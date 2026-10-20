"""期货标准研究面板组装与输出 (etl/build.py)。

按 P1 规划串联所有清洗步骤:
1. 加载交易日历 (calendar) 与合约静态元数据 (instruments_info, industry_mapping);
2. 构造主力合约连续映射 (dominant_mapping, rule=3, rank=1);
3. 关联日线行情 (bar_1d) 得到基础日频量价;
4. 计算比例复权因子与复权价格 (adjust), 并与米筐官方 ex_cum_factor 对账;
5. 流式解析分钟线 (bar_1m), 提取开盘 30m 执行价 (exec_price) 与 11 项微观结构特征 (minute_features);
6. 关联旧合约换月日执行价 (exec_vwap_old), 彻底杜绝换月跳空;
7. 关联展期收益率 (aux_roll), 仓单库存标准化 (aux_warehouse), 会员持仓排名 (aux_member);
8. 计算因果可交易品种池掩码 (universe);
9. 组装为标准宽长表 panel_daily.parquet, 强校验字段与 fields.py 规格完全一致;
10. 构建未来多期链式收益预测标签 labels.parquet (独立存放, 严防 DSL 泄漏);
11. 输出 fields.json 字段规格书与 meta.json 实验元数据。
"""

from __future__ import annotations

import json
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import polars as pl

from factorlib.core.config import DataConfig, UniverseConfig, load_universe_config
from factorlib.markets.futures.universe import build_universe

from .adjust import apply_adjustment, verify_against_official
from .aux_member import build_member_features
from .aux_roll import build_roll_yield
from .aux_warehouse import build_warehouse_features
from .calendar import load_calendar
from .daily import build_daily, load_bars_1d
from .dominant import describe_gaps, load_dominant
from .exec_price import minute_pairs
from .fields import DSL_FIELDS, LABEL_DESCRIPTIONS, PANEL_COLUMNS, PANEL_FIELDS, polars_dtype_name
from .instruments import first_listed_dates, load_instruments, load_sectors
from .io import file_sha256, source_fingerprint
from .labels import build_labels
from .minute_features import compute_minute_daily

logger = logging.getLogger(__name__)


def build_panel(
    cfg: DataConfig,
    ucfg: UniverseConfig | None = None,
    universe_path: Path | None = None,
    verify_adjust: bool = True,
    show_progress: bool = True,
) -> tuple[pl.DataFrame, pl.DataFrame, dict[str, Any]]:
    """构建期货标准研究面板并持久化。

    Parameters
    ----------
    cfg:
        数据清洗配置 (DataConfig)
    ucfg:
        品种池筛选配置 (UniverseConfig), 若未传则从 ``universe_path`` 或默认路径加载
    universe_path:
        universe.yaml 路径
    verify_adjust:
        是否与米筐官方复权因子 ex_cum_factor 对账
    show_progress:
        是否输出阶段耗时日志

    Returns
    -------
    (panel_daily, labels, meta)
        组装完成的面板 DataFrame、多期标签 DataFrame 以及元数据字典。
    """
    start_time = time.time()
    t0 = start_time

    if ucfg is None:
        if universe_path is not None and universe_path.exists():
            ucfg = load_universe_config(universe_path)
        else:
            default_u = Path("configs/futures/universe.yaml")
            if default_u.exists():
                ucfg = load_universe_config(default_u)
            else:
                ucfg = UniverseConfig()

    output_dir = cfg.panel_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    def _step(msg: str):
        nonlocal t0
        now = time.time()
        if show_progress:
            logger.info("[%6.2fs] %s", now - t0, msg)
        t0 = now

    logger.info("=== 开始构建期货标准研究面板 [%s -> %s] ===", cfg.start_date, cfg.end_date)

    # 1. 交易日历与静态元数据
    _step("加载交易日历与合约静态规格...")
    cal = load_calendar(cfg.raw_dir)
    instruments = load_instruments(cfg.raw_dir)
    first_listed = first_listed_dates(instruments)
    sectors = load_sectors(cfg.raw_dir)

    # 2. 主力合约连续映射
    _step("加载主力合约映射 (rule=3, rank=1)...")
    dominant = load_dominant(cfg.raw_dir, cfg, cal)
    gaps = describe_gaps(dominant)
    if not gaps.is_empty():
        logger.warning("主力映射表中存在 %d 个跨度空洞品种段", len(gaps))

    all_contracts = (
        dominant.select(["contract", "prev_contract"])
        .unpivot()
        .select(pl.col("value").alias("contract"))
        .unique()["contract"]
        .to_list()
    )

    # 3. 日线全量量价
    _step(f"读取日线行情 (涉及 {len(all_contracts)} 个合约)...")
    bars = load_bars_1d(cfg.raw_dir, all_contracts)
    daily = build_daily(dominant, bars, instruments, sectors)

    # 4. 比例复权与官方因子对账
    _step("执行因果比例复权...")
    daily, rolls = apply_adjustment(daily, bars)
    adjust_report: dict[str, Any] = {}
    if verify_adjust and cfg.adjust.verify_with_ex_factor:
        _step("与米筐官方 ex_cum_factor 对账...")
        verif = verify_against_official(daily, rolls, cfg.raw_dir, cfg)
        adjust_report = {
            "n_compared": verif.n_compared,
            "max_rel_err": verif.max_rel_err,
            "pass_tolerance": verif.ok,
            "tolerance": verif.tolerance,
            "failures_count": len(verif.failures),
            "fallback_rolls_count": len(verif.fallback_rolls),
        }
        if not verif.ok:
            logger.warning(
                "累计复权因子对账告警: %d 个品种相对误差超标 (max_rel_err=%.6e, 阈值=%.6e)",
                len(verif.failures),
                verif.max_rel_err,
                verif.tolerance,
            )
        else:
            logger.info(
                "累计复权因子对账通过! 最大相对误差: %.4e (阈值: %.4e)",
                verif.max_rel_err,
                verif.tolerance,
            )

    # 5. 分钟线流式处理 (执行价 + 微观结构特征)
    _step("按月流式处理分钟线 (开盘 30m VWAP + 微观结构特征)...")
    pairs = minute_pairs(daily, bars)
    min_daily = compute_minute_daily(cfg.raw_dir, pairs, cfg)

    # 6. 关联主/旧合约开盘 VWAP 执行价
    _step("组装主合约执行价与换月日旧合约平仓执行价...")
    exec_main = min_daily.rename({"order_book_id": "contract"})
    daily = daily.join(exec_main, on=["contract", "trading_date"], how="left")

    # 提取换月日旧合约在 T 日的执行价 exec_vwap_old
    rolls_exec = min_daily.select(
        pl.col("order_book_id").alias("prev_contract"),
        "trading_date",
        pl.col("exec_vwap").alias("_old_exec_vwap"),
    )
    daily = daily.join(rolls_exec, on=["prev_contract", "trading_date"], how="left")
    daily = daily.with_columns(
        exec_vwap_old=pl.when(pl.col("is_roll"))
        .then(pl.col("_old_exec_vwap").fill_null(pl.col("settle_raw")))
        .otherwise(pl.col("exec_vwap"))
    ).drop("_old_exec_vwap")

    # 7. 衍生特征清洗 (展期收益率、仓单库存、会员排名)
    max_date = daily["trading_date"].max()

    if cfg.features.roll_yield:
        _step("清洗展期收益率 (roll_yield)...")
        ry = build_roll_yield(cfg.raw_dir, cfg, daily, max_date)
        daily = daily.join(ry, on=["symbol", "trading_date"], how="left")
    else:
        daily = daily.with_columns(
            ry_main_sub=pl.lit(None, dtype=pl.Float64),
            ry_ann=pl.lit(None, dtype=pl.Float64),
            ry_near_main=pl.lit(None, dtype=pl.Float64),
        )

    if cfg.features.warehouse_stocks:
        _step("清洗仓单库存变动与 Z-Score (warehouse_stocks)...")
        wh = build_warehouse_features(cfg.raw_dir, cfg, daily, cal, max_date)
        daily = daily.join(wh, on=["symbol", "trading_date"], how="left")
    else:
        daily = daily.with_columns(
            wh_chg5=pl.lit(None, dtype=pl.Float64),
            wh_z60=pl.lit(None, dtype=pl.Float64),
        )

    if cfg.features.member_rank.enabled:
        _step("清洗会员持仓龙虎榜特征 (member_rank)...")
        mr = build_member_features(cfg.raw_dir, cfg, daily, max_date)
        daily = daily.join(mr, on=["symbol", "trading_date"], how="left")
    else:
        daily = daily.with_columns(
            mr_net_ratio=pl.lit(None, dtype=pl.Float64),
            mr_cr20=pl.lit(None, dtype=pl.Float64),
            mr_net_ratio_chg5=pl.lit(None, dtype=pl.Float64),
        )

    # 8. 品种池筛选
    _step("计算因果流动性品种池掩码 (in_universe)...")
    univ = build_universe(daily, first_listed, cal, ucfg)
    daily = daily.join(univ, on=["symbol", "trading_date"], how="left")

    # 复权全天 VWAP
    if "vwap_day" in daily.columns:
        daily = daily.with_columns(
            vwap=(pl.col("vwap_day") * pl.col("adj_factor")).cast(pl.Float64)
        )
    else:
        daily = daily.with_columns(vwap=pl.col("close"))

    # 9. 规范化与字段对齐
    _step("校验并对齐面板字段...")
    for spec in PANEL_FIELDS:
        name = spec.name
        if name not in daily.columns:
            daily = daily.with_columns(pl.lit(None).alias(name))
        expected_dtype = polars_dtype_name(spec)
        actual_dtype = str(daily.schema[name])
        if actual_dtype != expected_dtype:
            target_pl_dtype = getattr(pl, expected_dtype)
            daily = daily.with_columns(pl.col(name).cast(target_pl_dtype))

    panel_daily = daily.select(PANEL_COLUMNS).sort(["trading_date", "symbol"])

    # 10. 链式未来多期预测标签构建
    _step("构建链式多期前向收益预测标签 (labels.parquet)...")
    labels = build_labels(panel_daily, cal, cfg.label_horizons)

    # 11. 持久化与元数据记录
    _step("持久化面板长表与元数据文件...")
    panel_path = output_dir / "panel_daily.parquet"
    labels_path = output_dir / "labels.parquet"
    rolls_path = output_dir / "rolls.parquet"
    fields_path = output_dir / "fields.json"
    meta_path = output_dir / "meta.json"

    panel_daily.write_parquet(panel_path, compression="zstd")
    labels.write_parquet(labels_path, compression="zstd")
    rolls.write_parquet(rolls_path, compression="zstd")

    # 生成 fields.json
    fields_doc = {
        "panel_version": cfg.panel_version,
        "generated_at": datetime.now().isoformat(),
        "total_fields": len(PANEL_FIELDS),
        "dsl_fields_count": len(DSL_FIELDS),
        "fields": [spec.to_dict() for spec in PANEL_FIELDS],
        "labels": LABEL_DESCRIPTIONS,
    }
    with fields_path.open("w", encoding="utf-8") as fh:
        json.dump(fields_doc, fh, indent=2, ensure_ascii=False)

    total_time = time.time() - start_time
    source_fp = source_fingerprint(
        cfg.raw_dir,
        [
            "calendar",
            "instruments_info",
            "industry_mapping",
            "dominant_mapping",
            "bar_1d",
            "dominant_ex_factor",
            "roll_yield",
            "warehouse_stocks",
            "member_rank",
        ],
    )

    meta_doc = {
        "panel_version": cfg.panel_version,
        "generated_at": datetime.now().isoformat(),
        "build_duration_seconds": round(total_time, 2),
        "config_hash": cfg.hash(),
        "source_fingerprint": source_fp,
        "panel_sha256": file_sha256(panel_path),
        "labels_sha256": file_sha256(labels_path),
        "date_range": [
            str(panel_daily["trading_date"].min()),
            str(panel_daily["trading_date"].max()),
        ],
        "total_trading_days": panel_daily["trading_date"].n_unique(),
        "total_symbols": panel_daily["symbol"].n_unique(),
        "total_records": len(panel_daily),
        "in_universe_records": panel_daily["in_universe"].sum(),
        "config": cfg.to_dict(),
        "universe_config": ucfg.to_dict(),
        "adjust_verification": adjust_report,
    }
    with meta_path.open("w", encoding="utf-8") as fh:
        json.dump(meta_doc, fh, indent=2, ensure_ascii=False)

    logger.info("=== 期货标准研究面板构建圆满完成! 耗时: %.2fs ===", total_time)
    logger.info(
        "面板数据路径: %s (%d 行, %d 品种)",
        panel_path,
        len(panel_daily),
        panel_daily["symbol"].n_unique(),
    )
    logger.info("多期标签路径: %s", labels_path)
    logger.info("字段规格文件: %s", fields_path)

    return panel_daily, labels, meta_doc
