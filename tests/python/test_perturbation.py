"""未来数据扰动测试 (tests/python/test_perturbation.py)。

按技术白皮书 14.1 节规范:
- 针对期货标准研究面板 ETL 管道执行未来数据扰动验收测试 (Perturbation Test);
- 随机或显著篡改 T_0 日之后的所有原始行情与特征数据;
- 重新运行全量 ETL 计算, 严格断言在 T <= T_0 时刻的所有面板特征逐比特 (Bit-for-Bit) 完全一致,
  绝对杜绝未来函数 (Look-Ahead Bias) 与因果漂移;
- 同时断言 T > T_0 的特征发生显著变化 (证明扰动非空且实际生效),
  以及 T_0 处的未来预测标签 (y_1) 因前向持有跨入 T > T_0 而发生变化 (因果隔离验证)。
"""

from __future__ import annotations

from dataclasses import replace
from datetime import date, datetime
from pathlib import Path

import numpy as np
import polars as pl
import pytest
from factorlib.core.config import (
    DataConfig,
    DominantConfig,
    UniverseConfig,
    load_data_config,
    load_universe_config,
)
from factorlib.core.panel import Panel
from factorlib.markets.futures.etl.build import build_panel


def _generate_synthetic_raw(
    raw_dir: Path,
    dates: list[date],
    perturb_from: date | None = None,
) -> None:
    """生成包含主力合约换月、日线量价、分钟高频、期限结构、仓单与会员龙虎榜的自包含微型原始数据集。

    若指定 ``perturb_from``, 则在 ``trading_date > perturb_from`` 的记录上注入扰动。
    """
    raw_dir.mkdir(parents=True, exist_ok=True)

    # 1. calendar
    cal = pl.DataFrame({"trading_date": dates})
    (raw_dir / "calendar").mkdir(parents=True, exist_ok=True)
    cal.write_parquet(raw_dir / "calendar" / "calendar.parquet")

    # 2. instruments_info (RB 涉及 RB2405 与 RB2410 两个合约; CU 为单合约 CU2405)
    inst = pl.DataFrame(
        {
            "order_book_id": ["RB2405", "RB2410", "CU2405"],
            "underlying_symbol": ["RB", "RB", "CU"],
            "exchange": ["SHFE", "SHFE", "SHFE"],
            "contract_multiplier": [10.0, 10.0, 5.0],
            "margin_rate": [0.1, 0.1, 0.1],
            "listed_date": ["2023-01-01", "2023-06-01", "2023-01-01"],
            "maturity_date": ["2024-05-15", "2024-10-15", "2024-05-15"],
        }
    )
    (raw_dir / "instruments_info").mkdir(parents=True, exist_ok=True)
    inst.write_parquet(raw_dir / "instruments_info" / "instruments_info.parquet")

    # 3. industry_mapping
    ind = pl.DataFrame(
        {
            "underlying_symbol": ["RB", "CU"],
            "level1_industry": ["Black", "Nonferrous"],
        }
    )
    (raw_dir / "industry_mapping").mkdir(parents=True, exist_ok=True)
    ind.write_parquet(raw_dir / "industry_mapping" / "industry_mapping.parquet")

    # 4. dominant_mapping (RB 在 dates[5] 换月由 RB2405 切换为 RB2410)
    rb_contracts = ["RB2405"] * 5 + ["RB2410"] * 5
    cu_contracts = ["CU2405"] * 10
    dm = pl.DataFrame(
        {
            "date": dates + dates,
            "underlying_symbol": ["RB"] * 10 + ["CU"] * 10,
            "order_book_id": rb_contracts + cu_contracts,
            "rule": [3] * 20,
            "rank": [1] * 20,
        }
    )
    (raw_dir / "dominant_mapping").mkdir(parents=True, exist_ok=True)
    dm.write_parquet(raw_dir / "dominant_mapping" / "dominant_mapping.parquet")

    # 5. bar_1d
    rows_1d = []
    for d in dates:
        is_pert = (perturb_from is not None) and (d > perturb_from)
        price_scale = (1.5 + (d.day - 5) * 0.2) if is_pert else 1.0
        price_offset = 500.0 if is_pert else 0.0
        vol_scale = 3.0 if is_pert else 1.0

        # CU2405
        cu_close = (68200.0 + (d.day - 1) * 20.0) * price_scale + price_offset
        rows_1d.append(
            {
                "order_book_id": "CU2405",
                "trading_date": d,
                "open": cu_close - 50.0,
                "high": cu_close + 100.0,
                "low": cu_close - 100.0,
                "close": cu_close,
                "settlement": cu_close,
                "volume": 50000.0 * vol_scale,
                "total_turnover": 1.7e9 * vol_scale,
                "open_interest": 100000.0,
                "prev_settlement": cu_close - 20.0,
                "limit_up": cu_close * 1.08,
                "limit_down": cu_close * 0.92,
                "is_unilateral": 0,
            }
        )

        # RB2405 (前 6 天交易, 并在 dates[5] 换月日与 RB2410 重叠交易)
        if d <= dates[5]:
            rb5_close = (3820.0 + (d.day - 1) * 10.0) * price_scale + price_offset
            rows_1d.append(
                {
                    "order_book_id": "RB2405",
                    "trading_date": d,
                    "open": rb5_close - 10.0,
                    "high": rb5_close + 20.0,
                    "low": rb5_close - 20.0,
                    "close": rb5_close,
                    "settlement": rb5_close,
                    "volume": 200000.0 * vol_scale,
                    "total_turnover": 7.6e9 * vol_scale,
                    "open_interest": 500000.0,
                    "prev_settlement": rb5_close - 10.0,
                    "limit_up": rb5_close * 1.08,
                    "limit_down": rb5_close * 0.92,
                    "is_unilateral": 0,
                }
            )

        # RB2410 (从 dates[4] 之前就上市交易, 在 dates[5] 成为主力)
        if d >= dates[4]:
            rb10_close = (3720.0 + (d.day - 1) * 10.0) * price_scale + price_offset
            rows_1d.append(
                {
                    "order_book_id": "RB2410",
                    "trading_date": d,
                    "open": rb10_close - 10.0,
                    "high": rb10_close + 20.0,
                    "low": rb10_close - 20.0,
                    "close": rb10_close,
                    "settlement": rb10_close,
                    "volume": 150000.0 * vol_scale,
                    "total_turnover": 5.5e9 * vol_scale,
                    "open_interest": 400000.0,
                    "prev_settlement": rb10_close - 10.0,
                    "limit_up": rb10_close * 1.08,
                    "limit_down": rb10_close * 0.92,
                    "is_unilateral": 0,
                }
            )

    b1d = pl.DataFrame(rows_1d)
    (raw_dir / "bar_1d" / "full").mkdir(parents=True, exist_ok=True)
    b1d.write_parquet(raw_dir / "bar_1d" / "full" / "bar_1d_2024_01.parquet")

    # 6. bar_1m (每个合约每个交易日 2 根早盘分钟线)
    rows_1m = []
    for r in rows_1d:
        d = r["trading_date"]
        dt0 = datetime(d.year, d.month, d.day, 9, 1)
        dt1 = datetime(d.year, d.month, d.day, 9, 2)
        rows_1m.append(
            {
                "order_book_id": r["order_book_id"],
                "trading_date": d,
                "datetime": dt0,
                "open": r["open"],
                "high": r["high"],
                "low": r["low"],
                "close": r["close"],
                "volume": 500.0,
            }
        )
        rows_1m.append(
            {
                "order_book_id": r["order_book_id"],
                "trading_date": d,
                "datetime": dt1,
                "open": r["open"],
                "high": r["high"],
                "low": r["low"],
                "close": r["close"],
                "volume": 500.0,
            }
        )
    b1m = pl.DataFrame(rows_1m)
    (raw_dir / "bar_1m" / "full").mkdir(parents=True, exist_ok=True)
    b1m.write_parquet(raw_dir / "bar_1m" / "full" / "bar_1m_2024_01.parquet")

    # 7. roll_yield
    ry_rows = []
    for d in dates:
        is_pert = (perturb_from is not None) and (d > perturb_from)
        scale = 2.5 if is_pert else 1.0
        ry_rows.append(
            {
                "trading_date": d,
                "underlying_symbol": "RB",
                "roll_type": "main_sub",
                "rule": 3,
                "from_contract": "RB2405" if d < dates[5] else "RB2410",
                "to_contract": "RB2410" if d < dates[5] else "RB2501",
                "roll_yield": 0.02 * scale,
                "annualized_yield_trading": 0.08 * scale,
            }
        )
        ry_rows.append(
            {
                "trading_date": d,
                "underlying_symbol": "RB",
                "roll_type": "near_main",
                "rule": 3,
                "from_contract": "RB2403",
                "to_contract": "RB2405" if d < dates[5] else "RB2410",
                "roll_yield": 0.01 * scale,
                "annualized_yield_trading": 0.04 * scale,
            }
        )
    ry = pl.DataFrame(ry_rows)
    (raw_dir / "roll_yield" / "full").mkdir(parents=True, exist_ok=True)
    ry.write_parquet(raw_dir / "roll_yield" / "full" / "roll_yield_2024_01.parquet")

    # 8. warehouse_stocks
    wh_rows = []
    for d in dates:
        is_pert = (perturb_from is not None) and (d > perturb_from)
        offset = 50000.0 if is_pert else 0.0
        wh_rows.append(
            {"trading_date": d, "underlying_symbol": "RB", "on_warrant": 12000.0 + offset}
        )
        wh_rows.append(
            {"trading_date": d, "underlying_symbol": "CU", "on_warrant": 5000.0 + offset}
        )
    wh = pl.DataFrame(wh_rows)
    (raw_dir / "warehouse_stocks" / "full").mkdir(parents=True, exist_ok=True)
    wh.write_parquet(raw_dir / "warehouse_stocks" / "full" / "warehouse_stocks_2024_01.parquet")

    # 9. member_rank
    mr_rows = []
    for d in dates:
        is_pert = (perturb_from is not None) and (d > perturb_from)
        scale = 3.0 if is_pert else 1.0
        for rk in range(1, 6):
            mr_rows.append(
                {
                    "trading_date": d,
                    "order_book_id": "RB2405" if d < dates[5] else "RB2410",
                    "rank_by": "long",
                    "member_rank": rk,
                    "volume": 1000.0 * scale,
                }
            )
            mr_rows.append(
                {
                    "trading_date": d,
                    "order_book_id": "RB2405" if d < dates[5] else "RB2410",
                    "rank_by": "short",
                    "member_rank": rk,
                    "volume": 800.0 * scale,
                }
            )
    mr = pl.DataFrame(mr_rows)
    (raw_dir / "member_rank" / "full").mkdir(parents=True, exist_ok=True)
    mr.write_parquet(raw_dir / "member_rank" / "full" / "member_rank_2024_01.parquet")

    # 10. dominant_ex_factor 占位
    (raw_dir / "dominant_ex_factor" / "full").mkdir(parents=True, exist_ok=True)
    pl.DataFrame(
        {
            "underlying_symbol": ["RB"],
            "trading_date": [dates[0]],
            "ex_cum_factor": [1.0],
        }
    ).write_parquet(raw_dir / "dominant_ex_factor" / "full" / "dummy.parquet")


def test_perturbation_synthetic(tmp_path: Path):
    """自包含微型合成数据集的未来数据扰动测试 (覆盖换月事件前后与全量清洗算子)"""
    dates = [date(2024, 1, i) for i in range(1, 11)]
    # 设定基准截断日 T0 为第 5 天 (2024-01-05), 换月事件在第 6 天发生
    cutoff_date = dates[4]
    cutoff_idx = 5  # 前 5 个交易日为 T <= T0

    raw_base = tmp_path / "synth_raw_base"
    raw_pert = tmp_path / "synth_raw_pert"

    _generate_synthetic_raw(raw_base, dates, perturb_from=None)
    _generate_synthetic_raw(raw_pert, dates, perturb_from=cutoff_date)

    cfg_base = DataConfig(
        raw_dir=raw_base,
        derived_dir=tmp_path / "panel_base",
        panel_version="synth_base",
        start_date=dates[0],
        end_date=dates[-1],
        dominant=DominantConfig(rule=3, rank=1),
    )
    cfg_pert = DataConfig(
        raw_dir=raw_pert,
        derived_dir=tmp_path / "panel_pert",
        panel_version="synth_pert",
        start_date=dates[0],
        end_date=dates[-1],
        dominant=DominantConfig(rule=3, rank=1),
    )
    ucfg = UniverseConfig(min_adv20_turnover=1000.0, min_listed_days=0, min_days_to_expiry=0)

    build_panel(cfg_base, ucfg, verify_adjust=False, show_progress=False)
    build_panel(cfg_pert, ucfg, verify_adjust=False, show_progress=False)

    p_base = Panel.load(cfg_base.panel_dir)
    p_pert = Panel.load(cfg_pert.panel_dir)

    # 1. 核心断言: T <= T0 的所有特征矩阵逐比特完全一致
    for fname, base_mat in p_base.fields.items():
        pert_mat = p_pert.fields[fname]
        base_slice = base_mat[:cutoff_idx, :]
        pert_slice = pert_mat[:cutoff_idx, :]
        np.testing.assert_array_equal(
            base_slice,
            pert_slice,
            err_msg=f"合成数据未来扰动测试失败: 字段 {fname!r} 在 T <= {cutoff_date} 存在差异!",
        )

    # 2. 掩码在 T <= T0 完全一致
    np.testing.assert_array_equal(p_base.mask[:cutoff_idx, :], p_pert.mask[:cutoff_idx, :])

    # 3. T > T0 确实发生变化
    close_diff = np.nanmax(
        np.abs(p_base.fields["close"][cutoff_idx:, :] - p_pert.fields["close"][cutoff_idx:, :])
    )
    assert close_diff > 100.0, "T > T0 预期发生显著变化"

    # 4. 因果边界验证: T0 处的 y_1 预测标签因未来行情扰动而改变
    y1_t0_base = p_base.labels["y_1"][cutoff_idx - 1, :]
    y1_t0_pert = p_pert.labels["y_1"][cutoff_idx - 1, :]
    y1_t0_diff = np.nanmax(np.abs(y1_t0_base - y1_t0_pert))
    assert y1_t0_diff > 0.001, "T0 处的 y_1 标签预期受未来扰动影响"


def test_perturbation_real_data(tmp_path: Path):
    """基于真实 exports 数据集的未来数据扰动端到端测试。"""
    root_dir = Path(__file__).resolve().parent.parent.parent
    exports_dir = root_dir / "exports"
    config_path = root_dir / "configs/futures/data.yaml"
    universe_path = root_dir / "configs/futures/universe.yaml"

    if not exports_dir.exists():
        pytest.skip("未检测到 exports 原始数据目录，跳过实盘未来扰动测试")

    start_d = date(2024, 2, 1)
    end_d = date(2024, 3, 31)
    cutoff_date_str = "2024-02-29"

    cfg_base = load_data_config(config_path, root=root_dir)
    cfg_base = replace(
        cfg_base,
        start_date=start_d,
        end_date=end_d,
        panel_version="test_perturb_base",
        derived_dir=tmp_path / "derived",
    )
    ucfg = load_universe_config(universe_path)

    # 构建未扰动的基准面板
    build_panel(cfg_base, ucfg, verify_adjust=False, show_progress=False)
    p_base = Panel.load(cfg_base.panel_dir)

    dates = [str(d) for d in p_base.dates]
    assert cutoff_date_str in dates, f"截断日 {cutoff_date_str} 必须在基准时间轴内"
    cutoff_idx = dates.index(cutoff_date_str) + 1  # 包含 cutoff_date 的切片长度 (15)

    # 构造扰动原始数据集
    tmp_raw = tmp_path / "raw_perturbed"
    tmp_raw.mkdir(parents=True)

    for item in exports_dir.iterdir():
        if not item.is_dir() or item.name.startswith("_"):
            continue
        if item.name not in ("bar_1d", "bar_1m"):
            (tmp_raw / item.name).symlink_to(item)

    # 针对 bar_1d: 软链接 2024-02, 篡改 2024-03
    (tmp_raw / "bar_1d" / "full").mkdir(parents=True)
    for f in (exports_dir / "bar_1d" / "full").glob("*.parquet"):
        if f.name == "bar_1d_2024_03.parquet":
            df_03 = pl.read_parquet(f)
            df_perturbed = df_03.with_columns(
                open=pl.col("open") * 2.0 + 500.0,
                high=pl.col("high") * 2.0 + 550.0,
                low=pl.col("low") * 2.0 + 450.0,
                close=pl.col("close") * 2.0 + 500.0,
                settlement=pl.col("settlement") * 2.0 + 500.0,
                volume=pl.col("volume") * 3.0 + 1000.0,
                total_turnover=pl.col("total_turnover") * 5.0 + 1000000.0,
            )
            df_perturbed.write_parquet(tmp_raw / "bar_1d" / "full" / f.name)
        else:
            (tmp_raw / "bar_1d" / "full" / f.name).symlink_to(f)

    # 针对 bar_1m: 软链接 2024-02, 篡改 2024-03
    (tmp_raw / "bar_1m" / "full").mkdir(parents=True)
    for f in (exports_dir / "bar_1m" / "full").glob("*.parquet"):
        if f.name == "bar_1m_2024_03.parquet":
            df_1m = pl.read_parquet(f)
            df_1m_pert = df_1m.with_columns(
                open=pl.col("open") * 2.0 + 500.0,
                high=pl.col("high") * 2.0 + 550.0,
                low=pl.col("low") * 2.0 + 450.0,
                close=pl.col("close") * 2.0 + 500.0,
                volume=pl.col("volume") * 3.0 + 1000.0,
            )
            df_1m_pert.write_parquet(tmp_raw / "bar_1m" / "full" / f.name)
        else:
            (tmp_raw / "bar_1m" / "full" / f.name).symlink_to(f)

    cfg_pert = replace(
        cfg_base,
        raw_dir=tmp_raw,
        panel_version="test_perturb_mod",
    )
    build_panel(cfg_pert, ucfg, verify_adjust=False, show_progress=False)
    p_pert = Panel.load(cfg_pert.panel_dir)

    # 断言 1: T <= T_0 特征逐比特一致
    for fname, base_mat in p_base.fields.items():
        pert_mat = p_pert.fields[fname]
        base_slice = base_mat[:cutoff_idx, :]
        pert_slice = pert_mat[:cutoff_idx, :]
        np.testing.assert_array_equal(
            base_slice,
            pert_slice,
            err_msg=f"实盘未来扰动测试失败: 特征 {fname!r} 在 T <= {cutoff_date_str} 存在差异!",
        )

    # 断言 2: 品种池掩码一致
    np.testing.assert_array_equal(
        p_base.mask[:cutoff_idx, :],
        p_pert.mask[:cutoff_idx, :],
        err_msg="品种池掩码在 T <= T_0 存在差异!",
    )

    # 断言 3: T > T_0 发生变化
    close_diff = np.nanmax(
        np.abs(p_base.fields["close"][cutoff_idx:, :] - p_pert.fields["close"][cutoff_idx:, :])
    )
    assert close_diff > 100.0, (
        f"预期 T > T_0 的 close 字段发生明显变化，实际最大差异仅 {close_diff}"
    )

    # 断言 4: T0 处未来标签因跨入扰动期而改变
    y1_t0_base = p_base.labels["y_1"][cutoff_idx - 1, :]
    y1_t0_pert = p_pert.labels["y_1"][cutoff_idx - 1, :]
    y1_t0_diff = np.nanmax(np.abs(y1_t0_base - y1_t0_pert))
    assert y1_t0_diff > 0.001, f"T0 处的 y_1 预测标签预期受未来扰动影响，实际差异仅 {y1_t0_diff}"
