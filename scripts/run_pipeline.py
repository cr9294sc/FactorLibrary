#!/usr/bin/env python3
"""FactorLibrary 端到端全流程一键执行流水线 (scripts/run_pipeline.py)。

按白皮书第 3 节与 P7 验收规范设计，串通量化因子研发全链路：
1. 数据面板加载与时空坐标校验 (Panel Load & Validation)
2. 经典学术种子因子初始化与注册 (Seed Factors Init & Registration)
3. (可选) 强类型遗传规划自动化因子挖掘 (STGP Evolutionary Discovery & Admission)
4. Purged Walk-Forward 机器学习多因子合成 (ML Synthesis & Prediction)
5. 风险平价与波动率倒数加权组合构建 (Portfolio Optimization & Buffer Rebalancing)
6. Rust 极速实盘语义向量化回测 (Discrete Lots, Margin MTM, Roll-over Costs)
7. 交互式 HTML 诊断报告与全生命周期资产落盘 (Interactive Report & Asset Archiving)

用法示例:
    # 1. 默认全流程运行 (以种子因子与 ML 合成执行全闭环回测)
    python scripts/run_pipeline.py

    # 2. 单因子直接全流程快速串通
    python scripts/run_pipeline.py --factor "ts_zscore(close, 20)"

    # 3. 启用 GP 批量挖掘与全流程合成回测
    python scripts/run_pipeline.py --run-gp --gp-population 100 --gp-generations 5
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import logging
import sys
from pathlib import Path
from typing import Any

import numpy as np

# 将 python/ 目录加入 sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "python"))

from factorlib.core.backtest import BacktestResult, run_backtest
from factorlib.core.config import (
    BacktestConfig,
    load_backtest_config,
    load_gp_config,
    load_ml_config,
)
from factorlib.core.evaluate import evaluate_factor, get_seed_factors
from factorlib.core.panel import Panel
from factorlib.core.portfolio import build_portfolio_weights, compute_daily_returns_from_prices
from factorlib.core.registry import FactorRegistry
from factorlib.core.report import generate_html_report, summary_dict
from factorlib.core.runs import RunTracker
from factorlib.gp.engine import GPEngine
from factorlib.markets.futures.adapter import FuturesMarketAdapter
from factorlib.ml.pipeline import WalkForwardPipeline


def parse_args(args: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="FactorLibrary 端到端全流程一键执行流水线 (P7 整体验收)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--panel-dir",
        type=Path,
        default=Path("data/derived/futures/v1"),
        help="标准研究面板存储目录",
    )
    parser.add_argument(
        "--db",
        type=Path,
        default=Path("data/registry/factors.sqlite"),
        help="SQLite 因子注册库路径",
    )
    parser.add_argument(
        "--backtest-config",
        type=Path,
        default=Path("configs/futures/backtest.yaml"),
        help="回测与风控配置文件路径",
    )
    parser.add_argument(
        "--fees-config",
        type=Path,
        default=Path("configs/futures/fees.yaml"),
        help="期货品种合约规格与费率成本配置文件路径",
    )
    parser.add_argument(
        "--gp-config",
        type=Path,
        default=Path("configs/futures/gp_default.yaml"),
        help="GP 因子挖掘配置文件路径",
    )
    parser.add_argument(
        "--ml-config",
        type=Path,
        default=Path("configs/futures/ml_default.yaml"),
        help="ML 机器学习合成配置文件路径",
    )
    parser.add_argument(
        "--factor",
        type=str,
        default=None,
        help="直接指定单因子表达式进行全流程回测 (例如 'ts_zscore(close, 20)')",
    )
    parser.add_argument(
        "--run-gp",
        action="store_true",
        help="是否在流水线中执行强类型遗传规划 (STGP) 批量因子挖掘",
    )
    parser.add_argument(
        "--gp-generations",
        type=int,
        default=5,
        help="GP 挖掘演化代数",
    )
    parser.add_argument(
        "--gp-population",
        type=int,
        default=100,
        help="GP 挖掘总种群规模",
    )
    parser.add_argument(
        "--ml-model",
        type=str,
        default="equal_weight",
        choices=["equal_weight", "lgbm", "ridge"],
        help="机器学习多因子合成模型类型",
    )
    parser.add_argument(
        "--skip-seed",
        action="store_true",
        help="跳过经典文献种子因子入库检查",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="实验产出目录 (默认自动落盘至 runs/pipeline_YYYYMMDD_HHMMSS)",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="日志记录级别",
    )
    return parser.parse_args(args)


def run_pipeline(
    panel_dir: Path | str = Path("data/derived/futures/v1"),
    db_path: Path | str = Path("data/registry/factors.sqlite"),
    backtest_config_path: Path | str = Path("configs/futures/backtest.yaml"),
    fees_config_path: Path | str = Path("configs/futures/fees.yaml"),
    gp_config_path: Path | str = Path("configs/futures/gp_default.yaml"),
    ml_config_path: Path | str = Path("configs/futures/ml_default.yaml"),
    factor_expr: str | None = None,
    run_gp: bool = False,
    gp_generations: int = 5,
    gp_population: int = 100,
    ml_model: str = "equal_weight",
    skip_seed: bool = False,
    output_dir: Path | str | None = None,
    panel: Panel | None = None,
) -> dict[str, Any]:
    """端到端执行全流程量化研发流水线。

    Returns:
        包含回测指标摘要、输出文件路径与运行状态的字典。
    """
    panel_dir = Path(panel_dir)
    db_path = Path(db_path)
    backtest_config_path = Path(backtest_config_path)
    fees_config_path = Path(fees_config_path)
    gp_config_path = Path(gp_config_path)
    ml_config_path = Path(ml_config_path)

    # 1. 确定输出目录
    now_str = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    if output_dir is None:
        out_dir = Path("runs") / f"pipeline_{now_str}"
    else:
        out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    db_path.parent.mkdir(parents=True, exist_ok=True)
    registry = FactorRegistry(db_path=db_path)

    tracker = RunTracker(
        kind="backtest",
        run_id=out_dir.name,
        runs_root=out_dir.parent,
        registry=registry,
    )
    pipeline_meta: dict[str, Any] = {
        "pipeline_start_time": now_str,
        "run_gp": run_gp,
        "ml_model": ml_model,
    }

    logging.info("=" * 80)
    logging.info("  FactorLibrary 端到端全流程一键流水线启动 (Run ID: %s)", out_dir.name)
    logging.info("=" * 80)

    # 2. Stage 1: 加载与校验研究面板
    logging.info("[Stage 1/6] 检验并加载标准化研究面板...")
    if panel is None:
        if not panel_dir.exists():
            raise FileNotFoundError(f"面板目录不存在: {panel_dir}，请先执行 ETL 构建。")
        panel = Panel.load(panel_dir)

    T, N = panel.shape
    logging.info("  ✔ 面板加载完成: %d 交易日 x %d 品种, 包含 %d 特征", T, N, len(panel.fields))
    pipeline_meta["panel_T"] = T
    pipeline_meta["panel_N"] = N

    # 3. Stage 2: SQLite 因子库连接与种子因子初始化
    logging.info("[Stage 2/6] 连接 SQLite 因子注册表并检查基础因子资产...")

    if not skip_seed:
        # 检查是否已存在种子因子
        accepted_factors = registry.list_factors(status="accepted")
        if len(accepted_factors) == 0:
            logging.info("  因子库为空，正在自动评估并注册经典学术文献种子因子...")
            seeds = get_seed_factors()
            for s_item in seeds:
                if isinstance(s_item, dict):
                    s_name = s_item["name"]
                    s_expr = s_item["expr"]
                else:
                    s_name = str(s_item)
                    s_expr = str(s_item)
                eval_res = evaluate_factor(panel, s_expr)
                rec = eval_res.to_factor_record(
                    source="seed",
                    source_run="pipeline_seed_init",
                    data_version=str(panel.meta.get("panel_version", "v1")),
                    status="accepted",
                )
                try:
                    registry.register_factor(rec, run_id="pipeline_seed_init", upsert=True)
                    logging.info("    + 注册种子因子: %s (%s)", s_name, s_expr)
                except ValueError:
                    pass

    # 4. Stage 3: (可选) STGP 批量因子挖掘与准入
    if run_gp:
        import dataclasses

        logging.info(
            "[Stage 3/6] 启动强类型遗传规划 (STGP) 批量因子挖掘 (种群: %d, 代数: %d)...",
            gp_population,
            gp_generations,
        )
        gp_cfg = load_gp_config(gp_config_path)
        gp_cfg_override = dataclasses.replace(
            gp_cfg,
            population=gp_population,
            generations=gp_generations,
            islands=min(2, gp_cfg.islands),
            seed=42,
        )
        gp_tracker = RunTracker(
            kind="gp",
            run_id=f"gp_{out_dir.name}",
            runs_root=out_dir.parent,
            config_yaml=gp_cfg_override.to_yaml(),
            data_version=str(panel.meta.get("panel_version", "v1")),
            registry=registry,
        )
        engine = GPEngine(
            cfg=gp_cfg_override,
            panel=panel,
            registry=registry,
            tracker=gp_tracker,
        )
        gp_results = engine.run(
            run_id=gp_tracker.run_id,
            register_to_db=True,
        )
        logging.info("  ✔ GP 演化完成，累计准入发现候选因子 %d 个", len(gp_results))
        pipeline_meta["gp_admitted_count"] = len(gp_results)
    else:
        logging.info("[Stage 3/6] 跳过 STGP 演化挖掘阶段 (未指定 --run-gp)")

    # 5. Stage 4: 因子信号评分矩阵构建 (Scores Matrix: T x N)
    logging.info("[Stage 4/6] 生成因子信号评分矩阵 (Scores Matrix)...")
    scores = np.full((T, N), np.nan, dtype=np.float64, order="F")

    if factor_expr is not None:
        logging.info("  直接使用单因子 DSL 表达式高速求值: %s", factor_expr)
        rust_eng = panel.to_rust_engine()
        scores = rust_eng.eval(factor_expr)
        pipeline_meta["alpha_mode"] = "single_factor"
        pipeline_meta["factor_expr"] = factor_expr
    else:
        active_factors = registry.list_factors(status="accepted")
        logging.info("  从 SQLite 因子库提取 %d 个已准入因子进行多因子合成...", len(active_factors))

        if len(active_factors) == 0:
            # 备用动量因子
            logging.warning("  因子库无可用因子，自动回退到动量基准 'pct_change(close, 20)'")
            rust_eng = panel.to_rust_engine()
            scores = rust_eng.eval("pct_change(close, 20)")
            pipeline_meta["alpha_mode"] = "fallback_momentum"
        else:
            ml_cfg = load_ml_config(ml_config_path)
            logging.info("  执行 Purged Walk-Forward 机器学习多因子合成 (模型: %s)...", ml_model)
            try:
                pipeline_ml = WalkForwardPipeline.from_registry(
                    panel=panel,
                    config=ml_cfg,
                    registry=registry,
                )
                wf_result = pipeline_ml.run(strict_min_years=False)

                if ml_model in wf_result.predictions:
                    scores = wf_result.predictions[ml_model]
                else:
                    m_name = next(iter(wf_result.predictions.keys()))
                    scores = wf_result.predictions[m_name]

                # 保存预测文件到 runs 目录
                pred_parquet = out_dir / "predictions.parquet"
                wf_result.to_parquet(pred_parquet)
                pipeline_meta["alpha_mode"] = f"ml_{ml_model}"
                pipeline_meta["pred_parquet"] = str(pred_parquet)
                logging.info("  ✔ 机器学习多因子合成评分已完成并落盘: %s", pred_parquet)
            except RuntimeError as exc:
                logging.warning(
                    "  Walk-Forward 切分因样本时间跨度不足未能按年折叠 (%s)，回退为库内因子截面等权合成",
                    exc,
                )
                rust_eng = panel.to_rust_engine()
                factor_mats = []
                for f in active_factors[:6]:
                    mat = rust_eng.eval(f.canon_expr)
                    factor_mats.append(mat * float(f.sign))
                import warnings

                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", category=RuntimeWarning)
                    stacked = np.stack(factor_mats, axis=0)
                    scores = np.nanmean(stacked, axis=0)
                pipeline_meta["alpha_mode"] = "factors_equal_weight_fallback"

    # 6. Stage 5: 组合构建与权重优化 (Portfolio Optimization)
    logging.info("[Stage 5/6] 组合优化: 波动率倒数平价、收缩协方差与缓冲带调仓...")
    bt_cfg: BacktestConfig = load_backtest_config(backtest_config_path)
    close_field = panel.get_field("close")
    returns = compute_daily_returns_from_prices(close_field)
    weights = build_portfolio_weights(scores, returns, panel.mask, bt_cfg.portfolio)

    valid_weight_days = np.sum(np.any(~np.isnan(weights) & (weights != 0.0), axis=1))
    logging.info(
        "  ✔ 组合目标连续权重矩阵 W 计算完成 (有效调仓天数: %d / %d)", valid_weight_days, T
    )

    # 7. Stage 6: 执行 Rust 实盘口径向量化回测
    logging.info("[Stage 6/6] 调用 Rust 极速计算内核执行期货实盘语义回测...")
    adapter = FuturesMarketAdapter(fees_config_path=fees_config_path)
    bt_result: BacktestResult = run_backtest(panel, weights, config=bt_cfg, adapter=adapter)

    # 8. 产出报告与诊断汇总
    html_file = out_dir / "report.html"
    generate_html_report(
        bt_result, html_file, title=f"FactorLibrary 全流程流水线报告 - {out_dir.name}"
    )

    summary = summary_dict(bt_result)
    summary_file = out_dir / "summary.json"
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    # 导出逐日流水 CSV
    equity_file = out_dir / "equity.csv"
    import polars as pl

    daily_df = pl.DataFrame(
        {
            "trading_date": [str(d) for d in bt_result.dates],
            "equity": bt_result.equity.tolist(),
            "cash": bt_result.cash.tolist(),
            "margin_used": bt_result.margin_used.tolist(),
            "net_pnl": bt_result.net_pnl.tolist(),
            "turnover": bt_result.turnover.tolist(),
        }
    )
    daily_df.write_csv(equity_file)

    # 保存元数据并结束运行
    pipeline_meta["summary"] = summary
    tracker.save_json("pipeline_summary.json", pipeline_meta)
    tracker.finish(status="completed", n_evaluated=len(bt_result.dates))

    logging.info("=" * 80)
    logging.info("  ✔ FactorLibrary 端到端全流程流水线执行成功！")
    logging.info("  - 交互式 HTML 诊断报告: %s", html_file)
    logging.info("  - 评估指标汇总 JSON : %s", summary_file)
    logging.info("  - 年化收益率 (Annual Return): %.2f%%", summary.get("annual_return", 0.0) * 100)
    logging.info("  - 夏普比率 (Sharpe Ratio)  : %.2f", summary.get("sharpe", 0.0))
    logging.info("  - 最大回撤 (Max Drawdown)  : %.2f%%", summary.get("max_drawdown", 0.0) * 100)
    logging.info(
        "  - 日均单边换手率 (Daily Turnover): %.2f%%", summary.get("daily_turnover_mean", 0.0) * 100
    )
    logging.info("=" * 80)

    return {
        "status": "success",
        "run_id": out_dir.name,
        "output_dir": str(out_dir),
        "html_report": str(html_file),
        "summary_file": str(summary_file),
        "summary": summary,
        "backtest_result": bt_result,
    }


def main(cli_args: list[str] | None = None) -> int:
    args = parse_args(cli_args)

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="[%(asctime)s] [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    try:
        run_pipeline(
            panel_dir=args.panel_dir,
            db_path=args.db,
            backtest_config_path=args.backtest_config,
            fees_config_path=args.fees_config,
            gp_config_path=args.gp_config,
            ml_config_path=args.ml_config,
            factor_expr=args.factor,
            run_gp=args.run_gp,
            gp_generations=args.gp_generations,
            gp_population=args.gp_population,
            ml_model=args.ml_model,
            skip_seed=args.skip_seed,
            output_dir=args.output_dir,
        )
        return 0
    except Exception as exc:
        logging.exception("流水线执行失败: %s", exc)
        return 1


if __name__ == "__main__":
    sys.exit(main())
