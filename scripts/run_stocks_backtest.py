#!/usr/bin/env python3
"""FactorLibrary A 股统一回测执行入口 (scripts/run_stocks_backtest.py)。

完全独立于期货入口 (`scripts/run_backtest.py`)，践行「特征/模型在 `rq` 计算、执行统一落盘至
`barra` (`vwap_open_30m` 时钟) 经 `SignalBridge` 自动桥接」的单轨执行架构。

用法示例:
    # 1. 运行内置预设作业 (如 factorsim_match，直接读取 shifted_signal + 0.003 flat 费率 + 0.9 仓位)
    python scripts/run_stocks_backtest.py --profile factorsim_match

    # 2. 单因子 DSL 自动在 rq 求值并经 SignalBridge 桥接至 barra 执行 Barra QP 回测
    python scripts/run_stocks_backtest.py --factor "-ts_sum(c2c_ret, 5)" --postprocess standard

    # 3. ML / DL 样本外预测长表回测
    python scripts/run_stocks_backtest.py --pred runs/ml_xxx/predictions.parquet --profile default

    # 4. 外部宽表信号文件回测 (默认 T_exec 对齐，可用 --align 覆盖)
    python scripts/run_stocks_backtest.py --signal data/raw/stocks/barra/combo/signal/signal_data/shifted_signal.parquet
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from datetime import date, datetime
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "python"))

from factorlib.core.config import (
    StockBacktestConfig,
    StockFeesConfig,
    apply_stock_backtest_profile,
    load_stock_backtest_config,
    load_stock_fees_config,
)
from factorlib.core.evaluate import evaluate_factor_v4_stats
from factorlib.core.panel import Panel
from factorlib.core.probe import (
    FactorDefinition,
    evaluate_factor_definition,
    parse_postprocess_pipeline,
    probe_factor,
)
from factorlib.core.report import generate_html_report, summary_dict
from factorlib.core.synthetic import create_synthetic_stocks_panel
from factorlib.markets.profile import PanelSpec
from factorlib.markets.stocks.backtest import run_stock_backtest_pipeline
from factorlib.markets.stocks.bridge import BridgeAuditReport, SignalBridge
from factorlib.markets.stocks.profile import StocksProfile


def parse_args(args: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="FactorLibrary A 股统一信号桥接、Barra 组合优化与实盘状态机回测入口",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--profile",
        type=str,
        default="default",
        help="backtest.yaml 中预设作业名 (default | quantile | factorsim_match)",
    )
    parser.add_argument(
        "--factor",
        type=str,
        default=None,
        help="单因子 DSL 表达式 (在 rq 面板求值并自动经 SignalBridge 投影至 barra)",
    )
    parser.add_argument(
        "--factor-file",
        type=Path,
        default=None,
        help="因子定义文件路径 (.json / .yaml / .py)",
    )
    parser.add_argument(
        "--pred",
        type=Path,
        default=None,
        help="ML/DL 样本外预测长表 Parquet 路径 (含 trading_date, symbol, score/pred)",
    )
    parser.add_argument(
        "--signal",
        type=Path,
        default=None,
        help="外部宽表信号 Parquet 路径 (如 shifted_signal.parquet)",
    )
    parser.add_argument(
        "--postprocess",
        type=str,
        default="standard",
        help="单因子声明式后处理管线 (预设名或逗号简写)",
    )
    parser.add_argument(
        "--probe-only",
        action="store_true",
        help="仅在 rq 域执行静态探针校验并输出 JSON 报告",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["barra_qp", "quantile"],
        default=None,
        help="覆盖组合构建模式 (barra_qp | quantile)",
    )
    parser.add_argument(
        "--align",
        type=str,
        choices=["T_close", "T_exec"],
        default=None,
        help="覆盖 SignalBridge 时间对齐模式 (T_close | T_exec)",
    )
    parser.add_argument("--start-date", type=str, default=None, help="回测起始日期 (YYYY-MM-DD)")
    parser.add_argument("--end-date", type=str, default=None, help="回测结束日期 (YYYY-MM-DD)")
    parser.add_argument(
        "--fee-mode",
        type=str,
        choices=["asymmetric", "flat"],
        default=None,
        help="覆盖费率模式 (asymmetric | flat)",
    )
    parser.add_argument("--fee-rate", type=float, default=None, help="覆盖 flat 单边费率")
    parser.add_argument("--init-cash", type=float, default=None, help="覆盖初始资金")
    parser.add_argument("--reserve-cash", type=float, default=None, help="覆盖现金预留比例 (0, 1]")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/stocks/backtest.yaml"),
        help="A 股回测配置文件路径",
    )
    parser.add_argument(
        "--fees",
        type=Path,
        default=Path("configs/stocks/fees.yaml"),
        help="A 股费率配置文件路径",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="回测产物输出目录 (默认 runs/stocks_bt_YYYYMMDD_HHMMSS)",
    )
    parser.add_argument(
        "--synthetic",
        action="store_true",
        help="使用合成演示面板运行冒烟验证 (不读取磁盘 2GB Parquet)",
    )
    return parser.parse_args(args)


def _resolve_factor_definition(args: argparse.Namespace) -> FactorDefinition:
    if args.factor_file is not None:
        defn = FactorDefinition.from_file(args.factor_file)
        if args.postprocess is not None:
            defn.postprocess = parse_postprocess_pipeline(args.postprocess)
        return defn

    expr = args.factor or "-ts_sum(c2c_ret, 5)"
    return FactorDefinition(
        name="stocks_cli_factor",
        factor_type="expression",
        expression=expr,
        postprocess=parse_postprocess_pipeline(args.postprocess),
    )


def _slice_panel_and_scores(
    panel: Panel,
    scores: np.ndarray,
    start_d: date | None,
    end_d: date | None,
) -> tuple[Panel, np.ndarray]:
    if start_d is None and end_d is None:
        return panel, scores
    iso_dates = [str(d)[:10] for d in panel.dates]
    s_iso = start_d.isoformat() if start_d is not None else None
    e_iso = end_d.isoformat() if end_d is not None else None
    keep = np.array(
        [(s_iso is None or d >= s_iso) and (e_iso is None or d <= e_iso) for d in iso_dates],
        dtype=bool,
    )
    if not np.any(keep):
        raise ValueError(f"日期过滤 [{s_iso}, {e_iso}] 后无任何有效交易日")
    if np.all(keep):
        return panel, scores

    sliced = Panel(
        dates=panel.dates[keep],
        symbols=list(panel.symbols),
        fields={k: np.asfortranarray(v[keep, :]) for k, v in panel.fields.items()},
        mask=np.asfortranarray(panel.mask[keep, :]),
        labels={k: np.asfortranarray(v[keep, :]) for k, v in panel.labels.items()},
        aux_fields={k: np.asfortranarray(v[keep, :]) for k, v in panel.aux_fields.items()},
        meta=dict(panel.meta),
    )
    return sliced, np.asfortranarray(scores[keep, :])


def _load_pred_long_table_matrix(pred_path: Path) -> tuple[np.ndarray, list[str], list[str]]:
    df = pl.read_parquet(pred_path)
    col_score = "score" if "score" in df.columns else "pred"
    if col_score not in df.columns:
        raise ValueError(f"预测表 {pred_path} 缺少 'score' 或 'pred' 列: {df.columns}")

    dates = sorted({str(d)[:10] for d in df["trading_date"].to_list()})
    syms = sorted({str(s) for s in df["symbol"].to_list()})
    d_map = {d: i for i, d in enumerate(dates)}
    s_map = {s: j for j, s in enumerate(syms)}
    mat = np.full((len(dates), len(syms)), np.nan, dtype=np.float64, order="F")
    for d_val, s_val, v_val in df.select(["trading_date", "symbol", col_score]).iter_rows():
        if v_val is None:
            continue
        ti = d_map.get(str(d_val)[:10])
        nj = s_map.get(str(s_val))
        if ti is not None and nj is not None:
            mat[ti, nj] = float(v_val)
    return mat, dates, syms


def _apply_cli_overrides(
    bt_cfg: StockBacktestConfig,
    fees_cfg: StockFeesConfig,
    args: argparse.Namespace,
    root: Path,
) -> tuple[StockBacktestConfig, StockFeesConfig]:
    if args.init_cash is not None:
        bt_cfg = replace(bt_cfg, capital=replace(bt_cfg.capital, init_cash=float(args.init_cash)))
    if args.mode is not None:
        bt_cfg = replace(bt_cfg, portfolio=replace(bt_cfg.portfolio, mode=str(args.mode)))
    if args.reserve_cash is not None:
        bt_cfg = replace(
            bt_cfg,
            execution=replace(bt_cfg.execution, reserve_cash=float(args.reserve_cash)),
        )
    if args.start_date is not None:
        bt_cfg = replace(bt_cfg, start_date=date.fromisoformat(args.start_date))
    if args.end_date is not None:
        bt_cfg = replace(bt_cfg, end_date=date.fromisoformat(args.end_date))
    if args.align is not None:
        bt_cfg = replace(
            bt_cfg,
            signal_bridge=replace(bt_cfg.signal_bridge, input_signal_alignment=str(args.align)),
        )
    elif args.signal is not None:
        bt_cfg = replace(
            bt_cfg,
            signal_bridge=replace(bt_cfg.signal_bridge, input_signal_alignment="T_exec"),
        )
    if args.signal is not None:
        sig_p = args.signal if args.signal.is_absolute() else (root / args.signal)
        bt_cfg = replace(
            bt_cfg,
            signal_bridge=replace(
                bt_cfg.signal_bridge,
                source_dataset="external_parquet",
                external_signal_path=sig_p,
            ),
        )

    fee_updates: dict[str, Any] = {}
    if args.fee_mode is not None:
        fee_updates["fee_mode"] = str(args.fee_mode)
    if args.fee_rate is not None:
        fee_updates["fee_rate"] = float(args.fee_rate)
    if fee_updates:
        fees_cfg = replace(fees_cfg, **fee_updates)
    return bt_cfg, fees_cfg


def main(cli_args: list[str] | None = None, root_dir: Path | None = None) -> int:
    args = parse_args(cli_args)
    root = Path(root_dir) if root_dir is not None else Path(__file__).resolve().parent.parent
    profile = StocksProfile(root=root)

    if args.probe_only:
        defn = _resolve_factor_definition(args)
        rq_panel = (
            create_synthetic_stocks_panel(n_days=120, n_symbols=12, seed=42)
            if args.synthetic
            else profile.load_panel(PanelSpec(version="rq"))
        )
        rep = probe_factor(defn, panel=rq_panel)
        print(json.dumps(rep, indent=2, ensure_ascii=False))
        return 0 if rep.get("status") == "ok" else 1

    cfg_path = args.config if args.config.is_absolute() else (root / args.config)
    fees_path = args.fees if args.fees.is_absolute() else (root / args.fees)
    bt_cfg = (
        load_stock_backtest_config(cfg_path, root=root)
        if cfg_path.exists()
        else StockBacktestConfig()
    )
    fees_cfg = load_stock_fees_config(fees_path) if fees_path.exists() else StockFeesConfig()
    bt_cfg, fees_cfg = apply_stock_backtest_profile(bt_cfg, fees_cfg, args.profile, root=root)
    bt_cfg, fees_cfg = _apply_cli_overrides(bt_cfg, fees_cfg, args, root)

    # 所有 A 股执行统一使用 barra 与 vwap_open_30m 时钟（合成冒烟除外）
    if not args.synthetic and bt_cfg.execution.entry_window != "vwap_open_30m":
        bt_cfg = replace(bt_cfg, execution=replace(bt_cfg.execution, entry_window="vwap_open_30m"))

    if args.synthetic:
        barra_panel = create_synthetic_stocks_panel(n_days=120, n_symbols=12, seed=42)
        if bt_cfg.portfolio.mode == "barra_qp":
            bt_cfg = replace(bt_cfg, portfolio=replace(bt_cfg.portfolio, mode="quantile"))
        bt_cfg = replace(bt_cfg, execution=replace(bt_cfg.execution, entry_window="open"))
    else:
        print("[INFO] 正在加载 A 股统一执行面板 `barra` (vwap_open_30m)...")
        barra_panel = profile.load_panel(PanelSpec(version="barra"))

    align = bt_cfg.signal_bridge.input_signal_alignment
    min_ratio = bt_cfg.signal_bridge.min_symbol_match_ratio
    probe_rep: dict[str, Any] | None = None
    v4_stats: dict[str, Any] | None = None
    bridge_report: BridgeAuditReport | None = None

    has_factor_input = args.factor is not None or args.factor_file is not None
    ext_sig = bt_cfg.signal_bridge.external_signal_path

    if args.pred is not None:
        pred_path = args.pred if args.pred.is_absolute() else (root / args.pred)
        print(f"[INFO] 读取预测长表 {pred_path} 并经 SignalBridge ({align}) 投影至 barra...")
        mat, p_dates, p_syms = _load_pred_long_table_matrix(pred_path)
        scores_barra, bridge_report = SignalBridge.project(
            scores_src=mat,
            source_dates=p_dates,
            source_symbols=p_syms,
            target_panel=barra_panel,
            input_signal_alignment=align,
            source_dataset=f"pred:{pred_path.name}",
            min_symbol_match_ratio=min_ratio,
        )
    elif not has_factor_input and ext_sig is not None:
        print(f"[INFO] 读取外部信号宽表 {ext_sig} 并经 SignalBridge ({align}) 投影至 barra...")
        scores_barra, bridge_report = SignalBridge.from_parquet(
            ext_sig,
            barra_panel,
            input_signal_alignment=align,
            min_symbol_match_ratio=min_ratio,
        )
    else:
        if not has_factor_input:
            print("[INFO] 未指定外部信号或因子，默认使用 '-ts_sum(c2c_ret, 5)' 进行演示回测")
        defn = _resolve_factor_definition(args)
        rq_panel = barra_panel if args.synthetic else profile.load_panel(PanelSpec(version="rq"))
        print(f"[INFO] 在 rq 域求值因子 ({defn.name}) 并计算统计归因...")
        scores_rq, probe_rep = evaluate_factor_definition(defn, rq_panel)
        v4_stats = evaluate_factor_v4_stats(scores_rq, rq_panel)
        print(f"[INFO] 经 SignalBridge ({align}) 将 rq 信号单向投影至 barra...")
        scores_barra, bridge_report = SignalBridge.from_panel(
            rq_panel,
            scores_rq,
            barra_panel,
            input_signal_alignment=align,
            min_symbol_match_ratio=min_ratio,
        )

    exec_panel, exec_scores = _slice_panel_and_scores(
        barra_panel, scores_barra, bt_cfg.start_date, bt_cfg.end_date
    )

    print(
        f"[INFO] 构建目标权重: mode={bt_cfg.portfolio.mode}, "
        f"T={exec_panel.T} 日 ({exec_panel.dates[0]} ~ {exec_panel.dates[-1]}), N={exec_panel.N}"
    )
    weights = profile.build_weights(
        exec_scores,
        np.zeros_like(exec_scores),
        exec_panel.mask,
        config=bt_cfg.portfolio,
        panel=exec_panel,
        reserve_cash=bt_cfg.execution.reserve_cash,
    )

    print(
        f"[INFO] 调用 Rust 状态机撮合: entry_window={bt_cfg.execution.entry_window}, "
        f"fee_mode={fees_cfg.fee_mode}, reserve_cash={bt_cfg.execution.reserve_cash}"
    )
    result = run_stock_backtest_pipeline(exec_panel, weights, config=bt_cfg, fees=fees_cfg)

    out_dir = (
        Path(args.output_dir)
        if args.output_dir is not None
        else (root / "runs" / f"stocks_bt_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    html_file = out_dir / "report.html"
    generate_html_report(
        result,
        html_file,
        title=f"FactorLibrary A 股回测报告 ({args.profile}) - {out_dir.name}",
        v4_stats=v4_stats,
        probe_report=probe_rep,
    )
    csv_file = out_dir / "daily_equity.csv"
    result.to_dataframe().write_csv(csv_file)

    summary = summary_dict(result, v4_stats=v4_stats, probe_report=probe_rep)
    summary["profile"] = args.profile
    summary["portfolio_mode"] = bt_cfg.portfolio.mode
    summary["entry_window"] = bt_cfg.execution.entry_window
    summary["fee_mode"] = fees_cfg.fee_mode
    summary["fee_rate"] = fees_cfg.effective_flat_fee_rate
    summary["reserve_cash"] = bt_cfg.execution.reserve_cash
    if bridge_report is not None:
        summary["bridge_audit"] = bridge_report.to_dict()

    metrics_file = out_dir / "metrics.json"
    metrics_file.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"[PASS] 诊断报告已落盘: {html_file}")
    print(f"[PASS] 指标汇总已落盘: {metrics_file}")
    print("\n" + result.summary())
    return 0


if __name__ == "__main__":
    sys.exit(main())
