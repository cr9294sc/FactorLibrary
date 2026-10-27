#!/usr/bin/env python3
"""FactorLibrary 期货实盘语义向量化回测与 6 阶段因子工作流执行脚本 (scripts/run_backtest.py)。

用法示例:
    # 1. 静态探针校验 (Stage 3: 毫秒级语法、隐式依赖与 4 级预热窗口检查)
    python scripts/run_backtest.py --factor "ts_zscore(close, 20)" \
        --postprocess "baseUniMask,trdMask,MAD:3.0,indNeut,zScore" --probe-only

    # 2. 单因子 + 声明式后处理管线 + 双层回测 (Layer 1 v4 统计归因 + Layer 2 实盘逐日盯市)
    python scripts/run_backtest.py --panel-dir data/derived/futures/v1 \
        --factor "ts_zscore(close, 20)" --postprocess standard

    # 3. 从因子定义文件 (.json / .yaml / .py) 运行双层回测
    python scripts/run_backtest.py --panel-dir data/derived/futures/v1 \
        --factor-file my_factor.json

    # 4. 采用机器学习多因子预测进行组合构建与回测
    python scripts/run_backtest.py --panel-dir data/derived/futures/v1 \
        --pred runs/ml_20261020_153000/predictions.parquet \
        --config configs/futures/backtest.yaml
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import polars as pl
from factorlib.core.backtest import run_backtest
from factorlib.core.config import load_backtest_config
from factorlib.core.evaluate import evaluate_factor_v4_stats
from factorlib.core.panel import Panel
from factorlib.core.portfolio import build_portfolio_weights, compute_daily_returns_from_prices
from factorlib.core.probe import (
    FactorDefinition,
    evaluate_factor_definition,
    parse_postprocess_pipeline,
    probe_factor,
)
from factorlib.core.report import generate_html_report, summary_dict
from factorlib.markets.futures.adapter import FuturesMarketAdapter


def parse_args(args: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="FactorLibrary 期货实盘语义组合构建与双层向量化回测流水线",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--panel-dir",
        type=Path,
        default=Path("data/derived/futures/v1"),
        help="标准研究面板存储目录",
    )
    parser.add_argument(
        "--pred",
        type=Path,
        default=None,
        help="机器学习样本外多因子预测长表 Parquet 路径 (含 trading_date, symbol, score)",
    )
    parser.add_argument(
        "--factor",
        type=str,
        default=None,
        help="单因子 DSL 表达式 (与 --pred 二选一，例如 'ts_zscore(close, 20)')",
    )
    parser.add_argument(
        "--factor-file",
        type=Path,
        default=None,
        help="因子定义文件路径 (.json / .yaml / .py，支持 DSL 或 Python compute(context))",
    )
    parser.add_argument(
        "--postprocess",
        type=str,
        default=None,
        help="声明式后处理管线 (预设名 standard/minimal/style_neutral、逗号简写如 'MAD:3.0,indNeut,zScore' 或 JSON 数组)",
    )
    parser.add_argument(
        "--probe-only",
        action="store_true",
        help="仅执行 Stage 3 静态探针校验并输出 JSON 报告，不运行全量回测",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/futures/backtest.yaml"),
        help="回测与风控配置文件路径",
    )
    parser.add_argument(
        "--fees",
        type=Path,
        default=Path("configs/futures/fees.yaml"),
        help="期货品种合约规格与费率成本配置文件路径",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="回测结果输出目录 (默认落盘至 runs/backtest_YYYYMMDD_HHMMSS)",
    )
    return parser.parse_args(args)


def _resolve_factor_definition(args: argparse.Namespace) -> FactorDefinition:
    if args.factor_file is not None:
        defn = FactorDefinition.from_file(args.factor_file)
        if args.postprocess is not None:
            defn.postprocess = parse_postprocess_pipeline(args.postprocess)
        return defn

    expr = args.factor or "ts_zscore(close, 20)"
    return FactorDefinition(
        name="cli_factor",
        factor_type="expression",
        expression=expr,
        postprocess=parse_postprocess_pipeline(args.postprocess),
    )


def main(cli_args: list[str] | None = None) -> int:
    args = parse_args(cli_args)

    # 若指定 --probe-only，直接运行静态探针并打印 JSON
    if args.probe_only:
        defn = _resolve_factor_definition(args)
        rep = probe_factor(defn)
        print(json.dumps(rep, indent=2, ensure_ascii=False))
        return 0 if rep.get("status") == "ok" else 1

    # 1. 验证输入参数
    if args.pred is None and args.factor is None and args.factor_file is None:
        print(
            "[INFO] 未指定 --pred / --factor / --factor-file，默认使用动量因子 'ts_zscore(close, 20)' 进行回测演示"
        )
        args.factor = "ts_zscore(close, 20)"

    if not args.panel_dir.exists():
        print(f"[ERROR] 面板目录不存在: {args.panel_dir}", file=sys.stderr)
        return 1

    cfg = load_backtest_config(args.config)
    adapter = FuturesMarketAdapter(fees_config_path=args.fees)

    # 2. 确定运行产出目录
    if args.output_dir is None:
        run_id = f"backtest_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        out_dir = Path("runs") / run_id
    else:
        out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] 正在从 {args.panel_dir} 加载研究面板...")
    panel = Panel.load(args.panel_dir)
    T, N = panel.shape
    print(f"[INFO] 面板加载完成: {T} 个交易日 x {N} 个品种")

    # 3. 生成或加载信号评分矩阵 scores (T x N)
    scores = np.full((T, N), np.nan, dtype=np.float64, order="F")
    probe_rep = None
    v4_stats = None

    if args.pred is not None:
        print(f"[INFO] 从 {args.pred} 读取多因子预测评分长表...")
        pred_df = pl.read_parquet(args.pred)
        date_map = {str(d): i for i, d in enumerate(panel.dates)}
        sym_map = {s: i for i, s in enumerate(panel.symbols)}

        col_score = "score" if "score" in pred_df.columns else "pred"
        for row in pred_df.iter_rows(named=True):
            d_str = str(row["trading_date"])
            s_str = str(row["symbol"])
            if d_str in date_map and s_str in sym_map:
                r_idx = date_map[d_str]
                c_idx = sym_map[s_str]
                val = row.get(col_score)
                if val is not None:
                    scores[r_idx, c_idx] = float(val)
    else:
        defn = _resolve_factor_definition(args)
        print(
            f"[INFO] 执行静态探针与因子求值 ({defn.factor_type}): "
            f"postprocess={[s.op for s in defn.postprocess]}"
        )
        scores, probe_rep = evaluate_factor_definition(defn, panel)
        v4_stats = evaluate_factor_v4_stats(scores, panel)

    # 4. 构建组合目标连续权重矩阵 W (T x N)
    print("[INFO] 开始组合构建: 波动率平价、收缩协方差、持仓截断与缓冲带调仓...")
    close_field = panel.get_field("close")
    returns = compute_daily_returns_from_prices(close_field)
    weights = build_portfolio_weights(scores, returns, panel.mask, cfg.portfolio)

    # 5. 执行 Rust 期货实盘状态机回测
    print("[INFO] 调用 Rust 内核执行逐日盯市 (Mark-to-Market) 状态机回测...")
    result = run_backtest(panel, weights, config=cfg, adapter=adapter)

    # 6. 生成交互式 HTML 诊断报告与 JSON 汇总
    html_file = out_dir / "report.html"
    generate_html_report(
        result,
        html_file,
        title=f"FactorLibrary 期货回测诊断报告 - {out_dir.name}",
        v4_stats=v4_stats,
        probe_report=probe_rep,
    )
    print(f"[PASS] 交互式 HTML 诊断报告已成功生成: {html_file}")

    # 导出逐日财务流水账 CSV
    df = result.to_dataframe()
    csv_file = out_dir / "daily_equity.csv"
    df.write_csv(csv_file)
    print(f"[PASS] 每日资产流水记录已导出: {csv_file}")

    # 保存元数据与摘要
    summary_file = out_dir / "metrics.json"
    with summary_file.open("w", encoding="utf-8") as fh:
        json.dump(
            summary_dict(result, v4_stats=v4_stats, probe_report=probe_rep),
            fh,
            indent=2,
            ensure_ascii=False,
        )

    print("\n" + result.summary())
    return 0


if __name__ == "__main__":
    sys.exit(main())
