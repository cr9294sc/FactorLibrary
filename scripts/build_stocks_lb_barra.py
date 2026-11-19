#!/usr/bin/env python3
"""
FactorLibrary local_backtest 独立数据源 (lb_barra_v1) 构建工具 (scripts/build_stocks_lb_barra.py)

用法:
    python scripts/build_stocks_lb_barra.py
    python scripts/build_stocks_lb_barra.py --quick
    python scripts/build_stocks_lb_barra.py --start-date 2020-01-01 --end-date 2026-11-10
"""

from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import replace
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "python"))

from factorlib.core.config import load_stock_data_config
from factorlib.markets.stocks.ares_bridge import build_lb_barra_panel


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="将 local_backtest (.ares + CNE5) 转换为独立标准面板 data/derived/stocks/lb_barra_v1/"
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/stocks/data.yaml"),
        help="A 股数据配置 YAML 路径 (默认: configs/stocks/data.yaml)",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="lb_barra_v1",
        help="目标数据集名称 (默认: lb_barra_v1)",
    )
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=None,
        help="覆盖 local_backtest/runtime_assets 根目录",
    )
    parser.add_argument(
        "--derived-dir",
        type=Path,
        default=None,
        help="覆盖派生面板输出根目录",
    )
    parser.add_argument(
        "--benchmark-csv",
        type=Path,
        default=None,
        help="覆盖基准指数 CSV 路径",
    )
    parser.add_argument(
        "--start-date",
        type=str,
        default=None,
        help="覆盖起始日期 (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--end-date",
        type=str,
        default=None,
        help="覆盖结束日期 (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="快速构建模式 (仅构建 2024-02-01 ~ 2024-03-31 窗口)",
    )
    parser.add_argument(
        "--skip-cov",
        action="store_true",
        help="跳过打包 BarraCNE5_COV/*.parquet 到 risk_store/factor_cov.npz",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="日志级别 (默认: INFO)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="[%(asctime)s] [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    root_dir = Path(__file__).resolve().parent.parent
    cfg_path = args.config if args.config.is_absolute() else (root_dir / args.config)
    if not cfg_path.exists():
        logging.error("配置文件不存在: %s", cfg_path)
        return 1

    cfg = load_stock_data_config(cfg_path, root=root_dir, dataset=args.dataset)
    overrides: dict[str, object] = {}
    if args.raw_dir is not None:
        overrides["raw_dir"] = (
            args.raw_dir if args.raw_dir.is_absolute() else (root_dir / args.raw_dir)
        )
    if args.derived_dir is not None:
        overrides["derived_dir"] = (
            args.derived_dir if args.derived_dir.is_absolute() else (root_dir / args.derived_dir)
        )
    if args.benchmark_csv is not None:
        overrides["benchmark_csv"] = (
            args.benchmark_csv
            if args.benchmark_csv.is_absolute()
            else (root_dir / args.benchmark_csv)
        )
    if args.quick:
        overrides["start_date"] = date(2024, 2, 1)
        overrides["end_date"] = date(2024, 3, 31)
    else:
        if args.start_date:
            overrides["start_date"] = date.fromisoformat(args.start_date)
        if args.end_date:
            overrides["end_date"] = date.fromisoformat(args.end_date)

    if overrides:
        cfg = replace(cfg, **overrides)  # type: ignore[arg-type]

    try:
        panel_df, labels_df, meta = build_lb_barra_panel(
            cfg,
            pack_cov=not args.skip_cov,
            show_progress=True,
        )
        risk = meta.get("risk_store", {})
        print("\n" + "=" * 64)
        print("  FactorLibrary A 股 Barra 独立数据源构建摘要 (lb_barra_v1)")
        print("=" * 64)
        print(f"  数据集 ID:        {meta['dataset_id']} (version={meta['panel_version']})")
        print(f"  总记录数:         {meta['total_records']:,} 行 ({len(panel_df.columns)} 列)")
        print(
            f"  有效交易日:       {meta['total_trading_days']} 天 "
            f"({meta['date_range'][0]} ~ {meta['date_range'][1]})"
        )
        print(f"  覆盖标的数量:     {meta['total_symbols']} 个")
        print(f"  在池记录数:       {meta['in_universe_records']:,} 行")
        print(f"  标签字段数:       {len(labels_df.columns) - 2} 个")
        print(
            f"  风险库 (COV/RET): {risk.get('factor_cov_count', 0)} 日 COV / "
            f"{risk.get('factor_returns_count', 0)} 日 FactorRet"
        )
        print(f"  总构建耗时:       {meta['build_duration_seconds']:.2f} 秒")
        print(f"  落盘目录:         {cfg.panel_dir.resolve()}")
        print("=" * 64 + "\n")
        return 0
    except Exception as exc:
        logging.exception("构建 lb_barra_v1 面板失败: %s", exc)
        return 1


if __name__ == "__main__":
    sys.exit(main())
