#!/usr/bin/env python3
"""
FactorLibrary 多市场标准研究面板构建脚本 (scripts/build_panel.py)

用法:
    python scripts/build_panel.py
    python scripts/build_panel.py --market futures --quick
    python scripts/build_panel.py --market stocks --version v1
    python scripts/build_panel.py --start-date 2024-01-01 --end-date 2024-03-31
    python scripts/build_panel.py --config configs/futures/data.yaml --universe configs/futures/universe.yaml
"""

from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import replace
from datetime import date
from pathlib import Path

# 将 python/ 目录加入 sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "python"))

from factorlib.core.config import (
    load_data_config,
    load_stock_data_config,
    load_universe_config,
)
from factorlib.markets.futures.etl.build import build_panel
from factorlib.markets.stocks.ares_bridge import build_lb_barra_panel
from factorlib.markets.stocks.panel_loader import build_stock_panel


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="FactorLibrary 多市场标准研究面板构建工具")
    parser.add_argument(
        "--market",
        type=str,
        default="futures",
        choices=["futures", "stocks"],
        help="目标市场域 (futures | stocks, 默认: futures)",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="数据配置 YAML 路径 (默认按市场选取 configs/<market>/data.yaml)",
    )
    parser.add_argument(
        "--universe",
        type=Path,
        default=None,
        help="品种池配置 YAML 路径 (默认按市场选取 configs/<market>/universe.yaml)",
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
        "--version",
        type=str,
        default=None,
        help="覆盖派生面板版本号 (如 v1)",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="快速演示模式 (仅构建 2024-02-01 ~ 2024-03-31 两月数据)",
    )
    parser.add_argument(
        "--no-verify",
        action="store_true",
        help="跳过官方复权因子 ex_cum_factor 对账 (仅期货)",
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
    default_cfg = Path(f"configs/{args.market}/data.yaml")
    default_univ = Path(f"configs/{args.market}/universe.yaml")
    raw_cfg = args.config if args.config is not None else default_cfg
    raw_univ = args.universe if args.universe is not None else default_univ

    config_path = raw_cfg if raw_cfg.is_absolute() else (root_dir / raw_cfg)
    universe_path = raw_univ if raw_univ.is_absolute() else (root_dir / raw_univ)

    if not config_path.exists():
        logging.error("数据配置文件不存在: %s", config_path)
        return 1

    overrides: dict[str, object] = {}
    if args.quick:
        overrides["start_date"] = date(2024, 2, 1)
        overrides["end_date"] = date(2024, 3, 31)
        logging.info("已启用 --quick 快速构建模式 (2024-02-01 ~ 2024-03-31)")
    else:
        if args.start_date:
            overrides["start_date"] = date.fromisoformat(args.start_date)
        if args.end_date:
            overrides["end_date"] = date.fromisoformat(args.end_date)

    if args.version:
        overrides["panel_version"] = args.version

    try:
        if args.market == "stocks":
            scfg = load_stock_data_config(
                config_path,
                root=root_dir,
                dataset=args.version,
            )
            if overrides:
                scfg = replace(scfg, **overrides)  # type: ignore[arg-type]
            if scfg.source == "local_backtest":
                panel_df, _labels_df, meta = build_lb_barra_panel(cfg=scfg, show_progress=True)
                title = "FactorLibrary A 股 Barra 独立研究面板构建摘要"
            else:
                panel_df, _labels_df, meta = build_stock_panel(cfg=scfg, show_progress=True)
                title = "FactorLibrary A 股标准研究面板构建摘要"
            panel_dir = scfg.panel_dir
        else:
            cfg = load_data_config(config_path, root=root_dir)
            ucfg = load_universe_config(universe_path) if universe_path.exists() else None
            if overrides:
                cfg = replace(cfg, **overrides)  # type: ignore[arg-type]
            panel_df, _labels_df, meta = build_panel(
                cfg=cfg,
                ucfg=ucfg,
                universe_path=universe_path,
                verify_adjust=not args.no_verify,
                show_progress=True,
            )
            panel_dir = cfg.panel_dir
            title = "FactorLibrary 期货标准研究面板构建摘要"

        print("\n" + "=" * 60)
        print(f"  {title}")
        print("=" * 60)
        print(f"  市场域:           {args.market}")
        print(f"  版本编号:         {meta['panel_version']}")
        print(f"  总记录数:         {meta['total_records']:,} 行")
        print(
            f"  有效交易日:       {meta['total_trading_days']} 天 ({meta['date_range'][0]} ~ {meta['date_range'][1]})"
        )
        print(f"  覆盖标的数量:     {meta['total_symbols']} 个")
        print(f"  在池记录数:       {meta['in_universe_records']:,} 行")
        print(f"  特征矩阵字段:     {meta['config']['panel_version']} ({len(panel_df.columns)} 列)")
        print(f"  预测标签周期:     {meta['config']['label_horizons']}")
        print(f"  总构建耗时:       {meta['build_duration_seconds']:.2f} 秒")
        print(f"  落盘目录:         {panel_dir.resolve()}")
        print("=" * 60 + "\n")
        return 0
    except Exception as exc:
        logging.exception("构建研究面板发生致命异常: %s", exc)
        return 1


if __name__ == "__main__":
    sys.exit(main())
