#!/usr/bin/env python3
"""一键导出 `factorsim` 兼容策略与作业配置 (scripts/export_factorsim_strategy.py)。

用法示例:
    # 从平台声明式 YAML 导出到 local_backtest/（覆盖旧版 strategy.py）
    python scripts/export_factorsim_strategy.py \
        --config configs/stocks/backtest.yaml \
        --out-dir local_backtest --overwrite

    # 直接用 local_backtest 现有作业参数回灌并规范化（保留其 start_ds/end_ds/fee_rate）
    python scripts/export_factorsim_strategy.py \
        --config local_backtest/config.yaml --out-dir local_backtest --overwrite

    # 仅打印渲染源码，不落盘
    python scripts/export_factorsim_strategy.py --print-only
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "python"))

from factorlib.markets.stocks.strategy_export import (  # noqa: E402
    DEFAULT_BACKTEST_PARAMS,
    DEFAULT_STRATEGY_PARAMS,
    StrategyExportError,
    export_factorsim_strategy,
    load_strategy_yaml,
    render_factorsim_config,
    render_factorsim_strategy,
    resolve_strategy_params,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="导出 factorsim 兼容 strategy.py 与 config.yaml（Phase 3 P1-2）",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/stocks/backtest.yaml"),
        help="平台声明式 YAML（含 strategy / backtest 段）",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("local_backtest"),
        help="导出目录（写入 strategy.py 与 config.yaml）",
    )
    parser.add_argument("--overwrite", action="store_true", help="允许覆盖已存在的目标文件")
    parser.add_argument("--print-only", action="store_true", help="仅打印渲染源码与配置，不写盘")
    parser.add_argument(
        "--json",
        action="store_true",
        help="与 --print-only 联用时以 JSON 输出配置与源码",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    strategy_raw: dict = {}
    backtest_raw: dict = {}
    if args.config.exists():
        try:
            strategy_raw, backtest_raw = load_strategy_yaml(args.config)
        except StrategyExportError as exc:
            print(f"[ERROR] {exc}", file=sys.stderr)
            return 2

    # factorsim 作业配置的嵌套结构（strategy_config / backtest_config）同样可回灌
    if args.config.exists() and not strategy_raw:
        import yaml

        raw = yaml.safe_load(args.config.read_text(encoding="utf-8")) or {}
        strategy_raw = dict(raw.get("strategy_config") or {})
        backtest_raw = dict(raw.get("backtest_config") or {})

    try:
        params = resolve_strategy_params(strategy_raw, backtest_raw)
    except StrategyExportError as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 2

    if args.print_only:
        cfg = render_factorsim_config(params, backtest_raw)
        src = render_factorsim_strategy(params)
        if args.json:
            print(json.dumps({"config": cfg, "strategy_source": src}, ensure_ascii=False, indent=2))
        else:
            print(src)
            print("# ---- config.yaml ----")
            import yaml

            print(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True))
        return 0

    try:
        paths = export_factorsim_strategy(
            args.out_dir,
            params=params,
            backtest=backtest_raw,
            overwrite=args.overwrite,
        )
    except StrategyExportError as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 2

    print("[OK] factorsim 兼容策略导出完成:")
    print(f"  strategy: {paths['strategy']}")
    print(f"  config:   {paths['config']}")
    print("  参数摘要:")
    for key in (
        "power",
        "ind_neutral_weight",
        "vol_weight",
        "memory",
        "buy_penalty",
        "sell_penalty",
        "buy_hurdle",
        "hurdle_penalty",
        "reserve_cash",
    ):
        default_key = key if key in params else None
        if default_key is not None:
            print(f"    {key} = {params[default_key]}")
    print(f"    mode = {params.get('mode', DEFAULT_STRATEGY_PARAMS['mode'])}")
    print(f"    tvr = {params.get('tvr', DEFAULT_STRATEGY_PARAMS['tvr'])}")
    print(f"    fee_rate = {backtest_raw.get('fee_rate', DEFAULT_BACKTEST_PARAMS['fee_rate'])}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
