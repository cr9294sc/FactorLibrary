#!/usr/bin/env python3
"""FactorLibrary 统一因子目录与环境上下文检索工具 (scripts/catalog.py)。

对标 FactorLab MCP 工作流的 Stage 0 (`get_context`) 与 Stage 1 (`catalog_stats` / `catalog_search` / `catalog_info`)。

用法示例:
    python scripts/catalog.py --action context
    python scripts/catalog.py --action stats --catalog all
    python scripts/catalog.py --action search --catalog postprocess
    python scripts/catalog.py --action info --name riskNeutMulti
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from factorlib.core.catalog import FactorCatalog


def parse_args(args: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="FactorLibrary 因子研究字段、算子、后处理与风格目录检索 CLI",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--panel-dir",
        type=Path,
        default=Path("data/derived/futures/v1"),
        help="标准研究面板目录",
    )
    parser.add_argument(
        "--action",
        type=str,
        choices=["context", "stats", "search", "info"],
        default="stats",
        help="执行操作: context (环境上下文) | stats (统计) | search (模糊检索) | info (详情)",
    )
    parser.add_argument(
        "--catalog",
        type=str,
        default="all",
        help="目录过滤: all | variables | functions | postprocess | styles",
    )
    parser.add_argument(
        "--query",
        type=str,
        default="",
        help="search 模式下的关键词",
    )
    parser.add_argument(
        "--category",
        type=str,
        default="",
        help="search 模式下的子分类过滤 (如 mask, outlier, neutralize, normalize, smooth)",
    )
    parser.add_argument(
        "--name",
        type=str,
        default="",
        help="info 模式下的精确条目名称 (如 MAD, riskNeutMulti, ry_ann)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=50,
        help="search 模式下的最大返回条数",
    )
    return parser.parse_args(args)


def main(cli_args: list[str] | None = None) -> int:
    args = parse_args(cli_args)
    cat = FactorCatalog(panel_dir=args.panel_dir)

    if args.action == "context":
        res = cat.get_context()
    elif args.action == "stats":
        res = cat.stats(catalog=args.catalog)
    elif args.action == "search":
        res = cat.search(
            query=args.query,
            catalog=args.catalog,
            category=args.category,
            limit=args.limit,
        )
    elif args.action == "info":
        if not args.name:
            print("[ERROR] --action info 必须提供 --name 参数", file=sys.stderr)
            return 1
        entry = cat.get_entry(args.name, catalog=args.catalog)
        if entry is None:
            print(f"[ERROR] 未找到目录条目: {args.name!r}", file=sys.stderr)
            return 1
        res = entry
    else:
        return 1

    print(json.dumps(res, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
