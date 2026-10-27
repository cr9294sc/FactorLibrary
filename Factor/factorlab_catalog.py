#!/usr/bin/env python3
"""FactorLab Catalog CLI - Search and inspect variables and operators offline or online."""

import os
import sys
import json
import argparse

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_FILE = os.path.join(BASE_DIR, "data", "catalog_full.json")


def load_catalog():
    if not os.path.exists(DATA_FILE):
        print(f"[!] Catalog cache not found at {DATA_FILE}. Run 'python3 scripts/fetch_catalog.py' first.")
        sys.exit(1)
    with open(DATA_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def cmd_stats(args, catalog):
    summary = catalog.get("summary", {})
    print("=" * 60)
    print("  FactorLab Catalog Summary")
    print("=" * 60)
    print(f"  Last Fetched    : {catalog.get('fetched_at', 'unknown')}")
    print(f"  Variables       : {summary.get('variables_count', 0)}")
    print(f"  Functions       : {summary.get('functions_count', 0)}")
    print(f"  Postprocess Ops : {summary.get('postprocess_count', 0)}")
    print(f"  Total Items     : {summary.get('total_items', 0)}")
    print("=" * 60)


def cmd_search(args, catalog):
    q = args.query.lower().strip()
    print(f"[*] Searching catalog for query: '{q}'\n")

    matched_vars = []
    for v in catalog.get("variables", []):
        text = f"{v.get('short_name', '')} {v.get('actual_name', '')} {v.get('category_label', '')} {v.get('category', '')}".lower()
        if q in text:
            matched_vars.append(v)

    matched_funcs = []
    for f in catalog.get("functions", []):
        aliases = " ".join(f.get("aliases", []))
        text = f"{f.get('name', '')} {f.get('category', '')} {f.get('category_label', '')} {f.get('description', '')} {aliases}".lower()
        if q in text:
            matched_funcs.append(f)

    matched_post = []
    for p in catalog.get("postprocess_operators", []):
        text = f"{p.get('name', '')} {p.get('description', '')}".lower()
        if q in text:
            matched_post.append(p)

    if matched_funcs:
        print(f"--- Functions / 算子 ({len(matched_funcs)}) ---")
        for f in matched_funcs[:25]:
            params = ", ".join(p.get("name") for p in f.get("parameters", []))
            print(f"  [Func] {f['name']}({params}) - {f.get('category_label', '')} : {f.get('description', '').split('。')[0]}")
        if len(matched_funcs) > 25:
            print(f"  ... and {len(matched_funcs) - 25} more functions.")
        print()

    if matched_post:
        print(f"--- Postprocess / 后处理算子 ({len(matched_post)}) ---")
        for p in matched_post:
            print(f"  [Post] {p['name']} : {p.get('description', '')}")
        print()

    if matched_vars:
        print(f"--- Variables / 变量 ({len(matched_vars)}) ---")
        for v in matched_vars[:30]:
            print(f"  [Var]  {v['short_name']} -> {v.get('actual_name')} ({v.get('frequency', '-')}, {v.get('category_label', '-')})")
        if len(matched_vars) > 30:
            print(f"  ... and {len(matched_vars) - 30} more variables.")
        print()

    if not matched_funcs and not matched_post and not matched_vars:
        print("  (No matching entries found)")


def cmd_info(args, catalog):
    name = args.name.strip()
    lower_name = name.lower()

    # 1. Search functions
    for f in catalog.get("functions", []):
        if f.get("name", "").lower() == lower_name or lower_name in [a.lower() for a in f.get("aliases", [])]:
            print("=" * 60)
            print(f"  Function: {f['name']}")
            print("=" * 60)
            print(f"Category      : {f.get('category')} ({f.get('category_label')})")
            if f.get("aliases"):
                print(f"Aliases       : {', '.join(f.get('aliases'))}")
            print(f"Description   : {f.get('description')}")
            print(f"Window Needed : {f.get('needs_window')}")
            print(f"Input Shape   : {f.get('input')}")
            print(f"Return Type   : {f.get('returns')}")
            print(f"NaN Policy    : {f.get('missing_policy')}")
            print("\nParameters:")
            for p in f.get("parameters", []):
                req = "required" if p.get("required") else "optional"
                default = f", default={p.get('default')}" if p.get("default") is not None else ""
                print(f"  - {p.get('name')} ({p.get('kind')}, {req}{default}): {p.get('description')}")
            if f.get("example"):
                print("\nExecution Example:")
                print(f.get("example"))
            print("=" * 60)
            return

    # 2. Search postprocess
    for p in catalog.get("postprocess_operators", []):
        if p.get("name", "").lower() == lower_name:
            doc = p.get("documentation", {})
            post_c = doc.get("postprocess_contract", {})
            print("=" * 60)
            print(f"  Postprocess Operator: {p['name']}")
            print("=" * 60)
            print(f"Description   : {p.get('description')}")
            if doc.get("formula"):
                print(f"Formula       : {doc.get('formula')}")
            print(f"Axis          : {doc.get('axis')}")
            print(f"Implicit Deps : {', '.join(post_c.get('implicit_dependencies', [])) or 'None'}")
            print(f"NaN Policy    : {doc.get('nan_policy')}")
            if post_c.get("note"):
                print(f"Notice        : {post_c.get('note')}")
            print("=" * 60)
            return

    # 3. Search variables
    for v in catalog.get("variables", []):
        if v.get("short_name", "").lower() == lower_name or v.get("actual_name", "").lower() == lower_name:
            print("=" * 60)
            print(f"  Variable: {v['short_name']}")
            print("=" * 60)
            print(f"Actual Name   : {v.get('actual_name')}")
            print(f"Category      : {v.get('category')} ({v.get('category_label')})")
            print(f"Frequency     : {v.get('frequency')}")
            print(f"Kind          : {v.get('kind')}")
            print(f"Axes          : {v.get('axes')}")
            print(f"Missing Policy: {v.get('missing_policy')}")
            print("=" * 60)
            return

    print(f"[!] No item named '{name}' found in catalog.")


def main():
    parser = argparse.ArgumentParser(description="FactorLab Catalog CLI")
    subparsers = parser.add_subparsers(dest="command")

    # stats
    subparsers.add_parser("stats", help="Show catalog summary counts")

    # search
    search_p = subparsers.add_parser("search", help="Search variables and functions")
    search_p.add_argument("query", help="Keyword to search")

    # info
    info_p = subparsers.add_parser("info", help="Get full specification of a variable or operator")
    info_p.add_argument("name", help="Name of variable or function")

    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        sys.exit(0)

    catalog = load_catalog()

    if args.command == "stats":
        cmd_stats(args, catalog)
    elif args.command == "search":
        cmd_search(args, catalog)
    elif args.command == "info":
        cmd_info(args, catalog)


if __name__ == "__main__":
    main()
