#!/usr/bin/env python3
"""FactorLibrary 强类型遗传规划 (STGP) 批量因子挖掘命令行工具 (scripts/run_gp.py)。

用法:
    python scripts/run_gp.py
    python scripts/run_gp.py --recipe rcp_9f3a2c10abcd
    python scripts/run_gp.py --market stocks --population 200 --generations 10 --seed 42
    python scripts/run_gp.py --config configs/futures/gp_default.yaml --panel data/derived/futures/v1
"""

from __future__ import annotations

import argparse
import copy
import dataclasses
import logging
import sys
import time
from pathlib import Path
from typing import Any

import yaml

# 将 python/ 目录加入 sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "python"))

from factorlib.core.config import (
    AdmissionConfig,
    GPConfig,
    gp_config_from_dict,
    load_gp_config,
)
from factorlib.core.registry import FactorRegistry, ResearchRecipeRecord
from factorlib.core.runs import RunTracker
from factorlib.gp.engine import GPEngine
from factorlib.markets.profile import PanelSpec
from factorlib.markets.registry import get_profile


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = copy.deepcopy(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(merged.get(k), dict):
            merged[k] = _deep_merge(merged[k], v)
        else:
            merged[k] = copy.deepcopy(v)
    return merged


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="FactorLibrary 强类型遗传规划 (STGP) 自动化因子挖掘入口"
    )
    parser.add_argument(
        "--recipe",
        type=str,
        default=None,
        help="冻结实验配方 ID (从 SQLite research_recipe 表加载配置与 FeatureSet 快照)",
    )
    parser.add_argument(
        "--market",
        type=str,
        default=None,
        choices=["futures", "stocks"],
        help="目标市场域 (futures | stocks，未传 --recipe 时默认 futures)",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="GP 配置文件路径 (默认按市场选取 configs/<market>/gp_default.yaml)",
    )
    parser.add_argument(
        "--panel",
        type=Path,
        default=None,
        help="派生研究面板目录路径 (默认按市场选取 data/derived/<market>/v1)",
    )
    parser.add_argument(
        "--extension",
        action="append",
        default=[],
        help="挂载的 1m/日频扩展特征包 ID（可多次传入，与配方内扩展包合并）",
    )
    parser.add_argument(
        "--db",
        type=Path,
        default=Path("data/registry/factors.sqlite"),
        help="SQLite 因子注册库路径 (默认: data/registry/factors.sqlite)",
    )
    parser.add_argument(
        "--runs-dir",
        type=Path,
        default=Path("runs"),
        help="实验运行轨迹输出根目录 (默认: runs)",
    )
    parser.add_argument(
        "--run-id",
        type=str,
        default=None,
        help="指定实验 run_id (默认自动按时间戳生成 gp_YYYYMMDD_HHMMSS)",
    )
    parser.add_argument(
        "--population",
        type=int,
        default=None,
        help="覆盖配置中的总种群大小 (例如 1000)",
    )
    parser.add_argument(
        "--generations",
        type=int,
        default=None,
        help="覆盖配置中的演化代数 (例如 50)",
    )
    parser.add_argument(
        "--islands",
        type=int,
        default=None,
        help="覆盖配置中的演化岛屿数 (例如 4)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="覆盖配置中的随机种子 (例如 42)",
    )
    parser.add_argument(
        "--min-valid-icir",
        type=float,
        default=None,
        help="覆盖准入管线验证集最小 |ICIR| 阈值 (默认读取配置文件)",
    )
    parser.add_argument(
        "--register-status",
        type=str,
        default=None,
        choices=["candidate", "accepted"],
        help="入库因子初始状态 (--recipe 模式默认 candidate，传统模式默认 accepted)",
    )
    parser.add_argument(
        "--no-register",
        action="store_true",
        help="仅执行挖掘与准入验证，不将结果写入 SQLite 因子库",
    )
    parser.add_argument(
        "--synthetic",
        action="store_true",
        help="使用内置合成演示面板运行 (当本地派生面板未构建时)",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="日志级别 (默认: INFO)",
    )
    return parser.parse_args(argv)


def _apply_cli_overrides(cfg: GPConfig, args: argparse.Namespace) -> GPConfig:
    updates: dict[str, object] = {}
    if args.population is not None:
        updates["population"] = int(args.population)
    if args.generations is not None:
        updates["generations"] = int(args.generations)
    if args.islands is not None:
        updates["islands"] = int(args.islands)
    if args.seed is not None:
        updates["seed"] = int(args.seed)
    if args.min_valid_icir is not None:
        updates["admission"] = dataclasses.replace(
            cfg.admission,
            min_valid_ic_ir=float(args.min_valid_icir),
        )
    elif isinstance(cfg.admission, AdmissionConfig):
        pass
    if updates:
        return dataclasses.replace(cfg, **updates)  # type: ignore[arg-type]
    return cfg


def _build_gp_config_from_recipe(
    recipe: ResearchRecipeRecord,
    base_cfg_path: Path | None,
) -> GPConfig:
    raw: dict[str, Any] = {}
    if base_cfg_path is not None and base_cfg_path.exists():
        loaded = yaml.safe_load(base_cfg_path.read_text(encoding="utf-8"))
        if isinstance(loaded, dict):
            raw = loaded
    raw = _deep_merge(raw, recipe.params)
    raw["seed"] = int(recipe.seed)
    raw["label"] = recipe.target_label
    if recipe.splits:
        raw["splits"] = copy.deepcopy(recipe.splits)
    return gp_config_from_dict(raw)


def _format_markdown_report(
    engine: GPEngine,
    run_id: str,
    elapsed_sec: float,
    recipe_id: str | None = None,
) -> str:
    summary = engine.last_admission_summary
    lines = [
        f"# STGP 因子挖掘实验报告 (`{run_id}`)",
        "",
        "## 1. 实验配置与概览",
        f"- **市场域**: `{engine.market}`",
        f"- **实验配方 ID**: `{recipe_id or 'cli_yaml'}`",
        f"- **随机种子**: `{engine.cfg.seed}`",
        f"- **预测目标标签**: `{engine.label_name}` (horizon = {engine.horizon})",
        f"- **种群规模**: `{engine.cfg.population}` ({engine.cfg.islands} 岛屿)",
        f"- **演化代数**: `{engine.cfg.generations}`",
        f"- **累计唯一求值表达式数**: `{engine.total_unique_evaluated}`",
        f"- **总运行耗时**: `{elapsed_sec:.2f}s`",
        "",
        "## 2. 五步准入筛选漏斗",
    ]
    if summary is not None:
        lines.extend(
            [
                f"- **Step 1 (Top-K 名人堂初筛)**: `{summary.n_step1_top_k}`",
                f"- **Step 2 (验证集同向 & |ICIR| >= {engine.cfg.admission.min_valid_ic_ir})**: `{summary.n_step2_valid_ic}`",
                f"- **Step 3 (截面换手率 <= {engine.cfg.admission.max_turnover})**: `{summary.n_step3_turnover}`",
                f"- **Step 4 (贪心正交化去相关 max|ρ| < {engine.cfg.admission.max_library_corr})**: `{summary.n_step4_orthogonal}`",
                f"- **Step 5 (原子化入库登记)**: `{summary.n_step5_registered}`",
                "",
                "## 3. 入围合格因子清单",
                "",
                "| ID | 状态 | 推荐方向 | 规范化表达式 | 训练集 ICIR | 验证集 ICIR | 验证集换手率 | 最大相关性 |",
                "|---|---|---|---|---|---|---|---|",
            ]
        )
        for f in summary.admitted:
            fid_str = str(f.factor_id) if f.factor_id is not None else "-"
            lines.append(
                f"| {fid_str} | {f.status} | {f.sign:+d} | `{f.canon_expr}` | {f.train_ic_ir:+.3f} | "
                f"{f.valid_ic_ir:+.3f} | {f.valid_turnover:.2%} | {f.max_corr:.3f} |"
            )
    return "\n".join(lines) + "\n"


def run_gp_cli(args: argparse.Namespace, root_dir: Path | None = None) -> int:
    root = Path(root_dir) if root_dir is not None else Path(__file__).resolve().parent.parent
    db_path = args.db if args.db.is_absolute() else (root / args.db)
    runs_dir = args.runs_dir if args.runs_dir.is_absolute() else (root / args.runs_dir)

    registry = FactorRegistry(db_path=db_path)
    recipe: ResearchRecipeRecord | None = None
    if args.recipe:
        recipe = registry.get_recipe(args.recipe)
        if recipe is None:
            logging.error("未在数据库 %s 中找到配方 ID: %s", db_path, args.recipe)
            return 1
        if recipe.kind != "gp":
            logging.error("配方 %s 类型为 %r，无法用于 GP 挖掘任务", recipe.recipe_id, recipe.kind)
            return 1
        market = recipe.market
        if args.market and args.market != market:
            logging.warning(
                "命令行 --market=%s 与配方所属市场 %s 不一致，强制使用配方市场 %s",
                args.market,
                market,
                market,
            )
    else:
        market = args.market or "futures"

    profile = get_profile(market)
    if hasattr(profile, "root"):
        profile.root = root  # type: ignore[attr-defined]

    default_cfg_path = profile.config_path("gp")
    cfg_path = (
        (args.config if args.config.is_absolute() else (root / args.config))
        if args.config is not None
        else default_cfg_path
    )

    if recipe is not None:
        cfg = _build_gp_config_from_recipe(recipe, cfg_path if cfg_path.exists() else None)
        extensions = list(dict.fromkeys([*recipe.extensions, *(args.extension or [])]))
        feature_set: dict[str, Any] | None = recipe.feature_set or None
        register_status = args.register_status or "candidate"
        panel_override = recipe.params.get("panel_dir")
        default_panel_path = (
            Path(panel_override)
            if panel_override and Path(panel_override).is_absolute()
            else (root / panel_override if panel_override else profile.paths().panel_dir)
        )
    else:
        if not cfg_path.exists():
            logging.error("GP 配置文件不存在: %s", cfg_path)
            return 1
        cfg = load_gp_config(cfg_path)
        extensions = list(dict.fromkeys(args.extension or []))
        feature_set = None
        register_status = args.register_status or "accepted"
        default_panel_path = profile.paths().panel_dir

    cfg = _apply_cli_overrides(cfg, args)

    if args.synthetic:
        from factorlib.core.synthetic import (
            create_synthetic_futures_panel,
            create_synthetic_stocks_panel,
        )

        logging.info("正在创建 [%s] 合成演示研究面板...", market)
        panel = (
            create_synthetic_stocks_panel()
            if market == "stocks"
            else create_synthetic_futures_panel()
        )
        if extensions:
            profile.attach_extensions(panel, extensions)
    else:
        panel_path = (
            (args.panel if args.panel.is_absolute() else (root / args.panel))
            if args.panel is not None
            else default_panel_path
        )
        if not panel_path.exists():
            logging.error("研究面板目录不存在: %s，请先构建对应市场研究面板", panel_path)
            return 1

        logging.info(
            "正在加载 [%s] 研究面板: %s (extensions=%s)...", market, panel_path, extensions
        )
        panel = profile.load_panel(PanelSpec(panel_dir=panel_path, extensions=extensions))
    logging.info(
        "面板加载完成: T=%d 交易日, N=%d 标的, %d 特征字段",
        panel.T,
        panel.N,
        len(panel.fields),
    )

    data_version = str(panel.meta.get("panel_version", recipe.data_version if recipe else "v1"))

    tracker = RunTracker(
        kind="gp",
        run_id=args.run_id,
        runs_root=runs_dir,
        config_yaml=cfg.to_yaml(),
        data_version=data_version,
        registry=registry,
        market=market,
        recipe_id=recipe.recipe_id if recipe is not None else None,
    )
    logging.info(
        "实验运行轨迹目录已创建: %s (run_id=%s, market=%s, recipe=%s)",
        tracker.run_dir,
        tracker.run_id,
        market,
        recipe.recipe_id if recipe is not None else "none",
    )

    t0 = time.perf_counter()
    try:
        gp_engine = GPEngine(
            cfg=cfg,
            panel=panel,
            registry=registry,
            tracker=tracker,
            market=market,
            recipe_id=recipe.recipe_id if recipe is not None else None,
            feature_set=feature_set,
            register_status=register_status,
        )
        admitted = gp_engine.run(
            run_id=tracker.run_id,
            register_to_db=not args.no_register,
        )
        elapsed_sec = time.perf_counter() - t0
        md_report = _format_markdown_report(
            gp_engine,
            tracker.run_id,
            elapsed_sec,
            recipe_id=recipe.recipe_id if recipe is not None else None,
        )
        tracker.save_text("summary.md", md_report)
    except Exception as exc:
        logging.exception("GP 挖掘运行异常中断: %s", exc)
        tracker.finish(status="failed")
        return 1

    print("\n" + "=" * 105)
    print(
        f"STGP 挖掘完成! [{market}] Run ID: {tracker.run_id} | 耗时: {elapsed_sec:.2f}s | "
        f"独立求值: {gp_engine.total_unique_evaluated} | 准入合格: {len(admitted)} (status={register_status})"
    )
    print("=" * 105)
    if admitted:
        print(
            f"{'ID':<6} {'方向':<6} {'Train ICIR':<12} {'Valid ICIR':<12} "
            f"{'换手率':<10} {'MaxCorr':<10} {'规范化表达式'}"
        )
        print("-" * 105)
        for f in admitted:
            fid_str = str(f.factor_id) if f.factor_id is not None else "-"
            print(
                f"{fid_str:<6} {f.sign:+d}    {f.train_ic_ir:+10.3f}  {f.valid_ic_ir:+10.3f}  "
                f"{f.valid_turnover:8.2%}  {f.max_corr:8.3f}  {f.canon_expr}"
            )
    else:
        print("本次挖掘未产生满足全部五步准入阈值的因子（可调大种群代数或检查样本窗口）。")
    print("=" * 105 + "\n")

    return 0


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="[%(asctime)s] [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )
    return run_gp_cli(args)


if __name__ == "__main__":
    sys.exit(main())
