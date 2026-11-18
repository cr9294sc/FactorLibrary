#!/usr/bin/env python3
"""FactorLibrary Walk-Forward 机器学习多因子合成运行入口 (scripts/run_ml.py)。

用法:
    python scripts/run_ml.py
    python scripts/run_ml.py --recipe rcp_9f3a2c10abcd
    python scripts/run_ml.py --market stocks --start-year 2023 --end-year 2024 --no-strict-min-years
    python scripts/run_ml.py --config configs/futures/ml_default.yaml --panel data/derived/futures/v1
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

from factorlib.core.config import MLConfig, load_ml_config, ml_config_from_dict
from factorlib.core.registry import FactorRegistry, ResearchRecipeRecord
from factorlib.core.runs import RunTracker
from factorlib.markets.profile import PanelSpec
from factorlib.markets.registry import get_profile
from factorlib.ml.pipeline import WalkForwardPipeline, WalkForwardResult


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
        description="FactorLibrary Walk-Forward 机器学习多因子合成入口"
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
        help="ML 配置文件路径 (默认按市场选取 configs/<market>/ml_default.yaml)",
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
        help="指定实验 run_id (默认自动按时间戳生成 ml_YYYYMMDD_HHMMSS)",
    )
    parser.add_argument(
        "--start-year",
        type=int,
        default=None,
        help="覆盖预测起始年份 (例如 2018)",
    )
    parser.add_argument(
        "--end-year",
        type=int,
        default=None,
        help="覆盖预测结束年份 (例如 2026)",
    )
    parser.add_argument(
        "--train-window",
        type=str,
        choices=["expanding", "rolling"],
        default=None,
        help="覆盖训练窗口模式 (expanding 或 rolling)",
    )
    parser.add_argument(
        "--purge-window",
        type=int,
        default=None,
        help="覆盖 Purge 隔离天数 h",
    )
    parser.add_argument(
        "--strict-min-years",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="是否强制要求历史训练年份满足 min_train_years (默认 True)",
    )
    parser.add_argument(
        "--out-parquet",
        type=Path,
        default=None,
        help="指定输出的 pred_panel.parquet 路径 (默认位于本次 run 目录中)",
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


def _apply_cli_overrides(cfg: MLConfig, args: argparse.Namespace) -> MLConfig:
    wf_updates: dict[str, object] = {}
    if args.start_year is not None:
        wf_updates["start_predict_year"] = int(args.start_year)
    if args.end_year is not None:
        wf_updates["end_predict_year"] = int(args.end_year)
    if args.train_window is not None:
        wf_updates["train_window"] = str(args.train_window)
    if args.purge_window is not None:
        wf_updates["purge_window"] = int(args.purge_window)

    if wf_updates:
        new_wf = dataclasses.replace(cfg.walk_forward, **wf_updates)  # type: ignore[arg-type]
        return dataclasses.replace(cfg, walk_forward=new_wf)
    return cfg


def _build_ml_config_from_recipe(
    recipe: ResearchRecipeRecord,
    base_cfg_path: Path | None,
) -> MLConfig:
    raw: dict[str, Any] = {}
    if base_cfg_path is not None and base_cfg_path.exists():
        loaded = yaml.safe_load(base_cfg_path.read_text(encoding="utf-8"))
        if isinstance(loaded, dict):
            raw = loaded
    raw = _deep_merge(raw, recipe.params)
    raw["target_label"] = recipe.target_label
    return ml_config_from_dict(raw)


def _format_markdown_report(
    result: WalkForwardResult,
    cfg: MLConfig,
    run_id: str,
    elapsed_sec: float,
    market: str = "futures",
    recipe_id: str | None = None,
) -> str:
    lines = [
        f"# Walk-Forward 机器学习多因子合成报告 (`{run_id}`)",
        "",
        "## 1. 运行配置与环境",
        f"- **市场域**: `{market}`",
        f"- **实验配方 ID**: `{recipe_id or 'cli_yaml'}`",
        f"- **预测目标标签**: `{cfg.target_label}`",
        f"- **预测时段**: `{cfg.walk_forward.start_predict_year}` ~ `{cfg.walk_forward.end_predict_year}`",
        f"- **窗口模式**: `{cfg.walk_forward.train_window}`",
        f"- **时序净化隔离带 (Purge)**: `{cfg.walk_forward.purge_window}` 天",
        f"- **参与特征因子数**: `{len(result.feature_names)}` 个",
        f"- **总耗时**: `{elapsed_sec:.2f}s`",
        "",
        "## 2. 样本外模型综合表现对账 (Out-Of-Sample)",
        "",
        result.summary_table(),
        "",
        "## 3. 逐年切片滚动对账明细",
        "",
        "| 折索引 | 预测年份 | 训练样本数 | 验证样本数 | 测试样本数 | EqualWeight | RollingICIR | Ridge | LightGBM |",
        "| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |",
    ]

    for f in result.fold_metrics:
        m = f.get("models", {})
        lines.append(
            f"| {f['fold_idx']} | {f['predict_year']} | {f['n_train']} | {f['n_val']} | {f['n_test']} | "
            f"{m.get('equal_weight', 0.0):+.4f} | "
            f"{m.get('rolling_icir', 0.0):+.4f} | "
            f"{m.get('ridge', 0.0):+.4f} | "
            f"{m.get('lightgbm', 0.0):+.4f} |"
        )

    lines.extend(["", "## 4. 因子库输入特征清单", ""])
    for i, name in enumerate(result.feature_names, 1):
        lines.append(f"{i}. `{name}`")

    return "\n".join(lines) + "\n"


def run_ml_cli(args: argparse.Namespace, root_dir: Path | None = None) -> int:
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
        if recipe.kind != "ml":
            logging.error("配方 %s 类型为 %r，无法用于 ML 合成任务", recipe.recipe_id, recipe.kind)
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

    default_cfg_path = profile.config_path("ml")
    cfg_path = (
        (args.config if args.config.is_absolute() else (root / args.config))
        if args.config is not None
        else default_cfg_path
    )

    if recipe is not None:
        cfg = _build_ml_config_from_recipe(recipe, cfg_path if cfg_path.exists() else None)
        extensions = list(dict.fromkeys([*recipe.extensions, *(args.extension or [])]))
        feature_set: dict[str, Any] | None = recipe.feature_set or None
        strict_min_years = bool(recipe.params.get("strict_min_years", args.strict_min_years))
        panel_override = recipe.params.get("panel_dir")
        default_panel_path = (
            Path(panel_override)
            if panel_override and Path(panel_override).is_absolute()
            else (root / panel_override if panel_override else profile.paths().panel_dir)
        )
    else:
        if not cfg_path.exists():
            logging.error("ML 配置文件不存在: %s", cfg_path)
            return 1
        cfg = load_ml_config(cfg_path)
        extensions = list(dict.fromkeys(args.extension or []))
        feature_set = None
        strict_min_years = bool(args.strict_min_years)
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
            logging.error("研究面板目录不存在: %s", panel_path)
            return 1

        logging.info(
            "正在加载 [%s] 研究面板: %s (extensions=%s)...", market, panel_path, extensions
        )
        panel = profile.load_panel(PanelSpec(panel_dir=panel_path, extensions=extensions))
    logging.info(
        "面板加载完成: T=%d 交易日, N=%d 标的, 日期范围: %s ~ %s",
        panel.T,
        panel.N,
        panel.dates[0],
        panel.dates[-1],
    )

    data_version = str(panel.meta.get("panel_version", recipe.data_version if recipe else "v1"))

    tracker = RunTracker(
        kind="ml",
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
        pipeline = WalkForwardPipeline.from_registry(
            panel=panel,
            config=cfg,
            registry=registry,
            market=market,
            feature_set=feature_set,
        )
        result = pipeline.run(strict_min_years=strict_min_years, tracker=tracker)
        elapsed_sec = time.perf_counter() - t0

        # 保存输出结果
        out_parquet = (
            args.out_parquet
            if args.out_parquet is not None
            else (tracker.run_dir / "pred_panel.parquet")
        )
        result.to_parquet(out_parquet)
        logging.info("样本外预测矩阵已导出至: %s", out_parquet)

        # 保存指标与 Markdown 报告
        tracker.save_json("metrics.json", result.metrics)
        tracker.save_json("fold_metrics.json", result.fold_metrics)
        tracker.save_json("feature_importances.json", result.feature_importances)

        report_md = _format_markdown_report(
            result,
            cfg,
            tracker.run_id,
            elapsed_sec,
            market=market,
            recipe_id=recipe.recipe_id if recipe is not None else None,
        )
        tracker.save_text("summary.md", report_md)
        tracker.finish(status="completed")

    except Exception as exc:
        logging.exception("ML 多因子合成运行异常: %s", exc)
        tracker.finish(status="failed")
        return 1

    print("\n" + "=" * 95)
    print(
        f"Walk-Forward 机器学习多因子合成完成! [{market}] Run ID: {tracker.run_id} | 耗时: {elapsed_sec:.2f}s"
    )
    print("=" * 95)
    print(result.summary_table())
    print("=" * 95 + "\n")

    return 0


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="[%(asctime)s] [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )
    return run_ml_cli(args)


if __name__ == "__main__":
    sys.exit(main())
