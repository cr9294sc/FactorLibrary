#!/usr/bin/env python3
"""FactorLibrary Walk-Forward 机器学习多因子合成运行入口 (scripts/run_ml.py)。

用法:
    python scripts/run_ml.py
    python scripts/run_ml.py --config configs/futures/ml_default.yaml --panel data/derived/futures/v1
    python scripts/run_ml.py --start-year 2020 --end-year 2024 --no-strict-min-years
"""

from __future__ import annotations

import argparse
import dataclasses
import logging
import sys
import time
from pathlib import Path

# 将 python/ 目录加入 sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "python"))

from factorlib.core.config import MLConfig, load_ml_config
from factorlib.core.panel import Panel
from factorlib.core.registry import FactorRegistry
from factorlib.core.runs import RunTracker
from factorlib.ml.pipeline import WalkForwardPipeline, WalkForwardResult


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="FactorLibrary Walk-Forward 机器学习多因子合成入口"
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/futures/ml_default.yaml"),
        help="ML 配置文件路径 (默认: configs/futures/ml_default.yaml)",
    )
    parser.add_argument(
        "--panel",
        type=Path,
        default=Path("data/derived/futures/v1"),
        help="派生研究面板目录路径 (默认: data/derived/futures/v1)",
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
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="日志级别 (默认: INFO)",
    )
    return parser.parse_args()


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


def _format_markdown_report(
    result: WalkForwardResult,
    cfg: MLConfig,
    run_id: str,
    elapsed_sec: float,
) -> str:
    lines = [
        f"# Walk-Forward 机器学习多因子合成报告 (`{run_id}`)",
        "",
        "## 1. 运行配置与环境",
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


def main() -> int:
    args = parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="[%(asctime)s] [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    root_dir = Path(__file__).resolve().parent.parent
    cfg_path = args.config if args.config.is_absolute() else (root_dir / args.config)
    panel_path = args.panel if args.panel.is_absolute() else (root_dir / args.panel)
    db_path = args.db if args.db.is_absolute() else (root_dir / args.db)
    runs_dir = args.runs_dir if args.runs_dir.is_absolute() else (root_dir / args.runs_dir)

    if not cfg_path.exists():
        logging.error("ML 配置文件不存在: %s", cfg_path)
        return 1
    if not panel_path.exists():
        logging.error("研究面板目录不存在: %s", panel_path)
        return 1

    cfg = load_ml_config(cfg_path)
    cfg = _apply_cli_overrides(cfg, args)

    logging.info("正在加载研究面板: %s ...", panel_path)
    panel = Panel.load(panel_path)
    logging.info(
        "面板加载完成: T=%d 交易日, N=%d 品种, 日期范围: %s ~ %s",
        panel.T,
        panel.N,
        panel.dates[0],
        panel.dates[-1],
    )

    registry = FactorRegistry(db_path=db_path)
    data_version = str(panel.meta.get("panel_version", "v1"))

    tracker = RunTracker(
        kind="ml",
        run_id=args.run_id,
        runs_root=runs_dir,
        config_yaml=cfg.to_yaml(),
        data_version=data_version,
        registry=registry,
    )
    logging.info("实验运行轨迹目录已创建: %s (run_id=%s)", tracker.run_dir, tracker.run_id)

    t0 = time.perf_counter()
    try:
        pipeline = WalkForwardPipeline.from_registry(
            panel=panel,
            config=cfg,
            registry=registry,
        )
        result = pipeline.run(strict_min_years=args.strict_min_years)
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

        report_md = _format_markdown_report(result, cfg, tracker.run_id, elapsed_sec)
        tracker.save_text("summary.md", report_md)
        tracker.finish(status="completed")

    except Exception as exc:
        logging.exception("ML 多因子合成运行异常: %s", exc)
        tracker.finish(status="failed")
        return 1

    print("\n" + "=" * 95)
    print(
        f"Walk-Forward 机器学习多因子合成完成! Run ID: {tracker.run_id} | 耗时: {elapsed_sec:.2f}s"
    )
    print("=" * 95)
    print(result.summary_table())
    print("=" * 95 + "\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())
