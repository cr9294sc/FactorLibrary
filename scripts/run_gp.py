#!/usr/bin/env python3
"""FactorLibrary 强类型遗传规划 (STGP) 批量因子挖掘命令行工具 (scripts/run_gp.py)。

用法:
    python scripts/run_gp.py
    python scripts/run_gp.py --config configs/futures/gp_default.yaml --panel data/derived/futures/v1
    python scripts/run_gp.py --population 200 --generations 10 --seed 42
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

from factorlib.core.config import AdmissionConfig, GPConfig, load_gp_config
from factorlib.core.panel import Panel
from factorlib.core.registry import FactorRegistry
from factorlib.core.runs import RunTracker
from factorlib.gp.engine import GPEngine


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="FactorLibrary 强类型遗传规划 (STGP) 自动化因子挖掘入口"
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/futures/gp_default.yaml"),
        help="GP 配置文件路径 (默认: configs/futures/gp_default.yaml)",
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
        "--no-register",
        action="store_true",
        help="仅执行挖掘与准入验证，不将结果写入 SQLite 因子库",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="日志级别 (默认: INFO)",
    )
    return parser.parse_args()


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


def _format_markdown_report(
    engine: GPEngine,
    run_id: str,
    elapsed_sec: float,
) -> str:
    summary = engine.last_admission_summary
    lines = [
        f"# STGP 因子挖掘实验报告 (`{run_id}`)",
        "",
        "## 1. 实验配置与概览",
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
                "| ID | 推荐方向 | 规范化表达式 | 训练集 ICIR | 验证集 ICIR | 验证集换手率 | 最大相关性 |",
                "|---|---|---|---|---|---|---|",
            ]
        )
        for f in summary.admitted:
            fid_str = str(f.factor_id) if f.factor_id is not None else "-"
            lines.append(
                f"| {fid_str} | {f.sign:+d} | `{f.canon_expr}` | {f.train_ic_ir:+.3f} | "
                f"{f.valid_ic_ir:+.3f} | {f.valid_turnover:.2%} | {f.max_corr:.3f} |"
            )
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
        logging.error("GP 配置文件不存在: %s", cfg_path)
        return 1
    if not panel_path.exists():
        logging.error("研究面板目录不存在: %s，请先运行 scripts/build_panel.py", panel_path)
        return 1

    cfg = load_gp_config(cfg_path)
    cfg = _apply_cli_overrides(cfg, args)

    logging.info("正在加载研究面板: %s ...", panel_path)
    panel = Panel.load(panel_path)
    logging.info(
        "面板加载完成: T=%d 交易日, N=%d 品种, %d 特征字段",
        panel.T,
        panel.N,
        len(panel.fields),
    )

    registry = FactorRegistry(db_path=db_path)
    data_version = str(panel.meta.get("panel_version", "v1"))

    tracker = RunTracker(
        kind="gp",
        run_id=args.run_id,
        runs_root=runs_dir,
        config_yaml=cfg.to_yaml(),
        data_version=data_version,
        registry=registry,
    )
    logging.info("实验运行轨迹目录已创建: %s (run_id=%s)", tracker.run_dir, tracker.run_id)

    t0 = time.perf_counter()
    try:
        gp_engine = GPEngine(
            cfg=cfg,
            panel=panel,
            registry=registry,
            tracker=tracker,
        )
        admitted = gp_engine.run(
            run_id=tracker.run_id,
            register_to_db=not args.no_register,
        )
        elapsed_sec = time.perf_counter() - t0
        md_report = _format_markdown_report(gp_engine, tracker.run_id, elapsed_sec)
        tracker.save_text("summary.md", md_report)
    except Exception as exc:
        logging.exception("GP 挖掘运行异常中断: %s", exc)
        tracker.finish(status="failed")
        return 1

    print("\n" + "=" * 105)
    print(
        f"STGP 挖掘完成! Run ID: {tracker.run_id} | 耗时: {elapsed_sec:.2f}s | "
        f"独立求值: {gp_engine.total_unique_evaluated} | 准入合格: {len(admitted)}"
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


if __name__ == "__main__":
    sys.exit(main())
