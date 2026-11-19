#!/usr/bin/env python3
"""FactorLibrary `1d` DL Alpha PyTorch 滚动训练与物化入口 (scripts/run_dl.py)。

支持：
- `--recipe <id>` 冻结实验配方驱动（Web 端 P4 一键后台派发或终端直接执行）；
- 双市场统一 `1d` 时序 Dataset 构建（Fold 内防泄漏预处理 + 标签去 Beta）；
- 四大模型族：`gated_mlp` / `gru` / `tcn` / `st_attention`；
- Purged Walk-Forward 多种子集成 + 嵌套 Lockbox 隔离 + 日内截面 IC Loss；
- OOS 矩阵 `.npz` 物化 + `dl_alpha` `candidate` 状态入库 + 可选三关正交准入预检。
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "python"))

from factorlib.core.admission_audit import audit_candidates
from factorlib.core.registry import FactorRegistry, ResearchRecipeRecord
from factorlib.core.runs import RunTracker
from factorlib.core.synthetic import (
    create_synthetic_futures_panel,
    create_synthetic_stocks_panel,
)
from factorlib.dl.materialize import evaluate_and_materialize_dl_alpha
from factorlib.dl.models import default_model_hparams, normalize_model_arch
from factorlib.dl.trainer import DLTrainConfig, train_dl_walk_forward
from factorlib.markets.profile import PanelSpec
from factorlib.markets.registry import get_profile
from factorlib.ml.dataset import parse_label_horizon


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="FactorLibrary 1d DL Alpha Purged Walk-Forward PyTorch 训练入口"
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
        help="目标市场域 (futures | stocks)",
    )
    parser.add_argument(
        "--panel",
        type=Path,
        default=None,
        help="派生研究面板目录路径",
    )
    parser.add_argument(
        "--db",
        type=Path,
        default=Path("data/registry/factors.sqlite"),
        help="SQLite 因子注册库路径",
    )
    parser.add_argument(
        "--runs-dir",
        type=Path,
        default=Path("runs"),
        help="实验运行轨迹输出根目录",
    )
    parser.add_argument(
        "--run-id",
        type=str,
        default=None,
        help="指定实验 run_id",
    )
    parser.add_argument(
        "--arch",
        type=str,
        default=None,
        help="模型骨架覆盖 (gated_mlp | gru | tcn | st_attention)",
    )
    parser.add_argument(
        "--target-label",
        type=str,
        default=None,
        help="覆盖预测目标标签 (如 y_5 或纯特质标签 y_idio_5)",
    )
    parser.add_argument(
        "--seq-len",
        type=int,
        default=None,
        help="回看时间步长 L (默认 20)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="每折每种子的最大训练轮数 (默认 18)",
    )
    parser.add_argument(
        "--n-folds",
        type=int,
        default=None,
        help="Walk-Forward 折数",
    )
    parser.add_argument(
        "--n-seeds",
        type=int,
        default=None,
        help="每折随机种子集成数 (3~5)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="计算设备 (auto | cpu | cuda)",
    )
    parser.add_argument(
        "--audit",
        action="store_true",
        help="训练入库后立即执行三关正交准入预检并写入 audit_report.json",
    )
    parser.add_argument(
        "--synthetic",
        action="store_true",
        help="使用内置合成演示面板运行",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="日志级别",
    )
    return parser.parse_args(argv)


def _opt_float(val: Any, default: float | None = None) -> float | None:
    if val is None or val == "":
        return default
    return float(val)


def _opt_int(val: Any, default: int | None = None) -> int | None:
    if val is None or val == "":
        return default
    return int(val)


def run_dl_cli(args: argparse.Namespace, root_dir: Path | None = None) -> int:
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
        if recipe.kind != "dl":
            logging.error("配方 %s 类型为 %r，无法用于 DL 训练任务", recipe.recipe_id, recipe.kind)
            return 1
        market = recipe.market
    else:
        market = args.market or "futures"

    profile = get_profile(market)
    if hasattr(profile, "root"):
        profile.root = root  # type: ignore[attr-defined]

    extensions = list(recipe.extensions) if recipe is not None else []
    if args.synthetic:
        panel = (
            create_synthetic_stocks_panel()
            if market == "stocks"
            else create_synthetic_futures_panel()
        )
        if extensions:
            profile.attach_extensions(panel, extensions)
    else:
        panel_override = recipe.params.get("panel_dir") if recipe else None
        default_panel_path = (
            Path(panel_override)
            if panel_override and Path(panel_override).is_absolute()
            else (root / panel_override if panel_override else profile.paths().panel_dir)
        )
        panel_path = (
            (args.panel if args.panel.is_absolute() else (root / args.panel))
            if args.panel is not None
            else default_panel_path
        )
        if not panel_path.exists():
            logging.warning("研究面板目录 %s 不存在，回退使用内置合成演示面板", panel_path)
            panel = (
                create_synthetic_stocks_panel()
                if market == "stocks"
                else create_synthetic_futures_panel()
            )
        else:
            panel = profile.load_panel(PanelSpec(panel_dir=panel_path, extensions=extensions))

    data_version = str(panel.meta.get("panel_version", recipe.data_version if recipe else "v1"))
    params = dict(recipe.params) if recipe is not None else {}
    target_label = (
        args.target_label
        if getattr(args, "target_label", None)
        else (recipe.target_label if recipe is not None else "y_5")
    )
    if target_label not in panel.labels:
        target_label = "y_5" if "y_5" in panel.labels else next(iter(panel.labels.keys()))

    horizon = parse_label_horizon(target_label, default=5)
    arch_key = normalize_model_arch(args.arch or params.get("model"), market=market)
    arch_defaults = default_model_hparams(market=market, arch=arch_key)

    seq_len = max(2, int(args.seq_len or params.get("seq_len") or 20))
    if panel.T < 50:
        seq_len = min(seq_len, max(4, panel.T // 4))

    n_folds = max(2, min(8, int(args.n_folds or params.get("n_folds") or 4)))
    n_seeds = max(1, min(5, int(args.n_seeds or params.get("n_seeds") or 3)))
    epochs = max(1, int(args.epochs or params.get("epochs") or 18))
    purge_window = max(horizon, int(params.get("purge_window", horizon)))

    dl_cfg = DLTrainConfig(
        market=market,
        arch=arch_key,
        target_label=target_label,
        seq_len=seq_len,
        hidden_dim=_opt_int(params.get("hidden_dim"), arch_defaults["hidden_dim"]),
        n_layers=_opt_int(params.get("n_layers"), arch_defaults["n_layers"]),
        dropout=_opt_float(params.get("dropout"), arch_defaults["dropout"]),
        lr=_opt_float(params.get("lr"), arch_defaults["lr"]),
        weight_decay=_opt_float(params.get("weight_decay"), arch_defaults["weight_decay"]),
        epochs=epochs,
        patience=max(2, int(params.get("patience", 6))),
        batch_days=_opt_int(params.get("batch_days"), arch_defaults["batch_days"]),
        n_folds=n_folds,
        n_seeds=n_seeds,
        purge_window=purge_window,
        embargo_window=max(0, int(params.get("embargo_window", 0))),
        train_window=str(params.get("train_window", "expanding")),
        base_seed=int(recipe.seed) if recipe is not None else 42,
        feature_norm=str(params.get("feature_norm", "rank_gauss")),
        label_neutralize=str(params.get("label_neutralize", "demean")),
        causal_ts_window=_opt_int(params.get("causal_ts_window")),
        turnover_weight=float(params.get("turnover_penalty", 0.05)),
        lockbox_start_date=params.get("lockbox_start_date"),
        lockbox_ratio=float(params.get("lockbox_ratio", 0.0)),
        device=str(params.get("device") or args.device or "auto"),
        min_names=max(3, min(5, panel.N // 2)),
    )

    tracker = RunTracker(
        kind="dl",
        run_id=args.run_id,
        runs_root=runs_dir,
        config_yaml=json.dumps({**params, "resolved_arch": arch_key}, ensure_ascii=False, indent=2),
        data_version=data_version,
        registry=registry,
        market=market,
        recipe_id=recipe.recipe_id if recipe is not None else None,
    )

    t0 = time.perf_counter()
    log_lines: list[str] = [
        f"[{tracker.run_id}] PyTorch 1d DL Alpha 启动 | market={market} arch={arch_key} "
        f"seq_len={dl_cfg.seq_len} epochs={dl_cfg.epochs} folds={dl_cfg.n_folds} "
        f"seeds={dl_cfg.n_seeds} purge={dl_cfg.purge_window} neutralize={dl_cfg.label_neutralize}"
    ]

    def _on_progress(prog: dict[str, Any], msg: str) -> None:
        line = f"[{tracker.run_id}] {msg}"
        log_lines.append(line)
        logging.info(msg)
        tracker.update_heartbeat(progress=prog, log_tail="\n".join(log_lines[-30:]))

    try:
        fs_fields = list((recipe.feature_set or {}).get("base_fields", [])) if recipe else []
        ext_fields = list((recipe.feature_set or {}).get("ext_fields", [])) if recipe else []
        req_fields = [f for f in [*fs_fields, *ext_fields] if f in panel.fields]
        if not req_fields:
            req_fields = [
                f
                for f in (
                    "close",
                    "close_hfq",
                    "volume",
                    "log_amount",
                    "c2c_ret",
                    "ry_ann",
                    "wh_z60",
                    "mr_net_ratio",
                    "turnover_rate",
                )
                if f in panel.fields
            ]

        train_res = train_dl_walk_forward(
            panel,
            feature_names=req_fields,
            config=dl_cfg,
            progress_callback=_on_progress,
        )

        # 生成唯一候选名称（体现市场、模型架构与配方尾缀）
        rid_tag = recipe.recipe_id if recipe is not None else tracker.run_id
        cand_name = f"dl_{arch_key}_{rid_tag[-6:]}"
        mat_dir = db_path.parent / "materialized"
        fid, eval_metrics = evaluate_and_materialize_dl_alpha(
            panel=panel,
            registry=registry,
            oos_matrix=train_res.oos_matrix,
            market=market,
            name=cand_name,
            recipe_id=rid_tag,
            content_hash=recipe.content_hash if recipe is not None else tracker.run_id,
            run_id=tracker.run_id,
            horizon=horizon,
            min_names=max(3, min(5, panel.N // 2)),
            fold_diagnostics=train_res.fold_diagnostics,
            model_meta=train_res.model_meta,
            postprocess=profile.default_postprocess(),
            status="candidate",
            mat_dir=mat_dir,
        )

        audit_reports: list[dict[str, Any]] = []
        if args.audit or params.get("auto_audit"):
            audit_reports = audit_candidates(
                panel,
                registry,
                [fid],
                market=market,
                horizon=horizon,
                min_names=max(3, min(5, panel.N // 2)),
            )
            tracker.save_json("audit_report.json", audit_reports)
            if audit_reports:
                ar = audit_reports[0]
                log_lines.append(
                    f"[{tracker.run_id}] 三关准入预检: Gate1={ar['gate1_pass']} "
                    f"Gate2={ar['gate2_pass']} (resid_ICIR={ar.get('residual_icir', 0):+.3f}, "
                    f"t={ar.get('residual_t_stat', 0):+.2f}) "
                    f"Gate3={ar['gate3_pass']} (ΔSharpe={ar.get('delta_sharpe', 0):+.3f})"
                )

        elapsed_sec = time.perf_counter() - t0
        primary_icir = eval_metrics[0].ic_ir if eval_metrics else train_res.selection_icir
        log_lines.append(
            f"[{tracker.run_id}] 完成 OOS 物化入库: factor_id={fid} ({cand_name}), "
            f"OOS RankICIR={primary_icir:+.3f}, status=candidate, elapsed={elapsed_sec:.2f}s"
        )
        tracker.save_json("fold_diagnostics.json", train_res.fold_diagnostics)
        tracker.save_json("model_meta.json", train_res.model_meta)
        tracker.save_text("run_summary.md", "\n".join(log_lines) + "\n")
        tracker.update_heartbeat(
            progress={
                "stage": "完成 (dl_alpha candidate 已入库)",
                "pct": 1.0,
                "valid_icir": round(float(train_res.selection_icir), 4),
                "oos_icir": round(float(primary_icir), 4),
                "lockbox_icir": round(float(train_res.lockbox_icir), 4)
                if train_res.lockbox_icir is not None
                else None,
                "arch": arch_key,
                "factor_id": fid,
            },
            log_tail="\n".join(log_lines[-30:]),
        )
        tracker.finish(status="completed", n_evaluated=1)
        print(
            f"DL Alpha PyTorch complete: run_id={tracker.run_id}, factor_id={fid} ({cand_name}), "
            f"arch={arch_key}, selection_icir={train_res.selection_icir:+.4f}"
        )
        return 0
    except Exception as exc:
        logging.exception("DL Alpha 运行失败: %s", exc)
        log_lines.append(f"[{tracker.run_id}] ERROR: {exc}")
        tracker.update_heartbeat(log_tail="\n".join(log_lines[-30:]))
        tracker.finish(status="failed")
        return 1


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="[%(asctime)s] [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )
    return run_dl_cli(args)


if __name__ == "__main__":
    sys.exit(main())
