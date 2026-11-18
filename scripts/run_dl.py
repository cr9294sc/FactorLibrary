#!/usr/bin/env python3
"""FactorLibrary DL Alpha 滚动训练入口 (scripts/run_dl.py)。

Phase 2 阶段实现 `--recipe <id>` 驱动的 Purged Walk-Forward 因果线性/岭回归基线占位训练器，
严格执行 Fold 内防泄漏标准化、Purge 隔离、多种子集成、OOS 矩阵拼接物化 (`.npz`) 与
`candidate` 状态入库，并持续向 `RunTracker` 汇报 Fold/Seed 实时心跳与日志。
（Phase 3 将在此统一入口与契约下接入 PyTorch GRU / TCN / 时空解耦 Attention 网络。）
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "python"))

from factorlib.core.evaluate import _evaluate_factor_matrix_core
from factorlib.core.registry import FactorRegistry, ResearchRecipeRecord
from factorlib.core.runs import RunTracker
from factorlib.core.styles import _spearman_corr_1d
from factorlib.core.synthetic import (
    create_synthetic_futures_panel,
    create_synthetic_stocks_panel,
)
from factorlib.markets.profile import PanelSpec
from factorlib.markets.registry import get_profile


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="FactorLibrary DL Alpha Purged Walk-Forward 训练入口 (Phase 2 基线 / Phase 3 PyTorch)"
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


def _cs_zscore_matrix(mat: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """每日截面标准化（仅用当日可用截面样本，绝无跨期前视）。"""
    T, N = mat.shape
    out = np.full((T, N), np.nan, dtype=np.float64, order="F")
    for t in range(T):
        vm = mask[t, :] & np.isfinite(mat[t, :])
        idx = np.where(vm)[0]
        if len(idx) < 3:
            continue
        vals = mat[t, idx]
        mu = float(np.mean(vals))
        sd = float(np.std(vals))
        if sd > 1e-12:
            out[t, idx] = np.clip((vals - mu) / sd, -4.0, 4.0)
        else:
            out[t, idx] = 0.0
    return out


def _daily_rankicir(pred: np.ndarray, label: np.ndarray, mask: np.ndarray, rows: range) -> float:
    ics: list[float] = []
    for t in rows:
        vm = mask[t, :] & np.isfinite(pred[t, :]) & np.isfinite(label[t, :])
        idx = np.where(vm)[0]
        if len(idx) < 5:
            continue
        rho = _spearman_corr_1d(pred[t, idx], label[t, idx])
        if math.isfinite(rho):
            ics.append(float(rho))
    if len(ics) < 2:
        return 0.0
    arr = np.asarray(ics, dtype=np.float64)
    sd = float(np.std(arr, ddof=1))
    return float(np.mean(arr) / sd) if sd > 1e-12 else 0.0


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
    target_label = recipe.target_label if recipe is not None else "y_5"
    if target_label not in panel.labels:
        target_label = "y_5" if "y_5" in panel.labels else next(iter(panel.labels.keys()))

    horizon = (
        int(target_label[2:]) if target_label.startswith("y_") and target_label[2:].isdigit() else 5
    )
    purge_window = max(horizon, int(params.get("purge_window", horizon)))
    n_seeds = max(1, min(5, int(params.get("n_seeds", 3))))
    n_folds = max(2, min(6, int(params.get("n_folds", 4))))
    model_arch = str(params.get("model", "Gated-MLP (Phase2 Baseline)"))

    tracker = RunTracker(
        kind="dl",
        run_id=args.run_id,
        runs_root=runs_dir,
        config_yaml=json.dumps(params, ensure_ascii=False, indent=2),
        data_version=data_version,
        registry=registry,
        market=market,
        recipe_id=recipe.recipe_id if recipe is not None else None,
    )

    t0 = time.perf_counter()
    log_lines: list[str] = [
        f"[{tracker.run_id}] [Phase 2 Baseline Preview] Phase 3 will plug in PyTorch GRU/TCN/Attention",
        f"[{tracker.run_id}] market={market}, model={model_arch}, target={target_label}, purge={purge_window}, seeds={n_seeds}, folds={n_folds}",
    ]

    try:
        fs_fields = list((recipe.feature_set or {}).get("base_fields", [])) if recipe else []
        usable_fields = [f for f in fs_fields if f in panel.fields]
        if not usable_fields:
            usable_fields = [
                f
                for f in (
                    "close",
                    "volume",
                    "ry_ann",
                    "wh_z60",
                    "mr_net_ratio",
                    "turnover_rate",
                    "mkt_cap",
                )
                if f in panel.fields
            ]
        if not usable_fields:
            usable_fields = list(panel.fields.keys())[:4]

        # 构造标准化特征列表 (T, N, K)
        norm_feats = [_cs_zscore_matrix(panel.fields[f], panel.valid_mask) for f in usable_fields]
        X_cube = np.stack(norm_feats, axis=-1)  # (T, N, K)
        y_mat = np.asarray(panel.labels[target_label], dtype=np.float64)
        T, N, K = X_cube.shape

        # 构建严格 Purged Walk-Forward 折切分
        min_train = max(25, T // 3)
        fold_step = max(8, (T - min_train) // n_folds)

        oos_pred = np.full((T, N), np.nan, dtype=np.float64, order="F")
        fold_diags: list[dict[str, Any]] = []
        total_steps = n_folds * n_seeds
        done_steps = 0

        for fold_idx in range(n_folds):
            test_start = min_train + fold_idx * fold_step
            test_end = min(T, test_start + fold_step) if fold_idx < n_folds - 1 else T
            if test_start >= T - 2:
                break

            val_len = max(8, min(20, (test_start - purge_window) // 4))
            val_end = max(1, test_start - purge_window)
            val_start = max(0, val_end - val_len)
            train_end = max(5, val_start - purge_window)
            if train_end < 10:
                continue

            # 组装训练集特征-标签（剔除 NaN）
            tr_X = X_cube[:train_end].reshape(-1, K)
            tr_y = y_mat[:train_end].reshape(-1)
            tr_m = (
                panel.valid_mask[:train_end].reshape(-1)
                & np.all(np.isfinite(tr_X), axis=1)
                & np.isfinite(tr_y)
            )
            tr_X_clean = np.nan_to_num(tr_X[tr_m], nan=0.0)
            tr_y_clean = tr_y[tr_m]
            if len(tr_y_clean) < max(10, K + 2):
                continue

            seed_preds: list[np.ndarray] = []
            base_seed = int(recipe.seed) if recipe else 42
            best_val_icir = 0.0

            for s_idx in range(n_seeds):
                rng = np.random.default_rng(base_seed + fold_idx * 100 + s_idx)
                # 带轻微随机扰动正则的因果 Ridge 模拟多种子集成
                ridge_alpha = float(params.get("weight_decay", 1e-3)) * 1e4 + rng.uniform(1.0, 8.0)
                xtx = tr_X_clean.T @ tr_X_clean + ridge_alpha * np.eye(K)
                xty = tr_X_clean.T @ tr_y_clean
                w = np.linalg.solve(xtx, xty)
                # 预测 valid + test 窗并做每日截面 Z-Score
                raw_val = np.tensordot(
                    np.nan_to_num(X_cube[val_start:val_end], nan=0.0), w, axes=([2], [0])
                )
                val_icir = _daily_rankicir(
                    raw_val,
                    y_mat[val_start:val_end],
                    panel.valid_mask[val_start:val_end],
                    range(raw_val.shape[0]),
                )
                best_val_icir = max(best_val_icir, val_icir)

                raw_test = np.tensordot(
                    np.nan_to_num(X_cube[test_start:test_end], nan=0.0), w, axes=([2], [0])
                )
                norm_test = _cs_zscore_matrix(raw_test, panel.valid_mask[test_start:test_end])
                seed_preds.append(norm_test)

                done_steps += 1
                msg = (
                    f"Fold {fold_idx + 1}/{n_folds} · Seed {s_idx + 1}/{n_seeds} "
                    f"train=[0:{train_end}] valid=[{val_start}:{val_end}] test=[{test_start}:{test_end}] "
                    f"Valid RankICIR={val_icir:+.3f}"
                )
                log_lines.append(f"[{tracker.run_id}] {msg}")
                tracker.update_heartbeat(
                    progress={
                        "stage": f"Fold {fold_idx + 1}/{n_folds} · Seed {s_idx + 1}/{n_seeds}",
                        "pct": round(done_steps / max(1, total_steps), 4),
                        "valid_icir": round(best_val_icir, 4),
                        "fold": fold_idx + 1,
                        "n_folds": n_folds,
                        "seed": s_idx + 1,
                        "n_seeds": n_seeds,
                    },
                    log_tail="\n".join(log_lines[-30:]),
                )
                time.sleep(0.05)

            if seed_preds:
                fused = np.nanmean(np.stack(seed_preds, axis=0), axis=0)
                oos_pred[test_start:test_end] = fused
                fold_diags.append(
                    {
                        "fold": fold_idx + 1,
                        "train_range": [str(panel.dates[0]), str(panel.dates[train_end - 1])],
                        "valid_range": [str(panel.dates[val_start]), str(panel.dates[val_end - 1])],
                        "test_range": [
                            str(panel.dates[test_start]),
                            str(panel.dates[test_end - 1]),
                        ],
                        "valid_icir": best_val_icir,
                    }
                )

        # 统一评价并物化注册为 candidate dl_alpha 因子
        engine = panel.to_rust_engine()
        canon_tag = f"dl_alpha_{recipe.recipe_id if recipe else tracker.run_id}"
        eval_res = _evaluate_factor_matrix_core(
            engine=engine,
            dates=np.array([str(d) for d in panel.dates]),
            factor_mat=oos_pred,
            canon_expr=canon_tag,
            canon_hash=f"dl_{recipe.content_hash[:16]}"
            if recipe and recipe.content_hash
            else tracker.run_id,
            horizons=(horizon,),
            min_names=5,
        )

        cand_name = f"dl_alpha_{(recipe.recipe_id or tracker.run_id)[-6:]}"
        fid = registry.register_dl_alpha(
            name=cand_name,
            market=market,
            matrix=oos_pred,
            dates=panel.dates,
            symbols=panel.symbols,
            recipe_id=recipe.recipe_id if recipe else None,
            run_id=tracker.run_id,
            data_version=data_version,
            postprocess=profile.default_postprocess(),
            fold_diagnostics=fold_diags,
            metrics=eval_res.all_metrics,
            status="candidate",
        )
        elapsed_sec = time.perf_counter() - t0
        log_lines.append(
            f"[{tracker.run_id}] 完成 OOS 拼接与物化入库: factor_id={fid} ({cand_name}), status=candidate, elapsed={elapsed_sec:.2f}s"
        )
        tracker.save_json("fold_diagnostics.json", fold_diags)
        tracker.save_text("run_summary.md", "\n".join(log_lines) + "\n")
        tracker.update_heartbeat(
            progress={
                "stage": "完成 (candidate 已入库)",
                "pct": 1.0,
                "valid_icir": round(fold_diags[-1]["valid_icir"], 4) if fold_diags else 0.0,
                "factor_id": fid,
            },
            log_tail="\n".join(log_lines[-30:]),
        )
        tracker.finish(status="completed", n_evaluated=1)
        print(f"DL Alpha baseline complete: run_id={tracker.run_id}, factor_id={fid} ({cand_name})")
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
