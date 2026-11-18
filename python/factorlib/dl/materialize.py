"""DL Alpha OOS 因子物化与统一注册模块 (dl/materialize.py)。

严格遵循架构设计 §5.4：
- 仅将严格滚动拼接出的纯样本外 OOS `(T, N)` 矩阵写入 `.npz`；
- 在 SQLite 注册为 `factor_type = "dl_alpha"`、`source = "dl_alpha"`、`status = "candidate"`；
- 自动调用核心单因子统一评价管线并锁定初始不可变版本快照（关联 `recipe_id` 与逐折诊断）。
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from factorlib.core._hooks import resolve_panel_market, resolve_profile
from factorlib.core.evaluate import _evaluate_factor_matrix_core
from factorlib.core.panel import Panel
from factorlib.core.registry import FactorMetricsRecord, FactorRegistry


def evaluate_and_materialize_dl_alpha(
    *,
    panel: Panel,
    registry: FactorRegistry,
    oos_matrix: np.ndarray,
    market: str | None = None,
    name: str,
    recipe_id: str,
    content_hash: str = "",
    run_id: str | None = None,
    horizon: int = 5,
    min_names: int = 5,
    fold_diagnostics: list[dict[str, Any]] | None = None,
    model_meta: dict[str, Any] | None = None,
    postprocess: list[dict[str, Any]] | None = None,
    status: str = "candidate",
    mat_dir: str | Path | None = None,
) -> tuple[int, Sequence[FactorMetricsRecord]]:
    """使用 Rust 内核评估纯 OOS `dl_alpha` 矩阵，并物化为 `.npz` + 注册 SQLite。"""
    norm_m = str(market or resolve_panel_market(panel)).strip().lower()
    profile = resolve_profile(norm_m)
    engine = panel.to_rust_engine()
    data_version = str(panel.meta.get("panel_version", "v1"))

    oos_f = np.asfortranarray(oos_matrix, dtype=np.float64)
    canon_tag = f"dl_alpha({recipe_id}:{name})"
    hash_tag = f"dl_{content_hash[:16]}" if content_hash else f"dl_{recipe_id}"

    # 切分出有有效预测的纯 OOS 区间，供全期与 OOS 指标对齐统计
    valid_rows = np.where(np.any(np.isfinite(oos_f) & panel.valid_mask, axis=1))[0]
    dates_str = np.array([str(d) for d in panel.dates])
    splits: dict[str, tuple[str, str]] | None = None
    if len(valid_rows) > 0:
        oos_s = dates_str[int(valid_rows[0])]
        oos_e = dates_str[int(valid_rows[-1])]
        splits = {"test": (str(oos_s), str(oos_e))}

    eval_res = _evaluate_factor_matrix_core(
        engine=engine,
        dates=dates_str,
        factor_mat=oos_f,
        canon_expr=canon_tag,
        canon_hash=hash_tag,
        horizons=(int(horizon),),
        splits=splits,
        min_names=min_names,
        force_sign=1,
    )

    # 注册使用 OOS 有效区间符号（不因历史空值而翻转）
    chosen_pp = postprocess if postprocess is not None else profile.default_postprocess()
    factor_id = registry.register_dl_alpha(
        name=name,
        market=norm_m,
        matrix=oos_f,
        dates=panel.dates,
        symbols=panel.symbols,
        recipe_id=recipe_id,
        run_id=run_id,
        data_version=data_version,
        sign=1,
        mode="cs",
        status=status,
        postprocess=chosen_pp,
        fold_diagnostics=fold_diagnostics or [],
        metrics=eval_res.all_metrics,
        model_meta=model_meta or {},
        mat_dir=mat_dir,
    )
    return factor_id, eval_res.all_metrics
