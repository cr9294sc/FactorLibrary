"""候选因子三关正交增量准入审计与多因子 A/B 对比引擎 (core/admission_audit.py)。

严格遵循架构设计 §5.4 与 M7/M8 规范：
- 关 1 (基础门槛 + 两两去重): 校验 OOS |ICIR|、年度方向一致性、截面换手率，以及与现存 accepted/active 库（含公式、GP、DL）的两两截面 Rank 相关系数 max|ρ| < 0.60（调用 Rust `factor_correlation_matrix` / `cross_correlation`）。
- 关 2 (Spanning Test 多因子正交残差检验): 每日横截面对已有 active 因子库做多元 OLS 回归，校验正交残差序列的 Residual RankICIR 与 t-stat。
- 关 3 (二层合成消融校验): 对比 `Base Active Pool` 与 `Base Active Pool + Candidate` 在因果 Ridge 滚动合成下的 OOS 夏普与回撤净改善 ΔSharpe。
- 多因子 A/B 对比 (`compare_factors`): 强制对齐 2~6 个因子的严格交集 OOS 日期区间，计算同台指标、两两截面 Rank 相关性热力图与目标因子对基准组的正交残差累积 IC 曲线。
"""

from __future__ import annotations

import json
import math
from collections.abc import Sequence
from typing import Any

import numpy as np

from factorlib.core._hooks import resolve_panel_market, resolve_profile
from factorlib.core.evaluate import _evaluate_factor_matrix_core
from factorlib.core.panel import Panel
from factorlib.core.probe import FactorDefinition, evaluate_factor_definition
from factorlib.core.registry import FactorRecord, FactorRegistry
from factorlib.core.styles import _spearman_corr_1d


def _factor_record_to_definition(rec: FactorRecord, profile: Any | None = None) -> FactorDefinition:
    """将 SQLite `FactorRecord` 还原为可求值与统一后处理的 `FactorDefinition`。"""
    if rec.definition_json:
        try:
            raw = json.loads(rec.definition_json)
            if isinstance(raw, dict):
                defn = FactorDefinition.from_dict(raw)
                if not defn.materialization_path and rec.materialization_path:
                    defn.materialization_path = rec.materialization_path
                if not defn.recipe_id and rec.recipe_id:
                    defn.recipe_id = rec.recipe_id
                if not defn.postprocess and profile is not None:
                    defn = FactorDefinition(
                        name=defn.name,
                        factor_type=defn.factor_type,
                        expression=defn.expression,
                        code=defn.code,
                        postprocess=defn.postprocess,
                        description=defn.description,
                        market=defn.market or rec.market,
                        recipe_id=defn.recipe_id or rec.recipe_id,
                        materialization_path=defn.materialization_path or rec.materialization_path,
                    )
                return defn
        except Exception:
            pass

    default_pp = profile.default_postprocess() if profile is not None else "standard"
    return FactorDefinition(
        name=rec.name or f"factor_{rec.id or 0}",
        factor_type=rec.factor_type or "expression",
        expression=rec.canon_expr if rec.factor_type == "expression" else "",
        postprocess=default_pp,
        market=rec.market,
        recipe_id=rec.recipe_id,
        materialization_path=rec.materialization_path,
    )


def _evaluate_record_matrix(
    rec: FactorRecord,
    panel: Panel,
    engine: Any,
    profile: Any,
) -> tuple[np.ndarray, int]:
    """求值单条因子记录并返回 `(directed_matrix, sign)`。"""
    defn = _factor_record_to_definition(rec, profile=profile)
    mat, _ = evaluate_factor_definition(defn, panel, engine=engine, profile=profile)
    sign = 1 if int(rec.sign or 1) >= 0 else -1
    directed = np.asfortranarray(np.asarray(mat, dtype=np.float64) * float(sign), dtype=np.float64)
    return directed, sign


def _resolve_target_label(panel: Panel, horizon: int) -> tuple[str, np.ndarray]:
    """选择目标收益标签矩阵。"""
    lbl_name = f"y_{int(horizon)}"
    if lbl_name in panel.labels:
        return lbl_name, np.asarray(panel.labels[lbl_name], dtype=np.float64)
    for fallback in ("y_5", "y_1", "r_exec_next"):
        if fallback in panel.labels:
            return fallback, np.asarray(panel.labels[fallback], dtype=np.float64)
    first_k = next(iter(panel.labels.keys()))
    return first_k, np.asarray(panel.labels[first_k], dtype=np.float64)


def _daily_cs_ols_residual_ic(
    target_mat: np.ndarray,
    bench_mats: Sequence[np.ndarray],
    label_mat: np.ndarray,
    mask: np.ndarray,
    row_indices: np.ndarray,
    min_names: int = 5,
) -> dict[str, Any]:
    """对每日截面执行 `target ~ 1 + X_bench` OLS 回归，计算正交残差 RankIC 序列与 t 统计量。"""
    ic_vals: list[float] = []
    dates_idx: list[int] = []

    for t in row_indices:
        y_row = target_mat[t, :]
        r_row = label_mat[t, :]
        valid = mask[t, :] & np.isfinite(y_row) & np.isfinite(r_row)
        for bm in bench_mats:
            valid = valid & np.isfinite(bm[t, :])
        idx = np.where(valid)[0]
        m = len(idx)
        if m < max(min_names, len(bench_mats) + 2):
            continue

        y_vec = y_row[idx]
        if bench_mats:
            x_cols = [np.ones(m, dtype=np.float64)]
            for bm in bench_mats:
                col = bm[t, idx]
                std_c = float(np.std(col))
                if std_c > 1e-12:
                    x_cols.append((col - float(np.mean(col))) / std_c)
            if len(x_cols) > 1:
                X = np.column_stack(x_cols)
                beta, _, _, _ = np.linalg.lstsq(X, y_vec, rcond=None)
                resid = y_vec - X @ beta
            else:
                resid = y_vec - float(np.mean(y_vec))
        else:
            resid = y_vec

        if float(np.std(resid)) <= 1e-12:
            continue
        rho = _spearman_corr_1d(resid, r_row[idx])
        if math.isfinite(rho):
            ic_vals.append(float(rho))
            dates_idx.append(int(t))

    arr = np.asarray(ic_vals, dtype=np.float64)
    n_days = len(arr)
    if n_days == 0:
        return {
            "n_days": 0,
            "mean": 0.0,
            "std": 0.0,
            "icir": 0.0,
            "t_stat": 0.0,
            "daily_ic": [],
            "cum_ic": [],
            "row_indices": [],
        }
    mean_v = float(np.mean(arr))
    std_v = float(np.std(arr, ddof=1)) if n_days > 1 else 0.0
    icir = mean_v / std_v if std_v > 1e-12 else 0.0
    t_stat = (mean_v / std_v) * math.sqrt(n_days) if std_v > 1e-12 and n_days > 1 else 0.0
    cum_ic = np.cumsum(arr).tolist()
    return {
        "n_days": n_days,
        "mean": mean_v,
        "std": std_v,
        "icir": icir,
        "t_stat": t_stat,
        "daily_ic": arr.tolist(),
        "cum_ic": [float(x) for x in cum_ic],
        "row_indices": dates_idx,
    }


def _quick_ridge_oos_sharpe(
    feature_mats: Sequence[np.ndarray],
    ret_1d: np.ndarray,
    mask: np.ndarray,
    row_indices: np.ndarray,
    min_names: int = 5,
    alpha: float = 10.0,
) -> dict[str, float]:
    """轻量级因果扩展窗 Ridge 多因子合成并计算净多空夏普（供关 3 消融校验使用）。"""
    if not feature_mats or len(row_indices) < 10:
        return {"sharpe": 0.0, "ann_ret": 0.0, "max_dd": 0.0}

    T_sub = len(row_indices)
    K = len(feature_mats)
    warmup = max(5, min(20, T_sub // 3))

    # 预计算每日截面的 Ridge 充分统计量 (X^T X, X^T y) 避免重复全量拼接
    xtx_cum = np.zeros((K, K), dtype=np.float64)
    xty_cum = np.zeros(K, dtype=np.float64)
    port_rets: list[float] = []

    for step_idx, t in enumerate(row_indices):
        r_row = ret_1d[t, :]
        valid = mask[t, :] & np.isfinite(r_row)
        cols = []
        for fm in feature_mats:
            col = fm[t, :]
            valid = valid & np.isfinite(col)
            cols.append(col)
        idx = np.where(valid)[0]
        if len(idx) < min_names:
            continue

        X_t = np.column_stack([c[idx] for c in cols])
        # 当日截面标准化特征与去均值收益
        mu_x = np.mean(X_t, axis=0, keepdims=True)
        sd_x = np.std(X_t, axis=0, keepdims=True)
        sd_x = np.where(sd_x > 1e-12, sd_x, 1.0)
        X_norm = (X_t - mu_x) / sd_x
        y_t = r_row[idx]
        y_centered = y_t - float(np.mean(y_t))

        if step_idx >= warmup:
            if K == 1:
                score = X_norm[:, 0]
            else:
                mat_a = xtx_cum + alpha * np.eye(K, dtype=np.float64)
                try:
                    w = np.linalg.solve(mat_a, xty_cum)
                except np.linalg.LinAlgError:
                    w = np.ones(K, dtype=np.float64) / K
                if float(np.sum(np.abs(w))) < 1e-12:
                    w = np.ones(K, dtype=np.float64) / K
                score = X_norm @ w

            # 截面美元中性多空代理收益（按排序去均值权重）
            s_centered = score - float(np.mean(score))
            denom = float(np.sum(np.abs(s_centered)))
            if denom > 1e-12:
                weights = s_centered / denom
                port_rets.append(float(np.dot(weights, y_t)))

        # 在 T 日收益观测完成后（严格因果：仅用于 T+1 之后的预测）更新历史充分统计量
        xtx_cum += X_norm.T @ X_norm
        xty_cum += X_norm.T @ y_centered

    if len(port_rets) < 3:
        return {"sharpe": 0.0, "ann_ret": 0.0, "max_dd": 0.0}

    r_arr = np.asarray(port_rets, dtype=np.float64)
    ann_ret = float(np.mean(r_arr) * 250.0)
    ann_vol = float(np.std(r_arr, ddof=1) * math.sqrt(250.0))
    sharpe = ann_ret / ann_vol if ann_vol > 1e-12 else 0.0
    nav = np.cumprod(1.0 + r_arr)
    peaks = np.maximum.accumulate(nav)
    max_dd = float(np.min((nav - peaks) / np.maximum(peaks, 1e-12)))
    return {"sharpe": sharpe, "ann_ret": ann_ret, "max_dd": max_dd}


def audit_candidates(
    panel: Panel,
    registry: FactorRegistry,
    factor_ids: Sequence[int],
    market: str = "futures",
    *,
    horizon: int = 5,
    min_valid_icir: float = 0.25,
    max_turnover: float = 0.85,
    max_corr_limit: float = 0.60,
    min_residual_icir: float = 0.15,
    min_residual_t: float = 1.5,
    min_names: int = 5,
) -> list[dict[str, Any]]:
    """对候选因子列表执行三关递进式准入审计，并返回每条因子的审计报告。"""
    norm_m = str(market).strip().lower() or resolve_panel_market(panel)
    profile = resolve_profile(norm_m)
    engine = panel.to_rust_engine()
    T, _ = panel.shape
    dates = [str(d) for d in panel.dates]

    lbl_name, label_mat = _resolve_target_label(panel, horizon=horizon)
    ret_1d_name, ret_1d = _resolve_target_label(panel, horizon=1)
    del lbl_name, ret_1d_name

    # 使用 OOS / 后半程窗口作为默认检验窗口（若区间较短则覆盖全样本后 50% 或全部）
    oos_start = max(0, T // 3) if T >= 60 else 0
    oos_rows = np.arange(oos_start, T, dtype=np.int64)

    # 加载现存 accepted / active 基线库矩阵（排除正在审计的候选 ID 本身）
    cand_id_set = {int(fid) for fid in factor_ids}
    lib_records = [
        r
        for r in (
            registry.list_factors(status="active", market=norm_m)
            + registry.list_factors(status="accepted", market=norm_m)
        )
        if r.id not in cand_id_set
    ]
    lib_mats: list[np.ndarray] = []
    lib_names: list[str] = []
    for rec in lib_records:
        try:
            mat, _ = _evaluate_record_matrix(rec, panel, engine, profile)
            lib_mats.append(mat)
            lib_names.append(rec.name or rec.canon_expr[:24])
        except Exception:
            continue

    # 预先计算 Base Active Pool 消融基准夏普
    base_ablation = _quick_ridge_oos_sharpe(
        lib_mats, ret_1d=ret_1d, mask=panel.valid_mask, row_indices=oos_rows, min_names=min_names
    )

    results: list[dict[str, Any]] = []
    for fid in factor_ids:
        rec = registry.get_factor(int(fid), include_metrics=True)
        if rec is None or rec.market != norm_m:
            results.append(
                {
                    "factor_id": int(fid),
                    "name": f"factor_{fid}",
                    "error": "因子不存在或不属于当前市场",
                    "gate1_pass": False,
                    "gate2_pass": False,
                    "gate3_pass": False,
                    "all_passed": False,
                }
            )
            continue

        fname = rec.name or f"factor_{rec.id}"
        try:
            cand_mat, _ = _evaluate_record_matrix(rec, panel, engine, profile)
        except Exception as exc:
            results.append(
                {
                    "factor_id": int(fid),
                    "name": fname,
                    "error": str(exc),
                    "gate1_pass": False,
                    "gate2_pass": False,
                    "gate3_pass": False,
                    "all_passed": False,
                }
            )
            continue

        # ------------------------------------------------------------------
        # 关 1: 单因子 OOS |ICIR| + 分年方向一致性 + 换手率 + 两两去重 max|ρ| < 0.60
        # ------------------------------------------------------------------
        oos_sub = np.asfortranarray(cand_mat[oos_start:T])
        rank_ic_res = engine.rank_ic(
            oos_sub,
            label_name=f"y_{horizon}"
            if f"y_{horizon}" in engine.label_names
            else engine.label_names[0],
            min_names=min_names,
            rank=True,
            horizon=horizon,
            start_row=oos_start,
            end_row=T,
        )
        oos_icir = float(rank_ic_res.get("ic_ir", 0.0))
        oos_ic_mean = float(rank_ic_res.get("ic_mean", 0.0))
        to_res = engine.factor_turnover(
            oos_sub, min_names=min_names, start_row=oos_start, end_row=T
        )
        turnover = float(to_res.get("turnover", 0.0))
        if not math.isfinite(turnover):
            turnover = 0.0

        # 分年度方向一致性
        years = sorted({dates[i][:4] for i in oos_rows if len(dates[i]) >= 4})
        same_sign_years = 0
        valid_years = 0
        base_sign = 1 if oos_ic_mean >= 0.0 else -1
        for yr in years:
            y_idx = [i for i in oos_rows if dates[i].startswith(yr)]
            if len(y_idx) < 5:
                continue
            y_ic = []
            for t in y_idx:
                v_m = (
                    panel.valid_mask[t, :]
                    & np.isfinite(cand_mat[t, :])
                    & np.isfinite(label_mat[t, :])
                )
                ix = np.where(v_m)[0]
                if len(ix) >= min_names:
                    rho = _spearman_corr_1d(cand_mat[t, ix], label_mat[t, ix])
                    if math.isfinite(rho):
                        y_ic.append(rho)
            if y_ic:
                valid_years += 1
                if (1 if float(np.mean(y_ic)) >= 0.0 else -1) == base_sign:
                    same_sign_years += 1
        dir_consistency = float(same_sign_years / valid_years) if valid_years > 0 else 1.0

        max_corr = 0.0
        most_corr_with = None
        for lm, lname in zip(lib_mats, lib_names, strict=True):
            rho = float(
                engine.cross_correlation(
                    cand_mat,
                    lm,
                    min_names=min_names,
                    rank=True,
                    start_row=oos_start,
                    end_row=T,
                )
            )
            if math.isfinite(rho) and abs(rho) > max_corr:
                max_corr = abs(rho)
                most_corr_with = lname

        gate1_pass = (
            abs(oos_icir) >= min_valid_icir
            and turnover <= max_turnover
            and max_corr < max_corr_limit
            and (valid_years < 2 or dir_consistency >= 0.5)
        )

        # ------------------------------------------------------------------
        # 关 2: Spanning Test (对已有库因子正交残差的 RankICIR 与 t-stat)
        # ------------------------------------------------------------------
        span = _daily_cs_ols_residual_ic(
            cand_mat,
            lib_mats,
            label_mat=label_mat,
            mask=panel.valid_mask,
            row_indices=oos_rows,
            min_names=min_names,
        )
        resid_icir = float(span["icir"])
        resid_t = float(span["t_stat"])
        gate2_pass = abs(resid_icir) >= min_residual_icir and abs(resid_t) >= min_residual_t

        # ------------------------------------------------------------------
        # 关 3: 二层合成消融校验 (Base vs Base + Candidate)
        # ------------------------------------------------------------------
        with_ablation = _quick_ridge_oos_sharpe(
            [*lib_mats, cand_mat],
            ret_1d=ret_1d,
            mask=panel.valid_mask,
            row_indices=oos_rows,
            min_names=min_names,
        )
        delta_sharpe = float(with_ablation["sharpe"] - base_ablation["sharpe"])
        delta_ann_ret = float(with_ablation["ann_ret"] - base_ablation["ann_ret"])
        gate3_pass = delta_sharpe > 0.0

        all_passed = bool(gate1_pass and gate2_pass and gate3_pass)
        audit_payload = {
            "factor_id": int(fid),
            "name": fname,
            "source": rec.source,
            "factor_type": rec.factor_type,
            "status": rec.status,
            "oos_icir": oos_icir,
            "oos_ic_mean": oos_ic_mean,
            "turnover": turnover,
            "max_corr": max_corr,
            "most_corr_with": most_corr_with,
            "dir_consistency": dir_consistency,
            "residual_icir": resid_icir,
            "residual_t_stat": resid_t,
            "base_sharpe": base_ablation["sharpe"],
            "with_cand_sharpe": with_ablation["sharpe"],
            "delta_sharpe": delta_sharpe,
            "delta_ann_ret": delta_ann_ret,
            "gate1_pass": gate1_pass,
            "gate2_pass": gate2_pass,
            "gate3_pass": gate3_pass,
            "all_passed": all_passed,
            "oos_window": [dates[oos_start] if dates else "", dates[-1] if dates else ""],
        }
        results.append(audit_payload)

    return results


def compare_factors(
    panel: Panel,
    registry: FactorRegistry,
    factor_ids: Sequence[int],
    market: str = "futures",
    *,
    horizon: int = 5,
    min_names: int = 5,
    n_quantiles: int = 5,
) -> dict[str, Any]:
    """多因子 A/B 同台公平对比：严格交集 OOS 窗口、Rust Rank 相关性热力图与正交残差累积 IC。"""
    if len(factor_ids) < 2 or len(factor_ids) > 6:
        raise ValueError("多因子 A/B 对比需选择 2~6 个因子")

    norm_m = str(market).strip().lower() or resolve_panel_market(panel)
    profile = resolve_profile(norm_m)
    engine = panel.to_rust_engine()
    T, _ = panel.shape
    dates = [str(d) for d in panel.dates]
    _, label_mat = _resolve_target_label(panel, horizon=horizon)

    records: list[FactorRecord] = []
    directed_mats: list[np.ndarray] = []
    for fid in factor_ids:
        rec = registry.get_factor(int(fid), include_metrics=True)
        if rec is None or rec.market != norm_m:
            raise ValueError(f"未找到市场 {norm_m!r} 下的因子 id={fid}")
        mat, _ = _evaluate_record_matrix(rec, panel, engine, profile)
        records.append(rec)
        directed_mats.append(mat)

    # 1. 计算严格交集 OOS 有效日期掩码（所有对比因子与标签在当日均有 >= min_names 有效标的）
    inter_mask = np.ones(T, dtype=bool)
    for mat in directed_mats:
        valid_counts = np.sum(panel.valid_mask & np.isfinite(mat) & np.isfinite(label_mat), axis=1)
        inter_mask &= valid_counts >= min_names

    inter_rows = np.where(inter_mask)[0]
    if len(inter_rows) == 0:
        raise ValueError("所选因子在当前面板上不存在有效的严格交集日期窗口")

    s_row = int(inter_rows[0])
    e_row = int(inter_rows[-1]) + 1
    inter_dates = [dates[i] for i in inter_rows]

    # 2. 调用 Rust `factor_correlation_matrix` 计算 M x M 截面 Rank 相关矩阵（零 Python 循环）
    sliced_mats = [np.asfortranarray(m[s_row:e_row]) for m in directed_mats]
    corr_mat = np.asarray(
        engine.factor_correlation_matrix(
            sliced_mats,
            min_names=min_names,
            rank=True,
            start_row=s_row,
            end_row=e_row,
        ),
        dtype=np.float64,
    )
    corr_mat = np.where(np.isfinite(corr_mat), corr_mat, 0.0)

    # 3. 各因子在交集窗口下的标准化指标与累积 RankIC 曲线
    factors_summary: list[dict[str, Any]] = []
    cum_ic_series: list[dict[str, Any]] = []
    for rec, mat in zip(records, directed_mats, strict=True):
        eval_res = _evaluate_factor_matrix_core(
            engine=engine,
            dates=np.array(dates),
            factor_mat=mat,
            canon_expr=rec.canon_expr,
            canon_hash=rec.canon_hash,
            n_nodes=rec.n_nodes,
            depth=rec.depth,
            horizons=(horizon,),
            splits={"full": (dates[s_row], dates[e_row - 1])},
            min_names=min_names,
            n_quantiles=n_quantiles,
            force_sign=1,  # mat 已乘以 rec.sign
        )
        m_rec = eval_res.all_metrics[0] if eval_res.all_metrics else None
        daily_ic: list[float] = []
        for t in inter_rows:
            vm = panel.valid_mask[t, :] & np.isfinite(mat[t, :]) & np.isfinite(label_mat[t, :])
            ix = np.where(vm)[0]
            rho = _spearman_corr_1d(mat[t, ix], label_mat[t, ix]) if len(ix) >= min_names else 0.0
            daily_ic.append(0.0 if not math.isfinite(rho) else float(rho))
        cum_vals = np.cumsum(np.asarray(daily_ic, dtype=np.float64)).tolist()
        fname = rec.name or f"factor_{rec.id}"
        factors_summary.append(
            {
                "id": rec.id,
                "name": fname,
                "source": rec.source,
                "factor_type": rec.factor_type,
                "status": rec.status,
                "ic_mean": m_rec.ic_mean if m_rec else 0.0,
                "ic_ir": m_rec.ic_ir if m_rec else 0.0,
                "ic_win_rate": m_rec.ic_win_rate if m_rec else 0.0,
                "coverage": m_rec.coverage if m_rec else 0.0,
                "turnover": m_rec.turnover if m_rec else 0.0,
                "ls_sharpe": m_rec.ls_sharpe if m_rec else 0.0,
                "ls_annual_ret": m_rec.ls_annual_ret if m_rec else 0.0,
                "ls_max_dd": m_rec.ls_max_dd if m_rec else 0.0,
            }
        )
        cum_ic_series.append({"id": rec.id, "name": fname, "values": [float(v) for v in cum_vals]})

    # 4. 目标因子 (idx 0) 对其余基准组 (idx 1..K-1) 的正交残差累积 IC
    target_rec = records[0]
    target_mat = directed_mats[0]
    bench_mats = directed_mats[1:]
    span_res = _daily_cs_ols_residual_ic(
        target_mat,
        bench_mats,
        label_mat=label_mat,
        mask=panel.valid_mask,
        row_indices=inter_rows,
        min_names=min_names,
    )
    raw_target_ic = cum_ic_series[0]["values"]

    return {
        "market": norm_m,
        "horizon": int(horizon),
        "intersection_window": {
            "start_date": inter_dates[0],
            "end_date": inter_dates[-1],
            "n_days": len(inter_dates),
            "dates": inter_dates,
        },
        "factors": factors_summary,
        "corr_matrix": corr_mat.tolist(),
        "cum_ic_series": cum_ic_series,
        "orthogonal_residual": {
            "target_id": target_rec.id,
            "target_name": target_rec.name or f"factor_{target_rec.id}",
            "benchmark_names": [r.name or f"factor_{r.id}" for r in records[1:]],
            "residual_icir": span_res["icir"],
            "residual_t_stat": span_res["t_stat"],
            "residual_mean_ic": span_res["mean"],
            "dates": [dates[i] for i in span_res["row_indices"]],
            "cum_residual_ic": span_res["cum_ic"],
            "cum_raw_ic": raw_target_ic,
        },
    }
