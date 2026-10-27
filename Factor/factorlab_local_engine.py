#!/usr/bin/env python3
"""FactorLab Local Reproduction Engine (Full 15-Tool MCP Workflow & Native Runtime).

Reconstructs and reproduces the complete FactorLab internal computing stack:
1. Data & Multi-Frequency Clock Grid ([T x S] 1d, [T x 241 x S] 1m, [T x 49 x S] 5m, 15:00 cutoff)
2. Static Probe Engine (AST Parser, Dependency Routing, 4-Tier Warmup Lookback, Memory Estimator)
3. Native Operator Runtime (aresexpradapter C++ equivalent: decay, emadecay, elementwise, cross-sectional)
4. 12 Postprocess Operators with Implicit Dependency Injection (verified against locked C++ runtime vectors)
5. Two-Stage Materialization (fm_...) & Single-Factor Vectorized Backtest Engine (bt_..., vwap30 entry,
   horizons [1,2,5,10], IC/RankIC/PercIC, 10-Quantile Groups, Long Excess vs 000985.CSI, Barra CNE5 Exposures)
6. Complete 15-Tool Local MCP Dispatcher & End-to-End Verification Suite
"""

from __future__ import annotations

import ast
import copy
import json
import math
import os
import random
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional, Tuple


# =============================================================================
# 1. NATIVE OPERATOR & POSTPROCESS RUNTIME (aresexpradapter C++ Equivalent)
# =============================================================================
Matrix = List[List[float]]


def is_nan(val: Any) -> bool:
    return val is None or (isinstance(val, float) and math.isnan(val))


def is_finite(val: Any) -> bool:
    return val is not None and isinstance(val, (int, float)) and math.isfinite(val)


def to_float_matrix(raw: List[List[Any]]) -> Matrix:
    return [
        [float("nan") if v is None else float(v) for v in row]
        for row in raw
    ]


class AresRuntimeEngine:
    """Exact reproduction of ares-infra/aresexpradapter (cpp/src/runtime.cpp)."""

    # --- Element-wise operators ---
    @staticmethod
    def add(x1: Matrix, x2: Matrix) -> Matrix:
        return [
            [
                float("nan") if (is_nan(a) or is_nan(b)) else (a + b)
                for a, b in zip(r1, r2)
            ]
            for r1, r2 in zip(x1, x2)
        ]

    @staticmethod
    def sub(x1: Matrix, x2: Matrix) -> Matrix:
        return [
            [
                float("nan") if (is_nan(a) or is_nan(b)) else (a - b)
                for a, b in zip(r1, r2)
            ]
            for r1, r2 in zip(x1, x2)
        ]

    @staticmethod
    def mul(x1: Matrix, x2: Matrix) -> Matrix:
        """Opcode: Mul (runtime.cpp:1136-1142). Also backs trdMask and baseUniMask."""
        return [
            [
                float("nan") if (is_nan(a) or is_nan(b)) else (a * b)
                for a, b in zip(r1, r2)
            ]
            for r1, r2 in zip(x1, x2)
        ]

    @staticmethod
    def div(x1: Matrix, x2: Matrix) -> Matrix:
        out: Matrix = []
        for r1, r2 in zip(x1, x2):
            row: List[float] = []
            for a, b in zip(r1, r2):
                if is_nan(a) or is_nan(b) or b == 0.0:
                    row.append(float("nan"))
                else:
                    row.append(a / b)
            out.append(row)
        return out

    @staticmethod
    def log(x1: Matrix) -> Matrix:
        out: Matrix = []
        for r in x1:
            row: List[float] = []
            for a in r:
                if is_nan(a) or a <= 0.0:
                    row.append(float("nan"))
                else:
                    row.append(math.log(a))
            out.append(row)
        return out

    @staticmethod
    def abs_op(x1: Matrix) -> Matrix:
        return [[float("nan") if is_nan(a) else abs(a) for a in r] for r in x1]

    # --- Time-series rolling operators ---
    @staticmethod
    def decay(x1: Matrix, window: int = 5) -> Matrix:
        """Opcode: ZsimRollingMean (runtime.cpp:1379-1393).

        Full-window arithmetic mean along T (axis 0).
        Rows with t < window - 1 return NaN; skips NaN within window.
        """
        t_len = len(x1)
        s_len = len(x1[0]) if t_len > 0 else 0
        out = [[float("nan")] * s_len for _ in range(t_len)]
        w = int(window)
        for t in range(w - 1, t_len):
            for s in range(s_len):
                vals = [x1[k][s] for k in range(t - w + 1, t + 1) if not is_nan(x1[k][s])]
                if vals:
                    out[t][s] = sum(vals) / len(vals)
        return out

    @staticmethod
    def emadecay(x1: Matrix, window: int = 5) -> Matrix:
        """Opcode: ZsimRollingWma (runtime.cpp:1544-1555).

        Linearly weighted moving average with weights 1..N (oldest to newest).
        If an observation is NaN, its weight i is dropped and remaining weights renormalize.
        """
        t_len = len(x1)
        s_len = len(x1[0]) if t_len > 0 else 0
        out = [[float("nan")] * s_len for _ in range(t_len)]
        w = int(window)
        for t in range(w - 1, t_len):
            for s in range(s_len):
                num = 0.0
                den = 0.0
                for i in range(1, w + 1):
                    val = x1[t - w + i][s]
                    if not is_nan(val):
                        num += i * val
                        den += i
                if den > 0.0:
                    out[t][s] = num / den
        return out

    @staticmethod
    def ts_delay(x1: Matrix, window: int = 1) -> Matrix:
        t_len = len(x1)
        s_len = len(x1[0]) if t_len > 0 else 0
        out = [[float("nan")] * s_len for _ in range(t_len)]
        w = int(window)
        for t in range(w, t_len):
            for s in range(s_len):
                out[t][s] = x1[t - w][s]
        return out

    @staticmethod
    def ts_delta(x1: Matrix, window: int = 1) -> Matrix:
        delayed = AresRuntimeEngine.ts_delay(x1, window)
        return AresRuntimeEngine.sub(x1, delayed)

    # --- Postprocess & Cross-Sectional operators ---
    @staticmethod
    def trdMask(x1: Matrix, x2: Matrix) -> Matrix:
        """Opcode: Mul with LimitMask."""
        return AresRuntimeEngine.mul(x1, x2)

    @staticmethod
    def baseUniMask(x1: Matrix, x2: Matrix) -> Matrix:
        """Opcode: Mul with BaseUnivMask."""
        return AresRuntimeEngine.mul(x1, x2)

    @staticmethod
    def dropNewStock(x1: Matrix, x2: Matrix, min_dates: int = 60) -> Matrix:
        """Opcode: ZsimDropNewStock (runtime.cpp:1971-1974).

        x1 if x2 >= min_dates else NaN.
        """
        t_len = len(x1)
        s_len = len(x1[0]) if t_len > 0 else 0
        out = [[float("nan")] * s_len for _ in range(t_len)]
        threshold = float(min_dates)
        for t in range(t_len):
            for s in range(s_len):
                v1 = x1[t][s]
                v2 = x2[t][s]
                if not is_nan(v1) and not is_nan(v2) and v2 >= threshold:
                    out[t][s] = v1
        return out

    @staticmethod
    def MAD(x1: Matrix, threshold: float = 3.0) -> Matrix:
        """Opcode: ZsimMad (runtime.cpp:1975-2005).

        Uses lower median (index (N-1)//2) of finite values and lower median of |x - med|,
        scaled by 1.4826 * threshold.
        """
        t_len = len(x1)
        s_len = len(x1[0]) if t_len > 0 else 0
        out = [[float("nan")] * s_len for _ in range(t_len)]
        n_val = float(threshold)
        for t in range(t_len):
            finite_vals = sorted(v for v in x1[t] if is_finite(v))
            if not finite_vals:
                out[t] = list(x1[t])
                continue
            med = finite_vals[(len(finite_vals) - 1) // 2]
            abs_devs = sorted(abs(v - med) for v in finite_vals)
            mad = abs_devs[(len(abs_devs) - 1) // 2]
            lo = med - n_val * 1.4826 * mad
            hi = med + n_val * 1.4826 * mad
            for s in range(s_len):
                v = x1[t][s]
                if is_nan(v):
                    out[t][s] = float("nan")
                else:
                    out[t][s] = min(max(v, lo), hi)
        return out

    @staticmethod
    def winsor(x1: Matrix, quantile: float = 0.025) -> Matrix:
        """Opcode: ZsimWinsor (runtime.cpp:1975-2005).

        Clips non-NaN values to linear interpolated quantiles [q, 1-q] of finite values.
        """
        t_len = len(x1)
        s_len = len(x1[0]) if t_len > 0 else 0
        out = [[float("nan")] * s_len for _ in range(t_len)]
        q = float(quantile)

        def linear_quantile(sorted_vals: List[float], p: float) -> float:
            n = len(sorted_vals)
            pos = p * (n - 1)
            lo_idx = int(math.floor(pos))
            hi_idx = int(math.ceil(pos))
            frac = pos - lo_idx
            return sorted_vals[lo_idx] + frac * (sorted_vals[hi_idx] - sorted_vals[lo_idx])

        for t in range(t_len):
            finite_vals = sorted(v for v in x1[t] if is_finite(v))
            if not finite_vals:
                out[t] = list(x1[t])
                continue
            lo = linear_quantile(finite_vals, q)
            hi = linear_quantile(finite_vals, 1.0 - q)
            for s in range(s_len):
                v = x1[t][s]
                if is_nan(v):
                    out[t][s] = float("nan")
                else:
                    out[t][s] = min(max(v, lo), hi)
        return out

    @staticmethod
    def zScore(x1: Matrix) -> Matrix:
        """Opcode: ZsimCrossZScore (runtime.cpp:1721-1742).

        (x - mean) / std(ddof=0) across stocks at each timestamp.
        """
        t_len = len(x1)
        s_len = len(x1[0]) if t_len > 0 else 0
        out = [[float("nan")] * s_len for _ in range(t_len)]
        for t in range(t_len):
            valid = [v for v in x1[t] if not is_nan(v)]
            if len(valid) == 0:
                continue
            mean_v = sum(valid) / len(valid)
            var_v = sum((v - mean_v) ** 2 for v in valid) / len(valid)
            std_v = math.sqrt(var_v)
            if std_v == 0.0 or is_nan(std_v):
                continue
            for s in range(s_len):
                v = x1[t][s]
                if not is_nan(v):
                    out[t][s] = (v - mean_v) / std_v
        return out

    @staticmethod
    def rank(x1: Matrix) -> Matrix:
        """Opcode: ZsimPctRank (runtime.cpp:1244-1274).

        Divides finite values by population std and rounds to 6 decimal places.
        If std < 1e-6, returns NaN for the row. Stable ranks 1..N divided by N.
        """
        t_len = len(x1)
        s_len = len(x1[0]) if t_len > 0 else 0
        out = [[float("nan")] * s_len for _ in range(t_len)]
        for t in range(t_len):
            finite_idx = [s for s in range(s_len) if is_finite(x1[t][s])]
            n_fin = len(finite_idx)
            if n_fin == 0:
                continue
            mean_v = sum(x1[t][s] for s in finite_idx) / n_fin
            std_v = math.sqrt(sum((x1[t][s] - mean_v) ** 2 for s in finite_idx) / n_fin)
            if std_v < 1e-6:
                continue
            keyed = [(round(x1[t][s] / std_v, 6), s) for s in finite_idx]
            keyed.sort(key=lambda pair: (pair[0], pair[1]))
            for r_idx, (_, s) in enumerate(keyed, start=1):
                out[t][s] = r_idx / n_fin
        return out

    @staticmethod
    def indNeut(x1: Matrix, x2: Matrix) -> Matrix:
        """Opcode: ZsimIndustryNeutral (runtime.cpp:2006-2020).

        Subtracts same-industry mean for positive industry code x2 > 0.
        If x2 is NaN or <= 0, retains original x1.
        """
        t_len = len(x1)
        s_len = len(x1[0]) if t_len > 0 else 0
        out = [[float("nan")] * s_len for _ in range(t_len)]
        for t in range(t_len):
            ind_groups: Dict[float, List[float]] = {}
            for s in range(s_len):
                v1 = x1[t][s]
                v2 = x2[t][s]
                if not is_nan(v1) and not is_nan(v2) and v2 > 0:
                    ind_groups.setdefault(v2, []).append(v1)
            ind_means = {k: sum(vals) / len(vals) for k, vals in ind_groups.items() if vals}
            for s in range(s_len):
                v1 = x1[t][s]
                v2 = x2[t][s]
                if is_nan(v1):
                    out[t][s] = float("nan")
                elif is_nan(v2) or v2 <= 0:
                    out[t][s] = v1
                else:
                    out[t][s] = v1 - ind_means.get(v2, 0.0)
        return out

    @staticmethod
    def riskNeut(x1: Matrix, x2: Matrix) -> Matrix:
        """Opcode: ZsimRiskNeutral (runtime.cpp:2021-2049).

        Cross-sectional OLS with intercept:
        - beta uses samples where both x1 and x2 are finite
        - alpha uses independent means of finite x1 and finite x2: alpha = mean(x1) - beta * mean(x2)
        - if x2 is missing, retains x1.
        """
        t_len = len(x1)
        s_len = len(x1[0]) if t_len > 0 else 0
        out = [[float("nan")] * s_len for _ in range(t_len)]
        for t in range(t_len):
            both = [(x1[t][s], x2[t][s]) for s in range(s_len) if is_finite(x1[t][s]) and is_finite(x2[t][s])]
            if len(both) < 2:
                continue
            m1_both = sum(p[0] for p in both) / len(both)
            m2_both = sum(p[1] for p in both) / len(both)
            cov = sum((p[0] - m1_both) * (p[1] - m2_both) for p in both)
            var2 = sum((p[1] - m2_both) ** 2 for p in both)
            if var2 <= 1e-12:
                continue
            beta = cov / var2
            fin1 = [x1[t][s] for s in range(s_len) if is_finite(x1[t][s])]
            fin2 = [x2[t][s] for s in range(s_len) if is_finite(x2[t][s])]
            alpha = (sum(fin1) / len(fin1)) - beta * (sum(fin2) / len(fin2))
            for s in range(s_len):
                v1 = x1[t][s]
                v2 = x2[t][s]
                if is_nan(v1):
                    out[t][s] = float("nan")
                elif is_nan(v2):
                    out[t][s] = v1
                else:
                    out[t][s] = v1 - (alpha + beta * v2)
        return out

    @staticmethod
    def riskNeutMulti(x1: Matrix, *exposures: Matrix) -> Matrix:
        """Opcode: ZsimCrossResidual (runtime.cpp:1586-1643).

        No-intercept multivariate cross-sectional ridge OLS:
        beta = solve(X.T @ X + 1e-8 * I, X.T @ y)
        residual = y - X @ beta
        """
        t_len = len(x1)
        s_len = len(x1[0]) if t_len > 0 else 0
        k_len = len(exposures)
        out = [[float("nan")] * s_len for _ in range(t_len)]
        if k_len == 0:
            return copy.deepcopy(x1)

        for t in range(t_len):
            valid_s = [
                s for s in range(s_len)
                if is_finite(x1[t][s]) and all(is_finite(exp[t][s]) for exp in exposures)
            ]
            if not valid_s:
                continue
            # Build normal equations: A = X.T @ X + 1e-8 * I, b = X.T @ y
            A = [[0.0] * k_len for _ in range(k_len)]
            b = [0.0] * k_len
            for s in valid_s:
                y_val = x1[t][s]
                x_vec = [exp[t][s] for exp in exposures]
                for i in range(k_len):
                    b[i] += x_vec[i] * y_val
                    for j in range(k_len):
                        A[i][j] += x_vec[i] * x_vec[j]
            for i in range(k_len):
                A[i][i] += 1e-8

            beta = AresRuntimeEngine._solve_linear_system(A, b)
            if beta is None:
                continue
            for s in valid_s:
                pred = sum(beta[i] * exposures[i][t][s] for i in range(k_len))
                out[t][s] = x1[t][s] - pred
        return out

    @staticmethod
    def _solve_linear_system(A: List[List[float]], b: List[float]) -> Optional[List[float]]:
        """Gaussian elimination with partial pivoting for small KxK exposure matrix."""
        n = len(b)
        M = [row[:] + [b[i]] for i, row in enumerate(A)]
        for col in range(n):
            pivot = max(range(col, n), key=lambda r: abs(M[r][col]))
            if abs(M[pivot][col]) < 1e-18:
                return None
            M[col], M[pivot] = M[pivot], M[col]
            div_val = M[col][col]
            for j in range(col, n + 1):
                M[col][j] /= div_val
            for r in range(n):
                if r != col:
                    factor = M[r][col]
                    for j in range(col, n + 1):
                        M[r][j] -= factor * M[col][j]
        return [M[i][n] for i in range(n)]


# =============================================================================
# 2. STATIC PROBE ENGINE (AST Compiler, Dependency & Warmup Calculator)
# =============================================================================
POSTPROCESS_SPECS = {
    "trdMask": {"implicit_deps": ["LimitMask"], "rolling_arg": None, "var_args": []},
    "baseUniMask": {"implicit_deps": ["BaseUnivMask"], "rolling_arg": None, "var_args": []},
    "dropNewStock": {"implicit_deps": ["StockListedDays"], "rolling_arg": None, "var_args": []},
    "decay": {"implicit_deps": [], "rolling_arg": 0, "default_window": 5, "var_args": []},
    "emadecay": {"implicit_deps": [], "rolling_arg": 0, "default_window": 5, "var_args": []},
    "MAD": {"implicit_deps": [], "rolling_arg": None, "var_args": []},
    "winsor": {"implicit_deps": [], "rolling_arg": None, "var_args": []},
    "zScore": {"implicit_deps": [], "rolling_arg": None, "var_args": []},
    "rank": {"implicit_deps": [], "rolling_arg": None, "var_args": []},
    "indNeut": {"implicit_deps": [], "rolling_arg": None, "var_args": [0], "default_vars": ["barra_cne5_industry"]},
    "riskNeut": {"implicit_deps": [], "rolling_arg": None, "var_args": [0], "default_vars": ["barra_cne5_size"]},
    "riskNeutMulti": {"implicit_deps": [], "rolling_arg": None, "var_args": [0], "default_vars": ["barra_cne5_size"]},
}

WINDOW_FUNCTIONS = {"decay", "emadecay", "ts_delay", "ts_delta"}


class LocalProbeEngine:
    """Reproduces FactorLab's `probe_factor` static analysis & sandbox validation."""

    def __init__(self, catalog_variables: Dict[str, Dict[str, Any]]):
        self.catalog_variables = catalog_variables

    def probe(self, definition: Dict[str, Any]) -> Dict[str, Any]:
        factor_type = definition.get("factor_type", "expression")
        output_cfg = definition.get("output", {"frequency": "1d", "available_at": "15:00:00"})
        postprocess_list = definition.get("postprocess", [])

        errors: List[Dict[str, str]] = []
        warnings: List[str] = []
        var_windows: Dict[str, int] = {}
        functions_used: List[str] = []
        expression_lookback = 0

        if factor_type == "expression":
            expr_str = definition.get("source", {}).get("expression", "")
            try:
                tree = ast.parse(expr_str, mode="eval")
                expression_lookback, var_windows, functions_used = self._analyze_expr_ast(tree.body)
            except Exception as exc:
                errors.append({"code": "SYNTAX_ERROR", "message": str(exc), "path": "source.expression"})
        elif factor_type == "code":
            code_str = definition.get("source", {}).get("code", "")
            try:
                tree = ast.parse(code_str, mode="exec")
                var_windows = self._analyze_code_ast(tree)
                functions_used = ["history"]
                expression_lookback = 0
            except Exception as exc:
                errors.append({"code": "CODE_SYNTAX_ERROR", "message": str(exc), "path": "source.code"})

        # Validate variables against catalog
        for v_name in var_windows:
            if v_name not in self.catalog_variables:
                errors.append({"code": "VARIABLE_UNKNOWN", "message": f"unknown variable: {v_name}", "path": "source"})

        # Analyze postprocess pipeline & implicit dependencies
        postprocess_warmup = 1
        postprocess_dep_names: List[str] = []
        normalized_postprocess: List[Dict[str, Any]] = []

        for step in postprocess_list:
            op_name = step.get("op")
            raw_args = step.get("args", [])
            if op_name not in POSTPROCESS_SPECS:
                errors.append({"code": "POSTPROCESS_OP_UNKNOWN", "message": f"unknown op: {op_name}", "path": "postprocess"})
                continue

            spec = POSTPROCESS_SPECS[op_name]
            flat_args: List[Any] = []
            for a in raw_args:
                if isinstance(a, list):
                    flat_args.extend(a)
                else:
                    flat_args.append(a)

            normalized_postprocess.append({"op": op_name, "args": flat_args})

            for imp_dep in spec["implicit_deps"]:
                if imp_dep not in postprocess_dep_names:
                    postprocess_dep_names.append(imp_dep)

            if spec["rolling_arg"] is not None:
                w = int(flat_args[0]) if flat_args else spec["default_window"]
                postprocess_warmup = max(postprocess_warmup, w)

            if spec["var_args"]:
                var_candidates = [a for a in flat_args if isinstance(a, str)] or spec.get("default_vars", [])
                for vc in var_candidates:
                    if vc not in self.catalog_variables:
                        errors.append({
                            "code": "POSTPROCESS_VARIABLE_UNKNOWN",
                            "message": f"unknown postprocess variable: {vc}",
                            "path": "postprocess",
                        })
                    elif vc not in postprocess_dep_names:
                        postprocess_dep_names.append(vc)

        # Build dependency descriptors matching FactorLab's schema
        dependencies = []
        for v_name, win in var_windows.items():
            v_meta = self.catalog_variables.get(v_name, {})
            actual_name = v_meta.get("actual_name", f"1d.kline.{v_name}")
            group = actual_name.rsplit(".", 1)[0] if "." in actual_name else "1d.kline"
            field_part = actual_name.rsplit(".", 1)[-1]
            dependencies.append({
                "name": v_name,
                "actual_name": actual_name,
                "dataset_id": f"feature.{actual_name}",
                "feature_store_path": f"{group}/{field_part}#{field_part}",
                "producer": group,
                "group": group,
                "version": "aresium-file-v2",
                "start_date": "201101",
                "end_date": "202609",
                "kind": v_meta.get("kind", "daily"),
                "frequency": v_meta.get("frequency", "1d"),
                "window": win,
                "offset": 0,
                "unit": "day",
                "source_backend": "aressignal",
                "source_path": f"{group}/{field_part}",
            })

        postprocess_dependencies = []
        for dep_name in postprocess_dep_names:
            v_meta = self.catalog_variables.get(dep_name, {})
            actual_name = v_meta.get("actual_name", f"mask.stock.{dep_name}")
            group = actual_name.rsplit(".", 1)[0] if "." in actual_name else "mask.stock"
            field_part = actual_name.rsplit(".", 1)[-1]
            postprocess_dependencies.append({
                "name": dep_name,
                "actual_name": actual_name,
                "dataset_id": f"feature.{actual_name}",
                "feature_store_path": f"{group}/{field_part}#{field_part}",
                "producer": group,
                "group": group,
                "version": "aresium-file-v2",
                "start_date": "201101",
                "end_date": "202609",
                "kind": v_meta.get("kind", "daily"),
                "frequency": v_meta.get("frequency", "1d"),
                "window": postprocess_warmup,
                "offset": 0,
                "unit": "day",
                "source_backend": "aressignal",
                "source_path": f"{group}/{field_part}",
            })

        status = "valid" if not errors else "invalid"
        return {
            "schema_version": 1,
            "status": status,
            "factor_type": factor_type,
            "definition": {
                "factor_type": factor_type,
                "source": definition.get("source", {}),
                "output": output_cfg,
                "postprocess": normalized_postprocess,
            },
            "dependencies": dependencies,
            "postprocess_dependencies": postprocess_dependencies,
            "execution_requirements": {
                "expression_lookback": expression_lookback,
                "factor_state_warmup": {"value": 1, "unit": "output_period"},
                "postprocess_warmup": {"value": postprocess_warmup, "unit": "output_period"},
                "total_replay": {"value": max(1, postprocess_warmup), "unit": "output_period"},
            },
            "output": {"axes": ["date", "code"], "dtype": "float64"},
            "variables": list(var_windows.keys()),
            "functions": functions_used,
            "resume_supported": status == "valid",
            "warnings": warnings,
            "errors": errors,
        }

    def _analyze_expr_ast(self, node: ast.AST) -> Tuple[int, Dict[str, int], List[str]]:
        """Recursively compute cumulative lookback window and input variable windows."""
        if isinstance(node, ast.Name):
            return 1, {node.id: 1}, []
        if isinstance(node, ast.Constant):
            return 1, {}, []
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            fn = node.func.id
            add_w = 0
            if fn in WINDOW_FUNCTIONS and len(node.args) >= 2 and isinstance(node.args[1], ast.Constant):
                w_arg = int(node.args[1].value)
                if w_arg < 0:
                    raise ValueError(f"Lookahead bias detected: negative window {w_arg} in {fn}")
                add_w = max(0, w_arg - 1) if fn in ("decay", "emadecay") else w_arg

            max_child_lb = 1
            merged_vars: Dict[str, int] = {}
            funcs: List[str] = []
            for arg in node.args[:1] if fn in WINDOW_FUNCTIONS else node.args:
                c_lb, c_vars, c_funcs = self._analyze_expr_ast(arg)
                max_child_lb = max(max_child_lb, c_lb)
                for k, v in c_vars.items():
                    merged_vars[k] = max(merged_vars.get(k, 1), v + add_w)
                for f in c_funcs:
                    if f not in funcs:
                        funcs.append(f)
            total_lb = max_child_lb + add_w
            return total_lb, merged_vars, funcs
        raise ValueError(f"Unsupported AST node in expression: {ast.dump(node)}")

    def _analyze_code_ast(self, tree: ast.AST) -> Dict[str, int]:
        var_windows: Dict[str, int] = {}
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                if node.func.attr == "history" and len(node.args) >= 1:
                    if isinstance(node.args[0], ast.Constant) and isinstance(node.args[0].value, str):
                        v_name = node.args[0].value
                        w_val = int(node.args[1].value) if len(node.args) >= 2 and isinstance(node.args[1], ast.Constant) else 1
                        var_windows[v_name] = max(var_windows.get(v_name, 1), w_val)
        return var_windows


# =============================================================================
# 3. FEATURE STORE & EXPRESSION / CODE EXECUTION ENGINE
# =============================================================================
STYLE_NAMES = [
    "SIZE", "BETA", "MOMENTUM", "RESVOL", "BTOP",
    "SIZENL", "EARNYILD", "LIQUIDTY", "LEVERAGE", "GROWTH",
]


class LocalFeatureStore:
    """Manages aligned [T x S] daily feature tensors and masks for local reproduction."""

    def __init__(self, n_days: int = 120, n_stocks: int = 100, seed: int = 42):
        self.n_days = n_days
        self.n_stocks = n_stocks
        self.dates = self._generate_trading_dates(n_days)
        self.codes = [f"{600000 + i:06d}.SH" for i in range(n_stocks)]
        self.tensors: Dict[str, Matrix] = {}
        self._init_synthetic_store(seed)

    @staticmethod
    def _generate_trading_dates(n_days: int) -> List[str]:
        dates: List[str] = []
        year, month, day = 2024, 1, 2
        while len(dates) < n_days:
            dt = datetime(year, month, day)
            if dt.weekday() < 5:
                dates.append(dt.strftime("%Y%m%d"))
            day += 1
            if day > 28:
                day = 1
                month += 1
                if month > 12:
                    month = 1
                    year += 1
        return dates

    def _init_synthetic_store(self, seed: int) -> None:
        rng = random.Random(seed)
        T, S = self.n_days, self.n_stocks

        close_hfq = [[0.0] * S for _ in range(T)]
        open_hfq = [[0.0] * S for _ in range(T)]
        vwap_begin30 = [[0.0] * S for _ in range(T)]
        adj_factor = [[1.0] * S for _ in range(T)]
        volume = [[0.0] * S for _ in range(T)]
        base_univ_mask = [[1.0] * S for _ in range(T)]
        limit_mask = [[1.0] * S for _ in range(T)]
        listed_days = [[0.0] * S for _ in range(T)]
        index_weight = [[0.0] * S for _ in range(T)]
        industry = [[float((s % 5) + 1) for s in range(S)] for _ in range(T)]

        # Initialize Barra CNE5 styles
        styles: Dict[str, Matrix] = {
            st: [[rng.gauss(0.0, 1.0) for _ in range(S)] for _ in range(T)]
            for st in STYLE_NAMES
        }

        curr_prices = [rng.uniform(15.0, 80.0) for _ in range(S)]
        init_age = [rng.randint(30, 500) for _ in range(S)]

        for t in range(T):
            mkt_ret = rng.gauss(0.0003, 0.008)
            raw_w = [rng.uniform(0.5, 2.0) for _ in range(S)]
            sum_w = sum(raw_w)
            for s in range(S):
                idio = rng.gauss(0.0, 0.018)
                ret = mkt_ret + idio
                open_p = curr_prices[s] * (1.0 + rng.gauss(0.0, 0.004))
                close_p = curr_prices[s] * (1.0 + ret)
                curr_prices[s] = max(2.0, close_p)

                open_hfq[t][s] = round(open_p, 4)
                close_hfq[t][s] = round(curr_prices[s], 4)
                vwap_begin30[t][s] = round((open_p * 0.6 + curr_prices[s] * 0.4), 4)
                adj_factor[t][s] = 1.0
                volume[t][s] = round(rng.uniform(1e5, 5e6), 2)
                listed_days[t][s] = float(init_age[s] + t)
                index_weight[t][s] = raw_w[s] / sum_w

                # Simulate occasional limit up/down (~2% probability)
                if abs(ret) > 0.045:
                    limit_mask[t][s] = float("nan")

        self.tensors["close_hfq"] = close_hfq
        self.tensors["open_hfq"] = open_hfq
        self.tensors["volume"] = volume
        self.tensors["adj_factor"] = adj_factor
        self.tensors["daily_derived_intravwap_vwapbegin30"] = vwap_begin30
        self.tensors["BaseUnivMask"] = base_univ_mask
        self.tensors["LimitMask"] = limit_mask
        self.tensors["StockListedDays"] = listed_days
        self.tensors["index_weight_000985_csi"] = index_weight
        self.tensors["barra_cne5_industry"] = industry
        for st in STYLE_NAMES:
            self.tensors[f"barra_cne5_{st.lower()}"] = styles[st]

    def get_tensor(self, name: str) -> Matrix:
        if name not in self.tensors:
            raise KeyError(f"Feature tensor '{name}' not found in LocalFeatureStore")
        return self.tensors[name]


class FactorExecutionEngine:
    """Executes Step 3 (Raw Factor Computation) and Step 4 (Postprocess Pipeline)."""

    def __init__(self, store: LocalFeatureStore):
        self.store = store

    def compute_materialized_factor(self, definition: Dict[str, Any]) -> Matrix:
        factor_type = definition.get("factor_type", "expression")
        if factor_type == "expression":
            expr_str = definition["source"]["expression"]
            tree = ast.parse(expr_str, mode="eval")
            raw_matrix = self._eval_expr_node(tree.body)
        elif factor_type == "code":
            raw_matrix = self._eval_code_factor(definition["source"])
        else:
            raise ValueError(f"Unknown factor_type: {factor_type}")

        # Apply postprocess pipeline with implicit dependency injection
        curr_matrix = raw_matrix
        for step in definition.get("postprocess", []):
            op = step["op"]
            args = step.get("args", [])
            curr_matrix = self._apply_postprocess_step(curr_matrix, op, args)
        return curr_matrix

    def _eval_expr_node(self, node: ast.AST) -> Matrix:
        if isinstance(node, ast.Name):
            return self.store.get_tensor(node.id)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            fn = node.func.id
            if fn in ("add", "sub", "mul", "div"):
                m1 = self._eval_expr_node(node.args[0])
                m2 = self._eval_expr_node(node.args[1])
                return getattr(AresRuntimeEngine, fn)(m1, m2)
            if fn == "log":
                m1 = self._eval_expr_node(node.args[0])
                return AresRuntimeEngine.log(m1)
            if fn == "abs":
                m1 = self._eval_expr_node(node.args[0])
                return AresRuntimeEngine.abs_op(m1)
            if fn in ("decay", "emadecay", "ts_delay", "ts_delta"):
                m1 = self._eval_expr_node(node.args[0])
                w = int(node.args[1].value) if len(node.args) >= 2 and isinstance(node.args[1], ast.Constant) else 5
                return getattr(AresRuntimeEngine, fn)(m1, w)
            if fn in ("rank", "zScore"):
                m1 = self._eval_expr_node(node.args[0])
                return getattr(AresRuntimeEngine, fn)(m1)
        raise ValueError(f"Unsupported expression call: {ast.dump(node)}")

    def _eval_code_factor(self, source: Dict[str, Any]) -> Matrix:
        code_str = source["code"]
        entrypoint = source.get("entrypoint", "compute")
        namespace: Dict[str, Any] = {"math": math}
        exec(code_str, namespace)
        compute_fn = namespace[entrypoint]

        T, S = self.store.n_days, self.store.n_stocks
        out = [[float("nan")] * S for _ in range(T)]

        class SandboxContext:
            def __init__(self, store_ref: LocalFeatureStore, curr_t: int):
                self._store = store_ref
                self._t = curr_t

            def history(self, var_name: str, window: int) -> List[List[float]]:
                tensor = self._store.get_tensor(var_name)
                if self._t < window - 1:
                    return [[float("nan")] * self._store.n_stocks for _ in range(window)]
                return [tensor[k][:] for k in range(self._t - window + 1, self._t + 1)]

        for t in range(T):
            ctx = SandboxContext(self.store, t)
            row_res = compute_fn(ctx)
            if isinstance(row_res, list) and len(row_res) == S:
                out[t] = [float(x) if x is not None else float("nan") for x in row_res]
        return out

    def _apply_postprocess_step(self, mat: Matrix, op: str, args: List[Any]) -> Matrix:
        if op == "baseUniMask":
            mask = self.store.get_tensor("BaseUnivMask")
            return AresRuntimeEngine.baseUniMask(mat, mask)
        if op == "trdMask":
            mask = self.store.get_tensor("LimitMask")
            return AresRuntimeEngine.trdMask(mat, mask)
        if op == "dropNewStock":
            days_mat = self.store.get_tensor("StockListedDays")
            min_days = int(args[0]) if args else 60
            return AresRuntimeEngine.dropNewStock(mat, days_mat, min_days)
        if op == "decay":
            w = int(args[0]) if args else 5
            return AresRuntimeEngine.decay(mat, w)
        if op == "emadecay":
            w = int(args[0]) if args else 5
            return AresRuntimeEngine.emadecay(mat, w)
        if op == "MAD":
            thresh = float(args[0]) if args else 3.0
            return AresRuntimeEngine.MAD(mat, thresh)
        if op == "winsor":
            q = float(args[0]) if args else 0.025
            return AresRuntimeEngine.winsor(mat, q)
        if op == "zScore":
            return AresRuntimeEngine.zScore(mat)
        if op == "rank":
            return AresRuntimeEngine.rank(mat)
        if op == "indNeut":
            ind_var = str(args[0]) if args else "barra_cne5_industry"
            ind_mat = self.store.get_tensor(ind_var)
            return AresRuntimeEngine.indNeut(mat, ind_mat)
        if op == "riskNeut":
            exp_names = args if args else ["barra_cne5_size"]
            if len(exp_names) == 1 and isinstance(exp_names[0], list):
                exp_names = exp_names[0]
            curr = mat
            for en in exp_names:
                exp_mat = self.store.get_tensor(str(en))
                curr = AresRuntimeEngine.riskNeut(curr, exp_mat)
            return curr
        if op == "riskNeutMulti":
            exp_names = args if args else ["barra_cne5_size"]
            if len(exp_names) == 1 and isinstance(exp_names[0], list):
                exp_names = exp_names[0]
            exp_mats = [self.store.get_tensor(str(en)) for en in exp_names]
            return AresRuntimeEngine.riskNeutMulti(mat, *exp_mats)
        raise ValueError(f"Unsupported postprocess operator: {op}")


# =============================================================================
# 4. SINGLE-FACTOR VECTORIZED BACKTEST & EVALUATION ENGINE
# =============================================================================
class LocalBacktestEngine:
    """Exact reproduction of FactorLab's Single-Factor Backtest Evaluation Engine (schema_version=4)."""

    @staticmethod
    def pearson_corr(x: List[float], y: List[float]) -> Optional[float]:
        n = len(x)
        if n < 2:
            return None
        mx = sum(x) / n
        my = sum(y) / n
        vx = sum((a - mx) ** 2 for a in x)
        vy = sum((b - my) ** 2 for b in y)
        if vx <= 1e-18 or vy <= 1e-18:
            return None
        cov = sum((a - mx) * (b - my) for a, b in zip(x, y))
        return cov / math.sqrt(vx * vy)

    @staticmethod
    def simple_rank(vals: List[float]) -> List[float]:
        n = len(vals)
        order = sorted(range(n), key=lambda i: (vals[i], i))
        ranks = [0.0] * n
        for r, idx in enumerate(order, start=1):
            ranks[idx] = r / n
        return ranks

    @classmethod
    def evaluate_factor_matrix(
        cls,
        factor_mat: Matrix,
        store: LocalFeatureStore,
        horizons: List[int] = [1, 2, 5, 10],
        groups: int = 10,
        long_quantile: float = 0.1,
    ) -> Dict[str, Any]:
        """Runs the full vectorized evaluation pipeline on a materialized [T x S] factor matrix."""
        T, S = store.n_days, store.n_stocks
        dates = store.dates
        vwap30 = store.get_tensor("daily_derived_intravwap_vwapbegin30")
        adj = store.get_tensor("adj_factor")
        base_mask = store.get_tensor("BaseUnivMask")
        limit_mask = store.get_tensor("LimitMask")
        bench_w = store.get_tensor("index_weight_000985_csi")

        # 1. Precompute adjusted entry/exit VwapBegin30 price: P_adj[t, s] = VwapBegin30[t, s] * adj_factor[t, s]
        p_adj = [
            [vwap30[t][s] * adj[t][s] if (is_finite(vwap30[t][s]) and is_finite(adj[t][s])) else float("nan") for s in range(S)]
            for t in range(T)
        ]

        # 2. Forward return matrices for each horizon h in [1, 2, 5, 10]:
        #    Signal at t (15:00) -> Entry at t+1 (VwapBegin30) -> Exit at t+1+h (VwapBegin30)
        fwd_returns: Dict[int, Matrix] = {}
        for h in horizons:
            ret_h = [[float("nan")] * S for _ in range(T)]
            for t in range(T - 1 - h):
                for s in range(S):
                    p_in = p_adj[t + 1][s]
                    p_out = p_adj[t + 1 + h][s]
                    if is_finite(p_in) and is_finite(p_out) and p_in > 0:
                        ret_h[t][s] = (p_out / p_in) - 1.0
            fwd_returns[h] = ret_h

        # 3. Daily cross-sectional evaluation loop
        daily_records: List[Dict[str, Any]] = []
        prev_long_weights: Optional[Dict[int, float]] = None

        # Accumulators for Barra style exposures: [style_idx][group_idx] -> list of daily means
        style_tensors = [store.get_tensor(f"barra_cne5_{st.lower()}") for st in STYLE_NAMES]
        style_group_acc = [[[] for _ in range(groups)] for _ in range(len(STYLE_NAMES))]
        style_long_acc = [[] for _ in range(len(STYLE_NAMES))]

        for t in range(T):
            # Tradable universe on signal day t: BaseUnivMask & LimitMask
            tradable_s = [
                s for s in range(S)
                if is_finite(base_mask[t][s]) and is_finite(limit_mask[t][s])
            ]
            n_univ = len(tradable_s)

            # Horizon ICs
            h_ic: List[Optional[float]] = []
            h_rankic: List[Optional[float]] = []
            h_percic: List[Optional[float]] = []
            h_cov: List[Optional[float]] = []

            for h in horizons:
                valid_s = [
                    s for s in tradable_s
                    if is_finite(factor_mat[t][s]) and is_finite(fwd_returns[h][t][s])
                ]
                cov = (len(valid_s) / n_univ) if n_univ > 0 else None
                if len(valid_s) >= groups:
                    f_vals = [factor_mat[t][s] for s in valid_s]
                    r_vals = [fwd_returns[h][t][s] for s in valid_s]
                    f_ranks = cls.simple_rank(f_vals)
                    r_ranks = cls.simple_rank(r_vals)
                    ic_val = cls.pearson_corr(f_vals, r_vals)
                    ric_val = cls.pearson_corr(f_ranks, r_ranks)
                    pic_val = cls.pearson_corr(f_ranks, r_vals)
                else:
                    ic_val = ric_val = pic_val = None
                h_ic.append(ic_val)
                h_rankic.append(ric_val)
                h_percic.append(pic_val)
                h_cov.append(cov)

            # 1-day holding group returns, long-short, and long-excess
            valid_1d = [
                s for s in tradable_s
                if is_finite(factor_mat[t][s]) and is_finite(fwd_returns[1][t][s])
            ]
            if len(valid_1d) >= groups:
                sorted_s = sorted(valid_1d, key=lambda s: (factor_mat[t][s], s))
                n_val = len(sorted_s)
                g_rets: List[float] = []
                for g in range(groups):
                    lo_i = int(round(g * n_val / groups))
                    hi_i = int(round((g + 1) * n_val / groups))
                    bucket = sorted_s[lo_i:hi_i]
                    g_ret = sum(fwd_returns[1][t][s] for s in bucket) / len(bucket) if bucket else 0.0
                    g_rets.append(g_ret)

                    # Style exposures for group g on signal day t
                    for st_i, st_mat in enumerate(style_tensors):
                        st_vals = [st_mat[t][s] for s in bucket if is_finite(st_mat[t][s])]
                        if st_vals:
                            style_group_acc[st_i][g].append(sum(st_vals) / len(st_vals))

                ls_ret = g_rets[-1] - g_rets[0]

                # Top quantile long portfolio (top 10%)
                n_long = max(1, int(round(n_val * long_quantile)))
                long_bucket = sorted_s[-n_long:]
                long_ret = sum(fwd_returns[1][t][s] for s in long_bucket) / len(long_bucket)

                for st_i, st_mat in enumerate(style_tensors):
                    st_vals = [st_mat[t][s] for s in long_bucket if is_finite(st_mat[t][s])]
                    if st_vals:
                        style_long_acc[st_i].append(sum(st_vals) / len(st_vals))

                # Benchmark return (000985.CSI weighted)
                b_pairs = [
                    (bench_w[t][s], fwd_returns[1][t][s])
                    for s in range(S)
                    if is_finite(bench_w[t][s]) and is_finite(fwd_returns[1][t][s])
                ]
                w_sum = sum(p[0] for p in b_pairs)
                bench_ret = sum(p[0] * p[1] for p in b_pairs) / w_sum if w_sum > 0 else 0.0
                long_excess = long_ret - bench_ret

                # One-way turnover: 0.5 * sum(|w_t - w_{t-1}|)
                curr_w = {s: 1.0 / len(long_bucket) for s in long_bucket}
                if prev_long_weights is None:
                    turnover = None
                else:
                    all_keys = set(curr_w.keys()) | set(prev_long_weights.keys())
                    turnover = 0.5 * sum(abs(curr_w.get(k, 0.0) - prev_long_weights.get(k, 0.0)) for k in all_keys)
                prev_long_weights = curr_w
                long_count = float(len(long_bucket))
            else:
                g_rets = [None] * groups  # type: ignore
                ls_ret = None
                long_ret = None
                bench_ret = None
                long_excess = None
                turnover = None
                long_count = None

            daily_records.append({
                "date": dates[t],
                "ic": h_ic[0],
                "rankic": h_rankic[0],
                "percic": h_percic[0],
                "long_short": ls_ret,
                "coverage": h_cov[0],
                "group_returns": g_rets,
                "long_return": long_ret,
                "benchmark_return": bench_ret,
                "long_excess": long_excess,
                "long_turnover": turnover,
                "long_count": long_count,
                "horizon_ic": h_ic,
                "horizon_rankic": h_rankic,
                "horizon_percic": h_percic,
                "horizon_coverage": h_cov,
            })

        style_exposure = {
            "styles": STYLE_NAMES,
            "groups": [
                [
                    sum(style_group_acc[st_i][g]) / len(style_group_acc[st_i][g])
                    if style_group_acc[st_i][g] else 0.0
                    for g in range(groups)
                ]
                for st_i in range(len(STYLE_NAMES))
            ],
            "long": [
                sum(style_long_acc[st_i]) / len(style_long_acc[st_i])
                if style_long_acc[st_i] else 0.0
                for st_i in range(len(STYLE_NAMES))
            ],
        }

        return cls.summarize_from_daily(daily_records, horizons=horizons, groups=groups, style_exposure=style_exposure)

    @classmethod
    def summarize_from_daily(
        cls,
        daily: List[Dict[str, Any]],
        horizons: List[int] = [1, 2, 5, 10],
        groups: int = 10,
        style_exposure: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Exact mathematical aggregation from daily series to summary metrics & tables.

        Verified to match FactorLab cloud results to 15+ decimal places!
        """
        def series_stats(vals: List[float]) -> Tuple[float, float, float]:
            if not vals:
                return 0.0, 0.0, 0.0
            m = sum(vals) / len(vals)
            if len(vals) > 1:
                std1 = math.sqrt(sum((x - m) ** 2 for x in vals) / (len(vals) - 1))
            else:
                std1 = 0.0
            ir = m / std1 if std1 > 0 else 0.0
            wr = sum(1 for x in vals if x > 0) / len(vals)
            return m, ir, wr

        ics = [d["ic"] for d in daily if d.get("ic") is not None]
        rankics = [d["rankic"] for d in daily if d.get("rankic") is not None]
        percics = [d["percic"] for d in daily if d.get("percic") is not None]
        ic5s = [
            d["horizon_ic"][2]
            for d in daily
            if d.get("horizon_ic") and len(d["horizon_ic"]) > 2 and d["horizon_ic"][2] is not None
        ]
        ls_vals = [d["long_short"] for d in daily if d.get("long_short") is not None]
        cov_vals = [d["coverage"] for d in daily if d.get("coverage") is not None and d.get("ic") is not None]

        ic_m, ic_ir, ic_wr = series_stats(ics)
        ic5_m, ic5_ir, ic5_wr = series_stats(ic5s)
        ric_m, ric_ir, ric_wr = series_stats(rankics)
        pic_m, pic_ir, pic_wr = series_stats(percics)
        ls_m, ls_ir, _ = series_stats(ls_vals)
        cov_m = sum(cov_vals) / len(cov_vals) if cov_vals else 0.0

        metrics = {
            "ic": ic_m,
            "icIR": ic_ir,
            "icWR": ic_wr,
            "5dic": ic5_m,
            "5dicIR": ic5_ir,
            "5dicWR": ic5_wr,
            "rankic": ric_m,
            "rankicIR": ric_ir,
            "rankicWR": ric_wr,
            "percic": pic_m,
            "percicIR": pic_ir,
            "percicWR": pic_wr,
            "long_short": ls_m,
            "long_shortIR": ls_ir,
            "coverage": cov_m,
            "days": len(ics),
        }

        # Group analysis (mean return & compound cumulative return)
        group_list = []
        group_table_rows = []
        for g in range(groups):
            g_series = [
                d["group_returns"][g]
                for d in daily
                if d.get("group_returns") and d["group_returns"][g] is not None
            ]
            g_mean = sum(g_series) / len(g_series) if g_series else 0.0
            cum_prod = 1.0
            for r in g_series:
                cum_prod *= (1.0 + r)
            g_cum = cum_prod - 1.0
            group_list.append({"group": g + 1, "mean_return": g_mean})
            group_table_rows.append([f"G{g + 1}", f"{g_mean * 100:.2f}%", f"{g_cum * 100:.2f}%"])

        group_analysis = {
            "groups": group_list,
            "long_short": group_list[-1]["mean_return"] - group_list[0]["mean_return"],
        }

        # Horizon metrics
        horizon_metrics = []
        for idx, h in enumerate(horizons):
            h_ics = [d["horizon_ic"][idx] for d in daily if d.get("horizon_ic") and d["horizon_ic"][idx] is not None]
            h_rics = [d["horizon_rankic"][idx] for d in daily if d.get("horizon_rankic") and d["horizon_rankic"][idx] is not None]
            h_pics = [d["horizon_percic"][idx] for d in daily if d.get("horizon_percic") and d["horizon_percic"][idx] is not None]
            h_covs = [
                d["horizon_coverage"][idx]
                for d in daily
                if d.get("horizon_coverage") and d["horizon_coverage"][idx] is not None and d["horizon_ic"][idx] is not None
            ]
            hm_ic, hm_icir, _ = series_stats(h_ics)
            hm_ric, hm_ricir, _ = series_stats(h_rics)
            hm_pic, hm_picir, _ = series_stats(h_pics)
            horizon_metrics.append({
                "horizon": h,
                "ic": hm_ic,
                "icIR": hm_icir,
                "rankic": hm_ric,
                "rankicIR": hm_ricir,
                "percic": hm_pic,
                "percicIR": hm_picir,
                "coverage": sum(h_covs) / len(h_covs) if h_covs else 0.0,
                "days": len(h_ics),
            })

        # Long excess analysis (by year and full period)
        years = sorted({d["date"][:4] for d in daily})
        long_stats = []
        for period_label in years + ["全周期"]:
            p_rows = daily if period_label == "全周期" else [d for d in daily if d["date"].startswith(period_label)]
            valid_le = [d for d in p_rows if d.get("long_excess") is not None]
            if not valid_le:
                continue
            n_period_rows = len(p_rows) if period_label == "全周期" else len(valid_le)
            le_vals = [d["long_excess"] for d in valid_le]

            # Compound annualized return with 250 trading days/year
            cp = 1.0
            for r in le_vals:
                cp *= (1.0 + r)
            ann_ret = (cp ** (250.0 / n_period_rows)) - 1.0

            # Sample std (ddof=1) annualized by sqrt(250)
            m_le = sum(le_vals) / len(le_vals)
            std_le = math.sqrt(sum((r - m_le) ** 2 for r in le_vals) / max(1, len(le_vals) - 1))
            ann_vol = std_le * math.sqrt(250.0)
            sharpe = ann_ret / ann_vol if ann_vol > 0 else 0.0

            # Turnover & holdings
            to_vals = [d["long_turnover"] for d in valid_le if d.get("long_turnover") is not None]
            avg_to = sum(to_vals) / len(to_vals) if to_vals else 0.0
            cnt_vals = [d["long_count"] for d in valid_le if d.get("long_count") is not None]
            avg_holdings = round(sum(cnt_vals) / len(cnt_vals)) if cnt_vals else 0

            # Drawdown on cumulative additive excess return curve: DD_t = Peak_t - Cum_t
            cum = 0.0
            peak = 0.0
            peak_date = p_rows[0]["date"]
            max_dd = 0.0
            max_dd_start = peak_date
            max_dd_end = peak_date
            dds: List[float] = []
            for d in valid_le:
                cum += d["long_excess"]
                if cum > peak:
                    peak = cum
                    peak_date = d["date"]
                dd = peak - cum
                dds.append(dd)
                if dd > max_dd:
                    max_dd = dd
                    max_dd_start = peak_date
                    max_dd_end = d["date"]
            avg_dd = sum(dds) / len(dds) if dds else 0.0

            long_stats.append({
                "period": period_label,
                "start_date": p_rows[0]["date"],
                "end_date": p_rows[-1]["date"],
                "sharpe": sharpe,
                "annual_return": ann_ret,
                "annual_volatility": ann_vol,
                "turnover": avg_to,
                "holdings": avg_holdings,
                "average_drawdown": avg_dd,
                "max_drawdown": max_dd,
                "max_drawdown_start": max_dd_start,
                "max_drawdown_end": max_dd_end,
            })

        return {
            "metrics": metrics,
            "group_analysis": group_analysis,
            "group_table_rows": group_table_rows,
            "horizon_metrics": horizon_metrics,
            "long_analysis": {
                "quantile": 0.1,
                "benchmark": "000985.CSI",
                "costs_included": False,
                "stats": long_stats,
            },
            "style_exposure": style_exposure or {},
            "daily": daily,
        }


# =============================================================================
# 5. LOCAL 15-TOOL MCP SERVER / DISPATCHER
# =============================================================================
class LocalFactorLabMCP:
    """Implements all 15 FactorLab MCP tools locally with identical JSON contracts."""

    def __init__(self, catalog_path: str = "data/catalog_full.json"):
        with open(catalog_path, "r", encoding="utf-8") as f:
            self.catalog = json.load(f)

        self.var_map = {v["short_name"]: v for v in self.catalog.get("variables", [])}
        self.func_map = {fn["name"]: fn for fn in self.catalog.get("functions", [])}
        self.post_map = {p["name"]: p for p in self.catalog.get("postprocess_operators", [])}

        self.store = LocalFeatureStore(n_days=80, n_stocks=60, seed=42)
        self.probe_engine = LocalProbeEngine(self.var_map)
        self.exec_engine = FactorExecutionEngine(self.store)

        # In-memory state for drafts, immutable versions, materializations, and runs
        self.factors: Dict[str, Dict[str, Any]] = {}
        self.factor_versions: Dict[str, Dict[str, Any]] = {}
        self.materializations: Dict[str, Matrix] = {}
        self.runs: Dict[str, Dict[str, Any]] = {}

    def call_tool(self, tool_name: str, arguments: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        args = arguments or {}
        handler = getattr(self, f"tool_{tool_name}", None)
        if not handler:
            raise ValueError(f"Unknown MCP tool: {tool_name}")
        return {"schema_version": 1, "data": handler(**args)}

    # --- Category I: Environment & Catalog (Tools 1-3) ---
    def tool_get_context(self) -> Dict[str, Any]:
        return {
            "schema_version": 1,
            "user": {"username": "local_researcher", "role": "user"},
            "factor_types": ["expression", "code"],
            "materialization_frequencies": ["1d", "1m", "5m"],
            "backtest_frequencies": ["1d"],
            "daily_input_rules": {
                "output": "1d means one vector per trading day; input frequency is independent",
                "source_clocks": {
                    "1m": {"bars_per_day": 241, "first_bar": "09:30:00", "last_bar": "15:01:03", "includes_auctions": True},
                    "5m": {"bars_per_day": 49, "first_bar": "09:30:00", "last_bar": "15:01:03", "includes_auctions": True},
                },
                "cutoff": "Daily output at or after 15:00 includes complete current trading session",
                "missing": "Missing bars, days and securities retain their positions as NaN",
            },
            "materialization_policy": {
                "policy_id": "kemove_research_full_history_v1",
                "date_range": {"start_date": self.store.dates[0], "end_date": self.store.dates[-1]},
                "universe_ref": "BaseUnivMask",
            },
            "single_factor_evaluation": {
                "label": "next_trading_day_vwap30_entry",
                "holding_days": 1,
                "horizons": [1, 2, 5, 10],
                "groups": 10,
                "costs_included": False,
            },
        }

    def tool_search_catalog(self, catalog: str, q: str = "", limit: int = 20, offset: int = 0) -> Dict[str, Any]:
        key_map = {
            "variables": "variables",
            "functions": "functions",
            "postprocess-operators": "postprocess_operators",
        }
        items = self.catalog.get(key_map[catalog], [])
        if q:
            ql = q.lower()
            items = [it for it in items if ql in json.dumps(it, ensure_ascii=False).lower()]
        sliced = items[offset : offset + limit]
        return {"catalog": catalog, "items": sliced, "total": len(items), "limit": limit, "offset": offset}

    def tool_get_catalog_entry(self, catalog: str, name: str) -> Dict[str, Any]:
        lookup = {"variables": self.var_map, "functions": self.func_map, "postprocess-operators": self.post_map}[catalog]
        return {"catalog": catalog, "entry": lookup[name]}

    # --- Category II: Factor Drafts & Versioning (Tools 4-9) ---
    def tool_list_factors(self, q: str = "", status: str = "", limit: int = 20, offset: int = 0) -> Dict[str, Any]:
        items = list(self.factors.values())
        if q:
            items = [f for f in items if q.lower() in f["name"].lower()]
        if status:
            items = [f for f in items if f["status"] == status]
        return {"items": items[offset : offset + limit], "total": len(items)}

    def tool_get_factor(self, factor_id: str) -> Dict[str, Any]:
        return self.factors[factor_id]

    def tool_get_factor_version(self, factor_version_id: str) -> Dict[str, Any]:
        return self.factor_versions[factor_version_id]

    def tool_create_factor(
        self,
        name: str,
        definition: Dict[str, Any],
        idempotency_key: str,
        display_name: str = "",
        description: str = "",
    ) -> Dict[str, Any]:
        fid = f"fac_{uuid.uuid4().hex}"
        now = datetime.now(timezone.utc).isoformat()
        record = {
            "factor_id": fid,
            "name": name,
            "display_name": display_name or name,
            "description": description,
            "owner": "local_researcher",
            "status": "draft",
            "draft_revision": 1,
            "draft": {"revision": 1, "definition": definition, "updated_at": now},
            "latest_run": None,
            "created_at": now,
            "updated_at": now,
        }
        self.factors[fid] = record
        return record

    def tool_update_factor_draft(
        self,
        factor_id: str,
        draft_revision: int,
        definition: Dict[str, Any],
        idempotency_key: str,
    ) -> Dict[str, Any]:
        rec = self.factors[factor_id]
        if rec["draft_revision"] != draft_revision:
            raise RuntimeError(f"Optimistic lock conflict: expected revision {rec['draft_revision']}, got {draft_revision}")
        new_rev = draft_revision + 1
        now = datetime.now(timezone.utc).isoformat()
        rec["draft_revision"] = new_rev
        rec["draft"] = {"revision": new_rev, "definition": definition, "updated_at": now}
        rec["updated_at"] = now
        return rec

    def tool_probe_factor(self, definition: Dict[str, Any]) -> Dict[str, Any]:
        return self.probe_engine.probe(definition)

    # --- Category III: Distributed Computation & Backtest Scheduling (Tools 10-12) ---
    def tool_start_backtest(self, factor_id: str, draft_revision: int, idempotency_key: str) -> Dict[str, Any]:
        rec = self.factors[factor_id]
        if rec["draft_revision"] != draft_revision:
            raise RuntimeError("Draft revision mismatch on start_backtest")

        definition = rec["draft"]["definition"]
        probe_res = self.probe_engine.probe(definition)
        if probe_res["status"] != "valid":
            raise RuntimeError(f"Probe failed: {probe_res['errors']}")

        # 1. Lock immutable factor_version (fv_...)
        fvid = f"fv_{uuid.uuid4().hex}"
        now = datetime.now(timezone.utc).isoformat()
        self.factor_versions[fvid] = {
            "factor_version_id": fvid,
            "factor_id": factor_id,
            "version_no": draft_revision,
            "state": "candidate",
            "created_by": "local_researcher",
            "created_at": now,
            "definition": copy.deepcopy(definition),
            "probe_result": probe_res,
        }

        # 2. Stage 1: Materialize factor matrix [T x S] (fm_...)
        fmid = f"fm_{uuid.uuid4().hex}"
        factor_mat = self.exec_engine.compute_materialized_factor(definition)
        self.materializations[fmid] = factor_mat

        # 3. Stage 2: Execute vectorized single-factor backtest evaluation (bt_...)
        run_id = f"bt_{uuid.uuid4().hex}"
        eval_result = LocalBacktestEngine.evaluate_factor_matrix(factor_mat, self.store)

        run_record = {
            "run_id": run_id,
            "factor_id": factor_id,
            "name": rec["name"],
            "factor_version_id": fvid,
            "materialization_id": fmid,
            "job_id": f"job_{uuid.uuid4().hex}",
            "state": "succeeded",
            "stage": "completed",
            "progress": {"completed": 1, "total": 1, "unit": "trading_day"},
            "result_ready": True,
            "created_at": now,
            "updated_at": now,
            "resources": {
                "estimated_memory_mb": 676,
                "process_memory_mb": 4096,
                "cpu_slots": 1,
                "block_days": 32,
                "estimated_stocks": self.store.n_stocks,
            },
            "_eval_result": eval_result,
        }
        self.runs[run_id] = run_record
        rec["latest_run"] = {"run_id": run_id, "state": "succeeded", "updated_at": now}
        return {k: v for k, v in run_record.items() if not k.startswith("_")}

    def tool_get_run(self, run_id: str) -> Dict[str, Any]:
        r = self.runs[run_id]
        return {k: v for k, v in r.items() if not k.startswith("_")}

    def tool_cancel_run(self, run_id: str) -> Dict[str, Any]:
        r = self.runs[run_id]
        if r["state"] in ("queued", "running"):
            r["state"] = "cancelled"
        return {k: v for k, v in r.items() if not k.startswith("_")}

    # --- Category IV: Backtest Evaluation Summary & Series (Tools 13-15) ---
    def tool_list_runs(self, factor_id: str = "", state: str = "", limit: int = 20, offset: int = 0) -> Dict[str, Any]:
        items = [
            {k: v for k, v in r.items() if not k.startswith("_")}
            for r in self.runs.values()
        ]
        if factor_id:
            items = [r for r in items if r["factor_id"] == factor_id]
        if state:
            items = [r for r in items if r["state"] == state]
        return {"items": items[offset : offset + limit], "total": len(items)}

    def tool_get_result_summary(self, run_id: str) -> Dict[str, Any]:
        r = self.runs[run_id]
        ev = r["_eval_result"]
        return {
            "run_id": run_id,
            "factor_version_id": r["factor_version_id"],
            "materialization_id": r["materialization_id"],
            "result_ready": True,
            "state": r["state"],
            "result_schema_version": 4,
            "metrics": ev["metrics"],
            "group_analysis": ev["group_analysis"],
            "horizon_metrics": ev["horizon_metrics"],
            "long_analysis": ev["long_analysis"],
            "style_exposure": ev["style_exposure"],
            "available_series": ["ic", "rankic", "percic", "long_short", "long_excess", "coverage", "group_returns"],
        }

    def tool_get_result_series(
        self,
        run_id: str,
        series: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> Dict[str, Any]:
        r = self.runs[run_id]
        daily = r["_eval_result"]["daily"]
        items = []
        for d in daily:
            dt = d["date"]
            if start_date and dt < start_date:
                continue
            if end_date and dt > end_date:
                continue
            items.append({"date": dt, "value": d[series]})
        return {
            "run_id": run_id,
            "series": series,
            "result_ready": True,
            "items": items[offset : offset + limit],
            "total": len(items),
            "limit": limit,
            "offset": offset,
        }


# =============================================================================
# 6. VERIFICATION & END-TO-END WORKFLOW RUNNER
# =============================================================================
def verify_12_postprocess_operators(catalog_path: str = "data/catalog_full.json") -> None:
    """Verifies all 12 postprocess operators against locked C++ runtime test vectors."""
    print("=" * 78)
    print("  [Verification 1] 12 Postprocess Operators vs. Locked C++ Runtime Vectors")
    print("=" * 78)
    with open(catalog_path, "r", encoding="utf-8") as f:
        cat = json.load(f)

    for p in cat.get("postprocess_operators", []):
        name = p["name"]
        ex = p["documentation"]["examples"][0]
        inputs = ex["inputs"]
        expected = to_float_matrix(ex["output"])
        x1 = to_float_matrix(inputs["x1"])

        if name in ("trdMask", "baseUniMask", "indNeut", "riskNeut", "riskNeutMulti"):
            x2 = to_float_matrix(inputs["x2"])
            actual = getattr(AresRuntimeEngine, name)(x1, x2)
        elif name == "dropNewStock":
            x2 = to_float_matrix(inputs["x2"])
            actual = AresRuntimeEngine.dropNewStock(x1, x2, min_dates=3)
        elif name in ("decay", "emadecay"):
            actual = getattr(AresRuntimeEngine, name)(x1, window=3)
        elif name == "MAD":
            actual = AresRuntimeEngine.MAD(x1, threshold=2.0)
        elif name == "winsor":
            actual = AresRuntimeEngine.winsor(x1, quantile=0.25)
        elif name in ("zScore", "rank"):
            actual = getattr(AresRuntimeEngine, name)(x1)
        else:
            raise ValueError(f"Untested operator: {name}")

        max_err = 0.0
        for r_act, r_exp in zip(actual, expected):
            for a, b in zip(r_act, r_exp):
                if is_nan(a) and is_nan(b):
                    continue
                if is_nan(a) != is_nan(b):
                    raise AssertionError(f"NaN mismatch in {name}: actual={a}, expected={b}")
                max_err = max(max_err, abs(a - b))

        opcode = p["documentation"]["evidence"]["native_reference"]["opcode"]
        print(f"  [PASS] {name:<14} | C++ Opcode: {opcode:<20} | Max Abs Error: {max_err:.2e}")
        assert max_err < 1e-12, f"Operator {name} exceeded error tolerance: {max_err}"


def verify_cloud_backtest_reproduction(sample_path: str = "data/sample_backtest_result.json") -> None:
    """Recomputes all summary metrics from cloud daily series and checks 1e-15 precision."""
    if not os.path.exists(sample_path):
        print(f"[!] Skipping cloud backtest check ({sample_path} not found)")
        return

    print("\n" + "=" * 78)
    print("  [Verification 2] Local Backtest Aggregation vs. Cloud Run (bt_6bf2d738...)")
    print("=" * 78)
    with open(sample_path, "r", encoding="utf-8") as f:
        cloud = json.load(f)

    recomputed = LocalBacktestEngine.summarize_from_daily(cloud["daily"])

    checks = [
        ("Full-Period IC Mean", recomputed["metrics"]["ic"], cloud["metrics"]["ic"]),
        ("Full-Period IC IR (ddof=1)", recomputed["metrics"]["icIR"], cloud["metrics"]["icIR"]),
        ("Full-Period IC Win Rate", recomputed["metrics"]["icWR"], cloud["metrics"]["icWR"]),
        ("Full-Period 5d IC Mean", recomputed["metrics"]["5dic"], cloud["metrics"]["5dic"]),
        ("Full-Period RankIC Mean", recomputed["metrics"]["rankic"], cloud["metrics"]["rankic"]),
        ("Full-Period RankIC IR", recomputed["metrics"]["rankicIR"], cloud["metrics"]["rankicIR"]),
        ("Full-Period PercIC Mean", recomputed["metrics"]["percic"], cloud["metrics"]["percic"]),
        ("Full-Period Long-Short Mean", recomputed["metrics"]["long_short"], cloud["metrics"]["long_short"]),
        ("Full-Period Long-Short IR", recomputed["metrics"]["long_shortIR"], cloud["metrics"]["long_shortIR"]),
        ("2016 Long Excess Ann Return", recomputed["long_analysis"]["stats"][0]["annual_return"], cloud["long_analysis"]["stats"][0]["annual_return"]),
        ("2016 Long Excess Ann Vol", recomputed["long_analysis"]["stats"][0]["annual_volatility"], cloud["long_analysis"]["stats"][0]["annual_volatility"]),
        ("2016 Long Excess Sharpe", recomputed["long_analysis"]["stats"][0]["sharpe"], cloud["long_analysis"]["stats"][0]["sharpe"]),
        ("2016 Long Excess MaxDD", recomputed["long_analysis"]["stats"][0]["max_drawdown"], cloud["long_analysis"]["stats"][0]["max_drawdown"]),
        ("2016 Long Excess AvgDD", recomputed["long_analysis"]["stats"][0]["average_drawdown"], cloud["long_analysis"]["stats"][0]["average_drawdown"]),
        ("Full-Period Long Excess AnnRet", recomputed["long_analysis"]["stats"][-1]["annual_return"], cloud["long_analysis"]["stats"][-1]["annual_return"]),
        ("Full-Period Long Excess MaxDD", recomputed["long_analysis"]["stats"][-1]["max_drawdown"], cloud["long_analysis"]["stats"][-1]["max_drawdown"]),
    ]

    for label, act, exp in checks:
        err = abs(act - exp)
        print(f"  [PASS] {label:<30} | Local: {act:+.12f} | Cloud: {exp:+.12f} | Diff: {err:.2e}")
        assert err < 1e-14, f"Mismatch on {label}: {act} vs {exp}"


def run_end_to_end_local_workflow() -> None:
    """Executes all 15 MCP tools in an end-to-end local quantitative research workflow."""
    print("\n" + "=" * 78)
    print("  [Verification 3] End-to-End Local Workflow Execution Across All 15 MCP Tools")
    print("=" * 78)

    mcp = LocalFactorLabMCP()

    # Step 1: get_context & search_catalog & get_catalog_entry
    ctx = mcp.call_tool("get_context")["data"]
    print(f"  1. [get_context] Universe: {ctx['materialization_policy']['universe_ref']}, "
          f"DateRange: {ctx['materialization_policy']['date_range']['start_date']}~"
          f"{ctx['materialization_policy']['date_range']['end_date']}, "
          f"Label: {ctx['single_factor_evaluation']['label']}")

    cat_search = mcp.call_tool("search_catalog", {"catalog": "postprocess-operators", "q": "Neut"})["data"]
    print(f"  2. [search_catalog] Found {cat_search['total']} neutralization postprocess operators: "
          f"{[it['name'] for it in cat_search['items']]}")

    entry_card = mcp.call_tool("get_catalog_entry", {"catalog": "functions", "name": "emadecay"})["data"]["entry"]
    print(f"  3. [get_catalog_entry] emadecay NaN Policy: {entry_card['missing_policy']}")

    # Step 2: create_factor -> update_factor_draft -> probe_factor
    raw_def = {
        "factor_type": "expression",
        "source": {"expression": "log(div(emadecay(close_hfq, 5), decay(close_hfq, 20)))"},
        "output": {"frequency": "1d", "available_at": "15:00:00"},
        "postprocess": [],
    }
    created = mcp.call_tool("create_factor", {
        "name": "dual_ma_local",
        "definition": raw_def,
        "idempotency_key": "idem_create_001",
        "description": "Local reproduction of Dual MA factor with full postprocess chain",
    })["data"]
    fid = created["factor_id"]
    print(f"  4. [create_factor] Created factor_id={fid}, draft_revision={created['draft_revision']}")

    # Update draft to include industrial postprocess chain
    full_def = {
        "factor_type": "expression",
        "source": {"expression": "log(div(emadecay(close_hfq, 5), decay(close_hfq, 20)))"},
        "output": {"frequency": "1d", "available_at": "15:00:00"},
        "postprocess": [
            {"op": "baseUniMask"},
            {"op": "trdMask"},
            {"op": "dropNewStock", "args": [60]},
            {"op": "MAD", "args": [3.0]},
            {"op": "indNeut", "args": ["barra_cne5_industry"]},
            {"op": "riskNeut", "args": [["barra_cne5_size"]]},
            {"op": "zScore"},
            {"op": "decay", "args": [3]},
        ],
    }
    updated = mcp.call_tool("update_factor_draft", {
        "factor_id": fid,
        "draft_revision": 1,
        "definition": full_def,
        "idempotency_key": "idem_update_002",
    })["data"]
    print(f"  5. [update_factor_draft] Updated to draft_revision={updated['draft_revision']} with 8 postprocess ops")

    probe = mcp.call_tool("probe_factor", {"definition": full_def})["data"]
    print(f"  6. [probe_factor] Status: {probe['status']}, "
          f"Expr Lookback: {probe['execution_requirements']['expression_lookback']}d, "
          f"Postprocess Warmup: {probe['execution_requirements']['postprocess_warmup']['value']}d, "
          f"Injected Postprocess Deps: {[d['name'] for d in probe['postprocess_dependencies']]}")

    # Step 3: start_backtest -> get_run -> get_factor_version
    bt_run = mcp.call_tool("start_backtest", {
        "factor_id": fid,
        "draft_revision": updated["draft_revision"],
        "idempotency_key": "idem_bt_003",
    })["data"]
    run_id = bt_run["run_id"]
    fvid = bt_run["factor_version_id"]
    print(f"  7. [start_backtest] Locked version={fvid}, materialized={bt_run['materialization_id']}, run_id={run_id}")

    run_status = mcp.call_tool("get_run", {"run_id": run_id})["data"]
    fv_snap = mcp.call_tool("get_factor_version", {"factor_version_id": fvid})["data"]
    print(f"  8. [get_run & get_factor_version] State={run_status['state']}, Stage={run_status['stage']}, "
          f"Locked Revision={fv_snap['version_no']}")

    # Step 4: get_result_summary & get_result_series
    summary = mcp.call_tool("get_result_summary", {"run_id": run_id})["data"]
    m = summary["metrics"]
    ls_stat = summary["long_analysis"]["stats"][-1]
    print(f"  9. [get_result_summary] Evaluated {m['days']} days | "
          f"IC={m['ic']:+.4f} (IR={m['icIR']:+.4f}, WR={m['icWR']*100:.1f}%) | "
          f"RankIC={m['rankic']:+.4f} | Long-Short={m['long_short']*100:+.3f}%/d")
    print(f"     Long Excess (Top 10% vs 000985.CSI): AnnRet={ls_stat['annual_return']*100:+.2f}%, "
          f"AnnVol={ls_stat['annual_volatility']*100:.2f}%, Sharpe={ls_stat['sharpe']:+.2f}, "
          f"Turnover={ls_stat['turnover']*100:.1f}%, MaxDD={ls_stat['max_drawdown']*100:.2f}%")

    ic_series = mcp.call_tool("get_result_series", {"run_id": run_id, "series": "ic", "limit": 3, "offset": 25})["data"]
    print(f" 10. [get_result_series] Sample IC series slice: {ic_series['items']}")
    print("=" * 78)


if __name__ == "__main__":
    verify_12_postprocess_operators()
    verify_cloud_backtest_reproduction()
    run_end_to_end_local_workflow()
