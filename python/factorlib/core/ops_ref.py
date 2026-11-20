"""68 个向量化算子的纯 NumPy 参考实现 (core/ops_ref.py)。

按照白皮书 16.2 节与 P2 验收标准设计：
- 本模块提供所有 68 个逐元素、时序滚动、横截面及板块分组算子的纯 Python/NumPy 黄金参考实现；
- 用于与 `factorlib_core` Rust 原生计算内核进行基于性质的随机矩阵对账测试（Property-based Testing）；
- 严格遵循相同的缺失值约定：所有非有限浮点数（`NaN`、`+Inf`、`-Inf`）统一视作 `NaN` 缺失值处理。
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np

EPS: float = 1e-15
STD_EPS: float = 1e-12
VAR_EPS: float = 1e-24


def _clean(arr: np.ndarray) -> np.ndarray:
    """将输入二维数组转为 float64，并将 ±Inf 统一清洗为 NaN。"""
    out = np.array(arr, dtype=np.float64, copy=True, order="F")
    out[~np.isfinite(out)] = np.nan
    return out


# ============================================================================
# 1. 一元逐元素算子 (12 个)
# ============================================================================


def neg(x: np.ndarray) -> np.ndarray:
    xc = _clean(x)
    return _clean(-xc)


def abs_(x: np.ndarray) -> np.ndarray:
    xc = _clean(x)
    return _clean(np.abs(xc))


def sign(x: np.ndarray) -> np.ndarray:
    xc = _clean(x)
    out = np.full_like(xc, np.nan)
    mask = np.isfinite(xc)
    vals = xc[mask]
    res = np.zeros_like(vals)
    res[vals > 0.0] = 1.0
    res[vals < 0.0] = -1.0
    out[mask] = res
    return out


def inv(x: np.ndarray) -> np.ndarray:
    xc = _clean(x)
    out = np.full_like(xc, np.nan)
    mask = np.isfinite(xc) & (np.abs(xc) >= EPS)
    out[mask] = 1.0 / xc[mask]
    return _clean(out)


def slog(x: np.ndarray) -> np.ndarray:
    xc = _clean(x)
    out = np.full_like(xc, np.nan)
    mask = np.isfinite(xc)
    v = xc[mask]
    s = np.where(v > 0.0, 1.0, np.where(v < 0.0, -1.0, 0.0))
    out[mask] = s * np.log1p(np.abs(v))
    return _clean(out)


def ssqrt(x: np.ndarray) -> np.ndarray:
    xc = _clean(x)
    out = np.full_like(xc, np.nan)
    mask = np.isfinite(xc)
    v = xc[mask]
    s = np.where(v > 0.0, 1.0, np.where(v < 0.0, -1.0, 0.0))
    out[mask] = s * np.sqrt(np.abs(v))
    return _clean(out)


def square(x: np.ndarray) -> np.ndarray:
    xc = _clean(x)
    with np.errstate(over="ignore", invalid="ignore"):
        return _clean(xc * xc)


def cube(x: np.ndarray) -> np.ndarray:
    xc = _clean(x)
    with np.errstate(over="ignore", invalid="ignore"):
        return _clean(xc * xc * xc)


def sigmoid(x: np.ndarray) -> np.ndarray:
    xc = _clean(x)
    out = np.full_like(xc, np.nan)
    mask = np.isfinite(xc)
    v = xc[mask]
    pos = v >= 0.0
    res = np.empty_like(v)
    with np.errstate(over="ignore", under="ignore"):
        res[pos] = 1.0 / (1.0 + np.exp(-v[pos]))
        ev = np.exp(v[~pos])
        res[~pos] = ev / (1.0 + ev)
    out[mask] = res
    return _clean(out)


def tanh(x: np.ndarray) -> np.ndarray:
    xc = _clean(x)
    return _clean(np.tanh(xc))


def relu(x: np.ndarray) -> np.ndarray:
    xc = _clean(x)
    out = np.full_like(xc, np.nan)
    mask = np.isfinite(xc)
    out[mask] = np.maximum(xc[mask], 0.0)
    return out


def atan(x: np.ndarray) -> np.ndarray:
    xc = _clean(x)
    return _clean(np.arctan(xc))


# ============================================================================
# 2. 二元逐元素算子 (8 个)
# ============================================================================


def add(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    with np.errstate(over="ignore", invalid="ignore"):
        return _clean(_clean(x) + _clean(y))


def sub(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    with np.errstate(over="ignore", invalid="ignore"):
        return _clean(_clean(x) - _clean(y))


def mul(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    with np.errstate(over="ignore", invalid="ignore"):
        return _clean(_clean(x) * _clean(y))


def div(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    xc = _clean(x)
    yc = _clean(y)
    out = np.full_like(xc, np.nan)
    mask = np.isfinite(xc) & np.isfinite(yc) & (np.abs(yc) >= EPS)
    with np.errstate(over="ignore", invalid="ignore"):
        out[mask] = xc[mask] / yc[mask]
    return _clean(out)


def max2(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    xc = _clean(x)
    yc = _clean(y)
    out = np.full_like(xc, np.nan)
    mask = np.isfinite(xc) & np.isfinite(yc)
    out[mask] = np.maximum(xc[mask], yc[mask])
    return out


def min2(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    xc = _clean(x)
    yc = _clean(y)
    out = np.full_like(xc, np.nan)
    mask = np.isfinite(xc) & np.isfinite(yc)
    out[mask] = np.minimum(xc[mask], yc[mask])
    return out


def hypot(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    xc = _clean(x)
    yc = _clean(y)
    with np.errstate(over="ignore", invalid="ignore"):
        return _clean(np.hypot(xc, yc))


def copysign(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    xc = _clean(x)
    yc = _clean(y)
    out = np.full_like(xc, np.nan)
    mask = np.isfinite(xc) & np.isfinite(yc)
    vy = yc[mask]
    s = np.where(vy > 0.0, 1.0, np.where(vy < 0.0, -1.0, 0.0))
    out[mask] = np.abs(xc[mask]) * s
    return _clean(out)


# ============================================================================
# 3. 标量常数算子 (6 个)
# ============================================================================


def addc(x: np.ndarray, c: float) -> np.ndarray:
    xc = _clean(x)
    if not np.isfinite(c):
        return np.full_like(xc, np.nan)
    with np.errstate(over="ignore", invalid="ignore"):
        return _clean(xc + float(c))


def subc(x: np.ndarray, c: float) -> np.ndarray:
    xc = _clean(x)
    if not np.isfinite(c):
        return np.full_like(xc, np.nan)
    with np.errstate(over="ignore", invalid="ignore"):
        return _clean(xc - float(c))


def mulc(x: np.ndarray, c: float) -> np.ndarray:
    xc = _clean(x)
    if not np.isfinite(c):
        return np.full_like(xc, np.nan)
    with np.errstate(over="ignore", invalid="ignore"):
        return _clean(xc * float(c))


def divc(x: np.ndarray, c: float) -> np.ndarray:
    xc = _clean(x)
    if not np.isfinite(c) or abs(float(c)) < EPS:
        return np.full_like(xc, np.nan)
    with np.errstate(over="ignore", invalid="ignore"):
        return _clean(xc / float(c))


def spow(x: np.ndarray, c: float) -> np.ndarray:
    xc = _clean(x)
    if not np.isfinite(c):
        return np.full_like(xc, np.nan)
    cf = float(c)
    out = np.full_like(xc, np.nan)
    T, N = xc.shape
    for j in range(N):
        for t in range(T):
            v = xc[t, j]
            if not np.isfinite(v):
                continue
            av = abs(v)
            if av < EPS and cf < 0.0:
                continue
            if av == 0.0:
                out[t, j] = 1.0 if cf == 0.0 else 0.0
                continue
            s = 1.0 if v > 0.0 else -1.0
            with np.errstate(over="ignore", invalid="ignore"):
                val = s * (av**cf)
            if np.isfinite(val):
                out[t, j] = val
    return out


def clip(x: np.ndarray, c: float) -> np.ndarray:
    xc = _clean(x)
    if not np.isfinite(c):
        return np.full_like(xc, np.nan)
    bound = abs(float(c))
    out = np.full_like(xc, np.nan)
    mask = np.isfinite(xc)
    out[mask] = np.clip(xc[mask], -bound, bound)
    return out


# ============================================================================
# 4. 比较、逻辑与三元条件算子 (10 个)
# ============================================================================


def gt(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    xc, yc = _clean(x), _clean(y)
    out = np.full_like(xc, np.nan)
    m = np.isfinite(xc) & np.isfinite(yc)
    out[m] = np.where(xc[m] > yc[m], 1.0, 0.0)
    return out


def lt(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    xc, yc = _clean(x), _clean(y)
    out = np.full_like(xc, np.nan)
    m = np.isfinite(xc) & np.isfinite(yc)
    out[m] = np.where(xc[m] < yc[m], 1.0, 0.0)
    return out


def ge(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    xc, yc = _clean(x), _clean(y)
    out = np.full_like(xc, np.nan)
    m = np.isfinite(xc) & np.isfinite(yc)
    out[m] = np.where(xc[m] >= yc[m], 1.0, 0.0)
    return out


def le(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    xc, yc = _clean(x), _clean(y)
    out = np.full_like(xc, np.nan)
    m = np.isfinite(xc) & np.isfinite(yc)
    out[m] = np.where(xc[m] <= yc[m], 1.0, 0.0)
    return out


def gt_const(x: np.ndarray, c: float) -> np.ndarray:
    xc = _clean(x)
    if not np.isfinite(c):
        return np.full_like(xc, np.nan)
    out = np.full_like(xc, np.nan)
    m = np.isfinite(xc)
    out[m] = np.where(xc[m] > float(c), 1.0, 0.0)
    return out


def lt_const(x: np.ndarray, c: float) -> np.ndarray:
    xc = _clean(x)
    if not np.isfinite(c):
        return np.full_like(xc, np.nan)
    out = np.full_like(xc, np.nan)
    m = np.isfinite(xc)
    out[m] = np.where(xc[m] < float(c), 1.0, 0.0)
    return out


def and_cond(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    ac, bc = _clean(a), _clean(b)
    out = np.full_like(ac, np.nan)
    m = np.isfinite(ac) & np.isfinite(bc)
    out[m] = np.where((ac[m] > 0.5) & (bc[m] > 0.5), 1.0, 0.0)
    return out


def or_cond(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    ac, bc = _clean(a), _clean(b)
    out = np.full_like(ac, np.nan)
    m = np.isfinite(ac) & np.isfinite(bc)
    out[m] = np.where((ac[m] > 0.5) | (bc[m] > 0.5), 1.0, 0.0)
    return out


def not_cond(a: np.ndarray) -> np.ndarray:
    ac = _clean(a)
    out = np.full_like(ac, np.nan)
    m = np.isfinite(ac)
    out[m] = np.where(ac[m] <= 0.5, 1.0, 0.0)
    return out


def where(cond: np.ndarray, then_b: np.ndarray, else_b: np.ndarray) -> np.ndarray:
    cc = _clean(cond)
    tc = _clean(then_b)
    ec = _clean(else_b)
    out = np.full_like(cc, np.nan)
    m_true = np.isfinite(cc) & (cc > 0.5)
    m_false = np.isfinite(cc) & (cc <= 0.5)
    out[m_true] = tc[m_true]
    out[m_false] = ec[m_false]
    return out


# ============================================================================
# 5. 时序滚动一元算子 (19 个)
# ============================================================================


def delay(x: np.ndarray, d: int) -> np.ndarray:
    xc = _clean(x)
    T, N = xc.shape
    out = np.full((T, N), np.nan, dtype=np.float64, order="F")
    if d == 0:
        return xc
    if d < 0 or d >= T:
        return out
    out[d:, :] = xc[:-d, :]
    return out


def delta(x: np.ndarray, d: int) -> np.ndarray:
    xc = _clean(x)
    T, N = xc.shape
    out = np.full((T, N), np.nan, dtype=np.float64, order="F")
    if d <= 0 or d >= T:
        return out
    with np.errstate(over="ignore", invalid="ignore"):
        out[d:, :] = xc[d:, :] - xc[:-d, :]
    return _clean(out)


def pct_change(x: np.ndarray, d: int) -> np.ndarray:
    xc = _clean(x)
    T, N = xc.shape
    out = np.full((T, N), np.nan, dtype=np.float64, order="F")
    if d <= 0 or d >= T:
        return out
    cur = xc[d:, :]
    prev = xc[:-d, :]
    mask = np.isfinite(cur) & np.isfinite(prev) & (np.abs(prev) >= EPS)
    sub_out = np.full_like(cur, np.nan)
    with np.errstate(over="ignore", invalid="ignore"):
        sub_out[mask] = (cur[mask] - prev[mask]) / prev[mask]
    out[d:, :] = sub_out
    return _clean(out)


def log_return(x: np.ndarray, d: int) -> np.ndarray:
    xc = _clean(x)
    T, N = xc.shape
    out = np.full((T, N), np.nan, dtype=np.float64, order="F")
    if d <= 0 or d >= T:
        return out
    cur = xc[d:, :]
    prev = xc[:-d, :]
    mask = np.isfinite(cur) & np.isfinite(prev) & (cur > EPS) & (prev > EPS)
    sub_out = np.full_like(cur, np.nan)
    sub_out[mask] = np.log(cur[mask] / prev[mask])
    out[d:, :] = sub_out
    return _clean(out)


def ts_sum(x: np.ndarray, d: int) -> np.ndarray:
    xc = _clean(x)
    T, N = xc.shape
    out = np.full((T, N), np.nan, dtype=np.float64, order="F")
    if d <= 0 or d > T:
        return out
    for j in range(N):
        col = xc[:, j]
        for t in range(d - 1, T):
            win = col[t + 1 - d : t + 1]
            valid = win[np.isfinite(win)]
            if len(valid) >= 1:
                val = float(np.sum(valid))
                if np.isfinite(val):
                    out[t, j] = val
    return out


def ts_mean(x: np.ndarray, d: int) -> np.ndarray:
    xc = _clean(x)
    T, N = xc.shape
    out = np.full((T, N), np.nan, dtype=np.float64, order="F")
    if d <= 0 or d > T:
        return out
    for j in range(N):
        col = xc[:, j]
        for t in range(d - 1, T):
            win = col[t + 1 - d : t + 1]
            valid = win[np.isfinite(win)]
            if len(valid) >= 1:
                val = float(np.mean(valid))
                if np.isfinite(val):
                    out[t, j] = val
    return out


def ts_var(x: np.ndarray, d: int) -> np.ndarray:
    xc = _clean(x)
    T, N = xc.shape
    out = np.full((T, N), np.nan, dtype=np.float64, order="F")
    if d < 2 or d > T:
        return out
    for j in range(N):
        col = xc[:, j]
        for t in range(d - 1, T):
            win = col[t + 1 - d : t + 1]
            valid = win[np.isfinite(win)]
            k = len(valid)
            if k >= 2:
                mean = float(np.mean(valid))
                if not np.isfinite(mean):
                    continue
                diff = valid - mean
                m2 = float(np.sum(diff * diff))
                scale = max(1.0, abs(mean))
                if m2 <= VAR_EPS * k * scale * scale:
                    m2 = 0.0
                v = m2 / (k - 1)
                if np.isfinite(v):
                    out[t, j] = v
    return out


def ts_std(x: np.ndarray, d: int) -> np.ndarray:
    v = ts_var(x, d)
    out = np.full_like(v, np.nan)
    m = np.isfinite(v)
    out[m] = np.sqrt(v[m])
    return out


def ts_zscore(x: np.ndarray, d: int) -> np.ndarray:
    xc = _clean(x)
    T, N = xc.shape
    out = np.full((T, N), np.nan, dtype=np.float64, order="F")
    if d < 2 or d > T:
        return out
    for j in range(N):
        col = xc[:, j]
        for t in range(d - 1, T):
            cur = col[t]
            if not np.isfinite(cur):
                continue
            win = col[t + 1 - d : t + 1]
            valid = win[np.isfinite(win)]
            k = len(valid)
            if k >= 2:
                mean = float(np.mean(valid))
                if not np.isfinite(mean):
                    continue
                diff = valid - mean
                m2 = float(np.sum(diff * diff))
                scale = max(1.0, abs(mean))
                if m2 <= VAR_EPS * k * scale * scale:
                    continue
                std = float(np.sqrt(m2 / (k - 1)))
                if np.isfinite(std) and std > STD_EPS * scale:
                    z = (cur - mean) / std
                    if np.isfinite(z):
                        out[t, j] = z
    return out


def ts_min(x: np.ndarray, d: int) -> np.ndarray:
    xc = _clean(x)
    T, N = xc.shape
    out = np.full((T, N), np.nan, dtype=np.float64, order="F")
    if d <= 0 or d > T:
        return out
    for j in range(N):
        col = xc[:, j]
        for t in range(d - 1, T):
            win = col[t + 1 - d : t + 1]
            valid = win[np.isfinite(win)]
            if len(valid) >= 1:
                out[t, j] = float(np.min(valid))
    return out


def ts_max(x: np.ndarray, d: int) -> np.ndarray:
    xc = _clean(x)
    T, N = xc.shape
    out = np.full((T, N), np.nan, dtype=np.float64, order="F")
    if d <= 0 or d > T:
        return out
    for j in range(N):
        col = xc[:, j]
        for t in range(d - 1, T):
            win = col[t + 1 - d : t + 1]
            valid = win[np.isfinite(win)]
            if len(valid) >= 1:
                out[t, j] = float(np.max(valid))
    return out


def ts_argmin(x: np.ndarray, d: int) -> np.ndarray:
    xc = _clean(x)
    T, N = xc.shape
    out = np.full((T, N), np.nan, dtype=np.float64, order="F")
    if d <= 0 or d > T:
        return out
    for j in range(N):
        col = xc[:, j]
        for t in range(d - 1, T):
            win = col[t + 1 - d : t + 1]
            if np.any(np.isfinite(win)):
                out[t, j] = float(np.nanargmin(win))
    return out


def ts_argmax(x: np.ndarray, d: int) -> np.ndarray:
    xc = _clean(x)
    T, N = xc.shape
    out = np.full((T, N), np.nan, dtype=np.float64, order="F")
    if d <= 0 or d > T:
        return out
    for j in range(N):
        col = xc[:, j]
        for t in range(d - 1, T):
            win = col[t + 1 - d : t + 1]
            if np.any(np.isfinite(win)):
                out[t, j] = float(np.nanargmax(win))
    return out


def ts_rank(x: np.ndarray, d: int) -> np.ndarray:
    xc = _clean(x)
    T, N = xc.shape
    out = np.full((T, N), np.nan, dtype=np.float64, order="F")
    if d <= 0 or d > T:
        return out
    for j in range(N):
        col = xc[:, j]
        for t in range(d - 1, T):
            cur = col[t]
            if not np.isfinite(cur):
                continue
            win = col[t + 1 - d : t + 1]
            valid = win[np.isfinite(win)]
            k = len(valid)
            if k == 1:
                out[t, j] = 0.5
            elif k >= 2:
                tol = STD_EPS * max(1.0, abs(float(cur)))
                c_less = float(np.sum(valid < cur - tol))
                c_eq = max(1.0, float(np.sum(np.abs(valid - cur) <= tol)))
                out[t, j] = (c_less + 0.5 * (c_eq - 1.0)) / (k - 1.0)
    return out


def ts_skew(x: np.ndarray, d: int) -> np.ndarray:
    xc = _clean(x)
    T, N = xc.shape
    out = np.full((T, N), np.nan, dtype=np.float64, order="F")
    if d < 3 or d > T:
        return out
    for j in range(N):
        col = xc[:, j]
        for t in range(d - 1, T):
            win = col[t + 1 - d : t + 1]
            valid = win[np.isfinite(win)]
            k = len(valid)
            if k >= 3:
                mean = float(np.mean(valid))
                if not np.isfinite(mean):
                    continue
                scale = max(1.0, abs(mean))
                diff = valid - mean
                m2 = float(np.mean(diff**2))
                m3 = float(np.mean(diff**3))
                if np.isfinite(m2) and np.isfinite(m3) and m2 > VAR_EPS * scale * scale:
                    val = m3 / (m2**1.5)
                    if np.isfinite(val):
                        out[t, j] = val
    return out


def ts_kurt(x: np.ndarray, d: int) -> np.ndarray:
    xc = _clean(x)
    T, N = xc.shape
    out = np.full((T, N), np.nan, dtype=np.float64, order="F")
    if d < 4 or d > T:
        return out
    for j in range(N):
        col = xc[:, j]
        for t in range(d - 1, T):
            win = col[t + 1 - d : t + 1]
            valid = win[np.isfinite(win)]
            k = len(valid)
            if k >= 4:
                mean = float(np.mean(valid))
                if not np.isfinite(mean):
                    continue
                scale = max(1.0, abs(mean))
                diff = valid - mean
                m2 = float(np.mean(diff**2))
                m4 = float(np.mean(diff**4))
                if np.isfinite(m2) and np.isfinite(m4) and m2 > VAR_EPS * scale * scale:
                    val = m4 / (m2 * m2) - 3.0
                    if np.isfinite(val):
                        out[t, j] = val
    return out


def ts_ema(x: np.ndarray, d: int) -> np.ndarray:
    xc = _clean(x)
    T, N = xc.shape
    out = np.full((T, N), np.nan, dtype=np.float64, order="F")
    if d <= 0 or T == 0:
        return out
    alpha = 2.0 / (d + 1.0)
    one_minus_alpha = 1.0 - alpha
    for j in range(N):
        ema = np.nan
        for t in range(T):
            v = xc[t, j]
            if np.isfinite(v):
                if np.isnan(ema):
                    ema = v
                else:
                    ema = alpha * v + one_minus_alpha * ema
                    if not np.isfinite(ema):
                        ema = np.nan
                out[t, j] = ema
            else:
                out[t, j] = np.nan
    return out


def ts_decay_linear(x: np.ndarray, d: int) -> np.ndarray:
    xc = _clean(x)
    T, N = xc.shape
    out = np.full((T, N), np.nan, dtype=np.float64, order="F")
    if d <= 0 or d > T:
        return out
    weights = np.arange(1, d + 1, dtype=np.float64)
    for j in range(N):
        col = xc[:, j]
        for t in range(d - 1, T):
            win = col[t + 1 - d : t + 1]
            m = np.isfinite(win)
            if np.any(m):
                w_sum = float(np.sum(weights[m]))
                val_sum = float(np.sum(weights[m] * win[m]))
                if w_sum > 0.0 and np.isfinite(val_sum):
                    res = val_sum / w_sum
                    if np.isfinite(res):
                        out[t, j] = res
    return out


def ts_mad(x: np.ndarray, d: int) -> np.ndarray:
    xc = _clean(x)
    T, N = xc.shape
    out = np.full((T, N), np.nan, dtype=np.float64, order="F")
    if d <= 0 or d > T:
        return out
    for j in range(N):
        col = xc[:, j]
        for t in range(d - 1, T):
            win = col[t + 1 - d : t + 1]
            valid = win[np.isfinite(win)]
            if len(valid) >= 1:
                mean = float(np.mean(valid))
                if np.isfinite(mean):
                    val = float(np.mean(np.abs(valid - mean)))
                    if np.isfinite(val):
                        out[t, j] = val
    return out


# ============================================================================
# 6. 时序滚动二元算子 (4 个)
# ============================================================================


def ts_cov(x: np.ndarray, y: np.ndarray, d: int) -> np.ndarray:
    xc, yc = _clean(x), _clean(y)
    T, N = xc.shape
    out = np.full((T, N), np.nan, dtype=np.float64, order="F")
    if d < 2 or d > T:
        return out
    for j in range(N):
        cx, cy = xc[:, j], yc[:, j]
        for t in range(d - 1, T):
            wx = cx[t + 1 - d : t + 1]
            wy = cy[t + 1 - d : t + 1]
            m = np.isfinite(wx) & np.isfinite(wy)
            k = int(np.sum(m))
            if k >= 2:
                vx, vy = wx[m], wy[m]
                mx, my = float(np.mean(vx)), float(np.mean(vy))
                if np.isfinite(mx) and np.isfinite(my):
                    c_xy = float(np.sum((vx - mx) * (vy - my)))
                    if np.isfinite(c_xy):
                        val = c_xy / (k - 1)
                        if np.isfinite(val):
                            out[t, j] = val
    return out


def ts_corr(x: np.ndarray, y: np.ndarray, d: int) -> np.ndarray:
    xc, yc = _clean(x), _clean(y)
    T, N = xc.shape
    out = np.full((T, N), np.nan, dtype=np.float64, order="F")
    if d < 2 or d > T:
        return out
    for j in range(N):
        cx, cy = xc[:, j], yc[:, j]
        for t in range(d - 1, T):
            wx = cx[t + 1 - d : t + 1]
            wy = cy[t + 1 - d : t + 1]
            m = np.isfinite(wx) & np.isfinite(wy)
            k = int(np.sum(m))
            if k >= 2:
                vx, vy = wx[m], wy[m]
                mx, my = float(np.mean(vx)), float(np.mean(vy))
                if not (np.isfinite(mx) and np.isfinite(my)):
                    continue
                dx = vx - mx
                dy = vy - my
                m2_x = float(np.sum(dx * dx))
                m2_y = float(np.sum(dy * dy))
                c_xy = float(np.sum(dx * dy))
                sx = max(1.0, abs(mx))
                sy = max(1.0, abs(my))
                if (
                    np.isfinite(m2_x)
                    and np.isfinite(m2_y)
                    and np.isfinite(c_xy)
                    and m2_x > VAR_EPS * k * sx * sx
                    and m2_y > VAR_EPS * k * sy * sy
                ):
                    denom = float(np.sqrt(m2_x * m2_y))
                    if denom > 0.0 and np.isfinite(denom):
                        out[t, j] = float(np.clip(c_xy / denom, -1.0, 1.0))
    return out


def ts_beta(x: np.ndarray, y: np.ndarray, d: int) -> np.ndarray:
    xc, yc = _clean(x), _clean(y)
    T, N = xc.shape
    out = np.full((T, N), np.nan, dtype=np.float64, order="F")
    if d < 2 or d > T:
        return out
    for j in range(N):
        cx, cy = xc[:, j], yc[:, j]
        for t in range(d - 1, T):
            wx = cx[t + 1 - d : t + 1]
            wy = cy[t + 1 - d : t + 1]
            m = np.isfinite(wx) & np.isfinite(wy)
            k = int(np.sum(m))
            if k >= 2:
                vx, vy = wx[m], wy[m]
                mx, my = float(np.mean(vx)), float(np.mean(vy))
                if not (np.isfinite(mx) and np.isfinite(my)):
                    continue
                dx = vx - mx
                dy = vy - my
                m2_y = float(np.sum(dy * dy))
                c_xy = float(np.sum(dx * dy))
                sy = max(1.0, abs(my))
                if np.isfinite(m2_y) and np.isfinite(c_xy) and m2_y > VAR_EPS * k * sy * sy:
                    val = c_xy / m2_y
                    if np.isfinite(val):
                        out[t, j] = val
    return out


def ts_resid(x: np.ndarray, y: np.ndarray, d: int) -> np.ndarray:
    xc, yc = _clean(x), _clean(y)
    T, N = xc.shape
    out = np.full((T, N), np.nan, dtype=np.float64, order="F")
    if d < 2 or d > T:
        return out
    for j in range(N):
        cx, cy = xc[:, j], yc[:, j]
        for t in range(d - 1, T):
            cur_x, cur_y = cx[t], cy[t]
            if not (np.isfinite(cur_x) and np.isfinite(cur_y)):
                continue
            wx = cx[t + 1 - d : t + 1]
            wy = cy[t + 1 - d : t + 1]
            m = np.isfinite(wx) & np.isfinite(wy)
            k = int(np.sum(m))
            if k >= 2:
                vx, vy = wx[m], wy[m]
                mx, my = float(np.mean(vx)), float(np.mean(vy))
                if not (np.isfinite(mx) and np.isfinite(my)):
                    continue
                dx = vx - mx
                dy = vy - my
                m2_y = float(np.sum(dy * dy))
                c_xy = float(np.sum(dx * dy))
                sy = max(1.0, abs(my))
                if np.isfinite(m2_y) and np.isfinite(c_xy) and m2_y > VAR_EPS * k * sy * sy:
                    beta = c_xy / m2_y
                    resid = (cur_x - mx) - beta * (cur_y - my)
                    if np.isfinite(resid):
                        out[t, j] = resid
    return out


# ============================================================================
# 7. 横截面与行业分组算子 (9 个)
# ============================================================================


def _rank_1d_normalized(row: np.ndarray) -> np.ndarray:
    out = np.full_like(row, np.nan, dtype=np.float64)
    m = np.isfinite(row)
    idx = np.where(m)[0]
    k = len(idx)
    if k == 0:
        return out
    if k == 1:
        out[idx[0]] = 0.5
        return out
    # 严格对齐 Rust compute_normalized_ranks 的左锚点分段排序算法
    order = np.argsort(row[idx], kind="stable")
    sorted_idx = idx[order]
    sorted_vals = row[sorted_idx]
    denom = float(k - 1)
    i = 0
    while i < k:
        vi = float(sorted_vals[i])
        tol = STD_EPS * max(1.0, abs(vi))
        j = i + 1
        while j < k and abs(float(sorted_vals[j]) - vi) <= tol:
            j += 1
        avg_rank = 0.5 * float(i + j - 1) / denom
        for p in range(i, j):
            out[sorted_idx[p]] = avg_rank
        i = j
    return out


def cs_rank(x: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
    if mask is not None:
        x = np.where(mask, x, np.nan)
    xc = _clean(x)
    T, N = xc.shape
    out = np.full((T, N), np.nan, dtype=np.float64, order="F")
    for t in range(T):
        out[t, :] = _rank_1d_normalized(xc[t, :])
    return out


def cs_demean(x: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
    if mask is not None:
        x = np.where(mask, x, np.nan)
    xc = _clean(x)
    T, N = xc.shape
    out = np.full((T, N), np.nan, dtype=np.float64, order="F")
    for t in range(T):
        row = xc[t, :]
        m = np.isfinite(row)
        if np.any(m):
            mean = float(np.mean(row[m]))
            if np.isfinite(mean):
                out[t, m] = row[m] - mean
    return _clean(out)


def cs_zscore(x: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
    if mask is not None:
        x = np.where(mask, x, np.nan)
    xc = _clean(x)
    T, N = xc.shape
    out = np.full((T, N), np.nan, dtype=np.float64, order="F")
    for t in range(T):
        row = xc[t, :]
        m = np.isfinite(row)
        k = int(np.sum(m))
        if k >= 2:
            valid = row[m]
            mean = float(np.mean(valid))
            if not np.isfinite(mean):
                continue
            diff = valid - mean
            std = float(np.sqrt(np.sum(diff * diff) / (k - 1)))
            scale = max(1.0, abs(mean))
            if np.isfinite(std) and std > STD_EPS * scale:
                out[t, m] = (valid - mean) / std
    return _clean(out)


def cs_scale(x: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
    if mask is not None:
        x = np.where(mask, x, np.nan)
    xc = _clean(x)
    T, N = xc.shape
    out = np.full((T, N), np.nan, dtype=np.float64, order="F")
    for t in range(T):
        row = xc[t, :]
        m = np.isfinite(row)
        if np.any(m):
            abs_sum = float(np.sum(np.abs(row[m])))
            if np.isfinite(abs_sum) and abs_sum >= EPS:
                out[t, m] = row[m] / abs_sum
    return _clean(out)


def cs_winsorize(x: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
    if mask is not None:
        x = np.where(mask, x, np.nan)
    xc = _clean(x)
    T, N = xc.shape
    out = np.full((T, N), np.nan, dtype=np.float64, order="F")
    for t in range(T):
        row = xc[t, :]
        m = np.isfinite(row)
        if np.any(m):
            valid = row[m]
            med = float(np.median(valid))
            mad = float(np.median(np.abs(valid - med)))
            radius = 5.0 * 1.4826 * mad
            lo = med - radius
            hi = med + radius
            if np.isfinite(lo) and np.isfinite(hi):
                out[t, m] = np.clip(valid, lo, hi)
    return _clean(out)


def cs_neutralize(y: np.ndarray, x: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
    if mask is not None:
        y = np.where(mask, y, np.nan)
        x = np.where(mask, x, np.nan)
    yc, xc = _clean(y), _clean(x)
    T, N = yc.shape
    out = np.full((T, N), np.nan, dtype=np.float64, order="F")
    for t in range(T):
        rx, ry = xc[t, :], yc[t, :]
        m = np.isfinite(rx) & np.isfinite(ry)
        k = int(np.sum(m))
        if k >= 2:
            vx, vy = rx[m], ry[m]
            mx, my = float(np.mean(vx)), float(np.mean(vy))
            if not (np.isfinite(mx) and np.isfinite(my)):
                continue
            dx = vx - mx
            dy = vy - my
            m2_x = float(np.sum(dx * dx))
            c_xy = float(np.sum(dx * dy))
            sx = max(1.0, abs(mx))
            if np.isfinite(m2_x) and np.isfinite(c_xy) and m2_x > VAR_EPS * k * sx * sx:
                beta = c_xy / m2_x
                out[t, m] = (vy - my) - beta * dx
    return _clean(out)


def group_demean(x: np.ndarray, group: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
    if mask is not None:
        x = np.where(mask, x, np.nan)
        group = np.where(mask, group, np.nan)
    xc, gc = _clean(x), _clean(group)
    T, N = xc.shape
    out = np.full((T, N), np.nan, dtype=np.float64, order="F")
    for t in range(T):
        rx, rg = xc[t, :], gc[t, :]
        m = np.isfinite(rx) & np.isfinite(rg)
        if not np.any(m):
            continue
        gids = np.round(rg[m]).astype(np.int64)
        for gid in np.unique(gids):
            gm = np.isfinite(rx) & np.isfinite(rg) & (np.round(rg) == gid)
            mean = float(np.mean(rx[gm]))
            if np.isfinite(mean):
                out[t, gm] = rx[gm] - mean
    return _clean(out)


def group_rank(x: np.ndarray, group: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
    if mask is not None:
        x = np.where(mask, x, np.nan)
        group = np.where(mask, group, np.nan)
    xc, gc = _clean(x), _clean(group)
    T, N = xc.shape
    out = np.full((T, N), np.nan, dtype=np.float64, order="F")
    for t in range(T):
        rx, rg = xc[t, :], gc[t, :]
        m = np.isfinite(rx) & np.isfinite(rg)
        if not np.any(m):
            continue
        gids = np.round(rg[m]).astype(np.int64)
        for gid in np.unique(gids):
            gm = np.isfinite(rx) & np.isfinite(rg) & (np.round(rg) == gid)
            idx = np.where(gm)[0]
            out[t, idx] = _rank_1d_normalized(rx[idx])
    return out


def group_zscore(x: np.ndarray, group: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
    if mask is not None:
        x = np.where(mask, x, np.nan)
        group = np.where(mask, group, np.nan)
    xc, gc = _clean(x), _clean(group)
    T, N = xc.shape
    out = np.full((T, N), np.nan, dtype=np.float64, order="F")
    for t in range(T):
        rx, rg = xc[t, :], gc[t, :]
        m = np.isfinite(rx) & np.isfinite(rg)
        if not np.any(m):
            continue
        gids = np.round(rg[m]).astype(np.int64)
        for gid in np.unique(gids):
            gm = np.isfinite(rx) & np.isfinite(rg) & (np.round(rg) == gid)
            vals = rx[gm]
            k = len(vals)
            if k >= 2:
                mean = float(np.mean(vals))
                if not np.isfinite(mean):
                    continue
                diff = vals - mean
                std = float(np.sqrt(np.sum(diff * diff) / (k - 1)))
                scale = max(1.0, abs(mean))
                if np.isfinite(std) and std > STD_EPS * scale:
                    out[t, gm] = (vals - mean) / std
    return _clean(out)


# 算子名称到参考实现函数的映射字典（共 68 个算子）
OPS_REF_REGISTRY: dict[str, Callable[..., np.ndarray]] = {
    # 一元逐元素 (12)
    "neg": neg,
    "abs": abs_,
    "sign": sign,
    "inv": inv,
    "slog": slog,
    "ssqrt": ssqrt,
    "square": square,
    "cube": cube,
    "sigmoid": sigmoid,
    "tanh": tanh,
    "relu": relu,
    "atan": atan,
    # 二元逐元素 (8)
    "add": add,
    "sub": sub,
    "mul": mul,
    "div": div,
    "max2": max2,
    "min2": min2,
    "hypot": hypot,
    "copysign": copysign,
    # 标量常数 (6)
    "addc": addc,
    "subc": subc,
    "mulc": mulc,
    "divc": divc,
    "spow": spow,
    "clip": clip,
    # 比较、逻辑与三元条件 (10)
    "gt": gt,
    "lt": lt,
    "ge": ge,
    "le": le,
    "gt_const": gt_const,
    "lt_const": lt_const,
    "and_cond": and_cond,
    "or_cond": or_cond,
    "not_cond": not_cond,
    "where": where,
    # 时序滚动一元 (19)
    "delay": delay,
    "delta": delta,
    "pct_change": pct_change,
    "log_return": log_return,
    "ts_sum": ts_sum,
    "ts_mean": ts_mean,
    "ts_std": ts_std,
    "ts_var": ts_var,
    "ts_min": ts_min,
    "ts_max": ts_max,
    "ts_argmin": ts_argmin,
    "ts_argmax": ts_argmax,
    "ts_rank": ts_rank,
    "ts_zscore": ts_zscore,
    "ts_skew": ts_skew,
    "ts_kurt": ts_kurt,
    "ts_ema": ts_ema,
    "ts_decay_linear": ts_decay_linear,
    "ts_mad": ts_mad,
    # 时序滚动二元 (4)
    "ts_cov": ts_cov,
    "ts_corr": ts_corr,
    "ts_beta": ts_beta,
    "ts_resid": ts_resid,
    # 横截面与分组 (9)
    "cs_rank": cs_rank,
    "cs_zscore": cs_zscore,
    "cs_demean": cs_demean,
    "cs_scale": cs_scale,
    "cs_winsorize": cs_winsorize,
    "cs_neutralize": cs_neutralize,
    "group_demean": group_demean,
    "group_rank": group_rank,
    "group_zscore": group_zscore,
}


# ==============================================================================
# 白皮书 P3 评估指标库纯 NumPy 高精度参考实现
# ==============================================================================


def ref_pearson_correlation(x: np.ndarray, y: np.ndarray) -> float:
    """纯 NumPy Pearson 相关系数参考实现。"""
    n = len(x)
    if n < 2 or n != len(y):
        return float(np.nan)
    mx = float(np.mean(x))
    my = float(np.mean(y))
    dx = x - mx
    dy = y - my
    cov = float(np.sum(dx * dy))
    var_x = float(np.sum(dx * dx))
    var_y = float(np.sum(dy * dy))
    if var_x <= 1e-14 or var_y <= 1e-14:
        return float(np.nan)
    r = cov / np.sqrt(var_x * var_y)
    return float(np.clip(r, -1.0, 1.0))


def ref_fractional_ranks(vals: np.ndarray) -> np.ndarray:
    """纯 NumPy 并列平均排名 (1-based 平均秩) 参考实现。"""
    k = len(vals)
    if k == 0:
        return np.empty(0, dtype=np.float64)
    if k == 1:
        return np.array([1.0], dtype=np.float64)
    order = np.argsort(vals, kind="mergesort")
    sorted_vals = vals[order]
    ranks = np.zeros(k, dtype=np.float64)
    i = 0
    while i < k:
        vi = sorted_vals[i]
        tol = STD_EPS * max(1.0, abs(float(vi)))
        j = i + 1
        while j < k and abs(sorted_vals[j] - vi) <= tol:
            j += 1
        avg_rank = 0.5 * (i + 1 + j)
        for idx in range(i, j):
            ranks[order[idx]] = avg_rank
        i = j
    return ranks


def ref_rank_ic(
    factor: np.ndarray,
    label: np.ndarray,
    mask: np.ndarray,
    min_names: int = 5,
    rank: bool = True,
    horizon: int = 1,
) -> dict[str, Any]:
    """截面 IC 纯 NumPy 参考实现。"""
    T, N = factor.shape
    daily_ic = np.full(T, np.nan, dtype=np.float64)

    valid_f = 0
    total_univ = 0

    for t in range(T):
        m_row = mask[t]
        f_row = factor[t]
        l_row = label[t]

        valid_idx = np.where(m_row & np.isfinite(f_row) & np.isfinite(l_row))[0]
        total_univ += int(np.sum(m_row))
        valid_f += int(np.sum(m_row & np.isfinite(f_row)))

        if len(valid_idx) < min_names or len(valid_idx) < 2:
            continue

        x_vals = f_row[valid_idx]
        y_vals = l_row[valid_idx]

        if rank:
            rx = ref_fractional_ranks(x_vals)
            ry = ref_fractional_ranks(y_vals)
            daily_ic[t] = ref_pearson_correlation(rx, ry)
        else:
            daily_ic[t] = ref_pearson_correlation(x_vals, y_vals)

    valid_ic = daily_ic[np.isfinite(daily_ic)]
    m = len(valid_ic)
    coverage = float(valid_f / total_univ) if total_univ > 0 else (m / T if T > 0 else 0.0)

    if m == 0:
        return {
            "ic_mean": 0.0,
            "ic_std": 0.0,
            "ic_ir": 0.0,
            "ic_t": 0.0,
            "ic_win_rate": 0.0,
            "coverage": coverage,
            "valid_days": 0,
            "daily_ic": daily_ic,
        }

    mean = float(np.mean(valid_ic))
    std = float(np.std(valid_ic, ddof=1)) if m > 1 else 0.0
    ir = float(mean / std) if std > 1e-12 else 0.0
    h = max(horizon, 1)
    t_stat = ir * np.sqrt(m / h)

    if mean > 0:
        wins = int(np.sum(valid_ic > 0))
    elif mean < 0:
        wins = int(np.sum(valid_ic < 0))
    else:
        wins = int(np.sum(valid_ic == 0))
    win_rate = float(wins / m)

    return {
        "ic_mean": mean,
        "ic_std": std,
        "ic_ir": ir,
        "ic_t": float(t_stat),
        "ic_win_rate": win_rate,
        "coverage": coverage,
        "valid_days": m,
        "daily_ic": daily_ic,
    }


def ref_one_way_weight_turnover(prev_indices: list[int], curr_indices: list[int]) -> float:
    """计算两组标的集合之间的单边等权持仓权重换手率: 0.5 * sum |w_curr - w_prev|"""
    if len(prev_indices) == 0 or len(curr_indices) == 0:
        return float(np.nan)
    s_prev = set(prev_indices)
    s_curr = set(curr_indices)
    kp = float(len(s_prev))
    kc = float(len(s_curr))
    o = float(len(s_prev & s_curr))
    sum_diff = o * abs(1.0 / kc - 1.0 / kp) + (1.0 - o / kc) + (1.0 - o / kp)
    return float(np.clip(0.5 * sum_diff, 0.0, 1.0))


def ref_quantile_returns(
    factor: np.ndarray,
    forward_ret: np.ndarray,
    mask: np.ndarray,
    n_quantiles: int = 5,
    min_names: int = 5,
    sign: int = 1,
) -> dict[str, Any]:
    """分位数分组与多空收益纯 NumPy 参考实现。"""
    T, N = factor.shape
    group_daily_returns = [np.full(T, np.nan, dtype=np.float64) for _ in range(n_quantiles)]
    ls_daily_returns = np.full(T, np.nan, dtype=np.float64)
    daily_top_list: list[list[int]] = []
    top_q = (n_quantiles - 1) if sign >= 0 else 0

    for t in range(T):
        m_row = mask[t]
        f_row = factor[t]
        r_row = forward_ret[t]

        valid_idx = np.where(m_row & np.isfinite(f_row) & np.isfinite(r_row))[0]
        k_names = len(valid_idx)
        if k_names < min_names or k_names < n_quantiles or n_quantiles < 2:
            daily_top_list.append([])
            continue

        # 稳定排序: 因子值升序，相同时按标的下标排序
        sub_f = f_row[valid_idx]
        sorted_pos = np.argsort(sub_f, kind="mergesort")
        sorted_idx = valid_idx[sorted_pos]

        g_rets = []
        top_indices = []
        for q in range(n_quantiles):
            i_start = int(np.round(q * k_names / n_quantiles))
            i_end = int(np.round((q + 1) * k_names / n_quantiles))
            i_end = min(i_end, k_names)
            q_idx = sorted_idx[i_start:i_end]
            if len(q_idx) == 0:
                g_rets.append(np.nan)
            else:
                g_rets.append(float(np.mean(r_row[q_idx])))
            group_daily_returns[q][t] = g_rets[-1]
            if q == top_q:
                top_indices = sorted(q_idx.tolist())
        daily_top_list.append(top_indices)

        r_bottom = g_rets[0]
        r_top = g_rets[-1]
        if np.isfinite(r_bottom) and np.isfinite(r_top):
            ls_daily_returns[t] = (r_top - r_bottom) if sign >= 0 else (r_bottom - r_top)

    top_turnover_series = np.full(T, np.nan, dtype=np.float64)
    for t in range(1, T):
        prev_top = daily_top_list[t - 1]
        curr_top = daily_top_list[t]
        if len(prev_top) > 0 and len(curr_top) > 0:
            top_turnover_series[t] = ref_one_way_weight_turnover(prev_top, curr_top)

    valid_to = top_turnover_series[np.isfinite(top_turnover_series)]
    mean_top_turnover = float(np.mean(valid_to)) if len(valid_to) > 0 else 0.0

    valid_mask = np.isfinite(ls_daily_returns)
    m = int(np.sum(valid_mask))

    if m == 0:
        return {
            "n_quantiles": n_quantiles,
            "group_returns": [0.0] * n_quantiles,
            "group_annual_returns": [0.0] * n_quantiles,
            "ls_daily_returns": ls_daily_returns,
            "ls_mean": 0.0,
            "ls_std": 0.0,
            "ls_annual_ret": 0.0,
            "ls_sharpe": 0.0,
            "ls_max_dd": 0.0,
            "monotonicity": 0.0,
            "valid_days": 0,
            "group_daily_returns": group_daily_returns,
            "top_turnover": mean_top_turnover,
            "mean_top_turnover": mean_top_turnover,
            "top_turnover_series": top_turnover_series,
            "top_turnover_valid_days": int(len(valid_to)),
        }

    group_returns = [float(np.mean(g[valid_mask])) for g in group_daily_returns]
    group_annual_returns = [r * 252.0 for r in group_returns]

    valid_ls = ls_daily_returns[valid_mask]
    ls_mean = float(np.mean(valid_ls))
    ls_std = float(np.std(valid_ls, ddof=1)) if m > 1 else 0.0
    ls_annual_ret = ls_mean * 252.0
    ls_sharpe = float(np.sqrt(252.0) * ls_mean / ls_std) if ls_std > 1e-12 else 0.0

    # 累计复利最大回撤
    wealth = 1.0
    peak = 1.0
    max_dd = 0.0
    for r in valid_ls:
        wealth = max(wealth * (1.0 + r), 0.0)
        if wealth > peak:
            peak = wealth
        if peak > 0.0:
            dd = (peak - wealth) / peak
            if dd > max_dd:
                max_dd = dd

    # 单调性得分
    expected_order = group_annual_returns if sign >= 0 else list(reversed(group_annual_returns))
    q_indices = np.arange(n_quantiles, dtype=np.float64)
    rq = ref_fractional_ranks(q_indices)
    rr = ref_fractional_ranks(np.array(expected_order, dtype=np.float64))
    mono = ref_pearson_correlation(rq, rr)
    monotonicity = float(mono) if np.isfinite(mono) else 0.0

    return {
        "n_quantiles": n_quantiles,
        "group_returns": group_returns,
        "group_annual_returns": group_annual_returns,
        "ls_daily_returns": ls_daily_returns,
        "ls_mean": ls_mean,
        "ls_std": ls_std,
        "ls_annual_ret": ls_annual_ret,
        "ls_sharpe": ls_sharpe,
        "ls_max_dd": float(max_dd),
        "monotonicity": monotonicity,
        "valid_days": m,
        "group_daily_returns": group_daily_returns,
        "top_turnover": mean_top_turnover,
        "mean_top_turnover": mean_top_turnover,
        "top_turnover_series": top_turnover_series,
        "top_turnover_valid_days": int(len(valid_to)),
    }


def ref_turnover(
    factor: np.ndarray,
    mask: np.ndarray,
    min_names: int = 5,
) -> dict[str, Any]:
    """因子截面排名自相关与换手率纯 NumPy 参考实现。"""
    T, N = factor.shape
    if T < 2:
        return {
            "turnover": 0.0,
            "rank_autocorr": 1.0,
            "daily_autocorr": np.empty(0, dtype=np.float64),
            "valid_days": 0,
        }

    daily_autocorr = np.full(T - 1, np.nan, dtype=np.float64)

    for t in range(1, T):
        m_prev = mask[t - 1]
        m_curr = mask[t]
        f_prev = factor[t - 1]
        f_curr = factor[t]

        valid_idx = np.where(m_prev & m_curr & np.isfinite(f_prev) & np.isfinite(f_curr))[0]
        if len(valid_idx) < min_names or len(valid_idx) < 2:
            continue

        rx = ref_fractional_ranks(f_prev[valid_idx])
        ry = ref_fractional_ranks(f_curr[valid_idx])
        daily_autocorr[t - 1] = ref_pearson_correlation(rx, ry)

    valid_vals = daily_autocorr[np.isfinite(daily_autocorr)]
    m = len(valid_vals)

    if m == 0:
        return {
            "turnover": 1.0,
            "rank_autocorr": 0.0,
            "daily_autocorr": daily_autocorr,
            "valid_days": 0,
        }

    mean_autocorr = float(np.mean(valid_vals))
    turnover = float(1.0 - mean_autocorr)

    return {
        "turnover": turnover,
        "rank_autocorr": mean_autocorr,
        "rank_autocorr_turnover": turnover,
        "daily_autocorr": daily_autocorr,
        "valid_days": m,
    }


def ref_top_quantile_turnover(
    factor: np.ndarray,
    mask: np.ndarray,
    n_quantiles: int = 5,
    min_names: int = 5,
    sign: int = 1,
) -> dict[str, Any]:
    """多头 Top 分位日度单边持仓权重换手率纯 NumPy 参考实现。"""
    T, N = factor.shape
    if T < 2:
        return {
            "turnover": 0.0,
            "mean_top_turnover": 0.0,
            "daily_turnover": np.full(T, np.nan, dtype=np.float64),
            "valid_days": 0,
        }

    daily_top_list: list[list[int]] = []
    top_q = (n_quantiles - 1) if sign >= 0 else 0

    for t in range(T):
        m_row = mask[t]
        f_row = factor[t]
        valid_idx = np.where(m_row & np.isfinite(f_row))[0]
        k_names = len(valid_idx)
        if k_names < min_names or k_names < n_quantiles or n_quantiles < 2:
            daily_top_list.append([])
            continue

        sub_f = f_row[valid_idx]
        sorted_pos = np.argsort(sub_f, kind="mergesort")
        sorted_idx = valid_idx[sorted_pos]

        top_start = int(np.round(top_q * k_names / n_quantiles))
        top_end = min(int(np.round((top_q + 1) * k_names / n_quantiles)), k_names)
        top_idx = sorted(sorted_idx[top_start:top_end].tolist())
        daily_top_list.append(top_idx)

    daily_turnover = np.full(T, np.nan, dtype=np.float64)
    for t in range(1, T):
        prev_top = daily_top_list[t - 1]
        curr_top = daily_top_list[t]
        if len(prev_top) > 0 and len(curr_top) > 0:
            daily_turnover[t] = ref_one_way_weight_turnover(prev_top, curr_top)

    valid_vals = daily_turnover[np.isfinite(daily_turnover)]
    m = len(valid_vals)
    turnover = float(np.mean(valid_vals)) if m > 0 else 0.0

    return {
        "turnover": turnover,
        "mean_top_turnover": turnover,
        "daily_turnover": daily_turnover,
        "valid_days": m,
    }


def ref_pairwise_correlation(
    factor_a: np.ndarray,
    factor_b: np.ndarray,
    mask: np.ndarray,
    min_names: int = 5,
    rank: bool = True,
) -> float:
    """双因子截面平均相关系数纯 NumPy 参考实现。"""
    T, N = factor_a.shape
    daily_corrs = []

    for t in range(T):
        m_row = mask[t]
        fa = factor_a[t]
        fb = factor_b[t]

        valid_idx = np.where(m_row & np.isfinite(fa) & np.isfinite(fb))[0]
        if len(valid_idx) < min_names or len(valid_idx) < 2:
            continue

        xa = fa[valid_idx]
        xb = fb[valid_idx]

        if rank:
            ra = ref_fractional_ranks(xa)
            rb = ref_fractional_ranks(xb)
            rho = ref_pearson_correlation(ra, rb)
        else:
            rho = ref_pearson_correlation(xa, xb)

        if np.isfinite(rho):
            daily_corrs.append(rho)

    if not daily_corrs:
        return 0.0
    return float(np.mean(daily_corrs))


def ref_correlation_matrix(
    factors: list[np.ndarray],
    mask: np.ndarray,
    min_names: int = 5,
    rank: bool = True,
) -> np.ndarray:
    """多因子截面相关系数矩阵纯 NumPy 参考实现。"""
    M = len(factors)
    mat = np.zeros((M, M), dtype=np.float64)
    for i in range(M):
        mat[i, i] = 1.0

    for i in range(M):
        for j in range(i + 1, M):
            rho = ref_pairwise_correlation(factors[i], factors[j], mask, min_names, rank)
            mat[i, j] = rho
            mat[j, i] = rho

    return mat


# ============================================================================
# 10. 12 项声明式后处理算子纯 NumPy 参考实现 (C++ aresexpradapter 对齐)
# ============================================================================


def pp_mask_mul_ref(x1: np.ndarray, x2: np.ndarray) -> np.ndarray:
    """baseUniMask / trdMask 参考实现: x1 * x2 (任一为 NaN 则置 NaN)。"""
    a = np.asarray(x1, dtype=np.float64)
    b = np.asarray(x2, dtype=np.float64)
    out = np.full(a.shape, np.nan, dtype=np.float64, order="F")
    mask = np.isfinite(a) & np.isfinite(b)
    with np.errstate(over="ignore", invalid="ignore"):
        prod = a[mask] * b[mask]
    prod[~np.isfinite(prod)] = np.nan
    out[mask] = prod
    return out


def pp_drop_new_stock_ref(
    x1: np.ndarray,
    x2: np.ndarray,
    min_days: float = 60.0,
) -> np.ndarray:
    """dropNewStock 参考实现: 当 x2 >= min_days 时保留 x1, 否则输出 NaN。"""
    a = np.asarray(x1, dtype=np.float64)
    b = np.asarray(x2, dtype=np.float64)
    out = np.full(a.shape, np.nan, dtype=np.float64, order="F")
    mask = np.isfinite(a) & np.isfinite(b) & (b >= float(min_days))
    out[mask] = a[mask]
    return out


def pp_mad_ref(x1: np.ndarray, threshold: float = 3.0) -> np.ndarray:
    """MAD 参考实现: 下中位数 (N-1)//2 与 1.4826 * threshold 极值裁剪。"""
    a = np.asarray(x1, dtype=np.float64)
    t_len, s_len = a.shape
    out = np.full((t_len, s_len), np.nan, dtype=np.float64, order="F")
    n_val = float(threshold)
    if not np.isfinite(n_val):
        return out
    for t in range(t_len):
        row = a[t]
        fin = np.sort(row[np.isfinite(row)])
        k = len(fin)
        if k == 0:
            continue
        med = float(fin[(k - 1) // 2])
        abs_devs = np.sort(np.abs(fin - med))
        mad = float(abs_devs[(k - 1) // 2])
        lo = med - n_val * 1.4826 * mad
        hi = med + n_val * 1.4826 * mad
        mask = np.isfinite(row)
        out[t, mask] = np.clip(row[mask], lo, hi)
    return out


def pp_winsor_ref(x1: np.ndarray, quantile: float = 0.025) -> np.ndarray:
    """winsor 参考实现: 线性内插分位数 [q, 1-q] 裁剪。"""
    a = np.asarray(x1, dtype=np.float64)
    t_len, s_len = a.shape
    out = np.full((t_len, s_len), np.nan, dtype=np.float64, order="F")
    q = float(quantile)
    if not np.isfinite(q):
        return out
    q = min(max(q, 0.0), 0.5)

    def _lin_q(sorted_vals: np.ndarray, p: float) -> float:
        n = len(sorted_vals)
        if n == 1:
            return float(sorted_vals[0])
        pos = p * (n - 1)
        lo_i = int(np.floor(pos))
        hi_i = int(np.ceil(pos))
        frac = pos - lo_i
        return float(sorted_vals[lo_i] + frac * (sorted_vals[hi_i] - sorted_vals[lo_i]))

    for t in range(t_len):
        row = a[t]
        fin = np.sort(row[np.isfinite(row)])
        if len(fin) == 0:
            continue
        lo = _lin_q(fin, q)
        hi = _lin_q(fin, 1.0 - q)
        mask = np.isfinite(row)
        out[t, mask] = np.clip(row[mask], lo, hi)
    return out


def pp_ind_neut_ref(x1: np.ndarray, x2: np.ndarray) -> np.ndarray:
    """indNeut 参考实现: 对 x2 > 0 减去同板块均值; 若 x2 缺失或 <= 0 则保留原值 x1。"""
    a = np.asarray(x1, dtype=np.float64)
    b = np.asarray(x2, dtype=np.float64)
    t_len, s_len = a.shape
    out = np.full((t_len, s_len), np.nan, dtype=np.float64, order="F")
    for t in range(t_len):
        groups: dict[float, list[float]] = {}
        for s in range(s_len):
            v1 = float(a[t, s])
            v2 = float(b[t, s])
            if np.isfinite(v1) and np.isfinite(v2) and v2 > 0.0:
                groups.setdefault(v2, []).append(v1)
        means = {k: sum(vs) / len(vs) for k, vs in groups.items() if vs}
        for s in range(s_len):
            v1 = float(a[t, s])
            v2 = float(b[t, s])
            if not np.isfinite(v1):
                out[t, s] = np.nan
            elif not np.isfinite(v2) or v2 <= 0.0:
                out[t, s] = v1
            else:
                out[t, s] = v1 - means.get(v2, 0.0)
    return out


def pp_risk_neut_ref(x1: np.ndarray, x2: np.ndarray) -> np.ndarray:
    """riskNeut 参考实现: 带截距单变量截面 OLS 残差 (缺失暴露保留原值)。"""
    a = np.asarray(x1, dtype=np.float64)
    b = np.asarray(x2, dtype=np.float64)
    t_len, s_len = a.shape
    out = np.full((t_len, s_len), np.nan, dtype=np.float64, order="F")
    for t in range(t_len):
        r1 = a[t]
        r2 = b[t]
        both_mask = np.isfinite(r1) & np.isfinite(r2)
        if np.sum(both_mask) < 2:
            continue
        b1 = r1[both_mask]
        b2 = r2[both_mask]
        m1_b = float(np.mean(b1))
        m2_b = float(np.mean(b2))
        cov = float(np.sum((b1 - m1_b) * (b2 - m2_b)))
        var2 = float(np.sum((b2 - m2_b) ** 2))
        if var2 <= 1e-12:
            continue
        beta = cov / var2
        fin1 = r1[np.isfinite(r1)]
        fin2 = r2[np.isfinite(r2)]
        alpha = float(np.mean(fin1)) - beta * float(np.mean(fin2))
        for s in range(s_len):
            v1 = float(r1[s])
            v2 = float(r2[s])
            if not np.isfinite(v1):
                out[t, s] = np.nan
            elif not np.isfinite(v2):
                out[t, s] = v1
            else:
                out[t, s] = v1 - (alpha + beta * v2)
    return out


def pp_risk_neut_multi_ref(
    x1: np.ndarray,
    exposures: list[np.ndarray],
) -> np.ndarray:
    """riskNeutMulti 参考实现: 无截距多变量截面 1e-8 岭回归残差。"""
    a = np.asarray(x1, dtype=np.float64)
    if not exposures:
        return a.copy(order="F")
    exps = [np.asarray(e, dtype=np.float64) for e in exposures]
    t_len, s_len = a.shape
    k_len = len(exps)
    out = np.full((t_len, s_len), np.nan, dtype=np.float64, order="F")
    for t in range(t_len):
        valid_mask = np.isfinite(a[t]).copy()
        for e in exps:
            valid_mask &= np.isfinite(e[t])
        valid_idx = np.where(valid_mask)[0]
        if len(valid_idx) == 0:
            continue
        y_vec = a[t, valid_idx]
        x_mat = np.column_stack([e[t, valid_idx] for e in exps])
        a_sys = x_mat.T @ x_mat + 1e-8 * np.eye(k_len, dtype=np.float64)
        b_sys = x_mat.T @ y_vec
        try:
            beta = np.linalg.solve(a_sys, b_sys)
        except np.linalg.LinAlgError:
            continue
        out[t, valid_idx] = y_vec - (x_mat @ beta)
    return out


def pp_zscore_ref(x1: np.ndarray) -> np.ndarray:
    """zScore 参考实现: 每日截面总体标准差 (ddof=0) 标准化。"""
    a = np.asarray(x1, dtype=np.float64)
    t_len, s_len = a.shape
    out = np.full((t_len, s_len), np.nan, dtype=np.float64, order="F")
    for t in range(t_len):
        row = a[t]
        mask = np.isfinite(row)
        vals = row[mask]
        if len(vals) == 0:
            continue
        mean_v = float(np.mean(vals))
        std_v = float(np.sqrt(np.mean((vals - mean_v) ** 2)))
        if std_v == 0.0 or not np.isfinite(std_v):
            continue
        out[t, mask] = (vals - mean_v) / std_v
    return out


def pp_rank_ref(x1: np.ndarray) -> np.ndarray:
    """rank 参考实现: 6 位精度截断稳定排序百分位排名。"""
    a = np.asarray(x1, dtype=np.float64)
    t_len, s_len = a.shape
    out = np.full((t_len, s_len), np.nan, dtype=np.float64, order="F")
    for t in range(t_len):
        row = a[t]
        fin_idx = [s for s in range(s_len) if np.isfinite(row[s])]
        n_fin = len(fin_idx)
        if n_fin == 0:
            continue
        vals = np.array([row[s] for s in fin_idx], dtype=np.float64)
        mean_v = float(np.mean(vals))
        std_v = float(np.sqrt(np.mean((vals - mean_v) ** 2)))
        if std_v < 1e-6:
            continue
        keyed = [(round(float(row[s]) / std_v, 6), s) for s in fin_idx]
        keyed.sort(key=lambda pair: (pair[0], pair[1]))
        for r_idx, (_, s) in enumerate(keyed, start=1):
            out[t, s] = r_idx / n_fin
    return out


def pp_decay_ref(x1: np.ndarray, window: int = 5) -> np.ndarray:
    """decay 参考实现: t < window - 1 置 NaN 的沿时间轴滚动均值。"""
    a = np.asarray(x1, dtype=np.float64)
    t_len, s_len = a.shape
    out = np.full((t_len, s_len), np.nan, dtype=np.float64, order="F")
    w = int(window)
    if w <= 0 or w > t_len:
        return out
    for t in range(w - 1, t_len):
        for s in range(s_len):
            vals = [float(a[k, s]) for k in range(t - w + 1, t + 1) if np.isfinite(a[k, s])]
            if vals:
                out[t, s] = sum(vals) / len(vals)
    return out


def pp_emadecay_ref(x1: np.ndarray, window: int = 5) -> np.ndarray:
    """emadecay 参考实现: t < window - 1 置 NaN 的 1..W 线性加权移动平均 (缺失权重重归一化)。"""
    a = np.asarray(x1, dtype=np.float64)
    t_len, s_len = a.shape
    out = np.full((t_len, s_len), np.nan, dtype=np.float64, order="F")
    w = int(window)
    if w <= 0 or w > t_len:
        return out
    for t in range(w - 1, t_len):
        for s in range(s_len):
            num = 0.0
            den = 0.0
            for i in range(1, w + 1):
                val = float(a[t - w + i, s])
                if np.isfinite(val):
                    num += i * val
                    den += i
            if den > 0.0:
                out[t, s] = num / den
    return out


# ============================================================================
# 10. 统一 T+1 前向预测标签参考实现 (M1.3)
# ============================================================================


def forward_returns_ref(
    entry_price: np.ndarray,
    exit_price: np.ndarray,
    entry_ok: np.ndarray,
    exit_ok: np.ndarray,
    horizons: list[int] | tuple[int, ...],
    strict: bool = True,
) -> dict[int, np.ndarray]:
    """点对点 T+1 前向收益率标签参考实现，用于与 Rust `build_forward_labels` 对账。"""
    ep = _clean(entry_price)
    xp = _clean(exit_price)
    eok = np.asarray(entry_ok, dtype=bool)
    xok = np.asarray(exit_ok, dtype=bool)
    t_len, n_cols = ep.shape

    next_exit: np.ndarray | None = None
    if not strict and t_len > 0:
        next_exit = np.full((t_len, n_cols), -1, dtype=np.int64)
        for j in range(n_cols):
            nxt = -1
            for tau in range(t_len - 1, -1, -1):
                if xok[tau, j] and np.isfinite(xp[tau, j]) and xp[tau, j] > 0.0:
                    nxt = tau
                next_exit[tau, j] = nxt

    out: dict[int, np.ndarray] = {}
    for h in horizons:
        h_int = int(h)
        mat = np.full((t_len, n_cols), np.nan, dtype=np.float64, order="F")
        if h_int <= 0 or t_len <= 1 + h_int:
            out[h_int] = mat
            continue
        max_rows = t_len - 1 - h_int
        for j in range(n_cols):
            for t in range(max_rows):
                t_in = t + 1
                if not eok[t_in, j]:
                    continue
                p_in = ep[t_in, j]
                if not (np.isfinite(p_in) and p_in > 0.0):
                    continue
                t_exit = t + 1 + h_int
                if strict:
                    if not xok[t_exit, j]:
                        continue
                    p_out = xp[t_exit, j]
                    if np.isfinite(p_out) and p_out > 0.0:
                        mat[t, j] = p_out / p_in - 1.0
                else:
                    assert next_exit is not None
                    out_idx = int(next_exit[t_exit, j])
                    if out_idx >= 0:
                        p_out = xp[out_idx, j]
                        if np.isfinite(p_out) and p_out > 0.0:
                            mat[t, j] = p_out / p_in - 1.0
        out[h_int] = mat
    return out


def compound_forward_labels_ref(
    step_returns: np.ndarray,
    horizons: list[int] | tuple[int, ...],
) -> dict[str, np.ndarray]:
    """链式单步执行收益的多期复合参考实现，用于与 Rust `compound_forward_labels` 对账。"""
    sr = _clean(step_returns)
    t_len, n_cols = sr.shape
    out: dict[str, np.ndarray] = {}

    r_exec_next = np.full((t_len, n_cols), np.nan, dtype=np.float64, order="F")
    if t_len >= 2:
        r_exec_next[:-1, :] = sr[1:, :]
    out["r_exec_next"] = r_exec_next

    for h in horizons:
        h_int = int(h)
        mat = np.full((t_len, n_cols), np.nan, dtype=np.float64, order="F")
        if h_int <= 0 or t_len <= h_int:
            out[f"y_{h_int}"] = mat
            continue
        for j in range(n_cols):
            for t in range(t_len - h_int):
                seg = sr[t + 1 : t + 1 + h_int, j]
                if np.all(np.isfinite(seg)):
                    mat[t, j] = float(np.prod(1.0 + seg) - 1.0)
        out[f"y_{h_int}"] = mat
    return out


def spanning_test_ref(
    target: np.ndarray,
    benchmarks: list[np.ndarray],
    label: np.ndarray,
    mask: np.ndarray,
    min_names: int = 5,
    horizon: int = 5,
) -> dict[str, object]:
    """多因子 Spanning Test 正交残差检验纯 Python/NumPy 参考实现。"""
    from scipy.stats import rankdata

    tgt = _clean(target)
    lbl = _clean(label)
    bm_list = [_clean(b) for b in benchmarks]
    m_arr = np.asarray(mask, dtype=bool)
    t_len, _ = tgt.shape
    req_min = max(2, int(min_names), len(bm_list) + 2)

    daily_ic = np.full(t_len, np.nan, dtype=np.float64)
    valid_rows: list[int] = []
    valid_ics: list[float] = []

    for row in range(t_len):
        vm = m_arr[row, :] & np.isfinite(tgt[row, :]) & np.isfinite(lbl[row, :])
        for bm in bm_list:
            vm = vm & np.isfinite(bm[row, :])
        idx = np.where(vm)[0]
        m = len(idx)
        if m < req_min:
            continue

        y_vec = tgt[row, idx].astype(np.float64)
        orig_var = float(np.var(y_vec))
        if not np.isfinite(orig_var) or orig_var <= 1e-24:
            continue

        if bm_list:
            cols = [np.ones(m, dtype=np.float64)]
            for bm in bm_list:
                cv = bm[row, idx].astype(np.float64)
                sd = float(np.std(cv))
                if np.isfinite(sd) and sd > 1e-12:
                    cols.append((cv - float(np.mean(cv))) / sd)
            if len(cols) > 1:
                x_mat = np.column_stack(cols)
                beta, _, _, _ = np.linalg.lstsq(x_mat, y_vec, rcond=1e-10)
                resid = y_vec - x_mat @ beta
            else:
                resid = y_vec - float(np.mean(y_vec))
        else:
            resid = y_vec

        r_var = float(np.var(resid))
        if not np.isfinite(r_var) or r_var <= 1e-24 or r_var <= 1e-18 * orig_var:
            continue

        r_ranks = rankdata(resid, method="average")
        l_ranks = rankdata(lbl[row, idx], method="average")
        r_d = r_ranks - float(np.mean(r_ranks))
        l_d = l_ranks - float(np.mean(l_ranks))
        denom = float(np.sqrt(np.sum(r_d * r_d) * np.sum(l_d * l_d)))
        if denom <= 1e-14:
            continue
        rho = float(np.clip(np.sum(r_d * l_d) / denom, -1.0, 1.0))
        daily_ic[row] = rho
        valid_rows.append(row)
        valid_ics.append(rho)

    n_days = len(valid_ics)
    h = max(1, int(horizon))
    if n_days == 0:
        return {
            "valid_days": 0,
            "effective_n": 0.0,
            "mean_ic": 0.0,
            "std_ic": 0.0,
            "icir": 0.0,
            "t_stat": 0.0,
            "t_stat_adj": 0.0,
            "daily_ic": daily_ic,
            "cum_ic": np.array([], dtype=np.float64),
            "row_indices": [],
        }

    arr = np.asarray(valid_ics, dtype=np.float64)
    mean_ic = float(np.mean(arr))
    std_ic = float(np.std(arr, ddof=1)) if n_days > 1 else 0.0
    icir = mean_ic / std_ic if std_ic > 1e-12 else 0.0
    eff_n = float(n_days) / float(h)
    t_stat = icir * float(np.sqrt(n_days)) if std_ic > 1e-12 and n_days > 1 else 0.0
    t_stat_adj = icir * float(np.sqrt(eff_n)) if std_ic > 1e-12 and n_days > 1 else 0.0
    return {
        "valid_days": n_days,
        "effective_n": eff_n,
        "mean_ic": mean_ic,
        "std_ic": std_ic,
        "icir": icir,
        "t_stat": t_stat,
        "t_stat_adj": t_stat_adj,
        "daily_ic": daily_ic,
        "cum_ic": np.cumsum(arr),
        "row_indices": valid_rows,
    }


# ==============================================================================
# FactorLab MCP 1d 算子全量对齐：Rust 内核 (factorlib_core) 的 NumPy 黄金参考实现
# ==============================================================================
#
# 本段严格镜像 `rust/factorlib_core/src/ops/{elem,cs,ts,multi}.rs` 的数学定义与
# 边界语义（NaN 掩码、并列值容差、窗口满窗条件、总体/样本矩口径、岭正则项），
# 作为 `tests/python/test_rust_core.py` 中 `max |y_rust - y_ref| < 1e-9` 对账基准。

_RANK_TOL_BASE = 1e-12


def _tol_for(v: float) -> float:
    """与 Rust `STD_EPS * max(1, |v|)` 完全一致的并列判定容差。"""
    return _RANK_TOL_BASE * max(1.0, abs(float(v)))


def _clean_scalar(v: float) -> float:
    return float(v) if np.isfinite(v) else float("nan")


def _broadcast_row_agg(x: np.ndarray, agg: np.ndarray) -> np.ndarray:
    """把逐行聚合值广播回 (T, N)，仅在实际取值有限的位置输出。"""
    finite = np.isfinite(x)
    return np.where(finite, agg[:, None], np.nan).astype(np.float64)


def _merge_sorted_tolerant(sorted_vals: list[float]) -> list[float]:
    """按 Rust `uniq` 逻辑对升序列表做容差合并（与最后保留值比较）。"""
    uniq: list[float] = []
    for v in sorted_vals:
        if not uniq:
            uniq.append(v)
        else:
            last = uniq[-1]
            if abs(v - last) > _tol_for(last):
                uniq.append(v)
    return uniq


def _bsearch_tolerant(uniq: list[float], v: float) -> int:
    """复刻 Rust `binary_search_by` 的容差二分查找，未命中返回插入点。"""
    size = len(uniq)
    if size == 0:
        return 0

    def cmp(u: float) -> int:
        if abs(u - v) <= _tol_for(v):
            return 0
        return -1 if u < v else 1

    base = 0
    while size > 1:
        half = size // 2
        mid = base + half
        c = cmp(uniq[mid])
        if c == 0:
            return mid
        if c > 0:
            # u > v（Greater）→ base 保持不变
            pass
        else:
            base = mid
        size -= half
    c = cmp(uniq[base])
    if c == 0:
        return base
    return base + (1 if c < 0 else 0)


def _linear_quantile_sorted(sorted_vals: np.ndarray, p: float) -> float:
    """与 Rust `linear_quantile_sorted` 完全一致的线性内插分位数。"""
    k = len(sorted_vals)
    if k == 0:
        return float("nan")
    if k == 1:
        return float(sorted_vals[0])
    q = min(max(float(p), 0.0), 1.0)
    pos = q * (k - 1)
    lo = int(np.floor(pos))
    hi = int(np.ceil(pos))
    frac = pos - lo
    return float(sorted_vals[lo] + frac * (sorted_vals[hi] - sorted_vals[lo]))


# ------------------------------------------------------------------------------
# 数学变换扩充与缺失值处理
# ------------------------------------------------------------------------------


def exp_(x: np.ndarray) -> np.ndarray:
    """`exp(x)`：逐元素自然指数（上溢 `+Inf` 归一为 NaN）。"""
    out = np.full(x.shape, np.nan, dtype=np.float64, order="F")
    finite = np.isfinite(x)
    with np.errstate(over="ignore", invalid="ignore"):
        vals = np.exp(x[finite])
    out[finite] = np.where(np.isfinite(vals), vals, np.nan)
    return out


def ceil(x: np.ndarray) -> np.ndarray:
    """`ceil(x)`：逐元素向上取整。"""
    return np.where(np.isfinite(x), np.ceil(x), np.nan).astype(np.float64)


def sign_power(x: np.ndarray) -> np.ndarray:
    """`sign_power(x)`：带符号平方 sign(x) * x^2。"""
    return np.where(np.isfinite(x), np.sign(x) * x * x, np.nan).astype(np.float64)


def unique_count(x: np.ndarray) -> np.ndarray:
    """`unique_count(x)`：逐截面不同非 NaN 数值个数（容差合并）。"""
    T, N = x.shape
    agg = np.full(T, np.nan, dtype=np.float64)
    for t in range(T):
        vals = np.sort(x[t, np.isfinite(x[t, :])])
        if vals.size == 0:
            continue
        count = 1
        for i in range(1, len(vals)):
            tol = _tol_for(vals[i])
            if abs(vals[i] - vals[i - 1]) > tol:
                count += 1
        agg[t] = float(count)
    return _broadcast_row_agg(x, agg)


def is_inf(x: np.ndarray) -> np.ndarray:
    """`is_inf(x)`：正负无穷 → 1.0，有限值 → 0.0，NaN 保持 NaN。"""
    out = np.full(x.shape, np.nan, dtype=np.float64, order="F")
    nan_mask = np.isnan(x)
    out[~nan_mask] = np.where(np.isinf(x[~nan_mask]), 1.0, 0.0)
    return out


def is_nan(x: np.ndarray) -> np.ndarray:
    """`is_nan(x)`：NaN → 1.0，其余 → 0.0。"""
    return np.where(np.isnan(x), 1.0, 0.0).astype(np.float64)


def purify(x: np.ndarray) -> np.ndarray:
    """`purify(x)`：将正负无穷转换为 NaN。"""
    return np.where(np.isfinite(x), x, np.nan).astype(np.float64)


def zero_to_nan(x: np.ndarray) -> np.ndarray:
    """`zero_to_nan(x)`：将精确零值转换为 NaN（非有限值同样归一为 NaN）。"""
    finite = np.isfinite(x)
    return np.where(finite & (x != 0.0), x, np.nan).astype(np.float64)


def fillna(x: np.ndarray, fill_value: float) -> np.ndarray:
    """`fillna(x, c)`：仅把 NaN 替换为 fill_value。"""
    out = np.full(x.shape, np.nan, dtype=np.float64, order="F")
    if not np.isfinite(fill_value):
        return out
    finite = np.isfinite(x)
    out[finite] = x[finite]
    out[np.isnan(x)] = float(fill_value)
    return out


def round_decimals(x: np.ndarray, decimals: float) -> np.ndarray:
    """`round(x, decimals)`：按 decimals 位舍入（向偶数舍入）。"""
    if not np.isfinite(decimals):
        return np.full(x.shape, np.nan, dtype=np.float64, order="F")
    d = int(np.clip(int(np.round(decimals)), -15, 15))
    scale = 10.0**d
    finite = np.isfinite(x)
    out = np.full(x.shape, np.nan, dtype=np.float64, order="F")
    out[finite] = np.round(x[finite] * scale) / scale
    return out


def perc_long(x: np.ndarray, percentile: float) -> np.ndarray:
    """`perc_long(x, p)`：减截面分位点后正负配对的多空数值。"""
    T, N = x.shape
    out = np.full(x.shape, np.nan, dtype=np.float64, order="F")
    if not np.isfinite(percentile):
        return out
    q = min(max(float(percentile), 0.0), 1.0)
    for t in range(T):
        row = x[t, :]
        finite = np.isfinite(row)
        if not np.any(finite):
            continue
        vals = np.sort(row[finite])
        quant = _linear_quantile_sorted(vals, q)
        d = row - quant
        pos_sum = float(np.sum(d[finite & (d > 0.0)]))
        neg_cnt = int(np.sum(finite & (d < 0.0)))
        neg_fill = -(pos_sum / neg_cnt) if neg_cnt > 0 else 0.0
        res = np.full(N, np.nan, dtype=np.float64)
        res[finite & (d > 0.0)] = d[finite & (d > 0.0)]
        res[finite & (d < 0.0)] = neg_fill
        res[finite & (d == 0.0)] = 0.0
        out[t, :] = res
    return out


# ------------------------------------------------------------------------------
# 横截面统计扩充
# ------------------------------------------------------------------------------


def cs_sum(x: np.ndarray) -> np.ndarray:
    finite = np.isfinite(x)
    cnt = finite.sum(axis=1)
    sums = np.where(finite, x, 0.0).sum(axis=1)
    agg = np.where(cnt >= 1, sums, np.nan)
    return _broadcast_row_agg(x, agg)


def cs_mean(x: np.ndarray) -> np.ndarray:
    finite = np.isfinite(x)
    cnt = finite.sum(axis=1)
    sums = np.where(finite, x, 0.0).sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        agg = np.where(cnt >= 1, sums / np.maximum(cnt, 1), np.nan)
    return _broadcast_row_agg(x, agg)


def _cs_mean_m2(x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    finite = np.isfinite(x)
    cnt = finite.sum(axis=1).astype(np.float64)
    safe = np.where(finite, x, 0.0)
    sums = safe.sum(axis=1)
    means = np.where(cnt >= 1, sums / np.maximum(cnt, 1.0), np.nan)
    mask = finite.astype(np.float64)
    with np.errstate(invalid="ignore"):
        d = np.where(finite, x - means[:, None], 0.0)
    m2 = (d * d * mask).sum(axis=1)
    return cnt, means, m2


def cs_var(x: np.ndarray) -> np.ndarray:
    cnt, means, m2 = _cs_mean_m2(x)
    agg = np.where((cnt >= 1) & np.isfinite(means), m2 / np.maximum(cnt, 1.0), np.nan)
    return _broadcast_row_agg(x, agg)


def cs_std(x: np.ndarray) -> np.ndarray:
    cnt, means, m2 = _cs_mean_m2(x)
    with np.errstate(invalid="ignore"):
        agg = np.where((cnt >= 1) & np.isfinite(means), np.sqrt(m2 / np.maximum(cnt, 1.0)), np.nan)
    return _broadcast_row_agg(x, agg)


def cs_max(x: np.ndarray) -> np.ndarray:
    finite = np.isfinite(x)
    with np.errstate(invalid="ignore"):
        agg = np.where(finite.any(axis=1), np.nanmax(np.where(finite, x, -np.inf), axis=1), np.nan)
    return _broadcast_row_agg(x, agg)


def cs_min(x: np.ndarray) -> np.ndarray:
    finite = np.isfinite(x)
    with np.errstate(invalid="ignore"):
        agg = np.where(finite.any(axis=1), np.nanmin(np.where(finite, x, np.inf), axis=1), np.nan)
    return _broadcast_row_agg(x, agg)


def cs_nonan_num(x: np.ndarray) -> np.ndarray:
    agg = np.isfinite(x).sum(axis=1).astype(np.float64)
    return _broadcast_row_agg(x, agg)


def cs_argmax(x: np.ndarray) -> np.ndarray:
    T, N = x.shape
    agg = np.full(T, np.nan, dtype=np.float64)
    for t in range(T):
        row = x[t, :]
        finite = np.isfinite(row)
        if not np.any(finite):
            continue
        idx = np.flatnonzero(finite)
        vals = row[idx]
        best = 0
        for k in range(1, len(idx)):
            if vals[k] > vals[best]:
                best = k
        agg[t] = float(idx[best])
    return _broadcast_row_agg(x, agg)


def cs_argmin(x: np.ndarray) -> np.ndarray:
    T, N = x.shape
    agg = np.full(T, np.nan, dtype=np.float64)
    for t in range(T):
        row = x[t, :]
        finite = np.isfinite(row)
        if not np.any(finite):
            continue
        idx = np.flatnonzero(finite)
        vals = row[idx]
        best = 0
        for k in range(1, len(idx)):
            if vals[k] < vals[best]:
                best = k
        agg[t] = float(idx[best])
    return _broadcast_row_agg(x, agg)


def cs_median(x: np.ndarray) -> np.ndarray:
    T, _N = x.shape
    agg = np.full(T, np.nan, dtype=np.float64)
    for t in range(T):
        vals = np.sort(x[t, np.isfinite(x[t, :])])
        if vals.size == 0:
            continue
        agg[t] = float(vals[(len(vals) - 1) // 2])
    return _broadcast_row_agg(x, agg)


def cs_quantile(x: np.ndarray, q: float) -> np.ndarray:
    T, _N = x.shape
    agg = np.full(T, np.nan, dtype=np.float64)
    if not np.isfinite(q):
        return _broadcast_row_agg(x, agg)
    for t in range(T):
        vals = np.sort(x[t, np.isfinite(x[t, :])])
        if vals.size == 0:
            continue
        agg[t] = _linear_quantile_sorted(vals, q)
    return _broadcast_row_agg(x, agg)


def cs_mad(x: np.ndarray) -> np.ndarray:
    finite = np.isfinite(x)
    cnt = finite.sum(axis=1).astype(np.float64)
    safe = np.where(finite, x, 0.0)
    means = np.where(cnt >= 1, safe.sum(axis=1) / np.maximum(cnt, 1.0), np.nan)
    with np.errstate(invalid="ignore"):
        dev = np.where(finite, np.abs(x - means[:, None]), 0.0)
    agg = np.where((cnt >= 1) & np.isfinite(means), dev.sum(axis=1) / np.maximum(cnt, 1.0), np.nan)
    return _broadcast_row_agg(x, agg)


def _cs_moments(x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    finite = np.isfinite(x)
    cnt = finite.sum(axis=1).astype(np.float64)
    safe = np.where(finite, x, 0.0)
    means = np.where(cnt >= 1, safe.sum(axis=1) / np.maximum(cnt, 1.0), np.nan)
    with np.errstate(invalid="ignore"):
        d = np.where(finite, x - means[:, None], 0.0)
    d2 = d * d
    m2 = d2.sum(axis=1) / np.maximum(cnt, 1.0)
    m3 = (d2 * d).sum(axis=1) / np.maximum(cnt, 1.0)
    m4 = (d2 * d2).sum(axis=1) / np.maximum(cnt, 1.0)
    return cnt, m2, m3, m4


def cs_skew(x: np.ndarray) -> np.ndarray:
    cnt, m2, m3, _m4 = _cs_moments(x)
    finite = np.isfinite(x)
    safe = np.where(finite, x, 0.0)
    mu = safe.sum(axis=1) / np.maximum(cnt, 1.0)
    scale = np.maximum(1.0, np.abs(mu))
    with np.errstate(invalid="ignore"):
        ok = (cnt >= 3) & np.isfinite(mu) & (m2 > 1e-24 * scale * scale)
        agg = np.where(ok, m3 / np.power(m2, 1.5), np.nan)
    return _broadcast_row_agg(x, agg)


def cs_kurt(x: np.ndarray) -> np.ndarray:
    cnt, m2, _m3, m4 = _cs_moments(x)
    finite = np.isfinite(x)
    safe = np.where(finite, x, 0.0)
    mu = safe.sum(axis=1) / np.maximum(cnt, 1.0)
    scale = np.maximum(1.0, np.abs(mu))
    with np.errstate(invalid="ignore"):
        ok = (cnt >= 4) & np.isfinite(mu) & (m2 > 1e-24 * scale * scale)
        agg = np.where(ok, m4 / (m2 * m2) - 3.0, np.nan)
    return _broadcast_row_agg(x, agg)


def cs_pct_rank(x: np.ndarray) -> np.ndarray:
    T, N = x.shape
    out = np.full(x.shape, np.nan, dtype=np.float64, order="F")
    for t in range(T):
        finite = np.isfinite(x[t, :])
        idx = np.flatnonzero(finite)
        if idx.size == 0:
            continue
        vals = x[t, idx]
        order = np.lexsort((idx, vals))
        k = float(idx.size)
        for rank, pos in enumerate(order):
            out[t, idx[pos]] = (rank + 1) / k
    return out


def _cs_dense(x: np.ndarray, pct: bool) -> np.ndarray:
    T, N = x.shape
    out = np.full(x.shape, np.nan, dtype=np.float64, order="F")
    for t in range(T):
        finite = np.isfinite(x[t, :])
        idx = np.flatnonzero(finite)
        if idx.size == 0:
            continue
        vals = x[t, idx]
        order = np.lexsort((idx, vals))
        sorted_idx = idx[order]
        sorted_vals = vals[order]
        rank = 0
        i = 0
        while i < len(sorted_vals):
            base_v = sorted_vals[i]
            tol = _tol_for(base_v)
            j = i + 1
            while j < len(sorted_vals) and abs(sorted_vals[j] - base_v) <= tol:
                j += 1
            rank += 1
            for k in range(i, j):
                out[t, sorted_idx[k]] = float(rank)
            i = j
        if pct and rank > 0:
            row_mask = np.isfinite(out[t, :])
            out[t, row_mask] = out[t, row_mask] / float(rank)
    return out


def cs_dense_rank(x: np.ndarray) -> np.ndarray:
    return _cs_dense(x, False)


def cs_dense_pct_rank(x: np.ndarray) -> np.ndarray:
    return _cs_dense(x, True)


def cs_group(x: np.ndarray, groups: float) -> np.ndarray:
    if not np.isfinite(groups) or groups < 1.0:
        return np.full(x.shape, np.nan, dtype=np.float64, order="F")
    g = max(1.0, float(np.floor(groups)))
    pct = cs_pct_rank(x)
    grp = np.floor(pct * g)
    grp = np.clip(grp, 0.0, g - 1.0)
    return np.where(np.isfinite(pct), grp, np.nan).astype(np.float64)


def _cs_pair_moments(
    y: np.ndarray, x: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    both = np.isfinite(y) & np.isfinite(x)
    cnt = both.sum(axis=1).astype(np.float64)
    ys = np.where(both, y, 0.0)
    xs = np.where(both, x, 0.0)
    with np.errstate(invalid="ignore", divide="ignore"):
        my = np.where(cnt >= 1, ys.sum(axis=1) / np.maximum(cnt, 1.0), np.nan)
        mx = np.where(cnt >= 1, xs.sum(axis=1) / np.maximum(cnt, 1.0), np.nan)
    with np.errstate(invalid="ignore"):
        dy = np.where(both, y - my[:, None], 0.0)
        dx = np.where(both, x - mx[:, None], 0.0)
    cxy = (dy * dx).sum(axis=1)
    m2x = (dx * dx).sum(axis=1)
    m2y = (dy * dy).sum(axis=1)
    return cnt, my, mx, cxy, m2x, m2y


def cs_cov(y: np.ndarray, x: np.ndarray) -> np.ndarray:
    cnt, _my, _mx, cxy, _m2x, _m2y = _cs_pair_moments(y, x)
    with np.errstate(invalid="ignore", divide="ignore"):
        agg = np.where(cnt >= 2, cxy / np.maximum(cnt - 1.0, 1.0), np.nan)
    return _broadcast_row_agg(y, agg)


def cs_beta(y: np.ndarray, x: np.ndarray) -> np.ndarray:
    cnt, _my, mx, cxy, m2x, _m2y = _cs_pair_moments(y, x)
    scale = np.maximum(1.0, np.abs(mx))
    with np.errstate(invalid="ignore", divide="ignore"):
        ok = (cnt >= 2) & (m2x > 1e-24 * cnt * scale * scale)
        agg = np.where(ok, cxy / np.where(ok, m2x, 1.0), np.nan)
    return _broadcast_row_agg(y, agg)


def cs_corr(y: np.ndarray, x: np.ndarray) -> np.ndarray:
    cnt, my, mx, cxy, m2x, m2y = _cs_pair_moments(y, x)
    sy = np.maximum(1.0, np.abs(my))
    sx = np.maximum(1.0, np.abs(mx))
    with np.errstate(invalid="ignore", divide="ignore"):
        ok = (cnt >= 2) & (m2x > 1e-24 * cnt * sx * sx) & (m2y > 1e-24 * cnt * sy * sy)
        denom = np.sqrt(np.where(ok, m2x * m2y, np.nan))
        vals = np.where(ok & (denom > 0.0), np.clip(cxy / denom, -1.0, 1.0), np.nan)
    return _broadcast_row_agg(y, vals)


def barra5_zScore(x: np.ndarray, weight: np.ndarray) -> np.ndarray:
    """市值加权均值 + 等权总体标准差标准化（Barra 5 口径）。"""
    valid = np.isfinite(x) & np.isfinite(weight) & (weight >= 0.0)
    cnt = valid.sum(axis=1).astype(np.float64)
    w = np.where(valid, weight, 0.0)
    xv = np.where(valid, x, 0.0)
    wsum = w.sum(axis=1)
    wxsum = (w * xv).sum(axis=1)
    esum = xv.sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        ew_mean = np.where(cnt >= 1, esum / np.maximum(cnt, 1.0), np.nan)
        wmean = np.where((cnt >= 1) & (wsum > 0.0), wxsum / np.where(wsum > 0.0, wsum, 1.0), np.nan)
    with np.errstate(invalid="ignore"):
        d = np.where(valid, x - wmean[:, None], 0.0)
    m2 = (d * d).sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        s = np.sqrt(m2 / np.maximum(cnt, 1.0))
        scale = np.maximum(1.0, np.abs(ew_mean))
        ok = (cnt >= 1) & np.isfinite(wmean) & np.isfinite(s) & (s > 1e-12 * scale)
        out = np.where(
            valid & ok[:, None], (x - wmean[:, None]) / np.where(ok, s, 1.0)[:, None], np.nan
        )
    return out.astype(np.float64, order="F")


# ------------------------------------------------------------------------------
# 时序窗口扩充算子
# ------------------------------------------------------------------------------


def ts_median(x: np.ndarray, d: int) -> np.ndarray:
    """`ts_median(x, d)`：满窗内有限观测的下中位数。"""
    T, N = x.shape
    out = np.full(x.shape, np.nan, dtype=np.float64, order="F")
    if d == 0 or d > T:
        return out
    for col in range(N):
        for row in range(d - 1, T):
            vals = x[row + 1 - d : row + 1, col]
            vals = vals[np.isfinite(vals)]
            if vals.size == 0:
                continue
            vals = np.sort(vals)
            out[row, col] = float(vals[(len(vals) - 1) // 2])
    return out


def ts_quantile(x: np.ndarray, q: float, d: int) -> np.ndarray:
    """`ts_quantile(x, q, d)`：满窗内有限观测的线性内插 q 分位数。"""
    T, N = x.shape
    out = np.full(x.shape, np.nan, dtype=np.float64, order="F")
    if d == 0 or d > T or not np.isfinite(q):
        return out
    for col in range(N):
        for row in range(d - 1, T):
            vals = x[row + 1 - d : row + 1, col]
            vals = vals[np.isfinite(vals)]
            if vals.size == 0:
                continue
            vals = np.sort(vals)
            out[row, col] = _linear_quantile_sorted(vals, q)
    return out


def ts_roll(x: np.ndarray, d: int) -> np.ndarray:
    """`ts_roll(x, d)`：沿时间轴循环移位，首尾环绕。"""
    T, _N = x.shape
    out = np.full(x.shape, np.nan, dtype=np.float64, order="F")
    if T == 0:
        return out
    shift = d % T
    for row in range(T):
        src = (row + T - shift) % T
        out[row, :] = np.where(np.isfinite(x[src, :]), x[src, :], np.nan)
    return out


def ts_prod(x: np.ndarray, d: int) -> np.ndarray:
    """`ts_prod(x, d)`：满窗内有限观测连乘积。"""
    T, N = x.shape
    out = np.full(x.shape, np.nan, dtype=np.float64, order="F")
    if d == 0 or d > T:
        return out
    for col in range(N):
        for row in range(d - 1, T):
            vals = x[row + 1 - d : row + 1, col]
            vals = vals[np.isfinite(vals)]
            if vals.size == 0:
                continue
            with np.errstate(over="ignore", invalid="ignore"):
                prod = float(np.prod(vals))
            out[row, col] = prod if np.isfinite(prod) else np.nan
    return out


def ts_nonan_num(x: np.ndarray, d: int) -> np.ndarray:
    """`ts_nonan_num(x, d)`：满窗内非 NaN 计数。"""
    T, N = x.shape
    out = np.full(x.shape, np.nan, dtype=np.float64, order="F")
    if d == 0 or d > T:
        return out
    for col in range(N):
        for row in range(d - 1, T):
            out[row, col] = float(np.isfinite(x[row + 1 - d : row + 1, col]).sum())
    return out


def ts_pct_rank(x: np.ndarray, d: int) -> np.ndarray:
    """`ts_pct_rank(x, d)`：当前值在满窗内的百分位排名（1 基平均名次 / 有效样本数）。"""
    T, N = x.shape
    out = np.full(x.shape, np.nan, dtype=np.float64, order="F")
    if d == 0 or d > T:
        return out
    for col in range(N):
        for row in range(d - 1, T):
            cur = x[row, col]
            if not np.isfinite(cur):
                continue
            vals = x[row + 1 - d : row + 1, col]
            vals = vals[np.isfinite(vals)]
            k = vals.size
            if k == 0:
                continue
            tol = _tol_for(cur)
            c_less = float(np.sum(vals < cur - tol))
            c_eq = float(np.sum(np.abs(vals - cur) <= tol))
            out[row, col] = (c_less + 0.5 * (c_eq + 1.0)) / float(k)
    return out


def ts_group(x: np.ndarray, groups: int) -> np.ndarray:
    """`ts_group(x, groups)`：按一周期收益率的因果历史百分位浮点整除分组。"""
    T, N = x.shape
    out = np.full(x.shape, np.nan, dtype=np.float64, order="F")
    if T == 0 or groups == 0:
        return out
    g = float(groups)
    ret = pct_change(x, 1)
    for col in range(N):
        hist: list[float] = []
        for row in range(T):
            r = ret[row, col]
            if not np.isfinite(r):
                continue
            hist.append(float(r))
            arr = np.asarray(hist, dtype=np.float64)
            k = arr.size
            tol = _tol_for(r)
            c_less = float(np.sum(arr < r - tol))
            c_eq = float(np.sum(np.abs(arr - r) <= tol))
            pct = (c_less + 0.5 * (c_eq + 1.0)) / float(k)
            out[row, col] = min(max(float(np.floor(pct * g)), 0.0), g - 1.0)
    return out


def ts_dw(x1: np.ndarray, x2: np.ndarray) -> np.ndarray:
    """`ts_dw(x1, x2)`：扩张窗口残差 Durbin-Watson 统计量。"""
    T, N = x1.shape
    out = np.full(x1.shape, np.nan, dtype=np.float64, order="F")
    for col in range(N):
        cnt = 0
        mean_x = 0.0
        mean_y = 0.0
        m2_x = 0.0
        c_xy = 0.0
        prev_resid = np.nan
        dw_num = 0.0
        dw_den = 0.0
        n_resid = 0
        for row in range(T):
            vx = x2[row, col]
            vy = x1[row, col]
            if not (np.isfinite(vx) and np.isfinite(vy)):
                continue
            cnt += 1
            kf = float(cnt)
            dx = vx - mean_x
            dy = vy - mean_y
            mean_x += dx / kf
            mean_y += dy / kf
            dx2 = vx - mean_x
            dy2 = vy - mean_y
            m2_x += dx * dx2
            c_xy += dx * dy2
            scale_x = max(1.0, abs(mean_x))
            if cnt >= 2 and m2_x > 1e-24 * kf * scale_x * scale_x:
                slope = c_xy / m2_x
                intercept = mean_y - slope * mean_x
            else:
                slope = 0.0
                intercept = mean_y
            resid = vy - (intercept + slope * vx)
            dw_den += resid * resid
            if np.isfinite(prev_resid):
                dw_num += (resid - prev_resid) ** 2
            prev_resid = resid
            n_resid += 1
            if n_resid >= 3 and dw_den > 0.0:
                out[row, col] = dw_num / dw_den
    return out


def _expanding_rank_core(x: np.ndarray, pct: bool, dense: bool) -> np.ndarray:
    T, N = x.shape
    out = np.full(x.shape, np.nan, dtype=np.float64, order="F")
    for col in range(N):
        pairs: list[tuple[int, float]] = [
            (row, float(x[row, col])) for row in range(T) if np.isfinite(x[row, col])
        ]
        if not pairs:
            continue
        uniq = _merge_sorted_tolerant(sorted(v for _, v in pairs))
        m = len(uniq)
        cnts = [0] * m
        seen = [False] * m
        for row, v in pairs:
            b = _bsearch_tolerant(uniq, v)
            b = min(max(b, 0), m - 1) if m > 0 else 0
            cnts[b] += 1
            seen[b] = True
            total = float(sum(cnts))
            n_distinct = float(sum(1 for s in seen if s))
            if dense:
                num = float(sum(1 for k in range(b + 1) if seen[k]))
                den = n_distinct
            else:
                num = float(sum(cnts[: b + 1]))
                den = total
            if pct:
                out[row, col] = num / den if den > 0.0 else np.nan
            else:
                out[row, col] = num
    return out


def rank_ts(x: np.ndarray) -> np.ndarray:
    """`rank_ts(x)`：扩张窗口 1 基原始名次（严格因果）。"""
    return _expanding_rank_core(x, False, False)


def pct_rank_ts(x: np.ndarray) -> np.ndarray:
    """`pct_rank_ts(x)`：扩张窗口名次百分比。"""
    return _expanding_rank_core(x, True, False)


def dense_rank_ts(x: np.ndarray) -> np.ndarray:
    """`dense_rank_ts(x)`：扩张窗口稠密名次。"""
    return _expanding_rank_core(x, False, True)


def pct_dense_rank_ts(x: np.ndarray) -> np.ndarray:
    """`pct_dense_rank_ts(x)`：扩张窗口稠密名次百分比。"""
    return _expanding_rank_core(x, True, True)


def ts_dense_rank(x: np.ndarray) -> np.ndarray:
    """`ts_dense_rank(x)`：等价于 `dense_rank_ts`。"""
    return _expanding_rank_core(x, False, True)


def ts_pct_dense_rank(x: np.ndarray) -> np.ndarray:
    """`ts_pct_dense_rank(x)`：等价于 `pct_dense_rank_ts`。"""
    return _expanding_rank_core(x, True, True)


def ts_cumsum(x: np.ndarray) -> np.ndarray:
    """`ts_cumsum(x)`：沿时间轴对有限观测累加。"""
    T, N = x.shape
    out = np.full(x.shape, np.nan, dtype=np.float64, order="F")
    for col in range(N):
        acc = 0.0
        for row in range(T):
            v = x[row, col]
            if np.isfinite(v):
                acc += v
                out[row, col] = acc if np.isfinite(acc) else np.nan
    return out


def ts_cumprod(x: np.ndarray) -> np.ndarray:
    """`ts_cumprod(x)`：沿时间轴对有限观测累乘。"""
    T, N = x.shape
    out = np.full(x.shape, np.nan, dtype=np.float64, order="F")
    for col in range(N):
        acc = 1.0
        for row in range(T):
            v = x[row, col]
            if np.isfinite(v):
                acc *= v
                out[row, col] = acc if np.isfinite(acc) else np.nan
    return out


def ts_ffill(x: np.ndarray) -> np.ndarray:
    """`ts_ffill(x)`：沿时间轴前向填充 NaN。"""
    T, N = x.shape
    out = np.full(x.shape, np.nan, dtype=np.float64, order="F")
    for col in range(N):
        last = np.nan
        for row in range(T):
            v = x[row, col]
            if np.isfinite(v):
                last = v
                out[row, col] = v
            elif np.isfinite(last):
                out[row, col] = last
    return out


# ------------------------------------------------------------------------------
# 变参多元（加权）最小二乘回归算子
# ------------------------------------------------------------------------------

MULTI_RIDGE_EPS = 1.0e-10


def _solve_normal(a_mat: np.ndarray, b_vec: np.ndarray) -> np.ndarray | None:
    """求解正规方程（等价于 Rust 列主元高斯-约旦消元，奇异时返回 None）。"""
    try:
        return np.linalg.solve(a_mat, b_vec)
    except np.linalg.LinAlgError:
        return None


def _obs_mask(
    y_row: np.ndarray, exps: list[np.ndarray], weights: np.ndarray | None, row: int
) -> np.ndarray:
    """与 Rust `obs_valid` 完全一致的逐点有效性判定。"""
    mask = np.isfinite(y_row)
    for e in exps:
        mask &= np.isfinite(e[row, :])
    if weights is not None:
        w_row = weights[row, :]
        mask &= np.isfinite(w_row) & (w_row >= 0.0)
    return mask


def _cs_multi_core(
    y: np.ndarray,
    exps: list[np.ndarray],
    weights: np.ndarray | None,
    intercept: bool,
    want_beta: bool,
) -> np.ndarray:
    T, _N = y.shape
    out = np.full(y.shape, np.nan, dtype=np.float64, order="F")
    k = len(exps)
    if k == 0:
        return out
    dim = k + (1 if intercept else 0)
    for row in range(T):
        mask = _obs_mask(y[row, :], exps, weights, row)
        cnt = int(mask.sum())
        if cnt == 0 or cnt < dim:
            continue
        w_scale = 0.0
        w_vec = np.ones(cnt, dtype=np.float64)
        if weights is not None:
            w_vec = weights[row, mask].astype(np.float64)
            w_scale = float(w_vec.max())
            if not w_scale > 0.0:
                continue
            w_vec = w_vec / w_scale
        cols = [e[row, mask] for e in exps]
        if intercept:
            cols.append(np.ones(cnt, dtype=np.float64))
        x_mat = np.column_stack(cols)
        y_vec = y[row, mask]
        # 增广加权设计矩阵 [sqrt(w) X; sqrt(ridge) I] 上做 QR/SVD 最小二乘（与 Rust 同构）
        sw = np.sqrt(w_vec)
        xa = x_mat * sw[:, None]
        ba = y_vec * sw
        sqrt_ridge = float(np.sqrt(MULTI_RIDGE_EPS))
        aug = np.vstack([xa, sqrt_ridge * np.eye(dim)])
        aug_b = np.concatenate([ba, np.zeros(dim, dtype=np.float64)])
        try:
            beta = np.linalg.lstsq(aug, aug_b, rcond=None)[0]
        except np.linalg.LinAlgError:
            continue
        if not np.all(np.isfinite(beta)):
            continue
        if want_beta:
            out[row, mask] = beta[0]
        else:
            pred = np.zeros(cnt, dtype=np.float64)
            for i in range(k):
                pred += beta[i] * x_mat[:, i]
            if intercept:
                pred += beta[k]
            out[row, mask] = y_vec - pred
    return out


def _ts_multi_core(
    y: np.ndarray,
    exps: list[np.ndarray],
    weights: np.ndarray | None,
    intercept: bool,
    want_beta: bool,
) -> np.ndarray:
    T, N = y.shape
    out = np.full(y.shape, np.nan, dtype=np.float64, order="F")
    k = len(exps)
    if k == 0:
        return out
    dim = k + (1 if intercept else 0)
    for col in range(N):
        a_mat = np.zeros((dim, dim), dtype=np.float64)
        b_vec = np.zeros(dim, dtype=np.float64)
        cnt = 0
        w_scale = 0.0
        for row in range(T):
            yv = y[row, col]
            if not np.isfinite(yv):
                continue
            ok = True
            for e in exps:
                if not np.isfinite(e[row, col]):
                    ok = False
                    break
            raw_w = 1.0
            if ok and weights is not None:
                raw_w = float(weights[row, col])
                if not np.isfinite(raw_w) or raw_w < 0.0:
                    ok = False
            if not ok:
                continue
            if weights is not None:
                w_scale = max(w_scale, raw_w)
                if not w_scale > 0.0:
                    continue
                wv = raw_w / w_scale
            else:
                wv = 1.0
            xs = np.array([e[row, col] for e in exps] + ([1.0] if intercept else []))
            a_mat += wv * np.outer(xs, xs)
            b_vec += wv * xs * yv
            cnt += 1
            if cnt < dim + 1:
                continue
            sol = _solve_normal(a_mat + MULTI_RIDGE_EPS * np.eye(dim), b_vec)
            if sol is None or not np.all(np.isfinite(sol)):
                continue
            if want_beta:
                out[row, col] = sol[0]
            else:
                pred = sum(sol[i] * exps[i][row, col] for i in range(k))
                if intercept:
                    pred += sol[k]
                out[row, col] = yv - pred
    return out


def cs_multi_beta(y: np.ndarray, *args) -> np.ndarray:
    """`cs_multi_beta(y, exp_1..exp_K [, intercept])`：首个暴露的截面偏回归系数。"""
    exps, _w, intercept = _split_multi_args(args)
    return _cs_multi_core(y, exps, None, intercept, True)


def cs_multi_resi(y: np.ndarray, *args) -> np.ndarray:
    """`cs_multi_resi(y, exp_1..exp_K [, intercept])`：横截面多元 OLS 残差。"""
    exps, _w, intercept = _split_multi_args(args)
    return _cs_multi_core(y, exps, None, intercept, False)


def cs_multi_wls_beta(y: np.ndarray, *args) -> np.ndarray:
    """`cs_multi_wls_beta(y, w, exp_1..exp_K [, intercept])`：加权截面偏回归系数。"""
    exps, w, intercept = _split_multi_args(args, weighted=True)
    return _cs_multi_core(y, exps, w, intercept, True)


def cs_multi_wls_resi(y: np.ndarray, *args) -> np.ndarray:
    """`cs_multi_wls_resi(y, w, exp_1..exp_K [, intercept])`：加权截面多元 WLS 残差。"""
    exps, w, intercept = _split_multi_args(args, weighted=True)
    return _cs_multi_core(y, exps, w, intercept, False)


def ts_multi_beta(y: np.ndarray, *args) -> np.ndarray:
    """`ts_multi_beta(y, exp_1..exp_K [, intercept])`：首个暴露的扩张窗口时序偏回归系数。"""
    exps, _w, intercept = _split_multi_args(args)
    return _ts_multi_core(y, exps, None, intercept, True)


def ts_multi_resi(y: np.ndarray, *args) -> np.ndarray:
    """`ts_multi_resi(y, exp_1..exp_K [, intercept])`：扩张窗口时序多元 OLS 残差。"""
    exps, _w, intercept = _split_multi_args(args)
    return _ts_multi_core(y, exps, None, intercept, False)


def ts_multi_wls_beta(y: np.ndarray, *args) -> np.ndarray:
    """`ts_multi_wls_beta(y, w, exp_1..exp_K [, intercept])`：加权时序偏回归系数。"""
    exps, w, intercept = _split_multi_args(args, weighted=True)
    return _ts_multi_core(y, exps, w, intercept, True)


def ts_multi_wls_resi(y: np.ndarray, *args) -> np.ndarray:
    """`ts_multi_wls_resi(y, w, exp_1..exp_K [, intercept])`：加权时序 WLS 残差。"""
    exps, w, intercept = _split_multi_args(args, weighted=True)
    return _ts_multi_core(y, exps, w, intercept, False)


def _split_multi_args(
    args: tuple, *, weighted: bool = False
) -> tuple[list[np.ndarray], np.ndarray | None, bool]:
    """拆分 `(y, [w], exps..., [intercept])` 参数（Python 参考实现侧）。"""
    items = list(args)
    intercept = False
    if items and np.isscalar(items[-1]):
        intercept = bool(float(items[-1]) != 0.0)
        items = items[:-1]
    w: np.ndarray | None = None
    if weighted:
        w = items[0]
        items = items[1:]
    exps = [np.asarray(e, dtype=np.float64) for e in items]
    return exps, w, intercept


# FactorLab MCP 1d 全量算子参考实现注册表扩充（与 Rust `ops/mod.rs` 一一对应）
OPS_REF_REGISTRY.update(
    {
        # 数学变换与缺失值处理 (8)
        "exp": exp_,
        "ceil": ceil,
        "sign_power": sign_power,
        "unique_count": unique_count,
        "is_inf": is_inf,
        "is_nan": is_nan,
        "purify": purify,
        "zero_to_nan": zero_to_nan,
        # 横截面统计扩充 (16)
        "cs_sum": cs_sum,
        "cs_mean": cs_mean,
        "cs_std": cs_std,
        "cs_var": cs_var,
        "cs_max": cs_max,
        "cs_min": cs_min,
        "cs_median": cs_median,
        "cs_nonan_num": cs_nonan_num,
        "cs_argmax": cs_argmax,
        "cs_argmin": cs_argmin,
        "cs_pct_rank": cs_pct_rank,
        "cs_dense_rank": cs_dense_rank,
        "cs_dense_pct_rank": cs_dense_pct_rank,
        "cs_skew": cs_skew,
        "cs_kurt": cs_kurt,
        "cs_mad": cs_mad,
        # 横截面双变量统计与市值加权标准化 (5)
        "cs_beta": cs_beta,
        "cs_corr": cs_corr,
        "cs_cov": cs_cov,
        "barra5_zScore": barra5_zScore,
        "ts_dw": ts_dw,
        # 常数参数算子 (7)
        "fillna": fillna,
        "round": round_decimals,
        "perc_long": perc_long,
        "cs_quantile": cs_quantile,
        "cs_group": cs_group,
        "winsor": pp_winsor_ref,
        "MAD": pp_mad_ref,
        # 时序窗口扩充 (6)
        "ts_roll": ts_roll,
        "ts_prod": ts_prod,
        "ts_median": ts_median,
        "ts_nonan_num": ts_nonan_num,
        "ts_pct_rank": ts_pct_rank,
        "ts_group": ts_group,
        # 时序三元分位 (1)
        "ts_quantile": ts_quantile,
        # 扩张窗口因果排名与累积 (9)
        "rank_ts": rank_ts,
        "pct_rank_ts": pct_rank_ts,
        "dense_rank_ts": dense_rank_ts,
        "pct_dense_rank_ts": pct_dense_rank_ts,
        "ts_cumprod": ts_cumprod,
        "ts_cumsum": ts_cumsum,
        "ts_ffill": ts_ffill,
        "ts_dense_rank": ts_dense_rank,
        "ts_pct_dense_rank": ts_pct_dense_rank,
        # 变参多元（加权）回归 (8)
        "cs_multi_beta": cs_multi_beta,
        "cs_multi_resi": cs_multi_resi,
        "cs_multi_wls_beta": cs_multi_wls_beta,
        "cs_multi_wls_resi": cs_multi_wls_resi,
        "ts_multi_beta": ts_multi_beta,
        "ts_multi_resi": ts_multi_resi,
        "ts_multi_wls_beta": ts_multi_wls_beta,
        "ts_multi_wls_resi": ts_multi_wls_resi,
    }
)
