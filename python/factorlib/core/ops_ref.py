"""68 个向量化算子的纯 NumPy 参考实现 (core/ops_ref.py)。

按照白皮书 16.2 节与 P2 验收标准设计：
- 本模块提供所有 68 个逐元素、时序滚动、横截面及板块分组算子的纯 Python/NumPy 黄金参考实现；
- 用于与 `factorlib_core` Rust 原生计算内核进行基于性质的随机矩阵对账测试（Property-based Testing）；
- 严格遵循相同的缺失值约定：所有非有限浮点数（`NaN`、`+Inf`、`-Inf`）统一视作 `NaN` 缺失值处理。
"""

from __future__ import annotations

from collections.abc import Callable

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
    out = np.full_like(row, np.nan)
    m = np.isfinite(row)
    idx = np.where(m)[0]
    k = len(idx)
    if k == 0:
        return out
    if k == 1:
        out[idx[0]] = 0.5
        return out
    vals = row[idx]
    for pos, col_j in enumerate(idx):
        v = vals[pos]
        tol = STD_EPS * max(1.0, abs(float(v)))
        c_eq = float(np.sum(np.abs(vals - v) <= tol))
        c_less = float(np.sum(vals < (v - tol)))
        out[col_j] = (c_less + 0.5 * (c_eq - 1.0)) / (k - 1.0)
    return out


def cs_rank(x: np.ndarray) -> np.ndarray:
    xc = _clean(x)
    T, N = xc.shape
    out = np.full((T, N), np.nan, dtype=np.float64, order="F")
    for t in range(T):
        out[t, :] = _rank_1d_normalized(xc[t, :])
    return out


def cs_demean(x: np.ndarray) -> np.ndarray:
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


def cs_zscore(x: np.ndarray) -> np.ndarray:
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


def cs_scale(x: np.ndarray) -> np.ndarray:
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


def cs_winsorize(x: np.ndarray) -> np.ndarray:
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


def cs_neutralize(y: np.ndarray, x: np.ndarray) -> np.ndarray:
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


def group_demean(x: np.ndarray, group: np.ndarray) -> np.ndarray:
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


def group_rank(x: np.ndarray, group: np.ndarray) -> np.ndarray:
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


def group_zscore(x: np.ndarray, group: np.ndarray) -> np.ndarray:
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
