"""A 股模块化融合组合优化器 (markets/stocks/optimizer.py)。

闭环定位：``Investment Research -> Positions``，把「预测信号 + Barra 风险模型」
转换为可下单的目标权重，并提供持仓漂移递推。本模块**自包含**（仅依赖 numpy 与
可选的 cvxpy/clarabel/mosek），不修改任何既有配置文件。

三层模块化架构
---------------

1. **信号塑形层** :class:`SignalShaper`
   逐日截面居中秩映射 -> 软行业中性化 -> 低波倾斜 -> 头部幂次放大 -> EMA 慢速记忆。
   除 EMA 状态外所有中间量只依赖当日截面，严格无未来函数。
2. **动态风控守卫** :class:`DynamicRiskGuard`
   监测 Barra ``SIZE`` 因子日收益的短期累计跌幅与滚动（仅历史）Z-Score，
   在微盘流动性危机期自适应放大低波惩罚、收缩头部幂次并注入大市值正向倾斜。
3. **凸优化求解层** :class:`ConvexOptimizer`
   Barra 42 因子低秩协方差 QP（可选双期 Horizon 规划），内嵌非对称买卖换手惩罚与
   防追高跳升门槛；求解器按 ``MOSEK -> CLARABEL -> OSQP`` 顺序探测，并实现
   Level1 硬约束 -> Level2 L1 松弛 -> Level3 真实持仓归一化保底的三级容错。
   无 cvxpy 环境可使用 ``force_numpy=True`` 的纯 NumPy 降级模式。

持仓漂移闭环
------------

:func:`drift_positions` 按真实价格变动递推昨日持仓；:class:`PositionDriftTracker`
逐步维护 ``last_hold``（严格只使用当期的成交价与收盘价，无未来信息）。

配置解析
--------

全部配置均为 ``@dataclass(frozen=True)``，并提供 ``from_dict`` 解析嵌套 YAML 字典
（未知键忽略、类型宽松转换、非法取值抛 ``ValueError``）。
"""

from __future__ import annotations

import contextlib
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, fields
from typing import Any

import numpy as np

try:  # pragma: no cover - 依赖探测分支
    import cvxpy as cp
except ImportError:  # pragma: no cover - 无 cvxpy 的纯 NumPy 环境
    cp = None  # type: ignore[assignment]

_EPS = 1e-12

#: Barra CNE5 十个风格因子（大写短名，与 ``BarraRiskModelStore.factors`` 对齐）
_STYLE_FACTORS: tuple[str, ...] = (
    "BETA",
    "BTOP",
    "EARNYILD",
    "GROWTH",
    "LEVERAGE",
    "LIQUIDTY",
    "MOMENTUM",
    "RESVOL",
    "SIZE",
    "SIZENL",
)

_SUSPEND_FIELD_CANDIDATES: tuple[str, ...] = (
    "is_suspended",
    "suspend_stock",
    "barra_suspend",
    "suspend",
)
_INDUSTRY_FIELD_CANDIDATES: tuple[str, ...] = (
    "barra_industry",
    "barra_sector",
    "industry_code",
    "sector",
)
_MKT_CAP_FIELD_CANDIDATES: tuple[str, ...] = ("mkt_cap", "market_cap", "float_mkt_cap")
_RESVOL_FIELD_CANDIDATES: tuple[str, ...] = ("barra_resvol", "resvol")
_SPEC_VAR_FIELD_CANDIDATES: tuple[str, ...] = (
    "spec_var",
    "barra_spec_var",
    "specific_variance",
    "specific_var",
)


# ==============================================================================
# 通用工具函数
# ==============================================================================


def _coerce_like(value: Any, default: Any) -> Any:
    """按默认值类型对 YAML 标量做宽松类型转换（用于 ``from_dict``）。"""
    if default is None:
        return value
    if isinstance(default, bool):
        return bool(value)
    if isinstance(default, int):
        return int(value)
    if isinstance(default, float):
        return float(value)
    if isinstance(default, tuple):
        if value is None:
            return None
        items = tuple(value)
        if items and all(isinstance(seed, int) for seed in default):
            return tuple(int(v) for v in items)
        if items and all(isinstance(seed, float) for seed in default):
            return tuple(float(v) for v in items)
        return items
    if isinstance(default, dict):
        return {str(k).upper(): float(v) for k, v in dict(value or {}).items()}
    if isinstance(default, str):
        return str(value)
    return value


def _kwargs_from_raw(raw: Mapping[str, Any], defaults: Any, names: Sequence[str]) -> dict[str, Any]:
    """从原始字典中提取已知键并按默认值类型转换，未知键忽略。"""
    out: dict[str, Any] = {}
    for name in names:
        if name in raw:
            out[name] = _coerce_like(raw[name], getattr(defaults, name))
    return out


def _pop_section(raw: dict[str, Any], keys: Sequence[str]) -> dict[str, Any]:
    """弹出并合并嵌套配置小节（兼容多个别名键）。"""
    merged: dict[str, Any] = {}
    for key in keys:
        value = raw.pop(key, None)
        if isinstance(value, Mapping):
            merged.update(dict(value))
    return merged


def _average_rank_1d(values: np.ndarray) -> np.ndarray:
    """一维平均秩（1-based），并列样本取平均名次。"""
    n = int(values.size)
    ranks = np.empty(n, dtype=np.float64)
    if n == 0:
        return ranks
    order = np.argsort(values, kind="mergesort")
    sorted_vals = values[order]
    boundaries = np.flatnonzero(sorted_vals[1:] != sorted_vals[:-1]) + 1
    starts = np.concatenate((np.array([0], dtype=np.int64), boundaries))
    ends = np.concatenate((boundaries, np.array([n], dtype=np.int64)))
    avg = 0.5 * (starts.astype(np.float64) + ends.astype(np.float64) + 1.0)
    ranks[order] = np.repeat(avg, ends - starts)
    return ranks


def centered_rank_scores(values: np.ndarray) -> np.ndarray:
    """单行居中秩映射 ``r = (2*rank(s) - (k+1)) / (k-1)``。

    - ``k`` 为当日有效（有限）样本数；缺失样本输出 0.0；
    - ``k <= 1`` 时整行为 0.0；
    - 采用平均秩处理并列，输出取值范围 ``[-1, 1]``。
    """
    arr = np.asarray(values, dtype=np.float64).ravel()
    out = np.zeros(arr.size, dtype=np.float64)
    valid = np.isfinite(arr)
    k = int(np.count_nonzero(valid))
    if k <= 1:
        return out
    ranks = _average_rank_1d(arr[valid])
    out[valid] = (2.0 * ranks - (k + 1.0)) / (k - 1.0)
    return out


def soft_industry_neutralize(
    ranks: np.ndarray,
    industry: np.ndarray | None,
    eta: float,
    *,
    valid: np.ndarray | None = None,
) -> np.ndarray:
    """软行业中性化 ``r' = r - eta * mean_{j in industry(i)}(r)``。

    行业编码缺失（NaN）的股票不施加行业均值修正；``eta`` 为 1.0 时组内均值为 0。
    ``valid`` 可显式指定参与行业均值计算的样本（例如剔除当日信号缺失的股票）。
    """
    r = np.asarray(ranks, dtype=np.float64).ravel().copy()
    if industry is None or float(eta) == 0.0:
        return r
    ind = np.asarray(industry, dtype=np.float64).ravel()
    if ind.size != r.size:
        raise ValueError(f"industry 长度 {ind.size} 与 ranks 长度 {r.size} 不一致")
    mask = np.isfinite(ind)
    if valid is not None:
        mask &= np.asarray(valid, dtype=bool).ravel()
    if not np.any(mask):
        return r
    targets = np.flatnonzero(mask)
    codes = np.rint(ind[mask]).astype(np.int64)
    for code in np.unique(codes):
        sel = targets[codes == code]
        if sel.size == 0:
            continue
        r[sel] -= float(eta) * float(np.mean(r[sel]))
    return r


def _demean_standardize(row: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """截面去均值并除以标准差（``center_rank=False`` 时的原始信号标准化分支）。"""
    out = np.zeros(row.size, dtype=np.float64)
    if not np.any(valid):
        return out
    vals = row[valid]
    mean = float(np.mean(vals))
    std = float(np.std(vals))
    if std > _EPS:
        out[valid] = (vals - mean) / std
    return out


def _as_row_matrix(values: Any, n_rows: int, n_cols: int, name: str) -> np.ndarray | None:
    """把一维截面数组广播为 (T, N) 行矩阵，并校验形状。"""
    if values is None:
        return None
    arr = np.asarray(values, dtype=np.float64)
    if arr.ndim == 1:
        if arr.size != n_cols:
            raise ValueError(f"{name} 长度 {arr.size} 与股票数 {n_cols} 不一致")
        arr = np.broadcast_to(arr, (n_rows, n_cols))
    if arr.shape != (n_rows, n_cols):
        raise ValueError(f"{name} 形状 {arr.shape} 与 (T={n_rows}, N={n_cols}) 不一致")
    return arr


def _first_field(panel: Any, candidates: Sequence[str]) -> np.ndarray | None:
    """按候选名顺序从面板 ``aux_fields``/``fields`` 中读取首个可用字段。"""
    for key in candidates:
        try:
            if panel.has_any_field(key):
                return panel.get_any_field(key)
        except (AttributeError, KeyError):  # pragma: no cover - 非标准面板对象
            continue
    return None


def _vector_or_none(value: Any, n_symbols: int, name: str) -> np.ndarray | None:
    """把可选的一维向量转换为 float64 数组并校验长度。"""
    if value is None:
        return None
    arr = np.asarray(value, dtype=np.float64).ravel()
    if arr.size != n_symbols:
        raise ValueError(f"{name} 长度 {arr.size} 与股票数 {n_symbols} 不一致")
    return arr


def _bool_vector(value: Any, n_symbols: int, name: str, default: bool) -> np.ndarray:
    """把可选的布尔掩码转换为定长 bool 数组。"""
    if value is None:
        return np.full(n_symbols, default, dtype=bool)
    arr = np.asarray(value)
    if arr.size != n_symbols:
        raise ValueError(f"{name} 长度 {arr.size} 与股票数 {n_symbols} 不一致")
    return arr.astype(bool).ravel().copy()


def _trailing_sum(values: np.ndarray, window: int) -> np.ndarray:
    """严格因果的尾部滚动求和（第 t 行为 ``values[max(0, t-window+1) : t+1]`` 之和）。"""
    n = values.size
    csum = np.concatenate((np.zeros(1, dtype=np.float64), np.cumsum(values, dtype=np.float64)))
    idx = np.arange(n)
    lo = np.maximum(0, idx - int(window) + 1)
    return csum[idx + 1] - csum[lo]


def _trailing_zscore(values: np.ndarray, window: int, *, min_obs: int = 20) -> np.ndarray:
    """严格因果的滚动 Z-Score：仅使用 ``t`` 之前（不含 ``t``）的 ``window`` 个观测。"""
    n = values.size
    out = np.zeros(n, dtype=np.float64)
    c1 = np.concatenate((np.zeros(1, dtype=np.float64), np.cumsum(values, dtype=np.float64)))
    c2 = np.concatenate(
        (np.zeros(1, dtype=np.float64), np.cumsum(values * values, dtype=np.float64))
    )
    for t in range(n):
        hi = t
        lo = max(0, t - int(window))
        count = hi - lo
        if count < int(min_obs):
            continue
        mean = (c1[hi] - c1[lo]) / float(count)
        var = (c2[hi] - c2[lo]) / float(count) - mean * mean
        std = float(np.sqrt(max(var, 0.0)))
        if std <= 1e-12:
            continue
        out[t] = (float(values[t]) - mean) / std
    return out


def _solve_ratio(total: float) -> float:
    """安全倒数：避免除零产生的 inf/nan。"""
    return 0.0 if abs(total) <= _EPS else 1.0 / total


# ==============================================================================
# 1) 信号塑形层
# ==============================================================================


@dataclass(frozen=True)
class SignalShaperConfig:
    """信号塑形配置。

    Attributes
    ----------
    center_rank:
        是否使用居中秩映射（``False`` 时改为原始截面标准化）。
    ind_neutral_weight:
        软行业中性化权重 ``eta_ind``，取值范围 ``[0, 1]``。
    vol_weight:
        低波倾斜权重 ``eta_vol``（非负）。
    power:
        头部幂次放大指数 ``p``，要求 ``>= 1.0``。
    memory:
        EMA 慢速记忆系数 ``mu``，取值范围 ``[0, 1)``。
    """

    center_rank: bool = True
    ind_neutral_weight: float = 0.42
    vol_weight: float = 0.12
    power: float = 1.8
    memory: float = 0.75

    def __post_init__(self) -> None:
        if not (0.0 <= float(self.ind_neutral_weight) <= 1.0):
            raise ValueError(
                f"shaper.ind_neutral_weight 必须在 [0, 1] 之间, 实际 {self.ind_neutral_weight}"
            )
        if float(self.vol_weight) < 0.0:
            raise ValueError(f"shaper.vol_weight 必须 >= 0, 实际 {self.vol_weight}")
        if float(self.power) < 1.0:
            raise ValueError(f"shaper.power 必须 >= 1.0, 实际 {self.power}")
        if not (0.0 <= float(self.memory) < 1.0):
            raise ValueError(f"shaper.memory 必须在 [0, 1) 之间, 实际 {self.memory}")

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any] | None) -> SignalShaperConfig:
        """解析嵌套 YAML 字典（未知键忽略，非法取值抛 ``ValueError``）。"""
        data = dict(raw or {})
        if isinstance(data.get("signal_shaper"), Mapping):
            data = dict(data["signal_shaper"])
        base = cls()
        names = tuple(f.name for f in fields(cls))
        return cls(**_kwargs_from_raw(data, base, names))


class SignalShaper:
    """信号塑形器（逐日截面处理，NaN 保留为 NaN，跨日仅维护 EMA 状态）。

    单日处理顺序（严格只依赖当日截面）：

    1. 居中秩映射 ``r = (2*rank(s) - (k+1)) / (k-1)``，缺失置 0.0，``k <= 1`` 整行为 0；
    2. 软行业中性化 ``r' = r - eta_ind * mean_{j in industry(i)}(r)``；
    3. 低波倾斜 ``a = rank_score(r') - eta_vol * rank_score(resvol)``；
    4. 幂次放大 ``alpha = sign(a) * |a|^p``；
    5. 风控可选注入 ``alpha += size_tilt * rank_score(size)``（默认 ``size_tilt=0``）；
    6. EMA 慢速记忆 ``state_t = mu*state_{t-1} + (1-mu)*alpha_t``，首日 ``state=alpha``。
    """

    def __init__(self, cfg: SignalShaperConfig | None = None) -> None:
        self.cfg = cfg if cfg is not None else SignalShaperConfig()
        self._state: np.ndarray | None = None

    def reset(self) -> None:
        """清空 EMA 慢速记忆状态（新一轮回测/换组合时调用）。"""
        self._state = None

    @property
    def state(self) -> np.ndarray | None:
        """返回 EMA 状态的副本；尚未处理过任何截面时返回 ``None``。"""
        return None if self._state is None else self._state.copy()

    def shape_scores(
        self,
        scores: np.ndarray,
        *,
        industry: np.ndarray | None = None,
        resvol: np.ndarray | None = None,
        vol_weight: float | None = None,
        power: float | None = None,
        size: np.ndarray | None = None,
        size_tilt: float = 0.0,
    ) -> np.ndarray:
        """对 ``(T, N)`` 原始信号逐日塑形并返回 ``(T, N)`` 目标 alpha（Fortran 顺序）。

        Parameters
        ----------
        scores:
            原始信号矩阵 ``(T, N)``（亦接受单日 ``(N,)``）。
        industry:
            行业编码矩阵 ``(T, N)`` 或单日 ``(N,)``；缺失用 NaN 表示。
        resvol:
            低波风格（RESVOL）暴露矩阵；缺失时低波倾斜项视为 0。
        vol_weight, power:
            可选的当日覆盖值（供 :class:`DynamicRiskGuard` 动态调节），
            默认取配置中的 ``vol_weight`` / ``power``。
        size:
            市值矩阵；仅在 ``size_tilt != 0`` 时用于注入大市值正向倾斜。
        size_tilt:
            大市值正向倾斜强度（防守体制下由风控守卫给出，默认 0）。
        """
        raw = np.asarray(scores, dtype=np.float64)
        if raw.ndim == 1:
            raw = raw.reshape(1, -1)
        if raw.ndim != 2:
            raise ValueError(f"scores 必须为 (T, N) 二维矩阵, 实际 {raw.shape}")
        n_rows, n_cols = raw.shape
        ind_mat = _as_row_matrix(industry, n_rows, n_cols, "industry")
        vol_mat = _as_row_matrix(resvol, n_rows, n_cols, "resvol")
        size_mat = _as_row_matrix(size, n_rows, n_cols, "size")

        eta_ind = float(self.cfg.ind_neutral_weight)
        eff_vol = max(0.0, float(self.cfg.vol_weight if vol_weight is None else vol_weight))
        eff_power = max(0.0, float(self.cfg.power if power is None else power))
        mu = float(self.cfg.memory)
        tilt = float(size_tilt)

        state = self._state
        if state is not None and state.size != n_cols:
            state = None
        out = np.full((n_rows, n_cols), np.nan, dtype=np.float64)
        for t in range(n_rows):
            row = raw[t]
            valid = np.isfinite(row)
            if not np.any(valid):
                alpha = np.zeros(n_cols, dtype=np.float64)
            else:
                # 步骤 1：居中秩映射（或原始标准化）
                if self.cfg.center_rank:
                    ranks = centered_rank_scores(row)
                else:
                    ranks = _demean_standardize(row, valid)
                # 步骤 2：软行业中性化
                neutral_ranks = soft_industry_neutralize(
                    ranks, ind_mat[t] if ind_mat is not None else None, eta_ind, valid=valid
                )
                # 步骤 3：低波倾斜
                tilted = centered_rank_scores(neutral_ranks)
                if vol_mat is not None and eff_vol > 0.0:
                    vol_ranks = centered_rank_scores(vol_mat[t])
                    tilted = tilted - eff_vol * vol_ranks
                # 步骤 4：头部幂次放大
                alpha = np.sign(tilted) * np.power(np.abs(tilted), eff_power)
                # 步骤 5：风控注入的大市值正向倾斜
                if size_mat is not None and tilt != 0.0:
                    alpha = alpha + tilt * centered_rank_scores(size_mat[t])
            # 步骤 6：EMA 慢速记忆（仅在有有效信号的样本上更新状态）
            if state is None:
                state = np.where(valid, alpha, 0.0)
            else:
                state = np.where(valid, mu * state + (1.0 - mu) * alpha, state)
            out[t] = np.where(valid, state, np.nan)
        self._state = state
        return np.asfortranarray(out, dtype=np.float64)


# ==============================================================================
# 2) 动态风控守卫
# ==============================================================================


@dataclass(frozen=True)
class DynamicRiskGuardConfig:
    """动态风控守卫配置。

    Attributes
    ----------
    enabled:
        是否启用风控守卫（关闭时全部交易日视为常态体制）。
    lookback_days:
        SIZE 累计收益的回看窗口（交易日，默认 10）。
    stress_z_threshold:
        滚动 Z-Score 冲击触发阈值（取负向：``z <= -threshold``）。
    cumret_threshold:
        SIZE 累计收益触发阈值（默认 -3%）。
    vol_multiplier:
        防守体制下低波倾斜权重的放大倍数。
    power_scale:
        防守体制下头部幂次的收缩倍数（``>= 0``）。
    size_tilt:
        防守体制下注入的大市值正向倾斜强度。
    zscore_window:
        Z-Score 的滚动窗口（默认 252 个交易日，仅使用历史样本）。
    """

    enabled: bool = True
    lookback_days: int = 10
    stress_z_threshold: float = 1.5
    cumret_threshold: float = -0.03
    vol_multiplier: float = 1.8
    power_scale: float = 0.85
    size_tilt: float = 0.15
    zscore_window: int = 252

    def __post_init__(self) -> None:
        if int(self.lookback_days) < 1:
            raise ValueError(f"risk_guard.lookback_days 必须 >= 1, 实际 {self.lookback_days}")
        if int(self.zscore_window) < 2:
            raise ValueError(f"risk_guard.zscore_window 必须 >= 2, 实际 {self.zscore_window}")
        if float(self.stress_z_threshold) <= 0.0:
            raise ValueError(
                f"risk_guard.stress_z_threshold 必须 > 0, 实际 {self.stress_z_threshold}"
            )
        if float(self.cumret_threshold) > 0.0:
            raise ValueError(f"risk_guard.cumret_threshold 必须 <= 0, 实际 {self.cumret_threshold}")
        if float(self.vol_multiplier) <= 0.0:
            raise ValueError(f"risk_guard.vol_multiplier 必须 > 0, 实际 {self.vol_multiplier}")
        if float(self.power_scale) < 0.0:
            raise ValueError(f"risk_guard.power_scale 必须 >= 0, 实际 {self.power_scale}")
        if float(self.size_tilt) < 0.0:
            raise ValueError(f"risk_guard.size_tilt 必须 >= 0, 实际 {self.size_tilt}")

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any] | None) -> DynamicRiskGuardConfig:
        """解析嵌套 YAML 字典（未知键忽略，非法取值抛 ``ValueError``）。"""
        data = dict(raw or {})
        for alias in ("risk_guard", "riskguard", "guard", "adaptive_stress_guard"):
            if isinstance(data.get(alias), Mapping):
                data = dict(data[alias])
                break
        base = cls()
        names = tuple(f.name for f in fields(cls))
        return cls(**_kwargs_from_raw(data, base, names))


@dataclass(frozen=True)
class RiskGuardState:
    """逐日风控体制状态（长度均为交易日数 ``T``）。

    Attributes
    ----------
    active:
        是否处于防守体制。
    stress_score:
        压力分数（正值表示承压；取负向 Z-Score 与累计跌幅比例的较大者）。
    vol_weight_scale:
        低波倾斜权重乘子（防守期 ``vol_multiplier``，否则 1.0）。
    power_scale:
        头部幂次乘子（防守期 ``power_scale``，否则 1.0）。
    size_tilt:
        希望注入的大市值正向倾斜强度（防守期 ``size_tilt``，否则 0.0）。
    """

    active: np.ndarray
    stress_score: np.ndarray
    vol_weight_scale: np.ndarray
    power_scale: np.ndarray
    size_tilt: np.ndarray

    def __post_init__(self) -> None:
        active = np.asarray(self.active, dtype=bool).ravel()
        stress = np.asarray(self.stress_score, dtype=np.float64).ravel()
        vol_scale = np.asarray(self.vol_weight_scale, dtype=np.float64).ravel()
        power_scale = np.asarray(self.power_scale, dtype=np.float64).ravel()
        tilt = np.asarray(self.size_tilt, dtype=np.float64).ravel()
        lengths = {active.size, stress.size, vol_scale.size, power_scale.size, tilt.size}
        if len(lengths) != 1:
            raise ValueError(f"RiskGuardState 各字段长度必须一致, 实际 {sorted(lengths)}")
        object.__setattr__(self, "active", active)
        object.__setattr__(self, "stress_score", stress)
        object.__setattr__(self, "vol_weight_scale", vol_scale)
        object.__setattr__(self, "power_scale", power_scale)
        object.__setattr__(self, "size_tilt", tilt)


class DynamicRiskGuard:
    """动态风控守卫：识别微盘流动性危机并给出当日塑形参数修正。

    判定逻辑（第 ``t`` 日严格只用 ``<= t`` 的数据）：

    - ``cum10_t``：SIZE 因子日收益的 ``lookback_days`` 日累计收益；
    - ``z_t``：``cum10`` 相对「此前 252 个交易日」样本的滚动 Z-Score（不含当日）；
    - ``cum10_t <= cumret_threshold`` 或 ``z_t <= -stress_z_threshold`` 时进入防守体制。
    """

    def __init__(
        self,
        cfg: DynamicRiskGuardConfig | None = None,
        *,
        factor_names: Sequence[str] | None = None,
    ) -> None:
        self.cfg = cfg if cfg is not None else DynamicRiskGuardConfig()
        self.factor_names = tuple(str(n) for n in factor_names) if factor_names else ()

    def assess(
        self,
        factor_returns: np.ndarray,
        dates: Sequence[Any] | None = None,
        *,
        factor_names: Sequence[str] | None = None,
    ) -> RiskGuardState:
        """评估 ``(T, K)`` Barra 因子日收益矩阵的逐日风控体制状态。

        ``SIZE`` 列按名字定位（大小写不敏感）；无法定位（缺失因子名）时
        ``active`` 全为 ``False``，即不触发防守体制。
        """
        fr = np.asarray(factor_returns, dtype=np.float64)
        if fr.ndim != 2:
            raise ValueError(f"factor_returns 必须为 (T, K) 二维矩阵, 实际 {fr.shape}")
        n_days, n_factors = fr.shape
        if dates is not None and len(dates) != n_days:
            raise ValueError(f"dates 长度 {len(dates)} 与 factor_returns 行数 {n_days} 不一致")
        names = tuple(
            str(n) for n in (factor_names if factor_names is not None else self.factor_names)
        )
        if len(names) != n_factors:
            names = ()

        size_idx: int | None = None
        for pos, name in enumerate(names):
            if name.strip().upper() == "SIZE":
                size_idx = pos
                break

        if not self.cfg.enabled or size_idx is None:
            return RiskGuardState(
                active=np.zeros(n_days, dtype=bool),
                stress_score=np.zeros(n_days, dtype=np.float64),
                vol_weight_scale=np.ones(n_days, dtype=np.float64),
                power_scale=np.ones(n_days, dtype=np.float64),
                size_tilt=np.zeros(n_days, dtype=np.float64),
            )

        size_ret = np.where(np.isfinite(fr[:, size_idx]), fr[:, size_idx], 0.0)
        cum_ret = _trailing_sum(size_ret, int(self.cfg.lookback_days))
        zscore = _trailing_zscore(cum_ret, int(self.cfg.zscore_window))
        threshold = float(self.cfg.cumret_threshold)
        stress_floor = max(abs(threshold), _EPS)
        active = (cum_ret <= threshold) | (zscore <= -abs(float(self.cfg.stress_z_threshold)))
        drop_score = np.where(cum_ret < 0.0, -cum_ret / stress_floor, 0.0)
        stress_score = np.maximum(-zscore, drop_score)
        return RiskGuardState(
            active=active,
            stress_score=stress_score,
            vol_weight_scale=np.where(active, float(self.cfg.vol_multiplier), 1.0),
            power_scale=np.where(active, float(self.cfg.power_scale), 1.0),
            size_tilt=np.where(active, float(self.cfg.size_tilt), 0.0),
        )

    def apply(
        self,
        state: RiskGuardState,
        t: int,
        shaper_cfg: SignalShaperConfig,
    ) -> tuple[float, float, float]:
        """返回第 ``t`` 日调整后的 ``(vol_weight, power, size_tilt)`` 三元组。

        不修改传入的 ``shaper_cfg``（frozen dataclass），返回值为新标量。
        """
        if not isinstance(state, RiskGuardState):
            raise TypeError(f"state 必须为 RiskGuardState, 实际 {type(state).__name__}")
        idx = int(t)
        if idx < 0 or idx >= state.active.size:
            raise ValueError(f"t={t} 超出 RiskGuardState 长度 {state.active.size}")
        vol_weight = float(shaper_cfg.vol_weight) * float(state.vol_weight_scale[idx])
        power = float(shaper_cfg.power) * float(state.power_scale[idx])
        return vol_weight, power, float(state.size_tilt[idx])


# ==============================================================================
# 3) 凸优化求解层
# ==============================================================================


@dataclass(frozen=True)
class ConvexOptimizerConfig:
    """单期多因子 QP / 双期 Horizon 规划的求解配置（全部带默认值）。"""

    benchmark_weight_mode: str = "mkt_cap_proxy"
    proxy_rank_range: tuple[int, int] | None = (301, 800)
    alpha_scale: float = 1.0
    risk_aversion: float = 2.5
    turnover_penalty: float = 0.0015
    include_specific_risk: bool = True
    cash_buffer_max: float = 0.02
    max_single_weight: float = 0.03
    max_active_weight: float = 0.015
    max_daily_turnover: float = 0.15
    style_deviation_default: float = 0.25
    style_deviation_overrides: dict[str, float] = field(
        default_factory=lambda: {"SIZE": 0.15, "BETA": 0.20}
    )
    industry_deviation: float = 0.02
    respect_t_close_suspend: bool = True
    horizon_mode: bool = False
    persistence: float = 0.35
    discount: float = 0.8
    buy_hurdle: bool = True
    hurdle_penalty: float = 0.45
    asymmetric_trade_penalty: tuple[float, float] = (1.00, 0.45)
    solver_priority: tuple[str, ...] = ("MOSEK", "CLARABEL", "OSQP")
    mosek_license_file: str | None = None
    slack_penalty: float = 1000.0
    allow_slack_fallback: bool = True
    max_iterations: int = 200

    def __post_init__(self) -> None:
        if self.benchmark_weight_mode not in ("constituent", "mkt_cap_proxy", "absolute_neutral"):
            raise ValueError(
                "optimizer.benchmark_weight_mode 必须为 'constituent' / 'mkt_cap_proxy' / "
                f"'absolute_neutral', 实际 {self.benchmark_weight_mode!r}"
            )
        if self.proxy_rank_range is not None:
            if len(self.proxy_rank_range) != 2:
                raise ValueError("optimizer.proxy_rank_range 必须为二元组或 None")
            lo, hi = int(self.proxy_rank_range[0]), int(self.proxy_rank_range[1])
            if lo < 1 or hi < lo:
                raise ValueError(f"optimizer.proxy_rank_range 非法: {self.proxy_rank_range!r}")
            object.__setattr__(self, "proxy_rank_range", (lo, hi))
        if float(self.alpha_scale) < 0.0:
            raise ValueError(f"optimizer.alpha_scale 必须 >= 0, 实际 {self.alpha_scale}")
        if float(self.risk_aversion) < 0.0:
            raise ValueError(f"optimizer.risk_aversion 必须 >= 0, 实际 {self.risk_aversion}")
        if float(self.turnover_penalty) < 0.0:
            raise ValueError(f"optimizer.turnover_penalty 必须 >= 0, 实际 {self.turnover_penalty}")
        if not (0.0 <= float(self.cash_buffer_max) < 1.0):
            raise ValueError(
                f"optimizer.cash_buffer_max 必须在 [0, 1) 之间, 实际 {self.cash_buffer_max}"
            )
        if not (0.0 < float(self.max_single_weight) <= 1.0):
            raise ValueError(
                f"optimizer.max_single_weight 必须在 (0, 1] 之间, 实际 {self.max_single_weight}"
            )
        if float(self.max_active_weight) < 0.0:
            raise ValueError(
                f"optimizer.max_active_weight 必须 >= 0, 实际 {self.max_active_weight}"
            )
        if float(self.max_daily_turnover) <= 0.0:
            raise ValueError(
                f"optimizer.max_daily_turnover 必须 > 0, 实际 {self.max_daily_turnover}"
            )
        if float(self.style_deviation_default) < 0.0:
            raise ValueError("optimizer.style_deviation_default 必须 >= 0")
        if float(self.industry_deviation) < 0.0:
            raise ValueError("optimizer.industry_deviation 必须 >= 0")
        if not (0.0 <= float(self.persistence) < 1.0):
            raise ValueError(f"optimizer.persistence 必须在 [0, 1) 之间, 实际 {self.persistence}")
        if not (0.0 < float(self.discount) <= 1.0):
            raise ValueError(f"optimizer.discount 必须在 (0, 1] 之间, 实际 {self.discount}")
        if float(self.hurdle_penalty) < 0.0:
            raise ValueError(f"optimizer.hurdle_penalty 必须 >= 0, 实际 {self.hurdle_penalty}")
        trade_pen = tuple(self.asymmetric_trade_penalty)
        if len(trade_pen) != 2:
            raise ValueError("optimizer.asymmetric_trade_penalty 必须为二元组 (buy, sell)")
        if min(float(trade_pen[0]), float(trade_pen[1])) < 0.0:
            raise ValueError("optimizer.asymmetric_trade_penalty 两个分量都必须 >= 0")
        object.__setattr__(
            self, "asymmetric_trade_penalty", (float(trade_pen[0]), float(trade_pen[1]))
        )
        overrides = {str(k).upper(): float(v) for k, v in self.style_deviation_overrides.items()}
        if any(v < 0.0 for v in overrides.values()):
            raise ValueError("optimizer.style_deviation_overrides 的取值必须 >= 0")
        object.__setattr__(self, "style_deviation_overrides", overrides)
        solvers = tuple(str(s).strip().upper() for s in self.solver_priority if str(s).strip())
        if not solvers:
            raise ValueError("optimizer.solver_priority 不能为空")
        object.__setattr__(self, "solver_priority", solvers)
        if self.mosek_license_file is not None and not isinstance(self.mosek_license_file, str):
            raise ValueError("optimizer.mosek_license_file 必须为字符串或 None")
        if float(self.slack_penalty) <= 0.0:
            raise ValueError(f"optimizer.slack_penalty 必须 > 0, 实际 {self.slack_penalty}")
        if int(self.max_iterations) < 1:
            raise ValueError(f"optimizer.max_iterations 必须 >= 1, 实际 {self.max_iterations}")

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any] | None) -> ConvexOptimizerConfig:
        """解析嵌套 YAML 字典（兼容 ``convex`` / ``optimizer`` 子节，未知键忽略）。"""
        data = dict(raw or {})
        for alias in ("convex", "optimizer", "convex_optimizer", "solve"):
            if isinstance(data.get(alias), Mapping):
                data = dict(data[alias])
                break
        base = cls()
        return cls(**_kwargs_from_raw(data, base, _CONVEX_FIELD_NAMES))


@dataclass
class OptimizerResult:
    """单日优化求解结果。

    Attributes
    ----------
    weights:
        ``(N,)`` 目标权重（纯多头、已满足求解后的可行性截断）。
    objective:
        目标函数值（最大化方向；NumPy 降级路径为解析复算值）。
    solver_used:
        实际使用的求解器名称（``MOSEK`` / ``CLARABEL`` / ``OSQP`` / ``NUMPY`` / ``FALLBACK``）。
    fallback_level:
        容错层级：1=硬约束求解成功，2=引入 L1 松弛，3=持仓归一化保底，0=纯 NumPy 降级。
    status:
        求解状态字符串（``optimal`` / ``fallback`` / ``optimal_numpy`` 等）。
    diagnostics:
        诊断信息（现金缓冲、换手、风格偏离、初始建仓豁免等）。
    """

    weights: np.ndarray
    objective: float
    solver_used: str
    fallback_level: int
    status: str
    diagnostics: dict[str, Any] = field(default_factory=dict)


_SOLVER_CACHE: tuple[str, ...] | None = None


def available_cvxpy_solvers() -> tuple[str, ...]:
    """返回当前环境 cvxpy 已安装的求解器名称元组；cvxpy 缺失时返回空元组。"""
    global _SOLVER_CACHE
    if _SOLVER_CACHE is None:
        if cp is None:  # pragma: no cover - 由测试通过 monkeypatch 覆盖
            _SOLVER_CACHE = ()
        else:
            try:
                _SOLVER_CACHE = tuple(str(s) for s in cp.installed_solvers())
            except Exception:  # pragma: no cover - 求解器插件异常
                _SOLVER_CACHE = ()
    return _SOLVER_CACHE


@dataclass(frozen=True)
class _SolveContext:
    """单日求解的规范化输入上下文（内部使用）。"""

    alpha: np.ndarray
    alpha_ema: np.ndarray | None
    w_prev: np.ndarray
    wb: np.ndarray | None
    w_base: np.ndarray
    exposures: np.ndarray | None
    style_columns: tuple[int, ...]
    style_names: tuple[str, ...]
    industry: np.ndarray | None
    spec_var: np.ndarray | None
    factor_cov_sqrt: np.ndarray | None
    mask_investable: np.ndarray
    mask_locked: np.ndarray
    forecast: np.ndarray | None
    cash_min: float
    neutral: bool
    use_active_bound: bool
    horizon: bool
    initial_build: bool
    max_single: float
    max_active: float

    @property
    def n_symbols(self) -> int:
        return int(self.alpha.size)


class ConvexOptimizer:
    """凸优化求解器（Barra 低秩协方差 QP，可选双期 Horizon 规划）。

    目标函数（单期，最大化）::

        alpha_scale * alpha^T w
        - buy_pen * sum(b) - sell_pen * sum(s)
        - hurdle_penalty * sum(max(alpha - ema_alpha, 0) * b)
        - risk_aversion/2 * (||L^T X^T (w - wb)||^2 + sum(spec_var * (w - wb)^2))

    其中 ``b, s >= 0`` 且 ``b - s = w - w_prev``。约束覆盖满仓/现金缓冲、单票上限、
    主动权重上限、风格/行业偏离、换手上限、停牌与不可投资锁定。
    """

    def __init__(
        self,
        cfg: ConvexOptimizerConfig | None = None,
        *,
        force_numpy: bool = False,
    ) -> None:
        self.cfg = cfg if cfg is not None else ConvexOptimizerConfig()
        self.force_numpy = bool(force_numpy)

    # ------------------------------------------------------------------
    # 公开求解入口
    # ------------------------------------------------------------------
    def solve(
        self,
        alpha: np.ndarray,
        *,
        w_prev: np.ndarray | None = None,
        wb: np.ndarray | None = None,
        exposures: np.ndarray | None = None,
        industry: np.ndarray | None = None,
        spec_var: np.ndarray | None = None,
        factor_cov_sqrt: np.ndarray | None = None,
        style_columns: Sequence[int] | None = None,
        style_names: Sequence[str] | None = None,
        alpha_ema: np.ndarray | None = None,
        mask_active: np.ndarray | None = None,
        mask_suspend: np.ndarray | None = None,
        forecast: np.ndarray | None = None,
        cash_buffer: float | None = None,
    ) -> OptimizerResult:
        """求解单日目标权重。

        Parameters
        ----------
        alpha:
            ``(N,)`` 当日预期收益/塑形后信号；非有限处视为不可投资。
        w_prev:
            漂移后的真实持仓 ``(N,)``（昨日收盘盯市后的权重）。
        wb:
            基准权重 ``(N,)``；``None`` 或 ``absolute_neutral`` 模式时按绝对中性处理。
        exposures:
            ``(N, K)`` 因子暴露矩阵（与 ``factor_cov_sqrt``/``style_columns`` 对齐）。
        industry:
            ``(N,)`` 行业编码；缺失用 NaN 表示。
        spec_var:
            ``(N,)`` 特质方差（日频方差量纲）。
        factor_cov_sqrt:
            ``(K, K)`` 因子协方差平方根 ``L``，满足 ``L L^T = F``。
        style_columns:
            风格因子在 ``exposures`` 中的列下标（默认全部列）。
        style_names:
            与 ``exposures`` 列或 ``style_columns`` 对齐的风格名（用于偏离覆盖）。
        alpha_ema:
            alpha 的慢速 EMA（用于防追高跳升门槛）。
        mask_active:
            可投资掩码（universe）；``None`` 表示全部可投资。
        mask_suspend:
            停牌掩码；受 ``respect_t_close_suspend`` 控制是否锁定。
        forecast:
            第二期（T+1）预测信号，仅在 ``horizon_mode`` 且 ``risk_aversion == 0`` 时生效。
        cash_buffer:
            现金缓冲下界（默认 0），上界为 ``cfg.cash_buffer_max``。
        """
        ctx = self._build_context(
            alpha,
            w_prev=w_prev,
            wb=wb,
            exposures=exposures,
            industry=industry,
            spec_var=spec_var,
            factor_cov_sqrt=factor_cov_sqrt,
            style_columns=style_columns,
            style_names=style_names,
            alpha_ema=alpha_ema,
            mask_active=mask_active,
            mask_suspend=mask_suspend,
            forecast=forecast,
            cash_buffer=cash_buffer,
        )
        if self.force_numpy:
            return self._solve_numpy(ctx)
        if cp is None:
            raise RuntimeError(
                "当前环境缺少 cvxpy，无法执行凸优化求解；"
                '请执行 pip install -e ".[barra]" 安装 Barra 优化依赖，'
                "或使用 BarraPortfolioOptimizer(..., force_numpy=True) 走纯 NumPy 降级模式。"
            )
        return self._solve_cvxpy(ctx)

    # ------------------------------------------------------------------
    # 输入规范化
    # ------------------------------------------------------------------
    def _build_context(
        self,
        alpha: Any,
        *,
        w_prev: Any,
        wb: Any,
        exposures: Any,
        industry: Any,
        spec_var: Any,
        factor_cov_sqrt: Any,
        style_columns: Sequence[int] | None,
        style_names: Sequence[str] | None,
        alpha_ema: Any,
        mask_active: Any,
        mask_suspend: Any,
        forecast: Any,
        cash_buffer: float | None,
    ) -> _SolveContext:
        alpha_arr = np.asarray(alpha, dtype=np.float64).ravel()
        n = int(alpha_arr.size)
        if n == 0:
            raise ValueError("alpha 不能为空")

        prev = _vector_or_none(w_prev, n, "w_prev")
        if prev is None:
            prev = np.zeros(n, dtype=np.float64)
        prev = np.where(np.isfinite(prev), np.clip(prev, 0.0, None), 0.0)

        active = _bool_vector(mask_active, n, "mask_active", True)
        investable = active & np.isfinite(alpha_arr)
        alpha_clean = np.where(investable, alpha_arr, 0.0)

        suspend = _bool_vector(mask_suspend, n, "mask_suspend", False)
        if self.cfg.respect_t_close_suspend:
            locked = (~investable) | suspend
        else:
            locked = ~investable

        wb_arr = _vector_or_none(wb, n, "wb")
        if wb_arr is not None:
            wb_arr = np.where(np.isfinite(wb_arr), wb_arr, 0.0)
        neutral = self.cfg.benchmark_weight_mode == "absolute_neutral" or wb_arr is None
        w_base = np.zeros(n, dtype=np.float64) if neutral else wb_arr.copy()

        ema_arr = _vector_or_none(alpha_ema, n, "alpha_ema")
        if ema_arr is not None:
            ema_arr = np.where(np.isfinite(ema_arr), ema_arr, 0.0)

        expo: np.ndarray | None = None
        style_cols: tuple[int, ...] = ()
        col_names: tuple[str, ...] = ()
        if exposures is not None:
            arr = np.asarray(exposures, dtype=np.float64)
            if arr.ndim == 1:
                arr = arr.reshape(-1, 1)
            if arr.ndim != 2 or arr.shape[0] != n:
                raise ValueError(f"exposures 形状非法: {arr.shape}, 期望 (N={n}, K)")
            if arr.shape[1] > 0:
                expo = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
        if expo is not None:
            if style_columns is None:
                style_cols = tuple(range(expo.shape[1]))
            else:
                style_cols = tuple(int(c) for c in style_columns)
                if any(c < 0 or c >= expo.shape[1] for c in style_cols):
                    raise ValueError("style_columns 存在越界下标")
            raw_names = tuple(str(x) for x in style_names) if style_names else ()
            if len(raw_names) == expo.shape[1]:
                col_names = tuple(raw_names[c].upper() for c in style_cols)
            elif len(raw_names) == len(style_cols):
                col_names = tuple(x.upper() for x in raw_names)
            else:
                col_names = tuple(f"STYLE_{c}" for c in style_cols)

        ind_arr: np.ndarray | None = None
        if industry is not None:
            raw_ind = np.asarray(industry, dtype=np.float64).ravel()
            if raw_ind.size != n:
                raise ValueError(f"industry 长度 {raw_ind.size} 与股票数 {n} 不一致")
            ind_arr = np.where(np.isfinite(raw_ind), np.rint(raw_ind), -1.0)

        sv: np.ndarray | None = None
        if spec_var is not None and self.cfg.include_specific_risk:
            raw_sv = np.asarray(spec_var, dtype=np.float64).ravel()
            if raw_sv.size != n:
                raise ValueError(f"spec_var 长度 {raw_sv.size} 与股票数 {n} 不一致")
            sv = np.clip(np.where(np.isfinite(raw_sv), raw_sv, 0.0), 0.0, None)

        cov_sqrt: np.ndarray | None = None
        if factor_cov_sqrt is not None and expo is not None and self.cfg.risk_aversion > 0.0:
            arr = np.asarray(factor_cov_sqrt, dtype=np.float64)
            if arr.shape != (expo.shape[1], expo.shape[1]):
                raise ValueError(
                    f"factor_cov_sqrt 形状 {arr.shape} 必须为 ({expo.shape[1]}, {expo.shape[1]})"
                )
            cov_sqrt = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)

        fc: np.ndarray | None = None
        if forecast is not None:
            fc = _vector_or_none(forecast, n, "forecast")
            if fc is not None:
                fc = np.where(np.isfinite(fc), fc, 0.0)

        cash_min = 0.0 if cash_buffer is None else float(cash_buffer)
        if cash_min < 0.0:
            raise ValueError(f"cash_buffer 必须 >= 0, 实际 {cash_min}")
        if cash_min > float(self.cfg.cash_buffer_max) + 1e-12:
            raise ValueError(
                f"cash_buffer={cash_min} 超过 optimizer.cash_buffer_max={self.cfg.cash_buffer_max}"
            )

        horizon = (
            bool(self.cfg.horizon_mode) and float(self.cfg.risk_aversion) == 0.0 and fc is not None
        )
        initial_build = float(np.sum(np.abs(prev))) <= 1e-12
        return _SolveContext(
            alpha=alpha_clean,
            alpha_ema=ema_arr,
            w_prev=prev,
            wb=wb_arr,
            w_base=w_base,
            exposures=expo,
            style_columns=style_cols,
            style_names=col_names,
            industry=ind_arr,
            spec_var=sv,
            factor_cov_sqrt=cov_sqrt,
            mask_investable=investable,
            mask_locked=locked,
            forecast=fc,
            cash_min=cash_min,
            neutral=neutral,
            use_active_bound=(not neutral) and wb_arr is not None,
            horizon=horizon,
            initial_build=initial_build,
            max_single=float(self.cfg.max_single_weight),
            max_active=float(self.cfg.max_active_weight),
        )

    # ------------------------------------------------------------------
    # 纯 NumPy 降级路径
    # ------------------------------------------------------------------
    def _solve_numpy(self, ctx: _SolveContext) -> OptimizerResult:
        """信号排序 + 上限截断 + 逐次归一化的纯 NumPy 目标权重构造。"""
        cash = min(max(ctx.cash_min, 0.0), float(self.cfg.cash_buffer_max))
        invest = 1.0 - cash
        prev = ctx.w_prev
        w = prev.copy()
        free = ~ctx.mask_locked
        turnover_clipped = False
        if np.any(free):
            target_free = invest - float(np.sum(w[ctx.mask_locked]))
            target_free = max(0.0, min(target_free, float(np.count_nonzero(free)) * ctx.max_single))
            pref = _preference_scores(ctx.alpha[free])
            wb_free = None if (ctx.neutral or ctx.wb is None) else ctx.wb[free]
            if wb_free is not None and ctx.use_active_bound:
                lo = np.maximum(0.0, wb_free - ctx.max_active)
                hi = np.minimum(ctx.max_single, wb_free + ctx.max_active)
            else:
                lo = np.zeros(int(np.count_nonzero(free)), dtype=np.float64)
                hi = np.full(int(np.count_nonzero(free)), ctx.max_single, dtype=np.float64)
            lo = np.minimum(lo, hi)
            w[free] = _allocate_with_bounds(pref, lo, hi, target_free)
        else:
            w[:] = prev
        if not ctx.initial_build:
            diff = w - prev
            gross = float(np.sum(np.abs(diff)))
            if gross > float(self.cfg.max_daily_turnover) and gross > _EPS:
                w = prev + (float(self.cfg.max_daily_turnover) / gross) * diff
                turnover_clipped = True
        w = np.clip(np.where(np.isfinite(w), w, 0.0), 0.0, ctx.max_single)
        if np.any(ctx.mask_locked):
            w[ctx.mask_locked] = ctx.w_prev[ctx.mask_locked]
        w = np.ascontiguousarray(w, dtype=np.float64)
        return OptimizerResult(
            weights=w,
            objective=self._objective_value(ctx, w),
            solver_used="NUMPY",
            fallback_level=0,
            status="optimal_numpy",
            diagnostics=self._diagnostics(
                ctx,
                w,
                cash=cash,
                solver="NUMPY",
                status="optimal_numpy",
                numpy_mode=True,
                turnover_clipped=turnover_clipped,
            ),
        )

    # ------------------------------------------------------------------
    # cvxpy 求解与三级容错
    # ------------------------------------------------------------------
    def _solve_cvxpy(self, ctx: _SolveContext) -> OptimizerResult:
        solvers = self._solver_order()
        attempts: list[str] = []
        last_error = "无可用求解器"
        for level in (1, 2):
            if level == 2 and not self.cfg.allow_slack_fallback:
                break
            problem, variables = self._build_problem(ctx, relaxed=(level == 2))
            for solver in solvers:
                attempts.append(solver)
                try:
                    with self._mosek_license_env():
                        problem.solve(solver=solver, **self._solver_kwargs(solver))
                except Exception as exc:  # noqa: BLE001 - 逐求解器容错
                    last_error = f"{solver}: {type(exc).__name__}: {exc}"
                    continue
                status = str(problem.status)
                if status in ("optimal", "optimal_inaccurate"):
                    cash = self._cash_value(variables, ctx)
                    weights = self._postprocess_weights(variables["w"].value, ctx)
                    tolerance = 1e-4
                    if level == 1 and not self._feasible_within_tolerance(
                        ctx, weights, cash, tol=tolerance
                    ):
                        last_error = f"{solver}: 解未满足硬约束, 降级至松弛层级"
                        continue
                    if abs(float(np.sum(weights)) - (1.0 - cash)) > tolerance:
                        last_error = (
                            f"{solver}: 权重和 {float(np.sum(weights)):.6f} 与满仓目标不一致"
                        )
                        continue
                    value = float(problem.value) if problem.value is not None else float("nan")
                    return OptimizerResult(
                        weights=weights,
                        objective=value,
                        solver_used=solver,
                        fallback_level=level,
                        status=status,
                        diagnostics=self._diagnostics(
                            ctx,
                            weights,
                            cash=cash,
                            solver=solver,
                            status=status,
                            slack=self._slack_total(variables["slacks"]),
                            extra={"attempts": tuple(attempts)},
                        ),
                    )
                last_error = f"{solver}: status={status}"
        return self._fallback_result(ctx, reason=last_error, attempts=tuple(attempts))

    def _solver_order(self) -> list[str]:
        """按配置优先级筛选当前环境已安装的 cvxpy 求解器。"""
        installed = set(available_cvxpy_solvers())
        order = [s for s in self.cfg.solver_priority if s in installed]
        if not order:
            for fallback in ("CLARABEL", "SCS", "OSQP", "SCIPY", "HIGHS"):
                if fallback in installed:
                    order.append(fallback)
                    break
        return order

    def _solver_kwargs(self, solver: str) -> dict[str, Any]:
        """构造求解器专属参数（MOSEK 强制单线程）。"""
        if solver == "MOSEK":
            return {"mosek_params": {"MSK_IPAR_NUM_THREADS": 1}}
        if solver == "CLARABEL":
            return {"max_iter": int(self.cfg.max_iterations)}
        if solver == "OSQP":
            return {"max_iter": int(self.cfg.max_iterations), "eps_abs": 1e-8, "eps_rel": 1e-8}
        if solver == "SCS":
            return {"max_iters": int(self.cfg.max_iterations)}
        return {}

    @contextlib.contextmanager
    def _mosek_license_env(self) -> Any:
        """在求解期间临时指向 ``MOSEKLM_LICENSE_FILE``（退出后还原）。"""
        path = self.cfg.mosek_license_file
        if path is None:
            yield
            return
        previous = os.environ.get("MOSEKLM_LICENSE_FILE")
        os.environ["MOSEKLM_LICENSE_FILE"] = str(path)
        try:
            yield
        finally:
            if previous is None:
                os.environ.pop("MOSEKLM_LICENSE_FILE", None)
            else:
                os.environ["MOSEKLM_LICENSE_FILE"] = previous

    def _build_problem(self, ctx: _SolveContext, *, relaxed: bool) -> tuple[Any, dict[str, Any]]:
        """构造 cvxpy 问题；``relaxed=True`` 时引入风格/行业/换手的 L1 松弛。"""
        n = ctx.n_symbols
        w = cp.Variable(n)
        buy = cp.Variable(n, nonneg=True)
        sell = cp.Variable(n, nonneg=True)
        cash = cp.Variable()
        constraints: list[Any] = [
            cp.sum(w) == 1.0 - cash,
            cash >= ctx.cash_min,
            cash <= float(self.cfg.cash_buffer_max),
            w >= 0.0,
            w <= ctx.max_single,
            buy - sell == w - ctx.w_prev,
        ]
        style_slacks: list[Any] = []
        industry_slacks: list[Any] = []
        turnover_slacks: list[Any] = []
        locked = np.flatnonzero(ctx.mask_locked)
        if locked.size > 0:
            constraints.append(w[locked] == ctx.w_prev[locked])
        if ctx.use_active_bound and ctx.wb is not None:
            constraints.append(cp.abs(w - ctx.wb) <= ctx.max_active)
        self._append_exposure_constraints(
            constraints, w, ctx, relaxed, style_slacks, industry_slacks
        )
        if not ctx.initial_build:
            if relaxed:
                turnover_slack = cp.Variable(nonneg=True)
                constraints.append(
                    cp.sum(buy + sell) <= float(self.cfg.max_daily_turnover) + turnover_slack
                )
                turnover_slacks.append(turnover_slack)
            else:
                constraints.append(cp.sum(buy + sell) <= float(self.cfg.max_daily_turnover))

        objective: Any = float(self.cfg.alpha_scale) * cp.sum(cp.multiply(ctx.alpha, w))
        buy_pen, sell_pen = self.cfg.asymmetric_trade_penalty
        objective = objective - buy_pen * cp.sum(buy) - sell_pen * cp.sum(sell)
        if ctx.alpha_ema is not None and self.cfg.buy_hurdle and float(self.cfg.hurdle_penalty) > 0:
            jump = np.maximum(ctx.alpha - ctx.alpha_ema, 0.0)
            if np.any(jump > 0.0):
                objective = objective - float(self.cfg.hurdle_penalty) * cp.sum(
                    cp.multiply(jump, buy)
                )
        if float(self.cfg.risk_aversion) > 0.0:
            objective = objective - 0.5 * float(self.cfg.risk_aversion) * self._risk_expression(
                w, ctx
            )

        w2: Any = None
        if ctx.horizon:
            w2 = cp.Variable(n)
            constraints.extend(
                [
                    cp.sum(w2) == 1.0 - cash,
                    w2 >= 0.0,
                    w2 <= ctx.max_single,
                ]
            )
            if locked.size > 0:
                constraints.append(w2[locked] == w[locked])
            if ctx.use_active_bound and ctx.wb is not None:
                constraints.append(cp.abs(w2 - ctx.wb) <= ctx.max_active)
            self._append_exposure_constraints(
                constraints, w2, ctx, relaxed, style_slacks, industry_slacks
            )
            if relaxed:
                turnover_slack_2 = cp.Variable(nonneg=True)
                constraints.append(
                    cp.norm1(w2 - w) <= float(self.cfg.max_daily_turnover) + turnover_slack_2
                )
                turnover_slacks.append(turnover_slack_2)
            else:
                constraints.append(cp.norm1(w2 - w) <= float(self.cfg.max_daily_turnover))
            objective = objective + float(self.cfg.discount) * (
                cp.sum(cp.multiply(ctx.forecast, w2))
                - float(self.cfg.turnover_penalty) * cp.norm1(w2 - w)
            )

        slack_terms = style_slacks + industry_slacks + turnover_slacks
        if slack_terms:
            slack_total = slack_terms[0]
            for term in slack_terms[1:]:
                slack_total = slack_total + term
            objective = objective - float(self.cfg.slack_penalty) * slack_total

        problem = cp.Problem(cp.Maximize(objective), constraints)
        return problem, {"w": w, "w2": w2, "cash": cash, "slacks": slack_terms}

    def _append_exposure_constraints(
        self,
        constraints: list[Any],
        var: Any,
        ctx: _SolveContext,
        relaxed: bool,
        style_slacks: list[Any],
        industry_slacks: list[Any],
    ) -> None:
        """追加风格与行业暴露偏离约束（可带 L1 松弛）。"""
        if ctx.exposures is not None:
            for pos, col in enumerate(ctx.style_columns):
                delta = self._style_delta(ctx.style_names[pos])
                dev = cp.sum(cp.multiply(ctx.exposures[:, col], var - ctx.w_base))
                if relaxed:
                    slack = cp.Variable(nonneg=True)
                    constraints.append(dev <= delta + slack)
                    constraints.append(dev >= -delta - slack)
                    style_slacks.append(slack)
                else:
                    constraints.append(dev <= delta)
                    constraints.append(dev >= -delta)
        if ctx.industry is not None:
            valid_codes = np.unique(ctx.industry[ctx.industry >= 0.0])
            delta_ind = float(self.cfg.industry_deviation)
            for code in valid_codes:
                indicator = (ctx.industry == code).astype(np.float64)
                dev = cp.sum(cp.multiply(indicator, var - ctx.w_base))
                if relaxed:
                    slack = cp.Variable(nonneg=True)
                    constraints.append(dev <= delta_ind + slack)
                    constraints.append(dev >= -delta_ind - slack)
                    industry_slacks.append(slack)
                else:
                    constraints.append(dev <= delta_ind)
                    constraints.append(dev >= -delta_ind)

    def _style_delta(self, name: str) -> float:
        """取风格偏离上限（``style_deviation_overrides`` 覆盖默认值）。"""
        overrides = self.cfg.style_deviation_overrides
        return float(overrides.get(str(name).upper(), self.cfg.style_deviation_default))

    def _risk_expression(self, var: Any, ctx: _SolveContext) -> Any:
        """构造低秩因子风险 + 对角特质风险二次型 ``||L^T y||^2 + sum(spec_var*resid^2)``。"""
        expr: Any = 0.0
        residual = var - ctx.w_base
        if ctx.exposures is not None:
            y = ctx.exposures.T @ residual
            if ctx.factor_cov_sqrt is not None:
                expr = expr + cp.sum_squares(ctx.factor_cov_sqrt.T @ y)
            if ctx.spec_var is not None:
                expr = expr + cp.sum(cp.multiply(ctx.spec_var, cp.square(residual)))
        elif ctx.spec_var is not None:
            expr = expr + cp.sum(cp.multiply(ctx.spec_var, cp.square(residual)))
        return expr

    def _cash_value(self, variables: dict[str, Any], ctx: _SolveContext) -> float:
        """读取求解后的现金缓冲并裁剪到合法区间。"""
        cash = variables["cash"].value
        if cash is None:
            return float(ctx.cash_min)
        return float(
            np.clip(float(np.asarray(cash).ravel()[0]), ctx.cash_min, self.cfg.cash_buffer_max)
        )

    def _slack_total(self, slacks: Sequence[Any]) -> float:
        """统计松弛变量总量（用于诊断）。"""
        total = 0.0
        for slack in slacks:
            value = getattr(slack, "value", None)
            if value is not None:
                total += float(np.asarray(value).ravel()[0])
        return float(total)

    def _postprocess_weights(self, raw: Any, ctx: _SolveContext) -> np.ndarray:
        """求解结果数值清理：去 NaN、截断到 ``[0, max_single]``、回填锁定持仓。"""
        if raw is None:  # pragma: no cover - 求解器异常状态
            return self._fallback_weights(ctx)
        weights = np.asarray(raw, dtype=np.float64).ravel()
        weights = np.where(np.isfinite(weights), weights, 0.0)
        weights = np.clip(weights, 0.0, ctx.max_single)
        if np.any(ctx.mask_locked):
            weights[ctx.mask_locked] = ctx.w_prev[ctx.mask_locked]
        return np.ascontiguousarray(weights, dtype=np.float64)

    def _fallback_weights(self, ctx: _SolveContext) -> np.ndarray:
        """Level3 保底权重：``w_prev`` 归一化（无持仓时按等权建仓）。"""
        invest = 1.0 - min(max(ctx.cash_min, 0.0), float(self.cfg.cash_buffer_max))
        prev = ctx.w_prev
        total = float(np.sum(prev))
        if total > _EPS:
            weights = prev * (invest * _solve_ratio(total))
        else:
            weights = np.zeros(ctx.n_symbols, dtype=np.float64)
            free = ~ctx.mask_locked
            count = int(np.count_nonzero(free))
            if count > 0:
                weights[free] = min(invest / float(count), ctx.max_single)
        weights = self._postprocess_weights(weights, ctx)
        if np.any(ctx.mask_locked):
            weights[ctx.mask_locked] = ctx.w_prev[ctx.mask_locked]
        return weights

    def _fallback_result(
        self,
        ctx: _SolveContext,
        *,
        reason: str,
        attempts: tuple[str, ...] = (),
    ) -> OptimizerResult:
        """三级容错的 Level3：返回 ``w_prev`` 归一化结果并记录失败原因。"""
        weights = self._fallback_weights(ctx)
        cash = min(max(ctx.cash_min, 0.0), float(self.cfg.cash_buffer_max))
        return OptimizerResult(
            weights=weights,
            objective=self._objective_value(ctx, weights),
            solver_used="FALLBACK",
            fallback_level=3,
            status="fallback",
            diagnostics=self._diagnostics(
                ctx,
                weights,
                cash=cash,
                solver="FALLBACK",
                status="fallback",
                extra={"reason": reason, "attempts": attempts},
            ),
        )

    def _diagnostics(
        self,
        ctx: _SolveContext,
        weights: np.ndarray,
        *,
        cash: float,
        solver: str,
        status: str,
        slack: float = 0.0,
        numpy_mode: bool = False,
        turnover_clipped: bool = False,
        extra: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """汇总求解诊断信息。"""
        diagnostics: dict[str, Any] = {
            "cash_buffer": float(cash),
            "invested": float(np.sum(weights)),
            "turnover": float(np.sum(np.abs(weights - ctx.w_prev))),
            "initial_build": bool(ctx.initial_build),
            "horizon": bool(ctx.horizon),
            "n_locked": int(np.count_nonzero(ctx.mask_locked)),
            "n_style": len(ctx.style_columns),
            "max_style_deviation": self._max_style_deviation(ctx, weights),
            "solver": solver,
            "status": status,
            "numpy_mode": bool(numpy_mode),
        }
        if slack > 0.0:
            diagnostics["slack_used"] = float(slack)
        if turnover_clipped:
            diagnostics["turnover_clipped"] = True
        if extra:
            diagnostics.update(dict(extra))
        return diagnostics

    def _max_style_deviation(self, ctx: _SolveContext, weights: np.ndarray) -> float:
        """计算求解后风格偏离的最大绝对值（诊断用）。"""
        if ctx.exposures is None or not ctx.style_columns:
            return 0.0
        residual = weights - ctx.w_base
        values = [abs(float(np.dot(ctx.exposures[:, col], residual))) for col in ctx.style_columns]
        return float(max(values)) if values else 0.0

    def _feasible_within_tolerance(
        self,
        ctx: _SolveContext,
        weights: np.ndarray,
        cash: float,
        *,
        tol: float,
    ) -> bool:
        """校验 Level1 解是否满足全部硬约束（数值容差内）。

        求解器可能返回 ``optimal_inaccurate``；若其违反满仓/上限/风格/行业/换手约束，
        则视为该级失败并降级至 Level2 松弛层级。
        """
        if abs(float(np.sum(weights)) - (1.0 - cash)) > tol:
            return False
        if float(np.min(weights)) < -tol:
            return False
        if float(np.max(weights)) > ctx.max_single + tol:
            return False
        if (
            ctx.use_active_bound
            and ctx.wb is not None
            and float(np.max(np.abs(weights - ctx.wb))) > ctx.max_active + tol
        ):
            return False
        if not ctx.initial_build:
            turnover = float(np.sum(np.abs(weights - ctx.w_prev)))
            if turnover > float(self.cfg.max_daily_turnover) + tol:
                return False
        residual = weights - ctx.w_base
        if ctx.exposures is not None:
            for pos, col in enumerate(ctx.style_columns):
                deviation = abs(float(np.dot(ctx.exposures[:, col], residual)))
                if deviation > self._style_delta(ctx.style_names[pos]) + tol:
                    return False
        if ctx.industry is not None:
            for code in np.unique(ctx.industry[ctx.industry >= 0.0]):
                indicator = (ctx.industry == code).astype(np.float64)
                bound = float(self.cfg.industry_deviation) + tol
                if abs(float(np.dot(indicator, residual))) > bound:
                    return False
        return True

    def _objective_value(self, ctx: _SolveContext, weights: np.ndarray) -> float:
        """按与 cvxpy 目标一致的口径复算 NumPy 目标值（诊断/降级路径使用）。"""
        residual = weights - ctx.w_base
        value = float(self.cfg.alpha_scale) * float(np.dot(ctx.alpha, weights))
        diff = weights - ctx.w_prev
        buys = np.clip(diff, 0.0, None)
        sells = np.clip(-diff, 0.0, None)
        buy_pen, sell_pen = self.cfg.asymmetric_trade_penalty
        value -= buy_pen * float(np.sum(buys)) + sell_pen * float(np.sum(sells))
        if ctx.alpha_ema is not None and self.cfg.buy_hurdle:
            jump = np.maximum(ctx.alpha - ctx.alpha_ema, 0.0)
            value -= float(self.cfg.hurdle_penalty) * float(np.dot(jump, buys))
        if float(self.cfg.risk_aversion) > 0.0:
            quad = 0.0
            if ctx.exposures is not None:
                y = ctx.exposures.T @ residual
                if ctx.factor_cov_sqrt is not None:
                    z = ctx.factor_cov_sqrt.T @ y
                    quad += float(np.dot(z, z))
                if ctx.spec_var is not None:
                    quad += float(np.dot(ctx.spec_var, residual * residual))
            elif ctx.spec_var is not None:
                quad += float(np.dot(ctx.spec_var, residual * residual))
            value -= 0.5 * float(self.cfg.risk_aversion) * quad
        return float(value)


def _preference_scores(alpha: np.ndarray) -> np.ndarray:
    """把当日 alpha 转为 ``[0, 1]`` 的排序偏好（1 为最优），用于 NumPy 降级分配。"""
    n = int(alpha.size)
    out = np.zeros(n, dtype=np.float64)
    valid = np.isfinite(alpha)
    count = int(np.count_nonzero(valid))
    if count <= 0:
        return np.full(n, 1.0, dtype=np.float64)
    ranks = _average_rank_1d(alpha[valid])
    out[valid] = ranks / float(max(count, 1))
    return out


def _allocate_with_bounds(
    preference: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    total: float,
) -> np.ndarray:
    """在逐票上下界内按偏好比例分配权重，并精确命中 ``sum(w) == total``。

    采用「排序 -> 偏好比例分配 -> 上限截断 -> 逐次归一化（二分比例乘子）」策略：
    ``w = lower + min(lambda * preference, upper - lower)``，对 ``lambda`` 二分即可
    保证单调命中目标投资比例；最后把浮点残差按偏好在剩余容量的股票上补齐。
    """
    n = int(preference.size)
    lo = np.clip(np.asarray(lower, dtype=np.float64).ravel(), 0.0, None)
    hi = np.clip(np.asarray(upper, dtype=np.float64).ravel(), 0.0, None)
    lo = np.minimum(lo, hi)
    capacity = np.maximum(hi - lo, 0.0)
    target = float(np.clip(total, 0.0, float(np.sum(lo) + np.sum(capacity))))
    need = target - float(np.sum(lo))
    if need <= _EPS or n == 0:
        return lo

    pref = np.asarray(preference, dtype=np.float64).ravel()
    if pref.size != n:
        raise ValueError("preference 长度与上下界不一致")
    pref = np.where(np.isfinite(pref) & (pref > 0.0), pref, 0.0)
    if float(np.max(pref)) <= _EPS:
        pref = np.ones(n, dtype=np.float64)

    def allocate(multiplier: float) -> np.ndarray:
        return np.minimum(multiplier * pref, capacity)

    high = 1.0
    while float(np.sum(allocate(high))) < need and high < 1e12:
        high *= 2.0
    low = 0.0
    for _ in range(100):
        mid = 0.5 * (low + high)
        if float(np.sum(allocate(mid))) < need:
            low = mid
        else:
            high = mid
    extra = allocate(high)
    extra_sum = float(np.sum(extra))
    if extra_sum > _EPS:
        extra = extra * (need * _solve_ratio(extra_sum))
    extra = np.minimum(extra, capacity)
    residual = need - float(np.sum(extra))
    if residual > 0.0:
        order = np.argsort(-pref, kind="mergesort")
        for idx in order:
            room = capacity[idx] - extra[idx]
            if room <= 0.0:
                continue
            add = min(residual, float(room))
            extra[idx] += add
            residual -= add
            if residual <= _EPS:
                break
    return lo + extra


# ==============================================================================
# 4) 门面：Barra 组合优化器
# ==============================================================================


@dataclass(frozen=True)
class BarraOptimizerConfig(ConvexOptimizerConfig):
    """Barra 组合优化器总配置（在求解配置之上叠加塑形与风控配置）。"""

    shaper: SignalShaperConfig = field(default_factory=SignalShaperConfig)
    risk_guard: DynamicRiskGuardConfig = field(default_factory=DynamicRiskGuardConfig)
    seq_shape: bool = True

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any] | None) -> BarraOptimizerConfig:
        """解析嵌套 YAML 字典（``shaper`` / ``risk_guard`` / 求解参数平铺均可）。"""
        data = dict(raw or {})
        shaper_raw = _pop_section(data, ("shaper", "signal_shaper"))
        guard_raw = _pop_section(
            data, ("risk_guard", "riskguard", "guard", "adaptive_stress_guard")
        )
        convex_raw = _pop_section(data, ("convex", "optimizer", "convex_optimizer", "solve"))
        merged = dict(convex_raw)
        for name in _CONVEX_FIELD_NAMES:
            if name in data:
                merged[name] = data[name]
        convex = ConvexOptimizerConfig.from_dict(merged)
        kwargs = {name: getattr(convex, name) for name in _CONVEX_FIELD_NAMES}
        return cls(
            shaper=SignalShaperConfig.from_dict(shaper_raw),
            risk_guard=DynamicRiskGuardConfig.from_dict(guard_raw),
            seq_shape=bool(data.get("seq_shape", True)),
            **kwargs,
        )


@dataclass(frozen=True)
class _ExposureBundle:
    """逐日因子暴露的按需重建器（避免物化 ``(T, N, 42)`` 大张量）。"""

    names: tuple[str, ...]
    style_columns: tuple[int, ...]
    style_mats: tuple[np.ndarray, ...]
    industry_columns: tuple[int, ...]
    industry_codes: tuple[int, ...]
    country_columns: tuple[int, ...]
    industry_field: np.ndarray | None
    n_symbols: int
    from_risk_store: bool

    def row(self, t: int) -> np.ndarray:
        """返回第 ``t`` 日的 ``(N, F)`` 暴露矩阵。"""
        matrix = np.zeros((self.n_symbols, len(self.names)), dtype=np.float64)
        for col, mat in zip(self.style_columns, self.style_mats, strict=True):
            matrix[:, col] = mat[t]
        if self.industry_field is not None:
            codes_t = np.rint(
                np.where(np.isfinite(self.industry_field[t]), self.industry_field[t], -1.0)
            ).astype(np.int64)
            for col, code in zip(self.industry_columns, self.industry_codes, strict=True):
                matrix[:, col] = (codes_t == code).astype(np.float64)
        for col in self.country_columns:
            matrix[:, col] = 1.0
        return matrix


class BarraPortfolioOptimizer:
    """Barra 组合优化器门面：信号塑形 -> 动态风控 -> 凸优化 -> 逐日目标权重。

    ``w_b`` 三级降级：``constituent``（外部传入成分权重矩阵或字典）-> ``mkt_cap_proxy``
    （按市值排序取 ``proxy_rank_range`` 段并按市值归一化）-> ``absolute_neutral``（``w_b = 0``）。
    """

    def __init__(
        self,
        cfg: BarraOptimizerConfig | None = None,
        *,
        risk_store: Any | None = None,
        style_exposures: Mapping[str, np.ndarray] | np.ndarray | None = None,
        force_numpy: bool = False,
        factor_names: Sequence[str] | None = None,
        benchmark_weights: Any | None = None,
    ) -> None:
        self.cfg = cfg if cfg is not None else BarraOptimizerConfig()
        self.risk_store = risk_store
        self.style_exposures = style_exposures
        self.factor_names = tuple(str(n) for n in factor_names) if factor_names else ()
        self.benchmark_weights = benchmark_weights
        self.shaper = SignalShaper(self.cfg.shaper)
        store_names = tuple(str(n) for n in getattr(risk_store, "factors", ()) or ())
        self.guard = DynamicRiskGuard(
            self.cfg.risk_guard,
            factor_names=self.factor_names or store_names,
        )
        self.optimizer = ConvexOptimizer(self.cfg, force_numpy=force_numpy)

    # ------------------------------------------------------------------
    # 主入口
    # ------------------------------------------------------------------
    def build_weights(
        self,
        panel: Any,
        scores: np.ndarray,
        *,
        factor_returns: np.ndarray | None = None,
        dates: Sequence[Any] | None = None,
        last_hold: np.ndarray | None = None,
        reserve_cash: float = 1.0,
    ) -> np.ndarray:
        """逐日滚动求解目标权重矩阵 ``(T, N, order='F')``。

        Parameters
        ----------
        panel:
            研究面板（需提供 ``mask``、``fields``/``aux_fields``；字段缺失自动容错）。
        scores:
            ``(T, N)`` 原始信号；``cfg.seq_shape`` 为真时先经 :class:`SignalShaper`。
        factor_returns:
            可选 ``(T, K)`` Barra 因子日收益，用于驱动 :class:`DynamicRiskGuard`。
        dates:
            可选日期序列（默认取 ``panel.dates``），用于风险库协方差按日对齐。
        last_hold:
            可选初始漂移持仓（第 0 日 ``w_prev``），之后由上一日目标权重递推。
        reserve_cash:
            目标投资比例（``1 - 现金``），取值 ``(0, 1]``。
        """
        if not (0.0 < float(reserve_cash) <= 1.0):
            raise ValueError(f"reserve_cash 必须在 (0, 1] 之间, 实际 {reserve_cash}")
        n_days, n_symbols = panel.shape
        raw_scores = np.asarray(scores, dtype=np.float64)
        if raw_scores.shape != (n_days, n_symbols):
            raise ValueError(f"scores 形状 {raw_scores.shape} 与面板 {(n_days, n_symbols)} 不一致")
        date_list = tuple(str(d) for d in (dates if dates is not None else panel.dates))
        if len(date_list) != n_days:
            raise ValueError(f"dates 长度 {len(date_list)} 与面板交易日数 {n_days} 不一致")

        bundle = self._resolve_exposures(panel)
        industry = bundle.industry_field
        suspend = self._resolve_suspend(panel)
        mkt_cap = _first_field(panel, _MKT_CAP_FIELD_CANDIDATES)
        resvol = _first_field(panel, _RESVOL_FIELD_CANDIDATES)
        spec_var = self._resolve_spec_variance(panel, panel.dates, panel.symbols)
        constituent = self._resolve_constituent_matrix(panel)

        guard_state: RiskGuardState | None = None
        if factor_returns is not None and self.cfg.risk_guard.enabled:
            fr = np.asarray(factor_returns, dtype=np.float64)
            if fr.ndim != 2 or fr.shape[0] != n_days:
                raise ValueError(
                    f"factor_returns 形状 {fr.shape} 必须为 (T={n_days}, K) 的二维矩阵"
                )
            guard_state = self.guard.assess(
                fr, date_list, factor_names=self._guard_names(fr.shape[1])
            )

        cash_min = max(0.0, min(1.0 - float(reserve_cash), float(self.cfg.cash_buffer_max)))
        weights = np.zeros((n_days, n_symbols), dtype=np.float64, order="F")
        prev = np.zeros(n_symbols, dtype=np.float64)
        if last_hold is not None:
            prev = np.asarray(last_hold, dtype=np.float64).ravel()
            if prev.size != n_symbols:
                raise ValueError(f"last_hold 长度 {prev.size} 与股票数 {n_symbols} 不一致")
            prev = np.where(np.isfinite(prev), np.clip(prev, 0.0, None), 0.0)
        alpha_ema = np.zeros(n_symbols, dtype=np.float64)

        for t in range(n_days):
            mask_active = np.asarray(panel.mask[t], dtype=bool).copy()
            ind_row = industry[t] if industry is not None else None
            resvol_row = resvol[t] if resvol is not None else None
            cap_row = mkt_cap[t] if mkt_cap is not None else None
            shaped = self._shape_row(raw_scores[t], ind_row, resvol_row, cap_row, guard_state, t)
            ema_input = alpha_ema.copy()
            alpha_ema = float(self.cfg.persistence) * alpha_ema + (
                1.0 - float(self.cfg.persistence)
            ) * np.where(np.isfinite(shaped), shaped, 0.0)

            wb_row = self._benchmark_row(panel, t, cap_row, mask_active, constituent)
            exposures_t = bundle.row(t) if len(bundle.names) > 0 else None
            cov_sqrt = self._factor_cov_sqrt(panel, t, bundle)
            result = self.solve_one_day(
                shaped,
                exposures=exposures_t,
                industry=ind_row,
                spec_var=spec_var[t] if spec_var is not None else None,
                factor_cov_sqrt=cov_sqrt,
                wb=wb_row,
                w_prev=prev,
                alpha_ema=ema_input,
                mask_active=mask_active,
                mask_suspend=suspend[t] if suspend is not None else None,
                style_columns=bundle.style_columns,
                style_names=bundle.names,
                cash_buffer=cash_min,
            )
            weights[t] = result.weights
            prev = result.weights
        return np.asfortranarray(weights, dtype=np.float64)

    def solve_one_day(
        self,
        scores_t: np.ndarray,
        *,
        exposures: np.ndarray | None = None,
        industry: np.ndarray | None = None,
        spec_var: np.ndarray | None = None,
        factor_cov_sqrt: np.ndarray | None = None,
        wb: np.ndarray | None = None,
        w_prev: np.ndarray | None = None,
        alpha_ema: np.ndarray | None = None,
        mask_active: np.ndarray | None = None,
        mask_suspend: np.ndarray | None = None,
        forecast: np.ndarray | None = None,
        style_columns: Sequence[int] | None = None,
        style_names: Sequence[str] | None = None,
        cash_buffer: float = 0.0,
    ) -> OptimizerResult:
        """求解单日目标权重（``scores_t`` 应为已塑形的 alpha）。"""
        return self.optimizer.solve(
            scores_t,
            w_prev=w_prev,
            wb=wb,
            exposures=exposures,
            industry=industry,
            spec_var=spec_var,
            factor_cov_sqrt=factor_cov_sqrt,
            style_columns=style_columns,
            style_names=style_names,
            alpha_ema=alpha_ema,
            mask_active=mask_active,
            mask_suspend=mask_suspend,
            forecast=forecast,
            cash_buffer=cash_buffer,
        )

    # ------------------------------------------------------------------
    # 内部：字段解析、塑形、基准与风险库
    # ------------------------------------------------------------------
    def _shape_row(
        self,
        row: np.ndarray,
        industry_row: np.ndarray | None,
        resvol_row: np.ndarray | None,
        cap_row: np.ndarray | None,
        guard_state: RiskGuardState | None,
        t: int,
    ) -> np.ndarray:
        """对单日截面塑形（含风控动态参数覆盖）。"""
        if self.cfg.seq_shape:
            if guard_state is not None:
                vol_weight, power, size_tilt = self.guard.apply(guard_state, t, self.cfg.shaper)
            else:
                vol_weight = float(self.cfg.shaper.vol_weight)
                power = float(self.cfg.shaper.power)
                size_tilt = 0.0
            return self.shaper.shape_scores(
                row,
                industry=industry_row,
                resvol=resvol_row,
                vol_weight=vol_weight,
                power=power,
                size=cap_row,
                size_tilt=size_tilt,
            )[0]
        shaped = np.asarray(row, dtype=np.float64).copy()
        if guard_state is not None and cap_row is not None:
            _, _, size_tilt = self.guard.apply(guard_state, t, self.cfg.shaper)
            if size_tilt != 0.0:
                shaped = shaped + size_tilt * centered_rank_scores(cap_row)
        return shaped

    def _guard_names(self, n_factors: int) -> tuple[str, ...]:
        """定位 SIZE 因子所需的因子名列表（优先构造参数，其次风险库）。"""
        if len(self.factor_names) == n_factors:
            return self.factor_names
        store_names = tuple(str(n) for n in getattr(self.risk_store, "factors", ()) or ())
        if len(store_names) == n_factors:
            return store_names
        return ()

    def _factor_cov_sqrt(self, panel: Any, t: int, bundle: _ExposureBundle) -> np.ndarray | None:
        """读取第 ``t`` 日因子协方差平方根（仅风险库暴露路径可用）。"""
        store = self.risk_store
        if store is None or not bundle.from_risk_store or not bundle.names:
            return None
        getter = getattr(store, "get_factor_cov_sqrt", None)
        if getter is None:
            return None
        try:
            sqrt = getter(str(panel.dates[t])[:10])
        except Exception:  # noqa: BLE001 - 风险库缺该日数据时容错
            return None
        if sqrt is None:
            return None
        arr = np.asarray(sqrt, dtype=np.float64)
        if arr.shape != (len(bundle.names), len(bundle.names)):
            return None
        return np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)

    def _resolve_exposures(self, panel: Any) -> _ExposureBundle:
        """构建按需重建的因子暴露 bundle（优先风险库 42 因子结构）。"""
        n_days, n_symbols = panel.shape
        store = self.risk_store
        store_names = tuple(str(n) for n in getattr(store, "factors", ()) or ())
        industry_field = _first_field(panel, _INDUSTRY_FIELD_CANDIDATES)
        if store is not None and store_names:
            style_set = {str(n).upper() for n in getattr(store, "style_factors", ()) or ()}
            style_cols: list[int] = []
            style_mats: list[np.ndarray] = []
            industry_cols: list[int] = []
            industry_codes: list[int] = []
            country_cols: list[int] = []
            for col, name in enumerate(store_names):
                upper = name.upper()
                if upper in style_set:
                    field = _first_field(panel, (f"barra_{name.lower()}",))
                    if field is None:
                        continue
                    style_cols.append(col)
                    style_mats.append(field)
                elif upper == "COUNTRY":
                    country_cols.append(col)
                elif name.isdigit():
                    industry_cols.append(col)
                    industry_codes.append(int(name))
            return _ExposureBundle(
                names=store_names,
                style_columns=tuple(style_cols),
                style_mats=tuple(style_mats),
                industry_columns=tuple(industry_cols),
                industry_codes=tuple(industry_codes),
                country_columns=tuple(country_cols),
                industry_field=industry_field,
                n_symbols=n_symbols,
                from_risk_store=True,
            )

        style_cols = []
        style_mats = []
        names: list[str] = []
        external = self.style_exposures
        if isinstance(external, Mapping):
            for key, value in external.items():
                mat = np.asarray(value, dtype=np.float64)
                if mat.shape != (n_days, n_symbols):
                    raise ValueError(
                        f"style_exposures[{key!r}] 形状 {mat.shape} 与面板 "
                        f"{(n_days, n_symbols)} 不一致"
                    )
                names.append(str(key).upper())
                style_cols.append(len(names) - 1)
                style_mats.append(mat)
        elif external is not None:
            arr = np.asarray(external, dtype=np.float64)
            if arr.ndim == 2:
                arr = arr[:, :, None]
            if arr.ndim != 3 or arr.shape[:2] != (n_days, n_symbols):
                raise ValueError(f"style_exposures 形状 {arr.shape} 必须为 (T, N, K) 或 (T, N)")
            for k in range(arr.shape[2]):
                names.append(_STYLE_FACTORS[k] if k < len(_STYLE_FACTORS) else f"STYLE{k}")
                style_cols.append(k)
                style_mats.append(arr[:, :, k])
        else:
            for name in _STYLE_FACTORS:
                field = panel.fields.get(f"barra_{name.lower()}")
                if field is None:
                    continue
                names.append(name)
                style_cols.append(len(names) - 1)
                style_mats.append(field)
        return _ExposureBundle(
            names=tuple(names),
            style_columns=tuple(style_cols),
            style_mats=tuple(style_mats),
            industry_columns=(),
            industry_codes=(),
            country_columns=(),
            industry_field=industry_field,
            n_symbols=n_symbols,
            from_risk_store=False,
        )

    def _resolve_suspend(self, panel: Any) -> np.ndarray | None:
        """读取停牌标记矩阵（多别名容错，缺失返回 ``None``）。"""
        raw = _first_field(panel, _SUSPEND_FIELD_CANDIDATES)
        if raw is None:
            return None
        return np.isfinite(raw) & (raw > 0.5)

    def _resolve_spec_variance(
        self,
        panel: Any,
        dates: Sequence[Any],
        symbols: Sequence[str],
    ) -> np.ndarray | None:
        """读取/估计对角特质方差矩阵（字段优先，风险库文件兜底，缺失返回 ``None``）。"""
        raw = _first_field(panel, _SPEC_VAR_FIELD_CANDIDATES)
        if raw is not None:
            return np.clip(np.where(np.isfinite(raw), raw, 0.0), 0.0, None)
        vol = _first_field(panel, ("spec_vol_ewma", "barra_spec_vol"))
        if vol is not None:
            vol_clean = np.where(np.isfinite(vol), np.clip(vol, 0.0, None), 0.0)
            return np.asfortranarray(vol_clean * vol_clean, dtype=np.float64)
        store = self.risk_store
        loader = getattr(store, "load_specific_variance", None) if store is not None else None
        if loader is None:
            return None
        try:
            loaded = loader(dates, symbols)
        except Exception:  # noqa: BLE001 - 风险库缺特质方差文件时容错
            return None
        if loaded is None:
            return None
        arr = np.asarray(loaded, dtype=np.float64)
        if arr.shape != panel.shape:
            return None
        return np.clip(np.where(np.isfinite(arr), arr, 0.0), 0.0, None)

    def _resolve_constituent_matrix(self, panel: Any) -> np.ndarray | None:
        """解析外部传入的成分股权重（``(T, N)`` 矩阵 / ``{symbol: weight}`` / ``{date: row}``）。"""
        raw = self.benchmark_weights
        if raw is None:
            return None
        n_days, n_symbols = panel.shape
        if isinstance(raw, np.ndarray):
            arr = np.asarray(raw, dtype=np.float64)
            if arr.shape == (n_symbols,):
                return np.broadcast_to(arr, (n_days, n_symbols)).copy()
            if arr.shape == (n_days, n_symbols):
                return arr
            raise ValueError(
                f"benchmark_weights 形状 {arr.shape} 必须为 (N={n_symbols},) 或 "
                f"(T={n_days}, N={n_symbols})"
            )
        if not isinstance(raw, Mapping) or not raw:
            return None
        values = list(raw.values())
        if all(np.isscalar(v) for v in values):
            matrix = np.zeros((n_days, n_symbols), dtype=np.float64)
            symbol_index = _symbol_index(panel.symbols)
            for key, value in raw.items():
                pos = symbol_index.get(_normalize_symbol(key))
                if pos is not None:
                    matrix[:, pos] = float(value)
            return matrix
        if all(isinstance(v, np.ndarray) or np.isscalar(v) for v in values):
            date_rows: dict[str, np.ndarray] = {}
            for key, value in raw.items():
                row = np.asarray(value, dtype=np.float64).ravel()
                if row.size != n_symbols:
                    raise ValueError(
                        f"benchmark_weights[{key!r}] 长度 {row.size} 与股票数 {n_symbols} 不一致"
                    )
                date_rows[str(key)[:10]] = row
            matrix = np.zeros((n_days, n_symbols), dtype=np.float64)
            for t in range(n_days):
                row = date_rows.get(str(panel.dates[t])[:10])
                if row is not None:
                    matrix[t] = row
            return matrix
        return None

    def _benchmark_row(
        self,
        panel: Any,
        t: int,
        cap_row: np.ndarray | None,
        mask_active: np.ndarray,
        constituent: np.ndarray | None,
    ) -> np.ndarray | None:
        """构造第 ``t`` 日基准权重 ``w_b``（三级降级），绝对中性模式返回 ``None``。"""
        mode = self.cfg.benchmark_weight_mode
        if mode == "absolute_neutral":
            return None
        if mode == "constituent" and constituent is not None:
            row = np.asarray(constituent[t], dtype=np.float64).copy()
            row = np.where(np.isfinite(row), np.clip(row, 0.0, None), 0.0)
            total = float(np.sum(row))
            if total > _EPS:
                return row / total
        proxy = self._proxy_benchmark_row(cap_row, mask_active)
        if proxy is not None:
            return proxy
        return self._equal_benchmark_row(mask_active)

    def _proxy_benchmark_row(
        self, cap_row: np.ndarray | None, mask_active: np.ndarray
    ) -> np.ndarray | None:
        """按市值排序取 ``proxy_rank_range`` 段并归一化的基准代理。"""
        if cap_row is None or self.cfg.proxy_rank_range is None:
            return None
        valid = mask_active & np.isfinite(cap_row) & (cap_row > 0.0)
        candidates = np.flatnonzero(valid)
        if candidates.size == 0:
            return None
        order = candidates[np.argsort(-cap_row[candidates], kind="mergesort")]
        lo = max(1, int(self.cfg.proxy_rank_range[0]))
        hi = min(int(order.size), int(self.cfg.proxy_rank_range[1]))
        if lo > hi:
            return None
        selected = order[lo - 1 : hi]
        caps = cap_row[selected]
        total = float(np.sum(caps))
        if total <= _EPS:
            return None
        row = np.zeros(cap_row.size, dtype=np.float64)
        row[selected] = caps / total
        return row

    def _equal_benchmark_row(self, mask_active: np.ndarray) -> np.ndarray:
        """等权基准兜底（无市值字段或代理区间为空时使用）。"""
        row = np.zeros(mask_active.size, dtype=np.float64)
        selected = np.flatnonzero(mask_active)
        if selected.size == 0:
            return row
        row[selected] = 1.0 / float(selected.size)
        return row


def _normalize_symbol(symbol: Any) -> str:
    """把股票代码归一化为 6 位数字（剥离交易所后缀）。"""
    text = str(symbol).strip()
    if "." in text:
        text = text.split(".", 1)[0]
    return text.zfill(6) if text.isdigit() and len(text) < 6 else text


def _symbol_index(symbols: Sequence[Any]) -> dict[str, int]:
    """构建归一化代码 -> 列下标映射。"""
    return {_normalize_symbol(sym): pos for pos, sym in enumerate(symbols)}


# ==============================================================================
# 5) 持仓漂移闭环
# ==============================================================================


def drift_positions(
    prev_weights: np.ndarray,
    prev_exec_price: np.ndarray,
    prev_close_price: np.ndarray,
    *,
    reserve_cash: float = 1.0,
) -> np.ndarray:
    """按真实价格变动漂移昨日持仓并归一化到 ``reserve_cash``。

    ``w_i' = w_i * (p_close_i / p_exec_i)``；成交价/收盘价缺失或非正时该标的不做漂移
    （``w_i' = w_i``）。归一化 ``w_i'' = reserve_cash * w_i' / sum(w')``；
    全部价格无效或漂移后总权重非正时原样返回 ``prev_weights``。
    """
    weights = np.asarray(prev_weights, dtype=np.float64).ravel()
    exec_price = np.asarray(prev_exec_price, dtype=np.float64).ravel()
    close_price = np.asarray(prev_close_price, dtype=np.float64).ravel()
    if exec_price.size != weights.size or close_price.size != weights.size:
        raise ValueError(
            "prev_weights / prev_exec_price / prev_close_price 长度必须一致, "
            f"实际 {weights.size} / {exec_price.size} / {close_price.size}"
        )
    usable = (
        np.isfinite(exec_price)
        & np.isfinite(close_price)
        & (exec_price > 0.0)
        & (close_price > 0.0)
    )
    if not np.any(usable):
        return weights.copy()
    ratio = np.ones_like(weights)
    ratio[usable] = close_price[usable] / exec_price[usable]
    drifted = weights * ratio
    total = float(np.nansum(drifted))
    if not np.isfinite(total) or total <= _EPS:
        return weights.copy()
    return np.ascontiguousarray(reserve_cash * drifted / total, dtype=np.float64)


class PositionDriftTracker:
    """逐日持仓漂移追踪器（严格因果，只使用传入的当期成交价与收盘价）。

    每个交易日调用 :meth:`update` 传入当日目标权重与该日的成交价、收盘价，
    内部按当日「成交价 -> 收盘价」的价格变动盯市并归一化，``last_hold`` 即最新
    的漂移后真实持仓（次日作为 ``w_prev`` 传入优化器）。
    """

    def __init__(self, *, reserve_cash: float = 1.0) -> None:
        if not (0.0 < float(reserve_cash) <= 1.0):
            raise ValueError(f"reserve_cash 必须在 (0, 1] 之间, 实际 {reserve_cash}")
        self.reserve_cash = float(reserve_cash)
        self._last_hold: np.ndarray | None = None

    @property
    def last_hold(self) -> np.ndarray | None:
        """返回最新漂移后持仓的副本；尚未更新时返回 ``None``。"""
        return None if self._last_hold is None else self._last_hold.copy()

    def reset(self) -> None:
        """清空内部持仓状态。"""
        self._last_hold = None

    def update(
        self,
        target_weights: np.ndarray,
        exec_price_row: np.ndarray | None = None,
        close_price_row: np.ndarray | None = None,
        *,
        reserve_cash: float | None = None,
    ) -> np.ndarray:
        """更新并返回漂移后的持仓（价格行缺失时仅按目标权重归一化）。"""
        weights = np.asarray(target_weights, dtype=np.float64).ravel()
        cash = self.reserve_cash if reserve_cash is None else float(reserve_cash)
        if not (0.0 < cash <= 1.0):
            raise ValueError(f"reserve_cash 必须在 (0, 1] 之间, 实际 {cash}")
        if exec_price_row is None or close_price_row is None:
            total = float(np.nansum(weights))
            drifted = (
                weights.copy()
                if total <= _EPS
                else np.ascontiguousarray(cash * weights / total, dtype=np.float64)
            )
        else:
            drifted = drift_positions(weights, exec_price_row, close_price_row, reserve_cash=cash)
        self._last_hold = np.ascontiguousarray(drifted, dtype=np.float64)
        return self._last_hold.copy()


#: 供 ``profile.py`` 等上层模块按需导入的公开 API 清单
__all__ = [
    "BarraOptimizerConfig",
    "BarraPortfolioOptimizer",
    "ConvexOptimizer",
    "ConvexOptimizerConfig",
    "DynamicRiskGuard",
    "DynamicRiskGuardConfig",
    "OptimizerResult",
    "PositionDriftTracker",
    "RiskGuardState",
    "SignalShaper",
    "SignalShaperConfig",
    "available_cvxpy_solvers",
    "centered_rank_scores",
    "drift_positions",
    "soft_industry_neutralize",
]

_CONVEX_FIELD_NAMES: tuple[str, ...] = tuple(f.name for f in fields(ConvexOptimizerConfig))
