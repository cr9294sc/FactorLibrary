"""A 股 Barra CNE5 风险模型存储、因果 Constrained WLS 特质收益分解与动态特质方差内核
(markets/stocks/barra.py)。

按照架构设计 §3.2 ~ §3.3 实现：
1. `BarraRiskModelStore`：加载并管理 `risk_store/`（`store.json`、`factor_returns.parquet`、
   `factor_cov.npz` 与可选 `specific_variance.parquet`），提供对称化与最小特征值截断
   ($\\lambda_{\\min} \\ge 10^{-10}$) 后的日频因子协方差阵 $F_t$ 与平方根因子 $L_t$（$F_t = L_t L_t^\\top$）；
2. `compute_constrained_wls_idiosyncratic_returns`：因果横截面 Constrained WLS 动态分解，
   施加 Barra 市值加权行业收益零和约束 $\\sum_k \\omega_{k, t-\\Delta} f^{\\text{ind}}_{k,t} = 0$；
3. `estimate_dynamic_specific_variance`：严格因果滚动估计日频对角特质方差 $\\Delta_t$
   （EWMA + Newey-West 滞后修正 + 市值分层贝叶斯收缩 + 波动率体制调整 VRA）。
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from factorlib.markets.stocks.ares_bridge import BARRA_FACTOR_ORDER, BARRA_STYLE_FACTORS


def _to_iso_date(val: Any) -> str:
    if isinstance(val, datetime):
        return val.date().isoformat()
    if isinstance(val, date):
        return val.isoformat()
    s = str(val).strip()
    if len(s) == 8 and s.isdigit():
        return f"{s[:4]}-{s[4:6]}-{s[6:8]}"
    return s[:10]


def _normalize_sym6(sym: Any) -> str:
    s = str(sym).strip()
    if "." in s:
        s = s.split(".", 1)[0]
    return s.zfill(6) if s.isdigit() and len(s) < 6 else s


def ensure_psd_matrix(mat: np.ndarray, *, min_eigenvalue: float = 1e-10) -> np.ndarray:
    """对对称方阵执行对称化与最小特征值截断，保证数值半正定与因子分解稳定性。"""
    sym = 0.5 * (np.asarray(mat, dtype=np.float64) + np.asarray(mat, dtype=np.float64).T)
    evals, evecs = np.linalg.eigh(sym)
    clipped = np.maximum(evals, float(min_eigenvalue))
    return (evecs * clipped) @ evecs.T


def factor_cov_cholesky_or_eig_sqrt(
    cov: np.ndarray,
    *,
    min_eigenvalue: float = 1e-10,
) -> np.ndarray:
    """计算 PSD 截断后的对称平方根因子 $L_t$，满足 $L_t L_t^\\top = F_t^{\\text{psd}}$。"""
    sym = 0.5 * (np.asarray(cov, dtype=np.float64) + np.asarray(cov, dtype=np.float64).T)
    evals, evecs = np.linalg.eigh(sym)
    clipped = np.maximum(evals, float(min_eigenvalue))
    return evecs * np.sqrt(clipped)


@dataclass
class BarraRiskModelStore:
    """旁路 Barra CNE5 风险模型存储器（`risk_store/`）。"""

    store_dir: Path
    meta: dict[str, Any]
    factors: tuple[str, ...]
    style_factors: tuple[str, ...]
    factor_ret_dates: np.ndarray  # ISO 'YYYY-MM-DD'
    factor_returns: np.ndarray  # (T_fr, K)
    cov_dates: np.ndarray  # ISO 'YYYY-MM-DD'
    factor_cov: np.ndarray  # (T_cov, K, K)

    def __post_init__(self) -> None:
        self._fr_date_to_idx: dict[str, int] = {
            str(d): i for i, d in enumerate(self.factor_ret_dates.tolist())
        }
        self._cov_date_to_idx: dict[str, int] = {
            str(d): i for i, d in enumerate(self.cov_dates.tolist())
        }
        self._cov_dates_sorted: list[str] = list(self.cov_dates.tolist())

    @classmethod
    def load(cls, store_dir: str | Path) -> BarraRiskModelStore:
        """从 ``risk_store/`` 目录加载元数据、因子收益与日频因子协方差张量。"""
        sdir = Path(store_dir)
        store_json = sdir / "store.json"
        if not store_json.exists():
            raise FileNotFoundError(f"未找到风险库元数据文件: {store_json}")
        meta = json.loads(store_json.read_text(encoding="utf-8"))
        factors = tuple(str(x) for x in meta.get("factors", BARRA_FACTOR_ORDER))
        style_factors = tuple(str(x) for x in meta.get("style_factors", BARRA_STYLE_FACTORS))

        fr_path = sdir / "factor_returns.parquet"
        if fr_path.exists():
            fr_df = pl.read_parquet(fr_path).sort("trading_date")
            fr_dates = np.array([_to_iso_date(d) for d in fr_df["trading_date"].to_list()])
            fr_cols = [c for c in factors if c in fr_df.columns]
            fr_mat = fr_df.select(fr_cols).to_numpy().astype(np.float64)
        else:
            fr_dates = np.array([], dtype=str)
            fr_mat = np.empty((0, len(factors)), dtype=np.float64)

        cov_path = sdir / "factor_cov.npz"
        if cov_path.exists():
            with np.load(cov_path, allow_pickle=False) as npz:
                cov_dates = np.array([_to_iso_date(d) for d in npz["dates"].tolist()])
                cov_factors = [str(x) for x in npz["factors"].tolist()]
                cov_tensor = np.asarray(npz["cov"], dtype=np.float64)
                if cov_factors != list(factors) and cov_tensor.size > 0:
                    f_map = {name: i for i, name in enumerate(cov_factors)}
                    perm = [f_map[f] for f in factors]
                    cov_tensor = cov_tensor[:, perm, :][:, :, perm]
        else:
            cov_dates = np.array([], dtype=str)
            cov_tensor = np.empty((0, len(factors), len(factors)), dtype=np.float64)

        return cls(
            store_dir=sdir,
            meta=meta,
            factors=factors,
            style_factors=style_factors,
            factor_ret_dates=fr_dates,
            factor_returns=fr_mat,
            cov_dates=cov_dates,
            factor_cov=cov_tensor,
        )

    def _resolve_cov_idx(self, dt: Any, *, asof_fallback: bool = True) -> int | None:
        if len(self._cov_dates_sorted) == 0:
            return None
        iso = _to_iso_date(dt)
        if iso in self._cov_date_to_idx:
            return self._cov_date_to_idx[iso]
        if not asof_fallback:
            return None
        # 严格因果截断：仅回退至 <= iso 的最近可用交易日
        import bisect

        pos = bisect.bisect_right(self._cov_dates_sorted, iso) - 1
        return pos if pos >= 0 else None

    def get_factor_cov(
        self,
        dt: Any,
        *,
        min_eigenvalue: float = 1e-10,
        asof_fallback: bool = True,
    ) -> np.ndarray | None:
        """获取指定交易日（或严格因果回退）的 PSD 保护日频因子协方差矩阵 $F_t$。"""
        idx = self._resolve_cov_idx(dt, asof_fallback=asof_fallback)
        if idx is None:
            return None
        raw = self.factor_cov[idx]
        if not np.all(np.isfinite(raw)):
            raw = np.nan_to_num(raw, nan=0.0)
        return ensure_psd_matrix(raw, min_eigenvalue=min_eigenvalue)

    def get_factor_cov_sqrt(
        self,
        dt: Any,
        *,
        min_eigenvalue: float = 1e-10,
        asof_fallback: bool = True,
    ) -> np.ndarray | None:
        """获取指定交易日因子协方差平方根因子 $L_t$（$L_t L_t^\\top = F_t$）。"""
        idx = self._resolve_cov_idx(dt, asof_fallback=asof_fallback)
        if idx is None:
            return None
        raw = self.factor_cov[idx]
        if not np.all(np.isfinite(raw)):
            raw = np.nan_to_num(raw, nan=0.0)
        return factor_cov_cholesky_or_eig_sqrt(raw, min_eigenvalue=min_eigenvalue)

    def align_factor_returns(self, target_dates: Sequence[Any]) -> np.ndarray:
        """按目标日期序列对齐因子日收益率矩阵 ``(T, K)``。"""
        T = len(target_dates)
        K = len(self.factors)
        out = np.full((T, K), np.nan, dtype=np.float64)
        for t, d in enumerate(target_dates):
            idx = self._fr_date_to_idx.get(_to_iso_date(d))
            if idx is not None:
                out[t, :] = self.factor_returns[idx, :]
        return out

    def save_specific_variance(
        self,
        dates: Sequence[Any],
        symbols: Sequence[str],
        spec_var: np.ndarray,
    ) -> Path:
        """将滚动估计的日频特质方差 $\\Delta_t$ 写盘至 ``risk_store/specific_variance.parquet``。"""
        T, N = spec_var.shape
        if len(dates) != T or len(symbols) != N:
            raise ValueError(
                f"spec_var 形状 {spec_var.shape} 与 (T={len(dates)}, N={len(symbols)}) 不一致"
            )
        parsed_dates = [date.fromisoformat(_to_iso_date(d)) for d in dates]
        date_col = pl.Series(
            "trading_date",
            np.repeat(np.array(parsed_dates, dtype="datetime64[D]"), N),
            dtype=pl.Date,
        )
        sym_col = pl.Series("symbol", np.tile(np.array(list(symbols)), T), dtype=pl.String)
        df = pl.DataFrame(
            {
                "trading_date": date_col,
                "symbol": sym_col,
                "spec_var": np.asarray(spec_var, dtype=np.float64).reshape(-1, order="C"),
            }
        )
        out_path = self.store_dir / "specific_variance.parquet"
        df.write_parquet(out_path, compression="zstd")
        return out_path

    def load_specific_variance(
        self,
        target_dates: Sequence[Any],
        target_symbols: Sequence[str],
    ) -> np.ndarray | None:
        """从 ``risk_store/specific_variance.parquet`` 加载并按 6 位代码对齐至目标网格。"""
        sv_path = self.store_dir / "specific_variance.parquet"
        if not sv_path.exists():
            return None
        df = pl.read_parquet(sv_path)
        t_dates = [_to_iso_date(d) for d in target_dates]
        t_syms = [_normalize_sym6(s) for s in target_symbols]
        d_map = {d: i for i, d in enumerate(t_dates)}
        s_map = {s: j for j, s in enumerate(t_syms)}

        T, N = len(t_dates), len(t_syms)
        out = np.full((T, N), np.nan, dtype=np.float64, order="F")
        src_dates = [_to_iso_date(d) for d in df["trading_date"].to_list()]
        src_syms = [_normalize_sym6(s) for s in df["symbol"].to_list()]
        src_vals = df["spec_var"].to_numpy()

        r_idx: list[int] = []
        c_idx: list[int] = []
        v_idx: list[int] = []
        for k, (d_str, s_str) in enumerate(zip(src_dates, src_syms, strict=True)):
            ti = d_map.get(d_str)
            nj = s_map.get(s_str)
            if ti is not None and nj is not None:
                r_idx.append(ti)
                c_idx.append(nj)
                v_idx.append(k)
        if r_idx:
            out[np.array(r_idx), np.array(c_idx)] = src_vals[np.array(v_idx)]
        return out


@dataclass
class DynamicWLSResult:
    """因果 Constrained WLS 动态特质收益分解结果。"""

    spec_returns: np.ndarray  # (T, N, order='F')
    factor_returns: np.ndarray  # (T, K)
    factor_names: tuple[str, ...]
    industry_codes: tuple[int, ...]
    n_obs_per_day: np.ndarray  # (T,)


def compute_constrained_wls_idiosyncratic_returns(
    returns: np.ndarray,
    style_exposures: dict[str, np.ndarray] | np.ndarray,
    industry: np.ndarray,
    mkt_cap: np.ndarray,
    *,
    valid_mask: np.ndarray | None = None,
    exposure_lag_days: int = 2,
    weight_scheme: str = "sqrt_mkt_cap",
    industry_zero_sum_constraint: bool = True,
    return_winsor_abs: float | None = 0.35,
    ridge_eps: float = 1e-10,
    min_obs: int = 40,
    style_names: Sequence[str] | None = None,
    industry_codes: Sequence[int] | None = None,
) -> DynamicWLSResult:
    """因果横截面 Constrained WLS 动态特质收益分解。

    在每个交易日 ``t >= exposure_lag_days`` 上，仅使用事前暴露 $X_{t-\\Delta}$、
    市值权重 $W_{t-\\Delta}$ 与当日收益 $r_t$ 进行截面加权最小二乘回归：
    - 施加 Barra CNE5 标准市值加权行业零和约束 $\\sum_k \\omega_{k, t-\\Delta} f^{\\text{ind}}_{k,t} = 0$；
    - 估计 $\\hat{f}_t$ 时可选对极端收益 $|r_{i,t}| > \\text{return\\_winsor\\_abs}$ 缩尾；
    - 计算残差 $u_{i,t} = r_{i,t} - X_{i,t-\\Delta} \\hat{f}_t$ 时保留原始未缩尾收益。
    """
    r_mat = np.asarray(returns, dtype=np.float64)
    T, N = r_mat.shape
    if exposure_lag_days < 0:
        raise ValueError(f"exposure_lag_days 必须 >= 0, 实际 {exposure_lag_days}")
    if weight_scheme not in ("sqrt_mkt_cap", "equal"):
        raise ValueError(f"weight_scheme 必须为 'sqrt_mkt_cap' 或 'equal', 实际 {weight_scheme!r}")

    if isinstance(style_exposures, dict):
        s_names = tuple(style_names) if style_names is not None else tuple(style_exposures.keys())
        style_cube = np.stack(
            [np.asarray(style_exposures[k], dtype=np.float64) for k in s_names], axis=-1
        )
    else:
        style_cube = np.asarray(style_exposures, dtype=np.float64)
        if style_cube.ndim != 3 or style_cube.shape[:2] != (T, N):
            raise ValueError(f"style_exposures 形状 {style_cube.shape} 必须为 (T={T}, N={N}, K_s)")
        s_names = (
            tuple(style_names)
            if style_names is not None
            else tuple(f"STYLE_{k}" for k in range(style_cube.shape[2]))
        )

    k_styles = style_cube.shape[2]
    ind_mat = np.asarray(industry, dtype=np.float64)
    cap_mat = np.asarray(mkt_cap, dtype=np.float64)
    if ind_mat.shape != (T, N) or cap_mat.shape != (T, N):
        raise ValueError("industry 与 mkt_cap 形状必须与 returns (T, N) 一致")
    if valid_mask is not None and valid_mask.shape != (T, N):
        raise ValueError("valid_mask 形状必须与 returns (T, N) 一致")

    ind_codes = tuple(
        int(c) for c in (industry_codes if industry_codes is not None else range(1, 32))
    )
    k_ind = len(ind_codes)
    ind_to_pos = {c: idx for idx, c in enumerate(ind_codes)}
    factor_names = s_names + ("country",) + tuple(str(c) for c in ind_codes)
    n_factors = k_styles + 1 + k_ind

    spec_ret = np.full((T, N), np.nan, dtype=np.float64, order="F")
    f_ret = np.full((T, n_factors), np.nan, dtype=np.float64)
    n_obs = np.zeros(T, dtype=np.int32)

    style_fin = np.all(np.isfinite(style_cube), axis=2)

    for t in range(exposure_lag_days, T):
        lag_t = t - exposure_lag_days
        r_t = r_mat[t, :]
        cap_lag = cap_mat[lag_t, :]
        ind_lag = ind_mat[lag_t, :]
        base_ok = (
            np.isfinite(r_t)
            & style_fin[lag_t, :]
            & np.isfinite(cap_lag)
            & (cap_lag > 0.0)
            & np.isfinite(ind_lag)
        )
        if valid_mask is not None:
            base_ok &= valid_mask[lag_t, :]

        cand_idx = np.flatnonzero(base_ok)
        if len(cand_idx) < min_obs:
            continue

        # 将行业编码映射至 0..k_ind-1，过滤未知行业码
        ind_rounded = np.rint(ind_lag[cand_idx]).astype(np.int64)
        ind_pos = np.array([ind_to_pos.get(int(c), -1) for c in ind_rounded], dtype=np.int64)
        in_ind = ind_pos >= 0
        if int(np.sum(in_ind)) < min_obs:
            continue

        u_idx = cand_idx[in_ind]
        u_ind_pos = ind_pos[in_ind]
        m = len(u_idx)
        n_obs[t] = m

        # 构造事前暴露矩阵 X (m, n_factors): [styles, country, ind_dummies]
        X = np.zeros((m, n_factors), dtype=np.float64)
        X[:, :k_styles] = style_cube[lag_t, u_idx, :]
        X[:, k_styles] = 1.0
        X[np.arange(m), k_styles + 1 + u_ind_pos] = 1.0

        caps = cap_lag[u_idx]
        w = np.sqrt(caps) if weight_scheme == "sqrt_mkt_cap" else np.ones(m, dtype=np.float64)
        w = w / max(float(np.mean(w)), 1e-12)

        r_raw = r_t[u_idx]
        r_est = (
            np.clip(r_raw, -float(return_winsor_abs), float(return_winsor_abs))
            if return_winsor_abs is not None and return_winsor_abs > 0.0
            else r_raw
        )

        # 构造约束消除矩阵 R (n_factors, n_free)
        ind_caps = np.bincount(u_ind_pos, weights=caps, minlength=k_ind)
        active_inds = np.flatnonzero(ind_caps > 0.0)
        if len(active_inds) == 0:
            continue

        if industry_zero_sum_constraint and len(active_inds) >= 2:
            # 选取市值最大行业作为消元基准，确保数值最稳健
            ref_pos = int(active_inds[np.argmax(ind_caps[active_inds])])
            other_inds = [int(k) for k in active_inds if int(k) != ref_pos]
            n_free = k_styles + 1 + len(other_inds)
            R = np.zeros((n_factors, n_free), dtype=np.float64)
            R[: k_styles + 1, : k_styles + 1] = np.eye(k_styles + 1, dtype=np.float64)
            ref_cap = float(ind_caps[ref_pos])
            for j_col, k_ind_pos in enumerate(other_inds):
                col = k_styles + 1 + j_col
                R[k_styles + 1 + k_ind_pos, col] = 1.0
                R[k_styles + 1 + ref_pos, col] = -float(ind_caps[k_ind_pos]) / ref_cap
        elif industry_zero_sum_constraint and len(active_inds) == 1:
            # 仅单个行业时，其行业因子必须为 0
            n_free = k_styles + 1
            R = np.zeros((n_factors, n_free), dtype=np.float64)
            R[: k_styles + 1, : k_styles + 1] = np.eye(k_styles + 1, dtype=np.float64)
        else:
            # 无行业零和约束时去掉 country 截距避免共线性，或保留全部活跃行业+country 岭正则
            n_free = k_styles + 1 + len(active_inds)
            R = np.zeros((n_factors, n_free), dtype=np.float64)
            R[: k_styles + 1, : k_styles + 1] = np.eye(k_styles + 1, dtype=np.float64)
            for j_col, k_ind_pos in enumerate(active_inds):
                R[k_styles + 1 + int(k_ind_pos), k_styles + 1 + j_col] = 1.0

        XR = X @ R  # (m, n_free)
        sqrt_w = np.sqrt(w)
        X_w = XR * sqrt_w[:, None]
        y_w = r_est * sqrt_w

        gram = X_w.T @ X_w
        if ridge_eps > 0.0:
            gram = gram + np.eye(n_free, dtype=np.float64) * float(ridge_eps)
        rhs = X_w.T @ y_w

        try:
            theta = np.linalg.solve(gram, rhs)
        except np.linalg.LinAlgError:
            theta, *_ = np.linalg.lstsq(gram, rhs, rcond=None)

        f_t = R @ theta
        f_ret[t, :] = f_t

        # 对全部事前暴露和当日收益均有限的股票输出特质残差 (使用原始未缩尾收益)
        eval_ok = np.isfinite(r_t) & style_fin[lag_t, :] & np.isfinite(ind_lag)
        e_idx = np.flatnonzero(eval_ok)
        if len(e_idx) > 0:
            e_ind_pos = np.array(
                [ind_to_pos.get(int(round(float(c))), -1) for c in ind_lag[e_idx]], dtype=np.int64
            )
            e_valid = e_ind_pos >= 0
            e_idx = e_idx[e_valid]
            e_ind_pos = e_ind_pos[e_valid]
            if len(e_idx) > 0:
                pred = (
                    style_cube[lag_t, e_idx, :] @ f_t[:k_styles]
                    + f_t[k_styles]
                    + f_t[k_styles + 1 + e_ind_pos]
                )
                spec_ret[t, e_idx] = r_t[e_idx] - pred

    return DynamicWLSResult(
        spec_returns=spec_ret,
        factor_returns=f_ret,
        factor_names=factor_names,
        industry_codes=ind_codes,
        n_obs_per_day=n_obs,
    )


def estimate_dynamic_specific_variance(
    spec_ret: np.ndarray,
    mkt_cap: np.ndarray,
    *,
    valid_mask: np.ndarray | None = None,
    window_days: int = 252,
    half_life_days: float = 42.0,
    newey_west_lags: int = 2,
    min_obs: int = 30,
    shrinkage_enabled: bool = True,
    mkt_cap_deciles: int = 10,
    prior_strength: float = 20.0,
    vra_enabled: bool = True,
    vra_half_life_days: float = 21.0,
) -> np.ndarray:
    """严格因果滚动估计每日对角特质方差矩阵 $\\Delta_t$（日频方差量纲，``(T, N, order='F')``）。

    管线步骤（每步在 ``t`` 日仅访问 ``<= t`` 数据）：
    1. 截断窗口 EWMA 原始方差 + $q$ 阶 Bartlett 核 Newey-West 自相关修正（$0.25 \\hat{v}^{\\text{raw}}$ 正定保护）；
    2. 市值分层贝叶斯收缩（按当日市值分 $D$ 组向同组市值加权均值收缩）；
    3. 波动率体制调整 VRA（以昨日预测标准差标准化今日残差并按短半衰期平滑乘子）。
    """
    u = np.asarray(spec_ret, dtype=np.float64)
    caps = np.asarray(mkt_cap, dtype=np.float64)
    T, N = u.shape
    if caps.shape != (T, N):
        raise ValueError(f"mkt_cap 形状 {caps.shape} 必须与 spec_ret {(T, N)} 一致")
    if valid_mask is not None and valid_mask.shape != (T, N):
        raise ValueError("valid_mask 形状必须与 spec_ret 一致")
    if window_days <= 0 or half_life_days <= 0.0 or min_obs <= 0:
        raise ValueError("window_days, half_life_days, min_obs 必须全部 > 0")

    lam = float(0.5 ** (1.0 / half_life_days))
    lam_win = float(lam**window_days)
    q_lags = min(max(0, int(newey_west_lags)), max(0, int(window_days) - 1))

    fin = np.isfinite(u)
    u_clean = np.where(fin, u, 0.0)
    u2 = u_clean * u_clean
    fin_f = fin.astype(np.float64)

    # 流式追踪滞后 0 阶（方差）与滞后 1..q 阶（自协方差）的 EWMA 截断窗口累加器
    num0 = np.zeros(N, dtype=np.float64)
    den0 = np.zeros(N, dtype=np.float64)
    den0_sq = np.zeros(N, dtype=np.float64)  # 用于计算 Kish 有效观测数 n_eff
    cnt = np.zeros(N, dtype=np.int32)
    lam2 = lam * lam
    lam2_win = float(lam2**window_days)

    lag_nums = [np.zeros(N, dtype=np.float64) for _ in range(q_lags)]
    lag_dens = [np.zeros(N, dtype=np.float64) for _ in range(q_lags)]
    lag_decay_win = [
        float(lam ** (window_days - ell)) if window_days > ell else 0.0
        for ell in range(1, q_lags + 1)
    ]

    sigma_shrunk = np.full((T, N), np.nan, dtype=np.float64, order="F")

    for t in range(T):
        f_t = fin[t, :]
        ff_t = fin_f[t, :]
        num0 = lam * num0 + u2[t, :]
        den0 = lam * den0 + ff_t
        den0_sq = lam2 * den0_sq + ff_t
        cnt = cnt + f_t.astype(np.int32)

        if t >= window_days:
            old = t - window_days
            num0 = np.maximum(num0 - lam_win * u2[old, :], 0.0)
            den0 = np.maximum(den0 - lam_win * fin_f[old, :], 0.0)
            den0_sq = np.maximum(den0_sq - lam2_win * fin_f[old, :], 0.0)
            cnt = cnt - fin[old, :].astype(np.int32)

        # 更新 1..q 阶滞后自协方差累加器
        for ell_idx in range(q_lags):
            ell = ell_idx + 1
            if t >= ell:
                pair_ok = f_t & fin[t - ell, :]
                pair_f = pair_ok.astype(np.float64)
                prod_t = np.where(pair_ok, u_clean[t, :] * u_clean[t - ell, :], 0.0)
            else:
                pair_f = np.zeros(N, dtype=np.float64)
                prod_t = np.zeros(N, dtype=np.float64)

            lag_nums[ell_idx] = lam * lag_nums[ell_idx] + prod_t
            lag_dens[ell_idx] = lam * lag_dens[ell_idx] + pair_f

            if t >= window_days:
                old_hi = t - window_days + ell
                old_lo = t - window_days
                if old_hi < t and old_lo >= 0:
                    # 当旧样本对 (old_hi, old_lo) 滑出窗口时剔除其贡献
                    old_ok = fin[old_hi, :] & fin[old_lo, :]
                    old_prod = np.where(old_ok, u_clean[old_hi, :] * u_clean[old_lo, :], 0.0)
                    w_old = lag_decay_win[ell_idx]
                    lag_nums[ell_idx] -= w_old * old_prod
                    lag_dens[ell_idx] = np.maximum(
                        lag_dens[ell_idx] - w_old * old_ok.astype(np.float64), 0.0
                    )

        has_var = (cnt >= min_obs) & (den0 > 1e-12)
        v_raw = np.full(N, np.nan, dtype=np.float64)
        if np.any(has_var):
            v_raw[has_var] = np.maximum(num0[has_var] / den0[has_var], 1e-16)

        v_nw = v_raw.copy()
        if q_lags > 0 and np.any(has_var):
            nw_adj = np.zeros(N, dtype=np.float64)
            for ell_idx in range(q_lags):
                ell = ell_idx + 1
                bartlett = 1.0 - float(ell) / float(q_lags + 1)
                d_ell = lag_dens[ell_idx]
                ok_ell = has_var & (d_ell > 1e-12)
                if np.any(ok_ell):
                    gamma_ell = lag_nums[ell_idx][ok_ell] / d_ell[ok_ell]
                    nw_adj[ok_ell] += 2.0 * bartlett * gamma_ell
            v_nw[has_var] = np.maximum(0.25 * v_raw[has_var], v_raw[has_var] + nw_adj[has_var])

        sigma_nw = np.where(has_var, np.sqrt(np.maximum(v_nw, 1e-16)), np.nan)

        # Kish 有效样本量 n_eff
        n_eff = np.zeros(N, dtype=np.float64)
        ok_kish = den0_sq > 1e-15
        n_eff[ok_kish] = (den0[ok_kish] * den0[ok_kish]) / den0_sq[ok_kish]

        if not shrinkage_enabled:
            sigma_shrunk[t, :] = sigma_nw
            continue

        # 2. 市值分层贝叶斯收缩
        cap_t = caps[t, :]
        cap_ok = np.isfinite(cap_t) & (cap_t > 0.0)
        if valid_mask is not None:
            pool_ok = cap_ok & valid_mask[t, :]
        else:
            pool_ok = cap_ok

        fit_ok = pool_ok & np.isfinite(sigma_nw)
        fit_idx = np.flatnonzero(fit_ok)
        if len(fit_idx) < 5:
            sigma_shrunk[t, :] = sigma_nw
            continue

        n_groups = max(1, min(int(mkt_cap_deciles), len(fit_idx) // 3))
        if n_groups > 1:
            qs = np.linspace(0.0, 100.0, n_groups + 1)[1:-1]
            edges = np.percentile(cap_t[fit_idx], qs)
            fit_grp = np.searchsorted(edges, cap_t[fit_idx], side="right")
            all_grp = np.searchsorted(edges, np.where(cap_ok, cap_t, 0.0), side="right")
        else:
            fit_grp = np.zeros(len(fit_idx), dtype=np.int64)
            all_grp = np.zeros(N, dtype=np.int64)

        global_prior = float(np.average(sigma_nw[fit_idx], weights=cap_t[fit_idx]))
        grp_prior = np.full(n_groups, global_prior, dtype=np.float64)
        for g in range(n_groups):
            in_g = fit_grp == g
            if np.any(in_g):
                w_g = cap_t[fit_idx[in_g]]
                s_g = sigma_nw[fit_idx[in_g]]
                w_sum = float(np.sum(w_g))
                if w_sum > 0.0:
                    grp_prior[g] = float(np.sum(w_g * s_g) / w_sum)

        # 对所有当日有市值的股票进行收缩 (若观测数不足 min_obs，则完全使用组先验)
        target_mask = pool_ok | np.isfinite(sigma_nw)
        t_idx = np.flatnonzero(target_mask)
        prior_vec = np.where(cap_ok[t_idx], grp_prior[all_grp[t_idx]], global_prior)
        has_nw_t = np.isfinite(sigma_nw[t_idx])
        gamma = np.where(
            has_nw_t,
            float(prior_strength) / (float(prior_strength) + np.maximum(n_eff[t_idx], 0.0)),
            1.0,
        )
        nw_safe = np.where(has_nw_t, sigma_nw[t_idx], prior_vec)
        sigma_shrunk[t, t_idx] = (1.0 - gamma) * nw_safe + gamma * prior_vec

    if not vra_enabled:
        return np.asfortranarray(sigma_shrunk * sigma_shrunk, dtype=np.float64)

    # 3. 因果波动率体制调整 (VRA): 使用 t-1 日预测标准差标准化 t 日残差
    lam_vra = float(0.5 ** (1.0 / max(1.0, float(vra_half_life_days))))
    vra_var_ewma = 1.0
    spec_var = np.asfortranarray(sigma_shrunk * sigma_shrunk, dtype=np.float64)

    for t in range(1, T):
        prev_sig = sigma_shrunk[t - 1, :]
        ok_z = fin[t, :] & np.isfinite(prev_sig) & (prev_sig > 1e-8)
        if valid_mask is not None:
            ok_z &= valid_mask[t, :]
        if int(np.sum(ok_z)) >= 10:
            z2 = (u_clean[t, ok_z] / prev_sig[ok_z]) ** 2
            b2_t = float(np.mean(np.clip(z2, 0.0, 25.0)))
            vra_var_ewma = lam_vra * vra_var_ewma + (1.0 - lam_vra) * b2_t
        vra_mult = float(np.clip(vra_var_ewma, 0.5, 2.0))
        spec_var[t, :] = spec_var[t, :] * vra_mult

    return spec_var
