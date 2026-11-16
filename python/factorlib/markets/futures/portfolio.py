"""期货实盘语义组合构建与目标波动率定标 (markets/futures/portfolio.py)。

基于 `core/portfolio.py` 的通用截断与缓冲带原语，叠加期货专属的：
- 波动率倒数平价缩放 (Inverse Volatility Scaling)；
- 基于 Ledoit-Wolf 收缩协方差的组合目标波动率定标 (Target Volatility Scaling)。
"""

from __future__ import annotations

import warnings

import numpy as np
from sklearn.covariance import LedoitWolf

from factorlib.core.config import PortfolioConfig
from factorlib.core.portfolio import (
    apply_turnover_buffer,
    clip_and_normalize_weights,
    demean_cross_section,
)


def _estimate_shrinkage_covariance(
    returns_sub: np.ndarray,
    active_indices: list[int],
    fallback_vols: np.ndarray,
) -> np.ndarray:
    """使用 Ledoit-Wolf 估计收益率收缩协方差矩阵 (年化)。"""
    del active_indices
    w_len, k = returns_sub.shape
    if k == 1 or w_len <= 1:
        diag_vals = np.maximum(fallback_vols * fallback_vols, 1e-6)
        return np.diag(diag_vals)

    clean_r = returns_sub.copy()
    for col in range(k):
        vals = clean_r[:, col]
        mask_nan = ~np.isfinite(vals)
        if np.all(mask_nan):
            clean_r[:, col] = 0.0
        elif np.any(mask_nan):
            mean_val = float(np.nanmean(vals))
            clean_r[mask_nan, col] = mean_val

    if w_len >= 10:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                lw = LedoitWolf(assume_centered=False)
                lw.fit(clean_r)
                cov = lw.covariance_ * 252.0
                cov = 0.5 * (cov + cov.T)
                min_eig = np.min(np.linalg.eigvalsh(cov))
                if min_eig < 1e-8:
                    cov += (1e-8 - min_eig) * np.eye(k)
                return cov
        except Exception:
            pass

    try:
        sample_cov = np.cov(clean_r, rowvar=False) * 252.0
        if sample_cov.ndim == 0:
            sample_cov = np.array([[float(sample_cov)]])
        diag = np.diag(np.diag(sample_cov))
        cov = 0.8 * sample_cov + 0.2 * diag
        cov = 0.5 * (cov + cov.T)
        return cov
    except Exception:
        diag_vals = np.maximum(fallback_vols * fallback_vols, 1e-6)
        return np.diag(diag_vals)


def build_futures_portfolio_weights(
    scores: np.ndarray,
    returns: np.ndarray,
    mask: np.ndarray,
    cfg: PortfolioConfig | None = None,
) -> np.ndarray:
    """从截面预测评分映射为期货实盘目标连续权重矩阵 W (T x N, order='F')。"""
    if cfg is None:
        cfg = PortfolioConfig()

    T, N = scores.shape
    if returns.shape != (T, N) or mask.shape != (T, N):
        raise ValueError(
            f"scores {scores.shape}, returns {returns.shape}, mask {mask.shape} 维度必须完全一致"
        )

    weights = np.zeros((T, N), dtype=np.float64, order="F")

    for t in range(T):
        valid_mask = mask[t, :] & np.isfinite(scores[t, :])
        active_idx = np.where(valid_mask)[0]
        k_active = len(active_idx)

        if k_active == 0:
            continue

        raw_scores = scores[t, active_idx].copy()
        demeaned_scores = demean_cross_section(raw_scores) if cfg.mode == "cs" else raw_scores

        if np.all(np.abs(demeaned_scores) < 1e-12):
            continue

        vol_start = max(0, t - cfg.vol_lookback + 1)
        r_vol_window = returns[vol_start : t + 1, active_idx]

        ann_vols = np.zeros(k_active, dtype=np.float64)
        for i_col in range(k_active):
            col_ret = r_vol_window[:, i_col]
            valid_rets = col_ret[np.isfinite(col_ret)]
            if len(valid_rets) >= 5:
                vol = float(np.std(valid_rets, ddof=1) * np.sqrt(252.0))
            else:
                vol = 0.20
            ann_vols[i_col] = max(vol, 1e-4)

        inv_vol_w = demeaned_scores / ann_vols

        cov_start = max(0, t - cfg.cov_shrink_lookback + 1)
        r_cov_window = returns[cov_start : t + 1, active_idx]
        cov_matrix = _estimate_shrinkage_covariance(r_cov_window, list(active_idx), ann_vols)

        port_var = float(inv_vol_w @ cov_matrix @ inv_vol_w)
        port_vol = np.sqrt(max(port_var, 1e-12))

        gross_abs = float(np.sum(np.abs(inv_vol_w)))
        if port_vol > 1e-6 and gross_abs > 1e-12:
            scale_vol = cfg.target_volatility / port_vol
            scale_lev = cfg.max_gross_leverage / gross_abs
            target_w_active = inv_vol_w * min(scale_vol, scale_lev)
        else:
            target_w_active = inv_vol_w

        target_w_active = clip_and_normalize_weights(
            target_w_active,
            max_single_weight=cfg.max_single_weight,
            max_gross_leverage=cfg.max_gross_leverage,
            long_only=False,
        )

        target_w_full = np.zeros(N, dtype=np.float64)
        target_w_full[active_idx] = target_w_active

        if t == 0:
            weights[0, :] = target_w_full
        else:
            weights[t, :] = apply_turnover_buffer(
                weights[t - 1, :],
                target_w_full,
                valid_mask,
                rebalance_buffer=cfg.rebalance_buffer,
                max_gross_leverage=cfg.max_gross_leverage,
            )

    return np.asfortranarray(weights, dtype=np.float64)
