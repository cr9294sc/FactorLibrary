"""期货实盘语义组合构建与连续权重映射 (core/portfolio.py)。

按白皮书 12.1 节与 14 节设计:
1. 截面去均值标准化 (Cross-Sectional De-meaning，仅 CS 模式);
2. 波动率倒数平价缩放 (Inverse Volatility Scaling);
3. 基于 Ledoit-Wolf 收缩协方差的组合目标波动率定标 (Target Volatility Scaling);
4. 单品种权重截断与杠杆率集中度约束 (Truncation & Leverage Constraints);
5. 调仓缓冲带机制 (Turnover Buffer) 减少无谓磨损。
"""

from __future__ import annotations

import warnings

import numpy as np
from sklearn.covariance import LedoitWolf

from factorlib.core.config import PortfolioConfig


def compute_daily_returns_from_prices(prices: np.ndarray) -> np.ndarray:
    """根据复权/原始价格矩阵计算日度时序收益率。

    Parameters
    ----------
    prices : np.ndarray
        形状 (T, N) 的价格矩阵。

    Returns
    -------
    np.ndarray
        形状 (T, N) 的日收益率矩阵, 首日为 0.0, 缺失处为 NaN。
    """
    T, N = prices.shape
    returns = np.full((T, N), np.nan, dtype=np.float64, order="F")
    if T <= 1:
        if T == 1:
            returns[0, :] = 0.0
        return returns

    returns[0, :] = 0.0
    for i in range(N):
        col = prices[:, i]
        for t in range(1, T):
            p_prev = col[t - 1]
            p_curr = col[t]
            if np.isfinite(p_prev) and np.isfinite(p_curr) and p_prev > 0.0:
                returns[t, i] = (p_curr - p_prev) / p_prev
            else:
                returns[t, i] = np.nan
    return returns


def _estimate_shrinkage_covariance(
    returns_sub: np.ndarray,
    active_indices: list[int],
    fallback_vols: np.ndarray,
) -> np.ndarray:
    """使用 Ledoit-Wolf 估计收益率收缩协方差矩阵 (年化)。

    Parameters
    ----------
    returns_sub : np.ndarray
        形状 (W, K) 的活跃品种历史收益率窗口数据。
    active_indices : list[int]
        活跃品种索引。
    fallback_vols : np.ndarray
        各活跃品种的年化已实现波动率 (K,)。

    Returns
    -------
    np.ndarray
        形状 (K, K) 的年化协方差矩阵。
    """
    w_len, k = returns_sub.shape
    if k == 1 or w_len <= 1:
        diag_vals = np.maximum(fallback_vols * fallback_vols, 1e-6)
        return np.diag(diag_vals)

    # 填充缺失值 (以各品种均值或 0 填充)
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
                # 确保半正定
                cov = 0.5 * (cov + cov.T)
                min_eig = np.min(np.linalg.eigvalsh(cov))
                if min_eig < 1e-8:
                    cov += (1e-8 - min_eig) * np.eye(k)
                return cov
        except Exception:
            pass

    # 降级备用: 样本协方差 + 对角收缩
    try:
        sample_cov = np.cov(clean_r, rowvar=False) * 252.0
        if sample_cov.ndim == 0:
            sample_cov = np.array([[float(sample_cov)]])
        diag = np.diag(np.diag(sample_cov))
        cov = 0.8 * sample_cov + 0.2 * diag
        cov = 0.5 * (cov + cov.T)
        return cov
    except Exception:
        # 完全对角阵
        diag_vals = np.maximum(fallback_vols * fallback_vols, 1e-6)
        return np.diag(diag_vals)


def build_portfolio_weights(
    scores: np.ndarray,
    returns: np.ndarray,
    mask: np.ndarray,
    cfg: PortfolioConfig,
) -> np.ndarray:
    """从截面预测评分映射为严谨实盘目标连续权重矩阵 W (T x N)。

    Parameters
    ----------
    scores : np.ndarray
        形状 (T, N) 的预测信号/模型评分矩阵。
    returns : np.ndarray
        形状 (T, N) 的因果历史日收益率矩阵。
    mask : np.ndarray
        形状 (T, N) 的可交易品种池布尔矩阵。
    cfg : PortfolioConfig
        组合权重构建与风控约束配置。

    Returns
    -------
    np.ndarray
        形状 (T, N) 的连续目标权重矩阵 (Fortran-order, float64)。
    """
    T, N = scores.shape
    if returns.shape != (T, N) or mask.shape != (T, N):
        raise ValueError(
            f"scores {scores.shape}, returns {returns.shape}, mask {mask.shape} 维度必须完全一致"
        )

    weights = np.zeros((T, N), dtype=np.float64, order="F")

    for t in range(T):
        # 1. 提取 T 日有效可交易品种
        valid_mask = mask[t, :] & np.isfinite(scores[t, :])
        active_idx = np.where(valid_mask)[0]
        k_active = len(active_idx)

        if k_active == 0:
            continue

        raw_scores = scores[t, active_idx].copy()

        # 2. 截面去均值 (仅 CS 模式)
        if cfg.mode == "cs":
            mean_score = float(np.mean(raw_scores))
            demeaned_scores = raw_scores - mean_score
        else:
            demeaned_scores = raw_scores

        # 若去均值后全为 0，跳过
        if np.all(np.abs(demeaned_scores) < 1e-12):
            continue

        # 3. 波动率倒数平价缩放 (Inverse Volatility Scaling)
        # 回看近 vol_lookback 日的历史收益率 (严格因果, 仅用 <= t 日的数据)
        vol_start = max(0, t - cfg.vol_lookback + 1)
        r_vol_window = returns[vol_start : t + 1, active_idx]  # (W, K)

        ann_vols = np.zeros(k_active, dtype=np.float64)
        for i_col in range(k_active):
            col_ret = r_vol_window[:, i_col]
            valid_rets = col_ret[np.isfinite(col_ret)]
            if len(valid_rets) >= 5:
                vol = float(np.std(valid_rets, ddof=1) * np.sqrt(252.0))
            else:
                vol = 0.20  # 缺省历史波动率 20%
            ann_vols[i_col] = max(vol, 1e-4)

        # 逆波动率初步加权
        inv_vol_w = demeaned_scores / ann_vols

        # 4. 目标波动率定标 (Target Volatility Scaling)
        cov_start = max(0, t - cfg.cov_shrink_lookback + 1)
        r_cov_window = returns[cov_start : t + 1, active_idx]
        cov_matrix = _estimate_shrinkage_covariance(r_cov_window, list(active_idx), ann_vols)

        port_var = float(inv_vol_w @ cov_matrix @ inv_vol_w)
        port_vol = np.sqrt(max(port_var, 1e-12))

        gross_abs = float(np.sum(np.abs(inv_vol_w)))
        if port_vol > 1e-6 and gross_abs > 1e-12:
            scale_vol = cfg.target_volatility / port_vol
            scale_lev = cfg.max_gross_leverage / gross_abs
            scale = min(scale_vol, scale_lev)
            target_w_active = inv_vol_w * scale
        else:
            target_w_active = inv_vol_w

        # 5. 单品种最大权重截断与杠杆率集中度约束
        target_w_active = np.clip(target_w_active, -cfg.max_single_weight, cfg.max_single_weight)
        current_gross = float(np.sum(np.abs(target_w_active)))
        if current_gross > cfg.max_gross_leverage:
            target_w_active = target_w_active * (cfg.max_gross_leverage / current_gross)

        # 6. 调仓缓冲带机制 (Turnover Buffer)
        target_w_full = np.zeros(N, dtype=np.float64)
        target_w_full[active_idx] = target_w_active

        if t == 0:
            weights[0, :] = target_w_full
        else:
            w_prev = weights[t - 1, :]
            for j in range(N):
                if not valid_mask[j]:
                    # 离开品种池或停牌/数据缺失，权重归零
                    weights[t, j] = 0.0
                else:
                    diff = abs(target_w_full[j] - w_prev[j])
                    if diff < cfg.rebalance_buffer:
                        # 变动在缓冲带内，保持前一日权重
                        weights[t, j] = w_prev[j]
                    else:
                        weights[t, j] = target_w_full[j]

            # 再次检查缓冲后总杠杆约束
            t_gross = float(np.sum(np.abs(weights[t, :])))
            if t_gross > cfg.max_gross_leverage:
                weights[t, :] = weights[t, :] * (cfg.max_gross_leverage / t_gross)

    return np.asfortranarray(weights, dtype=np.float64)
