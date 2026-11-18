"""A 股分位等权多头组合构建器 (markets/stocks/portfolio.py)。

按重构计划 M1.6 设计：
- 分位等权多头（默认 Top 10%）；
- 支持可选行业中性选股 (`industry_neutral=True` 时在每个申万一级行业内按分位精选)；
- 单只个股权重上限 (`max_single_weight`) 与总杠杆归一化；
- 换手缓冲带 (`rebalance_buffer`)；
- **严禁使用全截面协方差矩阵**（避免大截面 N=5000 的病态求逆与内存膨胀）。
"""

from __future__ import annotations

import math

import numpy as np

from factorlib.core.config import StockPortfolioConfig
from factorlib.core.portfolio import apply_turnover_buffer, clip_and_normalize_weights


def build_stocks_portfolio_weights(
    scores: np.ndarray,
    returns: np.ndarray,
    mask: np.ndarray,
    cfg: StockPortfolioConfig | None = None,
    sector: np.ndarray | None = None,
) -> np.ndarray:
    """构建 A 股分位等权多头目标权重矩阵 W (T x N, order='F')。"""
    del returns  # 股票组合构建不使用全协方差或历史波动率矩阵
    if cfg is None:
        cfg = StockPortfolioConfig()

    T, N = scores.shape
    if mask.shape != (T, N):
        raise ValueError(f"scores {scores.shape} 与 mask {mask.shape} 维度必须完全一致")
    if sector is not None and sector.shape != (T, N):
        raise ValueError(f"sector {sector.shape} 与 scores {scores.shape} 维度必须完全一致")

    weights = np.zeros((T, N), dtype=np.float64, order="F")

    for t in range(T):
        s_row = scores[t, :]
        valid_mask = mask[t, :] & np.isfinite(s_row)
        active_idx = np.flatnonzero(valid_mask)
        k_active = len(active_idx)
        if k_active == 0:
            continue

        s_act = s_row[active_idx]
        if cfg.industry_neutral and sector is not None:
            sec_row = sector[t, active_idx]
            finite_sec = np.isfinite(sec_row)
            if np.any(finite_sec):
                codes = np.where(finite_sec, np.rint(sec_row).astype(np.int32), -1)
                order = np.lexsort((-s_act, codes))
                sorted_codes = codes[order]
                sorted_idx = active_idx[order]
                _, group_starts, group_counts = np.unique(
                    sorted_codes, return_index=True, return_counts=True
                )
                sel_parts = [
                    sorted_idx[st_i : st_i + max(1, int(math.ceil(cnt * cfg.top_quantile)))]
                    for st_i, cnt in zip(group_starts.tolist(), group_counts.tolist(), strict=False)
                ]
                selected_arr = (
                    np.concatenate(sel_parts) if sel_parts else np.empty(0, dtype=np.int64)
                )
            else:
                k_top = max(1, int(math.ceil(k_active * cfg.top_quantile)))
                order = np.argsort(-s_act, kind="mergesort")
                selected_arr = active_idx[order[:k_top]]
        else:
            k_top = max(1, int(math.ceil(k_active * cfg.top_quantile)))
            order = np.argsort(-s_act, kind="mergesort")
            selected_arr = active_idx[order[:k_top]]

        n_sel = len(selected_arr)
        if n_sel == 0:
            continue

        raw_w = np.full(n_sel, float(cfg.max_gross_leverage) / float(n_sel), dtype=np.float64)
        capped_w = clip_and_normalize_weights(
            raw_w,
            max_single_weight=cfg.max_single_weight,
            max_gross_leverage=cfg.max_gross_leverage,
            long_only=cfg.long_only,
        )

        target_w_full = np.zeros(N, dtype=np.float64)
        target_w_full[selected_arr] = capped_w

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
