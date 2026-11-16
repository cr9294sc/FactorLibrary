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
        valid_mask = mask[t, :] & np.isfinite(scores[t, :])
        active_idx = np.where(valid_mask)[0]
        k_active = len(active_idx)
        if k_active == 0:
            continue

        selected: list[int] = []
        if cfg.industry_neutral and sector is not None:
            sec_row = sector[t, active_idx]
            finite_sec = np.isfinite(sec_row)
            if np.any(finite_sec):
                buckets: dict[int, list[int]] = {}
                for pos, idx in enumerate(active_idx):
                    code = int(round(sec_row[pos])) if finite_sec[pos] else -1
                    buckets.setdefault(code, []).append(int(idx))
                for _code, idxs in buckets.items():
                    m_sec = len(idxs)
                    k_top = max(1, int(math.ceil(m_sec * cfg.top_quantile)))
                    idxs_sorted = sorted(idxs, key=lambda j: float(scores[t, j]), reverse=True)
                    selected.extend(idxs_sorted[:k_top])
            else:
                k_top = max(1, int(math.ceil(k_active * cfg.top_quantile)))
                order = np.argsort(scores[t, active_idx], kind="mergesort")[::-1]
                selected = active_idx[order[:k_top]].tolist()
        else:
            k_top = max(1, int(math.ceil(k_active * cfg.top_quantile)))
            order = np.argsort(scores[t, active_idx], kind="mergesort")[::-1]
            selected = active_idx[order[:k_top]].tolist()

        if not selected:
            continue

        raw_w = np.zeros(len(selected), dtype=np.float64)
        raw_w[:] = float(cfg.max_gross_leverage) / float(len(selected))
        capped_w = clip_and_normalize_weights(
            raw_w,
            max_single_weight=cfg.max_single_weight,
            max_gross_leverage=cfg.max_gross_leverage,
            long_only=cfg.long_only,
        )

        target_w_full = np.zeros(N, dtype=np.float64)
        for pos, col_idx in enumerate(selected):
            target_w_full[col_idx] = capped_w[pos]

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
