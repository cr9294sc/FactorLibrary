"""通用组合权重辅助工具与多市场权重调度层 (core/portfolio.py)。

按重构架构 M1.6 设计：
- 保留市场无关的截面去均值 (`demean_cross_section`)、权重截断与杠杆归一化
  (`clip_and_normalize_weights`)、调仓缓冲带 (`apply_turnover_buffer`) 与
  价格转收益率 (`compute_daily_returns_from_prices`)；
- 期货专用 `LedoitWolf` 协方差定标已迁至 `markets/futures/portfolio.py`；
- 股票分位等权多头组合构建位于 `markets/stocks/portfolio.py`；
- `build_portfolio_weights` 根据 `MarketProfile` 分发至对应市场实现。
"""

from __future__ import annotations

from typing import Any

import numpy as np

from factorlib.core._hooks import resolve_panel_market, resolve_profile
from factorlib.core.config import PortfolioConfig


def compute_daily_returns_from_prices(prices: np.ndarray) -> np.ndarray:
    """根据复权/原始价格矩阵计算日度时序收益率 (形状 (T, N), 首日 0.0, 缺失处 NaN)。"""
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


def demean_cross_section(raw_scores: np.ndarray) -> np.ndarray:
    """横截面去均值中心化。"""
    if len(raw_scores) == 0:
        return raw_scores.copy()
    return raw_scores - float(np.mean(raw_scores))


def clip_and_normalize_weights(
    raw_weights: np.ndarray,
    max_single_weight: float,
    max_gross_leverage: float,
    long_only: bool = False,
) -> np.ndarray:
    """对单日活跃标的目标权重执行单标的截断与总杠杆上限归一化。"""
    low = 0.0 if long_only else -float(max_single_weight)
    clipped = np.clip(raw_weights, low, float(max_single_weight))
    gross = float(np.sum(np.abs(clipped)))
    if gross > max_gross_leverage and gross > 1e-12:
        clipped = clipped * (float(max_gross_leverage) / gross)
    return clipped


def apply_turnover_buffer(
    weights_row_prev: np.ndarray,
    target_w_full: np.ndarray,
    valid_mask_row: np.ndarray,
    rebalance_buffer: float,
    max_gross_leverage: float,
) -> np.ndarray:
    """单日调仓缓冲带与出池清仓过滤。"""
    n_cols = len(target_w_full)
    out = np.zeros(n_cols, dtype=np.float64)
    for j in range(n_cols):
        if not valid_mask_row[j]:
            out[j] = 0.0
        elif abs(target_w_full[j] - weights_row_prev[j]) < rebalance_buffer:
            out[j] = weights_row_prev[j]
        else:
            out[j] = target_w_full[j]
    t_gross = float(np.sum(np.abs(out)))
    if t_gross > max_gross_leverage and t_gross > 1e-12:
        out = out * (max_gross_leverage / t_gross)
    return out


def build_portfolio_weights(
    scores: np.ndarray,
    returns: np.ndarray,
    mask: np.ndarray,
    cfg: Any | None = None,
    profile: Any | None = None,
    panel: Any | None = None,
) -> np.ndarray:
    """按 `MarketProfile` 分发构造目标连续权重矩阵 W (T x N, order='F')。"""
    resolved_profile = profile
    if resolved_profile is None:
        market = resolve_panel_market(panel, default="futures") if panel is not None else "futures"
        resolved_profile = resolve_profile(market)
    effective_cfg = cfg if cfg is not None else PortfolioConfig()
    return resolved_profile.build_weights(
        scores=scores,
        returns=returns,
        mask=mask,
        config=effective_cfg,
        panel=panel,
    )
