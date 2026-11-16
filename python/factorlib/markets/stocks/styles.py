"""A 股风格因子因果计算实现 (markets/stocks/styles.py)。"""

from __future__ import annotations

import numpy as np

from factorlib.core.ops_ref import ts_std as ts_std_ref
from factorlib.core.panel import Panel
from factorlib.core.styles import (
    _causal_pct_change_allow_short,
    _causal_rolling_mean_allow_partial,
    _normalize_style_matrix,
)

STOCK_STYLE_NAMES: tuple[str, ...] = (
    "SIZE",
    "VOLATILITY",
    "MOMENTUM",
    "LIQUIDITY",
)


def compute_stocks_styles(panel: Panel) -> dict[str, np.ndarray]:
    """根据当前股票 Panel 因果计算标准风格暴露矩阵字典 (形状 [T, N])。"""
    T, N = panel.shape
    fields = panel.fields
    styles: dict[str, np.ndarray] = {}

    close_mat = fields.get("close_hfq", fields.get("close"))

    # 1. SIZE / LIQUIDITY: 基于 20 日均成交额对数（缺失时回退 log_amount）
    if "amount" in fields:
        amt_ma = _causal_rolling_mean_allow_partial(fields["amount"], 20)
        with np.errstate(invalid="ignore", divide="ignore"):
            raw_size = np.where(np.isfinite(amt_ma) & (amt_ma > 0.0), np.log(amt_ma), np.nan)
    elif "log_amount" in fields:
        raw_size = _causal_rolling_mean_allow_partial(fields["log_amount"], 20)
    elif close_mat is not None and "volume" in fields:
        amt_ma = _causal_rolling_mean_allow_partial(close_mat * fields["volume"], 20)
        with np.errstate(invalid="ignore", divide="ignore"):
            raw_size = np.where(np.isfinite(amt_ma) & (amt_ma > 0.0), np.log(amt_ma), np.nan)
    else:
        raw_size = np.full((T, N), np.nan, dtype=np.float64)
    styles["SIZE"] = _normalize_style_matrix(raw_size, mask=panel.valid_mask)
    styles["LIQUIDITY"] = styles["SIZE"]

    # 2. VOLATILITY: c2c_ret 或 close_hfq 日收益 20 日标准差
    if "c2c_ret" in fields:
        raw_vol = ts_std_ref(fields["c2c_ret"], min(20, max(2, T)))
    elif close_mat is not None:
        ret1 = _causal_pct_change_allow_short(close_mat, 1)
        raw_vol = ts_std_ref(ret1, min(20, max(2, T)))
    else:
        raw_vol = np.full((T, N), np.nan, dtype=np.float64)
    styles["VOLATILITY"] = _normalize_style_matrix(raw_vol, mask=panel.valid_mask)

    # 3. MOMENTUM: 20 日复权收盘动量
    if close_mat is not None:
        raw_mom = _causal_pct_change_allow_short(close_mat, 20)
    else:
        raw_mom = np.full((T, N), np.nan, dtype=np.float64)
    styles["MOMENTUM"] = _normalize_style_matrix(raw_mom, mask=panel.valid_mask)

    return styles
