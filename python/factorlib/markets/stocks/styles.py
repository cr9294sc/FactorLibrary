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
    """根据当前股票 Panel 因果计算标准风格暴露矩阵字典 (形状 [T, N])，支持面板级缓存。"""
    T, N = panel.shape
    fields = panel.fields

    # 缓存键：绑定面板形状与核心输入字段内存标识及末行校验值，确保摄动测试修改字段后自动失效重算
    close_mat = fields.get("close_hfq", fields.get("close"))
    amt_mat = fields.get("amount", fields.get("log_amount"))
    c2c_mat = fields.get("c2c_ret")
    cache_key = (
        T,
        N,
        id(panel.valid_mask),
        int(np.sum(panel.valid_mask[-1])) if T > 0 else 0,
        id(close_mat),
        float(np.nan_to_num(close_mat[-1, 0], nan=-999.0))
        if close_mat is not None and close_mat.size
        else 0.0,
        id(amt_mat),
        float(np.nan_to_num(amt_mat[-1, 0], nan=-999.0))
        if amt_mat is not None and amt_mat.size
        else 0.0,
        id(c2c_mat),
        float(np.nan_to_num(c2c_mat[-1, 0], nan=-999.0))
        if c2c_mat is not None and c2c_mat.size
        else 0.0,
    )
    cached = getattr(panel, "_cached_stocks_styles", None)
    if cached is not None and cached[0] == cache_key:
        return cached[1]

    styles: dict[str, np.ndarray] = {}

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
    vol_win = min(20, max(2, T))
    if "c2c_ret" in fields:
        if T >= vol_win and N > 200:
            raw_vol = np.asarray(
                panel.to_rust_engine().eval(f"ts_std(c2c_ret, {vol_win})"), dtype=np.float64
            )
        else:
            raw_vol = ts_std_ref(fields["c2c_ret"], vol_win)
    elif close_mat is not None:
        ret1 = _causal_pct_change_allow_short(close_mat, 1)
        raw_vol = ts_std_ref(ret1, vol_win)
    else:
        raw_vol = np.full((T, N), np.nan, dtype=np.float64)
    styles["VOLATILITY"] = _normalize_style_matrix(raw_vol, mask=panel.valid_mask)

    # 3. MOMENTUM: 20 日复权收盘动量
    if close_mat is not None:
        raw_mom = _causal_pct_change_allow_short(close_mat, 20)
    else:
        raw_mom = np.full((T, N), np.nan, dtype=np.float64)
    styles["MOMENTUM"] = _normalize_style_matrix(raw_mom, mask=panel.valid_mask)

    object.__setattr__(panel, "_cached_stocks_styles", (cache_key, styles))
    return styles
