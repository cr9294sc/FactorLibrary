"""期货 6 大标准风格因子因果计算实现 (markets/futures/styles.py)。"""

from __future__ import annotations

import numpy as np

from factorlib.core.ops_ref import ts_std as ts_std_ref
from factorlib.core.panel import Panel
from factorlib.core.styles import (
    _causal_pct_change_allow_short,
    _causal_rolling_mean_allow_partial,
    _normalize_style_matrix,
)


def compute_futures_styles(panel: Panel) -> dict[str, np.ndarray]:
    """根据当前 Panel 因果计算 6 大期货标准风格因子矩阵字典 (各矩阵形状为 [T, N])。"""
    T, N = panel.shape
    fields = panel.fields
    styles: dict[str, np.ndarray] = {}

    # 1. SIZE: log(ts_mean(turnover, 20))
    if "turnover" in fields:
        turn_ma = _causal_rolling_mean_allow_partial(fields["turnover"], 20)
    elif panel.has_any_field("adv20_turnover"):
        turn_ma = np.array(panel.get_any_field("adv20_turnover"), dtype=np.float64, copy=True)
    elif "close" in fields and "volume" in fields:
        turn_ma = _causal_rolling_mean_allow_partial(fields["close"] * fields["volume"], 20)
    else:
        turn_ma = np.full((T, N), np.nan, dtype=np.float64)

    with np.errstate(invalid="ignore", divide="ignore"):
        raw_size = np.where(np.isfinite(turn_ma) & (turn_ma > 0.0), np.log(turn_ma), np.nan)
    styles["SIZE"] = _normalize_style_matrix(raw_size, mask=panel.valid_mask)

    # 2. VOLATILITY: ts_mean(rv, 20) 或日收益标准差
    if "rv" in fields:
        raw_vol = _causal_rolling_mean_allow_partial(fields["rv"], 20)
    elif "close" in fields:
        ret1 = _causal_pct_change_allow_short(fields["close"], 1)
        raw_vol = ts_std_ref(ret1, min(20, max(2, T)))
    else:
        raw_vol = np.full((T, N), np.nan, dtype=np.float64)
    styles["VOLATILITY"] = _normalize_style_matrix(raw_vol, mask=panel.valid_mask)

    # 3. MOMENTUM: pct_change(close, 20)
    if "close" in fields:
        raw_mom = _causal_pct_change_allow_short(fields["close"], 20)
    else:
        raw_mom = np.full((T, N), np.nan, dtype=np.float64)
    styles["MOMENTUM"] = _normalize_style_matrix(raw_mom, mask=panel.valid_mask)

    # 4. CARRY: ry_ann 或 ry_main_sub
    if "ry_ann" in fields:
        raw_carry = np.array(fields["ry_ann"], dtype=np.float64, copy=True)
    elif "ry_main_sub" in fields:
        raw_carry = np.array(fields["ry_main_sub"], dtype=np.float64, copy=True)
    else:
        raw_carry = np.full((T, N), np.nan, dtype=np.float64)
    styles["CARRY"] = _normalize_style_matrix(raw_carry, mask=panel.valid_mask)

    # 5. INVENTORY: -wh_z60 或 -wh_chg5
    if "wh_z60" in fields:
        raw_inv = -np.array(fields["wh_z60"], dtype=np.float64, copy=True)
    elif "wh_chg5" in fields:
        raw_inv = -np.array(fields["wh_chg5"], dtype=np.float64, copy=True)
    else:
        raw_inv = np.full((T, N), np.nan, dtype=np.float64)
    styles["INVENTORY"] = _normalize_style_matrix(raw_inv, mask=panel.valid_mask)

    # 6. POSITIONING: mr_net_ratio 或 mr_net_ratio_chg5
    if "mr_net_ratio" in fields:
        raw_pos = np.array(fields["mr_net_ratio"], dtype=np.float64, copy=True)
    elif "mr_net_ratio_chg5" in fields:
        raw_pos = np.array(fields["mr_net_ratio_chg5"], dtype=np.float64, copy=True)
    else:
        raw_pos = np.full((T, N), np.nan, dtype=np.float64)
    styles["POSITIONING"] = _normalize_style_matrix(raw_pos, mask=panel.valid_mask)

    return styles
