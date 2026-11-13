"""期货 6 大标准风格因子定义、因果计算与风格暴露归因 (core/styles.py)。

对标 FactorLab MCP 架构中的 Barra 风险/风格因子体系，专为商品与金融期货设计：
1. SIZE (规模与流动性): log(ts_mean(turnover, 20))
2. VOLATILITY (已实现波动率): ts_mean(rv, 20) 或 ts_std(pct_change(close, 1), 20)
3. MOMENTUM (中短期趋势动量): pct_change(close, 20)
4. CARRY (期限结构展期收益): ry_ann (主力-次主力年化展期收益率)
5. INVENTORY (基本面库存紧缺度): -wh_z60 (仓单 60 日滚动 Z-Score 取反)
6. POSITIONING (主力席位净持仓分歧): mr_net_ratio (会员多空前 20 净多头占比)

所有风格因子均严格遵循因果性（仅使用截至 T 日已知数据），并在每个交易日横截面上
经过 C++ 级语义的 `MAD(3.0) -> zScore(ddof=0)` 标准化处理。
既可作为 `riskNeut` / `riskNeutMulti` 后处理算子的风险暴露输入，也可用于回测报告中的风格暴露归因。
"""

from __future__ import annotations

from typing import Any

import factorlib_core
import numpy as np

from factorlib.core.ops_ref import ts_mean as ts_mean_ref
from factorlib.core.ops_ref import ts_std as ts_std_ref
from factorlib.core.panel import Panel

STYLE_NAMES: tuple[str, ...] = (
    "SIZE",
    "VOLATILITY",
    "MOMENTUM",
    "CARRY",
    "INVENTORY",
    "POSITIONING",
)


def _pearson_corr_1d(x: np.ndarray, y: np.ndarray) -> float:
    """计算一维有效向量的 Pearson 相关系数。"""
    if len(x) < 3:
        return np.nan
    mx = float(np.mean(x))
    my = float(np.mean(y))
    dx = x - mx
    dy = y - my
    vx = float(np.sum(dx * dx))
    vy = float(np.sum(dy * dy))
    if vx <= 1e-16 or vy <= 1e-16:
        return np.nan
    return float(np.sum(dx * dy) / np.sqrt(vx * vy))


def _spearman_corr_1d(x: np.ndarray, y: np.ndarray) -> float:
    """计算一维有效向量的 Spearman 秩相关系数（支持并列平均秩）。"""
    n = len(x)
    if n < 3:
        return np.nan

    def _avg_ranks(arr: np.ndarray) -> np.ndarray:
        order = np.argsort(arr, kind="mergesort")
        ranks = np.empty(n, dtype=np.float64)
        i = 0
        while i < n:
            j = i + 1
            v = arr[order[i]]
            while j < n and arr[order[j]] == v:
                j += 1
            avg_r = 0.5 * (i + j - 1) + 1.0
            for m in range(i, j):
                ranks[order[m]] = avg_r
            i = j
        return ranks

    return _pearson_corr_1d(_avg_ranks(x), _avg_ranks(y))


STYLE_DEFINITIONS: list[dict[str, Any]] = [
    {
        "name": "SIZE",
        "display_name": "规模与流动性 (Size & Liquidity)",
        "category": "style",
        "formula": "zScore(MAD(log(ts_mean(turnover, 20)), 3.0))",
        "required_fields": ["turnover"],
        "lookback": 20,
        "description": "主力合约 20 日均成交额对数，刻画品种容量与市场沉淀资金规模",
    },
    {
        "name": "VOLATILITY",
        "display_name": "波动率风险 (Realized Volatility)",
        "category": "style",
        "formula": "zScore(MAD(ts_mean(rv, 20), 3.0))",
        "required_fields": ["rv"],
        "lookback": 20,
        "description": "日内高频已实现波动率 rv 的 20 日均值（缺失时回退为日收益 20 日标准差）",
    },
    {
        "name": "MOMENTUM",
        "display_name": "截面动量 (Cross-Sectional Momentum)",
        "category": "style",
        "formula": "zScore(MAD(pct_change(close, 20), 3.0))",
        "required_fields": ["close"],
        "lookback": 21,
        "description": "比例复权收盘价 20 日累计收益率，刻画短中期趋势暴露",
    },
    {
        "name": "CARRY",
        "display_name": "期限结构展期 (Term Structure Carry)",
        "category": "style",
        "formula": "zScore(MAD(ry_ann, 3.0))",
        "required_fields": ["ry_ann"],
        "lookback": 1,
        "description": "主力与次主力合约年化展期收益率，贴水 (Backwardation) 为正暴露",
    },
    {
        "name": "INVENTORY",
        "display_name": "库存紧缺度 (Inventory Scarcity)",
        "category": "style",
        "formula": "zScore(MAD(-wh_z60, 3.0))",
        "required_fields": ["wh_z60"],
        "lookback": 1,
        "description": "注册仓单 60 日时序 Z-Score 的负值，低库存/去库对应正暴露",
    },
    {
        "name": "POSITIONING",
        "display_name": "席位净持仓 (Member Net Positioning)",
        "category": "style",
        "formula": "zScore(MAD(mr_net_ratio, 3.0))",
        "required_fields": ["mr_net_ratio"],
        "lookback": 1,
        "description": "前 20 主力会员席位净多头持仓占比 (L-S)/(L+S)",
    },
]


def _normalize_style_matrix(raw: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
    """对原始风格矩阵逐行执行 MAD(3.0) 缩尾 + zScore(ddof=0) 标准化。"""
    filtered = np.where(mask, raw, np.nan) if mask is not None else raw
    f_raw = np.asfortranarray(filtered, dtype=np.float64)
    mad_clipped = np.asarray(factorlib_core.apply_postprocess_op("MAD", f_raw, param=3.0))
    z_out = np.asarray(
        factorlib_core.apply_postprocess_op("zScore", np.asfortranarray(mad_clipped))
    )
    return np.asfortranarray(z_out, dtype=np.float64)


def _causal_rolling_mean_allow_partial(x: np.ndarray, window: int) -> np.ndarray:
    """严格因果的滚动均值：若有效长度达到 window 则采用标准 ts_mean，对前段样本做因果累积均值兜底。"""
    res = ts_mean_ref(x, window)
    T, N = x.shape
    # 对前 window-1 行（或因缺失未满 window 的行），使用截至当前行 t 的因果可用均值补充，避免短样本全 NaN
    for j in range(N):
        col = x[:, j]
        for t in range(T):
            if np.isnan(res[t, j]) and np.isfinite(col[t]):
                start = max(0, t + 1 - window)
                seg = col[start : t + 1]
                valid = seg[np.isfinite(seg)]
                if len(valid) >= 1:
                    res[t, j] = float(np.mean(valid))
    return res


def _causal_pct_change_allow_short(close: np.ndarray, window: int) -> np.ndarray:
    """严格因果的收益率计算：优先使用 t - window，若 t < window 且 t >= 1 则使用首日价格兜底。"""
    T, N = close.shape
    out = np.full((T, N), np.nan, dtype=np.float64)
    for j in range(N):
        col = close[:, j]
        for t in range(1, T):
            lag = min(window, t)
            prev = col[t - lag]
            curr = col[t]
            if np.isfinite(curr) and np.isfinite(prev) and abs(prev) > 1e-12:
                out[t, j] = curr / prev - 1.0
    return out


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


def compute_style_exposures(
    factor_mat: np.ndarray,
    styles_dict: dict[str, np.ndarray],
    mask: np.ndarray | None = None,
    n_quantiles: int = 5,
    sign: int = 1,
) -> dict[str, Any]:
    """计算因子在 6 大期货风格维度上的分组暴露、多头超额暴露与截面秩相关系数。"""
    T, N = factor_mat.shape
    style_names = [s for s in STYLE_NAMES if s in styles_dict]

    group_accum: dict[str, dict[str, list[float]]] = {
        f"G{q + 1}": {s: [] for s in style_names} for q in range(n_quantiles)
    }
    corr_accum: dict[str, list[float]] = {s: [] for s in style_names}

    directed_factor = factor_mat * (1.0 if sign >= 0 else -1.0)

    for t in range(T):
        f_row = directed_factor[t, :]
        valid_f = np.isfinite(f_row)
        if mask is not None:
            valid_f = valid_f & mask[t, :]

        valid_idx = np.where(valid_f)[0]
        m = len(valid_idx)
        if m < n_quantiles:
            continue

        # 按因子值升序切分 n_quantiles 组 (G1 最小, G{n_quantiles} 最大即多头组)
        order = valid_idx[np.argsort(f_row[valid_idx], kind="mergesort")]
        for q in range(n_quantiles):
            start_i = int(round(q * m / n_quantiles))
            end_i = int(round((q + 1) * m / n_quantiles))
            g_idx = order[start_i:end_i]
            if len(g_idx) == 0:
                continue
            g_key = f"G{q + 1}"
            for s in style_names:
                s_vals = styles_dict[s][t, g_idx]
                s_valid = s_vals[np.isfinite(s_vals)]
                if len(s_valid) > 0:
                    group_accum[g_key][s].append(float(np.mean(s_valid)))

        # 计算因子与各风格的截面秩相关（支持并列平均秩）
        for s in style_names:
            s_row = styles_dict[s][t, :]
            both = valid_f & np.isfinite(s_row)
            b_idx = np.where(both)[0]
            if len(b_idx) >= 5:
                c = _spearman_corr_1d(f_row[b_idx], s_row[b_idx])
                if np.isfinite(c):
                    corr_accum[s].append(c)

    group_exposures: dict[str, dict[str, float]] = {}
    for g_key, s_map in group_accum.items():
        group_exposures[g_key] = {
            s: (float(np.mean(vals)) if len(vals) > 0 else 0.0) for s, vals in s_map.items()
        }

    top_group = f"G{n_quantiles}"
    bot_group = "G1"
    long_exp = {s: group_exposures.get(top_group, {}).get(s, 0.0) for s in style_names}
    short_exp = {s: group_exposures.get(bot_group, {}).get(s, 0.0) for s in style_names}
    ls_exp = {s: long_exp[s] - short_exp[s] for s in style_names}
    rank_corr = {
        s: (float(np.mean(corr_accum[s])) if len(corr_accum[s]) > 0 else 0.0) for s in style_names
    }

    return {
        "styles": style_names,
        "group_exposures": group_exposures,
        "long_exposure": long_exp,
        "short_exposure": short_exp,
        "long_short_exposure": ls_exp,
        "rank_corr": rank_corr,
    }
