"""合成研究面板生成器 (core/synthetic.py)。

提供期货与 A 股两大研究域的内置合成演示面板（均采用 Fortran 列优先存储、严格因果标签与独立 aux_fields），
用于在尚未构建本地派生面板时保证工作台、回测与测试套件开箱即用。
"""

from __future__ import annotations

import datetime as dt

import numpy as np

from factorlib.core.panel import Panel


def create_synthetic_futures_panel(n_days: int = 120, seed: int = 42) -> Panel:
    """构建具备完整 33 个标准字段、多板块分布与真实因子信噪比的合成期货研究面板。"""
    rng = np.random.default_rng(seed)
    symbols = ["RB", "I", "CU", "AL", "TA", "MA", "M", "Y", "IF", "AU"]
    sector_map = [1.0, 1.0, 2.0, 2.0, 3.0, 3.0, 4.0, 4.0, 5.0, 6.0]
    N = len(symbols)
    T = max(40, int(n_days))

    start_dt = dt.date(2023, 1, 3)
    dates: list[str] = []
    curr_dt = start_dt
    while len(dates) < T:
        if curr_dt.weekday() < 5:
            dates.append(curr_dt.isoformat())
        curr_dt += dt.timedelta(days=1)

    ry_ann = np.zeros((T, N), dtype=np.float64, order="F")
    wh_z60 = np.zeros((T, N), dtype=np.float64, order="F")
    mr_net_ratio = np.zeros((T, N), dtype=np.float64, order="F")
    rv = np.zeros((T, N), dtype=np.float64, order="F")

    ry_ann[0, :] = rng.normal(0.0, 0.06, N)
    wh_z60[0, :] = rng.normal(0.0, 0.8, N)
    mr_net_ratio[0, :] = rng.normal(0.0, 0.12, N)
    rv[0, :] = rng.uniform(0.010, 0.025, N)

    for t in range(1, T):
        ry_ann[t, :] = 0.92 * ry_ann[t - 1, :] + rng.normal(0.0, 0.02, N)
        wh_z60[t, :] = 0.90 * wh_z60[t - 1, :] + rng.normal(0.0, 0.25, N)
        mr_net_ratio[t, :] = 0.85 * mr_net_ratio[t - 1, :] + rng.normal(0.0, 0.05, N)
        rv[t, :] = np.clip(0.80 * rv[t - 1, :] + rng.uniform(0.002, 0.006, N), 0.006, 0.045)

    base_prices = np.array(
        [3800.0, 820.0, 68000.0, 18500.0, 5800.0, 2500.0, 3900.0, 8200.0, 4000.0, 450.0]
    )
    close = np.zeros((T, N), dtype=np.float64, order="F")
    close[0, :] = base_prices

    for t in range(1, T):
        alpha_signal = (
            0.03 * ry_ann[t - 1, :] - 0.0012 * wh_z60[t - 1, :] + 0.008 * mr_net_ratio[t - 1, :]
        )
        noise = rng.normal(0.0002, 0.013, N)
        ret_t = np.clip(alpha_signal + noise, -0.07, 0.07)
        close[t, :] = close[t - 1, :] * (1.0 + ret_t)

    open_p = np.asfortranarray(close * (1.0 + rng.normal(0.0, 0.002, (T, N))))
    high_p = np.asfortranarray(
        np.maximum(open_p, close) * (1.0 + np.abs(rng.normal(0.004, 0.003, (T, N))))
    )
    low_p = np.asfortranarray(
        np.minimum(open_p, close) * (1.0 - np.abs(rng.normal(0.004, 0.003, (T, N))))
    )
    settle_p = np.asfortranarray(0.5 * (open_p + close))
    vwap_p = np.asfortranarray(0.3 * open_p + 0.3 * high_p + 0.4 * close)
    exec_vwap = np.asfortranarray(open_p * (1.0 + rng.normal(0.0, 0.0008, (T, N))))

    volume = np.asfortranarray(rng.uniform(20000.0, 180000.0, (T, N)))
    multipliers = np.array([10.0, 100.0, 5.0, 5.0, 5.0, 10.0, 10.0, 10.0, 300.0, 1000.0])
    turnover = np.asfortranarray(close * volume * multipliers[np.newaxis, :])
    open_interest = np.asfortranarray(volume * rng.uniform(1.8, 3.5, (T, N)))
    oi_chg = np.zeros((T, N), dtype=np.float64, order="F")
    oi_chg[1:, :] = open_interest[1:, :] - open_interest[:-1, :]

    wh_chg5 = np.zeros((T, N), dtype=np.float64, order="F")
    wh_chg5[5:, :] = (wh_z60[5:, :] - wh_z60[:-5, :]) * 0.1
    mr_chg5 = np.zeros((T, N), dtype=np.float64, order="F")
    mr_chg5[5:, :] = mr_net_ratio[5:, :] - mr_net_ratio[:-5, :]

    sector_mat = np.asfortranarray(np.tile(np.array(sector_map, dtype=np.float64), (T, 1)))
    listed_days = np.asfortranarray(
        np.tile(np.arange(120, 120 + T, dtype=np.float64)[:, np.newaxis], (1, N))
    )
    ones_mask = np.ones((T, N), dtype=np.float64, order="F")

    fields: dict[str, np.ndarray] = {
        "open": open_p,
        "high": high_p,
        "low": low_p,
        "close": close,
        "settle": settle_p,
        "vwap": vwap_p,
        "volume": volume,
        "turnover": turnover,
        "open_interest": open_interest,
        "oi_chg": oi_chg,
        "rv": rv,
        "rv_up": np.asfortranarray(rv * rng.uniform(0.45, 0.65, (T, N))),
        "rv_down": np.asfortranarray(rv * rng.uniform(0.35, 0.55, (T, N))),
        "ret_last30": np.asfortranarray(rng.normal(0.0, 0.004, (T, N))),
        "ret_first30": np.asfortranarray(rng.normal(0.0, 0.005, (T, N))),
        "spread_night_day": np.asfortranarray(rng.normal(0.0, 0.006, (T, N))),
        "vol_conc": np.asfortranarray(rng.uniform(0.15, 0.45, (T, N))),
        "amp": np.asfortranarray((high_p - low_p) / np.maximum(close, 1e-6)),
        "skew_intra": np.asfortranarray(rng.normal(0.0, 0.5, (T, N))),
        "kurt_intra": np.asfortranarray(rng.uniform(2.5, 6.0, (T, N))),
        "vwap_dev": np.asfortranarray((close - vwap_p) / np.maximum(vwap_p, 1e-6)),
        "ry_main_sub": np.asfortranarray(ry_ann / 4.0),
        "ry_ann": ry_ann,
        "ry_near_main": np.asfortranarray(ry_ann * 0.5),
        "wh_chg5": wh_chg5,
        "wh_z60": wh_z60,
        "mr_net_ratio": mr_net_ratio,
        "mr_cr20": np.asfortranarray(rng.uniform(0.45, 0.75, (T, N))),
        "mr_net_ratio_chg5": mr_chg5,
        "sector": sector_mat,
        "listed_days": listed_days,
        "in_universe_mask": ones_mask,
        "limit_mask": ones_mask.copy(order="F"),
    }

    labels: dict[str, np.ndarray] = {}
    for h in (1, 3, 5, 10, 20):
        yh = np.full((T, N), np.nan, dtype=np.float64, order="F")
        if T > h + 1:
            yh[: -(h + 1), :] = exec_vwap[1 + h :, :] / exec_vwap[1:-h, :] - 1.0
        labels[f"y_{h}"] = yh
    labels["r_exec_next"] = labels["y_1"].copy(order="F")

    valid_mask = np.ones((T, N), dtype=bool, order="F")
    aux_fields: dict[str, np.ndarray] = {
        "exec_vwap": exec_vwap,
        "exec_vwap_old": exec_vwap.copy(order="F"),
    }

    return Panel(
        dates=np.array(dates),
        symbols=symbols,
        fields=fields,
        mask=valid_mask,
        labels=labels,
        aux_fields=aux_fields,
        meta={"panel_version": "synthetic_v1", "synthetic": True, "market": "futures"},
    )


def create_synthetic_stocks_panel(
    n_days: int = 240,
    n_symbols: int = 40,
    seed: int = 108,
) -> Panel:
    """构建具备完整 A 股日频行情、申万行业编码、复权/原始价与因果标签的合成 A 股研究面板。"""
    rng = np.random.default_rng(seed)
    base_names = [
        "600519.XSHG",
        "000858.XSHE",
        "300750.XSHE",
        "601318.XSHG",
        "600036.XSHG",
        "000333.XSHE",
        "002594.XSHE",
        "600900.XSHG",
        "688981.XSHG",
        "601899.XSHG",
        "600276.XSHG",
        "002415.XSHE",
        "601012.XSHG",
        "000001.XSHE",
        "600030.XSHG",
        "601166.XSHG",
        "002475.XSHE",
        "300059.XSHE",
        "600887.XSHG",
        "601398.XSHG",
    ]
    N = max(12, int(n_symbols))
    symbols = list(base_names[:N])
    while len(symbols) < N:
        idx = len(symbols) + 1
        exch = "XSHG" if idx % 2 == 0 else "XSHE"
        code = f"{600000 + idx:06d}" if exch == "XSHG" else f"{100 + idx:06d}"
        symbols.append(f"{code}.{exch}")

    T = max(60, int(n_days))
    start_dt = dt.date(2023, 1, 3)
    dates: list[str] = []
    curr_dt = start_dt
    while len(dates) < T:
        if curr_dt.weekday() < 5:
            dates.append(curr_dt.isoformat())
        curr_dt += dt.timedelta(days=1)

    # 8 个申万模拟一级行业
    sector_arr = np.array([(i % 8) + 1.0 for i in range(N)], dtype=np.float64)
    sector_mat = np.asfortranarray(np.tile(sector_arr, (T, 1)))

    # 构造带温和反转/低波/换手 alpha 的 A 股价格过程
    mkt_ret = rng.normal(0.0003, 0.011, T)
    close = np.zeros((T, N), dtype=np.float64, order="F")
    close[0, :] = rng.uniform(12.0, 220.0, N)
    turnover_rate = np.zeros((T, N), dtype=np.float64, order="F")
    turnover_rate[0, :] = rng.uniform(0.005, 0.045, N)

    prev_rev = np.zeros(N, dtype=np.float64)
    for t in range(1, T):
        turnover_rate[t, :] = np.clip(
            0.78 * turnover_rate[t - 1, :] + rng.uniform(0.001, 0.012, N),
            0.002,
            0.12,
        )
        # 短期均值回归 + 低换手溢价 + 市场因子
        alpha = -0.08 * prev_rev - 0.05 * (
            turnover_rate[t - 1, :] - np.mean(turnover_rate[t - 1, :])
        )
        idio = rng.normal(0.0, 0.018, N)
        ret_t = np.clip(mkt_ret[t] + alpha + idio, -0.098, 0.098)
        close[t, :] = np.maximum(1.0, close[t - 1, :] * (1.0 + ret_t))
        prev_rev = ret_t

    open_p = np.asfortranarray(close * (1.0 + rng.normal(0.0, 0.004, (T, N))))
    high_p = np.asfortranarray(
        np.maximum(open_p, close) * (1.0 + np.abs(rng.normal(0.008, 0.005, (T, N))))
    )
    low_p = np.asfortranarray(
        np.minimum(open_p, close) * (1.0 - np.abs(rng.normal(0.008, 0.005, (T, N))))
    )
    vwap_p = np.asfortranarray(0.25 * open_p + 0.25 * high_p + 0.25 * low_p + 0.25 * close)

    c2c_ret = np.zeros((T, N), dtype=np.float64, order="F")
    c2c_ret[1:, :] = close[1:, :] / np.maximum(close[:-1, :], 1e-6) - 1.0

    mkt_cap_base = rng.uniform(5e9, 4e11, N)
    mkt_cap = np.asfortranarray((close / close[0:1, :]) * mkt_cap_base[np.newaxis, :])
    amount = np.asfortranarray(np.maximum(1e6, mkt_cap * turnover_rate * 0.35))
    volume = np.asfortranarray(amount / np.maximum(vwap_p, 1e-3))
    log_amount = np.asfortranarray(np.log(np.maximum(amount, 1.0)))

    pe = np.asfortranarray(rng.uniform(8.0, 65.0, (T, N)) * (close / close[0:1, :]))
    pb = np.asfortranarray(rng.uniform(0.9, 8.5, (T, N)) * (close / close[0:1, :]))

    listed_days = np.asfortranarray(
        np.tile(np.arange(180, 180 + T, dtype=np.float64)[:, np.newaxis], (1, N))
    )
    ones_mask = np.ones((T, N), dtype=np.float64, order="F")
    zeros_mask = np.zeros((T, N), dtype=np.float64, order="F")

    # 模拟少量停牌/涨跌停（~2%）
    limit_mask = ones_mask.copy(order="F")
    can_buy = ones_mask.copy(order="F")
    can_sell = ones_mask.copy(order="F")
    is_susp = zeros_mask.copy(order="F")
    susp_draw = rng.random((T, N)) < 0.015
    limit_mask[susp_draw] = np.nan
    can_buy[susp_draw] = 0.0
    can_sell[susp_draw] = 0.0
    is_susp[susp_draw] = 1.0

    fields: dict[str, np.ndarray] = {
        "open": open_p,
        "high": high_p,
        "low": low_p,
        "close": close,
        "close_hfq": close.copy(order="F"),
        "open_adj": open_p.copy(order="F"),
        "high_adj": high_p.copy(order="F"),
        "low_adj": low_p.copy(order="F"),
        "vwap_adj": vwap_p,
        "volume": volume,
        "amount": amount,
        "turnover": amount,
        "turnover_rate": turnover_rate,
        "log_amount": log_amount,
        "c2c_ret": c2c_ret,
        "mkt_cap": mkt_cap,
        "pe": pe,
        "pb": pb,
        "sector": sector_mat,
        "industry_code": sector_mat.copy(order="F"),
        "listed_days": listed_days,
        "in_universe_mask": ones_mask,
        "limit_mask": limit_mask,
    }

    labels: dict[str, np.ndarray] = {}
    for h in (1, 2, 5, 10):
        yh = np.full((T, N), np.nan, dtype=np.float64, order="F")
        if T > h + 1:
            yh[: -(h + 1), :] = open_p[1 + h :, :] / np.maximum(open_p[1:-h, :], 1e-6) - 1.0
        labels[f"y_{h}"] = yh
    labels["r_exec_next"] = labels["y_1"].copy(order="F")

    bench_ret = np.asfortranarray(np.tile(mkt_ret[:, np.newaxis], (1, N)))
    limit_up = np.asfortranarray(close * 1.10)
    limit_down = np.asfortranarray(close * 0.90)

    aux_fields: dict[str, np.ndarray] = {
        "open_raw": open_p.copy(order="F"),
        "close_raw": close.copy(order="F"),
        "high_raw": high_p.copy(order="F"),
        "low_raw": low_p.copy(order="F"),
        "ex_cum_factor": ones_mask.copy(order="F"),
        "limit_up": limit_up,
        "limit_down": limit_down,
        "is_suspended": is_susp,
        "is_st": zeros_mask.copy(order="F"),
        "can_buy_mask": can_buy,
        "can_sell_mask": can_sell,
        "bench_ret": bench_ret,
    }

    return Panel(
        dates=np.array(dates),
        symbols=symbols,
        fields=fields,
        mask=np.ones((T, N), dtype=bool, order="F"),
        labels=labels,
        aux_fields=aux_fields,
        meta={"panel_version": "synthetic_stocks_v1", "synthetic": True, "market": "stocks"},
    )
