"""测试 ETL 各核心清洗组件数学正确性 (tests/python/test_etl_components.py)。"""

from __future__ import annotations

import datetime as dt
from datetime import date

import numpy as np
import polars as pl
from factorlib.core.config import MinuteFeaturesConfig, UniverseConfig
from factorlib.markets.futures.etl.calendar import dense_grid, lag_by_idx
from factorlib.markets.futures.etl.exec_price import aggregate_exec
from factorlib.markets.futures.etl.labels import build_labels
from factorlib.markets.futures.etl.minute_features import aggregate_features
from factorlib.markets.futures.universe import build_universe


def test_calendar_and_dense_grid():
    cal = pl.DataFrame(
        {
            "trading_date": [
                date(2024, 1, 1),
                date(2024, 1, 2),
                date(2024, 1, 3),
                date(2024, 1, 4),
                date(2024, 1, 5),
            ],
            "date_idx": [0, 1, 2, 3, 4],
        }
    ).with_columns(pl.col("date_idx").cast(pl.Int64))

    # 品种 A 缺失 date_idx=2 (2024-01-03)
    df_sparse = pl.DataFrame(
        {
            "symbol": ["A", "A", "A"],
            "trading_date": [date(2024, 1, 1), date(2024, 1, 2), date(2024, 1, 4)],
            "date_idx": [0, 1, 3],
            "val": [10.0, 20.0, 40.0],
        }
    ).with_columns(pl.col("date_idx").cast(pl.Int64))

    # 展开稠密网格
    grid = dense_grid(df_sparse.select(["symbol", "date_idx"]), cal, key="symbol")
    assert len(grid) == 4  # 0, 1, 2, 3
    assert grid["date_idx"].to_list() == [0, 1, 2, 3]

    # 按 date_idx 滞后测试
    merged = grid.join(df_sparse, on=["symbol", "date_idx", "trading_date"], how="left")
    lagged = lag_by_idx(merged, ["val"], n=1, key="symbol")

    # date_idx=3 的滞后1位是 date_idx=2 (在 sparse 中缺失为 None), 绝非错误的取到 date_idx=1 的 20.0
    val_lag1 = lagged.filter(pl.col("date_idx") == 3)["val_lag1"][0]
    assert val_lag1 is None

    # date_idx=1 的滞后1位是 date_idx=0 (10.0)
    assert lagged.filter(pl.col("date_idx") == 1)["val_lag1"][0] == 10.0


def test_exec_price_aggregate():
    # 模拟两根分钟 bar
    # 郑商所历史异常场景: volume > 0 但 total_turnover == 0
    # 验证 VWAP 严格按 sum(close * volume) / sum(volume) 计算
    d0 = dt.datetime(2024, 1, 2, 21, 1)
    d1 = dt.datetime(2024, 1, 2, 21, 2)
    df = pl.DataFrame(
        {
            "order_book_id": ["RB2405", "RB2405"],
            "trading_date": [date(2024, 1, 3), date(2024, 1, 3)],
            "datetime": [d0, d1],
            "symbol": ["RB", "RB"],
            "close": [3800.0, 3810.0],
            "volume": [100.0, 200.0],
        }
    )

    res = aggregate_exec(df, window_minutes=30)
    assert len(res) == 1
    # VWAP = (3800 * 100 + 3810 * 200) / 300 = 3806.6666666666665
    expected_vwap = (3800.0 * 100 + 3810.0 * 200) / 300.0
    assert abs(res["exec_vwap"][0] - expected_vwap) < 1e-6
    assert res["exec_volume"][0] == 300.0
    assert res["exec_nbars"][0] == 2


def test_minute_features_math():
    # 构造确定性的 5 根分钟 Bar
    # 价格序列: 100.0, 102.0, 101.0, 103.0, 105.0
    # 成交量: 10, 20, 30, 40, 50 (总成交量 150)
    # 前结算价: 100.0
    t_start = dt.datetime(2024, 3, 4, 9, 31)
    times = [t_start + dt.timedelta(minutes=i) for i in range(5)]
    closes = [100.0, 102.0, 101.0, 103.0, 105.0]
    highs = [100.5, 102.5, 101.5, 103.5, 105.5]
    lows = [99.5, 101.5, 100.5, 102.0, 104.5]
    volumes = [10.0, 20.0, 30.0, 40.0, 50.0]

    df = pl.DataFrame(
        {
            "order_book_id": ["CU2405"] * 5,
            "trading_date": [date(2024, 3, 4)] * 5,
            "datetime": times,
            "open": closes,
            "high": highs,
            "low": lows,
            "close": closes,
            "volume": volumes,
            "prev_settlement": [100.0] * 5,
        }
    )

    mcfg = MinuteFeaturesConfig()
    res = aggregate_features(df, mcfg)

    # 1. 验证已实现波动率 (RV)
    r = np.diff(np.log(closes))
    expected_rv = np.sqrt(np.sum(r**2))
    assert abs(res["rv"][0] - expected_rv) < 1e-6

    # 2. 验证振幅 (amp)
    expected_amp = (max(highs) - min(lows)) / 100.0
    assert abs(res["amp"][0] - expected_amp) < 1e-6

    # 3. 验证成交量集中度 (vol_conc = sum((v/sum(v))^2))
    w = np.array(volumes) / sum(volumes)
    expected_conc = np.sum(w**2)
    assert abs(res["vol_conc"][0] - expected_conc) < 1e-6

    # 4. 验证 VWAP 偏离度
    vwap_day = np.sum(np.array(closes) * np.array(volumes)) / np.sum(volumes)
    expected_dev = (closes[-1] - vwap_day) / vwap_day
    assert abs(res["vwap_dev"][0] - expected_dev) < 1e-6


def test_universe_filtering():
    cal = pl.DataFrame(
        {
            "trading_date": [date(2024, 1, i) for i in range(1, 25)],
            "date_idx": list(range(24)),
        }
    ).with_columns(pl.col("date_idx").cast(pl.Int64))

    first_listed = pl.DataFrame(
        {
            "symbol": ["A"],
            "first_listed_date": [date(2023, 1, 1)],  # 上市超过 1 年
        }
    )

    # 构造 24 个交易日数据
    # 前 19 天 turnover = 1000 万, 从第 20 天起 turnover = 6000 万 (使 ADV20 逐步爬升)
    daily = pl.DataFrame(
        {
            "symbol": ["A"] * 24,
            "trading_date": [date(2024, 1, i) for i in range(1, 25)],
            "date_idx": list(range(24)),
            "turnover": [10_000_000.0] * 19 + [60_000_000.0] * 5,
            "traded": [True] * 24,
            "close_raw": [100.0] * 24,
            "maturity_date": [date(2024, 2, 1)] * 24,  # 距到期日充分大
        }
    ).with_columns(pl.col("date_idx").cast(pl.Int64))

    ucfg = UniverseConfig(min_adv20_turnover=50_000_000.0, min_listed_days=10, min_days_to_expiry=5)
    univ = build_universe(daily, first_listed, cal, ucfg)

    # 前 19 天窗口未达 20 天或 ADV20 不足 5000 万, 均为 False
    assert not univ["in_universe"][:19].any()


def test_labels_no_lookahead():
    cal = pl.DataFrame(
        {
            "trading_date": [date(2024, 1, i) for i in range(1, 10)],
            "date_idx": list(range(9)),
        }
    ).with_columns(pl.col("date_idx").cast(pl.Int64))

    # 构造价格序列: 日期 1 到 9
    # T=0 (2024-01-01): exec_vwap=100.0
    # T=1 (2024-01-02): exec_vwap=102.0, exec_vwap_old=102.0  (r_{0->1} = 2%)
    # T=2 (2024-01-03): exec_vwap=105.06, exec_vwap_old=105.06 (r_{1->2} = 3%)
    # T=3 (2024-01-04): exec_vwap=110.313, exec_vwap_old=110.313 (r_{2->3} = 5%)
    panel = pl.DataFrame(
        {
            "symbol": ["RB"] * 4,
            "trading_date": [
                date(2024, 1, 1),
                date(2024, 1, 2),
                date(2024, 1, 3),
                date(2024, 1, 4),
            ],
            "date_idx": [0, 1, 2, 3],
            "exec_vwap": [100.0, 102.0, 105.06, 110.313],
            "exec_vwap_old": [100.0, 102.0, 105.06, 110.313],
        }
    ).with_columns(pl.col("date_idx").cast(pl.Int64))

    labels = build_labels(panel, cal, horizons=[1, 2])

    # 验证严格防未来函数设计:
    # 对于 T=0 (2024-01-01):
    # 信号在 T=0 收盘后生成, 建仓在 T=1 开盘 VWAP (102.0)
    # y_1(T=0) 为持有 1 天从 T=1 到 T=2 的收益: 105.06 / 102.0 - 1 = 3.0% (而非泄露从 T=0 到 T=1 的 2%!)
    row0 = labels.filter(pl.col("trading_date") == date(2024, 1, 1))
    assert abs(row0["y_1"][0] - 0.03) < 1e-5
    # P0-01: r_exec_next 必须同步平移至 T+1->T+2, 与 y_1 严格等价 (3.0%), 严禁泄漏 T=0 日内收益 (2.0%)
    assert abs(row0["r_exec_next"][0] - 0.03) < 1e-5
    assert row0["r_exec_next"][0] == row0["y_1"][0]

    # y_2(T=0) 为持有 2 天从 T=1 到 T=3 的累计复利: 110.313 / 102.0 - 1 = (1 + 0.03) * (1 + 0.05) - 1 = 8.15%
    assert abs(row0["y_2"][0] - 0.0815) < 1e-5


def test_intraday_factor_no_lookahead_correlation():
    """P0-01 回归测试: 验证日内因子 (close/open - 1) 与 y_1 及平移后的 r_exec_next 无前视伪高相关。

    若标签误用当期未平移早盘收益 r_{T->T+1}, 因其区间包含了 T 日日内走势,
    会导致纯随机行情的日内涨跌幅对标签产生高达 +0.8 左右的虚假 RankIC。
    修复后 (T+1->T+2), 因果严格隔离, 截面平均 RankIC 应当收敛于 0。
    """
    rng = np.random.default_rng(42)
    n_days = 80
    n_symbols = 25
    symbols = [f"S{i:02d}" for i in range(n_symbols)]
    dates = [date(2024, 1, 1) + dt.timedelta(days=i) for i in range(n_days)]

    cal = pl.DataFrame(
        {
            "trading_date": dates,
            "date_idx": list(range(n_days)),
        }
    ).with_columns(pl.col("date_idx").cast(pl.Int64))

    # 生成随机游走价格序列
    # 日内收益率与隔夜收益率均为独立白噪声
    rows = []
    for s in symbols:
        price = 100.0
        for d_idx, d in enumerate(dates):
            intraday_ret = rng.normal(0.0, 0.02)
            overnight_ret = rng.normal(0.0, 0.01)
            open_p = price * (1.0 + overnight_ret)
            close_p = open_p * (1.0 + intraday_ret)
            exec_vwap = open_p  # 早盘开盘 VWAP 约等于 open
            rows.append(
                {
                    "symbol": s,
                    "trading_date": d,
                    "date_idx": d_idx,
                    "open": open_p,
                    "close": close_p,
                    "exec_vwap": exec_vwap,
                    "exec_vwap_old": exec_vwap,
                }
            )
            price = close_p

    df_panel = pl.DataFrame(rows).with_columns(pl.col("date_idx").cast(pl.Int64))
    labels = build_labels(df_panel, cal, horizons=[1])

    # 合并因子与标签
    merged = df_panel.join(labels, on=["trading_date", "symbol"], how="inner").with_columns(
        factor=pl.col("close") / pl.col("open") - 1.0,
    )

    # 逐日计算 factor 与 y_1 及 r_exec_next 的截面 RankIC
    def spearman_rank_corr(x: np.ndarray, y: np.ndarray) -> float:
        valid = np.isfinite(x) & np.isfinite(y)
        if valid.sum() < 5:
            return np.nan
        rx = np.argsort(np.argsort(x[valid]))
        ry = np.argsort(np.argsort(y[valid]))
        c = np.corrcoef(rx, ry)[0, 1]
        return float(c)

    rank_ics_y1 = []
    rank_ics_rexec = []
    for d in dates[:-2]:  # 排除最后无标签的两天
        sub = merged.filter(pl.col("trading_date") == d)
        f_vals = sub["factor"].to_numpy()
        y1_vals = sub["y_1"].to_numpy()
        r_vals = sub["r_exec_next"].to_numpy()

        ic_y1 = spearman_rank_corr(f_vals, y1_vals)
        ic_r = spearman_rank_corr(f_vals, r_vals)
        if np.isfinite(ic_y1):
            rank_ics_y1.append(ic_y1)
        if np.isfinite(ic_r):
            rank_ics_rexec.append(ic_r)

    mean_ic_y1 = float(np.mean(rank_ics_y1))
    mean_ic_r = float(np.mean(rank_ics_rexec))

    # 断言 1: y_1 与 r_exec_next 的 RankIC 必须完全一致
    assert abs(mean_ic_y1 - mean_ic_r) < 1e-12

    # 断言 2: 无前视偏差时，独立白噪声行情的日内涨跌幅对未来收益无预测力 (|IC| < 0.1)
    # 若存在前视偏差 (未平移)，此处 mean_ic 会高达 +0.7 ~ +0.8
    assert abs(mean_ic_y1) < 0.1, f"检测到前视偏差泄漏! mean_ic_y1 = {mean_ic_y1:.4f}"


def test_roll_exec_vwap_old_fallback_cascade():
    """P0-02 回归测试: 验证换月日旧合约平仓执行价的三级严格同合约级联回退机制。

    1) 优先使用旧合约早盘 30m VWAP (_old_exec_vwap);
    2) 若缺失则回退至旧合约当日日线开盘价 (_old_bar_open);
    3) 若全天零成交则回退至旧合约昨结 (_old_bar_prev_settle / shift(1) 昨结);
    4) 严禁回退至新合约当天的任何价格 (如当日新合约 settle_raw)。
    """
    # 构造 4 个换月日样本场景
    # 日期: T=0 (正常日), T=1 (L1 命中), T=2 (L2 命中), T=3 (L3a 命中), T=4 (L3b 命中)
    dates = [date(2024, 1, i) for i in range(1, 6)]
    daily = pl.DataFrame(
        {
            "trading_date": dates,
            "symbol": ["RB"] * 5,
            "contract": ["RB2401", "RB2405", "RB2410", "RB2501", "RB2505"],
            "prev_contract": ["RB2401", "RB2401", "RB2405", "RB2410", "RB2501"],
            "is_roll": [False, True, True, True, True],
            "exec_vwap": [100.0, 200.0, 300.0, 400.0, 500.0],  # 新合约早盘开盘价
            "settle_raw": [102.0, 205.0, 305.0, 405.0, 505.0],  # 新合约当日收盘结算价
        }
    )

    # 1. 模拟早盘分钟线提取结果: 仅 T=1 提取到了旧合约的早盘 30m VWAP
    rolls_exec = pl.DataFrame(
        {
            "prev_contract": ["RB2401"],
            "trading_date": [date(2024, 1, 2)],
            "_old_exec_vwap": [103.5],
        }
    )
    daily = daily.join(rolls_exec, on=["prev_contract", "trading_date"], how="left")

    # 2. 模拟日线 bar_1d 提取结果:
    # T=2 旧合约早盘无成交但全天有成交，开盘价 206.0，昨结 205.0
    # T=3 旧合约全天零成交 (open=0)，但有昨结 304.0
    # T=4 旧合约在 bar_1d 中完全无记录 (如已摘牌)
    old_bars = pl.DataFrame(
        {
            "prev_contract": ["RB2405", "RB2410"],
            "trading_date": [date(2024, 1, 3), date(2024, 1, 4)],
            "_old_bar_open": [206.0, None],
            "_old_bar_prev_settle": [205.0, 304.0],
        }
    )
    daily = daily.join(old_bars, on=["prev_contract", "trading_date"], how="left")

    # 预排序确保 shift(1) 正确
    daily = daily.sort(["symbol", "trading_date"])

    # 执行三级严格级联回退
    daily = daily.with_columns(
        exec_vwap_old=pl.when(pl.col("is_roll"))
        .then(
            pl.col("_old_exec_vwap")
            .fill_null(pl.col("_old_bar_open"))
            .fill_null(pl.col("_old_bar_prev_settle"))
            .fill_null(pl.col("settle_raw").shift(1).over("symbol"))
        )
        .otherwise(pl.col("exec_vwap"))
    ).drop(["_old_exec_vwap", "_old_bar_open", "_old_bar_prev_settle"])

    res = daily.sort("trading_date")

    # 逐日断言:
    # T=0: 非换月日，exec_vwap_old == exec_vwap == 100.0
    assert res["exec_vwap_old"][0] == 100.0

    # T=1: 换月日，L1 命中: 取旧合约早盘 30m VWAP (103.5)
    assert res["exec_vwap_old"][1] == 103.5

    # T=2: 换月日，L2 命中: 取旧合约日线真实开盘价 (206.0)，绝非新合约结算价 305.0
    assert res["exec_vwap_old"][2] == 206.0

    # T=3: 换月日，L3a 命中: 取旧合约日线 prev_settlement (304.0)，绝非新合约结算价 405.0
    assert res["exec_vwap_old"][3] == 304.0

    # T=4: 换月日，L3b 命中: bar_1d 缺失，回退至上一日旧合约结算价 (405.0)，绝非当日新合约结算价 505.0
    assert res["exec_vwap_old"][4] == 405.0

    # 关键因果红线校验: 在所有换月日 (T=1..4)，exec_vwap_old 严禁等于当日新合约 settle_raw
    for i in range(1, 5):
        assert res["exec_vwap_old"][i] != res["settle_raw"][i], (
            f"Row {i} 错误使用了新合约当日收盘结算价未来信息: "
            f"exec_vwap_old={res['exec_vwap_old'][i]}, settle_raw={res['settle_raw'][i]}"
        )


def test_minute_features_physical_30m_window_and_open_denominator():
    """验证 P1-06: 稀疏分钟线严格按物理 30 分钟时间窗及首根 open 计算 ret_first30 与 ret_last30。"""
    times = [
        dt.datetime(2024, 3, 4, 9, 1),
        dt.datetime(2024, 3, 4, 9, 20),
        dt.datetime(2024, 3, 4, 11, 0),
        dt.datetime(2024, 3, 4, 14, 35),
        dt.datetime(2024, 3, 4, 15, 0),
    ]
    df = pl.DataFrame(
        {
            "order_book_id": ["SI2405"] * 5,
            "trading_date": [date(2024, 3, 4)] * 5,
            "datetime": times,
            "open": [100.0, 101.0, 105.0, 120.0, 118.0],
            "high": [102.0, 106.0, 121.0, 121.0, 127.0],
            "low": [99.0, 100.5, 104.0, 117.0, 117.5],
            "close": [101.0, 105.0, 120.0, 118.0, 126.0],
            "volume": [10.0, 20.0, 30.0, 40.0, 50.0],
            "prev_settlement": [100.0] * 5,
        }
    )

    res = aggregate_features(df, MinuteFeaturesConfig())
    # [09:01, 09:31) 包含前两根 (open[0]=100.0, close[1]=105.0) -> +5%
    assert abs(res["ret_first30"][0] - 0.05) < 1e-12
    # (14:30, 15:00] 包含后两根 (open[3]=120.0, close[4]=126.0) -> +5%
    assert abs(res["ret_last30"][0] - 0.05) < 1e-12


def test_alt_data_lag_days_and_deterministic_dedup(tmp_path):
    """验证 P1-06: 盘后另类数据按日期排序后 keep='last' 确定性去重及 alt_data_lag_days=1 滞后开关。"""
    from factorlib.core.config import DataConfig
    from factorlib.markets.futures.etl.aux_warehouse import build_warehouse_features

    cal = pl.DataFrame(
        {
            "trading_date": [date(2024, 1, 2), date(2024, 1, 3), date(2024, 1, 4)],
            "date_idx": [0, 1, 2],
        }
    ).with_columns(pl.col("date_idx").cast(pl.Int64))
    daily = cal.with_columns(pl.lit("CU").alias("symbol"))

    # 故意乱序写入重复行 (100.0 与 200.0)，排序后 keep='last' 应确定性保留较大的 200.0
    wh_dir = tmp_path / "warehouse_stocks"
    wh_dir.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(
        {
            "trading_date": [
                date(2024, 1, 2),
                date(2024, 1, 2),
                date(2024, 1, 3),
                date(2024, 1, 4),
            ],
            "underlying_symbol": ["CU", "CU", "CU", "CU"],
            "on_warrant": [200.0, 100.0, 300.0, 400.0],
        }
    ).write_parquet(wh_dir / "warehouse_stocks.parquet")

    cfg_lag0 = DataConfig(
        raw_dir=tmp_path,
        derived_dir=tmp_path / "derived",
        start_date=date(2024, 1, 1),
        end_date=date(2024, 1, 10),
        alt_data_lag_days=0,
    )
    # 设 chg_lag=1 便于短样本测试
    from dataclasses import replace

    from factorlib.core.config import WarehouseConfig

    wcfg = WarehouseConfig(chg_lag=1, z_window=2, z_min_samples=2, eps=0.0)
    cfg0 = replace(cfg_lag0, warehouse=wcfg)
    cfg1 = replace(cfg0, alt_data_lag_days=1)

    res0 = build_warehouse_features(tmp_path, cfg0, daily, cal, date(2024, 1, 10))
    res1 = build_warehouse_features(tmp_path, cfg1, daily, cal, date(2024, 1, 10))

    # 1月3日 stock=300 vs 1月2日去重保留 stock=200 => wh_chg5[1] == (300 - 200) / 200 = 0.5
    assert abs(res0["wh_chg5"][1] - 0.5) < 1e-12
    # alt_data_lag_days=1 时，1月3日特征应前推至 1月4日，且 1月3日自身为空
    assert res1["wh_chg5"][1] is None
    assert abs(res1["wh_chg5"][2] - 0.5) < 1e-12
