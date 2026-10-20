"""测试 ETL 各核心清洗组件数学正确性 (tests/python/test_etl_components.py)。"""

from __future__ import annotations

import datetime as dt
from datetime import date
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from factorlib.core.config import MinuteFeaturesConfig, UniverseConfig, WarehouseConfig
from factorlib.markets.futures.etl.calendar import add_date_idx, dense_grid, lag_by_idx
from factorlib.markets.futures.etl.exec_price import aggregate_exec
from factorlib.markets.futures.etl.labels import build_labels
from factorlib.markets.futures.etl.minute_features import aggregate_features
from factorlib.markets.futures.universe import build_universe


def test_calendar_and_dense_grid():
    cal = pl.DataFrame({
        "trading_date": [date(2024, 1, 1), date(2024, 1, 2), date(2024, 1, 3), date(2024, 1, 4), date(2024, 1, 5)],
        "date_idx": [0, 1, 2, 3, 4],
    }).with_columns(pl.col("date_idx").cast(pl.Int64))

    # 品种 A 缺失 date_idx=2 (2024-01-03)
    df_sparse = pl.DataFrame({
        "symbol": ["A", "A", "A"],
        "trading_date": [date(2024, 1, 1), date(2024, 1, 2), date(2024, 1, 4)],
        "date_idx": [0, 1, 3],
        "val": [10.0, 20.0, 40.0],
    }).with_columns(pl.col("date_idx").cast(pl.Int64))

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
    df = pl.DataFrame({
        "order_book_id": ["RB2405", "RB2405"],
        "trading_date": [date(2024, 1, 3), date(2024, 1, 3)],
        "datetime": [d0, d1],
        "symbol": ["RB", "RB"],
        "close": [3800.0, 3810.0],
        "volume": [100.0, 200.0],
    })

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

    df = pl.DataFrame({
        "order_book_id": ["CU2405"] * 5,
        "trading_date": [date(2024, 3, 4)] * 5,
        "datetime": times,
        "open": closes,
        "high": highs,
        "low": lows,
        "close": closes,
        "volume": volumes,
        "prev_settlement": [100.0] * 5,
    })

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
    cal = pl.DataFrame({
        "trading_date": [date(2024, 1, i) for i in range(1, 25)],
        "date_idx": list(range(24)),
    }).with_columns(pl.col("date_idx").cast(pl.Int64))

    first_listed = pl.DataFrame({
        "symbol": ["A"],
        "first_listed_date": [date(2023, 1, 1)],  # 上市超过 1 年
    })

    # 构造 24 个交易日数据
    # 前 19 天 turnover = 1000 万, 从第 20 天起 turnover = 6000 万 (使 ADV20 逐步爬升)
    daily = pl.DataFrame({
        "symbol": ["A"] * 24,
        "trading_date": [date(2024, 1, i) for i in range(1, 25)],
        "date_idx": list(range(24)),
        "turnover": [10_000_000.0] * 19 + [60_000_000.0] * 5,
        "traded": [True] * 24,
        "close_raw": [100.0] * 24,
        "maturity_date": [date(2024, 2, 1)] * 24,  # 距到期日充分大
    }).with_columns(pl.col("date_idx").cast(pl.Int64))

    ucfg = UniverseConfig(min_adv20_turnover=50_000_000.0, min_listed_days=10, min_days_to_expiry=5)
    univ = build_universe(daily, first_listed, cal, ucfg)

    # 前 19 天窗口未达 20 天或 ADV20 不足 5000 万, 均为 False
    assert not univ["in_universe"][:19].any()


def test_labels_no_lookahead():
    cal = pl.DataFrame({
        "trading_date": [date(2024, 1, i) for i in range(1, 10)],
        "date_idx": list(range(9)),
    }).with_columns(pl.col("date_idx").cast(pl.Int64))

    # 构造价格序列: 日期 1 到 9
    # T=0 (2024-01-01): exec_vwap=100.0
    # T=1 (2024-01-02): exec_vwap=102.0, exec_vwap_old=102.0  (r_{0->1} = 2%)
    # T=2 (2024-01-03): exec_vwap=105.06, exec_vwap_old=105.06 (r_{1->2} = 3%)
    # T=3 (2024-01-04): exec_vwap=110.313, exec_vwap_old=110.313 (r_{2->3} = 5%)
    panel = pl.DataFrame({
        "symbol": ["RB"] * 4,
        "trading_date": [date(2024, 1, 1), date(2024, 1, 2), date(2024, 1, 3), date(2024, 1, 4)],
        "date_idx": [0, 1, 2, 3],
        "exec_vwap": [100.0, 102.0, 105.06, 110.313],
        "exec_vwap_old": [100.0, 102.0, 105.06, 110.313],
    }).with_columns(pl.col("date_idx").cast(pl.Int64))

    labels = build_labels(panel, cal, horizons=[1, 2])

    # 验证严格防未来函数设计:
    # 对于 T=0 (2024-01-01):
    # 信号在 T=0 收盘后生成, 建仓在 T=1 开盘 VWAP (102.0)
    # y_1(T=0) 为持有 1 天从 T=1 到 T=2 的收益: 105.06 / 102.0 - 1 = 3.0% (而非泄露从 T=0 到 T=1 的 2%!)
    row0 = labels.filter(pl.col("trading_date") == date(2024, 1, 1))
    assert abs(row0["y_1"][0] - 0.03) < 1e-5

    # y_2(T=0) 为持有 2 天从 T=1 到 T=3 的累计复利: 110.313 / 102.0 - 1 = (1 + 0.03) * (1 + 0.05) - 1 = 8.15%
    assert abs(row0["y_2"][0] - 0.0815) < 1e-5
