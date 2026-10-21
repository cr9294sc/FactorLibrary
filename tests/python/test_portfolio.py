"""测试期货实盘组合构建与权重映射 (tests/python/test_portfolio.py)。"""

from __future__ import annotations

import numpy as np
import pytest
from factorlib.core.config import PortfolioConfig
from factorlib.core.portfolio import (
    build_portfolio_weights,
    compute_daily_returns_from_prices,
)


def test_compute_daily_returns():
    # 构造简单价格矩阵 (3, 2)
    prices = np.array(
        [
            [100.0, 50.0],
            [105.0, 48.0],
            [110.0, 50.4],
        ],
        dtype=np.float64,
    )
    rets = compute_daily_returns_from_prices(prices)
    assert rets.shape == (3, 2)
    assert rets[0, 0] == 0.0 and rets[0, 1] == 0.0
    assert pytest.approx(rets[1, 0], 1e-6) == 0.05
    assert pytest.approx(rets[1, 1], 1e-6) == -0.04
    assert pytest.approx(rets[2, 0], 1e-6) == (110.0 - 105.0) / 105.0
    assert pytest.approx(rets[2, 1], 1e-6) == (50.4 - 48.0) / 48.0


def test_build_portfolio_weights_basic():
    T = 100
    N = 10
    np.random.seed(42)

    # 模拟收益率与评分
    returns = np.random.normal(0.0005, 0.02, size=(T, N))
    scores = np.random.normal(0.0, 1.0, size=(T, N))
    mask = np.ones((T, N), dtype=bool)
    # 让某个品种在部分交易日离开池子
    mask[20:30, 0] = False

    cfg = PortfolioConfig(
        mode="cs",
        target_volatility=0.10,
        vol_lookback=20,
        cov_shrink_lookback=40,
        max_single_weight=0.15,
        max_gross_leverage=2.0,
        rebalance_buffer=0.01,
    )

    weights = build_portfolio_weights(scores, returns, mask, cfg)

    assert weights.shape == (T, N)
    assert weights.flags.f_contiguous

    # 1. 验证离开池子的品种权重必须为 0
    assert np.all(weights[20:30, 0] == 0.0)

    # 2. 验证单品种绝对权重上限约束
    assert np.all(np.abs(weights) <= cfg.max_single_weight + 1e-7)

    # 3. 验证总杠杆率约束
    gross_lev = np.sum(np.abs(weights), axis=1)
    assert np.all(gross_lev <= cfg.max_gross_leverage + 1e-7)


def test_build_portfolio_weights_turnover_buffer():
    T = 20
    N = 4
    np.random.seed(42)
    returns = np.random.normal(0.0005, 0.015, size=(T, N))
    scores = np.random.normal(0.0, 1.0, size=(T, N))
    # 让第 15 天得分与第 14 天仅差 0.001
    scores[15, :] = scores[14, :] + 0.001
    # 让第 16 天得分与第 15 天出现巨大反转
    scores[16, :] = -scores[15, :]

    mask = np.ones((T, N), dtype=bool)

    cfg = PortfolioConfig(
        mode="cs",
        target_volatility=0.10,
        vol_lookback=10,
        cov_shrink_lookback=10,
        max_single_weight=0.30,
        max_gross_leverage=2.0,
        rebalance_buffer=0.05,
    )

    weights = build_portfolio_weights(scores, returns, mask, cfg)

    # 第 15 天得分变动极小，触发缓冲带，权重应与第 14 天完全一致
    np.testing.assert_allclose(weights[15, :], weights[14, :], atol=1e-7)
    # 第 16 天反转，超出缓冲带，权重必须调整
    assert not np.allclose(weights[16, :], weights[15, :], atol=1e-4)


def test_build_portfolio_weights_dimension_mismatch():
    scores = np.ones((10, 5))
    returns = np.ones((10, 4))  # 列数不匹配
    mask = np.ones((10, 5), dtype=bool)
    cfg = PortfolioConfig()

    with pytest.raises(ValueError, match="维度必须完全一致"):
        build_portfolio_weights(scores, returns, mask, cfg)
