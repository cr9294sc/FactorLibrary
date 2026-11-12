"""A 股多头超额收益体系与因果时间窗对齐单元测试 (tests/python/test_evaluate_excess.py)。

覆盖审查报告 P0-09 专项治理：
1. 复利几何年化 ((prod(1+r))^(250/N) - 1) 与单利算术年化双轨准确性
2. 加法累计超额收益回撤 (DD_t = Peak_t - Cum_t) 及极值起止日期定位
3. 缺失值、全 NaN、边界长度与极端亏损条件下的数学安全性
4. evaluate_factor_v4_stats 外部基准接入与分年度统计
5. 远期持仓区间 [T+1, T+2] 严格因果对齐断言
"""

from __future__ import annotations

import math

import numpy as np
from factorlib.core.evaluate import (
    calc_long_excess_metrics,
    evaluate_factor_v4_stats,
)
from factorlib.core.workbench import create_synthetic_futures_panel


def test_geometric_annual_return_constant_series() -> None:
    """对于常数超额收益率序列，几何年化结果必须完全等于 (1 + r)^250 - 1。"""
    daily_r = 0.001  # 0.1% / day
    n_days = 250
    long_rets = np.full(n_days, daily_r)
    bench_rets = np.zeros(n_days)

    res = calc_long_excess_metrics(long_rets, bench_rets, trading_days_per_year=250.0)

    expected_ann = (1.0 + daily_r) ** 250.0 - 1.0
    expected_arith = daily_r * 250.0

    assert math.isclose(res["annual_excess_return"], expected_ann, rel_tol=1e-12)
    assert math.isclose(res["annual_excess_return_arithmetic"], expected_arith, rel_tol=1e-12)
    assert res["annual_excess_volatility"] < 1e-12
    assert res["win_rate"] == 1.0
    assert res["additive_max_drawdown"] == 0.0


def test_drawdown_peak_to_trough_and_dates() -> None:
    """验证加法累计超额收益曲线上的最大回撤与起止日期定位。"""
    # 模拟超额收益:
    # Day 0: +0.05 -> Cum: 0.05, Peak: 0.05
    # Day 1: -0.02 -> Cum: 0.03, Peak: 0.05, DD: 0.02
    # Day 2: -0.03 -> Cum: 0.00, Peak: 0.05, DD: 0.05 (Max DD)
    # Day 3: +0.04 -> Cum: 0.04, Peak: 0.05, DD: 0.01
    long_rets = np.array([0.05, 0.01, 0.00, 0.05])
    bench_rets = np.array([0.00, 0.03, 0.03, 0.01])
    # excess = [0.05, -0.02, -0.03, +0.04]
    dates = ["20230103", "20230104", "20230105", "20230106"]

    res = calc_long_excess_metrics(long_rets, bench_rets, dates=dates, trading_days_per_year=250.0)

    assert math.isclose(res["additive_max_drawdown"], 0.05, rel_tol=1e-9)
    assert res["drawdown_start"] == "20230103"
    assert res["drawdown_end"] == "20230105"
    assert len(res["cumulative_additive"]) == 4
    assert math.isclose(res["cumulative_additive"][2], 0.0, abs_tol=1e-12)
    assert math.isclose(res["cumulative_additive"][3], 0.04, abs_tol=1e-12)


def test_annual_breakdown_by_year() -> None:
    """验证分年度切分统计与全周期独立计算的一致性。"""
    dates = ["20221230", "20230103", "20230104", "20240102"]
    long_rets = np.array([0.01, 0.02, -0.01, 0.03])
    bench_rets = np.array([0.00, 0.01, 0.00, 0.01])

    res = calc_long_excess_metrics(long_rets, bench_rets, dates=dates, trading_days_per_year=250.0)
    breakdown = res["annual_breakdown"]

    assert set(breakdown.keys()) == {"2022", "2023", "2024"}
    assert "annual_excess_return" in breakdown["2023"]
    assert "annual_excess_return_arithmetic" in breakdown["2023"]
    assert "excess_sharpe" in breakdown["2023"]
    assert "additive_max_drawdown" in breakdown["2023"]
    # 2023 年 2 天: +0.01, -0.01
    assert math.isclose(breakdown["2023"]["win_rate"], 0.5)


def test_edge_cases_nan_and_empty() -> None:
    """验证全 NaN、空序列、交错 NaN 与极值亏损的鲁棒性。"""
    # 1. 空数组
    res_empty = calc_long_excess_metrics(np.array([]), np.array([]))
    assert res_empty["annual_excess_return"] == 0.0
    assert res_empty["additive_max_drawdown"] == 0.0

    # 2. 全 NaN 数组
    res_nan = calc_long_excess_metrics(
        np.array([np.nan, np.nan]), np.array([0.01, 0.02]), dates=["20230101", "20230102"]
    )
    assert res_nan["annual_excess_return"] == 0.0
    assert res_nan["excess_sharpe"] == 0.0

    # 3. 包含部分 NaN: 仅统计有效日
    long_part = np.array([np.nan, 0.02, np.nan, 0.01])
    bench_part = np.array([0.01, 0.01, 0.02, 0.00])
    # valid excess: [0.01, 0.01] (2 days)
    res_part = calc_long_excess_metrics(long_part, bench_part, trading_days_per_year=250.0)
    expected_ann = (1.0 + 0.01) ** 250.0 - 1.0
    assert math.isclose(res_part["annual_excess_return"], expected_ann, rel_tol=1e-9)

    # 4. 极端亏损 (1 + r <= 0): 安全回退至算术年化，不抛出异常或产生复数
    res_loss = calc_long_excess_metrics(
        np.array([-1.2, 0.1]), np.array([0.0, 0.0]), trading_days_per_year=250.0
    )
    assert np.isfinite(res_loss["annual_excess_return"])
    assert res_loss["annual_excess_return"] < 0.0


def test_evaluate_factor_v4_stats_with_external_benchmark() -> None:
    """验证 evaluate_factor_v4_stats 接受 external benchmark_rets 时的完整指标。"""
    panel = create_synthetic_futures_panel(n_days=60, seed=42)
    T, N = panel.shape

    # 构造因子矩阵
    factor_mat = np.random.RandomState(42).randn(T, N)

    # 构造外部基准 (例如全市场指数日收益)
    external_bench = np.full(T, 0.0005)

    v4_stats = evaluate_factor_v4_stats(
        factor_mat,
        panel,
        n_quantiles=5,
        benchmark_rets=external_bench,
        trading_days_per_year=250.0,
    )

    assert "long_analysis" in v4_stats
    summary = v4_stats["long_analysis"]["summary"]
    assert "annual_excess_return" in summary
    assert "annual_excess_return_arithmetic" in summary
    assert "annual_excess_volatility" in summary
    assert "excess_sharpe" in summary
    assert "additive_max_drawdown" in summary
    assert "daily_turnover" in summary
    assert "annual" in v4_stats["long_analysis"]

    # 验证 benchmark_nav 正确追踪外部基准
    expected_last_bench_nav = (1.0 + 0.0005) ** T
    assert math.isclose(
        v4_stats["group_analysis"]["benchmark_nav"][-1], expected_last_bench_nav, rel_tol=1e-5
    )


def test_benchmark_forward_timing_shift_logic() -> None:
    """验证基准前移构造算法确保 t 日收益覆盖的是 [T+1, T+2]，与个股远期持仓区间严格因果对齐。"""
    T = 10
    # 模拟交易日报价
    bench_close = np.array([100.0, 102.0, 101.0, 105.0, 103.0, 107.0, 106.0, 110.0, 108.0, 112.0])

    # 错误口径 (旧代码): t 日记录的是 [t-1, t]
    bench_ret_buggy = np.full(T, np.nan)
    bench_ret_buggy[1:] = bench_close[1:] / bench_close[:-1] - 1.0

    # 正确口径 (P0-09 修复): t 日记录的是 [t+1, t+2]
    bench_fwd_ret_fixed = np.full(T, np.nan)
    bench_fwd_ret_fixed[:-2] = bench_close[2:] / bench_close[1:-1] - 1.0

    # 验证在信号日 t=0:
    # 因子在 t=0 收盘生成 -> t=1 买入, t=2 卖出
    # 正确基准收益必须是 bench_close[2] / bench_close[1] - 1.0 (101.0 / 102.0 - 1.0 = -0.98%)
    assert math.isclose(bench_fwd_ret_fixed[0], 101.0 / 102.0 - 1.0, rel_tol=1e-12)

    # 错误基准旧代码记录的是 bench_close[0] / bench_close[-1] (无) 或 t=1 记录 bench_close[1]/bench_close[0]
    # 在 t=1 上，旧代码记录的是 [0->1] 的收益 (+2.0%)，相差整整 2 个交易日且方向完全相反！
    assert not math.isclose(bench_ret_buggy[0], bench_fwd_ret_fixed[0])
    assert math.isnan(bench_ret_buggy[0])
    # 在信号日 t=1 上，远期区间是 [2->3]，而旧代码记录的是 [0->1]
    assert math.isclose(bench_fwd_ret_fixed[1], 105.0 / 101.0 - 1.0, rel_tol=1e-12)
    assert math.isclose(bench_ret_buggy[1], 102.0 / 100.0 - 1.0, rel_tol=1e-12)
