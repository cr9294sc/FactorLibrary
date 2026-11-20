"""A 股模块化融合优化器单元测试 (tests/python/test_stocks_optimizer.py)。

覆盖范围：
1. ``SignalShaper``：输出范围、NaN 保留、幂次单调、行业中性化、EMA 收敛与无未来函数；
2. ``DynamicRiskGuard``：SIZE 暴跌触发防守体制、常态不触发、严格因果与尺寸缺失容错；
3. ``ConvexOptimizer``：纯 NumPy 降级路径与 cvxpy 真实求解的约束满足性、兜底层级；
4. ``drift_positions`` / ``PositionDriftTracker``：归一化与价格缺失容错；
5. ``BarraPortfolioOptimizer.build_weights``：端到端形状/顺序/权重和/非负/无 NaN。

全部使用合成数据（小规模面板，无需真实 Barra 数据资产）。
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
from factorlib.core.panel import Panel
from factorlib.markets.stocks import optimizer as opt

# Barra CNE5 因子名（与 BARRA_FACTOR_ORDER 一致：10 风格 + country + 31 行业）
_FACTOR_NAMES: tuple[str, ...] = (
    "MOMENTUM",
    "RESVOL",
    "BETA",
    "LIQUIDTY",
    "LEVERAGE",
    "GROWTH",
    "BTOP",
    "EARNYILD",
    "SIZE",
    "SIZENL",
    "country",
    *(str(i) for i in range(1, 32)),
)
_SIZE_COL = _FACTOR_NAMES.index("SIZE")
_HAS_CVXPY = opt.cp is not None
_HAS_SOLVER = bool(set(opt.available_cvxpy_solvers()) & {"CLARABEL", "SCS", "OSQP", "MOSEK"})


# ==============================================================================
# 合成夹具
# ==============================================================================


def _make_panel(n_days: int = 30, n_symbols: int = 40, seed: int = 7) -> Panel:
    """构造含 2 个风格、3 个行业、市值、停牌与失效股票的合成面板。"""
    rng = np.random.default_rng(seed)
    dates = np.array([np.datetime64("2024-01-02") + np.timedelta64(i, "D") for i in range(n_days)])
    symbols = [f"{600000 + i:06d}.XSHG" for i in range(n_symbols)]
    mkt_cap = np.asfortranarray(np.exp(rng.normal(10.0, 1.0, size=(n_days, n_symbols))))
    industry_codes = np.tile(np.array([1.0, 2.0, 3.0]), n_symbols // 3 + 1)[:n_symbols]
    industry = np.asfortranarray(np.broadcast_to(industry_codes, (n_days, n_symbols)).copy())
    fields = {
        "mkt_cap": mkt_cap,
        "barra_industry": industry,
        "barra_size": np.asfortranarray(rng.normal(0.0, 1.0, size=(n_days, n_symbols))),
        "barra_beta": np.asfortranarray(rng.normal(0.0, 1.0, size=(n_days, n_symbols))),
        "barra_resvol": np.asfortranarray(rng.uniform(0.01, 0.06, size=(n_days, n_symbols))),
        "spec_vol_ewma": np.asfortranarray(rng.uniform(0.005, 0.03, size=(n_days, n_symbols))),
    }
    suspend = np.zeros((n_days, n_symbols), dtype=np.float64, order="F")
    if n_days > 7 and n_symbols > 3:
        suspend[5, 3] = 1.0
        suspend[7, 0] = 1.0
    mask = np.ones((n_days, n_symbols), dtype=bool, order="F")
    if n_days > 12 and n_symbols > 5:
        mask[12, 5] = False
    return Panel(
        dates=dates,
        symbols=symbols,
        fields=fields,
        mask=mask,
        aux_fields={"is_suspended": np.asfortranarray(suspend)},
    )


def _make_factor_returns(
    n_days: int = 60,
    *,
    crash: bool = True,
    seed: int = 13,
) -> np.ndarray:
    """构造 Barra 42 因子日收益矩阵：SIZE 列人为制造 10 日累计暴跌。"""
    rng = np.random.default_rng(seed)
    returns = rng.normal(0.0, 0.002, size=(n_days, len(_FACTOR_NAMES)))
    if crash:
        returns[:, _SIZE_COL] = 0.0
        returns[25:30, _SIZE_COL] = -0.02
    else:
        returns[:, _SIZE_COL] = 0.001
    return returns


def _make_cross_section(n_symbols: int = 40, seed: int = 11) -> dict[str, np.ndarray]:
    """构造单日优化输入（alpha、基准、持仓、暴露、行业与特质方差）。"""
    rng = np.random.default_rng(seed)
    return {
        "alpha": rng.normal(0.0, 1.0, size=n_symbols),
        "wb": np.full(n_symbols, 1.0 / n_symbols),
        "prev": np.full(n_symbols, 1.0 / n_symbols),
        "exposures": rng.normal(0.0, 1.0, size=(n_symbols, 2)),
        "factor_cov_sqrt": np.eye(2, dtype=np.float64) * 0.02,
        "industry": np.tile(np.array([1.0, 2.0, 3.0]), n_symbols // 3 + 1)[:n_symbols],
        "spec_var": np.full(n_symbols, 1e-4),
    }


# ==============================================================================
# 1) SignalShaper
# ==============================================================================


def test_signal_shaper_range_and_nan_preserved() -> None:
    """输出范围受控、NaN 严格保留、EMA 不产生额外 NaN。"""
    rng = np.random.default_rng(101)
    scores = rng.normal(0.0, 1.0, size=(20, 15))
    scores[3, 4] = np.nan
    scores[3, 5] = np.nan
    scores[9, :] = np.nan
    industry = np.tile(np.array([1.0, 2.0, 3.0]), 5)[:15]
    resvol = rng.uniform(0.01, 0.05, size=(20, 15))

    shaper = opt.SignalShaper()
    shaped = shaper.shape_scores(scores, industry=industry, resvol=resvol)

    assert shaped.shape == scores.shape
    assert shaped.flags.f_contiguous
    np.testing.assert_array_equal(np.isfinite(shaped), np.isfinite(scores))
    # |rank_score| <= 1，低波倾斜后可放大到 (1 + vol_weight)，再经幂次放大
    bound = (1.0 + shaper.cfg.vol_weight) ** shaper.cfg.power
    assert np.nanmax(np.abs(shaped)) <= bound + 1e-9
    assert not np.any(np.abs(shaped) == np.inf)


def test_signal_shaper_power_is_monotone_head_amplifier() -> None:
    """power >= 1 时对 |rank| <= 1 的信号是单调收缩放大，且不改变符号。"""
    rng = np.random.default_rng(202)
    scores = rng.normal(0.0, 1.0, size=(10, 25))
    base = opt.SignalShaperConfig(power=1.0, ind_neutral_weight=0.0, vol_weight=0.0, memory=0.0)
    strong = replace(base, power=2.5)
    weak_out = opt.SignalShaper(base).shape_scores(scores)
    strong_out = opt.SignalShaper(strong).shape_scores(scores)

    assert np.all(np.abs(strong_out) <= np.abs(weak_out) + 1e-12)
    np.testing.assert_array_equal(np.sign(strong_out), np.sign(weak_out))


def test_signal_shaper_industry_neutralization_means_zero() -> None:
    """``ind_neutral_weight=1.0`` 时软行业中性化后的组内均值为 0（与公开原语一致）。"""
    rng = np.random.default_rng(303)
    n_days, n_symbols = 12, 30
    scores = rng.normal(0.0, 1.0, size=(n_days, n_symbols))
    codes = np.tile(np.array([1.0, 2.0, 3.0]), n_symbols // 3)[:n_symbols]
    industry = np.broadcast_to(codes, (n_days, n_symbols))

    cfg = opt.SignalShaperConfig(ind_neutral_weight=1.0, vol_weight=0.0, power=1.0, memory=0.0)
    shaped = opt.SignalShaper(cfg).shape_scores(scores, industry=industry)

    ranks = np.stack([opt.centered_rank_scores(scores[t]) for t in range(n_days)])
    neutral = np.stack(
        [opt.soft_industry_neutralize(ranks[t], industry[t], 1.0) for t in range(n_days)]
    )
    expected = np.stack([opt.centered_rank_scores(neutral[t]) for t in range(n_days)])
    np.testing.assert_allclose(shaped, expected, atol=1e-12)

    for code in (1.0, 2.0, 3.0):
        sel = np.rint(codes) == code
        assert np.max(np.abs(np.mean(neutral[:, sel], axis=1))) < 1e-12
    # 中性化后行业均值不再残留原始信号的行业偏移
    raw_means = np.abs(np.mean(ranks[:, np.rint(codes) == 1.0], axis=1))
    assert np.max(raw_means) > 0.0


def test_signal_shaper_ema_memory_converges() -> None:
    """EMA 慢速记忆按 ``mu*state + (1-mu)*alpha`` 递推并收敛到静态 alpha。"""
    rng = np.random.default_rng(404)
    row = rng.normal(0.0, 1.0, size=(1, 12))
    cfg = opt.SignalShaperConfig(memory=0.75, ind_neutral_weight=0.0, vol_weight=0.0, power=1.0)
    shaper = opt.SignalShaper(cfg)
    outputs = np.vstack([shaper.shape_scores(row) for _ in range(80)])
    static = opt.SignalShaper(replace(cfg, memory=0.0)).shape_scores(row)

    np.testing.assert_allclose(outputs[-1], np.ravel(static), atol=1e-6)
    expected_second = 0.75 * outputs[0] + 0.25 * np.ravel(static)
    np.testing.assert_allclose(outputs[1], expected_second, atol=1e-12)
    flipped = opt.SignalShaper(cfg)
    first = flipped.shape_scores(row)[0]
    second = flipped.shape_scores(-row)[0]
    np.testing.assert_allclose(second, 0.5 * first, atol=1e-12)


def test_signal_shaper_no_lookahead() -> None:
    """篡改 ``t`` 之后的数据不影响 ``t`` 及之前的输出（严格因果）。"""
    rng = np.random.default_rng(505)
    n_days, n_symbols = 14, 10
    scores = rng.normal(0.0, 1.0, size=(n_days, n_symbols))
    industry = np.tile(np.array([1.0, 2.0]), n_symbols // 2)
    resvol = rng.uniform(0.01, 0.05, size=(n_days, n_symbols))

    baseline = opt.SignalShaper().shape_scores(scores, industry=industry, resvol=resvol)
    perturbed = scores.copy()
    split = 8
    perturbed[split:] = rng.normal(0.0, 5.0, size=(n_days - split, n_symbols))
    modified = opt.SignalShaper().shape_scores(perturbed, industry=industry, resvol=resvol)

    np.testing.assert_array_equal(baseline[:split], modified[:split])
    assert not np.array_equal(baseline[split:], modified[split:])


# ==============================================================================
# 2) DynamicRiskGuard
# ==============================================================================


def test_dynamic_risk_guard_triggers_on_size_crash() -> None:
    """SIZE 暴跌序列触发防守体制，并给出放大后的低波权重与收缩后的幂次。"""
    guard = opt.DynamicRiskGuard(opt.DynamicRiskGuardConfig(), factor_names=_FACTOR_NAMES)
    state = guard.assess(_make_factor_returns(crash=True), dates=None)

    active = np.flatnonzero(state.active)
    assert active.size > 0
    assert 29 <= active.max() < 60
    assert not state.active[:20].any()
    assert np.all(state.vol_weight_scale[active] == pytest.approx(1.8))
    assert np.all(state.power_scale[active] == pytest.approx(0.85))
    assert np.all(state.size_tilt[active] == pytest.approx(0.15))
    assert np.all(state.stress_score[active] > 0.0)

    shaper_cfg = opt.SignalShaperConfig()
    vol_weight, power, size_tilt = guard.apply(state, int(active.max()), shaper_cfg)
    assert vol_weight == pytest.approx(shaper_cfg.vol_weight * 1.8)
    assert power == pytest.approx(shaper_cfg.power * 0.85)
    assert size_tilt == pytest.approx(0.15)
    # 不修改原配置对象
    assert shaper_cfg == opt.SignalShaperConfig()


def test_dynamic_risk_guard_normal_series_inactive_and_causal() -> None:
    """常态序列不触发防守体制；截断未来数据后历史判定完全一致。"""
    guard = opt.DynamicRiskGuard(opt.DynamicRiskGuardConfig(), factor_names=_FACTOR_NAMES)
    normal = _make_factor_returns(crash=False)
    state = guard.assess(normal)
    assert not state.active.any()
    assert np.all(state.vol_weight_scale == pytest.approx(1.0))
    shaper_cfg = opt.SignalShaperConfig()
    assert guard.apply(state, 0, shaper_cfg) == pytest.approx(
        (shaper_cfg.vol_weight, shaper_cfg.power, 0.0)
    )

    crash = _make_factor_returns(crash=True)
    full = guard.assess(crash)
    split = 30
    truncated = guard.assess(crash[: split + 1])
    np.testing.assert_array_equal(full.active[: split + 1], truncated.active)
    np.testing.assert_allclose(full.stress_score[: split + 1], truncated.stress_score)
    np.testing.assert_allclose(full.vol_weight_scale[: split + 1], truncated.vol_weight_scale)


def test_dynamic_risk_guard_missing_size_factor_is_inactive() -> None:
    """无法定位 SIZE 因子（因子名缺失）时不得触发防守体制。"""
    guard = opt.DynamicRiskGuard(opt.DynamicRiskGuardConfig(), factor_names=_FACTOR_NAMES)
    returns = _make_factor_returns(crash=True)
    state = guard.assess(returns, factor_names=())
    assert not state.active.any()
    # 错位名称数量也视为不可定位
    state_short = guard.assess(returns, factor_names=("SIZE",))
    assert not state_short.active.any()


# ==============================================================================
# 3) ConvexOptimizer
# ==============================================================================


def test_convex_optimizer_numpy_path_constraints() -> None:
    """纯 NumPy 降级路径满足满仓、单票上限、换手上限、停牌锁定与主动权重上限。"""
    n_symbols = 40
    data = _make_cross_section(n_symbols)
    cfg = opt.ConvexOptimizerConfig(
        cash_buffer_max=0.02,
        max_single_weight=0.05,
        max_active_weight=0.02,
        max_daily_turnover=0.10,
        style_deviation_overrides={},
    )
    optimizer = opt.ConvexOptimizer(cfg, force_numpy=True)
    mask_active = np.ones(n_symbols, dtype=bool)
    mask_active[7] = False
    mask_suspend = np.zeros(n_symbols, dtype=bool)
    mask_suspend[3] = True

    result = optimizer.solve(
        data["alpha"],
        w_prev=data["prev"],
        wb=data["wb"],
        mask_active=mask_active,
        mask_suspend=mask_suspend,
    )
    weights = result.weights

    assert result.solver_used == "NUMPY"
    assert result.fallback_level == 0
    assert weights.shape == (n_symbols,)
    assert np.all(np.isfinite(weights))
    assert np.all(weights >= -1e-12)
    assert np.all(weights <= cfg.max_single_weight + 1e-12)
    invest = 1.0 - float(result.diagnostics["cash_buffer"])
    assert weights.sum() == pytest.approx(invest, abs=1e-9)
    assert np.abs(weights - data["prev"]).sum() <= cfg.max_daily_turnover + 1e-9
    assert np.abs(weights - data["wb"]).max() <= cfg.max_active_weight + 1e-9
    assert weights[3] == pytest.approx(data["prev"][3])
    assert weights[7] == pytest.approx(data["prev"][7])


def test_convex_optimizer_numpy_initial_build_and_turnover_clip() -> None:
    """初始建仓豁免换手约束；换手约束收紧时总量按上限截断且不越界。"""
    n_symbols = 30
    data = _make_cross_section(n_symbols)
    cfg = opt.ConvexOptimizerConfig(
        cash_buffer_max=0.0,
        max_single_weight=0.10,
        max_active_weight=0.5,
        max_daily_turnover=0.05,
        style_deviation_overrides={},
    )
    optimizer = opt.ConvexOptimizer(cfg, force_numpy=True)

    initial = optimizer.solve(data["alpha"], w_prev=np.zeros(n_symbols), wb=data["wb"])
    assert initial.diagnostics["initial_build"] is True
    assert initial.weights.sum() == pytest.approx(1.0, abs=1e-9)

    prev = np.full(n_symbols, 1.0 / n_symbols)
    clipped = optimizer.solve(data["alpha"], w_prev=prev, wb=data["wb"])
    assert np.abs(clipped.weights - prev).sum() <= cfg.max_daily_turnover + 1e-9
    assert clipped.weights.sum() <= 1.0 + 1e-9
    assert clipped.diagnostics["turnover_clipped"] is True
    assert np.all(clipped.weights >= 0.0)


@pytest.mark.skipif(not (_HAS_CVXPY and _HAS_SOLVER), reason="cvxpy 或求解器不可用")
def test_convex_optimizer_cvxpy_constraints() -> None:
    """cvxpy 真实求解满足满仓/现金、上下限、风格行业偏离与换手上限（1e-6 容差）。"""
    n_symbols = 40
    data = _make_cross_section(n_symbols)
    cfg = opt.ConvexOptimizerConfig(
        cash_buffer_max=0.02,
        max_single_weight=0.06,
        max_active_weight=0.03,
        max_daily_turnover=0.5,
        style_deviation_default=0.8,
        style_deviation_overrides={},
        industry_deviation=0.05,
        solver_priority=("CLARABEL", "SCS", "OSQP"),
        allow_slack_fallback=False,
    )
    optimizer = opt.ConvexOptimizer(cfg)
    result = optimizer.solve(
        data["alpha"],
        w_prev=data["prev"],
        wb=data["wb"],
        exposures=data["exposures"],
        factor_cov_sqrt=data["factor_cov_sqrt"],
        industry=data["industry"],
        spec_var=data["spec_var"],
        style_names=("SIZE", "BETA"),
    )
    weights = result.weights
    tol = 1e-6

    assert result.fallback_level == 1
    assert result.status in ("optimal", "optimal_inaccurate")
    invest = 1.0 - float(result.diagnostics["cash_buffer"])
    assert weights.sum() == pytest.approx(invest, abs=1e-6)
    assert np.all(weights >= -tol)
    assert np.all(weights <= cfg.max_single_weight + tol)
    assert np.abs(weights - data["wb"]).max() <= cfg.max_active_weight + tol
    assert np.abs(weights - data["prev"]).sum() <= cfg.max_daily_turnover + tol

    residual = weights - data["wb"]
    style_dev = np.abs(data["exposures"].T @ residual)
    assert np.all(style_dev <= cfg.style_deviation_default + tol)
    for code in (1.0, 2.0, 3.0):
        indicator = (data["industry"] == code).astype(np.float64)
        assert abs(float(indicator @ residual)) <= cfg.industry_deviation + tol
    assert np.isfinite(result.objective)


@pytest.mark.skipif(not (_HAS_CVXPY and _HAS_SOLVER), reason="cvxpy 或求解器不可用")
def test_convex_optimizer_horizon_mode() -> None:
    """``horizon_mode=True`` 且 ``risk_aversion=0`` 时双期 Horizon 规划可正常求解。"""
    n_symbols = 20
    rng = np.random.default_rng(808)
    alpha = rng.normal(0.0, 1.0, size=n_symbols)
    forecast = rng.normal(0.0, 1.0, size=n_symbols)
    wb = np.full(n_symbols, 1.0 / n_symbols)
    cfg = opt.ConvexOptimizerConfig(
        cash_buffer_max=0.01,
        max_single_weight=0.15,
        max_active_weight=0.2,
        max_daily_turnover=0.5,
        risk_aversion=0.0,
        horizon_mode=True,
        style_deviation_overrides={},
        solver_priority=("CLARABEL", "SCS", "OSQP"),
        allow_slack_fallback=False,
    )
    result = opt.ConvexOptimizer(cfg).solve(alpha, w_prev=wb, wb=wb, forecast=forecast)
    assert result.fallback_level == 1
    assert result.weights.sum() <= 1.0 + 1e-6
    assert np.all(result.weights >= -1e-6)


@pytest.mark.skipif(not _HAS_CVXPY, reason="未安装 cvxpy")
def test_convex_optimizer_fallback_level3_on_infeasible() -> None:
    """不可行的硬约束（单票上限过小）触发 Level3 持仓归一化保底。"""
    n_symbols = 6
    cfg = opt.ConvexOptimizerConfig(
        cash_buffer_max=0.0,
        max_single_weight=0.1,
        max_active_weight=1.0,
        allow_slack_fallback=True,
        style_deviation_overrides={},
        solver_priority=("CLARABEL", "SCS", "OSQP"),
    )
    prev = np.full(n_symbols, 1.0 / n_symbols)
    result = opt.ConvexOptimizer(cfg).solve(np.ones(n_symbols), w_prev=prev, wb=prev)
    assert result.fallback_level == 3
    assert result.solver_used == "FALLBACK"
    assert result.status == "fallback"
    assert result.weights.shape == (n_symbols,)
    assert np.all(np.isfinite(result.weights))
    assert np.all(result.weights >= 0.0)
    assert result.weights.sum() <= 1.0 + 1e-9


def test_convex_optimizer_raises_without_cvxpy(monkeypatch: pytest.MonkeyPatch) -> None:
    """cvxpy 缺失且未启用 force_numpy 时抛出带安装提示的 RuntimeError。"""
    monkeypatch.setattr(opt, "cp", None)
    optimizer = opt.ConvexOptimizer(opt.ConvexOptimizerConfig())
    with pytest.raises(RuntimeError, match="cvxpy"):
        optimizer.solve(np.ones(4), w_prev=np.zeros(4))


# ==============================================================================
# 4) 持仓漂移闭环
# ==============================================================================


def test_drift_positions_normalizes_and_handles_missing_prices() -> None:
    """漂移后按 reserve_cash 归一化；价格缺失/为 0 时该标的不漂移；全无效则原样返回。"""
    weights = np.array([0.5, 0.5])
    drifted = opt.drift_positions(
        weights,
        np.array([10.0, 10.0]),
        np.array([11.0, 10.0]),
        reserve_cash=0.98,
    )
    assert drifted.sum() == pytest.approx(0.98)
    assert drifted[0] > drifted[1]

    partial = opt.drift_positions(weights, np.array([10.0, 0.0]), np.array([11.0, 10.0]))
    gross = 0.5 * (11.0 / 10.0) + 0.5
    assert partial.sum() == pytest.approx(1.0)
    assert partial[0] == pytest.approx(0.5 * (11.0 / 10.0) / gross)
    assert partial[1] == pytest.approx(0.5 / gross)

    nan_price = opt.drift_positions(weights, np.array([np.nan, np.nan]), np.array([np.nan, np.nan]))
    np.testing.assert_array_equal(nan_price, weights)

    zero_total = opt.drift_positions(np.zeros(2), np.array([10.0, 10.0]), np.array([11.0, 12.0]))
    np.testing.assert_array_equal(zero_total, np.zeros(2))


def test_position_drift_tracker_state_and_tolerance() -> None:
    """追踪器逐步维护 last_hold，价格缺失不产生 NaN，reset 清空状态。"""
    tracker = opt.PositionDriftTracker(reserve_cash=0.95)
    assert tracker.last_hold is None

    first = tracker.update(np.array([0.6, 0.4]), np.array([10.0, 20.0]), np.array([10.5, 19.0]))
    assert first.sum() == pytest.approx(0.95)
    np.testing.assert_allclose(tracker.last_hold, first)

    second = tracker.update(np.array([0.6, 0.4]), np.array([np.nan, 20.0]), np.array([10.5, 19.0]))
    assert second.sum() == pytest.approx(0.95)
    assert np.all(np.isfinite(tracker.last_hold))

    # 不传价格时退化为按目标权重归一化
    third = tracker.update(np.array([0.3, 0.7]))
    assert third.sum() == pytest.approx(0.95)
    np.testing.assert_allclose(third, np.array([0.285, 0.665]), atol=1e-12)

    tracker.reset()
    assert tracker.last_hold is None


# ==============================================================================
# 5) 门面端到端
# ==============================================================================


def _make_end_to_end_config(**overrides: object) -> opt.BarraOptimizerConfig:
    """端到端测试使用的松约束配置。"""
    base = opt.BarraOptimizerConfig(
        cash_buffer_max=0.02,
        max_single_weight=0.05,
        max_active_weight=0.05,
        max_daily_turnover=1.0,
        industry_deviation=0.3,
        style_deviation_default=0.8,
        style_deviation_overrides={},
        proxy_rank_range=(1, 15),
    )
    return replace(base, **overrides) if overrides else base


def test_build_weights_end_to_end_numpy() -> None:
    """NumPy 降级模式的端到端输出：形状、(T,N) Fortran 顺序、权重和、非负、无 NaN。"""
    panel = _make_panel(n_days=25, n_symbols=40, seed=21)
    rng = np.random.default_rng(22)
    scores = rng.normal(0.0, 1.0, size=(25, 40))
    scores[4, 4] = np.nan
    scores[9, 11] = np.nan
    factor_returns = _make_factor_returns(n_days=25, crash=True, seed=23)

    optimizer = opt.BarraPortfolioOptimizer(
        _make_end_to_end_config(),
        force_numpy=True,
        factor_names=_FACTOR_NAMES,
    )
    weights = optimizer.build_weights(
        panel, scores, factor_returns=factor_returns, dates=panel.dates
    )

    assert weights.shape == (25, 40)
    assert weights.flags.f_contiguous
    assert np.all(np.isfinite(weights))
    assert np.all(weights >= 0.0)
    row_sums = weights.sum(axis=1)
    assert np.all(row_sums <= 1.0 + 1e-9)
    assert np.all(row_sums > 0.5)


@pytest.mark.skipif(not (_HAS_CVXPY and _HAS_SOLVER), reason="cvxpy 或求解器不可用")
def test_build_weights_end_to_end_cvxpy() -> None:
    """cvxpy 模式的小规模端到端回测：逐日可行权重、无 NaN、权重和不超过 1。"""
    panel = _make_panel(n_days=4, n_symbols=20, seed=31)
    rng = np.random.default_rng(32)
    scores = rng.normal(0.0, 1.0, size=(4, 20))
    factor_returns = _make_factor_returns(n_days=4, crash=False, seed=33)
    cfg = _make_end_to_end_config(
        max_single_weight=0.2,
        max_active_weight=0.15,
        max_daily_turnover=0.5,
        risk_aversion=1.0,
        solver_priority=("CLARABEL", "SCS", "OSQP"),
    )
    optimizer = opt.BarraPortfolioOptimizer(cfg, factor_names=_FACTOR_NAMES, force_numpy=False)
    weights = optimizer.build_weights(
        panel, scores, factor_returns=factor_returns, dates=panel.dates
    )

    assert weights.shape == (4, 20)
    assert weights.flags.f_contiguous
    assert np.all(np.isfinite(weights))
    assert np.all(weights >= -1e-6)
    assert np.all(weights.sum(axis=1) <= 1.0 + 1e-6)


def test_build_weights_handles_missing_optional_fields() -> None:
    """面板缺失市值/风格/停牌字段时仍可运行（全部读取容错）。"""
    n_days, n_symbols = 6, 12
    industry = np.tile(np.array([1.0, 2.0, 3.0]), n_symbols // 3)[None, :]
    panel = Panel(
        dates=np.array(
            [np.datetime64("2024-03-01") + np.timedelta64(i, "D") for i in range(n_days)]
        ),
        symbols=[f"{i:06d}.XSHE" for i in range(n_symbols)],
        fields={
            "barra_industry": np.asfortranarray(np.broadcast_to(industry, (n_days, n_symbols))),
        },
        mask=np.ones((n_days, n_symbols), dtype=bool, order="F"),
    )
    scores = np.random.default_rng(41).normal(0.0, 1.0, size=(n_days, n_symbols))
    optimizer = opt.BarraPortfolioOptimizer(
        _make_end_to_end_config(max_single_weight=0.2, proxy_rank_range=None),
        force_numpy=True,
    )
    weights = optimizer.build_weights(panel, scores)
    assert weights.shape == (n_days, n_symbols)
    assert np.all(np.isfinite(weights))
    assert np.all(weights.sum(axis=1) <= 1.0 + 1e-9)


# ==============================================================================
# 6) 配置解析与校验
# ==============================================================================


def test_configs_from_dict_and_validation() -> None:
    """嵌套 YAML 字典解析（未知键忽略、类型转换）与非法取值校验。"""
    raw = {
        "shaper": {
            "ind_neutral_weight": 0.5,
            "vol_weight": 0.2,
            "power": 2.0,
            "memory": 0.6,
            "unknown": 1,
        },
        "risk_guard": {
            "lookback_days": 5,
            "vol_multiplier": 2.0,
            "cumret_threshold": -0.05,
            "unknown": 2,
        },
        "optimizer": {
            "risk_aversion": 3.0,
            "solver_priority": ["CLARABEL"],
            "style_deviation_overrides": {"size": 0.1},
            "proxy_rank_range": [1, 100],
            "asymmetric_trade_penalty": [1.0, 0.5],
            "unknown": 3,
        },
        "seq_shape": False,
        "unknown_key": "ignored",
    }
    cfg = opt.BarraOptimizerConfig.from_dict(raw)
    assert isinstance(cfg, opt.ConvexOptimizerConfig)
    assert cfg.shaper.ind_neutral_weight == pytest.approx(0.5)
    assert cfg.shaper.memory == pytest.approx(0.6)
    assert cfg.risk_guard.lookback_days == 5
    assert cfg.risk_guard.cumret_threshold == pytest.approx(-0.05)
    assert cfg.risk_aversion == pytest.approx(3.0)
    assert cfg.solver_priority == ("CLARABEL",)
    assert cfg.style_deviation_overrides == {"SIZE": 0.1}
    assert cfg.proxy_rank_range == (1, 100)
    assert cfg.asymmetric_trade_penalty == (1.0, 0.5)
    assert cfg.seq_shape is False

    flat = opt.BarraOptimizerConfig.from_dict({"max_single_weight": 0.02})
    assert flat.max_single_weight == pytest.approx(0.02)

    with pytest.raises(ValueError):
        opt.SignalShaperConfig(memory=1.0)
    with pytest.raises(ValueError):
        opt.SignalShaperConfig(power=0.5)
    with pytest.raises(ValueError):
        opt.SignalShaperConfig(ind_neutral_weight=1.5)
    with pytest.raises(ValueError):
        opt.DynamicRiskGuardConfig(lookback_days=0)
    with pytest.raises(ValueError):
        opt.DynamicRiskGuardConfig(power_scale=-0.1)
    with pytest.raises(ValueError):
        opt.ConvexOptimizerConfig(max_single_weight=0.0)
    with pytest.raises(ValueError):
        opt.ConvexOptimizerConfig(benchmark_weight_mode="unknown")
    with pytest.raises(ValueError):
        opt.ConvexOptimizerConfig(solver_priority=())
    with pytest.raises(ValueError):
        opt.ConvexOptimizerConfig(proxy_rank_range=(100, 10))
    with pytest.raises(ValueError):
        opt.BarraPortfolioOptimizer(opt.BarraOptimizerConfig()).build_weights(
            _make_panel(n_days=3, n_symbols=6),
            np.zeros((3, 6)),
            reserve_cash=1.5,
        )


@pytest.mark.skipif(not (_HAS_CVXPY and _HAS_SOLVER), reason="cvxpy 或求解器不可用")
def test_convex_optimizer_level2_slack_fallback() -> None:
    """硬约束不可行但可松弛时进入 Level2，并记录松弛总量。"""
    n_symbols = 30
    data = _make_cross_section(n_symbols)
    cfg = opt.ConvexOptimizerConfig(
        cash_buffer_max=0.01,
        max_single_weight=0.1,
        max_active_weight=0.2,
        max_daily_turnover=0.5,
        style_deviation_default=0.5,
        style_deviation_overrides={},
        industry_deviation=0.0,
        allow_slack_fallback=True,
        solver_priority=("CLARABEL", "SCS", "OSQP"),
    )
    result = opt.ConvexOptimizer(cfg).solve(
        data["alpha"],
        w_prev=data["prev"],
        wb=data["wb"],
        industry=data["industry"],
        cash_buffer=0.01,
    )
    assert result.fallback_level == 2
    assert result.diagnostics.get("slack_used", 0.0) > 0.0
    assert result.weights.sum() == pytest.approx(0.99, abs=1e-6)
