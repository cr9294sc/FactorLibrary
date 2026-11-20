"""Phase 3 P1-1/P1-2 集成测试：`StocksProfile` 的 Barra 凸优化组合构建与双费率/漂移闭环接线。

覆盖：
1. `configs/stocks/{backtest,barra}.yaml` 的嵌套优化器配置解析与拍平；
2. `StocksProfile.build_weights` 在 `mode='quantile'` 与 `mode='barra_qp'` 下的分发；
3. `barra_qp` 端到端约束（满仓/纯多头/单票上限/换手上限）与 Fortran 布局；
4. `reserve_cash` 现金预留归一化与 `PositionDriftTracker` 漂移闭环；
5. 双费率模式（`fee_mode` 切换）在执行层的端到端差异。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from factorlib.core.config import (
    StockBacktestConfig,
    StockExecutionConfig,
    StockFeesConfig,
    StockPortfolioConfig,
    load_stock_backtest_config,
    load_stock_barra_config,
    load_stock_fees_config,
)
from factorlib.core.panel import Panel
from factorlib.markets.stocks.backtest import run_stock_backtest_pipeline
from factorlib.markets.stocks.optimizer import PositionDriftTracker, drift_positions
from factorlib.markets.stocks.profile import StocksProfile, normalize_optimizer_config

ROOT = Path(__file__).resolve().parents[2]


def _make_barra_panel(T: int = 24, N: int = 40, seed: int = 11) -> Panel:
    rng = np.random.default_rng(seed)
    ret = rng.normal(0.0005, 0.02, size=(T, N))
    close = np.cumprod(1.0 + ret, axis=0) * 10.0
    fields: dict[str, np.ndarray] = {
        "c2c_ret": np.asfortranarray(ret),
        "close_hfq": np.asfortranarray(close),
        "open_adj": np.asfortranarray(close * 0.998),
        "mkt_cap": np.asfortranarray(np.abs(rng.lognormal(mean=24.0, sigma=1.0, size=(T, N)))),
        "barra_industry": np.asfortranarray(rng.integers(1, 5, size=(T, N)).astype(np.float64)),
        "in_universe_mask": np.asfortranarray(np.ones((T, N))),
        "limit_mask": np.asfortranarray(np.ones((T, N))),
    }
    for name in (
        "beta",
        "btop",
        "earnyild",
        "growth",
        "leverage",
        "liquidty",
        "momentum",
        "resvol",
        "size",
        "sizenl",
    ):
        fields[f"barra_{name}"] = np.asfortranarray(rng.normal(0.0, 1.0, size=(T, N)))
    mask = np.ones((T, N), dtype=bool, order="F")
    labels = {"y_1": np.asfortranarray(rng.normal(0.0, 0.02, size=(T, N)))}
    return Panel(
        dates=np.array([f"2024{1 + i // 28:02d}{1 + i % 28:02d}" for i in range(T)]),
        symbols=[f"S{j:03d}" for j in range(N)],
        fields=fields,
        mask=mask,
        labels=labels,
        aux_fields={"is_suspended": np.asfortranarray(np.zeros((T, N)))},
        meta={"market": "stocks", "capabilities": {"has_barra_exposures": True}},
    )


@pytest.fixture()
def profile_no_risk_store(monkeypatch: pytest.MonkeyPatch) -> StocksProfile:
    """构造不依赖磁盘风险库的 profile（保证单测hermetic）。"""
    prof = StocksProfile(root=ROOT)
    monkeypatch.setattr(prof, "load_risk_model", lambda *a, **k: None)
    from factorlib.core.config import StockBarraConfig

    monkeypatch.setattr(prof, "barra_config", lambda: StockBarraConfig(risk_store_dir=None))
    return prof


# ------------------------------------------------------------------------------
# 1. 配置解析
# ------------------------------------------------------------------------------


def test_backtest_yaml_optimizer_section_flattened() -> None:
    bt = load_stock_backtest_config(ROOT / "configs" / "stocks" / "backtest.yaml")
    assert bt.portfolio.mode in ("quantile", "barra_qp")
    assert "objective" in bt.portfolio.optimizer
    assert "solver" in bt.portfolio.optimizer
    assert bt.strategy["power"] == pytest.approx(1.8)

    flat = normalize_optimizer_config(bt.portfolio.optimizer, bt)
    assert flat["risk_aversion"] == pytest.approx(2.5)
    assert flat["turnover_penalty"] == pytest.approx(0.0015)
    assert flat["max_single_weight"] == pytest.approx(0.03)
    assert flat["style_deviation_overrides"] == {"SIZE": 0.15, "BETA": 0.20}
    assert flat["solver_priority"] == ["MOSEK", "CLARABEL", "OSQP"]
    assert flat["horizon_mode"] is True
    assert flat["asymmetric_trade_penalty"] == (1.0, 0.45)
    assert flat["shaper"]["ind_neutral_weight"] == pytest.approx(0.42)
    assert flat["shaper"]["power"] == pytest.approx(1.8)
    assert flat["risk_guard"]["enabled"] is True
    assert flat["risk_guard"]["lookback_days"] == 10


def test_barra_yaml_contract() -> None:
    cfg = load_stock_barra_config(ROOT / "configs" / "stocks" / "barra.yaml", root=ROOT)
    assert cfg.risk_dataset == "lb_barra_v1"
    assert cfg.return_clock == "vwap_open_30m"
    assert len(cfg.style_factors) == 10
    assert cfg.industry_field == "barra_industry"
    assert cfg.min_eigenvalue == pytest.approx(1e-10)
    assert cfg.idiosyncratic_mode == "auto"


def test_portfolio_mode_validation() -> None:
    with pytest.raises(ValueError):
        StockPortfolioConfig(mode="invalid")
    with pytest.raises(ValueError):
        StockPortfolioConfig(mode="barra_qp", optimizer={})


# ------------------------------------------------------------------------------
# 2. build_weights 分发与 barra_qp 约束
# ------------------------------------------------------------------------------


def test_build_weights_quantile_mode_unchanged(profile_no_risk_store: StocksProfile) -> None:
    panel = _make_barra_panel()
    rng = np.random.default_rng(3)
    scores = rng.normal(size=panel.shape)
    cfg = StockPortfolioConfig(mode="quantile", top_quantile=0.2, max_single_weight=0.1)
    w = profile_no_risk_store.build_weights(
        scores=scores, returns=np.zeros_like(scores), mask=panel.mask, config=cfg, panel=panel
    )
    assert w.shape == panel.shape
    assert w.flags.f_contiguous
    assert np.all(w >= -1e-12)
    row_sums = w.sum(axis=1)
    # 分位等权基线在单票上限约束下不强制满仓（Top20% × 上限 0.1 = 0.8）
    assert np.all(row_sums > 0.0)
    assert np.all(row_sums <= 1.0 + 1e-6)


def test_build_weights_barra_qp_numpy_mode(profile_no_risk_store: StocksProfile) -> None:
    panel = _make_barra_panel()
    rng = np.random.default_rng(5)
    scores = rng.normal(size=panel.shape)
    bt = load_stock_backtest_config(ROOT / "configs" / "stocks" / "backtest.yaml")
    cfg = StockPortfolioConfig(
        mode="barra_qp",
        top_quantile=bt.portfolio.top_quantile,
        max_single_weight=bt.portfolio.max_single_weight,
        optimizer=bt.portfolio.optimizer,
    )
    w = profile_no_risk_store.build_weights(
        scores=scores,
        returns=np.zeros_like(scores),
        mask=panel.mask,
        config=cfg,
        panel=panel,
        force_numpy_optimizer=True,
    )
    assert w.shape == panel.shape
    assert w.flags.f_contiguous
    assert np.all(np.isfinite(w))
    assert np.all(w >= -1e-12)
    cap = float(cfg.optimizer["constraints"]["max_single_weight"])
    assert np.all(w <= cap + 1e-9)
    assert np.all(w.sum(axis=1) <= 1.0 + 1e-9)


def test_build_weights_barra_qp_requires_panel(profile_no_risk_store: StocksProfile) -> None:
    cfg = StockPortfolioConfig(mode="barra_qp", optimizer={"objective": {"risk_aversion": 1.0}})
    with pytest.raises(ValueError):
        profile_no_risk_store.build_weights(
            scores=np.zeros((3, 4)),
            returns=np.zeros((3, 4)),
            mask=np.ones((3, 4), dtype=bool),
            config=cfg,
            panel=None,
        )


# ------------------------------------------------------------------------------
# 3. 漂移闭环
# ------------------------------------------------------------------------------


def test_drift_positions_reserve_cash_normalization() -> None:
    prev = np.array([0.5, 0.3, 0.2])
    exec_p = np.array([10.0, 20.0, 5.0])
    close_p = np.array([11.0, 18.0, 5.0])
    drifted = drift_positions(prev, exec_p, close_p, reserve_cash=0.9)
    raw = prev * (close_p / exec_p)
    expected = 0.9 * raw / raw.sum()
    np.testing.assert_allclose(drifted, expected, rtol=1e-12)
    assert drifted.sum() == pytest.approx(0.9)

    # 价格缺失/非正的标的不参与漂移
    close_bad = np.array([np.nan, 18.0, 5.0])
    drifted_bad = drift_positions(prev, exec_p, close_bad, reserve_cash=1.0)
    assert np.isfinite(drifted_bad).all()
    assert drifted_bad.sum() == pytest.approx(1.0)


def test_position_drift_tracker_last_hold() -> None:
    tracker = PositionDriftTracker(reserve_cash=0.9)
    w0 = np.array([0.4, 0.35, 0.25])
    tracker.update(w0, np.array([10.0, 20.0, 5.0]), np.array([10.0, 20.0, 5.0]))
    # 首日价格无变动，仅做 reserve_cash 归一化（0.9 × w0）
    np.testing.assert_allclose(tracker.last_hold, 0.9 * w0, rtol=1e-12)
    assert tracker.last_hold.sum() == pytest.approx(0.9)

    tracker.update(
        np.array([0.5, 0.3, 0.2]),
        np.array([11.0, 19.0, 5.5]),
        np.array([12.0, 18.0, 5.0]),
    )
    assert tracker.last_hold.sum() == pytest.approx(0.9)
    assert np.all(tracker.last_hold >= 0.0)


# ------------------------------------------------------------------------------
# 4. 双费率模式执行层
# ------------------------------------------------------------------------------


def _fees(fee_mode: str, fee_rate: float = 0.003) -> StockFeesConfig:
    return StockFeesConfig(fee_mode=fee_mode, fee_rate=fee_rate)


def _exec_cfg() -> StockBacktestConfig:
    return StockBacktestConfig(execution=StockExecutionConfig(lot_size=100, reserve_cash=0.9))


def test_fee_mode_config_contract() -> None:
    asym = load_stock_fees_config(ROOT / "configs" / "stocks" / "fees.yaml")
    assert asym.fee_mode == "asymmetric"
    assert asym.effective_flat_fee_rate == 0.0
    flat = _fees("flat", 0.003)
    assert flat.effective_flat_fee_rate == pytest.approx(0.003)

    with pytest.raises(ValueError):
        StockFeesConfig(fee_mode="invalid")
    with pytest.raises(ValueError):
        StockFeesConfig(fee_mode="flat", fee_rate=0.0)


def test_backtest_dual_fee_mode_divergence() -> None:
    T, N = 4, 3
    weights = np.zeros((T, N), dtype=np.float64, order="F")
    weights[:, 0] = 0.5
    weights[:, 1] = 0.3
    weights[:, 2] = 0.2
    rng = np.random.default_rng(9)
    exec_p = np.asfortranarray(10.0 + rng.normal(0.0, 0.2, size=(T, N)))
    close_p = np.asfortranarray(exec_p * 1.001)
    # 第 1 日空仓 → 第 1 日权重（row0）不触发交易；第 2 日买入（row1），
    # 第 3 日清仓（row2=0）以产生卖出印花税
    weights[0, :] = 0.0
    weights[2, :] = 0.0
    weights[3, :] = 0.0
    fields = {
        "open_adj": exec_p,
        "close_hfq": close_p,
        "is_suspended": np.asfortranarray(np.zeros((T, N))),
        "can_buy_mask": np.asfortranarray(np.ones((T, N))),
        "can_sell_mask": np.asfortranarray(np.ones((T, N))),
    }
    panel = Panel(
        dates=np.array([f"2024010{i}" for i in range(1, T + 1)]),
        symbols=["A", "B", "C"],
        fields=fields,
        mask=np.ones((T, N), dtype=bool, order="F"),
        labels={},
        aux_fields={
            "is_suspended": fields["is_suspended"],
            "can_buy_mask": fields["can_buy_mask"],
            "can_sell_mask": fields["can_sell_mask"],
        },
        meta={"market": "stocks", "capabilities": {"has_directional_limit": True}},
    )

    res_asym = run_stock_backtest_pipeline(
        panel=panel, weights=weights, config=_exec_cfg(), fees=_fees("asymmetric")
    )
    res_flat = run_stock_backtest_pipeline(
        panel=panel, weights=weights, config=_exec_cfg(), fees=_fees("flat", 0.003)
    )
    assert res_asym.extras["fee_mode"] == "asymmetric"
    assert res_flat.extras["fee_mode"] == "flat"
    assert res_asym.extras["reserve_cash"] == pytest.approx(0.9)
    # flat 模式：印花税恒为 0，且费用严格对称
    assert float(np.sum(np.abs(res_flat.extras["tax"]))) == pytest.approx(0.0)
    assert float(res_asym.metrics["total_tax"]) > 0.0
    # flat 0.003 双边费率显著高于默认非对称税费（万 1 佣金 + 万 5 印花税）
    assert float(res_flat.metrics["total_fees"]) > float(res_asym.metrics["total_fees"])
    # 净值恒等式不受费率模式影响
    for res in (res_asym, res_flat):
        for d in range(1, T):
            assert abs(res.equity[d] - (res.equity[d - 1] + res.net_pnl[d])) < 1e-6
