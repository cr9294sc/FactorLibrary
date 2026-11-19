"""A 股持仓级回测、MarketProfile 抽象与组合构建单元测试 (tests/python/test_stock_backtest.py)。"""

from __future__ import annotations

from pathlib import Path

import factorlib_core
import numpy as np
import pytest
from factorlib.core.backtest import BacktestResult, run_backtest
from factorlib.core.config import StockDataConfig, StockPortfolioConfig
from factorlib.core.evaluate import evaluate_factor, run_factor_workflow
from factorlib.markets.profile import MarketProfile, PanelSpec
from factorlib.markets.registry import get_profile, list_profiles
from factorlib.markets.stocks.panel_loader import StockPanelData, build_stock_panel
from factorlib.markets.stocks.portfolio import build_stocks_portfolio_weights
from hypothesis import given, settings
from hypothesis import strategies as st

from tests.python.oracles.stock_ledger_ref import run_stock_ledger_ref
from tests.python.test_perturbation import _generate_synthetic_stock_raw


def _make_synthetic_stock_panel_data(T: int = 40, N: int = 12, seed: int = 7) -> StockPanelData:
    rng = np.random.default_rng(seed)
    dates = [f"202401{d + 1:02d}" if d < 31 else f"202402{d - 30:02d}" for d in range(T)]
    symbols = [f"0000{i + 1:02d}.XSHE" for i in range(N)]

    rets = rng.normal(0.0005, 0.02, size=(T, N))
    close_hfq = np.asfortranarray(10.0 * np.cumprod(1.0 + rets, axis=0))
    open_adj = np.asfortranarray(close_hfq * (1.0 + rng.normal(0.0, 0.005, size=(T, N))))
    high_adj = np.asfortranarray(np.maximum(open_adj, close_hfq) * 1.01)
    low_adj = np.asfortranarray(np.minimum(open_adj, close_hfq) * 0.99)
    vwap_adj = np.asfortranarray(0.5 * (open_adj + close_hfq))
    volume = np.asfortranarray(rng.uniform(1e5, 5e6, size=(T, N)))
    amount = np.asfortranarray(volume * close_hfq)
    log_amount = np.asfortranarray(np.log(amount))
    c2c_ret = np.asfortranarray(rets)
    ex_mat = np.ones((T, N), dtype=np.float64, order="F")

    base_univ_mask = np.ones((T, N), dtype=bool, order="F")
    limit_mask = np.ones((T, N), dtype=np.float64, order="F")
    can_buy_mask = np.ones((T, N), dtype=bool, order="F")
    can_sell_mask = np.ones((T, N), dtype=bool, order="F")
    is_susp = np.zeros((T, N), dtype=bool, order="F")
    is_st = np.zeros((T, N), dtype=bool, order="F")
    listed_days = np.asfortranarray(
        np.tile(np.arange(T, dtype=np.float64)[:, None] + 200.0, (1, N))
    )
    ind_code = np.asfortranarray(np.tile(np.array([1.0, 2.0, 3.0] * (N // 3))[:N], (T, 1)))
    bench = rng.normal(0.0003, 0.01, size=T)
    bench[-2:] = np.nan

    return StockPanelData(
        dates=dates,
        symbols=symbols,
        fields={
            "close": close_hfq.copy(),
            "open": open_adj.copy(),
            "high": high_adj.copy(),
            "low": low_adj.copy(),
            "close_hfq": close_hfq,
            "open_adj": open_adj,
            "high_adj": high_adj,
            "low_adj": low_adj,
            "vwap_adj": vwap_adj,
            "amount": amount,
            "volume": volume,
            "log_amount": log_amount,
            "c2c_ret": c2c_ret,
            "ex_cum_factor": ex_mat,
        },
        base_univ_mask=base_univ_mask,
        limit_mask=limit_mask,
        is_suspended=is_susp,
        is_st=is_st,
        stock_listed_days=listed_days,
        industry_code=ind_code,
        bench_fwd_open_1=bench,
        bench_fwd_close_1=bench,
        bench_ret=bench,
        can_buy_mask=can_buy_mask,
        can_sell_mask=can_sell_mask,
    )


def test_market_profile_registry_and_contracts() -> None:
    """验证 MarketProfile 注册中心与期货/股票双市场协议契约。"""
    assert set(list_profiles()) >= {"futures", "stocks"}
    f_prof = get_profile("futures")
    s_prof = get_profile("stocks")
    assert isinstance(f_prof, MarketProfile)
    assert isinstance(s_prof, MarketProfile)

    f_rules = f_prof.execution_rules()
    s_rules = s_prof.execution_rules()
    assert f_rules.t_plus_one is False and f_rules.allow_short is True and f_rules.lot_size == 1
    assert s_rules.t_plus_one is True and s_rules.allow_short is False and s_rules.lot_size == 100

    s_cost = s_prof.cost_model()
    assert s_cost.min_commission >= 0.0
    assert s_cost.stamp_duty_rate > 0.0


def test_stock_backtest_limit_blocking_suspension_and_odd_lot_sell() -> None:
    """验证涨跌停阻断、停牌按最后有效价盯市、买入整手100股与清仓零股语义。"""
    T, N = 5, 2
    weights = np.zeros((T, N), dtype=np.float64, order="F")
    weights[0, 0] = 0.50  # Day 1 买入 stock 0
    weights[0, 1] = 0.40  # Day 1 尝试买入 stock 1 (一字涨停阻断)
    weights[1, 0] = 0.00  # Day 2 清仓 stock 0
    weights[1, 1] = 0.00

    exec_p = np.array(
        [
            [10.0, 10.0],
            [10.0, 11.0],
            [12.0, 11.0],
            [12.0, 11.0],
            [12.0, 11.0],
        ],
        dtype=np.float64,
        order="F",
    )
    close_p = np.array(
        [
            [10.0, 10.0],
            [11.0, 11.0],
            [12.0, 11.0],
            [99.0, 11.0],  # Day 3 stock 0 停牌，脏收盘价不得污染持仓盯市
            [12.0, 11.0],
        ],
        dtype=np.float64,
        order="F",
    )
    can_buy = np.ones((T, N), dtype=bool, order="F")
    can_buy[1, 1] = False  # Day 1 stock 1 涨停不可买
    can_sell = np.ones((T, N), dtype=bool, order="F")
    is_susp = np.zeros((T, N), dtype=bool, order="F")
    is_susp[3, 0] = True

    res = factorlib_core.run_stock_backtest(
        weights,
        exec_p,
        close_p,
        can_buy,
        can_sell,
        is_susp,
        init_cash=100_000.0,
        lot_size=100,
        commission_rate=0.0001,
        min_commission=5.0,
        stamp_duty_rate=0.0005,
        transfer_fee_rate=0.00001,
        slip_ticks=0.0,
        tick_size=0.01,
        slip_bps=0.0,
    )
    ref = run_stock_ledger_ref(
        weights,
        exec_p,
        close_p,
        can_buy,
        can_sell,
        is_susp,
        init_cash=100_000.0,
        lot_size=100,
        commission_rate=0.0001,
        min_commission=5.0,
        stamp_duty_rate=0.0005,
        transfer_fee_rate=0.00001,
        slip_ticks=0.0,
        tick_size=0.01,
        slip_bps=0.0,
    )

    np.testing.assert_allclose(res["equity"], ref["equity"], atol=1e-9, rtol=0.0)
    np.testing.assert_array_equal(res["positions"], ref["positions"])
    np.testing.assert_array_equal(res["blocked_orders"], ref["blocked_orders"])
    assert res["positions"][1, 0] == 5000
    assert res["positions"][1, 1] == 0
    assert res["blocked_orders"][1] == 1
    assert res["positions"][2, 0] == 0
    assert res["tax"][2] > 0.0

    # 检验 allow_short=True 显式拒绝
    with pytest.raises(ValueError, match="暂未实现"):
        factorlib_core.run_stock_backtest(
            weights,
            exec_p,
            close_p,
            can_buy,
            can_sell,
            is_susp,
            allow_short=True,
        )


@settings(max_examples=25, deadline=None)
@given(
    seed=st.integers(min_value=1, max_value=10_000),
    slip_bps=st.sampled_from([0.0, 2.5]),
)
def test_stock_backtest_parity_with_python_oracle(seed: int, slip_bps: float) -> None:
    """随机权重/价格/停牌/涨跌停场景下，Rust 内核与纯 Python 逐日账本 |Δequity| < 1e-9。"""
    rng = np.random.default_rng(seed)
    T, N = 15, 5

    weights = np.asfortranarray(rng.uniform(-0.1, 0.4, size=(T, N)), dtype=np.float64)
    exec_p = np.asfortranarray(rng.uniform(5.0, 80.0, size=(T, N)), dtype=np.float64)
    close_p = np.asfortranarray(exec_p * rng.uniform(0.95, 1.05, size=(T, N)), dtype=np.float64)
    is_susp = np.asfortranarray(rng.random((T, N)) < 0.10, dtype=bool)
    can_buy = np.asfortranarray((rng.random((T, N)) > 0.10) & (~is_susp), dtype=bool)
    can_sell = np.asfortranarray((rng.random((T, N)) > 0.10) & (~is_susp), dtype=bool)

    res = factorlib_core.run_stock_backtest(
        weights,
        exec_p,
        close_p,
        can_buy,
        can_sell,
        is_susp,
        init_cash=1_000_000.0,
        slip_bps=slip_bps,
    )
    ref = run_stock_ledger_ref(
        weights,
        exec_p,
        close_p,
        can_buy,
        can_sell,
        is_susp,
        init_cash=1_000_000.0,
        slip_bps=slip_bps,
    )

    np.testing.assert_allclose(res["equity"], ref["equity"], atol=1e-9, rtol=0.0)
    np.testing.assert_allclose(res["cash"], ref["cash"], atol=1e-9, rtol=0.0)
    np.testing.assert_allclose(res["tax"], ref["tax"], atol=1e-9, rtol=0.0)
    np.testing.assert_allclose(res["fee"], ref["fee"], atol=1e-9, rtol=0.0)
    np.testing.assert_allclose(res["slippage"], ref["slippage"], atol=1e-9, rtol=0.0)
    np.testing.assert_array_equal(res["positions"], ref["positions"])
    np.testing.assert_array_equal(res["blocked_orders"], ref["blocked_orders"])


def test_stocks_portfolio_and_dual_profile_workflow() -> None:
    """验证股票分位等权组合构建（含行业中性）与 StocksProfile 端到端评价回测工作流。"""
    sdata = _make_synthetic_stock_panel_data(T=40, N=12, seed=11)
    panel = sdata.to_panel()

    assert panel.meta["market"] == "stocks"
    assert {"y_1", "y_2", "y_5", "y_10"}.issubset(panel.labels.keys())
    assert "bench_fwd_open_1" in panel.aux_fields
    assert "bench_fwd_open_1" not in panel.fields

    # 1. 测试分位等权与行业中性组合构建（不使用全协方差）
    scores = np.asfortranarray(panel.fields["c2c_ret"], dtype=np.float64)
    scfg = StockPortfolioConfig(top_quantile=0.25, industry_neutral=True, max_single_weight=0.25)
    w = build_stocks_portfolio_weights(
        scores=scores,
        returns=panel.fields["c2c_ret"],
        mask=panel.valid_mask,
        cfg=scfg,
        sector=panel.fields["sector"],
    )
    assert np.all(w >= 0.0)
    assert np.all(w <= 0.25 + 1e-12)
    assert np.all(np.sum(w, axis=1) <= 1.0 + 1e-12)

    # 2. 通过 core.backtest.run_backtest 分发到 StocksProfile
    s_prof = get_profile("stocks")
    bt_res = run_backtest(panel, w, profile=s_prof)
    assert isinstance(bt_res, BacktestResult)
    assert np.all(bt_res.margin_used == 0.0)
    assert np.all(bt_res.roll_cost == 0.0)
    assert "tax" in bt_res.extras and "blocked_orders" in bt_res.extras
    df = bt_res.to_dataframe()
    assert "tax" in df.columns and "blocked_orders" in df.columns

    # 3. evaluate_factor 与 run_factor_workflow 在 StocksProfile 上完整跑通
    eval_res = evaluate_factor(
        panel,
        "cs_zscore(c2c_ret)",
        horizons=(1, 5),
        postprocess=s_prof.default_postprocess(),
        profile=s_prof,
    )
    assert len(eval_res.all_metrics) > 0

    wf = run_factor_workflow(
        panel=panel,
        definition="cs_zscore(c2c_ret)",
        postprocess=s_prof.default_postprocess(),
        materialize=False,
        lock_version=False,
        profile=s_prof,
    )
    assert wf.backtest_result.metrics["total_return"] is not None
    assert wf.v4_stats["style_exposure"]["styles"]


def test_build_stock_panel_and_profile_load_derived(tmp_path: Path) -> None:
    """验证 M1.4 build_stock_panel 落盘与 StocksProfile.load_panel 加载闭环。"""
    from datetime import date

    dates = [date(2024, 1, i) for i in range(2, 10)]
    raw_dir = tmp_path / "raw_stocks"
    derived_dir = tmp_path / "derived_stocks"
    _generate_synthetic_stock_raw(raw_dir, dates, perturb_from=None)

    cfg = StockDataConfig(
        raw_dir=raw_dir,
        derived_dir=derived_dir,
        panel_version="v1",
        start_date=dates[0],
        end_date=dates[-1],
    )
    _pdf, _ldf, meta = build_stock_panel(cfg, show_progress=False)
    assert (cfg.panel_dir / "panel_daily.parquet").exists()
    assert (cfg.panel_dir / "labels.parquet").exists()
    assert (cfg.panel_dir / "meta.json").exists()
    assert meta["market"] == "stocks"

    prof = get_profile("stocks")
    panel = prof.load_panel(PanelSpec(version="v1", panel_dir=cfg.panel_dir))
    assert panel.meta["market"] == "stocks"
    assert "y_1" in panel.labels and "y_5" in panel.labels
    assert "bench_fwd_open_1" in panel.aux_fields
    assert "bench_fwd_open_1" not in panel.fields


def test_stock_backtest_vwap_open_30m_anti_lookahead() -> None:
    """验证 entry_window='vwap_open_30m' 必须使用 vwap_open_30m_adj，严禁静默回退全天 vwap_adj。"""
    from factorlib.core.config import StockBacktestConfig, StockExecutionConfig
    from factorlib.markets.stocks.backtest import (
        _resolve_exec_and_close_prices,
        run_stock_backtest_pipeline,
    )

    sdata = _make_synthetic_stock_panel_data(T=10, N=6, seed=11)
    panel_no_30m = sdata.to_panel(horizons=(1,))
    assert "vwap_adj" in panel_no_30m.fields
    assert not panel_no_30m.has_any_field("vwap_open_30m_adj")

    # 1. hfq 口径缺少 vwap_open_30m_adj 时禁止静默回退到全天 vwap_adj
    with pytest.raises(ValueError, match="vwap_open_30m_adj"):
        _resolve_exec_and_close_prices(
            panel_no_30m, price_basis="hfq", entry_window="vwap_open_30m"
        )

    # 2. raw 口径缺少 vwap_open_30m_raw / exec_vwap 时禁止静默回退
    with pytest.raises(ValueError, match="vwap_open_30m_raw"):
        _resolve_exec_and_close_prices(
            panel_no_30m, price_basis="raw", entry_window="vwap_open_30m"
        )

    # 3. 显式提供 vwap_open_30m_adj 时正确选取，且标注 bilateral_conservative
    vwap30 = panel_no_30m.fields["open_adj"] * 1.005
    panel_no_30m.fields["vwap_open_30m_adj"] = np.asfortranarray(vwap30)
    panel_no_30m.meta["capabilities"] = {"has_directional_limit": False}

    exec_p, close_p = _resolve_exec_and_close_prices(
        panel_no_30m, price_basis="hfq", entry_window="vwap_open_30m"
    )
    np.testing.assert_allclose(exec_p, vwap30)
    np.testing.assert_allclose(close_p, panel_no_30m.fields["close_hfq"])

    weights = np.full(panel_no_30m.shape, 0.25, dtype=np.float64, order="F")
    bt_cfg = StockBacktestConfig(
        execution=StockExecutionConfig(entry_window="vwap_open_30m", price_basis="hfq")
    )
    res = run_stock_backtest_pipeline(panel_no_30m, weights, config=bt_cfg)
    assert res.extras["limit_mask_mode"] == "bilateral_conservative"
    assert res.extras["entry_window"] == "vwap_open_30m"
