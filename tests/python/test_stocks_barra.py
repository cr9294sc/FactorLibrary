"""Phase 2 Barra 风险模型内核、SignalBridge/RiskView、自适应风格与 ML/DL 特质标签测试
(tests/python/test_stocks_barra.py)。
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from factorlib.core.config import (
    LightGBMConfig,
    MLConfig,
    StockDataConfig,
    StockUniverseFilterConfig,
    WalkForwardConfig,
)
from factorlib.core.panel import Panel
from factorlib.core.probe import apply_postprocess_pipeline, probe_factor
from factorlib.dl.dataset import prepare_dl_panel_data
from factorlib.dl.trainer import DLTrainConfig, train_dl_walk_forward
from factorlib.markets.profile import PanelSpec
from factorlib.markets.stocks.ares_bridge import (
    BARRA_FACTOR_ORDER,
    BARRA_STYLE_FACTORS,
    build_lb_barra_panel,
)
from factorlib.markets.stocks.barra import (
    BarraRiskModelStore,
    compute_constrained_wls_idiosyncratic_returns,
    estimate_dynamic_specific_variance,
)
from factorlib.markets.stocks.bridge import RiskView, SignalBridge
from factorlib.markets.stocks.profile import StocksProfile
from factorlib.ml.dataset import MLDatasetBuilder, parse_label_horizon
from factorlib.ml.pipeline import WalkForwardPipeline

from tests.python.test_stocks_lb_barra_phase1 import _make_synthetic_lb_assets


def _make_synthetic_rq_panel(T: int = 45, N: int = 6, seed: int = 31) -> Panel:
    rng = np.random.default_rng(seed)
    dates = np.array(
        [
            date(2024, 1, 2) if i == 0 else date.fromordinal(date(2024, 1, 2).toordinal() + i)
            for i in range(T)
        ],
        dtype="datetime64[D]",
    )
    symbols = [
        "000001.XSHE",
        "000002.XSHE",
        "600000.XSHG",
        "600519.XSHG",
        "300001.XSHE",
        "000333.XSHE",
    ][:N]
    rets = rng.normal(0.0005, 0.015, size=(T, N))
    close_hfq = np.asfortranarray(10.0 * np.cumprod(1.0 + rets, axis=0))
    open_adj = np.asfortranarray(close_hfq * (1.0 + rng.normal(0.0, 0.003, size=(T, N))))
    amount = np.asfortranarray(rng.uniform(1e7, 5e8, size=(T, N)))
    volume = np.asfortranarray(amount / np.maximum(close_hfq, 1.0))
    log_amount = np.asfortranarray(np.log(amount))
    c2c_ret = np.asfortranarray(rets)
    mask = np.ones((T, N), dtype=bool, order="F")
    ind = np.asfortranarray(np.tile(np.arange(1.0, N + 1.0), (T, 1)))
    is_susp = np.zeros((T, N), dtype=np.float64, order="F")
    y_1 = np.full((T, N), np.nan, dtype=np.float64, order="F")
    y_1[:-2, :] = open_adj[2:, :] / open_adj[1:-1, :] - 1.0
    y_5 = np.full((T, N), np.nan, dtype=np.float64, order="F")
    y_5[:-6, :] = open_adj[6:, :] / open_adj[1:-5, :] - 1.0

    return Panel(
        dates=dates,
        symbols=symbols,
        fields={
            "close_hfq": close_hfq,
            "open_adj": open_adj,
            "amount": amount,
            "volume": volume,
            "log_amount": log_amount,
            "c2c_ret": c2c_ret,
            "industry_code": ind,
            "sector": ind,
        },
        mask=mask,
        labels={"y_1": y_1, "y_5": y_5},
        aux_fields={"is_suspended": is_susp},
        meta={
            "market": "stocks",
            "panel_version": "rq",
            "dataset_id": "rq",
            "source": "ricequant",
        },
    )


def test_constrained_wls_orthogonality_and_zero_sum_constraint() -> None:
    """验证因果 Constrained WLS 的加权正交性、行业零和约束与真实因子还原能力。"""
    rng = np.random.default_rng(2026)
    T, N = 15, 180
    k_styles = 10
    k_ind = 6
    lag = 2

    styles = rng.normal(0.0, 1.0, size=(T, N, k_styles))
    ind = np.tile(np.arange(1, k_ind + 1, dtype=np.float64), (T, N // k_ind))
    mkt_cap = rng.uniform(1e9, 1e11, size=(T, N))

    # 构造严格满足行业市值零和约束的真实因子收益 + 纯特质收益
    true_f_style = rng.normal(0.0, 0.008, size=(T, k_styles))
    true_f_country = rng.normal(0.0, 0.01, size=T)
    true_f_ind = np.zeros((T, k_ind), dtype=np.float64)
    true_u = rng.normal(0.0, 0.012, size=(T, N))
    returns = np.full((T, N), np.nan, dtype=np.float64)

    for t in range(lag, T):
        caps_lag = mkt_cap[t - lag]
        ind_caps = np.array([np.sum(caps_lag[ind[t - lag] == k]) for k in range(1, k_ind + 1)])
        raw_ind = rng.normal(0.0, 0.01, size=k_ind)
        # 投影消除加权和使其满足 sum(omega_k * f_ind_k) = 0
        raw_ind -= np.sum(ind_caps * raw_ind) / np.sum(ind_caps)
        true_f_ind[t] = raw_ind
        ind_pos = ind[t - lag].astype(int) - 1
        returns[t] = (
            styles[t - lag] @ true_f_style[t] + true_f_country[t] + raw_ind[ind_pos] + true_u[t]
        )

    res = compute_constrained_wls_idiosyncratic_returns(
        returns,
        styles,
        ind,
        mkt_cap,
        exposure_lag_days=lag,
        return_winsor_abs=0.50,  # 确保合成数据不触发缩尾，检验精确 KKT 正交性
        ridge_eps=0.0,
        min_obs=50,
        industry_codes=tuple(range(1, k_ind + 1)),
    )

    for t in range(lag, T):
        caps_lag = mkt_cap[t - lag]
        ind_caps = np.array([np.sum(caps_lag[ind[t - lag] == k]) for k in range(1, k_ind + 1)])
        omega = ind_caps / np.sum(ind_caps)
        f_ind_hat = res.factor_returns[t, k_styles + 1 :]
        # 1. 行业市值加权零和误差 < 1e-10
        assert abs(float(np.sum(omega * f_ind_hat))) < 1e-10

        # 2. X^T W u_t 无穷范数 < 1e-8
        w = np.sqrt(caps_lag)
        w = w / np.mean(w)
        u_t = res.spec_returns[t]
        ind_pos = ind[t - lag].astype(int) - 1
        X = np.zeros((N, k_styles + 1 + k_ind), dtype=np.float64)
        X[:, :k_styles] = styles[t - lag]
        X[:, k_styles] = 1.0
        X[np.arange(N), k_styles + 1 + ind_pos] = 1.0
        grad = X.T @ (w * u_t)
        assert float(np.max(np.abs(grad))) < 1e-8


def test_barra_risk_store_and_dynamic_specific_variance(tmp_path: Path) -> None:
    """验证 BarraRiskModelStore（PSD 保护/因果 asof）与动态特质方差 Δ_t（NW + 收缩 + VRA）。"""
    raw_dir = tmp_path / "local_assets"
    derived_dir = tmp_path / "data" / "derived" / "stocks"
    fx = _make_synthetic_lb_assets(raw_dir, T=70, seed=42)

    cfg = StockDataConfig(
        raw_dir=raw_dir,
        derived_dir=derived_dir,
        panel_version="barra",
        dataset_id="barra",
        source="barra",
        benchmark_csv=fx["bench_csv"],
        entry_window="vwap_open_30m",
        exec_price_mode="vwap_open_30m_adj",
        start_date=fx["dates"][0],
        end_date=fx["dates"][-1],
        universe_filters=StockUniverseFilterConfig(
            use_source_base_univ_mask=True, exclude_bjse=True
        ),
        include_idiosyncratic_labels=True,
    )
    build_lb_barra_panel(cfg, pack_cov=True, show_progress=False)

    prof = StocksProfile(root=tmp_path)
    store = prof.load_risk_model("barra")
    assert isinstance(store, BarraRiskModelStore)
    assert len(store.factors) == len(BARRA_FACTOR_ORDER)

    # 验证 COV 平方根重构与严格因果 asof 回退（早于首日返回 None）
    earliest_cov = store.cov_dates[0]
    assert store.get_factor_cov("2015-01-01", asof_fallback=True) is None
    F_t = store.get_factor_cov(earliest_cov, min_eigenvalue=1e-10)
    L_t = store.get_factor_cov_sqrt(earliest_cov, min_eigenvalue=1e-10)
    assert F_t is not None and L_t is not None
    evals = np.linalg.eigvalsh(F_t)
    assert float(np.min(evals)) >= 1e-10 - 1e-15
    np.testing.assert_allclose(L_t @ L_t.T, F_t, atol=1e-12)

    # 验证动态特质方差 Δ_t 与落盘往返
    rng = np.random.default_rng(7)
    T_sv, N_sv = 80, 30
    u_sv = rng.normal(0.0, 0.02, size=(T_sv, N_sv))
    u_sv[:60, 0] = np.nan  # 模拟新股/停牌复牌仅少数近期观测
    caps_sv = rng.uniform(1e9, 1e11, size=(T_sv, N_sv))

    sv_nw = estimate_dynamic_specific_variance(
        u_sv,
        caps_sv,
        window_days=60,
        half_life_days=21.0,
        newey_west_lags=2,
        min_obs=15,
        shrinkage_enabled=False,
        vra_enabled=False,
    )
    sv_full = estimate_dynamic_specific_variance(
        u_sv,
        caps_sv,
        window_days=60,
        half_life_days=21.0,
        newey_west_lags=2,
        min_obs=15,
        shrinkage_enabled=True,
        mkt_cap_deciles=5,
        prior_strength=20.0,
        vra_enabled=True,
        vra_half_life_days=10.0,
    )
    # 低观测新股在未收缩前 <15 日为 NaN，经市值分层收缩后在最后一日获得稳定先验估计
    assert np.isnan(sv_nw[70, 0])
    assert np.isfinite(sv_full[70, 0]) and sv_full[70, 0] > 0.0

    sv_dates = [date.fromordinal(date(2024, 1, 2).toordinal() + i) for i in range(T_sv)]
    sv_syms = [f"{i + 1:06d}" for i in range(N_sv)]
    out_pq = store.save_specific_variance(sv_dates, sv_syms, sv_full)
    assert out_pq.exists()
    reloaded = store.load_specific_variance(sv_dates, sv_syms)
    assert reloaded is not None
    np.testing.assert_allclose(reloaded, sv_full, equal_nan=True)


def test_signal_bridge_alignment_and_parquet_shift(tmp_path: Path) -> None:
    """验证 SignalBridge 代码归一、覆盖率告警与 T_exec 前移 1 日 IC 还原。"""
    raw_dir = tmp_path / "local_assets"
    derived_dir = tmp_path / "data" / "derived" / "stocks"
    fx = _make_synthetic_lb_assets(raw_dir, T=45, seed=19)
    cfg = StockDataConfig(
        raw_dir=raw_dir,
        derived_dir=derived_dir,
        panel_version="barra",
        dataset_id="barra",
        source="barra",
        benchmark_csv=fx["bench_csv"],
        entry_window="vwap_open_30m",
        exec_price_mode="vwap_open_30m_adj",
        start_date=fx["dates"][0],
        end_date=fx["dates"][-1],
        universe_filters=StockUniverseFilterConfig(
            use_source_base_univ_mask=True, exclude_bjse=False
        ),
    )
    build_lb_barra_panel(cfg, pack_cov=False, show_progress=False)

    prof = StocksProfile(root=tmp_path)
    lb_panel = prof.load_panel(PanelSpec(version="barra", panel_dir=cfg.panel_dir))
    rq_panel = _make_synthetic_rq_panel(T=45, N=4, seed=21)

    # 1. T_close 桥接 + 低匹配率告警
    rng = np.random.default_rng(5)
    rq_scores = rng.normal(0.0, 1.0, size=rq_panel.shape)
    proj, report = prof.bridge_signal(
        rq_scores,
        rq_panel,
        lb_panel,
        input_signal_alignment="T_close",
        min_symbol_match_ratio=0.90,
    )
    assert proj.shape == lb_panel.shape
    assert report.symbol_match_count == 4
    assert report.symbol_match_ratio == pytest.approx(4 / 5)
    assert report.warnings  # 80% < 90% 触发覆盖率告警

    # 2. T_exec 外部 shifted_signal.parquet：信号行索引为执行日 T_exec = T_close + 1
    y1 = lb_panel.labels["y_1"]
    shifted_rows = np.full(lb_panel.shape, np.nan, dtype=np.float64)
    # 在执行日 t+1 存放与 y1[t] 正相关的信号
    shifted_rows[1:, :] = y1[:-1, :] * 10.0 + rng.normal(
        0.0, 0.01, size=(lb_panel.T - 1, lb_panel.N)
    )
    pq_path = tmp_path / "shifted_signal.parquet"
    pq_tbl = pa.table(
        {sym: shifted_rows[:, j] for j, sym in enumerate(lb_panel.symbols)}
        | {"__index_level_0__": [int(str(d).replace("-", "")) for d in lb_panel.dates]}
    )
    pq.write_table(pq_tbl, pq_path)

    aligned, rep_exec = SignalBridge.from_parquet(
        pq_path, lb_panel, input_signal_alignment="T_exec", min_symbol_match_ratio=0.90
    )
    assert rep_exec.symbol_match_ratio == 1.0
    assert not rep_exec.warnings
    ok = lb_panel.mask & np.isfinite(aligned) & np.isfinite(y1)
    assert float(np.corrcoef(aligned[ok], y1[ok])[0, 1]) > 0.90


def test_risk_view_adaptive_styles_and_ml_dl_y_idio(tmp_path: Path) -> None:
    """验证 RiskView 只读锁、10 维 Barra 风格自适应及 ML/DL y_idio_h 打通。"""
    raw_dir = tmp_path / "local_assets"
    derived_dir = tmp_path / "data" / "derived" / "stocks"
    fx = _make_synthetic_lb_assets(raw_dir, T=55, seed=88)
    cfg = StockDataConfig(
        raw_dir=raw_dir,
        derived_dir=derived_dir,
        panel_version="barra",
        dataset_id="barra",
        source="barra",
        benchmark_csv=fx["bench_csv"],
        entry_window="vwap_open_30m",
        exec_price_mode="vwap_open_30m_adj",
        start_date=fx["dates"][0],
        end_date=fx["dates"][-1],
        universe_filters=StockUniverseFilterConfig(
            use_source_base_univ_mask=True, exclude_bjse=False
        ),
        include_idiosyncratic_labels=True,
    )
    build_lb_barra_panel(cfg, pack_cov=False, show_progress=False)

    prof = StocksProfile(root=tmp_path)
    lb_panel = prof.load_panel(PanelSpec(version="barra", panel_dir=cfg.panel_dir))
    rq_panel = _make_synthetic_rq_panel(T=55, N=4, seed=89)

    # 未挂载 RiskView 前，rq_panel 仅回退 4 维近似风格，且缺少 y_idio_5
    styles_4 = prof.style_factors(rq_panel)
    assert set(styles_4.keys()) == {"SIZE", "LIQUIDITY", "VOLATILITY", "MOMENTUM"}
    with pytest.raises(KeyError, match="RiskView"):
        MLDatasetBuilder(rq_panel, {"f1": rq_panel.fields["c2c_ret"]}, target_label="y_idio_5")

    # 挂载只读 RiskView (含动态计算 rq 时钟 y_idio_1 / y_idio_5)
    attached = RiskView.attach(
        rq_panel,
        lb_panel,
        compute_dynamic_idio=True,
        dynamic_horizons=(1, 5),
    )
    assert "barra_beta" in attached["fields"]
    assert "y_idio_1" in rq_panel.labels and "y_idio_5" in rq_panel.labels
    assert rq_panel.fields["barra_beta"].flags.writeable is False
    assert rq_panel.labels["y_idio_5"].flags.writeable is False
    with pytest.raises(ValueError):
        rq_panel.fields["barra_beta"][0, 0] = 999.0

    # 挂载后 style_factors 自适应升级为 10 维 Barra CNE5 风格 + 兼容别名
    styles_10 = prof.style_factors(rq_panel)
    for s_name in BARRA_STYLE_FACTORS:
        assert s_name in styles_10
    assert "LIQUIDITY" in styles_10 and "VOLATILITY" in styles_10

    # probe_factor 与 apply_postprocess_pipeline 支持 riskNeutMulti:BETA,BTOP,SIZE,SIZENL
    p_res = probe_factor(
        {
            "name": "barra_neut_factor",
            "factor_type": "expression",
            "expression": "-ts_sum(spec_ret_vwap30, 5)",
            "postprocess": ["baseUniMask", "riskNeutMulti:BETA,BTOP,SIZE,SIZENL", "zScore"],
        },
        panel=lb_panel,
    )
    assert p_res["status"] == "ok", p_res["errors"]
    pp_mat = apply_postprocess_pipeline(
        -lb_panel.fields["spec_ret_vwap30"],
        ["baseUniMask", "riskNeutMulti:BETA,BTOP,SIZE,SIZENL", "zScore"],
        lb_panel,
        profile=prof,
    )
    assert pp_mat.shape == lb_panel.shape

    # 验证 parse_label_horizon 对 y_idio_h 与 ML/DL 训练流水线的打通
    assert parse_label_horizon("y_idio_1") == 1
    assert parse_label_horizon("y_idio_5") == 5
    assert parse_label_horizon("y_idio_10") == 10
    assert DLTrainConfig(target_label="y_idio_10").resolved()["horizon"] == 10

    # ML: 在挂载 y_idio_1 的 rq_panel 上运行 WalkForwardPipeline
    ml_cfg = MLConfig(
        target_label="y_idio_1",
        models=("equal_weight", "ridge"),
        walk_forward=WalkForwardConfig(
            start_predict_year=2024,
            end_predict_year=2024,
            train_window="expanding",
            purge_window=1,
            validation_size_days=5,
            min_train_years=1,
        ),
        lightgbm=LightGBMConfig(),
    )
    # 构造 2022~2024 跨年面板用于 ML Walk-Forward 冒烟 (train=2022, val=2023, test=2024)
    ml_panel = _make_synthetic_rq_panel(T=55, N=5, seed=71)
    ml_dates = np.array(
        [
            date(2022, 12, 1 + i)
            if i < 20
            else (date(2023, 12, i - 19) if i < 35 else date(2024, 1, i - 34))
            for i in range(55)
        ],
        dtype="datetime64[D]",
    )
    ml_panel.dates = ml_dates
    ml_panel.labels["y_idio_1"] = ml_panel.labels["y_1"].copy()
    ml_panel.labels["y_idio_1"].flags.writeable = False
    wf = WalkForwardPipeline(
        panel=ml_panel,
        config=ml_cfg,
        factor_matrices={"f1": ml_panel.fields["c2c_ret"], "f2": ml_panel.fields["log_amount"]},
    ).run(strict_min_years=False)
    assert wf.target_label == "y_idio_1"
    assert "ridge" in wf.predictions

    # DL: prepare_dl_panel_data + train_dl_walk_forward 支持 lb_barra_v1 与 y_idio_1
    dl_prep = prepare_dl_panel_data(
        lb_panel,
        feature_names=(),
        target_label="y_idio_1",
        market="stocks",
        min_names=3,
    )
    assert dl_prep.target_label == "y_idio_1"
    assert "vwap_open_30m_adj" in dl_prep.feature_names

    dl_cfg = DLTrainConfig(
        market="stocks",
        arch="gated_mlp",
        target_label="y_idio_1",
        seq_len=5,
        hidden_dim=8,
        n_layers=1,
        epochs=1,
        n_folds=2,
        n_seeds=1,
        purge_window=1,
        prefer_yearly_folds=False,
        device="cpu",
        min_names=3,
    )
    dl_res = train_dl_walk_forward(lb_panel, feature_names=dl_prep.feature_names[:4], config=dl_cfg)
    assert dl_res.oos_matrix.shape == lb_panel.shape

    # 卸载 RiskView 恢复原状
    detached = RiskView.detach(rq_panel)
    assert "barra_beta" in detached["fields"]
    assert "barra_beta" not in rq_panel.fields


def test_rq_auto_attaches_barra_risk_view_and_cli_backtest(tmp_path: Path) -> None:
    """验证加载 rq 时自动挂载 barra vwap_open_30m 标签与风格暴露，并打通 run_stocks_backtest.py。"""
    import json
    import sys

    import polars as pl

    from tests.python.test_stocks_lb_barra_phase1 import _make_synthetic_lb_assets

    raw_dir = tmp_path / "data" / "raw" / "stocks" / "barra"
    derived_dir = tmp_path / "data" / "derived" / "stocks"
    fx = _make_synthetic_lb_assets(raw_dir, T=45, seed=31)
    b_cfg = StockDataConfig(
        raw_dir=raw_dir,
        derived_dir=derived_dir,
        panel_version="barra",
        dataset_id="barra",
        source="barra",
        benchmark_csv=fx["bench_csv"],
        entry_window="vwap_open_30m",
        exec_price_mode="vwap_open_30m_adj",
        start_date=fx["dates"][0],
        end_date=fx["dates"][-1],
        universe_filters=StockUniverseFilterConfig(
            use_source_base_univ_mask=True, exclude_bjse=False
        ),
        include_idiosyncratic_labels=True,
    )
    build_lb_barra_panel(b_cfg, pack_cov=False, show_progress=False)

    # 构造落盘的 rq 面板（使用与 barra 重叠的日期与对应 .XSHE/.XSHG 代码）
    prof = StocksProfile(root=tmp_path)
    lb_panel = prof.load_panel(PanelSpec(version="barra"))
    T, N = lb_panel.shape
    rq_syms = [
        f"{s}.XSHG"
        if s.startswith("6")
        else (f"{s}.XSHE" if s.startswith(("0", "3")) else f"{s}.BJ")
        for s in lb_panel.symbols
    ]
    date_col = pl.Series(
        "trading_date",
        np.repeat(np.array(lb_panel.dates, dtype="datetime64[D]"), N),
        dtype=pl.Date,
    )
    sym_col = pl.Series("symbol", np.tile(np.array(rq_syms), T), dtype=pl.String)
    close_adj = lb_panel.fields["close_adj"]
    rq_dir = derived_dir / "rq"
    rq_dir.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(
        {
            "trading_date": date_col,
            "symbol": sym_col,
            "in_universe": lb_panel.mask.reshape(-1, order="C"),
            "close_hfq": close_adj.reshape(-1, order="C"),
            "open_adj": close_adj.reshape(-1, order="C"),
            "c2c_ret": lb_panel.fields["c2c_ret"].reshape(-1, order="C"),
            "volume": np.full(T * N, 1e6, dtype=np.float64),
            "amount": np.full(T * N, 1e7, dtype=np.float64),
            "sector": lb_panel.fields["barra_industry"].reshape(-1, order="C"),
            "listed_days": np.full(T * N, 500.0, dtype=np.float64),
        }
    ).write_parquet(rq_dir / "panel_daily.parquet")
    pl.DataFrame(
        {
            "trading_date": date_col,
            "symbol": sym_col,
            "y_1": np.zeros(T * N, dtype=np.float64),
            "y_5": np.zeros(T * N, dtype=np.float64),
        }
    ).write_parquet(rq_dir / "labels.parquet")
    (rq_dir / "meta.json").write_text(
        json.dumps(
            {
                "market": "stocks",
                "panel_version": "rq",
                "dataset_id": "rq",
                "source": "ricequant",
                "capabilities": {
                    "dataset_id": "rq",
                    "source": "ricequant",
                    "symbol_format": "rq",
                    "return_clock": "open_adj",
                    "has_ohlcv": True,
                    "has_volume_amount": True,
                    "has_directional_limit": True,
                    "has_listed_days": True,
                },
            }
        ),
        encoding="utf-8",
    )

    # 加载 rq 时自动通过 RiskView 挂载 barra 的 vwap_open_30m 标签与风格暴露
    rq_loaded = prof.load_panel(PanelSpec(version="rq"))
    assert rq_loaded.meta.get("labels_clock") == "vwap_open_30m"
    assert "y_idio_1" in rq_loaded.labels
    assert "barra_beta" in rq_loaded.fields
    np.testing.assert_allclose(
        rq_loaded.labels["y_1"][5, 0],
        lb_panel.labels["y_1"][5, 0],
        atol=1e-12,
    )

    # 端到端验证独立 A 股统一回测脚本 run_stocks_backtest.py
    scripts_dir = Path(__file__).resolve().parents[2] / "scripts"
    if str(scripts_dir) not in sys.path:
        sys.path.insert(0, str(scripts_dir))
    import run_stocks_backtest  # type: ignore[import-not-found]

    out_dir = tmp_path / "bt_out"
    rc = run_stocks_backtest.main(
        [
            "--factor",
            "-ts_sum(c2c_ret, 3)",
            "--postprocess",
            "minimal",
            "--mode",
            "quantile",
            "--config",
            str(Path(__file__).resolve().parents[2] / "configs/stocks/backtest.yaml"),
            "--fees",
            str(Path(__file__).resolve().parents[2] / "configs/stocks/fees.yaml"),
            "--output-dir",
            str(out_dir),
        ],
        root_dir=tmp_path,
    )
    assert rc == 0
    metrics = json.loads((out_dir / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["entry_window"] == "vwap_open_30m"
    assert "bridge_audit" in metrics
    assert metrics["bridge_audit"]["date_overlap_count"] > 0
    assert (out_dir / "report.html").exists()
