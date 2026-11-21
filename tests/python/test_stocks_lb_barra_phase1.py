"""Phase 1 独立数据源 lb_barra_v1 构建管线与防前视单元测试 (tests/python/test_stocks_lb_barra_phase1.py)。"""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from factorlib.core.config import (
    StockBacktestConfig,
    StockDataConfig,
    StockExecutionConfig,
    StockUniverseFilterConfig,
)
from factorlib.core.probe import probe_factor
from factorlib.markets.profile import PanelSpec
from factorlib.markets.stocks.ares_bridge import (
    BARRA_FACTOR_ORDER,
    BARRA_STYLE_FACTORS,
    build_lb_barra_panel,
    compute_causal_ex_cum_factor,
    read_ares_table,
)
from factorlib.markets.stocks.backtest import run_stock_backtest_pipeline
from factorlib.markets.stocks.profile import StocksProfile


def _write_ares_table(
    table_dir: Path,
    dates_int: np.ndarray,
    symbols: list[str],
    mat: np.ndarray,
    *,
    chunk_rows: int = 10,
) -> None:
    table_dir.mkdir(parents=True, exist_ok=True)
    n_rows, n_cols = mat.shape
    assert len(dates_int) == n_rows
    assert len(symbols) == n_cols

    np.save(table_dir / "index.npy", np.asarray(dates_int, dtype=np.int64))
    np.save(table_dir / "columns.npy", np.asarray(symbols, dtype=object))

    n_chunks = (n_rows + chunk_rows - 1) // chunk_rows
    for k in range(n_chunks):
        sub = np.ascontiguousarray(mat[k * chunk_rows : (k + 1) * chunk_rows, :], dtype=np.float64)
        sub.tofile(table_dir / f"{k}.ares")

    meta = np.array([np.float64, 1, n_rows, n_cols, chunk_rows, n_chunks], dtype=object)
    np.save(table_dir / "meta.npy", meta)


def _make_synthetic_lb_assets(
    raw_dir: Path,
    *,
    T: int = 45,
    seed: int = 123,
    perturb_from: int | None = None,
) -> dict[str, Any]:
    """构造完整可验证的合成 local_backtest/runtime_assets 目录。"""
    rng = np.random.default_rng(seed)
    dates_py = [
        date(2024, 1, 2) if i == 0 else date.fromordinal(date(2024, 1, 2).toordinal() + i)
        for i in range(T)
    ]
    dates_int = np.array([int(d.strftime("%Y%m%d")) for d in dates_py], dtype=np.int64)
    symbols = ["000001", "000002", "600000", "600519", "430047"]  # 含 1 只北交所 430047
    N = len(symbols)

    # 构造真实未复权价 + 在第 15 日股票 0 发生 2:1 拆股除权
    base_rets = rng.normal(0.0005, 0.015, size=(T, N))
    true_close = 20.0 * np.cumprod(1.0 + base_rets, axis=0)
    true_vwap = true_close * (1.0 + rng.normal(0.0, 0.004, size=(T, N)))
    if perturb_from is not None:
        true_close[perturb_from:, :] *= 1.8
        true_vwap[perturb_from:, :] *= 1.9

    close_raw = true_close.copy()
    vwap_raw = true_vwap.copy()
    pre_close_raw = np.zeros_like(close_raw)
    pre_close_raw[0, :] = 20.0
    pre_close_raw[1:, :] = close_raw[:-1, :]

    split_t = 15
    close_raw[split_t:, 0] *= 0.5
    vwap_raw[split_t:, 0] *= 0.5
    pre_close_raw[split_t, 0] = close_raw[split_t - 1, 0] * 0.5
    if split_t + 1 < T:
        pre_close_raw[split_t + 1 :, 0] = close_raw[split_t:-1, 0]

    # StockMask2 乘法语义 (1.0=正常, NaN=停牌/涨跌停/不在池)
    base_mask = np.ones((T, N), dtype=np.float64)
    limit_mask = np.ones((T, N), dtype=np.float64)
    susp_mask = np.ones((T, N), dtype=np.float64)

    susp_mask[10, 1] = np.nan  # t=10 股票 1 停牌
    close_raw[10, 1] = np.nan
    pre_close_raw[10, 1] = np.nan
    if 11 < T:
        pre_close_raw[11, 1] = close_raw[9, 1]
    limit_mask[12, 2] = np.nan  # t=12 股票 2 涨跌停
    base_mask[20, 3] = np.nan  # t=20 股票 3 不在基础池

    if perturb_from is not None:
        susp_mask[perturb_from:, 0] = np.nan
        limit_mask[perturb_from:, 1] = np.nan

    mkt_cap = rng.uniform(5e9, 2e11, size=(T, N))
    industry = np.tile(np.array([1.0, 2.0, 3.0, 4.0, 5.0]), (T, 1))
    if perturb_from is not None:
        mkt_cap[perturb_from:, :] *= 3.0

    ashare = raw_dir / "Cache" / "AshareCache"
    _write_ares_table(ashare / "1d_DailyKline/DailyKline.close", dates_int, symbols, close_raw)
    _write_ares_table(
        ashare / "1d_DailyKline/DailyKline.pre_close", dates_int, symbols, pre_close_raw
    )
    _write_ares_table(ashare / "1d_IntraVwap/IntraVwap.VwapBegin30", dates_int, symbols, vwap_raw)
    _write_ares_table(ashare / "1d_DailyFdm/DailyFdm.mkt_cap", dates_int, symbols, mkt_cap)
    _write_ares_table(
        ashare / "1d_StockMask2/StockMask2.BaseUnivMask", dates_int, symbols, base_mask
    )
    _write_ares_table(ashare / "1d_StockMask2/StockMask2.LimitMask", dates_int, symbols, limit_mask)
    _write_ares_table(
        ashare / "1d_StockMask2/StockMask2.SuspendStock", dates_int, symbols, susp_mask
    )
    _write_ares_table(ashare / "1d_BarraCNE5/BarraCNE5.INDUSTRY", dates_int, symbols, industry)

    for s_name in BARRA_STYLE_FACTORS:
        s_mat = rng.normal(0.0, 1.0, size=(T, N))
        if perturb_from is not None:
            s_mat[perturb_from:, :] += 5.0
        _write_ares_table(ashare / f"1d_BarraCNE5/BarraCNE5.{s_name}", dates_int, symbols, s_mat)

    # CNE5SpecRet.parquet (含首行全 NaN 测试有效日期过滤)
    spec_ret = rng.normal(0.0, 0.015, size=(T, N))
    spec_ret[0, :] = np.nan
    if perturb_from is not None:
        spec_ret[perturb_from:, :] += 0.25

    ret_dir = raw_dir / "Cache" / "BaseCache" / "BarraCNE5_RET"
    ret_dir.mkdir(parents=True, exist_ok=True)
    spec_tbl = pa.table(
        {sym: spec_ret[:, j] for j, sym in enumerate(symbols)} | {"date": dates_int}
    )
    pq.write_table(spec_tbl, ret_dir / "CNE5SpecRet.parquet")

    # CNE5FactorRet.parquet
    f_cols = list(BARRA_FACTOR_ORDER)
    fr_mat = rng.normal(0.0, 0.01, size=(T, len(f_cols)))
    fr_mat[0, :] = np.nan
    fr_tbl = pa.table(
        {col: fr_mat[:, j] for j, col in enumerate(f_cols)} | {"__index_level_0__": dates_int}
    )
    pq.write_table(fr_tbl, ret_dir / "CNE5FactorRet.parquet")

    # BarraCNE5_COV/*.parquet (最近 5 个交易日)
    cov_dir = raw_dir / "Cache" / "BaseCache" / "BarraCNE5_COV"
    cov_dir.mkdir(parents=True, exist_ok=True)
    for d_int in dates_int[-5:]:
        a = rng.normal(0.0, 0.01, size=(len(f_cols), len(f_cols)))
        cov_m = a @ a.T + np.eye(len(f_cols)) * 1e-4
        ct = pa.table(
            {col: cov_m[:, j] for j, col in enumerate(f_cols)} | {"__index_level_0__": f_cols}
        )
        pq.write_table(ct, cov_dir / f"{int(d_int)}.parquet")

    # benchmark_000905_SH.csv
    bench_csv = raw_dir / "benchmark_000905_SH.csv"
    b_close = 6000.0 * np.cumprod(1.0 + rng.normal(0.0002, 0.01, size=T))
    lines = ["ts_code,trade_date,close"] + [
        f"000905.SH,{int(d)},{b:.4f}" for d, b in zip(dates_int, b_close, strict=True)
    ]
    bench_csv.write_text("\n".join(lines) + "\n", encoding="utf-8")

    return {
        "dates": dates_py,
        "dates_int": dates_int,
        "symbols": symbols,
        "true_close": true_close,
        "true_vwap": true_vwap,
        "spec_ret": spec_ret,
        "bench_csv": bench_csv,
    }


def test_read_ares_and_causal_ex_cum_factor(tmp_path: Path) -> None:
    dates_int = np.array([20240102, 20240103, 20240104, 20240105], dtype=np.int64)
    syms = ["000001", "000002"]
    close_raw = np.array([[20.0, 10.0], [10.0, 10.5], [10.2, np.nan], [10.5, 10.5]])
    pre_close = np.array([[20.0, 10.0], [10.0, 10.0], [10.0, np.nan], [10.2, 10.5]])

    _write_ares_table(tmp_path / "tbl", dates_int, syms, close_raw, chunk_rows=2)
    d_out, s_out, m_out = read_ares_table(tmp_path / "tbl")
    assert np.array_equal(d_out, dates_int)
    assert s_out == syms
    np.testing.assert_allclose(m_out, close_raw, equal_nan=True)

    ex = compute_causal_ex_cum_factor(close_raw, pre_close)
    # 股票 0 在 t=1 发生 2:1 拆股: ex[1] = 20 / 10 = 2.0, close_adj[1] = 20.0
    np.testing.assert_allclose(ex[:, 0], [1.0, 2.0, 2.0, 2.0])
    # 股票 1 在 t=2 停牌，t=3 复牌时 last_close=10.5, pre_close=10.5 -> ex 不变
    np.testing.assert_allclose(ex[:, 1], [1.0, 1.0, 1.0, 1.0])


def test_build_lb_barra_panel_end_to_end_and_probe_guard(tmp_path: Path) -> None:
    raw_dir = tmp_path / "local_assets"
    derived_dir = tmp_path / "data" / "derived" / "stocks"
    fx = _make_synthetic_lb_assets(raw_dir, T=45, seed=99)

    cfg = StockDataConfig(
        raw_dir=raw_dir,
        derived_dir=derived_dir,
        panel_version="barra",
        dataset_id="barra",
        source="barra",
        benchmark_id="000905.SH",
        benchmark_csv=fx["bench_csv"],
        price_basis="hfq",
        entry_window="vwap_open_30m",
        exec_price_mode="vwap_open_30m_adj",
        start_date=fx["dates"][0],
        end_date=fx["dates"][-1],
        universe_filters=StockUniverseFilterConfig(
            use_source_base_univ_mask=True,
            exclude_bjse=True,
        ),
        label_horizons=(1, 2, 5),
        include_idiosyncratic_labels=True,
    )
    panel_df, labels_df, meta = build_lb_barra_panel(cfg, pack_cov=True, show_progress=False)

    assert meta["total_trading_days"] == 44  # 第 0 行 CNE5SpecRet 全 NaN 被跳过
    assert meta["total_symbols"] == 5
    assert meta["capabilities"]["has_ohlcv"] is False
    assert meta["capabilities"]["has_listed_days"] is False
    assert meta["capabilities"]["has_directional_limit"] is False
    assert meta["capabilities"]["has_vwap_open_30m"] is True
    assert (cfg.panel_dir / "risk_store" / "factor_returns.parquet").exists()
    assert (cfg.panel_dir / "risk_store" / "factor_cov.npz").exists()
    assert (cfg.panel_dir / "risk_store" / "store.json").exists()
    assert "y_idio_1" in labels_df.columns and "y_idio_2" in labels_df.columns

    # 通过 StocksProfile 加载独立派生面板
    prof = StocksProfile(root=tmp_path)
    panel = prof.load_panel(PanelSpec(version="barra", panel_dir=cfg.panel_dir))
    assert panel.shape == (44, 5)
    assert "vwap_open_30m_adj" in panel.fields
    assert "spec_ret_vwap30" in panel.fields
    assert "spec_vol_ewma" in panel.fields
    assert "barra_beta" in panel.fields and "barra_liquidty" in panel.fields
    assert "close_raw" in panel.aux_fields and "close_raw" not in panel.fields
    assert "vwap_open_30m_raw" in panel.aux_fields and "vwap_open_30m_raw" not in panel.fields

    # 校验反极性掩码与 exclude_bjse=True
    bj_idx = panel.symbols.index("430047")
    assert not np.any(panel.mask[:, bj_idx])
    # 原全序列 t=10 (对应切片后 t=9) 股票 1 停牌
    assert panel.aux_fields["is_suspended"][9, 1] == 1.0
    assert panel.aux_fields["can_buy_mask"][9, 1] == 1.0
    # 原全序列 t=12 (对应切片后 t=11) 股票 600000 涨跌停
    s600000 = panel.symbols.index("600000")
    assert panel.aux_fields["can_buy_mask"][11, s600000] == 0.0
    assert panel.aux_fields["can_sell_mask"][11, s600000] == 0.0

    # 校验 y_idio_1[T] == CNE5SpecRet[T+2] 与 y_idio_2[T] == (1+u[T+2])*(1+u[T+3])-1
    u_mat = panel.fields["spec_ret_vwap30"]
    np.testing.assert_allclose(panel.labels["y_idio_1"][0, 0], u_mat[2, 0], atol=1e-12)
    expected_y2 = (1.0 + u_mat[2, 0]) * (1.0 + u_mat[3, 0]) - 1.0
    np.testing.assert_allclose(panel.labels["y_idio_2"][0, 0], expected_y2, atol=1e-12)

    # 校验 probe_factor 依据 capabilities 拦截缺失字段与 dropNewStock
    bad_ohlcv = probe_factor("ts_mean(volume, 5)", panel=panel)
    assert bad_ohlcv["status"] == "error"

    bad_new = probe_factor(
        {
            "name": "test_new",
            "factor_type": "expression",
            "expression": "-ts_sum(spec_ret_vwap30, 5)",
            "postprocess": ["baseUniMask", "dropNewStock:120"],
        },
        panel=panel,
    )
    assert bad_new["status"] == "error"
    assert any("dropNewStock" in e for e in bad_new["errors"])

    good_spec = probe_factor(
        {
            "name": "spec_rev_5",
            "factor_type": "expression",
            "expression": "-ts_sum(spec_ret_vwap30, 5)",
            "postprocess": prof.default_postprocess("barra"),
        },
        panel=panel,
    )
    assert good_spec["status"] == "ok", good_spec["errors"]

    # 校验在 lb_barra_v1 面板上以 vwap_open_30m 执行回测跑通
    weights = np.full(panel.shape, 0.2, dtype=np.float64, order="F")
    bt_cfg = StockBacktestConfig(
        execution=StockExecutionConfig(entry_window="vwap_open_30m", price_basis="hfq")
    )
    res = run_stock_backtest_pipeline(panel, weights, config=bt_cfg)
    assert res.extras["limit_mask_mode"] == "bilateral_conservative"
    assert len(res.equity) == 44


def test_lb_barra_strict_causality_perturbation(tmp_path: Path) -> None:
    """因果摄动测试：篡改 T > T_perturb 的未来数据，历史 T <= T_perturb 必须绝对 0 差值。"""
    T = 45
    t_perturb = 30
    raw_base = tmp_path / "base_raw"
    raw_pert = tmp_path / "pert_raw"
    _make_synthetic_lb_assets(raw_base, T=T, seed=77, perturb_from=None)
    _make_synthetic_lb_assets(raw_pert, T=T, seed=77, perturb_from=t_perturb)

    def _load(rdir: Path, pdir: Path):
        cfg = StockDataConfig(
            raw_dir=rdir,
            derived_dir=pdir,
            panel_version="barra",
            source="barra",
            entry_window="vwap_open_30m",
            exec_price_mode="vwap_open_30m_adj",
            start_date=date(2024, 1, 2),
            end_date=date(2024, 3, 30),
            include_idiosyncratic_labels=True,
        )
        build_lb_barra_panel(cfg, pack_cov=False, show_progress=False)
        return StocksProfile(root=pdir.parent.parent).load_panel(
            PanelSpec(version="barra", panel_dir=cfg.panel_dir)
        )

    p_base = _load(raw_base, tmp_path / "derived_base")
    p_pert = _load(raw_pert, tmp_path / "derived_pert")

    # 原始第 0 行被过滤，因此 t_perturb 在切片后对应索引 t_cut = t_perturb - 1
    t_cut = t_perturb - 1
    for col in (
        "close_adj",
        "vwap_open_30m_adj",
        "c2c_ret",
        "vwap30_ret",
        "spec_ret_vwap30",
        "spec_vol_ewma",
    ):
        np.testing.assert_array_equal(p_base.fields[col][:t_cut, :], p_pert.fields[col][:t_cut, :])
    np.testing.assert_array_equal(p_base.mask[:t_cut, :], p_pert.mask[:t_cut, :])
