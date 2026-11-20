"""Phase 3 P0-1 验收测试：FactorLab MCP 1d 变量全量对齐与别名兼容层。

覆盖：
1. `configs/stocks/data.yaml` 的 `rq_v2` 数据集声明与 `extended_fields` 解析；
2. `StockPanelLoader` 扩充字段（daily_basic 估值换手 / num_trades / 三大指数权重 / 日内收益）
   在合成夹具上的端到端构建；
3. 指数成分权重严格 as-of backward 对齐与成分池掩码极性；
4. `core/variable_aliases` 别名归一（幂等、真实字段优先、极性反转变量拒绝对齐）；
5. `probe_factor` / `evaluate_factor` 入口的别名归一与 `canon_expr` 真实字段化；
6. `FactorCatalog` 别名条目可检索。
"""

from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path

import numpy as np
import polars as pl
import pytest
from factorlib.core.catalog import FactorCatalog
from factorlib.core.config import load_stock_datasets_config
from factorlib.core.panel import Panel
from factorlib.core.probe import FactorDefinition, evaluate_factor_definition, probe_factor
from factorlib.core.variable_aliases import (
    RQ_VARIABLE_ALIASES,
    UNMAPPED_MCP_VARIABLES,
    alias_catalog_entries,
    resolve_expression_aliases,
)
from factorlib.markets.stocks.panel_loader import StockPanelLoader

ROOT = Path(__file__).resolve().parents[2]


# ------------------------------------------------------------------------------
# 1. 配置声明
# ------------------------------------------------------------------------------


def test_rq_v2_dataset_declared_with_extended_fields():
    default_ds, datasets = load_stock_datasets_config(
        ROOT / "configs" / "stocks" / "data.yaml", root=ROOT
    )
    assert "rq_v2" in datasets
    cfg = datasets["rq_v2"]
    assert cfg.panel_version == "v2"
    assert cfg.extended_fields is True
    assert cfg.start_date <= cfg.end_date
    assert cfg.source == "ricequant"
    # v1 保持零变更兼容（默认不引入扩充字段）
    assert datasets["rq_v1"].extended_fields is False
    assert default_ds in datasets


# ------------------------------------------------------------------------------
# 2. 合成夹具上的扩充字段构建
# ------------------------------------------------------------------------------


def _write_parquet(path: Path, df: pl.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(path)


@pytest.fixture()
def synthetic_raw_dir(tmp_path: Path) -> Path:
    """构造含 bar_1d / daily_basic / trade_status / benchmark 的最小合成原始库。"""
    raw = tmp_path / "raw"
    days = [date(2024, 1, 2) + timedelta(days=i) for i in range(6)]
    syms = ["000001.XSHE", "600519.XSHG", "000002.XSHE"]

    bar_rows = []
    for d in days:
        for k, s in enumerate(syms):
            base = 10.0 + k
            bar_rows.append(
                {
                    "trading_date": d,
                    "order_book_id": s,
                    "open": base,
                    "high": base * 1.02,
                    "low": base * 0.98,
                    "close": base * 1.01,
                    "prev_close": base * 0.99,
                    "limit_up": base * 1.1,
                    "limit_down": base * 0.9,
                    "volume": 1.0e6,
                    "total_turnover": 1.0e7,
                    "num_trades": 1000.0 + 100.0 * k,
                }
            )
    _write_parquet(
        raw / "bar_1d" / "bar_1d_2024.parquet",
        pl.DataFrame(bar_rows).sort(["order_book_id", "trading_date"]),
    )

    db_rows = []
    for d in days:
        for k, s in enumerate(syms):
            db_rows.append(
                {
                    "trading_date": d,
                    "order_book_id": s,
                    "market_cap": 1.0e10 * (k + 1),
                    "a_share_market_val_in_circulation": 8.0e9 * (k + 1),
                    "pe_ratio_ttm": 10.0 + k,
                    "pb_ratio_lf": 1.0 + 0.1 * k,
                    "ps_ratio_ttm": 2.0 + k,
                    "turnover_rate": 0.5 + 0.1 * k,
                }
            )
    _write_parquet(
        raw / "daily_basic" / "daily_basic_2024.parquet",
        pl.DataFrame(db_rows).sort(["order_book_id", "trading_date"]),
    )

    ts_rows = [
        {"trading_date": d, "order_book_id": s, "is_st": False, "is_suspended": False}
        for d in days
        for s in syms
    ]
    _write_parquet(raw / "trade_status" / "trade_status_2024.parquet", pl.DataFrame(ts_rows))

    _write_parquet(
        raw / "instruments_info" / "instruments_info.parquet",
        pl.DataFrame(
            {
                "order_book_id": syms,
                "listed_date": ["2010-01-01"] * len(syms),
                "de_listed_date": ["0000-00-00"] * len(syms),
            }
        ),
    )

    # 指数成分权重：000300 在 day3 起把 000002 剔除（验证 as-of 对齐与掩码极性）
    sorted_syms = sorted(syms)
    iw_rows = []
    for s in sorted_syms[:2]:
        iw_rows.append(
            {
                "snapshot_date": days[0],
                "index_id": "000300.XSHG",
                "order_book_id": s,
                "weight": 0.5,
            }
        )
    iw_rows.append(
        {
            "snapshot_date": days[3],
            "index_id": "000300.XSHG",
            "order_book_id": "000001.XSHE",
            "weight": 1.0,
        }
    )
    _write_parquet(raw / "benchmark" / "index_weights_2024.parquet", pl.DataFrame(iw_rows))
    _write_parquet(
        raw / "benchmark" / "index_bar_1d.parquet",
        pl.DataFrame(
            {
                "order_book_id": ["000985.XSHG"] * len(days),
                "trading_date": days,
                "open": [4000.0] * len(days),
                "high": [4010.0] * len(days),
                "low": [3990.0] * len(days),
                "close": [4005.0] * len(days),
                "prev_close": [4000.0] * len(days),
                "volume": [1.0e9] * len(days),
                "total_turnover": [1.0e10] * len(days),
            }
        ),
    )
    return raw


def test_extended_fields_build_and_semantics(synthetic_raw_dir: Path) -> None:
    loader = StockPanelLoader(synthetic_raw_dir)
    base = loader.load(
        start_date=date(2024, 1, 1), end_date=date(2024, 12, 31), extended_fields=False
    )
    ext = loader.load(
        start_date=date(2024, 1, 1), end_date=date(2024, 12, 31), extended_fields=True
    )

    new_fields = set(ext.fields) - set(base.fields)
    for name in (
        "mkt_cap",
        "circ_mv",
        "pe_ratio_ttm",
        "pb_ratio_lf",
        "ps_ratio_ttm",
        "turnover_rate",
        "log_mkt_cap",
        "log_circ_mv",
        "num_trades",
        "avg_trade_size",
        "log_num_trades",
        "daily_derived_returns_intra",
        "pre_close",
        "close_change",
        "index_weight_000300_sh",
        "trade_universe_hs300",
    ):
        assert name in new_fields, f"缺少扩充字段: {name}"

    # 数值校验：log_mkt_cap == ln(mkt_cap)
    np.testing.assert_allclose(
        ext.fields["log_mkt_cap"], np.log(ext.fields["mkt_cap"]), rtol=1e-12, atol=0.0
    )
    # 日内收益率 == close / open - 1（复权比值）
    intra_ref = ext.fields["close_hfq"] / ext.fields["open_adj"] - 1.0
    np.testing.assert_allclose(
        ext.fields["daily_derived_returns_intra"], intra_ref, rtol=1e-12, atol=1e-14
    )
    # avg_trade_size == amount / num_trades
    np.testing.assert_allclose(
        ext.fields["avg_trade_size"],
        ext.fields["amount"] / ext.fields["num_trades"],
        rtol=1e-12,
        atol=0.0,
    )
    # close_change == close - pre_close
    np.testing.assert_allclose(
        ext.fields["close_change"],
        ext.fields["close"] - ext.fields["pre_close"],
        rtol=1e-12,
        atol=1e-12,
    )
    # 基础面板零变更
    for key, mat in base.fields.items():
        np.testing.assert_allclose(ext.fields[key], mat, rtol=0.0, atol=0.0)


def test_index_weight_asof_alignment_and_mask_polarity(synthetic_raw_dir: Path) -> None:
    loader = StockPanelLoader(synthetic_raw_dir)
    data = loader.load(
        start_date=date(2024, 1, 1), end_date=date(2024, 12, 31), extended_fields=True
    )
    w = data.fields["index_weight_000300_sh"]
    mask = data.fields["trade_universe_hs300"]
    syms = list(data.symbols)
    i_000001 = syms.index("000001.XSHE")
    i_000002 = syms.index("000002.XSHE")
    i_600519 = syms.index("600519.XSHG")

    # 首个快照前（第 0 日之前无快照）→ NaN；第 0 日起生效
    assert np.isfinite(w[0, i_000001])
    np.testing.assert_allclose(w[0, i_000001], 0.5)
    # 第 3 日新快照把 000001 权重调为 1.0，000002 掉出成分（严格 as-of 且不回填旧值）
    np.testing.assert_allclose(w[3, i_000001], 1.0)
    assert np.isnan(w[3, i_000002])
    # 非成分标的权重恒为 NaN
    assert np.isnan(w[:, i_600519]).all()
    # 成分池掩码极性：成分内 1.0，成分外 NaN
    assert mask[0, i_000001] == 1.0
    assert np.isnan(mask[0, i_600519])
    assert np.isnan(mask[3, i_000002])


# ------------------------------------------------------------------------------
# 3. 别名归一化
# ------------------------------------------------------------------------------


def test_alias_resolution_is_idempotent_and_field_priority() -> None:
    fields = {
        "c2c_ret",
        "open_adj",
        "in_universe_mask",
        "barra_size",
        "mkt_cap",
        "close_hfq",
        "log_amount",
    }
    expr = (
        "neg(add(cs_zscore(ts_corr(close_hfq, log_amount, 20)), "
        "cs_zscore(ts_std(daily_derived_returns_c2c_1d, 20))))"
    )
    resolved, rep = resolve_expression_aliases(expr, fields)
    assert "daily_derived_returns_c2c_1d" not in resolved
    assert "ts_std(c2c_ret, 20)" in resolved
    assert rep.applied["daily_derived_returns_c2c_1d"] == "c2c_ret"

    # 幂等：重复解析不再变化
    again, rep2 = resolve_expression_aliases(resolved, fields)
    assert again == resolved
    assert rep2.applied_count == 0

    # 真实字段优先：面板已存在同名字段时绝不改写
    fields_with_alias = set(fields) | {"daily_derived_returns_c2c_1d"}
    keep, rep3 = resolve_expression_aliases(expr, fields_with_alias)
    assert "daily_derived_returns_c2c_1d" in keep
    assert "daily_derived_returns_c2c_1d" in rep3.skipped_conflict

    # 目标字段缺失 → 不改写（避免静默 NaN）
    missing, rep4 = resolve_expression_aliases("ts_std(daily_derived_returns_c2c_1d, 5)", {"x"})
    assert "daily_derived_returns_c2c_1d" in missing
    assert "daily_derived_returns_c2c_1d" in rep4.skipped_missing


def test_barra_style_alias_and_polarity_guard() -> None:
    fields = {"barra_size", "barra_beta", "barra_industry", "mkt_cap", "c2c_ret"}
    resolved, rep = resolve_expression_aliases("add(barra_cne5_size, barra_cne5_beta)", fields)
    assert resolved == "add(barra_size, barra_beta)"
    assert rep.applied_count == 2

    # 大写下标别名（SIZE）在没有真实字段时映射到 barra_size
    resolved2, _ = resolve_expression_aliases("cs_zscore(SIZE)", fields)
    assert resolved2 == "cs_zscore(barra_size)"

    # 极性反转变量绝不登记别名，仅提示
    resolved3, rep3 = resolve_expression_aliases("multiply_like(SuspendStock)", fields)
    assert "SuspendStock" in resolved3
    assert "SuspendStock" in rep3.unmapped_used
    assert "SuspendStock" in UNMAPPED_MCP_VARIABLES
    assert "SuspendStock" not in RQ_VARIABLE_ALIASES


def test_alias_catalog_entries_filtered_by_available_fields() -> None:
    entries = alias_catalog_entries({"c2c_ret", "barra_size"})
    names = {e["name"] for e in entries}
    assert "daily_derived_returns_c2c_1d" in names
    assert "barra_cne5_size" in names
    # 目标字段缺失的别名不登记
    assert "daily_kline_open_hfq" not in names
    for e in entries:
        assert e["catalog"] == "variables"
        assert e["category"] == "factorlab_alias"


# ------------------------------------------------------------------------------
# 4. probe / evaluate 入口打通
# ------------------------------------------------------------------------------


def _make_linear_panel(T: int = 30, N: int = 6) -> Panel:
    rng = np.random.default_rng(7)
    ret = rng.normal(0.0, 0.02, size=(T, N))
    close = np.cumprod(1.0 + ret, axis=0) * 10.0
    fields = {
        "c2c_ret": np.asfortranarray(ret),
        "close_hfq": np.asfortranarray(close),
        "open_adj": np.asfortranarray(close * 0.995),
        "high_adj": np.asfortranarray(close * 1.01),
        "low_adj": np.asfortranarray(close * 0.99),
        "amount": np.asfortranarray(np.abs(rng.normal(1.0e8, 1.0e7, size=(T, N)))),
        "volume": np.asfortranarray(np.abs(rng.normal(1.0e6, 1.0e5, size=(T, N)))),
        "log_amount": np.asfortranarray(np.log(np.abs(rng.normal(1.0e8, 1e7, (T, N))))),
        "in_universe_mask": np.asfortranarray(np.ones((T, N))),
        "limit_mask": np.asfortranarray(np.ones((T, N))),
        "stock_listed_days": np.asfortranarray(np.full((T, N), 500.0)),
        "listed_days": np.asfortranarray(np.full((T, N), 500.0)),
    }
    mask = np.ones((T, N), dtype=bool, order="F")
    labels = {
        "y_1": np.asfortranarray(rng.normal(0.0, 0.02, size=(T, N))),
        "y_5": np.asfortranarray(rng.normal(0.0, 0.02, size=(T, N))),
    }
    return Panel(
        dates=np.array([f"202401{i:02d}" for i in range(1, T + 1)]),
        symbols=[f"S{j:03d}" for j in range(N)],
        fields=fields,
        mask=mask,
        labels=labels,
        aux_fields={},
        meta={"market": "stocks"},
    )


def test_probe_and_evaluate_resolve_aliases() -> None:
    panel = _make_linear_panel()
    defn = FactorDefinition(
        name="alias_probe",
        factor_type="expression",
        expression="neg(ts_std(daily_derived_returns_c2c_1d, 5))",
        postprocess=["baseUniMask", "trdMask", "zScore"],
    )
    rep = probe_factor(defn, panel=panel)
    assert rep["status"] == "ok", rep["errors"]
    assert "c2c_ret" in rep["canon_expr"]
    assert "daily_derived_returns_c2c_1d" not in rep["canon_expr"]
    assert rep["variable_alias_report"]["applied"]["daily_derived_returns_c2c_1d"] == "c2c_ret"
    assert any("变量别名归一" in w for w in rep["stage_order_warnings"])

    mat, rep2 = evaluate_factor_definition(defn, panel)
    assert mat.shape == panel.shape
    assert rep2["status"] == "ok"


def test_probe_reports_unmapped_polarity_variable() -> None:
    panel = _make_linear_panel()
    defn = FactorDefinition(
        name="unmapped_probe",
        factor_type="expression",
        expression="ts_std(SuspendStock, 5)",
    )
    rep = probe_factor(defn, panel=panel)
    # 变量仍是未知字段：静态检查必须报错，但同时给出极性说明告警
    assert rep["status"] == "error"
    assert any("无本地面板等价字段" in w for w in rep["stage_order_warnings"])


# ------------------------------------------------------------------------------
# 5. 目录检索
# ------------------------------------------------------------------------------


def test_catalog_exposes_alias_entries_for_panel() -> None:
    panel = _make_linear_panel()
    cat = FactorCatalog(panel_dir="data/derived/stocks/v1", market="stocks", panel=panel)
    alias_names = {v["name"] for v in cat.variables if v.get("category") == "factorlab_alias"}
    assert "daily_derived_returns_c2c_1d" in alias_names
    hit = cat.get_entry("daily_derived_returns_c2c_1d", catalog="variables")
    assert hit is not None and hit["resolves_to"] == "c2c_ret"
