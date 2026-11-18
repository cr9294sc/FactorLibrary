"""Phase 1 控制面与双市场底座重构验收测试 (tests/python/test_phase1_control_plane.py)。

覆盖：
1. SQLite WAL 并发模式、旧版单市场库幂等迁移与 `(market, canon_hash)` 双市场隔离；
2. `feature_set` 与不可变 `research_recipe`（深拷贝冻结、内容哈希去重、SQLite UPDATE 触发器阻断）；
3. `dl_alpha` 因子注册、`.npz` 物化、探针与统一后处理求值、以及 ML 合成管线加载；
4. `MarketProfile` 扩展包挂载接口、`lag_days` / `asof_rule` 校验、1m 构建器与因果摄动零泄露；
5. `scripts/run_gp.py --recipe <id>` 与 `scripts/run_ml.py --recipe <id>` 真实端到端执行闭环。
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import numpy as np
import polars as pl
import pytest
from factorlib.core.catalog import FactorCatalog
from factorlib.core.config import MLConfig, WalkForwardConfig
from factorlib.core.extensions import (
    ExtensionPack,
    ExtensionPackMeta,
    save_extension_pack,
)
from factorlib.core.panel import Panel
from factorlib.core.probe import FactorDefinition, evaluate_factor_definition, probe_factor
from factorlib.core.registry import (
    ExperimentRunRecord,
    FactorMetricsRecord,
    FactorRecord,
    FactorRegistry,
    FeatureSetRecord,
)
from factorlib.core.workbench import create_synthetic_futures_panel
from factorlib.markets.futures.profile import FuturesProfile
from factorlib.markets.profile import MarketProfile
from factorlib.markets.registry import get_profile
from factorlib.markets.stocks.profile import StocksProfile
from factorlib.ml.pipeline import WalkForwardPipeline

import scripts.run_gp as run_gp_mod
import scripts.run_ml as run_ml_mod


def _save_panel_as_parquet_dir(panel: Panel, out_dir: Path) -> Path:
    """将内存 Panel 写盘为标准 `panel_daily.parquet` + `labels.parquet` + `meta.json`。"""
    out_dir.mkdir(parents=True, exist_ok=True)
    T, N = panel.shape
    rows_panel: dict[str, list[object]] = {
        "trading_date": [],
        "symbol": [],
        "in_universe": [],
    }
    field_names = [k for k in panel.fields if k not in ("in_universe_mask", "limit_mask", "sector")]
    aux_names = list(panel.aux_fields.keys())
    for k in field_names + aux_names:
        rows_panel[k] = []

    has_sector = "sector" in panel.fields
    if has_sector:
        rows_panel["sector"] = []

    rows_labels: dict[str, list[object]] = {"trading_date": [], "symbol": []}
    for lname in panel.labels:
        rows_labels[lname] = []

    for t in range(T):
        d_str = str(panel.dates[t])
        for n in range(N):
            sym = panel.symbols[n]
            rows_panel["trading_date"].append(d_str)
            rows_panel["symbol"].append(sym)
            rows_panel["in_universe"].append(bool(panel.mask[t, n]))
            if has_sector:
                rows_panel["sector"].append(f"S{int(panel.fields['sector'][t, n])}")
            for k in field_names:
                rows_panel[k].append(float(panel.fields[k][t, n]))
            for k in aux_names:
                rows_panel[k].append(float(panel.aux_fields[k][t, n]))

            rows_labels["trading_date"].append(d_str)
            rows_labels["symbol"].append(sym)
            for lname, lmat in panel.labels.items():
                rows_labels[lname].append(float(lmat[t, n]))

    pl.DataFrame(rows_panel).write_parquet(out_dir / "panel_daily.parquet")
    pl.DataFrame(rows_labels).write_parquet(out_dir / "labels.parquet")
    (out_dir / "meta.json").write_text(json.dumps(panel.meta, ensure_ascii=False), encoding="utf-8")
    return out_dir


def test_sqlite_wal_market_isolation_and_legacy_migration(tmp_path: Path) -> None:
    """验证 WAL 模式开启、旧版单市场 DB 原子迁移与双市场隔离。"""
    db_path = tmp_path / "legacy_factors.sqlite"

    # 1. 构造旧版单市场表结构（无 market、feature_set、research_recipe）
    conn = sqlite3.connect(str(db_path))
    conn.executescript(
        """
        CREATE TABLE factor (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            canon_expr TEXT NOT NULL,
            canon_hash TEXT NOT NULL UNIQUE,
            mode TEXT NOT NULL,
            sign INTEGER NOT NULL,
            n_nodes INTEGER NOT NULL,
            depth INTEGER NOT NULL,
            source TEXT NOT NULL,
            source_run TEXT,
            data_version TEXT NOT NULL,
            status TEXT NOT NULL DEFAULT 'candidate',
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        );
        CREATE TABLE factor_metrics (
            factor_id INTEGER NOT NULL REFERENCES factor(id) ON DELETE CASCADE,
            split TEXT NOT NULL,
            horizon INTEGER NOT NULL,
            ic_mean REAL NOT NULL,
            ic_std REAL NOT NULL,
            ic_ir REAL NOT NULL,
            ic_t REAL NOT NULL,
            ic_win_rate REAL NOT NULL,
            coverage REAL NOT NULL,
            turnover REAL NOT NULL,
            ls_sharpe REAL NOT NULL,
            ls_annual_ret REAL NOT NULL,
            ls_max_dd REAL NOT NULL,
            computed_at TEXT NOT NULL,
            PRIMARY KEY (factor_id, split, horizon)
        );
        CREATE TABLE experiment_run (
            run_id TEXT PRIMARY KEY,
            kind TEXT NOT NULL,
            config_yaml TEXT NOT NULL,
            data_version TEXT NOT NULL,
            git_sha TEXT NOT NULL,
            n_evaluated INTEGER DEFAULT 0,
            started_at TEXT NOT NULL,
            finished_at TEXT,
            status TEXT NOT NULL CHECK (status IN ('running', 'completed', 'failed'))
        );
        INSERT INTO factor (
            canon_expr, canon_hash, mode, sign, n_nodes, depth,
            source, source_run, data_version, status, created_at, updated_at
        ) VALUES (
            'pct_change(close, 20)', 'hash_shared_mom20', 'cs', 1, 3, 2,
            'seed', 'legacy_run', 'v1', 'accepted', '2026-01-01T00:00:00Z', '2026-01-01T00:00:00Z'
        );
        """
    )
    conn.close()

    # 2. 迁移初始化并校验 WAL 与备份文件
    reg = FactorRegistry(db_path=db_path)
    assert reg.journal_mode() == "wal"
    backups = list(tmp_path.glob("legacy_factors.sqlite.bak.*"))
    assert len(backups) == 1

    # 旧数据默认归入 futures 市场
    fut_rec = reg.get_factor_by_hash("hash_shared_mom20", market="futures")
    assert fut_rec is not None
    assert fut_rec.market == "futures"

    # 同指纹可在 stocks 市场独立入库，互不冲突
    stk_id = reg.register_factor(
        FactorRecord(
            market="stocks",
            canon_expr="pct_change(close, 20)",
            canon_hash="hash_shared_mom20",
            source="manual",
            status="accepted",
        )
    )
    assert stk_id != fut_rec.id
    assert len(reg.list_factors(market="futures")) == 1
    assert len(reg.list_factors(market="stocks")) == 1

    # 迁移后的 experiment_run 支持 'dl' 与 'cancelled'
    reg.record_run(
        ExperimentRunRecord(
            run_id="dl_cancelled_01",
            market="stocks",
            kind="dl",
            recipe_id="rcp_test001",
            config_yaml="epochs: 10",
            data_version="v1",
            git_sha="abc123",
            status="cancelled",
            pid=12345,
        )
    )
    loaded_run = reg.get_run("dl_cancelled_01")
    assert loaded_run is not None
    assert loaded_run.market == "stocks"
    assert loaded_run.status == "cancelled"
    assert loaded_run.recipe_id == "rcp_test001"


def test_feature_set_and_immutable_research_recipe(tmp_path: Path) -> None:
    """验证 FeatureSet CRUD、ResearchRecipe 深拷贝冻结、幂等哈希及 SQLite 不可变约束。"""
    reg = FactorRegistry(db_path=tmp_path / "ctrl.sqlite")

    fs = FeatureSetRecord(
        name="fs_fut_core",
        market="futures",
        description="期货核心特征池",
        base_fields=["close", "volume", "ry_ann"],
        extensions=["micro_fut_1m@v1"],
        operators=["ts_mean", "cs_zscore", "sub"],
        windows=[5, 10, 20],
        default_postprocess=[{"op": "baseUniMask"}, {"op": "zScore"}],
    )
    fs_id = reg.save_feature_set(fs)
    assert fs_id > 0

    recipe = reg.create_recipe(
        market="futures",
        kind="gp",
        feature_set_id=fs_id,
        target_label="y_5",
        splits={"train": ["2023-01-01", "2023-03-31"], "valid": ["2023-04-01", "2023-05-31"]},
        params={"population": 40, "generations": 2},
        seed=99,
        name="exp_gp_v1",
    )
    assert recipe.recipe_id is not None and recipe.recipe_id.startswith("rcp_")
    assert recipe.feature_set["base_fields"] == ["close", "volume", "ry_ann"]
    assert recipe.extensions == ["micro_fut_1m@v1"]

    # 修改 FeatureSet 不得影响已冻结的 Recipe 快照
    fs_updated = FeatureSetRecord(
        name="fs_fut_core",
        market="futures",
        base_fields=["open", "high"],
        extensions=[],
        operators=["ts_std"],
        windows=[60],
    )
    reg.save_feature_set(fs_updated, upsert=True)
    reloaded_recipe = reg.get_recipe(recipe.recipe_id)
    assert reloaded_recipe is not None
    assert reloaded_recipe.feature_set["base_fields"] == ["close", "volume", "ry_ann"]
    assert reloaded_recipe.extensions == ["micro_fut_1m@v1"]

    # 同内容创建配方返回相同 recipe_id（内容寻址幂等）
    dup_recipe = reg.create_recipe(
        market="futures",
        kind="gp",
        feature_set=reloaded_recipe.feature_set,
        target_label="y_5",
        splits={"train": ["2023-01-01", "2023-03-31"], "valid": ["2023-04-01", "2023-05-31"]},
        params={"population": 40, "generations": 2},
        seed=99,
    )
    assert dup_recipe.recipe_id == recipe.recipe_id

    # SQLite 触发器拒绝直接 UPDATE research_recipe
    with pytest.raises(sqlite3.DatabaseError, match="immutable"):
        with reg._get_connection() as conn:
            conn.execute(
                "UPDATE research_recipe SET target_label = 'y_10' WHERE recipe_id = ?",
                (recipe.recipe_id,),
            )


def test_dl_alpha_contract_probe_postprocess_and_ml_pipeline(tmp_path: Path) -> None:
    """验证 dl_alpha 因子注册、.npz 物化、同构后处理求值与 WalkForwardPipeline 集成。"""
    panel = create_synthetic_futures_panel(n_days=90, seed=7)
    reg = FactorRegistry(db_path=tmp_path / "dl_test.sqlite")

    rng = np.random.default_rng(123)
    raw_oos = np.asfortranarray(
        panel.fields["ry_ann"] + rng.normal(0.0, 0.02, panel.shape),
        dtype=np.float64,
    )
    fid = reg.register_dl_alpha(
        name="gru_futures_v1",
        market="futures",
        matrix=raw_oos,
        dates=panel.dates,
        symbols=panel.symbols,
        recipe_id="rcp_dl_demo_01",
        run_id="dl_run_01",
        data_version="synthetic_v1",
        postprocess=[{"op": "baseUniMask"}, {"op": "trdMask"}, {"op": "zScore"}],
        fold_diagnostics=[{"fold": 0, "valid_icir": 0.45}],
        metrics=[FactorMetricsRecord(split="valid", horizon=5, ic_ir=0.45)],
        status="accepted",
    )

    rec = reg.get_factor(fid)
    assert rec is not None
    assert rec.factor_type == "dl_alpha"
    assert rec.source == "dl_alpha"
    assert rec.market == "futures"
    assert rec.recipe_id == "rcp_dl_demo_01"
    assert rec.materialization_path is not None
    assert Path(rec.materialization_path).exists()
    assert len(reg.list_factor_versions(fid)) == 1

    # 通过统一探针与后处理管线求值
    defn = FactorDefinition(
        name="gru_futures_v1",
        factor_type="dl_alpha",
        market="futures",
        recipe_id="rcp_dl_demo_01",
        materialization_path=rec.materialization_path,
        postprocess="standard",
    )
    rep = probe_factor(defn, panel=panel)
    assert rep["status"] == "ok"

    pp_mat, _ = evaluate_factor_definition(defn, panel)
    assert pp_mat.shape == panel.shape
    assert pp_mat.flags.f_contiguous
    assert np.isfinite(pp_mat).sum() > 0

    # WalkForwardPipeline.from_registry 支持混合加载 expression + dl_alpha 因子
    reg.register_factor(
        FactorRecord(
            market="futures",
            canon_expr="pct_change(close, 5)",
            canon_hash="mom5_hash",
            source="seed",
            status="accepted",
        )
    )
    ml_cfg = MLConfig(
        target_label="y_5",
        features_status=("accepted",),
        walk_forward=WalkForwardConfig(
            start_predict_year=2023,
            end_predict_year=2023,
            min_train_years=1,
            purge_window=5,
            validation_size_days=10,
        ),
        models=("equal_weight", "ridge"),
    )
    pipe = WalkForwardPipeline.from_registry(
        panel=panel, config=ml_cfg, registry=reg, market="futures"
    )
    assert len(pipe.feature_names) == 2
    assert any("dl_alpha" in n for n in pipe.feature_names)


def test_extension_pack_contract_mounting_and_causality(tmp_path: Path) -> None:
    """验证 MarketProfile 扩展包构建、防泄漏契约校验、挂载与摄动零泄露。"""
    f_prof = get_profile("futures")
    s_prof = get_profile("stocks")
    assert isinstance(f_prof, MarketProfile)
    assert isinstance(s_prof, MarketProfile)

    # 1. 缺少 lag_days 或 asof_rule 的扩展包必须拒绝解析
    bad_dir = tmp_path / "bad_pack"
    bad_dir.mkdir(parents=True)
    (bad_dir / "pack.json").write_text(
        json.dumps({"pack_id": "bad", "market": "futures", "asof_rule": "T_close"}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="lag_days"):
        ExtensionPack.from_dir(bad_dir)

    (bad_dir / "pack.json").write_text(
        json.dumps({"pack_id": "bad", "market": "futures", "lag_days": 0}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="asof_rule"):
        ExtensionPack.from_dir(bad_dir)

    # 2. intraday_tensor 预留类型不得挂载到 1d 面板
    tensor_dir = tmp_path / "tensor_pack"
    tensor_dir.mkdir(parents=True)
    (tensor_dir / "pack.json").write_text(
        json.dumps(
            {
                "pack_id": "intra_bins",
                "market": "futures",
                "kind": "intraday_tensor",
                "lag_days": 0,
                "asof_rule": "T_close",
            }
        ),
        encoding="utf-8",
    )
    panel = create_synthetic_futures_panel(n_days=50, seed=11)
    with pytest.raises(NotImplementedError):
        panel.attach_extension(tensor_dir)

    # 3. 禁止跨市场挂载
    stk_builder = StocksProfile(root=tmp_path).minute_pack_builder()
    stk_df = pl.DataFrame(
        {
            "trading_date": [str(panel.dates[0])],
            "symbol": [panel.symbols[0]],
            "stk_rv": [0.02],
        }
    )
    stk_pack_dir = tmp_path / "data" / "extensions" / "stocks" / "stk_micro"
    stk_builder.build_pack(stk_pack_dir, pack_id="stk_micro", daily_df=stk_df)
    with pytest.raises(ValueError, match="跨市场"):
        panel.attach_extension(stk_pack_dir)

    # 4. 期货 1m 扩展包构建 + lag_days=1 因果平移 + 未来摄动零泄露
    rows_base: list[dict[str, object]] = []
    rows_pert: list[dict[str, object]] = []
    cutoff_idx = 20
    for t_idx, d_str in enumerate(panel.dates):
        for s_idx, sym in enumerate(panel.symbols):
            val = float(t_idx * 0.01 + s_idx * 0.001)
            rows_base.append({"trading_date": str(d_str), "symbol": sym, "m1_rv_ext": val})
            pert_val = val + 999.0 if t_idx > cutoff_idx else val
            rows_pert.append({"trading_date": str(d_str), "symbol": sym, "m1_rv_ext": pert_val})

    fut_prof = FuturesProfile(root=tmp_path)
    fut_builder = fut_prof.minute_pack_builder()
    base_pack_dir = tmp_path / "data" / "extensions" / "futures" / "micro_test"
    pert_pack_dir = tmp_path / "pert_pack"
    fut_builder.build_pack(
        base_pack_dir,
        pack_id="micro_test",
        daily_df=pl.DataFrame(rows_base),
        lag_days=1,
        asof_rule="T+1_open",
    )
    save_extension_pack(
        pert_pack_dir,
        ExtensionPackMeta(
            pack_id="micro_test",
            market="futures",
            fields=("m1_rv_ext",),
            lag_days=1,
            asof_rule="T+1_open",
        ),
        pl.DataFrame(rows_pert),
    )

    assert [m.pack_id for m in fut_prof.list_extensions()] == ["micro_test"]

    p_base = create_synthetic_futures_panel(n_days=50, seed=11)
    p_pert = create_synthetic_futures_panel(n_days=50, seed=11)
    fut_prof.attach_extensions(p_base, ["micro_test"])
    p_pert.attach_extension(pert_pack_dir)

    assert "m1_rv_ext" in p_base.fields
    assert np.all(np.isnan(p_base.fields["m1_rv_ext"][0, :]))  # lag_days=1 首行必为 NaN
    # T <= cutoff_idx + 1 (因 lag=1, cutoff_idx+1 使用的是 cutoff_idx 的值) 前必须完全 bit 等价
    np.testing.assert_array_equal(
        p_base.fields["m1_rv_ext"][: cutoff_idx + 2, :],
        p_pert.fields["m1_rv_ext"][: cutoff_idx + 2, :],
    )

    # Catalog 动态识别已挂载扩展特征
    cat = FactorCatalog(market="futures", panel=p_base)
    ext_entries = [v for v in cat.variables if v["name"] == "m1_rv_ext"]
    assert len(ext_entries) == 1
    assert ext_entries[0]["category"] == "extension"


def test_run_gp_and_run_ml_with_recipe_id(tmp_path: Path) -> None:
    """验证 run_gp.py --recipe <id> 与 run_ml.py --recipe <id> 完整控制面闭环。"""
    panel = create_synthetic_futures_panel(n_days=580, seed=42)
    _save_panel_as_parquet_dir(panel, tmp_path / "data" / "derived" / "futures" / "v1")
    db_path = tmp_path / "data" / "registry" / "factors.sqlite"
    runs_dir = tmp_path / "runs"

    # 构建一个 1m 扩展特征包，包含与 y_5 高度正相关的扩展特征 ext_alpha_sig
    ext_rows = []
    for t in range(panel.T):
        for n in range(panel.N):
            ext_rows.append(
                {
                    "trading_date": str(panel.dates[t]),
                    "symbol": panel.symbols[n],
                    "ext_alpha_sig": float(panel.fields["ry_ann"][t, n] * 2.0),
                }
            )
    fut_prof = FuturesProfile(root=tmp_path)
    fut_prof.minute_pack_builder().build_pack(
        tmp_path / "data" / "extensions" / "futures" / "micro_gp_ext",
        pack_id="micro_gp_ext",
        daily_df=pl.DataFrame(ext_rows),
        lag_days=0,
        asof_rule="T_close",
    )

    reg = FactorRegistry(db_path=db_path)
    fs_id = reg.save_feature_set(
        FeatureSetRecord(
            name="fs_gp_recipe_test",
            market="futures",
            base_fields=["close", "ry_ann"],
            extensions=["micro_gp_ext"],
            operators=["ts_mean", "cs_zscore", "add"],
            windows=[3, 5],
        )
    )

    gp_recipe = reg.create_recipe(
        market="futures",
        kind="gp",
        feature_set_id=fs_id,
        target_label="y_5",
        splits={
            "train": [str(panel.dates[0]), str(panel.dates[300])],
            "valid": [str(panel.dates[301]), str(panel.dates[-1])],
        },
        params={
            "population": 16,
            "generations": 2,
            "islands": 2,
            "migration": {"interval": 2, "k_individuals": 1},
            "limits": {"init_min_depth": 1, "init_max_depth": 3, "max_depth": 4, "max_nodes": 12},
            "admission": {"min_valid_ic_ir": 0.0, "require_sign_consistency": False},
        },
        seed=7,
    )

    # 1. 执行 run_gp --recipe <id>
    gp_args = run_gp_mod.parse_args(
        [
            "--recipe",
            gp_recipe.recipe_id,
            "--db",
            str(db_path),
            "--runs-dir",
            str(runs_dir),
            "--run-id",
            "gp_rcp_test_01",
        ]
    )
    rc_gp = run_gp_mod.run_gp_cli(gp_args, root_dir=tmp_path)
    assert rc_gp == 0

    gp_run_rec = reg.get_run("gp_rcp_test_01")
    assert gp_run_rec is not None
    assert gp_run_rec.status == "completed"
    assert gp_run_rec.recipe_id == gp_recipe.recipe_id
    assert gp_run_rec.market == "futures"

    candidates = reg.list_factors(
        market="futures", status="candidate", recipe_id=gp_recipe.recipe_id
    )
    assert len(candidates) >= 1

    # 晋升部分因子用于 ML 合成配方验证
    for f in candidates[:2]:
        assert f.id is not None
        reg.update_factor_status(f.id, status="accepted")

    # 2. 执行 run_ml --recipe <id>
    ml_recipe = reg.create_recipe(
        market="futures",
        kind="ml",
        feature_set_id=fs_id,
        target_label="y_5",
        params={
            "features_status": ["accepted"],
            "models": ["equal_weight", "ridge"],
            "walk_forward": {
                "start_predict_year": 2025,
                "end_predict_year": 2025,
                "min_train_years": 1,
                "purge_window": 5,
                "validation_size_days": 20,
            },
            "strict_min_years": False,
        },
        seed=7,
    )
    ml_args = run_ml_mod.parse_args(
        [
            "--recipe",
            ml_recipe.recipe_id,
            "--db",
            str(db_path),
            "--runs-dir",
            str(runs_dir),
            "--run-id",
            "ml_rcp_test_01",
        ]
    )
    rc_ml = run_ml_mod.run_ml_cli(ml_args, root_dir=tmp_path)
    assert rc_ml == 0

    ml_run_rec = reg.get_run("ml_rcp_test_01")
    assert ml_run_rec is not None
    assert ml_run_rec.status == "completed"
    assert ml_run_rec.recipe_id == ml_recipe.recipe_id
    assert (runs_dir / "ml_rcp_test_01" / "pred_panel.parquet").exists()
    assert (runs_dir / "ml_rcp_test_01" / "metrics.json").exists()
    assert (runs_dir / "ml_rcp_test_01" / "summary.md").exists()


def test_run_gp_stocks_recipe_isolation(tmp_path: Path) -> None:
    """验证 stocks 市场下 run_gp --recipe <id> 端到端闭环及与 futures 因子库严格隔离。"""
    panel_raw = create_synthetic_futures_panel(n_days=90, seed=19)
    panel = Panel(
        dates=panel_raw.dates,
        symbols=["000001.XSHE", "600519.XSHG", "000858.XSHE", "601318.XSHG", "300750.XSHE"],
        fields={
            "open": np.asfortranarray(panel_raw.fields["open"][:, :5]),
            "high": np.asfortranarray(panel_raw.fields["high"][:, :5]),
            "low": np.asfortranarray(panel_raw.fields["low"][:, :5]),
            "close": np.asfortranarray(panel_raw.fields["close"][:, :5]),
            "volume": np.asfortranarray(panel_raw.fields["volume"][:, :5]),
            "sector": np.asfortranarray(panel_raw.fields["sector"][:, :5]),
            "listed_days": np.asfortranarray(panel_raw.fields["listed_days"][:, :5]),
        },
        mask=np.asfortranarray(panel_raw.mask[:, :5]),
        labels={"y_5": np.asfortranarray(panel_raw.labels["y_5"][:, :5])},
        aux_fields={"open_raw": np.asfortranarray(panel_raw.fields["open"][:, :5])},
        meta={"panel_version": "stk_v1", "market": "stocks"},
    )
    _save_panel_as_parquet_dir(panel, tmp_path / "data" / "derived" / "stocks" / "v1")
    db_path = tmp_path / "data" / "registry" / "factors.sqlite"
    runs_dir = tmp_path / "runs"

    reg = FactorRegistry(db_path=db_path)
    # 先写入一条 futures 因子确认不会被 stocks GP 污染
    reg.register_factor(
        FactorRecord(
            market="futures",
            canon_expr="pct_change(close, 5)",
            canon_hash="fut_only_hash",
            source="seed",
            status="accepted",
        )
    )

    fs_id = reg.save_feature_set(
        FeatureSetRecord(
            name="fs_stocks_gp",
            market="stocks",
            base_fields=["close", "volume"],
            operators=["ts_mean", "cs_zscore", "pct_change"],
            windows=[3, 5],
        )
    )
    stk_recipe = reg.create_recipe(
        market="stocks",
        kind="gp",
        feature_set_id=fs_id,
        target_label="y_5",
        splits={
            "train": [str(panel.dates[0]), str(panel.dates[49])],
            "valid": [str(panel.dates[50]), str(panel.dates[-1])],
        },
        params={
            "population": 12,
            "generations": 1,
            "islands": 2,
            "migration": {"interval": 2, "k_individuals": 1},
            "limits": {"init_min_depth": 1, "init_max_depth": 3, "max_depth": 4, "max_nodes": 10},
            "admission": {"min_valid_ic_ir": 0.0, "require_sign_consistency": False},
        },
        seed=11,
    )

    gp_args = run_gp_mod.parse_args(
        [
            "--recipe",
            stk_recipe.recipe_id,
            "--db",
            str(db_path),
            "--runs-dir",
            str(runs_dir),
            "--run-id",
            "gp_stk_rcp_01",
        ]
    )
    assert run_gp_mod.run_gp_cli(gp_args, root_dir=tmp_path) == 0

    stk_cands = reg.list_factors(
        market="stocks", status="candidate", recipe_id=stk_recipe.recipe_id
    )
    assert len(stk_cands) >= 1
    assert len(reg.list_factors(market="futures")) == 1
