"""FactorLibrary 双市场研发工作台与 FastAPI REST API 测试套件 (tests/python/test_workbench.py)。

涵盖：
1. `FactorWorkbenchService` 双市场 (`futures` | `stocks`) 惰性加载、合成面板与真实面板切换；
2. P1 体检、真实 `1d` K 线浏览器、`1m` 扩展包挂载与 `FeatureSet` CRUD；
3. P2 双模态因子定义、声明式后处理与静态探针预热推导；
4. P3 双层回测执行、不可变版本锁定、HTML 报告及多因子 A/B 严格交集对比；
5. P4 配方冻结 (`ResearchRecipe`) 与独立后台子进程派发/轮询/取消闭环；
6. P5 因子库隔离、三关准入批量审计 (`core/admission_audit.py`) 与状态流转；
7. FastAPI `TestClient` 路由隔离、`/api/pipeline/run` 410 下线校验与 ES 模块静态资源打包检验。
"""

from __future__ import annotations

import time
from pathlib import Path

from factorlib.core.synthetic import (
    create_synthetic_futures_panel,
    create_synthetic_stocks_panel,
)
from factorlib.core.workbench import FactorWorkbenchService
from factorlib.workbench_server import create_app, parse_args, resolve_frontend_dir
from fastapi.testclient import TestClient


def test_synthetic_futures_and_stocks_panel_creation():
    """验证期货与 A 股内置合成面板均具备完整标准字段、掩码与因果标签。"""
    f_panel = create_synthetic_futures_panel(n_days=60, seed=10)
    assert f_panel.shape == (60, 10)
    assert "close" in f_panel.fields
    assert "ry_ann" in f_panel.fields
    assert "wh_z60" in f_panel.fields
    assert "y_5" in f_panel.labels

    s_panel = create_synthetic_stocks_panel(n_days=80, n_symbols=20, seed=20)
    assert s_panel.shape == (80, 20)
    assert "close" in s_panel.fields
    assert "turnover_rate" in s_panel.fields
    assert "sector" in s_panel.fields
    assert "y_5" in s_panel.labels
    assert "bench_ret" in s_panel.aux_fields


def test_workbench_service_dual_market_context_kline_and_featureset(tmp_path: Path):
    """测试双市场上下文隔离、1d 真实 K 线提取与 FeatureSet CRUD。"""
    db_path = tmp_path / "test_factors.sqlite"
    runs_dir = tmp_path / "runs"
    svc = FactorWorkbenchService(
        panel_dir=Path("data/derived/futures/v1"),
        db_path=db_path,
        runs_dir=runs_dir,
        force_synthetic=True,
    )

    f_ctx = svc.get_context(market="futures")
    s_ctx = svc.get_context(market="stocks")
    assert f_ctx["market"] == "futures"
    assert s_ctx["market"] == "stocks"
    assert f_ctx["panel_info"]["n_symbols"] == 10
    assert s_ctx["panel_info"]["n_symbols"] == 40
    assert f_ctx["execution_rules"]["allow_short"] is True
    assert s_ctx["execution_rules"]["allow_short"] is False

    # K 线读取真实 OHLCV 与组合归一化序列
    kl = svc.get_kline(market="stocks", range_bars=40)
    assert kl["market"] == "stocks"
    assert len(kl["bars"]) == 40
    assert len(kl["symbols"]) == 40
    assert len(kl["basket_series"]) >= 1

    # FeatureSet 市场隔离 CRUD
    f_sets = svc.list_feature_sets(market="futures")
    s_sets = svc.list_feature_sets(market="stocks")
    assert len(f_sets) >= 1 and f_sets[0]["market"] == "futures"
    assert len(s_sets) >= 1 and s_sets[0]["market"] == "stocks"

    saved = svc.save_feature_set(
        {
            "name": "fs_custom_alpha",
            "description": "自定义测试特征池",
            "base_fields": ["close", "volume"],
            "operators": ["ts_mean", "cs_zscore"],
            "windows": [5, 20],
        },
        market="stocks",
    )
    assert saved["feature_set"]["name"] == "fs_custom_alpha"
    assert saved["feature_set"]["market"] == "stocks"


def test_workbench_service_probe_and_draft(tmp_path: Path):
    """测试双模态因子探针与草稿保存。"""
    svc = FactorWorkbenchService(
        db_path=tmp_path / "test_factors.sqlite",
        runs_dir=tmp_path / "runs",
        force_synthetic=True,
    )

    defn_expr = {
        "name": "test_carry",
        "factor_type": "expression",
        "expression": "ry_ann + pct_change(close, 5)",
        "postprocess": "baseUniMask,trdMask,MAD:3.0,indNeut,zScore,emadecay:3",
    }
    probe_res = svc.probe(defn_expr, market="futures")
    assert probe_res["status"] == "ok"
    assert probe_res["execution_requirements"]["total_required_lookback"] >= 5

    draft_res = svc.save_draft(defn_expr, market="futures")
    assert draft_res["status"] == "ok"
    assert draft_res["factor_id"] is not None


def test_workbench_service_run_workflow_dual_layer_both_markets(tmp_path: Path):
    """测试期货与 A 股双层回测工作流、物化与版本锁定。"""
    svc = FactorWorkbenchService(
        db_path=tmp_path / "test_factors.sqlite",
        runs_dir=tmp_path / "runs",
        force_synthetic=True,
    )

    res_f = svc.run_workflow(
        {
            "name": "alpha_fut_test",
            "factor_type": "expression",
            "expression": "cs_zscore(ry_ann) + cs_zscore(pct_change(close, 10))",
            "postprocess": "standard",
            "n_quantiles": 5,
        },
        market="futures",
    )
    assert res_f["status"] == "ok"
    assert res_f["version_locked"] == 1
    assert res_f["v4_stats"]["schema_version"] == "v4"
    assert len(res_f["backtest"]["series"]["equity"]) > 0

    res_s = svc.run_workflow(
        {
            "name": "alpha_stk_test",
            "factor_type": "expression",
            "expression": "-cs_zscore(pct_change(close, 10))",
            "postprocess": "minimal",
            "n_quantiles": 5,
        },
        market="stocks",
    )
    assert res_s["status"] == "ok"
    assert res_s["market"] == "stocks"
    assert Path(res_s["report_path"]).exists()


def test_workbench_audit_and_ab_compare(tmp_path: Path):
    """测试三关准入审计 (关1去重 + 关2正交残差 + 关3消融) 与多因子 A/B 对比。"""
    svc = FactorWorkbenchService(
        db_path=tmp_path / "test_factors.sqlite",
        runs_dir=tmp_path / "runs",
        force_synthetic=True,
    )
    init_res = svc.init_seed_factors(market="futures")
    seed_ids = init_res["registered_ids"]
    assert len(seed_ids) >= 3

    # 激活前 2 个种子因子作为基线库，第 3 个作为 candidate 进行三关审计
    svc.update_factor_status(seed_ids[0], status="active", market="futures")
    svc.update_factor_status(seed_ids[1], status="active", market="futures")
    svc.update_factor_status(seed_ids[2], status="candidate", market="futures")

    audit_res = svc.batch_audit_factors(
        {"factor_ids": [seed_ids[2]], "horizon": 5}, market="futures"
    )
    assert audit_res["status"] == "ok"
    assert len(audit_res["results"]) == 1
    rep = audit_res["results"][0]
    assert "gate1_pass" in rep and "gate2_pass" in rep and "gate3_pass" in rep
    assert "residual_icir" in rep and "delta_sharpe" in rep

    # 多因子 A/B 对比 (2~6 因子)
    cmp_res = svc.compare_factors_ab({"factor_ids": seed_ids[:3], "horizon": 5}, market="futures")
    assert len(cmp_res["factors"]) == 3
    assert len(cmp_res["corr_matrix"]) == 3
    assert cmp_res["intersection_window"]["n_days"] > 0
    assert len(cmp_res["orthogonal_residual"]["cum_residual_ic"]) > 0


def test_workbench_fastapi_endpoints_and_async_subprocess(tmp_path: Path):
    """测试 FastAPI 单栈路由、市场隔离、410 废弃校验及异步子进程完整闭环。"""
    svc = FactorWorkbenchService(
        db_path=tmp_path / "api_factors.sqlite",
        runs_dir=tmp_path / "runs",
        force_synthetic=True,
    )
    app = create_app(service=svc)

    with TestClient(app) as client:
        # 1. 前端静态模块化资源
        for path in (
            "/",
            "/styles.css",
            "/main.js",
            "/api.js",
            "/state.js",
            "/router.js",
            "/components/charts.js",
            "/pages/data.js",
            "/pages/lab.js",
            "/pages/compare.js",
            "/pages/mining.js",
            "/pages/registry.js",
        ):
            r = client.get(path)
            assert r.status_code == 200, f"GET {path} failed: {r.status_code}"

        # 2. 旧阻塞接口 /api/pipeline/run 必须返回 410 Gone
        r_gone = client.post("/api/pipeline/run", json={})
        assert r_gone.status_code == 410

        # 3. 非法市场名校验
        r_bad_mk = client.get("/api/crypto/context")
        assert r_bad_mk.status_code == 422

        # 4. 非法 JSON 请求体返回 400
        r_bad_json = client.post(
            "/api/futures/probe",
            content=b"{invalid json",
            headers={"Content-Type": "application/json"},
        )
        assert r_bad_json.status_code == 400

        # 5. 双市场隔离上下文与探针
        r_f = client.get("/api/futures/context")
        r_s = client.get("/api/stocks/context")
        assert r_f.status_code == 200 and r_f.json()["market"] == "futures"
        assert r_s.status_code == 200 and r_s.json()["market"] == "stocks"

        # 6. 异步后台子进程派发 (DL Alpha Phase 2 基线) -> 轮询完成 -> 产出 candidate
        r_launch = client.post(
            "/api/futures/runs",
            json={
                "kind": "dl",
                "target_label": "y_5",
                "seed": 42,
                "params": {"n_folds": 2, "n_seeds": 2, "purge_window": 5},
            },
        )
        assert r_launch.status_code == 202
        launch_data = r_launch.json()
        run_id = launch_data["run_id"]
        assert launch_data["pid"] > 0
        assert launch_data["recipe_id"].startswith("rcp_")

        # 轮询等待独立子进程结束
        final_detail = None
        for _ in range(40):
            r_det = client.get(f"/api/futures/runs/{run_id}")
            assert r_det.status_code == 200
            det = r_det.json()
            if det["status"] in ("completed", "failed", "cancelled"):
                final_detail = det
                break
            time.sleep(0.25)

        assert final_detail is not None, "后台异步子进程未在预期时间内完成"
        assert final_detail["status"] == "completed", final_detail.get("log_tail")
        assert len(final_detail["produced_factors"]) >= 1
        assert final_detail["produced_factors"][0]["status"] == "candidate"
        assert final_detail["produced_factors"][0]["factor_type"] == "dl_alpha"

        # 确认 stocks 市场下的 run 与 factor 均与 futures 严格隔离
        assert all(x["run_id"] != run_id for x in client.get("/api/stocks/runs").json())
        assert client.get("/api/stocks/factors").json() == []


def test_workbench_frontend_packaging_and_cli_args():
    """验证前端 ES 模块静态资源已收编至 factorlib 包内且服务端 CLI 参数解析正常。"""
    fdir = resolve_frontend_dir()
    assert fdir.is_dir()
    assert fdir.parent.name == "factorlib"
    for asset in (
        "index.html",
        "styles.css",
        "main.js",
        "api.js",
        "state.js",
        "router.js",
        "components/charts.js",
        "pages/data.js",
        "pages/lab.js",
        "pages/compare.js",
        "pages/mining.js",
        "pages/registry.js",
    ):
        assert (fdir / asset).is_file(), f"缺少前端资源 {asset}"

    args = parse_args(
        [
            "--host",
            "0.0.0.0",
            "--port",
            "9001",
            "--market",
            "stocks",
            "--synthetic",
            "--log-level",
            "DEBUG",
        ]
    )
    assert args.host == "0.0.0.0"
    assert args.port == 9001
    assert args.market == "stocks"
    assert args.synthetic is True
    assert args.log_level == "DEBUG"
