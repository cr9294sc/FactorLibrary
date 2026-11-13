"""FactorLibrary 研发工作台服务与 HTTP REST API 测试套件 (tests/python/test_workbench.py)。

涵盖：
1. `FactorWorkbenchService` 本地服务初始化与合成演示面板 / 真实面板切换；
2. Stage 0–1 目录检索与上下文接口；
3. Stage 2–3 双模态因子定义、声明式后处理与静态探针预热推导；
4. Stage 4–5 双层回测执行、不可变版本锁定与 HTML 诊断报告生成；
5. SQLite 因子库资产管理与生命周期流转；
6. 端到端流水线调用；
7. 标准库多线程 HTTP 服务端静态资源与 REST API 路由真实调用验证。
"""

from __future__ import annotations

import json
import threading
import urllib.error
import urllib.request
from pathlib import Path

import pytest
from factorlib.core.workbench import FactorWorkbenchService, create_synthetic_futures_panel
from factorlib.workbench_server import WorkbenchHTTPHandler, parse_args


def test_synthetic_futures_panel_creation():
    """验证合成期货面板具备完整 33 个标准字段、有效掩码与标签。"""
    panel = create_synthetic_futures_panel(n_days=60, seed=10)
    assert panel.shape == (60, 10)
    assert "close" in panel.fields
    assert "ry_ann" in panel.fields
    assert "wh_z60" in panel.fields
    assert "mr_net_ratio" in panel.fields
    assert "sector" in panel.fields
    assert "listed_days" in panel.fields
    assert "y_1" in panel.labels
    assert "y_5" in panel.labels
    assert panel.valid_mask.shape == (60, 10)


def test_workbench_service_context_and_catalog(tmp_path: Path):
    """测试工作台 Stage 0 上下文与 Stage 1 目录检索。"""
    db_path = tmp_path / "test_factors.sqlite"
    runs_dir = tmp_path / "runs"
    svc = FactorWorkbenchService(
        panel_dir=Path("data/derived/futures/v1"),
        db_path=db_path,
        runs_dir=runs_dir,
        force_synthetic=True,
    )

    ctx = svc.get_context()
    assert "panel_info" in ctx
    assert ctx["panel_info"]["n_days"] >= 40
    assert ctx["panel_info"]["n_symbols"] == 10
    assert "registry_summary" in ctx
    assert "authoring_templates" in ctx
    assert len(ctx["authoring_templates"]) >= 3

    # Stage 1 目录统计与检索
    stats = svc.catalog_stats(catalog="all")
    assert stats["total"] > 50

    search_res = svc.catalog_search(query="MAD", catalog="postprocess")
    assert len(search_res) >= 1
    assert search_res[0]["name"] == "MAD"

    info_res = svc.catalog_info(name="riskNeutMulti", catalog="postprocess")
    assert info_res is not None
    assert info_res["category"] == "neutralize"


def test_workbench_service_probe_and_draft(tmp_path: Path):
    """测试 Stage 2 因子草稿与 Stage 3 静态探针校验。"""
    db_path = tmp_path / "test_factors.sqlite"
    runs_dir = tmp_path / "runs"
    svc = FactorWorkbenchService(db_path=db_path, runs_dir=runs_dir, force_synthetic=True)

    # 1. DSL 表达式模式探针
    defn_expr = {
        "name": "test_carry",
        "factor_type": "expression",
        "expression": "ry_ann + pct_change(close, 5)",
        "postprocess": "baseUniMask,trdMask,MAD:3.0,indNeut,zScore,emadecay:3",
    }
    probe_res = svc.probe(defn_expr)
    assert probe_res["status"] == "ok"
    assert "emadecay" in probe_res["canon_expr"]
    assert probe_res["execution_requirements"]["total_required_lookback"] >= 5

    # 2. 保存草稿
    draft_res = svc.save_draft(defn_expr)
    assert draft_res["status"] == "ok"
    assert draft_res["factor_id"] is not None

    # 3. Python 因果沙箱代码探针
    defn_py = {
        "name": "test_py",
        "factor_type": "code",
        "code": "def compute(context):\n    c = context.history('close', 10)\n    return c[-1] / c[0] - 1.0\n",
        "postprocess": "standard",
    }
    probe_py = svc.probe(defn_py)
    assert probe_py["status"] == "ok"
    assert probe_py["execution_requirements"]["expression_lookback"] == 10


def test_workbench_service_run_workflow_dual_layer(tmp_path: Path):
    """测试 Stage 4–5 双层回测工作流、物化、版本锁定与 HTML 报告生成。"""
    db_path = tmp_path / "test_factors.sqlite"
    runs_dir = tmp_path / "runs"
    svc = FactorWorkbenchService(db_path=db_path, runs_dir=runs_dir, force_synthetic=True)

    payload = {
        "name": "alpha_workflow_test",
        "factor_type": "expression",
        "expression": "cs_zscore(ry_ann) + cs_zscore(pct_change(close, 10))",
        "postprocess": "standard",
        "materialize": True,
        "lock_version": True,
        "n_quantiles": 5,
    }

    res = svc.run_workflow(payload)
    assert res["status"] == "ok"
    assert res["factor_id"] is not None
    assert res["version_locked"] == 1

    # 验证 Layer 1 Schema v4 统计数据结构
    v4 = res["v4_stats"]
    assert v4["schema_version"] == "v4"
    assert "ic_analysis" in v4
    assert "group_analysis" in v4
    assert len(v4["group_analysis"]["summary"]) == 5
    assert "long_analysis" in v4
    assert "style_exposure" in v4

    # 验证 Layer 2 期货实盘逐日盯市回测数据结构
    bt = res["backtest"]
    assert "metrics" in bt
    assert "series" in bt
    assert len(bt["series"]["equity"]) > 0
    assert "cost_breakdown" in bt

    # 验证独立 HTML 报告生成
    report_file = Path(res["report_path"])
    assert report_file.exists()
    html_content = report_file.read_text(encoding="utf-8")
    assert "FactorLibrary" in html_content
    assert "alpha_workflow_test" in html_content

    # 再次回测同一因子，版本号应自动递增为 v2
    res2 = svc.run_workflow(payload)
    assert res2["version_locked"] == 2


def test_workbench_service_registry_lifecycle(tmp_path: Path):
    """测试 SQLite 因子库资产管理与状态机流转。"""
    db_path = tmp_path / "test_factors.sqlite"
    runs_dir = tmp_path / "runs"
    svc = FactorWorkbenchService(db_path=db_path, runs_dir=runs_dir, force_synthetic=True)

    # 1. 一键初始化种子因子
    init_res = svc.init_seed_factors()
    assert init_res["status"] == "ok"
    assert len(init_res["registered_ids"]) >= 1

    # 2. 查询因子列表
    factors = svc.list_factors()
    assert len(factors) >= 1
    fid = factors[0]["id"]

    # 3. 状态流转 candidate -> active
    up_res = svc.update_factor_status(fid, status="active", sign=1)
    assert up_res["status"] == "ok"
    assert up_res["factor"]["status"] == "active"

    # 4. 删除因子
    del_res = svc.delete_factor(fid)
    assert del_res["status"] == "ok"
    assert del_res["deleted"] is True


def test_workbench_service_pipeline_and_runs(tmp_path: Path):
    """测试端到端流水线调用与 runs 记录检索。"""
    db_path = tmp_path / "test_factors.sqlite"
    runs_dir = tmp_path / "runs"
    svc = FactorWorkbenchService(db_path=db_path, runs_dir=runs_dir, force_synthetic=True)

    res = svc.run_e2e_pipeline(
        {
            "factor_expr": "pct_change(close, 10)",
            "run_gp": False,
            "ml_model": "equal_weight",
            "skip_seed": True,
        }
    )
    assert res["status"] == "ok"
    assert res["run_id"] is not None

    runs = svc.list_runs()
    assert len(runs) >= 1
    assert runs[0]["run_id"] == res["run_id"]

    report_html = svc.get_run_report_html(res["run_id"])
    assert report_html is not None
    assert "FactorLibrary" in report_html


def test_workbench_http_server_endpoints(tmp_path: Path):
    """测试多线程 HTTP 服务端静态资源与 REST API 路由。"""
    db_path = tmp_path / "test_factors.sqlite"
    runs_dir = tmp_path / "runs"
    svc = FactorWorkbenchService(db_path=db_path, runs_dir=runs_dir, force_synthetic=True)

    from http.server import ThreadingHTTPServer

    WorkbenchHTTPHandler.service = svc
    # 随机绑定可用端口
    server = ThreadingHTTPServer(("127.0.0.1", 0), WorkbenchHTTPHandler)
    port = server.server_port

    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    base_url = f"http://127.0.0.1:{port}"

    try:
        # 1. 验证静态资源
        with urllib.request.urlopen(f"{base_url}/") as resp:
            assert resp.status == 200
            html = resp.read().decode("utf-8")
            assert "FactorLibrary" in html

        with urllib.request.urlopen(f"{base_url}/styles.css") as resp:
            assert resp.status == 200

        with urllib.request.urlopen(f"{base_url}/app.js") as resp:
            assert resp.status == 200

        # 2. 验证 REST API: GET /api/context
        with urllib.request.urlopen(f"{base_url}/api/context") as resp:
            assert resp.status == 200
            ctx = json.loads(resp.read().decode("utf-8"))
            assert "panel_info" in ctx

        # 3. 验证 REST API: POST /api/probe
        probe_data = json.dumps(
            {
                "name": "http_probe_test",
                "factor_type": "expression",
                "expression": "pct_change(close, 5)",
                "postprocess": "standard",
            }
        ).encode("utf-8")
        req = urllib.request.Request(
            f"{base_url}/api/probe",
            data=probe_data,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req) as resp:
            assert resp.status == 200
            rep = json.loads(resp.read().decode("utf-8"))
            assert rep["status"] == "ok"

        # 4. 空请求体调用 POST /api/factors/init-seeds 仍应正常返回 200
        init_req = urllib.request.Request(
            f"{base_url}/api/factors/init-seeds",
            data=b"",
            method="POST",
        )
        with urllib.request.urlopen(init_req) as resp:
            assert resp.status == 200
            init_rep = json.loads(resp.read().decode("utf-8"))
            assert init_rep["status"] == "ok"

        # 5. 非法 JSON 与非对象负载应返回 HTTP 400 Bad Request
        for method, endpoint, bad_payload in (
            ("POST", "/api/probe", b"{invalid json"),
            ("POST", "/api/probe", b"[1, 2, 3]"),
            ("PATCH", "/api/factors/1/status", b"{invalid json"),
        ):
            bad_req = urllib.request.Request(
                f"{base_url}{endpoint}",
                data=bad_payload,
                headers={"Content-Type": "application/json"},
                method=method,
            )
            with pytest.raises(urllib.error.HTTPError) as exc_info:
                urllib.request.urlopen(bad_req)
            assert exc_info.value.code == 400
            err_body = json.loads(exc_info.value.read().decode("utf-8"))
            assert "error" in err_body

    finally:
        server.shutdown()
        server.server_close()


def test_workbench_frontend_packaging_and_cli_args():
    """验证前端静态资源已收编至 factorlib 包内且服务端 CLI 参数解析正常。"""
    fdir = WorkbenchHTTPHandler.frontend_dir
    assert fdir.is_dir()
    assert fdir.parent.name == "factorlib"
    for asset in ("index.html", "styles.css", "app.js"):
        assert (fdir / asset).is_file()

    args = parse_args(
        ["--host", "0.0.0.0", "--port", "9001", "--synthetic", "--log-level", "DEBUG"]
    )
    assert args.host == "0.0.0.0"
    assert args.port == 9001
    assert args.synthetic is True
    assert args.log_level == "DEBUG"
