#!/usr/bin/env python3
"""FactorLibrary 研发工作台 FastAPI 单栈服务端 (factorlib/workbench_server.py)。

遵循架构重构计划 M5.3 与 Phase 2 规范：
- 统一采用 FastAPI + Uvicorn 单栈 HTTP 服务，托管零构建原生 ES 模块前端；
- 所有业务 API 均强制带有 `{market}` (`futures` | `stocks`) 路径前缀，实现双市场严格隔离；
- 废弃阻塞式 `/api/pipeline/run`（返回 HTTP 410 Gone 并指引至 `POST /api/{market}/runs`），
  全面升级为独立后台子进程派发与实时进度/日志看板；
- 打通 `FeatureSet` CRUD、`1m` 扩展包挂载、`1d` K 线浏览、多因子 A/B 同台对比与三关准入批量审计。
"""

from __future__ import annotations

import argparse
import logging
import sys
from enum import StrEnum
from pathlib import Path
from typing import Any

import uvicorn
from fastapi import FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from factorlib.core.workbench import FactorWorkbenchService

DEFAULT_SERVICE: FactorWorkbenchService | None = None


class MarketName(StrEnum):
    """支持的研究市场域枚举。"""

    futures = "futures"
    stocks = "stocks"


class PanelSwitchRequest(BaseModel):
    mode: str = "auto"
    panel_dir: str | None = None


class ExtensionMountRequest(BaseModel):
    pack_id: str
    action: str = "mount"


class FeatureSetSaveRequest(BaseModel):
    name: str
    description: str = ""
    base_fields: list[str] = Field(default_factory=list)
    extensions: list[str] = Field(default_factory=list)
    operators: list[str] | None = None
    windows: list[int] | None = None
    aux_factor_ids: list[int] = Field(default_factory=list)
    default_postprocess: list[dict[str, Any]] = Field(default_factory=list)
    upsert: bool = True


class FactorStatusRequest(BaseModel):
    status: str = "accepted"
    sign: int | None = None


class BatchFactorStatusRequest(BaseModel):
    factor_ids: list[int] = Field(default_factory=list)
    status: str = "accepted"
    sign: int | None = None


class BatchAuditRequest(BaseModel):
    factor_ids: list[int] = Field(default_factory=list)
    horizon: int = 5
    min_valid_icir: float = 0.20
    max_turnover: float = 0.85
    max_corr_limit: float = 0.60
    auto_promote: bool = False


class CompareFactorsRequest(BaseModel):
    factor_ids: list[int] = Field(default_factory=list)
    horizon: int = 5


class RecipeCreateRequest(BaseModel):
    kind: str = "dl"
    name: str = ""
    feature_set_id: int | None = None
    feature_set: dict[str, Any] | None = None
    target_label: str = "y_5"
    splits: dict[str, Any] = Field(default_factory=dict)
    params: dict[str, Any] = Field(default_factory=dict)
    extensions: list[str] | None = None
    seed: int = 42


class RunLaunchRequest(BaseModel):
    kind: str = "dl"
    recipe_id: str | None = None
    name: str = ""
    feature_set_id: int | None = None
    feature_set: dict[str, Any] | None = None
    target_label: str = "y_5"
    splits: dict[str, Any] = Field(default_factory=dict)
    params: dict[str, Any] = Field(default_factory=dict)
    extensions: list[str] | None = None
    seed: int = 42


def resolve_frontend_dir() -> Path:
    """定位前端静态资源目录（优先包内路径，兼容源码根目录回退）。"""
    pkg_frontend = Path(__file__).resolve().parent / "frontend"
    if pkg_frontend.is_dir():
        return pkg_frontend
    return Path(__file__).resolve().parents[2] / "frontend"


def get_workbench_service(
    panel_dir: str | Path = "data/derived/futures/v1",
    db_path: str | Path = "data/registry/factors.sqlite",
    runs_dir: str | Path = "runs",
    force_synthetic: bool = False,
    market: str = "futures",
) -> FactorWorkbenchService:
    global DEFAULT_SERVICE
    if DEFAULT_SERVICE is None:
        DEFAULT_SERVICE = FactorWorkbenchService(
            panel_dir=panel_dir,
            db_path=db_path,
            runs_dir=runs_dir,
            force_synthetic=force_synthetic,
            market=market,
        )
    return DEFAULT_SERVICE


def set_workbench_service(service: FactorWorkbenchService | None) -> None:
    """替换全局默认工作台服务实例（便于 CLI 启动与测试注入）。"""
    global DEFAULT_SERVICE
    DEFAULT_SERVICE = service


async def _read_json_dict(req: Request) -> dict[str, Any]:
    raw = await req.body()
    if not raw:
        return {}
    import json

    try:
        parsed = json.loads(raw.decode("utf-8"))
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Invalid JSON request body: {exc}") from exc
    if not isinstance(parsed, dict):
        raise HTTPException(status_code=400, detail="JSON request body must be an object")
    return parsed


def create_app(service: FactorWorkbenchService | None = None) -> FastAPI:
    """构建并配置 FactorLibrary 双市场工作台 FastAPI 应用实例。"""
    if service is not None:
        set_workbench_service(service)

    def _svc() -> FactorWorkbenchService:
        return service if service is not None else get_workbench_service()

    fastapi_app = FastAPI(
        title="FactorLibrary Workbench API",
        version="0.3.0",
        description="FactorLibrary 双市场量化因子混合投研工作台 API (Phase 2)",
    )
    fastapi_app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @fastapi_app.exception_handler(ValueError)
    async def _value_error_handler(_req: Request, exc: ValueError) -> JSONResponse:
        return JSONResponse(status_code=400, content={"error": str(exc)})

    # ------------------------------------------------------------------
    # M5.3 废弃路由提示: 旧同步阻塞式 /api/pipeline/run 返回 410 Gone
    # ------------------------------------------------------------------
    @fastapi_app.api_route("/api/pipeline/run", methods=["GET", "POST"])
    async def deprecated_pipeline_run() -> JSONResponse:
        return JSONResponse(
            status_code=status.HTTP_410_GONE,
            content={
                "error": "阻塞式 /api/pipeline/run 已下线 (HTTP 410 Gone)。请改用异步子进程派发接口 POST /api/{market}/runs。",
                "replacement": "/api/{market}/runs",
            },
        )

    # ------------------------------------------------------------------
    # 市场隔离 REST API (/api/{market}/...)
    # ------------------------------------------------------------------
    @fastapi_app.get("/api/{market}/context")
    def api_context(market: MarketName) -> dict[str, Any]:
        return _svc().get_context(market=market.value)

    @fastapi_app.post("/api/{market}/panel/switch")
    def api_switch_panel(market: MarketName, body: PanelSwitchRequest) -> dict[str, Any]:
        return _svc().switch_panel_source(
            mode=body.mode, panel_dir=body.panel_dir, market=market.value
        )

    @fastapi_app.get("/api/{market}/panel/kline")
    def api_panel_kline(
        market: MarketName,
        symbol: str | None = None,
        range_bars: int = 120,
        adj: str = "hfq",
        basket: str = "",
    ) -> dict[str, Any]:
        basket_list = [s.strip() for s in basket.split(",") if s.strip()] if basket else None
        return _svc().get_kline(
            market=market.value,
            symbol=symbol,
            range_bars=range_bars,
            adj=adj,
            basket=basket_list,
        )

    @fastapi_app.get("/api/{market}/extensions")
    def api_list_extensions(market: MarketName) -> list[dict[str, Any]]:
        return _svc().list_extensions(market=market.value)

    @fastapi_app.post("/api/{market}/extensions/mount")
    def api_mount_extension(market: MarketName, body: ExtensionMountRequest) -> dict[str, Any]:
        return _svc().mount_extension(pack_id=body.pack_id, action=body.action, market=market.value)

    @fastapi_app.get("/api/{market}/feature-sets")
    def api_list_feature_sets(market: MarketName) -> list[dict[str, Any]]:
        return _svc().list_feature_sets(market=market.value)

    @fastapi_app.post("/api/{market}/feature-sets")
    def api_save_feature_set(market: MarketName, body: FeatureSetSaveRequest) -> dict[str, Any]:
        return _svc().save_feature_set(body.model_dump(), market=market.value)

    @fastapi_app.delete("/api/{market}/feature-sets/{fs_id}")
    def api_delete_feature_set(market: MarketName, fs_id: int) -> dict[str, Any]:
        del market
        return _svc().delete_feature_set(fs_id)

    @fastapi_app.get("/api/{market}/catalog/stats")
    def api_catalog_stats(market: MarketName, catalog: str = "all") -> dict[str, Any]:
        return _svc().catalog_stats(catalog=catalog, market=market.value)

    @fastapi_app.get("/api/{market}/catalog/search")
    def api_catalog_search(
        market: MarketName,
        query: str = "",
        catalog: str = "all",
        category: str = "",
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        return _svc().catalog_search(
            query=query,
            catalog=catalog,
            category=category,
            limit=limit,
            market=market.value,
        )

    @fastapi_app.get("/api/{market}/catalog/info")
    def api_catalog_info(market: MarketName, name: str, catalog: str = "all") -> dict[str, Any]:
        entry = _svc().catalog_info(name=name, catalog=catalog, market=market.value)
        if entry is None:
            raise HTTPException(status_code=404, detail=f"Entry not found: {name}")
        return entry

    @fastapi_app.post("/api/{market}/probe")
    async def api_probe(market: MarketName, req: Request) -> dict[str, Any]:
        body = await _read_json_dict(req)
        return _svc().probe(body, market=market.value)

    @fastapi_app.post("/api/{market}/factors/draft")
    async def api_save_draft(market: MarketName, req: Request) -> dict[str, Any]:
        body = await _read_json_dict(req)
        return _svc().save_draft(body, market=market.value)

    @fastapi_app.post("/api/{market}/workflow/run")
    async def api_run_workflow(market: MarketName, req: Request) -> dict[str, Any]:
        body = await _read_json_dict(req)
        return _svc().run_workflow(body, market=market.value)

    @fastapi_app.get("/api/{market}/factors")
    def api_list_factors(
        market: MarketName,
        status: str | None = None,
        source: str | None = None,
        mode: str | None = None,
        factor_type: str | None = None,
        source_run: str | None = None,
        min_icir: float | None = None,
        split: str = "train",
        horizon: int = 5,
    ) -> list[dict[str, Any]]:
        return _svc().list_factors(
            status=status,
            source=source,
            mode=mode,
            factor_type=factor_type,
            source_run=source_run,
            min_icir=min_icir,
            split=split,
            horizon=horizon,
            market=market.value,
        )

    @fastapi_app.get("/api/{market}/factors/{factor_id}")
    def api_get_factor(market: MarketName, factor_id: int) -> dict[str, Any]:
        res = _svc().get_factor_detail(factor_id, market=market.value)
        if res is None:
            raise HTTPException(status_code=404, detail="Factor not found")
        return res

    @fastapi_app.patch("/api/{market}/factors/{factor_id}/status")
    def api_update_status(
        market: MarketName, factor_id: int, body: FactorStatusRequest
    ) -> dict[str, Any]:
        if body.sign is not None and body.sign not in (-1, 1):
            raise HTTPException(status_code=400, detail="sign must be -1 or 1")
        return _svc().update_factor_status(
            factor_id=factor_id,
            status=body.status,
            sign=body.sign,
            market=market.value,
        )

    @fastapi_app.post("/api/{market}/factors/batch-status")
    def api_batch_status(market: MarketName, body: BatchFactorStatusRequest) -> dict[str, Any]:
        return _svc().batch_update_factor_status(
            factor_ids=body.factor_ids,
            status=body.status,
            sign=body.sign,
            market=market.value,
        )

    @fastapi_app.post("/api/{market}/factors/audit")
    def api_batch_audit(market: MarketName, body: BatchAuditRequest) -> dict[str, Any]:
        return _svc().batch_audit_factors(body.model_dump(), market=market.value)

    @fastapi_app.delete("/api/{market}/factors/{factor_id}")
    def api_delete_factor(market: MarketName, factor_id: int) -> dict[str, Any]:
        return _svc().delete_factor(factor_id, market=market.value)

    @fastapi_app.post("/api/{market}/factors/init-seeds")
    def api_init_seeds(market: MarketName) -> dict[str, Any]:
        return _svc().init_seed_factors(market=market.value)

    @fastapi_app.post("/api/{market}/compare")
    def api_compare_factors(market: MarketName, body: CompareFactorsRequest) -> dict[str, Any]:
        return _svc().compare_factors_ab(body.model_dump(), market=market.value)

    @fastapi_app.get("/api/{market}/recipes")
    def api_list_recipes(
        market: MarketName, kind: str | None = None, limit: int = 50
    ) -> list[dict[str, Any]]:
        return _svc().list_recipes(market=market.value, kind=kind, limit=limit)

    @fastapi_app.post("/api/{market}/recipes")
    def api_create_recipe(market: MarketName, body: RecipeCreateRequest) -> dict[str, Any]:
        return _svc().create_recipe(body.model_dump(), market=market.value)

    @fastapi_app.get("/api/{market}/runs")
    def api_list_runs(market: MarketName) -> list[dict[str, Any]]:
        return _svc().list_runs(market=market.value)

    @fastapi_app.post("/api/{market}/runs", status_code=status.HTTP_202_ACCEPTED)
    def api_launch_run(market: MarketName, body: RunLaunchRequest) -> dict[str, Any]:
        return _svc().launch_run(body.model_dump(), market=market.value)

    @fastapi_app.get("/api/{market}/runs/{run_id}")
    def api_get_run_detail(market: MarketName, run_id: str) -> dict[str, Any]:
        res = _svc().get_run_detail(run_id, market=market.value)
        if res is None:
            raise HTTPException(status_code=404, detail="Run not found")
        return res

    @fastapi_app.post("/api/{market}/runs/{run_id}/cancel")
    def api_cancel_run(market: MarketName, run_id: str) -> dict[str, Any]:
        return _svc().cancel_run(run_id, market=market.value)

    @fastapi_app.get("/api/{market}/runs/{run_id}/report", response_class=HTMLResponse)
    def api_run_report(market: MarketName, run_id: str) -> str:
        del market
        html = _svc().get_run_report_html(run_id)
        if not html:
            raise HTTPException(status_code=404, detail="Report not found")
        return html

    frontend_path = resolve_frontend_dir()
    if frontend_path.exists():
        fastapi_app.mount(
            "/", StaticFiles(directory=str(frontend_path), html=True), name="frontend"
        )

    return fastapi_app


# 供 `uvicorn factorlib.workbench_server:app` 直接挂载的默认 ASGI 实例
app = create_app()


def parse_args(args: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="FactorLibrary 双市场量化因子研发工作台服务启动器 (FastAPI + Uvicorn)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--host", type=str, default="127.0.0.1", help="监听地址")
    parser.add_argument("--port", type=int, default=8080, help="监听端口")
    parser.add_argument(
        "--market",
        type=str,
        default="futures",
        choices=["futures", "stocks"],
        help="默认预热市场域",
    )
    parser.add_argument(
        "--panel-dir",
        type=Path,
        default=Path("data/derived/futures/v1"),
        help="标准研究面板目录",
    )
    parser.add_argument(
        "--db",
        type=Path,
        default=Path("data/registry/factors.sqlite"),
        help="SQLite 因子库路径",
    )
    parser.add_argument(
        "--runs-dir",
        type=Path,
        default=Path("runs"),
        help="实验产出目录",
    )
    parser.add_argument(
        "--synthetic",
        action="store_true",
        help="强制启用内置合成演示面板 (不读取本地 Parquet)",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="日志输出级别",
    )
    return parser.parse_args(args)


def main(cli_args: list[str] | None = None) -> int:
    args = parse_args(cli_args)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="[%(asctime)s] [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    try:
        svc = FactorWorkbenchService(
            panel_dir=args.panel_dir,
            db_path=args.db,
            runs_dir=args.runs_dir,
            force_synthetic=args.synthetic,
            market=args.market,
        )
        set_workbench_service(svc)
        logging.info("=" * 70)
        logging.info(" FactorLibrary 双市场研发工作台已就绪 (FastAPI + Uvicorn)")
        logging.info(" 工作台 Web 界面: http://%s:%d", args.host, args.port)
        logging.info(" REST API 文档  : http://%s:%d/docs", args.host, args.port)
        logging.info("=" * 70)
        uvicorn.run(
            app,
            host=args.host,
            port=args.port,
            log_level=args.log_level.lower(),
        )
        return 0
    except KeyboardInterrupt:
        logging.info("收到中断信号，正在安全关闭工作台服务...")
        return 0
    except Exception as exc:
        logging.exception("工作台服务运行失败: %s", exc)
        return 1


if __name__ == "__main__":
    sys.exit(main())
