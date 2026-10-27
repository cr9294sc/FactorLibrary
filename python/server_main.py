#!/usr/bin/env python3
"""FactorLibrary 研发工作台 REST API 与 Web 服务端 (python/server_main.py)。

支持双模启动：
1. 标准库模式 (零外部依赖，内置 ThreadingHTTPServer，在基础 Python 虚拟环境中开箱即用)；
2. FastAPI 模式 (若已安装 fastapi 与 uvicorn，自动挂载并暴露全局 ASGI 应用 `app`)。
托管前端单页应用 (frontend/index.html) 并提供全套 6 阶段量化因子研发 RESTful API。
"""

from __future__ import annotations

import argparse
import json
import logging
import mimetypes
import sys
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

# 确保 python/ 目录可正常导入
sys.path.insert(0, str(Path(__file__).resolve().parent))

from factorlib.core.workbench import FactorWorkbenchService

# 全局默认服务实例
DEFAULT_SERVICE: FactorWorkbenchService | None = None


def get_workbench_service(
    panel_dir: str | Path = "data/derived/futures/v1",
    db_path: str | Path = "data/registry/factors.sqlite",
    runs_dir: str | Path = "runs",
    force_synthetic: bool = False,
) -> FactorWorkbenchService:
    global DEFAULT_SERVICE
    if DEFAULT_SERVICE is None:
        DEFAULT_SERVICE = FactorWorkbenchService(
            panel_dir=panel_dir,
            db_path=db_path,
            runs_dir=runs_dir,
            force_synthetic=force_synthetic,
        )
    return DEFAULT_SERVICE


def _json_response(handler: BaseHTTPRequestHandler, data: Any, status: int = 200) -> None:
    body = json.dumps(data, ensure_ascii=False, indent=2).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json; charset=utf-8")
    handler.send_header("Content-Length", str(len(body)))
    handler.send_header("Access-Control-Allow-Origin", "*")
    handler.send_header("Access-Control-Allow-Headers", "Content-Type, Authorization")
    handler.send_header("Access-Control-Allow-Methods", "GET, POST, PATCH, DELETE, OPTIONS")
    handler.end_headers()
    handler.wfile.write(body)


def _html_response(handler: BaseHTTPRequestHandler, html_str: str, status: int = 200) -> None:
    body = html_str.encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "text/html; charset=utf-8")
    handler.send_header("Content-Length", str(len(body)))
    handler.end_headers()
    handler.wfile.write(body)


def _serve_static_file(handler: BaseHTTPRequestHandler, file_path: Path, status: int = 200) -> None:
    if not file_path.exists() or not file_path.is_file():
        _json_response(handler, {"error": "File not found"}, status=404)
        return
    ctype, _ = mimetypes.guess_type(str(file_path))
    if ctype is None:
        ctype = "application/octet-stream"
    body = file_path.read_bytes()
    handler.send_response(status)
    handler.send_header("Content-Type", f"{ctype}; charset=utf-8" if "text" in ctype else ctype)
    handler.send_header("Content-Length", str(len(body)))
    handler.end_headers()
    handler.wfile.write(body)


class WorkbenchHTTPHandler(BaseHTTPRequestHandler):
    """基于标准库的可并发 HTTP 请求处理器。"""

    service: FactorWorkbenchService = None  # type: ignore[assignment]
    frontend_dir: Path = Path(__file__).resolve().parent.parent / "frontend"

    def log_message(self, format: str, *args: Any) -> None:
        # 重定向至 Python 标准 logging 模块
        logging.info("%s - %s", self.address_string(), format % args)

    def do_OPTIONS(self) -> None:
        self.send_response(HTTPStatus.NO_CONTENT)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "Content-Type, Authorization")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, PATCH, DELETE, OPTIONS")
        self.end_headers()

    def _read_json_body(self) -> dict[str, Any]:
        clen = int(self.headers.get("Content-Length", 0))
        if clen <= 0:
            return {}
        raw = self.rfile.read(clen).decode("utf-8")
        try:
            return json.loads(raw)
        except Exception:
            return {}

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        path = parsed.path
        qs = parse_qs(parsed.query)

        # 1. 静态前端资源路由
        if path in ("/", "/index.html"):
            _serve_static_file(self, self.frontend_dir / "index.html")
            return
        if path in ("/styles.css", "/app.js"):
            _serve_static_file(self, self.frontend_dir / path.lstrip("/"))
            return

        # 2. REST API 路由
        try:
            if path == "/api/context":
                _json_response(self, self.service.get_context())
            elif path == "/api/catalog/stats":
                cat = qs.get("catalog", ["all"])[0]
                _json_response(self, self.service.catalog_stats(catalog=cat))
            elif path == "/api/catalog/search":
                q = qs.get("query", [""])[0]
                cat = qs.get("catalog", ["all"])[0]
                category = qs.get("category", [""])[0]
                limit = int(qs.get("limit", [100])[0])
                _json_response(
                    self,
                    self.service.catalog_search(
                        query=q, catalog=cat, category=category, limit=limit
                    ),
                )
            elif path == "/api/catalog/info":
                name = qs.get("name", [""])[0]
                cat = qs.get("catalog", ["all"])[0]
                entry = self.service.catalog_info(name=name, catalog=cat)
                if entry is None:
                    _json_response(self, {"error": f"Entry not found: {name}"}, status=404)
                else:
                    _json_response(self, entry)
            elif path == "/api/factors":
                status = qs.get("status", [None])[0]
                source = qs.get("source", [None])[0]
                mode = qs.get("mode", [None])[0]
                min_icir = float(qs["min_icir"][0]) if "min_icir" in qs else None
                split = qs.get("split", ["train"])[0]
                horizon = int(qs.get("horizon", [5])[0])
                factors = self.service.list_factors(
                    status=status,
                    source=source,
                    mode=mode,
                    min_icir=min_icir,
                    split=split,
                    horizon=horizon,
                )
                _json_response(self, factors)
            elif path.startswith("/api/factors/"):
                sub = path[len("/api/factors/") :]
                if "/" not in sub and sub.isdigit():
                    detail = self.service.get_factor_detail(int(sub))
                    if detail is None:
                        _json_response(self, {"error": "Factor not found"}, status=404)
                    else:
                        _json_response(self, detail)
                else:
                    _json_response(self, {"error": "Not found"}, status=404)
            elif path == "/api/runs":
                _json_response(self, self.service.list_runs())
            elif path.startswith("/api/runs/") and path.endswith("/report"):
                parts = path.strip("/").split("/")
                if len(parts) == 4 and parts[1] == "runs":
                    run_id = parts[2]
                    html = self.service.get_run_report_html(run_id)
                    if html:
                        _html_response(self, html)
                    else:
                        _json_response(self, {"error": "Report not found"}, status=404)
                else:
                    _json_response(self, {"error": "Not found"}, status=404)
            else:
                _json_response(self, {"error": "Not found"}, status=404)
        except Exception as exc:
            logging.exception("处理 GET 请求失败: %s", exc)
            _json_response(self, {"error": str(exc)}, status=500)

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        path = parsed.path
        body = self._read_json_body()

        try:
            if path == "/api/probe":
                _json_response(self, self.service.probe(body))
            elif path == "/api/factors/draft":
                res = self.service.save_draft(body)
                _json_response(self, res)
            elif path == "/api/workflow/run":
                res = self.service.run_workflow(body)
                _json_response(self, res)
            elif path == "/api/factors/init-seeds":
                res = self.service.init_seed_factors()
                _json_response(self, res)
            elif path == "/api/panel/switch":
                mode = str(body.get("mode", "auto"))
                p_dir = body.get("panel_dir")
                res = self.service.switch_panel_source(mode=mode, panel_dir=p_dir)
                _json_response(self, res)
            elif path == "/api/pipeline/run":
                res = self.service.run_e2e_pipeline(body)
                _json_response(self, res)
            else:
                _json_response(self, {"error": "Not found"}, status=404)
        except Exception as exc:
            logging.exception("处理 POST 请求失败: %s", exc)
            _json_response(self, {"error": str(exc)}, status=500)

    def do_PATCH(self) -> None:
        parsed = urlparse(self.path)
        path = parsed.path
        body = self._read_json_body()

        try:
            if path.startswith("/api/factors/") and path.endswith("/status"):
                parts = path.strip("/").split("/")
                fid = int(parts[2])
                status = str(body.get("status", "accepted"))
                sign = int(body["sign"]) if "sign" in body and body["sign"] in (-1, 1) else None
                res = self.service.update_factor_status(fid, status=status, sign=sign)
                _json_response(self, res)
            else:
                _json_response(self, {"error": "Not found"}, status=404)
        except Exception as exc:
            logging.exception("处理 PATCH 请求失败: %s", exc)
            _json_response(self, {"error": str(exc)}, status=500)

    def do_DELETE(self) -> None:
        parsed = urlparse(self.path)
        path = parsed.path

        try:
            if path.startswith("/api/factors/"):
                sub = path[len("/api/factors/") :]
                if sub.isdigit():
                    fid = int(sub)
                    res = self.service.delete_factor(fid)
                    _json_response(self, res)
                else:
                    _json_response(self, {"error": "Not found"}, status=404)
            else:
                _json_response(self, {"error": "Not found"}, status=404)
        except Exception as exc:
            logging.exception("处理 DELETE 请求失败: %s", exc)
            _json_response(self, {"error": str(exc)}, status=500)


def run_std_server(
    service: FactorWorkbenchService,
    host: str = "127.0.0.1",
    port: int = 8080,
) -> ThreadingHTTPServer:
    """启动零额外依赖的标准多线程 HTTP 服务。"""
    WorkbenchHTTPHandler.service = service
    server = ThreadingHTTPServer((host, port), WorkbenchHTTPHandler)
    logging.info("=" * 70)
    logging.info(" FactorLibrary 研发工作台已就绪 (Python Standard HTTP Server)")
    logging.info(" 工作台 Web 界面: http://%s:%d", host, port)
    logging.info(" REST API 入口  : http://%s:%d/api/context", host, port)
    logging.info("=" * 70)
    return server


# =============================================================================
# FastAPI 可选导出支持 (若环境安装了 fastapi，可直接由 uvicorn 挂载)
# =============================================================================
try:
    from fastapi import FastAPI, HTTPException, Request
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import HTMLResponse
    from fastapi.staticfiles import StaticFiles

    app = FastAPI(
        title="FactorLibrary Workbench API",
        version="0.2.0",
        description="FactorLibrary 6 阶段期货量化因子研发与实盘回测工作台 API",
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/api/context")
    def api_context() -> dict[str, Any]:
        return get_workbench_service().get_context()

    @app.get("/api/catalog/stats")
    def api_catalog_stats(catalog: str = "all") -> dict[str, Any]:
        return get_workbench_service().catalog_stats(catalog=catalog)

    @app.get("/api/catalog/search")
    def api_catalog_search(
        query: str = "", catalog: str = "all", category: str = "", limit: int = 100
    ) -> list[dict[str, Any]]:
        return get_workbench_service().catalog_search(
            query=query, catalog=catalog, category=category, limit=limit
        )

    @app.get("/api/catalog/info")
    def api_catalog_info(name: str, catalog: str = "all") -> dict[str, Any]:
        entry = get_workbench_service().catalog_info(name=name, catalog=catalog)
        if entry is None:
            raise HTTPException(status_code=404, detail="Entry not found")
        return entry

    @app.post("/api/probe")
    async def api_probe(req: Request) -> dict[str, Any]:
        body = await req.json()
        return get_workbench_service().probe(body)

    @app.post("/api/factors/draft")
    async def api_save_draft(req: Request) -> dict[str, Any]:
        body = await req.json()
        return get_workbench_service().save_draft(body)

    @app.post("/api/workflow/run")
    async def api_run_workflow(req: Request) -> dict[str, Any]:
        body = await req.json()
        return get_workbench_service().run_workflow(body)

    @app.get("/api/factors")
    def api_list_factors(
        status: str | None = None,
        source: str | None = None,
        mode: str | None = None,
        min_icir: float | None = None,
        split: str = "train",
        horizon: int = 5,
    ) -> list[dict[str, Any]]:
        return get_workbench_service().list_factors(
            status=status,
            source=source,
            mode=mode,
            min_icir=min_icir,
            split=split,
            horizon=horizon,
        )

    @app.get("/api/factors/{factor_id}")
    def api_get_factor(factor_id: int) -> dict[str, Any]:
        res = get_workbench_service().get_factor_detail(factor_id)
        if res is None:
            raise HTTPException(status_code=404, detail="Factor not found")
        return res

    @app.patch("/api/factors/{factor_id}/status")
    async def api_update_status(factor_id: int, req: Request) -> dict[str, Any]:
        body = await req.json()
        return get_workbench_service().update_factor_status(
            factor_id=factor_id,
            status=body.get("status", "accepted"),
            sign=body.get("sign"),
        )

    @app.delete("/api/factors/{factor_id}")
    def api_delete_factor(factor_id: int) -> dict[str, Any]:
        return get_workbench_service().delete_factor(factor_id)

    @app.post("/api/factors/init-seeds")
    def api_init_seeds() -> dict[str, Any]:
        return get_workbench_service().init_seed_factors()

    @app.post("/api/panel/switch")
    async def api_switch_panel(req: Request) -> dict[str, Any]:
        body = await req.json()
        return get_workbench_service().switch_panel_source(
            mode=body.get("mode", "auto"), panel_dir=body.get("panel_dir")
        )

    @app.post("/api/pipeline/run")
    async def api_run_pipeline(req: Request) -> dict[str, Any]:
        body = await req.json()
        return get_workbench_service().run_e2e_pipeline(body)

    @app.get("/api/runs")
    def api_list_runs() -> list[dict[str, Any]]:
        return get_workbench_service().list_runs()

    @app.get("/api/runs/{run_id}/report", response_class=HTMLResponse)
    def api_run_report(run_id: str) -> str:
        html = get_workbench_service().get_run_report_html(run_id)
        if not html:
            raise HTTPException(status_code=404, detail="Report not found")
        return html

    frontend_path = Path(__file__).resolve().parent.parent / "frontend"
    if frontend_path.exists():
        app.mount("/", StaticFiles(directory=str(frontend_path), html=True), name="frontend")
except ImportError:
    app = None  # type: ignore[assignment]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="FactorLibrary 因子研发工作台服务器",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--host", type=str, default="127.0.0.1", help="监听地址")
    parser.add_argument("--port", type=int, default=8080, help="监听端口")
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
        help="强制使用内置合成演示面板 (不读取本地 Parquet)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    service = get_workbench_service(
        panel_dir=args.panel_dir,
        db_path=args.db,
        runs_dir=args.runs_dir,
        force_synthetic=args.synthetic,
    )

    server = run_std_server(service, host=args.host, port=args.port)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        logging.info("收到中断信号，正在安全关闭工作台服务...")
        server.shutdown()
        server.server_close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
