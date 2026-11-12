"""FactorLab 线上平台 MCP JSON-RPC 轻量客户端辅助模块 (core/mcp_client.py)。

用于本地对账脚本 (`scripts/backtest_multi_factors.py` 等) 按需拉取线上因子定义与回测基准摘要。
认证凭证从仓库根目录 `.env`（已 gitignore）或环境变量 `FACTORLAB_MCP_TOKEN` 读取。
"""

from __future__ import annotations

import json
import os
import urllib.request
from pathlib import Path
from typing import Any


def _load_dotenv_if_present() -> None:
    """从仓库根目录或当前工作目录加载 `.env` 环境变量（若存在）。"""
    candidates = [
        Path(__file__).resolve().parents[3] / ".env",
        Path.cwd() / ".env",
    ]
    for env_path in candidates:
        if env_path.is_file():
            for line in env_path.read_text(encoding="utf-8").splitlines():
                stripped = line.strip()
                if stripped and not stripped.startswith("#") and "=" in stripped:
                    key, val = stripped.split("=", 1)
                    os.environ.setdefault(key.strip(), val.strip().strip("'\""))
            break


_load_dotenv_if_present()

FACTORLAB_URL: str = os.environ.get("FACTORLAB_MCP_URL", "https://factorlab.aresquant.cn/mcp")


def call_mcp_tool(tool_name: str, arguments: dict[str, Any] | None = None) -> dict[str, Any]:
    """调用 FactorLab 线上 MCP 工具接口并返回结构化 JSON 字典。"""
    token = os.environ.get("FACTORLAB_MCP_TOKEN", "")
    payload = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "tools/call",
        "params": {
            "name": tool_name,
            "arguments": arguments or {},
        },
    }
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json, text/event-stream",
        "User-Agent": "FactorLibrary/1.0",
    }
    if token:
        headers["Authorization"] = f"Bearer {token}"

    req = urllib.request.Request(
        FACTORLAB_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        res: dict[str, Any] = json.loads(resp.read().decode("utf-8"))
        if "result" in res and isinstance(res["result"], dict):
            structured = res["result"].get("structuredContent")
            if isinstance(structured, dict):
                return structured
        return res
