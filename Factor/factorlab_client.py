"""FactorLab MCP Python Helper Client."""

import json
import os
import urllib.request

# Load .env if present
_env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
if os.path.exists(_env_path):
    with open(_env_path, encoding="utf-8") as _f:
        for _line in _f:
            _line = _line.strip()
            if _line and not _line.startswith("#") and "=" in _line:
                _k, _v = _line.split("=", 1)
                os.environ.setdefault(_k.strip(), _v.strip())

FACTORLAB_URL = os.environ.get("FACTORLAB_MCP_URL", "https://factorlab.aresquant.cn/mcp")
FACTORLAB_TOKEN = os.environ.get(
    "FACTORLAB_MCP_TOKEN", "flpat_9K1qstqRTdtl2Eo023YyPV2_h9SCGLN8FKk-nMG0nbY"
)


def call_mcp_tool(tool_name: str, arguments: dict = None) -> dict:
    """Call an MCP tool on FactorLab."""
    if arguments is None:
        arguments = {}
    payload = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "tools/call",
        "params": {
            "name": tool_name,
            "arguments": arguments,
        },
    }
    req = urllib.request.Request(
        FACTORLAB_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Accept": "application/json, text/event-stream",
            "Authorization": f"Bearer {FACTORLAB_TOKEN}",
            "User-Agent": "Antigravity/1.0",
        },
        method="POST",
    )
    with urllib.request.urlopen(req) as resp:
        res = json.loads(resp.read().decode("utf-8"))
        if "result" in res and "structuredContent" in res["result"]:
            return res["result"]["structuredContent"]
        return res


if __name__ == "__main__":
    context = call_mcp_tool("get_context")
    print("User:", context.get("data", {}).get("user"))
    factors = call_mcp_tool("list_factors")
    print("Factors count:", len(factors.get("data", {}).get("items", [])))
    for item in factors.get("data", {}).get("items", []):
        print(f"  - [{item.get('factor_id')}] {item.get('name')} (Status: {item.get('status')})")
