"""FactorLab MCP Python Helper Client."""

import os
import json
import urllib.request

FACTORLAB_URL = os.environ.get("FACTORLAB_MCP_URL", "https://factorlab.aresquant.cn/mcp")
FACTORLAB_TOKEN = os.environ.get("FACTORLAB_MCP_TOKEN", "flpat_TTyS3h4J1n8mwM-pm_79ix6S3tHSST40DLzcqVWvYek")

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
