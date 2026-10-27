"""Fetch all catalog entries (variables, functions, postprocess-operators) from FactorLab MCP."""

import os
import json
import urllib.request
import time

MCP_URL = os.environ.get("FACTORLAB_MCP_URL", "https://factorlab.aresquant.cn/mcp")
MCP_TOKEN = os.environ.get("FACTORLAB_MCP_TOKEN", "flpat_TTyS3h4J1n8mwM-pm_79ix6S3tHSST40DLzcqVWvYek")

OUTPUT_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "catalog_full.json")


def fetch_catalog_items(catalog_name: str) -> list:
    """Fetch all items from a given catalog using pagination."""
    items = []
    offset = 0
    limit = 100
    print(f"[*] Fetching catalog '{catalog_name}'...")

    while True:
        payload = {
            "jsonrpc": "2.0",
            "id": int(time.time() * 1000) % 1000000,
            "method": "tools/call",
            "params": {
                "name": "search_catalog",
                "arguments": {
                    "catalog": catalog_name,
                    "limit": limit,
                    "offset": offset,
                },
            },
        }

        req = urllib.request.Request(
            MCP_URL,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Accept": "application/json, text/event-stream",
                "Authorization": f"Bearer {MCP_TOKEN}",
                "User-Agent": "FactorLabDocBuilder/1.0",
            },
            method="POST",
        )

        with urllib.request.urlopen(req) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            structured = data.get("result", {}).get("structuredContent", {}).get("data", {})
            total = structured.get("total", 0)
            batch = structured.get("items", [])
            items.extend(batch)
            print(f"    - fetched {len(batch)} items (offset {offset}/{total})")

            if len(batch) < limit or offset + len(batch) >= total:
                break
            offset += len(batch)

    print(f"[+] Finished '{catalog_name}': total {len(items)} items retrieved.\n")
    return items


def main():
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

    variables = fetch_catalog_items("variables")
    functions = fetch_catalog_items("functions")
    postprocess_operators = fetch_catalog_items("postprocess-operators")

    catalog_data = {
        "fetched_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "summary": {
            "variables_count": len(variables),
            "functions_count": len(functions),
            "postprocess_count": len(postprocess_operators),
            "total_items": len(variables) + len(functions) + len(postprocess_operators),
        },
        "variables": variables,
        "functions": functions,
        "postprocess_operators": postprocess_operators,
    }

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(catalog_data, f, ensure_ascii=False, indent=2)

    print(f"[SUCCESS] Wrote full catalog to: {OUTPUT_PATH}")
    print(f"Summary: {json.dumps(catalog_data['summary'], indent=2)}")


if __name__ == "__main__":
    main()
