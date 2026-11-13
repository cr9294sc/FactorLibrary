#!/usr/bin/env python3
"""FactorLibrary 研发工作台一键启动脚本 (scripts/run_web.py)。

用法示例:
    # 1. 默认启动 (挂载本地 data/derived/futures/v1 面板，监听 127.0.0.1:8080)
    python scripts/run_web.py

    # 2. 指定端口与地址
    python scripts/run_web.py --host 0.0.0.0 --port 9000

    # 3. 强制使用合成演示面板 (无本地数据文件时可用)
    python scripts/run_web.py --synthetic
"""

from __future__ import annotations

import sys
from pathlib import Path

# 兼容源码目录直接运行 (无需预先 editable install)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "python"))

from factorlib.workbench_server import main, parse_args  # noqa: E402

__all__ = ["main", "parse_args"]

if __name__ == "__main__":
    sys.exit(main())
