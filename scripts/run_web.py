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

import argparse
import logging
import sys
from pathlib import Path

# 将 python/ 目录加入 sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "python"))

from factorlib.core.workbench import FactorWorkbenchService
from factorlib.workbench_server import run_std_server


def parse_args(args: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="FactorLibrary 现代化 6 阶段量化因子研发工作台服务启动器",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--host", type=str, default="127.0.0.1", help="服务监听 IP 地址")
    parser.add_argument("--port", type=int, default=8080, help="服务监听端口")
    parser.add_argument(
        "--panel-dir",
        type=Path,
        default=Path("data/derived/futures/v1"),
        help="标准研究面板存储目录",
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
        help="强制启用内置合成演示面板 (适合快速体验与测试)",
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
        service = FactorWorkbenchService(
            panel_dir=args.panel_dir,
            db_path=args.db,
            runs_dir=args.runs_dir,
            force_synthetic=args.synthetic,
        )
        server = run_std_server(service, host=args.host, port=args.port)
        server.serve_forever()
        return 0
    except KeyboardInterrupt:
        logging.info("收到中断信号，服务正常退出。")
        return 0
    except Exception as exc:
        logging.exception("工作台服务运行失败: %s", exc)
        return 1


if __name__ == "__main__":
    sys.exit(main())
