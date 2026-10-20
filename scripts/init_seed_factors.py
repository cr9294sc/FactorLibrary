#!/usr/bin/env python3
"""FactorLibrary 经典文献种子因子评估与注册入库脚本 (scripts/init_seed_factors.py)。

用法:
    python scripts/init_seed_factors.py
    python scripts/init_seed_factors.py --panel data/derived/futures/v1 --db data/registry/factors.sqlite
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# 将 python/ 目录加入 sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "python"))

from factorlib.core.evaluate import evaluate_factor, get_seed_factors
from factorlib.core.panel import Panel
from factorlib.core.registry import ExperimentRunRecord, FactorRegistry


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="FactorLibrary 经典文献种子因子评估与注册入库工具")
    parser.add_argument(
        "--panel",
        type=Path,
        default=Path("data/derived/futures/v1"),
        help="派生研究面板路径 (默认: data/derived/futures/v1)",
    )
    parser.add_argument(
        "--db",
        type=Path,
        default=Path("data/registry/factors.sqlite"),
        help="SQLite 因子库路径 (默认: data/registry/factors.sqlite)",
    )
    parser.add_argument(
        "--run-id",
        type=str,
        default="seed_init_v1",
        help="实验运行批次标识符 (默认: seed_init_v1)",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="日志级别 (默认: INFO)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="[%(asctime)s] [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    root_dir = Path(__file__).resolve().parent.parent
    panel_path = args.panel if args.panel.is_absolute() else (root_dir / args.panel)
    db_path = args.db if args.db.is_absolute() else (root_dir / args.db)

    if not panel_path.exists():
        logging.error("面板目录不存在: %s，请先运行 scripts/build_panel.py 构建面板", panel_path)
        return 1

    logging.info("正在加载研究面板: %s ...", panel_path)
    panel = Panel.load(panel_path)
    logging.info(
        "面板加载完成: %d 交易日, %d 品种, %d 特征",
        panel.T,
        panel.N,
        len(panel.fields),
    )

    registry = FactorRegistry(db_path=db_path)
    logging.info("连接因子注册库: %s", db_path)

    # 记录实验开始
    run_rec = ExperimentRunRecord(
        run_id=args.run_id,
        kind="eval",
        config_yaml="seed_factors_initialization",
        data_version=str(panel.meta.get("panel_version", "v1")),
        git_sha="p3_seed_init",
        status="running",
    )
    registry.record_run(run_rec)

    seeds = get_seed_factors()
    logging.info("开始评估与入库 %d 个经典学术文献种子因子...", len(seeds))

    print("\n" + "=" * 90)
    print(
        f"{'种子因子名称':<24} {'推荐方向':<8} {'表达式':<26} {'IC均值':<10} {'ICIR':<8} {'多空年化':<10} {'换手率':<8}"
    )
    print("=" * 90)

    registered_count = 0
    for s_def in seeds:
        res = evaluate_factor(
            engine_or_panel=panel,
            expr=s_def["expr"],
            force_sign=s_def["sign"],
        )
        rec = res.to_factor_record(
            source="seed",
            source_run=args.run_id,
            status="accepted",
        )
        registry.register_factor(rec, run_id=args.run_id, upsert=True)
        registered_count += 1

        # 提取展示指标 (取 full 或 train 切片的 h=5)
        m_show = None
        for m in res.all_metrics:
            if m.horizon == 5:
                m_show = m
                break
        if m_show is None and res.all_metrics:
            m_show = res.all_metrics[0]

        ic_mean_str = f"{m_show.ic_mean:+.4f}" if m_show else "N/A"
        ic_ir_str = f"{m_show.ic_ir:+.2f}" if m_show else "N/A"
        ls_ret_str = f"{m_show.ls_annual_ret:+.2%}" if m_show else "N/A"
        turnover_str = f"{m_show.turnover:.2%}" if m_show else "N/A"
        sign_str = "+1 (正向)" if s_def["sign"] > 0 else "-1 (反向)"

        print(
            f"{s_def['name']:<24} {sign_str:<8} {s_def['expr']:<26} "
            f"{ic_mean_str:<10} {ic_ir_str:<8} {ls_ret_str:<10} {turnover_str:<8}"
        )

    print("=" * 90 + "\n")

    registry.finish_run(args.run_id, status="completed", n_evaluated=registered_count)
    logging.info("全部 %d 个种子因子成功评估并原子化入库!", registered_count)
    logging.info("因子库存储路径: %s", db_path)

    return 0


if __name__ == "__main__":
    sys.exit(main())
