"""展期收益率 / 期限结构清洗 (etl/aux_roll.py)。

米筐 ``roll_yield`` 按 (品种, 日期, roll_type, rule) 提供官方展期收益率:

- ``main_sub``:  主力 -> 次主力, ``roll_yield = (settle(main) - settle(sub)) / settle(sub)``;
  ``annualized_yield_trading`` 为按交易日 (252/ΔDays) 年化, 与白皮书 5.3 节公式口径一致;
- ``near_main``: 近月 -> 主力。

只保留与面板主力合约一致的记录: ``main_sub.from_contract == c_T`` 且 ``near_main.to_contract == c_T``
(审计实测一致率均为 100%), 不一致的记录置 null, 保证期限结构与持仓合约同源。

发布时点: 展期收益率基于 T 日结算价, 随 T 日盘后日线数据一并发布, 在 T 日盘后信号生成时刻已知。
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import polars as pl

from factorlib.core.config import DataConfig

from .io import scan_dataset


def build_roll_yield(raw_dir: Path, cfg: DataConfig, daily: pl.DataFrame, end_date: date) -> pl.DataFrame:
    """返回 ``symbol, trading_date, ry_main_sub, ry_ann, ry_near_main``。"""
    ry = (
        scan_dataset(
            raw_dir,
            "roll_yield",
            [
                "trading_date",
                "underlying_symbol",
                "roll_type",
                "rule",
                "from_contract",
                "to_contract",
                "roll_yield",
                "annualized_yield_trading",
            ],
        )
        .filter(
            (pl.col("rule") == cfg.dominant.rule)
            & pl.col("trading_date").is_between(cfg.start_date, end_date)
        )
        .rename({"underlying_symbol": "symbol"})
        .collect()
    )
    keys = daily.select(["symbol", "trading_date", "contract"])

    main_sub = (
        ry.filter(pl.col("roll_type") == "main_sub")
        .unique(subset=["symbol", "trading_date"], keep="first")
        .join(keys, on=["symbol", "trading_date"], how="inner")
        .filter(pl.col("from_contract") == pl.col("contract"))
        .select(
            "symbol",
            "trading_date",
            pl.col("roll_yield").alias("ry_main_sub"),
            pl.col("annualized_yield_trading").alias("ry_ann"),
        )
    )
    near_main = (
        ry.filter(pl.col("roll_type") == "near_main")
        .unique(subset=["symbol", "trading_date"], keep="first")
        .join(keys, on=["symbol", "trading_date"], how="inner")
        .filter(pl.col("to_contract") == pl.col("contract"))
        .select("symbol", "trading_date", pl.col("roll_yield").alias("ry_near_main"))
    )
    return keys.select(["symbol", "trading_date"]).join(
        main_sub, on=["symbol", "trading_date"], how="left"
    ).join(near_main, on=["symbol", "trading_date"], how="left")
