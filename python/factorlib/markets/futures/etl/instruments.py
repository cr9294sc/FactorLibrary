"""静态合约规格与板块清洗 (etl/instruments.py)。

``instruments_info`` 是合约静态快照表: 每合约一行乘数/保证金率/上市日/到期日,
不包含历史动态保证金率与手续费 (审计 R4), 后者由 ``configs/futures/fees.yaml`` 补偿。
"""

from __future__ import annotations

from pathlib import Path

import polars as pl

from .io import read_single

_DATE_FMT = "%Y-%m-%d"


def _to_date(col: str) -> pl.Expr:
    """字符串日期 -> Date; 空串/占位符 (如 0000-00-00) 一律转为 null。"""
    return pl.col(col).str.to_date(_DATE_FMT, strict=False)


def load_instruments(raw_dir: Path) -> pl.DataFrame:
    """合约静态规格表, 键 ``contract`` (= order_book_id)。

    列: contract, symbol, exchange, multiplier, margin_rate, listed_date, maturity_date
    """
    inst = read_single(raw_dir, "instruments_info")
    out = (
        inst.select(
            pl.col("order_book_id").alias("contract"),
            pl.col("underlying_symbol").alias("symbol"),
            pl.col("exchange"),
            pl.col("contract_multiplier").cast(pl.Float64).alias("multiplier"),
            pl.col("margin_rate").cast(pl.Float64),
            _to_date("listed_date").alias("listed_date"),
            _to_date("maturity_date").alias("maturity_date"),
        )
        .unique(subset=["contract"], keep="first", maintain_order=True)
        .sort("contract")
    )
    return out


def first_listed_dates(instruments: pl.DataFrame) -> pl.DataFrame:
    """每个品种最早合约上市日 (品种上市锚点), 列: symbol, first_listed_date。"""
    return (
        instruments.filter(pl.col("listed_date").is_not_null())
        .group_by("symbol")
        .agg(pl.col("listed_date").min().alias("first_listed_date"))
    )


def load_sectors(raw_dir: Path) -> pl.DataFrame:
    """品种 -> 一级板块 (用于 group 中性化), 列: symbol, sector。"""
    ind = read_single(raw_dir, "industry_mapping")
    return (
        ind.select(
            pl.col("underlying_symbol").alias("symbol"),
            pl.col("level1_industry").alias("sector"),
        )
        .unique(subset=["symbol"], keep="first", maintain_order=True)
    )
