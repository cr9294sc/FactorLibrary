"""主力合约连续映射构建 (etl/dominant.py)。

规则 (审计 R1/R2 已验证):
- 米筐 ``dominant_mapping`` 原生包含 rule=3 (跳过最后交易日) / rank=1 (主力), 直接过滤使用;
- ``date == T`` 的记录即交易日 T 当天有效的主力合约 (T-1 收盘后依据持仓量选定,
  T 日开盘前已知), **严禁**再向前/向后平移;
- 日内不切换合约。换月日 ``is_roll=True``, ``prev_contract`` 为换月前一有效行的旧合约。
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import polars as pl

from factorlib.core.config import DataConfig

from .calendar import add_date_idx
from .io import scan_dataset


def load_dominant(
    raw_dir: Path,
    cfg: DataConfig,
    cal: pl.DataFrame,
    end_date: date | None = None,
) -> pl.DataFrame:
    """构建主力映射表。

    返回列: ``trading_date, symbol, contract, prev_contract, is_roll, date_idx``,
    按 (symbol, trading_date) 升序; 每品种首行 ``prev_contract == contract``。
    """
    end = min(cfg.end_date, end_date) if end_date is not None else cfg.end_date
    dm = (
        scan_dataset(
            raw_dir,
            "dominant_mapping",
            ["date", "underlying_symbol", "rule", "rank", "order_book_id"],
        )
        .filter((pl.col("rule") == cfg.dominant.rule) & (pl.col("rank") == cfg.dominant.rank))
        .filter(pl.col("date").is_between(cfg.start_date, end))
        .select(
            pl.col("date").alias("trading_date"),
            pl.col("underlying_symbol").alias("symbol"),
            pl.col("order_book_id").alias("contract"),
        )
        .collect()
    )
    if dm.is_empty():
        raise ValueError("dominant_mapping 在给定规则与时间范围内无任何记录")

    dup = dm.select(["trading_date", "symbol"]).is_duplicated().sum()
    if dup:
        raise ValueError(f"dominant_mapping 在 (date, symbol) 上存在 {dup} 条重复记录")

    dm = add_date_idx(dm, cal).sort(["symbol", "trading_date"])
    dm = dm.with_columns(
        prev_contract=pl.col("contract").shift(1).over("symbol").fill_null(pl.col("contract"))
    ).with_columns(is_roll=pl.col("contract") != pl.col("prev_contract"))
    return dm.select(["trading_date", "symbol", "contract", "prev_contract", "is_roll", "date_idx"])


def describe_gaps(dominant: pl.DataFrame) -> pl.DataFrame:
    """列出映射表中存在日期空洞 (相邻行 date_idx 差 > 1) 的位置, 供审计与日志使用。"""
    return (
        dominant.sort(["symbol", "date_idx"])
        .with_columns(gap=(pl.col("date_idx") - pl.col("date_idx").shift(1).over("symbol")))
        .filter(pl.col("gap") > 1)
        .select(["symbol", "trading_date", "gap"])
    )
