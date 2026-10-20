"""交易日历对齐与稠密网格工具 (etl/calendar.py)。

所有“回看 n 日”的滚动/滞后计算都必须基于**交易日历索引** ``date_idx`` 而非行偏移,
否则品种停牌/缺行 (如 FU/LR/WR 在映射表中存在日期空洞) 会让 ``shift(5)`` 实际回看
超过 5 个交易日。
"""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Sequence

import polars as pl

from .io import read_single


def load_calendar(raw_dir: Path) -> pl.DataFrame:
    """加载交易日历, 返回 ``trading_date`` (升序, 唯一) 与连续整数索引 ``date_idx``。"""
    cal = read_single(raw_dir, "calendar").select("trading_date").unique().sort("trading_date")
    return cal.with_row_index("date_idx").with_columns(pl.col("date_idx").cast(pl.Int64))


def add_date_idx(df: pl.DataFrame, cal: pl.DataFrame, date_col: str = "trading_date") -> pl.DataFrame:
    """为 ``df`` 附加 ``date_idx`` 列; 日期不在日历内则报错 (审计已确认行情日期 100% 落在日历内)。"""
    out = df.join(cal, left_on=date_col, right_on="trading_date", how="left")
    if out["date_idx"].null_count() > 0:
        bad = out.filter(pl.col("date_idx").is_null())[date_col].unique().sort().head(5).to_list()
        raise ValueError(f"存在不在交易日历内的日期: {bad}")
    return out


def asof_date_idx(dates: pl.Series, cal: pl.DataFrame) -> pl.Series:
    """对任意日期 (可能为非交易日) 取不晚于它的最近交易日索引 (向后 asof)。"""
    left = pl.DataFrame({"d": dates}).with_row_index("_row")
    out = left.sort("d").join_asof(
        cal.rename({"trading_date": "d"}).sort("d"), on="d", strategy="backward"
    )
    return out.sort("_row")["date_idx"]


def dense_grid(
    df: pl.DataFrame,
    cal: pl.DataFrame,
    key: str = "symbol",
    end_date: date | None = None,
) -> pl.DataFrame:
    """按品种 [首行, 末行] 区间展开稠密交易日网格, 列: ``key`` / ``trading_date`` / ``date_idx``。

    ``df`` 需含 ``key`` 与 ``date_idx``。若给定 ``end_date`` 则网格上界取 ``end_date`` 对应的索引。
    """
    bounds = df.group_by(key).agg(
        pl.col("date_idx").min().alias("_lo"), pl.col("date_idx").max().alias("_hi")
    )
    grid = (
        bounds.with_columns(date_idx=pl.int_ranges(pl.col("_lo"), pl.col("_hi") + 1))
        .explode("date_idx")
        .drop(["_lo", "_hi"])
        .join(cal, on="date_idx", how="left")
        .select([key, "trading_date", "date_idx"])
        .sort([key, "date_idx"])
    )
    if end_date is not None:
        grid = grid.filter(pl.col("trading_date") <= end_date)
    return grid


def lag_by_idx(
    df: pl.DataFrame,
    cols: Sequence[str],
    n: int,
    key: str = "symbol",
    suffix: str | None = None,
) -> pl.DataFrame:
    """取 ``date_idx - n`` 处的取值 (按日历索引而非行偏移), 新增列名为 ``{col}{suffix}``。"""
    suffix = suffix if suffix is not None else f"_lag{n}"
    right = df.select(
        [key, (pl.col("date_idx") + n).alias("date_idx"), *[pl.col(c).alias(f"{c}{suffix}") for c in cols]]
    )
    return df.join(right, on=[key, "date_idx"], how="left")
