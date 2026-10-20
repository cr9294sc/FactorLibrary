"""日线基础量价清洗 (etl/daily.py)。

以主力映射 (symbol, T, c_T) 为骨架, 左连接 ``bar_1d`` 中合约 c_T 在 T 日的量价, 并附加静态
合约规格与板块。本阶段保留**原始 (未复权) 价格**, 复权由 :mod:`.adjust` 完成。
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import polars as pl

from .io import scan_dataset

_BAR_COLUMNS = (
    "order_book_id",
    "trading_date",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "total_turnover",
    "open_interest",
    "settlement",
    "prev_settlement",
    "limit_up",
    "limit_down",
    "is_unilateral",
)


def load_bars_1d(raw_dir: Path, contracts: Sequence[str]) -> pl.DataFrame:
    """读取给定合约集合的日线, 键 (contract, trading_date), 价量列统一转 Float64。

    额外计算 ``oi_prev`` (同一合约上一有效交易日持仓量), 只回看同合约历史, 不含任何未来信息;
    换月日用它计算 ``oi_chg`` 可避免跨合约持仓量作差。
    """
    bars = (
        scan_dataset(raw_dir, "bar_1d", _BAR_COLUMNS)
        .filter(pl.col("order_book_id").is_in(pl.Series(list(contracts)).implode()))
        .collect()
    )
    bars = (
        bars.rename({"order_book_id": "contract"})
        .with_columns(
            pl.col(
                [
                    "open",
                    "high",
                    "low",
                    "close",
                    "settlement",
                    "prev_settlement",
                    "limit_up",
                    "limit_down",
                ]
            ).cast(pl.Float64),
        )
        .sort(["contract", "trading_date"])
        .with_columns(oi_prev=pl.col("open_interest").shift(1).over("contract"))
    )
    if bars.select(["contract", "trading_date"]).is_duplicated().any():
        raise ValueError("bar_1d 在 (order_book_id, trading_date) 上存在重复记录")
    return bars


def build_daily(
    dominant: pl.DataFrame,
    bars: pl.DataFrame,
    instruments: pl.DataFrame,
    sectors: pl.DataFrame,
) -> pl.DataFrame:
    """组装日频基础表 (未复权)。

    成交额口径: 优先使用 ``total_turnover``; 若出现 ``volume>0`` 但成交额为 0 (郑商所历史异常,
    审计 R3 在分钟线上确认, 日线主力合约上实测为 0 例) 则以 ``volume × settlement × multiplier`` 近似。
    """
    spec = instruments.select(["contract", "multiplier", "margin_rate", "maturity_date"])
    df = (
        dominant.join(bars, on=["contract", "trading_date"], how="left")
        .join(spec, on="contract", how="left")
        .join(sectors, on="symbol", how="left")
    )
    est_turnover = pl.col("volume") * pl.col("settlement") * pl.col("multiplier")
    df = df.with_columns(
        turnover_fallback=(pl.col("volume") > 0) & (pl.col("total_turnover").fill_null(0.0) <= 0),
    ).with_columns(
        turnover=pl.when(pl.col("turnover_fallback"))
        .then(est_turnover)
        .otherwise(pl.col("total_turnover").cast(pl.Float64)),
        oi_chg=pl.col("open_interest") - pl.col("oi_prev"),
        traded=pl.col("volume").fill_null(0.0) > 0,
    )
    df = df.select(
        "trading_date",
        "symbol",
        "contract",
        "prev_contract",
        "is_roll",
        "date_idx",
        "sector",
        "multiplier",
        "margin_rate",
        "maturity_date",
        pl.col("open").alias("open_raw"),
        pl.col("high").alias("high_raw"),
        pl.col("low").alias("low_raw"),
        pl.col("close").alias("close_raw"),
        pl.col("settlement").alias("settle_raw"),
        pl.col("prev_settlement").alias("prev_settle_raw"),
        pl.col("volume").cast(pl.Float64),
        "turnover",
        pl.col("open_interest").cast(pl.Float64),
        "oi_chg",
        "limit_up",
        "limit_down",
        "is_unilateral",
        "traded",
        "turnover_fallback",
    ).sort(["trading_date", "symbol"])
    return df
