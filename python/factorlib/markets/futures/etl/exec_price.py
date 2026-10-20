"""分钟级开盘 30m 执行价 (etl/exec_price.py)。

执行价定义 (白皮书 4.1 / 审计 R3):

    P_exec(c, T) = sum(close_m * volume_m) / sum(volume_m),   m ∈ 合约 c 在交易日 T 开盘后 30 分钟窗口

- 窗口起点取该品种当日**所有参与计算的合约分钟 bar 的最早时间** (夜盘品种为 21:01 标签,
  日盘品种为 09:01/09:31 标签; 米筐 1m bar 以结束时刻标注), 窗口为 ``[open, open+30min)``,
  因此恰好覆盖 30 根 1 分钟 bar。窗口完全基于时间而非 bar 序号, 对缺 bar 的低流动性合约稳健;
- **严禁**使用 ``total_turnover / volume``: 郑商所历史成交额大量置零 (审计 R3), 会得到 0 价格;
- 窗口内总成交量为 0 时执行价记为 null (无成交, 不可执行)。

本模块的聚合函数与 :mod:`.minute_features` 共用同一次按月分钟线流式遍历。
"""

from __future__ import annotations

from pathlib import Path

import polars as pl

from .io import iter_minute_months


def minute_pairs(daily: pl.DataFrame, bars: pl.DataFrame) -> pl.DataFrame:
    """确定需要读取分钟线的 (合约, 交易日) 集合。

    - 当日主力合约 ``c_T`` 在 T 日: 用于 ``exec_vwap`` 与微观结构特征;
    - 换月日的旧合约 ``c_{T-1}`` 在 T 日: 用于 ``exec_vwap_old`` (旧合约平仓价, 链式收益用);
    - 仅保留日线成交量 > 0 的合约-日, 避免扫描无成交的死品种。

    返回列: ``order_book_id, trading_date, symbol, prev_settlement``。
    """
    cur = daily.select(
        pl.col("contract").alias("order_book_id"), "trading_date", "symbol"
    )
    old = daily.filter(pl.col("is_roll")).select(
        pl.col("prev_contract").alias("order_book_id"), "trading_date", "symbol"
    )
    pairs = pl.concat([cur, old]).unique(subset=["order_book_id", "trading_date"], keep="first")
    traded = bars.filter(pl.col("volume") > 0).select(
        pl.col("contract").alias("order_book_id"),
        "trading_date",
        pl.col("prev_settlement"),
    )
    return pairs.join(traded, on=["order_book_id", "trading_date"], how="inner").sort(
        ["trading_date", "order_book_id"]
    )


def aggregate_exec(df: pl.DataFrame, window_minutes: int) -> pl.DataFrame:
    """对单月分钟 bar 聚合开盘窗口执行价。

    ``df`` 需含 ``symbol, order_book_id, trading_date, datetime, close, volume``。
    返回列: ``order_book_id, trading_date, exec_vwap, exec_volume, exec_nbars``。
    """
    session_open = df.group_by(["symbol", "trading_date"]).agg(
        pl.col("datetime").min().alias("_session_open")
    )
    window = df.join(session_open, on=["symbol", "trading_date"], how="inner").filter(
        pl.col("datetime") < pl.col("_session_open") + pl.duration(minutes=window_minutes)
    )
    out = window.group_by(["order_book_id", "trading_date"]).agg(
        (pl.col("close") * pl.col("volume")).sum().alias("_pv"),
        pl.col("volume").sum().alias("exec_volume"),
        pl.len().cast(pl.Int64).alias("exec_nbars"),
    )
    return out.select(
        "order_book_id",
        "trading_date",
        pl.when(pl.col("exec_volume") > 0)
        .then(pl.col("_pv") / pl.col("exec_volume"))
        .otherwise(None)
        .alias("exec_vwap"),
        "exec_volume",
        "exec_nbars",
    )


def compute_exec_prices(raw_dir: Path, pairs: pl.DataFrame, window_minutes: int = 30) -> pl.DataFrame:
    """独立遍历分钟线, 仅计算执行价 (主要用于测试与单独重算)。"""
    parts = [aggregate_exec(df, window_minutes) for _, df in iter_minute_months(raw_dir, pairs)]
    if not parts:
        return pl.DataFrame(
            schema={
                "order_book_id": pl.String,
                "trading_date": pl.Date,
                "exec_vwap": pl.Float64,
                "exec_volume": pl.Float64,
                "exec_nbars": pl.Int64,
            }
        )
    return pl.concat(parts)
