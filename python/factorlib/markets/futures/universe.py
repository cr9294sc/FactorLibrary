"""期货可交易品种池 (markets/futures/universe.py)。

按 ``configs/futures/universe.yaml`` 构造因果的逐日品种池掩码 ``in_universe`` (T 日是否在池):

1. 当日有成交 (日线 volume > 0) 且收盘价有效;
2. 过去 20 个交易日 (含 T) 日均成交额 ``adv20_turnover >= min_adv20_turnover``, 窗口需满 20 个有效日;
3. 品种上市天数 ``listed_days >= min_listed_days`` (自品种最早合约上市日起算的交易日数);
4. 当日主力合约距到期日历天数 ``days_to_expiry >= min_days_to_expiry`` (避开交割月流动性枯竭)。

所有条件只使用 <=T 的数据 (T 日主力合约与到期日在 T 日开盘前已知)。
截面最少品种数 ``min_names_per_day`` 属于评估阶段规则, 不在掩码中施加, 仅在审计中统计。
"""

from __future__ import annotations

import polars as pl

from factorlib.core.config import UniverseConfig

from .etl.calendar import asof_date_idx

ADV_WINDOW = 20


def build_universe(
    daily: pl.DataFrame,
    first_listed: pl.DataFrame,
    cal: pl.DataFrame,
    ucfg: UniverseConfig,
) -> pl.DataFrame:
    """返回 ``symbol, trading_date, adv20_turnover, listed_days, days_to_expiry, in_universe``。

    Parameters
    ----------
    daily:
        日频基础表, 需含 ``symbol, trading_date, date_idx, turnover, traded, close_raw, maturity_date``。
    first_listed:
        ``symbol, first_listed_date`` (见 :func:`etl.instruments.first_listed_dates`)。
    """
    anchor = first_listed.with_columns(
        _anchor_idx=asof_date_idx(first_listed["first_listed_date"], cal)
    ).select(["symbol", "_anchor_idx"])

    df = (
        daily.select(
            "symbol",
            "trading_date",
            "date_idx",
            "turnover",
            "traded",
            "close_raw",
            "maturity_date",
        )
        .sort(["symbol", "date_idx"])
        .join(anchor, on="symbol", how="left")
        .with_columns(
            pl.col("_anchor_idx").fill_null(pl.col("date_idx").min().over("symbol")),
            adv20_turnover=pl.col("turnover")
            .rolling_mean_by("date_idx", window_size=f"{ADV_WINDOW}i", min_samples=ADV_WINDOW, closed="right")
            .over("symbol"),
            days_to_expiry=(pl.col("maturity_date") - pl.col("trading_date")).dt.total_days(),
        )
        .with_columns(listed_days=pl.col("date_idx") - pl.col("_anchor_idx"))
    )
    df = df.with_columns(
        in_universe=(
            pl.col("traded")
            & (pl.col("close_raw") > 0)
            & (pl.col("adv20_turnover") >= ucfg.min_adv20_turnover)
            & (pl.col("listed_days") >= ucfg.min_listed_days)
            & (pl.col("days_to_expiry") >= ucfg.min_days_to_expiry)
        ).fill_null(False)
    )
    return df.select(
        ["symbol", "trading_date", "adv20_turnover", "listed_days", "days_to_expiry", "in_universe"]
    )
