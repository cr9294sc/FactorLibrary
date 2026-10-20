"""会员持仓龙虎榜聚合 (etl/aux_member.py)。

对面板主力合约 ``c_T`` 在 T 日的会员排名, 取多头/空头持仓前 ``top_n`` (默认 20) 名之和:

    L = sum_{rank<=N} long_volume,   S = sum_{rank<=N} short_volume
    mr_net_ratio       = (L - S) / (L + S)
    mr_cr20            = (L + S) / open_interest_T          (取值范围约 [0, 2])
    mr_net_ratio_chg5  = mr_net_ratio_T - mr_net_ratio_{T-5}

注意:
- ``member_rank.volume`` 在 ``rank_by in {long, short}`` 时即会员持买/持卖仓量;
- 数据自 2012-01 起, 之前为 null;
- ``mr_net_ratio_chg5`` 的回看同样按交易日历索引, 跨换月日时前后为不同合约, 属于比率在时间上的
  比较而非持仓量直接作差, 该口径与白皮书 5.3 节一致;
- 龙虎榜盘后 (约 16:30-17:00) 发布, 早于 T+1 夜盘/日盘执行窗口, 在信号生成时刻已知。
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import polars as pl

from factorlib.core.config import DataConfig

from .calendar import lag_by_idx
from .io import scan_dataset


def build_member_features(
    raw_dir: Path,
    cfg: DataConfig,
    daily: pl.DataFrame,
    end_date: date,
) -> pl.DataFrame:
    """返回 ``symbol, trading_date, mr_net_ratio, mr_cr20, mr_net_ratio_chg5``。"""
    mcfg = cfg.features.member_rank
    keys = daily.select(["symbol", "trading_date", "date_idx", "contract", "open_interest"])
    pair_lf = keys.select(
        pl.col("contract").alias("order_book_id"), "trading_date", "symbol"
    ).lazy()

    top = (
        scan_dataset(
            raw_dir,
            "member_rank",
            ["trading_date", "order_book_id", "rank_by", "member_rank", "volume"],
        )
        .filter(
            pl.col("rank_by").is_in(["long", "short"])
            & (pl.col("member_rank") <= mcfg.top_n)
            & pl.col("trading_date").is_between(cfg.start_date, end_date)
        )
        .join(pair_lf, on=["order_book_id", "trading_date"], how="inner")
        .group_by(["symbol", "trading_date"])
        .agg(
            pl.col("volume").filter(pl.col("rank_by") == "long").sum().alias("_long"),
            pl.col("volume").filter(pl.col("rank_by") == "short").sum().alias("_short"),
        )
        .collect()
    )

    base = keys.join(top, on=["symbol", "trading_date"], how="left").with_columns(
        _tot=pl.col("_long") + pl.col("_short")
    )
    base = base.with_columns(
        mr_net_ratio=pl.when(pl.col("_tot") > 0).then((pl.col("_long") - pl.col("_short")) / pl.col("_tot")).otherwise(None),
        mr_cr20=pl.when((pl.col("_tot") > 0) & (pl.col("open_interest") > 0))
        .then(pl.col("_tot") / pl.col("open_interest"))
        .otherwise(None),
    )
    base = lag_by_idx(base, ["mr_net_ratio"], mcfg.delta_lag, suffix="_lag").with_columns(
        mr_net_ratio_chg5=pl.col("mr_net_ratio") - pl.col("mr_net_ratio_lag")
    )
    return base.select(["symbol", "trading_date", "mr_net_ratio", "mr_cr20", "mr_net_ratio_chg5"])
