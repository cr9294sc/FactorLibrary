"""日内微观结构特征聚合 (etl/minute_features.py)。

按白皮书 5.2 节, 对每个 (合约, 交易日) 的全部 1 分钟 bar 向量化提取日频特征。记
``p_m = close_m``, ``r_m = ln(p_m / p_{m-1})``。全部特征仅使用 T 日 15:00 及之前的 bar。

输出特征 (共 11 项, 白皮书称“10 项”但公式列举为 11 项, 此处全部实现):

===================  =============================================================================
rv                   已实现波动率 sqrt(sum r_m^2)
rv_up / rv_down      上行/下行已实现波动率
ret_last30           尾盘 30 分钟动量 p_M / p_{M-30} - 1
ret_first30          开盘 30 分钟收益 p_30 / p_1 - 1
spread_night_day     夜盘收益 - 日盘收益 (无夜盘品种为 null); 段收益 = 段末 close / 段首 open - 1
vol_conc             日内成交量集中度 (Herfindahl) sum (v_m / sum v)^2
amp                  日内振幅 (max(high) - min(low)) / 前结算价 (同合约原始价, 量纲一致)
skew_intra           日内收益偏度 (总体矩, 1/M 归一)
kurt_intra           日内收益超额峰度 (总体矩, 减 3)
vwap_dev             (收盘 - 日内 VWAP) / 日内 VWAP
===================  =============================================================================

另输出 ``vwap_day`` (全天 sum(close*volume)/sum(volume)) 与 ``n_bars``。

**与白皮书文字的两处细化** (均写入 ``fields.json`` 说明):

1. ``r_m`` 仅在**连续交易段内**计算: 相邻 bar 间隔超过 ``max_gap_minutes`` (默认 60) 的收益
   (夜盘->日盘隔夜跳空、午休) 视为跨段跳空, 不计入 rv / rv_up / rv_down / skew / kurt,
   否则隔夜跳空会主导 RV 而非反映日内波动;
2. 振幅用 bar 的 high/low 而非 close 的极值, 否则会系统性低估振幅。
"""

from __future__ import annotations

from pathlib import Path

import polars as pl

from factorlib.core.config import DataConfig, MinuteFeaturesConfig

from .exec_price import aggregate_exec
from .io import iter_minute_months

FEATURE_COLUMNS = (
    "rv",
    "rv_up",
    "rv_down",
    "ret_last30",
    "ret_first30",
    "spread_night_day",
    "vol_conc",
    "amp",
    "skew_intra",
    "kurt_intra",
    "vwap_dev",
)

_KEYS = ["order_book_id", "trading_date"]


def aggregate_features(df: pl.DataFrame, mcfg: MinuteFeaturesConfig) -> pl.DataFrame:
    """对单月分钟 bar 聚合 11 项微观结构特征。

    ``df`` 需含 ``order_book_id, trading_date, datetime, open, high, low, close, volume,
    prev_settlement`` 且已按 (合约, 交易日, datetime) 排序。
    返回列: ``order_book_id, trading_date, n_bars, vwap_day`` + :data:`FEATURE_COLUMNS`。
    """
    hour = pl.col("datetime").dt.hour()
    d = df.with_columns(
        _gap=pl.col("datetime").diff().over(_KEYS),
        _lr=pl.when(pl.col("close") > 0)
        .then(pl.col("close").log())
        .otherwise(None)
        .diff()
        .over(_KEYS),
        _night=(hour >= mcfg.night_start_hour) | (hour < mcfg.night_end_hour),
    ).with_columns(
        _r=pl.when(pl.col("_gap") <= pl.duration(minutes=mcfg.max_gap_minutes))
        .then(pl.col("_lr"))
        .otherwise(None)
    )

    r = pl.col("_r")
    close = pl.col("close")
    vol = pl.col("volume")
    n = pl.len()

    agg = d.group_by(_KEYS, maintain_order=True).agg(
        n.cast(pl.Int64).alias("n_bars"),
        r.count().alias("_n_r"),
        (r**2).sum().alias("_rv2"),
        pl.when(r > 0).then(r**2).otherwise(0.0).sum().alias("_rv2_up"),
        pl.when(r < 0).then(r**2).otherwise(0.0).sum().alias("_rv2_down"),
        pl.when(n > 30)
        .then(close.last() / close.slice(n - 31, 1).first() - 1.0)
        .otherwise(None)
        .alias("ret_last30"),
        (close.slice(29, 1).first() / close.first() - 1.0).alias("ret_first30"),
        (
            close.filter(pl.col("_night")).last() / pl.col("open").filter(pl.col("_night")).first()
            - 1.0
        ).alias("_ret_night"),
        (
            close.filter(~pl.col("_night")).last()
            / pl.col("open").filter(~pl.col("_night")).first()
            - 1.0
        ).alias("_ret_day"),
        vol.sum().alias("_v"),
        (vol**2).sum().alias("_v2"),
        (close * vol).sum().alias("_pv"),
        (pl.col("high").max() - pl.col("low").min()).alias("_hl"),
        pl.col("prev_settlement").first().alias("_prev_settle"),
        close.last().alias("_close_last"),
        r.skew(bias=True).alias("_skew"),
        r.kurtosis(fisher=True, bias=True).alias("_kurt"),
    )

    has_r = pl.col("_n_r") > 0
    has_v = pl.col("_v") > 0
    out = agg.select(
        "order_book_id",
        "trading_date",
        "n_bars",
        pl.when(has_v).then(pl.col("_pv") / pl.col("_v")).otherwise(None).alias("vwap_day"),
        pl.when(has_r).then(pl.col("_rv2").sqrt()).otherwise(None).alias("rv"),
        pl.when(has_r).then(pl.col("_rv2_up").sqrt()).otherwise(None).alias("rv_up"),
        pl.when(has_r).then(pl.col("_rv2_down").sqrt()).otherwise(None).alias("rv_down"),
        "ret_last30",
        "ret_first30",
        (pl.col("_ret_night") - pl.col("_ret_day")).alias("spread_night_day"),
        pl.when(has_v).then(pl.col("_v2") / pl.col("_v") ** 2).otherwise(None).alias("vol_conc"),
        pl.when(pl.col("_prev_settle") > 0)
        .then(pl.col("_hl") / pl.col("_prev_settle"))
        .otherwise(None)
        .alias("amp"),
        pl.col("_skew").fill_nan(None).alias("skew_intra"),
        pl.col("_kurt").fill_nan(None).alias("kurt_intra"),
        pl.when(has_v)
        .then(
            (pl.col("_close_last") - pl.col("_pv") / pl.col("_v")) / (pl.col("_pv") / pl.col("_v"))
        )
        .otherwise(None)
        .alias("vwap_dev"),
    )
    return out


def compute_minute_daily(raw_dir: Path, pairs: pl.DataFrame, cfg: DataConfig) -> pl.DataFrame:
    """单次按月流式遍历分钟线, 同时产出执行价与微观结构特征。

    返回键 (order_book_id, trading_date) 唯一的日频表, 列为
    ``exec_vwap, exec_volume, exec_nbars, n_bars, vwap_day`` + :data:`FEATURE_COLUMNS`。
    ``cfg.features.intraday_minute_features`` 为 False 时仅产出执行价列。
    """
    parts = []
    for _, df in iter_minute_months(raw_dir, pairs):
        ex = aggregate_exec(df, cfg.exec_price.window_minutes)
        if cfg.features.intraday_minute_features:
            feats = aggregate_features(df, cfg.minute_features)
            ex = ex.join(feats, on=_KEYS, how="full", coalesce=True)
        parts.append(ex)
        del df  # 立即释放单月原始分钟数据 (审计 R7)
    if not parts:
        return pl.DataFrame(
            schema={"order_book_id": pl.String, "trading_date": pl.Date, "exec_vwap": pl.Float64}
        )
    return pl.concat(parts, how="diagonal").sort(_KEYS)
