"""比例复权价格计算与官方因子对账 (etl/adjust.py)。

``prev_close_ratio`` 比例复权。设日期 ``s`` 发生主力合约 A -> B 的换月, 换月比例

    k_s = close_{s-1}(A) / close_{s-1}(B)

两个收盘价均为 ``s`` 之前最近一个有效交易日 (通常为 s-1) 的收盘价, 在 s 日开盘前已知。

**因果锚定 (相对白皮书 5.1 的刻意细化)**: 白皮书以“最新合约价位”为锚, 即
``P_adj(T) = P_raw(T) * prod_{s_j > T} 1/k_{s_j}``, 该公式在 T 日会用到 T 之后的换月比例,
导致同一历史日期的复权价水平随新换月不断漂移, 违反“T 日面板只依赖 <=T 数据”的红线
(未来扰动测试无法通过)。本实现改为以每品种**首个面板日价位为锚**:

    F(T)      = prod_{s_j <= T} k_{s_j}          (F 在首日 = 1)
    P_adj(T)  = P_raw(T) * F(T)

两者仅相差一个按品种固定的全局常数, 所有收益率/比例类算子结果完全一致; 且米筐官方
``ex_cum_factor`` 同样为“自历史起点向后累乘”的口径, 与本实现可直接对账。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import polars as pl

from factorlib.core.config import DataConfig

from .io import scan_dataset

#: 随复权因子缩放的价格字段: (原始列, 复权后列)
PRICE_PAIRS = (
    ("open_raw", "open"),
    ("high_raw", "high"),
    ("low_raw", "low"),
    ("close_raw", "close"),
    ("settle_raw", "settle"),
)


@dataclass
class AdjustVerification:
    """复权因子对账结果。"""

    tolerance: float
    n_compared: int
    max_rel_err: float
    failures: pl.DataFrame  # 列: symbol, max_rel_err, first_bad_date
    fallback_rolls: pl.DataFrame  # 因缺少前收盘价而回退 k=1 的换月, 列: symbol, trading_date, contract, prev_contract

    @property
    def ok(self) -> bool:
        return self.failures.is_empty()


def compute_roll_ratios(daily: pl.DataFrame, bars: pl.DataFrame) -> pl.DataFrame:
    """计算每个换月日的比例因子 k_s。

    返回列: ``symbol, trading_date, contract, prev_contract, k, k_fallback``。
    若新旧合约任一在 s 日之前无有效收盘价 (如死品种的新合约首日), ``k`` 回退为 1.0 并标记
    ``k_fallback=True`` (这类品种成交量为 0, 不会进入品种池)。
    """
    rolls = daily.filter(pl.col("is_roll")).select(["symbol", "trading_date", "contract", "prev_contract"])
    closes = (
        bars.select(["contract", "trading_date", "close"])
        .filter(pl.col("close") > 0)
        .sort("trading_date")
    )
    old = closes.rename({"contract": "prev_contract", "close": "close_old"})
    new = closes.rename({"close": "close_new"})
    out = (
        rolls.sort("trading_date")
        .join_asof(old, on="trading_date", by="prev_contract", strategy="backward",
                   allow_exact_matches=False, check_sortedness=False)
        .join_asof(new, on="trading_date", by="contract", strategy="backward",
                   allow_exact_matches=False, check_sortedness=False)
    )
    out = out.with_columns(
        k_raw=pl.col("close_old") / pl.col("close_new"),
    ).with_columns(
        k_fallback=pl.col("k_raw").is_null(),
        k=pl.col("k_raw").fill_null(1.0),
    )
    return out.select(["symbol", "trading_date", "contract", "prev_contract", "k", "k_fallback"]).sort(
        ["symbol", "trading_date"]
    )


def apply_adjustment(daily: pl.DataFrame, bars: pl.DataFrame) -> tuple[pl.DataFrame, pl.DataFrame]:
    """为日频表附加 ``adj_factor`` 与复权价格列, 返回 (daily, rolls)。"""
    rolls = compute_roll_ratios(daily, bars)
    df = (
        daily.sort(["symbol", "trading_date"])
        .join(rolls.select(["symbol", "trading_date", "k"]), on=["symbol", "trading_date"], how="left")
        .with_columns(pl.col("k").fill_null(1.0))
        .with_columns(adj_factor=pl.col("k").cum_prod().over("symbol"))
        .drop("k")
    )
    df = df.with_columns([(pl.col(raw) * pl.col("adj_factor")).alias(adj) for raw, adj in PRICE_PAIRS])
    return df.sort(["trading_date", "symbol"]), rolls


def verify_against_official(
    daily: pl.DataFrame,
    rolls: pl.DataFrame,
    raw_dir: Path,
    cfg: DataConfig,
) -> AdjustVerification:
    """与米筐 ``dominant_ex_factor.ex_cum_factor`` 对账累计复权因子。

    对每个品种, 取我们与官方都有值的日期序列, 各自除以首个共同日的取值后逐日比较相对误差,
    消除两侧锚定位置不同的影响。最大相对误差超过 ``cfg.adjust.verify_tolerance`` 的品种
    记为失败 (黄色警报, 由调用方记录日志, 不中断构建)。
    """
    official = (
        scan_dataset(
            raw_dir,
            "dominant_ex_factor",
            ["date", "underlying_symbol", "rule", "rank", "adjust_method", "ex_cum_factor"],
        )
        .filter(
            (pl.col("rule") == cfg.dominant.rule)
            & (pl.col("rank") == cfg.dominant.rank)
            & (pl.col("adjust_method") == cfg.adjust.method)
        )
        .select(
            pl.col("date").alias("trading_date"),
            pl.col("underlying_symbol").alias("symbol"),
            pl.col("ex_cum_factor").alias("official"),
        )
        .collect()
    )
    joined = (
        daily.select(["symbol", "trading_date", "adj_factor"])
        .join(official, on=["symbol", "trading_date"], how="inner")
        .drop_nulls(["adj_factor", "official"])
        .sort(["symbol", "trading_date"])
        .with_columns(
            rel_err=(
                (pl.col("adj_factor") / pl.col("adj_factor").first().over("symbol"))
                / (pl.col("official") / pl.col("official").first().over("symbol"))
                - 1.0
            ).abs()
        )
    )
    tol = cfg.adjust.verify_tolerance
    failures = (
        joined.filter(pl.col("rel_err") > tol)
        .group_by("symbol")
        .agg(
            pl.col("rel_err").max().alias("max_rel_err"),
            pl.col("trading_date").min().alias("first_bad_date"),
        )
        .sort("max_rel_err", descending=True)
    )
    fallback = rolls.filter(pl.col("k_fallback")).select(["symbol", "trading_date", "contract", "prev_contract"])
    max_err = float(joined["rel_err"].max()) if joined.height else 0.0
    return AdjustVerification(
        tolerance=tol,
        n_compared=joined.height,
        max_rel_err=max_err,
        failures=failures,
        fallback_rolls=fallback,
    )
