"""链式未来多期标签构建 (etl/labels.py)。

**标签仅用于训练与评估, 严禁作为 DSL 特征**, 因此写入独立的 ``labels.parquet``,
不进入 ``panel_daily.parquet``。

时空约定 (Q7): T 日 15:00 收盘后生成信号, **在 T+1 开盘 30 分钟 VWAP 建仓**, 持有 h 日,
在 T+1+h 日同窗口 VWAP 平仓。

定义面板行 d 上的单日链式 VWAP-to-VWAP 收益 (持有 c_d 从 d 日执行价到 d+1 日执行价):

    r_{d->d+1} = P_exec(c_d, d+1) / P_exec(c_d, d) - 1 = exec_vwap_old[d+1] / exec_vwap[d] - 1

其中 ``exec_vwap_old[d+1]`` 是面板第 d+1 行记录的“旧合约 c_d 在 d+1 日的执行价”。换月日
(c_{d+1} != c_d) 旧合约在 d+1 以其自身真实成交价平仓、新合约另以自身成交价开仓, 因此**不存在
跨合约作差的换月跳空**。

h 期标签:

    y_h(T) = prod_{k=1}^{h} (1 + r_{T+k -> T+k+1}) - 1

**相对白皮书 4.1 节文字的修正**: 白皮书写作 ``prod_{k=1}^{h}(1 + r_{T+k-1 -> T+k})``, 即从 **T 日**执行价
(T 日早盘, 早于 T 日收盘信号) 起算, 其收益区间覆盖了特征本身所处的交易日, 会把 T 日日内走势
(如 ``close/open-1`` 类因子) 直接泄露进标签并使 IC 虚高, 同时与“T+1 建仓”的执行约定自相矛盾。
本实现的标签起点为 T+1 日执行价, 与回测一致、无信息重叠。

任一环节价格缺失 (停牌/无成交) 时 ``y_h`` 为 null (严格, 不做填补)。
"""

from __future__ import annotations

from typing import Sequence

import polars as pl

from .calendar import dense_grid


def build_labels(
    panel: pl.DataFrame,
    cal: pl.DataFrame,
    horizons: Sequence[int],
) -> pl.DataFrame:
    """返回 ``trading_date, symbol, r_exec_next, y_{h}...``。

    ``panel`` 需含 ``symbol, trading_date, date_idx, exec_vwap, exec_vwap_old``。
    在品种稠密交易日网格上计算, 使得 T+k 严格对应第 k 个交易日而非第 k 行。
    """
    grid = dense_grid(panel.select(["symbol", "date_idx"]), cal).join(
        panel.select(["symbol", "date_idx", "exec_vwap", "exec_vwap_old"]),
        on=["symbol", "date_idx"],
        how="left",
    ).sort(["symbol", "date_idx"])

    # r_{d->d+1}: 分子取下一交易日行的 exec_vwap_old (即 c_d 在 d+1 日的价格)
    grid = grid.with_columns(
        _old_next=pl.col("exec_vwap_old").shift(-1).over("symbol"),
    ).with_columns(
        r_exec_next=pl.when((pl.col("exec_vwap") > 0) & (pl.col("_old_next") > 0))
        .then(pl.col("_old_next") / pl.col("exec_vwap") - 1.0)
        .otherwise(None)
    )
    grid = grid.with_columns(_lg=pl.col("r_exec_next").log1p())

    # y_h(T) 需要 r_{T+1->T+2} .. r_{T+h->T+h+1}, 即 lg 在行 T+1..T+h。先把 lg 前移一行
    # (行 T 持有 lg[T+1]), 再取从 T 起向前的 h 行窗口和 (稠密网格上行偏移 == 交易日偏移),
    # 这样 T 日自身的 r_{T->T+1} (起点为 T 日早盘, 早于 T 日收盘信号) 不会进入标签。
    grid = grid.with_columns(_lg_next=pl.col("_lg").shift(-1).over("symbol"))
    exprs = []
    for h in horizons:
        fwd_sum = (
            pl.col("_lg_next")
            .reverse()
            .rolling_sum(window_size=h, min_samples=h)
            .reverse()
            .over("symbol")
        )
        exprs.append(fwd_sum.exp().sub(1.0).alias(f"y_{h}"))
    grid = grid.with_columns(exprs)

    out = panel.select(["trading_date", "symbol", "date_idx"]).join(
        grid.select(["symbol", "date_idx", "r_exec_next", *[f"y_{h}" for h in horizons]]),
        on=["symbol", "date_idx"],
        how="left",
    )
    return out.drop("date_idx").sort(["trading_date", "symbol"])
