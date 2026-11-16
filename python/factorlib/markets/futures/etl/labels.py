"""链式未来多期标签构建 (etl/labels.py)。

**标签仅用于训练与评估, 严禁作为 DSL 特征**, 因此写入独立的 ``labels.parquet``,
不进入 ``panel_daily.parquet``。

时空约定 (Q7): T 日 15:00 收盘后生成信号, **在 T+1 开盘 30 分钟 VWAP 建仓**, 持有 h 日,
在 T+1+h 日同窗口 VWAP 平仓。

按重构架构 M1.3：
- 保留换月链式单步执行收益 ``r_{d->d+1} = exec_vwap_old[d+1] / exec_vwap[d] - 1`` 的 Polars 稠密网格计算；
- ``r_exec_next`` 平移与 h 期复合标签计算统一下沉至 Rust 内核 ``factorlib_core.compound_forward_labels``。
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import factorlib_core
import numpy as np
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
    h_list = [int(h) for h in horizons]
    if panel.is_empty():
        schema: dict[str, Any] = {
            "trading_date": panel.schema.get("trading_date", pl.Date),
            "symbol": pl.String,
            "r_exec_next": pl.Float64,
        }
        for h in h_list:
            schema[f"y_{h}"] = pl.Float64
        return pl.DataFrame(schema=schema)

    grid = (
        dense_grid(panel.select(["symbol", "date_idx"]), cal)
        .join(
            panel.select(["symbol", "date_idx", "exec_vwap", "exec_vwap_old"]),
            on=["symbol", "date_idx"],
            how="left",
        )
        .sort(["symbol", "date_idx"])
    )

    # r_{d->d+1}: 分子取下一交易日行的 exec_vwap_old (即 c_d 在 d+1 日的价格)
    grid = grid.with_columns(
        _old_next=pl.col("exec_vwap_old").shift(-1).over("symbol"),
    ).with_columns(
        _r_step=pl.when((pl.col("exec_vwap") > 0) & (pl.col("_old_next") > 0))
        .then(pl.col("_old_next") / pl.col("exec_vwap") - 1.0)
        .otherwise(None)
    )

    min_idx = int(grid["date_idx"].min())
    max_idx = int(grid["date_idx"].max())
    t_span = max_idx - min_idx + 1
    unique_syms = sorted(grid["symbol"].unique().to_list())
    n_syms = len(unique_syms)
    sym_map = {s: idx for idx, s in enumerate(unique_syms)}

    g_r = (grid["date_idx"].to_numpy() - min_idx).astype(np.int64)
    g_c = grid["symbol"].replace_strict(sym_map).to_numpy().astype(np.int64)
    step_mat = np.full((t_span, n_syms), np.nan, dtype=np.float64, order="F")
    step_mat[g_r, g_c] = grid["_r_step"].cast(pl.Float64).to_numpy()

    rust_labels = factorlib_core.compound_forward_labels(step_mat, h_list)

    p_base = panel.select(["trading_date", "symbol", "date_idx"])
    p_r = (p_base["date_idx"].to_numpy() - min_idx).astype(np.int64)
    p_c = p_base["symbol"].replace_strict(sym_map).to_numpy().astype(np.int64)

    cols: dict[str, pl.Series] = {
        "trading_date": p_base["trading_date"],
        "symbol": p_base["symbol"],
        "r_exec_next": pl.Series("r_exec_next", np.asarray(rust_labels["r_exec_next"])[p_r, p_c]),
    }
    for h in h_list:
        key = f"y_{h}"
        cols[key] = pl.Series(key, np.asarray(rust_labels[key])[p_r, p_c])

    return pl.DataFrame(cols).sort(["trading_date", "symbol"])
