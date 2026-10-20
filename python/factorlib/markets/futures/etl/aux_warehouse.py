"""仓单库存标准化 (etl/aux_warehouse.py)。

各品种仓单物理单位不一致 (吨/张/千克), **严禁跨品种直接比较绝对值**, 统一转为品种内相对量:

    wh_chg5 = (stock_T - stock_{T-5}) / (stock_{T-5} + eps)
    wh_z60  = (stock_T - mu_60) / (sigma_60 + eps)

- 回看按**交易日历索引**计算 (品种稠密网格), 缺失日不会让 T-5 偏移;
- ``mu_60 / sigma_60`` 为含 T 日在内最近 60 个交易日的滚动均值与样本标准差 (ddof=1), 窗口内有效样本
  不足 ``z_min_samples`` 时为 null; 全部只使用 <=T 数据;
- ``eps`` 取 1 个计数单位 (默认 1.0): 仓单为吨/张量级的计数量, 1.0 的平滑项既避免分母为 0 又不会
  在库存逼近 0 时数值爆炸 (白皮书只写 eps, 未给量级)。
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import polars as pl

from factorlib.core.config import DataConfig

from .calendar import dense_grid, lag_by_idx
from .io import scan_dataset


def build_warehouse_features(
    raw_dir: Path,
    cfg: DataConfig,
    daily: pl.DataFrame,
    cal: pl.DataFrame,
    end_date: date,
) -> pl.DataFrame:
    """返回 ``symbol, trading_date, wh_chg5, wh_z60`` (与 ``daily`` 的 symbol-date 行一一对应)。"""
    wcfg = cfg.warehouse
    stock = (
        scan_dataset(raw_dir, "warehouse_stocks", ["trading_date", "underlying_symbol", "on_warrant"])
        .filter(pl.col("trading_date").is_between(cfg.start_date, end_date))
        .rename({"underlying_symbol": "symbol", "on_warrant": "stock"})
        .unique(subset=["symbol", "trading_date"], keep="first")
        .collect()
    )
    keys = daily.select(["symbol", "trading_date", "date_idx"])
    grid = dense_grid(keys, cal).join(stock, on=["symbol", "trading_date"], how="left").sort(["symbol", "date_idx"])

    grid = lag_by_idx(grid, ["stock"], wcfg.chg_lag, suffix="_lag")
    grid = grid.with_columns(
        wh_chg5=(pl.col("stock") - pl.col("stock_lag")) / (pl.col("stock_lag") + wcfg.eps),
        _mu=pl.col("stock")
        .rolling_mean_by("date_idx", window_size=f"{wcfg.z_window}i", min_samples=wcfg.z_min_samples, closed="right")
        .over("symbol"),
        _sd=pl.col("stock")
        .rolling_std_by("date_idx", window_size=f"{wcfg.z_window}i", min_samples=wcfg.z_min_samples, closed="right")
        .over("symbol"),
    ).with_columns(wh_z60=(pl.col("stock") - pl.col("_mu")) / (pl.col("_sd") + wcfg.eps))
    return keys.select(["symbol", "trading_date"]).join(
        grid.select(["symbol", "trading_date", "wh_chg5", "wh_z60"]),
        on=["symbol", "trading_date"],
        how="left",
    )
