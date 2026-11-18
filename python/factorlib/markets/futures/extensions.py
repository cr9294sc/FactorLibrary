"""期货市场 1m 微观结构扩展特征包构建器 (markets/futures/extensions.py)。

职责与 `1d` 主面板完全解耦：
- 隔离处理期货夜盘跨日 `trading_date` 归属、`max_gap_minutes` 交易段跳空保护、
  换月边界保护与郑商所成交额置零保护（复用 `etl/minute_features.py`）；
- 输出标准化扩展包 `{pack_dir}/{features.parquet, pack.json}`，显式声明
  `lag_days=0` 与 `asof_rule="T_close"`（仅使用截至 T 日 15:00 的分钟 bar）。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import polars as pl

from factorlib.core.config import DataConfig, load_data_config
from factorlib.core.extensions import ExtensionPackMeta, save_extension_pack
from factorlib.markets.futures.etl.minute_features import FEATURE_COLUMNS, compute_minute_daily


class FuturesMinutePackBuilder:
    """期货 1m 日频扩展特征包离线抽取器。"""

    market: str = "futures"

    def __init__(self, root: Path | None = None) -> None:
        self.root = Path(root) if root is not None else Path(".")

    def build_pack(
        self,
        output_dir: Path,
        pack_id: str = "micro_futures_1m",
        version: str = "v1",
        *,
        daily_df: pl.DataFrame | None = None,
        raw_dir: Path | None = None,
        pairs: pl.DataFrame | None = None,
        data_config: DataConfig | None = None,
        lag_days: int = 0,
        asof_rule: str = "T_close",
        description: str = "期货 1m 日内微观结构扩展特征包（rv/ret_last30/vwap_dev 等）",
        **kwargs: Any,
    ) -> ExtensionPackMeta:
        """构建并落盘期货 `1m` 日频扩展特征包。"""
        del kwargs
        out_path = Path(output_dir)

        if daily_df is not None:
            df = daily_df
        else:
            cfg = data_config
            if cfg is None:
                cfg_path = self.root / "configs" / "futures" / "data.yaml"
                if cfg_path.exists():
                    cfg = load_data_config(cfg_path, root=self.root)
            if cfg is None:
                raise ValueError("构建期货 1m 扩展包需要提供 `daily_df` 或有效的 `data_config`")
            src_raw = Path(raw_dir) if raw_dir is not None else cfg.raw_dir
            if pairs is None:
                raise ValueError("从原始分钟线抽取期货扩展包需要提供主力映射表 `pairs`")

            minute_daily = compute_minute_daily(src_raw, pairs, cfg)
            # 按 (order_book_id, trading_date) 映射回品种主代码 `symbol`
            if "symbol" in pairs.columns:
                df = minute_daily.join(
                    pairs.select(["order_book_id", "trading_date", "symbol"]).unique(),
                    on=["order_book_id", "trading_date"],
                    how="inner",
                )
            else:
                df = minute_daily.rename({"order_book_id": "symbol"})

            keep_cols = ["trading_date", "symbol", *[c for c in FEATURE_COLUMNS if c in df.columns]]
            df = df.select(keep_cols)

        feature_cols = tuple(c for c in df.columns if c not in ("trading_date", "symbol"))
        meta = ExtensionPackMeta(
            pack_id=pack_id,
            market=self.market,
            version=version,
            kind="daily_features",
            fields=feature_cols,
            lag_days=lag_days,
            asof_rule=asof_rule,
            description=description,
        )
        return save_extension_pack(out_path, meta, df)
