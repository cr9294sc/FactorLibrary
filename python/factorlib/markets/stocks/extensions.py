"""A 股市场 1m 扩展特征包构建契约预留 (markets/stocks/extensions.py)。

遵循混合投研架构设计 §3.3.2 契约：
- 隔离处理 A 股 09:30–11:30 / 13:00–15:00 连续竞价、集合竞价剔除、后复权对齐与分片并行；
- 第一阶段仅建立接口协议与测试/预置扩展包挂载能力，原始 A 股 1m 离线流式抽取器将在后续阶段接入。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import polars as pl

from factorlib.core.extensions import ExtensionPackMeta, save_extension_pack


class StocksMinutePackBuilder:
    """A 股 1m 日频扩展特征包抽取协议实现（原始 1m 抽取预留）。"""

    market: str = "stocks"

    def __init__(self, root: Path | None = None) -> None:
        self.root = Path(root) if root is not None else Path(".")

    def build_pack(
        self,
        output_dir: Path,
        pack_id: str = "micro_stocks_1m",
        version: str = "v1",
        *,
        daily_df: pl.DataFrame | None = None,
        lag_days: int = 0,
        asof_rule: str = "T_close",
        description: str = "A 股 1m 日内连续竞价微观结构扩展特征包",
        **kwargs: Any,
    ) -> ExtensionPackMeta:
        """若传入已聚合的 `daily_df` 则标准化落盘，否则提示原始 1m 抽取接口预留。"""
        del kwargs
        if daily_df is None:
            raise NotImplementedError(
                "A 股原始 1m 分钟线离线流式抽取器尚在预留阶段；"
                "可传入已聚合日频扩展特征 DataFrame (`daily_df`) 进行打包测试。"
            )
        feature_cols = tuple(c for c in daily_df.columns if c not in ("trading_date", "symbol"))
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
        return save_extension_pack(Path(output_dir), meta, daily_df)
