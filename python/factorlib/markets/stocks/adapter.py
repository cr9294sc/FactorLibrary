"""A 股股票市场适配器占位实现 (markets/stocks/adapter.py)。

按白皮书 17 节设计, 为二期 A 股市场因子接入保留标准契约。
第一期直接抛出 NotImplementedError。
"""

from __future__ import annotations

import numpy as np

from factorlib.core.panel import Panel
from factorlib.markets.base import ContractSpec, CostModel, MarketAdapter


class StockMarketAdapter(MarketAdapter):
    name = "stocks"

    def calendar(self) -> np.ndarray:
        raise NotImplementedError("A 股股票市场适配器将在工程计划二期 (Phase 2) 完整实现。")

    def universe_mask(self, panel: Panel) -> np.ndarray:
        raise NotImplementedError("A 股股票市场适配器将在工程计划二期 (Phase 2) 完整实现。")

    def contract_spec(self, symbol: str) -> ContractSpec:
        # A 股标准规格: 1 手 = 100 股, 乘数恒为 1.0
        return ContractSpec(multiplier=1.0, lot_size=100, tick_size=0.01)

    def cost_model(self, symbol: str) -> CostModel:
        raise NotImplementedError("A 股股票市场适配器将在工程计划二期 (Phase 2) 完整实现。")
