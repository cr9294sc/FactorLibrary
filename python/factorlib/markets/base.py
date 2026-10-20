"""市场适配器抽象基类与通用规格定义 (markets/base.py)。

实现「市场无关计算内核 (Core)」与「市场适配层 (MarketAdapter)」的解耦:
- 适配器统一输出规整的面板矩阵与品种池掩码;
- 隔离不同市场的交易规则 (期货 T+0 / 双向杠杆 / 换月展期 vs 股票 T+1 / 仅多头 / 送转分红复权);
- 第一期实现商品与股指期货市场适配器 (FuturesMarketAdapter), 第二期无缝接入 A 股市场。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np

from factorlib.core.panel import Panel


@dataclass(frozen=True)
class ContractSpec:
    """合约基本交易规格。"""

    multiplier: float = 1.0
    lot_size: int = 1
    tick_size: float = 1.0
    margin_rate: float = 0.10


@dataclass(frozen=True)
class CostModel:
    """多维交易成本模型。"""

    fee_type: str = "rate"  # "rate" (按成交额比例) | "fixed" (按手固定金额)
    open_fee: float = 0.00005
    close_fee: float = 0.00005
    close_today_fee: float = 0.00005
    slippage_ticks: float = 1.0


class MarketAdapter(ABC):
    """市场适配器抽象基类。"""

    name: str = "base"

    @abstractmethod
    def calendar(self) -> np.ndarray:
        """获取市场统一交易日历。"""
        pass

    @abstractmethod
    def universe_mask(self, panel: Panel) -> np.ndarray:
        """计算给定面板在各交易日的可交易品种池布尔矩阵 (T x N)。"""
        pass

    @abstractmethod
    def contract_spec(self, symbol: str) -> ContractSpec:
        """获取指定品种的合约规格。"""
        pass

    @abstractmethod
    def cost_model(self, symbol: str) -> CostModel:
        """获取指定品种的交易成本模型。"""
        pass
