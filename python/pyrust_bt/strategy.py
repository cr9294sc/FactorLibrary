from __future__ import annotations
from typing import Any, Dict


class Strategy:
    def on_start(self, ctx: Any) -> None:
        """回测开始时调用。"""
        pass

    def next(self, bar: Dict[str, Any]) -> str | Dict[str, Any] | None:
        """
        每根 bar 调用。返回以下之一：
        - 字符串："BUY" 或 "SELL"（市价单、默认 size=1）
        - 字典：{"action": "BUY"|"SELL", "type": "market"|"limit", "size": float, "price"?: float}
        - None：不下单
        """
        return None

    def on_order(self, event: Dict[str, Any]) -> None:
        """订单事件：submitted/filled 等。"""
        pass

    def on_trade(self, event: Dict[str, Any]) -> None:
        """成交事件。包含 order_id/side/price/size。"""
        pass

    def on_stop(self) -> None:
        """回测结束时调用。"""
        pass 