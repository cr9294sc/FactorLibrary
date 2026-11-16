"""核心层依赖倒置钩子 (core/_hooks.py)。

使 `core/` 在完全不导入 `factorlib.markets` 的前提下，能够将 `profile=None` 的
旧式调用按 `panel.meta["market"]` 惰性分发到已注册的 `MarketProfile` 实现。
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

_PROFILE_RESOLVER: Callable[[str], Any] | None = None


def register_profile_resolver(resolver: Callable[[str], Any]) -> None:
    """由顶层包或市场注册中心注入按市场名称解析 `MarketProfile` 的回调。"""
    global _PROFILE_RESOLVER
    _PROFILE_RESOLVER = resolver


def resolve_profile(market: str = "futures") -> Any:
    """解析指定市场名称对应的 `MarketProfile` 实例。"""
    if _PROFILE_RESOLVER is None:
        raise RuntimeError(
            "尚未注册 MarketProfile 解析器，请显式传入 `profile` 或先导入 `factorlib.markets.registry`。"
        )
    return _PROFILE_RESOLVER(market)


def resolve_panel_market(panel: Any, default: str = "futures") -> str:
    """从 Panel.meta 中安全读取所属市场名称。"""
    meta = getattr(panel, "meta", None)
    if isinstance(meta, dict):
        market = meta.get("market")
        if isinstance(market, str) and market.strip():
            return market.strip().lower()
    return default
