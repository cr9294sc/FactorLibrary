"""市场 Profile 注册中心 (markets/registry.py)。"""

from __future__ import annotations

from collections.abc import Callable

from factorlib.markets.profile import MarketProfile

PROFILES: dict[str, Callable[[], MarketProfile]] = {}


def register_profile(name: str, factory: Callable[[], MarketProfile]) -> None:
    """注册市场 Profile 工厂函数。"""
    key = str(name).strip().lower()
    if not key:
        raise ValueError("market profile name 不能为空")
    PROFILES[key] = factory


def get_profile(name: str = "futures") -> MarketProfile:
    """按市场名称获取对应 `MarketProfile` 实例。"""
    key = str(name).strip().lower()
    if key not in PROFILES:
        raise KeyError(f"未知的市场 Profile: {name!r}，已注册市场: {sorted(PROFILES.keys())}")
    return PROFILES[key]()


def list_profiles() -> list[str]:
    """返回所有已注册市场名称列表。"""
    return sorted(PROFILES.keys())


def _register_builtin_profiles() -> None:
    def _make_futures() -> MarketProfile:
        from factorlib.markets.futures.profile import FuturesProfile

        return FuturesProfile()

    def _make_stocks() -> MarketProfile:
        from factorlib.markets.stocks.profile import StocksProfile

        return StocksProfile()

    register_profile("futures", _make_futures)
    register_profile("stocks", _make_stocks)


_register_builtin_profiles()
