"""FactorLibrary: 多市场量化因子挖掘与回测平台 (Python 研究层)。"""

from factorlib.core._hooks import register_profile_resolver


def _lazy_resolve_profile(market: str = "futures"):
    from factorlib.markets.registry import get_profile

    return get_profile(market)


register_profile_resolver(_lazy_resolve_profile)
