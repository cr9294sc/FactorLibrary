"""A 股股票市场适配层与面板加载体系 (markets/stocks)。"""

from factorlib.markets.stocks.ares_bridge import (
    build_lb_barra_panel,
    compute_causal_ex_cum_factor,
    load_lb_barra_panel,
    read_ares_table,
)
from factorlib.markets.stocks.backtest import StocksBacktester, run_stock_backtest_pipeline
from factorlib.markets.stocks.barra import (
    BarraRiskModelStore,
    DynamicWLSResult,
    compute_constrained_wls_idiosyncratic_returns,
    estimate_dynamic_specific_variance,
)
from factorlib.markets.stocks.bridge import BridgeAuditReport, RiskView, SignalBridge
from factorlib.markets.stocks.panel_loader import (
    STOCK_AUX_BACKTEST_FIELDS,
    StockPanelData,
    StockPanelLoader,
    build_stock_panel,
)
from factorlib.markets.stocks.profile import StocksProfile

__all__ = [
    "BarraRiskModelStore",
    "BridgeAuditReport",
    "DynamicWLSResult",
    "RiskView",
    "STOCK_AUX_BACKTEST_FIELDS",
    "SignalBridge",
    "StockPanelData",
    "StockPanelLoader",
    "StocksBacktester",
    "StocksProfile",
    "build_lb_barra_panel",
    "build_stock_panel",
    "compute_causal_ex_cum_factor",
    "compute_constrained_wls_idiosyncratic_returns",
    "estimate_dynamic_specific_variance",
    "load_lb_barra_panel",
    "read_ares_table",
    "run_stock_backtest_pipeline",
]
