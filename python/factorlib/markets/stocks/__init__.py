"""A 股股票市场适配层与面板加载体系 (markets/stocks)。"""

from factorlib.markets.stocks.ares_bridge import (
    build_lb_barra_panel,
    compute_causal_ex_cum_factor,
    load_lb_barra_panel,
    read_ares_table,
)
from factorlib.markets.stocks.backtest import StocksBacktester, run_stock_backtest_pipeline
from factorlib.markets.stocks.panel_loader import (
    STOCK_AUX_BACKTEST_FIELDS,
    StockPanelData,
    StockPanelLoader,
    build_stock_panel,
)
from factorlib.markets.stocks.profile import StocksProfile

__all__ = [
    "STOCK_AUX_BACKTEST_FIELDS",
    "StockPanelData",
    "StockPanelLoader",
    "StocksBacktester",
    "StocksProfile",
    "build_lb_barra_panel",
    "build_stock_panel",
    "compute_causal_ex_cum_factor",
    "load_lb_barra_panel",
    "read_ares_table",
    "run_stock_backtest_pipeline",
]
