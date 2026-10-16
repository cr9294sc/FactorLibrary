from .api import BacktestEngine, BacktestConfig
from .strategy import Strategy
from .analyzers import (
    compute_drawdown_segments,
    round_trips_from_trades,
    export_trades_to_csv,
    export_round_trips_to_csv,
    factor_backtest,
    compute_performance_metrics,
    generate_analysis_report,
)

# Import vectorized functions from Rust if available
try:
    from engine_rust import compute_sma, compute_rsi
except ImportError:
    compute_sma = None
    compute_rsi = None

__all__ = [
    "BacktestEngine",
    "BacktestConfig",
    "Strategy",
    "compute_drawdown_segments",
    "round_trips_from_trades",
    "export_trades_to_csv",
    "export_round_trips_to_csv",
    "factor_backtest",
    "compute_performance_metrics",
    "generate_analysis_report",
    "compute_sma",
    "compute_rsi",
] 