"""FactorLibrary Rust 极速计算内核 Python 导出入口 (factorlib_core)。"""

from .factorlib_core import (
    Engine,
    calc_correlation_matrix,
    calc_ic,
    calc_pairwise_correlation,
    calc_quantile_returns,
    calc_turnover,
    check_expr,
    list_operators,
    parse_and_canonicalize,
    run_fast_backtest,
    run_futures_backtest,
)

__all__ = [
    "Engine",
    "check_expr",
    "list_operators",
    "parse_and_canonicalize",
    "calc_ic",
    "calc_quantile_returns",
    "calc_turnover",
    "calc_pairwise_correlation",
    "calc_correlation_matrix",
    "run_fast_backtest",
    "run_futures_backtest",
]
