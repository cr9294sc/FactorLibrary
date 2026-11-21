"""多市场实盘语义回测调度层与统一结果容器 (core/backtest.py)。

按重构架构 M1 设计：
- 统一封装跨市场强类型 `BacktestResult`（新增 `extras: dict[str, Any]` 承载市场专属扩展字段，
  如股票市场的 `tax`、`blocked_orders`、`benchmark_returns`、`excess_returns`）；
- `run_backtest(panel, weights, config=None, profile=None)` 仅负责按 `MarketProfile` 分发，
  不再直接依赖期货市场适配器实现。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import factorlib_core
import numpy as np
import polars as pl

from factorlib.core._hooks import resolve_panel_market, resolve_profile
from factorlib.core.panel import Panel


@dataclass
class BacktestResult:
    """统一多市场实盘回测结果容器。"""

    dates: np.ndarray
    symbols: list[str]
    equity: np.ndarray
    returns: np.ndarray
    cash: np.ndarray
    margin_used: np.ndarray
    leverage: np.ndarray
    gross_pnl: np.ndarray
    net_pnl: np.ndarray
    fee: np.ndarray
    slippage: np.ndarray
    roll_cost: np.ndarray
    turnover: np.ndarray
    positions: np.ndarray
    metrics: dict[str, Any] = field(default_factory=dict)
    extras: dict[str, Any] = field(default_factory=dict)

    def to_dataframe(self) -> pl.DataFrame:
        """导出每日净值与财务流水平账 DataFrame。"""
        dates_col: Any = (
            [str(d) for d in self.dates]
            if isinstance(self.dates, np.ndarray) and self.dates.dtype == object
            else self.dates
        )
        cols: dict[str, Any] = {
            "trading_date": dates_col,
            "equity": self.equity,
            "returns": self.returns,
            "cash": self.cash,
            "margin_used": self.margin_used,
            "leverage": self.leverage,
            "gross_pnl": self.gross_pnl,
            "net_pnl": self.net_pnl,
            "fee": self.fee,
            "slippage": self.slippage,
            "roll_cost": self.roll_cost,
            "turnover": self.turnover,
        }
        t_len = len(self.dates)
        for k, v in self.extras.items():
            if isinstance(v, np.ndarray) and v.ndim == 1 and len(v) == t_len:
                cols[k] = v
            elif isinstance(v, list) and len(v) == t_len:
                cols[k] = v
        return pl.DataFrame(cols)

    def summary(self) -> str:
        """格式化输出回测关键绩效指标汇总字符串。"""
        m = self.metrics
        lines = [
            "==================================================",
            "        FactorLibrary 实盘语义回测绩效汇总        ",
            "==================================================",
            f" 初始资金 (Capital)       : {self.equity[0]:,.2f}",
            f" 期末权益 (Final Equity)  : {self.equity[-1]:,.2f}",
            f" 累计收益率 (Total Return): {m.get('total_return', 0.0) * 100:.2f}%",
            f" 年化收益率 (Annual Return): {m.get('annual_return', 0.0) * 100:.2f}%",
            f" 年化波动率 (Annual Vol)  : {m.get('annual_volatility', 0.0) * 100:.2f}%",
            f" 夏普比率 (Sharpe Ratio)  : {m.get('sharpe', 0.0):.2f}",
            f" 索提诺比率 (Sortino Ratio): {m.get('sortino', 0.0):.2f}",
            f" 卡尔玛比率 (Calmar Ratio): {m.get('calmar', 0.0):.2f}",
            f" 最大回撤 (Max Drawdown)  : {m.get('max_drawdown', 0.0) * 100:.2f}%",
            f" 最大回撤天数 (Max DD Days): {m.get('max_drawdown_days', 0)} 天",
            f" 胜率 (Win Rate)          : {m.get('win_rate', 0.0) * 100:.2f}%",
            f" 盈亏比 (Profit/Loss Ratio): {m.get('profit_loss_ratio', 0.0):.2f}",
            f" 日均换手率 (Daily Turnover): {m.get('daily_turnover_mean', 0.0) * 100:.2f}%",
            f" 年化换手率 (Annual Turnover): {m.get('annual_turnover', 0.0):.2f}x",
            f" 累计交易佣金 (Total Fees) : {m.get('total_fees', 0.0):,.2f}",
            f" 累计滑点成本 (Total Slip) : {m.get('total_slippage', 0.0):,.2f}",
            f" 累计换月成本 (Total Roll) : {m.get('total_roll_cost', 0.0):,.2f}",
            f" 累计印花税 (Total Tax)   : {m.get('total_tax', 0.0):,.2f}",
            f" 保证金最大占用比率       : {m.get('max_margin_utilization', 0.0) * 100:.2f}%",
            f" 最大杠杆倍数 (Max Leverage): {m.get('max_leverage', 0.0):.2f}x",
            "==================================================",
        ]
        return "\n".join(lines)


def run_backtest(
    panel: Panel,
    weights: np.ndarray,
    config: Any | None = None,
    adapter: Any | None = None,
    profile: Any | None = None,
) -> BacktestResult:
    """按 `MarketProfile` 分发执行实盘持仓级回测。"""
    resolved_profile = profile
    if resolved_profile is None:
        market = getattr(adapter, "name", None) if adapter is not None else None
        if not isinstance(market, str) or not market:
            market = resolve_panel_market(panel, default="futures")
        resolved_profile = resolve_profile(market)

    kwargs: dict[str, Any] = {}
    if adapter is not None:
        kwargs["adapter"] = adapter
    return resolved_profile.backtester().run(panel, weights, config=config, **kwargs)


def run_fast_backtest(
    weights: np.ndarray,
    returns: np.ndarray,
    fee_rate: float = 0.0001,
    slippage_bps: float = 1.0,
    init_cash: float = 10_000_000.0,
) -> dict[str, Any]:
    """执行快速连续权重回测 (基于 Rust 内核)。"""
    return factorlib_core.run_fast_backtest(
        weights=np.asfortranarray(weights, dtype=np.float64),
        returns=np.asfortranarray(returns, dtype=np.float64),
        fee_rate=fee_rate,
        slippage_bps=slippage_bps,
        init_cash=init_cash,
    )
