"""期货实盘语义回测 Python 调度层与结果封装 (core/backtest.py)。

按白皮书 12.2 节设计:
- 衔接 Panel 数据容器与 Rust 计算内核 `factorlib_core.run_futures_backtest` / `run_fast_backtest`;
- 统一获取期货各标的乘数、保证金率、手续费率与跳数滑点;
- 产出包含每日净值、收益、仓位、杠杆、成本明细与综合评估指标的强类型 `BacktestResult`。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import factorlib_core
import numpy as np
import polars as pl

from factorlib.core.config import BacktestConfig
from factorlib.core.panel import Panel
from factorlib.markets.futures.adapter import FuturesMarketAdapter


@dataclass
class BacktestResult:
    """期货实盘回测结果容器。

    Attributes
    ----------
    dates : np.ndarray
        一维交易日数组 (长度 T)。
    symbols : list[str]
        品种代码列表 (长度 N)。
    equity : np.ndarray
        每日总权益净值序列 (长度 T)。
    returns : np.ndarray
        每日净收益率序列 (长度 T)。
    cash : np.ndarray
        每日可用资金序列 (长度 T)。
    margin_used : np.ndarray
        每日保证金占用序列 (长度 T)。
    leverage : np.ndarray
        每日实际杠杆倍数序列 (长度 T)。
    gross_pnl : np.ndarray
        每日盯市毛盈亏序列 (长度 T)。
    net_pnl : np.ndarray
        每日净盈亏序列 (长度 T)。
    fee : np.ndarray
        每日手续费序列 (长度 T)。
    slippage : np.ndarray
        每日冲击滑点序列 (长度 T)。
    roll_cost : np.ndarray
        每日换月展期摩擦成本序列 (长度 T)。
    turnover : np.ndarray
        每日名义调仓换手率序列 (长度 T)。
    positions : np.ndarray
        每日收盘持仓整数手数矩阵 (T, N)。
    metrics : dict[str, Any]
        多维回测评估指标字典。
    """

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

    def to_dataframe(self) -> pl.DataFrame:
        """导出每日净值与财务流水平账 DataFrame。"""
        return pl.DataFrame(
            {
                "trading_date": self.dates,
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
        )

    def summary(self) -> str:
        """格式化输出回测关键绩效指标汇总字符串。"""
        m = self.metrics
        lines = [
            "==================================================",
            "        FactorLibrary 期货实盘回测绩效汇总        ",
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
            f" 保证金最大占用比率       : {m.get('max_margin_utilization', 0.0) * 100:.2f}%",
            f" 最大杠杆倍数 (Max Leverage): {m.get('max_leverage', 0.0):.2f}x",
            "==================================================",
        ]
        return "\n".join(lines)


def run_backtest(
    panel: Panel,
    weights: np.ndarray,
    config: BacktestConfig | None = None,
    adapter: FuturesMarketAdapter | None = None,
) -> BacktestResult:
    """执行期货实盘离散手数与逐日盯市回测。

    Parameters
    ----------
    panel : Panel
        研究面板容器，需包含行情价格与合约元数据。
    weights : np.ndarray
        形状为 (T, N) 的连续目标权重矩阵 (Fortran-order)。
    config : BacktestConfig, optional
        回测参数配置，默认使用默认配置。
    adapter : FuturesMarketAdapter, optional
        期货市场适配器，用于读取品种规格与费率模型。

    Returns
    -------
    BacktestResult
        完整的实盘回测结果对象。
    """
    if config is None:
        config = BacktestConfig()
    if adapter is None:
        adapter = FuturesMarketAdapter(
            fees_config_path=Path("configs/futures/fees.yaml"),
        )

    T, N = panel.shape
    if weights.shape != (T, N):
        raise ValueError(f"weights 形状 {weights.shape} 必须与 panel 形状 ({T}, {N}) 一致")

    # 1. 提取执行价格 exec_vwap 与换月平旧价 exec_vwap_old（优先 aux_fields，兼容手工构造面板）
    if panel.has_any_field("exec_vwap"):
        exec_price = panel.get_any_field("exec_vwap")
    elif panel.has_any_field("open_raw"):
        exec_price = panel.get_any_field("open_raw")
    elif panel.has_any_field("close_raw"):
        exec_price = panel.get_any_field("close_raw")
    else:
        exec_price = panel.get_field("close")

    if panel.has_any_field("exec_vwap_old"):
        exec_price_old = panel.get_any_field("exec_vwap_old")
    else:
        exec_price_old = exec_price

    # 2. 提取结算价格 settle_price
    if panel.has_any_field("settle_raw"):
        settle_price = panel.get_any_field("settle_raw")
    elif panel.has_any_field("settle"):
        settle_price = panel.get_any_field("settle")
    else:
        settle_price = exec_price

    # 3. 提取涨跌停与单边市标记
    if panel.has_any_field("limit_up"):
        limit_up = panel.get_any_field("limit_up")
    else:
        limit_up = np.full((T, N), np.inf, dtype=np.float64, order="F")

    if panel.has_any_field("limit_down"):
        limit_down = panel.get_any_field("limit_down")
    else:
        limit_down = np.full((T, N), 0.0, dtype=np.float64, order="F")

    if panel.has_any_field("is_unilateral"):
        is_unilateral = panel.get_any_field("is_unilateral") > 0.5
    else:
        is_unilateral = np.zeros((T, N), dtype=bool, order="F")

    if panel.has_any_field("is_roll"):
        is_roll = panel.get_any_field("is_roll") > 0.5
    else:
        is_roll = np.zeros((T, N), dtype=bool, order="F")

    # 4. 组装品种规格 specs
    specs = []
    for sym in panel.symbols:
        spec = adapter.contract_spec(sym)
        cost = adapter.cost_model(sym)
        specs.append(
            {
                "symbol": sym,
                "multiplier": float(spec.multiplier),
                "margin_rate": float(spec.margin_rate),
                "fee_type": str(cost.fee_type),
                "open_fee": float(cost.open_fee),
                "close_fee": float(cost.close_fee),
                "close_today_fee": float(cost.close_today_fee),
                "tick_size": float(spec.tick_size),
                "slip_ticks": float(cost.slippage_ticks),
            }
        )

    # 5. 调用 Rust 计算内核
    raw_res = factorlib_core.run_futures_backtest(
        target_weights=np.asfortranarray(weights, dtype=np.float64),
        exec_price=np.asfortranarray(exec_price, dtype=np.float64),
        exec_price_old=np.asfortranarray(exec_price_old, dtype=np.float64),
        settle_price=np.asfortranarray(settle_price, dtype=np.float64),
        limit_up=np.asfortranarray(limit_up, dtype=np.float64),
        limit_down=np.asfortranarray(limit_down, dtype=np.float64),
        is_unilateral=np.asfortranarray(is_unilateral, dtype=bool),
        is_roll=np.asfortranarray(is_roll, dtype=bool),
        specs=specs,
        init_cash=float(config.capital.init_cash),
        max_margin_ratio=float(config.execution.max_margin_ratio),
        margin_safety_multiplier=float(config.execution.margin_safety_multiplier),
        lot_rounding=str(config.execution.lot_rounding),
    )

    return BacktestResult(
        dates=panel.dates,
        symbols=panel.symbols,
        equity=raw_res["equity"],
        returns=raw_res["returns"],
        cash=raw_res["cash"],
        margin_used=raw_res["margin_used"],
        leverage=raw_res["leverage"],
        gross_pnl=raw_res["gross_pnl"],
        net_pnl=raw_res["net_pnl"],
        fee=raw_res["fee"],
        slippage=raw_res["slippage"],
        roll_cost=raw_res["roll_cost"],
        turnover=raw_res["turnover"],
        positions=raw_res["positions"],
        metrics=raw_res["metrics"],
    )


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
