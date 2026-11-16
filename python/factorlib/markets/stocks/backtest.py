"""A 股持仓级向量化回测执行器 (markets/stocks/backtest.py)。"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import factorlib_core
import numpy as np

from factorlib.core.backtest import BacktestResult
from factorlib.core.config import (
    StockBacktestConfig,
    StockFeesConfig,
    load_stock_backtest_config,
    load_stock_fees_config,
)
from factorlib.core.panel import Panel


def _resolve_exec_and_close_prices(
    panel: Panel,
    price_basis: str,
    entry_window: str,
) -> tuple[np.ndarray, np.ndarray]:
    """根据 price_basis ('hfq' | 'raw') 与 entry_window 解析股票执行价与盯市收盘价。"""
    if price_basis == "raw":
        if entry_window == "vwap_open_30m" and panel.has_any_field("exec_vwap"):
            exec_p = panel.get_any_field("exec_vwap")
        elif panel.has_any_field("open_raw"):
            exec_p = panel.get_any_field("open_raw")
        elif panel.has_any_field("open"):
            exec_p = panel.get_any_field("open")
        else:
            exec_p = panel.get_any_field("open_adj")

        if panel.has_any_field("close_raw"):
            close_p = panel.get_any_field("close_raw")
        elif panel.has_any_field("close"):
            close_p = panel.get_any_field("close")
        else:
            close_p = panel.get_any_field("close_hfq")
        return exec_p, close_p

    # 默认 hfq 复权口径
    if entry_window == "vwap_open_30m" and panel.has_any_field("vwap_adj"):
        exec_p = panel.get_any_field("vwap_adj")
    elif panel.has_any_field("open_adj"):
        exec_p = panel.get_any_field("open_adj")
    elif panel.has_any_field("open_raw"):
        exec_p = panel.get_any_field("open_raw")
    else:
        exec_p = panel.get_any_field("close_hfq")

    if panel.has_any_field("close_hfq"):
        close_p = panel.get_any_field("close_hfq")
    elif panel.has_any_field("close"):
        close_p = panel.get_any_field("close")
    else:
        close_p = exec_p
    return exec_p, close_p


def run_stock_backtest_pipeline(
    panel: Panel,
    weights: np.ndarray,
    config: StockBacktestConfig | None = None,
    fees: StockFeesConfig | None = None,
    benchmark_returns: np.ndarray | None = None,
) -> BacktestResult:
    """执行 A 股 T+1 整手持仓级回测并返回统一 `BacktestResult`。"""
    bt_cfg = config if config is not None else StockBacktestConfig()
    fee_cfg = fees if fees is not None else StockFeesConfig()

    T, N = panel.shape
    if weights.shape != (T, N):
        raise ValueError(f"weights 形状 {weights.shape} 必须与 panel 形状 ({T}, {N}) 一致")

    exec_p, close_p = _resolve_exec_and_close_prices(
        panel,
        price_basis=bt_cfg.execution.price_basis,
        entry_window=bt_cfg.execution.entry_window,
    )

    if panel.has_any_field("is_suspended"):
        is_susp = panel.get_any_field("is_suspended") > 0.5
    else:
        is_susp = np.zeros((T, N), dtype=bool, order="F")

    if panel.has_any_field("can_buy_mask"):
        can_buy = panel.get_any_field("can_buy_mask") > 0.5
    elif "limit_mask" in panel.fields:
        can_buy = np.isfinite(panel.fields["limit_mask"])
    else:
        can_buy = np.ones((T, N), dtype=bool, order="F")

    if panel.has_any_field("can_sell_mask"):
        can_sell = panel.get_any_field("can_sell_mask") > 0.5
    elif "limit_mask" in panel.fields:
        can_sell = np.isfinite(panel.fields["limit_mask"])
    else:
        can_sell = np.ones((T, N), dtype=bool, order="F")

    bench_arr = benchmark_returns
    if bench_arr is None and panel.has_any_field("bench_ret"):
        cand = panel.get_any_field("bench_ret")[:, 0]
        if np.any(np.isfinite(cand)):
            bench_arr = np.ascontiguousarray(cand, dtype=np.float64)

    raw_res = factorlib_core.run_stock_backtest(
        target_weights=np.asfortranarray(weights, dtype=np.float64),
        exec_price=np.asfortranarray(exec_p, dtype=np.float64),
        close_price=np.asfortranarray(close_p, dtype=np.float64),
        can_buy_mask=np.asfortranarray(can_buy, dtype=bool),
        can_sell_mask=np.asfortranarray(can_sell, dtype=bool),
        is_suspended=np.asfortranarray(is_susp, dtype=bool),
        benchmark_returns=np.ascontiguousarray(bench_arr, dtype=np.float64)
        if bench_arr is not None
        else None,
        init_cash=float(bt_cfg.capital.init_cash),
        lot_size=int(bt_cfg.execution.lot_size),
        t_plus_one=bool(bt_cfg.execution.t_plus_one),
        allow_short=bool(bt_cfg.execution.allow_short),
        commission_rate=float(fee_cfg.commission_rate),
        min_commission=float(fee_cfg.min_commission),
        stamp_duty_rate=float(fee_cfg.stamp_duty_rate),
        transfer_fee_rate=float(fee_cfg.transfer_fee_rate),
        slip_ticks=float(fee_cfg.slip_ticks),
        tick_size=float(fee_cfg.tick_size),
        slip_bps=float(fee_cfg.slip_bps),
    )

    zeros_t = np.zeros(T, dtype=np.float64)
    extras: dict[str, Any] = {
        "tax": raw_res["tax"],
        "blocked_orders": raw_res["blocked_orders"],
        "excess_returns": raw_res["excess_returns"],
    }
    if bench_arr is not None:
        clean_b = np.where(np.isfinite(bench_arr), bench_arr, 0.0)
        extras["benchmark_returns"] = clean_b
        extras["benchmark_nav"] = np.cumprod(1.0 + clean_b)

    return BacktestResult(
        dates=panel.dates,
        symbols=panel.symbols,
        equity=raw_res["equity"],
        returns=raw_res["returns"],
        cash=raw_res["cash"],
        margin_used=zeros_t,
        leverage=raw_res["leverage"],
        gross_pnl=raw_res["gross_pnl"],
        net_pnl=raw_res["net_pnl"],
        fee=raw_res["fee"],
        slippage=raw_res["slippage"],
        roll_cost=zeros_t.copy(),
        turnover=raw_res["turnover"],
        positions=raw_res["positions"],
        metrics=raw_res["metrics"],
        extras=extras,
    )


class StocksBacktester:
    """A 股 `Backtester` 协议实现。"""

    def __init__(
        self,
        fees_config_path: Path = Path("configs/stocks/fees.yaml"),
        backtest_config_path: Path = Path("configs/stocks/backtest.yaml"),
    ) -> None:
        self.fees_config_path = Path(fees_config_path)
        self.backtest_config_path = Path(backtest_config_path)

    def run(
        self,
        panel: Panel,
        weights: np.ndarray,
        config: Any | None = None,
        **kwargs: Any,
    ) -> BacktestResult:
        fees_cfg = (
            load_stock_fees_config(self.fees_config_path)
            if self.fees_config_path.exists()
            else StockFeesConfig()
        )
        if isinstance(config, StockBacktestConfig):
            bt_cfg = config
        elif self.backtest_config_path.exists():
            bt_cfg = load_stock_backtest_config(self.backtest_config_path)
        else:
            bt_cfg = StockBacktestConfig()

        return run_stock_backtest_pipeline(
            panel=panel,
            weights=weights,
            config=bt_cfg,
            fees=fees_cfg,
            benchmark_returns=kwargs.get("benchmark_returns"),
        )
