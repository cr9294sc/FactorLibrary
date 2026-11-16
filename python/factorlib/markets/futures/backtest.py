"""期货实盘离散手数与逐日盯市回测执行器 (markets/futures/backtest.py)。"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import factorlib_core
import numpy as np

from factorlib.core.backtest import BacktestResult
from factorlib.core.config import BacktestConfig
from factorlib.core.panel import Panel
from factorlib.markets.futures.adapter import FuturesMarketAdapter


def run_futures_backtest_pipeline(
    panel: Panel,
    weights: np.ndarray,
    config: BacktestConfig | None = None,
    adapter: FuturesMarketAdapter | None = None,
) -> BacktestResult:
    """执行期货实盘离散手数与逐日盯市回测。"""
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


class FuturesBacktester:
    """期货 `Backtester` 协议实现。"""

    def __init__(self, fees_config_path: Path = Path("configs/futures/fees.yaml")) -> None:
        self.fees_config_path = Path(fees_config_path)

    def run(
        self,
        panel: Panel,
        weights: np.ndarray,
        config: Any | None = None,
        **kwargs: Any,
    ) -> BacktestResult:
        adapter = kwargs.get("adapter")
        if adapter is None:
            adapter = FuturesMarketAdapter(fees_config_path=self.fees_config_path)
        bt_cfg = config if isinstance(config, BacktestConfig) else BacktestConfig()
        return run_futures_backtest_pipeline(
            panel=panel, weights=weights, config=bt_cfg, adapter=adapter
        )
