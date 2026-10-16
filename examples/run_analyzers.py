from __future__ import annotations
from typing import Any, Dict, List
import os
import sys

# Allow running from repo root
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "python"))

from pyrust_bt.api import BacktestConfig, BacktestEngine
from pyrust_bt.data import load_csv_to_bars
from pyrust_bt.strategy import Strategy
from pyrust_bt.analyzers import compute_drawdown_segments, round_trips_from_trades, export_trades_to_csv, factor_backtest

# Try vectorized SMA from Rust if available
try:
    from engine_rust import compute_sma as _rs_sma
except Exception:
    _rs_sma = None


def compute_sma_py(closes: List[float], window: int) -> List[float | None]:
    if window <= 0:
        raise ValueError("window must be > 0")
    out: List[float | None] = []
    s = 0.0
    for i, v in enumerate(closes):
        s += v
        if i + 1 < window:
            out.append(None)
        elif i + 1 == window:
            out.append(s / window)
        else:
            s -= closes[i - window]
            out.append(s / window)
    return out


class FactorStrategy(Strategy):
    def __init__(self, window: int = 3, size: float = 1.0) -> None:
        self.window = window
        self.size = size
        self._closes: List[float] = []

    def next(self, bar: Dict[str, Any]) -> Dict[str, Any] | None:
        close = float(bar["close"])  # type: ignore[assignment]
        self._closes.append(close)
        if len(self._closes) < self.window:
            return None
        sma = sum(self._closes[-self.window :]) / self.window
        if close > sma:
            return {"action": "BUY", "type": "market", "size": self.size}
        elif close < sma:
            return {"action": "SELL", "type": "market", "size": self.size}
        return None


def main() -> None:
    cfg = BacktestConfig(
        start="2020-01-01",
        end="2020-12-31",
        cash=10000.0,
        commission_rate=0.0005,
        slippage_bps=2.0,
        batch_size=2000,
    )
    engine = BacktestEngine(cfg)

    data_path = os.path.join(os.path.dirname(__file__), "data", "sample.csv")
    bars = load_csv_to_bars(data_path, symbol="SAMPLE")

    # Pre-compute factor: close - SMA(window)
    window = 3
    closes = [float(b["close"]) for b in bars]
    if _rs_sma is not None:
        sma = _rs_sma(closes, window)
    else:
        sma = compute_sma_py(closes, window)
    for i, b in enumerate(bars):
        f = None if sma[i] is None else (closes[i] - float(sma[i]))
        b["factor"] = 0.0 if f is None else float(f)

    strat = FactorStrategy(window=window, size=1.0)
    res = engine.run(strat, bars)

    equity_curve = res["equity_curve"]
    trades = res["trades"]

    segs = compute_drawdown_segments(equity_curve)
    print("Top drawdown segments:")
    for s in segs[:3]:
        print(s)

    rts = round_trips_from_trades(trades)
    print("Round trips (first 3):", rts[:3])

    out_csv = os.path.join(os.path.dirname(__file__), "data", "trades_out.csv")
    export_trades_to_csv(trades, out_csv)
    print("Trades exported to:", out_csv)

    # Factor backtest over bars (factor is precomputed into bars)
    fb = factor_backtest(bars, factor_key="factor", quantiles=3, forward=1)
    print("Factor backtest:", fb)


if __name__ == "__main__":
    main() 