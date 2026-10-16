from __future__ import annotations
from typing import Any, Dict, List
import os
import sys

# Allow running from repo root
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "python"))

from pyrust_bt.api import BacktestConfig
from pyrust_bt.data import load_csv_to_bars
from pyrust_bt.optimize import grid_search
from pyrust_bt.strategy import Strategy


class DummySMAStrategy(Strategy):
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
    )

    data_path = os.path.join(os.path.dirname(__file__), "data", "sample.csv")
    bars = load_csv_to_bars(data_path, symbol="SAMPLE")

    results = grid_search(
        cfg,
        bars,
        DummySMAStrategy,
        param_grid={"window": [3, 4, 5, 6, 7], "size": [1.0]},
        score_key="total_return",
    )

    print("Top results (by total_return):")
    for params, res in results[:3]:
        print(params, res.get("stats"))


if __name__ == "__main__":
    main() 