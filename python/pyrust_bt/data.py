from __future__ import annotations
from typing import Any, Dict, List
import csv


def load_csv_to_bars(path: str, symbol: str = "SPY") -> List[Dict[str, Any]]:
    """
    Load CSV with headers: datetime,open,high,low,close,volume
    Returns list of dicts suitable for the Rust engine MVP.
    """
    bars: List[Dict[str, Any]] = []
    with open(path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            bars.append(
                {
                    "datetime": row.get("datetime"),
                    "open": float(row["open"]) if row.get("open") else 0.0,
                    "high": float(row["high"]) if row.get("high") else 0.0,
                    "low": float(row["low"]) if row.get("low") else 0.0,
                    "close": float(row["close"]) if row.get("close") else 0.0,
                    "volume": float(row["volume"]) if row.get("volume") else 0.0,
                    "symbol": symbol,
                }
            )
    return bars 