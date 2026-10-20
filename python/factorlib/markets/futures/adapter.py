"""期货市场适配器实现 (markets/futures/adapter.py)。

按白皮书 3.1 节与 17 节设计:
- 封装期货交易日历、合约规格与费率成本模型;
- 协同 ETL 管道产出标准研究面板并加载为内存容器。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import yaml

from factorlib.core.panel import Panel
from factorlib.markets.base import ContractSpec, CostModel, MarketAdapter
from factorlib.markets.futures.etl.calendar import load_calendar


class FuturesMarketAdapter(MarketAdapter):
    name = "futures"

    def __init__(
        self,
        raw_dir: Path = Path("./exports"),
        fees_config_path: Path = Path("./configs/futures/fees.yaml"),
    ) -> None:
        self.raw_dir = Path(raw_dir)
        self.fees_config_path = Path(fees_config_path)
        self._fees_cache: dict | None = None

    def _load_fees(self) -> dict:
        if self._fees_cache is None:
            if self.fees_config_path.exists():
                with self.fees_config_path.open("r", encoding="utf-8") as fh:
                    self._fees_cache = yaml.safe_load(fh) or {}
            else:
                self._fees_cache = {}
        return self._fees_cache

    def calendar(self) -> np.ndarray:
        cal = load_calendar(self.raw_dir)
        return cal["trading_date"].to_numpy()

    def universe_mask(self, panel: Panel) -> np.ndarray:
        return panel.mask

    def contract_spec(self, symbol: str) -> ContractSpec:
        fees = self._load_fees()
        prod = fees.get("products", {}).get(symbol, {})
        default = fees.get("default", {})

        multiplier = float(prod.get("multiplier", default.get("multiplier", 10.0)))
        tick_size = float(prod.get("tick_size", default.get("tick_size", 1.0)))
        margin_rate = float(prod.get("margin_rate", default.get("margin_rate", 0.10)))

        return ContractSpec(
            multiplier=multiplier,
            lot_size=1,
            tick_size=tick_size,
            margin_rate=margin_rate,
        )

    def cost_model(self, symbol: str) -> CostModel:
        fees = self._load_fees()
        prod = fees.get("products", {}).get(symbol, {})
        default = fees.get("default", {})

        fee_type = str(prod.get("fee_type", default.get("fee_type", "rate")))
        open_fee = float(prod.get("open", default.get("open", 0.00005)))
        close_fee = float(prod.get("close", default.get("close", 0.00005)))
        close_today_fee = float(prod.get("close_today", default.get("close_today", 0.00005)))
        slip_ticks = float(prod.get("slip_ticks", default.get("slip_ticks", 1.0)))

        return CostModel(
            fee_type=fee_type,
            open_fee=open_fee,
            close_fee=close_fee,
            close_today_fee=close_today_fee,
            slippage_ticks=slip_ticks,
        )
