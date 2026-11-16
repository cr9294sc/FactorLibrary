"""期货市场 Profile 实现 (`FuturesProfile`, markets/futures/profile.py)。"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import factorlib_core
import numpy as np
import yaml

from factorlib.core.catalog import RECOMMENDED_POSTPROCESS_PRESETS
from factorlib.core.config import PortfolioConfig
from factorlib.core.panel import Panel
from factorlib.core.probe import PostprocessSpec
from factorlib.markets.futures.backtest import FuturesBacktester
from factorlib.markets.futures.etl.fields import AUX_BACKTEST_FIELDS
from factorlib.markets.futures.portfolio import build_futures_portfolio_weights
from factorlib.markets.futures.styles import compute_futures_styles
from factorlib.markets.profile import (
    Backtester,
    CostModelSpec,
    ExecutionRules,
    PanelSpec,
    ResearchPaths,
)


class FuturesProfile:
    """商品与金融期货研究域 `MarketProfile` 实现。"""

    name: str = "futures"

    def __init__(self, root: Path | None = None) -> None:
        self.root = Path(root) if root is not None else Path(".")

    def paths(self, version: str = "v1") -> ResearchPaths:
        derived_dir = self.root / "data" / "derived" / "futures"
        return ResearchPaths(
            market="futures",
            version=version,
            raw_dir=self.root / "exports",
            derived_dir=derived_dir,
            panel_dir=derived_dir / version,
            db_path=self.root / "data" / "factors.sqlite",
            runs_dir=self.root / "runs",
            config_dir=self.root / "configs" / "futures",
        )

    def load_panel(self, spec: PanelSpec | None = None) -> Panel:
        s = spec or PanelSpec()
        panel_dir = s.panel_dir or self.paths(s.version).panel_dir
        panel = Panel.load(
            panel_dir,
            fields_subset=s.fields_subset,
            load_labels=s.load_labels,
            aux_field_names=AUX_BACKTEST_FIELDS,
        )
        panel.meta.setdefault("market", self.name)
        return panel

    def build_labels(self, panel: Panel, horizons: Sequence[int]) -> dict[str, np.ndarray]:
        """基于 panel 中的 `exec_vwap` 与 `exec_vwap_old` 调用 Rust 构造期货链式标签。"""
        T, N = panel.shape
        exec_p = (
            panel.get_any_field("exec_vwap")
            if panel.has_any_field("exec_vwap")
            else panel.get_field("close")
        )
        exec_old = (
            panel.get_any_field("exec_vwap_old") if panel.has_any_field("exec_vwap_old") else exec_p
        )
        step = np.full((T, N), np.nan, dtype=np.float64, order="F")
        if T >= 2:
            valid = (
                (exec_p[:-1] > 0.0)
                & (exec_old[1:] > 0.0)
                & np.isfinite(exec_p[:-1])
                & np.isfinite(exec_old[1:])
            )
            step[:-1] = np.where(valid, exec_old[1:] / exec_p[:-1] - 1.0, np.nan)
        raw = factorlib_core.compound_forward_labels(step, [int(h) for h in horizons])
        return {k: np.asfortranarray(v, dtype=np.float64) for k, v in raw.items()}

    def style_factors(self, panel: Panel) -> dict[str, np.ndarray]:
        return compute_futures_styles(panel)

    def default_postprocess(self) -> PostprocessSpec:
        return list(RECOMMENDED_POSTPROCESS_PRESETS["standard"])

    def execution_rules(self) -> ExecutionRules:
        return ExecutionRules(
            t_plus_one=False,
            allow_short=True,
            lot_size=1,
            price_limit_aware=True,
            entry_window="vwap_open_30m",
            price_basis="raw",
        )

    def backtester(self) -> Backtester:
        fees_path = self.paths().config_dir / "fees.yaml"
        return FuturesBacktester(fees_config_path=fees_path)

    def cost_model(self) -> CostModelSpec:
        fees_path = self.paths().config_dir / "fees.yaml"
        raw: dict[str, Any] = {}
        if fees_path.exists():
            with fees_path.open("r", encoding="utf-8") as fh:
                raw = yaml.safe_load(fh) or {}
        default = raw.get("default", {})
        return CostModelSpec(
            market=self.name,
            fee_type=str(default.get("fee_type", "rate")),
            buy_fee_rate=float(default.get("open", 0.00005)),
            sell_fee_rate=float(default.get("close", 0.00005)),
            slip_ticks=float(default.get("slip_ticks", 1.0)),
            per_symbol=dict(raw.get("products", {})),
        )

    def build_weights(
        self,
        scores: np.ndarray,
        returns: np.ndarray,
        mask: np.ndarray,
        config: Any | None = None,
        panel: Panel | None = None,
    ) -> np.ndarray:
        del panel
        cfg = config if isinstance(config, PortfolioConfig) else PortfolioConfig()
        return build_futures_portfolio_weights(scores=scores, returns=returns, mask=mask, cfg=cfg)
