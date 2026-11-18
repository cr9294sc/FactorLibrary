"""A 股市场 Profile 实现 (`StocksProfile`, markets/stocks/profile.py)。"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import factorlib_core
import numpy as np

from factorlib.core.config import (
    StockFeesConfig,
    StockPortfolioConfig,
    load_stock_data_config,
    load_stock_fees_config,
)
from factorlib.core.extensions import ExtensionPackMeta, discover_extension_packs
from factorlib.core.panel import Panel
from factorlib.core.probe import PostprocessSpec
from factorlib.markets.profile import (
    Backtester,
    CostModelSpec,
    ExecutionRules,
    MinutePackBuilder,
    PanelSpec,
    ResearchPaths,
)
from factorlib.markets.stocks.backtest import StocksBacktester
from factorlib.markets.stocks.extensions import StocksMinutePackBuilder
from factorlib.markets.stocks.panel_loader import STOCK_AUX_BACKTEST_FIELDS, StockPanelLoader
from factorlib.markets.stocks.portfolio import build_stocks_portfolio_weights
from factorlib.markets.stocks.styles import compute_stocks_styles


class StocksProfile:
    """A 股日频研究域 `MarketProfile` 实现。"""

    name: str = "stocks"

    def __init__(self, root: Path | None = None) -> None:
        self.root = Path(root) if root is not None else Path(".")

    def paths(self, version: str = "v1") -> ResearchPaths:
        derived_dir = self.root / "data" / "derived" / "stocks"
        return ResearchPaths(
            market="stocks",
            version=version,
            raw_dir=self.root / "data" / "raw" / "stocks",
            derived_dir=derived_dir,
            panel_dir=derived_dir / version,
            db_path=self.root / "data" / "registry" / "factors.sqlite",
            runs_dir=self.root / "runs",
            config_dir=self.root / "configs" / "stocks",
            extensions_dir=self.root / "data" / "extensions" / "stocks",
        )

    def config_path(self, kind: str) -> Path:
        mapping = {
            "data": "data.yaml",
            "universe": "universe.yaml",
            "fees": "fees.yaml",
            "backtest": "backtest.yaml",
            "gp": "gp_default.yaml",
            "ml": "ml_default.yaml",
        }
        fname = mapping.get(kind.strip().lower())
        if fname is None:
            raise ValueError(f"未知的配置类型 {kind!r}，可选: {sorted(mapping.keys())}")
        return self.paths().config_dir / fname

    def load_panel(self, spec: PanelSpec | None = None) -> Panel:
        s = spec or PanelSpec()
        p_info = self.paths(s.version)
        panel_dir = s.panel_dir or p_info.panel_dir
        if (panel_dir / "panel_daily.parquet").exists():
            panel = Panel.load(
                panel_dir,
                fields_subset=s.fields_subset,
                load_labels=s.load_labels,
                aux_field_names=STOCK_AUX_BACKTEST_FIELDS,
            )
            panel.meta.setdefault("market", self.name)
            if s.extensions:
                self.attach_extensions(panel, s.extensions)
            return panel

        # 回退从原始 stocks Parquet 目录加载并构造完整 labels + aux_fields
        data_cfg_path = s.data_config_path or (p_info.config_dir / "data.yaml")
        if data_cfg_path.exists():
            dcfg = load_stock_data_config(data_cfg_path, root=self.root)
            raw_dir = s.raw_dir or dcfg.raw_dir
            start_d = s.start_date or dcfg.start_date
            end_d = s.end_date or dcfg.end_date
            bench_id = dcfg.benchmark_id
            exec_mode = dcfg.exec_price_mode
            horizons = dcfg.label_horizons
        else:
            raw_dir = s.raw_dir or p_info.raw_dir
            start_d = s.start_date or "2015-11-01"
            end_d = s.end_date or "2024-12-31"
            bench_id = "000985.XSHG"
            exec_mode = "open_adj"
            horizons = (1, 2, 5, 10)

        loader = StockPanelLoader(raw_dir)
        sdata = loader.load(
            start_date=start_d,
            end_date=end_d,
            benchmark_id=bench_id,
            exec_price_mode=exec_mode,
        )
        panel = sdata.to_panel(horizons=horizons)
        if s.extensions:
            self.attach_extensions(panel, s.extensions)
        return panel

    def list_extensions(self) -> list[ExtensionPackMeta]:
        ext_dir = self.paths().extensions_dir
        if ext_dir is None:
            return []
        return discover_extension_packs(ext_dir, market=self.name)

    def attach_extensions(
        self,
        panel: Panel,
        pack_ids: Sequence[str],
        *,
        overwrite: bool = False,
    ) -> Panel:
        ext_dir = self.paths().extensions_dir
        for pid in pack_ids:
            panel.attach_extension(pid, extensions_dir=ext_dir, overwrite=overwrite)
        return panel

    def minute_pack_builder(self) -> MinutePackBuilder:
        return StocksMinutePackBuilder(root=self.root)

    def build_labels(self, panel: Panel, horizons: Sequence[int]) -> dict[str, np.ndarray]:
        """调用 Rust `build_forward_labels` 统一生成 A 股 T+1 预测收益标签。"""
        T, N = panel.shape
        price_mat = next(
            (
                panel.fields[k]
                for k in ("open_adj", "vwap_adj", "close_hfq", "close")
                if k in panel.fields
            ),
            None,
        )
        if price_mat is None:
            price_mat = panel.get_any_field("open_raw")

        is_susp = (
            panel.get_any_field("is_suspended") > 0.5
            if panel.has_any_field("is_suspended")
            else ~panel.mask
        )
        vol_ok = (
            panel.fields["volume"] > 0.0
            if "volume" in panel.fields
            else np.ones((T, N), dtype=bool)
        )
        price_ok = np.isfinite(price_mat) & (price_mat > 0.0)
        entry_ok = np.asfortranarray((~is_susp) & vol_ok & price_ok, dtype=bool)
        exit_ok = np.asfortranarray(price_ok, dtype=bool)
        f_price = np.asfortranarray(price_mat, dtype=np.float64)

        h_list = [int(h) for h in horizons]
        raw = factorlib_core.build_forward_labels(
            f_price, f_price, entry_ok, exit_ok, h_list, strict=True
        )
        out = {f"y_{h}": np.asfortranarray(raw[h], dtype=np.float64) for h in h_list}
        if 1 in raw:
            out["r_exec_next"] = np.asfortranarray(raw[1], dtype=np.float64)
        return out

    def style_factors(self, panel: Panel) -> dict[str, np.ndarray]:
        return compute_stocks_styles(panel)

    def default_postprocess(self) -> PostprocessSpec:
        return [
            "baseUniMask",
            "trdMask",
            "dropNewStock:120",
            "MAD:3",
            "zScore",
            "indNeut:sector",
            "decay:5",
            "rank",
        ]

    def execution_rules(self) -> ExecutionRules:
        return ExecutionRules(
            t_plus_one=True,
            allow_short=False,
            lot_size=100,
            price_limit_aware=True,
            entry_window="open",
            price_basis="hfq",
        )

    def backtester(self) -> Backtester:
        cfg_dir = self.paths().config_dir
        return StocksBacktester(
            fees_config_path=cfg_dir / "fees.yaml",
            backtest_config_path=cfg_dir / "backtest.yaml",
        )

    def cost_model(self) -> CostModelSpec:
        fees_path = self.paths().config_dir / "fees.yaml"
        fcfg = load_stock_fees_config(fees_path) if fees_path.exists() else StockFeesConfig()
        buy_rate = fcfg.commission_rate + fcfg.transfer_fee_rate
        sell_rate = buy_rate + fcfg.stamp_duty_rate
        return CostModelSpec(
            market=self.name,
            fee_type="asymmetric_rate",
            buy_fee_rate=buy_rate,
            sell_fee_rate=sell_rate,
            min_commission=fcfg.min_commission,
            stamp_duty_rate=fcfg.stamp_duty_rate,
            transfer_fee_rate=fcfg.transfer_fee_rate,
            slip_ticks=fcfg.slip_ticks,
            slip_bps=fcfg.slip_bps,
        )

    def build_weights(
        self,
        scores: np.ndarray,
        returns: np.ndarray,
        mask: np.ndarray,
        config: Any | None = None,
        panel: Panel | None = None,
    ) -> np.ndarray:
        cfg = config if isinstance(config, StockPortfolioConfig) else StockPortfolioConfig()
        sector = (
            panel.fields.get("sector", panel.fields.get("industry_code"))
            if panel is not None
            else None
        )
        return build_stocks_portfolio_weights(
            scores=scores,
            returns=returns,
            mask=mask,
            cfg=cfg,
            sector=sector,
        )
