"""多市场统一抽象协议 `MarketProfile` 与执行/路径/成本规格定义 (markets/profile.py)。"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

import numpy as np

if TYPE_CHECKING:
    from factorlib.core.backtest import BacktestResult
    from factorlib.core.extensions import ExtensionPackMeta
    from factorlib.core.panel import Panel
    from factorlib.core.probe import PostprocessSpec


@dataclass(frozen=True)
class ExecutionRules:
    """市场交易与撮合机制规则。"""

    t_plus_one: bool
    allow_short: bool
    lot_size: int
    price_limit_aware: bool
    entry_window: str  # "vwap_open_30m" | "open"
    price_basis: str  # "raw" | "hfq"


@dataclass(frozen=True)
class ResearchPaths:
    """单市场研究数据、因子库与实验产物目录规范。"""

    market: str
    version: str
    raw_dir: Path
    derived_dir: Path
    panel_dir: Path
    db_path: Path
    runs_dir: Path
    config_dir: Path
    extensions_dir: Path | None = None


@dataclass(frozen=True)
class PanelSpec:
    """面板加载请求规格。"""

    version: str = "v1"
    fields_subset: Sequence[str] | None = None
    load_labels: bool = True
    panel_dir: Path | None = None
    raw_dir: Path | None = None
    start_date: date | str | None = None
    end_date: date | str | None = None
    data_config_path: Path | None = None
    extensions: Sequence[str] = ()
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class CostModelSpec:
    """市场统一费率与滑点摘要规格。"""

    market: str
    fee_type: str  # "rate" | "fixed" | "asymmetric_rate"
    buy_fee_rate: float = 0.0
    sell_fee_rate: float = 0.0
    min_commission: float = 0.0
    stamp_duty_rate: float = 0.0
    transfer_fee_rate: float = 0.0
    slip_ticks: float = 1.0
    slip_bps: float = 0.0
    per_symbol: dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class Backtester(Protocol):
    """市场回测执行器协议。"""

    def run(
        self,
        panel: Panel,
        weights: np.ndarray,
        config: Any | None = None,
        **kwargs: Any,
    ) -> BacktestResult: ...


@runtime_checkable
class MinutePackBuilder(Protocol):
    """各市场 1m 高频日频扩展特征包离线抽取协议。"""

    market: str

    def build_pack(
        self,
        output_dir: Path,
        pack_id: str,
        version: str = "v1",
        **kwargs: Any,
    ) -> ExtensionPackMeta: ...


@runtime_checkable
class MarketProfile(Protocol):
    """市场领域隔离协议：统一封装路径、面板加载、扩展挂载、标签、风格、后处理、组合构建与回测。"""

    name: str

    def paths(self, version: str = "v1") -> ResearchPaths: ...

    def config_path(self, kind: str) -> Path: ...

    def load_panel(self, spec: PanelSpec | None = None) -> Panel: ...

    def list_extensions(self) -> list[ExtensionPackMeta]: ...

    def attach_extensions(
        self,
        panel: Panel,
        pack_ids: Sequence[str],
        *,
        overwrite: bool = False,
    ) -> Panel: ...

    def minute_pack_builder(self) -> MinutePackBuilder: ...

    def build_labels(self, panel: Panel, horizons: Sequence[int]) -> dict[str, np.ndarray]: ...

    def style_factors(self, panel: Panel) -> dict[str, np.ndarray]: ...

    def default_postprocess(self) -> PostprocessSpec: ...

    def execution_rules(self) -> ExecutionRules: ...

    def backtester(self) -> Backtester: ...

    def cost_model(self) -> CostModelSpec: ...

    def build_weights(
        self,
        scores: np.ndarray,
        returns: np.ndarray,
        mask: np.ndarray,
        config: Any | None = None,
        panel: Panel | None = None,
    ) -> np.ndarray: ...
