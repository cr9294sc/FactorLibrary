"""A 股市场 Profile 实现 (`StocksProfile`, markets/stocks/profile.py)。"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import factorlib_core
import numpy as np

from factorlib.core.config import (
    ConfigError,
    StockDataConfig,
    StockFeesConfig,
    StockPortfolioConfig,
    load_stock_data_config,
    load_stock_datasets_config,
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
from factorlib.markets.stocks.ares_bridge import load_lb_barra_panel
from factorlib.markets.stocks.backtest import StocksBacktester
from factorlib.markets.stocks.barra import BarraRiskModelStore
from factorlib.markets.stocks.bridge import BridgeAuditReport, SignalBridge
from factorlib.markets.stocks.extensions import StocksMinutePackBuilder
from factorlib.markets.stocks.panel_loader import STOCK_AUX_BACKTEST_FIELDS, StockPanelLoader
from factorlib.markets.stocks.portfolio import build_stocks_portfolio_weights
from factorlib.markets.stocks.styles import compute_stocks_styles


def _symbol_key(value: Any) -> str:
    """跨源证券代码归一：`000001.XSHE` / `SZ000001` / `000001` 统一为 `000001`。"""
    s = str(value).strip().upper()
    if "." in s:
        s = s.split(".", 1)[0]
    if s.startswith(("SH", "SZ", "BJ")):
        s = s[2:]
    return s.zfill(6) if s.isdigit() and len(s) < 6 else s


def _normalize_date(value: Any) -> str:
    """把 `date` / `datetime` / `YYYYMMDD` / `YYYY-MM-DD` 统一归一化为 ISO 日期字符串。"""
    from datetime import date as _date
    from datetime import datetime as _datetime

    if isinstance(value, _datetime):
        return value.date().isoformat()
    if isinstance(value, _date):
        return value.isoformat()
    s = str(value).strip()
    if len(s) == 8 and s.isdigit():
        return f"{s[:4]}-{s[4:6]}-{s[6:8]}"
    return s[:10]


# 设计文档 §4.4 的嵌套优化器 YAML 段 -> `BarraOptimizerConfig` 平铺字段映射
_OBJECTIVE_FIELDS: tuple[str, ...] = (
    "alpha_scale",
    "risk_aversion",
    "turnover_penalty",
    "include_specific_risk",
)
_CONSTRAINT_FIELDS: tuple[str, ...] = (
    "cash_buffer_max",
    "max_single_weight",
    "max_active_weight",
    "max_daily_turnover",
    "style_deviation_default",
    "style_deviation_overrides",
    "industry_deviation",
    "respect_t_close_suspend",
)
_SOLVER_FIELD_MAP: dict[str, str] = {
    "priority": "solver_priority",
    "mosek_license_file": "mosek_license_file",
    "mosek_num_threads": "mosek_num_threads",
    "slack_penalty": "slack_penalty",
    "allow_slack_fallback": "allow_slack_fallback",
    "max_iterations": "max_iterations",
}
_SHAPER_FIELD_MAP: dict[str, str] = {
    "center_rank": "center_rank",
    "ind_neutral_weight": "ind_neutral_weight",
    "vol_weight": "vol_weight",
    "power": "power",
    "memory": "memory",
}
_GUARD_FIELD_MAP: dict[str, str] = {
    "adaptive_stress_guard": "enabled",
    "stress_lookback": "lookback_days",
    "stress_z_threshold": "stress_z_threshold",
    "stress_cumret_threshold": "cumret_threshold",
    "vol_multiplier": "vol_multiplier",
    "power_scale": "power_scale",
    "size_tilt": "size_tilt",
    "stress_z_window": "zscore_window",
}


def normalize_optimizer_config(
    raw: dict[str, Any] | None,
    backtest_cfg: Any | None = None,
) -> dict[str, Any]:
    """把设计文档 §4.4 的嵌套 YAML（objective / constraints / solver）拍平为优化器字段。

    同时把 `backtest.yaml` 顶层 `strategy` 段的信号塑形与风控守卫参数注入
    `shaper` / `risk_guard` 子配置，实现「同一份 YAML 同时驱动原生优化器与策略导出」。
    """
    flat: dict[str, Any] = dict(raw or {})
    objective = flat.pop("objective", None) or {}
    constraints = flat.pop("constraints", None) or {}
    solver = flat.pop("solver", None) or {}
    if isinstance(objective, dict):
        for key in _OBJECTIVE_FIELDS:
            if key in objective:
                flat[key] = objective[key]
    if isinstance(constraints, dict):
        for key in _CONSTRAINT_FIELDS:
            if key in constraints:
                flat[key] = constraints[key]
    if isinstance(solver, dict):
        for src, dst in _SOLVER_FIELD_MAP.items():
            if src in solver:
                flat[dst] = solver[src]

    if isinstance(flat.get("proxy_rank_range"), list):
        flat["proxy_rank_range"] = tuple(int(v) for v in flat["proxy_rank_range"])

    strategy = getattr(backtest_cfg, "strategy", None) or {}
    if strategy:
        shaper = dict(flat.get("shaper") or {})
        guard = dict(flat.get("risk_guard") or {})
        for src, dst in _SHAPER_FIELD_MAP.items():
            if src in strategy:
                shaper.setdefault(dst, strategy[src])
        for src, dst in _GUARD_FIELD_MAP.items():
            if src in strategy:
                guard.setdefault(dst, strategy[src])
        for key in ("buy_hurdle", "hurdle_penalty", "persistence", "discount", "mode"):
            if key not in strategy:
                continue
            if key == "mode":
                flat.setdefault("horizon_mode", str(strategy[key]) == "horizon")
            else:
                flat.setdefault(key, strategy[key])
        asym = strategy.get("asymmetric_trade_penalty")
        if isinstance(asym, dict):
            flat.setdefault(
                "asymmetric_trade_penalty",
                (float(asym.get("buy", 1.0)), float(asym.get("sell", 0.45))),
            )
        if shaper:
            flat["shaper"] = shaper
        if guard:
            flat["risk_guard"] = guard
    return flat


def _default_rq_capabilities(version: str = "v1") -> dict[str, Any]:
    ds_id = version if str(version).startswith("rq_") else f"rq_{version}"
    return {
        "dataset_id": ds_id,
        "source": "ricequant",
        "symbol_format": "rq",
        "return_clock": "open_adj",
        "has_ohlcv": True,
        "has_volume_amount": True,
        "has_directional_limit": True,
        "has_listed_days": True,
        "has_barra_exposures": False,
        "has_barra_risk_store": False,
        "has_vwap_open_30m": False,
    }


class StocksProfile:
    """A 股日频研究域 `MarketProfile` 实现。"""

    name: str = "stocks"

    def __init__(self, root: Path | None = None) -> None:
        self.root = Path(root) if root is not None else Path(".")

    def _resolve_panel_version(self, version: str) -> str:
        """将逻辑数据集名（如 ``rq_v1`` / ``lb_barra_v1``）映射为物理目录版本名。"""
        cfg_path = self.root / "configs" / "stocks" / "data.yaml"
        if cfg_path.exists():
            try:
                _, ds_map = load_stock_datasets_config(cfg_path, root=self.root)
                if version in ds_map:
                    return ds_map[version].panel_version
            except ConfigError:
                pass
        if version == "rq_v1":
            return "v1"
        return version

    def list_datasets(self, data_config_path: Path | None = None) -> dict[str, StockDataConfig]:
        """列出 `configs/stocks/data.yaml` 中声明的全部独立股票数据集。"""
        cfg_path = data_config_path or (self.root / "configs" / "stocks" / "data.yaml")
        if not cfg_path.exists():
            return {}
        _, ds_map = load_stock_datasets_config(cfg_path, root=self.root)
        return ds_map

    def dataset_config(
        self,
        dataset: str | None = None,
        data_config_path: Path | None = None,
    ) -> StockDataConfig:
        """读取指定数据集（如 ``rq_v1`` / ``lb_barra_v1``）的配置对象。"""
        cfg_path = data_config_path or (self.root / "configs" / "stocks" / "data.yaml")
        return load_stock_data_config(cfg_path, root=self.root, dataset=dataset)

    def load_risk_model(
        self,
        version: str = "lb_barra_v1",
        *,
        risk_store_dir: Path | None = None,
    ) -> BarraRiskModelStore:
        """加载指定数据集对应的旁路 ``BarraRiskModelStore``。"""
        if risk_store_dir is None:
            barra_cfg = self.barra_config()
            store_dir = (
                Path(barra_cfg.risk_store_dir)
                if barra_cfg.risk_store_dir is not None
                else (self.paths(version).panel_dir / "risk_store")
            )
        else:
            store_dir = Path(risk_store_dir)
        return BarraRiskModelStore.load(store_dir)

    def bridge_signal(
        self,
        scores_src: np.ndarray,
        source_panel: Panel,
        target_panel: Panel,
        *,
        input_signal_alignment: str = "T_close",
        min_symbol_match_ratio: float = 0.90,
        start_date: Any | None = None,
        end_date: Any | None = None,
    ) -> tuple[np.ndarray, BridgeAuditReport]:
        """运行期轻量单向信号桥接（如 ``rq_v1 -> lb_barra_v1``）。"""
        return SignalBridge.from_panel(
            source_panel=source_panel,
            scores_src=scores_src,
            target_panel=target_panel,
            input_signal_alignment=input_signal_alignment,
            min_symbol_match_ratio=min_symbol_match_ratio,
            start_date=start_date,
            end_date=end_date,
        )

    def paths(self, version: str = "v1") -> ResearchPaths:
        phys_version = self._resolve_panel_version(version)
        derived_dir = self.root / "data" / "derived" / "stocks"
        return ResearchPaths(
            market="stocks",
            version=phys_version,
            raw_dir=self.root / "data" / "raw" / "stocks",
            derived_dir=derived_dir,
            panel_dir=derived_dir / phys_version,
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
            panel.meta.setdefault("panel_version", p_info.version)
            panel.meta.setdefault("capabilities", _default_rq_capabilities(p_info.version))
            if s.extensions:
                self.attach_extensions(panel, s.extensions)
            return panel

        # 回退从原始目录加载并构造完整 labels + aux_fields
        data_cfg_path = s.data_config_path or (p_info.config_dir / "data.yaml")
        if data_cfg_path.exists():
            try:
                dcfg = load_stock_data_config(data_cfg_path, root=self.root, dataset=s.version)
            except ConfigError:
                dcfg = load_stock_data_config(data_cfg_path, root=self.root)
            raw_dir = s.raw_dir or dcfg.raw_dir
            start_d = s.start_date or dcfg.start_date
            end_d = s.end_date or dcfg.end_date
            bench_id = dcfg.benchmark_id
            exec_mode = dcfg.exec_price_mode
            horizons = dcfg.label_horizons
            source = dcfg.source
        else:
            dcfg = None
            raw_dir = s.raw_dir or p_info.raw_dir
            start_d = s.start_date or "2015-11-01"
            end_d = s.end_date or "2024-12-31"
            bench_id = "000985.XSHG"
            exec_mode = "open_adj"
            horizons = (1, 2, 5, 10)
            source = "local_backtest" if "lb_barra" in s.version else "ricequant"

        if source == "local_backtest":
            from dataclasses import replace

            lb_cfg = dcfg or StockDataConfig(
                raw_dir=raw_dir,
                derived_dir=p_info.derived_dir,
                panel_version=p_info.version,
                source="local_backtest",
                benchmark_id="000905.SH",
                entry_window="vwap_open_30m",
                exec_price_mode="vwap_open_30m_adj",
                include_idiosyncratic_labels=True,
                dataset_id=s.version,
            )
            if s.raw_dir is not None:
                lb_cfg = replace(lb_cfg, raw_dir=Path(s.raw_dir))
            if s.start_date is not None:
                from datetime import date as _date

                sd = (
                    s.start_date
                    if isinstance(s.start_date, _date)
                    else _date.fromisoformat(str(s.start_date))
                )
                lb_cfg = replace(lb_cfg, start_date=sd)
            if s.end_date is not None:
                from datetime import date as _date

                ed = (
                    s.end_date
                    if isinstance(s.end_date, _date)
                    else _date.fromisoformat(str(s.end_date))
                )
                lb_cfg = replace(lb_cfg, end_date=ed)
            panel = load_lb_barra_panel(lb_cfg)
            if s.extensions:
                self.attach_extensions(panel, s.extensions)
            return panel

        loader = StockPanelLoader(raw_dir)
        sdata = loader.load(
            start_date=start_d,
            end_date=end_d,
            benchmark_id=bench_id,
            exec_price_mode=exec_mode,
        )
        panel = sdata.to_panel(horizons=horizons)
        panel.meta.setdefault("capabilities", _default_rq_capabilities(p_info.version))
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
        caps = panel.meta.get("capabilities") if isinstance(panel.meta, dict) else None
        prefer_vwap30 = isinstance(caps, dict) and caps.get("return_clock") == "vwap_open_30m"
        search_order = (
            ("vwap_open_30m_adj", "open_adj", "vwap_adj", "close_hfq", "close_adj", "close")
            if prefer_vwap30
            else ("open_adj", "vwap_open_30m_adj", "vwap_adj", "close_hfq", "close_adj", "close")
        )
        price_mat = next(
            (panel.fields[k] for k in search_order if k in panel.fields),
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

    def default_postprocess(self, version: str | None = None) -> PostprocessSpec:
        if version is not None and "lb_barra" in version:
            return [
                "baseUniMask",
                "trdMask",
                "MAD:3",
                "zScore",
                "indNeut:sector",
                "decay:5",
                "rank",
            ]
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

    def execution_rules(self, version: str | None = None) -> ExecutionRules:
        entry_win = "vwap_open_30m" if (version is not None and "lb_barra" in version) else "open"
        return ExecutionRules(
            t_plus_one=True,
            allow_short=False,
            lot_size=100,
            price_limit_aware=True,
            entry_window=entry_win,
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
        *,
        factor_returns: np.ndarray | None = None,
        last_hold: np.ndarray | None = None,
        reserve_cash: float | None = None,
        force_numpy_optimizer: bool = False,
    ) -> np.ndarray:
        """按 `portfolio.mode` 分发组合构建：分位等权基线或模块化融合 Barra 凸优化。"""
        cfg = config if isinstance(config, StockPortfolioConfig) else StockPortfolioConfig()
        del returns  # 两种模式均不依赖全截面历史收益矩阵（避免 N=5000 病态协方差）
        if cfg.mode == "barra_qp":
            return self._build_barra_qp_weights(
                scores=scores,
                mask=mask,
                cfg=cfg,
                panel=panel,
                factor_returns=factor_returns,
                last_hold=last_hold,
                reserve_cash=reserve_cash,
                force_numpy=force_numpy_optimizer,
            )
        sector = (
            panel.fields.get("sector", panel.fields.get("industry_code"))
            if panel is not None
            else None
        )
        return build_stocks_portfolio_weights(
            scores=scores,
            returns=np.zeros_like(scores),
            mask=mask,
            cfg=cfg,
            sector=sector,
        )

    def build_barra_optimizer(
        self,
        cfg: StockPortfolioConfig,
        *,
        panel: Panel | None = None,
        force_numpy: bool = False,
    ) -> Any:
        """构造 `BarraPortfolioOptimizer`（含风险库、基准权重与求解器配置）。"""
        from factorlib.markets.stocks.optimizer import (
            BarraOptimizerConfig,
            BarraPortfolioOptimizer,
        )

        barra_cfg = self.barra_config()
        risk_store = None
        try:
            risk_dir = barra_cfg.risk_store_dir or (
                self.paths("lb_barra_v1").panel_dir / "risk_store"
            )
            if Path(risk_dir).exists() and any(Path(risk_dir).glob("store.json")):
                risk_store = self.load_risk_model("lb_barra_v1", risk_store_dir=Path(risk_dir))
        except Exception:
            risk_store = None

        opt_cfg = BarraOptimizerConfig.from_dict(
            normalize_optimizer_config(dict(cfg.optimizer), self.backtest_config())
        )
        benchmark_weights = None
        if risk_store is not None and opt_cfg.benchmark_weight_mode == "constituent":
            benchmark_weights = self._load_benchmark_weights(opt_cfg, panel)
        return BarraPortfolioOptimizer(
            opt_cfg,
            risk_store=risk_store,
            benchmark_weights=benchmark_weights,
            force_numpy=force_numpy,
        )

    def backtest_config(self) -> Any:
        """加载 `configs/stocks/backtest.yaml`（缺失时回退默认契约）。"""
        from factorlib.core.config import load_stock_backtest_config

        path = self.paths().config_dir / "backtest.yaml"
        if path.exists():
            try:
                return load_stock_backtest_config(path)
            except ConfigError:
                return None
        return None

    @staticmethod
    def _load_benchmark_weights(opt_cfg: Any, panel: Panel | None) -> Any:
        """加载逐日成分权重（`constituent` 模式）；缺失时返回 `None` 触发代理降级。"""
        path_str = getattr(opt_cfg, "benchmark_weights_path", None)
        if not path_str or panel is None:
            return None
        p = Path(str(path_str))
        if not p.exists():
            return None
        import polars as pl

        df = pl.read_parquet(p)
        needed = {"trading_date", "symbol", "weight"}
        if not needed.issubset(set(df.columns)):
            return None
        T, N = panel.shape
        dates = [_normalize_date(d) for d in panel.dates]
        d_map = {d: i for i, d in enumerate(dates)}
        # 跨源代码归一：米筐 `000001.XSHE` 与 lb_barra 6 位代码 `000001` 统一为 6 位逻辑键
        s_map = {_symbol_key(s): j for j, s in enumerate(panel.symbols)}
        out = np.zeros((T, N), dtype=np.float64, order="F")
        hit = 0
        for d_val, s_val, w_val in df.select(["trading_date", "symbol", "weight"]).iter_rows():
            ti = d_map.get(_normalize_date(d_val))
            nj = s_map.get(_symbol_key(s_val))
            if ti is None or nj is None:
                continue
            out[ti, nj] = float(w_val)
            hit += 1
        # 零命中或全零权重视为不可用，交由 `mkt_cap_proxy` 降级
        if hit == 0 or not np.any(out > 0.0):
            return None
        return out

    def _build_barra_qp_weights(
        self,
        *,
        scores: np.ndarray,
        mask: np.ndarray,
        cfg: StockPortfolioConfig,
        panel: Panel | None,
        factor_returns: np.ndarray | None,
        last_hold: np.ndarray | None,
        reserve_cash: float | None,
        force_numpy: bool,
    ) -> np.ndarray:
        """模块化融合凸优化（信号塑形 → 动态风控守卫 → Barra 低秩 QP + 漂移闭环）。"""
        if panel is None:
            raise ValueError(
                "portfolio.mode='barra_qp' 必须提供 panel（需要 Barra 暴露与市值字段）"
            )
        optimizer = self.build_barra_optimizer(cfg, panel=panel, force_numpy=force_numpy)
        effective_cash = float(reserve_cash) if reserve_cash is not None else 1.0
        if reserve_cash is None:
            bt_cfg = self.backtest_config()
            if bt_cfg is not None:
                effective_cash = float(getattr(bt_cfg.execution, "reserve_cash", 1.0))
        fr = factor_returns
        if fr is None:
            store = getattr(optimizer, "risk_store", None)
            if store is not None:
                fr = store.align_factor_returns(panel.dates)
        weights = optimizer.build_weights(
            panel,
            scores,
            factor_returns=fr,
            dates=list(panel.dates),
            last_hold=last_hold,
            reserve_cash=effective_cash,
        )
        return np.asfortranarray(np.asarray(weights, dtype=np.float64), dtype=np.float64)

    def barra_config(self) -> Any:
        """加载 `configs/stocks/barra.yaml`（缺失时回退默认 Barra 契约）。"""
        from factorlib.core.config import StockBarraConfig, load_stock_barra_config

        path = self.paths().config_dir / "barra.yaml"
        if path.exists():
            try:
                return load_stock_barra_config(path, root=self.root)
            except ConfigError:
                return StockBarraConfig()
        return StockBarraConfig()
