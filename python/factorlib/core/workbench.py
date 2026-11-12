"""FactorLibrary 统一研发工作台服务层 (core/workbench.py)。

连接前端交互界面（现代单页 Web 工作台 SPA、REST API 与 Streamlit）与后端核心引擎：
- 智能面板加载：优先挂载本地 `data/derived/futures/v1`，缺失或切换演示模式时自动生成具备真实产业链经济语义的合成期货面板；
- Stage 0 & 1：统一因子目录检索 (`FactorCatalog`: 字段、68+ 算子、12 个后处理算子、6 大期货风格因子)；
- Stage 2 & 3：双模态因子定义 (`expression` / `code`)、12 算子后处理链编排与毫秒级静态探针 (`probe_factor`)；
- Stage 4 & 5：因子矩阵物化 (`.npz`)、不可变版本快照锁定 (`factor_version`)、Layer 1 (Schema v4 统计分布与风格归因) + Layer 2 (期货实盘逐日盯市回测)；
- 因子库资产全生命周期管理 (`candidate -> accepted -> active -> deprecated`) 与端到端流水线 (`GP` / `ML` / `Backtest`) 编排。
"""

from __future__ import annotations

import dataclasses
import datetime as dt
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from factorlib.core.backtest import BacktestResult
from factorlib.core.catalog import (
    POSTPROCESS_SPECS,
    RECOMMENDED_POSTPROCESS_PRESETS,
    FactorCatalog,
)
from factorlib.core.config import (
    BacktestConfig,
    PortfolioConfig,
    load_backtest_config,
)
from factorlib.core.evaluate import (
    get_seed_factors,
    register_seed_factors,
    run_factor_workflow,
)
from factorlib.core.panel import Panel
from factorlib.core.probe import (
    FactorDefinition,
    parse_postprocess_pipeline,
    probe_factor,
)
from factorlib.core.registry import FactorRegistry
from factorlib.core.report import generate_html_report, summary_dict
from factorlib.core.styles import STYLE_DEFINITIONS


def _sanitize_json_value(obj: Any) -> Any:
    """递归将 numpy 标量、数组、NaN/Inf 与 Path 转换为严格合规的标准 JSON 结构。"""
    if obj is None or isinstance(obj, (bool, str, int)):
        return obj
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, np.generic):
        val = obj.item()
        if isinstance(val, float) and not math.isfinite(val):
            return None
        return val
    if isinstance(obj, np.ndarray):
        return _sanitize_json_value(obj.tolist())
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, dict):
        return {str(k): _sanitize_json_value(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [_sanitize_json_value(x) for x in obj]
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return _sanitize_json_value(dataclasses.asdict(obj))
    return str(obj)


# 预设双模态因子编写模板，供前端 IDE 一键载入
AUTHORING_TEMPLATES: list[dict[str, Any]] = [
    {
        "id": "tpl_carry_mom",
        "name": "carry_momentum_combo",
        "title": "展期收益 + 截面动量复合因子 (DSL + 标准后处理)",
        "factor_type": "expression",
        "expression": "cs_zscore(ry_ann) + cs_zscore(pct_change(close, 20))",
        "code": "",
        "postprocess": RECOMMENDED_POSTPROCESS_PRESETS["standard"],
        "description": "结合主力-次主力年化展期收益率与 20 日价格动量，经板块中性化与 Z-Score 标准化。",
    },
    {
        "id": "tpl_inv_pos",
        "name": "scarcity_smart_money",
        "title": "低库存紧缺 + 主力席位净多头共振 (DSL + 多风格中性化)",
        "factor_type": "expression",
        "expression": "-wh_z60 + ts_mean(mr_net_ratio_chg5, 5)",
        "code": "",
        "postprocess": RECOMMENDED_POSTPROCESS_PRESETS["style_neutral"],
        "description": "捕捉注册仓单去库与前 20 会员净多头增仓共振，剥离规模、波动率与展期风格暴露。",
    },
    {
        "id": "tpl_micro_rev",
        "name": "intraday_skew_reversal",
        "title": "高频日内偏度与尾盘动量修正 (DSL + EMA 平滑)",
        "factor_type": "expression",
        "expression": "ts_mean(ret_last30, 10) - 0.5 * ts_mean(skew_intra, 10)",
        "code": "",
        "postprocess": [
            {"op": "baseUniMask"},
            {"op": "trdMask"},
            {"op": "MAD", "param": 3.0},
            {"op": "indNeut"},
            {"op": "zScore"},
            {"op": "emadecay", "window": 5},
        ],
        "description": "融合尾盘 30 分钟聪明钱动量与日内分钟收益偏度反转，叠加 5 日指数衰减平滑降低换手。",
    },
    {
        "id": "tpl_py_vol_adj_mom",
        "name": "py_vol_adjusted_momentum",
        "title": "Python 因果沙箱：夏普动量与持仓量确认 (compute(context))",
        "factor_type": "code",
        "expression": "",
        "code": (
            "def compute(context):\n"
            '    close = context.history("close", 20)\n'
            '    oi = context.history("open_interest", 10)\n'
            "    ret20 = close[-1] / close[0] - 1.0\n"
            "    daily_ret = close[1:] / close[:-1] - 1.0\n"
            "    vol20 = np.nanstd(daily_ret, axis=0) + 1e-6\n"
            "    oi_boost = np.clip(oi[-1] / (np.nanmean(oi, axis=0) + 1e-6), 0.5, 2.0)\n"
            "    return (ret20 / vol20) * oi_boost\n"
        ),
        "postprocess": RECOMMENDED_POSTPROCESS_PRESETS["standard"],
        "description": "使用严格因果的 context.history(field, window) 窗口切片编写自定义 Python 截面因子。",
    },
]


def create_synthetic_futures_panel(n_days: int = 120, seed: int = 42) -> Panel:
    """构建具备完整 33 个标准字段、多板块分布与真实因子信噪比的合成期货研究面板。

    用于当本地不存在 `data/derived/futures/v1` 或用户在前端开启「合成演示面板」时，
    保证 6 阶段因子研发工作流、双层回测、GP 与 ML 管线随时可交互运行。
    """
    rng = np.random.default_rng(seed)
    symbols = ["RB", "I", "CU", "AL", "TA", "MA", "M", "Y", "IF", "AU"]
    sector_map = [1.0, 1.0, 2.0, 2.0, 3.0, 3.0, 4.0, 4.0, 5.0, 6.0]
    N = len(symbols)
    T = max(40, int(n_days))

    start_dt = dt.date(2023, 1, 3)
    dates: list[str] = []
    curr_dt = start_dt
    while len(dates) < T:
        if curr_dt.weekday() < 5:
            dates.append(curr_dt.isoformat())
        curr_dt += dt.timedelta(days=1)

    # 构造具备持续性的潜在基本面特征 (展期 ry_ann, 库存 wh_z60, 席位 mr_net_ratio)
    ry_ann = np.zeros((T, N), dtype=np.float64, order="F")
    wh_z60 = np.zeros((T, N), dtype=np.float64, order="F")
    mr_net_ratio = np.zeros((T, N), dtype=np.float64, order="F")
    rv = np.zeros((T, N), dtype=np.float64, order="F")

    ry_ann[0, :] = rng.normal(0.0, 0.06, N)
    wh_z60[0, :] = rng.normal(0.0, 0.8, N)
    mr_net_ratio[0, :] = rng.normal(0.0, 0.12, N)
    rv[0, :] = rng.uniform(0.010, 0.025, N)

    for t in range(1, T):
        ry_ann[t, :] = 0.92 * ry_ann[t - 1, :] + rng.normal(0.0, 0.02, N)
        wh_z60[t, :] = 0.90 * wh_z60[t - 1, :] + rng.normal(0.0, 0.25, N)
        mr_net_ratio[t, :] = 0.85 * mr_net_ratio[t - 1, :] + rng.normal(0.0, 0.05, N)
        rv[t, :] = np.clip(0.80 * rv[t - 1, :] + rng.uniform(0.002, 0.006, N), 0.006, 0.045)

    # 构造与展期、库存、席位及动量弱相关的因果日度收益率
    base_prices = np.array(
        [3800.0, 820.0, 68000.0, 18500.0, 5800.0, 2500.0, 3900.0, 8200.0, 4000.0, 450.0]
    )
    close = np.zeros((T, N), dtype=np.float64, order="F")
    close[0, :] = base_prices

    for t in range(1, T):
        alpha_signal = (
            0.03 * ry_ann[t - 1, :] - 0.0012 * wh_z60[t - 1, :] + 0.008 * mr_net_ratio[t - 1, :]
        )
        noise = rng.normal(0.0002, 0.013, N)
        ret_t = np.clip(alpha_signal + noise, -0.07, 0.07)
        close[t, :] = close[t - 1, :] * (1.0 + ret_t)

    open_p = np.asfortranarray(close * (1.0 + rng.normal(0.0, 0.002, (T, N))))
    high_p = np.asfortranarray(
        np.maximum(open_p, close) * (1.0 + np.abs(rng.normal(0.004, 0.003, (T, N))))
    )
    low_p = np.asfortranarray(
        np.minimum(open_p, close) * (1.0 - np.abs(rng.normal(0.004, 0.003, (T, N))))
    )
    settle_p = np.asfortranarray(0.5 * (open_p + close))
    vwap_p = np.asfortranarray(0.3 * open_p + 0.3 * high_p + 0.4 * close)
    exec_vwap = np.asfortranarray(open_p * (1.0 + rng.normal(0.0, 0.0008, (T, N))))

    volume = np.asfortranarray(rng.uniform(20000.0, 180000.0, (T, N)))
    multipliers = np.array([10.0, 100.0, 5.0, 5.0, 5.0, 10.0, 10.0, 10.0, 300.0, 1000.0])
    turnover = np.asfortranarray(close * volume * multipliers[np.newaxis, :])
    open_interest = np.asfortranarray(volume * rng.uniform(1.8, 3.5, (T, N)))
    oi_chg = np.zeros((T, N), dtype=np.float64, order="F")
    oi_chg[1:, :] = open_interest[1:, :] - open_interest[:-1, :]

    wh_chg5 = np.zeros((T, N), dtype=np.float64, order="F")
    wh_chg5[5:, :] = (wh_z60[5:, :] - wh_z60[:-5, :]) * 0.1
    mr_chg5 = np.zeros((T, N), dtype=np.float64, order="F")
    mr_chg5[5:, :] = mr_net_ratio[5:, :] - mr_net_ratio[:-5, :]

    sector_mat = np.asfortranarray(np.tile(np.array(sector_map, dtype=np.float64), (T, 1)))
    listed_days = np.asfortranarray(
        np.tile(np.arange(120, 120 + T, dtype=np.float64)[:, np.newaxis], (1, N))
    )
    ones_mask = np.ones((T, N), dtype=np.float64, order="F")

    fields: dict[str, np.ndarray] = {
        "open": open_p,
        "high": high_p,
        "low": low_p,
        "close": close,
        "settle": settle_p,
        "vwap": vwap_p,
        "exec_vwap": exec_vwap,
        "exec_vwap_old": exec_vwap.copy(order="F"),
        "volume": volume,
        "turnover": turnover,
        "open_interest": open_interest,
        "oi_chg": oi_chg,
        "rv": rv,
        "rv_up": np.asfortranarray(rv * rng.uniform(0.45, 0.65, (T, N))),
        "rv_down": np.asfortranarray(rv * rng.uniform(0.35, 0.55, (T, N))),
        "ret_last30": np.asfortranarray(rng.normal(0.0, 0.004, (T, N))),
        "ret_first30": np.asfortranarray(rng.normal(0.0, 0.005, (T, N))),
        "spread_night_day": np.asfortranarray(rng.normal(0.0, 0.006, (T, N))),
        "vol_conc": np.asfortranarray(rng.uniform(0.15, 0.45, (T, N))),
        "amp": np.asfortranarray((high_p - low_p) / np.maximum(close, 1e-6)),
        "skew_intra": np.asfortranarray(rng.normal(0.0, 0.5, (T, N))),
        "kurt_intra": np.asfortranarray(rng.uniform(2.5, 6.0, (T, N))),
        "vwap_dev": np.asfortranarray((close - vwap_p) / np.maximum(vwap_p, 1e-6)),
        "ry_main_sub": np.asfortranarray(ry_ann / 4.0),
        "ry_ann": ry_ann,
        "ry_near_main": np.asfortranarray(ry_ann * 0.5),
        "wh_chg5": wh_chg5,
        "wh_z60": wh_z60,
        "mr_net_ratio": mr_net_ratio,
        "mr_cr20": np.asfortranarray(rng.uniform(0.45, 0.75, (T, N))),
        "mr_net_ratio_chg5": mr_chg5,
        "sector": sector_mat,
        "listed_days": listed_days,
        "in_universe_mask": ones_mask,
        "limit_mask": ones_mask.copy(order="F"),
    }

    labels: dict[str, np.ndarray] = {}
    for h in (1, 3, 5, 10, 20):
        yh = np.full((T, N), np.nan, dtype=np.float64, order="F")
        if T > h + 1:
            yh[: -(h + 1), :] = exec_vwap[1 + h :, :] / exec_vwap[1:-h, :] - 1.0
        labels[f"y_{h}"] = yh
    labels["r_exec_next"] = labels["y_1"].copy(order="F")

    valid_mask = np.ones((T, N), dtype=bool, order="F")
    return Panel(
        dates=np.array(dates),
        symbols=symbols,
        fields=fields,
        mask=valid_mask,
        labels=labels,
        meta={"panel_version": "synthetic_v1", "synthetic": True},
    )


class FactorWorkbenchService:
    """FactorLibrary 前后端一体化工作台核心服务。"""

    def __init__(
        self,
        panel_dir: str | Path = "data/derived/futures/v1",
        db_path: str | Path = "data/registry/factors.sqlite",
        runs_dir: str | Path = "runs",
        backtest_config_path: str | Path = "configs/futures/backtest.yaml",
        fees_config_path: str | Path = "configs/futures/fees.yaml",
        gp_config_path: str | Path = "configs/futures/gp_default.yaml",
        ml_config_path: str | Path = "configs/futures/ml_default.yaml",
        force_synthetic: bool = False,
    ) -> None:
        self.panel_dir = Path(panel_dir)
        self.db_path = Path(db_path)
        self.runs_dir = Path(runs_dir)
        self.backtest_config_path = Path(backtest_config_path)
        self.fees_config_path = Path(fees_config_path)
        self.gp_config_path = Path(gp_config_path)
        self.ml_config_path = Path(ml_config_path)

        self.runs_dir.mkdir(parents=True, exist_ok=True)
        self.registry = FactorRegistry(db_path=self.db_path)
        self.catalog = FactorCatalog(panel_dir=self.panel_dir)

        self.panel_source: str = "synthetic"
        self.panel: Panel = self._load_panel(force_synthetic=force_synthetic)

    def _load_panel(self, force_synthetic: bool = False) -> Panel:
        if not force_synthetic and (self.panel_dir / "panel_daily.parquet").exists():
            try:
                p = Panel.load(self.panel_dir)
                self.panel_source = "real"
                return p
            except Exception:
                pass
        self.panel_source = "synthetic"
        return create_synthetic_futures_panel(n_days=120, seed=42)

    def switch_panel_source(
        self, mode: str = "auto", panel_dir: str | Path | None = None
    ) -> dict[str, Any]:
        """动态切换真实研究面板与合成演示面板。"""
        if panel_dir is not None:
            self.panel_dir = Path(panel_dir)
            self.catalog = FactorCatalog(panel_dir=self.panel_dir)
        m = mode.strip().lower()
        if m == "synthetic":
            self.panel = self._load_panel(force_synthetic=True)
        else:
            self.panel = self._load_panel(force_synthetic=False)
        return self.get_context()

    # =========================================================================
    # Stage 0 & Stage 1: 环境上下文与目录检索
    # =========================================================================

    def get_context(self) -> dict[str, Any]:
        """Stage 0 获取工作台完整上下文、面板规格、因子库统计与内置模板。"""
        base_ctx = self.catalog.get_context()
        T, N = self.panel.shape
        all_factors = self.registry.list_factors()
        status_counts = {"candidate": 0, "accepted": 0, "active": 0, "deprecated": 0}
        for f in all_factors:
            status_counts[f.status] = status_counts.get(f.status, 0) + 1

        bt_cfg = (
            load_backtest_config(self.backtest_config_path)
            if self.backtest_config_path.exists()
            else BacktestConfig()
        )

        ctx = {
            **base_ctx,
            "panel_info": {
                "source": self.panel_source,
                "panel_dir": str(self.panel_dir),
                "has_real_panel": (self.panel_dir / "panel_daily.parquet").exists(),
                "n_days": int(T),
                "n_symbols": int(N),
                "start_date": str(self.panel.dates[0]) if T > 0 else "",
                "end_date": str(self.panel.dates[-1]) if T > 0 else "",
                "symbols": list(self.panel.symbols),
                "n_fields": len(self.panel.fields),
                "labels": list(self.panel.labels.keys()),
            },
            "registry_summary": {
                "total": len(all_factors),
                "by_status": status_counts,
                "db_path": str(self.db_path),
            },
            "postprocess_specs": POSTPROCESS_SPECS,
            "style_definitions": STYLE_DEFINITIONS,
            "seed_factors": get_seed_factors(),
            "authoring_templates": AUTHORING_TEMPLATES,
            "default_portfolio": {
                "init_cash": float(bt_cfg.capital.init_cash),
                "target_volatility": float(bt_cfg.portfolio.target_volatility),
                "vol_lookback": int(bt_cfg.portfolio.vol_lookback),
                "max_single_weight": float(bt_cfg.portfolio.max_single_weight),
                "max_gross_leverage": float(bt_cfg.portfolio.max_gross_leverage),
                "rebalance_buffer": float(bt_cfg.portfolio.rebalance_buffer),
            },
        }
        return _sanitize_json_value(ctx)

    def catalog_stats(self, catalog: str = "all") -> dict[str, Any]:
        return _sanitize_json_value(self.catalog.stats(catalog=catalog))

    def catalog_search(
        self,
        query: str = "",
        catalog: str = "all",
        category: str = "",
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        return _sanitize_json_value(
            self.catalog.search(query=query, catalog=catalog, category=category, limit=limit)
        )

    def catalog_info(self, name: str, catalog: str = "all") -> dict[str, Any] | None:
        return _sanitize_json_value(self.catalog.get_entry(name=name, catalog=catalog))

    # =========================================================================
    # Stage 2 & Stage 3: 因子定义、草稿保存与毫秒级静态探针
    # =========================================================================

    def _parse_definition(
        self, payload: dict[str, Any] | FactorDefinition | str
    ) -> FactorDefinition:
        if isinstance(payload, FactorDefinition):
            return payload
        if isinstance(payload, str):
            return FactorDefinition(
                name="inline_factor", factor_type="expression", expression=payload
            )
        return FactorDefinition.from_dict(payload)

    def probe(self, payload: dict[str, Any] | FactorDefinition | str) -> dict[str, Any]:
        """Stage 3 执行毫秒级静态探针校验。"""
        try:
            defn = self._parse_definition(payload)
            rep = probe_factor(defn, panel=self.panel)
            return _sanitize_json_value(rep)
        except Exception as exc:
            return {
                "schema_version": "v1",
                "status": "error",
                "errors": [str(exc)],
                "stage_order_warnings": [],
                "execution_requirements": {
                    "expression_lookback": 1,
                    "postprocess_lookback": 0,
                    "total_required_lookback": 1,
                    "warmup_bars": 0,
                },
            }

    def save_draft(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Stage 2 保存或更新因子草稿记录。"""
        defn = self._parse_definition(payload)
        rep = probe_factor(defn, panel=self.panel)
        if rep["status"] != "ok":
            raise ValueError("; ".join(rep.get("errors", ["静态探针未通过"])))

        mode = str(payload.get("mode", "cs"))
        sign = int(payload.get("sign", 1))
        fid = self.registry.save_draft(
            definition=defn,
            probe_report=rep,
            mode=mode,
            sign=sign,
            source="manual",
            data_version=str(self.panel.meta.get("panel_version", "v1")),
        )
        detail = self.get_factor_detail(fid)
        return _sanitize_json_value(
            {
                "status": "ok",
                "factor_id": fid,
                "probe_report": rep,
                "factor": detail,
            }
        )

    # =========================================================================
    # Stage 4 & Stage 5: 因子求值、物化、版本锁定与双层回测 (Layer 1 v4 + Layer 2 MTM)
    # =========================================================================

    def _serialize_backtest_result(self, bt: BacktestResult) -> dict[str, Any]:
        equity = np.asarray(bt.equity, dtype=np.float64)
        peaks = np.maximum.accumulate(equity) if len(equity) > 0 else np.array([])
        drawdown = ((equity - peaks) / np.maximum(peaks, 1e-12)).tolist() if len(equity) > 0 else []

        # 按年-月汇总收益热力数据
        monthly_map: dict[str, dict[str, float]] = {}
        for d_str, r in zip(bt.dates, bt.returns, strict=False):
            ds = str(d_str)
            if len(ds) >= 7:
                y_str = ds[:4]
                m_str = ds[5:7]
                if y_str not in monthly_map:
                    monthly_map[y_str] = {f"{m:02d}": 0.0 for m in range(1, 13)}
                if m_str in monthly_map[y_str] and np.isfinite(r):
                    monthly_map[y_str][m_str] = (1.0 + monthly_map[y_str][m_str]) * (
                        1.0 + float(r)
                    ) - 1.0

        monthly_rows = []
        for y_str in sorted(monthly_map.keys()):
            y_cum = 1.0
            for m_val in monthly_map[y_str].values():
                y_cum *= 1.0 + m_val
            monthly_rows.append(
                {
                    "year": y_str,
                    "months": monthly_map[y_str],
                    "year_total": y_cum - 1.0,
                }
            )

        # 极端交易日 Top 5
        rets = np.asarray(bt.returns, dtype=np.float64)
        order = np.argsort(np.where(np.isfinite(rets), rets, 0.0))
        k = min(5, len(rets))
        best_days = [
            {"date": str(bt.dates[i]), "return": float(rets[i]), "net_pnl": float(bt.net_pnl[i])}
            for i in order[-k:][::-1]
        ]
        worst_days = [
            {"date": str(bt.dates[i]), "return": float(rets[i]), "net_pnl": float(bt.net_pnl[i])}
            for i in order[:k]
        ]

        gross_sum = float(np.sum(bt.gross_pnl))
        fee_sum = float(np.sum(bt.fee))
        slip_sum = float(np.sum(bt.slippage))
        roll_sum = float(np.sum(bt.roll_cost))
        net_sum = float(np.sum(bt.net_pnl))
        cost_drag = ((fee_sum + slip_sum) / gross_sum) if gross_sum > 1e-6 else 0.0

        return {
            "metrics": dict(bt.metrics),
            "dates": [str(d) for d in bt.dates],
            "symbols": list(bt.symbols),
            "series": {
                "equity": equity.tolist(),
                "returns": bt.returns.tolist(),
                "drawdown": drawdown,
                "cash": bt.cash.tolist(),
                "margin_used": bt.margin_used.tolist(),
                "margin_ratio": (bt.margin_used / np.maximum(equity, 1e-6)).tolist(),
                "turnover": bt.turnover.tolist(),
                "gross_pnl": bt.gross_pnl.tolist(),
                "net_pnl": bt.net_pnl.tolist(),
            },
            "cost_breakdown": {
                "gross_pnl": gross_sum,
                "fees": fee_sum,
                "slippage": slip_sum,
                "roll_cost": roll_sum,
                "net_pnl": net_sum,
                "cost_drag": cost_drag,
            },
            "monthly_returns": monthly_rows,
            "extreme_days": {
                "best": best_days,
                "worst": worst_days,
            },
        }

    def run_workflow(self, payload: dict[str, Any]) -> dict[str, Any]:
        """执行完整 6 阶段因子研发工作流（静态探针 -> 双模态求值与后处理 -> 物化与版本锁定 -> 双层回测）。"""
        defn = self._parse_definition(payload)
        if "postprocess" in payload and payload["postprocess"] is not None:
            defn.postprocess = parse_postprocess_pipeline(payload["postprocess"])

        bt_cfg = (
            load_backtest_config(self.backtest_config_path)
            if self.backtest_config_path.exists()
            else BacktestConfig()
        )
        port_overrides = payload.get("portfolio") or {}
        if "init_cash" in port_overrides and port_overrides["init_cash"] is not None:
            bt_cfg = dataclasses.replace(
                bt_cfg,
                capital=dataclasses.replace(
                    bt_cfg.capital, init_cash=float(port_overrides["init_cash"])
                ),
            )

        port_cfg: PortfolioConfig = bt_cfg.portfolio
        for field_name in (
            "target_volatility",
            "max_single_weight",
            "max_gross_leverage",
            "rebalance_buffer",
        ):
            if field_name in port_overrides and port_overrides[field_name] is not None:
                port_cfg = dataclasses.replace(
                    port_cfg, **{field_name: float(port_overrides[field_name])}
                )
        if "vol_lookback" in port_overrides and port_overrides["vol_lookback"] is not None:
            port_cfg = dataclasses.replace(
                port_cfg, vol_lookback=int(port_overrides["vol_lookback"])
            )
        bt_cfg = dataclasses.replace(bt_cfg, portfolio=port_cfg)

        materialize = bool(payload.get("materialize", True))
        lock_version = bool(payload.get("lock_version", True))
        n_quantiles = int(payload.get("n_quantiles", 5))
        force_sign = int(payload["force_sign"]) if payload.get("force_sign") in (-1, 1) else None

        wf_res = run_factor_workflow(
            panel=self.panel,
            definition=defn,
            portfolio_config=port_cfg,
            backtest_config=bt_cfg,
            registry=self.registry,
            materialize=materialize,
            lock_version=lock_version,
            force_sign=force_sign,
            n_quantiles=n_quantiles,
        )

        # 生成独立交互式 HTML 报告落盘至 runs/
        now_tag = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_name = "".join(c if c.isalnum() or c in ("_", "-") else "_" for c in defn.name)[:32]
        run_id = f"wf_{now_tag}_{safe_name}"
        out_dir = self.runs_dir / run_id
        out_dir.mkdir(parents=True, exist_ok=True)
        html_path = out_dir / "report.html"
        generate_html_report(
            wf_res.backtest_result,
            html_path,
            title=f"FactorLibrary 双层因子诊断报告 - {defn.name}",
            v4_stats=wf_res.v4_stats,
            probe_report=wf_res.probe_report,
        )

        return _sanitize_json_value(
            {
                "status": "ok",
                "run_id": run_id,
                "report_path": str(html_path),
                "report_url": f"/api/runs/{run_id}/report",
                "factor_id": wf_res.factor_id,
                "version_locked": wf_res.version_locked,
                "materialization_path": str(wf_res.materialization_path)
                if wf_res.materialization_path
                else None,
                "definition": wf_res.definition.to_dict(),
                "probe_report": wf_res.probe_report,
                "sign": wf_res.evaluation.sign,
                "evaluation_metrics": [m.to_dict() for m in wf_res.evaluation.all_metrics],
                "v4_stats": wf_res.v4_stats,
                "backtest": self._serialize_backtest_result(wf_res.backtest_result),
            }
        )

    # =========================================================================
    # 因子库资产与不可变版本中心 (SQLite Factor Registry)
    # =========================================================================

    def list_factors(
        self,
        status: str | None = None,
        source: str | None = None,
        mode: str | None = None,
        min_icir: float | None = None,
        split: str = "train",
        horizon: int = 5,
    ) -> list[dict[str, Any]]:
        records = self.registry.list_factors(
            status=status or None,
            source=source or None,
            mode=mode or None,
            min_icir=min_icir,
            split=split,
            horizon=horizon,
        )
        out: list[dict[str, Any]] = []
        for r in records:
            d = r.to_dict()
            versions = self.registry.list_factor_versions(r.id) if r.id is not None else []
            d["version_count"] = len(versions)
            d["latest_version"] = versions[-1].version if versions else 0
            # 提取目标 split/horizon 代表性指标便于前端表格直接渲染
            primary_m = None
            for m in r.metrics:
                if m.split == split and m.horizon == horizon:
                    primary_m = m.to_dict()
                    break
            if primary_m is None and r.metrics:
                primary_m = r.metrics[0].to_dict()
            d["primary_metric"] = primary_m
            out.append(d)
        return _sanitize_json_value(out)

    def get_factor_detail(self, factor_id: int) -> dict[str, Any] | None:
        rec = self.registry.get_factor(factor_id, include_metrics=True)
        if rec is None:
            return None
        d = rec.to_dict()
        versions = self.registry.list_factor_versions(factor_id)
        parsed_versions = []
        for v in versions:
            vd = v.to_dict()
            try:
                vd["definition"] = json.loads(v.definition_json) if v.definition_json else {}
            except Exception:
                vd["definition"] = {}
            try:
                vd["probe"] = json.loads(v.probe_json) if v.probe_json else None
            except Exception:
                vd["probe"] = None
            try:
                vd["eval_summary"] = (
                    json.loads(v.eval_summary_json) if v.eval_summary_json else None
                )
            except Exception:
                vd["eval_summary"] = None
            parsed_versions.append(vd)
        d["versions"] = parsed_versions
        try:
            d["definition"] = json.loads(rec.definition_json) if rec.definition_json else None
        except Exception:
            d["definition"] = None
        return _sanitize_json_value(d)

    def update_factor_status(
        self, factor_id: int, status: str, sign: int | None = None
    ) -> dict[str, Any]:
        self.registry.update_factor_status(factor_id=factor_id, status=status, sign=sign)
        detail = self.get_factor_detail(factor_id)
        return _sanitize_json_value({"status": "ok", "factor": detail})

    def delete_factor(self, factor_id: int) -> dict[str, Any]:
        deleted = self.registry.delete_factor(factor_id)
        return {"status": "ok" if deleted else "not_found", "deleted": deleted}

    def init_seed_factors(self) -> dict[str, Any]:
        ids = register_seed_factors(
            registry=self.registry,
            engine_or_panel=self.panel,
            run_id="workbench_seed_init",
            status="accepted",
        )
        return _sanitize_json_value(
            {
                "status": "ok",
                "registered_ids": ids,
                "factors": self.list_factors(),
            }
        )

    # =========================================================================
    # 自动化因子挖掘 (GP)、机器学习合成 (ML) 与全链路流水线
    # =========================================================================

    def run_e2e_pipeline(self, payload: dict[str, Any]) -> dict[str, Any]:
        """调用端到端研发流水线（支持单因子、多因子 ML 合成、可选 STGP 演化挖掘）。"""
        from scripts.run_pipeline import run_pipeline

        factor_expr = payload.get("factor_expr") or None
        if isinstance(factor_expr, str) and not factor_expr.strip():
            factor_expr = None
        run_gp = bool(payload.get("run_gp", False))
        gp_generations = int(payload.get("gp_generations", 3))
        gp_population = int(payload.get("gp_population", 40))
        ml_model = str(payload.get("ml_model", "equal_weight"))
        skip_seed = bool(payload.get("skip_seed", False))

        now_str = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = self.runs_dir / f"pipeline_{now_str}"

        res = run_pipeline(
            panel_dir=self.panel_dir,
            db_path=self.db_path,
            backtest_config_path=self.backtest_config_path,
            fees_config_path=self.fees_config_path,
            gp_config_path=self.gp_config_path,
            ml_config_path=self.ml_config_path,
            factor_expr=factor_expr,
            run_gp=run_gp,
            gp_generations=gp_generations,
            gp_population=gp_population,
            ml_model=ml_model,
            skip_seed=skip_seed,
            output_dir=out_dir,
            panel=self.panel,
        )

        bt_res: BacktestResult = res["backtest_result"]
        run_id = res["run_id"]
        return _sanitize_json_value(
            {
                "status": "ok",
                "run_id": run_id,
                "output_dir": res["output_dir"],
                "report_path": res["html_report"],
                "report_url": f"/api/runs/{run_id}/report",
                "summary": summary_dict(bt_res),
                "backtest": self._serialize_backtest_result(bt_res),
            }
        )

    def list_runs(self) -> list[dict[str, Any]]:
        """列出 runs/ 目录及 SQLite 注册表中的历史实验产出。"""
        db_runs = {r.run_id: r.to_dict() for r in self.registry.list_runs()}
        items: list[dict[str, Any]] = []
        seen: set[str] = set()

        if self.runs_dir.exists():
            for child in sorted(self.runs_dir.iterdir(), key=lambda p: p.name, reverse=True):
                if not child.is_dir():
                    continue
                rid = child.name
                seen.add(rid)
                report_html = child / "report.html"
                summary_data: dict[str, Any] = {}
                for s_name in (
                    "summary.json",
                    "metrics.json",
                    "report.json",
                    "pipeline_summary.json",
                ):
                    s_file = child / s_name
                    if s_file.exists():
                        try:
                            loaded = json.loads(s_file.read_text(encoding="utf-8"))
                            if isinstance(loaded, dict):
                                summary_data = loaded.get("summary", loaded)
                                break
                        except Exception:
                            pass

                db_rec = db_runs.get(rid, {})
                items.append(
                    {
                        "run_id": rid,
                        "kind": db_rec.get(
                            "kind",
                            "workflow"
                            if rid.startswith("wf_")
                            else ("pipeline" if rid.startswith("pipeline_") else "backtest"),
                        ),
                        "status": db_rec.get("status", "completed"),
                        "started_at": db_rec.get("started_at"),
                        "has_report": report_html.exists(),
                        "report_url": f"/api/runs/{rid}/report" if report_html.exists() else None,
                        "summary": {
                            "total_return": summary_data.get("total_return"),
                            "annual_return": summary_data.get("annual_return"),
                            "sharpe": summary_data.get("sharpe"),
                            "max_drawdown": summary_data.get("max_drawdown"),
                            "daily_turnover_mean": summary_data.get("daily_turnover_mean"),
                        },
                    }
                )

        for rid, rec in db_runs.items():
            if rid not in seen:
                items.append(
                    {
                        "run_id": rid,
                        "kind": rec.get("kind", "eval"),
                        "status": rec.get("status", "completed"),
                        "started_at": rec.get("started_at"),
                        "has_report": False,
                        "report_url": None,
                        "summary": {},
                    }
                )

        return _sanitize_json_value(items[:50])

    def get_run_report_html(self, run_id: str) -> str | None:
        """读取指定实验运行目录下的 report.html 内容。"""
        safe_id = Path(run_id).name
        report_file = self.runs_dir / safe_id / "report.html"
        if report_file.exists():
            return report_file.read_text(encoding="utf-8")
        return None
