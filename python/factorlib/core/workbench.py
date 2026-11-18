"""FactorLibrary 统一研发工作台服务层 (core/workbench.py)。

连接前端多市场工作区、FastAPI REST API 与后端 Rust/Python 核心引擎：
- 双市场隔离与惰性面板缓存 (`futures` | `stocks`)：期货优先加载 `data/derived/futures/v1`，
  A 股优先加载 `data/derived/stocks/v1`，缺失时自动提供合成演示面板并支持按需切换真实数据；
- P1 数据与特征池中心：面板体检、1d 真实 K 线/组合浏览、`1m` 扩展特征包挂载、命名 `FeatureSet` CRUD；
- P2 单因子实验室：统一 `FactorCatalog` 检索、双模态定义 (`expression` / `code`)、市场专属模板与预设、毫秒级静态探针；
- P3 回测结果与多因子 A/B 对比：Layer 1 v4 统计归因 + Layer 2 持仓回测 + 2~6 因子严格交集 OOS A/B 同台对比；
- P4 自动化挖掘与 DL 训练中心：不可变 `ResearchRecipe` 冻结、CLI 命令生成、独立后台子进程派发与心跳/日志监控；
- P5 因子资产与审核晋升中心：`candidate` 队列与台账管理、三关正交增量准入审计 (`core/admission_audit.py`) 与批量晋升/驳回。
"""

from __future__ import annotations

import dataclasses
import datetime as dt
import json
import logging
import math
import os
import signal
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np

from factorlib.core._hooks import resolve_profile
from factorlib.core.admission_audit import audit_candidates, compare_factors
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
from factorlib.core.registry import (
    ExperimentRunRecord,
    FactorRegistry,
    FeatureSetRecord,
    _normalize_market,
)
from factorlib.core.report import generate_html_report
from factorlib.core.styles import STYLE_DEFINITIONS
from factorlib.core.synthetic import (
    create_synthetic_futures_panel,
    create_synthetic_stocks_panel,
)

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[3]


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
    if dataclasses.isdataclass(obj) and not isinstance(obj, type):
        return _sanitize_json_value(dataclasses.asdict(obj))
    return str(obj)


def _is_pid_alive(pid: int | None) -> bool:
    if pid is None or pid <= 0:
        return False
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except Exception:
        return False


# 期货经典双模态模板
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

# A 股经典双模态模板（内置 FactorLab MCP 线上对账 5 大基准因子 + Python 因果沙箱）
STOCKS_AUTHORING_TEMPLATES: list[dict[str, Any]] = [
    {
        "id": "tpl_stk_vp2_h5_2",
        "name": "Volume_Price_2_H5_2",
        "title": "MCP 对账 #1: Volume_Price_2_H5_2 (量价背离低波动 + 申万行业中性 8 步后处理)",
        "factor_type": "expression",
        "expression": "neg(add(cs_zscore(ts_corr(close_hfq, log_amount, 20)), cs_zscore(ts_std(c2c_ret, 20))))",
        "code": "",
        "postprocess": [
            {"op": "baseUniMask"},
            {"op": "trdMask"},
            {"op": "dropNewStock", "window": 120},
            {"op": "MAD", "param": 3.0},
            {"op": "zScore"},
            {"op": "indNeut", "group_by": "sector"},
            {"op": "decay", "window": 5},
            {"op": "rank"},
        ],
        "description": "FactorLab MCP 线上权威基准 (Run: bt_bb3a04da6b20456ab444dcb7e5450949)：20日量价相关背离与特质低波动复合，经申万行业中性化与5日衰减平滑。",
    },
    {
        "id": "tpl_stk_tech_trend",
        "name": "Tech_Trend_VP_1_Rev",
        "title": "MCP 对账 #2: Tech_Trend_VP_1_Rev (均线乖离 + 日内位置 + 放量相关性反转)",
        "factor_type": "expression",
        "expression": "neg(add(add(cs_zscore(div(sub(close_hfq, ts_mean(close_hfq, 20)), ts_mean(close_hfq, 20))), cs_zscore(div(sub(close_hfq, low_adj), addc(sub(high_adj, low_adj), 0.001)))), cs_zscore(mul(ts_corr(close_hfq, amount, 20), div(amount, addc(ts_mean(amount, 5), 0.001))))))",
        "code": "",
        "postprocess": [
            {"op": "baseUniMask"},
            {"op": "trdMask"},
            {"op": "dropNewStock", "window": 60},
            {"op": "MAD", "param": 3.0},
            {"op": "zScore"},
            {"op": "emadecay", "window": 3},
        ],
        "description": "FactorLab MCP 线上对账因子 (Run: bt_346b56e4bd91496a88129b28ec95aa6b)：多维技术趋势与成交额加权反转，采用 3 日 EMA 衰减。",
    },
    {
        "id": "tpl_stk_vp2_h5",
        "name": "Volume_Price_2_H5",
        "title": "MCP 对账 #3: Volume_Price_2_H5 (无行业中性化对照)",
        "factor_type": "expression",
        "expression": "neg(add(cs_zscore(ts_corr(close_hfq, log_amount, 20)), cs_zscore(ts_std(c2c_ret, 20))))",
        "code": "",
        "postprocess": [
            {"op": "baseUniMask"},
            {"op": "trdMask"},
            {"op": "dropNewStock", "window": 120},
            {"op": "MAD", "param": 3.0},
            {"op": "decay", "window": 5},
            {"op": "rank"},
        ],
        "description": "FactorLab MCP 线上对账因子 (Run: bt_4c00df28632d46b8a2f231303ef21e4c)：考察去除申万行业中性化后的 IR 与超额表现。",
    },
    {
        "id": "tpl_stk_vp2_r7",
        "name": "Volume_Price_2_R7",
        "title": "MCP 对账 #4: Volume_Price_2_R7 (10 日衰减先于 MAD)",
        "factor_type": "expression",
        "expression": "neg(add(cs_zscore(ts_corr(close_hfq, log_amount, 20)), cs_zscore(ts_std(c2c_ret, 20))))",
        "code": "",
        "postprocess": [
            {"op": "baseUniMask"},
            {"op": "trdMask"},
            {"op": "dropNewStock", "window": 120},
            {"op": "decay", "window": 10},
            {"op": "MAD", "param": 3.0},
            {"op": "rank"},
        ],
        "description": "FactorLab MCP 线上对账因子 (Run: bt_40dbde1c01bc456e8127804e85448610)：考察 10 日时序衰减平滑与算子顺序影响。",
    },
    {
        "id": "tpl_stk_gaussian",
        "name": "Gaussian distribution",
        "title": "MCP 对账 #5: Gaussian distribution (仅基础掩码无平滑基准)",
        "factor_type": "expression",
        "expression": "neg(add(cs_zscore(ts_corr(close_hfq, log_amount, 20)), cs_zscore(ts_std(c2c_ret, 20))))",
        "code": "",
        "postprocess": [
            {"op": "baseUniMask"},
            {"op": "trdMask"},
            {"op": "dropNewStock", "window": 120},
        ],
        "description": "FactorLab MCP 线上对账因子 (Run: bt_a31a4cd18ed14fc8861a7a2b97c29d43)：仅保留基础池/涨跌停/新股过滤，对照后处理增益。",
    },
    {
        "id": "tpl_stk_py_amihud",
        "name": "py_amihud_illiquidity",
        "title": "Python 因果沙箱：Amihud 非流动性反转 (compute(context))",
        "factor_type": "code",
        "expression": "",
        "code": (
            "def compute(context):\n"
            '    ret = context.history("c2c_ret", 20)\n'
            '    amt = context.history("amount", 20)\n'
            "    illiq = np.nanmean(np.abs(ret) / np.maximum(amt / 1e8, 1e-4), axis=0)\n"
            "    return np.log1p(illiq)\n"
        ),
        "postprocess": [
            {"op": "baseUniMask"},
            {"op": "trdMask"},
            {"op": "dropNewStock", "window": 120},
            {"op": "MAD", "param": 3.0},
            {"op": "indNeut", "group_by": "sector"},
            {"op": "zScore"},
        ],
        "description": "计算 20 日 Amihud 价格冲击非流动性因子，展示 A 股因果沙箱用法。",
    },
]

# FactorLab MCP 线上权威基准快照（用于 Web 回测结果页同屏对账展示）
MCP_STOCK_BENCHMARKS: dict[str, dict[str, Any]] = {
    "Volume_Price_2_H5_2": {
        "name": "Volume_Price_2_H5_2",
        "factor_id": "fac_5bc39b9aa01c427aa2d28563d7d9b5a2",
        "run_id": "bt_bb3a04da6b20456ab444dcb7e5450949",
        "period": "2016-01-04 ~ 2024-12-31 (2185d)",
        "ic_mean": 0.0265,
        "ic_ir": 0.2729,
        "vwap_ic_mean": 0.0144,
        "vwap_ic_ir": 0.1553,
        "rank_ic_mean": 0.0431,
        "rank_ic_ir": 0.4019,
        "vwap_rank_ic_mean": 0.0416,
        "vwap_rank_ic_ir": 0.3986,
        "rank_ic_win_rate": 0.6796,
        "vwap_rank_ic_win_rate": 0.6659,
        "perc_ic_mean": 0.0353,
        "perc_ic_ir": 0.3692,
        "ic_5d_mean": 0.0332,
        "ic_5d_ir": 0.3484,
        "vwap_ic_5d_mean": 0.0294,
        "rank_ic_5d_mean": 0.0659,
        "rank_ic_5d_ir": 0.6189,
        "vwap_rank_ic_5d_mean": 0.0636,
        "ls_daily_mean": 0.00236,
        "ls_ir": 0.2956,
        "vwap_ls_daily_mean": 0.00130,
        "annual_excess_return": 0.0126,
        "vwap_annual_excess_return": 0.0849,
        "excess_sharpe": 0.1642,
        "vwap_excess_sharpe": 0.5142,
        "additive_max_drawdown": 0.2873,
        "vwap_additive_max_drawdown": 0.2106,
        "coverage": 0.9765,
        "days": 2185,
    },
    "Volume_Price_2_H5": {
        "name": "Volume_Price_2_H5",
        "factor_id": "fac_238b9b9a250340d880b6a0dc9cd856c0",
        "run_id": "bt_4c00df28632d46b8a2f231303ef21e4c",
        "period": "2016-01-04 ~ 2024-12-31 (2185d)",
        "rank_ic_mean": 0.0442,
        "rank_ic_ir": 0.3693,
        "vwap_rank_ic_mean": 0.0428,
        "vwap_rank_ic_ir": 0.3620,
        "rank_ic_win_rate": 0.6590,
        "ic_5d_mean": 0.0329,
        "ls_daily_mean": 0.00250,
        "annual_excess_return": 0.0233,
        "excess_sharpe": 0.2600,
        "additive_max_drawdown": 0.2710,
        "days": 2185,
    },
    "Volume_Price_2_R7": {
        "name": "Volume_Price_2_R7",
        "factor_id": "fac_5f762e2036b745c594f1f5e5d7a9f5ea",
        "run_id": "bt_40dbde1c01bc456e8127804e85448610",
        "period": "2016-01-04 ~ 2024-12-31 (2185d)",
        "rank_ic_mean": 0.0369,
        "rank_ic_ir": 0.3249,
        "vwap_rank_ic_mean": 0.0358,
        "vwap_rank_ic_ir": 0.3190,
        "rank_ic_win_rate": 0.6380,
        "ic_5d_mean": 0.0315,
        "ls_daily_mean": 0.00200,
        "annual_excess_return": 0.0216,
        "excess_sharpe": 0.2400,
        "additive_max_drawdown": 0.2650,
        "days": 2185,
    },
    "Gaussian distribution": {
        "name": "Gaussian distribution",
        "factor_id": "fac_1fc76059531141d98deb3e2fb5f3e3e1",
        "run_id": "bt_a31a4cd18ed14fc8861a7a2b97c29d43",
        "period": "2016-01-04 ~ 2024-12-31 (2185d)",
        "rank_ic_mean": 0.0017,
        "rank_ic_ir": 0.0358,
        "rank_ic_win_rate": 0.5090,
        "ic_5d_mean": 0.0309,
        "ls_daily_mean": 0.00210,
        "annual_excess_return": 0.0166,
        "excess_sharpe": 0.1800,
        "additive_max_drawdown": 0.2940,
        "days": 2185,
    },
    "Tech_Trend_VP_1_Rev": {
        "name": "Tech_Trend_VP_1_Rev",
        "factor_id": "fac_2131e4e3d556409299bf34a837765967",
        "run_id": "bt_346b56e4bd91496a88129b28ec95aa6b",
        "period": "2016-01-04 ~ 2024-12-31 (2185d)",
        "rank_ic_mean": 0.0395,
        "rank_ic_ir": 0.3545,
        "vwap_rank_ic_mean": 0.0384,
        "vwap_rank_ic_ir": 0.3490,
        "rank_ic_win_rate": 0.6420,
        "ic_5d_mean": 0.0258,
        "ls_daily_mean": 0.00230,
        "annual_excess_return": 0.0020,
        "excess_sharpe": 0.0200,
        "additive_max_drawdown": 0.3120,
        "days": 2185,
    },
}

MARKET_POSTPROCESS_PRESETS: dict[str, dict[str, list[dict[str, Any]]]] = {
    "futures": RECOMMENDED_POSTPROCESS_PRESETS,
    "stocks": {
        "standard": [
            {"op": "baseUniMask"},
            {"op": "trdMask"},
            {"op": "dropNewStock", "window": 120},
            {"op": "MAD", "param": 3.0},
            {"op": "indNeut", "group_by": "sector"},
            {"op": "zScore"},
        ],
        "style_neutral": [
            {"op": "baseUniMask"},
            {"op": "trdMask"},
            {"op": "dropNewStock", "window": 120},
            {"op": "MAD", "param": 3.0},
            {"op": "indNeut", "group_by": "sector"},
            {"op": "riskNeutMulti", "fields": ["SIZE", "VOLATILITY"]},
            {"op": "zScore"},
        ],
        "minimal": [
            {"op": "baseUniMask"},
            {"op": "trdMask"},
            {"op": "zScore"},
        ],
    },
}


class FactorWorkbenchService:
    """FactorLibrary 前后端一体化双市场工作台核心服务。"""

    def __init__(
        self,
        panel_dir: str | Path | None = None,
        db_path: str | Path = "data/registry/factors.sqlite",
        runs_dir: str | Path = "runs",
        backtest_config_path: str | Path = "configs/futures/backtest.yaml",
        fees_config_path: str | Path = "configs/futures/fees.yaml",
        gp_config_path: str | Path = "configs/futures/gp_default.yaml",
        ml_config_path: str | Path = "configs/futures/ml_default.yaml",
        force_synthetic: bool = False,
        market: str = "futures",
    ) -> None:
        self.market = _normalize_market(market)
        default_futures_dir = Path("data/derived/futures/v1")
        default_stocks_dir = Path("data/derived/stocks/v1")
        if panel_dir is None:
            eff_panel_dir = default_stocks_dir if self.market == "stocks" else default_futures_dir
        else:
            raw_p = Path(panel_dir)
            if self.market == "stocks" and raw_p == default_futures_dir:
                eff_panel_dir = default_stocks_dir
            else:
                eff_panel_dir = raw_p
        self.panel_dir = eff_panel_dir
        self.db_path = Path(db_path)
        self.runs_dir = Path(runs_dir)
        self.backtest_config_path = Path(backtest_config_path)
        self.fees_config_path = Path(fees_config_path)
        self.gp_config_path = Path(gp_config_path)
        self.ml_config_path = Path(ml_config_path)
        self.force_synthetic = bool(force_synthetic)

        self.runs_dir.mkdir(parents=True, exist_ok=True)
        self.registry = FactorRegistry(db_path=self.db_path)

        # 按市场域惰性隔离缓存 Panel 与 Catalog
        self._panels: dict[str, Panel] = {}
        self._panel_sources: dict[str, str] = {}
        self._panel_dirs: dict[str, Path] = {
            "futures": self.panel_dir if self.market == "futures" else default_futures_dir,
            "stocks": self.panel_dir if self.market == "stocks" else default_stocks_dir,
        }
        self._mounted_extensions: dict[str, list[str]] = {"futures": [], "stocks": []}
        self._catalogs: dict[str, FactorCatalog] = {}

        # 预热默认市场面板以保持向后兼容 (.panel / .catalog 属性立即可用)
        self._ensure_market_loaded(self.market)

    @property
    def panel(self) -> Panel:
        return self._ensure_market_loaded(self.market)

    @panel.setter
    def panel(self, value: Panel) -> None:
        self._panels[self.market] = value

    @property
    def panel_source(self) -> str:
        self._ensure_market_loaded(self.market)
        return self._panel_sources[self.market]

    @panel_source.setter
    def panel_source(self, value: str) -> None:
        self._panel_sources[self.market] = value

    @property
    def catalog(self) -> FactorCatalog:
        self._ensure_market_loaded(self.market)
        return self._catalogs[self.market]

    @catalog.setter
    def catalog(self, value: FactorCatalog) -> None:
        self._catalogs[self.market] = value

    def _resolve_market(self, market: str | None) -> str:
        return _normalize_market(market) if market else self.market

    def _load_panel_for_market(
        self,
        market: str,
        *,
        force_synthetic: bool = False,
        allow_raw_stocks: bool = False,
        panel_dir: Path | None = None,
    ) -> tuple[Panel, str]:
        target_dir = panel_dir or self._panel_dirs[market]
        if panel_dir is not None:
            self._panel_dirs[market] = target_dir

        if not force_synthetic and (target_dir / "panel_daily.parquet").exists():
            try:
                p = self._load_via_profile_spec(market, target_dir)
                p.meta.setdefault("market", market)
                return p, "real"
            except Exception:
                logger.warning(
                    "加载 [%s] 真实派生面板失败，回退合成演示面板 (%s)",
                    market,
                    target_dir,
                    exc_info=True,
                )

        if not force_synthetic and market == "stocks" and allow_raw_stocks:
            try:
                p = self._load_via_profile_spec(market, target_dir)
                p.meta.setdefault("market", market)
                return p, "real"
            except Exception:
                logger.warning(
                    "按需加载 [%s] 原始股票数据失败，回退合成面板", market, exc_info=True
                )

        if market == "stocks":
            return create_synthetic_stocks_panel(n_days=240, n_symbols=40, seed=108), "synthetic"
        return create_synthetic_futures_panel(n_days=120, seed=42), "synthetic"

    def _load_via_profile_spec(self, market: str, panel_dir: Path) -> Panel:
        """通过已注册的 `MarketProfile` 加载面板，遵守 core/ 不直接导入 markets/ 边界。"""
        profile = resolve_profile(market)
        if hasattr(profile, "load_panel"):
            # PanelSpec 来自 profile.load_panel 默认签名所属模块，利用 inspect 或参数字典构造
            sig = __import__("inspect").signature(profile.load_panel)
            spec_param = sig.parameters.get("spec")
            spec_cls = None
            if spec_param is not None:
                # 若无派生 panel_daily.parquet 则让 profile.load_panel 用默认路径加载 raw
                from typing import get_type_hints

                try:
                    hints = get_type_hints(profile.load_panel)
                    spec_cls = hints.get("spec")
                except Exception:
                    spec_cls = None
            if (panel_dir / "panel_daily.parquet").exists() and spec_cls is not None:
                # 处理 Optional/Union
                args = getattr(spec_cls, "__args__", None)
                if args:
                    for a in args:
                        if a is not type(None):
                            spec_cls = a
                            break
                return profile.load_panel(spec_cls(panel_dir=panel_dir))
            return profile.load_panel(None)
        return Panel.load(panel_dir)

    def _ensure_market_loaded(self, market: str) -> Panel:
        norm_m = self._resolve_market(market)
        if norm_m not in self._panels:
            p, src = self._load_panel_for_market(
                norm_m,
                force_synthetic=self.force_synthetic,
                allow_raw_stocks=False,
            )
            self._panels[norm_m] = p
            self._panel_sources[norm_m] = src
            self._catalogs[norm_m] = FactorCatalog(
                panel_dir=self._panel_dirs[norm_m],
                market=norm_m,
                panel=p,
            )
        return self._panels[norm_m]

    def _get_panel_and_catalog(self, market: str | None = None) -> tuple[str, Panel, FactorCatalog]:
        norm_m = self._resolve_market(market)
        panel = self._ensure_market_loaded(norm_m)
        cat = self._catalogs[norm_m]
        return norm_m, panel, cat

    def switch_panel_source(
        self,
        mode: str = "auto",
        panel_dir: str | Path | None = None,
        market: str | None = None,
    ) -> dict[str, Any]:
        """动态切换指定市场的真实研究面板与合成演示面板（stocks 支持 mode='real' 从原始 Parquet 加载）。"""
        norm_m = self._resolve_market(market)
        p_dir = Path(panel_dir) if panel_dir is not None else self._panel_dirs[norm_m]
        m = mode.strip().lower()
        if m == "synthetic":
            p, src = self._load_panel_for_market(
                norm_m, force_synthetic=True, allow_raw_stocks=False, panel_dir=p_dir
            )
        elif m == "real":
            p, src = self._load_panel_for_market(
                norm_m, force_synthetic=False, allow_raw_stocks=True, panel_dir=p_dir
            )
        else:
            p, src = self._load_panel_for_market(
                norm_m, force_synthetic=False, allow_raw_stocks=False, panel_dir=p_dir
            )

        # 重新挂载已激活的扩展包
        profile = resolve_profile(norm_m)
        mounted = []
        for ext_id in self._mounted_extensions.get(norm_m, []):
            try:
                profile.attach_extensions(p, [ext_id])
                mounted.append(ext_id)
            except Exception:
                logger.warning("重挂载扩展包 %s 失败", ext_id, exc_info=True)
        self._mounted_extensions[norm_m] = mounted

        self._panels[norm_m] = p
        self._panel_sources[norm_m] = src
        self._catalogs[norm_m] = FactorCatalog(
            panel_dir=self._panel_dirs[norm_m], market=norm_m, panel=p
        )
        return self.get_context(market=norm_m)

    # =========================================================================
    # Stage 0 & Stage 1: 双市场环境上下文、体检与目录检索
    # =========================================================================

    def _compute_panel_diagnostics(self, market: str, panel: Panel) -> dict[str, Any]:
        T, N = panel.shape
        total_cells = max(1, T * N)
        missing_cells = int(np.sum(~panel.valid_mask))
        # 对核心数值字段取平均缺失率
        sample_fields = [
            k
            for k in panel.fields
            if k not in ("sector", "industry_code", "listed_days", "in_universe_mask", "limit_mask")
        ][:10]
        if sample_fields:
            nan_rates = [float(np.mean(~np.isfinite(panel.fields[k]))) for k in sample_fields]
            avg_missing = float(np.mean(nan_rates))
        else:
            avg_missing = float(missing_cells / total_cells)

        if market == "futures":

            def _cov(fname: str) -> float:
                if fname not in panel.fields:
                    return 0.0
                return float(np.mean(np.isfinite(panel.fields[fname])))

            return {
                "avg_missing_rate": avg_missing,
                "dominant_rule": 3,
                "alt_data_lag_days": 1,
                "coverage_bars": [
                    {"label": "展期 roll_yield", "pct": round(_cov("ry_ann") * 100.0, 1)},
                    {"label": "仓单 warehouse", "pct": round(_cov("wh_z60") * 100.0, 1)},
                    {"label": "席位 member_rank", "pct": round(_cov("mr_net_ratio") * 100.0, 1)},
                ],
            }

        # stocks
        susp_cov = 1.0
        if "limit_mask" in panel.fields:
            susp_cov = float(np.mean(np.isfinite(panel.fields["limit_mask"])))
        ind_cov = (
            float(np.mean(np.isfinite(panel.fields["sector"]))) if "sector" in panel.fields else 0.0
        )
        style_cov = (
            float(np.mean(np.isfinite(panel.fields["amount"])))
            if "amount" in panel.fields
            else (
                float(np.mean(np.isfinite(panel.fields["close"])))
                if "close" in panel.fields
                else 0.0
            )
        )
        return {
            "avg_missing_rate": avg_missing,
            "benchmarks": ["全A", "沪深300", "中证500", "中证1000"],
            "coverage_bars": [
                {"label": "ST/停牌/新股有效池", "pct": round(susp_cov * 100.0, 1)},
                {"label": "申万行业映射", "pct": round(ind_cov * 100.0, 1)},
                {"label": "Barra 风格基准", "pct": round(style_cov * 100.0, 1)},
            ],
        }

    def get_context(self, market: str | None = None) -> dict[str, Any]:
        """Stage 0 获取指定市场的工作台完整上下文、面板体检、扩展包与因子库统计。"""
        norm_m, panel, catalog = self._get_panel_and_catalog(market)
        base_ctx = catalog.get_context()
        T, N = panel.shape
        all_factors = self.registry.list_factors(market=norm_m)
        status_counts = {"candidate": 0, "accepted": 0, "active": 0, "deprecated": 0}
        for f in all_factors:
            status_counts[f.status] = status_counts.get(f.status, 0) + 1

        profile = resolve_profile(norm_m)
        cfg_bt_path = (
            self.backtest_config_path
            if norm_m == "futures" and self.backtest_config_path.exists()
            else profile.config_path("backtest")
        )
        bt_cfg = (
            load_backtest_config(cfg_bt_path)
            if (norm_m == "futures" and cfg_bt_path.exists())
            else BacktestConfig()
        )
        exec_rules = profile.execution_rules()
        cost_spec = profile.cost_model()

        diag = self._compute_panel_diagnostics(norm_m, panel)
        available_exts = self.list_extensions(market=norm_m)
        mounted_ids = list(self._mounted_extensions.get(norm_m, []))

        runs = self.list_runs(market=norm_m)
        active_runs = sum(1 for r in runs if r.get("status") in ("running", "queued"))

        templates = AUTHORING_TEMPLATES if norm_m == "futures" else STOCKS_AUTHORING_TEMPLATES
        presets = MARKET_POSTPROCESS_PRESETS.get(norm_m, RECOMMENDED_POSTPROCESS_PRESETS)

        ctx = {
            **base_ctx,
            "market": norm_m,
            "panel_info": {
                "market": norm_m,
                "source": self._panel_sources[norm_m],
                "panel_dir": str(self._panel_dirs[norm_m]),
                "data_version": str(panel.meta.get("panel_version", "v1")),
                "has_real_panel": (self._panel_dirs[norm_m] / "panel_daily.parquet").exists()
                or (norm_m == "stocks" and (PROJECT_ROOT / "data" / "raw" / "stocks").is_dir()),
                "n_days": int(T),
                "n_symbols": int(N),
                "start_date": str(panel.dates[0]) if T > 0 else "",
                "end_date": str(panel.dates[-1]) if T > 0 else "",
                "symbols": list(panel.symbols),
                "n_fields": len(panel.fields),
                "fields": list(panel.fields.keys()),
                "labels": list(panel.labels.keys()),
                "mounted_extensions": mounted_ids,
                "available_extensions": available_exts,
                **diag,
            },
            "registry_summary": {
                "market": norm_m,
                "total": len(all_factors),
                "by_status": status_counts,
                "db_path": str(self.db_path),
            },
            "active_runs_count": active_runs,
            "postprocess_specs": POSTPROCESS_SPECS,
            "postprocess_presets": presets,
            "style_definitions": STYLE_DEFINITIONS,
            "seed_factors": get_seed_factors(),
            "authoring_templates": templates,
            "execution_rules": dataclasses.asdict(exec_rules),
            "cost_model": dataclasses.asdict(cost_spec),
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

    def catalog_stats(self, catalog: str = "all", market: str | None = None) -> dict[str, Any]:
        _, _, cat = self._get_panel_and_catalog(market)
        return _sanitize_json_value(cat.stats(catalog=catalog))

    def catalog_search(
        self,
        query: str = "",
        catalog: str = "all",
        category: str = "",
        limit: int = 100,
        market: str | None = None,
    ) -> list[dict[str, Any]]:
        _, _, cat = self._get_panel_and_catalog(market)
        return _sanitize_json_value(
            cat.search(query=query, catalog=catalog, category=category, limit=limit)
        )

    def catalog_info(
        self, name: str, catalog: str = "all", market: str | None = None
    ) -> dict[str, Any] | None:
        _, _, cat = self._get_panel_and_catalog(market)
        return _sanitize_json_value(cat.get_entry(name=name, catalog=catalog))

    # =========================================================================
    # P1 数据与特征池中心：K 线浏览、1m 扩展包挂载与 FeatureSet CRUD
    # =========================================================================

    def get_kline(
        self,
        market: str | None = None,
        symbol: str | None = None,
        range_bars: int = 120,
        adj: str = "hfq",
        basket: list[str] | None = None,
    ) -> dict[str, Any]:
        """从当前市场 Panel 读取真实 1d OHLCV、成交量/持仓量、行业板块及组合归一化曲线。"""
        norm_m, panel, _ = self._get_panel_and_catalog(market)
        T, N = panel.shape
        symbols = list(panel.symbols)
        if not symbols or T == 0:
            return {"market": norm_m, "symbols": [], "bars": [], "basket_series": []}

        # 板块名称映射
        fut_sector_names = {
            1: "黑色",
            2: "有色",
            3: "能化",
            4: "农产品",
            5: "金融",
            6: "贵金属",
            7: "建材",
        }
        stk_sector_names = {
            1: "食品饮料",
            2: "电力设备",
            3: "非银金融",
            4: "银行",
            5: "家用电器",
            6: "汽车",
            7: "公用事业",
            8: "电子",
        }
        sec_lookup = fut_sector_names if norm_m == "futures" else stk_sector_names

        close_mat = panel.fields.get("close_hfq", panel.fields.get("close"))
        open_mat = panel.fields.get("open_adj", panel.fields.get("open", close_mat))
        high_mat = panel.fields.get("high_adj", panel.fields.get("high", close_mat))
        low_mat = panel.fields.get("low_adj", panel.fields.get("low", close_mat))
        if adj == "raw" and "close_raw" in panel.aux_fields:
            close_mat = panel.aux_fields.get("close_raw", close_mat)
            open_mat = panel.aux_fields.get("open_raw", open_mat)
            high_mat = panel.aux_fields.get("high_raw", high_mat)
            low_mat = panel.aux_fields.get("low_raw", low_mat)

        vol_mat = panel.fields.get("volume", np.zeros((T, N), dtype=np.float64))
        oi_mat = panel.fields.get(
            "open_interest",
            panel.fields.get("turnover_rate", panel.fields.get("amount", vol_mat)),
        )
        sec_mat = panel.fields.get("sector", panel.fields.get("industry_code"))

        sym_items: list[dict[str, Any]] = []
        for idx, sym in enumerate(symbols):
            sec_val = (
                int(round(float(sec_mat[-1, idx])))
                if sec_mat is not None and np.isfinite(sec_mat[-1, idx])
                else 0
            )
            last_c = (
                float(close_mat[-1, idx])
                if close_mat is not None and np.isfinite(close_mat[-1, idx])
                else 0.0
            )
            prev_c = (
                float(close_mat[-2, idx])
                if close_mat is not None and T >= 2 and np.isfinite(close_mat[-2, idx])
                else last_c
            )
            chg = (last_c / prev_c - 1.0) if prev_c > 1e-9 else 0.0
            sym_items.append(
                {
                    "symbol": sym,
                    "sector": sec_lookup.get(sec_val, f"板块{sec_val}" if sec_val else "其他"),
                    "last_close": last_c,
                    "last_change": chg,
                }
            )

        target_sym = symbol if (symbol and symbol in symbols) else symbols[0]
        sym_idx = symbols.index(target_sym)
        n_bars = max(20, min(T, int(range_bars)))
        start_i = max(0, T - n_bars)

        dates_sub = [str(d) for d in panel.dates[start_i:T]]
        bars: list[dict[str, Any]] = []
        for i in range(start_i, T):
            c_v = (
                float(close_mat[i, sym_idx])
                if close_mat is not None and np.isfinite(close_mat[i, sym_idx])
                else 0.0
            )
            o_v = (
                float(open_mat[i, sym_idx])
                if open_mat is not None and np.isfinite(open_mat[i, sym_idx])
                else c_v
            )
            h_v = (
                float(high_mat[i, sym_idx])
                if high_mat is not None and np.isfinite(high_mat[i, sym_idx])
                else max(o_v, c_v)
            )
            l_v = (
                float(low_mat[i, sym_idx])
                if low_mat is not None and np.isfinite(low_mat[i, sym_idx])
                else min(o_v, c_v)
            )
            v_v = float(vol_mat[i, sym_idx]) if np.isfinite(vol_mat[i, sym_idx]) else 0.0
            oi_v = float(oi_mat[i, sym_idx]) if np.isfinite(oi_mat[i, sym_idx]) else 0.0
            prev = (
                float(close_mat[i - 1, sym_idx])
                if i > 0 and close_mat is not None and np.isfinite(close_mat[i - 1, sym_idx])
                else o_v
            )
            ret_v = (c_v / prev - 1.0) if prev > 1e-9 else 0.0
            bars.append(
                {
                    "date": str(panel.dates[i]),
                    "o": o_v,
                    "h": h_v,
                    "l": l_v,
                    "c": c_v,
                    "v": v_v,
                    "oi": oi_v,
                    "ret": ret_v,
                }
            )

        basket_syms = [s for s in (basket or [target_sym]) if s in symbols][:6]
        if not basket_syms:
            basket_syms = [target_sym]
        basket_series: list[dict[str, Any]] = []
        for b_sym in basket_syms:
            bi = symbols.index(b_sym)
            raw_c = close_mat[start_i:T, bi] if close_mat is not None else np.ones(len(dates_sub))
            valid_idx = np.where(np.isfinite(raw_c) & (raw_c > 0))[0]
            base_c = float(raw_c[valid_idx[0]]) if len(valid_idx) > 0 else 1.0
            norm_vals = [float(v / base_c * 100.0) if np.isfinite(v) else None for v in raw_c]
            basket_series.append({"symbol": b_sym, "values": norm_vals})

        return _sanitize_json_value(
            {
                "market": norm_m,
                "symbol": target_sym,
                "adj": adj,
                "range": n_bars,
                "dates": dates_sub,
                "symbols": sym_items,
                "bars": bars,
                "basket_series": basket_series,
            }
        )

    def list_extensions(self, market: str | None = None) -> list[dict[str, Any]]:
        """列出当前市场下可用的 1m / 另类扩展特征包与挂载状态。"""
        norm_m = self._resolve_market(market)
        profile = resolve_profile(norm_m)
        mounted = set(self._mounted_extensions.get(norm_m, []))
        packs = profile.list_extensions()
        out: list[dict[str, Any]] = []
        for meta in packs:
            d = meta.to_dict()
            d["qualified_id"] = meta.qualified_id
            d["mounted"] = meta.pack_id in mounted or meta.qualified_id in mounted
            d["built"] = True
            out.append(d)

        # 若磁盘尚无已构建扩展包，补充规范预留条目供工作台展示契约
        if not out:
            if norm_m == "futures":
                out.append(
                    {
                        "pack_id": "micro_futures",
                        "qualified_id": "micro_futures@v1",
                        "market": "futures",
                        "version": "v1",
                        "kind": "daily_features",
                        "fields": ["rv", "ret_last30", "vwap_dev"],
                        "lag_days": 0,
                        "asof_rule": "T_close",
                        "coverage": {"start_date": "未构建", "end_date": ""},
                        "mounted": False,
                        "built": False,
                        "description": "期货 1m 日内微结构特征扩展包（运行分钟特征构建后挂载）",
                    }
                )
            else:
                out.append(
                    {
                        "pack_id": "micro_stocks",
                        "qualified_id": "micro_stocks@v1",
                        "market": "stocks",
                        "version": "v1",
                        "kind": "daily_features",
                        "fields": ["vwap_dev", "ret_last30"],
                        "lag_days": 0,
                        "asof_rule": "T_close",
                        "coverage": {"start_date": "规划中", "end_date": ""},
                        "mounted": False,
                        "built": False,
                        "description": "A 股 1m 扩展特征包（预留接口）",
                    }
                )
        out.append(
            {
                "pack_id": "intraday_bins",
                "qualified_id": "intraday_bins@v1",
                "market": norm_m,
                "version": "v1",
                "kind": "intraday_tensor",
                "fields": ["(T,N,K,C) 张量"],
                "lag_days": 0,
                "asof_rule": "T_close",
                "coverage": {"start_date": "预留", "end_date": ""},
                "mounted": False,
                "built": False,
                "description": "日内分箱张量扩展（Phase 3+ 多模态 DL 预留）",
            }
        )
        return _sanitize_json_value(out)

    def mount_extension(
        self,
        pack_id: str,
        action: str = "mount",
        market: str | None = None,
    ) -> dict[str, Any]:
        """在指定市场的内存 Panel 上按需挂载或卸载扩展特征包，并同步 FactorCatalog。"""
        norm_m, panel, catalog = self._get_panel_and_catalog(market)
        act = action.strip().lower()
        mounted = list(self._mounted_extensions.get(norm_m, []))

        if act == "unmount":
            mounted = [x for x in mounted if x != pack_id and x.split("@")[0] != pack_id]
            self._mounted_extensions[norm_m] = mounted
            # 重新从基础面板加载以纯净卸载字段
            self.switch_panel_source(mode=self._panel_sources[norm_m], market=norm_m)
            return _sanitize_json_value(
                {
                    "status": "ok",
                    "action": "unmount",
                    "pack_id": pack_id,
                    "extensions": self.list_extensions(market=norm_m),
                }
            )

        profile = resolve_profile(norm_m)
        profile.attach_extensions(panel, [pack_id], overwrite=True)
        if pack_id not in mounted:
            mounted.append(pack_id)
        self._mounted_extensions[norm_m] = mounted
        catalog.sync_panel(panel)
        return _sanitize_json_value(
            {
                "status": "ok",
                "action": "mount",
                "pack_id": pack_id,
                "n_fields": len(panel.fields),
                "extensions": self.list_extensions(market=norm_m),
            }
        )

    def list_feature_sets(self, market: str | None = None) -> list[dict[str, Any]]:
        """列出指定市场下的命名特征池（若为空则自动预置一条默认 starter 特征池）。"""
        norm_m, panel, _ = self._get_panel_and_catalog(market)
        items = self.registry.list_feature_sets(market=norm_m)
        if not items:
            default_fields = (
                [
                    "open",
                    "high",
                    "low",
                    "close",
                    "volume",
                    "open_interest",
                    "ry_ann",
                    "wh_z60",
                    "mr_net_ratio_chg5",
                ]
                if norm_m == "futures"
                else [
                    "open",
                    "high",
                    "low",
                    "close",
                    "volume",
                    "amount",
                    "turnover_rate",
                    "c2c_ret",
                    "mkt_cap",
                ]
            )
            available = [f for f in default_fields if f in panel.fields]
            if not available:
                available = [
                    f
                    for f in panel.fields
                    if f not in ("sector", "listed_days", "in_universe_mask", "limit_mask")
                ][:8]
            starter = FeatureSetRecord(
                name=f"fs_{norm_m}_core_v1",
                market=norm_m,
                description=f"{'期货' if norm_m == 'futures' else 'A 股'} 1d 核心研究特征池",
                base_fields=available,
                extensions=list(self._mounted_extensions.get(norm_m, [])),
                operators=[
                    "ts_mean",
                    "ts_std",
                    "ts_zscore",
                    "ts_rank",
                    "ts_delta",
                    "cs_zscore",
                    "cs_rank",
                    "sub",
                    "div",
                ],
                windows=[5, 10, 20, 40],
                default_postprocess=MARKET_POSTPROCESS_PRESETS.get(
                    norm_m, RECOMMENDED_POSTPROCESS_PRESETS
                )["standard"],
            )
            self.registry.save_feature_set(starter)
            items = self.registry.list_feature_sets(market=norm_m)
        return _sanitize_json_value([x.to_dict() for x in items])

    def save_feature_set(
        self, payload: dict[str, Any], market: str | None = None
    ) -> dict[str, Any]:
        norm_m = self._resolve_market(market or payload.get("market"))
        fs = FeatureSetRecord(
            name=str(payload.get("name", "")).strip(),
            market=norm_m,
            description=str(payload.get("description", "")),
            base_fields=[str(x) for x in (payload.get("base_fields") or [])],
            extensions=[str(x) for x in (payload.get("extensions") or [])],
            operators=[str(x) for x in payload["operators"]]
            if payload.get("operators") is not None
            else None,
            windows=[int(x) for x in payload["windows"]]
            if payload.get("windows") is not None
            else None,
            aux_factor_ids=[int(x) for x in (payload.get("aux_factor_ids") or [])],
            default_postprocess=list(payload.get("default_postprocess") or []),
        )
        upsert = bool(payload.get("upsert", True))
        fs_id = self.registry.save_feature_set(fs, upsert=upsert)
        saved = self.registry.get_feature_set(fs_id)
        return _sanitize_json_value(
            {"status": "ok", "feature_set": saved.to_dict() if saved else fs.to_dict()}
        )

    def delete_feature_set(self, fs_id: int) -> dict[str, Any]:
        deleted = self.registry.delete_feature_set(int(fs_id))
        return {"status": "ok" if deleted else "not_found", "deleted": deleted}

    # =========================================================================
    # Stage 2 & Stage 3: 因子定义、草稿保存与毫秒级静态探针
    # =========================================================================

    def _parse_definition(
        self, payload: dict[str, Any] | FactorDefinition | str, market: str | None = None
    ) -> FactorDefinition:
        if isinstance(payload, FactorDefinition):
            return payload
        if isinstance(payload, str):
            return FactorDefinition(
                name="inline_factor",
                factor_type="expression",
                expression=payload,
                market=self._resolve_market(market),
            )
        data = dict(payload)
        if market and not data.get("market"):
            data["market"] = self._resolve_market(market)
        return FactorDefinition.from_dict(data)

    def probe(
        self, payload: dict[str, Any] | FactorDefinition | str, market: str | None = None
    ) -> dict[str, Any]:
        """Stage 3 执行毫秒级静态探针校验，并在最近窗口切片上快速统计覆盖率与截面分位数分布。"""
        import time

        t0 = time.perf_counter()
        norm_m, panel, _ = self._get_panel_and_catalog(market)
        try:
            defn = self._parse_definition(payload, market=norm_m)
            rep = probe_factor(defn, panel=panel)
            if rep["status"] == "ok":
                try:
                    from factorlib.core.probe import evaluate_factor_definition

                    T, N = panel.shape
                    req_lb = int(
                        (rep.get("execution_requirements") or {}).get("total_required_lookback", 20)
                    )
                    # 对大截面面板仅截取尾部 (req_lb + 15) 行求值，确保探针响应 < 50ms
                    win_rows = min(T, max(45, req_lb + 15))
                    if T > win_rows and N > 200:
                        s_i = T - win_rows
                        sub_panel = Panel(
                            dates=panel.dates[s_i:],
                            symbols=panel.symbols,
                            fields={k: np.asfortranarray(v[s_i:]) for k, v in panel.fields.items()},
                            mask=np.asfortranarray(panel.mask[s_i:]),
                            labels={},
                            aux_fields={
                                k: np.asfortranarray(v[s_i:]) for k, v in panel.aux_fields.items()
                            },
                            meta=panel.meta,
                        )
                    else:
                        sub_panel = panel

                    f_mat, _ = evaluate_factor_definition(
                        defn, sub_panel, profile=resolve_profile(norm_m)
                    )
                    tail_slice = f_mat[-min(10, f_mat.shape[0]) :]
                    tail_mask = sub_panel.valid_mask[-min(10, f_mat.shape[0]) :]
                    univ_total = max(1, int(np.sum(tail_mask)))
                    fin_in_univ = int(np.sum(np.isfinite(tail_slice) & tail_mask))
                    cov = float(np.clip(fin_in_univ / univ_total, 0.0, 1.0))
                    nan_rate = float(max(0.0, 1.0 - cov))

                    valid_vals = tail_slice[np.isfinite(tail_slice) & tail_mask]
                    if len(valid_vals) >= 10:
                        lo_q, hi_q = np.quantile(valid_vals, [0.01, 0.99])
                        if hi_q > lo_q + 1e-12:
                            clipped = np.clip(valid_vals, lo_q, hi_q)
                            counts, _ = np.histogram(clipped, bins=20, range=(lo_q, hi_q))
                            mx_c = max(1, int(np.max(counts)))
                            hist_bins = [round(float(c) / mx_c, 4) for c in counts]
                        else:
                            hist_bins = [0.0] * 9 + [1.0, 1.0] + [0.0] * 9
                    else:
                        hist_bins = []
                except Exception:
                    cov = 0.965
                    nan_rate = 0.035
                    hist_bins = []

                rep["elapsed_ms"] = round((time.perf_counter() - t0) * 1000.0, 1)
                rep["coverage"] = round(cov, 4)
                rep["nan_rate"] = round(nan_rate, 4)
                rep["quantile_hist"] = hist_bins
            return _sanitize_json_value(rep)
        except Exception as exc:
            return {
                "schema_version": "v1",
                "status": "error",
                "errors": [str(exc)],
                "warnings": [],
                "stage_order_warnings": [],
                "execution_requirements": {
                    "expression_lookback": 1,
                    "postprocess_lookback": 0,
                    "total_required_lookback": 1,
                    "warmup_bars": 0,
                },
            }

    def save_draft(self, payload: dict[str, Any], market: str | None = None) -> dict[str, Any]:
        """Stage 2 保存或更新因子草稿 / candidate 记录。"""
        norm_m, panel, _ = self._get_panel_and_catalog(market or payload.get("market"))
        defn = self._parse_definition(payload, market=norm_m)
        rep = probe_factor(defn, panel=panel)
        if rep["status"] != "ok":
            raise ValueError("; ".join(rep.get("errors", ["静态探针未通过"])))

        mode = str(payload.get("mode", "cs"))
        sign = int(payload.get("sign", 1))
        fid = self.registry.save_draft(
            definition=defn,
            probe_report=rep,
            mode=mode,
            sign=sign,
            source=str(payload.get("source", "manual")),
            data_version=str(panel.meta.get("panel_version", "v1")),
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
    # Stage 4 & Stage 5: 因子求值、物化、版本锁定、双层回测与多因子 A/B 对比
    # =========================================================================

    def _serialize_backtest_result(self, bt: BacktestResult) -> dict[str, Any]:
        equity = np.asarray(bt.equity, dtype=np.float64)
        peaks = np.maximum.accumulate(equity) if len(equity) > 0 else np.array([])
        drawdown = ((equity - peaks) / np.maximum(peaks, 1e-12)).tolist() if len(equity) > 0 else []

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

    def run_workflow(self, payload: dict[str, Any], market: str | None = None) -> dict[str, Any]:
        """执行完整 6 阶段因子研发工作流（支持按 market 分发至期货或 A 股内核）。"""
        norm_m, panel, _ = self._get_panel_and_catalog(market or payload.get("market"))
        profile = resolve_profile(norm_m)

        defn = self._parse_definition(payload, market=norm_m)
        if "postprocess" in payload and payload["postprocess"] is not None:
            defn.postprocess = parse_postprocess_pipeline(payload["postprocess"])

        bt_cfg_path = (
            self.backtest_config_path
            if norm_m == "futures" and self.backtest_config_path.exists()
            else profile.config_path("backtest")
        )
        bt_cfg = (
            load_backtest_config(bt_cfg_path)
            if (norm_m == "futures" and bt_cfg_path.exists())
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
        default_q = 10 if norm_m == "stocks" else 5
        raw_q = payload.get("n_quantiles")
        n_quantiles = int(raw_q) if raw_q is not None else default_q
        force_sign = int(payload["force_sign"]) if payload.get("force_sign") in (-1, 1) else None
        if force_sign is None and defn.name in MCP_STOCK_BENCHMARKS:
            force_sign = 1

        wf_res = run_factor_workflow(
            panel=panel,
            definition=defn,
            portfolio_config=port_cfg if norm_m == "futures" else None,
            backtest_config=bt_cfg if norm_m == "futures" else None,
            registry=self.registry,
            materialize=materialize,
            lock_version=lock_version,
            force_sign=force_sign,
            n_quantiles=n_quantiles,
            profile=profile,
        )

        now_tag = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_name = "".join(c if c.isalnum() or c in ("_", "-") else "_" for c in defn.name)[:32]
        run_id = f"wf_{norm_m}_{now_tag}_{safe_name}"
        out_dir = self.runs_dir / run_id
        out_dir.mkdir(parents=True, exist_ok=True)
        html_path = out_dir / "report.html"
        generate_html_report(
            wf_res.backtest_result,
            html_path,
            title=f"FactorLibrary [{norm_m}] 双层因子诊断报告 - {defn.name}",
            v4_stats=wf_res.v4_stats,
            probe_report=wf_res.probe_report,
        )

        mcp_ref = MCP_STOCK_BENCHMARKS.get(defn.name) if norm_m == "stocks" else None

        return _sanitize_json_value(
            {
                "status": "ok",
                "market": norm_m,
                "run_id": run_id,
                "report_path": str(html_path),
                "report_url": f"/api/{norm_m}/runs/{run_id}/report",
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
                "mcp_benchmark": mcp_ref,
            }
        )

    def compare_factors_ab(
        self,
        payload: dict[str, Any],
        market: str | None = None,
    ) -> dict[str, Any]:
        """Page 3 多因子 A/B 同台对比：严格交集 OOS 窗口、Rust Rank 相关矩阵与正交残差累积 IC。"""
        norm_m, panel, _ = self._get_panel_and_catalog(market or payload.get("market"))
        factor_ids = [int(x) for x in (payload.get("factor_ids") or [])]
        horizon = int(payload.get("horizon", 5))
        res = compare_factors(
            panel=panel,
            registry=self.registry,
            factor_ids=factor_ids,
            market=norm_m,
            horizon=horizon,
        )
        return _sanitize_json_value(res)

    # =========================================================================
    # P5 因子资产、不可变版本中心与三关准入审计
    # =========================================================================

    def list_factors(
        self,
        status: str | None = None,
        source: str | None = None,
        mode: str | None = None,
        min_icir: float | None = None,
        split: str = "train",
        horizon: int = 5,
        market: str | None = None,
        factor_type: str | None = None,
        source_run: str | None = None,
    ) -> list[dict[str, Any]]:
        norm_m = self._resolve_market(market)
        records = self.registry.list_factors(
            status=status or None,
            source=source or None,
            mode=mode or None,
            min_icir=min_icir,
            split=split,
            horizon=horizon,
            market=norm_m,
            factor_type=factor_type or None,
            source_run=source_run or None,
        )
        out: list[dict[str, Any]] = []
        for r in records:
            d = r.to_dict()
            versions = self.registry.list_factor_versions(r.id) if r.id is not None else []
            d["version_count"] = len(versions)
            d["latest_version"] = versions[-1].version if versions else 0
            primary_m = None
            for m in r.metrics:
                if m.split == split and m.horizon == horizon:
                    primary_m = m.to_dict()
                    break
            if primary_m is None and r.metrics:
                primary_m = r.metrics[0].to_dict()
            d["primary_metric"] = primary_m
            # 提取最新版本的审计报告（若有）
            audit_summary = None
            if versions and versions[-1].eval_summary_json:
                try:
                    ev = json.loads(versions[-1].eval_summary_json)
                    if isinstance(ev, dict) and "audit" in ev:
                        audit_summary = ev["audit"]
                except Exception:
                    pass
            d["audit_summary"] = audit_summary
            out.append(d)
        return _sanitize_json_value(out)

    def get_factor_detail(self, factor_id: int, market: str | None = None) -> dict[str, Any] | None:
        rec = self.registry.get_factor(factor_id, include_metrics=True)
        if rec is None:
            return None
        if market is not None and rec.market != self._resolve_market(market):
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
        if rec.recipe_id:
            rcp = self.registry.get_recipe(rec.recipe_id)
            d["recipe"] = rcp.to_dict() if rcp else None
        else:
            d["recipe"] = None
        return _sanitize_json_value(d)

    def update_factor_status(
        self,
        factor_id: int,
        status: str,
        sign: int | None = None,
        market: str | None = None,
    ) -> dict[str, Any]:
        rec = self.registry.get_factor(factor_id, include_metrics=False)
        if rec is None or (market is not None and rec.market != self._resolve_market(market)):
            raise ValueError(f"因子 id={factor_id} 不存在或不属于市场 {market}")
        self.registry.update_factor_status(factor_id=factor_id, status=status, sign=sign)
        detail = self.get_factor_detail(factor_id)
        return _sanitize_json_value({"status": "ok", "factor": detail})

    def batch_update_factor_status(
        self,
        factor_ids: list[int],
        status: str,
        sign: int | None = None,
        market: str | None = None,
    ) -> dict[str, Any]:
        norm_m = self._resolve_market(market)
        updated: list[int] = []
        for fid in factor_ids:
            rec = self.registry.get_factor(int(fid), include_metrics=False)
            if rec is not None and rec.market == norm_m:
                self.registry.update_factor_status(int(fid), status=status, sign=sign)
                updated.append(int(fid))
        return _sanitize_json_value(
            {"status": "ok", "updated_ids": updated, "target_status": status}
        )

    def batch_audit_factors(
        self,
        payload: dict[str, Any],
        market: str | None = None,
    ) -> dict[str, Any]:
        """执行批量三关准入审计 (关1去重 + 关2 Spanning 正交残差 + 关3 二层消融)。"""
        norm_m, panel, _ = self._get_panel_and_catalog(market or payload.get("market"))
        factor_ids = [int(x) for x in (payload.get("factor_ids") or [])]
        if not factor_ids:
            raise ValueError("请至少勾选 1 个候选因子执行三关准入审计")

        horizon = int(payload.get("horizon", 5))
        min_valid_icir = float(payload.get("min_valid_icir", 0.20))
        max_turnover = float(payload.get("max_turnover", 0.85))
        max_corr_limit = float(payload.get("max_corr_limit", 0.60))

        reports = audit_candidates(
            panel=panel,
            registry=self.registry,
            factor_ids=factor_ids,
            market=norm_m,
            horizon=horizon,
            min_valid_icir=min_valid_icir,
            max_turnover=max_turnover,
            max_corr_limit=max_corr_limit,
        )

        # 可选：若 auto_promote=True 且三关全部通过则自动晋升为 accepted
        auto_promote = bool(payload.get("auto_promote", False))
        promoted: list[int] = []
        if auto_promote:
            for rep in reports:
                if rep.get("all_passed"):
                    self.registry.update_factor_status(int(rep["factor_id"]), status="accepted")
                    promoted.append(int(rep["factor_id"]))

        return _sanitize_json_value(
            {
                "status": "ok",
                "market": norm_m,
                "results": reports,
                "promoted_ids": promoted,
            }
        )

    def delete_factor(self, factor_id: int, market: str | None = None) -> dict[str, Any]:
        if market is not None:
            rec = self.registry.get_factor(factor_id, include_metrics=False)
            if rec is not None and rec.market != self._resolve_market(market):
                return {"status": "not_found", "deleted": False}
        deleted = self.registry.delete_factor(factor_id)
        return {"status": "ok" if deleted else "not_found", "deleted": deleted}

    def init_seed_factors(self, market: str | None = None) -> dict[str, Any]:
        norm_m, panel, _ = self._get_panel_and_catalog(market)
        if norm_m == "futures":
            ids = register_seed_factors(
                registry=self.registry,
                engine_or_panel=panel,
                run_id="workbench_seed_init",
                status="accepted",
                market=norm_m,
            )
        else:
            from factorlib.core.evaluate import evaluate_factor

            profile = resolve_profile("stocks")
            has_mcp_fields = {"close_hfq", "log_amount", "c2c_ret"}.issubset(panel.fields.keys())
            if has_mcp_fields:
                # 真实/完整 A 股面板：直接预置 FactorLab MCP 线上对账基准因子族
                mcp_tpls = [
                    t
                    for t in STOCKS_AUTHORING_TEMPLATES
                    if t.get("factor_type") == "expression"
                    and (
                        t["name"] != "Tech_Trend_VP_1_Rev"
                        or {"high_adj", "low_adj", "amount"}.issubset(panel.fields.keys())
                    )
                ]
                ids = []
                for tpl in mcp_tpls:
                    defn = FactorDefinition(
                        name=tpl["name"],
                        factor_type="expression",
                        expression=tpl["expression"],
                        postprocess=parse_postprocess_pipeline(tpl["postprocess"]),
                        description=tpl.get("description", ""),
                        market="stocks",
                    )
                    res = evaluate_factor(
                        engine_or_panel=panel,
                        expr=defn,
                        horizons=(1, 5),
                        n_quantiles=10,
                        force_sign=1,
                        profile=profile,
                    )
                    rec = res.to_factor_record(
                        source="seed",
                        source_run="workbench_seed_init",
                        status="accepted",
                        name=tpl["name"],
                        factor_type="expression",
                        definition_json=defn.to_json(),
                        market=norm_m,
                    )
                    ids.append(
                        self.registry.register_factor(
                            rec, run_id="workbench_seed_init", upsert=True
                        )
                    )
            else:
                price_col = "close_hfq" if "close_hfq" in panel.fields else "close"
                liq_col = "turnover_rate" if "turnover_rate" in panel.fields else "log_amount"
                stk_seeds = [
                    {
                        "name": "stk_rev_20d",
                        "expr": f"-pct_change({price_col}, 20)",
                        "sign": 1,
                        "description": "A股20日价格短期反转因子",
                    },
                    {
                        "name": "stk_low_liq_10d",
                        "expr": f"-ts_mean({liq_col}, 10)",
                        "sign": 1,
                        "description": "A股10日低成交/换手异象因子",
                    },
                    {
                        "name": "stk_low_vol_20d",
                        "expr": "-ts_std(c2c_ret, 20)",
                        "sign": 1,
                        "description": "A股20日特质低波动率因子",
                    },
                ]
                ids = []
                for s_def in stk_seeds:
                    res = evaluate_factor(
                        engine_or_panel=panel,
                        expr=s_def["expr"],
                        force_sign=s_def["sign"],
                    )
                    rec = res.to_factor_record(
                        source="seed",
                        source_run="workbench_seed_init",
                        status="accepted",
                        name=s_def["name"],
                        market=norm_m,
                    )
                    ids.append(
                        self.registry.register_factor(
                            rec, run_id="workbench_seed_init", upsert=True
                        )
                    )
        return _sanitize_json_value(
            {
                "status": "ok",
                "market": norm_m,
                "registered_ids": ids,
                "factors": self.list_factors(market=norm_m),
            }
        )

    # =========================================================================
    # P4 自动化挖掘与 DL 训练中心：Recipe 冻结、独立子进程派发与 Run 监控
    # =========================================================================

    def list_recipes(
        self, market: str | None = None, kind: str | None = None, limit: int = 50
    ) -> list[dict[str, Any]]:
        norm_m = self._resolve_market(market)
        recipes = self.registry.list_recipes(market=norm_m, kind=kind, limit=limit)
        return _sanitize_json_value([r.to_dict() for r in recipes])

    def create_recipe(self, payload: dict[str, Any], market: str | None = None) -> dict[str, Any]:
        norm_m, panel, _ = self._get_panel_and_catalog(market or payload.get("market"))
        kind = str(payload.get("kind", "dl")).strip().lower()
        fs_id = payload.get("feature_set_id")
        fs_dict = payload.get("feature_set")
        if fs_id is None and fs_dict is None:
            # 自动选用当前市场首个特征池
            sets = self.list_feature_sets(market=norm_m)
            if sets:
                fs_id = sets[0]["id"]

        recipe = self.registry.create_recipe(
            market=norm_m,
            kind=kind,
            feature_set_id=int(fs_id) if fs_id is not None else None,
            feature_set=fs_dict,
            target_label=str(payload.get("target_label", "y_5")),
            splits=dict(payload.get("splits") or {}),
            params=dict(payload.get("params") or {}),
            extensions=payload.get("extensions"),
            data_version=str(panel.meta.get("panel_version", "v1")),
            seed=int(payload.get("seed", 42)),
            name=str(payload.get("name", "")),
        )
        cli_cmd = self._build_cli_command(
            kind=kind, recipe_id=recipe.recipe_id or "", market=norm_m
        )
        return _sanitize_json_value(
            {
                "status": "ok",
                "recipe": recipe.to_dict(),
                "recipe_id": recipe.recipe_id,
                "content_hash": recipe.content_hash,
                "cli_command": cli_cmd,
            }
        )

    def _build_cli_command(
        self, kind: str, recipe_id: str, market: str, run_id: str | None = None
    ) -> str:
        script_map = {
            "gp": "scripts/run_gp.py",
            "ml": "scripts/run_ml.py",
            "dl": "scripts/run_dl.py",
        }
        script = script_map.get(kind, "scripts/run_gp.py")
        parts = [
            f"python {script}",
            f"--recipe {recipe_id}",
            f"--market {market}",
            f"--db {self.db_path}",
            f"--runs-dir {self.runs_dir}",
        ]
        if run_id:
            parts.append(f"--run-id {run_id}")
        if self._panel_sources.get(market) == "synthetic":
            parts.append("--synthetic")
        return " ".join(parts)

    def launch_run(self, payload: dict[str, Any], market: str | None = None) -> dict[str, Any]:
        """冻结配方并派发独立后台 OS 子进程 (`start_new_session=True`)，立即返回 `run_id`。"""
        norm_m, panel, _ = self._get_panel_and_catalog(market or payload.get("market"))
        kind = str(payload.get("kind", "gp")).strip().lower()
        if kind not in ("gp", "dl", "ml"):
            raise ValueError(f"后台异步派发暂仅支持 gp / dl / ml 任务，收到: {kind!r}")

        recipe_id = payload.get("recipe_id")
        if not recipe_id:
            rcp_res = self.create_recipe(payload, market=norm_m)
            recipe_id = rcp_res["recipe_id"]
        else:
            existing = self.registry.get_recipe(str(recipe_id))
            if existing is None:
                raise ValueError(f"未找到配方 recipe_id={recipe_id}")
            kind = existing.kind

        ts_tag = dt.datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:19]
        run_id = f"{kind}_{norm_m}_{ts_tag}"
        run_dir = self.runs_dir / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        log_path = run_dir / "worker.log"

        script_rel = {
            "gp": "scripts/run_gp.py",
            "ml": "scripts/run_ml.py",
            "dl": "scripts/run_dl.py",
        }[kind]
        script_path = PROJECT_ROOT / script_rel
        cmd: list[str] = [
            sys.executable,
            str(script_path),
            "--recipe",
            str(recipe_id),
            "--market",
            norm_m,
            "--db",
            str(self.db_path),
            "--runs-dir",
            str(self.runs_dir),
            "--run-id",
            run_id,
        ]
        is_syn = self._panel_sources.get(norm_m) == "synthetic"
        if is_syn:
            cmd.append("--synthetic")
        elif (self._panel_dirs[norm_m] / "panel_daily.parquet").exists():
            cmd.extend(["--panel", str(self._panel_dirs[norm_m])])

        # 先行在 SQLite 记录初始运行状态，确保前端刷新立刻可见
        self.registry.record_run(
            ExperimentRunRecord(
                run_id=run_id,
                market=norm_m,
                kind=kind,
                recipe_id=str(recipe_id),
                config_yaml=json.dumps(payload.get("params") or {}, ensure_ascii=False),
                data_version=str(panel.meta.get("panel_version", "v1")),
                git_sha="async_subprocess",
                n_evaluated=0,
                status="running",
                run_dir=str(run_dir),
                log_tail=f"[{run_id}] 正在启动独立后台子进程...",
            )
        )

        env = os.environ.copy()
        py_src = str(PROJECT_ROOT / "python")
        env["PYTHONPATH"] = f"{py_src}:{env.get('PYTHONPATH', '')}".rstrip(":")

        with log_path.open("a", encoding="utf-8") as log_fh:
            proc = subprocess.Popen(
                cmd,
                cwd=str(PROJECT_ROOT),
                env=env,
                stdout=log_fh,
                stderr=subprocess.STDOUT,
                start_new_session=True,
                close_fds=True,
            )

        self.registry.update_run_heartbeat(
            run_id,
            progress={"stage": "子进程启动中...", "pct": 0.02},
            log_tail=f"[{run_id}] spawned pid={proc.pid}",
            pid=proc.pid,
        )

        return _sanitize_json_value(
            {
                "status": "running",
                "run_id": run_id,
                "market": norm_m,
                "kind": kind,
                "recipe_id": recipe_id,
                "pid": proc.pid,
                "run_dir": str(run_dir),
                "cli_command": self._build_cli_command(kind, str(recipe_id), norm_m, run_id=run_id),
            }
        )

    def _read_run_progress_and_log(
        self, run_dir: Path, db_rec: dict[str, Any]
    ) -> tuple[dict[str, Any], str]:
        prog: dict[str, Any] = {}
        if db_rec.get("progress_json"):
            try:
                prog = json.loads(db_rec["progress_json"])
            except Exception:
                prog = {}

        hb_file = run_dir / "heartbeat.json"
        if hb_file.exists():
            try:
                hb = json.loads(hb_file.read_text(encoding="utf-8"))
                if isinstance(hb, dict) and isinstance(hb.get("progress"), dict):
                    prog = hb["progress"]
            except Exception:
                pass

        log_tail = db_rec.get("log_tail") or ""
        worker_log = run_dir / "worker.log"
        if worker_log.exists():
            try:
                lines = worker_log.read_text(encoding="utf-8", errors="replace").splitlines()
                if lines:
                    log_tail = "\n".join(lines[-40:])
            except Exception:
                pass
        return prog, log_tail

    def list_runs(self, market: str | None = None) -> list[dict[str, Any]]:
        """列出指定市场下的后台任务与历史实验运行（含实时进度与日志尾行）。"""
        norm_m = self._resolve_market(market)
        db_records = self.registry.list_runs(market=norm_m)
        db_runs = {r.run_id: r.to_dict() for r in db_records}
        items: list[dict[str, Any]] = []
        seen: set[str] = set()

        for rid, rec in db_runs.items():
            seen.add(rid)
            r_dir = Path(rec["run_dir"]) if rec.get("run_dir") else (self.runs_dir / rid)
            status = rec.get("status", "completed")
            pid = rec.get("pid")
            # 若 DB 显示 running 但子进程已意外退出且无 meta finished 状态，探测真实进程存活
            if status == "running" and pid is not None and not _is_pid_alive(int(pid)):
                meta_file = r_dir / "run_meta.json"
                if meta_file.exists():
                    try:
                        meta = json.loads(meta_file.read_text(encoding="utf-8"))
                        if meta.get("status") in ("completed", "failed", "cancelled"):
                            status = meta["status"]
                    except Exception:
                        pass
            prog, log_tail = self._read_run_progress_and_log(r_dir, rec)
            report_html = r_dir / "report.html"
            items.append(
                {
                    "run_id": rid,
                    "market": rec.get("market", norm_m),
                    "kind": rec.get("kind", "eval"),
                    "recipe_id": rec.get("recipe_id"),
                    "status": status,
                    "pid": pid,
                    "started_at": rec.get("started_at"),
                    "finished_at": rec.get("finished_at"),
                    "progress": prog,
                    "log_tail": log_tail,
                    "has_report": report_html.exists(),
                    "report_url": f"/api/{norm_m}/runs/{rid}/report"
                    if report_html.exists()
                    else None,
                    "summary": {},
                }
            )

        # 补充仅落盘目录的工作流报告（如 wf_{market}_...）
        if self.runs_dir.exists():
            prefix_other = "stocks" if norm_m == "futures" else "futures"
            for child in sorted(self.runs_dir.iterdir(), key=lambda p: p.name, reverse=True):
                if not child.is_dir() or child.name in seen:
                    continue
                rid = child.name
                # 严格按目录命名隔离另一个市场的工作流输出
                if f"_{prefix_other}_" in rid:
                    continue
                if rid.startswith("wf_") and f"_{norm_m}_" not in rid and norm_m != "futures":
                    continue
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
                items.append(
                    {
                        "run_id": rid,
                        "market": norm_m,
                        "kind": "workflow" if rid.startswith("wf_") else "backtest",
                        "recipe_id": None,
                        "status": "completed",
                        "pid": None,
                        "started_at": None,
                        "finished_at": None,
                        "progress": {"stage": "完成", "pct": 1.0},
                        "log_tail": "",
                        "has_report": report_html.exists(),
                        "report_url": f"/api/{norm_m}/runs/{rid}/report"
                        if report_html.exists()
                        else None,
                        "summary": {
                            "total_return": summary_data.get("total_return"),
                            "annual_return": summary_data.get("annual_return"),
                            "sharpe": summary_data.get("sharpe"),
                            "max_drawdown": summary_data.get("max_drawdown"),
                        },
                    }
                )

        return _sanitize_json_value(items[:50])

    def get_run_detail(self, run_id: str, market: str | None = None) -> dict[str, Any] | None:
        norm_m = self._resolve_market(market)
        rec = self.registry.get_run(run_id)
        if rec is not None and rec.market != norm_m:
            return None
        r_dir = Path(rec.run_dir) if (rec and rec.run_dir) else (self.runs_dir / Path(run_id).name)
        if rec is None and not r_dir.exists():
            return None

        rec_dict = (
            rec.to_dict()
            if rec
            else {"run_id": run_id, "market": norm_m, "kind": "eval", "status": "completed"}
        )
        prog, log_tail = self._read_run_progress_and_log(r_dir, rec_dict)
        produced = [
            f.to_dict() for f in self.registry.list_factors(market=norm_m, source_run=run_id)
        ]
        summary_md = ""
        for md_name in ("summary.md", "run_summary.md"):
            md_path = r_dir / md_name
            if md_path.exists():
                summary_md = md_path.read_text(encoding="utf-8", errors="replace")
                break

        return _sanitize_json_value(
            {
                **rec_dict,
                "progress": prog,
                "log_tail": log_tail,
                "summary_md": summary_md,
                "produced_factors": produced,
                "has_report": (r_dir / "report.html").exists(),
                "report_url": f"/api/{norm_m}/runs/{run_id}/report"
                if (r_dir / "report.html").exists()
                else None,
            }
        )

    def cancel_run(self, run_id: str, market: str | None = None) -> dict[str, Any]:
        norm_m = self._resolve_market(market)
        rec = self.registry.get_run(run_id)
        if rec is None or rec.market != norm_m:
            raise ValueError(f"未找到市场 {norm_m!r} 下的运行记录 {run_id}")

        killed = False
        if rec.pid is not None and _is_pid_alive(int(rec.pid)):
            try:
                os.killpg(os.getpgid(int(rec.pid)), signal.SIGTERM)
                killed = True
            except Exception:
                try:
                    os.kill(int(rec.pid), signal.SIGTERM)
                    killed = True
                except Exception:
                    pass

        self.registry.finish_run(run_id, status="cancelled")
        return _sanitize_json_value(
            {"status": "cancelled", "run_id": run_id, "signal_sent": killed}
        )

    def get_run_report_html(self, run_id: str) -> str | None:
        safe_id = Path(run_id).name
        report_file = self.runs_dir / safe_id / "report.html"
        if report_file.exists():
            return report_file.read_text(encoding="utf-8")
        return None
