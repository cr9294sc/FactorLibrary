"""统一因子研究目录与上下文发现引擎 (core/catalog.py)。

对标 FactorLab MCP 工作流的 Stage 0 & Stage 1 (`get_context` + `catalog_stats` / `catalog_search` / `catalog_info`)，
将期货量化平台的数据字段 (`variables`)、Rust 算子原语 (`functions`)、12 个声明式后处理算子 (`postprocess`)
以及 6 大期货标准风格因子 (`styles`) 统一纳管并提供检索接口。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import factorlib_core

from factorlib.core.styles import STYLE_DEFINITIONS

# 12 个声明式后处理算子规格字典 (与 rust/factorlib_core/src/ops/postprocess.rs 严格对齐)
POSTPROCESS_SPECS: list[dict[str, Any]] = [
    {
        "name": "baseUniMask",
        "catalog": "postprocess",
        "category": "mask",
        "stage_order": 1,
        "formula": "out[t, i] = x1[t, i] * in_universe_mask[t, i]",
        "default_args": {},
        "implicit_dependencies": ["in_universe_mask"],
        "lookback": 1,
        "description": "基础品种池准入掩码：非品种池内（in_universe=False）置为 NaN",
    },
    {
        "name": "trdMask",
        "catalog": "postprocess",
        "category": "mask",
        "stage_order": 1,
        "formula": "out[t, i] = x1[t, i] * limit_mask[t, i]",
        "default_args": {},
        "implicit_dependencies": ["limit_mask"],
        "lookback": 1,
        "description": "可交易状态掩码：停牌、无成交或单边涨跌停（is_unilateral=True）置为 NaN",
    },
    {
        "name": "dropNewStock",
        "catalog": "postprocess",
        "category": "mask",
        "stage_order": 1,
        "formula": "out[t, i] = NaN if listed_days[t, i] < min_days else x1[t, i]",
        "default_args": {"window": 60},
        "implicit_dependencies": ["listed_days"],
        "lookback": 1,
        "description": "次新品种剔除：上市交易日不足 window 天（默认 60 日）的品种置为 NaN",
    },
    {
        "name": "MAD",
        "catalog": "postprocess",
        "category": "outlier",
        "stage_order": 2,
        "formula": "clamp(x1, med - p * 1.4826 * mad, med + p * 1.4826 * mad)",
        "default_args": {"param": 3.0},
        "implicit_dependencies": [],
        "lookback": 1,
        "description": "截面绝对中位差去极值（采用下中位数索引 (k-1)//2 与 1.4826 正态一致因子）",
    },
    {
        "name": "winsor",
        "catalog": "postprocess",
        "category": "outlier",
        "stage_order": 2,
        "formula": "clamp(x1, sorted[floor(p*k)], sorted[k - 1 - floor(p*k)])",
        "default_args": {"param": 0.025},
        "implicit_dependencies": [],
        "lookback": 1,
        "description": "截面分位数缩尾去极值（双侧对称截断比例 param，默认 0.025）",
    },
    {
        "name": "indNeut",
        "catalog": "postprocess",
        "category": "neutralize",
        "stage_order": 3,
        "formula": "out[t, i] = x1[t, i] - mean_{j in sector(i)}(x1[t, j])",
        "default_args": {"group_by": "sector"},
        "implicit_dependencies": ["sector"],
        "lookback": 1,
        "description": "期货板块（黑色/有色/能化/农产品/贵金属/股指/国债）内去均值中性化",
    },
    {
        "name": "riskNeut",
        "catalog": "postprocess",
        "category": "neutralize",
        "stage_order": 3,
        "formula": "out = x1 - (cov(x1, x2) / var(x2)) * (x2 - mean(x2))",
        "default_args": {"field": "SIZE"},
        "implicit_dependencies": ["SIZE"],
        "lookback": 20,
        "description": "单风格/单风险变量截面一元线性回归残差中性化（中心化斜率投影，保留截面均值）",
    },
    {
        "name": "riskNeutMulti",
        "catalog": "postprocess",
        "category": "neutralize",
        "stage_order": 3,
        "formula": "out = y - sum_m beta_m * X_m (via (X^T X) beta = X^T y)",
        "default_args": {"fields": ["SIZE", "VOLATILITY", "CARRY"]},
        "implicit_dependencies": ["SIZE", "VOLATILITY", "CARRY"],
        "lookback": 20,
        "description": "多风格因子截面多元无截距 OLS 正交残差中性化（高斯列主元消元求解）",
    },
    {
        "name": "zScore",
        "catalog": "postprocess",
        "category": "normalize",
        "stage_order": 4,
        "formula": "out[t, i] = (x1[t, i] - mean(x1[t])) / std_pop(x1[t], ddof=0)",
        "default_args": {},
        "implicit_dependencies": [],
        "lookback": 1,
        "description": "截面总体标准差 Z-Score 标准化（ddof=0）",
    },
    {
        "name": "rank",
        "catalog": "postprocess",
        "category": "normalize",
        "stage_order": 4,
        "formula": "round_ties_even((avg_rank - 0.5) / k, 6)",
        "default_args": {},
        "implicit_dependencies": [],
        "lookback": 1,
        "description": "截面均匀分布百分位秩变换 (0, 1)，并列取平均秩并保留 6 位银行家舍入",
    },
    {
        "name": "decay",
        "catalog": "postprocess",
        "category": "smooth",
        "stage_order": 5,
        "formula": "out[t, i] = mean(x1[max(0, t-d+1)..t, i])",
        "default_args": {"window": 5},
        "implicit_dependencies": [],
        "lookback": 5,
        "description": "时序等权滑动平均平滑（忽略窗口内 NaN，严格因果）",
    },
    {
        "name": "emadecay",
        "catalog": "postprocess",
        "category": "smooth",
        "stage_order": 5,
        "formula": "alpha = 2/(d+1); s_k = alpha*v_k + (1-alpha)*s_{k-1}",
        "default_args": {"window": 5},
        "implicit_dependencies": [],
        "lookback": 5,
        "description": "时序窗口内指数移动平均平滑（每时刻独立回看窗口 d 内有效观测递推）",
    },
]

# 常用标准后处理预设链
RECOMMENDED_POSTPROCESS_PRESETS: dict[str, list[dict[str, Any]]] = {
    "standard": [
        {"op": "baseUniMask"},
        {"op": "trdMask"},
        {"op": "MAD", "param": 3.0},
        {"op": "indNeut"},
        {"op": "zScore"},
    ],
    "minimal": [
        {"op": "baseUniMask"},
        {"op": "trdMask"},
        {"op": "zScore"},
    ],
    "style_neutral": [
        {"op": "baseUniMask"},
        {"op": "trdMask"},
        {"op": "MAD", "param": 3.0},
        {"op": "indNeut"},
        {"op": "riskNeutMulti", "fields": ["SIZE", "VOLATILITY", "CARRY"]},
        {"op": "zScore"},
    ],
}

# 内置默认 DSL 字段兜底清单 (当未挂载 data/derived/futures/v1/fields.json 时使用)
_FALLBACK_FIELDS: list[dict[str, Any]] = [
    {
        "name": "open",
        "dtype": "float64",
        "role": "dsl",
        "group": "price_adj",
        "description": "比例复权开盘价",
    },
    {
        "name": "high",
        "dtype": "float64",
        "role": "dsl",
        "group": "price_adj",
        "description": "比例复权最高价",
    },
    {
        "name": "low",
        "dtype": "float64",
        "role": "dsl",
        "group": "price_adj",
        "description": "比例复权最低价",
    },
    {
        "name": "close",
        "dtype": "float64",
        "role": "dsl",
        "group": "price_adj",
        "description": "比例复权收盘价",
    },
    {
        "name": "settle",
        "dtype": "float64",
        "role": "dsl",
        "group": "price_adj",
        "description": "比例复权结算价",
    },
    {
        "name": "vwap",
        "dtype": "float64",
        "role": "dsl",
        "group": "price_adj",
        "description": "比例复权全天 VWAP",
    },
    {
        "name": "volume",
        "dtype": "float64",
        "role": "dsl",
        "group": "volume",
        "description": "主力合约成交量 (手)",
    },
    {
        "name": "turnover",
        "dtype": "float64",
        "role": "dsl",
        "group": "volume",
        "description": "主力合约成交额 (元)",
    },
    {
        "name": "open_interest",
        "dtype": "float64",
        "role": "dsl",
        "group": "volume",
        "description": "主力合约持仓量 (手)",
    },
    {
        "name": "oi_chg",
        "dtype": "float64",
        "role": "dsl",
        "group": "volume",
        "description": "同合约持仓量日变化",
    },
    {
        "name": "rv",
        "dtype": "float64",
        "role": "dsl",
        "group": "microstructure",
        "description": "日内已实现波动率",
    },
    {
        "name": "rv_up",
        "dtype": "float64",
        "role": "dsl",
        "group": "microstructure",
        "description": "上行已实现波动率",
    },
    {
        "name": "rv_down",
        "dtype": "float64",
        "role": "dsl",
        "group": "microstructure",
        "description": "下行已实现波动率",
    },
    {
        "name": "ret_last30",
        "dtype": "float64",
        "role": "dsl",
        "group": "microstructure",
        "description": "尾盘 30 分钟动量",
    },
    {
        "name": "ret_first30",
        "dtype": "float64",
        "role": "dsl",
        "group": "microstructure",
        "description": "开盘 30 分钟收益",
    },
    {
        "name": "spread_night_day",
        "dtype": "float64",
        "role": "dsl",
        "group": "microstructure",
        "description": "夜盘减日盘收益差",
    },
    {
        "name": "vol_conc",
        "dtype": "float64",
        "role": "dsl",
        "group": "microstructure",
        "description": "日内成交量集中度",
    },
    {
        "name": "amp",
        "dtype": "float64",
        "role": "dsl",
        "group": "microstructure",
        "description": "日内高低振幅",
    },
    {
        "name": "skew_intra",
        "dtype": "float64",
        "role": "dsl",
        "group": "microstructure",
        "description": "日内分钟收益偏度",
    },
    {
        "name": "kurt_intra",
        "dtype": "float64",
        "role": "dsl",
        "group": "microstructure",
        "description": "日内分钟收益超额峰度",
    },
    {
        "name": "vwap_dev",
        "dtype": "float64",
        "role": "dsl",
        "group": "microstructure",
        "description": "收盘价对 VWAP 偏离度",
    },
    {
        "name": "ry_main_sub",
        "dtype": "float64",
        "role": "dsl",
        "group": "term_structure",
        "description": "主力-次主力展期收益率",
    },
    {
        "name": "ry_ann",
        "dtype": "float64",
        "role": "dsl",
        "group": "term_structure",
        "description": "主力-次主力年化展期收益率",
    },
    {
        "name": "ry_near_main",
        "dtype": "float64",
        "role": "dsl",
        "group": "term_structure",
        "description": "近月-主力展期收益率",
    },
    {
        "name": "wh_chg5",
        "dtype": "float64",
        "role": "dsl",
        "group": "inventory",
        "description": "注册仓单 5 日变动率",
    },
    {
        "name": "wh_z60",
        "dtype": "float64",
        "role": "dsl",
        "group": "inventory",
        "description": "注册仓单 60 日滚动 Z-Score",
    },
    {
        "name": "mr_net_ratio",
        "dtype": "float64",
        "role": "dsl",
        "group": "positioning",
        "description": "前 20 会员净多头占比",
    },
    {
        "name": "mr_cr20",
        "dtype": "float64",
        "role": "dsl",
        "group": "positioning",
        "description": "前 20 会员持仓集中度",
    },
    {
        "name": "mr_net_ratio_chg5",
        "dtype": "float64",
        "role": "dsl",
        "group": "positioning",
        "description": "净多头占比 5 日变化",
    },
    {
        "name": "sector",
        "dtype": "float64",
        "role": "aux",
        "group": "meta",
        "description": "一级板块整数编码 (1..K)",
    },
    {
        "name": "listed_days",
        "dtype": "float64",
        "role": "aux",
        "group": "universe",
        "description": "品种上市交易日数",
    },
    {
        "name": "in_universe_mask",
        "dtype": "float64",
        "role": "aux",
        "group": "universe",
        "description": "品种池准入掩码 (1.0 / NaN)",
    },
    {
        "name": "limit_mask",
        "dtype": "float64",
        "role": "aux",
        "group": "universe",
        "description": "可交易状态掩码 (1.0 / NaN)",
    },
]


class FactorCatalog:
    """统一管理因子开发所需的字段、算子、后处理步骤与风格字典。"""

    def __init__(self, panel_dir: str | Path = "data/derived/futures/v1") -> None:
        self.panel_dir = Path(panel_dir)
        self.variables: list[dict[str, Any]] = self._load_variables()
        self.functions: list[dict[str, Any]] = self._load_functions()
        self.postprocess: list[dict[str, Any]] = list(POSTPROCESS_SPECS)
        self.styles: list[dict[str, Any]] = [{**s, "catalog": "styles"} for s in STYLE_DEFINITIONS]

    def _load_variables(self) -> list[dict[str, Any]]:
        fields_file = self.panel_dir / "fields.json"
        raw_fields: list[dict[str, Any]] = []
        if fields_file.exists():
            try:
                data = json.loads(fields_file.read_text(encoding="utf-8"))
                raw_fields = data.get("fields", [])
            except Exception:
                raw_fields = list(_FALLBACK_FIELDS)
        else:
            raw_fields = list(_FALLBACK_FIELDS)

        out: list[dict[str, Any]] = []
        seen: set[str] = set()
        for f in raw_fields:
            name = str(f.get("name", ""))
            if not name or name in seen:
                continue
            seen.add(name)
            out.append(
                {
                    "name": name,
                    "catalog": "variables",
                    "category": str(f.get("group", f.get("role", "dsl"))),
                    "role": str(f.get("role", "dsl")),
                    "dtype": str(f.get("dtype", "float64")),
                    "description": str(f.get("description", "")),
                }
            )

        # 确保隐式掩码字段在目录中可见
        for extra in [
            {
                "name": "in_universe_mask",
                "catalog": "variables",
                "category": "universe",
                "role": "aux",
                "dtype": "float64",
                "description": "品种池准入掩码 (1.0 有效, NaN 剔除)，供 baseUniMask 使用",
            },
            {
                "name": "limit_mask",
                "catalog": "variables",
                "category": "universe",
                "role": "aux",
                "dtype": "float64",
                "description": "可交易掩码 (1.0 正常, NaN 停牌或单边涨跌停)，供 trdMask 使用",
            },
        ]:
            if extra["name"] not in seen:
                seen.add(extra["name"])
                out.append(extra)

        return out

    def _load_functions(self) -> list[dict[str, Any]]:
        ops_raw = factorlib_core.list_operators()
        out: list[dict[str, Any]] = []

        if isinstance(ops_raw, list):
            for item in ops_raw:
                op = str(item.get("name", ""))
                cat = str(item.get("category", "unary"))
                arg_types = item.get("arg_types", ["Series"])
                sig = f"{op}({', '.join(arg_types)})"
                out.append(
                    {
                        "name": op,
                        "catalog": "functions",
                        "category": cat,
                        "arity": sig,
                        "description": f"Rust 内核向量化算子 {sig}",
                    }
                )
        elif isinstance(ops_raw, dict):
            for op in ops_raw.get("unary", []):
                out.append(
                    {
                        "name": op,
                        "catalog": "functions",
                        "category": "cross_sectional" if op.startswith("cs_") else "unary",
                        "arity": "unary(x)",
                        "description": f"一元算子 {op}(x)",
                    }
                )

        # 补充 FactorLab 兼容别名说明
        aliases = [
            ("log", "unary", "log(x)", "有符号安全对数别名 -> slog(x)"),
            ("sqrt", "unary", "sqrt(x)", "有符号安全平方根别名 -> ssqrt(x)"),
            ("ts_delay", "time_series", "ts_delay(x, d)", "滞后算子别名 -> delay(x, d)"),
            ("ts_delta", "time_series", "ts_delta(x, d)", "差分算子别名 -> delta(x, d)"),
            ("decay", "time_series", "decay(x, d)", "等权移动平均别名 -> ts_mean(x, d)"),
            (
                "emadecay",
                "time_series",
                "emadecay(x, d)",
                "线性衰减加权别名 -> ts_decay_linear(x, d)",
            ),
        ]
        existing = {item["name"] for item in out}
        for name, cat, sig, desc in aliases:
            if name not in existing:
                out.append(
                    {
                        "name": name,
                        "catalog": "functions",
                        "category": cat,
                        "arity": sig,
                        "description": desc,
                    }
                )
        return out

    def get_context(self) -> dict[str, Any]:
        """Stage 0 环境初始化与上下文发现接口。"""
        return {
            "platform": "FactorLibrary Futures Engine (Rust + Python)",
            "panel_dir": str(self.panel_dir),
            "counts": {
                "variables": len(self.variables),
                "functions": len(self.functions),
                "postprocess": len(self.postprocess),
                "styles": len(self.styles),
            },
            "authoring_modes": ["expression", "code"],
            "postprocess_presets": RECOMMENDED_POSTPROCESS_PRESETS,
            "styles": [s["name"] for s in self.styles],
            "workflow_stages": [
                "Stage 0: get_context (环境与算子目录发现)",
                "Stage 1: catalog_search / catalog_info (字段、算子与后处理字典检索)",
                "Stage 2: FactorDefinition (DSL 表达式或 Python compute(context) + 12 算子后处理链)",
                "Stage 3: probe_factor (静态语法校验、隐式依赖提取与 4 级预热窗口推导)",
                "Stage 4: evaluate_factor_definition & Materialization (Rust 极速求值与物化缓存)",
                "Stage 5: run_factor_workflow (Layer 1 统计分布归因 + Layer 2 期货实盘逐日盯市回测)",
            ],
        }

    def stats(self, catalog: str = "all") -> dict[str, Any]:
        """Stage 1 统计各目录分类下的条目数。"""
        items = self._select_catalog(catalog)
        by_catalog: dict[str, int] = {}
        by_category: dict[str, int] = {}
        for it in items:
            c_name = it.get("catalog", "unknown")
            cat = it.get("category", "unknown")
            by_catalog[c_name] = by_catalog.get(c_name, 0) + 1
            by_category[f"{c_name}:{cat}"] = by_category.get(f"{c_name}:{cat}", 0) + 1
        return {
            "total": len(items),
            "by_catalog": by_catalog,
            "by_category": by_category,
        }

    def search(
        self,
        query: str = "",
        catalog: str = "all",
        category: str = "",
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        """Stage 1 模糊检索变量、算子、后处理步骤或风格因子。"""
        items = self._select_catalog(catalog)
        q = query.strip().lower()
        cat_filter = category.strip().lower()

        matched: list[dict[str, Any]] = []
        for it in items:
            if cat_filter and it.get("category", "").lower() != cat_filter:
                continue
            if q:
                haystack = (
                    f"{it.get('name', '')} {it.get('category', '')} "
                    f"{it.get('description', '')} {it.get('formula', '')}"
                ).lower()
                if q not in haystack:
                    continue
            matched.append(it)
            if len(matched) >= limit:
                break
        return matched

    def get_entry(self, name: str, catalog: str = "all") -> dict[str, Any] | None:
        """Stage 1 精确查询单个字段、算子或后处理条目的详细规格。"""
        items = self._select_catalog(catalog)
        for it in items:
            if it.get("name") == name:
                return it
        # 大小写不敏感二次匹配
        target = name.lower()
        for it in items:
            if str(it.get("name", "")).lower() == target:
                return it
        return None

    def _select_catalog(self, catalog: str) -> list[dict[str, Any]]:
        c = catalog.strip().lower()
        if c in ("", "all"):
            return self.variables + self.functions + self.postprocess + self.styles
        if c in ("var", "vars", "variable", "variables", "field", "fields"):
            return self.variables
        if c in ("func", "funcs", "function", "functions", "op", "ops"):
            return self.functions
        if c in ("pp", "post", "postprocess"):
            return self.postprocess
        if c in ("style", "styles", "risk"):
            return self.styles
        raise ValueError(
            f"未知的 catalog 类型: {catalog!r}，可选: all | variables | functions | postprocess | styles"
        )
