"""FactorLab MCP 标准变量别名与本地面板字段的双向兼容映射 (core/variable_aliases.py)。

严格对齐 `docs/factorlab_mcp_catalog.md` 第五章「底层基础特征与变量库」的官方变量短名，
在**表达式字符串层**完成别名归一（把 `daily_derived_returns_c2c_1d` 重写为 `c2c_ret` 等），
从而：

1. **零内存放大**：不向面板与 Rust 内核复制别名列（`(T=4000, N=5600)` 单列约 180MB），
   只在 DSL 文本进入解析器前做标识符级替换；
2. **语义严格等价**：仅登记**数值完全同义**的别名（极性反转、口径不同或本地缺失的官方
   变量一律**不登记**，由 `UNMAPPED_MCP_VARIABLES` 显式记录并在体检报告中提示）；
3. **双域隔离**：`rq` 量价基本面域与 `lb_barra` Barra 风险域各自维护独立别名表，
   跨域字段（如 `barra_cne5_size`）仅在对应数据集上生效，杜绝静默 `NaN`。

设计红线：
- 别名解析绝不改写算子名或函数调用（仅替换**已确认是变量**的标识符）；
- 当同名真实字段已存在时，**优先使用真实字段**（别名不覆盖实字段）；
- 解析结果必须可复现、幂等（同一表达式重复解析结果不变）。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

# ==============================================================================
# 1. 米筐量价基本面域 (rq_v1 / rq_v2) 变量别名
# ==============================================================================

RQ_VARIABLE_ALIASES: dict[str, str] = {
    # --- 量价行情 (price) ---
    "daily_kline_open_hfq": "open_adj",
    "daily_kline_high_hfq": "high_adj",
    "daily_kline_low_hfq": "low_adj",
    "daily_kline_adj_factor": "ex_cum_factor",
    "daily_kline_pct_chg": "c2c_ret",
    "daily_kline_real_pre_close": "pre_close",
    "daily_kline_change": "close_change",
    "daily_kline_vol_hfq": "volume",
    # --- 衍生统计 (other) ---
    "daily_derived_returns_c2c_1d": "c2c_ret",
    "daily_derived_returns_intra": "daily_derived_returns_intra",
    # --- 基本面与估值 (fundamental, 仅 extended_fields 面板可用) ---
    "daily_fdm_mkt_cap": "mkt_cap",
    # --- 股票池掩码 (universe) ---
    "BaseUnivMask": "in_universe_mask",
    "LimitMask": "limit_mask",
    "StockListedDays": "stock_listed_days",
    "stock_mask_nonewstockmask": "listed_days",
    # --- 指数成分权重 (industry) ---
    "index_weight_000300_sh": "index_weight_000300_sh",
    "index_weight_000905_sh": "index_weight_000905_sh",
    "index_weight_000852_sh": "index_weight_000852_sh",
}

# ==============================================================================
# 2. lb_barra_v1 Barra 风险域变量别名（含 RiskView 只读挂载的 10 风格）
# ==============================================================================

BARRA_VARIABLE_ALIASES: dict[str, str] = {
    # 10 维 Barra CNE5 同步风格暴露
    "barra_cne5_beta": "barra_beta",
    "barra_cne5_btop": "barra_btop",
    "barra_cne5_earnyild": "barra_earnyild",
    "barra_cne5_growth": "barra_growth",
    "barra_cne5_leverage": "barra_leverage",
    "barra_cne5_liquidty": "barra_liquidty",
    "barra_cne5_liquidity": "barra_liquidty",
    "barra_cne5_momentum": "barra_momentum",
    "barra_cne5_resvol": "barra_resvol",
    "barra_cne5_size": "barra_size",
    "barra_cne5_sizenl": "barra_sizenl",
    "barra_cne5_industry": "barra_industry",
    # 大写下标风格别名（本地 styles.py 中的标准风格键）
    "BETA": "barra_beta",
    "BTOP": "barra_btop",
    "EARNYILD": "barra_earnyild",
    "GROWTH": "barra_growth",
    "LEVERAGE": "barra_leverage",
    "LIQUIDTY": "barra_liquidty",
    "MOMENTUM": "barra_momentum",
    "RESVOL": "barra_resvol",
    "SIZE": "barra_size",
    "SIZENL": "barra_sizenl",
    # 特质收益与市值（Barra 域原生字段）
    "daily_fdm_mkt_cap": "mkt_cap",
}

# ==============================================================================
# 3. 官方变量中无法本地等价映射的清单（口径不同 / 极性反转 / 本地缺失）
# ==============================================================================

UNMAPPED_MCP_VARIABLES: dict[str, str] = {
    "SuspendStock": "官方为「非停牌掩码」(1=可交易)，本地 `is_suspended` 为 1=停牌，极性相反，禁止别名",
    "stock_mask_ststock": "官方为「剔除 ST 掩码」(NaN=剔除)，本地 `is_st` 为 1=ST，极性相反，禁止别名",
    "stock_mask_up_limit": "本地提供 `limit_up` 绝对涨停参考价，与官方布尔掩码不同口径",
    "stock_mask_down_limit": "本地提供 `limit_down` 绝对跌停参考价，与官方布尔掩码不同口径",
    "stock_mask_risk_level": "本地无风险警示等级 (0/1/2/3) 字段",
    "trade_universe_asharebj": "本地无北交所板块掩码",
    "trade_universe_asharecyb": "本地无创业板板块掩码",
    "trade_universe_asharesh": "本地无上交所板块掩码",
    "trade_universe_asharesz": "本地无深交所板块掩码",
    "trade_universe_asharedelistrisk": "本地无退市风险警示池",
    "trade_universe_ewm_amt": "本地可由 `ts_mean(amount, 30)` 表达，不单独登记别名",
    "trade_universe_top1500": "本地无平滑成交额 TOP1500 池",
    "trade_universe_top3000": "本地无平滑成交额 TOP3000 池",
    "daily_fdm_totalshare": "本地 daily_basic 未落盘总股本",
    "daily_fdm_floatshare": "本地 daily_basic 未落盘流通股本",
    "daily_fdm_dv_ttm": "本地 daily_basic 未落盘股息率 TTM",
    "sub_barra_cne5_beta": "子因子族 (25 项) 仅云端提供，本地只托管 10 维合成风格",
    "pre_close": "本地 `pre_close_raw` 为除权参考前收盘价，建议直接使用该本地字段名",
}

# ==============================================================================
# 4. 别名解析器
# ==============================================================================

_IDENT_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


@dataclass(frozen=True)
class VariableAliasReport:
    """变量别名归一化体检报告。"""

    applied: dict[str, str] = field(default_factory=dict)
    skipped_conflict: tuple[str, ...] = ()
    skipped_missing: tuple[str, ...] = ()
    unmapped_used: tuple[str, ...] = ()

    @property
    def applied_count(self) -> int:
        return len(self.applied)

    def to_dict(self) -> dict[str, Any]:
        return {
            "applied": dict(self.applied),
            "applied_count": len(self.applied),
            "skipped_conflict": list(self.skipped_conflict),
            "skipped_missing": list(self.skipped_missing),
            "unmapped_used": list(self.unmapped_used),
        }


def merged_aliases(*alias_maps: dict[str, str] | None) -> dict[str, str]:
    """按顺序合并多张别名表（后者覆盖前者）。"""
    out: dict[str, str] = {}
    for m in alias_maps:
        if m:
            out.update(m)
    return out


def resolve_expression_aliases(
    expr: str,
    available_fields: Any = None,
    *,
    aliases: dict[str, str] | None = None,
) -> tuple[str, VariableAliasReport]:
    """把 DSL 表达式中的 FactorLab MCP 变量别名归一化为本地面板字段名。

    Args:
        expr: 原始 DSL 表达式（前缀算子调用或中缀语法糖）。
        available_fields: 当前面板可用字段集合（可为 ``set`` / ``dict_keys`` / 序列）。
            当为 ``None`` 时不做「目标字段是否存在」校验（宽松模式）。
        aliases: 自定义别名表；缺省时自动按可用字段域推断（见 `default_alias_table`）。

    Returns:
        ``(归一化后的表达式, 体检报告)``。表达式仅在存在可替换标识符时才会变化。
    """
    fields: set[str] | None
    if available_fields is None:
        fields = None
    else:
        fields = {str(f) for f in available_fields}

    table = aliases if aliases is not None else default_alias_table(fields)

    applied: dict[str, str] = {}
    conflicts: list[str] = []
    missing: list[str] = []
    unmapped_used: list[str] = []

    def _replace(match: re.Match[str]) -> str:
        token = match.group(0)
        if token in UNMAPPED_MCP_VARIABLES:
            if token not in unmapped_used:
                unmapped_used.append(token)
            return token
        target = table.get(token)
        if target is None:
            return token
        if token == target:
            # 同名别名（如 index_weight_*）无需替换
            return token
        # 真实字段优先：若面板已存在同名字段，则绝不改写
        if fields is not None and token in fields:
            if token not in conflicts:
                conflicts.append(token)
            return token
        if fields is not None and target not in fields:
            if token not in missing:
                missing.append(token)
            return token
        applied[token] = target
        return target

    resolved = _IDENT_RE.sub(_replace, expr)
    report = VariableAliasReport(
        applied=applied,
        skipped_conflict=tuple(conflicts),
        skipped_missing=tuple(missing),
        unmapped_used=tuple(unmapped_used),
    )
    return resolved, report


def default_alias_table(available_fields: set[str] | None) -> dict[str, str]:
    """按当前面板字段域自动选择别名表（Barra 域与米筐量价域并集，安全幂等）。"""
    table = dict(RQ_VARIABLE_ALIASES)
    table.update(BARRA_VARIABLE_ALIASES)
    # 大写下标风格别名（BETA/SIZE/...）仅在面板无同名实字段时生效，
    # 由 `resolve_expression_aliases` 的「真实字段优先」规则保证。
    if available_fields is not None and not any(k.startswith("barra_") for k in available_fields):
        for key in (
            "BETA",
            "BTOP",
            "EARNYILD",
            "GROWTH",
            "LEVERAGE",
            "LIQUIDTY",
            "MOMENTUM",
            "RESVOL",
            "SIZE",
            "SIZENL",
        ):
            table.pop(key, None)
    return table


def alias_catalog_entries(
    available_fields: Any = None,
    *,
    aliases: dict[str, str] | None = None,
) -> list[dict[str, Any]]:
    """列出可在当前面板上生效的别名条目（供 `FactorCatalog` / Workbench 检索展示）。"""
    fields = {str(f) for f in available_fields} if available_fields is not None else None
    table = aliases if aliases is not None else default_alias_table(fields)
    out: list[dict[str, Any]] = []
    for alias, target in sorted(table.items()):
        if alias == target:
            continue
        if fields is not None and target not in fields:
            continue
        if fields is not None and alias in fields:
            continue
        out.append(
            {
                "name": alias,
                "catalog": "variables",
                "category": "factorlab_alias",
                "role": "dsl",
                "dtype": "float64",
                "resolves_to": target,
                "description": f"FactorLab MCP 变量别名 -> 本地面板字段 `{target}`",
            }
        )
    return out
