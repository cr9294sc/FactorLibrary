"""面板字段规格注册表 (etl/fields.py)。

``panel_daily.parquet`` 的列与 ``fields.json`` 字段规格书均由本注册表生成, 构建时强校验两者一致,
避免文档与数据漂移。

角色 (``role``):

- ``key``: 主键/标识列;
- ``dsl``: 可被表达式 DSL 引用的特征字段, **全部满足 T 日 15:00 收盘 (含盘后发布的日度衍生数据)
  之前已知**;
- ``aux``: 辅助列 (执行价/原始价/品种池/合约规格), 供回测与品种池使用, **不暴露给 DSL**。

预测标签位于独立的 ``labels.parquet``, 不在本注册表中。
"""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class FieldSpec:
    name: str
    dtype: str  # date | str | bool | int64 | float64
    role: str  # key | dsl | aux
    group: str
    source: str
    description: str

    def to_dict(self) -> dict[str, str]:
        return asdict(self)


def _f(name: str, dtype: str, role: str, group: str, source: str, description: str) -> FieldSpec:
    return FieldSpec(name, dtype, role, group, source, description)


PANEL_FIELDS: tuple[FieldSpec, ...] = (
    # ---- 主键 ----
    _f("trading_date", "date", "key", "key", "calendar", "米筐交易日 (夜盘归属下一交易日)"),
    _f("symbol", "str", "key", "key", "dominant_mapping", "品种代码 (underlying_symbol)"),
    _f(
        "contract",
        "str",
        "key",
        "key",
        "dominant_mapping",
        "T 日主力合约 (rule=3, rank=1), 日内不换月",
    ),
    # ---- 元信息 / 换月 ----
    _f(
        "prev_contract",
        "str",
        "aux",
        "meta",
        "dominant_mapping",
        "前一有效行的主力合约; 换月日 != contract",
    ),
    _f(
        "is_roll",
        "bool",
        "aux",
        "meta",
        "dominant_mapping",
        "T 日是否发生主力换月 (contract != prev_contract)",
    ),
    _f("date_idx", "int64", "aux", "meta", "calendar", "交易日历连续整数索引 (回看/对齐用)"),
    _f("sector", "str", "aux", "meta", "industry_mapping", "一级板块 (group 中性化用)"),
    # ---- 复权后价格 (DSL) ----
    _f(
        "open",
        "float64",
        "dsl",
        "price_adj",
        "bar_1d+adjust",
        "比例复权开盘价 = open_raw * adj_factor",
    ),
    _f("high", "float64", "dsl", "price_adj", "bar_1d+adjust", "比例复权最高价"),
    _f("low", "float64", "dsl", "price_adj", "bar_1d+adjust", "比例复权最低价"),
    _f("close", "float64", "dsl", "price_adj", "bar_1d+adjust", "比例复权收盘价"),
    _f("settle", "float64", "dsl", "price_adj", "bar_1d+adjust", "比例复权结算价"),
    _f(
        "vwap",
        "float64",
        "dsl",
        "price_adj",
        "bar_1m+adjust",
        "比例复权全天 VWAP = sum(close*volume)/sum(volume) * adj_factor",
    ),
    # ---- 量仓 (DSL) ----
    _f("volume", "float64", "dsl", "volume", "bar_1d", "主力合约成交量 (手)"),
    _f(
        "turnover",
        "float64",
        "dsl",
        "volume",
        "bar_1d",
        "主力合约成交额 (元); 异常置零时以 volume*settle*multiplier 兜底",
    ),
    _f(
        "open_interest",
        "float64",
        "dsl",
        "volume",
        "bar_1d",
        "主力合约持仓量 (手, 原始量级, 跨换月不可直接比较)",
    ),
    _f(
        "oi_chg",
        "float64",
        "dsl",
        "volume",
        "bar_1d",
        "持仓量日变化 = OI_T(c_T) - OI_{T-1}(c_T), 同合约作差, 换月日首日为 null",
    ),
    # ---- 日内微观结构 (DSL) ----
    _f(
        "rv",
        "float64",
        "dsl",
        "microstructure",
        "bar_1m",
        "已实现波动率 sqrt(sum r_m^2), 仅连续交易段内收益",
    ),
    _f("rv_up", "float64", "dsl", "microstructure", "bar_1m", "上行已实现波动率"),
    _f("rv_down", "float64", "dsl", "microstructure", "bar_1m", "下行已实现波动率"),
    _f(
        "ret_last30",
        "float64",
        "dsl",
        "microstructure",
        "bar_1m",
        "尾盘 30 分钟动量 p_M/p_{M-30}-1",
    ),
    _f("ret_first30", "float64", "dsl", "microstructure", "bar_1m", "开盘 30 分钟收益 p_30/p_1-1"),
    _f(
        "spread_night_day",
        "float64",
        "dsl",
        "microstructure",
        "bar_1m",
        "夜盘收益-日盘收益, 无夜盘品种为 null",
    ),
    _f("vol_conc", "float64", "dsl", "microstructure", "bar_1m", "日内成交量集中度 Herfindahl"),
    _f(
        "amp",
        "float64",
        "dsl",
        "microstructure",
        "bar_1m",
        "日内振幅 (max high - min low)/前结算价",
    ),
    _f("skew_intra", "float64", "dsl", "microstructure", "bar_1m", "日内收益偏度"),
    _f("kurt_intra", "float64", "dsl", "microstructure", "bar_1m", "日内收益超额峰度"),
    _f("vwap_dev", "float64", "dsl", "microstructure", "bar_1m", "收盘价对日内 VWAP 偏离度"),
    # ---- 期限结构 (DSL) ----
    _f(
        "ry_main_sub",
        "float64",
        "dsl",
        "term_structure",
        "roll_yield",
        "主力-次主力展期收益率 (settle(main)-settle(sub))/settle(sub)",
    ),
    _f(
        "ry_ann",
        "float64",
        "dsl",
        "term_structure",
        "roll_yield",
        "主力-次主力年化展期收益率 (252/交易日差)",
    ),
    _f("ry_near_main", "float64", "dsl", "term_structure", "roll_yield", "近月-主力展期收益率"),
    # ---- 仓单 (DSL) ----
    _f(
        "wh_chg5",
        "float64",
        "dsl",
        "inventory",
        "warehouse_stocks",
        "仓单 5 日变动率 (品种内标准化)",
    ),
    _f("wh_z60", "float64", "dsl", "inventory", "warehouse_stocks", "仓单 60 日滚动时序 Z-Score"),
    # ---- 会员持仓 (DSL) ----
    _f(
        "mr_net_ratio",
        "float64",
        "dsl",
        "positioning",
        "member_rank",
        "会员多空前 20 净多头占比 (L-S)/(L+S)",
    ),
    _f(
        "mr_cr20",
        "float64",
        "dsl",
        "positioning",
        "member_rank",
        "会员前 20 持仓集中度 (L+S)/OI, 取值约 [0,2]",
    ),
    _f(
        "mr_net_ratio_chg5", "float64", "dsl", "positioning", "member_rank", "净多头占比 5 日变动量"
    ),
    # ---- 原始价 / 涨跌停 (aux) ----
    _f("open_raw", "float64", "aux", "raw", "bar_1d", "未复权开盘价 (合约实际价)"),
    _f("high_raw", "float64", "aux", "raw", "bar_1d", "未复权最高价"),
    _f("low_raw", "float64", "aux", "raw", "bar_1d", "未复权最低价"),
    _f("close_raw", "float64", "aux", "raw", "bar_1d", "未复权收盘价"),
    _f("settle_raw", "float64", "aux", "raw", "bar_1d", "未复权结算价 (逐日盯市用)"),
    _f("prev_settle_raw", "float64", "aux", "raw", "bar_1d", "未复权前结算价"),
    _f("limit_up", "float64", "aux", "raw", "bar_1d", "涨停价 (原始)"),
    _f("limit_down", "float64", "aux", "raw", "bar_1d", "跌停价 (原始)"),
    _f("is_unilateral", "bool", "aux", "raw", "bar_1d", "是否单边市 (不可开平仓)"),
    _f(
        "adj_factor",
        "float64",
        "aux",
        "raw",
        "adjust",
        "因果复权因子 F(T)=prod_{s<=T} k_s, 每品种首日为 1",
    ),
    # ---- 执行价 (aux) ----
    _f(
        "exec_vwap",
        "float64",
        "aux",
        "exec",
        "bar_1m",
        "P_exec(c_T, T): 合约 c_T 在 T 日开盘 30m 的 sum(close*vol)/sum(vol) (原始价)",
    ),
    _f(
        "exec_vwap_old",
        "float64",
        "aux",
        "exec",
        "bar_1m",
        "P_exec(c_{T-1}, T): 旧合约在 T 日开盘 30m VWAP, 换月日平旧仓价; 非换月日等于 exec_vwap",
    ),
    _f("exec_volume", "float64", "aux", "exec", "bar_1m", "开盘 30m 窗口成交量 (手)"),
    # ---- 合约规格 (aux) ----
    _f("multiplier", "float64", "aux", "spec", "instruments_info", "合约乘数 (静态快照)"),
    _f(
        "margin_rate",
        "float64",
        "aux",
        "spec",
        "instruments_info",
        "基础保证金率 (静态快照, 非动态历史)",
    ),
    _f("days_to_expiry", "int64", "aux", "spec", "instruments_info", "T 日距主力合约到期日历天数"),
    # ---- 品种池 (aux) ----
    _f("traded", "bool", "aux", "universe", "bar_1d", "T 日主力合约有成交 (volume>0)"),
    _f("adv20_turnover", "float64", "aux", "universe", "bar_1d", "含 T 日的 20 日平均成交额"),
    _f(
        "listed_days",
        "int64",
        "aux",
        "universe",
        "instruments_info",
        "自品种最早合约上市日起的交易日数",
    ),
    _f("in_universe", "bool", "aux", "universe", "universe.yaml", "T 日是否在可交易品种池内"),
)

PANEL_COLUMNS: list[str] = [f.name for f in PANEL_FIELDS]
DSL_FIELDS: list[str] = [f.name for f in PANEL_FIELDS if f.role == "dsl"]

#: 标签列的规格 (写入 fields.json 的 labels 段, 仅文档用途)
LABEL_DESCRIPTIONS: dict[str, str] = {
    "r_exec_next": "r_{T->T+1}: 持有 c_T 从 T 日到 T+1 日的 VWAP-to-VWAP 收益 (链式构件)",
    "y_{h}": "y_h(T)=prod_{k=1..h}(1+r_{T+k->T+k+1})-1: 信号 T 日收盘, T+1 开盘 30m VWAP 建仓, 持有 h 日的真实累计收益",
}

_POLARS_DTYPES = {
    "date": "Date",
    "str": "String",
    "bool": "Boolean",
    "int64": "Int64",
    "float64": "Float64",
}


def polars_dtype_name(spec: FieldSpec) -> str:
    return _POLARS_DTYPES[spec.dtype]
