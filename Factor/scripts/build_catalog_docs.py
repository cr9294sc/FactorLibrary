"""Generate comprehensive, structured Markdown documentation for FactorLab catalog."""

import os
import json
from collections import defaultdict

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_FILE = os.path.join(BASE_DIR, "data", "catalog_full.json")
DOCS_DIR = os.path.join(BASE_DIR, "docs", "catalog")


def load_data():
    with open(DATA_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def clean_table_cell(text, max_len=None):
    if not text:
        return "-"
    cleaned = text.replace("\r\n", " ").replace("\n", " ").replace("|", "\\|").strip()
    if max_len and len(cleaned) > max_len:
        return cleaned[:max_len] + "..."
    return cleaned


def format_param_table(params):
    if not params:
        return "*(无参数)*"
    rows = ["| 参数名 | 类型 | 必填 | 默认值 | 说明 |", "| :--- | :--- | :---: | :---: | :--- |"]
    for p in params:
        name = p.get("name", "")
        kind = p.get("kind", "")
        req = "是" if p.get("required") else "否"
        default = str(p.get("default")) if p.get("default") is not None else "-"
        desc = clean_table_cell(p.get("description", ""))
        rows.append(f"| `{name}` | `{kind}` | {req} | `{default}` | {desc} |")
    return "\n".join(rows)


def build_variable_docs(variables):
    os.makedirs(os.path.join(DOCS_DIR, "variables"), exist_ok=True)

    # Grouping variables
    price_vars = [v for v in variables if v.get("category") == "price"]
    minute_vars = [v for v in variables if v.get("category") == "minute"]
    daily_intraday_vars = [v for v in variables if v.get("category") == "daily_intraday"]
    universe_vars = [v for v in variables if v.get("category") == "universe"]
    industry_vars = [v for v in variables if v.get("category") == "industry"]
    
    other_vars = [v for v in variables if v.get("category") == "other"]
    barra_vars = [v for v in other_vars if v.get("short_name", "").startswith("barra_")]
    sub_sw_vars = [v for v in other_vars if v.get("short_name", "").startswith("sub_") or v.get("short_name", "").startswith("sw_")]
    balance_vars = [v for v in other_vars if v.get("short_name", "").startswith("f_1d_balance_sheet")]
    income_vars = [v for v in other_vars if v.get("short_name", "").startswith("f_1d_income")]
    cashflow_vars = [v for v in other_vars if v.get("short_name", "").startswith("f_1d_cash_flow")]
    daily_derived_vars = [v for v in other_vars if v.get("short_name", "").startswith("daily")]

    # 1. 01_price_kline.md
    render_var_page(
        os.path.join(DOCS_DIR, "variables", "01_price_kline.md"),
        title="日线行情变量 (Price Kline)",
        desc="包含 A 股市场基础日线量价指标，包括未复权及后复权（hfq）开高低收、成交量、成交额以及复权因子等。",
        items=price_vars,
    )

    # 2. 02_minute_highfreq.md
    render_var_page(
        os.path.join(DOCS_DIR, "variables", "02_minute_highfreq.md"),
        title="高频分钟行情变量 (Minute High-Frequency)",
        desc="包含 1 分钟（1m）与 5 分钟（5m）切片高频数据字段，包括分钟 K 线、成交分布、加权均价等共 162 个变量。",
        items=minute_vars,
        group_by="frequency",
    )

    # 3. 03_daily_intraday.md
    render_var_page(
        os.path.join(DOCS_DIR, "variables", "03_daily_intraday.md"),
        title="日内衍生特征变量 (Daily Intraday)",
        desc="基于日内微观结构与分时量价计算生成的日频衍生统计特征（如 IntraVwap、Twap、早盘/尾盘分时指标等共 54 个）。",
        items=daily_intraday_vars,
    )

    # 4. 04_universe_and_masks.md
    render_var_page(
        os.path.join(DOCS_DIR, "variables", "04_universe_and_masks.md"),
        title="股票池与交易状态掩码 (Universe & Masks)",
        desc="包含基础可选股票池（BaseUnivMask）、涨跌停掩码（LimitMask）、ST 股票、停牌、上市天数过滤等 29 个状态标记。",
        items=universe_vars,
    )

    # 5. 05_industry_and_benchmark.md
    render_var_page(
        os.path.join(DOCS_DIR, "variables", "05_industry_and_benchmark.md"),
        title="行业分类与指数基准变量 (Industry & Benchmark)",
        desc="包含申万一级/二级/三级行业分类代码、子行业归属、以及中证全指、沪深300等核心基准指数权重（共 48 个）。",
        items=industry_vars + sub_sw_vars,
    )

    # 6. 06_barra_risk_factors.md
    render_var_page(
        os.path.join(DOCS_DIR, "variables", "06_barra_risk_factors.md"),
        title="Barra 风格与风险因子 (Barra Risk Factors)",
        desc="包含 Barra 中国股票风险模型 CNE5（11 个风格因子）与 CNTR（20 个风格因子）的核心风险暴露矩阵，常用于多因子中性化与风险对齐。",
        items=barra_vars,
        group_by_prefix={
            "barra_cne5": "Barra CNE5 风格因子 (11 个)",
            "barra_cntr": "Barra CNTR 风格因子 (20 个)",
        }
    )

    # 7. 07_financial_balance_sheet.md
    render_var_page(
        os.path.join(DOCS_DIR, "variables", "07_financial_balance_sheet.md"),
        title="财务基本面：资产负债表 (Balance Sheet)",
        desc="上市公司资产负债表核心科目（货币资金、应收账款、存货、资产总计、短期负债、所有者权益等共 150 个变量）。",
        items=balance_vars,
    )

    # 8. 08_financial_income_and_cashflow.md
    render_var_page(
        os.path.join(DOCS_DIR, "variables", "08_financial_income_and_cashflow.md"),
        title="财务基本面：利润表与现金流量表 (Income & Cash Flow)",
        desc="利润表科目（营业收入、营业成本、净利润、研发费用等共 86 个）及现金流量表科目（经营、投资、筹资现金流共 89 个）。",
        items=income_vars + cashflow_vars,
        group_by_prefix={
            "f_1d_income": "利润表科目 (Income Statement)",
            "f_1d_cash_flow": "现金流量表科目 (Cash Flow Statement)",
        }
    )

    # 9. 09_daily_derived_indicators.md
    render_var_page(
        os.path.join(DOCS_DIR, "variables", "09_daily_derived_indicators.md"),
        title="日频统计与财务衍生指标 (Daily Derived)",
        desc="由平台预计算的估值比率（PE, PB, PS）、换手率、流动市值、波动率等日频指标（共 36 个）。",
        items=daily_derived_vars,
    )


def render_var_page(path, title, desc, items, group_by=None, group_by_prefix=None):
    lines = [
        f"# {title}",
        "",
        desc,
        "",
        f"> **总计收录**：`{len(items)}` 个变量",
        "",
        "---",
        "",
    ]

    if group_by_prefix:
        for prefix, grp_title in group_by_prefix.items():
            grp_items = [it for it in items if it.get("short_name", "").startswith(prefix)]
            lines.append(f"## {grp_title} ({len(grp_items)} 个)")
            lines.append("")
            lines.append(render_var_table(grp_items))
            lines.append("")
    elif group_by:
        groups = defaultdict(list)
        for it in items:
            groups[it.get(group_by, "其它")].append(it)
        for grp_val, grp_items in sorted(groups.items()):
            lines.append(f"## 分组：{grp_val} ({len(grp_items)} 个)")
            lines.append("")
            lines.append(render_var_table(grp_items))
            lines.append("")
    else:
        lines.append(render_var_table(items))
        lines.append("")

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def render_var_table(items):
    rows = [
        "| 变量名称 (Short Name) | 底层标识 (Actual Name) | 频度 | 维度 (Axes) | 数据类型 (Kind) | 缺失值处理策略 |",
        "| :--- | :--- | :---: | :---: | :---: | :--- |",
    ]
    for it in sorted(items, key=lambda x: x.get("short_name", "")):
        sname = f"`{it.get('short_name')}`"
        aname = f"`{it.get('actual_name')}`"
        freq = it.get("frequency", "-")
        axes = it.get("axes", "-")
        kind = it.get("kind", "-")
        policy = clean_table_cell(it.get("missing_policy", "-"))
        rows.append(f"| {sname} | {aname} | {freq} | {axes} | {kind} | {policy} |")
    return "\n".join(rows)


def build_function_docs(functions):
    os.makedirs(os.path.join(DOCS_DIR, "functions"), exist_ok=True)

    cat_map = {
        "time_series": ("01_time_series_window.md", "时序滑动窗口算子 (Time Series Rolling)"),
        "time_series_shift": ("02_time_series_shift.md", "时序位移与差分算子 (Time Series Shift & Delta)"),
        "cross_section": ("03_cross_section.md", "截面排序与截面统计算子 (Cross Sectional)"),
        "math_and_elementwise": ("04_math_and_elementwise.md", "数学变换与逐元素算术算子 (Math & Arithmetic)"),
        "logic_and_nan": ("05_logic_and_nan.md", "逻辑分支与缺失值净化算子 (Logic & Missing Value)"),
        "operator": ("06_factor_operators.md", "因子标准化与专用处理算子 (Factor Special Operators)"),
    }

    # Split into groups
    grouped = defaultdict(list)
    for f in functions:
        cat = f.get("category")
        if cat in ["math", "arithmetic"]:
            grouped["math_and_elementwise"].append(f)
        elif cat in ["nan", "comparison"]:
            grouped["logic_and_nan"].append(f)
        else:
            grouped[cat].append(f)

    for group_key, (fname, gtitle) in cat_map.items():
        items = grouped.get(group_key, [])
        fpath = os.path.join(DOCS_DIR, "functions", fname)
        render_function_page(fpath, gtitle, items)


def render_function_page(path, title, items):
    lines = [
        f"# {title}",
        "",
        f"收录该类别下的 `{len(items)}` 个计算算子，附带完整的数学语义、参数签名、输入输出契约与矩阵计算用例。",
        "",
        "---",
        "",
        "## 快速速查目录",
        "",
        "| 算子名 | 参数列表 | 返回值 | 窗口依赖 | 说明摘要 |",
        "| :--- | :--- | :--- | :---: | :--- |",
    ]

    for it in sorted(items, key=lambda x: x.get("name", "")):
        name = it.get("name")
        params_str = ", ".join(p.get("name") for p in it.get("parameters", []))
        sig = f"`{name}({params_str})`"
        raw_ret = clean_table_cell(it.get('returns', 'matrix'), max_len=45)
        ret = f"`{raw_ret}`"
        needs_win = "是" if it.get("needs_window") else "否"
        desc = clean_table_cell(it.get("description", "").split("。")[0] + "。")
        lines.append(f"| [{name}](#{name.lower()}) | {sig} | {ret} | {needs_win} | {desc} |")

    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## 算子详细规格卡片")
    lines.append("")

    for it in sorted(items, key=lambda x: x.get("name", "")):
        name = it.get("name")
        params = it.get("parameters", [])
        params_str = ", ".join(p.get("name") for p in params)
        lines.append(f"### <a id=\"{name.lower()}\"></a> `{name}({params_str})`")
        lines.append("")
        lines.append(f"- **功能说明**：{it.get('description')}")
        lines.append(f"- **所属分类**：`{it.get('category')}` ({it.get('category_label')})")
        if it.get("aliases"):
            lines.append(f"- **别名 (Aliases)**：{', '.join(f'`{a}`' for a in it.get('aliases'))}")
        lines.append(f"- **输入形状契约**：`{it.get('input')}`")
        lines.append(f"- **输出形状契约**：`{it.get('returns')}`")
        lines.append(f"- **滑动窗口要求**：{'必须指定正整数 window' if it.get('needs_window') else '无需滑动窗口'}")
        lines.append(f"- **缺失值与非有限值处理 (NaN / Inf)**：{it.get('missing_policy')}")
        lines.append("")
        lines.append("**参数列表**：")
        lines.append("")
        lines.append(format_param_table(params))
        lines.append("")
        if it.get("example"):
            lines.append("**调用示例与数值映射**：")
            lines.append("```text")
            lines.append(it.get("example").strip())
            lines.append("```")
            lines.append("")
        lines.append("---")
        lines.append("")

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def build_postprocess_docs(postprocess_ops):
    os.makedirs(os.path.join(DOCS_DIR, "postprocess"), exist_ok=True)
    fpath = os.path.join(DOCS_DIR, "postprocess", "01_postprocess_guide.md")

    lines = [
        "# FactorLab 因子后处理算子指南 (Postprocess Operators)",
        "",
        "在 FactorLab 中，后处理（Postprocess）是在因子原始矩阵计算完成后、提交回测评估前执行的标准流水线阶段。",
        "后处理通常用于**数据过滤、平滑、去极值、标准化与风格中性化**。",
        "",
        "> [!IMPORTANT]",
        "> **平台自动注入规则**：",
        "> 后处理算子在平台中通过指定算子名和标量参数（如 `window`、`quant`）配置，平台会在运行时**自动注入所依赖的掩码与暴露矩阵**（例如 `LimitMask`、`BaseUnivMask`、行业暴露代码等）。编写因子时无需手动复制原生函数的全量输入参数。",
        "",
        f"**全量收录**：`{len(postprocess_ops)}` 个后处理算子。",
        "",
        "---",
        "",
        "## 后处理算子汇总表",
        "",
        "| 算子名 (Operator) | 阶段分类 | 自动注入依赖 (Implicit Dependencies) | 核心功能摘要 |",
        "| :--- | :---: | :--- | :--- |",
    ]

    for p in postprocess_ops:
        name = p.get("name")
        doc = p.get("documentation", {})
        post_contract = doc.get("postprocess_contract", {})
        deps = ", ".join(f"`{d}`" for d in post_contract.get("implicit_dependencies", [])) or "无"
        desc = clean_table_cell(p.get("description", ""))
        lines.append(f"| [{name}](#{name.lower()}) | 因子后处理 | {deps} | {desc} |")

    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## 后处理算子详细规范")
    lines.append("")

    for p in postprocess_ops:
        name = p.get("name")
        doc = p.get("documentation", {})
        input_c = doc.get("input_contract", {})
        output_c = doc.get("output_contract", {})
        post_c = doc.get("postprocess_contract", {})
        examples = doc.get("examples", [])

        lines.append(f"### <a id=\"{name.lower()}\"></a> `{name}`")
        lines.append("")
        lines.append(f"- **功能概述**：{p.get('description')}")
        if doc.get("formula"):
            lines.append(f"- **计算数学公式**：`{doc.get('formula')}`")
        lines.append(f"- **计算轴 (Axis)**：`{doc.get('axis', 'stocks')}`")
        deps = ", ".join(f"`{d}`" for d in post_c.get("implicit_dependencies", [])) or "无"
        lines.append(f"- **隐式注入依赖**：{deps}")
        lines.append(f"- **输出数据类型与形状**：`{output_c.get('dtype', 'float64')}`，保持 `[{output_c.get('shape', ['T', 'S'])[0]} × {output_c.get('shape', ['T', 'S'])[1]}]`")
        lines.append(f"- **缺失值与非有限值策略**：{doc.get('nan_policy', output_c.get('missing', '无特别说明'))}")
        lines.append(f"- **后处理配置提示**：{post_c.get('note', '无')}")
        lines.append("")
        
        args = p.get("arguments", [])
        if args:
            lines.append("**配置参数 (Arguments)**：")
            lines.append("")
            lines.append("| 参数名 | 说明 |")
            lines.append("| :--- | :--- |")
            for a in args:
                lines.append(f"| `{a}` | 后处理调节参数 |")
            lines.append("")

        if examples:
            lines.append("**运行时执行验证用例**：")
            ex = examples[0]
            lines.append(f"- **调用表达式**：`{ex.get('expression')}`")
            lines.append(f"- **输出验证级别**：`{ex.get('verification')}`")
            lines.append("")

        lines.append("---")
        lines.append("")

    with open(fpath, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def build_readme(data):
    v_cnt = data["summary"]["variables_count"]
    f_cnt = data["summary"]["functions_count"]
    p_cnt = data["summary"]["postprocess_count"]
    total = data["summary"]["total_items"]
    fetched_at = data["fetched_at"]

    content = f"""# FactorLab 因子平台变量与算子完整知识库 (Catalog Documentation)

本文档体系基于 FactorLab MCP 协议元数据自动构建，全面收录了 FactorLab 因子研究工作台当前支持的 **{total} 个核心对象**：
- **{v_cnt} 个变量 (Variables)**：日线量价、分钟高频、日内衍生、股票池掩码、行业与宽基基准、Barra 风险因子及上市公司财务三大表。
- **{f_cnt} 个计算算子函数 (Functions)**：涵盖时序窗口、时序位移、截面排序、数学变换、逻辑分支、缺失值处理与因子处理。
- **{p_cnt} 个后处理算子 (Postprocess Operators)**：涵盖交易掩码、衰减平滑、去极值截断、Z-Score 标准化及多因子风格中性化。

*最后同步时间：`{fetched_at}`*

---

## 一、知识库文档结构总览

```
docs/catalog/
├── README.md                                # 本总览与快速导航
├── variables/                               # 变量体系 (703 个)
│   ├── 01_price_kline.md                    # 日线行情量价 (18 个)
│   ├── 02_minute_highfreq.md                # 1分钟与5分钟高频切片 (162 个)
│   ├── 03_daily_intraday.md                 # 日内衍生与分时指标 (54 个)
│   ├── 04_universe_and_masks.md             # 股票池掩码与交易状态 (29 个)
│   ├── 05_industry_and_benchmark.md         # 申万行业与指数基准 (48 个)
│   ├── 06_barra_risk_factors.md             # Barra 风格因子 (31 个：CNE5 + CNTR)
│   ├── 07_financial_balance_sheet.md        # 财务基本面：资产负债表 (150 个)
│   ├── 08_financial_income_and_cashflow.md  # 财务基本面：利润与现金流 (175 个)
│   └── 09_daily_derived_indicators.md       # 日间衍生统计特征 (36 个)
├── functions/                               # 计算算子函数 (106 个)
│   ├── 01_time_series_window.md             # 时序滑动窗口算子 (38 个)
│   ├── 02_time_series_shift.md              # 时序位移与差分 (4 个)
│   ├── 03_cross_section.md                  # 截面排序与统计 (27 个)
│   ├── 04_math_and_elementwise.md           # 数学变换与逐元素算术 (19 个)
│   ├── 05_logic_and_nan.md                  # 逻辑分支与缺失值处理 (7 个)
│   └── 06_factor_operators.md               # 因子处理与标准化 (11 个)
└── postprocess/                             # 后处理体系 (12 个)
    └── 01_postprocess_guide.md              # 12 种后处理算子规范与自动注入
```

---

## 二、变量体系速查面板 (703 个)

| 类别 | 文档入口 | 数量 | 典型代表 | 频度 | 说明 |
| :--- | :--- | :---: | :--- | :---: | :--- |
| **日线行情** | [01_price_kline.md](./variables/01_price_kline.md) | 18 | `close_hfq`, `volume`, `amount`, `adj_factor` | 1d | 开高低收、成交额量、后复权价格与复权因子 |
| **分钟高频** | [02_minute_highfreq.md](./variables/02_minute_highfreq.md) | 162 | `1m.kline.close`, `5m.kline.volume` | 1m / 5m | 1分钟与5分钟多维度切片高频数据 |
| **日内衍生** | [03_daily_intraday.md](./variables/03_daily_intraday.md) | 54 | `IntraVwap.VwapBegin30`, `IntraTwap.*` | 1d | 基于高频切片聚合生成的日频分时统计特征 |
| **股票池掩码** | [04_universe_and_masks.md](./variables/04_universe_and_masks.md) | 29 | `BaseUnivMask`, `LimitMask`, `STMask` | 1d | 可交易池、涨跌停、停牌及新股过滤掩码 |
| **行业与指数** | [05_industry_and_benchmark.md](./variables/05_industry_and_benchmark.md) | 48 | `sw_l1`, `sub_ind`, `000985.CSI`, `000300.SH` | 1d | 申万各级行业代码与宽基指数成分权重 |
| **Barra 风险** | [06_barra_risk_factors.md](./variables/06_barra_risk_factors.md) | 31 | `barra_cne5_beta`, `barra_cntr_size`, `barra_cntr_momentum` | 1d | Barra CNE5 (11个) 与 CNTR (20个) 风格因子暴露矩阵 |
| **资产负债表** | [07_financial_balance_sheet.md](./variables/07_financial_balance_sheet.md) | 150 | `f_1d_balance_sheet_tot_assets`, `..._monetary_cap` | 1d | 上市公司资产负债表完整原始科目 |
| **利润现金流** | [08_financial_income_and_cashflow.md](./variables/08_financial_income_and_cashflow.md) | 175 | `f_1d_income_oper_rev`, `f_1d_cash_flow_net_cash_flows_oper_act` | 1d | 营业收入、净利润、毛利及各项现金流科目 |
| **日间衍生** | [09_daily_derived_indicators.md](./variables/09_daily_derived_indicators.md) | 36 | `daily_turnover_rate`, `daily_free_market_cap` | 1d | 换手率、自由流通市值、估值比率等日常特征 |

---

## 三、算子函数速查面板 (106 个)

| 类别 | 文档入口 | 数量 | 典型算子 | 特点与窗口要求 |
| :--- | :--- | :---: | :--- | :--- |
| **时序滑动窗口** | [01_time_series_window.md](./functions/01_time_series_window.md) | 38 | `decay`, `emadecay`, `ts_mean`, `ts_std`, `ts_corr`, `ts_rank` | 必须提供 `window` 观察期参数，按时间维度滚动计算 |
| **时序位移差分** | [02_time_series_shift.md](./functions/02_time_series_shift.md) | 4 | `ts_shift`, `ts_delta`, `ts_pct_chg`, `ts_roll` | 针对时序轴进行行平移、前向差分与变化率计算 |
| **截面排序与运算** | [03_cross_section.md](./functions/03_cross_section.md) | 27 | `cs_rank`, `cs_pct_rank`, `cs_zscore`, `cs_mad`, `cs_beta` | 针对每日同一时刻的全部股票截面进行横向统计与排序 |
| **数学变换与逐元素**| [04_math_and_elementwise.md](./functions/04_math_and_elementwise.md) | 19 | `abs`, `log`, `exp`, `scale`, `sign_power`, `add`, `mul` | 逐元素代数运算与非线性数学变换，不改变矩阵维度 |
| **逻辑分支与缺失值**| [05_logic_and_nan.md](./functions/05_logic_and_nan.md) | 7 | `ifelse`, `where`, `fillna`, `is_nan`, `purify` | 条件分支选择、非有限值（NaN/Inf）识别与净化处理 |
| **因子处理专用** | [06_factor_operators.md](./functions/06_factor_operators.md) | 11 | `MAD`, `winsor`, `zScore`, `indNeut`, `riskNeutMulti` | 因子研发专用去极值、中位数缩尾、行业与风险中性化 |

---

## 四、后处理算子速查面板 (12 个)

详细指南见：[01_postprocess_guide.md](./postprocess/01_postprocess_guide.md)

1. **掩码过滤**：`trdMask`（按可交易过滤）、`baseUniMask`（基础池过滤）、`dropNewStock`（次新股过滤）
2. **时序平滑**：`decay`（滑动均值平滑）、`emadecay`（线性衰减加权平滑）
3. **极值与标准化**：`MAD`（绝对中位差截断）、`winsor`（两端分位数缩尾）、`zScore`（截面标准化）、`rank`（百分位排序）
4. **中性化**：`indNeut`（行业中性化）、`riskNeut`（单因子风险中性化）、`riskNeutMulti`（多因子多元回归残差中性化）

---

## 五、命令行快速检索工具 (CLI)

本项目在根目录提供了 `factorlab_catalog.py` 脚本，无需翻阅文档，在终端中可直接秒级查阅：

```bash
# 1. 全局模糊搜索
python3 factorlab_catalog.py search "decay"

# 2. 搜索变量
python3 factorlab_catalog.py var "volume"

# 3. 搜索算子函数
python3 factorlab_catalog.py func "rank"

# 4. 查看具体算子或变量的完整参数与规格详情
python3 factorlab_catalog.py info "emadecay"
python3 factorlab_catalog.py info "close_hfq"
```
"""
    with open(os.path.join(DOCS_DIR, "README.md"), "w", encoding="utf-8") as f:
        f.write(content.strip() + "\n")


def main():
    print("[*] Loading catalog data from JSON...")
    data = load_data()
    variables = data.get("variables", [])
    functions = data.get("functions", [])
    postprocess_ops = data.get("postprocess_operators", [])

    print(f"[*] Building variable docs ({len(variables)} items)...")
    build_variable_docs(variables)

    print(f"[*] Building function docs ({len(functions)} items)...")
    build_function_docs(functions)

    print(f"[*] Building postprocess docs ({len(postprocess_ops)} items)...")
    build_postprocess_docs(postprocess_ops)

    print("[*] Building README index...")
    build_readme(data)

    print("[SUCCESS] All Markdown documentation generated under docs/catalog/!")


if __name__ == "__main__":
    main()
