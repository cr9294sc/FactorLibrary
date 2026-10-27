# FactorLab 因子平台变量与算子完整知识库 (Catalog Documentation)

本文档体系基于 FactorLab MCP 协议元数据自动构建，全面收录了 FactorLab 因子研究工作台当前支持的 **821 个核心对象**：
- **703 个变量 (Variables)**：日线量价、分钟高频、日内衍生、股票池掩码、行业与宽基基准、Barra 风险因子及上市公司财务三大表。
- **106 个计算算子函数 (Functions)**：涵盖时序窗口、时序位移、截面排序、数学变换、逻辑分支、缺失值处理与因子处理。
- **12 个后处理算子 (Postprocess Operators)**：涵盖交易掩码、衰减平滑、去极值截断、Z-Score 标准化及多因子风格中性化。

*最后同步时间：`2026-10-26T02:53:25Z`*

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
