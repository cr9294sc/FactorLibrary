# FactorLab 因子平台 MCP 资产全景速查手册

> **文档定位**：本手册系统整理了通过 Model Context Protocol (MCP) 接入的 **FactorLab（AresQuant）** 因子挖掘与回测云平台上的全部**因子资产**、**计算算子**与**特征变量**。供本地量化研究员、自动化遗传规划（GP）挖掘算子库扩充及跨平台因子迁移参考。

## 一、MCP 服务基础连接与回测环境配置

| 配置项 | 说明 / 参数值 |
| :--- | :--- |
| **MCP 服务端点 (URL)** | `https://factorlab.aresquant.cn/mcp` |
| **协议架构** | JSON-RPC 2.0 over HTTP (POST / Server-Sent Events) |
| **当前鉴权用户** | `226020016 (角色: user)` |
| **因子支持类型** | `expression, code` (表达式 / 独立代码) |
| **数据物化频率** | `1d, 1m, 5m` |
| **回测时钟频率** | `1d` |
| **固定回测历史区间** | `20160101 ~ 20241231` |
| **基准股票池 (Universe)** | `BaseUnivMask` |
| **单因子评价基准标签** | `next_trading_day_vwap30_entry` (持仓: 1d, 分组: 10组) |
| **预测周期 (Horizons)** | `[1, 2, 5, 10] 天` |

### 数据时钟与截断规则 (Data Clocks & Cutoff)
- **日线输出口径**：1d means one vector per trading day; input frequency is independent
- **分钟时钟**：1m 每天 239 根 (09:30:00 ~ 15:00:00)；5m 每天 49 根 (09:30:00 ~ 15:01:03)。
- **截断时刻**：Daily output at or after 15:00 is end-of-day research and includes the complete current trading session, including the 1m closing bar labelled 15:00:00 and the 5m closing bar labelled 15:01:03; earlier cutoffs expose only bars up to that time. This does not certify live publication latency.
- **缺失值策略**：Missing bars, days and securities retain their positions as NaN

---

## 二、已开发托管因子库 (Custom Factors - 8 项)

平台已托管 8 个自定义挖掘/研发因子，均基于表达式语法树（Expression AST）构建，并附带链式后处理（Postprocess Pipeline）与全历史回测验证数据：

### 2.1 因子回测核心绩效横向对比

| 因子名称 | IC 均值 | IC_IR | IC 胜率 | RankIC | RankIC_IR | RankIC 胜率 | 多空收益 (日) | 覆盖率 | 状态 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Volume_Price_2_H5_2** | 0.0265 | 0.2729 | 61.6% | **0.0431** | **0.4019** | 68.0% | 23.64 bps | 97.6% | `draft` |
| **Volume_Price_2_H5** | 0.0277 | 0.2128 | 58.5% | **0.0475** | **0.3244** | 64.1% | 25.35 bps | 97.6% | `draft` |
| **Volume_Price_2_R7** | 0.0255 | 0.1924 | 57.8% | **0.0448** | **0.2979** | 62.5% | 23.24 bps | 97.7% | `draft` |
| **Volume_Price_2** | 0.0254 | 0.1920 | 57.8% | **0.0448** | **0.2980** | 62.5% | 23.08 bps | 97.7% | `draft` |
| **Gaussian distribution** | - | - | - | **0.0017** | **0.0303** | 51.4% | 1.48 bps | 95.3% | `draft` |
| **Tech_Trend_VP_Constrained** | 0.0264 | 0.3478 | 63.6% | **0.0329** | **0.4161** | 65.5% | 20.56 bps | 96.0% | `draft` |
| **Tech_Trend_VP_1_Rev** | 0.0308 | 0.2620 | 60.2% | **0.0395** | **0.3197** | 61.8% | 24.16 bps | 99.7% | `draft` |
| **Volume_Price_1** | 0.0124 | 0.1202 | 57.0% | **0.0181** | **0.1492** | 57.8% | 5.21 bps | 97.7% | `draft` |

### 2.2 因子定义与表达式详案

#### 2.2.1 Volume_Price_2_H5_2 (Volume_Price_2_H5_2)
- **Factor ID**: `fac_5bc39b9aa01c427aa2d28563d7d9b5a2` (最新 Run: `bt_bb3a04da6b20456ab444dcb7e5450949`)
- **原始表达式 (Expression AST)**:
  ```text
  add(sub(BaseUnivMask, BaseUnivMask), mul(-1.0, add(cs_zscore(ts_corr(close_hfq, log(amount), 20)), cs_zscore(ts_std(daily_derived_returns_c2c_1d, 20)))))
  ```
- **后处理流水线 (Postprocess Pipeline)**:
  `trdMask()` -> `dropNewStock(120)` -> `MAD(3)` -> `indNeut('barra_cne5_industry')` -> `decay(5)` -> `rank()`
- **全量回测指标 (2016-2024, 2185 交易日)**:
  - IC: `0.0265` (IR: `0.2729`, 胜率: `61.6%`)
  - RankIC: `0.0431` (IR: `0.4019`, 胜率: `68.0%`)
  - 5日 RankIC: `0.0332` (5d IR: `0.3484`)
  - 多空组收益: `23.64 bps/天` (多空 IR: `0.2956`)
  - 股票池覆盖率: `97.65%`

#### 2.2.2 Volume_Price_2_H5 (Volume_Price_2_H5)
- **Factor ID**: `fac_238b9b9a250340d880b6a0dc9cd856c0` (最新 Run: `bt_4c00df28632d46b8a2f231303ef21e4c`)
- **原始表达式 (Expression AST)**:
  ```text
  add(sub(BaseUnivMask, BaseUnivMask), mul(-1.0, add(cs_zscore(ts_corr(close_hfq, log(amount), 20)), cs_zscore(ts_std(daily_derived_returns_c2c_1d, 20)))))
  ```
- **后处理流水线 (Postprocess Pipeline)**:
  `trdMask()` -> `dropNewStock(120)` -> `MAD(3.0)` -> `decay(5)` -> `rank()`
- **全量回测指标 (2016-2024, 2185 交易日)**:
  - IC: `0.0277` (IR: `0.2128`, 胜率: `58.5%`)
  - RankIC: `0.0475` (IR: `0.3244`, 胜率: `64.1%`)
  - 5日 RankIC: `0.0343` (5d IR: `0.2718`)
  - 多空组收益: `25.35 bps/天` (多空 IR: `0.2399`)
  - 股票池覆盖率: `97.65%`

#### 2.2.3 Volume_Price_2_R7 (Volume_Price_2_R7)
- **Factor ID**: `fac_5f762e2036b745c594f1f5e5d7a9f5ea` (最新 Run: `bt_40dbde1c01bc456e8127804e85448610`)
- **描述**: Volume_Price_2 R7 独立保存与复现版本 (对应 fv_0538d9bbcba440a78e8b308138c8e17f)
- **原始表达式 (Expression AST)**:
  ```text
  add(sub(BaseUnivMask, BaseUnivMask), mul(-1.0, add(cs_zscore(ts_corr(close_hfq, log(amount), 20)), cs_zscore(ts_std(daily_derived_returns_c2c_1d, 20)))))
  ```
- **后处理流水线 (Postprocess Pipeline)**:
  `trdMask()` -> `dropNewStock(120)` -> `decay(10)` -> `MAD(3.0)` -> `rank()`
- **全量回测指标 (2016-2024, 2185 交易日)**:
  - IC: `0.0255` (IR: `0.1924`, 胜率: `57.8%`)
  - RankIC: `0.0448` (IR: `0.2979`, 胜率: `62.5%`)
  - 5日 RankIC: `0.0326` (5d IR: `0.2519`)
  - 多空组收益: `23.24 bps/天` (多空 IR: `0.2167`)
  - 股票池覆盖率: `97.68%`

#### 2.2.4 Volume_Price_2 (Volume_Price_2)
- **Factor ID**: `fac_2ae042705b8241eba2c2d695423da171` (最新 Run: `bt_fd55b3711ef4465692e7f91959bec401`)
- **原始表达式 (Expression AST)**:
  ```text
  add(sub(LimitMask, LimitMask), mul(-1.0, add(cs_zscore(ts_corr(close_hfq, log(amount), 20)), cs_zscore(ts_std(daily_derived_returns_c2c_1d, 20)))))
  ```
- **后处理流水线 (Postprocess Pipeline)**:
  `baseUniMask()` -> `dropNewStock(120)` -> `decay(10)` -> `MAD(3.0)` -> `rank()`
- **全量回测指标 (2016-2024, 2185 交易日)**:
  - IC: `0.0254` (IR: `0.1920`, 胜率: `57.8%`)
  - RankIC: `0.0448` (IR: `0.2980`, 胜率: `62.5%`)
  - 5日 RankIC: `0.0324` (5d IR: `0.2516`)
  - 多空组收益: `23.08 bps/天` (多空 IR: `0.2152`)
  - 股票池覆盖率: `97.68%`

#### 2.2.5 Gaussian distribution (高斯分布)
- **Factor ID**: `fac_1fc76059531141d98deb3e2fb5f3e3e1` (最新 Run: `bt_a31a4cd18ed14fc8861a7a2b97c29d43`)
- **原始表达式 (Expression AST)**:
  ```text
  mul(-1.0, add(cs_zscore(ts_corr(close_hfq, log(amount), 20)), cs_zscore(ts_std(daily_derived_returns_c2c_1d, 20))))
  ```
- **后处理流水线 (Postprocess Pipeline)**:
  `baseUniMask()` -> `trdMask()` -> `dropNewStock(120)`
- **全量回测指标 (2016-2024, 2185 交易日)**:
  - IC: `-` (IR: `-`, 胜率: `-`)
  - RankIC: `0.0017` (IR: `0.0303`, 胜率: `51.4%`)
  - 5日 RankIC: `-` (5d IR: `-`)
  - 多空组收益: `1.48 bps/天` (多空 IR: `0.0178`)
  - 股票池覆盖率: `95.27%`

#### 2.2.6 Tech_Trend_VP_Constrained (Tech_Trend_VP_Constrained (多重约束中性化版))
- **Factor ID**: `fac_ee2bc6b4269344fa8198729c4502905f` (最新 Run: `bt_28fab47e9fe7438192ba08eb9e5fe60f`)
- **描述**: 反向技术量价因子，增加行业中性化、Barra四风格正交回归、次新股120日与极值收紧约束
- **原始表达式 (Expression AST)**:
  ```text
  mul(-1.0, add(add(cs_zscore(div(sub(close_hfq, ts_mean(close_hfq, 20)), ts_mean(close_hfq, 20))), cs_zscore(div(sub(close_hfq, low), add(sub(high, low), 0.001)))), cs_zscore(mul(ts_corr(close_hfq, amount, 20), div(amount, add(ts_mean(amount, 5), 0.001))))))
  ```
- **后处理流水线 (Postprocess Pipeline)**:
  `baseUniMask()` -> `trdMask()` -> `dropNewStock(120)` -> `MAD(2.5)` -> `indNeut('barra_cne5_industry')` -> `riskNeutMulti(['barra_cne5_size', 'barra_cne5_beta', 'barra_cne5_liquidty', 'barra_cne5_resvol'])` -> `zScore()` -> `emadecay(3)`
- **全量回测指标 (2016-2024, 2185 交易日)**:
  - IC: `0.0264` (IR: `0.3478`, 胜率: `63.6%`)
  - RankIC: `0.0329` (IR: `0.4161`, 胜率: `65.5%`)
  - 5日 RankIC: `0.0325` (5d IR: `0.4532`)
  - 多空组收益: `20.56 bps/天` (多空 IR: `0.3211`)
  - 股票池覆盖率: `96.00%`

#### 2.2.7 Tech_Trend_VP_1_Rev (Tech_Trend_VP_1_Rev (反向特质反转))
- **Factor ID**: `fac_2131e4e3d556409299bf34a837765967` (最新 Run: `bt_346b56e4bd91496a88129b28ec95aa6b`)
- **描述**: 取负后的反向技术量价因子
- **原始表达式 (Expression AST)**:
  ```text
  mul(-1.0, add(add(cs_zscore(div(sub(close_hfq, ts_mean(close_hfq, 20)), ts_mean(close_hfq, 20))), cs_zscore(div(sub(close_hfq, low), add(sub(high, low), 0.001)))), cs_zscore(mul(ts_corr(close_hfq, amount, 20), div(amount, add(ts_mean(amount, 5), 0.001))))))
  ```
- **后处理流水线 (Postprocess Pipeline)**:
  `baseUniMask()` -> `trdMask()` -> `dropNewStock(60)` -> `MAD(3.0)` -> `zScore()` -> `emadecay(3)`
- **全量回测指标 (2016-2024, 2185 交易日)**:
  - IC: `0.0308` (IR: `0.2620`, 胜率: `60.2%`)
  - RankIC: `0.0395` (IR: `0.3197`, 胜率: `61.8%`)
  - 5日 RankIC: `0.0380` (5d IR: `0.3455`)
  - 多空组收益: `24.16 bps/天` (多空 IR: `0.2228`)
  - 股票池覆盖率: `99.72%`

#### 2.2.8 Volume_Price_1 (Volume_Price_1)
- **Factor ID**: `fac_dc0aebbdea354ecaadf046ffd4a9e0a7` (最新 Run: `bt_84683d8c87414bb58a114809b6626219`)
- **原始表达式 (Expression AST)**:
  ```text
  mul(mul(-1.0, cs_zscore(daily_derived_returns_intra)), exp(mul(-0.5, add(power(cs_zscore(daily_derived_returns_intra), 2), power(cs_zscore(amount), 2)))))
  ```
- **后处理流水线 (Postprocess Pipeline)**:
  `baseUniMask()` -> `trdMask()`
- **全量回测指标 (2016-2024, 2185 交易日)**:
  - IC: `0.0124` (IR: `0.1202`, 胜率: `57.0%`)
  - RankIC: `0.0181` (IR: `0.1492`, 胜率: `57.8%`)
  - 5日 RankIC: `0.0075` (5d IR: `0.0755`)
  - 多空组收益: `5.21 bps/天` (多空 IR: `0.0640`)
  - 股票池覆盖率: `97.66%`

---

## 三、因子后处理算子 (Postprocess Operators - 12 项)

后处理算子作用于表达式求值后的日截面矩阵序列，按声明顺序链式执行，主要用于极端值处理、风险/行业中性化、时序平滑与截面标准化：

| 算子名称 | 功能描述 | 参数列表与默认值 | 缺失值与边界说明 |
| :--- | :--- | :--- | :--- |
| **`trdMask`** | 按可交易条件过滤输入，不可交易位置返回 NaN。 | *无参数* | 遵循标准 NaN 屏蔽规则 |
| **`baseUniMask`** | 按基础股票池条件过滤输入，池外位置返回 NaN。 | *无参数* | 遵循标准 NaN 屏蔽规则 |
| **`dropNewStock`** | 根据上市时长过滤新股，不满足 min_dates 的位置返回 NaN。 | `minimum_days: integer = 60` | 遵循标准 NaN 屏蔽规则 |
| **`decay`** | 完整窗口内的滚动算术均值。 | `window: integer = 5` | 遵循标准 NaN 屏蔽规则 |
| **`emadecay`** | 按旧到新 1..N 权重线性加权的滚动衰减，不是指数衰减。 | `window: integer = 5` | 遵循标准 NaN 屏蔽规则 |
| **`MAD`** | 按截面中位数和经 1.4826 缩放的 MAD 截断极值。 | `threshold: number = 3.0` | 遵循标准 NaN 屏蔽规则 |
| **`winsor`** | 按两端 quant 分位数对极端值进行缩尾处理。 | `quantile: number = 0.025` | 遵循标准 NaN 屏蔽规则 |
| **`zScore`** | 对每个交易时点的股票截面执行 Z-score 标准化。 | *无参数* | 遵循标准 NaN 屏蔽规则 |
| **`rank`** | 逐时点截面百分位排名：从 1 开始的稳定原始名次除以有效样本数。 | *无参数* | 遵循标准 NaN 屏蔽规则 |
| **`indNeut`** | 在每个时点按 x2 行业代码对 x1 减去同行业非缺失均值。 | `industry: variable = barra_cne5_industry` | 遵循标准 NaN 屏蔽规则 |
| **`riskNeut`** | 对因子执行风险暴露中性化。 | `exposures: variables = ['barra_cne5_size']` | 遵循标准 NaN 屏蔽规则 |
| **`riskNeutMulti`** | 以 x1 为被解释变量、后续矩阵为暴露，做无截距截面多元回归并返回残差。 | `exposures: variables = ['barra_cne5_size']` | 遵循标准 NaN 屏蔽规则 |

### 3.1 常见后处理算子用法解析

1. **`trdMask` / `baseUniMask`**：交易日可交易性掩码与基准股票池过滤，不可交易/池外位置置为 NaN。
2. **`dropNewStock(minimum_days)`**：新股过滤，默认 60 个自然日，过滤掉上市初期高波动筹码。
3. **`MAD(threshold)` / `winsor(quantile)`**：中位数绝对偏差去极值与双端缩尾，MAD 默认阈值 3.0，配合 1.4826 正态一致性缩放。
4. **`indNeut(industry)`**：行业中性化，默认依据申万/Barra 行业分组减去行业内均值。
5. **`riskNeutMulti(exposures)`**：多元风险中性化，利用无截距 OLS 投影残差，剥离指定风格（如 `size`, `beta`, `liquidty`, `resvol`）。
6. **`decay(window)` / `emadecay(window)`**：滚动窗口衰减与线性加权移动平滑，抑制因子换手率。
7. **`zScore` / `rank`**：截面标准化与截面分位数排序归一化至 $[0, 1]$。

---

## 四、表达式计算算子库 (Functions / Expression Operators - 106 项)

FactorLab 表达式引擎包含 **106** 个原子计算算子，涵盖 8 大类别的逐元素代数、数学变换、截面统计、时序滚动与高级矩阵回归运算：


### 4.1 逐元素运算 (7 个算子)

| 算子 | 参数签名 | 滚动窗口? | 功能描述 | 典型调用示例 |
| :--- | :--- | :---: | :--- | :--- |
| **`add`** | `add(x1, x2)` | 否 | 逐元素计算 x1 + x2。 | `add(x1,x2)` |
| **`div`** | `div(x1, x2)` | 否 | 安全逐元素相除；绝对值小于 1e-6 的分母转为 NaN。 | `div(x1,x2)` |
| **`greater`** | `greater(x1, x2)` | 否 | 逐元素取两个输入的最大值，不是“大于”比较。 | `greater(x1,x2)` |
| **`less`** | `less(x1, x2)` | 否 | 逐元素取两个输入的最小值，不是“小于”比较。 | `less(x1,x2)` |
| **`mul`** | `mul(x1, x2)` | 否 | 逐元素计算 x1 * x2。 | `mul(x1,x2)` |
| **`power`** | `power(x1, n)` | 否 | 逐元素计算 x1 的 n 次幂。 | `power(x1,2)` |
| **`sub`** | `sub(x1, x2)` | 否 | 逐元素计算 x1 - x2。 | `sub(x1,x2)` |

### 4.2 条件与比较 (2 个算子)

| 算子 | 参数签名 | 滚动窗口? | 功能描述 | 典型调用示例 |
| :--- | :--- | :---: | :--- | :--- |
| **`ifelse`** | `ifelse(x1, x2, x3)` | 否 | 逐元素按条件 x1 选择 x2 或 x3；输出所选分支的数值。 | `ifelse(x1,x2,x3)` |
| **`where`** | `where(x1, x2, x3)` | 否 | 逐元素按条件 x1 选择 x2 或 x3；输出所选分支的数值。 | `where(x1,x2,x3)` |

### 4.3 数学变换 (12 个算子)

| 算子 | 参数签名 | 滚动窗口? | 功能描述 | 典型调用示例 |
| :--- | :--- | :---: | :--- | :--- |
| **`abs`** | `abs(x1)` | 否 | 逐元素取绝对值。 | `abs(x1)` |
| **`ceil`** | `ceil(x1)` | 否 | 逐元素向上取整。 | `ceil(x1)` |
| **`clip`** | `clip(x1, x2, x3)` | 否 | 将每个元素限制在 [lo, hi] 区间内。 | `clip(x1,x2,x3)` |
| **`exp`** | `exp(x1)` | 否 | 逐元素计算自然指数 e^x。 | `exp(x1)` |
| **`log`** | `log(x1)` | 否 | 逐元素计算自然对数；输入小于或等于零时返回 NaN。 | `log(x1)` |
| **`perc_long`** | `perc_long(x1, percentile)` | 否 | 先减去截面分位点，正值保留，负值统一替换为分摊多头总量的负值；输出多空数值。 | `perc_long(x1,0.5)` |
| **`round`** | `round(x1, decimals)` | 否 | 逐元素按 decimals 位舍入；恰好半值采用向偶数舍入。 | `round(x1,2)` |
| **`scale`** | `scale(x1)` | 否 | 按截面绝对值之和缩放输入。 | `scale(x1)` |
| **`sign`** | `sign(x1)` | 否 | 逐元素返回输入的符号。 | `sign(x1)` |
| **`sign_power`** | `sign_power(x1)` | 否 | 逐元素计算带符号平方 sign(x1) * x1²；只接收一个矩阵参数。 | `sign_power(x1)` |
| **`sqrt`** | `sqrt(x1)` | 否 | 逐元素计算平方根；小于或等于零的输入均产生 NaN。 | `sqrt(x1)` |
| **`unique_count`** | `unique_count(x1)` | 否 | 逐截面统计不同非 NaN 数值的个数；正负 Inf 也计入。 | `unique_count(x1)` |

### 4.4 缺失值与极值处理 (5 个算子)

| 算子 | 参数签名 | 滚动窗口? | 功能描述 | 典型调用示例 |
| :--- | :--- | :---: | :--- | :--- |
| **`fillna`** | `fillna(x1, fill_value)` | 否 | 仅把 NaN 替换为 fill_value；其他值包括正负无穷保持不变。 | `fillna(x1,1)` |
| **`is_inf`** | `is_inf(x1)` | 否 | 逐元素判断是否为正负无穷。 | `is_inf(x1)` |
| **`is_nan`** | `is_nan(x1)` | 否 | 逐元素判断是否为 NaN。 | `is_nan(x1)` |
| **`purify`** | `purify(x1)` | 否 | 将正负无穷等非有限值转换为 NaN。 | `purify(x1)` |
| **`zero_to_nan`** | `zero_to_nan(x1)` | 否 | 将零值转换为 NaN。 | `zero_to_nan(x1)` |

### 4.5 截面运算 (Cross-Section) (27 个算子)

| 算子 | 参数签名 | 滚动窗口? | 功能描述 | 典型调用示例 |
| :--- | :--- | :---: | :--- | :--- |
| **`cs_argmax`** | `cs_argmax(x1)` | 否 | 返回每个交易时点股票截面最大值所在的位置。 | `cs_argmax(x1)` |
| **`cs_argmin`** | `cs_argmin(x1)` | 否 | 返回每个交易时点股票截面最小值所在的位置。 | `cs_argmin(x1)` |
| **`cs_beta`** | `cs_beta(x1, x2)` | 否 | 按交易时点计算 x1 相对 x2 的截面回归 beta。 | `cs_beta(x1,x2)` |
| **`cs_corr`** | `cs_corr(x1, x2)` | 否 | 计算每个交易时点两个股票截面之间的相关系数。 | `cs_corr(x1,x2)` |
| **`cs_cov`** | `cs_cov(x1, x2)` | 否 | 计算每个交易时点两个股票截面之间的协方差。 | `cs_cov(x1,x2)` |
| **`cs_dense_pct_rank`** | `cs_dense_pct_rank(x1)` | 否 | 在每个交易时点执行稠密百分位排名。 | `cs_dense_pct_rank(x1)` |
| **`cs_dense_rank`** | `cs_dense_rank(x1)` | 否 | 在每个交易时点执行稠密排名，相同值使用同一名次。 | `cs_dense_rank(x1)` |
| **`cs_group`** | `cs_group(x1, groups)` | 否 | 截面百分位排名经浮点整除得到组号。 | `cs_group(x1,3)` |
| **`cs_kurt`** | `cs_kurt(x1)` | 否 | 计算每个交易时点的股票截面峰度。 | `cs_kurt(x1)` |
| **`cs_mad`** | `cs_mad(x1)` | 否 | 计算相对均值的平均绝对偏差；不是相对中位数的 MAD。 | `cs_mad(x1)` |
| **`cs_max`** | `cs_max(x1)` | 否 | 计算每个交易时点的股票截面最大值。 | `cs_max(x1)` |
| **`cs_mean`** | `cs_mean(x1)` | 否 | 计算每个交易时点的股票截面均值。 | `cs_mean(x1)` |
| **`cs_median`** | `cs_median(x1)` | 否 | 取计算轴有限值排序后的下中位数；偶数样本取中间较小值。 | `cs_median(x1)` |
| **`cs_min`** | `cs_min(x1)` | 否 | 计算每个交易时点的股票截面最小值。 | `cs_min(x1)` |
| **`cs_multi_beta`** | `cs_multi_beta(x1, x2, intercept)` | 否 | 对每个交易时点执行多元截面回归并返回回归系数。 | `cs_multi_beta(x1,x2,0)` |
| **`cs_multi_resi`** | `cs_multi_resi(x1, x2, intercept)` | 否 | 按交易时点使用 x1 解释 x2，并返回截面回归残差。 | `cs_multi_resi(x1,x2,0)` |
| **`cs_multi_wls_beta`** | `cs_multi_wls_beta(x1, x2, x3, intercept)` | 否 | 对每个交易时点执行加权多元截面回归并返回回归系数。 | `cs_multi_wls_beta(x1,x2,x3,0)` |
| **`cs_multi_wls_resi`** | `cs_multi_wls_resi(x1, x2, x3, intercept)` | 否 | 对每个交易时点执行加权多元截面回归并返回残差。 | `cs_multi_wls_resi(x1,x2,x3,0)` |
| **`cs_nonan_num`** | `cs_nonan_num(x1)` | 否 | 统计每个交易时点股票截面中的非 NaN 数量。 | `cs_nonan_num(x1)` |
| **`cs_pct_rank`** | `cs_pct_rank(x1)` | 否 | 逐时点截面百分位排名：从 1 开始的稳定原始名次除以有效样本数。 | `cs_pct_rank(x1)` |
| **`cs_quantile`** | `cs_quantile(x1, q)` | 否 | 计算每个交易时点股票截面的 q 分位数。 | `cs_quantile(x1,0.5)` |
| **`cs_rank`** | `cs_rank(x1)` | 否 | 逐时点截面稳定排序，输出从 1 开始的原始名次。 | `cs_rank(x1)` |
| **`cs_skew`** | `cs_skew(x1)` | 否 | 计算每个交易时点的股票截面偏度。 | `cs_skew(x1)` |
| **`cs_std`** | `cs_std(x1)` | 否 | 计算每个交易时点的股票截面标准差。 | `cs_std(x1)` |
| **`cs_sum`** | `cs_sum(x1)` | 否 | 计算每个交易时点的股票截面总和。 | `cs_sum(x1)` |
| **`cs_var`** | `cs_var(x1)` | 否 | 计算每个交易时点的股票截面方差。 | `cs_var(x1)` |
| **`cs_zscore`** | `cs_zscore(x1)` | 否 | 按交易时点对股票截面执行 Z-score 标准化。 | `cs_zscore(x1)` |

### 4.6 时序位移与差分 (Time-Series Shift) (4 个算子)

| 算子 | 参数签名 | 滚动窗口? | 功能描述 | 典型调用示例 |
| :--- | :--- | :---: | :--- | :--- |
| **`ts_delta`** | `ts_delta(x1, period)` | 否 | 计算当前值减去 period 个周期前的值。 | `ts_delta(x1,1)` |
| **`ts_pct_chg`** | `ts_pct_chg(x1, period)` | 否 | 计算相对 period 个周期前数值的变化率。 | `ts_pct_chg(x1,1)` |
| **`ts_roll`** | `ts_roll(x1, period)` | 否 | 沿时间轴循环移位，首尾环绕。 | `ts_roll(x1,1)` |
| **`ts_shift`** | `ts_shift(x1, period)` | 否 | 沿时间轴移动 period 行，移入部分填 NaN。 | `ts_shift(x1,1)` |

### 4.7 时序窗口滚动 (Time-Series Window) (38 个算子)

| 算子 | 参数签名 | 滚动窗口? | 功能描述 | 典型调用示例 |
| :--- | :--- | :---: | :--- | :--- |
| **`decay`** | `decay(x1, window)` | 是 | 完整窗口内的滚动算术均值。 | `decay(x1,3)` |
| **`dense_rank_ts`** | `dense_rank_ts(x1)` | 否 | 在输入的时间序列窗口内执行稠密排名。 | `dense_rank_ts(x1)` |
| **`emadecay`** | `emadecay(x1, window)` | 是 | 按旧到新 1..N 权重线性加权的滚动衰减，不是指数衰减。 | `emadecay(x1,3)` |
| **`pct_dense_rank_ts`** | `pct_dense_rank_ts(x1)` | 否 | 在输入的时间序列窗口内返回稠密百分位排名。 | `pct_dense_rank_ts(x1)` |
| **`pct_rank_ts`** | `pct_rank_ts(x1)` | 否 | 在整段输入时间历史内对每只股票的所有时点做稳定百分位排名，输出整段矩阵。 | `pct_rank_ts(x1)` |
| **`rank_ts`** | `rank_ts(x1)` | 否 | 在整段输入时间历史内对每只股票的所有时点做稳定原始排名，输出整段排名矩阵。 | `rank_ts(x1)` |
| **`ts_argmax`** | `ts_argmax(x1)` | 否 | 沿每只股票的时间轴计算最大值位置。 | `ts_argmax(x1)` |
| **`ts_argmin`** | `ts_argmin(x1)` | 否 | 沿每只股票的时间轴计算最小值位置。 | `ts_argmin(x1)` |
| **`ts_beta`** | `ts_beta(x1, x2, window)` | 是 | 计算 x1 相对 x2 在过去 N 个周期的滚动 beta。 | `ts_beta(x1,x2,3)` |
| **`ts_corr`** | `ts_corr(x1, x2, window)` | 是 | 计算两组输入过去 N 个周期的滚动相关系数。 | `ts_corr(x1,x2,3)` |
| **`ts_cov`** | `ts_cov(x1, x2, window)` | 是 | 计算两组输入过去 N 个周期的滚动协方差。 | `ts_cov(x1,x2,3)` |
| **`ts_cumprod`** | `ts_cumprod(x1)` | 否 | 沿每只股票的时间轴计算累计乘积。 | `ts_cumprod(x1)` |
| **`ts_cumsum`** | `ts_cumsum(x1)` | 否 | 沿每只股票的时间轴计算累计和。 | `ts_cumsum(x1)` |
| **`ts_dense_rank`** | `ts_dense_rank(x1)` | 否 | 返回最后一行在整段输入历史内的稠密名次；当前接口不接受 window。 | `ts_dense_rank(x1)` |
| **`ts_dw`** | `ts_dw(x1, x2)` | 否 | 源定义的残差 Durbin-Watson 统计量；差分沿股票轴。 | `ts_dw(x1,x2)` |
| **`ts_ema`** | `ts_ema(x1, window)` | 是 | 当前锁定源实现的窗口 EMA 权重在有效位置相等，数值表现为窗口均值。 | `ts_ema(x1,3)` |
| **`ts_ffill`** | `ts_ffill(x1)` | 否 | 沿每只股票的时间轴向前填充 NaN。 | `ts_ffill(x1)` |
| **`ts_group`** | `ts_group(x1, groups)` | 否 | 按一周期收益率做浮点整除分组；不是窗口排名分组。 | `ts_group(x1,3)` |
| **`ts_kurt`** | `ts_kurt(x1, window)` | 是 | 对每只股票计算过去 N 个周期的峰度。 | `ts_kurt(x1,3)` |
| **`ts_mad`** | `ts_mad(x1, window)` | 是 | 计算相对均值的平均绝对偏差；不是相对中位数的 MAD。 | `ts_mad(x1,3)` |
| **`ts_max`** | `ts_max(x1, window)` | 是 | 对每只股票计算过去 N 个周期的最大值。 | `ts_max(x1,3)` |
| **`ts_mean`** | `ts_mean(x1, window)` | 是 | 对每只股票计算过去 N 个周期的滚动均值。 | `ts_mean(x1,3)` |
| **`ts_median`** | `ts_median(x1, window)` | 是 | 取计算轴有限值排序后的下中位数；偶数样本取中间较小值。 | `ts_median(x1,3)` |
| **`ts_min`** | `ts_min(x1, window)` | 是 | 对每只股票计算过去 N 个周期的最小值。 | `ts_min(x1,3)` |
| **`ts_multi_beta`** | `ts_multi_beta(x1, x2, intercept)` | 否 | 沿时间轴执行多元回归并返回回归系数。 | `ts_multi_beta(x1,x2,0)` |
| **`ts_multi_resi`** | `ts_multi_resi(x1, x2, intercept)` | 否 | 沿时间轴执行多元回归并返回残差。 | `ts_multi_resi(x1,x2,0)` |
| **`ts_multi_wls_beta`** | `ts_multi_wls_beta(x1, x2, x3, intercept)` | 否 | 沿时间轴执行加权多元回归并返回回归系数。 | `ts_multi_wls_beta(x1,x2,x3,0)` |
| **`ts_multi_wls_resi`** | `ts_multi_wls_resi(x1, x2, x3, intercept)` | 否 | 沿时间轴执行加权多元回归并返回残差。 | `ts_multi_wls_resi(x1,x2,x3,0)` |
| **`ts_nonan_num`** | `ts_nonan_num(x1, window)` | 是 | 保留源 roll 维度口径的滚动非 NaN 计数。 | `ts_nonan_num(x1,3)` |
| **`ts_pct_dense_rank`** | `ts_pct_dense_rank(x1)` | 否 | 返回最后一行在整段输入历史内的稠密名次百分比；当前接口不接受 window。 | `ts_pct_dense_rank(x1)` |
| **`ts_pct_rank`** | `ts_pct_rank(x1, window)` | 是 | 计算当前值在每只股票过去 N 个周期中的百分位排名。 | `ts_pct_rank(x1,3)` |
| **`ts_prod`** | `ts_prod(x1, window)` | 是 | 计算每只股票过去 N 个周期的连乘积。 | `ts_prod(x1,3)` |
| **`ts_quantile`** | `ts_quantile(x1, q, window)` | 是 | 计算每只股票过去 N 个周期的 q 分位数。 | `ts_quantile(x1,0.5,3)` |
| **`ts_rank`** | `ts_rank(x1, window)` | 是 | 计算当前值在每只股票过去 N 个周期中的排名。 | `ts_rank(x1,3)` |
| **`ts_skew`** | `ts_skew(x1, window)` | 是 | 对每只股票计算过去 N 个周期的偏度。 | `ts_skew(x1,3)` |
| **`ts_std`** | `ts_std(x1, window)` | 是 | 对每只股票计算过去 N 个周期的标准差。 | `ts_std(x1,3)` |
| **`ts_sum`** | `ts_sum(x1, window)` | 是 | 对每只股票计算过去 N 个周期的滚动总和。 | `ts_sum(x1,3)` |
| **`ts_var`** | `ts_var(x1, window)` | 是 | 对每只股票计算过去 N 个周期的方差。 | `ts_var(x1,3)` |

### 4.8 因子复合处理 (Factor Operators) (11 个算子)

| 算子 | 参数签名 | 滚动窗口? | 功能描述 | 典型调用示例 |
| :--- | :--- | :---: | :--- | :--- |
| **`barra5_zScore`** | `barra5_zScore(x1, x2)` | 否 | 用 x2 市值权重计算截面均值、等权总体标准差进行标准化。 | `barra5_zScore(x1,x2)` |
| **`baseUniMask`** | `baseUniMask(x1, x2)` | 否 | 按基础股票池条件过滤输入，池外位置返回 NaN。 | `baseUniMask(x1,x2)` |
| **`dropNewStock`** | `dropNewStock(x1, x2, min_dates)` | 否 | 根据上市时长过滤新股，不满足 min_dates 的位置返回 NaN。 | `dropNewStock(x1,x2,3)` |
| **`indNeut`** | `indNeut(x1, x2)` | 否 | 在每个时点按 x2 行业代码对 x1 减去同行业非缺失均值。 | `indNeut(x1,x2)` |
| **`MAD`** | `MAD(x1, n)` | 否 | 按截面中位数和经 1.4826 缩放的 MAD 截断极值。 | `MAD(x1,2)` |
| **`rank`** | `rank(x1)` | 否 | 逐时点截面百分位排名：从 1 开始的稳定原始名次除以有效样本数。 | `rank(x1)` |
| **`riskNeut`** | `riskNeut(x1, x2)` | 否 | 对因子执行风险暴露中性化。 | `riskNeut(x1,x2)` |
| **`riskNeutMulti`** | `riskNeutMulti(x1, x2)` | 否 | 以 x1 为被解释变量、后续矩阵为暴露，做无截距截面多元回归并返回残差。 | `riskNeutMulti(x1,x2)` |
| **`trdMask`** | `trdMask(x1, x2)` | 否 | 按可交易条件过滤输入，不可交易位置返回 NaN。 | `trdMask(x1,x2)` |
| **`winsor`** | `winsor(x1, quant)` | 否 | 按两端 quant 分位数对极端值进行缩尾处理。 | `winsor(x1,0.25)` |
| **`zScore`** | `zScore(x1)` | 否 | 对每个交易时点的股票截面执行 Z-score 标准化。 | `zScore(x1)` |

---

## 五、底层基础特征与变量库 (Variables & Features - 759 项)

FactorLab 提供了 759 个高维度原始特征矩阵（支持 `1d`、`1m`、`5m` 频率），涵盖量价行情、Barra 风险模型暴露、股票池掩码、财务报表与分析师预期：

### 5.1 变量资产大类分布统计

| 变量类别代码 | 类别中文名称 | 变量总数 | 支持频率 | 典型变量举例 |
| :--- | :--- | :---: | :--- | :--- |
| `price` | 日线量价行情 | **16** | 1d | `open`, `close_hfq`, `amount`, `daily_kline_pct_chg` |
| `universe` | 股票池掩码与状态 | **21** | 1d | `BaseUnivMask`, `LimitMask`, `SuspendStock`, `STStock` |
| `risk` | 风险模型风格暴露 (Barra) | **94** | 1d | `barra_cne5_beta`, `barra_cne5_size`, `barra_cne5_momentum` |
| `fundamental` | 基本面与估值指标 | **4** | 1d | `daily_fdm_mkt_cap`, `daily_fdm_totalshare`, `daily_fdm_dv_ttm` |
| `industry` | 行业与指数成分权重 | **27** | 1d | `index_weight_000300_sh`, `index_weight_000905_sh` |
| `minute` | 日内高频微结构 (分钟线) | **162** | 1m, 5m | `min1_kline_close`, `min5_kline_volume`, `min1_kline_vwap` |
| `forecast` | 卖方分析师一致预期 | **51** | 1d | `forecast_net_profit_avg`, `forecast_pe_avg` |
| `statement` | 三大财务报表原生科目 | **325** | 1d (点时性) | 资产负债表、利润表、现金流量表 325 项明细科目 |
| `other` | 衍生统计与综合特征 | **59** | 1d | `daily_derived_returns_c2c_1d`, `daily_derived_returns_intra` |

### 5.2 核心量价行情、股票池掩码与估值 (Price, Universe & Fundamental)

| 变量短名称 (Short Name) | 平台内部标识 (Actual Name) | 维度/口径 | 字段定义与业务语义 |
| :--- | :--- | :--- | :--- |
| **`amount`** | `1d.kline.amount` | `1d` / `price` | 成交金额。 |
| **`close`** | `1d.kline.close` | `1d` / `price` | 收盘价。 |
| **`close_hfq`** | `1d.kline.close_hfq` | `1d` / `price` | 收盘价的后复权序列。 |
| **`daily_kline_adj_factor`** | `1d.kline.adj_factor` | `1d` / `price` | 累计后复权因子。 |
| **`daily_kline_change`** | `1d.kline.change` | `1d` / `price` | 日行情涨跌额。 |
| **`daily_kline_high_hfq`** | `1d.kline.high_hfq` | `1d` / `price` | 最高价的后复权序列。 |
| **`daily_kline_low_hfq`** | `1d.kline.low_hfq` | `1d` / `price` | 最低价的后复权序列。 |
| **`daily_kline_open_hfq`** | `1d.kline.open_hfq` | `1d` / `price` | 开盘价的后复权序列。 |
| **`daily_kline_pct_chg`** | `1d.kline.pct_chg` | `1d` / `price` | 当日涨跌幅。 |
| **`daily_kline_real_pre_close`** | `1d.kline.real_pre_close` | `1d` / `price` | 上一个有效交易日的收盘价（未复权）。 |
| **`daily_kline_vol_hfq`** | `1d.kline.vol_hfq` | `1d` / `price` | 成交量的后复权序列。 |
| **`high`** | `1d.kline.high` | `1d` / `price` | 最高价。 |
| **`low`** | `1d.kline.low` | `1d` / `price` | 最低价。 |
| **`open`** | `1d.kline.open` | `1d` / `price` | 开盘价。 |
| **`pre_close`** | `1d.kline.pre_close` | `1d` / `price` | 除权参考前收盘价。 |
| **`volume`** | `1d.kline.vol` | `1d` / `price` | 成交量。 |
| **`BaseUnivMask`** | `mask.stock.BaseUnivMask` | `1d` / `universe` | 基础股票池掩码。 |
| **`LimitMask`** | `mask.stock.LimitMask` | `1d` / `universe` | 涨跌停交易限制掩码。 |
| **`stock_mask_down_limit`** | `mask.stock.down_limit` | `1d` / `universe` | 当日跌停参考价。 |
| **`stock_mask_nonewstockmask`** | `mask.stock.NoNewStockMask` | `1d` / `universe` | 上市超过 60 个自然日的股票掩码。 |
| **`stock_mask_risk_level`** | `mask.stock.risk_level` | `1d` / `universe` | 当日风险警示等级：0 无警示，1 ST，2 *ST（含高风险警示），3 PT。 |
| **`stock_mask_ststock`** | `mask.stock.STStock` | `1d` / `universe` | 剔除特殊处理股票的掩码。 |
| **`stock_mask_up_limit`** | `mask.stock.up_limit` | `1d` / `universe` | 当日涨停参考价。 |
| **`StockListedDays`** | `mask.stock.StockListedDays` | `1d` / `universe` | 股票上市后的自然日天数。 |
| **`SuspendStock`** | `mask.stock.SuspendStock` | `1d` / `universe` | 非停牌股票掩码。 |
| **`trade_universe_asharebj`** | `mask.universe.AshareBJ` | `1d` / `universe` | 北交所股票为 1，其余为 NaN。 |
| **`trade_universe_asharecyb`** | `mask.universe.AshareCYB` | `1d` / `universe` | 创业板股票为 1，其余为 NaN。 |
| **`trade_universe_asharedelistrisk`** | `mask.universe.AshareDelistRisk` | `1d` / `universe` | 退市风险警示股票池：当日简称为 *ST 的股票为 1，其余为 NaN。 |
| **`trade_universe_asharesh`** | `mask.universe.AshareSH` | `1d` / `universe` | 上交所 A 股为 1，其余为 NaN。 |
| **`trade_universe_asharesz`** | `mask.universe.AshareSZ` | `1d` / `universe` | 深交所 A 股为 1，其余为 NaN。 |
| **`trade_universe_ewm_amt`** | `mask.universe.ewm_amt` | `1d` / `universe` | 日成交额的 30 日半衰期指数加权均值。 |
| **`trade_universe_hs300`** | `mask.universe.HS300` | `1d` / `universe` | 沪深 300 成分股为 1，其余为 NaN。 |
| **`trade_universe_top1500`** | `mask.universe.TOP1500` | `1d` / `universe` | 按前一交易日平滑成交额排名的前 1500 只股票的股票池掩码。 |
| **`trade_universe_top3000`** | `mask.universe.TOP3000` | `1d` / `universe` | 按前一交易日平滑成交额排名的前 3000 只股票的股票池掩码。 |
| **`trade_universe_zz1000`** | `mask.universe.ZZ1000` | `1d` / `universe` | 中证 1000 成分股为 1，其余为 NaN。 |
| **`trade_universe_zz500`** | `mask.universe.ZZ500` | `1d` / `universe` | 中证 500 成分股为 1，其余为 NaN。 |
| **`trade_universe_zz800`** | `mask.universe.ZZ800` | `1d` / `universe` | 中证 800 成分股为 1，其余为 NaN。 |
| **`daily_fdm_dv_ttm`** | `1d.fdm.dv_ttm` | `1d` / `fundamental` |  |
| **`daily_fdm_floatshare`** | `1d.fdm.floatshare` | `1d` / `fundamental` | 已公告财报中的期末总股本。 |
| **`daily_fdm_mkt_cap`** | `1d.fdm.mkt_cap` | `1d` / `fundamental` | 上述股本乘当日未复权收盘价。 |
| **`daily_fdm_totalshare`** | `1d.fdm.totalshare` | `1d` / `fundamental` | 总股本。 |

### 5.3 Barra CNE5 风格因子与风险暴露 (Risk Models - CNE5 核心子集)

| 风格因子代码 | 平台内部字段 | 因子中文含义 | 合成与标准化算法口径 |
| :--- | :--- | :--- | :--- |
| **`barra_cne5_beta`** | `1d.barra_cne5.BETA` | CNE5 市场贝塔合成风格暴露。 | 由子因子 BETA 做截面标准化得到 |
| **`barra_cne5_btop`** | `1d.barra_cne5.BTOP` | CNE5 账面市值比合成风格暴露。 | 由子因子 BTOP 做截面标准化得到 |
| **`barra_cne5_earnyild`** | `1d.barra_cne5.EARNYILD` | CNE5 盈利收益率合成风格暴露。 | 由子因子 CETOP、ETOP、EPFWD 合成：各自截面标准化后加权求和，再做一次截面标准化 |
| **`barra_cne5_growth`** | `1d.barra_cne5.GROWTH` | CNE5 成长合成风格暴露。 | 由子因子 EGRO、SGRO、EGRLF、EGRSF 合成：各自截面标准化后加权求和，再做一次截面标准化 |
| **`barra_cne5_industry`** | `1d.barra_cne5.INDUSTRY` | CNE5 行业分类合成风格暴露。 | 申万 2021 一级行业代码，只在样本内的股票上有值 |
| **`barra_cne5_leverage`** | `1d.barra_cne5.LEVERAGE` | CNE5 杠杆合成风格暴露。 | 由子因子 MLEV、DTOA、BLEV 合成：各自截面标准化后加权求和，再做一次截面标准化 |
| **`barra_cne5_liquidty`** | `1d.barra_cne5.LIQUIDTY` | CNE5 流动性合成风格暴露。 | 由子因子 STOM、STOQ、STOA 合成：各自截面标准化后加权求和，再做一次截面标准化 |
| **`barra_cne5_momentum`** | `1d.barra_cne5.MOMENTUM` | CNE5 动量合成风格暴露。 | 由子因子 RSTR 做截面标准化得到 |
| **`barra_cne5_resvol`** | `1d.barra_cne5.RESVOL` | CNE5 残差波动合成风格暴露。 | 由子因子 DASTD、CMRA、HSIGMA 合成：各自截面标准化后加权求和，再做一次截面标准化 |
| **`barra_cne5_size`** | `1d.barra_cne5.SIZE` | CNE5 规模合成风格暴露。 | 由子因子 LNCAP 做截面标准化得到 |
| **`barra_cne5_sizenl`** | `1d.barra_cne5.SIZENL` | CNE5 非线性规模合成风格暴露。 | 由子因子 NLSIZE 做截面标准化得到 |
| **`sub_barra_cne5_beta`** | `1d.sub_barra_cne5.BETA` | CNE5 市场贝塔子因子。 | 已去极值（3 倍 MAD）并做截面标准化；只在上市满 180 个自然日、且在基础股票池内的股票上计算；具体输入和窗口与常见 Barra 定义不一定相同。 |
| **`sub_barra_cne5_blev`** | `1d.sub_barra_cne5.BLEV` | CNE5 账面杠杆子因子。 | 已去极值（3 倍 MAD）并做截面标准化；只在上市满 180 个自然日、且在基础股票池内的股票上计算；具体输入和窗口与常见 Barra 定义不一定相同。 |
| **`sub_barra_cne5_btop`** | `1d.sub_barra_cne5.BTOP` | CNE5 账面市值比子因子。 | 已去极值（3 倍 MAD）并做截面标准化；只在上市满 180 个自然日、且在基础股票池内的股票上计算；具体输入和窗口与常见 Barra 定义不一定相同。 |
| **`sub_barra_cne5_cetop`** | `1d.sub_barra_cne5.CETOP` | CNE5 现金盈利收益率子因子。 | 已去极值（3 倍 MAD）并做截面标准化；只在上市满 180 个自然日、且在基础股票池内的股票上计算；具体输入和窗口与常见 Barra 定义不一定相同。 |
| **`sub_barra_cne5_cmra`** | `1d.sub_barra_cne5.CMRA` | CNE5 累积区间收益范围子因子。 | 已去极值（3 倍 MAD）并做截面标准化；只在上市满 180 个自然日、且在基础股票池内的股票上计算；具体输入和窗口与常见 Barra 定义不一定相同。 |
| **`sub_barra_cne5_dastd`** | `1d.sub_barra_cne5.DASTD` | CNE5 日收益波动子因子。 | 已去极值（3 倍 MAD）并做截面标准化；只在上市满 180 个自然日、且在基础股票池内的股票上计算；具体输入和窗口与常见 Barra 定义不一定相同。 |
| **`sub_barra_cne5_dtoa`** | `1d.sub_barra_cne5.DTOA` | CNE5 负债资产比子因子。 | 已去极值（3 倍 MAD）并做截面标准化；只在上市满 180 个自然日、且在基础股票池内的股票上计算；具体输入和窗口与常见 Barra 定义不一定相同。 |
| **`sub_barra_cne5_egrlf`** | `1d.sub_barra_cne5.EGRLF` | CNE5 长期预测盈利增长子因子。 | 已去极值（3 倍 MAD）并做截面标准化；只在上市满 180 个自然日、且在基础股票池内的股票上计算；具体输入和窗口与常见 Barra 定义不一定相同。 |
| **`sub_barra_cne5_egro`** | `1d.sub_barra_cne5.EGRO` | CNE5 历史盈利增长子因子。 | 已去极值（3 倍 MAD）并做截面标准化；只在上市满 180 个自然日、且在基础股票池内的股票上计算；具体输入和窗口与常见 Barra 定义不一定相同。 |
| **`sub_barra_cne5_egrsf`** | `1d.sub_barra_cne5.EGRSF` | CNE5 短期预测盈利增长子因子。 | 已去极值（3 倍 MAD）并做截面标准化；只在上市满 180 个自然日、且在基础股票池内的股票上计算；具体输入和窗口与常见 Barra 定义不一定相同。 |
| **`sub_barra_cne5_epfwd`** | `1d.sub_barra_cne5.EPFWD` | CNE5 预测盈利收益率子因子。 | 已去极值（3 倍 MAD）并做截面标准化；只在上市满 180 个自然日、且在基础股票池内的股票上计算；具体输入和窗口与常见 Barra 定义不一定相同。 |
| **`sub_barra_cne5_etop`** | `1d.sub_barra_cne5.ETOP` | CNE5 历史盈利收益率子因子。 | 已去极值（3 倍 MAD）并做截面标准化；只在上市满 180 个自然日、且在基础股票池内的股票上计算；具体输入和窗口与常见 Barra 定义不一定相同。 |
| **`sub_barra_cne5_hsigma`** | `1d.sub_barra_cne5.HSIGMA` | CNE5 历史残差波动子因子。 | 已去极值（3 倍 MAD）并做截面标准化；只在上市满 180 个自然日、且在基础股票池内的股票上计算；具体输入和窗口与常见 Barra 定义不一定相同。 |
| **`sub_barra_cne5_lncap`** | `1d.sub_barra_cne5.LNCAP` | CNE5 对数规模子因子。 | 已去极值（3 倍 MAD）并做截面标准化；只在上市满 180 个自然日、且在基础股票池内的股票上计算；具体输入和窗口与常见 Barra 定义不一定相同。 |
| **`sub_barra_cne5_mlev`** | `1d.sub_barra_cne5.MLEV` | CNE5 市场杠杆子因子。 | 已去极值（3 倍 MAD）并做截面标准化；只在上市满 180 个自然日、且在基础股票池内的股票上计算；具体输入和窗口与常见 Barra 定义不一定相同。 |
| **`sub_barra_cne5_nlsize`** | `1d.sub_barra_cne5.NLSIZE` | CNE5 非线性规模子因子。 | 已去极值（3 倍 MAD）并做截面标准化；只在上市满 180 个自然日、且在基础股票池内的股票上计算；具体输入和窗口与常见 Barra 定义不一定相同。 |
| **`sub_barra_cne5_rstr`** | `1d.sub_barra_cne5.RSTR` | CNE5 相对强弱动量子因子。 | 已去极值（3 倍 MAD）并做截面标准化；只在上市满 180 个自然日、且在基础股票池内的股票上计算；具体输入和窗口与常见 Barra 定义不一定相同。 |
| **`sub_barra_cne5_sgro`** | `1d.sub_barra_cne5.SGRO` | CNE5 销售增长子因子。 | 已去极值（3 倍 MAD）并做截面标准化；只在上市满 180 个自然日、且在基础股票池内的股票上计算；具体输入和窗口与常见 Barra 定义不一定相同。 |
| **`sub_barra_cne5_stoa`** | `1d.sub_barra_cne5.STOA` | CNE5 年度换手子因子。 | 已去极值（3 倍 MAD）并做截面标准化；只在上市满 180 个自然日、且在基础股票池内的股票上计算；具体输入和窗口与常见 Barra 定义不一定相同。 |
| **`sub_barra_cne5_stom`** | `1d.sub_barra_cne5.STOM` | CNE5 月度换手子因子。 | 已去极值（3 倍 MAD）并做截面标准化；只在上市满 180 个自然日、且在基础股票池内的股票上计算；具体输入和窗口与常见 Barra 定义不一定相同。 |
| **`sub_barra_cne5_stoq`** | `1d.sub_barra_cne5.STOQ` | CNE5 季度换手子因子。 | 已去极值（3 倍 MAD）并做截面标准化；只在上市满 180 个自然日、且在基础股票池内的股票上计算；具体输入和窗口与常见 Barra 定义不一定相同。 |

### 5.4 日内分钟微结构特征精选 (Minute Bars - 1m / 5m)

| 分钟特征代码 | 频率 | 平台内部字段 | 描述说明 |
| :--- | :---: | :--- | :--- |
| **`f_1m_bars_grid1mbar_close`** | `1m` | `1m.bars.Grid1mBar.close` | 窗口内最后有效成交价。 |
| **`f_1m_bars_grid1mbar_cumsum_amt`** | `1m` | `1m.bars.Grid1mBar.cumsum_amt` | 窗口末快照的当日累计成交额。 |
| **`f_1m_bars_grid1mbar_cumsum_num`** | `1m` | `1m.bars.Grid1mBar.cumsum_num` | 窗口末快照的当日累计成交笔数。 |
| **`f_1m_bars_grid1mbar_cumsum_vol`** | `1m` | `1m.bars.Grid1mBar.cumsum_vol` | 窗口末快照的当日累计成交量。 |
| **`f_1m_bars_grid1mbar_first_total_ask_num`** | `1m` | `1m.bars.Grid1mBar.first_total_ask_num` | 窗口内最后一个快照的委托卖出总笔数。 |
| **`f_1m_bars_grid1mbar_first_total_ask_vol`** | `1m` | `1m.bars.Grid1mBar.first_total_ask_vol` | 窗口内最后一个快照的委托卖出总量。 |
| **`f_1m_bars_grid1mbar_first_total_bid_num`** | `1m` | `1m.bars.Grid1mBar.first_total_bid_num` | 窗口内最后一个快照的委托买入总笔数。 |
| **`f_1m_bars_grid1mbar_first_total_bid_vol`** | `1m` | `1m.bars.Grid1mBar.first_total_bid_vol` | 窗口内最后一个快照的委托买入总量。 |
| **`f_1m_bars_grid1mbar_first_weighted_ask_prc`** | `1m` | `1m.bars.Grid1mBar.first_weighted_ask_prc` | 窗口内最后一个快照的加权平均委卖价。 |
| **`f_1m_bars_grid1mbar_first_weighted_bid_prc`** | `1m` | `1m.bars.Grid1mBar.first_weighted_bid_prc` | 窗口内最后一个快照的加权平均委买价。 |
| **`f_1m_bars_grid1mbar_high`** | `1m` | `1m.bars.Grid1mBar.high` | 窗口内快照所携带 high 字段的最大值。 |
| **`f_1m_bars_grid1mbar_low`** | `1m` | `1m.bars.Grid1mBar.low` | 窗口内快照所携带 low 字段的最小值。 |
| **`f_1m_bars_grid1mbar_number`** | `1m` | `1m.bars.Grid1mBar.number` | 窗口内成交笔数。 |
| **`f_1m_bars_grid1mbar_turnover`** | `1m` | `1m.bars.Grid1mBar.turnover` | 窗口内成交额。 |
| **`f_1m_bars_grid1mbar_volume`** | `1m` | `1m.bars.Grid1mBar.volume` | 窗口内成交量。 |
| **`f_1m_bars_grid1mbar_vwap`** | `1m` | `1m.bars.Grid1mBar.vwap` | 窗口内成交量加权均价。 |
| **`f_1m_bars_grid1mtradebar_amount`** | `1m` | `1m.bars.Grid1mTradeBar.amount` | 窗口内成交额。 |
| **`f_1m_bars_grid1mtradebar_buy_amount`** | `1m` | `1m.bars.Grid1mTradeBar.buy_amount` | 窗口内买方成交额。 |
| **`f_1m_bars_grid1mtradebar_buy_number`** | `1m` | `1m.bars.Grid1mTradeBar.buy_number` | 窗口内买方成交笔数。 |
| **`f_1m_bars_grid1mtradebar_buy_volume`** | `1m` | `1m.bars.Grid1mTradeBar.buy_volume` | 窗口内买方成交量。 |
| **`f_1m_bars_grid1mtradebar_close`** | `1m` | `1m.bars.Grid1mTradeBar.close` | 窗口内收盘价。 |
| **`f_1m_bars_grid1mtradebar_high`** | `1m` | `1m.bars.Grid1mTradeBar.high` | 窗口内最高价。 |
| **`f_1m_bars_grid1mtradebar_low`** | `1m` | `1m.bars.Grid1mTradeBar.low` | 窗口内最低价。 |
| **`f_1m_bars_grid1mtradebar_number`** | `1m` | `1m.bars.Grid1mTradeBar.number` | 窗口内成交笔数。 |
| **`f_1m_bars_grid1mtradebar_open`** | `1m` | `1m.bars.Grid1mTradeBar.open` | 窗口内开盘价。 |
| **`f_1m_bars_grid1mtradebar_sell_amount`** | `1m` | `1m.bars.Grid1mTradeBar.sell_amount` | 窗口内卖方成交额。 |
| **`f_1m_bars_grid1mtradebar_sell_number`** | `1m` | `1m.bars.Grid1mTradeBar.sell_number` | 窗口内卖方成交笔数。 |
| **`f_1m_bars_grid1mtradebar_sell_volume`** | `1m` | `1m.bars.Grid1mTradeBar.sell_volume` | 窗口内卖方成交量。 |
| **`f_1m_bars_grid1mtradebar_volume`** | `1m` | `1m.bars.Grid1mTradeBar.volume` | 窗口内成交量。 |
| **`f_1m_bars_grid1mtradebar_vwap`** | `1m` | `1m.bars.Grid1mTradeBar.vwap` | 窗口内成交量加权均价。 |
| *...（其余 132 个分钟级特征）* | - | - | 完整数据请参阅平台变量目录 |

---

## 六、与本地平台 (FactorLibrary) 算子与因子映射指南

为了将 FactorLab 云端的因子迁移至我们本地的 **Rust 高性能计算内核 (`factorlib_core`)** 与 **Python 向量化回测状态机**，以下为语法映射与实现指引：

### 6.1 核心算子对齐表 (Cloud FactorLab <-> Local FactorLibrary)

| 功能分类 | FactorLab 云端表达式 | 本地 Rust 内核实现 (`factorlib_core::ops`) | 本地 Python 参考实现 (`ops_ref.py`) |
| :--- | :--- | :--- | :--- |
| **时序滑动均值** | `ts_mean(x, 20)` | `ops::ts::ts_mean(x, 20)` (Welford 递推) | `ops_ref.ts_mean(x, 20)` |
| **时序滚动相关系数** | `ts_corr(x, y, 20)` | `ops::ts::ts_corr(x, y, 20)` | `ops_ref.ts_corr(x, y, 20)` |
| **时序滚动标准差** | `ts_std(x, 20)` | `ops::ts::ts_std(x, 20)` | `ops_ref.ts_std(x, 20)` |
| **截面 Z-Score** | `cs_zscore(x)` | `ops::cs::cs_zscore(x)` | `ops_ref.cs_zscore(x)` |
| **截面百分位排名** | `cs_rank(x)` | `ops::cs::cs_rank(x)` | `ops_ref.cs_rank(x)` |
| **行业均值去中心化** | `indNeut(ind)` | `ops::neutralize::ind_neutralize(x, ind)` | `ops_ref.ind_neutralize(x, ind)` |
| **MAD 去极值** | `MAD(3.0)` | `ops::clip::mad_winsorize(x, 3.0)` | `ops_ref.mad_winsorize(x, 3.0)` |
| **线性递减加权衰减**| `emadecay(window)` | `ops::ts::linear_decay(x, window)` | `ops_ref.linear_decay(x, window)` |

### 6.2 明星因子本地一键移植代码模板

以 FactorLab 综合表现最佳的 **`Tech_Trend_VP_Constrained`** (IC=0.0264, RankIC=0.0329, RankIC_IR=0.4161) 为例，在本地 Python/Rust 引擎中的计算流如下：

```python
import numpy as np
from factorlib.core import ops_ref as ops
# 或者直接调用 Rust 加速内核:
# import factorlib_core as flc

def compute_tech_trend_vp_constrained(close_hfq, high, low, amount, ind_code, risk_matrix):
    """
    FactorLab 'Tech_Trend_VP_Constrained' 本地复现版本
    """
    # 1. 动量偏离项
    ma20 = ops.ts_mean(close_hfq, 20)
    term1 = ops.cs_zscore((close_hfq - ma20) / ma20)
    
    # 2. 价格截面位置项 (K线振幅占比)
    term2 = ops.cs_zscore((close_hfq - low) / (high - low + 0.001))
    
    # 3. 量价相关性与放量共振项
    corr_pv = ops.ts_corr(close_hfq, amount, 20)
    amt_ratio = amount / (ops.ts_mean(amount, 5) + 0.001)
    term3 = ops.cs_zscore(corr_pv * amt_ratio)
    
    # 4. 反向合成
    raw_factor = -1.0 * (term1 + term2 + term3)
    
    # 5. 后处理流水线 (MAD -> 行业中性化 -> 风格中性化 -> 截面标准化 -> 衰减)
    step1 = ops.mad_winsorize(raw_factor, 2.5)
    step2 = ops.ind_neutralize(step1, ind_code)
    step3 = ops.multi_residualize(step2, risk_matrix)  # 剥离 size, beta, liquidty, resvol
    step4 = ops.cs_zscore(step3)
    final_factor = ops.linear_decay(step4, 3)
    
    return final_factor
```
