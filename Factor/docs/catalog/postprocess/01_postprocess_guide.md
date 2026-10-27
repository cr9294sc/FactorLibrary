# FactorLab 因子后处理算子指南 (Postprocess Operators)

在 FactorLab 中，后处理（Postprocess）是在因子原始矩阵计算完成后、提交回测评估前执行的标准流水线阶段。
后处理通常用于**数据过滤、平滑、去极值、标准化与风格中性化**。

> [!IMPORTANT]
> **平台自动注入规则**：
> 后处理算子在平台中通过指定算子名和标量参数（如 `window`、`quant`）配置，平台会在运行时**自动注入所依赖的掩码与暴露矩阵**（例如 `LimitMask`、`BaseUnivMask`、行业暴露代码等）。编写因子时无需手动复制原生函数的全量输入参数。

**全量收录**：`12` 个后处理算子。

---

## 后处理算子汇总表

| 算子名 (Operator) | 阶段分类 | 自动注入依赖 (Implicit Dependencies) | 核心功能摘要 |
| :--- | :---: | :--- | :--- |
| [trdMask](#trdmask) | 因子后处理 | `LimitMask` | 按可交易条件过滤输入，不可交易位置返回 NaN。 |
| [baseUniMask](#baseunimask) | 因子后处理 | `BaseUnivMask` | 按基础股票池条件过滤输入，池外位置返回 NaN。 |
| [dropNewStock](#dropnewstock) | 因子后处理 | `StockListedDays` | 根据上市时长过滤新股，不满足 min_dates 的位置返回 NaN。 |
| [decay](#decay) | 因子后处理 | 无 | 完整窗口内的滚动算术均值。 |
| [emadecay](#emadecay) | 因子后处理 | 无 | 按旧到新 1..N 权重线性加权的滚动衰减，不是指数衰减。 |
| [MAD](#mad) | 因子后处理 | 无 | 按截面中位数和经 1.4826 缩放的 MAD 截断极值。 |
| [winsor](#winsor) | 因子后处理 | 无 | 按两端 quant 分位数对极端值进行缩尾处理。 |
| [zScore](#zscore) | 因子后处理 | 无 | 对每个交易时点的股票截面执行 Z-score 标准化。 |
| [rank](#rank) | 因子后处理 | 无 | 逐时点截面百分位排名：从 1 开始的稳定原始名次除以有效样本数。 |
| [indNeut](#indneut) | 因子后处理 | 无 | 在每个时点按 x2 行业代码对 x1 减去同行业非缺失均值。 |
| [riskNeut](#riskneut) | 因子后处理 | 无 | 对因子执行风险暴露中性化。 |
| [riskNeutMulti](#riskneutmulti) | 因子后处理 | 无 | 以 x1 为被解释变量、后续矩阵为暴露，做无截距截面多元回归并返回残差。 |

---

## 后处理算子详细规范

### <a id="trdmask"></a> `trdMask`

- **功能概述**：按可交易条件过滤输入，不可交易位置返回 NaN。
- **计算数学公式**：`x1 * x2`
- **计算轴 (Axis)**：`stocks`
- **隐式注入依赖**：`LimitMask`
- **输出数据类型与形状**：`float64`，保持 `[T × S]`
- **缺失值与非有限值策略**：逐元素相乘：1 保留值，NaN 掩码使输出 NaN，0 掩码输出 0（不是 NaN）。原生函数不读取或自动生成掩码。
- **后处理配置提示**：后处理通过 OP 配置传标量参数，平台自动注入声明的掩码/暴露矩阵；不要复制原生函数的全部 tensor 参数。

**运行时执行验证用例**：
- **调用表达式**：`trdMask(x1,x2)`
- **输出验证级别**：`executed_against_locked_native_runtime`

---

### <a id="baseunimask"></a> `baseUniMask`

- **功能概述**：按基础股票池条件过滤输入，池外位置返回 NaN。
- **计算数学公式**：`x1 * x2`
- **计算轴 (Axis)**：`stocks`
- **隐式注入依赖**：`BaseUnivMask`
- **输出数据类型与形状**：`float64`，保持 `[T × S]`
- **缺失值与非有限值策略**：逐元素相乘：1 保留值，NaN 掩码使输出 NaN，0 掩码输出 0（不是 NaN）。原生函数不读取或自动生成掩码。
- **后处理配置提示**：后处理通过 OP 配置传标量参数，平台自动注入声明的掩码/暴露矩阵；不要复制原生函数的全部 tensor 参数。

**运行时执行验证用例**：
- **调用表达式**：`baseUniMask(x1,x2)`
- **输出验证级别**：`executed_against_locked_native_runtime`

---

### <a id="dropnewstock"></a> `dropNewStock`

- **功能概述**：根据上市时长过滤新股，不满足 min_dates 的位置返回 NaN。
- **计算数学公式**：`x1 if x2 >= min_dates else NaN`
- **计算轴 (Axis)**：`stocks`
- **隐式注入依赖**：`StockListedDays`
- **输出数据类型与形状**：`float64`，保持 `[T × S]`
- **缺失值与非有限值策略**：上市天数 x2 为 NaN 或 x2 < min_dates 时输出 NaN；恰好等于阈值保留 x1。x1 原有 NaN 保留。
- **后处理配置提示**：后处理通过 OP 配置传标量参数，平台自动注入声明的掩码/暴露矩阵；不要复制原生函数的全部 tensor 参数。

**配置参数 (Arguments)**：

| 参数名 | 说明 |
| :--- | :--- |
| `{'name': 'minimum_days', 'kind': 'integer', 'default': 60, 'minimum': 1}` | 后处理调节参数 |

**运行时执行验证用例**：
- **调用表达式**：`dropNewStock(x1,x2,3)`
- **输出验证级别**：`executed_against_locked_native_runtime`

---

### <a id="decay"></a> `decay`

- **功能概述**：完整窗口内的滚动算术均值。
- **计算数学公式**：`sum(valid x)/count(valid x)`
- **计算轴 (Axis)**：`time`
- **隐式注入依赖**：无
- **输出数据类型与形状**：`float64`，保持 `[T × S]`
- **缺失值与非有限值策略**：统计时跳过 NaN，不自动剔除 Inf；没有非 NaN 样本则返回 NaN。显式滚动窗口不足时返回 NaN，不用更短窗口代替。
- **后处理配置提示**：后处理通过 OP 配置传标量参数，平台自动注入声明的掩码/暴露矩阵；不要复制原生函数的全部 tensor 参数。

**配置参数 (Arguments)**：

| 参数名 | 说明 |
| :--- | :--- |
| `{'name': 'window', 'kind': 'integer', 'default': 5, 'minimum': 1}` | 后处理调节参数 |

**运行时执行验证用例**：
- **调用表达式**：`decay(x1,3)`
- **输出验证级别**：`executed_against_locked_native_runtime`

---

### <a id="emadecay"></a> `emadecay`

- **功能概述**：按旧到新 1..N 权重线性加权的滚动衰减，不是指数衰减。
- **计算数学公式**：`sum(i*x_i for i=1..N, valid) / sum(i for i=1..N, valid)`
- **计算轴 (Axis)**：`time`
- **隐式注入依赖**：无
- **输出数据类型与形状**：`float64`，保持 `[T × S]`
- **缺失值与非有限值策略**：NaN 所在位置权重剔除，剩余权重重新归一化。
- **后处理配置提示**：后处理通过 OP 配置传标量参数，平台自动注入声明的掩码/暴露矩阵；不要复制原生函数的全部 tensor 参数。

**配置参数 (Arguments)**：

| 参数名 | 说明 |
| :--- | :--- |
| `{'name': 'window', 'kind': 'integer', 'default': 5, 'minimum': 1}` | 后处理调节参数 |

**运行时执行验证用例**：
- **调用表达式**：`emadecay(x1,3)`
- **输出验证级别**：`executed_against_locked_native_runtime`

---

### <a id="mad"></a> `MAD`

- **功能概述**：按截面中位数和经 1.4826 缩放的 MAD 截断极值。
- **计算数学公式**：`clip(x, median(x)-n*1.4826*MAD, median(x)+n*1.4826*MAD)`
- **计算轴 (Axis)**：`stocks`
- **隐式注入依赖**：无
- **输出数据类型与形状**：`float64`，保持 `[T × S]`
- **缺失值与非有限值策略**：用有限值的下中位数及绝对偏差下中位数计算边界；NaN 保留，Inf 在有边界时被截断；无有限值时原样返回。
- **后处理配置提示**：后处理通过 OP 配置传标量参数，平台自动注入声明的掩码/暴露矩阵；不要复制原生函数的全部 tensor 参数。

**配置参数 (Arguments)**：

| 参数名 | 说明 |
| :--- | :--- |
| `{'name': 'threshold', 'kind': 'number', 'default': 3.0, 'minimum': 0.0}` | 后处理调节参数 |

**运行时执行验证用例**：
- **调用表达式**：`MAD(x1,2)`
- **输出验证级别**：`executed_against_locked_native_runtime`

---

### <a id="winsor"></a> `winsor`

- **功能概述**：按两端 quant 分位数对极端值进行缩尾处理。
- **计算数学公式**：`return torch.clip(x,min = f,max = u)`
- **计算轴 (Axis)**：`stocks`
- **隐式注入依赖**：无
- **输出数据类型与形状**：`float64`，保持 `[T × S]`
- **缺失值与非有限值策略**：用有限值计算两端线性分位点，再截断非 NaN 输入（含 Inf）；NaN 保留。全无有限值时保持原输入。
- **后处理配置提示**：后处理通过 OP 配置传标量参数，平台自动注入声明的掩码/暴露矩阵；不要复制原生函数的全部 tensor 参数。

**配置参数 (Arguments)**：

| 参数名 | 说明 |
| :--- | :--- |
| `{'name': 'quantile', 'kind': 'number', 'default': 0.025, 'minimum': 0.0}` | 后处理调节参数 |

**运行时执行验证用例**：
- **调用表达式**：`winsor(x1,0.25)`
- **输出验证级别**：`executed_against_locked_native_runtime`

---

### <a id="zscore"></a> `zScore`

- **功能概述**：对每个交易时点的股票截面执行 Z-score 标准化。
- **计算数学公式**：`(x1 - mean(x1,axis=stocks)) / std(x1,axis=stocks,ddof=0)`
- **计算轴 (Axis)**：`stocks`
- **隐式注入依赖**：无
- **输出数据类型与形状**：`float64`，保持 `[T × S]`
- **缺失值与非有限值策略**：均值和总体标准差跳过 NaN，不剔除 Inf；原有 NaN 保留，常数截面除以零产生 NaN。
- **后处理配置提示**：后处理通过 OP 配置传标量参数，平台自动注入声明的掩码/暴露矩阵；不要复制原生函数的全部 tensor 参数。

**运行时执行验证用例**：
- **调用表达式**：`zScore(x1)`
- **输出验证级别**：`executed_against_locked_native_runtime`

---

### <a id="rank"></a> `rank`

- **功能概述**：逐时点截面百分位排名：从 1 开始的稳定原始名次除以有效样本数。
- **计算数学公式**：`cs_rank(x1) / count(finite(x1)); range=(0,1]`
- **计算轴 (Axis)**：`stocks`
- **隐式注入依赖**：无
- **输出数据类型与形状**：`float64`，保持 `[T × S]`
- **缺失值与非有限值策略**：仅排列有限值；NaN/Inf 位置返回 NaN。先除以总体标准差并舍入 6 位，标准差 < 1e-6 时整条计算轴返回 NaN；并列值按原轴顺序依次分配不同名次。
- **后处理配置提示**：后处理通过 OP 配置传标量参数，平台自动注入声明的掩码/暴露矩阵；不要复制原生函数的全部 tensor 参数。

**运行时执行验证用例**：
- **调用表达式**：`rank(x1)`
- **输出验证级别**：`executed_against_locked_native_runtime`

---

### <a id="indneut"></a> `indNeut`

- **功能概述**：在每个时点按 x2 行业代码对 x1 减去同行业非缺失均值。
- **计算数学公式**：`x1 - mean(x1 of stocks with same positive x2 code)`
- **计算轴 (Axis)**：`stocks`
- **隐式注入依赖**：无
- **输出数据类型与形状**：`float64`，保持 `[T × S]`
- **缺失值与非有限值策略**：x1 为 NaN 时保留 NaN；x2 缺失或 <=0 时保留原始 x1；有效行业内均值跳过 x1 的 NaN。
- **后处理配置提示**：后处理通过 OP 配置传标量参数，平台自动注入声明的掩码/暴露矩阵；不要复制原生函数的全部 tensor 参数。

**配置参数 (Arguments)**：

| 参数名 | 说明 |
| :--- | :--- |
| `{'name': 'industry', 'kind': 'variable', 'default': 'CNE5S_IND'}` | 后处理调节参数 |

**运行时执行验证用例**：
- **调用表达式**：`indNeut(x1,x2)`
- **输出验证级别**：`executed_against_locked_native_runtime`

---

### <a id="riskneut"></a> `riskNeut`

- **功能概述**：对因子执行风险暴露中性化。
- **计算数学公式**：`return resi`
- **计算轴 (Axis)**：`stocks`
- **隐式注入依赖**：无
- **输出数据类型与形状**：`float64`，保持 `[T × S]`
- **缺失值与非有限值策略**：beta 仅用 x1/x2 同时有限的样本，alpha 的两个均值分别跳过各自 NaN；x2 缺失时保留 x1，x1 缺失仍为 NaN；暴露方差不足时可返回 NaN。
- **后处理配置提示**：后处理通过 OP 配置传标量参数，平台自动注入声明的掩码/暴露矩阵；不要复制原生函数的全部 tensor 参数。

**配置参数 (Arguments)**：

| 参数名 | 说明 |
| :--- | :--- |
| `{'name': 'exposures', 'kind': 'variables', 'default': ['CNE5S_SIZE']}` | 后处理调节参数 |

**运行时执行验证用例**：
- **调用表达式**：`riskNeut(x1,x2)`
- **输出验证级别**：`executed_against_locked_native_runtime`

---

### <a id="riskneutmulti"></a> `riskNeutMulti`

- **功能概述**：以 x1 为被解释变量、后续矩阵为暴露，做无截距截面多元回归并返回残差。
- **计算数学公式**：`beta = solve(X.T @ W @ X + 1e-8*I, X.T @ W @ y); residual = y - X @ beta; W=I for OLS`
- **计算轴 (Axis)**：`stocks`
- **隐式注入依赖**：无
- **输出数据类型与形状**：`float64`，保持 `[T × S]`
- **缺失值与非有限值策略**：拟合使用响应和所有解释变量均非 NaN 的完整样本；WLS 还要求乘以 sqrt(weight) 后非 NaN，负权重被排除。正规方程对角线加 1e-8；无法求解时输出 NaN。残差在原样本上计算，因此响应或解释变量缺失的位置返回 NaN。
- **后处理配置提示**：后处理通过 OP 配置传标量参数，平台自动注入声明的掩码/暴露矩阵；不要复制原生函数的全部 tensor 参数。

**配置参数 (Arguments)**：

| 参数名 | 说明 |
| :--- | :--- |
| `{'name': 'exposures', 'kind': 'variables', 'default': ['CNE5S_SIZE']}` | 后处理调节参数 |

**运行时执行验证用例**：
- **调用表达式**：`riskNeutMulti(x1,x2)`
- **输出验证级别**：`executed_against_locked_native_runtime`

---
