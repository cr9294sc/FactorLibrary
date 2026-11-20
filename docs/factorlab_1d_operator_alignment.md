# FactorLab MCP 1d 算子与变量全量对齐规约（Phase 3 P0）

> **文档定位**：`docs/factorlab_mcp_catalog.md`（云端 FactorLab / AresQuant 手册）与本地
> `rust/factorlib_core` + `python/factorlib/core/ops_ref.py` 的**逐算子对齐契约**。
> 本文件是「算子/变量全量对齐」的唯一验收依据：任何算子语义、边界或偏差都必须在此登记。
>
> **对齐状态**：FactorLab MCP 第四章 **106 / 106** 个 1d 表达式算子全部落地；
> Rust 内核注册算子总数 **128**（含 22 个原生扩展与别名目标），Python 参考实现 **128 / 128** 一对一镜像。

---

## 1. 对齐总览

| 维度 | 云端 FactorLab MCP | 本地 FactorLibrary | 状态 |
|---|---|---|---|
| 1d 表达式算子 | 106 | 106（128 个注册项含别名/扩展） | ✅ 全量 |
| 后处理算子 | 12 | 12 (`ops/postprocess.rs`) | ✅ 全量 |
| 变量短名 | 759（含 1m/5m/报表/分析师） | 别名层 + `stocks/v2` 50 字段 + `lb_barra_v1` Barra/风险库 | ✅ 1d 子集全量 |
| 数值对账 | — | `max abs err < 1e-9`（Rust vs NumPy） | ✅ 全绿 |

**注册算子分类（128 项）**

| 类别 | 数量 | 算子 |
|---|---:|---|
| `elem_unary` | 20 | `neg, abs, sign, inv, slog, ssqrt, square, cube, sigmoid, tanh, relu, atan, exp, ceil, sign_power, unique_count, is_inf, is_nan, purify, zero_to_nan` |
| `elem_binary` | 8 | `add, sub, mul, div, max2, min2, hypot, copysign` |
| `elem_const` | 13 | `addc, subc, mulc, divc, spow, clip, fillna, round, perc_long, cs_quantile, cs_group, winsor, MAD` |
| `cmp_series` / `cmp_const` | 4 / 2 | `gt, lt, ge, le` / `gt_const, lt_const` |
| `logic_unary` / `logic_binary` | 1 / 2 | `not_cond` / `and_cond, or_cond` |
| `cond_ternary` | 1 | `where`（别名 `ifelse`） |
| `cs_unary` | 21 | `cs_rank, cs_zscore, cs_demean, cs_scale, cs_winsorize, cs_sum, cs_mean, cs_std, cs_var, cs_max, cs_min, cs_median, cs_nonan_num, cs_argmax, cs_argmin, cs_pct_rank, cs_dense_rank, cs_dense_pct_rank, cs_skew, cs_kurt, cs_mad` |
| `cs_binary` | 6 | `cs_neutralize, cs_beta, cs_corr, cs_cov, barra5_zScore, ts_dw` |
| `group_binary` | 3 | `group_demean, group_rank, group_zscore` |
| `ts_unary` | 25 | `delay, delta, pct_change, log_return, ts_sum, ts_mean, ts_std, ts_var, ts_min, ts_max, ts_argmin, ts_argmax, ts_rank, ts_zscore, ts_skew, ts_kurt, ts_ema, ts_decay_linear, ts_mad, ts_roll, ts_prod, ts_median, ts_nonan_num, ts_pct_rank, ts_group` |
| `ts_binary` | 5 | `ts_cov, ts_corr, ts_beta, ts_resid, ts_quantile` |
| `ts_full_history` | 9 | `rank_ts, pct_rank_ts, dense_rank_ts, pct_dense_rank_ts, ts_cumprod, ts_cumsum, ts_ffill, ts_dense_rank, ts_pct_dense_rank` |
| `cs_nary_regression` | 4 | `cs_multi_beta*, cs_multi_resi*, cs_multi_wls_beta*, cs_multi_wls_resi*` |
| `ts_nary_regression` | 4 | `ts_multi_beta*, ts_multi_resi*, ts_multi_wls_beta*, ts_multi_wls_resi*` |

> `*` 标记为**变参算子**（`OpSpec.variadic = true`），支持 K ≥ 1 个暴露矩阵。

---

## 2. FactorLab 106 算子 → 本地实现映射

### 2.1 逐元素运算（7）

| FactorLab | 本地实现 | 说明 |
|---|---|---|
| `add(x1,x2)` | `elem::op_add` | 逐元素加 |
| `sub(x1,x2)` | `elem::op_sub` | 逐元素减 |
| `mul(x1,x2)` | `elem::op_mul` | 逐元素乘 |
| `div(x1,x2)` | `elem::op_div` | 安全除法：`|x2| < 1e-15` → NaN |
| `greater(x1,x2)` | `max2` 别名 | 逐元素取大（**不是比较**，与云端一致） |
| `less(x1,x2)` | `min2` 别名 | 逐元素取小（**不是比较**，与云端一致） |
| `power(x1,n)` | `spow` 别名 | 保号幂次：`sign(x1)·|x1|^n` |

### 2.2 条件与比较（2）

| FactorLab | 本地实现 | 说明 |
|---|---|---|
| `where(x1,x2,x3)` / `ifelse` | `elem::op_where` | 条件 `x1 > 0.5` 为真取 `x2`，否则 `x3`；`x1` 为 NaN 时输出 NaN |
| 中缀 `> < >= <=` | `gt/lt/ge/le` + `gt_const/lt_const` | 与常数比较时静态告警复权价格绝对尺度风险 |

### 2.3 数学变换（12）

| FactorLab | 本地实现 | 偏差与说明 |
|---|---|---|
| `abs(x1)` | `elem::op_abs` | — |
| `ceil(x1)` | `elem::op_ceil` | — |
| `clip(x1,lo,hi)` | `spow`/`clip(x, c)` | 本地为**对称截断** `[-|c|, |c|]`（云端为 `[lo,hi]` 双参数，DSL 常数槽位只接受单常数） |
| `exp(x1)` | `elem::op_exp` | 上溢 `+Inf` 归一为 NaN（平台 NaN 策略） |
| `log(x1)` | `slog` 别名 | 有符号安全对数 `sign·ln(1+|x|)`；**平台约定**，与云端 `ln` 且 `x<=0 → NaN` 不同口径，见 §4.3 |
| `perc_long(x1,p)` | `elem::op_perc_long` | 减截面 `p` 分位后正值保留，负值统一替换为 `-Σ正/负计数`，零净额多空 |
| `round(x1,d)` | `elem::op_round_decimals` | 向偶数舍入（`round_ties_even`），`d` 截断至 `[-15,15]` |
| `scale(x1)` | `cs_scale` 别名 | 按截面绝对值之和缩放 |
| `sign(x1)` | `elem::op_sign` | — |
| `sign_power(x1)` | `elem::op_sign_power` | `sign(x1)·x1²` |
| `sqrt(x1)` | `ssqrt` 别名 | 有符号安全平方根 `sign·√|x|`；**平台约定**，见 §4.3 |
| `unique_count(x1)` | `elem::op_unique_count` | 截面不同非 NaN 值个数（相邻值容差 `1e-12·max(1,|v|)` 合并） |

### 2.4 缺失值与极值处理（5）

| FactorLab | 本地实现 | 边界语义 |
|---|---|---|
| `fillna(x1,v)` | `elem::op_fillna` | 仅 NaN → `v`；`±Inf` 归一为 NaN（平台策略） |
| `is_inf(x1)` | `elem::op_is_inf` | `±Inf → 1.0`，有限 → `0.0`，NaN → NaN |
| `is_nan(x1)` | `elem::op_is_nan` | NaN → `1.0`，其余（含 `±Inf`）→ `0.0` |
| `purify(x1)` | `elem::op_purify` | `±Inf → NaN`；NaN → NaN |
| `zero_to_nan(x1)` | `elem::op_zero_to_nan` | 精确 `0.0 → NaN` |

> 上述 4 个探测/清洗算子**跳过 universe 掩码**（`UnaryOp::is_nan_probe()`），
> 以保留面板原始缺失信息。

### 2.5 截面运算（27）

| FactorLab | 本地实现 | 口径 |
|---|---|---|
| `cs_rank(x1)` | `cs::op_cs_rank` | 平均秩归一化到 `[0,1]`，`k=1 → 0.5` |
| `cs_pct_rank(x1)` / `rank` | `cs::op_cs_pct_rank` | 稳定原始名次 `(1..k)/k`（并列按列号先后，取值 `(0,1]`） |
| `cs_dense_rank(x1)` | `cs::op_cs_dense_rank` | 稠密名次 `1..D` |
| `cs_dense_pct_rank(x1)` | `cs::op_cs_dense_pct_rank` | `dense/D` |
| `cs_group(x1,g)` | `cs::op_cs_group` | `floor(pct_rank·g)` 截断至 `[0, g-1]` |
| `cs_sum/mean/max/min` | `cs::op_cs_sum/mean/max/min` | 有限值截面聚合，逐点广播（仅在原值有限处输出） |
| `cs_std/cs_var` | `cs::op_cs_std/var` | **总体矩 `ddof=0`**（与 `zScore` 后处理口径一致） |
| `cs_median` | `cs::op_cs_median` | **下中位数** `sorted[(k-1)/2]` |
| `cs_quantile(x1,q)` | `cs::op_cs_quantile` | 线性内插（`numpy.quantile` 默认口径） |
| `cs_mad` | `cs::op_cs_mad` | 相对**均值**的平均绝对偏差（非中位数 MAD） |
| `cs_skew` | `cs::op_cs_skew` | 总体矩 `m3/m2^1.5`，需 ≥3 有效样本 |
| `cs_kurt` | `cs::op_cs_kurt` | 超额峰度 `m4/m2²−3`，需 ≥4 有效样本 |
| `cs_nonan_num` | `cs::op_cs_nonan_num` | 非 NaN 计数 |
| `cs_argmax/cs_argmin` | `cs::op_cs_argmax/argmin` | **0 基列位置**（并列取最小列号） |
| `cs_beta` | `cs::op_cs_beta` | 中心化 OLS 斜率 `cov/var` |
| `cs_corr` | `cs::op_cs_corr` | Pearson，截断 `[-1,1]` |
| `cs_cov` | `cs::op_cs_cov` | 样本协方差 `ddof=1` |
| `cs_multi_beta/resi` | `multi::op_multi` | 变参；`(y, exp_1..exp_K [, intercept])` |
| `cs_multi_wls_beta/resi` | `multi::op_multi` | 变参；`(y, w, exp_1..exp_K [, intercept])` |
| `cs_zscore` / `zScore` | `cs::op_cs_zscore` | 样本标准差 `ddof=1`（与后处理 `zScore` 的 `ddof=0` 略有差异，见 §4.4） |

### 2.6 时序位移与差分（4）

| FactorLab | 本地实现 | 说明 |
|---|---|---|
| `ts_shift(x,d)` | `delay` 别名 | 移入 NaN |
| `ts_delta(x,d)` | `delta` 别名 | `x[t]-x[t-d]` |
| `ts_pct_chg(x,d)` | `pct_change` 别名 | `x[t]/x[t-d]-1`，分母 `<1e-15` → NaN |
| `ts_roll(x,d)` | `ts::op_ts_roll` | 循环移位 `out[t]=x[(t+T-d)%T]` |

### 2.7 时序窗口滚动（38）

| FactorLab | 本地实现 | 口径与偏差 |
|---|---|---|
| `ts_mean` / `decay` | `ts::op_ts_mean` | 双栈 Welford `O(1)`；**满窗**（`t≥d-1`）才输出，窗内 NaN 跳过 |
| `ts_std` | `ts::op_ts_std` | **样本标准差 `ddof=1`** |
| `ts_var` | `ts::op_ts_var` | **样本方差 `ddof=1`** |
| `ts_sum/min/max` | `ts::op_ts_sum/min/max` | 满窗；NaN 跳过 |
| `ts_median` | `ts::op_ts_median` | 满窗下中位数 |
| `ts_quantile(x,q,d)` | `ts::op_ts_quantile` | 满窗线性内插分位 |
| `ts_prod` | `ts::op_ts_prod` | 满窗有限值连乘 |
| `ts_nonan_num` | `ts::op_ts_nonan_num` | 满窗非 NaN 计数 |
| `ts_rank` | `ts::op_ts_rank` | 窗口平均秩归一化 `[0,1]`（`k=1 → 0.5`） |
| `ts_pct_rank` | `ts::op_ts_pct_rank` | 1 基平均名次 `/k`（取值 `(0,1]`） |
| `ts_group(x,g)` | `ts::op_ts_group` | 一周期收益率的因果历史百分位 `floor(pct·g)` |
| `ts_mad` | `ts::op_ts_mad` | 相对均值平均绝对偏差 |
| `ts_skew/kurt` | `ts::op_ts_skew/kurt` | 总体矩；`ts_kurt` 需 `d≥4` |
| `ts_argmax/argmin` | `ts::op_ts_argmax/argmin` | 窗口内相对偏移 `0..d-1`（并列取最早） |
| `ts_ema` | `ts::op_ts_ema` | `α=2/(d+1)` 递推 EMA（跨窗口连续，NaN 跳过不重置） |
| `emadecay` | `ts_decay_linear` 别名 | 权重 `1..W`（旧→新）线性加权 |
| `ts_corr/cov/beta/resid` | `ts::op_ts_corr/cov/beta/resid` | 双栈二元协矩 `O(1)`；`corr` 截断 `[-1,1]` |
| `ts_multi_beta/resi` | `multi::op_multi` | 扩张窗口因果多元 OLS；第 `dim+1` 个有效观测起输出 |
| `ts_multi_wls_beta/resi` | `multi::op_multi` | 同上，权重逐日因果归一化 |
| `rank_ts` | `ts::op_rank_ts` | **扩张窗口因果版**：名次仅由 `≤t` 历史决定（见 §4.1） |
| `pct_rank_ts` | `ts::op_pct_rank_ts` | 名次 `/ 有效样本数` |
| `dense_rank_ts` / `ts_dense_rank` | `ts::op_dense_rank_ts` / `op_ts_dense_rank` | 扩张窗口稠密名次 |
| `pct_dense_rank_ts` / `ts_pct_dense_rank` | `ts::op_pct_dense_rank_ts` / `op_ts_pct_dense_rank` | `dense/D` |
| `ts_cumsum` | `ts::op_ts_cumsum` | 有限值累加（NaN 位置输出 NaN，不入累加器） |
| `ts_cumprod` | `ts::op_ts_cumprod` | 有限值累乘 |
| `ts_ffill` | `ts::op_ts_ffill` | 前向填充 |
| `ts_dw(x1,x2)` | `ts::op_ts_dw` | 扩张窗口残差 Durbin-Watson（见 §4.2） |

### 2.8 因子复合处理（11）

| FactorLab | 本地实现 | 说明 |
|---|---|---|
| `barra5_zScore(x1,x2)` | `cs::op_barra5_zscore` | 以 `x2`（市值）为权重算截面均值；**等权总体标准差**标准化 |
| `MAD(x1,n)` | `postprocess::pp_mad` | 下中位数 + `1.4826` 缩放裁剪；`n` 取绝对值并强制有界（防负阈值 panic） |
| `winsor(x1,q)` | `postprocess::pp_winsor` | 线性内插分位 `[q, 1-q]` 裁剪；`q` 截断至 `[0,0.5]` |
| `zScore(x1)` | `cs_zscore` 别名 | 见 §4.4 |
| `rank(x1)` | `cs_pct_rank` 别名 | 见 §2.5 |
| `indNeut(x1,x2)` | `group_demean` 别名 | 按整数行业编码减去**同组非缺失均值** |
| `riskNeut(x1,x2)` | `cs_neutralize` 别名 | **带截距**单变量 OLS 残差 |
| `riskNeutMulti(x1,x2,…)` | `cs_multi_resi` 别名 | **无截距**多元 OLS 残差（变参，尾部常数可开截距） |
| `baseUniMask(x1,x2)` | `postprocess::pp_mask_mul` | 仅后处理链可用（需隐式 `in_universe_mask`），见 §4.5 |
| `trdMask(x1,x2)` | `postprocess::pp_mask_mul` | 仅后处理链可用（需隐式 `limit_mask`），见 §4.5 |
| `dropNewStock(x1,x2,min)` | `postprocess::pp_drop_new_stock` | 仅后处理链可用（需隐式 `listed_days`），见 §4.5 |

---

## 3. 变量短名 → 本地面板字段对齐

变量别名统一在 `python/factorlib/core/variable_aliases.py` 中登记，并在**表达式字符串层**
（`probe_factor` / `evaluate_factor` 入口）归一化为真实字段名，实现零内存放大的别名兼容。

### 3.1 米筐量价基本面域（`rq_v1` / `rq_v2`）

| FactorLab 变量短名 | 本地字段 | `rq_v1` | `rq_v2` |
|---|---|:--:|:--:|
| `daily_derived_returns_c2c_1d` | `c2c_ret` | ✅ | ✅ |
| `daily_derived_returns_intra` | `daily_derived_returns_intra` | ❌ | ✅ |
| `daily_kline_open_hfq` | `open_adj` | ✅ | ✅ |
| `daily_kline_high_hfq` / `low_hfq` | `high_adj` / `low_adj` | ✅ | ✅ |
| `daily_kline_adj_factor` | `ex_cum_factor` | ✅ | ✅ |
| `daily_kline_pct_chg` | `c2c_ret` | ✅ | ✅ |
| `daily_kline_real_pre_close` | `pre_close` | ❌ | ✅ |
| `daily_kline_change` | `close_change` | ❌ | ✅ |
| `daily_kline_vol_hfq` | `volume` | ✅ | ✅ |
| `daily_fdm_mkt_cap` | `mkt_cap` | ❌ | ✅ |
| `BaseUnivMask` | `in_universe_mask` | ✅ | ✅ |
| `LimitMask` | `limit_mask` | ✅ | ✅ |
| `StockListedDays` | `stock_listed_days` | ✅ | ✅ |
| `stock_mask_nonewstockmask` | `listed_days` | ✅ | ✅ |
| `index_weight_000300_sh/000905_sh/000852_sh` | 同名字段 | ❌ | ✅ |
| `trade_universe_hs300/zz500/zz1000` | 同名字段（成分内 `1.0`，其余 `NaN`） | ❌ | ✅ |

### 3.2 Barra 风险域（`lb_barra_v1`）

| FactorLab 变量短名 | 本地字段 |
|---|---|
| `barra_cne5_{beta,btop,earnyild,growth,leverage,liquidty,momentum,resvol,size,sizenl}` | `barra_{同后缀小写}` |
| `barra_cne5_industry` | `barra_industry` |
| `BETA/BTOP/EARNYILD/GROWTH/LEVERAGE/LIQUIDTY/MOMENTUM/RESVOL/SIZE/SIZENL` | `barra_{同后缀小写}` |
| `daily_fdm_mkt_cap` | `mkt_cap` |

### 3.3 显式拒绝别名（极性反转 / 口径不同 / 本地缺失）

`UNMAPPED_MCP_VARIABLES` 中登记的变量**不建立别名**，仅在使用时给出说明性告警，避免静默错误：

| 变量 | 拒绝原因 |
|---|---|
| `SuspendStock` | 云端为「非停牌掩码」(1=可交易)，本地 `is_suspended` 为 1=停牌，**极性相反** |
| `stock_mask_ststock` | 云端为「剔除 ST 掩码」，本地 `is_st` 为 1=ST，**极性相反** |
| `stock_mask_up_limit` / `down_limit` | 本地为绝对涨跌停参考价，与云端布尔掩码不同口径 |
| `stock_mask_risk_level` | 本地无风险警示等级 (0/1/2/3) |
| `trade_universe_ashare{bj,cyb,sh,sz}` / `asharedelistrisk` | 本地无板块/退市风险池掩码 |
| `trade_universe_{ewm_amt,top1500,top3000}` | 可由 `ts_mean(amount,30)` 等表达，不单独登记 |
| `daily_fdm_{totalshare,floatshare,dv_ttm}` | 本地 `daily_basic` 未落盘 |
| `sub_barra_cne5_*`（25 项） | 子因子族仅云端提供，本地只托管 10 维合成风格 |

---

## 4. 已登记偏差与设计决策（必须知晓）

### 4.1 全历史排名类算子采用**因果扩张窗口**
云端 `rank_ts` / `pct_rank_ts` / `dense_rank_ts` / `pct_dense_rank_ts` 的描述为
「在整段输入时间历史内排名」，直接实现会引入**未来函数**（`t` 时刻名次依赖 `t+1..T` 取值），
违反 `AGENTS.md` 反未来函数红线。本地统一实现为**扩张窗口因果版**：`t` 时刻名次仅由
`≤ t` 的历史决定；在序列末点（`t = T-1`）与云端语义**完全一致**。
复杂度：Fenwick 树 `O(T log D)`，`T=4000, N=5600` 全量可接受。

### 4.2 `ts_dw` 采用因果扩张残差 Durbin-Watson
云端未给出闭式定义。本地定义：对每只标的，以截至 `t` 的全部有效样本做**带截距 OLS** 残差递推
$r_s = x_{1,s} - (a_t + b_t x_{2,s})$，输出
$\mathrm{DW}_t = \sum_{s=1}^{t}(r_s - r_{s-1})^2 / \sum_{s=1}^{t} r_s^2$（需 ≥3 有效观测）。
中心化统计量采用 Welford 增量更新，单步 `O(1)`。

### 4.3 `log` / `sqrt` 为有符号安全变换
平台自 P2 起即约定 `log` → `slog = sign(x)·ln(1+|x|)`、`sqrt` → `ssqrt = sign(x)·√|x|`，
以便在含负值面板上仍可求值。这与云端「`ln(x)`，`x≤0 → NaN`」**不同口径**；
需要严格云端语义时请显式使用 `purify` + 自定义常数变换，或直接使用 `ssqrt`/`slog` 的数学性质。
> 该口径差异已在 `docs/futures_platform_plan.md` 与 `docs/api_reference.md` 登记。

### 4.4 `zScore` 别名的 `ddof` 差异
- 表达式算子 `cs_zscore`（`zScore` 别名）：样本标准差 `ddof=1`（与 `group_zscore`/`ts_zscore` 一致）；
- 后处理算子 `zScore`（`postprocess::pp_zscore`）：总体标准差 `ddof=0`（与云端 `ZsimCrossZScore` 一致）。
两者在 `N ≳ 500` 的 A 股截面上数值差异 `< 0.1%`；**因子准入对账须使用后处理链口径**。

### 4.5 掩码类算子仅在后处理链暴露
`baseUniMask` / `trdMask` / `dropNewStock` 需要隐式面板依赖（`in_universe_mask` /
`limit_mask` / `listed_days`），DSL 求值器不注入隐式依赖，因此这三个算子**只**通过
12 步声明式后处理链（`PostprocessStep`）生效，不提供表达式别名。这是刻意的架构边界，
避免用户在表达式中写出隐式依赖未知的因子。

### 4.6 变参多元回归的参数布局
```
cs_multi_beta / cs_multi_resi        : (y, exp_1, ..., exp_K [, intercept])
cs_multi_wls_beta / _resi            : (y, w,   exp_1, ..., exp_K [, intercept])
ts_multi_beta / ts_multi_resi        : 同上（沿时间轴扩张窗口）
ts_multi_wls_beta / _resi            : 同上
```
- `intercept` 为**可选尾部常数**，非零即启用截距（缺省 0 = 无截距）；
- `*_beta` 返回**首个暴露 `exp_1` 的偏回归系数**（控制其余暴露后的净载荷）并逐点广播；
  单暴露时即普通 OLS 斜率。多暴露的完整系数向量请使用后处理 `riskNeutMulti` 或
  `BarraRiskModelStore`；
- 数值稳定性：
  - 横截面族（`cs_multi_*`）在增广加权设计矩阵 `[√w·X; √ε·I]` 上做 **Householder QR**
    最小二乘（`ε = 1e-10`），条件数远优于正规方程；
  - 时序族（`ts_multi_*`）使用增量正规方程 + 列主元高斯消元（`O(dim²)` 增量、`O(dim³)` 日求解）；
  - 加权形态先将权重按有效样本最大权重**尺度归一化**（WLS 解对权重整体缩放不变），
    彻底避免 A 股市值量纲 `10^10~10^12` 导致的条件数爆炸。

### 4.7 `cs_argmax/cs_argmin` 输出 0 基列位置
返回截面极值所在的**0 基列索引**（并列取最小列号），而非 1 基名次；
`ts_argmax/ts_argmin` 返回窗口内相对偏移 `0..d-1`（并列取最早出现）。

### 4.8 `±Inf` 的平台 NaN 策略
- **面板装载边界**（`Panel` / `Engine` 构造）统一把 `±Inf` 归一为 `NaN`（`into_fortran_f64`）；
- **算子内部**保留原始 `±Inf` 语义，以便 `is_inf` / `purify` 能正确探测；
- 因此通过 DSL 求值时输入矩阵中的 `±Inf` 已在装载边界被归一，`is_inf(x)` 恒为 `0.0`（或 NaN）。
  需要探测原始时请使用 Python 侧算子直调或 `purify` 之前的中间结果。

---

## 5. 验收门禁

```bash
# Rust 侧：算子注册数 + FactorLab 106 项覆盖 + 类型检查阻断
cargo test --manifest-path rust/factorlib_core/Cargo.toml

# Python 侧：128 算子逐一对账（max abs err < 1e-9）+ 别名解析 + 目录检索
pytest tests/python/test_rust_core.py tests/python/test_stocks_v2_variables.py -q
```

对账基准因子：**仅保留 `Volume_Price_2_H5_2`** 作为跨数据源与 MCP 算子/数据对齐的
唯一黄金基准（其余 4 个云端草稿因子模板已从工作台模板与基准快照中清理）。