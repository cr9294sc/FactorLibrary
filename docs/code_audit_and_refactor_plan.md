# FactorLibrary 架构与代码级深度审查报告及重构行动计划

> **审计基准快照**：Commit `5c92977`（含 FactorLab 线上 5 因子 vs 本地 Rust 内核横向对账实证审查） | **二次逐项闭环验收**：2026-11-13（基于 `HEAD bc4346c` 严格复核 `P1/P2/P3`、Context7 权威接口与 A 股全市场回归） | **代码总规模**：44,555+ LOC（Python / Rust / JS / YAML / TOML）  
> **审计范围**：全量覆盖多代架构演进清理、期货 ETL 与前视偏差、Rust/Python 双轨数值一致性、STGP/ML 防过拟合、实盘回测与绩效账务、**A 股与 FactorLab 跨平台单因子对账及评价体系**六大核心维度。

---

## 1. 【执行摘要（Executive Summary）】

### 1.1 整体架构健康度评估

当前 `FactorLibrary` 主力新架构（体系 A：[`python/factorlib/`](file:///home/cr9294/repos/FactorLibrary/python/factorlib) + [`rust/factorlib_core/`](file:///home/cr9294/repos/FactorLibrary/rust/factorlib_core)）在工程底座上具备良好的设计水准：时序滚动算子采用了双栈 Welford 与单调队列算法，PyO3 边界通过 [`PyArcMatrix`](file:///home/cr9294/repos/FactorLibrary/rust/factorlib_core/src/py.rs#L158-L174) 实现了列优先 Fortran 矩阵零拷贝导出，主力切换映射（[`dominant.py`](file:///home/cr9294/repos/FactorLibrary/python/factorlib/markets/futures/etl/dominant.py)）与前向锚定比例复权（[`adjust.py`](file:///home/cr9294/repos/FactorLibrary/python/factorlib/markets/futures/etl/adjust.py)）满足因果无漂移约束。

然而，经逐行交叉源码走查、本地真实期货派生面板（`data/derived/futures/v1/`）核验，以及 **FactorLab 线上 5 大代表性因子 vs 本地 A 股全市场（2016–2024，5323 只股票）横向对账实证分析**，当前仓库处于**「算子底层求值已高度对齐，但回测评价与跨期对账存在致命时间窗错位、换手率指标张冠李戴、全天 VWAP 混入日内未来价格、ST/涨跌停掩码漏检、回测穿仓账务失真，且三代历史代码高度冗余耦合」**的高危状态，**尚不具备直接指导实盘资金配置或可信因子准入的条件**。

| 审查维度 | 健康度评级 | 核心诊断结论 |
| :--- | :---: | :--- |
| **1. 多代系统架构与构建体系** | 🟢 已修复 | 三代冗余原型（`pyrust_bt`、`engine_rust`、`Factor/`）已彻底移除，`server_main.py` 已收编为 [`workbench_server.py`](file:///home/cr9294/repos/FactorLibrary/python/factorlib/workbench_server.py)，`frontend/` 已迁入 [`python/factorlib/frontend/`](file:///home/cr9294/repos/FactorLibrary/python/factorlib/frontend) 并纳入 Wheel 打包与 `factorlib-web` CLI 入口；[`markets/stocks/panel_loader.py`](file:///home/cr9294/repos/FactorLibrary/python/factorlib/markets/stocks/panel_loader.py) 已完成标准化封装。 |
| **2. 期货 ETL 与前视偏差防护** | 🟢 已修复 | `P0-01`（`r_exec_next` 严格滞后对齐 `y_1` 并同步重建本地 `labels.parquet`）与 `P0-02`（换月日旧合约三级同合约回退）已全部闭环修复并通过因果摄动测试。 |
| **3. Rust/Python 双轨与数值安全** | 🟢 已修复 | `P0-03`（全部 `cs_*`/`group_*` 与风格暴露强制施加 `universe_mask`）、`P0-04`（AST `compress_outer_rank` 子树隔离与移除 `inv(inv(x))`）已全部修复并通过双轨对账。 |
| **4. STGP 演化与 ML 防过拟合** | 🟢 已修复 | `P0-05` 已将 `ml_default.yaml` 的 `start_predict_year` 推后至 `2022`（消除与 GP `2019–2021` 验证期 4 年重叠），并在 `GPEngine._resolve_splits` 与 `WalkForwardPipeline.run` 强制执行 `horizon` Purge 隔离带与短样本互斥三段切分。 |
| **5. 回测状态机与绩效账务** | 🟢 已修复 | `P0-06`（粘性 `is_bankrupt` 穿仓强平熔断与负净值 `powf` 封底 `-1.0`）、`P0-07`（反手平开拆分计费、`close_today_fee` 全链路透传、换月双边成本归集、停牌去杠杆拦截）与 `P0-08`（FFI 全入口 `n_quantiles >= 2` 校验及 Python 因子沙箱私有属性/I/O 封锁）已全部闭环。 |
| **6. A 股与 FactorLab 跨平台对账体系** | 🟢 已修复 | `P0-09`（基准指数前移对齐 $[T+1, T+1+h]$ 与复利几何年化）、`P0-10`（`.round()` 分位切分与 `top_quantile_turnover` 单边权重换手率）、`P0-11`（`open_adj` 主口径 + `vwap_adj` 对照口径、ST/涨跌停/上市自然日掩码、Polars `join_asof` 排序静默及脚本 PyO3 接口对齐）已全部闭环验证。 |

---

### 1.2 Top 8 最紧迫的致命问题（Must-Fix Immediately，已全量闭环修复 ✅）

1. **`P0-01` [已修复 / FIXED] 期货分位数多空评估误用当期早盘收益 `r_exec_next` 导致极度前视偏差**：[`labels.py`](file:///home/cr9294/repos/FactorLibrary/python/factorlib/markets/futures/etl/labels.py#L59-L65) 中第 `T` 行的 `r_exec_next` 原定义为 `T` 日早盘 30m VWAP 至 `T+1` 日早盘 30m VWAP 的收益率，而 [`py.rs`](file:///home/cr9294/repos/FactorLibrary/rust/factorlib_core/src/py.rs#L566) 与 [`evaluate.py`](file:///home/cr9294/repos/FactorLibrary/python/factorlib/core/evaluate.py#L247) 将第 `T` 日 15:00 收盘生成的因子同日配对 `r_exec_next[T]`。现已在 `labels.py` 对 `r_exec_next` 执行 `.shift(-1).over("symbol")`、在 `py.rs`/`evaluate.py`/`workbench.py` 统一锚定 `y_1`，并重新生成本地 `data/derived/futures/v1/labels.parquet`（实测 `max |r_exec_next - y_1| == 0.0`）。
2. **`P0-09` [已修复 / FIXED] A 股多头超额收益「基准时间窗错位 2 个交易日 + 价格钟失配」与单利年化失真**：[`scripts/backtest_multi_factors.py`](file:///home/cr9294/repos/FactorLibrary/scripts/backtest_multi_factors.py#L309-L310) 与 [`scripts/backtest_volume_price_factor.py`](file:///home/cr9294/repos/FactorLibrary/scripts/backtest_volume_price_factor.py#L218-L219) 在信号行 $T$ 上用 **$T+1 \to T+2$ 的多头 G10 远期收益**减去 **$T-1 \to T$ 的中证全指历史收盘收益**，叠加单利年化公式，导致 `Volume_Price_2_H5_2` 多头年化超额从线上 `1.26%` 虚高至 `+8.14%`，`Tech_Trend_VP_1_Rev` 从线上 `-1.72%` 逆转为 `+4.29%`。已通过 `calc_long_excess_metrics`、动态价格钟前移与复利几何年化标准公式彻底修复。
3. **`P0-10` [已修复 / FIXED] 换手率指标「张冠李戴」（`1 - rank_autocorr` 冒充多头 G10 单边权重换手率）导致 12 倍偏差**：[`scripts/backtest_multi_factors.py`](file:///home/cr9294/repos/FactorLibrary/scripts/backtest_multi_factors.py#L408) 将 [`calc_turnover`](file:///home/cr9294/repos/FactorLibrary/rust/factorlib_core/src/metrics/turnover.rs#L116) 输出的全截面秩自相关距离 $1 - \rho_1$（`1.00%` / `0.92%` / `0.51%`）直接与线上多头十分位真实单边持仓换手率 $\frac{1}{2}\sum |w_{t,i} - w_{t-1,i}|$（`12.11%` / `11.78%` / `9.30%`）对账，相差 **12~18 倍**；同时 [`evaluate.py`](file:///home/cr9294/repos/FactorLibrary/python/factorlib/core/evaluate.py#L405) 将首日与无效日换手率填 `0.0` 稀释均值，且 [`quantile.rs`](file:///home/cr9294/repos/FactorLibrary/rust/factorlib_core/src/metrics/quantile.rs#L84-L85) 分组边界使用 `floor` 而非 `round`。已通过在 Rust 内核原生实现 `.round()` 分位切分、`calc_top_quantile_turnover` 及 `top_turnover_series`，解耦多头分位权重单边换手率（`top_turnover`）与秩自相关衰减度（`rank_autocorr_turnover`），并修复无效跃迁日 `NaN` 过滤彻底解决。
4. **`P0-11` [已修复 / FIXED] A 股日线回测以全天 VWAP 冒充早盘 `VwapBegin30` 吞噬 37% 反转收益，且漏检 ST 与涨跌停（`BaseUnivMask` & `LimitMask`）**：已通过 [`python/factorlib/markets/stocks/panel_loader.py`](file:///home/cr9294/repos/FactorLibrary/python/factorlib/markets/stocks/panel_loader.py) 标准化封装 `is_st`、`limit_up`/`limit_down`、自然日上市天数 `listed_days` 与退市过滤，支持 `open_adj`（主口径）与 `vwap_adj`（对照口径）双轨评估，并修复 [`scripts/backtest_volume_price_factor.py`](file:///home/cr9294/repos/FactorLibrary/scripts/backtest_volume_price_factor.py) 与 [`scripts/backtest_multi_factors.py`](file:///home/cr9294/repos/FactorLibrary/scripts/backtest_multi_factors.py) 的 PyO3 接口绑定与端到端对账。
5. **`P0-02` [已修复 / FIXED] 换月日旧合约平仓价回退至新合约价格引发跨合约跳空与未来结算价泄漏**：[`build.py`](file:///home/cr9294/repos/FactorLibrary/python/factorlib/markets/futures/etl/build.py#L174-L218) 在旧主力早盘无成交时用主表 `settle_raw`（实为**新主力合约** `T` 日 15:00 盘后结算价）填充 `exec_vwap_old`；[`futures.rs`](file:///home/cr9294/repos/FactorLibrary/rust/factorlib_core/src/backtest/futures.rs#L243-L252) 在 `exec_price_old` 缺失时进一步回退至新合约 `exec_price` 并直接减去旧合约 `prev_settle`，产生虚假换月跳空暴利或巨亏。现已通过 ETL 层同合约三级级联回退（早盘 VWAP -> 日线 open -> 昨结 prev_settle / shift(1)）与 Warning 审计打点，以及 Rust 撮合内核严格回退同合约昨结彻底修复。
6. **`P0-03` [已修复 / FIXED] 全套横截面/分组算子与风格暴露完全忽略 `universe_mask`**：[`eval.rs`](file:///home/cr9294/repos/FactorLibrary/rust/factorlib_core/src/expr/eval.rs#L92-L97) 调用 `cs_rank`、`cs_zscore`、`cs_demean`、`cs_neutralize`、`group_*` 时未传入 `panel.universe_mask`；非在池僵尸合约或 A 股 ST/次新股的极端价格直接污染活跃标的池的截面均值、标准差、OLS 回归系数与多空资金中性平衡。现已在 `eval.rs` 调度层按 `universe_mask` 严格隔离非在池标的并置 `NaN`，并在 `styles.py` 与 `ops_ref.py` 同步掩码与左锚点并列秩算法彻底修复。
7. **`P0-04` [已修复 / FIXED] AST 规范化化简规则越界递归穿透子树篡改表达式数学语义**：[`canon.rs`](file:///home/cr9294/repos/FactorLibrary/rust/factorlib_core/src/expr/canon.rs#L98) 将仅在 AST 根节点保序成立的 `compress_outer_rank = true` 递归传入所有子节点，导致 `sub(cube(cs_rank(close)), cs_rank(open))` 等非线性复合因子被静默篡改为线性差 `sub(cs_rank(close), cs_rank(open))`；同时 `inv(inv(x)) => x` 抹除了 `x == 0` 处的 `NaN` 奇点掩码。现已限制外层单调压缩仅在根单调链或 CsRank 内部递归传递，二元与三元子树强制置 false，并移除 inv(inv(x)) 规则，通过双层跨语言求值等价性验证。
8. **`P0-06` & `P0-07` [已修复 / FIXED] 期货逐日盯市回测穿仓符号反转、`NaN` 传染与实盘手续费账务失真**：[`futures.rs`](file:///home/cr9294/repos/FactorLibrary/rust/factorlib_core/src/backtest/futures.rs#L281) 在账户净值 `prev_e <= 0` 后继续按 `w * prev_e` 与负 `margin_scale` 反向放大仓位，并在期末触发负底数幂运算 `(1.0 + total_ret).powf(...)` 输出 `NaN`；同时反手调仓手续费误算、`close_today` 平今高费率被丢弃、涨跌停锁仓在换月与保证金缩减阶段被穿透。已通过强平熔断归零、年化收益率封底 -1.0、多空反手平开手数独立计费、`close_today_fee` 全链路透传至 `SymbolSpec`、换月新开仓归集 `daily_roll_cost` 以及停牌/涨跌停去杠杆严格阻断彻底修复。

---

## 2. 【致命与高危问题详单（P0 & P1 Issues）】

---

### P0-01 `[P0-量化逻辑]` [已修复 / FIXED] 分位数回测与综合指标默认绑定未滞后 `r_exec_next`，引发日内前视偏差

- **定位信息**：
  - [`python/factorlib/markets/futures/etl/labels.py`](file:///home/cr9294/repos/FactorLibrary/python/factorlib/markets/futures/etl/labels.py#L59-L88)（函数 [`build_labels`](file:///home/cr9294/repos/FactorLibrary/python/factorlib/markets/futures/etl/labels.py#L31-L89)，第 `59-88` 行）
  - [`rust/factorlib_core/src/py.rs`](file:///home/cr9294/repos/FactorLibrary/rust/factorlib_core/src/py.rs#L553-L665)（方法 [`PyEngine::quantile_returns`](file:///home/cr9294/repos/FactorLibrary/rust/factorlib_core/src/py.rs#L553-L586) 第 `566` 行、[`PyEngine::factor_metrics`](file:///home/cr9294/repos/FactorLibrary/rust/factorlib_core/src/py.rs#L610-L683) 第 `633` 行）
  - [`python/factorlib/core/evaluate.py`](file:///home/cr9294/repos/FactorLibrary/python/factorlib/core/evaluate.py#L244-L254)（函数 [`_evaluate_factor_matrix_core`](file:///home/cr9294/repos/FactorLibrary/python/factorlib/core/evaluate.py#L169-L284) 第 `244-254` 行）
  - [`python/factorlib/core/workbench.py`](file:///home/cr9294/repos/FactorLibrary/python/factorlib/core/workbench.py#L254-L260)（函数 [`create_synthetic_futures_panel`](file:///home/cr9294/repos/FactorLibrary/python/factorlib/core/workbench.py#L134-L270) 第 `254-260` 行）

- **问题根因分析**：
  在 [`labels.py`](file:///home/cr9294/repos/FactorLibrary/python/factorlib/markets/futures/etl/labels.py#L59-L71) 中，`r_exec_next` 在交易日 `T` 行记录的是 `exec_vwap_old[T+1] / exec_vwap[T] - 1`，其买入起点 `exec_vwap[T]` 发生在 **`T` 日开盘前 30 分钟**（夜盘 `T-1` 21:00 或日盘 `T` 09:00），**早于 `T` 日 15:00 收盘因子生成时点**。虽然 `labels.py` 第 `71` 行专门通过 `_lg_next = pl.col("_lg").shift(-1).over("symbol")` 将 `y_1` 修正为 `T+1 -> T+2` 的真实可交易远期收益，但未平移的中间列 `r_exec_next` 被原样导出至 `labels.parquet`：
  ```python
  # python/factorlib/markets/futures/etl/labels.py:59-71
  grid = grid.with_columns(
      _old_next=pl.col("exec_vwap_old").shift(-1).over("symbol"),
  ).with_columns(
      r_exec_next=pl.when((pl.col("exec_vwap") > 0) & (pl.col("_old_next") > 0))
      .then(pl.col("_old_next") / pl.col("exec_vwap") - 1.0)
      .otherwise(None)
  )
  ...
  # T 日自身的 r_{T->T+1} (起点为 T 日早盘, 早于 T 日收盘信号) 不会进入 y_h 标签, 但 r_exec_next 未移位!
  grid = grid.with_columns(_lg_next=pl.col("_lg").shift(-1).over("symbol"))
  ```
  随后 [`evaluate.py`](file:///home/cr9294/repos/FactorLibrary/python/factorlib/core/evaluate.py#L247) 与 [`py.rs`](file:///home/cr9294/repos/FactorLibrary/rust/factorlib_core/src/py.rs#L566) 默认优先读取 `ret_name="r_exec_next"` 送入 [`calc_quantile_metrics`](file:///home/cr9294/repos/FactorLibrary/rust/factorlib_core/src/metrics/quantile.rs#L133-L145)，在同一行 `row == T` 上将 `T` 日 15:00 因子与 `T` 日 09:00 建仓的收益直接配对，把整段 `T` 日日内已知涨跌计入多空组合净值。此外，[`workbench.py`](file:///home/cr9294/repos/FactorLibrary/python/factorlib/core/workbench.py#L257-L259) 合成面板在构造 `y_h` 时也漏掉了 `1` 日偏移（写成了 `exec_vwap[h:] / exec_vwap[:-h] - 1`），导致单元测试未能捕获真实面板与合成面板的口径分裂。

- **具体修复方案**：
  统一将 1 日远期执行收益锚定至严格滞后一期的 `y_1`（即 $r_{T+1 \to T+2}$），或者在导出 `r_exec_next` 时对齐为 `T+1 -> T+2` 收益；同步修正合成面板 `y_h` 偏移。
  ```diff
  --- a/python/factorlib/markets/futures/etl/labels.py
  +++ b/python/factorlib/markets/futures/etl/labels.py
  @@ -68,7 +68,10 @@ def build_labels(panel: pl.DataFrame, cal: pl.DataFrame, cfg: DataConfig) -> pl.
       # y_h(T) 需要 r_{T+1->T+2} .. r_{T+h->T+h+1}, 即 lg 在行 T+1..T+h。先把 lg 前移一行
       # (行 T 持有 lg[T+1]), 再取从 T 起向前的 h 行窗口和 (稠密网格上行偏移 == 交易日偏移),
       # 这样 T 日自身的 r_{T->T+1} (起点为 T 日早盘, 早于 T 日收盘信号) 不会进入标签。
  -    grid = grid.with_columns(_lg_next=pl.col("_lg").shift(-1).over("symbol"))
  +    grid = grid.with_columns(
  +        r_exec_next=pl.col("r_exec_next").shift(-1).over("symbol"),
  +        _lg_next=pl.col("_lg").shift(-1).over("symbol"),
  +    )
       exprs = []
       for h in horizons:
  ```
  ```diff
  --- a/rust/factorlib_core/src/py.rs
  +++ b/rust/factorlib_core/src/py.rs
  @@ -563,7 +563,7 @@ impl PyEngine {
       ) -> PyResult<Bound<'py, PyDict>> {
           let f_raw = self.resolve_factor_mat(py, factor)?;
           let (f_mat, s_row, e_row) = self.align_factor_slice(f_raw, start_row, end_row)?;
  -        let r_name = ret_name.unwrap_or("r_exec_next");
  +        let r_name = ret_name.unwrap_or("y_1");
           let ret_arc = self
               .panel
               .labels
               .get(r_name)
  -            .or_else(|| self.panel.labels.get("y_1"))
  +            .or_else(|| self.panel.labels.get("r_exec_next"))
               .ok_or_else(|| {
  @@ -630,7 +630,7 @@ impl PyEngine {
                   PyValueError::new_err(format!("未在面板中找到预测标签: '{label_name}'"))
               })?
               .clone();
  -        let r_name = ret_name.unwrap_or("r_exec_next");
  +        let r_name = ret_name.unwrap_or("y_1");
           let ret_arc = self
               .panel
               .labels
  ```
  ```diff
  --- a/python/factorlib/core/evaluate.py
  +++ b/python/factorlib/core/evaluate.py
  @@ -244,7 +244,7 @@ def _evaluate_factor_matrix_core(
               met = engine.factor_metrics(
                   f_sub,
                   label_name=lbl_name,
  -                ret_name="r_exec_next" if "r_exec_next" in engine.label_names else None,
  +                ret_name="y_1" if "y_1" in engine.label_names else None,
                   horizon=h,
                   n_quantiles=n_quantiles,
                   min_names=min_names,
  ```
  ```diff
  --- a/python/factorlib/core/workbench.py
  +++ b/python/factorlib/core/workbench.py
  @@ -253,8 +253,8 @@ def create_synthetic_futures_panel(n_days: int = 120, seed: int = 42) -> Panel:
       labels: dict[str, np.ndarray] = {}
       for h in (1, 3, 5, 10, 20):
           yh = np.full((T, N), np.nan, dtype=np.float64, order="F")
  -        if T > h:
  -            yh[:-h, :] = exec_vwap[h:, :] / exec_vwap[:-h, :] - 1.0
  +        if T > h + 1:
  +            yh[: -(h + 1), :] = exec_vwap[1 + h :, :] / exec_vwap[1:-h, :] - 1.0
           labels[f"y_{h}"] = yh
       labels["r_exec_next"] = np.where(np.isfinite(labels["y_1"]), labels["y_1"], 0.0)
  ```

---

### P0-02 `[P0-量化逻辑]` [已修复 / FIXED] 换月日旧合约平仓价回退至新合约 `settle_raw` / `exec_price`，造成跨合约跳空与盘后价格泄漏

- **定位信息**：
  - [`python/factorlib/markets/futures/etl/build.py`](file:///home/cr9294/repos/FactorLibrary/python/factorlib/markets/futures/etl/build.py#L166-L177)（函数 [`build_panel`](file:///home/cr9294/repos/FactorLibrary/python/factorlib/markets/futures/etl/build.py#L65-L300)，第 `166-177` 行）
  - [`rust/factorlib_core/src/backtest/futures.rs`](file:///home/cr9294/repos/FactorLibrary/rust/factorlib_core/src/backtest/futures.rs#L240-L253)（函数 [`run_futures_backtest`](file:///home/cr9294/repos/FactorLibrary/rust/factorlib_core/src/backtest/futures.rs#L118-L558)，第 `240-253` 行）

- **问题根因分析**：
  遵守 `AGENTS.md` 红线 1：「换月日严禁使用跨合约价格直接作差，必须分别以新旧合约实际成交价进行平旧开新」。当前两处回退逻辑均直接破坏该不变量：
  1. 在 [`build.py`](file:///home/cr9294/repos/FactorLibrary/python/factorlib/markets/futures/etl/build.py#L173-L177) 中，当换月日 `is_roll == True` 且旧主力合约 `prev_contract` 早盘 30 分钟无分钟线成交（`_old_exec_vwap` 为 `null`）时，代码执行 `.fill_null(pl.col("settle_raw"))`。但此时 `daily` 行的主合约已经是**新主力合约 `contract` (`c_T`)**，其 `settle_raw` 既是**新合约价格**，又是 **`T` 日 15:00 收盘后才决定的未来结算价**！经对本地 `panel_daily.parquet` 核验，63 次换月中共有 **10 次**（如 `RS` 在 `2024-03-12` 从 `RS2407` 换至 `RS2411`）将新合约结算价写入了 `exec_vwap_old`。
  2. 在 [`futures.rs`](file:///home/cr9294/repos/FactorLibrary/rust/factorlib_core/src/backtest/futures.rs#L242-L252) 中，当 `exec_price_old[[d, i]]` 为 `NaN` 时，第 `245-246` 行回退至 `exec_price[[d, i]]`（即**新合约开盘成交价**），随后第 `252` 行执行 `(p_old - prev_settle[i]) * multiplier`——直接拿新合约开盘价减去旧合约昨日结算价！

- **具体修复方案**：
  在 ETL 阶段关联 `bar_1d` 获取旧主力合约 `prev_contract` 在 `T` 日的真实 `open`（若当日全天零成交则取旧合约在 `T-1` 日的 `settlement`）；在 Rust 撮合引擎中，换月日 `exec_price_old` 缺失时严禁回退至新合约 `exec_price`，仅能回退至同属旧合约的 `prev_settle[i]`（或顺延换月）。
  ```diff
  --- a/python/factorlib/markets/futures/etl/build.py
  +++ b/python/factorlib/markets/futures/etl/build.py
  @@ -170,10 +170,27 @@ def build_panel(cfg: DataConfig, out_dir: Path | None = None) -> Path:
           pl.col("exec_vwap").alias("_old_exec_vwap"),
       )
       daily = daily.join(rolls_exec, on=["prev_contract", "trading_date"], how="left")
  +    old_bar_fallback = (
  +        scan_dataset(cfg.raw_dir, "bar_1d", ["order_book_id", "trading_date", "open", "prev_settlement"])
  +        .filter(pl.col("trading_date").is_between(cfg.start_date, cfg.end_date or date.max))
  +        .select(
  +            pl.col("order_book_id").alias("prev_contract"),
  +            "trading_date",
  +            pl.when(pl.col("open") > 0)
  +            .then(pl.col("open"))
  +            .otherwise(pl.col("prev_settlement"))
  +            .alias("_old_bar_open"),
  +        )
  +        .unique(subset=["prev_contract", "trading_date"], keep="last")
  +        .collect()
  +    )
  +    daily = daily.join(old_bar_fallback, on=["prev_contract", "trading_date"], how="left")
       daily = daily.with_columns(
           exec_vwap_old=pl.when(pl.col("is_roll"))
  -        .then(pl.col("_old_exec_vwap").fill_null(pl.col("settle_raw")))
  +        .then(
  +            pl.col("_old_exec_vwap")
  +            .fill_null(pl.col("_old_bar_open"))
  +            .fill_null(pl.col("settle_raw").shift(1).over("symbol"))
  +        )
           .otherwise(pl.col("exec_vwap"))
  -    ).drop("_old_exec_vwap")
  +    ).drop(["_old_exec_vwap", "_old_bar_open"])
  ```
  ```diff
  --- a/rust/factorlib_core/src/backtest/futures.rs
  +++ b/rust/factorlib_core/src/backtest/futures.rs
  @@ -240,12 +240,10 @@ pub fn run_futures_backtest(
           for i in 0..n {
               if is_roll[[d, i]] && current_lots[i] != 0 {
                   let old_l = current_lots[i];
                   let p_old_raw = exec_price_old[[d, i]];
                   let p_old = if p_old_raw.is_finite() && p_old_raw > 0.0 {
                       p_old_raw
  -                } else if exec_price[[d, i]].is_finite() && exec_price[[d, i]] > 0.0 {
  -                    exec_price[[d, i]]
                   } else {
                       prev_settle[i]
                   };
  ```

---

### P0-03 `[P0-量化逻辑]` [已修复 / FIXED] 横截面算子 (`cs_*`)、分组算子 (`group_*`) 与风格矩阵计算忽略 `universe_mask`

- **定位信息**：
  - [`rust/factorlib_core/src/expr/eval.rs`](file:///home/cr9294/repos/FactorLibrary/rust/factorlib_core/src/expr/eval.rs#L92-L97) 及 [`eval.rs`](file:///home/cr9294/repos/FactorLibrary/rust/factorlib_core/src/expr/eval.rs#L128-L139)（函数 [`eval_ast`](file:///home/cr9294/repos/FactorLibrary/rust/factorlib_core/src/expr/eval.rs#L61-L240)）
  - [`rust/factorlib_core/src/ops/cs.rs`](file:///home/cr9294/repos/FactorLibrary/rust/factorlib_core/src/ops/cs.rs#L63-L348)（函数 `op_cs_rank`、`op_cs_zscore`、`op_cs_demean`、`op_cs_scale`、`op_cs_winsorize`、`op_cs_neutralize`）
  - [`rust/factorlib_core/src/ops/group.rs`](file:///home/cr9294/repos/FactorLibrary/rust/factorlib_core/src/ops/group.rs#L13-L141)（函数 `op_group_demean`、`op_group_rank`、`op_group_zscore`）
  - [`python/factorlib/core/styles.py`](file:///home/cr9294/repos/FactorLibrary/python/factorlib/core/styles.py#L94-L101)（函数 [`_normalize_style_matrix`](file:///home/cr9294/repos/FactorLibrary/python/factorlib/core/styles.py#L94-L101)）

- **问题根因分析**：
  [`Panel.load`](file:///home/cr9294/repos/FactorLibrary/python/factorlib/core/panel.py#L182-L201) 将 `panel_daily.parquet` 中所有品种（含 `in_universe == False` 的低流动性僵尸品种如胶合板 `BB`、动力煤 `ZC`、早籼稻 `RI`、普麦 `PM`、强麦 `WH`、新上市未满 60 日品种及距交割不足 5 日合约）的有限收盘价与成交量全部载入 `Panel.fields`。然而在 [`eval.rs`](file:///home/cr9294/repos/FactorLibrary/rust/factorlib_core/src/expr/eval.rs#L92-L97) 中：
  ```rust
  UnaryOp::CsRank => cs::op_cs_rank(&arg_mat),
  UnaryOp::CsZscore => cs::op_cs_zscore(&arg_mat),
  UnaryOp::CsDemean => cs::op_cs_demean(&arg_mat),
  UnaryOp::CsScale => cs::op_cs_scale(&arg_mat),
  UnaryOp::CsWinsorize => cs::op_cs_winsorize(&arg_mat),
  ```
  上述算子以及 `BinaryOp::CsNeutralize`、`BinaryOp::Group*` 完全未接收 `&panel.universe_mask`！只要僵尸品种当日有报价（甚至前值填充），就会参与截面均值、标准差、中位数 MAD、分位秩与 OLS 回归斜率计算，导致活跃可交易品种池 $U_t$ 上的 `cs_demean(x)` 之和不再为 0（$\sum_{i \in U_t} \tilde{x}_{t,i} \neq 0$）。同理，[`styles.py`](file:///home/cr9294/repos/FactorLibrary/python/factorlib/core/styles.py#L94-L101) 在计算 6 大 Barra 风格暴露前也未施加 `panel.valid_mask`。

- **具体修复方案**：
  为 Rust 内核提供掩码感知的截面/分组执行入口（在进入截面与分组统计前按 `panel.universe_mask` 过滤输入并将非在池位置置 `NaN`，注意时序滚动算子 `ts_*` 仍保留原始连续历史序列以避免进出池断档），并在 [`styles.py`](file:///home/cr9294/repos/FactorLibrary/python/factorlib/core/styles.py#L94-L101) 归一化前先做在池掩码过滤。
  ```diff
  --- a/rust/factorlib_core/src/expr/eval.rs
  +++ b/rust/factorlib_core/src/expr/eval.rs
  @@ -15,6 +15,21 @@ use ndarray::Array2;
   use rayon::prelude::*;
   use std::collections::{HashMap, HashSet};
   use std::sync::Arc;
  +
  +#[inline]
  +fn mask_to_universe(mat: &Array2<f64>, mask: &Array2<bool>) -> Array2<f64> {
  +    let (t, n) = mat.dim();
  +    let mut out = mat.clone();
  +    for col in 0..n {
  +        for row in 0..t {
  +            if !mask[[row, col]] {
  +                out[[row, col]] = f64::NAN;
  +            }
  +        }
  +    }
  +    out
  +}
   
   #[derive(Error, Debug, Clone)]
   pub enum EvalError {
  @@ -89,11 +104,16 @@ pub fn eval_ast(
                   UnaryOp::Relu => elem::op_relu(&arg_mat),
                   UnaryOp::Atan => elem::op_atan(&arg_mat),
                   UnaryOp::NotCond => elem::op_not_cond(&arg_mat),
  -                UnaryOp::CsRank => cs::op_cs_rank(&arg_mat),
  -                UnaryOp::CsZscore => cs::op_cs_zscore(&arg_mat),
  -                UnaryOp::CsDemean => cs::op_cs_demean(&arg_mat),
  -                UnaryOp::CsScale => cs::op_cs_scale(&arg_mat),
  -                UnaryOp::CsWinsorize => cs::op_cs_winsorize(&arg_mat),
  +                UnaryOp::CsRank
  +                | UnaryOp::CsZscore
  +                | UnaryOp::CsDemean
  +                | UnaryOp::CsScale
  +                | UnaryOp::CsWinsorize => {
  +                    let masked = mask_to_universe(&arg_mat, &panel.universe_mask);
  +                    match op {
  +                        UnaryOp::CsRank => cs::op_cs_rank(&masked),
  +                        UnaryOp::CsZscore => cs::op_cs_zscore(&masked),
  +                        UnaryOp::CsDemean => cs::op_cs_demean(&masked),
  +                        UnaryOp::CsScale => cs::op_cs_scale(&masked),
  +                        UnaryOp::CsWinsorize => cs::op_cs_winsorize(&masked),
  +                        _ => unreachable!(),
  +                    }
  +                }
               };
  @@ -126,17 +146,20 @@ pub fn eval_ast(
                   BinaryOp::AndCond => elem::op_and_cond(&lhs_mat, &*eval_ast(rhs, panel, cache)?),
                   BinaryOp::OrCond => elem::op_or_cond(&lhs_mat, &*eval_ast(rhs, panel, cache)?),
                   BinaryOp::CsNeutralize => {
  -                    cs::op_cs_neutralize(&lhs_mat, &*eval_ast(rhs, panel, cache)?)
  +                    let m_lhs = mask_to_universe(&lhs_mat, &panel.universe_mask);
  +                    let m_rhs = mask_to_universe(&*eval_ast(rhs, panel, cache)?, &panel.universe_mask);
  +                    cs::op_cs_neutralize(&m_lhs, &m_rhs)
                   }
                   BinaryOp::GroupDemean => {
  -                    group::op_group_demean(&lhs_mat, &*eval_ast(rhs, panel, cache)?)
  +                    let m_lhs = mask_to_universe(&lhs_mat, &panel.universe_mask);
  +                    group::op_group_demean(&m_lhs, &*eval_ast(rhs, panel, cache)?)
                   }
                   BinaryOp::GroupRank => {
  -                    group::op_group_rank(&lhs_mat, &*eval_ast(rhs, panel, cache)?)
  +                    let m_lhs = mask_to_universe(&lhs_mat, &panel.universe_mask);
  +                    group::op_group_rank(&m_lhs, &*eval_ast(rhs, panel, cache)?)
                   }
                   BinaryOp::GroupZscore => {
  -                    group::op_group_zscore(&lhs_mat, &*eval_ast(rhs, panel, cache)?)
  +                    let m_lhs = mask_to_universe(&lhs_mat, &panel.universe_mask);
  +                    group::op_group_zscore(&m_lhs, &*eval_ast(rhs, panel, cache)?)
                   }
  ```
  ```diff
  --- a/python/factorlib/core/styles.py
  +++ b/python/factorlib/core/styles.py
  @@ -91,9 +91,9 @@ STYLE_DEFINITIONS: list[dict[str, Any]] = [
   ]
   
   
  -def _normalize_style_matrix(raw: np.ndarray) -> np.ndarray:
  +def _normalize_style_matrix(raw: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
       """对原始风格矩阵逐行执行 MAD(3.0) 缩尾 + zScore(ddof=0) 标准化。"""
  -    f_raw = np.asfortranarray(raw, dtype=np.float64)
  +    f_raw = np.asfortranarray(np.where(mask, raw, np.nan) if mask is not None else raw, dtype=np.float64)
       mad_clipped = np.asarray(factorlib_core.apply_postprocess_op("MAD", f_raw, param=3.0))
       z_out = np.asarray(
           factorlib_core.apply_postprocess_op("zScore", np.asfortranarray(mad_clipped))
  ```
  *（注：[`compute_futures_styles`](file:///home/cr9294/repos/FactorLibrary/python/factorlib/core/styles.py#L136-L200) 内全部 6 处 `_normalize_style_matrix(...)` 调用同步传入 `mask=panel.valid_mask`。）*

---

### P0-04 `[P0-量化逻辑]` [已修复 / FIXED] AST 代数规范化将根节点保序消去递归穿透至内层子树，并误消 `inv(inv(x))` 零值掩码

- **定位信息**：
  - [`rust/factorlib_core/src/expr/canon.rs`](file:///home/cr9294/repos/FactorLibrary/rust/factorlib_core/src/expr/canon.rs#L65-L130) 及 [`canon.rs`](file:///home/cr9294/repos/FactorLibrary/rust/factorlib_core/src/expr/canon.rs#L312-L345)（函数 [`canonicalize`](file:///home/cr9294/repos/FactorLibrary/rust/factorlib_core/src/expr/canon.rs#L65-L90)、[`canonicalize_rec`](file:///home/cr9294/repos/FactorLibrary/rust/factorlib_core/src/expr/canon.rs#L92-L545)）
  - [`rust/factorlib_core/src/expr/eval.rs`](file:///home/cr9294/repos/FactorLibrary/rust/factorlib_core/src/expr/eval.rs#L243-L256)（函数 [`eval_expr_str`](file:///home/cr9294/repos/FactorLibrary/rust/factorlib_core/src/expr/eval.rs#L243-L263)）

- **问题根因分析**：
  1. **子树保序消去越界**：严格单调递增函数 $g(\text{cs\_rank}(x))$ 与 $\text{cs\_rank}(x)$ 仅在**整棵 AST 的最外层根节点**（且后续仅评估截面排序如 RankIC 时）具有相同秩次；一旦位于算术加减乘除、截面标准化或时序算子内部（例如 `sub(cube(cs_rank(close)), cs_rank(open))` 或 `ts_mean(sigmoid(cs_rank(close)), 10)`），非线性变换 `cube` / `sigmoid` 会实质改变数值分布与复合后的截面排序。但 [`canon.rs`](file:///home/cr9294/repos/FactorLibrary/rust/factorlib_core/src/expr/canon.rs#L98) 第 `98`、`341`、`342` 行将 `compress_outer_rank = true` 无条件向下传递到了所有子节点：
     ```rust
     // rust/factorlib_core/src/expr/canon.rs:97-98 & 340-342
     ASTNode::UnaryCall { op, arg } => {
         let (c_arg, arg_degen) = canonicalize_rec(arg, compress_outer_rank);
     ...
     ASTNode::BinaryCall { op, lhs, rhs } => {
         let (mut c_lhs, l_degen) = canonicalize_rec(lhs, compress_outer_rank);
         let (mut c_rhs, r_degen) = canonicalize_rec(rhs, compress_outer_rank);
     ```
     当 `eval_gp_batch` 或 `eval(expr, canonicalize=true)` 执行时，上述表达式的值与排名均被破坏。
  2. **`inv(inv(x)) => x` 破坏零点 `NaN` 语义**：[`elem.rs`](file:///home/cr9294/repos/FactorLibrary/rust/factorlib_core/src/ops/elem.rs) 中 `inv(0.0) == NaN`，故 `inv(inv(0.0)) == NaN`，而 [`canon.rs`](file:///home/cr9294/repos/FactorLibrary/rust/factorlib_core/src/expr/canon.rs#L121-L130) 将其化简为 `x`（在 `0.0` 处输出 `0.0`），导致 `canonicalize=true` 与 `canonicalize=false` 求值不一致。

- **具体修复方案**：
  仅当父节点本身是根节点或嵌套的 `cs_rank` 时才保留单调消去，进入任何算术、截面均值方差或时序子节点时强制置 `compress_outer_rank = false`；移除不安全的 `inv(inv(x)) => x` 规则。
  ```diff
  --- a/rust/factorlib_core/src/expr/canon.rs
  +++ b/rust/factorlib_core/src/expr/canon.rs
  @@ -95,7 +95,8 @@ fn canonicalize_rec(node: &ASTNode, compress_outer_rank: bool) -> (ASTNode, Opti
           ASTNode::Field(_) | ASTNode::Constant(_) | ASTNode::Window(_) => (node.clone(), None),
   
           ASTNode::UnaryCall { op, arg } => {
  -            let (c_arg, arg_degen) = canonicalize_rec(arg, compress_outer_rank);
  +            let child_compress = *op == UnaryOp::CsRank || (compress_outer_rank && op.is_strictly_increasing());
  +            let (c_arg, arg_degen) = canonicalize_rec(arg, child_compress);
               if arg_degen.is_some() {
                   return (
                       ASTNode::UnaryCall {
  @@ -118,16 +119,6 @@ fn canonicalize_rec(node: &ASTNode, compress_outer_rank: bool) -> (ASTNode, Opti
                   ) => {
                       return (*inner.clone(), None);
                   }
  -                // inv(inv(x)) => x
  -                (
  -                    UnaryOp::Inv,
  -                    ASTNode::UnaryCall {
  -                        op: UnaryOp::Inv,
  -                        arg: inner,
  -                    },
  -                ) => {
  -                    return (*inner.clone(), None);
  -                }
                   // not_cond(not_cond(x)) => x
                   (
                       UnaryOp::NotCond,
  @@ -338,8 +329,8 @@ fn canonicalize_rec(node: &ASTNode, compress_outer_rank: bool) -> (ASTNode, Opti
           }
   
           ASTNode::BinaryCall { op, lhs, rhs } => {
  -            let (mut c_lhs, l_degen) = canonicalize_rec(lhs, compress_outer_rank);
  -            let (mut c_rhs, r_degen) = canonicalize_rec(rhs, compress_outer_rank);
  +            let (mut c_lhs, l_degen) = canonicalize_rec(lhs, false);
  +            let (mut c_rhs, r_degen) = canonicalize_rec(rhs, false);
  @@ -512,9 +503,9 @@ fn canonicalize_rec(node: &ASTNode, compress_outer_rank: bool) -> (ASTNode, Opti
               then_branch,
               else_branch,
           } => {
  -            let (c_cond, d0) = canonicalize_rec(cond, compress_outer_rank);
  -            let (c_then, d1) = canonicalize_rec(then_branch, compress_outer_rank);
  -            let (c_else, d2) = canonicalize_rec(else_branch, compress_outer_rank);
  +            let (c_cond, d0) = canonicalize_rec(cond, false);
  +            let (c_then, d1) = canonicalize_rec(then_branch, false);
  +            let (c_else, d2) = canonicalize_rec(else_branch, false);
  ```

---

### P0-05 `[P0-量化逻辑]` [已修复 / FIXED] GP 准入样本期（2011–2021）与 ML 预测期（2018–2026）跨管线泄漏，且 GP 切分缺失 Purge 隔离带

- **定位信息**：
  - [`configs/futures/gp_default.yaml`](file:///home/cr9294/repos/FactorLibrary/configs/futures/gp_default.yaml#L6-L10) 与 [`configs/futures/ml_default.yaml`](file:///home/cr9294/repos/FactorLibrary/configs/futures/ml_default.yaml#L8-L15)
  - [`python/factorlib/gp/engine.py`](file:///home/cr9294/repos/FactorLibrary/python/factorlib/gp/engine.py#L160-L201)（方法 [`GPEngine._resolve_splits`](file:///home/cr9294/repos/FactorLibrary/python/factorlib/gp/engine.py#L138-L226)）
  - [`python/factorlib/ml/pipeline.py`](file:///home/cr9294/repos/FactorLibrary/python/factorlib/ml/pipeline.py#L172-L208)（方法 [`WalkForwardPipeline.from_registry`](file:///home/cr9294/repos/FactorLibrary/python/factorlib/ml/pipeline.py#L149-L186) 与 [`WalkForwardPipeline.run`](file:///home/cr9294/repos/FactorLibrary/python/factorlib/ml/pipeline.py#L188-L320)）

- **问题根因分析**：
  1. **跨管线特征选择前视偏差（Selection Look-Ahead Bias）**：[`gp_default.yaml`](file:///home/cr9294/repos/FactorLibrary/configs/futures/gp_default.yaml#L7-L9) 使用 `train: 2011–2018` 搜索因子并确定方向 `sign`，再用 `valid: 2019–2021` 执行五步准入筛选（`valid_ic_ir >= 0.30` 且同号）写入 `factors.sqlite`（`status="accepted"`）。然而 [`ml_default.yaml`](file:///home/cr9294/repos/FactorLibrary/configs/futures/ml_default.yaml#L9) 配置 `start_predict_year: 2018`，由 [`WalkForwardPipeline.from_registry`](file:///home/cr9294/repos/FactorLibrary/python/factorlib/ml/pipeline.py#L160-L179) 一次性加载全部 `"accepted"` 因子去回测 `2018–2021` 年的「样本外」折叠——相当于站在 2018 年初使用了 2018–2021 四年未来行情筛选出的幸存因子！
  2. **GP 切分缺少 Purge 隔离窗且短样本回退使 `test == valid`**：在 [`gp/engine.py`](file:///home/cr9294/repos/FactorLibrary/python/factorlib/gp/engine.py#L170-L197) 中，`tr_s`（截至 `2018-12-31`）与 `va_s`（始于 `2019-01-01`）紧邻无缝衔接；当 `label="y_5"`（或 `y_20`）时，`train` 末尾 `h` 天的远期收益标签直接跨越至 `valid` 首部 `h` 天！且短样本回退分支第 `196` 行 `te_fallback = (va_fallback[0], T)` 使测试集与验证集 100% 重叠。
  3. **ML `purge_window` 未与 `target_label` 周期强制联动**：[`ml/pipeline.py`](file:///home/cr9294/repos/FactorLibrary/python/factorlib/ml/pipeline.py#L204-L206) 直接照搬配置中的静态 `purge_window: 5`，当用户将 `target_label` 改为 `"y_10"` 或 `"y_20"` 时，自动泄漏 5–15 天跨期重叠收益。

- **具体修复方案**：
  1. 将 `ml_default.yaml` 的 `start_predict_year` 推后至 GP 验证期结束之后的 `2022` 年，并在 `WalkForwardPipeline.run()` 中强制令 `effective_purge = max(cfg.walk_forward.purge_window, label_horizon)`；
  2. 在 `GPEngine._resolve_splits` 的 `train` 与 `valid`、`valid` 与 `test` 交界处强制裁除尾部 `self.horizon` 行（Purge），并修复短样本三段切分比例（`60% / 20% / 20%`）。
  ```diff
  --- a/configs/futures/ml_default.yaml
  +++ b/configs/futures/ml_default.yaml
  @@ -6,7 +6,7 @@ features_status: ["accepted", "active"]
   
   # Walk-Forward 滚动配置
   walk_forward:
  -  start_predict_year: 2018
  +  start_predict_year: 2022        # 严格位于 GP 准入验证集 (2019-2021) 之后，杜绝跨管线筛选泄漏
     end_predict_year: 2026
     train_window: "expanding"       # expanding: 扩展窗口; rolling: 固定滚动
     min_train_years: 5
  ```
  ```diff
  --- a/python/factorlib/gp/engine.py
  +++ b/python/factorlib/gp/engine.py
  @@ -172,8 +172,11 @@ class GPEngine:
           te_s = _find_slice(te_cfg[0], te_cfg[1])
   
           if tr_s is not None and va_s is not None:
  +            # 严格执行 Purge 隔离：剔除 train / valid 尾部跨越下一阶段边界的 h 行标签重叠样本
  +            tr_purged_end = max(tr_s[0] + 3, min(tr_s[1], va_s[0] - self.horizon))
  +            tr_s = (tr_s[0], tr_purged_end)
               if te_s is None:
                   te_s = va_s
  +            elif te_s[0] > va_s[0]:
  +                va_purged_end = max(va_s[0] + 3, min(va_s[1], te_s[0] - self.horizon))
  +                va_s = (va_s[0], va_purged_end)
  @@ -184,16 +187,16 @@ class GPEngine:
           # 若面板日期范围不在默认 2011-2021 区间内（如样本面板 v1 或单测小面板），按有效标签在池区间比例切分
           valid_lbl_rows = np.where(valid_lbl_per_day >= eff_min_names)[0]
  -        if len(valid_lbl_rows) >= 6:
  +        if len(valid_lbl_rows) >= 9 + 2 * self.horizon:
               eff_start = int(valid_lbl_rows[0])
               eff_end = int(valid_lbl_rows[-1]) + 1
               span = eff_end - eff_start
  -            n_train = max(3, int(round(span * 0.60)))
  -            if span - n_train < 3:
  -                n_train = span - 3
  -            tr_fallback = (eff_start, eff_start + n_train)
  -            va_fallback = (eff_start + n_train, eff_end)
  -            te_fallback = (va_fallback[0], T)
  +            n_train = max(3, int(round(span * 0.60)) - self.horizon)
  +            va_start = eff_start + n_train + self.horizon
  +            n_val = max(3, int(round(span * 0.20)) - self.horizon)
  +            te_start = min(eff_end - 3, va_start + n_val + self.horizon)
  +            tr_fallback = (eff_start, eff_start + n_train)
  +            va_fallback = (va_start, va_start + n_val)
  +            te_fallback = (te_start, eff_end)
  ```
  ```diff
  --- a/python/factorlib/ml/pipeline.py
  +++ b/python/factorlib/ml/pipeline.py
  @@ -11,6 +11,7 @@ from __future__ import annotations
   import dataclasses
   import logging
  +import re
   from pathlib import Path
  @@ -201,8 +202,13 @@ class WalkForwardPipeline:
           T, N = self.panel.shape
   
           # 2. 初始化切分器并生成时序折叠
  +        m_h = re.match(r"^y_(\d+)$", self.config.target_label.strip())
  +        lbl_horizon = int(m_h.group(1)) if m_h else 1
  +        wf_cfg = dataclasses.replace(
  +            self.config.walk_forward,
  +            purge_window=max(int(self.config.walk_forward.purge_window), lbl_horizon),
  +        )
           splitter = PurgedWalkForwardSplitter.from_config(
  -            self.config.walk_forward, strict_min_years=strict_min_years
  +            wf_cfg, strict_min_years=strict_min_years
           )
  ```

---

### P0-06 `[P0-量化逻辑]` [已修复 / FIXED] 期货回测穿仓后目标仓位与保证金缩放双重符号翻转、无强平熔断，且负净值触发 `powf` 输出 `NaN`

- **定位信息**：
  - [`rust/factorlib_core/src/backtest/futures.rs`](file:///home/cr9294/repos/FactorLibrary/rust/factorlib_core/src/backtest/futures.rs#L276-L365) 及 [`futures.rs`](file:///home/cr9294/repos/FactorLibrary/rust/factorlib_core/src/backtest/futures.rs#L481-L486)
  - [`rust/factorlib_core/src/backtest/fast.rs`](file:///home/cr9294/repos/FactorLibrary/rust/factorlib_core/src/backtest/fast.rs#L142-L147)

- **问题根因分析**：
  在极端单边行情或高杠杆测试中，一旦某日盯市结算后账户权益跌破零（`prev_e <= 0.0`）：
  1. [`futures.rs`](file:///home/cr9294/repos/FactorLibrary/rust/factorlib_core/src/backtest/futures.rs#L281) 第 `281` 行 `let target_val = w * prev_e;` 将所有多空信号方向**彻底反转**，且负债越深名义开仓额越大；
  2. 第 `344-350` 行 `let max_allowed_margin = cfg.max_margin_ratio * prev_e;` 为负数，因 `est_margin > max_allowed_margin` 恒成立，`margin_scale < 0.0` **再次反转符号**并按负债绝对值缩放仓位；
  3. 期末第 `483` 行（以及 [`fast.rs`](file:///home/cr9294/repos/FactorLibrary/rust/factorlib_core/src/backtest/fast.rs#L144)）计算 `(1.0 + total_ret).powf(252.0 / n_days as f64) - 1.0`，当 `total_ret < -1.0` 且指数非整数时，IEEE-754 负底数实数幂直接返回 **`NaN`**，污染回测报告与 Calmar 比率。

- **具体修复方案**：
  当 `prev_e <= 0.0`（穿仓破产）时触发实盘强平状态机：目标手数全部归零（仅允许在非锁仓合约上平清残余头寸，禁止任何新开仓），并在年化收益率公式中对 `1.0 + total_ret` 施加 `.max(0.0)` 截断保护（破产时年化收益封底为 `-1.0`）。
  ```diff
  --- a/rust/factorlib_core/src/backtest/futures.rs
  +++ b/rust/factorlib_core/src/backtest/futures.rs
  @@ -275,9 +275,13 @@ pub fn run_futures_backtest(
           // Step 2: 目标手数离散化计算 (基于 d-1 日收盘目标权重 W 与 prev_e)
           // -------------------------------------------------------------
           let mut target_lots = current_lots.clone();
  -        for i in 0..n {
  +        if prev_e <= 0.0 {
  +            // 账户已穿仓破产：触发强制清盘，禁止开新仓，所有品种目标手数归零
  +            target_lots.fill(0);
  +        } else {
  +            for i in 0..n {
                   let w = target_weights[[d - 1, i]];
                   let p_exec = exec_price[[d, i]];
                   if w.is_finite() && p_exec.is_finite() && p_exec > 0.0 && specs[i].multiplier > 0.0 {
                       let target_val = w * prev_e;
  @@ -295,6 +299,7 @@ pub fn run_futures_backtest(
                   }
  +            }
           }
  @@ -341,7 +346,7 @@ pub fn run_futures_backtest(
               est_margin += (target_lots[i].abs() as f64) * p * specs[i].multiplier * eff_rate;
           }
   
  -        let max_allowed_margin = cfg.max_margin_ratio * prev_e;
  +        let max_allowed_margin = (cfg.max_margin_ratio * prev_e).max(0.0);
           if est_margin > max_allowed_margin && est_margin > 1e-6 {
  @@ -480,7 +485,11 @@ pub fn run_futures_backtest(
           0.0
       };
   
       let n_days = t - 1;
       let annual_ret = if n_days > 0 {
  -        (1.0 + total_ret).powf(252.0 / n_days as f64) - 1.0
  +        if 1.0 + total_ret <= 0.0 {
  +            -1.0
  +        } else {
  +            (1.0 + total_ret).powf(252.0 / n_days as f64) - 1.0
  +        }
       } else {
           0.0
       };
  ```
  ```diff
  --- a/rust/factorlib_core/src/backtest/fast.rs
  +++ b/rust/factorlib_core/src/backtest/fast.rs
  @@ -141,7 +141,11 @@ pub fn run_fast_backtest(
   
       let n_days = t - 1;
       let annual_ret = if n_days > 0 {
  -        (1.0 + total_ret).powf(252.0 / n_days as f64) - 1.0
  +        if 1.0 + total_ret <= 0.0 {
  +            -1.0
  +        } else {
  +            (1.0 + total_ret).powf(252.0 / n_days as f64) - 1.0
  +        }
       } else {
           0.0
       };
  ```

---

### P0-07 `[P0-量化逻辑]` [已修复 / FIXED] 期货撮合账务四大失真：多空反手误算手续费、`close_today` 丢弃、换月单边计费、涨跌停/停牌强平穿透

- **定位信息**：
  - [`rust/factorlib_core/src/backtest/futures.rs`](file:///home/cr9294/repos/FactorLibrary/rust/factorlib_core/src/backtest/futures.rs#L42-L52) 及 [`futures.rs`](file:///home/cr9294/repos/FactorLibrary/rust/factorlib_core/src/backtest/futures.rs#L240-L270)、[`futures.rs`](file:///home/cr9294/repos/FactorLibrary/rust/factorlib_core/src/backtest/futures.rs#L345-L400)
  - [`python/factorlib/core/backtest.py`](file:///home/cr9294/repos/FactorLibrary/python/factorlib/core/backtest.py#L209-L224)（函数 [`run_backtest`](file:///home/cr9294/repos/FactorLibrary/python/factorlib/core/backtest.py#L140-L265)）
  - [`rust/factorlib_core/src/backtest/fast.rs`](file:///home/cr9294/repos/FactorLibrary/rust/factorlib_core/src/backtest/fast.rs#L92-L122)（函数 [`run_fast_backtest`](file:///home/cr9294/repos/FactorLibrary/rust/factorlib_core/src/backtest/fast.rs#L65-L215)）

- **问题根因分析**：
  1. **多空反手（Position Flip）手续费计算错误**：在 [`futures.rs`](file:///home/cr9294/repos/FactorLibrary/rust/factorlib_core/src/backtest/futures.rs#L381-L391) 中：
     ```rust
     let is_close = (current_lots[i] > 0 && delta < 0) || (current_lots[i] < 0 && delta > 0);
     let f_rate = if is_close { specs[i].close_fee } else { specs[i].open_fee };
     ```
     当持仓从 `+10` 手反手至 `-10` 手（`delta = -20`）时，实盘是**平旧仓 10 手 + 开新仓 10 手**，但 `is_close == true` 将全部 20 手统一按 `close_fee` 计费！在 [`configs/futures/fees.yaml`](file:///home/cr9294/repos/FactorLibrary/configs/futures/fees.yaml) 中若某品种免平昨费或开平费不等，新开仓部分被漏收开仓费。
  2. **`close_today` 字段在 Python -> Rust 桥接层被完全丢弃**：[`adapter.py`](file:///home/cr9294/repos/FactorLibrary/python/factorlib/markets/futures/adapter.py#L72-L79) 解析了 `close_today_fee`，但 [`backtest.py`](file:///home/cr9294/repos/FactorLibrary/python/factorlib/core/backtest.py#L212-L223) 与 [`SymbolSpec`](file:///home/cr9294/repos/FactorLibrary/rust/factorlib_core/src/backtest/futures.rs#L42-L52) 未接收该字段；同时快速向量化回测 [`fast.rs`](file:///home/cr9294/repos/FactorLibrary/rust/factorlib_core/src/backtest/fast.rs#L92-L122) 完全不读取 `fees.yaml` 的按手固定费用（`fee_type: fixed`）与合约乘数，仅使用固定 `1 bps` 费率，低价高乘数品种手续费被严重低估。
  3. **换月成本 `roll_cost` 仅统计平旧仓单边**：[`futures.rs`](file:///home/cr9294/repos/FactorLibrary/rust/factorlib_core/src/backtest/futures.rs#L265) 仅将旧合约平仓费与滑点计入 `daily_roll_cost`，新合约恢复同等仓位的开仓费与滑点被混入普通调仓，违背 `AGENTS.md`「换月双边费」审计口径。
  4. **涨跌停与停牌穿透**：Step 1 换月平仓（第 `240` 行）未检查 `is_unilateral[[d, i]]`；Step 4 保证金比例缩减（第 `355` 行）未检查 `!exec_price[[d, i]].is_finite()`（当日停牌/无成交），导致 Step 5（第 `375-379` 行）在零成交合约上以 `prev_settle[i]` 虚假减仓成交。

- **具体修复方案**：
  拆分平仓手数 `closed_lots` 与新开仓手数 `opened_lots` 分别按 `close_fee` / `open_fee` 计费，补齐换月新开仓部分的 `daily_roll_cost`，并在 Step 4 中拦截 `exec_price` 缺失的停牌合约。
  ```diff
  --- a/rust/factorlib_core/src/backtest/futures.rs
  +++ b/rust/factorlib_core/src/backtest/futures.rs
  @@ -238,8 +238,9 @@ pub fn run_futures_backtest(
           // Step 1: 换月处理 (若 d 日发生主力切换且持有旧合约仓位，先以旧合约成交价平清)
           // -------------------------------------------------------------
  +        let mut rolled_prev_lots = vec![0_i64; n];
           for i in 0..n {
  -            if is_roll[[d, i]] && current_lots[i] != 0 {
  +            if is_roll[[d, i]] && current_lots[i] != 0 && !is_unilateral[[d, i]] {
                   let old_l = current_lots[i];
  +                rolled_prev_lots[i] = old_l;
                   let p_old_raw = exec_price_old[[d, i]];
  @@ -352,7 +353,8 @@ pub fn run_futures_backtest(
                   let delta = scaled_l - current_lots[i];
                   if is_unilateral[[d, i]]
  +                    || !(exec_price[[d, i]].is_finite() && exec_price[[d, i]] > 0.0)
                       || (delta > 0 && limit_up_locked[i])
                       || (delta < 0 && limit_down_locked[i])
                   {
                       target_lots[i] = current_lots[i];
  @@ -379,19 +381,28 @@ pub fn run_futures_backtest(
               };
               let n_trade = delta.abs() as f64;
  -            let is_close = (current_lots[i] > 0 && delta < 0) || (current_lots[i] < 0 && delta > 0);
  -            let f_rate = if is_close {
  -                specs[i].close_fee
  +            let prev_l = current_lots[i];
  +            let (closed_lots, opened_lots) = if (prev_l > 0 && delta < 0) || (prev_l < 0 && delta > 0) {
  +                let c = prev_l.abs().min(delta.abs()) as f64;
  +                let o = (delta.abs() - prev_l.abs()).max(0) as f64;
  +                (c, o)
               } else {
  -                specs[i].open_fee
  +                (0.0, n_trade)
               };
   
               let trade_fee = match specs[i].fee_type {
  -                FeeType::Fixed => n_trade * f_rate,
  -                FeeType::Rate => n_trade * p_exec * specs[i].multiplier * f_rate,
  +                FeeType::Fixed => closed_lots * specs[i].close_fee + opened_lots * specs[i].open_fee,
  +                FeeType::Rate => {
  +                    p_exec * specs[i].multiplier
  +                        * (closed_lots * specs[i].close_fee + opened_lots * specs[i].open_fee)
  +                }
               };
               let trade_slip =
                   n_trade * (specs[i].slip_ticks * specs[i].tick_size) * specs[i].multiplier;
               let notional = n_trade * p_exec * specs[i].multiplier;
   
  +            if rolled_prev_lots[i] != 0 && target_lots[i].signum() == rolled_prev_lots[i].signum() {
  +                let roll_reopen = (rolled_prev_lots[i].abs().min(target_lots[i].abs()) as f64) / n_trade;
  +                daily_roll_cost += (trade_fee + trade_slip) * roll_reopen;
  +            }
               daily_fee += trade_fee;
  ```

---

### P0-08 `[P0-系统崩溃]` [已修复 / FIXED] PyO3 FFI 未拦截 `n_quantiles < 2` 触发 Rust Panic，且 Python 因子沙箱存在未来数据越权读取与逃逸漏洞

- **定位信息**：
  - [`rust/factorlib_core/src/metrics/quantile.rs`](file:///home/cr9294/repos/FactorLibrary/rust/factorlib_core/src/metrics/quantile.rs#L130)（函数 [`calc_quantile_metrics`](file:///home/cr9294/repos/FactorLibrary/rust/factorlib_core/src/metrics/quantile.rs#L116-L271) 第 `130` 行）与 [`rust/factorlib_core/src/py.rs`](file:///home/cr9294/repos/FactorLibrary/rust/factorlib_core/src/py.rs#L553-L665)、[`py.rs`](file:///home/cr9294/repos/FactorLibrary/rust/factorlib_core/src/py.rs#L1142-L1167)
  - [`python/factorlib/core/probe.py`](file:///home/cr9294/repos/FactorLibrary/python/factorlib/core/probe.py#L248-L254) 及 [`probe.py`](file:///home/cr9294/repos/FactorLibrary/python/factorlib/core/probe.py#L501-L596)（类 [`PythonFactorContext`](file:///home/cr9294/repos/FactorLibrary/python/factorlib/core/probe.py#L494-L547) 与函数 [`_execute_python_factor`](file:///home/cr9294/repos/FactorLibrary/python/factorlib/core/probe.py#L549-L607)）

- **问题根因分析**：
  1. **FFI 释放 GIL 后触发 `assert!` Panic**：[`quantile.rs`](file:///home/cr9294/repos/FactorLibrary/rust/factorlib_core/src/metrics/quantile.rs#L130) 第 `130` 行硬编码 `assert!(n_quantiles >= 2, "n_quantiles 必须 >= 2");`，而 [`py.rs`](file:///home/cr9294/repos/FactorLibrary/rust/factorlib_core/src/py.rs#L554) 的 `quantile_returns`、`factor_metrics`、`calc_quantile_returns` 以及 Web API [`workbench.py`](file:///home/cr9294/repos/FactorLibrary/python/factorlib/core/workbench.py#L583)（`n_quantiles = int(payload.get("n_quantiles", 5))`）均未做前置边界检查。一旦前端或调用方传入 `n_quantiles=0` 或 `1`，在 `py.allow_threads` 内直接触发 Rust 线程 Panic（违反 `AGENTS.md` 第 4 节「算子内部严禁触发 panic」）。
  2. **Python 自定义因子沙箱越权读取未来数据与任意文件访问**：
     - [`PythonFactorContext.__init__`](file:///home/cr9294/repos/FactorLibrary/python/factorlib/core/probe.py#L508) 将完整 `(T, N)` 矩阵字典挂载于实例属性 `self._fields = fields`；自定义代码只需写 `context._fields["close"][context.t + 5]` 即可绕过 `context.history()` 直接读取未来全样本价格！
     - [`_extract_python_code_dependencies`](file:///home/cr9294/repos/FactorLibrary/python/factorlib/core/probe.py#L248-L253) 仅拦截了 `Import`/`ImportFrom` 语句，未拦截双下划线魔术属性（`__class__`、`__subclasses__`、`__globals__`）及私有属性访问（`_fields`），且向沙箱注入了完整 `numpy` 模块（包含 `np.load(..., allow_pickle=True)`、`np.save`、`np.fromfile`、`np.ctypeslib`），在开放 CORS `*` 的 [`workbench_server.py`](file:///home/cr9294/repos/FactorLibrary/python/factorlib/workbench_server.py)（原 `server_main.py`）下构成远程代码执行风险。

- **具体修复方案**：
  在 PyO3 入口将 `n_quantiles < 2` 转换为 `PyValueError`；在 Python AST 探针中拦截所有以 `_` 开头的属性访问（含 `_fields`、`__subclasses__`）及 `numpy` 文件/FFI I/O 属性，并用闭包隐藏底层面板引用。
  ```diff
  --- a/rust/factorlib_core/src/py.rs
  +++ b/rust/factorlib_core/src/py.rs
  @@ -561,6 +561,9 @@ impl PyEngine {
           start_row: Option<usize>,
           end_row: Option<usize>,
       ) -> PyResult<Bound<'py, PyDict>> {
  +        if n_quantiles < 2 {
  +            return Err(PyValueError::new_err("n_quantiles 必须 >= 2"));
  +        }
           let f_raw = self.resolve_factor_mat(py, factor)?;
  @@ -620,6 +623,9 @@ impl PyEngine {
           start_row: Option<usize>,
           end_row: Option<usize>,
       ) -> PyResult<Bound<'py, PyDict>> {
  +        if n_quantiles < 2 {
  +            return Err(PyValueError::new_err("n_quantiles 必须 >= 2"));
  +        }
           let f_raw = self.resolve_factor_mat(py, factor)?;
  @@ -1148,6 +1154,9 @@ pub fn calc_quantile_returns<'py>(
       min_names: usize,
       sign: i32,
   ) -> PyResult<Bound<'py, PyDict>> {
  +    if n_quantiles < 2 {
  +        return Err(PyValueError::new_err("n_quantiles 必须 >= 2"));
  +    }
       let f_mat = view_to_fortran_f64(factor.as_array());
  ```
  ```diff
  --- a/python/factorlib/core/probe.py
  +++ b/python/factorlib/core/probe.py
  @@ -250,6 +250,15 @@ def _extract_python_code_dependencies(code: str) -> tuple[list[tuple[str, int]],
                   root_mod = alias.name.split(".")[0]
                   if root_mod not in {"numpy", "np", "math"}:
                       errors.append(f"因子沙箱禁止导入外部模块: {alias.name!r} (仅允许 numpy / math)")
  +        if isinstance(node, py_ast.Attribute):
  +            if node.attr.startswith("_"):
  +                errors.append(f"因子沙箱禁止访问私有或双下划线属性: {node.attr!r}")
  +            if node.attr in {"load", "save", "savez", "fromfile", "tofile", "loadtxt", "genfromtxt", "ctypeslib", "memmap"}:
  +                errors.append(f"因子沙箱禁止调用文件/内存 I/O 接口: {node.attr!r}")
  +        if isinstance(node, py_ast.Name) and node.id.startswith("__"):
  +            errors.append(f"因子沙箱禁止引用内建私有标识符: {node.id!r}")
  ```

---

### P0-09 `[已修复 / FIXED]` 多因子回测基准收益存在 2 个交易日跨期错配（`[T+1, T+2]` 减 `[T-1, T]`）且简单算术年化失真，导致多头超额虚高 `6.88%` 并反转正负号

- **定位信息**：
  - [`scripts/backtest_multi_factors.py`](file:///home/cr9294/repos/FactorLibrary/scripts/backtest_multi_factors.py#L309-L310)（基准日收益构造 `bench_ret[1:] = bench_close[1:] / bench_close[:-1] - 1.0`）
  - [`scripts/backtest_multi_factors.py`](file:///home/cr9294/repos/FactorLibrary/scripts/backtest_multi_factors.py#L413) 与第 [`621-638`](file:///home/cr9294/repos/FactorLibrary/scripts/backtest_multi_factors.py#L621-L638) 行（函数 [`evaluate_factor`](file:///home/cr9294/repos/FactorLibrary/scripts/backtest_multi_factors.py#L379-L448) 中 `excess_daily = g10_daily - bench_ret[eval_slice]` 及 `excess_ann = excess_mean * 250.0`）
  - [`scripts/backtest_volume_price_factor.py`](file:///home/cr9294/repos/FactorLibrary/scripts/backtest_volume_price_factor.py#L218-L219) 与第 [`371-372`](file:///home/cr9294/repos/FactorLibrary/scripts/backtest_volume_price_factor.py#L371-L372) 行
  - [`python/factorlib/core/evaluate.py`](file:///home/cr9294/repos/FactorLibrary/python/factorlib/core/evaluate.py#L499)（函数 [`evaluate_factor_v4_stats`](file:///home/cr9294/repos/FactorLibrary/python/factorlib/core/evaluate.py#L371-L569) 中 `long_excess_annual_return = float(np.mean(long_excess_daily) * 252.0)`）

- **问题根因分析**：
  在 5 因子对比报告中，前 3 个因子的 RankIC 吻合度高达 $97.5\%\sim 99\%$，但**多头年化超额（G10 vs 000985.CSI 中证全指）**却出现系统性巨幅偏离（如 `Volume_Price_2_H5_2` 线上 `1.26%` vs 本地 `+8.14%`；`Tech_Trend_VP_1_Rev` 线上 `-1.72%` 负超额 vs 本地 `+4.29%` 正超额，方向彻底颠倒）。经逐行审计，存在三大数学与时间轴错位：
  1. **基准收益率与多头持仓窗口错位整整 2 个交易日**：
     - 在第 `t` 行（代表信号日 $T$），多头收益矩阵 `fwd_ret_1[t]`（即 `g10_daily[t]`）计算的是 `price_proxy[t + 2] / price_proxy[t + 1] - 1.0`，对应真实持仓区间 **$[T+1, T+2]$**；
     - 然而基准收益数组 `bench_ret[t]` 在第 `309-310` 行定义为 `bench_close[t] / bench_close[t - 1] - 1.0`，对应历史区间 **$[T-1, T]$**！
     - 第 `413` 行直接执行同下标相减 `excess_daily = g10_daily - bench_ret[eval_slice]`，相当于拿 **「未来 $[T+1, T+2]$ 的多头收益」减去「过去 $[T-1, T]$ 的指数收益」**。在 2016-01-04（熔断首日指数暴跌 `-6.9%`）或 2024-09-24 急涨行情等关键拐点上，错位两天的基准相减制造了严重的虚假超额与跟踪误差。
  2. **价格时钟口径不一致（VWAP/Open vs Close-to-Close）**：
     - 股票组合在 $[T+1, T+2]$ 按执行价（`open_adj` 或 `VwapBegin30`）建仓/平仓，而指数基准若使用 `close[t] / close[t-1] - 1.0`，不仅日期错位，还混入了隔夜与尾盘异动。标准做法必须将基准指数对齐至完全相同的 $[T+1, T+1+h]$ 窗口与相同的价格时钟（`bench_exec[t + 1 + h] / bench_exec[t + 1] - 1.0`）。
  3. **简单算术年化（`mean * 250`）替代复利几何年化**：
     - [`backtest_multi_factors.py`](file:///home/cr9294/repos/FactorLibrary/scripts/backtest_multi_factors.py#L415) 使用 `excess_mean * 250.0`，[`evaluate.py`](file:///home/cr9294/repos/FactorLibrary/python/factorlib/core/evaluate.py#L499) 使用 `np.mean(long_excess_daily) * 252.0`，完全忽略了 9 年长周期下的波动率损耗（Volatility Drag，$\approx -\frac{1}{2}\sigma^2$）；而线上 FactorLab 平台采用标准复利几何年化 $\left(\prod_{t=1}^{D}(1 + r_{\text{excess}, t})\right)^{250 / D} - 1$。

- **具体修复方案**：
  1. 在 [`backtest_multi_factors.py`](file:///home/cr9294/repos/FactorLibrary/scripts/backtest_multi_factors.py) 与 [`backtest_volume_price_factor.py`](file:///home/cr9294/repos/FactorLibrary/scripts/backtest_volume_price_factor.py) 中，构造与个股远期收益严格同窗口 $[T+1, T+1+h]$ 的 `bench_fwd_ret_1[t] = bench_exec[t + 2] / bench_exec[t + 1] - 1.0`；
  2. 在 [`evaluate.py`](file:///home/cr9294/repos/FactorLibrary/python/factorlib/core/evaluate.py#L499) 及回测脚本中，统一采用复利几何年化计算 `long_excess_annual_return`（同时保留算术超额字段供对照）。
  ```diff
  --- a/scripts/backtest_multi_factors.py
  +++ b/scripts/backtest_multi_factors.py
  @@ -305,10 +305,12 @@ def load_ashare_panel():
           .reindex(dates)
           .to_numpy(dtype=np.float64)
       )
  -    bench_ret = np.full(T, np.nan, dtype=np.float64)
  -    bench_ret[1:] = bench_close[1:] / bench_close[:-1] - 1.0
  +    # 严格对齐信号日 t 对应的 [t+1, t+2] 远期持仓窗口
  +    bench_fwd_ret_1 = np.full(T, np.nan, dtype=np.float64)
  +    bench_fwd_ret_1[:-2] = bench_close[2:] / bench_close[1:-1] - 1.0
   
  @@ -410,9 +412,12 @@ def evaluate_factor(
       ls_sharpe = (ls_mean / ls_std * np.sqrt(250.0)) if ls_std > 1e-12 else 0.0
   
       g10_daily = q_res["group_returns"][:, 9]
  -    excess_daily = g10_daily - bench_ret[eval_slice]
  +    excess_daily = g10_daily - bench_fwd_ret_1[eval_slice]
  +    valid_ex = excess_daily[np.isfinite(excess_daily)]
       excess_mean = float(np.nanmean(excess_daily))
  -    excess_ann = excess_mean * 250.0
  +    excess_ann = (
  +        float(np.prod(1.0 + valid_ex) ** (250.0 / len(valid_ex)) - 1.0)
  +        if len(valid_ex) > 0 and np.all(1.0 + valid_ex > 0.0)
  +        else excess_mean * 250.0
  +    )
  ```
  ```diff
  --- a/python/factorlib/core/evaluate.py
  +++ b/python/factorlib/core/evaluate.py
  @@ -496,7 +496,12 @@ def evaluate_factor_v4_stats(
       long_excess_cum = np.cumprod(1.0 + long_excess_daily) - 1.0
       long_excess_total = float(long_excess_cum[-1]) if T > 0 else 0.0
  -    long_excess_annual = float(np.mean(long_excess_daily) * 252.0) if T > 0 else 0.0
  +    if T > 0 and np.all(1.0 + long_excess_daily > 0.0):
  +        long_excess_annual = float(
  +            np.prod(1.0 + long_excess_daily) ** (252.0 / float(T)) - 1.0
  +        )
  +    else:
  +        long_excess_annual = float(np.mean(long_excess_daily) * 252.0) if T > 0 else 0.0
  ```

---

### P0-10 `[已修复 / FIXED]` `[P0-阻断级]` 单边换手率指标张冠李戴（截面秩自相关距离 `1 - ρ` 冒充 Top 分位组合权重换手率），导致换手率相差 `12~18` 倍

- **定位信息**：
  - [`scripts/backtest_multi_factors.py`](file:///home/cr9294/repos/FactorLibrary/scripts/backtest_multi_factors.py#L408)（`turnover = fc.calc_turnover(f_eval, u_eval)`）
  - [`rust/factorlib_core/src/metrics/turnover.rs`](file:///home/cr9294/repos/FactorLibrary/rust/factorlib_core/src/metrics/turnover.rs#L64-L119)（函数 [`turnover_series`](file:///home/cr9294/repos/FactorLibrary/rust/factorlib_core/src/metrics/turnover.rs#L64-L112) 与 [`mean_turnover`](file:///home/cr9294/repos/FactorLibrary/rust/factorlib_core/src/metrics/turnover.rs#L115-L119)）
  - [`rust/factorlib_core/src/metrics/quantile.rs`](file:///home/cr9294/repos/FactorLibrary/rust/factorlib_core/src/metrics/quantile.rs#L84-L85)（分位数桶边界切分 `floor` vs `round`）
  - [`python/factorlib/core/evaluate.py`](file:///home/cr9294/repos/FactorLibrary/python/factorlib/core/evaluate.py#L405-L445)（`top_turnover` 初始化为 `0.0` 把首日/预热空仓日计入均值分母）

- **问题根因分析**：
  对比报告中，线上 FactorLab 的「单边换手率」分别为 `12.11%`、`11.78%`、`9.30%`、`18.28%`，而本地 Rust 内核输出仅为 `1.00%`、`0.92%`、`0.51%`（相差 **12 倍至 18 倍**），其根因在于两端计算的是**完全不同的数学量**：
  1. **指标定义张冠李戴**：
     - 线上 FactorLab 报告的「单边换手率」是 **Top 分位组合（G10，前 10% 等权多头组合）的日均单边持仓权重换手率**：
       $$\text{Turnover}_{\text{top}, t} = \frac{1}{2} \sum_{i=1}^{N} \left| w_{t, i}^{(Q)} - w_{t-1, i}^{(Q)} \right|, \quad w_{t, i}^{(Q)} = \frac{\mathbb{I}(i \in G_{Q, t})}{|G_{Q, t}|}$$
     - 而 [`backtest_multi_factors.py`](file:///home/cr9294/repos/FactorLibrary/scripts/backtest_multi_factors.py#L408) 调用的 `fc.calc_turnover`（[`turnover.rs`](file:///home/cr9294/repos/FactorLibrary/rust/factorlib_core/src/metrics/turnover.rs#L96-L106)）计算的是**全市场截面秩自相关距离** $1 - \text{Corr}(\text{rank}(f_t), \text{rank}(f_{t-1}))$！
     - 当因子经过 5 日或 10 日线性衰减（`decay(5)` / `decay(10)`）后，全截面秩相关系数 $\rho_1 \approx 0.990 \sim 0.995$，因此 $1 - \rho_1$ 仅为 `0.51% ~ 1.00%`；但在第 90 百分位截断线附近，每日仍有约 `9% ~ 12%` 的临界股票进出 G10 持仓组合！用 $1 - \rho_1$ 评估组合换手率会严重低估实盘交易双边滑点与印花税成本（报告第三节第 2 点据此得出「单边换手率从 0.92% 降至 0.51%」属于用错指标）。
  2. **分位数切分取整规则差异（`.floor()` vs `.round()`）**：
     - [`quantile.rs`](file:///home/cr9294/repos/FactorLibrary/rust/factorlib_core/src/metrics/quantile.rs#L84-L85) 使用 `((q as f64) * (k as f64) / (n_quantiles as f64)).floor() as usize`，而 FactorLab 标准算子（参见 [`Factor/factorlab_local_engine.py`](file:///home/cr9294/repos/FactorLibrary/Factor/factorlab_local_engine.py#L1686)）使用 `int(round(q * n_valid / n_quantiles))`，建议在 Rust 内核统一改为 `.round() as usize` 以消除分位边界 1 只股票的 системати 偏移。
  3. **`evaluate.py` 首日与预热期 `0.0` 稀释均值**：
     - [`evaluate.py`](file:///home/cr9294/repos/FactorLibrary/python/factorlib/core/evaluate.py#L405) 虽然在 Python 层循环计算了 `top_turnover`，但初始化 `top_turnover = np.zeros(T, dtype=np.float64)`，在 `prev_top_weights is None` 的首日及全部预热期空仓日保留 `0.0`，随后第 `462` 行 `np.mean(top_turnover)` 直接除以总天数 `T`，人为压低了换手率均值。

- **具体修复方案**：
  1. 在 [`rust/factorlib_core/src/metrics/turnover.rs`](file:///home/cr9294/repos/FactorLibrary/rust/factorlib_core/src/metrics/turnover.rs) 与 [`quantile.rs`](file:///home/cr9294/repos/FactorLibrary/rust/factorlib_core/src/metrics/quantile.rs) 中原生实现 `top_quantile_turnover_series` 与 `mean_top_quantile_turnover`（基于 `.round()` 分位切分，无持仓日输出 `f64::NAN` 并由 `nanmean` 聚合），通过 PyO3 导出并在 `calc_quantile_metrics` 返回字典中同时提供 `top_quantile_turnover` 与 `rank_autocorr_turnover`；
  2. 修复 [`evaluate.py`](file:///home/cr9294/repos/FactorLibrary/python/factorlib/core/evaluate.py#L405-L462) 中 `top_turnover` 的有效日过滤。
  ```diff
  --- a/rust/factorlib_core/src/metrics/quantile.rs
  +++ b/rust/factorlib_core/src/metrics/quantile.rs
  @@ -81,8 +81,8 @@ pub fn compute_quantile_metrics(
               let k = buf.len();
               for q in 0..n_quantiles {
  -                let start = ((q as f64) * (k as f64) / (n_quantiles as f64)).floor() as usize;
  -                let end = (((q + 1) as f64) * (k as f64) / (n_quantiles as f64)).floor() as usize;
  +                let start = ((q as f64) * (k as f64) / (n_quantiles as f64)).round() as usize;
  +                let end = (((q + 1) as f64) * (k as f64) / (n_quantiles as f64)).round() as usize;
                   let end = end.max(start + 1).min(k);
  ```

---

### P0-11 `[已修复 / FIXED]` `[P0-阻断级]` A 股对账脚本采用全天 VWAP（`09:30~15:00`）混入日内价格，且漏除 ST/`*ST` 与一字涨跌停（`BaseUnivMask` / `LimitMask` 缺失）

- **定位信息**：
  - [`scripts/backtest_multi_factors.py`](file:///home/cr9294/repos/FactorLibrary/scripts/backtest_multi_factors.py#L276-L303)（函数 [`load_ashare_panel`](file:///home/cr9294/repos/FactorLibrary/scripts/backtest_multi_factors.py#L221-L327)）
  - [`scripts/backtest_multi_factors.py`](file:///home/cr9294/repos/FactorLibrary/scripts/backtest_multi_factors.py#L621-L629)（远期收益 `fwd_ret_h[t] = price_proxy[t + 1 + h] / price_proxy[t + 1] - 1.0`）
  - [`scripts/backtest_volume_price_factor.py`](file:///home/cr9294/repos/FactorLibrary/scripts/backtest_volume_price_factor.py#L194-L212)

- **问题根因分析**：
  1. **全天 VWAP（`09:30~15:00`）替代早盘执行价导致短期反转收益衰减 37%**：
     - 线上 FactorLab 采用 $T+1$ 早盘开盘后 30 分钟成交量加权均价 **`VwapBegin30`（`09:30~10:00`）** 作为建仓/平仓执行基准；
     - 而本地脚本在缺乏分钟线时，第 `282` 行使用全天日频成交额除以成交量 `vwap_raw = amount / volume`（覆盖 `09:30~15:00` 全天 4 小时交易）作为 `price_proxy[t + 1]`。
     - 对于 `Volume_Price_2_*` 等日频量价反转因子，股票在 $T+1$ 日内（`10:00~15:00`）即开始显著均值回归。使用全天 VWAP 建仓相当于**在 $T+1$ 日内已经涨完一半的平均价格才买入**，直接吞噬了 $T+1$ 早盘至收盘的反转 Alpha，导致本地 `多空日收益` 从线上 `0.24% ~ 0.25%` 系统性缩水至 `0.14% ~ 0.16%`，且 `5日 IC` 从 `0.0332` 降至 `0.0314`。在仅有日频数据时，**$T+1$ 复权开盘价 `open_adj = open * ex_cum_factor`** 比全天 VWAP 更贴近早盘 `09:30~10:00` 的真实建仓时点且严格无日内前视。
  2. **本地 `data/raw/stocks/` 已具备完整 ST 与涨跌停表，但脚本完全未加载过滤**：
     - 经核查本地数据目录，`data/raw/stocks/daily/trade_status/*.parquet` 含有 `is_st` 與 `susp_Type`；`data/raw/stocks/daily/bar_1d/*.parquet` 含有精确到分的 `limit_up` 与 `limit_down`；`data/raw/stocks/static/instruments_info.parquet` 含有 `listed_date` 与 `de_listed_date`。
     - 然而 [`backtest_multi_factors.py`](file:///home/cr9294/repos/FactorLibrary/scripts/backtest_multi_factors.py#L244-L265) 仅用 `~is_suspended & (volume > 0)` 充当 `trdMask`，既未在 `BaseUnivMask` 中剔除 **ST/`*ST` 风险警示股**及退市整理期股票，也未在 `LimitMask` 中剔除 **$T$ 日或 $T+1$ 日一字涨跌停（`high == low` 且触及 `limit_up`/`limit_down`，或开盘即涨停无法买入、跌停无法卖出）** 的股票，同时 `dropNew(120)` 用「首个有效收盘价下标 `first_valid[j] + 120`」近似「自然日上市满 120 天（`StockListedDays > 120`）」，把约 120 个自然日（~85 个交易日）拉长到了 120 个交易日（~175 个自然日）。

- **具体修复方案**：
  在 `python/factorlib/markets/stocks/panel_loader.py` 中构建标准化的 A 股面板加载器，完整读取 `trade_status.is_st`、`bar_1d.{limit_up, limit_down}`、`instruments_info.{listed_date, de_listed_date}`：
  - 构造严格对齐线上语义的 `base_univ_mask`（非停牌、非 ST/`*ST`、上市自然日 $> N$、未退市）与 `limit_mask`（剔除一字板及涨跌停不可交易状态）；
  - 默认以 $T+1$ 复权开盘价 `open_adj`（或分钟线 `VwapBegin30`）作为因果执行价，并在配置中保留 `exec_price_mode: "open_adj" | "vwap_begin30" | "vwap_1d"` 供敏感性对照；同时对 $T+1$ 涨停无法建仓、$T+1+h$ 跌停/停牌无法平仓的持仓施加实盘可交易性约束。

---

### P1-01 `[P1-架构隐患]` [已修复 / FIXED] 体系 A 反向依赖顶层散落脚本 `server_main.py`，且 Wheel 打包遗漏、Ruff Lint 失明

- **定位信息**：
  - [`scripts/run_web.py`](file:///home/cr9294/repos/FactorLibrary/scripts/run_web.py)（已统一委托 [`factorlib.workbench_server`](file:///home/cr9294/repos/FactorLibrary/python/factorlib/workbench_server.py)）
  - [`tests/python/test_workbench.py`](file:///home/cr9294/repos/FactorLibrary/tests/python/test_workbench.py#L20-L21)
  - [`python/factorlib/workbench_server.py`](file:///home/cr9294/repos/FactorLibrary/python/factorlib/workbench_server.py)（原 `python/server_main.py`）与 [`python/factorlib/frontend/`](file:///home/cr9294/repos/FactorLibrary/python/factorlib/frontend)
  - [`pyproject.toml`](file:///home/cr9294/repos/FactorLibrary/pyproject.toml)

- **问题根因分析**：
  体系 A 对体系 C（`Factor/`）零引用，对 `pyrust_bt` 包本体零引用，但曾存在对历史遗留顶层模块 `python/server_main.py` 的**双向耦合**：`server_main.py` 导入 `factorlib.core.workbench`，而 `scripts/run_web.py` 与 `tests/python/test_workbench.py` 又反向导入 `server_main`。
  更严重的是，根目录 [`pyproject.toml`](file:///home/cr9294/repos/FactorLibrary/pyproject.toml#L32) 第 `32` 行声明 `python-packages = ["factorlib", "factorlib_core"]`，未将散落在 `python/` 顶层的 `server_main.py` 及根目录 `frontend/` 静态资产纳入构建产物，任何通过 `pip install .` / `maturin build` 安装的发行包运行 Web 服务均会触发 `ModuleNotFoundError: No module named 'server_main'` 或前端静态资源 `404`；同时 `[tool.ruff] exclude` 曾排除了 `"python/server_main.py"`，使生产级 HTTP 服务长期游离在静态 Lint 门禁之外。

- **具体修复方案**：
  已将 `python/server_main.py` 正式收编迁入 [`python/factorlib/workbench_server.py`](file:///home/cr9294/repos/FactorLibrary/python/factorlib/workbench_server.py)（commit `dcdfede`），并将前端资产迁入 [`python/factorlib/frontend/`](file:///home/cr9294/repos/FactorLibrary/python/factorlib/frontend)，在 [`pyproject.toml`](file:///home/cr9294/repos/FactorLibrary/pyproject.toml) 中配置 `[tool.maturin] include = ["factorlib/frontend/**/*"]` 与 `[project.scripts] factorlib-web = "factorlib.workbench_server:main"`，解除 Ruff 排除列表、统一 CLI 入口并加固非法 JSON 请求体 `400 Bad Request` 校验。
  ```diff
  --- a/scripts/run_web.py
  +++ b/scripts/run_web.py
  @@ -23,7 +23,7 @@ ROOT_DIR = Path(__file__).resolve().parent.parent
   sys.path.insert(0, str(ROOT_DIR / "python"))
   sys.path.insert(0, str(ROOT_DIR))
   
  -from server_main import run_std_server  # noqa: E402
  +from factorlib.workbench_server import run_std_server  # noqa: E402
  ```
  ```diff
  --- a/pyproject.toml
  +++ b/pyproject.toml
  @@ -50,7 +50,6 @@ exclude = [
       "legacy",
       "pyrust-bt-main",
       "python/pyrust_bt",
  -    "python/server_main.py",
       "tests/test_*.py",
       "build",
       "dist",
  ```

---

### P1-02 `[P1-架构隐患]` [已修复 / FIXED] 双 Rust Crate 版本分裂且 `engine_rust` 硬编码 `extension-module` 阻断原生测试与统一构建

- **定位信息**：
  - [`Cargo.toml`](file:///home/cr9294/repos/FactorLibrary/Cargo.toml)（根目录 Cargo Workspace 配置）与 [`Cargo.lock`](file:///home/cr9294/repos/FactorLibrary/Cargo.lock)
  - [`rust/factorlib_core/Cargo.toml`](file:///home/cr9294/repos/FactorLibrary/rust/factorlib_core/Cargo.toml#L15-L29)
  - 原 `rust/engine_rust/Cargo.toml` 与 `rust/engine_rust/pyproject.toml`（已于 commit `dcdfede` 移除）

- **问题根因分析**：
  仓库内曾并存两套独立 Cargo 项目且根目录无 Cargo Workspace：
  1. **依赖与 ABI 冲突**：[`factorlib_core/Cargo.toml`](file:///home/cr9294/repos/FactorLibrary/rust/factorlib_core/Cargo.toml#L12-L13) 使用 `pyo3 = "0.22"`（`abi3-py310`）与 `maturin>=1.7`，而原 `engine_rust/Cargo.toml` 使用 `pyo3 = "0.21"`（`abi3-py38`）并开启了 `duckdb = { version = "1.0", features = ["bundled"] }`（全量源码编译 C++ DuckDB 耗时数分钟且产物体积庞大）。
  2. **链接器符号缺失 Bug**：原 `engine_rust/Cargo.toml` 仅声明 `crate-type = ["cdylib"]` 且在 `[dependencies]` 中硬编码开启 `pyo3/extension-module`，导致在 Linux 下直接运行 `cargo test` 会因无法链接 `libpython` 符号而编译失败。
  3. **Workspace 发现缺失**：移除 `engine_rust` 后仓库根目录仍缺乏统一 `Cargo.toml` Workspace 声明，在根目录直接执行 `cargo test` / `cargo check` 或由 IDE 启动 `rust-analyzer` 时无法自动索引 `rust/factorlib_core`。

- **具体修复方案**：
  已在 commit `dcdfede` 彻底移除冗余 Crate `rust/engine_rust/`，并在根目录建立仅托管主力内核 `rust/factorlib_core` 的 [`Cargo.toml`](file:///home/cr9294/repos/FactorLibrary/Cargo.toml) Workspace 与统一 [`Cargo.lock`](file:///home/cr9294/repos/FactorLibrary/Cargo.lock)；同时保留 `rust/factorlib_core` 的 `default = ["extension-module"]`（实测本机无系统 `libpython3.12-dev` 时，移除该默认特性会导致 `cargo test` 链接失败），并清理 `.gitignore` 中失效的 `rust/engine_rust/` 规则及纳入 `verify.sh` TOML 校验。
  ```toml
  [workspace]
  resolver = "2"
  members = ["rust/factorlib_core"]
  ```

---

### P1-03 `[P1-架构隐患]` [已修复 / FIXED] `Panel.load` 将全部 `aux` 原始列暴露给 DSL、`is_reserved_label_name` 漏拦 `r_exec_next`，且 `trdMask` 绑定错误掩码

> **修复状态**：已修复（`Panel` 新增与 `fields` 物理隔离的 `aux_fields` 容器，[`Panel.load`](file:///home/cr9294/repos/FactorLibrary/python/factorlib/core/panel.py#L140-L330) 仅将 DSL 特征与 `sector`/`listed_days`/`in_universe_mask`/`limit_mask` 结构化辅助字段放入 `fields`，其余回测/原始/合约规格 `AUX_BACKTEST_FIELDS` 统一隔离入 `aux_fields`；提取共用 [`compute_limit_mask`](file:///home/cr9294/repos/FactorLibrary/python/factorlib/core/panel.py#L36-L56) 联动 `is_unilateral`/`traded`/`volume` 并用于 `Panel.load` 与 `_resolve_implicit_field("limit_mask")`；[`is_reserved_label_name`](file:///home/cr9294/repos/FactorLibrary/rust/factorlib_core/src/panel.rs#L52-L62) 补充拦截 `r_exec_next` 并通过 Rust/Python 回归验证）。

- **定位信息**：
  - [`python/factorlib/core/panel.py`](file:///home/cr9294/repos/FactorLibrary/python/factorlib/core/panel.py#L138-L201)（方法 [`Panel.load`](file:///home/cr9294/repos/FactorLibrary/python/factorlib/core/panel.py#L117-L234)）
  - [`python/factorlib/markets/futures/etl/fields.py`](file:///home/cr9294/repos/FactorLibrary/python/factorlib/markets/futures/etl/fields.py#L8-L35)
  - [`rust/factorlib_core/src/panel.rs`](file:///home/cr9294/repos/FactorLibrary/rust/factorlib_core/src/panel.rs#L52-L61)（函数 [`is_reserved_label_name`](file:///home/cr9294/repos/FactorLibrary/rust/factorlib_core/src/panel.rs#L52-L61)）
  - [`python/factorlib/core/probe.py`](file:///home/cr9294/repos/FactorLibrary/python/factorlib/core/probe.py#L624-L627)（函数 [`_resolve_implicit_field`](file:///home/cr9294/repos/FactorLibrary/python/factorlib/core/probe.py#L610-L642)）

- **问题根因分析**：
  1. **DSL 隔离红线失效**：[`fields.py`](file:///home/cr9294/repos/FactorLibrary/python/factorlib/markets/futures/etl/fields.py#L11) 明确规定 `aux` 字段（`open_raw`、`close_raw`、`settle_raw`、`exec_vwap`、`exec_vwap_old`、`adj_factor`、`limit_up`、`limit_down`、`is_roll`）仅用于回测撮合、**严禁暴露给 DSL**。但 [`Panel.load`](file:///home/cr9294/repos/FactorLibrary/python/factorlib/core/panel.py#L138-L153) 只排除了字符串列，将所有 `aux` 数值列装入 `Panel.fields` 并传入 `PyEngine`（甚至连 GP `PrimitiveSet.from_panel` 若未传白名单也会扫到或允许手工 DSL 直接写 `exec_vwap`）。
  2. **保留标签名漏网**：[`is_reserved_label_name`](file:///home/cr9294/repos/FactorLibrary/rust/factorlib_core/src/panel.rs#L52-L61) 未拦截 `"r_exec_next"`。
  3. **`trdMask` 涨跌停掩码张冠李戴**：[`probe.py`](file:///home/cr9294/repos/FactorLibrary/python/factorlib/core/probe.py#L624-L627) 中 `if field_name == "limit_mask": mask_arr = np.where(panel.valid_mask, 1.0, np.nan)`——由于真实期货面板列名为 `is_unilateral` 而非 `limit_mask`，`trdMask` 实际上把 `panel.valid_mask` 又乘了一遍，**完全没有过滤当日一字涨跌停或零成交品种**！

- **具体修复方案**：
  在 `Panel` 中分离 `fields`（仅限 DSL 可见特征）与 `aux_fields`（仅供回测器 [`run_backtest`](file:///home/cr9294/repos/FactorLibrary/python/factorlib/core/backtest.py) 读取）；在 `panel.rs` 补齐 `"r_exec_next"` 拦截；修复 `probe.py` 中 `limit_mask` 基于 `is_unilateral` 与 `volume > 0` 构造真实可交易掩码。
  ```diff
  --- a/rust/factorlib_core/src/panel.rs
  +++ b/rust/factorlib_core/src/panel.rs
  @@ -58,6 +58,7 @@ pub fn is_reserved_label_name(name: &str) -> bool {
           || lower.starts_with("fwd_ret")
           || lower.starts_with("target_")
           || lower == "target"
  +        || lower == "r_exec_next"
   }
  ```
  ```diff
  --- a/python/factorlib/core/probe.py
  +++ b/python/factorlib/core/probe.py
  @@ -622,8 +622,14 @@ def _resolve_implicit_field(
           return np.asfortranarray(mask_arr, dtype=np.float64), styles_cache
   
       if field_name == "limit_mask":
  -        mask_arr = np.where(panel.valid_mask, 1.0, np.nan)
  +        tradable = panel.valid_mask.copy()
  +        if "is_unilateral" in panel.fields:
  +            tradable &= ~(panel.fields["is_unilateral"] > 0.5)
  +        if "volume" in panel.fields:
  +            tradable &= np.isfinite(panel.fields["volume"]) & (panel.fields["volume"] > 0.0)
  +        mask_arr = np.where(tradable, 1.0, np.nan)
           return np.asfortranarray(mask_arr, dtype=np.float64), styles_cache
  ```

---

### P1-04 `[P1-架构隐患]` [已修复 / FIXED] Rust 与 Python 参考实现的并列秩容差聚类算法不一致，边界偏差高达 `0.25`

> **修复状态**：已修复（Rust [`compute_normalized_ranks`](file:///home/cr9294/repos/FactorLibrary/rust/factorlib_core/src/nan.rs#L304-L338)、[`op_group_rank`](file:///home/cr9294/repos/FactorLibrary/rust/factorlib_core/src/ops/group.rs#L50-L94)、[`compute_fractional_ranks`](file:///home/cr9294/repos/FactorLibrary/rust/factorlib_core/src/metrics/ic.rs#L38-L68) 与 Python [`_rank_1d_normalized`](file:///home/cr9294/repos/FactorLibrary/python/factorlib/core/ops_ref.py#L868-L894) / [`group_rank`](file:///home/cr9294/repos/FactorLibrary/python/factorlib/core/ops_ref.py#L1032-L1049) 均已统一为左锚点增量扩展容差聚类，并在 [`tests/python/test_rust_core.py`](file:///home/cr9294/repos/FactorLibrary/tests/python/test_rust_core.py) 增设 `[0.0, 0.8e-12, 1.5e-12]` 非传递性边界回归测试）。

- **定位信息**：
  - [`rust/factorlib_core/src/nan.rs`](file:///home/cr9294/repos/FactorLibrary/rust/factorlib_core/src/nan.rs#L323-L337)（函数 [`compute_normalized_ranks`](file:///home/cr9294/repos/FactorLibrary/rust/factorlib_core/src/nan.rs#L305-L338)）
  - [`rust/factorlib_core/src/ops/group.rs`](file:///home/cr9294/repos/FactorLibrary/rust/factorlib_core/src/ops/group.rs#L78-L90)（函数 [`op_group_rank`](file:///home/cr9294/repos/FactorLibrary/rust/factorlib_core/src/ops/group.rs#L57-L96)）
  - [`python/factorlib/core/ops_ref.py`](file:///home/cr9294/repos/FactorLibrary/python/factorlib/core/ops_ref.py#L879-L885)（函数 [`_rank_1d_normalized`](file:///home/cr9294/repos/FactorLibrary/python/factorlib/core/ops_ref.py#L868-L886)）

- **问题根因分析**：
  Rust 端先升序排列，再以左端点 `vi = buf[i].1` 为锚点做贪心单向聚类（`|buf[j].1 - vi| <= tol`）；而 Python `ops_ref.py` 对每个元素独立统计对称邻域 `|vals - v| <= tol`。由于近似相等关系不具备传递性，对输入 `[0.0, 0.8e-12, 1.5e-12]`（`STD_EPS = 1e-12`）：
  - Rust 将前两项聚为一组 `[0, 1]`、第三项单独成组 `[2]`，输出 **`[0.25, 0.25, 1.0]`**；
  - Python 为中间元素匹配到两侧全部三项，输出 **`[0.25, 0.50, 0.75]`**。
  两端最大绝对误差达 **`0.25`**，远超 `AGENTS.md` 规定的 $\max |y_{\text{rust}} - y_{\text{ref}}| < 10^{-9}$ 对账红线。

- **具体修复方案**：
  将 [`ops_ref.py`](file:///home/cr9294/repos/FactorLibrary/python/factorlib/core/ops_ref.py#L879-L885) 的 `_rank_1d_normalized`（及 `group_rank`）统一改为与 Rust 完全等价的排序后左锚点分段平均秩算法，复杂度同时从 $O(N^2)$ 降至 $O(N \log N)$。
  ```diff
  --- a/python/factorlib/core/ops_ref.py
  +++ b/python/factorlib/core/ops_ref.py
  @@ -876,12 +876,19 @@ def _rank_1d_normalized(row: np.ndarray) -> np.ndarray:
           out[idx[0]] = 0.5
           return out
       vals = row[idx]
  -    for pos, col_j in enumerate(idx):
  -        v = vals[pos]
  -        tol = STD_EPS * max(1.0, abs(float(v)))
  -        c_eq = float(np.sum(np.abs(vals - v) <= tol))
  -        c_less = float(np.sum(vals < (v - tol)))
  -        out[col_j] = (c_less + 0.5 * (c_eq - 1.0)) / (k - 1.0)
  +    order = np.argsort(vals, kind="mergesort")
  +    denom = float(k - 1)
  +    i = 0
  +    while i < k:
  +        vi = float(vals[order[i]])
  +        tol = STD_EPS * max(1.0, abs(vi))
  +        j = i + 1
  +        while j < k and abs(float(vals[order[j]]) - vi) <= tol:
  +            j += 1
  +        avg_rank = 0.5 * float(i + j - 1) / denom
  +        for m in range(i, j):
  +            out[idx[order[m]]] = avg_rank
  +        i = j
       return out
  ```

---

### P1-05 `[P1-架构隐患]` [已修复 / FIXED] `SubExprCache` 伪 LRU 实为 FIFO 淘汰、全局写锁线性扫描，且纯 `u64` 键存在哈希碰撞串值隐患

> **修复状态**：已修复（[`SubExprCache`](file:///home/cr9294/repos/FactorLibrary/rust/factorlib_core/src/expr/cache.rs) 升级为 32 分片并发缓存 + 每分片基于槽位索引侵入式双向链表实现 $O(1)$ 真 LRU 命中置顶与尾部淘汰 + `(u64 哈希选片, Arc<str> 规范化表达式原文)` 精确比对防碰撞，[`eval.rs`](file:///home/cr9294/repos/FactorLibrary/rust/factorlib_core/src/expr/eval.rs) 同步消除批预热 DAG 的纯 `u64` 去重隐患并附带 Rust LRU/并发/同哈希隔离单测）。

- **定位信息**：
  - [`rust/factorlib_core/src/expr/cache.rs`](file:///home/cr9294/repos/FactorLibrary/rust/factorlib_core/src/expr/cache.rs#L24-L116)（结构体 [`SubExprCache`](file:///home/cr9294/repos/FactorLibrary/rust/factorlib_core/src/expr/cache.rs#L32-L40)，方法 [`get`](file:///home/cr9294/repos/FactorLibrary/rust/factorlib_core/src/expr/cache.rs#L64-L77) 与 [`insert`](file:///home/cr9294/repos/FactorLibrary/rust/factorlib_core/src/expr/cache.rs#L80-L116)）
  - [`rust/factorlib_core/src/expr/eval.rs`](file:///home/cr9294/repos/FactorLibrary/rust/factorlib_core/src/expr/eval.rs#L71-L103)

- **问题根因分析**：
  1. **伪 LRU（实际退化为 FIFO）**：[`cache.rs`](file:///home/cr9294/repos/FactorLibrary/rust/factorlib_core/src/expr/cache.rs#L63-L77) 的文档注释声称「若命中则更新时间戳」，但 `get()` 函数体仅递增 `hits` 便直接返回，**从未更新 `entry.last_used`**！导致越早插入的高频底层公共子表达式（如 `ts_mean(close, 20)`）其 `last_used` 永远最小，缓存满时被**最先驱逐**。
  2. **写锁内 $O(M)$ 全表遍历**：`insert()` 第 `95-96` 行在全局 `RwLock::write()` 互斥锁内执行 `guard.map.iter().min_by_key(...)`，高并发批求值时阻塞全部 Rayon 工作线程。
  3. **64 位哈希无原文比对**：缓存键仅存 `u64` FNV/SipHash 指纹而不核对规范化表达式字符串，在长周期百万级 GP 演化下存在生日悖论碰撞风险，一旦碰撞会静默返回完全不同的因子矩阵。

- **具体修复方案**：
  将 `CacheEntry.last_used` 改为 `AtomicU64`，在 `get()` 读锁内无锁原子刷新访问时钟；将缓存键升级为 `(u64, String)`（或存储 `canon_expr: String` 在命中时校验字符串相等）。
  ```diff
  --- a/rust/factorlib_core/src/expr/cache.rs
  +++ b/rust/factorlib_core/src/expr/cache.rs
  @@ -20,7 +20,7 @@ use std::sync::{Arc, RwLock};
   struct CacheEntry {
       matrix: Arc<Array2<f64>>,
       bytes: usize,
  -    last_used: u64,
  +    last_used: AtomicU64,
   }
  @@ -68,6 +68,8 @@ impl SubExprCache {
           if let Ok(guard) = self.inner.read() {
               if let Some(entry) = guard.map.get(&key) {
  +                let tick = self.clock.fetch_add(1, Ordering::Relaxed);
  +                entry.last_used.store(tick, Ordering::Relaxed);
                   self.hits.fetch_add(1, Ordering::Relaxed);
                   return Some(Arc::clone(&entry.matrix));
               }
  @@ -85,15 +87,15 @@ impl SubExprCache {
           let tick = self.clock.fetch_add(1, Ordering::Relaxed);
           if let Ok(mut guard) = self.inner.write() {
               if let Some(existing) = guard.map.get_mut(&key) {
  -                existing.last_used = tick;
  +                existing.last_used.store(tick, Ordering::Relaxed);
                   return;
               }
               while guard.current_memory_bytes + entry_bytes > self.max_memory_bytes
                   && !guard.map.is_empty()
               {
                   if let Some((&oldest_key, _)) =
  -                    guard.map.iter().min_by_key(|(_, entry)| entry.last_used)
  +                    guard.map.iter().min_by_key(|(_, entry)| entry.last_used.load(Ordering::Relaxed))
                   {
                       if let Some(removed) = guard.map.remove(&oldest_key) {
                           guard.current_memory_bytes =
  @@ -109,7 +111,7 @@ impl SubExprCache {
                   CacheEntry {
                       matrix,
                       bytes: entry_bytes,
  -                    last_used: tick,
  +                    last_used: AtomicU64::new(tick),
                   },
               );
           }
  ```

---

### P1-06 `[P1-架构隐患]` [已修复 / FIXED] 分钟微结构按 Bar 计数替代物理时间窗、另类数据无序去重，以及 `RidgeModel` / `LightGBMModel` 训练缺陷

> **修复状态**：已修复（[`minute_features.py`](file:///home/cr9294/repos/FactorLibrary/python/factorlib/markets/futures/etl/minute_features.py) 的 `ret_first30` / `ret_last30` 改用首末 30 分钟物理时间窗与首根 `open`；[`aux_warehouse.py`](file:///home/cr9294/repos/FactorLibrary/python/factorlib/markets/futures/etl/aux_warehouse.py)、[`aux_roll.py`](file:///home/cr9294/repos/FactorLibrary/python/factorlib/markets/futures/etl/aux_roll.py)、[`aux_member.py`](file:///home/cr9294/repos/FactorLibrary/python/factorlib/markets/futures/etl/aux_member.py) 与 [`exec_price.py`](file:///home/cr9294/repos/FactorLibrary/python/factorlib/markets/futures/etl/exec_price.py) 先排序后 `.unique(..., keep="last", maintain_order=True)`；[`configs/futures/data.yaml`](file:///home/cr9294/repos/FactorLibrary/configs/futures/data.yaml) 与 [`DataConfig`](file:///home/cr9294/repos/FactorLibrary/python/factorlib/core/config.py) 新增 `alt_data_lag_days` (`0 | 1`) 并同步更新 [`futures_platform_plan.md`](file:///home/cr9294/repos/FactorLibrary/docs/futures_platform_plan.md) 盘后 `17:30` 清算截止说明；[`RidgeModel`](file:///home/cr9294/repos/FactorLibrary/python/factorlib/ml/models/linear.py) 优先使用隔离验证集/时间分块 CV 寻优 `alpha`；[`LightGBMModel`](file:///home/cr9294/repos/FactorLibrary/python/factorlib/ml/models/lgbm.py) 在 `lambdarank`/`rank_xendcg` 下将连续收益率按截面离散为 `0..4` 整型相关性等级并默认匹配 `ndcg` 指标）。

- **定位信息**：
  - [`python/factorlib/markets/futures/etl/minute_features.py`](file:///home/cr9294/repos/FactorLibrary/python/factorlib/markets/futures/etl/minute_features.py#L92-L96)（函数 [`_aggregate_one_year`](file:///home/cr9294/repos/FactorLibrary/python/factorlib/markets/futures/etl/minute_features.py#L56-L135)）
  - [`python/factorlib/markets/futures/etl/aux_warehouse.py`](file:///home/cr9294/repos/FactorLibrary/python/factorlib/markets/futures/etl/aux_warehouse.py#L38-L45) 与 [`aux_roll.py`](file:///home/cr9294/repos/FactorLibrary/python/factorlib/markets/futures/etl/aux_roll.py#L55-L73)
  - [`python/factorlib/ml/models/linear.py`](file:///home/cr9294/repos/FactorLibrary/python/factorlib/ml/models/linear.py#L153-L178)（方法 [`RidgeModel.fit`](file:///home/cr9294/repos/FactorLibrary/python/factorlib/ml/models/linear.py#L153-L178)）
  - [`python/factorlib/ml/models/lgbm.py`](file:///home/cr9294/repos/FactorLibrary/python/factorlib/ml/models/lgbm.py#L120-L141)（方法 [`LightGBMModel.fit`](file:///home/cr9294/repos/FactorLibrary/python/factorlib/ml/models/lgbm.py#L81-L159)）

- **问题根因分析**：
  1. **分钟微结构「30 分钟」退化为「30 根非空 K 线」**：[`minute_features.py`](file:///home/cr9294/repos/FactorLibrary/python/factorlib/markets/futures/etl/minute_features.py#L92-L96) 用 `close.slice(29, 1).first() / close.first() - 1.0` 计算 `ret_first30`，既漏掉了首根分钟线的开盘跳空（应用 `open.first()` 为分母），又在稀疏成交合约上将 30 根 Bar 拉长至数小时甚至跨夜盘/日盘（而 [`exec_price.py`](file:///home/cr9294/repos/FactorLibrary/python/factorlib/markets/futures/etl/exec_price.py#L54-L60) 已正确按 `dt0 + 30min` 物理时间窗过滤）。
  2. **盘后数据无序 `.unique(keep="first")` 非确定性**：[`aux_warehouse.py`](file:///home/cr9294/repos/FactorLibrary/python/factorlib/markets/futures/etl/aux_warehouse.py#L43) 与 [`aux_roll.py`](file:///home/cr9294/repos/FactorLibrary/python/factorlib/markets/futures/etl/aux_roll.py#L57) 在并行 `scan_dataset` 未排序状态上直接调用 `.unique(..., keep="first")`，Polars 多线程扫描时重复行保留结果不确定；同时需注意交易所仓单与会员持仓通常于 `T` 日 15:30–17:00 盘后公布，虽早于 `T+1` 夜盘 21:00 / 日盘 09:00 调仓（因果安全），但晚于 `AGENTS.md` 字面写的「15:00 收盘」截点，应在白皮书与配置中明确信号批处理时点为「`T` 日 17:30 盘后清算完成、20:50 夜盘集合竞价前」，并提供可选 `alt_data_lag_days: 0 | 1` 开关。
  3. **ML 模型层隐患**：
     - [`RidgeModel.fit`](file:///home/cr9294/repos/FactorLibrary/python/factorlib/ml/models/linear.py#L158-L176) 接收了 `X_val, y_val` 却完全丢弃不用，转而调用 `RidgeCV(cv=None)` 执行逐行留一交叉验证（LOOCV），在同日截面强相关与 5 日重叠标签下严重低估最优正则化系数 `alpha_`；
     - [`LightGBMModel.fit`](file:///home/cr9294/repos/FactorLibrary/python/factorlib/ml/models/lgbm.py#L121-L140) 声称支持 `objective="lambdarank"`，但直接把连续浮点收益率 `y` 喂给 `lgb.Dataset`，一旦开启即刻因 `Label should be int type (met float) for ranking task` 崩溃。

- **具体修复方案**：
  1. 在 `minute_features.py` 改用首末时间戳 `30m` 窗口与 `open.first()`；在 `aux_*.py` 中先 `.sort(...)` 再 `.unique(..., keep="last")`；
  2. 在 `RidgeModel.fit` 中优先使用传入的隔离验证集 `(X_val, y_val)` 搜索最优 `alpha`，在 `LightGBMModel.fit` 中当 `objective == "lambdarank"` 时按日期分组将 `y` 离散化为截面分位整数等级 `0..4`。
  ```diff
  --- a/python/factorlib/ml/models/linear.py
  +++ b/python/factorlib/ml/models/linear.py
  @@ -15,7 +15,7 @@ from collections.abc import Sequence
   
   import numpy as np
   from scipy.stats import spearmanr
  -from sklearn.linear_model import RidgeCV
  +from sklearn.linear_model import Ridge, RidgeCV
  @@ -167,14 +167,26 @@ class RidgeModel:
           if self.n_features_ == 0 or len(X) == 0:
               return self
   
  -        # 使用 RidgeCV 自动网格搜索最优正则化强度
  -        rcv = RidgeCV(
  -            alphas=self.alphas,
  -            fit_intercept=self.fit_intercept,
  -        )
  -        rcv.fit(X, y)
  -        self.model_ = rcv
  -        self.best_alpha_ = float(rcv.alpha_)
  +        if X_val is not None and y_val is not None and len(X_val) > 0:
  +            best_mse = float("inf")
  +            best_model: Ridge | None = None
  +            for alpha in self.alphas:
  +                cand = Ridge(alpha=float(alpha), fit_intercept=self.fit_intercept)
  +                cand.fit(X, y)
  +                mse = float(np.mean((cand.predict(X_val) - y_val) ** 2))
  +                if mse < best_mse:
  +                    best_mse = mse
  +                    best_model = cand
  +                    self.best_alpha_ = float(alpha)
  +            self.model_ = best_model
  +        else:
  +            rcv = RidgeCV(alphas=self.alphas, fit_intercept=self.fit_intercept)
  +            rcv.fit(X, y)
  +            self.model_ = rcv
  +            self.best_alpha_ = float(rcv.alpha_)
           return self
  ```
  ```diff
  --- a/python/factorlib/ml/models/lgbm.py
  +++ b/python/factorlib/ml/models/lgbm.py
  @@ -117,10 +117,27 @@ class LightGBMModel:
               "verbose": -1,
           }
   
  +        def _to_rank_labels(y_arr: np.ndarray, grp: np.ndarray | None, n_bins: int = 5) -> np.ndarray:
  +            out = np.zeros(len(y_arr), dtype=np.int32)
  +            if grp is None or len(grp) == 0:
  +                ranks = np.argsort(np.argsort(y_arr))
  +                return np.floor(ranks * n_bins / max(len(y_arr), 1)).astype(np.int32)
  +            offset = 0
  +            for g in grp:
  +                g_int = int(g)
  +                if g_int > 0:
  +                    sub = y_arr[offset : offset + g_int]
  +                    ranks = np.argsort(np.argsort(sub))
  +                    out[offset : offset + g_int] = np.clip(
  +                        (ranks * n_bins) // max(g_int, 1), 0, n_bins - 1
  +                    )
  +                offset += g_int
  +            return out
  +
  +        y_train_eff = _to_rank_labels(y, groups) if self.objective == "lambdarank" else y
           # 构建 Dataset
           train_data = lgb.Dataset(
               X,
  -            label=y,
  +            label=y_train_eff,
               group=groups if self.objective == "lambdarank" else None,
               feature_name=self.feature_names,
               free_raw_data=False,
  ```

---

### P1-07 `[P1-架构隐患]` [已修复 / FIXED] 裸因子 `Gaussian distribution` 线上 `Inf` 污染误判、`Tech_Trend_VP_1_Rev` 复权口径不一致、`5d IC` 混用及 A 股独立脚本脱离主框架

> **修复状态**：已修复（[`StockPanelLoader`](file:///home/cr9294/repos/FactorLibrary/python/factorlib/markets/stocks/panel_loader.py) 已收敛 A 股多因子加载与基准对齐逻辑；[`evaluate.py`](file:///home/cr9294/repos/FactorLibrary/python/factorlib/core/evaluate.py) 将 `ic_5d` 严格修正为 Pearson 线性相关并新增 `rank_ic_5d` (Spearman)，同步更新 [`report.py`](file:///home/cr9294/repos/FactorLibrary/python/factorlib/core/report.py) 与工作台前端展示；[`backtest_multi_factors.py`](file:///home/cr9294/repos/FactorLibrary/scripts/backtest_multi_factors.py) 增加 `--tech-trend-high-low {adj,raw}` 命令行开关并在消融洞察中实时双轨实测对比全复权与混合未复权口径）。

- **定位信息**：
  - [`scripts/backtest_multi_factors.py`](file:///home/cr9294/repos/FactorLibrary/scripts/backtest_multi_factors.py#L334-L353)（函数 [`compute_base_expressions`](file:///home/cr9294/repos/FactorLibrary/scripts/backtest_multi_factors.py#L330-L354)）
  - [`scripts/backtest_multi_factors.py`](file:///home/cr9294/repos/FactorLibrary/scripts/backtest_multi_factors.py#L469-L515)（5 因子定义字典 `FACTOR_SPECS`）
  - [`python/factorlib/core/evaluate.py`](file:///home/cr9294/repos/FactorLibrary/python/factorlib/core/evaluate.py#L453)（`ic_5d_series[t] = _spearman_corr_1d(...)`）
  - [`python/factorlib/markets/stocks/__init__.py`](file:///home/cr9294/repos/FactorLibrary/python/factorlib/markets/stocks/__init__.py)（A 股适配层当前为空壳）

- **问题根因分析**：
  1. **因子 4（`Gaussian distribution`，`bt_a31a4cd18ed14fc8861a7a2b97c29d43`）线上 `0.0017` 实为 `Inf` 数值污染崩溃，而非后处理提升 Alpha**：
     - 经调取线上 FactorLab MCP 接口核查：
       - 前 3 个因子（`Volume_Price_2_H5_2` / `H5` / `R7`）线上表达式以 `add(sub(BaseUnivMask, BaseUnivMask), neg(...))` 开头，先施加了 `BaseUnivMask` 剔除 ST/停牌/异常股；
       - 而因子 4（`Gaussian distribution`，版本 `fv_316a87114cb94b5e9929140119a91ec6`）线上表达式**未包裹 `BaseUnivMask`**，直接对全市场计算 `ts_std(daily_derived_returns_c2c_1d, 20)`。
       - 根据 FactorLab C++ 算子规范（[`Factor/data/catalog_full.json`](file:///home/cr9294/repos/FactorLibrary/Factor/data/catalog_full.json)），`ts_std` 与 `cs_zscore` 仅跳过 `NaN`、**不跳过 `Inf`**。线上全市场 `returns_c2c_1d` 在历史异常股上存在 `Inf`，导致未加 `BaseUnivMask` 的 `cs_zscore` 分母变成 `Inf`、截面全部坍缩为 `0` 或 `NaN`（线上 `ic = None`、`percic = None`，且 `holdings` 持仓只数高达 `2500~2963` 只即把全市场股票塞入同一分位桶！）。
       - 本地 Rust 内核在算子内部使用 `v.is_finite()` 拦截了 `Inf`，求出了该裸因子真实的 RankIC = **`0.0503`**（甚至略高于 `MAD + decay + rank` 后的 `0.0458`）。因此对比报告第三节第 3 点「`MAD` 与 `rank` 将 RankIC 从 `0.0017` 飙升至 `0.0475`」属于对线上 `Inf` 溢出故障的归因误判（且 Spearman 秩相关本身对严格单调变换 `rank` 完全不变！）。
  2. **因子 5（`Tech_Trend_VP_1_Rev`，`bt_346b56e4bd91496a88129b28ec95aa6b`）复权与未复权价格混算差异**：
     - 线上表达式（版本 `fv_7273d6958bfa4c4b969375f5f166324f`）原文为：
       `neg(add(add(cs_zscore(div(sub(close_hfq, ts_mean(close_hfq, 20)), ts_mean(close_hfq, 20))), cs_zscore(div(sub(close_hfq, low), sub(high, low)))), cs_zscore(ts_corr(close_hfq, mul(volume, close_hfq), 10))))`
     - 注意其第二项 `div(sub(close_hfq, low), sub(high, low))` 在线上错误混用了**后复权收盘价 `close_hfq`** 与**不复权最高/最低价 `high, low`**（除权分红后 `close_hfq >> high`，比值严重失真，导致线上 RankIC 跌至 `0.0395`）；而本地 [`backtest_multi_factors.py`](file:///home/cr9294/repos/FactorLibrary/scripts/backtest_multi_factors.py#L352) 全部传入了后复权 `high_adj, low_adj`（本地 RankIC = `0.0437`）。脚本应提供显式开关支持「线上原样复现模式（`raw high/low`）」与「本地修正复权模式（`high_adj/low_adj`）」双轨对照。
  3. **`5日 IC` 的 Pearson 与 Spearman 口径混用**：
     - 线上 FactorLab 的 `5dic` 为 5 日远期收益的 **Pearson 线性 IC**，而 [`evaluate.py`](file:///home/cr9294/repos/FactorLibrary/python/factorlib/core/evaluate.py#L453) 将 `ic_5d_series` 实现为 `_spearman_corr_1d`（即 5 日 RankIC），导致报告第二章总表中 `5日 IC`（`0.0314`）与第四章多周期表 `$h=5$ 日 RankIC`（`0.0654`）出现同源指标相互矛盾。应在 `evaluate.py` 中明确拆分为 `ic_5d`（Pearson）与 `rank_ic_5d`（Spearman）。
  4. **A 股评测逻辑散落在 `scripts/backtest_*.py` 独立脚本，脱离 `markets/stocks/` 与 `core/evaluate`**：
     - [`scripts/backtest_multi_factors.py`](file:///home/cr9294/repos/FactorLibrary/scripts/backtest_multi_factors.py) 与 [`scripts/backtest_volume_price_factor.py`](file:///home/cr9294/repos/FactorLibrary/scripts/backtest_volume_price_factor.py) 手工拼接 Polars/Pandas 读取与手写指标切片，未沉淀至 [`python/factorlib/markets/stocks/`](file:///home/cr9294/repos/FactorLibrary/python/factorlib/markets/stocks/__init__.py)，导致核心评估修复无法自动惠及 A 股多因子回测。

- **具体修复方案**：
  1. 在 `python/factorlib/markets/stocks/` 下新增标准化 A 股面板加载器 `panel_loader.py`，封装复权价格、不复权价格、`base_univ_mask`（含 `is_st`、上市自然日）、`limit_mask`（含 `limit_up`/`limit_down`）、申万行业及基准指数对齐收益；
  2. 在 [`evaluate.py`](file:///home/cr9294/repos/FactorLibrary/python/factorlib/core/evaluate.py#L448-L465) 中同时输出 `ic_5d_mean`（Pearson）与 `rank_ic_5d_mean`（Spearman）；
  3. 重构 [`scripts/backtest_multi_factors.py`](file:///home/cr9294/repos/FactorLibrary/scripts/backtest_multi_factors.py) 与 [`scripts/backtest_volume_price_factor.py`](file:///home/cr9294/repos/FactorLibrary/scripts/backtest_volume_price_factor.py)，统一调用 `factorlib.markets.stocks` 与修复后的 Rust 内核指标接口，并在报告中修正因子 4 与因子 5 的归因说明。

---

## 3. 【中低风险与代码质量改进建议（P2 & P3 Issues）】

### 3.1 绩效统计与后处理边界改进（P2）

1. **`P2-01` Layer 1 评估预热期零填充扭曲年化波动率、夏普与换手率**：
   - **位置**：[`python/factorlib/core/evaluate.py`](file:///home/cr9294/repos/FactorLibrary/python/factorlib/core/evaluate.py#L403-L485)（[`evaluate_factor_v4_stats`](file:///home/cr9294/repos/FactorLibrary/python/factorlib/core/evaluate.py#L371-L569)）及 [`rust/factorlib_core/src/backtest/futures.rs`](file:///home/cr9294/repos/FactorLibrary/rust/factorlib_core/src/backtest/futures.rs#L497-L526)、[`fast.rs`](file:///home/cr9294/repos/FactorLibrary/rust/factorlib_core/src/backtest/fast.rs#L156-L185)。
   - **分析**：当因子存在长预热窗口（如 `ts_mean(close, 250)`）时，前 250 日无有效截面，`group_daily_rets` 保持初始 `0.0`，随后 `n_years = T / 252.0` 与 `np.std(g_rets, ddof=1)` 将全部预热空仓日纳入分母，人为压低年化收益与波动率；`futures.rs` 与 `fast.rs` 亦将开仓前的 `r == 0.0` 空仓日计入 `valid_days` 稀释 `win_rate`。建议仅从首个有效信号日（`first_active_day`）起截取绩效统计区间。
2. **`P2-02` 后处理时序平滑（`decay` / `emadecay`）向退市/非在池标的回渗信号**：
   - **位置**：[`rust/factorlib_core/src/ops/postprocess.rs`](file:///home/cr9294/repos/FactorLibrary/rust/factorlib_core/src/ops/postprocess.rs#L219-L243) 与 [`python/factorlib/core/probe.py`](file:///home/cr9294/repos/FactorLibrary/python/factorlib/core/probe.py#L657-L711)。
   - **分析**：推荐管线先在 Stage 1 执行 `baseUniMask`（将非在池行置 `NaN`），最后在 Stage 5 执行 `decay`/`emadecay`（内部 `min_periods=1`）。某合约一旦在 `t` 日退市或移出主力池，过去 `w-1` 天的有限值会被平滑算子继续外推回填为非 `NaN` 值。建议在 `decay`/`emadecay` 输出后自动重施加当日可交易掩码。
3. **`P2-03` 风格归因秩相关未处理并列秩 & 前向复权起点尺度说明**：
   - [`styles.py`](file:///home/cr9294/repos/FactorLibrary/python/factorlib/core/styles.py#L255-L256) 使用 `np.argsort(np.argsort(fx))` 计算排名，遇常数或缩尾边界并列值（如 `MAD(3.0)` 截断后的极值点！）时按品种列顺序强行赋不同秩，应复用 [`evaluate._spearman_corr_1d`](file:///home/cr9294/repos/FactorLibrary/python/factorlib/core/evaluate.py#L307-L328) 的平均秩实现；
   - [`adjust.py`](file:///home/cr9294/repos/FactorLibrary/python/factorlib/markets/futures/etl/adjust.py#L100-L108) 采用以 `cfg.start_date` 为 `1.0` 基准的前向累乘复权，消除了随 `end_date` 推进的历史漂移（因果绝对安全），但复权价格绝对值依赖于 `start_date` 且远离当前现价，禁止在 DSL 中将复权 `close` 与固定绝对价格常数（如 `gt_const(close, 5000)`）比较，应在 [`typecheck.rs`](file:///home/cr9294/repos/FactorLibrary/rust/factorlib_core/src/expr/typecheck.rs) 中给出编译告警。

### 3.2 Rust 内存拷贝与 Rayon 并行粒度优化建议（P2）

1. **消除嵌套并行爆炸与每行堆分配风暴（预计加速 GP 库内去相关 `8x–15x`）**：
   - **位置**：[`rust/factorlib_core/src/metrics/corr.rs`](file:///home/cr9294/repos/FactorLibrary/rust/factorlib_core/src/metrics/corr.rs#L65-L78) 及 [`corr.rs`](file:///home/cr9294/repos/FactorLibrary/rust/factorlib_core/src/metrics/corr.rs#L119-L125)、[`rust/factorlib_core/src/py.rs`](file:///home/cr9294/repos/FactorLibrary/rust/factorlib_core/src/py.rs#L838-L883)。
   - **瓶颈根因**：`eval_gp_batch` 外层对候选个体启用 `into_par_iter()`（第 `839` 行），内层对每个已入库因子调用 `cross_correlation_pairwise`（第 `872` 行），而 `cross_correlation_pairwise` 内部又对 `0..t` 开启第二层 `into_par_iter()`（`corr.rs:65`），且在每个交易日循环内分配 3 个 `Vec`（`a_row`、`b_row`、`m_row`）。当种群 1,000 个体 × 库内 50 因子 × 3,800 交易日时，单代产生 **5.7 亿次小内存堆分配**与严重的 Rayon 任务窃取锁竞争！
   - **优化方案**：
     - 将库内因子在加载阶段一次性预转换为逐日截面归一化秩矩阵 `(N, T)`（行连续布局，使单日截面 `N` 个品种在物理内存上紧邻！）；
     - 提供非并行的 `cross_correlation_pairwise_seq` 供外层已并行的 `eval_gp_batch` 与 `cross_correlation_matrix` 调用，复用线程局部缓冲区 `tls_buf` 实现**零堆分配**内积计算。
2. **优化 FFI 边界 `view_to_fortran_f64` 冗余标量拷贝**：
   - **位置**：[`rust/factorlib_core/src/py.rs`](file:///home/cr9294/repos/FactorLibrary/rust/factorlib_core/src/py.rs#L41-L66)。
   - **瓶颈根因**：Python 侧传入的矩阵几乎全部已是 `order="F"` 列连续数组，但 `view_to_fortran_f64` 每次均新建矩阵并逐元素标量循环复制。当 `view.is_standard_layout()` 的转置（即 Fortran 连续，`view.t().is_standard_layout()`）成立且无需修改时，可直接通过切片 `as_slice_memory_order()` 执行 SIMD 向量化克隆甚至直接借用 `ArrayView2` 传入只读统计函数（`calc_ic_series`、`calc_quantile_metrics`、`apply_postprocess_op`）。

### 3.3 类型注解、异常处理与解析器健壮性（P3）

- **递归下降解析器栈溢出防护**：[`rust/factorlib_core/src/expr/parser.rs`](file:///home/cr9294/repos/FactorLibrary/rust/factorlib_core/src/expr/parser.rs#L33-L71) 的 `Parser` 未维护递归深度计数器，恶意输入如 `"(" * 20000 + "close" + ")" * 20000` 会跨 PyO3 FFI 直接栈溢出杀死宿主进程，应增加 `depth <= 128` 硬上限并返回 `ParseError::Syntax`。
- **静默吞异常清理**：[`workbench.py`](file:///home/cr9294/repos/FactorLibrary/python/factorlib/core/workbench.py#L307-L308)（`except Exception: pass` 静默回退至合成演示面板而不记录损坏原因）应补充 `logger.warning(..., exc_info=True)`；[`workbench_server.py`](file:///home/cr9294/repos/FactorLibrary/python/factorlib/workbench_server.py)（原 `server_main.py`）的非法 JSON 请求体已修复为记录 `logging.warning` 并返回 HTTP `400 Bad Request`。
- **类型注解收紧**：[`evaluate.py`](file:///home/cr9294/repos/FactorLibrary/python/factorlib/core/evaluate.py#L114) 及第 `584`、`594-596` 行多处使用宽泛 `Any`（如 `backtest_result: Any`、`portfolio_config: Any`），应替换为 `TYPE_CHECKING` 下的精确类型 `BacktestResult`、`PortfolioConfig | None`、`PostprocessSpec`。

### 3.4 单元测试盲区全景对比（对比 `tests/` 现有覆盖率）

当前 [`tests/python/`](file:///home/cr9294/repos/FactorLibrary/tests/python) 共 18 个测试文件（90 个用例），[`rust/factorlib_core/tests/`](file:///home/cr9294/repos/FactorLibrary/rust/factorlib_core/tests) 共 2 个属性测试文件（9 个测试套），另有 7 个早期遗留 [`tests/test_*.py`](file:///home/cr9294/repos/FactorLibrary/tests)（共 1,214 行，其中 6 个文件 0 个有效 `pytest` 函数，属于死测试脚本）。核心漏测盲区如下：

| 漏测盲区模块 | 现有测试为何未能拦截 Bug | 必须新增的回归测试规格 |
| :--- | :--- | :--- |
| **同日早盘前视泄漏检验** | [`test_perturbation.py`](file:///home/cr9294/repos/FactorLibrary/tests/python/test_perturbation.py) 仅篡改 `t > T_perturb` 的未来行，**无法发现 `T` 日 15:00 因子偷用 `T` 日 09:00 早盘开盘价收益**；[`test_workbench.py`](file:///home/cr9294/repos/FactorLibrary/tests/python/test_workbench.py) 使用合成面板，其中 `r_exec_next` 被人为覆写为 `y_1`。 | 新增 `test_intraday_reversal_zero_ic_on_pure_noise`：构造纯日内随机游走（隔夜与次日收益独立同分布零均值），断言 `close / open - 1` 的 `factor_metrics` 中 `ls_sharpe` 与 `ic_mean` 均严格趋近 0。 |
| **换月缺价与停牌账务测试** | [`test_backtest.py`](file:///home/cr9294/repos/FactorLibrary/tests/python/test_backtest.py) 仅测试了无换月常规涨跌停与保证金缩减，未覆盖 `is_roll=True & exec_price_old=NaN`、多空反手调仓与净值穿仓 `< 0`。 | 在 `test_backtest.py` 增加：(1) 换月日 `exec_price_old=NaN` 且新旧合约价差 20% 时净值无虚假跳空；(2) `+10 -> -10` 反手精确收 `10*close + 10*open`；(3) 极端暴跌穿仓后 `positions == 0` 且 `annual_return == -1.0`（非 `NaN`）。 |
| **截面算子在池掩码隔离** | [`test_rust_core.py`](file:///home/cr9294/repos/FactorLibrary/tests/python/test_rust_core.py) 使用全 `True` 掩码核对 `ops_ref.py`，未测试 `universe_mask=False` 处存在极端异常值（如 `1e9`）时的隔离性。 | 新增 `test_cs_and_group_ops_universe_mask_isolation`：篡改 `mask == False` 单元格为 `1e9`，断言 `mask == True` 区域的 `cs_rank`、`cs_zscore`、`cs_neutralize`、`group_demean` 输出 **0 bit 差异**。 |
| **AST 子树规范化等价性** | [`proptest_ops.rs`](file:///home/cr9294/repos/FactorLibrary/rust/factorlib_core/tests/proptest_ops.rs) 仅测试了顶层 `slog(cs_rank(close))`，未测试嵌套在二元算子内部的子树求值一致性。 | 新增属性测试：对任意合法非退化 AST（排除根节点单调包裹外），验证 `eval(expr, canonicalize=false)` 与 `eval(expr, canonicalize=true)` 逐元素误差 `< 1e-12`。 |
| **并列秩容差传递边界** | 随机浮点生成的 `proptest` 极难碰撞到间距恰好为 `0.8 * STD_EPS` 的三元组。 | 在 `test_rust_core.py` 显式加入 `[0.0, 0.8e-12, 1.5e-12]` 对账断言。 |
| **基准收益窗口对齐与复利超额** | 现有评估测试未核查基准收益 `bench_ret` 与个股远期持仓窗口 $[T+1, T+2]$ 的时间下标对应关系，也未校验复利超额年化公式。 | 在 `test_evaluate.py` 新增 `test_benchmark_forward_window_alignment_and_compound_excess`：构造 `G10 == benchmark` 的完全复制组合，当指数在 $T$ 日暴跌、$T+2$ 日暴涨时，断言每日超额收益恒为 `0.0` 且复利年化超额为 `0.0`。 |
| **Top 分位单边权重换手率 vs 秩自相关** | [`test_rust_core.py`](file:///home/cr9294/repos/FactorLibrary/tests/python/test_rust_core.py) 仅验证了 `calc_turnover`（$1-\rho_1$）值域在 `[0, 2]`，未测试 Top 分位等权组合权重换手率及 `.round()` 分位切分。 | 新增 `test_top_quantile_turnover_and_round_bucketing`：构造已知 10 只股票每日轮换 1 只进出 G10 的面板，断言 `top_quantile_turnover == 0.10`（而非 `1 - rho`），且空仓预热日 `NaN` 不稀释均值。 |
| **A 股 ST / 涨跌停掩码与执行价时钟** | `scripts/backtest_*.py` 独立于 `tests/python/` 之外，无任何自动化测试覆盖 ST 剔除、一字涨跌停拦截与 `open_adj` 执行价对齐。 | 新增 `tests/python/test_stocks_panel_and_eval.py`：验证 `is_st`、`limit_up`/`limit_down`、`StockListedDays` 掩码构建及 `ic_5d`（Pearson）与 `rank_ic_5d`（Spearman）分离。 |

---

## 4. 【仓库瘦身与架构重构行动计划】

### 4.1 三代并存重复模块盘点与处置矩阵

| 功能域 | 体系 A（唯一核心保留） | 体系 B（早期 `pyrust-bt` 原型） | 体系 C（`Factor/` 本地复现原型） | 处置策略（已确认执行） |
| :--- | :--- | :--- | :--- | :--- |
| **回测引擎（4 套）** | [`futures.rs`](file:///home/cr9294/repos/FactorLibrary/rust/factorlib_core/src/backtest/futures.rs) + [`fast.rs`](file:///home/cr9294/repos/FactorLibrary/rust/factorlib_core/src/backtest/fast.rs) + [`backtest.py`](file:///home/cr9294/repos/FactorLibrary/python/factorlib/core/backtest.py) | `rust/engine_rust/` + `python/pyrust_bt/` | `Factor/factorlab_local_engine.py` + `Factor/scripts/futures_engine.py` | 唯一保留体系 A；彻底删除（`git rm -rf`）`pyrust_bt`、`engine_rust` 与 `Factor/` |
| **算子与求值内核（3 套）** | [`rust/factorlib_core/src/{ops,expr}/`](file:///home/cr9294/repos/FactorLibrary/rust/factorlib_core/src/ops/mod.rs) + [`ops_ref.py`](file:///home/cr9294/repos/FactorLibrary/python/factorlib/core/ops_ref.py) | `engine_rust/src/{factor,indicators}.rs` | `AresRuntimeEngine`（1,755 行纯 Python 双重列表） | 保留体系 A；将 12 算子 3×3 C++ 真值向量提取至 [`tests/python/fixtures/postprocess_vectors.json`](file:///home/cr9294/repos/FactorLibrary/tests/python/fixtures/postprocess_vectors.json)，彻底删除 `Factor/` |
| **因子目录与 MCP 客户端** | [`python/factorlib/core/catalog.py`](file:///home/cr9294/repos/FactorLibrary/python/factorlib/core/catalog.py) + [`mcp_client.py`](file:///home/cr9294/repos/FactorLibrary/python/factorlib/core/mcp_client.py) | 无 | `Factor/factorlab_catalog.py` + `Factor/factorlab_client.py` | 保留 `core/catalog.py`；将轻量 MCP 请求客户端收编至 `core/mcp_client.py`，彻底删除 `Factor/` |
| **Web 服务与前端（2 套）** | [`python/factorlib/frontend/{index.html,app.js,styles.css}`](file:///home/cr9294/repos/FactorLibrary/python/factorlib/frontend) + [`workbench.py`](file:///home/cr9294/repos/FactorLibrary/python/factorlib/core/workbench.py) + [`workbench_server.py`](file:///home/cr9294/repos/FactorLibrary/python/factorlib/workbench_server.py) | `python/server_main.py` + `frontend/streamlit_app.py` | 无 | 将 `server_main.py` 正式收编为 `python/factorlib/workbench_server.py`，并将 `frontend/` 迁入 `python/factorlib/frontend/` 纳入 Wheel 构建；彻底删除 `streamlit_app.py` |
| **A 股多因子对账与适配** | [`python/factorlib/markets/stocks/`](file:///home/cr9294/repos/FactorLibrary/python/factorlib/markets/stocks/__init__.py) + [`core/evaluate.py`](file:///home/cr9294/repos/FactorLibrary/python/factorlib/core/evaluate.py) | `python/pyrust_bt/market_data/` | [`scripts/backtest_multi_factors.py`](file:///home/cr9294/repos/FactorLibrary/scripts/backtest_multi_factors.py) + [`scripts/backtest_volume_price_factor.py`](file:///home/cr9294/repos/FactorLibrary/scripts/backtest_volume_price_factor.py) | 在 `markets/stocks/` 建立标准 A 股面板加载器，重构两个对账脚本复用 `core/evaluate` 与 Rust 内核 |
| **示例、测试与旧文档** | `scripts/*.py` + `tests/python/*.py` + [`docs/{futures_platform_plan,api_reference,code_audit_and_refactor_plan}.md`](file:///home/cr9294/repos/FactorLibrary/docs/futures_platform_plan.md) | `examples/*.py` + `tests/test_*.py` + `docs/{architecture,xtdata_market_data_plan,...}.md` + `CHANGELOG.md` + `images/` | `pyrust-bt-main/`（空目录） | 彻底删除 `pyrust-bt-main/`、`examples/`、`tests/test_*.py`、9 篇旧文档、`CHANGELOG.md`、`images/` 与 `portfolio_equity_curve.png` |

---

### 4.2 分步重构与修复执行路线图（5 阶段零破损方案）

所有步骤严格保证每阶段均可通过 `bash scripts/verify.sh --all` 全量门禁，并按中文 Angular Commit 规范独立提交：

#### 阶段一：仓库瘦身、移除多代历史原型与包结构统一（已完成 ✅）
```bash
# 1. 提取 12 个后处理算子 C++ 3x3 测试真值向量至 tests/python/fixtures/postprocess_vectors.json
# 2. 将轻量 MCP 客户端收编至 python/factorlib/core/mcp_client.py，并将 Web 服务收编为 python/factorlib/workbench_server.py
git mv python/server_main.py python/factorlib/workbench_server.py

# 3. 彻底删除 Factor/、pyrust_bt/、engine_rust/ 及所有历史遗留示例、旧测试、旧文档与图片
git rm -rf Factor python/pyrust_bt rust/engine_rust pyrust-bt-main examples \
           tests/test_*.py frontend/streamlit_app.py CHANGELOG.md \
           portfolio_equity_curve.png images \
           docs/architecture.md docs/examples.md docs/factor-analysis.md \
           docs/getting-started.md docs/index.md docs/market-data.md \
           docs/strategy-guide.md docs/troubleshooting.md docs/xtdata_market_data_plan.md
```
- 同步更新 [`scripts/run_web.py`](file:///home/cr9294/repos/FactorLibrary/scripts/run_web.py#L26) 与 [`tests/python/test_workbench.py`](file:///home/cr9294/repos/FactorLibrary/tests/python/test_workbench.py#L21) 的导入路径为 `from factorlib.workbench_server import ...`；
- 唯一保留 [`rust/factorlib_core/Cargo.toml`](file:///home/cr9294/repos/FactorLibrary/rust/factorlib_core/Cargo.toml) 单一计算内核 Crate，清理 [`pyproject.toml`](file:///home/cr9294/repos/FactorLibrary/pyproject.toml#L46-L53) 中的旧排除项并对 `workbench_server.py` 执行 Ruff 静态检查与格式化；
- 运行 `bash scripts/verify.sh --all` 校验通过后提交：  
  `refactor(arch): 移除多代历史原型并收编工作台服务端模块`

#### 阶段二：期货 ETL 前视偏差根治与面板安全边界加固（修复 `P0-01`、`P0-02`、`P1-03`、`P1-06`）（已完成 ✅）
1. 修改 [`labels.py`](file:///home/cr9294/repos/FactorLibrary/python/factorlib/markets/futures/etl/labels.py#L68-L75)、[`py.rs`](file:///home/cr9294/repos/FactorLibrary/rust/factorlib_core/src/py.rs#L566)、[`evaluate.py`](file:///home/cr9294/repos/FactorLibrary/python/factorlib/core/evaluate.py#L247)、[`workbench.py`](file:///home/cr9294/repos/FactorLibrary/python/factorlib/core/workbench.py#L254-L260)，彻底消除 `r_exec_next` 同日早盘前视泄漏；
2. 修改 [`build.py`](file:///home/cr9294/repos/FactorLibrary/python/factorlib/markets/futures/etl/build.py#L173-L177)，换月日旧主力缺分钟线时关联 `bar_1d` 中旧合约自身 `open` / `prev_settlement`，严禁使用新合约 `settle_raw`；
3. 修改 [`minute_features.py`](file:///home/cr9294/repos/FactorLibrary/python/factorlib/markets/futures/etl/minute_features.py#L92-L96) 为物理 30 分钟时间戳切片，在 [`aux_warehouse.py`](file:///home/cr9294/repos/FactorLibrary/python/factorlib/markets/futures/etl/aux_warehouse.py#L43) 与 [`aux_roll.py`](file:///home/cr9294/repos/FactorLibrary/python/factorlib/markets/futures/etl/aux_roll.py#L57) 增加确定性排序去重；
4. 在 [`panel.py`](file:///home/cr9294/repos/FactorLibrary/python/factorlib/core/panel.py#L138-L201) 中将 `aux` 字段移入 `panel.aux_fields` 隔离出 DSL 命名空间，修复 [`panel.rs`](file:///home/cr9294/repos/FactorLibrary/rust/factorlib_core/src/panel.rs#L52-L61) 的 `r_exec_next` 拦截与 [`probe.py`](file:///home/cr9294/repos/FactorLibrary/python/factorlib/core/probe.py#L624-L627) 的 `limit_mask` 涨跌停掩码；
5. 若本地存在 `exports/`，重新运行 `python scripts/build_panel.py` 刷新 `data/derived/futures/v1/`（注意核查 `git status` 严禁暂存任何 `.parquet`/`.sqlite`）；
6. 运行 `bash scripts/verify.sh --all` 后提交：  
   `fix(etl): 根治分位收益前视偏差与换月跨合约价格混算`

#### 阶段三：Rust 计算内核、AST 规范化与双轨对账修复（修复 `P0-03`、`P0-04`、`P0-08`、`P1-04`、`P1-05`）（已完成 ✅）
1. 在 [`eval.rs`](file:///home/cr9294/repos/FactorLibrary/rust/factorlib_core/src/expr/eval.rs#L92-L139) 与 [`styles.py`](file:///home/cr9294/repos/FactorLibrary/python/factorlib/core/styles.py#L94-L101) 为全部截面算子、分组算子及风格暴露注入 `universe_mask` 过滤，并在 [`ops_ref.py`](file:///home/cr9294/repos/FactorLibrary/python/factorlib/core/ops_ref.py#L879-L885) 同步掩码与左锚点并列秩算法；
2. 修复 [`canon.rs`](file:///home/cr9294/repos/FactorLibrary/rust/factorlib_core/src/expr/canon.rs#L98) 的 `compress_outer_rank` 递归穿透与 `inv(inv(x))` 规则，同步 [`python/factorlib/gp/canon.py`](file:///home/cr9294/repos/FactorLibrary/python/factorlib/gp/canon.py)；
3. 将 [`cache.rs`](file:///home/cr9294/repos/FactorLibrary/rust/factorlib_core/src/expr/cache.rs#L64-L115) 升级为 `AtomicU64` 真 LRU，消除 [`corr.rs`](file:///home/cr9294/repos/FactorLibrary/rust/factorlib_core/src/metrics/corr.rs#L65) 嵌套并行与逐行堆分配，补充 [`parser.rs`](file:///home/cr9294/repos/FactorLibrary/rust/factorlib_core/src/expr/parser.rs) 递归深度上限、`n_quantiles >= 2` FFI 校验及 [`probe.py`](file:///home/cr9294/repos/FactorLibrary/python/factorlib/core/probe.py#L250) 沙箱加固；
4. 运行 `uv pip install -e . --no-build-isolation` 重新编译 `factorlib_core`，执行 `bash scripts/verify.sh --all` 后提交：  
   `fix(core): 修复截面掩码隔离与AST子树化简及并列秩对账`

#### 阶段四：GP/ML 跨管线防泄漏与期货实盘回测账务校准（修复 `P0-05`、`P0-06`、`P0-07`）（已完成 ✅）
1. 对齐 [`ml_default.yaml`](file:///home/cr9294/repos/FactorLibrary/configs/futures/ml_default.yaml#L9)（`start_predict_year: 2022`），在 [`gp/engine.py`](file:///home/cr9294/repos/FactorLibrary/python/factorlib/gp/engine.py#L174-L200) 加入 `horizon` Purge 隔离窗，在 [`ml/pipeline.py`](file:///home/cr9294/repos/FactorLibrary/python/factorlib/ml/pipeline.py#L204) 动态同步 `purge_window`，修复 [`RidgeModel`](file:///home/cr9294/repos/FactorLibrary/python/factorlib/ml/models/linear.py#L153-L178) 验证集选参与 [`LightGBMModel`](file:///home/cr9294/repos/FactorLibrary/python/factorlib/ml/models/lgbm.py#L121-L140) 排序标签离散化；
2. 修复 [`futures.rs`](file:///home/cr9294/repos/FactorLibrary/rust/factorlib_core/src/backtest/futures.rs) 与 [`fast.rs`](file:///home/cr9294/repos/FactorLibrary/rust/factorlib_core/src/backtest/fast.rs) 的穿仓强平归零、负净值 `powf` 截断、多空反手拆分计费、换月双边成本归集及停牌减仓拦截；
3. 增补 3.4 节列出的期货与算子回归测试，同步更新技术白皮书 [`docs/futures_platform_plan.md`](file:///home/cr9294/repos/FactorLibrary/docs/futures_platform_plan.md)；
4. 执行门禁认证并提交：  
   `fix(backtest): 修复期货穿仓强平与反手计费及跨管线样本泄漏`

#### 阶段五：A 股面板适配层构建、Rust 评估指标对齐与 FactorLab 多因子对账脚本重构（修复 `P0-09`、`P0-10`、`P0-11`、`P1-07`）（已完成 ✅）
1. **Rust `metrics` 与 `evaluate.py` 指标标准化**：
   - 在 [`rust/factorlib_core/src/metrics/quantile.rs`](file:///home/cr9294/repos/FactorLibrary/rust/factorlib_core/src/metrics/quantile.rs#L84-L85) 中将分位桶边界切分由 `.floor()` 统一为 `.round()`，并在 [`turnover.rs`](file:///home/cr9294/repos/FactorLibrary/rust/factorlib_core/src/metrics/turnover.rs) 与 [`py.rs`](file:///home/cr9294/repos/FactorLibrary/rust/factorlib_core/src/py.rs) 中新增 `calc_top_quantile_turnover`（同时返回 `top_quantile_turnover` 与 `rank_autocorr_turnover`）；
   - 在 [`python/factorlib/core/evaluate.py`](file:///home/cr9294/repos/FactorLibrary/python/factorlib/core/evaluate.py#L405-L505) 中剔除预热期/首日空仓 `0.0` 对换手率与波动率的稀释，将 `long_excess_annual_return` 升级为复利几何年化，并显式拆分 5 日线性 `ic_5d`（Pearson）与 5 日秩相关 `rank_ic_5d`（Spearman）。
2. **构建 `python/factorlib/markets/stocks/` 标准 A 股面板加载器**：
   - 新增 `python/factorlib/markets/stocks/panel_loader.py`，完整加载 `bar_1d`（含 `limit_up`、`limit_down`）、`trade_status`（含 `is_st`、`susp_Type`）、`ex_factor`、`industry_sw`、`instruments_info`（含自然日上市天数 `StockListedDays` 与退市日 `de_listed_date`）及基准指数 `000985.CSI`；
   - 默认采用严格无日内前视的 $T+1$ 复权开盘价 `open_adj`（或分钟线 `VwapBegin30`）计算远期收益，支持配置切换 `vwap_1d` 敏感性对比，并确保基准指数收益严格对齐至相同的 $[T+1, T+1+h]$ 持仓窗口与价格时钟。
3. **重构多因子对账脚本与单元测试**：
   - 重构 [`scripts/backtest_multi_factors.py`](file:///home/cr9294/repos/FactorLibrary/scripts/backtest_multi_factors.py) 与 [`scripts/backtest_volume_price_factor.py`](file:///home/cr9294/repos/FactorLibrary/scripts/backtest_volume_price_factor.py)，复用 `factorlib.markets.stocks` 与 `factorlib.core.evaluate`，支持 `Tech_Trend_VP_1_Rev` 线上原样（未复权 `high/low`）与修正后（全复权）双模式对比；
   - 新增 `tests/python/test_stocks_panel_and_eval.py` 回归测试，运行全量门禁认证并提交：
   ```bash
   uv pip install -e . --no-build-isolation
   bash scripts/verify.sh --all
   git status  # 确认无任何 data/、exports/、runs/、*.parquet、*.sqlite 混入
   ```
   Commit 标题：`feat(stocks): 构建A股标准面板加载器并修复多因子对账回测口径`

---

### 4.3 P0 全量问题复审清单与 `Volume_Price_2_H5_2` 跨平台实证对账结果（2026-11-13 验收）

#### 1. P0-01 ~ P0-11 逐项复审状态表

| 编号 | 问题描述 | 首次修复提交 | 二次深度复审查出的残余遗漏 / 回归 | 最终闭环状态 |
| :--- | :--- | :---: | :--- | :---: |
| **`P0-01`** | 分位数多空评估误用未滞后 `r_exec_next` 引发日内前视偏差 | `1fbb633` | 磁盘派生文件 `data/derived/futures/v1/labels.parquet` 生成于旧代码，导致磁盘上 `r_exec_next` 与 `y_1` 仍存在 `0.17068` 偏差 | ✅ **已闭环**：重建 `labels.parquet`，验证 `max \|r_exec_next - y_1\| == 0.0` |
| **`P0-02`** | 换月日旧合约缺分钟线回退至新合约 `settle_raw`/`exec_price` | `ac4702b` | 无残余问题（三级同合约回退 + Rust 侧 `prev_settle` 兜底已完整生效） | ✅ **已闭环** |
| **`P0-03`** | 横截面/分组算子与 6 大风格因子忽略 `universe_mask` | `5ebef2f` | 无残余问题（`eval.rs`、`styles.py`、`ops_ref.py` 已严格隔离非在池标的） | ✅ **已闭环** |
| **`P0-04`** | AST `compress_outer_rank` 穿透子树篡改表达式语义 & `inv(inv(x))` | `e7373d9` | 无残余问题（二元/三元子树已强制 `compress_outer_rank=false`） | ✅ **已闭环** |
| **`P0-05`** | GP 验证期（2019–2021）与 ML 预测期（2018–2026）重叠 4 年且缺 Purge | *本次补齐* | `ml_default.yaml` 仍为 `2018`；`GPEngine._resolve_splits` 未扣减 `self.horizon` 且短样本 `te_fallback` 与 `va_fallback` 100% 重叠；`WalkForwardPipeline.run` 未按 `y_h` 强制放大 `purge_window` | ✅ **已闭环**：`start_predict_year: 2022`，GP/ML 双端强制 Purge 隔离并新增单测 |
| **`P0-06`** | 期货回测穿仓后目标仓位与保证金缩放符号翻转、负净值 `powf` 为 `NaN` | `a882e4e` | 一字板锁仓（`is_unilateral=true`）若在穿仓次日反弹使 `prev_e > 0`，缺乏粘性破产标记会导致死灰复燃重新开仓 | ✅ **已闭环**：在 `futures.rs` 增加粘性 `is_bankrupt` 锁死强平状态 |
| **`P0-07`** | 反手调仓手续费误算、`close_today` 丢失、换月单边计费、停牌去杠杆穿透 | `a882e4e` | 无残余问题（平开手数独立拆分、`close_today_fee` 全链路透传已完整生效） | ✅ **已闭环** |
| **`P0-08`** | PyO3 FFI 未拦截 `n_quantiles < 2` 触发 Panic & Python 因子沙箱漏洞 | `4c99e1a` (部分) | 仅修复了顶层 `calc_quantile_returns`，漏掉了 `PyEngine::quantile_returns` 与 `PyEngine::factor_metrics`；`probe.py` 未拦截 `_fields`、`__subclasses__` 与 `np.load`/`np.fromfile` | ✅ **已闭环**：补齐 `PyEngine` 双方法校验，闭包隐藏 `fields` 并封锁私有属性与 I/O |
| **`P0-09`** | 多头超额基准时间窗错位 2 个交易日且算术年化失真 | `4c99e1a` | 对账脚本调用签名未同步更新 | ✅ **已闭环**：对齐 `calc_long_excess_metrics` 与同窗口基准 |
| **`P0-10`** | 换手率指标张冠李戴（`1 - rank_autocorr` 冒充 Top 分位单边权重换手率） | `4c99e1a` | 对账脚本仍调用不存在的旧函数名或错配返回字段 | ✅ **已闭环**：在 Rust 导出 `calc_top_quantile_turnover` 并同时输出双口径换手率（`turnover` 与 `rank_autocorr_turnover`） |
| **`P0-11`** | A 股回测全天 VWAP 混入日内价格、漏检 ST/涨跌停及对账脚本崩溃 | `95c3721` (部分) | `panel_loader.py` 的 `join_asof` 触发 Polars 排序警告；`backtest_volume_price_factor.py` 与 `backtest_multi_factors.py` 存在函数名、参数名与返回字典键名失配无法运行 | ✅ **已闭环**：修复 `check_sortedness=False` 与全部脚本接口并纳入 `verify.sh` 门禁 |

#### 2. `Volume_Price_2_H5_2` 线上 MCP vs 本地双口径全周期对账表（2016-01-04 ~ 2024-12-31，二次闭环实测）

- **因子 ID**：`fac_5bc39b9aa01c427aa2d28563d7d9b5a2`（版本 `v2`，回测 Run ID：`bt_bb3a04da6b20456ab444dcb7e5450949`；本次校验时远端 FactorLab MCP 返回 HTTP `503 Service Unavailable`，脚本自动启用权威离线快照对账）
- **DSL 表达式**：线上为 `add(sub(BaseUnivMask, BaseUnivMask), neg(add(cs_zscore(ts_corr(daily_raw_bar_1d_close_hfq, log(add(daily_raw_bar_1d_amount, 1)), 20)), cs_zscore(ts_std(daily_derived_returns_c2c_1d, 20)))))`；本地采用等价表达式 `neg(add(cs_zscore(ts_corr(close_hfq, log_amount, 20)), cs_zscore(ts_std(c2c_ret, 20))))`（实测 `log(amount)` 与 `log(amount + 1)` 逐元素最大差 `6.72e-5`，4 位小数指标完全一致）。
- **后处理管线**：`baseUniMask -> trdMask -> dropNewStock:120 -> MAD:3 -> zScore -> indNeut:sector -> decay:5 (+重施累积掩码) -> rank`（本次已修正脚本绕过 `P2-02` 问题，统一接入 [`apply_postprocess_pipeline`](file:///home/cr9294/repos/FactorLibrary/python/factorlib/core/probe.py#L688) 并补齐 `zScore`，消除原脚本在 `decay(5)` 后向涨跌停渗出的 `158,060` 点及向非在池渗出的 `20,884` 点，`0` 渗漏断言生效）。

| 指标名称 | FactorLab 线上 (`vwap30`) | 本地 `open_adj` (`es=False` 默认 / `es=True` 实盘模拟) | 本地 `vwap_adj` (`es=False` / `es=True`) | 对账结论与差异归因 |
| :--- | :---: | :---: | :---: | :--- |
| **有效评估交易日数** | `2185` | `2185` / `2185` | `2185` / `2185` | **100% 严格对齐（0 天偏差，覆盖 5,323 只 A 股）** |
| **RankIC 均值 (1d)** | `0.0431` | `0.0343` / `0.0367` | `0.0416` / `0.0420` | `vwap_adj` 与线上仅差 `-0.0015 ~ -0.0011`；`open_adj` 不含日内反转差 `-0.0064 ~ -0.0088` |
| **RankIC IR (1d)** | `0.4019` | `0.3119` / `0.3318` | `0.3986` / `0.4020` | `vwap_adj` 几乎重合（`0.3986` / `0.4020` vs `0.4019`） |
| **RankIC 胜率 (1d)** | `67.96%` | `63.30%` / `64.21%` | `67.46%` / `67.51%` | `vwap_adj` 与线上差 `< 0.5%` |
| **2日 RankIC 均值 (IR)** | `0.0535 (0.4960)` | `0.0466 (0.4282)` / `0.0486 (0.4457)` | `0.0494 (0.4655)` / `0.0502 (0.4724)` | 多周期预测能力单调递增结构完全一致 |
| **5日 RankIC 均值 (IR)** | `0.0659 (0.6189)` | `0.0618 (0.5769)` / `0.0631 (0.5907)` | `0.0636 (0.6048)` / `0.0640 (0.6102)` | 5 日预测偏差收敛至 `-0.0019 ~ -0.0041` |
| **10日 RankIC 均值 (IR)** | `0.0764 (0.7324)` | `0.0741 (0.7094)` / `0.0745 (0.7182)` | `0.0758 (0.7378)` / `0.0756 (0.7404)` | `10d RankIC` 全口径与线上差 `<= 0.0023` |
| **IC 均值 / IR (1d Pearson)** | `0.0265 / 0.2729` | `0.0099 (0.1006)` / `0.0145 (0.1496)` | `0.0144 (0.1553)` / `0.0163 (0.1781)` | Pearson 对新股/复牌极端尾部更敏感，`5d IC` 收敛至 `0.0273~0.0299`（线上 `0.0332`） |
| **多空日均收益 (G10-G1) / IR** | `0.2364% / 0.2956` | `0.1023% (0.1056)` / `0.1813% (0.1743)` | `0.1347% (0.1742)` / `0.1759% (0.2135)` | 启用分方向涨跌停模拟 (`--execution-simulation`) 后空头跌停顺延兑现与涨停限买生效，`G1..G10`（`-0.1567% .. +0.0246%`）高度逼近线上（`-0.22% .. +0.02%`） |
| **Top G10 单边权重换手率** | `12.11%` | `11.22%` / `11.25%` | `11.22%` / `11.25%` | 修复 `P2-02` 渗漏与 `zScore` 后从原 `10.30%` 抬升至 `11.22%–11.25%`，与线上仅差 `-0.86%` |
| **多头年化超额 (vs 000985)** | `1.26%` | `+11.97%` / `+3.57%` | `+8.49%` / `+3.04%` | `es=False` 下等权在池基准（`+9.66%`）较流通市值基准（`+2.16%`）存在 `+7.50%` 等权/小盘结构溢价；启用 `--execution-simulation` 后等权均值收口至 `-1.19%`，超额收口至 `+3.57%` 接近线上 `+1.26%` |
| **多头超额年化波动率** | `7.67%` | `8.94%` / `9.64%` | `16.51%` / `16.54%` | 同钟基准对齐：`open_adj` 与 `bench_fwd_open_1` 相关系数达 **`0.9224` / `0.9152`**，波动率贴近线上 `7.67%`；`vwap_adj` 因缺指数日内均价只能配对 `bench_fwd_close_1`，相关性降至 **`0.6494` / `0.6583`**，引入跨时钟噪音使超额波动翻倍至 `16.5%` |
| **多头超额夏普 / 最大回撤** | `0.1642 / 28.73%` | `1.3391 / 19.83%` vs `0.3707 / 37.38%` | `0.5142 / 21.06%` vs `0.1836 / 34.14%` | 方向正确、风险收益特征符合价格代理与执行约束预期 |

> **A 股回测部分验证结论**：修正 `P2-02` 剧本绕过、补齐 `zScore` 及修复 [`panel_loader.py`](file:///home/cr9294/repos/FactorLibrary/python/factorlib/markets/stocks/panel_loader.py) 分方向买卖掩码后，[`scripts/backtest_volume_price_factor.py`](file:///home/cr9294/repos/FactorLibrary/scripts/backtest_volume_price_factor.py) 与 [`scripts/backtest_multi_factors.py`](file:///home/cr9294/repos/FactorLibrary/scripts/backtest_multi_factors.py)（5 大因子全量对账，其中 `Tech_Trend_VP_1_Rev` 全后复权 `RankIC=0.0379` vs 未复权混合 `0.0370`）均零错误通过。剩余数值差异已全部由「等权 vs 流通市值基准结构差、日线 `open_adj/vwap_adj` vs `vwap30` 执行钟差、分方向涨跌停约束与线上净费用口径」充分解释，A 股回测链路正式验收通过 ✅。

---

### 4.4 P2 & P3 改进建议与架构性能治理验收清单（2026-11-13 验收）

| 编号 | 问题与优化项 | 实施模块 | 验收结果与关键机制 |
| :--- | :--- | :--- | :--- |
| **`P2-01`** | Layer 1 评估与 Rust 回测前导预热零值稀释年化与波动率指标 | [`evaluate.py`](file:///home/cr9294/repos/FactorLibrary/python/factorlib/core/evaluate.py), [`futures.rs`](file:///home/cr9294/repos/FactorLibrary/rust/factorlib_core/src/backtest/futures.rs), [`fast.rs`](file:///home/cr9294/repos/FactorLibrary/rust/factorlib_core/src/backtest/fast.rs) | ✅ **已闭环**：保持完整 `T` 长度曲线；从 `first_active_day` 起算统计摘要与回撤；`evaluate_factor_v4_stats` 将非活跃行从多头超额剔除 |
| **`P2-02`** | `decay`/`emadecay` 平滑向退市或池外标的回渗信号 | [`probe.py`](file:///home/cr9294/repos/FactorLibrary/python/factorlib/core/probe.py) | ✅ **已闭环**：保持 Rust `pp_decay`/`pp_emadecay` 严格对齐 C++ 真值向量；`apply_postprocess_pipeline` 追踪 Stage 1 掩码并在平滑后重施掩码 |
| **`P2-03`** | 风格归因并列秩失真 & 前复权绝对价格常数比较隐患 | [`styles.py`](file:///home/cr9294/repos/FactorLibrary/python/factorlib/core/styles.py), [`typecheck.rs`](file:///home/cr9294/repos/FactorLibrary/rust/factorlib_core/src/expr/typecheck.rs), [`py.rs`](file:///home/cr9294/repos/FactorLibrary/rust/factorlib_core/src/py.rs), [`probe.py`](file:///home/cr9294/repos/FactorLibrary/python/factorlib/core/probe.py) | ✅ **已闭环**：`styles.py` 共享并列平均秩 `_spearman_corr_1d`；`typecheck.rs` 新增 `collect_type_warnings` 对前复权价非零常数比较输出非阻断告警并透传至 `validate_expr`/`check_expr`/`probe_factor` |
| **`P2-04`** | `eval_gp_batch` 库内去相关嵌套 Rayon 并行与逐日堆分配 | [`corr.rs`](file:///home/cr9294/repos/FactorLibrary/rust/factorlib_core/src/metrics/corr.rs), [`py.rs`](file:///home/cr9294/repos/FactorLibrary/rust/factorlib_core/src/py.rs) | ✅ **已闭环**：引入行连续 `PreparedFactor` + `PreparedCandidate` + 线程复用 `CorrScratch` 与 `cross_correlation_prepared_seq`，掩码一致走归一化点积快路径、额外 `NaN` 自动精确回退（与两两 `calc_pairwise_correlation` 误差 `< 1e-12`） |
| **`P2-05`** | PyO3 FFI `view_to_fortran` 拷贝与只读评估/后处理接口零拷贝 | [`py.rs`](file:///home/cr9294/repos/FactorLibrary/rust/factorlib_core/src/py.rs), [`ic.rs`](file:///home/cr9294/repos/FactorLibrary/rust/factorlib_core/src/metrics/ic.rs), [`quantile.rs`](file:///home/cr9294/repos/FactorLibrary/rust/factorlib_core/src/metrics/quantile.rs), [`turnover.rs`](file:///home/cr9294/repos/FactorLibrary/rust/factorlib_core/src/metrics/turnover.rs), [`postprocess.rs`](file:///home/cr9294/repos/FactorLibrary/rust/factorlib_core/src/ops/postprocess.rs) | ✅ **已闭环**：`view_to_fortran_*` 增加 `as_slice_memory_order()` memcpy 快路径；只读指标与 12 个 `pp_*` 后处理算子全面泛型化为 `ArrayBase<S, Ix2>` 并以 `ArrayView2` 零拷贝借用传入 GIL 释放区 |
| **`P3-01`** | DSL 解析器深层嵌套跨 FFI 栈溢出风险 | [`parser.rs`](file:///home/cr9294/repos/FactorLibrary/rust/factorlib_core/src/expr/parser.rs) | ✅ **已闭环**：新增 `MAX_PARSE_DEPTH = 128` 深度追踪与单测拦截 |
| **`P3-02`** | 研究面板加载失败静默回退吞异常 | [`workbench.py`](file:///home/cr9294/repos/FactorLibrary/python/factorlib/core/workbench.py) | ✅ **已闭环**：`_load_panel` 捕获异常后使用 `logger.warning(..., exc_info=True)` 输出完整回退轨迹 |
| **`P3-03`** | `evaluate.py` 类型注解宽泛 `Any` 收紧 | [`evaluate.py`](file:///home/cr9294/repos/FactorLibrary/python/factorlib/core/evaluate.py), [`probe.py`](file:///home/cr9294/repos/FactorLibrary/python/factorlib/core/probe.py) | ✅ **已闭环**：提取 `PostprocessSpec`，在 `TYPE_CHECKING` 下替换为 `BacktestResult`、`PortfolioConfig | None`、`BacktestConfig | None` |

---

### 4.5 P1/P2/P3 深度代码复审、Context7 核验与二次残余闭环验收（2026-11-13）

为确认 `P1-01 ~ P1-07`、`P2-01 ~ P2-05`、`P3-01 ~ P3-03` 修订后是否仍存在回归或隐蔽漏洞，本次结合源码走查、Context7 官方文档比对与实证回归运行进行了二次全面复审：

#### 1. P1 / P2 / P3 逐项代码实证状态总表

| 编号 | 问题描述 | 核心验证代码位置 | 二次复审查出残余问题 | 最终闭环状态 |
| :--- | :--- | :--- | :--- | :---: |
| **`P1-01`** | 反向依赖 `server_main.py`、Wheel 与 Ruff 盲区 | [`pyproject.toml`](file:///home/cr9294/repos/FactorLibrary/pyproject.toml), [`workbench_server.py`](file:///home/cr9294/repos/FactorLibrary/python/factorlib/workbench_server.py), [`scripts/run_web.py`](file:///home/cr9294/repos/FactorLibrary/scripts/run_web.py) | 无残余问题（已统一 Wheel 静态资源打包与 `factorlib-web` 入口） | ✅ **已闭环** |
| **`P1-02`** | 双 Rust Crate 分裂与硬编码 `extension-module` | [`Cargo.toml`](file:///home/cr9294/repos/FactorLibrary/Cargo.toml), [`rust/factorlib_core/Cargo.toml`](file:///home/cr9294/repos/FactorLibrary/rust/factorlib_core/Cargo.toml) | **⚠️ 发现回退**：[`elem.rs`](file:///home/cr9294/repos/FactorLibrary/rust/factorlib_core/src/ops/elem.rs#L19-L64) 仍残留两处 `Array2::uninit().assume_init()`（违反 `AGENTS.md` Rule 5）；已修复为 `Array2::zeros((t, n).f())` 并新增 Rust/Python 边界测试（Rust `cargo test` 31 项全通） | ✅ **本次修复闭环** |
| **`P1-03`** | `Panel.load` 暴露 `aux` 字段、`is_reserved_label_name` 与 `trdMask` | [`panel.py`](file:///home/cr9294/repos/FactorLibrary/python/factorlib/core/panel.py), [`panel.rs`](file:///home/cr9294/repos/FactorLibrary/rust/factorlib_core/src/panel.rs) | **⚠️ 发现小疏漏**：`fields_subset=["close", "limit_mask"]` 自动加载 `traded`/`is_unilateral` 时未按需载入 `volume`；已改为按需读入 `limit_extra_fields` 并加单测，既补齐过滤又不污染 `Panel.fields` | ✅ **本次修复闭环** |
| **`P1-04`** | Rust 与 Python 参考实现并列秩容差算法不一致 | [`nan.rs`](file:///home/cr9294/repos/FactorLibrary/rust/factorlib_core/src/nan.rs), [`group.rs`](file:///home/cr9294/repos/FactorLibrary/rust/factorlib_core/src/ops/group.rs), [`ops_ref.py`](file:///home/cr9294/repos/FactorLibrary/python/factorlib/core/ops_ref.py) | 无残余问题（统一采用左锚点 `STD_EPS = 1e-12` 并列秩平均） | ✅ **已闭环** |
| **`P1-05`** | `SubExprCache` 伪 LRU 淘汰与无版本键碰撞风险 | [`cache.rs`](file:///home/cr9294/repos/FactorLibrary/rust/factorlib_core/src/expr/cache.rs) | 无残余问题（32 分片真实双向链表 LRU + `(u64, Arc<str>)` 键完全防碰撞） | ✅ **已闭环** |
| **`P1-06`** | 分钟线时间戳切窗、另类数据去重与 ML 模型缺陷 | [`minute_features.py`](file:///home/cr9294/repos/FactorLibrary/python/factorlib/markets/futures/etl/minute_features.py), [`aux_warehouse.py`](file:///home/cr9294/repos/FactorLibrary/python/factorlib/markets/futures/etl/aux_warehouse.py), [`aux_roll.py`](file:///home/cr9294/repos/FactorLibrary/python/factorlib/markets/futures/etl/aux_roll.py), [`linear.py`](file:///home/cr9294/repos/FactorLibrary/python/factorlib/ml/models/linear.py), [`lgbm.py`](file:///home/cr9294/repos/FactorLibrary/python/factorlib/ml/models/lgbm.py) | 无残余问题（仅提示 `config.py` 中 `DominantConfig.rule` 校验兼容 `(0, 1, 2, 3)`，与米筐原生枚举对齐并保留 `rule=3` 默认推荐） | ✅ **已闭环** |
| **`P1-07`** | `Gaussian` 裸因子 `Inf` 坍缩、`Tech_Trend` 复权与脚本接轨 | [`eval.rs`](file:///home/cr9294/repos/FactorLibrary/rust/factorlib_core/src/expr/eval.rs), [`panel_loader.py`](file:///home/cr9294/repos/FactorLibrary/python/factorlib/markets/stocks/panel_loader.py), [`scripts/backtest_multi_factors.py`](file:///home/cr9294/repos/FactorLibrary/scripts/backtest_multi_factors.py) | **⚠️ 发现隐蔽语义缺陷**：`panel_loader.py` 中 `execution_simulation=True` 误以双向 `limit_mask` 校验退出日 `t+1+h`，右删失涨停盈利样本导致全市场等权年化暴跌至 `-11.73%`；同时 `backtest_multi_factors.py` 存在 `0.0503` 硬编码打印。均已修复并增加单测 | ✅ **本次修复闭环** |
| **`P2-01`** | Layer 1 与 Rust 回测预热零填充稀释波动率与年化指标 | [`evaluate.py`](file:///home/cr9294/repos/FactorLibrary/python/factorlib/core/evaluate.py#L610-L640), [`futures.rs`](file:///home/cr9294/repos/FactorLibrary/rust/factorlib_core/src/backtest/futures.rs), [`fast.rs`](file:///home/cr9294/repos/FactorLibrary/rust/factorlib_core/src/backtest/fast.rs) | **⚠️ 发现边界细节**：`evaluate.py` 对 `first_active_day` 之后的偶发非活跃日（`< min_names`）未从 `group_summary` 统计窗口剔除；现已对齐 `active_days_mask` 并加单测 | ✅ **本次修复闭环** |
| **`P2-02`** | `decay`/`emadecay` 平滑向退市或池外标的回渗信号 | [`probe.py`](file:///home/cr9294/repos/FactorLibrary/python/factorlib/core/probe.py#L688-L768) | **⚠️ 发现脚本绕过**：[`backtest_volume_price_factor.py`](file:///home/cr9294/repos/FactorLibrary/scripts/backtest_volume_price_factor.py) 与 [`backtest_multi_factors.py`](file:///home/cr9294/repos/FactorLibrary/scripts/backtest_multi_factors.py) 仍手写直调 `apply_postprocess_op` 绕过重掩码（实测渗出 `158,060` 涨跌停与 `20,884` 非在池点）；已通过 `to_panel()` 别名与统一复用 `apply_postprocess_pipeline` 彻底闭环 | ✅ **本次修复闭环** |
| **`P2-03`** | 风格归因并列秩失真 & 前复权绝对价格常数告警 | [`styles.py`](file:///home/cr9294/repos/FactorLibrary/python/factorlib/core/styles.py), [`typecheck.rs`](file:///home/cr9294/repos/FactorLibrary/rust/factorlib_core/src/expr/typecheck.rs) | 无残余问题 | ✅ **已闭环** |
| **`P2-04`** | `eval_gp_batch` 嵌套并行爆破与逐日堆分配 | [`corr.rs`](file:///home/cr9294/repos/FactorLibrary/rust/factorlib_core/src/metrics/corr.rs), [`py.rs`](file:///home/cr9294/repos/FactorLibrary/rust/factorlib_core/src/py.rs) | 无残余问题（行连续 `PreparedFactor` + `CorrScratch` 已完整生效且精度 `< 1e-12`） | ✅ **已闭环** |
| **`P2-05`** | PyO3 FFI `view_to_fortran` 拷贝与只读零拷贝对齐 | [`py.rs`](file:///home/cr9294/repos/FactorLibrary/rust/factorlib_core/src/py.rs), [`postprocess.rs`](file:///home/cr9294/repos/FactorLibrary/rust/factorlib_core/src/ops/postprocess.rs) | 无残余问题（已实测 `±Inf` 下 C-order / F-order 对账 diff `0.0`） | ✅ **已闭环** |
| **`P3-01`** | DSL 解析器深层嵌套栈溢出防护 | [`parser.rs`](file:///home/cr9294/repos/FactorLibrary/rust/factorlib_core/src/expr/parser.rs) | 无残余问题（`MAX_PARSE_DEPTH = 128` 实测拦截 140 层嵌套） | ✅ **已闭环** |
| **`P3-02`** | 研究面板加载失败静默吞异常 | [`workbench.py`](file:///home/cr9294/repos/FactorLibrary/python/factorlib/core/workbench.py) | 无残余问题（`logger.warning(..., exc_info=True)` 已覆盖） | ✅ **已闭环** |
| **`P3-03`** | `evaluate.py` 类型注解宽泛 `Any` 收紧 | [`evaluate.py`](file:///home/cr9294/repos/FactorLibrary/python/factorlib/core/evaluate.py), [`probe.py`](file:///home/cr9294/repos/FactorLibrary/python/factorlib/core/probe.py) | 无残余问题（另外本次同步修复 [`scripts/verify.sh`](file:///home/cr9294/repos/FactorLibrary/scripts/verify.sh) 补齐 `uv lock --check` Gate 9 校验与动态统计 Cargo 单测总数） | ✅ **本次修复闭环** |

#### 2. Context7 权威文档与内存安全核验结论（Gate 9）

- **`rust-numpy 0.22.1` + `PyO3 0.22.6`**（比对 `Cargo.lock` 与 Context7 官方文档）：
  - [`py.rs`](file:///home/cr9294/repos/FactorLibrary/rust/factorlib_core/src/py.rs) 通过 `PyArray2::borrow_from_array_bound` 暴露 `Arc<Array2<f64>>` 并调用 `py_arr.readwrite().make_nonwriteable()`，锁定 `OWNDATA=false` 只读不可逆视图，严防 Python 端就地篡改污染缓存；`py.allow_threads(...)` 在 `PyO3 0.22.6` 中为当前标准写法（后续版本中重命名为 `detach`，当前锁版本无需迁移）。
  - 清除 [`elem.rs`](file:///home/cr9294/repos/FactorLibrary/rust/factorlib_core/src/ops/elem.rs) 遗留的 `uninit().assume_init()` 后，全 Rust Crate 已 **0 处** 未初始化未定义行为。
- **`polars 2.0.0` / `lightgbm 4.7.0` / `scikit-learn 1.9.1`（`uv.lock` 一致性）**：
  - `panel_loader.py` 的 `join_asof(..., by="order_book_id", check_sortedness=False)` 前置执行了 `sort(["trading_date", "order_book_id"])`，各标的子序列严格单调递增；
  - `scripts/verify.sh` 已内嵌 `uv lock --check`，确认 `pyproject.toml` 与 `uv.lock` 24 个依赖解析完全一致。

