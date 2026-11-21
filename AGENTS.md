# AGENTS.md - FactorLibrary 协同开发智能体操作指南

本项目是一个由 Rust 极速计算内核与 Python 科学计算生态融合构建的**多市场（期货 & A 股双数据源）量化因子挖掘、组合优化与实盘级回测平台（FactorLibrary）**。
所有在此代码库工作的智能体（Agents）必须严格遵守以下**量化研究闭环范式（Quantitative Research Closed-Loop Model）**、工程规范与行为准则。

---

## 顶层量化研究闭环架构规范（Quantitative Research Closed-Loop Paradigm）

本平台的一切功能演进、模块划分、配置设计与实验调优，必须严格遵循「**投资研究（Investment Research） → 交易执行（Execution） → 绩效监控与反馈调优（Monitor & Adjustment）**」六阶闭环范式：

```text
                       ┌──────────────────── Adjustment (参数/风控/模型反馈调优) ───────────────────┐
                       ▼                                       ▼                                  │
┌──────────────────────────────────────────────────────────────────────────────┐  ┌────────────┐  ┌┴───────────┐
│                         Investment Research (投资研究)                        │  │ Execution  │  │  Monitor   │
│  ┌──────────┐      ┌───────────┐      ┌───────────────┐      ┌────────────┐  │  │ ┌────────┐ │  │ ┌────────┐ │
│  │ 1. Data  │ ───► │ 2.Factors │ ───► │ 3.Predictions │ ───► │ 4.Positions│ ─┼─►│ │5.Orders│ ┼─►│ │6.Return│ │
│  └──────────┘      └───────────┘      └───────────────┘      └────────────┘  │  │ └────────┘ │  │ └────────┘ │
└──────────────────────────────────────────────────────────────────────────────┘  └────────────┘  └────────────┘
```

### 六阶闭环模块映射与开发契约

| 闭环阶段 | 所属域 | 核心职责与数学/工程边界 | 对应代码模块与配置资产 |
| :--- | :--- | :--- | :--- |
| **1. Data**<br>（双源底座与视图桥接） | Investment Research | • **物理分库、各守时钟、运行期单向投影**：A 股托管 `stocks/rq`（米筐 2015–2026 全量 OHLCV、`daily_basic` 估值换手、`num_trades`、指数成分权重，特征时钟 `open_adj`）与 `stocks/barra`（Barra CNE5 10 风格、31 行业、`VwapBegin30`、特质收益 `CNE5SpecRet`、日频协方差 `COV`，执行时钟 `vwap_open_30m`）两套独立数据集，零物理合表；<br>• **变量别名兼容**：统一支持 FactorLab MCP 标准变量别名（如 `daily_derived_returns_c2c_1d`、`daily_derived_returns_intra`、`barra_cne5_*`）；<br>• **跨源投影**：加载 `rq` 时自动通过 `RiskView` 只读挂载 `barra` 的 `vwap_open_30m` 标签与风格暴露，通过 `SignalBridge`（支持 `T_close`/`T_exec` 显式对齐）跨源投递信号至 `barra`。 | • `rust/factorlib_core/src/panel.rs`<br>• `python/factorlib/core/panel.py`<br>• `python/factorlib/markets/stocks/{panel_loader,ares_bridge,bridge,barra}.py`<br>• `configs/{futures,stocks}/data.yaml` |
| **2. Factors**<br>（算子库与因子生产线） | Investment Research | • **Rust 极速算子库**：维护逐元素、时序滚动（$O(1)$ Welford/单调队列）、横截面、多元加权回归残差（`cs_multi_wls_resi`）、Barra 市值加权标准化（`barra5_zScore`）等核心 1d 算子及 FactorLab 同名别名；<br>• **双模态定义与 12 步声明式后处理**：支持 DSL AST 表达式与因果 Python 代码，链式执行 `baseUniMask → trdMask → dropNewStock → MAD/winsor → indNeut → riskNeutMulti → decay/emadecay → zScore/rank`；<br>• **黄金对账基准**：锁定 `Volume_Price_2_H5_2` 为跨数据源与线上 MCP 算子/后处理数值对齐的唯一黄金基准因子。 | • `rust/factorlib_core/src/ops/`<br>• `rust/factorlib_core/src/expr/`<br>• `python/factorlib/core/{ops_ref,probe,dsl}.py`<br>• `python/factorlib/gp/` |
| **3. Predictions**<br>（横截面收益预测与合成） | Investment Research | • **双轨预测目标**：同时支持总收益标签（`y_1/2/5/10`）与剥离市场/行业/风格后的**纯特质收益标签**（`y_idio_1/2/5/10`）；<br>• **多因子合成与外部信号接入**：支持 Purged Walk-Forward ML（LightGBM/Ridge）、PyTorch DL Alpha 合成，以及通过 `SignalBridge` 导入外部预合成信号（如 `shifted_signal.parquet`）。 | • `python/factorlib/ml/`<br>• `python/factorlib/dl/`<br>• `python/factorlib/markets/stocks/bridge.py`<br>• `configs/{futures,stocks}/ml_default.yaml` |
| **4. Positions**<br>（模块化组合优化器） | Investment Research | • **三层模块化融合架构（`optimizer.py`）**：<br>  1. **信号塑形层（SignalShaper）**：截面排序归一化、软行业中性化（`ind_neutral_weight`）、低波因子倾斜（`-vol_weight * RESVOL`）、非线性头部幂次放大（`power`）与 EMA 慢速记忆（`memory`）；<br>  2. **动态风控守卫（DynamicRiskGuard）**：监测 Barra `FactorRet['SIZE']` 等风格短期异动，在微盘流动性危机期自适应收紧风格敞口与低波惩罚；<br>  3. **凸优化求解层（ConvexOptimizer）**：支持「双期跨期规划 Horizon LP」与「Barra 42 因子低秩协方差 QP/SOCP」组合，内嵌非对称买卖换手惩罚（`asymmetric_trade_penalty`）与防追高跳升门槛（`buy_hurdle`），支持 `MOSEK`/`CLARABEL` 双轨求解；<br>• **持仓漂移闭环**：逐日根据真实价格涨跌、停牌/涨跌停阻断与 `reserve_cash` 动态递推昨日真实漂移持仓 `last_hold`。 | • `python/factorlib/markets/stocks/{optimizer,portfolio}.py`<br>• `python/factorlib/core/portfolio.py`<br>• `configs/stocks/{barra,backtest}.yaml` |
| **5. Orders**<br>（实盘语义订单与撮合） | Execution | • **严格因果执行时钟**：A 股所有回测统一落盘至 `barra` 域以 $T+1$ `vwap_open_30m_adj` 执行（严禁静默回退全天 VWAP）；<br>• **市场微观约束与双费率模式**：A 股 T+1 锁仓、100 股整手、停牌与涨跌停阻断（`blocked_orders`）、期货换月平旧开新；同时支持「A 股真实非对称佣金/印花税/过户费/滑点」与「统一单边费率（如 `fee_rate: 0.003` 对账模式）」。 | • `rust/factorlib_core/src/backtest/{stocks,futures}.rs`<br>• `python/factorlib/markets/stocks/backtest.py`<br>• `configs/{futures,stocks}/fees.yaml` |
| **6. Returns & Adjustment**<br>（绩效归因与反馈调优） | Monitor & Feedback | • **多维绩效监控与 Barra 归因**：输出分年度/全周期净值、Sharpe、最大回撤、特质 RankIC，以及 Barra 42 因子（国家 + 10 风格 + 31 行业 + 特质 Alpha + 交易成本）收益与风险归因；<br>• **Adjustment 反馈调优闭环**：针对特定极端行情段（如 2024 年初微盘股回撤段）定位风险暴露源，通过声明式 YAML 参数与 `--profile` 预设调节 **Predictions**（标签/中性化）、**Positions**（`power`/`ind_neutral_weight`/`vol_weight`/`adaptive_stress_guard`/`asymmetric_trade_penalty`）与 **Orders**（`reserve_cash`/换手阈值），并支持一键导出外部兼容策略文件。 | • `rust/factorlib_core/src/metrics/`<br>• `python/factorlib/core/{evaluate,report,registry}.py`<br>• `scripts/run_stocks_backtest.py`、`scripts/run_backtest.py` |

---

## Git Commit Convention

- After every code change, write a Git commit title in **Chinese**, following the Angular commit convention.
- Commit title format: `<type>(<scope>): <subject>`, where `type` is one of `feat`, `fix`, `refactor`, `docs`, `style`, `test`, `chore`; `scope` is an optional module or feature area (e.g. `data`, `factor`, `backtest`); and `subject` is a concise Chinese description, typically under 50 characters.

### Commit 示例

- `feat(factor): 添加截面 RankIC 与多空分位收益计算算子`
- `fix(etl): 修复郑商所分钟线成交额为零导致的 VWAP 异常`
- `refactor(core): 重构 AST 规范化化简逻辑并支持子表达式缓存`
- `docs(plan): 完善期货因子挖掘平台技术白皮书`
- `chore(env): 初始化 uv 虚拟环境与 rustup 工具链`
- `test(ops): 添加基于 proptest 的时序滚动算子对账测试`

---

## 核心架构与目录分工

1. **`python/factorlib/`**：主 Python 算法包
   - `core/`：面板数据抽象、DSL 语法糖、静态探针与 12 步后处理（`probe.py`）、评价体系与 Barra 归因（`evaluate.py`）、组合构建、因子库 ORM、运行追踪
   - `gp/`：强类型遗传规划（STGP）自动化因子挖掘系统（原语、变异交叉、多岛模型、准入管线）
   - `ml/` & `dl/`：Purged Walk-Forward 机器学习与 PyTorch 深度学习多因子合成管线（支持总收益 `y_h` 与纯特质收益 `y_idio_h` 双轨训练）
   - `markets/futures/`：期货市场适配层（ETL、主力映射、微结构特征提取、合约规格、费率模型）
   - `markets/stocks/`：A 股双数据源市场适配层（米筐 `panel_loader.py`、`.ares` 转换 `ares_bridge.py`、跨源桥接 `bridge.py`、动态特质收益与风险库 `barra.py`、模块化融合凸优化器 `optimizer.py`、回测执行 `backtest.py`）
2. **`rust/factorlib_core/`**：Rust 原生计算内核（通过 PyO3 导出为 `factorlib_core`）
   - `panel.rs`：列优先 Fortran 矩阵与对齐掩码
   - `ops/`：向量化逐元素、时序滚动（$O(1)$ Welford / 单调队列）、横截面、多元回归残差与后处理算子（与 FactorLab MCP 1d 核心算子库及同名别名对齐）
   - `expr/`：AST 语法树、词法语法分析器、静态类型检查、规范化化简、LRU 缓存
   - `metrics/`：RankIC、特质 RankIC、分位收益、换手率、相关性矩阵计算
   - `backtest/`：期货（`futures.rs`）与 A 股（`stocks.rs`）实盘语义向量化回测状态机（手数/整手离散化、T+1 锁仓、涨跌停/停牌阻断、换月双边费/A 股双费率模式、逐日盯市）
3. **`configs/futures/` & `configs/stocks/`**：声明式 YAML 配置文件（双源数据声明 `data.yaml`、Barra 风险模型 `barra.yaml`、品种/股票池、费率、GP/ML 超参、模块化组合优化与回测配置 `backtest.yaml`、本地 `mosek.lic`）
4. **`legacy/`**：原 pyrust-bt 股票回测原型存档（供对照参考，不直接修改）
5. **`exports/`**（已 gitignore）：只读的米筐原始期货 Parquet 数据资产（约 12.7GB）
6. **`data/`**（已 gitignore）：原始股票数据（`raw/stocks/{rq,barra}`）、派生研究面板长表（`derived/futures/v1`、`derived/stocks/{rq,barra}` + `barra/risk_store/`）与本地 SQLite 因子库
7. **`runs/`**（已 gitignore）：实验产出目录（指标、归因报告、调参日志、HTML 诊断报告）

---

## 智能体行为与质量红线

1. **严禁未来函数（No Look-Ahead Bias）**：
   - 统一遵循米筐 `trading_date`。$T$ 日信号只能使用截至 $T$ 日 15:00 收盘已知的数据。
   - 撮合成交统一使用 $T+1$ 开盘前 30 分钟的 VWAP（$\sum(\text{close} \times \text{volume}) / \sum \text{volume}$）。
   - 换月日严禁使用跨合约价格直接作差，必须分别以新旧合约实际成交价进行平旧开新。
   - 机器学习训练必须进行严密的时序隔离（Purge Window = $h$）。
2. **数据资产保护**：
   - 严禁将 `exports/`、`data/`、`runs/`、`*.parquet`、`*.sqlite` 等数据文件提交至 Git 仓库。
   - 在执行 `git add` 之前必须核查 `git status`，杜绝大文件入库。
3. **Rust / Python 协同规范**：
   - 耗时计算（表达式求值、时序滚动、指标统计、实盘撮合）下沉到 Rust，并在密集批处理前显式释放 Python GIL。
   - 所有算子修改均需通过 Python NumPy 参考实现（`ops_ref.py`）进行对账测试。
4. **文档与注释规范**：
   - 保持代码文档与架构设计（`docs/futures_platform_plan.md`）严格同步。
   - 提交代码必须遵循上方的 Git Commit 规范。
5. **Context7 权威文档审查与跨语言安全红线（Context7 Authoritative Verification & Memory Safety）**：
   - 凡涉及核心第三方库（PyO3、rust-numpy、Polars、LightGBM、Scikit-Learn 等）的跨语言接口、版本升级、破坏性变更（Breaking Changes）或生命周期与内存安全约定（如 FFI 边界、只读/可变视图借用、并发线程安全），智能体**严禁凭经验臆测编写 API 调用**，必须通过 Context7 MCP 工具（`resolve-library-id` 与 `query-docs`）检索最新官方权威文档、API 签名与废弃声明；
   - 跨语言导出共享矩阵（如 Rust `Arc<Array2<f64>>` 通过 `PyArray2::borrow_from_array_bound` 暴露给 Python）时，必须通过 `py_arr.readwrite().make_nonwriteable()` 强制锁定为只读视图，严防 Python 端就地修改（in-place mutation）造成底座面板与全局 LRU 缓存的隐式污染；
   - 严禁在 Rust 算子中使用 `Array2::uninit().assume_init()` 等产生未定义行为（UB）的代码，统一使用 `Array2::zeros` 或 `alloc_nan_matrix` 保证安全。

---

## 代码提交前检查认证规范（Pre-Commit Verification Gates）

所有智能体与开发者在完成任务、执行 `git commit` 之前，**必须**完整运行并通过以下质量认证门禁。任何一项未通过均严禁提交代码！

### 一键全量认证命令

```bash
# 全量门禁校验（推荐在 git commit 前执行）
bash scripts/verify.sh --all

# 快速校验（日常开发快速反馈：格式、静态 Lint 与配置检查）
bash scripts/verify.sh --fast
```

### 分语言与分资产认证逻辑

#### 1. Rust 内核代码认证（Rust Gate）
- **`rust-analyzer` 静态诊断与 LSP 扫描**：
  ```bash
  rust-analyzer diagnostics rust/factorlib_core
  ```
  要求：全 Crate 扫描必须 0 致命诊断错误，杜绝无效类型、未决宏或死锁隐患。
- **编译器静态语法与类型检查**：
  ```bash
  cargo check --manifest-path rust/factorlib_core/Cargo.toml --all-targets
  ```
- **Clippy 静态分析（零警告准入）**：
  ```bash
  cargo clippy --manifest-path rust/factorlib_core/Cargo.toml --all-targets -- -D warnings
  ```
  要求：严格禁止任何 Warning 入库，保证最高级别的内存安全与最佳实践。
- **代码排版风格规范**：
  ```bash
  cargo fmt --manifest-path rust/factorlib_core/Cargo.toml -- --check
  ```
- **单元测试与 Proptest 属性测试**：
  ```bash
  cargo test --manifest-path rust/factorlib_core/Cargo.toml
  ```
  要求：包含全部 68 算子注册测试、AST 规范化与化简测试、非法语法阻断测试，以及基于 `proptest` 的双栈 Welford 与单调队列极限性质测试全部 100% 通过。

#### 2. Python 算法层代码认证（Python Gate）
- **Ruff 静态 Lint 扫描与死代码检测**：
  ```bash
  ruff check python/factorlib tests/python scripts
  ```
  要求：无未捕获异常、未解析引用、未使用的变量或导入（自动修复可用 `ruff check --fix`）。
- **Ruff 代码格式一致性检查**：
  ```bash
  ruff format --check python/factorlib tests/python scripts
  ```
  要求：保持 100 字符行长、双引号等统一代码排版（格式化可用 `ruff format`）。
- **自动化单元与集成测试套件**：
  ```bash
  pytest tests/python
  ```
  要求：配置解析、ETL 构建、面板内存容器及 Rust 内核绑定的全部测试用例 100% 通过。

#### 3. 配置文件验证逻辑（YAML / TOML Configs Gate）
- **语法合规性检查**：
  - 自动遍历 `configs/futures/*.yaml`、`pyproject.toml` 及 `rust/factorlib_core/Cargo.toml`，确保能够被标准解析器（`yaml.safe_load` 与 `tomllib.load`）正确解析，无缩进或格式畸变。
- **业务语义与范围规则校验**：
  ```bash
  pytest tests/python/test_config.py
  ```
  - **主力规则校验**：`dominant.rule` 必须为有效枚举（`[1, 2, 3]`，默认强制采用 `rule=3` 避开交割月流动性枯竭）；
  - **撮合安全约束**：开盘成交价窗口 `window_minutes > 0`，且 `calc_method` 严格锁定为 `vwap_close_volume`（基于收盘价与成交量加权），**严禁**使用 `turnover/volume` 规避郑商所历史成交额置零除零隐患；
  - **费率与保证金**：基础费率非负，合约乘数大于 0。

#### 4. 算子跨语言高精度对账逻辑（Operator Parity Gate）
- **NumPy 参考实现对齐**：
  - 任何新增或修改的计算内核算子，必须在 `python/factorlib/core/ops_ref.py` 中提供纯 Python/NumPy 高精度参考实现；
  - 在 `tests/python/test_rust_core.py` 中注册对账测试，并在 `rust/factorlib_core/tests/proptest_ops.rs` 中提供对应性质测试。
- **极端数值边界检验**：
  - 必须在高密度 NaN（如 22% 缺失率）、全 NaN 序列、全常量序列、单点有效、±Inf、以及窗口大于历史长度的极端场景下，验证 Rust 内核输出与参考实现的数值差异严格满足：
    $$\max |y_{\text{rust}} - y_{\text{ref}}| < 10^{-9}$$
  - 算子内部严禁触发 panic、浮点异常崩溃或未定义行为。

#### 5. 反未来函数与因果摄动审计（Anti-Lookahead Perturbation Gate）
- **因果摄动回归测试**：
  ```bash
  pytest tests/python/test_perturbation.py
  ```
  - 在全历史面板中随机选取基准交易日 $T_{\text{perturb}}$，人为篡改 $T > T_{\text{perturb}}$ 的后续行情数据；
  - 检验 $T \le T_{\text{perturb}}$ 的所有历史因子值、复权因子、截面标签与主力合约映射必须保持 **0 差值（绝对 bit 等价）**；
- **AST 语法树防未来语法阻断**：
  - 通过 `rust/factorlib_core/src/expr/typecheck.rs` 静态拦截任何前向未来算子（如 `shift(-k)`、前向移动均值）及未注册未知变量。

#### 6. Git 暂存区与数据安全隔离红线（Data Leakage Gate）
- **暂存区（Staging Area）自动审计**：
  ```bash
  git diff --cached --name-only
  ```
- **禁止入库清单**：
  - 严禁将数据资产目录（`exports/`、`data/`、`runs/`）暂存；
  - 严禁暂存数据文件：`*.parquet`、`*.sqlite`、`*.arrow`、`*.feather`、`*.h5`、`*.bin`、`*.tar`、`*.zip`；
  - 严格限制单文件提交大小：任何单个暂存文件不得超过 **1MB**（1,048,576 字节）。

#### 7. 脚本与 CLI 工具健壮性校验（CLI & Scripts Gate）
- 针对 `scripts/` 目录中的维护与生产脚本（如 `build_panel.py`、`audit_data.py`），验证其 `--help` 命令行入口可正常解析并返回状态码 0，确保生产调用入口健康。

#### 8. 文档与白皮书同步规范（Documentation Consistency Gate）
- 任何架构升级、算子扩充或配置变更，必须同步更新技术白皮书 `docs/futures_platform_plan.md`；
- Git Commit Message 必须严格遵循中文 Angular Commit 规范，格式为 `<type>(<scope>): <subject>`。

#### 9. 权威 API 规范与依赖对齐门禁（Context7 API Compliance & Dependency Gate）
- **Context7 文档核验规范**：
  - 针对 Rust/PyO3 跨语言绑定与 Python 科学计算栈的关键调用，必须比对 Context7 查得的对应版本 API 文档，杜绝已被弃用（Deprecated）或行为发生变更的接口；
- **只读不可变性与内存安全审计**：
  - `Engine.eval()` 与算子批求值输出的 NumPy 数组必须确保 `flags.writeable == False`，若尝试就地赋值必须抛出 `ValueError`；
- **锁文件与依赖版本一致性**：
  - `pyproject.toml` 中的任何依赖增减必须同步刷新并检验 `uv.lock`（执行 `uv lock --check`），确保可重复构建。
