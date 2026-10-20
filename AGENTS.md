# AGENTS.md - FactorLibrary 协同开发智能体操作指南

本项目是一个由 Rust 极速计算内核与 Python 科学计算生态融合构建的**期货量化因子挖掘与回测平台（FactorLibrary）**。
所有在此代码库工作的智能体（Agents）必须严格遵守以下工程规范与行为准则。

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
   - `core/`：面板数据抽象、DSL 语法糖、评价体系、组合构建、因子库 ORM、运行追踪
   - `gp/`：强类型遗传规划（STGP）自动化因子挖掘系统（原语、变异交叉、多岛模型、准入管线）
   - `ml/`：Purged Walk-Forward 机器学习多因子合成管线（特征工程、LightGBM、岭回归）
   - `markets/futures/`：期货市场适配层（ETL、主力映射、微结构特征提取、合约规格、费率模型）
   - `markets/stocks/`：A 股市场适配层（预留二期实现）
2. **`rust/factorlib_core/`**：Rust 原生计算内核（通过 PyO3 导出为 `factorlib_core`）
   - `panel.rs`：列优先 Fortran 矩阵与对齐掩码
   - `ops/`：50+ 向量化逐元素、时序滚动（$O(1)$ Welford / 单调队列）、横截面算子
   - `expr/`：AST 语法树、词法语法分析器、静态类型检查、规范化化简、LRU 缓存
   - `metrics/`：RankIC、分位收益、换手率、相关性矩阵计算
   - `backtest/`：期货实盘语义向量化回测状态机（手数离散化、涨跌停阻断、换月双边费、逐日盯市）
3. **`configs/futures/`**：YAML 配置文件（数据、品种池、费率、GP 超参、ML 超参、回测配置）
4. **`legacy/`**：原 pyrust-bt 股票回测原型存档（供对照参考，不直接修改）
5. **`exports/`**（已 gitignore）：只读的米筐原始 Parquet 数据资产（约 12.7GB）
6. **`data/`**（已 gitignore）：派生研究面板长表与本地 SQLite 因子库
7. **`runs/`**（已 gitignore）：实验产出目录（指标、日志、HTML 诊断报告）

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

