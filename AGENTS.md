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
