# FactorLibrary: 工业级期货量化因子挖掘与实盘回测平台

<div align="center">

[![Python](https://img.shields.io/badge/Python-3.12+-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![Rust](https://img.shields.io/badge/Rust-2021_Edition-000000?logo=rust&logoColor=white)](https://www.rust-lang.org/)
[![PyO3](https://img.shields.io/badge/PyO3-0.22+-blue)](https://pyo3.rs/)
[![Polars](https://img.shields.io/badge/Polars-1.0+-CD792C?logo=polars&logoColor=white)](https://pola.rs/)
[![LightGBM](https://img.shields.io/badge/LightGBM-4.0+-008080)](https://lightgbm.readthedocs.io/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

<h3>极速 Rust 原生计算内核 × 现代 Python 科学计算生态</h3>
<p><b>高频降采样微观结构特征 → DSL 符号表达式 → 强类型遗传规划 (STGP) 批量挖掘 → SQLite 因子库去相关管理 → Purged Walk-Forward 机器学习多因子合成 → 严密防未来函数与期货实盘语义回测</b></p>

</div>

---

## 目录 (Table of Contents)

- [平台简介](#平台简介)
- [核心架构与数据流](#核心架构与数据流)
- [八大核心特性](#八大核心特性)
- [快速上手](#快速上手)
  - [1. 环境安装与构建](#1-环境安装与构建)
  - [2. 提交前门禁快速验证](#2-提交前门禁快速验证)
- [数据管理与面板构建](#数据管理与面板构建)
  - [1. 原始数据资产](#1-原始数据资产)
  - [2. 数据资产深度自检与审计](#2-数据资产深度自检与审计)
  - [3. 标准化研究面板构建](#3-标准化研究面板构建)
  - [4. 派生面板结构解析](#4-派生面板结构解析)
- [端到端一键流水线 (CLI)](#端到端一键流水线-cli)
  - [场景 A：单因子全流程极速回测](#场景-a单因子全流程极速回测)
  - [场景 B：因子资产库联动合成回测](#场景-b因子资产库联动合成回测)
  - [场景 C：STGP 演化挖掘 + ML 合成全流程](#场景-cstgp-演化挖掘--ml-合成全流程)
  - [回测结果与实验产物解读](#回测结果与实验产物解读)
- [分模块生产级 CLI 脚本指南](#分模块生产级-cli-脚本指南)
  - [1. 种子因子入库 (`init_seed_factors.py`)](#1-种子因子入库-init_seed_factorspy)
  - [2. 强类型遗传规划独立挖掘 (`run_gp.py`)](#2-强类型遗传规划独立挖掘-run_gppy)
  - [3. Purged Walk-Forward 机器学习独立合成 (`run_ml.py`)](#3-purged-walk-forward-机器学习独立合成-run_mlpy)
  - [4. 期货实盘向量化回测 (`run_backtest.py`)](#4-期货实盘向量化回测-run_backtestpy)
  - [5. 现代化 6 阶段量化因子研发 Web 工作台 (`run_web.py`)](#5-现代化-6-阶段量化因子研发-web-工作台-run_webpy)
  - [6. 全资产代码质量认证门禁 (`verify.sh`)](#6-全资产代码质量认证门禁-verifysh)
  - [7. A 股股票因子回测与跨平台对账脚本](#7-a-股股票因子回测与跨平台对账脚本)
- [Python SDK 核心编程范式](#python-sdk-核心编程范式)
  - [1. 面板容器操作 (`factorlib.core.panel`)](#1-面板容器操作-factorlibcorepanel)
  - [2. DSL 表达式构建与静态检查 (`factorlib.core.dsl`)](#2-dsl-表达式构建与静态检查-factorlibcoredsl)
  - [3. Rust 极速计算内核交互 (`factorlib_core.Engine`)](#3-rust-极速计算内核交互-factorlib_coreengine)
  - [4. SQLite 因子全生命周期注册库 (`factorlib.core.registry`)](#4-sqlite-因子全生命周期注册库-factorlibcoreregistry)
  - [5. 因子综合评估体系 (`factorlib.core.evaluate`)](#5-因子综合评估体系-factorlibcoreevaluate)
  - [6. 组合优化与实盘回测状态机 (`factorlib.core.portfolio` / `backtest`)](#6-组合优化与实盘回测状态机-factorlibcoreportfolio--backtest)
- [68 项高性能算子速查表](#68-项高性能算子速查表)
- [严密防未来函数与交易语义规范](#严密防未来函数与交易语义规范)
- [配置文件索引与规范](#配置文件索引与规范)
- [代码库目录导航](#代码库目录导航)
- [A 股股票市场适配与全市场回测对账 (Phase 2 已可用)](#a-股股票市场适配与全市场回测对账-phase-2-已可用)

---

## 平台简介

**FactorLibrary** 是一个专为中国商品与股指期货市场打造的工业级量化因子挖掘与实盘回测平台。平台严格遵循「**市场无关计算内核（Core）**」与「**市场适配层（MarketAdapter）**」的绝对解耦设计原则，核心热路径采用 Rust 原生实现并通过 PyO3 导出为极速扩展库，上层采用 Python 构建灵活强大的量化研究全生命周期工作流。

系统全流程实现**严密防未来函数**因果对齐机制，从原始微观高频行情流式清洗、VWAP 撮合价提取，到 AST 代数化简、强类型遗传规划（STGP）批量挖掘、五步准入漏斗、Purged Walk-Forward 机器学习多因子合成，直至手数离散化实盘语义回测与交互式 Plotly HTML 报告产出，支持百代万级表达式高并发快速评估。同时预留了完整的 A 股股票市场扩展适配接口规范。

---

## 核心架构与数据流

```mermaid
flowchart TD
    subgraph DataLayer["1. 数据管道与市场适配 (factorlib.markets.futures.etl)"]
        Raw["米筐原始 Parquet (exports/ 只读)"]
        ETL["Polars 流式清洗与微结构特征提取"]
        Derived["标准化研究面板 (data/derived/futures/v1)"]
        Raw --> ETL --> Derived
    end

    subgraph RustCore["2. Rust 极速计算内核 (factorlib_core)"]
        PanelMem["T x N Fortran 列优先连续内存矩阵"]
        AST["AST 词法语法解析 / 静态类型检查 / 代数化简"]
        LRU["全局子表达式 ahash LRU 缓存"]
        RayonEval["Rayon 多核批并行求值引擎"]
        Ops["68 项向量化、增量时序与截面算子"]
        Metrics["RankIC / 分位收益 / 换手率 / 相关性矩阵"]
        Derived --> PanelMem
        PanelMem --> RayonEval
        AST --> LRU --> RayonEval --> Ops --> Metrics
    end

    subgraph ResearchFlow["3. 因子挖掘与机器学习流 (factorlib)"]
        STGP["强类型遗传规划 (STGP 环形多岛拓扑迁移)"]
        Admission["五步准入漏斗 (ICIR / 换手率 / 贪心正交去相关)"]
        Registry[("SQLite 因子资产全生命周期注册库 (factors.sqlite)")]
        ML["Purged Walk-Forward 机器学习多因子合成 (LightGBM/Ridge)"]
        Portfolio["风险平价 / Ledoit-Wolf 协方差收缩 / 调仓缓冲带"]
        STGP <-->|"批求值 / 适应度"| RustCore
        STGP --> Admission --> Registry
        Registry --> ML --> Portfolio
    end

    subgraph BacktestEngine["4. 期货实盘语义向量化回测 (factorlib_core::backtest)"]
        TargetW["目标连续权重矩阵 W (T x N)"]
        DiscreteLots["合约乘数与开盘真实 VWAP 离散化为整数手"]
        Limits["涨跌停锁死与单边市禁止开平仓"]
        RollOver["换月展期状态机 (平旧开新 / 双边交易费与滑点)"]
        MTM["逐日盯市 (Mark-to-Market) 记账与动态保证金风控"]
        Portfolio --> TargetW --> DiscreteLots --> Limits --> RollOver --> MTM
    end

    subgraph Reporting["5. 可视化诊断与资产追踪 (runs/)"]
        HTML["交互式 Plotly HTML 诊断报告 (report.html)"]
        Summary["财务绩效指标汇总 (summary.json)"]
        Flow["每日逐日记账流水 (equity.csv)"]
        MTM --> HTML
        MTM --> Summary
        MTM --> Flow
    end
```

---

## 八大核心特性

1. **Rust 原生极速计算内核**：
   - 基于连续内存的 $(T, N)$ Fortran-order 列优先宽矩阵，极致发挥 CPU L1/L2 缓存与 SIMD 向量化性能；
   - 68 项算子（包含时序增量 Welford $O(1)$ 均值/方差、单调队列极值、横截面百分比排名与板块中性化）；
   - 多表达式通过 Rayon 释放全部多核并发，全局子表达式 LRU 缓存避免重复求值。
2. **强类型遗传规划（STGP）批量因子挖掘**：
   - 强类型节点系统（`Series`, `Window`, `Constant`, `Bool`），从语法树生成期杜绝类型畸变与无效求值；
   - 支持交叉、子树变异、点变异、提升突变（Hoist Mutation 强力抑制代码膨胀）与环形多岛拓扑迁移。
3. **五步自动化准入漏斗管线**：
   - Top-K 名人堂初筛 $\to$ 验证集同向显著性约束 $\to$ 截面换手率软门槛 $\to$ 贪心正交去相关（$\max |\rho| < 0.60$） $\to$ 原子化入库。
4. **SQLite 因子资产全生命周期状态机**：
   - 基于唯一代数指纹 `canon_hash` 精确去重，四阶段生命周期流转：`candidate` $\to$ `accepted` $\to$ `active` $\to$ `deprecated`；
   - 完备记录实验批次轨迹（`run_tracker`）与跨样本段（Train/Valid/Test）多预测期数（1, 3, 5, 10, 20）评估指标。
5. **Purged Walk-Forward 机器学习多因子合成**：
   - 按年扩展（Expanding）与滚动（Rolling）重训机制，训练/验证/测试严密时序隔离带（Purge Window = $h$），杜绝标签重叠前瞻泄露；
   - 截面中位数极值截断（MAD Winsorization）与排序高斯化（Rank Gauss）；
   - 支持 Equal Weight、Rolling ICIR、Ridge 与 LightGBM 截面排序回归。
6. **期货实盘交易语义向量化回测**：
   - 统一遵循米筐 `trading_date`，信号基于 $T$ 日收盘生成，执行于 $T+1$ 开盘前 30 分钟真实成交价 $\text{VWAP} = \sum(\text{close} \times \text{volume}) / \sum \text{volume}$；
   - 目标连续权重动态转为品种整数手数；
   - 涨跌停锁死阻断、换月平旧开新双边手续费与滑点、动态保证金超限强平阻断；
   - 严格满足每日盯市记账恒等式：$\text{Cash}_d + \sum (\text{lots}_{i,d} \times \text{settle}_{i,d} \times \text{Multiplier}_i) == \text{Equity}_d$。
7. **全资产交付与质量认证门禁体系**：
   - 内置 `bash scripts/verify.sh --all`，涵盖暂存区数据安全审计、`uv lock --check`、YAML/TOML 配置校验、Rust `rust-analyzer` / 零警告 Clippy / `cargo test`（31 项）、Python Ruff / Pytest（139 项）、与因果摄动防未来函数审计（2 项）及 12 项 CLI 入口体检。
8. **A 股股票市场适配与统一无渗漏后处理回测**：
   - 已实现 [`StockMarketAdapter`](python/factorlib/markets/stocks/adapter.py) 与 [`StockPanelLoader`](python/factorlib/markets/stocks/panel_loader.py)，支持申万行业/流通市值中性化、[`apply_postprocess_pipeline`](python/factorlib/core/probe.py) 末步掩码强制重施，以及方向性涨跌停撮合模拟（`can_buy_mask` 拦截涨停买入、`can_sell_mask` 遇跌停/停牌自动顺延平仓）。

---

## 快速上手

### 1. 环境安装与构建

推荐使用现代 Python 包管理器 `uv`（或原生 `python3.12 -m venv`）配合 Rust 工具链（`rustup` / `cargo`）：

```bash
# 1. 确保 Rust 工具链就绪 (rustc >= 1.80)
rustc --version
cargo --version

# 2. 创建并激活 Python 3.12 虚拟环境
# 方式 A (推荐 uv):
uv venv --python 3.12 .venv
source .venv/bin/activate

# 方式 B (原生 python3.12):
python3.12 -m venv .venv
source .venv/bin/activate

# 3. 安装 Python 核心依赖与开发套件
pip install -e ".[dev]"

# 4. 编译并挂载 Rust 核心扩展库 (release 极速模式)
maturin develop --release --manifest-path rust/factorlib_core/Cargo.toml
```

### 2. 提交前门禁快速验证

安装完成后，可立即执行门禁脚本快速检验环境与依赖完整性：

```bash
# 快速静态校验 (格式、Lint 与配置检查)
bash scripts/verify.sh --fast

# 全量门禁认证 (含 uv lock、31 项 Rust 测试、139 项 Python 单元测试、2 项因果摄动审计与 CLI 体检)
bash scripts/verify.sh --all
```

### 3. 启动现代化 6 阶段量化因子研发工作台 (Web Studio)

平台提供零外部构建依赖的现代量化因子单页工作台（SPA），无缝串联 6 阶段研发流程：

```bash
# 启动零依赖量化研发 Web 单页工作台 (浏览器访问 http://127.0.0.1:8080)
python scripts/run_web.py

# 强制启用内置合成演示面板 (无须本地数据即可开箱体验)
python scripts/run_web.py --synthetic
```

---

## 数据管理与面板构建

### 1. 原始数据资产

平台默认支持米筐（Ricequant）格式期货历史行情数据资产（只读存放于 `exports/` 目录，约 12.7GB）：
- `bar_1d/`：全品种日线 OHLCV、持仓量、结算价与成交额；
- `bar_1m/`：全品种分钟线，用于流式提取 $T+1$ 开盘前 30 分钟真实成交价 VWAP；
- `calendar/`：统一交易日历 (`calendar.parquet`)；
- `dominant_mapping/`：主力合约映射字典（`rule=3` 避开交割月流动性枯竭）；
- `roll_yield/`：展期收益率衍生数据；
- `warehouse_stocks/`：注册仓单库存数据；
- `member_rank/`：期货交易所龙虎榜会员前 20 大持仓数据。

### 2. 数据资产深度自检与审计

在首次使用前，可调用数据审计工具对原始数据资产进行全量完整性扫描：

```bash
# 快速自检模式
python scripts/audit_data.py --quick

# 全量深度审计并输出 Markdown 诊断报告
python scripts/audit_data.py --output data/derived/futures/audit_report.md
```

### 3. 标准化研究面板构建

调用 ETL 流水线将原始分品种数据流式清洗为内存友好的宽表与标签集：

```bash
# 快速演示模式 (仅构建 2024-02-01 ~ 2024-03-31 两个月演示面板)
python scripts/build_panel.py --quick

# 全历史标准化面板构建 (覆盖 2018-01-01 至 2026-08-31)
python scripts/build_panel.py --start-date 2018-01-01 --end-date 2026-08-31 --version v1
```

### 4. 派生面板结构解析

构建完成的研究面板落盘于 `data/derived/futures/v1/`：
- `panel_daily.parquet`：宽表格式日线特征矩阵（含价格、微观结构特征、基本面衍生指标）；
- `labels.parquet`：预测目标收益率标签（例如 $y_1, y_3, y_5, y_{10}, y_{20}$）；
- `rolls.parquet`：主力换月展期事件记录表（旧主力、新主力、换月生效日）；
- `fields.json`：特征字段名、维度与数据类型元数据；
- `meta.json`：面板版本、覆盖日期范围、标的品种池指纹。

---

## 端到端一键流水线 (CLI)

`scripts/run_pipeline.py` 是平台最核心的端到端调度入口，串联「面板加载 $\to$ 种子因子注册 $\to$ STGP 演化挖掘 $\to$ 机器学习多因子合成 $\to$ 风险平价组合优化 $\to$ Rust 期货实盘状态机回测 $\to$ Plotly 交互式 HTML 诊断报告生成」。

### 场景 A：单因子全流程极速回测

若您希望快速评估单个因子表达式的实盘收益与回撤表现，直接传入 `--factor`：

```bash
# 评估 20 日价格动量单因子
python scripts/run_pipeline.py --factor "pct_change(close, 20)"

# 评估 20 日时序 Z-Score 因子
python scripts/run_pipeline.py --factor "ts_zscore(close, 20)"

# 评估微观结构复合因子
python scripts/run_pipeline.py --factor "cs_rank(ts_mean(sub(close, open), 20))"
```

### 场景 B：因子资产库联动合成回测

若不指定 `--factor`，流水线将自动连接 SQLite 因子资产库，提取所有标记为 `accepted` 的因子资产，通过机器学习管线执行合成回测：

```bash
# 使用因子库资产执行截面等权 (Equal Weight) 合成回测
python scripts/run_pipeline.py --ml-model equal_weight

# 使用岭回归 (Ridge) 滚动重训合成回测
python scripts/run_pipeline.py --ml-model ridge

# 使用 LightGBM 决策树截面排序回归合成回测
python scripts/run_pipeline.py --ml-model lgbm
```

### 场景 C：STGP 演化挖掘 + ML 合成全流程

一键开启遗传规划自动化挖掘，让算法自主进化新因子，并通过五步准入漏斗筛选入库后执行回测：

```bash
python scripts/run_pipeline.py \
  --run-gp \
  --gp-population 200 \
  --gp-generations 10 \
  --ml-model equal_weight \
  --output-dir runs/my_gp_experiment
```

### 回测结果与实验产物解读

每次运行完成后，实验产物自动统一保存于 `runs/pipeline_YYYYMMDD_HHMMSS/`（或自定义 `--output-dir`）中：

| 产物文件 | 说明 | 核心价值 |
|---|---|---|
| `report.html` | **交互式 Plotly 诊断报告** | 包含累计净值走势曲线、最大回撤水下分布、月度收益热力矩阵、每日单边换手率柱状图 |
| `summary.json` | **财务绩效汇总指标** | 年化收益率 (`annual_return`)、夏普比率 (`sharpe`)、最大回撤 (`max_drawdown`)、Calmar 比率、日均换手率 (`daily_turnover_mean`) |
| `equity.csv` | **每日盯市账户记账流水** | 逐日记载：`trading_date`, `equity`, `cash`, `margin_used`, `gross_leverage`, `net_pnl`, `turnover` |
| `pipeline_summary.json` | **全链路元数据与耗时** | 记录面板尺寸、特征数量、调仓天数、因子表达式与阶段耗时统计 |

---

## 分模块生产级 CLI 脚本指南

除了全流程流水线外，平台提供了完备的独立模块化命令行工具，便于分阶段研究与离线批处理：

### 1. 种子因子入库 (`init_seed_factors.py`)

评估经典文献 4 大种子因子（价格动量、展期收益率、5日仓单变动、5日主力净多变动），计算各样本区间指标并写入 SQLite 因子库：

```bash
python scripts/init_seed_factors.py \
  --panel data/derived/futures/v1 \
  --db data/registry/factors.sqlite \
  --run-id seed_init_v1
```

### 2. 强类型遗传规划独立挖掘 (`run_gp.py`)

启动多岛模型（Islands Topology）并行演化，配合五步准入漏斗（Top-K 初筛 $\to$ 验证集同向显著性 $\to$ 换手率软门槛 $\to$ 贪心正交去相关 $\to$ 自动入库）：

```bash
python scripts/run_gp.py \
  --config configs/futures/gp_default.yaml \
  --population 500 \
  --generations 30 \
  --islands 4 \
  --seed 42 \
  --min-valid-icir 0.3 \
  --runs-dir runs
```

### 3. Purged Walk-Forward 机器学习独立合成 (`run_ml.py`)

从因子库加载合格因子矩阵，执行 Purged 扩展窗口逐年滚动重训，产出样本外多模型预测评分矩阵：

```bash
python scripts/run_ml.py \
  --config configs/futures/ml_default.yaml \
  --start-year 2018 \
  --end-year 2026 \
  --train-window expanding \
  --purge-window 5
```

### 4. 期货实盘向量化回测 (`run_backtest.py`)

使用单因子表达式或机器学习产出的 `pred_panel.parquet` 执行期货实盘撮合状态机回测：

```bash
# 单因子表达式回测
python scripts/run_backtest.py \
  --factor "ts_zscore(close, 20)" \
  --config configs/futures/backtest.yaml \
  --fees configs/futures/fees.yaml

# 机器学习预测宽表回测
python scripts/run_backtest.py \
  --pred runs/ml_run_v1/predictions.parquet \
  --config configs/futures/backtest.yaml \
  --fees configs/futures/fees.yaml
```

### 5. 现代化 6 阶段量化因子研发 Web 工作台 (`run_web.py`)

```bash
# 启动零依赖量化研发 Web 单页工作台
python scripts/run_web.py --port 8080
```

### 6. 全资产代码质量认证门禁 (`verify.sh`)

所有开发者在提交代码前**必须**运行的质量认证脚本，提供多种执行模式：

```bash
bash scripts/verify.sh --all        # 全量六大门禁检验 (推荐提交前执行)
bash scripts/verify.sh --fast       # 快速日常静态校验 (格式、类型与配置)
bash scripts/verify.sh --rust       # 仅运行 Rust 内核 Clippy, Fmt 与单元测试
bash scripts/verify.sh --python     # 仅运行 Python Ruff, Pytest 与因果摄动审计
bash scripts/verify.sh --git-only   # 仅检查 Git 暂存区数据安全防泄漏
```

### 7. A 股股票因子回测与跨平台对账脚本

基于 `data/raw/stocks/`（全市场 5,323 只 A 股，`2015-11-02 ~ 2024-12-31`），平台提供统一走 `apply_postprocess_pipeline` 无渗漏后处理的股票评测与 FactorLab 对账脚本：

```bash
# 下载/同步米筐 A 股日线与元数据至 data/raw/stocks/
python scripts/download_rq_stocks.py --help

# 单因子 Volume_Price_2_H5_2 全周期回测与 FactorLab 对账 (open_adj / vwap_adj 双口径)
python scripts/backtest_volume_price_factor.py

# 开启方向性涨跌停撮合模拟 (T+1 涨停不可买、T+1+h 跌停/停牌自动顺延卖出)
python scripts/backtest_volume_price_factor.py --execution-simulation

# 5 大经典因子多维度本地回测对比 (支持 --tech-trend-high-low adj|raw 与 --execution-simulation)
python scripts/backtest_multi_factors.py --execution-simulation
```

---

## Python SDK 核心编程范式

在 Jupyter Notebook、研究脚本或生产服务中，可直接调用 `factorlib` 与 `factorlib_core`：

### 1. 面板容器操作 (`factorlib.core.panel`)

```python
from factorlib.core.panel import Panel

# 从派生目录加载标准化面板
panel = Panel.load("data/derived/futures/v1")
print(f"面板规模: {panel.T} 交易日 x {panel.N} 品种")
print(f"覆盖特征: {list(panel.fields.keys())[:5]}")

# 提取特征矩阵 (均为 Fortran 列优先 (T, N) 连续内存数组)
close = panel.get_field("close")
volume = panel.get_field("volume")
mask = panel.mask  # 可交易品种有效掩码 (T, N) bool

# 按日期区间切片
sub_panel = panel.slice_dates("2024-01-01", "2024-03-31")
```

### 2. DSL 表达式构建与静态检查 (`factorlib.core.dsl`)

系统提供链式语法糖 `Expr` 与终端字段构造器 `F`，支持原生 Python 算术运算符重载：

```python
from factorlib.core.dsl import F, Expr, canonicalize_expr, validate_expr

# 1. 链式构建中缀语法糖
alpha = (F("close") - F("open")).ts_mean(20).cs_rank()
print(str(alpha))  # 输出: 'cs_rank(ts_mean(sub(close, open), 20))'

# 2. 静态类型检查与防未来函数拦截
check = validate_expr(str(alpha), available_fields=list(panel.fields.keys()))
print("语法校验通过:", check["valid"])

# 非法未来函数调用将被强力拦截
invalid_check = validate_expr("ts_delay(close, -5)")
assert not invalid_check["valid"], "成功阻断未来前向移位算子！"

# 3. AST 代数规范化与唯一哈希指纹
canon_info = alpha.canonicalize(available_fields=list(panel.fields.keys()))
print(f"规范化表达式: {canon_info.canon_expr}")
print(f"代数指纹哈希: {canon_info.canon_hash}")
```

### 3. Rust 极速计算内核交互 (`factorlib_core.Engine`)

```python
# 将 Panel 数据直通注入 Rust 原生引擎 (零数据拷贝，Fortran 内存直连)
engine = panel.to_rust_engine(cache_mb=128, n_threads=4)

# 单表达式求值 (毫秒级)
mat = engine.eval("ts_zscore(close, 20)")

# 多表达式批并行求值 (充分利用 CPU 全部多核并发)
batch_exprs = [
    "ts_mean(close, 5)",
    "ts_mean(close, 20)",
    "pct_change(close, 10)",
    "cs_rank(ts_std(close, 20))",
]
results = engine.eval_batch(batch_exprs)
```

### 4. SQLite 因子全生命周期注册库 (`factorlib.core.registry`)

```python
from factorlib.core.registry import FactorRecord, FactorRegistry

registry = FactorRegistry("data/registry/factors.sqlite")

# 查询库内已准入因子列表
factors = registry.list_factors(status="accepted")
for f in factors:
    print(f"ID {f.id}: [{f.status}] {f.canon_expr} (方向: {f.sign}, 来源: {f.source})")

# 更新因子生命周期状态
# candidate -> accepted -> active -> deprecated
if factors:
    registry.update_factor_status(factors[0].id, "active")
```

### 5. 因子综合评估体系 (`factorlib.core.evaluate`)

```python
from factorlib.core.evaluate import evaluate_factor

# 针对单因子执行多时段切分与多预测周期 (h=1,3,5,10,20) 全维度评估
eval_res = evaluate_factor(panel, "pct_change(close, 20)")

print("推荐方向 (Sign):", eval_res.sign)
for split_name, h_dict in eval_res.metrics_by_split.items():
    if 5 in h_dict:
        m = h_dict[5]
        print(f"[{split_name}] h=5: RankIC={m.ic_mean:.4f}, ICIR={m.ic_ir:.2f}, 多空年化={m.ls_annual_ret:.2%}")

# 直接转化为因子注册表记录落盘
record = eval_res.to_factor_record(source="manual", status="candidate")
registry.register_factor(record, run_id="my_manual_test")
```

### 6. 组合优化与实盘回测状态机 (`factorlib.core.portfolio` / `backtest`)

```python
import yaml
from factorlib.core.backtest import run_backtest
from factorlib.core.portfolio import build_portfolio_weights
from factorlib.core.report import generate_html_report
from factorlib.markets.futures.adapter import FuturesMarketAdapter

# 1. 加载配置与市场适配器
with open("configs/futures/backtest.yaml", encoding="utf-8") as f:
    bt_cfg = yaml.safe_load(f)
adapter = FuturesMarketAdapter(fees_path="configs/futures/fees.yaml")

# 2. 组合优化: 波动率倒数平价、收缩协方差与调仓缓冲带
scores = engine.eval("ts_zscore(close, 20)")
target_weights = build_portfolio_weights(
    scores=scores,
    returns=panel.fields["close"],
    mask=panel.mask,
    config=bt_cfg["portfolio"],
)

# 3. 执行 Rust 期货实盘语义向量化回测
result = run_backtest(
    panel=panel,
    weights=target_weights,
    config=bt_cfg,
    adapter=adapter,
)

# 4. 打印财务统计并导出交互式 HTML 诊断报告
print(f"年化收益: {result.metrics['annual_return']:.2%}")
print(f"夏普比率: {result.metrics['sharpe']:.2f}")
print(f"最大回撤: {result.metrics['max_drawdown']:.2%}")

generate_html_report(result, output_path="runs/my_report.html", title="单因子回测报告")
```

---

## 68 项高性能算子速查表

Rust 内核原生注册全部 68 项向量化与时序算子，全面覆盖技术分析、微观结构统计与横截面中性化需求：

| 算子类别 | 数量 | 算子名称与语法签名 | 算法特性与说明 |
|---|:---:|---|---|
| **逐元素一元算子** | 12 | `neg(x)`, `abs(x)`, `sign(x)`, `inv(x)`, `slog(x)`, `ssqrt(x)`, `square(x)`, `cube(x)`, `sigmoid(x)`, `tanh(x)`, `relu(x)`, `atan(x)` | 逐点向量化，安全除零与负数平滑净化，无崩溃隐患。 |
| **逐元素二元算子** | 8 | `add(x, y)`, `sub(x, y)`, `mul(x, y)`, `div(x, y)`, `max2(x, y)`, `min2(x, y)`, `hypot(x, y)`, `copysign(x, y)` | 矩阵广播计算，自动遵循 IEEE 754 NaN 传播准则。 |
| **标量常数算子** | 6 | `addc(x, c)`, `subc(x, c)`, `mulc(x, c)`, `divc(x, c)`, `spow(x, c)`, `clip(x, c)` | 与浮点标量 $c$ 逐元素运算；`spow` 自动保留符号。 |
| **时序滚动算子** | 25 | `delay(x, d)`, `delta(x, d)`, `pct_change(x, d)`, `log_return(x, d)` | 时序平移与一阶差分，$O(1)$ 环形缓冲区极速推进。 |
| | | `ts_mean(x, d)`, `ts_sum(x, d)`, `ts_var(x, d)`, `ts_std(x, d)` | 基于双栈增量 **Welford $O(1)$ 算法**，杜绝浮点精度累积漂移。 |
| | | `ts_skew(x, d)`, `ts_kurt(x, d)` | 三阶偏度与四阶峰度在线增量累加器。 |
| | | `ts_min(x, d)`, `ts_max(x, d)`, `ts_argmax(x, d)`, `ts_argmin(x, d)` | 基于**单调双端队列 (Monotonic Deque)** 实现均摊 $O(1)$ 极值搜索。 |
| | | `ts_rank(x, d)`, `ts_zscore(x, d)`, `ts_mad(x, d)` | 时序百分比排名、时序滚动 Z-Score 与中位数绝对偏差。 |
| | | `ts_decay_linear(x, d)`, `ts_ema(x, d)` | 线性衰减权重平滑与指数移动平均。 |
| | | `ts_corr(x, y, d)`, `ts_cov(x, y, d)`, `ts_beta(x, y, d)` | 增量双变量 Welford $O(1)$ 协方差、相关系数与资产 Beta。 |
| | | `ts_wmean(x, w, d)`, `ts_sharpe(x, d)` | 加权滚动均值与时序年化夏普比率。 |
| **横截面算子** | 9 | `cs_rank(x)` | 截面百分比排名（排序归一化至 $[0, 1]$）。 |
| | | `cs_zscore(x)`, `cs_scale(x)`, `cs_demean(x)` | 截面标准化、绝对值和归一化与截面去均值。 |
| | | `cs_winsorize(x)`, `cs_mad(x)` | 截面 MAD 稳健去极值截断与截面中位数绝对偏差。 |
| | | `cs_quantile(x, q)` | 截面分位数切分。 |
| | | `cs_neutralize(x, g)` | 截面正交化中性化投影。 |
| **行业/板块算子** | 3 | `group_rank(x, g)`, `group_demean(x, g)`, `group_zscore(x, g)` | 按板块或申万行业分组标识 $g$ 执行组内中性化。 |
| **逻辑与条件算子** | 5 | `gt(x, y)`, `lt(x, y)`, `gt_const(x, c)`, `lt_const(x, c)`, `if_else(cond, x, y)` | 矩阵布尔掩码比较与三元条件分支选择。 |

---

## 严密防未来函数与交易语义规范

平台在量化研究全链路设立严格的因果隔离机制：

1. **时序因果对齐**：
   - 统一使用米筐 `trading_date`；
   - $T$ 日所有因子值、复权因子、截面标签与主力合约映射只能读取截至 $T$ 日 15:00 收盘已知的数据；
   - 静态类型检查器阻断任何前向负数位移（如 `delay(x, -1)`）及未来标签。
2. **真实实盘撮合成交机制**：
   - 严禁使用 $T$ 日收盘价直接作为成交价（收盘价已包含因果信息）；
   - 执行撮合统一使用 $T+1$ 开盘前 30 分钟基于 1m bar 计算的真实加权平均价：
     $$\text{VWAP}_{T+1} = \frac{\sum_{t=1}^{30} (\text{close}_t \times \text{volume}_t)}{\sum_{t=1}^{30} \text{volume}_t}$$
   - 严禁使用 `turnover / volume`，彻底规避郑商所历史成交额置零除零隐患。
3. **换月展期状态机与双边摩擦**：
   - 遵循 `dominant_mapping`（`rule=3` 避开交割月流动性枯竭）；
   - 换月日旧合约平仓与新合约开仓严格独立记账，按各自真实成交价双边计收手续费与滑点，严禁跨合约价格直接作差。
4. **机器学习时序隔离 (Purged Walk-Forward)**：
   - 训练集与验证集末尾精确剔除 $h$ 天标签重叠区间（Purge Window = $h$），消除标签重叠带来的样本穿透；
   - 验证集与测试集前端设置静默隔离期（Embargo Window）。
5. **逐日盯市 (Mark-to-Market) 守恒定律**：
   - 每日收盘记账严格满足资金守恒恒等式：
     $$\text{Cash}_d + \sum_{i=1}^N (\text{lots}_{i,d} \times \text{settle}_{i,d} \times \text{Multiplier}_i) == \text{Equity}_d$$
     $$\text{Equity}_d - \text{Equity}_{d-1} == \text{NetPnL}_d$$

---

## 配置文件索引与规范

所有生产级配置文件均存放于 `configs/futures/` 目录：

| 配置文件 | 核心配置项与业务语义 |
|---|---|
| `data.yaml` | 原始米筐数据路径 (`exports/`)、派生面板路径、主力映射规则 (`rule=3, rank=1`)、复权方法与开盘 30 分钟 VWAP 撮合窗口。 |
| `universe.yaml` | 品种池准入规则：上市天数门槛（$\ge 60$ 天）、流动性门槛（20日日均成交额 $\ge 5000$ 万）、距离交割日最小天数（$\ge 5$ 天）。 |
| `fees.yaml` | 全期货品种合约规格：合约乘数、最小变动价位（tick_size）、开仓/平仓/平今手续费率与保证金率。 |
| `gp_default.yaml` | 遗传规划挖掘超参：种群规模 (1000)、演化代数 (50)、岛屿数 (4)、迁移间隔、树深与节点上限、准入阈值。 |
| `ml_default.yaml` | Walk-Forward 机器学习超参：滚动年份范围 (2018~2026)、扩展窗口模式、Purge 隔离天数、LightGBM 树参数。 |
| `backtest.yaml` | 回测资金与风控：初始本金 (1000 万)、目标年化波动率 (10%)、最大杠杆上限 (300%)、调仓缓冲带 (1%)、保证金安全乘数 (1.1)。 |

---

## 代码库目录导航

```text
FactorLibrary/
├── python/factorlib/                      # 主 Python 算法包
│   ├── core/                              # 面板容器、DSL 语法糖、评估体系、组合优化、回测状态机、报告、因子库 ORM
│   ├── gp/                                # 强类型遗传规划自动化因子挖掘系统 (原语、变异、多岛拓扑、准入漏斗)
│   ├── ml/                                # Purged Walk-Forward 机器学习多因子合成 (预处理、切分、LGBM、Ridge)
│   └── markets/                           # 市场适配层
│       ├── futures/                       # 中国商品与股指期货市场适配器、品种池与 ETL 清洗流
│       └── stocks/                        # A 股股票市场适配器与面板加载器 (StockMarketAdapter, StockPanelLoader)
├── rust/factorlib_core/                   # Rust 极速计算内核
│   ├── src/ops/                           # 68 项向量化、逐元素、时序增量 Welford 与单调队列算子
│   ├── src/expr/                          # AST 语法树、词法语法分析、静态类型检查、代数规范化、LRU 缓存
│   ├── src/metrics/                       # RankIC、分位数收益、换手率、相关性矩阵计算
│   └── src/backtest/                      # 期货实盘状态机 (离散手数、涨跌停、换月成本、逐日盯市)
├── configs/futures/                       # 生产级 YAML 配置文件 (数据、品种池、费率、GP、ML、回测)
├── scripts/                               # 生产维护与执行脚本 (verify.sh, audit, build_panel, gp, ml, backtest, 股票对账)
├── tests/python/                          # Python 自动化单元与集成测试套件 (139 项测试 + 2 项因果摄动测试)
├── docs/                                  # 技术文档中心
│   ├── futures_platform_plan.md           # 期货与 A 股量化平台技术白皮书 (技术基石与理论依据)
│   ├── code_audit_and_refactor_plan.md    # 架构与代码级深度审查报告与闭环验收记录
│   └── api_reference.md                   # 平台全模块 API 参考手册
├── exports/                               # 只读米筐原始 Parquet 数据资产 (已 gitignore)
├── data/                                  # 派生研究面板与本地 SQLite 因子库 (已 gitignore)
├── runs/                                  # 实验产出目录 (指标汇总、流水账单、HTML 诊断报告，已 gitignore)
├── AGENTS.md                              # 智能体协同开发工程规范与质量门禁红线
└── pyproject.toml                         # 项目工程与依赖配置
```

---

## A 股股票市场适配与全市场回测对账 (Phase 2 已可用)

针对 A 股股票市场研究与 FactorLab 跨平台对账，系统已完整实现规范适配、全市场面板加载与零渗漏统一后处理评测链路：
1. **市场适配器 (`factorlib.markets.stocks.adapter.StockMarketAdapter`)**：1 手 = 100 股，乘数恒为 1.0，最小变动价位 0.01 元，无杠杆（保证金率 1.0），非对称印花税（仅卖出征收）及板块涨跌停识别（主板 $\pm 10\%$、科创板/创业板 $\pm 20\%$、北交所 $\pm 30\%$、ST $\pm 5\%$）；
2. **股票全市场面板加载器 (`factorlib.markets.stocks.panel_loader.StockPanelLoader`)**：
   - 将 `data/raw/stocks/`（5,323 只全市场股票）重塑为 Fortran 列优先 `StockPanelData`，并通过 `to_panel()` 无缝暴露 `listed_days` / `sector` 零拷贝别名以复用 `apply_postprocess_pipeline`；
   - 预计算方向性撮合约束 `can_buy_mask`（仅拦截开仓日停牌/涨停）与 `can_sell_mask`（仅拦截平仓日停牌/跌停）；
3. **方向性涨跌停撮合与递延平仓**：调用 `compute_forward_returns(execution_simulation=True)` 时，对 $t+1$ 不可买入信号执行遮罩，并在 $t+1+h$ 遇跌停/停牌时通过向量化后向扫描自动顺延至下一可卖出日成交，不再误杀涨停持仓赢家；
4. **跨平台对账验证**：`scripts/backtest_volume_price_factor.py` 与 `scripts/backtest_multi_factors.py` 支持 `open_adj` / `vwap_adj` 双口径与 `--execution-simulation` 切换，已完成与 FactorLab `Volume_Price_2_H5_2` (`bt_bb3a04da6b20456ab444dcb7e5450949`) 全周期（`2016-01-04 ~ 2024-12-31`）对账验证。

---

## 许可证与贡献规范

- 本项目采用 MIT 许可证；
- 代码提交前**必须**运行并通过门禁：`bash scripts/verify.sh --all`；
- Git Commit 提交信息必须遵循中文 Angular 规范：`<type>(<scope>): <subject>`（例如 `feat(factor): 添加截面 RankIC 计算算子`）。
