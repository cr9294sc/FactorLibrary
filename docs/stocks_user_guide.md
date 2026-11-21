# FactorLibrary A 股量化全流程用户使用手册

> **适用范围**：A 股全市场（`data/derived/stocks/{rq, barra}`）研究、因子编写与探针校验、GP/ML/DL 多因子、本地 **MOSEK** Barra 组合优化与 Rust 实盘级回测全闭环。  
> **推荐环境**：所有命令均在仓库根目录使用 `.venv/bin/python`（或已激活的 Python 3.12 虚拟环境）运行。

---

## 0. 全局六阶闭环总览

FactorLibrary A 股链路严格遵循 **「投资研究 → 交易执行 → 绩效监控」** 六阶闭环，并做到**零手动切表、零时钟错配**：

```text
  [1. Data: rq & barra]
   ├─ 特征域 `rq`   (2,677 交易日 x 5,478 标的, 50+ 量价/估值/指数权重字段)
   │    └─ 加载时自动通过 RiskView 挂载 `barra` 的 vwap_open_30m 标签 (y_h, y_idio_h) 与 10 风格暴露
   └─ 执行域 `barra` (2,628 交易日 x 5,642 标的, VwapBegin30 + CNE5 + 42x42 COV risk_store)
          │
          ▼
  [2. Factors] ── DSL 表达式 / Python compute(context) + 12 步后处理 + probe_factor 探针
          │       或 GP 多岛演化自动挖掘 (scripts/run_gp.py)
          ▼
  [3. Predictions] ── Purged Walk-Forward ML (scripts/run_ml.py) / DL (scripts/run_dl.py) / 外部信号 Parquet
          │
          ▼  (SignalBridge 自动完成 6位代码归一 + T_close/T_exec 对齐)
  [4. Positions] ── SignalShaper → DynamicRiskGuard → BarraPortfolioOptimizer (本地 MOSEK 求解)
          │
          ▼
  [5. Orders] ── Rust 股票撮合状态机 (vwap_open_30m_adj, T+1, 100股整手, 停牌/涨跌停阻断, 双费率)
          │
          ▼
  [6. Returns] ── report.html + metrics.json (v4_stats + bridge_audit) + daily_equity.csv
```

---

## 1. Data：数据目录与面板构建

### 1.1 标准目录结构

```text
data/
├── raw/stocks/
│   ├── rq/                  # 米筐原始 Parquet (bar_1d, daily_basic, industry_sw, trade_status 等)
│   └── barra/               # Barra 原始资产 (Cache/AshareCache/*.ares, BarraCNE5_{RET,COV}, benchmark_000905_SH.csv, shifted_signal.parquet)
└── derived/stocks/
    ├── rq/                  # 米筐派生研究面板 (panel_daily.parquet, labels.parquet, meta.json, fields.json)
    └── barra/               # Barra 派生执行面板 (+ risk_store/{factor_cov.npz, factor_returns.parquet, store.json})
configs/stocks/
├── data.yaml                # 声明 rq 与 barra 两个数据集（逻辑名 == 目录名）
├── barra.yaml               # Barra CNE5 风险模型与动态特质方差参数
├── backtest.yaml            # 组合优化、SignalBridge、执行与作业预设 (default | quantile | factorsim_match)
├── fees.yaml                # 双费率模式 (asymmetric | flat)
└── mosek.lic                # 本地 MOSEK 许可证（已由 .gitignore 忽略）
```

### 1.2 数据更新与面板重建（仅在拉取新行情后执行）

```bash
# 1. 增量同步米筐 A 股原始数据到 data/raw/stocks/rq/
.venv/bin/python scripts/download_rq_stocks.py --end-date 2026-11-10

# 2. 构建/更新米筐派生面板 data/derived/stocks/rq/
.venv/bin/python scripts/build_panel.py --market stocks --version rq

# 3. 构建/更新 Barra 派生面板与 risk_store -> data/derived/stocks/barra/
.venv/bin/python scripts/build_stocks_lb_barra.py
```

---

## 2. Factors：编写自定义因子与静态探针检查

平台支持 **DSL 表达式** 与 **因果 Python 函数** 双模态编写，统一经 12 步零渗漏后处理管线与 `probe_factor` 静态探针校验。

### 2.1 方式 A：直接传入 DSL 表达式（最快）

支持 Rust 内核全部 **106 个 FactorLab 1d 算子**（128 个注册项含别名）及标准变量短名（如 `daily_derived_returns_c2c_1d` 自动归一为 `c2c_ret`、`barra_cne5_size` 归一为 `barra_size`）：

```bash
# 先做毫秒级静态探针校验（检查语法、未来函数、依赖字段与预热窗口）
.venv/bin/python scripts/run_stocks_backtest.py \
  --factor "neg(add(cs_zscore(ts_corr(close_hfq, log_amount, 20)), cs_zscore(ts_std(c2c_ret, 20))))" \
  --postprocess "baseUniMask,trdMask,dropNewStock:120,MAD:3,zScore,indNeut:sector,decay:5,rank" \
  --probe-only
```

常用 `--postprocess` 预设与参数说明：
- `standard`：`baseUniMask,trdMask,MAD:3,indNeut:sector,zScore`
- `minimal`：`baseUniMask,trdMask,zScore`
- `style_neutral`：包含行业与风格中性化
- 支持自定义逗号串，例如 `"baseUniMask,trdMask,dropNewStock:120,MAD:3,indNeut:sector,riskNeutMulti:SIZE,RESVOL,decay:5,rank"`。

### 2.2 方式 B：编写因子定义文件（`.yaml` / `.json` / `.py`）

#### YAML 表达式定义示例（`my_rev_factor.yaml`）

```yaml
name: short_rev_5d
factor_type: expression
expression: "-ts_sum(c2c_ret, 5)"
postprocess:
  - op: baseUniMask
  - op: trdMask
  - op: dropNewStock
    window: 120
  - op: MAD
    param: 3.0
  - op: indNeut
    group_by: sector
  - op: decay
    window: 5
  - op: zScore
```

#### Python 因果函数定义示例（`my_py_factor.py`）

> 因果沙箱规范：仅通过 `context["field"]` 读取 `(T, N)` 只读矩阵，严禁使用前向负位移或读取 `labels` / `aux_fields`。

```python
import numpy as np


def compute(context):
    close = context["close_hfq"]
    amount = context["amount"]
    ret = context["c2c_ret"]

    # 示例：20 日换手金额波动倒数 × 短期反转
    prev_close = np.vstack([np.full((1, close.shape[1]), np.nan), close[:-1, :]])
    daily_chg = (close - prev_close) / np.where(prev_close > 0, prev_close, np.nan)
    vol_proxy = np.sqrt(np.maximum(amount, 1.0))
    return -ret / vol_proxy - 0.1 * daily_chg
```

校验并直接回测文件因子：

```bash
.venv/bin/python scripts/run_stocks_backtest.py --factor-file my_rev_factor.yaml --probe-only
.venv/bin/python scripts/run_stocks_backtest.py --factor-file my_py_factor.py --postprocess standard --mode quantile
```

### 2.3 自动化显式挖掘（STGP）

```bash
.venv/bin/python scripts/run_gp.py \
  --market stocks \
  --config configs/stocks/gp_default.yaml \
  --population 200 --generations 10 --islands 4 --seed 42
```

---

## 3. Predictions：ML / DL 多因子合成与外部信号

### 3.1 Purged Walk-Forward 机器学习合成

```bash
.venv/bin/python scripts/run_ml.py \
  --market stocks \
  --config configs/stocks/ml_default.yaml \
  --target-label y_idio_5 \
  --start-year 2023 --end-year 2026
# 输出样本外预测长表：runs/ml_YYYYMMDD_HHMMSS/predictions.parquet
```

### 3.2 PyTorch 1d DL Alpha 合成

```bash
.venv/bin/python scripts/run_dl.py \
  --market stocks \
  --arch st_attention \
  --target-label y_idio_5 \
  --seq-len 20 --epochs 18 --n-folds 4 --n-seeds 3
# 输出样本外预测长表：runs/dl_YYYYMMDD_HHMMSS/predictions.parquet
```

### 3.3 外部预合成信号宽表

直接传入宽表 Parquet（索引/首列为日期、列名为 6 位或带交易所后缀股票代码，如 `data/raw/stocks/barra/combo/signal/signal_data/shifted_signal.parquet`），[`SignalBridge`](file:///home/cr9294/repos/FactorLibrary/python/factorlib/markets/stocks/bridge.py#L139) 会自动处理代码归一与 `T_close` / `T_exec` 对齐。

---

## 4. Positions：Barra 组合优化与本地 MOSEK 求解全流程

平台通过 [`BarraPortfolioOptimizer`](file:///home/cr9294/repos/FactorLibrary/python/factorlib/markets/stocks/optimizer.py) 实现 **SignalShaper → DynamicRiskGuard → ConvexOptimizer** 三层融合求解。

### 4.1 确认本地 MOSEK 许可证就绪

1. 许可证文件位于 [`configs/stocks/mosek.lic`](file:///home/cr9294/repos/FactorLibrary/configs/stocks/backtest.yaml#L53)（已在 `.gitignore` 中忽略）。
2. [`configs/stocks/backtest.yaml`](file:///home/cr9294/repos/FactorLibrary/configs/stocks/backtest.yaml#L51-L57) 默认求解器优先级：
   ```yaml
   solver:
     priority: ["MOSEK", "CLARABEL", "OSQP"]
     mosek_license_file: "./configs/stocks/mosek.lic"
     mosek_num_threads: 1
     allow_slack_fallback: true
     slack_penalty: 1000.0
   ```
3. 可通过 5 行 Python 快速验证 `MOSEK` 单日求解状态与约束满足度：

```bash
.venv/bin/python - << 'EOF'
from pathlib import Path
import numpy as np
from factorlib.core.config import load_stock_backtest_config
from factorlib.markets.profile import PanelSpec
from factorlib.markets.stocks.profile import StocksProfile

prof = StocksProfile()
bt_cfg = load_stock_backtest_config(Path("configs/stocks/backtest.yaml"))
opt = prof.build_barra_optimizer(bt_cfg.portfolio)
barra = prof.load_panel(PanelSpec(version="barra"))
t = barra.T - 1
res = opt.solve_one_day(
    barra.fields["c2c_ret"][t],
    exposures=np.stack([barra.fields[f"barra_{s.lower()}"][t] for s in opt.risk_store.style_factors], axis=1),
    industry=barra.fields["barra_industry"][t],
    mask_active=barra.mask[t],
    mask_suspend=(barra.aux_fields["is_suspended"][t] > 0.5),
)
print("solver:", res.solver_used, "status:", res.status, "n_pos:", int(np.sum(res.weights > 1e-6)), "max_w:", float(np.max(res.weights)))
EOF
```

### 4.2 优化核心参数说明（`configs/stocks/backtest.yaml`）

- **`portfolio.optimizer`（硬约束与风险模型）**：
  - `objective.risk_aversion` / `turnover_penalty` / `include_specific_risk`：日频风险厌恶与换手惩罚；
  - `constraints.max_single_weight` / `max_active_weight` / `max_daily_turnover`：单票、偏离度与换手硬约束；
  - `constraints.style_deviation_default` / `style_deviation_overrides`（如 `SIZE: 0.15`, `BETA: 0.20`）与 `industry_deviation: 0.02`：Barra 10 风格与 31 行业偏离控制；
  - `benchmark_weight_mode`：`mkt_cap_proxy`（如 `[301, 800]` 模拟中证 500）、`constituent` 或 `absolute_neutral`。
- **`strategy`（信号塑形与风控守卫）**：
  - `power`（头部幂次放大）、`ind_neutral_weight`（软行业中性）、`vol_weight`（低波 `RESVOL` 惩罚）、`memory`（EMA 记忆）；
  - `adaptive_stress_guard`（监测 `FactorRet['SIZE']` 10 日异动并在微盘回撤体制下收紧敞口）；
  - `mode: horizon`（双期跨期规划，`persistence` / `discount`）、`asymmetric_trade_penalty`（买 `1.00` / 卖 `0.45`）与 `buy_hurdle`。

---

## 5. Orders & Returns：统一运行 A 股回测

全部 A 股回测统一使用独立入口 [`scripts/run_stocks_backtest.py`](file:///home/cr9294/repos/FactorLibrary/scripts/run_stocks_backtest.py)，执行域固定在 `barra`（`vwap_open_30m`），支持 CLI 参数覆盖，**无需反复修改 YAML**：

### 5.1 四大高频场景命令

```bash
# 场景 1：单因子 DSL + Barra QP (本地 MOSEK 求解 + A 股非对称税费)
.venv/bin/python scripts/run_stocks_backtest.py \
  --factor "-ts_sum(c2c_ret, 5)" \
  --postprocess standard \
  --profile default \
  --start-date 2024-01-01 --end-date 2026-11-10

# 场景 2：单因子快速分位等权基线 (仍走 SignalBridge -> barra vwap_open_30m)
.venv/bin/python scripts/run_stocks_backtest.py \
  --factor "-ts_sum(c2c_ret, 5)" \
  --postprocess standard \
  --profile quantile

# 场景 3：ML / DL 样本外预测长表回测
.venv/bin/python scripts/run_stocks_backtest.py \
  --pred runs/ml_20261121_120000/predictions.parquet \
  --profile default

# 场景 4：作业 3 / factorsim 对账预设 (自动加载 shifted_signal.parquet + 0.003 flat 费率 + 0.9 仓位 + T_exec)
.venv/bin/python scripts/run_stocks_backtest.py \
  --profile factorsim_match
  # 如需先跑小窗口冒烟，可追加: --start-date 2024-01-02 --end-date 2024-03-31
```

### 5.2 回测产物说明

每次回测默认输出至 `runs/stocks_bt_YYYYMMDD_HHMMSS/`（或 `--output-dir` 指定目录）：

| 文件 | 内容说明 |
|---|---|
| `report.html` | 独立交互式诊断报告（净值与回撤曲线、月度收益热力表、交易成本拆解瀑布图、v4 分层/IC 图表） |
| `metrics.json` | 组合绩效指标、桥接审计报告 (`bridge_audit`)、探针报告 (`probe_report`) 及 `v4_stats`（含 IC/RankIC/分组收益/10 维风格暴露） |
| `daily_equity.csv` | 逐日资金权益、收益率、现金、杠杆、毛/净盈亏、佣金、滑点、印花税、换手率、阻断单数及基准收益 |

---

## 6. 完整 Python SDK 端到端实例（写因子 → 探针 → 桥接 → MOSEK 优化 → 回测）

```python
from pathlib import Path
import numpy as np
from factorlib.core.config import load_stock_backtest_config, load_stock_fees_config
from factorlib.core.evaluate import evaluate_factor_v4_stats
from factorlib.core.probe import FactorDefinition, evaluate_factor_definition
from factorlib.core.report import generate_html_report
from factorlib.markets.profile import PanelSpec
from factorlib.markets.stocks.backtest import run_stock_backtest_pipeline
from factorlib.markets.stocks.bridge import SignalBridge
from factorlib.markets.stocks.profile import StocksProfile

# 1. Data: 加载 rq（自动挂载 barra 的 vwap_open_30m 标签与 10 维风格）与执行域 barra
prof = StocksProfile()
rq_panel = prof.load_panel(PanelSpec(version="rq"))
barra_panel = prof.load_panel(PanelSpec(version="barra"))

# 2. Factors: 编写因子 + 探针校验 + 后处理求值 + Layer 1 v4 统计评价
defn = FactorDefinition(
    name="vp_rev_combo",
    factor_type="expression",
    expression="neg(add(cs_zscore(ts_corr(close_hfq, log_amount, 20)), cs_zscore(ts_std(c2c_ret, 20))))",
    postprocess=["baseUniMask", "trdMask", "dropNewStock:120", "MAD:3", "zScore", "indNeut:sector", "decay:5", "rank"],
)
scores_rq, probe_rep = evaluate_factor_definition(defn, rq_panel)
v4_stats = evaluate_factor_v4_stats(scores_rq, rq_panel, profile=prof)

# 3. SignalBridge: 跨源单向投影至 barra (T_close 对齐)
scores_barra, bridge_rep = SignalBridge.from_panel(rq_panel, scores_rq, barra_panel, input_signal_alignment="T_close")
print("Bridge coverage:", bridge_rep.symbol_match_ratio, bridge_rep.non_null_coverage_ratio)

# 4. Positions: 本地 MOSEK + Barra 42 因子低秩 QP 组合优化（演示取最近 60 个交易日）
bt_cfg = load_stock_backtest_config(Path("configs/stocks/backtest.yaml"))
fees_cfg = load_stock_fees_config(Path("configs/stocks/fees.yaml"))
weights = prof.build_weights(
    scores_barra,
    np.zeros_like(scores_barra),
    barra_panel.mask,
    config=bt_cfg.portfolio,
    panel=barra_panel,
    reserve_cash=bt_cfg.execution.reserve_cash,
)

# 5. Orders & Returns: Rust 实盘撮合 + HTML 报告输出
res = run_stock_backtest_pipeline(barra_panel, weights, config=bt_cfg, fees=fees_cfg)
print(res.summary())
generate_html_report(res, Path("runs/sdk_demo/report.html"), v4_stats=v4_stats, probe_report=probe_rep)
```
