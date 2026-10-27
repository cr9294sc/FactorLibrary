# Barra 风格与风险因子 (Barra Risk Factors)

包含 Barra 中国股票风险模型 CNE5（11 个风格因子）与 CNTR（20 个风格因子）的核心风险暴露矩阵，常用于多因子中性化与风险对齐。

> **总计收录**：`31` 个变量

---

## Barra CNE5 风格因子 (11 个) (11 个)

| 变量名称 (Short Name) | 底层标识 (Actual Name) | 频度 | 维度 (Axes) | 数据类型 (Kind) | 缺失值处理策略 |
| :--- | :--- | :---: | :---: | :---: | :--- |
| `barra_cne5_beta` | `1d.barra_cne5.BETA` | 1d | date × code | daily | 缺失日期和证券读取为 NaN；上游生产口径尚未核实。 |
| `barra_cne5_btop` | `1d.barra_cne5.BTOP` | 1d | date × code | daily | 缺失日期和证券读取为 NaN；上游生产口径尚未核实。 |
| `barra_cne5_earnyild` | `1d.barra_cne5.EARNYILD` | 1d | date × code | daily | 缺失日期和证券读取为 NaN；上游生产口径尚未核实。 |
| `barra_cne5_growth` | `1d.barra_cne5.GROWTH` | 1d | date × code | daily | 缺失日期和证券读取为 NaN；上游生产口径尚未核实。 |
| `barra_cne5_industry` | `1d.barra_cne5.INDUSTRY` | 1d | date × code | daily | 缺失日期和证券读取为 NaN；上游生产口径尚未核实。 |
| `barra_cne5_leverage` | `1d.barra_cne5.LEVERAGE` | 1d | date × code | daily | 缺失日期和证券读取为 NaN；上游生产口径尚未核实。 |
| `barra_cne5_liquidty` | `1d.barra_cne5.LIQUIDTY` | 1d | date × code | daily | 缺失日期和证券读取为 NaN；上游生产口径尚未核实。 |
| `barra_cne5_momentum` | `1d.barra_cne5.MOMENTUM` | 1d | date × code | daily | 缺失日期和证券读取为 NaN；上游生产口径尚未核实。 |
| `barra_cne5_resvol` | `1d.barra_cne5.RESVOL` | 1d | date × code | daily | 缺失日期和证券读取为 NaN；上游生产口径尚未核实。 |
| `barra_cne5_size` | `1d.barra_cne5.SIZE` | 1d | date × code | daily | 缺失日期和证券读取为 NaN；上游生产口径尚未核实。 |
| `barra_cne5_sizenl` | `1d.barra_cne5.SIZENL` | 1d | date × code | daily | 缺失日期和证券读取为 NaN；上游生产口径尚未核实。 |

## Barra CNTR 风格因子 (20 个) (20 个)

| 变量名称 (Short Name) | 底层标识 (Actual Name) | 频度 | 维度 (Axes) | 数据类型 (Kind) | 缺失值处理策略 |
| :--- | :--- | :---: | :---: | :---: | :--- |
| `barra_cntr_analystsentiment` | `1d.barra_cntr.analystsentiment` | 1d | date × code | daily | 缺失日期和证券读取为 NaN；上游生产口径尚未核实。 |
| `barra_cntr_beta` | `1d.barra_cntr.beta` | 1d | date × code | daily | 缺失日期和证券读取为 NaN；上游生产口径尚未核实。 |
| `barra_cntr_booktoprice` | `1d.barra_cntr.booktoprice` | 1d | date × code | daily | 缺失日期和证券读取为 NaN；上游生产口径尚未核实。 |
| `barra_cntr_dividendyield` | `1d.barra_cntr.dividendyield` | 1d | date × code | daily | 缺失日期和证券读取为 NaN；上游生产口径尚未核实。 |
| `barra_cntr_earningsquality` | `1d.barra_cntr.earningsquality` | 1d | date × code | daily | 缺失日期和证券读取为 NaN；上游生产口径尚未核实。 |
| `barra_cntr_earningsvariability` | `1d.barra_cntr.earningsvariability` | 1d | date × code | daily | 缺失日期和证券读取为 NaN；上游生产口径尚未核实。 |
| `barra_cntr_earningsyield` | `1d.barra_cntr.earningsyield` | 1d | date × code | daily | 缺失日期和证券读取为 NaN；上游生产口径尚未核实。 |
| `barra_cntr_growth` | `1d.barra_cntr.growth` | 1d | date × code | daily | 缺失日期和证券读取为 NaN；上游生产口径尚未核实。 |
| `barra_cntr_industrymomentum` | `1d.barra_cntr.industrymomentum` | 1d | date × code | daily | 缺失日期和证券读取为 NaN；上游生产口径尚未核实。 |
| `barra_cntr_investmentquality` | `1d.barra_cntr.investmentquality` | 1d | date × code | daily | 缺失日期和证券读取为 NaN；上游生产口径尚未核实。 |
| `barra_cntr_leverage` | `1d.barra_cntr.leverage` | 1d | date × code | daily | 缺失日期和证券读取为 NaN；上游生产口径尚未核实。 |
| `barra_cntr_liquidity` | `1d.barra_cntr.liquidity` | 1d | date × code | daily | 缺失日期和证券读取为 NaN；上游生产口径尚未核实。 |
| `barra_cntr_longtermreversal` | `1d.barra_cntr.longtermreversal` | 1d | date × code | daily | 缺失日期和证券读取为 NaN；上游生产口径尚未核实。 |
| `barra_cntr_midcap` | `1d.barra_cntr.midcap` | 1d | date × code | daily | 缺失日期和证券读取为 NaN；上游生产口径尚未核实。 |
| `barra_cntr_momentum` | `1d.barra_cntr.momentum` | 1d | date × code | daily | 缺失日期和证券读取为 NaN；上游生产口径尚未核实。 |
| `barra_cntr_profitability` | `1d.barra_cntr.profitability` | 1d | date × code | daily | 缺失日期和证券读取为 NaN；上游生产口径尚未核实。 |
| `barra_cntr_residualvolatility` | `1d.barra_cntr.residualvolatility` | 1d | date × code | daily | 缺失日期和证券读取为 NaN；上游生产口径尚未核实。 |
| `barra_cntr_season` | `1d.barra_cntr.season` | 1d | date × code | daily | 缺失日期和证券读取为 NaN；上游生产口径尚未核实。 |
| `barra_cntr_shorttermreversal` | `1d.barra_cntr.shorttermreversal` | 1d | date × code | daily | 缺失日期和证券读取为 NaN；上游生产口径尚未核实。 |
| `barra_cntr_size` | `1d.barra_cntr.size` | 1d | date × code | daily | 缺失日期和证券读取为 NaN；上游生产口径尚未核实。 |
