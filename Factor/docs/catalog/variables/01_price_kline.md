# 日线行情变量 (Price Kline)

包含 A 股市场基础日线量价指标，包括未复权及后复权（hfq）开高低收、成交量、成交额以及复权因子等。

> **总计收录**：`18` 个变量

---

| 变量名称 (Short Name) | 底层标识 (Actual Name) | 频度 | 维度 (Axes) | 数据类型 (Kind) | 缺失值处理策略 |
| :--- | :--- | :---: | :---: | :---: | :--- |
| `amount` | `1d.kline.amount` | 1d | date × code | daily | 缺失日期和证券读取为 NaN；上游生产口径尚未核实。 |
| `close` | `1d.kline.close` | 1d | date × code | daily | 缺失日期和证券读取为 NaN；上游生产口径尚未核实。 |
| `close_hfq` | `1d.kline.close_hfq` | 1d | date × code | daily | 缺失日期和证券读取为 NaN；上游生产口径尚未核实。 |
| `daily_kline_adj_factor` | `1d.kline.adj_factor` | 1d | date × code | daily | 缺失日期和证券读取为 NaN；上游生产口径尚未核实。 |
| `daily_kline_change` | `1d.kline.change` | 1d | date × code | daily | 缺失日期和证券读取为 NaN；上游生产口径尚未核实。 |
| `daily_kline_high_hfq` | `1d.kline.high_hfq` | 1d | date × code | daily | 缺失日期和证券读取为 NaN；上游生产口径尚未核实。 |
| `daily_kline_low_hfq` | `1d.kline.low_hfq` | 1d | date × code | daily | 缺失日期和证券读取为 NaN；上游生产口径尚未核实。 |
| `daily_kline_open_hfq` | `1d.kline.open_hfq` | 1d | date × code | daily | 缺失日期和证券读取为 NaN；上游生产口径尚未核实。 |
| `daily_kline_pct_chg` | `1d.kline.pct_chg` | 1d | date × code | daily | 缺失日期和证券读取为 NaN；上游生产口径尚未核实。 |
| `daily_kline_real_pre_close` | `1d.kline.real_pre_close` | 1d | date × code | daily | 缺失日期和证券读取为 NaN；上游生产口径尚未核实。 |
| `daily_kline_state_adj_factor` | `1d.kline_state.adj_factor` | 1d | date × code | daily | 缺失日期和证券读取为 NaN；上游生产口径尚未核实。 |
| `daily_kline_state_real_pre_close` | `1d.kline_state.real_pre_close` | 1d | date × code | daily | 缺失日期和证券读取为 NaN；上游生产口径尚未核实。 |
| `daily_kline_vol_hfq` | `1d.kline.vol_hfq` | 1d | date × code | daily | 缺失日期和证券读取为 NaN；上游生产口径尚未核实。 |
| `high` | `1d.kline.high` | 1d | date × code | daily | 缺失日期和证券读取为 NaN；上游生产口径尚未核实。 |
| `low` | `1d.kline.low` | 1d | date × code | daily | 缺失日期和证券读取为 NaN；上游生产口径尚未核实。 |
| `open` | `1d.kline.open` | 1d | date × code | daily | 缺失日期和证券读取为 NaN；上游生产口径尚未核实。 |
| `pre_close` | `1d.kline.pre_close` | 1d | date × code | daily | 缺失日期和证券读取为 NaN；上游生产口径尚未核实。 |
| `volume` | `1d.kline.vol` | 1d | date × code | daily | 缺失日期和证券读取为 NaN；上游生产口径尚未核实。 |
