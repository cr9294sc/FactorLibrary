# 股票池与交易状态掩码 (Universe & Masks)

包含基础可选股票池（BaseUnivMask）、涨跌停掩码（LimitMask）、ST 股票、停牌、上市天数过滤等 29 个状态标记。

> **总计收录**：`29` 个变量

---

| 变量名称 (Short Name) | 底层标识 (Actual Name) | 频度 | 维度 (Axes) | 数据类型 (Kind) | 缺失值处理策略 |
| :--- | :--- | :---: | :---: | :---: | :--- |
| `BaseUnivMask` | `mask.stock.BaseUnivMask` | 1d | date × code | daily | 缺失日期和证券读取为 NaN；上游生产口径尚未核实。 |
| `LimitMask` | `mask.stock.LimitMask` | 1d | date × code | daily | 缺失日期和证券读取为 NaN；上游生产口径尚未核实。 |
| `StockListedDays` | `mask.stock.StockListedDays` | 1d | date × code | daily | 缺失日期和证券读取为 NaN；上游生产口径尚未核实。 |
| `SuspendStock` | `mask.stock.SuspendStock` | 1d | date × code | daily | 缺失日期和证券读取为 NaN；上游生产口径尚未核实。 |
| `mask_stock_fixed_baseunivmask` | `mask.stock.fixed.BaseUnivMask` | 1d | date × code | daily | 缺失日期和证券读取为 NaN；上游生产口径尚未核实。 |
| `mask_stock_fixed_down_limit` | `mask.stock.fixed.down_limit` | 1d | date × code | daily | 缺失日期和证券读取为 NaN；上游生产口径尚未核实。 |
| `mask_stock_fixed_limitmask` | `mask.stock.fixed.LimitMask` | 1d | date × code | daily | 缺失日期和证券读取为 NaN；上游生产口径尚未核实。 |
| `mask_stock_fixed_nonewstockmask` | `mask.stock.fixed.NoNewStockMask` | 1d | date × code | daily | 缺失日期和证券读取为 NaN；上游生产口径尚未核实。 |
| `mask_stock_fixed_stocklisteddays` | `mask.stock.fixed.StockListedDays` | 1d | date × code | daily | 缺失日期和证券读取为 NaN；上游生产口径尚未核实。 |
| `mask_stock_fixed_ststock` | `mask.stock.fixed.STStock` | 1d | date × code | daily | 缺失日期和证券读取为 NaN；上游生产口径尚未核实。 |
| `mask_stock_fixed_suspendstock` | `mask.stock.fixed.SuspendStock` | 1d | date × code | daily | 缺失日期和证券读取为 NaN；上游生产口径尚未核实。 |
| `mask_stock_fixed_up_limit` | `mask.stock.fixed.up_limit` | 1d | date × code | daily | 缺失日期和证券读取为 NaN；上游生产口径尚未核实。 |
| `stock_mask_down_limit` | `mask.stock.down_limit` | 1d | date × code | daily | 缺失日期和证券读取为 NaN；上游生产口径尚未核实。 |
| `stock_mask_nonewstockmask` | `mask.stock.NoNewStockMask` | 1d | date × code | daily | 缺失日期和证券读取为 NaN；上游生产口径尚未核实。 |
| `stock_mask_risk_level` | `mask.stock.risk_level` | 1d | date × code | daily | 缺失日期和证券读取为 NaN；上游生产口径尚未核实。 |
| `stock_mask_ststock` | `mask.stock.STStock` | 1d | date × code | daily | 缺失日期和证券读取为 NaN；上游生产口径尚未核实。 |
| `stock_mask_up_limit` | `mask.stock.up_limit` | 1d | date × code | daily | 缺失日期和证券读取为 NaN；上游生产口径尚未核实。 |
| `trade_universe_asharebj` | `mask.universe.AshareBJ` | 1d | date × code | daily | 缺失日期和证券读取为 NaN；上游生产口径尚未核实。 |
| `trade_universe_asharecyb` | `mask.universe.AshareCYB` | 1d | date × code | daily | 缺失日期和证券读取为 NaN；上游生产口径尚未核实。 |
| `trade_universe_asharedelistrisk` | `mask.universe.AshareDelistRisk` | 1d | date × code | daily | 缺失日期和证券读取为 NaN；上游生产口径尚未核实。 |
| `trade_universe_asharesh` | `mask.universe.AshareSH` | 1d | date × code | daily | 缺失日期和证券读取为 NaN；上游生产口径尚未核实。 |
| `trade_universe_asharesz` | `mask.universe.AshareSZ` | 1d | date × code | daily | 缺失日期和证券读取为 NaN；上游生产口径尚未核实。 |
| `trade_universe_ewm_amt` | `mask.universe.ewm_amt` | 1d | date × code | daily | 缺失日期和证券读取为 NaN；上游生产口径尚未核实。 |
| `trade_universe_hs300` | `mask.universe.HS300` | 1d | date × code | daily | 缺失日期和证券读取为 NaN；上游生产口径尚未核实。 |
| `trade_universe_top1500` | `mask.universe.TOP1500` | 1d | date × code | daily | 缺失日期和证券读取为 NaN；上游生产口径尚未核实。 |
| `trade_universe_top3000` | `mask.universe.TOP3000` | 1d | date × code | daily | 缺失日期和证券读取为 NaN；上游生产口径尚未核实。 |
| `trade_universe_zz1000` | `mask.universe.ZZ1000` | 1d | date × code | daily | 缺失日期和证券读取为 NaN；上游生产口径尚未核实。 |
| `trade_universe_zz500` | `mask.universe.ZZ500` | 1d | date × code | daily | 缺失日期和证券读取为 NaN；上游生产口径尚未核实。 |
| `trade_universe_zz800` | `mask.universe.ZZ800` | 1d | date × code | daily | 缺失日期和证券读取为 NaN；上游生产口径尚未核实。 |
