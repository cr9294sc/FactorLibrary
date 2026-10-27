# 日内衍生特征变量 (Daily Intraday)

基于日内微观结构与分时量价计算生成的日频衍生统计特征（如 IntraVwap、Twap、早盘/尾盘分时指标等共 54 个）。

> **总计收录**：`54` 个变量

---

| 变量名称 (Short Name) | 底层标识 (Actual Name) | 频度 | 维度 (Axes) | 数据类型 (Kind) | 缺失值处理策略 |
| :--- | :--- | :---: | :---: | :---: | :--- |
| `daily_derived_dailyvolratio_aftbuyvolratio` | `daily-derived.DailyVolRatio.AftBuyVolRatio` | 1d | date × code | daily | 读取缺失日期或未匹配证券时返回 NaN，不自动前向填充或补零。生产器内部的填充规则另见字段说明。 |
| `daily_derived_dailyvolratio_aftsellvolratio` | `daily-derived.DailyVolRatio.AftSellVolRatio` | 1d | date × code | daily | 读取缺失日期或未匹配证券时返回 NaN，不自动前向填充或补零。生产器内部的填充规则另见字段说明。 |
| `daily_derived_dailyvolratio_aftsmallbuyvolratio` | `daily-derived.DailyVolRatio.AftSmallBuyVolRatio` | 1d | date × code | daily | 读取缺失日期或未匹配证券时返回 NaN，不自动前向填充或补零。生产器内部的填充规则另见字段说明。 |
| `daily_derived_dailyvolratio_aftsmallsellvolratio` | `daily-derived.DailyVolRatio.AftSmallSellVolRatio` | 1d | date × code | daily | 读取缺失日期或未匹配证券时返回 NaN，不自动前向填充或补零。生产器内部的填充规则另见字段说明。 |
| `daily_derived_dailyvolratio_aftvolratio` | `daily-derived.DailyVolRatio.AftVolRatio` | 1d | date × code | daily | 读取缺失日期或未匹配证券时返回 NaN，不自动前向填充或补零。生产器内部的填充规则另见字段说明。 |
| `daily_derived_dailyvolratio_morbuyvolratio` | `daily-derived.DailyVolRatio.MorBuyVolRatio` | 1d | date × code | daily | 读取缺失日期或未匹配证券时返回 NaN，不自动前向填充或补零。生产器内部的填充规则另见字段说明。 |
| `daily_derived_dailyvolratio_morsellratio` | `daily-derived.DailyVolRatio.MorSellRatio` | 1d | date × code | daily | 读取缺失日期或未匹配证券时返回 NaN，不自动前向填充或补零。生产器内部的填充规则另见字段说明。 |
| `daily_derived_dailyvolratio_morsmallbuyvolratio` | `daily-derived.DailyVolRatio.MorSmallBuyVolRatio` | 1d | date × code | daily | 读取缺失日期或未匹配证券时返回 NaN，不自动前向填充或补零。生产器内部的填充规则另见字段说明。 |
| `daily_derived_dailyvolratio_morsmallsellvolratio` | `daily-derived.DailyVolRatio.MorSmallSellVolRatio` | 1d | date × code | daily | 读取缺失日期或未匹配证券时返回 NaN，不自动前向填充或补零。生产器内部的填充规则另见字段说明。 |
| `daily_derived_dailyvolratio_morvolratio` | `daily-derived.DailyVolRatio.MorVolRatio` | 1d | date × code | daily | 读取缺失日期或未匹配证券时返回 NaN，不自动前向填充或补零。生产器内部的填充规则另见字段说明。 |
| `daily_derived_dailyvolratio_smallbuyvolratio` | `daily-derived.DailyVolRatio.SmallBuyVolRatio` | 1d | date × code | daily | 读取缺失日期或未匹配证券时返回 NaN，不自动前向填充或补零。生产器内部的填充规则另见字段说明。 |
| `daily_derived_dailyvolratio_smallsellvolratio` | `daily-derived.DailyVolRatio.SmallSellVolRatio` | 1d | date × code | daily | 读取缺失日期或未匹配证券时返回 NaN，不自动前向填充或补零。生产器内部的填充规则另见字段说明。 |
| `daily_derived_intravwap_vwap30_093000` | `daily-derived.IntraVwap.Vwap30.093000` | 1d | date × code | daily | 读取缺失日期或未匹配证券时返回 NaN，不自动前向填充或补零。生产器内部的填充规则另见字段说明。 |
| `daily_derived_intravwap_vwap30_100000` | `daily-derived.IntraVwap.Vwap30.100000` | 1d | date × code | daily | 读取缺失日期或未匹配证券时返回 NaN，不自动前向填充或补零。生产器内部的填充规则另见字段说明。 |
| `daily_derived_intravwap_vwap30_103000` | `daily-derived.IntraVwap.Vwap30.103000` | 1d | date × code | daily | 读取缺失日期或未匹配证券时返回 NaN，不自动前向填充或补零。生产器内部的填充规则另见字段说明。 |
| `daily_derived_intravwap_vwap30_110000` | `daily-derived.IntraVwap.Vwap30.110000` | 1d | date × code | daily | 读取缺失日期或未匹配证券时返回 NaN，不自动前向填充或补零。生产器内部的填充规则另见字段说明。 |
| `daily_derived_intravwap_vwap30_130000` | `daily-derived.IntraVwap.Vwap30.130000` | 1d | date × code | daily | 读取缺失日期或未匹配证券时返回 NaN，不自动前向填充或补零。生产器内部的填充规则另见字段说明。 |
| `daily_derived_intravwap_vwap30_133000` | `daily-derived.IntraVwap.Vwap30.133000` | 1d | date × code | daily | 读取缺失日期或未匹配证券时返回 NaN，不自动前向填充或补零。生产器内部的填充规则另见字段说明。 |
| `daily_derived_intravwap_vwap30_140000` | `daily-derived.IntraVwap.Vwap30.140000` | 1d | date × code | daily | 读取缺失日期或未匹配证券时返回 NaN，不自动前向填充或补零。生产器内部的填充规则另见字段说明。 |
| `daily_derived_intravwap_vwap30_143000` | `daily-derived.IntraVwap.Vwap30.143000` | 1d | date × code | daily | 读取缺失日期或未匹配证券时返回 NaN，不自动前向填充或补零。生产器内部的填充规则另见字段说明。 |
| `daily_derived_intravwap_vwapbegin120` | `daily-derived.IntraVwap.VwapBegin120` | 1d | date × code | daily | 读取缺失日期或未匹配证券时返回 NaN，不自动前向填充或补零。生产器内部的填充规则另见字段说明。 |
| `daily_derived_intravwap_vwapbegin15` | `daily-derived.IntraVwap.VwapBegin15` | 1d | date × code | daily | 读取缺失日期或未匹配证券时返回 NaN，不自动前向填充或补零。生产器内部的填充规则另见字段说明。 |
| `daily_derived_intravwap_vwapbegin30` | `daily-derived.IntraVwap.VwapBegin30` | 1d | date × code | daily | 读取缺失日期或未匹配证券时返回 NaN，不自动前向填充或补零。生产器内部的填充规则另见字段说明。 |
| `daily_derived_intravwap_vwapbegin60` | `daily-derived.IntraVwap.VwapBegin60` | 1d | date × code | daily | 读取缺失日期或未匹配证券时返回 NaN，不自动前向填充或补零。生产器内部的填充规则另见字段说明。 |
| `daily_derived_intravwap_vwapend120` | `daily-derived.IntraVwap.VwapEnd120` | 1d | date × code | daily | 读取缺失日期或未匹配证券时返回 NaN，不自动前向填充或补零。生产器内部的填充规则另见字段说明。 |
| `daily_derived_intravwap_vwapend15` | `daily-derived.IntraVwap.VwapEnd15` | 1d | date × code | daily | 读取缺失日期或未匹配证券时返回 NaN，不自动前向填充或补零。生产器内部的填充规则另见字段说明。 |
| `daily_derived_intravwap_vwapend30` | `daily-derived.IntraVwap.VwapEnd30` | 1d | date × code | daily | 读取缺失日期或未匹配证券时返回 NaN，不自动前向填充或补零。生产器内部的填充规则另见字段说明。 |
| `daily_derived_intravwap_vwapend60` | `daily-derived.IntraVwap.VwapEnd60` | 1d | date × code | daily | 读取缺失日期或未匹配证券时返回 NaN，不自动前向填充或补零。生产器内部的填充规则另见字段说明。 |
| `daily_derived_returns_c2c_10d` | `daily-derived.Returns.c2c_10d` | 1d | date × code | daily | 读取缺失日期或未匹配证券时返回 NaN，不自动前向填充或补零。生产器内部的填充规则另见字段说明。 |
| `daily_derived_returns_c2c_1d` | `daily-derived.Returns.c2c_1d` | 1d | date × code | daily | 读取缺失日期或未匹配证券时返回 NaN，不自动前向填充或补零。生产器内部的填充规则另见字段说明。 |
| `daily_derived_returns_c2c_20d` | `daily-derived.Returns.c2c_20d` | 1d | date × code | daily | 读取缺失日期或未匹配证券时返回 NaN，不自动前向填充或补零。生产器内部的填充规则另见字段说明。 |
| `daily_derived_returns_c2c_5d` | `daily-derived.Returns.c2c_5d` | 1d | date × code | daily | 读取缺失日期或未匹配证券时返回 NaN，不自动前向填充或补零。生产器内部的填充规则另见字段说明。 |
| `daily_derived_returns_intra` | `daily-derived.Returns.intra` | 1d | date × code | daily | 读取缺失日期或未匹配证券时返回 NaN，不自动前向填充或补零。生产器内部的填充规则另见字段说明。 |
| `daily_derived_returns_o2o_10d` | `daily-derived.Returns.o2o_10d` | 1d | date × code | daily | 读取缺失日期或未匹配证券时返回 NaN，不自动前向填充或补零。生产器内部的填充规则另见字段说明。 |
| `daily_derived_returns_o2o_1d` | `daily-derived.Returns.o2o_1d` | 1d | date × code | daily | 读取缺失日期或未匹配证券时返回 NaN，不自动前向填充或补零。生产器内部的填充规则另见字段说明。 |
| `daily_derived_returns_o2o_20d` | `daily-derived.Returns.o2o_20d` | 1d | date × code | daily | 读取缺失日期或未匹配证券时返回 NaN，不自动前向填充或补零。生产器内部的填充规则另见字段说明。 |
| `daily_derived_returns_o2o_5d` | `daily-derived.Returns.o2o_5d` | 1d | date × code | daily | 读取缺失日期或未匹配证券时返回 NaN，不自动前向填充或补零。生产器内部的填充规则另见字段说明。 |
| `daily_derived_returns_overnight` | `daily-derived.Returns.overnight` | 1d | date × code | daily | 读取缺失日期或未匹配证券时返回 NaN，不自动前向填充或补零。生产器内部的填充规则另见字段说明。 |
| `daily_derived_returns_vwap30_c2c_093000` | `daily-derived.Returns.vwap30_c2c_093000` | 1d | date × code | daily | 缺失日期和证券读取为 NaN；上游生产口径尚未核实。 |
| `daily_derived_returns_vwap30_c2c_100000` | `daily-derived.Returns.vwap30_c2c_100000` | 1d | date × code | daily | 缺失日期和证券读取为 NaN；上游生产口径尚未核实。 |
| `daily_derived_returns_vwap30_c2c_103000` | `daily-derived.Returns.vwap30_c2c_103000` | 1d | date × code | daily | 缺失日期和证券读取为 NaN；上游生产口径尚未核实。 |
| `daily_derived_returns_vwap30_c2c_110000` | `daily-derived.Returns.vwap30_c2c_110000` | 1d | date × code | daily | 缺失日期和证券读取为 NaN；上游生产口径尚未核实。 |
| `daily_derived_returns_vwap30_c2c_130000` | `daily-derived.Returns.vwap30_c2c_130000` | 1d | date × code | daily | 缺失日期和证券读取为 NaN；上游生产口径尚未核实。 |
| `daily_derived_returns_vwap30_c2c_133000` | `daily-derived.Returns.vwap30_c2c_133000` | 1d | date × code | daily | 缺失日期和证券读取为 NaN；上游生产口径尚未核实。 |
| `daily_derived_returns_vwap30_c2c_140000` | `daily-derived.Returns.vwap30_c2c_140000` | 1d | date × code | daily | 缺失日期和证券读取为 NaN；上游生产口径尚未核实。 |
| `daily_derived_returns_vwap30_c2c_143000` | `daily-derived.Returns.vwap30_c2c_143000` | 1d | date × code | daily | 缺失日期和证券读取为 NaN；上游生产口径尚未核实。 |
| `daily_twap_intratwap_twapbegin120` | `daily-twap.IntraTwap.TwapBegin120` | 1d | date × code | daily | 读取缺失日期或未匹配证券时返回 NaN，不自动前向填充或补零。生产器内部的填充规则另见字段说明。 |
| `daily_twap_intratwap_twapbegin15` | `daily-twap.IntraTwap.TwapBegin15` | 1d | date × code | daily | 读取缺失日期或未匹配证券时返回 NaN，不自动前向填充或补零。生产器内部的填充规则另见字段说明。 |
| `daily_twap_intratwap_twapbegin30` | `daily-twap.IntraTwap.TwapBegin30` | 1d | date × code | daily | 读取缺失日期或未匹配证券时返回 NaN，不自动前向填充或补零。生产器内部的填充规则另见字段说明。 |
| `daily_twap_intratwap_twapbegin60` | `daily-twap.IntraTwap.TwapBegin60` | 1d | date × code | daily | 读取缺失日期或未匹配证券时返回 NaN，不自动前向填充或补零。生产器内部的填充规则另见字段说明。 |
| `daily_twap_intratwap_twapend120` | `daily-twap.IntraTwap.TwapEnd120` | 1d | date × code | daily | 读取缺失日期或未匹配证券时返回 NaN，不自动前向填充或补零。生产器内部的填充规则另见字段说明。 |
| `daily_twap_intratwap_twapend15` | `daily-twap.IntraTwap.TwapEnd15` | 1d | date × code | daily | 读取缺失日期或未匹配证券时返回 NaN，不自动前向填充或补零。生产器内部的填充规则另见字段说明。 |
| `daily_twap_intratwap_twapend30` | `daily-twap.IntraTwap.TwapEnd30` | 1d | date × code | daily | 读取缺失日期或未匹配证券时返回 NaN，不自动前向填充或补零。生产器内部的填充规则另见字段说明。 |
| `daily_twap_intratwap_twapend60` | `daily-twap.IntraTwap.TwapEnd60` | 1d | date × code | daily | 读取缺失日期或未匹配证券时返回 NaN，不自动前向填充或补零。生产器内部的填充规则另见字段说明。 |
