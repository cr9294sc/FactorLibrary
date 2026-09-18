# FactorLibrary 期货数据仓库说明文档

## 概述

本数据仓库包含从米筐科技 (Ricequant) 导出的中国期货市场历史数据，存储格式为 Parquet，总数据量约 **12.7 GB**。数据涵盖期货合约行情、主力连续合约、复权因子、会员持仓排名、展期收益率、仓单等多个维度。

## 数据存储结构

```
exports/
├── _schema/                    # 数据结构定义
│   ├── column_types.json      # 字段类型映射（Arrow/PostgreSQL）
│   └── timescaledb_ddl.sql    # TimescaleDB 建表语句
├── _state/                     # 数据状态管理
├── _tmp/                       # 临时文件目录
├── calendar/                   # 交易日历
├── instruments_info/           # 合约基本信息
├── industry_mapping/           # 行业分类映射
├── bar_1d/                     # 期货合约日线数据
├── bar_1m/                     # 期货合约分钟线数据
├── dominant_bar_1m/            # 主力连续合约分钟线
├── dominant_mapping/           # 主力合约映射关系
├── dominant_ex_factor/         # 主力连续合约复权因子
├── ex_factor/                  # 复权因子
├── member_rank/                # 会员持仓排名
├── roll_yield/                 # 展期收益率
└── warehouse_stocks/           # 仓单数据
```

## 数据集详细说明

### 1. 交易日历 (calendar)

**文件**: `calendar/calendar.parquet`  
**大小**: 36 KB  
**官方文档**: N/A（辅助数据）

#### 字段说明
| 字段名 | 数据类型 | 说明 |
|--------|----------|------|
| trading_date | DATE | 交易日期 |

#### 数据范围
- 包含期货市场所有交易日信息

---

### 2. 合约基本信息 (instruments_info)

**文件**: `instruments_info/instruments_info.parquet`  
**大小**: 214 KB  
**官方文档**: Ricequant API - 期货合约信息查询

#### 字段说明
| 字段名 | 数据类型 | 可空 | 说明 |
|--------|----------|------|------|
| order_book_id | TEXT | 否 | 合约代码（如 IF2401, CU2403） |
| underlying_symbol | TEXT | 是 | 品种代码（如 IF, CU, RB） |
| symbol | TEXT | 是 | 合约简称 |
| exchange | TEXT | 是 | 交易所代码（CFFEX/SHFE/DCE/CZCE/INE/GFEX） |
| contract_multiplier | DOUBLE | 是 | 合约乘数 |
| margin_rate | DOUBLE | 是 | 保证金率 |
| listed_date | TEXT | 是 | 上市日期 |
| de_listed_date | TEXT | 是 | 退市日期 |
| maturity_date | TEXT | 是 | 到期日 |
| trading_hours | TEXT | 是 | 交易时段 |
| round_lot | DOUBLE | 是 | 最小交易单位 |
| type | TEXT | 是 | 合约类型 |
| product | TEXT | 是 | 产品类型 |
| industry_name | TEXT | 是 | 所属行业 |
| start_delivery_date | TEXT | 是 | 开始交割日 |
| end_delivery_date | TEXT | 是 | 结束交割日 |

#### 数据特点
- 包含所有历史和当前期货合约的静态信息
- 覆盖五大期货交易所（中金所、上期所、大商所、郑商所、上能源、广期所）

---

### 3. 行业分类映射 (industry_mapping)

**文件**: `industry_mapping/industry_mapping.parquet`  
**大小**: 5.1 KB  
**官方文档**: N/A（辅助数据）

#### 字段说明
| 字段名 | 数据类型 | 可空 | 说明 |
|--------|----------|------|------|
| underlying_symbol | TEXT | 否 | 品种代码 |
| level1_industry | TEXT | 否 | 一级行业分类 |
| level2_industry | TEXT | 否 | 二级行业分类 |
| product_name_zh | TEXT | 是 | 产品中文名称 |
| industry_name_db | TEXT | 是 | 数据库行业名称 |
| sample_symbol | TEXT | 是 | 样例合约代码 |
| notes | TEXT | 是 | 备注说明 |

#### 行业分类体系
例如：
- **黑色金属**: 铁矿石(I)、螺纹钢(RB)、热轧卷板(HC)等
- **有色金属**: 铜(CU)、铝(AL)、锌(ZN)等
- **能源化工**: 原油(SC)、PTA(TA)、甲醇(MA)等
- **农产品**: 豆粕(M)、玉米(C)、白糖(SR)等

---

### 4. 期货合约日线数据 (bar_1d)

**文件数量**: 200 个 Parquet 文件  
**数据量**: 114 MB  
**时间范围**: 2010-01 至 2026-08  
**官方文档**: [futures.get_price (frequency='1d')](https://www.ricequant.com/doc/rqdata/python/generic-api#rqdata-API-get_price)

#### 字段说明
| 字段名 | 数据类型 | 可空 | 说明 | 对应官方字段 |
|--------|----------|------|------|-------------|
| datetime | TIMESTAMP | 否 | 交易时间戳 | datetime |
| order_book_id | VARCHAR | 否 | 合约代码 | order_book_id |
| open | DOUBLE | 是 | 开盘价 | open |
| high | DOUBLE | 是 | 最高价 | high |
| low | DOUBLE | 是 | 最低价 | low |
| close | DOUBLE | 是 | 收盘价 | close |
| volume | DOUBLE | 是 | 成交量 | volume |
| total_turnover | DOUBLE | 是 | 成交额 | total_turnover |
| open_interest | DOUBLE | 是 | 持仓量 | open_interest |
| trading_date | DATE | 是 | 交易日期 | - |
| prev_settlement | DOUBLE | 是 | 昨结算价 | prev_settlement |
| prev_close | DOUBLE | 是 | 昨收盘价 | - |
| settlement | DOUBLE | 是 | 结算价 | settlement |
| limit_up | DOUBLE | 是 | 涨停价 | limit_up |
| limit_down | DOUBLE | 是 | 跌停价 | limit_down |
| day_session_open | DOUBLE | 是 | 日盘开盘价 | - |
| is_unilateral | BOOLEAN | 是 | 是否单边市 | - |
| created_at | TIMESTAMPTZ | 是 | 创建时间 | - |
| updated_at | TIMESTAMPTZ | 是 | 更新时间 | - |

#### 文件组织
- 按年月分区存储：`bar_1d_YYYY_MM.parquet`
- 每个文件包含该月所有合约的日线数据

#### 对应 Ricequant API
```python
# 获取单个合约日线数据
rqdatac.futures.get_price('CU2403', start_date='20240101', end_date='20240131', frequency='1d')
```

---

### 5. 期货合约分钟线数据 (bar_1m)

**文件数量**: 200 个 Parquet 文件  
**数据量**: 9.5 GB（最大数据集）  
**时间范围**: 2010-01 至 2026-08  
**官方文档**: [futures.get_price (frequency='1m')](https://www.ricequant.com/doc/rqdata/python/generic-api#rqdata-API-get_price)

#### 字段说明
| 字段名 | 数据类型 | 可空 | 说明 | 对应官方字段 |
|--------|----------|------|------|-------------|
| datetime | TIMESTAMP | 否 | 分钟级时间戳 | datetime |
| order_book_id | VARCHAR | 否 | 合约代码 | order_book_id |
| open | DOUBLE | 是 | 开盘价 | open |
| high | DOUBLE | 是 | 最高价 | high |
| low | DOUBLE | 是 | 最低价 | low |
| close | DOUBLE | 是 | 收盘价 | close |
| volume | DOUBLE | 是 | 成交量 | volume |
| total_turnover | DOUBLE | 是 | 成交额 | total_turnover |
| open_interest | DOUBLE | 是 | 持仓量 | open_interest |
| trading_date | DATE | 是 | 交易日期（夜盘对应次日） | trading_date |
| is_unilateral | BOOLEAN | 是 | 是否单边市 | - |
| created_at | TIMESTAMPTZ | 是 | 创建时间 | - |
| updated_at | TIMESTAMPTZ | 是 | 更新时间 | - |

#### 数据特点
- 单个文件大小：2.8 MB ~ 27 MB
- 包含日盘和夜盘数据
- trading_date 字段处理了夜盘跨日问题（夜盘交易归属次日交易日）

#### 对应 Ricequant API
```python
# 获取分钟线数据
rqdatac.futures.get_price('RB2405', start_date='20240301', end_date='20240305', frequency='1m')
```

---

### 6. 主力连续合约分钟线 (dominant_bar_1m)

**文件数量**: 200 个 Parquet 文件  
**数据量**: 2.8 GB  
**时间范围**: 2010-01 至 2026-08  
**官方文档**: [futures.get_dominant_price](https://www.ricequant.com/doc/rqdata/python/futures-mod#rqdata-API-futures-get_dominant_price)

#### 字段说明
| 字段名 | 数据类型 | 可空 | 说明 | 对应官方字段 |
|--------|----------|------|------|-------------|
| datetime | TIMESTAMP | 否 | 分钟级时间戳 | datetime |
| trading_date | DATE | 否 | 交易日期 | trading_date |
| underlying_symbol | VARCHAR | 否 | 品种代码 | underlying_symbol |
| rule | INTEGER | 否 | 主力选取规则（0/1/2/3） | rule |
| rank | INTEGER | 否 | 主力级别（1=主力,2=次主力） | rank |
| order_book_id | VARCHAR | 是 | 当前主力合约代码 | dominant_id |
| adjust_method | VARCHAR | 否 | 复权方法 | adjust_method |
| ex_factor | DOUBLE | 是 | 复权因子 | - |
| ex_cum_factor | DOUBLE | 是 | 累计复权因子 | - |
| open | DOUBLE | 是 | 开盘价 | open |
| high | DOUBLE | 是 | 最高价 | high |
| low | DOUBLE | 是 | 最低价 | low |
| close | DOUBLE | 是 | 收盘价 | close |
| volume | DOUBLE | 是 | 成交量 | volume |
| total_turnover | DOUBLE | 是 | 成交额 | total_turnover |
| open_interest | DOUBLE | 是 | 持仓量 | open_interest |
| created_at | TIMESTAMPTZ | 是 | 创建时间 | - |
| updated_at | TIMESTAMPTZ | 是 | 更新时间 | - |

#### 主力合约选取规则 (rule)
根据官方文档：

- **rule=0** (默认): 当同品种其他合约持仓量超过当前主力合约 1.1 倍时切换，每个合约只能做一次主力/次主力
- **rule=1**: 只考虑最大/第二大昨仓
- **rule=2**: 昨日成交量与持仓量同为最大/第二大
- **rule=3**: 在 rule=0 基础上，最后一个交易日不能成为主力

#### 复权方法 (adjust_method)
- **prev_close_spread**: 基于切换前一日收盘价价差复权
- **open_spread**: 基于切换当日开盘价价差复权
- **prev_close_ratio**: 基于切换前一日收盘价比例复权
- **open_ratio**: 基于切换当日开盘价比例复权

#### 对应 Ricequant API
```python
# 获取主力连续合约数据
rqdatac.futures.get_dominant_price(
    underlying_symbols='IF',
    start_date='20210901',
    end_date='20210902',
    frequency='1m',
    adjust_type='pre',
    adjust_method='prev_close_spread',
    rule=0,
    rank=1
)
```

---

### 7. 主力合约映射关系 (dominant_mapping)

**文件数量**: 200 个 Parquet 文件  
**数据量**: 1.9 MB  
**时间范围**: 2010-01 至 2026-08  
**官方文档**: [futures.get_dominant](https://www.ricequant.com/doc/rqdata/python/futures-mod#rqdata-API-futures-get_dominant)

#### 字段说明
| 字段名 | 数据类型 | 可空 | 说明 | 对应官方字段 |
|--------|----------|------|------|-------------|
| date | DATE | 否 | 日期 | date |
| underlying_symbol | VARCHAR | 否 | 品种代码 | underlying_symbol |
| rule | INTEGER | 否 | 主力选取规则 | rule |
| rank | INTEGER | 否 | 主力级别 | rank |
| order_book_id | VARCHAR | 是 | 主力合约代码 | order_book_id |
| tertiary | VARCHAR | 是 | 次次主力合约代码 | - |
| created_at | TIMESTAMPTZ | 是 | 创建时间 | - |
| updated_at | TIMESTAMPTZ | 是 | 更新时间 | - |

#### 数据用途
- 记录每日各品种的主力、次主力合约切换历史
- 用于追溯某一时点的主力合约是哪个具体合约

#### 对应 Ricequant API
```python
# 获取某品种主力合约
rqdatac.futures.get_dominant('IF', start_date='20210801', end_date='20210831', rule=0, rank=1)
```

---

### 8. 主力连续合约复权因子 (dominant_ex_factor)

**文件数量**: 200 个 Parquet 文件  
**数据量**: 24 MB  
**时间范围**: 2010-01 至 2026-08  
**官方文档**: [futures.get_ex_factor](https://www.ricequant.com/doc/rqdata/python/futures-mod#rqdata-API-futures-get_ex_factor)

#### 字段说明
| 字段名 | 数据类型 | 可空 | 说明 | 对应官方字段 |
|--------|----------|------|------|-------------|
| date | DATE | 否 | 除权日（主力切换日） | ex_date |
| underlying_symbol | VARCHAR | 否 | 品种代码 | underlying_symbol |
| rule | INTEGER | 否 | 主力选取规则 | rule |
| rank | INTEGER | 否 | 主力级别 | rank |
| order_book_id | VARCHAR | 是 | 新主力合约代码 | - |
| adjust_method | VARCHAR | 否 | 复权方法 | adjust_method |
| ex_factor | DOUBLE | 是 | 复权因子 | ex_factor |
| ex_cum_factor | DOUBLE | 是 | 累计复权因子 | ex_cum_factor |
| created_at | TIMESTAMPTZ | 是 | 创建时间 | - |
| updated_at | TIMESTAMPTZ | 是 | 更新时间 | - |

#### 数据说明
- 记录主力合约切换时的复权因子
- 用于将不同时期的主力合约价格调整到可比口径

#### 对应 Ricequant API
```python
# 获取复权因子
rqdatac.futures.get_ex_factor(
    underlying_symbols='IF',
    start_date='20210601',
    end_date='20210902',
    adjust_method='prev_close_spread',
    rule=0,
    rank=1
)
```

---

### 9. 复权因子 (ex_factor)

**文件数量**: 290 个 Parquet 文件  
**数据量**: 6.7 MB  
**时间范围**: 2000-01 至 2026-08（数据历史最长）  
**官方文档**: [futures.get_ex_factor](https://www.ricequant.com/doc/rqdata/python/futures-mod#rqdata-API-futures-get_ex_factor)

#### 字段说明
| 字段名 | 数据类型 | 可空 | 说明 |
|--------|----------|------|------|
| ex_date | DATE | 否 | 除权日 |
| underlying_symbol | VARCHAR | 否 | 品种代码 |
| adjust_method | VARCHAR | 否 | 复权方法 |
| rule | INTEGER | 否 | 主力选取规则 |
| rank | INTEGER | 否 | 主力级别 |
| ex_factor | DOUBLE | 是 | 复权因子 |
| ex_cum_factor | DOUBLE | 是 | 累计复权因子 |
| ex_end_date | DATE | 是 | 复权因子截止日期 |
| created_at | TIMESTAMPTZ | 是 | 创建时间 |
| updated_at | TIMESTAMPTZ | 是 | 更新时间 |

#### 数据特点
- 最早可追溯到 2000 年
- 包含所有复权方法的因子数据

---

### 10. 会员持仓排名 (member_rank)

**文件数量**: 176 个 Parquet 文件  
**数据量**: 205 MB  
**时间范围**: 2012-01 至 2026-08  
**官方文档**: [futures.get_member_rank](https://www.ricequant.com/doc/rqdata/python/futures-mod#rqdata-API-futures-get_member_rank)

#### 字段说明
| 字段名 | 数据类型 | 可空 | 说明 | 对应官方字段 |
|--------|----------|------|------|-------------|
| trading_date | DATE | 否 | 交易日期 | trading_date |
| order_book_id | VARCHAR | 否 | 合约代码 | commodity_id |
| rank_by | VARCHAR | 否 | 排名依据（volume/long/short） | rank_by |
| member_name | VARCHAR | 否 | 期货公司名称 | member_name |
| member_rank | INTEGER | 否 | 排名 | rank |
| volume | DOUBLE | 是 | 交易量或持仓量 | volume |
| volume_change | DOUBLE | 是 | 变动量 | volume_change |
| created_at | TIMESTAMPTZ | 是 | 创建时间 | - |
| updated_at | TIMESTAMPTZ | 是 | 更新时间 | - |

#### 排名依据 (rank_by)
- **volume**: 成交量排名
- **long**: 持买仓量排名
- **short**: 持卖仓量排名

#### 数据用途
- 分析期货公司持仓分布
- 追踪大户持仓变化
- 研究主力资金动向

#### 对应 Ricequant API
```python
# 获取会员排名数据
rqdatac.futures.get_member_rank('A1901', trading_date='20180910', rank_by='short')
```

---

### 11. 展期收益率 (roll_yield)

**文件数量**: 200 个 Parquet 文件  
**数据量**: 18 MB  
**时间范围**: 2010-01 至 2026-08  
**官方文档**: [futures.get_roll_yield](https://www.ricequant.com/doc/rqdata/python/futures-mod#rqdata-API-futures-get_roll_yield)

#### 字段说明
| 字段名 | 数据类型 | 可空 | 说明 | 对应官方字段 |
|--------|----------|------|------|-------------|
| trading_date | DATE | 否 | 交易日期 | date |
| underlying_symbol | VARCHAR | 否 | 品种代码 | underlying_symbol |
| roll_type | VARCHAR | 否 | 展期类型 | type |
| rule | INTEGER | 否 | 主力选取规则 | rule |
| from_contract | VARCHAR | 是 | 滚动前合约 | from_contract |
| to_contract | VARCHAR | 是 | 滚动后合约 | to_contract |
| roll_yield | DOUBLE | 是 | 展期收益率 | yield |
| annualized_yield | DOUBLE | 是 | 自然日年化收益率 | annualized_yield |
| annualized_yield_trading | DOUBLE | 是 | 交易日年化收益率 | annualized_yield_trading |
| created_at | TIMESTAMPTZ | 是 | 创建时间 | - |
| updated_at | TIMESTAMPTZ | 是 | 更新时间 | - |

#### 展期类型 (roll_type)
- **main_sub**: 主力 → 次主力
- **near_main**: 近月 → 主力

#### 计算公式
- **main_sub 展期收益率** = (主力结算价 - 次主力结算价) / 次主力结算价
- **near_main 展期收益率** = (近月结算价 - 主力结算价) / 主力结算价
- **年化收益率** = 展期收益率 × (365 / 两合约到期自然日差)
- **年化收益率(交易日)** = 展期收益率 × (252 / 两合约到期交易日差)

#### 数据用途
- 分析期货市场升贴水结构
- 评估展期成本
- 构建商品期货 CTA 策略

#### 对应 Ricequant API
```python
# 获取展期收益率
rqdatac.futures.get_roll_yield('CU', start_date='20250811', end_date='20250815', type='main_sub', rule=0)
```

---

### 12. 仓单数据 (warehouse_stocks)

**文件数量**: 200 个 Parquet 文件  
**数据量**: 1.8 MB  
**时间范围**: 2010-01 至 2026-08  
**官方文档**: [futures.get_warehouse_stocks](https://www.ricequant.com/doc/rqdata/python/futures-mod#rqdata-API-futures-get_warehouse_stocks)

#### 字段说明
| 字段名 | 数据类型 | 可空 | 说明 | 对应官方字段 |
|--------|----------|------|------|-------------|
| trading_date | DATE | 否 | 交易日期 | date |
| underlying_symbol | VARCHAR | 否 | 品种代码 | underlying_symbol |
| on_warrant | DOUBLE | 是 | 注册仓单量 | on_warrant |
| exchange | VARCHAR | 是 | 交易所 | exchange |
| effective_forecast | DOUBLE | 是 | 有效预报（仅郑商所） | effective_forecast |
| warrant_units | VARCHAR | 是 | 仓单单位（仅郑商所） | warrant_units |
| contract_multiplier | DOUBLE | 是 | 合约乘数 | contract_multiplier |
| deliverable | DOUBLE | 是 | 可交割量（仅上期所） | deliverable |
| created_at | TIMESTAMPTZ | 是 | 创建时间 | - |
| updated_at | TIMESTAMPTZ | 是 | 更新时间 | - |

#### 数据特点
- 不同交易所披露的字段有所差异
- 上期所提供 deliverable（可交割量）
- 郑商所提供 effective_forecast（有效预报）和 warrant_units（仓单单位）

#### 数据用途
- 分析库存水平
- 判断供需关系
- 辅助基本面分析

#### 对应 Ricequant API
```python
# 获取仓单数据
rqdatac.futures.get_warehouse_stocks(['CF','SR'], start_date='20251120', end_date='20251120')
```

---

## 数据质量说明

### 郑商所数据特殊处理

根据官方文档说明，郑商所数据存在以下特殊处理：

1. **合约代码补齐**: CF701 补齐为 CF1701，避免年份混淆
2. **成交额异常处理**: tick 数据中成交额存在递减异常，分钟数据合成时若出现负值则设为 0
3. **日线数据合成**: 由当日 tick 合成，不包含盘后交易数据

### 品种代码切换

部分商品期货存在品种代码变更历史：

| 品种 | 旧代码 | 新代码 | 切换时间 |
|------|--------|--------|----------|
| 强麦 | WS | WH | 1305 及以前 |
| 普麦 | WT | PM | 1211 及以前 |
| 菜籽油 | RO | OI | 1305 及以前 |
| 早籼稻 | ER | RI | 1305 及以前 |
| 甲醇 | ME | MA | 1505 及以前 |
| 动力煤 | TC | ZC | 1604 及以前 |

主力连续合约中，变更前后作为两种合约处理（如 WS88 和 WH88）。

### 五档行情说明

- 上期所提供五档行情数据（2020-05-12 至今）
- 其他交易所仅提供一档行情

---

## 数据使用建议

### 1. 数据加载性能优化

由于单个数据集（如 bar_1m）数据量较大，建议：

- 使用 DuckDB 或 Polars 进行高效列式查询
- 按月份文件进行分区查询，避免加载全部数据
- 利用 Parquet 文件的列裁剪特性，只读取需要的字段

### 2. 主力连续合约使用

- **回测建议**: 使用 rule=0, rank=1, adjust_method='prev_close_spread'
- **实盘建议**: 结合 dominant_mapping 获取实时主力合约代码
- **注意事项**: 主力切换时会产生跳空，需要关注 ex_factor

### 3. 数据更新策略

当前数据截止到 **2026-08**，建议：

- 日线/分钟线数据：每日盘后更新
- 主力连续合约：每日盘后更新（主力可能切换）
- 会员排名/仓单：每周更新
- 基础信息数据：每月更新

### 4. TimescaleDB 导入

项目已提供 TimescaleDB DDL 脚本（`_schema/timescaledb_ddl.sql`），可直接导入到时序数据库：

```bash
psql -U postgres -d futures_db -f exports/_schema/timescaledb_ddl.sql
```

---

## 数据统计总览

| 数据集 | 文件数 | 数据量 | 时间范围 | 更新频率建议 |
|--------|--------|--------|----------|--------------|
| calendar | 1 | 36 KB | 全量 | 年度 |
| instruments_info | 1 | 214 KB | 全量 | 月度 |
| industry_mapping | 1 | 5.1 KB | 全量 | 年度 |
| bar_1d | 200 | 114 MB | 2010-01 ~ 2026-08 | 日度 |
| bar_1m | 200 | 9.5 GB | 2010-01 ~ 2026-08 | 日度 |
| dominant_bar_1m | 200 | 2.8 GB | 2010-01 ~ 2026-08 | 日度 |
| dominant_mapping | 200 | 1.9 MB | 2010-01 ~ 2026-08 | 日度 |
| dominant_ex_factor | 200 | 24 MB | 2010-01 ~ 2026-08 | 日度 |
| ex_factor | 290 | 6.7 MB | 2000-01 ~ 2026-08 | 日度 |
| member_rank | 176 | 205 MB | 2012-01 ~ 2026-08 | 周度 |
| roll_yield | 200 | 18 MB | 2010-01 ~ 2026-08 | 日度 |
| warehouse_stocks | 200 | 1.8 MB | 2010-01 ~ 2026-08 | 周度 |
| **总计** | **1,868** | **~12.7 GB** | **2000-01 ~ 2026-08** | - |

---

## 官方文档映射

| 本地数据集 | Ricequant API | 官方文档链接 |
|-----------|---------------|-------------|
| bar_1d | futures.get_price | [get_price](https://www.ricequant.com/doc/rqdata/python/generic-api#rqdata-API-get_price) |
| bar_1m | futures.get_price | [get_price](https://www.ricequant.com/doc/rqdata/python/generic-api#rqdata-API-get_price) |
| dominant_bar_1m | futures.get_dominant_price | [get_dominant_price](https://www.ricequant.com/doc/rqdata/python/futures-mod#rqdata-API-futures-get_dominant_price) |
| dominant_mapping | futures.get_dominant | [get_dominant](https://www.ricequant.com/doc/rqdata/python/futures-mod#rqdata-API-futures-get_dominant) |
| dominant_ex_factor | futures.get_ex_factor | [get_ex_factor](https://www.ricequant.com/doc/rqdata/python/futures-mod#rqdata-API-futures-get_ex_factor) |
| member_rank | futures.get_member_rank | [get_member_rank](https://www.ricequant.com/doc/rqdata/python/futures-mod#rqdata-API-futures-get_member_rank) |
| roll_yield | futures.get_roll_yield | [get_roll_yield](https://www.ricequant.com/doc/rqdata/python/futures-mod#rqdata-API-futures-get_roll_yield) |
| warehouse_stocks | futures.get_warehouse_stocks | [get_warehouse_stocks](https://www.ricequant.com/doc/rqdata/python/futures-mod#rqdata-API-futures-get_warehouse_stocks) |

---

## 数据审计结论

### ✅ 数据完整性
- 所有数据集均按月份分区存储，文件命名规范统一
- 关键字段定义明确，与官方文档对应关系清晰
- 提供了完整的 schema 定义和 TimescaleDB DDL

### ✅ 数据覆盖度
- 行情数据覆盖 2010 年至今（部分追溯到 2000 年）
- 涵盖五大交易所所有活跃品种
- 包含日线、分钟线、主力连续、衍生指标等多维度数据

### ⚠️ 注意事项
1. 郑商所数据存在历史遗留问题（合约代码补齐、成交额处理）
2. 部分品种存在代码变更历史，需要注意主力连续合约的处理
3. 五档行情数据仅上期所提供（2020-05 起）
4. 数据更新截止到 2026-08，建议定期增量更新

### 📊 数据规模
- 总文件数: 1,868 个 Parquet 文件
- 总数据量: 约 12.7 GB
- 最大单数据集: bar_1m (9.5 GB)
- 数据历史跨度: 26 年（2000-2026）

---

**文档生成时间**: 2026-09-18  
**数据版本**: 截至 2026-08  
**审计依据**: Ricequant 官方文档 + 本地 _schema 定义
