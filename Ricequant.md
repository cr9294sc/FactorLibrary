期货行情数据说明 [​](#rqdata-API-futures-overview)
------------------------------------------

可获取期货合约的日行情、分钟行情、tick 行情数据，具体调用方式请参考 [API-get_price](https://www.ricequant.com/doc/rqdata/python/generic-api#rqdata-API-get_price).

注意事项

关于五档数据，目前只有上期所提供（提供范围为 2020-05-12 至今）。其余商所只提供一档。

郑商所相关数据约定 [​](#rqdata-API-futures-zss-agreement)
------------------------------------------------

由于郑商所本身的一些问题，我们在数据上做了一定的处理，规则如下：

*   合约代码：郑商所的合约代码我们统一做了补齐，例如 CF701 补齐为 CF1701 以避免 2007 年的合约与 2017 年的合约无法分辨的问题。
    
*   成交额及成交量：由于郑商所的历史数据中 tick 成交额存在错误（该累积量存在随时间递减的情况），导致整个数据不可信。所以合成过程中若分钟成交额`TotalTurnover`出现负数，则将该分钟字段设置为 0，但分钟成交量`TotalVolumeTrade`正常。
    
*   日线合成：日线数据由当日 tick 数据合成，未包含盘后交易的成交数据（该数据交易所未披露完全）。
    
*   品种代码切换：一些商品期货交易品种存在合约代码修改的问题。因为变动前后它们的合约乘数产生了变化，所以目前我们将修改前后的期货合约**当做两种合约**来处理。而实际上，它们交易的又是同一品种，价格水平具备连续性。所以在主力连续与指数连续合约的处理上，我们做了区分处理。举例来说，强麦的主力连续合约我们有 “强麦主力连续（旧）” 和“强麦主力连续”，分别对应'WS88'和'WH88'的合约代码。一些变动的期货代码列举如下：
    

<table tabindex="0"><thead><tr><th>品种</th><th>品种代码（变动前）</th><th>品种代码（变动后）</th></tr></thead><tbody><tr><td>强麦</td><td>WS（1305 及以前）</td><td>WH</td></tr><tr><td>普麦</td><td>WT (1211 及以前)</td><td>PM</td></tr><tr><td>菜籽油</td><td>RO（1305 及以前）</td><td>OI</td></tr><tr><td>早籼稻</td><td>ER（1305 及以前）</td><td>RI</td></tr><tr><td>甲醇</td><td>ME（1505 及以前）</td><td>MA</td></tr><tr><td>动力煤</td><td>TC（1604 及以前）</td><td>ZC</td></tr></tbody></table>

连续合约 [​](#rqdata-API-futures-continuous)
----------------------------------------

需要注意，由于期货合约存续的特殊性，针对每一品种的期货合约，系统中都增加了**主力、次主力连续合约**以及**指数连续合约**两个人工合成的合约来满足使用需求。

*   主力连续合约：合约首次上市时，以当日收盘同品种持仓量最大者作为从第二个交易日开始的主力合约。当同品种其他合约持仓量在收盘后超过当前主力合约 1.1 倍时，从第二个交易日开始进行主力合约的切换。 日内不会进行主力合约的切换。主力连续合约是由该品种期货不同时期主力合约接续而成，代码以 88、888、889 结尾。
    
    *   例如 IF88、IF888、IF889：
        *   IF88 为合约量价数据的简单拼接，未做平滑处理
        *   IF888 为对价格进行了 "前复权平滑" 处理，处理规则如下：以主力合约切换前一天（T-1 日）新、旧两个主力合约收盘价做差，之后将 T-1 日及以前的主力连续合约的所有价格水平整体加上或减去该价差，以 "整体抬升" 或 "整体下降" 主力合约的价格水平，成交量、持仓量均不作调整，成交额统一设置为 0.
        *   IF889 为对价格进行 “后复权平滑” 处理，处理规则如下：以主力合约切换当天（T 日）旧、新两个主力合约开盘价做差， 之后将 T 日及以后的主力连续合约的所有价格水平整体加上或减去该价差，成交量、持仓量均不作调整，成交额统一设置为 0
*   次主力连续合约：比当前主力合约远月，未做过主力合约，也未做过次主力的合约中以累计持仓量第二大者作为次主力连续合约，代码以 88A2 结尾，例如 AU88A2。
    
*   指数连续合约：由当前品种全部可交易合约以累计持仓量为权重加权平均得到，代码以 99 结尾，例如 IF99。
    

### futures.get_dominant - 获取主力合约 [​](#rqdata-API-futures-get_dominant)

python

```
futures.get_dominant(underlying_symbol, start_date=None, end_date=None, rule=0, rank=1, market='cn')
```

获取某一期货品种在一段时间内的主力合约数据。例如可查询主力、次主力、次次主力合约，也支持 `rule=0`、`rule=1`、`rule=2`、`rule=3` 等不同主力合约选取规则。

#### 参数 [​](#rqdata-API-futures-get_dominant-params)

<table tabindex="0"><thead><tr><th>参数</th><th>类型</th><th>说明</th></tr></thead><tbody><tr><td>underlying_symbol</td><td><em>str</em></td><td><strong>必填参数</strong>，期货合约品种，例如沪深 300 股指期货为'IF'</td></tr><tr><td>start_date</td><td><em>int, str, datetime.date, datetime.datetime, pandas.Timestamp</em></td><td>开始日期，默认为期货品种最早上市日期后一交易日</td></tr><tr><td>end_date</td><td><em>int, str, datetime.date, datetime.datetime, pandas.Timestamp</em></td><td>结束日期，默认为当前日期</td></tr><tr><td>rule</td><td><em>int</em></td><td>主力合约选取规则。<br>默认 rule=0，当同品种其他合约持仓量在收盘后超过当前主力合约 1.1 倍时，从第二个交易日开始进行主力合约的切换。每个合约只能做一次主力 / 次主力合约，不会重复出现。针对股指期货，只在当月和次月选择主力合约。<br>当 rule=1 时，主力 / 次主力合约的选取只考虑最大 / 第二大昨仓这个条件。<br>当 rule=2 时，采用昨日成交量与持仓量同为最大 / 第二大的合约为当日主力 / 次主力。<br>当 rule=3 时，在 rule=0 选取规则上，考虑在最后一个交易日不能成为主力 / 次主力合约。</td></tr><tr><td>rank</td><td><em>int</em></td><td>默认 rank=1。<br>1 - 主力合约（支持所有期货）<br>2 - 次主力合约（支持所有期货；针对股指期货，需满足 <strong>rule=1</strong> 或 <strong>rule=2</strong>）<br>3 - 次次主力合约（支持所有期货；针对股指期货，需满足 <strong>rule=1</strong> 或 <strong>rule=2</strong>）</td></tr><tr><td>market</td><td><em>str</em></td><td>默认是中国内地市场 ('cn') 。可选'cn' - 中国内地市场；</td></tr></tbody></table>

#### 返回 [​](#rqdata-API-futures-get_dominant-return)

_Pandas.Series_ - 主力合约代码

#### 范例 [​](#rqdata-API-futures-get_dominant-example)

*   获取某一天的主力合约代码

python

```
[In]
futures.get_dominant('IF', '20160801')
[Out]
date
20160801    IF1608
```

*   获取从上市到某天之间的主力合约代码

python

```
[In]
futures.get_dominant('IC', end_date='20150501')
[Out]
date
20150417    IC1505
20150420    IC1505
20150421    IC1505
20150422    IC1505
20150423    IC1505
20150424    IC1505
20150427    IC1505
20150428    IC1505
20150429    IC1505
20150430    IC1505
20150501    IC1505
```

### futures.get_contracts - 获取期货可交易合约列表 [​](#rqdata-API-futures-get_contracts)

python

```
futures.get_contracts(underlying_symbol, date=None, market='cn')
```

获取指定期货品种在指定日期可交易的合约列表，结果按到期月份排序。

#### 参数 [​](#rqdata-API-futures-get_contracts-params)

<table tabindex="0"><thead><tr><th>参数</th><th>类型</th><th>说明</th></tr></thead><tbody><tr><td>underlying_symbol</td><td><em>str</em></td><td><strong>必填参数</strong>，期货合约品种，例如沪深 300 股指期货为'IF'</td></tr><tr><td>date</td><td><em>int, str, datetime.date, datetime.datetime, pandas.Timestamp</em></td><td>查询日期，默认为当日</td></tr><tr><td>market</td><td><em>str</em></td><td>默认是中国内地市场 ('cn') 。可选'cn' - 中国内地市场；</td></tr></tbody></table>

#### 返回 [​](#rqdata-API-futures-get_contracts-return)

_str list_ - 可交易的 order_book_id list

#### 范例 [​](#rqdata-API-futures-get_contracts-example)

python

```
[In]
futures.get_contracts('IF', '20160801')
[Out]
['IF1608', 'IF1609', 'IF1612', 'IF1703']
```

### futures.get_dominant_price - 获取期货主力连续合约行情数据 [​](#rqdata-API-futures-get_dominant_price)

python

```
futures.get_dominant_price(underlying_symbols,start_date=None,end_date=None,frequency='1d',fields=None,adjust_type='pre', adjust_method='prev_close_spread',rule=0,rank=1)
```

获取期货主力连续合约行情数据，支持日线、分钟线和 tick 级别数据。例如可按 `pre`、`none` 等复权方式查询，也支持 `prev_close_spread`、`open_spread` 等不同平滑方法。

#### 参数 [​](#rqdata-API-futures-get_dominant_price-params)

<table tabindex="0"><thead><tr><th>参数</th><th>类型</th><th>说明</th></tr></thead><tbody><tr><td>underlying_symbols</td><td><em>str&nbsp;or&nbsp;str list</em></td><td><strong>必填参数</strong>，期货合约品种，可传入 &nbsp;underlying_symbol, underlying_symbol list</td></tr><tr><td>start_date</td><td><em>int, str, datetime.date, datetime.datetime, pandas.Timestamp</em></td><td>开始日期</td></tr><tr><td>end_date</td><td><em>int, str, datetime.date, datetime.datetime, pandas.Timestamp</em></td><td>结束日期，start_date ,end_date 不传参数时默认返回最近三个月的数据</td></tr><tr><td>frequency</td><td><em>str</em></td><td>历史数据的频率。 支持 / 日 / 分钟 / tick 级别的历史数据，默认为'1d'。1m- 分钟线，1d - 日线，分钟可选取不同频率，例如'5m'代表 5 分钟线</td></tr><tr><td>fields</td><td><em>str&nbsp;or&nbsp;str list</em></td><td>查询字段，可选字段见下方返回，默认返回所有字段</td></tr><tr><td>adjust_type</td><td><em>str</em></td><td>复权方式，默认为'pre'，<br>none - 不复权 ，pre -&nbsp; 前复权， post -&nbsp; 后复权，</td></tr><tr><td>adjust_method</td><td><em>str</em></td><td>复权方法<br>'prev_close_spread'：基于主力合约切换前一个交易日收盘价价差进行复权<br>'open_spread'：基于主力合约切换当日开盘价价差进行复权<br>'prev_close_ratio'：基于主力合约切换前一个交易日收盘价比例进行复权<br>'open_ratio'：基于主力合约切换当日开盘价比例进行复权'<br>默认为'prev_close_spread';<br>adjust_type 为 None 时，adjust_method 复权方法设置无效</td></tr><tr><td>rule</td><td><em>int</em></td><td>主力合约选取规则。<br>默认 rule=0，当同品种其他合约持仓量在收盘后超过当前主力合约 1.1 倍时，从第二个交易日开始进行主力合约的切换。每个合约只能做一次主力 / 次主力合约，不会重复出现。针对股指期货，只在当月和次月选择主力合约。<br>当 rule=1 时，主力 / 次主力合约的选取只考虑最大 / 第二大昨仓这个条件。<br>当 rule=2 时，采用昨日成交量与持仓量同为最大 / 第二大的合约为当日主力 / 次主力。<br>当 rule=3 时，在 rule=0 选取规则上，考虑在最后一个交易日不能成为主力 / 次主力合约。</td></tr><tr><td>rank</td><td><em>int</em></td><td>默认 rank=1<br>1 - 主力合约（支持所有期货）<br>2 - 次主力合约（支持所有期货；针对股指期货，需满足 <strong>rule=1</strong> 或 <strong>rule=2</strong>）<br>3 - 次次主力合约（支持所有期货；针对股指期货，需满足 <strong>rule=1</strong> 或 <strong>rule=2</strong>）</td></tr><tr><td>time_slice</td><td><em>str</em>, <em>datetime.time</em></td><td>开始、结束时间段。默认返回当天所有数据。<br>支持分钟 / tick 级别的切分，详见下方范例。</td></tr></tbody></table>

#### 返回 [​](#rqdata-API-futures-get_dominant_price-return)

*   bar 数据

<table tabindex="0"><thead><tr><th>字段</th><th>类型</th><th>说明</th></tr></thead><tbody><tr><td>open</td><td><em>float</em></td><td>开盘价</td></tr><tr><td>close</td><td><em>float</em></td><td>收盘价</td></tr><tr><td>high</td><td><em>float</em></td><td>最高价</td></tr><tr><td>low</td><td><em>float</em></td><td>最低价</td></tr><tr><td>limit_up</td><td><em>float</em></td><td>涨停价（仅限日线数据）</td></tr><tr><td>limit_down</td><td><em>float</em></td><td>跌停价（仅限日线数据）</td></tr><tr><td>total_turnover</td><td><em>float</em></td><td>成交额</td></tr><tr><td>volume</td><td><em>float</em></td><td>成交量</td></tr><tr><td>settlement</td><td><em>float</em></td><td>结算价 （仅限日线数据）</td></tr><tr><td>prev_settlement</td><td><em>float</em></td><td>昨日结算价（仅限日线数据）</td></tr><tr><td>open_interest</td><td><em>float</em></td><td>累计持仓量</td></tr><tr><td>trading_date</td><td><em>pandas.Timestamp</em></td><td>交易日期（仅限分钟线数据），对应期货夜盘的情况</td></tr><tr><td>dominant_id</td><td><em>str</em></td><td>主力合约</td></tr></tbody></table>

*   tick 数据

<table tabindex="0"><thead><tr><th>字段</th><th>类型</th><th>说明</th></tr></thead><tbody><tr><td>open</td><td><em>float</em></td><td>当日开盘价</td></tr><tr><td>high</td><td><em>float</em></td><td>当日最高价</td></tr><tr><td>low</td><td><em>float</em></td><td>当日最低价</td></tr><tr><td>last</td><td><em>float</em></td><td>最新价</td></tr><tr><td>prev_close</td><td><em>float</em></td><td>昨日收盘价</td></tr><tr><td>total_turnover</td><td><em>float</em></td><td>成交额</td></tr><tr><td>volume</td><td><em>float</em></td><td>成交量</td></tr><tr><td>limit_up</td><td><em>float</em></td><td>涨停价</td></tr><tr><td>limit_down</td><td><em>float</em></td><td>跌停价</td></tr><tr><td>open_interest</td><td><em>float</em></td><td>累计持仓量</td></tr><tr><td>datetime</td><td><em>pandas.Timestamp</em></td><td>交易所时间戳</td></tr><tr><td>a1~a5</td><td><em>float</em></td><td>卖一至五档报盘价格</td></tr><tr><td>a1_v~a5_v</td><td><em>float</em></td><td>卖一至五档报盘量</td></tr><tr><td>b1~b5</td><td><em>float</em></td><td>买一至五档报盘价</td></tr><tr><td>b1_v~b5_v</td><td><em>float</em></td><td>买一至五档报盘量</td></tr><tr><td>change_rate</td><td><em>float</em></td><td>涨跌幅</td></tr><tr><td>trading_date</td><td><em>pandas.Timestamp</em></td><td>交易日期，对应期货夜盘的情况</td></tr><tr><td>prev_settlement</td><td><em>float</em></td><td>昨日结算价</td></tr><tr><td>dominant_id</td><td><em>str</em></td><td>主力合约</td></tr></tbody></table>

#### 范例 [​](#rqdata-API-futures-get_dominant_price-example)

*   获取期货主力连续合约前复权日线行情

python

```
[In]
futures.get_dominant_price(underlying_symbols='IF',start_date=20210901,end_date=20210902,frequency='1d',fields=None,adjust_type='pre', adjust_method='prev_close_spread')
[Out]
 		                        settlement      volume          limit_down	open	open_interest	total_turnover	limit_up	close	low	prev_settlement	high
underlying_symbol	date
IF	                2021-09-01	4855.0	        130017.0        4290.2	        4767.2	        143730.0	0	5243.4	        4856.4	4737.2	4766.8	        4898.6
                        2021-09-02	4854.0	        73853.0	        4369.6	        4855.0	        128436.0	0	5340.4	        4854.2	4830.2	4855.0	        4879.0
```

*   获取期货主力连续合约不复权分钟线行情

python

```
[In]
futures.get_dominant_price(underlying_symbols='IF',start_date=20210901,end_date=20210901,frequency='1m',fields=None,adjust_type='none', adjust_method='prev_close_spread')
[Out]
                                                trading_date	volume	open	open_interest	total_turnover	close	low	high
underlying_symbol       datetime
IF	                2021-09-01 09:31:00	2021-09-01	2087.0	4767.2	140180.0	2.990109e+09	4779.0	4767.0	4781.8
                        2021-09-01 09:32:00	2021-09-01	1044.0	4779.6	139408.0	1.496143e+09	4773.2	4772.0	4780.6
                        2021-09-01 09:33:00	2021-09-01	964.0	4773.2	138709.0	1.379020e+09	4763.4	4763.0	4773.2
                        2021-09-01 09:34:00	2021-09-01	1239.0	4763.4	137894.0	1.768014e+09	4751.4	4750.8	4763.4
                        2021-09-01 09:35:00	2021-09-01	1126.0	4750.8	137099.0	1.605260e+09	4755.2	4748.0	4755.6
                        ···
                        2021-09-01 14:58:00	2021-09-01	589.0	4854.4	143113.0	8.574656e+08	4855.2	4851.0	4855.4
                        2021-09-01 14:59:00	2021-09-01	445.0	4855.0	143410.0	6.483443e+08	4856.8	4853.8	4858.2
                        2021-09-01 15:00:00	2021-09-01	611.0	4857.4	143730.0	8.907135e+08	4856.4	4855.8	4861.0
```

*   获取期货主力连续合约 20210901 - 20210901 每个交易日开盘 09:31 至 09:32 的历史分钟行情 (切分规则：先获取 20210901 - 20210901 所有的数据再进行切分，消耗的还是所有数据的流量)

python

```
[In]
futures.get_dominant_price(underlying_symbols='IF',start_date=20210901,end_date=20210902,frequency='1m',fields=None,adjust_type='pre', adjust_method='prev_close_spread',time_slice=('09:31','09:32'))
[Out]
                                trading_date	dominant_id	open	close	high	low	total_turnover	volume	open_interest
underlying_symbol	datetime									
IF	2021-09-01 09:31:00	2021-09-01	IF2109	4390.6	4402.4	4405.2	4390.4	0	2087.0	140180.0
        2021-09-01 09:32:00	2021-09-01	IF2109	4403.0	4396.6	4404.0	4395.4	0	1044.0	139408.0
        2021-09-02 09:31:00	2021-09-02	IF2109	4478.4	4490.6	4491.2	4477.0	0	1855.0	142804.0
        2021-09-02 09:32:00	2021-09-02	IF2109	4490.6	4480.6	4490.6	4478.6	0	1335.0	141883.0
```

### futures.get_ex_factor - 获取期货主力连续合约复权因子 [​](#rqdata-API-futures-get_ex_factor)

python

```
futures.get_ex_factor(underlying_symbols, start_date=None, end_date=None, adjust_method='prev_close_spread', rule=0, rank=1, market='cn')
```

获取期货主力连续合约复权因子数据。例如复权方法可选 `prev_close_spread`、`open_spread`、`prev_close_ratio`，也支持 `open_ratio`。

#### 参数 [​](#rqdata-API-futures-get_ex_factor-params)

<table tabindex="0"><thead><tr><th>参数</th><th>类型</th><th>说明</th></tr></thead><tbody><tr><td>underlying_symbols</td><td><em>str&nbsp;or&nbsp;list</em></td><td><strong>必填参数</strong>，品种代码</td></tr><tr><td>start_date</td><td><em>int, str, datetime.date, datetime.datetime, pandas.Timestamp</em></td><td>开始日期 ，不传入时返回全部</td></tr><tr><td>end_date</td><td><em>int, str, datetime.date, datetime.datetime, pandas.Timestamp</em></td><td>结束日期 ，不传入时返回全部</td></tr><tr><td>adjust_method</td><td><em>str</em></td><td>复权方法<br>'prev_close_spread'：基于主力合约切换前一个交易日收盘价价差进行复权<br>'open_spread'：基于主力合约切换当日开盘价价差进行复权<br>'prev_close_ratio'：基于主力合约切换前一个交易日收盘价比例进行复权<br>'open_ratio'：基于主力合约切换当日开盘价比例进行复权'<br>默认为'prev_close_spread'</td></tr><tr><td>rule</td><td><em>int</em></td><td>主力合约选取规则。<br>默认 rule=0，当同品种其他合约持仓量在收盘后超过当前主力合约 1.1 倍时，从第二个交易日开始进行主力合约的切换。每个合约只能做一次主力 / 次主力合约，不会重复出现。针对股指期货，只在当月和次月选择主力合约。<br>当 rule=1 时，主力 / 次主力合约的选取只考虑最大 / 第二大昨仓这个条件。<br>当 rule=2 时，采用昨日成交量与持仓量同为最大 / 第二大的合约为当日主力 / 次主力。<br>当 rule=3 时，在 rule=0 选取规则上，考虑在最后一个交易日不能成为主力 / 次主力合约。</td></tr><tr><td>rank</td><td><em>int</em></td><td>默认 rank=1<br>1 - 主力合约（支持所有期货）<br>2 - 次主力合约（支持所有期货；针对股指期货，需满足 <strong>rule=1</strong> 或 <strong>rule=2</strong>）<br>3 - 次次主力合约（支持所有期货；针对股指期货，需满足 <strong>rule=1</strong> 或 <strong>rule=2</strong>）</td></tr><tr><td>market</td><td>str</td><td>默认是中国内地市场 ('cn') 。可选'cn' - 中国内地市场</td></tr></tbody></table>

#### 返回 [​](#rqdata-API-futures-get_ex_factor-return)

*   返回 pandas dataframe  包含了复权因子的日期和对应的各项数值

<table tabindex="0"><thead><tr><th>参数</th><th>类型</th><th>说明</th></tr></thead><tbody><tr><td>ex_date</td><td><em>pandas.Timestamp</em></td><td>除权除息日（ 主力合约切换日）</td></tr><tr><td>underlying_symbol</td><td><em>str</em></td><td>品种代码</td></tr><tr><td>ex_factor</td><td><em>float</em></td><td>复权因子</td></tr><tr><td>ex_cum_factor</td><td><em>float</em></td><td>累计复权因子</td></tr><tr><td>ex_end_date</td><td><em>pandas.Timestamp</em></td><td>复权因子所在期的截止日期</td></tr></tbody></table>

#### 范例 [​](#rqdata-API-futures-get_ex_factor-example)

python

```
[In]
futures.get_ex_factor(underlying_symbols='IF', start_date=20210601, end_date=20210902,adjust_method='prev_close_spread', market='cn')
[Out]
	underlying_symbol	ex_factor	ex_end_date	ex_cum_factor
ex_date
2021-06-18	IF	        32.8	        2021-07-15	1165.8
2021-07-16	IF	        16.8	        2021-08-12	1182.6
2021-08-13	IF	        29.0	        NaT	        1211.6
```

### futures.get_contract_multiplier - 获取期货品种合约乘数 [​](#rqdata-API-futures-get_contract_multiplier)

python

```
futures.get_contract_multiplier(underlying_symbols, start_date=None, end_date=None, market='cn')
```

获取期货品种的合约乘数数据，支持按时间范围查询。

#### 参数 [​](#rqdata-API-futures-get_contract_multiplier-params)

<table tabindex="0"><thead><tr><th>参数</th><th>类型</th><th>说明</th></tr></thead><tbody><tr><td>underlying_symbols</td><td><em>str</em> or <em>str list</em></td><td><strong>必填参数</strong>，期货合约品种</td></tr><tr><td>start_date</td><td><em>int, str, datetime.date, datetime.datetime, pandas.Timestamp</em></td><td>开始日期，不传入时返回所有数据</td></tr><tr><td>end_date</td><td><em>int, str, datetime.date, datetime.datetime, pandas.Timestamp</em></td><td>结束日期，不传入时返回所有数据</td></tr><tr><td>market</td><td><em>str</em></td><td>目前只支持中国市场 ('cn')</td></tr></tbody></table>

#### 返回 [​](#rqdata-API-futures-get_contract_multiplier-return)

*   _pandas DataFrame_

<table tabindex="0"><thead><tr><th>字段</th><th>类型</th><th>说明</th></tr></thead><tbody><tr><td>underlying_symbol</td><td><em>str</em></td><td>期货合约品种</td></tr><tr><td>date</td><td><em>pandas.Timestamp</em></td><td>交易日期</td></tr><tr><td>exchange</td><td><em>str</em></td><td>期货品种对应交易所</td></tr><tr><td>contract_multiplier</td><td><em>float</em></td><td>合约乘数</td></tr></tbody></table>

#### 范例 [​](#rqdata-API-futures-get_contract_multiplier-example)

python

```
In[]:
futures.get_contract_multiplier(['FB','I'], start_date='20191128', end_date='20191203', market='cn')
Out[]:
		                          exchange	contract_multiplier
underlying_symbol	date
FB	              2019-11-28	DCE	      500.0
                  2019-11-29	DCE	      500.0
                  2019-12-02	DCE	      10.0
                  2019-12-03	DCE	      10.0
I	              2019-11-28	DCE	      100.0
                  2019-11-29	DCE	      100.0
                  2019-12-02	DCE	      100.0
                  2019-12-03	DCE	      100.0
```

### futures.get_exchange_daily - 获取期货交易所日线数据 [​](#rqdata-API-futures-get_exchange_daily)

python

```
futures.get_exchange_daily(order_book_ids, start_date=None, end_date=None, fields=None,  market='cn')
```

获取期货交易所日线数据，支持按合约、时间范围和字段查询。

#### 参数 [​](#rqdata-API-futures-get_exchange_daily-params)

<table tabindex="0"><thead><tr><th>参数</th><th>类型</th><th>说明</th></tr></thead><tbody><tr><td>order_book_ids</td><td><em>str</em> or <em>str list</em></td><td><strong>必填参数</strong>，可输入 order_book_id, order_book_id list</td></tr><tr><td>start_date</td><td><em>int, str, datetime.date, datetime.datetime, pandas.Timestamp</em></td><td>开始日期</td></tr><tr><td>end_date</td><td><em>int, str, datetime.date, datetime.datetime, pandas.Timestamp</em></td><td>结束日期，start_date ,end_date 不传参数时默认返回最近三个月的数据</td></tr><tr><td>fields</td><td><em>str OR str list</em></td><td>查询字段，可选字段见下方返回，默认返回所有字段</td></tr><tr><td>market</td><td><em>str</em></td><td>默认是中国内地市场 ('cn') 。可选'cn' - 中国内地市场</td></tr></tbody></table>

#### 返回 [​](#rqdata-API-futures-get_exchange_daily-return)

_pandas DataFrame_

<table tabindex="0"><thead><tr><th>字段</th><th>类型</th><th>说明</th></tr></thead><tbody><tr><td>order_book_id</td><td><em>str</em></td><td>开盘价</td></tr><tr><td>date</td><td><em>float</em></td><td>交易日期</td></tr><tr><td>open</td><td><em>float</em></td><td>开盘价</td></tr><tr><td>close</td><td><em>float</em></td><td>收盘价</td></tr><tr><td>high</td><td><em>float</em></td><td>最高价</td></tr><tr><td>low</td><td><em>float</em></td><td>最低价</td></tr><tr><td>total_turnover</td><td><em>float</em></td><td>成交额</td></tr><tr><td>volume</td><td><em>float</em></td><td>成交量</td></tr><tr><td>settlement</td><td><em>float</em></td><td>结算价</td></tr><tr><td>prev_settlement</td><td><em>float</em></td><td>昨日结算价</td></tr><tr><td>open_interest</td><td><em>float</em></td><td>累计持仓量</td></tr></tbody></table>

#### 范例 [​](#rqdata-API-futures-get_exchange_daily-example)

python

```
In[]:
futures.get_exchange_daily('A2409', start_date='20240801', end_date='20240816', market='cn')
Out[]:
		                          open   close    high     low  total_turnover    volume  settlement  prev_settlement  open_interest
order_book_id date
A2409         2024-08-01  4549.0  4576.0  4581.0  4530.0    5.480035e+09  120316.0      4554.0           4543.0       102030.0
              2024-08-02  4580.0  4614.0  4634.0  4573.0    6.643340e+09  144204.0      4606.0           4554.0       104363.0
              2024-08-05  4617.0  4590.0  4628.0  4582.0    4.818868e+09  104612.0      4606.0           4606.0       100831.0
              2024-08-06  4584.0  4586.0  4602.0  4560.0    4.223373e+09   92163.0      4582.0           4606.0        90101.0
              2024-08-07  4586.0  4582.0  4601.0  4569.0    3.611244e+09   78789.0      4583.0           4582.0        85198.0
              2024-08-08  4572.0  4585.0  4593.0  4569.0    2.368785e+09   51666.0      4584.0           4583.0        79301.0
              2024-08-09  4584.0  4575.0  4598.0  4553.0    4.023708e+09   87879.0      4578.0           4584.0        73742.0
              2024-08-12  4575.0  4559.0  4578.0  4553.0    2.552710e+09   55909.0      4565.0           4578.0        67254.0
              2024-08-13  4559.0  4540.0  4565.0  4536.0    2.781623e+09   61125.0      4550.0           4565.0        62705.0
              2024-08-14  4538.0  4513.0  4551.0  4500.0    3.146009e+09   69587.0      4520.0           4550.0        54827.0
              2024-08-15  4520.0  4487.0  4536.0  4482.0    2.088194e+05   46333.0      4506.0           4520.0        49281.0
              2024-08-16  4488.0  4464.0  4507.0  4449.0    1.966178e+05   43952.0      4473.0           4506.0        38939.0
```

### futures.get_continuous_contracts - 获取期货当月等连续合约 [​](#rqdata-API-futures.get_continuous_contracts)

python

```
futures.get_continuous_contracts(underlying_symbol, start_date, end_date, type='front_month', market='cn')
```

获取股指期货和商品期货的连续合约数据。例如类型可选 `front_month`、`next_month`，股指期货还支持 `current_quarter` 和 `next_quarter`。

#### 参数 [​](#rqdata-API-futures-get_continuous_contracts-params)

<table tabindex="0"><thead><tr><th>参数</th><th>类型</th><th>说明</th></tr></thead><tbody><tr><td>underlying_symbol</td><td><em>str</em></td><td><strong>必填参数</strong>，期货合约品种，目前仅支持<b>股指、商品期货</b></td></tr><tr><td>start_date</td><td><em>int, str, datetime.date, datetime.datetime, pandas.Timestamp</em></td><td>开始日期</td></tr><tr><td>end_date</td><td><em>int, str, datetime.date, datetime.datetime, pandas.Timestamp</em></td><td>结束日期</td></tr><tr><td>type</td><td><em>str</em></td><td>类型，默认为 front_month<br>front_month - 近月（支持股指、商品期货）<br>next_month - 次月（支持股指、商品期货）<br>current_quarter - 季月（仅支持股指期货）<br>next_quarter - 远季（仅支持股指期货）</td></tr><tr><td>market</td><td><em>str</em></td><td>默认是中国内地市场 ('cn') 。可选'cn' - 中国内地市场；</td></tr></tbody></table>

#### 返回 [​](#rqdata-API-futures-get_continuous_contracts-return)

_Pandas Series_ - 连续合约 order_book_id

#### 范例 [​](#rqdata-API-futures-get_continuous_contracts-example)

*   获取 IF 一段时间内的近月合约

python

```
In[]:
futures.get_continuous_contracts('IF', 20250616, 20250701,'front_month')
Out[]:
date
2025-06-16    IF2506
2025-06-17    IF2506
2025-06-18    IF2506
2025-06-19    IF2506
2025-06-20    IF2506
2025-06-23    IF2507
2025-06-24    IF2507
2025-06-25    IF2507
2025-06-26    IF2507
2025-06-27    IF2507
2025-06-30    IF2507
2025-07-01    IF2507
Name: order_book_id, dtype: object
```

*   获取 CU 一段时间内的次月合约

python

```
In[]:
futures.get_continuous_contracts('CU', 20250612, 20250620,'next_month')
Out[]:
date
2025-06-12    CU2507
2025-06-13    CU2507
2025-06-16    CU2507
2025-06-17    CU2508
2025-06-18    CU2508
2025-06-19    CU2508
2025-06-20    CU2508
Name: order_book_id, dtype: object
```

期货其他数据 [​](#rqdata-API-futures-ext)
-----------------------------------

### futures.get_member_rank - 获取期货会员持仓等排名情况 [​](#rqdata-API-futures-get_member_rank)

python

```
futures.get_member_rank(obj, trading_date=None, rank_by='volume')
```

获取期货某合约或品种的会员排名数据。例如可按成交量、持买仓量、持卖仓量查询排名，分别对应 `volume`、`long`、`short`。

#### 参数 [​](#rqdata-API-futures-get_member_rank-params)

<table tabindex="0"><thead><tr><th>参数</th><th>类型</th><th>说明</th></tr></thead><tbody><tr><td>obj</td><td><em>str</em></td><td><strong>必填参数</strong>，可以是期货的具体合约或者品种</td></tr><tr><td>trading_date</td><td><em>int, str, datetime.date, datetime.datetime, pandas.Timestamp</em></td><td>查询日期，默认为当日</td></tr><tr><td>start_date</td><td><em>int, str, datetime.date, datetime.datetime, pandas.Timestamp</em></td><td>开始日期。需要传入该参数时，必须打上'start_date='字样</td></tr><tr><td>end_date</td><td><em>int, str, datetime.date, datetime.datetime, pandas.Timestamp</em></td><td>结束日期。需要传入该参数时，必须打上'end_date='字样</td></tr><tr><td>rank_by</td><td><em>str</em></td><td>排名依据，默认为 volume<br>volume - 交易量统计排名，long - 持买仓量统计排名，short - 持卖仓量统计排名</td></tr></tbody></table>

#### 返回 [​](#rqdata-API-futures-get_member_rank-return)

-_pandas DataFrame_

<table tabindex="0"><thead><tr><th>字段</th><th>类型</th><th>说明</th></tr></thead><tbody><tr><td>commodity_id</td><td><em>str</em></td><td>期货品种代码或期货合约代码</td></tr><tr><td>member_name</td><td><em>str</em></td><td>期货商名称</td></tr><tr><td>rank</td><td><em>int</em></td><td>排名</td></tr><tr><td>volume</td><td><em>float</em></td><td>交易量或持仓量视乎参数 rank_by 的设定</td></tr><tr><td>volume_change</td><td><em>float</em></td><td>交易量或持仓量较之前的变动</td></tr></tbody></table>

#### 范例 [​](#rqdata-API-futures-get_member_rank-example)

*   获取期货合约为标的的会员排名：

python

```
[In]
futures.get_member_rank('A1901',trading_date=20180910,rank_by='short')
[Out]
 	      commodity_id 	member_name 	rank 	volume 	volume_change
trading_date
2018-09-10 	A1901 	     国投安信 	     1 	     20143 	5065
2018-09-10 	A1901 	     五矿经易 	     2 	     14909 	4465
2018-09-10 	A1901 	     华安期货 	     3 	     9360 	3464
2018-09-10 	A1901 	     国泰君安 	     4 	     7915 	-26
2018-09-10 	A1901 	     永安期货 	     5 	     6683 	998
2018-09-10 	A1901 	     中信期货 	     6 	     6587 	-583
2018-09-10 	A1901 	     华泰期货 	     7 	     5918 	-430
2018-09-10 	A1901 	     东证期货 	     8 	     5075 	1837
2018-09-10 	A1901 	     中国国际 	     9 	     4792 	2169
2018-09-10 	A1901 	     国富期货 	     10 	   4632 	-213
2018-09-10 	A1901 	     浙商期货 	     11 	   4160 	-513
2018-09-10 	A1901 	     新湖期货 	     12 	   3960 	119
2018-09-10 	A1901 	     中金期货 	     13 	   3868 	-25
2018-09-10 	A1901 	     光大期货 	     14 	   3694 	2566
2018-09-10 	A1901 	     摩根大通 	     15 	   3644 	0
2018-09-10 	A1901 	     银河期货 	     16 	   3173 	559
2018-09-10 	A1901 	     兴证期货 	     17 	   3151 	-251
2018-09-10 	A1901 	     方正中期 	     18 	   2206 	146
2018-09-10 	A1901 	     一德期货 	     19 	   2017 	838
2018-09-10 	A1901 	     南华期货 	     20 	   1949 	-190
```

*   获取期货品种为标的的会员排名：

python

```
[In]
futures.get_member_rank('CU',trading_date=20180910,rank_by='short')
[Out]
        commodity_id 	member_name 	rank 	volume 	volume_change
trading_date
2018-09-10 	CU 	        五矿经易 	       1 	29160 	 302
2018-09-10 	CU 	        中信期货 	       2 	27136 	 -535
2018-09-10 	CU 	        永安期货 	       3 	16521 	 753
2018-09-10 	CU 	        海通期货 	       4 	15994 	 -161
2018-09-10 	CU 	        中粮期货 	       5 	14572 	 -614
2018-09-10 	CU 	        国泰君安 	       6 	8852 	   245
2018-09-10 	CU 	        金瑞期货 	       7 	8668 	   -703
2018-09-10 	CU 	        迈科期货 	       8 	8320 	   94
2018-09-10 	CU 	        建信期货 	       9 	6688 	   -57
2018-09-10 	CU 	        广发期货 	      10 	5847 	   -34
2018-09-10 	CU 	        华安期货 	      11 	5451 	   -289
2018-09-10 	CU 	        格林大华 	      12 	5330 	   -217
2018-09-10 	CU 	        中银国际 	      13 	5190 	   487
2018-09-10 	CU 	        铜冠金源 	      14 	4896 	   139
2018-09-10 	CU 	        兴证期货 	      15 	4636 	   -459
2018-09-10 	CU 	        方正中期 	      16 	4587 	   -92
2018-09-10 	CU 	        国投安信 	      17 	4567 	   79
2018-09-10 	CU 	        东方财富 	      18 	4551 	   127
2018-09-10 	CU 	        新湖期货 	      19 	4269 	   -60
2018-09-10 	CU 	        国贸期货 	      20 	3396 	   -522
```

*   获取期货品种为标的的指定时间段会员排名，_单个合约的指定时间段会员排名获取方式雷同_

python

```
[In] :
futures.get_member_rank('RB',start_date='20180910',end_date='20180915').tail(5)
[Out]:
             commodity_id member_name  rank  volume  volume_change
trading_date
2018-09-14             RB        中辉期货    16   53761          -5682
2018-09-14             RB        宏源期货    17   51777          -9398
2018-09-14             RB        广发期货    18   51279          -2157
2018-09-14             RB        国贸期货    19   51145         -23497
2018-09-14             RB        上海中期    20   47731          -4862
```

### futures.get_warehouse_stocks - 获取期货仓单数据 [​](#rqdata-API-futures-get_warehouse_stocks)

python

```
futures.get_warehouse_stocks(underlying_symbols, start_date=None, end_date=None, market='cn')
```

获取期货品种的注册仓单数据，支持按时间范围查询。

#### 参数 [​](#rqdata-API-futures-get_warehouse_stocks-params)

<table tabindex="0"><thead><tr><th>参数</th><th>类型</th><th>说明</th></tr></thead><tbody><tr><td>underlying_symbols</td><td><em>str</em> or <em>str list</em></td><td><strong>必填参数</strong>，期货合约品种，可传入 &nbsp;underlying_symbol, underlying_symbol list</td></tr><tr><td>start_date</td><td><em>int, str, datetime.date, datetime.datetime, pandas.Timestamp</em></td><td>开始日期</td></tr><tr><td>end_date</td><td><em>int, str, datetime.date, datetime.datetime, pandas.Timestamp</em></td><td>结束日期，start_date ,end_date 不传参数时默认返回最近一年的数据</td></tr><tr><td>market</td><td><em>str</em></td><td>目前只支持中国市场 ('cn')</td></tr></tbody></table>

#### 返回 [​](#rqdata-API-futures-get_warehouse_stocks-return)

*   _pandas DataFrame_

<table tabindex="0"><thead><tr><th>字段</th><th>类型</th><th>说明</th></tr></thead><tbody><tr><td>on_warrant</td><td><em>float</em></td><td>注册仓单量</td></tr><tr><td>exchange</td><td><em>str</em></td><td>期货品种对应交易所</td></tr><tr><td>effective_forecast</td><td><em>float</em></td><td>有效预报。仅支持郑商所（CZCE）合约</td></tr><tr><td>warrant_units</td><td><em>str</em></td><td>仓单单位。仅支持郑商所（CZCE）合约</td></tr><tr><td>contract_multiplier</td><td><em>float</em></td><td>合约乘数。部分品种/交易所会返回。</td></tr><tr><td>deliverable</td><td><em>float</em></td><td>符合交割品质的货物数量。仅支持上期所（SHFE）合约</td></tr></table>

#### 范例 [​](#rqdata-API-futures-get_warehouse_stocks-example)

python

```
In [4]: futures.get_warehouse_stocks('CF',start_date=20191201,end_date=20191205)
Out[4]:
                            on_warrant exchange  effective_forecast  warrant_units   deliverable
date     underlying_symbol
20191202 CF                      19425     CZCE                4753              8          NaN
20191203 CF                      19921     CZCE                4696              8          NaN
20191204 CF                      19997     CZCE                5005              8          NaN
20191205 CF                      20603     CZCE                4752              8          NaN
```

python

```
In [4]: futures.get_warehouse_stocks(['CF','SR'],start_date=20251120,end_date=20251120)
Out[4]:
                                on_warrant	exchange	effective_forecast	warrant_units	contract_multiplier	deliverable
date	   underlying_symbol						
20251120   CF	                3503.0	        CZCE	1147.0	                8	                5	        None
           SR	                7982.0	        CZCE	183.0	                1	                10	        None
```

### futures.get_basis - 获取股指期货每日升贴水数据 [​](#rqdata-API-futures-get_basis)

python

```
futures.get_basis(order_book_ids, start_date=None, end_date=None, fields=None, frequency='1d', dividend_adjusted=False, market='cn')
```

获取股指期货升贴水数据，支持日线、分钟线和 tick 级别查询。例如可查询每日升贴水数据，也支持分钟级和 tick 级升贴水数据。此处分红调整数据来源于 [API-get_predicted_dividend_point](https://www.ricequant.com/doc/rqdata/python/futures-mod#rqdata-API-futures-get_predicted_dividend_point)。

#### 参数 [​](#rqdata-API-futures-get_basis-params)

<table tabindex="0"><thead><tr><th>参数</th><th>类型</th><th>说明</th></tr></thead><tbody><tr><td>order_book_ids</td><td><em>str_or_str_list</em></td><td>合约代码，可传入 order_book_id, order_book_id list</td></tr><tr><td>start_date</td><td><em>int, str, datetime.date, datetime.datetime, pandas.Timestamp</em></td><td>开始日期</td></tr><tr><td>end_date</td><td><em>int, str, datetime.date, datetime.datetime, pandas.Timestamp</em></td><td>结束日期，start_date ,end_date 不传参数时默认返回最近三个月的数据</td></tr><tr><td>fields</td><td><em>str_or_str_list</em></td><td>查询字段，可选字段见下方返回，默认返回所有字段</td></tr><tr><td>frequency</td><td><em>str</em></td><td>频率，支持 / 日 / 分钟 / tick 级别的历史数据，默认为'1d'。<br>1d - 日线<br>1m - 分钟线<br>tick</td></tr><tr><td>dividend_adjusted</td><td><em>bool</em></td><td>是否进行分红调整，默认为 False</td></tr></tbody></table>

#### 返回 [​](#rqdata-API-futures-get_basis-return)

pandas multi-index DataFrame

<table tabindex="0"><thead><tr><th>字段</th><th>类型</th><th>说明</th></tr></thead><tbody><tr><td>open</td><td><em>float</em></td><td>开盘价</td></tr><tr><td>high</td><td><em>float</em></td><td>最高价</td></tr><tr><td>low</td><td><em>float</em></td><td>最低价</td></tr><tr><td>close</td><td><em>float</em></td><td>收盘价</td></tr><tr><td>index</td><td><em>str</em></td><td>指数合约</td></tr><tr><td>close_index</td><td><em>float</em></td><td>指数收盘价</td></tr><tr><td>basis</td><td><em>float</em></td><td>升贴水，等于期货合约收盘价 - 对应指数收盘价</td></tr><tr><td>basis_rate</td><td><em>float</em></td><td>升贴水率 (%)，（期货合约收盘价 - 对应指数收盘价）*100 / 对应指数收盘价</td></tr><tr><td>basis_annual_rate</td><td><em>float</em></td><td>年化升贴水率（%), basis_rate *(250 / 合约到期剩余交易日）</td></tr><tr><td>settlement</td><td><em>float</em></td><td>结算价</td></tr><tr><td>settle_basis</td><td><em>float</em></td><td>升贴水，等于期货合约结算价 - 对应指数收盘价</td></tr><tr><td>settle_basis_rate</td><td><em>float</em></td><td>升贴水率 (%)，（期货合约结算价 - 对应指数收盘价）*100 / 对应指数收盘价</td></tr><tr><td>settle_basis_annual_rate</td><td><em>float</em></td><td>年化升贴水率（%), settle_basis_rate*(250 / 合约到期剩余交易日）</td></tr></tbody></table>

#### 范例 [​](#rqdata-API-futures-get_basis-example)

*   获取 IF2106 和 IH2106 的升贴水数据

python

```
[In]
futures.get_basis(['IF2106','IH2106'],'20210412','20210413')
[Out]
		                low	open	high	basis	basis_rate	index	close	basis_annual_rate	close_index
order_book_id	date
IH2106	2021-04-12	3395.0	3438.0	3446.4	-64.2654	-1.848265	000016.XSHG	3412.8	-10.268142	3477.0654
        2021-04-13	3384.2	3430.2	3432.6	-61.4592	-1.775529	000016.XSHG	3400.0	-10.088231	3461.4592
IF2106	2021-04-12	4854.0	4938.2	4956.0	-81.7459	-1.652185	000300.XSHG	4866.0	-9.178804	4947.7459
        2021-04-13	4844.0	4891.2	4905.2	-74.2438	-1.503019	000300.XSHG	4865.4	-8.539882	4939.6438
```

### futures.get_current_basis - 获取股指期货实时升贴水数据 [​](#rqdata-API-futures-get_current_basis)

python

```
futures.get_current_basis(order_book_ids,dividend_adjusted=False,market='cn')
```

获取股指期货实时升贴水数据，计算逻辑与 `get_basis` 一致。此处分红调整数据来源于 [API-get_predicted_dividend_point](https://www.ricequant.com/doc/rqdata/python/futures-mod#rqdata-API-futures-get_predicted_dividend_point)，采用当日分红预测点位进行调整。

#### 参数 [​](#rqdata-API-futures-get_current_basis-params)

<table tabindex="0"><thead><tr><th>参数</th><th>类型</th><th>说明</th></tr></thead><tbody><tr><td>order_book_ids</td><td><em>str_or_str_list</em></td><td><strong>必填参数</strong>，合约代码，可传入 order_book_id, order_book_id list</td></tr><tr><td>dividend_adjusted</td><td><em>bool</em></td><td>是否进行分红调整，默认为 False</td></tr><tr><td>market</td><td><em>str</em></td><td>默认是中国内地市场 ('cn') 。可选'cn' - 中国内地市场；</td></tr></tbody></table>

#### 返回 [​](#rqdata-API-futures-get_current_basis-return)

pandas multi-index DataFrame

<table tabindex="0"><thead><tr><th>字段</th><th>类型</th><th>说明</th></tr></thead><tbody><tr><td>order_book_id</td><td><em>str</em></td><td>合约代码</td></tr><tr><td>datetime</td><td><em>pandas.Timestamp</em></td><td>最新一行 tick 的时间戳</td></tr><tr><td>index</td><td><em>str</em></td><td>指数合约</td></tr><tr><td>index_px</td><td><em>float</em></td><td>指数最新价格</td></tr><tr><td>future_px</td><td><em>float</em></td><td>期货最新价格</td></tr><tr><td>basis</td><td><em>float</em></td><td>升贴水，等于期货合约收盘价 - 对应指数收盘价</td></tr><tr><td>basis_rate</td><td><em>float</em></td><td>升贴水率 (%)，（期货合约收盘价 - 对应指数收盘价）*100 / 对应指数收盘价</td></tr><tr><td>basis_annual_rate</td><td><em>float</em></td><td>年化升贴水率（%), basis_rate *(250 / 合约到期剩余交易日）</td></tr></tbody></table>

#### 范例 [​](#rqdata-API-futures-get_current_basis-example)

*   获取 IF2403 的实时升贴水数据

python

```
[In]
futures.get_current_basis('IF2403')
[Out]
                     index                datetime   index_px  future_px   basis  basis_rate  basis_annual_rate
order_book_id
IF2403         000300.XSHG 2024-02-26 15:23:08.200  3453.3585     3445.4 -7.9585   -0.230457            -4.1153
```

### futures.get_trading_parameters - 获取期货交易参数信息 [​](#rqdata-API-futures-get_trading_parameters)

python

```
futures.get_trading_parameters(order_book_ids, start_date=None, end_date=None, fields=None, market='cn')
```

获取期货交易参数信息。例如可获取保证金、手续费、限仓等交易参数数据。

注意事项

*   start_date 和 end_date 需同时传入或同时不传入。当不传入 start_date , end_date 参数时，查询时间在交易日 T 日 6.30 pm 之前，返回 T 日的数据；查询时点在 6.30pm 之后，返回交易日 T+1 日的数据。
*   保证金、手续费数据提供范围为 2010.04 月至今；限仓数据各交易所提供范围见下方表格  
    

<table tabindex="0"><thead><tr><th>交易所</th><th>时间范围</th></tr></thead><tbody><tr><td>中金所</td><td>2010-04-16 至今</td></tr><tr><td>上期所</td><td>2013-10-08 至今</td></tr><tr><td>大商所</td><td>2018-12-21 至今</td></tr><tr><td>郑商所</td><td>2021-04-12 至今</td></tr><tr><td>上能源</td><td>2021-06-11 至今</td></tr><tr><td>广期所</td><td>2022-12-23 至今</td></tr></tbody></table>

#### 参数 [​](#rqdata-API-futures-get_trading_parameters-params)

<table tabindex="0"><thead><tr><th>参数</th><th>类型</th><th>说明</th></tr></thead><tbody><tr><td>order_book_ids</td><td><em>str&nbsp;or&nbsp;list</em></td><td><strong>必填参数</strong>，可输入 order_book_id, order_book_id list</td></tr><tr><td>start_date</td><td><em>int, str, datetime.date, datetime.datetime, pandas.Timestamp</em></td><td>开始日期，若不指定日期，则默认为当前交易日期</td></tr><tr><td>end_date</td><td><em>int, str, datetime.date, datetime.datetime, pandas.Timestamp</em></td><td>结束日期 ，若不指定日期，则默认为当前交易日期</td></tr><tr><td>fields</td><td><em>str OR str list</em></td><td>查询字段，可选字段见下方返回，默认返回所有字段</td></tr><tr><td>market</td><td><em>str</em></td><td>默认是中国内地市场 ('cn') 。可选'cn' - 中国内地市场</td></tr></tbody></table>

#### 返回 [​](#rqdata-API-futures-get_trading_parameters-return)

*   返回   multi-index DataFrame

<table tabindex="0"><thead><tr><th>字段</th><th>类型</th><th>说明</th><th>备注</th></tr></thead><tbody><tr><td>order_book_ids</td><td><em>str</em></td><td>合约代码</td><td></td></tr><tr><td>trading_date</td><td><em>pandas.Timestamp</em></td><td>交易日期</td><td></td></tr><tr><td>long_margin_ratio</td><td><em>float</em></td><td>多头保证金率</td><td>盘前更新</td></tr><tr><td>short_margin_ratio</td><td><em>float</em></td><td>空头保证金率</td><td>盘前更新</td></tr><tr><td>commission_type</td><td><em>str</em></td><td>手续费类型（按成交量 / 按成交额）</td><td>盘后更新</td></tr><tr><td>open_commission</td><td><em>float</em></td><td>开仓手续费</td><td>盘后更新</td></tr><tr><td>close_commission</td><td><em>float</em></td><td>平仓手续费</td><td>盘后更新</td></tr><tr><td>discount_rate</td><td><em>float</em></td><td>平今折扣率</td><td>盘后更新</td></tr><tr><td>close_commission_today</td><td><em>float</em></td><td>平今仓手续费 / 率</td><td>盘后更新</td></tr><tr><td>non_member_limit_rate</td><td><em>float</em></td><td>非期货会员持仓限额比例</td><td>郑商所盘后更新，盘中返回 Nan，其他商所盘前更新</td></tr><tr><td>client_limit_rate</td><td><em>float</em></td><td>客户持仓限额比例，盘前更新</td><td>同上</td></tr><tr><td>non_member_limit</td><td><em>float</em></td><td>非期货会员持仓限额 (手)</td><td>同上</td></tr><tr><td>client_limit</td><td><em>float</em></td><td>客户持仓限额 (手)</td><td>同上</td></tr><tr><td>min_order_quantity</td><td><em>float</em></td><td>最小开仓下单量 (手)</td><td>盘前更新</td></tr><tr><td>max_order_quantity</td><td><em>float</em></td><td>最大开仓下单量 (手)</td><td>盘前更新</td></tr><tr><td>min_margin_ratio</td><td><em>float</em></td><td>最低交易保证金</td><td>盘前更新</td></tr><tr><td>trade_unit</td><td><em>float</em></td><td>交易单位</td><td>盘前更新</td></tr><tr><td>price_unit</td><td><em>float</em></td><td>报价单位</td><td>盘前更新</td></tr></tbody></table>

#### 范例 [​](#rqdata-API-futures-get_trading_parameters-example)

*   获取 IF2312 当天的交易参数信息

python

```
[In]
futures.get_trading_parameters('IF2312')
[Out]
                            long_margin_ratio	short_margin_ratio	commission_type	open_commission	...	client_limit	min_order_quantity	max_order_quantity	min_margin_ratio
order_book_id	trading_date
IF2312	2023-12-05	0.12	0.12	by_money	0.000023	...	5000.0	1.0	NaN	0.08
```

*   获取 CU2312 指定日期的交易参数信息

python

```
[In]
futures.get_trading_parameters('CU2312',start_date=20231201,end_date=20231205)
[Out]
                            long_margin_ratio	short_margin_ratio	commission_type	open_commission	...	client_limit	min_order_quantity	max_order_quantity	min_margin_ratio
order_book_id	trading_date
CU2312	2023-12-01	0.15	0.15	by_money	0.00005	...	1000.0	1.0	NaN	0.05
2023-12-04	0.15	0.15	by_money	0.00005	...	1000.0	1.0	NaN	0.05
2023-12-05	0.15	0.15	by_money	0.00005	...	1000.0	1.0	NaN	0.05
```

### futures.get_roll_yield - 获取商品期货展期收益率数据 [​](#rqdata-API-futures-get_roll_yield)

python

```
futures.get_roll_yield(underlying_symbol, start_date=None, end_date=None, type='main_sub', rule=0, market='cn')
```

获取商品期货展期收益率数据。例如类型可选 `main_sub` 和 `near_main`，也支持不同主力合约选取规则。

#### 参数 [​](#rqdata-API-futures-get_roll_yield-params)

<table tabindex="0"><thead><tr><th>参数</th><th>类型</th><th>说明</th></tr></thead><tbody><tr><td>underlying_symbol</td><td><em>str&nbsp;or&nbsp;list</em></td><td><strong>必填参数</strong>，品种代码</td></tr><tr><td>start_date</td><td><em>int, str, datetime.date, datetime.datetime, pandas.Timestamp</em></td><td>开始日期，默认为 None，表示返回最近三个月的数据</td></tr><tr><td>end_date</td><td><em>int, str, datetime.date, datetime.datetime, pandas.Timestamp</em></td><td>结束日期 ，默认为 None，表示返回最近三个月的数据</td></tr><tr><td>type</td><td><em>str</em></td><td>'main_sub'：基于主力、次主力计算的展期收益率<br>'near_main'：基于近月、主力计算的展期收益率<br>默认为 main_sub</td></tr><tr><td>rule</td><td><em>int</em></td><td>主力合约选取规则。<br>默认 rule=0，当同品种其他合约持仓量在收盘后超过当前主力合约 1.1 倍时，从第二个交易日开始进行主力合约的切换。每个合约只能做一次主力 / 次主力合约，不会重复出现。针对股指期货，只在当月和次月选择主力合约。<br>当 rule=1 时，主力 / 次主力合约的选取只考虑最大 / 第二大昨仓这个条件。<br>当 rule=2 时，采用昨日成交量与持仓量同为最大 / 第二大的合约为当日主力 / 次主力。<br>当 rule=3 时，在 rule=0 选取规则上，考虑在最后一个交易日不能成为主力 / 次主力合约。</td></tr><tr><td>market</td><td><em>str</em></td><td>默认是中国内地市场 ('cn') 。可选'cn' - 中国内地市场</td></tr></tbody></table>

#### 返回 [​](#rqdata-API-futures-get_roll_yield-return)

<table tabindex="0"><thead><tr><th>字段</th><th>类型</th><th>说明</th></tr></thead><tbody><tr><td>underlying_symbol</td><td><em>str</em></td><td>品种代码，目前仅支持商品期货</td></tr><tr><td>date</td><td><em>pandas.Timestamp</em></td><td>交易日期</td></tr><tr><td>from_contract</td><td><em>str</em></td><td>滚动前的合约，取决于 type 参数<br>当 type 为 ‘main_sub’ 时，from_contract 为主力合约<br>当 type 为'near_main' 时，from_contract 为近月合约</td></tr><tr><td>to_contract</td><td><em>str</em></td><td>滚动后的合约，取决于 type 参数<br>当 type 为'main_sub' 时，to_contract 为次主力合约<br>当 type 为'near_main' 时，to_contract 为主力合约</td></tr><tr><td>yield</td><td><em>float</em></td><td>展期收益率<br>当 type 为'main_sub' 时，展期收益率 =（主力合约结算价－次主力合约结算价）/ 次主力合约结算价<br>当 type 为'near_main' 时，展期收益率 =（近月合约结算价－主力合约结算价）/ 主力合约价结算价</td></tr><tr><td>annualized_yield</td><td><em>float</em></td><td>自然日年化展期收益率（展期收益率 * (365 / 两个合约到期日之间的自然日差)）</td></tr><tr><td>annualized_yield_trading</td><td><em>float</em></td><td>交易日年化展期收益率（展期收益率 * (252 / 两个合约到期日之间的交易日差)）</td></tr></tbody></table>

#### 范例 [​](#rqdata-API-futures-get_roll_yield-example)

*   获取 CU 基于主力、次主力计算的一段时间的展期收益率

python

```
[In]
futures.get_roll_yield('CU',20250811, 20250815, type='main_sub', rule=0)
[Out]
                                from_contract	to_contract	yield	annualized_yield	annualized_yield_trading
underlying_symbol	date					
CU	               2025-08-11	CU2509	CU2510	-0.000127	-0.001541	-0.001878
                2025-08-12	CU2509	CU2510	-0.000380	-0.004622	-0.005631
                2025-08-13	CU2509	CU2510	-0.000504	-0.006131	-0.007470
                2025-08-14	CU2509	CU2510	0.000000	0.000000	0.000000
                2025-08-15	CU2509	CU2510	0.000127	0.001541	0.001878
```

### futures.get_predicted_dividend_point - 获取股指期货预测分红点位 [​](#rqdata-API-futures-get_predicted_dividend_point)

python

```
futures.get_predicted_dividend_point(order_book_id，start_date, end_date)
```

计算逻辑说明

分红点位成分股每股分红金额成分股股价成分股权重指数收盘价

其中，＜ 的成分股分红点位才会被计算在内， 是成分股  的除权除息日。

此处，若公司已公布具体分红方案，则采用实际分红金融及除权除息日纳入计算。若公司未公布具体分红方案则会对其进行预测，分红金额的预测存在以下两种情况： (1) 若公司已公布分红预案或是决案，分红金额将采用已公布方案中的数据；  
(2) 若未公布则按下述公式进行预测，净利润采用最新披露的年报 / 业绩快报 / 业绩预告用于计算，若无对应数据则为过去三年对应季度的平均值，股息支付率则始终为过去三年的平均值。

每股分红金额预测分红季度净利润过去三年平均股息支付率预测日总股本股息支付率现金分红总额净利润

对于除权除息日的预测同样存在两种情况： (1) 若公司已公布分红预案或是决案，除权除息日为方案公布日 + avg(gap)；  
(2) 若未公布，则认为该年该股票分红情况同历史最近年度的分红情况一致.

#### 参数 [​](#rqdata-API-futures-get_predicted_dividend_point-params)

<table tabindex="0"><thead><tr><th>参数</th><th>类型</th><th>说明</th></tr></thead><tbody><tr><td>order_book_id</td><td><em>str&nbsp;or&nbsp;str_list</em></td><td><strong>必填参数</strong>，合约代码</td></tr><tr><td>start_date</td><td><em>int, str, datetime.date, datetime.datetime, pandas.Timestamp</em></td><td>开始日期</td></tr><tr><td>end_date</td><td><em>int, str, datetime.date, datetime.datetime, pandas.Timestamp</em></td><td>结束日期</td></tr></tbody></table>

#### 返回 [​](#rqdata-API-futures-get_predicted_dividend_point-return)

_pandas DataFrame_

<table tabindex="0"><thead><tr><th>字段</th><th>类型</th><th>说明</th></tr></thead><tbody><tr><td>order_book_id</td><td><em>str</em></td><td>合约代码</td></tr><tr><td>date</td><td><em>pandas.Timestamp</em></td><td>交易日期</td></tr><tr><td>dividend_points</td><td><em>float</em></td><td>预测分红点位</td></tr></tbody></table>

#### 范例 [​](#rqdata-API-futures-get_predicted_dividend_point-example)

*   获取 ['IH2512', 'IH2601', 'IH2603', 'IH2606'] 在 2025-11-28 的预测分红点位

python

```
[In]
futures.get_predicted_dividend_point(['IH2512', 'IH2601', 'IH2603', 'IH2606'], 20251128, 20251128)
[Out]

                                dividend_points
order_book_id	   date	
IH2512	        2025-11-28	4.214274
IH2601	        2025-11-28	6.006543
IH2603	        2025-11-28	15.350197
IH2606	        2025-11-28	34.223083
```
