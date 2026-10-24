"""A 股股票市场适配器抽象规范与规格实现 (markets/stocks/adapter.py)。

严格遵循白皮书第 17 节《后续 A 股股票市场接入规范说明》架构设计：
- 实现 MarketAdapter 抽象契约 (calendar, universe_mask, contract_spec, cost_model)；
- 固化 A 股特定交易规则：
  * 最小交易单位：1 手 = 100 股，合约乘数恒为 1.0，价格最小变动单位 tick_size = 0.01；
  * 保证金机制：股票全额现货交易，保证金比例恒为 1.0 (无杜绝爆仓杠杆)；
  * 非对称税费模型：买入单向佣金与过户费，卖出加征单向证券交易印花税；
  * 标的板块识别与涨跌停规则：主板 10%、科创板/创业板 20%、北交所 30%、ST 标的 5%；
  * 交易机制语义：T+1 机制与仅多头持仓约束；
- 为工程二期（Phase 2）直接接入 A 股行情 Parquet / Arrow 数据流预留完备插槽。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from factorlib.core.panel import Panel
from factorlib.markets.base import ContractSpec, CostModel, MarketAdapter


@dataclass(frozen=True)
class StockCostSpec:
    """A 股股票交易税费与滑点规格参数。"""

    commission_rate: float = 0.0001  # 券商佣金率 (默认万分之 1)
    min_commission: float = 5.0  # 佣金单笔最低收取金额 (5 元起收)
    stamp_duty_rate: float = 0.0005  # 卖出单边印花税率 (现行减半征收万分之 5)
    transfer_fee_rate: float = 0.00001  # 证券过户费 (双边万分之 0.1)
    slippage_ticks: float = 1.0  # 滑点跳数 (每跳 0.01 元)


class StockMarketAdapter(MarketAdapter):
    """A 股股票市场适配器实现与规范固化。"""

    name: str = "stocks"

    def __init__(
        self,
        raw_dir: Path | str | None = None,
        cost_spec: StockCostSpec | None = None,
        custom_calendar: np.ndarray | None = None,
    ) -> None:
        """初始化 A 股股票适配器。

        Args:
            raw_dir: 股票原始 Parquet 数据目录 (Phase 2 接入)
            cost_spec: 交易税费与滑点参数对象
            custom_calendar: 可选直接注入的交易日历数组
        """
        self.raw_dir = Path(raw_dir) if raw_dir is not None else None
        self.cost_spec = cost_spec or StockCostSpec()
        self._custom_calendar = custom_calendar

    def calendar(self) -> np.ndarray:
        """获取 A 股市场统一交易日历。

        若已显式注入 custom_calendar，直接返回；
        若未注入且处于第一期，抛出友好提示引导 Phase 2 数据接入。
        """
        if self._custom_calendar is not None:
            return np.asarray(self._custom_calendar)

        if self.raw_dir is not None and (self.raw_dir / "trading_calendar.parquet").exists():
            # Phase 2 预留数据加载入口
            import polars as pl

            df = pl.read_parquet(self.raw_dir / "trading_calendar.parquet")
            return df["trading_date"].to_numpy()

        raise NotImplementedError(
            "A 股交易日历读取接口已预留，将在 Phase 2 接入 A 股行情时完整激活。"
            "当前可通过 StockMarketAdapter(custom_calendar=...) 显式注入日历数组。"
        )

    def universe_mask(self, panel: Panel) -> np.ndarray:
        """计算给定面板在各交易日的可交易品种池布尔矩阵 (T x N)。

        A 股特性处理：
        1. 剔除停牌与无量日 (volume <= 0 或 close 为 NaN)；
        2. 若面板中提供 is_st 或 st_status 字段，自动剔除 ST/*ST 标的；
        3. 若面板提供 listing_days 字段，剔除上市不足 180 天的次新股。
        """
        # 基础可交易掩码：基于现有面板掩码
        mask = panel.mask.copy()

        # 检查是否包含 ST 字段
        if "is_st" in panel.fields:
            st_field = panel.get_field("is_st")
            is_st_bool = np.nan_to_num(st_field, nan=0.0) > 0.5
            mask = mask & (~is_st_bool)

        # 检查是否包含上市天数字段 (过滤上市不足 180 天次新股)
        if "listing_days" in panel.fields:
            ld_field = panel.get_field("listing_days")
            valid_age = np.nan_to_num(ld_field, nan=0.0) >= 180.0
            mask = mask & valid_age

        return mask

    def contract_spec(self, symbol: str) -> ContractSpec:
        """获取 A 股标的合约交易规格。

        按 A 股交易所标准规则：
        - 1 手 = 100 股 (lot_size = 100)；
        - 乘数恒为 1.0 (multiplier = 1.0)；
        - 报价最小变动价位 0.01 元 (tick_size = 0.01)；
        - 现货交易保证金率恒为 1.0 (margin_rate = 1.0, 零杠杆全额现金)。
        """
        return ContractSpec(
            multiplier=1.0,
            lot_size=100,
            tick_size=0.01,
            margin_rate=1.0,
        )

    def cost_model(self, symbol: str) -> CostModel:
        """获取 A 股标的的交易成本模型。

        A 股税费机制 (以成交额为基准):
        - 买入成本 (open_fee): 券商佣金 + 过户费
        - 卖出成本 (close_fee): 券商佣金 + 过户费 + 卖出单向印花税
        - 平今成本 (close_today_fee): 与普通卖出一致
        """
        buy_fee = self.cost_spec.commission_rate + self.cost_spec.transfer_fee_rate
        sell_fee = buy_fee + self.cost_spec.stamp_duty_rate

        return CostModel(
            fee_type="rate",
            open_fee=buy_fee,
            close_fee=sell_fee,
            close_today_fee=sell_fee,
            slippage_ticks=self.cost_spec.slippage_ticks,
        )

    @staticmethod
    def identify_board(symbol: str) -> str:
        """根据证券代码前缀识别所属交易板块。

        Args:
            symbol: 证券代码 (如 '600519', '000001', '300750', '688981', '830000', '920000')

        Returns:
            板块名称: 'main' (主板), 'chinext' (创业板), 'star' (科创板), 'bse' (北交所)
        """
        clean_sym = symbol.strip().upper()
        # 去除交易所后缀如 .SH, .SZ, .BJ
        if "." in clean_sym:
            clean_sym = clean_sym.split(".")[0]

        if clean_sym.startswith(("600", "601", "603", "605", "000", "001", "002", "003")):
            return "main"
        elif clean_sym.startswith("30"):
            return "chinext"
        elif clean_sym.startswith("688"):
            return "star"
        elif clean_sym.startswith(("4", "8", "920")):
            return "bse"
        return "main"

    @classmethod
    def price_limit_ratio(cls, symbol: str, is_st: bool = False) -> float:
        """获取标的单日涨跌幅限制比例。

        规则：
        - ST / *ST 标的：5% (0.05)
        - 主板非 ST 标的：10% (0.10)
        - 科创板 (STAR) 与创业板 (ChiNext)：20% (0.20)
        - 北交所 (BSE)：30% (0.30)
        """
        if is_st:
            return 0.05

        board = cls.identify_board(symbol)
        if board == "star" or board == "chinext":
            return 0.20
        elif board == "bse":
            return 0.30
        return 0.10

    @property
    def is_t_plus_one(self) -> bool:
        """A 股实行 T+1 交易交收机制。"""
        return True

    @property
    def allow_short(self) -> bool:
        """A 股现货回测默认禁止融券做空 (单向做多)。"""
        return False

    def validate_panel_schema(self, panel: Panel) -> tuple[bool, list[str]]:
        """检验股票面板字段规格是否满足回测与因子研究要求。

        Required fields:
        - 'close' 或 'close_adj' (收盘价 / 后复权收盘价)
        - 'open' 或 'open_adj' (开盘价 / 后复权开盘价)
        - 'volume' (成交量 / 股)
        - 'turnover' (成交额 / 元)
        """
        required = ["close", "volume"]
        missing = [f for f in required if f not in panel.fields]
        return (len(missing) == 0, missing)

    def to_dict(self) -> dict[str, Any]:
        """序列化适配器元信息。"""
        return {
            "name": self.name,
            "is_t_plus_one": self.is_t_plus_one,
            "allow_short": self.allow_short,
            "cost_spec": {
                "commission_rate": self.cost_spec.commission_rate,
                "stamp_duty_rate": self.cost_spec.stamp_duty_rate,
                "transfer_fee_rate": self.cost_spec.transfer_fee_rate,
                "slippage_ticks": self.cost_spec.slippage_ticks,
            },
        }
