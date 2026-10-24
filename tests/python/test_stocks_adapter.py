"""A 股股票市场适配器规格与契约单元测试 (tests/python/test_stocks_adapter.py)。

按白皮书第 17 节设计，验证股票市场适配器的规范固化与接口一致性。
"""

from __future__ import annotations

import numpy as np
import pytest
from factorlib.core.panel import Panel
from factorlib.markets.base import ContractSpec, CostModel, MarketAdapter
from factorlib.markets.stocks.adapter import StockCostSpec, StockMarketAdapter


def test_stock_adapter_inheritance_and_contract_spec() -> None:
    adapter = StockMarketAdapter()
    assert isinstance(adapter, MarketAdapter)
    assert adapter.name == "stocks"
    assert adapter.is_t_plus_one is True
    assert adapter.allow_short is False

    spec = adapter.contract_spec("600519")
    assert isinstance(spec, ContractSpec)
    assert spec.multiplier == 1.0
    assert spec.lot_size == 100
    assert spec.tick_size == 0.01
    assert spec.margin_rate == 1.0


def test_stock_cost_model_asymmetry() -> None:
    cost_spec = StockCostSpec(
        commission_rate=0.0001,
        stamp_duty_rate=0.0005,
        transfer_fee_rate=0.00001,
        slippage_ticks=1.0,
    )
    adapter = StockMarketAdapter(cost_spec=cost_spec)
    cost = adapter.cost_model("300750")

    assert isinstance(cost, CostModel)
    assert cost.fee_type == "rate"
    # 买入: 佣金 + 过户费
    assert pytest.approx(cost.open_fee, rel=1e-6) == 0.00011
    # 卖出: 佣金 + 过户费 + 卖出单向印花税
    assert pytest.approx(cost.close_fee, rel=1e-6) == 0.00061
    assert pytest.approx(cost.close_today_fee, rel=1e-6) == 0.00061
    assert cost.slippage_ticks == 1.0


def test_board_identification() -> None:
    assert StockMarketAdapter.identify_board("600519") == "main"
    assert StockMarketAdapter.identify_board("600519.SH") == "main"
    assert StockMarketAdapter.identify_board("000001.SZ") == "main"
    assert StockMarketAdapter.identify_board("002415") == "main"
    assert StockMarketAdapter.identify_board("300750.SZ") == "chinext"
    assert StockMarketAdapter.identify_board("688981.SH") == "star"
    assert StockMarketAdapter.identify_board("830000.BJ") == "bse"
    assert StockMarketAdapter.identify_board("920001") == "bse"


def test_price_limit_ratio() -> None:
    # 主板
    assert StockMarketAdapter.price_limit_ratio("600519") == 0.10
    assert StockMarketAdapter.price_limit_ratio("000001") == 0.10
    # 创业板与科创板
    assert StockMarketAdapter.price_limit_ratio("300750") == 0.20
    assert StockMarketAdapter.price_limit_ratio("688981") == 0.20
    # 北交所
    assert StockMarketAdapter.price_limit_ratio("830000") == 0.30
    # 任意板块 ST
    assert StockMarketAdapter.price_limit_ratio("600519", is_st=True) == 0.05
    assert StockMarketAdapter.price_limit_ratio("300750", is_st=True) == 0.05
    assert StockMarketAdapter.price_limit_ratio("688981", is_st=True) == 0.05


def test_calendar_handling() -> None:
    # 默认未指定且无 Phase 2 数据时抛出清晰的 NotImplementedError
    adapter = StockMarketAdapter()
    with pytest.raises(NotImplementedError, match="A 股交易日历读取接口已预留"):
        adapter.calendar()

    # 显式注入交易日历
    custom_dates = np.array([20240102, 20240103, 20240104])
    adapter_with_cal = StockMarketAdapter(custom_calendar=custom_dates)
    cal = adapter_with_cal.calendar()
    np.testing.assert_array_equal(cal, custom_dates)


def test_universe_mask_with_st_and_listing_days() -> None:
    # 构造小型测试面板 (T=4, N=2)
    dates = np.array([20240102, 20240103, 20240104, 20240105])
    symbols = ["000001", "300750"]
    mask = np.ones((4, 2), dtype=bool)

    # 字段包含 is_st 与 listing_days
    # symbol 0: 正常股票, 上市天数 > 180
    # symbol 1: 第 2 天被标记为 ST (row 1), 且上市天数仅 100 天 (全部行 < 180)
    close = np.full((4, 2), 10.0, order="F")
    volume = np.full((4, 2), 1000.0, order="F")
    is_st = np.array([[0.0, 0.0], [0.0, 1.0], [0.0, 1.0], [0.0, 0.0]], dtype=np.float64, order="F")
    listing_days = np.array(
        [[200.0, 100.0], [201.0, 101.0], [202.0, 102.0], [203.0, 103.0]],
        dtype=np.float64,
        order="F",
    )

    panel = Panel(
        dates=dates,
        symbols=symbols,
        fields={
            "close": close,
            "volume": volume,
            "is_st": is_st,
            "listing_days": listing_days,
        },
        mask=mask,
    )

    adapter = StockMarketAdapter()
    filtered_mask = adapter.universe_mask(panel)

    # symbol 0 全部通过过滤
    assert np.all(filtered_mask[:, 0])
    # symbol 1 因上市不足 180 天全部被过滤掉
    assert not np.any(filtered_mask[:, 1])


def test_schema_validation_and_serialization() -> None:
    dates = np.array([20240102, 20240103])
    symbols = ["600519"]
    close = np.full((2, 1), 100.0, order="F")
    panel = Panel(
        dates=dates,
        symbols=symbols,
        fields={"close": close},
        mask=np.ones((2, 1), dtype=bool),
    )

    adapter = StockMarketAdapter()
    valid, missing = adapter.validate_panel_schema(panel)
    assert not valid
    assert "volume" in missing

    d = adapter.to_dict()
    assert d["name"] == "stocks"
    assert d["is_t_plus_one"] is True
    assert "cost_spec" in d
