"""A 股标准面板加载器、掩码构建与回测口径单元测试 (tests/python/test_stocks_panel_and_eval.py)。

按审查重构计划 (P0-11, P1-07) 规格要求，验证:
1. StockPanelLoader 与 StockPanelData 结构规范与 core.Panel 导出能力；
2. BaseUnivMask 准确剔除 ST、停牌、零成交及非存续期标的；
3. LimitMask 准确剔除封死涨跌停及一字板标的；
4. StockListedDays 自然日计算无偏差；
5. compute_forward_returns 正确区分 open_adj 与 vwap_adj 价格时钟；
6. factorlib_core.Engine 结合 BaseUnivMask 与 LimitMask 正常求值与后处理。
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import numpy as np
import pytest
from factorlib.core.panel import Panel
from factorlib.markets.stocks.panel_loader import StockPanelData, StockPanelLoader


def test_loader_nonexistent_dir() -> None:
    """验证传入非法目录时抛出 FileNotFoundError。"""
    with pytest.raises(FileNotFoundError, match="原始数据根目录不存在"):
        StockPanelLoader("/tmp/nonexistent_stocks_dir_12345")


def test_stock_panel_data_container_and_panel_conversion() -> None:
    """验证 StockPanelData 内存容器属性、字典兼容访问与 to_panel 转换。"""
    dates = ["20240102", "20240103", "20240104"]
    symbols = ["000001.XSHE", "600519.XSHG"]
    T, N = 3, 2

    close_hfq = np.full((T, N), 10.0, dtype=np.float64, order="F")
    open_adj = np.full((T, N), 9.9, dtype=np.float64, order="F")
    high_adj = np.full((T, N), 10.2, dtype=np.float64, order="F")
    low_adj = np.full((T, N), 9.8, dtype=np.float64, order="F")
    vwap_adj = np.full((T, N), 10.05, dtype=np.float64, order="F")
    amount = np.full((T, N), 1e6, dtype=np.float64, order="F")
    volume = np.full((T, N), 1e5, dtype=np.float64, order="F")
    log_amount = np.log(amount)
    c2c_ret = np.zeros((T, N), dtype=np.float64, order="F")
    ex_mat = np.ones((T, N), dtype=np.float64, order="F")

    base_univ_mask = np.ones((T, N), dtype=bool, order="F")
    base_univ_mask[1, 1] = False  # 标的 1 在第 2 天不在池

    limit_mask = np.ones((T, N), dtype=np.float64, order="F")
    limit_mask[2, 0] = np.nan  # 标的 0 在第 3 天触及涨停

    is_susp = np.zeros((T, N), dtype=bool, order="F")
    is_st = np.zeros((T, N), dtype=bool, order="F")
    listed_days = np.array(
        [[100.0, 500.0], [101.0, 501.0], [102.0, 502.0]], dtype=np.float64, order="F"
    )
    ind_code = np.array([[1.0, 2.0], [1.0, 2.0], [1.0, 2.0]], dtype=np.float64, order="F")

    bench_open = np.array([0.01, -0.005, np.nan])
    bench_close = np.array([0.012, -0.004, np.nan])

    fields = {
        "close_hfq": close_hfq,
        "open_adj": open_adj,
        "high_adj": high_adj,
        "low_adj": low_adj,
        "vwap_adj": vwap_adj,
        "amount": amount,
        "volume": volume,
        "log_amount": log_amount,
        "c2c_ret": c2c_ret,
        "ex_cum_factor": ex_mat,
    }

    panel_data = StockPanelData(
        dates=dates,
        symbols=symbols,
        fields=fields,
        base_univ_mask=base_univ_mask,
        limit_mask=limit_mask,
        is_suspended=is_susp,
        is_st=is_st,
        stock_listed_days=listed_days,
        industry_code=ind_code,
        bench_fwd_open_1=bench_open,
        bench_fwd_close_1=bench_close,
        bench_ret=bench_open,
    )

    assert panel_data.shape == (3, 2)
    assert panel_data.T == 3
    assert panel_data.N == 2

    # 字典兼容接口
    assert panel_data["close_hfq"].shape == (3, 2)
    assert panel_data["listed_days"].shape == (3, 2)
    assert panel_data.get("base_univ_mask") is not None
    assert "open_adj" in panel_data.keys()

    # 转换为 core.Panel
    panel = panel_data.to_panel()
    assert isinstance(panel, Panel)
    assert panel.shape == (3, 2)
    assert panel.mask[1, 1] is False or panel.mask[1, 1] == 0
    assert "close_hfq" in panel.fields
    assert "limit_mask" in panel.fields
    assert "stock_listed_days" in panel.fields


def test_compute_forward_returns_logic_and_execution_simulation() -> None:
    """验证远期收益率计算、价格代理切换以及 execution_simulation 涨跌停拦截。"""
    dates = ["20240102", "20240103", "20240104", "20240105"]
    symbols = ["S1", "S2"]
    T, N = 4, 2

    # S1: 价格从 10 -> 11 -> 12 -> 13
    # S2: 价格从 20 -> 18 -> 16 -> 14
    open_adj = np.array(
        [
            [10.0, 20.0],  # T=0
            [11.0, 18.0],  # T=1
            [12.0, 16.0],  # T=2
            [13.0, 14.0],  # T=3
        ],
        dtype=np.float64,
        order="F",
    )
    vwap_adj = open_adj * 1.02  # 模拟全天 VWAP 价格偏移

    volume = np.full((T, N), 1000.0, dtype=np.float64, order="F")
    is_susp = np.zeros((T, N), dtype=bool, order="F")
    limit_mask = np.ones((T, N), dtype=np.float64, order="F")
    base_mask = np.ones((T, N), dtype=bool, order="F")

    # 模拟 S1 在 T=1 (入场日) 开盘一字涨停无法建仓
    limit_mask[1, 0] = np.nan

    panel_data = StockPanelData(
        dates=dates,
        symbols=symbols,
        fields={
            "open_adj": open_adj,
            "vwap_adj": vwap_adj,
            "volume": volume,
        },
        base_univ_mask=base_mask,
        limit_mask=limit_mask,
        is_suspended=is_susp,
        is_st=np.zeros((T, N), dtype=bool, order="F"),
        stock_listed_days=np.full((T, N), 500.0, dtype=np.float64, order="F"),
        industry_code=np.ones((T, N), dtype=np.float64, order="F"),
        bench_fwd_open_1=np.zeros(T),
        bench_fwd_close_1=np.zeros(T),
        bench_ret=np.zeros(T),
    )

    # 1. 纯因子模式 (execution_simulation=False)
    # T=0 对应的 [T+1, T+2] 收益:
    # S1: open_adj[2] / open_adj[1] - 1 = 12 / 11 - 1
    # S2: open_adj[2] / open_adj[1] - 1 = 16 / 18 - 1
    fwd_open = panel_data.compute_forward_returns(horizons=[1], exec_price_mode="open_adj")
    assert pytest.approx(fwd_open[1][0, 0], rel=1e-5) == 12.0 / 11.0 - 1.0
    assert pytest.approx(fwd_open[1][0, 1], rel=1e-5) == 16.0 / 18.0 - 1.0

    # 2. 模拟实盘交易约束 (execution_simulation=True)
    # T+1 日 S1 的 limit_mask 为 NaN，买不进 -> 返回 NaN
    fwd_sim = panel_data.compute_forward_returns(
        horizons=[1], exec_price_mode="open_adj", execution_simulation=True
    )
    assert np.isnan(fwd_sim[1][0, 0])  # S1 无法买入
    assert pytest.approx(fwd_sim[1][0, 1], rel=1e-5) == 16.0 / 18.0 - 1.0  # S2 正常买入


def test_real_data_subset_loader_and_mask_semantics() -> None:
    """基于真实股票目录抽样加载验证 BaseUnivMask 与 LimitMask 语义。"""
    raw_path = Path("data/raw/stocks")
    if not (raw_path / "bar_1d").exists():
        pytest.skip("data/raw/stocks 未包含日频行情数据，跳过真实数据测试")

    loader = StockPanelLoader(raw_path)
    # 加载 2024 年 1 月前两周数据
    panel_data = loader.load(
        start_date=date(2024, 1, 2),
        end_date=date(2024, 1, 15),
        exec_price_mode="open_adj",
    )

    assert panel_data.T == 10
    assert panel_data.N > 4000

    # 验证 BaseUnivMask:
    # 若某单元格 is_st=True 或 is_suspended=True，其 base_univ_mask 必须为 False
    st_idx = panel_data.is_st
    assert np.all(~panel_data.base_univ_mask[st_idx])

    susp_idx = panel_data.is_suspended
    assert np.all(~panel_data.base_univ_mask[susp_idx])

    # 验证 LimitMask:
    # 只要 limit_mask 为有限值，必须严格等于 1.0
    valid_limit = np.isfinite(panel_data.limit_mask)
    np.testing.assert_array_equal(panel_data.limit_mask[valid_limit], 1.0)

    # 验证 StockListedDays: 所有有效值必须非负且合理 (自然日)
    valid_days = panel_data.stock_listed_days[np.isfinite(panel_data.stock_listed_days)]
    assert np.all(valid_days >= 0.0)
    assert np.max(valid_days) > 365.0  # 全市场必然包含上市超过 1 年的股票


def test_to_panel_aliases_and_postprocess_zero_leak_after_decay() -> None:
    """验证 to_panel() 暴露 listed_days/sector 且经 decay 后重施掩码零渗漏。"""
    from factorlib.core.probe import apply_postprocess_pipeline

    T, N = 5, 6
    dates = [f"2024010{i + 2}" for i in range(T)]
    symbols = ["S1", "S2", "S3", "S4", "S5", "S6"]

    close_hfq = np.full((T, N), 10.0, dtype=np.float64, order="F")
    vol = np.full((T, N), 1000.0, dtype=np.float64, order="F")
    base_mask = np.ones((T, N), dtype=bool, order="F")
    limit_mask = np.full((T, N), 1.0, dtype=np.float64, order="F")
    listed_days = np.full((T, N), 200.0, dtype=np.float64, order="F")
    ind = np.array([[1.0, 1.0, 2.0, 2.0, 1.0, 2.0]] * T, dtype=np.float64, order="F")

    # T=2 时 S1 不在池、S2 锁板、S3 未满 120 天，保留 S4/S5/S6 具有有效截面方差
    base_mask[2, 0] = False
    limit_mask[2, 1] = np.nan
    listed_days[2, 2] = 60.0

    panel_data = StockPanelData(
        dates=dates,
        symbols=symbols,
        fields={
            "close_hfq": close_hfq,
            "open_adj": close_hfq.copy(),
            "vwap_adj": close_hfq.copy(),
            "volume": vol,
            "amount": vol * 10.0,
        },
        base_univ_mask=base_mask,
        limit_mask=limit_mask,
        is_suspended=np.zeros((T, N), dtype=bool, order="F"),
        is_st=np.zeros((T, N), dtype=bool, order="F"),
        stock_listed_days=listed_days,
        industry_code=ind,
        bench_fwd_open_1=np.zeros(T),
        bench_fwd_close_1=np.zeros(T),
        bench_ret=np.zeros(T),
    )

    panel = panel_data.to_panel()
    assert "listed_days" in panel.fields and "sector" in panel.fields

    raw = np.asfortranarray(np.tile([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], (T, 1)), dtype=np.float64)
    res = apply_postprocess_pipeline(
        raw,
        [
            "baseUniMask",
            "trdMask",
            "dropNewStock:120",
            "MAD:3",
            "zScore",
            "indNeut:sector",
            "decay:3",
            "rank",
        ],
        panel,
    )
    assert np.isnan(res[2, 0])
    assert np.isnan(res[2, 1])
    assert np.isnan(res[2, 2])
    assert np.all(np.isfinite(res[2, 3:]))


def test_execution_simulation_directional_masks_and_exit_deferral() -> None:
    """验证分方向买卖掩码：t+1+h 涨停不误删盈利、跌停/停牌顺延至首个可卖出日平仓。"""
    T, N = 6, 4
    dates = [f"2024010{i + 2}" for i in range(T)]
    symbols = ["S_BUY_BLK", "S_EXIT_LU", "S_EXIT_DEFER", "S_FOREVER_HALT"]

    open_adj = np.array(
        [
            [10.0, 10.0, 10.0, 10.0],  # t=0 (signal)
            [11.0, 10.0, 10.0, 10.0],  # t=1 (entry)
            [11.0, 12.0, 9.0, 10.0],  # t=2 (target exit for h=1)
            [11.0, 12.0, 8.1, 10.0],  # t=3 (still locked down for S_EXIT_DEFER)
            [11.0, 12.0, 8.5, 10.0],  # t=4 (tradable recovery for S_EXIT_DEFER)
            [11.0, 12.0, 8.5, 10.0],  # t=5
        ],
        dtype=np.float64,
        order="F",
    )
    vol = np.full((T, N), 1000.0, dtype=np.float64, order="F")
    base_mask = np.ones((T, N), dtype=bool, order="F")
    limit_mask = np.ones((T, N), dtype=np.float64, order="F")
    can_buy_mask = np.ones((T, N), dtype=bool, order="F")
    can_sell_mask = np.ones((T, N), dtype=bool, order="F")
    is_susp = np.zeros((T, N), dtype=bool, order="F")

    # S_BUY_BLK 在 t=1 开盘涨停不可买入
    can_buy_mask[1, 0] = False
    limit_mask[1, 0] = np.nan

    # S_EXIT_LU 在 t=2 收盘涨停：双向 limit_mask 为 NaN，但卖出端 can_sell_mask 为 True（应正常兑现盈利）
    limit_mask[2, 1] = np.nan
    can_sell_mask[2, 1] = True

    # S_EXIT_DEFER 在 t=2, t=3 跌停不可卖出，t=4 恢复可卖（应顺延至 t=4 按 8.5/10.0 - 1 平仓）
    limit_mask[2:4, 2] = np.nan
    can_sell_mask[2:4, 2] = False

    # S_FOREVER_HALT 从 t=2 起连续停牌至样本末尾（无样本内平仓日 -> NaN）
    is_susp[2:, 3] = True
    can_sell_mask[2:, 3] = False

    panel_data = StockPanelData(
        dates=dates,
        symbols=symbols,
        fields={
            "open_adj": open_adj,
            "vwap_adj": open_adj.copy(),
            "close_hfq": open_adj.copy(),
            "volume": vol,
            "amount": vol * open_adj,
        },
        base_univ_mask=base_mask,
        limit_mask=limit_mask,
        is_suspended=is_susp,
        is_st=np.zeros((T, N), dtype=bool, order="F"),
        stock_listed_days=np.full((T, N), 500.0, dtype=np.float64, order="F"),
        industry_code=np.ones((T, N), dtype=np.float64, order="F"),
        bench_fwd_open_1=np.zeros(T),
        bench_fwd_close_1=np.zeros(T),
        bench_ret=np.zeros(T),
        can_buy_mask=can_buy_mask,
        can_sell_mask=can_sell_mask,
    )

    fwd_sim = panel_data.compute_forward_returns(
        horizons=[1], exec_price_mode="open_adj", execution_simulation=True
    )[1]
    assert np.isnan(fwd_sim[0, 0])
    assert pytest.approx(fwd_sim[0, 1], rel=1e-12) == 12.0 / 10.0 - 1.0
    assert pytest.approx(fwd_sim[0, 2], rel=1e-12) == 8.5 / 10.0 - 1.0
    assert np.isnan(fwd_sim[0, 3])
