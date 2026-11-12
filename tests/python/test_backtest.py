"""测试期货实盘回测状态机与快速回测 (tests/python/test_backtest.py)。"""

from __future__ import annotations

import numpy as np
from factorlib.core.backtest import BacktestResult, run_backtest, run_fast_backtest
from factorlib.core.config import BacktestConfig, CapitalConfig, ExecutionConfig, PortfolioConfig
from factorlib.core.panel import Panel


def test_run_fast_backtest_basic():
    T = 10
    N = 2
    # 构造简单权重与收益率
    weights = np.zeros((T, N), dtype=np.float64, order="F")
    returns = np.zeros((T, N), dtype=np.float64, order="F")

    weights[:, 0] = 0.5
    returns[:, 0] = 0.01  # 品种 0 每天收益 1%

    res = run_fast_backtest(weights, returns, fee_rate=0.0001, slippage_bps=1.0)
    assert "equity" in res
    assert "metrics" in res
    assert len(res["equity"]) == T
    assert res["equity"][-1] > res["equity"][0]
    assert res["metrics"]["total_return"] > 0.0
    assert res["metrics"]["sharpe"] > 0.0


def test_run_futures_backtest_identity_and_limit_lock():
    # 构造手工可审计推演样本
    # T=5 日, N=2 品种 (RB, I)
    dates = np.array([f"2024-01-0{i}" for i in range(1, 6)])
    symbols = ["RB", "I"]
    T = len(dates)
    N = len(symbols)

    exec_vwap = np.full((T, N), 100.0, dtype=np.float64, order="F")
    exec_vwap_old = np.full((T, N), 100.0, dtype=np.float64, order="F")
    settle_raw = np.full((T, N), 100.0, dtype=np.float64, order="F")
    limit_up = np.full((T, N), np.inf, dtype=np.float64, order="F")
    limit_down = np.full((T, N), 0.0, dtype=np.float64, order="F")
    is_unilateral = np.zeros((T, N), dtype=np.float64, order="F")
    is_roll = np.zeros((T, N), dtype=np.float64, order="F")

    # Day 1: 开仓 RB (多头权重 20%), 执行价 100, 收盘结算价 105 (盈利)
    exec_vwap[1, 0] = 100.0
    settle_raw[1, 0] = 105.0

    # Day 2: 换月展期日！
    # 旧合约平仓执行价 106.0, 新合约开仓执行价 200.0, 新合约结算价 204.0
    is_roll[2, 0] = 1.0
    exec_vwap_old[2, 0] = 106.0
    exec_vwap[2, 0] = 200.0
    settle_raw[2, 0] = 204.0

    # Day 3: 触发涨停锁死！RB 计划加仓，但 limit_up == exec_vwap，买单应被拒绝
    limit_up[3, 0] = 204.0
    exec_vwap[3, 0] = 204.0
    settle_raw[3, 0] = 204.0

    # Day 4: 平仓 RB (目标权重 0.0), 执行价 204.0, 结算价 204.0
    exec_vwap[4, 0] = 204.0
    settle_raw[4, 0] = 204.0

    panel = Panel(
        dates=dates,
        symbols=symbols,
        fields={
            "exec_vwap": exec_vwap,
            "exec_vwap_old": exec_vwap_old,
            "settle_raw": settle_raw,
            "limit_up": limit_up,
            "limit_down": limit_down,
            "is_unilateral": is_unilateral,
            "is_roll": is_roll,
            "close": settle_raw,
        },
        mask=np.ones((T, N), dtype=bool),
    )

    weights = np.zeros((T, N), dtype=np.float64, order="F")
    weights[0, 0] = 0.20  # Day 0 收盘决策: Day 1 买入 20%
    weights[1, 0] = 0.20  # Day 1 收盘决策: Day 2 维持 20%
    weights[2, 0] = 0.40  # Day 2 收盘决策: Day 3 拟加仓至 40% (将被涨停阻止)
    weights[3, 0] = 0.00  # Day 3 收盘决策: Day 4 平仓至 0%

    cfg = BacktestConfig(
        capital=CapitalConfig(init_cash=1_000_000.0),
        portfolio=PortfolioConfig(mode="cs"),
        execution=ExecutionConfig(max_margin_ratio=0.50, margin_safety_multiplier=1.0),
    )

    res: BacktestResult = run_backtest(panel, weights, config=cfg)

    # 1. 验证逐日盯市守恒定律 E_d - E_{d-1} == net_pnl_d
    for d in range(1, T):
        diff = abs(res.equity[d] - (res.equity[d - 1] + res.net_pnl[d]))
        assert diff < 1e-6, f"Day {d} 资金恒等式失衡: diff={diff}"

    # 2. 验证换月日展期核算：
    # 旧合约以 106.0 平仓，获得 (106 - 105) * lots * mult 盈亏
    # 新合约以 200.0 开仓，收盘 204.0，获得 (204 - 200) * new_lots * mult 盈亏
    # 绝不能出现 (200 - 106) 的伪跳空损益！
    assert res.roll_cost[2] > 0.0  # 产生换月手续费与滑点
    assert res.gross_pnl[2] > 0.0
    assert res.gross_pnl[2] < 100_000.0  # 无虚假价格跳空


def test_run_futures_backtest_roll_nan_fallback_no_cross_contract_gap():
    """P0-02 回归测试: 换月日旧合约平仓执行价缺失 (NaN) 时，必须回退至旧合约昨日结算价。

    平仓盈亏严格为 0，绝对不发生跨合约价格直接作差导致巨额跳空盈亏。
    """
    dates = np.array([f"2024-01-0{i}" for i in range(1, 5)])
    symbols = ["RB"]
    T = len(dates)
    N = len(symbols)

    exec_vwap = np.full((T, N), 100.0, dtype=np.float64, order="F")
    exec_vwap_old = np.full((T, N), 100.0, dtype=np.float64, order="F")
    settle_raw = np.full((T, N), 100.0, dtype=np.float64, order="F")
    limit_up = np.full((T, N), np.inf, dtype=np.float64, order="F")
    limit_down = np.full((T, N), 0.0, dtype=np.float64, order="F")
    is_unilateral = np.zeros((T, N), dtype=np.float64, order="F")
    is_roll = np.zeros((T, N), dtype=np.float64, order="F")

    # Day 1: 开仓 RB 20%, 执行价 100, 结算价 105 (昨结转入 Day 2 为 105)
    exec_vwap[1, 0] = 100.0
    settle_raw[1, 0] = 105.0

    # Day 2: 换月展期日！
    # 旧合约平仓执行价缺失 (NaN)
    # 新合约执行价 200.0, 结算价 204.0
    is_roll[2, 0] = 1.0
    exec_vwap_old[2, 0] = np.nan
    exec_vwap[2, 0] = 200.0
    settle_raw[2, 0] = 204.0

    # Day 3: 平仓
    exec_vwap[3, 0] = 204.0
    settle_raw[3, 0] = 204.0

    panel = Panel(
        dates=dates,
        symbols=symbols,
        fields={
            "exec_vwap": exec_vwap,
            "exec_vwap_old": exec_vwap_old,
            "settle_raw": settle_raw,
            "limit_up": limit_up,
            "limit_down": limit_down,
            "is_unilateral": is_unilateral,
            "is_roll": is_roll,
            "close": settle_raw,
        },
        mask=np.ones((T, N), dtype=bool),
    )

    weights = np.zeros((T, N), dtype=np.float64, order="F")
    weights[0, 0] = 0.20  # Day 1 开仓 20%
    weights[1, 0] = 0.20  # Day 2 换月维持 20%
    weights[2, 0] = 0.00  # Day 3 平仓

    cfg = BacktestConfig(
        capital=CapitalConfig(init_cash=1_000_000.0),
        portfolio=PortfolioConfig(mode="cs"),
        execution=ExecutionConfig(max_margin_ratio=0.50, margin_safety_multiplier=1.0),
    )

    res: BacktestResult = run_backtest(panel, weights, config=cfg)

    # 1. 资金守恒验证
    for d in range(1, T):
        diff = abs(res.equity[d] - (res.equity[d - 1] + res.net_pnl[d]))
        assert diff < 1e-6, f"Day {d} 资金恒等式失衡: diff={diff}"

    # 2. 关键断言:
    # 旧合约在 Day 2 平仓，因缺失 exec_vwap_old 回退至昨日结算价 105.0，
    # 故旧合约平仓盈亏严格为 (105.0 - 105.0) * lots * mult = 0.0！
    # 新合约在 Day 2 以 200.0 开仓，收盘结算价 204.0，盈利 (204.0 - 200.0) * new_lots * mult
    # 若错误回退至新合约开盘价 200.0，旧合约平仓盈亏将高达 (200.0 - 105.0) = +95.0 点的巨大跨合约跳空！
    new_lots = res.positions[2, 0]
    assert new_lots > 0
    expected_new_contract_pnl = new_lots * (204.0 - 200.0) * 10.0  # multiplier=10
    assert abs(res.gross_pnl[2] - expected_new_contract_pnl) < 1e-6
    # 确认换月成本（手续费+滑点）按昨结正常计提
    assert res.roll_cost[2] > 0.0
    # 确认 Day 3 平仓归零
    assert res.positions[3, 0] == 0


def test_run_futures_backtest_margin_scale_down():
    # 测试保证金超限时等比例缩减持仓
    dates = np.array(["2024-01-01", "2024-01-02"])
    symbols = ["RB"]
    T = len(dates)
    N = len(symbols)

    exec_vwap = np.full((T, N), 1000.0, dtype=np.float64, order="F")
    settle_raw = np.full((T, N), 1000.0, dtype=np.float64, order="F")

    panel = Panel(
        dates=dates,
        symbols=symbols,
        fields={
            "exec_vwap": exec_vwap,
            "settle_raw": settle_raw,
            "close": settle_raw,
        },
        mask=np.ones((T, N), dtype=bool),
    )

    # 目标权重设为 1000% (超高杠杆，必触发保证金上限 50%)
    weights = np.zeros((T, N), dtype=np.float64, order="F")
    weights[0, 0] = 10.0

    cfg = BacktestConfig(
        capital=CapitalConfig(init_cash=100_000.0),
        execution=ExecutionConfig(max_margin_ratio=0.50),
    )

    res = run_backtest(panel, weights, config=cfg)
    # 实际保证金占用不得超过最大上限 50%
    assert res.margin_used[1] <= 50_000.0 + 1e-4


def test_run_futures_backtest_bankruptcy_safe_no_nan():
    """P0-06 端到端测试: 账户穿仓破产时触发强平熔断，年化收益率保底 -1.0，杜绝 NaN。"""
    dates = np.array(["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04"])
    symbols = ["RB"]
    T = len(dates)
    N = len(symbols)

    exec_vwap = np.full((T, N), 100.0, dtype=np.float64, order="F")
    settle_raw = np.full((T, N), 100.0, dtype=np.float64, order="F")

    # Day 1: 开仓价 100，收盘结算价暴跌至 10 (杠杆放大致穿仓)
    exec_vwap[1, 0] = 100.0
    settle_raw[1, 0] = 10.0
    exec_vwap[2, 0] = 10.0
    settle_raw[2, 0] = 10.0
    exec_vwap[3, 0] = 10.0
    settle_raw[3, 0] = 10.0

    panel = Panel(
        dates=dates,
        symbols=symbols,
        fields={
            "exec_vwap": exec_vwap,
            "exec_vwap_old": exec_vwap,
            "settle_raw": settle_raw,
            "close": settle_raw,
        },
        mask=np.ones((T, N), dtype=bool),
    )

    weights = np.zeros((T, N), dtype=np.float64, order="F")
    weights[0, 0] = 5.0  # 5 倍杠杆
    weights[1, 0] = 5.0
    weights[2, 0] = 5.0

    cfg = BacktestConfig(
        capital=CapitalConfig(init_cash=100_000.0),
        execution=ExecutionConfig(max_margin_ratio=1.0, margin_safety_multiplier=1.0),
    )

    res = run_backtest(panel, weights, config=cfg)

    # Day 1 穿仓：净值跌为负数
    assert res.equity[1] < 0.0
    # Day 2 强平清零：仓位归零
    assert res.positions[2, 0] == 0
    assert res.positions[3, 0] == 0

    # 验证指标：total_return < -1.0, annual_return 严格为 -1.0 且非 NaN
    assert res.metrics["total_return"] < -1.0
    assert res.metrics["annual_return"] == -1.0
    assert not np.isnan(res.metrics["annual_return"])
    assert not np.isnan(res.metrics["calmar"])


def test_run_futures_backtest_position_flip_and_close_today_spec():
    """P0-07 端到端测试: 多空反手手续费拆分与 close_today_fee 全链路透传。"""
    dates = np.array(["2024-01-01", "2024-01-02", "2024-01-03"])
    symbols = ["RB"]
    T = len(dates)
    N = len(symbols)

    exec_vwap = np.full((T, N), 100.0, dtype=np.float64, order="F")
    settle_raw = np.full((T, N), 100.0, dtype=np.float64, order="F")

    panel = Panel(
        dates=dates,
        symbols=symbols,
        fields={
            "exec_vwap": exec_vwap,
            "exec_vwap_old": exec_vwap,
            "settle_raw": settle_raw,
            "close": settle_raw,
        },
        mask=np.ones((T, N), dtype=bool),
    )

    weights = np.zeros((T, N), dtype=np.float64, order="F")
    weights[0, 0] = 0.10  # Day 1 买入
    weights[1, 0] = -0.10  # Day 2 反手做空

    cfg = BacktestConfig(
        capital=CapitalConfig(init_cash=1_000_000.0),
        execution=ExecutionConfig(max_margin_ratio=0.50, margin_safety_multiplier=1.0),
    )

    res = run_backtest(panel, weights, config=cfg)
    lots_d1 = res.positions[1, 0]
    lots_d2 = res.positions[2, 0]
    assert lots_d1 > 0
    assert lots_d2 == -lots_d1

    # RB: open_fee=0.0001, close_fee=0.0001, close_today_fee=0.0001, mult=10
    # Day 2 反手交易 2 * lots_d1 手 (平昨 lots_d1 + 开新 lots_d1)
    expected_fee_d2 = 100.0 * 10.0 * (lots_d1 * 2) * 0.0001
    assert abs(res.fee[2] - expected_fee_d2) < 1e-4


def test_run_futures_backtest_roll_bidirectional_cost():
    """P0-07 端到端测试: 换月展期摩擦成本必须包含平旧与开新双边开平仓费用及滑点。"""
    dates = np.array(["2024-01-01", "2024-01-02", "2024-01-03"])
    symbols = ["RB"]
    T = len(dates)
    N = len(symbols)

    exec_vwap = np.full((T, N), 100.0, dtype=np.float64, order="F")
    exec_vwap_old = np.full((T, N), 100.0, dtype=np.float64, order="F")
    settle_raw = np.full((T, N), 100.0, dtype=np.float64, order="F")
    is_roll = np.zeros((T, N), dtype=np.float64, order="F")

    # Day 2 换月
    is_roll[1, 0] = 0.0
    is_roll[2, 0] = 1.0

    panel = Panel(
        dates=dates,
        symbols=symbols,
        fields={
            "exec_vwap": exec_vwap,
            "exec_vwap_old": exec_vwap_old,
            "settle_raw": settle_raw,
            "is_roll": is_roll,
            "close": settle_raw,
        },
        mask=np.ones((T, N), dtype=bool),
    )

    weights = np.zeros((T, N), dtype=np.float64, order="F")
    weights[0, 0] = 0.20  # Day 1 买入
    weights[1, 0] = 0.20  # Day 2 换月同等维持

    cfg = BacktestConfig(
        capital=CapitalConfig(init_cash=1_000_000.0),
        execution=ExecutionConfig(max_margin_ratio=0.50, margin_safety_multiplier=1.0),
    )

    res = run_backtest(panel, weights, config=cfg)

    # 换月日 Day 2 展期成本包含平旧和开新双边成本
    lots = res.positions[1, 0]
    mult = 10.0
    slip_ticks = 1.0
    tick_size = 1.0
    slip_per_side = lots * slip_ticks * tick_size * mult
    fee_per_side = lots * 100.0 * mult * 0.0001
    expected_roll_cost = 2 * (fee_per_side + slip_per_side)

    assert abs(res.roll_cost[2] - expected_roll_cost) < 1e-4
