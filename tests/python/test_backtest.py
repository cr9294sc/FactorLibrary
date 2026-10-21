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

    # 3. 验证 Day 3 涨停阻断：加仓至 40% 失败，持仓手数保持不变
    assert res.positions[3, 0] == res.positions[2, 0]

    # 4. 验证 Day 4 成功平仓归零
    assert res.positions[4, 0] == 0

    # 5. 验证 DataFrame 导出与 summary
    df = res.to_dataframe()
    assert len(df) == T
    assert "equity" in df.columns
    assert "returns" in df.columns

    summary_text = res.summary()
    assert "FactorLibrary 期货实盘回测绩效汇总" in summary_text
    assert "夏普比率" in summary_text


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
