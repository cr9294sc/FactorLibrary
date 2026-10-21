"""测试 HTML 交互式报告生成器 (tests/python/test_report.py)。"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from factorlib.core.backtest import BacktestResult
from factorlib.core.report import generate_html_report, summary_dict


def test_generate_html_report(tmp_path: Path):
    dates = np.array([f"2024-01-0{i}" for i in range(1, 6)])
    symbols = ["RB", "CU"]
    T = len(dates)
    N = len(symbols)

    equity = np.array([10_000_000.0, 10_050_000.0, 10_020_000.0, 10_080_000.0, 10_100_000.0])
    returns = np.array([0.0, 0.005, -0.003, 0.006, 0.002])
    cash = equity * 0.7
    margin_used = equity * 0.3
    leverage = np.full(T, 1.5)
    gross_pnl = np.array([0.0, 52000.0, -28000.0, 63000.0, 21000.0])
    fee = np.array([0.0, 1500.0, 1200.0, 2000.0, 800.0])
    slippage = np.array([0.0, 500.0, 800.0, 1000.0, 200.0])
    roll_cost = np.array([0.0, 0.0, 1000.0, 0.0, 0.0])
    net_pnl = gross_pnl - fee - slippage
    turnover = np.array([0.0, 0.3, 0.2, 0.4, 0.1])
    positions = np.zeros((T, N), dtype=np.int64)

    metrics = {
        "total_return": 0.01,
        "annual_return": 0.05,
        "annual_volatility": 0.08,
        "sharpe": 1.25,
        "sortino": 1.80,
        "calmar": 2.10,
        "max_drawdown": 0.003,
        "max_drawdown_days": 1,
        "win_rate": 0.75,
        "profit_loss_ratio": 2.2,
        "daily_turnover_mean": 0.25,
        "annual_turnover": 63.0,
        "total_fees": float(np.sum(fee)),
        "total_slippage": float(np.sum(slippage)),
        "total_roll_cost": float(np.sum(roll_cost)),
        "max_margin_utilization": 0.30,
        "max_leverage": 1.5,
    }

    res = BacktestResult(
        dates=dates,
        symbols=symbols,
        equity=equity,
        returns=returns,
        cash=cash,
        margin_used=margin_used,
        leverage=leverage,
        gross_pnl=gross_pnl,
        net_pnl=net_pnl,
        fee=fee,
        slippage=slippage,
        roll_cost=roll_cost,
        turnover=turnover,
        positions=positions,
        metrics=metrics,
    )

    out_html = tmp_path / "report.html"
    p = generate_html_report(res, out_html, title="单元测试回测报告")

    assert p.exists()
    assert p == out_html
    html_text = p.read_text(encoding="utf-8")
    assert "单元测试回测报告" in html_text
    assert "<svg" in html_text
    assert "heatmap-table" in html_text
    assert "waterfall-container" in html_text
    assert "10,100,000.00" in html_text or "1.00%" in html_text

    # 验证配套 JSON 文件输出
    out_json = tmp_path / "report.json"
    assert out_json.exists()
    data = json.loads(out_json.read_text(encoding="utf-8"))
    assert data["total_return"] == 0.01
    assert data["n_trading_days"] == 5
    assert data["sharpe"] == 1.25

    # 验证 summary_dict
    s_d = summary_dict(res)
    assert s_d["start_date"] == "2024-01-01"
    assert s_d["end_date"] == "2024-01-05"
