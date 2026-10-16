from __future__ import annotations
from typing import Any, Dict, List
import os
import sys
import time

# Allow running from repo root
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "python"))

from pyrust_bt.api import BacktestConfig, BacktestEngine
from pyrust_bt.data import load_csv_to_bars
from pyrust_bt.strategy import Strategy
from pyrust_bt.analyzers import (
    compute_drawdown_segments,
    round_trips_from_trades,
    compute_performance_metrics,
    generate_analysis_report,
    export_round_trips_to_csv,
)

# Try to import vectorized functions
try:
    from engine_rust import compute_sma, compute_rsi
    print("✓ 使用 Rust 向量化指标")
except ImportError:
    print("⚠ Rust 向量化指标不可用，使用 Python 实现")
    compute_sma = None
    compute_rsi = None


class OptimizedSMAStrategy(Strategy):
    def __init__(self, sma_window: int = 10, rsi_window: int = 14, size: float = 1.0) -> None:
        self.sma_window = sma_window
        self.rsi_window = rsi_window
        self.size = size
        self._closes: List[float] = []
        self._bar_count = 0
        # 缓存指标值以减少计算
        self._last_sma: float | None = None
        self._last_rsi: float | None = None

    def on_start(self, ctx: Any) -> None:
        print(f"策略启动：SMA({self.sma_window}), RSI({self.rsi_window})")

    def next(self, bar: Dict[str, Any]) -> Dict[str, Any] | None:
        close = float(bar["close"])  # type: ignore[assignment]
        self._closes.append(close)
        self._bar_count += 1

        # 简化的SMA计算（滑动窗口）
        if len(self._closes) >= self.sma_window:
            sma = sum(self._closes[-self.sma_window:]) / self.sma_window
            self._last_sma = sma
        
        # 简化的RSI计算（每100根bar计算一次）
        if self._bar_count % 100 == 0 and len(self._closes) >= self.rsi_window + 1:
            gains = 0.0
            losses = 0.0
            for i in range(-self.rsi_window, 0):
                change = self._closes[i] - self._closes[i-1]
                if change > 0:
                    gains += change
                else:
                    losses += abs(change)
            
            if losses > 0:
                rs = gains / losses
                self._last_rsi = 100 - (100 / (1 + rs))
            else:
                self._last_rsi = 100.0

        # 交易逻辑
        if self._last_sma is not None and self._last_rsi is not None:
            if close > self._last_sma and self._last_rsi < 30:
                return {"action": "BUY", "type": "market", "size": self.size}
            elif close < self._last_sma and self._last_rsi > 70:
                return {"action": "SELL", "type": "market", "size": self.size}

        return None

    def on_stop(self) -> None:
        print(f"策略结束，共处理 {self._bar_count} 根bar")


def run_performance_comparison() -> None:
    print("=== pyrust-bt 性能优化测试 ===")
    
    # 数据加载
    data_path = os.path.join(os.path.dirname(__file__), "data", "sh600000_min.csv")
    if not os.path.exists(data_path):
        print(f"数据文件不存在: {data_path}")
        print("使用 sample.csv")
        data_path = os.path.join(os.path.dirname(__file__), "data", "sample.csv")
    
    print(f"加载数据: {data_path}")
    start_time = time.time()
    bars = load_csv_to_bars(data_path, symbol="TEST")
    load_time = time.time() - start_time
    print(f"数据加载完成: {len(bars)} 根bar, 耗时 {load_time:.3f}s")

    # 测试不同批处理大小
    batch_sizes = [500, 1000, 2000, 5000]
    results = []

    for batch_size in batch_sizes:
        print(f"\n--- 测试批处理大小: {batch_size} ---")
        
        cfg = BacktestConfig(
            start="2016-01-01",
            end="2025-12-31",
            cash=100000.0,
            commission_rate=0.0005,
            slippage_bps=2.0,
            batch_size=batch_size,
        )

        engine = BacktestEngine(cfg)
        strategy = OptimizedSMAStrategy(sma_window=20, rsi_window=14, size=10.0)
        
        print("开始回测...")
        start_time = time.time()
        result = engine.run(strategy, bars)
        backtest_time = time.time() - start_time
        
        bars_per_second = len(bars) / backtest_time if backtest_time > 0 else 0
        print(f"回测完成，耗时 {backtest_time:.3f}s ({bars_per_second:.0f} bars/s)")
        
        results.append({
            'batch_size': batch_size,
            'time': backtest_time,
            'bars_per_second': bars_per_second,
            'equity': result['equity'],
            'trades': len(result['trades']),
        })

    # 性能对比
    print(f"\n=== 性能对比结果 ===")
    print(f"{'批处理大小':<10} {'耗时(s)':<10} {'速度(bars/s)':<15} {'净值':<12} {'交易次数'}")
    print("-" * 65)
    
    for r in results:
        print(f"{r['batch_size']:<10} {r['time']:<10.3f} {r['bars_per_second']:<15.0f} {r['equity']:<12.2f} {r['trades']}")
    
    # 找出最佳性能
    best = max(results, key=lambda x: x['bars_per_second'])
    print(f"\n最佳性能: 批处理大小 {best['batch_size']}, {best['bars_per_second']:.0f} bars/s")

    # 详细分析最佳结果
    print(f"\n=== 最佳配置详细分析 ===")
    cfg_best = BacktestConfig(
        start="2016-01-01",
        end="2025-12-31", 
        cash=100000.0,
        commission_rate=0.0005,
        slippage_bps=2.0,
        batch_size=best['batch_size'],
    )
    
    engine_best = BacktestEngine(cfg_best)
    strategy_best = OptimizedSMAStrategy(sma_window=20, rsi_window=14, size=10.0)
    
    start_time = time.time()
    result_best = engine_best.run(strategy_best, bars)
    backtest_time = time.time() - start_time
    
    # 分析时间
    start_time = time.time()
    equity_curve = result_best["equity_curve"]
    trades = result_best["trades"]
    
    dd_segments = compute_drawdown_segments(equity_curve)
    round_trips = round_trips_from_trades(trades, bars)
    performance = compute_performance_metrics(equity_curve)
    report = generate_analysis_report(equity_curve, trades, round_trips, dd_segments)
    
    analysis_time = time.time() - start_time
    
    print(f"回测执行: {backtest_time:.3f}s ({len(bars)/backtest_time:.0f} bars/s)")
    print(f"结果分析: {analysis_time:.3f}s")
    print(f"总耗时: {load_time + backtest_time + analysis_time:.3f}s")
    
    # 输出关键统计
    stats = result_best.get("stats", {})
    print(f"\n净值: {result_best['equity']:,.2f}")
    print(f"总收益: {stats.get('total_return', 0):.4f}")
    print(f"最大回撤: {stats.get('max_drawdown', 0):.4f}")
    print(f"夏普比率: {stats.get('sharpe', 0):.4f}")
    print(f"交易次数: {len(trades)}")


def main() -> None:
    run_performance_comparison()


if __name__ == "__main__":
    main() 