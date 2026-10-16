"""
基于持仓权重文件的回测系统
Portfolio backtest system based on holdings weight files
"""

from __future__ import annotations
from typing import Any, Dict, List
import os
import sys
import csv
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

# 允许从项目根目录运行
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "python"))

from pyrust_bt.api import BacktestEngine, BacktestConfig
from pyrust_bt.strategy import Strategy


def load_multi_asset_market_data(data_path: str) -> Dict[str, List[Dict[str, Any]]]:
    """
    从CSV文件加载多资产市场数据
    Load multi-asset market data from CSV file

    CSV格式: symbol,datetime,open,high,low,close,volume
    """
    feeds: Dict[str, List[Dict[str, Any]]] = {}

    if not os.path.exists(data_path):
        print(f"警告: 数据文件不存在: {data_path}")
        print("正在生成示例数据...")
        return generate_sample_market_data()

    with open(data_path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            sym = row["symbol"]
            feeds.setdefault(sym, []).append(
                {
                    "symbol": sym,
                    "datetime": row["datetime"],
                    "open": float(row["open"]) if row.get("open") else 0.0,
                    "high": float(row["high"]) if row.get("high") else 0.0,
                    "low": float(row["low"]) if row.get("low") else 0.0,
                    "close": float(row["close"]) if row.get("close") else 0.0,
                    "volume": float(row["volume"]) if row.get("volume") else 0.0,
                }
            )

    # 确保每个feed的数据按datetime排序
    for k in feeds:
        feeds[k].sort(key=lambda x: x["datetime"])

    return feeds


def generate_sample_market_data() -> Dict[str, List[Dict[str, Any]]]:
    """
    生成示例市场数据
    Generate sample market data for demonstration
    """
    print("正在生成示例市场数据...")

    # 定义资产列表
    assets = ["AAA", "BBB", "CCC", "DDD", "EEE"]

    # 定义时间范围 (2022年)
    start_date = datetime(2022, 1, 1)
    end_date = datetime(2022, 12, 31)

    feeds: Dict[str, List[Dict[str, Any]]] = {}

    # 为每个资产生成价格数据
    for asset in assets:
        # 初始价格
        base_price = np.random.uniform(50, 200)

        # 生成每日数据
        current_date = start_date
        while current_date <= end_date:
            # 跳过周末
            if current_date.weekday() < 5:  # 0-4 是周一到周五
                # 生成随机价格变动
                daily_return = np.random.normal(0, 0.02)  # 2% 日波动率

                if asset in feeds:
                    last_price = feeds[asset][-1]["close"]
                else:
                    last_price = base_price

                # 计算新价格
                open_price = last_price * (1 + np.random.normal(0, 0.005))
                close_price = open_price * (1 + daily_return)
                high_price = max(open_price, close_price) * (1 + abs(np.random.normal(0, 0.005)))
                low_price = min(open_price, close_price) * (1 - abs(np.random.normal(0, 0.005)))
                volume = np.random.uniform(1000, 10000)

                bar = {
                    "symbol": asset,
                    "datetime": current_date.strftime("%Y-%m-%d"),
                    "open": round(open_price, 2),
                    "high": round(high_price, 2),
                    "low": round(low_price, 2),
                    "close": round(close_price, 2),
                    "volume": round(volume, 0),
                }

                feeds.setdefault(asset, []).append(bar)

            current_date += timedelta(days=1)

    print(f"已生成 {len(feeds)} 个资产的示例数据")
    return feeds


class PortfolioRebalancingStrategy(Strategy):
    """
    基于外部持仓文件的投资组合再平衡策略
    Portfolio rebalancing strategy based on external holdings file
    """

    def __init__(self, holdings_path: str, rebalance_tolerance: float = 0.05) -> None:
        """
        初始化策略
        Args:
            holdings_path: 持仓文件路径
            rebalance_tolerance: 再平衡容忍度 (权重偏差超过此值时触发再平衡)
        """
        self.holdings = self._load_holdings(holdings_path)
        self.rebalanced_dates = set()
        self.rebalance_tolerance = rebalance_tolerance
        self.last_portfolio_value = 0.0

        print(f"加载持仓文件: {holdings_path}")
        print(f"调仓日期: {sorted(self.holdings.keys())}")

    def _load_holdings(self, path: str) -> Dict[str, Dict[str, float]]:
        """
        从CSV文件加载持仓数据
        Load holdings data from CSV file

        CSV格式: date,asset,weight
        返回: {date_str: {asset_symbol: weight}}
        """
        holdings_map: Dict[str, Dict[str, float]] = {}

        if not os.path.exists(path):
            print(f"警告: 持仓文件不存在: {path}")
            return holdings_map

        df = pd.read_csv(path, parse_dates=['date'])

        for _, row in df.iterrows():
            # 格式化日期为 YYYY-MM-DD 以匹配市场数据格式
            date_str = row['date'].strftime('%Y-%m-%d')
            if date_str not in holdings_map:
                holdings_map[date_str] = {}
            holdings_map[date_str][row['asset']] = row['weight']

        # 验证权重总和
        for date, weights in holdings_map.items():
            total_weight = sum(weights.values())
            if abs(total_weight - 1.0) > 0.01:
                print(f"警告: {date} 的权重总和不为1: {total_weight:.3f}")

        return holdings_map

    def _calculate_target_positions(self, target_weights: Dict[str, float],
                                  current_prices: Dict[str, float],
                                  total_value: float) -> Dict[str, float]:
        """
        计算目标持仓数量
        Calculate target position sizes based on weights and current prices
        """
        target_positions = {}
        for symbol, weight in target_weights.items():
            target_value = total_value * weight
            if symbol in current_prices and current_prices[symbol] > 0:
                target_positions[symbol] = target_value / current_prices[symbol]
            else:
                print(f"警告: 无法获取 {symbol} 的当前价格")
                target_positions[symbol] = 0.0
        return target_positions

    def next_multi(self, update_slice: Dict[str, Dict[str, Any]],
                  ctx: Dict[str, Any]) -> List[Dict[str, Any]] | None:
        """
        多资产策略主要逻辑
        Main strategy logic for multi-asset backtest
        """
        # 从update_slice中获取当前日期
        if not update_slice:
            return None

        current_dt_str = next(iter(update_slice.values()))["datetime"]

        # 检查是否为调仓日期
        if current_dt_str in self.holdings and current_dt_str not in self.rebalanced_dates:
            return self._execute_rebalancing(current_dt_str, update_slice, ctx)

        return None

    def _execute_rebalancing(self, date_str: str,
                           update_slice: Dict[str, Dict[str, Any]],
                           ctx: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        执行投资组合再平衡
        Execute portfolio rebalancing
        """
        self.rebalanced_dates.add(date_str)
        target_weights = self.holdings[date_str]

        print(f"\n[{date_str}] 执行投资组合再平衡")
        print(f"目标权重: {target_weights}")

        # 获取当前价格
        current_prices = {}
        for symbol, bar in update_slice.items():
            current_prices[symbol] = bar["close"]

        # 获取当前投资组合价值
        total_equity = ctx.get("equity", self.last_portfolio_value)
        self.last_portfolio_value = total_equity

        print(f"当前投资组合总价值: {total_equity:,.2f}")

        # 计算目标持仓
        target_positions = self._calculate_target_positions(
            target_weights, current_prices, total_equity
        )

        # 获取当前持仓
        current_positions = ctx.get("positions", {})
        current_shares = {}
        for symbol, pos_info in current_positions.items():
            if isinstance(pos_info, dict):
                current_shares[symbol] = pos_info.get("position", 0.0)
            else:
                current_shares[symbol] = float(pos_info) if pos_info else 0.0

        # 生成交易指令
        actions = []

        # 处理目标持仓中的资产
        for symbol, target_shares in target_positions.items():
            current_shares_count = current_shares.get(symbol, 0.0)
            shares_diff = target_shares - current_shares_count

            if abs(shares_diff) > 0.01:  # 避免微小交易
                action = "BUY" if shares_diff > 0 else "SELL"
                actions.append({
                    "action": action,
                    "type": "market",
                    "size": abs(shares_diff),
                    "symbol": symbol,
                })

                # print(f"  {action} {symbol}: {abs(shares_diff):.2f} 股")

        # 清理不再持有的资产
        for symbol in current_shares:
            if symbol not in target_weights and current_shares[symbol] > 0:
                actions.append({
                    "action": "SELL",
                    "type": "market",
                    "size": current_shares[symbol],
                    "symbol": symbol,
                })
                # print(f"  SELL {symbol}: {current_shares[symbol]:.2f} 股 (清理)")

        print(f"生成 {len(actions)} 个交易指令")
        return actions

    def on_trade(self, event: Dict[str, Any]) -> None:
        """交易回调"""
        symbol = event.get("symbol", "Unknown")
        side = event.get("side", "Unknown")
        price = event.get("price", 0)
        size = event.get("size", 0)
        # print(f"  交易成交: {side} {symbol} {size:.2f}股 @ {price:.2f}")


def run_portfolio_backtest(market_data_path: str = None,
                          holdings_path: str = None,
                          start_date: str = "2022-01-01",
                          end_date: str = "2022-12-31",
                          initial_cash: float = 1_000_000.0) -> Dict[str, Any]:
    """
    运行投资组合回测
    Run portfolio backtest
    """
    print("=" * 60)
    print("基于持仓权重文件的投资组合回测系统")
    print("Portfolio Backtest System Based on Holdings Weight Files")
    print("=" * 60)

    # 配置回测参数
    cfg = BacktestConfig(
        start=start_date,
        end=end_date,
        cash=initial_cash,
        commission_rate=0.,  # 0.1% 手续费
        slippage_bps=0,        # 5 bps 滑点
        batch_size=5000,         # 大批量处理提高性能
    )
    engine = BacktestEngine(cfg)

    # 加载市场数据
    if market_data_path is None:
        market_data_path = os.path.join(os.path.dirname(__file__), "data", "2022.csv")

    feeds = load_multi_asset_market_data(market_data_path)
    print(f"已加载 {len(feeds)} 个资产的市场数据")

    # 检查数据时间范围
    all_dates = set()
    for symbol, bars in feeds.items():
        if bars:
            all_dates.add(bars[0]["datetime"])
            all_dates.add(bars[-1]["datetime"])

    if all_dates:
        print(f"数据时间范围: {min(all_dates)} 至 {max(all_dates)}")

    # 加载持仓文件
    if holdings_path is None:
        holdings_path = os.path.join(os.path.dirname(__file__), "holdings.csv")

    if not os.path.exists(holdings_path):
        print(f"错误: 持仓文件不存在: {holdings_path}")
        return {}

    # 初始化策略
    strategy = PortfolioRebalancingStrategy(holdings_path)

    # 运行回测
    print(f"\n开始回测...")
    print(f"初始资金: {initial_cash:,.2f}")
    print(f"回测期间: {start_date} 至 {end_date}")

    result = engine.run_multi(strategy, feeds)

    return result


def analyze_and_display_results(result: Dict[str, Any]) -> None:
    """
    分析并显示回测结果
    Analyze and display backtest results
    """
    print("\n" + "=" * 60)
    print("回测结果分析 / Backtest Results Analysis")
    print("=" * 60)

    if not result:
        print("回测结果为空")
        return

    # 基本结果
    final_equity = result.get('equity', 0)
    cash = result.get('cash', 0)
    realized_pnl = result.get('realized_pnl', 0)

    print(f"最终净值: {final_equity:,.2f}")
    print(f"现金余额: {cash:,.2f}")
    print(f"已实现盈亏: {realized_pnl:,.2f}")

    # 计算收益率
    initial_cash = 1_000_000.0  # 假设初始资金
    total_return = (final_equity - initial_cash) / initial_cash * 100
    print(f"总收益率: {total_return:.2f}%")

    # 统计信息
    stats = result.get("stats", {})
    if stats:
        print(f"\n风险收益指标:")
        print(f"  年化收益率: {stats.get('annualized_return', 0):.2%}")
        print(f"  年化波动率: {stats.get('volatility', 0):.2%}")
        print(f"  夏普比率: {stats.get('sharpe', 0):.3f}")
        print(f"  最大回撤: {stats.get('max_drawdown', 0):.2%}")
        print(f"  卡尔玛比率: {stats.get('calmar', 0):.3f}")

        print(f"\n交易统计:")
        print(f"  总交易次数: {stats.get('total_trades', 0)}")
        print(f"  盈利交易: {stats.get('winning_trades', 0)}")
        print(f"  亏损交易: {stats.get('losing_trades', 0)}")
        print(f"  胜率: {stats.get('win_rate', 0):.2%}")

    # 交易记录
    trades = result.get('trades', [])
    if trades:
        print(f"\n最近交易记录 (最后10笔):")
        for trade in trades[-10:]:
            symbol = trade.get('symbol', 'Unknown')
            side = trade.get('side', 'Unknown')
            price = trade.get('price', 0)
            size = trade.get('size', 0)
            print(f"  {side} {symbol} {size:.2f}股 @ {price:.2f}")


def generate_equity_curve_plot(result: Dict[str, Any], save_path: str = "portfolio_equity_curve.png") -> None:
    """
    生成净值曲线图
    Generate equity curve plot
    """
    if 'equity_curve' not in result or not result['equity_curve']:
        print("没有净值曲线数据")
        return

    print(f"\n正在生成净值曲线图...")

    try:
        import matplotlib
        matplotlib.use('Agg')  # 使用非交互式后端
        import matplotlib.pyplot as plt
        import matplotlib.dates as mdates
        from datetime import datetime

        # 转换数据为DataFrame
        equity_df = pd.DataFrame(result['equity_curve'])
        equity_df['datetime'] = pd.to_datetime(equity_df['datetime'])
        equity_df.set_index('datetime', inplace=True)

        # 计算累计收益率
        initial_equity = equity_df['equity'].iloc[0]
        equity_df['returns'] = (equity_df['equity'] / initial_equity - 1) * 100

        # 创建图表
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

        # 净值曲线 - 使用numpy数组避免多维索引问题
        dates = equity_df.index.values
        equity_values = equity_df['equity'].values

        ax1.plot(dates, equity_values, 'b-', linewidth=2, label='投资组合净值')
        ax1.set_title('投资组合净值曲线 / Portfolio Equity Curve', fontsize=14, fontweight='bold')
        ax1.set_ylabel('净值 / Equity', fontsize=12)
        ax1.grid(True, alpha=0.3)
        ax1.legend()

        # 格式化x轴日期
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45)

        # 累计收益率曲线
        returns_values = equity_df['returns'].values
        ax2.plot(dates, returns_values, 'g-', linewidth=2, label='累计收益率')
        ax2.axhline(y=0, color='r', linestyle='--', alpha=0.7)
        ax2.set_title('累计收益率 / Cumulative Returns', fontsize=12, fontweight='bold')
        ax2.set_ylabel('收益率 (%) / Returns (%)', fontsize=12)
        ax2.set_xlabel('日期 / Date', fontsize=12)
        ax2.grid(True, alpha=0.3)
        ax2.legend()

        # 格式化x轴日期
        ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45)

        # 调整布局
        plt.tight_layout()

        # 保存图表
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"净值曲线图已保存: {os.path.abspath(save_path)}")

        # 尝试显示图表（在某些环境中可能不工作）
        try:
            plt.show()
        except:
            pass

    except ImportError:
        print("警告: 无法生成图表。请安装matplotlib: pip install matplotlib")
    except Exception as e:
        print(f"生成图表时发生错误: {e}")


def main():
    """主函数"""
    # 运行回测
    result = run_portfolio_backtest(
        start_date="2022-01-01",
        end_date="2022-12-31",
        initial_cash=1_000_000.0
    )

    if result:
        # 分析和显示结果
        analyze_and_display_results(result)

        # 生成净值曲线图
        generate_equity_curve_plot(result)

        print(f"\n回测完成！")
    else:
        print("回测失败！")


if __name__ == "__main__":
    main()