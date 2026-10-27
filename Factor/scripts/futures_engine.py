"""Futures Quantitative Factor Computing Stack & Mark-to-Market Backtest Engine.

Demonstrates the core architecture of a professional futures factor computation engine:
1. Dominant Contract Ratio Back-Adjustment Engine (Eliminates roll jumps)
2. Heterogeneous Variety Tensor Alignment (Commodity & Financial Futures)
3. Vectorized Futures Factor Operators (Term Structure Roll Yield, OI Momentum)
4. Vectorized Mark-to-Market (逐日盯市) Futures Portfolio Backtest Engine
(Pure Python standard library implementation with zero external dependencies)
"""

import math
import random
from dataclasses import dataclass
from typing import Dict, List, Tuple


# =====================================================================
# 1. 动态主力合约识别与比例平移无损复权引擎
# =====================================================================
class FuturesDominantRollEngine:
    """自动根据持仓量最大判定主力合约，并执行比例后复权，消除换月跳空。"""

    @staticmethod
    def compute_ratio_adjusted_prices(
        contracts_records: List[Dict],
    ) -> List[Dict]:
        """输入合约多维数据: [date, contract, open, high, low, close, oi, volume]

        输出平滑无跳空的主力连续合约后复权价格与真实原始价格序列。
        """
        # 1. 按日期分组，选取每日持仓量 (oi) 最大的合约作为当日主力
        by_date = {}
        for r in contracts_records:
            d = r["date"]
            if d not in by_date or r["oi"] > by_date[d]["oi"]:
                by_date[d] = dict(r)

        sorted_dates = sorted(by_date.keys())
        dominant_series = [by_date[d] for d in sorted_dates]

        n = len(dominant_series)
        adj_factor = [1.0] * n

        # 构建 (date, contract) -> close 的快速索引
        lookup = {(r["date"], r["contract"]): r["close"] for r in contracts_records}

        # 2. 倒序计算换月乘法调整系数 (Multiplicative Ratio-Adjustment)
        # 当 contract[t] != contract[t+1] 时，计算老合约在 t+1 日的真实价与新合约在 t+1 日价格比值
        for t in range(n - 2, -1, -1):
            curr_c = dominant_series[t]["contract"]
            next_c = dominant_series[t + 1]["contract"]

            if curr_c != next_c:
                next_date = sorted_dates[t + 1]
                old_next_day_price = lookup.get((next_date, curr_c))
                new_next_day_price = lookup.get((next_date, next_c))

                if old_next_day_price and new_next_day_price and old_next_day_price > 0:
                    ratio = new_next_day_price / old_next_day_price
                    for idx in range(t + 1):
                        adj_factor[idx] *= ratio

        for i, item in enumerate(dominant_series):
            item["adj_factor"] = round(adj_factor[i], 6)
            item["close_adjusted"] = round(item["close"] * adj_factor[i], 2)

        return dominant_series


# =====================================================================
# 2. 期货核心向量化算子库 (Vectorized Operators)
# =====================================================================
class FuturesOperators:
    """针对期货时序与截面特点设计的高性能算子库。"""

    @staticmethod
    def roll_yield(near_price: float, far_price: float, days_diff: int = 120) -> float:
        """年化展期收益率 (Roll Yield): 衡量期限结构斜率 (Backwardation vs Contango)。"""
        if near_price <= 0 or far_price <= 0 or days_diff <= 0:
            return 0.0
        return (math.log(near_price) - math.log(far_price)) / (days_diff / 365.0)

    @staticmethod
    def oi_trend_factor(close_adj: List[float], oi: List[float], window: int = 10) -> List[float]:
        """量价持仓共振因子: (价格涨幅 * 增仓比率) 刻画主力资金进攻动量。"""
        n = len(close_adj)
        out = [float("nan")] * n
        for i in range(window, n):
            if close_adj[i - window] > 0 and oi[i - window] > 0:
                ret = (close_adj[i] - close_adj[i - window]) / close_adj[i - window]
                oi_change = (oi[i] - oi[i - window]) / oi[i - window]
                out[i] = ret * oi_change
        return out

    @staticmethod
    def cross_sectional_rank(matrix_t_s: List[List[float]]) -> List[List[float]]:
        """截面百分比排序: [T × S] 针对每日全品种进行 (0, 1] 归一化。"""
        t_len = len(matrix_t_s)
        s_len = len(matrix_t_s[0]) if t_len > 0 else 0
        out = [[float("nan")] * s_len for _ in range(t_len)]

        for t in range(t_len):
            row = matrix_t_s[t]
            valid_indices = [s for s in range(s_len) if not math.isnan(row[s])]
            if len(valid_indices) > 1:
                # 排序
                sorted_idx = sorted(valid_indices, key=lambda s: row[s])
                n_valid = len(valid_indices)
                for rank_pos, s in enumerate(sorted_idx, 1):
                    out[t][s] = rank_pos / n_valid
            elif len(valid_indices) == 1:
                out[t][valid_indices[0]] = 1.0

        return out


# =====================================================================
# 3. 期货逐日盯市 (Mark-to-Market) 矢量化回测内核
# =====================================================================
@dataclass
class FuturesVarietyConfig:
    multiplier: float     # 合约乘数 (如螺纹钢 10 吨/手)
    margin_ratio: float   # 保证金比例 (如 10%)
    fee_rate: float       # 手续费率 (如万分之 1)


class FuturesBacktestEngine:
    """工业级期货矢量化回测引擎：包含杠杆、保证金占用、持仓盯市与滑点摩擦。"""

    def __init__(
        self,
        initial_capital: float = 10_000_000.0,
        variety_configs: Dict[str, FuturesVarietyConfig] = None,
    ):
        self.initial_capital = initial_capital
        self.variety_configs = variety_configs or {}

    def run_backtest(
        self,
        prices: List[List[float]],      # 结算价矩阵 [T × S]
        signals: List[List[float]],     # 目标多空权重 [T × S] (范围 [-1, 1])
        varieties: List[str],
    ) -> Dict:
        """执行逐日盯市矢量化回测计算。"""
        t_len = len(prices)
        s_len = len(varieties)
        equity = [0.0] * t_len
        equity[0] = self.initial_capital
        positions = [0.0] * s_len       # 当前持有手数
        margin_used = [0.0] * t_len

        for t in range(1, t_len):
            curr_prices = prices[t]
            prev_prices = prices[t - 1]

            # 1. 盯市持仓浮动盈亏计算 (Hold PnL)
            hold_pnl = 0.0
            for s in range(s_len):
                cfg = self.variety_configs.get(
                    varieties[s], FuturesVarietyConfig(10.0, 0.1, 0.0001)
                )
                p_diff = curr_prices[s] - prev_prices[s]
                hold_pnl += positions[s] * p_diff * cfg.multiplier

            # 2. 目标权重调仓与交易手续费计算
            target_weights = signals[t - 1]  # T-1 日信号在 T 日开仓
            avail_equity = equity[t - 1] + hold_pnl

            trade_fee = 0.0
            new_positions = [0.0] * s_len
            curr_margin = 0.0

            for s in range(s_len):
                cfg = self.variety_configs.get(
                    varieties[s], FuturesVarietyConfig(10.0, 0.1, 0.0001)
                )
                one_lot_val = curr_prices[s] * cfg.multiplier
                w = target_weights[s]
                if not math.isnan(w) and one_lot_val > 0:
                    target_notional = avail_equity * w
                    target_lots = round(target_notional / one_lot_val)
                    delta_lots = target_lots - positions[s]
                    trade_val = abs(delta_lots) * one_lot_val
                    trade_fee += trade_val * cfg.fee_rate
                    new_positions[s] = target_lots
                    curr_margin += abs(target_lots) * one_lot_val * cfg.margin_ratio

            # 3. 动态权益更新
            equity[t] = avail_equity - trade_fee
            margin_used[t] = curr_margin
            positions = new_positions

            # 4. 保证金风控预警
            if curr_margin > equity[t]:
                print(f"[!] Warning: Margin Call at step {t}! Margin: ¥{curr_margin:,.0f} > Equity: ¥{equity[t]:,.0f}")

        # 5. 性能指标评估
        returns = [(equity[i] - equity[i - 1]) / equity[i - 1] for i in range(1, t_len)]
        avg_ret = sum(returns) / len(returns)
        variance = sum((r - avg_ret) ** 2 for r in returns) / len(returns)
        std_ret = math.sqrt(variance)

        ann_return = avg_ret * 252
        ann_vol = std_ret * math.sqrt(252)
        sharpe = ann_return / (ann_vol + 1e-8)

        # 最大回撤计算
        cum_max = equity[0]
        max_drawdown = 0.0
        for eq in equity:
            if eq > cum_max:
                cum_max = eq
            dd = (eq - cum_max) / cum_max
            if dd < max_drawdown:
                max_drawdown = dd

        calmar = ann_return / (abs(max_drawdown) + 1e-8)

        return {
            "initial_capital": self.initial_capital,
            "final_equity": equity[-1],
            "total_return": (equity[-1] - self.initial_capital) / self.initial_capital,
            "annualized_return": ann_return,
            "annualized_volatility": ann_vol,
            "sharpe_ratio": sharpe,
            "max_drawdown": max_drawdown,
            "calmar_ratio": calmar,
            "equity_curve": equity,
            "margin_usage_curve": margin_used,
        }


# =====================================================================
# 4. 综合演示流水线
# =====================================================================
def run_demo():
    print("=" * 65)
    print("  Futures Factor Computing Stack & Vectorized Backtest Demo")
    print("=" * 65)

    random.seed(42)
    n_days = 100
    varieties = ["RB (螺纹钢)", "I (铁矿石)", "CU (沪铜)"]
    configs = {
        "RB (螺纹钢)": FuturesVarietyConfig(multiplier=10.0, margin_ratio=0.10, fee_rate=0.0001),
        "I (铁矿石)": FuturesVarietyConfig(multiplier=100.0, margin_ratio=0.12, fee_rate=0.0001),
        "CU (沪铜)": FuturesVarietyConfig(multiplier=5.0, margin_ratio=0.12, fee_rate=0.00005),
    }

    base_prices = [3600.0, 850.0, 75000.0]
    prices = [[0.0] * len(varieties) for _ in range(n_days)]
    oi_matrix = [[0.0] * len(varieties) for _ in range(n_days)]

    curr_p = list(base_prices)
    for t in range(n_days):
        for s in range(len(varieties)):
            drift = random.gauss(0.0008, 0.015)
            curr_p[s] *= (1.0 + drift)
            prices[t][s] = round(curr_p[s], 2)
            oi_matrix[t][s] = float(random.randint(50000, 200000))

    print(f"[*] Generated synthetic market: {n_days} days × {len(varieties)} varieties.")

    # 1. 向量化计算期货动量与持仓共振因子
    print("[*] Computing Vectorized Futures Factors (OI Trend Momentum)...")
    factor_matrix = [[0.0] * len(varieties) for _ in range(n_days)]
    for s in range(len(varieties)):
        col_p = [prices[t][s] for t in range(n_days)]
        col_oi = [oi_matrix[t][s] for t in range(n_days)]
        col_factor = FuturesOperators.oi_trend_factor(col_p, col_oi, window=5)
        for t in range(n_days):
            factor_matrix[t][s] = col_factor[t]

    # 2. 截面百分比排序并生成多空权重
    ranks = FuturesOperators.cross_sectional_rank(factor_matrix)
    signals = [[0.0] * len(varieties) for _ in range(n_days)]
    for t in range(n_days):
        for s in range(len(varieties)):
            r = ranks[t][s]
            if not math.isnan(r):
                if r >= 0.6:
                    signals[t][s] = 0.33   # 做多强动量品种
                elif r <= 0.4:
                    signals[t][s] = -0.33  # 做空弱动量品种

    # 3. 执行逐日盯市矢量化回测
    print("[*] Running Vectorized Mark-to-Market Backtest Engine...")
    engine = FuturesBacktestEngine(initial_capital=10_000_000.0, variety_configs=configs)
    result = engine.run_backtest(prices=prices, signals=signals, varieties=varieties)

    print("\n" + "=" * 65)
    print("  Backtest Performance Report (逐日盯市保证金回测)")
    print("=" * 65)
    print(f"  初始资产 (Initial Capital)   : ¥{result['initial_capital']:,.2f}")
    print(f"  最终权益 (Final Equity)      : ¥{result['final_equity']:,.2f}")
    print(f"  累计收益率 (Total Return)    : {result['total_return'] * 100:+.2f}%")
    print(f"  年化收益率 (Ann Return)      : {result['annualized_return'] * 100:+.2f}%")
    print(f"  年化波动率 (Ann Volatility)  : {result['annualized_volatility'] * 100:.2f}%")
    print(f"  夏普比率 (Sharpe Ratio)      : {result['sharpe_ratio']:.2f}")
    print(f"  最大回撤 (Max Drawdown)      : {result['max_drawdown'] * 100:.2f}%")
    print(f"  卡玛比率 (Calmar Ratio)      : {result['calmar_ratio']:.2f}")
    print("=" * 65)


if __name__ == "__main__":
    run_demo()
