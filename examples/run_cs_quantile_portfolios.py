from __future__ import annotations
from typing import Any, Dict, List
import os
import sys
import numpy as np
import pandas as pd

# 允许从 repo 根目录运行本文件
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "python"))

from pyrust_bt.api import BacktestConfig, BacktestEngine
from pyrust_bt.strategy import Strategy
from pyrust_bt.analyzers import export_trades_to_csv


def generate_cs_momentum_sample() -> pd.DataFrame:
    """
    生成 2020-01-02 ~ 2025-12-31 多股票日频样例数据，含动量因子：
      - mom_20d / mom_126d / mom_252d
    返回 DataFrame（不分桶）。
    """
    rng = np.random.default_rng(7)
    symbols = [
        "600000.SH", "600519.SH", "000001.SZ", "000651.SZ", "300750.SZ",
        "601318.SH", "002594.SZ", "600036.SH", "601988.SH", "000333.SZ",
    ]
    dates = pd.bdate_range("2020-01-02", "2025-12-31")

    frames: List[pd.DataFrame] = []
    for sym in symbols:
        n = len(dates)
        mu, sigma = 0.0003, 0.02
        rets = rng.normal(mu, sigma, size=n)
        price0 = float(rng.uniform(8.0, 120.0))
        close = price0 * np.cumprod(1.0 + rets)
        df = pd.DataFrame({"datetime": dates, "symbol": sym, "close": close})
        df["mom_20d"] = df["close"] / df["close"].shift(20) - 1.0
        df["mom_126d"] = df["close"] / df["close"].shift(126) - 1.0
        df["mom_252d"] = df["close"] / df["close"].shift(252) - 1.0
        frames.append(df)

    full = pd.concat(frames, ignore_index=True)
    full["datetime"] = pd.to_datetime(full["datetime"])  # 确保是时间类型
    return full


def add_quantile_buckets(df: pd.DataFrame, factor: str, quantiles: int = 5) -> pd.DataFrame:
    """按日期做截面分桶，生成 __q 列（0..quantiles-1，Int64 可空整型）。"""
    df = df.copy()
    # 逐日分组计算分位分桶，忽略该日 NaN 因子
    def _bucket_one_day(g: pd.DataFrame) -> pd.Series:
        x = pd.to_numeric(g[factor], errors="coerce")
        ranks_pct = x.rank(pct=True, method="first")
        bucket = np.floor(ranks_pct * quantiles).clip(0, quantiles - 1)
        return bucket.astype("Int64")

    # 仅对该日的因子列进行分桶，避免未来版本行为改变
    df["__q"] = (
        df.groupby("datetime", group_keys=False)[factor]
        .apply(lambda s: np.floor(s.rank(pct=True, method="first") * quantiles).clip(0, quantiles - 1).astype("Int64"))
    )
    return df


def build_feeds(df: pd.DataFrame) -> Dict[str, List[Dict[str, Any]]]:
    """构建 run_multi 所需的 feeds: {symbol: [bar, ...]}，每个 bar 含 symbol/datetime/open/high/low/close/volume/q。"""
    out: Dict[str, List[Dict[str, Any]]] = {}
    df = df.sort_values(["symbol", "datetime"])  # 保证顺序
    for sym, g in df.groupby("symbol", sort=False):
        rows: List[Dict[str, Any]] = []
        for _, r in g.iterrows():
            rows.append({
                "symbol": sym,
                "datetime": r["datetime"].strftime("%Y-%m-%d"),
                "open": float(r["close"]),  # 简化示例：用 close 代替 OHLC
                "high": float(r["close"]),
                "low": float(r["close"]),
                "close": float(r["close"]),
                "volume": 0.0,
                # 注意：引擎不保证透传自定义字段，这里不依赖 q 下发到回调
            })
        out[sym] = rows
    return out


def build_quantile_membership(df: pd.DataFrame, quantiles: int = 5) -> Dict[int, Dict[str, set[str]]]:
    """构建每个分位在每个交易日的成分集合映射：{q: {date_str: set(symbols)}}"""
    mem: Dict[int, Dict[str, set[str]]] = {q: {} for q in range(quantiles)}
    g = df.dropna(subset=["__q"]).groupby(["datetime", "__q"], sort=False)["symbol"]
    for (dt, q), symbols in g:
        date_str = dt.strftime("%Y-%m-%d")
        q_int = int(q)
        mem[q_int].setdefault(date_str, set()).update(map(str, symbols.tolist()))
    return mem


class QuantileRebalanceStrategy(Strategy):
    """逐日再平衡：目标分位的成分持有 1 手，其余为空仓。"""

    def __init__(self, target_q: int, membership: Dict[str, set[str]], size: float = 1.0) -> None:
        self.target_q = target_q
        self.size = size
        self._held: set[str] = set()
        self._membership_by_date = membership  # date_str -> set(symbol)

    def next_multi(self, update_slice: Dict[str, Dict[str, Any]], ctx: Dict[str, Any]) -> List[Dict[str, Any]] | None:
        # 获取此次更新的交易日（多源一致）
        any_bar = next(iter(update_slice.values())) if update_slice else None
        curr_date = str(any_bar.get("datetime")) if any_bar else None
        desired: set[str] = set() if curr_date is None else self._membership_by_date.get(curr_date, set())

        actions: List[Dict[str, Any]] = []
        # 卖出不再需要的
        for sym in list(self._held):
            if sym not in desired:
                actions.append({"action": "SELL", "type": "market", "size": self.size, "symbol": sym})
                self._held.remove(sym)
        # 买入新增的
        for sym in desired:
            if sym not in self._held:
                actions.append({"action": "BUY", "type": "market", "size": self.size, "symbol": sym})
                self._held.add(sym)

        return actions or None


def run_quantile_portfolios(feeds: Dict[str, List[Dict[str, Any]]], memberships: Dict[int, Dict[str, set[str]]], quantiles: int = 5) -> None:
    cfg = BacktestConfig(
        start=min(b["datetime"] for rows in feeds.values() for b in rows),
        end=max(b["datetime"] for rows in feeds.values() for b in rows),
        cash=1_000_000.0,
        commission_rate=0.0005,
        slippage_bps=2.0,
        batch_size=2000,
    )
    engine = BacktestEngine(cfg)

    out_dir = os.path.join(os.path.dirname(__file__), "cs_quantile_runs")
    os.makedirs(out_dir, exist_ok=True)

    for q in range(quantiles):
        strat = QuantileRebalanceStrategy(target_q=q, membership=memberships.get(q, {}), size=1.0)
        result = engine.run_multi(strat, feeds)

        print(f"Quantile {q}: stats=", result.get("stats"))

        # 导出交易清单（如引擎返回了 trades）
        trades = result.get("trades")
        if trades:
            export_trades_to_csv(trades, os.path.join(out_dir, f"trades_q{q}.csv"))

        # 导出净值曲线
        eq = result.get("equity_curve") or result.get("equity")
        if eq:
            # equity_curve: list[dict]; equity: list[float] or list[dict]
            df_eq = pd.DataFrame(eq)
            df_eq.to_csv(os.path.join(out_dir, f"equity_q{q}.csv"), index=False)


def main() -> None:
    # 1) 生成数据 + 计算分桶（以 mom_126d 为例）
    df = generate_cs_momentum_sample()
    df = add_quantile_buckets(df, factor="mom_126d", quantiles=5)

    # 2) 构建 feeds 并回测各分位组合
    feeds = build_feeds(df)
    memberships = build_quantile_membership(df, quantiles=5)
    run_quantile_portfolios(feeds, memberships, quantiles=5)


if __name__ == "__main__":
    main()


