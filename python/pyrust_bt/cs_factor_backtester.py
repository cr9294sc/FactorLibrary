from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import os
import math
import json
import csv
import numpy as np
import pandas as pd
from datetime import datetime


@dataclass
class BacktestConfigCS:
    """配置：截面因子分层回测（仅需当期因子与下一期收益）。"""
    factor_col: str = "factor"
    ret_next_col: str = "ret_next"  # 下一期收益（例如 next-day return）
    datetime_col: str = "datetime"
    symbol_col: str = "symbol"
    quantiles: int = 5
    winsorize: Optional[Tuple[float, float]] = None  # (0.01, 0.99)
    standardize: bool = False  # 按当日截面 z-score
    min_obs_per_day: int = 30  # 当日最少样本
    initial_equity: float = 1.0


@dataclass
class BacktestResultCS:
    """结果：分位组合与多空组合的收益/净值/统计。"""
    returns_by_quantile: Dict[int, pd.Series]
    equity_by_quantile: Dict[int, pd.Series]
    stats_by_quantile: Dict[int, Dict[str, float]]
    long_short_return: Optional[pd.Series] = None
    long_short_equity: Optional[pd.Series] = None
    long_short_stats: Optional[Dict[str, float]] = None
    turnover_by_quantile: Optional[Dict[int, pd.Series]] = None


class CrossSectionFactorBacktester:
    """
    截面因子分层回测器（不依赖交易撮合），直接用当期因子与下一期收益做组合收益/净值与评价。
    输入数据需包含：datetime, symbol, factor_col, ret_next_col。
    """

    def __init__(self, df: pd.DataFrame, cfg: BacktestConfigCS) -> None:
        self.df = df.copy()
        self.cfg = cfg
        self._validate_and_prepare()

    def _validate_and_prepare(self) -> None:
        c = self.cfg
        need_cols = {c.datetime_col, c.symbol_col, c.factor_col, c.ret_next_col}
        missing = need_cols.difference(set(self.df.columns))
        if missing:
            raise ValueError(f"missing required columns: {sorted(missing)}")

        # 标准化时间与类型
        self.df[c.datetime_col] = pd.to_datetime(self.df[c.datetime_col], errors="coerce")
        self.df = self.df.dropna(subset=[c.datetime_col, c.symbol_col])
        self.df.sort_values([c.datetime_col, c.symbol_col], inplace=True)

        # 截面内预处理（winsorize / standardize）
        fac = pd.to_numeric(self.df[c.factor_col], errors="coerce")
        self.df[c.factor_col] = fac
        if c.winsorize is not None:
            lo, hi = c.winsorize
            self.df[c.factor_col] = (
                self.df.groupby(c.datetime_col)[c.factor_col]
                .transform(lambda x: x.clip(x.quantile(lo), x.quantile(hi)))
            )
        if c.standardize:
            self.df[c.factor_col] = (
                self.df.groupby(c.datetime_col)[c.factor_col]
                .transform(lambda x: (x - x.mean()) / (x.std(ddof=0) + 1e-12))
            )

        # 保证收益为浮点
        self.df[c.ret_next_col] = pd.to_numeric(self.df[c.ret_next_col], errors="coerce")

    def _assign_quantiles(self) -> pd.Series:
        c = self.cfg
        # 逐日分桶（忽略该日 NaN）
        def _one_day(s: pd.Series) -> pd.Series:
            r = s.rank(pct=True, method="first")
            b = np.floor(r * c.quantiles).clip(0, c.quantiles - 1)
            return b.astype("Int64")

        q = (
            self.df.groupby(self.cfg.datetime_col, group_keys=False)[self.cfg.factor_col]
            .apply(_one_day)
        )
        return q

    def _compute_daily_quantile_returns(self, q_series: pd.Series) -> Dict[int, pd.Series]:
        c = self.cfg
        df = self.df.copy()
        df["__q"] = q_series
        df = df.dropna(subset=["__q"]).copy()
        # 按日期/分位聚合下一期收益（等权）
        grp = df.groupby([c.datetime_col, "__q"], sort=False)[c.ret_next_col].mean()
        # 拆成各分位的时间序列
        ret_by_q: Dict[int, pd.Series] = {}
        for q in range(c.quantiles):
            s = grp.xs(q, level=1, drop_level=False).droplevel(1) if (q in grp.index.get_level_values(1)) else pd.Series(dtype=float)
            ret_by_q[q] = s
        return ret_by_q

    def _compute_turnover(self, q_series: pd.Series) -> Dict[int, pd.Series]:
        c = self.cfg
        df = self.df.copy()
        df["__q"] = q_series
        df = df.dropna(subset=["__q"]).copy()
        turn_by_q: Dict[int, List[float]] = {q: [] for q in range(c.quantiles)}
        idx_by_q: Dict[int, List[pd.Timestamp]] = {q: [] for q in range(c.quantiles)}

        for dt, g in df.groupby(c.datetime_col, sort=False):
            for q in range(c.quantiles):
                members = set(g.loc[g["__q"] == q, c.symbol_col].astype(str).tolist())
                # 取前一日集合
                prev_dt = None
                # 简化：用上一组的日期
                # 我们在聚合后统一对齐（近似）
                turn = np.nan
                turn_by_q[q].append(turn)
                idx_by_q[q].append(dt)

        # 更严谨的 turnover 计算（逐日集合交并）
        out: Dict[int, pd.Series] = {}
        for q in range(c.quantiles):
            series = pd.Series(index=idx_by_q[q], data=turn_by_q[q], dtype=float).sort_index()
            # 用真正集合比较计算
            vals: List[float] = []
            prev_set: Optional[set] = None
            for dt, g in df.groupby(c.datetime_col, sort=True):
                cur = set(g.loc[g["__q"] == q, c.symbol_col].astype(str).tolist())
                if prev_set is None:
                    vals.append(np.nan)
                else:
                    inter = len(prev_set.intersection(cur))
                    denom = max(1, len(prev_set))
                    vals.append(1.0 - inter / denom)
                prev_set = cur
            series.loc[:] = vals
            out[q] = series
        return out

    @staticmethod
    def _compute_stats_from_returns(ret: pd.Series, trading_days: int = 252) -> Dict[str, float]:
        ret = ret.dropna()
        if ret.empty:
            return {k: 0.0 for k in [
                "total_return", "annualized_return", "volatility", "sharpe",
                "calmar", "max_drawdown", "max_dd_duration", "win_rate"
            ]}
        eq = (1.0 + ret).cumprod()
        total_return = float(eq.iloc[-1] - 1.0)
        mean_ret = float(ret.mean())
        vol = float(ret.std(ddof=0))
        ann_ret = mean_ret * trading_days
        ann_vol = vol * math.sqrt(trading_days)
        sharpe = (ann_ret) / (ann_vol + 1e-12)
        # 回撤
        peak = -np.inf
        max_dd = 0.0
        max_dd_dur = 0
        cur_dur = 0
        for v in eq.values:
            if v > peak:
                peak = v
                cur_dur = 0
            else:
                cur_dur += 1
                dd = 0.0 if peak <= 0 else (peak - v) / peak
                if dd > max_dd:
                    max_dd = dd
                if cur_dur > max_dd_dur:
                    max_dd_dur = cur_dur
        calmar = ann_ret / (max_dd + 1e-12)
        win_rate = float((ret > 0).mean())
        return {
            "total_return": total_return,
            "annualized_return": ann_ret,
            "volatility": ann_vol,
            "sharpe": sharpe,
            "calmar": calmar,
            "max_drawdown": float(max_dd),
            "max_dd_duration": float(max_dd_dur),
            "win_rate": win_rate,
        }

    def run(self, compute_long_short: bool = True) -> BacktestResultCS:
        q = self._assign_quantiles()
        ret_by_q = self._compute_daily_quantile_returns(q)
        eq_by_q: Dict[int, pd.Series] = {}
        stats_by_q: Dict[int, Dict[str, float]] = {}

        for k, sr in ret_by_q.items():
            sr = sr.sort_index()
            eq = (1.0 + sr).cumprod() * self.cfg.initial_equity
            eq_by_q[k] = eq
            stats_by_q[k] = self._compute_stats_from_returns(sr)

        ls_ret = None
        ls_eq = None
        ls_stats = None
        if compute_long_short and self.cfg.quantiles >= 2:
            hi = self.cfg.quantiles - 1
            lo = 0
            # 多空：高分位 - 低分位（等权），对齐索引并集
            a0 = ret_by_q.get(hi, pd.Series(dtype=float))
            b0 = ret_by_q.get(lo, pd.Series(dtype=float))
            idx = a0.index.union(b0.index)
            a = a0.reindex(idx)
            b = b0.reindex(idx)
            ls_ret = (a - b).dropna()
            ls_eq = (1.0 + ls_ret).cumprod() * self.cfg.initial_equity
            ls_stats = self._compute_stats_from_returns(ls_ret)

        turnover = self._compute_turnover(q)
        return BacktestResultCS(
            returns_by_quantile=ret_by_q,
            equity_by_quantile=eq_by_q,
            stats_by_quantile=stats_by_q,
            long_short_return=ls_ret,
            long_short_equity=ls_eq,
            long_short_stats=ls_stats,
            turnover_by_quantile=turnover,
        )

    def export(self, result: BacktestResultCS, out_dir: str) -> None:
        os.makedirs(out_dir, exist_ok=True)
        # 导出分位净值/收益
        for q, sr in result.returns_by_quantile.items():
            sr.to_csv(os.path.join(out_dir, f"ret_q{q}.csv"), header=["ret"], index_label="date")
        for q, sr in result.equity_by_quantile.items():
            sr.to_csv(os.path.join(out_dir, f"equity_q{q}.csv"), header=["equity"], index_label="date")
        # 多空
        if result.long_short_return is not None:
            result.long_short_return.to_csv(os.path.join(out_dir, "ret_long_short.csv"), header=["ret"], index_label="date")
        if result.long_short_equity is not None:
            result.long_short_equity.to_csv(os.path.join(out_dir, "equity_long_short.csv"), header=["equity"], index_label="date")
        # 统计
        with open(os.path.join(out_dir, "stats.json"), "w", encoding="utf-8") as f:
            json.dump({str(k): v for k, v in result.stats_by_quantile.items()}, f, indent=2, ensure_ascii=False)
        if result.long_short_stats is not None:
            with open(os.path.join(out_dir, "stats_long_short.json"), "w", encoding="utf-8") as f:
                json.dump(result.long_short_stats, f, indent=2, ensure_ascii=False)
        # 换手
        if result.turnover_by_quantile is not None:
            for q, sr in result.turnover_by_quantile.items():
                sr.to_csv(os.path.join(out_dir, f"turnover_q{q}.csv"), header=["turnover"], index_label="date")


if __name__ == "__main__":
    # 可直接运行的小样本演示：构造 100 支股票、两年、随机因子 + next-day return
    rng = np.random.default_rng(123)
    dates = pd.bdate_range("2020-01-02", "2021-12-31")
    symbols = [f"STK{i:04d}" for i in range(1, 101)]
    rows: List[Dict[str, Any]] = []
    for dt in dates:
        fac = rng.standard_normal(len(symbols))
        nxt = rng.normal(0.0005, 0.02, size=len(symbols))
        for sym, fv, rv in zip(symbols, fac, nxt):
            rows.append({
                "datetime": dt,
                "symbol": sym,
                "factor": float(fv),
                "ret_next": float(rv),
            })
    demo_df = pd.DataFrame(rows)
    config = BacktestConfigCS(factor_col="factor", ret_next_col="ret_next", quantiles=5, winsorize=(0.01, 0.99), standardize=True)
    bt = CrossSectionFactorBacktester(demo_df, config)
    res = bt.run(compute_long_short=True)
    out_dir = os.path.join(os.path.dirname(__file__), "../../examples/results", "cs_backtester_demo")
    bt.export(res, out_dir=os.path.abspath(out_dir))
    print("Exported to:", os.path.abspath(out_dir))


