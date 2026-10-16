from __future__ import annotations
from typing import Any, Dict, List
import os
import sys
import numpy as np
import pandas as pd


# 允许从 repo 根目录运行本文件
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "python"))

from pyrust_bt.multi_factor_analyzer import MultiFactorAnalyzer


def generate_cs_momentum_sample() -> List[Dict[str, Any]]:
    """
    生成 2020-01-02 ~ 2025-12-31 的多股票日频样例数据，并计算动量因子：
      - mom_20d: 20日动量
      - mom_126d: 126日动量（约半年）
      - mom_252d: 252日动量（约一年）
    返回值为 bars 列表（dict），可直接喂给 MultiFactorAnalyzer。
    """
    rng = np.random.default_rng(42)
    symbols = [
        "600000.SH",
        "600519.SH",
        "000001.SZ",
        "000651.SZ",
        "300750.SZ",
        "601318.SH",
        "002594.SZ",
        "600036.SH",
        "601988.SH",
        "000333.SZ",
    ]

    dates = pd.bdate_range("2020-01-02", "2025-12-31")  # 工作日

    frames: List[pd.DataFrame] = []
    for sym in symbols:
        n = len(dates)
        # 简单几何布朗运动近似：每日收益 ~ N(mu, sigma)
        mu = 0.0003
        sigma = 0.02
        rets = rng.normal(mu, sigma, size=n)
        price0 = float(rng.uniform(8.0, 120.0))
        close = price0 * np.cumprod(1.0 + rets)

        df = pd.DataFrame({
            "datetime": dates,
            "symbol": sym,
            "close": close,
        })

        # 计算动量因子（shift 之后才有值，起始段为 NaN）
        df["mom_20d"] = df["close"] / df["close"].shift(20) - 1.0
        df["mom_126d"] = df["close"] / df["close"].shift(126) - 1.0
        df["mom_252d"] = df["close"] / df["close"].shift(252) - 1.0

        frames.append(df)

    full = pd.concat(frames, ignore_index=True)
    # 转为字符串日期，便于跨平台/序列化
    full["datetime"] = full["datetime"].dt.strftime("%Y-%m-%d")
    # 只保留需要列
    cols = ["datetime", "symbol", "close", "mom_20d", "mom_126d", "mom_252d"]
    full = full[cols]
    return full.to_dict("records")


def main() -> None:
    # 1) 生成样例数据
    bars = generate_cs_momentum_sample()

    # 2) 构建分析器（横截面口径：逐日分桶 + 截面IC）
    mfa = MultiFactorAnalyzer(
        bars,
        datetime_col="datetime",
        symbol_col="symbol",
        window_size=60,
        nan_policy="drop",
        winsorize=(0.01, 0.99),
        standardize=True,
        horizon_unit="bars",
    )

    # 3) 进行因子评价（分位数=5，前瞻期：1/5/20 个交易日）
    report = mfa.analyze_all_factors(
        quantiles=5,
        forward_periods=[1, 5, 20],
        method="cs",
    )

    # 4) 打印结果摘要
    print("=== Factor Ranking (Top 5) ===")
    for i, (name, score) in enumerate(report.factor_ranking[:5], start=1):
        print(f"{i:02d}. {name}: score={score:.4f}")

    print("\n=== Summary Stats ===")
    for k, v in report.summary_stats.items():
        print(f"{k}: {v}")

    # 5) 可选：导出详细报告
    out_dir = os.path.join(os.path.dirname(__file__), "results", "cs_momentum_reports")
    mfa.export_report(report, output_dir=out_dir)
    print(f"\nReports exported to: {out_dir}")


if __name__ == "__main__":
    main()


