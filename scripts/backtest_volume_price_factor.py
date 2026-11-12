#!/usr/bin/env python3
"""Volume_Price_2_H5_2 因子本地股票回测与 FactorLab 线上对账脚本 (scripts/backtest_volume_price_factor.py)。

步骤:
1. 通过 FactorLab MCP 接口拉取线上回测基准结果 (Run ID: bt_bb3a04da6b20456ab444dcb7e5450949)
2. 通过 StockPanelLoader 加载本地 A 股全市场历史数据 (data/raw/stocks/, 2015-11 ~ 2024-12)
3. 使用 factorlib_core Rust 计算内核执行因子表达式求值与 6 阶段后处理 (严格应用 LimitMask 与 BaseUnivMask)
4. 执行单因子回测 (IC/RankIC, 10 分组收益, 多空组合, 多头超额 vs 000985.CSI, 换手率, 基于 open_adj 早盘基准)
5. 生成线上 vs 本地高精度对账对比报告
"""

from __future__ import annotations

import os
import sys
import time
from datetime import date
from typing import Any

import factorlib_core
import numpy as np

# 引入本地 FactorLab 客户端与核心评价指标
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "python")))
from factorlib.core.evaluate import calc_long_excess_metrics  # noqa: E402
from factorlib.core.mcp_client import call_mcp_tool  # noqa: E402
from factorlib.markets.stocks import StockPanelData, StockPanelLoader  # noqa: E402


def fetch_online_factorlab_results() -> dict[str, Any]:
    """从 FactorLab MCP 服务拉取 Volume_Price_2_H5_2 线上回测结果。"""
    print("[1/5] 正在通过 MCP 调取 FactorLab 线上回测数据...")
    factor_id = "fac_5bc39b9aa01c427aa2d28563d7d9b5a2"
    factor_info = call_mcp_tool("get_factor", {"factor_id": factor_id})
    latest_run_id = factor_info.get("data", {}).get("latest_run", {}).get("run_id")
    print(f"      线上因子名称: {factor_info.get('data', {}).get('name')}")
    print(f"      最新回测 Run ID: {latest_run_id}")

    summary = call_mcp_tool("get_result_summary", {"run_id": latest_run_id})
    data = summary.get("data", {})
    return data


def load_local_stock_data(
    start_date: date = date(2015, 11, 1),
    end_date: date = date(2024, 12, 31),
    exec_price_mode: str = "open_adj",
) -> StockPanelData:
    """通过 StockPanelLoader 加载本地全市场日频行情与基础信息。"""
    print("[2/5] 正在通过 StockPanelLoader 构建 A 股标准截面面板与掩码矩阵...")
    loader = StockPanelLoader("data/raw/stocks")
    panel_data = loader.load(
        start_date=start_date,
        end_date=end_date,
        exec_price_mode=exec_price_mode,
    )
    print(
        f"      日期网格: {len(panel_data.dates)} 天 ({panel_data.dates[0]} 至 {panel_data.dates[-1]}), "
        f"全市场标的: {len(panel_data.symbols)} 只, 耗时: {panel_data.meta.get('load_cost_seconds', 0.0):.2f} 秒"
    )
    return panel_data


def compute_factor_and_postprocess(
    dates: list[str],
    symbols: list[str],
    panel_data: StockPanelData,
) -> np.ndarray:
    """使用 factorlib_core Rust 计算内核执行表达式计算与 6 阶段后处理。"""
    print("[3/5] 正在通过 factorlib_core Rust 计算内核执行因子求值与后处理流水线...")
    t0 = time.time()

    # 1. 基础股票池有效掩码 (绑定 BaseUnivMask 剔除 ST、停牌与存续期外标的)
    base_mask = np.asfortranarray(panel_data.base_univ_mask)

    # 2. Rust Engine 求值
    fields = {
        "close_hfq": panel_data["close_hfq"],
        "log_amount": panel_data["log_amount"],
        "c2c_ret": panel_data["c2c_ret"],
    }
    engine = factorlib_core.Engine(fields, base_mask)

    # 表达式: neg(add(cs_zscore(ts_corr(close_hfq, log_amount, 20)), cs_zscore(ts_std(c2c_ret, 20))))
    expr = "neg(add(cs_zscore(ts_corr(close_hfq, log_amount, 20)), cs_zscore(ts_std(c2c_ret, 20))))"
    raw_factor = np.asarray(engine.eval(expr))
    print(f"      - [阶段 3] 因子表达式 AST 求值完毕 (有效点数: {np.sum(np.isfinite(raw_factor))})")

    # 3. 阶段四: 后处理流水线 (对齐 FactorLab postprocess)
    # 步骤 1: trdMask (过滤涨跌停与一字板不可交易标的)
    f_step1 = factorlib_core.apply_postprocess_op("trdMask", raw_factor, x2=panel_data.limit_mask)

    # 步骤 2: dropNewStock(120) (剔除上市未满 120 自然日的次新股)
    f_step2 = factorlib_core.apply_postprocess_op(
        "dropNewStock", f_step1, x2=panel_data.stock_listed_days, param=120.0
    )

    # 步骤 3: MAD(3) (3 倍中位数绝对偏差去极值)
    f_step3 = factorlib_core.apply_postprocess_op("MAD", f_step2, param=3.0)

    # 步骤 4: indNeut (申万一级行业截面残差中性化)
    f_step4 = factorlib_core.apply_postprocess_op(
        "indNeut", f_step3, x2=panel_data["industry_code"]
    )

    # 步骤 5: decay(5) (5日线性衰减平滑)
    f_step5 = factorlib_core.apply_postprocess_op("decay", f_step4, param=5.0)

    # 步骤 6: rank (截面百分比排序归一化)
    final_factor = factorlib_core.apply_postprocess_op("rank", f_step5)

    print(f"      - [阶段 4] 6 步后处理算子执行完毕 (耗时: {time.time() - t0:.2f}s)")
    return final_factor


def run_local_backtest_evaluation(
    dates: list[str],
    symbols: list[str],
    factor_mat: np.ndarray,
    panel_data: StockPanelData,
    start_eval_date: str = "20160104",
    end_eval_date: str = "20241231",
    exec_price_mode: str = "open_adj",
) -> dict[str, Any]:
    """执行 2016-2024 全周期与分年度单因子回测评估。"""
    print(
        f"[4/5] 正在执行全周期 ({start_eval_date} 至 {end_eval_date}) 与分年度单因子回测 (模式: {exec_price_mode})..."
    )
    t0 = time.time()

    # 1. 远期收益率计算: 按照 FactorLab 规则采用早盘代理 (默认 open_adj)
    fwd_returns = panel_data.compute_forward_returns(
        horizons=[1, 2, 5, 10],
        exec_price_mode=exec_price_mode,
        execution_simulation=False,
    )
    fwd_ret_1d = fwd_returns[1]
    fwd_ret_2d = fwd_returns[2]
    fwd_ret_5d = fwd_returns[5]
    fwd_ret_10d = fwd_returns[10]

    # 2. 切片到评估区间 [20160104, 20241231]
    eval_indices = [i for i, d in enumerate(dates) if start_eval_date <= d <= end_eval_date]
    sub_dates = [dates[i] for i in eval_indices]
    sub_factor = factor_mat[eval_indices]
    sub_ret_1d = fwd_ret_1d[eval_indices]
    sub_ret_2d = fwd_ret_2d[eval_indices]
    sub_ret_5d = fwd_ret_5d[eval_indices]
    sub_ret_10d = fwd_ret_10d[eval_indices]
    sub_bench_ret = panel_data.bench_ret[eval_indices]

    # 有效回测掩码 (因子有效且收益率有效)
    eval_mask = np.isfinite(sub_factor) & np.isfinite(sub_ret_1d)

    # 3. 计算 IC 与 RankIC
    ic_res = factorlib_core.calc_ic(sub_factor, sub_ret_1d, eval_mask, rank=False, horizon=1)
    rankic_res = factorlib_core.calc_ic(sub_factor, sub_ret_1d, eval_mask, rank=True, horizon=1)

    eval_mask_5d = np.isfinite(sub_factor) & np.isfinite(sub_ret_5d)
    ic_5d_res = factorlib_core.calc_ic(sub_factor, sub_ret_5d, eval_mask_5d, rank=False, horizon=5)
    rankic_5d_res = factorlib_core.calc_ic(
        sub_factor, sub_ret_5d, eval_mask_5d, rank=True, horizon=5
    )

    eval_mask_2d = np.isfinite(sub_factor) & np.isfinite(sub_ret_2d)
    rankic_2d_res = factorlib_core.calc_ic(
        sub_factor, sub_ret_2d, eval_mask_2d, rank=True, horizon=2
    )

    eval_mask_10d = np.isfinite(sub_factor) & np.isfinite(sub_ret_10d)
    rankic_10d_res = factorlib_core.calc_ic(
        sub_factor, sub_ret_10d, eval_mask_10d, rank=True, horizon=10
    )

    # 4. 计算 10 分组多空收益
    q_res = factorlib_core.calc_quantile_metrics(
        sub_factor, sub_ret_1d, eval_mask, n_quantiles=10, horizon=1
    )

    # 5. 计算组合多头 (G10) 权重换手率与全截面秩自相关
    top_quantile_turnover = factorlib_core.calc_top_quantile_turnover(
        sub_factor, eval_mask, n_quantiles=10, top_quantile=9
    )
    rank_autocorr_turnover = factorlib_core.calc_turnover(sub_factor, eval_mask)

    # 6. 计算标准多头超额收益
    g10_daily_ret = np.asarray(q_res["group_returns"])[:, 9]
    excess_metrics = calc_long_excess_metrics(
        long_returns=g10_daily_ret,
        bench_returns=sub_bench_ret,
        dates=sub_dates,
    )

    # 7. 分年度回测统计
    years = sorted({d[:4] for d in sub_dates})
    annual_breakdown = []
    for y in years:
        y_indices = [i for i, d in enumerate(sub_dates) if d.startswith(y)]
        y_factor = sub_factor[y_indices]
        y_ret_1d = sub_ret_1d[y_indices]
        y_mask = eval_mask[y_indices]
        y_bench = sub_bench_ret[y_indices]
        y_dates = [sub_dates[i] for i in y_indices]

        y_ic = factorlib_core.calc_ic(y_factor, y_ret_1d, y_mask, rank=False, horizon=1)
        y_rankic = factorlib_core.calc_ic(y_factor, y_ret_1d, y_mask, rank=True, horizon=1)
        y_q = factorlib_core.calc_quantile_metrics(
            y_factor, y_ret_1d, y_mask, n_quantiles=10, horizon=1
        )
        y_turnover = factorlib_core.calc_top_quantile_turnover(
            y_factor, y_mask, n_quantiles=10, top_quantile=9
        )
        y_g10_ret = np.asarray(y_q["group_returns"])[:, 9]
        y_excess = calc_long_excess_metrics(
            long_returns=y_g10_ret,
            bench_returns=y_bench,
            dates=y_dates,
        )

        annual_breakdown.append(
            {
                "year": y,
                "days": len(y_indices),
                "ic": y_ic["ic_mean"],
                "ic_ir": y_ic["ic_ir"],
                "rankic": y_rankic["ic_mean"],
                "rankic_ir": y_rankic["ic_ir"],
                "rankic_win_rate": y_rankic["ic_win_rate"],
                "long_short": y_q["long_short_mean"],
                "long_short_ir": y_q["long_short_ir"],
                "long_short_sharpe": y_q["long_short_sharpe"],
                "long_short_max_dd": y_q["long_short_max_drawdown"],
                "excess_annual_ret": y_excess["excess_annual_return"],
                "excess_sharpe": y_excess["excess_sharpe"],
                "excess_max_dd": y_excess["excess_max_drawdown"],
                "excess_win_rate": y_excess["excess_win_rate"],
                "turnover": y_turnover,
            }
        )

    cost_time = time.time() - t0
    print(f"      - 回测评估完毕 (耗时: {cost_time:.2f}s)")

    return {
        "dates": sub_dates,
        "n_days": len(sub_dates),
        "ic": ic_res["ic_mean"],
        "ic_ir": ic_res["ic_ir"],
        "ic_win_rate": ic_res["ic_win_rate"],
        "5dic": ic_5d_res["ic_mean"],
        "5dic_ir": ic_5d_res["ic_ir"],
        "rankic": rankic_res["ic_mean"],
        "rankic_ir": rankic_res["ic_ir"],
        "rankic_win_rate": rankic_res["ic_win_rate"],
        "rankic_2d": rankic_2d_res["ic_mean"],
        "rankic_5d": rankic_5d_res["ic_mean"],
        "rankic_10d": rankic_10d_res["ic_mean"],
        "long_short": q_res["long_short_mean"],
        "long_short_ir": q_res["long_short_ir"],
        "long_short_sharpe": q_res["long_short_sharpe"],
        "long_short_max_dd": q_res["long_short_max_drawdown"],
        "excess_annual_ret": excess_metrics["excess_annual_return"],
        "excess_sharpe": excess_metrics["excess_sharpe"],
        "excess_max_dd": excess_metrics["excess_max_drawdown"],
        "excess_win_rate": excess_metrics["excess_win_rate"],
        "turnover": top_quantile_turnover,
        "rank_autocorr_turnover": rank_autocorr_turnover,
        "group_returns": q_res["group_returns"],
        "annual_breakdown": annual_breakdown,
    }


def print_comparison_report(online_data: dict[str, Any], local_data: dict[str, Any]):
    """打印线上 FactorLab 与本地 Rust 计算内核的对比报告。"""
    print("\n" + "=" * 90)
    print(" 📊 FactorLab 线上平台 vs 本地 Rust 内核全量回测对账报告 (P0-11 修正版)")
    print("=" * 90)

    on_m = online_data.get("metrics", {})
    tables = online_data.get("tables", {})

    print(f"\n【1. 全周期概览 (2016-01-04 至 2024-12-31, 评估交易日: {local_data['n_days']})】")
    print(
        f"{'指标名称':<25} | {'线上 FactorLab':<18} | {'本地 Rust 内核':<18} | {'绝对/相对偏差':<15}"
    )
    print("-" * 85)

    metrics_to_compare = [
        ("RankIC (截面秩相关)", "rankic", local_data["rankic"], "{:.4f}"),
        ("RankIC IR", "rankicIR", local_data["rankic_ir"], "{:.4f}"),
        ("RankIC 胜率", "rankicWR", local_data["rankic_win_rate"], "{:.2%}"),
        ("IC (Pearson 线性相关)", "ic", local_data["ic"], "{:.4f}"),
        ("IC IR", "icIR", local_data["ic_ir"], "{:.4f}"),
        ("5日 IC", "5dic", local_data["5dic"], "{:.4f}"),
        ("多空日均收益", "long_short", local_data["long_short"], "{:.4%}"),
        ("多空 IR", "long_shortIR", local_data["long_short_ir"], "{:.4f}"),
        ("Top 分位换手率", None, local_data["turnover"], "{:.2%}"),
        ("截面秩自相关距离", None, local_data["rank_autocorr_turnover"], "{:.2%}"),
    ]

    on_excess_row = None
    for row in tables.get("long_excess_analysis", []):
        if row.get("period_type") == "full_period":
            on_excess_row = row
            break

    for label, on_key, loc_val, fmt in metrics_to_compare:
        if on_key is not None:
            on_val = on_m.get(on_key)
        elif label == "Top 分位换手率" and on_excess_row:
            on_val = on_excess_row.get("turnover")
        else:
            on_val = None

        on_str = fmt.format(on_val) if on_val is not None else "N/A"
        loc_str = fmt.format(loc_val) if loc_val is not None else "N/A"
        diff_str = (
            f"{loc_val - on_val:+.4f}"
            if on_val is not None and isinstance(on_val, (int, float))
            else "-"
        )
        print(f"{label:<25} | {on_str:<18} | {loc_str:<18} | {diff_str:<15}")

    if on_excess_row:
        print("\n【2. 多头组合 (G10) 相对中证全指 (000985.CSI) 超额表现对比】")
        print(
            f"{'指标名称':<25} | {'线上 FactorLab':<18} | {'本地 Rust 内核':<18} | {'绝对/相对偏差':<15}"
        )
        print("-" * 85)
        excess_cmps = [
            ("年化超额收益率", "annual_return", local_data["excess_annual_ret"], "{:.2%}"),
            ("超额夏普比率", "sharpe", local_data["excess_sharpe"], "{:.2f}"),
            ("超额最大回撤", "max_drawdown", local_data["excess_max_dd"], "{:.2%}"),
            ("超额胜率", "win_rate", local_data["excess_win_rate"], "{:.2%}"),
        ]
        for label, k, loc_val, fmt in excess_cmps:
            on_val = on_excess_row.get(k)
            on_str = fmt.format(on_val) if on_val is not None else "N/A"
            loc_str = fmt.format(loc_val) if loc_val is not None else "N/A"
            diff_str = (
                f"{loc_val - on_val:+.4f}"
                if on_val is not None and isinstance(on_val, (int, float))
                else "-"
            )
            print(f"{label:<25} | {on_str:<18} | {loc_str:<18} | {diff_str:<15}")

    print("\n【3. 多周期 RankIC 结构与单调性】")
    print(f"  - 1日 RankIC:  {local_data['rankic']:.4f}")
    print(f"  - 2日 RankIC:  {local_data['rankic_2d']:.4f}")
    print(f"  - 5日 RankIC:  {local_data['rankic_5d']:.4f}")
    print(f"  - 10日 RankIC: {local_data['rankic_10d']:.4f}")

    print("\n【4. 分年度对账细分 (2016 - 2024)】")
    print(
        f"{'年份':<6} | {'天数':<4} | {'RankIC':<8} | {'IR':<6} | {'胜率':<7} | {'多空日收益':<9} | {'年化超额':<9} | {'超额夏普':<8} | {'换手率':<8}"
    )
    print("-" * 90)
    for row in local_data["annual_breakdown"]:
        print(
            f"{row['year']:<6} | {row['days']:<4} | {row['rankic']:<8.4f} | {row['rankic_ir']:<6.2f} | "
            f"{row['rankic_win_rate']:<7.1%} | {row['long_short']:<9.2%} | {row['excess_annual_ret']:<9.2%} | "
            f"{row['excess_sharpe']:<8.2f} | {row['turnover']:<8.2%}"
        )
    print("=" * 90)


def main():
    print("=" * 90)
    print(" 🚀 Volume_Price_2_H5_2 因子 A 股全市场本地回测与全量对账启动 (P0-11 修正版)")
    print("=" * 90)

    # 1. 抓取线上数据
    online_data = fetch_online_factorlab_results()

    # 2. 构建本地标准 A 股面板
    panel_data = load_local_stock_data(
        start_date=date(2015, 11, 1),
        end_date=date(2024, 12, 31),
        exec_price_mode="open_adj",
    )
    dates = panel_data.dates
    symbols = panel_data.symbols

    # 3. 计算因子
    factor_mat = compute_factor_and_postprocess(dates, symbols, panel_data)

    # 4. 执行本地回测 (默认 open_adj 早盘代理)
    local_data = run_local_backtest_evaluation(
        dates=dates,
        symbols=symbols,
        factor_mat=factor_mat,
        panel_data=panel_data,
        exec_price_mode="open_adj",
    )

    # 5. 输出对比报告
    print_comparison_report(online_data, local_data)


if __name__ == "__main__":
    main()
