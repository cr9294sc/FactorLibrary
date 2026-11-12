#!/usr/bin/env python3
"""
Volume_Price_2_H5_2 因子本地股票回测与 FactorLab 线上对账脚本

步骤:
1. 通过 FactorLab MCP 接口拉取线上回测基准结果 (Run ID: bt_bb3a04da6b20456ab444dcb7e5450949)
2. 加载本地 A 股全市场历史数据 (data/raw/stocks/, 2015-11 ~ 2024-12)
3. 使用 factorlib_core Rust 计算内核执行因子表达式求值与 6 阶段后处理
4. 执行单因子回测 (IC/RankIC, 10 分组收益, 多空组合, 多头超额 vs 000985.CSI, 换手率)
5. 生成线上 vs 本地高精度对账对比报告
"""

from __future__ import annotations

import glob
import math
import os
import sys
import time
from datetime import date, datetime
from typing import Any

import factorlib_core
import numpy as np
import polars as pl

# 引入本地 FactorLab 客户端
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from Factor.factorlab_client import call_mcp_tool


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
) -> tuple[list[str], list[str], dict[str, np.ndarray]]:
    """加载本地全市场日频行情与基础信息，重构为对齐的 [T x N] 面板矩阵。"""
    print("[2/5] 正在加载本地 A 股原始 Parquet 行情并构建面板矩阵...")
    t0 = time.time()

    # 1. 扫描日线行情
    bar_files = sorted(
        glob.glob("data/raw/stocks/bar_1d/bar_1d_201[5-9].parquet")
        + glob.glob("data/raw/stocks/bar_1d/bar_1d_202[0-4].parquet")
    )

    df_bar = (
        pl.scan_parquet(bar_files)
        .filter((pl.col("trading_date") >= start_date) & (pl.col("trading_date") <= end_date))
        .select(["trading_date", "order_book_id", "open", "close", "volume", "total_turnover"])
        .collect()
        .sort(["order_book_id", "trading_date"])
    )

    dates = sorted(df_bar["trading_date"].unique().to_list())
    symbols = sorted(df_bar["order_book_id"].unique().to_list())
    T, N = len(dates), len(symbols)
    print(
        f"      日期网格: {len(dates)} 天 ({dates[0]} 至 {dates[-1]}), 全市场标的: {len(symbols)} 只"
    )

    # 2. 复权因子 join_asof
    df_ex = (
        pl.read_parquet("data/raw/stocks/ex_factor/ex_factor.parquet")
        .select(["order_book_id", "ex_date", "ex_cum_factor"])
        .sort(["order_book_id", "ex_date"])
    )
    df_bar = df_bar.join_asof(
        df_ex,
        left_on="trading_date",
        right_on="ex_date",
        by="order_book_id",
        strategy="backward",
    ).with_columns(pl.col("ex_cum_factor").fill_null(1.0))

    # 3. 停牌状态 join
    ts_files = sorted(
        glob.glob("data/raw/stocks/trade_status/trade_status_201[5-9].parquet")
        + glob.glob("data/raw/stocks/trade_status/trade_status_202[0-4].parquet")
    )
    df_ts = (
        pl.scan_parquet(ts_files)
        .filter((pl.col("trading_date") >= start_date) & (pl.col("trading_date") <= end_date))
        .select(["trading_date", "order_book_id", "is_suspended"])
        .collect()
    )
    df_bar = df_bar.join(df_ts, on=["trading_date", "order_book_id"], how="left")
    df_bar = df_bar.with_columns(pl.col("is_suspended").fill_null(True))

    # 4. 申万一级行业 join_asof
    ind_files = sorted(
        glob.glob("data/raw/stocks/industry_sw/industry_sw_201[5-9].parquet")
        + glob.glob("data/raw/stocks/industry_sw/industry_sw_202[0-4].parquet")
    )
    df_ind = (
        pl.scan_parquet(ind_files)
        .select(["snapshot_date", "order_book_id", "first_industry_name"])
        .collect()
        .sort(["order_book_id", "snapshot_date"])
    )
    df_bar = df_bar.join_asof(
        df_ind,
        left_on="trading_date",
        right_on="snapshot_date",
        by="order_book_id",
        strategy="backward",
    )
    all_industries = sorted(
        [x for x in df_ind["first_industry_name"].unique().to_list() if x is not None]
    )
    ind_map = {name: float(idx + 1) for idx, name in enumerate(all_industries)}
    df_bar = df_bar.with_columns(
        pl.col("first_industry_name")
        .replace_strict(ind_map, default=float("nan"))
        .alias("industry_code")
    )

    # 5. 上市天数计算 (StockListedDays)
    cal_dates = (
        pl.read_parquet("data/raw/stocks/calendar/trading_calendar.parquet")["trading_date"]
        .sort()
        .to_numpy()
    )
    inst = pl.read_parquet("data/raw/stocks/instruments_info/instruments_info.parquet").select(
        ["order_book_id", "listed_date"]
    )
    listed_map = {}
    for row in inst.iter_rows():
        sym, l_date_str = row[0], row[1]
        if l_date_str:
            try:
                l_d = datetime.strptime(l_date_str, "%Y-%m-%d").date()
                idx = np.searchsorted(cal_dates, l_d)
                listed_map[sym] = idx
            except Exception:
                pass

    # 映射坐标索引
    date_map = {d: i for i, d in enumerate(dates)}
    sym_map = {s: i for i, s in enumerate(symbols)}

    df_bar = df_bar.with_columns(
        [
            pl.col("trading_date").replace_strict(date_map).alias("t_idx"),
            pl.col("order_book_id").replace_strict(sym_map).alias("s_idx"),
        ]
    )

    t_arr = df_bar["t_idx"].to_numpy()
    s_arr = df_bar["s_idx"].to_numpy()

    # 初始化 (T, N) 矩阵
    close_mat = np.full((T, N), np.nan, dtype=np.float64)
    open_mat = np.full((T, N), np.nan, dtype=np.float64)
    amount_mat = np.full((T, N), np.nan, dtype=np.float64)
    vol_mat = np.full((T, N), np.nan, dtype=np.float64)
    ex_mat = np.full((T, N), 1.0, dtype=np.float64)
    susp_mat = np.full((T, N), True, dtype=bool)
    ind_mat = np.full((T, N), np.nan, dtype=np.float64)

    close_mat[t_arr, s_arr] = df_bar["close"].to_numpy()
    open_mat[t_arr, s_arr] = df_bar["open"].to_numpy()
    amount_mat[t_arr, s_arr] = df_bar["total_turnover"].to_numpy()
    vol_mat[t_arr, s_arr] = df_bar["volume"].to_numpy()
    ex_mat[t_arr, s_arr] = df_bar["ex_cum_factor"].to_numpy()
    susp_mat[t_arr, s_arr] = df_bar["is_suspended"].to_numpy()
    ind_mat[t_arr, s_arr] = df_bar["industry_code"].to_numpy()

    # 后复权收盘价 close_hfq
    close_hfq = close_mat * ex_mat
    open_adj = open_mat * ex_mat

    # VWAP 成交均价 (成交额 / 成交量 * 复权因子)
    with np.errstate(divide="ignore", invalid="ignore"):
        vwap_adj = (amount_mat / np.where(vol_mat > 0, vol_mat, np.nan)) * ex_mat

    # 上市天数矩阵 StockListedDays
    listed_days_mat = np.full((T, N), np.nan, dtype=np.float64)
    cal_date_indices = np.searchsorted(cal_dates, dates)
    for s_idx, sym in enumerate(symbols):
        l_idx = listed_map.get(sym)
        if l_idx is not None:
            listed_days_mat[:, s_idx] = (cal_date_indices - l_idx).astype(np.float64)

    # 收益率矩阵 c2c_ret: close_hfq[t] / close_hfq[t-1] - 1
    c2c_ret = np.full((T, N), np.nan, dtype=np.float64)
    c2c_ret[1:] = (close_hfq[1:] / close_hfq[:-1]) - 1.0

    # log(amount)
    with np.errstate(divide="ignore", invalid="ignore"):
        log_amount = np.log(np.where(amount_mat > 0, amount_mat, np.nan))

    # 基准中证全指 000985.XSHG
    df_bench = (
        pl.read_parquet("data/raw/stocks/benchmark/index_bar_1d.parquet")
        .filter(
            (pl.col("order_book_id") == "000985.XSHG")
            & (pl.col("trading_date") >= start_date)
            & (pl.col("trading_date") <= end_date)
        )
        .sort("trading_date")
    )
    bench_dict = {row[1]: row[5] for row in df_bench.iter_rows()}  # close
    bench_close = np.array([bench_dict.get(d, np.nan) for d in dates], dtype=np.float64)
    bench_ret = np.full(T, np.nan, dtype=np.float64)
    bench_ret[1:] = (bench_close[1:] / bench_close[:-1]) - 1.0

    print(f"      数据构建完成，耗时: {time.time() - t0:.2f} 秒")

    matrices = {
        "close_hfq": close_hfq,
        "open_adj": open_adj,
        "vwap_adj": vwap_adj,
        "amount": amount_mat,
        "log_amount": log_amount,
        "c2c_ret": c2c_ret,
        "is_suspended": susp_mat,
        "listed_days": listed_days_mat,
        "industry_code": ind_mat,
        "bench_ret": bench_ret,
    }
    date_strs = [d.strftime("%Y%m%d") for d in dates]
    return date_strs, symbols, matrices


def compute_factor_and_postprocess(
    dates: list[str],
    symbols: list[str],
    matrices: dict[str, np.ndarray],
) -> np.ndarray:
    """使用 factorlib_core Rust 计算内核执行表达式计算与 6 阶段后处理。"""
    print("[3/5] 正在通过 factorlib_core Rust 计算内核执行因子求值与后处理流水线...")
    t0 = time.time()

    # 1. 基础股票池有效掩码 (有收盘价且有成交量)
    base_mask = np.isfinite(matrices["close_hfq"]) & np.isfinite(matrices["log_amount"])

    # 2. Rust Engine 求值
    fields = {
        "close_hfq": matrices["close_hfq"],
        "log_amount": matrices["log_amount"],
        "c2c_ret": matrices["c2c_ret"],
    }
    engine = factorlib_core.Engine(fields, base_mask)

    # 表达式: neg(add(cs_zscore(ts_corr(close_hfq, log_amount, 20)), cs_zscore(ts_std(c2c_ret, 20))))
    expr = "neg(add(cs_zscore(ts_corr(close_hfq, log_amount, 20)), cs_zscore(ts_std(c2c_ret, 20))))"
    raw_factor = np.asarray(engine.eval(expr))
    print(f"      - [阶段 3] 因子表达式 AST 求值完毕 (有效点数: {np.sum(np.isfinite(raw_factor))})")

    # 3. 阶段四: 后处理流水线 (对齐 FactorLab postprocess)
    # 步骤 1: trdMask (过滤停牌股与非交易股)
    # trdMask 矩阵: 未停牌且有成交量为 1.0, 否则 NaN
    trd_mask_arr = np.where(~matrices["is_suspended"] & (matrices["amount"] > 0), 1.0, np.nan)
    f_step1 = factorlib_core.apply_postprocess_op("trdMask", raw_factor, x2=trd_mask_arr)

    # 步骤 2: dropNewStock(120) (剔除上市未满 120 交易日的次新股)
    f_step2 = factorlib_core.apply_postprocess_op(
        "dropNewStock", f_step1, x2=matrices["listed_days"], param=120.0
    )

    # 步骤 3: MAD(3) (3 倍中位数绝对偏差去极值)
    f_step3 = factorlib_core.apply_postprocess_op("MAD", f_step2, param=3.0)

    # 步骤 4: indNeut (申万一级行业截面残差中性化)
    f_step4 = factorlib_core.apply_postprocess_op("indNeut", f_step3, x2=matrices["industry_code"])

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
    matrices: dict[str, np.ndarray],
    start_eval_date: str = "20160104",
    end_eval_date: str = "20241231",
) -> dict[str, Any]:
    """执行 2016-2024 全周期与分年度单因子回测评估。"""
    print(f"[4/5] 正在执行全周期 ({start_eval_date} 至 {end_eval_date}) 与分年度单因子回测...")
    t0 = time.time()
    T, N = len(dates), len(symbols)

    # 1. 远期收益率计算: 按照 FactorLab 规则
    # t 日 15:00 信号 -> t+1 日 VwapBegin30 进场 -> t+1+h 日 VwapBegin30 出场
    # 本地使用 vwap_adj (全天均价) 或 open_adj 作为代理价格 P_adj
    price_proxy = matrices["vwap_adj"]  # 或 open_adj

    fwd_ret_1d = np.full((T, N), np.nan, dtype=np.float64)
    fwd_ret_2d = np.full((T, N), np.nan, dtype=np.float64)
    fwd_ret_5d = np.full((T, N), np.nan, dtype=np.float64)
    fwd_ret_10d = np.full((T, N), np.nan, dtype=np.float64)

    for t in range(T - 11):
        p_in = price_proxy[t + 1]
        valid_in = np.isfinite(p_in) & (p_in > 0) & (~matrices["is_suspended"][t + 1])
        for h, fwd in [(1, fwd_ret_1d), (2, fwd_ret_2d), (5, fwd_ret_5d), (10, fwd_ret_10d)]:
            p_out = price_proxy[t + 1 + h]
            valid_out = np.isfinite(p_out) & (p_out > 0)
            valid_both = valid_in & valid_out
            ret = np.full(N, np.nan, dtype=np.float64)
            ret[valid_both] = (p_out[valid_both] / p_in[valid_both]) - 1.0
            fwd[t] = ret

    # 2. 切片到评估区间 [20160104, 20241231]
    eval_indices = [i for i, d in enumerate(dates) if start_eval_date <= d <= end_eval_date]
    sub_dates = [dates[i] for i in eval_indices]
    sub_factor = factor_mat[eval_indices]
    sub_ret_1d = fwd_ret_1d[eval_indices]
    sub_ret_2d = fwd_ret_2d[eval_indices]
    sub_ret_5d = fwd_ret_5d[eval_indices]
    sub_ret_10d = fwd_ret_10d[eval_indices]
    sub_bench_ret = matrices["bench_ret"][eval_indices]

    # 有效回测掩码 (因子有效且收益率有效且非停牌)
    eval_mask = np.isfinite(sub_factor) & np.isfinite(sub_ret_1d)

    # 3. Rust 指标计算
    # 全周期 IC & RankIC (1日)
    ic_res = factorlib_core.calc_ic(sub_factor, sub_ret_1d, eval_mask, rank=False, horizon=1)
    rankic_res = factorlib_core.calc_ic(sub_factor, sub_ret_1d, eval_mask, rank=True, horizon=1)

    # 5日 IC & RankIC
    eval_mask_5d = np.isfinite(sub_factor) & np.isfinite(sub_ret_5d)
    ic_5d_res = factorlib_core.calc_ic(sub_factor, sub_ret_5d, eval_mask_5d, rank=False, horizon=5)
    rankic_5d_res = factorlib_core.calc_ic(
        sub_factor, sub_ret_5d, eval_mask_5d, rank=True, horizon=5
    )

    # 2日 & 10日 RankIC
    eval_mask_2d = np.isfinite(sub_factor) & np.isfinite(sub_ret_2d)
    rankic_2d_res = factorlib_core.calc_ic(
        sub_factor, sub_ret_2d, eval_mask_2d, rank=True, horizon=2
    )
    eval_mask_10d = np.isfinite(sub_factor) & np.isfinite(sub_ret_10d)
    rankic_10d_res = factorlib_core.calc_ic(
        sub_factor, sub_ret_10d, eval_mask_10d, rank=True, horizon=10
    )

    # 4. 10 分位数收益 (Q1..Q10) 与多空日收益
    q_res = factorlib_core.calc_quantile_returns(
        sub_factor, sub_ret_1d, eval_mask, n_quantiles=10, min_names=10, sign=1
    )
    turnover_res = factorlib_core.calc_turnover(sub_factor, eval_mask, min_names=10)

    # 多头超额 (G10 vs 000985.CSI)
    # 取 G10 每日收益
    g10_daily = np.asarray(q_res["group_daily_returns"][9])
    # 相对中证全指超额
    valid_days = len(g10_daily)
    excess_daily = g10_daily - sub_bench_ret[:valid_days]
    excess_mean = float(np.nanmean(excess_daily))
    excess_std = float(np.nanstd(excess_daily, ddof=1))
    excess_annual_ret = excess_mean * 250.0
    excess_annual_vol = excess_std * math.sqrt(250.0) if excess_std > 0 else 0.0
    excess_sharpe = excess_annual_ret / excess_annual_vol if excess_annual_vol > 0 else 0.0

    # 累计超额与最大回撤
    cum_excess = np.nancumsum(excess_daily)
    peak = np.maximum.accumulate(cum_excess)
    drawdowns = peak - cum_excess
    max_dd = float(np.nanmax(drawdowns)) if len(drawdowns) > 0 else 0.0

    # 5. 分年度统计 (2016 .. 2024)
    annual_stats = []
    years = sorted({d[:4] for d in sub_dates})
    for y in years:
        y_indices = [i for i, d in enumerate(sub_dates) if d.startswith(y)]
        if not y_indices:
            continue
        y_factor = sub_factor[y_indices]
        y_ret_1d = sub_ret_1d[y_indices]
        y_ret_5d = sub_ret_5d[y_indices]
        y_mask_1d = eval_mask[y_indices]
        y_mask_5d = eval_mask_5d[y_indices]
        y_bench = sub_bench_ret[y_indices]

        y_ic = factorlib_core.calc_ic(y_factor, y_ret_1d, y_mask_1d, rank=False, horizon=1)
        y_rankic = factorlib_core.calc_ic(y_factor, y_ret_1d, y_mask_1d, rank=True, horizon=1)
        y_ic_5d = factorlib_core.calc_ic(y_factor, y_ret_5d, y_mask_5d, rank=False, horizon=5)
        y_q = factorlib_core.calc_quantile_returns(
            y_factor, y_ret_1d, y_mask_1d, n_quantiles=10, min_names=10, sign=1
        )

        # 多头超额 (G10 vs Bench)
        y_g10 = np.asarray(y_q["group_daily_returns"][9])
        y_v_days = len(y_g10)
        y_excess = y_g10 - y_bench[:y_v_days]
        y_ex_mean = float(np.nanmean(y_excess))
        y_ex_std = float(np.nanstd(y_excess, ddof=1))
        y_ann_ret = y_ex_mean * 250.0
        y_ann_vol = y_ex_std * math.sqrt(250.0) if y_ex_std > 0 else 0.0
        y_sharpe = y_ann_ret / y_ann_vol if y_ann_vol > 0 else 0.0

        y_cum_ex = np.nancumsum(y_excess)
        y_peak = np.maximum.accumulate(y_cum_ex)
        y_dd = float(np.nanmax(y_peak - y_cum_ex)) if len(y_cum_ex) > 0 else 0.0

        annual_stats.append(
            {
                "year": y,
                "days": len(y_indices),
                "ic": y_ic["ic_mean"],
                "ic_ir": y_ic["ic_ir"],
                "rankic": y_rankic["ic_mean"],
                "rankic_ir": y_rankic["ic_ir"],
                "ic_5d": y_ic_5d["ic_mean"],
                "ic_5d_ir": y_ic_5d["ic_ir"],
                "ls_daily": y_q["ls_mean"],
                "excess_ann_ret": y_ann_ret,
                "excess_sharpe": y_sharpe,
                "excess_max_dd": y_dd,
            }
        )

    local_results = {
        "dates_count": len(sub_dates),
        "valid_days": ic_res["valid_days"],
        "ic": ic_res,
        "rankic": rankic_res,
        "ic_5d": ic_5d_res,
        "rankic_5d": rankic_5d_res,
        "rankic_2d": rankic_2d_res,
        "rankic_10d": rankic_10d_res,
        "quantile": q_res,
        "turnover": turnover_res,
        "long_excess": {
            "annual_return": excess_annual_ret,
            "annual_vol": excess_annual_vol,
            "sharpe": excess_sharpe,
            "max_dd": max_dd,
        },
        "annual_breakdown": annual_stats,
    }
    print(f"      回测完成，耗时: {time.time() - t0:.2f}s")
    return local_results


def print_comparison_tables(online: dict[str, Any], local: dict[str, Any]):
    """打印线上与本地回测对比表格。"""
    print("\n" + "=" * 90)
    print(" 🚀 Volume_Price_2_H5_2 因子回测结果对账总览 (FactorLab 线上 vs 本地 Rust 回测框架)")
    print("=" * 90)

    # 1. 全周期核心指标对比
    print("\n【全周期核心统计指标对比 (2016-01-04 ~ 2024-12-31)】")
    header = f"{'指标名称':<18} | {'FactorLab 线上结果':<22} | {'本地 Rust 回测结果':<22} | {'绝对差异 / 对账结论':<20}"
    print("-" * 90)
    print(header)
    print("-" * 90)

    m_on = online.get("metrics", {})
    t_on = online.get("tables", [])

    items = [
        ("IC 均值", f"{m_on.get('ic', 0):.4f}", f"{local['ic']['ic_mean']:.4f}"),
        ("IC IR", f"{m_on.get('icIR', 0):.4f}", f"{local['ic']['ic_ir']:.4f}"),
        (
            "IC 胜率",
            f"{m_on.get('icWR', 0) * 100:.2f}%",
            f"{local['ic']['ic_win_rate'] * 100:.2f}%",
        ),
        ("RankIC 均值", f"{m_on.get('rankic', 0):.4f}", f"{local['rankic']['ic_mean']:.4f}"),
        ("RankIC IR", f"{m_on.get('rankicIR', 0):.4f}", f"{local['rankic']['ic_ir']:.4f}"),
        (
            "RankIC 胜率",
            f"{m_on.get('rankicWR', 0) * 100:.2f}%",
            f"{local['rankic']['ic_win_rate'] * 100:.2f}%",
        ),
        ("5日 IC 均值", f"{m_on.get('5dic', 0):.4f}", f"{local['ic_5d']['ic_mean']:.4f}"),
        ("5日 IC IR", f"{m_on.get('5dicIR', 0):.4f}", f"{local['ic_5d']['ic_ir']:.4f}"),
        (
            "多空日均收益",
            f"{m_on.get('long_short', 0) * 100:.2f}%",
            f"{local['quantile']['ls_mean'] * 100:.2f}%",
        ),
        (
            "多空夏普比率",
            f"{m_on.get('long_shortIR', 0):.4f}",
            f"{local['quantile']['ls_sharpe']:.4f}",
        ),
        ("单边换手率", "12.11%", f"{local['turnover']['turnover'] * 100:.2f}%"),
        ("多头年化超额 (vs全指)", "1.26%", f"{local['long_excess']['annual_return'] * 100:.2f}%"),
        ("多头超额夏普", "0.1642", f"{local['long_excess']['sharpe']:.4f}"),
        ("有效回测天数", "2185 天", f"{local['valid_days']} 天"),
    ]

    for name, v_on, v_loc in items:
        # 提取浮点数值计算差异
        try:
            val_on = float(v_on.replace("%", "").replace(" 天", ""))
            val_loc = float(v_loc.replace("%", "").replace(" 天", ""))
            diff = val_loc - val_on
            diff_str = f"{diff:+.4f}" if "%" not in v_on else f"{diff:+.2f}%"
        except Exception:
            diff_str = "--"
        print(f"{name:<18} | {v_on:<22} | {v_loc:<22} | {diff_str:<20}")

    # 2. 10 分组收益对比 (G1 .. G10)
    print("\n【10 分组收益单调性对比 (Quantile Groups G1 ~ G10)】")
    print("-" * 75)
    print(
        f"{'分位组':<8} | {'FactorLab 日均收益':<20} | {'本地 Rust 日均收益':<20} | {'收益单调性走势':<15}"
    )
    print("-" * 75)
    t_grp_on = next((t for t in t_on if t.get("id") == "group_metrics"), {})
    on_group_rows = t_grp_on.get("rows", [])
    local_grp_rets = local["quantile"]["group_returns"]

    for idx in range(10):
        g_name = f"G{idx + 1}"
        on_ret_str = on_group_rows[idx][1] if idx < len(on_group_rows) else "--"
        loc_ret = local_grp_rets[idx] * 100.0
        bar = "█" * int(max(0, loc_ret * 200 + 10))
        print(f"{g_name:<8} | {on_ret_str:<20} | {loc_ret:+.3f}%{'':<13} | {bar}")

    # 3. 多周期 RankIC 衰减对比
    print("\n【多周期预测能力衰减对比 (RankIC Across Horizons)】")
    print("-" * 75)
    print(
        f"{'预测周期':<10} | {'FactorLab RankIC':<20} | {'本地 Rust RankIC':<20} | {'本地 RankIC IR':<15}"
    )
    print("-" * 75)
    horizons_data = [
        (
            "1 日 (h=1)",
            "0.0431",
            f"{local['rankic']['ic_mean']:.4f}",
            f"{local['rankic']['ic_ir']:.4f}",
        ),
        (
            "2 日 (h=2)",
            "0.0535",
            f"{local['rankic_2d']['ic_mean']:.4f}",
            f"{local['rankic_2d']['ic_ir']:.4f}",
        ),
        (
            "5 日 (h=5)",
            "0.0659",
            f"{local['rankic_5d']['ic_mean']:.4f}",
            f"{local['rankic_5d']['ic_ir']:.4f}",
        ),
        (
            "10 日 (h=10)",
            "0.0764",
            f"{local['rankic_10d']['ic_mean']:.4f}",
            f"{local['rankic_10d']['ic_ir']:.4f}",
        ),
    ]
    for h_name, h_on, h_loc, h_ir in horizons_data:
        print(f"{h_name:<10} | {h_on:<20} | {h_loc:<20} | {h_ir:<15}")

    # 4. 分年度对比 (2016 .. 2024)
    print("\n【分年度 RankIC 与超额表现对账 (2016 ~ 2024)】")
    print("-" * 90)
    print(
        f"{'年份':<6} | {'线上 RankIC':<14} | {'本地 RankIC':<14} | {'线上 RankIC IR':<16} | {'本地 RankIC IR':<16} | {'本地多头超额':<12}"
    )
    print("-" * 90)
    t_annual_on = next((t for t in t_on if t.get("id") == "yearly_metrics"), {})
    on_ann_dict = {r[0]: r for r in t_annual_on.get("rows", [])}

    for stat in local["annual_breakdown"]:
        y = stat["year"]
        on_row = on_ann_dict.get(y, [])
        on_rankic = on_row[5] if len(on_row) > 5 else "--"
        on_rankic_ir = on_row[6] if len(on_row) > 6 else "--"
        print(
            f"{y:<6} | {on_rankic:<14} | {stat['rankic']:.4f}{'':<8} | {on_rankic_ir:<16} | {stat['rankic_ir']:.4f}{'':<10} | {stat['excess_ann_ret'] * 100:+.2f}%"
        )

    print("=" * 90)


def main():
    print("=" * 90)
    print(" 🎯 FactorLab 线上因子 Volume_Price_2_H5_2 vs 本地 Rust 计算引擎回测对账流程启动")
    print("=" * 90)

    # 1. 抓取线上数据
    online_data = fetch_online_factorlab_results()

    # 2. 构建本地数据
    dates, symbols, matrices = load_local_stock_data(
        start_date=date(2015, 11, 1),
        end_date=date(2024, 12, 31),
    )

    # 3. 因子计算与后处理
    factor_mat = compute_factor_and_postprocess(dates, symbols, matrices)

    # 4. 本地回测评估
    local_data = run_local_backtest_evaluation(
        dates=dates,
        symbols=symbols,
        factor_mat=factor_mat,
        matrices=matrices,
        start_eval_date="20160104",
        end_eval_date="20241231",
    )

    # 5. 打印对比报告
    print_comparison_tables(online_data, local_data)


if __name__ == "__main__":
    main()
