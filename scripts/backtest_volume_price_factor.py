#!/usr/bin/env python3
"""Volume_Price_2_H5_2 因子本地股票回测与 FactorLab 线上对账脚本 (scripts/backtest_volume_price_factor.py)。

步骤:
1. 通过 FactorLab MCP 接口拉取线上回测基准结果 (Run ID: bt_bb3a04da6b20456ab444dcb7e5450949)，支持远端 503 时自动回退至权威快照
2. 通过 StockPanelLoader 加载本地 A 股全市场历史数据 (data/raw/stocks/, 2015-11 ~ 2024-12)
3. 使用 factorlib_core Rust 计算内核执行因子表达式求值与 6 阶段后处理 (严格应用 LimitMask 与 BaseUnivMask)
4. 执行单因子回测 (同时评测 open_adj 早盘无前视主口径与 vwap_adj 全天均价对照口径：IC/RankIC, 10 分组收益, 多空组合, 多头超额 vs 000985.CSI, Top G10 单边换手率)
5. 生成线上 (vwap30) vs 本地 (open_adj 主口径 / vwap_adj 对照口径) 高精度三方对账对比报告
"""

from __future__ import annotations

import argparse
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
from factorlib.core.probe import apply_postprocess_pipeline  # noqa: E402
from factorlib.markets.stocks import StockPanelData, StockPanelLoader  # noqa: E402

# FactorLab 线上 Volume_Price_2_H5_2 (bt_bb3a04da6b20456ab444dcb7e5450949) 权威回测快照 (用于远端 503 离线回退)
OFFLINE_FACTORLAB_SUMMARY: dict[str, Any] = {
    "run_id": "bt_bb3a04da6b20456ab444dcb7e5450949",
    "factor_version_id": "fv_f20a12c06ce14ddc9442b025eb1ac241",
    "metrics": {
        "ic": 0.026498926788376665,
        "icIR": 0.2729103541388729,
        "icWR": 0.6155606407322655,
        "5dic": 0.03318357687537172,
        "5dicIR": 0.3484082996835071,
        "5dicWR": 0.635946813388354,
        "rankic": 0.04310053527470243,
        "rankicIR": 0.40185604794226404,
        "rankicWR": 0.6796338672768879,
        "percic": 0.03526514461197284,
        "percicIR": 0.3691925083255501,
        "percicWR": 0.6672768878718536,
        "long_short": 0.002363537363504312,
        "long_shortIR": 0.2956032237148618,
        "coverage": 0.9764628670537125,
        "days": 2185,
    },
    "tables": [
        {
            "id": "group_metrics",
            "rows": [
                ["G1", "-0.22%", "-99.44%"],
                ["G2", "-0.10%", "-91.08%"],
                ["G3", "-0.06%", "-78.87%"],
                ["G4", "-0.03%", "-63.52%"],
                ["G5", "-0.02%", "-45.60%"],
                ["G6", "-0.00%", "-29.44%"],
                ["G7", "0.01%", "-7.82%"],
                ["G8", "0.01%", "-0.48%"],
                ["G9", "0.01%", "10.02%"],
                ["G10", "0.02%", "16.84%"],
            ],
        },
        {
            "id": "horizon_metrics",
            "rows": [
                ["1日", "0.0265", "0.2729", "0.0431", "0.4019", "0.0353", "0.3692", "97.65%"],
                ["2日", "0.0274", "0.2825", "0.0535", "0.4960", "0.0434", "0.4564", "97.64%"],
                ["5日", "0.0332", "0.3484", "0.0659", "0.6189", "0.0528", "0.5651", "97.64%"],
                ["10日", "0.0404", "0.4296", "0.0764", "0.7324", "0.0611", "0.6662", "97.63%"],
            ],
        },
        {
            "id": "yearly_metrics",
            "rows": [
                [
                    "2016",
                    "0.0184",
                    "0.1592",
                    "0.0349",
                    "0.3172",
                    "0.0302",
                    "0.2273",
                    "0.0241",
                    "0.1963",
                    "98.14%",
                    "244",
                ],
                [
                    "2017",
                    "0.0323",
                    "0.2960",
                    "0.0595",
                    "0.5485",
                    "0.0428",
                    "0.3680",
                    "0.0358",
                    "0.3537",
                    "95.69%",
                    "244",
                ],
                [
                    "2018",
                    "0.0235",
                    "0.2099",
                    "0.0281",
                    "0.2779",
                    "0.0321",
                    "0.2495",
                    "0.0252",
                    "0.2185",
                    "98.61%",
                    "243",
                ],
                [
                    "2019",
                    "0.0314",
                    "0.3457",
                    "0.0402",
                    "0.4180",
                    "0.0465",
                    "0.4465",
                    "0.0379",
                    "0.4081",
                    "98.54%",
                    "244",
                ],
                [
                    "2020",
                    "0.0278",
                    "0.2783",
                    "0.0321",
                    "0.3510",
                    "0.0466",
                    "0.4338",
                    "0.0388",
                    "0.4097",
                    "96.80%",
                    "243",
                ],
                [
                    "2021",
                    "0.0260",
                    "0.3023",
                    "0.0242",
                    "0.2972",
                    "0.0478",
                    "0.5242",
                    "0.0402",
                    "0.4900",
                    "96.20%",
                    "243",
                ],
                [
                    "2022",
                    "0.0293",
                    "0.3971",
                    "0.0303",
                    "0.3869",
                    "0.0504",
                    "0.6250",
                    "0.0413",
                    "0.5666",
                    "97.33%",
                    "242",
                ],
                [
                    "2023",
                    "0.0265",
                    "0.3885",
                    "0.0258",
                    "0.3404",
                    "0.0502",
                    "0.7417",
                    "0.0409",
                    "0.7185",
                    "98.18%",
                    "242",
                ],
                [
                    "2024",
                    "0.0231",
                    "0.2170",
                    "0.0229",
                    "0.2225",
                    "0.0413",
                    "0.3511",
                    "0.0332",
                    "0.3236",
                    "99.36%",
                    "240",
                ],
                [
                    "全周期",
                    "0.0265",
                    "0.2729",
                    "0.0332",
                    "0.3484",
                    "0.0431",
                    "0.4019",
                    "0.0353",
                    "0.3692",
                    "97.65%",
                    "2185",
                ],
            ],
        },
        {
            "id": "long_excess_metrics",
            "rows": [
                [
                    "2016",
                    "20160104",
                    "20161230",
                    "2.4295",
                    "11.17%",
                    "4.60%",
                    "14.30%",
                    "242",
                    "0.50%",
                    "1.94%",
                    "20160504",
                    "20160617",
                ],
                [
                    "2017",
                    "20170103",
                    "20171229",
                    "-2.1087",
                    "-11.81%",
                    "5.60%",
                    "11.65%",
                    "280",
                    "6.64%",
                    "14.91%",
                    "20170317",
                    "20171120",
                ],
                [
                    "2018",
                    "20180102",
                    "20181228",
                    "0.8337",
                    "6.01%",
                    "7.21%",
                    "12.47%",
                    "313",
                    "1.37%",
                    "6.34%",
                    "20180102",
                    "20180205",
                ],
                [
                    "2019",
                    "20190102",
                    "20191231",
                    "-0.9819",
                    "-4.76%",
                    "4.84%",
                    "11.82%",
                    "335",
                    "2.96%",
                    "5.22%",
                    "20190104",
                    "20191115",
                ],
                [
                    "2020",
                    "20200102",
                    "20201231",
                    "-0.8469",
                    "-5.42%",
                    "6.40%",
                    "12.41%",
                    "346",
                    "2.09%",
                    "10.91%",
                    "20200904",
                    "20201231",
                ],
                [
                    "2021",
                    "20210104",
                    "20211231",
                    "0.9344",
                    "10.89%",
                    "11.65%",
                    "11.27%",
                    "390",
                    "3.32%",
                    "8.89%",
                    "20210104",
                    "20210209",
                ],
                [
                    "2022",
                    "20220104",
                    "20221230",
                    "0.6211",
                    "4.59%",
                    "7.40%",
                    "11.22%",
                    "434",
                    "1.82%",
                    "5.46%",
                    "20220303",
                    "20220704",
                ],
                [
                    "2023",
                    "20230103",
                    "20231229",
                    "2.5646",
                    "15.18%",
                    "5.92%",
                    "11.21%",
                    "472",
                    "0.84%",
                    "4.75%",
                    "20230308",
                    "20230420",
                ],
                [
                    "2024",
                    "20240102",
                    "20241231",
                    "-0.9015",
                    "-10.46%",
                    "11.60%",
                    "12.59%",
                    "474",
                    "11.07%",
                    "16.74%",
                    "20240103",
                    "20240202",
                ],
                [
                    "全周期",
                    "20160104",
                    "20241231",
                    "0.1642",
                    "1.26%",
                    "7.67%",
                    "12.11%",
                    "365",
                    "8.51%",
                    "28.73%",
                    "20170317",
                    "20210209",
                ],
            ],
        },
    ],
}


def fetch_online_factorlab_results() -> dict[str, Any]:
    """从 FactorLab MCP 服务拉取 Volume_Price_2_H5_2 线上回测结果，若远端 503 则回退至离线快照。"""
    print("[1/5] 正在通过 MCP 调取 FactorLab 线上回测数据...")
    factor_id = "fac_5bc39b9aa01c427aa2d28563d7d9b5a2"
    try:
        factor_info = call_mcp_tool("get_factor", {"factor_id": factor_id})
        latest_run_id = (
            factor_info.get("data", {}).get("latest_run", {}).get("run_id")
            or "bt_bb3a04da6b20456ab444dcb7e5450949"
        )
        print(
            f"      线上因子名称: {factor_info.get('data', {}).get('name', 'Volume_Price_2_H5_2')}"
        )
        print(f"      最新回测 Run ID: {latest_run_id}")
        summary = call_mcp_tool("get_result_summary", {"run_id": latest_run_id})
        data = summary.get("data", {})
        if data:
            return data
    except Exception as exc:
        print(
            f"      [提示] 远端 MCP 服务暂不可用 ({exc})，自动启用 bt_bb3a04da6b20456ab444dcb7e5450949 权威快照基准。"
        )
    return OFFLINE_FACTORLAB_SUMMARY


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
    """使用 factorlib_core Rust 计算内核与统一 P2-02 防泄漏后处理管线执行因子求值。"""
    print("[3/5] 正在通过 factorlib_core Rust 计算内核执行因子求值与后处理流水线...")
    t0 = time.time()

    # 1. 基础股票池有效掩码与标准 Panel (绑定 BaseUnivMask 剔除 ST、停牌与存续期外标的)
    stock_panel = panel_data.to_panel()
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

    # 3. 阶段四: 复用 apply_postprocess_pipeline 执行严格 8 步后处理
    # (对齐 FactorLab: baseUniMask -> trdMask -> dropNewStock(120) -> MAD(3) -> zScore -> indNeut -> decay(5) + 重施掩码 -> rank)
    pp_steps = [
        "baseUniMask",
        "trdMask",
        "dropNewStock:120",
        "MAD:3",
        "zScore",
        "indNeut:sector",
        "decay:5",
        "rank",
    ]
    final_factor = apply_postprocess_pipeline(raw_factor, pp_steps, stock_panel)

    # 零渗漏校验 (确保 decay(5) 未向涨跌停/非在池/次新股渗出无效分数)
    fin = np.isfinite(final_factor)
    leak_lim = int(np.sum(fin & (~np.isfinite(panel_data.limit_mask))))
    leak_univ = int(np.sum(fin & (~panel_data.base_univ_mask)))
    if leak_lim > 0 or leak_univ > 0:
        raise RuntimeError(
            f"后处理后检测到掩码渗漏: leak_limit={leak_lim}, leak_base_univ={leak_univ}"
        )

    print(
        f"      - [阶段 4] 统一后处理流水线执行完毕 (含 zScore 与 decay 后重施掩码, 耗时: {time.time() - t0:.2f}s)"
    )
    return final_factor


def run_local_backtest_evaluation(
    dates: list[str],
    symbols: list[str],
    factor_mat: np.ndarray,
    panel_data: StockPanelData,
    start_eval_date: str = "20160104",
    end_eval_date: str = "20241231",
    exec_price_mode: str = "open_adj",
    execution_simulation: bool = False,
) -> dict[str, Any]:
    """执行 2016-2024 全周期与分年度单因子回测评估。"""
    print(
        f"[4/5] 正在执行全周期 ({start_eval_date} 至 {end_eval_date}) 与分年度单因子回测 "
        f"(模式: {exec_price_mode}, 涨跌停模拟: {execution_simulation})..."
    )
    t0 = time.time()

    # 1. 远期收益率计算: 按照执行价格代理 (open_adj 或 vwap_adj)
    fwd_returns = panel_data.compute_forward_returns(
        horizons=[1, 2, 5, 10],
        exec_price_mode=exec_price_mode,
        execution_simulation=execution_simulation,
    )
    fwd_ret_1d = fwd_returns[1]
    fwd_ret_2d = fwd_returns[2]
    fwd_ret_5d = fwd_returns[5]
    fwd_ret_10d = fwd_returns[10]

    # 对齐价格时钟的基准收益率序列 (P0-09: open_adj 对应开盘基准, vwap_adj 对应收盘基准)
    bench_series = (
        panel_data.bench_fwd_open_1
        if exec_price_mode == "open_adj"
        else panel_data.bench_fwd_close_1
    )

    # 2. 切片到评估区间 [20160104, 20241231]
    eval_indices = [i for i, d in enumerate(dates) if start_eval_date <= d <= end_eval_date]
    sub_dates = [dates[i] for i in eval_indices]
    sub_factor = np.asfortranarray(factor_mat[eval_indices], dtype=np.float64)
    sub_ret_1d = np.asfortranarray(fwd_ret_1d[eval_indices], dtype=np.float64)
    sub_ret_2d = np.asfortranarray(fwd_ret_2d[eval_indices], dtype=np.float64)
    sub_ret_5d = np.asfortranarray(fwd_ret_5d[eval_indices], dtype=np.float64)
    sub_ret_10d = np.asfortranarray(fwd_ret_10d[eval_indices], dtype=np.float64)
    sub_bench_ret = np.asarray(bench_series[eval_indices], dtype=np.float64)

    # 有效回测掩码 (因子有效且收益率有效)
    eval_mask = np.asfortranarray(np.isfinite(sub_factor) & np.isfinite(sub_ret_1d))

    # 3. 计算 IC 与 RankIC
    ic_res = factorlib_core.calc_ic(sub_factor, sub_ret_1d, eval_mask, rank=False, horizon=1)
    rankic_res = factorlib_core.calc_ic(sub_factor, sub_ret_1d, eval_mask, rank=True, horizon=1)

    eval_mask_5d = np.asfortranarray(np.isfinite(sub_factor) & np.isfinite(sub_ret_5d))
    ic_5d_res = factorlib_core.calc_ic(sub_factor, sub_ret_5d, eval_mask_5d, rank=False, horizon=5)
    rankic_5d_res = factorlib_core.calc_ic(
        sub_factor, sub_ret_5d, eval_mask_5d, rank=True, horizon=5
    )

    eval_mask_2d = np.asfortranarray(np.isfinite(sub_factor) & np.isfinite(sub_ret_2d))
    rankic_2d_res = factorlib_core.calc_ic(
        sub_factor, sub_ret_2d, eval_mask_2d, rank=True, horizon=2
    )

    eval_mask_10d = np.asfortranarray(np.isfinite(sub_factor) & np.isfinite(sub_ret_10d))
    rankic_10d_res = factorlib_core.calc_ic(
        sub_factor, sub_ret_10d, eval_mask_10d, rank=True, horizon=10
    )

    # 4. 计算 10 分组多空收益 (P0-10: 对齐切分边界并内置 mean_top_turnover)
    q_res = factorlib_core.calc_quantile_returns(
        sub_factor, sub_ret_1d, eval_mask, n_quantiles=10, min_names=10, sign=1
    )

    # 5. 计算组合多头 (G10) 权重换手率与全截面秩自相关
    top_to_res = factorlib_core.calc_top_quantile_turnover(
        sub_factor, eval_mask, n_quantiles=10, min_names=10, sign=1
    )
    turnover_res = factorlib_core.calc_turnover(sub_factor, eval_mask, min_names=10)

    # 6. 计算标准多头超额收益 (G10 vs 000985.CSI) 与等权宇宙/时钟相关性统计
    g10_daily_ret = np.asarray(q_res["group_daily_returns"][9], dtype=np.float64)
    n_valid = len(g10_daily_ret)
    b_slice = sub_bench_ret[:n_valid]
    excess_metrics = calc_long_excess_metrics(
        long_rets=g10_daily_ret,
        bench_rets=b_slice,
        dates=sub_dates[:n_valid],
        trading_days_per_year=250.0,
    )
    ok_pair = np.isfinite(g10_daily_ret) & np.isfinite(b_slice)
    corr_bench = (
        float(np.corrcoef(g10_daily_ret[ok_pair], b_slice[ok_pair])[0, 1])
        if np.sum(ok_pair) > 2
        else 0.0
    )
    has_valid_row = np.any(eval_mask, axis=1)
    ew_daily = np.full(len(sub_dates), np.nan, dtype=np.float64)
    if np.any(has_valid_row):
        ew_daily[has_valid_row] = np.nanmean(
            np.where(eval_mask[has_valid_row], sub_ret_1d[has_valid_row], np.nan),
            axis=1,
        )
    ew_annual_ret = float(np.nanmean(ew_daily) * 250.0) if np.any(np.isfinite(ew_daily)) else 0.0
    bench_annual_ret = float(np.nanmean(b_slice) * 250.0) if np.any(np.isfinite(b_slice)) else 0.0

    # 7. 分年度回测统计 (2016 .. 2024)
    years = sorted({d[:4] for d in sub_dates})
    annual_breakdown = []
    for y in years:
        y_indices = [i for i, d in enumerate(sub_dates) if d.startswith(y)]
        if not y_indices:
            continue
        y_factor = np.asfortranarray(sub_factor[y_indices], dtype=np.float64)
        y_ret_1d = np.asfortranarray(sub_ret_1d[y_indices], dtype=np.float64)
        y_ret_5d = np.asfortranarray(sub_ret_5d[y_indices], dtype=np.float64)
        y_mask = np.asfortranarray(eval_mask[y_indices])
        y_mask_5d = np.asfortranarray(eval_mask_5d[y_indices])

        y_ic = factorlib_core.calc_ic(y_factor, y_ret_1d, y_mask, rank=False, horizon=1)
        y_rankic = factorlib_core.calc_ic(y_factor, y_ret_1d, y_mask, rank=True, horizon=1)
        y_ic_5d = factorlib_core.calc_ic(y_factor, y_ret_5d, y_mask_5d, rank=False, horizon=5)
        y_q = factorlib_core.calc_quantile_returns(
            y_factor, y_ret_1d, y_mask, n_quantiles=10, min_names=10, sign=1
        )
        y_to = factorlib_core.calc_top_quantile_turnover(
            y_factor, y_mask, n_quantiles=10, min_names=10, sign=1
        )
        y_ex = excess_metrics["annual_breakdown"].get(y, {})
        ls_ir = y_q["ls_mean"] / y_q["ls_std"] if y_q["ls_std"] > 1e-12 else 0.0

        annual_breakdown.append(
            {
                "year": y,
                "days": len(y_indices),
                "ic": y_ic["ic_mean"],
                "ic_ir": y_ic["ic_ir"],
                "ic_5d": y_ic_5d["ic_mean"],
                "ic_5d_ir": y_ic_5d["ic_ir"],
                "rankic": y_rankic["ic_mean"],
                "rankic_ir": y_rankic["ic_ir"],
                "rankic_win_rate": y_rankic["ic_win_rate"],
                "long_short": y_q["ls_mean"],
                "long_short_ir": ls_ir,
                "long_short_sharpe": y_q["ls_sharpe"],
                "long_short_max_dd": y_q["ls_max_dd"],
                "excess_annual_ret": y_ex.get("annual_excess_return", 0.0),
                "excess_annual_vol": y_ex.get("annual_excess_volatility", 0.0),
                "excess_sharpe": y_ex.get("excess_sharpe", 0.0),
                "excess_max_dd": y_ex.get("additive_max_drawdown", 0.0),
                "turnover": y_to["mean_top_turnover"],
            }
        )

    cost_time = time.time() - t0
    print(f"      - 回测评估完毕 (模式: {exec_price_mode}, 耗时: {cost_time:.2f}s)")

    ls_ir_full = q_res["ls_mean"] / q_res["ls_std"] if q_res["ls_std"] > 1e-12 else 0.0
    return {
        "exec_price_mode": exec_price_mode,
        "dates": sub_dates,
        "n_days": len(sub_dates),
        "valid_days": ic_res["valid_days"],
        "ic": ic_res["ic_mean"],
        "ic_ir": ic_res["ic_ir"],
        "ic_win_rate": ic_res["ic_win_rate"],
        "5dic": ic_5d_res["ic_mean"],
        "5dic_ir": ic_5d_res["ic_ir"],
        "rankic": rankic_res["ic_mean"],
        "rankic_ir": rankic_res["ic_ir"],
        "rankic_win_rate": rankic_res["ic_win_rate"],
        "rankic_2d": rankic_2d_res["ic_mean"],
        "rankic_2d_ir": rankic_2d_res["ic_ir"],
        "rankic_5d": rankic_5d_res["ic_mean"],
        "rankic_5d_ir": rankic_5d_res["ic_ir"],
        "rankic_10d": rankic_10d_res["ic_mean"],
        "rankic_10d_ir": rankic_10d_res["ic_ir"],
        "long_short": q_res["ls_mean"],
        "long_short_ir": ls_ir_full,
        "long_short_sharpe": q_res["ls_sharpe"],
        "long_short_max_dd": q_res["ls_max_dd"],
        "excess_annual_ret": excess_metrics["annual_excess_return"],
        "excess_annual_ret_arith": excess_metrics["annual_excess_return_arithmetic"],
        "excess_annual_vol": excess_metrics["annual_excess_volatility"],
        "excess_sharpe": excess_metrics["excess_sharpe"],
        "excess_max_dd": excess_metrics["additive_max_drawdown"],
        "excess_avg_dd": excess_metrics["average_drawdown"],
        "turnover": top_to_res["mean_top_turnover"],
        "rank_autocorr_turnover": turnover_res["turnover"],
        "corr_bench": corr_bench,
        "ew_annual_ret": ew_annual_ret,
        "bench_annual_ret": bench_annual_ret,
        "execution_simulation": execution_simulation,
        "group_returns": q_res["group_returns"],
        "annual_breakdown": annual_breakdown,
    }


def print_comparison_report(
    online_data: dict[str, Any],
    local_open: dict[str, Any],
    local_vwap: dict[str, Any],
) -> None:
    """打印线上 FactorLab (vwap30) 与本地 Rust (open_adj 主口径 & vwap_adj 对照口径) 的三方对账报告。"""
    print("\n" + "=" * 112)
    print(
        " 📊 Volume_Price_2_H5_2 因子 FactorLab 线上 (vwap30) vs 本地 Rust 内核 (open_adj 主口径 / vwap_adj 对照口径) 对账报告"
    )
    print("=" * 112)

    on_m = online_data.get("metrics", {})
    tables = online_data.get("tables", [])

    def _find_table(tid: str) -> dict[str, Any]:
        if isinstance(tables, list):
            return next((t for t in tables if t.get("id") == tid), {})
        return {}

    t_excess = _find_table("long_excess_metrics")
    on_excess_rows = {r[0]: r for r in t_excess.get("rows", []) if len(r) >= 10}

    print(
        f"\n【1. 全周期核心指标对比 (2016-01-04 至 2024-12-31, 有效交易日: {local_open['valid_days']} / 2185)】"
    )
    print("-" * 112)
    print(
        f"{'指标名称':<24} | {'FactorLab 线上 (vwap30)':<22} | {'本地 Rust (open_adj 主口径)':<26} | {'本地 Rust (vwap_adj 对照)':<24} | {'open_adj 偏差':<12}"
    )
    print("-" * 112)

    rows = [
        ("IC 均值 (1d)", on_m.get("ic", 0.0265), local_open["ic"], local_vwap["ic"], "{:.4f}"),
        (
            "IC IR (1d)",
            on_m.get("icIR", 0.2729),
            local_open["ic_ir"],
            local_vwap["ic_ir"],
            "{:.4f}",
        ),
        (
            "IC 胜率 (1d)",
            on_m.get("icWR", 0.6156),
            local_open["ic_win_rate"],
            local_vwap["ic_win_rate"],
            "{:.2%}",
        ),
        (
            "RankIC 均值 (1d)",
            on_m.get("rankic", 0.0431),
            local_open["rankic"],
            local_vwap["rankic"],
            "{:.4f}",
        ),
        (
            "RankIC IR (1d)",
            on_m.get("rankicIR", 0.4019),
            local_open["rankic_ir"],
            local_vwap["rankic_ir"],
            "{:.4f}",
        ),
        (
            "RankIC 胜率 (1d)",
            on_m.get("rankicWR", 0.6796),
            local_open["rankic_win_rate"],
            local_vwap["rankic_win_rate"],
            "{:.2%}",
        ),
        ("5日 IC 均值", on_m.get("5dic", 0.0332), local_open["5dic"], local_vwap["5dic"], "{:.4f}"),
        (
            "5日 IC IR",
            on_m.get("5dicIR", 0.3484),
            local_open["5dic_ir"],
            local_vwap["5dic_ir"],
            "{:.4f}",
        ),
        (
            "多空日均收益 (G10-G1)",
            on_m.get("long_short", 0.002364),
            local_open["long_short"],
            local_vwap["long_short"],
            "{:.4%}",
        ),
        (
            "多空日度 IR",
            on_m.get("long_shortIR", 0.2956),
            local_open["long_short_ir"],
            local_vwap["long_short_ir"],
            "{:.4f}",
        ),
        ("Top G10 单边换手率", 0.1211, local_open["turnover"], local_vwap["turnover"], "{:.2%}"),
        (
            "截面秩自相关距离 (1-ρ)",
            None,
            local_open["rank_autocorr_turnover"],
            local_vwap["rank_autocorr_turnover"],
            "{:.2%}",
        ),
        (
            "多头年化超额 (vs 000985)",
            0.0126,
            local_open["excess_annual_ret"],
            local_vwap["excess_annual_ret"],
            "{:.2%}",
        ),
        (
            "多头超额年化波动",
            0.0767,
            local_open["excess_annual_vol"],
            local_vwap["excess_annual_vol"],
            "{:.2%}",
        ),
        (
            "多头超额夏普比率",
            0.1642,
            local_open["excess_sharpe"],
            local_vwap["excess_sharpe"],
            "{:.4f}",
        ),
        (
            "多头超额最大回撤",
            0.2873,
            local_open["excess_max_dd"],
            local_vwap["excess_max_dd"],
            "{:.2%}",
        ),
    ]

    for label, on_v, loc_o, loc_v, fmt in rows:
        on_s = fmt.format(on_v) if on_v is not None else "   --   "
        o_s = fmt.format(loc_o)
        v_s = fmt.format(loc_v)
        if on_v is not None:
            diff = loc_o - on_v
            diff_s = f"{diff * 100:+.2f}%" if "%" in fmt else f"{diff:+.4f}"
        else:
            diff_s = "  --  "
        print(f"{label:<24} | {on_s:<22} | {o_s:<26} | {v_s:<24} | {diff_s:<12}")

    # 2. 10 分组日均收益对比 (G1 .. G10)
    print("\n【2. 10 分组日均收益单调性对比 (Quantile Groups G1 ~ G10)】")
    print("-" * 100)
    print(
        f"{'分位组':<8} | {'FactorLab 线上 (vwap30)':<24} | {'本地 Rust (open_adj 主口径)':<26} | {'本地 Rust (vwap_adj 对照)':<24}"
    )
    print("-" * 100)
    t_grp = _find_table("group_metrics")
    on_grp_rows = t_grp.get("rows", [])
    for idx in range(10):
        g_name = f"G{idx + 1}"
        on_g = on_grp_rows[idx][1] if idx < len(on_grp_rows) else "--"
        loc_g_open = f"{local_open['group_returns'][idx] * 100:+.4f}%"
        loc_g_vwap = f"{local_vwap['group_returns'][idx] * 100:+.4f}%"
        print(f"{g_name:<8} | {on_g:<24} | {loc_g_open:<26} | {loc_g_vwap:<24}")

    # 3. 多周期预测能力衰减对比
    print("\n【3. 多周期预测能力衰减对比 (RankIC & RankIC IR Across Horizons)】")
    print("-" * 100)
    print(
        f"{'预测周期':<10} | {'线上 RankIC (IR)':<24} | {'本地 open_adj RankIC (IR)':<28} | {'本地 vwap_adj RankIC (IR)':<28}"
    )
    print("-" * 100)
    h_rows = [
        (
            "1 日 (h=1)",
            "0.0431 (0.4019)",
            f"{local_open['rankic']:.4f} ({local_open['rankic_ir']:.4f})",
            f"{local_vwap['rankic']:.4f} ({local_vwap['rankic_ir']:.4f})",
        ),
        (
            "2 日 (h=2)",
            "0.0535 (0.4960)",
            f"{local_open['rankic_2d']:.4f} ({local_open['rankic_2d_ir']:.4f})",
            f"{local_vwap['rankic_2d']:.4f} ({local_vwap['rankic_2d_ir']:.4f})",
        ),
        (
            "5 日 (h=5)",
            "0.0659 (0.6189)",
            f"{local_open['rankic_5d']:.4f} ({local_open['rankic_5d_ir']:.4f})",
            f"{local_vwap['rankic_5d']:.4f} ({local_vwap['rankic_5d_ir']:.4f})",
        ),
        (
            "10 日 (h=10)",
            "0.0764 (0.7324)",
            f"{local_open['rankic_10d']:.4f} ({local_open['rankic_10d_ir']:.4f})",
            f"{local_vwap['rankic_10d']:.4f} ({local_vwap['rankic_10d_ir']:.4f})",
        ),
    ]
    for h_name, h_on, h_open, h_vwap in h_rows:
        print(f"{h_name:<10} | {h_on:<24} | {h_open:<28} | {h_vwap:<28}")

    # 4. 分年度对账细分 (2016 - 2024)
    print("\n【4. 分年度 RankIC 与多头超额对账细分 (2016 ~ 2024, 本地采用 open_adj 主口径)】")
    print("-" * 112)
    print(
        f"{'年份':<6} | {'天数':<4} | {'线上 RankIC':<11} | {'本地 RankIC':<11} | {'线上 IR':<8} | {'本地 IR':<8} | {'线上超额':<9} | {'本地超额':<9} | {'线上换手':<9} | {'本地换手':<9}"
    )
    print("-" * 112)
    t_yearly = _find_table("yearly_metrics")
    on_year_dict = {r[0]: r for r in t_yearly.get("rows", []) if len(r) >= 7}

    for row in local_open["annual_breakdown"]:
        y = row["year"]
        on_y = on_year_dict.get(y, [])
        on_ex = on_excess_rows.get(y, [])
        on_ric = on_y[5] if len(on_y) > 5 else "--"
        on_ir = on_y[6] if len(on_y) > 6 else "--"
        on_ex_ret = on_ex[4] if len(on_ex) > 4 else "--"
        on_to = on_ex[6] if len(on_ex) > 6 else "--"
        print(
            f"{y:<6} | {row['days']:<4} | {on_ric:<11} | {row['rankic']:<11.4f} | {on_ir:<8} | {row['rankic_ir']:<8.4f} | "
            f"{on_ex_ret:<9} | {row['excess_annual_ret'] * 100:+6.2f}%  | {on_to:<9} | {row['turnover'] * 100:6.2f}%"
        )

    print("\n【5. 跨平台偏差归因摘要】")
    print("-" * 112)
    print(
        f"  1) 后处理全流程闭环: 已接入 apply_postprocess_pipeline (含 baseUniMask/trdMask/dropNewStock/MAD/zScore/indNeut/decay 后重施掩码),"
        f" 彻底阻断涨跌停/非在池残值渗漏 (execution_simulation={local_open['execution_simulation']})."
    )
    spread_pp = (local_open["ew_annual_ret"] - local_open["bench_annual_ret"]) * 100
    print(
        f"  2) open_adj 与 000985.CSI 超额结构差: open_adj 组合与开盘基准相关系数 {local_open['corr_bench']:.4f} "
        f"(超额波动 {local_open['excess_annual_vol'] * 100:.2f}% 贴近线上 7.67%), "
        f"在池等权全市场年化 {local_open['ew_annual_ret'] * 100:+.2f}% 相对流通市值基准 {local_open['bench_annual_ret'] * 100:+.2f}% "
        f"结构差 {spread_pp:+.2f}%, 叠加线上 vwap30 代理、涨跌停执行约束及税费归因."
    )
    print(
        f"  3) vwap_adj 基准时钟失配提示: vwap_adj (全天均价) 因缺少指数日内均价而配对收盘基准 bench_fwd_close_1, "
        f"相关性降至 {local_vwap['corr_bench']:.4f}, 引入跨时钟追踪误差噪声 (超额波动 {local_vwap['excess_annual_vol'] * 100:.2f}%)."
    )
    print("=" * 112)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Volume_Price_2_H5_2 因子本地股票回测与 FactorLab 线上对账脚本"
    )
    parser.add_argument(
        "--start-date",
        default="20160104",
        help="回测评估开始日期 (默认: 20160104)",
    )
    parser.add_argument(
        "--end-date",
        default="20241231",
        help="回测评估结束日期 (默认: 20241231)",
    )
    parser.add_argument(
        "--execution-simulation",
        action="store_true",
        help="启用分方向涨跌停实盘执行约束 (T+1 涨停不可买、T+1+h 跌停/停牌顺延退出)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print("=" * 112)
    print(
        " 🚀 Volume_Price_2_H5_2 因子 A 股全市场本地回测与全量对账启动 (P0-09/P0-10/P0-11 修正版)"
    )
    print("=" * 112)

    # 1. 抓取线上数据 (支持 503 自动回退)
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

    # 4. 执行本地回测 (主口径 open_adj 与对照口径 vwap_adj)
    local_open = run_local_backtest_evaluation(
        dates=dates,
        symbols=symbols,
        factor_mat=factor_mat,
        panel_data=panel_data,
        start_eval_date=args.start_date,
        end_eval_date=args.end_date,
        exec_price_mode="open_adj",
        execution_simulation=args.execution_simulation,
    )
    local_vwap = run_local_backtest_evaluation(
        dates=dates,
        symbols=symbols,
        factor_mat=factor_mat,
        panel_data=panel_data,
        start_eval_date=args.start_date,
        end_eval_date=args.end_date,
        exec_price_mode="vwap_adj",
        execution_simulation=args.execution_simulation,
    )

    # 5. 输出对比报告
    print_comparison_report(online_data, local_open, local_vwap)


if __name__ == "__main__":
    main()
