"""单因子与多因子多维度评价管线、FactorLab Schema v4 统计归因与双层回测工作流 (core/evaluate.py)。

严格遵循白皮书第 8 章、10 章与 FactorLab 6 阶段工作流规范：
- 支持单因子在多样本切分区间（Train 2011-2018, Valid 2019-2021, Test 2022-2026.08, Full）
  与多预测期数（h in [1, 3, 5, 10, 20]）下的高并发评估；
- 自动根据训练集 Rank IC 符号决定因子的推荐方向 sign (1 或 -1)；
- 实现 FactorLab Schema v4 Layer 1 统计分布评估 (`evaluate_factor_v4_stats`)：
  - 5 类 IC 序列 (`ic`, `rank_ic`, `perc_ic` 首尾 20% 极值相关, `ic_5d` 线性 Pearson, `rank_ic_5d` 秩 Spearman) 及分年度 `ddof=1` IR 汇总；
  - 分位数 `G1..G5` 复利净值与单调性检验；
  - 多头超额分析 (`long_analysis`)：相对等权商品池基准的超额收益、加法口径最大回撤与换手率；
  - 6 大期货标准风格暴露归因 (`style_exposure`)；
- 提供端到端因子工作流编排 `run_factor_workflow()`，串联静态探针、求值物化、Layer 1 统计评价与 Layer 2 期货实盘逐日盯市回测。
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import factorlib_core
import numpy as np

from factorlib.core._hooks import resolve_panel_market, resolve_profile
from factorlib.core.panel import Panel
from factorlib.core.probe import (
    FactorDefinition,
    PostprocessSpec,
    evaluate_factor_definition,
    parse_postprocess_pipeline,
)
from factorlib.core.registry import FactorMetricsRecord, FactorRecord, FactorRegistry
from factorlib.core.styles import (
    _pearson_corr_1d,
    _spearman_corr_1d,
    compute_style_exposures,
)

if TYPE_CHECKING:
    from factorlib.core.backtest import BacktestResult

# 白皮书 Q6 与 13.4 节规定的标准样本切分区间
DEFAULT_SPLITS: dict[str, tuple[str, str]] = {
    "train": ("2011-01-01", "2018-12-31"),
    "valid": ("2019-01-01", "2021-12-31"),
    "test": ("2022-01-01", "2026-08-31"),
}

DEFAULT_HORIZONS: tuple[int, ...] = (1, 3, 5, 10, 20)


@dataclass
class FactorEvaluationResult:
    """因子综合评估结果容器。"""

    canon_expr: str
    canon_hash: str
    sign: int
    factor_matrix: np.ndarray
    metrics_by_split: dict[str, dict[int, FactorMetricsRecord]] = field(default_factory=dict)
    all_metrics: list[FactorMetricsRecord] = field(default_factory=list)
    n_nodes: int = 1
    depth: int = 1
    probe_report: dict[str, Any] | None = None
    v4_stats: dict[str, Any] | None = None

    def to_factor_record(
        self,
        source: str = "manual",
        source_run: str | None = None,
        data_version: str = "v1",
        status: str = "candidate",
        name: str | None = None,
        factor_type: str = "expression",
        definition_json: str | None = None,
        market: str = "futures",
        recipe_id: str | None = None,
    ) -> FactorRecord:
        """转换为可直接写入 SQLite 注册表的 FactorRecord 对象。"""
        return FactorRecord(
            market=market,
            canon_expr=self.canon_expr,
            canon_hash=self.canon_hash,
            mode="cs",
            sign=self.sign,
            n_nodes=self.n_nodes,
            depth=self.depth,
            source=source,
            source_run=source_run,
            recipe_id=recipe_id,
            data_version=data_version,
            status=status,
            metrics=list(self.all_metrics),
            name=name,
            factor_type=factor_type,
            definition_json=definition_json,
        )


def _resolve_engine_and_panel(
    engine_or_panel: factorlib_core.Engine | Panel,
) -> tuple[factorlib_core.Engine, np.ndarray, list[str]]:
    """解析底层 Engine 实例、日期数组与符号列表。"""
    if isinstance(engine_or_panel, Panel):
        panel = engine_or_panel
        engine = panel.to_rust_engine()
        dates = np.array([str(d) for d in panel.dates])
        symbols = list(panel.symbols)
    elif hasattr(engine_or_panel, "dates") and hasattr(engine_or_panel, "symbols"):
        engine = engine_or_panel
        dates = np.array(engine.dates)
        symbols = list(engine.symbols)
    else:
        raise TypeError("输入必须为 Panel 或 factorlib_core.Engine 实例")
    return engine, dates, symbols


def evaluate_factor(
    engine_or_panel: factorlib_core.Engine | Panel,
    expr: str | FactorDefinition,
    horizons: Sequence[int] = DEFAULT_HORIZONS,
    splits: dict[str, tuple[str, str]] | None = None,
    min_names: int = 5,
    n_quantiles: int = 5,
    force_sign: int | None = None,
    postprocess: PostprocessSpec | None = None,
    profile: Any | None = None,
) -> FactorEvaluationResult:
    """对单个 DSL 因子表达式或 FactorDefinition 执行全时段多预测期数综合评价。"""
    engine, dates, _ = _resolve_engine_and_panel(engine_or_panel)
    panel_obj = engine_or_panel if isinstance(engine_or_panel, Panel) else None

    probe_rep: dict[str, Any] | None = None
    if isinstance(expr, FactorDefinition) or postprocess is not None:
        if isinstance(expr, FactorDefinition):
            defn = expr
        else:
            defn = FactorDefinition(
                name="inline_factor",
                factor_type="expression",
                expression=str(expr),
                postprocess=parse_postprocess_pipeline(postprocess),
            )
        if panel_obj is not None:
            factor_mat, probe_rep = evaluate_factor_definition(
                defn, panel_obj, engine=engine, profile=profile
            )
            canon_expr = probe_rep["canon_expr"]
            canon_hash = probe_rep["canon_hash"]
            n_nodes = int(probe_rep["n_nodes"])
            depth = int(probe_rep["depth"])
        else:
            canon_info = engine.canonicalize(defn.expression)
            canon_expr = canon_info["canon_expr"]
            canon_hash = canon_info["canon_hash"]
            n_nodes = canon_info["n_nodes"]
            depth = canon_info["depth"]
            factor_mat = np.asarray(engine.eval(canon_expr))
    else:
        canon_info = engine.canonicalize(expr)
        canon_expr = canon_info["canon_expr"]
        canon_hash = canon_info["canon_hash"]
        n_nodes = canon_info["n_nodes"]
        depth = canon_info["depth"]
        factor_mat = np.asarray(engine.eval(canon_expr))

    return _evaluate_factor_matrix_core(
        engine=engine,
        dates=dates,
        factor_mat=factor_mat,
        canon_expr=canon_expr,
        canon_hash=canon_hash,
        n_nodes=n_nodes,
        depth=depth,
        horizons=horizons,
        splits=splits,
        min_names=min_names,
        n_quantiles=n_quantiles,
        force_sign=force_sign,
        probe_report=probe_rep,
    )


def _evaluate_factor_matrix_core(
    engine: factorlib_core.Engine,
    dates: np.ndarray,
    factor_mat: np.ndarray,
    canon_expr: str,
    canon_hash: str,
    n_nodes: int = 1,
    depth: int = 1,
    horizons: Sequence[int] = DEFAULT_HORIZONS,
    splits: dict[str, tuple[str, str]] | None = None,
    min_names: int = 5,
    n_quantiles: int = 5,
    force_sign: int | None = None,
    probe_report: dict[str, Any] | None = None,
) -> FactorEvaluationResult:
    """对给定 (T, N) 因子矩阵执行多切片多预测期数 Rust 评估。"""
    T, _ = factor_mat.shape

    eval_splits: dict[str, tuple[str, str]] = {}
    if splits is not None:
        eval_splits.update(splits)
    else:
        eval_splits.update(DEFAULT_SPLITS)

    if "full" not in eval_splits and len(dates) > 0:
        eval_splits["full"] = (str(dates[0]), str(dates[-1]))

    split_indices: dict[str, np.ndarray] = {}
    for split_name, (start_d, end_d) in eval_splits.items():
        mask_idx = np.where((dates >= start_d) & (dates <= end_d))[0]
        if len(mask_idx) > 0:
            split_indices[split_name] = mask_idx

    if "train" not in split_indices or len(split_indices["train"]) == 0:
        split_indices["train"] = np.arange(T)

    # 自动推断因子方向 sign
    if force_sign is not None:
        sign = 1 if force_sign >= 0 else -1
    else:
        train_idx = split_indices.get("train", np.arange(T))
        s_train = int(train_idx[0])
        e_train = int(train_idx[-1]) + 1
        f_train = np.asfortranarray(factor_mat[s_train:e_train])
        preferred_label = "y_5" if "y_5" in engine.label_names else engine.label_names[0]
        train_ic_res = engine.rank_ic(
            f_train,
            label_name=preferred_label,
            min_names=min_names,
            rank=True,
            horizon=5,
            start_row=s_train,
            end_row=e_train,
        )
        train_mean = train_ic_res["ic_mean"]
        sign = 1 if train_mean >= 0 else -1

    metrics_by_split: dict[str, dict[int, FactorMetricsRecord]] = {}
    all_metrics: list[FactorMetricsRecord] = []

    for split_name, idx_range in split_indices.items():
        metrics_by_split[split_name] = {}
        s_row = int(idx_range[0])
        e_row = int(idx_range[-1]) + 1
        f_sub = np.asfortranarray(factor_mat[s_row:e_row])

        for h in horizons:
            lbl_name = f"y_{h}"
            if lbl_name not in engine.label_names:
                lbl_name = (
                    "y_5"
                    if "y_5" in engine.label_names
                    else ("y_1" if "y_1" in engine.label_names else engine.label_names[0])
                )

            met = engine.factor_metrics(
                f_sub,
                label_name=lbl_name,
                ret_name="y_1"
                if "y_1" in engine.label_names
                else ("r_exec_next" if "r_exec_next" in engine.label_names else None),
                horizon=h,
                n_quantiles=n_quantiles,
                min_names=min_names,
                sign=sign,
                start_row=s_row,
                end_row=e_row,
            )

            rec = FactorMetricsRecord(
                split=split_name,
                horizon=h,
                ic_mean=float(met["ic_mean"]),
                ic_std=float(met["ic_std"]),
                ic_ir=float(met["ic_ir"]),
                ic_t=float(met["ic_t"]),
                ic_win_rate=float(met["ic_win_rate"]),
                coverage=float(met["coverage"]),
                turnover=float(met.get("top_turnover", met["turnover_val"])),
                ls_sharpe=float(met["ls_sharpe"]),
                ls_annual_ret=float(met["ls_annual_ret"]),
                ls_max_dd=float(met["ls_max_dd"]),
                top_turnover=float(met.get("top_turnover", met["turnover_val"])),
                rank_autocorr_turnover=float(met.get("rank_autocorr_turnover", 0.0)),
            )

            metrics_by_split[split_name][h] = rec
            all_metrics.append(rec)

    return FactorEvaluationResult(
        canon_expr=canon_expr,
        canon_hash=canon_hash,
        sign=sign,
        factor_matrix=factor_mat,
        metrics_by_split=metrics_by_split,
        all_metrics=all_metrics,
        n_nodes=n_nodes,
        depth=depth,
        probe_report=probe_report,
    )


# ==============================================================================
# Layer 1: FactorLab Schema v4 统计分布评估 (IC/PercIC/5dIC + 分组净值 + 多头超额 + 风格暴露)
# ==============================================================================


def _summarize_ic_slice(
    ic_arr: np.ndarray,
    rank_ic_arr: np.ndarray,
    perc_ic_arr: np.ndarray,
    ic_5d_arr: np.ndarray,
    rank_ic_5d_arr: np.ndarray | None = None,
) -> dict[str, float]:
    """按样本标准差 (ddof=1) 汇总指定时段内的 IC 统计指标（显式区分 5d Pearson IC 与 5d RankIC）。"""

    def _mean_std_ir_t(arr: np.ndarray) -> tuple[float, float, float, float]:
        valid = arr[np.isfinite(arr)]
        n_v = len(valid)
        if n_v == 0:
            return 0.0, 0.0, 0.0, 0.0
        mean_v = float(np.mean(valid))
        std_v = float(np.std(valid, ddof=1)) if n_v > 1 else 0.0
        ir_v = mean_v / std_v if std_v > 1e-12 else 0.0
        t_v = ir_v * float(np.sqrt(n_v))
        return mean_v, std_v, ir_v, t_v

    ic_m, ic_s, ic_ir, ic_t = _mean_std_ir_t(ic_arr)
    ric_m, ric_s, ric_ir, ric_t = _mean_std_ir_t(rank_ic_arr)
    pic_m, _, pic_ir, pic_t = _mean_std_ir_t(perc_ic_arr)
    ic5_m, _, ic5_ir, ic5_t = _mean_std_ir_t(ic_5d_arr)
    ric5_m, _, ric5_ir, ric5_t = (
        _mean_std_ir_t(rank_ic_5d_arr) if rank_ic_5d_arr is not None else (0.0, 0.0, 0.0, 0.0)
    )

    valid_ic = ic_arr[np.isfinite(ic_arr)]
    ic_win_rate = float(np.mean(valid_ic > 0.0)) if len(valid_ic) > 0 else 0.0
    valid_ric = rank_ic_arr[np.isfinite(rank_ic_arr)]
    win_rate = float(np.mean(valid_ric > 0.0)) if len(valid_ric) > 0 else 0.0

    return {
        "ic_mean": ic_m,
        "ic_std": ic_s,
        "ic_ir": ic_ir,
        "ic_t": ic_t,
        "ic_win_rate": ic_win_rate,
        "rank_ic_mean": ric_m,
        "rank_ic_std": ric_s,
        "rank_ic_ir": ric_ir,
        "rank_ic_t": ric_t,
        "perc_ic_mean": pic_m,
        "perc_ic_ir": pic_ir,
        "perc_ic_t": pic_t,
        "ic_5d_mean": ic5_m,
        "ic_5d_ir": ic5_ir,
        "ic_5d_t": ic5_t,
        "rank_ic_5d_mean": ric5_m,
        "rank_ic_5d_ir": ric5_ir,
        "rank_ic_5d_t": ric5_t,
        "win_rate": win_rate,
        "valid_days": float(len(valid_ric)),
    }


def calc_long_excess_metrics(
    long_rets: np.ndarray,
    bench_rets: np.ndarray,
    dates: list[str] | None = None,
    trading_days_per_year: float = 250.0,
    compute_annual: bool = True,
) -> dict[str, Any]:
    """计算多头超额收益系列指标（严格对齐 FactorLab 规范：复利几何年化、加法回撤、分年统计）。

    Parameters
    ----------
    long_rets : np.ndarray
        多头（如 G10）日收益率 1D 数组
    bench_rets : np.ndarray
        对齐到相同持仓窗口的基准日收益率 1D 数组
    dates : list[str] | None
        日期字符串列表（如 "20160104"），用于分年度切分与回撤起止点定位
    trading_days_per_year : float
        年化交易日基数（默认 250.0）
    compute_annual : bool
        是否递归切分计算分年度统计 (默认 True)

    Returns
    -------
    dict[str, Any]
        含 annual_excess_return (几何复利), annual_excess_return_arithmetic (算术单利),
        annual_excess_volatility, excess_sharpe, win_rate,
        additive_max_drawdown, average_drawdown, drawdown_start, drawdown_end,
        annual_breakdown, cumulative_additive, cumulative_compound
    """
    l_arr = np.asarray(long_rets, dtype=np.float64)
    b_arr = np.asarray(bench_rets, dtype=np.float64)
    n_total = min(len(l_arr), len(b_arr))
    if n_total == 0:
        return {
            "annual_excess_return": 0.0,
            "annual_excess_return_arithmetic": 0.0,
            "annual_excess_volatility": 0.0,
            "excess_sharpe": 0.0,
            "win_rate": 0.0,
            "average_drawdown": 0.0,
            "additive_max_drawdown": 0.0,
            "drawdown_start": "",
            "drawdown_end": "",
            "annual_breakdown": {},
            "cumulative_additive": [],
            "cumulative_compound": [],
        }

    excess_daily = l_arr[:n_total] - b_arr[:n_total]
    valid_mask = np.isfinite(excess_daily)
    valid_ex = excess_daily[valid_mask]
    n_valid = len(valid_ex)

    if n_valid > 0 and np.all(1.0 + valid_ex > 0.0):
        # 几何复利年化: (prod(1 + r))^(250 / N) - 1
        log_sum = float(np.sum(np.log1p(valid_ex)))
        ann_ret = float(np.expm1(log_sum * (trading_days_per_year / float(n_valid))))
    elif n_valid > 0:
        ann_ret = float(np.mean(valid_ex) * trading_days_per_year)
    else:
        ann_ret = 0.0

    ann_ret_arith = float(np.mean(valid_ex) * trading_days_per_year) if n_valid > 0 else 0.0
    ann_vol = (
        float(np.std(valid_ex, ddof=1) * np.sqrt(trading_days_per_year)) if n_valid > 1 else 0.0
    )
    sharpe = ann_ret / ann_vol if ann_vol > 1e-12 else 0.0
    win_rate = float(np.mean(valid_ex > 0.0)) if n_valid > 0 else 0.0

    # 加法累计超额收益曲线与最大回撤（FactorLab 规范: DD_t = Peak_t - Cum_t）
    cum_clean = np.where(valid_mask, excess_daily, 0.0)
    cum_excess_additive = np.cumsum(cum_clean)
    cum_excess_compound = np.cumprod(1.0 + cum_clean) - 1.0

    cum = 0.0
    peak = 0.0
    first_date = dates[0] if dates and len(dates) > 0 else ""
    peak_date = first_date
    max_dd = 0.0
    max_dd_start = first_date
    max_dd_end = first_date
    dds: list[float] = []

    for i in range(n_total):
        if not valid_mask[i]:
            continue
        r = excess_daily[i]
        cum += r
        d_str = dates[i] if dates and i < len(dates) else str(i)
        if cum > peak:
            peak = cum
            peak_date = d_str
        dd = peak - cum
        dds.append(dd)
        if dd > max_dd:
            max_dd = dd
            max_dd_start = peak_date
            max_dd_end = d_str
    avg_dd = float(np.mean(dds)) if dds else 0.0

    annual_breakdown: dict[str, dict[str, Any]] = {}
    if compute_annual and dates and len(dates) >= n_total:
        sub_dates = dates[:n_total]
        years = sorted({d[:4] for d in sub_dates if len(d) >= 4})
        for y in years:
            y_indices = [i for i, d in enumerate(sub_dates) if d.startswith(y)]
            if not y_indices:
                continue
            y_long = l_arr[y_indices]
            y_bench = b_arr[y_indices]
            y_d = [sub_dates[i] for i in y_indices]
            y_res = calc_long_excess_metrics(
                y_long,
                y_bench,
                dates=y_d,
                trading_days_per_year=trading_days_per_year,
                compute_annual=False,
            )
            annual_breakdown[y] = y_res

    return {
        "annual_excess_return": ann_ret,
        "annual_excess_return_arithmetic": ann_ret_arith,
        "annual_excess_volatility": ann_vol,
        "excess_sharpe": sharpe,
        "win_rate": win_rate,
        "average_drawdown": avg_dd,
        "additive_max_drawdown": max_dd,
        "drawdown_start": max_dd_start,
        "drawdown_end": max_dd_end,
        "annual_breakdown": annual_breakdown,
        "cumulative_additive": [float(x) for x in cum_excess_additive],
        "cumulative_compound": [float(x) for x in cum_excess_compound],
    }


def evaluate_factor_v4_stats(
    factor_matrix: np.ndarray,
    panel: Panel,
    n_quantiles: int = 5,
    sign: int = 1,
    min_names: int = 5,
    benchmark_rets: np.ndarray | None = None,
    trading_days_per_year: float = 250.0,
    profile: Any | None = None,
    eval_start_date: str | None = None,
    eval_end_date: str | None = None,
) -> dict[str, Any]:
    """执行 FactorLab Schema v4 标准统计分布评价 (Layer 1 Statistical Evaluation)。"""
    raw_T, N = factor_matrix.shape
    raw_dates = [str(d) for d in panel.dates]

    # 对包含 2015-11 ~ 2015-12 预热缓冲区的全历史 A 股标准面板，默认从 2016-01-01 开始统计评价（对齐 2016-2024 权威对账区间）
    if (
        eval_start_date is None
        and raw_T > 1000
        and len(raw_dates) > 0
        and raw_dates[0].replace("-", "").startswith("201511")
    ):
        eval_start_date = "20160101"

    # 先在全长 panel 上取好风格字典（利用面板级缓存），再按评估窗口切片
    resolved_profile = (
        profile if profile is not None else resolve_profile(resolve_panel_market(panel))
    )
    full_styles_dict = resolved_profile.style_factors(panel)

    if eval_start_date is not None or eval_end_date is not None:
        s_norm = eval_start_date.replace("-", "") if eval_start_date else "00000000"
        e_norm = eval_end_date.replace("-", "") if eval_end_date else "99999999"
        eval_idx = [
            i for i, d in enumerate(raw_dates) if s_norm <= d.replace("-", "")[:8] <= e_norm
        ]
        if not eval_idx:
            eval_idx = list(range(raw_T))
    else:
        eval_idx = list(range(raw_T))

    dates = [raw_dates[i] for i in eval_idx]
    T = len(dates)
    sub_factor = np.asarray(factor_matrix[eval_idx], dtype=np.float64)
    directed = sub_factor * (1.0 if sign >= 0 else -1.0)
    mask = panel.valid_mask[eval_idx]
    styles_dict = {k: v[eval_idx] for k, v in full_styles_dict.items()}

    # 提取 1 日执行收益与 5 日累计收益标签
    if "y_1" in panel.labels:
        ret_1d = panel.labels["y_1"][eval_idx]
    elif "r_exec_next" in panel.labels:
        ret_1d = panel.labels["r_exec_next"][eval_idx]
    else:
        first_lbl = (
            next(iter(panel.labels.values()))[eval_idx] if panel.labels else np.zeros((T, N))
        )
        ret_1d = first_lbl

    if "y_5" in panel.labels:
        ret_5d = panel.labels["y_5"][eval_idx]
    else:
        ret_5d = ret_1d

    ic_series = np.full(T, np.nan, dtype=np.float64)
    rank_ic_series = np.full(T, np.nan, dtype=np.float64)
    perc_ic_series = np.full(T, np.nan, dtype=np.float64)
    ic_5d_series = np.full(T, np.nan, dtype=np.float64)
    rank_ic_5d_series = np.full(T, np.nan, dtype=np.float64)

    group_daily_rets = {f"G{q + 1}": np.zeros(T, dtype=np.float64) for q in range(n_quantiles)}
    active_days_mask = np.zeros(T, dtype=bool)
    if benchmark_rets is not None:
        b_full = np.asarray(benchmark_rets, dtype=np.float64)
        bench_daily_rets = b_full[eval_idx] if len(b_full) == raw_T else b_full[:T]
    else:
        bench_daily_rets = np.full(T, np.nan, dtype=np.float64)
    top_turnover = np.full(T, np.nan, dtype=np.float64)
    prev_top_set: set[int] = set()
    coverage_series = np.full(T, np.nan, dtype=np.float64)

    if N > 200:
        # 大截面（如 A 股 5000+ 标的）直接调用 Rust Rayon 并行算子计算 IC 与分组收益
        dir_f = np.asfortranarray(directed, dtype=np.float64)
        r1_f = np.asfortranarray(ret_1d, dtype=np.float64)
        r5_f = np.asfortranarray(ret_5d, dtype=np.float64)
        v1_mask = np.asfortranarray(mask & np.isfinite(dir_f) & np.isfinite(r1_f), dtype=bool)
        v5_mask = np.asfortranarray(mask & np.isfinite(dir_f) & np.isfinite(r5_f), dtype=bool)

        ic_res_1 = factorlib_core.calc_ic(
            dir_f, r1_f, v1_mask, min_names=min_names, rank=False, horizon=1
        )
        ric_res_1 = factorlib_core.calc_ic(
            dir_f, r1_f, v1_mask, min_names=min_names, rank=True, horizon=1
        )
        ic_res_5 = factorlib_core.calc_ic(
            dir_f, r5_f, v5_mask, min_names=min_names, rank=False, horizon=5
        )
        ric_res_5 = factorlib_core.calc_ic(
            dir_f, r5_f, v5_mask, min_names=min_names, rank=True, horizon=5
        )

        ic_series = np.asarray(ic_res_1["daily_ic"], dtype=np.float64)
        rank_ic_series = np.asarray(ric_res_1["daily_ic"], dtype=np.float64)
        ic_5d_series = np.asarray(ic_res_5["daily_ic"], dtype=np.float64)
        rank_ic_5d_series = np.asarray(ric_res_5["daily_ic"], dtype=np.float64)

        q_res = factorlib_core.calc_quantile_returns(
            dir_f, r1_f, v1_mask, n_quantiles=n_quantiles, min_names=min_names, sign=1
        )
        g_raw = q_res["group_daily_returns"]
        ls_raw = np.asarray(q_res["ls_daily_returns"], dtype=np.float64)
        active_days_mask = np.isfinite(ls_raw)
        for q in range(n_quantiles):
            arr_q = np.asarray(g_raw[q], dtype=np.float64)
            group_daily_rets[f"G{q + 1}"] = np.where(np.isfinite(arr_q), arr_q, 0.0)
        top_turnover = np.asarray(q_res["top_turnover_series"], dtype=np.float64)

        # 使用 O(N) argpartition 计算首尾 20% PercIC 与覆盖率
        univ_counts = np.sum(mask, axis=1)
        valid_counts = np.sum(v1_mask, axis=1)
        with np.errstate(invalid="ignore", divide="ignore"):
            coverage_series = np.where(univ_counts > 0, valid_counts / univ_counts, np.nan)

        for t in range(T):
            m = int(valid_counts[t])
            if m < min_names:
                continue
            row_m = v1_mask[t, :]
            fx = directed[t, row_m]
            ry = ret_1d[t, row_m]
            if benchmark_rets is None:
                bench_daily_rets[t] = float(np.mean(ry))
            tail_k = max(2, int(np.ceil(0.2 * m)))
            if 2 * tail_k <= m:
                part_low = np.argpartition(fx, tail_k - 1)[:tail_k]
                part_high = np.argpartition(fx, m - tail_k)[m - tail_k :]
                ext_pos = np.concatenate([part_low, part_high])
                perc_ic_series[t] = _pearson_corr_1d(fx[ext_pos], ry[ext_pos])
            else:
                perc_ic_series[t] = ic_series[t]
    else:
        for t in range(T):
            f_row = directed[t, :]
            r1_row = ret_1d[t, :]
            r5_row = ret_5d[t, :]

            u_cnt = int(np.sum(mask[t, :]))
            valid_1d = mask[t, :] & np.isfinite(f_row) & np.isfinite(r1_row)
            idx_1d = np.where(valid_1d)[0]
            m = len(idx_1d)
            if u_cnt > 0 and m > 0:
                coverage_series[t] = float(m / u_cnt)

            if m >= min_names:
                active_days_mask[t] = True
                fx = f_row[idx_1d]
                ry = r1_row[idx_1d]
                ic_series[t] = _pearson_corr_1d(fx, ry)
                rank_ic_series[t] = _spearman_corr_1d(fx, ry)

                # PercIC: 首尾 20% 极值品种的截面 Pearson 相关系数
                order = np.argsort(fx, kind="mergesort")
                tail_k = max(2, int(np.ceil(0.2 * m)))
                if 2 * tail_k <= m:
                    ext_pos = np.concatenate([order[:tail_k], order[m - tail_k :]])
                else:
                    ext_pos = order
                perc_ic_series[t] = _pearson_corr_1d(fx[ext_pos], ry[ext_pos])

                # 等权商品池基准日收益 (Equal-Weight Universe Benchmark: 若未传入外部基准)
                if benchmark_rets is None:
                    bench_daily_rets[t] = float(np.mean(ry))

                # 分位数分组收益 G1 (最低) .. G{n_quantiles} (最高/多头组)
                for q in range(n_quantiles):
                    s_i = int(round(q * m / n_quantiles))
                    e_i = int(round((q + 1) * m / n_quantiles))
                    g_pos = order[s_i:e_i]
                    if len(g_pos) > 0:
                        group_daily_rets[f"G{q + 1}"][t] = float(np.mean(ry[g_pos]))
                    if q == n_quantiles - 1:
                        curr_top_set = set(idx_1d[g_pos].tolist())
                        if prev_top_set and curr_top_set:
                            kp = float(len(prev_top_set))
                            kc = float(len(curr_top_set))
                            o = float(len(prev_top_set & curr_top_set))
                            sum_diff = (
                                o * abs(1.0 / kc - 1.0 / kp) + (1.0 - o / kc) + (1.0 - o / kp)
                            )
                            top_turnover[t] = float(np.clip(0.5 * sum_diff, 0.0, 1.0))
                        prev_top_set = curr_top_set

            valid_5d = mask[t, :] & np.isfinite(f_row) & np.isfinite(r5_row)
            idx_5d = np.where(valid_5d)[0]
            if len(idx_5d) >= min_names:
                fx5 = f_row[idx_5d]
                ry5 = r5_row[idx_5d]
                ic_5d_series[t] = _pearson_corr_1d(fx5, ry5)
                rank_ic_5d_series[t] = _spearman_corr_1d(fx5, ry5)

    # 按年份与全样本汇总 IC
    years = sorted({d[:4] for d in dates if len(d) >= 4})
    ic_annual: dict[str, dict[str, float]] = {}
    for y in years:
        y_idx = [i for i, d in enumerate(dates) if d.startswith(y)]
        y_sum = _summarize_ic_slice(
            ic_series[y_idx],
            rank_ic_series[y_idx],
            perc_ic_series[y_idx],
            ic_5d_series[y_idx],
            rank_ic_5d_series[y_idx],
        )
        y_cov = coverage_series[y_idx][np.isfinite(coverage_series[y_idx])]
        y_sum["coverage"] = float(np.mean(y_cov)) if len(y_cov) > 0 else 0.0
        ic_annual[y] = y_sum
    ic_full = _summarize_ic_slice(
        ic_series, rank_ic_series, perc_ic_series, ic_5d_series, rank_ic_5d_series
    )
    valid_cov_all = coverage_series[np.isfinite(coverage_series)]
    ic_full["coverage"] = float(np.mean(valid_cov_all)) if len(valid_cov_all) > 0 else 0.0

    # 分组复利净值与年化收益（净值序列保持完整 T 长度，统计量从首个有效活跃日开始以免前导 0 稀释波动率与年化指标）
    group_nav_series: dict[str, list[float]] = {}
    group_summary: dict[str, dict[str, float]] = {}
    first_active_day = int(np.argmax(active_days_mask)) if np.any(active_days_mask) else None

    for g_key, g_rets in group_daily_rets.items():
        nav = np.cumprod(1.0 + g_rets)
        group_nav_series[g_key] = [float(x) for x in nav]
        if first_active_day is None:
            group_summary[g_key] = {
                "daily_mean": 0.0,
                "total_return": 0.0,
                "annual_return": 0.0,
                "annual_volatility": 0.0,
                "sharpe": 0.0,
                "max_drawdown": 0.0,
                "win_rate": 0.0,
            }
            continue
        active_rets = g_rets[first_active_day:][active_days_mask[first_active_day:]]
        active_len = len(active_rets)
        n_years = max(active_len / trading_days_per_year, 1.0 / trading_days_per_year)
        tot_ret = float(nav[-1] - 1.0) if len(nav) > 0 else 0.0
        ann_ret = float((1.0 + max(tot_ret, -0.9999)) ** (1.0 / n_years) - 1.0)
        vol = (
            float(np.std(active_rets, ddof=1) * np.sqrt(trading_days_per_year))
            if active_len > 1
            else 0.0
        )
        sharpe = ann_ret / vol if vol > 1e-12 else 0.0
        peaks = np.maximum.accumulate(nav)
        max_dd = float(np.max((peaks - nav) / np.maximum(peaks, 1e-12))) if len(nav) > 0 else 0.0
        win_r = float(np.mean(active_rets > 0.0)) if active_len > 0 else 0.0
        group_summary[g_key] = {
            "daily_mean": float(np.mean(active_rets)) if active_len > 0 else 0.0,
            "total_return": tot_ret,
            "annual_return": ann_ret,
            "annual_volatility": vol,
            "sharpe": sharpe,
            "max_drawdown": max_dd,
            "win_rate": win_r,
        }

    # 多空对冲 (Top - Bottom) 统计
    top_key = f"G{n_quantiles}"
    bot_key = "G1"
    ls_daily_arr = group_daily_rets[top_key] - group_daily_rets[bot_key]
    ls_nav = np.cumprod(1.0 + ls_daily_arr)
    if first_active_day is not None:
        ls_act = ls_daily_arr[first_active_day:][active_days_mask[first_active_day:]]
        ls_mean = float(np.mean(ls_act)) if len(ls_act) > 0 else 0.0
        ls_std = float(np.std(ls_act, ddof=1)) if len(ls_act) > 1 else 0.0
        ls_ir = ls_mean / ls_std if ls_std > 1e-12 else 0.0
        ls_ann = ls_mean * trading_days_per_year
        ls_vol = ls_std * float(np.sqrt(trading_days_per_year))
        ls_sharpe = ls_ann / ls_vol if ls_vol > 1e-12 else 0.0
        ls_peaks = np.maximum.accumulate(ls_nav)
        ls_mdd = float(np.max((ls_peaks - ls_nav) / np.maximum(ls_peaks, 1e-12)))
        ls_win = float(np.mean(ls_act > 0.0)) if len(ls_act) > 0 else 0.0
    else:
        ls_mean = ls_ir = ls_ann = ls_vol = ls_sharpe = ls_mdd = ls_win = 0.0

    # 多头超额分析 (Long Excess Analysis: Top Group vs Benchmark；未达 min_names 的预热日置 NaN，避免混入 -benchmark)
    long_rets_for_excess = np.where(active_days_mask, group_daily_rets[top_key], np.nan)
    excess_res = calc_long_excess_metrics(
        long_rets_for_excess,
        bench_daily_rets,
        dates=dates,
        trading_days_per_year=trading_days_per_year,
    )

    valid_to = top_turnover[np.isfinite(top_turnover)]
    mean_top_turnover = float(np.mean(valid_to)) if len(valid_to) > 0 else 0.0

    long_annual = excess_res["annual_breakdown"]
    for y, y_dict in long_annual.items():
        y_idx = [i for i, d in enumerate(dates) if d.startswith(y)]
        y_to = top_turnover[y_idx][np.isfinite(top_turnover[y_idx])]
        y_dict["daily_turnover"] = float(np.mean(y_to)) if len(y_to) > 0 else 0.0
        y_ls = ls_daily_arr[y_idx][active_days_mask[y_idx]]
        y_ls_m = float(np.mean(y_ls)) if len(y_ls) > 0 else 0.0
        y_ls_s = float(np.std(y_ls, ddof=1)) if len(y_ls) > 1 else 0.0
        y_dict["ls_mean"] = y_ls_m
        y_dict["ls_ir"] = y_ls_m / y_ls_s if y_ls_s > 1e-12 else 0.0

    style_exp = compute_style_exposures(
        sub_factor,
        styles_dict,
        mask=mask,
        n_quantiles=n_quantiles,
        sign=sign,
    )

    top_s = group_summary[top_key]
    bot_s = group_summary[bot_key]
    portfolio_summary = {
        "long": {
            "label": f"多头 ({top_key})",
            "annual_return": top_s["annual_return"],
            "sharpe": top_s["sharpe"],
            "calmar": top_s["annual_return"] / top_s["max_drawdown"]
            if top_s["max_drawdown"] > 1e-6
            else 0.0,
            "max_drawdown": top_s["max_drawdown"],
            "win_rate": top_s["win_rate"],
            "turnover": mean_top_turnover,
            "excess_annual": excess_res["annual_excess_return"],
        },
        "short": {
            "label": f"空头 ({bot_key})",
            "annual_return": bot_s["annual_return"],
            "sharpe": bot_s["sharpe"],
            "calmar": bot_s["annual_return"] / bot_s["max_drawdown"]
            if bot_s["max_drawdown"] > 1e-6
            else 0.0,
            "max_drawdown": bot_s["max_drawdown"],
            "win_rate": bot_s["win_rate"],
            "turnover": mean_top_turnover,
            "excess_annual": 0.0,
        },
        "long_short": {
            "label": f"多空 ({top_key}-{bot_key})",
            "daily_mean": ls_mean,
            "ir": ls_ir,
            "annual_return": ls_ann,
            "sharpe": ls_sharpe,
            "calmar": ls_ann / ls_mdd if ls_mdd > 1e-6 else 0.0,
            "max_drawdown": ls_mdd,
            "win_rate": ls_win,
            "turnover": mean_top_turnover,
            "excess_annual": ls_ann,
            "nav_series": [float(x) for x in ls_nav],
        },
        "long_excess": {
            "label": f"{top_key} 超额 (vs 基准)",
            "annual_return": excess_res["annual_excess_return"],
            "sharpe": excess_res["excess_sharpe"],
            "calmar": (
                excess_res["annual_excess_return"] / excess_res["additive_max_drawdown"]
                if excess_res["additive_max_drawdown"] > 1e-6
                else 0.0
            ),
            "max_drawdown": excess_res["additive_max_drawdown"],
            "win_rate": excess_res["win_rate"],
            "turnover": mean_top_turnover,
            "excess_annual": excess_res["annual_excess_return"],
        },
    }

    clean_bench_rets = np.where(np.isfinite(bench_daily_rets), bench_daily_rets, 0.0)
    return {
        "schema_version": "v4",
        "sign": sign,
        "dates": dates,
        "ic_analysis": {
            "summary": ic_full,
            "annual": ic_annual,
            "series": {
                "ic": [float(x) if np.isfinite(x) else None for x in ic_series],
                "rank_ic": [float(x) if np.isfinite(x) else None for x in rank_ic_series],
                "perc_ic": [float(x) if np.isfinite(x) else None for x in perc_ic_series],
                "ic_5d": [float(x) if np.isfinite(x) else None for x in ic_5d_series],
                "rank_ic_5d": [float(x) if np.isfinite(x) else None for x in rank_ic_5d_series],
            },
        },
        "group_analysis": {
            "n_quantiles": n_quantiles,
            "summary": group_summary,
            "nav_series": group_nav_series,
            "benchmark_nav": [float(x) for x in np.cumprod(1.0 + clean_bench_rets)],
        },
        "long_analysis": {
            "summary": {
                "annual_excess_return": excess_res["annual_excess_return"],
                "annual_excess_return_arithmetic": excess_res["annual_excess_return_arithmetic"],
                "annual_excess_volatility": excess_res["annual_excess_volatility"],
                "excess_sharpe": excess_res["excess_sharpe"],
                "additive_max_drawdown": excess_res["additive_max_drawdown"],
                "average_drawdown": excess_res["average_drawdown"],
                "drawdown_start": excess_res["drawdown_start"],
                "drawdown_end": excess_res["drawdown_end"],
                "win_rate": excess_res["win_rate"],
                "daily_turnover": mean_top_turnover,
            },
            "annual": long_annual,
            "cum_excess_additive": excess_res["cumulative_additive"],
            "cum_excess_compound": excess_res["cumulative_compound"],
        },
        "portfolio_summary": portfolio_summary,
        "style_exposure": style_exp,
    }


# ==============================================================================
# 端到端双层因子工作流 (Probe + Materialization + Layer 1 v4 + Layer 2 Futures MTM)
# ==============================================================================


@dataclass
class DualLayerWorkflowResult:
    """串联静态探针、因子物化、Layer 1 统计分布归因与 Layer 2 期货实盘盯市回测的完整结果。"""

    definition: FactorDefinition
    probe_report: dict[str, Any]
    evaluation: FactorEvaluationResult
    v4_stats: dict[str, Any]
    backtest_result: BacktestResult
    weights: np.ndarray
    factor_id: int | None = None
    version_locked: int | None = None
    materialization_path: Path | None = None


def run_factor_workflow(
    panel: Panel,
    definition: FactorDefinition | dict[str, Any] | str,
    postprocess: PostprocessSpec | None = None,
    portfolio_config: Any | None = None,
    backtest_config: Any | None = None,
    registry: FactorRegistry | None = None,
    materialize: bool = True,
    lock_version: bool = True,
    force_sign: int | None = None,
    n_quantiles: int = 5,
    profile: Any | None = None,
) -> DualLayerWorkflowResult:
    """一键运行完整 6 阶段因子工作流：
    Stage 2/3 (定义与静态探针) -> Stage 4 (Rust 求值、后处理与物化) -> Stage 5 (Layer 1 统计评价 + Layer 2 实盘回测)。
    """
    from factorlib.core.backtest import run_backtest
    from factorlib.core.portfolio import (
        build_portfolio_weights,
        compute_daily_returns_from_prices,
    )

    resolved_profile = (
        profile if profile is not None else resolve_profile(resolve_panel_market(panel))
    )

    if isinstance(definition, FactorDefinition):
        defn = definition
        if postprocess is not None:
            defn.postprocess = parse_postprocess_pipeline(postprocess)
    elif isinstance(definition, dict):
        defn = FactorDefinition.from_dict(definition)
        if postprocess is not None:
            defn.postprocess = parse_postprocess_pipeline(postprocess)
    else:
        defn = FactorDefinition(
            name="workflow_factor",
            factor_type="expression",
            expression=str(definition),
            postprocess=parse_postprocess_pipeline(postprocess),
        )

    # Stage 3 & Stage 4: 静态探针校验 + 主因子求值 + 12 算子声明式后处理
    engine = panel.to_rust_engine()
    factor_mat, probe_rep = evaluate_factor_definition(
        defn, panel, engine=engine, profile=resolved_profile
    )

    # Stage 5 Layer 1 (Part A): 多样本切分、多预测窗口评价并推断方向 sign
    avail_horizons = tuple(
        int(lbl[2:])
        for lbl in sorted(engine.label_names)
        if lbl.startswith("y_") and lbl[2:].isdigit()
    )
    target_horizons = (1, 5) if panel.N > 200 else DEFAULT_HORIZONS
    eval_horizons = tuple(h for h in target_horizons if h in avail_horizons) or avail_horizons
    eval_res = _evaluate_factor_matrix_core(
        engine=engine,
        dates=np.array([str(d) for d in panel.dates]),
        factor_mat=factor_mat,
        canon_expr=probe_rep["canon_expr"],
        canon_hash=probe_rep["canon_hash"],
        n_nodes=int(probe_rep["n_nodes"]),
        depth=int(probe_rep["depth"]),
        horizons=eval_horizons,
        n_quantiles=n_quantiles,
        force_sign=force_sign,
        probe_report=probe_rep,
    )

    # Stage 5 Layer 1 (Part B): FactorLab Schema v4 统计分布评估与风格暴露归因
    bench_rets = None
    if "bench_ret" in panel.aux_fields:
        candidate_bench = panel.aux_fields["bench_ret"][:, 0]
        if np.any(np.isfinite(candidate_bench)):
            bench_rets = candidate_bench

    v4_stats = evaluate_factor_v4_stats(
        factor_matrix=factor_mat,
        panel=panel,
        n_quantiles=n_quantiles,
        sign=eval_res.sign,
        benchmark_rets=bench_rets,
        profile=resolved_profile,
    )
    eval_res.v4_stats = v4_stats

    # Stage 5 Layer 2: 市场组合优化 + Rust 实盘持仓级回测
    bt_cfg = backtest_config
    port_cfg = (
        portfolio_config
        if portfolio_config is not None
        else (bt_cfg.portfolio if bt_cfg is not None else None)
    )

    directed_scores = np.asfortranarray(factor_mat * float(eval_res.sign), dtype=np.float64)
    close_field = (
        panel.fields.get("close")
        if "close" in panel.fields
        else panel.fields.get("close_hfq", panel.fields.get("open_adj"))
    )
    if close_field is None:
        raise KeyError("面板中未找到收盘价字段 ('close' / 'close_hfq' / 'open_adj')")
    hist_returns = compute_daily_returns_from_prices(close_field)
    weights = build_portfolio_weights(
        scores=directed_scores,
        returns=hist_returns,
        mask=panel.valid_mask,
        cfg=port_cfg,
        profile=resolved_profile,
        panel=panel,
    )
    bt_res = run_backtest(panel=panel, weights=weights, config=bt_cfg, profile=resolved_profile)

    # 可选注册表入库、物化与不可变版本锁定
    fid: int | None = None
    mat_path: Path | None = None
    ver_num: int | None = None
    if registry is not None:
        rec = eval_res.to_factor_record(
            source="dl_alpha" if defn.factor_type == "dl_alpha" else "manual",
            name=defn.name,
            factor_type=defn.factor_type,
            definition_json=defn.to_json(),
            market=resolved_profile.name,
            recipe_id=getattr(defn, "recipe_id", None),
        )
        fid = registry.register_factor(rec, upsert=True)
        if materialize:
            mat_path = registry.materialize_factor(fid, factor_mat, panel.dates, panel.symbols)
        if lock_version:
            ver_rec = registry.lock_factor_version(
                fid,
                probe_report=probe_rep,
                eval_summary={
                    "ic_summary": v4_stats["ic_analysis"]["summary"],
                    "long_summary": v4_stats["long_analysis"]["summary"],
                    "backtest_metrics": bt_res.metrics,
                },
            )
            ver_num = ver_rec.version

    return DualLayerWorkflowResult(
        definition=defn,
        probe_report=probe_rep,
        evaluation=eval_res,
        v4_stats=v4_stats,
        backtest_result=bt_res,
        weights=weights,
        factor_id=fid,
        version_locked=ver_num,
        materialization_path=mat_path,
    )


# ==============================================================================
# 经典文献种子因子定义
# ==============================================================================


def get_seed_factors() -> list[dict[str, Any]]:
    """获取学术文献与期货多因子工业界公认的 4 大经典种子因子定义规格。"""
    return [
        {
            "name": "momentum_20d",
            "expr": "pct_change(close, 20)",
            "sign": 1,
            "description": "20日价格收益动量 (Jegadeesh & Titman 1993, Moskowitz et al. 2012 期货时序与截面动量)",
        },
        {
            "name": "roll_yield_main_sub",
            "expr": "ry_main_sub",
            "sign": 1,
            "description": "主力-次主力展期收益率 (Erb & Harvey 2006, 现货溢价 Backwardation 驱动正向展期收益)",
        },
        {
            "name": "warehouse_stock_chg5",
            "expr": "wh_chg5",
            "sign": -1,
            "description": "5日仓单库存变动率 (Gorton et al. 2013, 注册仓单累积代表供大于求，预期未来价格下跌)",
        },
        {
            "name": "member_net_long_chg5",
            "expr": "mr_net_ratio_chg5",
            "sign": 1,
            "description": "会员持仓前20大主力席位5日净多头占比变动量 (主力聪明钱持仓异动，正向预测未来价格)",
        },
    ]


def register_seed_factors(
    registry: FactorRegistry,
    engine_or_panel: factorlib_core.Engine | Panel,
    run_id: str = "seed_init",
    status: str = "accepted",
    market: str | None = None,
) -> list[int]:
    """批量评估并将 4 大经典种子因子原子化登记入库 factors.sqlite。"""
    seed_defs = get_seed_factors()
    registered_ids = []
    eff_market = market or (
        resolve_panel_market(engine_or_panel) if isinstance(engine_or_panel, Panel) else "futures"
    )

    for s_def in seed_defs:
        res = evaluate_factor(
            engine_or_panel=engine_or_panel,
            expr=s_def["expr"],
            force_sign=s_def["sign"],
        )
        rec = res.to_factor_record(
            source="seed",
            source_run=run_id,
            status=status,
            name=s_def["name"],
            market=eff_market,
        )
        fid = registry.register_factor(rec, run_id=run_id, upsert=True)
        registered_ids.append(fid)

    return registered_ids
