"""单因子与多因子多维度评价管线与种子因子注册 (core/evaluate.py)。

严格遵循白皮书第 8 章、10 章与 P3 验收规范：
- 支持单因子在多样本切分区间（Train 2011-2018, Valid 2019-2021, Test 2022-2026.08, Full）
  与多预测期数（h in [1, 3, 5, 10, 20]）下的高并发评估；
- 自动根据训练集 Rank IC 符号决定因子的推荐方向 sign (1 或 -1)；
- 组装 IC（均值、IR、t 统计量、胜率、覆盖率）、分位数收益（多空年化、夏普、最大回撤、单调性）、换手率；
- 定义并验证经典学术文献 4 大种子因子（动量、展期收益率、仓单变动、会员持仓异动）；
- 与 SQLite 因子注册表无缝联动。
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import factorlib_core
import numpy as np

from factorlib.core.panel import Panel
from factorlib.core.registry import FactorMetricsRecord, FactorRecord, FactorRegistry

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

    def to_factor_record(
        self,
        source: str = "manual",
        source_run: str | None = None,
        data_version: str = "v1",
        status: str = "candidate",
    ) -> FactorRecord:
        """转换为可直接写入 SQLite 注册表的 FactorRecord 对象。"""
        return FactorRecord(
            canon_expr=self.canon_expr,
            canon_hash=self.canon_hash,
            mode="cs",
            sign=self.sign,
            n_nodes=self.n_nodes,
            depth=self.depth,
            source=source,
            source_run=source_run,
            data_version=data_version,
            status=status,
            metrics=list(self.all_metrics),
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
    expr: str,
    horizons: Sequence[int] = DEFAULT_HORIZONS,
    splits: dict[str, tuple[str, str]] | None = None,
    min_names: int = 5,
    n_quantiles: int = 5,
    force_sign: int | None = None,
) -> FactorEvaluationResult:
    """对单个 DSL 因子表达式执行全时段多预测期数综合评价。

    Parameters
    ----------
    engine_or_panel:
        Rust 计算引擎或 Python 面板容器
    expr:
        待评估的因子 DSL 文本
    horizons:
        预测期数列表 (默认 [1, 3, 5, 10, 20])
    splits:
        时段切分字典，默认使用 train(2011-2018)/valid(2019-2021)/test(2022-2026.08) 及 full
    min_names:
        横截面最小有效标的数 (默认 5)
    n_quantiles:
        分位数分组数 (默认 5)
    force_sign:
        强制指定方向 (1 为正向, -1 为反向; None 则由训练集自动推断)
    """
    engine, dates, _ = _resolve_engine_and_panel(engine_or_panel)

    # 规范化与静态检查
    canon_info = engine.canonicalize(expr)
    canon_expr = canon_info["canon_expr"]
    canon_hash = canon_info["canon_hash"]
    n_nodes = canon_info["n_nodes"]
    depth = canon_info["depth"]

    # 求值全量因子矩阵 (T, N)
    factor_mat = np.asarray(engine.eval(canon_expr))
    T, N = factor_mat.shape

    # 确定时段切分
    eval_splits: dict[str, tuple[str, str]] = {}
    if splits is not None:
        eval_splits.update(splits)
    else:
        eval_splits.update(DEFAULT_SPLITS)

    # 增加全样本 full 切分
    if "full" not in eval_splits and len(dates) > 0:
        eval_splits["full"] = (str(dates[0]), str(dates[-1]))

    # 计算各切片对应的连续行索引
    split_indices: dict[str, np.ndarray] = {}
    for split_name, (start_d, end_d) in eval_splits.items():
        mask_idx = np.where((dates >= start_d) & (dates <= end_d))[0]
        if len(mask_idx) > 0:
            split_indices[split_name] = mask_idx

    # 若未包含任何有效日（例如小样本测试集日期不在 2011-2018 区间内），则以全部现有日期作为 train
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
        # 优先使用 y_5 标签判断方向，不存在则使用 y_1 或任一可用标签
        preferred_label = "y_5" if "y_5" in engine.label_names else engine.label_names[0]
        # 使用底层 rank_ic 计算该切片的 IC 均值
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

    # 遍历所有切片与预测窗口计算多维指标
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
                # 若无对应期数标签则尝试使用最接近的或首个标签
                lbl_name = (
                    "y_5"
                    if "y_5" in engine.label_names
                    else ("y_1" if "y_1" in engine.label_names else engine.label_names[0])
                )

            # 调用 Rust 内核一键综合评估
            met = engine.factor_metrics(
                f_sub,
                label_name=lbl_name,
                ret_name="r_exec_next" if "r_exec_next" in engine.label_names else None,
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
                turnover=float(met["turnover_val"]),
                ls_sharpe=float(met["ls_sharpe"]),
                ls_annual_ret=float(met["ls_annual_ret"]),
                ls_max_dd=float(met["ls_max_dd"]),
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
) -> list[int]:
    """批量评估并将 4 大经典种子因子原子化登记入库 factors.sqlite。"""
    seed_defs = get_seed_factors()
    registered_ids = []

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
        )
        fid = registry.register_factor(rec, run_id=run_id, upsert=True)
        registered_ids.append(fid)

    return registered_ids
