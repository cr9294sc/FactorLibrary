"""Purged Walk-Forward 多因子机器学习合成流水线 (ml/pipeline.py)。

严格遵循白皮书第 11 章架构：
- 加载并执行因子库特征预处理 (方向对齐、MAD Winsorization、Rank Gauss)；
- 使用 Purged & Embargoed 扩展/滚动切分器按年推进，杜绝任何未来数据穿越；
- 并行拟合 Equal Weight、Rolling ICIR、Ridge 与 LightGBM；
- 记录逐年样本外预测矩阵与特征增益，统计长周期 RankIC / ICIR / 胜率对账；
- 导出样本外综合预测面板 `pred_panel.parquet`。
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
from scipy.stats import spearmanr

from factorlib.core.config import MLConfig
from factorlib.core.panel import Panel
from factorlib.core.registry import FactorRecord, FactorRegistry
from factorlib.ml.dataset import MLDatasetBuilder
from factorlib.ml.models import (
    EqualWeightModel,
    LightGBMModel,
    RidgeModel,
    RollingICIRModel,
)
from factorlib.ml.splits import PurgedWalkForwardSplitter, WalkForwardFold


@dataclass
class WalkForwardResult:
    """Walk-Forward 机器学习多因子合成运行结果产物。"""

    predictions: dict[str, np.ndarray]  # 各模型对应的 (T, N) 样本外预测评分矩阵
    metrics: dict[str, dict[str, float]]  # 各模型在全测试时段内的汇总指标 (RankIC, ICIR 等)
    fold_metrics: list[dict[str, Any]]  # 逐年逐折样本外指标追踪明细
    feature_importances: dict[int, dict[str, float]]  # 逐折 LightGBM 特征重要性
    feature_names: list[str]
    dates: np.ndarray
    symbols: list[str]
    mask: np.ndarray
    target_label: str
    target_matrix: np.ndarray

    def to_parquet(self, out_path: str | Path) -> None:
        """向量化保存样本外多模型预测宽表为 parquet 文件。"""
        p = Path(out_path)
        p.parent.mkdir(parents=True, exist_ok=True)

        d_idx, s_idx = np.where(self.mask)
        data_dict: dict[str, Any] = {
            "trading_date": [str(self.dates[t]) for t in d_idx],
            "symbol": [self.symbols[s] for s in s_idx],
        }

        # 写入真实目标标签
        data_dict[self.target_label] = self.target_matrix[d_idx, s_idx]

        # 写入各模型预测列
        for m_name, mat in self.predictions.items():
            data_dict[f"pred_{m_name}"] = mat[d_idx, s_idx]

        df = pl.DataFrame(data_dict)
        df.write_parquet(p)

    def summary_table(self) -> str:
        """生成 Markdown 格式的多模型全样本外绩效对比表格。"""
        lines = [
            "| 模型名称 (Model) | 样本外 RankIC | RankIC Std | ICIR | t-stat | 胜率 (Win Rate) | 评估天数 |",
            "| :--- | :--- | :--- | :--- | :--- | :--- | :--- |",
        ]
        for m_name, m_stats in self.metrics.items():
            lines.append(
                f"| `{m_name}` | {m_stats.get('ic_mean', 0.0):.4f} | "
                f"{m_stats.get('ic_std', 0.0):.4f} | "
                f"{m_stats.get('ic_ir', 0.0):.4f} | "
                f"{m_stats.get('ic_t', 0.0):.2f} | "
                f"{m_stats.get('ic_win_rate', 0.0) * 100:.1f}% | "
                f"{int(m_stats.get('valid_days', 0))} |"
            )
        return "\n".join(lines)


def _calc_rank_ic(
    pred: np.ndarray,
    target: np.ndarray,
    mask: np.ndarray,
    min_names: int = 5,
) -> dict[str, float]:
    """计算单个模型预测矩阵 (T, N) 在指定时段内的截面 RankIC 及统计量。"""
    T = pred.shape[0]
    daily_ics: list[float] = []

    for t in range(T):
        valid = mask[t, :] & (~np.isnan(pred[t, :])) & (~np.isnan(target[t, :]))
        if np.sum(valid) < min_names:
            continue
        corr, _ = spearmanr(pred[t, valid], target[t, valid])
        if not np.isnan(corr):
            daily_ics.append(float(corr))

    arr = np.array(daily_ics)
    n_days = len(arr)
    if n_days == 0:
        return {
            "ic_mean": 0.0,
            "ic_std": 0.0,
            "ic_ir": 0.0,
            "ic_t": 0.0,
            "ic_win_rate": 0.0,
            "valid_days": 0,
        }

    mean = float(np.mean(arr))
    std = float(np.std(arr, ddof=1)) if n_days > 1 else 0.0
    ir = mean / std if std > 1e-8 else 0.0
    t_stat = ir * np.sqrt(n_days)
    win_rate = float(np.sum(arr > 0)) / n_days

    return {
        "ic_mean": mean,
        "ic_std": std,
        "ic_ir": ir,
        "ic_t": t_stat,
        "ic_win_rate": win_rate,
        "valid_days": n_days,
    }


class WalkForwardPipeline:
    """Walk-Forward 机器学习多因子合成流水线。"""

    def __init__(
        self,
        panel: Panel,
        config: MLConfig,
        factor_matrices: dict[str, np.ndarray],
        factor_signs: dict[str, int] | None = None,
        feature_names: list[str] | None = None,
    ) -> None:
        self.panel = panel
        self.config = config
        self.factor_matrices = factor_matrices
        self.factor_signs = factor_signs or {}
        self.feature_names = feature_names or sorted(factor_matrices.keys())

    @classmethod
    def from_registry(
        cls,
        panel: Panel,
        config: MLConfig,
        registry: FactorRegistry,
    ) -> WalkForwardPipeline:
        """从 SQLite 因子库提取指定状态的因子并构建流水线。"""
        # 按照配置过滤因子 (accepted, active 等)
        all_factors = registry.list_factors()
        target_factors: list[FactorRecord] = [
            f for f in all_factors if f.status in config.features_status
        ]

        if not target_factors:
            # 若指定状态无因子，退回尝试 candidate 或全部因子
            target_factors = all_factors
            if not target_factors:
                raise ValueError("因子注册库中没有任何因子，无法执行多因子合成")

        engine = panel.to_rust_engine()
        factor_matrices: dict[str, np.ndarray] = {}
        factor_signs: dict[str, int] = {}

        for f in target_factors:
            name = f.canon_expr
            factor_matrices[name] = np.asarray(engine.eval(f.canon_expr))
            factor_signs[name] = f.sign

        return cls(
            panel=panel,
            config=config,
            factor_matrices=factor_matrices,
            factor_signs=factor_signs,
            feature_names=sorted(factor_matrices.keys()),
        )

    def run(self, strict_min_years: bool = True) -> WalkForwardResult:
        """执行全流程 Purged Walk-Forward 滚动重训与合成。"""
        # 1. 因子特征预处理
        ds_builder = MLDatasetBuilder(
            panel=self.panel,
            factor_matrices=self.factor_matrices,
            factor_signs=self.factor_signs,
            target_label=self.config.target_label,
            preprocess=True,
            rank_gauss_target=False,
        )

        target_mat = np.asfortranarray(self.panel.get_label(self.config.target_label))
        T, N = self.panel.shape

        # 2. 初始化切分器并生成时序折叠 (P0-05: 强制校验 purge_window >= 标签预测周期 h)
        lbl_horizon = 1
        if "_" in self.config.target_label:
            tail = self.config.target_label.rsplit("_", 1)[-1].rstrip("dD")
            if tail.isdigit():
                lbl_horizon = max(1, int(tail))
        effective_purge = max(self.config.walk_forward.purge_window, lbl_horizon)
        wf_cfg = replace(self.config.walk_forward, purge_window=effective_purge)

        splitter = PurgedWalkForwardSplitter.from_config(wf_cfg, strict_min_years=strict_min_years)
        folds: list[WalkForwardFold] = splitter.split(self.panel.dates)
        if not folds:
            raise RuntimeError(
                f"在面板日期范围内无法生成任何 Walk-Forward 折叠 "
                f"({self.config.walk_forward.start_predict_year} ~ "
                f"{self.config.walk_forward.end_predict_year})。请检查年份配置或设置 strict_min_years=False。"
            )

        # 初始化输出矩阵字典 (各模型全样本外预测矩阵)
        pred_matrices: dict[str, np.ndarray] = {
            m: np.full((T, N), np.nan, dtype=np.float64, order="F") for m in self.config.models
        }

        fold_records: list[dict[str, Any]] = []
        feature_importances: dict[int, dict[str, float]] = {}

        # 3. 逐折推进演化与训练
        for fold in folds:
            pred_year = fold.predict_year
            train_ds, val_ds, test_ds = ds_builder.get_fold_datasets(fold)

            if test_ds.n_samples == 0:
                continue

            fold_record: dict[str, Any] = {
                "fold_idx": fold.fold_idx,
                "predict_year": pred_year,
                "n_train": train_ds.n_samples,
                "n_val": val_ds.n_samples,
                "n_test": test_ds.n_samples,
                "models": {},
            }

            # 模型 1: Equal Weight
            if "equal_weight" in self.config.models:
                ew = EqualWeightModel().fit(train_ds.X)
                pred_ew = ew.predict(test_ds.X)
                ds_builder.reconstruct_panel(
                    pred_ew,
                    test_ds.date_indices,
                    test_ds.symbol_indices,
                    out_matrix=pred_matrices["equal_weight"],
                )
                fold_record["models"]["equal_weight"] = float(spearmanr(pred_ew, test_ds.y)[0])

            # 模型 2: Rolling ICIR
            if "rolling_icir" in self.config.models:
                icir = RollingICIRModel().fit(
                    train_ds.X, train_ds.y, date_indices=train_ds.date_indices
                )
                pred_icir = icir.predict(test_ds.X)
                ds_builder.reconstruct_panel(
                    pred_icir,
                    test_ds.date_indices,
                    test_ds.symbol_indices,
                    out_matrix=pred_matrices["rolling_icir"],
                )
                fold_record["models"]["rolling_icir"] = float(spearmanr(pred_icir, test_ds.y)[0])

            # 模型 3: Ridge
            if "ridge" in self.config.models:
                ridge = RidgeModel().fit(
                    train_ds.X,
                    train_ds.y,
                    date_indices=train_ds.date_indices,
                    X_val=val_ds.X,
                    y_val=val_ds.y,
                )
                pred_ridge = ridge.predict(test_ds.X)
                ds_builder.reconstruct_panel(
                    pred_ridge,
                    test_ds.date_indices,
                    test_ds.symbol_indices,
                    out_matrix=pred_matrices["ridge"],
                )
                fold_record["models"]["ridge"] = float(spearmanr(pred_ridge, test_ds.y)[0])

            # 模型 4: LightGBM
            if "lightgbm" in self.config.models:
                lgbm = LightGBMModel.from_config(
                    self.config.lightgbm, feature_names=ds_builder.feature_names
                )
                lgbm.fit(
                    train_ds.X,
                    train_ds.y,
                    date_indices=train_ds.date_indices,
                    X_val=val_ds.X,
                    y_val=val_ds.y,
                    groups=train_ds.groups,
                    groups_val=val_ds.groups,
                )
                pred_lgb = lgbm.predict(test_ds.X)
                ds_builder.reconstruct_panel(
                    pred_lgb,
                    test_ds.date_indices,
                    test_ds.symbol_indices,
                    out_matrix=pred_matrices["lightgbm"],
                )
                fold_record["models"]["lightgbm"] = float(spearmanr(pred_lgb, test_ds.y)[0])
                feature_importances[pred_year] = lgbm.get_feature_importances()

            fold_records.append(fold_record)

        # 4. 汇总全测试集时段的截面 RankIC 指标
        all_test_dates = np.concatenate([f.test_indices for f in folds])
        all_test_mask = np.zeros(self.panel.shape, dtype=bool)
        all_test_mask[all_test_dates, :] = self.panel.mask[all_test_dates, :]

        metrics_summary: dict[str, dict[str, float]] = {}
        for m in self.config.models:
            m_metrics = _calc_rank_ic(
                pred=pred_matrices[m],
                target=target_mat,
                mask=all_test_mask,
            )
            metrics_summary[m] = m_metrics

        return WalkForwardResult(
            predictions=pred_matrices,
            metrics=metrics_summary,
            fold_metrics=fold_records,
            feature_importances=feature_importances,
            feature_names=ds_builder.feature_names,
            dates=self.panel.dates,
            symbols=self.panel.symbols,
            mask=self.panel.mask,
            target_label=self.config.target_label,
            target_matrix=target_mat,
        )
