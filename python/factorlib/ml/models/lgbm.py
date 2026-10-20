"""LightGBM 截面排序回归模型封装 (ml/models/lgbm.py)。

严格遵循白皮书 11.3 节架构与超参约束：
- 防过拟合硬约束：max_depth=4, num_leaves=15, feature_fraction=0.7, bagging_fraction=0.8；
- 目标函数：支持连续回归 (regression_l2) 与截面排序 (lambdarank)；
- 早停机制：在验证集上严格执行 early_stopping_rounds=50，保留最优泛化权重；
- 提供特征增益 (Gain) 与分裂数 (Split) 多维重要性输出。
"""

from __future__ import annotations

from typing import Any

import lightgbm as lgb
import numpy as np

from factorlib.core.config import LightGBMConfig


class LightGBMModel:
    """LightGBM 梯度提升决策树多因子合成模型。"""

    def __init__(
        self,
        objective: str = "regression_l2",
        metric: str = "rmse",
        boosting_type: str = "gbdt",
        max_depth: int = 4,
        num_leaves: int = 15,
        learning_rate: float = 0.03,
        n_estimators: int = 1000,
        min_child_samples: int = 200,
        feature_fraction: float = 0.70,
        bagging_fraction: float = 0.80,
        bagging_freq: int = 1,
        early_stopping_rounds: int = 50,
        seed: int = 42,
        feature_names: list[str] | None = None,
    ) -> None:
        self.objective = objective
        self.metric = metric
        self.boosting_type = boosting_type
        self.max_depth = max_depth
        self.num_leaves = num_leaves
        self.learning_rate = learning_rate
        self.n_estimators = n_estimators
        self.min_child_samples = min_child_samples
        self.feature_fraction = feature_fraction
        self.bagging_fraction = bagging_fraction
        self.bagging_freq = bagging_freq
        self.early_stopping_rounds = early_stopping_rounds
        self.seed = seed
        self.feature_names = feature_names

        self.booster_: lgb.Booster | None = None
        self.best_iteration_: int = 0
        self.n_features_: int = 0

    @classmethod
    def from_config(
        cls, cfg: LightGBMConfig, feature_names: list[str] | None = None
    ) -> LightGBMModel:
        """从 LightGBMConfig 配置实例初始化。"""
        return cls(
            objective=cfg.objective,
            metric=cfg.metric,
            boosting_type=cfg.boosting_type,
            max_depth=cfg.max_depth,
            num_leaves=cfg.num_leaves,
            learning_rate=cfg.learning_rate,
            n_estimators=cfg.n_estimators,
            min_child_samples=cfg.min_child_samples,
            feature_fraction=cfg.feature_fraction,
            bagging_fraction=cfg.bagging_fraction,
            bagging_freq=cfg.bagging_freq,
            early_stopping_rounds=cfg.early_stopping_rounds,
            seed=cfg.seed,
            feature_names=feature_names,
        )

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        date_indices: np.ndarray | None = None,
        X_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
        groups: np.ndarray | None = None,
        groups_val: np.ndarray | None = None,
    ) -> LightGBMModel:
        """在训练集上拟合，并在验证集上执行早停评估。"""
        if X.ndim != 2:
            raise ValueError(f"X 必须为二维数组，实际形状 {X.shape}")
        if len(X) != len(y):
            raise ValueError(f"样本数不匹配: len(X)={len(X)} vs len(y)={len(y)}")

        self.n_features_ = X.shape[1]
        if self.n_features_ == 0 or len(X) == 0:
            return self

        # 根据样本量动态适配最小叶子样本数 (小测试集或少标的时防止报错)
        effective_min_child = max(1, min(self.min_child_samples, len(X) // 3))

        # 构建原生 LightGBM 参数字典
        lgb_params: dict[str, Any] = {
            "objective": self.objective,
            "metric": self.metric,
            "boosting_type": self.boosting_type,
            "max_depth": self.max_depth,
            "num_leaves": self.num_leaves,
            "learning_rate": self.learning_rate,
            "min_child_samples": effective_min_child,
            "feature_fraction": self.feature_fraction,
            "bagging_fraction": self.bagging_fraction,
            "bagging_freq": self.bagging_freq,
            "seed": self.seed,
            "verbose": -1,
        }

        # 构建 Dataset
        train_data = lgb.Dataset(
            X,
            label=y,
            group=groups if self.objective == "lambdarank" else None,
            feature_name=self.feature_names,
            free_raw_data=False,
        )

        valid_sets = []
        callbacks = []

        if X_val is not None and y_val is not None and len(X_val) > 0:
            val_data = lgb.Dataset(
                X_val,
                label=y_val,
                group=groups_val if self.objective == "lambdarank" else None,
                reference=train_data,
                feature_name=self.feature_names,
                free_raw_data=False,
            )
            valid_sets.append(val_data)
            if self.early_stopping_rounds > 0:
                callbacks.append(
                    lgb.early_stopping(
                        stopping_rounds=self.early_stopping_rounds,
                        verbose=False,
                    )
                )

        self.booster_ = lgb.train(
            lgb_params,
            train_data,
            num_boost_round=self.n_estimators,
            valid_sets=valid_sets if len(valid_sets) > 0 else None,
            callbacks=callbacks if len(callbacks) > 0 else None,
        )

        self.best_iteration_ = self.booster_.best_iteration
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """执行模型样本外预测。"""
        if self.booster_ is None:
            raise RuntimeError("模型尚未拟合，请先调用 fit()")
        if X.shape[1] != self.n_features_:
            raise ValueError(f"特征数量不匹配: 期望 {self.n_features_}，实际输入 {X.shape[1]}")
        if len(X) == 0:
            return np.empty((0,), dtype=np.float64)

        return self.booster_.predict(X, num_iteration=self.best_iteration_)

    def get_feature_importances(self, importance_type: str = "gain") -> dict[str, float]:
        """获取特征重要性映射字典。"""
        if self.booster_ is None:
            raise RuntimeError("模型尚未拟合")

        raw_imp = self.booster_.feature_importance(importance_type=importance_type)
        if self.feature_names and len(self.feature_names) == len(raw_imp):
            return {name: float(val) for name, val in zip(self.feature_names, raw_imp, strict=True)}
        return {f"feat_{i}": float(v) for i, v in enumerate(raw_imp)}
