"""基准线性与启发式多因子合成模型 (ml/models/linear.py)。

严格遵循白皮书 11.3 节模型体系：
1. Equal Weight (等权基线)：因子 Rank Gauss 标准化后简单平均；
2. Rolling ICIR Weight (动态信息比率加权)：根据样本内各因子日频 RankIC 计算 ICIR 赋权；
3. Ridge 回归：带 L2 正则化的线性截面回归，自动网格搜索最优 alpha。
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge, RidgeCV


class EqualWeightModel:
    """等权基线模型 (简单平均)。"""

    def __init__(self) -> None:
        self.n_features_: int = 0
        self.weights_: np.ndarray | None = None

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray | None = None,
        date_indices: np.ndarray | None = None,
    ) -> EqualWeightModel:
        if X.ndim != 2:
            raise ValueError(f"X 必须为二维数组，实际形状 {X.shape}")
        self.n_features_ = X.shape[1]
        if self.n_features_ == 0:
            self.weights_ = np.empty((0,), dtype=np.float64)
        else:
            self.weights_ = np.full(self.n_features_, 1.0 / self.n_features_, dtype=np.float64)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.weights_ is None:
            raise RuntimeError("模型尚未拟合，请先调用 fit()")
        if X.shape[1] != self.n_features_:
            raise ValueError(f"特征数量不匹配: 期望 {self.n_features_}，实际输入 {X.shape[1]}")
        if len(X) == 0:
            return np.empty((0,), dtype=np.float64)
        return X @ self.weights_


class RollingICIRModel:
    """样本内/滚动 RankIC 信息比率 (ICIR) 加权模型。

    公式:
        w_f = max(0, ICIR_f) / sum(max(0, ICIR_j))
    若全为负或零，则平滑退化为等权。
    """

    def __init__(self, min_names_per_day: int = 5) -> None:
        self.min_names_per_day = min_names_per_day
        self.n_features_: int = 0
        self.weights_: np.ndarray | None = None
        self.icir_: np.ndarray | None = None

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        date_indices: np.ndarray | None = None,
    ) -> RollingICIRModel:
        if X.ndim != 2:
            raise ValueError(f"X 必须为二维数组，实际形状 {X.shape}")
        if len(X) != len(y):
            raise ValueError(f"样本数不匹配: len(X)={len(X)} vs len(y)={len(y)}")

        self.n_features_ = X.shape[1]
        if self.n_features_ == 0:
            self.weights_ = np.empty((0,), dtype=np.float64)
            self.icir_ = np.empty((0,), dtype=np.float64)
            return self

        if date_indices is None or len(np.unique(date_indices)) < 2:
            # 没有日期分组时退化为全样本截面相关性
            icirs = np.zeros(self.n_features_, dtype=np.float64)
            for k in range(self.n_features_):
                corr, _ = spearmanr(X[:, k], y)
                icirs[k] = float(corr) if not np.isnan(corr) else 0.0
        else:
            unique_dates = np.unique(date_indices)
            daily_ics = [[] for _ in range(self.n_features_)]

            for d in unique_dates:
                d_mask = date_indices == d
                if np.sum(d_mask) < self.min_names_per_day:
                    continue
                y_sub = y[d_mask]
                for k in range(self.n_features_):
                    x_sub = X[d_mask, k]
                    corr, _ = spearmanr(x_sub, y_sub)
                    if not np.isnan(corr):
                        daily_ics[k].append(float(corr))

            icirs = np.zeros(self.n_features_, dtype=np.float64)
            for k in range(self.n_features_):
                arr = np.array(daily_ics[k])
                if len(arr) >= 2:
                    std = np.std(arr, ddof=1)
                    if std > 1e-8:
                        icirs[k] = np.mean(arr) / std
                    else:
                        icirs[k] = 0.0
                elif len(arr) == 1:
                    icirs[k] = arr[0]
                else:
                    icirs[k] = 0.0

        self.icir_ = icirs
        pos_icir = np.maximum(0.0, icirs)
        s = np.sum(pos_icir)
        if s > 1e-12:
            self.weights_ = pos_icir / s
        else:
            # 全部小于等于 0 时退化为等权
            self.weights_ = np.full(self.n_features_, 1.0 / self.n_features_, dtype=np.float64)

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.weights_ is None:
            raise RuntimeError("模型尚未拟合，请先调用 fit()")
        if X.shape[1] != self.n_features_:
            raise ValueError(f"特征数量不匹配: 期望 {self.n_features_}，实际输入 {X.shape[1]}")
        if len(X) == 0:
            return np.empty((0,), dtype=np.float64)
        return X @ self.weights_


class RidgeModel:
    """自适应 L2 正则化岭回归模型。"""

    def __init__(
        self,
        alphas: Sequence[float] | None = None,
        fit_intercept: bool = True,
    ) -> None:
        self.alphas = (
            np.logspace(-2, 4, 20) if alphas is None else np.asarray(alphas, dtype=np.float64)
        )
        self.fit_intercept = fit_intercept
        self.model_: Ridge | RidgeCV | None = None
        self.n_features_: int = 0
        self.best_alpha_: float = 1.0

    def _select_alpha_on_val(
        self, X: np.ndarray, y: np.ndarray, X_val: np.ndarray, y_val: np.ndarray
    ) -> tuple[Ridge, float]:
        best_mse = float("inf")
        best_model: Ridge | None = None
        best_alpha = float(self.alphas[0])
        for alpha in self.alphas:
            cand = Ridge(alpha=float(alpha), fit_intercept=self.fit_intercept)
            cand.fit(X, y)
            mse = float(np.mean((cand.predict(X_val) - y_val) ** 2))
            if mse < best_mse:
                best_mse = mse
                best_model = cand
                best_alpha = float(alpha)
        assert best_model is not None
        return best_model, best_alpha

    def _select_alpha_date_blocked(
        self, X: np.ndarray, y: np.ndarray, date_indices: np.ndarray
    ) -> tuple[Ridge, float]:
        unique_dates = np.unique(date_indices)
        n_blocks = min(5, len(unique_dates))
        blocks = np.array_split(unique_dates, n_blocks)
        best_mse = float("inf")
        best_alpha = float(self.alphas[0])

        for alpha in self.alphas:
            sq_err = 0.0
            count = 0
            for val_dates in blocks:
                val_mask = np.isin(date_indices, val_dates)
                train_mask = ~val_mask
                if not np.any(train_mask) or not np.any(val_mask):
                    continue
                cand = Ridge(alpha=float(alpha), fit_intercept=self.fit_intercept)
                cand.fit(X[train_mask], y[train_mask])
                diff = cand.predict(X[val_mask]) - y[val_mask]
                sq_err += float(np.sum(diff * diff))
                count += int(np.sum(val_mask))
            mse = sq_err / count if count > 0 else float("inf")
            if mse < best_mse:
                best_mse = mse
                best_alpha = float(alpha)

        final_model = Ridge(alpha=best_alpha, fit_intercept=self.fit_intercept)
        final_model.fit(X, y)
        return final_model, best_alpha

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        date_indices: np.ndarray | None = None,
        X_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
    ) -> RidgeModel:
        if X.ndim != 2:
            raise ValueError(f"X 必须为二维数组，实际形状 {X.shape}")
        if len(X) != len(y):
            raise ValueError(f"样本数不匹配: len(X)={len(X)} vs len(y)={len(y)}")

        self.n_features_ = X.shape[1]
        if self.n_features_ == 0 or len(X) == 0:
            return self

        if X_val is not None and y_val is not None and len(X_val) > 0:
            # 优先使用时间隔离的验证集 (X_val, y_val) 寻优正则化强度，防止截面 LOOCV 低估 alpha
            self.model_, self.best_alpha_ = self._select_alpha_on_val(X, y, X_val, y_val)
        elif date_indices is not None and len(np.unique(date_indices)) >= 3:
            # 未传入显式验证集但提供日期索引时，按时间日期分块交叉验证避免同日截面信息泄漏
            self.model_, self.best_alpha_ = self._select_alpha_date_blocked(X, y, date_indices)
        else:
            rcv = RidgeCV(
                alphas=self.alphas,
                fit_intercept=self.fit_intercept,
            )
            rcv.fit(X, y)
            self.model_ = rcv
            self.best_alpha_ = float(rcv.alpha_)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.model_ is None:
            raise RuntimeError("模型尚未拟合，请先调用 fit()")
        if X.shape[1] != self.n_features_:
            raise ValueError(f"特征数量不匹配: 期望 {self.n_features_}，实际输入 {X.shape[1]}")
        if len(X) == 0:
            return np.empty((0,), dtype=np.float64)
        return self.model_.predict(X)

    @property
    def coef_(self) -> np.ndarray:
        if self.model_ is None:
            raise RuntimeError("模型尚未拟合")
        return self.model_.coef_

    @property
    def intercept_(self) -> float:
        if self.model_ is None:
            raise RuntimeError("模型尚未拟合")
        return float(self.model_.intercept_)
