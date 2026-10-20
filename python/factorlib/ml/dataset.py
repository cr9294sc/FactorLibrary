"""机器学习特征工程预处理与数据集构建器 (ml/dataset.py)。

严格遵循白皮书 11.1 节标准预处理流水线：
1. 方向对齐：因子乘以符号标量 sign，使所有特征与标签正向对齐；
2. 截面极值中位数截断 (MAD Winsorization)：
   超出 [med - 5 * 1.4826 MAD, med + 5 * 1.4826 MAD] 的离群点进行截断；
3. 截面排序高斯化 (Rank Gauss Transform)：
   x_tilde = ndtri((rank - 0.5) / N_t)，消除量纲并映射为标准正态分布；
4. 构建纯净二维表格数据集 TabularDataset，支持向 LightGBM / Ridge / 组合引擎无缝传输。
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from scipy.special import ndtri
from scipy.stats import rankdata

from factorlib.core.panel import Panel
from factorlib.core.registry import FactorRecord
from factorlib.ml.splits import WalkForwardFold


def mad_winsorize(
    mat: np.ndarray,
    mask: np.ndarray | None = None,
    n_mad: float = 5.0,
) -> np.ndarray:
    """对 (T, N) 宽矩阵按行 (逐交易日) 执行截面中位数极值截断 (MAD Winsorization)。

    Parameters
    ----------
    mat:
        输入二维矩阵，形状 (T, N)
    mask:
        可交易标的掩码矩阵 (T, N)，若提供则仅对有效标的参与统计
    n_mad:
        MAD 倍数阈值 (默认 5.0)

    Returns
    -------
    np.ndarray
        截断后的 (T, N) Fortran 矩阵，非有效位置保留 NaN
    """
    out = np.array(mat, dtype=np.float64, copy=True, order="F")
    T, N = out.shape

    for t in range(T):
        row = out[t, :]
        if mask is not None:
            valid_mask = mask[t, :] & (~np.isnan(row))
        else:
            valid_mask = ~np.isnan(row)

        if not np.any(valid_mask):
            continue

        valid_vals = row[valid_mask]
        if len(valid_vals) < 3:
            continue

        med = np.median(valid_vals)
        mad = np.median(np.abs(valid_vals - med))

        if mad > 1e-12:
            bound = n_mad * 1.4826 * mad
            lower = med - bound
            upper = med + bound
            out[t, valid_mask] = np.clip(valid_vals, lower, upper)

    return np.asfortranarray(out)


def rank_gauss_transform(
    mat: np.ndarray,
    mask: np.ndarray | None = None,
) -> np.ndarray:
    """对 (T, N) 宽矩阵按行 (逐交易日) 执行截面排序高斯化 (Rank Gauss Transform)。

    公式:
        tilde_x_{i,t} = Phi^{-1}((rank(x_{i,t}) - 0.5) / N_t)

    Parameters
    ----------
    mat:
        输入二维矩阵，形状 (T, N)
    mask:
        可交易标的掩码矩阵 (T, N)

    Returns
    -------
    np.ndarray
        高斯化后的 (T, N) Fortran 矩阵，非有效标的保持 NaN
    """
    out = np.full(mat.shape, np.nan, dtype=np.float64, order="F")
    T, N = mat.shape

    for t in range(T):
        row = mat[t, :]
        if mask is not None:
            valid_mask = mask[t, :] & (~np.isnan(row))
        else:
            valid_mask = ~np.isnan(row)

        valid_idx = np.where(valid_mask)[0]
        n_valid = len(valid_idx)
        if n_valid == 0:
            continue
        if n_valid == 1:
            out[t, valid_idx[0]] = 0.0
            continue

        valid_vals = row[valid_idx]
        ranks = rankdata(valid_vals, method="average")
        quantiles = (ranks - 0.5) / n_valid
        out[t, valid_idx] = ndtri(quantiles)

    return np.asfortranarray(out)


def align_directions(
    factor_matrices: dict[str, np.ndarray],
    factor_signs: dict[str, int],
) -> dict[str, np.ndarray]:
    """根据注册符号将所有因子矩阵方向对齐，使其与预测标签预期保持正相关。"""
    aligned: dict[str, np.ndarray] = {}
    for name, mat in factor_matrices.items():
        sign = factor_signs.get(name, 1)
        if sign == -1:
            aligned[name] = np.asfortranarray(-mat)
        else:
            aligned[name] = np.asfortranarray(mat)
    return aligned


def preprocess_factors(
    factor_matrices: dict[str, np.ndarray],
    factor_signs: dict[str, int] | None = None,
    mask: np.ndarray | None = None,
    n_mad: float = 5.0,
) -> dict[str, np.ndarray]:
    """执行完整的因子预处理流：方向对齐 -> MAD 截断 -> Rank Gauss 高斯化。"""
    if factor_signs is not None:
        aligned = align_directions(factor_matrices, factor_signs)
    else:
        aligned = factor_matrices

    preprocessed: dict[str, np.ndarray] = {}
    for name, mat in aligned.items():
        mat_winsor = mad_winsorize(mat, mask=mask, n_mad=n_mad)
        mat_gauss = rank_gauss_transform(mat_winsor, mask=mask)
        preprocessed[name] = mat_gauss
    return preprocessed


@dataclass(frozen=True)
class TabularDataset:
    """平铺格式机器学习训练/评估数据集。"""

    X: np.ndarray  # 形状 (M, K)
    y: np.ndarray  # 形状 (M,)
    date_indices: np.ndarray  # 对应在面板 dates 中的行索引 (M,)
    symbol_indices: np.ndarray  # 对应在面板 symbols 中的列索引 (M,)
    groups: np.ndarray  # 每日有效样本数列表 (按天聚合, 用于 ranking 模型)
    feature_names: list[str]

    @property
    def n_samples(self) -> int:
        return len(self.y)

    @property
    def n_features(self) -> int:
        return self.X.shape[1] if self.X.ndim == 2 else 0


class MLDatasetBuilder:
    """多因子机器学习数据集构建与格式编排器。"""

    def __init__(
        self,
        panel: Panel,
        factor_matrices: dict[str, np.ndarray],
        factor_signs: dict[str, int] | None = None,
        target_label: str = "y_5",
        target_matrix: np.ndarray | None = None,
        preprocess: bool = True,
        rank_gauss_target: bool = False,
    ) -> None:
        self.panel = panel
        self.target_label = target_label
        self.rank_gauss_target = rank_gauss_target

        # 准备目标标签矩阵 (T, N)
        if target_matrix is not None:
            self.y_matrix = np.asfortranarray(target_matrix)
        else:
            self.y_matrix = np.asfortranarray(panel.get_label(target_label))

        if self.rank_gauss_target:
            self.y_matrix = rank_gauss_transform(self.y_matrix, mask=panel.mask)

        # 预处理因子特征
        if preprocess:
            self.features = preprocess_factors(
                factor_matrices, factor_signs=factor_signs, mask=panel.mask
            )
        else:
            self.features = {k: np.asfortranarray(v) for k, v in factor_matrices.items()}

        self.feature_names = sorted(self.features.keys())

    @classmethod
    def from_registry(
        cls,
        panel: Panel,
        registry_factors: Sequence[FactorRecord],
        target_label: str = "y_5",
        preprocess: bool = True,
        rank_gauss_target: bool = False,
    ) -> MLDatasetBuilder:
        """从 Panel 和 FactorRecord 列表直接求值并构建。"""
        engine = panel.to_rust_engine()
        factor_matrices: dict[str, np.ndarray] = {}
        factor_signs: dict[str, int] = {}

        for rec in registry_factors:
            name = rec.canon_expr
            mat = np.asarray(engine.eval(rec.canon_expr))
            factor_matrices[name] = mat
            factor_signs[name] = rec.sign

        return cls(
            panel=panel,
            factor_matrices=factor_matrices,
            factor_signs=factor_signs,
            target_label=target_label,
            preprocess=preprocess,
            rank_gauss_target=rank_gauss_target,
        )

    def build_slice(self, row_indices: np.ndarray) -> TabularDataset:
        """根据指定的行索引 (交易日切片) 提取并拼装样本。"""
        if len(row_indices) == 0:
            return TabularDataset(
                X=np.empty((0, len(self.feature_names)), dtype=np.float64),
                y=np.empty((0,), dtype=np.float64),
                date_indices=np.empty((0,), dtype=np.int64),
                symbol_indices=np.empty((0,), dtype=np.int64),
                groups=np.empty((0,), dtype=np.int32),
                feature_names=self.feature_names,
            )

        K = len(self.feature_names)

        # 堆叠 K 个特征成为 (T_slice, N, K)
        # 为提升效率，逐日收集样本
        x_rows: list[np.ndarray] = []
        y_rows: list[float] = []
        d_idx: list[int] = []
        s_idx: list[int] = []
        groups: list[int] = []

        feat_arrays = [self.features[fname] for fname in self.feature_names]

        for t in row_indices:
            day_mask = self.panel.mask[t, :]
            y_t = self.y_matrix[t, :]
            # 有效样本条件：在品种池内、y 不为 NaN、且特征无 NaN
            valid = day_mask & (~np.isnan(y_t))
            for f_arr in feat_arrays:
                valid &= ~np.isnan(f_arr[t, :])

            valid_symbols = np.where(valid)[0]
            n_valid = len(valid_symbols)
            if n_valid == 0:
                continue

            groups.append(n_valid)
            for s in valid_symbols:
                x_vals = [f_arr[t, s] for f_arr in feat_arrays]
                x_rows.append(x_vals)
                y_rows.append(y_t[s])
                d_idx.append(t)
                s_idx.append(s)

        if len(x_rows) == 0:
            X = np.empty((0, K), dtype=np.float64)
            y = np.empty((0,), dtype=np.float64)
        else:
            X = np.array(x_rows, dtype=np.float64)
            y = np.array(y_rows, dtype=np.float64)

        return TabularDataset(
            X=X,
            y=y,
            date_indices=np.array(d_idx, dtype=np.int64),
            symbol_indices=np.array(s_idx, dtype=np.int64),
            groups=np.array(groups, dtype=np.int32),
            feature_names=self.feature_names,
        )

    def get_fold_datasets(
        self, fold: WalkForwardFold
    ) -> tuple[TabularDataset, TabularDataset, TabularDataset]:
        """获取某一折的训练集、验证集与测试集。"""
        train_ds = self.build_slice(fold.train_indices)
        val_ds = self.build_slice(fold.val_indices)
        test_ds = self.build_slice(fold.test_indices)
        return train_ds, val_ds, test_ds

    def reconstruct_panel(
        self,
        predictions: np.ndarray,
        date_indices: np.ndarray,
        symbol_indices: np.ndarray,
        out_matrix: np.ndarray | None = None,
    ) -> np.ndarray:
        """将一维平铺预测结果回填至 (T, N) 宽矩阵。"""
        T, N = self.panel.shape
        if out_matrix is None:
            mat = np.full((T, N), np.nan, dtype=np.float64, order="F")
        else:
            mat = out_matrix

        mat[date_indices, symbol_indices] = predictions
        return mat
