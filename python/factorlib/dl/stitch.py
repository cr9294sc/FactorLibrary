"""Walk-Forward OOS 预测拼接与多种子截面融合器 (dl/stitch.py)。

严格遵循架构设计 §5.3 & M8.2 规范：
- 每个 Fold 内多种子预测每日按有效标的截面标准化 (CS Z-Score) 后等权融合；
- 逐折将纯测试预测窗 `test_indices` 无重叠拼接回全历史 `(T, N)` Fortran 矩阵；
- Selection Period 与 Lockbox Period 分离记录，绝不允许重叠写穿。
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def cs_zscore_matrix(
    mat: np.ndarray,
    valid_mask: np.ndarray,
    clip: float = 4.0,
    min_names: int = 3,
) -> np.ndarray:
    """每日横截面均值-标准差标准化（仅使用当日有效截面，零未来泄漏）。"""
    arr = np.asarray(mat, dtype=np.float64)
    mask = np.asarray(valid_mask, dtype=bool)
    if arr.shape != mask.shape:
        raise ValueError(f"mat 形状 {arr.shape} 与 valid_mask 形状 {mask.shape} 不一致")

    t_len, n_cols = arr.shape
    out = np.full((t_len, n_cols), np.nan, dtype=np.float64, order="F")
    for t in range(t_len):
        vm = mask[t, :] & np.isfinite(arr[t, :])
        idx = np.where(vm)[0]
        if len(idx) < min_names:
            continue
        vals = arr[t, idx]
        mu = float(np.mean(vals))
        sd = float(np.std(vals))
        if sd > 1e-12:
            z = np.clip((vals - mu) / sd, -clip, clip)
        else:
            z = np.zeros_like(vals)
        out[t, idx] = z
    return out


def bag_seed_predictions(
    seed_matrices: Sequence[np.ndarray],
    valid_mask: np.ndarray,
    *,
    clip: float = 4.0,
    min_names: int = 3,
) -> np.ndarray:
    """对同折多种子预测执行每日截面 Z-Score 后等权集成（Seed Bagging）。"""
    if not seed_matrices:
        raise ValueError("seed_matrices 不能为空")

    normed = [
        cs_zscore_matrix(mat, valid_mask, clip=clip, min_names=min_names) for mat in seed_matrices
    ]
    stacked = np.stack(normed, axis=0)  # (S, T, N)
    finite_cnt = np.sum(np.isfinite(stacked), axis=0)
    summed = np.nansum(stacked, axis=0)
    fused = np.where(finite_cnt > 0, summed / np.maximum(finite_cnt, 1), np.nan)
    return cs_zscore_matrix(fused, valid_mask, clip=clip, min_names=min_names)


def stitch_oos_folds(
    total_days: int,
    n_symbols: int,
    fold_predictions: Sequence[tuple[np.ndarray, np.ndarray]],
) -> np.ndarray:
    """将各折 `(test_indices, pred_submatrix)` 无重叠拼接为 `(T, N)` 全周期纯 OOS 矩阵。

    Parameters
    ----------
    total_days:
        面板总交易日数 `T`。
    n_symbols:
        面板标的数量 `N`。
    fold_predictions:
        元素为 `(test_indices, pred_submatrix)` 的序列；`pred_submatrix` 形状 `(len(test_indices), N)`。
    """
    oos = np.full((total_days, n_symbols), np.nan, dtype=np.float64, order="F")
    seen_rows: set[int] = set()

    for fold_no, (test_idx, sub_mat) in enumerate(fold_predictions):
        idx = np.asarray(test_idx, dtype=np.int64)
        mat = np.asarray(sub_mat, dtype=np.float64)
        if mat.shape != (len(idx), n_symbols):
            raise ValueError(
                f"第 {fold_no} 折预测矩阵形状 {mat.shape} 与 test_indices 长度 {len(idx)} / N={n_symbols} 不匹配"
            )
        for row_i in idx.tolist():
            if row_i < 0 or row_i >= total_days:
                raise IndexError(f"test index {row_i} 越界 [0, {total_days})")
            if row_i in seen_rows:
                raise ValueError(f"检测到 OOS 测试日索引重叠: row={row_i}")
            seen_rows.add(row_i)
        oos[idx, :] = mat

    return np.asfortranarray(oos, dtype=np.float64)
