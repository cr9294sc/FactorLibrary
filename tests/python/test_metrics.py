"""P3 验收测试：Rust 指标库与纯 Python 参考实现 (ops_ref.py) 全量对账与性质测试。

涵盖白皮书 P3 全部验收指标与 Operator Parity Gate (门禁 4)：
1. 截面 Rank IC / Pearson IC 对账：在含高密度 NaN、并列值、±Inf 的噪声矩阵上比对，误差严格 < 1e-9；
2. 分位数多空收益、夏普比率、最大回撤、单调性得分对账，误差严格 < 1e-9；
3. 截面换手率与排名自相关对账，误差严格 < 1e-9；
4. 因子矩阵截面相关系数矩阵对账，误差严格 < 1e-9；
5. 极端边界测试：全 NaN 矩阵、零方差常数矩阵、样本数不足 min_names 等边界场景无 Panic。
"""

from __future__ import annotations

import factorlib_core
import numpy as np
import pytest
from factorlib.core import ops_ref


def _make_noisy_test_matrices(
    seed: int, T: int = 40, N: int = 15, nan_prob: float = 0.20
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """生成带有高密度 NaN、并列值与 ±Inf 的随机测试矩阵。"""
    rng = np.random.default_rng(seed)
    f = rng.normal(loc=10.0, scale=5.0, size=(T, N))
    lbl = rng.normal(loc=0.01, scale=0.03, size=(T, N))

    # 注入并列值
    tie_mask = rng.random((T, N)) < 0.10
    f[tie_mask] = np.round(f[tie_mask], 0)
    lbl[tie_mask] = np.round(lbl[tie_mask], 2)

    # 注入 NaN
    f[rng.random((T, N)) < nan_prob] = np.nan
    lbl[rng.random((T, N)) < nan_prob] = np.nan

    # 注入少量 ±Inf
    inf_f = rng.random((T, N)) < 0.02
    f[inf_f] = np.where(rng.random(np.sum(inf_f)) < 0.5, np.inf, -np.inf)

    # 构造 mask: 部分标的停牌不在池
    mask = rng.random((T, N)) > 0.15

    return (
        np.asfortranarray(f, dtype=np.float64),
        np.asfortranarray(lbl, dtype=np.float64),
        np.asfortranarray(mask, dtype=bool),
    )


def test_rank_ic_parity_against_ops_ref():
    """测试 calc_ic 与 ops_ref.ref_rank_ic 误差 < 1e-9。"""
    for seed in [42, 101, 2026]:
        f, lbl, mask = _make_noisy_test_matrices(seed)

        # 1. Rank IC (rank=True)
        rust_ic = factorlib_core.calc_ic(f, lbl, mask, min_names=5, rank=True, horizon=5)
        py_ic = ops_ref.ref_rank_ic(f, lbl, mask, min_names=5, rank=True, horizon=5)

        assert abs(rust_ic["ic_mean"] - py_ic["ic_mean"]) < 1e-9
        assert abs(rust_ic["ic_std"] - py_ic["ic_std"]) < 1e-9
        assert abs(rust_ic["ic_ir"] - py_ic["ic_ir"]) < 1e-9
        assert abs(rust_ic["ic_t"] - py_ic["ic_t"]) < 1e-9
        assert abs(rust_ic["ic_win_rate"] - py_ic["ic_win_rate"]) < 1e-9
        assert abs(rust_ic["coverage"] - py_ic["coverage"]) < 1e-9
        assert rust_ic["valid_days"] == py_ic["valid_days"]

        # 逐日 IC 时序对账
        r_daily = rust_ic["daily_ic"]
        p_daily = py_ic["daily_ic"]
        np.testing.assert_allclose(
            np.where(np.isnan(r_daily), 0.0, r_daily),
            np.where(np.isnan(p_daily), 0.0, p_daily),
            atol=1e-9,
            rtol=1e-9,
        )

        # 2. Pearson IC (rank=False)
        rust_p = factorlib_core.calc_ic(f, lbl, mask, min_names=5, rank=False, horizon=1)
        py_p = ops_ref.ref_rank_ic(f, lbl, mask, min_names=5, rank=False, horizon=1)
        assert abs(rust_p["ic_mean"] - py_p["ic_mean"]) < 1e-9
        assert abs(rust_p["ic_std"] - py_p["ic_std"]) < 1e-9


def test_quantile_returns_parity_against_ops_ref():
    """测试 calc_quantile_returns 与 ops_ref.ref_quantile_returns 误差 < 1e-9。"""
    for seed in [123, 456, 789]:
        f, lbl, mask = _make_noisy_test_matrices(seed)

        rust_qm = factorlib_core.calc_quantile_returns(
            f, lbl, mask, n_quantiles=5, min_names=5, sign=1
        )
        py_qm = ops_ref.ref_quantile_returns(f, lbl, mask, n_quantiles=5, min_names=5, sign=1)

        assert abs(rust_qm["ls_mean"] - py_qm["ls_mean"]) < 1e-9
        assert abs(rust_qm["ls_std"] - py_qm["ls_std"]) < 1e-9
        assert abs(rust_qm["ls_annual_ret"] - py_qm["ls_annual_ret"]) < 1e-9
        assert abs(rust_qm["ls_sharpe"] - py_qm["ls_sharpe"]) < 1e-9
        assert abs(rust_qm["ls_max_dd"] - py_qm["ls_max_dd"]) < 1e-9
        assert abs(rust_qm["monotonicity"] - py_qm["monotonicity"]) < 1e-9
        assert rust_qm["valid_days"] == py_qm["valid_days"]
        assert abs(rust_qm["top_turnover"] - py_qm["top_turnover"]) < 1e-9
        np.testing.assert_allclose(
            np.where(np.isnan(rust_qm["top_turnover_series"]), 0.0, rust_qm["top_turnover_series"]),
            np.where(np.isnan(py_qm["top_turnover_series"]), 0.0, py_qm["top_turnover_series"]),
            atol=1e-9,
        )

        for q in range(5):
            assert abs(rust_qm["group_returns"][q] - py_qm["group_returns"][q]) < 1e-9
            assert abs(rust_qm["group_annual_returns"][q] - py_qm["group_annual_returns"][q]) < 1e-9

        # 反向 sign=-1
        rust_qm_rev = factorlib_core.calc_quantile_returns(
            f, lbl, mask, n_quantiles=5, min_names=5, sign=-1
        )
        py_qm_rev = ops_ref.ref_quantile_returns(f, lbl, mask, n_quantiles=5, min_names=5, sign=-1)
        assert abs(rust_qm_rev["ls_mean"] - py_qm_rev["ls_mean"]) < 1e-9
        assert abs(rust_qm_rev["monotonicity"] - py_qm_rev["monotonicity"]) < 1e-9
        assert abs(rust_qm_rev["top_turnover"] - py_qm_rev["top_turnover"]) < 1e-9


def test_turnover_parity_against_ops_ref():
    """测试 calc_turnover 与 ops_ref.ref_turnover 误差 < 1e-9。"""
    for seed in [55, 66, 77]:
        f, _, mask = _make_noisy_test_matrices(seed)

        rust_to = factorlib_core.calc_turnover(f, mask, min_names=5)
        py_to = ops_ref.ref_turnover(f, mask, min_names=5)

        assert abs(rust_to["turnover"] - py_to["turnover"]) < 1e-9
        assert abs(rust_to["rank_autocorr"] - py_to["rank_autocorr"]) < 1e-9
        assert abs(rust_to["rank_autocorr_turnover"] - py_to["rank_autocorr_turnover"]) < 1e-9
        assert rust_to["rank_autocorr_turnover"] == rust_to["turnover"]
        assert rust_to["valid_days"] == py_to["valid_days"]

        r_auto = rust_to["daily_autocorr"]
        p_auto = py_to["daily_autocorr"]
        np.testing.assert_allclose(
            np.where(np.isnan(r_auto), 0.0, r_auto),
            np.where(np.isnan(p_auto), 0.0, p_auto),
            atol=1e-9,
            rtol=1e-9,
        )


def test_top_quantile_turnover_parity_and_known_rotation():
    """测试 calc_top_quantile_turnover 与 ops_ref 对齐，并测试已知 10 选 1 轮换换手率严格等于 0.10。"""
    # 1. 随机矩阵双轨对账
    for seed in [101, 202, 303]:
        f, _, mask = _make_noisy_test_matrices(seed)
        rust_to = factorlib_core.calc_top_quantile_turnover(
            f, mask, n_quantiles=5, min_names=5, sign=1
        )
        py_to = ops_ref.ref_top_quantile_turnover(f, mask, n_quantiles=5, min_names=5, sign=1)

        assert abs(rust_to["turnover"] - py_to["turnover"]) < 1e-9
        assert abs(rust_to["mean_top_turnover"] - py_to["mean_top_turnover"]) < 1e-9
        assert rust_to["valid_days"] == py_to["valid_days"]
        np.testing.assert_allclose(
            np.where(np.isnan(rust_to["daily_turnover"]), 0.0, rust_to["daily_turnover"]),
            np.where(np.isnan(py_to["daily_turnover"]), 0.0, py_to["daily_turnover"]),
            atol=1e-9,
            rtol=1e-9,
        )

    # 2. 构造 100 只股票、10 分位 (每组 10 只股票)、2 个交易日的面板
    # Day 0: 股票 0..99 因子值 0..99，G10 组为标的 90..99
    # Day 1: 标的 99 因子值降至 -1.0，其他不变。G10 组变为 89..98 (9 只重叠，1 只出圈，严格换手 10.0%)
    T, N = 2, 100
    f_known = np.zeros((T, N), dtype=np.float64, order="F")
    f_known[0, :] = np.arange(N, dtype=np.float64)
    f_known[1, :] = np.arange(N, dtype=np.float64)
    f_known[1, 99] = -1.0
    mask_known = np.ones((T, N), dtype=bool, order="F")

    known_res = factorlib_core.calc_top_quantile_turnover(
        f_known, mask_known, n_quantiles=10, min_names=10, sign=1
    )
    assert known_res["valid_days"] == 1
    assert np.isnan(known_res["daily_turnover"][0])
    assert abs(known_res["daily_turnover"][1] - 0.10) < 1e-12
    assert abs(known_res["turnover"] - 0.10) < 1e-12

    # 对比截面秩自相关距离 1 - rho: 单只标的秩次移动导致 1 - rho 约 0.059
    rank_to = factorlib_core.calc_turnover(f_known, mask_known, min_names=10)
    assert rank_to["turnover"] < known_res["turnover"]
    assert abs(rank_to["turnover"] - 0.05940594059405946) < 1e-6


def test_correlation_matrix_parity_against_ops_ref():
    """测试 calc_pairwise_correlation 与 calc_correlation_matrix 误差 < 1e-9。"""
    rng = np.random.default_rng(999)
    T, N = 30, 10
    f1 = np.asfortranarray(rng.normal(size=(T, N)))
    f2 = np.asfortranarray(rng.normal(size=(T, N)))
    f3 = np.asfortranarray(rng.normal(size=(T, N)))
    mask = np.ones((T, N), dtype=bool, order="F")

    # 成对相关
    rho_rust = factorlib_core.calc_pairwise_correlation(f1, f2, mask, min_names=5, rank=True)
    rho_py = ops_ref.ref_pairwise_correlation(f1, f2, mask, min_names=5, rank=True)
    assert abs(rho_rust - rho_py) < 1e-9

    # 矩阵相关
    mat_rust = factorlib_core.calc_correlation_matrix([f1, f2, f3], mask, min_names=5, rank=True)
    mat_py = ops_ref.ref_correlation_matrix([f1, f2, f3], mask, min_names=5, rank=True)
    np.testing.assert_allclose(mat_rust, mat_py, atol=1e-9, rtol=1e-9)


def test_metrics_extreme_edge_cases():
    """极端边缘测试：全 NaN、常数矩阵、零有效标的、单标的在池等无崩溃测试。"""
    T, N = 10, 5
    all_nan = np.full((T, N), np.nan, dtype=np.float64, order="F")
    all_zero = np.zeros((T, N), dtype=np.float64, order="F")
    mask_true = np.ones((T, N), dtype=bool, order="F")
    mask_false = np.zeros((T, N), dtype=bool, order="F")

    # 1. 全 NaN
    ic_nan = factorlib_core.calc_ic(all_nan, all_zero, mask_true, min_names=5)
    assert ic_nan["valid_days"] == 0
    assert ic_nan["ic_mean"] == 0.0

    qm_nan = factorlib_core.calc_quantile_returns(all_nan, all_zero, mask_true, n_quantiles=3)
    assert qm_nan["valid_days"] == 0
    assert qm_nan["ls_annual_ret"] == 0.0

    to_nan = factorlib_core.calc_turnover(all_nan, mask_true, min_names=3)
    assert to_nan["valid_days"] == 0
    assert to_nan["turnover"] == 1.0

    # 2. 全 False mask
    ic_nomask = factorlib_core.calc_ic(all_zero, all_zero, mask_false, min_names=3)
    assert ic_nomask["valid_days"] == 0

    # 3. 常数方差为 0
    const_f = np.ones((T, N), dtype=np.float64, order="F") * 42.0
    const_l = np.ones((T, N), dtype=np.float64, order="F") * 10.0
    ic_const = factorlib_core.calc_ic(const_f, const_l, mask_true, min_names=3)
    assert ic_const["valid_days"] == 0  # 零方差无有效相关系数

    # 4. P0-08: n_quantiles < 2 必须优雅抛出 ValueError 而非触发跨 FFI Rust Panic
    with pytest.raises(ValueError, match="n_quantiles"):
        factorlib_core.calc_quantile_returns(all_zero, all_zero, mask_true, n_quantiles=1)
    with pytest.raises(ValueError, match="n_quantiles"):
        factorlib_core.calc_top_quantile_turnover(all_zero, mask_true, n_quantiles=1)

    eng = factorlib_core.Engine(
        fields={"close": all_zero},
        mask=mask_true,
        labels={"y_1": all_zero, "y_5": all_zero},
    )
    with pytest.raises(ValueError, match="n_quantiles"):
        eng.quantile_returns(all_zero, n_quantiles=1)
    with pytest.raises(ValueError, match="n_quantiles"):
        eng.top_quantile_turnover(all_zero, n_quantiles=1)
    with pytest.raises(ValueError, match="n_quantiles"):
        eng.factor_metrics(all_zero, n_quantiles=1)
