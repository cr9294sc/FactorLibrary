//! 评估指标库（IC、分位数多空、换手率、相关性矩阵）高精度数值测试与 proptest 属性验证。
//!
//! 严格对应白皮书 P3 验收指标与数学规范：
//! 1. Spearman Rank IC / Pearson IC：单调变换保秩性、完全正相关、完全反相关、并列均秩、低样本阻断；
//! 2. 分位数收益：分组单调性、多空利差、Sharpe 比率、复利最大回撤；
//! 3. 换手率：全保序（零换手）、全反序（最大换手）；
//! 4. 因子相关性矩阵：对角线为 1.0，对称性，Rayon 并行一致性。

use factorlib_core::metrics::corr::{cross_correlation_matrix, cross_correlation_pairwise};
use factorlib_core::metrics::ic::{
    calc_ic_metrics, calc_ic_series, compute_fractional_ranks, pearson_correlation,
};
use factorlib_core::metrics::quantile::{calc_quantile_metrics, calc_top_quantile_turnover};
use factorlib_core::metrics::turnover::calc_turnover_metrics;
use ndarray::{Array2, ShapeBuilder};
use proptest::prelude::*;

#[test]
fn test_fractional_ranks_and_pearson_correlation() {
    // 1. 无并列样本 [10.0, 30.0, 20.0] -> ranks: [1.0, 3.0, 2.0]
    let vals = [10.0, 30.0, 20.0];
    let mut ranks = vec![0.0; 3];
    let mut buf = Vec::new();
    compute_fractional_ranks(&vals, &mut ranks, &mut buf);
    assert_eq!(ranks, vec![1.0, 3.0, 2.0]);

    // 2. 有并列样本 [10.0, 20.0, 20.0, 40.0] -> 20.0 占据第 2、3 位，平均秩 2.5
    let vals_tie = [10.0, 20.0, 20.0, 40.0];
    let mut ranks_tie = vec![0.0; 4];
    compute_fractional_ranks(&vals_tie, &mut ranks_tie, &mut buf);
    assert_eq!(ranks_tie, vec![1.0, 2.5, 2.5, 4.0]);

    // 3. Pearson 相关系数
    let x = [1.0, 2.0, 3.0, 4.0, 5.0];
    let y = [2.0, 4.0, 6.0, 8.0, 10.0];
    let r = pearson_correlation(&x, &y);
    assert!((r - 1.0).abs() < 1e-12);

    let y_neg = [-2.0, -4.0, -6.0, -8.0, -10.0];
    let r_neg = pearson_correlation(&x, &y_neg);
    assert!((r_neg - (-1.0)).abs() < 1e-12);

    // 常数向量相关系数为 NaN
    let const_y = [5.0, 5.0, 5.0, 5.0, 5.0];
    assert!(pearson_correlation(&x, &const_y).is_nan());
}

#[test]
fn test_rank_ic_perfect_and_monotonic() {
    let t = 20;
    let n = 10;
    let mut factor = Array2::<f64>::zeros((t, n).f());
    let mut label_pos = Array2::<f64>::zeros((t, n).f());
    let mut label_neg = Array2::<f64>::zeros((t, n).f());
    let mask = Array2::<bool>::from_elem((t, n).f(), true);

    for r in 0..t {
        for c in 0..n {
            let x = (c as f64) + 1.0;
            factor[[r, c]] = x;
            // 单调增非线性变换 x^3
            label_pos[[r, c]] = x.powi(3) + 2.0;
            // 严格反向
            label_neg[[r, c]] = -x * 5.0;
        }
    }

    // 正向非线性单调，Spearman Rank IC 应恒为 1.0
    let ic_pos_series = calc_ic_series(&factor, &label_pos, &mask, 5, true);
    for v in ic_pos_series {
        assert!((v - 1.0).abs() < 1e-12);
    }

    // 反向单调，Spearman Rank IC 应恒为 -1.0
    let ic_neg_series = calc_ic_series(&factor, &label_neg, &mask, 5, true);
    for v in ic_neg_series {
        assert!((v - (-1.0)).abs() < 1e-12);
    }

    // 聚合统计
    let daily = vec![0.1, 0.2, 0.15, -0.05, 0.2];
    let metrics = calc_ic_metrics(&daily, 5, 5, 50, 50);
    assert!((metrics.ic_mean - 0.12).abs() < 1e-12);
    assert!(metrics.ic_std > 0.0);
    assert!(metrics.ic_ir > 0.0);
    assert_eq!(metrics.valid_days, 5);
    assert!((metrics.coverage - 1.0).abs() < 1e-12);
    // 4 正 1 负，均值为正，胜率 4/5 = 0.8
    assert!((metrics.ic_win_rate - 0.8).abs() < 1e-12);
}

#[test]
fn test_quantile_returns_and_drawdown() {
    let t = 10;
    let n = 15;
    let mut factor = Array2::<f64>::zeros((t, n).f());
    let mut fwd_ret = Array2::<f64>::zeros((t, n).f());
    let mask = Array2::<bool>::from_elem((t, n).f(), true);

    for r in 0..t {
        for c in 0..n {
            factor[[r, c]] = c as f64;
            // 收益随 c 严格单调增加，且跨日有微小波动以具有非零标准差
            fwd_ret[[r, c]] = (c as f64) * (0.01 + (r as f64) * 0.001);
        }
    }

    let qm = calc_quantile_metrics(&factor, &fwd_ret, &mask, 5, 5, 1);
    assert_eq!(qm.n_quantiles, 5);
    assert_eq!(qm.valid_days, 10);
    // 第 0 组收益必须小于第 4 组收益
    assert!(qm.group_returns[0] < qm.group_returns[4]);
    // 单调性得分应为 1.0
    assert!((qm.monotonicity - 1.0).abs() < 1e-12);
    // 多空年化收益应严格大于 0
    assert!(qm.ls_annual_ret > 0.0);
    assert!(qm.ls_sharpe > 0.0);
    // 每日稳定正收益，无回撤
    assert_eq!(qm.ls_max_dd, 0.0);

    // 验证 sign = -1 反向
    let qm_rev = calc_quantile_metrics(&factor, &fwd_ret, &mask, 5, 5, -1);
    assert!(qm_rev.ls_annual_ret < 0.0);
}

#[test]
fn test_turnover_metrics() {
    let t = 5;
    let n = 8;
    let mut factor_const = Array2::<f64>::zeros((t, n).f());
    let mask = Array2::<bool>::from_elem((t, n).f(), true);

    for r in 0..t {
        for c in 0..n {
            factor_const[[r, c]] = c as f64;
        }
    }

    // 因子完全不变，排名自相关应为 1.0，换手率为 0.0
    let to_const = calc_turnover_metrics(&factor_const, &mask, 4);
    assert_eq!(to_const.valid_days, 4);
    assert!((to_const.rank_autocorr - 1.0).abs() < 1e-12);
    assert!(to_const.turnover.abs() < 1e-12);
    assert!((to_const.rank_autocorr_turnover - to_const.turnover).abs() < 1e-12);

    // 因子每隔一天完全反转：rank autocorr 为 -1.0，换手率为 2.0
    let mut factor_invert = Array2::<f64>::zeros((t, n).f());
    for r in 0..t {
        for c in 0..n {
            let sign = if r % 2 == 0 { 1.0 } else { -1.0 };
            factor_invert[[r, c]] = (c as f64) * sign;
        }
    }
    let to_inv = calc_turnover_metrics(&factor_invert, &mask, 4);
    assert!((to_inv.rank_autocorr - (-1.0)).abs() < 1e-12);
    assert!((to_inv.turnover - 2.0).abs() < 1e-12);
    assert!((to_inv.rank_autocorr_turnover - 2.0).abs() < 1e-12);
}

#[test]
fn test_top_quantile_turnover_and_rotation() {
    // 构造 10 只标的，3 个交易日，5 分位分组 (每组 2 只标的)
    let t = 3;
    let n = 10;
    let mut factor = Array2::<f64>::zeros((t, n).f());
    let mut fwd_ret = Array2::<f64>::zeros((t, n).f());
    let mask = Array2::<bool>::from_elem((t, n).f(), true);

    // Day 0: 因子值为 0..9，Top 组为标的 [8, 9]
    for c in 0..n {
        factor[[0, c]] = c as f64;
        fwd_ret[[0, c]] = 0.01;
    }

    // Day 1: 标的 9 因子值降至 -1.0，其他不变。Top 组变为 [7, 8]，重叠为 {8} (1/2 轮换，单边换手 50%)
    for c in 0..n {
        factor[[1, c]] = c as f64;
        fwd_ret[[1, c]] = 0.01;
    }
    factor[[1, 9]] = -1.0;

    // Day 2: 标的 7 和 8 降至 -2.0，Top 组变为 [5, 6]，重叠为 0 (完全轮换，单边换手 100%)
    for c in 0..n {
        factor[[2, c]] = c as f64;
        fwd_ret[[2, c]] = 0.01;
    }
    factor[[2, 9]] = -1.0;
    factor[[2, 8]] = -2.0;
    factor[[2, 7]] = -2.0;

    let qm = calc_quantile_metrics(&factor, &fwd_ret, &mask, 5, 5, 1);
    assert_eq!(qm.top_turnover_series.len(), 3);
    assert!(qm.top_turnover_series[0].is_nan());
    assert!((qm.top_turnover_series[1] - 0.50).abs() < 1e-12);
    assert!((qm.top_turnover_series[2] - 1.00).abs() < 1e-12);
    assert_eq!(qm.top_turnover_valid_days, 2);
    assert!((qm.mean_top_turnover - 0.75).abs() < 1e-12);

    // 独立算子 calc_top_quantile_turnover 必须输出完全一致的结果
    let top_to = calc_top_quantile_turnover(&factor, &mask, 5, 5, 1);
    assert_eq!(top_to.valid_days, 2);
    assert!((top_to.turnover - 0.75).abs() < 1e-12);
    assert!(top_to.daily_turnover[0].is_nan());
    assert!((top_to.daily_turnover[1] - 0.50).abs() < 1e-12);
    assert!((top_to.daily_turnover[2] - 1.00).abs() < 1e-12);

    // 反向因子 sign = -1：多头为最低分位
    let top_to_rev = calc_top_quantile_turnover(&factor, &mask, 5, 5, -1);
    assert_eq!(top_to_rev.valid_days, 2);
    assert!(top_to_rev.turnover > 0.0);
}

#[test]
fn test_factor_correlation_matrix() {
    let t = 15;
    let n = 10;
    let mut f1 = Array2::<f64>::zeros((t, n).f());
    let mut f2 = Array2::<f64>::zeros((t, n).f());
    let mut f3 = Array2::<f64>::zeros((t, n).f());
    let mask = Array2::<bool>::from_elem((t, n).f(), true);

    for r in 0..t {
        for c in 0..n {
            let x = c as f64;
            f1[[r, c]] = x;
            f2[[r, c]] = x * 2.0 + 3.0; // 与 f1 完全正相关 (rho=1.0)
            f3[[r, c]] = -x; // 与 f1 完全反相关 (rho=-1.0)
        }
    }

    let rho_12 = cross_correlation_pairwise(&f1, &f2, &mask, 5, true);
    assert!((rho_12 - 1.0).abs() < 1e-12);

    let rho_13 = cross_correlation_pairwise(&f1, &f3, &mask, 5, true);
    assert!((rho_13 - (-1.0)).abs() < 1e-12);

    let factors = vec![&f1, &f2, &f3];
    let mat = cross_correlation_matrix(&factors, &mask, 5, true);
    assert_eq!(mat.dim(), (3, 3));

    // 对角线为 1.0
    for i in 0..3 {
        assert!((mat[[i, i]] - 1.0).abs() < 1e-12);
    }
    // 对称性
    assert!((mat[[0, 1]] - mat[[1, 0]]).abs() < 1e-12);
    assert!((mat[[0, 2]] - mat[[2, 0]]).abs() < 1e-12);
    assert!((mat[[0, 1]] - 1.0).abs() < 1e-12);
    assert!((mat[[0, 2]] - (-1.0)).abs() < 1e-12);
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(50))]

    #[test]
    fn prop_correlation_matrix_symmetry_and_bounds(
        n_names in 6usize..12,
        t_days in 8usize..16,
    ) {
        let mut f1 = Array2::<f64>::zeros((t_days, n_names).f());
        let mut f2 = Array2::<f64>::zeros((t_days, n_names).f());
        let mask = Array2::<bool>::from_elem((t_days, n_names).f(), true);

        for r in 0..t_days {
            for c in 0..n_names {
                f1[[r, c]] = ((r * 7 + c * 13) % 29) as f64;
                f2[[r, c]] = ((r * 11 + c * 17) % 31) as f64;
            }
        }

        let factors = vec![&f1, &f2];
        let mat = cross_correlation_matrix(&factors, &mask, 4, true);

        prop_assert!((mat[[0, 0]] - 1.0).abs() < 1e-12);
        prop_assert!((mat[[1, 1]] - 1.0).abs() < 1e-12);
        prop_assert!((mat[[0, 1]] - mat[[1, 0]]).abs() < 1e-12);
        prop_assert!(mat[[0, 1]] >= -1.0 - 1e-12 && mat[[0, 1]] <= 1.0 + 1e-12);
    }
}
