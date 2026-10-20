//! 分位数收益与多空对冲组合评价 (metrics/quantile.rs)。
//!
//! 严格遵循白皮书 8.2 节规范：
//! - 每日将有效标的按因子值从小到大均分为 $K$ 组（默认 $K=5$）；
//! - 组内等权收益 $R_k(t) = \frac{1}{|\mathcal{G}_{k,t}|}\sum_{i \in \mathcal{G}_{k,t}} r_{i, t \to t+1}$；
//! - 多空对冲收益：若因子正向（sign=1）则多最高组、空最低组；若反向（sign=-1）则多最低组、空最高组；
//! - 年化收益率 $\text{Ret}_{\text{LS}} = 252 \times \mu_{\text{LS}}$；
//! - 多空夏普比率 $\text{Sharpe}_{\text{LS}} = \sqrt{252} \times \mu_{\text{LS}} / \sigma_{\text{LS}}$；
//! - 多空最大回撤 $\text{MaxDD}_{\text{LS}}$ 基于累计复利净值曲线；
//! - 单调性得分 $\text{Monotonicity}$：各组累计年化收益率与分位序号的 Spearman 秩相关系数。

use crate::metrics::ic::{compute_fractional_ranks, pearson_correlation};
use crate::nan::cmp_f64;
use ndarray::Array2;
use rayon::prelude::*;

/// 分位数分组与多空组合评估指标。
#[derive(Debug, Clone, PartialEq)]
pub struct QuantileMetrics {
    /// 分位数分组数 K (例如 5)
    pub n_quantiles: usize,
    /// 各组逐日收益时序 [K][T]
    pub group_daily_returns: Vec<Vec<f64>>,
    /// 各组日度均值收益 (长度 K)
    pub group_returns: Vec<f64>,
    /// 各组年化均值收益 (252 * group_returns, 长度 K)
    pub group_annual_returns: Vec<f64>,
    /// 多空对冲逐日收益序列 (长度 T)
    pub ls_daily_returns: Vec<f64>,
    /// 多空日度收益均值
    pub ls_mean: f64,
    /// 多空日度收益样本标准差 (ddof=1)
    pub ls_std: f64,
    /// 多空年化收益率 (252 * ls_mean)
    pub ls_annual_ret: f64,
    /// 多空年化夏普比率
    pub ls_sharpe: f64,
    /// 多空累计复利最大回撤 (0.0 ~ 1.0)
    pub ls_max_dd: f64,
    /// 各组收益率关于分位序号的单调性等级相关系数 (-1.0 ~ 1.0)
    pub monotonicity: f64,
    /// 满足分组条件的有效交易日数
    pub valid_days: usize,
}

/// 计算单个交易日截面上的分位数分组收益与多空利差。
pub fn calc_daily_quantile_returns(
    factor_row: &[f64],
    ret_row: &[f64],
    mask_row: &[bool],
    n_quantiles: usize,
    min_names: usize,
    sign: i32,
) -> (Vec<f64>, f64) {
    let n = factor_row.len();
    let mut pairs: Vec<(usize, f64, f64)> = Vec::with_capacity(n);

    for i in 0..n {
        if mask_row[i] {
            let f = factor_row[i];
            let r = ret_row[i];
            if f.is_finite() && r.is_finite() {
                pairs.push((i, f, r));
            }
        }
    }

    let k_names = pairs.len();
    if k_names < min_names || k_names < n_quantiles || n_quantiles < 2 {
        return (vec![f64::NAN; n_quantiles], f64::NAN);
    }

    // 稳定排序：因子值升序，相同按标的下标排序
    pairs.sort_unstable_by(|a, b| match cmp_f64(a.1, b.1) {
        std::cmp::Ordering::Equal => a.0.cmp(&b.0),
        other => other,
    });

    let mut g_rets = vec![f64::NAN; n_quantiles];
    let k_float = k_names as f64;
    let nq_float = n_quantiles as f64;

    for (q, ret_val) in g_rets.iter_mut().enumerate() {
        let i_start = ((q as f64) * k_float / nq_float).floor() as usize;
        let i_end = (((q + 1) as f64) * k_float / nq_float).floor() as usize;
        let count = i_end - i_start;
        if count == 0 {
            *ret_val = f64::NAN;
            continue;
        }
        let mut sum_r = 0.0_f64;
        for item in &pairs[i_start..i_end] {
            sum_r += item.2;
        }
        *ret_val = sum_r / (count as f64);
    }

    // 多空收益计算
    let r_bottom = g_rets[0];
    let r_top = g_rets[n_quantiles - 1];
    let ls = if r_bottom.is_finite() && r_top.is_finite() {
        if sign >= 0 {
            r_top - r_bottom
        } else {
            r_bottom - r_top
        }
    } else {
        f64::NAN
    };

    (g_rets, ls)
}

/// 全量面板多日分位数与多空收益计算。
pub fn calc_quantile_metrics(
    factor: &Array2<f64>,
    forward_ret: &Array2<f64>,
    mask: &Array2<bool>,
    n_quantiles: usize,
    min_names: usize,
    sign: i32,
) -> QuantileMetrics {
    let (t, n) = factor.dim();
    assert_eq!(
        forward_ret.dim(),
        (t, n),
        "factor 与 forward_ret 矩阵形状必须一致"
    );
    assert_eq!(mask.dim(), (t, n), "factor 与 mask 矩阵形状必须一致");
    assert!(n_quantiles >= 2, "n_quantiles 必须 >= 2");

    // 并行计算每日的分组收益与多空收益
    let daily_results: Vec<(Vec<f64>, f64)> = (0..t)
        .into_par_iter()
        .map(|row| {
            let mut f_row = Vec::with_capacity(n);
            let mut r_row = Vec::with_capacity(n);
            let mut m_row = Vec::with_capacity(n);
            for col in 0..n {
                f_row.push(factor[[row, col]]);
                r_row.push(forward_ret[[row, col]]);
                m_row.push(mask[[row, col]]);
            }
            calc_daily_quantile_returns(&f_row, &r_row, &m_row, n_quantiles, min_names, sign)
        })
        .collect();

    let mut group_daily_returns: Vec<Vec<f64>> = vec![vec![f64::NAN; t]; n_quantiles];
    let mut ls_daily_returns = vec![f64::NAN; t];

    for row in 0..t {
        let (ref grets, ls) = daily_results[row];
        for q in 0..n_quantiles {
            group_daily_returns[q][row] = grets[q];
        }
        ls_daily_returns[row] = ls;
    }

    // 统计有效日
    let valid_indices: Vec<usize> = (0..t)
        .filter(|&row| ls_daily_returns[row].is_finite())
        .collect();
    let m = valid_indices.len();

    if m == 0 {
        return QuantileMetrics {
            n_quantiles,
            group_daily_returns,
            group_returns: vec![0.0; n_quantiles],
            group_annual_returns: vec![0.0; n_quantiles],
            ls_daily_returns,
            ls_mean: 0.0,
            ls_std: 0.0,
            ls_annual_ret: 0.0,
            ls_sharpe: 0.0,
            ls_max_dd: 0.0,
            monotonicity: 0.0,
            valid_days: 0,
        };
    }

    let mf = m as f64;

    // 计算各组平均收益
    let mut group_returns = vec![0.0_f64; n_quantiles];
    let mut group_annual_returns = vec![0.0_f64; n_quantiles];

    for q in 0..n_quantiles {
        let sum_q: f64 = valid_indices
            .iter()
            .map(|&row| group_daily_returns[q][row])
            .sum();
        let mean_q = sum_q / mf;
        group_returns[q] = mean_q;
        group_annual_returns[q] = mean_q * 252.0;
    }

    // 多空均值与样本标准差
    let ls_sum: f64 = valid_indices.iter().map(|&row| ls_daily_returns[row]).sum();
    let ls_mean = ls_sum / mf;

    let ls_std = if m > 1 {
        let var_sum: f64 = valid_indices
            .iter()
            .map(|&row| (ls_daily_returns[row] - ls_mean).powi(2))
            .sum();
        (var_sum / ((m - 1) as f64)).sqrt()
    } else {
        0.0
    };

    let ls_annual_ret = ls_mean * 252.0;
    let ls_sharpe = if ls_std > 1e-12 {
        (252.0_f64).sqrt() * ls_mean / ls_std
    } else {
        0.0
    };

    // 计算累计复利最大回撤
    let mut wealth = 1.0_f64;
    let mut peak = 1.0_f64;
    let mut max_dd = 0.0_f64;

    for &row in &valid_indices {
        let r = ls_daily_returns[row];
        wealth = (wealth * (1.0 + r)).max(0.0);
        if wealth > peak {
            peak = wealth;
        }
        if peak > 0.0 {
            let dd = (peak - wealth) / peak;
            if dd > max_dd {
                max_dd = dd;
            }
        }
    }

    // 单调性计算：根据 sign 对齐方向计算等级相关系数
    let mut expected_order_returns = vec![0.0_f64; n_quantiles];
    for q in 0..n_quantiles {
        expected_order_returns[q] = if sign >= 0 {
            group_annual_returns[q]
        } else {
            group_annual_returns[n_quantiles - 1 - q]
        };
    }

    let q_indices: Vec<f64> = (0..n_quantiles).map(|i| i as f64).collect();
    let mut q_ranks = vec![0.0_f64; n_quantiles];
    let mut r_ranks = vec![0.0_f64; n_quantiles];
    let mut buf = Vec::with_capacity(n_quantiles);
    compute_fractional_ranks(&q_indices, &mut q_ranks, &mut buf);
    compute_fractional_ranks(&expected_order_returns, &mut r_ranks, &mut buf);
    let mono = pearson_correlation(&q_ranks, &r_ranks);
    let monotonicity = if mono.is_finite() { mono } else { 0.0 };

    QuantileMetrics {
        n_quantiles,
        group_daily_returns,
        group_returns,
        group_annual_returns,
        ls_daily_returns,
        ls_mean,
        ls_std,
        ls_annual_ret,
        ls_sharpe,
        ls_max_dd: max_dd,
        monotonicity,
        valid_days: m,
    }
}
