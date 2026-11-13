//! 因子换手率与横截面排名自相关指标 (metrics/turnover.rs)。
//!
//! 严格遵循白皮书 8.3 节规范：
//! - 截面换手率代理指标：$\text{Turnover} = 1 - \frac{1}{|\mathcal{D}|-1}\sum_{t=2}^{|\mathcal{D}|} \text{Corr}(\text{rank}(\mathbf{f}_t), \text{rank}(\mathbf{f}_{t-1}))$；
//! - 仅在相邻两个交易日同时在池且因子值有限的共同有效标的交集上计算；
//! - 换手率取值范围理论上在 $[0.0, 2.0]$，典型因子位于 $[0.05, 0.85]$；
//! - 支持 Rayon 相邻日并发计算。

use crate::metrics::ic::{compute_fractional_ranks, pearson_correlation};
use ndarray::{ArrayBase, Data, Ix2};
use rayon::prelude::*;

/// 因子换手率与自相关评价指标。
#[derive(Debug, Clone, PartialEq)]
pub struct TurnoverMetrics {
    /// 日度平均换手率代理指标 (1.0 - rank_autocorr，保留向后兼容)
    pub turnover: f64,
    /// 因子截面排名一阶自相关系数均值
    pub rank_autocorr: f64,
    /// 因子截面排名一阶自相关衰减度 (1.0 - rank_autocorr，显式语义)
    pub rank_autocorr_turnover: f64,
    /// 相邻交易日截面排名自相关时序 (长度 T-1, 无效对填充 NaN)
    pub daily_autocorr: Vec<f64>,
    /// 有效相邻交易日对数
    pub valid_days: usize,
}

/// 计算单对相邻交易日 (t-1, t) 共同标的上的截面排名自相关。
pub fn calc_daily_autocorr(
    prev_factor_row: &[f64],
    curr_factor_row: &[f64],
    prev_mask_row: &[bool],
    curr_mask_row: &[bool],
    min_names: usize,
) -> f64 {
    let n = prev_factor_row.len();
    let mut x_vals = Vec::with_capacity(n);
    let mut y_vals = Vec::with_capacity(n);

    for i in 0..n {
        if prev_mask_row[i] && curr_mask_row[i] {
            let f_prev = prev_factor_row[i];
            let f_curr = curr_factor_row[i];
            if f_prev.is_finite() && f_curr.is_finite() {
                x_vals.push(f_prev);
                y_vals.push(f_curr);
            }
        }
    }

    let k = x_vals.len();
    if k < min_names || k < 2 {
        return f64::NAN;
    }

    let mut x_ranks = vec![0.0_f64; k];
    let mut y_ranks = vec![0.0_f64; k];
    let mut buf = Vec::with_capacity(k);
    compute_fractional_ranks(&x_vals, &mut x_ranks, &mut buf);
    compute_fractional_ranks(&y_vals, &mut y_ranks, &mut buf);

    pearson_correlation(&x_ranks, &y_ranks)
}

/// 并行计算整个面板在所有相邻交易日上的换手率指标。
pub fn calc_turnover_metrics<S1, S2>(
    factor: &ArrayBase<S1, Ix2>,
    mask: &ArrayBase<S2, Ix2>,
    min_names: usize,
) -> TurnoverMetrics
where
    S1: Data<Elem = f64> + Sync,
    S2: Data<Elem = bool> + Sync,
{
    let (t, n) = factor.dim();
    assert_eq!(mask.dim(), (t, n), "factor 与 mask 矩阵形状必须一致");

    if t < 2 {
        return TurnoverMetrics {
            turnover: 0.0,
            rank_autocorr: 1.0,
            rank_autocorr_turnover: 0.0,
            daily_autocorr: Vec::new(),
            valid_days: 0,
        };
    }

    struct RowBuf {
        f_prev: Vec<f64>,
        f_curr: Vec<f64>,
        m_prev: Vec<bool>,
        m_curr: Vec<bool>,
    }

    let daily_autocorr: Vec<f64> = (1..t)
        .into_par_iter()
        .map_init(
            || RowBuf {
                f_prev: vec![0.0; n],
                f_curr: vec![0.0; n],
                m_prev: vec![false; n],
                m_curr: vec![false; n],
            },
            |buf, row| {
                for col in 0..n {
                    buf.f_prev[col] = factor[[row - 1, col]];
                    buf.f_curr[col] = factor[[row, col]];
                    buf.m_prev[col] = mask[[row - 1, col]];
                    buf.m_curr[col] = mask[[row, col]];
                }
                calc_daily_autocorr(
                    &buf.f_prev,
                    &buf.f_curr,
                    &buf.m_prev,
                    &buf.m_curr,
                    min_names,
                )
            },
        )
        .collect();

    let valid_autocorr: Vec<f64> = daily_autocorr
        .iter()
        .copied()
        .filter(|v| v.is_finite())
        .collect();
    let m = valid_autocorr.len();

    if m == 0 {
        return TurnoverMetrics {
            turnover: 1.0,
            rank_autocorr: 0.0,
            rank_autocorr_turnover: 1.0,
            daily_autocorr,
            valid_days: 0,
        };
    }

    let sum: f64 = valid_autocorr.iter().sum();
    let mean_autocorr = sum / (m as f64);
    let turnover = 1.0 - mean_autocorr;

    TurnoverMetrics {
        turnover,
        rank_autocorr: mean_autocorr,
        rank_autocorr_turnover: turnover,
        daily_autocorr,
        valid_days: m,
    }
}
