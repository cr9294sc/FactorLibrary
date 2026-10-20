//! 因子截面相关系数矩阵与去相关计算 (metrics/corr.rs)。
//!
//! 严格对应白皮书 8 章、9.4 节与 9.5 节规范：
//! - 计算因子间在有效标的池上的逐日截面相关性，并取时序均值；
//! - 支持 Spearman 秩相关（rank=true）与 Pearson 相关（rank=false）；
//! - 支持双因子快速对账以及 M 个因子的高并发相关性矩阵并行计算；
//! - 满足对角线恒为 1.0，对称矩阵性质 $\mathbf{C}_{j, k} = \mathbf{C}_{k, j}$。

use crate::metrics::ic::{compute_fractional_ranks, pearson_correlation};
use ndarray::Array2;
use rayon::prelude::*;

/// 计算两个因子在单日截面共同有效标的上的相关系数。
pub fn calc_daily_factor_correlation(
    a_row: &[f64],
    b_row: &[f64],
    mask_row: &[bool],
    min_names: usize,
    rank: bool,
) -> f64 {
    let n = a_row.len();
    let mut x_vals = Vec::with_capacity(n);
    let mut y_vals = Vec::with_capacity(n);

    for i in 0..n {
        if mask_row[i] {
            let va = a_row[i];
            let vb = b_row[i];
            if va.is_finite() && vb.is_finite() {
                x_vals.push(va);
                y_vals.push(vb);
            }
        }
    }

    let k = x_vals.len();
    if k < min_names || k < 2 {
        return f64::NAN;
    }

    if rank {
        let mut x_ranks = vec![0.0_f64; k];
        let mut y_ranks = vec![0.0_f64; k];
        let mut buf = Vec::with_capacity(k);
        compute_fractional_ranks(&x_vals, &mut x_ranks, &mut buf);
        compute_fractional_ranks(&y_vals, &mut y_ranks, &mut buf);
        pearson_correlation(&x_ranks, &y_ranks)
    } else {
        pearson_correlation(&x_vals, &y_vals)
    }
}

/// 计算两个因子矩阵之间的全时段平均横截面相关系数。
pub fn cross_correlation_pairwise(
    a: &Array2<f64>,
    b: &Array2<f64>,
    mask: &Array2<bool>,
    min_names: usize,
    rank: bool,
) -> f64 {
    let (t, n) = a.dim();
    assert_eq!(b.dim(), (t, n), "因子矩阵形状必须一致");
    assert_eq!(mask.dim(), (t, n), "mask 矩阵形状必须一致");

    let daily_corrs: Vec<f64> = (0..t)
        .into_par_iter()
        .map(|row| {
            let mut a_row = Vec::with_capacity(n);
            let mut b_row = Vec::with_capacity(n);
            let mut m_row = Vec::with_capacity(n);
            for col in 0..n {
                a_row.push(a[[row, col]]);
                b_row.push(b[[row, col]]);
                m_row.push(mask[[row, col]]);
            }
            calc_daily_factor_correlation(&a_row, &b_row, &m_row, min_names, rank)
        })
        .collect();

    let valid: Vec<f64> = daily_corrs.into_iter().filter(|v| v.is_finite()).collect();
    if valid.is_empty() {
        0.0
    } else {
        let sum: f64 = valid.iter().sum();
        sum / (valid.len() as f64)
    }
}

/// 并行计算 M 个因子之间的 M x M 截面相关系数矩阵。
pub fn cross_correlation_matrix(
    factors: &[&Array2<f64>],
    mask: &Array2<bool>,
    min_names: usize,
    rank: bool,
) -> Array2<f64> {
    let m = factors.len();
    let mut out = Array2::<f64>::zeros((m, m));
    if m == 0 {
        return out;
    }

    for i in 0..m {
        out[[i, i]] = 1.0;
    }

    if m == 1 {
        return out;
    }

    // 生成上三角所有配对索引
    let mut pairs = Vec::with_capacity(m * (m - 1) / 2);
    for i in 0..m {
        for j in (i + 1)..m {
            pairs.push((i, j));
        }
    }

    // 并行计算每对因子的相关系数
    let pair_results: Vec<((usize, usize), f64)> = pairs
        .into_par_iter()
        .map(|(i, j)| {
            let rho = cross_correlation_pairwise(factors[i], factors[j], mask, min_names, rank);
            ((i, j), rho)
        })
        .collect();

    for ((i, j), rho) in pair_results {
        out[[i, j]] = rho;
        out[[j, i]] = rho;
    }

    out
}
