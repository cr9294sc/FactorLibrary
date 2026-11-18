//! 多因子正交残差检验 / Spanning Test 计算内核 (metrics/spanning.rs)。
//!
//! 严格对应设计文档 §5.4 关 2 与 M8.3 规范：
//! - 对每个有效交易日横截面执行 OLS 正交回归：$f_{\text{cand}, t} \sim 1 + \mathbf{F}_{\text{active}, t}$；
//! - 采用二次修正 Gram-Schmidt (MGS2) 正交投影求解残差 $\epsilon_t = (I - P_X) f_{\text{cand}, t}$，
//!   对高度共线或常数退化基准柱具备严格数值稳定性，无需任意正则扰动；
//! - 计算每日残差与未来标签的 Spearman Rank IC 序列，并统计 `mean`、`std`、`icir`、
//!   原始 `t_stat` 以及经持有期自由度修正的样本量校正 `t_stat_adj` ($t_{\text{adj}} = \text{ICIR} \sqrt{N / h}$)。

use crate::metrics::ic::{compute_fractional_ranks, pearson_correlation};
use ndarray::{ArrayBase, ArrayView2, Data, Ix2};
use rayon::prelude::*;

/// Spanning Test 汇总结果结构体。
#[derive(Debug, Clone, PartialEq)]
pub struct SpanningMetrics {
    pub valid_days: usize,
    pub effective_n: f64,
    pub mean_ic: f64,
    pub std_ic: f64,
    pub icir: f64,
    pub t_stat: f64,
    pub t_stat_adj: f64,
    pub daily_ic: Vec<f64>,
    pub cum_ic: Vec<f64>,
    pub valid_row_indices: Vec<usize>,
}

/// 在单个交易日截面上通过二次修正 Gram-Schmidt (MGS2) 计算 `target` 对 `benchmarks` 的正交残差与标签的 Spearman Rank IC。
/// 若有效样本不足或残差/标签退化为常数，返回 `f64::NAN`。
#[allow(clippy::needless_range_loop)]
pub fn calc_daily_spanning_ic(
    target_row: &[f64],
    bench_rows: &[&[f64]],
    label_row: &[f64],
    mask_row: &[bool],
    min_names: usize,
) -> f64 {
    let n = target_row.len();
    if label_row.len() != n || mask_row.len() != n {
        return f64::NAN;
    }
    let req_min = min_names.max(2).max(bench_rows.len() + 2);

    let mut valid_cols: Vec<usize> = Vec::with_capacity(n);
    'col_loop: for c in 0..n {
        if !mask_row[c] {
            continue;
        }
        let tv = target_row[c];
        let lv = label_row[c];
        if !tv.is_finite() || !lv.is_finite() {
            continue;
        }
        for br in bench_rows {
            if !br[c].is_finite() {
                continue 'col_loop;
            }
        }
        valid_cols.push(c);
    }

    let m = valid_cols.len();
    if m < req_min {
        return f64::NAN;
    }

    let mut resid: Vec<f64> = valid_cols.iter().map(|&c| target_row[c]).collect();
    let mf = m as f64;
    let orig_mean = resid.iter().sum::<f64>() / mf;
    let orig_var = resid
        .iter()
        .map(|&v| (v - orig_mean) * (v - orig_mean))
        .sum::<f64>()
        / mf;
    if !orig_var.is_finite() || orig_var <= 1e-24 {
        return f64::NAN;
    }

    let label_vals: Vec<f64> = valid_cols.iter().map(|&c| label_row[c]).collect();

    if !bench_rows.is_empty() {
        let mf = m as f64;
        let inv_sqrt_m = 1.0 / mf.sqrt();
        // 构建标准正交基 Q，q_0 = 截距项 1/sqrt(m)
        let mut basis: Vec<Vec<f64>> = Vec::with_capacity(bench_rows.len() + 1);
        let q0 = vec![inv_sqrt_m; m];
        let dot0: f64 = resid.iter().map(|&v| v * inv_sqrt_m).sum();
        for r in &mut resid {
            *r -= dot0 * inv_sqrt_m;
        }
        basis.push(q0);

        for br in bench_rows {
            let mut col_vec: Vec<f64> = valid_cols.iter().map(|&c| br[c]).collect();
            let mean_c = col_vec.iter().sum::<f64>() / mf;
            let mut var_c = 0.0_f64;
            for v in &col_vec {
                let d = *v - mean_c;
                var_c += d * d;
            }
            let std_c = (var_c / mf).sqrt();
            if !std_c.is_finite() || std_c <= 1e-12 {
                continue;
            }
            for v in &mut col_vec {
                *v = (*v - mean_c) / std_c;
            }

            // MGS2 正交化：对已有正交基投影两次以达到机器精度正交
            for _pass in 0..2 {
                for q in &basis {
                    let dot: f64 = col_vec.iter().zip(q.iter()).map(|(a, b)| a * b).sum();
                    for (v, &qi) in col_vec.iter_mut().zip(q.iter()) {
                        *v -= dot * qi;
                    }
                }
            }

            let norm_sq: f64 = col_vec.iter().map(|&v| v * v).sum();
            let norm = norm_sq.sqrt();
            // 若新方向与已有基准几乎完全共线，则忽略以免数值放大噪声
            if !norm.is_finite() || norm <= 1e-10 * mf.sqrt() {
                continue;
            }
            let inv_norm = 1.0 / norm;
            for v in &mut col_vec {
                *v *= inv_norm;
            }

            let proj: f64 = resid.iter().zip(col_vec.iter()).map(|(r, q)| r * q).sum();
            for (r, &qi) in resid.iter_mut().zip(col_vec.iter()) {
                *r -= proj * qi;
            }
            basis.push(col_vec);
        }
    }

    // 校验残差非退化常数（相对方差 < 1e-18 视为被基准完全线性张成）
    let r_mean = resid.iter().sum::<f64>() / mf;
    let r_var = resid
        .iter()
        .map(|&v| (v - r_mean) * (v - r_mean))
        .sum::<f64>()
        / mf;
    if !r_var.is_finite() || r_var <= 1e-24 || r_var <= 1e-18 * orig_var {
        return f64::NAN;
    }

    let mut r_ranks = vec![0.0_f64; m];
    let mut l_ranks = vec![0.0_f64; m];
    let mut buf = Vec::with_capacity(m);
    compute_fractional_ranks(&resid, &mut r_ranks, &mut buf);
    compute_fractional_ranks(&label_vals, &mut l_ranks, &mut buf);
    pearson_correlation(&r_ranks, &l_ranks)
}

/// 在全量 `(T, N)` 矩阵及行切片上并行计算多因子 Spanning Test 正交残差检验指标。
pub fn calc_spanning_metrics<S1, S2, S3>(
    target: &ArrayBase<S1, Ix2>,
    benchmarks: &[ArrayView2<'_, f64>],
    label: &ArrayBase<S2, Ix2>,
    mask: &ArrayBase<S3, Ix2>,
    min_names: usize,
    horizon: usize,
) -> SpanningMetrics
where
    S1: Data<Elem = f64> + Sync,
    S2: Data<Elem = f64> + Sync,
    S3: Data<Elem = bool> + Sync,
{
    let (t, n) = target.dim();
    assert_eq!(label.dim(), (t, n), "target 与 label 维度必须一致");
    assert_eq!(mask.dim(), (t, n), "target 与 mask 维度必须一致");
    for b in benchmarks {
        assert_eq!(b.dim(), (t, n), "benchmark 维度必须与 target 一致");
    }

    let daily_all: Vec<f64> = (0..t)
        .into_par_iter()
        .map(|row| {
            let mut t_row = vec![0.0_f64; n];
            let mut l_row = vec![0.0_f64; n];
            let mut m_row = vec![false; n];
            let mut b_storage: Vec<Vec<f64>> = vec![vec![0.0_f64; n]; benchmarks.len()];

            for col in 0..n {
                t_row[col] = target[[row, col]];
                l_row[col] = label[[row, col]];
                m_row[col] = mask[[row, col]];
            }
            for (bi, bm) in benchmarks.iter().enumerate() {
                for col in 0..n {
                    b_storage[bi][col] = bm[[row, col]];
                }
            }
            let b_refs: Vec<&[f64]> = b_storage.iter().map(|v| v.as_slice()).collect();
            calc_daily_spanning_ic(&t_row, &b_refs, &l_row, &m_row, min_names)
        })
        .collect();

    let mut valid_ics: Vec<f64> = Vec::with_capacity(t);
    let mut valid_rows: Vec<usize> = Vec::with_capacity(t);
    for (row, &val) in daily_all.iter().enumerate() {
        if val.is_finite() {
            valid_ics.push(val);
            valid_rows.push(row);
        }
    }

    let valid_days = valid_ics.len();
    let h = horizon.max(1) as f64;
    if valid_days == 0 {
        return SpanningMetrics {
            valid_days: 0,
            effective_n: 0.0,
            mean_ic: 0.0,
            std_ic: 0.0,
            icir: 0.0,
            t_stat: 0.0,
            t_stat_adj: 0.0,
            daily_ic: daily_all,
            cum_ic: Vec::new(),
            valid_row_indices: Vec::new(),
        };
    }

    let mean_ic = valid_ics.iter().sum::<f64>() / (valid_days as f64);
    let std_ic = if valid_days > 1 {
        let var = valid_ics
            .iter()
            .map(|&v| (v - mean_ic) * (v - mean_ic))
            .sum::<f64>()
            / ((valid_days - 1) as f64);
        var.sqrt()
    } else {
        0.0
    };

    let icir = if std_ic > 1e-12 {
        mean_ic / std_ic
    } else {
        0.0
    };
    let effective_n = (valid_days as f64) / h;
    let t_stat = if std_ic > 1e-12 && valid_days > 1 {
        icir * (valid_days as f64).sqrt()
    } else {
        0.0
    };
    let t_stat_adj = if std_ic > 1e-12 && valid_days > 1 {
        icir * effective_n.sqrt()
    } else {
        0.0
    };

    let mut cum_ic = Vec::with_capacity(valid_days);
    let mut acc = 0.0_f64;
    for &ic in &valid_ics {
        acc += ic;
        cum_ic.push(acc);
    }

    SpanningMetrics {
        valid_days,
        effective_n,
        mean_ic,
        std_ic,
        icir,
        t_stat,
        t_stat_adj,
        daily_ic: daily_all,
        cum_ic,
        valid_row_indices: valid_rows,
    }
}
