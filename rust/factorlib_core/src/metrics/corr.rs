//! 因子截面相关系数矩阵与去相关计算 (metrics/corr.rs)。
//!
//! 严格对应白皮书 8 章、9.4 节与 9.5 节规范：
//! - 计算因子间在有效标的池上的逐日截面相关性，并取时序均值；
//! - 支持 Spearman 秩相关（rank=true）与 Pearson 相关（rank=false）；
//! - 支持双因子快速对账以及 M 个因子的高并发相关性矩阵并行计算；
//! - 满足对角线恒为 1.0，对称矩阵性质 $\mathbf{C}_{j, k} = \mathbf{C}_{k, j}$。

use crate::metrics::ic::{compute_fractional_ranks, pearson_correlation};
use ndarray::{Array2, ArrayBase, Data, Ix2};
use rayon::prelude::*;

/// 可复用的日度相关系数计算临时缓冲区，避免循环内重复堆分配。
#[derive(Default, Clone)]
pub struct CorrScratch {
    x_vals: Vec<f64>,
    y_vals: Vec<f64>,
    x_ranks: Vec<f64>,
    y_ranks: Vec<f64>,
    rank_buf: Vec<(usize, f64)>,
}

/// 预转换的库内因子行连续表示 `(T, N)`（用于 GP 批评估零分配去相关）。
pub struct PreparedFactor {
    pub t: usize,
    pub n: usize,
    /// 行优先存储的数值矩阵（非有限值统一置 `NaN`）
    vals_row_major: Vec<f64>,
    /// 每行对应 `mask && finite` 掩码字节（0/1）
    valid_row_major: Vec<u8>,
    /// 每行有效标的数量
    valid_counts: Vec<usize>,
    /// 行优先存储的归一化秩（均值 0、L2 范数 1），仅当该行有效且方差非零时有意义
    norm_ranks_row_major: Vec<f64>,
    /// 标记每行预秩向量是否有效（`valid_count >= min_names.max(2)` 且非全常数）
    row_rank_ok: Vec<bool>,
}

impl PreparedFactor {
    /// 从 `(T, N)` 因子矩阵与在池掩码构建行连续预备表示（含归一化秩快路径缓存）。
    pub fn from_matrix<S1, S2>(
        mat: &ArrayBase<S1, Ix2>,
        mask: &ArrayBase<S2, Ix2>,
        min_names: usize,
        compute_ranks: bool,
    ) -> Self
    where
        S1: Data<Elem = f64>,
        S2: Data<Elem = bool>,
    {
        let (t, n) = mat.dim();
        assert_eq!(mask.dim(), (t, n), "因子矩阵与 mask 形状必须一致");

        let total = t * n;
        let mut vals_row_major = vec![f64::NAN; total];
        let mut valid_row_major = vec![0u8; total];
        let mut valid_counts = vec![0usize; t];
        let mut norm_ranks_row_major = if compute_ranks {
            vec![0.0_f64; total]
        } else {
            Vec::new()
        };
        let mut row_rank_ok = vec![false; t];

        let mut row_vals = Vec::with_capacity(n);
        let mut row_cols = Vec::with_capacity(n);
        let mut row_ranks = Vec::with_capacity(n);
        let mut rank_buf = Vec::with_capacity(n);

        for r in 0..t {
            let base = r * n;
            row_vals.clear();
            row_cols.clear();
            for c in 0..n {
                let v = mat[[r, c]];
                let fv = if v.is_finite() { v } else { f64::NAN };
                vals_row_major[base + c] = fv;
                if mask[[r, c]] && fv.is_finite() {
                    valid_row_major[base + c] = 1;
                    row_vals.push(fv);
                    row_cols.push(c);
                }
            }
            let k = row_vals.len();
            valid_counts[r] = k;

            if compute_ranks && k >= min_names && k >= 2 {
                row_ranks.resize(k, 0.0);
                compute_fractional_ranks(&row_vals, &mut row_ranks, &mut rank_buf);
                let kf = k as f64;
                let mean = row_ranks.iter().sum::<f64>() / kf;
                let mut var_sum = 0.0_f64;
                for rk in &row_ranks {
                    let d = *rk - mean;
                    var_sum += d * d;
                }
                if var_sum > 1e-14 {
                    let inv_norm = 1.0 / var_sum.sqrt();
                    for (idx, &col) in row_cols.iter().enumerate() {
                        norm_ranks_row_major[base + col] = (row_ranks[idx] - mean) * inv_norm;
                    }
                    row_rank_ok[r] = true;
                }
            }
        }

        Self {
            t,
            n,
            vals_row_major,
            valid_row_major,
            valid_counts,
            norm_ranks_row_major,
            row_rank_ok,
        }
    }

    #[inline]
    pub fn row_vals(&self, row: usize) -> &[f64] {
        let base = row * self.n;
        &self.vals_row_major[base..base + self.n]
    }

    #[inline]
    pub fn row_valid(&self, row: usize) -> &[u8] {
        let base = row * self.n;
        &self.valid_row_major[base..base + self.n]
    }

    #[inline]
    pub fn row_norm_ranks(&self, row: usize) -> &[f64] {
        let base = row * self.n;
        &self.norm_ranks_row_major[base..base + self.n]
    }
}

/// 可复用的候选因子日度预备缓冲区（用于 GP 批评估零堆分配去相关）。
#[derive(Default)]
pub struct PreparedCandidate {
    pub t: usize,
    pub n: usize,
    vals_row_major: Vec<f64>,
    valid_row_major: Vec<u8>,
    valid_counts: Vec<usize>,
    norm_ranks_row_major: Vec<f64>,
    row_rank_ok: Vec<bool>,
    row_vals_scratch: Vec<f64>,
    row_cols_scratch: Vec<usize>,
    row_ranks_scratch: Vec<f64>,
    rank_buf_scratch: Vec<(usize, f64)>,
}

impl PreparedCandidate {
    /// 将 `(T, N)` 候选矩阵切片填充入复用缓冲区（无重复分配）。
    pub fn fill_from_matrix<S: Data<Elem = f64>>(
        &mut self,
        mat: &ArrayBase<S, Ix2>,
        mask_row_major: &[bool],
        min_names: usize,
        compute_ranks: bool,
    ) {
        let (t, n) = mat.dim();
        assert_eq!(mask_row_major.len(), t * n);
        self.t = t;
        self.n = n;
        let total = t * n;

        self.vals_row_major.resize(total, f64::NAN);
        self.valid_row_major.resize(total, 0);
        self.valid_counts.resize(t, 0);
        self.row_rank_ok.resize(t, false);
        if compute_ranks {
            self.norm_ranks_row_major.resize(total, 0.0);
        }

        for r in 0..t {
            let base = r * n;
            self.row_vals_scratch.clear();
            self.row_cols_scratch.clear();
            for c in 0..n {
                let v = mat[[r, c]];
                let fv = if v.is_finite() { v } else { f64::NAN };
                self.vals_row_major[base + c] = fv;
                if mask_row_major[base + c] && fv.is_finite() {
                    self.valid_row_major[base + c] = 1;
                    self.row_vals_scratch.push(fv);
                    self.row_cols_scratch.push(c);
                } else {
                    self.valid_row_major[base + c] = 0;
                    if compute_ranks {
                        self.norm_ranks_row_major[base + c] = 0.0;
                    }
                }
            }
            let k = self.row_vals_scratch.len();
            self.valid_counts[r] = k;
            self.row_rank_ok[r] = false;

            if compute_ranks && k >= min_names && k >= 2 {
                self.row_ranks_scratch.resize(k, 0.0);
                compute_fractional_ranks(
                    &self.row_vals_scratch,
                    &mut self.row_ranks_scratch,
                    &mut self.rank_buf_scratch,
                );
                let kf = k as f64;
                let mean = self.row_ranks_scratch.iter().sum::<f64>() / kf;
                let mut var_sum = 0.0_f64;
                for rk in &self.row_ranks_scratch {
                    let d = *rk - mean;
                    var_sum += d * d;
                }
                if var_sum > 1e-14 {
                    let inv_norm = 1.0 / var_sum.sqrt();
                    for (idx, &col) in self.row_cols_scratch.iter().enumerate() {
                        self.norm_ranks_row_major[base + col] =
                            (self.row_ranks_scratch[idx] - mean) * inv_norm;
                    }
                    self.row_rank_ok[r] = true;
                }
            }
        }
    }
}

/// 零堆分配计算候选因子与预备库内因子之间的全时段平均横截面相关系数。
pub fn cross_correlation_prepared_seq(
    cand: &PreparedCandidate,
    lib: &PreparedFactor,
    min_names: usize,
    rank: bool,
    scratch: &mut CorrScratch,
) -> f64 {
    let t = cand.t;
    let n = cand.n;
    assert_eq!((lib.t, lib.n), (t, n), "因子矩阵形状必须一致");

    let mut sum_corr = 0.0_f64;
    let mut valid_days = 0usize;

    for r in 0..t {
        let base = r * n;
        let c_valid = &cand.valid_row_major[base..base + n];
        let l_valid = lib.row_valid(r);

        let rho = if rank
            && cand.valid_counts[r] == lib.valid_counts[r]
            && cand.valid_counts[r] >= min_names
            && cand.valid_counts[r] >= 2
            && c_valid == l_valid
        {
            if !cand.row_rank_ok[r] || !lib.row_rank_ok[r] {
                f64::NAN
            } else {
                let ca = &cand.norm_ranks_row_major[base..base + n];
                let cb = lib.row_norm_ranks(r);
                let mut dot = 0.0_f64;
                for c in 0..n {
                    dot += ca[c] * cb[c];
                }
                dot.clamp(-1.0, 1.0)
            }
        } else {
            let ca_vals = &cand.vals_row_major[base..base + n];
            let lb_vals = lib.row_vals(r);
            scratch.x_vals.clear();
            scratch.y_vals.clear();
            for c in 0..n {
                if c_valid[c] != 0 && l_valid[c] != 0 {
                    scratch.x_vals.push(ca_vals[c]);
                    scratch.y_vals.push(lb_vals[c]);
                }
            }
            let k = scratch.x_vals.len();
            if k < min_names || k < 2 {
                f64::NAN
            } else if rank {
                scratch.x_ranks.resize(k, 0.0);
                scratch.y_ranks.resize(k, 0.0);
                compute_fractional_ranks(
                    &scratch.x_vals,
                    &mut scratch.x_ranks,
                    &mut scratch.rank_buf,
                );
                compute_fractional_ranks(
                    &scratch.y_vals,
                    &mut scratch.y_ranks,
                    &mut scratch.rank_buf,
                );
                pearson_correlation(&scratch.x_ranks, &scratch.y_ranks)
            } else {
                pearson_correlation(&scratch.x_vals, &scratch.y_vals)
            }
        };

        if rho.is_finite() {
            sum_corr += rho;
            valid_days += 1;
        }
    }

    if valid_days == 0 {
        0.0
    } else {
        sum_corr / (valid_days as f64)
    }
}

/// 复用 `CorrScratch` 计算两个因子在单日截面共同有效标的上的相关系数（零额外堆分配）。
pub fn calc_daily_factor_correlation_with_scratch(
    a_row: &[f64],
    b_row: &[f64],
    mask_row: &[bool],
    min_names: usize,
    rank: bool,
    scratch: &mut CorrScratch,
) -> f64 {
    let n = a_row.len();
    scratch.x_vals.clear();
    scratch.y_vals.clear();

    for i in 0..n {
        if mask_row[i] {
            let va = a_row[i];
            let vb = b_row[i];
            if va.is_finite() && vb.is_finite() {
                scratch.x_vals.push(va);
                scratch.y_vals.push(vb);
            }
        }
    }

    let k = scratch.x_vals.len();
    if k < min_names || k < 2 {
        return f64::NAN;
    }

    if rank {
        scratch.x_ranks.resize(k, 0.0);
        scratch.y_ranks.resize(k, 0.0);
        compute_fractional_ranks(&scratch.x_vals, &mut scratch.x_ranks, &mut scratch.rank_buf);
        compute_fractional_ranks(&scratch.y_vals, &mut scratch.y_ranks, &mut scratch.rank_buf);
        pearson_correlation(&scratch.x_ranks, &scratch.y_ranks)
    } else {
        pearson_correlation(&scratch.x_vals, &scratch.y_vals)
    }
}

/// 计算两个因子在单日截面共同有效标的上的相关系数。
pub fn calc_daily_factor_correlation(
    a_row: &[f64],
    b_row: &[f64],
    mask_row: &[bool],
    min_names: usize,
    rank: bool,
) -> f64 {
    let mut scratch = CorrScratch::default();
    calc_daily_factor_correlation_with_scratch(
        a_row,
        b_row,
        mask_row,
        min_names,
        rank,
        &mut scratch,
    )
}

/// 顺序计算两个因子矩阵之间的全时段平均横截面相关系数（零堆分配，供外层已并行的批处理调用）。
pub fn cross_correlation_pairwise_seq<S1, S2, S3>(
    a: &ArrayBase<S1, Ix2>,
    b: &ArrayBase<S2, Ix2>,
    mask: &ArrayBase<S3, Ix2>,
    min_names: usize,
    rank: bool,
    scratch: &mut CorrScratch,
) -> f64
where
    S1: Data<Elem = f64>,
    S2: Data<Elem = f64>,
    S3: Data<Elem = bool>,
{
    let (t, n) = a.dim();
    assert_eq!(b.dim(), (t, n), "因子矩阵形状必须一致");
    assert_eq!(mask.dim(), (t, n), "mask 矩阵形状必须一致");

    let mut sum_corr = 0.0_f64;
    let mut valid_days = 0usize;

    for row in 0..t {
        scratch.x_vals.clear();
        scratch.y_vals.clear();
        for col in 0..n {
            if mask[[row, col]] {
                let va = a[[row, col]];
                let vb = b[[row, col]];
                if va.is_finite() && vb.is_finite() {
                    scratch.x_vals.push(va);
                    scratch.y_vals.push(vb);
                }
            }
        }
        let k = scratch.x_vals.len();
        if k < min_names || k < 2 {
            continue;
        }
        let rho = if rank {
            scratch.x_ranks.resize(k, 0.0);
            scratch.y_ranks.resize(k, 0.0);
            compute_fractional_ranks(&scratch.x_vals, &mut scratch.x_ranks, &mut scratch.rank_buf);
            compute_fractional_ranks(&scratch.y_vals, &mut scratch.y_ranks, &mut scratch.rank_buf);
            pearson_correlation(&scratch.x_ranks, &scratch.y_ranks)
        } else {
            pearson_correlation(&scratch.x_vals, &scratch.y_vals)
        };
        if rho.is_finite() {
            sum_corr += rho;
            valid_days += 1;
        }
    }

    if valid_days == 0 {
        0.0
    } else {
        sum_corr / (valid_days as f64)
    }
}

/// 并行计算两个因子矩阵之间的全时段平均横截面相关系数。
pub fn cross_correlation_pairwise<S1, S2, S3>(
    a: &ArrayBase<S1, Ix2>,
    b: &ArrayBase<S2, Ix2>,
    mask: &ArrayBase<S3, Ix2>,
    min_names: usize,
    rank: bool,
) -> f64
where
    S1: Data<Elem = f64> + Sync,
    S2: Data<Elem = f64> + Sync,
    S3: Data<Elem = bool> + Sync,
{
    let (t, n) = a.dim();
    assert_eq!(b.dim(), (t, n), "因子矩阵形状必须一致");
    assert_eq!(mask.dim(), (t, n), "mask 矩阵形状必须一致");

    let daily_corrs: Vec<f64> = (0..t)
        .into_par_iter()
        .map_init(CorrScratch::default, |scratch, row| {
            scratch.x_vals.clear();
            scratch.y_vals.clear();
            for col in 0..n {
                if mask[[row, col]] {
                    let va = a[[row, col]];
                    let vb = b[[row, col]];
                    if va.is_finite() && vb.is_finite() {
                        scratch.x_vals.push(va);
                        scratch.y_vals.push(vb);
                    }
                }
            }
            let k = scratch.x_vals.len();
            if k < min_names || k < 2 {
                return f64::NAN;
            }
            if rank {
                scratch.x_ranks.resize(k, 0.0);
                scratch.y_ranks.resize(k, 0.0);
                compute_fractional_ranks(
                    &scratch.x_vals,
                    &mut scratch.x_ranks,
                    &mut scratch.rank_buf,
                );
                compute_fractional_ranks(
                    &scratch.y_vals,
                    &mut scratch.y_ranks,
                    &mut scratch.rank_buf,
                );
                pearson_correlation(&scratch.x_ranks, &scratch.y_ranks)
            } else {
                pearson_correlation(&scratch.x_vals, &scratch.y_vals)
            }
        })
        .collect();

    let mut sum = 0.0_f64;
    let mut cnt = 0usize;
    for v in daily_corrs {
        if v.is_finite() {
            sum += v;
            cnt += 1;
        }
    }
    if cnt == 0 {
        0.0
    } else {
        sum / (cnt as f64)
    }
}

/// 并行计算 M 个因子之间的 M x M 截面相关系数矩阵（外层按因子对并行，内层顺序复用缓冲区）。
pub fn cross_correlation_matrix<S1, S3>(
    factors: &[&ArrayBase<S1, Ix2>],
    mask: &ArrayBase<S3, Ix2>,
    min_names: usize,
    rank: bool,
) -> Array2<f64>
where
    S1: Data<Elem = f64> + Sync,
    S3: Data<Elem = bool> + Sync,
{
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

    // 预转换为行连续表示，并在外层因子对级别并行计算（消除嵌套并行爆炸）
    let prepared: Vec<PreparedFactor> = factors
        .par_iter()
        .map(|f| PreparedFactor::from_matrix(*f, mask, min_names, rank))
        .collect();

    let mut pairs = Vec::with_capacity(m * (m - 1) / 2);
    for i in 0..m {
        for j in (i + 1)..m {
            pairs.push((i, j));
        }
    }

    let pair_results: Vec<((usize, usize), f64)> = pairs
        .into_par_iter()
        .map_init(CorrScratch::default, |scratch, (i, j)| {
            let mut sum_corr = 0.0_f64;
            let mut valid_days = 0usize;
            let pa = &prepared[i];
            let pb = &prepared[j];
            for r in 0..pa.t {
                let va = pa.row_valid(r);
                let vb = pb.row_valid(r);
                let rho = if rank
                    && pa.valid_counts[r] == pb.valid_counts[r]
                    && pa.valid_counts[r] >= min_names
                    && pa.valid_counts[r] >= 2
                    && va == vb
                {
                    if !pa.row_rank_ok[r] || !pb.row_rank_ok[r] {
                        f64::NAN
                    } else {
                        let ra = pa.row_norm_ranks(r);
                        let rb = pb.row_norm_ranks(r);
                        let mut dot = 0.0_f64;
                        for c in 0..pa.n {
                            dot += ra[c] * rb[c];
                        }
                        dot.clamp(-1.0, 1.0)
                    }
                } else {
                    let xa = pa.row_vals(r);
                    let xb = pb.row_vals(r);
                    scratch.x_vals.clear();
                    scratch.y_vals.clear();
                    for c in 0..pa.n {
                        if va[c] != 0 && vb[c] != 0 {
                            scratch.x_vals.push(xa[c]);
                            scratch.y_vals.push(xb[c]);
                        }
                    }
                    let k = scratch.x_vals.len();
                    if k < min_names || k < 2 {
                        f64::NAN
                    } else if rank {
                        scratch.x_ranks.resize(k, 0.0);
                        scratch.y_ranks.resize(k, 0.0);
                        compute_fractional_ranks(
                            &scratch.x_vals,
                            &mut scratch.x_ranks,
                            &mut scratch.rank_buf,
                        );
                        compute_fractional_ranks(
                            &scratch.y_vals,
                            &mut scratch.y_ranks,
                            &mut scratch.rank_buf,
                        );
                        pearson_correlation(&scratch.x_ranks, &scratch.y_ranks)
                    } else {
                        pearson_correlation(&scratch.x_vals, &scratch.y_vals)
                    }
                };
                if rho.is_finite() {
                    sum_corr += rho;
                    valid_days += 1;
                }
            }
            let mean_rho = if valid_days == 0 {
                0.0
            } else {
                sum_corr / (valid_days as f64)
            };
            ((i, j), mean_rho)
        })
        .collect();

    for ((i, j), rho) in pair_results {
        out[[i, j]] = rho;
        out[[j, i]] = rho;
    }

    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::ShapeBuilder;

    #[test]
    fn test_prepared_corr_exact_parity_with_nans_and_ties() {
        let t = 15;
        let n = 12;
        let mut a = Array2::<f64>::zeros((t, n).f());
        let mut b = Array2::<f64>::zeros((t, n).f());
        let mut mask = Array2::<bool>::from_elem((t, n).f(), true);

        for r in 0..t {
            for c in 0..n {
                a[[r, c]] = ((r * 7 + c * 3) % 5) as f64;
                b[[r, c]] = ((r * 5 + c * 11) % 7) as f64;
                if (r + c) % 9 == 0 {
                    a[[r, c]] = f64::NAN;
                }
                if (r * 3 + c) % 11 == 0 {
                    b[[r, c]] = f64::NAN;
                }
                if c == 0 && r % 2 == 0 {
                    mask[[r, c]] = false;
                }
            }
        }

        let expected_spearman = cross_correlation_pairwise(&a, &b, &mask, 4, true);
        let expected_pearson = cross_correlation_pairwise(&a, &b, &mask, 4, false);

        let mask_ref = &mask;
        let mask_rm: Vec<bool> = (0..t)
            .flat_map(|r| (0..n).map(move |c| mask_ref[[r, c]]))
            .collect();
        let lib_s = PreparedFactor::from_matrix(&b, &mask, 4, true);
        let mut cand_s = PreparedCandidate::default();
        cand_s.fill_from_matrix(&a, &mask_rm, 4, true);
        let mut scratch = CorrScratch::default();
        let got_spearman = cross_correlation_prepared_seq(&cand_s, &lib_s, 4, true, &mut scratch);
        assert!(
            (got_spearman - expected_spearman).abs() < 1e-12,
            "Spearman parity mismatch: {got_spearman} vs {expected_spearman}"
        );

        let lib_p = PreparedFactor::from_matrix(&b, &mask, 4, false);
        let mut cand_p = PreparedCandidate::default();
        cand_p.fill_from_matrix(&a, &mask_rm, 4, false);
        let got_pearson = cross_correlation_prepared_seq(&cand_p, &lib_p, 4, false, &mut scratch);
        assert!(
            (got_pearson - expected_pearson).abs() < 1e-12,
            "Pearson parity mismatch: {got_pearson} vs {expected_pearson}"
        );

        let mat = cross_correlation_matrix(&[&a, &b], &mask, 4, true);
        assert!((mat[[0, 1]] - expected_spearman).abs() < 1e-12);
        assert!((mat[[1, 0]] - expected_spearman).abs() < 1e-12);
        assert!((mat[[0, 0]] - 1.0).abs() < 1e-12);
    }
}
