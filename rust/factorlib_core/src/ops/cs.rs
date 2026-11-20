//! 横截面算子实现 (ops/cs.rs)。
//!
//! 针对 Fortran 列优先 `(T, N)` 矩阵进行了极致的 CPU L1 Cache 局部性优化：
//! - `cs_demean`, `cs_zscore`, `cs_scale`, `cs_neutralize`：采用按列顺序扫描累加到 L1 常驻行统计向量
//!   （长度 $T=4000$ 仅占 32KB），实现 100% 步长 1（Stride-1）连续内存读写，彻底消除跨列跳跃 Cache Miss；
//! - `cs_rank`, `cs_winsorize`：采用分块转置（Tiled Transpose）至行连续缓冲区后计算，再分块写回列优先矩阵。

use crate::nan::{clean_f64, cmp_f64, compute_normalized_ranks, EPS, STD_EPS, VAR_EPS};
use crate::panel::alloc_nan_matrix;
use ndarray::Array2;

const TILE: usize = 32;

/// 分块将 Fortran `(T, N)` 矩阵转置为行连续 `Vec<f64>`（形状 `T x N` C-order）。
fn transpose_f_to_c(x: &Array2<f64>) -> Vec<f64> {
    let (t, n) = x.dim();
    let mut c_buf = vec![f64::NAN; t * n];
    for c0 in (0..n).step_by(TILE) {
        let c_end = (c0 + TILE).min(n);
        for r0 in (0..t).step_by(TILE) {
            let r_end = (r0 + TILE).min(t);
            for col in c0..c_end {
                let xc = x.column(col);
                for row in r0..r_end {
                    c_buf[row * n + col] = xc[row];
                }
            }
        }
    }
    c_buf
}

/// 分块将行连续 `c_buf`（`T x N` C-order）转置回 Fortran `(T, N)` 矩阵。
fn transpose_c_to_f(c_buf: &[f64], t: usize, n: usize) -> Array2<f64> {
    let mut out = alloc_nan_matrix(t, n);
    for c0 in (0..n).step_by(TILE) {
        let c_end = (c0 + TILE).min(n);
        for r0 in (0..t).step_by(TILE) {
            let r_end = (r0 + TILE).min(t);
            for col in c0..c_end {
                let mut oc = out.column_mut(col);
                for row in r0..r_end {
                    oc[row] = c_buf[row * n + col];
                }
            }
        }
    }
    out
}

#[inline]
fn slice_median_sorted(sorted: &[f64]) -> f64 {
    let k = sorted.len();
    if k == 0 {
        f64::NAN
    } else if k % 2 == 1 {
        sorted[k / 2]
    } else {
        0.5 * (sorted[k / 2 - 1] + sorted[k / 2])
    }
}

pub fn op_cs_rank(x: &Array2<f64>) -> Array2<f64> {
    let (t, n) = x.dim();
    if t == 0 || n == 0 {
        return alloc_nan_matrix(t, n);
    }
    let c_in = transpose_f_to_c(x);
    let mut c_out = vec![f64::NAN; t * n];
    let mut buf: Vec<(usize, f64)> = Vec::with_capacity(n);

    for row in 0..t {
        let r_slice = &c_in[row * n..(row + 1) * n];
        let o_slice = &mut c_out[row * n..(row + 1) * n];
        compute_normalized_ranks(r_slice, o_slice, &mut buf);
    }
    transpose_c_to_f(&c_out, t, n)
}

pub fn op_cs_demean(x: &Array2<f64>) -> Array2<f64> {
    let (t, n) = x.dim();
    let mut out = alloc_nan_matrix(t, n);
    if t == 0 || n == 0 {
        return out;
    }
    let mut sums = vec![0.0_f64; t];
    let mut cnts = vec![0u32; t];

    for col in 0..n {
        let xc = x.column(col);
        for row in 0..t {
            let v = xc[row];
            if v.is_finite() {
                sums[row] += v;
                cnts[row] += 1;
            }
        }
    }

    let mut means = vec![f64::NAN; t];
    for row in 0..t {
        if cnts[row] >= 1 {
            let m = sums[row] / (cnts[row] as f64);
            if m.is_finite() {
                means[row] = m;
            }
        }
    }

    for col in 0..n {
        let xc = x.column(col);
        let mut oc = out.column_mut(col);
        for row in 0..t {
            let v = xc[row];
            let m = means[row];
            if v.is_finite() && m.is_finite() {
                oc[row] = clean_f64(v - m);
            }
        }
    }
    out
}

pub fn op_cs_zscore(x: &Array2<f64>) -> Array2<f64> {
    let (t, n) = x.dim();
    let mut out = alloc_nan_matrix(t, n);
    if t == 0 || n == 0 {
        return out;
    }
    let mut sums = vec![0.0_f64; t];
    let mut cnts = vec![0u32; t];

    for col in 0..n {
        let xc = x.column(col);
        for row in 0..t {
            let v = xc[row];
            if v.is_finite() {
                sums[row] += v;
                cnts[row] += 1;
            }
        }
    }

    let mut means = vec![f64::NAN; t];
    for row in 0..t {
        if cnts[row] >= 2 {
            let m = sums[row] / (cnts[row] as f64);
            if m.is_finite() {
                means[row] = m;
            }
        }
    }

    let mut m2s = vec![0.0_f64; t];
    for col in 0..n {
        let xc = x.column(col);
        for row in 0..t {
            let v = xc[row];
            let m = means[row];
            if v.is_finite() && m.is_finite() {
                let d = v - m;
                m2s[row] += d * d;
            }
        }
    }

    let mut inv_stds = vec![f64::NAN; t];
    for row in 0..t {
        let m = means[row];
        if m.is_finite() {
            let std = (m2s[row] / ((cnts[row] - 1) as f64)).sqrt();
            let scale = 1.0_f64.max(m.abs());
            if std.is_finite() && std > STD_EPS * scale {
                inv_stds[row] = std;
            }
        }
    }

    for col in 0..n {
        let xc = x.column(col);
        let mut oc = out.column_mut(col);
        for row in 0..t {
            let v = xc[row];
            let std = inv_stds[row];
            if v.is_finite() && std.is_finite() {
                oc[row] = clean_f64((v - means[row]) / std);
            }
        }
    }
    out
}

pub fn op_cs_scale(x: &Array2<f64>) -> Array2<f64> {
    let (t, n) = x.dim();
    let mut out = alloc_nan_matrix(t, n);
    if t == 0 || n == 0 {
        return out;
    }
    let mut abs_sums = vec![0.0_f64; t];
    for col in 0..n {
        let xc = x.column(col);
        for row in 0..t {
            let v = xc[row];
            if v.is_finite() {
                abs_sums[row] += v.abs();
            }
        }
    }
    for col in 0..n {
        let xc = x.column(col);
        let mut oc = out.column_mut(col);
        for row in 0..t {
            let v = xc[row];
            let s = abs_sums[row];
            if v.is_finite() && s.is_finite() && s >= EPS {
                oc[row] = clean_f64(v / s);
            }
        }
    }
    out
}

pub fn op_cs_winsorize(x: &Array2<f64>) -> Array2<f64> {
    let (t, n) = x.dim();
    if t == 0 || n == 0 {
        return alloc_nan_matrix(t, n);
    }
    let c_in = transpose_f_to_c(x);
    let mut c_out = vec![f64::NAN; t * n];
    let mut vals: Vec<f64> = Vec::with_capacity(n);
    let mut devs: Vec<f64> = Vec::with_capacity(n);

    for row in 0..t {
        let r_slice = &c_in[row * n..(row + 1) * n];
        let o_slice = &mut c_out[row * n..(row + 1) * n];
        vals.clear();
        for &v in r_slice {
            if v.is_finite() {
                vals.push(v);
            }
        }
        if vals.is_empty() {
            continue;
        }
        vals.sort_unstable_by(|a, b| cmp_f64(*a, *b));
        let med = slice_median_sorted(&vals);
        devs.clear();
        for &v in &vals {
            devs.push((v - med).abs());
        }
        devs.sort_unstable_by(|a, b| cmp_f64(*a, *b));
        let mad = slice_median_sorted(&devs);
        let radius = 5.0 * 1.4826 * mad;
        let lo = med - radius;
        let hi = med + radius;
        if lo.is_finite() && hi.is_finite() && lo <= hi {
            for col in 0..n {
                let v = r_slice[col];
                if v.is_finite() {
                    o_slice[col] = clean_f64(v.clamp(lo, hi));
                }
            }
        }
    }
    transpose_c_to_f(&c_out, t, n)
}

/// 横截面中性化：对每一行截面，将 `y` 对 `x` 做带截距 OLS 回归并输出残差。
pub fn op_cs_neutralize(y: &Array2<f64>, x: &Array2<f64>) -> Array2<f64> {
    let (t, n) = y.dim();
    debug_assert_eq!((t, n), x.dim());
    let mut out = alloc_nan_matrix(t, n);
    if t == 0 || n == 0 {
        return out;
    }

    let mut sum_x = vec![0.0_f64; t];
    let mut sum_y = vec![0.0_f64; t];
    let mut cnts = vec![0u32; t];

    for col in 0..n {
        let xc = x.column(col);
        let yc = y.column(col);
        for row in 0..t {
            let vx = xc[row];
            let vy = yc[row];
            if vx.is_finite() && vy.is_finite() {
                sum_x[row] += vx;
                sum_y[row] += vy;
                cnts[row] += 1;
            }
        }
    }

    let mut mx = vec![f64::NAN; t];
    let mut my = vec![f64::NAN; t];
    for row in 0..t {
        if cnts[row] >= 2 {
            let kf = cnts[row] as f64;
            mx[row] = sum_x[row] / kf;
            my[row] = sum_y[row] / kf;
        }
    }

    let mut c_xy = vec![0.0_f64; t];
    let mut m2_x = vec![0.0_f64; t];
    for col in 0..n {
        let xc = x.column(col);
        let yc = y.column(col);
        for row in 0..t {
            let vx = xc[row];
            let vy = yc[row];
            if vx.is_finite() && vy.is_finite() && mx[row].is_finite() && my[row].is_finite() {
                let dx = vx - mx[row];
                let dy = vy - my[row];
                c_xy[row] += dx * dy;
                m2_x[row] += dx * dx;
            }
        }
    }

    let mut betas = vec![f64::NAN; t];
    for row in 0..t {
        if mx[row].is_finite() && my[row].is_finite() {
            let kf = cnts[row] as f64;
            let sx = 1.0_f64.max(mx[row].abs());
            if m2_x[row].is_finite() && c_xy[row].is_finite() && m2_x[row] > VAR_EPS * kf * sx * sx
            {
                betas[row] = c_xy[row] / m2_x[row];
            }
        }
    }

    for col in 0..n {
        let xc = x.column(col);
        let yc = y.column(col);
        let mut oc = out.column_mut(col);
        for row in 0..t {
            let vx = xc[row];
            let vy = yc[row];
            let beta = betas[row];
            if vx.is_finite() && vy.is_finite() && beta.is_finite() {
                oc[row] = clean_f64((vy - my[row]) - beta * (vx - mx[row]));
            }
        }
    }
    out
}

// ============================================================================
// 9. FactorLab 1d 横截面统计扩充算子
// ============================================================================

/// 逐行广播写入：仅当 `x[row, col]` 有限时写入 `agg[row]`。
#[inline]
fn broadcast_row_agg(x: &Array2<f64>, agg: &[f64]) -> Array2<f64> {
    let (t, n) = x.dim();
    let mut out = alloc_nan_matrix(t, n);
    for col in 0..n {
        let xc = x.column(col);
        let mut oc = out.column_mut(col);
        for row in 0..t {
            let v = xc[row];
            let a = agg[row];
            if v.is_finite() && a.is_finite() {
                oc[row] = a;
            }
        }
    }
    out
}

/// `cs_sum(x)`：每个交易时点股票截面有限值总和。
pub fn op_cs_sum(x: &Array2<f64>) -> Array2<f64> {
    let (t, n) = x.dim();
    let mut agg = vec![f64::NAN; t];
    let mut sums = vec![0.0_f64; t];
    let mut cnts = vec![0u32; t];
    for col in 0..n {
        let xc = x.column(col);
        for row in 0..t {
            let v = xc[row];
            if v.is_finite() {
                sums[row] += v;
                cnts[row] += 1;
            }
        }
    }
    for row in 0..t {
        if cnts[row] >= 1 {
            agg[row] = clean_f64(sums[row]);
        }
    }
    broadcast_row_agg(x, &agg)
}

/// `cs_mean(x)`：每个交易时点股票截面算术均值。
pub fn op_cs_mean(x: &Array2<f64>) -> Array2<f64> {
    let (t, n) = x.dim();
    let mut agg = vec![f64::NAN; t];
    let mut sums = vec![0.0_f64; t];
    let mut cnts = vec![0u32; t];
    for col in 0..n {
        let xc = x.column(col);
        for row in 0..t {
            let v = xc[row];
            if v.is_finite() {
                sums[row] += v;
                cnts[row] += 1;
            }
        }
    }
    for row in 0..t {
        if cnts[row] >= 1 {
            agg[row] = clean_f64(sums[row] / (cnts[row] as f64));
        }
    }
    broadcast_row_agg(x, &agg)
}

/// `cs_var(x)`：截面总体方差（`ddof=0`）。
pub fn op_cs_var(x: &Array2<f64>) -> Array2<f64> {
    let (t, n) = x.dim();
    let mut means = vec![f64::NAN; t];
    let mut m2s = vec![0.0_f64; t];
    let mut cnts = vec![0u32; t];
    let mut sums = vec![0.0_f64; t];
    for col in 0..n {
        let xc = x.column(col);
        for row in 0..t {
            let v = xc[row];
            if v.is_finite() {
                sums[row] += v;
                cnts[row] += 1;
            }
        }
    }
    for row in 0..t {
        if cnts[row] >= 1 {
            means[row] = sums[row] / (cnts[row] as f64);
        }
    }
    for col in 0..n {
        let xc = x.column(col);
        for row in 0..t {
            let v = xc[row];
            let m = means[row];
            if v.is_finite() && m.is_finite() {
                let d = v - m;
                m2s[row] += d * d;
            }
        }
    }
    let mut agg = vec![f64::NAN; t];
    for row in 0..t {
        if means[row].is_finite() && cnts[row] >= 1 {
            agg[row] = clean_f64(m2s[row] / (cnts[row] as f64));
        }
    }
    broadcast_row_agg(x, &agg)
}

/// `cs_std(x)`：截面总体标准差（`ddof=0`），与 `zScore` 后处理口径一致。
pub fn op_cs_std(x: &Array2<f64>) -> Array2<f64> {
    let (t, n) = x.dim();
    let mut means = vec![f64::NAN; t];
    let mut m2s = vec![0.0_f64; t];
    let mut cnts = vec![0u32; t];
    let mut sums = vec![0.0_f64; t];
    for col in 0..n {
        let xc = x.column(col);
        for row in 0..t {
            let v = xc[row];
            if v.is_finite() {
                sums[row] += v;
                cnts[row] += 1;
            }
        }
    }
    for row in 0..t {
        if cnts[row] >= 1 {
            means[row] = sums[row] / (cnts[row] as f64);
        }
    }
    for col in 0..n {
        let xc = x.column(col);
        for row in 0..t {
            let v = xc[row];
            let m = means[row];
            if v.is_finite() && m.is_finite() {
                let d = v - m;
                m2s[row] += d * d;
            }
        }
    }
    let mut agg = vec![f64::NAN; t];
    for row in 0..t {
        if means[row].is_finite() && cnts[row] >= 1 {
            agg[row] = clean_f64((m2s[row] / (cnts[row] as f64)).sqrt());
        }
    }
    broadcast_row_agg(x, &agg)
}

/// `cs_max(x)` / `cs_min(x)`：截面极值。
pub fn op_cs_max(x: &Array2<f64>) -> Array2<f64> {
    cs_extreme(x, true)
}

pub fn op_cs_min(x: &Array2<f64>) -> Array2<f64> {
    cs_extreme(x, false)
}

fn cs_extreme(x: &Array2<f64>, want_max: bool) -> Array2<f64> {
    let (t, n) = x.dim();
    let mut agg = vec![f64::NAN; t];
    for col in 0..n {
        let xc = x.column(col);
        for row in 0..t {
            let v = xc[row];
            if v.is_finite() {
                let cur = agg[row];
                if !cur.is_finite() || (want_max && v > cur) || (!want_max && v < cur) {
                    agg[row] = v;
                }
            }
        }
    }
    broadcast_row_agg(x, &agg)
}

/// `cs_nonan_num(x)`：截面非 `NaN` 数量。
pub fn op_cs_nonan_num(x: &Array2<f64>) -> Array2<f64> {
    let (t, n) = x.dim();
    let mut agg = vec![f64::NAN; t];
    let mut cnts = vec![0u32; t];
    for col in 0..n {
        let xc = x.column(col);
        for row in 0..t {
            if xc[row].is_finite() {
                cnts[row] += 1;
            }
        }
    }
    for row in 0..t {
        agg[row] = cnts[row] as f64;
    }
    broadcast_row_agg(x, &agg)
}

/// `cs_argmax(x)` / `cs_argmin(x)`：截面极值所在的 0 基列位置（并列取最小列号）。
pub fn op_cs_argmax(x: &Array2<f64>) -> Array2<f64> {
    cs_extreme_arg(x, true)
}

pub fn op_cs_argmin(x: &Array2<f64>) -> Array2<f64> {
    cs_extreme_arg(x, false)
}

fn cs_extreme_arg(x: &Array2<f64>, want_max: bool) -> Array2<f64> {
    let (t, n) = x.dim();
    let mut best_val = vec![f64::NAN; t];
    let mut best_idx = vec![f64::NAN; t];
    for col in 0..n {
        let xc = x.column(col);
        for row in 0..t {
            let v = xc[row];
            if v.is_finite() {
                let cur = best_val[row];
                if !cur.is_finite() || (want_max && v > cur) || (!want_max && v < cur) {
                    best_val[row] = v;
                    best_idx[row] = col as f64;
                }
            }
        }
    }
    broadcast_row_agg(x, &best_idx)
}

/// 将 `(col, val)` 缓冲区按数值升序（并列保持列号升序）排序。
#[inline]
fn sort_row_values(buf: &mut [(usize, f64)]) {
    buf.sort_unstable_by(|a, b| crate::nan::cmp_f64(a.1, b.1).then_with(|| a.0.cmp(&b.0)));
}

/// 收集单行有限值及其列号。
#[inline]
fn collect_row(buf: &mut Vec<(usize, f64)>, c_row: &[f64], row: usize, n: usize) {
    buf.clear();
    let base = row * n;
    for col in 0..n {
        let v = c_row[base + col];
        if v.is_finite() {
            buf.push((col, v));
        }
    }
}

/// `cs_median(x)`：截面下中位数（偶数样本取中间较小值）。
#[allow(clippy::needless_range_loop)]
pub fn op_cs_median(x: &Array2<f64>) -> Array2<f64> {
    let (t, n) = x.dim();
    if t == 0 || n == 0 {
        return alloc_nan_matrix(t, n);
    }
    let c_in = transpose_f_to_c(x);
    let mut agg = vec![f64::NAN; t];
    let mut buf: Vec<(usize, f64)> = Vec::with_capacity(n);
    for row in 0..t {
        collect_row(&mut buf, &c_in, row, n);
        if buf.is_empty() {
            continue;
        }
        sort_row_values(&mut buf);
        agg[row] = buf[(buf.len() - 1) / 2].1;
    }
    broadcast_row_agg(x, &agg)
}

/// `cs_quantile(x, q)`：截面 `q` 分位数（线性内插，与 `numpy.quantile` 默认口径一致）。
#[allow(clippy::needless_range_loop)]
pub fn op_cs_quantile(x: &Array2<f64>, q: f64) -> Array2<f64> {
    let (t, n) = x.dim();
    if t == 0 || n == 0 || !q.is_finite() {
        return alloc_nan_matrix(t, n);
    }
    let qq = q.clamp(0.0, 1.0);
    let c_in = transpose_f_to_c(x);
    let mut agg = vec![f64::NAN; t];
    let mut buf: Vec<(usize, f64)> = Vec::with_capacity(n);
    let mut vals: Vec<f64> = Vec::with_capacity(n);
    for row in 0..t {
        collect_row(&mut buf, &c_in, row, n);
        if buf.is_empty() {
            continue;
        }
        sort_row_values(&mut buf);
        vals.clear();
        vals.extend(buf.iter().map(|p| p.1));
        agg[row] = linear_quantile_sorted(&vals, qq);
    }
    broadcast_row_agg(x, &agg)
}

/// `cs_mad(x)`：截面相对均值的平均绝对偏差（非中位数 MAD）。
pub fn op_cs_mad(x: &Array2<f64>) -> Array2<f64> {
    let (t, n) = x.dim();
    let mut means = vec![f64::NAN; t];
    let mut abs_sum = vec![0.0_f64; t];
    let mut cnts = vec![0u32; t];
    let mut sums = vec![0.0_f64; t];
    for col in 0..n {
        let xc = x.column(col);
        for row in 0..t {
            let v = xc[row];
            if v.is_finite() {
                sums[row] += v;
                cnts[row] += 1;
            }
        }
    }
    for row in 0..t {
        if cnts[row] >= 1 {
            means[row] = sums[row] / (cnts[row] as f64);
        }
    }
    for col in 0..n {
        let xc = x.column(col);
        for row in 0..t {
            let v = xc[row];
            let m = means[row];
            if v.is_finite() && m.is_finite() {
                abs_sum[row] += (v - m).abs();
            }
        }
    }
    let mut agg = vec![f64::NAN; t];
    for row in 0..t {
        if means[row].is_finite() && cnts[row] >= 1 {
            agg[row] = clean_f64(abs_sum[row] / (cnts[row] as f64));
        }
    }
    broadcast_row_agg(x, &agg)
}

/// 截面高阶矩辅助：返回 `(mean, m2, m3, m4, count)`。
fn cs_moments(x: &Array2<f64>, row: usize) -> (f64, f64, f64, f64, usize) {
    let n = x.ncols();
    let mut sum = 0.0_f64;
    let mut cnt = 0usize;
    for col in 0..n {
        let v = x[[row, col]];
        if v.is_finite() {
            sum += v;
            cnt += 1;
        }
    }
    if cnt == 0 {
        return (f64::NAN, 0.0, 0.0, 0.0, 0);
    }
    let mean = sum / (cnt as f64);
    let mut m2 = 0.0_f64;
    let mut m3 = 0.0_f64;
    let mut m4 = 0.0_f64;
    for col in 0..n {
        let v = x[[row, col]];
        if v.is_finite() {
            let d = v - mean;
            let d2 = d * d;
            m2 += d2;
            m3 += d2 * d;
            m4 += d2 * d2;
        }
    }
    let kf = cnt as f64;
    (mean, m2 / kf, m3 / kf, m4 / kf, cnt)
}

/// `cs_skew(x)`：截面偏度（总体矩 `m3 / m2^1.5`，需至少 3 个有效观测）。
#[allow(clippy::needless_range_loop)]
pub fn op_cs_skew(x: &Array2<f64>) -> Array2<f64> {
    let (t, _n) = x.dim();
    let mut agg = vec![f64::NAN; t];
    for row in 0..t {
        let (mean, m2, m3, _m4, cnt) = cs_moments(x, row);
        if cnt >= 3 && mean.is_finite() {
            let scale = 1.0_f64.max(mean.abs());
            if m2 > VAR_EPS * scale * scale {
                agg[row] = clean_f64(m3 / m2.powf(1.5));
            }
        }
    }
    broadcast_row_agg(x, &agg)
}

/// `cs_kurt(x)`：截面超额峰度（总体矩 `m4 / m2^2 - 3`，需至少 4 个有效观测）。
#[allow(clippy::needless_range_loop)]
pub fn op_cs_kurt(x: &Array2<f64>) -> Array2<f64> {
    let (t, _n) = x.dim();
    let mut agg = vec![f64::NAN; t];
    for row in 0..t {
        let (mean, m2, _m3, m4, cnt) = cs_moments(x, row);
        if cnt >= 4 && mean.is_finite() {
            let scale = 1.0_f64.max(mean.abs());
            if m2 > VAR_EPS * scale * scale {
                agg[row] = clean_f64(m4 / (m2 * m2) - 3.0);
            }
        }
    }
    broadcast_row_agg(x, &agg)
}

/// `cs_pct_rank(x)` / `rank(x)`：稳定原始名次（1..k）除以有效样本数。
pub fn op_cs_pct_rank(x: &Array2<f64>) -> Array2<f64> {
    let (t, n) = x.dim();
    if t == 0 || n == 0 {
        return alloc_nan_matrix(t, n);
    }
    let c_in = transpose_f_to_c(x);
    let mut c_out = vec![f64::NAN; t * n];
    let mut buf: Vec<(usize, f64)> = Vec::with_capacity(n);
    for row in 0..t {
        collect_row(&mut buf, &c_in, row, n);
        if buf.is_empty() {
            continue;
        }
        sort_row_values(&mut buf);
        let kf = buf.len() as f64;
        for (i, &(col, _)) in buf.iter().enumerate() {
            c_out[row * n + col] = ((i + 1) as f64) / kf;
        }
    }
    transpose_c_to_f(&c_out, t, n)
}

/// `cs_dense_rank(x)`：稠密名次（1..D，并列共享同一名次）。
pub fn op_cs_dense_rank(x: &Array2<f64>) -> Array2<f64> {
    cs_dense(x, false)
}

/// `cs_dense_pct_rank(x)`：稠密名次除以不同取值个数 D。
pub fn op_cs_dense_pct_rank(x: &Array2<f64>) -> Array2<f64> {
    cs_dense(x, true)
}

fn cs_dense(x: &Array2<f64>, pct: bool) -> Array2<f64> {
    let (t, n) = x.dim();
    if t == 0 || n == 0 {
        return alloc_nan_matrix(t, n);
    }
    let c_in = transpose_f_to_c(x);
    let mut c_out = vec![f64::NAN; t * n];
    let mut buf: Vec<(usize, f64)> = Vec::with_capacity(n);
    for row in 0..t {
        collect_row(&mut buf, &c_in, row, n);
        if buf.is_empty() {
            continue;
        }
        sort_row_values(&mut buf);
        let mut rank = 0usize;
        let mut i = 0usize;
        while i < buf.len() {
            let base_v = buf[i].1;
            let tol = STD_EPS * 1.0_f64.max(base_v.abs());
            let mut j = i + 1;
            while j < buf.len() && (buf[j].1 - base_v).abs() <= tol {
                j += 1;
            }
            rank += 1;
            let assign = rank as f64;
            for &(col, _) in &buf[i..j] {
                c_out[row * n + col] = assign;
            }
            i = j;
        }
        if pct && rank > 0 {
            let denom = rank as f64;
            for j in 0..n {
                let v = c_out[row * n + j];
                if v.is_finite() {
                    c_out[row * n + j] = v / denom;
                }
            }
        }
    }
    transpose_c_to_f(&c_out, t, n)
}

/// `cs_group(x, groups)`：截面百分位排名后浮点整除分组，组号 `0..groups-1`。
pub fn op_cs_group(x: &Array2<f64>, groups: f64) -> Array2<f64> {
    let (t, n) = x.dim();
    if t == 0 || n == 0 || !groups.is_finite() || groups < 1.0 {
        return alloc_nan_matrix(t, n);
    }
    let g = groups.floor().max(1.0);
    let pct = op_cs_pct_rank(x);
    let mut out = alloc_nan_matrix(t, n);
    for col in 0..n {
        let pc = pct.column(col);
        let mut oc = out.column_mut(col);
        for row in 0..t {
            let p = pc[row];
            if p.is_finite() {
                let grp = (p * g).floor().clamp(0.0, g - 1.0);
                oc[row] = clean_f64(grp);
            }
        }
    }
    out
}

/// `cs_cov(x1, x2)`：截面样本协方差（`ddof=1`）。
pub fn op_cs_cov(y: &Array2<f64>, x: &Array2<f64>) -> Array2<f64> {
    let (t, n) = y.dim();
    debug_assert_eq!((t, n), x.dim());
    let mut agg = vec![f64::NAN; t];
    let mut my = vec![0.0_f64; t];
    let mut mx = vec![0.0_f64; t];
    let mut cnts = vec![0u32; t];
    for col in 0..n {
        let yc = y.column(col);
        let xc = x.column(col);
        for row in 0..t {
            let vy = yc[row];
            let vx = xc[row];
            if vy.is_finite() && vx.is_finite() {
                my[row] += vy;
                mx[row] += vx;
                cnts[row] += 1;
            }
        }
    }
    for row in 0..t {
        if cnts[row] >= 1 {
            my[row] /= cnts[row] as f64;
            mx[row] /= cnts[row] as f64;
        }
    }
    let mut cxy = vec![0.0_f64; t];
    for col in 0..n {
        let yc = y.column(col);
        let xc = x.column(col);
        for row in 0..t {
            let vy = yc[row];
            let vx = xc[row];
            if vy.is_finite() && vx.is_finite() && cnts[row] >= 2 {
                cxy[row] += (vy - my[row]) * (vx - mx[row]);
            }
        }
    }
    for row in 0..t {
        if cnts[row] >= 2 {
            agg[row] = clean_f64(cxy[row] / ((cnts[row] - 1) as f64));
        }
    }
    broadcast_row_agg(y, &agg)
}

/// `cs_beta(x1, x2)`：截面回归斜率 $\beta = \text{cov}(x_1, x_2) / \text{var}(x_2)$。
pub fn op_cs_beta(y: &Array2<f64>, x: &Array2<f64>) -> Array2<f64> {
    let (t, n) = y.dim();
    debug_assert_eq!((t, n), x.dim());
    let mut agg = vec![f64::NAN; t];
    let mut my = vec![0.0_f64; t];
    let mut mx = vec![0.0_f64; t];
    let mut cnts = vec![0u32; t];
    for col in 0..n {
        let yc = y.column(col);
        let xc = x.column(col);
        for row in 0..t {
            let vy = yc[row];
            let vx = xc[row];
            if vy.is_finite() && vx.is_finite() {
                my[row] += vy;
                mx[row] += vx;
                cnts[row] += 1;
            }
        }
    }
    for row in 0..t {
        if cnts[row] >= 1 {
            my[row] /= cnts[row] as f64;
            mx[row] /= cnts[row] as f64;
        }
    }
    let mut cxy = vec![0.0_f64; t];
    let mut m2x = vec![0.0_f64; t];
    for col in 0..n {
        let yc = y.column(col);
        let xc = x.column(col);
        for row in 0..t {
            let vy = yc[row];
            let vx = xc[row];
            if vy.is_finite() && vx.is_finite() && cnts[row] >= 2 {
                let dy = vy - my[row];
                let dx = vx - mx[row];
                cxy[row] += dy * dx;
                m2x[row] += dx * dx;
            }
        }
    }
    for row in 0..t {
        if cnts[row] >= 2 {
            let scale = 1.0_f64.max(mx[row].abs());
            if m2x[row] > VAR_EPS * (cnts[row] as f64) * scale * scale {
                agg[row] = clean_f64(cxy[row] / m2x[row]);
            }
        }
    }
    broadcast_row_agg(y, &agg)
}

/// `cs_corr(x1, x2)`：截面 Pearson 相关系数（截断至 `[-1, 1]`）。
pub fn op_cs_corr(y: &Array2<f64>, x: &Array2<f64>) -> Array2<f64> {
    let (t, n) = y.dim();
    debug_assert_eq!((t, n), x.dim());
    let mut agg = vec![f64::NAN; t];
    let mut my = vec![0.0_f64; t];
    let mut mx = vec![0.0_f64; t];
    let mut cnts = vec![0u32; t];
    for col in 0..n {
        let yc = y.column(col);
        let xc = x.column(col);
        for row in 0..t {
            let vy = yc[row];
            let vx = xc[row];
            if vy.is_finite() && vx.is_finite() {
                my[row] += vy;
                mx[row] += vx;
                cnts[row] += 1;
            }
        }
    }
    for row in 0..t {
        if cnts[row] >= 1 {
            my[row] /= cnts[row] as f64;
            mx[row] /= cnts[row] as f64;
        }
    }
    let mut cxy = vec![0.0_f64; t];
    let mut m2x = vec![0.0_f64; t];
    let mut m2y = vec![0.0_f64; t];
    for col in 0..n {
        let yc = y.column(col);
        let xc = x.column(col);
        for row in 0..t {
            let vy = yc[row];
            let vx = xc[row];
            if vy.is_finite() && vx.is_finite() && cnts[row] >= 2 {
                let dy = vy - my[row];
                let dx = vx - mx[row];
                cxy[row] += dy * dx;
                m2x[row] += dx * dx;
                m2y[row] += dy * dy;
            }
        }
    }
    for row in 0..t {
        if cnts[row] >= 2 {
            let kf = cnts[row] as f64;
            let sy = 1.0_f64.max(my[row].abs());
            let sx = 1.0_f64.max(mx[row].abs());
            if m2x[row] > VAR_EPS * kf * sx * sx && m2y[row] > VAR_EPS * kf * sy * sy {
                let denom = (m2x[row] * m2y[row]).sqrt();
                if denom > 0.0 && denom.is_finite() {
                    agg[row] = clean_f64((cxy[row] / denom).clamp(-1.0, 1.0));
                }
            }
        }
    }
    broadcast_row_agg(y, &agg)
}

/// `barra5_zScore(x1, x2)`：以 `x2`（市值）为权重计算截面加权均值，使用等权总体标准差标准化。
pub fn op_barra5_zscore(x: &Array2<f64>, weight: &Array2<f64>) -> Array2<f64> {
    let (t, n) = x.dim();
    debug_assert_eq!((t, n), weight.dim());
    let mut out = alloc_nan_matrix(t, n);
    let mut wsum = vec![0.0_f64; t];
    let mut wxsum = vec![0.0_f64; t];
    let mut esum = vec![0.0_f64; t];
    let mut cnts = vec![0u32; t];
    for col in 0..n {
        let xc = x.column(col);
        let wc = weight.column(col);
        for row in 0..t {
            let vx = xc[row];
            let w = wc[row];
            if vx.is_finite() && w.is_finite() && w >= 0.0 {
                wsum[row] += w;
                wxsum[row] += w * vx;
                esum[row] += vx;
                cnts[row] += 1;
            }
        }
    }
    let mut means = vec![f64::NAN; t];
    let mut ew_means = vec![f64::NAN; t];
    for row in 0..t {
        if cnts[row] >= 1 {
            ew_means[row] = esum[row] / (cnts[row] as f64);
            if wsum[row] > 0.0 {
                means[row] = wxsum[row] / wsum[row];
            }
        }
    }
    let mut m2s = vec![0.0_f64; t];
    for col in 0..n {
        let xc = x.column(col);
        let wc = weight.column(col);
        for row in 0..t {
            let vx = xc[row];
            let w = wc[row];
            let m = means[row];
            if vx.is_finite() && w.is_finite() && w >= 0.0 && m.is_finite() {
                let d = vx - m;
                m2s[row] += d * d;
            }
        }
    }
    let mut stds = vec![f64::NAN; t];
    for row in 0..t {
        if means[row].is_finite() && cnts[row] >= 1 {
            let s = (m2s[row] / (cnts[row] as f64)).sqrt();
            let scale = 1.0_f64.max(ew_means[row].abs());
            if s.is_finite() && s > STD_EPS * scale {
                stds[row] = s;
            }
        }
    }
    for col in 0..n {
        let xc = x.column(col);
        let wc = weight.column(col);
        let mut oc = out.column_mut(col);
        for row in 0..t {
            let vx = xc[row];
            let w = wc[row];
            let s = stds[row];
            if vx.is_finite() && w.is_finite() && w >= 0.0 && s.is_finite() {
                oc[row] = clean_f64((vx - means[row]) / s);
            }
        }
    }
    out
}
/// 对已升序排序的有限值切片计算线性内插分位数（与 `numpy.quantile` 默认口径一致）。
#[inline]
fn linear_quantile_sorted(sorted_vals: &[f64], p: f64) -> f64 {
    let k = sorted_vals.len();
    if k == 0 {
        return f64::NAN;
    }
    if k == 1 {
        return sorted_vals[0];
    }
    let pos = p.clamp(0.0, 1.0) * ((k - 1) as f64);
    let lo_idx = pos.floor() as usize;
    let hi_idx = pos.ceil() as usize;
    let frac = pos - (lo_idx as f64);
    sorted_vals[lo_idx] + frac * (sorted_vals[hi_idx] - sorted_vals[lo_idx])
}
