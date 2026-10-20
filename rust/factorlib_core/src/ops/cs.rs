//! 横截面算子实现 (ops/cs.rs)。
//!
//! 针对 Fortran 列优先 `(T, N)` 矩阵进行了极致的 CPU L1 Cache 局部性优化：
//! - `cs_demean`, `cs_zscore`, `cs_scale`, `cs_neutralize`：采用按列顺序扫描累加到 L1 常驻行统计向量
//!   （长度 $T=4000$ 仅占 32KB），实现 100% 步长 1（Stride-1）连续内存读写，彻底消除跨列跳跃 Cache Miss；
//! - `cs_rank`, `cs_winsorize`：采用分块转置（Tiled Transpose）至行连续缓冲区后计算，再分块写回列优先矩阵。

use crate::nan::{clean_f64, cmp_f64, compute_normalized_ranks, EPS, STD_EPS, VAR_EPS};
use crate::panel::alloc_nan_matrix;
use ndarray::{Array2, ShapeBuilder};

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
    let mut out = unsafe { Array2::<f64>::uninit((t, n).f()).assume_init() };
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
        if lo.is_finite() && hi.is_finite() {
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
            if m2_x[row].is_finite()
                && c_xy[row].is_finite()
                && m2_x[row] > VAR_EPS * kf * sx * sx
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
