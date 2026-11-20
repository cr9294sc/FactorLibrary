//! 高性能时序滚动算子实现 (ops/ts.rs)。
//!
//! 严格遵循白皮书 6.2 节的高性能时序滚动设计：
//! - `ts_mean`, `ts_std`, `ts_var`, `ts_sum`, `ts_zscore`：采用基于双栈（Two-Stack Queue）的
//!   Chan 并行 Welford 增量状态机，时间复杂度均摊 $O(1)$，且完全杜绝减法灾难性抵消；
//! - `ts_min`, `ts_max`, `ts_argmin`, `ts_argmax`：采用双端单调队列（`VecDeque<(usize, f64)>`），
//!   每个元素至多入队出队一次，单步均摊复杂度 $O(1)$；
//! - `ts_rank`：维护有序滑动窗口并进行二分查找（`partition_point`），单步复杂度 $O(\log d)$；
//! - `ts_cov`, `ts_corr`, `ts_beta`, `ts_resid`：采用基于双栈的二元协矩 Welford 状态机，单步均摊 $O(1)$。

use crate::nan::{
    clean_f64, cmp_f64, TwoStackBivariateWelford, TwoStackWelford, EPS, STD_EPS, VAR_EPS,
};
use crate::panel::alloc_nan_matrix;
use ndarray::Array2;
use std::collections::VecDeque;

// ============================================================================
// 1. 基础时序算子: delay, delta, pct_change, log_return
// ============================================================================

pub fn op_delay(x: &Array2<f64>, d: usize) -> Array2<f64> {
    let (t, n) = x.dim();
    let mut out = alloc_nan_matrix(t, n);
    if d == 0 {
        for col in 0..n {
            let xc = x.column(col);
            let mut oc = out.column_mut(col);
            for row in 0..t {
                oc[row] = clean_f64(xc[row]);
            }
        }
        return out;
    }
    if d >= t {
        return out;
    }
    for col in 0..n {
        let xc = x.column(col);
        let mut oc = out.column_mut(col);
        for row in d..t {
            oc[row] = clean_f64(xc[row - d]);
        }
    }
    out
}

pub fn op_delta(x: &Array2<f64>, d: usize) -> Array2<f64> {
    let (t, n) = x.dim();
    let mut out = alloc_nan_matrix(t, n);
    if d == 0 || d >= t {
        return out;
    }
    for col in 0..n {
        let xc = x.column(col);
        let mut oc = out.column_mut(col);
        for row in d..t {
            let cur = xc[row];
            let prev = xc[row - d];
            if cur.is_finite() && prev.is_finite() {
                oc[row] = clean_f64(cur - prev);
            }
        }
    }
    out
}

pub fn op_pct_change(x: &Array2<f64>, d: usize) -> Array2<f64> {
    let (t, n) = x.dim();
    let mut out = alloc_nan_matrix(t, n);
    if d == 0 || d >= t {
        return out;
    }
    for col in 0..n {
        let xc = x.column(col);
        let mut oc = out.column_mut(col);
        for row in d..t {
            let cur = xc[row];
            let prev = xc[row - d];
            if cur.is_finite() && prev.is_finite() && prev.abs() >= EPS {
                oc[row] = clean_f64((cur - prev) / prev);
            }
        }
    }
    out
}

pub fn op_log_return(x: &Array2<f64>, d: usize) -> Array2<f64> {
    let (t, n) = x.dim();
    let mut out = alloc_nan_matrix(t, n);
    if d == 0 || d >= t {
        return out;
    }
    for col in 0..n {
        let xc = x.column(col);
        let mut oc = out.column_mut(col);
        for row in d..t {
            let cur = xc[row];
            let prev = xc[row - d];
            if cur.is_finite() && prev.is_finite() && cur > EPS && prev > EPS {
                oc[row] = clean_f64((cur / prev).ln());
            }
        }
    }
    out
}

// ============================================================================
// 2. 双栈 Welford O(1) 增量时序算子: ts_sum, ts_mean, ts_std, ts_var, ts_zscore
// ============================================================================

pub fn op_ts_sum(x: &Array2<f64>, d: usize) -> Array2<f64> {
    let (t, n) = x.dim();
    let mut out = alloc_nan_matrix(t, n);
    if d == 0 || d > t {
        return out;
    }
    let mut sw = TwoStackWelford::with_capacity(d + 1);
    for col in 0..n {
        sw.clear();
        let xc = x.column(col);
        let mut oc = out.column_mut(col);
        for row in 0..t {
            sw.push(xc[row]);
            if row >= d {
                sw.pop();
            }
            if row + 1 >= d {
                let st = sw.query();
                if st.n >= 1 {
                    oc[row] = clean_f64(st.sum);
                }
            }
        }
    }
    out
}

pub fn op_ts_mean(x: &Array2<f64>, d: usize) -> Array2<f64> {
    let (t, n) = x.dim();
    let mut out = alloc_nan_matrix(t, n);
    if d == 0 || d > t {
        return out;
    }
    let mut sw = TwoStackWelford::with_capacity(d + 1);
    for col in 0..n {
        sw.clear();
        let xc = x.column(col);
        let mut oc = out.column_mut(col);
        for row in 0..t {
            sw.push(xc[row]);
            if row >= d {
                sw.pop();
            }
            if row + 1 >= d {
                let st = sw.query();
                if st.n >= 1 {
                    oc[row] = clean_f64(st.mean);
                }
            }
        }
    }
    out
}

pub fn op_ts_std(x: &Array2<f64>, d: usize) -> Array2<f64> {
    let (t, n) = x.dim();
    let mut out = alloc_nan_matrix(t, n);
    if d < 2 || d > t {
        return out;
    }
    let mut sw = TwoStackWelford::with_capacity(d + 1);
    for col in 0..n {
        sw.clear();
        let xc = x.column(col);
        let mut oc = out.column_mut(col);
        for row in 0..t {
            sw.push(xc[row]);
            if row >= d {
                sw.pop();
            }
            if row + 1 >= d {
                let st = sw.query();
                oc[row] = st.sample_std();
            }
        }
    }
    out
}

pub fn op_ts_var(x: &Array2<f64>, d: usize) -> Array2<f64> {
    let (t, n) = x.dim();
    let mut out = alloc_nan_matrix(t, n);
    if d < 2 || d > t {
        return out;
    }
    let mut sw = TwoStackWelford::with_capacity(d + 1);
    for col in 0..n {
        sw.clear();
        let xc = x.column(col);
        let mut oc = out.column_mut(col);
        for row in 0..t {
            sw.push(xc[row]);
            if row >= d {
                sw.pop();
            }
            if row + 1 >= d {
                let st = sw.query();
                oc[row] = st.sample_var();
            }
        }
    }
    out
}

pub fn op_ts_zscore(x: &Array2<f64>, d: usize) -> Array2<f64> {
    let (t, n) = x.dim();
    let mut out = alloc_nan_matrix(t, n);
    if d < 2 || d > t {
        return out;
    }
    let mut sw = TwoStackWelford::with_capacity(d + 1);
    for col in 0..n {
        sw.clear();
        let xc = x.column(col);
        let mut oc = out.column_mut(col);
        for row in 0..t {
            let cur = xc[row];
            sw.push(cur);
            if row >= d {
                sw.pop();
            }
            if row + 1 >= d && cur.is_finite() {
                let st = sw.query();
                if st.n >= 2 {
                    let std = st.sample_std();
                    let scale = 1.0_f64.max(st.mean.abs());
                    if std.is_finite() && std > STD_EPS * scale {
                        oc[row] = clean_f64((cur - st.mean) / std);
                    }
                }
            }
        }
    }
    out
}

// ============================================================================
// 3. 单调双端队列 O(1) 极值算子: ts_min, ts_max, ts_argmin, ts_argmax
// ============================================================================

pub fn op_ts_min(x: &Array2<f64>, d: usize) -> Array2<f64> {
    let (t, n) = x.dim();
    let mut out = alloc_nan_matrix(t, n);
    if d == 0 || d > t {
        return out;
    }
    let mut dq: VecDeque<(usize, f64)> = VecDeque::with_capacity(d + 1);
    for col in 0..n {
        dq.clear();
        let xc = x.column(col);
        let mut oc = out.column_mut(col);
        for row in 0..t {
            while let Some(&(idx, _)) = dq.front() {
                if idx + d <= row {
                    dq.pop_front();
                } else {
                    break;
                }
            }
            let cur = xc[row];
            if cur.is_finite() {
                while let Some(&(_, back_val)) = dq.back() {
                    if back_val >= cur {
                        dq.pop_back();
                    } else {
                        break;
                    }
                }
                dq.push_back((row, cur));
            }
            if row + 1 >= d {
                if let Some(&(_, min_val)) = dq.front() {
                    oc[row] = min_val;
                }
            }
        }
    }
    out
}

pub fn op_ts_max(x: &Array2<f64>, d: usize) -> Array2<f64> {
    let (t, n) = x.dim();
    let mut out = alloc_nan_matrix(t, n);
    if d == 0 || d > t {
        return out;
    }
    let mut dq: VecDeque<(usize, f64)> = VecDeque::with_capacity(d + 1);
    for col in 0..n {
        dq.clear();
        let xc = x.column(col);
        let mut oc = out.column_mut(col);
        for row in 0..t {
            while let Some(&(idx, _)) = dq.front() {
                if idx + d <= row {
                    dq.pop_front();
                } else {
                    break;
                }
            }
            let cur = xc[row];
            if cur.is_finite() {
                while let Some(&(_, back_val)) = dq.back() {
                    if back_val <= cur {
                        dq.pop_back();
                    } else {
                        break;
                    }
                }
                dq.push_back((row, cur));
            }
            if row + 1 >= d {
                if let Some(&(_, max_val)) = dq.front() {
                    oc[row] = max_val;
                }
            }
        }
    }
    out
}

/// 窗口 `[row + 1 - d ..= row]` 内最小值首次出现的相对偏移索引（`0.0 ..= (d-1) as f64`）。
/// 使用严格 `>` 弹出队尾，保证并列最小值保留最早出现的索引，均摊复杂度 $O(1)$。
pub fn op_ts_argmin(x: &Array2<f64>, d: usize) -> Array2<f64> {
    let (t, n) = x.dim();
    let mut out = alloc_nan_matrix(t, n);
    if d == 0 || d > t {
        return out;
    }
    let mut dq: VecDeque<(usize, f64)> = VecDeque::with_capacity(d + 1);
    for col in 0..n {
        dq.clear();
        let xc = x.column(col);
        let mut oc = out.column_mut(col);
        for row in 0..t {
            while let Some(&(idx, _)) = dq.front() {
                if idx + d <= row {
                    dq.pop_front();
                } else {
                    break;
                }
            }
            let cur = xc[row];
            if cur.is_finite() {
                while let Some(&(_, back_val)) = dq.back() {
                    if back_val > cur {
                        dq.pop_back();
                    } else {
                        break;
                    }
                }
                dq.push_back((row, cur));
            }
            if row + 1 >= d {
                let win_start = row + 1 - d;
                if let Some(&(min_idx, _)) = dq.front() {
                    oc[row] = (min_idx - win_start) as f64;
                }
            }
        }
    }
    out
}

/// 窗口 `[row + 1 - d ..= row]` 内最大值首次出现的相对偏移索引（`0.0 ..= (d-1) as f64`）。
pub fn op_ts_argmax(x: &Array2<f64>, d: usize) -> Array2<f64> {
    let (t, n) = x.dim();
    let mut out = alloc_nan_matrix(t, n);
    if d == 0 || d > t {
        return out;
    }
    let mut dq: VecDeque<(usize, f64)> = VecDeque::with_capacity(d + 1);
    for col in 0..n {
        dq.clear();
        let xc = x.column(col);
        let mut oc = out.column_mut(col);
        for row in 0..t {
            while let Some(&(idx, _)) = dq.front() {
                if idx + d <= row {
                    dq.pop_front();
                } else {
                    break;
                }
            }
            let cur = xc[row];
            if cur.is_finite() {
                while let Some(&(_, back_val)) = dq.back() {
                    if back_val < cur {
                        dq.pop_back();
                    } else {
                        break;
                    }
                }
                dq.push_back((row, cur));
            }
            if row + 1 >= d {
                let win_start = row + 1 - d;
                if let Some(&(max_idx, _)) = dq.front() {
                    oc[row] = (max_idx - win_start) as f64;
                }
            }
        }
    }
    out
}

// ============================================================================
// 4. 滚动分位排名算子: ts_rank (维护有序窗口，二分查找 O(log d))
// ============================================================================

pub fn op_ts_rank(x: &Array2<f64>, d: usize) -> Array2<f64> {
    let (t, n) = x.dim();
    let mut out = alloc_nan_matrix(t, n);
    if d == 0 || d > t {
        return out;
    }
    let mut sorted_win: Vec<f64> = Vec::with_capacity(d + 1);
    for col in 0..n {
        sorted_win.clear();
        let xc = x.column(col);
        let mut oc = out.column_mut(col);
        for row in 0..t {
            if row >= d {
                let old = xc[row - d];
                if old.is_finite() {
                    let pos = sorted_win.partition_point(|&v| cmp_f64(v, old).is_lt());
                    if pos < sorted_win.len() {
                        sorted_win.remove(pos);
                    }
                }
            }
            let cur = xc[row];
            if cur.is_finite() {
                let pos = sorted_win.partition_point(|&v| cmp_f64(v, cur).is_lt());
                sorted_win.insert(pos, cur);
            }
            if row + 1 >= d && cur.is_finite() {
                let k = sorted_win.len();
                if k == 1 {
                    oc[row] = 0.5;
                } else if k >= 2 {
                    let tol = STD_EPS * 1.0_f64.max(cur.abs());
                    let lo = sorted_win.partition_point(|&v| v < cur - tol);
                    let hi = sorted_win.partition_point(|&v| v <= cur + tol);
                    let c_less = lo as f64;
                    let c_eq = (hi - lo).max(1) as f64;
                    oc[row] = (c_less + 0.5 * (c_eq - 1.0)) / ((k - 1) as f64);
                }
            }
        }
    }
    out
}

// ============================================================================
// 5. 高阶矩与加权移动平均算子: ts_skew, ts_kurt, ts_ema, ts_decay_linear, ts_mad
// ============================================================================

pub fn op_ts_skew(x: &Array2<f64>, d: usize) -> Array2<f64> {
    let (t, n) = x.dim();
    let mut out = alloc_nan_matrix(t, n);
    if d < 3 || d > t {
        return out;
    }
    let mut sw = TwoStackWelford::with_capacity(d + 1);
    for col in 0..n {
        sw.clear();
        let xc = x.column(col);
        let mut oc = out.column_mut(col);
        for row in 0..t {
            sw.push(xc[row]);
            if row >= d {
                sw.pop();
            }
            if row + 1 >= d {
                let st = sw.query();
                if st.n >= 3 {
                    let mean = st.mean;
                    let scale = 1.0_f64.max(mean.abs());
                    let win_start = row + 1 - d;
                    let mut s2 = 0.0;
                    let mut s3 = 0.0;
                    for idx in win_start..=row {
                        let v = xc[idx];
                        if v.is_finite() {
                            let diff = v - mean;
                            let diff2 = diff * diff;
                            s2 += diff2;
                            s3 += diff2 * diff;
                        }
                    }
                    let kf = st.n as f64;
                    let m2 = s2 / kf;
                    let m3 = s3 / kf;
                    if m2.is_finite() && m3.is_finite() && m2 > VAR_EPS * scale * scale {
                        oc[row] = clean_f64(m3 / m2.powf(1.5));
                    }
                }
            }
        }
    }
    out
}

pub fn op_ts_kurt(x: &Array2<f64>, d: usize) -> Array2<f64> {
    let (t, n) = x.dim();
    let mut out = alloc_nan_matrix(t, n);
    if d < 4 || d > t {
        return out;
    }
    let mut sw = TwoStackWelford::with_capacity(d + 1);
    for col in 0..n {
        sw.clear();
        let xc = x.column(col);
        let mut oc = out.column_mut(col);
        for row in 0..t {
            sw.push(xc[row]);
            if row >= d {
                sw.pop();
            }
            if row + 1 >= d {
                let st = sw.query();
                if st.n >= 4 {
                    let mean = st.mean;
                    let scale = 1.0_f64.max(mean.abs());
                    let win_start = row + 1 - d;
                    let mut s2 = 0.0;
                    let mut s4 = 0.0;
                    for idx in win_start..=row {
                        let v = xc[idx];
                        if v.is_finite() {
                            let diff = v - mean;
                            let diff2 = diff * diff;
                            s2 += diff2;
                            s4 += diff2 * diff2;
                        }
                    }
                    let kf = st.n as f64;
                    let m2 = s2 / kf;
                    let m4 = s4 / kf;
                    if m2.is_finite() && m4.is_finite() && m2 > VAR_EPS * scale * scale {
                        oc[row] = clean_f64(m4 / (m2 * m2) - 3.0);
                    }
                }
            }
        }
    }
    out
}

pub fn op_ts_ema(x: &Array2<f64>, d: usize) -> Array2<f64> {
    let (t, n) = x.dim();
    let mut out = alloc_nan_matrix(t, n);
    if d == 0 || t == 0 {
        return out;
    }
    let alpha = 2.0 / ((d as f64) + 1.0);
    let one_minus_alpha = 1.0 - alpha;
    for col in 0..n {
        let xc = x.column(col);
        let mut oc = out.column_mut(col);
        let mut ema = f64::NAN;
        for row in 0..t {
            let v = xc[row];
            if v.is_finite() {
                ema = if ema.is_nan() {
                    v
                } else {
                    clean_f64(alpha * v + one_minus_alpha * ema)
                };
                oc[row] = ema;
            } else {
                oc[row] = f64::NAN;
            }
        }
    }
    out
}

pub fn op_ts_decay_linear(x: &Array2<f64>, d: usize) -> Array2<f64> {
    let (t, n) = x.dim();
    let mut out = alloc_nan_matrix(t, n);
    if d == 0 || d > t {
        return out;
    }
    for col in 0..n {
        let xc = x.column(col);
        let mut oc = out.column_mut(col);
        for row in (d - 1)..t {
            let win_start = row + 1 - d;
            let mut w_sum = 0.0;
            let mut val_sum = 0.0;
            for i in 0..d {
                let v = xc[win_start + i];
                if v.is_finite() {
                    let w = (i + 1) as f64;
                    w_sum += w;
                    val_sum += w * v;
                }
            }
            if w_sum > 0.0 {
                oc[row] = clean_f64(val_sum / w_sum);
            }
        }
    }
    out
}

pub fn op_ts_mad(x: &Array2<f64>, d: usize) -> Array2<f64> {
    let (t, n) = x.dim();
    let mut out = alloc_nan_matrix(t, n);
    if d == 0 || d > t {
        return out;
    }
    let mut sw = TwoStackWelford::with_capacity(d + 1);
    for col in 0..n {
        sw.clear();
        let xc = x.column(col);
        let mut oc = out.column_mut(col);
        for row in 0..t {
            sw.push(xc[row]);
            if row >= d {
                sw.pop();
            }
            if row + 1 >= d {
                let st = sw.query();
                if st.n >= 1 {
                    let mean = st.mean;
                    let win_start = row + 1 - d;
                    let mut abs_sum = 0.0;
                    for idx in win_start..=row {
                        let v = xc[idx];
                        if v.is_finite() {
                            abs_sum += (v - mean).abs();
                        }
                    }
                    oc[row] = clean_f64(abs_sum / (st.n as f64));
                }
            }
        }
    }
    out
}

// ============================================================================
// 6. 二元协矩双栈 Welford O(1) 算子: ts_cov, ts_corr, ts_beta, ts_resid
// ============================================================================

pub fn op_ts_cov(x: &Array2<f64>, y: &Array2<f64>, d: usize) -> Array2<f64> {
    let (t, n) = x.dim();
    debug_assert_eq!((t, n), y.dim());
    let mut out = alloc_nan_matrix(t, n);
    if d < 2 || d > t {
        return out;
    }
    let mut sw = TwoStackBivariateWelford::with_capacity(d + 1);
    for col in 0..n {
        sw.clear();
        let xc = x.column(col);
        let yc = y.column(col);
        let mut oc = out.column_mut(col);
        for row in 0..t {
            sw.push(xc[row], yc[row]);
            if row >= d {
                sw.pop();
            }
            if row + 1 >= d {
                let st = sw.query();
                if st.n >= 2 && st.c_xy.is_finite() {
                    oc[row] = clean_f64(st.c_xy / ((st.n - 1) as f64));
                }
            }
        }
    }
    out
}

pub fn op_ts_corr(x: &Array2<f64>, y: &Array2<f64>, d: usize) -> Array2<f64> {
    let (t, n) = x.dim();
    debug_assert_eq!((t, n), y.dim());
    let mut out = alloc_nan_matrix(t, n);
    if d < 2 || d > t {
        return out;
    }
    let mut sw = TwoStackBivariateWelford::with_capacity(d + 1);
    for col in 0..n {
        sw.clear();
        let xc = x.column(col);
        let yc = y.column(col);
        let mut oc = out.column_mut(col);
        for row in 0..t {
            sw.push(xc[row], yc[row]);
            if row >= d {
                sw.pop();
            }
            if row + 1 >= d {
                let st = sw.query();
                if st.n >= 2 {
                    let nf = st.n as f64;
                    let sx = 1.0_f64.max(st.mean_x.abs());
                    let sy = 1.0_f64.max(st.mean_y.abs());
                    if st.m2_x.is_finite()
                        && st.m2_y.is_finite()
                        && st.c_xy.is_finite()
                        && st.m2_x > VAR_EPS * nf * sx * sx
                        && st.m2_y > VAR_EPS * nf * sy * sy
                    {
                        let denom = (st.m2_x * st.m2_y).sqrt();
                        if denom > 0.0 && denom.is_finite() {
                            oc[row] = clean_f64((st.c_xy / denom).clamp(-1.0, 1.0));
                        }
                    }
                }
            }
        }
    }
    out
}

pub fn op_ts_beta(x: &Array2<f64>, y: &Array2<f64>, d: usize) -> Array2<f64> {
    let (t, n) = x.dim();
    debug_assert_eq!((t, n), y.dim());
    let mut out = alloc_nan_matrix(t, n);
    if d < 2 || d > t {
        return out;
    }
    let mut sw = TwoStackBivariateWelford::with_capacity(d + 1);
    for col in 0..n {
        sw.clear();
        let xc = x.column(col);
        let yc = y.column(col);
        let mut oc = out.column_mut(col);
        for row in 0..t {
            sw.push(xc[row], yc[row]);
            if row >= d {
                sw.pop();
            }
            if row + 1 >= d {
                let st = sw.query();
                if st.n >= 2 {
                    let nf = st.n as f64;
                    let sy = 1.0_f64.max(st.mean_y.abs());
                    if st.m2_y.is_finite()
                        && st.c_xy.is_finite()
                        && st.m2_y > VAR_EPS * nf * sy * sy
                    {
                        oc[row] = clean_f64(st.c_xy / st.m2_y);
                    }
                }
            }
        }
    }
    out
}

pub fn op_ts_resid(x: &Array2<f64>, y: &Array2<f64>, d: usize) -> Array2<f64> {
    let (t, n) = x.dim();
    debug_assert_eq!((t, n), y.dim());
    let mut out = alloc_nan_matrix(t, n);
    if d < 2 || d > t {
        return out;
    }
    let mut sw = TwoStackBivariateWelford::with_capacity(d + 1);
    for col in 0..n {
        sw.clear();
        let xc = x.column(col);
        let yc = y.column(col);
        let mut oc = out.column_mut(col);
        for row in 0..t {
            let cur_x = xc[row];
            let cur_y = yc[row];
            sw.push(cur_x, cur_y);
            if row >= d {
                sw.pop();
            }
            if row + 1 >= d && cur_x.is_finite() && cur_y.is_finite() {
                let st = sw.query();
                if st.n >= 2 {
                    let nf = st.n as f64;
                    let sy = 1.0_f64.max(st.mean_y.abs());
                    if st.m2_y.is_finite()
                        && st.c_xy.is_finite()
                        && st.m2_y > VAR_EPS * nf * sy * sy
                    {
                        let beta = st.c_xy / st.m2_y;
                        let resid = (cur_x - st.mean_x) - beta * (cur_y - st.mean_y);
                        oc[row] = clean_f64(resid);
                    }
                }
            }
        }
    }
    out
}

// ============================================================================
// 7. FactorLab 1d 时序窗口扩充算子: ts_median / ts_quantile / ts_roll /
//    ts_prod / ts_nonan_num / ts_pct_rank / ts_group
// ============================================================================

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

/// `ts_median(x, d)`：窗口内有限观测的下中位数（偶数样本取中间较小值）。
pub fn op_ts_median(x: &Array2<f64>, d: usize) -> Array2<f64> {
    let (t, n) = x.dim();
    let mut out = alloc_nan_matrix(t, n);
    if d == 0 || d > t {
        return out;
    }
    let mut vals: Vec<f64> = Vec::with_capacity(d + 1);
    for col in 0..n {
        let xc = x.column(col);
        let mut oc = out.column_mut(col);
        for row in (d - 1)..t {
            vals.clear();
            for idx in (row + 1 - d)..=row {
                let v = xc[idx];
                if v.is_finite() {
                    vals.push(v);
                }
            }
            if vals.is_empty() {
                continue;
            }
            vals.sort_unstable_by(|a, b| cmp_f64(*a, *b));
            oc[row] = vals[(vals.len() - 1) / 2];
        }
    }
    out
}

/// `ts_quantile(x, q, d)`：窗口内有限观测的 `q` 分位数（线性内插）。
pub fn op_ts_quantile(x: &Array2<f64>, q: f64, d: usize) -> Array2<f64> {
    let (t, n) = x.dim();
    let mut out = alloc_nan_matrix(t, n);
    if d == 0 || d > t || !q.is_finite() {
        return out;
    }
    let qq = q.clamp(0.0, 1.0);
    let mut vals: Vec<f64> = Vec::with_capacity(d + 1);
    for col in 0..n {
        let xc = x.column(col);
        let mut oc = out.column_mut(col);
        for row in (d - 1)..t {
            vals.clear();
            for idx in (row + 1 - d)..=row {
                let v = xc[idx];
                if v.is_finite() {
                    vals.push(v);
                }
            }
            if vals.is_empty() {
                continue;
            }
            vals.sort_unstable_by(|a, b| cmp_f64(*a, *b));
            oc[row] = clean_f64(linear_quantile_sorted(&vals, qq));
        }
    }
    out
}

/// `ts_roll(x, d)`：沿时间轴循环移位 `d` 行（首尾环绕，`out[t] = x[(t + T - d) % T]`）。
pub fn op_ts_roll(x: &Array2<f64>, d: usize) -> Array2<f64> {
    let (t, n) = x.dim();
    let mut out = alloc_nan_matrix(t, n);
    if t == 0 {
        return out;
    }
    let shift = d % t;
    for col in 0..n {
        let xc = x.column(col);
        let mut oc = out.column_mut(col);
        for row in 0..t {
            let src = (row + t - shift) % t;
            oc[row] = clean_f64(xc[src]);
        }
    }
    out
}

/// `ts_prod(x, d)`：窗口内有限观测连乘积（全缺失时返回 `NaN`）。
pub fn op_ts_prod(x: &Array2<f64>, d: usize) -> Array2<f64> {
    let (t, n) = x.dim();
    let mut out = alloc_nan_matrix(t, n);
    if d == 0 || d > t {
        return out;
    }
    for col in 0..n {
        let xc = x.column(col);
        let mut oc = out.column_mut(col);
        for row in (d - 1)..t {
            let mut prod = 1.0_f64;
            let mut cnt = 0usize;
            for idx in (row + 1 - d)..=row {
                let v = xc[idx];
                if v.is_finite() {
                    prod *= v;
                    cnt += 1;
                }
            }
            if cnt > 0 {
                oc[row] = clean_f64(prod);
            }
        }
    }
    out
}

/// `ts_nonan_num(x, d)`：窗口内非 `NaN` 有效观测计数（满窗后才输出，空窗输出 `0.0`）。
pub fn op_ts_nonan_num(x: &Array2<f64>, d: usize) -> Array2<f64> {
    let (t, n) = x.dim();
    let mut out = alloc_nan_matrix(t, n);
    if d == 0 || d > t {
        return out;
    }
    for col in 0..n {
        let xc = x.column(col);
        let mut oc = out.column_mut(col);
        for row in (d - 1)..t {
            let mut cnt = 0usize;
            for idx in (row + 1 - d)..=row {
                if xc[idx].is_finite() {
                    cnt += 1;
                }
            }
            oc[row] = cnt as f64;
        }
    }
    out
}

/// `ts_pct_rank(x, d)`：当前值在窗口内的百分位排名（1 基平均名次除以有效样本数，取值 `(0, 1]`）。
pub fn op_ts_pct_rank(x: &Array2<f64>, d: usize) -> Array2<f64> {
    let (t, n) = x.dim();
    let mut out = alloc_nan_matrix(t, n);
    if d == 0 || d > t {
        return out;
    }
    let mut vals: Vec<f64> = Vec::with_capacity(d + 1);
    for col in 0..n {
        let xc = x.column(col);
        let mut oc = out.column_mut(col);
        for row in (d - 1)..t {
            let cur = xc[row];
            if !cur.is_finite() {
                continue;
            }
            vals.clear();
            for idx in (row + 1 - d)..=row {
                let v = xc[idx];
                if v.is_finite() {
                    vals.push(v);
                }
            }
            let k = vals.len();
            if k == 0 {
                continue;
            }
            let tol = STD_EPS * 1.0_f64.max(cur.abs());
            let c_less = vals.iter().filter(|&&v| v < cur - tol).count() as f64;
            let c_eq = vals.iter().filter(|&&v| (v - cur).abs() <= tol).count() as f64;
            let avg_rank = c_less + 0.5 * (c_eq + 1.0);
            oc[row] = clean_f64(avg_rank / (k as f64));
        }
    }
    out
}

/// `ts_group(x, groups)`：按一周期收益率的因果历史百分位浮点整除分组，组号 `0..groups-1`。
pub fn op_ts_group(x: &Array2<f64>, groups: usize) -> Array2<f64> {
    let (t, n) = x.dim();
    let mut out = alloc_nan_matrix(t, n);
    if t == 0 || groups == 0 {
        return out;
    }
    let g = groups as f64;
    let ret = op_pct_change(x, 1);
    let mut hist: Vec<f64> = Vec::with_capacity(t);
    for col in 0..n {
        let rc = ret.column(col);
        let mut oc = out.column_mut(col);
        hist.clear();
        for row in 0..t {
            let r = rc[row];
            if !r.is_finite() {
                continue;
            }
            hist.push(r);
            let k = hist.len();
            let tol = STD_EPS * 1.0_f64.max(r.abs());
            let c_less = hist.iter().filter(|&&v| v < r - tol).count() as f64;
            let c_eq = hist.iter().filter(|&&v| (v - r).abs() <= tol).count() as f64;
            let pct = (c_less + 0.5 * (c_eq + 1.0)) / (k as f64);
            let grp = (pct * g).floor().clamp(0.0, g - 1.0);
            oc[row] = clean_f64(grp);
        }
    }
    out
}

/// `ts_dw(x1, x2)`：扩张窗口残差 Durbin-Watson 统计量（沿时间轴差分，需至少 3 个有效观测）。
///
/// 每个时点 $t$ 以截至 $t$ 的全部有效样本做带截距 OLS 残差递推
/// （$r_s = x_1 - (a_t + b_t x_2)$），输出
/// $\mathrm{DW}_t = \sum_{s=1}^{t}(r_s - r_{s-1})^2 \big/ \sum_{s=1}^{t} r_s^2$。
pub fn op_ts_dw(x1: &Array2<f64>, x2: &Array2<f64>) -> Array2<f64> {
    let (t, n) = x1.dim();
    debug_assert_eq!((t, n), x2.dim());
    let mut out = alloc_nan_matrix(t, n);
    for col in 0..n {
        let c1 = x1.column(col);
        let c2 = x2.column(col);
        let mut oc = out.column_mut(col);

        let mut cnt = 0usize;
        let mut mean_x = 0.0_f64;
        let mut mean_y = 0.0_f64;
        let mut m2_x = 0.0_f64;
        let mut c_xy = 0.0_f64;
        let mut prev_resid = f64::NAN;
        let mut dw_num = 0.0_f64;
        let mut dw_den = 0.0_f64;
        let mut n_resid = 0usize;

        for row in 0..t {
            let vx = c2[row];
            let vy = c1[row];
            if !vx.is_finite() || !vy.is_finite() {
                continue;
            }
            cnt += 1;
            let kf = cnt as f64;
            let dx = vx - mean_x;
            let dy = vy - mean_y;
            mean_x += dx / kf;
            mean_y += dy / kf;
            let dx2 = vx - mean_x;
            let dy2 = vy - mean_y;
            m2_x += dx * dx2;
            c_xy += dx * dy2;
            let scale_x = 1.0_f64.max(mean_x.abs());
            let (slope, intercept) = if cnt >= 2 && m2_x > VAR_EPS * kf * scale_x * scale_x {
                let b = c_xy / m2_x;
                (b, mean_y - b * mean_x)
            } else {
                (0.0, mean_y)
            };
            let resid = vy - (intercept + slope * vx);
            dw_den += resid * resid;
            if prev_resid.is_finite() {
                let dr = resid - prev_resid;
                dw_num += dr * dr;
            }
            prev_resid = resid;
            n_resid += 1;

            if n_resid >= 3 && dw_den > 0.0 {
                oc[row] = clean_f64(dw_num / dw_den);
            }
        }
    }
    out
}
// ============================================================================
// 8. 全历史（扩张窗口）因果排名算子与累积算子:
//    rank_ts / pct_rank_ts / dense_rank_ts / pct_dense_rank_ts /
//    ts_dense_rank / ts_pct_dense_rank / ts_cumsum / ts_cumprod / ts_ffill
// ============================================================================

/// 一维 Fenwick 树（Binary Indexed Tree），用于扩张窗口计数查询。
struct Fenwick {
    tree: Vec<i32>,
    n: usize,
}

impl Fenwick {
    fn new(n: usize) -> Self {
        Self {
            tree: vec![0_i32; n + 1],
            n,
        }
    }

    #[inline]
    fn add(&mut self, idx0: usize, delta: i32) {
        let mut i = idx0 + 1;
        while i <= self.n {
            self.tree[i] += delta;
            i += i.isolate_lowest_one();
        }
    }

    /// 返回前缀和 `[0 ..= idx0]`。
    #[inline]
    fn prefix(&self, idx0: usize) -> i32 {
        let mut i = idx0 + 1;
        let mut s = 0_i32;
        while i > 0 {
            s += self.tree[i];
            i -= i.isolate_lowest_one();
        }
        s
    }
}

/// 扩张窗口因果排名的核心实现。
///
/// - `pct = false, dense = false` → `rank_ts`（1 基原始名次，并列按时间先后次序）
/// - `pct = true,  dense = false` → `pct_rank_ts`（名次 / 有效样本数）
/// - `pct = false, dense = true`  → `dense_rank_ts` / `ts_dense_rank`（稠密名次 1..D）
/// - `pct = true,  dense = true`  → `pct_dense_rank_ts` / `ts_pct_dense_rank`（稠密名次 / D）
///
/// 严格因果：时点 `t` 的名次仅由 `[0..=t]` 的历史取值决定，绝不使用未来数据。
fn expanding_rank_core(x: &Array2<f64>, pct: bool, dense: bool) -> Array2<f64> {
    let (t, n) = x.dim();
    let mut out = alloc_nan_matrix(t, n);
    if t == 0 || n == 0 {
        return out;
    }
    let mut uniq: Vec<f64> = Vec::with_capacity(t);
    let mut pairs: Vec<(usize, f64)> = Vec::with_capacity(t);

    for col in 0..n {
        let xc = x.column(col);
        uniq.clear();
        pairs.clear();
        for row in 0..t {
            let v = xc[row];
            if v.is_finite() {
                uniq.push(v);
                pairs.push((row, v));
            }
        }
        if uniq.is_empty() {
            continue;
        }
        uniq.sort_unstable_by(|a, b| cmp_f64(*a, *b));
        let mut m = 0usize;
        for i in 0..uniq.len() {
            let v = uniq[i];
            if m == 0 {
                uniq[m] = v;
                m += 1;
            } else {
                let last = uniq[m - 1];
                let tol = STD_EPS * 1.0_f64.max(last.abs());
                if (v - last).abs() > tol {
                    uniq[m] = v;
                    m += 1;
                }
            }
        }
        uniq.truncate(m);

        let mut cnt_tree = Fenwick::new(m);
        let mut seen_tree = Fenwick::new(m);
        let mut seen: Vec<bool> = vec![false; m];

        let mut oc = out.column_mut(col);
        for &(row, v) in pairs.iter() {
            let tol = STD_EPS * 1.0_f64.max(v.abs());
            let bin = uniq
                .binary_search_by(|u| {
                    if (u - v).abs() <= tol {
                        std::cmp::Ordering::Equal
                    } else {
                        cmp_f64(*u, v)
                    }
                })
                .unwrap_or_else(|pos| pos.min(m.saturating_sub(1)));
            cnt_tree.add(bin, 1);
            if !seen[bin] {
                seen[bin] = true;
                seen_tree.add(bin, 1);
            }
            let total = cnt_tree.prefix(m - 1) as f64;
            let n_distinct = seen_tree.prefix(m - 1) as f64;
            let (num, den) = if dense {
                (seen_tree.prefix(bin) as f64, n_distinct)
            } else {
                (cnt_tree.prefix(bin) as f64, total)
            };
            oc[row] = if pct {
                if den > 0.0 {
                    clean_f64(num / den)
                } else {
                    f64::NAN
                }
            } else {
                clean_f64(num)
            };
        }
    }
    out
}

/// `rank_ts(x)`：扩张窗口 1 基原始名次（严格因果版整段历史排名）。
pub fn op_rank_ts(x: &Array2<f64>) -> Array2<f64> {
    expanding_rank_core(x, false, false)
}

/// `pct_rank_ts(x)`：扩张窗口名次百分比（名次 / 有效样本数）。
pub fn op_pct_rank_ts(x: &Array2<f64>) -> Array2<f64> {
    expanding_rank_core(x, true, false)
}

/// `dense_rank_ts(x)`：扩张窗口稠密名次（1..D）。
pub fn op_dense_rank_ts(x: &Array2<f64>) -> Array2<f64> {
    expanding_rank_core(x, false, true)
}

/// `pct_dense_rank_ts(x)`：扩张窗口稠密名次百分比（稠密名次 / D）。
pub fn op_pct_dense_rank_ts(x: &Array2<f64>) -> Array2<f64> {
    expanding_rank_core(x, true, true)
}

/// `ts_dense_rank(x)`：等价于 `dense_rank_ts`（取整段历史末点即云端原语义）。
pub fn op_ts_dense_rank(x: &Array2<f64>) -> Array2<f64> {
    expanding_rank_core(x, false, true)
}

/// `ts_pct_dense_rank(x)`：等价于 `pct_dense_rank_ts`（稠密名次百分比）。
pub fn op_ts_pct_dense_rank(x: &Array2<f64>) -> Array2<f64> {
    expanding_rank_core(x, true, true)
}

/// `ts_cumsum(x)`：沿时间轴对有限观测累加（`NaN` 位置输出 `NaN`，累加器不受其影响）。
pub fn op_ts_cumsum(x: &Array2<f64>) -> Array2<f64> {
    let (t, n) = x.dim();
    let mut out = alloc_nan_matrix(t, n);
    for col in 0..n {
        let xc = x.column(col);
        let mut oc = out.column_mut(col);
        let mut acc = 0.0_f64;
        for row in 0..t {
            let v = xc[row];
            if v.is_finite() {
                acc += v;
                oc[row] = clean_f64(acc);
            }
        }
    }
    out
}

/// `ts_cumprod(x)`：沿时间轴对有限观测累乘（`NaN` 位置输出 `NaN`，累乘器不受其影响）。
pub fn op_ts_cumprod(x: &Array2<f64>) -> Array2<f64> {
    let (t, n) = x.dim();
    let mut out = alloc_nan_matrix(t, n);
    for col in 0..n {
        let xc = x.column(col);
        let mut oc = out.column_mut(col);
        let mut acc = 1.0_f64;
        let mut started = false;
        for row in 0..t {
            let v = xc[row];
            if v.is_finite() {
                acc *= v;
                started = true;
                oc[row] = clean_f64(acc);
            }
        }
        if !started {
            for row in 0..t {
                oc[row] = f64::NAN;
            }
        }
    }
    out
}

/// `ts_ffill(x)`：沿时间轴前向填充 `NaN`（首个有效值之前保持 `NaN`）。
pub fn op_ts_ffill(x: &Array2<f64>) -> Array2<f64> {
    let (t, n) = x.dim();
    let mut out = alloc_nan_matrix(t, n);
    for col in 0..n {
        let xc = x.column(col);
        let mut oc = out.column_mut(col);
        let mut last = f64::NAN;
        for row in 0..t {
            let v = xc[row];
            if v.is_finite() {
                last = v;
                oc[row] = v;
            } else if last.is_finite() {
                oc[row] = last;
            }
        }
    }
    out
}
