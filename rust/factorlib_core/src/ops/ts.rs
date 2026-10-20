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
