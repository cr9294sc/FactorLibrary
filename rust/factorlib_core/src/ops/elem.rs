//! 向量化逐元素算子实现 (ops/elem.rs)。
//!
//! 涵盖四类逐元素运算：
//! 1. 一元逐元素算子 (`neg`, `abs`, `sign`, `inv`, `slog`, `ssqrt`, `square`, `cube`, `sigmoid`, `tanh`, `relu`, `atan`)
//! 2. 二元逐元素算子 (`add`, `sub`, `mul`, `div`, `max2`, `min2`, `hypot`, `copysign`)
//! 3. 标量常数算子 (`addc`, `subc`, `mulc`, `divc`, `spow`, `clip`)
//! 4. 比较、布尔逻辑与三元条件算子 (`gt`, `lt`, `ge`, `le`, `gt_const`, `lt_const`, `and_cond`, `or_cond`, `not_cond`, `where`)

use crate::nan::{clean_f64, EPS};
use crate::panel::alloc_nan_matrix;
use ndarray::{Array2, ShapeBuilder};

#[inline]
fn map_unary<F>(x: &Array2<f64>, f: F) -> Array2<f64>
where
    F: Fn(f64) -> f64,
{
    let (t, n) = x.dim();
    if let Some(xs) = x.as_slice_memory_order() {
        if !x.is_standard_layout() || t <= 1 || n <= 1 {
            let mut out = Array2::<f64>::zeros((t, n).f());
            let os = out.as_slice_memory_order_mut().unwrap();
            for (o, &v) in os.iter_mut().zip(xs.iter()) {
                *o = if v.is_finite() {
                    clean_f64(f(v))
                } else {
                    f64::NAN
                };
            }
            return out;
        }
    }
    let mut out = alloc_nan_matrix(t, n);
    for ((r, c), &v) in x.indexed_iter() {
        out[[r, c]] = if v.is_finite() {
            clean_f64(f(v))
        } else {
            f64::NAN
        };
    }
    out
}

#[inline]
fn map_binary<F>(x: &Array2<f64>, y: &Array2<f64>, f: F) -> Array2<f64>
where
    F: Fn(f64, f64) -> f64,
{
    let (t, n) = x.dim();
    debug_assert_eq!((t, n), y.dim());
    if let (Some(xs), Some(ys)) = (x.as_slice_memory_order(), y.as_slice_memory_order()) {
        if (!x.is_standard_layout() && !y.is_standard_layout()) || t <= 1 || n <= 1 {
            let mut out = Array2::<f64>::zeros((t, n).f());
            let os = out.as_slice_memory_order_mut().unwrap();
            for (o, (&vx, &vy)) in os.iter_mut().zip(xs.iter().zip(ys.iter())) {
                *o = if vx.is_finite() && vy.is_finite() {
                    clean_f64(f(vx, vy))
                } else {
                    f64::NAN
                };
            }
            return out;
        }
    }
    let mut out = alloc_nan_matrix(t, n);
    for col in 0..n {
        let xc = x.column(col);
        let yc = y.column(col);
        let mut oc = out.column_mut(col);
        for row in 0..t {
            let vx = xc[row];
            let vy = yc[row];
            oc[row] = if vx.is_finite() && vy.is_finite() {
                clean_f64(f(vx, vy))
            } else {
                f64::NAN
            };
        }
    }
    out
}

// ============================================================================
// 1. 一元逐元素算子 (12 个)
// ============================================================================

pub fn op_neg(x: &Array2<f64>) -> Array2<f64> {
    map_unary(x, |v| -v)
}

pub fn op_abs(x: &Array2<f64>) -> Array2<f64> {
    map_unary(x, |v| v.abs())
}

pub fn op_sign(x: &Array2<f64>) -> Array2<f64> {
    map_unary(x, |v| {
        if v > 0.0 {
            1.0
        } else if v < 0.0 {
            -1.0
        } else {
            0.0
        }
    })
}

pub fn op_inv(x: &Array2<f64>) -> Array2<f64> {
    map_unary(x, |v| if v.abs() >= EPS { 1.0 / v } else { f64::NAN })
}

pub fn op_slog(x: &Array2<f64>) -> Array2<f64> {
    map_unary(x, |v| {
        let s = if v > 0.0 {
            1.0
        } else if v < 0.0 {
            -1.0
        } else {
            0.0
        };
        s * v.abs().ln_1p()
    })
}

pub fn op_ssqrt(x: &Array2<f64>) -> Array2<f64> {
    map_unary(x, |v| {
        let s = if v > 0.0 {
            1.0
        } else if v < 0.0 {
            -1.0
        } else {
            0.0
        };
        s * v.abs().sqrt()
    })
}

pub fn op_square(x: &Array2<f64>) -> Array2<f64> {
    map_unary(x, |v| v * v)
}

pub fn op_cube(x: &Array2<f64>) -> Array2<f64> {
    map_unary(x, |v| v * v * v)
}

pub fn op_sigmoid(x: &Array2<f64>) -> Array2<f64> {
    map_unary(x, |v| {
        if v >= 0.0 {
            1.0 / (1.0 + (-v).exp())
        } else {
            let ev = v.exp();
            ev / (1.0 + ev)
        }
    })
}

pub fn op_tanh(x: &Array2<f64>) -> Array2<f64> {
    map_unary(x, |v| v.tanh())
}

pub fn op_relu(x: &Array2<f64>) -> Array2<f64> {
    map_unary(x, |v| if v > 0.0 { v } else { 0.0 })
}

pub fn op_atan(x: &Array2<f64>) -> Array2<f64> {
    map_unary(x, |v| v.atan())
}

// ============================================================================
// 2. 二元逐元素算子 (8 个)
// ============================================================================

pub fn op_add(x: &Array2<f64>, y: &Array2<f64>) -> Array2<f64> {
    map_binary(x, y, |a, b| a + b)
}

pub fn op_sub(x: &Array2<f64>, y: &Array2<f64>) -> Array2<f64> {
    map_binary(x, y, |a, b| a - b)
}

pub fn op_mul(x: &Array2<f64>, y: &Array2<f64>) -> Array2<f64> {
    map_binary(x, y, |a, b| a * b)
}

pub fn op_div(x: &Array2<f64>, y: &Array2<f64>) -> Array2<f64> {
    map_binary(x, y, |a, b| if b.abs() >= EPS { a / b } else { f64::NAN })
}

pub fn op_max2(x: &Array2<f64>, y: &Array2<f64>) -> Array2<f64> {
    map_binary(x, y, |a, b| if a >= b { a } else { b })
}

pub fn op_min2(x: &Array2<f64>, y: &Array2<f64>) -> Array2<f64> {
    map_binary(x, y, |a, b| if a <= b { a } else { b })
}

pub fn op_hypot(x: &Array2<f64>, y: &Array2<f64>) -> Array2<f64> {
    map_binary(x, y, |a, b| a.hypot(b))
}

pub fn op_copysign(x: &Array2<f64>, y: &Array2<f64>) -> Array2<f64> {
    map_binary(x, y, |a, b| {
        let s = if b > 0.0 {
            1.0
        } else if b < 0.0 {
            -1.0
        } else {
            0.0
        };
        a.abs() * s
    })
}

// ============================================================================
// 3. 标量常数算子 (6 个)
// ============================================================================

pub fn op_addc(x: &Array2<f64>, c: f64) -> Array2<f64> {
    if !c.is_finite() {
        return alloc_nan_matrix(x.nrows(), x.ncols());
    }
    map_unary(x, |v| v + c)
}

pub fn op_subc(x: &Array2<f64>, c: f64) -> Array2<f64> {
    if !c.is_finite() {
        return alloc_nan_matrix(x.nrows(), x.ncols());
    }
    map_unary(x, |v| v - c)
}

pub fn op_mulc(x: &Array2<f64>, c: f64) -> Array2<f64> {
    if !c.is_finite() {
        return alloc_nan_matrix(x.nrows(), x.ncols());
    }
    map_unary(x, |v| v * c)
}

pub fn op_divc(x: &Array2<f64>, c: f64) -> Array2<f64> {
    if !c.is_finite() || c.abs() < EPS {
        return alloc_nan_matrix(x.nrows(), x.ncols());
    }
    map_unary(x, |v| v / c)
}

pub fn op_spow(x: &Array2<f64>, c: f64) -> Array2<f64> {
    if !c.is_finite() {
        return alloc_nan_matrix(x.nrows(), x.ncols());
    }
    map_unary(x, |v| {
        let av = v.abs();
        if av < EPS && c < 0.0 {
            return f64::NAN;
        }
        if av == 0.0 {
            return if c == 0.0 { 1.0 } else { 0.0 };
        }
        let s = if v > 0.0 { 1.0 } else { -1.0 };
        s * av.powf(c)
    })
}

pub fn op_clip(x: &Array2<f64>, c: f64) -> Array2<f64> {
    if !c.is_finite() {
        return alloc_nan_matrix(x.nrows(), x.ncols());
    }
    let bound = c.abs();
    map_unary(x, |v| v.clamp(-bound, bound))
}

// ============================================================================
// 4. 比较、布尔逻辑与三元条件算子 (10 个)
// ============================================================================

pub fn op_gt(x: &Array2<f64>, y: &Array2<f64>) -> Array2<f64> {
    map_binary(x, y, |a, b| if a > b { 1.0 } else { 0.0 })
}

pub fn op_lt(x: &Array2<f64>, y: &Array2<f64>) -> Array2<f64> {
    map_binary(x, y, |a, b| if a < b { 1.0 } else { 0.0 })
}

pub fn op_ge(x: &Array2<f64>, y: &Array2<f64>) -> Array2<f64> {
    map_binary(x, y, |a, b| if a >= b { 1.0 } else { 0.0 })
}

pub fn op_le(x: &Array2<f64>, y: &Array2<f64>) -> Array2<f64> {
    map_binary(x, y, |a, b| if a <= b { 1.0 } else { 0.0 })
}

pub fn op_gt_const(x: &Array2<f64>, c: f64) -> Array2<f64> {
    if !c.is_finite() {
        return alloc_nan_matrix(x.nrows(), x.ncols());
    }
    map_unary(x, |v| if v > c { 1.0 } else { 0.0 })
}

pub fn op_lt_const(x: &Array2<f64>, c: f64) -> Array2<f64> {
    if !c.is_finite() {
        return alloc_nan_matrix(x.nrows(), x.ncols());
    }
    map_unary(x, |v| if v < c { 1.0 } else { 0.0 })
}

pub fn op_and_cond(a: &Array2<f64>, b: &Array2<f64>) -> Array2<f64> {
    map_binary(a, b, |va, vb| if va > 0.5 && vb > 0.5 { 1.0 } else { 0.0 })
}

pub fn op_or_cond(a: &Array2<f64>, b: &Array2<f64>) -> Array2<f64> {
    map_binary(a, b, |va, vb| if va > 0.5 || vb > 0.5 { 1.0 } else { 0.0 })
}

pub fn op_not_cond(a: &Array2<f64>) -> Array2<f64> {
    map_unary(a, |va| if va <= 0.5 { 1.0 } else { 0.0 })
}

pub fn op_where(cond: &Array2<f64>, then_b: &Array2<f64>, else_b: &Array2<f64>) -> Array2<f64> {
    let (t, n) = cond.dim();
    debug_assert_eq!((t, n), then_b.dim());
    debug_assert_eq!((t, n), else_b.dim());
    let mut out = alloc_nan_matrix(t, n);
    for col in 0..n {
        let cc = cond.column(col);
        let tc = then_b.column(col);
        let ec = else_b.column(col);
        let mut oc = out.column_mut(col);
        for row in 0..t {
            let vc = cc[row];
            if !vc.is_finite() {
                oc[row] = f64::NAN;
            } else if vc > 0.5 {
                oc[row] = clean_f64(tc[row]);
            } else {
                oc[row] = clean_f64(ec[row]);
            }
        }
    }
    out
}

// ============================================================================
// 5. FactorLab 1d 数学变换扩充 (4 个): exp, ceil, sign_power, unique_count
// ============================================================================

/// `exp(x)`：逐元素自然指数；上溢至 `+Inf` 时按平台 NaN 策略统一归一为 `NaN`。
pub fn op_exp(x: &Array2<f64>) -> Array2<f64> {
    map_unary(x, |v| v.exp())
}

/// `ceil(x)`：逐元素向上取整。
pub fn op_ceil(x: &Array2<f64>) -> Array2<f64> {
    map_unary(x, |v| v.ceil())
}

/// `sign_power(x)`：带符号平方 $\text{sign}(x) \cdot x^2$（单调、保号、幂次放大）。
pub fn op_sign_power(x: &Array2<f64>) -> Array2<f64> {
    map_unary(x, |v| {
        let s = if v > 0.0 {
            1.0
        } else if v < 0.0 {
            -1.0
        } else {
            0.0
        };
        s * v * v
    })
}

/// `unique_count(x)`：逐截面统计不同非 `NaN` 数值个数（`±Inf` 已按平台策略归一为 `NaN`）。
pub fn op_unique_count(x: &Array2<f64>) -> Array2<f64> {
    let (t, n) = x.dim();
    let mut out = alloc_nan_matrix(t, n);
    if t == 0 || n == 0 {
        return out;
    }
    let mut vals: Vec<f64> = Vec::with_capacity(n);
    for row in 0..t {
        vals.clear();
        for col in 0..n {
            let v = x[[row, col]];
            if v.is_finite() {
                vals.push(v);
            }
        }
        if vals.is_empty() {
            continue;
        }
        vals.sort_unstable_by(|a, b| crate::nan::cmp_f64(*a, *b));
        let mut count = 1usize;
        for i in 1..vals.len() {
            let tol = crate::nan::STD_EPS * 1.0_f64.max(vals[i].abs());
            if (vals[i] - vals[i - 1]).abs() > tol {
                count += 1;
            }
        }
        let cnt_f = count as f64;
        for col in 0..n {
            if x[[row, col]].is_finite() {
                out[[row, col]] = cnt_f;
            }
        }
    }
    out
}

// ============================================================================
// 6. FactorLab 1d 缺失值与极值处理 (4 个): is_inf, is_nan, purify, zero_to_nan
// ============================================================================

/// 逐元素批量映射，保留 `NaN` 原样（仅供缺失值探测/清洗类算子使用）。
#[inline]
fn map_raw<F>(x: &Array2<f64>, f: F) -> Array2<f64>
where
    F: Fn(f64) -> f64,
{
    let (t, n) = x.dim();
    let mut out = alloc_nan_matrix(t, n);
    for col in 0..n {
        let xc = x.column(col);
        let mut oc = out.column_mut(col);
        for row in 0..t {
            let v = xc[row];
            oc[row] = if v.is_nan() { f64::NAN } else { f(v) };
        }
    }
    out
}

/// `is_inf(x)`：正负无穷返回 `1.0`，有限值返回 `0.0`，`NaN` 保持 `NaN`。
pub fn op_is_inf(x: &Array2<f64>) -> Array2<f64> {
    map_raw(x, |v| if v.is_infinite() { 1.0 } else { 0.0 })
}

/// `is_nan(x)`：`NaN` 返回 `1.0`，其余（含 `±Inf`）返回 `0.0`。
pub fn op_is_nan(x: &Array2<f64>) -> Array2<f64> {
    let (t, n) = x.dim();
    let mut out = alloc_nan_matrix(t, n);
    for col in 0..n {
        let xc = x.column(col);
        let mut oc = out.column_mut(col);
        for row in 0..t {
            oc[row] = if xc[row].is_nan() { 1.0 } else { 0.0 };
        }
    }
    out
}

/// `purify(x)`：将正负无穷等非有限值（除 `NaN` 外）转换为 `NaN`。
pub fn op_purify(x: &Array2<f64>) -> Array2<f64> {
    map_raw(x, |v| if v.is_finite() { v } else { f64::NAN })
}

/// `zero_to_nan(x)`：将精确零值转换为 `NaN`。
pub fn op_zero_to_nan(x: &Array2<f64>) -> Array2<f64> {
    map_unary(x, |v| if v == 0.0 { f64::NAN } else { v })
}

// ============================================================================
// 7. FactorLab 1d 常数参数算子 (3 个): fillna, round, perc_long
// ============================================================================

/// `fillna(x, fill_value)`：仅把 `NaN` 替换为 `fill_value`（`±Inf` 依平台策略归一为 `NaN`）。
pub fn op_fillna(x: &Array2<f64>, fill_value: f64) -> Array2<f64> {
    let (t, n) = x.dim();
    let mut out = alloc_nan_matrix(t, n);
    if !fill_value.is_finite() {
        return out;
    }
    for col in 0..n {
        let xc = x.column(col);
        let mut oc = out.column_mut(col);
        for row in 0..t {
            let v = xc[row];
            oc[row] = if v.is_finite() {
                v
            } else if v.is_nan() {
                fill_value
            } else {
                f64::NAN
            };
        }
    }
    out
}

/// `round(x, decimals)`：逐元素按 `decimals` 位舍入，恰好半值采用向偶数舍入。
pub fn op_round_decimals(x: &Array2<f64>, decimals: f64) -> Array2<f64> {
    if !decimals.is_finite() {
        return alloc_nan_matrix(x.nrows(), x.ncols());
    }
    let d = (decimals.round() as i32).clamp(-15, 15);
    let scale = 10.0_f64.powi(d);
    map_unary(x, |v| (v * scale).round_ties_even() / scale)
}

/// `perc_long(x, percentile)`：先减截面分位点，正值保留、负值统一替换为
/// `-sum(正值) / count(负值)` 以均摊多头总量，形成零净额多空数值。
pub fn op_perc_long(x: &Array2<f64>, percentile: f64) -> Array2<f64> {
    let (t, n) = x.dim();
    let mut out = alloc_nan_matrix(t, n);
    if t == 0 || n == 0 || !percentile.is_finite() {
        return out;
    }
    let q = percentile.clamp(0.0, 1.0);
    let mut vals: Vec<f64> = Vec::with_capacity(n);

    for row in 0..t {
        vals.clear();
        for col in 0..n {
            let v = x[[row, col]];
            if v.is_finite() {
                vals.push(v);
            }
        }
        if vals.is_empty() {
            continue;
        }
        vals.sort_unstable_by(|a, b| crate::nan::cmp_f64(*a, *b));
        let k = vals.len();
        let pos = q * ((k - 1) as f64);
        let lo = pos.floor() as usize;
        let hi = pos.ceil() as usize;
        let frac = pos - (lo as f64);
        let quant = vals[lo] + frac * (vals[hi] - vals[lo]);

        let mut pos_sum = 0.0_f64;
        let mut neg_cnt = 0usize;
        for col in 0..n {
            let v = x[[row, col]];
            if v.is_finite() {
                let d = v - quant;
                if d > 0.0 {
                    pos_sum += d;
                } else if d < 0.0 {
                    neg_cnt += 1;
                }
            }
        }
        let neg_fill = if neg_cnt > 0 {
            -(pos_sum / (neg_cnt as f64))
        } else {
            0.0
        };
        for col in 0..n {
            let v = x[[row, col]];
            if v.is_finite() {
                let d = v - quant;
                out[[row, col]] = clean_f64(if d > 0.0 {
                    d
                } else if d < 0.0 {
                    neg_fill
                } else {
                    0.0
                });
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_factorlab_missing_value_probe_semantics() {
        // ±Inf / NaN / 零值 / 有限值的四分类边界（平台 NaN 策略：Inf 归一为 NaN 仅在面板装载边界，
        // 算子内部必须保留原始 Inf 语义以支撑 is_inf / purify 的探测能力）
        let x = array![[f64::INFINITY, f64::NEG_INFINITY, f64::NAN, 0.0, 2.5]];
        let inf_mask = op_is_inf(&x);
        assert_eq!(inf_mask[[0, 0]], 1.0);
        assert_eq!(inf_mask[[0, 1]], 1.0);
        assert!(inf_mask[[0, 2]].is_nan());
        assert_eq!(inf_mask[[0, 3]], 0.0);
        assert_eq!(inf_mask[[0, 4]], 0.0);

        let nan_mask = op_is_nan(&x);
        assert_eq!(nan_mask[[0, 0]], 0.0);
        assert_eq!(nan_mask[[0, 1]], 0.0);
        assert_eq!(nan_mask[[0, 2]], 1.0);
        assert_eq!(nan_mask[[0, 4]], 0.0);

        let purified = op_purify(&x);
        assert!(purified[[0, 0]].is_nan());
        assert!(purified[[0, 1]].is_nan());
        assert!(purified[[0, 2]].is_nan());
        assert_eq!(purified[[0, 3]], 0.0);
        assert_eq!(purified[[0, 4]], 2.5);

        let z2n = op_zero_to_nan(&x);
        assert!(z2n[[0, 3]].is_nan());
        assert_eq!(z2n[[0, 4]], 2.5);
        assert!(z2n[[0, 0]].is_nan());
        assert!(z2n[[0, 2]].is_nan());

        let filled = op_fillna(&x, -1.0);
        assert_eq!(filled[[0, 2]], -1.0);
        assert_eq!(filled[[0, 3]], 0.0);
        assert_eq!(filled[[0, 4]], 2.5);
        assert!(filled[[0, 0]].is_nan());
    }

    #[test]
    fn test_factorlab_math_transform_ops() {
        let x = array![[-2.0, -0.5, 0.0, 0.5, 2.0, f64::NAN]];
        let sp = op_sign_power(&x);
        assert_eq!(sp[[0, 0]], -4.0);
        assert_eq!(sp[[0, 1]], -0.25);
        assert_eq!(sp[[0, 2]], 0.0);
        assert_eq!(sp[[0, 3]], 0.25);
        assert_eq!(sp[[0, 4]], 4.0);
        assert!(sp[[0, 5]].is_nan());

        let cl = op_ceil(&x);
        assert_eq!(cl[[0, 1]], 0.0);
        assert_eq!(cl[[0, 3]], 1.0);

        let ex = op_exp(&x);
        assert!((ex[[0, 2]] - 1.0).abs() < 1e-12);
        assert!((ex[[0, 4]] - 2.0_f64.exp()).abs() < 1e-9);

        // 向偶数舍入（round half to even）
        let r = array![[0.5, 1.5, 2.5, -0.5, -1.5, f64::NAN]];
        let rd = op_round_decimals(&r, 0.0);
        assert_eq!(rd[[0, 0]], 0.0);
        assert_eq!(rd[[0, 1]], 2.0);
        assert_eq!(rd[[0, 2]], 2.0);
        assert_eq!(rd[[0, 3]], -0.0);
        assert_eq!(rd[[0, 4]], -2.0);
    }

    #[test]
    fn test_elem_zeros_alloc_small_and_corder_shapes() {
        let a_row = array![[-2.0, f64::NAN, 3.0, f64::INFINITY]]; // (1, 4) C-order
        let b_row = array![[1.0, 2.0, -0.5, 4.0]];
        let abs_row = op_abs(&a_row);
        assert_eq!(abs_row[[0, 0]], 2.0);
        assert!(abs_row[[0, 1]].is_nan());
        assert_eq!(abs_row[[0, 2]], 3.0);
        assert!(abs_row[[0, 3]].is_nan());

        let add_row = op_add(&a_row, &b_row);
        assert_eq!(add_row[[0, 0]], -1.0);
        assert!(add_row[[0, 1]].is_nan());
        assert_eq!(add_row[[0, 2]], 2.5);
        assert!(add_row[[0, 3]].is_nan());

        let a_col = Array2::from_shape_vec((4, 1).f(), vec![-1.0, 2.0, f64::NAN, 0.0]).unwrap();
        let b_col = Array2::from_shape_vec((4, 1).f(), vec![3.0, -4.0, 1.0, 5.0]).unwrap();
        let add_col = op_add(&a_col, &b_col);
        assert_eq!(add_col[[0, 0]], 2.0);
        assert_eq!(add_col[[1, 0]], -2.0);
        assert!(add_col[[2, 0]].is_nan());
        assert_eq!(add_col[[3, 0]], 5.0);
    }
}
