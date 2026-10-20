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
            let mut out = unsafe { Array2::<f64>::uninit((t, n).f()).assume_init() };
            let os = out.as_slice_memory_order_mut().unwrap();
            for (o, &v) in os.iter_mut().zip(xs.iter()) {
                *o = if v.is_finite() { clean_f64(f(v)) } else { f64::NAN };
            }
            return out;
        }
    }
    let mut out = alloc_nan_matrix(t, n);
    for ((r, c), &v) in x.indexed_iter() {
        out[[r, c]] = if v.is_finite() { clean_f64(f(v)) } else { f64::NAN };
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
            let mut out = unsafe { Array2::<f64>::uninit((t, n).f()).assume_init() };
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
