//! 变参多元（加权）最小二乘回归算子实现 (ops/multi.rs)。
//!
//! 对齐 `docs/factorlab_mcp_catalog.md` §4.5/§4.7 的 `cs_multi_*` / `ts_multi_*` 家族：
//! - `cs_multi_beta` / `cs_multi_resi`：逐交易时点横截面多元 OLS；
//! - `cs_multi_wls_beta` / `cs_multi_wls_resi`：逐交易时点横截面多元 WLS（权重矩阵 `w`）；
//! - `ts_multi_beta` / `ts_multi_resi`：逐标的扩张窗口因果多元 OLS；
//! - `ts_multi_wls_beta` / `ts_multi_wls_resi`：逐标的扩张窗口因果多元 WLS。
//!
//! 参数布局（`exps` 为 K 个暴露矩阵，`intercept` 决定是否附加常数项）：
//! - 非加权：`(y, exp_1, ..., exp_K [, intercept])`
//! - 加权：`(y, w, exp_1, ..., exp_K [, intercept])`
//!
//! 输出语义：
//! - `*_resi` 返回逐点残差 $r = y - X\hat{\beta}$（仅在全部输入有限的点上输出，其余为 `NaN`）；
//! - `*_beta` 返回**首个暴露矩阵 `exp_1` 的偏回归系数**（控制其余暴露后的净载荷），逐点广播；
//!   单暴露情形下即普通 OLS 斜率。多暴露的完整系数向量应通过 12 步后处理链
//!   `riskNeutMulti` 或 `BarraRiskModelStore` 获取。
//!
//! 数值稳定性：权重矩阵在装配正规方程前按有效样本权重最大值做**尺度归一化**
//! （WLS 的解对权重整体缩放不变），彻底避免 A 股市值量纲 $10^{10}\sim10^{12}$ 造成的
//! 正规方程条件数爆炸与跨算法（高斯消元 vs LAPACK）数值分歧。

use crate::nan::{clean_f64, EPS};
use crate::panel::alloc_nan_matrix;
use ndarray::Array2;

/// 求解小规模 $K \times K$ 线性方程组 $A x = b$（列主元高斯-约旦消元）。
#[allow(clippy::needless_range_loop)]
fn solve_linear_system(a: &mut [Vec<f64>], b: &[f64]) -> Option<Vec<f64>> {
    let k = b.len();
    if k == 0 {
        return Some(Vec::new());
    }
    let mut aug: Vec<Vec<f64>> = Vec::with_capacity(k);
    for (i, row) in a.iter().enumerate() {
        let mut r = Vec::with_capacity(k + 1);
        r.extend_from_slice(row);
        r.push(b[i]);
        aug.push(r);
    }
    for col in 0..k {
        let mut pivot = col;
        let mut max_abs = aug[col][col].abs();
        for (r, aug_row) in aug.iter().enumerate().skip(col + 1) {
            let v = aug_row[col].abs();
            if v > max_abs {
                max_abs = v;
                pivot = r;
            }
        }
        if max_abs.is_nan() || max_abs <= 1e-18 {
            return None;
        }
        if pivot != col {
            aug.swap(col, pivot);
        }
        let div_val = aug[col][col];
        for j in col..=k {
            aug[col][j] /= div_val;
        }
        for r in 0..k {
            if r != col {
                let factor = aug[r][col];
                if factor != 0.0 {
                    for j in col..=k {
                        let pivot_val = aug[col][j];
                        aug[r][j] -= factor * pivot_val;
                    }
                }
            }
        }
    }
    Some((0..k).map(|i| aug[i][k]).collect())
}

/// Householder QR 最小二乘求解（数值稳定版，等价于岭正则正规方程解）。
///
/// 输入 `a` 为 `(m, n)` 行主序设计矩阵，`b` 为 `(m,)` 响应向量；原地做 Householder 变换后回代。
#[allow(clippy::needless_range_loop)]
fn lstsq_qr(a: &mut [Vec<f64>], b: &mut [f64], n: usize) -> Option<Vec<f64>> {
    let m = a.len();
    if m < n || n == 0 {
        return None;
    }
    let mut v = vec![0.0_f64; m];
    for k in 0..n {
        let mut norm2 = 0.0_f64;
        for i in k..m {
            norm2 += a[i][k] * a[i][k];
        }
        let norm = norm2.sqrt();
        if norm.is_nan() || norm <= 1e-300 {
            return None;
        }
        let alpha = if a[k][k] > 0.0 { -norm } else { norm };
        v.iter_mut().for_each(|x| *x = 0.0);
        for i in k..m {
            v[i] = a[i][k];
        }
        v[k] -= alpha;
        let mut vnorm2 = 0.0_f64;
        for i in k..m {
            vnorm2 += v[i] * v[i];
        }
        if vnorm2.is_nan() || vnorm2 <= 0.0 {
            continue;
        }
        for j in k..n {
            let mut dot = 0.0_f64;
            for i in k..m {
                dot += v[i] * a[i][j];
            }
            let f = 2.0 * dot / vnorm2;
            for i in k..m {
                a[i][j] -= f * v[i];
            }
        }
        let mut dot = 0.0_f64;
        for i in k..m {
            dot += v[i] * b[i];
        }
        let f = 2.0 * dot / vnorm2;
        for i in k..m {
            b[i] -= f * v[i];
        }
    }
    let mut x = vec![0.0_f64; n];
    for i in (0..n).rev() {
        let mut s = b[i];
        for j in (i + 1)..n {
            s -= a[i][j] * x[j];
        }
        if a[i][i].abs() < 1e-300 {
            return None;
        }
        x[i] = s / a[i][i];
    }
    Some(x)
}

/// 正规方程岭正则项（防止极端停牌日或共线暴露导致奇异）。
pub const MULTI_RIDGE_EPS: f64 = 1.0e-10;

/// 判断单点观测是否有效（`y`、全部暴露有限，且加权时权重有限非负）。
#[inline]
fn obs_valid(
    y_val: f64,
    exps: &[&Array2<f64>],
    weights: Option<&Array2<f64>>,
    row: usize,
    col: usize,
) -> bool {
    if !y_val.is_finite() {
        return false;
    }
    for exp in exps {
        if !exp[[row, col]].is_finite() {
            return false;
        }
    }
    if let Some(w) = weights {
        let v = w[[row, col]];
        if !v.is_finite() || v < 0.0 {
            return false;
        }
    }
    true
}

#[allow(clippy::too_many_arguments)]
fn cs_multi_core(
    y: &Array2<f64>,
    exps: &[&Array2<f64>],
    weights: Option<&Array2<f64>>,
    intercept: bool,
    want_beta: bool,
) -> Array2<f64> {
    let (t, n) = y.dim();
    let mut out = alloc_nan_matrix(t, n);
    let k = exps.len();
    if k == 0 || t == 0 || n == 0 {
        return out;
    }
    for exp in exps {
        debug_assert_eq!((t, n), exp.dim());
    }
    if let Some(w) = weights {
        debug_assert_eq!((t, n), w.dim());
    }

    let dim = k + usize::from(intercept);

    for row in 0..t {
        // 1) 预扫描：有效观测数与权重归一化尺度
        let mut cnt = 0usize;
        let mut w_scale = 0.0_f64;
        for col in 0..n {
            if !obs_valid(y[[row, col]], exps, weights, row, col) {
                continue;
            }
            cnt += 1;
            if let Some(w) = weights {
                w_scale = w_scale.max(w[[row, col]]);
            }
        }
        if cnt == 0 || cnt < dim {
            continue;
        }
        let inv_scale = if weights.is_some() {
            if w_scale.is_nan() || w_scale <= 0.0 {
                continue;
            }
            1.0 / w_scale
        } else {
            1.0
        };

        // 2) 装配增广加权设计矩阵 [sqrt(w) X; sqrt(ridge) I]，以 Householder QR 求最小二乘解
        let mut aug: Vec<Vec<f64>> = Vec::with_capacity(cnt + dim);
        let mut aug_b: Vec<f64> = Vec::with_capacity(cnt + dim);
        for col in 0..n {
            let y_val = y[[row, col]];
            if !obs_valid(y_val, exps, weights, row, col) {
                continue;
            }
            let wv = match weights {
                Some(w) => w[[row, col]] * inv_scale,
                None => 1.0,
            };
            let sw = wv.sqrt();
            let mut r = vec![0.0_f64; dim];
            for i in 0..k {
                r[i] = sw * exps[i][[row, col]];
            }
            if intercept {
                r[k] = sw;
            }
            aug.push(r);
            aug_b.push(sw * y_val);
        }
        let sqrt_ridge = MULTI_RIDGE_EPS.sqrt();
        for i in 0..dim {
            let mut r = vec![0.0_f64; dim];
            r[i] = sqrt_ridge;
            aug.push(r);
            aug_b.push(0.0);
        }
        let Some(beta) = lstsq_qr(&mut aug, &mut aug_b, dim) else {
            continue;
        };

        // 3) 回填残差或首个暴露的偏回归系数
        for col in 0..n {
            let y_val = y[[row, col]];
            if !obs_valid(y_val, exps, weights, row, col) {
                continue;
            }
            if want_beta {
                out[[row, col]] = clean_f64(beta[0]);
            } else {
                let mut pred = 0.0_f64;
                for i in 0..k {
                    pred += beta[i] * exps[i][[row, col]];
                }
                if intercept {
                    pred += beta[k];
                }
                out[[row, col]] = clean_f64(y_val - pred);
            }
        }
    }
    out
}

#[allow(clippy::too_many_arguments)]
fn ts_multi_core(
    y: &Array2<f64>,
    exps: &[&Array2<f64>],
    weights: Option<&Array2<f64>>,
    intercept: bool,
    want_beta: bool,
) -> Array2<f64> {
    let (t, n) = y.dim();
    let mut out = alloc_nan_matrix(t, n);
    let k = exps.len();
    if k == 0 || t == 0 || n == 0 {
        return out;
    }
    for exp in exps {
        debug_assert_eq!((t, n), exp.dim());
    }
    if let Some(w) = weights {
        debug_assert_eq!((t, n), w.dim());
    }
    let dim = k + usize::from(intercept);
    let mut xs = vec![0.0_f64; dim];

    for col in 0..n {
        let yc = y.column(col);
        let mut a_mat = vec![vec![0.0_f64; dim]; dim];
        let mut b_vec = vec![0.0_f64; dim];
        let mut cnt = 0usize;
        // 严格因果的权重尺度：仅使用截至当前时点已观测到的最大权重
        let mut w_scale = 0.0_f64;

        for row in 0..t {
            let yv = yc[row];
            if !obs_valid(yv, exps, weights, row, col) {
                continue;
            }
            let raw_w = match weights {
                Some(w) => w[[row, col]],
                None => 1.0,
            };
            if weights.is_some() {
                w_scale = w_scale.max(raw_w);
                if w_scale.is_nan() || w_scale <= 0.0 {
                    continue;
                }
            }
            let wv = if weights.is_some() {
                raw_w / w_scale
            } else {
                1.0
            };
            for i in 0..k {
                xs[i] = exps[i][[row, col]];
            }
            if intercept {
                xs[k] = 1.0;
            }
            for i in 0..dim {
                b_vec[i] += wv * xs[i] * yv;
                for j in 0..dim {
                    a_mat[i][j] += wv * xs[i] * xs[j];
                }
            }
            cnt += 1;

            if cnt < dim + 1 {
                continue;
            }
            let mut scaled = a_mat.clone();
            for (i, a_row) in scaled.iter_mut().enumerate() {
                a_row[i] += MULTI_RIDGE_EPS;
            }
            let Some(sol) = solve_linear_system(&mut scaled, &b_vec) else {
                continue;
            };
            let mut oc = out.column_mut(col);
            if want_beta {
                oc[row] = clean_f64(sol[0]);
            } else {
                let mut pred = 0.0_f64;
                for i in 0..k {
                    pred += sol[i] * exps[i][[row, col]];
                }
                if intercept {
                    pred += sol[k];
                }
                oc[row] = clean_f64(yv - pred);
            }
        }
    }
    out
}

/// 统一分派入口：按算子类型调用横截面或时序多元（加权）回归核心。
pub fn op_multi(
    op: crate::ops::NaryOp,
    y: &Array2<f64>,
    exps: &[&Array2<f64>],
    weights: Option<&Array2<f64>>,
    intercept: bool,
) -> Array2<f64> {
    let want_beta = op.returns_beta();
    if op.is_time_series() {
        ts_multi_core(y, exps, weights, intercept, want_beta)
    } else {
        cs_multi_core(y, exps, weights, intercept, want_beta)
    }
}

/// 判断常量是否为“真”截距开关（非零即开启；`NaN` 视为关闭）。
#[inline]
pub fn intercept_flag(c: f64) -> bool {
    c.is_finite() && c.abs() > EPS
}
