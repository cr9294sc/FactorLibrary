//! 12 项声明式后处理算子实现 (ops/postprocess.rs)。
//!
//! 严格对齐 FactorLab C++ 运行时 (`ares-infra/aresexpradapter`, `runtime.cpp`) 的数学定义与边界语义，
//! 支持隐式依赖注入与链式执行，所有输出均为 `(T, N)` Fortran 列优先矩阵：
//! 1. `baseUniMask` (`Mul` with `in_universe_mask`)
//! 2. `trdMask` (`Mul` with `limit_mask`)
//! 3. `dropNewStock` (`ZsimDropNewStock`, `listed_days >= min_days`)
//! 4. `MAD` (`ZsimMad`, 下中位数 `(N-1)/2` 与 `1.4826 * threshold` 裁剪)
//! 5. `winsor` (`ZsimWinsor`, 线性内插分位数 `[q, 1-q]` 裁剪)
//! 6. `indNeut` (`ZsimIndustryNeutral`, 对 `sector > 0` 减去同板块均值，缺失或 `<= 0` 保留原值)
//! 7. `riskNeut` (`ZsimRiskNeutral`, 带截距截面 OLS 残差，缺失暴露保留原值)
//! 8. `riskNeutMulti` (`ZsimCrossResidual`, 无截距多变量截面 $10^{-8}$ 岭回归残差)
//! 9. `zScore` (`ZsimCrossZScore`, 总体标准差 `ddof=0` 截面标准化)
//! 10. `rank` (`ZsimPctRank`, $\sigma_{\text{ddof}=0} < 10^{-6}$ 保护 + 6 位小数截断稳定排序百分位排名)
//! 11. `decay` (`ZsimRollingMean`, $t < W-1$ 置 `NaN` 的沿时间轴滚动算术均值)
//! 12. `emadecay` (`ZsimRollingWma`, $t < W-1$ 置 `NaN` 的 $1\dots W$ 线性加权移动平均，缺失权重重归一化)

use crate::nan::cmp_f64;
use crate::panel::alloc_nan_matrix;
use ahash::AHashMap;
use ndarray::Array2;

/// 1 & 2. `baseUniMask` / `trdMask`: 逐元素乘法掩码过滤（Opcode: `Mul`）。
pub fn pp_mask_mul(x1: &Array2<f64>, x2: &Array2<f64>) -> Array2<f64> {
    let (t, n) = x1.dim();
    debug_assert_eq!((t, n), x2.dim());
    let mut out = alloc_nan_matrix(t, n);
    for col in 0..n {
        let c1 = x1.column(col);
        let c2 = x2.column(col);
        let mut oc = out.column_mut(col);
        for row in 0..t {
            let v1 = c1[row];
            let v2 = c2[row];
            if v1.is_finite() && v2.is_finite() {
                let prod = v1 * v2;
                if prod.is_finite() {
                    oc[row] = prod;
                }
            }
        }
    }
    out
}

/// 3. `dropNewStock`: 过滤上市不足 `min_days` 的标的（Opcode: `ZsimDropNewStock`）。
pub fn pp_drop_new_stock(x1: &Array2<f64>, x2: &Array2<f64>, min_days: f64) -> Array2<f64> {
    let (t, n) = x1.dim();
    debug_assert_eq!((t, n), x2.dim());
    let mut out = alloc_nan_matrix(t, n);
    for col in 0..n {
        let c1 = x1.column(col);
        let c2 = x2.column(col);
        let mut oc = out.column_mut(col);
        for row in 0..t {
            let v1 = c1[row];
            let v2 = c2[row];
            if v1.is_finite() && v2.is_finite() && v2 >= min_days {
                oc[row] = v1;
            }
        }
    }
    out
}

/// 4. `MAD`: 截面下中位数绝对偏差极值裁剪（Opcode: `ZsimMad`）。
///
/// 采用下中位数（排序后下标 `(k - 1) / 2`）：
/// $m = \text{median}_{\text{lower}}(x_1)$，$\text{mad} = \text{median}_{\text{lower}}(|x_1 - m|)$，
/// 裁剪至 $[m - \text{threshold} \cdot 1.4826 \cdot \text{mad}, \; m + \text{threshold} \cdot 1.4826 \cdot \text{mad}]$。
pub fn pp_mad(x1: &Array2<f64>, threshold: f64) -> Array2<f64> {
    let (t, n) = x1.dim();
    let mut out = alloc_nan_matrix(t, n);
    if t == 0 || n == 0 || !threshold.is_finite() {
        return out;
    }

    let mut finite_vals: Vec<f64> = Vec::with_capacity(n);
    let mut abs_devs: Vec<f64> = Vec::with_capacity(n);

    for row in 0..t {
        finite_vals.clear();
        for col in 0..n {
            let v = x1[[row, col]];
            if v.is_finite() {
                finite_vals.push(v);
            }
        }
        if finite_vals.is_empty() {
            continue;
        }
        finite_vals.sort_by(|a, b| cmp_f64(*a, *b));
        let k = finite_vals.len();
        let med = finite_vals[(k - 1) / 2];

        abs_devs.clear();
        for &v in &finite_vals {
            abs_devs.push((v - med).abs());
        }
        abs_devs.sort_by(|a, b| cmp_f64(*a, *b));
        let mad = abs_devs[(k - 1) / 2];

        let radius = threshold * 1.4826 * mad;
        let lo = med - radius;
        let hi = med + radius;

        for col in 0..n {
            let v = x1[[row, col]];
            if v.is_finite() {
                out[[row, col]] = v.clamp(lo, hi);
            }
        }
    }
    out
}

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

/// 5. `winsor`: 截面线性内插分位数裁剪（Opcode: `ZsimWinsor`）。
pub fn pp_winsor(x1: &Array2<f64>, quantile: f64) -> Array2<f64> {
    let (t, n) = x1.dim();
    let mut out = alloc_nan_matrix(t, n);
    if t == 0 || n == 0 || !quantile.is_finite() {
        return out;
    }

    let q = quantile.clamp(0.0, 0.5);
    let mut finite_vals: Vec<f64> = Vec::with_capacity(n);

    for row in 0..t {
        finite_vals.clear();
        for col in 0..n {
            let v = x1[[row, col]];
            if v.is_finite() {
                finite_vals.push(v);
            }
        }
        if finite_vals.is_empty() {
            continue;
        }
        finite_vals.sort_by(|a, b| cmp_f64(*a, *b));
        let lo = linear_quantile_sorted(&finite_vals, q);
        let hi = linear_quantile_sorted(&finite_vals, 1.0 - q);

        for col in 0..n {
            let v = x1[[row, col]];
            if v.is_finite() {
                out[[row, col]] = v.clamp(lo, hi);
            }
        }
    }
    out
}

/// 6. `indNeut`: 行业/期货板块截面去均值（Opcode: `ZsimIndustryNeutral`）。
///
/// 对 `x2 > 0` 的有效板块编码减去同板块均值；若 `x2` 为 `NaN` 或 `<= 0`，**保留原值 `x1`**。
pub fn pp_ind_neut(x1: &Array2<f64>, x2: &Array2<f64>) -> Array2<f64> {
    let (t, n) = x1.dim();
    debug_assert_eq!((t, n), x2.dim());
    let mut out = alloc_nan_matrix(t, n);
    let mut groups: AHashMap<u64, (f64, usize)> = AHashMap::with_capacity(16);

    for row in 0..t {
        groups.clear();
        for col in 0..n {
            let v1 = x1[[row, col]];
            let v2 = x2[[row, col]];
            if v1.is_finite() && v2.is_finite() && v2 > 0.0 {
                let key = v2.to_bits();
                let entry = groups.entry(key).or_insert((0.0, 0));
                entry.0 += v1;
                entry.1 += 1;
            }
        }
        for col in 0..n {
            let v1 = x1[[row, col]];
            let v2 = x2[[row, col]];
            if !v1.is_finite() {
                out[[row, col]] = f64::NAN;
            } else if !v2.is_finite() || v2 <= 0.0 {
                out[[row, col]] = v1;
            } else {
                let key = v2.to_bits();
                if let Some(&(sum, cnt)) = groups.get(&key) {
                    if cnt > 0 {
                        out[[row, col]] = v1 - (sum / (cnt as f64));
                    } else {
                        out[[row, col]] = v1;
                    }
                } else {
                    out[[row, col]] = v1;
                }
            }
        }
    }
    out
}

/// 7. `riskNeut`: 带截距单变量截面 OLS 风险中性化（Opcode: `ZsimRiskNeutral`）。
///
/// - 斜率 $\beta$ 使用 $(x_1, x_2)$ 同时有限的样本计算；
/// - 截距 $\alpha = \text{mean}(x_1) - \beta \cdot \text{mean}(x_2)$ 分别在各自有限样本上独立计算均值；
/// - 若 $x_2$ 缺失（`NaN`），保留原值 $x_1$。
pub fn pp_risk_neut(x1: &Array2<f64>, x2: &Array2<f64>) -> Array2<f64> {
    let (t, n) = x1.dim();
    debug_assert_eq!((t, n), x2.dim());
    let mut out = alloc_nan_matrix(t, n);

    for row in 0..t {
        let mut sum1_both = 0.0_f64;
        let mut sum2_both = 0.0_f64;
        let mut cnt_both = 0usize;

        let mut sum1_all = 0.0_f64;
        let mut cnt1_all = 0usize;
        let mut sum2_all = 0.0_f64;
        let mut cnt2_all = 0usize;

        for col in 0..n {
            let v1 = x1[[row, col]];
            let v2 = x2[[row, col]];
            let f1 = v1.is_finite();
            let f2 = v2.is_finite();
            if f1 {
                sum1_all += v1;
                cnt1_all += 1;
            }
            if f2 {
                sum2_all += v2;
                cnt2_all += 1;
            }
            if f1 && f2 {
                sum1_both += v1;
                sum2_both += v2;
                cnt_both += 1;
            }
        }

        if cnt_both < 2 || cnt1_all == 0 || cnt2_all == 0 {
            continue;
        }

        let m1_both = sum1_both / (cnt_both as f64);
        let m2_both = sum2_both / (cnt_both as f64);

        let mut cov = 0.0_f64;
        let mut var2 = 0.0_f64;
        for col in 0..n {
            let v1 = x1[[row, col]];
            let v2 = x2[[row, col]];
            if v1.is_finite() && v2.is_finite() {
                let d1 = v1 - m1_both;
                let d2 = v2 - m2_both;
                cov += d1 * d2;
                var2 += d2 * d2;
            }
        }

        if var2 <= 1e-12 {
            continue;
        }

        let beta = cov / var2;
        let mean1_indep = sum1_all / (cnt1_all as f64);
        let mean2_indep = sum2_all / (cnt2_all as f64);
        let alpha = mean1_indep - beta * mean2_indep;

        for col in 0..n {
            let v1 = x1[[row, col]];
            let v2 = x2[[row, col]];
            if !v1.is_finite() {
                out[[row, col]] = f64::NAN;
            } else if !v2.is_finite() {
                out[[row, col]] = v1;
            } else {
                out[[row, col]] = v1 - (alpha + beta * v2);
            }
        }
    }
    out
}

/// 求解小规模 $K \times K$ 线性方程组 $A \beta = b$（高斯-约旦列主元消元法）。
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
        if max_abs < 1e-18 {
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

/// 8. `riskNeutMulti`: 无截距多变量截面岭回归（Opcode: `ZsimCrossResidual`）。
///
/// 正规方程对角线加 $10^{-8}$ 岭正则项：$\beta = (X^T X + 10^{-8} I)^{-1} X^T y$，
/// 仅在 $y$ 与所有暴露 $X$ 均有限的样本上拟合并输出残差 $r = y - X\beta$。
pub fn pp_risk_neut_multi(y: &Array2<f64>, exposures: &[&Array2<f64>]) -> Array2<f64> {
    let (t, n) = y.dim();
    let k = exposures.len();
    if k == 0 {
        return y.clone();
    }
    for exp in exposures {
        debug_assert_eq!((t, n), exp.dim());
    }

    let mut out = alloc_nan_matrix(t, n);
    let mut valid_cols: Vec<usize> = Vec::with_capacity(n);

    for row in 0..t {
        valid_cols.clear();
        'col_loop: for col in 0..n {
            if !y[[row, col]].is_finite() {
                continue;
            }
            for exp in exposures {
                if !exp[[row, col]].is_finite() {
                    continue 'col_loop;
                }
            }
            valid_cols.push(col);
        }

        if valid_cols.is_empty() {
            continue;
        }

        let mut a_mat = vec![vec![0.0_f64; k]; k];
        let mut b_vec = vec![0.0_f64; k];

        for &col in &valid_cols {
            let y_val = y[[row, col]];
            for i in 0..k {
                let xi = exposures[i][[row, col]];
                b_vec[i] += xi * y_val;
                for j in 0..k {
                    let xj = exposures[j][[row, col]];
                    a_mat[i][j] += xi * xj;
                }
            }
        }

        for (i, a_row) in a_mat.iter_mut().enumerate() {
            a_row[i] += 1e-8;
        }

        if let Some(beta) = solve_linear_system(&mut a_mat, &b_vec) {
            for &col in &valid_cols {
                let mut pred = 0.0_f64;
                for i in 0..k {
                    pred += beta[i] * exposures[i][[row, col]];
                }
                out[[row, col]] = y[[row, col]] - pred;
            }
        }
    }
    out
}

/// 9. `zScore`: 每日截面总体标准差（`ddof=0`）标准化（Opcode: `ZsimCrossZScore`）。
pub fn pp_zscore(x1: &Array2<f64>) -> Array2<f64> {
    let (t, n) = x1.dim();
    let mut out = alloc_nan_matrix(t, n);
    if t == 0 || n == 0 {
        return out;
    }

    for row in 0..t {
        let mut sum = 0.0_f64;
        let mut cnt = 0usize;
        for col in 0..n {
            let v = x1[[row, col]];
            if v.is_finite() {
                sum += v;
                cnt += 1;
            }
        }
        if cnt == 0 {
            continue;
        }
        let mean = sum / (cnt as f64);
        let mut m2 = 0.0_f64;
        for col in 0..n {
            let v = x1[[row, col]];
            if v.is_finite() {
                let d = v - mean;
                m2 += d * d;
            }
        }
        let std = (m2 / (cnt as f64)).sqrt();
        if !std.is_finite() || std == 0.0 {
            continue;
        }
        for col in 0..n {
            let v = x1[[row, col]];
            if v.is_finite() {
                out[[row, col]] = (v - mean) / std;
            }
        }
    }
    out
}

/// 与 Python `round(val, 6)` 一致的银行家舍入（round half to even）。
#[inline]
fn round_6_ties_even(val: f64) -> f64 {
    let scaled = val * 1_000_000.0;
    scaled.round_ties_even() / 1_000_000.0
}

/// 10. `rank`: 每日截面 6 位精度截断稳定排序百分位排名（Opcode: `ZsimPctRank`）。
pub fn pp_rank(x1: &Array2<f64>) -> Array2<f64> {
    let (t, n) = x1.dim();
    let mut out = alloc_nan_matrix(t, n);
    if t == 0 || n == 0 {
        return out;
    }

    let mut keyed: Vec<(f64, usize)> = Vec::with_capacity(n);

    for row in 0..t {
        let mut sum = 0.0_f64;
        let mut cnt = 0usize;
        for col in 0..n {
            let v = x1[[row, col]];
            if v.is_finite() {
                sum += v;
                cnt += 1;
            }
        }
        if cnt == 0 {
            continue;
        }
        let mean = sum / (cnt as f64);
        let mut m2 = 0.0_f64;
        for col in 0..n {
            let v = x1[[row, col]];
            if v.is_finite() {
                let d = v - mean;
                m2 += d * d;
            }
        }
        let std = (m2 / (cnt as f64)).sqrt();
        if !std.is_finite() || std < 1e-6 {
            continue;
        }

        keyed.clear();
        for col in 0..n {
            let v = x1[[row, col]];
            if v.is_finite() {
                keyed.push((round_6_ties_even(v / std), col));
            }
        }
        keyed.sort_by(|a, b| cmp_f64(a.0, b.0).then_with(|| a.1.cmp(&b.1)));
        let n_fin = keyed.len() as f64;
        for (idx, &(_, col)) in keyed.iter().enumerate() {
            out[[row, col]] = ((idx + 1) as f64) / n_fin;
        }
    }
    out
}

/// 11. `decay`: 沿时间轴 $T$ 满窗滚动算术均值（Opcode: `ZsimRollingMean`）。
///
/// 当行下标 `row < window - 1` 时强制返回 `NaN`；否则跳过窗内 `NaN` 计算算术平均。
pub fn pp_decay(x1: &Array2<f64>, window: usize) -> Array2<f64> {
    let (t, n) = x1.dim();
    let mut out = alloc_nan_matrix(t, n);
    if t == 0 || n == 0 || window == 0 || window > t {
        return out;
    }

    for col in 0..n {
        let xc = x1.column(col);
        let mut oc = out.column_mut(col);
        for row in (window - 1)..t {
            let mut sum = 0.0_f64;
            let mut cnt = 0usize;
            for k in (row + 1 - window)..=row {
                let v = xc[k];
                if v.is_finite() {
                    sum += v;
                    cnt += 1;
                }
            }
            if cnt > 0 {
                oc[row] = sum / (cnt as f64);
            }
        }
    }
    out
}

/// 12. `emadecay`: 沿时间轴 $T$ 满窗线性加权移动平均 WMA（Opcode: `ZsimRollingWma`）。
///
/// 权重为 $1 \dots W$（从最旧到最新）。若窗口内某观测为 `NaN`，剔除其对应权重并重归一化剩余权重；
/// 当 `row < window - 1` 时强制返回 `NaN`。
pub fn pp_emadecay(x1: &Array2<f64>, window: usize) -> Array2<f64> {
    let (t, n) = x1.dim();
    let mut out = alloc_nan_matrix(t, n);
    if t == 0 || n == 0 || window == 0 || window > t {
        return out;
    }

    for col in 0..n {
        let xc = x1.column(col);
        let mut oc = out.column_mut(col);
        for row in (window - 1)..t {
            let mut num = 0.0_f64;
            let mut den = 0.0_f64;
            let base = row + 1 - window;
            for i in 1..=window {
                let v = xc[base + i - 1];
                if v.is_finite() {
                    let w = i as f64;
                    num += w * v;
                    den += w;
                }
            }
            if den > 0.0 {
                oc[row] = num / den;
            }
        }
    }
    out
}
