//! 统一 T+1 前向预测标签计算内核 (`labels.rs`)。
//!
//! 消除期货与 A 股两套 T+1 标签语义重复：
//! - `forward_returns`: 通用点对点前向收益计算，支持严格模式 (`strict=true`: 到期不可交易置 NaN)
//!   与顺延模式 (`strict=false`: 到期不可交易顺延至下一可平仓交易日)；
//! - `compound_chained_returns`: 期货换月链式单步收益 (`r_{d->d+1}`) 的多期复合计算，
//!   与 `r_exec_next` 严格对齐 `T+1` 入场窗口。

use ndarray::{Array2, ArrayView2, ShapeBuilder};

#[inline]
fn is_valid_price(p: f64) -> bool {
    p.is_finite() && p > 0.0
}

/// 计算通用点对点 T+1 前向收益率矩阵列表。
///
/// 约定：
/// - `T` 日收盘后生成信号，在 `T+1` 日以 `entry_price[T+1]` 建仓；
/// - 持有 `h` 个交易日，在 `T+1+h` 日（或顺延至下一可平仓日）以 `exit_price` 平仓；
/// - 有效信号行范围为 `t + 1 + h <= T - 1`；
/// - 任意价格为 NaN/±Inf/<=0 或不可交易时返回 NaN。
pub fn forward_returns(
    entry_price: ArrayView2<'_, f64>,
    exit_price: ArrayView2<'_, f64>,
    entry_ok: ArrayView2<'_, bool>,
    exit_ok: ArrayView2<'_, bool>,
    horizons: &[usize],
    strict: bool,
) -> Vec<Array2<f64>> {
    let (t_len, n_cols) = entry_price.dim();
    assert_eq!(
        exit_price.dim(),
        (t_len, n_cols),
        "exit_price 维度必须与 entry_price 一致"
    );
    assert_eq!(
        entry_ok.dim(),
        (t_len, n_cols),
        "entry_ok 维度必须与 entry_price 一致"
    );
    assert_eq!(
        exit_ok.dim(),
        (t_len, n_cols),
        "exit_ok 维度必须与 entry_price 一致"
    );

    // 顺延模式下预先按列倒序扫描构造 next_exit_idx[tau, j]：>= tau 的最早可平仓行索引
    let next_exit_idx: Option<Array2<isize>> = if !strict && t_len > 0 {
        let mut idx_mat = Array2::<isize>::from_elem((t_len, n_cols).f(), -1);
        for j in 0..n_cols {
            let mut nxt: isize = -1;
            for tau in (0..t_len).rev() {
                if exit_ok[[tau, j]] && is_valid_price(exit_price[[tau, j]]) {
                    nxt = tau as isize;
                }
                idx_mat[[tau, j]] = nxt;
            }
        }
        Some(idx_mat)
    } else {
        None
    };

    let mut results = Vec::with_capacity(horizons.len());

    for &h in horizons {
        let mut out = Array2::<f64>::from_elem((t_len, n_cols).f(), f64::NAN);
        if h == 0 || t_len <= 1 + h {
            results.push(out);
            continue;
        }
        let max_rows = t_len - 1 - h; // t in 0..max_rows <=> t + 1 + h <= t_len - 1

        for j in 0..n_cols {
            for t in 0..max_rows {
                let t_in = t + 1;
                if !entry_ok[[t_in, j]] {
                    continue;
                }
                let p_in = entry_price[[t_in, j]];
                if !is_valid_price(p_in) {
                    continue;
                }

                let t_exit_target = t + 1 + h;
                if strict {
                    if !exit_ok[[t_exit_target, j]] {
                        continue;
                    }
                    let p_out = exit_price[[t_exit_target, j]];
                    if is_valid_price(p_out) {
                        out[[t, j]] = p_out / p_in - 1.0;
                    }
                } else if let Some(ref idx_mat) = next_exit_idx {
                    let out_idx = idx_mat[[t_exit_target, j]];
                    if out_idx >= 0 {
                        let p_out = exit_price[[out_idx as usize, j]];
                        if is_valid_price(p_out) {
                            out[[t, j]] = p_out / p_in - 1.0;
                        }
                    }
                }
            }
        }

        results.push(out);
    }

    results
}

/// 基于链式单步执行收益 `step_returns[d, j] = r_{d->d+1}` 计算 `r_exec_next` 与多期复合标签 `y_h`。
///
/// 约定：
/// - `r_exec_next[T, j] = step_returns[T + 1, j]`；
/// - `y_h[T, j] = prod_{k=1..h} (1 + step_returns[T + k, j]) - 1`；
/// - 窗口 `[T + 1, T + h]` 内任意一步缺失 (NaN/±Inf) 则 `y_h[T, j]` 为 NaN。
pub fn compound_chained_returns(
    step_returns: ArrayView2<'_, f64>,
    horizons: &[usize],
) -> (Array2<f64>, Vec<Array2<f64>>) {
    let (t_len, n_cols) = step_returns.dim();

    let mut r_exec_next = Array2::<f64>::from_elem((t_len, n_cols).f(), f64::NAN);
    if t_len >= 2 {
        for j in 0..n_cols {
            for t in 0..(t_len - 1) {
                let r = step_returns[[t + 1, j]];
                if r.is_finite() {
                    r_exec_next[[t, j]] = r;
                }
            }
        }
    }

    let mut out_list = Vec::with_capacity(horizons.len());
    for &h in horizons {
        let mut out = Array2::<f64>::from_elem((t_len, n_cols).f(), f64::NAN);
        if h == 0 || t_len <= h {
            out_list.push(out);
            continue;
        }
        let max_rows = t_len - h; // t in 0..max_rows <=> t + h <= t_len - 1
        for j in 0..n_cols {
            for t in 0..max_rows {
                let mut all_valid = true;
                let mut all_gt_neg_one = true;
                let mut log_sum = 0.0_f64;
                let mut prod = 1.0_f64;
                for k in 1..=h {
                    let r = step_returns[[t + k, j]];
                    if !r.is_finite() {
                        all_valid = false;
                        break;
                    }
                    prod *= 1.0 + r;
                    if r > -1.0 {
                        log_sum += r.ln_1p();
                    } else {
                        all_gt_neg_one = false;
                    }
                }
                if all_valid {
                    out[[t, j]] = if all_gt_neg_one {
                        log_sum.exp_m1()
                    } else {
                        prod - 1.0
                    };
                }
            }
        }
        out_list.push(out);
    }

    (r_exec_next, out_list)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_forward_returns_strict_and_deferral() {
        let price = array![
            [10.0, 10.0, 10.0],
            [11.0, 10.0, 10.0],
            [11.0, 12.0, 9.0],
            [11.0, 12.0, 8.1],
            [11.0, 12.0, 8.5],
        ];
        let entry_ok = array![
            [true, true, true],
            [false, true, true],
            [true, true, true],
            [true, true, true],
            [true, true, true],
        ];
        let exit_ok = array![
            [true, true, true],
            [true, true, true],
            [true, true, false],
            [true, true, false],
            [true, true, true],
        ];

        let res_strict = forward_returns(
            price.view(),
            price.view(),
            entry_ok.view(),
            exit_ok.view(),
            &[1],
            true,
        );
        assert!(res_strict[0][[0, 0]].is_nan());
        assert!((res_strict[0][[0, 1]] - 0.2).abs() < 1e-12);
        assert!(res_strict[0][[0, 2]].is_nan());

        let res_defer = forward_returns(
            price.view(),
            price.view(),
            entry_ok.view(),
            exit_ok.view(),
            &[1],
            false,
        );
        assert!(res_defer[0][[0, 0]].is_nan());
        assert!((res_defer[0][[0, 1]] - 0.2).abs() < 1e-12);
        assert!((res_defer[0][[0, 2]] - (8.5 / 10.0 - 1.0)).abs() < 1e-12);
    }

    #[test]
    fn test_compound_chained_returns_basic() {
        let step = array![[0.01, 0.02], [0.10, -0.05], [0.20, 0.10], [0.05, f64::NAN]];
        let (r1, ys) = compound_chained_returns(step.view(), &[1, 2]);
        assert!((r1[[0, 0]] - 0.10).abs() < 1e-12);
        assert!((ys[0][[0, 0]] - 0.10).abs() < 1e-12);
        assert!((ys[1][[0, 0]] - (1.10 * 1.20 - 1.0)).abs() < 1e-12);
        assert!((ys[1][[0, 1]] - (0.95 * 1.10 - 1.0)).abs() < 1e-12);
        // t=1 跨入 t+2=3 (col 1 为 NaN)
        assert!(ys[1][[1, 1]].is_nan());
        assert!((ys[1][[1, 0]] - (1.20 * 1.05 - 1.0)).abs() < 1e-12);
    }
}
