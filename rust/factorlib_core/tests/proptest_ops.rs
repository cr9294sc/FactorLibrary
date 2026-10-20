//! 基于 `proptest` 的 68 算子数值对账、DSL 类型安全与代数规范化性质测试。
//!
//! 严格对应白皮书 P2 验收标准与 16.2 节要求：
//! - 随机生成包含正负无穷、大数、并列值、高密度 `NaN` 的 $T \times N$ 矩阵；
//! - 对全部时序滚动增量算法（双栈 Welford、双端单调队列、二分有序窗）与朴素窗口切片重算逐元素比对，
//!   断言误差 $< 10^{-9}$；
//! - 校验 7.2 节表格中全部 10 项非法签名阻断示例与 `y_*` 未来函数标签阻断；
//! - 校验 7.3 节交换律、幂等性、单调不变性与退化树阻断。

use factorlib_core::expr::cache::SubExprCache;
use factorlib_core::expr::canon::canonicalize;
use factorlib_core::expr::eval::{eval_batch_str, eval_expr_str};
use factorlib_core::expr::parser::parse_expr;
use factorlib_core::expr::typecheck::typecheck_factor_expr;
use factorlib_core::nan::{clean_f64, STD_EPS, VAR_EPS};
use factorlib_core::ops::{cs, elem, list_all_operators, ts};
use factorlib_core::panel::{alloc_nan_matrix, into_fortran_f64, Panel};
use ndarray::{Array2, ShapeBuilder};
use proptest::prelude::*;
use std::collections::{HashMap, HashSet};
use std::sync::Arc;

fn assert_matrices_close(actual: &Array2<f64>, expected: &Array2<f64>, tol: f64, ctx: &str) {
    assert_eq!(actual.dim(), expected.dim(), "{ctx}: 矩阵维度不一致");
    let (t, n) = actual.dim();
    for r in 0..t {
        for c in 0..n {
            let a = actual[[r, c]];
            let e = expected[[r, c]];
            if a.is_nan() && e.is_nan() {
                continue;
            }
            assert!(
                a.is_finite() && e.is_finite(),
                "{ctx} at ({r}, {c}): NaN 不一致 actual={a}, expected={e}"
            );
            let diff = (a - e).abs();
            let scale = 1.0_f64.max(a.abs()).max(e.abs());
            assert!(
                diff <= tol * scale,
                "{ctx} at ({r}, {c}): 误差超标 actual={a}, expected={e}, diff={diff}, scale={scale}"
            );
        }
    }
}

/// 生成包含有限值、并列值、高密度 NaN 及 ±Inf 的随机测试元素策略。
fn noisy_f64_strategy() -> impl Strategy<Value = f64> {
    prop_oneof![
        6 => -1e5_f64..1e5_f64,
        2 => prop::sample::select(vec![-10.0, -1.0, 0.0, 0.5, 1.0, 1.0, 2.0, 10.0]),
        2 => Just(f64::NAN),
        1 => prop::sample::select(vec![f64::INFINITY, f64::NEG_INFINITY]),
    ]
}

fn matrix_strategy(t: usize, n: usize) -> impl Strategy<Value = Array2<f64>> {
    proptest::collection::vec(noisy_f64_strategy(), t * n).prop_map(move |v| {
        let mut arr = Array2::<f64>::zeros((t, n).f());
        for c in 0..n {
            for r in 0..t {
                arr[[r, c]] = v[c * t + r];
            }
        }
        arr
    })
}

/// 朴素切片参考实现：用于验证双栈 Welford 与单调队列增量算法。
fn ref_ts_mean_std_var_sum_zscore(
    x: &Array2<f64>,
    d: usize,
) -> (
    Array2<f64>,
    Array2<f64>,
    Array2<f64>,
    Array2<f64>,
    Array2<f64>,
) {
    let (t, n) = x.dim();
    let mut sum_out = alloc_nan_matrix(t, n);
    let mut mean_out = alloc_nan_matrix(t, n);
    let mut var_out = alloc_nan_matrix(t, n);
    let mut std_out = alloc_nan_matrix(t, n);
    let mut z_out = alloc_nan_matrix(t, n);

    if d == 0 || d > t {
        return (sum_out, mean_out, var_out, std_out, z_out);
    }

    for col in 0..n {
        for row in (d - 1)..t {
            let win_start = row + 1 - d;
            let mut valid = Vec::with_capacity(d);
            for s in win_start..=row {
                let v = x[[s, col]];
                if v.is_finite() {
                    valid.push(v);
                }
            }
            let k = valid.len();
            if k >= 1 {
                let sum: f64 = valid.iter().sum();
                let mean = sum / (k as f64);
                sum_out[[row, col]] = clean_f64(sum);
                mean_out[[row, col]] = clean_f64(mean);
                if k >= 2 && d >= 2 {
                    let m2: f64 = valid.iter().map(|&v| (v - mean) * (v - mean)).sum();
                    let scale = 1.0_f64.max(mean.abs());
                    let m2_c = if m2 <= VAR_EPS * (k as f64) * scale * scale {
                        0.0
                    } else {
                        m2
                    };
                    let var = clean_f64(m2_c / ((k - 1) as f64));
                    var_out[[row, col]] = var;
                    if var.is_finite() {
                        let std = var.sqrt();
                        std_out[[row, col]] = std;
                        let cur = x[[row, col]];
                        if cur.is_finite() && std > STD_EPS * scale {
                            z_out[[row, col]] = clean_f64((cur - mean) / std);
                        }
                    }
                }
            }
        }
    }
    (sum_out, mean_out, var_out, std_out, z_out)
}

fn ref_ts_extrema_and_rank(
    x: &Array2<f64>,
    d: usize,
) -> (
    Array2<f64>,
    Array2<f64>,
    Array2<f64>,
    Array2<f64>,
    Array2<f64>,
) {
    let (t, n) = x.dim();
    let mut min_out = alloc_nan_matrix(t, n);
    let mut max_out = alloc_nan_matrix(t, n);
    let mut argmin_out = alloc_nan_matrix(t, n);
    let mut argmax_out = alloc_nan_matrix(t, n);
    let mut rank_out = alloc_nan_matrix(t, n);

    if d == 0 || d > t {
        return (min_out, max_out, argmin_out, argmax_out, rank_out);
    }

    for col in 0..n {
        for row in (d - 1)..t {
            let win_start = row + 1 - d;
            let mut min_v = f64::INFINITY;
            let mut max_v = f64::NEG_INFINITY;
            let mut min_pos = None;
            let mut max_pos = None;
            let mut valid = Vec::with_capacity(d);

            for (offset, s) in (win_start..=row).enumerate() {
                let v = x[[s, col]];
                if v.is_finite() {
                    valid.push(v);
                    if v < min_v {
                        min_v = v;
                        min_pos = Some(offset);
                    }
                    if v > max_v {
                        max_v = v;
                        max_pos = Some(offset);
                    }
                }
            }
            if let Some(p) = min_pos {
                min_out[[row, col]] = min_v;
                argmin_out[[row, col]] = p as f64;
            }
            if let Some(p) = max_pos {
                max_out[[row, col]] = max_v;
                argmax_out[[row, col]] = p as f64;
            }
            let cur = x[[row, col]];
            if cur.is_finite() {
                let k = valid.len();
                if k == 1 {
                    rank_out[[row, col]] = 0.5;
                } else if k >= 2 {
                    let c_less = valid.iter().filter(|&&v| v < cur).count() as f64;
                    let c_eq = valid.iter().filter(|&&v| v == cur).count() as f64;
                    rank_out[[row, col]] = (c_less + 0.5 * (c_eq - 1.0)) / ((k - 1) as f64);
                }
            }
        }
    }
    (min_out, max_out, argmin_out, argmax_out, rank_out)
}

fn ref_ts_bivariate(
    x: &Array2<f64>,
    y: &Array2<f64>,
    d: usize,
) -> (Array2<f64>, Array2<f64>, Array2<f64>, Array2<f64>) {
    let (t, n) = x.dim();
    let mut cov_out = alloc_nan_matrix(t, n);
    let mut corr_out = alloc_nan_matrix(t, n);
    let mut beta_out = alloc_nan_matrix(t, n);
    let mut resid_out = alloc_nan_matrix(t, n);

    if d < 2 || d > t {
        return (cov_out, corr_out, beta_out, resid_out);
    }

    for col in 0..n {
        for row in (d - 1)..t {
            let win_start = row + 1 - d;
            let mut pairs = Vec::with_capacity(d);
            for s in win_start..=row {
                let vx = x[[s, col]];
                let vy = y[[s, col]];
                if vx.is_finite() && vy.is_finite() {
                    pairs.push((vx, vy));
                }
            }
            let k = pairs.len();
            if k >= 2 {
                let kf = k as f64;
                let mx: f64 = pairs.iter().map(|p| p.0).sum::<f64>() / kf;
                let my: f64 = pairs.iter().map(|p| p.1).sum::<f64>() / kf;
                let mut m2_x = 0.0;
                let mut m2_y = 0.0;
                let mut c_xy = 0.0;
                for &(vx, vy) in &pairs {
                    let dx = vx - mx;
                    let dy = vy - my;
                    m2_x += dx * dx;
                    m2_y += dy * dy;
                    c_xy += dx * dy;
                }
                if c_xy.is_finite() {
                    cov_out[[row, col]] = clean_f64(c_xy / ((k - 1) as f64));
                }
                let sx = 1.0_f64.max(mx.abs());
                let sy = 1.0_f64.max(my.abs());
                if m2_y.is_finite() && c_xy.is_finite() && m2_y > VAR_EPS * kf * sy * sy {
                    let beta = clean_f64(c_xy / m2_y);
                    beta_out[[row, col]] = beta;
                    let cur_x = x[[row, col]];
                    let cur_y = y[[row, col]];
                    if cur_x.is_finite() && cur_y.is_finite() {
                        resid_out[[row, col]] = clean_f64((cur_x - mx) - (c_xy / m2_y) * (cur_y - my));
                    }
                    if m2_x.is_finite() && m2_x > VAR_EPS * kf * sx * sx {
                        let denom = (m2_x * m2_y).sqrt();
                        if denom > 0.0 && denom.is_finite() {
                            corr_out[[row, col]] = clean_f64((c_xy / denom).clamp(-1.0, 1.0));
                        }
                    }
                }
            }
        }
    }
    (cov_out, corr_out, beta_out, resid_out)
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(64))]

    #[test]
    fn prop_ts_welford_and_monotonic_queue(
        x in matrix_strategy(35, 6),
        d in 1usize..15usize,
    ) {
        let (exp_sum, exp_mean, exp_var, exp_std, exp_z) = ref_ts_mean_std_var_sum_zscore(&x, d);
        assert_matrices_close(&ts::op_ts_sum(&x, d), &exp_sum, 1e-9, "ts_sum");
        assert_matrices_close(&ts::op_ts_mean(&x, d), &exp_mean, 1e-9, "ts_mean");
        assert_matrices_close(&ts::op_ts_var(&x, d), &exp_var, 1e-9, "ts_var");
        assert_matrices_close(&ts::op_ts_std(&x, d), &exp_std, 1e-9, "ts_std");
        assert_matrices_close(&ts::op_ts_zscore(&x, d), &exp_z, 1e-9, "ts_zscore");

        let (exp_min, exp_max, exp_argmin, exp_argmax, exp_rank) = ref_ts_extrema_and_rank(&x, d);
        assert_matrices_close(&ts::op_ts_min(&x, d), &exp_min, 1e-12, "ts_min");
        assert_matrices_close(&ts::op_ts_max(&x, d), &exp_max, 1e-12, "ts_max");
        assert_matrices_close(&ts::op_ts_argmin(&x, d), &exp_argmin, 1e-12, "ts_argmin");
        assert_matrices_close(&ts::op_ts_argmax(&x, d), &exp_argmax, 1e-12, "ts_argmax");
        assert_matrices_close(&ts::op_ts_rank(&x, d), &exp_rank, 1e-12, "ts_rank");
    }

    #[test]
    fn prop_ts_bivariate_welford(
        x in matrix_strategy(30, 5),
        y in matrix_strategy(30, 5),
        d in 2usize..12usize,
    ) {
        let (exp_cov, exp_corr, exp_beta, exp_resid) = ref_ts_bivariate(&x, &y, d);
        assert_matrices_close(&ts::op_ts_cov(&x, &y, d), &exp_cov, 1e-9, "ts_cov");
        assert_matrices_close(&ts::op_ts_corr(&x, &y, d), &exp_corr, 1e-9, "ts_corr");
        assert_matrices_close(&ts::op_ts_beta(&x, &y, d), &exp_beta, 1e-9, "ts_beta");
        assert_matrices_close(&ts::op_ts_resid(&x, &y, d), &exp_resid, 1e-9, "ts_resid");
    }

    #[test]
    fn prop_elem_and_cs_identities(
        x in matrix_strategy(20, 8),
        y in matrix_strategy(20, 8),
    ) {
        // add 交换律
        assert_matrices_close(&elem::op_add(&x, &y), &elem::op_add(&y, &x), 1e-12, "add commutative");
        // mul 交换律
        assert_matrices_close(&elem::op_mul(&x, &y), &elem::op_mul(&y, &x), 1e-12, "mul commutative");
        // max2 & min2 交换律
        assert_matrices_close(&elem::op_max2(&x, &y), &elem::op_max2(&y, &x), 1e-12, "max2 commutative");
        assert_matrices_close(&elem::op_min2(&x, &y), &elem::op_min2(&y, &x), 1e-12, "min2 commutative");
        // neg(neg(x)) == clean(x)
        let clean_x = into_fortran_f64(&x);
        assert_matrices_close(&elem::op_neg(&elem::op_neg(&x)), &clean_x, 1e-12, "neg(neg(x))");
        // abs(neg(x)) == abs(x)
        assert_matrices_close(&elem::op_abs(&elem::op_neg(&x)), &elem::op_abs(&x), 1e-12, "abs(neg(x))");
        // cs_rank 幂等性: cs_rank(cs_rank(x)) == cs_rank(x)
        let r1 = cs::op_cs_rank(&x);
        let r2 = cs::op_cs_rank(&r1);
        assert_matrices_close(&r1, &r2, 1e-12, "cs_rank idempotency");
        // cs_rank 保序性: cs_rank(slog(x)) == cs_rank(x)
        let r_slog = cs::op_cs_rank(&elem::op_slog(&x));
        assert_matrices_close(&r1, &r_slog, 1e-12, "cs_rank(slog(x))");
        // cs_demean 幂等性
        let d1 = cs::op_cs_demean(&x);
        let d2 = cs::op_cs_demean(&d1);
        assert_matrices_close(&d1, &d2, 1e-9, "cs_demean idempotency");
    }
}

#[test]
fn test_all_68_operators_registered() {
    let ops = list_all_operators();
    assert_eq!(ops.len(), 68);
}

#[test]
fn test_typechecker_blocks_all_whitepaper_illegal_examples() {
    let fields: HashSet<String> = ["open", "high", "low", "close", "volume", "sector_id"]
        .into_iter()
        .map(String::from)
        .collect();
    let labels: HashSet<String> = ["y_1", "y_5", "custom_label"]
        .into_iter()
        .map(String::from)
        .collect();

    // 白皮书 7.2 节表格中的全部 10 个非法用法示例 + 未来函数标签访问测试
    let illegal_cases = [
        "abs(20)",
        "add(close, 10)",
        "spow(close, open)",
        "where(close, open, high)",
        "delay(close, open)",
        "ts_mean(20, close)",
        "ts_rank(close, -5)",
        "ts_corr(close, 20, volume)",
        "cs_rank(20)",
        "cs_neutralize(close, 10)",
        "gt(close, open)", // 根节点为 BoolCond 而非 Series
        "y_5",
        "ts_mean(y_1, 5)",
        "add(close, custom_label)",
        "unknown_field_xyz",
    ];

    for expr in illegal_cases {
        let res = parse_expr(expr)
            .map_err(|e| e.to_string())
            .and_then(|ast| {
                typecheck_factor_expr(&ast, Some(&fields), Some(&labels))
                    .map_err(|e| e.to_string())
            });
        assert!(res.is_err(), "期望非法表达式 '{expr}' 被阻断，但通过了检查");
    }
}

#[test]
fn test_canonicalization_and_degenerate_pruning() {
    // 1. 交换律归一化
    let c1 = canonicalize(&parse_expr("add(volume, close)").unwrap());
    let c2 = canonicalize(&parse_expr("add(close, volume)").unwrap());
    assert_eq!(c1.canon_expr, c2.canon_expr);
    assert_eq!(c1.canon_hash, c2.canon_hash);

    // ts_corr 前两个参数交换律
    let tc1 = canonicalize(&parse_expr("ts_corr(volume, close, 20)").unwrap());
    let tc2 = canonicalize(&parse_expr("ts_corr(close, volume, 20)").unwrap());
    assert_eq!(tc1.canon_expr, tc2.canon_expr);
    assert_eq!(tc1.canon_hash, tc2.canon_hash);

    // 2. 幂等性与代数恒等化简
    let id1 = canonicalize(&parse_expr("cs_rank(cs_rank(close))").unwrap());
    assert_eq!(id1.canon_expr, "cs_rank(close)");

    let id2 = canonicalize(&parse_expr("abs(neg(abs(close)))").unwrap());
    assert_eq!(id2.canon_expr, "abs(close)");

    let id3 = canonicalize(&parse_expr("neg(neg(close))").unwrap());
    assert_eq!(id3.canon_expr, "close");

    let id4 = canonicalize(&parse_expr("delay(delay(close, 5), 10)").unwrap());
    assert_eq!(id4.canon_expr, "delay(close, 15)");

    // 3. 单调不变性压缩
    let m1 = canonicalize(&parse_expr("slog(cs_rank(close))").unwrap());
    assert_eq!(m1.canon_expr, "cs_rank(close)");

    let m2 = canonicalize(&parse_expr("cs_rank(ssqrt(close))").unwrap());
    assert_eq!(m2.canon_expr, "cs_rank(close)");

    // 4. 退化结构阻断
    let d1 = canonicalize(&parse_expr("sub(close, close)").unwrap());
    assert!(d1.is_degenerate);

    let d2 = canonicalize(&parse_expr("div(add(open, close), add(close, open))").unwrap());
    assert!(d2.is_degenerate);

    let d3 = canonicalize(&parse_expr("ts_corr(close, close, 20)").unwrap());
    assert!(d3.is_degenerate);
}

#[test]
fn test_dag_eval_batch_and_lru_cache() {
    let t = 50;
    let n = 8;
    let dates: Vec<String> = (0..t).map(|i| format!("2024-01-{i:02}")).collect();
    let symbols: Vec<String> = (0..n).map(|j| format!("S{j}")).collect();
    let mask = Array2::from_elem((t, n).f(), true);

    let mut close = Array2::<f64>::zeros((t, n).f());
    let mut open = Array2::<f64>::zeros((t, n).f());
    let mut grp = Array2::<f64>::zeros((t, n).f());
    for c in 0..n {
        for r in 0..t {
            close[[r, c]] = 100.0 + (r as f64) * 0.5 + (c as f64) * 3.0;
            open[[r, c]] = 99.5 + (r as f64) * 0.45 + (c as f64) * 3.0;
            grp[[r, c]] = (c % 2) as f64;
        }
    }

    let mut fields = HashMap::new();
    fields.insert("close".to_string(), Arc::new(close));
    fields.insert("open".to_string(), Arc::new(open));
    fields.insert("grp".to_string(), Arc::new(grp));

    let panel = Panel::new(dates, symbols, fields, mask, HashMap::new()).unwrap();
    let cache = SubExprCache::from_megabytes(16);

    let exprs = vec![
        "cs_rank(ts_mean(close, 10))".to_string(),
        "cs_zscore(ts_mean(close, 10))".to_string(),
        "add(ts_mean(close, 10), ts_std(open, 10))".to_string(),
        "group_demean(ts_mean(close, 10), grp)".to_string(),
        "where(gt(close, open), ts_mean(close, 10), ts_std(open, 10))".to_string(),
    ];

    let batch_res = eval_batch_str(&exprs, &panel, Some(&cache), false);
    assert_eq!(batch_res.len(), 5);
    for (idx, r) in batch_res.iter().enumerate() {
        let mat = r.as_ref().unwrap();
        let single = eval_expr_str(&exprs[idx], &panel, None, false).unwrap();
        assert_matrices_close(mat, &single, 1e-12, &exprs[idx]);
    }

    let stats = cache.stats();
    assert!(stats.entries >= 5);
    assert!(stats.hits >= 4);
}
