//! 行业/板块分组横截面算子实现 (ops/group.rs)。
//!
//! 支持按板块/行业整数编码矩阵 `group` 在每个交易日横截面内分组计算：
//! - `group_demean(x, group)`：板块内去均值
//! - `group_rank(x, group)`：板块内百分比排名
//! - `group_zscore(x, group)`：板块内 Z-Score 标准化

use crate::nan::{clean_f64, cmp_f64, STD_EPS};
use crate::panel::alloc_nan_matrix;
use ahash::AHashMap;
use ndarray::Array2;

pub fn op_group_demean(x: &Array2<f64>, group: &Array2<f64>) -> Array2<f64> {
    let (t, n) = x.dim();
    debug_assert_eq!((t, n), group.dim());
    let mut out = alloc_nan_matrix(t, n);
    let mut stats: AHashMap<i64, (f64, usize)> = AHashMap::with_capacity(16);

    for row in 0..t {
        stats.clear();
        for col in 0..n {
            let vx = x[[row, col]];
            let vg = group[[row, col]];
            if vx.is_finite() && vg.is_finite() {
                let gid = vg.round() as i64;
                let entry = stats.entry(gid).or_insert((0.0, 0));
                entry.0 += vx;
                entry.1 += 1;
            }
        }
        for col in 0..n {
            let vx = x[[row, col]];
            let vg = group[[row, col]];
            if vx.is_finite() && vg.is_finite() {
                let gid = vg.round() as i64;
                if let Some(&(sum, cnt)) = stats.get(&gid) {
                    if cnt >= 1 {
                        let mean = sum / (cnt as f64);
                        out[[row, col]] = clean_f64(vx - mean);
                    }
                }
            }
        }
    }
    out
}

pub fn op_group_rank(x: &Array2<f64>, group: &Array2<f64>) -> Array2<f64> {
    let (t, n) = x.dim();
    debug_assert_eq!((t, n), group.dim());
    let mut out = alloc_nan_matrix(t, n);
    let mut buckets: AHashMap<i64, Vec<(usize, f64)>> = AHashMap::with_capacity(16);

    for row in 0..t {
        for v in buckets.values_mut() {
            v.clear();
        }
        for col in 0..n {
            let vx = x[[row, col]];
            let vg = group[[row, col]];
            if vx.is_finite() && vg.is_finite() {
                let gid = vg.round() as i64;
                buckets.entry(gid).or_default().push((col, vx));
            }
        }
        for members in buckets.values_mut() {
            let k = members.len();
            if k == 0 {
                continue;
            }
            if k == 1 {
                out[[row, members[0].0]] = 0.5;
                continue;
            }
            members.sort_unstable_by(|a, b| cmp_f64(a.1, b.1));
            let denom = (k - 1) as f64;
            let mut i = 0;
            while i < k {
                let mut j = i + 1;
                let base_v = members[i].1;
                let tol = STD_EPS * 1.0_f64.max(base_v.abs());
                while j < k && (members[j].1 - base_v).abs() <= tol {
                    j += 1;
                }
                let avg_rank = 0.5 * ((i + j - 1) as f64) / denom;
                for idx in i..j {
                    out[[row, members[idx].0]] = avg_rank;
                }
                i = j;
            }
        }
    }
    out
}

pub fn op_group_zscore(x: &Array2<f64>, group: &Array2<f64>) -> Array2<f64> {
    let (t, n) = x.dim();
    debug_assert_eq!((t, n), group.dim());
    let mut out = alloc_nan_matrix(t, n);
    let mut buckets: AHashMap<i64, Vec<(usize, f64)>> = AHashMap::with_capacity(16);

    for row in 0..t {
        for v in buckets.values_mut() {
            v.clear();
        }
        for col in 0..n {
            let vx = x[[row, col]];
            let vg = group[[row, col]];
            if vx.is_finite() && vg.is_finite() {
                let gid = vg.round() as i64;
                buckets.entry(gid).or_default().push((col, vx));
            }
        }
        for members in buckets.values() {
            let k = members.len();
            if k < 2 {
                continue;
            }
            let sum: f64 = members.iter().map(|m| m.1).sum();
            let mean = sum / (k as f64);
            if !mean.is_finite() {
                continue;
            }
            let m2: f64 = members
                .iter()
                .map(|m| {
                    let d = m.1 - mean;
                    d * d
                })
                .sum();
            let std = (m2 / ((k - 1) as f64)).sqrt();
            let scale = 1.0_f64.max(mean.abs());
            if std.is_finite() && std > STD_EPS * scale {
                for &(col, val) in members {
                    out[[row, col]] = clean_f64((val - mean) / std);
                }
            }
        }
    }
    out
}
