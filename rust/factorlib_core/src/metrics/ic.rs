//! 截面 Spearman Rank IC 与时序 IC 指标计算 (metrics/ic.rs)。
//!
//! 严格遵循白皮书 8.1 节规范：
//! - $r_{\text{IC}}(t) = \text{Corr}(\text{rank}(\mathbf{f}_t), \text{rank}(\mathbf{y}_t))$；
//! - 采用标准并列平均秩（Average Ties）与 Pearson 秩相关算法；
//! - 计算 IC 均值 $\mu_{\text{IC}}$、样本标准差 $\sigma_{\text{IC}}$、信息比率 $\text{ICIR} = \mu / \sigma$；
//! - 多周期重叠偏误修正 $t$ 统计量：$t_{\text{stat}} = \text{ICIR} \times \sqrt{|\mathcal{D}| / h}$；
//! - IC 符号胜率：$\text{WinRate}_{\text{IC}} = \frac{1}{|\mathcal{D}|}\sum \mathbb{I}(\text{sign}(r(t)) == \text{sign}(\mu))$；
//! - 标的池有效覆盖率：$\text{Coverage}$；
//! - 支持 Rayon 行并发极速计算。

use crate::nan::{cmp_f64, STD_EPS};
use ndarray::Array2;
use rayon::prelude::*;

/// 因子 IC 评价汇总指标结构体。
#[derive(Debug, Clone, PartialEq)]
pub struct IcMetrics {
    /// 截面 IC 序列时序均值
    pub ic_mean: f64,
    /// 截面 IC 序列时序样本标准差 (ddof=1)
    pub ic_std: f64,
    /// 因子信息比率 (ICIR = ic_mean / ic_std)
    pub ic_ir: f64,
    /// 经持有期自由度调整的 t 统计量 (t = ICIR * sqrt(N / h))
    pub ic_t: f64,
    /// IC 符号同向胜率 (0.0 ~ 1.0)
    pub ic_win_rate: f64,
    /// 在池标的有效因子值覆盖率 (0.0 ~ 1.0)
    pub coverage: f64,
    /// 满足最小截面样本数的有效交易日天数
    pub valid_days: usize,
    /// 逐日 IC 时序数组 (无效日填充 NaN)
    pub daily_ic: Vec<f64>,
}

/// 计算一维切片的并列平均排名（1-based 平均秩）。
pub fn compute_fractional_ranks(vals: &[f64], out_ranks: &mut [f64], buf: &mut Vec<(usize, f64)>) {
    let k = vals.len();
    if k == 0 {
        return;
    }
    if k == 1 {
        out_ranks[0] = 1.0;
        return;
    }
    buf.clear();
    for (i, &v) in vals.iter().enumerate() {
        buf.push((i, v));
    }
    buf.sort_unstable_by(|a, b| cmp_f64(a.1, b.1));

    let mut i = 0;
    while i < k {
        let vi = buf[i].1;
        let tol = STD_EPS * 1.0_f64.max(vi.abs());
        let mut j = i + 1;
        while j < k && (buf[j].1 - vi).abs() <= tol {
            j += 1;
        }
        // [i, j) 范围内的元素具有相同取值，赋予 1-based 平均排名：0.5 * ((i+1) + j)
        let avg_rank = 0.5 * ((i + 1 + j) as f64);
        for idx in i..j {
            out_ranks[buf[idx].0] = avg_rank;
        }
        i = j;
    }
}

/// 计算两个一维等长向量之间的 Pearson 相关系数。
#[inline]
pub fn pearson_correlation(x: &[f64], y: &[f64]) -> f64 {
    let n = x.len();
    if n < 2 || n != y.len() {
        return f64::NAN;
    }
    let nf = n as f64;
    let mut sum_x = 0.0_f64;
    let mut sum_y = 0.0_f64;
    for i in 0..n {
        sum_x += x[i];
        sum_y += y[i];
    }
    let mean_x = sum_x / nf;
    let mean_y = sum_y / nf;

    let mut cov = 0.0_f64;
    let mut var_x = 0.0_f64;
    let mut var_y = 0.0_f64;

    for i in 0..n {
        let dx = x[i] - mean_x;
        let dy = y[i] - mean_y;
        cov += dx * dy;
        var_x += dx * dx;
        var_y += dy * dy;
    }

    if var_x <= 1e-14 || var_y <= 1e-14 {
        f64::NAN
    } else {
        let r = cov / (var_x * var_y).sqrt();
        r.clamp(-1.0, 1.0)
    }
}

/// 计算单个交易日截面上的相关系数 (Spearman Rank IC 或 Pearson IC)。
pub fn calc_daily_cross_sectional_ic(
    factor_row: &[f64],
    label_row: &[f64],
    mask_row: &[bool],
    min_names: usize,
    rank: bool,
) -> f64 {
    let n = factor_row.len();
    let mut x_vals = Vec::with_capacity(n);
    let mut y_vals = Vec::with_capacity(n);

    for i in 0..n {
        if mask_row[i] {
            let f = factor_row[i];
            let y = label_row[i];
            if f.is_finite() && y.is_finite() {
                x_vals.push(f);
                y_vals.push(y);
            }
        }
    }

    let k = x_vals.len();
    if k < min_names || k < 2 {
        return f64::NAN;
    }

    if rank {
        let mut x_ranks = vec![0.0_f64; k];
        let mut y_ranks = vec![0.0_f64; k];
        let mut buf = Vec::with_capacity(k);
        compute_fractional_ranks(&x_vals, &mut x_ranks, &mut buf);
        compute_fractional_ranks(&y_vals, &mut y_ranks, &mut buf);
        pearson_correlation(&x_ranks, &y_ranks)
    } else {
        pearson_correlation(&x_vals, &y_vals)
    }
}

/// 并行计算整个面板在所有交易日上的截面 IC 时序。
pub fn calc_ic_series(
    factor: &Array2<f64>,
    label: &Array2<f64>,
    mask: &Array2<bool>,
    min_names: usize,
    rank: bool,
) -> Vec<f64> {
    let (t, n) = factor.dim();
    assert_eq!(label.dim(), (t, n), "factor 与 label 矩阵形状必须一致");
    assert_eq!(mask.dim(), (t, n), "factor 与 mask 矩阵形状必须一致");

    // 每一天为一截面，按交易日并行
    (0..t)
        .into_par_iter()
        .map(|row| {
            let mut f_row = Vec::with_capacity(n);
            let mut l_row = Vec::with_capacity(n);
            let mut m_row = Vec::with_capacity(n);
            for col in 0..n {
                f_row.push(factor[[row, col]]);
                l_row.push(label[[row, col]]);
                m_row.push(mask[[row, col]]);
            }
            calc_daily_cross_sectional_ic(&f_row, &l_row, &m_row, min_names, rank)
        })
        .collect()
}

/// 聚合逐日 IC 时序并生成完整的多维评估指标。
pub fn calc_ic_metrics(
    daily_ic: &[f64],
    horizon: usize,
    total_days: usize,
    valid_factor_count: usize,
    total_universe_count: usize,
) -> IcMetrics {
    let valid_ic: Vec<f64> = daily_ic.iter().copied().filter(|v| v.is_finite()).collect();
    let m = valid_ic.len();

    let coverage = if total_universe_count > 0 {
        (valid_factor_count as f64) / (total_universe_count as f64)
    } else if total_days > 0 {
        (m as f64) / (total_days as f64)
    } else {
        0.0
    };

    if m == 0 {
        return IcMetrics {
            ic_mean: 0.0,
            ic_std: 0.0,
            ic_ir: 0.0,
            ic_t: 0.0,
            ic_win_rate: 0.0,
            coverage,
            valid_days: 0,
            daily_ic: daily_ic.to_vec(),
        };
    }

    let mf = m as f64;
    let sum: f64 = valid_ic.iter().sum();
    let mean = sum / mf;

    let std = if m > 1 {
        let var_sum: f64 = valid_ic.iter().map(|v| (v - mean).powi(2)).sum();
        (var_sum / ((m - 1) as f64)).sqrt()
    } else {
        0.0
    };

    let ir = if std > 1e-12 { mean / std } else { 0.0 };

    let h = horizon.max(1) as f64;
    let t_stat = ir * (mf / h).sqrt();

    let wins: usize = if mean > 0.0 {
        valid_ic.iter().filter(|&&v| v > 0.0).count()
    } else if mean < 0.0 {
        valid_ic.iter().filter(|&&v| v < 0.0).count()
    } else {
        valid_ic.iter().filter(|&&v| v == 0.0).count()
    };
    let win_rate = (wins as f64) / mf;

    IcMetrics {
        ic_mean: mean,
        ic_std: std,
        ic_ir: ir,
        ic_t: t_stat,
        ic_win_rate: win_rate,
        coverage,
        valid_days: m,
        daily_ic: daily_ic.to_vec(),
    }
}
