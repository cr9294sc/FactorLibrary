//! 快速连续权重回测引擎 (`backtest/fast.rs`)。
//!
//! 用于单因子或粗筛阶段的高性能连续权重回测，扣除基础比例手续费与滑点。

use ndarray::Array2;

#[derive(Debug, Clone)]
pub struct FastBacktestConfig {
    pub fee_rate: f64,
    pub slippage_bps: f64,
    pub init_cash: f64,
}

impl Default for FastBacktestConfig {
    fn default() -> Self {
        Self {
            fee_rate: 0.0001,
            slippage_bps: 1.0,
            init_cash: 10_000_000.0,
        }
    }
}

#[derive(Debug, Clone)]
pub struct FastBacktestResult {
    pub equity: Vec<f64>,
    pub daily_returns: Vec<f64>,
    pub turnover: Vec<f64>,
    pub fees: Vec<f64>,
    pub slippage: Vec<f64>,
    pub total_return: f64,
    pub annual_return: f64,
    pub annual_volatility: f64,
    pub sharpe: f64,
    pub sortino: f64,
    pub calmar: f64,
    pub max_drawdown: f64,
    pub max_drawdown_days: usize,
    pub win_rate: f64,
    pub daily_turnover_mean: f64,
    pub annual_turnover: f64,
}

pub fn run_fast_backtest(
    weights: &Array2<f64>,
    returns: &Array2<f64>,
    cfg: &FastBacktestConfig,
) -> FastBacktestResult {
    let (t, n) = weights.dim();
    assert_eq!(
        returns.dim(),
        (t, n),
        "weights 与 returns 矩阵维度必须完全一致"
    );

    if t == 0 {
        return FastBacktestResult {
            equity: vec![],
            daily_returns: vec![],
            turnover: vec![],
            fees: vec![],
            slippage: vec![],
            total_return: 0.0,
            annual_return: 0.0,
            annual_volatility: 0.0,
            sharpe: 0.0,
            sortino: 0.0,
            calmar: 0.0,
            max_drawdown: 0.0,
            max_drawdown_days: 0,
            win_rate: 0.0,
            daily_turnover_mean: 0.0,
            annual_turnover: 0.0,
        };
    }

    let mut equity = Vec::with_capacity(t);
    let mut daily_returns = Vec::with_capacity(t);
    let mut turnover = Vec::with_capacity(t);
    let mut fees = Vec::with_capacity(t);
    let mut slippage = Vec::with_capacity(t);

    equity.push(cfg.init_cash);
    daily_returns.push(0.0);
    turnover.push(0.0);
    fees.push(0.0);
    slippage.push(0.0);

    let slip_rate = cfg.slippage_bps * 1e-4;
    let mut first_active_day: Option<usize> = None;

    for d in 1..t {
        let prev_e = equity[d - 1];
        if prev_e <= 0.0 {
            // P0-06: 净值跌破零，触发穿仓熔断，冻结资产并清空后续交易与换手
            equity.push(0.0);
            daily_returns.push(0.0);
            turnover.push(0.0);
            fees.push(0.0);
            slippage.push(0.0);
            continue;
        }

        // d 日执行的仓位是 d-1 日收盘后确定的权重
        let w_curr = weights.row(d - 1);
        let w_prev = if d >= 2 {
            weights.row(d - 2)
        } else {
            ndarray::ArrayView1::from(&[])
        };

        // 计算调仓换手率
        let mut d_turnover = 0.0;
        for i in 0..n {
            let wc = if w_curr.is_empty() { 0.0 } else { w_curr[i] };
            let wp = if w_prev.is_empty() { 0.0 } else { w_prev[i] };
            let wc_val = if wc.is_finite() { wc } else { 0.0 };
            let wp_val = if wp.is_finite() { wp } else { 0.0 };
            d_turnover += (wc_val - wp_val).abs();
        }

        let d_fee = d_turnover * cfg.fee_rate;
        let d_slip = d_turnover * slip_rate;

        // 计算组合毛收益率
        let mut gross_ret = 0.0;
        let ret_row = returns.row(d);
        for i in 0..n {
            let w = if w_curr.is_empty() { 0.0 } else { w_curr[i] };
            let r = ret_row[i];
            if w.is_finite() && r.is_finite() {
                gross_ret += w * r;
            }
        }

        let net_ret = gross_ret - (d_fee + d_slip);
        let curr_e = (prev_e * (1.0 + net_ret)).max(0.0);

        let has_active_weight =
            !w_curr.is_empty() && w_curr.iter().any(|&w| w.is_finite() && w != 0.0);
        if first_active_day.is_none() && (d_turnover > 0.0 || has_active_weight) {
            first_active_day = Some(d);
        }

        equity.push(curr_e);
        daily_returns.push(net_ret);
        turnover.push(d_turnover);
        fees.push(d_fee * prev_e);
        slippage.push(d_slip * prev_e);
    }

    // 统计指标计算 (P2-01: 剔除前导预热空仓日)
    let total_ret = if cfg.init_cash > 0.0 {
        equity.last().copied().unwrap_or(cfg.init_cash) / cfg.init_cash - 1.0
    } else {
        0.0
    };

    let stat_start = first_active_day.unwrap_or(t);
    let n_days = t.saturating_sub(stat_start);
    let annual_ret = if n_days > 0 {
        // P0-06: 净值穿仓 1.0 + total_ret <= 0 时封底截断为 -1.0，防止负底数幂运算触发 IEEE-754 NaN
        if 1.0 + total_ret <= 0.0 {
            -1.0
        } else {
            (1.0 + total_ret).powf(252.0 / n_days as f64) - 1.0
        }
    } else {
        0.0
    };

    // 波动率与夏普
    let mut sum_r = 0.0;
    let mut sum_sq_r = 0.0;
    let mut win_count = 0;
    let mut valid_days = 0;
    let mut downside_sq = 0.0;

    for &r in daily_returns.iter().skip(stat_start) {
        sum_r += r;
        sum_sq_r += r * r;
        if r > 0.0 {
            win_count += 1;
        }
        if r < 0.0 {
            downside_sq += r * r;
        }
        valid_days += 1;
    }

    let (ann_vol, sharpe, sortino, win_rate) = if valid_days >= 2 {
        let mean_r = sum_r / valid_days as f64;
        let var_r = (sum_sq_r - valid_days as f64 * mean_r * mean_r) / (valid_days - 1) as f64;
        let std_r = var_r.max(0.0).sqrt();
        let ann_v = std_r * (252.0_f64).sqrt();
        let sh = if ann_v > 1e-6 {
            (mean_r * 252.0) / ann_v
        } else {
            0.0
        };
        let down_std = (downside_sq / valid_days as f64).sqrt() * (252.0_f64).sqrt();
        let so = if down_std > 1e-6 {
            (mean_r * 252.0) / down_std
        } else {
            0.0
        };
        (ann_v, sh, so, win_count as f64 / valid_days as f64)
    } else {
        (0.0, 0.0, 0.0, 0.0)
    };

    // 最大回撤
    let eq_anchor = stat_start.saturating_sub(1);
    let mut peak = if stat_start < t {
        equity[eq_anchor]
    } else {
        0.0
    };
    let mut max_dd = 0.0;
    let mut max_dd_days = 0;
    let mut current_dd_days = 0;

    if stat_start < t {
        for &e in equity.iter().skip(stat_start) {
            if e > peak {
                peak = e;
                current_dd_days = 0;
            } else {
                current_dd_days += 1;
                if peak > 0.0 {
                    let dd = (peak - e) / peak;
                    if dd > max_dd {
                        max_dd = dd;
                    }
                }
                if current_dd_days > max_dd_days {
                    max_dd_days = current_dd_days;
                }
            }
        }
    }

    let calmar = if max_dd > 1e-6 {
        annual_ret / max_dd
    } else {
        0.0
    };

    let mean_turnover = if valid_days > 0 {
        turnover.iter().skip(stat_start).sum::<f64>() / valid_days as f64
    } else {
        0.0
    };

    FastBacktestResult {
        equity,
        daily_returns,
        turnover,
        fees,
        slippage,
        total_return: total_ret,
        annual_return: annual_ret,
        annual_volatility: ann_vol,
        sharpe,
        sortino,
        calmar,
        max_drawdown: max_dd,
        max_drawdown_days: max_dd_days,
        win_rate,
        daily_turnover_mean: mean_turnover,
        annual_turnover: mean_turnover * 252.0,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;

    #[test]
    fn test_fast_backtest_bankruptcy_no_nan() {
        let t = 4;
        let n = 1;
        let mut weights = Array2::<f64>::zeros((t, n));
        weights[[0, 0]] = 1.0;
        weights[[1, 0]] = 1.0;
        weights[[2, 0]] = 1.0;

        let mut returns = Array2::<f64>::zeros((t, n));
        // Day 1 亏损 120% (穿仓)
        returns[[1, 0]] = -1.20;
        returns[[2, 0]] = 0.10;
        returns[[3, 0]] = 0.10;

        let cfg = FastBacktestConfig {
            fee_rate: 0.0001,
            slippage_bps: 1.0,
            init_cash: 100_000.0,
        };

        let res = run_fast_backtest(&weights, &returns, &cfg);

        // Day 1 穿仓后净值应为 0 (熔断冻结)
        assert_eq!(res.equity[1], 0.0);
        assert_eq!(res.equity[2], 0.0);
        assert_eq!(res.equity[3], 0.0);

        // 验证指标无 NaN 且年化收益保底 -1.0
        assert_eq!(res.total_return, -1.0);
        assert_eq!(res.annual_return, -1.0);
        assert!(!res.annual_return.is_nan());
        assert!(!res.calmar.is_nan());
    }
}
