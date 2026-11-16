//! 回测公共账务结构与绩效指标计算 (`backtest/common.rs`)。
//!
//! 抽取期货与股票持仓级回测状态机共用的收益、波动率、夏普、索提诺、卡尔玛、
//! 最大回撤与换手率计算逻辑，确保跨市场绩效评估口径严格一致。

/// 通用组合绩效指标集合。
#[derive(Debug, Clone, Default)]
pub struct CommonMetrics {
    pub total_return: f64,
    pub annual_return: f64,
    pub annual_volatility: f64,
    pub sharpe: f64,
    pub sortino: f64,
    pub calmar: f64,
    pub max_drawdown: f64,
    pub max_drawdown_days: usize,
    pub win_rate: f64,
    pub profit_loss_ratio: f64,
    pub daily_turnover_mean: f64,
    pub annual_turnover: f64,
    pub total_fees: f64,
    pub total_slippage: f64,
}

/// 基于每日净值、收益率与换手率序列计算组合绩效指标（自动剔除前导预热空仓日）。
pub fn compute_common_metrics(
    equity: &[f64],
    returns: &[f64],
    turnover: &[f64],
    fee: &[f64],
    slippage: &[f64],
    init_cash: f64,
    first_active_day: Option<usize>,
) -> CommonMetrics {
    let t = equity.len();
    let total_ret = if init_cash > 0.0 {
        equity.last().copied().unwrap_or(init_cash) / init_cash - 1.0
    } else {
        0.0
    };

    let stat_start = first_active_day.unwrap_or(t);
    let n_days = t.saturating_sub(stat_start);
    let annual_ret = if n_days > 0 {
        // 净值穿仓 1.0 + total_ret <= 0 时封底截断为 -1.0，防止负底数幂运算触发 IEEE-754 NaN
        if 1.0 + total_ret <= 0.0 {
            -1.0
        } else {
            (1.0 + total_ret).powf(252.0 / n_days as f64) - 1.0
        }
    } else {
        0.0
    };

    let mut sum_r = 0.0;
    let mut sum_sq_r = 0.0;
    let mut win_count = 0;
    let mut valid_days = 0;
    let mut downside_sq = 0.0;
    let mut sum_win = 0.0;
    let mut sum_loss = 0.0;
    let mut loss_count = 0;

    for &r in returns.iter().skip(stat_start) {
        sum_r += r;
        sum_sq_r += r * r;
        if r > 0.0 {
            win_count += 1;
            sum_win += r;
        } else if r < 0.0 {
            loss_count += 1;
            sum_loss += r.abs();
            downside_sq += r * r;
        }
        valid_days += 1;
    }

    let (ann_vol, sharpe, sortino, win_rate, pl_ratio) = if valid_days >= 2 {
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
        let wr = win_count as f64 / valid_days as f64;
        let avg_win = if win_count > 0 {
            sum_win / win_count as f64
        } else {
            0.0
        };
        let avg_loss = if loss_count > 0 {
            sum_loss / loss_count as f64
        } else {
            0.0
        };
        let plr = if avg_loss > 1e-8 {
            avg_win / avg_loss
        } else if avg_win > 0.0 {
            999.0
        } else {
            0.0
        };
        (ann_v, sh, so, wr, plr)
    } else {
        (0.0, 0.0, 0.0, 0.0, 0.0)
    };

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

    CommonMetrics {
        total_return: total_ret,
        annual_return: annual_ret,
        annual_volatility: ann_vol,
        sharpe,
        sortino,
        calmar,
        max_drawdown: max_dd,
        max_drawdown_days: max_dd_days,
        win_rate,
        profit_loss_ratio: pl_ratio,
        daily_turnover_mean: mean_turnover,
        annual_turnover: mean_turnover * 252.0,
        total_fees: fee.iter().sum(),
        total_slippage: slippage.iter().sum(),
    }
}
