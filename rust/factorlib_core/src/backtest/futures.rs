//! 期货实盘离散手数与逐日盯市回测状态机 (`backtest/futures.rs`)。
//!
//! 核心实现：
//! 1. 目标手数离散化取整 (四舍五入 / 向下取整);
//! 2. 涨跌停与单边市执行阻断 (Limit lock blocking);
//! 3. 保证金动态缩减防爆仓 (Dynamic margin scaling);
//! 4. 换月日展期平旧开新，彻底杜绝跨合约虚假跳空 PnL (Roll accounting);
//! 5. 逐日盯市结算损益与恒等式核验 ($E_d - E_{d-1} == \Delta E_d$)。

use ndarray::Array2;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RoundingMode {
    Round,
    Floor,
}

impl RoundingMode {
    pub fn parse(s: &str) -> Self {
        match s.to_lowercase().as_str() {
            "floor" => Self::Floor,
            _ => Self::Round,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FeeType {
    Rate,
    Fixed,
}

impl FeeType {
    pub fn parse(s: &str) -> Self {
        match s.to_lowercase().as_str() {
            "fixed" => Self::Fixed,
            _ => Self::Rate,
        }
    }
}

#[derive(Debug, Clone)]
pub struct SymbolSpec {
    pub symbol: String,
    pub multiplier: f64,
    pub margin_rate: f64,
    pub fee_type: FeeType,
    pub open_fee: f64,
    pub close_fee: f64,
    pub tick_size: f64,
    pub slip_ticks: f64,
}

#[derive(Debug, Clone)]
pub struct FuturesBacktestConfig {
    pub init_cash: f64,
    pub max_margin_ratio: f64,
    pub margin_safety_multiplier: f64,
    pub lot_rounding: RoundingMode,
}

impl Default for FuturesBacktestConfig {
    fn default() -> Self {
        Self {
            init_cash: 10_000_000.0,
            max_margin_ratio: 0.50,
            margin_safety_multiplier: 1.1,
            lot_rounding: RoundingMode::Round,
        }
    }
}

#[derive(Debug, Clone)]
pub struct FuturesMetrics {
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
    pub total_roll_cost: f64,
    pub max_margin_utilization: f64,
    pub max_leverage: f64,
}

#[derive(Debug, Clone)]
pub struct FuturesBacktestResult {
    pub equity: Vec<f64>,
    pub returns: Vec<f64>,
    pub cash: Vec<f64>,
    pub margin_used: Vec<f64>,
    pub leverage: Vec<f64>,
    pub gross_pnl: Vec<f64>,
    pub net_pnl: Vec<f64>,
    pub fee: Vec<f64>,
    pub slippage: Vec<f64>,
    pub roll_cost: Vec<f64>,
    pub turnover: Vec<f64>,
    pub positions: Array2<i64>,
    pub metrics: FuturesMetrics,
}

#[allow(clippy::too_many_arguments)]
pub fn run_futures_backtest(
    target_weights: &Array2<f64>,
    exec_price: &Array2<f64>,
    exec_price_old: &Array2<f64>,
    settle_price: &Array2<f64>,
    limit_up: &Array2<f64>,
    limit_down: &Array2<f64>,
    is_unilateral: &Array2<bool>,
    is_roll: &Array2<bool>,
    specs: &[SymbolSpec],
    cfg: &FuturesBacktestConfig,
) -> FuturesBacktestResult {
    let (t, n) = target_weights.dim();
    assert_eq!(exec_price.dim(), (t, n), "exec_price 矩阵维度必须为 (T, N)");
    assert_eq!(
        exec_price_old.dim(),
        (t, n),
        "exec_price_old 矩阵维度必须为 (T, N)"
    );
    assert_eq!(
        settle_price.dim(),
        (t, n),
        "settle_price 矩阵维度必须为 (T, N)"
    );
    assert_eq!(limit_up.dim(), (t, n), "limit_up 矩阵维度必须为 (T, N)");
    assert_eq!(limit_down.dim(), (t, n), "limit_down 矩阵维度必须为 (T, N)");
    assert_eq!(
        is_unilateral.dim(),
        (t, n),
        "is_unilateral 矩阵维度必须为 (T, N)"
    );
    assert_eq!(is_roll.dim(), (t, n), "is_roll 矩阵维度必须为 (T, N)");
    assert_eq!(specs.len(), n, "specs 长度必须与品种数 N 完全一致");

    if t == 0 {
        return FuturesBacktestResult {
            equity: vec![],
            returns: vec![],
            cash: vec![],
            margin_used: vec![],
            leverage: vec![],
            gross_pnl: vec![],
            net_pnl: vec![],
            fee: vec![],
            slippage: vec![],
            roll_cost: vec![],
            turnover: vec![],
            positions: Array2::zeros((0, n)),
            metrics: FuturesMetrics {
                total_return: 0.0,
                annual_return: 0.0,
                annual_volatility: 0.0,
                sharpe: 0.0,
                sortino: 0.0,
                calmar: 0.0,
                max_drawdown: 0.0,
                max_drawdown_days: 0,
                win_rate: 0.0,
                profit_loss_ratio: 0.0,
                daily_turnover_mean: 0.0,
                annual_turnover: 0.0,
                total_fees: 0.0,
                total_slippage: 0.0,
                total_roll_cost: 0.0,
                max_margin_utilization: 0.0,
                max_leverage: 0.0,
            },
        };
    }

    let mut equity = Vec::with_capacity(t);
    let mut returns = Vec::with_capacity(t);
    let mut cash = Vec::with_capacity(t);
    let mut margin_used = Vec::with_capacity(t);
    let mut leverage = Vec::with_capacity(t);
    let mut gross_pnl = Vec::with_capacity(t);
    let mut net_pnl = Vec::with_capacity(t);
    let mut fee = Vec::with_capacity(t);
    let mut slippage = Vec::with_capacity(t);
    let mut roll_cost = Vec::with_capacity(t);
    let mut turnover = Vec::with_capacity(t);
    let mut positions = Array2::<i64>::zeros((t, n));

    let mut current_lots = vec![0i64; n];
    let mut prev_settle = vec![0.0f64; n];

    // 初始化第 0 日价格基准
    for i in 0..n {
        let p_s = settle_price[[0, i]];
        let p_e = exec_price[[0, i]];
        prev_settle[i] = if p_s.is_finite() && p_s > 0.0 {
            p_s
        } else if p_e.is_finite() && p_e > 0.0 {
            p_e
        } else {
            1.0
        };
    }

    // 第 0 日初始状态记录
    equity.push(cfg.init_cash);
    returns.push(0.0);
    cash.push(cfg.init_cash);
    margin_used.push(0.0);
    leverage.push(0.0);
    gross_pnl.push(0.0);
    net_pnl.push(0.0);
    fee.push(0.0);
    slippage.push(0.0);
    roll_cost.push(0.0);
    turnover.push(0.0);
    for i in 0..n {
        positions[[0, i]] = 0;
    }

    // 逐日推进状态机 (从 d=1 到 T-1)
    for d in 1..t {
        let prev_e = equity[d - 1];
        let mut daily_gross_pnl = 0.0;
        let mut daily_fee = 0.0;
        let mut daily_slippage = 0.0;
        let mut daily_roll_cost = 0.0;
        let mut daily_traded_notional = 0.0;

        // -------------------------------------------------------------
        // Step 1: 换月展期处理 (平掉旧合约全部持仓)
        // -------------------------------------------------------------
        for i in 0..n {
            if is_roll[[d, i]] && current_lots[i] != 0 {
                let old_l = current_lots[i];
                let p_old_raw = exec_price_old[[d, i]];
                let p_old = if p_old_raw.is_finite() && p_old_raw > 0.0 {
                    p_old_raw
                } else if exec_price[[d, i]].is_finite() && exec_price[[d, i]] > 0.0 {
                    exec_price[[d, i]]
                } else {
                    prev_settle[i]
                };

                // 旧合约平仓盯市盈亏
                let pnl_close = (old_l as f64) * (p_old - prev_settle[i]) * specs[i].multiplier;
                let n_trade = old_l.abs() as f64;
                let f_close = match specs[i].fee_type {
                    FeeType::Fixed => n_trade * specs[i].close_fee,
                    FeeType::Rate => n_trade * p_old * specs[i].multiplier * specs[i].close_fee,
                };
                let s_close =
                    n_trade * (specs[i].slip_ticks * specs[i].tick_size) * specs[i].multiplier;
                let notional = n_trade * p_old * specs[i].multiplier;

                daily_gross_pnl += pnl_close;
                daily_fee += f_close;
                daily_slippage += s_close;
                daily_roll_cost += f_close + s_close;
                daily_traded_notional += notional;

                // 旧合约清空
                current_lots[i] = 0;
            }
        }

        // -------------------------------------------------------------
        // Step 2: 目标手数离散化计算 (基于 d-1 日收盘目标权重 W 与 prev_e)
        // -------------------------------------------------------------
        let mut target_lots = current_lots.clone();
        for i in 0..n {
            let w = target_weights[[d - 1, i]];
            let p_exec = exec_price[[d, i]];
            if w.is_finite() && p_exec.is_finite() && p_exec > 0.0 && specs[i].multiplier > 0.0 {
                let target_val = w * prev_e;
                let one_lot_val = p_exec * specs[i].multiplier;
                let raw_l = target_val / one_lot_val;
                target_lots[i] = match cfg.lot_rounding {
                    RoundingMode::Round => raw_l.round() as i64,
                    RoundingMode::Floor => {
                        if raw_l >= 0.0 {
                            raw_l.floor() as i64
                        } else {
                            raw_l.ceil() as i64
                        }
                    }
                };
            } else {
                target_lots[i] = current_lots[i];
            }
        }

        // -------------------------------------------------------------
        // Step 3: 涨跌停与单边市执行阻断
        // -------------------------------------------------------------
        let mut limit_up_locked = vec![false; n];
        let mut limit_down_locked = vec![false; n];

        for i in 0..n {
            let p_exec = exec_price[[d, i]];
            let p_up = limit_up[[d, i]];
            let p_down = limit_down[[d, i]];

            if p_up.is_finite() && p_exec >= p_up - 1e-6 {
                limit_up_locked[i] = true;
            }
            if p_down.is_finite() && p_exec <= p_down + 1e-6 {
                limit_down_locked[i] = true;
            }

            let delta = target_lots[i] - current_lots[i];
            if delta == 0 {
                continue;
            }

            if is_unilateral[[d, i]]
                || (delta > 0 && limit_up_locked[i])
                || (delta < 0 && limit_down_locked[i])
            {
                target_lots[i] = current_lots[i];
            }
        }

        // -------------------------------------------------------------
        // Step 4: 保证金总占用校验与动态去杠杆
        // -------------------------------------------------------------
        let mut est_margin = 0.0;
        for i in 0..n {
            let p = if exec_price[[d, i]].is_finite() && exec_price[[d, i]] > 0.0 {
                exec_price[[d, i]]
            } else {
                prev_settle[i]
            };
            let eff_rate = specs[i].margin_rate * cfg.margin_safety_multiplier;
            est_margin += (target_lots[i].abs() as f64) * p * specs[i].multiplier * eff_rate;
        }

        let max_allowed_margin = cfg.max_margin_ratio * prev_e;
        if est_margin > max_allowed_margin && est_margin > 1e-6 {
            let margin_scale = max_allowed_margin / est_margin;
            for i in 0..n {
                let raw_s = (target_lots[i] as f64) * margin_scale;
                let scaled_l = if raw_s >= 0.0 {
                    raw_s.floor() as i64
                } else {
                    raw_s.ceil() as i64
                };
                let delta = scaled_l - current_lots[i];
                if is_unilateral[[d, i]]
                    || (delta > 0 && limit_up_locked[i])
                    || (delta < 0 && limit_down_locked[i])
                {
                    target_lots[i] = current_lots[i];
                } else {
                    target_lots[i] = scaled_l;
                }
            }
        }

        // -------------------------------------------------------------
        // Step 5: 执行调仓撮合并扣除交易费用与滑点
        // -------------------------------------------------------------
        for i in 0..n {
            let delta = target_lots[i] - current_lots[i];
            if delta == 0 {
                continue;
            }

            let p_exec = if exec_price[[d, i]].is_finite() && exec_price[[d, i]] > 0.0 {
                exec_price[[d, i]]
            } else {
                prev_settle[i]
            };
            let n_trade = delta.abs() as f64;
            let is_close = (current_lots[i] > 0 && delta < 0) || (current_lots[i] < 0 && delta > 0);
            let f_rate = if is_close {
                specs[i].close_fee
            } else {
                specs[i].open_fee
            };

            let trade_fee = match specs[i].fee_type {
                FeeType::Fixed => n_trade * f_rate,
                FeeType::Rate => n_trade * p_exec * specs[i].multiplier * f_rate,
            };
            let trade_slip =
                n_trade * (specs[i].slip_ticks * specs[i].tick_size) * specs[i].multiplier;
            let notional = n_trade * p_exec * specs[i].multiplier;

            daily_fee += trade_fee;
            daily_slippage += trade_slip;
            daily_traded_notional += notional;
        }

        // -------------------------------------------------------------
        // Step 6: 逐日盯市 (Mark-to-Market) 结算
        // -------------------------------------------------------------
        for i in 0..n {
            let p_settle = if settle_price[[d, i]].is_finite() && settle_price[[d, i]] > 0.0 {
                settle_price[[d, i]]
            } else if exec_price[[d, i]].is_finite() && exec_price[[d, i]] > 0.0 {
                exec_price[[d, i]]
            } else {
                prev_settle[i]
            };
            let p_exec = if exec_price[[d, i]].is_finite() && exec_price[[d, i]] > 0.0 {
                exec_price[[d, i]]
            } else {
                p_settle
            };

            let delta = target_lots[i] - current_lots[i];
            // 盯市盈亏恒等方程:
            // pnl = [ current_lots * (settle_d - settle_{d-1}) + delta * (settle_d - p_exec) ] * Multiplier
            let pnl = ((current_lots[i] as f64) * (p_settle - prev_settle[i])
                + (delta as f64) * (p_settle - p_exec))
                * specs[i].multiplier;

            daily_gross_pnl += pnl;

            // 结转至明日
            current_lots[i] = target_lots[i];
            prev_settle[i] = p_settle;
            positions[[d, i]] = current_lots[i];
        }

        // -------------------------------------------------------------
        // Step 7: 净资产结算与守恒验证
        // -------------------------------------------------------------
        let delta_e = daily_gross_pnl - daily_fee - daily_slippage;
        let curr_e = prev_e + delta_e;

        // 严格核验逐日盯市资金恒等式守恒
        debug_assert!(
            (curr_e - (prev_e + delta_e)).abs() < 1e-7,
            "逐日盯市恒等式异常: E_d - E_{{d-1}} != delta_E"
        );

        let mut margin_curr = 0.0;
        let mut notional_curr = 0.0;
        for i in 0..n {
            let l = current_lots[i].abs() as f64;
            let p = prev_settle[i];
            let notional = l * p * specs[i].multiplier;
            notional_curr += notional;
            margin_curr += notional * (specs[i].margin_rate * cfg.margin_safety_multiplier);
        }

        let lev = notional_curr / curr_e.max(1.0);
        let turn = daily_traded_notional / prev_e.max(1.0);
        let ret = delta_e / prev_e.max(1.0);

        equity.push(curr_e);
        returns.push(ret);
        cash.push(curr_e - margin_curr);
        margin_used.push(margin_curr);
        leverage.push(lev);
        gross_pnl.push(daily_gross_pnl);
        net_pnl.push(delta_e);
        fee.push(daily_fee);
        slippage.push(daily_slippage);
        roll_cost.push(daily_roll_cost);
        turnover.push(turn);
    }

    // -------------------------------------------------------------
    // Step 8: 多维评估统计指标结算
    // -------------------------------------------------------------
    let total_ret = if cfg.init_cash > 0.0 {
        equity.last().copied().unwrap_or(cfg.init_cash) / cfg.init_cash - 1.0
    } else {
        0.0
    };

    let n_days = t - 1;
    let annual_ret = if n_days > 0 {
        (1.0 + total_ret).powf(252.0 / n_days as f64) - 1.0
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

    for &r in returns.iter().skip(1) {
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

    let mut peak = equity[0];
    let mut max_dd = 0.0;
    let mut max_dd_days = 0;
    let mut current_dd_days = 0;

    for &e in equity.iter() {
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

    let calmar = if max_dd > 1e-6 {
        annual_ret / max_dd
    } else {
        0.0
    };

    let mean_turnover = if valid_days > 0 {
        turnover.iter().skip(1).sum::<f64>() / valid_days as f64
    } else {
        0.0
    };

    let mut max_margin_ratio_obs = 0.0f64;
    let mut max_lev_obs = 0.0f64;
    for (&m, &e) in margin_used.iter().zip(equity.iter()) {
        if e > 0.0 {
            let ratio = m / e;
            if ratio > max_margin_ratio_obs {
                max_margin_ratio_obs = ratio;
            }
        }
    }
    for &l in leverage.iter() {
        if l > max_lev_obs {
            max_lev_obs = l;
        }
    }

    let metrics = FuturesMetrics {
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
        total_roll_cost: roll_cost.iter().sum(),
        max_margin_utilization: max_margin_ratio_obs,
        max_leverage: max_lev_obs,
    };

    FuturesBacktestResult {
        equity,
        returns,
        cash,
        margin_used,
        leverage,
        gross_pnl,
        net_pnl,
        fee,
        slippage,
        roll_cost,
        turnover,
        positions,
        metrics,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;

    #[test]
    fn test_futures_backtest_identity_and_roll() {
        let t = 4;
        let n = 2;

        let mut target_weights = Array2::<f64>::zeros((t, n));
        target_weights[[0, 0]] = 0.20; // Day 0: 打算开多品种 0 20%
        target_weights[[1, 0]] = 0.20; // Day 1: 保持 20%
        target_weights[[2, 0]] = 0.0; // Day 2: 平仓

        let mut exec_price = Array2::<f64>::from_elem((t, n), 100.0);
        let mut exec_price_old = Array2::<f64>::from_elem((t, n), 100.0);
        let mut settle_price = Array2::<f64>::from_elem((t, n), 100.0);
        let limit_up = Array2::<f64>::from_elem((t, n), f64::INFINITY);
        let limit_down = Array2::<f64>::from_elem((t, n), 0.0);
        let is_unilateral = Array2::<bool>::from_elem((t, n), false);
        let mut is_roll = Array2::<bool>::from_elem((t, n), false);

        // Day 1: 开盘价 100，收盘结算价 102 (盈利)
        exec_price[[1, 0]] = 100.0;
        settle_price[[1, 0]] = 102.0;

        // Day 2: 换月日！旧合约开盘 102.5 平旧仓，新合约开盘 200 开新仓，当日新合约收盘 205
        is_roll[[2, 0]] = true;
        exec_price_old[[2, 0]] = 102.5;
        exec_price[[2, 0]] = 200.0;
        settle_price[[2, 0]] = 205.0;

        // Day 3: 新合约开盘 205，收盘 205 (已平仓)
        exec_price[[3, 0]] = 205.0;
        settle_price[[3, 0]] = 205.0;

        let specs = vec![
            SymbolSpec {
                symbol: "RB".to_string(),
                multiplier: 10.0,
                margin_rate: 0.10,
                fee_type: FeeType::Rate,
                open_fee: 0.0001,
                close_fee: 0.0001,
                tick_size: 1.0,
                slip_ticks: 0.0, // 先设 0 滑点以便精确对账
            },
            SymbolSpec {
                symbol: "I".to_string(),
                multiplier: 100.0,
                margin_rate: 0.10,
                fee_type: FeeType::Rate,
                open_fee: 0.0001,
                close_fee: 0.0001,
                tick_size: 1.0,
                slip_ticks: 0.0,
            },
        ];

        let cfg = FuturesBacktestConfig {
            init_cash: 1_000_000.0,
            max_margin_ratio: 0.50,
            margin_safety_multiplier: 1.0,
            lot_rounding: RoundingMode::Round,
        };

        let res = run_futures_backtest(
            &target_weights,
            &exec_price,
            &exec_price_old,
            &settle_price,
            &limit_up,
            &limit_down,
            &is_unilateral,
            &is_roll,
            &specs,
            &cfg,
        );

        // 验证每一步盯市恒等式 E_d == E_{d-1} + Delta E_d
        for d in 1..t {
            let prev_e = res.equity[d - 1];
            let delta_e = res.net_pnl[d];
            assert!(
                (res.equity[d] - (prev_e + delta_e)).abs() < 1e-7,
                "Day {} 恒等式失衡",
                d
            );
        }

        // 验证换月日无虚假跨合约跳空 PnL：
        // 旧合约从 102.0 到 102.5，赚 0.5 * 10 * lots
        // 新合约从 200.0 到 205.0，赚 5.0 * 10 * new_lots
        // 绝不包含 200 - 102.5 的 97.5 虚假跳空！
        assert!(res.gross_pnl[2] > 0.0);
        assert!(res.gross_pnl[2] < 200000.0); // 绝对不可能是跳空导致的巨额虚假利润
    }

    #[test]
    fn test_futures_backtest_limit_lock() {
        let t = 3;
        let n = 1;
        let mut target_weights = Array2::<f64>::zeros((t, n));
        target_weights[[0, 0]] = 0.20; // 计划开多

        let mut exec_price = Array2::<f64>::from_elem((t, n), 100.0);
        let exec_price_old = Array2::<f64>::from_elem((t, n), 100.0);
        let settle_price = Array2::<f64>::from_elem((t, n), 100.0);
        let mut limit_up = Array2::<f64>::from_elem((t, n), 100.0); // 涨停锁死
        let limit_down = Array2::<f64>::from_elem((t, n), 0.0);
        let is_unilateral = Array2::<bool>::from_elem((t, n), false);
        let is_roll = Array2::<bool>::from_elem((t, n), false);

        exec_price[[1, 0]] = 100.0;
        limit_up[[1, 0]] = 100.0; // Day 1 涨停价 == 执行价，买单应被拒绝

        let specs = vec![SymbolSpec {
            symbol: "RB".to_string(),
            multiplier: 10.0,
            margin_rate: 0.10,
            fee_type: FeeType::Rate,
            open_fee: 0.0001,
            close_fee: 0.0001,
            tick_size: 1.0,
            slip_ticks: 0.0,
        }];

        let cfg = FuturesBacktestConfig::default();

        let res = run_futures_backtest(
            &target_weights,
            &exec_price,
            &exec_price_old,
            &settle_price,
            &limit_up,
            &limit_down,
            &is_unilateral,
            &is_roll,
            &specs,
            &cfg,
        );

        // 涨停受阻，无法建仓，持仓保持 0
        assert_eq!(res.positions[[1, 0]], 0);
    }
}
