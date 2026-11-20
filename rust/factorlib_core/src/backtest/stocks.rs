//! A 股持仓级向量化回测状态机 (`backtest/stocks.rs`)。
//!
//! 严格遵循 A 股实盘交易语义：
//! 1. 执行价格：T+1 早盘执行价 (`open_adj` / `open_raw` / `vwap_adj`)；
//! 2. 整手与零股：买入向下取整到 `lot_size`（默认 100 股）；全额卖出清仓允许零股；
//! 3. T+1 交收约束：当日买入的股票次日才可卖出；
//! 4. 涨跌停阻断：`can_buy_mask` / `can_sell_mask` 阻断未成交订单并计入 `blocked_orders`；
//! 5. 停牌处理：停牌期间不可交易，持仓按最后有效收盘价盯市；
//! 6. 双费率模式：
//!    - `flat_fee_rate == 0`（默认，`asymmetric`）：买入收佣金（含最低佣金）+ 过户费，
//!      卖出加征印花税，完全复刻 A 股真实非对称税费；
//!    - `flat_fee_rate > 0`（`flat` 对账模式）：买卖双边统一按成交额单边费率计费
//!      （例如 `fee_rate = 0.003`），不计最低佣金与印花税，等价于 `factorsim` 作业回测口径；
//!    滑点在两种模式下均支持 tick 与成交额 bps 比例；
//! 7. 做空控制：默认关闭；`allow_short=true` 暂未实现，显式报错拒绝。

use crate::backtest::common::compute_common_metrics;
use ndarray::{Array2, ShapeBuilder};

#[inline]
fn is_valid_price(p: f64) -> bool {
    p.is_finite() && p > 0.0
}

#[derive(Debug, Clone)]
pub struct StockBacktestConfig {
    pub init_cash: f64,
    pub lot_size: i64,
    pub t_plus_one: bool,
    pub allow_short: bool,
    pub commission_rate: f64,
    pub min_commission: f64,
    pub stamp_duty_rate: f64,
    pub transfer_fee_rate: f64,
    pub slip_ticks: f64,
    pub tick_size: f64,
    pub slip_bps: f64,
    /// 统一单边费率（`> 0` 时进入 `flat` 对账模式，忽略佣金/印花税/过户费的非对称结构）
    pub flat_fee_rate: f64,
}

impl Default for StockBacktestConfig {
    fn default() -> Self {
        Self {
            init_cash: 10_000_000.0,
            lot_size: 100,
            t_plus_one: true,
            allow_short: false,
            commission_rate: 0.0001,
            min_commission: 5.0,
            stamp_duty_rate: 0.0005,
            transfer_fee_rate: 0.00001,
            slip_ticks: 1.0,
            tick_size: 0.01,
            slip_bps: 0.0,
            flat_fee_rate: 0.0,
        }
    }
}

#[derive(Debug, Clone)]
pub struct StockMetrics {
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
    pub total_tax: f64,
    pub total_blocked_orders: usize,
    pub max_leverage: f64,
    pub bench_total_return: f64,
    pub bench_annual_return: f64,
    pub annual_excess_return: f64,
    pub excess_sharpe: f64,
}

#[derive(Debug, Clone)]
pub struct StockBacktestResult {
    pub equity: Vec<f64>,
    pub returns: Vec<f64>,
    pub cash: Vec<f64>,
    pub leverage: Vec<f64>,
    pub gross_pnl: Vec<f64>,
    pub net_pnl: Vec<f64>,
    pub fee: Vec<f64>,
    pub slippage: Vec<f64>,
    pub tax: Vec<f64>,
    pub turnover: Vec<f64>,
    pub blocked_orders: Vec<i64>,
    pub excess_returns: Vec<f64>,
    pub positions: Array2<i64>,
    pub metrics: StockMetrics,
}

/// 计算单笔订单成本 `(fee, tax, slippage)`。
///
/// - `flat_fee_rate > 0`：双边统一单边费率模式，`fee = notional * flat_fee_rate`，`tax = 0`；
/// - 否则：A 股真实非对称税费（买卖佣金含最低佣金 + 过户费，卖出加征印花税）。
#[inline]
fn calc_order_costs(delta: i64, p_exec: f64, cfg: &StockBacktestConfig) -> (f64, f64, f64) {
    if delta == 0 {
        return (0.0, 0.0, 0.0);
    }
    let qty = delta.abs() as f64;
    let notional = qty * p_exec;
    let slip = qty * (cfg.slip_ticks * cfg.tick_size) + notional * (cfg.slip_bps / 10_000.0);
    if cfg.flat_fee_rate > 0.0 {
        return (notional * cfg.flat_fee_rate, 0.0, slip);
    }
    let comm = if cfg.commission_rate > 0.0 || cfg.min_commission > 0.0 {
        (notional * cfg.commission_rate).max(cfg.min_commission)
    } else {
        0.0
    };
    let transfer = notional * cfg.transfer_fee_rate;
    let tax = if delta < 0 {
        notional * cfg.stamp_duty_rate
    } else {
        0.0
    };
    (comm + transfer, tax, slip)
}

#[allow(clippy::too_many_arguments)]
pub fn run_stock_backtest(
    target_weights: &Array2<f64>,
    exec_price: &Array2<f64>,
    close_price: &Array2<f64>,
    can_buy_mask: &Array2<bool>,
    can_sell_mask: &Array2<bool>,
    is_suspended: &Array2<bool>,
    benchmark_returns: Option<&[f64]>,
    cfg: &StockBacktestConfig,
) -> Result<StockBacktestResult, String> {
    if cfg.allow_short {
        return Err(
            "A 股融券做空回测暂未实现 (预留接口，当前仅支持 allow_short=false)".to_string(),
        );
    }
    if cfg.lot_size < 1 {
        return Err(format!("lot_size 必须 >= 1，实际 {}", cfg.lot_size));
    }
    if !cfg.flat_fee_rate.is_finite() || !(0.0..1.0).contains(&cfg.flat_fee_rate) {
        return Err(format!(
            "flat_fee_rate 必须位于 [0, 1) 区间，实际 {}",
            cfg.flat_fee_rate
        ));
    }

    let (t, n) = target_weights.dim();
    if exec_price.dim() != (t, n)
        || close_price.dim() != (t, n)
        || can_buy_mask.dim() != (t, n)
        || can_sell_mask.dim() != (t, n)
        || is_suspended.dim() != (t, n)
    {
        return Err("所有行情与掩码矩阵维度必须完全一致 (T, N)".to_string());
    }
    if let Some(bench) = benchmark_returns {
        if bench.len() != t {
            return Err(format!(
                "benchmark_returns 长度 ({}) 必须与时间轴 T ({t}) 一致",
                bench.len()
            ));
        }
    }

    if t == 0 {
        return Ok(StockBacktestResult {
            equity: vec![],
            returns: vec![],
            cash: vec![],
            leverage: vec![],
            gross_pnl: vec![],
            net_pnl: vec![],
            fee: vec![],
            slippage: vec![],
            tax: vec![],
            turnover: vec![],
            blocked_orders: vec![],
            excess_returns: vec![],
            positions: Array2::zeros((0, n).f()),
            metrics: StockMetrics {
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
                total_tax: 0.0,
                total_blocked_orders: 0,
                max_leverage: 0.0,
                bench_total_return: 0.0,
                bench_annual_return: 0.0,
                annual_excess_return: 0.0,
                excess_sharpe: 0.0,
            },
        });
    }

    let mut equity = Vec::with_capacity(t);
    let mut returns = Vec::with_capacity(t);
    let mut cash = Vec::with_capacity(t);
    let mut leverage = Vec::with_capacity(t);
    let mut gross_pnl = Vec::with_capacity(t);
    let mut net_pnl = Vec::with_capacity(t);
    let mut fee = Vec::with_capacity(t);
    let mut slippage = Vec::with_capacity(t);
    let mut tax = Vec::with_capacity(t);
    let mut turnover = Vec::with_capacity(t);
    let mut blocked_orders = Vec::with_capacity(t);
    let mut excess_returns = vec![0.0_f64; t];
    let mut positions = Array2::<i64>::zeros((t, n).f());

    let mut shares = vec![0_i64; n];
    let mut prev_close = vec![0.0_f64; n];
    let mut cash_bal = cfg.init_cash;

    for i in 0..n {
        let p_c = close_price[[0, i]];
        let p_e = exec_price[[0, i]];
        prev_close[i] = if !is_suspended[[0, i]] && is_valid_price(p_c) {
            p_c
        } else if !is_suspended[[0, i]] && is_valid_price(p_e) {
            p_e
        } else {
            0.0
        };
    }

    equity.push(cfg.init_cash);
    returns.push(0.0);
    cash.push(cfg.init_cash);
    leverage.push(0.0);
    gross_pnl.push(0.0);
    net_pnl.push(0.0);
    fee.push(0.0);
    slippage.push(0.0);
    tax.push(0.0);
    turnover.push(0.0);
    blocked_orders.push(0);

    let lot = cfg.lot_size;
    let lot_f = lot as f64;
    let mut first_active_day: Option<usize> = None;

    for d in 1..t {
        let prev_e = equity[d - 1];
        // T+1: 仅昨日收盘已有持仓今日可卖，今日新买入的份额次日才可卖
        let sellable = shares.clone();
        let mut exec_delta = vec![0_i64; n];
        let mut day_blocked = 0_i64;

        // -------------------------------------------------------------
        // Step 1: 解析目标权重并生成买卖意向（整手/清仓零股 + 停牌/涨跌停阻断）
        // -------------------------------------------------------------
        for i in 0..n {
            let w = target_weights[[d - 1, i]];
            let p_exec = exec_price[[d, i]];
            let susp = is_suspended[[d, i]];

            if !w.is_finite() {
                continue;
            }

            if w <= 0.0 {
                if shares[i] > 0 {
                    // 清仓全额卖出（允许零股）
                    if susp || !can_sell_mask[[d, i]] || !is_valid_price(p_exec) {
                        day_blocked += 1;
                    } else {
                        let qty = if cfg.t_plus_one {
                            sellable[i]
                        } else {
                            shares[i]
                        };
                        exec_delta[i] = -qty;
                    }
                }
                continue;
            }

            // w > 0.0
            if susp || !is_valid_price(p_exec) {
                if shares[i] == 0 {
                    day_blocked += 1;
                } else {
                    // 目标权重偏离现有持仓但停牌无法交易
                    day_blocked += 1;
                }
                continue;
            }

            let target_val = w * prev_e.max(0.0);
            let raw_target_shares = target_val / p_exec;
            let cur_s = shares[i] as f64;

            if raw_target_shares >= cur_s {
                let buy_qty = (((raw_target_shares - cur_s) / lot_f).floor() as i64).max(0) * lot;
                if buy_qty > 0 {
                    if !can_buy_mask[[d, i]] {
                        day_blocked += 1;
                    } else {
                        exec_delta[i] = buy_qty;
                    }
                }
            } else {
                let sell_qty = (((cur_s - raw_target_shares) / lot_f).floor() as i64).max(0) * lot;
                let max_sell = if cfg.t_plus_one {
                    sellable[i]
                } else {
                    shares[i]
                };
                let capped_sell = sell_qty.min(max_sell);
                if capped_sell > 0 {
                    if !can_sell_mask[[d, i]] {
                        day_blocked += 1;
                    } else {
                        exec_delta[i] = -capped_sell;
                    }
                }
            }
        }

        // -------------------------------------------------------------
        // Step 2: 先计入卖单回笼现金，再对买单做可用现金约束缩放（不允许现金透支）
        // -------------------------------------------------------------
        let mut avail_cash = cash_bal;
        for i in 0..n {
            if exec_delta[i] < 0 {
                let p_exec = exec_price[[d, i]];
                let (f, tx, sl) = calc_order_costs(exec_delta[i], p_exec, cfg);
                let proceeds = (-exec_delta[i] as f64) * p_exec - (f + tx + sl);
                avail_cash += proceeds;
            }
        }

        let compute_buy_cash_needed = |deltas: &[i64]| -> f64 {
            let mut sum = 0.0;
            for i in 0..n {
                if deltas[i] > 0 {
                    let p_exec = exec_price[[d, i]];
                    let (f, _tx, sl) = calc_order_costs(deltas[i], p_exec, cfg);
                    sum += (deltas[i] as f64) * p_exec + f + sl;
                }
            }
            sum
        };

        let mut total_buy_need = compute_buy_cash_needed(&exec_delta);
        let cash_cap = avail_cash.max(0.0);
        if total_buy_need > cash_cap {
            if cash_cap <= 0.0 {
                for delta in &mut exec_delta {
                    if *delta > 0 {
                        *delta = 0;
                    }
                }
            } else {
                let scale = (cash_cap / total_buy_need).clamp(0.0, 1.0);
                for delta in &mut exec_delta {
                    if *delta > 0 {
                        *delta = (((*delta as f64 * scale) / lot_f).floor() as i64).max(0) * lot;
                    }
                }
                total_buy_need = compute_buy_cash_needed(&exec_delta);
                // 若因最低佣金导致微小超额，逐手回退直到严格满足现金约束
                while total_buy_need > cash_cap + 1e-9 {
                    let mut trimmed = false;
                    for i in 0..n {
                        if exec_delta[i] >= lot {
                            exec_delta[i] -= lot;
                            trimmed = true;
                            total_buy_need = compute_buy_cash_needed(&exec_delta);
                            if total_buy_need <= cash_cap + 1e-9 {
                                break;
                            }
                        }
                    }
                    if !trimmed {
                        break;
                    }
                }
            }
        }

        // -------------------------------------------------------------
        // Step 3: 撮合成交结算费用与逐日盯市盈亏
        // -------------------------------------------------------------
        let mut daily_fee = 0.0;
        let mut daily_tax = 0.0;
        let mut daily_slippage = 0.0;
        let mut daily_traded_notional = 0.0;
        let mut daily_gross_pnl = 0.0;
        let mut market_value = 0.0;

        for i in 0..n {
            let delta = exec_delta[i];
            let p_exec_raw = exec_price[[d, i]];
            let p_close_raw = close_price[[d, i]];
            let susp = is_suspended[[d, i]];

            if delta != 0 {
                let p_ex = p_exec_raw;
                let (f, tx, sl) = calc_order_costs(delta, p_ex, cfg);
                daily_fee += f;
                daily_tax += tx;
                daily_slippage += sl;
                daily_traded_notional += (delta.abs() as f64) * p_ex;
                cash_bal -= (delta as f64) * p_ex + f + tx + sl;
            }

            // 停牌日严格保持最后有效价 prev_close[i]
            let p_mtm = if !susp && is_valid_price(p_close_raw) {
                p_close_raw
            } else if !susp && is_valid_price(p_exec_raw) {
                p_exec_raw
            } else {
                prev_close[i]
            };

            let p_ex_for_pnl = if is_valid_price(p_exec_raw) {
                p_exec_raw
            } else {
                p_mtm
            };

            let pnl = (shares[i] as f64) * (p_mtm - prev_close[i])
                + (delta as f64) * (p_mtm - p_ex_for_pnl);
            daily_gross_pnl += pnl;

            shares[i] += delta;
            prev_close[i] = p_mtm;
            positions[[d, i]] = shares[i];
            market_value += (shares[i] as f64) * p_mtm;
        }

        let daily_net_pnl = daily_gross_pnl - daily_fee - daily_tax - daily_slippage;
        let curr_e = cash_bal + market_value;

        debug_assert!(
            (curr_e - (prev_e + daily_net_pnl)).abs() < 1e-6,
            "股票逐日盯市恒等式异常: curr_e={curr_e}, expected={}",
            prev_e + daily_net_pnl
        );

        let lev = market_value / curr_e.max(1.0);
        let turn = daily_traded_notional / prev_e.max(1.0);
        let ret = daily_net_pnl / prev_e.max(1.0);

        if first_active_day.is_none()
            && (daily_traded_notional > 0.0
                || shares.iter().any(|&s| s != 0)
                || daily_net_pnl != 0.0)
        {
            first_active_day = Some(d);
        }

        if let Some(bench) = benchmark_returns {
            let b_r = if bench[d].is_finite() { bench[d] } else { 0.0 };
            excess_returns[d] = ret - b_r;
        } else {
            excess_returns[d] = ret;
        }

        equity.push(curr_e);
        returns.push(ret);
        cash.push(cash_bal);
        leverage.push(lev);
        gross_pnl.push(daily_gross_pnl);
        net_pnl.push(daily_net_pnl);
        fee.push(daily_fee);
        slippage.push(daily_slippage);
        tax.push(daily_tax);
        turnover.push(turn);
        blocked_orders.push(day_blocked);
    }

    // -------------------------------------------------------------
    // Step 4: 汇总绩效指标与基准超额指标
    // -------------------------------------------------------------
    let common = compute_common_metrics(
        &equity,
        &returns,
        &turnover,
        &fee,
        &slippage,
        cfg.init_cash,
        first_active_day,
    );

    let max_lev_obs = leverage.iter().copied().fold(0.0_f64, f64::max);

    let stat_start = first_active_day.unwrap_or(t);
    let n_active = t.saturating_sub(stat_start);
    let (bench_total_ret, bench_ann_ret, ann_excess_ret, excess_sharpe) = if let Some(bench) =
        benchmark_returns
    {
        if n_active > 0 {
            let mut b_nav = 1.0_f64;
            let mut ex_sum = 0.0_f64;
            let mut ex_sq = 0.0_f64;
            for d in stat_start..t {
                let b_r = if bench[d].is_finite() { bench[d] } else { 0.0 };
                b_nav *= 1.0 + b_r;
                let ex = excess_returns[d];
                ex_sum += ex;
                ex_sq += ex * ex;
            }
            let b_tot = b_nav - 1.0;
            let b_ann = if 1.0 + b_tot <= 0.0 {
                -1.0
            } else {
                (1.0 + b_tot).powf(252.0 / n_active as f64) - 1.0
            };
            let ex_ann = (1.0 + common.annual_return) / (1.0 + b_ann).max(1e-8) - 1.0;
            let ex_sh = if n_active >= 2 {
                let mean_ex = ex_sum / n_active as f64;
                let var_ex = (ex_sq - n_active as f64 * mean_ex * mean_ex) / (n_active - 1) as f64;
                let ann_ex_vol = var_ex.max(0.0).sqrt() * (252.0_f64).sqrt();
                if ann_ex_vol > 1e-6 {
                    (mean_ex * 252.0) / ann_ex_vol
                } else {
                    0.0
                }
            } else {
                0.0
            };
            (b_tot, b_ann, ex_ann, ex_sh)
        } else {
            (0.0, 0.0, 0.0, 0.0)
        }
    } else {
        (0.0, 0.0, common.annual_return, common.sharpe)
    };

    let metrics = StockMetrics {
        total_return: common.total_return,
        annual_return: common.annual_return,
        annual_volatility: common.annual_volatility,
        sharpe: common.sharpe,
        sortino: common.sortino,
        calmar: common.calmar,
        max_drawdown: common.max_drawdown,
        max_drawdown_days: common.max_drawdown_days,
        win_rate: common.win_rate,
        profit_loss_ratio: common.profit_loss_ratio,
        daily_turnover_mean: common.daily_turnover_mean,
        annual_turnover: common.annual_turnover,
        total_fees: common.total_fees,
        total_slippage: common.total_slippage,
        total_tax: tax.iter().sum(),
        total_blocked_orders: blocked_orders.iter().sum::<i64>().max(0) as usize,
        max_leverage: max_lev_obs,
        bench_total_return: bench_total_ret,
        bench_annual_return: bench_ann_ret,
        annual_excess_return: ann_excess_ret,
        excess_sharpe,
    };

    Ok(StockBacktestResult {
        equity,
        returns,
        cash,
        leverage,
        gross_pnl,
        net_pnl,
        fee,
        slippage,
        tax,
        turnover,
        blocked_orders,
        excess_returns,
        positions,
        metrics,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_stock_backtest_t_plus_one_and_lot_rounding() {
        let weights = array![[0.5, 0.0], [0.0, 0.0], [0.0, 0.0]];
        let exec_p = array![[10.0, 10.0], [10.0, 10.0], [12.0, 10.0]];
        let close_p = array![[10.0, 10.0], [11.0, 10.0], [12.0, 10.0]];
        let can_buy = Array2::from_elem((3, 2), true);
        let can_sell = Array2::from_elem((3, 2), true);
        let is_susp = Array2::from_elem((3, 2), false);

        let cfg = StockBacktestConfig {
            init_cash: 100_000.0,
            lot_size: 100,
            commission_rate: 0.0001,
            min_commission: 5.0,
            stamp_duty_rate: 0.0005,
            transfer_fee_rate: 0.00001,
            slip_ticks: 0.0,
            tick_size: 0.01,
            slip_bps: 0.0,
            ..Default::default()
        };

        let res = run_stock_backtest(
            &weights, &exec_p, &close_p, &can_buy, &can_sell, &is_susp, None, &cfg,
        )
        .unwrap();

        assert_eq!(res.positions[[1, 0]], 5000);
        assert_eq!(res.positions[[2, 0]], 0);
        assert!(res.tax[2] > 0.0);
        assert_eq!(res.tax[1], 0.0);
        for d in 1..3 {
            assert!((res.equity[d] - (res.equity[d - 1] + res.net_pnl[d])).abs() < 1e-8);
        }
    }

    #[test]
    fn test_flat_fee_mode_is_symmetric_and_ignores_tax() {
        // 第一天买入 5000 股 @10 元，第二天清仓卖出 @12 元
        let weights = array![[0.5, 0.0], [0.0, 0.0], [0.0, 0.0]];
        let exec_p = array![[10.0, 10.0], [10.0, 10.0], [12.0, 10.0]];
        let close_p = array![[10.0, 10.0], [11.0, 10.0], [12.0, 10.0]];
        let can_buy = Array2::from_elem((3, 2), true);
        let can_sell = Array2::from_elem((3, 2), true);
        let is_susp = Array2::from_elem((3, 2), false);

        let cfg = StockBacktestConfig {
            init_cash: 100_000.0,
            commission_rate: 0.0001,
            min_commission: 5.0,
            stamp_duty_rate: 0.0005,
            transfer_fee_rate: 0.00001,
            slip_ticks: 0.0,
            slip_bps: 0.0,
            flat_fee_rate: 0.003,
            ..Default::default()
        };
        let res = run_stock_backtest(
            &weights, &exec_p, &close_p, &can_buy, &can_sell, &is_susp, None, &cfg,
        )
        .unwrap();

        // 买入日：5000 股 × 10 元 × 0.003 = 150 元
        assert!((res.fee[1] - 150.0).abs() < 1e-9);
        assert_eq!(res.tax[1], 0.0);
        // 卖出日：5000 股 × 12 元 × 0.003 = 180 元，且印花税恒为 0
        assert!((res.fee[2] - 180.0).abs() < 1e-9);
        assert_eq!(res.tax[2], 0.0);
        assert!((res.metrics.total_tax).abs() < 1e-12);
        for d in 1..3 {
            assert!((res.equity[d] - (res.equity[d - 1] + res.net_pnl[d])).abs() < 1e-8);
        }

        // 对账口径：统一单边费率必须严格对称（买入/卖出同费率）
        let asym = StockBacktestConfig {
            flat_fee_rate: 0.0,
            ..cfg.clone()
        };
        let res2 = run_stock_backtest(
            &weights, &exec_p, &close_p, &can_buy, &can_sell, &is_susp, None, &asym,
        )
        .unwrap();
        assert!(res2.tax[2] > 0.0);
        assert_ne!(res2.fee[1], res.fee[1]);
    }

    #[test]
    fn test_flat_fee_rate_validation() {
        let z = Array2::<f64>::zeros((2, 1));
        let b = Array2::from_elem((2, 1), true);
        let s = Array2::from_elem((2, 1), false);
        let cfg = StockBacktestConfig {
            flat_fee_rate: -0.01,
            ..Default::default()
        };
        let err = run_stock_backtest(&z, &z, &z, &b, &b, &s, None, &cfg).unwrap_err();
        assert!(err.contains("flat_fee_rate"));
    }

    #[test]
    fn test_allow_short_rejected() {
        let cfg = StockBacktestConfig {
            allow_short: true,
            ..Default::default()
        };
        let z = Array2::<f64>::zeros((2, 1));
        let b = Array2::<bool>::from_elem((2, 1), true);
        let s = Array2::<bool>::from_elem((2, 1), false);
        let err = run_stock_backtest(&z, &z, &z, &b, &b, &s, None, &cfg).unwrap_err();
        assert!(err.contains("暂未实现"));
    }
}
