//! 基于 `proptest` 的股票持仓级回测状态机性质测试 (`proptest_backtest.rs`)。

use factorlib_core::backtest::stocks::{run_stock_backtest, StockBacktestConfig};
use ndarray::{Array2, ShapeBuilder};
use proptest::prelude::*;

proptest! {
    #![proptest_config(ProptestConfig::with_cases(32))]

    #[test]
    fn prop_stock_backtest_accounting_and_constraints(
        t in 5usize..25,
        n in 2usize..8,
        raw_weights in proptest::collection::vec(-0.2f64..0.6, 25 * 8),
        raw_prices in proptest::collection::vec(5.0f64..100.0, 25 * 8),
        flags in proptest::collection::vec(0u8..10, 25 * 8),
    ) {
        let mut weights = Array2::<f64>::zeros((t, n).f());
        let mut exec_p = Array2::<f64>::zeros((t, n).f());
        let mut close_p = Array2::<f64>::zeros((t, n).f());
        let mut can_buy = Array2::<bool>::from_elem((t, n).f(), true);
        let mut can_sell = Array2::<bool>::from_elem((t, n).f(), true);
        let mut is_susp = Array2::<bool>::from_elem((t, n).f(), false);

        for c in 0..n {
            for r in 0..t {
                let idx = r * n + c;
                weights[[r, c]] = raw_weights[idx % raw_weights.len()];
                let p = raw_prices[idx % raw_prices.len()];
                exec_p[[r, c]] = p;
                close_p[[r, c]] = p * 1.01;
                let fl = flags[idx % flags.len()];
                if fl == 0 {
                    is_susp[[r, c]] = true;
                    can_buy[[r, c]] = false;
                    can_sell[[r, c]] = false;
                } else if fl == 1 {
                    can_buy[[r, c]] = false;
                } else if fl == 2 {
                    can_sell[[r, c]] = false;
                }
            }
        }

        let cfg = StockBacktestConfig {
            init_cash: 2_000_000.0,
            ..Default::default()
        };
        let res = run_stock_backtest(
            &weights,
            &exec_p,
            &close_p,
            &can_buy,
            &can_sell,
            &is_susp,
            None,
            &cfg,
        )
        .unwrap();

        for d in 1..t {
            // 1. 逐日盯市恒等式: E_d - E_{d-1} == net_pnl_d
            let diff = (res.equity[d] - (res.equity[d - 1] + res.net_pnl[d])).abs();
            prop_assert!(diff < 1e-6, "Day {d} MTM 恒等式偏差: {diff}");
            // 2. 现金非负 (纯现货不融资)
            prop_assert!(res.cash[d] >= -1e-6, "Day {d} 现金透支: {}", res.cash[d]);
            // 3. 非负持仓且严禁做空
            for c in 0..n {
                prop_assert!(res.positions[[d, c]] >= 0);
            }
        }
    }
}
