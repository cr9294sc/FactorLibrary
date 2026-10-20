//! 表达式批量并行求值基准性能压测 (benches/eval_bench.rs)。
//!
//! 压测场景：4000 交易日 × 80 期货品种面板上并行求值 1000 个复合因子表达式。

use factorlib_core::expr::cache::SubExprCache;
use factorlib_core::expr::eval::eval_batch_str;
use factorlib_core::panel::Panel;
use ndarray::{Array2, ShapeBuilder};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::Instant;

fn main() {
    let t = 4000usize;
    let n = 80usize;
    let dates: Vec<String> = (0..t).map(|i| format!("d_{i}")).collect();
    let symbols: Vec<String> = (0..n).map(|j| format!("SYM_{j}")).collect();
    let mask = Array2::from_elem((t, n).f(), true);

    let field_names = ["open", "high", "low", "close", "volume", "rv", "oi"];
    let mut fields = HashMap::new();
    for (f_idx, &fname) in field_names.iter().enumerate() {
        let mut mat = Array2::<f64>::zeros((t, n).f());
        for c in 0..n {
            for r in 0..t {
                let seed = ((r * 131 + c * 17 + f_idx * 31) % 1000) as f64;
                mat[[r, c]] = 100.0 + seed * 0.1;
            }
        }
        fields.insert(fname.to_string(), Arc::new(mat));
    }

    let panel = Panel::new(dates, symbols, fields, mask, HashMap::new()).unwrap();
    let windows = [3, 5, 10, 20, 40, 60, 120, 250];
    let unary_ops = ["cs_rank", "cs_zscore", "cs_demean", "slog", "ssqrt", "tanh"];
    let ts_ops = [
        "ts_mean",
        "ts_std",
        "ts_min",
        "ts_max",
        "ts_zscore",
        "ts_rank",
        "delta",
        "pct_change",
    ];
    let bin_ops = ["add", "sub", "mul", "div", "max2", "min2"];

    let mut exprs = Vec::with_capacity(1000);
    for i in 0..1000 {
        let f1 = field_names[i % field_names.len()];
        let f2 = field_names[(i * 3 + 1) % field_names.len()];
        let w1 = windows[i % windows.len()];
        let w2 = windows[(i / 3) % windows.len()];
        let ts1 = ts_ops[i % ts_ops.len()];
        let ts2 = ts_ops[(i + 2) % ts_ops.len()];
        let bop = bin_ops[i % bin_ops.len()];
        let uop = unary_ops[i % unary_ops.len()];
        exprs.push(format!(
            "{uop}({bop}({ts1}({f1}, {w1}), {ts2}({f2}, {w2})))"
        ));
    }

    let cache = SubExprCache::from_megabytes(2048);
    let t0 = Instant::now();
    let results = eval_batch_str(&exprs, &panel, Some(&cache), true);
    let elapsed = t0.elapsed().as_secs_f64();
    let ok_count = results.iter().filter(|r| r.is_ok()).count();
    let throughput = (exprs.len() as f64) / elapsed;
    println!(
        "[eval_bench] 4000x80 面板批求值 {} 表达式 (成功 {}): 耗时 {:.3}s, 吞吐量 {:.1} expr/s",
        exprs.len(),
        ok_count,
        elapsed,
        throughput
    );
}
