//! 期货实盘语义向量化回测状态机 (`backtest`)。
//!
//! 包含两个子模块：
//! - `fast`：快速无摩擦/简易费率连续权重回测 (用于因子初筛与粗评)
//! - `futures`：期货实盘离散手数、涨跌停阻断、换月双边费与逐日盯市 (Mark-to-Market) 完整状态机

pub mod fast;
pub mod futures;

pub use fast::{run_fast_backtest, FastBacktestConfig, FastBacktestResult};
pub use futures::{
    run_futures_backtest, FeeType, FuturesBacktestConfig, FuturesBacktestResult, FuturesMetrics,
    RoundingMode, SymbolSpec,
};
