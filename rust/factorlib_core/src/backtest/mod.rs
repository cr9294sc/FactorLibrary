//! 多市场实盘语义向量化回测状态机 (`backtest`)。
//!
//! 包含四个子模块：
//! - `common`：跨市场公共绩效指标与账务计算
//! - `fast`：快速无摩擦/简易费率连续权重回测 (用于因子初筛与粗评)
//! - `futures`：期货实盘离散手数、涨跌停阻断、换月双边费与逐日盯市完整状态机
//! - `stocks`：A 股 T+1 整手买卖、涨跌停阻断、停牌盯市与非对称税费持仓级状态机

pub mod common;
pub mod fast;
pub mod futures;
pub mod stocks;

pub use common::{compute_common_metrics, CommonMetrics};
pub use fast::{run_fast_backtest, FastBacktestConfig, FastBacktestResult};
pub use futures::{
    run_futures_backtest, FeeType, FuturesBacktestConfig, FuturesBacktestResult, FuturesMetrics,
    RoundingMode, SymbolSpec,
};
pub use stocks::{run_stock_backtest, StockBacktestConfig, StockBacktestResult, StockMetrics};
