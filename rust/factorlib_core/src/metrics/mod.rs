//! 评估指标计算模块 (metrics/mod.rs)。
//!
//! 包含白皮书第 8 章与 P3 阶段全部核心评估指标：
//! - `ic`: 截面 Spearman Rank IC 与 Pearson IC、ICIR、t 统计量、胜率与有效覆盖率；
//! - `quantile`: 分位数分组收益、多空对冲收益与夏普比率、最大回撤、单调性得分；
//! - `turnover`: 因子截面排名自相关与日度换手率代理指标；
//! - `corr`: 因子截面相关系数矩阵与贪心去相关计算。

pub mod corr;
pub mod ic;
pub mod quantile;
pub mod turnover;

pub use corr::{
    cross_correlation_matrix, cross_correlation_pairwise, cross_correlation_pairwise_seq,
    cross_correlation_prepared_seq, CorrScratch, PreparedCandidate, PreparedFactor,
};
pub use ic::{calc_ic_metrics, calc_ic_series, IcMetrics};
pub use quantile::{
    calc_quantile_metrics, calc_top_quantile_turnover, QuantileMetrics, TopQuantileTurnoverMetrics,
};
pub use turnover::{calc_turnover_metrics, TurnoverMetrics};
