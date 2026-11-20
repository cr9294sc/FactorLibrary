//! 算子集合定义、静态签名表与统一分派入口 (ops/mod.rs)。
//!
//! 严格对齐 `docs/factorlab_mcp_catalog.md` 第四章的 106 个 FactorLab 1d 表达式算子，
//! 并提供同名别名（`greater`/`less`/`ifelse`/`log`/`sqrt`/`scale`/`rank`/`zScore`/`MAD`/
//! `winsor`/`indNeut`/`riskNeut`/`riskNeutMulti`/`ts_shift`/`ts_pct_chg`/`decay`/`emadecay` 等）。
//!
//! 注册算子分四类：
//! - `UnaryOp`：一元算子（逐元素、横截面、全历史时序、缺失值处理）；
//! - `BinaryOp`：二元算子（逐元素、比较、时序滚动窗口、横截面双变量、分组、常数参数）；
//! - `TernaryOp`：三元算子（`where`、时序二元滚动、`ts_quantile`）；
//! - `NaryOp`：变参多元截面/时序回归算子（`cs_multi_*` / `ts_multi_*` 家族，支持 K 个暴露矩阵）。

pub mod cs;
pub mod elem;
pub mod group;
pub mod multi;
pub mod postprocess;
pub mod ts;

use crate::expr::types::DataType;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum UnaryOp {
    // 算术一元 (12)
    Neg,
    Abs,
    Sign,
    Inv,
    Slog,
    Ssqrt,
    Square,
    Cube,
    Sigmoid,
    Tanh,
    Relu,
    Atan,
    // 逻辑一元 (1)
    NotCond,
    // 横截面一元 (5)
    CsRank,
    CsZscore,
    CsDemean,
    CsScale,
    CsWinsorize,
    // 数学变换扩充 (4)
    Exp,
    Ceil,
    SignPower,
    UniqueCount,
    // 缺失值与极值处理 (4)
    IsInf,
    IsNan,
    Purify,
    ZeroToNan,
    // 横截面统计扩充 (16)
    CsSum,
    CsMean,
    CsStd,
    CsVar,
    CsMax,
    CsMin,
    CsMedian,
    CsNoNanNum,
    CsArgMax,
    CsArgMin,
    CsPctRank,
    CsDenseRank,
    CsDensePctRank,
    CsSkew,
    CsKurt,
    CsMad,
    // 全历史时序排名 (4)
    RankTs,
    PctRankTs,
    DenseRankTs,
    PctDenseRankTs,
    // 全历史时序累积与填充 (5)
    TsCumProd,
    TsCumSum,
    TsFfill,
    TsDenseRank,
    TsPctDenseRank,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BinaryOp {
    // 算术二元 (Series, Series) -> Series (8)
    Add,
    Sub,
    Mul,
    Div,
    Max2,
    Min2,
    Hypot,
    Copysign,
    // 常数运算 (Series, Constant) -> Series (6)
    Addc,
    Subc,
    Mulc,
    Divc,
    Spow,
    Clip,
    // 比较运算 (Series, Series) -> BoolCond (4)
    Gt,
    Lt,
    Ge,
    Le,
    // 常数比较 (Series, Constant) -> BoolCond (2)
    GtConst,
    LtConst,
    // 逻辑二元 (BoolCond, BoolCond) -> BoolCond (2)
    AndCond,
    OrCond,
    // 时序滚动一元 (Series, Window) -> Series (19)
    Delay,
    Delta,
    PctChange,
    LogReturn,
    TsSum,
    TsMean,
    TsStd,
    TsVar,
    TsMin,
    TsMax,
    TsArgmin,
    TsArgmax,
    TsRank,
    TsZscore,
    TsSkew,
    TsKurt,
    TsEma,
    TsDecayLinear,
    TsMad,
    // 横截面二元 (Series, Series) -> Series (1)
    CsNeutralize,
    // 行业/板块分组 (Series, Series) -> Series (3)
    GroupDemean,
    GroupRank,
    GroupZscore,
    // 缺失值填充、常数参数与截面分位 (Series, Constant) -> Series (7)
    FillNa,
    RoundDec,
    PercLong,
    CsQuantile,
    CsGroup,
    WinsorExpr,
    MadWinsor,
    // 横截面双变量统计与市值加权标准化 (Series, Series) -> Series (5)
    CsBeta,
    CsCorr,
    CsCov,
    Barra5ZScore,
    TsDw,
    // 时序滚动窗口扩充 (Series, Window) -> Series (6)
    TsRoll,
    TsProd,
    TsMedian,
    TsNoNanNum,
    TsPctRank,
    TsGroup,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TernaryOp {
    // 条件分支 (BoolCond, Series, Series) -> Series (1)
    Where,
    // 时序二元滚动 (Series, Series, Window) -> Series (4)
    TsCov,
    TsCorr,
    TsBeta,
    TsResid,
    // 时序分位 (Series, Constant, Window) -> Series (1)
    TsQuantile,
}

/// 变参多元回归算子族（`cs_multi_*` / `ts_multi_*`）。
///
/// 参数布局（`intercept` 为可选尾部常数，缺省 0 即无截距）：
/// - `CsMultiBeta` / `CsMultiResi`：`(y, exp_1, ..., exp_K [, intercept])`，K >= 1
/// - `CsMultiWlsBeta` / `CsMultiWlsResi`：`(y, w, exp_1, ..., exp_K [, intercept])`，K >= 1
/// - `TsMultiBeta` / `TsMultiResi`：`(y, exp_1, ..., exp_K [, intercept])`，K >= 1
/// - `TsMultiWlsBeta` / `TsMultiWlsResi`：`(y, w, exp_1, ..., exp_K [, intercept])`，K >= 1
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum NaryOp {
    CsMultiBeta,
    CsMultiResi,
    CsMultiWlsBeta,
    CsMultiWlsResi,
    TsMultiBeta,
    TsMultiResi,
    TsMultiWlsBeta,
    TsMultiWlsResi,
}

#[derive(Debug, Clone)]
pub struct OpSpec {
    pub name: &'static str,
    pub category: &'static str,
    pub arg_types: Vec<DataType>,
    pub return_type: DataType,
    pub commutative: bool,
    pub variadic: bool,
}
impl UnaryOp {
    pub const ALL: &'static [UnaryOp] = &[
        UnaryOp::Neg,
        UnaryOp::Abs,
        UnaryOp::Sign,
        UnaryOp::Inv,
        UnaryOp::Slog,
        UnaryOp::Ssqrt,
        UnaryOp::Square,
        UnaryOp::Cube,
        UnaryOp::Sigmoid,
        UnaryOp::Tanh,
        UnaryOp::Relu,
        UnaryOp::Atan,
        UnaryOp::NotCond,
        UnaryOp::CsRank,
        UnaryOp::CsZscore,
        UnaryOp::CsDemean,
        UnaryOp::CsScale,
        UnaryOp::CsWinsorize,
        UnaryOp::Exp,
        UnaryOp::Ceil,
        UnaryOp::SignPower,
        UnaryOp::UniqueCount,
        UnaryOp::IsInf,
        UnaryOp::IsNan,
        UnaryOp::Purify,
        UnaryOp::ZeroToNan,
        UnaryOp::CsSum,
        UnaryOp::CsMean,
        UnaryOp::CsStd,
        UnaryOp::CsVar,
        UnaryOp::CsMax,
        UnaryOp::CsMin,
        UnaryOp::CsMedian,
        UnaryOp::CsNoNanNum,
        UnaryOp::CsArgMax,
        UnaryOp::CsArgMin,
        UnaryOp::CsPctRank,
        UnaryOp::CsDenseRank,
        UnaryOp::CsDensePctRank,
        UnaryOp::CsSkew,
        UnaryOp::CsKurt,
        UnaryOp::CsMad,
        UnaryOp::RankTs,
        UnaryOp::PctRankTs,
        UnaryOp::DenseRankTs,
        UnaryOp::PctDenseRankTs,
        UnaryOp::TsCumProd,
        UnaryOp::TsCumSum,
        UnaryOp::TsFfill,
        UnaryOp::TsDenseRank,
        UnaryOp::TsPctDenseRank,
    ];

    pub fn from_name(name: &str) -> Option<Self> {
        match name {
            "neg" => Some(UnaryOp::Neg),
            "abs" => Some(UnaryOp::Abs),
            "sign" => Some(UnaryOp::Sign),
            "inv" => Some(UnaryOp::Inv),
            "slog" | "log" => Some(UnaryOp::Slog),
            "ssqrt" | "sqrt" => Some(UnaryOp::Ssqrt),
            "square" => Some(UnaryOp::Square),
            "cube" => Some(UnaryOp::Cube),
            "sigmoid" => Some(UnaryOp::Sigmoid),
            "tanh" => Some(UnaryOp::Tanh),
            "relu" => Some(UnaryOp::Relu),
            "atan" => Some(UnaryOp::Atan),
            "not_cond" => Some(UnaryOp::NotCond),
            "cs_rank" => Some(UnaryOp::CsRank),
            "cs_zscore" | "zScore" => Some(UnaryOp::CsZscore),
            "cs_demean" => Some(UnaryOp::CsDemean),
            "cs_scale" | "scale" => Some(UnaryOp::CsScale),
            "cs_winsorize" => Some(UnaryOp::CsWinsorize),
            "exp" => Some(UnaryOp::Exp),
            "ceil" => Some(UnaryOp::Ceil),
            "sign_power" => Some(UnaryOp::SignPower),
            "unique_count" => Some(UnaryOp::UniqueCount),
            "is_inf" => Some(UnaryOp::IsInf),
            "is_nan" => Some(UnaryOp::IsNan),
            "purify" => Some(UnaryOp::Purify),
            "zero_to_nan" => Some(UnaryOp::ZeroToNan),
            "cs_sum" => Some(UnaryOp::CsSum),
            "cs_mean" => Some(UnaryOp::CsMean),
            "cs_std" => Some(UnaryOp::CsStd),
            "cs_var" => Some(UnaryOp::CsVar),
            "cs_max" => Some(UnaryOp::CsMax),
            "cs_min" => Some(UnaryOp::CsMin),
            "cs_median" => Some(UnaryOp::CsMedian),
            "cs_nonan_num" => Some(UnaryOp::CsNoNanNum),
            "cs_argmax" => Some(UnaryOp::CsArgMax),
            "cs_argmin" => Some(UnaryOp::CsArgMin),
            "cs_pct_rank" | "rank" => Some(UnaryOp::CsPctRank),
            "cs_dense_rank" => Some(UnaryOp::CsDenseRank),
            "cs_dense_pct_rank" => Some(UnaryOp::CsDensePctRank),
            "cs_skew" => Some(UnaryOp::CsSkew),
            "cs_kurt" => Some(UnaryOp::CsKurt),
            "cs_mad" => Some(UnaryOp::CsMad),
            "rank_ts" => Some(UnaryOp::RankTs),
            "pct_rank_ts" => Some(UnaryOp::PctRankTs),
            "dense_rank_ts" => Some(UnaryOp::DenseRankTs),
            "pct_dense_rank_ts" => Some(UnaryOp::PctDenseRankTs),
            "ts_cumprod" => Some(UnaryOp::TsCumProd),
            "ts_cumsum" => Some(UnaryOp::TsCumSum),
            "ts_ffill" => Some(UnaryOp::TsFfill),
            "ts_dense_rank" => Some(UnaryOp::TsDenseRank),
            "ts_pct_dense_rank" => Some(UnaryOp::TsPctDenseRank),
            _ => None,
        }
    }

    pub fn name(&self) -> &'static str {
        match self {
            UnaryOp::Neg => "neg",
            UnaryOp::Abs => "abs",
            UnaryOp::Sign => "sign",
            UnaryOp::Inv => "inv",
            UnaryOp::Slog => "slog",
            UnaryOp::Ssqrt => "ssqrt",
            UnaryOp::Square => "square",
            UnaryOp::Cube => "cube",
            UnaryOp::Sigmoid => "sigmoid",
            UnaryOp::Tanh => "tanh",
            UnaryOp::Relu => "relu",
            UnaryOp::Atan => "atan",
            UnaryOp::NotCond => "not_cond",
            UnaryOp::CsRank => "cs_rank",
            UnaryOp::CsZscore => "cs_zscore",
            UnaryOp::CsDemean => "cs_demean",
            UnaryOp::CsScale => "cs_scale",
            UnaryOp::CsWinsorize => "cs_winsorize",
            UnaryOp::Exp => "exp",
            UnaryOp::Ceil => "ceil",
            UnaryOp::SignPower => "sign_power",
            UnaryOp::UniqueCount => "unique_count",
            UnaryOp::IsInf => "is_inf",
            UnaryOp::IsNan => "is_nan",
            UnaryOp::Purify => "purify",
            UnaryOp::ZeroToNan => "zero_to_nan",
            UnaryOp::CsSum => "cs_sum",
            UnaryOp::CsMean => "cs_mean",
            UnaryOp::CsStd => "cs_std",
            UnaryOp::CsVar => "cs_var",
            UnaryOp::CsMax => "cs_max",
            UnaryOp::CsMin => "cs_min",
            UnaryOp::CsMedian => "cs_median",
            UnaryOp::CsNoNanNum => "cs_nonan_num",
            UnaryOp::CsArgMax => "cs_argmax",
            UnaryOp::CsArgMin => "cs_argmin",
            UnaryOp::CsPctRank => "cs_pct_rank",
            UnaryOp::CsDenseRank => "cs_dense_rank",
            UnaryOp::CsDensePctRank => "cs_dense_pct_rank",
            UnaryOp::CsSkew => "cs_skew",
            UnaryOp::CsKurt => "cs_kurt",
            UnaryOp::CsMad => "cs_mad",
            UnaryOp::RankTs => "rank_ts",
            UnaryOp::PctRankTs => "pct_rank_ts",
            UnaryOp::DenseRankTs => "dense_rank_ts",
            UnaryOp::PctDenseRankTs => "pct_dense_rank_ts",
            UnaryOp::TsCumProd => "ts_cumprod",
            UnaryOp::TsCumSum => "ts_cumsum",
            UnaryOp::TsFfill => "ts_ffill",
            UnaryOp::TsDenseRank => "ts_dense_rank",
            UnaryOp::TsPctDenseRank => "ts_pct_dense_rank",
        }
    }

    pub fn category(&self) -> &'static str {
        match self {
            UnaryOp::NotCond => "logic_unary",
            UnaryOp::CsRank
            | UnaryOp::CsZscore
            | UnaryOp::CsDemean
            | UnaryOp::CsScale
            | UnaryOp::CsWinsorize
            | UnaryOp::CsSum
            | UnaryOp::CsMean
            | UnaryOp::CsStd
            | UnaryOp::CsVar
            | UnaryOp::CsMax
            | UnaryOp::CsMin
            | UnaryOp::CsMedian
            | UnaryOp::CsNoNanNum
            | UnaryOp::CsArgMax
            | UnaryOp::CsArgMin
            | UnaryOp::CsPctRank
            | UnaryOp::CsDenseRank
            | UnaryOp::CsDensePctRank
            | UnaryOp::CsSkew
            | UnaryOp::CsKurt
            | UnaryOp::CsMad => "cs_unary",
            UnaryOp::RankTs
            | UnaryOp::PctRankTs
            | UnaryOp::DenseRankTs
            | UnaryOp::PctDenseRankTs
            | UnaryOp::TsCumProd
            | UnaryOp::TsCumSum
            | UnaryOp::TsFfill
            | UnaryOp::TsDenseRank
            | UnaryOp::TsPctDenseRank => "ts_full_history",
            _ => "elem_unary",
        }
    }

    pub fn signature(&self) -> (DataType, DataType) {
        match self {
            UnaryOp::NotCond => (DataType::BoolCond, DataType::BoolCond),
            _ => (DataType::Series, DataType::Series),
        }
    }

    /// 是否为“缺失值探测/清洗”类算子（输出 0/1 指示或剥离非有限值，不做 universe 掩码过滤）。
    pub fn is_nan_probe(&self) -> bool {
        matches!(
            self,
            UnaryOp::IsInf | UnaryOp::IsNan | UnaryOp::Purify | UnaryOp::ZeroToNan
        )
    }

    /// 是否在定义域内严格单调递增（用于 `cs_rank` 单调不变性化简）。
    pub fn is_strictly_increasing(&self) -> bool {
        matches!(
            self,
            UnaryOp::Slog | UnaryOp::Ssqrt | UnaryOp::Cube | UnaryOp::Atan
        ) || matches!(self, UnaryOp::Sigmoid | UnaryOp::Tanh | UnaryOp::Exp)
    }
}
impl BinaryOp {
    pub const ALL: &'static [BinaryOp] = &[
        BinaryOp::Add,
        BinaryOp::Sub,
        BinaryOp::Mul,
        BinaryOp::Div,
        BinaryOp::Max2,
        BinaryOp::Min2,
        BinaryOp::Hypot,
        BinaryOp::Copysign,
        BinaryOp::Addc,
        BinaryOp::Subc,
        BinaryOp::Mulc,
        BinaryOp::Divc,
        BinaryOp::Spow,
        BinaryOp::Clip,
        BinaryOp::Gt,
        BinaryOp::Lt,
        BinaryOp::Ge,
        BinaryOp::Le,
        BinaryOp::GtConst,
        BinaryOp::LtConst,
        BinaryOp::AndCond,
        BinaryOp::OrCond,
        BinaryOp::Delay,
        BinaryOp::Delta,
        BinaryOp::PctChange,
        BinaryOp::LogReturn,
        BinaryOp::TsSum,
        BinaryOp::TsMean,
        BinaryOp::TsStd,
        BinaryOp::TsVar,
        BinaryOp::TsMin,
        BinaryOp::TsMax,
        BinaryOp::TsArgmin,
        BinaryOp::TsArgmax,
        BinaryOp::TsRank,
        BinaryOp::TsZscore,
        BinaryOp::TsSkew,
        BinaryOp::TsKurt,
        BinaryOp::TsEma,
        BinaryOp::TsDecayLinear,
        BinaryOp::TsMad,
        BinaryOp::CsNeutralize,
        BinaryOp::GroupDemean,
        BinaryOp::GroupRank,
        BinaryOp::GroupZscore,
        BinaryOp::FillNa,
        BinaryOp::RoundDec,
        BinaryOp::PercLong,
        BinaryOp::CsQuantile,
        BinaryOp::CsGroup,
        BinaryOp::WinsorExpr,
        BinaryOp::MadWinsor,
        BinaryOp::CsBeta,
        BinaryOp::CsCorr,
        BinaryOp::CsCov,
        BinaryOp::Barra5ZScore,
        BinaryOp::TsDw,
        BinaryOp::TsRoll,
        BinaryOp::TsProd,
        BinaryOp::TsMedian,
        BinaryOp::TsNoNanNum,
        BinaryOp::TsPctRank,
        BinaryOp::TsGroup,
    ];

    pub fn from_name(name: &str) -> Option<Self> {
        match name {
            "add" => Some(BinaryOp::Add),
            "sub" => Some(BinaryOp::Sub),
            "mul" => Some(BinaryOp::Mul),
            "div" => Some(BinaryOp::Div),
            "max2" | "greater" => Some(BinaryOp::Max2),
            "min2" | "less" => Some(BinaryOp::Min2),
            "hypot" => Some(BinaryOp::Hypot),
            "copysign" => Some(BinaryOp::Copysign),
            "addc" => Some(BinaryOp::Addc),
            "subc" => Some(BinaryOp::Subc),
            "mulc" => Some(BinaryOp::Mulc),
            "divc" => Some(BinaryOp::Divc),
            "spow" | "power" => Some(BinaryOp::Spow),
            "clip" => Some(BinaryOp::Clip),
            "gt" => Some(BinaryOp::Gt),
            "lt" => Some(BinaryOp::Lt),
            "ge" => Some(BinaryOp::Ge),
            "le" => Some(BinaryOp::Le),
            "gt_const" => Some(BinaryOp::GtConst),
            "lt_const" => Some(BinaryOp::LtConst),
            "and_cond" => Some(BinaryOp::AndCond),
            "or_cond" => Some(BinaryOp::OrCond),
            "delay" | "ts_shift" => Some(BinaryOp::Delay),
            "delta" | "ts_delta" => Some(BinaryOp::Delta),
            "pct_change" | "ts_pct_chg" => Some(BinaryOp::PctChange),
            "log_return" => Some(BinaryOp::LogReturn),
            "ts_sum" => Some(BinaryOp::TsSum),
            "ts_mean" | "decay" => Some(BinaryOp::TsMean),
            "ts_std" => Some(BinaryOp::TsStd),
            "ts_var" => Some(BinaryOp::TsVar),
            "ts_min" => Some(BinaryOp::TsMin),
            "ts_max" => Some(BinaryOp::TsMax),
            "ts_argmin" => Some(BinaryOp::TsArgmin),
            "ts_argmax" => Some(BinaryOp::TsArgmax),
            "ts_rank" => Some(BinaryOp::TsRank),
            "ts_zscore" => Some(BinaryOp::TsZscore),
            "ts_skew" => Some(BinaryOp::TsSkew),
            "ts_kurt" => Some(BinaryOp::TsKurt),
            "ts_ema" => Some(BinaryOp::TsEma),
            "ts_decay_linear" | "emadecay" => Some(BinaryOp::TsDecayLinear),
            "ts_mad" => Some(BinaryOp::TsMad),
            "cs_neutralize" | "riskNeut" => Some(BinaryOp::CsNeutralize),
            "group_demean" | "indNeut" => Some(BinaryOp::GroupDemean),
            "group_rank" => Some(BinaryOp::GroupRank),
            "group_zscore" => Some(BinaryOp::GroupZscore),
            "fillna" => Some(BinaryOp::FillNa),
            "round" => Some(BinaryOp::RoundDec),
            "perc_long" => Some(BinaryOp::PercLong),
            "cs_quantile" => Some(BinaryOp::CsQuantile),
            "cs_group" => Some(BinaryOp::CsGroup),
            "winsor" => Some(BinaryOp::WinsorExpr),
            "MAD" => Some(BinaryOp::MadWinsor),
            "cs_beta" => Some(BinaryOp::CsBeta),
            "cs_corr" => Some(BinaryOp::CsCorr),
            "cs_cov" => Some(BinaryOp::CsCov),
            "barra5_zScore" => Some(BinaryOp::Barra5ZScore),
            "ts_dw" => Some(BinaryOp::TsDw),
            "ts_roll" => Some(BinaryOp::TsRoll),
            "ts_prod" => Some(BinaryOp::TsProd),
            "ts_median" => Some(BinaryOp::TsMedian),
            "ts_nonan_num" => Some(BinaryOp::TsNoNanNum),
            "ts_pct_rank" => Some(BinaryOp::TsPctRank),
            "ts_group" => Some(BinaryOp::TsGroup),
            _ => None,
        }
    }

    pub fn name(&self) -> &'static str {
        match self {
            BinaryOp::Add => "add",
            BinaryOp::Sub => "sub",
            BinaryOp::Mul => "mul",
            BinaryOp::Div => "div",
            BinaryOp::Max2 => "max2",
            BinaryOp::Min2 => "min2",
            BinaryOp::Hypot => "hypot",
            BinaryOp::Copysign => "copysign",
            BinaryOp::Addc => "addc",
            BinaryOp::Subc => "subc",
            BinaryOp::Mulc => "mulc",
            BinaryOp::Divc => "divc",
            BinaryOp::Spow => "spow",
            BinaryOp::Clip => "clip",
            BinaryOp::Gt => "gt",
            BinaryOp::Lt => "lt",
            BinaryOp::Ge => "ge",
            BinaryOp::Le => "le",
            BinaryOp::GtConst => "gt_const",
            BinaryOp::LtConst => "lt_const",
            BinaryOp::AndCond => "and_cond",
            BinaryOp::OrCond => "or_cond",
            BinaryOp::Delay => "delay",
            BinaryOp::Delta => "delta",
            BinaryOp::PctChange => "pct_change",
            BinaryOp::LogReturn => "log_return",
            BinaryOp::TsSum => "ts_sum",
            BinaryOp::TsMean => "ts_mean",
            BinaryOp::TsStd => "ts_std",
            BinaryOp::TsVar => "ts_var",
            BinaryOp::TsMin => "ts_min",
            BinaryOp::TsMax => "ts_max",
            BinaryOp::TsArgmin => "ts_argmin",
            BinaryOp::TsArgmax => "ts_argmax",
            BinaryOp::TsRank => "ts_rank",
            BinaryOp::TsZscore => "ts_zscore",
            BinaryOp::TsSkew => "ts_skew",
            BinaryOp::TsKurt => "ts_kurt",
            BinaryOp::TsEma => "ts_ema",
            BinaryOp::TsDecayLinear => "ts_decay_linear",
            BinaryOp::TsMad => "ts_mad",
            BinaryOp::CsNeutralize => "cs_neutralize",
            BinaryOp::GroupDemean => "group_demean",
            BinaryOp::GroupRank => "group_rank",
            BinaryOp::GroupZscore => "group_zscore",
            BinaryOp::FillNa => "fillna",
            BinaryOp::RoundDec => "round",
            BinaryOp::PercLong => "perc_long",
            BinaryOp::CsQuantile => "cs_quantile",
            BinaryOp::CsGroup => "cs_group",
            BinaryOp::WinsorExpr => "winsor",
            BinaryOp::MadWinsor => "MAD",
            BinaryOp::CsBeta => "cs_beta",
            BinaryOp::CsCorr => "cs_corr",
            BinaryOp::CsCov => "cs_cov",
            BinaryOp::Barra5ZScore => "barra5_zScore",
            BinaryOp::TsDw => "ts_dw",
            BinaryOp::TsRoll => "ts_roll",
            BinaryOp::TsProd => "ts_prod",
            BinaryOp::TsMedian => "ts_median",
            BinaryOp::TsNoNanNum => "ts_nonan_num",
            BinaryOp::TsPctRank => "ts_pct_rank",
            BinaryOp::TsGroup => "ts_group",
        }
    }

    pub fn category(&self) -> &'static str {
        match self {
            BinaryOp::Add
            | BinaryOp::Sub
            | BinaryOp::Mul
            | BinaryOp::Div
            | BinaryOp::Max2
            | BinaryOp::Min2
            | BinaryOp::Hypot
            | BinaryOp::Copysign => "elem_binary",
            BinaryOp::Addc
            | BinaryOp::Subc
            | BinaryOp::Mulc
            | BinaryOp::Divc
            | BinaryOp::Spow
            | BinaryOp::Clip
            | BinaryOp::FillNa
            | BinaryOp::RoundDec
            | BinaryOp::PercLong
            | BinaryOp::CsQuantile
            | BinaryOp::CsGroup
            | BinaryOp::WinsorExpr
            | BinaryOp::MadWinsor => "elem_const",
            BinaryOp::Gt | BinaryOp::Lt | BinaryOp::Ge | BinaryOp::Le => "cmp_series",
            BinaryOp::GtConst | BinaryOp::LtConst => "cmp_const",
            BinaryOp::AndCond | BinaryOp::OrCond => "logic_binary",
            BinaryOp::CsNeutralize
            | BinaryOp::CsBeta
            | BinaryOp::CsCorr
            | BinaryOp::CsCov
            | BinaryOp::Barra5ZScore
            | BinaryOp::TsDw => "cs_binary",
            BinaryOp::GroupDemean | BinaryOp::GroupRank | BinaryOp::GroupZscore => "group_binary",
            _ => "ts_unary",
        }
    }

    pub fn signature(&self) -> (DataType, DataType, DataType) {
        match self {
            BinaryOp::Add
            | BinaryOp::Sub
            | BinaryOp::Mul
            | BinaryOp::Div
            | BinaryOp::Max2
            | BinaryOp::Min2
            | BinaryOp::Hypot
            | BinaryOp::Copysign
            | BinaryOp::CsNeutralize
            | BinaryOp::GroupDemean
            | BinaryOp::GroupRank
            | BinaryOp::GroupZscore
            | BinaryOp::CsBeta
            | BinaryOp::CsCorr
            | BinaryOp::CsCov
            | BinaryOp::Barra5ZScore
            | BinaryOp::TsDw => (DataType::Series, DataType::Series, DataType::Series),

            BinaryOp::Addc
            | BinaryOp::Subc
            | BinaryOp::Mulc
            | BinaryOp::Divc
            | BinaryOp::Spow
            | BinaryOp::Clip
            | BinaryOp::FillNa
            | BinaryOp::RoundDec
            | BinaryOp::PercLong
            | BinaryOp::CsQuantile
            | BinaryOp::CsGroup
            | BinaryOp::WinsorExpr
            | BinaryOp::MadWinsor => (DataType::Series, DataType::Constant, DataType::Series),

            BinaryOp::Gt | BinaryOp::Lt | BinaryOp::Ge | BinaryOp::Le => {
                (DataType::Series, DataType::Series, DataType::BoolCond)
            }

            BinaryOp::GtConst | BinaryOp::LtConst => {
                (DataType::Series, DataType::Constant, DataType::BoolCond)
            }

            BinaryOp::AndCond | BinaryOp::OrCond => {
                (DataType::BoolCond, DataType::BoolCond, DataType::BoolCond)
            }

            BinaryOp::Delay
            | BinaryOp::Delta
            | BinaryOp::PctChange
            | BinaryOp::LogReturn
            | BinaryOp::TsSum
            | BinaryOp::TsMean
            | BinaryOp::TsStd
            | BinaryOp::TsVar
            | BinaryOp::TsMin
            | BinaryOp::TsMax
            | BinaryOp::TsArgmin
            | BinaryOp::TsArgmax
            | BinaryOp::TsRank
            | BinaryOp::TsZscore
            | BinaryOp::TsSkew
            | BinaryOp::TsKurt
            | BinaryOp::TsEma
            | BinaryOp::TsDecayLinear
            | BinaryOp::TsMad
            | BinaryOp::TsRoll
            | BinaryOp::TsProd
            | BinaryOp::TsMedian
            | BinaryOp::TsNoNanNum
            | BinaryOp::TsPctRank
            | BinaryOp::TsGroup => (DataType::Series, DataType::Window, DataType::Series),
        }
    }

    pub fn is_commutative(&self) -> bool {
        matches!(
            self,
            BinaryOp::Add
                | BinaryOp::Mul
                | BinaryOp::Max2
                | BinaryOp::Min2
                | BinaryOp::Hypot
                | BinaryOp::AndCond
                | BinaryOp::OrCond
        )
    }
}
impl TernaryOp {
    pub const ALL: &'static [TernaryOp] = &[
        TernaryOp::Where,
        TernaryOp::TsCov,
        TernaryOp::TsCorr,
        TernaryOp::TsBeta,
        TernaryOp::TsResid,
        TernaryOp::TsQuantile,
    ];

    pub fn from_name(name: &str) -> Option<Self> {
        match name {
            "where" | "ifelse" => Some(TernaryOp::Where),
            "ts_cov" => Some(TernaryOp::TsCov),
            "ts_corr" => Some(TernaryOp::TsCorr),
            "ts_beta" => Some(TernaryOp::TsBeta),
            "ts_resid" => Some(TernaryOp::TsResid),
            "ts_quantile" => Some(TernaryOp::TsQuantile),
            _ => None,
        }
    }

    pub fn name(&self) -> &'static str {
        match self {
            TernaryOp::Where => "where",
            TernaryOp::TsCov => "ts_cov",
            TernaryOp::TsCorr => "ts_corr",
            TernaryOp::TsBeta => "ts_beta",
            TernaryOp::TsResid => "ts_resid",
            TernaryOp::TsQuantile => "ts_quantile",
        }
    }

    /// 分类标签：`ts_binary` 同时驱动 `analyze_lookback` 对第三参数窗口的预热期推导。
    pub fn category(&self) -> &'static str {
        match self {
            TernaryOp::Where => "cond_ternary",
            _ => "ts_binary",
        }
    }

    pub fn signature(&self) -> (DataType, DataType, DataType, DataType) {
        match self {
            TernaryOp::Where => (
                DataType::BoolCond,
                DataType::Series,
                DataType::Series,
                DataType::Series,
            ),
            TernaryOp::TsQuantile => (
                DataType::Series,
                DataType::Constant,
                DataType::Window,
                DataType::Series,
            ),
            TernaryOp::TsCov | TernaryOp::TsCorr | TernaryOp::TsBeta | TernaryOp::TsResid => (
                DataType::Series,
                DataType::Series,
                DataType::Window,
                DataType::Series,
            ),
        }
    }

    /// 前两个参数是否满足交换律（`ts_cov(x, y, d) == ts_cov(y, x, d)`，`ts_corr` 同理）。
    pub fn is_first_two_commutative(&self) -> bool {
        matches!(self, TernaryOp::TsCov | TernaryOp::TsCorr)
    }
}

impl NaryOp {
    pub const ALL: &'static [NaryOp] = &[
        NaryOp::CsMultiBeta,
        NaryOp::CsMultiResi,
        NaryOp::CsMultiWlsBeta,
        NaryOp::CsMultiWlsResi,
        NaryOp::TsMultiBeta,
        NaryOp::TsMultiResi,
        NaryOp::TsMultiWlsBeta,
        NaryOp::TsMultiWlsResi,
    ];

    pub fn from_name(name: &str) -> Option<Self> {
        match name {
            "cs_multi_beta" => Some(NaryOp::CsMultiBeta),
            "cs_multi_resi" | "riskNeutMulti" => Some(NaryOp::CsMultiResi),
            "cs_multi_wls_beta" => Some(NaryOp::CsMultiWlsBeta),
            "cs_multi_wls_resi" => Some(NaryOp::CsMultiWlsResi),
            "ts_multi_beta" => Some(NaryOp::TsMultiBeta),
            "ts_multi_resi" => Some(NaryOp::TsMultiResi),
            "ts_multi_wls_beta" => Some(NaryOp::TsMultiWlsBeta),
            "ts_multi_wls_resi" => Some(NaryOp::TsMultiWlsResi),
            _ => None,
        }
    }

    pub fn name(&self) -> &'static str {
        match self {
            NaryOp::CsMultiBeta => "cs_multi_beta",
            NaryOp::CsMultiResi => "cs_multi_resi",
            NaryOp::CsMultiWlsBeta => "cs_multi_wls_beta",
            NaryOp::CsMultiWlsResi => "cs_multi_wls_resi",
            NaryOp::TsMultiBeta => "ts_multi_beta",
            NaryOp::TsMultiResi => "ts_multi_resi",
            NaryOp::TsMultiWlsBeta => "ts_multi_wls_beta",
            NaryOp::TsMultiWlsResi => "ts_multi_wls_resi",
        }
    }

    pub fn category(&self) -> &'static str {
        match self {
            NaryOp::CsMultiBeta
            | NaryOp::CsMultiResi
            | NaryOp::CsMultiWlsBeta
            | NaryOp::CsMultiWlsResi => "cs_nary_regression",
            _ => "ts_nary_regression",
        }
    }

    /// 是否为加权（WLS）形态：第二个参数为权重矩阵 `w`。
    pub fn is_weighted(&self) -> bool {
        matches!(
            self,
            NaryOp::CsMultiWlsBeta
                | NaryOp::CsMultiWlsResi
                | NaryOp::TsMultiWlsBeta
                | NaryOp::TsMultiWlsResi
        )
    }

    /// 是否沿时间轴（`true`）还是沿横截面（`false`）执行回归。
    pub fn is_time_series(&self) -> bool {
        matches!(
            self,
            NaryOp::TsMultiBeta
                | NaryOp::TsMultiResi
                | NaryOp::TsMultiWlsBeta
                | NaryOp::TsMultiWlsResi
        )
    }

    /// 是否返回回归系数（`beta`）而非残差（`resi`）。
    pub fn returns_beta(&self) -> bool {
        matches!(
            self,
            NaryOp::CsMultiBeta
                | NaryOp::CsMultiWlsBeta
                | NaryOp::TsMultiBeta
                | NaryOp::TsMultiWlsBeta
        )
    }

    /// 最少参数个数（不含可选尾部 `intercept`）。
    pub fn min_args(&self) -> usize {
        if self.is_weighted() {
            3 // (y, w, exp_1)
        } else {
            2 // (y, exp_1)
        }
    }
}

/// 全部已注册算子总数（`UnaryOp` + `BinaryOp` + `TernaryOp` + `NaryOp`）。
pub const REGISTERED_OP_COUNT: usize =
    UnaryOp::ALL.len() + BinaryOp::ALL.len() + TernaryOp::ALL.len() + NaryOp::ALL.len();

/// 列出全部已注册的算子规格（`UnaryOp` + `BinaryOp` + `TernaryOp` + `NaryOp`）。
pub fn list_all_operators() -> Vec<OpSpec> {
    let mut specs = Vec::with_capacity(REGISTERED_OP_COUNT);
    for &op in UnaryOp::ALL {
        let (in_t, out_t) = op.signature();
        specs.push(OpSpec {
            name: op.name(),
            category: op.category(),
            arg_types: vec![in_t],
            return_type: out_t,
            commutative: false,
            variadic: false,
        });
    }
    for &op in BinaryOp::ALL {
        let (t0, t1, out_t) = op.signature();
        specs.push(OpSpec {
            name: op.name(),
            category: op.category(),
            arg_types: vec![t0, t1],
            return_type: out_t,
            commutative: op.is_commutative(),
            variadic: false,
        });
    }
    for &op in TernaryOp::ALL {
        let (t0, t1, t2, out_t) = op.signature();
        specs.push(OpSpec {
            name: op.name(),
            category: op.category(),
            arg_types: vec![t0, t1, t2],
            return_type: out_t,
            commutative: op.is_first_two_commutative(),
            variadic: false,
        });
    }
    for &op in NaryOp::ALL {
        let mut arg_types = Vec::with_capacity(4);
        arg_types.push(DataType::Series);
        if op.is_weighted() {
            arg_types.push(DataType::Series);
        }
        arg_types.push(DataType::Series);
        arg_types.push(DataType::Constant);
        specs.push(OpSpec {
            name: op.name(),
            category: op.category(),
            arg_types,
            return_type: DataType::Series,
            commutative: false,
            variadic: true,
        });
    }
    specs
}
