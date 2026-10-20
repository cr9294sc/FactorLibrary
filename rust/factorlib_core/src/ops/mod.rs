//! 算子集合定义、静态签名表与统一分派入口 (ops/mod.rs)。
//!
//! 包含全部 68 个向量化算子（18 个一元算子 + 45 个二元算子 + 5 个三元算子），
//! 提供名称解析、静态类型签名查询、代数性质（交换律、幂等性、单调性）查询与矩阵求值分派。

pub mod cs;
pub mod elem;
pub mod group;
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
}

#[derive(Debug, Clone)]
pub struct OpSpec {
    pub name: &'static str,
    pub category: &'static str,
    pub arg_types: Vec<DataType>,
    pub return_type: DataType,
    pub commutative: bool,
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
    ];

    pub fn from_name(name: &str) -> Option<Self> {
        match name {
            "neg" => Some(UnaryOp::Neg),
            "abs" => Some(UnaryOp::Abs),
            "sign" => Some(UnaryOp::Sign),
            "inv" => Some(UnaryOp::Inv),
            "slog" => Some(UnaryOp::Slog),
            "ssqrt" => Some(UnaryOp::Ssqrt),
            "square" => Some(UnaryOp::Square),
            "cube" => Some(UnaryOp::Cube),
            "sigmoid" => Some(UnaryOp::Sigmoid),
            "tanh" => Some(UnaryOp::Tanh),
            "relu" => Some(UnaryOp::Relu),
            "atan" => Some(UnaryOp::Atan),
            "not_cond" => Some(UnaryOp::NotCond),
            "cs_rank" => Some(UnaryOp::CsRank),
            "cs_zscore" => Some(UnaryOp::CsZscore),
            "cs_demean" => Some(UnaryOp::CsDemean),
            "cs_scale" => Some(UnaryOp::CsScale),
            "cs_winsorize" => Some(UnaryOp::CsWinsorize),
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
        }
    }

    pub fn category(&self) -> &'static str {
        match self {
            UnaryOp::NotCond => "logic_unary",
            UnaryOp::CsRank
            | UnaryOp::CsZscore
            | UnaryOp::CsDemean
            | UnaryOp::CsScale
            | UnaryOp::CsWinsorize => "cs_unary",
            _ => "elem_unary",
        }
    }

    pub fn signature(&self) -> (DataType, DataType) {
        match self {
            UnaryOp::NotCond => (DataType::BoolCond, DataType::BoolCond),
            _ => (DataType::Series, DataType::Series),
        }
    }

    /// 是否在整个实数域上严格单调递增（用于 `cs_rank` 单调不变性化简）。
    pub fn is_strictly_increasing(&self) -> bool {
        matches!(
            self,
            UnaryOp::Slog
                | UnaryOp::Ssqrt
                | UnaryOp::Cube
                | UnaryOp::Sigmoid
                | UnaryOp::Tanh
                | UnaryOp::Atan
        )
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
    ];

    pub fn from_name(name: &str) -> Option<Self> {
        match name {
            "add" => Some(BinaryOp::Add),
            "sub" => Some(BinaryOp::Sub),
            "mul" => Some(BinaryOp::Mul),
            "div" => Some(BinaryOp::Div),
            "max2" => Some(BinaryOp::Max2),
            "min2" => Some(BinaryOp::Min2),
            "hypot" => Some(BinaryOp::Hypot),
            "copysign" => Some(BinaryOp::Copysign),
            "addc" => Some(BinaryOp::Addc),
            "subc" => Some(BinaryOp::Subc),
            "mulc" => Some(BinaryOp::Mulc),
            "divc" => Some(BinaryOp::Divc),
            "spow" => Some(BinaryOp::Spow),
            "clip" => Some(BinaryOp::Clip),
            "gt" => Some(BinaryOp::Gt),
            "lt" => Some(BinaryOp::Lt),
            "ge" => Some(BinaryOp::Ge),
            "le" => Some(BinaryOp::Le),
            "gt_const" => Some(BinaryOp::GtConst),
            "lt_const" => Some(BinaryOp::LtConst),
            "and_cond" => Some(BinaryOp::AndCond),
            "or_cond" => Some(BinaryOp::OrCond),
            "delay" => Some(BinaryOp::Delay),
            "delta" => Some(BinaryOp::Delta),
            "pct_change" => Some(BinaryOp::PctChange),
            "log_return" => Some(BinaryOp::LogReturn),
            "ts_sum" => Some(BinaryOp::TsSum),
            "ts_mean" => Some(BinaryOp::TsMean),
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
            "ts_decay_linear" => Some(BinaryOp::TsDecayLinear),
            "ts_mad" => Some(BinaryOp::TsMad),
            "cs_neutralize" => Some(BinaryOp::CsNeutralize),
            "group_demean" => Some(BinaryOp::GroupDemean),
            "group_rank" => Some(BinaryOp::GroupRank),
            "group_zscore" => Some(BinaryOp::GroupZscore),
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
            | BinaryOp::Clip => "elem_const",
            BinaryOp::Gt | BinaryOp::Lt | BinaryOp::Ge | BinaryOp::Le => "cmp_series",
            BinaryOp::GtConst | BinaryOp::LtConst => "cmp_const",
            BinaryOp::AndCond | BinaryOp::OrCond => "logic_binary",
            BinaryOp::CsNeutralize => "cs_binary",
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
            | BinaryOp::GroupZscore => (DataType::Series, DataType::Series, DataType::Series),

            BinaryOp::Addc
            | BinaryOp::Subc
            | BinaryOp::Mulc
            | BinaryOp::Divc
            | BinaryOp::Spow
            | BinaryOp::Clip => (DataType::Series, DataType::Constant, DataType::Series),

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
            | BinaryOp::TsMad => (DataType::Series, DataType::Window, DataType::Series),
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
    ];

    pub fn from_name(name: &str) -> Option<Self> {
        match name {
            "where" => Some(TernaryOp::Where),
            "ts_cov" => Some(TernaryOp::TsCov),
            "ts_corr" => Some(TernaryOp::TsCorr),
            "ts_beta" => Some(TernaryOp::TsBeta),
            "ts_resid" => Some(TernaryOp::TsResid),
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
        }
    }

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
            TernaryOp::TsCov | TernaryOp::TsCorr | TernaryOp::TsBeta | TernaryOp::TsResid => (
                DataType::Series,
                DataType::Series,
                DataType::Window,
                DataType::Series,
            ),
        }
    }

    /// 前两个参数是否满足交换律（`ts_cov(x, y, d) == ts_cov(y, x, d)`，`ts_corr(x, y, d) == ts_corr(y, x, d)`）。
    pub fn is_first_two_commutative(&self) -> bool {
        matches!(self, TernaryOp::TsCov | TernaryOp::TsCorr)
    }
}

/// 列出全部已注册的算子规格（共 68 个）。
pub fn list_all_operators() -> Vec<OpSpec> {
    let mut specs = Vec::with_capacity(68);
    for &op in UnaryOp::ALL {
        let (in_t, out_t) = op.signature();
        specs.push(OpSpec {
            name: op.name(),
            category: op.category(),
            arg_types: vec![in_t],
            return_type: out_t,
            commutative: false,
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
        });
    }
    specs
}
