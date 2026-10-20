//! 强类型 DSL 类型系统定义 (expr/types.rs)。
//!
//! 严格对应白皮书 7.1 节与 7.2 节定义：
//! - `Series`：连续时序面板矩阵 $(T \times N)$
//! - `Window`：离散时序滑动窗口正整数（如 5, 10, 20）
//! - `Constant`：浮点常数标量（如 0.5, 2.0, -1.0）
//! - `BoolCond`：条件布尔掩码矩阵 $(T \times N)$

use std::fmt;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DataType {
    Series,
    Window,
    Constant,
    BoolCond,
}

impl DataType {
    pub fn as_str(&self) -> &'static str {
        match self {
            DataType::Series => "Series",
            DataType::Window => "Window",
            DataType::Constant => "Constant",
            DataType::BoolCond => "BoolCond",
        }
    }
}

impl fmt::Display for DataType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.as_str())
    }
}
