//! FactorLibrary Rust 极速计算内核 (`factorlib_core`)。
//!
//! 模块组成：
//! - `nan`：NaN 安全浮点运算与双栈 $O(1)$ Welford 状态机
//! - `panel`：Fortran 列优先面板矩阵容器与未来标签隔离
//! - `ops`：68 个逐元素、时序滚动、横截面与分组向量化算子
//! - `expr`：DSL 词法/语法分析器、强类型检查、代数规范化与子表达式 LRU 缓存
//! - `py`：PyO3 导出类 `Engine` 与工具函数

#![allow(clippy::useless_conversion)]

pub mod expr;
pub mod nan;
pub mod ops;
pub mod panel;
pub mod py;

use pyo3::prelude::*;

#[pymodule]
fn factorlib_core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<py::PyEngine>()?;
    m.add_function(wrap_pyfunction!(py::parse_and_canonicalize, m)?)?;
    m.add_function(wrap_pyfunction!(py::check_expr, m)?)?;
    m.add_function(wrap_pyfunction!(py::list_operators, m)?)?;
    Ok(())
}
