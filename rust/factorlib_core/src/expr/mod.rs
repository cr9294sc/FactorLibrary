//! DSL 表达式语法系统、类型检查、代数规范化与并行求值模块 (expr/mod.rs)。

pub mod ast;
pub mod cache;
pub mod canon;
pub mod eval;
pub mod lexer;
pub mod parser;
pub mod typecheck;
pub mod types;
