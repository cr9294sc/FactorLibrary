//! DSL 静态类型检查器 (expr/typecheck.rs)。
//!
//! 严格对应白皮书 7.2 节算子静态签名表：
//! - 校验每个算子调用的参数类型是否匹配（`Series`, `Window`, `Constant`, `BoolCond`）；
//! - 校验滑动窗口参数必须为正整数（$1 \le w \le 2500$）；
//! - 校验根节点返回类型必须为 `Series`（因子表达式必须产出连续面板矩阵）；
//! - 校验字段名是否存在于面板特征集，并拦截任何针对预测标签列的引用。

use crate::expr::ast::ASTNode;
use crate::expr::types::DataType;
use crate::panel::is_reserved_label_name;
use std::collections::HashSet;
use thiserror::Error;

#[derive(Error, Debug, Clone, PartialEq)]
pub enum TypeError {
    #[error("算子 '{op}' 第 {arg_idx} 个参数类型不匹配: 期望 {expected}, 实际为 {actual}")]
    ArgTypeMismatch {
        op: String,
        arg_idx: usize,
        expected: DataType,
        actual: DataType,
    },
    #[error("滑动窗口参数非法: 窗口大小必须 >= 1 且 <= 2500, 实际为 {0}")]
    InvalidWindowSize(usize),
    #[error("常数参数非法: 常数必须为有限浮点数, 实际为 {0}")]
    InvalidConstant(f64),
    #[error("严禁未来函数: 表达式禁止引用预测标签或保留字段 '{0}'")]
    ForbiddenLabelAccess(String),
    #[error("未知面板特征字段: '{0}'")]
    UnknownField(String),
    #[error("表达式根节点输出类型必须为 Series, 实际为 {0}")]
    RootNotSeries(DataType),
}

/// 递归推导并校验 AST 节点的输出类型。
pub fn infer_type(
    node: &ASTNode,
    available_fields: Option<&HashSet<String>>,
    forbidden_labels: Option<&HashSet<String>>,
) -> Result<DataType, TypeError> {
    match node {
        ASTNode::Field(name) => {
            if is_reserved_label_name(name) {
                return Err(TypeError::ForbiddenLabelAccess(name.clone()));
            }
            if let Some(labels) = forbidden_labels {
                if labels.contains(name) {
                    return Err(TypeError::ForbiddenLabelAccess(name.clone()));
                }
            }
            if let Some(fields) = available_fields {
                if !fields.contains(name) {
                    return Err(TypeError::UnknownField(name.clone()));
                }
            }
            Ok(DataType::Series)
        }
        ASTNode::Constant(c) => {
            if !c.is_finite() {
                return Err(TypeError::InvalidConstant(*c));
            }
            Ok(DataType::Constant)
        }
        ASTNode::Window(w) => {
            if *w == 0 || *w > 2500 {
                return Err(TypeError::InvalidWindowSize(*w));
            }
            Ok(DataType::Window)
        }
        ASTNode::UnaryCall { op, arg } => {
            let actual = infer_type(arg, available_fields, forbidden_labels)?;
            let (expected, out_type) = op.signature();
            if actual != expected {
                return Err(TypeError::ArgTypeMismatch {
                    op: op.name().to_string(),
                    arg_idx: 0,
                    expected,
                    actual,
                });
            }
            Ok(out_type)
        }
        ASTNode::BinaryCall { op, lhs, rhs } => {
            let actual_0 = infer_type(lhs, available_fields, forbidden_labels)?;
            let actual_1 = infer_type(rhs, available_fields, forbidden_labels)?;
            let (expected_0, expected_1, out_type) = op.signature();
            if actual_0 != expected_0 {
                return Err(TypeError::ArgTypeMismatch {
                    op: op.name().to_string(),
                    arg_idx: 0,
                    expected: expected_0,
                    actual: actual_0,
                });
            }
            if actual_1 != expected_1 {
                return Err(TypeError::ArgTypeMismatch {
                    op: op.name().to_string(),
                    arg_idx: 1,
                    expected: expected_1,
                    actual: actual_1,
                });
            }
            Ok(out_type)
        }
        ASTNode::TernaryCall {
            op,
            cond,
            then_branch,
            else_branch,
        } => {
            let actual_0 = infer_type(cond, available_fields, forbidden_labels)?;
            let actual_1 = infer_type(then_branch, available_fields, forbidden_labels)?;
            let actual_2 = infer_type(else_branch, available_fields, forbidden_labels)?;
            let (expected_0, expected_1, expected_2, out_type) = op.signature();
            if actual_0 != expected_0 {
                return Err(TypeError::ArgTypeMismatch {
                    op: op.name().to_string(),
                    arg_idx: 0,
                    expected: expected_0,
                    actual: actual_0,
                });
            }
            if actual_1 != expected_1 {
                return Err(TypeError::ArgTypeMismatch {
                    op: op.name().to_string(),
                    arg_idx: 1,
                    expected: expected_1,
                    actual: actual_1,
                });
            }
            if actual_2 != expected_2 {
                return Err(TypeError::ArgTypeMismatch {
                    op: op.name().to_string(),
                    arg_idx: 2,
                    expected: expected_2,
                    actual: actual_2,
                });
            }
            Ok(out_type)
        }
    }
}

/// 校验完整因子表达式：推导类型并确保根节点返回 `DataType::Series`。
pub fn typecheck_factor_expr(
    node: &ASTNode,
    available_fields: Option<&HashSet<String>>,
    forbidden_labels: Option<&HashSet<String>>,
) -> Result<DataType, TypeError> {
    let root_ty = infer_type(node, available_fields, forbidden_labels)?;
    if root_ty != DataType::Series {
        return Err(TypeError::RootNotSeries(root_ty));
    }
    Ok(root_ty)
}
