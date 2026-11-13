//! DSL 静态类型检查器 (expr/typecheck.rs)。
//!
//! 严格对应白皮书 7.2 节算子静态签名表：
//! - 校验每个算子调用的参数类型是否匹配（`Series`, `Window`, `Constant`, `BoolCond`）；
//! - 校验滑动窗口参数必须为正整数（$1 \le w \le 2500$）；
//! - 校验根节点返回类型必须为 `Series`（因子表达式必须产出连续面板矩阵）；
//! - 校验字段名是否存在于面板特征集，并拦截任何针对预测标签列的引用。

use crate::expr::ast::ASTNode;
use crate::expr::types::DataType;
use crate::ops::BinaryOp;
use crate::panel::is_reserved_label_name;
use std::collections::HashSet;
use thiserror::Error;

/// 前向累乘复权价格字段集合（绝对数值尺度依赖于 `start_date`，禁止与固定绝对常数直接比较）。
const ADJUSTED_PRICE_FIELDS: &[&str] = &["open", "high", "low", "close", "settle", "vwap"];

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

#[inline]
fn extract_adjusted_price_field(node: &ASTNode) -> Option<&str> {
    match node {
        ASTNode::Field(name) if ADJUSTED_PRICE_FIELDS.contains(&name.as_str()) => {
            Some(name.as_str())
        }
        ASTNode::UnaryCall { arg, .. } => extract_adjusted_price_field(arg),
        _ => None,
    }
}

#[inline]
fn extract_nonzero_constant(node: &ASTNode) -> Option<f64> {
    match node {
        ASTNode::Constant(c) if c.is_finite() && *c != 0.0 => Some(*c),
        ASTNode::Window(w) if *w != 0 => Some(*w as f64),
        _ => None,
    }
}

fn collect_warnings_rec(node: &ASTNode, out: &mut Vec<String>) {
    match node {
        ASTNode::Field(_) | ASTNode::Constant(_) | ASTNode::Window(_) => {}
        ASTNode::UnaryCall { arg, .. } => collect_warnings_rec(arg, out),
        ASTNode::BinaryCall { op, lhs, rhs } => {
            if matches!(
                op,
                BinaryOp::GtConst
                    | BinaryOp::LtConst
                    | BinaryOp::Gt
                    | BinaryOp::Lt
                    | BinaryOp::Ge
                    | BinaryOp::Le
            ) {
                if let (Some(field), Some(c)) = (
                    extract_adjusted_price_field(lhs),
                    extract_nonzero_constant(rhs),
                ) {
                    out.push(format!(
                        "复权价格字段 '{field}' 绝对尺度依赖于 start_date 基准，禁止在 DSL 中与固定绝对价格常数 ({c}) 比较 ({})",
                        op.name()
                    ));
                } else if let (Some(c), Some(field)) = (
                    extract_nonzero_constant(lhs),
                    extract_adjusted_price_field(rhs),
                ) {
                    out.push(format!(
                        "复权价格字段 '{field}' 绝对尺度依赖于 start_date 基准，禁止在 DSL 中与固定绝对价格常数 ({c}) 比较 ({})",
                        op.name()
                    ));
                }
            }
            collect_warnings_rec(lhs, out);
            collect_warnings_rec(rhs, out);
        }
        ASTNode::TernaryCall {
            cond,
            then_branch,
            else_branch,
            ..
        } => {
            collect_warnings_rec(cond, out);
            collect_warnings_rec(then_branch, out);
            collect_warnings_rec(else_branch, out);
        }
    }
}

/// 静态检查 AST 中的量化语义告警（例如复权价格与固定绝对常数比较）。
pub fn collect_type_warnings(node: &ASTNode) -> Vec<String> {
    let mut warnings = Vec::new();
    collect_warnings_rec(node, &mut warnings);
    warnings
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::expr::parser::parse_expr;

    #[test]
    fn test_adjusted_price_constant_comparison_warning() {
        let ast_warn = parse_expr("where(gt_const(close, 5000), 1.0, -1.0)").unwrap();
        let w1 = collect_type_warnings(&ast_warn);
        assert_eq!(w1.len(), 1);
        assert!(w1[0].contains("close") && w1[0].contains("5000"));

        let ast_infix = parse_expr("where(high >= 3000, 1.0, 0.0)").unwrap();
        let w2 = collect_type_warnings(&ast_infix);
        assert_eq!(w2.len(), 1);
        assert!(w2[0].contains("high") && w2[0].contains("3000"));

        // 与 0.0 的正负号检查不触发告警
        let ast_zero = parse_expr("where(gt_const(close, 0), 1.0, 0.0)").unwrap();
        assert!(collect_type_warnings(&ast_zero).is_empty());

        // 非复权价格字段（如 volume）不触发告警
        let ast_vol = parse_expr("where(gt_const(volume, 5000), 1.0, 0.0)").unwrap();
        assert!(collect_type_warnings(&ast_vol).is_empty());
    }
}
