//! AST 递归求值器与自底向上 DAG 并行批求值引擎 (expr/eval.rs)。
//!
//! 核心亮点：
//! 1. 单表达式求值：递归遍历 AST，优先查询 `SubExprCache`，命中时直接克隆 `Arc<Array2<f64>>`；
//! 2. 批量表达式求值（`eval_batch`）：
//!    - 将一批数千个 AST 树自动归并为按深度分层的有向无环图（Bottom-Up Level-Wise DAG）；
//!    - 同一深度层内的唯一子表达式通过 `rayon` 并行求值并写入 `SubExprCache`；
//!    - 彻底消除多线程同时求值相同公共子表达式的重复计算（Cache Stampede），实现超高吞吐。

use crate::expr::ast::ASTNode;
use crate::expr::cache::SubExprCache;
use crate::expr::canon::{canonicalize, canonicalize_value_preserving};
use crate::expr::parser::{parse_expr, ParseError};
use crate::expr::typecheck::{typecheck_factor_expr, TypeError};
use crate::ops::{cs, elem, group, ts, BinaryOp, TernaryOp, UnaryOp};
use crate::panel::{Panel, PanelError};
use ahash::AHashMap;
use ndarray::Array2;
use rayon::prelude::*;
use std::sync::Arc;
use thiserror::Error;

#[derive(Error, Debug, Clone)]
pub enum EvalError {
    #[error(transparent)]
    Parse(#[from] ParseError),
    #[error(transparent)]
    Type(#[from] TypeError),
    #[error(transparent)]
    Panel(#[from] PanelError),
    #[error("求值错误: 期望窗口参数 Window, 实际遇到 {0}")]
    ExpectedWindow(String),
    #[error("求值错误: 期望常数参数 Constant, 实际遇到 {0}")]
    ExpectedConstant(String),
    #[error("求值错误: 不能直接对标量或窗口叶子节点求值矩阵: {0}")]
    CannotEvalScalarLeaf(String),
    #[error("退化表达式已阻断: {0}")]
    DegenerateExpression(String),
}

fn extract_window(node: &ASTNode) -> Result<usize, EvalError> {
    match node {
        ASTNode::Window(w) => Ok(*w),
        other => Err(EvalError::ExpectedWindow(other.to_canon_string())),
    }
}

fn extract_constant(node: &ASTNode) -> Result<f64, EvalError> {
    match node {
        ASTNode::Constant(c) => Ok(*c),
        ASTNode::Window(w) => Ok(*w as f64),
        other => Err(EvalError::ExpectedConstant(other.to_canon_string())),
    }
}

/// 递归求值单个 `ASTNode`，返回共享列优先矩阵指针 `Arc<Array2<f64>>`。
pub fn eval_ast(
    node: &ASTNode,
    panel: &Panel,
    cache: Option<&SubExprCache>,
) -> Result<Arc<Array2<f64>>, EvalError> {
    match node {
        ASTNode::Field(name) => {
            let mat = panel.get_field(name)?;
            Ok(Arc::clone(mat))
        }
        ASTNode::Constant(_) | ASTNode::Window(_) => {
            Err(EvalError::CannotEvalScalarLeaf(node.to_canon_string()))
        }
        ASTNode::UnaryCall { op, arg } => {
            let key = node.canon_hash();
            if let Some(c) = cache {
                if let Some(hit) = c.get(key) {
                    return Ok(hit);
                }
            }
            let arg_mat = eval_ast(arg, panel, cache)?;
            let out = match op {
                UnaryOp::Neg => elem::op_neg(&arg_mat),
                UnaryOp::Abs => elem::op_abs(&arg_mat),
                UnaryOp::Sign => elem::op_sign(&arg_mat),
                UnaryOp::Inv => elem::op_inv(&arg_mat),
                UnaryOp::Slog => elem::op_slog(&arg_mat),
                UnaryOp::Ssqrt => elem::op_ssqrt(&arg_mat),
                UnaryOp::Square => elem::op_square(&arg_mat),
                UnaryOp::Cube => elem::op_cube(&arg_mat),
                UnaryOp::Sigmoid => elem::op_sigmoid(&arg_mat),
                UnaryOp::Tanh => elem::op_tanh(&arg_mat),
                UnaryOp::Relu => elem::op_relu(&arg_mat),
                UnaryOp::Atan => elem::op_atan(&arg_mat),
                UnaryOp::NotCond => elem::op_not_cond(&arg_mat),
                UnaryOp::CsRank => cs::op_cs_rank(&arg_mat),
                UnaryOp::CsZscore => cs::op_cs_zscore(&arg_mat),
                UnaryOp::CsDemean => cs::op_cs_demean(&arg_mat),
                UnaryOp::CsScale => cs::op_cs_scale(&arg_mat),
                UnaryOp::CsWinsorize => cs::op_cs_winsorize(&arg_mat),
            };
            let arc_out = Arc::new(out);
            if let Some(c) = cache {
                c.insert(key, Arc::clone(&arc_out));
            }
            Ok(arc_out)
        }
        ASTNode::BinaryCall { op, lhs, rhs } => {
            let key = node.canon_hash();
            if let Some(c) = cache {
                if let Some(hit) = c.get(key) {
                    return Ok(hit);
                }
            }
            let lhs_mat = eval_ast(lhs, panel, cache)?;
            let out = match op {
                // (Series, Series) -> Series / BoolCond
                BinaryOp::Add => elem::op_add(&lhs_mat, &*eval_ast(rhs, panel, cache)?),
                BinaryOp::Sub => elem::op_sub(&lhs_mat, &*eval_ast(rhs, panel, cache)?),
                BinaryOp::Mul => elem::op_mul(&lhs_mat, &*eval_ast(rhs, panel, cache)?),
                BinaryOp::Div => elem::op_div(&lhs_mat, &*eval_ast(rhs, panel, cache)?),
                BinaryOp::Max2 => elem::op_max2(&lhs_mat, &*eval_ast(rhs, panel, cache)?),
                BinaryOp::Min2 => elem::op_min2(&lhs_mat, &*eval_ast(rhs, panel, cache)?),
                BinaryOp::Hypot => elem::op_hypot(&lhs_mat, &*eval_ast(rhs, panel, cache)?),
                BinaryOp::Copysign => elem::op_copysign(&lhs_mat, &*eval_ast(rhs, panel, cache)?),
                BinaryOp::Gt => elem::op_gt(&lhs_mat, &*eval_ast(rhs, panel, cache)?),
                BinaryOp::Lt => elem::op_lt(&lhs_mat, &*eval_ast(rhs, panel, cache)?),
                BinaryOp::Ge => elem::op_ge(&lhs_mat, &*eval_ast(rhs, panel, cache)?),
                BinaryOp::Le => elem::op_le(&lhs_mat, &*eval_ast(rhs, panel, cache)?),
                BinaryOp::AndCond => elem::op_and_cond(&lhs_mat, &*eval_ast(rhs, panel, cache)?),
                BinaryOp::OrCond => elem::op_or_cond(&lhs_mat, &*eval_ast(rhs, panel, cache)?),
                BinaryOp::CsNeutralize => {
                    cs::op_cs_neutralize(&lhs_mat, &*eval_ast(rhs, panel, cache)?)
                }
                BinaryOp::GroupDemean => {
                    group::op_group_demean(&lhs_mat, &*eval_ast(rhs, panel, cache)?)
                }
                BinaryOp::GroupRank => {
                    group::op_group_rank(&lhs_mat, &*eval_ast(rhs, panel, cache)?)
                }
                BinaryOp::GroupZscore => {
                    group::op_group_zscore(&lhs_mat, &*eval_ast(rhs, panel, cache)?)
                }

                // (Series, Constant) -> Series / BoolCond
                BinaryOp::Addc => elem::op_addc(&lhs_mat, extract_constant(rhs)?),
                BinaryOp::Subc => elem::op_subc(&lhs_mat, extract_constant(rhs)?),
                BinaryOp::Mulc => elem::op_mulc(&lhs_mat, extract_constant(rhs)?),
                BinaryOp::Divc => elem::op_divc(&lhs_mat, extract_constant(rhs)?),
                BinaryOp::Spow => elem::op_spow(&lhs_mat, extract_constant(rhs)?),
                BinaryOp::Clip => elem::op_clip(&lhs_mat, extract_constant(rhs)?),
                BinaryOp::GtConst => elem::op_gt_const(&lhs_mat, extract_constant(rhs)?),
                BinaryOp::LtConst => elem::op_lt_const(&lhs_mat, extract_constant(rhs)?),

                // (Series, Window) -> Series
                BinaryOp::Delay => ts::op_delay(&lhs_mat, extract_window(rhs)?),
                BinaryOp::Delta => ts::op_delta(&lhs_mat, extract_window(rhs)?),
                BinaryOp::PctChange => ts::op_pct_change(&lhs_mat, extract_window(rhs)?),
                BinaryOp::LogReturn => ts::op_log_return(&lhs_mat, extract_window(rhs)?),
                BinaryOp::TsSum => ts::op_ts_sum(&lhs_mat, extract_window(rhs)?),
                BinaryOp::TsMean => ts::op_ts_mean(&lhs_mat, extract_window(rhs)?),
                BinaryOp::TsStd => ts::op_ts_std(&lhs_mat, extract_window(rhs)?),
                BinaryOp::TsVar => ts::op_ts_var(&lhs_mat, extract_window(rhs)?),
                BinaryOp::TsMin => ts::op_ts_min(&lhs_mat, extract_window(rhs)?),
                BinaryOp::TsMax => ts::op_ts_max(&lhs_mat, extract_window(rhs)?),
                BinaryOp::TsArgmin => ts::op_ts_argmin(&lhs_mat, extract_window(rhs)?),
                BinaryOp::TsArgmax => ts::op_ts_argmax(&lhs_mat, extract_window(rhs)?),
                BinaryOp::TsRank => ts::op_ts_rank(&lhs_mat, extract_window(rhs)?),
                BinaryOp::TsZscore => ts::op_ts_zscore(&lhs_mat, extract_window(rhs)?),
                BinaryOp::TsSkew => ts::op_ts_skew(&lhs_mat, extract_window(rhs)?),
                BinaryOp::TsKurt => ts::op_ts_kurt(&lhs_mat, extract_window(rhs)?),
                BinaryOp::TsEma => ts::op_ts_ema(&lhs_mat, extract_window(rhs)?),
                BinaryOp::TsDecayLinear => ts::op_ts_decay_linear(&lhs_mat, extract_window(rhs)?),
                BinaryOp::TsMad => ts::op_ts_mad(&lhs_mat, extract_window(rhs)?),
            };
            let arc_out = Arc::new(out);
            if let Some(c) = cache {
                c.insert(key, Arc::clone(&arc_out));
            }
            Ok(arc_out)
        }
        ASTNode::TernaryCall {
            op,
            cond,
            then_branch,
            else_branch,
        } => {
            let key = node.canon_hash();
            if let Some(c) = cache {
                if let Some(hit) = c.get(key) {
                    return Ok(hit);
                }
            }
            let out = match op {
                TernaryOp::Where => {
                    let c_mat = eval_ast(cond, panel, cache)?;
                    let t_mat = eval_ast(then_branch, panel, cache)?;
                    let e_mat = eval_ast(else_branch, panel, cache)?;
                    elem::op_where(&c_mat, &t_mat, &e_mat)
                }
                TernaryOp::TsCov => {
                    let x_mat = eval_ast(cond, panel, cache)?;
                    let y_mat = eval_ast(then_branch, panel, cache)?;
                    let w = extract_window(else_branch)?;
                    ts::op_ts_cov(&x_mat, &y_mat, w)
                }
                TernaryOp::TsCorr => {
                    let x_mat = eval_ast(cond, panel, cache)?;
                    let y_mat = eval_ast(then_branch, panel, cache)?;
                    let w = extract_window(else_branch)?;
                    ts::op_ts_corr(&x_mat, &y_mat, w)
                }
                TernaryOp::TsBeta => {
                    let x_mat = eval_ast(cond, panel, cache)?;
                    let y_mat = eval_ast(then_branch, panel, cache)?;
                    let w = extract_window(else_branch)?;
                    ts::op_ts_beta(&x_mat, &y_mat, w)
                }
                TernaryOp::TsResid => {
                    let x_mat = eval_ast(cond, panel, cache)?;
                    let y_mat = eval_ast(then_branch, panel, cache)?;
                    let w = extract_window(else_branch)?;
                    ts::op_ts_resid(&x_mat, &y_mat, w)
                }
            };
            let arc_out = Arc::new(out);
            if let Some(c) = cache {
                c.insert(key, Arc::clone(&arc_out));
            }
            Ok(arc_out)
        }
    }
}

/// 解析、校验并求值单个 DSL 字符串表达式。
pub fn eval_expr_str(
    expr: &str,
    panel: &Panel,
    cache: Option<&SubExprCache>,
    use_canonicalize: bool,
) -> Result<Arc<Array2<f64>>, EvalError> {
    let raw_ast = parse_expr(expr)?;
    let fields = panel.field_names_set();
    let labels = panel.label_names_set();
    typecheck_factor_expr(&raw_ast, Some(&fields), Some(&labels))?;

    let ast_to_eval = if use_canonicalize {
        let cr = canonicalize(&raw_ast);
        if cr.is_degenerate {
            return Err(EvalError::DegenerateExpression(
                cr.degenerate_reason
                    .unwrap_or_else(|| "退化表达式".to_string()),
            ));
        }
        cr.node
    } else {
        canonicalize_value_preserving(&raw_ast)
    };

    eval_ast(&ast_to_eval, panel, cache)
}

/// 递归收集语法树中所有非叶子求值节点（按深度分层去重）。
fn collect_dag_nodes_by_depth<'a>(node: &'a ASTNode, levels: &mut Vec<AHashMap<u64, &'a ASTNode>>) {
    match node {
        ASTNode::Field(_) | ASTNode::Constant(_) | ASTNode::Window(_) => {}
        ASTNode::UnaryCall { arg, .. } => {
            collect_dag_nodes_by_depth(arg, levels);
            let d = node.depth();
            if levels.len() <= d {
                levels.resize_with(d + 1, AHashMap::new);
            }
            levels[d].entry(node.canon_hash()).or_insert(node);
        }
        ASTNode::BinaryCall { lhs, rhs, .. } => {
            collect_dag_nodes_by_depth(lhs, levels);
            collect_dag_nodes_by_depth(rhs, levels);
            let d = node.depth();
            if levels.len() <= d {
                levels.resize_with(d + 1, AHashMap::new);
            }
            levels[d].entry(node.canon_hash()).or_insert(node);
        }
        ASTNode::TernaryCall {
            cond,
            then_branch,
            else_branch,
            ..
        } => {
            collect_dag_nodes_by_depth(cond, levels);
            collect_dag_nodes_by_depth(then_branch, levels);
            collect_dag_nodes_by_depth(else_branch, levels);
            let d = node.depth();
            if levels.len() <= d {
                levels.resize_with(d + 1, AHashMap::new);
            }
            levels[d].entry(node.canon_hash()).or_insert(node);
        }
    }
}

/// 批量并行求值一组 DSL 表达式（支持可选子表达式缓存与自底向上分层 DAG 调度）。
pub fn eval_batch_str(
    exprs: &[String],
    panel: &Panel,
    cache: Option<&SubExprCache>,
    use_canonicalize: bool,
) -> Vec<Result<Arc<Array2<f64>>, EvalError>> {
    let fields = panel.field_names_set();
    let labels = panel.label_names_set();

    // 1. 并行完成词法解析、静态类型检查与规范化化简
    let parsed_asts: Vec<Result<ASTNode, EvalError>> = exprs
        .par_iter()
        .map(|expr| {
            let raw_ast = parse_expr(expr)?;
            typecheck_factor_expr(&raw_ast, Some(&fields), Some(&labels))?;
            if use_canonicalize {
                let cr = canonicalize(&raw_ast);
                if cr.is_degenerate {
                    return Err(EvalError::DegenerateExpression(
                        cr.degenerate_reason
                            .unwrap_or_else(|| "退化表达式".to_string()),
                    ));
                }
                Ok(cr.node)
            } else {
                Ok(canonicalize_value_preserving(&raw_ast))
            }
        })
        .collect();

    // 2. 若启用了缓存且批量规模 >= 2，构建自底向上分层去重 DAG 并发预热求值
    if let Some(c) = cache {
        if exprs.len() >= 2 {
            let mut levels: Vec<AHashMap<u64, &ASTNode>> = Vec::with_capacity(16);
            for ast in parsed_asts.iter().flatten() {
                collect_dag_nodes_by_depth(ast, &mut levels);
            }
            for level_map in levels.iter().skip(2) {
                if level_map.is_empty() {
                    continue;
                }
                let nodes: Vec<&ASTNode> = level_map.values().copied().collect();
                nodes.par_iter().for_each(|&sub_node| {
                    let _ = eval_ast(sub_node, panel, Some(c));
                });
            }
        }
    }

    // 3. 并行收集所有根表达式结果
    parsed_asts
        .into_par_iter()
        .map(|res| match res {
            Ok(ast) => eval_ast(&ast, panel, cache),
            Err(e) => Err(e),
        })
        .collect()
}
