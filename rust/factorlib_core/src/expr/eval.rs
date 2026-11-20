//! AST 递归求值器与自底向上 DAG 并行批求值引擎 (expr/eval.rs)。
//!
//! 核心亮点：
//! 1. 单表达式求值：递归遍历 AST，优先查询 `SubExprCache`，命中时直接克隆 `Arc<Array2<f64>>`；
//! 2. 批量表达式求值（`eval_batch`）：
//!    - 将一批数千个 AST 树自动归并为按深度分层的有向无环图（Bottom-Up Level-Wise DAG）；
//!    - 同一深度层内的唯一子表达式通过 `rayon` 并行求值并写入 `SubExprCache`；
//!    - 彻底消除多线程同时求值相同公共子表达式的重复计算（Cache Stampede），实现超高吞吐；
//! 3. 严格 universe 掩码语义：全部横截面算子先按 `panel.universe_mask` 屏蔽池外标的，
//!    缺失值探测类算子（`is_nan`/`is_inf`/`purify`/`zero_to_nan`）则跳过掩码以保留原始缺失信息。

use crate::expr::ast::{hash_canon_str, ASTNode};
use crate::expr::cache::SubExprCache;
use crate::expr::canon::{canonicalize, canonicalize_value_preserving};
use crate::expr::parser::{parse_expr, ParseError};
use crate::expr::typecheck::{typecheck_factor_expr, TypeError};
use crate::ops::{cs, elem, group, multi, postprocess, ts, BinaryOp, TernaryOp, UnaryOp};
use crate::panel::{Panel, PanelError};
use ahash::AHashMap;
use ndarray::Array2;
use rayon::prelude::*;
use std::borrow::Cow;
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
    #[error("求值错误: 变参回归算子 '{op}' 参数不完整 ({found} 个)")]
    InvalidNaryArgs { op: String, found: usize },
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

/// 按品种池有效性掩码过滤矩阵：若存在非在池标的（mask == false），将其置为 NaN；全在池时短路零拷贝借用。
#[inline]
fn mask_to_universe<'a>(mat: &'a Array2<f64>, mask: &Array2<bool>) -> Cow<'a, Array2<f64>> {
    let (t, n) = mat.dim();
    debug_assert_eq!((t, n), mask.dim());
    let has_out_of_universe = mask.iter().any(|&m| !m);
    if !has_out_of_universe {
        return Cow::Borrowed(mat);
    }
    let mut out = mat.clone();
    for col in 0..n {
        for row in 0..t {
            if !mask[[row, col]] {
                out[[row, col]] = f64::NAN;
            }
        }
    }
    Cow::Owned(out)
}

/// 带分片防碰撞缓存的公共非叶子节点求值包装。
fn eval_nonleaf_cached<F>(
    node: &ASTNode,
    cache: Option<&SubExprCache>,
    compute: F,
) -> Result<Arc<Array2<f64>>, EvalError>
where
    F: FnOnce() -> Result<Array2<f64>, EvalError>,
{
    if let Some(c) = cache {
        let canon_expr = node.to_canon_string();
        let key_hash = hash_canon_str(&canon_expr);
        if let Some(hit) = c.get(key_hash, &canon_expr) {
            return Ok(hit);
        }
        let arc_out = Arc::new(compute()?);
        c.insert(key_hash, canon_expr, Arc::clone(&arc_out));
        Ok(arc_out)
    } else {
        Ok(Arc::new(compute()?))
    }
}

/// 一元算子分派（`masked` 为已按 universe 掩码过滤后的输入）。
fn dispatch_unary(op: UnaryOp, m: &Array2<f64>) -> Array2<f64> {
    match op {
        UnaryOp::Neg => elem::op_neg(m),
        UnaryOp::Abs => elem::op_abs(m),
        UnaryOp::Sign => elem::op_sign(m),
        UnaryOp::Inv => elem::op_inv(m),
        UnaryOp::Slog => elem::op_slog(m),
        UnaryOp::Ssqrt => elem::op_ssqrt(m),
        UnaryOp::Square => elem::op_square(m),
        UnaryOp::Cube => elem::op_cube(m),
        UnaryOp::Sigmoid => elem::op_sigmoid(m),
        UnaryOp::Tanh => elem::op_tanh(m),
        UnaryOp::Relu => elem::op_relu(m),
        UnaryOp::Atan => elem::op_atan(m),
        UnaryOp::NotCond => elem::op_not_cond(m),
        UnaryOp::Exp => elem::op_exp(m),
        UnaryOp::Ceil => elem::op_ceil(m),
        UnaryOp::SignPower => elem::op_sign_power(m),
        UnaryOp::UniqueCount => elem::op_unique_count(m),
        UnaryOp::IsInf => elem::op_is_inf(m),
        UnaryOp::IsNan => elem::op_is_nan(m),
        UnaryOp::Purify => elem::op_purify(m),
        UnaryOp::ZeroToNan => elem::op_zero_to_nan(m),
        UnaryOp::CsRank => cs::op_cs_rank(m),
        UnaryOp::CsZscore => cs::op_cs_zscore(m),
        UnaryOp::CsDemean => cs::op_cs_demean(m),
        UnaryOp::CsScale => cs::op_cs_scale(m),
        UnaryOp::CsWinsorize => cs::op_cs_winsorize(m),
        UnaryOp::CsSum => cs::op_cs_sum(m),
        UnaryOp::CsMean => cs::op_cs_mean(m),
        UnaryOp::CsStd => cs::op_cs_std(m),
        UnaryOp::CsVar => cs::op_cs_var(m),
        UnaryOp::CsMax => cs::op_cs_max(m),
        UnaryOp::CsMin => cs::op_cs_min(m),
        UnaryOp::CsMedian => cs::op_cs_median(m),
        UnaryOp::CsNoNanNum => cs::op_cs_nonan_num(m),
        UnaryOp::CsArgMax => cs::op_cs_argmax(m),
        UnaryOp::CsArgMin => cs::op_cs_argmin(m),
        UnaryOp::CsPctRank => cs::op_cs_pct_rank(m),
        UnaryOp::CsDenseRank => cs::op_cs_dense_rank(m),
        UnaryOp::CsDensePctRank => cs::op_cs_dense_pct_rank(m),
        UnaryOp::CsSkew => cs::op_cs_skew(m),
        UnaryOp::CsKurt => cs::op_cs_kurt(m),
        UnaryOp::CsMad => cs::op_cs_mad(m),
        UnaryOp::RankTs => ts::op_rank_ts(m),
        UnaryOp::PctRankTs => ts::op_pct_rank_ts(m),
        UnaryOp::DenseRankTs => ts::op_dense_rank_ts(m),
        UnaryOp::PctDenseRankTs => ts::op_pct_dense_rank_ts(m),
        UnaryOp::TsCumProd => ts::op_ts_cumprod(m),
        UnaryOp::TsCumSum => ts::op_ts_cumsum(m),
        UnaryOp::TsFfill => ts::op_ts_ffill(m),
        UnaryOp::TsDenseRank => ts::op_ts_dense_rank(m),
        UnaryOp::TsPctDenseRank => ts::op_ts_pct_dense_rank(m),
    }
}

/// 二元算子分派（`lhs`/`rhs` 已按需完成 universe 掩码过滤）。
fn dispatch_binary(
    op: BinaryOp,
    lhs: &Array2<f64>,
    rhs_series: Option<&Array2<f64>>,
    rhs_const: Option<f64>,
    rhs_window: Option<usize>,
) -> Array2<f64> {
    let series = |v: Option<&Array2<f64>>| -> Array2<f64> {
        v.cloned()
            .unwrap_or_else(|| Array2::<f64>::from_elem(lhs.raw_dim(), f64::NAN))
    };
    let cst = rhs_const.unwrap_or(f64::NAN);
    let win = rhs_window.unwrap_or(0);
    match op {
        BinaryOp::Add => elem::op_add(lhs, &series(rhs_series)),
        BinaryOp::Sub => elem::op_sub(lhs, &series(rhs_series)),
        BinaryOp::Mul => elem::op_mul(lhs, &series(rhs_series)),
        BinaryOp::Div => elem::op_div(lhs, &series(rhs_series)),
        BinaryOp::Max2 => elem::op_max2(lhs, &series(rhs_series)),
        BinaryOp::Min2 => elem::op_min2(lhs, &series(rhs_series)),
        BinaryOp::Hypot => elem::op_hypot(lhs, &series(rhs_series)),
        BinaryOp::Copysign => elem::op_copysign(lhs, &series(rhs_series)),
        BinaryOp::Gt => elem::op_gt(lhs, &series(rhs_series)),
        BinaryOp::Lt => elem::op_lt(lhs, &series(rhs_series)),
        BinaryOp::Ge => elem::op_ge(lhs, &series(rhs_series)),
        BinaryOp::Le => elem::op_le(lhs, &series(rhs_series)),
        BinaryOp::AndCond => elem::op_and_cond(lhs, &series(rhs_series)),
        BinaryOp::OrCond => elem::op_or_cond(lhs, &series(rhs_series)),
        BinaryOp::CsNeutralize => cs::op_cs_neutralize(lhs, &series(rhs_series)),
        BinaryOp::CsBeta => cs::op_cs_beta(lhs, &series(rhs_series)),
        BinaryOp::CsCorr => cs::op_cs_corr(lhs, &series(rhs_series)),
        BinaryOp::CsCov => cs::op_cs_cov(lhs, &series(rhs_series)),
        BinaryOp::Barra5ZScore => cs::op_barra5_zscore(lhs, &series(rhs_series)),
        BinaryOp::TsDw => ts::op_ts_dw(lhs, &series(rhs_series)),
        BinaryOp::GroupDemean => group::op_group_demean(lhs, &series(rhs_series)),
        BinaryOp::GroupRank => group::op_group_rank(lhs, &series(rhs_series)),
        BinaryOp::GroupZscore => group::op_group_zscore(lhs, &series(rhs_series)),
        BinaryOp::Addc => elem::op_addc(lhs, cst),
        BinaryOp::Subc => elem::op_subc(lhs, cst),
        BinaryOp::Mulc => elem::op_mulc(lhs, cst),
        BinaryOp::Divc => elem::op_divc(lhs, cst),
        BinaryOp::Spow => elem::op_spow(lhs, cst),
        BinaryOp::Clip => elem::op_clip(lhs, cst),
        BinaryOp::GtConst => elem::op_gt_const(lhs, cst),
        BinaryOp::LtConst => elem::op_lt_const(lhs, cst),
        BinaryOp::FillNa => elem::op_fillna(lhs, cst),
        BinaryOp::RoundDec => elem::op_round_decimals(lhs, cst),
        BinaryOp::PercLong => elem::op_perc_long(lhs, cst),
        BinaryOp::CsQuantile => cs::op_cs_quantile(lhs, cst),
        BinaryOp::CsGroup => cs::op_cs_group(lhs, cst),
        BinaryOp::WinsorExpr => postprocess::pp_winsor(lhs, cst),
        BinaryOp::MadWinsor => postprocess::pp_mad(lhs, cst),
        BinaryOp::Delay => ts::op_delay(lhs, win),
        BinaryOp::Delta => ts::op_delta(lhs, win),
        BinaryOp::PctChange => ts::op_pct_change(lhs, win),
        BinaryOp::LogReturn => ts::op_log_return(lhs, win),
        BinaryOp::TsSum => ts::op_ts_sum(lhs, win),
        BinaryOp::TsMean => ts::op_ts_mean(lhs, win),
        BinaryOp::TsStd => ts::op_ts_std(lhs, win),
        BinaryOp::TsVar => ts::op_ts_var(lhs, win),
        BinaryOp::TsMin => ts::op_ts_min(lhs, win),
        BinaryOp::TsMax => ts::op_ts_max(lhs, win),
        BinaryOp::TsArgmin => ts::op_ts_argmin(lhs, win),
        BinaryOp::TsArgmax => ts::op_ts_argmax(lhs, win),
        BinaryOp::TsRank => ts::op_ts_rank(lhs, win),
        BinaryOp::TsZscore => ts::op_ts_zscore(lhs, win),
        BinaryOp::TsSkew => ts::op_ts_skew(lhs, win),
        BinaryOp::TsKurt => ts::op_ts_kurt(lhs, win),
        BinaryOp::TsEma => ts::op_ts_ema(lhs, win),
        BinaryOp::TsDecayLinear => ts::op_ts_decay_linear(lhs, win),
        BinaryOp::TsMad => ts::op_ts_mad(lhs, win),
        BinaryOp::TsRoll => ts::op_ts_roll(lhs, win),
        BinaryOp::TsProd => ts::op_ts_prod(lhs, win),
        BinaryOp::TsMedian => ts::op_ts_median(lhs, win),
        BinaryOp::TsNoNanNum => ts::op_ts_nonan_num(lhs, win),
        BinaryOp::TsPctRank => ts::op_ts_pct_rank(lhs, win),
        BinaryOp::TsGroup => ts::op_ts_group(lhs, win),
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
        ASTNode::UnaryCall { op, arg } => eval_nonleaf_cached(node, cache, || {
            let arg_mat = eval_ast(arg, panel, cache)?;
            let masked = if op.is_nan_probe() || op.category() != "cs_unary" {
                Cow::Borrowed(&*arg_mat)
            } else {
                mask_to_universe(&arg_mat, &panel.universe_mask)
            };
            Ok(dispatch_unary(*op, &masked))
        }),
        ASTNode::BinaryCall { op, lhs, rhs } => eval_nonleaf_cached(node, cache, || {
            let lhs_mat = eval_ast(lhs, panel, cache)?;
            let (_t0, t1, _) = op.signature();
            let need_rhs_series = t1 == crate::expr::types::DataType::Series;
            let is_logic = matches!(op, BinaryOp::AndCond | BinaryOp::OrCond);
            let rhs_mat_opt = if need_rhs_series || is_logic {
                Some(eval_ast(rhs, panel, cache)?)
            } else {
                None
            };

            let mask_both = matches!(
                op,
                BinaryOp::CsNeutralize
                    | BinaryOp::GroupDemean
                    | BinaryOp::GroupRank
                    | BinaryOp::GroupZscore
                    | BinaryOp::CsBeta
                    | BinaryOp::CsCorr
                    | BinaryOp::CsCov
                    | BinaryOp::Barra5ZScore
            );
            let mask_lhs_only = matches!(
                op,
                BinaryOp::PercLong
                    | BinaryOp::CsQuantile
                    | BinaryOp::CsGroup
                    | BinaryOp::WinsorExpr
                    | BinaryOp::MadWinsor
            );

            let m_lhs: Cow<'_, Array2<f64>> = if mask_both || mask_lhs_only {
                mask_to_universe(&lhs_mat, &panel.universe_mask)
            } else {
                Cow::Borrowed(&*lhs_mat)
            };
            let m_rhs: Option<Cow<'_, Array2<f64>>> = rhs_mat_opt.as_ref().map(|r| {
                if mask_both {
                    mask_to_universe(r, &panel.universe_mask)
                } else {
                    Cow::Borrowed(&**r)
                }
            });

            let rhs_const = match t1 {
                crate::expr::types::DataType::Constant => Some(extract_constant(rhs)?),
                _ => None,
            };
            let rhs_window = match t1 {
                crate::expr::types::DataType::Window => Some(extract_window(rhs)?),
                _ => None,
            };

            Ok(dispatch_binary(
                *op,
                &m_lhs,
                m_rhs.as_deref(),
                rhs_const,
                rhs_window,
            ))
        }),
        ASTNode::TernaryCall {
            op,
            cond,
            then_branch,
            else_branch,
        } => eval_nonleaf_cached(node, cache, || {
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
                TernaryOp::TsQuantile => {
                    let x_mat = eval_ast(cond, panel, cache)?;
                    let q = extract_constant(then_branch)?;
                    let w = extract_window(else_branch)?;
                    ts::op_ts_quantile(&x_mat, q, w)
                }
            };
            Ok(out)
        }),
        ASTNode::NaryCall { op, args } => eval_nonleaf_cached(node, cache, || {
            let has_intercept =
                matches!(args.last().map(|a| a.as_ref()), Some(ASTNode::Constant(_)));
            let positional = if has_intercept {
                args.len() - 1
            } else {
                args.len()
            };
            if positional < op.min_args() {
                return Err(EvalError::InvalidNaryArgs {
                    op: op.name().to_string(),
                    found: positional,
                });
            }
            let intercept = if has_intercept {
                match args.last().map(|a| a.as_ref()) {
                    Some(ASTNode::Constant(c)) => multi::intercept_flag(*c),
                    _ => false,
                }
            } else {
                false
            };

            let y_mat = eval_ast(&args[0], panel, cache)?;
            let mut cursor = 1usize;
            let weights_mat = if op.is_weighted() {
                let w = eval_ast(&args[cursor], panel, cache)?;
                cursor += 1;
                Some(w)
            } else {
                None
            };
            let mut exp_mats: Vec<Arc<Array2<f64>>> = Vec::with_capacity(positional - cursor);
            for a in args.iter().take(positional).skip(cursor) {
                exp_mats.push(eval_ast(a, panel, cache)?);
            }

            let is_cs = !op.is_time_series();
            let m_y: Cow<'_, Array2<f64>> = if is_cs {
                mask_to_universe(&y_mat, &panel.universe_mask)
            } else {
                Cow::Borrowed(&*y_mat)
            };
            let m_w: Option<Cow<'_, Array2<f64>>> = weights_mat.as_ref().map(|w| {
                if is_cs {
                    mask_to_universe(w, &panel.universe_mask)
                } else {
                    Cow::Borrowed(&**w)
                }
            });
            let m_exps: Vec<Cow<'_, Array2<f64>>> = exp_mats
                .iter()
                .map(|m| {
                    if is_cs {
                        mask_to_universe(m, &panel.universe_mask)
                    } else {
                        Cow::Borrowed(&**m)
                    }
                })
                .collect();
            let exp_refs: Vec<&Array2<f64>> = m_exps.iter().map(|c| c.as_ref()).collect();

            Ok(multi::op_multi(
                *op,
                &m_y,
                &exp_refs,
                m_w.as_deref(),
                intercept,
            ))
        }),
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

/// 递归收集语法树中所有非叶子求值节点（按深度分层，基于规范化表达式字符串精确去重防碰撞）。
fn record_dag_node<'a>(node: &'a ASTNode, levels: &mut Vec<AHashMap<String, &'a ASTNode>>) {
    let d = node.depth();
    if levels.len() <= d {
        levels.resize_with(d + 1, AHashMap::new);
    }
    levels[d].entry(node.to_canon_string()).or_insert(node);
}

fn collect_dag_nodes_by_depth<'a>(
    node: &'a ASTNode,
    levels: &mut Vec<AHashMap<String, &'a ASTNode>>,
) {
    match node {
        ASTNode::Field(_) | ASTNode::Constant(_) | ASTNode::Window(_) => {}
        ASTNode::UnaryCall { arg, .. } => {
            collect_dag_nodes_by_depth(arg, levels);
            record_dag_node(node, levels);
        }
        ASTNode::BinaryCall { lhs, rhs, .. } => {
            collect_dag_nodes_by_depth(lhs, levels);
            collect_dag_nodes_by_depth(rhs, levels);
            record_dag_node(node, levels);
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
            record_dag_node(node, levels);
        }
        ASTNode::NaryCall { args, .. } => {
            for a in args {
                collect_dag_nodes_by_depth(a, levels);
            }
            record_dag_node(node, levels);
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
            let mut levels: Vec<AHashMap<String, &ASTNode>> = Vec::with_capacity(16);
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
