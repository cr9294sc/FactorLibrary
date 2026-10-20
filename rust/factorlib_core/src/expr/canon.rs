//! AST 代数等价规范化（Canonicalization）与退化结构阻断引擎 (expr/canon.rs)。
//!
//! 严格实现白皮书 7.3 节全部 4 类规则：
//! 1. 交换律排序（Commutative Sorting）：`add`, `mul`, `max2`, `min2`, `hypot`, `ts_corr`, `ts_cov`, `and_cond`, `or_cond`
//! 2. 幂等性与代数恒等化简（Idempotency & Identity Folding）：
//!    `cs_rank(cs_rank(x)) => cs_rank(x)`, `abs(abs(x)) => abs(x)`, `neg(neg(x)) => x`, `delay(delay(x, d1), d2) => delay(x, d1+d2)` 等
//! 3. 单调不变性压缩（Monotonic Invariance Compression）：
//!    剥离 `cs_rank` 内外包裹的严格单调递增无界/保序变换（如 `slog`, `ssqrt`）
//! 4. 退化结构阻断（Degenerate Pruning）：
//!    检测 `sub(x, x)`, `div(x, x)`, `ts_corr(x, x, w)`, `mulc(x, 0.0)` 等恒定常数退化树并标记阻断。

use crate::expr::ast::ASTNode;
use crate::nan::EPS;
use crate::ops::{BinaryOp, TernaryOp, UnaryOp};

#[derive(Debug, Clone)]
pub struct CanonResult {
    pub node: ASTNode,
    pub canon_expr: String,
    pub canon_hash: u64,
    pub canon_hash_hex: String,
    pub is_degenerate: bool,
    pub degenerate_reason: Option<String>,
    pub n_nodes: usize,
    pub depth: usize,
}

/// 对 AST 执行完整代数规范化化简与退化检测（迭代至不动点）。
pub fn canonicalize(root: &ASTNode) -> CanonResult {
    let mut cur = root.clone();
    let mut degen_reason = None;
    for _ in 0..8 {
        let (next, reason) = canonicalize_rec(&cur, true);
        if reason.is_some() {
            cur = next;
            degen_reason = reason;
            break;
        }
        if next == cur {
            break;
        }
        cur = next;
    }
    let canon_expr = cur.to_canon_string();
    let canon_hash = cur.canon_hash();
    let canon_hash_hex = format!("{:016x}", canon_hash);
    let n_nodes = cur.node_count();
    let depth = cur.depth();
    CanonResult {
        node: cur,
        canon_expr,
        canon_hash,
        canon_hash_hex,
        is_degenerate: degen_reason.is_some(),
        degenerate_reason: degen_reason,
        n_nodes,
        depth,
    }
}

/// 仅执行数值完全等价（Bit-Exact / Value-Preserving）的规范化（用于精确求值缓存键）。
pub fn canonicalize_value_preserving(root: &ASTNode) -> ASTNode {
    let mut cur = root.clone();
    for _ in 0..8 {
        let (next, reason) = canonicalize_rec(&cur, false);
        if reason.is_some() || next == cur {
            cur = next;
            break;
        }
        cur = next;
    }
    cur
}

fn order_nodes(a: ASTNode, b: ASTNode) -> (ASTNode, ASTNode) {
    let ha = a.canon_hash();
    let hb = b.canon_hash();
    if ha < hb {
        (a, b)
    } else if ha > hb {
        (b, a)
    } else {
        let sa = a.to_canon_string();
        let sb = b.to_canon_string();
        if sa <= sb {
            (a, b)
        } else {
            (b, a)
        }
    }
}

fn canonicalize_rec(
    node: &ASTNode,
    compress_outer_rank: bool,
) -> (ASTNode, Option<String>) {
    match node {
        ASTNode::Field(_) | ASTNode::Constant(_) | ASTNode::Window(_) => (node.clone(), None),

        ASTNode::UnaryCall { op, arg } => {
            let (c_arg, arg_degen) = canonicalize_rec(arg, compress_outer_rank);
            if arg_degen.is_some() {
                return (
                    ASTNode::UnaryCall {
                        op: *op,
                        arg: Box::new(c_arg),
                    },
                    arg_degen,
                );
            }

            // 规则 2 & 3：一元幂等性与单调性化简
            match (op, &c_arg) {
                // neg(neg(x)) => x
                (UnaryOp::Neg, ASTNode::UnaryCall { op: UnaryOp::Neg, arg: inner }) => {
                    return (*inner.clone(), None);
                }
                // inv(inv(x)) => x
                (UnaryOp::Inv, ASTNode::UnaryCall { op: UnaryOp::Inv, arg: inner }) => {
                    return (*inner.clone(), None);
                }
                // not_cond(not_cond(x)) => x
                (UnaryOp::NotCond, ASTNode::UnaryCall { op: UnaryOp::NotCond, arg: inner }) => {
                    return (*inner.clone(), None);
                }
                // abs(abs(x)) => abs(x), abs(neg(x)) => abs(x)
                (
                    UnaryOp::Abs,
                    ASTNode::UnaryCall {
                        op: UnaryOp::Abs | UnaryOp::Neg,
                        arg: inner,
                    },
                ) => {
                    return (
                        ASTNode::UnaryCall {
                            op: UnaryOp::Abs,
                            arg: inner.clone(),
                        },
                        None,
                    );
                }
                // square(neg(x)) => square(x), square(abs(x)) => square(x)
                (
                    UnaryOp::Square,
                    ASTNode::UnaryCall {
                        op: UnaryOp::Neg | UnaryOp::Abs,
                        arg: inner,
                    },
                ) => {
                    return (
                        ASTNode::UnaryCall {
                            op: UnaryOp::Square,
                            arg: inner.clone(),
                        },
                        None,
                    );
                }
                // sign(sign(x)) => sign(x), sign(slog(x)) => sign(x), sign(ssqrt(x)) => sign(x)
                (
                    UnaryOp::Sign,
                    ASTNode::UnaryCall {
                        op: UnaryOp::Sign | UnaryOp::Slog | UnaryOp::Ssqrt | UnaryOp::Cube,
                        arg: inner,
                    },
                ) => {
                    return (
                        ASTNode::UnaryCall {
                            op: UnaryOp::Sign,
                            arg: inner.clone(),
                        },
                        None,
                    );
                }
                // relu(relu(x)) => relu(x)
                (UnaryOp::Relu, ASTNode::UnaryCall { op: UnaryOp::Relu, arg: inner }) => {
                    return (
                        ASTNode::UnaryCall {
                            op: UnaryOp::Relu,
                            arg: inner.clone(),
                        },
                        None,
                    );
                }
                // cs_demean(cs_demean(x)) => cs_demean(x)
                (UnaryOp::CsDemean, ASTNode::UnaryCall { op: UnaryOp::CsDemean, arg: inner }) => {
                    return (
                        ASTNode::UnaryCall {
                            op: UnaryOp::CsDemean,
                            arg: inner.clone(),
                        },
                        None,
                    );
                }
                // cs_zscore(cs_zscore(x)) => cs_zscore(x), cs_zscore(cs_demean(x)) => cs_zscore(x)
                (
                    UnaryOp::CsZscore,
                    ASTNode::UnaryCall {
                        op: UnaryOp::CsZscore | UnaryOp::CsDemean,
                        arg: inner,
                    },
                ) => {
                    return (
                        ASTNode::UnaryCall {
                            op: UnaryOp::CsZscore,
                            arg: inner.clone(),
                        },
                        None,
                    );
                }
                // cs_scale(cs_scale(x)) => cs_scale(x)
                (UnaryOp::CsScale, ASTNode::UnaryCall { op: UnaryOp::CsScale, arg: inner }) => {
                    return (
                        ASTNode::UnaryCall {
                            op: UnaryOp::CsScale,
                            arg: inner.clone(),
                        },
                        None,
                    );
                }
                // cs_rank(cs_rank(x)) => cs_rank(x) 及内部保序严格单调变换压缩
                (UnaryOp::CsRank, ASTNode::UnaryCall { op: inner_op, arg: inner })
                    if *inner_op == UnaryOp::CsRank
                        || *inner_op == UnaryOp::CsDemean
                        || *inner_op == UnaryOp::CsZscore
                        || *inner_op == UnaryOp::CsScale
                        || inner_op.is_strictly_increasing() =>
                {
                    return (
                        ASTNode::UnaryCall {
                            op: UnaryOp::CsRank,
                            arg: inner.clone(),
                        },
                        None,
                    );
                }
                // cs_rank(addc(x, c)) / cs_rank(subc(x, c)) => cs_rank(x)
                (
                    UnaryOp::CsRank,
                    ASTNode::BinaryCall {
                        op: BinaryOp::Addc | BinaryOp::Subc,
                        lhs: inner,
                        ..
                    },
                ) => {
                    return (
                        ASTNode::UnaryCall {
                            op: UnaryOp::CsRank,
                            arg: inner.clone(),
                        },
                        None,
                    );
                }
                // cs_rank(mulc(x, c)) (c > 0) => cs_rank(x)
                (
                    UnaryOp::CsRank,
                    ASTNode::BinaryCall {
                        op: BinaryOp::Mulc | BinaryOp::Divc,
                        lhs: inner,
                        rhs,
                    },
                ) => {
                    if let ASTNode::Constant(c) = rhs.as_ref() {
                        if *c > EPS {
                            return (
                                ASTNode::UnaryCall {
                                    op: UnaryOp::CsRank,
                                    arg: inner.clone(),
                                },
                                None,
                            );
                        }
                    }
                }
                // 规则 4：在 cs_rank 外层包裹严格单调递增函数（如 slog(cs_rank(x)) => cs_rank(x)）
                (
                    outer_op,
                    ASTNode::UnaryCall {
                        op: UnaryOp::CsRank,
                        arg: inner,
                    },
                ) if compress_outer_rank && outer_op.is_strictly_increasing() => {
                    return (
                        ASTNode::UnaryCall {
                            op: UnaryOp::CsRank,
                            arg: inner.clone(),
                        },
                        None,
                    );
                }
                _ => {}
            }

            (
                ASTNode::UnaryCall {
                    op: *op,
                    arg: Box::new(c_arg),
                },
                None,
            )
        }

        ASTNode::BinaryCall { op, lhs, rhs } => {
            let (mut c_lhs, l_degen) = canonicalize_rec(lhs, compress_outer_rank);
            let (mut c_rhs, r_degen) = canonicalize_rec(rhs, compress_outer_rank);

            if let Some(reason) = l_degen.or(r_degen) {
                return (
                    ASTNode::BinaryCall {
                        op: *op,
                        lhs: Box::new(c_lhs),
                        rhs: Box::new(c_rhs),
                    },
                    Some(reason),
                );
            }

            // 规则 1：交换律排序
            if op.is_commutative() {
                let (first, second) = order_nodes(c_lhs, c_rhs);
                c_lhs = first;
                c_rhs = second;
            }

            // 规则 3：退化结构阻断
            let same_operands =
                c_lhs.canon_hash() == c_rhs.canon_hash() && c_lhs.to_canon_string() == c_rhs.to_canon_string();

            if same_operands {
                match op {
                    BinaryOp::Sub => {
                        return (
                            ASTNode::BinaryCall {
                                op: *op,
                                lhs: Box::new(c_lhs),
                                rhs: Box::new(c_rhs),
                            },
                            Some("sub(x, x) 退化为常数 0".to_string()),
                        );
                    }
                    BinaryOp::Div => {
                        return (
                            ASTNode::BinaryCall {
                                op: *op,
                                lhs: Box::new(c_lhs),
                                rhs: Box::new(c_rhs),
                            },
                            Some("div(x, x) 退化为常数 1".to_string()),
                        );
                    }
                    BinaryOp::CsNeutralize => {
                        return (
                            ASTNode::BinaryCall {
                                op: *op,
                                lhs: Box::new(c_lhs),
                                rhs: Box::new(c_rhs),
                            },
                            Some("cs_neutralize(x, x) 退化为常数 0".to_string()),
                        );
                    }
                    BinaryOp::Gt | BinaryOp::Lt | BinaryOp::Ge | BinaryOp::Le => {
                        return (
                            ASTNode::BinaryCall {
                                op: *op,
                                lhs: Box::new(c_lhs),
                                rhs: Box::new(c_rhs),
                            },
                            Some(format!("{}(x, x) 退化为恒定布尔值", op.name())),
                        );
                    }
                    BinaryOp::Max2 | BinaryOp::Min2 | BinaryOp::AndCond | BinaryOp::OrCond => {
                        return (c_lhs, None);
                    }
                    _ => {}
                }
            }

            // 规则 2：常数与时序算子化简
            match (op, &c_lhs, &c_rhs) {
                // delay(delay(x, d1), d2) => delay(x, d1 + d2)
                (
                    BinaryOp::Delay,
                    ASTNode::BinaryCall {
                        op: BinaryOp::Delay,
                        lhs: inner_x,
                        rhs: inner_w1,
                    },
                    ASTNode::Window(w2),
                ) => {
                    if let ASTNode::Window(w1) = inner_w1.as_ref() {
                        return (
                            ASTNode::BinaryCall {
                                op: BinaryOp::Delay,
                                lhs: inner_x.clone(),
                                rhs: Box::new(ASTNode::Window(w1 + w2)),
                            },
                            None,
                        );
                    }
                }
                // addc(x, 0.0) / subc(x, 0.0) => x
                (BinaryOp::Addc | BinaryOp::Subc, _, ASTNode::Constant(c)) if c.abs() < EPS => {
                    return (c_lhs, None);
                }
                // subc(x, c) => addc(x, -c) 统一规范化为 addc
                (BinaryOp::Subc, _, ASTNode::Constant(c)) => {
                    return canonicalize_rec(
                        &ASTNode::BinaryCall {
                            op: BinaryOp::Addc,
                            lhs: Box::new(c_lhs),
                            rhs: Box::new(ASTNode::Constant(-c)),
                        },
                        compress_outer_rank,
                    );
                }
                // addc(addc(x, c1), c2) => addc(x, c1 + c2)
                (
                    BinaryOp::Addc,
                    ASTNode::BinaryCall {
                        op: BinaryOp::Addc,
                        lhs: inner_x,
                        rhs: inner_c1,
                    },
                    ASTNode::Constant(c2),
                ) => {
                    if let ASTNode::Constant(c1) = inner_c1.as_ref() {
                        let sum_c = c1 + c2;
                        if sum_c.abs() < EPS {
                            return (*inner_x.clone(), None);
                        }
                        return (
                            ASTNode::BinaryCall {
                                op: BinaryOp::Addc,
                                lhs: inner_x.clone(),
                                rhs: Box::new(ASTNode::Constant(sum_c)),
                            },
                            None,
                        );
                    }
                }
                // mulc(x, 0.0) / spow(x, 0.0) / clip(x, 0.0) => 退化常数
                (
                    BinaryOp::Mulc | BinaryOp::Spow | BinaryOp::Clip,
                    _,
                    ASTNode::Constant(c),
                ) if c.abs() < EPS => {
                    let op_name = op.name();
                    return (
                        ASTNode::BinaryCall {
                            op: *op,
                            lhs: Box::new(c_lhs),
                            rhs: Box::new(c_rhs),
                        },
                        Some(format!("{op_name}(x, 0.0) 退化为常数")),
                    );
                }
                // mulc(x, 1.0) / divc(x, 1.0) / spow(x, 1.0) => x
                (
                    BinaryOp::Mulc | BinaryOp::Divc | BinaryOp::Spow,
                    _,
                    ASTNode::Constant(c),
                ) if (*c - 1.0).abs() < EPS => {
                    return (c_lhs, None);
                }
                // mulc(x, -1.0) / divc(x, -1.0) => neg(x)
                (BinaryOp::Mulc | BinaryOp::Divc, _, ASTNode::Constant(c))
                    if (*c + 1.0).abs() < EPS =>
                {
                    return canonicalize_rec(
                        &ASTNode::UnaryCall {
                            op: UnaryOp::Neg,
                            arg: Box::new(c_lhs),
                        },
                        compress_outer_rank,
                    );
                }
                // ts_*(Constant, w) => 退化
                (_, ASTNode::Constant(_), ASTNode::Window(_)) => {
                    return (
                        ASTNode::BinaryCall {
                            op: *op,
                            lhs: Box::new(c_lhs),
                            rhs: Box::new(c_rhs),
                        },
                        Some(format!("{}(常数, w) 退化为常数", op.name())),
                    );
                }
                _ => {}
            }

            (
                ASTNode::BinaryCall {
                    op: *op,
                    lhs: Box::new(c_lhs),
                    rhs: Box::new(c_rhs),
                },
                None,
            )
        }

        ASTNode::TernaryCall {
            op,
            cond,
            then_branch,
            else_branch,
        } => {
            let (mut c0, d0) = canonicalize_rec(cond, compress_outer_rank);
            let (mut c1, d1) = canonicalize_rec(then_branch, compress_outer_rank);
            let (c2, d2) = canonicalize_rec(else_branch, compress_outer_rank);

            if let Some(reason) = d0.or(d1).or(d2) {
                return (
                    ASTNode::TernaryCall {
                        op: *op,
                        cond: Box::new(c0),
                        then_branch: Box::new(c1),
                        else_branch: Box::new(c2),
                    },
                    Some(reason),
                );
            }

            if op.is_first_two_commutative() {
                let (first, second) = order_nodes(c0, c1);
                c0 = first;
                c1 = second;
            }

            match op {
                TernaryOp::Where => {
                    // where(cond, x, x) => x
                    if c1.canon_hash() == c2.canon_hash()
                        && c1.to_canon_string() == c2.to_canon_string()
                    {
                        return (c1, None);
                    }
                }
                TernaryOp::TsCorr | TernaryOp::TsResid | TernaryOp::TsBeta => {
                    // ts_corr(x, x, d) / ts_resid(x, x, d) / ts_beta(x, x, d) => 退化
                    if c0.canon_hash() == c1.canon_hash()
                        && c0.to_canon_string() == c1.to_canon_string()
                    {
                        return (
                            ASTNode::TernaryCall {
                                op: *op,
                                cond: Box::new(c0),
                                then_branch: Box::new(c1),
                                else_branch: Box::new(c2),
                            },
                            Some(format!("{}(x, x, w) 退化为常数", op.name())),
                        );
                    }
                }
                TernaryOp::TsCov => {}
            }

            (
                ASTNode::TernaryCall {
                    op: *op,
                    cond: Box::new(c0),
                    then_branch: Box::new(c1),
                    else_branch: Box::new(c2),
                },
                None,
            )
        }
    }
}
