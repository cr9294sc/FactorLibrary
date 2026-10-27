//! DSL 抽象语法树（AST）节点定义与确定性指纹计算 (expr/ast.rs)。

use crate::ops::{BinaryOp, TernaryOp, UnaryOp};
use ahash::RandomState;
use std::fmt;

/// 固定种子的确定性 AHash 状态机，确保跨进程、跨线程生成的 `canon_hash` 100% 恒定。
const DETERMINISTIC_HASHER: RandomState = RandomState::with_seeds(
    0x517cc1b727220a95,
    0x9e3779b97f4a7c15,
    0xbf58476d1ce4e5b9,
    0x94d049bb133111eb,
);

/// 对规范化字符串计算 64 位确定性 `ahash` 指纹。
#[inline]
pub fn hash_canon_str(s: &str) -> u64 {
    DETERMINISTIC_HASHER.hash_one(s)
}

/// 将浮点常数格式化为规范文本表示（消除 `-0.0` 并保留紧凑可逆精度）。
pub fn format_constant(c: f64) -> String {
    let v = if c == 0.0 { 0.0 } else { c };
    if v.fract() == 0.0 && v.abs() < 1e15 {
        format!("{v:.1}")
    } else {
        // 统一四舍五入至 10 位有效小数，防止浮点微扰导致哈希漂移
        let rounded = (v * 1e10).round() / 1e10;
        let s = format!("{rounded}");
        if s.contains('.') || s.contains('e') || s.contains('E') {
            s
        } else {
            format!("{s}.0")
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum ASTNode {
    /// 终端特征字段 (如 "close", "volume")
    Field(String),
    /// 浮点常数标量 (如 0.5, 2.0, -1.0)
    Constant(f64),
    /// 离散窗口参数 (如 5, 10, 20)
    Window(usize),
    /// 一元算子调用
    UnaryCall { op: UnaryOp, arg: Box<ASTNode> },
    /// 二元算子调用
    BinaryCall {
        op: BinaryOp,
        lhs: Box<ASTNode>,
        rhs: Box<ASTNode>,
    },
    /// 三元算子调用 (`where` 或 `ts_corr`/`ts_cov`/`ts_beta`/`ts_resid`)
    TernaryCall {
        op: TernaryOp,
        cond: Box<ASTNode>,
        then_branch: Box<ASTNode>,
        else_branch: Box<ASTNode>,
    },
}

impl ASTNode {
    /// 统计语法树总节点数（用于奥卡姆剃刀复杂度惩罚）。
    pub fn node_count(&self) -> usize {
        match self {
            ASTNode::Field(_) | ASTNode::Constant(_) | ASTNode::Window(_) => 1,
            ASTNode::UnaryCall { arg, .. } => 1 + arg.node_count(),
            ASTNode::BinaryCall { lhs, rhs, .. } => 1 + lhs.node_count() + rhs.node_count(),
            ASTNode::TernaryCall {
                cond,
                then_branch,
                else_branch,
                ..
            } => 1 + cond.node_count() + then_branch.node_count() + else_branch.node_count(),
        }
    }

    /// 计算语法树最大深度（叶子节点深度为 1）。
    pub fn depth(&self) -> usize {
        match self {
            ASTNode::Field(_) | ASTNode::Constant(_) | ASTNode::Window(_) => 1,
            ASTNode::UnaryCall { arg, .. } => 1 + arg.depth(),
            ASTNode::BinaryCall { lhs, rhs, .. } => 1 + lhs.depth().max(rhs.depth()),
            ASTNode::TernaryCall {
                cond,
                then_branch,
                else_branch,
                ..
            } => {
                1 + cond
                    .depth()
                    .max(then_branch.depth())
                    .max(else_branch.depth())
            }
        }
    }

    /// 序列化为紧凑规范化前缀表达式字符串。
    pub fn to_canon_string(&self) -> String {
        let mut buf = String::with_capacity(64);
        self.write_canon(&mut buf);
        buf
    }

    fn write_canon(&self, out: &mut String) {
        use std::fmt::Write;
        match self {
            ASTNode::Field(name) => out.push_str(name),
            ASTNode::Constant(c) => out.push_str(&format_constant(*c)),
            ASTNode::Window(w) => {
                let _ = write!(out, "{w}");
            }
            ASTNode::UnaryCall { op, arg } => {
                out.push_str(op.name());
                out.push('(');
                arg.write_canon(out);
                out.push(')');
            }
            ASTNode::BinaryCall { op, lhs, rhs } => {
                out.push_str(op.name());
                out.push('(');
                lhs.write_canon(out);
                out.push_str(", ");
                rhs.write_canon(out);
                out.push(')');
            }
            ASTNode::TernaryCall {
                op,
                cond,
                then_branch,
                else_branch,
            } => {
                out.push_str(op.name());
                out.push('(');
                cond.write_canon(out);
                out.push_str(", ");
                then_branch.write_canon(out);
                out.push_str(", ");
                else_branch.write_canon(out);
                out.push(')');
            }
        }
    }

    /// 计算当前节点的 64 位确定性 `ahash` 指纹。
    pub fn canon_hash(&self) -> u64 {
        hash_canon_str(&self.to_canon_string())
    }

    /// 计算十六进制字符串格式的 64 位唯一指纹。
    pub fn canon_hash_hex(&self) -> String {
        format!("{:016x}", self.canon_hash())
    }

    /// 递归推导表达式所需的总预热期（`expression_lookback`）、各终端特征所需窗口及调用的算子列表。
    pub fn analyze_lookback(&self) -> (usize, Vec<(String, usize)>, Vec<String>) {
        match self {
            ASTNode::Field(name) => (1, vec![(name.clone(), 1)], Vec::new()),
            ASTNode::Constant(_) | ASTNode::Window(_) => (1, Vec::new(), Vec::new()),
            ASTNode::UnaryCall { op, arg } => {
                let (lb, vars, mut ops) = arg.analyze_lookback();
                let op_name = op.name().to_string();
                if !ops.contains(&op_name) {
                    ops.push(op_name);
                }
                (lb, vars, ops)
            }
            ASTNode::BinaryCall { op, lhs, rhs } => {
                let add_w = match (&op, rhs.as_ref()) {
                    (
                        BinaryOp::Delay
                        | BinaryOp::Delta
                        | BinaryOp::PctChange
                        | BinaryOp::LogReturn,
                        ASTNode::Window(w),
                    ) => *w,
                    (op_val, ASTNode::Window(w)) if op_val.category() == "ts_unary" => {
                        w.saturating_sub(1)
                    }
                    _ => 0,
                };
                let (lb_l, vars_l, mut ops_l) = lhs.analyze_lookback();
                let (lb_r, vars_r, ops_r) = rhs.analyze_lookback();
                let max_child_lb = lb_l.max(lb_r);
                let mut merged_vars: Vec<(String, usize)> = Vec::new();
                for (k, v) in vars_l.into_iter().chain(vars_r.into_iter()) {
                    let req = v + add_w;
                    if let Some(existing) = merged_vars.iter_mut().find(|p| p.0 == k) {
                        if req > existing.1 {
                            existing.1 = req;
                        }
                    } else {
                        merged_vars.push((k, req));
                    }
                }
                for f in ops_r {
                    if !ops_l.contains(&f) {
                        ops_l.push(f);
                    }
                }
                let op_name = op.name().to_string();
                if !ops_l.contains(&op_name) {
                    ops_l.push(op_name);
                }
                (max_child_lb + add_w, merged_vars, ops_l)
            }
            ASTNode::TernaryCall {
                op,
                cond,
                then_branch,
                else_branch,
            } => {
                let add_w = match (&op, else_branch.as_ref()) {
                    (op_val, ASTNode::Window(w)) if op_val.category() == "ts_binary" => {
                        w.saturating_sub(1)
                    }
                    _ => 0,
                };
                let (lb_0, vars_0, mut ops_0) = cond.analyze_lookback();
                let (lb_1, vars_1, ops_1) = then_branch.analyze_lookback();
                let (lb_2, vars_2, ops_2) = else_branch.analyze_lookback();
                let max_child_lb = lb_0.max(lb_1).max(lb_2);
                let mut merged_vars: Vec<(String, usize)> = Vec::new();
                for (k, v) in vars_0
                    .into_iter()
                    .chain(vars_1.into_iter())
                    .chain(vars_2.into_iter())
                {
                    let req = v + add_w;
                    if let Some(existing) = merged_vars.iter_mut().find(|p| p.0 == k) {
                        if req > existing.1 {
                            existing.1 = req;
                        }
                    } else {
                        merged_vars.push((k, req));
                    }
                }
                for f in ops_1.into_iter().chain(ops_2.into_iter()) {
                    if !ops_0.contains(&f) {
                        ops_0.push(f);
                    }
                }
                let op_name = op.name().to_string();
                if !ops_0.contains(&op_name) {
                    ops_0.push(op_name);
                }
                (max_child_lb + add_w, merged_vars, ops_0)
            }
        }
    }
}

impl fmt::Display for ASTNode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.to_canon_string())
    }
}
