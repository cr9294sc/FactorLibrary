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
}

impl fmt::Display for ASTNode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.to_canon_string())
    }
}
