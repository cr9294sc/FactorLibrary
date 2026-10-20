//! DSL 递归下降语法解析器 (expr/parser.rs)。
//!
//! 同时支持：
//! 1. 规范前缀算子调用语法（如 `cs_rank(ts_mean(close, 20))`）
//! 2. 算术与比较中缀语法糖（如 `(close - open) / ts_std(close, 20)`）
//! 3. 词法/语法级未来函数标签拦截（禁止引用 `y_1`, `y_5` 等标签列）

use crate::expr::ast::ASTNode;
use crate::expr::lexer::{tokenize, LexError, Token, TokenKind};
use crate::expr::types::DataType;
use crate::ops::{BinaryOp, TernaryOp, UnaryOp};
use crate::panel::is_reserved_label_name;
use thiserror::Error;

#[derive(Error, Debug, Clone, PartialEq)]
pub enum ParseError {
    #[error(transparent)]
    Lex(#[from] LexError),
    #[error("语法错误 (位置 {pos}): {msg}")]
    Syntax { pos: usize, msg: String },
    #[error("未知算子名称: '{0}'")]
    UnknownOperator(String),
    #[error("算子 '{op}' 参数数量错误: 期望 {expected} 个, 实际传入 {actual} 个")]
    ArityMismatch {
        op: String,
        expected: usize,
        actual: usize,
    },
    #[error("严禁未来函数: 表达式禁止引用预测标签或保留字段 '{0}'")]
    ForbiddenLabelAccess(String),
}

struct Parser {
    tokens: Vec<Token>,
    cursor: usize,
}

impl Parser {
    fn new(tokens: Vec<Token>) -> Self {
        Self { tokens, cursor: 0 }
    }

    #[inline]
    fn peek(&self) -> &Token {
        &self.tokens[self.cursor]
    }

    #[inline]
    fn advance(&mut self) -> Token {
        let tok = self.tokens[self.cursor].clone();
        if self.cursor + 1 < self.tokens.len() {
            self.cursor += 1;
        }
        tok
    }

    fn expect(&mut self, expected: &TokenKind) -> Result<Token, ParseError> {
        let tok = self.peek().clone();
        if &tok.kind == expected {
            Ok(self.advance())
        } else {
            Err(ParseError::Syntax {
                pos: tok.pos,
                msg: format!("期望 {:?}, 实际遇到 {:?}", expected, tok.kind),
            })
        }
    }

    fn parse_expr(&mut self) -> Result<ASTNode, ParseError> {
        self.parse_comparison()
    }

    fn parse_comparison(&mut self) -> Result<ASTNode, ParseError> {
        let mut node = self.parse_additive()?;
        while let TokenKind::Gt | TokenKind::Lt | TokenKind::Ge | TokenKind::Le = self.peek().kind {
            let op_tok = self.advance();
            let rhs = self.parse_additive()?;
            node = match (&op_tok.kind, rhs) {
                (TokenKind::Gt, ASTNode::Constant(c)) => ASTNode::BinaryCall {
                    op: BinaryOp::GtConst,
                    lhs: Box::new(node),
                    rhs: Box::new(ASTNode::Constant(c)),
                },
                (TokenKind::Gt, ASTNode::Window(w)) => ASTNode::BinaryCall {
                    op: BinaryOp::GtConst,
                    lhs: Box::new(node),
                    rhs: Box::new(ASTNode::Constant(w as f64)),
                },
                (TokenKind::Lt, ASTNode::Constant(c)) => ASTNode::BinaryCall {
                    op: BinaryOp::LtConst,
                    lhs: Box::new(node),
                    rhs: Box::new(ASTNode::Constant(c)),
                },
                (TokenKind::Lt, ASTNode::Window(w)) => ASTNode::BinaryCall {
                    op: BinaryOp::LtConst,
                    lhs: Box::new(node),
                    rhs: Box::new(ASTNode::Constant(w as f64)),
                },
                (TokenKind::Gt, r) => ASTNode::BinaryCall {
                    op: BinaryOp::Gt,
                    lhs: Box::new(node),
                    rhs: Box::new(r),
                },
                (TokenKind::Lt, r) => ASTNode::BinaryCall {
                    op: BinaryOp::Lt,
                    lhs: Box::new(node),
                    rhs: Box::new(r),
                },
                (TokenKind::Ge, r) => ASTNode::BinaryCall {
                    op: BinaryOp::Ge,
                    lhs: Box::new(node),
                    rhs: Box::new(r),
                },
                (TokenKind::Le, r) => ASTNode::BinaryCall {
                    op: BinaryOp::Le,
                    lhs: Box::new(node),
                    rhs: Box::new(r),
                },
                _ => unreachable!(),
            };
        }
        Ok(node)
    }

    fn parse_additive(&mut self) -> Result<ASTNode, ParseError> {
        let mut node = self.parse_multiplicative()?;
        while let TokenKind::Plus | TokenKind::Minus = self.peek().kind {
            let is_plus = self.advance().kind == TokenKind::Plus;
            let rhs = self.parse_multiplicative()?;
            node = match (is_plus, rhs) {
                (true, ASTNode::Constant(c)) => ASTNode::BinaryCall {
                    op: BinaryOp::Addc,
                    lhs: Box::new(node),
                    rhs: Box::new(ASTNode::Constant(c)),
                },
                (true, ASTNode::Window(w)) => ASTNode::BinaryCall {
                    op: BinaryOp::Addc,
                    lhs: Box::new(node),
                    rhs: Box::new(ASTNode::Constant(w as f64)),
                },
                (false, ASTNode::Constant(c)) => ASTNode::BinaryCall {
                    op: BinaryOp::Subc,
                    lhs: Box::new(node),
                    rhs: Box::new(ASTNode::Constant(c)),
                },
                (false, ASTNode::Window(w)) => ASTNode::BinaryCall {
                    op: BinaryOp::Subc,
                    lhs: Box::new(node),
                    rhs: Box::new(ASTNode::Constant(w as f64)),
                },
                (true, r) => ASTNode::BinaryCall {
                    op: BinaryOp::Add,
                    lhs: Box::new(node),
                    rhs: Box::new(r),
                },
                (false, r) => ASTNode::BinaryCall {
                    op: BinaryOp::Sub,
                    lhs: Box::new(node),
                    rhs: Box::new(r),
                },
            };
        }
        Ok(node)
    }

    fn parse_multiplicative(&mut self) -> Result<ASTNode, ParseError> {
        let mut node = self.parse_unary()?;
        while let TokenKind::Star | TokenKind::Slash = self.peek().kind {
            let is_mul = self.advance().kind == TokenKind::Star;
            let rhs = self.parse_unary()?;
            node = match (is_mul, rhs) {
                (true, ASTNode::Constant(c)) => ASTNode::BinaryCall {
                    op: BinaryOp::Mulc,
                    lhs: Box::new(node),
                    rhs: Box::new(ASTNode::Constant(c)),
                },
                (true, ASTNode::Window(w)) => ASTNode::BinaryCall {
                    op: BinaryOp::Mulc,
                    lhs: Box::new(node),
                    rhs: Box::new(ASTNode::Constant(w as f64)),
                },
                (false, ASTNode::Constant(c)) => ASTNode::BinaryCall {
                    op: BinaryOp::Divc,
                    lhs: Box::new(node),
                    rhs: Box::new(ASTNode::Constant(c)),
                },
                (false, ASTNode::Window(w)) => ASTNode::BinaryCall {
                    op: BinaryOp::Divc,
                    lhs: Box::new(node),
                    rhs: Box::new(ASTNode::Constant(w as f64)),
                },
                (true, r) => ASTNode::BinaryCall {
                    op: BinaryOp::Mul,
                    lhs: Box::new(node),
                    rhs: Box::new(r),
                },
                (false, r) => ASTNode::BinaryCall {
                    op: BinaryOp::Div,
                    lhs: Box::new(node),
                    rhs: Box::new(r),
                },
            };
        }
        Ok(node)
    }

    fn parse_unary(&mut self) -> Result<ASTNode, ParseError> {
        match self.peek().kind {
            TokenKind::Minus => {
                self.advance();
                match self.peek().kind {
                    TokenKind::IntLit(iv) => {
                        self.advance();
                        Ok(ASTNode::Constant(-(iv as f64)))
                    }
                    TokenKind::FloatLit(fv) => {
                        self.advance();
                        Ok(ASTNode::Constant(-fv))
                    }
                    _ => {
                        let inner = self.parse_unary()?;
                        Ok(ASTNode::UnaryCall {
                            op: UnaryOp::Neg,
                            arg: Box::new(inner),
                        })
                    }
                }
            }
            TokenKind::Plus => {
                self.advance();
                self.parse_unary()
            }
            _ => self.parse_primary(),
        }
    }

    fn parse_primary(&mut self) -> Result<ASTNode, ParseError> {
        let tok = self.peek().clone();
        match tok.kind {
            TokenKind::LParen => {
                self.advance();
                let expr = self.parse_expr()?;
                self.expect(&TokenKind::RParen)?;
                Ok(expr)
            }
            TokenKind::IntLit(iv) => {
                self.advance();
                Ok(ASTNode::Window(iv))
            }
            TokenKind::FloatLit(fv) => {
                self.advance();
                Ok(ASTNode::Constant(fv))
            }
            TokenKind::Ident(name) => {
                self.advance();
                if self.peek().kind == TokenKind::LParen {
                    self.advance(); // consume '('
                    let mut args = Vec::with_capacity(3);
                    if self.peek().kind != TokenKind::RParen {
                        loop {
                            args.push(self.parse_expr()?);
                            if self.peek().kind == TokenKind::Comma {
                                self.advance();
                            } else {
                                break;
                            }
                        }
                    }
                    self.expect(&TokenKind::RParen)?;
                    build_call_node(&name, args)
                } else {
                    if is_reserved_label_name(&name) {
                        return Err(ParseError::ForbiddenLabelAccess(name));
                    }
                    Ok(ASTNode::Field(name))
                }
            }
            _ => Err(ParseError::Syntax {
                pos: tok.pos,
                msg: format!("非法的表达式起始 Token: {:?}", tok.kind),
            }),
        }
    }
}

/// 当参数槽位明确声明为 `Constant` 且传入了无小数点的整数字面量 `Window(w)` 时，自动提升为 `Constant(w as f64)`。
#[inline]
fn coerce_for_slot(node: ASTNode, expected: DataType) -> ASTNode {
    if expected == DataType::Constant {
        if let ASTNode::Window(w) = node {
            return ASTNode::Constant(w as f64);
        }
    }
    node
}

fn build_call_node(name: &str, mut args: Vec<ASTNode>) -> Result<ASTNode, ParseError> {
    if let Some(op) = UnaryOp::from_name(name) {
        if args.len() != 1 {
            return Err(ParseError::ArityMismatch {
                op: name.to_string(),
                expected: 1,
                actual: args.len(),
            });
        }
        let (t0, _) = op.signature();
        let a0 = coerce_for_slot(args.remove(0), t0);
        return Ok(ASTNode::UnaryCall {
            op,
            arg: Box::new(a0),
        });
    }

    if let Some(op) = BinaryOp::from_name(name) {
        if args.len() != 2 {
            return Err(ParseError::ArityMismatch {
                op: name.to_string(),
                expected: 2,
                actual: args.len(),
            });
        }
        let (t0, t1, _) = op.signature();
        let a1 = coerce_for_slot(args.pop().unwrap(), t1);
        let a0 = coerce_for_slot(args.pop().unwrap(), t0);
        return Ok(ASTNode::BinaryCall {
            op,
            lhs: Box::new(a0),
            rhs: Box::new(a1),
        });
    }

    if let Some(op) = TernaryOp::from_name(name) {
        if args.len() != 3 {
            return Err(ParseError::ArityMismatch {
                op: name.to_string(),
                expected: 3,
                actual: args.len(),
            });
        }
        let (t0, t1, t2, _) = op.signature();
        let a2 = coerce_for_slot(args.pop().unwrap(), t2);
        let a1 = coerce_for_slot(args.pop().unwrap(), t1);
        let a0 = coerce_for_slot(args.pop().unwrap(), t0);
        return Ok(ASTNode::TernaryCall {
            op,
            cond: Box::new(a0),
            then_branch: Box::new(a1),
            else_branch: Box::new(a2),
        });
    }

    Err(ParseError::UnknownOperator(name.to_string()))
}

/// 将 DSL 表达式字符串解析为 `ASTNode`。
pub fn parse_expr(input: &str) -> Result<ASTNode, ParseError> {
    let tokens = tokenize(input)?;
    let mut parser = Parser::new(tokens);
    let node = parser.parse_expr()?;
    parser.expect(&TokenKind::Eof)?;
    Ok(node)
}
