//! DSL 词法分析器 (expr/lexer.rs)。
//!
//! 支持标准函数式调用（如 `cs_rank(ts_mean(close, 20))`）以及中缀运算符语法糖
//! （如 `(close - open) / ts_std(close, 20)`）。

use thiserror::Error;

#[derive(Error, Debug, Clone, PartialEq)]
pub enum LexError {
    #[error("词法错误 (位置 {pos}): 未知字符 '{ch}'")]
    UnexpectedChar { pos: usize, ch: char },
    #[error("词法错误 (位置 {pos}): 非法数字字面量 '{lexeme}'")]
    InvalidNumber { pos: usize, lexeme: String },
}

#[derive(Debug, Clone, PartialEq)]
pub enum TokenKind {
    Ident(String),
    IntLit(usize),
    FloatLit(f64),
    LParen,
    RParen,
    Comma,
    Plus,
    Minus,
    Star,
    Slash,
    Gt,
    Lt,
    Ge,
    Le,
    Eof,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Token {
    pub kind: TokenKind,
    pub pos: usize,
}

pub fn tokenize(input: &str) -> Result<Vec<Token>, LexError> {
    let chars: Vec<(usize, char)> = input.char_indices().collect();
    let n = chars.len();
    let mut i = 0;
    let mut tokens = Vec::with_capacity(32);

    while i < n {
        let (pos, ch) = chars[i];
        if ch.is_whitespace() {
            i += 1;
            continue;
        }
        match ch {
            '(' => {
                tokens.push(Token {
                    kind: TokenKind::LParen,
                    pos,
                });
                i += 1;
            }
            ')' => {
                tokens.push(Token {
                    kind: TokenKind::RParen,
                    pos,
                });
                i += 1;
            }
            ',' => {
                tokens.push(Token {
                    kind: TokenKind::Comma,
                    pos,
                });
                i += 1;
            }
            '+' => {
                tokens.push(Token {
                    kind: TokenKind::Plus,
                    pos,
                });
                i += 1;
            }
            '-' => {
                tokens.push(Token {
                    kind: TokenKind::Minus,
                    pos,
                });
                i += 1;
            }
            '*' => {
                tokens.push(Token {
                    kind: TokenKind::Star,
                    pos,
                });
                i += 1;
            }
            '/' => {
                tokens.push(Token {
                    kind: TokenKind::Slash,
                    pos,
                });
                i += 1;
            }
            '>' => {
                if i + 1 < n && chars[i + 1].1 == '=' {
                    tokens.push(Token {
                        kind: TokenKind::Ge,
                        pos,
                    });
                    i += 2;
                } else {
                    tokens.push(Token {
                        kind: TokenKind::Gt,
                        pos,
                    });
                    i += 1;
                }
            }
            '<' => {
                if i + 1 < n && chars[i + 1].1 == '=' {
                    tokens.push(Token {
                        kind: TokenKind::Le,
                        pos,
                    });
                    i += 2;
                } else {
                    tokens.push(Token {
                        kind: TokenKind::Lt,
                        pos,
                    });
                    i += 1;
                }
            }
            c if c.is_ascii_digit() || c == '.' => {
                let start = i;
                let mut has_dot = c == '.';
                let mut has_exp = false;
                i += 1;
                while i < n {
                    let nc = chars[i].1;
                    if nc.is_ascii_digit() {
                        i += 1;
                    } else if nc == '.' && !has_dot && !has_exp {
                        has_dot = true;
                        i += 1;
                    } else if (nc == 'e' || nc == 'E') && !has_exp {
                        has_exp = true;
                        i += 1;
                        if i < n && (chars[i].1 == '+' || chars[i].1 == '-') {
                            i += 1;
                        }
                    } else {
                        break;
                    }
                }
                let end_byte = if i < n { chars[i].0 } else { input.len() };
                let lexeme = &input[chars[start].0..end_byte];
                if !has_dot && !has_exp {
                    if let Ok(iv) = lexeme.parse::<usize>() {
                        tokens.push(Token {
                            kind: TokenKind::IntLit(iv),
                            pos,
                        });
                    } else if let Ok(fv) = lexeme.parse::<f64>() {
                        tokens.push(Token {
                            kind: TokenKind::FloatLit(fv),
                            pos,
                        });
                    } else {
                        return Err(LexError::InvalidNumber {
                            pos,
                            lexeme: lexeme.to_string(),
                        });
                    }
                } else {
                    let fv = lexeme.parse::<f64>().map_err(|_| LexError::InvalidNumber {
                        pos,
                        lexeme: lexeme.to_string(),
                    })?;
                    tokens.push(Token {
                        kind: TokenKind::FloatLit(fv),
                        pos,
                    });
                }
            }
            c if c.is_ascii_alphabetic() || c == '_' => {
                let start = i;
                i += 1;
                while i < n {
                    let nc = chars[i].1;
                    if nc.is_ascii_alphanumeric() || nc == '_' {
                        i += 1;
                    } else {
                        break;
                    }
                }
                let end_byte = if i < n { chars[i].0 } else { input.len() };
                let ident = input[chars[start].0..end_byte].to_string();
                tokens.push(Token {
                    kind: TokenKind::Ident(ident),
                    pos,
                });
            }
            other => {
                return Err(LexError::UnexpectedChar { pos, ch: other });
            }
        }
    }

    tokens.push(Token {
        kind: TokenKind::Eof,
        pos: input.len(),
    });
    Ok(tokens)
}
