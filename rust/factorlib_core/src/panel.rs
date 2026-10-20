//! 核心研究面板数据结构与列优先矩阵容器 (panel.rs)。
//!
//! 按照白皮书 6.1 节设计：
//! - 内部所有连续面板字段使用 `ndarray::Array2<f64>`，形状为 `(T, N)`，采用 Fortran 列优先布局；
//! - 同一品种（列）在内存中连续排列，确保时序滚动算子零跳跃遍历、充分利用 CPU 缓存预取与 SIMD；
//! - `labels` 与 `fields` 严格物理隔离，DSL 语法树禁止访问任何标签列。

use ndarray::{Array2, ShapeBuilder};
use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use thiserror::Error;

#[derive(Error, Debug, Clone)]
pub enum PanelError {
    #[error("面板维度不匹配: 期望 ({expected_t}, {expected_n}), 实际 {name} 为 ({actual_t}, {actual_n})")]
    DimensionMismatch {
        name: String,
        expected_t: usize,
        expected_n: usize,
        actual_t: usize,
        actual_n: usize,
    },
    #[error("未找到特征字段: '{0}'")]
    FieldNotFound(String),
    #[error("严禁未来函数: 表达式禁止引用预测标签或保留字段 '{0}'")]
    ForbiddenLabelAccess(String),
}

/// 将任意布局的二维矩阵转换为标准 Fortran 列优先（Column-Major）布局的 `Array2<f64>`。
pub fn into_fortran_f64(arr: &Array2<f64>) -> Array2<f64> {
    let (t, n) = arr.dim();
    let mut f_arr = Array2::<f64>::zeros((t, n).f());
    for col in 0..n {
        let src_col = arr.column(col);
        let mut dst_col = f_arr.column_mut(col);
        for row in 0..t {
            let v = src_col[row];
            dst_col[row] = if v.is_finite() { v } else { f64::NAN };
        }
    }
    f_arr
}

/// 分配一个新的初始化为 `NaN` 的 Fortran 列优先矩阵 `(T, N)`。
#[inline]
pub fn alloc_nan_matrix(t: usize, n: usize) -> Array2<f64> {
    Array2::<f64>::from_elem((t, n).f(), f64::NAN)
}

/// 判断字段名是否属于系统保留的未来收益标签名。
#[inline]
pub fn is_reserved_label_name(name: &str) -> bool {
    let lower = name.to_ascii_lowercase();
    lower.starts_with("y_")
        || lower == "label"
        || lower.starts_with("label_")
        || lower.starts_with("ret_fwd")
        || lower.starts_with("fwd_ret")
        || lower.starts_with("target_")
        || lower == "target"
}

#[derive(Debug, Clone)]
pub struct Panel {
    /// 交易日时间轴 (长度 T)
    pub dates: Vec<String>,
    /// 品种代码符号列表 (长度 N, 如 ["AG", "AU", "CU", ...])
    pub symbols: Vec<String>,
    /// 基础字段矩阵映射表，键为字段名，值为 T x N 矩阵 (Fortran-order)
    pub fields: HashMap<String, Arc<Array2<f64>>>,
    /// 品种池有效性掩码 (True 代表当日在池)，形状 T x N
    pub universe_mask: Array2<bool>,
    /// 预测收益标签集合，键为标签名 (如 "y_5")，单独存储，防止 DSL 越权访问
    pub labels: HashMap<String, Arc<Array2<f64>>>,
}

impl Panel {
    pub fn new(
        dates: Vec<String>,
        symbols: Vec<String>,
        fields: HashMap<String, Arc<Array2<f64>>>,
        universe_mask: Array2<bool>,
        labels: HashMap<String, Arc<Array2<f64>>>,
    ) -> Result<Self, PanelError> {
        let t = dates.len();
        let n = symbols.len();

        let (mt, mn) = universe_mask.dim();
        if (mt, mn) != (t, n) {
            return Err(PanelError::DimensionMismatch {
                name: "universe_mask".to_string(),
                expected_t: t,
                expected_n: n,
                actual_t: mt,
                actual_n: mn,
            });
        }

        let mut clean_fields = HashMap::with_capacity(fields.len());
        for (k, v) in fields {
            let (ft, fn_) = v.dim();
            if (ft, fn_) != (t, n) {
                return Err(PanelError::DimensionMismatch {
                    name: format!("field:{k}"),
                    expected_t: t,
                    expected_n: n,
                    actual_t: ft,
                    actual_n: fn_,
                });
            }
            if v.is_standard_layout() && t > 1 && n > 1 {
                clean_fields.insert(k, Arc::new(into_fortran_f64(&v)));
            } else {
                clean_fields.insert(k, v);
            }
        }

        let mut clean_labels = HashMap::with_capacity(labels.len());
        for (k, v) in labels {
            let (lt, ln) = v.dim();
            if (lt, ln) != (t, n) {
                return Err(PanelError::DimensionMismatch {
                    name: format!("label:{k}"),
                    expected_t: t,
                    expected_n: n,
                    actual_t: lt,
                    actual_n: ln,
                });
            }
            if v.is_standard_layout() && t > 1 && n > 1 {
                clean_labels.insert(k, Arc::new(into_fortran_f64(&v)));
            } else {
                clean_labels.insert(k, v);
            }
        }

        Ok(Self {
            dates,
            symbols,
            fields: clean_fields,
            universe_mask,
            labels: clean_labels,
        })
    }

    #[inline]
    pub fn shape(&self) -> (usize, usize) {
        (self.dates.len(), self.symbols.len())
    }

    #[inline]
    pub fn num_dates(&self) -> usize {
        self.dates.len()
    }

    #[inline]
    pub fn num_symbols(&self) -> usize {
        self.symbols.len()
    }

    /// 获取指定特征字段矩阵引用；若尝试访问标签名或保留字则立即拒绝。
    pub fn get_field(&self, name: &str) -> Result<&Arc<Array2<f64>>, PanelError> {
        if is_reserved_label_name(name) || self.labels.contains_key(name) {
            return Err(PanelError::ForbiddenLabelAccess(name.to_string()));
        }
        self.fields
            .get(name)
            .ok_or_else(|| PanelError::FieldNotFound(name.to_string()))
    }

    pub fn field_names_set(&self) -> HashSet<String> {
        self.fields.keys().cloned().collect()
    }

    pub fn label_names_set(&self) -> HashSet<String> {
        self.labels.keys().cloned().collect()
    }
}
