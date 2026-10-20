//! PyO3 Python 绑定与 `Engine` 导出封装 (py.rs)。
//!
//! 严格对应白皮书 6.3 节、14.1 节与 P2 清单设计：
//! - 导出 `Engine` 类，接收 `fields`、`mask`、`labels`、`dates`、`symbols`、`cache_mb`、`n_threads`；
//! - 在执行密集计算（`eval`、`eval_batch`、`eval_batch_safe`、`canonicalize_batch`）前，
//!   统一通过 `py.allow_threads(...)` 显式释放 Python GIL；
//! - 导出模块级辅助函数：`parse_and_canonicalize`、`validate_expr`、`list_operators`。

use crate::expr::cache::SubExprCache;
use crate::expr::canon::{canonicalize, CanonResult};
use crate::expr::eval::{eval_batch_str, eval_expr_str};
use crate::expr::parser::parse_expr;
use crate::expr::typecheck::{infer_type, typecheck_factor_expr};
use crate::ops::list_all_operators;
use crate::panel::Panel;
use ndarray::{Array2, ArrayView2, ShapeBuilder};
use numpy::{PyArray2, PyReadonlyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use rayon::prelude::*;
use rayon::{ThreadPool, ThreadPoolBuilder};
use std::collections::{HashMap, HashSet};
use std::sync::Arc;

fn view_to_fortran_f64(view: ArrayView2<'_, f64>) -> Array2<f64> {
    let (t, n) = view.dim();
    let mut f_arr = Array2::<f64>::zeros((t, n).f());
    for col in 0..n {
        let src_col = view.column(col);
        let mut dst_col = f_arr.column_mut(col);
        for row in 0..t {
            let v = src_col[row];
            dst_col[row] = if v.is_finite() { v } else { f64::NAN };
        }
    }
    f_arr
}

fn view_to_fortran_bool(view: ArrayView2<'_, bool>) -> Array2<bool> {
    let (t, n) = view.dim();
    let mut f_arr = Array2::<bool>::from_elem((t, n).f(), false);
    for col in 0..n {
        let src_col = view.column(col);
        let mut dst_col = f_arr.column_mut(col);
        for row in 0..t {
            dst_col[row] = src_col[row];
        }
    }
    f_arr
}

fn extract_string_seq(obj: &Bound<'_, PyAny>) -> PyResult<Vec<String>> {
    let mut out = Vec::new();
    for item in obj.iter()? {
        let elem = item?;
        if let Ok(s) = elem.extract::<String>() {
            out.push(s);
        } else {
            out.push(elem.str()?.to_str()?.to_string());
        }
    }
    Ok(out)
}

fn canon_result_to_pydict<'py>(
    py: Python<'py>,
    cr: &CanonResult,
    ret_type: &str,
) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new_bound(py);
    d.set_item("canon_expr", &cr.canon_expr)?;
    d.set_item("canon_hash", &cr.canon_hash_hex)?;
    d.set_item("hash_u64", cr.canon_hash)?;
    d.set_item("is_degenerate", cr.is_degenerate)?;
    d.set_item("degenerate_reason", &cr.degenerate_reason)?;
    d.set_item("n_nodes", cr.n_nodes)?;
    d.set_item("depth", cr.depth)?;
    d.set_item("return_type", ret_type)?;
    Ok(d)
}

/// 零拷贝持有 `Arc<Array2<f64>>` 的 Python 容器对象，作为 `PyArray2::borrow_from_array_bound` 的生命周期宿主。
#[pyclass(frozen)]
pub struct PyArcMatrix {
    #[allow(dead_code)]
    inner: Arc<Array2<f64>>,
}

#[inline]
fn arc_matrix_to_pyarray<'py>(
    py: Python<'py>,
    mat_arc: Arc<Array2<f64>>,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let ptr: *const Array2<f64> = Arc::as_ptr(&mat_arc);
    let container = Bound::new(py, PyArcMatrix { inner: mat_arc })?;
    // SAFETY: `container` 持有 `Arc<Array2<f64>>`，且 NumPy 数组的 base 指针绑定到 `container`，
    // 在 Python GC 销毁该 NumPy 数组前，`ptr` 指向的堆内存始终有效且不可变。
    let py_arr = unsafe { PyArray2::borrow_from_array_bound(&*ptr, container.into_any()) };
    Ok(py_arr)
}

#[pyclass(name = "Engine")]
pub struct PyEngine {
    panel: Arc<Panel>,
    cache: Arc<SubExprCache>,
    pool: Option<Arc<ThreadPool>>,
    n_threads: usize,
}

#[pymethods]
impl PyEngine {
    #[new]
    #[pyo3(signature = (fields, mask, labels=None, dates=None, symbols=None, cache_mb=4096, n_threads=0))]
    pub fn new(
        fields: &Bound<'_, PyDict>,
        mask: PyReadonlyArray2<'_, bool>,
        labels: Option<&Bound<'_, PyDict>>,
        dates: Option<&Bound<'_, PyAny>>,
        symbols: Option<Vec<String>>,
        cache_mb: usize,
        n_threads: usize,
    ) -> PyResult<Self> {
        let mask_arr = view_to_fortran_bool(mask.as_array());
        let (t, n) = mask_arr.dim();

        let dates_vec = match dates {
            Some(d_obj) => extract_string_seq(d_obj)?,
            None => (0..t).map(|i| format!("d_{i}")).collect(),
        };
        let symbols_vec = symbols.unwrap_or_else(|| (0..n).map(|j| format!("s_{j}")).collect());

        let mut fields_map: HashMap<String, Arc<Array2<f64>>> =
            HashMap::with_capacity(fields.len());
        for (k, v) in fields.iter() {
            let key: String = k.extract()?;
            let arr_ro: PyReadonlyArray2<'_, f64> = v.extract().map_err(|_| {
                PyValueError::new_err(format!("字段 '{key}' 必须为二维 float64 NumPy 数组"))
            })?;
            let f_mat = view_to_fortran_f64(arr_ro.as_array());
            fields_map.insert(key, Arc::new(f_mat));
        }

        let mut labels_map: HashMap<String, Arc<Array2<f64>>> = HashMap::new();
        if let Some(lbl_dict) = labels {
            for (k, v) in lbl_dict.iter() {
                let key: String = k.extract()?;
                let arr_ro: PyReadonlyArray2<'_, f64> = v.extract().map_err(|_| {
                    PyValueError::new_err(format!("标签 '{key}' 必须为二维 float64 NumPy 数组"))
                })?;
                let f_mat = view_to_fortran_f64(arr_ro.as_array());
                labels_map.insert(key, Arc::new(f_mat));
            }
        }

        let panel = Panel::new(dates_vec, symbols_vec, fields_map, mask_arr, labels_map)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;

        let pool = if n_threads > 0 {
            let tp = ThreadPoolBuilder::new()
                .num_threads(n_threads)
                .build()
                .map_err(|e| PyValueError::new_err(format!("构建 Rayon 线程池失败: {e}")))?;
            Some(Arc::new(tp))
        } else {
            None
        };

        let effective_threads = if n_threads > 0 {
            n_threads
        } else {
            rayon::current_num_threads()
        };

        Ok(Self {
            panel: Arc::new(panel),
            cache: Arc::new(SubExprCache::from_megabytes(cache_mb)),
            pool,
            n_threads: effective_threads,
        })
    }

    #[getter]
    pub fn shape(&self) -> (usize, usize) {
        self.panel.shape()
    }

    #[getter]
    pub fn dates(&self) -> Vec<String> {
        self.panel.dates.clone()
    }

    #[getter]
    pub fn symbols(&self) -> Vec<String> {
        self.panel.symbols.clone()
    }

    #[getter]
    pub fn n_threads(&self) -> usize {
        self.n_threads
    }

    #[getter]
    pub fn field_names(&self) -> Vec<String> {
        let mut names: Vec<String> = self.panel.fields.keys().cloned().collect();
        names.sort();
        names
    }

    #[getter]
    pub fn label_names(&self) -> Vec<String> {
        let mut names: Vec<String> = self.panel.labels.keys().cloned().collect();
        names.sort();
        names
    }

    /// 求值单个 DSL 表达式，显式释放 Python GIL，零拷贝返回 `(T, N)` Fortran 列优先 NumPy 矩阵。
    #[pyo3(signature = (expr, use_cache=true, canonicalize=false))]
    pub fn eval<'py>(
        &self,
        py: Python<'py>,
        expr: &str,
        use_cache: bool,
        canonicalize: bool,
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let panel = Arc::clone(&self.panel);
        let cache = if use_cache {
            Some(Arc::clone(&self.cache))
        } else {
            None
        };
        let expr_owned = expr.to_string();

        let res = py.allow_threads(move || {
            let cache_ref = cache.as_deref();
            eval_expr_str(&expr_owned, &panel, cache_ref, canonicalize)
        });

        let mat_arc = res.map_err(|e| PyValueError::new_err(e.to_string()))?;
        arc_matrix_to_pyarray(py, mat_arc)
    }

    /// 并行批求值一组 DSL 表达式，显式释放 Python GIL 并利用自底向上分层 DAG 共享子表达式缓存。
    #[pyo3(signature = (exprs, use_cache=true, canonicalize=false))]
    pub fn eval_batch<'py>(
        &self,
        py: Python<'py>,
        exprs: Vec<String>,
        use_cache: bool,
        canonicalize: bool,
    ) -> PyResult<Bound<'py, PyList>> {
        let panel = Arc::clone(&self.panel);
        let cache = if use_cache {
            Some(Arc::clone(&self.cache))
        } else {
            None
        };
        let pool = self.pool.clone();

        let batch_results = py.allow_threads(move || {
            let cache_ref = cache.as_deref();
            match pool {
                Some(tp) => tp.install(|| eval_batch_str(&exprs, &panel, cache_ref, canonicalize)),
                None => eval_batch_str(&exprs, &panel, cache_ref, canonicalize),
            }
        });

        let out_list = PyList::empty_bound(py);
        for res in batch_results {
            let mat_arc = res.map_err(|e| PyValueError::new_err(e.to_string()))?;
            let py_arr = arc_matrix_to_pyarray(py, mat_arc)?;
            out_list.append(py_arr)?;
        }
        Ok(out_list)
    }

    /// 容错批求值：若批次中某个体退化或非法，对应位置返回 `None` 而不中断整批求值（专为 GP 演化设计）。
    #[pyo3(signature = (exprs, use_cache=true, canonicalize=true))]
    pub fn eval_batch_safe<'py>(
        &self,
        py: Python<'py>,
        exprs: Vec<String>,
        use_cache: bool,
        canonicalize: bool,
    ) -> PyResult<Bound<'py, PyList>> {
        let panel = Arc::clone(&self.panel);
        let cache = if use_cache {
            Some(Arc::clone(&self.cache))
        } else {
            None
        };
        let pool = self.pool.clone();

        let batch_results = py.allow_threads(move || {
            let cache_ref = cache.as_deref();
            match pool {
                Some(tp) => tp.install(|| eval_batch_str(&exprs, &panel, cache_ref, canonicalize)),
                None => eval_batch_str(&exprs, &panel, cache_ref, canonicalize),
            }
        });

        let out_list = PyList::empty_bound(py);
        for res in batch_results {
            match res {
                Ok(mat_arc) => {
                    let py_arr = arc_matrix_to_pyarray(py, mat_arc)?;
                    out_list.append(py_arr)?;
                }
                Err(_) => {
                    out_list.append(py.None())?;
                }
            }
        }
        Ok(out_list)
    }

    /// 对表达式执行静态类型检查与代数规范化，返回包含 `canon_expr`、`canon_hash`、`is_degenerate` 等信息的字典。
    pub fn canonicalize<'py>(&self, py: Python<'py>, expr: &str) -> PyResult<Bound<'py, PyDict>> {
        let ast = parse_expr(expr).map_err(|e| PyValueError::new_err(e.to_string()))?;
        let fields = self.panel.field_names_set();
        let labels = self.panel.label_names_set();
        let ret_ty = typecheck_factor_expr(&ast, Some(&fields), Some(&labels))
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        let cr = canonicalize(&ast);
        canon_result_to_pydict(py, &cr, ret_ty.as_str())
    }

    /// 并行对大批量表达式执行规范化化简与指纹计算。
    pub fn canonicalize_batch<'py>(
        &self,
        py: Python<'py>,
        exprs: Vec<String>,
    ) -> PyResult<Bound<'py, PyList>> {
        let fields = self.panel.field_names_set();
        let labels = self.panel.label_names_set();
        let pool = self.pool.clone();

        let results: Vec<Result<(CanonResult, &'static str), String>> =
            py.allow_threads(move || {
                let work = || {
                    exprs
                        .par_iter()
                        .map(|e| {
                            let ast = parse_expr(e).map_err(|err| err.to_string())?;
                            let ty = typecheck_factor_expr(&ast, Some(&fields), Some(&labels))
                                .map_err(|err| err.to_string())?;
                            Ok((canonicalize(&ast), ty.as_str()))
                        })
                        .collect()
                };
                match pool {
                    Some(tp) => tp.install(work),
                    None => work(),
                }
            });

        let out = PyList::empty_bound(py);
        for item in results {
            let (cr, ty_str) = item.map_err(PyValueError::new_err)?;
            out.append(canon_result_to_pydict(py, &cr, ty_str)?)?;
        }
        Ok(out)
    }

    /// 校验表达式语法、类型及防未来函数约束（不抛出异常，返回包含 `valid` 与 `error` 的字典）。
    pub fn validate_expr<'py>(&self, py: Python<'py>, expr: &str) -> PyResult<Bound<'py, PyDict>> {
        let d = PyDict::new_bound(py);
        match parse_expr(expr) {
            Ok(ast) => {
                let fields = self.panel.field_names_set();
                let labels = self.panel.label_names_set();
                match typecheck_factor_expr(&ast, Some(&fields), Some(&labels)) {
                    Ok(ty) => {
                        let cr = canonicalize(&ast);
                        d.set_item("valid", true)?;
                        d.set_item("return_type", ty.as_str())?;
                        d.set_item("error", py.None())?;
                        d.set_item("n_nodes", ast.node_count())?;
                        d.set_item("depth", ast.depth())?;
                        d.set_item("canon_expr", cr.canon_expr)?;
                        d.set_item("canon_hash", cr.canon_hash_hex)?;
                        d.set_item("is_degenerate", cr.is_degenerate)?;
                    }
                    Err(te) => {
                        d.set_item("valid", false)?;
                        d.set_item("return_type", py.None())?;
                        d.set_item("error", te.to_string())?;
                        d.set_item("n_nodes", ast.node_count())?;
                        d.set_item("depth", ast.depth())?;
                    }
                }
            }
            Err(pe) => {
                d.set_item("valid", false)?;
                d.set_item("return_type", py.None())?;
                d.set_item("error", pe.to_string())?;
                d.set_item("n_nodes", 0usize)?;
                d.set_item("depth", 0usize)?;
            }
        }
        Ok(d)
    }

    /// 获取全局子表达式缓存统计指标（条目数、占用字节、最大字节、命中次数、未命中次数）。
    pub fn cache_stats<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let st = self.cache.stats();
        let d = PyDict::new_bound(py);
        d.set_item("entries", st.entries)?;
        d.set_item("memory_bytes", st.current_memory_bytes)?;
        d.set_item("max_memory_bytes", st.max_memory_bytes)?;
        d.set_item("hits", st.hits)?;
        d.set_item("misses", st.misses)?;
        Ok(d)
    }

    /// 清空子表达式缓存。
    pub fn clear_cache(&self) {
        self.cache.clear();
    }
}

/// 无需实例化 `Engine` 的静态表达式解析与代数规范化函数。
#[pyfunction]
#[pyo3(signature = (expr, available_fields=None))]
pub fn parse_and_canonicalize<'py>(
    py: Python<'py>,
    expr: &str,
    available_fields: Option<Vec<String>>,
) -> PyResult<Bound<'py, PyDict>> {
    let ast = parse_expr(expr).map_err(|e| PyValueError::new_err(e.to_string()))?;
    let fields_set: Option<HashSet<String>> = available_fields.map(|v| v.into_iter().collect());
    let ret_ty = typecheck_factor_expr(&ast, fields_set.as_ref(), None)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    let cr = canonicalize(&ast);
    canon_result_to_pydict(py, &cr, ret_ty.as_str())
}

/// 静态表达式语法与类型检查函数（无需 `Panel` 实例即可校验任意子表达式或完整因子表达式）。
#[pyfunction]
#[pyo3(signature = (expr, available_fields=None, require_series_root=true))]
pub fn check_expr<'py>(
    py: Python<'py>,
    expr: &str,
    available_fields: Option<Vec<String>>,
    require_series_root: bool,
) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new_bound(py);
    let fields_set: Option<HashSet<String>> = available_fields.map(|v| v.into_iter().collect());
    match parse_expr(expr) {
        Ok(ast) => {
            let ty_res = if require_series_root {
                typecheck_factor_expr(&ast, fields_set.as_ref(), None)
            } else {
                infer_type(&ast, fields_set.as_ref(), None)
            };
            match ty_res {
                Ok(ty) => {
                    let cr = canonicalize(&ast);
                    d.set_item("valid", true)?;
                    d.set_item("return_type", ty.as_str())?;
                    d.set_item("error", py.None())?;
                    d.set_item("n_nodes", ast.node_count())?;
                    d.set_item("depth", ast.depth())?;
                    d.set_item("canon_expr", cr.canon_expr)?;
                    d.set_item("canon_hash", cr.canon_hash_hex)?;
                    d.set_item("is_degenerate", cr.is_degenerate)?;
                }
                Err(te) => {
                    d.set_item("valid", false)?;
                    d.set_item("return_type", py.None())?;
                    d.set_item("error", te.to_string())?;
                }
            }
        }
        Err(pe) => {
            d.set_item("valid", false)?;
            d.set_item("return_type", py.None())?;
            d.set_item("error", pe.to_string())?;
        }
    }
    Ok(d)
}

/// 列出内核注册的全部算子规格列表（共 68 个算子）。
#[pyfunction]
pub fn list_operators<'py>(py: Python<'py>) -> PyResult<Bound<'py, PyList>> {
    let specs = list_all_operators();
    let out = PyList::empty_bound(py);
    for sp in specs {
        let d = PyDict::new_bound(py);
        d.set_item("name", sp.name)?;
        d.set_item("category", sp.category)?;
        let args: Vec<&str> = sp.arg_types.iter().map(|t| t.as_str()).collect();
        d.set_item("arg_types", args)?;
        d.set_item("return_type", sp.return_type.as_str())?;
        d.set_item("commutative", sp.commutative)?;
        out.append(d)?;
    }
    Ok(out)
}
