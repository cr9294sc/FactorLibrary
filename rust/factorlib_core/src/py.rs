//! PyO3 Python 绑定与 `Engine` 导出封装 (py.rs)。
//!
//! 严格对应白皮书 6.3 节、14.1 节与 P2 清单设计：
//! - 导出 `Engine` 类，接收 `fields`、`mask`、`labels`、`dates`、`symbols`、`cache_mb`、`n_threads`；
//! - 在执行密集计算（`eval`、`eval_batch`、`eval_batch_safe`、`canonicalize_batch`）前，
//!   统一通过 `py.allow_threads(...)` 显式释放 Python GIL；
//! - 导出模块级辅助函数：`parse_and_canonicalize`、`validate_expr`、`list_operators`。

use crate::backtest::fast::{run_fast_backtest as run_fast_backtest_impl, FastBacktestConfig};
use crate::backtest::futures::{
    run_futures_backtest as run_futures_backtest_impl, FeeType, FuturesBacktestConfig,
    RoundingMode, SymbolSpec,
};
use crate::expr::cache::SubExprCache;
use crate::expr::canon::{canonicalize, CanonResult};
use crate::expr::eval::{eval_batch_str, eval_expr_str};
use crate::expr::parser::parse_expr;
use crate::expr::typecheck::{infer_type, typecheck_factor_expr};
use crate::metrics::corr::{cross_correlation_matrix, cross_correlation_pairwise};
use crate::metrics::ic::{
    calc_daily_cross_sectional_ic, calc_ic_metrics, calc_ic_series, IcMetrics,
};
use crate::metrics::quantile::{
    calc_quantile_metrics, calc_top_quantile_turnover as rust_calc_top_quantile_turnover,
    QuantileMetrics, TopQuantileTurnoverMetrics,
};
use crate::metrics::turnover::{calc_turnover_metrics, TurnoverMetrics};
use crate::ops::list_all_operators;
use crate::ops::postprocess::{
    pp_decay, pp_drop_new_stock, pp_emadecay, pp_ind_neut, pp_mad, pp_mask_mul, pp_rank,
    pp_risk_neut, pp_risk_neut_multi, pp_winsor, pp_zscore,
};
use crate::panel::Panel;
use ndarray::{s, Array2, ArrayView2, ShapeBuilder};
use numpy::{PyArray1, PyArray2, PyArrayMethods, PyReadonlyArray2};
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

fn ic_metrics_to_pydict<'py>(py: Python<'py>, m: &IcMetrics) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new_bound(py);
    d.set_item("ic_mean", m.ic_mean)?;
    d.set_item("ic_std", m.ic_std)?;
    d.set_item("ic_ir", m.ic_ir)?;
    d.set_item("ic_t", m.ic_t)?;
    d.set_item("ic_win_rate", m.ic_win_rate)?;
    d.set_item("coverage", m.coverage)?;
    d.set_item("valid_days", m.valid_days)?;
    d.set_item("daily_ic", PyArray1::from_slice_bound(py, &m.daily_ic))?;
    Ok(d)
}

fn quantile_metrics_to_pydict<'py>(
    py: Python<'py>,
    m: &QuantileMetrics,
) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new_bound(py);
    d.set_item("n_quantiles", m.n_quantiles)?;
    d.set_item("group_returns", PyList::new_bound(py, &m.group_returns))?;
    d.set_item(
        "group_annual_returns",
        PyList::new_bound(py, &m.group_annual_returns),
    )?;
    d.set_item("ls_mean", m.ls_mean)?;
    d.set_item("ls_std", m.ls_std)?;
    d.set_item("ls_annual_ret", m.ls_annual_ret)?;
    d.set_item("ls_sharpe", m.ls_sharpe)?;
    d.set_item("ls_max_dd", m.ls_max_dd)?;
    d.set_item("monotonicity", m.monotonicity)?;
    d.set_item("valid_days", m.valid_days)?;
    d.set_item(
        "ls_daily_returns",
        PyArray1::from_slice_bound(py, &m.ls_daily_returns),
    )?;
    let g_daily_list = PyList::empty_bound(py);
    for g in &m.group_daily_returns {
        g_daily_list.append(PyArray1::from_slice_bound(py, g))?;
    }
    d.set_item("group_daily_returns", g_daily_list)?;
    d.set_item("top_turnover", m.mean_top_turnover)?;
    d.set_item("mean_top_turnover", m.mean_top_turnover)?;
    d.set_item("top_turnover_valid_days", m.top_turnover_valid_days)?;
    d.set_item(
        "top_turnover_series",
        PyArray1::from_slice_bound(py, &m.top_turnover_series),
    )?;
    Ok(d)
}

fn turnover_metrics_to_pydict<'py>(
    py: Python<'py>,
    m: &TurnoverMetrics,
) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new_bound(py);
    d.set_item("turnover", m.turnover)?;
    d.set_item("rank_autocorr", m.rank_autocorr)?;
    d.set_item("rank_autocorr_turnover", m.rank_autocorr_turnover)?;
    d.set_item("valid_days", m.valid_days)?;
    d.set_item(
        "daily_autocorr",
        PyArray1::from_slice_bound(py, &m.daily_autocorr),
    )?;
    Ok(d)
}

fn top_quantile_turnover_to_pydict<'py>(
    py: Python<'py>,
    m: &TopQuantileTurnoverMetrics,
) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new_bound(py);
    d.set_item("turnover", m.turnover)?;
    d.set_item("mean_top_turnover", m.turnover)?;
    d.set_item("valid_days", m.valid_days)?;
    d.set_item(
        "daily_turnover",
        PyArray1::from_slice_bound(py, &m.daily_turnover),
    )?;
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
    py_arr.readwrite().make_nonwriteable();
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
                        let (lb, f_wins, ops_used) = ast.analyze_lookback();
                        let fw_dict = PyDict::new_bound(py);
                        for (k, w) in f_wins {
                            fw_dict.set_item(k, w)?;
                        }
                        d.set_item("valid", true)?;
                        d.set_item("return_type", ty.as_str())?;
                        d.set_item("error", py.None())?;
                        d.set_item("n_nodes", ast.node_count())?;
                        d.set_item("depth", ast.depth())?;
                        d.set_item("canon_expr", cr.canon_expr)?;
                        d.set_item("canon_hash", cr.canon_hash_hex)?;
                        d.set_item("is_degenerate", cr.is_degenerate)?;
                        d.set_item("lookback", lb)?;
                        d.set_item("field_windows", fw_dict)?;
                        d.set_item("operators", ops_used)?;
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

    /// 评估因子截面 Spearman Rank IC 或 Pearson IC 时序与多维统计指标。
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (factor, label_name="y_5", min_names=5, rank=true, horizon=1, start_row=None, end_row=None))]
    pub fn rank_ic<'py>(
        &self,
        py: Python<'py>,
        factor: &Bound<'_, PyAny>,
        label_name: &str,
        min_names: usize,
        rank: bool,
        horizon: usize,
        start_row: Option<usize>,
        end_row: Option<usize>,
    ) -> PyResult<Bound<'py, PyDict>> {
        let f_raw = self.resolve_factor_mat(py, factor)?;
        let (f_mat, s_row, e_row) = self.align_factor_slice(f_raw, start_row, end_row)?;
        let label_arc = self
            .panel
            .labels
            .get(label_name)
            .ok_or_else(|| {
                PyValueError::new_err(format!("未在面板中找到预测标签: '{label_name}'"))
            })?
            .clone();
        let label_sub = view_to_fortran_f64(label_arc.slice(s![s_row..e_row, ..]));
        let mask_sub = view_to_fortran_bool(self.panel.universe_mask.slice(s![s_row..e_row, ..]));

        let metrics = py.allow_threads(move || {
            let (t, n) = f_mat.dim();
            let mut valid_f = 0;
            let mut total_univ = 0;
            for row in 0..t {
                for col in 0..n {
                    if mask_sub[[row, col]] {
                        total_univ += 1;
                        if f_mat[[row, col]].is_finite() {
                            valid_f += 1;
                        }
                    }
                }
            }
            let daily = calc_ic_series(&f_mat, &label_sub, &mask_sub, min_names, rank);
            calc_ic_metrics(&daily, horizon, t, valid_f, total_univ)
        });

        ic_metrics_to_pydict(py, &metrics)
    }

    /// 评估因子分位数分组收益与多空对冲收益时序指标。
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (factor, ret_name=None, n_quantiles=5, min_names=5, sign=1, start_row=None, end_row=None))]
    pub fn quantile_returns<'py>(
        &self,
        py: Python<'py>,
        factor: &Bound<'_, PyAny>,
        ret_name: Option<&str>,
        n_quantiles: usize,
        min_names: usize,
        sign: i32,
        start_row: Option<usize>,
        end_row: Option<usize>,
    ) -> PyResult<Bound<'py, PyDict>> {
        let f_raw = self.resolve_factor_mat(py, factor)?;
        let (f_mat, s_row, e_row) = self.align_factor_slice(f_raw, start_row, end_row)?;
        if n_quantiles < 2 {
            return Err(PyValueError::new_err("n_quantiles 必须 >= 2"));
        }
        let r_name = ret_name.unwrap_or("y_1");
        let ret_arc = self
            .panel
            .labels
            .get(r_name)
            .or_else(|| self.panel.labels.get("r_exec_next"))
            .ok_or_else(|| {
                PyValueError::new_err(format!(
                    "未在面板中找到用于分位数评估的收益标签: '{r_name}'"
                ))
            })?
            .clone();
        let ret_sub = view_to_fortran_f64(ret_arc.slice(s![s_row..e_row, ..]));
        let mask_sub = view_to_fortran_bool(self.panel.universe_mask.slice(s![s_row..e_row, ..]));

        let metrics = py.allow_threads(move || {
            calc_quantile_metrics(&f_mat, &ret_sub, &mask_sub, n_quantiles, min_names, sign)
        });

        quantile_metrics_to_pydict(py, &metrics)
    }

    /// 评估因子横截面排名自相关与日度换手率代理指标。
    #[pyo3(signature = (factor, min_names=5, start_row=None, end_row=None))]
    pub fn factor_turnover<'py>(
        &self,
        py: Python<'py>,
        factor: &Bound<'_, PyAny>,
        min_names: usize,
        start_row: Option<usize>,
        end_row: Option<usize>,
    ) -> PyResult<Bound<'py, PyDict>> {
        let f_raw = self.resolve_factor_mat(py, factor)?;
        let (f_mat, s_row, e_row) = self.align_factor_slice(f_raw, start_row, end_row)?;
        let mask_sub = view_to_fortran_bool(self.panel.universe_mask.slice(s![s_row..e_row, ..]));

        let metrics = py.allow_threads(move || calc_turnover_metrics(&f_mat, &mask_sub, min_names));

        turnover_metrics_to_pydict(py, &metrics)
    }

    /// 评估因子多头 Top 分位日度单边持仓权重换手率。
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (factor, n_quantiles=5, min_names=5, sign=1, start_row=None, end_row=None))]
    pub fn top_quantile_turnover<'py>(
        &self,
        py: Python<'py>,
        factor: &Bound<'_, PyAny>,
        n_quantiles: usize,
        min_names: usize,
        sign: i32,
        start_row: Option<usize>,
        end_row: Option<usize>,
    ) -> PyResult<Bound<'py, PyDict>> {
        let f_raw = self.resolve_factor_mat(py, factor)?;
        let (f_mat, s_row, e_row) = self.align_factor_slice(f_raw, start_row, end_row)?;
        let mask_sub = view_to_fortran_bool(self.panel.universe_mask.slice(s![s_row..e_row, ..]));

        if n_quantiles < 2 {
            return Err(PyValueError::new_err("n_quantiles 必须 >= 2"));
        }

        let metrics = py.allow_threads(move || {
            rust_calc_top_quantile_turnover(&f_mat, &mask_sub, n_quantiles, min_names, sign)
        });

        top_quantile_turnover_to_pydict(py, &metrics)
    }

    /// 因子多维度综合评估（单次调用同时计算 IC、分位数多空、换手率，全程释放 GIL）。
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (factor, label_name="y_5", ret_name=None, horizon=5, n_quantiles=5, min_names=5, sign=1, start_row=None, end_row=None))]
    pub fn factor_metrics<'py>(
        &self,
        py: Python<'py>,
        factor: &Bound<'_, PyAny>,
        label_name: &str,
        ret_name: Option<&str>,
        horizon: usize,
        n_quantiles: usize,
        min_names: usize,
        sign: i32,
        start_row: Option<usize>,
        end_row: Option<usize>,
    ) -> PyResult<Bound<'py, PyDict>> {
        let f_raw = self.resolve_factor_mat(py, factor)?;
        let (f_mat, s_row, e_row) = self.align_factor_slice(f_raw, start_row, end_row)?;
        if n_quantiles < 2 {
            return Err(PyValueError::new_err("n_quantiles 必须 >= 2"));
        }
        let label_arc = self
            .panel
            .labels
            .get(label_name)
            .ok_or_else(|| {
                PyValueError::new_err(format!("未在面板中找到预测标签: '{label_name}'"))
            })?
            .clone();
        let r_name = ret_name.unwrap_or("y_1");
        let ret_arc = self
            .panel
            .labels
            .get(r_name)
            .or_else(|| self.panel.labels.get("r_exec_next"))
            .unwrap_or(&label_arc)
            .clone();
        let label_sub = view_to_fortran_f64(label_arc.slice(s![s_row..e_row, ..]));
        let ret_sub = view_to_fortran_f64(ret_arc.slice(s![s_row..e_row, ..]));
        let mask_sub = view_to_fortran_bool(self.panel.universe_mask.slice(s![s_row..e_row, ..]));

        let (ic_m, q_m, to_m) = py.allow_threads(move || {
            let (t, n) = f_mat.dim();
            let mut valid_f = 0;
            let mut total_univ = 0;
            for row in 0..t {
                for col in 0..n {
                    if mask_sub[[row, col]] {
                        total_univ += 1;
                        if f_mat[[row, col]].is_finite() {
                            valid_f += 1;
                        }
                    }
                }
            }
            let daily = calc_ic_series(&f_mat, &label_sub, &mask_sub, min_names, true);
            let ic_res = calc_ic_metrics(&daily, horizon, t, valid_f, total_univ);
            let q_res =
                calc_quantile_metrics(&f_mat, &ret_sub, &mask_sub, n_quantiles, min_names, sign);
            let to_res = calc_turnover_metrics(&f_mat, &mask_sub, min_names);
            (ic_res, q_res, to_res)
        });

        let d = PyDict::new_bound(py);
        d.set_item("ic", ic_metrics_to_pydict(py, &ic_m)?)?;
        d.set_item("quantile", quantile_metrics_to_pydict(py, &q_m)?)?;
        d.set_item("turnover", turnover_metrics_to_pydict(py, &to_m)?)?;
        d.set_item("ic_mean", ic_m.ic_mean)?;
        d.set_item("ic_std", ic_m.ic_std)?;
        d.set_item("ic_ir", ic_m.ic_ir)?;
        d.set_item("ic_t", ic_m.ic_t)?;
        d.set_item("ic_win_rate", ic_m.ic_win_rate)?;
        d.set_item("coverage", ic_m.coverage)?;
        d.set_item("turnover_val", q_m.mean_top_turnover)?;
        d.set_item("top_turnover", q_m.mean_top_turnover)?;
        d.set_item("rank_autocorr_turnover", to_m.turnover)?;
        d.set_item("ls_sharpe", q_m.ls_sharpe)?;
        d.set_item("ls_annual_ret", q_m.ls_annual_ret)?;
        d.set_item("ls_max_dd", q_m.ls_max_dd)?;
        d.set_item("monotonicity", q_m.monotonicity)?;
        Ok(d)
    }

    /// 计算两个因子在有效标的池上的平均横截面相关系数。
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (factor_a, factor_b, min_names=5, rank=true, start_row=None, end_row=None))]
    pub fn cross_correlation<'py>(
        &self,
        py: Python<'py>,
        factor_a: &Bound<'_, PyAny>,
        factor_b: &Bound<'_, PyAny>,
        min_names: usize,
        rank: bool,
        start_row: Option<usize>,
        end_row: Option<usize>,
    ) -> PyResult<f64> {
        let fa_raw = self.resolve_factor_mat(py, factor_a)?;
        let fb_raw = self.resolve_factor_mat(py, factor_b)?;
        let (fa, s_row, e_row) = self.align_factor_slice(fa_raw, start_row, end_row)?;
        let (fb, s_b, e_b) = self.align_factor_slice(fb_raw, Some(s_row), Some(e_row))?;
        if (s_row, e_row) != (s_b, e_b) || fa.dim() != fb.dim() {
            return Err(PyValueError::new_err(
                "factor_a 与 factor_b 的行切片维度不一致",
            ));
        }
        let mask_sub = view_to_fortran_bool(self.panel.universe_mask.slice(s![s_row..e_row, ..]));

        let rho = py.allow_threads(move || {
            cross_correlation_pairwise(&fa, &fb, &mask_sub, min_names, rank)
        });
        Ok(rho)
    }

    /// 并行计算一组因子的 M x M 截面相关系数矩阵。
    #[pyo3(signature = (factors, min_names=5, rank=true, start_row=None, end_row=None))]
    pub fn factor_correlation_matrix<'py>(
        &self,
        py: Python<'py>,
        factors: Vec<PyReadonlyArray2<'_, f64>>,
        min_names: usize,
        rank: bool,
        start_row: Option<usize>,
        end_row: Option<usize>,
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let panel_t = self.panel.shape().0;
        let s_row = start_row.unwrap_or(0);
        let e_row = end_row.unwrap_or_else(|| {
            if let Some(first) = factors.first() {
                let r = first.as_array().dim().0;
                if r < panel_t && start_row.is_none() {
                    r
                } else {
                    panel_t
                }
            } else {
                panel_t
            }
        });
        if s_row > e_row || e_row > panel_t {
            return Err(PyValueError::new_err("无效的行切片范围"));
        }

        let mut f_mats: Vec<Array2<f64>> = Vec::with_capacity(factors.len());
        for f in factors {
            let raw = view_to_fortran_f64(f.as_array());
            let (sliced, _, _) = self.align_factor_slice(raw, Some(s_row), Some(e_row))?;
            f_mats.push(sliced);
        }
        let mask_sub = view_to_fortran_bool(self.panel.universe_mask.slice(s![s_row..e_row, ..]));

        let mat = py.allow_threads(move || {
            let refs: Vec<&Array2<f64>> = f_mats.iter().collect();
            cross_correlation_matrix(&refs, &mask_sub, min_names, rank)
        });

        Ok(PyArray2::from_owned_array_bound(py, mat))
    }

    /// 遗传规划 (STGP) 专用高并发批求值与适应度指标计算（单次释放 GIL 完成 DAG 求值 + 切片 RankIC + 库内相关性）。
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (exprs, label_name="y_5", start_row=0, end_row=None, min_names=5, horizon=5, library_factors=None, use_cache=true))]
    pub fn eval_gp_batch<'py>(
        &self,
        py: Python<'py>,
        exprs: Vec<String>,
        label_name: &str,
        start_row: usize,
        end_row: Option<usize>,
        min_names: usize,
        horizon: usize,
        library_factors: Option<Vec<PyReadonlyArray2<'_, f64>>>,
        use_cache: bool,
    ) -> PyResult<Bound<'py, PyList>> {
        let (panel_t, panel_n) = self.panel.shape();
        let e_row = end_row.unwrap_or(panel_t).min(panel_t);
        let s_row = start_row.min(e_row);

        let label_arc = self
            .panel
            .labels
            .get(label_name)
            .ok_or_else(|| {
                PyValueError::new_err(format!("未在面板中找到预测标签: '{label_name}'"))
            })?
            .clone();

        let mut lib_mats: Vec<Array2<f64>> = Vec::new();
        if let Some(libs) = library_factors {
            for lf in libs {
                let raw = view_to_fortran_f64(lf.as_array());
                let (sliced, _, _) = self.align_factor_slice(raw, Some(s_row), Some(e_row))?;
                lib_mats.push(sliced);
            }
        }

        let panel = Arc::clone(&self.panel);
        let cache = if use_cache {
            Some(Arc::clone(&self.cache))
        } else {
            None
        };
        let pool = self.pool.clone();

        struct GpEvalItem {
            valid: bool,
            ic_mean: f64,
            ic_std: f64,
            ic_ir: f64,
            ic_t: f64,
            ic_win_rate: f64,
            coverage: f64,
            valid_days: usize,
            max_abs_corr: f64,
            mean_abs_corr: f64,
            error: Option<String>,
        }

        let batch_items: Vec<GpEvalItem> = py.allow_threads(move || {
            let work = || {
                let cache_ref = cache.as_deref();
                let eval_res = eval_batch_str(&exprs, &panel, cache_ref, true);

                let t_sub = e_row - s_row;
                let label_sub = view_to_fortran_f64(label_arc.slice(s![s_row..e_row, ..]));
                let mask_sub =
                    view_to_fortran_bool(panel.universe_mask.slice(s![s_row..e_row, ..]));

                let mut total_univ = 0usize;
                for r in 0..t_sub {
                    for c in 0..panel_n {
                        if mask_sub[[r, c]] {
                            total_univ += 1;
                        }
                    }
                }

                eval_res
                    .into_par_iter()
                    .map(|res| match res {
                        Ok(mat_arc) => {
                            let f_sub = view_to_fortran_f64(mat_arc.slice(s![s_row..e_row, ..]));
                            let mut valid_f = 0usize;
                            for r in 0..t_sub {
                                for c in 0..panel_n {
                                    if mask_sub[[r, c]] && f_sub[[r, c]].is_finite() {
                                        valid_f += 1;
                                    }
                                }
                            }
                            let mut daily_ic = Vec::with_capacity(t_sub);
                            let mut f_row = vec![0.0_f64; panel_n];
                            let mut l_row = vec![0.0_f64; panel_n];
                            let mut m_row = vec![false; panel_n];
                            for r in 0..t_sub {
                                for c in 0..panel_n {
                                    f_row[c] = f_sub[[r, c]];
                                    l_row[c] = label_sub[[r, c]];
                                    m_row[c] = mask_sub[[r, c]];
                                }
                                daily_ic.push(calc_daily_cross_sectional_ic(
                                    &f_row, &l_row, &m_row, min_names, true,
                                ));
                            }
                            let ic_m =
                                calc_ic_metrics(&daily_ic, horizon, t_sub, valid_f, total_univ);

                            let mut max_abs_corr = 0.0_f64;
                            let mut sum_abs_corr = 0.0_f64;
                            if !lib_mats.is_empty() && ic_m.valid_days > 0 {
                                for lib_f in &lib_mats {
                                    let rho = cross_correlation_pairwise(
                                        &f_sub, lib_f, &mask_sub, min_names, true,
                                    )
                                    .abs();
                                    if rho.is_finite() {
                                        if rho > max_abs_corr {
                                            max_abs_corr = rho;
                                        }
                                        sum_abs_corr += rho;
                                    }
                                }
                            }
                            let mean_abs_corr = if !lib_mats.is_empty() {
                                sum_abs_corr / (lib_mats.len() as f64)
                            } else {
                                0.0
                            };

                            GpEvalItem {
                                valid: true,
                                ic_mean: ic_m.ic_mean,
                                ic_std: ic_m.ic_std,
                                ic_ir: ic_m.ic_ir,
                                ic_t: ic_m.ic_t,
                                ic_win_rate: ic_m.ic_win_rate,
                                coverage: ic_m.coverage,
                                valid_days: ic_m.valid_days,
                                max_abs_corr,
                                mean_abs_corr,
                                error: None,
                            }
                        }
                        Err(e) => GpEvalItem {
                            valid: false,
                            ic_mean: 0.0,
                            ic_std: 0.0,
                            ic_ir: 0.0,
                            ic_t: 0.0,
                            ic_win_rate: 0.0,
                            coverage: 0.0,
                            valid_days: 0,
                            max_abs_corr: 0.0,
                            mean_abs_corr: 0.0,
                            error: Some(e.to_string()),
                        },
                    })
                    .collect()
            };

            match pool {
                Some(tp) => tp.install(work),
                None => work(),
            }
        });

        let out_list = PyList::empty_bound(py);
        for item in batch_items {
            let d = PyDict::new_bound(py);
            d.set_item("valid", item.valid)?;
            d.set_item("ic_mean", item.ic_mean)?;
            d.set_item("ic_std", item.ic_std)?;
            d.set_item("ic_ir", item.ic_ir)?;
            d.set_item("ic_t", item.ic_t)?;
            d.set_item("ic_win_rate", item.ic_win_rate)?;
            d.set_item("coverage", item.coverage)?;
            d.set_item("valid_days", item.valid_days)?;
            d.set_item("max_abs_corr", item.max_abs_corr)?;
            d.set_item("mean_abs_corr", item.mean_abs_corr)?;
            d.set_item("error", item.error)?;
            out_list.append(d)?;
        }
        Ok(out_list)
    }
}

impl PyEngine {
    fn resolve_factor_mat<'py>(
        &self,
        py: Python<'py>,
        factor_obj: &Bound<'py, PyAny>,
    ) -> PyResult<Array2<f64>> {
        if let Ok(expr_str) = factor_obj.extract::<String>() {
            let p = Arc::clone(&self.panel);
            let c = Arc::clone(&self.cache);
            let res = py.allow_threads(move || eval_expr_str(&expr_str, &p, Some(&c), false));
            let arc_mat = res.map_err(|e| PyValueError::new_err(e.to_string()))?;
            Ok((*arc_mat).clone())
        } else if let Ok(ro_arr) = factor_obj.extract::<PyReadonlyArray2<'_, f64>>() {
            Ok(view_to_fortran_f64(ro_arr.as_array()))
        } else {
            Err(PyValueError::new_err(
                "factor 必须为 DSL 表达式文本或二维 float64 NumPy 数组",
            ))
        }
    }

    fn align_factor_slice(
        &self,
        f_mat: Array2<f64>,
        start_row: Option<usize>,
        end_row: Option<usize>,
    ) -> PyResult<(Array2<f64>, usize, usize)> {
        let (panel_t, panel_n) = self.panel.shape();
        let (f_t, f_n) = f_mat.dim();
        if f_n != panel_n {
            return Err(PyValueError::new_err(format!(
                "因子列数 ({f_n}) 与面板品种数 ({panel_n}) 不一致"
            )));
        }
        let s_row = start_row.unwrap_or(0);
        let e_row = end_row.unwrap_or(if f_t < panel_t && start_row.is_none() {
            f_t
        } else {
            panel_t
        });
        if s_row > e_row || e_row > panel_t {
            return Err(PyValueError::new_err(format!(
                "无效的行切片范围: [{s_row}, {e_row}), 面板总行数={panel_t}"
            )));
        }
        let expected_sub = e_row - s_row;
        if f_t == expected_sub {
            Ok((f_mat, s_row, e_row))
        } else if f_t == panel_t {
            let sliced = view_to_fortran_f64(f_mat.slice(s![s_row..e_row, ..]));
            Ok((sliced, s_row, e_row))
        } else {
            Err(PyValueError::new_err(format!(
                "因子行数 ({f_t}) 与预期切片行数 ({expected_sub}) 或面板行数 ({panel_t}) 不匹配"
            )))
        }
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
                    let (lb, f_wins, ops_used) = ast.analyze_lookback();
                    let fw_dict = PyDict::new_bound(py);
                    for (k, w) in f_wins {
                        fw_dict.set_item(k, w)?;
                    }
                    d.set_item("valid", true)?;
                    d.set_item("return_type", ty.as_str())?;
                    d.set_item("error", py.None())?;
                    d.set_item("n_nodes", ast.node_count())?;
                    d.set_item("depth", ast.depth())?;
                    d.set_item("canon_expr", cr.canon_expr)?;
                    d.set_item("canon_hash", cr.canon_hash_hex)?;
                    d.set_item("is_degenerate", cr.is_degenerate)?;
                    d.set_item("lookback", lb)?;
                    d.set_item("field_windows", fw_dict)?;
                    d.set_item("operators", ops_used)?;
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

/// 计算两个矩阵在指定掩码下的截面 IC 序列与多维评价指标。
#[allow(clippy::too_many_arguments)]
#[pyfunction]
#[pyo3(signature = (factor, label, mask, min_names=5, rank=true, horizon=1))]
pub fn calc_ic<'py>(
    py: Python<'py>,
    factor: PyReadonlyArray2<'_, f64>,
    label: PyReadonlyArray2<'_, f64>,
    mask: PyReadonlyArray2<'_, bool>,
    min_names: usize,
    rank: bool,
    horizon: usize,
) -> PyResult<Bound<'py, PyDict>> {
    let f_mat = view_to_fortran_f64(factor.as_array());
    let l_mat = view_to_fortran_f64(label.as_array());
    let m_mat = view_to_fortran_bool(mask.as_array());

    let (t, n) = f_mat.dim();
    if l_mat.dim() != (t, n) || m_mat.dim() != (t, n) {
        return Err(PyValueError::new_err(
            "factor, label 与 mask 矩阵维度必须完全一致",
        ));
    }

    let metrics = py.allow_threads(move || {
        let mut valid_f = 0;
        let mut total_univ = 0;
        for row in 0..t {
            for col in 0..n {
                if m_mat[[row, col]] {
                    total_univ += 1;
                    if f_mat[[row, col]].is_finite() {
                        valid_f += 1;
                    }
                }
            }
        }
        let daily = calc_ic_series(&f_mat, &l_mat, &m_mat, min_names, rank);
        calc_ic_metrics(&daily, horizon, t, valid_f, total_univ)
    });

    ic_metrics_to_pydict(py, &metrics)
}

/// 计算因子矩阵在指定收益标签下的分位数分组与多空组合指标。
#[allow(clippy::too_many_arguments)]
#[pyfunction]
#[pyo3(signature = (factor, forward_ret, mask, n_quantiles=5, min_names=5, sign=1))]
pub fn calc_quantile_returns<'py>(
    py: Python<'py>,
    factor: PyReadonlyArray2<'_, f64>,
    forward_ret: PyReadonlyArray2<'_, f64>,
    mask: PyReadonlyArray2<'_, bool>,
    n_quantiles: usize,
    min_names: usize,
    sign: i32,
) -> PyResult<Bound<'py, PyDict>> {
    let f_mat = view_to_fortran_f64(factor.as_array());
    let r_mat = view_to_fortran_f64(forward_ret.as_array());
    let m_mat = view_to_fortran_bool(mask.as_array());

    let (t, n) = f_mat.dim();
    if r_mat.dim() != (t, n) || m_mat.dim() != (t, n) {
        return Err(PyValueError::new_err(
            "factor, forward_ret 与 mask 矩阵维度必须完全一致",
        ));
    }
    if n_quantiles < 2 {
        return Err(PyValueError::new_err("n_quantiles 必须 >= 2"));
    }

    let metrics = py.allow_threads(move || {
        calc_quantile_metrics(&f_mat, &r_mat, &m_mat, n_quantiles, min_names, sign)
    });

    quantile_metrics_to_pydict(py, &metrics)
}

/// 计算因子多头 Top 分位日度单边持仓权重换手率。
#[pyfunction]
#[pyo3(signature = (factor, mask, n_quantiles=5, min_names=5, sign=1))]
pub fn calc_top_quantile_turnover<'py>(
    py: Python<'py>,
    factor: PyReadonlyArray2<'_, f64>,
    mask: PyReadonlyArray2<'_, bool>,
    n_quantiles: usize,
    min_names: usize,
    sign: i32,
) -> PyResult<Bound<'py, PyDict>> {
    let f_mat = view_to_fortran_f64(factor.as_array());
    let m_mat = view_to_fortran_bool(mask.as_array());

    let (t, n) = f_mat.dim();
    if m_mat.dim() != (t, n) {
        return Err(PyValueError::new_err("factor 与 mask 矩阵维度必须完全一致"));
    }
    if n_quantiles < 2 {
        return Err(PyValueError::new_err("n_quantiles 必须 >= 2"));
    }

    let metrics = py.allow_threads(move || {
        rust_calc_top_quantile_turnover(&f_mat, &m_mat, n_quantiles, min_names, sign)
    });

    top_quantile_turnover_to_pydict(py, &metrics)
}

/// 计算因子横截面排名自相关与日度换手率代理指标。
#[pyfunction]
#[pyo3(signature = (factor, mask, min_names=5))]
pub fn calc_turnover<'py>(
    py: Python<'py>,
    factor: PyReadonlyArray2<'_, f64>,
    mask: PyReadonlyArray2<'_, bool>,
    min_names: usize,
) -> PyResult<Bound<'py, PyDict>> {
    let f_mat = view_to_fortran_f64(factor.as_array());
    let m_mat = view_to_fortran_bool(mask.as_array());

    let (t, n) = f_mat.dim();
    if m_mat.dim() != (t, n) {
        return Err(PyValueError::new_err("factor 与 mask 矩阵维度必须完全一致"));
    }

    let metrics = py.allow_threads(move || calc_turnover_metrics(&f_mat, &m_mat, min_names));

    turnover_metrics_to_pydict(py, &metrics)
}

/// 计算两个因子矩阵之间的横截面相关系数均值。
#[pyfunction]
#[pyo3(signature = (factor_a, factor_b, mask, min_names=5, rank=true))]
pub fn calc_pairwise_correlation<'py>(
    py: Python<'py>,
    factor_a: PyReadonlyArray2<'_, f64>,
    factor_b: PyReadonlyArray2<'_, f64>,
    mask: PyReadonlyArray2<'_, bool>,
    min_names: usize,
    rank: bool,
) -> PyResult<f64> {
    let fa = view_to_fortran_f64(factor_a.as_array());
    let fb = view_to_fortran_f64(factor_b.as_array());
    let m = view_to_fortran_bool(mask.as_array());

    let (t, n) = fa.dim();
    if fb.dim() != (t, n) || m.dim() != (t, n) {
        return Err(PyValueError::new_err(
            "factor_a, factor_b 与 mask 矩阵维度必须完全一致",
        ));
    }

    let rho = py.allow_threads(move || cross_correlation_pairwise(&fa, &fb, &m, min_names, rank));

    Ok(rho)
}

/// 计算多个因子矩阵之间的 M x M 截面相关系数矩阵。
#[pyfunction]
#[pyo3(signature = (factors, mask, min_names=5, rank=true))]
pub fn calc_correlation_matrix<'py>(
    py: Python<'py>,
    factors: Vec<PyReadonlyArray2<'_, f64>>,
    mask: PyReadonlyArray2<'_, bool>,
    min_names: usize,
    rank: bool,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let m_mat = view_to_fortran_bool(mask.as_array());
    let (t, n) = m_mat.dim();

    let mut f_mats: Vec<Array2<f64>> = Vec::with_capacity(factors.len());
    for f in factors {
        let mat = view_to_fortran_f64(f.as_array());
        if mat.dim() != (t, n) {
            return Err(PyValueError::new_err(
                "所有因子矩阵的维度必须与 mask 矩阵严格一致",
            ));
        }
        f_mats.push(mat);
    }

    let mat = py.allow_threads(move || {
        let refs: Vec<&Array2<f64>> = f_mats.iter().collect();
        cross_correlation_matrix(&refs, &m_mat, min_names, rank)
    });

    Ok(PyArray2::from_owned_array_bound(py, mat))
}

/// 快速连续权重回测函数。
#[pyfunction]
#[pyo3(signature = (weights, returns, fee_rate=0.0001, slippage_bps=1.0, init_cash=10000000.0))]
pub fn run_fast_backtest<'py>(
    py: Python<'py>,
    weights: PyReadonlyArray2<'_, f64>,
    returns: PyReadonlyArray2<'_, f64>,
    fee_rate: f64,
    slippage_bps: f64,
    init_cash: f64,
) -> PyResult<Bound<'py, PyDict>> {
    let w_mat = view_to_fortran_f64(weights.as_array());
    let r_mat = view_to_fortran_f64(returns.as_array());
    if w_mat.dim() != r_mat.dim() {
        return Err(PyValueError::new_err("weights 与 returns 维度必须一致"));
    }

    let cfg = FastBacktestConfig {
        fee_rate,
        slippage_bps,
        init_cash,
    };

    let res = py.allow_threads(move || run_fast_backtest_impl(&w_mat, &r_mat, &cfg));

    let out = PyDict::new_bound(py);
    out.set_item("equity", PyArray1::from_vec_bound(py, res.equity))?;
    out.set_item(
        "daily_returns",
        PyArray1::from_vec_bound(py, res.daily_returns),
    )?;
    out.set_item("turnover", PyArray1::from_vec_bound(py, res.turnover))?;
    out.set_item("fees", PyArray1::from_vec_bound(py, res.fees))?;
    out.set_item("slippage", PyArray1::from_vec_bound(py, res.slippage))?;

    let m = PyDict::new_bound(py);
    m.set_item("total_return", res.total_return)?;
    m.set_item("annual_return", res.annual_return)?;
    m.set_item("annual_volatility", res.annual_volatility)?;
    m.set_item("sharpe", res.sharpe)?;
    m.set_item("sortino", res.sortino)?;
    m.set_item("calmar", res.calmar)?;
    m.set_item("max_drawdown", res.max_drawdown)?;
    m.set_item("max_drawdown_days", res.max_drawdown_days)?;
    m.set_item("win_rate", res.win_rate)?;
    m.set_item("daily_turnover_mean", res.daily_turnover_mean)?;
    m.set_item("annual_turnover", res.annual_turnover)?;
    out.set_item("metrics", m)?;

    Ok(out)
}

/// 期货实盘离散手数与逐日盯市回测函数。
#[allow(clippy::too_many_arguments)]
#[pyfunction]
#[pyo3(signature = (
    target_weights,
    exec_price,
    exec_price_old,
    settle_price,
    limit_up,
    limit_down,
    is_unilateral,
    is_roll,
    specs,
    init_cash=10000000.0,
    max_margin_ratio=0.50,
    margin_safety_multiplier=1.1,
    lot_rounding="round"
))]
pub fn run_futures_backtest<'py>(
    py: Python<'py>,
    target_weights: PyReadonlyArray2<'_, f64>,
    exec_price: PyReadonlyArray2<'_, f64>,
    exec_price_old: PyReadonlyArray2<'_, f64>,
    settle_price: PyReadonlyArray2<'_, f64>,
    limit_up: PyReadonlyArray2<'_, f64>,
    limit_down: PyReadonlyArray2<'_, f64>,
    is_unilateral: PyReadonlyArray2<'_, bool>,
    is_roll: PyReadonlyArray2<'_, bool>,
    specs: Vec<Bound<'py, PyDict>>,
    init_cash: f64,
    max_margin_ratio: f64,
    margin_safety_multiplier: f64,
    lot_rounding: &str,
) -> PyResult<Bound<'py, PyDict>> {
    let w_mat = view_to_fortran_f64(target_weights.as_array());
    let ep_mat = view_to_fortran_f64(exec_price.as_array());
    let epo_mat = view_to_fortran_f64(exec_price_old.as_array());
    let sp_mat = view_to_fortran_f64(settle_price.as_array());
    let lu_mat = view_to_fortran_f64(limit_up.as_array());
    let ld_mat = view_to_fortran_f64(limit_down.as_array());
    let iu_mat = view_to_fortran_bool(is_unilateral.as_array());
    let ir_mat = view_to_fortran_bool(is_roll.as_array());

    let (t, n) = w_mat.dim();
    if ep_mat.dim() != (t, n)
        || epo_mat.dim() != (t, n)
        || sp_mat.dim() != (t, n)
        || lu_mat.dim() != (t, n)
        || ld_mat.dim() != (t, n)
        || iu_mat.dim() != (t, n)
        || ir_mat.dim() != (t, n)
    {
        return Err(PyValueError::new_err(
            "所有行情与执行矩阵维度必须完全一致 (T, N)",
        ));
    }

    if specs.len() != n {
        return Err(PyValueError::new_err(format!(
            "specs 长度 ({}) 必须与标的列数 N ({}) 严格相等",
            specs.len(),
            n
        )));
    }

    let mut rust_specs = Vec::with_capacity(n);
    for d in specs {
        let symbol: String = d
            .get_item("symbol")?
            .map(|v| v.extract())
            .unwrap_or(Ok("".to_string()))?;
        let multiplier: f64 = d
            .get_item("multiplier")?
            .map(|v| v.extract())
            .unwrap_or(Ok(10.0))?;
        let margin_rate: f64 = d
            .get_item("margin_rate")?
            .map(|v| v.extract())
            .unwrap_or(Ok(0.10))?;
        let fee_type_str: String = d
            .get_item("fee_type")?
            .map(|v| v.extract())
            .unwrap_or(Ok("rate".to_string()))?;
        let open_fee: f64 = d
            .get_item("open_fee")?
            .map(|v| v.extract())
            .unwrap_or(Ok(0.00005))?;
        let close_fee: f64 = d
            .get_item("close_fee")?
            .map(|v| v.extract())
            .unwrap_or(Ok(0.00005))?;
        let close_today_fee: f64 = d
            .get_item("close_today_fee")?
            .map(|v| v.extract())
            .unwrap_or(Ok(close_fee))?;
        let tick_size: f64 = d
            .get_item("tick_size")?
            .map(|v| v.extract())
            .unwrap_or(Ok(1.0))?;
        let slip_ticks: f64 = d
            .get_item("slip_ticks")?
            .map(|v| v.extract())
            .unwrap_or(Ok(1.0))?;

        rust_specs.push(SymbolSpec {
            symbol,
            multiplier,
            margin_rate,
            fee_type: FeeType::parse(&fee_type_str),
            open_fee,
            close_fee,
            close_today_fee,
            tick_size,
            slip_ticks,
        });
    }

    let cfg = FuturesBacktestConfig {
        init_cash,
        max_margin_ratio,
        margin_safety_multiplier,
        lot_rounding: RoundingMode::parse(lot_rounding),
    };

    let res = py.allow_threads(move || {
        run_futures_backtest_impl(
            &w_mat,
            &ep_mat,
            &epo_mat,
            &sp_mat,
            &lu_mat,
            &ld_mat,
            &iu_mat,
            &ir_mat,
            &rust_specs,
            &cfg,
        )
    });

    let out = PyDict::new_bound(py);
    out.set_item("equity", PyArray1::from_vec_bound(py, res.equity))?;
    out.set_item("returns", PyArray1::from_vec_bound(py, res.returns))?;
    out.set_item("cash", PyArray1::from_vec_bound(py, res.cash))?;
    out.set_item("margin_used", PyArray1::from_vec_bound(py, res.margin_used))?;
    out.set_item("leverage", PyArray1::from_vec_bound(py, res.leverage))?;
    out.set_item("gross_pnl", PyArray1::from_vec_bound(py, res.gross_pnl))?;
    out.set_item("net_pnl", PyArray1::from_vec_bound(py, res.net_pnl))?;
    out.set_item("fee", PyArray1::from_vec_bound(py, res.fee))?;
    out.set_item("slippage", PyArray1::from_vec_bound(py, res.slippage))?;
    out.set_item("roll_cost", PyArray1::from_vec_bound(py, res.roll_cost))?;
    out.set_item("turnover", PyArray1::from_vec_bound(py, res.turnover))?;
    out.set_item(
        "positions",
        PyArray2::from_owned_array_bound(py, res.positions),
    )?;

    let m = PyDict::new_bound(py);
    m.set_item("total_return", res.metrics.total_return)?;
    m.set_item("annual_return", res.metrics.annual_return)?;
    m.set_item("annual_volatility", res.metrics.annual_volatility)?;
    m.set_item("sharpe", res.metrics.sharpe)?;
    m.set_item("sortino", res.metrics.sortino)?;
    m.set_item("calmar", res.metrics.calmar)?;
    m.set_item("max_drawdown", res.metrics.max_drawdown)?;
    m.set_item("max_drawdown_days", res.metrics.max_drawdown_days)?;
    m.set_item("win_rate", res.metrics.win_rate)?;
    m.set_item("profit_loss_ratio", res.metrics.profit_loss_ratio)?;
    m.set_item("daily_turnover_mean", res.metrics.daily_turnover_mean)?;
    m.set_item("annual_turnover", res.metrics.annual_turnover)?;
    m.set_item("total_fees", res.metrics.total_fees)?;
    m.set_item("total_slippage", res.metrics.total_slippage)?;
    m.set_item("total_roll_cost", res.metrics.total_roll_cost)?;
    m.set_item("max_margin_utilization", res.metrics.max_margin_utilization)?;
    m.set_item("max_leverage", res.metrics.max_leverage)?;
    out.set_item("metrics", m)?;

    Ok(out)
}

/// 执行单个声明式后处理算子（12 项 C++ 精度对齐算子，全程释放 Python GIL）。
#[allow(clippy::too_many_arguments)]
#[pyfunction]
#[pyo3(signature = (op_name, x1, x2=None, exposures=None, param=None))]
pub fn apply_postprocess_op<'py>(
    py: Python<'py>,
    op_name: &str,
    x1: PyReadonlyArray2<'_, f64>,
    x2: Option<PyReadonlyArray2<'_, f64>>,
    exposures: Option<Vec<PyReadonlyArray2<'_, f64>>>,
    param: Option<f64>,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let m1 = view_to_fortran_f64(x1.as_array());
    let (t, n) = m1.dim();

    let m2_opt = match x2 {
        Some(arr) => {
            let m = view_to_fortran_f64(arr.as_array());
            if m.dim() != (t, n) {
                return Err(PyValueError::new_err(format!(
                    "后处理算子 '{op_name}' 的 x2 维度 {:?} 与 x1 维度 ({t}, {n}) 不一致",
                    m.dim()
                )));
            }
            Some(m)
        }
        None => None,
    };

    let exp_mats: Vec<Array2<f64>> = match exposures {
        Some(list) => {
            let mut v = Vec::with_capacity(list.len());
            for arr in list {
                let m = view_to_fortran_f64(arr.as_array());
                if m.dim() != (t, n) {
                    return Err(PyValueError::new_err(format!(
                        "后处理算子 '{op_name}' 的暴露矩阵维度 {:?} 与 x1 维度 ({t}, {n}) 不一致",
                        m.dim()
                    )));
                }
                v.push(m);
            }
            v
        }
        None => Vec::new(),
    };

    let op_owned = op_name.to_string();
    let res_mat: Result<Array2<f64>, String> = py.allow_threads(move || match op_owned.as_str() {
        "baseUniMask" | "trdMask" => {
            let m2 = m2_opt
                .as_ref()
                .ok_or_else(|| format!("算子 '{op_owned}' 需要提供掩码矩阵 x2"))?;
            Ok(pp_mask_mul(&m1, m2))
        }
        "dropNewStock" => {
            let m2 = m2_opt
                .as_ref()
                .ok_or_else(|| "算子 'dropNewStock' 需要提供上市天数矩阵 x2".to_string())?;
            let min_days = param.unwrap_or(60.0);
            Ok(pp_drop_new_stock(&m1, m2, min_days))
        }
        "MAD" => {
            let threshold = param.unwrap_or(3.0);
            Ok(pp_mad(&m1, threshold))
        }
        "winsor" => {
            let quantile = param.unwrap_or(0.025);
            Ok(pp_winsor(&m1, quantile))
        }
        "indNeut" => {
            let m2 = m2_opt
                .as_ref()
                .ok_or_else(|| "算子 'indNeut' 需要提供板块/行业编码矩阵 x2".to_string())?;
            Ok(pp_ind_neut(&m1, m2))
        }
        "riskNeut" => {
            if !exp_mats.is_empty() {
                let mut curr = m1;
                for exp in &exp_mats {
                    curr = pp_risk_neut(&curr, exp);
                }
                Ok(curr)
            } else {
                let m2 = m2_opt
                    .as_ref()
                    .ok_or_else(|| "算子 'riskNeut' 需要提供风险暴露矩阵 x2".to_string())?;
                Ok(pp_risk_neut(&m1, m2))
            }
        }
        "riskNeutMulti" => {
            if !exp_mats.is_empty() {
                let refs: Vec<&Array2<f64>> = exp_mats.iter().collect();
                Ok(pp_risk_neut_multi(&m1, &refs))
            } else if let Some(ref m2) = m2_opt {
                let refs = vec![m2];
                Ok(pp_risk_neut_multi(&m1, &refs))
            } else {
                Err("算子 'riskNeutMulti' 需要提供至少一个风险暴露矩阵".to_string())
            }
        }
        "zScore" => Ok(pp_zscore(&m1)),
        "rank" => Ok(pp_rank(&m1)),
        "decay" => {
            let w = param.unwrap_or(5.0).max(1.0) as usize;
            Ok(pp_decay(&m1, w))
        }
        "emadecay" => {
            let w = param.unwrap_or(5.0).max(1.0) as usize;
            Ok(pp_emadecay(&m1, w))
        }
        _ => Err(format!("未知的后处理算子: '{op_owned}'")),
    });

    let out = res_mat.map_err(PyValueError::new_err)?;
    Ok(PyArray2::from_owned_array_bound(py, out))
}
