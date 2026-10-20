//! NaN 安全浮点实用函数与增量滑动窗口聚合状态机 (nan.rs)。
//!
//! 核心设计：
//! 1. 统一将 `NaN`、`+Inf`、`-Inf` 视为缺失值（标准化为 `f64::NAN`），防止无穷大污染增量累加器；
//! 2. 实现基于双栈队列（Two-Stack Queue / SWAG）的 Chan 并行 Welford 增量状态机：
//!    - 完全消除传统滑窗移除旧值时的减法灾难性抵消（Catastrophic Cancellation）；
//!    - 单步时间复杂度严格均摊 $O(1)$，与窗口长度 $d$ 无关；
//!    - 数值精度与每次在窗口切片上重算完全一致（误差 $< 10^{-13}$）。

use std::cmp::Ordering;

pub const EPS: f64 = 1e-15;
pub const STD_EPS: f64 = 1e-12;
pub const VAR_EPS: f64 = 1e-24;

/// 若输入为有限浮点数则原样返回，否则（NaN 或 ±Inf）统一返回 `f64::NAN`。
#[inline(always)]
pub fn clean_f64(x: f64) -> f64 {
    if x.is_finite() {
        x
    } else {
        f64::NAN
    }
}

/// 判断浮点数是否为有效有限值。
#[inline(always)]
pub fn is_valid(x: f64) -> bool {
    x.is_finite()
}

/// 安全的浮点数全序比较（仅用于已确认 finite 的数值排序）。
#[inline(always)]
pub fn cmp_f64(a: f64, b: f64) -> Ordering {
    a.partial_cmp(&b).unwrap_or(Ordering::Equal)
}

/// 一元 Welford 聚合状态：维护有效样本数 `n`、总和 `sum`、均值 `mean`、二阶中心矩累积 `m2`。
#[derive(Debug, Clone, Copy)]
pub struct WelfordState {
    pub n: usize,
    pub sum: f64,
    pub mean: f64,
    pub m2: f64,
}

impl WelfordState {
    #[inline(always)]
    pub const fn empty() -> Self {
        Self {
            n: 0,
            sum: 0.0,
            mean: 0.0,
            m2: 0.0,
        }
    }

    #[inline(always)]
    pub fn from_value(x: f64) -> Self {
        if x.is_finite() {
            Self {
                n: 1,
                sum: x,
                mean: x,
                m2: 0.0,
            }
        } else {
            Self::empty()
        }
    }

    /// 使用 Chan (1979) 并行公式无减法合并两个不相交子窗口的 Welford 状态。
    #[inline(always)]
    pub fn combine(a: Self, b: Self) -> Self {
        if a.n == 0 {
            return b;
        }
        if b.n == 0 {
            return a;
        }
        let n = a.n + b.n;
        let sum = a.sum + b.sum;
        let nf = n as f64;
        let na = a.n as f64;
        let nb = b.n as f64;
        let delta = b.mean - a.mean;
        let mean = a.mean + delta * (nb / nf);
        let m2 = a.m2 + b.m2 + delta * delta * (na * nb / nf);
        Self { n, sum, mean, m2 }
    }

    #[inline(always)]
    pub fn sample_var(&self) -> f64 {
        if self.n < 2 {
            return f64::NAN;
        }
        let scale = 1.0_f64.max(self.mean.abs());
        let m2_clean = if self.m2 <= VAR_EPS * (self.n as f64) * scale * scale {
            0.0
        } else {
            self.m2
        };
        let v = m2_clean / ((self.n - 1) as f64);
        clean_f64(v)
    }

    #[inline(always)]
    pub fn sample_std(&self) -> f64 {
        let v = self.sample_var();
        if v.is_nan() {
            f64::NAN
        } else {
            v.sqrt()
        }
    }
}

/// 基于双栈（Two-Stack Queue）的 $O(1)$ 均摊滑动窗口 Welford 聚合器。
pub struct TwoStackWelford {
    front_agg: Vec<WelfordState>,
    back_vals: Vec<WelfordState>,
    back_agg: Vec<WelfordState>,
}

impl TwoStackWelford {
    pub fn with_capacity(cap: usize) -> Self {
        Self {
            front_agg: Vec::with_capacity(cap),
            back_vals: Vec::with_capacity(cap),
            back_agg: Vec::with_capacity(cap),
        }
    }

    pub fn clear(&mut self) {
        self.front_agg.clear();
        self.back_vals.clear();
        self.back_agg.clear();
    }

    #[inline(always)]
    pub fn push(&mut self, x: f64) {
        let state = WelfordState::from_value(x);
        self.back_vals.push(state);
        let agg = match self.back_agg.last() {
            Some(&prev) => WelfordState::combine(prev, state),
            None => state,
        };
        self.back_agg.push(agg);
    }

    #[inline(always)]
    pub fn pop(&mut self) {
        if self.front_agg.is_empty() {
            while let Some(state) = self.back_vals.pop() {
                let agg = match self.front_agg.last() {
                    Some(&prev) => WelfordState::combine(state, prev),
                    None => state,
                };
                self.front_agg.push(agg);
            }
            self.back_agg.clear();
        }
        self.front_agg.pop();
    }

    #[inline(always)]
    pub fn query(&self) -> WelfordState {
        match (self.front_agg.last(), self.back_agg.last()) {
            (Some(&f), Some(&b)) => WelfordState::combine(f, b),
            (Some(&f), None) => f,
            (None, Some(&b)) => b,
            (None, None) => WelfordState::empty(),
        }
    }
}

/// 二元协矩 Welford 状态：维护共同有效对数 `n`、`mean_x`、`mean_y`、`m2_x`、`m2_y`、`c_xy`。
#[derive(Debug, Clone, Copy)]
pub struct BivariateWelfordState {
    pub n: usize,
    pub mean_x: f64,
    pub mean_y: f64,
    pub m2_x: f64,
    pub m2_y: f64,
    pub c_xy: f64,
}

impl BivariateWelfordState {
    #[inline(always)]
    pub const fn empty() -> Self {
        Self {
            n: 0,
            mean_x: 0.0,
            mean_y: 0.0,
            m2_x: 0.0,
            m2_y: 0.0,
            c_xy: 0.0,
        }
    }

    #[inline(always)]
    pub fn from_pair(x: f64, y: f64) -> Self {
        if x.is_finite() && y.is_finite() {
            Self {
                n: 1,
                mean_x: x,
                mean_y: y,
                m2_x: 0.0,
                m2_y: 0.0,
                c_xy: 0.0,
            }
        } else {
            Self::empty()
        }
    }

    #[inline(always)]
    pub fn combine(a: Self, b: Self) -> Self {
        if a.n == 0 {
            return b;
        }
        if b.n == 0 {
            return a;
        }
        let n = a.n + b.n;
        let nf = n as f64;
        let na = a.n as f64;
        let nb = b.n as f64;
        let dx = b.mean_x - a.mean_x;
        let dy = b.mean_y - a.mean_y;
        let weight = na * nb / nf;
        Self {
            n,
            mean_x: a.mean_x + dx * (nb / nf),
            mean_y: a.mean_y + dy * (nb / nf),
            m2_x: a.m2_x + b.m2_x + dx * dx * weight,
            m2_y: a.m2_y + b.m2_y + dy * dy * weight,
            c_xy: a.c_xy + b.c_xy + dx * dy * weight,
        }
    }
}

/// 基于双栈的 $O(1)$ 均摊滑动窗口二元协矩聚合器。
pub struct TwoStackBivariateWelford {
    front_agg: Vec<BivariateWelfordState>,
    back_vals: Vec<BivariateWelfordState>,
    back_agg: Vec<BivariateWelfordState>,
}

impl TwoStackBivariateWelford {
    pub fn with_capacity(cap: usize) -> Self {
        Self {
            front_agg: Vec::with_capacity(cap),
            back_vals: Vec::with_capacity(cap),
            back_agg: Vec::with_capacity(cap),
        }
    }

    pub fn clear(&mut self) {
        self.front_agg.clear();
        self.back_vals.clear();
        self.back_agg.clear();
    }

    #[inline(always)]
    pub fn push(&mut self, x: f64, y: f64) {
        let state = BivariateWelfordState::from_pair(x, y);
        self.back_vals.push(state);
        let agg = match self.back_agg.last() {
            Some(&prev) => BivariateWelfordState::combine(prev, state),
            None => state,
        };
        self.back_agg.push(agg);
    }

    #[inline(always)]
    pub fn pop(&mut self) {
        if self.front_agg.is_empty() {
            while let Some(state) = self.back_vals.pop() {
                let agg = match self.front_agg.last() {
                    Some(&prev) => BivariateWelfordState::combine(state, prev),
                    None => state,
                };
                self.front_agg.push(agg);
            }
            self.back_agg.clear();
        }
        self.front_agg.pop();
    }

    #[inline(always)]
    pub fn query(&self) -> BivariateWelfordState {
        match (self.front_agg.last(), self.back_agg.last()) {
            (Some(&f), Some(&b)) => BivariateWelfordState::combine(f, b),
            (Some(&f), None) => f,
            (None, Some(&b)) => b,
            (None, None) => BivariateWelfordState::empty(),
        }
    }
}

/// 计算有限切片上的平均排名（归一化到 $[0, 1]$，若仅 1 个有效元素则为 $0.5$）。
/// `indices_and_vals` 作为临时可重用缓冲区传入以避免循环分配。
pub fn compute_normalized_ranks(
    row: &[f64],
    out: &mut [f64],
    buf: &mut Vec<(usize, f64)>,
) {
    buf.clear();
    for (j, &v) in row.iter().enumerate() {
        if v.is_finite() {
            buf.push((j, v));
        } else {
            out[j] = f64::NAN;
        }
    }
    let k = buf.len();
    if k == 0 {
        return;
    }
    if k == 1 {
        out[buf[0].0] = 0.5;
        return;
    }
    buf.sort_unstable_by(|a, b| cmp_f64(a.1, b.1));
    let denom = (k - 1) as f64;
    let mut i = 0;
    while i < k {
        let vi = buf[i].1;
        let tol = STD_EPS * 1.0_f64.max(vi.abs());
        let mut j = i + 1;
        while j < k && (buf[j].1 - vi).abs() <= tol {
            j += 1;
        }
        // 索引范围 [i, j) 具有相同数值，平均排名为 0.5 * (i + j - 1)
        let avg_rank = 0.5 * ((i + j - 1) as f64) / denom;
        for idx in i..j {
            out[buf[idx].0] = avg_rank;
        }
        i = j;
    }
}
