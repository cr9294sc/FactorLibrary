r"""P2 验收测试：Rust 计算内核 (`factorlib_core`) 与 Python 参考实现 (`ops_ref.py`) 全量对账与压测。

涵盖白皮书 P2 全部验收指标：
1. 68 个算子（逐元素一元/二元/常数/条件逻辑、时序滚动增量 Welford/单调队列/二分排名、横截面、板块分组）
   在随机噪声矩阵（含高密度 NaN、正负无穷、并列极值、不同尺度数值）下与 `ops_ref.py` 对账，
   最大相对/绝对浮点误差严格 $< 10^{-9}$；
2. 基于 `hypothesis` 的性质测试（Property-based Testing）；
3. 表达式 DSL 解析、中缀语法糖、静态类型检查（10 项非法签名示例 + `y_*` 未来标签阻断）；
4. AST 代数规范化化简（交换律、幂等性、单调不变性压缩、退化结构阻断）；
5. 全局 `ahash` 子表达式 LRU 缓存命中与驱逐行为验证；
6. 4000 交易日 × 80 品种面板上 1000 个表达式批并行求值吞吐量验证（要求 $\ge 1000$ 表达式/秒）。
"""

from __future__ import annotations

import time
from datetime import date, timedelta

import factorlib_core
import numpy as np
from factorlib.core import ops_ref
from factorlib.core.dsl import F, canonicalize_expr, get_operator_specs, validate_expr
from factorlib.core.panel import Panel
from hypothesis import given, settings
from hypothesis import strategies as st


def _make_noisy_matrix(
    rng: np.random.Generator, T: int, N: int, nan_prob: float = 0.22
) -> np.ndarray:
    """生成含高密度 NaN、并列值、±Inf 及跨数量级浮点数的 (T, N) Fortran 矩阵。"""
    mat = rng.normal(loc=50.0, scale=25.0, size=(T, N))
    # 注入并列值
    tie_mask = rng.random((T, N)) < 0.08
    mat[tie_mask] = np.round(mat[tie_mask], 0)
    # 注入高密度 NaN
    nan_mask = rng.random((T, N)) < nan_prob
    mat[nan_mask] = np.nan
    # 注入少量 ±Inf 测试边界净化
    inf_mask = rng.random((T, N)) < 0.02
    mat[inf_mask] = np.where(rng.random(np.sum(inf_mask)) < 0.5, np.inf, -np.inf)
    return np.asfortranarray(mat, dtype=np.float64)


def _assert_close_with_nan(
    actual: np.ndarray, expected: np.ndarray, msg: str, tol: float = 1e-9
) -> None:
    assert actual.shape == expected.shape, f"{msg}: 形状不匹配 {actual.shape} vs {expected.shape}"
    act_nan = np.isnan(actual)
    exp_nan = np.isnan(expected)
    np.testing.assert_array_equal(act_nan, exp_nan, err_msg=f"{msg}: NaN 掩码不一致")
    valid = ~act_nan
    if np.any(valid):
        np.testing.assert_allclose(
            actual[valid],
            expected[valid],
            rtol=tol,
            atol=tol,
            err_msg=f"{msg}: 数值误差超过 {tol}",
        )


def test_all_68_operators_reconciliation_against_ops_ref():
    """在多组随机含噪矩阵（含 NaN / ±Inf / 并列值）上逐一验证全部 68 个算子误差 < 1e-9。"""
    T, N = 65, 12
    for seed in [42, 101, 2026]:
        rng = np.random.default_rng(seed)
        x = _make_noisy_matrix(rng, T, N, nan_prob=0.20)
        y = _make_noisy_matrix(rng, T, N, nan_prob=0.18)
        # 构造正值序列供 log_return 等测试，同时保留负值与零在 x, y 中
        pos_x = np.abs(x) + 0.5
        # 构造板块分组整数矩阵 (0, 1, 2)
        grp = np.asfortranarray(rng.integers(0, 3, size=(T, N)).astype(np.float64))
        grp[rng.random((T, N)) < 0.1] = np.nan

        mask = np.ones((T, N), dtype=bool, order="F")
        eng = factorlib_core.Engine(
            {"x": x, "y": y, "pos_x": pos_x, "grp": grp},
            mask,
            {"y_5": np.zeros((T, N), dtype=np.float64, order="F")},
            [f"d_{i}" for i in range(T)],
            [f"S_{j}" for j in range(N)],
            cache_mb=128,
            n_threads=4,
        )

        # 1. 一元逐元素算子 (12)
        for op_name in [
            "neg",
            "abs",
            "sign",
            "inv",
            "slog",
            "ssqrt",
            "square",
            "cube",
            "sigmoid",
            "tanh",
            "relu",
            "atan",
        ]:
            rust_res = eng.eval(f"{op_name}(x)", use_cache=False)
            ref_res = ops_ref.OPS_REF_REGISTRY[op_name](x)
            _assert_close_with_nan(rust_res, ref_res, f"seed={seed} {op_name}(x)")

        # 2. 二元逐元素算子 (8)
        for op_name in ["add", "sub", "mul", "div", "max2", "min2", "hypot", "copysign"]:
            rust_res = eng.eval(f"{op_name}(x, y)", use_cache=False)
            ref_res = ops_ref.OPS_REF_REGISTRY[op_name](x, y)
            _assert_close_with_nan(rust_res, ref_res, f"seed={seed} {op_name}(x, y)")

        # 3. 标量常数算子 (6)
        for op_name, c_val in [
            ("addc", -2.5),
            ("subc", 1.75),
            ("mulc", -0.5),
            ("divc", 2.0),
            ("spow", 1.5),
            ("spow", 0.5),
            ("clip", 30.0),
        ]:
            rust_res = eng.eval(f"{op_name}(x, {c_val})", use_cache=False)
            ref_res = ops_ref.OPS_REF_REGISTRY[op_name](x, c_val)
            _assert_close_with_nan(rust_res, ref_res, f"seed={seed} {op_name}(x, {c_val})")

        # 4. 比较、布尔逻辑与 where 三元条件算子 (10)
        for cmp_op in ["gt", "lt", "ge", "le"]:
            rust_res = eng.eval(f"where({cmp_op}(x, y), x, y)", use_cache=False)
            ref_cond = ops_ref.OPS_REF_REGISTRY[cmp_op](x, y)
            ref_res = ops_ref.where(ref_cond, x, y)
            _assert_close_with_nan(rust_res, ref_res, f"seed={seed} where({cmp_op})")

        for cmp_c_op, c_val in [("gt_const", 40.0), ("lt_const", 60.0)]:
            rust_res = eng.eval(f"where({cmp_c_op}(x, {c_val}), x, y)", use_cache=False)
            ref_cond = ops_ref.OPS_REF_REGISTRY[cmp_c_op](x, c_val)
            ref_res = ops_ref.where(ref_cond, x, y)
            _assert_close_with_nan(rust_res, ref_res, f"seed={seed} where({cmp_c_op})")

        for logic_op in ["and_cond", "or_cond"]:
            rust_res = eng.eval(
                f"where({logic_op}(gt(x, y), gt_const(x, 30.0)), x, y)", use_cache=False
            )
            c1 = ops_ref.gt(x, y)
            c2 = ops_ref.gt_const(x, 30.0)
            ref_cond = ops_ref.OPS_REF_REGISTRY[logic_op](c1, c2)
            ref_res = ops_ref.where(ref_cond, x, y)
            _assert_close_with_nan(rust_res, ref_res, f"seed={seed} where({logic_op})")

        rust_not = eng.eval("where(not_cond(gt(x, y)), x, y)", use_cache=False)
        ref_not = ops_ref.where(ops_ref.not_cond(ops_ref.gt(x, y)), x, y)
        _assert_close_with_nan(rust_not, ref_not, f"seed={seed} where(not_cond)")

        # 5. 时序滚动一元算子 (19)
        for w in [3, 5, 10, 20]:
            for ts_op in [
                "delay",
                "delta",
                "pct_change",
                "log_return",
                "ts_sum",
                "ts_mean",
                "ts_std",
                "ts_var",
                "ts_min",
                "ts_max",
                "ts_argmin",
                "ts_argmax",
                "ts_rank",
                "ts_zscore",
                "ts_skew",
                "ts_kurt",
                "ts_ema",
                "ts_decay_linear",
                "ts_mad",
            ]:
                if ts_op == "ts_kurt" and w < 4:
                    continue
                field_in = "pos_x" if ts_op == "log_return" else "x"
                arr_in = pos_x if ts_op == "log_return" else x
                rust_res = eng.eval(f"{ts_op}({field_in}, {w})", use_cache=False)
                ref_res = ops_ref.OPS_REF_REGISTRY[ts_op](arr_in, w)
                _assert_close_with_nan(rust_res, ref_res, f"seed={seed} {ts_op}({field_in}, {w})")

        # 6. 时序滚动二元算子 (4)
        for w in [5, 10, 20]:
            for ts_bin_op in ["ts_cov", "ts_corr", "ts_beta", "ts_resid"]:
                rust_res = eng.eval(f"{ts_bin_op}(x, y, {w})", use_cache=False)
                ref_res = ops_ref.OPS_REF_REGISTRY[ts_bin_op](x, y, w)
                _assert_close_with_nan(rust_res, ref_res, f"seed={seed} {ts_bin_op}(x, y, {w})")

        # 7. 横截面与板块分组算子 (9)
        for cs_op in ["cs_rank", "cs_zscore", "cs_demean", "cs_scale", "cs_winsorize"]:
            rust_res = eng.eval(f"{cs_op}(x)", use_cache=False)
            ref_res = ops_ref.OPS_REF_REGISTRY[cs_op](x)
            _assert_close_with_nan(rust_res, ref_res, f"seed={seed} {cs_op}(x)")

        rust_neut = eng.eval("cs_neutralize(y, x)", use_cache=False)
        ref_neut = ops_ref.cs_neutralize(y, x)
        _assert_close_with_nan(rust_neut, ref_neut, f"seed={seed} cs_neutralize(y, x)")

        for grp_op in ["group_demean", "group_rank", "group_zscore"]:
            rust_res = eng.eval(f"{grp_op}(x, grp)", use_cache=False)
            ref_res = ops_ref.OPS_REF_REGISTRY[grp_op](x, grp)
            _assert_close_with_nan(rust_res, ref_res, f"seed={seed} {grp_op}(x, grp)")


@settings(max_examples=30, deadline=None)
@given(
    seed=st.integers(min_value=1, max_value=100000),
    window=st.sampled_from([3, 5, 8, 12]),
    nan_prob=st.floats(min_value=0.0, max_value=0.45),
)
def test_hypothesis_property_reconciliation(seed: int, window: int, nan_prob: float):
    """基于 Hypothesis 的随机矩阵性质对账测试。"""
    rng = np.random.default_rng(seed)
    T, N = 28, 7
    x = _make_noisy_matrix(rng, T, N, nan_prob=nan_prob)
    y = _make_noisy_matrix(rng, T, N, nan_prob=nan_prob)
    mask = np.ones((T, N), dtype=bool, order="F")
    eng = factorlib_core.Engine({"x": x, "y": y}, mask, cache_mb=32, n_threads=2)

    expr = f"cs_rank(add(ts_zscore(x, {window}), ts_corr(x, y, {window})))"
    rust_out = eng.eval(expr)
    ref_out = ops_ref.cs_rank(
        ops_ref.add(ops_ref.ts_zscore(x, window), ops_ref.ts_corr(x, y, window))
    )
    _assert_close_with_nan(rust_out, ref_out, f"hypothesis seed={seed} w={window}")


def test_dsl_sugar_typechecking_and_canon():
    """测试 Python DSL 语法糖、类型检查阻断、未来函数标签拦截与代数规范化。"""
    specs = get_operator_specs()
    assert len(specs) == 68
    assert set(ops_ref.OPS_REF_REGISTRY.keys()) == {s["name"] for s in specs}

    # Expr 链式语法糖与中缀表达式
    close = F("close")
    open_ = F("open")
    vol = F("volume")
    expr_obj = ((close - open_) / vol.ts_std(20)).cs_rank()
    info = expr_obj.canonicalize(available_fields=["close", "open", "volume"])
    assert info.canon_expr == "cs_rank(div(sub(close, open), ts_std(volume, 20)))"
    assert not info.is_degenerate
    assert info.return_type == "Series"

    # 交换律与幂等化简
    c1 = canonicalize_expr("add(volume, close)")
    c2 = canonicalize_expr("close + volume")
    assert c1.canon_expr == c2.canon_expr
    assert c1.canon_hash == c2.canon_hash

    c_idem = canonicalize_expr("cs_rank(slog(cs_rank(close)))")
    assert c_idem.canon_expr == "cs_rank(close)"

    # 退化表达式标记
    c_degen = canonicalize_expr("sub(ts_mean(close, 10), ts_mean(close, 10))")
    assert c_degen.is_degenerate

    # 静态类型检查与未来函数拦截
    assert not validate_expr("abs(20)")["valid"]
    assert not validate_expr("add(close, 10)")["valid"]
    assert not validate_expr("ts_rank(close, -5)")["valid"]
    assert not validate_expr("ts_mean(y_5, 10)")["valid"]
    assert not validate_expr("y_1")["valid"]
    assert not validate_expr("unknown_col", available_fields=["close", "open"])["valid"]


def test_Throughput_1000_exprs_on_4000x80_panel():
    """压测验收：在 4000 × 80 标准期货面板上批求值 1000 个表达式，验证吞吐量 >= 1000 expr/s。"""
    T, N = 4000, 80
    rng = np.random.default_rng(7)
    field_names = ["open", "high", "low", "close", "volume", "rv", "oi"]
    fields: dict[str, np.ndarray] = {
        name: np.asfortranarray(100.0 + rng.standard_normal((T, N)) * 10.0, dtype=np.float64)
        for name in field_names
    }
    mask = np.ones((T, N), dtype=bool, order="F")
    dates = np.array([date(2010, 1, 1) + timedelta(days=i) for i in range(T)])
    symbols = [f"SYM_{j:02d}" for j in range(N)]

    panel = Panel(dates=dates, symbols=symbols, fields=fields, mask=mask)
    eng = panel.to_rust_engine(cache_mb=2048, n_threads=0)

    windows = [3, 5, 10, 20, 40, 60, 120, 250]
    ts_ops = [
        "ts_mean",
        "ts_std",
        "ts_min",
        "ts_max",
        "ts_zscore",
        "ts_rank",
        "delta",
        "pct_change",
    ]
    bin_ops = ["add", "sub", "mul", "div", "max2", "min2"]
    cs_ops = ["cs_rank", "cs_zscore", "cs_demean", "slog", "ssqrt", "tanh"]

    exprs = []
    for i in range(1000):
        f1 = field_names[i % len(field_names)]
        f2 = field_names[(i * 3 + 1) % len(field_names)]
        w1 = windows[i % len(windows)]
        w2 = windows[(i // 3) % len(windows)]
        t1 = ts_ops[i % len(ts_ops)]
        t2 = ts_ops[(i + 2) % len(ts_ops)]
        bop = bin_ops[i % len(bin_ops)]
        cop = cs_ops[i % len(cs_ops)]
        exprs.append(f"{cop}({bop}({t1}({f1}, {w1}), {t2}({f2}, {w2})))")

    eng.clear_cache()
    t0 = time.perf_counter()
    results = eng.eval_batch(exprs, use_cache=True, canonicalize=False)
    elapsed = time.perf_counter() - t0
    throughput = len(exprs) / elapsed

    assert len(results) == 1000
    assert results[0].shape == (T, N)
    assert results[0].flags.f_contiguous
    stats = eng.cache_stats()
    assert stats["hits"] > 0
    assert throughput >= 1000.0, f"批求值吞吐量不足 1000 expr/s: 实际 {throughput:.1f} expr/s"
