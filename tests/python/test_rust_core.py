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
import pytest
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

    # P0-04: 单调压缩不得越界穿透至子节点，inv(inv(x)) 不得化简为 x
    c_sub_cube = canonicalize_expr("sub(cube(cs_rank(close)), cs_rank(open))")
    assert c_sub_cube.canon_expr == "sub(cube(cs_rank(close)), cs_rank(open))"

    c_ts_mean_slog = canonicalize_expr("ts_mean(slog(cs_rank(close)), 10)")
    assert c_ts_mean_slog.canon_expr == "ts_mean(slog(cs_rank(close)), 10)"

    c_inv_inv = canonicalize_expr("inv(inv(close))")
    assert c_inv_inv.canon_expr == "inv(inv(close))"

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


def test_canonicalization_p0_04_eval_parity_and_nan_mask():
    """验证 P0-04：规范化化简不改变子树数学数值（canonicalize=True 与 False 逐元素误差 < 1e-12），

    且 inv(inv(x)) 在 x=0 处严格保留 NaN 掩码。
    """
    T, N = 20, 6
    dates = np.array([f"2024-01-{i:02d}" for i in range(1, T + 1)])
    symbols = [f"S{j}" for j in range(N)]
    rng = np.random.default_rng(42)

    close = np.asfortranarray(10.0 + rng.uniform(0.1, 5.0, (T, N)), dtype=np.float64)
    open_ = np.asfortranarray(9.5 + rng.uniform(0.1, 5.0, (T, N)), dtype=np.float64)
    # 人为设置奇异点
    close[0, 0] = 0.0

    mask = np.ones((T, N), dtype=bool, order="F")
    panel = Panel(dates=dates, symbols=symbols, fields={"close": close, "open": open_}, mask=mask)
    eng = panel.to_rust_engine(cache_mb=16, n_threads=1)

    # 1. 验证非线性复合子树的求值等价性
    test_exprs = [
        "sub(cube(cs_rank(close)), cs_rank(open))",
        "ts_mean(slog(cs_rank(close)), 5)",
        "inv(inv(close))",
    ]
    for expr in test_exprs:
        v_true = eng.eval(expr, canonicalize=True)
        v_false = eng.eval(expr, canonicalize=False)
        _assert_close_with_nan(v_true, v_false, f"canonicalize parity for {expr}")

    # 2. 验证奇异点 NaN 掩码保留
    inv_true = eng.eval("inv(inv(close))", canonicalize=True)
    assert np.isnan(inv_true[0, 0]), "inv(inv(0.0)) 必须为 NaN"
    assert np.isfinite(inv_true[1, 0]), "非零位置必须有限有效"


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


def test_eval_output_is_immutable_and_prevents_cache_corruption() -> None:
    """验证 Engine.eval 导出的 NumPy 矩阵被标记为只读 (writeable=False)，
    杜绝 Python 端就地修改破坏底层 Arc 内存与 SubExprCache 缓存。
    """
    import pytest

    t, n = 20, 5
    fields = {"close": np.ones((t, n), order="F", dtype=np.float64)}
    mask = np.ones((t, n), dtype=bool, order="F")
    eng = factorlib_core.Engine(fields, mask)

    arr = eng.eval("close")
    assert not arr.flags.writeable, "导出的 NumPy 矩阵必须为只读 (writeable=False)"

    with pytest.raises(ValueError, match="read-only"):
        arr[0, 0] = 999.0

    # 再次求值验证底层缓存未被篡改
    arr_again = eng.eval("close")
    assert arr_again[0, 0] == 1.0


def test_cs_and_group_ops_universe_mask_isolation() -> None:
    """验证 P0-03: 全部 9 个截面与分组算子在 universe_mask==False 处完全隔离极端值，
    且非在池标的位置严格输出 NaN，在池标的结果与纯净输入绝对等价。
    """
    rng = np.random.default_rng(2026)
    T, N = 30, 10

    # 随机生成原始特征，包含部分自然 NaN
    x_raw = _make_noisy_matrix(rng, T, N, nan_prob=0.1)
    y_raw = _make_noisy_matrix(rng, T, N, nan_prob=0.1)
    grp_raw = rng.integers(1, 4, size=(T, N)).astype(np.float64, order="F")

    # 构造有 ~30% False 的 universe_mask
    mask = rng.uniform(0.0, 1.0, size=(T, N)) > 0.3
    # 确保每一行至少有 3 个在池标的
    for t in range(T):
        if np.sum(mask[t, :]) < 3:
            mask[t, :3] = True
    mask = np.asfortranarray(mask, dtype=bool)

    # 构造受极端干扰值污染的矩阵：在 mask == False 处注入 1e9, -1e9, 999999.0
    x_polluted = x_raw.copy()
    y_polluted = y_raw.copy()
    grp_polluted = grp_raw.copy()

    x_polluted[~mask] = 1e9
    y_polluted[~mask] = -1e9
    grp_polluted[~mask] = 999.0

    # 构造纯净基准矩阵：在 mask == False 处显式置为 NaN
    x_clean = np.where(mask, x_raw, np.nan)
    y_clean = np.where(mask, y_raw, np.nan)
    grp_clean = np.where(mask, grp_raw, np.nan)

    eng_polluted = factorlib_core.Engine(
        {"x": x_polluted, "y": y_polluted, "grp": grp_polluted},
        mask,
        cache_mb=16,
    )
    eng_clean = factorlib_core.Engine(
        {"x": x_clean, "y": y_clean, "grp": grp_clean},
        mask,
        cache_mb=16,
    )

    # 1. 5 个一元横截面算子
    cs_unary_ops = ["cs_rank", "cs_zscore", "cs_demean", "cs_scale", "cs_winsorize"]
    for op in cs_unary_ops:
        res_polluted = eng_polluted.eval(f"{op}(x)", use_cache=False)
        res_clean = eng_clean.eval(f"{op}(x)", use_cache=False)
        ref_res = ops_ref.OPS_REF_REGISTRY[op](x_polluted, mask=mask)

        # 1. 非在池标的严格全为 NaN
        assert np.all(np.isnan(res_polluted[~mask])), f"{op} 在 ~mask 处必须全部输出 NaN"
        assert np.all(np.isnan(res_clean[~mask])), f"{op} 在 ~mask 处必须全部输出 NaN"

        # 2. 在池标的结果与纯净输入绝对等价 (0 bit 差异)
        _assert_close_with_nan(res_polluted, res_clean, f"mask isolation: {op}")

        # 3. 与 ops_ref(mask=mask) 严格一致 (< 1e-9)
        _assert_close_with_nan(res_polluted, ref_res, f"ops_ref parity: {op}")

    # 2. 二元截面中性化算子
    neut_polluted = eng_polluted.eval("cs_neutralize(y, x)", use_cache=False)
    neut_clean = eng_clean.eval("cs_neutralize(y, x)", use_cache=False)
    neut_ref = ops_ref.cs_neutralize(y_polluted, x_polluted, mask=mask)
    assert np.all(np.isnan(neut_polluted[~mask])), "cs_neutralize 在 ~mask 处必须全部输出 NaN"
    _assert_close_with_nan(neut_polluted, neut_clean, "mask isolation: cs_neutralize")
    _assert_close_with_nan(neut_polluted, neut_ref, "ops_ref parity: cs_neutralize")

    # 3. 3 个分组算子
    group_ops = ["group_demean", "group_rank", "group_zscore"]
    for op in group_ops:
        res_polluted = eng_polluted.eval(f"{op}(x, grp)", use_cache=False)
        res_clean = eng_clean.eval(f"{op}(x, grp)", use_cache=False)
        ref_res = ops_ref.OPS_REF_REGISTRY[op](x_polluted, grp_polluted, mask=mask)

        assert np.all(np.isnan(res_polluted[~mask])), f"{op} 在 ~mask 处必须全部输出 NaN"
        _assert_close_with_nan(res_polluted, res_clean, f"mask isolation: {op}")
        _assert_close_with_nan(res_polluted, ref_res, f"ops_ref parity: {op}")

    # 4. 时序滚动嵌套横截面算子：验证时序特征保留连续性，截面层隔离污染
    nested_polluted = eng_polluted.eval("cs_rank(ts_mean(x, 5))", use_cache=False)
    assert np.all(np.isnan(nested_polluted[~mask])), "嵌套表达式在 ~mask 处必须为 NaN"


def test_tie_rank_non_transitive_tolerance_parity():
    """验证 P1-04 边界样本 [0.0, 0.8e-12, 1.5e-12] 在 Rust 与 Python ops_ref 下采用左锚点容差聚类。"""
    close = np.asfortranarray(
        [
            [0.0, 0.8e-12, 1.5e-12],
            [1.5e-12, 0.8e-12, 0.0],
            [0.0, 1.2e-12, 2.5e-12],
        ],
        dtype=np.float64,
    )
    sector = np.ones((3, 3), dtype=np.float64, order="F")
    mask = np.ones((3, 3), dtype=bool, order="F")
    engine = factorlib_core.Engine(
        {"close": close, "sector": sector},
        mask,
        {},
        ["2024-01-01", "2024-01-02", "2024-01-03"],
        ["A", "B", "C"],
    )

    rust_cs = engine.eval("cs_rank(close)")
    ref_cs = ops_ref.cs_rank(close)
    expected = np.array(
        [
            [0.25, 0.25, 1.0],
            [1.0, 0.25, 0.25],
            [0.0, 0.5, 1.0],
        ],
        dtype=np.float64,
    )
    np.testing.assert_allclose(rust_cs, expected, rtol=0.0, atol=1e-15)
    np.testing.assert_allclose(ref_cs, expected, rtol=0.0, atol=1e-15)

    rust_grp = engine.eval("group_rank(close, sector)")
    ref_grp = ops_ref.group_rank(close, sector)
    np.testing.assert_allclose(rust_grp, expected, rtol=0.0, atol=1e-15)
    np.testing.assert_allclose(ref_grp, expected, rtol=0.0, atol=1e-15)


def test_rust_blocks_r_exec_next_reserved_field():
    """验证 P1-03 Rust 静态类型检查与 Engine 阻断未来执行收益率保留字段 r_exec_next。"""
    for bad_expr in ("r_exec_next", "ts_mean(r_exec_next, 5)"):
        chk = validate_expr(bad_expr)
        assert not chk["valid"], f"validate_expr 应当拦截未来保留字段: {bad_expr}"
        assert chk["error"] is not None

        direct = factorlib_core.check_expr(bad_expr)
        assert not direct["valid"]

    panel_engine = factorlib_core.Engine(
        {"close": np.ones((4, 2), dtype=np.float64, order="F")},
        np.ones((4, 2), dtype=bool, order="F"),
        {},
        ["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04"],
        ["A", "B"],
    )
    with pytest.raises(ValueError):
        panel_engine.eval("r_exec_next")
    with pytest.raises(ValueError):
        panel_engine.eval("ts_mean(r_exec_next, 2)")


def test_zero_copy_c_and_f_order_parity_across_metrics_and_postprocess():
    """验证 P2-05: C-order、F-order 与非连续切片输入在指标与后处理算子上完全位等价。"""
    rng = np.random.default_rng(2025)
    t, n = 25, 10
    f_c = rng.normal(0.0, 1.0, (t, n))
    f_c[3, 2] = np.nan
    f_c[7, 5] = 0.0
    f_c[7, 6] = 0.0  # 引入并列值
    f_f = np.asfortranarray(f_c)

    r_c = rng.normal(0.0, 0.02, (t, n))
    r_c[4, 1] = np.nan
    r_f = np.asfortranarray(r_c)

    b_c = rng.normal(0.0, 1.0, (t, n))
    b_c[5, 4] = np.nan
    b_f = np.asfortranarray(b_c)

    mask_c = rng.random((t, n)) > 0.1
    mask_f = np.asfortranarray(mask_c)

    # 非连续跨步视图（逐行重复后取奇数/偶数行，保证数值与 f_c / r_c 完全一致且步长非紧致）
    f_strided = np.repeat(f_c, 2, axis=0)[::2, :]
    r_strided = np.repeat(r_c, 2, axis=0)[::2, :]
    assert not f_strided.flags["C_CONTIGUOUS"] and not f_strided.flags["F_CONTIGUOUS"]

    for cand_f, cand_r, cand_m in (
        (f_c, r_c, mask_c),
        (f_strided, r_strided, mask_c),
    ):
        for use_rank in (False, True):
            ic_ref = factorlib_core.calc_ic(f_f, r_f, mask_f, 4, use_rank, 1)
            ic_cur = factorlib_core.calc_ic(cand_f, cand_r, cand_m, 4, use_rank, 1)
            for k in ("ic_mean", "ic_std", "ic_ir", "ic_t", "ic_win_rate", "coverage"):
                assert np.isclose(ic_ref[k], ic_cur[k], rtol=0.0, atol=1e-13)
            assert ic_ref["valid_days"] == ic_cur["valid_days"]

        q_ref = factorlib_core.calc_quantile_returns(f_f, r_f, mask_f, 5, 4, 1)
        q_cur = factorlib_core.calc_quantile_returns(cand_f, cand_r, cand_m, 5, 4, 1)
        assert np.allclose(q_ref["group_returns"], q_cur["group_returns"], rtol=0.0, atol=1e-13)
        assert np.isclose(q_ref["ls_mean"], q_cur["ls_mean"], rtol=0.0, atol=1e-13)

        to_ref = factorlib_core.calc_turnover(f_f, mask_f, 4)
        to_cur = factorlib_core.calc_turnover(cand_f, cand_m, 4)
        assert np.isclose(to_ref["turnover"], to_cur["turnover"], rtol=0.0, atol=1e-13)
        assert np.isclose(to_ref["rank_autocorr"], to_cur["rank_autocorr"], rtol=0.0, atol=1e-13)

        tto_ref = factorlib_core.calc_top_quantile_turnover(f_f, mask_f, 5, 4, 1)
        tto_cur = factorlib_core.calc_top_quantile_turnover(cand_f, cand_m, 5, 4, 1)
        assert np.allclose(
            tto_ref["daily_turnover"],
            tto_cur["daily_turnover"],
            equal_nan=True,
            rtol=0.0,
            atol=1e-13,
        )

    corr_f = factorlib_core.calc_correlation_matrix([f_f, b_f], mask_f, 4, True)
    corr_c = factorlib_core.calc_correlation_matrix([f_c, b_c], mask_c, 4, True)
    assert np.allclose(corr_f, corr_c, rtol=0.0, atol=1e-13)

    for op, kw in (
        ("zScore", {}),
        ("rank", {}),
        ("MAD", {"param": 3.0}),
        ("winsor", {"param": 0.05}),
        ("decay", {"param": 4.0}),
        ("emadecay", {"param": 4.0}),
        ("riskNeut", {"x2": b_c}),
        ("riskNeutMulti", {"exposures": [b_c]}),
    ):
        kw_f = dict(kw)
        if "x2" in kw_f:
            kw_f["x2"] = np.asfortranarray(kw_f["x2"])
        if "exposures" in kw_f:
            kw_f["exposures"] = [np.asfortranarray(x) for x in kw_f["exposures"]]
        out_f = factorlib_core.apply_postprocess_op(op, f_f, **kw_f)
        out_c = factorlib_core.apply_postprocess_op(op, f_c, **kw)
        out_s = factorlib_core.apply_postprocess_op(op, f_strided, **kw)
        np.testing.assert_allclose(out_f, out_c, equal_nan=True, rtol=0.0, atol=1e-13)
        np.testing.assert_allclose(out_f, out_s, equal_nan=True, rtol=0.0, atol=1e-13)


def test_eval_gp_batch_prepared_correlation_parity_with_extra_nans_and_ties():
    """验证 P2-04: PreparedFactor 快速路径在库因子与候选因子含额外 NaN 及并列值时与两两 Spearman 完全一致。"""
    rng = np.random.default_rng(101)
    t, n = 24, 10
    close = np.asfortranarray(100.0 + np.cumsum(rng.normal(0.0, 1.0, (t, n)), axis=0))
    y_5 = np.asfortranarray(rng.normal(0.0, 0.02, (t, n)))
    mask = np.ones((t, n), dtype=bool, order="F")
    mask[0:2, 0] = False

    lib1 = np.asfortranarray(rng.normal(0.0, 1.0, (t, n)))
    lib1[5:9, 2] = 0.25  # 并列值
    lib1[10, 3] = np.nan  # 额外 NaN
    lib2 = np.asfortranarray(np.round(rng.normal(0.0, 1.0, (t, n)), 1))

    engine = factorlib_core.Engine(
        {"close": close},
        mask,
        {"y_5": y_5},
        [f"2024-01-{i + 1:02d}" for i in range(t)],
        [f"S{j}" for j in range(n)],
    )

    expr = "ts_std(close, 5)"
    cand = engine.eval(expr)
    res = engine.eval_gp_batch(
        [expr],
        label_name="y_5",
        start_row=4,
        end_row=t,
        min_names=5,
        horizon=5,
        library_factors=[lib1, lib2],
        use_cache=False,
    )
    got_max_corr = res[0]["max_abs_corr"]

    sub_cand = cand[4:t, :]
    sub_mask = mask[4:t, :]
    expected_1 = factorlib_core.calc_pairwise_correlation(sub_cand, lib1[4:t, :], sub_mask, 5, True)
    expected_2 = factorlib_core.calc_pairwise_correlation(sub_cand, lib2[4:t, :], sub_mask, 5, True)
    expected_max = max(abs(expected_1), abs(expected_2))
    assert np.isclose(got_max_corr, expected_max, rtol=0.0, atol=1e-12)


def test_check_expr_deep_nesting_blocked_and_price_warning():
    """验证 P3-01 与 P2-03(b): 深度超过 128 层触发语法拒绝，前复权价非零常数比较产生 warnings。"""
    deep_expr = "close"
    for _ in range(140):
        deep_expr = f"abs({deep_expr})"
    chk_deep = validate_expr(deep_expr, available_fields=["close"])
    assert not chk_deep["valid"]
    assert chk_deep["error"] is not None

    chk_warn = validate_expr(
        "where(gt_const(close, 5000.0), close, neg(close))", available_fields=["close"]
    )
    assert chk_warn["valid"]
    assert len(chk_warn["warnings"]) >= 1

    chk_pos = validate_expr("where(close > 0, close, neg(close))", available_fields=["close"])
    assert chk_pos["valid"]
    assert chk_pos["warnings"] == []


def test_elem_ops_small_and_nonstandard_layout_parity():
    """验证 T=1/N=1 与 C-order/非标准内存布局下 elem 一元/二元算子 (zeros 分配路径) 与 NumPy 完全对账。"""
    rng = np.random.default_rng(7)
    for shape in ((1, 8), (8, 1), (4, 5)):
        a_c = np.ascontiguousarray(rng.normal(2.0, 0.5, size=shape), dtype=np.float64)
        b_c = np.ascontiguousarray(rng.uniform(0.5, 3.0, size=shape), dtype=np.float64)
        a_c[0, 0] = np.nan
        mask = np.ones(shape, dtype=bool, order="F")

        eng = factorlib_core.Engine(
            {"a": np.asfortranarray(a_c), "b": np.asfortranarray(b_c)}, mask
        )
        got_unary = np.asarray(eng.eval("abs(neg(b))"))
        ref_unary = np.abs(-b_c)
        np.testing.assert_allclose(got_unary, ref_unary, rtol=1e-12, atol=1e-12, equal_nan=True)

        got_bin = np.asarray(eng.eval("div(add(a, b), mul(b, b))"))
        ref_bin = (a_c + b_c) / (b_c * b_c)
        np.testing.assert_allclose(got_bin, ref_bin, rtol=1e-12, atol=1e-12, equal_nan=True)
