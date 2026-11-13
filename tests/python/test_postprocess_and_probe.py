"""FactorLab 12 算子 C++ 级精度对账、静态探针、双模态因果摄动与端到端双层回测测试套件
(tests/python/test_postprocess_and_probe.py)。

涵盖：
1. 12 个声明式后处理算子 (`baseUniMask`, `trdMask`, `dropNewStock`, `MAD`, `winsor`, `indNeut`,
   `riskNeut`, `riskNeutMulti`, `zScore`, `rank`, `decay`, `emadecay`) 的 Rust 内核实现、
   Python NumPy 参考实现与 `Factor/data/catalog_full.json` 锁定 C++ 真值向量的 < 1e-12 精度对账；
2. DSL 兼容别名 (`log`, `sqrt`, `ts_delay`, `ts_delta`, `decay`, `emadecay`) 校验；
3. `probe_factor()` 静态探针 4 级预热回看窗口推导与隐式依赖窗口放大效应验证；
4. `FactorCatalog` 统一目录检索 (`get_context`, `stats`, `search`, `get_entry`)；
5. `expression` (Rust DSL) 与 `code` (Python `def compute(context):`) 双模态 + 后处理链的因果摄动零未来测试；
6. `run_factor_workflow()` 双层回测 (Layer 1 Schema v4 统计归因 + Layer 2 期货实盘逐日盯市) 与版本锁定测试。
"""

from __future__ import annotations

import json
from pathlib import Path

import factorlib_core
import numpy as np
import pytest
from factorlib.core.catalog import FactorCatalog
from factorlib.core.evaluate import run_factor_workflow
from factorlib.core.ops_ref import (
    pp_decay_ref,
    pp_drop_new_stock_ref,
    pp_emadecay_ref,
    pp_ind_neut_ref,
    pp_mad_ref,
    pp_mask_mul_ref,
    pp_rank_ref,
    pp_risk_neut_multi_ref,
    pp_risk_neut_ref,
    pp_winsor_ref,
    pp_zscore_ref,
)
from factorlib.core.panel import Panel
from factorlib.core.probe import (
    FactorDefinition,
    evaluate_factor_definition,
    parse_postprocess_pipeline,
    probe_factor,
)
from factorlib.core.registry import FactorRegistry


def _to_f64_mat(nested: list[list[float | None]]) -> np.ndarray:
    arr = np.array(
        [[np.nan if v is None else float(v) for v in row] for row in nested],
        dtype=np.float64,
    )
    return np.asfortranarray(arr)


def _assert_mat_close(
    actual: np.ndarray, expected: np.ndarray, tol: float = 1e-12, msg: str = ""
) -> None:
    assert actual.shape == expected.shape, f"{msg}: 形状不一致 {actual.shape} != {expected.shape}"
    nan_act = np.isnan(actual)
    nan_exp = np.isnan(expected)
    assert np.array_equal(nan_act, nan_exp), (
        f"{msg}: NaN 掩码不一致\nactual={actual}\nexpected={expected}"
    )
    valid = ~nan_exp
    if np.any(valid):
        max_diff = float(np.max(np.abs(actual[valid] - expected[valid])))
        assert max_diff < tol, (
            f"{msg}: 最大误差 {max_diff:.3e} >= {tol}\nactual={actual}\nexpected={expected}"
        )


def test_postprocess_12_ops_against_locked_cpp_vectors():
    """验证 12 个后处理算子的 Rust 实现与 NumPy 参考实现严格对齐 C++ 真值向量 (< 1e-12)。"""
    catalog_path = Path(__file__).resolve().parent / "fixtures" / "postprocess_vectors.json"
    assert catalog_path.exists(), f"缺少后处理锁定真值向量文件: {catalog_path}"

    data = json.loads(catalog_path.read_text(encoding="utf-8"))
    pp_entries = {e["name"]: e for e in data["postprocess_operators"]}
    assert len(pp_entries) == 12

    param_map: dict[str, float | None] = {
        "trdMask": None,
        "baseUniMask": None,
        "dropNewStock": 3.0,
        "decay": 3.0,
        "emadecay": 3.0,
        "MAD": 2.0,
        "winsor": 0.25,
        "zScore": None,
        "rank": None,
        "indNeut": None,
        "riskNeut": None,
        "riskNeutMulti": None,
    }

    for op_name, param_val in param_map.items():
        ex = pp_entries[op_name]["documentation"]["examples"][0]
        x1 = _to_f64_mat(ex["inputs"]["x1"])
        x2 = _to_f64_mat(ex["inputs"]["x2"]) if "x2" in ex["inputs"] else None
        expected = _to_f64_mat(ex["output"])

        # 1. 调用 Rust 内核
        if op_name == "riskNeutMulti":
            assert x2 is not None
            rust_out = np.asarray(factorlib_core.apply_postprocess_op(op_name, x1, exposures=[x2]))
        else:
            rust_out = np.asarray(
                factorlib_core.apply_postprocess_op(op_name, x1, x2=x2, param=param_val)
            )
        _assert_mat_close(rust_out, expected, tol=1e-12, msg=f"Rust vs C++ [{op_name}]")

        # 2. 调用 Python NumPy 参考实现
        if op_name in ("trdMask", "baseUniMask"):
            assert x2 is not None
            ref_out = pp_mask_mul_ref(x1, x2)
        elif op_name == "dropNewStock":
            assert x2 is not None
            ref_out = pp_drop_new_stock_ref(x1, x2, float(param_val or 3.0))
        elif op_name == "decay":
            ref_out = pp_decay_ref(x1, int(param_val or 3))
        elif op_name == "emadecay":
            ref_out = pp_emadecay_ref(x1, int(param_val or 3))
        elif op_name == "MAD":
            ref_out = pp_mad_ref(x1, float(param_val or 2.0))
        elif op_name == "winsor":
            ref_out = pp_winsor_ref(x1, float(param_val or 0.25))
        elif op_name == "zScore":
            ref_out = pp_zscore_ref(x1)
        elif op_name == "rank":
            ref_out = pp_rank_ref(x1)
        elif op_name == "indNeut":
            assert x2 is not None
            ref_out = pp_ind_neut_ref(x1, x2)
        elif op_name == "riskNeut":
            assert x2 is not None
            ref_out = pp_risk_neut_ref(x1, x2)
        elif op_name == "riskNeutMulti":
            assert x2 is not None
            ref_out = pp_risk_neut_multi_ref(x1, [x2])
        else:
            raise AssertionError(f"未覆盖算子: {op_name}")

        _assert_mat_close(ref_out, expected, tol=1e-12, msg=f"NumPy Ref vs C++ [{op_name}]")
        _assert_mat_close(rust_out, ref_out, tol=1e-12, msg=f"Rust vs NumPy Ref [{op_name}]")


def test_risk_neut_multi_multiple_exposures_orthogonality():
    """验证多元无截距 OLS 中性化 (riskNeutMulti) 在多风格暴露下的残差正交性。"""
    rng = np.random.default_rng(42)
    T, N = 10, 30
    e1 = np.asfortranarray(rng.standard_normal((T, N)))
    e2 = np.asfortranarray(rng.standard_normal((T, N)))
    noise = rng.standard_normal((T, N)) * 0.1
    y = np.asfortranarray(2.5 * e1 - 1.3 * e2 + noise)

    rust_res = np.asarray(
        factorlib_core.apply_postprocess_op("riskNeutMulti", y, exposures=[e1, e2])
    )
    ref_res = pp_risk_neut_multi_ref(y, [e1, e2])
    _assert_mat_close(rust_res, ref_res, tol=1e-12, msg="riskNeutMulti 2-exposure parity")

    # 残差应与 e1, e2 在每一行上正交 (点积接近 0，受 C++ 1e-8 对角正则化影响量级约 1e-8)
    for t in range(T):
        dot1 = float(np.dot(rust_res[t], e1[t]))
        dot2 = float(np.dot(rust_res[t], e2[t]))
        assert abs(dot1) < 1e-7
        assert abs(dot2) < 1e-7


def test_dsl_aliases_and_static_probe_window_amplification():
    """测试 DSL 兼容别名及 probe_factor 4 级预热回看窗口和隐式依赖放大效应。"""
    defn = FactorDefinition(
        name="alias_and_amp_test",
        factor_type="expression",
        expression="ts_delta(log(close), 5) + emadecay(sqrt(volume), 10)",
        postprocess=parse_postprocess_pipeline(
            [
                {"op": "baseUniMask"},
                {"op": "trdMask"},
                {"op": "MAD", "param": 3.0},
                {"op": "indNeut"},
                {"op": "riskNeutMulti", "fields": ["SIZE", "CARRY"]},
                {"op": "zScore"},
                {"op": "decay", "window": 5},
            ]
        ),
    )

    rep = probe_factor(defn)
    assert rep["status"] == "ok"
    assert rep["errors"] == []

    # 显式依赖检查: close 需要 6 (ts_delta 5 -> 5+1), volume 需要 10 (emadecay 10)
    exp_vars = {v["name"]: v["window"] for v in rep["dependencies"]["variables"]}
    assert exp_vars["close"] == 6
    assert exp_vars["volume"] == 10

    # 后处理中包含末端 decay(5)，因此 acc_smooth = 4
    # 前置隐式依赖:
    # - in_universe_mask: 1 + 4 = 5
    # - limit_mask: 1 + 4 = 5
    # - sector: 1 + 4 = 5
    # - SIZE (自身 lookback=20): 20 + 4 = 24
    # - CARRY (自身 lookback=1): 1 + 4 = 5
    pp_vars = {v["name"]: v["window"] for v in rep["postprocess_dependencies"]["variables"]}
    assert pp_vars["in_universe_mask"] == 5
    assert pp_vars["limit_mask"] == 5
    assert pp_vars["sector"] == 5
    assert pp_vars["SIZE"] == 24
    assert pp_vars["CARRY"] == 5

    req = rep["execution_requirements"]
    assert req["expression_lookback"] == 10
    assert req["postprocess_lookback"] == 5
    assert req["total_required_lookback"] == 24
    assert req["warmup_bars"] == 23


def test_probe_factor_blocks_lookahead_and_unsafe_imports():
    """测试静态探针拦截未来函数与非法 Python 导入。"""
    bad_dsl = FactorDefinition(
        name="bad_dsl",
        factor_type="expression",
        expression="delay(close, -1)",
    )
    rep_dsl = probe_factor(bad_dsl)
    assert rep_dsl["status"] == "error"
    assert len(rep_dsl["errors"]) > 0

    bad_py = FactorDefinition(
        name="bad_py",
        factor_type="code",
        code="import os\ndef compute(context):\n    return context.history('close', 5)[-1]",
    )
    rep_py = probe_factor(bad_py)
    assert rep_py["status"] == "error"
    assert any("禁止导入外部模块" in e for e in rep_py["errors"])


def test_catalog_context_and_search():
    """测试 FactorCatalog 的 Stage 0 与 Stage 1 接口。"""
    cat = FactorCatalog()
    ctx = cat.get_context()
    assert ctx["counts"]["postprocess"] == 12
    assert ctx["counts"]["styles"] == 6
    assert ctx["counts"]["functions"] >= 68

    pp_list = cat.search(catalog="postprocess", category="neutralize")
    names = {it["name"] for it in pp_list}
    assert names == {"indNeut", "riskNeut", "riskNeutMulti"}

    info = cat.get_entry("riskNeutMulti")
    assert info is not None
    assert info["category"] == "neutralize"


def _make_synthetic_panel(T: int = 30, N: int = 12, seed: int = 123) -> Panel:
    rng = np.random.default_rng(seed)
    dates = [f"2025-01-{i + 1:02d}" if i < 31 else f"2025-02-{i - 30:02d}" for i in range(T)]
    symbols = [f"SYM{j:02d}" for j in range(N)]

    close = np.asfortranarray(100.0 * np.exp(np.cumsum(rng.normal(0.0005, 0.015, (T, N)), axis=0)))
    volume = np.asfortranarray(rng.uniform(1000.0, 50000.0, (T, N)))
    turnover = np.asfortranarray(close * volume * 10.0)
    rv = np.asfortranarray(rng.uniform(0.008, 0.03, (T, N)))
    ry_ann = np.asfortranarray(rng.normal(0.0, 0.05, (T, N)))
    wh_z60 = np.asfortranarray(rng.normal(0.0, 1.0, (T, N)))
    mr_net_ratio = np.asfortranarray(rng.normal(0.0, 0.2, (T, N)))
    sector = np.asfortranarray(np.tile((np.arange(N) % 3) + 1.0, (T, 1)))
    listed_days = np.asfortranarray(np.full((T, N), 200.0))
    in_uni = np.asfortranarray(np.ones((T, N), dtype=np.float64))
    lim_mask = np.asfortranarray(np.ones((T, N), dtype=np.float64))

    fields = {
        "close": close,
        "open": close * 0.999,
        "high": close * 1.01,
        "low": close * 0.99,
        "settle": close,
        "vwap": close,
        "exec_vwap": close,
        "exec_vwap_old": close,
        "volume": volume,
        "turnover": turnover,
        "rv": rv,
        "ry_ann": ry_ann,
        "wh_z60": wh_z60,
        "mr_net_ratio": mr_net_ratio,
        "sector": sector,
        "listed_days": listed_days,
        "in_universe_mask": in_uni,
        "limit_mask": lim_mask,
    }

    y_1 = np.zeros((T, N), dtype=np.float64, order="F")
    y_1[:-1, :] = close[1:, :] / close[:-1, :] - 1.0
    y_5 = np.zeros((T, N), dtype=np.float64, order="F")
    y_5[:-5, :] = close[5:, :] / close[:-5, :] - 1.0

    labels = {
        "r_exec_next": y_1.copy(order="F"),
        "y_1": y_1,
        "y_5": y_5,
    }
    valid_mask = np.ones((T, N), dtype=bool, order="F")
    return Panel(
        dates=np.array(dates), symbols=symbols, fields=fields, labels=labels, mask=valid_mask
    )


def test_dual_mode_and_postprocess_perturbation_zero_lookahead():
    """验证 DSL 与 Python compute(context) 双模态 + 后处理链在篡改未来行情时严格保持 0 未来泄露。"""
    panel_orig = _make_synthetic_panel(T=28, N=12, seed=7)
    panel_pert = _make_synthetic_panel(T=28, N=12, seed=7)

    t_perturb = 15
    # 剧烈篡改 t > t_perturb 的所有未来行情
    for k in panel_pert.fields:
        if k not in ("sector", "listed_days", "in_universe_mask", "limit_mask"):
            panel_pert.fields[k][t_perturb + 1 :, :] *= 5.0
            panel_pert.fields[k][t_perturb + 1 :, 0] += 9999.0

    # 1. Expression 模式 + 完整风格中性化与平滑后处理链
    expr_def = FactorDefinition(
        name="pert_expr",
        factor_type="expression",
        expression="pct_change(close, 5) - ts_mean(rv, 5)",
        postprocess="baseUniMask,trdMask,MAD:3.0,indNeut,riskNeutMulti:SIZE+VOLATILITY,zScore,emadecay:3",
    )
    mat_orig_e, _ = evaluate_factor_definition(expr_def, panel_orig)
    mat_pert_e, _ = evaluate_factor_definition(expr_def, panel_pert)
    _assert_mat_close(
        mat_orig_e[: t_perturb + 1],
        mat_pert_e[: t_perturb + 1],
        tol=1e-15,
        msg="Expression + Postprocess Anti-Lookahead Perturbation",
    )

    # 2. Python Code 模式 (`def compute(context):`) + 后处理链
    py_def = FactorDefinition(
        name="pert_py",
        factor_type="code",
        code="""
def compute(context):
    c = context.history("close", 5)
    v = context.history("volume", 5)
    ret = c[-1] / c[0] - 1.0
    v_norm = v[-1] / np.nanmean(v, axis=0)
    return ret * v_norm
""",
        postprocess="baseUniMask,trdMask,winsor:0.05,indNeut,riskNeut:CARRY,rank,decay:3",
    )
    mat_orig_p, _ = evaluate_factor_definition(py_def, panel_orig)
    mat_pert_p, _ = evaluate_factor_definition(py_def, panel_pert)
    _assert_mat_close(
        mat_orig_p[: t_perturb + 1],
        mat_pert_p[: t_perturb + 1],
        tol=1e-15,
        msg="Python Code + Postprocess Anti-Lookahead Perturbation",
    )


def test_end_to_end_run_factor_workflow_and_versioning(tmp_path: Path):
    """验证 6 阶段端到端因子工作流：探针、物化、Layer 1 v4 统计归因、Layer 2 实盘盯市回测与版本锁定。"""
    panel = _make_synthetic_panel(T=32, N=15, seed=99)
    reg = FactorRegistry(db_path=tmp_path / "workflow_factors.sqlite")

    defn = FactorDefinition(
        name="carry_mom_combo",
        factor_type="expression",
        expression="ry_ann + pct_change(close, 5)",
        postprocess="standard",
        description="期限结构展期叠加短期动量并做标准截面后处理",
    )

    wf_res = run_factor_workflow(
        panel=panel,
        definition=defn,
        registry=reg,
        materialize=True,
        lock_version=True,
    )

    # 校验探针输出
    assert wf_res.probe_report["status"] == "ok"
    assert "|> baseUniMask()" in wf_res.probe_report["canon_expr"]

    # 校验 Layer 1 Schema v4 统计输出
    v4 = wf_res.v4_stats
    assert v4["schema_version"] == "v4"
    assert "ic_analysis" in v4
    assert "perc_ic_mean" in v4["ic_analysis"]["summary"]
    assert "group_analysis" in v4
    assert len(v4["group_analysis"]["summary"]) == 5
    assert "long_analysis" in v4
    assert "additive_max_drawdown" in v4["long_analysis"]["summary"]
    assert "style_exposure" in v4
    assert len(v4["style_exposure"]["styles"]) == 6

    # 校验 Layer 2 期货实盘逐日盯市回测输出
    assert wf_res.backtest_result is not None
    assert len(wf_res.backtest_result.equity) == 32
    assert wf_res.backtest_result.equity[0] > 0

    # 校验注册表物化文件与版本锁定
    assert wf_res.factor_id is not None
    assert wf_res.version_locked == 1
    assert wf_res.materialization_path is not None
    assert wf_res.materialization_path.exists()

    loaded = reg.load_materialized_factor(wf_res.factor_id)
    assert loaded is not None
    mat_loaded, dates_loaded, syms_loaded = loaded
    _assert_mat_close(mat_loaded, wf_res.evaluation.factor_matrix, tol=1e-15)
    assert len(dates_loaded) == 32
    assert len(syms_loaded) == 15

    versions = reg.list_factor_versions(wf_res.factor_id)
    assert len(versions) == 1
    assert versions[0].version == 1


def test_futures_styles_universe_mask_isolation():
    """验证 compute_futures_styles 在非在池标的 (mask == False) 处完全隔离极端值，
    且非在池标的位置严格输出 NaN，在池标的结果不受污染。
    """
    from factorlib.core.styles import compute_futures_styles

    panel_clean = _make_synthetic_panel(T=30, N=10, seed=42)
    panel_polluted = _make_synthetic_panel(T=30, N=10, seed=42)

    # 构造品种池掩码：
    # 1. 标的 8, 9 为全周期非在池僵尸合约 (如低流动性僵尸品种)
    # 2. 标的 7 从第 10 天起出池 (如临近交割退池)
    mask = np.ones((30, 10), dtype=bool, order="F")
    mask[:, 8:] = False
    mask[10:, 7] = False

    panel_clean.mask = mask
    panel_polluted.mask = mask

    # 在 panel_polluted 的 ~mask 区域注入极端污染值 (1e9)
    for k in panel_polluted.fields:
        if k not in ("sector", "listed_days", "in_universe_mask", "limit_mask"):
            panel_polluted.fields[k][~mask] = 1e9

    styles_clean = compute_futures_styles(panel_clean)
    styles_polluted = compute_futures_styles(panel_polluted)

    for s_name in ("SIZE", "VOLATILITY", "MOMENTUM", "CARRY", "INVENTORY", "POSITIONING"):
        sc = styles_clean[s_name]
        sp = styles_polluted[s_name]

        # 1. 非在池标的位置严格输出 NaN
        assert np.all(np.isnan(sc[~mask])), f"style {s_name} clean at ~mask must be NaN"
        assert np.all(np.isnan(sp[~mask])), f"style {s_name} polluted at ~mask must be NaN"

        # 2. 全周期在池标的 (0..6) 无论在全周期还是出池期，计算结果与基准保持 0 bit 差异
        _assert_mat_close(sc[:, :7], sp[:, :7], tol=1e-12, msg=f"style {s_name} active")

        # 3. 标的 7 在前 10 天（仍在池时）计算结果不受后 20 天退池极端值影响
        _assert_mat_close(sc[:10, 7], sp[:10, 7], tol=1e-12, msg=f"style {s_name} s7 early")


def test_python_factor_sandbox_security_p0_08():
    """验证 P0-08 Python 因子沙箱安全红线：阻断私有属性访问、双下划线逃逸与 numpy 文件 I/O。"""
    panel = _make_synthetic_panel(T=10, N=6, seed=7)

    # 1. 尝试访问 context._fields 越权读取未来数据
    bad_priv = FactorDefinition(
        name="bad_priv",
        factor_type="code",
        code="def compute(context):\n    return context._fields['close'][-1]\n",
    )
    rep1 = probe_factor(bad_priv, panel=panel)
    assert rep1["status"] == "error"
    assert any("私有或双下划线属性" in e for e in rep1["errors"])
    with pytest.raises(ValueError, match="私有或双下划线属性"):
        evaluate_factor_definition(bad_priv, panel)

    # 2. 尝试通过 numpy.loadtxt / fromfile 读取磁盘文件
    bad_io = FactorDefinition(
        name="bad_io",
        factor_type="code",
        code="import numpy as np\ndef compute(context):\n    return np.fromfile('/etc/passwd')\n",
    )
    rep2 = probe_factor(bad_io, panel=panel)
    assert rep2["status"] == "error"
    assert any("文件或内存 I/O" in e for e in rep2["errors"])

    # 3. 尝试引用双下划线内置名称
    bad_dunder = FactorDefinition(
        name="bad_dunder",
        factor_type="code",
        code="def compute(context):\n    x = __builtins__\n    return context.history('close', 1)[0]\n",
    )
    rep3 = probe_factor(bad_dunder, panel=panel)
    assert rep3["status"] == "error"
    assert any("双下划线" in e for e in rep3["errors"])


def test_trd_mask_uses_is_unilateral_traded_and_volume_when_limit_mask_absent():
    """验证 P1-03: 当 panel.fields 未预置 limit_mask 时，trdMask 联动拦截单边市/未交易/零成交量。"""
    from factorlib.core.probe import (
        apply_postprocess_pipeline,
        parse_postprocess_pipeline,
    )

    T, N = 2, 4
    dates = np.array(["2024-01-02", "2024-01-03"])
    symbols = ["A", "B", "C", "D"]
    mask = np.ones((T, N), dtype=bool, order="F")
    mask[0, 3] = False  # 标的 D 首日不在池

    # A: 正常可交易; B: 首日单边市 (is_unilateral=1); C: 首日 traded=0; D: 次日 volume=0
    volume = np.array(
        [
            [100.0, 100.0, 100.0, 100.0],
            [100.0, 100.0, 100.0, 0.0],
        ],
        dtype=np.float64,
        order="F",
    )
    is_unilateral = np.array(
        [
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
        ],
        dtype=np.float64,
        order="F",
    )
    traded = np.array(
        [
            [1.0, 1.0, 0.0, 1.0],
            [1.0, 1.0, 1.0, 1.0],
        ],
        dtype=np.float64,
        order="F",
    )

    panel = Panel(
        dates=dates,
        symbols=symbols,
        fields={
            "close": np.full((T, N), 100.0, dtype=np.float64, order="F"),
            "volume": volume,
        },
        aux_fields={
            "is_unilateral": is_unilateral,
            "traded": traded,
        },
        mask=mask,
    )
    assert "limit_mask" not in panel.fields

    raw = np.full((T, N), 2.5, dtype=np.float64, order="F")
    steps = parse_postprocess_pipeline([{"op": "trdMask"}])
    masked = apply_postprocess_pipeline(raw, steps, panel)

    expected_nan = np.array(
        [
            [False, True, True, True],
            [False, False, False, True],
        ],
        dtype=bool,
    )
    np.testing.assert_array_equal(np.isnan(masked), expected_nan)
    np.testing.assert_allclose(masked[~expected_nan], 2.5)
