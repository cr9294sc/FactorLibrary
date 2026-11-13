"""测试标准研究面板内存容器 (tests/python/test_panel.py)。"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import numpy as np
import polars as pl
import pytest
from factorlib.core.panel import Panel


def test_panel_direct_initialization():
    dates = np.array([date(2024, 1, 1), date(2024, 1, 2), date(2024, 1, 3)])
    symbols = ["AG", "AU", "CU"]
    T = len(dates)
    N = len(symbols)

    mask = np.ones((T, N), dtype=bool, order="C")  # 故意传入 C-order 数组测试自动转 F-order
    c_arr = np.arange(T * N, dtype=np.float64).reshape((T, N), order="C")
    y_arr = np.ones((T, N), dtype=np.float64, order="C")

    p = Panel(
        dates=dates,
        symbols=symbols,
        fields={"close": c_arr},
        mask=mask,
        labels={"y_1": y_arr},
        meta={"source": "test"},
    )

    assert p.shape == (3, 3)
    assert p.T == 3
    assert p.N == 3
    assert p.fields["close"].flags.f_contiguous
    assert p.labels["y_1"].flags.f_contiguous
    assert p.meta["source"] == "test"

    # get_field 与 get_label
    close_field = p.get_field("close")
    assert close_field.shape == (3, 3)
    with pytest.raises(KeyError, match="未找到字段"):
        p.get_field("non_existent_field")

    y1_label = p.get_label("y_1")
    assert y1_label.shape == (3, 3)
    with pytest.raises(KeyError, match="未找到标签"):
        p.get_label("non_existent_label")

    # to_rust_engine 成功构建 Rust 计算内核 Engine
    eng = p.to_rust_engine(cache_mb=64, n_threads=2)
    assert eng.shape == (3, 3)
    assert eng.symbols == ["AG", "AU", "CU"]
    res = eng.eval("addc(close, 1.0)")
    np.testing.assert_allclose(res, c_arr + 1.0)


def test_panel_dimension_mismatch():
    dates = np.array([date(2024, 1, 1), date(2024, 1, 2)])
    symbols = ["A", "B", "C"]  # T=2, N=3

    good_mask = np.ones((2, 3), dtype=bool)
    bad_mask = np.ones((3, 3), dtype=bool)

    # mask 形状不匹配
    with pytest.raises(ValueError, match="mask 形状"):
        Panel(dates=dates, symbols=symbols, fields={}, mask=bad_mask)

    # 字段形状不匹配
    bad_field = np.ones((2, 2), dtype=np.float64)
    with pytest.raises(ValueError, match="字段 'close' 形状"):
        Panel(dates=dates, symbols=symbols, fields={"close": bad_field}, mask=good_mask)

    # 标签形状不匹配
    bad_label = np.ones((4, 3), dtype=np.float64)
    with pytest.raises(ValueError, match="标签 'y_1' 形状"):
        Panel(dates=dates, symbols=symbols, fields={}, mask=good_mask, labels={"y_1": bad_label})


def test_panel_load_from_parquet(tmp_path: Path):
    # 模拟生成 panel_daily.parquet 与 labels.parquet
    dates = [date(2024, 1, 1), date(2024, 1, 2)]

    # 构造长表，故意让 (date=2024-01-02, symbol=AG) 缺失，测试 NaN 填充
    panel_df = pl.DataFrame(
        {
            "trading_date": [dates[0], dates[0], dates[1]],
            "symbol": ["AG", "CU", "CU"],
            "in_universe": [True, False, True],
            "close": [10.0, 20.0, 25.0],
            "volume": [100.0, 200.0, 250.0],
            "is_roll": [False, False, True],  # boolean 字段
            "sector": ["Precious", "Industrial", "Industrial"],  # 字符串辅助列, load 自动忽略
        }
    )

    labels_df = pl.DataFrame(
        {
            "trading_date": [dates[0], dates[0], dates[1]],
            "symbol": ["AG", "CU", "CU"],
            "y_1": [0.01, -0.02, 0.03],
        }
    )

    meta_doc = {"version": "test_v1", "author": "factorlib"}

    panel_path = tmp_path / "panel_daily.parquet"
    labels_path = tmp_path / "labels.parquet"
    meta_path = tmp_path / "meta.json"

    panel_df.write_parquet(panel_path)
    labels_df.write_parquet(labels_path)
    meta_path.write_text(json.dumps(meta_doc), encoding="utf-8")

    # 1. 完整加载
    panel = Panel.load(tmp_path)
    assert panel.T == 2
    assert panel.N == 2
    assert list(panel.symbols) == ["AG", "CU"]
    assert panel.meta["version"] == "test_v1"

    # 验证 Fortran 内存排布
    assert panel.mask.flags.f_contiguous
    assert panel.fields["close"].flags.f_contiguous
    assert panel.fields["volume"].flags.f_contiguous
    assert panel.labels["y_1"].flags.f_contiguous

    # 验证 mask
    # dates[0]: AG=True, CU=False
    # dates[1]: AG missing -> False, CU=True
    assert panel.mask[0, 0] is True or panel.mask[0, 0] == 1
    assert panel.mask[0, 1] is False or panel.mask[0, 1] == 0
    assert panel.mask[1, 0] is False or panel.mask[1, 0] == 0
    assert panel.mask[1, 1] is True or panel.mask[1, 1] == 1

    # 验证缺失值 NaN 填充
    close_mat = panel.get_field("close")
    assert close_mat[0, 0] == 10.0
    assert close_mat[0, 1] == 20.0
    assert np.isnan(close_mat[1, 0])  # AG 在第2天缺失 -> NaN
    assert close_mat[1, 1] == 25.0

    y1_mat = panel.get_label("y_1")
    assert y1_mat[0, 0] == 0.01
    assert y1_mat[0, 1] == -0.02
    assert np.isnan(y1_mat[1, 0])
    assert y1_mat[1, 1] == 0.03

    # 2. 字段子集加载 fields_subset
    sub_panel = Panel.load(tmp_path, fields_subset=["close"])
    assert "close" in sub_panel.fields
    assert "volume" not in sub_panel.fields

    # 3. 不加载标签 load_labels=False
    no_label_panel = Panel.load(tmp_path, load_labels=False)
    assert len(no_label_panel.labels) == 0

    # 4. 目录不存在 panel_daily.parquet 报错 FileNotFoundError
    with pytest.raises(FileNotFoundError, match="未找到面板文件"):
        Panel.load(tmp_path / "empty_dir")


def test_panel_load_existing_quick_panel():
    derived_v1 = Path("data/derived/futures/v1")
    if not (derived_v1 / "panel_daily.parquet").exists():
        pytest.skip("本地 data/derived/futures/v1 尚未生成，跳过实盘面板加载集成测试")

    p = Panel.load(derived_v1)
    assert p.T > 0
    assert p.N == 79
    assert "close" in p.fields
    assert "exec_vwap" in p.aux_fields
    assert "exec_vwap" not in p.fields
    assert "y_1" in p.labels
    assert p.fields["close"].flags.f_contiguous
    assert p.labels["y_1"].flags.f_contiguous


def test_panel_load_isolates_aux_fields_from_dsl_and_python_sandbox(tmp_path: Path):
    """验证 P1-03: Panel.load 将回测/原始/规格 aux 列隔离至 aux_fields，对 DSL 与 Python 因子不可见。"""
    from factorlib.core.probe import (
        FactorDefinition,
        evaluate_factor_definition,
        probe_factor,
    )

    panel_dir = tmp_path / "derived_aux_iso"
    panel_dir.mkdir(parents=True, exist_ok=True)

    # 2 日 x 2 品种，注入 traded=0 / is_unilateral=1 边界场景
    df = pl.DataFrame(
        {
            "trading_date": [
                date(2024, 1, 2),
                date(2024, 1, 2),
                date(2024, 1, 3),
                date(2024, 1, 3),
            ],
            "symbol": ["AG", "RB", "AG", "RB"],
            "in_universe": [True, True, True, True],
            "sector": ["metal", "black", "metal", "black"],
            "listed_days": [500, 500, 501, 501],
            "close": [5000.0, 3800.0, 5050.0, 3820.0],
            "volume": [10000.0, 20000.0, 11000.0, 0.0],
            "exec_vwap": [5005.0, 3805.0, 5055.0, 3825.0],
            "exec_vwap_old": [5005.0, 3805.0, 5050.0, 3825.0],
            "open_raw": [5000.0, 3800.0, 5040.0, 3810.0],
            "close_raw": [5000.0, 3800.0, 5050.0, 3820.0],
            "settle_raw": [5002.0, 3801.0, 5048.0, 3818.0],
            "limit_up": [5300.0, 4100.0, 5350.0, 4120.0],
            "limit_down": [4700.0, 3500.0, 4750.0, 3520.0],
            "is_unilateral": [0, 1, 0, 0],
            "adj_factor": [1.0, 1.0, 1.0, 1.0],
            "is_roll": [False, False, True, False],
            "traded": [1, 1, 1, 0],
            "adv20_turnover": [1e9, 2e9, 1e9, 2e9],
            "multiplier": [15.0, 10.0, 15.0, 10.0],
            "margin_rate": [0.1, 0.1, 0.1, 0.1],
            "days_to_expiry": [30, 40, 29, 39],
        }
    )
    df.write_parquet(panel_dir / "panel_daily.parquet")

    panel = Panel.load(panel_dir, load_labels=False)

    # DSL 特征与结构化分组/掩码字段保留在 panel.fields
    for expected_field in ("close", "volume", "sector", "listed_days", "limit_mask"):
        assert expected_field in panel.fields

    # 回测/原始/规格辅助列全部隔离在 panel.aux_fields，严禁进入 panel.fields
    expected_aux = (
        "exec_vwap",
        "exec_vwap_old",
        "open_raw",
        "close_raw",
        "settle_raw",
        "limit_up",
        "limit_down",
        "is_unilateral",
        "adj_factor",
        "is_roll",
        "traded",
        "adv20_turnover",
        "multiplier",
        "margin_rate",
        "days_to_expiry",
    )
    for col in expected_aux:
        assert col in panel.aux_fields, f"{col} 应当进入 panel.aux_fields"
        assert col not in panel.fields, f"{col} 严禁暴露在 panel.fields 中"

    # limit_mask 正确拦截 (0, 1) 单边市与 (1, 1) 停牌无量
    lmask = panel.fields["limit_mask"]
    assert lmask[0, 0] == 1.0
    assert np.isnan(lmask[0, 1])
    assert lmask[1, 0] == 1.0
    assert np.isnan(lmask[1, 1])

    # Rust Engine 对 aux 字段不可见
    engine = panel.to_rust_engine()
    for bad_field in ("exec_vwap", "is_roll", "settle_raw", "r_exec_next"):
        with pytest.raises(ValueError):
            engine.eval(bad_field)

    # Python 因子沙箱与静态探针对 aux 字段不可见
    from factorlib.core.probe import PythonFactorContext

    py_leak = FactorDefinition(
        name="py_aux_leak",
        factor_type="code",
        code="def compute(context):\n    return context.history('exec_vwap', 1)[0]\n",
    )
    rep = probe_factor(py_leak, panel=panel)
    assert rep["status"] == "error"
    with pytest.raises(ValueError, match="exec_vwap"):
        evaluate_factor_definition(py_leak, panel)
    ctx = PythonFactorContext(panel.fields, panel.dates, panel.symbols, current_t=1)
    with pytest.raises(KeyError):
        ctx.history("exec_vwap", 1)

    # fields_subset=["close", "limit_mask"] 自动加载 traded/is_unilateral 到 aux_fields 并正确计算 limit_mask
    sub = Panel.load(panel_dir, fields_subset=["close", "limit_mask"], load_labels=False)
    assert set(sub.fields.keys()) == {"close", "limit_mask"}
    assert "traded" in sub.aux_fields and "traded" not in sub.fields
    assert "is_unilateral" in sub.aux_fields and "is_unilateral" not in sub.fields
    np.testing.assert_array_equal(np.isnan(sub.fields["limit_mask"]), np.isnan(lmask))
