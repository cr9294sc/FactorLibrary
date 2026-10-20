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
    assert "exec_vwap" in p.fields
    assert "y_1" in p.labels
    assert p.fields["close"].flags.f_contiguous
    assert p.labels["y_1"].flags.f_contiguous
