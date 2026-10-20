"""测试配置加载与校验 (tests/python/test_config.py)。"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest

from factorlib.core.config import (
    ConfigError,
    DataConfig,
    DominantConfig,
    ExecPriceConfig,
    UniverseConfig,
    load_data_config,
    load_universe_config,
)


def test_dominant_config_validation():
    # 合法取值
    c = DominantConfig(rule=3, rank=1)
    assert c.rule == 3 and c.rank == 1

    # 非法 rule
    with pytest.raises(ConfigError, match="dominant.rule"):
        DominantConfig(rule=4, rank=1)

    # 非法 rank
    with pytest.raises(ConfigError, match="dominant.rank"):
        DominantConfig(rule=3, rank=0)


def test_exec_price_config_validation():
    c = ExecPriceConfig(window_minutes=30, calc_method="vwap_close_volume")
    assert c.window_minutes == 30

    # 严禁使用 turnover/volume (郑商所成交额置零风险)
    with pytest.raises(ConfigError, match="vwap_close_volume"):
        ExecPriceConfig(window_minutes=30, calc_method="turnover_volume")

    with pytest.raises(ConfigError, match="window_minutes"):
        ExecPriceConfig(window_minutes=0)


def test_load_data_config(tmp_path: Path):
    yaml_content = """
raw_dir: "./exports"
derived_dir: "./data/derived/futures"
panel_version: "v1_test"
dominant:
  rule: 3
  rank: 1
timeline:
  start_date: "2024-01-01"
  end_date: "2024-06-30"
labels:
  horizons: [1, 5, 20]
"""
    p = tmp_path / "data.yaml"
    p.write_text(yaml_content, encoding="utf-8")

    cfg = load_data_config(p, root=tmp_path)
    assert cfg.panel_version == "v1_test"
    assert cfg.dominant.rule == 3
    assert cfg.start_date == date(2024, 1, 1)
    assert cfg.end_date == date(2024, 6, 30)
    assert cfg.label_horizons == (1, 5, 20)

    # 验证语义哈希具有确定性
    h1 = cfg.hash()
    h2 = cfg.hash()
    assert h1 == h2
    assert isinstance(h1, str) and len(h1) == 64


def test_load_universe_config(tmp_path: Path):
    yaml_content = """
liquidity:
  min_adv20_turnover: 10000000.0
  min_listed_days: 30
  min_days_to_expiry: 3
evaluation:
  min_names_per_day: 5
"""
    p = tmp_path / "universe.yaml"
    p.write_text(yaml_content, encoding="utf-8")

    ucfg = load_universe_config(p)
    assert ucfg.min_adv20_turnover == 10_000_000.0
    assert ucfg.min_listed_days == 30
    assert ucfg.min_days_to_expiry == 3
    assert ucfg.min_names_per_day == 5
