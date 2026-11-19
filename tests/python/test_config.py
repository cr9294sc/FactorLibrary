"""测试配置加载与校验 (tests/python/test_config.py)。"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest
from factorlib.core.config import (
    CapitalConfig,
    ConfigError,
    DominantConfig,
    ExecPriceConfig,
    ExecutionConfig,
    GPConfig,
    LightGBMConfig,
    MLConfig,
    PortfolioConfig,
    TreeLimitsConfig,
    VariationConfig,
    WalkForwardConfig,
    load_backtest_config,
    load_data_config,
    load_gp_config,
    load_ml_config,
    load_stock_backtest_config,
    load_stock_data_config,
    load_stock_datasets_config,
    load_stock_fees_config,
    load_universe_config,
    validate_profile_configs,
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


def test_load_gp_config_and_validation(tmp_path: Path):
    default_gp_path = Path("configs/futures/gp_default.yaml")
    assert default_gp_path.exists()

    gcfg = load_gp_config(default_gp_path)
    assert isinstance(gcfg, GPConfig)
    assert gcfg.seed == 42
    assert gcfg.mode == "cs"
    assert gcfg.label == "y_5"
    assert gcfg.population == 1000
    assert gcfg.generations == 50
    assert gcfg.islands == 4
    assert gcfg.migration.interval == 10
    assert gcfg.limits.init_min_depth == 2
    assert gcfg.limits.init_max_depth == 5
    assert gcfg.limits.max_depth == 6
    assert gcfg.limits.max_nodes == 30
    assert gcfg.fitness.parsimony_penalty == 0.002
    assert gcfg.fitness.min_coverage == 0.70
    assert gcfg.admission.min_valid_ic_ir == 0.30
    assert gcfg.admission.max_library_corr == 0.60

    # 非法树深限制
    with pytest.raises(ConfigError, match="init_max_depth"):
        TreeLimitsConfig(init_min_depth=4, init_max_depth=2, max_depth=6)

    # 非法变异概率
    with pytest.raises(ConfigError, match="variation"):
        VariationConfig(
            crossover_prob=-0.1,
            subtree_mut_prob=0.5,
            point_mut_prob=0.2,
            hoist_mut_prob=0.2,
            param_mut_prob=0.2,
        )


def test_load_ml_config_and_validation(tmp_path: Path):
    default_ml_path = Path("configs/futures/ml_default.yaml")
    assert default_ml_path.exists()

    mcfg = load_ml_config(default_ml_path)
    assert isinstance(mcfg, MLConfig)
    assert mcfg.target_label == "y_5"
    assert "accepted" in mcfg.features_status
    assert mcfg.walk_forward.start_predict_year == 2022
    assert mcfg.walk_forward.end_predict_year == 2026
    assert mcfg.walk_forward.train_window == "expanding"
    assert mcfg.walk_forward.purge_window == 5
    assert mcfg.walk_forward.validation_size_days == 252
    assert mcfg.lightgbm.objective == "regression_l2"
    assert mcfg.lightgbm.max_depth == 4
    assert mcfg.lightgbm.num_leaves == 15
    assert mcfg.lightgbm.min_child_samples == 200
    assert mcfg.lightgbm.feature_fraction == 0.70
    assert mcfg.lightgbm.bagging_fraction == 0.80
    assert mcfg.lightgbm.early_stopping_rounds == 50

    # 序列化测试
    d = mcfg.to_dict()
    assert d["target_label"] == "y_5"
    assert "lightgbm" in d
    assert "walk_forward" in d
    yaml_str = mcfg.to_yaml()
    assert "regression_l2" in yaml_str

    # 非法 walk_forward 验证
    with pytest.raises(ConfigError, match="train_window"):
        WalkForwardConfig(train_window="invalid_mode")
    with pytest.raises(ConfigError, match="start_predict_year"):
        WalkForwardConfig(start_predict_year=2025, end_predict_year=2020)

    # 非法 lightgbm 验证
    with pytest.raises(ConfigError, match="feature_fraction"):
        LightGBMConfig(feature_fraction=0.0)
    with pytest.raises(ConfigError, match="max_depth"):
        LightGBMConfig(max_depth=0)

    # 非法 MLConfig
    with pytest.raises(ConfigError, match="target_label"):
        MLConfig(target_label="")
    with pytest.raises(ConfigError, match="features_status"):
        MLConfig(features_status=("unknown_status",))
    with pytest.raises(ConfigError, match="model"):
        MLConfig(models=("unsupported_model",))


def test_load_backtest_config(tmp_path: Path):
    yaml_content = """
capital:
  init_cash: 5000000.0

portfolio:
  mode: "cs"
  target_volatility: 0.12
  vol_lookback: 25
  cov_shrink_lookback: 50
  max_single_weight: 0.10
  max_gross_leverage: 2.5
  rebalance_buffer: 0.02

execution:
  max_margin_ratio: 0.60
  margin_safety_multiplier: 1.15
  lot_rounding: "floor"
"""
    p = tmp_path / "backtest.yaml"
    p.write_text(yaml_content, encoding="utf-8")

    bcfg = load_backtest_config(p)
    assert bcfg.capital.init_cash == 5000000.0
    assert bcfg.portfolio.mode == "cs"
    assert bcfg.portfolio.target_volatility == 0.12
    assert bcfg.portfolio.vol_lookback == 25
    assert bcfg.portfolio.cov_shrink_lookback == 50
    assert bcfg.portfolio.max_single_weight == 0.10
    assert bcfg.portfolio.max_gross_leverage == 2.5
    assert bcfg.portfolio.rebalance_buffer == 0.02
    assert bcfg.execution.max_margin_ratio == 0.60
    assert bcfg.execution.margin_safety_multiplier == 1.15
    assert bcfg.execution.lot_rounding == "floor"

    # 序列化测试
    d = bcfg.to_dict()
    assert d["capital"]["init_cash"] == 5000000.0
    y_str = bcfg.to_yaml()
    assert "init_cash: 5000000.0" in y_str

    # 非法参数校验
    with pytest.raises(ConfigError, match="init_cash"):
        CapitalConfig(init_cash=0.0)
    with pytest.raises(ConfigError, match="portfolio.mode"):
        PortfolioConfig(mode="invalid_mode")
    with pytest.raises(ConfigError, match="target_volatility"):
        PortfolioConfig(target_volatility=-0.1)
    with pytest.raises(ConfigError, match="vol_lookback"):
        PortfolioConfig(vol_lookback=1)
    with pytest.raises(ConfigError, match="cov_shrink_lookback"):
        PortfolioConfig(cov_shrink_lookback=1)
    with pytest.raises(ConfigError, match="max_single_weight"):
        PortfolioConfig(max_single_weight=0.0)
    with pytest.raises(ConfigError, match="max_single_weight"):
        PortfolioConfig(max_single_weight=1.5)
    with pytest.raises(ConfigError, match="max_gross_leverage"):
        PortfolioConfig(max_gross_leverage=0.0)
    with pytest.raises(ConfigError, match="rebalance_buffer"):
        PortfolioConfig(rebalance_buffer=-0.01)
    with pytest.raises(ConfigError, match="max_margin_ratio"):
        ExecutionConfig(max_margin_ratio=0.0)
    with pytest.raises(ConfigError, match="max_margin_ratio"):
        ExecutionConfig(max_margin_ratio=1.5)
    with pytest.raises(ConfigError, match="margin_safety_multiplier"):
        ExecutionConfig(margin_safety_multiplier=0.9)
    with pytest.raises(ConfigError, match="lot_rounding"):
        ExecutionConfig(lot_rounding="invalid_rounding")


def test_load_default_backtest_yaml():
    p = Path("configs/futures/backtest.yaml")
    assert p.exists(), "configs/futures/backtest.yaml 必须存在"
    cfg = load_backtest_config(p)
    assert cfg.capital.init_cash == 10_000_000.0
    assert cfg.portfolio.target_volatility == 0.10
    assert cfg.execution.max_margin_ratio == 0.50


def test_alt_data_lag_days_validation(tmp_path: Path):
    """验证 P1-06: timeline.alt_data_lag_days 仅允许 0 或 1。"""
    base_yaml = """
raw_dir: "./exports"
derived_dir: "./data/derived/futures"
timeline:
  start_date: "2020-01-01"
  end_date: "2024-01-01"
  alt_data_lag_days: {lag}
"""
    p1 = tmp_path / "ok.yaml"
    p1.write_text(base_yaml.format(lag=1), encoding="utf-8")
    cfg = load_data_config(p1, root=tmp_path)
    assert cfg.alt_data_lag_days == 1

    p2 = tmp_path / "bad.yaml"
    p2.write_text(base_yaml.format(lag=2), encoding="utf-8")
    with pytest.raises(ConfigError, match="alt_data_lag_days"):
        load_data_config(p2, root=tmp_path)


def test_stocks_configs_and_profile_validation(tmp_path: Path):
    """验证 M1.4 A 股整套配置文件加载及费率非负/min_commission>=0/lot_size==100 校验。"""
    res_f = validate_profile_configs("futures")
    res_s = validate_profile_configs("stocks")
    assert res_f["data"].panel_version == "v1"
    assert res_s["data"].price_basis == "hfq"
    assert res_s["fees"].lot_size == 100
    assert res_s["fees"].min_commission >= 0.0
    assert res_s["backtest"].execution.lot_size == 100

    # lot_size != 100 必须报错
    bad_fees = tmp_path / "bad_fees.yaml"
    bad_fees.write_text("lot_size: 1\nfees:\n  commission_rate: 0.0001\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="lot_size"):
        load_stock_fees_config(bad_fees)

    # 负费率或负 min_commission 必须报错
    bad_comm = tmp_path / "bad_comm.yaml"
    bad_comm.write_text("lot_size: 100\nfees:\n  min_commission: -1.0\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="min_commission"):
        load_stock_fees_config(bad_comm)

    bad_stamp = tmp_path / "bad_stamp.yaml"
    bad_stamp.write_text("lot_size: 100\nfees:\n  stamp_duty_rate: -0.001\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="stamp_duty_rate"):
        load_stock_fees_config(bad_stamp)

    # 非法 price_basis 必须报错
    bad_data = tmp_path / "bad_data.yaml"
    bad_data.write_text(
        'raw_dir: "./data/raw/stocks"\nderived_dir: "./data/derived/stocks"\nprice_basis: "qfq"\n',
        encoding="utf-8",
    )
    with pytest.raises(ConfigError, match="price_basis"):
        load_stock_data_config(bad_data, root=tmp_path)

    # 非法 execution.lot_size 必须报错
    bad_bt = tmp_path / "bad_bt.yaml"
    bad_bt.write_text("execution:\n  lot_size: 50\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="lot_size"):
        load_stock_backtest_config(bad_bt)


def test_stocks_dual_dataset_data_yaml(tmp_path: Path):
    """验证双数据源版 configs/stocks/data.yaml 加载与数据集切换。"""
    data_yaml = Path("configs/stocks/data.yaml")
    default_ds, ds_map = load_stock_datasets_config(data_yaml)
    assert default_ds == "rq_v1"
    assert set(ds_map.keys()) >= {"rq_v1", "lb_barra_v1"}

    rq_cfg = load_stock_data_config(data_yaml, dataset="rq_v1")
    assert rq_cfg.source == "ricequant"
    assert rq_cfg.panel_version == "v1"
    assert rq_cfg.entry_window == "open"
    assert rq_cfg.exec_price_mode == "open_adj"
    assert rq_cfg.include_idiosyncratic_labels is False

    # 按 panel_version='v1' 也能解析到 rq_v1
    rq_alias = load_stock_data_config(data_yaml, dataset="v1")
    assert rq_alias.panel_version == "v1"

    lb_cfg = load_stock_data_config(data_yaml, dataset="lb_barra_v1")
    assert lb_cfg.source == "local_backtest"
    assert lb_cfg.panel_version == "lb_barra_v1"
    assert lb_cfg.entry_window == "vwap_open_30m"
    assert lb_cfg.exec_price_mode == "vwap_open_30m_adj"
    assert lb_cfg.include_idiosyncratic_labels is True
    assert lb_cfg.universe_filters.use_source_base_univ_mask is True
    assert lb_cfg.universe_filters.exclude_bjse is True

    bad_src = tmp_path / "bad_src.yaml"
    bad_src.write_text(
        'raw_dir: "./r"\nderived_dir: "./d"\nsource: "unknown_src"\n',
        encoding="utf-8",
    )
    with pytest.raises(ConfigError, match="source"):
        load_stock_data_config(bad_src, root=tmp_path)
