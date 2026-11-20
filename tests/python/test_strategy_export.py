"""Phase 3 P1-2 验收测试：`factorsim` 兼容策略文件一键导出。

覆盖：
1. 参数集默认值、区间强校验与嵌套 `asymmetric_trade_penalty` 归并；
2. 渲染源码可 `compile()`、参数确定顺序、关键数学实现不得缺失（防止静默回归）；
3. `strategy.py` + `config.yaml` 落盘、覆盖保护与幂等；
4. CLI `--help` / `--print-only` / 落盘三条路径。
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest
import yaml
from factorlib.markets.stocks.strategy_export import (
    DEFAULT_STRATEGY_PARAMS,
    FactorsimStrategyParams,
    StrategyExportError,
    export_factorsim_strategy,
    load_strategy_yaml,
    render_factorsim_config,
    render_factorsim_strategy,
    resolve_strategy_params,
    validate_strategy_params,
)

ROOT = Path(__file__).resolve().parents[2]


def test_render_strategy_compiles_and_contains_full_math() -> None:
    src = render_factorsim_strategy({})
    compile(src, "strategy.py", "exec")

    # 三层架构与五类关键数学实现不得缺失
    for marker in (
        "class LowVolIndustryNeutralStrategy(StrategyBase)",
        "def rank_score(",
        "def _shape_signal(",
        "def _update_size_stress(",
        "def _previous_weights(",
        '_cfg("adaptive_stress_guard")',
        "asymmetric",
        "buy_penalty",
        "sell_penalty",
        "reserve_cash",
        "buy_hurdle",
        "hurdle_penalty",
        "persistence",
        "discount",
        "ind_neutral_weight",
        "vol_multiplier",
        "power_scale",
        "size_tilt",
    ):
        assert marker in src, f"导出策略缺少关键实现标记: {marker}"

    # 严禁把旧的「静默忽略」参数写死
    assert "PARAMS = {" in src
    assert "class LowVolIndustryNeutralStrategy" in src


def test_params_defaults_and_validation() -> None:
    resolved = FactorsimStrategyParams({}).to_dict()
    assert resolved["power"] == DEFAULT_STRATEGY_PARAMS["power"]
    assert resolved["ind_neutral_weight"] == pytest.approx(0.42)
    assert resolved["buy_penalty"] == pytest.approx(1.00)
    assert resolved["sell_penalty"] == pytest.approx(0.45)
    assert resolved["reserve_cash"] == pytest.approx(0.9)

    bad_cases = [
        {"h_max": 0.0},
        {"tvr": 0.0},
        {"mode": "invalid"},
        {"power": 0.5},
        {"memory": 1.0},
        {"reserve_cash": 0.0},
        {"ind_neutral_weight": 1.5},
        {"power_scale": 0.0},
        {"vol_multiplier": 0.0},
        {"stress_lookback": 1},
        {"stress_z_window": 5},
    ]
    for bad in bad_cases:
        with pytest.raises(StrategyExportError):
            validate_strategy_params({**DEFAULT_STRATEGY_PARAMS, **bad})


def test_resolve_strategy_params_merges_nested_asymmetric_penalty() -> None:
    merged = resolve_strategy_params(
        {
            "power": 1.6,
            "ind_neutral_weight": 0.50,
            "asymmetric_trade_penalty": {"buy": 1.20, "sell": 0.30},
            "adaptive_stress_guard": False,
        },
        {"reserve_cash": 0.85, "fee_rate": 0.003},
    )
    assert merged["power"] == pytest.approx(1.6)
    assert merged["ind_neutral_weight"] == pytest.approx(0.50)
    assert merged["buy_penalty"] == pytest.approx(1.20)
    assert merged["sell_penalty"] == pytest.approx(0.30)
    assert merged["adaptive_stress_guard"] is False
    assert merged["reserve_cash"] == pytest.approx(0.85)


def test_render_factorsim_config_structure() -> None:
    cfg = render_factorsim_config(
        {"power": 1.8, "asymmetric_trade_penalty": {"buy": 1.0, "sell": 0.45}},
        {"cash": 20_000_000.0, "reserve_cash": 0.9, "fee_rate": 0.003},
    )
    assert cfg["type"] == "backtest"
    strat = cfg["strategy_config"]
    assert strat["strategy_class"] == "LowVolIndustryNeutralStrategy"
    assert strat["power"] == pytest.approx(1.8)
    assert strat["asymmetric_trade_penalty"] == {"buy": 1.0, "sell": 0.45}
    # reserve_cash 只属于回测层，不得混入策略参数
    assert "reserve_cash" not in strat
    assert cfg["backtest_config"]["reserve_cash"] == pytest.approx(0.9)
    assert cfg["backtest_config"]["fee_rate"] == pytest.approx(0.003)


def test_export_writes_files_with_overwrite_protection(tmp_path: Path) -> None:
    out = tmp_path / "local_backtest"
    paths = export_factorsim_strategy(out, params={"power": 1.7}, backtest={"fee_rate": 0.003})
    assert paths["strategy"].exists() and paths["config"].exists()

    source = paths["strategy"].read_text(encoding="utf-8")
    compile(source, str(paths["strategy"]), "exec")
    assert "1.7" in source

    cfg = yaml.safe_load(paths["config"].read_text(encoding="utf-8"))
    assert cfg["strategy_config"]["power"] == pytest.approx(1.7)

    # 无 overwrite 时必须拒绝覆盖
    with pytest.raises(StrategyExportError):
        export_factorsim_strategy(out, params={"power": 1.2})

    # overwrite 生效
    paths2 = export_factorsim_strategy(out, params={"power": 1.2}, overwrite=True)
    assert "1.2" in paths2["strategy"].read_text(encoding="utf-8")


def test_load_strategy_yaml_reads_platform_config() -> None:
    strategy, backtest = load_strategy_yaml(ROOT / "configs" / "stocks" / "backtest.yaml")
    assert isinstance(strategy, dict) and isinstance(backtest, dict)


def test_export_cli_help_and_print_only() -> None:
    script = ROOT / "scripts" / "export_factorsim_strategy.py"
    help_res = subprocess.run(
        [sys.executable, str(script), "--help"], capture_output=True, text=True, check=False
    )
    assert help_res.returncode == 0, help_res.stderr
    assert "--out-dir" in help_res.stdout

    print_res = subprocess.run(
        [sys.executable, str(script), "--print-only"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert print_res.returncode == 0, print_res.stderr
    assert "class LowVolIndustryNeutralStrategy" in print_res.stdout
    assert "config.yaml" in print_res.stdout
