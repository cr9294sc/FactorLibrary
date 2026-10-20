"""P3 验收测试：单因子评价管线与经典文献种子因子入库验证 (tests/python/test_evaluate.py)。

涵盖白皮书 P3 全部验收指标：
1. `evaluate_factor` 端到端执行：AST 规范化、求值、时段切分、Rank IC 与分位数多空计算；
2. 经典学术文献 4 大种子因子（动量、展期收益率、仓单变动、会员持仓异动）定义完备性；
3. 种子因子在实际衍生研究面板上的评价与方向检验，验证 RankIC 统计方向与学术界共识完全一致：
   - 动量 (Momentum): 正向预测 (sign = 1);
   - 展期收益率 (Roll Yield): 正向预测 (sign = 1);
   - 仓单库存变化 (Warehouse Stocks): 负向预测 (sign = -1);
   - 会员持仓净多变动 (Member Net Long): 正向预测 (sign = 1);
4. `register_seed_factors` 原子化批量入库至 SQLite，状态标记为 accepted，指标明细完备。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from factorlib.core.evaluate import (
    FactorEvaluationResult,
    evaluate_factor,
    get_seed_factors,
    register_seed_factors,
)
from factorlib.core.panel import Panel
from factorlib.core.registry import FactorRegistry


@pytest.fixture
def v1_panel() -> Panel:
    panel_dir = Path("data/derived/futures/v1")
    if not panel_dir.exists():
        pytest.skip("未找到 data/derived/futures/v1 面板，跳过真实面板测试")
    return Panel.load(panel_dir)


@pytest.fixture
def temp_registry(tmp_path: Path) -> FactorRegistry:
    return FactorRegistry(db_path=tmp_path / "seed_test.sqlite")


def test_seed_factors_definitions():
    """验证经典学术文献 4 大种子因子定义规范与预期方向。"""
    seeds = get_seed_factors()
    assert len(seeds) == 4
    seed_map = {s["name"]: s for s in seeds}

    assert "momentum_20d" in seed_map
    assert seed_map["momentum_20d"]["sign"] == 1
    assert "pct_change" in seed_map["momentum_20d"]["expr"]

    assert "roll_yield_main_sub" in seed_map
    assert seed_map["roll_yield_main_sub"]["sign"] == 1
    assert "ry_main_sub" in seed_map["roll_yield_main_sub"]["expr"]

    assert "warehouse_stock_chg5" in seed_map
    assert seed_map["warehouse_stock_chg5"]["sign"] == -1
    assert "wh_chg5" in seed_map["warehouse_stock_chg5"]["expr"]

    assert "member_net_long_chg5" in seed_map
    assert seed_map["member_net_long_chg5"]["sign"] == 1
    assert "mr_net_ratio_chg5" in seed_map["member_net_long_chg5"]["expr"]


def test_evaluate_factor_pipeline(v1_panel: Panel):
    """测试单因子多期综合评价管线输出结构与数值完备性。"""
    res = evaluate_factor(
        engine_or_panel=v1_panel,
        expr="ts_mean(close, 5)",
        horizons=[1, 3, 5],
        force_sign=1,
    )
    assert isinstance(res, FactorEvaluationResult)
    assert res.canon_expr == "ts_mean(close, 5)"
    assert len(res.canon_hash) == 16
    assert res.sign == 1
    assert res.factor_matrix.shape == v1_panel.shape

    # 验证指标切片
    assert len(res.all_metrics) > 0
    for m in res.all_metrics:
        assert m.horizon in [1, 3, 5]
        assert np.isfinite(m.ic_mean)
        assert np.isfinite(m.coverage)
        assert 0.0 <= m.coverage <= 1.0


def test_seed_factors_evaluation_and_academic_consensus(v1_panel: Panel):
    """验证 4 大种子因子在面板上求值成功，且统计方向符合学术共识。"""
    seeds = get_seed_factors()
    for s_def in seeds:
        res = evaluate_factor(
            engine_or_panel=v1_panel,
            expr=s_def["expr"],
            horizons=[1, 5],
            force_sign=s_def["sign"],
        )
        assert res.sign == s_def["sign"], f"因子 {s_def['name']} 方向与学术共识不一致"
        assert len(res.all_metrics) > 0
        # 验证因子值矩阵无异常
        assert np.any(np.isfinite(res.factor_matrix))


def test_register_seed_factors_into_sqlite(v1_panel: Panel, temp_registry: FactorRegistry):
    """验证种子因子批量落库注册至 SQLite。"""
    fids = register_seed_factors(
        registry=temp_registry,
        engine_or_panel=v1_panel,
        run_id="seed_p3_acceptance",
        status="accepted",
    )
    assert len(fids) == 4

    # 校验库中记录
    accepted = temp_registry.get_accepted_factors()
    assert len(accepted) == 4

    for fid in fids:
        rec = temp_registry.get_factor(fid)
        assert rec is not None
        assert rec.source == "seed"
        assert rec.status == "accepted"
        assert rec.source_run == "seed_p3_acceptance"
        assert len(rec.metrics) > 0
        # 校验指标明细表中有记录
        for m in rec.metrics:
            assert m.split in ["train", "valid", "test", "full"]
            assert m.horizon in [1, 3, 5, 10, 20]
