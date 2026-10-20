"""P3 验收测试：SQLite 因子注册表 ORM 与生命周期状态机测试 (tests/python/test_registry.py)。

涵盖白皮书 10.1 节与 10.2 节全部规范：
1. 数据库物理 Schema、索引与 WAL 模式初始化；
2. 因子元数据与多周期评估指标原子事务入库；
3. 代数指纹 canon_hash 唯一性约束与同质因子阻断；
4. 因子生命周期流转状态机 (candidate -> accepted -> active -> deprecated)；
5. 多维度检索过滤 (status, min_icir, mode, source)；
6. 级联删除物理完整性；
7. 实验运行元数据上下文全流程追踪。
"""

from __future__ import annotations

from pathlib import Path

import pytest
from factorlib.core.registry import (
    ExperimentRunRecord,
    FactorMetricsRecord,
    FactorRecord,
    FactorRegistry,
)


@pytest.fixture
def registry(tmp_path: Path) -> FactorRegistry:
    db_file = tmp_path / "test_factors.sqlite"
    return FactorRegistry(db_path=db_file)


def test_schema_initialization(registry: FactorRegistry):
    """测试数据库表与索引是否正确创建并激活外键约束。"""
    with registry._get_connection() as conn:
        cur = conn.cursor()
        cur.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = {row["name"] for row in cur.fetchall()}
        assert "factor" in tables
        assert "factor_metrics" in tables
        assert "experiment_run" in tables

        cur.execute("SELECT name FROM sqlite_master WHERE type='index'")
        indices = {row["name"] for row in cur.fetchall()}
        assert "idx_factor_status" in indices
        assert "idx_factor_hash" in indices
        assert "idx_metrics_lookup" in indices


def test_factor_crud_and_metrics_cascade(registry: FactorRegistry):
    """测试因子的创建、查询、更新指标与级联删除。"""
    m1 = FactorMetricsRecord(
        split="train",
        horizon=5,
        ic_mean=0.08,
        ic_std=0.16,
        ic_ir=0.50,
        ic_t=3.25,
        ic_win_rate=0.68,
        coverage=0.92,
        turnover=0.25,
        ls_sharpe=1.85,
        ls_annual_ret=0.15,
        ls_max_dd=0.06,
    )
    m2 = FactorMetricsRecord(
        split="valid",
        horizon=5,
        ic_mean=0.06,
        ic_std=0.15,
        ic_ir=0.40,
        ic_t=2.10,
        ic_win_rate=0.62,
        coverage=0.91,
        turnover=0.26,
        ls_sharpe=1.45,
        ls_annual_ret=0.12,
        ls_max_dd=0.08,
    )

    factor = FactorRecord(
        canon_expr="pct_change(close, 20)",
        canon_hash="hash_mom20",
        mode="cs",
        sign=1,
        n_nodes=3,
        depth=2,
        source="manual",
        source_run="run_001",
        data_version="v1",
        status="candidate",
        metrics=[m1, m2],
    )

    # 1. 注册新因子
    fid = registry.register_factor(factor, run_id="run_001")
    assert fid > 0

    # 2. 按 ID 查询
    rec = registry.get_factor(fid)
    assert rec is not None
    assert rec.canon_expr == "pct_change(close, 20)"
    assert rec.canon_hash == "hash_mom20"
    assert rec.sign == 1
    assert len(rec.metrics) == 2

    # 3. 按 Hash 查询
    rec_by_hash = registry.get_factor_by_hash("hash_mom20")
    assert rec_by_hash is not None
    assert rec_by_hash.id == fid

    # 4. 追加测试集指标
    m3 = FactorMetricsRecord(
        split="test",
        horizon=5,
        ic_mean=0.07,
        ic_std=0.15,
        ic_ir=0.46,
        ic_t=2.50,
        ic_win_rate=0.65,
        coverage=0.90,
        turnover=0.24,
        ls_sharpe=1.60,
        ls_annual_ret=0.14,
        ls_max_dd=0.07,
    )
    registry.add_metrics(fid, m3)
    rec_updated = registry.get_factor(fid)
    assert rec_updated is not None
    assert len(rec_updated.metrics) == 3

    # 5. 级联删除验证
    deleted = registry.delete_factor(fid)
    assert deleted is True
    assert registry.get_factor(fid) is None
    # 验证子表 metrics 是否级联删除
    with registry._get_connection() as conn:
        cur = conn.cursor()
        cur.execute("SELECT count(*) as cnt FROM factor_metrics WHERE factor_id = ?", (fid,))
        assert cur.fetchone()["cnt"] == 0


def test_unique_hash_and_upsert(registry: FactorRegistry):
    """测试 canon_hash 唯一性约束与 upsert 覆盖逻辑。"""
    f1 = FactorRecord(
        canon_expr="ry_main_sub",
        canon_hash="hash_ry",
        mode="cs",
        sign=1,
        source="seed",
        status="candidate",
    )
    fid1 = registry.register_factor(f1)
    assert fid1 > 0

    # 重复插入相同哈希且 upsert=False，应抛出 ValueError
    with pytest.raises(ValueError, match="已存在于库中"):
        registry.register_factor(f1, upsert=False)

    # 启用 upsert=True，应成功更新状态并复用 ID
    f1_update = FactorRecord(
        canon_expr="ry_main_sub",
        canon_hash="hash_ry",
        mode="cs",
        sign=1,
        source="seed",
        status="accepted",
    )
    fid2 = registry.register_factor(f1_update, upsert=True)
    assert fid2 == fid1

    rec = registry.get_factor(fid1)
    assert rec is not None
    assert rec.status == "accepted"


def test_factor_lifecycle_and_filtering(registry: FactorRegistry):
    """测试因子状态流转与多维度检索过滤。"""
    f_cand = FactorRecord(
        canon_expr="wh_chg5",
        canon_hash="hash_wh",
        status="candidate",
        source="gp",
        metrics=[FactorMetricsRecord(split="train", horizon=5, ic_ir=0.25)],
    )
    f_acc = FactorRecord(
        canon_expr="mr_net_ratio_chg5",
        canon_hash="hash_mr",
        status="accepted",
        source="seed",
        metrics=[FactorMetricsRecord(split="train", horizon=5, ic_ir=0.55)],
    )
    f_act = FactorRecord(
        canon_expr="pct_change(close, 10)",
        canon_hash="hash_mom10",
        status="active",
        source="manual",
        metrics=[FactorMetricsRecord(split="train", horizon=5, ic_ir=0.45)],
    )

    registry.register_factor(f_cand)
    id_acc = registry.register_factor(f_acc)
    registry.register_factor(f_act)

    # 检索 accepted
    acc_list = registry.get_accepted_factors()
    assert len(acc_list) == 1
    assert acc_list[0].canon_hash == "hash_mr"

    # 检索 active
    act_list = registry.get_active_factors()
    assert len(act_list) == 1
    assert act_list[0].canon_hash == "hash_mom10"

    # 按 ICIR 阈值过滤 (ICIR >= 0.40)
    high_ir_list = registry.list_factors(min_icir=0.40, split="train", horizon=5)
    assert len(high_ir_list) == 2
    hashes = {r.canon_hash for r in high_ir_list}
    assert hashes == {"hash_mr", "hash_mom10"}

    # 状态推进: accepted -> active
    registry.update_factor_status(id_acc, status="active")
    assert len(registry.get_active_factors()) == 2

    # 状态退化: active -> deprecated
    registry.update_factor_status(id_acc, status="deprecated")
    assert len(registry.get_active_factors()) == 1
    dep_list = registry.list_factors(status="deprecated")
    assert len(dep_list) == 1
    assert dep_list[0].canon_hash == "hash_mr"

    # 非法状态检查
    with pytest.raises(ValueError, match="无效的状态"):
        registry.update_factor_status(id_acc, status="invalid_status")


def test_experiment_run_tracking(registry: FactorRegistry):
    """测试实验运行上下文追踪与状态流转。"""
    run = ExperimentRunRecord(
        run_id="gp_20261020_001",
        kind="gp",
        config_yaml="pop_size: 100\nn_generations: 20",
        data_version="v1",
        git_sha="abcdef123456",
        n_evaluated=0,
        status="running",
    )

    # 1. 记录运行开始
    registry.record_run(run)
    saved_run = registry.get_run("gp_20261020_001")
    assert saved_run is not None
    assert saved_run.status == "running"
    assert saved_run.kind == "gp"

    # 2. 结束运行
    registry.finish_run("gp_20261020_001", status="completed", n_evaluated=2000)
    finished_run = registry.get_run("gp_20261020_001")
    assert finished_run is not None
    assert finished_run.status == "completed"
    assert finished_run.n_evaluated == 2000
    assert finished_run.finished_at is not None

    # 3. 列表检索
    runs = registry.list_runs(kind="gp", status="completed")
    assert len(runs) == 1
    assert runs[0].run_id == "gp_20261020_001"
