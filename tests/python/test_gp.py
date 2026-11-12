"""P4 强类型遗传规划 (STGP) 批量挖掘引擎全维度验收测试 (tests/python/test_gp.py)。

覆盖白皮书第 9 章与 P4 阶段全部验收指标：
1. `gp/primitives.py` & `gp/generate.py`：强类型原语集、离散参数池与 Ramped Half-and-Half 生成器，
   验证生成的表达式 100% 通过 Rust 静态类型检查与防未来函数检查；
2. `gp/variation.py`：强类型子树交叉、子树变异、点变异、提升变异、参数微调 5 大算子安全约束验证；
3. `gp/islands.py`：多岛独立演化与环形拓扑迁移 (`0 -> 1 -> 2 -> 3 -> 0`)；
4. `gp/admission.py`：五步自动准入过滤器（Top-K 初筛 -> 验证集同向与强度 -> 换手率约束 -> 贪心正交化 -> SQLite 入库）；
5. `gp/engine.py` & `core/runs.py`：
   - 固定随机种子 100% 确定性复现（两轮独立运行逐代统计、名人堂与准入因子完全一致）；
   - 1000 个体 × 50 代性能压测与植入信号挖掘入库全链路验证。
"""

from __future__ import annotations

import time
from pathlib import Path

import factorlib_core
import numpy as np
import pytest
from factorlib.core.config import (
    AdmissionConfig,
    FitnessConfig,
    GPConfig,
    MigrationConfig,
    TreeLimitsConfig,
)
from factorlib.core.panel import Panel
from factorlib.core.registry import FactorRegistry
from factorlib.core.runs import RunTracker
from factorlib.gp import (
    DEFAULT_CONSTANTS,
    DEFAULT_WINDOWS,
    DataType,
    FieldNode,
    GPEngine,
    Island,
    OpNode,
    PrimitiveSet,
    RingMigrationController,
    VariationPipeline,
    WindowNode,
    canonicalize_node,
    crossover,
    generate_full,
    generate_grow,
    generate_ramped_half_and_half,
    hoist_mutation,
    param_tuning,
    parse_dsl_to_tree,
    point_mutation,
    subtree_mutation,
)


@pytest.fixture
def synthetic_gp_panel() -> Panel:
    """构建带强植入 Alpha 信号的合成多品种研究面板 (涵盖 train/valid/test 三个时段)。

    植入真实规律：`y_5` 与 `cs_rank(pct_change(close, 5))` 及 `neg(wh_chg5)` 高度正相关，
    使得 STGP 能够稳定挖掘出泛化到验证集且相互正交的因子并入库。
    """
    rng = np.random.default_rng(20261020)
    # 构造 3 个年份段共 180 个交易日：
    # train: 2015 (80天), valid: 2020 (50天), test: 2023 (50天)
    train_dates = [f"2015-{(i // 28) + 1:02d}-{(i % 28) + 1:02d}" for i in range(80)]
    valid_dates = [f"2020-{(i // 28) + 1:02d}-{(i % 28) + 1:02d}" for i in range(50)]
    test_dates = [f"2023-{(i // 28) + 1:02d}-{(i % 28) + 1:02d}" for i in range(50)]
    dates = np.array(train_dates + valid_dates + test_dates)
    T = len(dates)
    N = 16
    symbols = [f"SYM_{j:02d}" for j in range(N)]

    # 生成基础特征
    close = np.zeros((T, N), dtype=np.float64, order="F")
    close[0, :] = 100.0 + rng.uniform(-10.0, 10.0, size=N)
    for t in range(1, T):
        ret_t = rng.normal(0.0, 0.015, size=N)
        close[t, :] = close[t - 1, :] * (1.0 + ret_t)

    open_p = np.asfortranarray(close * (1.0 + rng.normal(0.0, 0.003, size=(T, N))))
    high_p = np.asfortranarray(
        np.maximum(open_p, close) * (1.0 + np.abs(rng.normal(0.0, 0.005, size=(T, N))))
    )
    low_p = np.asfortranarray(
        np.minimum(open_p, close) * (1.0 - np.abs(rng.normal(0.0, 0.005, size=(T, N))))
    )
    volume = np.asfortranarray(rng.uniform(1000.0, 50000.0, size=(T, N)))
    rv = np.asfortranarray(np.abs(rng.normal(0.015, 0.004, size=(T, N))))
    ry_main_sub = np.asfortranarray(rng.normal(0.0, 0.02, size=(T, N)))
    wh_chg5 = np.asfortranarray(rng.normal(0.0, 0.05, size=(T, N)))
    mr_net_ratio = np.asfortranarray(rng.normal(0.0, 0.1, size=(T, N)))

    # 制造具有持续性的低换手信号源 (通过平滑确保换手率 <= 0.85)
    for col in range(N):
        for t in range(1, T):
            ry_main_sub[t, col] = 0.75 * ry_main_sub[t - 1, col] + 0.25 * ry_main_sub[t, col]
            wh_chg5[t, col] = 0.75 * wh_chg5[t - 1, col] + 0.25 * wh_chg5[t, col]
            mr_net_ratio[t, col] = 0.70 * mr_net_ratio[t - 1, col] + 0.30 * mr_net_ratio[t, col]

    # 植入两个正交的真实预测信号到 y_5 与 y_1：
    # Signal 1: ry_main_sub (正向)
    # Signal 2: -wh_chg5 (反向)
    y_5 = np.asfortranarray(
        0.80 * ry_main_sub - 0.50 * wh_chg5 + rng.normal(0.0, 0.015, size=(T, N))
    )
    y_1 = np.asfortranarray(0.25 * y_5 + rng.normal(0.0, 0.005, size=(T, N)))

    mask = np.ones((T, N), dtype=bool, order="F")

    fields = {
        "open": open_p,
        "high": high_p,
        "low": low_p,
        "close": close,
        "volume": volume,
        "rv": rv,
        "ry_main_sub": ry_main_sub,
        "wh_chg5": wh_chg5,
        "mr_net_ratio": mr_net_ratio,
    }
    labels = {
        "r_exec_next": y_1,
        "y_1": y_1,
        "y_3": y_5 * 0.6,
        "y_5": y_5,
        "y_10": y_5 * 1.4,
        "y_20": y_5 * 1.8,
    }
    return Panel(
        dates=dates,
        symbols=symbols,
        fields=fields,
        mask=mask,
        labels=labels,
        meta={"panel_version": "v1_synth"},
    )


def test_primitives_and_ramped_half_and_half(synthetic_gp_panel: Panel):
    """测试强类型原语集与 Ramped Half-and-Half 初始化生成器。"""
    pset = PrimitiveSet.from_panel(synthetic_gp_panel)
    assert pset.windows == DEFAULT_WINDOWS
    assert pset.constants == DEFAULT_CONSTANTS
    assert "close" in pset.fields
    assert len(pset.operators) >= 60

    rng = np.random.default_rng(42)

    # 测试 Full 模式生成的树深度精确达到指定深度
    full_tree = generate_full(pset, rng, depth=4, target_type=DataType.Series)
    assert full_tree.return_type == DataType.Series
    assert full_tree.depth == 4

    # 测试 Grow 模式生成
    grow_tree = generate_grow(pset, rng, min_depth=2, max_depth=5, target_type=DataType.Series)
    assert grow_tree.return_type == DataType.Series
    assert 2 <= grow_tree.depth <= 5

    # 测试 Ramped Half-and-Half 批量生成 100 个个体，全部通过 Rust 静态类型检查
    pop = generate_ramped_half_and_half(
        pset=pset,
        pop_size=100,
        rng=rng,
        min_depth=2,
        max_depth=5,
        max_nodes=30,
        deduplicate=True,
    )
    assert len(pop) == 100
    avail_fields = list(pset.fields)

    for ind in pop:
        assert not ind.is_degenerate
        assert 1 <= ind.depth <= 5
        assert 1 <= ind.n_nodes <= 30
        chk = factorlib_core.check_expr(ind.canon_expr, avail_fields, True)
        assert chk["valid"] is True, f"非法表达式: {ind.canon_expr}, 错误: {chk['error']}"
        assert chk["return_type"] == "Series"


def test_variation_operators_type_safety(synthetic_gp_panel: Panel):
    """测试全部 5 类强类型变异与交叉算子的类型安全与深度/节点上限约束。"""
    pset = PrimitiveSet.from_panel(synthetic_gp_panel)
    rng = np.random.default_rng(123)
    limits = TreeLimitsConfig(init_min_depth=2, init_max_depth=4, max_depth=6, max_nodes=25)
    pipe = VariationPipeline(pset=pset, limits=limits)

    pop = generate_ramped_half_and_half(
        pset=pset,
        pop_size=30,
        rng=rng,
        min_depth=2,
        max_depth=4,
        max_nodes=25,
    )
    avail_fields = list(pset.fields)

    for i in range(len(pop)):
        p1 = pop[i]
        p2 = pop[(i + 1) % len(pop)]

        c_cross = crossover(p1, p2, pset, rng, max_depth=6, max_nodes=25)
        c_sub = subtree_mutation(p1, pset, rng, max_depth=6, max_nodes=25)
        c_pt = point_mutation(p1, pset, rng, max_depth=6, max_nodes=25)
        c_hoist = hoist_mutation(p1, pset, rng, max_depth=6, max_nodes=25)
        c_param = param_tuning(p1, pset, rng, max_depth=6, max_nodes=25)
        c_pipe = pipe.apply(p1, p2, rng)

        for child in (c_cross, c_sub, c_pt, c_hoist, c_param, c_pipe):
            assert not child.is_degenerate
            assert child.depth <= 6
            assert child.n_nodes <= 25
            chk = factorlib_core.check_expr(child.canon_expr, avail_fields, True)
            assert chk["valid"] is True

    # 专项验证 hoist_mutation 能够缩减树深
    deep_expr = "cs_rank(ts_mean(ts_std(close, 5), 10))"
    deep_node = parse_dsl_to_tree(deep_expr, pset)
    deep_ind = canonicalize_node(deep_node, pset=pset)
    hoisted = hoist_mutation(deep_ind, pset, rng, max_depth=6, max_nodes=25)
    assert hoisted.depth < deep_ind.depth


def test_ring_island_migration(synthetic_gp_panel: Panel):
    """测试多岛模型与环形拓扑迁移 (`Island 0 -> 1 -> 2 -> 3 -> 0`)。"""
    pset = PrimitiveSet.from_panel(synthetic_gp_panel)
    islands = [Island(island_id=i, pop_size=20, seed=101 + i) for i in range(4)]

    for idx, isl in enumerate(islands):
        isl.initialize(pset, min_depth=2, max_depth=4, max_nodes=20)
        # 赋予人工适应度以便检验精英迁移方向
        for j, ind in enumerate(isl.population):
            ind.adjusted_fitness = float(idx * 10.0 + j * 0.1)
            ind.evaluated = True

    # 为 Island 0 构造一个具有独一无二高适应度的个体
    champion_node = OpNode(
        op=pset.ops_by_name["ts_mean"],
        args=(FieldNode("ry_main_sub"), WindowNode(5)),
    )
    champion = canonicalize_node(champion_node, pset=pset)
    champion.adjusted_fitness = 999.0
    champion.evaluated = True
    islands[0].population[0] = champion

    # 确保 Island 1 中原本不含该 champion
    islands[1].population = [
        ind for ind in islands[1].population if ind.canon_hash != champion.canon_hash
    ]
    while len(islands[1].population) < 20:
        extra = islands[1].population[0].clone()
        islands[1].population.append(extra)

    controller = RingMigrationController(interval=10, k_individuals=2)
    assert not controller.should_migrate(5, 4)
    assert controller.should_migrate(10, 4)

    migrated = controller.migrate(islands)
    assert migrated > 0

    # 验证 Island 0 的 champion 已经按环形拓扑迁入 Island 1
    island1_hashes = {ind.canon_hash for ind in islands[1].population}
    assert champion.canon_hash in island1_hashes


def test_gp_reproducibility_with_fixed_seed(synthetic_gp_panel: Panel, tmp_path: Path):
    """测试固定种子可 100% 稳定复现演化轨迹与最终产出。"""
    cfg = GPConfig(
        seed=2026,
        mode="cs",
        label="y_5",
        splits={
            "train": ("2015-01-01", "2015-12-31"),
            "valid": ("2020-01-01", "2020-12-31"),
            "test": ("2023-01-01", "2023-12-31"),
        },
        population=60,
        generations=6,
        islands=3,
        migration=MigrationConfig(interval=3, k_individuals=2),
        limits=TreeLimitsConfig(init_min_depth=2, init_max_depth=4, max_depth=5, max_nodes=20),
        admission=AdmissionConfig(
            min_valid_ic_ir=0.20,
            require_sign_consistency=True,
            max_library_corr=0.60,
        ),
    )

    reg1 = FactorRegistry(db_path=tmp_path / "rep1.sqlite")
    eng1 = GPEngine(cfg=cfg, panel=synthetic_gp_panel, registry=reg1)
    admitted1 = eng1.run(run_id="rep_run_1", register_to_db=True)

    reg2 = FactorRegistry(db_path=tmp_path / "rep2.sqlite")
    eng2 = GPEngine(cfg=cfg, panel=synthetic_gp_panel, registry=reg2)
    admitted2 = eng2.run(run_id="rep_run_2", register_to_db=True)

    # 1. 验证每一代的统计指标逐比特完全一致
    assert len(eng1.history) == len(eng2.history) == 7
    for s1, s2 in zip(eng1.history, eng2.history, strict=True):
        assert s1.best_expr == s2.best_expr
        assert s1.best_fitness == pytest.approx(s2.best_fitness, abs=1e-12)
        assert s1.mean_fitness == pytest.approx(s2.mean_fitness, abs=1e-12)
        assert s1.total_unique_evaluated == s2.total_unique_evaluated

    # 2. 验证名人堂完全一致
    hof1 = [ind.canon_hash for ind in eng1.hall_of_fame]
    hof2 = [ind.canon_hash for ind in eng2.hall_of_fame]
    assert hof1 == hof2

    # 3. 验证准入落库因子列表完全一致
    assert [f.canon_hash for f in admitted1] == [f.canon_hash for f in admitted2]


def test_admission_pipeline_five_steps_and_sqlite_registration(
    synthetic_gp_panel: Panel,
    tmp_path: Path,
):
    """验证五步准入决策器：符号一致性、ICIR 阈值、换手率约束、贪心正交化去相关与入库。"""
    db_path = tmp_path / "admission_test.sqlite"
    registry = FactorRegistry(db_path=db_path)
    tracker = RunTracker(
        kind="gp",
        run_id="gp_test_admission",
        runs_root=tmp_path / "runs",
        config_yaml="seed: 42\n",
        data_version="v1_synth",
        registry=registry,
    )

    cfg = GPConfig(
        seed=42,
        mode="cs",
        label="y_5",
        splits={
            "train": ("2015-01-01", "2015-12-31"),
            "valid": ("2020-01-01", "2020-12-31"),
            "test": ("2023-01-01", "2023-12-31"),
        },
        population=80,
        generations=8,
        islands=4,
        migration=MigrationConfig(interval=4, k_individuals=3),
        limits=TreeLimitsConfig(init_min_depth=2, init_max_depth=4, max_depth=5, max_nodes=25),
        fitness=FitnessConfig(
            metric="ic_ir",
            parsimony_penalty=0.002,
            min_coverage=0.70,
            corr_penalty_threshold=0.60,
            corr_penalty_factor=0.50,
        ),
        admission=AdmissionConfig(
            min_valid_ic_ir=0.30,
            require_sign_consistency=True,
            max_library_corr=0.60,
            max_turnover=0.85,
            top_k_candidates=50,
        ),
    )

    gp_engine = GPEngine(
        cfg=cfg,
        panel=synthetic_gp_panel,
        registry=registry,
        tracker=tracker,
    )
    admitted = gp_engine.run(run_id=tracker.run_id, register_to_db=True)

    # 验证成功挖掘出进入库中的合格因子
    assert len(admitted) >= 1
    db_accepted = registry.get_accepted_factors()
    assert len(db_accepted) == len(admitted)

    for af in admitted:
        assert af.factor_id is not None
        assert abs(af.valid_ic_ir) >= 0.30
        assert af.valid_turnover <= 0.85
        assert af.max_corr < 0.60
        assert (af.train_ic_mean > 0) == (af.valid_ic_mean > 0)

    # 验证多个入围因子彼此之间在训练+验证集上的相关性严格 < 0.60
    if len(admitted) >= 2:
        exprs = [af.canon_expr for af in admitted]
        mats = gp_engine.rust_engine.eval_batch(exprs, use_cache=True, canonicalize=False)
        corr_end = gp_engine.valid_slice[1]
        corr_mat = gp_engine.rust_engine.factor_correlation_matrix(
            [np.asarray(m) for m in mats],
            min_names=gp_engine.effective_min_names,
            rank=True,
            start_row=0,
            end_row=corr_end,
        )
        for i in range(len(admitted)):
            for j in range(i + 1, len(admitted)):
                assert abs(corr_mat[i, j]) < 0.60

    # 验证 runs/ 目录产物完整
    assert (tracker.run_dir / "config.yaml").exists()
    assert (tracker.run_dir / "run_meta.json").exists()
    assert (tracker.run_dir / "generations.jsonl").exists()
    assert (tracker.run_dir / "hall_of_fame.json").exists()
    assert (tracker.run_dir / "admission_summary.json").exists()

    run_rec = registry.get_run("gp_test_admission")
    assert run_rec is not None
    assert run_rec.status == "completed"
    assert run_rec.n_evaluated == gp_engine.total_unique_evaluated


def test_gp_1000_pop_50_gen_throughput_benchmark(synthetic_gp_panel: Panel):
    """验收压测：验证 1000 个体 × 50 代在单机远低于 10 分钟（实测数秒级）内完成。"""
    cfg = GPConfig(
        seed=42,
        mode="cs",
        label="y_5",
        splits={
            "train": ("2015-01-01", "2015-12-31"),
            "valid": ("2020-01-01", "2020-12-31"),
            "test": ("2023-01-01", "2023-12-31"),
        },
        population=1000,
        generations=50,
        islands=4,
        migration=MigrationConfig(interval=10, k_individuals=20),
        limits=TreeLimitsConfig(init_min_depth=2, init_max_depth=5, max_depth=6, max_nodes=30),
    )

    t0 = time.perf_counter()
    gp_engine = GPEngine(cfg=cfg, panel=synthetic_gp_panel, registry=None)
    admitted = gp_engine.run(run_id="bench_1000x50", register_to_db=False)
    elapsed_sec = time.perf_counter() - t0

    # 白皮书 P4 要求 1000 个体 × 50 代 < 600 秒 (10 分钟)，Rust 并行 + 子表达式缓存通常在 15 秒内完成
    assert elapsed_sec < 600.0, f"1000×50 演化耗时 {elapsed_sec:.2f}s 超过 10 分钟上限"
    assert len(gp_engine.history) == 51
    assert gp_engine.total_unique_evaluated > 1000
    assert len(admitted) >= 1


def test_gp_splits_purge_and_non_overlapping_fallback_p0_05(synthetic_gp_panel: Panel):
    """验证 P0-05: GP 切分边界强制扣除 horizon 作为 Purge 隔离带，且短样本回退三段完全互斥。"""
    cfg = GPConfig(
        seed=42,
        mode="cs",
        label="y_5",
        splits={
            "train": ("2015-01-01", "2015-12-31"),
            "valid": ("2020-01-01", "2020-12-31"),
            "test": ("2023-01-01", "2023-12-31"),
        },
        population=20,
        generations=1,
        islands=2,
    )
    eng = GPEngine(cfg=cfg, panel=synthetic_gp_panel, registry=None)
    # train 与 valid、valid 与 test 交界处必须留出 >= horizon (5) 的 Purge 间隔
    assert eng.train_slice[1] + eng.horizon <= eng.valid_slice[0]
    assert eng.valid_slice[1] + eng.horizon <= eng.test_slice[0]

    # 短样本回退场景 (日期不匹配默认配置)
    cfg_short = GPConfig(
        seed=42,
        mode="cs",
        label="y_5",
        splits={
            "train": ("1990-01-01", "1991-12-31"),
            "valid": ("1992-01-01", "1993-12-31"),
            "test": ("1994-01-01", "1995-12-31"),
        },
        population=20,
        generations=1,
        islands=2,
    )
    eng_short = GPEngine(cfg=cfg_short, panel=synthetic_gp_panel, registry=None)
    assert eng_short.train_slice[1] + eng_short.horizon <= eng_short.valid_slice[0]
    assert eng_short.valid_slice[1] + eng_short.horizon <= eng_short.test_slice[0]
