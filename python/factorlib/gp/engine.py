"""强类型遗传规划 (STGP) 主循环演化控制器 (gp/engine.py)。

严格遵循白皮书第 9 章与 14.2 节架构设计：
- 整合多岛演化模型 (`Island`)、环形拓扑迁移 (`RingMigrationController`)、
  强类型变异交叉管线 (`VariationPipeline`)、Rust 极速批求值 (`Engine.eval_gp_batch`)、
  名人堂 (`HallOfFame`) 以及五步正交化准入管线 (`AdmissionPipeline`)；
- 综合适应度函数严格执行：
  `F(expr) = |ICIR_train| - lambda * NodesCount - Penalty_cov - Penalty_corr`；
- 固定随机种子下 100% 确定性复现演化轨迹与挖掘结果。
"""

from __future__ import annotations

import logging
import math
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from factorlib.core.config import GPConfig
from factorlib.core.panel import Panel
from factorlib.core.registry import FactorRegistry
from factorlib.core.runs import RunTracker
from factorlib.gp.admission import AdmissionPipeline, AdmissionSummary, AdmittedFactor
from factorlib.gp.canon import canonicalize_population
from factorlib.gp.islands import Island, RingMigrationController
from factorlib.gp.primitives import PrimitiveSet
from factorlib.gp.select import HallOfFame
from factorlib.gp.tree import Individual
from factorlib.gp.variation import VariationPipeline

logger = logging.getLogger(__name__)


@dataclass
class GenerationStats:
    """单代演化统计指标快照。"""

    generation: int
    pop_size: int
    unique_in_pop: int
    new_evaluated: int
    total_unique_evaluated: int
    best_fitness: float
    mean_fitness: float
    best_ic_ir: float
    best_ic_mean: float
    mean_nodes: float
    mean_depth: float
    migrated: int
    elapsed_ms: float
    best_expr: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _extract_horizon_from_label(label_name: str, default: int = 5) -> int:
    if label_name.startswith("y_"):
        tail = label_name[2:]
        if tail.isdigit():
            return max(1, int(tail))
    return default


class GPEngine:
    """强类型遗传规划 (STGP) 批量因子挖掘引擎。"""

    def __init__(
        self,
        cfg: GPConfig,
        panel: Panel,
        registry: FactorRegistry | None = None,
        pset: PrimitiveSet | None = None,
        tracker: RunTracker | None = None,
    ) -> None:
        self.cfg = cfg
        self.panel = panel
        self.registry = registry
        self.tracker = tracker
        self.rust_engine = panel.to_rust_engine()

        # 解析目标预测标签与持有期 h
        if self.cfg.label in self.rust_engine.label_names:
            self.label_name = self.cfg.label
        elif "y_5" in self.rust_engine.label_names:
            self.label_name = "y_5"
        elif self.rust_engine.label_names:
            self.label_name = self.rust_engine.label_names[0]
        else:
            raise ValueError("Panel 中未包含任何可用预测标签 (labels)")

        self.horizon = _extract_horizon_from_label(self.label_name, default=5)

        # 解析训练/验证/测试行索引切片
        (
            self.train_slice,
            self.valid_slice,
            self.test_slice,
            self.effective_splits,
            self.effective_min_names,
            self.train_active_days,
        ) = self._resolve_splits()

        # 根据有效训练集长度自适应确定最大可用窗口（长历史面板保持全部 250，短样本面板避免窗口 > 样本长导致覆盖率归零）
        max_cov_win: int | None = None
        if self.train_active_days < 300:
            max_cov_win = max(
                5,
                int(self.train_slice[1] * (1.0 - self.cfg.fitness.min_coverage + 0.20)),
            )

        self.pset = (
            pset if pset is not None else PrimitiveSet.from_panel(panel, max_window=max_cov_win)
        )
        self.variation = VariationPipeline(
            pset=self.pset,
            limits=self.cfg.limits,
            variation_cfg=self.cfg.variation,
        )
        self.migration_controller = RingMigrationController(
            interval=self.cfg.migration.interval,
            k_individuals=self.cfg.migration.k_individuals,
        )

        self.islands: list[Island] = []
        self.hall_of_fame = HallOfFame(capacity=self.cfg.hall_of_fame_size)
        self.history: list[GenerationStats] = []
        self.last_admission_summary: AdmissionSummary | None = None

        # 缓存已评估过的规范化哈希结果，避免跨代重复统计
        self._fitness_cache: dict[str, dict[str, Any]] = {}
        self._library_train_mats: list[np.ndarray] = self._load_library_train_matrices()

    def _resolve_splits(
        self,
    ) -> tuple[
        tuple[int, int],
        tuple[int, int],
        tuple[int, int],
        dict[str, tuple[str, str]],
        int,
        int,
    ]:
        """解析面板日期对应的 (train, valid, test) 连续行索引区间，并在短样本面板上优雅回退。"""
        dates_str = np.array([str(d) for d in self.panel.dates])
        T = len(dates_str)

        # 计算有效在池标的数以自适应 min_names
        active_per_day = np.sum(self.panel.mask, axis=1)
        max_active = int(np.max(active_per_day)) if T > 0 else 0
        eff_min_names = min(self.cfg.min_names, max(2, max_active))

        lbl_mat = self.panel.get_label(self.label_name)
        valid_lbl_per_day = np.sum(np.isfinite(lbl_mat) & self.panel.mask, axis=1)

        def _find_slice(start_d: str, end_d: str) -> tuple[int, int] | None:
            idx = np.where((dates_str >= start_d) & (dates_str <= end_d))[0]
            if len(idx) >= 3:
                return int(idx[0]), int(idx[-1]) + 1
            return None

        tr_cfg = self.cfg.splits.get("train", ("2011-01-01", "2018-12-31"))
        va_cfg = self.cfg.splits.get("valid", ("2019-01-01", "2021-12-31"))
        te_cfg = self.cfg.splits.get("test", ("2022-01-01", "2026-08-31"))

        tr_s = _find_slice(tr_cfg[0], tr_cfg[1])
        va_s = _find_slice(va_cfg[0], va_cfg[1])
        te_s = _find_slice(te_cfg[0], te_cfg[1])

        if tr_s is not None and va_s is not None:
            # P0-05: 在 train 与 valid、valid 与 test 交界处强制扣除 self.horizon 作为 Purge 隔离带，
            # 防止边界末尾 h 个交易日的前瞻收益标签跨越切分边界泄露
            if tr_s[1] > va_s[0] - self.horizon and (tr_s[1] - tr_s[0]) > self.horizon + 3:
                tr_s = (tr_s[0], max(tr_s[0] + 3, min(tr_s[1], va_s[0] - self.horizon)))

            if te_s is not None:
                if va_s[1] > te_s[0] - self.horizon and (va_s[1] - va_s[0]) > self.horizon + 3:
                    va_s = (va_s[0], max(va_s[0] + 3, min(va_s[1], te_s[0] - self.horizon)))
            else:
                # 若面板未覆盖默认 test 日期，优先取 valid 之后的独立区间；否则从 valid 尾部切出独立 test
                rem_start = min(T, va_s[1] + self.horizon)
                if T - rem_start >= 3:
                    te_s = (rem_start, T)
                elif (va_s[1] - va_s[0]) >= 6 + self.horizon:
                    mid = va_s[0] + max(3, (va_s[1] - va_s[0] - self.horizon) // 2)
                    te_s = (min(va_s[1] - 3, mid + self.horizon), va_s[1])
                    va_s = (va_s[0], mid)
                else:
                    mid = va_s[0] + max(1, (va_s[1] - va_s[0]) // 2)
                    te_s = (mid, va_s[1])
                    va_s = (va_s[0], mid)

            eff_splits = {
                "train": (str(dates_str[tr_s[0]]), str(dates_str[tr_s[1] - 1])),
                "valid": (str(dates_str[va_s[0]]), str(dates_str[va_s[1] - 1])),
                "test": (str(dates_str[te_s[0]]), str(dates_str[te_s[1] - 1])),
            }
            tr_active = int(np.sum(valid_lbl_per_day[tr_s[0] : tr_s[1]] >= eff_min_names))
            return tr_s, va_s, te_s, eff_splits, eff_min_names, max(1, tr_active)

        # 若面板日期范围不在默认 2011-2021 区间内（如样本面板 v1 或单测小面板），
        # P0-05: 严格按 60% / 20% / 20% 切分为三段完全互斥的 (train, valid, test) 并插入 Purge 隔离带
        valid_lbl_rows = np.where(valid_lbl_per_day >= eff_min_names)[0]
        if len(valid_lbl_rows) >= 9:
            eff_start = int(valid_lbl_rows[0])
            eff_end = int(valid_lbl_rows[-1]) + 1
            span = eff_end - eff_start
            purge = self.horizon if span >= 9 + 2 * self.horizon else max(0, (span - 9) // 2)
            usable = span - 2 * purge
            n_train = max(3, int(round(usable * 0.60)))
            n_valid = max(3, int(round(usable * 0.20)))
            if usable - n_train - n_valid < 3:
                n_valid = max(3, (usable - 3) // 2)
                n_train = max(3, usable - n_valid - 3)
            tr_fallback = (eff_start, eff_start + n_train)
            va_start = tr_fallback[1] + purge
            va_fallback = (va_start, va_start + n_valid)
            te_start = va_fallback[1] + purge
            te_fallback = (te_start, max(te_start + 1, eff_end))
        elif len(valid_lbl_rows) >= 6:
            eff_start = int(valid_lbl_rows[0])
            eff_end = int(valid_lbl_rows[-1]) + 1
            span = eff_end - eff_start
            n_train = max(2, span // 2)
            n_valid = max(2, (span - n_train) // 2)
            tr_fallback = (eff_start, eff_start + n_train)
            va_fallback = (eff_start + n_train, eff_start + n_train + n_valid)
            te_fallback = (eff_start + n_train + n_valid, eff_end)
        else:
            tr_fallback = (0, T)
            va_fallback = (0, T)
            te_fallback = (0, T)

        eff_splits = {
            "train": (
                str(dates_str[tr_fallback[0]]),
                str(dates_str[max(0, tr_fallback[1] - 1)]),
            ),
            "valid": (
                str(dates_str[va_fallback[0]]),
                str(dates_str[max(0, va_fallback[1] - 1)]),
            ),
            "test": (
                str(dates_str[te_fallback[0]]),
                str(dates_str[max(0, te_fallback[1] - 1)]),
            ),
        }
        tr_active = int(np.sum(valid_lbl_per_day[tr_fallback[0] : tr_fallback[1]] >= eff_min_names))
        logger.info(
            "面板日期范围 [%s, %s] 未完整覆盖配置切分，自动采用有效在池区间切分: train=%s (%d有效日), valid=%s",
            dates_str[0] if T > 0 else "N/A",
            dates_str[-1] if T > 0 else "N/A",
            eff_splits["train"],
            tr_active,
            eff_splits["valid"],
        )
        return tr_fallback, va_fallback, te_fallback, eff_splits, eff_min_names, max(1, tr_active)

    def _load_library_train_matrices(self) -> list[np.ndarray]:
        """预加载 SQLite 注册表中已入库因子的训练集矩阵，用于计算冗余惩罚。"""
        if self.registry is None:
            return []
        lib_factors = self.registry.list_factors(status="accepted") + self.registry.list_factors(
            status="active"
        )
        if not lib_factors:
            return []

        exprs = list({f.canon_expr for f in lib_factors})
        raw_mats = self.rust_engine.eval_batch_safe(exprs, use_cache=True, canonicalize=False)
        s_row, e_row = self.train_slice
        mats: list[np.ndarray] = []
        for m in raw_mats:
            if m is not None:
                arr = np.asfortranarray(np.asarray(m)[s_row:e_row])
                mats.append(arr)
        return mats

    @property
    def total_unique_evaluated(self) -> int:
        return len(self._fitness_cache)

    def initialize_population(self) -> None:
        """按配置构建多个岛屿并进行强类型 Ramped Half-and-Half 初始化与首代评估。"""
        self.islands.clear()
        n_islands = self.cfg.islands
        base_pop = self.cfg.population // n_islands
        remainder = self.cfg.population % n_islands

        for i in range(n_islands):
            pop_i = base_pop + (1 if i < remainder else 0)
            # 白皮书 9.1 节：各岛采用独立确定性种子 (如 base_seed + 101 + i)
            seed_i = self.cfg.seed + 101 + i
            island = Island(
                island_id=i,
                pop_size=pop_i,
                seed=seed_i,
                hof_capacity=self.cfg.hall_of_fame_size,
            )
            island.initialize(
                pset=self.pset,
                min_depth=self.cfg.limits.init_min_depth,
                max_depth=self.cfg.limits.init_max_depth,
                max_nodes=self.cfg.limits.max_nodes,
            )
            self.islands.append(island)

    def _compute_adjusted_fitness(
        self,
        raw_fitness: float,
        n_nodes: int,
        coverage: float,
        max_abs_corr: float,
    ) -> float:
        """按照白皮书 9.4 节公式计算综合适应度：
        F(expr) = |ICIR_train| - lambda * NodesCount - Penalty_cov - Penalty_corr
        """
        if not math.isfinite(raw_fitness) or coverage < self.cfg.fitness.min_coverage:
            return -10.0

        parsimony_pen = self.cfg.fitness.parsimony_penalty * float(n_nodes)
        corr_pen = (
            self.cfg.fitness.corr_penalty_factor * raw_fitness
            if max_abs_corr > self.cfg.fitness.corr_penalty_threshold
            else 0.0
        )
        return raw_fitness - parsimony_pen - corr_pen

    def evaluate_individuals(self, individuals: Sequence[Individual]) -> int:
        """对一批个体进行批量规范化、Rust 并行求值与综合适应度指派。返回新求值的唯一表达式数。"""
        if not individuals:
            return 0

        # 收集尚未在 _fitness_cache 中评估过的唯一非退化表达式
        pending_by_hash: dict[str, str] = {}
        for ind in individuals:
            if ind.is_degenerate or not ind.canon_hash:
                ind.raw_fitness = 0.0
                ind.adjusted_fitness = -10.0
                ind.evaluated = True
                continue
            if ind.depth > self.cfg.limits.max_depth or ind.n_nodes > self.cfg.limits.max_nodes:
                ind.raw_fitness = 0.0
                ind.adjusted_fitness = -10.0
                ind.evaluated = True
                continue
            if ind.canon_hash not in self._fitness_cache:
                pending_by_hash[ind.canon_hash] = ind.canon_expr

        new_eval_count = len(pending_by_hash)
        if new_eval_count > 0:
            hashes_list = list(pending_by_hash.keys())
            exprs_list = [pending_by_hash[h] for h in hashes_list]
            s_row, e_row = self.train_slice

            batch_res = self.rust_engine.eval_gp_batch(
                exprs_list,
                label_name=self.label_name,
                start_row=s_row,
                end_row=e_row,
                min_names=self.effective_min_names,
                horizon=self.horizon,
                library_factors=self._library_train_mats if self._library_train_mats else None,
                use_cache=True,
            )

            for h_key, res_dict in zip(hashes_list, batch_res, strict=True):
                self._fitness_cache[h_key] = dict(res_dict)

        # 将缓存指标回填至所有个体并计算综合适应度
        train_total_days = max(1, self.train_active_days)
        min_valid_days = max(2, int(math.floor(train_total_days * self.cfg.fitness.min_coverage)))

        for ind in individuals:
            if ind.is_degenerate or not ind.canon_hash:
                continue
            if ind.depth > self.cfg.limits.max_depth or ind.n_nodes > self.cfg.limits.max_nodes:
                continue

            cached = self._fitness_cache.get(ind.canon_hash)
            valid_days = int(cached.get("valid_days", 0)) if cached is not None else 0
            if cached is None or not cached.get("valid", False) or valid_days == 0:
                ind.raw_fitness = 0.0
                ind.adjusted_fitness = -10.0
                ind.evaluated = True
                continue

            ind.ic_mean = float(cached["ic_mean"])
            ind.ic_std = float(cached["ic_std"])
            ind.ic_ir = float(cached["ic_ir"])
            cell_cov = float(cached["coverage"])
            day_cov = float(valid_days) / float(train_total_days)
            ind.coverage = min(cell_cov, day_cov) if valid_days < min_valid_days else cell_cov
            ind.max_lib_corr = float(cached["max_abs_corr"])
            ind.mean_lib_corr = float(cached["mean_abs_corr"])

            raw_metric = abs(ind.ic_ir) if self.cfg.fitness.metric == "ic_ir" else abs(ind.ic_mean)
            if not math.isfinite(raw_metric):
                raw_metric = 0.0

            ind.raw_fitness = raw_metric
            ind.adjusted_fitness = self._compute_adjusted_fitness(
                raw_fitness=raw_metric,
                n_nodes=ind.n_nodes,
                coverage=ind.coverage,
                max_abs_corr=ind.max_lib_corr,
            )
            ind.evaluated = True

        return new_eval_count

    def _collect_all_population(self) -> list[Individual]:
        all_pop: list[Individual] = []
        for isl in self.islands:
            all_pop.extend(isl.population)
        return all_pop

    def _record_generation_stats(
        self,
        gen_idx: int,
        new_eval_count: int,
        migrated_count: int,
        elapsed_ms: float,
    ) -> GenerationStats:
        all_pop = self._collect_all_population()
        for isl in self.islands:
            isl.update_hall_of_fame()
        self.hall_of_fame.update(all_pop)

        valid_inds = [
            ind
            for ind in all_pop
            if not ind.is_degenerate
            and math.isfinite(ind.adjusted_fitness)
            and ind.adjusted_fitness > -5.0
        ]
        if not valid_inds:
            valid_inds = all_pop

        best_ind = max(
            valid_inds,
            key=lambda x: (
                x.adjusted_fitness if math.isfinite(x.adjusted_fitness) else -1e18,
                -x.n_nodes,
            ),
        )
        mean_fit = float(np.mean([x.adjusted_fitness for x in valid_inds])) if valid_inds else -10.0
        mean_nodes = float(np.mean([x.n_nodes for x in all_pop])) if all_pop else 0.0
        mean_depth = float(np.mean([x.depth for x in all_pop])) if all_pop else 0.0
        unique_hashes = len({x.canon_hash for x in all_pop if x.canon_hash})

        stats = GenerationStats(
            generation=gen_idx,
            pop_size=len(all_pop),
            unique_in_pop=unique_hashes,
            new_evaluated=new_eval_count,
            total_unique_evaluated=self.total_unique_evaluated,
            best_fitness=float(best_ind.adjusted_fitness),
            mean_fitness=mean_fit,
            best_ic_ir=float(best_ind.ic_ir),
            best_ic_mean=float(best_ind.ic_mean),
            mean_nodes=mean_nodes,
            mean_depth=mean_depth,
            migrated=migrated_count,
            elapsed_ms=round(elapsed_ms, 2),
            best_expr=best_ind.canon_expr,
        )
        self.history.append(stats)
        if self.tracker is not None:
            self.tracker.log_generation(stats)
        return stats

    def evolve_one_generation(self, gen_idx: int) -> GenerationStats:
        """单代演化：变异交叉 -> 规范化去重 -> 下发 Rust 批求值 -> 适应度指派 -> 环形迁移。"""
        t0 = time.perf_counter()

        if not self.islands:
            self.initialize_population()

        # 第 0 代：直接评估初始种群；后续代：先执行繁衍变异再评估
        if gen_idx > 0:
            for isl in self.islands:
                isl.breed_next_generation(
                    variation=self.variation,
                    tournament_size=self.cfg.tournament_size,
                    elitism=self.cfg.elitism,
                )
            # 批量规范化同步
            for isl in self.islands:
                isl.population = canonicalize_population(
                    isl.population,
                    pset=self.pset,
                    engine=self.rust_engine,
                    sync_tree=True,
                )

        all_pop = self._collect_all_population()
        new_eval_count = self.evaluate_individuals(all_pop)

        # 检查是否触发多岛环形迁移
        migrated_count = 0
        if self.migration_controller.should_migrate(gen_idx, len(self.islands)):
            migrated_count = self.migration_controller.migrate(self.islands)

        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        stats = self._record_generation_stats(
            gen_idx=gen_idx,
            new_eval_count=new_eval_count,
            migrated_count=migrated_count,
            elapsed_ms=elapsed_ms,
        )
        return stats

    def run(
        self,
        run_id: str | None = None,
        register_to_db: bool = True,
    ) -> list[AdmittedFactor]:
        """执行完整多代演化循环，收敛后调用五步准入管线进行筛选与落库。"""
        effective_run_id = run_id or (
            self.tracker.run_id if self.tracker is not None else f"gp_seed_{self.cfg.seed}"
        )
        data_version = str(self.panel.meta.get("panel_version", "v1"))

        if not self.islands:
            self.initialize_population()

        self.history.clear()

        # Gen 0 (初始种群评估) + Gen 1..generations 演化
        for gen_idx in range(self.cfg.generations + 1):
            stats = self.evolve_one_generation(gen_idx)
            if gen_idx == 0 or gen_idx % 5 == 0 or gen_idx == self.cfg.generations:
                logger.info(
                    "[Gen %02d/%02d] BestFit=%.4f | BestICIR=%+.3f | Unique=%d/%d | NewEval=%d | Nodes=%.1f | %.1fms | %s",
                    gen_idx,
                    self.cfg.generations,
                    stats.best_fitness,
                    stats.best_ic_ir,
                    stats.unique_in_pop,
                    stats.pop_size,
                    stats.new_evaluated,
                    stats.mean_nodes,
                    stats.elapsed_ms,
                    stats.best_expr,
                )

        # 合并所有岛屿的名人堂与全局名人堂
        for isl in self.islands:
            self.hall_of_fame.update(list(isl.hall_of_fame))

        candidates = self.hall_of_fame.top_k(self.cfg.admission.top_k_candidates)

        # 执行五步准入管线
        admission_pipe = AdmissionPipeline(
            cfg=self.cfg.admission,
            label_name=self.label_name,
            horizon=self.horizon,
            min_names=self.effective_min_names,
            mode=self.cfg.mode,
            data_version=data_version,
            splits=self.effective_splits,
        )
        corr_slice = (self.train_slice[0], self.valid_slice[1])
        summary = admission_pipe.run(
            candidates=candidates,
            engine=self.rust_engine,
            panel=self.panel,
            valid_slice=self.valid_slice,
            corr_slice=corr_slice,
            registry=self.registry,
            run_id=effective_run_id,
            register_to_db=register_to_db,
        )
        self.last_admission_summary = summary

        if self.tracker is not None:
            self.tracker.save_json(
                "hall_of_fame.json",
                [ind.to_dict() for ind in candidates],
            )
            self.tracker.save_json("admission_summary.json", summary.to_dict())
            self.tracker.finish(status="completed", n_evaluated=self.total_unique_evaluated)

        return summary.admitted
