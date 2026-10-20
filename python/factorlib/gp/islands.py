"""多岛演化模型与环形拓扑迁移管理 (gp/islands.py)。

严格遵循白皮书 9.1 节规范：
- 支持构建多个彼此独立的演化岛屿（默认 4 岛），各岛使用独立确定性随机种子（`np.random.default_rng`）；
- 采用环形拓扑结构（Ring Topology：`Island i -> Island (i + 1) % K`）；
- 每隔固定代数（默认 10 代），各岛将排名前 5%（或配置数量）的精英个体向下一相邻岛屿单向迁移，
  替换目标岛屿中适应度最差的个体，防止种群早熟收敛并维持跨岛基因多样性。
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from factorlib.gp.generate import generate_ramped_half_and_half
from factorlib.gp.primitives import PrimitiveSet
from factorlib.gp.select import HallOfFame, _fitness_sort_key, select_elites, tournament_select
from factorlib.gp.tree import Individual
from factorlib.gp.variation import VariationPipeline


class Island:
    """独立演化岛屿容器。"""

    def __init__(
        self,
        island_id: int,
        pop_size: int,
        seed: int,
        hof_capacity: int = 100,
    ) -> None:
        if pop_size < 1:
            raise ValueError(f"岛屿种群规模必须 >= 1, 实际为 {pop_size}")
        self.island_id = int(island_id)
        self.pop_size = int(pop_size)
        self.seed = int(seed)
        self.rng = np.random.default_rng(self.seed)
        self.population: list[Individual] = []
        self.hall_of_fame = HallOfFame(capacity=max(10, hof_capacity))

    def initialize(
        self,
        pset: PrimitiveSet,
        min_depth: int = 2,
        max_depth: int = 5,
        max_nodes: int = 30,
    ) -> list[Individual]:
        """采用 Ramped Half-and-Half 初始化本岛种群。"""
        self.population = generate_ramped_half_and_half(
            pset=pset,
            pop_size=self.pop_size,
            rng=self.rng,
            min_depth=min_depth,
            max_depth=max_depth,
            max_nodes=max_nodes,
            deduplicate=True,
        )
        return self.population

    def update_hall_of_fame(self) -> None:
        """将当前已评估的本岛种群同步更新至本岛名人堂。"""
        self.hall_of_fame.update(self.population)

    def breed_next_generation(
        self,
        variation: VariationPipeline,
        tournament_size: int = 7,
        elitism: int = 5,
    ) -> list[Individual]:
        """通过精英保留 + 锦标赛选择 + 五类强类型变异算子繁衍下一代种群。"""
        if not self.population:
            raise RuntimeError(f"Island {self.island_id} 种群尚未初始化")

        n_elite = min(max(0, int(elitism)), self.pop_size)
        next_pop: list[Individual] = select_elites(self.population, k=n_elite, deduplicate=True)
        seen_hashes: set[str] = {ind.canon_hash for ind in next_pop if ind.canon_hash}

        while len(next_pop) < self.pop_size:
            p1 = tournament_select(self.population, self.rng, tournament_size=tournament_size)
            p2 = tournament_select(self.population, self.rng, tournament_size=tournament_size)

            child = variation.apply(p1, p2, self.rng)
            # 若产生了已在下一代中重复的个体，尝试一次额外点变异或参数微调增加多样性
            if child.canon_hash in seen_hashes:
                alt = variation.subtree_mutation(p1, self.rng)
                if alt.canon_hash not in seen_hashes and not alt.is_degenerate:
                    child = alt

            if child.canon_hash:
                seen_hashes.add(child.canon_hash)
            next_pop.append(child)

        self.population = next_pop
        return self.population

    def get_migrants(self, k: int) -> list[Individual]:
        """提取本岛排名前 `k` 的精英个体用于跨岛迁移。"""
        return select_elites(self.population, k=k, deduplicate=True)

    def replace_worst(self, immigrants: Sequence[Individual]) -> int:
        """用迁入的精英个体替换本岛中表现最差的个体（自动跳过已存在的同哈希个体）。"""
        if not immigrants or not self.population:
            return 0

        # 按适应度从优到劣排序（最差个体在末尾）
        sorted_pop = sorted(self.population, key=_fitness_sort_key)
        existing_hashes = {ind.canon_hash for ind in sorted_pop if ind.canon_hash}

        unique_immigrants: list[Individual] = []
        for imm in immigrants:
            if imm.is_degenerate or not imm.canon_hash:
                continue
            if imm.canon_hash not in existing_hashes:
                unique_immigrants.append(imm.clone())
                existing_hashes.add(imm.canon_hash)

        if not unique_immigrants:
            return 0

        n_replace = min(len(unique_immigrants), max(1, len(sorted_pop) // 2))
        for i in range(n_replace):
            sorted_pop[-(i + 1)] = unique_immigrants[i]

        self.population = sorted_pop
        self.update_hall_of_fame()
        return n_replace


class RingMigrationController:
    """多岛环形拓扑迁移控制器（Ring Topology: `Island 0 -> 1 -> 2 -> 3 -> 0`）。"""

    def __init__(
        self,
        interval: int = 10,
        k_individuals: int | None = None,
        migration_rate: float = 0.05,
    ) -> None:
        if interval < 1:
            raise ValueError("migration.interval 必须 >= 1")
        self.interval = int(interval)
        self.k_individuals = k_individuals
        self.migration_rate = float(migration_rate)

    def should_migrate(self, gen_idx: int, n_islands: int) -> bool:
        """判断当前代数（1-based 或 0-based 正整数代）是否触发环形迁移。"""
        return n_islands >= 2 and gen_idx > 0 and (gen_idx % self.interval == 0)

    def resolve_migrant_count(self, island_pop_size: int) -> int:
        """计算单岛每次迁出的精英个体数量（优先遵循 5% 比例并受限于岛容量）。"""
        rate_k = max(1, int(round(island_pop_size * self.migration_rate)))
        if self.k_individuals is None or self.k_individuals <= 0:
            return rate_k
        # 若配置的 k_individuals 针对总种群（如 1000 * 2% = 20），限制不超过单岛的 20%
        max_per_island = max(1, island_pop_size // 5)
        return min(int(self.k_individuals), max_per_island)

    def migrate(self, islands: Sequence[Island]) -> int:
        """执行单向环形迁移：`Island i` 的精英迁入 `Island (i + 1) % N`。"""
        n_islands = len(islands)
        if n_islands < 2:
            return 0

        # 1. 先从各岛同步提取待迁出精英，避免链式覆盖污染
        outgoing: list[list[Individual]] = []
        for isl in islands:
            k = self.resolve_migrant_count(isl.pop_size)
            outgoing.append(isl.get_migrants(k))

        # 2. 按环形拓扑注入相邻下一岛屿
        total_migrated = 0
        for i, migrants in enumerate(outgoing):
            target_island = islands[(i + 1) % n_islands]
            total_migrated += target_island.replace_worst(migrants)

        return total_migrated
