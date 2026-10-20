"""锦标赛选择、精英保留与全局名人堂管理 (gp/select.py)。

严格遵循白皮书 3.1、9.1、9.5 与 14.2 节设计：
- 锦标赛选择（Tournament Selection）：按 `adjusted_fitness` 竞争，相同时优先选择节点数更少的简约树；
- 精英保留（Elitism）：保证每代岛屿最优个体不因随机变异丢失；
- 名人堂（HallOfFame）：按唯一指纹 `canon_hash` 去重维护演化历史 Top-K 精英个体，供准入管线初筛。
"""

from __future__ import annotations

import math
from collections.abc import Iterator, Sequence

import numpy as np

from factorlib.gp.tree import Individual


def _fitness_sort_key(ind: Individual) -> tuple[float, int, str]:
    """排序键：适应度越高越优 -> 节点数越少越优 (奥卡姆剃刀) -> 字典序确定性平局决胜。"""
    fit = ind.adjusted_fitness if math.isfinite(ind.adjusted_fitness) else -1e18
    return (-fit, ind.n_nodes, ind.canon_hash)


def tournament_select(
    population: Sequence[Individual],
    rng: np.random.Generator,
    tournament_size: int = 7,
) -> Individual:
    """从种群中随机抽取 `tournament_size` 个个体进行锦标赛竞争，返回最优者副本。"""
    n = len(population)
    if n == 0:
        raise ValueError("空种群无法执行锦标赛选择")
    k = max(1, min(int(tournament_size), n))
    indices = rng.choice(n, size=k, replace=False)
    contenders = [population[int(i)] for i in indices]
    winner = min(contenders, key=_fitness_sort_key)
    return winner.clone()


def select_elites(
    population: Sequence[Individual],
    k: int = 5,
    deduplicate: bool = True,
) -> list[Individual]:
    """选取种群中适应度最高的 Top-K 精英个体。"""
    if k <= 0 or not population:
        return []

    valid = [ind for ind in population if not ind.is_degenerate]
    if not valid:
        valid = list(population)

    sorted_pop = sorted(valid, key=_fitness_sort_key)
    elites: list[Individual] = []
    seen_hashes: set[str] = set()

    for ind in sorted_pop:
        if deduplicate and ind.canon_hash in seen_hashes:
            continue
        seen_hashes.add(ind.canon_hash)
        elites.append(ind.clone())
        if len(elites) >= k:
            break

    return elites


class HallOfFame:
    """跨代数、跨岛屿全局唯一名人堂（Hall of Fame）。"""

    def __init__(self, capacity: int = 200) -> None:
        if capacity < 1:
            raise ValueError("HallOfFame 容量必须 >= 1")
        self.capacity = int(capacity)
        self._by_hash: dict[str, Individual] = {}
        self._sorted: list[Individual] = []

    def update(self, individuals: Sequence[Individual]) -> None:
        """将新评估的一批个体合并入名人堂，自动按 `canon_hash` 去重并截断至 `capacity`。"""
        changed = False
        for ind in individuals:
            if ind.is_degenerate or not ind.canon_hash:
                continue
            if not math.isfinite(ind.adjusted_fitness) or ind.adjusted_fitness <= -5.0:
                continue
            existing = self._by_hash.get(ind.canon_hash)
            if existing is None or ind.adjusted_fitness > existing.adjusted_fitness:
                self._by_hash[ind.canon_hash] = ind.clone()
                changed = True

        if changed:
            all_items = sorted(self._by_hash.values(), key=_fitness_sort_key)
            if len(all_items) > self.capacity:
                all_items = all_items[: self.capacity]
                self._by_hash = {item.canon_hash: item for item in all_items}
            self._sorted = all_items

    def top_k(self, k: int | None = None) -> list[Individual]:
        """获取按适应度降序排列的前 `k` 个不重复精英个体。"""
        if k is None or k >= len(self._sorted):
            return [ind.clone() for ind in self._sorted]
        return [ind.clone() for ind in self._sorted[: max(0, int(k))]]

    def best(self) -> Individual | None:
        """返回当前排名第一的最优个体。"""
        return self._sorted[0].clone() if self._sorted else None

    def __len__(self) -> int:
        return len(self._sorted)

    def __iter__(self) -> Iterator[Individual]:
        return iter(self._sorted)
