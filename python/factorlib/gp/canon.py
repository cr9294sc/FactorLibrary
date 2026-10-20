"""Python 侧 AST 代数等价规范化与种群去重包装器 (gp/canon.py)。

严格遵循白皮书 3.1 节与 7.3 节设计：
- 调用 Rust 内核的代数化简引擎（交换律排序、幂等折叠、单调压缩、退化阻断）；
- 将化简后的紧凑 DSL 表达式同步回写为 `Individual` 的 `root`, `canon_expr`, `canon_hash`, `n_nodes`, `depth`；
- 支持单树实时规范化与整代种群 Rayon 批并行规范化去重。
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import factorlib_core

from factorlib.gp.primitives import PrimitiveSet
from factorlib.gp.tree import ASTNode, Individual, parse_dsl_to_tree


def canonicalize_node(
    node: ASTNode,
    pset: PrimitiveSet | None = None,
    sync_tree: bool = True,
) -> Individual:
    """对单个 `ASTNode` 执行 Rust 代数规范化并返回 `Individual`。"""
    raw_expr = node.to_expr()
    available = list(pset.fields) if pset is not None else None
    info: dict[str, Any] = factorlib_core.parse_and_canonicalize(raw_expr, available)

    canon_expr = str(info["canon_expr"])
    is_degen = bool(info["is_degenerate"])
    root_node = node
    if sync_tree and pset is not None and not is_degen and canon_expr != raw_expr:
        try:
            root_node = parse_dsl_to_tree(canon_expr, pset, expected_type=node.return_type)
        except Exception:
            root_node = node

    return Individual(
        root=root_node,
        canon_expr=canon_expr,
        canon_hash=str(info["canon_hash"]),
        hash_u64=int(info["hash_u64"]),
        n_nodes=int(info["n_nodes"]),
        depth=int(info["depth"]),
        is_degenerate=is_degen,
        degenerate_reason=info.get("degenerate_reason"),
    )


def canonicalize_individual(
    ind: Individual,
    pset: PrimitiveSet | None = None,
    sync_tree: bool = True,
) -> Individual:
    """对已有 `Individual` 重新执行代数规范化（重置适应度状态）。"""
    return canonicalize_node(ind.root, pset=pset, sync_tree=sync_tree)


def canonicalize_population(
    individuals: Sequence[Individual],
    pset: PrimitiveSet | None = None,
    engine: Any | None = None,
    sync_tree: bool = True,
) -> list[Individual]:
    """批量规范化一组个体（优先调用 Rust Engine 的 Rayon 并行 `canonicalize_batch`）。"""
    if not individuals:
        return []

    raw_exprs = [ind.root.to_expr() for ind in individuals]
    if engine is not None and hasattr(engine, "canonicalize_batch"):
        batch_infos = list(engine.canonicalize_batch(raw_exprs))
    else:
        available = list(pset.fields) if pset is not None else None
        batch_infos = [factorlib_core.parse_and_canonicalize(e, available) for e in raw_exprs]

    out: list[Individual] = []
    for ind, raw_e, info in zip(individuals, raw_exprs, batch_infos, strict=True):
        canon_expr = str(info["canon_expr"])
        is_degen = bool(info["is_degenerate"])
        root_node = ind.root
        if sync_tree and pset is not None and not is_degen and canon_expr != raw_e:
            try:
                root_node = parse_dsl_to_tree(canon_expr, pset, expected_type=ind.root.return_type)
            except Exception:
                root_node = ind.root

        out.append(
            Individual(
                root=root_node,
                canon_expr=canon_expr,
                canon_hash=str(info["canon_hash"]),
                hash_u64=int(info["hash_u64"]),
                raw_fitness=ind.raw_fitness,
                adjusted_fitness=ind.adjusted_fitness,
                n_nodes=int(info["n_nodes"]),
                depth=int(info["depth"]),
                ic_mean=ind.ic_mean,
                ic_std=ind.ic_std,
                ic_ir=ind.ic_ir,
                coverage=ind.coverage,
                max_lib_corr=ind.max_lib_corr,
                mean_lib_corr=ind.mean_lib_corr,
                is_degenerate=is_degen,
                degenerate_reason=info.get("degenerate_reason"),
                evaluated=ind.evaluated,
            )
        )
    return out


def deduplicate_individuals(
    individuals: Sequence[Individual],
    exclude_degenerate: bool = True,
) -> list[Individual]:
    """按 64 位规范化哈希 `canon_hash` 去重，并可选剔除退化个体。"""
    seen: dict[str, Individual] = {}
    for ind in individuals:
        if exclude_degenerate and ind.is_degenerate:
            continue
        h = ind.canon_hash
        if not h:
            continue
        prev = seen.get(h)
        if prev is None or ind.adjusted_fitness > prev.adjusted_fitness:
            seen[h] = ind
    return list(seen.values())
