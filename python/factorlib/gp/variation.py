"""强类型安全遗传变异与交叉算子管线 (gp/variation.py)。

严格遵循白皮书 9.3 节规范实现五类遗传算子：
1. 强类型子树交叉 (Typed Subtree Crossover, 默认 60%)
2. 子树变异 (Subtree Mutation, 默认 15%)
3. 点变异 (Point Mutation, 默认 10%)
4. 提升变异 (Hoist Mutation, 默认 5%，抑制 Bloat 代码膨胀)
5. 参数微调变异 (Parameter Tuning, 默认 10%)
"""

from __future__ import annotations

import numpy as np

from factorlib.core.config import TreeLimitsConfig, VariationConfig
from factorlib.gp.canon import canonicalize_node
from factorlib.gp.generate import generate_tree
from factorlib.gp.primitives import DataType, PrimitiveSet
from factorlib.gp.tree import ASTNode, ConstantNode, FieldNode, Individual, OpNode, WindowNode


def _is_valid_offspring(ind: Individual, max_depth: int, max_nodes: int) -> bool:
    return (
        not ind.is_degenerate
        and ind.depth <= max_depth
        and ind.n_nodes <= max_nodes
        and ind.root.return_type == DataType.Series
    )


def crossover(
    ind1: Individual,
    ind2: Individual,
    pset: PrimitiveSet,
    rng: np.random.Generator,
    max_depth: int = 6,
    max_nodes: int = 30,
    max_retries: int = 5,
) -> Individual:
    """强类型子树交叉：在父本 A 中选节点 u (类型 τ)，在父本 B 中选同类型 τ 节点 v 替换 u。"""
    nodes1 = ind1.root.iter_nodes_with_path()
    nodes2 = ind2.root.iter_nodes_with_path()

    # 优先选择非根节点
    non_root1 = [(p, n) for p, n in nodes1 if len(p) > 0]
    candidates_u = non_root1 if non_root1 else nodes1

    # 按类型对父本 B 的节点建立索引
    by_type2: dict[DataType, list[tuple[tuple[int, ...], ASTNode]]] = {}
    for p2, n2 in nodes2:
        by_type2.setdefault(n2.return_type, []).append((p2, n2))

    # 优先挑选 Series 或 BoolCond 子树进行交叉
    series_u = [
        (p, n) for p, n in candidates_u if n.return_type in (DataType.Series, DataType.BoolCond)
    ]

    for _ in range(max_retries):
        pool_u = series_u if (series_u and rng.random() < 0.85) else candidates_u
        u_idx = int(rng.integers(0, len(pool_u)))
        path_u, node_u = pool_u[u_idx]
        tau = node_u.return_type

        allowed_depth = max_depth - len(path_u)
        allowed_nodes = max_nodes - (ind1.root.n_nodes - node_u.n_nodes)

        compat_v = [
            n2
            for _, n2 in by_type2.get(tau, [])
            if n2.depth <= allowed_depth
            and n2.n_nodes <= allowed_nodes
            and n2.to_expr() != node_u.to_expr()
        ]
        if not compat_v:
            continue

        v_idx = int(rng.integers(0, len(compat_v)))
        node_v = compat_v[v_idx]
        new_root = ind1.root.replace_at_path(path_u, node_v)
        child = canonicalize_node(new_root, pset=pset, sync_tree=True)
        if _is_valid_offspring(child, max_depth=max_depth, max_nodes=max_nodes):
            return child

    return ind1.clone()


def subtree_mutation(
    ind: Individual,
    pset: PrimitiveSet,
    rng: np.random.Generator,
    max_depth: int = 6,
    max_nodes: int = 30,
    max_subtree_depth: int = 3,
    max_retries: int = 5,
) -> Individual:
    """子树变异：随机选取节点 u (类型 τ)，以 Grow 模式生成最大深度 <= 3 的类型 τ 新子树替换。"""
    all_nodes = ind.root.iter_nodes_with_path()
    series_nodes = [
        (p, n)
        for p, n in all_nodes
        if n.return_type in (DataType.Series, DataType.BoolCond)
        and (max_depth - len(p)) >= n.return_type.min_tree_depth
    ]
    pool = series_nodes if series_nodes else all_nodes

    for _ in range(max_retries):
        idx = int(rng.integers(0, len(pool)))
        path_u, node_u = pool[idx]
        tau = node_u.return_type

        avail_depth = max(tau.min_tree_depth, max_depth - len(path_u))
        sub_max_d = min(max_subtree_depth, avail_depth)
        sub_min_d = min(sub_max_d, max(tau.min_tree_depth, 2 if len(path_u) == 0 else 1))

        new_sub = generate_tree(
            pset=pset,
            rng=rng,
            target_type=tau,
            min_depth=sub_min_d,
            max_depth=sub_max_d,
            method="grow",
        )
        new_root = ind.root.replace_at_path(path_u, new_sub)
        child = canonicalize_node(new_root, pset=pset, sync_tree=True)
        if (
            _is_valid_offspring(child, max_depth=max_depth, max_nodes=max_nodes)
            and child.canon_hash != ind.canon_hash
        ):
            return child

    return ind.clone()


def point_mutation(
    ind: Individual,
    pset: PrimitiveSet,
    rng: np.random.Generator,
    max_depth: int = 6,
    max_nodes: int = 30,
    max_retries: int = 5,
) -> Individual:
    """点变异：随机选取一个算子节点，用同类型签名的同级算子替换（或替换终端特征）。"""
    all_nodes = ind.root.iter_nodes_with_path()
    op_nodes = [
        (p, n)
        for p, n in all_nodes
        if isinstance(n, OpNode) and len(pset.get_compatible_operators(n.op, exclude_self=True)) > 0
    ]
    field_nodes = [(p, n) for p, n in all_nodes if isinstance(n, FieldNode)]

    for _ in range(max_retries):
        if op_nodes and (not field_nodes or rng.random() < 0.80):
            idx = int(rng.integers(0, len(op_nodes)))
            path_u, op_node = op_nodes[idx]
            assert isinstance(op_node, OpNode)
            compat_ops = pset.get_compatible_operators(op_node.op, exclude_self=True)
            new_op = compat_ops[int(rng.integers(0, len(compat_ops)))]
            replaced: ASTNode = OpNode(op=new_op, args=op_node.args)
        elif field_nodes and len(pset.fields) > 1:
            idx = int(rng.integers(0, len(field_nodes)))
            path_u, f_node = field_nodes[idx]
            assert isinstance(f_node, FieldNode)
            other_fields = [f for f in pset.fields if f != f_node.name]
            new_f = other_fields[int(rng.integers(0, len(other_fields)))]
            replaced = FieldNode(new_f)
        else:
            break

        new_root = ind.root.replace_at_path(path_u, replaced)
        child = canonicalize_node(new_root, pset=pset, sync_tree=True)
        if (
            _is_valid_offspring(child, max_depth=max_depth, max_nodes=max_nodes)
            and child.canon_hash != ind.canon_hash
        ):
            return child

    return ind.clone()


def hoist_mutation(
    ind: Individual,
    pset: PrimitiveSet,
    rng: np.random.Generator,
    max_depth: int = 6,
    max_nodes: int = 30,
) -> Individual:
    """提升变异：随机选取子树节点直接提升替换其同类型的祖先算子节点，有效抑制膨胀 (Bloat)。"""
    all_nodes = ind.root.iter_nodes_with_path()
    # 收集所有 (祖先路径, 同类型后代节点) 对
    hoist_pairs: list[tuple[tuple[int, ...], ASTNode]] = []
    non_trivial_pairs: list[tuple[tuple[int, ...], ASTNode]] = []

    for path_anc, node_anc in all_nodes:
        if not isinstance(node_anc, OpNode):
            continue
        descendants = node_anc.iter_nodes_with_path()
        for rel_path, desc_node in descendants:
            if not rel_path:
                continue
            if desc_node.return_type == node_anc.return_type:
                hoist_pairs.append((path_anc, desc_node))
                # 若替换的是根节点，优先保留至少包含 1 个算子的后代子树
                if len(path_anc) > 0 or isinstance(desc_node, OpNode):
                    non_trivial_pairs.append((path_anc, desc_node))

    candidates = non_trivial_pairs if non_trivial_pairs else hoist_pairs
    if not candidates:
        return ind.clone()

    # 随机打乱尝试
    order = rng.permutation(len(candidates))
    for i in order[:5]:
        path_anc, desc_node = candidates[int(i)]
        new_root = ind.root.replace_at_path(path_anc, desc_node)
        child = canonicalize_node(new_root, pset=pset, sync_tree=True)
        if _is_valid_offspring(child, max_depth=max_depth, max_nodes=max_nodes):
            return child

    return ind.clone()


def param_tuning(
    ind: Individual,
    pset: PrimitiveSet,
    rng: np.random.Generator,
    max_depth: int = 6,
    max_nodes: int = 30,
    max_retries: int = 5,
) -> Individual:
    """参数微调变异：保持树形拓扑完全不变，随机扰动某个窗口整数或常数标量。"""
    all_nodes = ind.root.iter_nodes_with_path()
    param_nodes = [
        (p, n)
        for p, n in all_nodes
        if (isinstance(n, WindowNode) and len(pset.windows) > 1)
        or (isinstance(n, ConstantNode) and len(pset.constants) > 1)
    ]

    if not param_nodes:
        # 若树中不含窗口或常数节点，则退化为微调终端特征列
        return point_mutation(ind, pset, rng, max_depth=max_depth, max_nodes=max_nodes)

    for _ in range(max_retries):
        idx = int(rng.integers(0, len(param_nodes)))
        path_p, p_node = param_nodes[idx]

        if isinstance(p_node, WindowNode):
            other_w = [w for w in pset.windows if w != p_node.value]
            new_val_w = other_w[int(rng.integers(0, len(other_w)))]
            new_leaf: ASTNode = WindowNode(int(new_val_w))
        elif isinstance(p_node, ConstantNode):
            other_c = [c for c in pset.constants if abs(c - p_node.value) > 1e-9]
            if not other_c:
                other_c = list(pset.constants)
            new_val_c = other_c[int(rng.integers(0, len(other_c)))]
            new_leaf = ConstantNode(float(new_val_c))
        else:
            continue

        new_root = ind.root.replace_at_path(path_p, new_leaf)
        child = canonicalize_node(new_root, pset=pset, sync_tree=True)
        if (
            _is_valid_offspring(child, max_depth=max_depth, max_nodes=max_nodes)
            and child.canon_hash != ind.canon_hash
        ):
            return child

    return ind.clone()


class VariationPipeline:
    """强类型遗传变异与交叉统一调度管线。"""

    def __init__(
        self,
        pset: PrimitiveSet,
        limits: TreeLimitsConfig | None = None,
        variation_cfg: VariationConfig | None = None,
    ) -> None:
        self.pset = pset
        self.limits = limits or TreeLimitsConfig()
        self.cfg = variation_cfg or VariationConfig()

        raw_probs = np.array(
            [
                self.cfg.crossover_prob,
                self.cfg.subtree_mut_prob,
                self.cfg.point_mut_prob,
                self.cfg.hoist_mut_prob,
                self.cfg.param_mut_prob,
            ],
            dtype=np.float64,
        )
        self.probs = raw_probs / raw_probs.sum()
        self.cum_probs = np.cumsum(self.probs)

    def crossover(
        self,
        ind1: Individual,
        ind2: Individual,
        rng: np.random.Generator,
    ) -> Individual:
        return crossover(
            ind1,
            ind2,
            self.pset,
            rng,
            max_depth=self.limits.max_depth,
            max_nodes=self.limits.max_nodes,
        )

    def subtree_mutation(self, ind: Individual, rng: np.random.Generator) -> Individual:
        return subtree_mutation(
            ind,
            self.pset,
            rng,
            max_depth=self.limits.max_depth,
            max_nodes=self.limits.max_nodes,
        )

    def point_mutation(self, ind: Individual, rng: np.random.Generator) -> Individual:
        return point_mutation(
            ind,
            self.pset,
            rng,
            max_depth=self.limits.max_depth,
            max_nodes=self.limits.max_nodes,
        )

    def hoist_mutation(self, ind: Individual, rng: np.random.Generator) -> Individual:
        return hoist_mutation(
            ind,
            self.pset,
            rng,
            max_depth=self.limits.max_depth,
            max_nodes=self.limits.max_nodes,
        )

    def param_tuning(self, ind: Individual, rng: np.random.Generator) -> Individual:
        return param_tuning(
            ind,
            self.pset,
            rng,
            max_depth=self.limits.max_depth,
            max_nodes=self.limits.max_nodes,
        )

    def apply(
        self,
        parent1: Individual,
        parent2: Individual,
        rng: np.random.Generator,
    ) -> Individual:
        """按配置的五类变异概率分布随机抽取算子产生单个子代。"""
        u = float(rng.random())
        if u < self.cum_probs[0]:
            return self.crossover(parent1, parent2, rng)
        if u < self.cum_probs[1]:
            return self.subtree_mutation(parent1, rng)
        if u < self.cum_probs[2]:
            return self.point_mutation(parent1, rng)
        if u < self.cum_probs[3]:
            return self.hoist_mutation(parent1, rng)
        return self.param_tuning(parent1, rng)
