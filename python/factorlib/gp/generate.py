"""强类型 Ramped Half-and-Half 初始种群生成算法 (gp/generate.py)。

严格遵循白皮书 9.2 节规范：
- 支持限制树深（默认 `min_depth=2`, `max_depth=5`）与最大节点数；
- 种群 50% 采用 Grow 模式（随机选择算子或终端叶子，生成非对称形态树）；
- 种群 50% 采用 Full 模式（在达到最大深度前只选择非终端算子，生成平衡饱满树）；
- 强类型递归保护：对 `Series`, `Window`, `Constant`, `BoolCond` 四类节点严格按类型签名与最小合法深度展开，
  保证生成的每一棵语法树 100% 通过 Rust 静态类型检查器。
"""

from __future__ import annotations

import numpy as np

from factorlib.gp.canon import canonicalize_node
from factorlib.gp.primitives import DataType, OperatorPrimitive, PrimitiveSet
from factorlib.gp.tree import ASTNode, ConstantNode, FieldNode, Individual, OpNode, WindowNode


def _sample_field(
    pset: PrimitiveSet, rng: np.random.Generator, avoid: str | None = None
) -> FieldNode:
    if avoid is not None and len(pset.fields) > 1:
        candidates = [f for f in pset.fields if f != avoid]
        idx = int(rng.integers(0, len(candidates)))
        return FieldNode(candidates[idx])
    idx = int(rng.integers(0, len(pset.fields)))
    return FieldNode(pset.fields[idx])


def _sample_window(pset: PrimitiveSet, rng: np.random.Generator) -> WindowNode:
    idx = int(rng.integers(0, len(pset.windows)))
    return WindowNode(int(pset.windows[idx]))


def _sample_constant(pset: PrimitiveSet, rng: np.random.Generator) -> ConstantNode:
    idx = int(rng.integers(0, len(pset.constants)))
    return ConstantNode(float(pset.constants[idx]))


def _sample_operator(
    ops: tuple[OperatorPrimitive, ...],
    rng: np.random.Generator,
) -> OperatorPrimitive:
    idx = int(rng.integers(0, len(ops)))
    return ops[idx]


def generate_tree(
    pset: PrimitiveSet,
    rng: np.random.Generator,
    target_type: DataType = DataType.Series,
    min_depth: int = 2,
    max_depth: int = 5,
    method: str = "grow",
    term_prob: float = 0.35,
) -> ASTNode:
    """递归生成一棵满足强类型签名与深度约束的 `ASTNode` 子树。

    Parameters
    ----------
    pset:
        强类型原语集
    rng:
        独立的 NumPy 随机数生成器 (保证 100% 确定性可复现)
    target_type:
        期望生成的子树根节点返回类型
    min_depth:
        最小期望深度（叶子节点深度为 1）
    max_depth:
        最大允许深度（叶子节点深度为 1）
    method:
        'grow' 或 'full'
    term_prob:
        在 'grow' 模式下已满足最小深度时选择终端叶子节点的概率
    """
    if method not in ("grow", "full"):
        raise ValueError(f"生成模式必须为 'grow' 或 'full', 实际为 {method!r}")

    # 1. Window 与 Constant 永远是深度为 1 的终端叶子
    if target_type == DataType.Window:
        return _sample_window(pset, rng)
    if target_type == DataType.Constant:
        return _sample_constant(pset, rng)

    # 2. BoolCond 类型：无深度为 1 的叶子，最小深度为 2 (由比较算子生成)
    if target_type == DataType.BoolCond:
        eff_max = max(2, max_depth)
        if eff_max == 2 or (method == "grow" and min_depth <= 2 and rng.random() < 0.6):
            valid_ops = pset.get_operators(DataType.BoolCond, max_depth=2)
        else:
            valid_ops = pset.get_operators(DataType.BoolCond, max_depth=eff_max)
            if not valid_ops:
                valid_ops = pset.get_operators(DataType.BoolCond, max_depth=2)

        op = _sample_operator(valid_ops, rng)
        return _expand_operator(op, pset, rng, min_depth, eff_max, method, term_prob)

    # 3. Series 类型
    if max_depth <= 1:
        return _sample_field(pset, rng)

    if method == "grow" and min_depth <= 1 and rng.random() < term_prob:
        return _sample_field(pset, rng)

    valid_ops = pset.get_operators(DataType.Series, max_depth=max_depth)
    if not valid_ops:
        return _sample_field(pset, rng)

    op = _sample_operator(valid_ops, rng)
    return _expand_operator(op, pset, rng, min_depth, max_depth, method, term_prob)


def _expand_operator(
    op: OperatorPrimitive,
    pset: PrimitiveSet,
    rng: np.random.Generator,
    min_depth: int,
    max_depth: int,
    method: str,
    term_prob: float,
) -> OpNode:
    """根据算子签名递归生成所有强类型子节点，并规避左右同质的退化参数。"""
    children: list[ASTNode] = []
    first_series_expr: str | None = None

    for arg_t in op.arg_types:
        child_min = max(arg_t.min_tree_depth, min_depth - 1)
        child_max = max(arg_t.min_tree_depth, max_depth - 1)
        child = generate_tree(
            pset=pset,
            rng=rng,
            target_type=arg_t,
            min_depth=child_min,
            max_depth=child_max,
            method=method,
            term_prob=term_prob,
        )

        # 若二元/三元算子的两个 Series 参数完全相同，尝试重新采样避免 sub(x,x)/div(x,x)/ts_corr(x,x,w)
        if arg_t == DataType.Series:
            c_expr = child.to_expr()
            if first_series_expr is None:
                first_series_expr = c_expr
            elif c_expr == first_series_expr and len(pset.fields) > 1:
                if isinstance(child, FieldNode):
                    child = _sample_field(pset, rng, avoid=first_series_expr)
                else:
                    alt = generate_tree(
                        pset=pset,
                        rng=rng,
                        target_type=arg_t,
                        min_depth=child_min,
                        max_depth=child_max,
                        method=method,
                        term_prob=term_prob,
                    )
                    if alt.to_expr() != first_series_expr:
                        child = alt
                    else:
                        child = _sample_field(pset, rng, avoid=first_series_expr)

        children.append(child)

    return OpNode(op=op, args=tuple(children))


def generate_grow(
    pset: PrimitiveSet,
    rng: np.random.Generator,
    min_depth: int = 2,
    max_depth: int = 5,
    target_type: DataType = DataType.Series,
) -> ASTNode:
    """使用 Grow 模式生成非对称强类型语法树。"""
    return generate_tree(
        pset=pset,
        rng=rng,
        target_type=target_type,
        min_depth=min_depth,
        max_depth=max_depth,
        method="grow",
    )


def generate_full(
    pset: PrimitiveSet,
    rng: np.random.Generator,
    depth: int = 3,
    target_type: DataType = DataType.Series,
) -> ASTNode:
    """使用 Full 模式生成指定深度的饱满强类型语法树。"""
    return generate_tree(
        pset=pset,
        rng=rng,
        target_type=target_type,
        min_depth=depth,
        max_depth=depth,
        method="full",
    )


def generate_ramped_half_and_half(
    pset: PrimitiveSet,
    pop_size: int,
    rng: np.random.Generator,
    min_depth: int = 2,
    max_depth: int = 5,
    max_nodes: int = 30,
    deduplicate: bool = True,
    max_retries_per_ind: int = 8,
) -> list[Individual]:
    """采用 Ramped Half-and-Half 策略生成初始种群（50% Grow + 50% Full，跨深度均匀分布）。"""
    if pop_size <= 0:
        return []

    depths = list(range(max(2, min_depth), max(2, max_depth) + 1))
    n_depths = len(depths)

    population: list[Individual] = []
    seen_hashes: set[str] = set()

    for i in range(pop_size):
        target_d = depths[i % n_depths]
        # 50% Full, 50% Grow
        method = "full" if (i // n_depths) % 2 == 0 else "grow"

        chosen_ind: Individual | None = None
        for attempt in range(max_retries_per_ind):
            cur_method = method if attempt < max_retries_per_ind // 2 else "grow"
            node = generate_tree(
                pset=pset,
                rng=rng,
                target_type=DataType.Series,
                min_depth=min_depth if cur_method == "grow" else target_d,
                max_depth=target_d,
                method=cur_method,
            )
            ind = canonicalize_node(node, pset=pset, sync_tree=True)
            if ind.is_degenerate:
                continue
            if ind.depth > max_depth or ind.n_nodes > max_nodes:
                continue
            if deduplicate and ind.canon_hash in seen_hashes:
                chosen_ind = ind  # 保留作为超出重试上限时的合法回退
                continue
            chosen_ind = ind
            break

        if chosen_ind is None:
            # 保底生成一个简单合法的 2 层算子个体
            f_node = _sample_field(pset, rng)
            w_node = _sample_window(pset, rng)
            ts_op = pset.ops_by_name.get("ts_mean") or pset.get_operators(DataType.Series, 2)[0]
            if ts_op.arg_types == (DataType.Series, DataType.Window):
                fallback_root: ASTNode = OpNode(op=ts_op, args=(f_node, w_node))
            else:
                fallback_root = f_node
            chosen_ind = canonicalize_node(fallback_root, pset=pset, sync_tree=True)

        seen_hashes.add(chosen_ind.canon_hash)
        population.append(chosen_ind)

    return population
