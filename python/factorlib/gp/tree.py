"""STGP 语法树节点表示、解析转换与个体封装 (gp/tree.py)。

严格遵循白皮书 7.1 节与 9.3 节类图规范：
- 定义强类型不可变 AST 节点：`FieldNode`, `ConstantNode`, `WindowNode`, `OpNode`；
- 支持路径寻址遍历（`iter_nodes_with_path`）与函数式子树替换（`replace_at_path`）；
- 支持从规范化 DSL 文本反向解析构造强类型 `ASTNode`（`parse_dsl_to_tree`）；
- 定义 `Individual` 个体对象，完整承载 `root`, `canon_expr`, `canon_hash`, `raw_fitness`,
  `adjusted_fitness`, `n_nodes`, `depth` 及训练集统计指标。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from factorlib.gp.primitives import DataType, OperatorPrimitive, PrimitiveSet


def format_float_constant(val: float) -> str:
    """将浮点常数格式化为与 Rust 内核一致的规范文本。"""
    v = 0.0 if val == 0.0 else float(val)
    if v.is_integer() and abs(v) < 1e15:
        return f"{v:.1f}"
    rounded = round(v, 10)
    s = f"{rounded}"
    if "." not in s and "e" not in s and "E" not in s:
        s = f"{s}.0"
    return s


class ASTNode:
    """STGP 抽象语法树节点基类。"""

    @property
    def return_type(self) -> DataType:
        raise NotImplementedError

    @property
    def n_nodes(self) -> int:
        raise NotImplementedError

    @property
    def depth(self) -> int:
        raise NotImplementedError

    def to_expr(self) -> str:
        raise NotImplementedError

    def clone(self) -> ASTNode:
        """由于节点采用不可变设计，返回自身引用即等价于深拷贝。"""
        return self

    def iter_nodes_with_path(
        self,
        prefix: tuple[int, ...] = (),
    ) -> list[tuple[tuple[int, ...], ASTNode]]:
        """前序遍历收集所有 `(路径元组, 节点)`。根节点路径为 `()`。"""
        out: list[tuple[tuple[int, ...], ASTNode]] = [(prefix, self)]
        if isinstance(self, OpNode):
            for idx, child in enumerate(self.args):
                out.extend(child.iter_nodes_with_path(prefix + (idx,)))
        return out

    def replace_at_path(self, path: tuple[int, ...], new_node: ASTNode) -> ASTNode:
        """返回将指定 `path` 处子树替换为 `new_node` 后的新语法树。"""
        if not path:
            return new_node
        if not isinstance(self, OpNode):
            raise IndexError(f"叶子节点无法继续沿路径 {path} 替换")
        head, *tail = path
        if head < 0 or head >= len(self.args):
            raise IndexError(f"子节点索引 {head} 越界 (arity={len(self.args)})")
        new_args = list(self.args)
        new_args[head] = self.args[head].replace_at_path(tuple(tail), new_node)
        return OpNode(op=self.op, args=tuple(new_args))

    def __str__(self) -> str:
        return self.to_expr()


@dataclass(frozen=True)
class FieldNode(ASTNode):
    """终端特征字段节点 (`DataType.Series`)。"""

    name: str

    @property
    def return_type(self) -> DataType:
        return DataType.Series

    @property
    def n_nodes(self) -> int:
        return 1

    @property
    def depth(self) -> int:
        return 1

    def to_expr(self) -> str:
        return self.name


@dataclass(frozen=True)
class ConstantNode(ASTNode):
    """浮点常数标量节点 (`DataType.Constant`)。"""

    value: float

    @property
    def return_type(self) -> DataType:
        return DataType.Constant

    @property
    def n_nodes(self) -> int:
        return 1

    @property
    def depth(self) -> int:
        return 1

    def to_expr(self) -> str:
        return format_float_constant(self.value)


@dataclass(frozen=True)
class WindowNode(ASTNode):
    """离散滑动窗口整数节点 (`DataType.Window`)。"""

    value: int

    @property
    def return_type(self) -> DataType:
        return DataType.Window

    @property
    def n_nodes(self) -> int:
        return 1

    @property
    def depth(self) -> int:
        return 1

    def to_expr(self) -> str:
        return str(int(self.value))


@dataclass(frozen=True)
class OpNode(ASTNode):
    """非终端算子调用节点。"""

    op: OperatorPrimitive
    args: tuple[ASTNode, ...]

    def __post_init__(self) -> None:
        if len(self.args) != self.op.arity:
            raise ValueError(
                f"算子 '{self.op.name}' 期望 {self.op.arity} 个参数, 实际传入 {len(self.args)} 个"
            )
        for idx, (child, exp_t) in enumerate(zip(self.args, self.op.arg_types, strict=True)):
            if child.return_type != exp_t:
                raise TypeError(
                    f"算子 '{self.op.name}' 第 {idx} 个参数类型不匹配: "
                    f"期望 {exp_t.value}, 实际为 {child.return_type.value}"
                )

    @property
    def return_type(self) -> DataType:
        return self.op.return_type

    @property
    def n_nodes(self) -> int:
        return 1 + sum(c.n_nodes for c in self.args)

    @property
    def depth(self) -> int:
        return 1 + max(c.depth for c in self.args)

    def to_expr(self) -> str:
        inner = ", ".join(c.to_expr() for c in self.args)
        return f"{self.op.name}({inner})"


def _split_top_level_args(arg_str: str) -> list[str]:
    """按顶层逗号切分函数参数字符串。"""
    parts: list[str] = []
    depth = 0
    start = 0
    for idx, ch in enumerate(arg_str):
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
        elif ch == "," and depth == 0:
            parts.append(arg_str[start:idx].strip())
            start = idx + 1
    tail = arg_str[start:].strip()
    if tail:
        parts.append(tail)
    return parts


def parse_dsl_to_tree(
    expr: str,
    pset: PrimitiveSet,
    expected_type: DataType = DataType.Series,
) -> ASTNode:
    """将前缀函数式 DSL 表达式解析为强类型 `ASTNode` 语法树。"""
    s = expr.strip()
    if not s:
        raise ValueError("空表达式无法解析为 ASTNode")

    # 检查是否为函数调用 op(arg1, arg2, ...)
    if "(" in s and s.endswith(")"):
        paren_idx = s.index("(")
        op_name = s[:paren_idx].strip()
        inner = s[paren_idx + 1 : -1].strip()
        op = pset.ops_by_name.get(op_name)
        if op is None:
            # 回退从全量内核算子表查找（例如包含 group_binary 时）
            from factorlib.gp.primitives import load_core_operators

            all_ops = {o.name: o for o in load_core_operators(include_group_ops=True)}
            op = all_ops.get(op_name)
            if op is None:
                raise ValueError(f"未知的算子名称: {op_name!r}")

        raw_args = _split_top_level_args(inner)
        if len(raw_args) != op.arity:
            raise ValueError(f"算子 {op_name!r} 参数个数不匹配: {raw_args}")
        parsed_args = tuple(
            parse_dsl_to_tree(arg_s, pset, expected_type=arg_t)
            for arg_s, arg_t in zip(raw_args, op.arg_types, strict=True)
        )
        return OpNode(op=op, args=parsed_args)

    # 叶子节点：根据期望类型解析为 WindowNode / ConstantNode / FieldNode
    if expected_type == DataType.Window:
        return WindowNode(int(float(s)))
    if expected_type == DataType.Constant:
        return ConstantNode(float(s))

    # 若首字符为数字或正负号，判断为常数或窗口
    if s[0].isdigit() or (s[0] in "+-" and len(s) > 1 and (s[1].isdigit() or s[1] == ".")):
        if "." in s or "e" in s or "E" in s or s.startswith("-"):
            return ConstantNode(float(s))
        return WindowNode(int(s))

    return FieldNode(s)


@dataclass
class Individual:
    """遗传规划种群中的单个候选因子个体。"""

    root: ASTNode
    canon_expr: str = ""
    canon_hash: str = ""
    hash_u64: int = 0
    raw_fitness: float = float("-inf")
    adjusted_fitness: float = float("-inf")
    n_nodes: int = 0
    depth: int = 0
    ic_mean: float = 0.0
    ic_std: float = 0.0
    ic_ir: float = 0.0
    coverage: float = 0.0
    max_lib_corr: float = 0.0
    mean_lib_corr: float = 0.0
    is_degenerate: bool = False
    degenerate_reason: str | None = None
    evaluated: bool = False

    def __post_init__(self) -> None:
        if not self.canon_expr:
            self.canon_expr = self.root.to_expr()
        if self.n_nodes <= 0:
            self.n_nodes = self.root.n_nodes
        if self.depth <= 0:
            self.depth = self.root.depth

    @property
    def expr(self) -> str:
        """返回个体的规范化表达式字符串。"""
        return self.canon_expr or self.root.to_expr()

    @property
    def sign(self) -> int:
        """根据训练集 IC 均值确定推荐因子方向 (1 或 -1)。"""
        return 1 if self.ic_mean >= 0.0 else -1

    def clone(self) -> Individual:
        """深拷贝当前个体。"""
        return Individual(
            root=self.root.clone(),
            canon_expr=self.canon_expr,
            canon_hash=self.canon_hash,
            hash_u64=self.hash_u64,
            raw_fitness=self.raw_fitness,
            adjusted_fitness=self.adjusted_fitness,
            n_nodes=self.n_nodes,
            depth=self.depth,
            ic_mean=self.ic_mean,
            ic_std=self.ic_std,
            ic_ir=self.ic_ir,
            coverage=self.coverage,
            max_lib_corr=self.max_lib_corr,
            mean_lib_corr=self.mean_lib_corr,
            is_degenerate=self.is_degenerate,
            degenerate_reason=self.degenerate_reason,
            evaluated=self.evaluated,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "canon_expr": self.canon_expr,
            "canon_hash": self.canon_hash,
            "raw_fitness": self.raw_fitness,
            "adjusted_fitness": self.adjusted_fitness,
            "n_nodes": self.n_nodes,
            "depth": self.depth,
            "ic_mean": self.ic_mean,
            "ic_std": self.ic_std,
            "ic_ir": self.ic_ir,
            "coverage": self.coverage,
            "max_lib_corr": self.max_lib_corr,
            "sign": self.sign,
            "is_degenerate": self.is_degenerate,
        }
