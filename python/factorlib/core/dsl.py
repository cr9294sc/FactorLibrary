"""Python 侧 DSL 表达式构建器、语法糖与静态检查封装 (core/dsl.py)。

按照白皮书 3.1 节与第 7 节设计：
- 提供面向对象与函数式的因子表达式构建语法糖（`Expr` / `F`）；
- 对接 `factorlib_core` 的静态类型检查（`check_expr`）、代数等价规范化（`parse_and_canonicalize`）
  与算子目录查询（`list_operators`）。
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import factorlib_core


@dataclass(frozen=True)
class CanonInfo:
    """规范化后的表达式元数据对象。"""

    canon_expr: str
    canon_hash: str
    hash_u64: int
    is_degenerate: bool
    degenerate_reason: str | None
    n_nodes: int
    depth: int
    return_type: str


class Expr:
    """DSL 表达式链式构建语法糖包装器。

    示例
    ----
    >>> close = Expr("close")
    >>> open_ = Expr("open")
    >>> alpha = (close - open_).ts_mean(20).cs_rank()
    >>> str(alpha)
    'cs_rank(ts_mean(sub(close, open), 20))'
    """

    def __init__(self, expr_str: str) -> None:
        self._expr = expr_str.strip()

    def __str__(self) -> str:
        return self._expr

    def __repr__(self) -> str:
        return f"Expr({self._expr!r})"

    # 算术运算符重载
    def __neg__(self) -> Expr:
        return Expr(f"neg({self._expr})")

    def __abs__(self) -> Expr:
        return Expr(f"abs({self._expr})")

    def __add__(self, other: Expr | float | int) -> Expr:
        if isinstance(other, (int, float)):
            return Expr(f"addc({self._expr}, {float(other)})")
        return Expr(f"add({self._expr}, {other})")

    def __sub__(self, other: Expr | float | int) -> Expr:
        if isinstance(other, (int, float)):
            return Expr(f"subc({self._expr}, {float(other)})")
        return Expr(f"sub({self._expr}, {other})")

    def __mul__(self, other: Expr | float | int) -> Expr:
        if isinstance(other, (int, float)):
            return Expr(f"mulc({self._expr}, {float(other)})")
        return Expr(f"mul({self._expr}, {other})")

    def __truediv__(self, other: Expr | float | int) -> Expr:
        if isinstance(other, (int, float)):
            return Expr(f"divc({self._expr}, {float(other)})")
        return Expr(f"div({self._expr}, {other})")

    def __gt__(self, other: Expr | float | int) -> Expr:
        if isinstance(other, (int, float)):
            return Expr(f"gt_const({self._expr}, {float(other)})")
        return Expr(f"gt({self._expr}, {other})")

    def __lt__(self, other: Expr | float | int) -> Expr:
        if isinstance(other, (int, float)):
            return Expr(f"lt_const({self._expr}, {float(other)})")
        return Expr(f"lt({self._expr}, {other})")

    # 常用链式算子方法
    def delay(self, d: int) -> Expr:
        return Expr(f"delay({self._expr}, {int(d)})")

    def delta(self, d: int) -> Expr:
        return Expr(f"delta({self._expr}, {int(d)})")

    def pct_change(self, d: int) -> Expr:
        return Expr(f"pct_change({self._expr}, {int(d)})")

    def ts_mean(self, d: int) -> Expr:
        return Expr(f"ts_mean({self._expr}, {int(d)})")

    def ts_std(self, d: int) -> Expr:
        return Expr(f"ts_std({self._expr}, {int(d)})")

    def ts_min(self, d: int) -> Expr:
        return Expr(f"ts_min({self._expr}, {int(d)})")

    def ts_max(self, d: int) -> Expr:
        return Expr(f"ts_max({self._expr}, {int(d)})")

    def ts_rank(self, d: int) -> Expr:
        return Expr(f"ts_rank({self._expr}, {int(d)})")

    def ts_zscore(self, d: int) -> Expr:
        return Expr(f"ts_zscore({self._expr}, {int(d)})")

    def cs_rank(self) -> Expr:
        return Expr(f"cs_rank({self._expr})")

    def cs_zscore(self) -> Expr:
        return Expr(f"cs_zscore({self._expr})")

    def cs_demean(self) -> Expr:
        return Expr(f"cs_demean({self._expr})")

    def cs_winsorize(self) -> Expr:
        return Expr(f"cs_winsorize({self._expr})")

    def canonicalize(self, available_fields: Sequence[str] | None = None) -> CanonInfo:
        return canonicalize_expr(self._expr, available_fields=available_fields)


def F(field_name: str) -> Expr:
    """快捷构造终端特征字段 `Expr`。"""
    return Expr(field_name)


def canonicalize_expr(
    expr: str | Expr,
    available_fields: Sequence[str] | None = None,
) -> CanonInfo:
    """调用 Rust 内核解析并代数规范化表达式。"""
    fields_list = list(available_fields) if available_fields is not None else None
    raw: dict[str, Any] = factorlib_core.parse_and_canonicalize(str(expr), fields_list)
    return CanonInfo(
        canon_expr=raw["canon_expr"],
        canon_hash=raw["canon_hash"],
        hash_u64=raw["hash_u64"],
        is_degenerate=raw["is_degenerate"],
        degenerate_reason=raw["degenerate_reason"],
        n_nodes=raw["n_nodes"],
        depth=raw["depth"],
        return_type=raw["return_type"],
    )


def validate_expr(
    expr: str | Expr,
    available_fields: Sequence[str] | None = None,
    require_series_root: bool = True,
) -> dict[str, Any]:
    """静态校验表达式语法、强类型签名及防未来函数约束。"""
    fields_list = list(available_fields) if available_fields is not None else None
    return factorlib_core.check_expr(str(expr), fields_list, require_series_root)


def get_operator_specs() -> list[dict[str, Any]]:
    """获取 Rust 内核注册的全部算子规格列表。"""
    return factorlib_core.list_operators()
