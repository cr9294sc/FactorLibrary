"""强类型遗传规划 (STGP) 原语集与类型系统定义 (gp/primitives.py)。

严格遵循白皮书 7.1、7.2 与 9.2 节规范：
- 定义四大强类型枚举：`Series` (连续面板矩阵), `Window` (离散滑动窗口), `Constant` (浮点常数), `BoolCond` (条件布尔掩码)；
- 默认窗口参数池：`{3, 5, 10, 20, 40, 60, 120, 250}`；
- 默认常数参数池：`{-2.0, -1.0, -0.5, 0.5, 1.0, 2.0}`；
- 直接与 Rust 内核 `factorlib_core.list_operators()` 算子规格表同步，按返回类型、参数签名与最小合法子树深度建立索引。
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

import factorlib_core

from factorlib.core.panel import Panel


class DataType(StrEnum):
    """STGP 语法树节点静态数据类型枚举。"""

    Series = "Series"
    Window = "Window"
    Constant = "Constant"
    BoolCond = "BoolCond"

    @property
    def min_tree_depth(self) -> int:
        """生成该类型合法子树所需的最小深度（叶子节点深度为 1）。

        `BoolCond` 没有直接终端叶子，必须由比较算子（如 `gt(Series, Series)`）生成，因此最小深度为 2。
        """
        if self is DataType.BoolCond:
            return 2
        return 1


# 白皮书 9.2 节规定的全局离散参数池
DEFAULT_WINDOWS: tuple[int, ...] = (3, 5, 10, 20, 40, 60, 120, 250)
DEFAULT_CONSTANTS: tuple[float, ...] = (-2.0, -1.0, -0.5, 0.5, 1.0, 2.0)

# 面板中非 DSL 终端的辅助列（回测/执行/原始未复权价等），不作为 GP 终端特征暴露
AUX_NON_DSL_FIELDS: frozenset[str] = frozenset(
    {
        "trading_date",
        "symbol",
        "contract",
        "prev_contract",
        "is_roll",
        "date_idx",
        "sector",
        "open_raw",
        "high_raw",
        "low_raw",
        "close_raw",
        "settle_raw",
        "prev_settle_raw",
        "limit_up",
        "limit_down",
        "is_unilateral",
        "adj_factor",
        "exec_vwap",
        "exec_vwap_old",
        "exec_volume",
        "multiplier",
        "margin_rate",
        "days_to_expiry",
        "traded",
        "adv20_turnover",
        "listed_days",
        "in_universe",
    }
)


@dataclass(frozen=True)
class OperatorPrimitive:
    """强类型算子原语规格。"""

    name: str
    category: str
    arg_types: tuple[DataType, ...]
    return_type: DataType
    commutative: bool = False

    @property
    def arity(self) -> int:
        return len(self.arg_types)

    @property
    def signature_key(self) -> tuple[tuple[DataType, ...], DataType]:
        return (self.arg_types, self.return_type)

    @property
    def min_depth(self) -> int:
        """以该算子为根节点的合法子树所需的最小深度。"""
        if not self.arg_types:
            return 1
        return 1 + max(t.min_tree_depth for t in self.arg_types)


def load_core_operators(include_group_ops: bool = False) -> tuple[OperatorPrimitive, ...]:
    """从 Rust 内核加载算子规格列表并转换为 `OperatorPrimitive`。"""
    raw_specs: list[dict[str, Any]] = factorlib_core.list_operators()
    ops: list[OperatorPrimitive] = []
    for sp in raw_specs:
        cat = str(sp["category"])
        if not include_group_ops and cat == "group_binary":
            continue
        arg_types = tuple(DataType(str(t)) for t in sp["arg_types"])
        ret_type = DataType(str(sp["return_type"]))
        ops.append(
            OperatorPrimitive(
                name=str(sp["name"]),
                category=cat,
                arg_types=arg_types,
                return_type=ret_type,
                commutative=bool(sp["commutative"]),
            )
        )
    return tuple(ops)


class PrimitiveSet:
    """STGP 强类型原语集合容器。"""

    def __init__(
        self,
        fields: Sequence[str],
        windows: Sequence[int] = DEFAULT_WINDOWS,
        constants: Sequence[float] = DEFAULT_CONSTANTS,
        operators: Sequence[OperatorPrimitive] | None = None,
        include_group_ops: bool = False,
    ) -> None:
        cleaned_fields = tuple(sorted({str(f) for f in fields if str(f).strip()}))
        if not cleaned_fields:
            raise ValueError("PrimitiveSet 至少需要包含 1 个终端特征字段 (fields)")

        cleaned_windows = tuple(sorted({int(w) for w in windows if int(w) >= 1}))
        if not cleaned_windows:
            raise ValueError("PrimitiveSet 至少需要包含 1 个正整数窗口参数 (windows)")

        cleaned_constants = tuple(float(c) for c in constants)
        if not cleaned_constants:
            raise ValueError("PrimitiveSet 至少需要包含 1 个常数参数 (constants)")

        self.fields: tuple[str, ...] = cleaned_fields
        self.windows: tuple[int, ...] = cleaned_windows
        self.constants: tuple[float, ...] = cleaned_constants
        self.operators: tuple[OperatorPrimitive, ...] = (
            tuple(operators)
            if operators is not None
            else load_core_operators(include_group_ops=include_group_ops)
        )

        # 构建快速检索索引
        self.ops_by_name: dict[str, OperatorPrimitive] = {op.name: op for op in self.operators}

        by_ret: dict[DataType, list[OperatorPrimitive]] = {t: [] for t in DataType}
        by_sig: dict[tuple[tuple[DataType, ...], DataType], list[OperatorPrimitive]] = {}
        for op in self.operators:
            by_ret[op.return_type].append(op)
            by_sig.setdefault(op.signature_key, []).append(op)

        self.ops_by_return_type: dict[DataType, tuple[OperatorPrimitive, ...]] = {
            k: tuple(v) for k, v in by_ret.items()
        }
        self.ops_by_signature: dict[
            tuple[tuple[DataType, ...], DataType], tuple[OperatorPrimitive, ...]
        ] = {k: tuple(v) for k, v in by_sig.items()}

    @classmethod
    def from_panel(
        cls,
        panel: Panel,
        windows: Sequence[int] | None = None,
        constants: Sequence[float] | None = None,
        include_group_ops: bool = False,
        max_window: int | None = None,
    ) -> PrimitiveSet:
        """从 `Panel` 实例自动提取合法 DSL 特征字段构建 `PrimitiveSet`。"""
        all_fields = list(panel.fields.keys())
        dsl_fields = [f for f in all_fields if f not in AUX_NON_DSL_FIELDS]
        if not dsl_fields:
            dsl_fields = all_fields

        win_pool = list(windows) if windows is not None else list(DEFAULT_WINDOWS)
        if max_window is not None:
            filtered = [w for w in win_pool if w <= max_window]
            win_pool = filtered if filtered else [min(win_pool)]

        const_pool = constants if constants is not None else DEFAULT_CONSTANTS
        return cls(
            fields=dsl_fields,
            windows=win_pool,
            constants=const_pool,
            include_group_ops=include_group_ops,
        )

    def get_operators(
        self,
        return_type: DataType,
        max_depth: int | None = None,
    ) -> tuple[OperatorPrimitive, ...]:
        """获取指定返回类型且满足最大允许深度约束的算子集合。"""
        candidates = self.ops_by_return_type.get(return_type, ())
        if max_depth is None:
            return candidates
        return tuple(op for op in candidates if op.min_depth <= max_depth)

    def get_compatible_operators(
        self,
        op: OperatorPrimitive,
        exclude_self: bool = True,
    ) -> tuple[OperatorPrimitive, ...]:
        """获取与给定算子具有完全相同输入输出类型签名的同级算子集合（用于点变异）。"""
        same_sig = self.ops_by_signature.get(op.signature_key, ())
        if not exclude_self:
            return same_sig
        others = tuple(cand for cand in same_sig if cand.name != op.name)
        return others if others else same_sig
