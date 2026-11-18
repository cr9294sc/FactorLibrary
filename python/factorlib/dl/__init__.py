"""FactorLibrary `1d` 深度学习因子研究子系统 (`factorlib.dl`)。"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from factorlib.dl.trainer import DLTrainConfig, DLTrainResult, train_dl_walk_forward

__all__ = [
    "DLTrainConfig",
    "DLTrainResult",
    "train_dl_walk_forward",
]


def __getattr__(name: str) -> Any:
    if name in __all__:
        from factorlib.dl import trainer

        return getattr(trainer, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
