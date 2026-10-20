"""多因子合成预测模型族 (ml/models)。

包含基准等权、时序/样本内 ICIR 加权、Ridge 回归与 LightGBM 梯度提升决策树。
"""

from __future__ import annotations

from typing import Any

from factorlib.ml.models.lgbm import LightGBMModel
from factorlib.ml.models.linear import (
    EqualWeightModel,
    RidgeModel,
    RollingICIRModel,
)

__all__ = [
    "EqualWeightModel",
    "LightGBMModel",
    "RidgeModel",
    "RollingICIRModel",
    "create_model",
]


def create_model(model_name: str, **kwargs: Any) -> Any:
    """根据模型标识符工厂化实例化模型。"""
    normalized = model_name.strip().lower()
    if normalized in ("equal_weight", "eq_weight", "ew"):
        return EqualWeightModel()
    if normalized in ("rolling_icir", "icir"):
        return RollingICIRModel(**kwargs)
    if normalized in ("ridge", "ridge_regression"):
        return RidgeModel(**kwargs)
    if normalized in ("lightgbm", "lgb", "lgbm"):
        return LightGBMModel(**kwargs)
    raise ValueError(f"未知的模型类型: {model_name!r}")
