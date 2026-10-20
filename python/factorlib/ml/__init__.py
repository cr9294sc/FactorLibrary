"""FactorLibrary 机器学习多因子合成模块 (factorlib.ml)。

架构组件：
- splits: Purged & Embargoed 扩展/滚动窗口切分器
- dataset: 截面特征工程预处理标准流 (方向对齐、MAD Winsorization、Rank Gauss) 与平铺数据集构建
- models: Equal Weight、Rolling ICIR、Ridge 与 LightGBM 预测模型
- pipeline: Walk-Forward 自动滚动训练评估管线
"""

from __future__ import annotations

from factorlib.ml.dataset import (
    MLDatasetBuilder,
    TabularDataset,
    align_directions,
    mad_winsorize,
    preprocess_factors,
    rank_gauss_transform,
)
from factorlib.ml.models import (
    EqualWeightModel,
    LightGBMModel,
    RidgeModel,
    RollingICIRModel,
    create_model,
)
from factorlib.ml.pipeline import WalkForwardPipeline, WalkForwardResult
from factorlib.ml.splits import PurgedWalkForwardSplitter, WalkForwardFold

__all__ = [
    "EqualWeightModel",
    "LightGBMModel",
    "MLDatasetBuilder",
    "PurgedWalkForwardSplitter",
    "RidgeModel",
    "RollingICIRModel",
    "TabularDataset",
    "WalkForwardFold",
    "WalkForwardPipeline",
    "WalkForwardResult",
    "align_directions",
    "create_model",
    "mad_winsorize",
    "preprocess_factors",
    "rank_gauss_transform",
]
