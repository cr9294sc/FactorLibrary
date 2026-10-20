"""核心层: 配置 / 面板 / DSL / 评价 / 组合 / 因子库 (市场无关)。"""

from factorlib.core.evaluate import (
    FactorEvaluationResult,
    evaluate_factor,
    get_seed_factors,
    register_seed_factors,
)
from factorlib.core.panel import Panel
from factorlib.core.registry import (
    ExperimentRunRecord,
    FactorMetricsRecord,
    FactorRecord,
    FactorRegistry,
)

__all__ = [
    "Panel",
    "FactorRegistry",
    "FactorRecord",
    "FactorMetricsRecord",
    "ExperimentRunRecord",
    "FactorEvaluationResult",
    "evaluate_factor",
    "get_seed_factors",
    "register_seed_factors",
]
