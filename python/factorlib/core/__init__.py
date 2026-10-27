"""核心层: 配置 / 面板 / DSL / 催化目录 / 探针后处理 / 评价 / 组合 / 因子库 (市场无关)。"""

from factorlib.core.catalog import (
    POSTPROCESS_SPECS,
    RECOMMENDED_POSTPROCESS_PRESETS,
    FactorCatalog,
)
from factorlib.core.evaluate import (
    DualLayerWorkflowResult,
    FactorEvaluationResult,
    evaluate_factor,
    evaluate_factor_v4_stats,
    get_seed_factors,
    register_seed_factors,
    run_factor_workflow,
)
from factorlib.core.panel import Panel
from factorlib.core.probe import (
    FactorDefinition,
    PostprocessStep,
    PythonFactorContext,
    apply_postprocess_pipeline,
    evaluate_factor_definition,
    parse_postprocess_pipeline,
    probe_factor,
)
from factorlib.core.registry import (
    ExperimentRunRecord,
    FactorMetricsRecord,
    FactorRecord,
    FactorRegistry,
    FactorVersionRecord,
)
from factorlib.core.styles import (
    STYLE_DEFINITIONS,
    STYLE_NAMES,
    compute_futures_styles,
    compute_style_exposures,
)
from factorlib.core.workbench import (
    AUTHORING_TEMPLATES,
    FactorWorkbenchService,
    create_synthetic_futures_panel,
)

__all__ = [
    "Panel",
    "FactorCatalog",
    "POSTPROCESS_SPECS",
    "RECOMMENDED_POSTPROCESS_PRESETS",
    "FactorDefinition",
    "PostprocessStep",
    "PythonFactorContext",
    "parse_postprocess_pipeline",
    "probe_factor",
    "apply_postprocess_pipeline",
    "evaluate_factor_definition",
    "STYLE_NAMES",
    "STYLE_DEFINITIONS",
    "compute_futures_styles",
    "compute_style_exposures",
    "FactorRegistry",
    "FactorRecord",
    "FactorMetricsRecord",
    "FactorVersionRecord",
    "ExperimentRunRecord",
    "FactorEvaluationResult",
    "DualLayerWorkflowResult",
    "evaluate_factor",
    "evaluate_factor_v4_stats",
    "run_factor_workflow",
    "get_seed_factors",
    "register_seed_factors",
    "FactorWorkbenchService",
    "create_synthetic_futures_panel",
    "AUTHORING_TEMPLATES",
]
