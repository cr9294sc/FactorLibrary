"""强类型遗传规划 (STGP) 自动化因子挖掘系统 (factorlib.gp)。"""

from __future__ import annotations

from factorlib.gp.admission import AdmissionPipeline, AdmissionSummary, AdmittedFactor
from factorlib.gp.canon import (
    canonicalize_individual,
    canonicalize_node,
    canonicalize_population,
    deduplicate_individuals,
)
from factorlib.gp.engine import GenerationStats, GPEngine
from factorlib.gp.generate import (
    generate_full,
    generate_grow,
    generate_ramped_half_and_half,
    generate_tree,
)
from factorlib.gp.islands import Island, RingMigrationController
from factorlib.gp.primitives import (
    DEFAULT_CONSTANTS,
    DEFAULT_WINDOWS,
    DataType,
    OperatorPrimitive,
    PrimitiveSet,
)
from factorlib.gp.select import HallOfFame, select_elites, tournament_select
from factorlib.gp.tree import (
    ASTNode,
    ConstantNode,
    FieldNode,
    Individual,
    OpNode,
    WindowNode,
    parse_dsl_to_tree,
)
from factorlib.gp.variation import (
    VariationPipeline,
    crossover,
    hoist_mutation,
    param_tuning,
    point_mutation,
    subtree_mutation,
)

__all__ = [
    "ASTNode",
    "AdmissionPipeline",
    "AdmissionSummary",
    "AdmittedFactor",
    "ConstantNode",
    "DEFAULT_CONSTANTS",
    "DEFAULT_WINDOWS",
    "DataType",
    "FieldNode",
    "GPEngine",
    "GenerationStats",
    "HallOfFame",
    "Individual",
    "Island",
    "OpNode",
    "OperatorPrimitive",
    "PrimitiveSet",
    "RingMigrationController",
    "VariationPipeline",
    "WindowNode",
    "canonicalize_individual",
    "canonicalize_node",
    "canonicalize_population",
    "crossover",
    "deduplicate_individuals",
    "generate_full",
    "generate_grow",
    "generate_ramped_half_and_half",
    "generate_tree",
    "hoist_mutation",
    "param_tuning",
    "parse_dsl_to_tree",
    "point_mutation",
    "select_elites",
    "subtree_mutation",
    "tournament_select",
]
