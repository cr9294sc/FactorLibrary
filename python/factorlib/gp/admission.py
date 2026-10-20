"""STGP 因子出海入库与五步正交化去相关准入决策器 (gp/admission.py)。

严格遵循白皮书 9.5 节五步准入过滤器规范：
1. Top-K 初筛：合并所有岛屿的名人堂（Hall of Fame），按训练集适应度降序取前 200 个不重复个体；
2. 验证集符号与强度检验：`sign(IC_valid) == sign(IC_train)` 且 `|ICIR_valid| >= min_valid_ic_ir` (默认 0.30)；
3. 截面换手率约束：单日换手率代理指标满足 `Turnover <= max_turnover` (默认 0.85)；
4. 贪心正交化去相关（Greedy Orthogonalization）：按验证集 `|ICIR_valid|` 降序遍历候选因子，
   仅当其与已入库集合及本批已入围集合的所有因子的最大相关系数均 `< max_library_corr` (默认 0.60) 时准予入选；
5. 入库登记：将合格因子全时段多周期评估后原子化写入 SQLite，状态标记为 `accepted`。
"""

from __future__ import annotations

import logging
import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from factorlib.core.config import AdmissionConfig
from factorlib.core.evaluate import FactorEvaluationResult, evaluate_factor
from factorlib.core.panel import Panel
from factorlib.core.registry import FactorRegistry
from factorlib.gp.select import _fitness_sort_key
from factorlib.gp.tree import Individual

logger = logging.getLogger(__name__)


@dataclass
class AdmittedFactor:
    """通过五步准入管线并登记入库的合格因子资产。"""

    canon_expr: str
    canon_hash: str
    sign: int
    n_nodes: int
    depth: int
    train_ic_mean: float
    train_ic_ir: float
    train_fitness: float
    valid_ic_mean: float
    valid_ic_ir: float
    valid_turnover: float
    max_corr: float
    factor_id: int | None = None
    status: str = "accepted"
    evaluation: FactorEvaluationResult | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "factor_id": self.factor_id,
            "canon_expr": self.canon_expr,
            "canon_hash": self.canon_hash,
            "sign": self.sign,
            "n_nodes": self.n_nodes,
            "depth": self.depth,
            "train_ic_mean": self.train_ic_mean,
            "train_ic_ir": self.train_ic_ir,
            "train_fitness": self.train_fitness,
            "valid_ic_mean": self.valid_ic_mean,
            "valid_ic_ir": self.valid_ic_ir,
            "valid_turnover": self.valid_turnover,
            "max_corr": self.max_corr,
            "status": self.status,
        }


@dataclass
class AdmissionSummary:
    """五步准入漏斗统计汇总报告。"""

    n_step1_top_k: int = 0
    n_step2_valid_ic: int = 0
    n_step3_turnover: int = 0
    n_step4_orthogonal: int = 0
    n_step5_registered: int = 0
    admitted: list[AdmittedFactor] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_step1_top_k": self.n_step1_top_k,
            "n_step2_valid_ic": self.n_step2_valid_ic,
            "n_step3_turnover": self.n_step3_turnover,
            "n_step4_orthogonal": self.n_step4_orthogonal,
            "n_step5_registered": self.n_step5_registered,
            "admitted": [f.to_dict() for f in self.admitted],
        }


class AdmissionPipeline:
    """五步全自动因子准入与正交化去相关管线。"""

    def __init__(
        self,
        cfg: AdmissionConfig | None = None,
        label_name: str = "y_5",
        horizon: int = 5,
        min_names: int = 5,
        mode: str = "cs",
        data_version: str = "v1",
        splits: dict[str, tuple[str, str]] | None = None,
    ) -> None:
        self.cfg = cfg or AdmissionConfig()
        self.label_name = label_name
        self.horizon = int(horizon)
        self.min_names = int(min_names)
        self.mode = mode
        self.data_version = data_version
        self.splits = splits

    def run(
        self,
        candidates: Sequence[Individual],
        engine: Any,
        panel: Panel,
        valid_slice: tuple[int, int],
        corr_slice: tuple[int, int] | None = None,
        registry: FactorRegistry | None = None,
        run_id: str = "gp_run",
        register_to_db: bool = True,
    ) -> AdmissionSummary:
        """对名人堂候选个体执行完整五步准入筛选并原子化登记入库。"""
        summary = AdmissionSummary()
        if not candidates:
            return summary

        val_start, val_end = valid_slice
        c_start, c_end = corr_slice if corr_slice is not None else (0, val_end)

        # 获取库内已有的 accepted / active 因子哈希与矩阵
        existing_hashes: set[str] = set()
        library_matrices: list[np.ndarray] = []
        if registry is not None:
            lib_records = registry.list_factors(status="accepted") + registry.list_factors(
                status="active"
            )
            lib_exprs: list[str] = []
            for rec in lib_records:
                existing_hashes.add(rec.canon_hash)
                lib_exprs.append(rec.canon_expr)
            if lib_exprs:
                evaluated_libs = engine.eval_batch_safe(
                    lib_exprs, use_cache=True, canonicalize=False
                )
                for mat in evaluated_libs:
                    if mat is not None:
                        arr = np.asfortranarray(np.asarray(mat)[c_start:c_end])
                        library_matrices.append(arr)

        # ----------------------------------------------------------------------
        # Step 1: Top-K 初筛 (去重 + 剔除已在库内的同指纹因子 + 按训练适应度降序取前 K)
        # ----------------------------------------------------------------------
        unique_map: dict[str, Individual] = {}
        for ind in candidates:
            if ind.is_degenerate or not ind.canon_hash:
                continue
            if ind.canon_hash in existing_hashes:
                continue
            if not math.isfinite(ind.adjusted_fitness) or ind.adjusted_fitness <= -5.0:
                continue
            prev = unique_map.get(ind.canon_hash)
            if prev is None or ind.adjusted_fitness > prev.adjusted_fitness:
                unique_map[ind.canon_hash] = ind

        sorted_step1 = sorted(unique_map.values(), key=_fitness_sort_key)[
            : self.cfg.top_k_candidates
        ]
        summary.n_step1_top_k = len(sorted_step1)
        if not sorted_step1:
            return summary

        # 批量求值 Step 1 候选因子全量矩阵
        step1_exprs = [ind.canon_expr for ind in sorted_step1]
        step1_mats = engine.eval_batch_safe(step1_exprs, use_cache=True, canonicalize=True)

        # 在验证集切片上批量计算 RankIC
        valid_ic_results = engine.eval_gp_batch(
            step1_exprs,
            label_name=self.label_name,
            start_row=val_start,
            end_row=val_end,
            min_names=self.min_names,
            horizon=self.horizon,
            library_factors=None,
            use_cache=True,
        )

        # ----------------------------------------------------------------------
        # Step 2 & Step 3: 验证集符号一致性 + |ICIR_valid| 阈值 + 截面换手率约束
        # ----------------------------------------------------------------------
        # 元组结构: (ind, full_mat, val_ic_mean, val_ic_ir, val_turnover)
        step3_survivors: list[tuple[Individual, np.ndarray, float, float, float]] = []

        for ind, mat_obj, val_ic in zip(sorted_step1, step1_mats, valid_ic_results, strict=True):
            if mat_obj is None or not val_ic["valid"]:
                continue

            val_valid_days = int(val_ic.get("valid_days", 0))
            min_val_days = max(2, int((val_end - val_start) * 0.5))
            if val_valid_days < min_val_days:
                continue

            val_mean = float(val_ic["ic_mean"])
            val_ir = float(val_ic["ic_ir"])
            train_mean = float(ind.ic_mean)

            if not (math.isfinite(val_mean) and math.isfinite(val_ir)):
                continue

            # Step 2a: 符号一致性检验
            if self.cfg.require_sign_consistency:
                if abs(train_mean) < 1e-12 or abs(val_mean) < 1e-12:
                    continue
                train_sign = 1 if train_mean > 0.0 else -1
                val_sign = 1 if val_mean > 0.0 else -1
                if train_sign != val_sign:
                    continue

            # Step 2b: 验证集 |ICIR| 强度检验
            if abs(val_ir) < self.cfg.min_valid_ic_ir:
                continue

            summary.n_step2_valid_ic += 1

            # Step 3: 截面换手率检验
            full_arr = np.asfortranarray(np.asarray(mat_obj))
            to_res = engine.factor_turnover(
                full_arr,
                min_names=self.min_names,
                start_row=val_start,
                end_row=val_end,
            )
            turnover_val = float(to_res["turnover"])
            if not math.isfinite(turnover_val):
                turnover_val = 0.0
            if turnover_val > self.cfg.max_turnover:
                continue

            summary.n_step3_turnover += 1
            step3_survivors.append((ind, full_arr, val_mean, val_ir, turnover_val))

        if not step3_survivors:
            return summary

        # ----------------------------------------------------------------------
        # Step 4: 贪心正交化去相关 (按验证集 |ICIR_valid| 降序遍历)
        # ----------------------------------------------------------------------
        step3_survivors.sort(
            key=lambda item: (
                -abs(item[3]),
                -abs(item[0].ic_ir),
                item[0].n_nodes,
                item[0].canon_hash,
            )
        )

        accepted_pool_mats: list[np.ndarray] = list(library_matrices)
        step4_admitted: list[tuple[Individual, np.ndarray, float, float, float, float]] = []

        for ind, full_arr, val_mean, val_ir, turnover_val in step3_survivors:
            cand_slice = np.asfortranarray(full_arr[c_start:c_end])
            max_abs_rho = 0.0
            is_redundant = False

            for ref_mat in accepted_pool_mats:
                rho = abs(
                    float(
                        engine.cross_correlation(
                            cand_slice,
                            ref_mat,
                            min_names=self.min_names,
                            rank=True,
                            start_row=c_start,
                            end_row=c_end,
                        )
                    )
                )
                if math.isfinite(rho):
                    if rho > max_abs_rho:
                        max_abs_rho = rho
                    if rho >= self.cfg.max_library_corr:
                        is_redundant = True
                        break

            if is_redundant:
                continue

            accepted_pool_mats.append(cand_slice)
            step4_admitted.append((ind, full_arr, val_mean, val_ir, turnover_val, max_abs_rho))

        summary.n_step4_orthogonal = len(step4_admitted)

        # ----------------------------------------------------------------------
        # Step 5: 入库登记 (计算全样本段与多窗口指标，原子写入 SQLite)
        # ----------------------------------------------------------------------
        for ind, _, val_mean, val_ir, turnover_val, max_abs_rho in step4_admitted:
            rec_sign = 1 if ind.ic_mean >= 0.0 else -1
            eval_res: FactorEvaluationResult | None = None
            fid: int | None = None

            if register_to_db and registry is not None:
                eval_res = evaluate_factor(
                    engine_or_panel=panel,
                    expr=ind.canon_expr,
                    splits=self.splits,
                    min_names=self.min_names,
                    force_sign=rec_sign,
                )
                f_rec = eval_res.to_factor_record(
                    source="gp",
                    source_run=run_id,
                    data_version=self.data_version,
                    status="accepted",
                )
                f_rec.mode = self.mode
                fid = registry.register_factor(f_rec, run_id=run_id, upsert=True)
                summary.n_step5_registered += 1

            admitted_obj = AdmittedFactor(
                factor_id=fid,
                canon_expr=ind.canon_expr,
                canon_hash=ind.canon_hash,
                sign=rec_sign,
                n_nodes=ind.n_nodes,
                depth=ind.depth,
                train_ic_mean=ind.ic_mean,
                train_ic_ir=ind.ic_ir,
                train_fitness=ind.adjusted_fitness,
                valid_ic_mean=val_mean,
                valid_ic_ir=val_ir,
                valid_turnover=turnover_val,
                max_corr=max_abs_rho,
                status="accepted",
                evaluation=eval_res,
            )
            summary.admitted.append(admitted_obj)

        logger.info(
            "准入漏斗完成: TopK=%d -> ValidIC=%d -> Turnover=%d -> Orthogonal=%d -> Registered=%d",
            summary.n_step1_top_k,
            summary.n_step2_valid_ic,
            summary.n_step3_turnover,
            summary.n_step4_orthogonal,
            summary.n_step5_registered,
        )
        return summary
