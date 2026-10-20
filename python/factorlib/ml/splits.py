"""Purged & Embargoed Walk-Forward 扩展/滚动窗口切分器 (ml/splits.py)。

严格遵循白皮书 11.2 节架构设计：
- 采用扩展窗口 (Expanding Window) 或固定滚动窗口 (Rolling Window)；
- 在每个训练集与验证集末尾严格剔除 h 个交易日 (Purge Window = h)，杜绝长周期标签前瞻泄漏；
- 支持验证集与测试集起始端 Embargo 静默隔离；
- 输出强类型 WalkForwardFold 容器，支持全生命周期因果可追溯性审计。
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from factorlib.core.config import WalkForwardConfig


def _extract_year(val: Any) -> int:
    """提取日期对象的年份整数。"""
    if isinstance(val, (dt.date, dt.datetime)):
        return val.year
    if isinstance(val, np.datetime64):
        return int(str(val)[:4])
    s = str(val).strip()
    if len(s) >= 4 and s[:4].isdigit():
        return int(s[:4])
    raise ValueError(f"无法从值 {val!r} 解析年份")


@dataclass(frozen=True)
class WalkForwardFold:
    """单折 Walk-Forward 切分数据容器。"""

    fold_idx: int
    predict_year: int
    train_indices: np.ndarray
    val_indices: np.ndarray
    test_indices: np.ndarray
    train_dates: np.ndarray
    val_dates: np.ndarray
    test_dates: np.ndarray
    purge_train_indices: np.ndarray
    purge_val_indices: np.ndarray
    embargo_val_indices: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.int64))
    embargo_test_indices: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.int64))

    def __post_init__(self) -> None:
        # 校验时序严格单调与零重叠红线
        if len(self.train_indices) > 0 and len(self.val_indices) > 0:
            if np.max(self.train_indices) >= np.min(self.val_indices):
                raise ValueError("训练集最大索引必须严格小于验证集最小索引 (存在信息交叉)")

        if len(self.val_indices) > 0 and len(self.test_indices) > 0:
            if np.max(self.val_indices) >= np.min(self.test_indices):
                raise ValueError("验证集最大索引必须严格小于测试集最小索引 (存在信息交叉)")

        if len(self.purge_train_indices) > 0 and len(self.train_indices) > 0:
            if np.max(self.train_indices) >= np.min(self.purge_train_indices):
                raise ValueError("Purge 训练隔离区必须位于训练集有效数据之后")

        if len(self.purge_val_indices) > 0 and len(self.val_indices) > 0:
            if np.max(self.val_indices) >= np.min(self.purge_val_indices):
                raise ValueError("Purge 验证隔离区必须位于验证集有效数据之后")

    @property
    def n_train(self) -> int:
        return len(self.train_indices)

    @property
    def n_val(self) -> int:
        return len(self.val_indices)

    @property
    def n_test(self) -> int:
        return len(self.test_indices)

    @property
    def n_purge(self) -> int:
        return len(self.purge_train_indices) + len(self.purge_val_indices)


class PurgedWalkForwardSplitter:
    """Purged & Embargoed 滚动扩展窗口切分器。

    Parameters
    ----------
    start_predict_year:
        首个预测年份 (如 2018)
    end_predict_year:
        最终预测年份 (如 2026)
    train_window:
        'expanding' (扩展窗口) 或 'rolling' (固定窗口)
    min_train_years:
        最少历史训练年份数 (默认 5)
    purge_window:
        训练集与验证集尾部净化天数 h (必须与标签期数一致, 默认 5)
    validation_size_days:
        早停验证集交易日数 (默认 252, 约为 1 年)
    embargo_window:
        验证集/测试集起始端静默观察天数 (默认 0)
    strict_min_years:
        是否严格强制历史年份数 >= min_train_years (测试小样本时可置为 False)
    """

    def __init__(
        self,
        start_predict_year: int = 2018,
        end_predict_year: int = 2026,
        train_window: str = "expanding",
        min_train_years: int = 5,
        purge_window: int = 5,
        validation_size_days: int = 252,
        embargo_window: int = 0,
        strict_min_years: bool = True,
    ) -> None:
        if train_window not in ("expanding", "rolling"):
            raise ValueError(
                f"train_window 必须为 'expanding' 或 'rolling', 实际为 {train_window!r}"
            )
        if start_predict_year > end_predict_year:
            raise ValueError(
                f"start_predict_year ({start_predict_year}) 不能晚于 end_predict_year ({end_predict_year})"
            )
        if purge_window < 0:
            raise ValueError(f"purge_window 不能为负数: {purge_window}")
        if embargo_window < 0:
            raise ValueError(f"embargo_window 不能为负数: {embargo_window}")

        self.start_predict_year = start_predict_year
        self.end_predict_year = end_predict_year
        self.train_window = train_window
        self.min_train_years = min_train_years
        self.purge_window = purge_window
        self.validation_size_days = validation_size_days
        self.embargo_window = embargo_window
        self.strict_min_years = strict_min_years

    @classmethod
    def from_config(
        cls, cfg: WalkForwardConfig, strict_min_years: bool = True
    ) -> PurgedWalkForwardSplitter:
        """从 WalkForwardConfig 强类型配置实例构建切分器。"""
        return cls(
            start_predict_year=cfg.start_predict_year,
            end_predict_year=cfg.end_predict_year,
            train_window=cfg.train_window,
            min_train_years=cfg.min_train_years,
            purge_window=cfg.purge_window,
            validation_size_days=cfg.validation_size_days,
            embargo_window=cfg.embargo_window,
            strict_min_years=strict_min_years,
        )

    def split(self, dates: Sequence[Any]) -> list[WalkForwardFold]:
        """对给定的交易日期时间序列执行全流程 Purged Walk-Forward 切分。

        Parameters
        ----------
        dates:
            一维有序日期序列 (字符串、date、datetime 或 datetime64)

        Returns
        -------
        list[WalkForwardFold]
            生成的每折切分容器列表
        """
        arr_dates = np.asarray(dates)
        T = len(arr_dates)
        if T == 0:
            return []

        years = np.array([_extract_year(d) for d in arr_dates], dtype=np.int32)
        unique_years = sorted(set(years.tolist()))

        # 筛选满足预测区间的年份
        target_years = [
            y for y in unique_years if self.start_predict_year <= y <= self.end_predict_year
        ]

        folds: list[WalkForwardFold] = []
        for fold_idx, pred_year in enumerate(target_years):
            # 1. 确定测试集 (Year = pred_year)
            test_mask = years == pred_year
            raw_test_idx = np.where(test_mask)[0]
            if len(raw_test_idx) == 0:
                continue

            test_start = raw_test_idx[0]
            if test_start == 0:
                # 没有任何历史数据供训练与验证
                continue

            # Embargo 处理测试集
            if self.embargo_window > 0 and len(raw_test_idx) > self.embargo_window:
                embargo_test_idx = raw_test_idx[: self.embargo_window]
                test_idx = raw_test_idx[self.embargo_window :]
            else:
                embargo_test_idx = np.array([], dtype=np.int64)
                test_idx = raw_test_idx

            # 2. 确定候选验证集 (通常取 pred_year - 1 年，或往前至多 validation_size_days 天)
            prior_indices = np.arange(0, test_start, dtype=np.int64)
            val_year = pred_year - 1
            val_year_mask = years[prior_indices] == val_year
            if np.any(val_year_mask):
                cand_val_idx = prior_indices[val_year_mask]
            else:
                # 如果没有整年前置年，截取最后至多 validation_size_days 天且不占用超过半数历史
                max_val_len = min(self.validation_size_days, max(1, len(prior_indices) // 2))
                cand_val_idx = prior_indices[-max_val_len:]

            if len(cand_val_idx) <= self.purge_window:
                # 验证集天数不足以进行 Purge 净化
                continue

            # Purge 验证集末尾 h 天 (防止标签穿透至 test 集)
            if self.purge_window > 0:
                purge_val_idx = cand_val_idx[-self.purge_window :]
                eff_val_idx = cand_val_idx[: -self.purge_window]
            else:
                purge_val_idx = np.array([], dtype=np.int64)
                eff_val_idx = cand_val_idx

            # Embargo 验证集头部
            if self.embargo_window > 0 and len(eff_val_idx) > self.embargo_window:
                embargo_val_idx = eff_val_idx[: self.embargo_window]
                val_idx = eff_val_idx[self.embargo_window :]
            else:
                embargo_val_idx = np.array([], dtype=np.int64)
                val_idx = eff_val_idx

            # 3. 确定候选训练集 (验证集之前的所有交易日)
            val_start = cand_val_idx[0]
            cand_train_idx = prior_indices[prior_indices < val_start]
            if len(cand_train_idx) <= self.purge_window:
                # 训练数据不足
                continue

            train_years = sorted(set(years[cand_train_idx].tolist()))
            if self.strict_min_years and len(train_years) < self.min_train_years:
                # 严格模式下历史年限不足
                continue

            if self.train_window == "rolling" and len(train_years) > self.min_train_years:
                # 滚动窗口：只保留最近 min_train_years 年
                keep_years = set(train_years[-self.min_train_years :])
                mask_rolling = np.isin(years[cand_train_idx], list(keep_years))
                cand_train_idx = cand_train_idx[mask_rolling]

            # Purge 训练集末尾 h 天 (防止标签穿透至 val 集)
            if self.purge_window > 0:
                purge_train_idx = cand_train_idx[-self.purge_window :]
                train_idx = cand_train_idx[: -self.purge_window]
            else:
                purge_train_idx = np.array([], dtype=np.int64)
                train_idx = cand_train_idx

            if len(train_idx) == 0 or len(val_idx) == 0 or len(test_idx) == 0:
                continue

            folds.append(
                WalkForwardFold(
                    fold_idx=fold_idx,
                    predict_year=pred_year,
                    train_indices=train_idx,
                    val_indices=val_idx,
                    test_indices=test_idx,
                    train_dates=arr_dates[train_idx],
                    val_dates=arr_dates[val_idx],
                    test_dates=arr_dates[test_idx],
                    purge_train_indices=purge_train_idx,
                    purge_val_indices=purge_val_idx,
                    embargo_val_indices=embargo_val_idx,
                    embargo_test_indices=embargo_test_idx,
                )
            )

        return folds
