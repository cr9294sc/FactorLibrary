"""配置加载与强类型校验 (core/config.py)。

仅使用标准库 dataclasses 做强类型封装, 加载 ``configs/futures/*.yaml``。
本模块只服务于 P1 ETL 所需的 ``data.yaml`` 与 ``universe.yaml``, 后续阶段
(gp / ml / backtest 配置) 在此基础上扩展。
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from datetime import date
from pathlib import Path
from typing import Any

import yaml


class ConfigError(ValueError):
    """配置缺失或取值非法。"""


def _parse_date(value: Any, name: str) -> date:
    if isinstance(value, date):
        return value
    try:
        return date.fromisoformat(str(value))
    except ValueError as exc:  # pragma: no cover - 错误信息路径
        raise ConfigError(f"{name} 不是合法的 ISO 日期: {value!r}") from exc


def _section(raw: dict[str, Any], key: str) -> dict[str, Any]:
    value = raw.get(key, {})
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ConfigError(f"配置节 {key!r} 必须是映射, 实际为 {type(value).__name__}")
    return value


@dataclass(frozen=True)
class DominantConfig:
    rule: int = 3
    rank: int = 1

    def __post_init__(self) -> None:
        if self.rule not in (0, 1, 2, 3):
            raise ConfigError(f"dominant.rule 必须在 0..3, 实际 {self.rule}")
        if self.rank not in (1, 2, 3):
            raise ConfigError(f"dominant.rank 必须在 1..3, 实际 {self.rank}")


@dataclass(frozen=True)
class AdjustConfig:
    method: str = "prev_close_ratio"
    verify_with_ex_factor: bool = True
    verify_tolerance: float = 0.0005

    def __post_init__(self) -> None:
        if self.method != "prev_close_ratio":
            raise ConfigError(f"目前仅支持 adjust.method=prev_close_ratio, 实际 {self.method!r}")
        if self.verify_tolerance <= 0:
            raise ConfigError("adjust.verify_tolerance 必须 > 0")


@dataclass(frozen=True)
class ExecPriceConfig:
    window_minutes: int = 30
    calc_method: str = "vwap_close_volume"

    def __post_init__(self) -> None:
        if self.window_minutes <= 0:
            raise ConfigError("exec_price.window_minutes 必须 > 0")
        if self.calc_method != "vwap_close_volume":
            # 严禁使用 turnover/volume (郑商所成交额历史置零, 见审计 R3)
            raise ConfigError("exec_price.calc_method 只允许 vwap_close_volume")


@dataclass(frozen=True)
class MemberRankConfig:
    enabled: bool = True
    top_n: int = 20
    delta_lag: int = 5


@dataclass(frozen=True)
class FeaturesConfig:
    intraday_minute_features: bool = True
    roll_yield: bool = True
    warehouse_stocks: bool = True
    member_rank: MemberRankConfig = field(default_factory=MemberRankConfig)


@dataclass(frozen=True)
class MinuteFeaturesConfig:
    max_gap_minutes: int = 60
    night_start_hour: int = 18
    night_end_hour: int = 3


@dataclass(frozen=True)
class WarehouseConfig:
    chg_lag: int = 5
    z_window: int = 60
    z_min_samples: int = 30
    eps: float = 1.0

    def __post_init__(self) -> None:
        if not (0 < self.z_min_samples <= self.z_window):
            raise ConfigError("warehouse.z_min_samples 必须在 (0, z_window] 内")


@dataclass(frozen=True)
class DataConfig:
    raw_dir: Path
    derived_dir: Path
    panel_version: str = "v1"
    dominant: DominantConfig = field(default_factory=DominantConfig)
    adjust: AdjustConfig = field(default_factory=AdjustConfig)
    exec_price: ExecPriceConfig = field(default_factory=ExecPriceConfig)
    start_date: date = date(2010, 1, 1)
    end_date: date = date(2026, 8, 31)
    features: FeaturesConfig = field(default_factory=FeaturesConfig)
    minute_features: MinuteFeaturesConfig = field(default_factory=MinuteFeaturesConfig)
    warehouse: WarehouseConfig = field(default_factory=WarehouseConfig)
    label_horizons: tuple[int, ...] = (1, 3, 5, 10, 20)

    def __post_init__(self) -> None:
        if self.start_date > self.end_date:
            raise ConfigError("timeline.start_date 不得晚于 end_date")
        if not self.label_horizons or any(h <= 0 for h in self.label_horizons):
            raise ConfigError("labels.horizons 必须为正整数列表")

    @property
    def panel_dir(self) -> Path:
        """派生面板输出目录 ``derived_dir/panel_version``。"""
        return self.derived_dir / self.panel_version

    def to_dict(self) -> dict[str, Any]:
        """可 JSON 序列化的完整配置快照 (用于元数据哈希)。"""
        out = asdict(self)
        out["raw_dir"] = str(self.raw_dir)
        out["derived_dir"] = str(self.derived_dir)
        out["start_date"] = self.start_date.isoformat()
        out["end_date"] = self.end_date.isoformat()
        out["label_horizons"] = list(self.label_horizons)
        return out

    def hash(self) -> str:
        """配置语义哈希 (不含路径, 路径变化不改变面板语义)。"""
        payload = self.to_dict()
        payload.pop("raw_dir", None)
        payload.pop("derived_dir", None)
        blob = json.dumps(payload, sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class UniverseConfig:
    min_adv20_turnover: float = 50_000_000.0
    min_listed_days: int = 60
    min_days_to_expiry: int = 5
    min_names_per_day: int = 10

    def __post_init__(self) -> None:
        if self.min_listed_days < 0 or self.min_days_to_expiry < 0:
            raise ConfigError("universe 天数阈值不得为负")
        if self.min_names_per_day < 1:
            raise ConfigError("evaluation.min_names_per_day 必须 >= 1")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _load_yaml(path: Path) -> dict[str, Any]:
    path = Path(path)
    if not path.exists():
        raise ConfigError(f"配置文件不存在: {path}")
    with path.open("r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}
    if not isinstance(raw, dict):
        raise ConfigError(f"配置文件根节点必须是映射: {path}")
    return raw


def _resolve(path_str: str, root: Path) -> Path:
    p = Path(path_str)
    return p if p.is_absolute() else (root / p).resolve()


def load_data_config(path: Path, root: Path | None = None) -> DataConfig:
    """加载 ``data.yaml``。相对路径以 ``root`` (默认当前工作目录) 为基准解析。"""
    root = Path(root) if root is not None else Path.cwd()
    raw = _load_yaml(path)

    for key in ("raw_dir", "derived_dir"):
        if key not in raw:
            raise ConfigError(f"data.yaml 缺少必填项 {key}")

    dom = _section(raw, "dominant")
    adj = _section(raw, "adjust")
    ex = _section(raw, "exec_price")
    tl = _section(raw, "timeline")
    feats = _section(raw, "features")
    mf = _section(raw, "minute_features")
    wh = _section(raw, "warehouse")
    mr_extra = _section(raw, "member_rank")
    lab = _section(raw, "labels")

    mr_feat = feats.get("member_rank", {})
    if isinstance(mr_feat, bool):
        mr_feat = {"enabled": mr_feat}
    member = MemberRankConfig(
        enabled=bool(mr_feat.get("enabled", True)),
        top_n=int(mr_feat.get("top_n", 20)),
        delta_lag=int(mr_extra.get("delta_lag", 5)),
    )
    features = FeaturesConfig(
        intraday_minute_features=bool(feats.get("intraday_minute_features", True)),
        roll_yield=bool(feats.get("roll_yield", True)),
        warehouse_stocks=bool(feats.get("warehouse_stocks", True)),
        member_rank=member,
    )

    kwargs: dict[str, Any] = {
        "raw_dir": _resolve(raw["raw_dir"], root),
        "derived_dir": _resolve(raw["derived_dir"], root),
        "panel_version": str(raw.get("panel_version", "v1")),
        "dominant": DominantConfig(**dom),
        "adjust": AdjustConfig(**adj),
        "exec_price": ExecPriceConfig(**ex),
        "features": features,
        "minute_features": MinuteFeaturesConfig(**mf),
        "warehouse": WarehouseConfig(**wh),
    }
    if "start_date" in tl:
        kwargs["start_date"] = _parse_date(tl["start_date"], "timeline.start_date")
    if "end_date" in tl:
        kwargs["end_date"] = _parse_date(tl["end_date"], "timeline.end_date")
    if "horizons" in lab:
        kwargs["label_horizons"] = tuple(int(h) for h in lab["horizons"])
    return DataConfig(**kwargs)


def load_universe_config(path: Path) -> UniverseConfig:
    """加载 ``universe.yaml``。"""
    raw = _load_yaml(path)
    liq = _section(raw, "liquidity")
    ev = _section(raw, "evaluation")
    return UniverseConfig(
        min_adv20_turnover=float(liq.get("min_adv20_turnover", 50_000_000.0)),
        min_listed_days=int(liq.get("min_listed_days", 60)),
        min_days_to_expiry=int(liq.get("min_days_to_expiry", 5)),
        min_names_per_day=int(ev.get("min_names_per_day", 10)),
    )


# ==============================================================================
# P4: 强类型遗传规划 (STGP) 配置定义与校验
# ==============================================================================


@dataclass(frozen=True)
class MigrationConfig:
    interval: int = 10
    k_individuals: int = 20

    def __post_init__(self) -> None:
        if self.interval < 1:
            raise ConfigError(f"migration.interval 必须 >= 1, 实际 {self.interval}")
        if self.k_individuals < 0:
            raise ConfigError(f"migration.k_individuals 必须 >= 0, 实际 {self.k_individuals}")


@dataclass(frozen=True)
class TreeLimitsConfig:
    init_min_depth: int = 2
    init_max_depth: int = 5
    max_depth: int = 6
    max_nodes: int = 30

    def __post_init__(self) -> None:
        if self.init_min_depth < 1:
            raise ConfigError(f"limits.init_min_depth 必须 >= 1, 实际 {self.init_min_depth}")
        if self.init_max_depth < self.init_min_depth:
            raise ConfigError(
                f"limits.init_max_depth ({self.init_max_depth}) 必须 >= init_min_depth ({self.init_min_depth})"
            )
        if self.max_depth < self.init_max_depth:
            raise ConfigError(
                f"limits.max_depth ({self.max_depth}) 必须 >= init_max_depth ({self.init_max_depth})"
            )
        if self.max_nodes < 2:
            raise ConfigError(f"limits.max_nodes 必须 >= 2, 实际 {self.max_nodes}")


@dataclass(frozen=True)
class VariationConfig:
    crossover_prob: float = 0.60
    subtree_mut_prob: float = 0.15
    point_mut_prob: float = 0.10
    hoist_mut_prob: float = 0.05
    param_mut_prob: float = 0.10

    def __post_init__(self) -> None:
        probs = (
            self.crossover_prob,
            self.subtree_mut_prob,
            self.point_mut_prob,
            self.hoist_mut_prob,
            self.param_mut_prob,
        )
        if any(p < 0.0 for p in probs):
            raise ConfigError("variation 各项概率必须 >= 0")
        total = sum(probs)
        if total <= 0.0:
            raise ConfigError("variation 各项概率之和必须 > 0")


@dataclass(frozen=True)
class FitnessConfig:
    metric: str = "ic_ir"
    parsimony_penalty: float = 0.002
    min_coverage: float = 0.70
    corr_penalty_threshold: float = 0.60
    corr_penalty_factor: float = 0.50

    def __post_init__(self) -> None:
        if self.metric not in ("ic_ir", "ic_mean"):
            raise ConfigError(f"fitness.metric 必须为 'ic_ir' 或 'ic_mean', 实际 {self.metric!r}")
        if self.parsimony_penalty < 0.0:
            raise ConfigError("fitness.parsimony_penalty 必须 >= 0")
        if not (0.0 <= self.min_coverage <= 1.0):
            raise ConfigError("fitness.min_coverage 必须在 [0.0, 1.0] 内")
        if not (0.0 <= self.corr_penalty_threshold <= 1.0):
            raise ConfigError("fitness.corr_penalty_threshold 必须在 [0.0, 1.0] 内")
        if not (0.0 <= self.corr_penalty_factor <= 1.0):
            raise ConfigError("fitness.corr_penalty_factor 必须在 [0.0, 1.0] 内")


@dataclass(frozen=True)
class AdmissionConfig:
    min_valid_ic_ir: float = 0.30
    require_sign_consistency: bool = True
    max_library_corr: float = 0.60
    max_turnover: float = 0.85
    top_k_candidates: int = 200

    def __post_init__(self) -> None:
        if self.min_valid_ic_ir < 0.0:
            raise ConfigError("admission.min_valid_ic_ir 必须 >= 0")
        if not (0.0 <= self.max_library_corr <= 1.0):
            raise ConfigError("admission.max_library_corr 必须在 [0.0, 1.0] 内")
        if not (0.0 <= self.max_turnover <= 2.0):
            raise ConfigError("admission.max_turnover 必须在 [0.0, 2.0] 内")
        if self.top_k_candidates < 1:
            raise ConfigError("admission.top_k_candidates 必须 >= 1")


def _default_gp_splits() -> dict[str, tuple[str, str]]:
    return {
        "train": ("2011-01-01", "2018-12-31"),
        "valid": ("2019-01-01", "2021-12-31"),
        "test": ("2022-01-01", "2026-08-31"),
    }


@dataclass(frozen=True)
class GPConfig:
    seed: int = 42
    mode: str = "cs"
    label: str = "y_5"
    splits: dict[str, tuple[str, str]] = field(default_factory=_default_gp_splits)
    population: int = 1000
    generations: int = 50
    islands: int = 4
    tournament_size: int = 7
    elitism: int = 5
    hall_of_fame_size: int = 200
    min_names: int = 5
    migration: MigrationConfig = field(default_factory=MigrationConfig)
    limits: TreeLimitsConfig = field(default_factory=TreeLimitsConfig)
    variation: VariationConfig = field(default_factory=VariationConfig)
    fitness: FitnessConfig = field(default_factory=FitnessConfig)
    admission: AdmissionConfig = field(default_factory=AdmissionConfig)

    def __post_init__(self) -> None:
        if self.mode not in ("cs", "ts"):
            raise ConfigError(f"gp.mode 必须为 'cs' 或 'ts', 实际 {self.mode!r}")
        if self.population < 4:
            raise ConfigError(f"gp.population 必须 >= 4, 实际 {self.population}")
        if self.generations < 1:
            raise ConfigError(f"gp.generations 必须 >= 1, 实际 {self.generations}")
        if self.islands < 1 or self.islands > self.population:
            raise ConfigError(f"gp.islands 必须在 [1, {self.population}] 内, 实际 {self.islands}")
        if self.tournament_size < 1:
            raise ConfigError("gp.tournament_size 必须 >= 1")
        if self.elitism < 0:
            raise ConfigError("gp.elitism 必须 >= 0")

    def to_dict(self) -> dict[str, Any]:
        out = asdict(self)
        out["splits"] = {k: list(v) for k, v in self.splits.items()}
        return out

    def to_yaml(self) -> str:
        return yaml.safe_dump(self.to_dict(), sort_keys=False, allow_unicode=True)


def load_gp_config(path: Path) -> GPConfig:
    """加载并强校验 ``gp_default.yaml``。"""
    raw = _load_yaml(path)

    mig = _section(raw, "migration")
    lim = _section(raw, "limits")
    var = _section(raw, "variation")
    fit = _section(raw, "fitness")
    adm = _section(raw, "admission")
    raw_splits = _section(raw, "splits")

    splits = _default_gp_splits()
    for split_key, val in raw_splits.items():
        if not isinstance(val, (list, tuple)) or len(val) != 2:
            raise ConfigError(f"splits.{split_key} 必须为长度为 2 的起止日期列表")
        d0 = _parse_date(val[0], f"splits.{split_key}[0]")
        d1 = _parse_date(val[1], f"splits.{split_key}[1]")
        if d0 > d1:
            raise ConfigError(f"splits.{split_key} 起始日期不得晚于结束日期")
        splits[str(split_key)] = (d0.isoformat(), d1.isoformat())

    return GPConfig(
        seed=int(raw.get("seed", 42)),
        mode=str(raw.get("mode", "cs")),
        label=str(raw.get("label", "y_5")),
        splits=splits,
        population=int(raw.get("population", 1000)),
        generations=int(raw.get("generations", 50)),
        islands=int(raw.get("islands", 4)),
        tournament_size=int(raw.get("tournament_size", 7)),
        elitism=int(raw.get("elitism", 5)),
        hall_of_fame_size=int(raw.get("hall_of_fame_size", adm.get("top_k_candidates", 200))),
        min_names=int(raw.get("min_names", 5)),
        migration=MigrationConfig(**mig),
        limits=TreeLimitsConfig(**lim),
        variation=VariationConfig(**var),
        fitness=FitnessConfig(**fit),
        admission=AdmissionConfig(**adm),
    )


@dataclass(frozen=True)
class WalkForwardConfig:
    """Purged & Embargoed 滚动扩展/滚动窗口配置。"""

    start_predict_year: int = 2018
    end_predict_year: int = 2026
    train_window: str = "expanding"  # expanding | rolling
    min_train_years: int = 5
    purge_window: int = 5
    validation_size_days: int = 252
    embargo_window: int = 0

    def __post_init__(self) -> None:
        if self.train_window not in ("expanding", "rolling"):
            raise ConfigError(
                f"walk_forward.train_window 必须为 'expanding' 或 'rolling', 实际 {self.train_window!r}"
            )
        if self.start_predict_year > self.end_predict_year:
            raise ConfigError("walk_forward.start_predict_year 不得晚于 end_predict_year")
        if self.min_train_years < 1:
            raise ConfigError("walk_forward.min_train_years 必须 >= 1")
        if self.purge_window < 0:
            raise ConfigError("walk_forward.purge_window 必须 >= 0")
        if self.validation_size_days < 1:
            raise ConfigError("walk_forward.validation_size_days 必须 >= 1")
        if self.embargo_window < 0:
            raise ConfigError("walk_forward.embargo_window 必须 >= 0")


@dataclass(frozen=True)
class LightGBMConfig:
    """LightGBM 截面排序回归超参数配置 (防过拟合强约束)。"""

    objective: str = "regression_l2"
    metric: str = "rmse"
    boosting_type: str = "gbdt"
    max_depth: int = 4
    num_leaves: int = 15
    learning_rate: float = 0.03
    n_estimators: int = 1000
    min_child_samples: int = 200
    feature_fraction: float = 0.70
    bagging_fraction: float = 0.80
    bagging_freq: int = 1
    early_stopping_rounds: int = 50
    seed: int = 42

    def __post_init__(self) -> None:
        if self.max_depth < 1:
            raise ConfigError("lightgbm.max_depth 必须 >= 1")
        if self.num_leaves < 2:
            raise ConfigError("lightgbm.num_leaves 必须 >= 2")
        if self.learning_rate <= 0:
            raise ConfigError("lightgbm.learning_rate 必须 > 0")
        if self.n_estimators < 1:
            raise ConfigError("lightgbm.n_estimators 必须 >= 1")
        if self.min_child_samples < 1:
            raise ConfigError("lightgbm.min_child_samples 必须 >= 1")
        if not (0 < self.feature_fraction <= 1.0):
            raise ConfigError("lightgbm.feature_fraction 必须在 (0, 1] 之间")
        if not (0 < self.bagging_fraction <= 1.0):
            raise ConfigError("lightgbm.bagging_fraction 必须在 (0, 1] 之间")


@dataclass(frozen=True)
class MLConfig:
    """机器学习多因子合成整体配置。"""

    target_label: str = "y_5"
    features_status: tuple[str, ...] = ("accepted", "active")
    walk_forward: WalkForwardConfig = field(default_factory=WalkForwardConfig)
    lightgbm: LightGBMConfig = field(default_factory=LightGBMConfig)
    models: tuple[str, ...] = ("equal_weight", "rolling_icir", "ridge", "lightgbm")

    def __post_init__(self) -> None:
        if not self.target_label:
            raise ConfigError("ml.target_label 不能为空")
        for s in self.features_status:
            if s not in ("candidate", "accepted", "active", "deprecated"):
                raise ConfigError(f"无效的 features_status: {s!r}")
        valid_models = {"equal_weight", "rolling_icir", "ridge", "lightgbm"}
        for m in self.models:
            if m not in valid_models:
                raise ConfigError(f"无效的 model: {m!r}, 必须在 {valid_models} 中")

    def to_dict(self) -> dict[str, Any]:
        out = asdict(self)
        out["features_status"] = list(self.features_status)
        out["models"] = list(self.models)
        return out

    def to_yaml(self) -> str:
        return yaml.safe_dump(self.to_dict(), sort_keys=False, allow_unicode=True)


def load_ml_config(path: Path) -> MLConfig:
    """加载并强校验 ``ml_default.yaml``。"""
    raw = _load_yaml(path)
    wf_sec = _section(raw, "walk_forward")
    lgb_sec = _section(raw, "lightgbm")

    status_raw = raw.get("features_status", ["accepted", "active"])
    if isinstance(status_raw, str):
        status_tuple = (status_raw,)
    else:
        status_tuple = tuple(status_raw)

    models_raw = raw.get("models", ["equal_weight", "rolling_icir", "ridge", "lightgbm"])
    if isinstance(models_raw, str):
        models_tuple = (models_raw,)
    else:
        models_tuple = tuple(models_raw)

    return MLConfig(
        target_label=str(raw.get("target_label", "y_5")),
        features_status=status_tuple,
        walk_forward=WalkForwardConfig(**wf_sec),
        lightgbm=LightGBMConfig(**lgb_sec),
        models=models_tuple,
    )
