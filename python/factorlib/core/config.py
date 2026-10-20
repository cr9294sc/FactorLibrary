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
