"""1m 高频与另类日频扩展特征包契约与挂载加载器 (core/extensions.py)。

遵循混合投研架构设计 §3.3.2 契约：
- 核心 `1d` 主面板与 `1m` 扩展层职责彻底剥离；
- 每个扩展包落盘于独立目录（如 `data/extensions/<market>/<pack_id>/`），包含：
  - `pack.json`: 元数据清单，强制声明 `pack_id`, `market`, `version`, `kind`,
    `fields`, `lag_days`, `asof_rule` (`T_close` | `T+1_open`), `source_fingerprint`；
  - `features.parquet`: 按 `(trading_date, symbol)` 对齐的日频扩展特征长表；
- 严格防未来函数：未显式声明 `lag_days` 或 `asof_rule` 的扩展包拒绝挂载；
  挂载时按 `lag_days` 执行严格的向前平移（`t` 日数据仅在 `t + lag_days` 可见）。
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

VALID_PACK_KINDS: frozenset[str] = frozenset({"daily_features", "intraday_tensor"})
VALID_ASOF_RULES: frozenset[str] = frozenset({"T_close", "T+1_open"})
VALID_EXTENSION_MARKETS: frozenset[str] = frozenset({"futures", "stocks"})


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while chunk := fh.read(1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


@dataclass(frozen=True)
class ExtensionPackMeta:
    """日频扩展特征包元数据契约 (`pack.json`)。"""

    pack_id: str
    market: str
    version: str = "v1"
    kind: str = "daily_features"  # "daily_features" | "intraday_tensor"
    fields: tuple[str, ...] = ()
    lag_days: int = 0
    asof_rule: str = "T_close"  # "T_close" | "T+1_open"
    coverage: dict[str, Any] = field(default_factory=dict)
    source_fingerprint: str = ""
    description: str = ""
    created_at: str = field(default_factory=lambda: dt.datetime.now(dt.UTC).isoformat())

    def __post_init__(self) -> None:
        if not str(self.pack_id).strip():
            raise ValueError("ExtensionPackMeta.pack_id 不能为空")
        norm_market = str(self.market).strip().lower()
        if norm_market not in VALID_EXTENSION_MARKETS:
            raise ValueError(
                f"无效的扩展包市场域: {self.market!r}, 必须在 {sorted(VALID_EXTENSION_MARKETS)} 中"
            )
        object.__setattr__(self, "market", norm_market)

        if self.kind not in VALID_PACK_KINDS:
            raise ValueError(
                f"无效的扩展包类型 kind={self.kind!r}, 必须在 {sorted(VALID_PACK_KINDS)} 中"
            )
        if int(self.lag_days) < 0:
            raise ValueError(f"扩展包 lag_days 不得为负: {self.lag_days}")
        object.__setattr__(self, "lag_days", int(self.lag_days))

        if self.asof_rule not in VALID_ASOF_RULES:
            raise ValueError(
                f"无效的扩展包可得性规则 asof_rule={self.asof_rule!r}, "
                f"必须在 {sorted(VALID_ASOF_RULES)} 中"
            )
        object.__setattr__(self, "fields", tuple(str(f) for f in self.fields if str(f).strip()))

    @property
    def qualified_id(self) -> str:
        return f"{self.pack_id}@{self.version}"

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["fields"] = list(self.fields)
        return d

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, ensure_ascii=False)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> ExtensionPackMeta:
        if "lag_days" not in raw or raw["lag_days"] is None:
            raise ValueError("扩展包 pack.json 缺少必填防泄漏字段 `lag_days`")
        if "asof_rule" not in raw or not raw["asof_rule"]:
            raise ValueError("扩展包 pack.json 缺少必填防泄漏字段 `asof_rule`")
        if "pack_id" not in raw or not raw["pack_id"]:
            raise ValueError("扩展包 pack.json 缺少必填字段 `pack_id`")
        if "market" not in raw or not raw["market"]:
            raise ValueError("扩展包 pack.json 缺少必填字段 `market`")

        raw_fields = raw.get("fields", ())
        if raw_fields and isinstance(raw_fields[0], dict):
            fields_tuple = tuple(str(x.get("name", "")) for x in raw_fields if x.get("name"))
        else:
            fields_tuple = tuple(str(x) for x in raw_fields)

        return cls(
            pack_id=str(raw["pack_id"]),
            market=str(raw["market"]),
            version=str(raw.get("version", "v1")),
            kind=str(raw.get("kind", "daily_features")),
            fields=fields_tuple,
            lag_days=int(raw["lag_days"]),
            asof_rule=str(raw["asof_rule"]),
            coverage=dict(raw.get("coverage") or {}),
            source_fingerprint=str(raw.get("source_fingerprint", "")),
            description=str(raw.get("description", "")),
            created_at=str(raw.get("created_at") or dt.datetime.now(dt.UTC).isoformat()),
        )

    @classmethod
    def from_json_file(cls, pack_json_path: str | Path) -> ExtensionPackMeta:
        p = Path(pack_json_path)
        if not p.exists():
            raise FileNotFoundError(f"未找到扩展包清单文件: {p}")
        data = json.loads(p.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise ValueError(f"扩展包清单 {p} 必须是 JSON 对象")
        return cls.from_dict(data)


class ExtensionPack:
    """已加载契约并可按需挂载到 `Panel` 的日频扩展特征包。"""

    def __init__(self, pack_dir: str | Path, meta: ExtensionPackMeta) -> None:
        self.pack_dir = Path(pack_dir)
        self.meta = meta
        self.features_path = self.pack_dir / "features.parquet"

    @classmethod
    def from_dir(cls, pack_dir: str | Path) -> ExtensionPack:
        p = Path(pack_dir)
        meta = ExtensionPackMeta.from_json_file(p / "pack.json")
        pack = cls(pack_dir=p, meta=meta)
        if meta.kind == "daily_features" and not pack.features_path.exists():
            raise FileNotFoundError(f"扩展包缺少特征数据文件: {pack.features_path}")
        return pack

    def load_aligned_matrices(
        self,
        dates: np.ndarray,
        symbols: Sequence[str],
        fields_subset: Sequence[str] | None = None,
    ) -> dict[str, np.ndarray]:
        """读取 `features.parquet`，对齐到 `(T, N)` Fortran 矩阵并按 `lag_days` 强制因果平移。"""
        if self.meta.kind != "daily_features":
            raise NotImplementedError(
                f"扩展包 {self.meta.pack_id!r} 类型为 {self.meta.kind!r}，"
                "日内分箱张量暂未接入 1d 主面板挂载"
            )
        if not self.features_path.exists():
            raise FileNotFoundError(f"扩展包数据文件不存在: {self.features_path}")

        lf = pl.scan_parquet(str(self.features_path))
        schema = lf.collect_schema()
        all_cols = list(schema.names())
        if "trading_date" not in all_cols or "symbol" not in all_cols:
            raise ValueError(f"扩展包 {self.features_path} 必须包含 `trading_date` 与 `symbol` 列")

        candidate_cols = (
            list(self.meta.fields)
            if self.meta.fields
            else [c for c in all_cols if c not in ("trading_date", "symbol")]
        )
        if fields_subset is not None:
            wanted = set(fields_subset)
            target_cols = [c for c in candidate_cols if c in wanted and c in all_cols]
        else:
            target_cols = [c for c in candidate_cols if c in all_cols]

        if not target_cols:
            return {}

        df = pl.read_parquet(
            self.features_path,
            columns=["trading_date", "symbol", *target_cols],
        ).sort(["trading_date", "symbol"])

        # 统一交易日为 ISO 字符串进行对齐（兼容 Date / Datetime / String）
        date_series = df["trading_date"]
        if date_series.dtype == pl.Date:
            raw_dates = date_series.dt.strftime("%Y-%m-%d").to_list()
        elif isinstance(date_series.dtype, pl.Datetime):
            raw_dates = date_series.dt.date().dt.strftime("%Y-%m-%d").to_list()
        else:
            raw_dates = [str(d)[:10] for d in date_series.to_list()]

        raw_syms = [str(s) for s in df["symbol"].to_list()]
        panel_dates = [str(d)[:10] for d in dates]
        T = len(panel_dates)
        N = len(symbols)

        date_map = {d: i for i, d in enumerate(panel_dates)}
        symbol_map = {str(s): i for i, s in enumerate(symbols)}

        valid_rows: list[int] = []
        valid_r: list[int] = []
        valid_c: list[int] = []
        for idx, (d_str, sym) in enumerate(zip(raw_dates, raw_syms, strict=True)):
            r = date_map.get(d_str)
            c = symbol_map.get(sym)
            if r is not None and c is not None:
                valid_rows.append(idx)
                valid_r.append(r)
                valid_c.append(c)

        r_idx = np.array(valid_r, dtype=np.int32)
        c_idx = np.array(valid_c, dtype=np.int32)
        src_idx = np.array(valid_rows, dtype=np.int32)

        lag = int(self.meta.lag_days)
        out: dict[str, np.ndarray] = {}
        numeric_dtypes = (
            pl.Float32,
            pl.Float64,
            pl.Int8,
            pl.Int16,
            pl.Int32,
            pl.Int64,
            pl.UInt8,
            pl.UInt16,
            pl.UInt32,
            pl.UInt64,
            pl.Boolean,
        )
        for col_name in target_cols:
            series = df[col_name]
            if series.dtype not in numeric_dtypes:
                continue
            mat = np.full((T, N), np.nan, dtype=np.float64, order="F")
            if len(src_idx) > 0:
                vals = series.cast(pl.Float64).to_numpy()
                mat[r_idx, c_idx] = vals[src_idx]
            if lag > 0:
                shifted = np.full((T, N), np.nan, dtype=np.float64, order="F")
                if lag < T:
                    shifted[lag:, :] = mat[:-lag, :]
                mat = shifted
            out[col_name] = mat
        return out


def resolve_extension_pack(
    pack_ref: str | Path,
    extensions_dir: Path | None = None,
) -> ExtensionPack:
    """解析扩展包引用（支持目录路径、`pack_id` 或 `pack_id@version`）。"""
    ref_path = Path(pack_ref)
    if ref_path.is_dir() and (ref_path / "pack.json").exists():
        return ExtensionPack.from_dir(ref_path)

    raw_ref = str(pack_ref).strip()
    if not raw_ref:
        raise ValueError("扩展包标识不能为空")

    pack_id, _, req_version = raw_ref.partition("@")
    if extensions_dir is None:
        raise FileNotFoundError(f"无法定位扩展包 {raw_ref!r}：未提供 `extensions_dir` 且路径不存在")

    candidates = [extensions_dir / pack_id]
    if req_version:
        candidates.insert(0, extensions_dir / f"{pack_id}@{req_version}")
        candidates.insert(1, extensions_dir / pack_id / req_version)

    for cand in candidates:
        if cand.is_dir() and (cand / "pack.json").exists():
            pack = ExtensionPack.from_dir(cand)
            if req_version and pack.meta.version != req_version:
                raise ValueError(
                    f"扩展包 {pack_id!r} 版本不匹配: 请求 {req_version!r}, 实际 {pack.meta.version!r}"
                )
            return pack

    raise FileNotFoundError(f"在目录 {extensions_dir} 下未找到扩展包: {raw_ref!r}")


def discover_extension_packs(
    extensions_dir: Path,
    market: str | None = None,
) -> list[ExtensionPackMeta]:
    """扫描扩展包目录下全部合规 `pack.json` 清单并返回元数据列表。"""
    base = Path(extensions_dir)
    if not base.exists():
        return []

    found: list[ExtensionPackMeta] = []
    for pack_json in sorted(base.rglob("pack.json")):
        try:
            meta = ExtensionPackMeta.from_json_file(pack_json)
            if market is not None and meta.market != market.strip().lower():
                continue
            found.append(meta)
        except Exception:
            continue
    return found


def save_extension_pack(
    pack_dir: str | Path,
    meta: ExtensionPackMeta,
    df: pl.DataFrame,
) -> ExtensionPackMeta:
    """将日频扩展特征 DataFrame 与 `pack.json` 原子化落盘，并计算文件指纹。"""
    if "trading_date" not in df.columns or "symbol" not in df.columns:
        raise ValueError("保存扩展包要求 DataFrame 必须包含 `trading_date` 与 `symbol` 列")

    out_dir = Path(pack_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    features_path = out_dir / "features.parquet"

    sorted_df = df.sort(["trading_date", "symbol"])
    sorted_df.write_parquet(features_path)

    inferred_fields = meta.fields or tuple(
        c for c in sorted_df.columns if c not in ("trading_date", "symbol")
    )
    date_col = sorted_df["trading_date"]
    if sorted_df.height > 0:
        if date_col.dtype == pl.Date:
            start_d = date_col.min().isoformat()  # type: ignore[union-attr]
            end_d = date_col.max().isoformat()  # type: ignore[union-attr]
        else:
            start_d = str(date_col.min())[:10]
            end_d = str(date_col.max())[:10]
        n_syms = int(sorted_df["symbol"].n_unique())
    else:
        start_d = None
        end_d = None
        n_syms = 0

    coverage = dict(meta.coverage)
    coverage.setdefault("start_date", start_d)
    coverage.setdefault("end_date", end_d)
    coverage.setdefault("n_rows", int(sorted_df.height))
    coverage.setdefault("n_symbols", n_syms)

    fp = meta.source_fingerprint or _sha256_file(features_path)
    final_meta = ExtensionPackMeta(
        pack_id=meta.pack_id,
        market=meta.market,
        version=meta.version,
        kind=meta.kind,
        fields=inferred_fields,
        lag_days=meta.lag_days,
        asof_rule=meta.asof_rule,
        coverage=coverage,
        source_fingerprint=fp,
        description=meta.description,
        created_at=meta.created_at,
    )
    (out_dir / "pack.json").write_text(final_meta.to_json(), encoding="utf-8")
    return final_meta
