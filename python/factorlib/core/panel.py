"""标准化研究面板内存容器 (core/panel.py)。

按白皮书 6.1 节与 14.1 节设计:
- 所有二维连续面板字段均以 (T, N) 的 Fortran 列优先 (Column-Major) 布局存入内存,
  使单品种在连续交易日内的内存紧凑排列, 极大提升 CPU 缓存命中率与 SIMD 吞吐;
- 停牌、未上市或数据缺失处统一以 NaN 表达;
- 预测标签 (labels) 与特征字段 (fields) 物理隔离存储, 防止 DSL 语法树求值越权访问;
- 提供 `load` 方法, 从派生目录秒级加载 Parquet 长表并向量化重塑为 (T, N) 宽矩阵。
"""

from __future__ import annotations

import json
from collections.abc import Collection, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from factorlib.core._hooks import resolve_panel_market, resolve_profile
from factorlib.core.extensions import (
    ExtensionPack,
    resolve_extension_pack,
)

# 允许留在 panel.fields 中供分组与掩码算子使用的结构化辅助字段
STRUCTURAL_PANEL_AUX_FIELDS: frozenset[str] = frozenset(
    {"sector", "listed_days", "in_universe_mask", "limit_mask"}
)

# 未显式传入 aux_field_names 且无 fields.json 时的通用回测辅助字段默认集合
DEFAULT_AUX_BACKTEST_FIELDS: frozenset[str] = frozenset(
    {
        "exec_vwap",
        "exec_vwap_old",
        "open_raw",
        "close_raw",
        "high_raw",
        "low_raw",
        "settle_raw",
        "limit_up",
        "limit_down",
        "is_unilateral",
        "adj_factor",
        "ex_cum_factor",
        "is_roll",
        "traded",
        "adv20_turnover",
        "multiplier",
        "margin_rate",
        "days_to_expiry",
        "is_suspended",
        "is_st",
        "can_buy_mask",
        "can_sell_mask",
        "bench_fwd_open_1",
        "bench_fwd_close_1",
        "bench_ret",
    }
)


def _resolve_aux_field_names(
    dir_path: Path,
    aux_field_names: Collection[str] | None,
) -> frozenset[str]:
    """确定应隔离进 aux_fields 的列名集合（优先显式参数 -> fields.json -> 默认集合）。"""
    if aux_field_names is not None:
        return frozenset(aux_field_names)
    fields_json = dir_path / "fields.json"
    if fields_json.exists():
        try:
            with fields_json.open("r", encoding="utf-8") as fh:
                doc = json.load(fh)
            parsed = {
                str(spec["name"])
                for spec in doc.get("fields", [])
                if isinstance(spec, dict)
                and spec.get("role") == "aux"
                and str(spec.get("name")) not in STRUCTURAL_PANEL_AUX_FIELDS
            }
            if parsed:
                return frozenset(parsed)
        except Exception:
            pass
    return DEFAULT_AUX_BACKTEST_FIELDS


def _lookup_panel_field(
    name: str,
    aux_fields: dict[str, np.ndarray] | None = None,
    fields: dict[str, np.ndarray] | None = None,
) -> np.ndarray | None:
    """按 aux_fields 优先、fields 回退的顺序查找二维字段矩阵。"""
    if aux_fields is not None and name in aux_fields:
        return aux_fields[name]
    if fields is not None and name in fields:
        return fields[name]
    return None


def compute_limit_mask(
    valid_mask: np.ndarray,
    aux_fields: dict[str, np.ndarray] | None = None,
    fields: dict[str, np.ndarray] | None = None,
) -> np.ndarray:
    """根据在池掩码及停牌/单边市/涨跌停/成交量标记计算可交易掩码矩阵 (1.0 / NaN, order='F')。"""
    tradable = np.array(valid_mask, dtype=bool, copy=True, order="F")

    is_unilateral = _lookup_panel_field("is_unilateral", aux_fields=aux_fields, fields=fields)
    if is_unilateral is not None:
        tradable &= ~(np.isfinite(is_unilateral) & (is_unilateral > 0.5))

    traded = _lookup_panel_field("traded", aux_fields=aux_fields, fields=fields)
    if traded is not None:
        tradable &= np.isfinite(traded) & (traded > 0.5)

    volume = _lookup_panel_field("volume", aux_fields=aux_fields, fields=fields)
    if volume is not None:
        tradable &= np.isfinite(volume) & (volume > 0.0)

    is_suspended = _lookup_panel_field("is_suspended", aux_fields=aux_fields, fields=fields)
    if is_suspended is not None:
        tradable &= ~(np.isfinite(is_suspended) & (is_suspended > 0.5))

        limit_up = _lookup_panel_field("limit_up", aux_fields=aux_fields, fields=fields)
        limit_down = _lookup_panel_field("limit_down", aux_fields=aux_fields, fields=fields)
        close_raw = _lookup_panel_field("close_raw", aux_fields=aux_fields, fields=fields)
        if close_raw is None:
            close_raw = _lookup_panel_field("close", aux_fields=aux_fields, fields=fields)
        high_raw = _lookup_panel_field("high_raw", aux_fields=aux_fields, fields=fields)
        if high_raw is None:
            high_raw = _lookup_panel_field("high", aux_fields=aux_fields, fields=fields)
        low_raw = _lookup_panel_field("low_raw", aux_fields=aux_fields, fields=fields)
        if low_raw is None:
            low_raw = _lookup_panel_field("low", aux_fields=aux_fields, fields=fields)

        eps = 1e-4
        if limit_up is not None and close_raw is not None:
            tradable &= ~(
                np.isfinite(limit_up) & np.isfinite(close_raw) & (close_raw >= limit_up - eps)
            )
        if limit_down is not None and close_raw is not None:
            tradable &= ~(
                np.isfinite(limit_down) & np.isfinite(close_raw) & (close_raw <= limit_down + eps)
            )
        if high_raw is not None and low_raw is not None:
            flat_hl = (
                np.isfinite(high_raw) & np.isfinite(low_raw) & (np.abs(high_raw - low_raw) < eps)
            )
            if limit_up is not None:
                tradable &= ~(flat_hl & np.isfinite(limit_up) & (high_raw >= limit_up - eps))
            if limit_down is not None:
                tradable &= ~(flat_hl & np.isfinite(limit_down) & (low_raw <= limit_down + eps))

    return np.asfortranarray(np.where(tradable, 1.0, np.nan), dtype=np.float64)


@dataclass
class Panel:
    """期货/多资产截面研究面板内存容器。

    Attributes
    ----------
    dates:
        一维时间轴数组, 形状 (T,), 升序排列, dtype 通常为 datetime64[D] 或字符串
    symbols:
        品种/标的代码列表, 长度 N, 升序排列 (如 ['A', 'AG', 'AL', ..., 'ZN'])
    fields:
        特征字段映射表, 键为特征名 (如 'close', 'rv', 'volume') 及结构化分组/掩码字段,
        值为 (T, N) 的二维 float64 矩阵, 严格遵循 Fortran 顺序 (order='F')
    mask:
        可交易品种池掩码布尔矩阵, 形状 (T, N), True 表示该标的在该交易日有效在池
    labels:
        预测收益标签映射表, 键为标签名 (如 'y_1', 'y_5'),
        值为 (T, N) 的二维 float64 矩阵 (order='F')
    aux_fields:
        回测执行价格、未复权价格、涨跌停标记与合约规格等辅助字段映射表,
        与 `fields` 物理隔离以防止 DSL 语法树与 Python 因子上下文越权访问
    meta:
        构建元数据字典 (配置指纹、生成时间、版本等)
    """

    dates: np.ndarray
    symbols: list[str]
    fields: dict[str, np.ndarray]
    mask: np.ndarray
    labels: dict[str, np.ndarray] = field(default_factory=dict)
    aux_fields: dict[str, np.ndarray] = field(default_factory=dict)
    meta: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        T = len(self.dates)
        N = len(self.symbols)

        if self.mask.shape != (T, N):
            raise ValueError(f"mask 形状 {self.mask.shape} 与 (T={T}, N={N}) 不匹配")

        for group_name, mapping in (
            ("字段", self.fields),
            ("标签", self.labels),
            ("辅助字段", self.aux_fields),
        ):
            for k, v in mapping.items():
                if v.shape != (T, N):
                    raise ValueError(f"{group_name} {k!r} 形状 {v.shape} 与 (T={T}, N={N}) 不匹配")
                if not v.flags.f_contiguous:
                    mapping[k] = np.asfortranarray(v)

    @property
    def shape(self) -> tuple[int, int]:
        """(T, N): 交易日天数 x 品种数。"""
        return len(self.dates), len(self.symbols)

    @property
    def T(self) -> int:
        return len(self.dates)

    @property
    def N(self) -> int:
        return len(self.symbols)

    @property
    def valid_mask(self) -> np.ndarray:
        """与 mask 等价的属性别名。"""
        return self.mask

    def get_field(self, name: str) -> np.ndarray:
        """获取指定字段 (T, N) 矩阵。"""
        if name not in self.fields:
            raise KeyError(f"面板中未找到字段: {name!r}, 可选字段: {list(self.fields.keys())}")
        return self.fields[name]

    def get_label(self, name: str) -> np.ndarray:
        """获取指定标签 (T, N) 矩阵。"""
        if name not in self.labels:
            raise KeyError(f"面板中未找到标签: {name!r}, 可选标签: {list(self.labels.keys())}")
        return self.labels[name]

    def get_aux_field(self, name: str) -> np.ndarray:
        """获取指定回测/诊断辅助字段 (T, N) 矩阵。"""
        if name not in self.aux_fields:
            raise KeyError(
                f"面板中未找到辅助字段: {name!r}, 可选辅助字段: {list(self.aux_fields.keys())}"
            )
        return self.aux_fields[name]

    def has_any_field(self, name: str) -> bool:
        """判断字段是否存在于 `aux_fields` 或 `fields`（优先回测辅助字段，向后兼容手工构造面板）。"""
        return name in self.aux_fields or name in self.fields

    def get_any_field(self, name: str) -> np.ndarray:
        """优先从 `aux_fields` 读取字段，若不存在则回退到 `fields`。"""
        arr = _lookup_panel_field(name, aux_fields=self.aux_fields, fields=self.fields)
        if arr is None:
            raise KeyError(
                f"面板中未找到字段或辅助字段: {name!r}, "
                f"aux_fields={list(self.aux_fields.keys())}, fields={list(self.fields.keys())}"
            )
        return arr

    @classmethod
    def load(
        cls,
        path: str | Path,
        fields_subset: Sequence[str] | None = None,
        load_labels: bool = True,
        aux_field_names: Collection[str] | None = None,
    ) -> Panel:
        """从标准派生数据目录 (包含 panel_daily.parquet) 加载面板。

        Parameters
        ----------
        path:
            面板存储目录 (例如 ``data/derived/futures/v1``)
        fields_subset:
            可选, 仅加载指定特征字段子集以降低内存开销; 若为 None 则加载全部数值特征
        load_labels:
            是否同时加载 labels.parquet
        aux_field_names:
            可选, 需隔离至 ``aux_fields`` 的辅助字段名集合 (通常由 ``MarketProfile`` 传入)
        """
        dir_path = Path(path)
        panel_file = dir_path / "panel_daily.parquet"
        labels_file = dir_path / "labels.parquet"
        meta_file = dir_path / "meta.json"

        if not panel_file.exists():
            raise FileNotFoundError(f"未找到面板文件: {panel_file}")

        meta: dict[str, Any] = {}
        if meta_file.exists():
            with meta_file.open("r", encoding="utf-8") as fh:
                meta = json.load(fh)

        # 读取元信息列以确定 T 与 N 轴
        base_lf = pl.scan_parquet(str(panel_file))
        schema = base_lf.collect_schema()

        # 确定需要读取的列
        all_cols = list(schema.names())
        ignore_cols = {
            "trading_date",
            "symbol",
            "contract",
            "prev_contract",
            "date_idx",
            "sector",
            "in_universe",
        }

        aux_names_set = _resolve_aux_field_names(dir_path, aux_field_names)

        want_limit_mask = fields_subset is None or "limit_mask" in fields_subset
        unrequested_volume = False
        if fields_subset is not None:
            load_cols = [c for c in fields_subset if c in all_cols]
            if want_limit_mask:
                for dep_col in ("traded", "is_unilateral"):
                    if dep_col in all_cols and dep_col not in load_cols:
                        load_cols.append(dep_col)
                if "volume" in all_cols and "volume" not in load_cols:
                    load_cols.append("volume")
                    unrequested_volume = True
            load_sector = "sector" in fields_subset and "sector" in all_cols
        else:
            load_cols = [c for c in all_cols if c not in ignore_cols]
            load_sector = "sector" in all_cols

        extra_cols = ["trading_date", "symbol", "in_universe"]
        if load_sector:
            extra_cols.append("sector")
        needed_cols = list(set(extra_cols + load_cols))
        df = (
            pl.read_parquet(panel_file, columns=needed_cols)
            .sort(["trading_date", "symbol"])
            .with_columns(
                _r=pl.col("trading_date").rank("dense").cast(pl.Int32) - 1,
                _c=pl.col("symbol").rank("dense").cast(pl.Int32) - 1,
            )
        )

        unique_dates_list = df["trading_date"].unique(maintain_order=True).to_list()
        unique_dates = np.array(unique_dates_list)
        unique_symbols = df["symbol"].unique(maintain_order=True).sort().to_list()

        T = len(unique_dates)
        N = len(unique_symbols)

        row_indices = df["_r"].to_numpy()
        col_indices = df["_c"].to_numpy()

        # 构建 mask
        mask = np.zeros((T, N), dtype=bool, order="F")
        if "in_universe" in df.columns:
            mask_vals = df["in_universe"].fill_null(False).to_numpy()
            mask[row_indices, col_indices] = mask_vals

        # 构建各字段宽矩阵（DSL/结构化特征入 fields，回测/原始/规格辅助列隔离入 aux_fields）
        fields_dict: dict[str, np.ndarray] = {}
        aux_fields_dict: dict[str, np.ndarray] = {}
        limit_extra_fields: dict[str, np.ndarray] = {}
        for col_name in load_cols:
            if col_name == "sector":
                continue
            col_series = df[col_name]
            # 仅处理数值型与布尔型
            if col_series.dtype in (
                pl.Float32,
                pl.Float64,
                pl.Int32,
                pl.Int64,
                pl.UInt32,
                pl.UInt64,
                pl.Boolean,
            ):
                arr = np.full((T, N), np.nan, dtype=np.float64, order="F")
                vals = col_series.cast(pl.Float64).to_numpy()
                arr[row_indices, col_indices] = vals
                if col_name == "volume" and unrequested_volume:
                    limit_extra_fields[col_name] = arr
                elif col_name in aux_names_set:
                    aux_fields_dict[col_name] = arr
                else:
                    fields_dict[col_name] = arr

        # 编码板块字段 sector 为正整数矩阵 (1.0, 2.0, ..., K; 缺失为 NaN)
        if load_sector and "sector" in df.columns:
            sec_col = df["sector"]
            sec_arr = np.full((T, N), np.nan, dtype=np.float64, order="F")
            if sec_col.dtype in (
                pl.Float32,
                pl.Float64,
                pl.Int32,
                pl.Int64,
                pl.UInt32,
                pl.UInt64,
            ):
                sec_arr[row_indices, col_indices] = sec_col.cast(pl.Float64).to_numpy()
            else:
                sec_raw = sec_col.to_list()
                valid_secs = sorted({str(s) for s in sec_raw if s is not None and str(s).strip()})
                sec_to_code = {s: float(idx + 1) for idx, s in enumerate(valid_secs)}
                meta["sector_mapping"] = {int(v): k for k, v in sec_to_code.items()}
                sec_vals = np.array(
                    [sec_to_code.get(str(s), np.nan) if s is not None else np.nan for s in sec_raw],
                    dtype=np.float64,
                )
                sec_arr[row_indices, col_indices] = sec_vals
            fields_dict["sector"] = sec_arr

        # 补充 A 股结构化字段别名 (stock_listed_days -> listed_days, industry_code -> sector)
        if (
            "stock_listed_days" in fields_dict
            and "listed_days" not in fields_dict
            and (fields_subset is None or "listed_days" in fields_subset)
        ):
            fields_dict["listed_days"] = fields_dict["stock_listed_days"]
        if (
            "industry_code" in fields_dict
            and "sector" not in fields_dict
            and (fields_subset is None or "sector" in fields_subset)
        ):
            fields_dict["sector"] = fields_dict["industry_code"]

        # 当全量加载或显式请求时，注入后处理隐式掩码矩阵 (in_universe_mask 与 limit_mask)
        if fields_subset is None or "in_universe_mask" in (fields_subset or []):
            univ_mask_arr = np.where(mask, 1.0, np.nan).astype(np.float64, order="F")
            fields_dict["in_universe_mask"] = univ_mask_arr

        if want_limit_mask:
            fields_for_limit = (
                {**fields_dict, **limit_extra_fields} if limit_extra_fields else fields_dict
            )
            fields_dict["limit_mask"] = compute_limit_mask(
                mask, aux_fields=aux_fields_dict, fields=fields_for_limit
            )

        # 加载标签
        labels_dict: dict[str, np.ndarray] = {}
        if load_labels and labels_file.exists():
            ldf = pl.read_parquet(labels_file).sort(["trading_date", "symbol"])
            if (
                len(ldf) == len(df)
                and ldf["trading_date"].equals(df["trading_date"])
                and ldf["symbol"].equals(df["symbol"])
            ):
                l_row_idx = row_indices
                l_col_idx = col_indices
                valid_mask_lbl = None
            else:
                date_map = {d: i for i, d in enumerate(unique_dates_list)}
                symbol_map = {s: i for i, s in enumerate(unique_symbols)}
                l_dates = ldf["trading_date"].to_list()
                l_syms = ldf["symbol"].to_list()
                valid_idx = [
                    i
                    for i, (d, s) in enumerate(zip(l_dates, l_syms, strict=True))
                    if d in date_map and s in symbol_map
                ]
                l_row_idx = np.array([date_map[l_dates[i]] for i in valid_idx], dtype=np.int32)
                l_col_idx = np.array([symbol_map[l_syms[i]] for i in valid_idx], dtype=np.int32)
                valid_mask_lbl = valid_idx

            for col_name in ldf.columns:
                if col_name in ("trading_date", "symbol"):
                    continue
                arr = np.full((T, N), np.nan, dtype=np.float64, order="F")
                vals = ldf[col_name].cast(pl.Float64).to_numpy()
                if valid_mask_lbl is None:
                    arr[l_row_idx, l_col_idx] = vals
                else:
                    arr[l_row_idx, l_col_idx] = vals[valid_mask_lbl]
                labels_dict[col_name] = arr

        return cls(
            dates=unique_dates,
            symbols=unique_symbols,
            fields=fields_dict,
            mask=mask,
            labels=labels_dict,
            aux_fields=aux_fields_dict,
            meta=meta,
        )

    @property
    def engine(self) -> Any:
        """惰性缓存并返回当前 Panel 绑定的 Rust 计算内核实例。"""
        return self.to_rust_engine()

    def to_rust_engine(self, cache_mb: int = 4096, n_threads: int | None = None) -> Any:
        """将内存矩阵传递给 Rust 计算内核 (factorlib_core)，并在面板未变更时复用缓存实例。"""
        try:
            import factorlib_core
        except ImportError as exc:
            raise NotImplementedError(
                "Rust 极速计算内核 `factorlib_core` 尚未构建安装。"
                "请按照实施路线图推进计划 P2 (Rust 内核实现)。"
            ) from exc

        eff_threads = n_threads or 0
        state_key = (
            cache_mb,
            eff_threads,
            len(self.dates),
            len(self.symbols),
            id(self.mask),
            int(np.sum(self.mask[-1])) if self.mask.size else 0,
            tuple(
                (
                    k,
                    id(v),
                    float(np.nan_to_num(v[-1, 0], nan=-999.0)) if v.size else 0.0,
                    float(np.nan_to_num(v[-1, -1], nan=-999.0)) if v.size else 0.0,
                )
                for k, v in self.fields.items()
            ),
            tuple(
                (
                    k,
                    id(v),
                    float(np.nan_to_num(v[0, 0], nan=-999.0)) if v.size else 0.0,
                    float(np.nan_to_num(v[-1, -1], nan=-999.0)) if v.size else 0.0,
                )
                for k, v in self.labels.items()
            ),
        )
        cached = getattr(self, "_cached_engine_pair", None)
        if cached is not None and cached[0] == state_key:
            return cached[1]

        eng = factorlib_core.Engine(
            self.fields,
            self.mask,
            self.labels,
            [str(d) for d in self.dates],
            self.symbols,
            cache_mb=cache_mb,
            n_threads=eff_threads,
        )
        object.__setattr__(self, "_cached_engine_pair", (state_key, eng))
        return eng

    @property
    def attached_extensions(self) -> list[dict[str, Any]]:
        """返回当前面板已挂载的扩展特征包元数据列表。"""
        raw = self.meta.get("extensions", [])
        return list(raw) if isinstance(raw, list) else []

    def attach_extension(
        self,
        pack: ExtensionPack | str | Path,
        *,
        fields_subset: Sequence[str] | None = None,
        overwrite: bool = False,
        extensions_dir: Path | None = None,
    ) -> list[str]:
        """按需挂载日频扩展特征包（例如 1m 微观结构扩展包）到 `Panel.fields`。

        严格校验：
        1. 扩展包必须显式声明 `lag_days >= 0` 与 `asof_rule`；
        2. 禁止跨市场挂载（`pack.meta.market` 须与 `panel.meta['market']` 一致）；
        3. 禁止覆盖 `aux_fields` 与结构化字段；
        4. 按 `lag_days` 执行严格向前平移后以 Fortran 列优先写入 `self.fields`。
        """
        if isinstance(pack, ExtensionPack):
            ext_pack = pack
        else:
            resolved_dir = extensions_dir
            if resolved_dir is None and not Path(pack).is_dir():
                market = resolve_panel_market(self)
                prof = resolve_profile(market)
                resolved_dir = getattr(prof.paths(), "extensions_dir", None)
            ext_pack = resolve_extension_pack(pack, extensions_dir=resolved_dir)

        panel_market = self.meta.get("market")
        if isinstance(panel_market, str) and panel_market.strip():
            if ext_pack.meta.market != panel_market.strip().lower():
                raise ValueError(
                    f"禁止跨市场挂载扩展包: 面板市场为 {panel_market!r}, "
                    f"扩展包 {ext_pack.meta.pack_id!r} 市场为 {ext_pack.meta.market!r}"
                )

        aligned = ext_pack.load_aligned_matrices(
            dates=self.dates,
            symbols=self.symbols,
            fields_subset=fields_subset,
        )
        attached_cols: list[str] = []
        for col_name, mat in aligned.items():
            if col_name in STRUCTURAL_PANEL_AUX_FIELDS or col_name in self.aux_fields:
                raise ValueError(f"扩展包字段 {col_name!r} 与受保护结构/辅助字段冲突，拒绝挂载")
            if col_name in self.fields and not overwrite:
                continue
            self.fields[col_name] = np.asfortranarray(mat, dtype=np.float64)
            attached_cols.append(col_name)

        exts = [
            item
            for item in self.attached_extensions
            if isinstance(item, dict) and item.get("pack_id") != ext_pack.meta.pack_id
        ]
        exts.append(
            {
                "pack_id": ext_pack.meta.pack_id,
                "version": ext_pack.meta.version,
                "qualified_id": ext_pack.meta.qualified_id,
                "market": ext_pack.meta.market,
                "kind": ext_pack.meta.kind,
                "lag_days": ext_pack.meta.lag_days,
                "asof_rule": ext_pack.meta.asof_rule,
                "source_fingerprint": ext_pack.meta.source_fingerprint,
                "fields": attached_cols,
            }
        )
        self.meta["extensions"] = exts
        return attached_cols

    def detach_extension(self, pack_id: str) -> list[str]:
        """卸载指定 `pack_id`（或 `pack_id@version`）的已挂载扩展字段。"""
        target_id = str(pack_id).split("@", 1)[0].strip()
        removed: list[str] = []
        remaining: list[dict[str, Any]] = []
        for item in self.attached_extensions:
            if isinstance(item, dict) and item.get("pack_id") == target_id:
                for col in item.get("fields", []):
                    if col in self.fields:
                        del self.fields[col]
                        removed.append(col)
            elif isinstance(item, dict):
                remaining.append(item)
        self.meta["extensions"] = remaining
        return removed
