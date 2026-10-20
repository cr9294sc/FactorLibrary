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
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import polars as pl


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
        特征字段映射表, 键为特征名 (如 'close', 'rv', 'volume'),
        值为 (T, N) 的二维 float64 矩阵, 严格遵循 Fortran 顺序 (order='F')
    mask:
        可交易品种池掩码布尔矩阵, 形状 (T, N), True 表示该标的在该交易日有效在池
    labels:
        预测收益标签映射表, 键为标签名 (如 'y_1', 'y_5'),
        值为 (T, N) 的二维 float64 矩阵 (order='F')
    meta:
        构建元数据字典 (配置指纹、生成时间、版本等)
    """

    dates: np.ndarray
    symbols: List[str]
    fields: Dict[str, np.ndarray]
    mask: np.ndarray
    labels: Dict[str, np.ndarray] = field(default_factory=dict)
    meta: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        T = len(self.dates)
        N = len(self.symbols)

        if self.mask.shape != (T, N):
            raise ValueError(f"mask 形状 {self.mask.shape} 与 (T={T}, N={N}) 不匹配")

        for k, v in self.fields.items():
            if v.shape != (T, N):
                raise ValueError(f"字段 {k!r} 形状 {v.shape} 与 (T={T}, N={N}) 不匹配")
            if not v.flags.f_contiguous:
                self.fields[k] = np.asfortranarray(v)

        for k, v in self.labels.items():
            if v.shape != (T, N):
                raise ValueError(f"标签 {k!r} 形状 {v.shape} 与 (T={T}, N={N}) 不匹配")
            if not v.flags.f_contiguous:
                self.labels[k] = np.asfortranarray(v)

    @property
    def shape(self) -> Tuple[int, int]:
        """(T, N): 交易日天数 x 品种数。"""
        return len(self.dates), len(self.symbols)

    @property
    def T(self) -> int:
        return len(self.dates)

    @property
    def N(self) -> int:
        return len(self.symbols)

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

    @classmethod
    def load(
        cls,
        path: Union[str, Path],
        fields_subset: Optional[Sequence[str]] = None,
        load_labels: bool = True,
    ) -> "Panel":
        """从标准派生数据目录 (包含 panel_daily.parquet) 加载面板。

        Parameters
        ----------
        path:
            面板存储目录 (例如 ``data/derived/futures/v1``)
        fields_subset:
            可选, 仅加载指定特征字段子集以降低内存开销; 若为 None 则加载全部数值特征
        load_labels:
            是否同时加载 labels.parquet
        """
        dir_path = Path(path)
        panel_file = dir_path / "panel_daily.parquet"
        labels_file = dir_path / "labels.parquet"
        meta_file = dir_path / "meta.json"

        if not panel_file.exists():
            raise FileNotFoundError(f"未找到面板文件: {panel_file}")

        meta: Dict[str, Any] = {}
        if meta_file.exists():
            with meta_file.open("r", encoding="utf-8") as fh:
                meta = json.load(fh)

        # 读取元信息列以确定 T 与 N 轴
        base_lf = pl.scan_parquet(str(panel_file))
        schema = base_lf.collect_schema()

        # 确定需要读取的列
        all_cols = list(schema.names())
        ignore_cols = {"trading_date", "symbol", "contract", "prev_contract", "is_roll", "date_idx", "sector", "in_universe"}

        if fields_subset is not None:
            load_cols = [c for c in fields_subset if c in all_cols]
        else:
            load_cols = [c for c in all_cols if c not in ignore_cols]

        needed_cols = list(set(["trading_date", "symbol", "in_universe"] + load_cols))
        df = pl.read_parquet(panel_file, columns=needed_cols).sort(["trading_date", "symbol"])

        unique_dates_list = df["trading_date"].unique().sort().to_list()
        unique_dates = np.array(unique_dates_list)
        unique_symbols = df["symbol"].unique().sort().to_list()

        T = len(unique_dates)
        N = len(unique_symbols)

        # 建立整数索引映射以实现 O(1) 向量化填充
        date_map = {d: i for i, d in enumerate(unique_dates_list)}
        symbol_map = {s: i for i, s in enumerate(unique_symbols)}

        row_indices = np.array([date_map[d] for d in df["trading_date"].to_list()], dtype=np.int32)
        col_indices = np.array([symbol_map[s] for s in df["symbol"].to_list()], dtype=np.int32)

        # 构建 mask
        mask = np.zeros((T, N), dtype=bool, order="F")
        if "in_universe" in df.columns:
            mask_vals = df["in_universe"].fill_null(False).to_numpy()
            mask[row_indices, col_indices] = mask_vals

        # 构建各字段宽矩阵
        fields_dict: Dict[str, np.ndarray] = {}
        for col_name in load_cols:
            col_series = df[col_name]
            # 仅处理数值型与布尔型
            if col_series.dtype in (pl.Float32, pl.Float64, pl.Int32, pl.Int64, pl.UInt32, pl.UInt64, pl.Boolean):
                arr = np.full((T, N), np.nan, dtype=np.float64, order="F")
                vals = col_series.cast(pl.Float64).to_numpy()
                arr[row_indices, col_indices] = vals
                fields_dict[col_name] = arr

        # 加载标签
        labels_dict: Dict[str, np.ndarray] = {}
        if load_labels and labels_file.exists():
            ldf = pl.read_parquet(labels_file).sort(["trading_date", "symbol"])
            l_dates = ldf["trading_date"].to_list()
            l_syms = ldf["symbol"].to_list()
            valid_idx = [i for i, (d, s) in enumerate(zip(l_dates, l_syms)) if d in date_map and s in symbol_map]
            l_row_idx = np.array([date_map[l_dates[i]] for i in valid_idx], dtype=np.int32)
            l_col_idx = np.array([symbol_map[l_syms[i]] for i in valid_idx], dtype=np.int32)

            for col_name in ldf.columns:
                if col_name in ("trading_date", "symbol"):
                    continue
                arr = np.full((T, N), np.nan, dtype=np.float64, order="F")
                vals = ldf[col_name].cast(pl.Float64).to_numpy()
                arr[l_row_idx, l_col_idx] = vals[valid_idx]
                labels_dict[col_name] = arr

        return cls(
            dates=unique_dates,
            symbols=unique_symbols,
            fields=fields_dict,
            mask=mask,
            labels=labels_dict,
            meta=meta,
        )

    def to_rust_engine(self, cache_mb: int = 4096, n_threads: Optional[int] = None) -> Any:
        """将内存矩阵零拷贝传递给 Rust 计算内核 (factorlib_core)。

        注意: factorlib_core 将在工程计划 P2 阶段构建提供。
        """
        try:
            import factorlib_core
        except ImportError as exc:
            raise NotImplementedError(
                "Rust 极速计算内核 `factorlib_core` 尚未构建安装。"
                "请按照实施路线图推进计划 P2 (Rust 内核实现)。"
            ) from exc

        return factorlib_core.Engine(
            self.fields,
            self.mask,
            self.labels,
            [str(d) for d in self.dates],
            self.symbols,
            cache_mb=cache_mb,
            n_threads=n_threads or 0,
        )
