"""Parquet 安全读取与按月流式迭代 (etl/io.py)。

约定:
- ``exports/`` 为只读资产, 本模块只读不写;
- 分钟线 ``bar_1m`` 严禁一次性全量读取 (审计 R7), 一律通过 :func:`iter_minute_months`
  按月流式迭代, 单月数据处理完毕即释放;
- 米筐月度分区文件按 ``trading_date`` 所在月份切分 (夜盘 bar 的日历日可能落在上月末,
  但其 ``trading_date`` 归属本月), 因此按 ``trading_date`` 月份匹配文件是安全的。
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Iterator, Sequence
from pathlib import Path

import polars as pl

_MONTH_RE = re.compile(r"_(\d{4})_(\d{2})\.parquet$")

#: 分钟线读取的最小字段集 (投影下推, 降低 IO)
MINUTE_COLUMNS: tuple[str, ...] = (
    "order_book_id",
    "trading_date",
    "datetime",
    "open",
    "high",
    "low",
    "close",
    "volume",
)


class RawDataError(RuntimeError):
    """原始数据缺失或布局异常。"""


def dataset_files(raw_dir: Path, name: str) -> list[Path]:
    """返回数据集 ``name`` 的全部 Parquet 文件 (按文件名升序)。

    仅读取 ``full/`` 子目录; 单文件数据集 (calendar / instruments_info / industry_mapping)
    则直接返回 ``<name>/<name>.parquet``。
    """
    base = Path(raw_dir) / name
    full = base / "full"
    if full.is_dir():
        files = sorted(full.glob("*.parquet"))
    else:
        files = sorted(base.glob("*.parquet"))
    if not files:
        raise RawDataError(f"数据集 {name!r} 未找到任何 Parquet 文件: {base}")
    return files


def scan_dataset(raw_dir: Path, name: str, columns: Sequence[str] | None = None) -> pl.LazyFrame:
    """惰性扫描整个数据集 (适用于日线及以下体量的数据集, 不适用于 ``bar_1m``)。"""
    if name == "bar_1m":
        raise RawDataError("bar_1m 禁止一次性全量扫描, 请使用 iter_minute_months()")
    lf = pl.scan_parquet([str(p) for p in dataset_files(raw_dir, name)])
    if columns is not None:
        lf = lf.select(list(columns))
    return lf


def read_single(raw_dir: Path, name: str) -> pl.DataFrame:
    """读取单文件静态数据集。"""
    files = dataset_files(raw_dir, name)
    if len(files) != 1:
        raise RawDataError(f"数据集 {name!r} 预期单文件, 实际 {len(files)} 个")
    return pl.read_parquet(files[0])


def monthly_files(raw_dir: Path, name: str) -> dict[tuple[int, int], Path]:
    """解析 ``<name>_YYYY_MM.parquet`` 文件名, 返回 ``{(年, 月): 路径}``。"""
    out: dict[tuple[int, int], Path] = {}
    for path in dataset_files(raw_dir, name):
        m = _MONTH_RE.search(path.name)
        if m is None:
            raise RawDataError(f"无法从文件名解析年月: {path.name}")
        out[(int(m.group(1)), int(m.group(2)))] = path
    return out


def iter_minute_months(
    raw_dir: Path,
    pairs: pl.DataFrame,
    columns: Sequence[str] = MINUTE_COLUMNS,
) -> Iterator[tuple[tuple[int, int], pl.DataFrame]]:
    """按月流式迭代 1 分钟 bar, 仅保留 ``pairs`` 指定的 (合约, 交易日) 子集。

    Parameters
    ----------
    pairs:
        至少包含 ``order_book_id`` / ``trading_date`` 两列, 其余列 (如 ``symbol`` /
        ``prev_settlement``) 会通过内连接附加到分钟 bar 上。键必须唯一。

    Yields
    ------
    ((year, month), DataFrame)
        每月一个 DataFrame, 已按 (合约, 交易日, datetime) 排序, 价量列为 Float64。
        调用方处理完毕后应立即丢弃引用以释放内存。
    """
    if pairs.is_empty():
        return
    if pairs.select(["order_book_id", "trading_date"]).is_duplicated().any():
        raise ValueError("pairs 的 (order_book_id, trading_date) 键必须唯一")

    files = monthly_files(raw_dir, "bar_1m")
    by_month = pairs.with_columns(
        _y=pl.col("trading_date").dt.year().cast(pl.Int32),
        _m=pl.col("trading_date").dt.month().cast(pl.Int32),
    ).partition_by(["_y", "_m"], as_dict=True)
    for (year, month), part in sorted(by_month.items()):
        path = files.get((int(year), int(month)))
        if path is None:
            continue
        part = part.drop(["_y", "_m"])
        contracts = part["order_book_id"].unique().to_list()
        lo, hi = part["trading_date"].min(), part["trading_date"].max()
        df = (
            pl.scan_parquet(str(path))
            .select(list(columns))
            .filter(
                pl.col("order_book_id").is_in(pl.Series(contracts).implode())
                & pl.col("trading_date").is_between(lo, hi)
            )
            .join(part.lazy(), on=["order_book_id", "trading_date"], how="inner")
            .with_columns(pl.col(["open", "high", "low", "close", "volume"]).cast(pl.Float64))
            .sort(["order_book_id", "trading_date", "datetime"])
            .collect()
        )
        yield (int(year), int(month)), df


def source_fingerprint(raw_dir: Path, names: Sequence[str]) -> str:
    """原始文件指纹: 对 (相对路径, 字节数, mtime_ns) 做 SHA-256, 不读取文件内容。"""
    h = hashlib.sha256()
    root = Path(raw_dir)
    for name in sorted(names):
        for path in dataset_files(root, name):
            st = path.stat()
            h.update(f"{path.relative_to(root)}|{st.st_size}|{st.st_mtime_ns}\n".encode())
    return h.hexdigest()


def file_sha256(path: Path, chunk: int = 1 << 20) -> str:
    """文件内容 SHA-256 (用于派生面板的完整性标记)。"""
    h = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()
