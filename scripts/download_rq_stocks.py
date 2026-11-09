#!/usr/bin/env python3
"""
米筐 RQData A 股历史日频数据本地化导出与限额断点续传工具 (scripts/download_rq_stocks.py)

核心特性:
1. 双环境自举: `--help` 零第三方依赖响应; 实际下载时若当前环境未安装 rqdatac,
   自动切换至项目隔离虚拟环境 `.venv-rqsdk/bin/python` 执行。
2. 零密钥泄漏: 自动从本地 `.env` (权限 0600, 已 gitignore) 或环境变量读取许可证,
   启用 Zstandard 传输压缩与北交所覆盖 (`use_zstd=True, enable_bjse=True`)。
3. 防幸存者偏差与省流架构:
   - 拉取全历史上市与已退市全部 A 股 (`all_instruments(type='CS')`), 按各年度实际存续期过滤查询列表;
   - 日线仅拉取 `adjust_type='none'` 原始行情 + 独立 `ex_factor` 除权因子事件表, 节省 2/3 行情流量;
   - 按自然年正向 (2010 -> 2026) 原子打包 `[bar_1d, trade_status, daily_basic, industry_sw, index_weights]`。
4. 动态配额哨兵与整年边界停机:
   - 实时监测 `rqdatac.user.get_quota()`, 根据实测单股单日压缩流量动态预估下一完整年份所需配额;
   - 若剩余配额不足以完成下一整年, 主动在上一完整自然年年末 (如 2024-12-31) 安全停机;
   - 次日再次执行同一命令即可从 `_state/export_tasks.json` 自动断点续传。

用法:
    python scripts/download_rq_stocks.py
    python scripts/download_rq_stocks.py --status
    python scripts/download_rq_stocks.py --start-year 2010 --end-date 2026-11-07
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
RQSDK_VENV_PYTHON = REPO_ROOT / ".venv-rqsdk" / "bin" / "python"

BENCHMARK_INDICES: tuple[tuple[str, str], ...] = (
    ("000300.XSHG", "2005-04-08"),  # 沪深300
    ("000905.XSHG", "2007-01-15"),  # 中证500
    ("000985.XSHG", "2011-08-02"),  # 中证全指
    ("000852.XSHG", "2014-10-17"),  # 中证1000
)

WEIGHT_INDICES: tuple[tuple[str, str], ...] = (
    ("000300.XSHG", "2005-04-08"),
    ("000905.XSHG", "2007-01-15"),
    ("000852.XSHG", "2014-10-17"),
)

PRICE_FIELDS: list[str] = [
    "open",
    "high",
    "low",
    "close",
    "prev_close",
    "limit_up",
    "limit_down",
    "volume",
    "total_turnover",
    "num_trades",
]

FACTOR_FIELDS: list[str] = [
    "market_cap",
    "a_share_market_val_in_circulation",
    "pe_ratio_ttm",
    "pb_ratio_lf",
    "ps_ratio_ttm",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="米筐 RQData A 股历史日频数据本地化采集与整年断点续传工具"
    )
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=Path("data/raw/stocks"),
        help="股票原始 Parquet 落盘根目录 (默认: data/raw/stocks)",
    )
    parser.add_argument(
        "--start-year",
        type=int,
        default=2010,
        help="起始自然年份 (默认: 2010)",
    )
    parser.add_argument(
        "--end-date",
        type=str,
        default="2026-11-07",
        help="截止日期 YYYY-MM-DD (默认: 2026-11-07)",
    )
    parser.add_argument(
        "--safety-margin-mb",
        type=float,
        default=8.0,
        help="每日流量上限安全缓冲垫 MB (默认: 8.0)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1500,
        help="单批次查询股票数量上限 (默认: 1500)",
    )
    parser.add_argument(
        "--years",
        type=str,
        default=None,
        help="可选指定年份列表, 逗号分隔, 如 '2023,2024'",
    )
    parser.add_argument(
        "--include-bjse",
        action="store_true",
        default=False,
        help="是否包含北交所 (BJSE) 标的 (默认: False, 仅包含沪深主板/创业板/科创板 XSHG/XSHE, 避免无 BJSE 行情权限报错)",
    )
    parser.add_argument(
        "--status",
        action="store_true",
        help="仅查看本地落盘水位线与当前米筐账户配额, 不执行下载",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="强制重下已完成任务",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="日志级别 (默认: INFO)",
    )
    return parser.parse_args()


def _ensure_rqsdk_runtime() -> None:
    """若当前解释器缺少 rqdatac 且存在独立虚拟环境 .venv-rqsdk, 自动切换执行。"""
    try:
        import rqdatac  # noqa: F401

        return
    except ImportError:
        pass

    if (
        RQSDK_VENV_PYTHON.exists()
        and Path(sys.prefix).resolve() != (REPO_ROOT / ".venv-rqsdk").resolve()
        and os.environ.get("_FACTORLIB_RQSDK_REEXEC") != "1"
    ):
        env = os.environ.copy()
        env["_FACTORLIB_RQSDK_REEXEC"] = "1"
        os.execve(
            str(RQSDK_VENV_PYTHON),
            [str(RQSDK_VENV_PYTHON), *sys.argv],
            env,
        )

    raise RuntimeError(
        "当前 Python 环境未安装 rqdatac，且未找到 .venv-rqsdk/bin/python。"
        "请先执行: uv venv .venv-rqsdk --python 3.12 && "
        "uv pip install --python .venv-rqsdk/bin/python rqdatac pyarrow polars tqdm pyyaml"
    )


def _load_dotenv(dotenv_path: Path) -> None:
    """轻量读取仓库根目录 .env 中的键值 (不覆盖已有环境变量)。"""
    if not dotenv_path.exists():
        return
    for raw_line in dotenv_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, val = line.split("=", 1)
        key = key.strip()
        val = val.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = val


def _init_rqdatac(enable_bjse: bool = False) -> Any:
    """安全初始化 rqdatac 并开启 Zstd 压缩。"""
    import rqdatac

    _load_dotenv(REPO_ROOT / ".env")
    license_key = os.environ.get("RQSDK_LICENSE", "").strip()
    conf_uri = os.environ.get("RQDATAC2_CONF") or os.environ.get("RQDATAC_CONF")

    if license_key:
        rqdatac.init("license", license_key, use_zstd=True, enable_bjse=enable_bjse)
    elif conf_uri:
        rqdatac.init(uri=conf_uri, use_zstd=True, enable_bjse=enable_bjse)
    else:
        rqdatac.init(use_zstd=True, enable_bjse=enable_bjse)
    return rqdatac


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()


class StateStore:
    """管理 data/raw/stocks/_state/{export_tasks.json, watermarks.json} 持久化状态。"""

    def __init__(self, raw_dir: Path) -> None:
        self.raw_dir = raw_dir
        self.state_dir = raw_dir / "_state"
        self.tasks_file = self.state_dir / "export_tasks.json"
        self.watermarks_file = self.state_dir / "watermarks.json"
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.tasks_doc = self._read_json(
            self.tasks_file,
            default={"version": 1, "updated_at": _now_iso(), "tasks": {}},
        )
        self.watermarks = self._read_json(
            self.watermarks_file,
            default={
                "market": "stocks",
                "target_start_date": "2010-01-01",
                "target_end_date": "2026-11-07",
                "last_complete_year": None,
                "completed_years": [],
                "max_trading_date": None,
                "last_export": None,
            },
        )

    @staticmethod
    def _read_json(path: Path, default: dict[str, Any]) -> dict[str, Any]:
        if path.exists():
            with path.open("r", encoding="utf-8") as fh:
                return json.load(fh)
        return default

    @staticmethod
    def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
        tmp = path.with_suffix(".json.tmp")
        with tmp.open("w", encoding="utf-8") as fh:
            json.dump(payload, fh, ensure_ascii=False, indent=2, sort_keys=True)
        tmp.replace(path)

    def is_done(self, task_id: str, rel_path: str, force: bool = False) -> bool:
        if force:
            return False
        rec = self.tasks_doc["tasks"].get(task_id)
        if not rec or rec.get("status") != "completed":
            return False
        return (self.raw_dir / rel_path).exists()

    def mark_completed(
        self,
        task_id: str,
        table: str,
        rel_path: str,
        rows: int,
        wire_bytes: int,
        start: str,
        end: str,
        started_at: str,
        extra: dict[str, Any] | None = None,
    ) -> None:
        entry: dict[str, Any] = {
            "task_id": task_id,
            "table": table,
            "path": rel_path,
            "status": "completed",
            "rows": int(rows),
            "wire_bytes": int(max(0, wire_bytes)),
            "start": start,
            "end": end,
            "started_at": started_at,
            "completed_at": _now_iso(),
            "error": None,
        }
        if extra:
            entry.update(extra)
        self.tasks_doc["tasks"][task_id] = entry
        self.tasks_doc["updated_at"] = _now_iso()
        self._atomic_write_json(self.tasks_file, self.tasks_doc)

    def mark_year_complete(self, year: int, max_trading_date: str) -> None:
        completed = sorted({*self.watermarks.get("completed_years", []), int(year)})
        self.watermarks["completed_years"] = completed
        # 计算自起始年起连续完整的最高年份
        contiguous_last: int | None = None
        if completed:
            contiguous_last = completed[0]
            for y in completed[1:]:
                if y == contiguous_last + 1:
                    contiguous_last = y
                else:
                    break
        self.watermarks["last_complete_year"] = contiguous_last
        prev_max = self.watermarks.get("max_trading_date")
        if prev_max is None or max_trading_date > prev_max:
            self.watermarks["max_trading_date"] = max_trading_date
        self.watermarks["last_export"] = _now_iso()
        self._atomic_write_json(self.watermarks_file, self.watermarks)

    def empirical_bytes_per_stock_day(self) -> float | None:
        """从已完整落盘年度任务中统计每(活跃股票×交易日)的平均流量字节数。"""
        complete_set = set(self.watermarks.get("completed_years", []))
        if not complete_set:
            return None
        year_totals: dict[int, int] = {}
        year_stock_days: dict[int, int] = {}
        for tid, rec in self.tasks_doc["tasks"].items():
            if rec.get("status") != "completed" or not tid.startswith("year:"):
                continue
            parts = tid.split(":")
            if len(parts) < 3:
                continue
            y = int(parts[1])
            if y not in complete_set:
                continue
            year_totals[y] = year_totals.get(y, 0) + int(rec.get("wire_bytes", 0))
            sd = int(rec.get("stock_days", 0))
            if sd > 0:
                year_stock_days[y] = sd

        ratios = [
            year_totals[y] / year_stock_days[y]
            for y in sorted(year_totals)
            if y in year_stock_days and year_stock_days[y] > 0 and year_totals[y] > 0
        ]
        if not ratios:
            return None
        return float(ratios[-1])


def _atomic_write_parquet(df: Any, target_path: Path) -> int:
    """使用 Polars Zstd 压缩原子写入 Parquet 并返回行数。"""
    import polars as pl

    target_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = target_path.with_suffix(".parquet.tmp")
    if isinstance(df, pl.DataFrame):
        pl_df = df
    else:
        pl_df = pl.from_pandas(df)
    pl_df.write_parquet(tmp_path, compression="zstd", compression_level=3)
    tmp_path.replace(target_path)
    return len(pl_df)


def _chunked(seq: list[str], size: int) -> list[list[str]]:
    return [seq[i : i + size] for i in range(0, len(seq), size)]


def _quota_snapshot(rqdatac: Any) -> dict[str, Any]:
    q = rqdatac.user.get_quota()
    used = int(q.get("bytes_used", 0))
    limit = int(q.get("bytes_limit", 0))
    rem = (limit - used) if limit > 0 else -1
    return {
        "bytes_used": used,
        "bytes_limit": limit,
        "bytes_remaining": rem,
        "remaining_days": q.get("remaining_days"),
        "license_type": q.get("license_type"),
    }


def _format_mb(num_bytes: float) -> str:
    return f"{num_bytes / (1024.0 * 1024.0):.2f} MB"


def _active_stocks_in_range(
    inst_df: Any, start_str: str, end_str: str, include_bjse: bool = False
) -> list[str]:
    """筛选在 [start_str, end_str] 区间内处于已上市且未在区间开始前退市的股票代码列表。"""
    import polars as pl

    cond = (pl.col("listed_date_clean").is_null() | (pl.col("listed_date_clean") <= end_str)) & (
        pl.col("delisted_date_clean").is_null() | (pl.col("delisted_date_clean") >= start_str)
    )
    if not include_bjse and "exchange" in inst_df.columns:
        cond = cond & (pl.col("exchange") != "BJSE")

    filtered = inst_df.filter(cond)
    return sorted(filtered["order_book_id"].to_list())


def _month_anchor_dates(trading_dates: list[date]) -> list[date]:
    """提取年份内首交易日与每月最后一个交易日作为截面快照锚点。"""
    if not trading_dates:
        return []
    anchors: list[date] = [trading_dates[0]]
    for idx in range(len(trading_dates) - 1):
        curr_d = trading_dates[idx]
        next_d = trading_dates[idx + 1]
        if (curr_d.year, curr_d.month) != (next_d.year, next_d.month):
            anchors.append(curr_d)
    if trading_dates[-1] != anchors[-1]:
        anchors.append(trading_dates[-1])
    return sorted(set(anchors))


def _export_global_metadata(
    rqdatac: Any,
    state: StateStore,
    raw_dir: Path,
    end_date: str,
    batch_size: int,
    force: bool,
) -> tuple[Any, list[date]]:
    """导出全局元数据表: 交易日历、股票基础信息、全历史除权因子、宽基指数日线。"""
    import polars as pl

    # 1. 交易日历 (2005-01-01 ~ 2026-12-31)
    cal_rel = "calendar/trading_calendar.parquet"
    cal_path = raw_dir / cal_rel
    if not state.is_done("global:calendar", cal_rel, force=force):
        t_start = _now_iso()
        b0 = _quota_snapshot(rqdatac)["bytes_used"]
        cal_dates = rqdatac.get_trading_dates("2005-01-01", "2026-12-31")
        cal_df = pl.DataFrame({"trading_date": cal_dates}).with_columns(
            pl.col("trading_date").cast(pl.Date)
        )
        n_rows = _atomic_write_parquet(cal_df, cal_path)
        b1 = _quota_snapshot(rqdatac)["bytes_used"]
        state.mark_completed(
            "global:calendar",
            "calendar",
            cal_rel,
            n_rows,
            b1 - b0,
            "2005-01-01",
            "2026-12-31",
            t_start,
        )
        logging.info("已导出交易日历: %d 个交易日 (流量 %s)", n_rows, _format_mb(b1 - b0))
    else:
        cal_df = pl.read_parquet(cal_path)

    all_trading_dates: list[date] = cal_df["trading_date"].sort().to_list()

    # 2. 全量股票静态元数据 (含历史退市股与北交所)
    inst_rel = "instruments_info/instruments_info.parquet"
    inst_path = raw_dir / inst_rel
    if not state.is_done("global:instruments_info", inst_rel, force=force):
        t_start = _now_iso()
        b0 = _quota_snapshot(rqdatac)["bytes_used"]
        inst_pd = rqdatac.all_instruments(type="CS", market="cn")
        # 统一将对象列安全转为字符串避免混合类型报错
        for col in inst_pd.columns:
            if str(inst_pd[col].dtype) == "object":
                inst_pd[col] = inst_pd[col].astype(str)
        inst_df = (
            pl.from_pandas(inst_pd)
            .with_columns(
                listed_date_clean=pl.when(
                    pl.col("listed_date").is_in(["0000-00-00", "None", "nan", "NaT", ""])
                )
                .then(None)
                .otherwise(pl.col("listed_date").str.slice(0, 10)),
                delisted_date_clean=pl.when(
                    pl.col("de_listed_date").is_in(["0000-00-00", "None", "nan", "NaT", ""])
                )
                .then(None)
                .otherwise(pl.col("de_listed_date").str.slice(0, 10)),
            )
            .sort("order_book_id")
        )
        n_rows = _atomic_write_parquet(inst_df, inst_path)
        b1 = _quota_snapshot(rqdatac)["bytes_used"]
        state.mark_completed(
            "global:instruments_info",
            "instruments_info",
            inst_rel,
            n_rows,
            b1 - b0,
            "1990-01-01",
            end_date,
            t_start,
        )
        logging.info("已导出全量 A 股静态信息: %d 只标的 (流量 %s)", n_rows, _format_mb(b1 - b0))
    else:
        inst_df = pl.read_parquet(inst_path)

    all_ids = sorted(inst_df["order_book_id"].to_list())

    # 3. 全历史除权除息复权因子事件表 (稀疏事件表, 一次拉取全量极省流量)
    exf_rel = "ex_factor/ex_factor.parquet"
    exf_path = raw_dir / exf_rel
    if not state.is_done("global:ex_factor", exf_rel, force=force):
        t_start = _now_iso()
        b0 = _quota_snapshot(rqdatac)["bytes_used"]
        parts: list[pl.DataFrame] = []
        for batch in _chunked(all_ids, batch_size):
            ex_pd = rqdatac.get_ex_factor(batch, start_date="1990-01-01", end_date=end_date)
            if ex_pd is not None and not ex_pd.empty:
                ex_pd = ex_pd.reset_index()
                parts.append(
                    pl.from_pandas(ex_pd).select(
                        pl.col("order_book_id").cast(pl.String),
                        pl.col("ex_date").dt.date().alias("ex_date"),
                        pl.col("ex_factor").cast(pl.Float64),
                        pl.col("ex_cum_factor").cast(pl.Float64),
                    )
                )
        exf_df = (
            pl.concat(parts, how="vertical").sort(["order_book_id", "ex_date"])
            if parts
            else pl.DataFrame(
                schema={
                    "order_book_id": pl.String,
                    "ex_date": pl.Date,
                    "ex_factor": pl.Float64,
                    "ex_cum_factor": pl.Float64,
                }
            )
        )
        n_rows = _atomic_write_parquet(exf_df, exf_path)
        b1 = _quota_snapshot(rqdatac)["bytes_used"]
        state.mark_completed(
            "global:ex_factor",
            "ex_factor",
            exf_rel,
            n_rows,
            b1 - b0,
            "1990-01-01",
            end_date,
            t_start,
        )
        logging.info("已导出除权除息复权因子表: %d 条事件 (流量 %s)", n_rows, _format_mb(b1 - b0))

    # 4. 宽基基准指数日线行情
    idx_rel = "benchmark/index_bar_1d.parquet"
    idx_path = raw_dir / idx_rel
    if not state.is_done("global:index_bar_1d", idx_rel, force=force):
        t_start = _now_iso()
        b0 = _quota_snapshot(rqdatac)["bytes_used"]
        idx_ids = [item[0] for item in BENCHMARK_INDICES]
        idx_pd = rqdatac.get_price(
            idx_ids,
            start_date="2009-01-01",
            end_date=end_date,
            frequency="1d",
            fields=["open", "high", "low", "close", "prev_close", "volume", "total_turnover"],
            adjust_type="none",
            expect_df=True,
        )
        idx_pd = idx_pd.reset_index()
        idx_df = (
            pl.from_pandas(idx_pd)
            .rename({"date": "trading_date"})
            .with_columns(pl.col("trading_date").dt.date())
            .sort(["trading_date", "order_book_id"])
        )
        n_rows = _atomic_write_parquet(idx_df, idx_path)
        b1 = _quota_snapshot(rqdatac)["bytes_used"]
        state.mark_completed(
            "global:index_bar_1d",
            "index_bar_1d",
            idx_rel,
            n_rows,
            b1 - b0,
            "2009-01-01",
            end_date,
            t_start,
        )
        logging.info("已导出宽基基准指数日线: %d 行 (流量 %s)", n_rows, _format_mb(b1 - b0))

    return inst_df, all_trading_dates


def _export_year_bundle(
    rqdatac: Any,
    state: StateStore,
    raw_dir: Path,
    inst_df: Any,
    year: int,
    y_dates: list[date],
    batch_size: int,
    force: bool,
    include_bjse: bool = False,
) -> int:
    """原子导出单个自然年的全部 5 张股票核心日频与截面表, 返回当年总消耗字节。"""
    import numpy as np
    import polars as pl

    y_start = y_dates[0].isoformat()
    y_end = y_dates[-1].isoformat()
    active_ids = _active_stocks_in_range(inst_df, y_start, y_end, include_bjse=include_bjse)
    stock_days = len(active_ids) * len(y_dates)
    year_wire_bytes = 0

    logging.info(
        ">>> 开始同步 %d 年度数据包 [%s ~ %s] | 活跃/存续标的: %d 只 | 交易日: %d 天",
        year,
        y_start,
        y_end,
        len(active_ids),
        len(y_dates),
    )

    # 1. bar_1d: 不复权日线行情
    bar_tid = f"year:{year}:bar_1d"
    bar_rel = f"bar_1d/bar_1d_{year}.parquet"
    if not state.is_done(bar_tid, bar_rel, force=force):
        t_start = _now_iso()
        b0 = _quota_snapshot(rqdatac)["bytes_used"]
        bar_parts: list[pl.DataFrame] = []
        for batch in _chunked(active_ids, batch_size):
            px_pd = rqdatac.get_price(
                batch,
                start_date=y_start,
                end_date=y_end,
                frequency="1d",
                fields=PRICE_FIELDS,
                adjust_type="none",
                skip_suspended=True,
                expect_df=True,
            )
            if px_pd is not None and not px_pd.empty:
                px_pd = px_pd.reset_index()
                bar_parts.append(
                    pl.from_pandas(px_pd)
                    .rename({"date": "trading_date"})
                    .with_columns(pl.col("trading_date").dt.date())
                )
        bar_df = (
            pl.concat(bar_parts, how="vertical").sort(["trading_date", "order_book_id"])
            if bar_parts
            else pl.DataFrame()
        )
        n_rows = _atomic_write_parquet(bar_df, raw_dir / bar_rel)
        b1 = _quota_snapshot(rqdatac)["bytes_used"]
        used = b1 - b0
        year_wire_bytes += used
        state.mark_completed(
            bar_tid,
            "bar_1d",
            bar_rel,
            n_rows,
            used,
            y_start,
            y_end,
            t_start,
            extra={"stock_days": stock_days},
        )
        logging.info("  [%d] bar_1d 完成: %d 行 (流量 %s)", year, n_rows, _format_mb(used))

    # 2. trade_status: ST 与停牌真实历史掩码
    ts_tid = f"year:{year}:trade_status"
    ts_rel = f"trade_status/trade_status_{year}.parquet"
    if not state.is_done(ts_tid, ts_rel, force=force):
        t_start = _now_iso()
        b0 = _quota_snapshot(rqdatac)["bytes_used"]
        ts_parts: list[pl.DataFrame] = []
        for batch in _chunked(active_ids, batch_size):
            st_pd = rqdatac.is_st_stock(batch, start_date=y_start, end_date=y_end)
            sp_pd = rqdatac.is_suspended(batch, start_date=y_start, end_date=y_end)
            if st_pd is not None and not st_pd.empty and sp_pd is not None and not sp_pd.empty:
                # 对齐行列并向量化展平为长表
                common_dates = st_pd.index.intersection(sp_pd.index)
                common_cols = st_pd.columns.intersection(sp_pd.columns)
                st_sub = st_pd.loc[common_dates, common_cols]
                sp_sub = sp_pd.loc[common_dates, common_cols]
                t_len, n_len = st_sub.shape
                dates_rep = np.repeat(common_dates.to_numpy(dtype="datetime64[us]"), n_len)
                syms_tile = np.tile(np.asarray(common_cols, dtype=str), t_len)
                st_flat = st_sub.to_numpy(dtype=bool).ravel()
                sp_flat = sp_sub.to_numpy(dtype=bool).ravel()
                ts_parts.append(
                    pl.DataFrame(
                        {
                            "trading_date": dates_rep,
                            "order_book_id": syms_tile,
                            "is_st": st_flat,
                            "is_suspended": sp_flat,
                        }
                    ).with_columns(pl.col("trading_date").dt.date())
                )
        ts_df = (
            pl.concat(ts_parts, how="vertical").sort(["trading_date", "order_book_id"])
            if ts_parts
            else pl.DataFrame()
        )
        n_rows = _atomic_write_parquet(ts_df, raw_dir / ts_rel)
        b1 = _quota_snapshot(rqdatac)["bytes_used"]
        used = b1 - b0
        year_wire_bytes += used
        state.mark_completed(
            ts_tid,
            "trade_status",
            ts_rel,
            n_rows,
            used,
            y_start,
            y_end,
            t_start,
            extra={"stock_days": stock_days},
        )
        logging.info("  [%d] trade_status 完成: %d 行 (流量 %s)", year, n_rows, _format_mb(used))

    # 3. daily_basic: 每日换手率 + 总市值/流通市值/PE/PB/PS 估值
    db_tid = f"year:{year}:daily_basic"
    db_rel = f"daily_basic/daily_basic_{year}.parquet"
    if not state.is_done(db_tid, db_rel, force=force):
        t_start = _now_iso()
        b0 = _quota_snapshot(rqdatac)["bytes_used"]
        tr_parts: list[pl.DataFrame] = []
        fac_parts: list[pl.DataFrame] = []
        for batch in _chunked(active_ids, batch_size):
            tr_pd = rqdatac.get_turnover_rate(
                batch,
                start_date=y_start,
                end_date=y_end,
                fields=["today"],
                expect_df=True,
            )
            if tr_pd is not None and not tr_pd.empty:
                tr_pd = tr_pd.reset_index()
                date_col = "tradedate" if "tradedate" in tr_pd.columns else "date"
                tr_parts.append(
                    pl.from_pandas(tr_pd)
                    .rename({date_col: "trading_date", "today": "turnover_rate"})
                    .with_columns(
                        pl.col("trading_date").dt.date(),
                        pl.col("turnover_rate").cast(pl.Float64),
                    )
                    .select(["trading_date", "order_book_id", "turnover_rate"])
                )

            fac_pd = rqdatac.get_factor(
                batch,
                FACTOR_FIELDS,
                start_date=y_start,
                end_date=y_end,
                expect_df=True,
            )
            if fac_pd is not None and not fac_pd.empty:
                fac_pd = fac_pd.reset_index()
                fac_parts.append(
                    pl.from_pandas(fac_pd)
                    .rename({"date": "trading_date"})
                    .with_columns(
                        pl.col("trading_date").dt.date(),
                        *[pl.col(c).cast(pl.Float64) for c in FACTOR_FIELDS if c in fac_pd.columns],
                    )
                )

        tr_df = pl.concat(tr_parts, how="vertical") if tr_parts else None
        fac_df = pl.concat(fac_parts, how="vertical") if fac_parts else None
        if fac_df is not None and tr_df is not None:
            db_df = fac_df.join(
                tr_df, on=["trading_date", "order_book_id"], how="full", coalesce=True
            ).sort(["trading_date", "order_book_id"])
        elif fac_df is not None:
            db_df = fac_df.with_columns(turnover_rate=pl.lit(None, dtype=pl.Float64)).sort(
                ["trading_date", "order_book_id"]
            )
        elif tr_df is not None:
            db_df = tr_df.sort(["trading_date", "order_book_id"])
        else:
            db_df = pl.DataFrame()

        n_rows = _atomic_write_parquet(db_df, raw_dir / db_rel)
        b1 = _quota_snapshot(rqdatac)["bytes_used"]
        used = b1 - b0
        year_wire_bytes += used
        state.mark_completed(
            db_tid,
            "daily_basic",
            db_rel,
            n_rows,
            used,
            y_start,
            y_end,
            t_start,
            extra={"stock_days": stock_days},
        )
        logging.info("  [%d] daily_basic 完成: %d 行 (流量 %s)", year, n_rows, _format_mb(used))

    # 4. industry_sw: 申万 2021 一/二/三级历史月度锚点截面映射
    ind_tid = f"year:{year}:industry_sw"
    ind_rel = f"industry_sw/industry_sw_{year}.parquet"
    anchors = _month_anchor_dates(y_dates)
    if not state.is_done(ind_tid, ind_rel, force=force):
        t_start = _now_iso()
        b0 = _quota_snapshot(rqdatac)["bytes_used"]
        ind_parts: list[pl.DataFrame] = []
        for anchor_dt in anchors:
            dt_str = anchor_dt.isoformat()
            snap_ids = _active_stocks_in_range(inst_df, dt_str, dt_str, include_bjse=include_bjse)
            if not snap_ids:
                continue
            ind_pd = rqdatac.get_instrument_industry(
                snap_ids, source="sws_2021", level=0, date=dt_str
            )
            if ind_pd is not None and not ind_pd.empty:
                ind_pd = ind_pd.reset_index()
                for c in ind_pd.columns:
                    if str(ind_pd[c].dtype) == "object":
                        ind_pd[c] = ind_pd[c].astype(str)
                ind_parts.append(
                    pl.from_pandas(ind_pd).with_columns(
                        snapshot_date=pl.lit(anchor_dt, dtype=pl.Date)
                    )
                )
        ind_df = (
            pl.concat(ind_parts, how="vertical").sort(["snapshot_date", "order_book_id"])
            if ind_parts
            else pl.DataFrame()
        )
        n_rows = _atomic_write_parquet(ind_df, raw_dir / ind_rel)
        b1 = _quota_snapshot(rqdatac)["bytes_used"]
        used = b1 - b0
        year_wire_bytes += used
        state.mark_completed(
            ind_tid,
            "industry_sw",
            ind_rel,
            n_rows,
            used,
            y_start,
            y_end,
            t_start,
            extra={"stock_days": stock_days},
        )
        logging.info("  [%d] industry_sw 完成: %d 行 (流量 %s)", year, n_rows, _format_mb(used))

    # 5. index_weights: 沪深300 / 中证500 / 中证1000 月度历史成分权重
    iw_tid = f"year:{year}:index_weights"
    iw_rel = f"benchmark/index_weights_{year}.parquet"
    if not state.is_done(iw_tid, iw_rel, force=force):
        t_start = _now_iso()
        b0 = _quota_snapshot(rqdatac)["bytes_used"]
        iw_parts: list[pl.DataFrame] = []
        for anchor_dt in anchors:
            dt_str = anchor_dt.isoformat()
            for idx_id, listed_str in WEIGHT_INDICES:
                if dt_str < listed_str:
                    continue
                ser = rqdatac.index_weights(idx_id, date=dt_str)
                if ser is not None and len(ser) > 0:
                    iw_parts.append(
                        pl.DataFrame(
                            {
                                "snapshot_date": [anchor_dt] * len(ser),
                                "index_id": [idx_id] * len(ser),
                                "order_book_id": [str(k) for k in ser.index.tolist()],
                                "weight": [float(v) for v in ser.to_numpy()],
                            },
                            schema={
                                "snapshot_date": pl.Date,
                                "index_id": pl.String,
                                "order_book_id": pl.String,
                                "weight": pl.Float64,
                            },
                        )
                    )
        iw_df = (
            pl.concat(iw_parts, how="vertical").sort(["snapshot_date", "index_id", "order_book_id"])
            if iw_parts
            else pl.DataFrame()
        )
        n_rows = _atomic_write_parquet(iw_df, raw_dir / iw_rel)
        b1 = _quota_snapshot(rqdatac)["bytes_used"]
        used = b1 - b0
        year_wire_bytes += used
        state.mark_completed(
            iw_tid,
            "index_weights",
            iw_rel,
            n_rows,
            used,
            y_start,
            y_end,
            t_start,
            extra={"stock_days": stock_days},
        )
        logging.info("  [%d] index_weights 完成: %d 行 (流量 %s)", year, n_rows, _format_mb(used))

    state.mark_year_complete(year, y_end)
    q_now = _quota_snapshot(rqdatac)
    rem_str = _format_mb(q_now["bytes_remaining"]) if q_now["bytes_limit"] > 0 else "不限"
    logging.info(
        "<<< [%d 年度完整落盘] 本年新增流量: %s | 今日累计已用: %s | 今日剩余可用: %s",
        year,
        _format_mb(year_wire_bytes),
        _format_mb(q_now["bytes_used"]),
        rem_str,
    )
    return year_wire_bytes


def _print_status_summary(raw_dir: Path, state: StateStore, q: dict[str, Any]) -> None:
    total_disk = sum(f.stat().st_size for f in raw_dir.rglob("*.parquet") if f.is_file())
    limit_str = _format_mb(q["bytes_limit"]) if q["bytes_limit"] > 0 else "不受限"
    rem_str = _format_mb(q["bytes_remaining"]) if q["bytes_limit"] > 0 else "不受限"
    print("\n" + "=" * 68)
    print("  FactorLibrary 米筐股票日频数据仓库状态与配额摘要")
    print("=" * 68)
    print(f"  许可证类型:         {q['license_type']} (剩余有效期: {q['remaining_days']} 天)")
    print(f"  今日流量已用/上限:  {_format_mb(q['bytes_used'])} / {limit_str} (剩余: {rem_str})")
    print(f"  本地原始目录:       {raw_dir.resolve()}")
    print(f"  Parquet 落盘总计:   {_format_mb(total_disk)}")
    print(f"  已完成年份列表:     {state.watermarks.get('completed_years', [])}")
    print(f"  连续完整年份水位:   {state.watermarks.get('last_complete_year')}")
    print(f"  最新覆盖交易日:     {state.watermarks.get('max_trading_date')}")
    print("=" * 68 + "\n")


def main() -> int:
    args = parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="[%(asctime)s] [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    _ensure_rqsdk_runtime()
    rqdatac = _init_rqdatac(enable_bjse=args.include_bjse)
    from rqdatac.share.errors import QuotaExceeded

    raw_dir = args.raw_dir if args.raw_dir.is_absolute() else (REPO_ROOT / args.raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    state = StateStore(raw_dir)
    end_dt = date.fromisoformat(args.end_date)
    state.watermarks["target_start_date"] = f"{args.start_year}-01-01"
    state.watermarks["target_end_date"] = end_dt.isoformat()

    q_init = _quota_snapshot(rqdatac)
    if args.status:
        _print_status_summary(raw_dir, state, q_init)
        return 0

    logging.info(
        "RQData 连接成功 | 账户类型=%s | 剩余天数=%s | 今日剩余流量=%s / %s",
        q_init["license_type"],
        q_init["remaining_days"],
        _format_mb(q_init["bytes_remaining"]) if q_init["bytes_limit"] > 0 else "不限",
        _format_mb(q_init["bytes_limit"]) if q_init["bytes_limit"] > 0 else "不限",
    )

    t0 = time.time()
    safety_bytes = int(args.safety_margin_mb * 1024 * 1024)

    try:
        inst_df, all_cal_dates = _export_global_metadata(
            rqdatac=rqdatac,
            state=state,
            raw_dir=raw_dir,
            end_date=end_dt.isoformat(),
            batch_size=args.batch_size,
            force=args.force,
        )

        if args.years:
            target_years = sorted({int(x.strip()) for x in args.years.split(",") if x.strip()})
        else:
            target_years = list(range(args.start_year, end_dt.year + 1))

        for year in target_years:
            y_dates = [d for d in all_cal_dates if d.year == year and d <= end_dt]
            if not y_dates:
                continue

            # 检查该年 5 个子表是否已全部完成
            sub_tables = ["bar_1d", "trade_status", "daily_basic", "industry_sw", "index_weights"]
            rel_map = {
                "bar_1d": f"bar_1d/bar_1d_{year}.parquet",
                "trade_status": f"trade_status/trade_status_{year}.parquet",
                "daily_basic": f"daily_basic/daily_basic_{year}.parquet",
                "industry_sw": f"industry_sw/industry_sw_{year}.parquet",
                "index_weights": f"benchmark/index_weights_{year}.parquet",
            }
            if not args.force and all(
                state.is_done(f"year:{year}:{tb}", rel_map[tb], force=False) for tb in sub_tables
            ):
                state.mark_year_complete(year, y_dates[-1].isoformat())
                logging.info("跳过已完成年度: %d (%s ~ %s)", year, y_dates[0], y_dates[-1])
                continue

            # 进入新年度前执行动态流量配额预判, 确保停在完整自然年边界
            q_curr = _quota_snapshot(rqdatac)
            if q_curr["bytes_limit"] > 0:
                active_ids_y = _active_stocks_in_range(
                    inst_df,
                    y_dates[0].isoformat(),
                    y_dates[-1].isoformat(),
                    include_bjse=args.include_bjse,
                )
                stock_days_y = len(active_ids_y) * len(y_dates)
                bpsd = max(state.empirical_bytes_per_stock_day() or 78.0, 72.0)
                est_bytes = int(bpsd * stock_days_y * 1.08)

                if q_curr["bytes_remaining"] < (est_bytes + safety_bytes):
                    logging.warning(
                        "触发整年边界流量保护! 剩余配额 %s 小于 %d 年度预计所需 %s + 安全垫 %s。"
                        "已完整保留截至 %s 年末的数据，明日再次执行 `python scripts/download_rq_stocks.py` 即可自动续传!",
                        _format_mb(q_curr["bytes_remaining"]),
                        year,
                        _format_mb(est_bytes),
                        _format_mb(safety_bytes),
                        state.watermarks.get("last_complete_year") or (year - 1),
                    )
                    break

            _export_year_bundle(
                rqdatac=rqdatac,
                state=state,
                raw_dir=raw_dir,
                inst_df=inst_df,
                year=year,
                y_dates=y_dates,
                batch_size=args.batch_size,
                force=args.force,
                include_bjse=args.include_bjse,
            )

    except QuotaExceeded as exc:
        logging.warning(
            "收到米筐服务端当日流量限额通知 (%s)。已完成的所有完整年份均已安全落盘，"
            "当前连续完整年份水位线为: %s。明日再次运行同一命令即可无缝续传!",
            exc,
            state.watermarks.get("last_complete_year"),
        )

    elapsed = time.time() - t0
    q_final = _quota_snapshot(rqdatac)
    logging.info("本次同步任务结束，总耗时 %.1f 秒。", elapsed)
    _print_status_summary(raw_dir, state, q_final)
    return 0


if __name__ == "__main__":
    sys.exit(main())
