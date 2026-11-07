"""米筐股票日频采集器断点状态机与防幸存者偏差辅助逻辑测试 (tests/python/test_download_rq_stocks.py)。"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import polars as pl

from scripts.download_rq_stocks import (
    StateStore,
    _active_stocks_in_range,
    _month_anchor_dates,
)


def test_state_store_contiguous_watermark_and_bpsd(tmp_path: Path) -> None:
    """验证整年水位线连续性判定与单股单日流量实测估算。"""
    store = StateStore(tmp_path)
    (tmp_path / "bar_1d").mkdir(parents=True)
    dummy_file = tmp_path / "bar_1d" / "bar_1d_2010.parquet"
    dummy_file.write_bytes(b"parquet")

    store.mark_completed(
        task_id="year:2010:bar_1d",
        table="bar_1d",
        rel_path="bar_1d/bar_1d_2010.parquet",
        rows=1000,
        wire_bytes=80_000,
        start="2010-01-04",
        end="2010-12-31",
        started_at="2026-11-07T00:00:00+00:00",
        extra={"stock_days": 1000},
    )
    assert store.is_done("year:2010:bar_1d", "bar_1d/bar_1d_2010.parquet")
    # 未标记整年完成前不应纳入完整年份流量统计
    assert store.empirical_bytes_per_stock_day() is None

    store.mark_year_complete(2010, "2010-12-31")
    assert store.watermarks["last_complete_year"] == 2010
    assert store.empirical_bytes_per_stock_day() == 80.0

    # 若跳年完成 2012, 连续完整年份水位仍应锁定在 2010
    store.mark_year_complete(2012, "2012-12-31")
    assert store.watermarks["completed_years"] == [2010, 2012]
    assert store.watermarks["last_complete_year"] == 2010

    # 补齐 2011 后, 连续完整年份自动推进至 2012
    store.mark_year_complete(2011, "2011-12-30")
    assert store.watermarks["last_complete_year"] == 2012


def test_active_stocks_includes_delisted_without_survivorship_bias() -> None:
    """验证年度股票池过滤严格保留当年尚未退市的历史退市股, 剔除未上市与已提前退市标的。"""
    inst_df = pl.DataFrame(
        {
            "order_book_id": [
                "000001.XSHE",  # 长期存续
                "600001.XSHG",  # 2009 已退市 -> 2010 不应查询
                "600002.XSHG",  # 2010 年中退市 -> 2010 必须保留防幸存者偏差!
                "301999.XSHE",  # 2023 上市 -> 2010 不应查询
            ],
            "listed_date_clean": ["1991-04-03", "1998-01-01", "2001-05-01", "2023-06-01"],
            "delisted_date_clean": [None, "2009-11-15", "2010-06-30", None],
        }
    )
    active_2010 = _active_stocks_in_range(inst_df, "2010-01-04", "2010-12-31")
    assert active_2010 == ["000001.XSHE", "600002.XSHG"]


def test_month_anchor_dates() -> None:
    """验证首交易日与月末交易日截面锚点提取。"""
    dates = [
        date(2024, 1, 2),
        date(2024, 1, 31),
        date(2024, 2, 1),
        date(2024, 2, 29),
        date(2024, 3, 1),
        date(2024, 3, 15),
    ]
    anchors = _month_anchor_dates(dates)
    assert anchors == [
        date(2024, 1, 2),
        date(2024, 1, 31),
        date(2024, 2, 29),
        date(2024, 3, 15),
    ]
