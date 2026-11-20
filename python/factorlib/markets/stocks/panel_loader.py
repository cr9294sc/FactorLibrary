"""A 股标准面板加载器与截面数据容器 (markets/stocks/panel_loader.py)。

按白皮书第 17 节与审查重构计划 (P0-11, P1-07) 设计:
- 严格遵循因果无前视原则加载 A 股原始 Parquet 行情 (bar_1d, trade_status, instruments_info, ex_factor, industry_sw, benchmark)；
- 构造严格无日内前视的执行价代理 (默认 T+1 复权开盘价 open_adj, 修复全天 VWAP 吞噬 37% 反转收益缺陷)；
- 规范生成基础股票池掩码 BaseUnivMask (非停牌、非 ST/*ST、上市存续期) 与涨跌停掩码 LimitMask (封死涨跌停与一字板不可交易)；
- 按照自然日精确计算 StockListedDays，为 dropNewStock 算子提供无漂移时间轴；
- 封装为 StockPanelData 容器，向下兼容对账脚本字典访问，并提供 to_panel() 导出标准 core.Panel 实例。
"""

from __future__ import annotations

import glob
import json
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Any

import factorlib_core
import numpy as np
import polars as pl

from factorlib.core.config import StockDataConfig
from factorlib.core.panel import Panel

STOCK_AUX_BACKTEST_FIELDS: frozenset[str] = frozenset(
    {
        "open_raw",
        "close_raw",
        "pre_close_raw",
        "high_raw",
        "low_raw",
        "vwap_open_30m_raw",
        "ex_cum_factor",
        "limit_up",
        "limit_down",
        "is_suspended",
        "is_st",
        "can_buy_mask",
        "can_sell_mask",
        "bench_fwd_open_1",
        "bench_fwd_close_1",
        "bench_ret",
        "bench_c2c",
    }
)

_RAW_TO_AUX_ALIAS: dict[str, str] = {
    "open": "open_raw",
    "close": "close_raw",
    "high": "high_raw",
    "low": "low_raw",
}


@dataclass
class StockPanelData:
    """A 股截面研究面板数据容器。

    Attributes:
        dates: 交易日历列表 (格式通常为 YYYYMMDD 或 YYYY-MM-DD 字符串)
        symbols: 股票标的代码列表 (格式如 '000001.XSHE', '600519.XSHG')
        fields: 核心行情矩阵字典 (T x N, order='F')，包含 close_hfq, open_adj, high_adj, low_adj, vwap_adj, amount, log_amount, volume, c2c_ret
        base_univ_mask: 基础股票池布尔矩阵 (T x N)，剔除停牌、ST/*ST 与未上市/退市
        limit_mask: 涨跌停与一字板不可交易掩码矩阵 (T x N)，可交易为 1.0，封死/一字板为 NaN
        is_suspended: 停牌布尔矩阵 (T x N)
        is_st: 风险警示 ST/*ST 布尔矩阵 (T x N)
        stock_listed_days: 自然日上市天数矩阵 (T x N)
        industry_code: 申万一级行业编码数值矩阵 (T x N)
        bench_fwd_open_1: 基准指数 [T+1, T+2] 远期开盘价收益率向量 (T,)
        bench_fwd_close_1: 基准指数 [T+1, T+2] 远期收盘价收益率向量 (T,)
        bench_ret: 当前回测价格时钟对齐的基准远期收益率向量 (T,)
        meta: 加载与环境元数据字典
    """

    dates: list[str]
    symbols: list[str]
    fields: dict[str, np.ndarray]
    base_univ_mask: np.ndarray
    limit_mask: np.ndarray
    is_suspended: np.ndarray
    is_st: np.ndarray
    stock_listed_days: np.ndarray
    industry_code: np.ndarray
    bench_fwd_open_1: np.ndarray
    bench_fwd_close_1: np.ndarray
    bench_ret: np.ndarray
    can_buy_mask: np.ndarray | None = None
    can_sell_mask: np.ndarray | None = None
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def shape(self) -> tuple[int, int]:
        """(T, N): 交易日天数 x 标的只数。"""
        return len(self.dates), len(self.symbols)

    @property
    def T(self) -> int:
        return len(self.dates)

    @property
    def N(self) -> int:
        return len(self.symbols)

    def to_panel(self, horizons: Sequence[int] = (1, 2, 5, 10)) -> Panel:
        """导出为统一的 core.Panel 标准内存容器（含 Rust labels 与物理隔离 aux_fields）。"""
        T, N = self.shape
        panel_fields: dict[str, np.ndarray] = {}
        aux_fields: dict[str, np.ndarray] = {}

        for k, v in self.fields.items():
            if k in _RAW_TO_AUX_ALIAS:
                aux_fields[_RAW_TO_AUX_ALIAS[k]] = np.asfortranarray(v, dtype=np.float64)
            elif k in STOCK_AUX_BACKTEST_FIELDS:
                aux_fields[k] = np.asfortranarray(v, dtype=np.float64)
            else:
                panel_fields[k] = np.asfortranarray(v, dtype=np.float64)

        listed_arr = np.asfortranarray(self.stock_listed_days, dtype=np.float64)
        ind_arr = np.asfortranarray(self.industry_code, dtype=np.float64)
        panel_fields["stock_listed_days"] = listed_arr
        panel_fields["listed_days"] = listed_arr
        panel_fields["industry_code"] = ind_arr
        panel_fields["sector"] = ind_arr
        panel_fields["limit_mask"] = np.asfortranarray(self.limit_mask, dtype=np.float64)
        panel_fields["in_universe_mask"] = np.asfortranarray(
            np.where(self.base_univ_mask, 1.0, np.nan), dtype=np.float64
        )

        # 回测与基准辅助字段隔离至 aux_fields，严禁进入 fields
        aux_fields["is_suspended"] = np.asfortranarray(self.is_suspended, dtype=np.float64)
        aux_fields["is_st"] = np.asfortranarray(self.is_st, dtype=np.float64)
        if self.can_buy_mask is not None:
            aux_fields["can_buy_mask"] = np.asfortranarray(self.can_buy_mask, dtype=np.float64)
        else:
            aux_fields["can_buy_mask"] = np.asfortranarray(
                np.isfinite(self.limit_mask), dtype=np.float64
            )
        if self.can_sell_mask is not None:
            aux_fields["can_sell_mask"] = np.asfortranarray(self.can_sell_mask, dtype=np.float64)
        else:
            aux_fields["can_sell_mask"] = np.asfortranarray(
                np.isfinite(self.limit_mask), dtype=np.float64
            )

        for b_name, b_vec in (
            ("bench_fwd_open_1", self.bench_fwd_open_1),
            ("bench_fwd_close_1", self.bench_fwd_close_1),
            ("bench_ret", self.bench_ret),
        ):
            vec = np.asarray(b_vec, dtype=np.float64).reshape(T, 1)
            aux_fields[b_name] = np.asfortranarray(np.broadcast_to(vec, (T, N)).copy())

        # 调用 Rust 统一标签引擎构造预测标签
        labels: dict[str, np.ndarray] = {}
        pref_mode = str(self.meta.get("exec_price_mode", "open_adj"))
        price_mode = next(
            (
                m
                for m in (pref_mode, "open_adj", "vwap_adj", "close_hfq", "close")
                if m in self.fields
            ),
            None,
        )
        if price_mode is not None and len(horizons) > 0:
            fwd_map = self.compute_forward_returns(
                horizons=list(horizons), exec_price_mode=price_mode
            )
            for h, mat in fwd_map.items():
                labels[f"y_{h}"] = np.asfortranarray(mat, dtype=np.float64)
            if 1 in fwd_map:
                labels["r_exec_next"] = np.asfortranarray(fwd_map[1], dtype=np.float64)

        dates_arr = np.array(self.dates)
        meta = dict(self.meta)
        meta["market"] = "stocks"

        return Panel(
            dates=dates_arr,
            symbols=list(self.symbols),
            fields=panel_fields,
            mask=np.asfortranarray(self.base_univ_mask, dtype=bool),
            labels=labels,
            aux_fields=aux_fields,
            meta=meta,
        )

    def compute_forward_returns(
        self,
        horizons: list[int] | tuple[int, ...] = (1, 2, 5, 10),
        exec_price_mode: str = "open_adj",
        execution_simulation: bool = False,
    ) -> dict[int, np.ndarray]:
        """调用 Rust `build_forward_labels` 统一计算各预测周期的远期收益率矩阵 (T x N)。

        对齐语义:
        - 信号生成时点: T 日 15:00 收盘
        - 入场价格: T+1 日早盘执行基准 (如 open_adj)
        - 出场价格: T+1+h 日早盘执行基准 (如 open_adj)
        - 远期收益: R_{T, h} = price[T + 1 + h] / price[T + 1] - 1.0
        """
        price_mat = self.fields.get(exec_price_mode)
        if price_mat is None:
            raise KeyError(f"未找到指定的执行价矩阵: {exec_price_mode!r}")

        T, N = self.shape
        vol_mat = self.fields.get("volume")
        vol_ok = (vol_mat > 0.0) if vol_mat is not None else np.ones((T, N), dtype=bool)
        price_ok = np.isfinite(price_mat) & (price_mat > 0.0)

        if execution_simulation:
            buy_ok = (
                self.can_buy_mask if self.can_buy_mask is not None else np.isfinite(self.limit_mask)
            )
            sell_ok = (
                self.can_sell_mask
                if self.can_sell_mask is not None
                else np.isfinite(self.limit_mask)
            )
            entry_ok = np.asfortranarray(
                buy_ok & (~self.is_suspended) & vol_ok & price_ok, dtype=bool
            )
            exit_ok = np.asfortranarray(
                sell_ok & (~self.is_suspended) & vol_ok & price_ok, dtype=bool
            )
        else:
            entry_ok = np.asfortranarray((~self.is_suspended) & vol_ok & price_ok, dtype=bool)
            exit_ok = np.asfortranarray(price_ok, dtype=bool)

        f_price = np.asfortranarray(price_mat, dtype=np.float64)
        h_list = [int(h) for h in horizons]
        rust_out = factorlib_core.build_forward_labels(
            f_price,
            f_price,
            entry_ok,
            exit_ok,
            h_list,
            strict=not execution_simulation,
        )
        return {h: np.asarray(rust_out[h]) for h in h_list}

    # 字典兼容接口，确保现有回测脚本直接适配
    def __getitem__(self, key: str) -> Any:
        if key in self.fields:
            return self.fields[key]
        if hasattr(self, key):
            return getattr(self, key)
        if key == "listed_days":
            return self.stock_listed_days
        raise KeyError(f"StockPanelData 未找到键: {key!r}")

    def get(self, key: str, default: Any = None) -> Any:
        try:
            return self[key]
        except KeyError:
            return default

    def keys(self) -> list[str]:
        base_keys = [
            "dates",
            "symbols",
            "base_univ_mask",
            "limit_mask",
            "is_suspended",
            "is_st",
            "stock_listed_days",
            "listed_days",
            "industry_code",
            "bench_fwd_open_1",
            "bench_fwd_close_1",
            "bench_ret",
        ]
        return list(set(base_keys + list(self.fields.keys())))


class StockPanelLoader:
    """A 股全市场标准 Parquet 行情加载器与面板构建器。"""

    def __init__(self, raw_dir: Path | str = "data/raw/stocks") -> None:
        self.raw_dir = Path(raw_dir)
        if not self.raw_dir.exists():
            raise FileNotFoundError(f"A 股原始数据根目录不存在: {self.raw_dir.resolve()}")

    def _build_extended_fields(
        self,
        *,
        dates: list[date],
        symbols: list[str],
        date_map: dict[date, int],
        sym_map: dict[str, int],
        start_d: date,
        end_d: date,
        close_hfq: np.ndarray,
        open_adj: np.ndarray,
        amount_mat: np.ndarray,
        num_trades_mat: np.ndarray,
        prev_close_mat: np.ndarray,
        close_mat: np.ndarray,
        industry_code: np.ndarray,
        T: int,
        N: int,
    ) -> dict[str, np.ndarray]:
        """构造 FactorLab MCP 1d 全量变量对齐的扩充字段（`stocks/v2`）。

        严格因果：所有字段均为 T 日收盘（15:00）已知的量价/估值/成分权重快照，
        不含任何前视信息；`daily_derived_returns_intra` 使用同一复权基准的价格比值。
        """
        out: dict[str, np.ndarray] = {}

        # a) daily_basic 估值与换手字段
        db_dir = self.raw_dir / "daily_basic"
        db_files = sorted(glob.glob(str(db_dir / "daily_basic_*.parquet")))
        if db_files:
            db_lf = pl.scan_parquet(db_files)
            db_present = set(db_lf.collect_schema().names())
            db_wanted = [
                "trading_date",
                "order_book_id",
                "market_cap",
                "a_share_market_val_in_circulation",
                "pe_ratio_ttm",
                "pb_ratio_lf",
                "ps_ratio_ttm",
                "turnover_rate",
            ]
            db_use = [c for c in db_wanted if c in db_present]
            df_db = (
                db_lf.filter(
                    (pl.col("trading_date") >= start_d) & (pl.col("trading_date") <= end_d)
                )
                .select(db_use)
                .collect()
            )
            db_cols = {
                "mkt_cap": "market_cap",
                "circ_mv": "a_share_market_val_in_circulation",
                "pe_ratio_ttm": "pe_ratio_ttm",
                "pb_ratio_lf": "pb_ratio_lf",
                "ps_ratio_ttm": "ps_ratio_ttm",
                "turnover_rate": "turnover_rate",
            }
            for local_name, src_col in db_cols.items():
                out[local_name] = self._pivot_column(df_db, src_col, date_map, sym_map, T, N)
            with np.errstate(divide="ignore", invalid="ignore"):
                out["log_mkt_cap"] = np.log(np.where(out["mkt_cap"] > 0.0, out["mkt_cap"], np.nan))
                out["log_circ_mv"] = np.log(np.where(out["circ_mv"] > 0.0, out["circ_mv"], np.nan))
                out["log_mkt_cap"] = np.asfortranarray(out["log_mkt_cap"], dtype=np.float64)
                out["log_circ_mv"] = np.asfortranarray(out["log_circ_mv"], dtype=np.float64)

        # b) 成交笔数与平均单笔成交额
        if np.isfinite(num_trades_mat).any():
            out["num_trades"] = np.asfortranarray(num_trades_mat, dtype=np.float64)
            with np.errstate(divide="ignore", invalid="ignore"):
                avg_size = amount_mat / np.where(num_trades_mat > 0.0, num_trades_mat, np.nan)
                out["avg_trade_size"] = np.asfortranarray(avg_size, dtype=np.float64)
                out["log_num_trades"] = np.asfortranarray(
                    np.log(np.where(num_trades_mat > 0.0, num_trades_mat, np.nan)),
                    dtype=np.float64,
                )

        # b2) 除权参考前收盘价与涨跌额（供 daily_kline_real_pre_close / daily_kline_change 别名）
        if np.isfinite(prev_close_mat).any():
            out["pre_close"] = np.asfortranarray(prev_close_mat, dtype=np.float64)
            out["close_change"] = np.asfortranarray(close_mat - prev_close_mat, dtype=np.float64)

        # c) 日内收益率（收盘 / 开盘 - 1，复权比值不变）
        with np.errstate(divide="ignore", invalid="ignore"):
            intra = close_hfq / np.where(open_adj > 0.0, open_adj, np.nan) - 1.0
        out["daily_derived_returns_intra"] = np.asfortranarray(intra, dtype=np.float64)

        # d) 三大指数成分权重与成分池掩码 (000300 / 000905 / 000852)
        iw_dir = self.raw_dir / "benchmark"
        iw_files = sorted(glob.glob(str(iw_dir / "index_weights_*.parquet")))
        if iw_files:
            iw_lf = pl.scan_parquet(iw_files)
            iw_present = set(iw_lf.collect_schema().names())
            iw_wanted = ["snapshot_date", "index_id", "order_book_id", "weight"]
            df_iw = iw_lf.select([c for c in iw_wanted if c in iw_present]).collect()
            index_map = {
                "000300.XSHG": ("index_weight_000300_sh", "trade_universe_hs300"),
                "000905.XSHG": ("index_weight_000905_sh", "trade_universe_zz500"),
                "000852.XSHG": ("index_weight_000852_sh", "trade_universe_zz1000"),
            }
            for index_id, (w_name, mask_name) in index_map.items():
                w_mat = self._index_weight_matrix(df_iw, index_id, dates, sym_map, T, N)
                if w_mat is None:
                    continue
                out[w_name] = w_mat
                out[mask_name] = np.asfortranarray(
                    np.where(np.isfinite(w_mat), 1.0, np.nan), dtype=np.float64
                )

        # e) 行业编码别名（barra/sector 双名兼容）
        if industry_code is not None:
            out.setdefault("industry_code", industry_code)

        return out

    @staticmethod
    def _pivot_column(
        df: pl.DataFrame,
        col: str,
        date_map: dict[date, int],
        sym_map: dict[str, int],
        T: int,
        N: int,
    ) -> np.ndarray:
        """把长表某一列透视到 `(T, N)` Fortran 网格（未覆盖位置为 NaN）。"""
        mat = np.full((T, N), np.nan, dtype=np.float64, order="F")
        if col not in df.columns or df.height == 0:
            return mat
        sub = df.select(["trading_date", "order_book_id", col]).drop_nulls(col)
        if sub.height == 0:
            return mat
        # 方向安全映射：未在日期轴/标的轴上的记录（如指数成分外的标的）自动丢弃，
        # 绝不使用 replace_strict 的无默认值严格模式（数据源可能含面板外标的）。
        sub = (
            sub.with_columns(
                [
                    pl.col("trading_date").replace_strict(date_map, default=None).alias("t_idx"),
                    pl.col("order_book_id").replace_strict(sym_map, default=None).alias("s_idx"),
                ]
            )
            .drop_nulls(["t_idx", "s_idx"])
            .with_columns(
                [
                    pl.col("t_idx").cast(pl.Int64),
                    pl.col("s_idx").cast(pl.Int64),
                ]
            )
        )
        if sub.height == 0:
            return mat
        t_arr = sub["t_idx"].to_numpy()
        s_arr = sub["s_idx"].to_numpy()
        mat[t_arr, s_arr] = sub[col].to_numpy()
        return mat

    @staticmethod
    def _index_weight_matrix(
        df_iw: pl.DataFrame,
        index_id: str,
        dates: list[date],
        sym_map: dict[str, int],
        T: int,
        N: int,
    ) -> np.ndarray | None:
        """按快照日严格向前对齐（as-of backward）构造指数成分权重 `(T, N)` 矩阵。

        `snapshot_date <= trading_date` 的最近一期权重生效，快照之间保持不变；
        成分外标的权重为 NaN（与 FactorLab MCP `index_weight_*` 语义一致）。
        """
        sub = df_iw.filter(pl.col("index_id") == index_id).select(
            ["snapshot_date", "order_book_id", "weight"]
        )
        if sub.height == 0:
            return None
        snap_dates = sorted({d for d in sub["snapshot_date"].to_list() if d is not None})
        if not snap_dates:
            return None
        snap_idx = {d: i for i, d in enumerate(snap_dates)}
        snap_mat = np.full((len(snap_dates), N), np.nan, dtype=np.float64)
        s_series = sub["order_book_id"].to_list()
        d_series = sub["snapshot_date"].to_list()
        w_series = sub["weight"].to_numpy()
        r_idx: list[int] = []
        c_idx: list[int] = []
        v_idx: list[int] = []
        for k in range(sub.height):
            si = snap_idx.get(d_series[k])
            nj = sym_map.get(s_series[k])
            if si is None or nj is None:
                continue
            r_idx.append(si)
            c_idx.append(nj)
            v_idx.append(k)
        if not r_idx:
            return None
        snap_mat[np.array(r_idx), np.array(c_idx)] = w_series[np.array(v_idx)]
        # 分快照段整块广播到交易日轴（每段内权重保持不变）
        out = np.full((T, N), np.nan, dtype=np.float64, order="F")
        import bisect

        pos = -1
        seg_start = 0
        for t_i, cur_d in enumerate(dates):
            new_pos = bisect.bisect_right(snap_dates, cur_d) - 1
            if new_pos != pos:
                if pos >= 0 and t_i > seg_start:
                    out[seg_start:t_i, :] = snap_mat[pos, :]
                pos = new_pos
                seg_start = t_i
        if pos >= 0 and T > seg_start:
            out[seg_start:T, :] = snap_mat[pos, :]
        return out

    def load(
        self,
        start_date: date | str = date(2015, 11, 1),
        end_date: date | str = date(2024, 12, 31),
        benchmark_id: str = "000985.XSHG",
        exec_price_mode: str = "open_adj",
        extended_fields: bool = False,
    ) -> StockPanelData:
        """加载 A 股指定日期区间的日频全量截面面板并构建标准化掩码。

        Args:
            start_date: 起始日期 (包含)
            end_date: 截止日期 (包含)
            benchmark_id: 基准指数代码 (默认 000985.XSHG 中证全指)
            exec_price_mode: 默认执行价代理 ("open_adj" | "vwap_adj")

        Returns:
            StockPanelData 实例
        """
        t0 = time.time()
        start_d = (
            datetime.strptime(start_date, "%Y-%m-%d").date()
            if isinstance(start_date, str)
            else start_date
        )
        end_d = (
            datetime.strptime(end_date, "%Y-%m-%d").date()
            if isinstance(end_date, str)
            else end_date
        )

        # 1. 扫描日线行情文件 bar_1d
        bar_dir = self.raw_dir / "bar_1d"
        bar_files = sorted(glob.glob(str(bar_dir / "bar_1d_*.parquet")))
        if not bar_files:
            raise FileNotFoundError(f"在 {bar_dir} 未找到任何 bar_1d_*.parquet 数据文件")

        # 按实际落盘列自适应选取（合成测试夹具可能缺少 num_trades / prev_close 等可选列）
        bar_lf = pl.scan_parquet(bar_files)
        present_cols = set(bar_lf.collect_schema().names())
        wanted_cols = [
            "trading_date",
            "order_book_id",
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
        use_cols = [c for c in wanted_cols if c in present_cols]
        df_bar = (
            bar_lf.filter((pl.col("trading_date") >= start_d) & (pl.col("trading_date") <= end_d))
            .select(use_cols)
            .collect()
            .sort(["order_book_id", "trading_date"])
        )

        # 2. 关联除权因子 ex_factor (按 ex_date 进行 join_asof backward)
        ex_file = self.raw_dir / "ex_factor" / "ex_factor.parquet"
        if ex_file.exists():
            df_ex = (
                pl.read_parquet(ex_file)
                .select(["order_book_id", "ex_date", "ex_cum_factor"])
                .sort(["order_book_id", "ex_date"])
            )
            df_bar = df_bar.join_asof(
                df_ex,
                left_on="trading_date",
                right_on="ex_date",
                by="order_book_id",
                strategy="backward",
                check_sortedness=False,
            ).with_columns(pl.col("ex_cum_factor").fill_null(1.0))
        else:
            df_bar = df_bar.with_columns(pl.lit(1.0).alias("ex_cum_factor"))

        # 3. 关联交易与 ST 状态 trade_status (is_st, is_suspended)
        ts_dir = self.raw_dir / "trade_status"
        ts_files = sorted(glob.glob(str(ts_dir / "trade_status_*.parquet")))
        if ts_files:
            df_ts = (
                pl.scan_parquet(ts_files)
                .filter((pl.col("trading_date") >= start_d) & (pl.col("trading_date") <= end_d))
                .select(["trading_date", "order_book_id", "is_st", "is_suspended"])
                .collect()
            )
            df_bar = df_bar.join(df_ts, on=["trading_date", "order_book_id"], how="left")
        else:
            df_bar = df_bar.with_columns(
                [
                    pl.lit(False).alias("is_st"),
                    pl.lit(False).alias("is_suspended"),
                ]
            )

        # 填充缺失默认值
        df_bar = df_bar.with_columns(
            [
                pl.col("ex_cum_factor").fill_null(1.0),
                pl.col("is_st").fill_null(False),
                pl.col("is_suspended").fill_null(False),
            ]
        )

        # 4. 确定全局 dates 和 symbols 轴
        df_bar = df_bar.sort(["trading_date", "order_book_id"])
        dates = sorted(df_bar["trading_date"].unique().to_list())
        symbols = sorted(df_bar["order_book_id"].unique().to_list())
        T, N = len(dates), len(symbols)

        # 5. 关联申万行业分类 industry_sw
        ind_dir = self.raw_dir / "industry_sw"
        ind_files = sorted(glob.glob(str(ind_dir / "industry_sw_*.parquet")))
        if ind_files:
            df_ind = (
                pl.scan_parquet(ind_files)
                .select(["snapshot_date", "order_book_id", "first_industry_name"])
                .collect()
                .sort(["order_book_id", "snapshot_date"])
            )
            df_bar = df_bar.join_asof(
                df_ind,
                left_on="trading_date",
                right_on="snapshot_date",
                by="order_book_id",
                strategy="backward",
                check_sortedness=False,
            )
            all_industries = sorted(
                [x for x in df_ind["first_industry_name"].unique().to_list() if x is not None]
            )
            ind_map = {name: float(idx + 1) for idx, name in enumerate(all_industries)}
            df_bar = df_bar.with_columns(
                pl.col("first_industry_name")
                .replace_strict(ind_map, default=float("nan"))
                .alias("industry_code")
            )
        else:
            df_bar = df_bar.with_columns(pl.lit(float("nan")).alias("industry_code"))

        # 6. 读取标的静态信息 instruments_info (获取上市日期与退市日期)
        inst_file = self.raw_dir / "instruments_info" / "instruments_info.parquet"
        listed_dates: dict[str, date] = {}
        delisted_dates: dict[str, date] = {}
        if inst_file.exists():
            inst_df = pl.read_parquet(inst_file).select(
                ["order_book_id", "listed_date", "de_listed_date"]
            )
            for row in inst_df.iter_rows():
                s, l_str, d_str = row[0], row[1], row[2]
                if l_str:
                    try:
                        listed_dates[s] = datetime.strptime(l_str[:10], "%Y-%m-%d").date()
                    except Exception:
                        pass
                if d_str and d_str not in ("0000-00-00", "None", ""):
                    try:
                        delisted_dates[s] = datetime.strptime(d_str[:10], "%Y-%m-%d").date()
                    except Exception:
                        pass

        # 7. 构建矩阵映射索引
        date_map = {d: i for i, d in enumerate(dates)}
        sym_map = {s: i for i, s in enumerate(symbols)}

        df_bar = df_bar.with_columns(
            [
                pl.col("trading_date").replace_strict(date_map).alias("t_idx"),
                pl.col("order_book_id").replace_strict(sym_map).alias("s_idx"),
            ]
        )

        t_arr = df_bar["t_idx"].to_numpy()
        s_arr = df_bar["s_idx"].to_numpy()

        # 分配 Fortran 列优先连续内存矩阵
        close_mat = np.full((T, N), np.nan, dtype=np.float64, order="F")
        open_mat = np.full((T, N), np.nan, dtype=np.float64, order="F")
        high_mat = np.full((T, N), np.nan, dtype=np.float64, order="F")
        low_mat = np.full((T, N), np.nan, dtype=np.float64, order="F")
        amount_mat = np.full((T, N), np.nan, dtype=np.float64, order="F")
        vol_mat = np.full((T, N), np.nan, dtype=np.float64, order="F")
        ex_mat = np.full((T, N), 1.0, dtype=np.float64, order="F")
        limit_up_mat = np.full((T, N), np.nan, dtype=np.float64, order="F")
        limit_down_mat = np.full((T, N), np.nan, dtype=np.float64, order="F")
        num_trades_mat = np.full((T, N), np.nan, dtype=np.float64, order="F")
        prev_close_mat = np.full((T, N), np.nan, dtype=np.float64, order="F")

        is_susp_mat = np.full((T, N), True, dtype=bool, order="F")
        is_st_mat = np.full((T, N), False, dtype=bool, order="F")
        ind_mat = np.full((T, N), np.nan, dtype=np.float64, order="F")

        # 赋值
        close_mat[t_arr, s_arr] = df_bar["close"].to_numpy()
        open_mat[t_arr, s_arr] = df_bar["open"].to_numpy()
        high_mat[t_arr, s_arr] = df_bar["high"].to_numpy()
        low_mat[t_arr, s_arr] = df_bar["low"].to_numpy()
        amount_mat[t_arr, s_arr] = df_bar["total_turnover"].to_numpy()
        vol_mat[t_arr, s_arr] = df_bar["volume"].to_numpy()
        ex_mat[t_arr, s_arr] = df_bar["ex_cum_factor"].to_numpy()
        limit_up_mat[t_arr, s_arr] = df_bar["limit_up"].to_numpy()
        limit_down_mat[t_arr, s_arr] = df_bar["limit_down"].to_numpy()
        if "num_trades" in df_bar.columns:
            num_trades_mat[t_arr, s_arr] = df_bar["num_trades"].to_numpy()
        if "prev_close" in df_bar.columns:
            prev_close_mat[t_arr, s_arr] = df_bar["prev_close"].to_numpy()

        is_susp_mat[t_arr, s_arr] = df_bar["is_suspended"].to_numpy()
        is_st_mat[t_arr, s_arr] = df_bar["is_st"].to_numpy()
        ind_mat[t_arr, s_arr] = df_bar["industry_code"].to_numpy()

        # 8. 复权价格与派生指标矩阵计算
        close_hfq = close_mat * ex_mat
        open_adj = open_mat * ex_mat
        high_adj = high_mat * ex_mat
        low_adj = low_mat * ex_mat

        with np.errstate(divide="ignore", invalid="ignore"):
            vwap_adj = (amount_mat / np.where(vol_mat > 0.0, vol_mat, np.nan)) * ex_mat
            log_amount = np.log(np.where(amount_mat > 0.0, amount_mat, np.nan))

        c2c_ret = np.full((T, N), np.nan, dtype=np.float64, order="F")
        c2c_ret[1:] = (close_hfq[1:] / close_hfq[:-1]) - 1.0

        # 9. 构造自然日上市天数矩阵 StockListedDays 及存续期掩码
        listed_days_mat = np.full((T, N), np.nan, dtype=np.float64, order="F")
        in_listing_period = np.zeros((T, N), dtype=bool, order="F")

        for s_idx, sym in enumerate(symbols):
            l_d = listed_dates.get(sym)
            d_d = delisted_dates.get(sym)
            for t_idx, cur_d in enumerate(dates):
                if l_d is not None:
                    # 计算真实自然日差值
                    diff = (cur_d - l_d).days
                    if diff >= 0:
                        listed_days_mat[t_idx, s_idx] = float(diff)
                    # 存续期判定
                    is_listed = cur_d >= l_d
                    is_not_delisted = (d_d is None) or (cur_d <= d_d)
                    if is_listed and is_not_delisted:
                        in_listing_period[t_idx, s_idx] = True
                else:
                    # 若无静态信息，默认基于行情出现判定
                    if np.isfinite(close_mat[t_idx, s_idx]):
                        in_listing_period[t_idx, s_idx] = True

        # 10. 构造 BaseUnivMask (基础股票池掩码)
        # 规则: 处于上市存续期 & 非停牌 & 有成交量 & 非 ST/*ST & 价格有限
        base_univ_mask = (
            in_listing_period
            & (~is_susp_mat)
            & (vol_mat > 0.0)
            & (~is_st_mat)
            & np.isfinite(close_hfq)
            & np.isfinite(log_amount)
        )

        # 11. 构造 LimitMask (涨跌停与一字板不可交易掩码) 与分方向买卖可执行掩码
        # 规则:
        # - limit_mask: 信号日 T 的双边风控过滤掩码，可交易处为 1.0，收盘涨跌停/一字板处为 NaN (适配 trdMask)
        # - can_buy_mask: 买入端掩码，拦截开盘涨停 / 一字涨停 / 停牌 / 无成交 (不再误拦截可买入的跌停股)
        # - can_sell_mask: 卖出端掩码，拦截收盘跌停 / 一字跌停 / 停牌 / 无成交 (不再误删涨停盈利股)
        eps = 1e-4
        is_limit_up = np.isfinite(limit_up_mat) & (close_mat >= limit_up_mat - eps)
        is_open_limit_up = (
            np.isfinite(limit_up_mat) & np.isfinite(open_mat) & (open_mat >= limit_up_mat - eps)
        )
        is_limit_down = np.isfinite(limit_down_mat) & (close_mat <= limit_down_mat + eps)
        is_unopened_limit_up = (
            np.isfinite(high_mat)
            & np.isfinite(low_mat)
            & np.isfinite(limit_up_mat)
            & (np.abs(high_mat - low_mat) < eps)
            & (high_mat >= limit_up_mat - eps)
        )
        is_unopened_limit_down = (
            np.isfinite(high_mat)
            & np.isfinite(low_mat)
            & np.isfinite(limit_down_mat)
            & (np.abs(high_mat - low_mat) < eps)
            & (low_mat <= limit_down_mat + eps)
        )
        is_unopened_board = is_unopened_limit_up | is_unopened_limit_down
        is_locked_limit = is_limit_up | is_limit_down | is_unopened_board

        limit_mask = np.full((T, N), np.nan, dtype=np.float64, order="F")
        # 非锁板且基础数据有效的位置赋 1.0
        tradeable_limit = (~is_locked_limit) & (~is_susp_mat) & (vol_mat > 0.0)
        limit_mask[tradeable_limit] = 1.0

        can_buy_mask = np.asfortranarray(
            (~is_open_limit_up) & (~is_unopened_limit_up) & (~is_susp_mat) & (vol_mat > 0.0),
            dtype=bool,
        )
        can_sell_mask = np.asfortranarray(
            (~is_limit_down) & (~is_unopened_limit_down) & (~is_susp_mat) & (vol_mat > 0.0),
            dtype=bool,
        )

        # 11.5 FactorLab MCP 1d 变量全量对齐扩充字段 (stocks/v2)
        extended: dict[str, np.ndarray] = {}
        if extended_fields:
            extended = self._build_extended_fields(
                dates=dates,
                symbols=symbols,
                date_map=date_map,
                sym_map=sym_map,
                start_d=start_d,
                end_d=end_d,
                close_hfq=close_hfq,
                open_adj=open_adj,
                amount_mat=amount_mat,
                num_trades_mat=num_trades_mat,
                prev_close_mat=prev_close_mat,
                close_mat=close_mat,
                industry_code=ind_mat,
                T=T,
                N=N,
            )

        # 12. 读取基准指数收益率 (000985.XSHG)
        bench_file = self.raw_dir / "benchmark" / "index_bar_1d.parquet"
        bench_fwd_open_1 = np.full(T, np.nan, dtype=np.float64)
        bench_fwd_close_1 = np.full(T, np.nan, dtype=np.float64)

        if bench_file.exists():
            df_bench = (
                pl.read_parquet(bench_file)
                .filter(
                    (pl.col("order_book_id") == benchmark_id)
                    & (pl.col("trading_date") >= start_d)
                    & (pl.col("trading_date") <= end_d)
                )
                .sort("trading_date")
            )
            bench_open_dict: dict[Any, float] = {}
            bench_close_dict: dict[Any, float] = {}
            for td, op, cl in df_bench.select(["trading_date", "open", "close"]).iter_rows():
                bench_open_dict[td] = op
                bench_close_dict[td] = cl

            b_open = np.array([bench_open_dict.get(d, np.nan) for d in dates], dtype=np.float64)
            b_close = np.array([bench_close_dict.get(d, np.nan) for d in dates], dtype=np.float64)

            # 对齐信号日 t 对应的 [T+1, T+2] 远期持仓窗口
            bench_fwd_open_1[:-2] = (b_open[2:] / b_open[1:-1]) - 1.0
            bench_fwd_close_1[:-2] = (b_close[2:] / b_close[1:-1]) - 1.0

        bench_ret = bench_fwd_open_1 if exec_price_mode == "open_adj" else bench_fwd_close_1

        date_strs = [d.strftime("%Y%m%d") for d in dates]

        fields = {
            "close": close_mat,
            "open": open_mat,
            "high": high_mat,
            "low": low_mat,
            "close_hfq": close_hfq,
            "open_adj": open_adj,
            "high_adj": high_adj,
            "low_adj": low_adj,
            "vwap_adj": vwap_adj,
            "amount": amount_mat,
            "volume": vol_mat,
            "log_amount": log_amount,
            "c2c_ret": c2c_ret,
            "ex_cum_factor": ex_mat,
            "limit_up": limit_up_mat,
            "limit_down": limit_down_mat,
        }
        if extended:
            fields.update(extended)

        cost_time = time.time() - t0
        meta = {
            "start_date": start_d.strftime("%Y-%m-%d"),
            "end_date": end_d.strftime("%Y-%m-%d"),
            "benchmark_id": benchmark_id,
            "exec_price_mode": exec_price_mode,
            "n_days": T,
            "n_symbols": N,
            "load_cost_seconds": cost_time,
        }

        return StockPanelData(
            dates=date_strs,
            symbols=symbols,
            fields=fields,
            base_univ_mask=base_univ_mask,
            limit_mask=limit_mask,
            is_suspended=is_susp_mat,
            is_st=is_st_mat,
            stock_listed_days=listed_days_mat,
            industry_code=ind_mat,
            bench_fwd_open_1=bench_fwd_open_1,
            bench_fwd_close_1=bench_fwd_close_1,
            bench_ret=bench_ret,
            can_buy_mask=can_buy_mask,
            can_sell_mask=can_sell_mask,
            meta=meta,
        )


def build_stock_panel(
    cfg: StockDataConfig,
    show_progress: bool = True,
) -> tuple[pl.DataFrame, pl.DataFrame, dict[str, Any]]:
    """构建 A 股标准研究面板并持久化至 `cfg.panel_dir`（panel_daily.parquet / labels.parquet / meta.json）。"""
    del show_progress
    t0 = time.time()
    output_dir = cfg.panel_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    loader = StockPanelLoader(cfg.raw_dir)
    sdata = loader.load(
        start_date=cfg.start_date,
        end_date=cfg.end_date,
        benchmark_id=cfg.benchmark_id,
        exec_price_mode=cfg.exec_price_mode,
        extended_fields=bool(getattr(cfg, "extended_fields", False)),
    )

    # 基础股票池掩码 (BaseUnivMask) 保持与 StockPanelLoader.load 一致（剔除停牌、ST 与存续期外）；
    # 新股过滤阈值 (min_listed_days, 如 120 / 60) 交由后处理算子 dropNewStock 在滚动窗口求值后按需执行，
    # 避免在 AST 时序算子 (ts_corr / ts_std) 预热阶段提前屏蔽早期有效行情。
    uf = cfg.universe_filters
    if not uf.exclude_st:
        pass

    panel = sdata.to_panel(horizons=cfg.label_horizons)
    T, N = panel.shape
    parsed_dates = [datetime.strptime(str(d), "%Y%m%d").date() for d in panel.dates]

    date_col = pl.Series(
        "trading_date",
        np.repeat(np.array(parsed_dates, dtype="datetime64[D]"), N),
        dtype=pl.Date,
    )
    sym_col = pl.Series("symbol", np.tile(np.array(panel.symbols), T), dtype=pl.String)

    panel_cols: dict[str, Any] = {
        "trading_date": date_col,
        "symbol": sym_col,
        "in_universe": panel.mask.reshape(-1, order="C"),
    }
    for fname, mat in panel.fields.items():
        if fname in ("in_universe_mask", "limit_mask"):
            continue
        panel_cols[fname] = mat.reshape(-1, order="C")
    for aname, mat in panel.aux_fields.items():
        panel_cols[aname] = mat.reshape(-1, order="C")

    panel_df = pl.DataFrame(panel_cols).sort(["trading_date", "symbol"])

    label_cols: dict[str, Any] = {
        "trading_date": date_col,
        "symbol": sym_col,
    }
    for lname, mat in panel.labels.items():
        label_cols[lname] = mat.reshape(-1, order="C")
    labels_df = pl.DataFrame(label_cols).sort(["trading_date", "symbol"])

    panel_path = output_dir / "panel_daily.parquet"
    labels_path = output_dir / "labels.parquet"
    fields_path = output_dir / "fields.json"
    meta_path = output_dir / "meta.json"

    panel_df.write_parquet(panel_path, compression="zstd")
    labels_df.write_parquet(labels_path, compression="zstd")

    fields_doc = {
        "market": "stocks",
        "panel_version": cfg.panel_version,
        "generated_at": datetime.now().isoformat(),
        "fields": [
            {"name": k, "role": "dsl"}
            for k in panel.fields
            if k not in ("sector", "listed_days", "in_universe_mask", "limit_mask")
        ]
        + [{"name": k, "role": "aux"} for k in panel.aux_fields],
    }
    with fields_path.open("w", encoding="utf-8") as fh:
        json.dump(fields_doc, fh, indent=2, ensure_ascii=False)

    meta_doc: dict[str, Any] = {
        "market": "stocks",
        "panel_version": cfg.panel_version,
        "dataset_id": cfg.dataset_id or f"rq_{cfg.panel_version}",
        "source": cfg.source,
        "generated_at": datetime.now().isoformat(),
        "build_duration_seconds": round(time.time() - t0, 2),
        "config_hash": cfg.hash(),
        "date_range": [str(cfg.start_date), str(cfg.end_date)],
        "total_trading_days": T,
        "total_symbols": N,
        "total_records": len(panel_df),
        "in_universe_records": int(np.sum(panel.mask)),
        "capabilities": {
            "dataset_id": cfg.dataset_id or f"rq_{cfg.panel_version}",
            "source": cfg.source,
            "symbol_format": "rq",
            "return_clock": cfg.exec_price_mode,
            "has_ohlcv": True,
            "has_volume_amount": True,
            "has_directional_limit": True,
            "has_listed_days": True,
            "has_barra_exposures": False,
            "has_barra_risk_store": False,
            "has_vwap_open_30m": False,
            "has_fundamentals": bool(getattr(cfg, "extended_fields", False)),
            "has_index_weights": bool(getattr(cfg, "extended_fields", False)),
            "has_num_trades": bool(getattr(cfg, "extended_fields", False)),
            "extended_fields": bool(getattr(cfg, "extended_fields", False)),
        },
        "config": cfg.to_dict(),
    }
    with meta_path.open("w", encoding="utf-8") as fh:
        json.dump(meta_doc, fh, indent=2, ensure_ascii=False)

    return panel_df, labels_df, meta_doc
