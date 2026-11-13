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
import time
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from factorlib.core.panel import Panel


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

    def to_panel(self) -> Panel:
        """导出为统一的 core.Panel 标准内存容器。"""
        # 将 base_univ_mask 作为主在池掩码
        panel_fields = dict(self.fields)
        listed_arr = np.asfortranarray(self.stock_listed_days)
        ind_arr = np.asfortranarray(self.industry_code)
        panel_fields["stock_listed_days"] = listed_arr
        panel_fields["listed_days"] = listed_arr
        panel_fields["industry_code"] = ind_arr
        panel_fields["sector"] = ind_arr
        panel_fields["limit_mask"] = np.asfortranarray(self.limit_mask)

        # 构造日期 numpy 数组
        dates_arr = np.array(self.dates)

        return Panel(
            dates=dates_arr,
            symbols=list(self.symbols),
            fields=panel_fields,
            mask=np.asfortranarray(self.base_univ_mask),
            meta=dict(self.meta),
        )

    def compute_forward_returns(
        self,
        horizons: list[int] | tuple[int, ...] = (1, 2, 5, 10),
        exec_price_mode: str = "open_adj",
        execution_simulation: bool = False,
    ) -> dict[int, np.ndarray]:
        """计算各预测周期的远期收益率矩阵 (T x N)。

        对齐语义:
        - 信号生成时点: T 日 15:00 收盘
        - 入场价格: T+1 日早盘执行基准 (如 open_adj)
        - 出场价格: T+1+h 日早盘执行基准 (如 open_adj)
        - 远期收益: R_{T, h} = price[T + 1 + h] / price[T + 1] - 1.0

        Args:
            horizons: 预测周期列表 (如 [1, 2, 5, 10])
            exec_price_mode: "open_adj" (默认, 贴近早盘 VwapBegin30 且无日内前视) | "vwap_adj" (全天均价)
            execution_simulation: 若为 True，施加 T+1 涨停无法买入、T+1+h 跌停/停牌无法平仓的实盘约束

        Returns:
            {h: (T x N) 收益率矩阵}
        """
        price_mat = self.fields.get(exec_price_mode)
        if price_mat is None:
            raise KeyError(f"未找到指定的执行价矩阵: {exec_price_mode!r}")

        T, N = self.shape
        fwd_returns: dict[int, np.ndarray] = {}
        vol_mat = self.fields.get("volume")

        next_exit_idx: np.ndarray | None = None
        entry_tradable: np.ndarray | None = None
        if execution_simulation:
            buy_ok = (
                self.can_buy_mask if self.can_buy_mask is not None else np.isfinite(self.limit_mask)
            )
            sell_ok = (
                self.can_sell_mask
                if self.can_sell_mask is not None
                else np.isfinite(self.limit_mask)
            )
            vol_ok = (vol_mat > 0.0) if vol_mat is not None else np.ones((T, N), dtype=bool)
            price_ok = np.isfinite(price_mat) & (price_mat > 0.0)
            entry_tradable = buy_ok & (~self.is_suspended) & vol_ok & price_ok
            exit_tradable = sell_ok & (~self.is_suspended) & vol_ok & price_ok

            next_exit_idx = np.full((T, N), -1, dtype=np.int32)
            nxt = np.full(N, -1, dtype=np.int32)
            for tau in range(T - 1, -1, -1):
                nxt = np.where(exit_tradable[tau], tau, nxt)
                next_exit_idx[tau] = nxt

        col_idx = np.arange(N)
        for h in horizons:
            fwd = np.full((T, N), np.nan, dtype=np.float64, order="F")
            max_t = T - 1 - h
            if max_t <= 0:
                fwd_returns[h] = fwd
                continue

            for t in range(max_t):
                p_in = price_mat[t + 1]

                if execution_simulation:
                    assert entry_tradable is not None and next_exit_idx is not None
                    out_indices = next_exit_idx[t + 1 + h]
                    has_exit = out_indices >= 0
                    safe_out = np.where(has_exit, out_indices, 0)
                    p_out = price_mat[safe_out, col_idx]
                    valid = entry_tradable[t + 1] & has_exit & np.isfinite(p_out) & (p_out > 0.0)
                else:
                    p_out = price_mat[t + 1 + h]
                    vol_entry_ok = (vol_mat[t + 1] > 0.0) if vol_mat is not None else True
                    valid = (
                        np.isfinite(p_in)
                        & np.isfinite(p_out)
                        & (p_in > 0.0)
                        & (p_out > 0.0)
                        & (~self.is_suspended[t + 1])
                        & vol_entry_ok
                    )

                ret = np.full(N, np.nan, dtype=np.float64)
                ret[valid] = (p_out[valid] / p_in[valid]) - 1.0
                fwd[t] = ret

            fwd_returns[h] = fwd

        return fwd_returns

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

    def load(
        self,
        start_date: date | str = date(2015, 11, 1),
        end_date: date | str = date(2024, 12, 31),
        benchmark_id: str = "000985.XSHG",
        exec_price_mode: str = "open_adj",
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

        df_bar = (
            pl.scan_parquet(bar_files)
            .filter((pl.col("trading_date") >= start_d) & (pl.col("trading_date") <= end_d))
            .select(
                [
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
                ]
            )
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
