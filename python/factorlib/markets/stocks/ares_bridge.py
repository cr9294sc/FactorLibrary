"""Barra (.ares + Barra CNE5 Parquet) 独立数据源构建桥 (markets/stocks/ares_bridge.py)。

按照双数据源独立托管架构设计：
- 零依赖 `factorsim` 二进制与米筐 `data/raw/stocks/rq/` 目录；
- 解析 `data/raw/stocks/barra/Cache/AshareCache/` 下的分片 `.ares` 矩阵与
  `Cache/BaseCache/BarraCNE5_{RET,COV}` 风险模型资产；
- 严格遵守因果防前视规范：
  1. 基于 `DailyKline.close` 与 `DailyKline.pre_close` 因果递推后复权因子 `ex_cum_factor`；
  2. `StockMask2.{BaseUnivMask,LimitMask,SuspendStock}` 显式反极性转换；
  3. `CNE5SpecRet[T]` 直接作为滞后 0 期的当日特质收益特征 `spec_ret_vwap30`；
  4. `y_idio_h[T]` 严格使用 `[T+2 .. T+1+h]` 的复利纯特质收益并隔离至 `labels.parquet`；
- 落盘为独立标准面板 `data/derived/stocks/barra/` 与旁路 `risk_store/`。
"""

from __future__ import annotations

import json
import time
from datetime import date, datetime
from pathlib import Path
from typing import Any

import factorlib_core
import numpy as np
import polars as pl
import pyarrow.parquet as pq

from factorlib.core.config import StockDataConfig

BARRA_STYLE_FACTORS: tuple[str, ...] = (
    "BETA",
    "BTOP",
    "EARNYILD",
    "GROWTH",
    "LEVERAGE",
    "LIQUIDTY",
    "MOMENTUM",
    "RESVOL",
    "SIZE",
    "SIZENL",
)

BARRA_FACTOR_ORDER: tuple[str, ...] = (
    "MOMENTUM",
    "RESVOL",
    "BETA",
    "LIQUIDTY",
    "LEVERAGE",
    "GROWTH",
    "BTOP",
    "EARNYILD",
    "SIZE",
    "SIZENL",
    "country",
    *(str(i) for i in range(1, 32)),
)

_ARES_TABLES: dict[str, str] = {
    "close_raw": "1d_DailyKline/DailyKline.close",
    "pre_close_raw": "1d_DailyKline/DailyKline.pre_close",
    "vwap_open_30m_raw": "1d_IntraVwap/IntraVwap.VwapBegin30",
    "mkt_cap": "1d_DailyFdm/DailyFdm.mkt_cap",
    "base_univ_mask": "1d_StockMask2/StockMask2.BaseUnivMask",
    "limit_mask": "1d_StockMask2/StockMask2.LimitMask",
    "suspend_stock": "1d_StockMask2/StockMask2.SuspendStock",
    "barra_industry": "1d_BarraCNE5/BarraCNE5.INDUSTRY",
    **{f"barra_{f.lower()}": f"1d_BarraCNE5/BarraCNE5.{f}" for f in BARRA_STYLE_FACTORS},
}

_DEFAULT_CAPABILITIES: dict[str, Any] = {
    "dataset_id": "barra",
    "source": "barra",
    "symbol_format": "6digit",
    "return_clock": "vwap_open_30m",
    "has_ohlcv": False,
    "has_volume_amount": False,
    "has_directional_limit": False,
    "has_listed_days": False,
    "has_barra_exposures": True,
    "has_barra_risk_store": True,
    "has_vwap_open_30m": True,
}


def _date_to_yyyymmdd_int(d: date | int | str) -> int:
    if isinstance(d, date):
        return int(d.strftime("%Y%m%d"))
    s = str(d).strip().replace("-", "")
    return int(s[:8])


def _yyyymmdd_int_to_date(val: int) -> date:
    y, rem = divmod(int(val), 10000)
    m, d = divmod(rem, 100)
    return date(y, m, d)


def _normalize_sym6(sym: Any) -> str:
    s = str(sym).strip()
    if "." in s:
        s = s.split(".", 1)[0]
    return s.zfill(6) if s.isdigit() and len(s) < 6 else s


def read_ares_table(table_dir: Path) -> tuple[np.ndarray, list[str], np.ndarray]:
    """读取单个 `.ares` 分片表目录，返回 ``(dates_int64, symbols_str, matrix_float64)``。"""
    table_dir = Path(table_dir)
    if not table_dir.is_dir():
        raise FileNotFoundError(f".ares 表目录不存在: {table_dir}")

    idx_raw = np.load(table_dir / "index.npy", allow_pickle=True)
    cols_raw = np.load(table_dir / "columns.npy", allow_pickle=True)
    meta_raw = np.load(table_dir / "meta.npy", allow_pickle=True)

    n_rows = int(meta_raw[2])
    n_cols = int(meta_raw[3])
    n_chunks = int(meta_raw[5])
    dtype = np.dtype(meta_raw[0]) if len(meta_raw) > 0 else np.dtype(np.float64)

    chunks: list[np.ndarray] = []
    for k in range(n_chunks):
        chunk_path = table_dir / f"{k}.ares"
        if not chunk_path.exists():
            raise FileNotFoundError(f"缺少 .ares 分片文件: {chunk_path}")
        raw = np.fromfile(chunk_path, dtype=dtype)
        if raw.size % n_cols != 0:
            raise ValueError(f"{chunk_path} 元素数 ({raw.size}) 不能被列数 ({n_cols}) 整除")
        chunks.append(raw.reshape(-1, n_cols))

    mat = np.vstack(chunks) if chunks else np.empty((0, n_cols), dtype=dtype)
    if mat.shape != (n_rows, n_cols):
        raise ValueError(
            f"{table_dir} 拼接维度 {mat.shape} 与 meta.npy 声明 ({n_rows}, {n_cols}) 不一致"
        )

    dates_int = np.asarray([_date_to_yyyymmdd_int(x) for x in idx_raw], dtype=np.int64)
    symbols = [_normalize_sym6(c) for c in cols_raw]
    if len(dates_int) != n_rows or len(symbols) != n_cols:
        raise ValueError(f"{table_dir} index/columns 长度与矩阵形状不匹配")

    return dates_int, symbols, np.asarray(mat, dtype=np.float64)


def _align_matrix_to_grid(
    src_dates: np.ndarray,
    src_symbols: list[str],
    src_mat: np.ndarray,
    target_dates: np.ndarray,
    target_symbols: list[str],
    *,
    fill_value: float = np.nan,
) -> np.ndarray:
    """将源二维矩阵按 (dates_int, symbols) 对齐至目标网格 (T, N, order='F')。"""
    T = len(target_dates)
    N = len(target_symbols)
    if np.array_equal(src_dates, target_dates) and src_symbols == target_symbols:
        return np.asfortranarray(src_mat, dtype=np.float64)

    src_row_map = {int(d): i for i, d in enumerate(src_dates.tolist())}
    src_col_map = {s: j for j, s in enumerate(src_symbols)}

    row_pairs = [
        (ti, src_row_map[int(d)]) for ti, d in enumerate(target_dates) if int(d) in src_row_map
    ]
    col_pairs = [(nj, src_col_map[s]) for nj, s in enumerate(target_symbols) if s in src_col_map]

    out = np.full((T, N), fill_value, dtype=np.float64, order="F")
    if not row_pairs or not col_pairs:
        return out

    t_idx = np.array([p[0] for p in row_pairs], dtype=np.int64)
    s_r_idx = np.array([p[1] for p in row_pairs], dtype=np.int64)
    n_idx = np.array([p[0] for p in col_pairs], dtype=np.int64)
    s_c_idx = np.array([p[1] for p in col_pairs], dtype=np.int64)

    if src_symbols == target_symbols:
        out[t_idx, :] = src_mat[s_r_idx, :]
    else:
        sub = src_mat[np.ix_(s_r_idx, s_c_idx)]
        out[np.ix_(t_idx, n_idx)] = sub
    return out


def compute_causal_ex_cum_factor(
    close_raw: np.ndarray,
    pre_close_raw: np.ndarray,
) -> np.ndarray:
    """基于不复权 `close` 与 `pre_close` 严格因果递推后复权因子 (T, N, order='F')。

    递推公式（首日有效置 1.0，除权日乘以 ``last_close / pre_close[t]``）：
    ``ex[t]`` 仅依赖 ``close_raw[<= t-1]`` 与 ``pre_close_raw[<= t]``，绝不触碰未来数据。
    """
    T, N = close_raw.shape
    if pre_close_raw.shape != (T, N):
        raise ValueError(
            f"close_raw {close_raw.shape} 与 pre_close_raw {pre_close_raw.shape} 形状必须一致"
        )

    ex = np.ones((T, N), dtype=np.float64, order="F")
    last_close = np.full(N, np.nan, dtype=np.float64)
    last_ex = np.ones(N, dtype=np.float64)

    for t in range(T):
        c_t = close_raw[t, :]
        p_t = pre_close_raw[t, :]
        p_ok = np.isfinite(p_t) & (p_t > 0.0) & np.isfinite(last_close) & (last_close > 0.0)
        cur_ex = last_ex.copy()
        if np.any(p_ok):
            cur_ex[p_ok] = last_ex[p_ok] * (last_close[p_ok] / p_t[p_ok])
        ex[t, :] = cur_ex

        c_ok = np.isfinite(c_t) & (c_t > 0.0)
        if np.any(c_ok):
            last_close[c_ok] = c_t[c_ok]
            last_ex[c_ok] = cur_ex[c_ok]

    return ex


def compute_causal_ewma_vol(
    mat: np.ndarray,
    *,
    window: int = 252,
    half_life: float = 42.0,
    min_obs: int = 30,
) -> np.ndarray:
    """严格因果滚动 EWMA 波动率 (T, N, order='F')，仅使用 ``<= t`` 截面。"""
    if window <= 0 or half_life <= 0.0 or min_obs <= 0:
        raise ValueError("window, half_life, min_obs 必须全部 > 0")

    T, N = mat.shape
    lam = float(0.5 ** (1.0 / half_life))
    lam_win = float(lam**window)

    out = np.full((T, N), np.nan, dtype=np.float64, order="F")
    fin_mat = np.isfinite(mat)
    sq_mat = np.where(fin_mat, mat * mat, 0.0)
    fin_f = fin_mat.astype(np.float64)

    num = np.zeros(N, dtype=np.float64)
    den = np.zeros(N, dtype=np.float64)
    cnt = np.zeros(N, dtype=np.int32)

    for t in range(T):
        num = lam * num + sq_mat[t, :]
        den = lam * den + fin_f[t, :]
        cnt = cnt + fin_mat[t, :].astype(np.int32)
        if t >= window:
            old = t - window
            num = np.maximum(num - lam_win * sq_mat[old, :], 0.0)
            den = np.maximum(den - lam_win * fin_f[old, :], 0.0)
            cnt = cnt - fin_mat[old, :].astype(np.int32)

        valid = (cnt >= min_obs) & (den > 1e-15)
        if np.any(valid):
            out[t, valid] = np.sqrt(np.maximum(num[valid] / den[valid], 0.0))

    return out


def compute_idiosyncratic_forward_labels(
    spec_ret: np.ndarray,
    entry_ok: np.ndarray,
    exit_ok: np.ndarray,
    horizons: tuple[int, ...] | list[int],
) -> dict[str, np.ndarray]:
    """根据滞后 0 期的 ``CNE5SpecRet`` 严格因果构造纯特质远期收益标签 ``y_idio_h``。

    对齐语义：对于信号日 ``T`` 收盘（次日 ``T+1`` 早盘开盘 30m 入场、``T+1+h`` 早盘出场）：
    ``y_idio_h[T] = prod_{k=2}^{h+1} (1 + spec_ret[T+k]) - 1``，
    仅当 ``entry_ok[T+1]``、``exit_ok[T+1+h]`` 均为 True 且窗口内特质收益全部有限时有效。
    """
    T, N = spec_ret.shape
    fin_spec = np.isfinite(spec_ret)
    one_plus = np.where(fin_spec, 1.0 + spec_ret, 1.0)

    labels: dict[str, np.ndarray] = {}
    for h in horizons:
        h_int = int(h)
        if h_int <= 0:
            raise ValueError(f"label horizon 必须 > 0, 实际 {h_int}")
        out = np.full((T, N), np.nan, dtype=np.float64, order="F")
        max_t = T - h_int - 1
        if max_t > 0:
            valid = entry_ok[1 : max_t + 1] & exit_ok[1 + h_int : max_t + 1 + h_int]
            prod = np.ones((max_t, N), dtype=np.float64)
            for k in range(2, h_int + 2):
                sl = slice(k, max_t + k)
                valid &= fin_spec[sl]
                prod *= one_plus[sl]
            out[:max_t, :] = np.where(valid, prod - 1.0, np.nan)
        labels[f"y_idio_{h_int}"] = out

    return labels


def _load_benchmark_series(
    benchmark_csv: Path | None,
    dates_int: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """从独立基准 CSV (``ts_code,trade_date,close``) 构建 ``bench_fwd_close_1`` 与 ``bench_c2c``。"""
    T = len(dates_int)
    bench_fwd_close_1 = np.full(T, np.nan, dtype=np.float64)
    bench_c2c = np.full(T, np.nan, dtype=np.float64)
    if benchmark_csv is None or not Path(benchmark_csv).exists():
        return bench_fwd_close_1, bench_c2c

    df = pl.read_csv(benchmark_csv)
    date_col = "trade_date" if "trade_date" in df.columns else "trading_date"
    close_map: dict[int, float] = {}
    for td, cl in df.select([date_col, "close"]).iter_rows():
        if td is None or cl is None:
            continue
        close_map[_date_to_yyyymmdd_int(td)] = float(cl)

    b_close = np.array([close_map.get(int(d), np.nan) for d in dates_int], dtype=np.float64)
    if T >= 2:
        ok_c2c = (
            (b_close[1:] > 0.0)
            & (b_close[:-1] > 0.0)
            & np.isfinite(b_close[1:])
            & np.isfinite(b_close[:-1])
        )
        bench_c2c[1:] = np.where(ok_c2c, (b_close[1:] / b_close[:-1]) - 1.0, np.nan)
    if T >= 3:
        ok_fwd = (
            (b_close[2:] > 0.0)
            & (b_close[1:-1] > 0.0)
            & np.isfinite(b_close[2:])
            & np.isfinite(b_close[1:-1])
        )
        bench_fwd_close_1[:-2] = np.where(ok_fwd, (b_close[2:] / b_close[1:-1]) - 1.0, np.nan)
    return bench_fwd_close_1, bench_c2c


def _read_spec_ret_parquet(spec_ret_path: Path) -> tuple[np.ndarray, list[str], np.ndarray]:
    """读取 `CNE5SpecRet.parquet`，返回 ``(dates_int64, symbols_str, matrix_float64)``。"""
    tbl = pq.read_table(spec_ret_path)
    date_col = "date" if "date" in tbl.column_names else "__index_level_0__"
    dates_int = np.asarray(
        [_date_to_yyyymmdd_int(x) for x in tbl[date_col].to_pylist()], dtype=np.int64
    )
    sym_cols = [c for c in tbl.column_names if c != date_col]
    symbols = [_normalize_sym6(c) for c in sym_cols]
    mat = np.column_stack([tbl[c].to_numpy(zero_copy_only=False) for c in sym_cols]).astype(
        np.float64
    )
    return dates_int, symbols, mat


def _build_risk_store(
    raw_dir: Path,
    risk_store_dir: Path,
    start_int: int,
    end_int: int,
    *,
    dataset_id: str,
    pack_cov: bool = True,
) -> dict[str, Any]:
    """将 `CNE5FactorRet.parquet` 与 `BarraCNE5_COV/*.parquet` 转换为紧凑旁路 `risk_store/`。"""
    risk_store_dir.mkdir(parents=True, exist_ok=True)
    base_cache = raw_dir / "Cache" / "BaseCache"

    # 1. CNE5FactorRet.parquet
    fr_path = base_cache / "BarraCNE5_RET" / "CNE5FactorRet.parquet"
    if not fr_path.exists():
        raise FileNotFoundError(f"未找到 CNE5FactorRet.parquet: {fr_path}")

    fr_tbl = pq.read_table(fr_path)
    fr_idx_col = "__index_level_0__" if "__index_level_0__" in fr_tbl.column_names else "date"
    fr_dates_raw = [_date_to_yyyymmdd_int(x) for x in fr_tbl[fr_idx_col].to_pylist()]
    factor_cols = [f for f in BARRA_FACTOR_ORDER if f in fr_tbl.column_names]
    if len(factor_cols) != len(BARRA_FACTOR_ORDER):
        missing = sorted(set(BARRA_FACTOR_ORDER) - set(factor_cols))
        raise ValueError(f"CNE5FactorRet.parquet 缺少因子列: {missing}")

    fr_mat = np.column_stack(
        [fr_tbl[c].to_numpy(zero_copy_only=False) for c in factor_cols]
    ).astype(np.float64)
    valid_row = np.any(np.isfinite(fr_mat), axis=1)
    keep_idx = [
        i for i, d_int in enumerate(fr_dates_raw) if valid_row[i] and start_int <= d_int <= end_int
    ]
    fr_dates_py = [_yyyymmdd_int_to_date(fr_dates_raw[i]) for i in keep_idx]
    fr_sub = fr_mat[keep_idx, :]

    fr_df = pl.DataFrame(
        {
            "trading_date": pl.Series("trading_date", fr_dates_py, dtype=pl.Date),
            **{col: fr_sub[:, j] for j, col in enumerate(factor_cols)},
        }
    ).sort("trading_date")
    fr_df.write_parquet(risk_store_dir / "factor_returns.parquet", compression="zstd")

    # 2. BarraCNE5_COV/*.parquet -> factor_cov.npz
    cov_dir = base_cache / "BarraCNE5_COV"
    cov_dates: list[str] = []
    cov_mats: list[np.ndarray] = []
    if pack_cov and cov_dir.is_dir():
        for cov_file in sorted(cov_dir.glob("*.parquet")):
            stem = cov_file.stem
            if not stem.isdigit():
                continue
            d_int = int(stem)
            if not (start_int <= d_int <= end_int):
                continue
            ct = pq.read_table(cov_file)
            c_idx_col = "__index_level_0__" if "__index_level_0__" in ct.column_names else "index"
            row_names = [str(x) for x in ct[c_idx_col].to_pylist()]
            row_map = {name: r for r, name in enumerate(row_names)}
            raw_cov = np.column_stack(
                [ct[col].to_numpy(zero_copy_only=False) for col in factor_cols]
            ).astype(np.float64)
            if row_names != list(factor_cols):
                row_perm = [row_map[f] for f in factor_cols]
                raw_cov = raw_cov[row_perm, :]
            cov_dates.append(_yyyymmdd_int_to_date(d_int).isoformat())
            cov_mats.append(raw_cov)

        cov_tensor = (
            np.stack(cov_mats, axis=0)
            if cov_mats
            else np.empty((0, len(factor_cols), len(factor_cols)), dtype=np.float64)
        )
        np.savez_compressed(
            risk_store_dir / "factor_cov.npz",
            dates=np.array(cov_dates, dtype=str),
            factors=np.array(factor_cols, dtype=str),
            cov=cov_tensor,
        )

    store_meta = {
        "source": "barra",
        "dataset_id": dataset_id,
        "return_clock": "vwap_open_30m",
        "variance_unit": "daily",
        "factors": list(factor_cols),
        "style_factors": list(BARRA_STYLE_FACTORS),
        "factor_returns_count": len(fr_dates_py),
        "factor_returns_date_range": (
            [fr_dates_py[0].isoformat(), fr_dates_py[-1].isoformat()] if fr_dates_py else []
        ),
        "factor_cov_count": len(cov_dates),
        "factor_cov_date_range": [cov_dates[0], cov_dates[-1]] if cov_dates else [],
    }
    with (risk_store_dir / "store.json").open("w", encoding="utf-8") as fh:
        json.dump(store_meta, fh, indent=2, ensure_ascii=False)
    return store_meta


def _assemble_lb_barra_data(
    cfg: StockDataConfig,
) -> dict[str, Any]:
    """加载并对齐 ``local_backtest`` 数据源为内存矩阵集合（不写盘）。"""
    raw_dir = Path(cfg.raw_dir)
    ashare_dir = raw_dir / "Cache" / "AshareCache"
    spec_ret_path = raw_dir / "Cache" / "BaseCache" / "BarraCNE5_RET" / "CNE5SpecRet.parquet"
    if not ashare_dir.is_dir():
        raise FileNotFoundError(f"AshareCache 目录不存在: {ashare_dir}")
    if not spec_ret_path.exists():
        raise FileNotFoundError(f"CNE5SpecRet.parquet 不存在: {spec_ret_path}")

    # 1. 在全量 DailyKline 轴上因果递推后复权因子 ex_cum_factor
    d_close, syms_close, close_full = read_ares_table(ashare_dir / _ARES_TABLES["close_raw"])
    d_pre, syms_pre, pre_close_full = read_ares_table(ashare_dir / _ARES_TABLES["pre_close_raw"])
    if not np.array_equal(d_close, d_pre) or syms_close != syms_pre:
        pre_close_full = _align_matrix_to_grid(d_pre, syms_pre, pre_close_full, d_close, syms_close)

    # 确保全量行、列均升序排列，便于后续零拷贝构造有序长表
    if not np.all(np.diff(d_close) > 0):
        row_order = np.argsort(d_close, kind="mergesort")
        d_close = d_close[row_order]
        close_full = np.asfortranarray(close_full[row_order, :])
        pre_close_full = np.asfortranarray(pre_close_full[row_order, :])

    sym_order = sorted(range(len(syms_close)), key=lambda i: syms_close[i])
    if sym_order != list(range(len(syms_close))):
        syms_close = [syms_close[i] for i in sym_order]
        close_full = np.asfortranarray(close_full[:, sym_order])
        pre_close_full = np.asfortranarray(pre_close_full[:, sym_order])

    ex_full = compute_causal_ex_cum_factor(close_full, pre_close_full)
    close_adj_full = close_full * ex_full
    c2c_ret_full = np.full_like(close_adj_full, np.nan)
    if close_adj_full.shape[0] >= 2:
        with np.errstate(divide="ignore", invalid="ignore"):
            c2c_ret_full[1:, :] = (close_adj_full[1:, :] / close_adj_full[:-1, :]) - 1.0

    # 2. 对齐 VwapBegin30 并使用同一因果 ex_cum_factor 复权
    d_vwap, syms_vwap, vwap_raw_src = read_ares_table(
        ashare_dir / _ARES_TABLES["vwap_open_30m_raw"]
    )
    vwap_raw_full = _align_matrix_to_grid(d_vwap, syms_vwap, vwap_raw_src, d_close, syms_close)
    vwap_adj_full = vwap_raw_full * ex_full
    vwap30_ret_full = np.full_like(vwap_adj_full, np.nan)
    if vwap_adj_full.shape[0] >= 2:
        with np.errstate(divide="ignore", invalid="ignore"):
            vwap30_ret_full[1:, :] = (vwap_adj_full[1:, :] / vwap_adj_full[:-1, :]) - 1.0

    # 3. 读取 CNE5SpecRet 并确定 [start_date, end_date] 有效目标交易日轴
    d_spec, syms_spec, spec_raw = _read_spec_ret_parquet(spec_ret_path)
    valid_spec_dates = {
        int(d_spec[i]) for i in range(len(d_spec)) if np.any(np.isfinite(spec_raw[i, :]))
    }
    start_int = _date_to_yyyymmdd_int(cfg.start_date)
    end_int = _date_to_yyyymmdd_int(cfg.end_date)
    vwap_date_set = set(d_vwap.tolist())

    target_date_mask = np.array(
        [
            (start_int <= int(d) <= end_int)
            and (int(d) in vwap_date_set)
            and (int(d) in valid_spec_dates)
            for d in d_close
        ],
        dtype=bool,
    )
    if not np.any(target_date_mask):
        raise ValueError(f"在 [{cfg.start_date}, {cfg.end_date}] 范围内未找到有效对齐交易日")

    target_dates_int = d_close[target_date_mask]
    symbols = list(syms_close)
    T = len(target_dates_int)
    N = len(symbols)

    # 4. 切片主行情 & 复权字段，对齐其余 .ares 与 CNE5SpecRet
    close_raw = np.asfortranarray(close_full[target_date_mask, :])
    pre_close_raw = np.asfortranarray(pre_close_full[target_date_mask, :])
    ex_cum_factor = np.asfortranarray(ex_full[target_date_mask, :])
    close_adj = np.asfortranarray(close_adj_full[target_date_mask, :])
    c2c_ret = np.asfortranarray(c2c_ret_full[target_date_mask, :])
    vwap_open_30m_raw = np.asfortranarray(vwap_raw_full[target_date_mask, :])
    vwap_open_30m_adj = np.asfortranarray(vwap_adj_full[target_date_mask, :])
    vwap30_ret = np.asfortranarray(vwap30_ret_full[target_date_mask, :])

    spec_ret_vwap30 = _align_matrix_to_grid(d_spec, syms_spec, spec_raw, target_dates_int, symbols)
    spec_vol_ewma = compute_causal_ewma_vol(spec_ret_vwap30, window=252, half_life=42.0, min_obs=30)

    extra_fields: dict[str, np.ndarray] = {}
    for key in (
        "mkt_cap",
        "base_univ_mask",
        "limit_mask",
        "suspend_stock",
        "barra_industry",
        *(f"barra_{f.lower()}" for f in BARRA_STYLE_FACTORS),
    ):
        d_k, s_k, m_k = read_ares_table(ashare_dir / _ARES_TABLES[key])
        extra_fields[key] = _align_matrix_to_grid(d_k, s_k, m_k, target_dates_int, symbols)

    mkt_cap = extra_fields.pop("mkt_cap")
    with np.errstate(divide="ignore", invalid="ignore"):
        log_mkt_cap = np.where(np.isfinite(mkt_cap) & (mkt_cap > 0.0), np.log(mkt_cap), np.nan)

    # 5. 反极性转换 StockMask2 掩码 (1.0=正常 -> False停牌/True可交易)
    base_src = extra_fields.pop("base_univ_mask")
    limit_src = extra_fields.pop("limit_mask")
    susp_src = extra_fields.pop("suspend_stock")

    is_suspended = np.asfortranarray(np.isnan(susp_src), dtype=bool)
    can_buy_mask = np.asfortranarray(np.isfinite(limit_src), dtype=bool)
    can_sell_mask = np.asfortranarray(np.isfinite(limit_src), dtype=bool)

    price_valid = (
        np.isfinite(close_adj)
        & (close_adj > 0.0)
        & np.isfinite(vwap_open_30m_adj)
        & (vwap_open_30m_adj > 0.0)
    )
    if cfg.universe_filters.use_source_base_univ_mask:
        in_universe = np.asfortranarray(
            np.isfinite(base_src) & (~is_suspended) & price_valid, dtype=bool
        )
    else:
        in_universe = np.asfortranarray((~is_suspended) & price_valid, dtype=bool)

    if cfg.universe_filters.exclude_bjse:
        is_bjse = np.array([s.startswith(("4", "8", "92")) for s in symbols], dtype=bool)
        if np.any(is_bjse):
            in_universe[:, is_bjse] = False

    # 6. 基准指数收益率 (000905.SH CSV)
    bench_fwd_close_1, bench_c2c = _load_benchmark_series(cfg.benchmark_csv, target_dates_int)
    bench_fwd_mat = np.asfortranarray(
        np.broadcast_to(bench_fwd_close_1.reshape(T, 1), (T, N)).copy()
    )
    bench_c2c_mat = np.asfortranarray(np.broadcast_to(bench_c2c.reshape(T, 1), (T, N)).copy())

    # 7. 构建远期总收益标签 (vwap_open_30m_adj) 与纯特质远期标签 y_idio_h
    entry_ok = np.asfortranarray(
        (~is_suspended) & np.isfinite(vwap_open_30m_adj) & (vwap_open_30m_adj > 0.0),
        dtype=bool,
    )
    exit_ok = np.asfortranarray(
        np.isfinite(vwap_open_30m_adj) & (vwap_open_30m_adj > 0.0), dtype=bool
    )
    h_list = [int(h) for h in cfg.label_horizons]
    raw_fwd = factorlib_core.build_forward_labels(
        vwap_open_30m_adj, vwap_open_30m_adj, entry_ok, exit_ok, h_list, strict=True
    )
    labels_mats: dict[str, np.ndarray] = {
        f"y_{h}": np.asfortranarray(raw_fwd[h], dtype=np.float64) for h in h_list
    }
    if 1 in raw_fwd:
        labels_mats["r_exec_next"] = np.asfortranarray(raw_fwd[1], dtype=np.float64)

    if cfg.include_idiosyncratic_labels:
        labels_mats.update(
            compute_idiosyncratic_forward_labels(spec_ret_vwap30, entry_ok, exit_ok, h_list)
        )

    barra_ind = np.asfortranarray(extra_fields.pop("barra_industry"), dtype=np.float64)
    dsl_fields: dict[str, np.ndarray] = {
        "close_adj": close_adj,
        "vwap_open_30m_adj": vwap_open_30m_adj,
        "c2c_ret": np.asfortranarray(c2c_ret, dtype=np.float64),
        "vwap30_ret": np.asfortranarray(vwap30_ret, dtype=np.float64),
        "mkt_cap": np.asfortranarray(mkt_cap, dtype=np.float64),
        "log_mkt_cap": np.asfortranarray(log_mkt_cap, dtype=np.float64),
        **{k: np.asfortranarray(v, dtype=np.float64) for k, v in extra_fields.items()},
        "barra_industry": barra_ind,
        "industry_code": barra_ind,
        "spec_ret_vwap30": np.asfortranarray(spec_ret_vwap30, dtype=np.float64),
        "spec_vol_ewma": np.asfortranarray(spec_vol_ewma, dtype=np.float64),
    }
    aux_fields: dict[str, np.ndarray] = {
        "close_raw": close_raw,
        "pre_close_raw": pre_close_raw,
        "vwap_open_30m_raw": vwap_open_30m_raw,
        "ex_cum_factor": ex_cum_factor,
        "is_suspended": np.asfortranarray(is_suspended.astype(np.float64), dtype=np.float64),
        "can_buy_mask": np.asfortranarray(can_buy_mask.astype(np.float64), dtype=np.float64),
        "can_sell_mask": np.asfortranarray(can_sell_mask.astype(np.float64), dtype=np.float64),
        "bench_fwd_close_1": bench_fwd_mat,
        "bench_ret": bench_fwd_mat,
        "bench_c2c": bench_c2c_mat,
    }
    dataset_id = cfg.dataset_id or cfg.panel_version
    capabilities = {
        **_DEFAULT_CAPABILITIES,
        "dataset_id": dataset_id,
        "source": cfg.source,
        "return_clock": cfg.entry_window,
    }
    return {
        "target_dates_int": target_dates_int,
        "parsed_dates": [_yyyymmdd_int_to_date(int(d)) for d in target_dates_int],
        "symbols": symbols,
        "in_universe": in_universe,
        "dsl_fields": dsl_fields,
        "aux_fields": aux_fields,
        "labels_mats": labels_mats,
        "dataset_id": dataset_id,
        "capabilities": capabilities,
    }


def load_lb_barra_panel(cfg: StockDataConfig) -> Any:
    """直接从 ``local_backtest`` 原始资产加载内存 ``Panel``（回退路径，不写盘）。"""
    from factorlib.core.panel import Panel

    assembled = _assemble_lb_barra_data(cfg)
    dsl_fields = dict(assembled["dsl_fields"])
    aux_fields = dict(assembled["aux_fields"])
    in_universe = assembled["in_universe"]

    dsl_fields["sector"] = dsl_fields["industry_code"]
    dsl_fields["in_universe_mask"] = np.asfortranarray(
        np.where(in_universe, 1.0, np.nan), dtype=np.float64
    )
    limit_ok = (aux_fields["can_buy_mask"] > 0.5) & (~(aux_fields["is_suspended"] > 0.5))
    dsl_fields["limit_mask"] = np.asfortranarray(np.where(limit_ok, 1.0, np.nan), dtype=np.float64)
    dates_arr = np.array(assembled["parsed_dates"], dtype="datetime64[D]")
    return Panel(
        dates=dates_arr,
        symbols=assembled["symbols"],
        fields=dsl_fields,
        mask=in_universe,
        labels=assembled["labels_mats"],
        aux_fields=aux_fields,
        meta={
            "market": "stocks",
            "panel_version": cfg.panel_version,
            "dataset_id": assembled["dataset_id"],
            "source": cfg.source,
            "capabilities": assembled["capabilities"],
        },
    )


def build_lb_barra_panel(
    cfg: StockDataConfig,
    *,
    pack_cov: bool = True,
    show_progress: bool = True,
) -> tuple[pl.DataFrame, pl.DataFrame, dict[str, Any]]:
    """将 `data/raw/stocks/barra/` 转换为独立标准面板 `data/derived/stocks/barra/`。"""
    del show_progress
    t0 = time.time()
    output_dir = cfg.panel_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    assembled = _assemble_lb_barra_data(cfg)
    target_dates_int = assembled["target_dates_int"]
    parsed_dates = assembled["parsed_dates"]
    symbols = assembled["symbols"]
    in_universe = assembled["in_universe"]
    dsl_fields = assembled["dsl_fields"]
    aux_fields = assembled["aux_fields"]
    labels_mats = assembled["labels_mats"]
    dataset_id = assembled["dataset_id"]
    capabilities = assembled["capabilities"]

    T = len(parsed_dates)
    N = len(symbols)

    # 8. 组装并写盘 Parquet 长表 (panel_daily.parquet & labels.parquet)
    date_col = pl.Series(
        "trading_date",
        np.repeat(np.array(parsed_dates, dtype="datetime64[D]"), N),
        dtype=pl.Date,
    )
    sym_col = pl.Series("symbol", np.tile(np.array(symbols), T), dtype=pl.String)

    panel_cols: dict[str, Any] = {
        "trading_date": date_col,
        "symbol": sym_col,
        "in_universe": in_universe.reshape(-1, order="C"),
    }
    for name, mat in dsl_fields.items():
        panel_cols[name] = mat.reshape(-1, order="C")
    for name, mat in aux_fields.items():
        panel_cols[name] = mat.reshape(-1, order="C")

    panel_df = pl.DataFrame(panel_cols)
    label_cols: dict[str, Any] = {"trading_date": date_col, "symbol": sym_col}
    for lname, mat in labels_mats.items():
        label_cols[lname] = mat.reshape(-1, order="C")
    labels_df = pl.DataFrame(label_cols)

    panel_df.write_parquet(output_dir / "panel_daily.parquet", compression="zstd")
    labels_df.write_parquet(output_dir / "labels.parquet", compression="zstd")

    # 9. 构建旁路 risk_store/
    risk_meta = _build_risk_store(
        Path(cfg.raw_dir),
        output_dir / "risk_store",
        start_int=int(target_dates_int[0]),
        end_int=int(target_dates_int[-1]),
        dataset_id=dataset_id,
        pack_cov=pack_cov,
    )

    # 10. 写盘 fields.json 与 meta.json (含显式 capabilities)
    fields_doc = {
        "market": "stocks",
        "panel_version": cfg.panel_version,
        "dataset_id": dataset_id,
        "generated_at": datetime.now().isoformat(),
        "fields": [{"name": k, "role": "dsl"} for k in dsl_fields]
        + [{"name": k, "role": "aux"} for k in aux_fields],
    }
    with (output_dir / "fields.json").open("w", encoding="utf-8") as fh:
        json.dump(fields_doc, fh, indent=2, ensure_ascii=False)

    meta_doc: dict[str, Any] = {
        "market": "stocks",
        "panel_version": cfg.panel_version,
        "dataset_id": dataset_id,
        "source": cfg.source,
        "generated_at": datetime.now().isoformat(),
        "build_duration_seconds": round(time.time() - t0, 2),
        "config_hash": cfg.hash(),
        "date_range": [parsed_dates[0].isoformat(), parsed_dates[-1].isoformat()],
        "total_trading_days": T,
        "total_symbols": N,
        "total_records": len(panel_df),
        "in_universe_records": int(np.sum(in_universe)),
        "capabilities": capabilities,
        "risk_store": risk_meta,
        "config": cfg.to_dict(),
    }
    with (output_dir / "meta.json").open("w", encoding="utf-8") as fh:
        json.dump(meta_doc, fh, indent=2, ensure_ascii=False)

    return panel_df, labels_df, meta_doc
