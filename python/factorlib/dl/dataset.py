"""双市场统一 `1d` 时序 Dataset 构建器与 Fold 内防泄漏预处理器 (dl/dataset.py)。

严格遵循架构设计 §5.1、§5.3 与 M8.2 规范：
1. **特征物理隔离与按需扩展消费**：仅从 `FeatureSet` 白名单及 `panel.fields`（含按需挂载的
   `1m` 日频扩展包特征）抽取特征，严禁读取 `aux_fields` 与 `labels`；对底座只读数组强制拷贝，
   确保绝不原地修改共享面板，也不触发 PyTorch 只读警告；
2. **因果截面标准化 + Fold 内部缩放器**：
   - 每日截面独立执行 MAD 去极值 (`5 * 1.4826 * MAD`) 与 Rank-Gauss / CS Z-Score；
   - 任何涉及训练期分布参数估计的截断边界与标准化统计量（`FoldScaler`）仅在每个 Fold 的
     `train_indices` 内部 `fit`，再静态变换 `valid` 与 `test`，彻底切断跨界数据泄漏；
3. **标签去 Beta**：支持逐日截面 `demean`（去均值）与 `sector`（去行业/板块均值）中性化；
4. **Purged & Embargoed 切分与嵌套 Lockbox**：复用 `ml/splits.PurgedWalkForwardSplitter`，
   并在短样本/显式折数模式下按时序切分且严格断言 `purge_window >= horizon`；
5. **高内存效率滑窗采样**：仅缓存 Fold 缩放后的 `(T, N, C)` float32 立方体，按日批（Day Batch）或
   池化（Pooled）动态切片 `[t - L + 1 .. t]` 滑窗，支持大截面 A 股训练而不爆显存。
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.special import ndtri
from scipy.stats import rankdata

from factorlib.core.panel import Panel
from factorlib.ml.dataset import mad_winsorize
from factorlib.ml.splits import PurgedWalkForwardSplitter, WalkForwardFold


def _to_iso_date_str(val: Any) -> str:
    if isinstance(val, (dt.date, dt.datetime)):
        return val.strftime("%Y-%m-%d")
    s = str(val).strip()
    if len(s) == 8 and s.isdigit():
        return f"{s[:4]}-{s[4:6]}-{s[6:8]}"
    return s[:10]


def _extract_year(val: Any) -> int:
    s = _to_iso_date_str(val)
    if len(s) >= 4 and s[:4].isdigit():
        return int(s[:4])
    return 2024


def causal_rolling_ts_zscore(
    mat: np.ndarray,
    valid_mask: np.ndarray,
    window: int = 20,
    min_periods: int = 5,
    clip: float = 5.0,
) -> np.ndarray:
    """严格因果历史滚动时序 Z-Score：t 日仅依赖 `[max(0, t-W+1) .. t]` 历史观测。"""
    arr = np.array(mat, dtype=np.float64, copy=True)
    arr[~np.isfinite(arr)] = np.nan
    arr[~np.asarray(valid_mask, dtype=bool)] = np.nan
    t_len, n_cols = arr.shape
    out = np.full((t_len, n_cols), np.nan, dtype=np.float64)
    w = max(2, int(window))
    mp = max(2, min(w, int(min_periods)))

    for j in range(n_cols):
        col = arr[:, j]
        for t in range(t_len):
            v = col[t]
            if not np.isfinite(v):
                continue
            s = max(0, t - w + 1)
            hist = col[s : t + 1]
            fin = hist[np.isfinite(hist)]
            if len(fin) < mp:
                continue
            mu = float(np.mean(fin))
            sd = float(np.std(fin))
            if sd > 1e-12:
                out[t, j] = float(np.clip((v - mu) / sd, -clip, clip))
            else:
                out[t, j] = 0.0
    return out


def cross_sectional_normalize_feature(
    mat: np.ndarray,
    valid_mask: np.ndarray,
    *,
    method: str = "rank_gauss",
    n_mad: float = 5.0,
    clip: float = 4.0,
    min_names: int = 3,
    causal_ts_window: int | None = None,
) -> np.ndarray:
    """单特征严格逐日因果归一化（绝不读取未来交易日）。"""
    clean = np.array(mat, dtype=np.float64, copy=True)
    clean[~np.isfinite(clean)] = np.nan
    mask = np.asarray(valid_mask, dtype=bool)
    clean[~mask] = np.nan

    if causal_ts_window is not None and causal_ts_window > 1:
        clean = causal_rolling_ts_zscore(clean, mask, window=int(causal_ts_window))

    mode = str(method).strip().lower()
    if mode == "none":
        return np.asfortranarray(clean, dtype=np.float64)

    wins = mad_winsorize(clean, mask=mask, n_mad=n_mad)
    t_len, n_cols = wins.shape
    out = np.full((t_len, n_cols), np.nan, dtype=np.float64, order="F")

    for t in range(t_len):
        vm = mask[t, :] & np.isfinite(wins[t, :])
        idx = np.where(vm)[0]
        m = len(idx)
        if m < min_names:
            continue
        vals = wins[t, idx]
        if mode == "rank_gauss":
            if float(np.std(vals)) <= 1e-12:
                out[t, idx] = 0.0
            else:
                ranks = rankdata(vals, method="average")
                q = np.clip((ranks - 0.5) / m, 1e-4, 1.0 - 1e-4)
                out[t, idx] = np.clip(ndtri(q), -clip, clip)
        elif mode == "cs_zscore":
            mu = float(np.mean(vals))
            sd = float(np.std(vals))
            out[t, idx] = np.clip((vals - mu) / sd, -clip, clip) if sd > 1e-12 else 0.0
        else:
            raise ValueError(f"不支持的 DL 特征截面归一化模式: {method!r}")
    return out


def neutralize_label_cross_section(
    label_mat: np.ndarray,
    valid_mask: np.ndarray,
    *,
    method: str = "demean",
    sector_mat: np.ndarray | None = None,
    min_names: int = 3,
) -> np.ndarray:
    """对预测目标标签执行逐日截面去 Beta 中性化 (`none` | `demean` | `sector`)。"""
    lbl = np.array(label_mat, dtype=np.float64, copy=True)
    lbl[~np.isfinite(lbl)] = np.nan
    mask = np.asarray(valid_mask, dtype=bool)
    norm_method = str(method).strip().lower()

    if norm_method == "none":
        lbl[~mask] = np.nan
        return lbl
    if norm_method not in ("demean", "sector"):
        raise ValueError(f"不支持的标签中性化方式: {method!r}")

    t_len, n_cols = lbl.shape
    out = np.full((t_len, n_cols), np.nan, dtype=np.float64)

    for t in range(t_len):
        vm = mask[t, :] & np.isfinite(lbl[t, :])
        idx = np.where(vm)[0]
        if len(idx) < min_names:
            continue
        vals = lbl[t, idx]
        day_mean = float(np.mean(vals))

        if norm_method == "sector" and sector_mat is not None:
            sec_row = sector_mat[t, idx]
            valid_sec = np.isfinite(sec_row) & (sec_row >= 0)
            res = vals - day_mean
            if np.any(valid_sec):
                v_pos = np.where(valid_sec)[0]
                sec_codes = sec_row[v_pos].astype(np.int64)
                _, inv = np.unique(sec_codes, return_inverse=True)
                cnts = np.bincount(inv)
                sums = np.bincount(inv, weights=vals[v_pos])
                grp_means = np.where(cnts >= 2, sums / np.maximum(cnts, 1), day_mean)
                res[v_pos] = vals[v_pos] - grp_means[inv]
            out[t, idx] = res
        else:
            out[t, idx] = vals - day_mean

    return out


@dataclass
class FoldScaler:
    """Fold 内部参数化缩放器：严格仅在当前 Fold 的 `train_indices` 内部拟合分布边界。"""

    q_low: np.ndarray | None = None
    q_high: np.ndarray | None = None
    mean: np.ndarray | None = None
    std: np.ndarray | None = None
    low_pct: float = 0.5
    high_pct: float = 99.5
    clip_z: float = 6.0

    def fit(
        self,
        x_cube: np.ndarray,
        valid_mask: np.ndarray,
        train_indices: np.ndarray,
    ) -> FoldScaler:
        if len(train_indices) == 0:
            raise ValueError("FoldScaler.fit 收到空的 train_indices")
        t_len, n_cols, n_ch = x_cube.shape
        del t_len, n_cols
        tr_x = x_cube[train_indices]  # (T_tr, N, C)
        tr_m = np.asarray(valid_mask[train_indices], dtype=bool)

        q_lo = np.zeros(n_ch, dtype=np.float64)
        q_hi = np.zeros(n_ch, dtype=np.float64)
        means = np.zeros(n_ch, dtype=np.float64)
        stds = np.ones(n_ch, dtype=np.float64)

        for c in range(n_ch):
            ch_vals = tr_x[:, :, c]
            fin = ch_vals[tr_m & np.isfinite(ch_vals)]
            if len(fin) < 4:
                q_lo[c], q_hi[c], means[c], stds[c] = -10.0, 10.0, 0.0, 1.0
                continue
            lo = float(np.percentile(fin, self.low_pct))
            hi = float(np.percentile(fin, self.high_pct))
            if hi <= lo:
                lo, hi = float(np.min(fin)), float(np.max(fin))
            clipped = np.clip(fin, lo, hi)
            mu = float(np.mean(clipped))
            sd = float(np.std(clipped))
            q_lo[c] = lo
            q_hi[c] = hi
            means[c] = mu
            stds[c] = sd if sd > 1e-8 else 1.0

        self.q_low = q_lo
        self.q_high = q_hi
        self.mean = means
        self.std = stds
        return self

    def transform(self, x_cube: np.ndarray) -> np.ndarray:
        if self.q_low is None or self.q_high is None or self.mean is None or self.std is None:
            raise RuntimeError("FoldScaler 尚未 fit，无法调用 transform")
        out = np.array(x_cube, dtype=np.float32, copy=True)
        n_ch = out.shape[-1]
        for c in range(n_ch):
            ch = out[..., c]
            fin = np.isfinite(ch)
            vals = np.clip(ch[fin], float(self.q_low[c]), float(self.q_high[c]))
            vals = (vals - float(self.mean[c])) / float(self.std[c])
            vals = np.clip(vals, -self.clip_z, self.clip_z)
            ch[~fin] = 0.0
            ch[fin] = vals.astype(np.float32)
        return out


@dataclass(frozen=True)
class DLPreparedData:
    """双市场统一 DL `1d` 预处理结果面板容器（不含跨折状态，安全共享读视图）。"""

    market: str
    feature_names: list[str]
    target_label: str
    dates: np.ndarray
    symbols: list[str]
    x_raw_norm: np.ndarray  # (T, N, C) float64，每日因果截面归一化结果，待 FoldScaler 拟合
    raw_valid_mask: np.ndarray  # (T, N, C) bool
    y_mat: np.ndarray  # (T, N) float64
    valid_mask: np.ndarray  # (T, N) bool，可交易掩码 & 目标有效
    panel_mask: np.ndarray  # (T, N) bool，原始面板 valid_mask
    sector_mat: np.ndarray | None  # (T, N) float64
    n_sectors: int
    default_sector_ids: np.ndarray  # (N,) int64

    @property
    def shape(self) -> tuple[int, int, int]:
        return (
            int(self.x_raw_norm.shape[0]),
            int(self.x_raw_norm.shape[1]),
            int(self.x_raw_norm.shape[2]),
        )


def resolve_sector_ids_causal(
    panel: Panel,
    *,
    train_end_row: int | None = None,
) -> tuple[np.ndarray, int]:
    """仅利用截至 `train_end_row` 的已知分类信息提取标的一级板块/行业整数编码 `(N,)`。"""
    n_cols = panel.N
    sec_field = (
        "sector"
        if "sector" in panel.fields
        else ("industry_code" if "industry_code" in panel.fields else None)
    )
    if sec_field is None:
        return np.zeros(n_cols, dtype=np.int64), 1

    raw = np.array(panel.fields[sec_field], dtype=np.float64, copy=True)
    end_r = panel.T if train_end_row is None else max(1, min(panel.T, int(train_end_row) + 1))
    sub = raw[:end_r, :]

    codes: list[int] = []
    for j in range(n_cols):
        col = sub[:, j]
        fin = col[np.isfinite(col) & (col >= 0)]
        codes.append(int(round(float(fin[-1]))) if len(fin) > 0 else 0)

    uniq = sorted(set(codes))
    mapping = {val: idx for idx, val in enumerate(uniq)}
    mapped = np.array([mapping[v] for v in codes], dtype=np.int64)
    return mapped, max(1, len(uniq))


def prepare_dl_panel_data(
    panel: Panel,
    *,
    feature_names: Sequence[str],
    target_label: str = "y_5",
    market: str = "futures",
    feature_norm: str = "rank_gauss",
    label_neutralize: str = "demean",
    causal_ts_window: int | None = None,
    min_names: int = 3,
) -> DLPreparedData:
    """从 `Panel` 严格无前视地构造 1d DL 特征立方体与中性化标签。"""
    if target_label not in panel.labels:
        hint = (
            "（请使用 barra 数据集或先通过 RiskView.attach(..., compute_dynamic_idio=True) 挂载）"
            if str(target_label).startswith("y_idio_")
            else ""
        )
        raise ValueError(
            f"面板缺少目标标签 {target_label!r}{hint} (可用: {list(panel.labels.keys())})"
        )

    # 严格屏蔽 aux_fields 与 labels，确保特征仅来自合法观测字段（含已挂载 1m 扩展特征）
    blocked = set(panel.aux_fields.keys()) | set(panel.labels.keys())
    selected: list[str] = []
    for name in feature_names:
        if name in blocked and name not in panel.fields:
            raise ValueError(f"特征 {name!r} 属于未来/回测隔离字段，禁止进入 DL 输入特征集")
        if name in panel.fields and name not in selected:
            selected.append(name)

    if not selected:
        fallback_order = (
            "close",
            "close_hfq",
            "close_adj",
            "vwap_open_30m_adj",
            "volume",
            "log_amount",
            "log_mkt_cap",
            "c2c_ret",
            "vwap30_ret",
            "spec_ret_vwap30",
            "spec_vol_ewma",
            "ry_ann",
            "wh_z60",
            "mr_net_ratio",
            "turnover_rate",
        )
        selected = [f for f in fallback_order if f in panel.fields]
    if not selected:
        selected = [k for k in panel.fields.keys() if k not in ("sector", "industry_code")][:6]
    if not selected:
        raise ValueError("Panel 中未找到任何可用输入特征")

    panel_mask = np.array(panel.valid_mask, dtype=bool, copy=True)
    feat_mats: list[np.ndarray] = []
    valid_feats: list[np.ndarray] = []
    for f in selected:
        raw_mat = np.array(panel.fields[f], dtype=np.float64, copy=True)
        valid_feats.append(panel_mask & np.isfinite(raw_mat))
        normed = cross_sectional_normalize_feature(
            raw_mat,
            panel_mask,
            method=feature_norm,
            min_names=min_names,
            causal_ts_window=causal_ts_window,
        )
        feat_mats.append(normed)

    x_cube = np.stack(feat_mats, axis=-1)  # (T, N, C)
    raw_valid = np.stack(valid_feats, axis=-1)  # (T, N, C)

    sec_mat: np.ndarray | None = None
    if "sector" in panel.fields:
        sec_mat = np.array(panel.fields["sector"], dtype=np.float64, copy=True)
    elif "industry_code" in panel.fields:
        sec_mat = np.array(panel.fields["industry_code"], dtype=np.float64, copy=True)

    y_raw = np.array(panel.labels[target_label], dtype=np.float64, copy=True)
    y_neut = neutralize_label_cross_section(
        y_raw,
        panel_mask,
        method=label_neutralize,
        sector_mat=sec_mat,
        min_names=min_names,
    )
    target_valid = panel_mask & np.isfinite(y_neut)
    default_sec, n_sec = resolve_sector_ids_causal(panel, train_end_row=None)

    return DLPreparedData(
        market=str(market).strip().lower(),
        feature_names=selected,
        target_label=target_label,
        dates=np.array([_to_iso_date_str(d) for d in panel.dates]),
        symbols=list(panel.symbols),
        x_raw_norm=x_cube,
        raw_valid_mask=raw_valid,
        y_mat=y_neut,
        valid_mask=target_valid,
        panel_mask=panel_mask,
        sector_mat=sec_mat,
        n_sectors=n_sec,
        default_sector_ids=default_sec,
    )


def assert_folds_no_leakage(
    folds: Sequence[WalkForwardFold],
    *,
    purge_window: int,
    horizon: int,
) -> None:
    """强断言：验证每一折均满足 `purge_window >= horizon` 且测试日无重叠无穿越。"""
    if purge_window < horizon:
        raise ValueError(
            f"防泄漏校验失败：purge_window ({purge_window}) 必须 >= horizon ({horizon})"
        )
    seen_test: set[int] = set()
    for f in folds:
        if len(f.train_indices) == 0 or len(f.val_indices) == 0 or len(f.test_indices) == 0:
            raise ValueError(f"第 {f.fold_idx} 折存在空切分区间")
        tr_max = int(np.max(f.train_indices))
        val_min = int(np.min(f.val_indices))
        val_max = int(np.max(f.val_indices))
        te_min = int(np.min(f.test_indices))

        if val_min - tr_max - 1 < horizon:
            raise ValueError(
                f"第 {f.fold_idx} 折 Train→Val 隔离间距 {val_min - tr_max - 1} < horizon {horizon}"
            )
        if te_min - val_max - 1 < horizon:
            raise ValueError(
                f"第 {f.fold_idx} 折 Val→Test 隔离间距 {te_min - val_max - 1} < horizon {horizon}"
            )
        for ti in f.test_indices.tolist():
            if ti in seen_test:
                raise ValueError(f"第 {f.fold_idx} 折存在重复 OOS test 索引: {ti}")
            seen_test.add(ti)


def build_dl_folds(
    dates: Sequence[Any],
    *,
    horizon: int = 5,
    purge_window: int = 5,
    n_folds: int = 4,
    train_window: str = "expanding",
    embargo_window: int = 0,
    lockbox_start_date: str | None = None,
    lockbox_ratio: float = 0.0,
    prefer_yearly: bool = True,
) -> tuple[list[WalkForwardFold], WalkForwardFold | None]:
    """构建 Selection 期 `WalkForwardFold` 列表与可选的嵌套 Lockbox 盲测折。

    - 在多年期面板上优先调用 `PurgedWalkForwardSplitter`；
    - 当日历年数不足（如合成演示面板或高密滚动小面板）时，自动退化为均匀等步长
      Purged 时序切分，并始终强制执行 `assert_folds_no_leakage`。
    """
    eff_purge = max(int(horizon), int(purge_window))
    iso_dates = np.array([_to_iso_date_str(d) for d in dates])
    total_t = len(iso_dates)
    if total_t < max(30, 3 * eff_purge + 12):
        raise ValueError(
            f"面板交易日数 T={total_t} 过短，无法完成至少 1 折 Purged Walk-Forward 切分"
        )

    # 切分外层盲测 Lockbox 边界
    lock_start_row: int | None = None
    if lockbox_start_date:
        cutoff = _to_iso_date_str(lockbox_start_date)
        pos = np.where(iso_dates >= cutoff)[0]
        if len(pos) > 0 and int(pos[0]) >= max(30, 3 * eff_purge + 12):
            lock_start_row = int(pos[0])
    elif lockbox_ratio > 0.0:
        cand_lock = int(total_t * (1.0 - min(0.4, max(0.05, float(lockbox_ratio)))))
        if cand_lock >= max(30, 3 * eff_purge + 12) and total_t - cand_lock >= 10:
            lock_start_row = cand_lock

    sel_dates = iso_dates[:lock_start_row] if lock_start_row is not None else iso_dates
    sel_t = len(sel_dates)

    folds: list[WalkForwardFold] = []
    years = sorted({_extract_year(d) for d in sel_dates})
    if prefer_yearly and len(years) >= max(4, n_folds):
        splitter = PurgedWalkForwardSplitter(
            start_predict_year=years[2],
            end_predict_year=years[-1],
            train_window=train_window,
            min_train_years=1,
            purge_window=eff_purge,
            validation_size_days=min(126, max(15, sel_t // 8)),
            embargo_window=embargo_window,
            strict_min_years=False,
        )
        raw_folds = splitter.split(sel_dates)
        # 仅保留训练与验证规模充裕的折；若超出 n_folds 则取最后 n_folds 折或合并
        valid_raw = [f for f in raw_folds if f.n_train >= 30 and f.n_val >= 8 and f.n_test >= 5]
        if len(valid_raw) >= 2:
            if len(valid_raw) > n_folds:
                valid_raw = valid_raw[-n_folds:]
            for idx_f, rf in enumerate(valid_raw):
                folds.append(
                    WalkForwardFold(
                        fold_idx=idx_f,
                        predict_year=rf.predict_year,
                        train_indices=rf.train_indices,
                        val_indices=rf.val_indices,
                        test_indices=rf.test_indices,
                        train_dates=rf.train_dates,
                        val_dates=rf.val_dates,
                        test_dates=rf.test_dates,
                        purge_train_indices=rf.purge_train_indices,
                        purge_val_indices=rf.purge_val_indices,
                        embargo_val_indices=rf.embargo_val_indices,
                        embargo_test_indices=rf.embargo_test_indices,
                    )
                )

    if not folds:
        # 通用等步长 Chronological Purged Folds（保证任意足够长面板上都严格成立）
        n_f = max(2, int(n_folds))
        val_len = max(8, min(60, (sel_t - 2 * eff_purge) // 6))
        min_train = max(20, sel_t // 3, val_len + 2 * eff_purge)
        remain = sel_t - min_train - 2 * eff_purge - val_len
        if remain < n_f * 3:
            n_f = max(1, min(n_f, max(1, remain // 3)))
        step = max(3, (sel_t - min_train) // max(1, n_f))

        for k in range(n_f):
            test_start = min_train + k * step
            test_end = sel_t if k == n_f - 1 else min(sel_t, test_start + step)
            if test_start >= sel_t - 2 or test_end - test_start < 2:
                break
            val_end = test_start - eff_purge - embargo_window
            val_start = max(0, val_end - val_len)
            train_end = val_start - eff_purge
            train_start = (
                0 if train_window == "expanding" else max(0, train_end - max(min_train, step * 2))
            )
            if train_end - train_start < 15 or val_end - val_start < 5:
                continue

            tr_idx = np.arange(train_start, train_end, dtype=np.int64)
            pt_idx = np.arange(train_end, train_end + eff_purge, dtype=np.int64)
            emb_val = np.arange(train_end + eff_purge, val_start, dtype=np.int64)
            val_idx = np.arange(val_start, val_end, dtype=np.int64)
            pv_idx = np.arange(val_end, val_end + eff_purge, dtype=np.int64)
            emb_te = np.arange(val_end + eff_purge, test_start, dtype=np.int64)
            te_idx = np.arange(test_start, test_end, dtype=np.int64)
            folds.append(
                WalkForwardFold(
                    fold_idx=len(folds),
                    predict_year=_extract_year(iso_dates[test_start]),
                    train_indices=tr_idx,
                    val_indices=val_idx,
                    test_indices=te_idx,
                    train_dates=iso_dates[tr_idx],
                    val_dates=iso_dates[val_idx],
                    test_dates=iso_dates[te_idx],
                    purge_train_indices=pt_idx,
                    purge_val_indices=pv_idx,
                    embargo_val_indices=emb_val,
                    embargo_test_indices=emb_te,
                )
            )

    lockbox_fold: WalkForwardFold | None = None
    if lock_start_row is not None:
        val_len = max(10, min(80, (lock_start_row - 2 * eff_purge) // 5))
        val_end = lock_start_row - eff_purge - embargo_window
        val_start = max(0, val_end - val_len)
        train_end = val_start - eff_purge
        if train_end >= 20 and val_end - val_start >= 6 and total_t - lock_start_row >= 5:
            tr_idx = np.arange(0, train_end, dtype=np.int64)
            val_idx = np.arange(val_start, val_end, dtype=np.int64)
            te_idx = np.arange(lock_start_row, total_t, dtype=np.int64)
            lockbox_fold = WalkForwardFold(
                fold_idx=len(folds),
                predict_year=_extract_year(iso_dates[lock_start_row]),
                train_indices=tr_idx,
                val_indices=val_idx,
                test_indices=te_idx,
                train_dates=iso_dates[tr_idx],
                val_dates=iso_dates[val_idx],
                test_dates=iso_dates[te_idx],
                purge_train_indices=np.arange(train_end, train_end + eff_purge, dtype=np.int64),
                purge_val_indices=np.arange(val_end, val_end + eff_purge, dtype=np.int64),
                embargo_val_indices=np.arange(train_end + eff_purge, val_start, dtype=np.int64),
                embargo_test_indices=np.arange(val_end + eff_purge, lock_start_row, dtype=np.int64),
            )

    if not folds:
        raise ValueError("当前数据长度不足以切分出有效的 Walk-Forward Fold")

    all_to_check = [*folds, lockbox_fold] if lockbox_fold is not None else folds
    assert_folds_no_leakage(all_to_check, purge_window=eff_purge, horizon=horizon)
    return folds, lockbox_fold


def gather_day_windows(
    x_scaled: np.ndarray,
    y_mat: np.ndarray,
    valid_mask: np.ndarray,
    day_indices: np.ndarray,
    seq_len: int,
    *,
    raw_valid_mask: np.ndarray | None = None,
    min_hist_steps: int = 1,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """按交易日索引切片严格因果的历史滑窗 `(B_days, N, seq_len, C)`（t 日仅读 `[t-L+1..t]`）。"""
    _, n_cols, n_ch = x_scaled.shape
    b_len = len(day_indices)
    l_len = max(1, int(seq_len))
    x_win = np.zeros((b_len, n_cols, l_len, n_ch), dtype=np.float32)
    y_win = np.zeros((b_len, n_cols), dtype=np.float32)
    m_win = np.zeros((b_len, n_cols), dtype=bool)

    for bi, t_i in enumerate(day_indices.tolist()):
        start = t_i - l_len + 1
        if start >= 0:
            x_win[bi] = np.transpose(x_scaled[start : t_i + 1], (1, 0, 2))
        else:
            avail = t_i + 1
            x_win[bi, :, l_len - avail :, :] = np.transpose(x_scaled[0 : t_i + 1], (1, 0, 2))

        row_m = np.asarray(valid_mask[t_i], dtype=bool).copy()
        if raw_valid_mask is not None and min_hist_steps > 1:
            hist_s = max(0, start)
            # 每资产在回看窗口内拥有足够有效观测日才参与计算
            hist_cnt = np.sum(np.any(raw_valid_mask[hist_s : t_i + 1], axis=-1), axis=0)
            row_m &= hist_cnt >= min(min_hist_steps, t_i + 1)
        m_win[bi] = row_m
        y_row = y_mat[t_i]
        fin = row_m & np.isfinite(y_row)
        y_win[bi, fin] = y_row[fin].astype(np.float32)
        m_win[bi] = fin

    return x_win, y_win, m_win
