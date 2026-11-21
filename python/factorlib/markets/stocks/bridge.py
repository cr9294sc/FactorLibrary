"""运行期轻量单向桥接层：``SignalBridge`` 与只读 ``RiskView`` (markets/stocks/bridge.py)。

按照架构设计 §2.3 实现：
- 彻底不合并底层静态表、不写回米筐派生目录，仅在内存按 ``(trading_date, sym6)`` 做轻量投影；
- ``SignalBridge``：将 ``rq``（或外部 ``shifted_signal.parquet``）因子/预测信号投影至
  ``barra`` 网格，支持 ``T_close``（默认）与 ``T_exec``（自动前移 1 个交易日）时间对齐，
  并输出结构化 ``BridgeAuditReport`` 覆盖率审计报告；
- ``RiskView``：将 ``barra`` 的 10 维 Barra 风格、行业、市值或 ``y_idio_h`` 标签以**只读视图**
  （``flags.writeable == False``）按需附加至 ``rq`` 面板，并支持基于 ``RiskView`` 的动态 WLS 特质标签构造。
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq

from factorlib.core.panel import Panel
from factorlib.markets.stocks.ares_bridge import (
    BARRA_STYLE_FACTORS,
    compute_idiosyncratic_forward_labels,
)
from factorlib.markets.stocks.barra import compute_constrained_wls_idiosyncratic_returns

_LOG = logging.getLogger(__name__)


def _to_iso_date(val: Any) -> str:
    if isinstance(val, datetime):
        return val.date().isoformat()
    if isinstance(val, date):
        return val.isoformat()
    s = str(val).strip()
    if len(s) == 8 and s.isdigit():
        return f"{s[:4]}-{s[4:6]}-{s[6:8]}"
    return s[:10]


def _normalize_sym6(sym: Any) -> str:
    s = str(sym).strip()
    if "." in s:
        s = s.split(".", 1)[0]
    return s.zfill(6) if s.isdigit() and len(s) < 6 else s


def _project_matrix_to_target(
    src_mat: np.ndarray,
    src_dates: Sequence[Any],
    src_symbols: Sequence[str],
    target_dates: Sequence[Any],
    target_symbols: Sequence[str],
    *,
    shift_rows_back: int = 0,
    start_date: Any | None = None,
    end_date: Any | None = None,
) -> tuple[np.ndarray, list[str], int]:
    """按 ``(iso_date, sym6)`` 将源二维矩阵投影至目标网格 ``(T_tgt, N_tgt, order='F')``。"""
    src = np.asarray(src_mat, dtype=np.float64)
    T_src, N_src = src.shape
    if len(src_dates) != T_src or len(src_symbols) != N_src:
        raise ValueError(
            f"源矩阵形状 {src.shape} 与 (len(dates)={len(src_dates)}, len(symbols)={len(src_symbols)}) 不一致"
        )

    tgt_dates_iso = [_to_iso_date(d) for d in target_dates]
    tgt_syms6 = [_normalize_sym6(s) for s in target_symbols]
    T_tgt, N_tgt = len(tgt_dates_iso), len(tgt_syms6)

    start_iso = _to_iso_date(start_date) if start_date is not None else None
    end_iso = _to_iso_date(end_date) if end_date is not None else None

    tgt_date_to_idx = {d: i for i, d in enumerate(tgt_dates_iso)}
    src_sym_to_col: dict[str, int] = {}
    for j, s in enumerate(src_symbols):
        s6 = _normalize_sym6(s)
        if s6 not in src_sym_to_col:
            src_sym_to_col[s6] = j

    col_pairs = [
        (nj, src_sym_to_col[s6]) for nj, s6 in enumerate(tgt_syms6) if s6 in src_sym_to_col
    ]

    row_pairs: list[tuple[int, int, str]] = []
    for sr, d_raw in enumerate(src_dates):
        d_iso = _to_iso_date(d_raw)
        ti = tgt_date_to_idx.get(d_iso)
        if ti is None:
            continue
        mapped_t = ti - int(shift_rows_back)
        if mapped_t < 0 or mapped_t >= T_tgt:
            continue
        mapped_iso = tgt_dates_iso[mapped_t]
        if start_iso is not None and mapped_iso < start_iso:
            continue
        if end_iso is not None and mapped_iso > end_iso:
            continue
        row_pairs.append((mapped_t, sr, mapped_iso))

    out = np.full((T_tgt, N_tgt), np.nan, dtype=np.float64, order="F")
    if row_pairs and col_pairs:
        t_idx = np.array([p[0] for p in row_pairs], dtype=np.int64)
        s_r_idx = np.array([p[1] for p in row_pairs], dtype=np.int64)
        n_idx = np.array([p[0] for p in col_pairs], dtype=np.int64)
        s_c_idx = np.array([p[1] for p in col_pairs], dtype=np.int64)
        out[np.ix_(t_idx, n_idx)] = src[np.ix_(s_r_idx, s_c_idx)]

    matched_dates = sorted({p[2] for p in row_pairs})
    return out, matched_dates, len(col_pairs)


@dataclass(frozen=True)
class BridgeAuditReport:
    """跨源信号桥接体检审计报告。"""

    source_dataset: str
    target_dataset: str
    input_signal_alignment: str
    date_overlap_count: int
    date_range: tuple[str, str] | None
    symbol_match_count: int
    target_symbol_count: int
    symbol_match_ratio: float
    non_null_coverage_ratio: float
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        if d["date_range"] is not None:
            d["date_range"] = list(d["date_range"])
        return d


class SignalBridge:
    """单向信号桥接器：将任意源信号矩阵投影到目标面板（如 ``barra``）网格。"""

    @staticmethod
    def project(
        scores_src: np.ndarray,
        source_dates: Sequence[Any],
        source_symbols: Sequence[str],
        target_panel: Panel,
        *,
        input_signal_alignment: str = "T_close",
        source_dataset: str = "rq",
        min_symbol_match_ratio: float = 0.90,
        start_date: Any | None = None,
        end_date: Any | None = None,
    ) -> tuple[np.ndarray, BridgeAuditReport]:
        """按 ``(trading_date, sym6)`` 投影信号，支持 ``T_close`` 与 ``T_exec`` 对齐。"""
        align_mode = str(input_signal_alignment).strip()
        if align_mode not in ("T_close", "T_exec"):
            raise ValueError(
                f"input_signal_alignment 必须为 'T_close' 或 'T_exec', 实际 {align_mode!r}"
            )
        shift_back = 1 if align_mode == "T_exec" else 0

        projected, matched_dates, n_matched_syms = _project_matrix_to_target(
            scores_src,
            source_dates,
            source_symbols,
            target_panel.dates,
            target_panel.symbols,
            shift_rows_back=shift_back,
            start_date=start_date,
            end_date=end_date,
        )

        T_tgt, N_tgt = target_panel.shape
        sym_ratio = float(n_matched_syms) / float(max(1, N_tgt))

        warnings: list[str] = []
        if len(matched_dates) == 0:
            msg = f"SignalBridge ({source_dataset} -> target) 无重叠交易日"
            warnings.append(msg)
            _LOG.warning(msg)
        if sym_ratio < float(min_symbol_match_ratio):
            msg = (
                f"SignalBridge 标的匹配覆盖率 {sym_ratio:.2%} ({n_matched_syms}/{N_tgt}) "
                f"低于阈值 {min_symbol_match_ratio:.2%}"
            )
            warnings.append(msg)
            _LOG.warning(msg)

        # 计算重叠交易日内在池标的的非空信号覆盖率
        tgt_dates_iso = [_to_iso_date(d) for d in target_panel.dates]
        matched_set = set(matched_dates)
        row_mask = np.array([d in matched_set for d in tgt_dates_iso], dtype=bool)
        if np.any(row_mask):
            active_cells = target_panel.mask[row_mask, :]
            n_active = int(np.sum(active_cells))
            if n_active > 0:
                n_fin = int(np.sum(np.isfinite(projected[row_mask, :]) & active_cells))
                cov_ratio = float(n_fin) / float(n_active)
            else:
                cov_ratio = float(np.mean(np.isfinite(projected[row_mask, :])))
        else:
            cov_ratio = 0.0

        target_ds = str(
            target_panel.meta.get("dataset_id")
            or target_panel.meta.get("panel_version")
            or "target"
        )
        date_range = (matched_dates[0], matched_dates[-1]) if matched_dates else None
        report = BridgeAuditReport(
            source_dataset=str(source_dataset),
            target_dataset=target_ds,
            input_signal_alignment=align_mode,
            date_overlap_count=len(matched_dates),
            date_range=date_range,
            symbol_match_count=n_matched_syms,
            target_symbol_count=N_tgt,
            symbol_match_ratio=sym_ratio,
            non_null_coverage_ratio=cov_ratio,
            warnings=warnings,
        )
        return projected, report

    @classmethod
    def from_panel(
        cls,
        source_panel: Panel,
        scores_src: np.ndarray,
        target_panel: Panel,
        *,
        input_signal_alignment: str = "T_close",
        min_symbol_match_ratio: float = 0.90,
        start_date: Any | None = None,
        end_date: Any | None = None,
    ) -> tuple[np.ndarray, BridgeAuditReport]:
        """直接从源 ``Panel`` 的日期与标的轴向目标 ``Panel`` 投影因子评分矩阵。"""
        src_ds = str(
            source_panel.meta.get("dataset_id")
            or source_panel.meta.get("panel_version")
            or "source_panel"
        )
        return cls.project(
            scores_src=scores_src,
            source_dates=source_panel.dates,
            source_symbols=source_panel.symbols,
            target_panel=target_panel,
            input_signal_alignment=input_signal_alignment,
            source_dataset=src_ds,
            min_symbol_match_ratio=min_symbol_match_ratio,
            start_date=start_date,
            end_date=end_date,
        )

    @classmethod
    def from_parquet(
        cls,
        signal_path: str | Path,
        target_panel: Panel,
        *,
        input_signal_alignment: str = "T_exec",
        min_symbol_match_ratio: float = 0.90,
        start_date: Any | None = None,
        end_date: Any | None = None,
    ) -> tuple[np.ndarray, BridgeAuditReport]:
        """从宽表 Parquet（如 ``shifted_signal.parquet``）读取信号并投影至目标面板。"""
        p = Path(signal_path)
        if not p.exists():
            raise FileNotFoundError(f"外部信号 Parquet 文件不存在: {p}")

        tbl = pq.read_table(p)
        idx_candidates = ("__index_level_0__", "date", "trading_date", "trade_date")
        date_col = next((c for c in idx_candidates if c in tbl.column_names), None)
        if date_col is None:
            raise ValueError(
                f"{p} 未找到日期索引列 (支持: {idx_candidates}), 实际列: {tbl.column_names[:10]}"
            )

        src_dates = tbl[date_col].to_pylist()
        sym_cols = [c for c in tbl.column_names if c != date_col]
        src_mat = np.column_stack([tbl[c].to_numpy(zero_copy_only=False) for c in sym_cols]).astype(
            np.float64
        )

        return cls.project(
            scores_src=src_mat,
            source_dates=src_dates,
            source_symbols=sym_cols,
            target_panel=target_panel,
            input_signal_alignment=input_signal_alignment,
            source_dataset=f"parquet:{p.name}",
            min_symbol_match_ratio=min_symbol_match_ratio,
            start_date=start_date,
            end_date=end_date,
        )


_DEFAULT_RISK_FIELDS: tuple[str, ...] = (
    *(f"barra_{f.lower()}" for f in BARRA_STYLE_FACTORS),
    "barra_industry",
    "mkt_cap",
)


class RiskView:
    """只读风险视图：将 ``barra`` 的 Barra 暴露与/或特质标签投影至 ``rq`` 面板。

    所有投影矩阵强制设置 ``flags.writeable = False``，绝不修改磁盘原始 Parquet。
    """

    @staticmethod
    def attach(
        target_panel: Panel,
        risk_panel: Panel,
        *,
        fields: Sequence[str] | None = None,
        labels: Sequence[str] | None = None,
        compute_dynamic_idio: bool = False,
        dynamic_horizons: Sequence[int] = (1, 2, 5, 10),
        exposure_lag_days: int = 2,
        return_field: str | None = None,
        overwrite: bool = False,
    ) -> dict[str, Any]:
        """按 ``(trading_date, sym6)`` 将风险面板字段/标签只读挂载至 ``target_panel``。"""
        req_fields = (
            list(fields)
            if fields is not None
            else [f for f in _DEFAULT_RISK_FIELDS if f in risk_panel.fields]
        )
        attached_fields: list[str] = []
        for fname in req_fields:
            if fname not in risk_panel.fields:
                raise KeyError(f"risk_panel 中未找到字段: {fname!r}")
            if fname in target_panel.fields and not overwrite:
                continue
            proj, _, _ = _project_matrix_to_target(
                risk_panel.fields[fname],
                risk_panel.dates,
                risk_panel.symbols,
                target_panel.dates,
                target_panel.symbols,
            )
            proj.flags.writeable = False
            target_panel.fields[fname] = proj
            attached_fields.append(fname)

        attached_labels: list[str] = []
        if labels:
            for lname in labels:
                if lname not in risk_panel.labels:
                    raise KeyError(f"risk_panel 中未找到标签: {lname!r}")
                if lname in target_panel.labels and not overwrite:
                    continue
                proj_l, _, _ = _project_matrix_to_target(
                    risk_panel.labels[lname],
                    risk_panel.dates,
                    risk_panel.symbols,
                    target_panel.dates,
                    target_panel.symbols,
                )
                proj_l.flags.writeable = False
                target_panel.labels[lname] = proj_l
                attached_labels.append(lname)

        if compute_dynamic_idio:
            # 在目标面板自己的执行时钟上运行因果 Constrained WLS 生成纯特质标签 y_idio_h
            if return_field is not None:
                ret_mat = target_panel.get_field(return_field)
                eff_lag = int(exposure_lag_days)
                px_clock = target_panel.fields.get("open_adj", target_panel.fields.get("close_hfq"))
            elif "open_adj" in target_panel.fields:
                px_clock = target_panel.fields["open_adj"]
                ret_mat = np.full_like(px_clock, np.nan)
                with np.errstate(divide="ignore", invalid="ignore"):
                    ret_mat[1:, :] = (px_clock[1:, :] / px_clock[:-1, :]) - 1.0
                eff_lag = int(exposure_lag_days)
            elif "c2c_ret" in target_panel.fields:
                ret_mat = target_panel.fields["c2c_ret"]
                px_clock = target_panel.fields.get("close_hfq")
                eff_lag = 1
            else:
                raise KeyError("target_panel 缺少 open_adj 或 c2c_ret，无法计算动态特质标签")

            style_dict = {
                s: target_panel.fields[f"barra_{s.lower()}"]
                for s in BARRA_STYLE_FACTORS
                if f"barra_{s.lower()}" in target_panel.fields
            }
            if not style_dict or "mkt_cap" not in target_panel.fields:
                raise KeyError("计算动态特质标签需要 barra_* 风格矩阵与 mkt_cap")
            ind_mat = target_panel.fields.get(
                "barra_industry", target_panel.fields.get("industry_code")
            )
            if ind_mat is None:
                raise KeyError("计算动态特质标签需要 barra_industry 或 industry_code")

            wls_res = compute_constrained_wls_idiosyncratic_returns(
                ret_mat,
                style_dict,
                ind_mat,
                target_panel.fields["mkt_cap"],
                valid_mask=target_panel.mask,
                exposure_lag_days=eff_lag,
                style_names=tuple(style_dict.keys()),
            )
            is_susp = (
                target_panel.get_any_field("is_suspended") > 0.5
                if target_panel.has_any_field("is_suspended")
                else ~target_panel.mask
            )
            px_ok = (
                np.isfinite(px_clock) & (px_clock > 0.0)
                if px_clock is not None
                else np.isfinite(ret_mat)
            )
            entry_ok = np.asfortranarray((~is_susp) & px_ok, dtype=bool)
            exit_ok = np.asfortranarray(px_ok, dtype=bool)
            dyn_labels = compute_idiosyncratic_forward_labels(
                wls_res.spec_returns, entry_ok, exit_ok, list(dynamic_horizons)
            )
            for lname, lmat in dyn_labels.items():
                if lname in target_panel.labels and not overwrite:
                    continue
                lmat.flags.writeable = False
                target_panel.labels[lname] = lmat
                attached_labels.append(lname)

        # 触发 styles / engine 缓存失效
        for attr in ("_cached_stocks_styles", "_cached_engine_pair"):
            if hasattr(target_panel, attr):
                delattr(target_panel, attr)

        prev = target_panel.meta.get("risk_view", {})
        prev_fields = list(prev.get("fields", [])) if isinstance(prev, dict) else []
        prev_labels = list(prev.get("labels", [])) if isinstance(prev, dict) else []
        info = {
            "source_dataset": risk_panel.meta.get(
                "dataset_id", risk_panel.meta.get("panel_version")
            ),
            "fields": sorted(set(prev_fields) | set(attached_fields)),
            "labels": sorted(set(prev_labels) | set(attached_labels)),
        }
        target_panel.meta["risk_view"] = info
        return info

    @staticmethod
    def detach(target_panel: Panel) -> dict[str, list[str]]:
        """卸载通过 ``RiskView.attach`` 挂载的只读字段与标签。"""
        info = target_panel.meta.pop("risk_view", None)
        removed_fields: list[str] = []
        removed_labels: list[str] = []
        if isinstance(info, dict):
            for fname in info.get("fields", []):
                if fname in target_panel.fields:
                    del target_panel.fields[fname]
                    removed_fields.append(fname)
            for lname in info.get("labels", []):
                if lname in target_panel.labels:
                    del target_panel.labels[lname]
                    removed_labels.append(lname)
        for attr in ("_cached_stocks_styles", "_cached_engine_pair"):
            if hasattr(target_panel, attr):
                delattr(target_panel, attr)
        return {"fields": removed_fields, "labels": removed_labels}
