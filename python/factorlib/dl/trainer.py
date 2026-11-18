"""PyTorch DL Alpha Purged Walk-Forward 多种子滚动训练器 (dl/trainer.py)。

严格遵循架构设计 §5.1~5.4 与 M8.2 规范：
1. 每折独立创建 `FoldScaler`，仅在 Fold 内部 `train_indices` 上拟合分布截断与归一化参数；
2. 每折按 `n_seeds` 训练 `3~5` 个不同随机种子（自动选型期与 Lockbox 盲测完全物理隔离）；
3. 优化目标：`CompositeAlphaLoss`（日度截面 Pearson IC Loss + 小权重 Huber + 相邻交易日预测差分惩罚）；
4. 早停机制：每轮 Epoch 在验证集上严格评估 **Valid Daily RankICIR**，保留最优权重快照；
5. 同折多种子按每日截面 Z-Score 等权融合，逐折无重叠拼接出全周期纯样本外 OOS `(T, N)` 矩阵；
6. 实时回调进度到 `RunTracker`，支持 CPU 与 NVIDIA CUDA (`RTX 4070` 等) 自动调度与确定性复现。
"""

from __future__ import annotations

import copy
import math
import random
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, field
from typing import Any

import factorlib_core
import numpy as np
import torch

from factorlib.core.panel import Panel
from factorlib.dl.dataset import (
    DLPreparedData,
    FoldScaler,
    build_dl_folds,
    gather_day_windows,
    prepare_dl_panel_data,
    resolve_sector_ids_causal,
)
from factorlib.dl.loss import CompositeAlphaLoss
from factorlib.dl.models import (
    build_dl_model,
    default_model_hparams,
    normalize_model_arch,
)
from factorlib.dl.stitch import bag_seed_predictions, stitch_oos_folds
from factorlib.ml.splits import WalkForwardFold


@dataclass
class DLTrainConfig:
    """DL Alpha Purged Walk-Forward 训练超参数与风控配置。"""

    market: str = "futures"
    arch: str = "gru"
    target_label: str = "y_5"
    seq_len: int = 20
    hidden_dim: int | None = None
    n_layers: int | None = None
    dropout: float | None = None
    lr: float | None = None
    weight_decay: float | None = None
    epochs: int = 18
    patience: int = 6
    batch_days: int | None = None
    n_folds: int = 4
    n_seeds: int = 3
    purge_window: int = 5
    embargo_window: int = 0
    train_window: str = "expanding"
    base_seed: int = 42
    feature_norm: str = "rank_gauss"
    label_neutralize: str = "demean"
    causal_ts_window: int | None = None
    ic_weight: float = 1.0
    huber_weight: float = 0.15
    turnover_weight: float = 0.05
    lockbox_start_date: str | None = None
    lockbox_ratio: float = 0.0
    prefer_yearly_folds: bool = True
    device: str = "auto"
    min_names: int = 4
    min_hist_steps: int = 1
    deterministic: bool = True

    def resolved(self) -> dict[str, Any]:
        arch_key = normalize_model_arch(self.arch, market=self.market)
        defaults = default_model_hparams(market=self.market, arch=arch_key)
        horizon = (
            int(self.target_label[2:])
            if self.target_label.startswith("y_") and self.target_label[2:].isdigit()
            else 5
        )
        return {
            "market": str(self.market).strip().lower(),
            "arch": arch_key,
            "horizon": horizon,
            "purge_window": max(horizon, int(self.purge_window)),
            "hidden_dim": int(
                self.hidden_dim if self.hidden_dim is not None else defaults["hidden_dim"]
            ),
            "n_layers": int(self.n_layers if self.n_layers is not None else defaults["n_layers"]),
            "dropout": float(self.dropout if self.dropout is not None else defaults["dropout"]),
            "lr": float(self.lr if self.lr is not None else defaults["lr"]),
            "weight_decay": float(
                self.weight_decay if self.weight_decay is not None else defaults["weight_decay"]
            ),
            "batch_days": int(
                self.batch_days if self.batch_days is not None else defaults["batch_days"]
            ),
            "use_symbol_emb": bool(defaults["use_symbol_emb"]),
        }


@dataclass
class DLTrainResult:
    """DL Alpha Walk-Forward 完整产出。"""

    oos_matrix: np.ndarray
    selection_matrix: np.ndarray
    lockbox_matrix: np.ndarray | None
    selection_icir: float
    lockbox_icir: float | None
    fold_diagnostics: list[dict[str, Any]]
    feature_names: list[str]
    model_meta: dict[str, Any]
    logs: list[str] = field(default_factory=list)


def _select_device(pref: str) -> torch.device:
    p = str(pref or "auto").strip().lower()
    if p == "cpu":
        return torch.device("cpu")
    if p.startswith("cuda") and torch.cuda.is_available():
        return torch.device(p)
    if p == "auto" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _seed_everything(seed: int, deterministic: bool = True) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def _eval_rankicir_rust(
    pred_sub: np.ndarray,
    label_sub: np.ndarray,
    mask_sub: np.ndarray,
    min_names: int,
    horizon: int,
) -> tuple[float, float, int]:
    """调用 Rust `calc_ic` 对子窗口预测评估 RankICIR、RankIC Mean 与有效天数。"""
    p_f = np.asfortranarray(pred_sub, dtype=np.float64)
    l_f = np.asfortranarray(label_sub, dtype=np.float64)
    m_f = np.asfortranarray(mask_sub, dtype=bool)
    res = factorlib_core.calc_ic(
        p_f, l_f, m_f, min_names=max(3, int(min_names)), rank=True, horizon=max(1, int(horizon))
    )
    icir = float(res.get("ic_ir", 0.0))
    ic_mean = float(res.get("ic_mean", 0.0))
    valid_days = int(res.get("valid_days", 0))
    return (
        icir if math.isfinite(icir) else 0.0,
        ic_mean if math.isfinite(ic_mean) else 0.0,
        valid_days,
    )


@torch.no_grad()
def _predict_day_indices(
    model: torch.nn.Module,
    *,
    x_scaled: np.ndarray,
    y_mat: np.ndarray,
    panel_mask: np.ndarray,
    raw_valid_mask: np.ndarray,
    day_indices: np.ndarray,
    seq_len: int,
    batch_days: int,
    symbol_ids_t: torch.Tensor,
    sector_ids_t: torch.Tensor,
    device: torch.device,
    min_hist_steps: int,
) -> np.ndarray:
    model.eval()
    n_days = len(day_indices)
    n_syms = x_scaled.shape[1]
    preds = np.full((n_days, n_syms), np.nan, dtype=np.float64)

    for s in range(0, n_days, max(1, batch_days)):
        sub_idx = day_indices[s : s + batch_days]
        xw, _, mw = gather_day_windows(
            x_scaled,
            y_mat,
            panel_mask,
            sub_idx,
            seq_len,
            raw_valid_mask=raw_valid_mask,
            min_hist_steps=min_hist_steps,
        )
        # 推理时仅要求特征侧 mask（允许最后 horizon 天无真实标签仍产成样本外预测因子）
        x_t = torch.from_numpy(np.ascontiguousarray(xw)).to(device)
        m_t = torch.from_numpy(np.ascontiguousarray(mw)).to(device)
        out_t = model(x_t, symbol_ids=symbol_ids_t, sector_ids=sector_ids_t, valid_mask=m_t)
        out_np = out_t.detach().cpu().numpy().astype(np.float64)
        out_np[~mw] = np.nan
        preds[s : s + len(sub_idx)] = out_np
    return preds


def _predict_panel_only(
    model: torch.nn.Module,
    *,
    x_scaled: np.ndarray,
    panel_mask: np.ndarray,
    raw_valid_mask: np.ndarray,
    day_indices: np.ndarray,
    seq_len: int,
    batch_days: int,
    symbol_ids_t: torch.Tensor,
    sector_ids_t: torch.Tensor,
    device: torch.device,
    min_hist_steps: int,
) -> np.ndarray:
    """严格仅使用面板有效特征掩码做推理（不读取任何测试期 `y_mat`，彻底杜绝标签依赖）。"""
    dummy_y = np.zeros(panel_mask.shape, dtype=np.float64)
    return _predict_day_indices(
        model,
        x_scaled=x_scaled,
        y_mat=dummy_y,
        panel_mask=panel_mask,
        raw_valid_mask=raw_valid_mask,
        day_indices=day_indices,
        seq_len=seq_len,
        batch_days=batch_days,
        symbol_ids_t=symbol_ids_t,
        sector_ids_t=sector_ids_t,
        device=device,
        min_hist_steps=min_hist_steps,
    )


def _train_single_fold(
    *,
    fold: WalkForwardFold,
    is_lockbox: bool,
    prepared: DLPreparedData,
    panel: Panel,
    cfg: DLTrainConfig,
    resolved: dict[str, Any],
    device: torch.device,
    fold_display_idx: int,
    total_folds: int,
    progress_callback: Callable[[dict[str, Any], str], None] | None,
) -> tuple[np.ndarray, dict[str, Any]]:
    t_len, n_syms, n_feats = prepared.shape
    del t_len
    seq_len = max(1, int(cfg.seq_len))
    batch_days = max(1, int(resolved["batch_days"]))
    horizon = int(resolved["horizon"])

    # 1. 严格因果 FoldScaler：仅使用当前折 train_indices
    scaler = FoldScaler().fit(prepared.x_raw_norm, prepared.valid_mask, fold.train_indices)
    x_scaled = scaler.transform(prepared.x_raw_norm)

    train_end_max = int(np.max(fold.train_indices))
    sector_ids_np, n_sectors = resolve_sector_ids_causal(panel, train_end_row=train_end_max)
    symbol_ids_t = torch.arange(n_syms, dtype=torch.long, device=device)
    sector_ids_t = torch.from_numpy(np.ascontiguousarray(sector_ids_np)).to(device)

    raw_val_label = prepared.y_mat[fold.val_indices]
    val_label_mask = prepared.valid_mask[fold.val_indices]

    n_seeds = max(1, int(cfg.n_seeds))
    seed_val_preds: list[np.ndarray] = []
    seed_test_preds: list[np.ndarray] = []
    seed_best_icirs: list[float] = []

    loss_fn = CompositeAlphaLoss(
        ic_weight=cfg.ic_weight,
        huber_weight=cfg.huber_weight,
        turnover_weight=cfg.turnover_weight,
        min_names=cfg.min_names,
    )

    # 筛选具有足够有效截面标的的训练交易日
    tr_days_all = fold.train_indices[
        np.sum(prepared.valid_mask[fold.train_indices], axis=1) >= max(3, cfg.min_names)
    ]
    if len(tr_days_all) == 0:
        tr_days_all = fold.train_indices

    for s_idx in range(n_seeds):
        run_seed = int(cfg.base_seed) + fold_display_idx * 100 + s_idx
        _seed_everything(run_seed, deterministic=cfg.deterministic)

        model = build_dl_model(
            resolved["arch"],
            n_features=n_feats,
            market=resolved["market"],
            seq_len=seq_len,
            hidden_dim=resolved["hidden_dim"],
            n_layers=resolved["n_layers"],
            dropout=resolved["dropout"],
            n_symbols=n_syms,
            n_sectors=max(n_sectors, prepared.n_sectors),
            use_symbol_emb=resolved["use_symbol_emb"],
        ).to(device)

        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=resolved["lr"],
            weight_decay=resolved["weight_decay"],
        )
        n_epochs = max(1, int(cfg.epochs))
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(1, n_epochs), eta_min=resolved["lr"] * 0.1
        )

        best_val_icir = -1e9
        best_val_ic = 0.0
        best_state = copy.deepcopy(model.state_dict())
        no_improve = 0
        rng = np.random.default_rng(run_seed)

        for _epoch in range(1, n_epochs + 1):
            model.train()
            shuffled = tr_days_all.copy()
            rng.shuffle(shuffled)

            for b_start in range(0, len(shuffled), batch_days):
                day_b = shuffled[b_start : b_start + batch_days]
                xw, yw, mw = gather_day_windows(
                    x_scaled,
                    prepared.y_mat,
                    prepared.valid_mask,
                    day_b,
                    seq_len,
                    raw_valid_mask=prepared.raw_valid_mask,
                    min_hist_steps=cfg.min_hist_steps,
                )
                if int(np.sum(mw)) < cfg.min_names:
                    continue

                x_t = torch.from_numpy(np.ascontiguousarray(xw)).to(device)
                y_t = torch.from_numpy(np.ascontiguousarray(yw)).to(device)
                m_t = torch.from_numpy(np.ascontiguousarray(mw)).to(device)

                # 若需要换手差分惩罚且允许读取前一日 t-1 <= train_end（不越界！），构造严格因果 t-1 对齐输入
                prev_pred_t: torch.Tensor | None = None
                prev_mask_t: torch.Tensor | None = None
                if cfg.turnover_weight > 0.0:
                    prev_days = np.maximum(day_b - 1, 0)
                    same_mask = day_b > 0
                    pxw, _, pmw = gather_day_windows(
                        x_scaled,
                        prepared.y_mat,
                        prepared.panel_mask,
                        prev_days,
                        seq_len,
                        raw_valid_mask=prepared.raw_valid_mask,
                        min_hist_steps=cfg.min_hist_steps,
                    )
                    pmw[~same_mask, :] = False
                    if int(np.sum(pmw & mw)) > 0:
                        px_t = torch.from_numpy(np.ascontiguousarray(pxw)).to(device)
                        pm_t = torch.from_numpy(np.ascontiguousarray(pmw)).to(device)
                        prev_pred_t = model(
                            px_t,
                            symbol_ids=symbol_ids_t,
                            sector_ids=sector_ids_t,
                            valid_mask=pm_t,
                        )
                        prev_mask_t = pm_t

                optimizer.zero_grad(set_to_none=True)
                pred_t = model(
                    x_t,
                    symbol_ids=symbol_ids_t,
                    sector_ids=sector_ids_t,
                    valid_mask=m_t,
                )
                loss, _ = loss_fn(
                    pred_t,
                    y_t,
                    m_t,
                    prev_pred=prev_pred_t,
                    prev_mask=prev_mask_t,
                )
                if torch.isfinite(loss):
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()

            scheduler.step()

            # Validation epoch check via Rust RankICIR
            infer_batch = max(batch_days, 8)
            val_pred = _predict_panel_only(
                model,
                x_scaled=x_scaled,
                panel_mask=prepared.panel_mask,
                raw_valid_mask=prepared.raw_valid_mask,
                day_indices=fold.val_indices,
                seq_len=seq_len,
                batch_days=infer_batch,
                symbol_ids_t=symbol_ids_t,
                sector_ids_t=sector_ids_t,
                device=device,
                min_hist_steps=cfg.min_hist_steps,
            )
            val_icir, val_ic_mean, val_days = _eval_rankicir_rust(
                val_pred, raw_val_label, val_label_mask, cfg.min_names, horizon
            )
            # 有效覆盖率检查（要求验证期至少命中 2 天以上有效日，超短夹具回退到 IC Mean）
            score = val_icir if val_days >= 2 else val_ic_mean - 10.0
            if score > best_val_icir:
                best_val_icir = score
                best_val_ic = val_ic_mean
                best_state = copy.deepcopy(model.state_dict())
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= max(1, int(cfg.patience)):
                    break

        model.load_state_dict(best_state)
        infer_batch = max(batch_days, 8)
        final_val_pred = _predict_panel_only(
            model,
            x_scaled=x_scaled,
            panel_mask=prepared.panel_mask,
            raw_valid_mask=prepared.raw_valid_mask,
            day_indices=fold.val_indices,
            seq_len=seq_len,
            batch_days=infer_batch,
            symbol_ids_t=symbol_ids_t,
            sector_ids_t=sector_ids_t,
            device=device,
            min_hist_steps=cfg.min_hist_steps,
        )
        final_test_pred = _predict_panel_only(
            model,
            x_scaled=x_scaled,
            panel_mask=prepared.panel_mask,
            raw_valid_mask=prepared.raw_valid_mask,
            day_indices=fold.test_indices,
            seq_len=seq_len,
            batch_days=infer_batch,
            symbol_ids_t=symbol_ids_t,
            sector_ids_t=sector_ids_t,
            device=device,
            min_hist_steps=cfg.min_hist_steps,
        )

        seed_val_preds.append(final_val_pred)
        seed_test_preds.append(final_test_pred)
        seed_best_icirs.append(0.0 if best_val_icir < -1e5 else float(best_val_icir))

        if progress_callback is not None:
            step_num = fold_display_idx * n_seeds + (s_idx + 1)
            tot_steps = max(1, total_folds * n_seeds)
            tag = "Lockbox" if is_lockbox else f"Fold {fold_display_idx + 1}/{total_folds}"
            msg = (
                f"{tag} · Seed {s_idx + 1}/{n_seeds} ({resolved['arch']}) "
                f"train=[{fold.train_dates[0]}..{fold.train_dates[-1]}] "
                f"test=[{fold.test_dates[0]}..{fold.test_dates[-1]}] "
                f"Valid RankICIR={seed_best_icirs[-1]:+.3f} (IC={best_val_ic:+.4f})"
            )
            progress_callback(
                {
                    "stage": f"{tag} · Seed {s_idx + 1}/{n_seeds}",
                    "pct": round(step_num / tot_steps, 4),
                    "valid_icir": round(float(np.mean(seed_best_icirs)), 4),
                    "fold": fold_display_idx + 1,
                    "n_folds": total_folds,
                    "seed": s_idx + 1,
                    "n_seeds": n_seeds,
                    "arch": resolved["arch"],
                    "device": str(device),
                },
                msg,
            )

    fused_val = bag_seed_predictions(seed_val_preds, prepared.panel_mask[fold.val_indices])
    fused_test = bag_seed_predictions(seed_test_preds, prepared.panel_mask[fold.test_indices])

    fused_val_icir, _, _ = _eval_rankicir_rust(
        fused_val, raw_val_label, val_label_mask, cfg.min_names, horizon
    )
    te_icir, te_ic_mean, te_valid_days = _eval_rankicir_rust(
        fused_test,
        prepared.y_mat[fold.test_indices],
        prepared.valid_mask[fold.test_indices],
        cfg.min_names,
        horizon,
    )

    diag = {
        "fold": fold_display_idx + 1,
        "is_lockbox": bool(is_lockbox),
        "train_range": [str(fold.train_dates[0]), str(fold.train_dates[-1])],
        "valid_range": [str(fold.val_dates[0]), str(fold.val_dates[-1])],
        "test_range": [str(fold.test_dates[0]), str(fold.test_dates[-1])],
        "n_train": int(fold.n_train),
        "n_val": int(fold.n_val),
        "n_test": int(fold.n_test),
        "n_purge": int(fold.n_purge),
        "seed_val_icirs": [round(v, 4) for v in seed_best_icirs],
        "valid_icir": round(float(fused_val_icir), 4),
        "test_icir": round(float(te_icir), 4),
        "test_ic_mean": round(float(te_ic_mean), 4),
        "test_valid_days": int(te_valid_days),
    }
    return fused_test, diag


def train_dl_walk_forward(
    panel: Panel,
    *,
    feature_names: Sequence[str],
    config: DLTrainConfig,
    progress_callback: Callable[[dict[str, Any], str], None] | None = None,
) -> DLTrainResult:
    """执行双市场统一 `1d` DL Alpha Purged Walk-Forward 多种子滚动训练。"""
    resolved = config.resolved()
    device = _select_device(config.device)

    prepared = prepare_dl_panel_data(
        panel,
        feature_names=feature_names,
        target_label=config.target_label,
        market=resolved["market"],
        feature_norm=config.feature_norm,
        label_neutralize=config.label_neutralize,
        causal_ts_window=config.causal_ts_window,
        min_names=max(3, int(config.min_names)),
    )

    folds, lockbox_fold = build_dl_folds(
        prepared.dates,
        horizon=resolved["horizon"],
        purge_window=resolved["purge_window"],
        n_folds=max(2, int(config.n_folds)),
        train_window=config.train_window,
        embargo_window=config.embargo_window,
        lockbox_start_date=config.lockbox_start_date,
        lockbox_ratio=config.lockbox_ratio,
        prefer_yearly=config.prefer_yearly_folds,
    )

    total_folds = len(folds) + (1 if lockbox_fold is not None else 0)
    t_len, n_syms, _ = prepared.shape
    selection_parts: list[tuple[np.ndarray, np.ndarray]] = []
    lockbox_parts: list[tuple[np.ndarray, np.ndarray]] = []
    fold_diags: list[dict[str, Any]] = []
    logs: list[str] = [
        f"[DL] market={resolved['market']} arch={resolved['arch']} device={device} "
        f"shape=(T={t_len}, N={n_syms}, C={len(prepared.feature_names)}, L={config.seq_len}) "
        f"purge={resolved['purge_window']} folds={len(folds)}"
        + (" + 1 lockbox" if lockbox_fold else "")
    ]

    def _wrap_cb(prog: dict[str, Any], msg: str) -> None:
        logs.append(msg)
        if progress_callback is not None:
            progress_callback(prog, msg)

    for idx_f, fold in enumerate(folds):
        sub_pred, diag = _train_single_fold(
            fold=fold,
            is_lockbox=False,
            prepared=prepared,
            panel=panel,
            cfg=config,
            resolved=resolved,
            device=device,
            fold_display_idx=idx_f,
            total_folds=total_folds,
            progress_callback=_wrap_cb,
        )
        selection_parts.append((fold.test_indices, sub_pred))
        fold_diags.append(diag)

    selection_mat = stitch_oos_folds(t_len, n_syms, selection_parts)
    sel_icir, _, _ = _eval_rankicir_rust(
        selection_mat,
        prepared.y_mat,
        prepared.valid_mask & np.isfinite(selection_mat),
        config.min_names,
        resolved["horizon"],
    )

    lockbox_mat: np.ndarray | None = None
    lockbox_icir: float | None = None
    if lockbox_fold is not None:
        lb_pred, lb_diag = _train_single_fold(
            fold=lockbox_fold,
            is_lockbox=True,
            prepared=prepared,
            panel=panel,
            cfg=config,
            resolved=resolved,
            device=device,
            fold_display_idx=len(folds),
            total_folds=total_folds,
            progress_callback=_wrap_cb,
        )
        lockbox_parts.append((lockbox_fold.test_indices, lb_pred))
        fold_diags.append(lb_diag)
        lockbox_mat = stitch_oos_folds(t_len, n_syms, lockbox_parts)
        lb_val, _, _ = _eval_rankicir_rust(
            lockbox_mat,
            prepared.y_mat,
            prepared.valid_mask & np.isfinite(lockbox_mat),
            config.min_names,
            resolved["horizon"],
        )
        lockbox_icir = float(lb_val)
        all_parts = [*selection_parts, *lockbox_parts]
        oos_full = stitch_oos_folds(t_len, n_syms, all_parts)
    else:
        oos_full = selection_mat

    model_meta = {
        **asdict(config),
        **resolved,
        "device_used": str(device),
        "feature_names": prepared.feature_names,
        "n_sectors": prepared.n_sectors,
        "selection_icir": round(float(sel_icir), 4),
        "lockbox_icir": round(float(lockbox_icir), 4) if lockbox_icir is not None else None,
        "extensions_attached": list(panel.meta.get("extensions", [])),
    }

    return DLTrainResult(
        oos_matrix=oos_full,
        selection_matrix=selection_mat,
        lockbox_matrix=lockbox_mat,
        selection_icir=float(sel_icir),
        lockbox_icir=lockbox_icir,
        fold_diagnostics=fold_diags,
        feature_names=prepared.feature_names,
        model_meta=model_meta,
        logs=logs,
    )
