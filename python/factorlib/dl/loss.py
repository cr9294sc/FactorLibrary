"""DL Alpha 截面优化损失函数集 (dl/loss.py)。

严格遵循架构设计 §5.1.3：
- 日内截面 Pearson/Cosine IC Loss（按日分组或日批维度计算截面相关系数，对齐排序收益目标）；
- 小权重 Huber 锚定项，约束模型输出幅度并防止截面极值震荡；
- 相邻交易日预测差分惩罚（Turnover Diff Penalty），抑制无谓日度换手；
- 全向量化无 Python 循环实现（对日批张量 `(B_days, N)` 直接求解），支持 GPU 极速训练。
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def daily_pearson_ic(
    pred: torch.Tensor,
    target: torch.Tensor,
    valid_mask: torch.Tensor | None = None,
    min_names: int = 4,
    eps: float = 1e-8,
) -> tuple[torch.Tensor, torch.Tensor]:
    """计算日批维度 `dim=-1` 上每个有效交易日的截面 Pearson IC。

    Parameters
    ----------
    pred:
        预测得分张量 `(B, N)` 或一维单日向量 `(N,)`。
    target:
        目标收益张量 `(B, N)` 或 `(N,)`。
    valid_mask:
        布尔有效掩码 `(B, N)`；为 ``None`` 时使用两者均有限的位置。
    min_names:
        单日截面最少有效标的数量。
    eps:
        分母稳定项，避免零方差导致的 ``NaN`` 梯度。

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor]
        `(mean_ic, valid_day_count)`；若批内无有效交易日则返回 `(0.0, 0)`。
    """
    if pred.dim() == 1:
        pred = pred.unsqueeze(0)
        target = target.unsqueeze(0)
        if valid_mask is not None:
            valid_mask = valid_mask.unsqueeze(0)

    if valid_mask is None:
        mask = torch.isfinite(pred) & torch.isfinite(target)
    else:
        mask = valid_mask.bool() & torch.isfinite(target)

    p_clean = torch.where(mask, pred, torch.zeros_like(pred))
    y_clean = torch.where(mask, target, torch.zeros_like(target))
    w = mask.to(dtype=pred.dtype)

    counts = w.sum(dim=1, keepdim=True)
    denom_cnt = counts.clamp(min=1.0)
    p_mean = p_clean.sum(dim=1, keepdim=True) / denom_cnt
    y_mean = y_clean.sum(dim=1, keepdim=True) / denom_cnt

    dp = (p_clean - p_mean) * w
    dy = (y_clean - y_mean) * w

    cov = (dp * dy).sum(dim=1)
    var_p = (dp * dp).sum(dim=1)
    var_y = (dy * dy).sum(dim=1)

    ok_days = (counts.squeeze(1) >= float(min_names)) & (var_y > eps)
    n_ok = ok_days.sum()
    if int(n_ok.item()) == 0:
        return pred.sum() * 0.0, n_ok

    denom = torch.sqrt((var_p[ok_days] + eps) * (var_y[ok_days] + eps))
    ics = torch.clamp(cov[ok_days] / denom, min=-1.0, max=1.0)
    return ics.mean(), n_ok


def grouped_daily_ic_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    day_ids: torch.Tensor,
    min_names: int = 4,
    eps: float = 1e-8,
) -> torch.Tensor:
    """按一维 `day_ids` 分组计算截面 IC 损失 `1.0 - mean(IC)`（用于扁平池化批样本）。"""
    uniq_days = torch.unique(day_ids)
    ics: list[torch.Tensor] = []
    for d in uniq_days:
        sel = day_ids == d
        if int(sel.sum().item()) < min_names:
            continue
        p_d = pred[sel]
        y_d = target[sel]
        dp = p_d - p_d.mean()
        dy = y_d - y_d.mean()
        vy = (dy * dy).sum()
        if float(vy.detach().item()) <= eps:
            continue
        vp = (dp * dp).sum()
        ic = (dp * dy).sum() / torch.sqrt((vp + eps) * (vy + eps))
        ics.append(torch.clamp(ic, min=-1.0, max=1.0))

    if not ics:
        return pred.sum() * 0.0 + 1.0
    return 1.0 - torch.stack(ics).mean()


def cross_sectional_zscore_target(
    target: torch.Tensor,
    valid_mask: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """对日批目标 `(B, N)` 按有效截面标准化，供 Huber 锚定项对齐量纲。"""
    w = valid_mask.to(dtype=target.dtype)
    y = torch.where(valid_mask, target, torch.zeros_like(target))
    cnt = w.sum(dim=1, keepdim=True).clamp(min=1.0)
    mu = y.sum(dim=1, keepdim=True) / cnt
    diff = (y - mu) * w
    var = (diff * diff).sum(dim=1, keepdim=True) / cnt
    sd = torch.sqrt(var + eps)
    return torch.clamp(diff / sd, min=-4.0, max=4.0)


class CompositeAlphaLoss(nn.Module):
    """复合 DL Alpha 优化目标：日内截面 IC + 小权重 Huber + 相邻交易日换手差分惩罚。"""

    def __init__(
        self,
        ic_weight: float = 1.0,
        huber_weight: float = 0.15,
        turnover_weight: float = 0.05,
        huber_delta: float = 1.0,
        min_names: int = 4,
    ) -> None:
        super().__init__()
        self.ic_weight = float(ic_weight)
        self.huber_weight = float(huber_weight)
        self.turnover_weight = float(turnover_weight)
        self.huber_delta = float(huber_delta)
        self.min_names = int(min_names)

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        valid_mask: torch.Tensor,
        *,
        prev_pred: torch.Tensor | None = None,
        prev_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        mean_ic, n_ok = daily_pearson_ic(
            pred, target, valid_mask=valid_mask, min_names=self.min_names
        )
        ic_loss = 1.0 - mean_ic

        # 小权重 Huber 锚定（将 pred 对齐到日度标准化后的 target 量纲）
        if self.huber_weight > 0.0 and bool(valid_mask.any().item()):
            y_norm = cross_sectional_zscore_target(target, valid_mask)
            p_sel = pred[valid_mask]
            y_sel = y_norm[valid_mask]
            huber = F.huber_loss(p_sel, y_sel, delta=self.huber_delta)
        else:
            huber = pred.sum() * 0.0

        # 相邻交易日预测差分惩罚（按预测得分平滑变化抑换手）
        if (
            self.turnover_weight > 0.0
            and prev_pred is not None
            and prev_mask is not None
            and pred.shape == prev_pred.shape
        ):
            both = valid_mask.bool() & prev_mask.bool()
            if bool(both.any().item()):
                diff = pred[both] - prev_pred[both]
                turn_pen = (diff * diff).mean()
            else:
                turn_pen = pred.sum() * 0.0
        elif self.turnover_weight > 0.0 and pred.dim() == 2 and pred.shape[0] > 1:
            # 退化兜底：若批内按时间邻接排列，直接在 batch 内部邻接两行计算平滑差分
            both = valid_mask[1:].bool() & valid_mask[:-1].bool()
            if bool(both.any().item()):
                diff = pred[1:][both] - pred[:-1][both]
                turn_pen = (diff * diff).mean()
            else:
                turn_pen = pred.sum() * 0.0
        else:
            turn_pen = pred.sum() * 0.0

        total = (
            self.ic_weight * ic_loss + self.huber_weight * huber + self.turnover_weight * turn_pen
        )
        stats = {
            "loss": float(total.detach().item()),
            "ic": float(mean_ic.detach().item()),
            "huber": float(huber.detach().item()),
            "turnover_pen": float(turn_pen.detach().item()),
            "valid_days": float(n_ok.detach().item()),
        }
        return total, stats
