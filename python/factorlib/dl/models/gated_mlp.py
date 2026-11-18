"""Gated-MLP 基线时序因子网络 (dl/models/gated_mlp.py)。

采用多尺度因果时序汇聚（当前截面 $t$ + 短/全窗均值与波动 + 线性衰减加权）+
SwiGLU 门控残差前馈块 + 可选品种/板块低维嵌入，在期货与 A 股 `1d` 上充当高速防过拟合基线。
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class _SwiGLUBlock(nn.Module):
    def __init__(self, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim)
        self.gate_proj = nn.Linear(hidden_dim, hidden_dim * 2)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm(x)
        u, v = self.gate_proj(h).chunk(2, dim=-1)
        h = F.silu(u) * v
        h = self.drop(h)
        h = self.out_proj(h)
        return x + self.drop(h)


class GatedMLPAlpha(nn.Module):
    """多尺度时序汇总 + SwiGLU 门控残差 MLP 因子模型。"""

    def __init__(
        self,
        n_features: int,
        seq_len: int = 20,
        hidden_dim: int = 32,
        n_layers: int = 2,
        dropout: float = 0.35,
        n_symbols: int = 0,
        n_sectors: int = 0,
        use_symbol_emb: bool = False,
    ) -> None:
        super().__init__()
        self.n_features = int(n_features)
        self.seq_len = int(seq_len)
        self.hidden_dim = int(hidden_dim)

        # 多尺度因果 summary: last + mean_all + mean_short5 + std_all => 4 * C 维
        in_summary_dim = 4 * self.n_features
        emb_dim = 0
        self.sym_emb: nn.Embedding | None = None
        self.sec_emb: nn.Embedding | None = None
        if use_symbol_emb and n_symbols > 0:
            sym_d = max(4, min(16, hidden_dim // 4))
            self.sym_emb = nn.Embedding(n_symbols, sym_d)
            emb_dim += sym_d
        if n_sectors > 1:
            sec_d = max(4, min(12, hidden_dim // 4))
            self.sec_emb = nn.Embedding(n_sectors, sec_d)
            emb_dim += sec_d

        self.in_norm = nn.LayerNorm(in_summary_dim)
        self.in_proj = nn.Linear(in_summary_dim + emb_dim, hidden_dim)
        self.in_drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList(
            [_SwiGLUBlock(hidden_dim, dropout=dropout) for _ in range(max(1, n_layers))]
        )
        self.out_norm = nn.LayerNorm(hidden_dim)
        self.head = nn.Linear(hidden_dim, 1)

    def _summarize_window(self, x_seq: torch.Tensor) -> torch.Tensor:
        """从 `(M, L, C)` 窗口提取严格因果的多尺度汇总特征 `(M, 4*C)`。"""
        last = x_seq[:, -1, :]
        mean_all = x_seq.mean(dim=1)
        short_k = min(5, x_seq.shape[1])
        mean_short = x_seq[:, -short_k:, :].mean(dim=1)
        std_all = x_seq.std(dim=1, unbiased=False)
        return torch.cat([last, mean_all, mean_short, std_all], dim=-1)

    def forward(
        self,
        x: torch.Tensor,
        *,
        symbol_ids: torch.Tensor | None = None,
        sector_ids: torch.Tensor | None = None,
        valid_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del valid_mask
        is_day_batch = x.dim() == 4
        if is_day_batch:
            b_dim, n_dim, l_dim, c_dim = x.shape
            x_flat = x.reshape(b_dim * n_dim, l_dim, c_dim)
        else:
            b_dim, n_dim = 1, x.shape[0]
            x_flat = x

        feat = self.in_norm(self._summarize_window(x_flat))
        parts = [feat]

        if self.sym_emb is not None and symbol_ids is not None:
            sid = symbol_ids.reshape(-1)
            if is_day_batch and sid.shape[0] == n_dim:
                sid = sid.unsqueeze(0).expand(b_dim, n_dim).reshape(-1)
            parts.append(self.sym_emb(sid.clamp(min=0, max=self.sym_emb.num_embeddings - 1)))

        if self.sec_emb is not None and sector_ids is not None:
            scid = sector_ids.reshape(-1)
            if is_day_batch and scid.shape[0] == n_dim:
                scid = scid.unsqueeze(0).expand(b_dim, n_dim).reshape(-1)
            parts.append(self.sec_emb(scid.clamp(min=0, max=self.sec_emb.num_embeddings - 1)))

        h = self.in_drop(F.gelu(self.in_proj(torch.cat(parts, dim=-1))))
        for blk in self.blocks:
            h = blk(h)
        out = self.head(self.out_norm(h)).squeeze(-1)
        return out.reshape(b_dim, n_dim) if is_day_batch else out
