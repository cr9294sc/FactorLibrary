"""轻量因果 GRU 时序因子网络 (dl/models/gru.py)。

设计要点：
- 单向因果 `nn.GRU(batch_first=True)`（1~2 层），绝不使用 `bidirectional=True`；
- 融合末端隐藏态 $h_t$ 与可学习时序注意力加权池化，兼顾近期动量与历史状态；
- 支持期货小截面品种/板块低维嵌入，搭配高 Dropout 抑制金融低信噪比过拟合。
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class GRUAlpha(nn.Module):
    """因果单向 GRU + 注意力时间池化 Alpha 模型。"""

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
        del seq_len
        self.hidden_dim = int(hidden_dim)
        self.in_norm = nn.LayerNorm(n_features)
        self.in_proj = nn.Linear(n_features, hidden_dim)
        self.in_drop = nn.Dropout(dropout)

        n_gru = max(1, min(3, int(n_layers)))
        self.gru = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=n_gru,
            batch_first=True,
            dropout=dropout if n_gru > 1 else 0.0,
            bidirectional=False,
        )
        self.time_attn = nn.Linear(hidden_dim, 1)

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

        self.out_norm = nn.LayerNorm(hidden_dim * 2 + emb_dim)
        self.head_mlp = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2 + emb_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def encode_temporal(
        self,
        x_flat: torch.Tensor,
        *,
        symbol_ids: torch.Tensor | None = None,
        sector_ids: torch.Tensor | None = None,
        b_dim: int = 1,
        n_dim: int = 1,
        is_day_batch: bool = False,
    ) -> torch.Tensor:
        h_in = self.in_drop(F.gelu(self.in_proj(self.in_norm(x_flat))))
        seq_out, _ = self.gru(h_in)  # (M, L, H)
        last_h = seq_out[:, -1, :]
        attn_w = torch.softmax(self.time_attn(seq_out).squeeze(-1), dim=1).unsqueeze(-1)
        ctx_h = (seq_out * attn_w).sum(dim=1)
        parts = [last_h, ctx_h]

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

        return self.out_norm(torch.cat(parts, dim=-1))

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

        rep = self.encode_temporal(
            x_flat,
            symbol_ids=symbol_ids,
            sector_ids=sector_ids,
            b_dim=b_dim,
            n_dim=n_dim,
            is_day_batch=is_day_batch,
        )
        out = self.head_mlp(rep).squeeze(-1)
        return out.reshape(b_dim, n_dim) if is_day_batch else out
