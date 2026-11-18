"""严格因果扩张一维卷积网络 (dl/models/tcn.py)。

设计要点：
- 每个膨胀卷积块仅在时间维度左侧补零 `(kernel_size - 1) * dilation`，绝不向未来步向右扩展，
  从而保证 $t$ 时刻隐藏表征严格独立于未来时刻特征；
- 残差连接 + LayerNorm + GELU + Dropout，支持多感受野时序动态捕获。
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class _CausalDilatedBlock(nn.Module):
    def __init__(self, channels: int, kernel_size: int, dilation: int, dropout: float) -> None:
        super().__init__()
        self.left_pad = (kernel_size - 1) * dilation
        self.conv1 = nn.Conv1d(channels, channels, kernel_size, dilation=dilation)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size, dilation=dilation)
        self.norm1 = nn.LayerNorm(channels)
        self.norm2 = nn.LayerNorm(channels)
        self.drop = nn.Dropout(dropout)

    def _causal_conv(self, conv: nn.Conv1d, x_bct: torch.Tensor) -> torch.Tensor:
        padded = F.pad(x_bct, (self.left_pad, 0))
        return conv(padded)

    def forward(self, x_bct: torch.Tensor) -> torch.Tensor:
        h = self._causal_conv(self.conv1, x_bct)
        h = self.norm1(h.transpose(1, 2)).transpose(1, 2)
        h = self.drop(F.gelu(h))

        h = self._causal_conv(self.conv2, h)
        h = self.norm2(h.transpose(1, 2)).transpose(1, 2)
        h = self.drop(F.gelu(h))
        return x_bct + h


class CausalTCNAlpha(nn.Module):
    """因果扩张 1D-TCN 时序 Alpha 模型。"""

    def __init__(
        self,
        n_features: int,
        seq_len: int = 20,
        hidden_dim: int = 32,
        n_layers: int = 3,
        kernel_size: int = 3,
        dropout: float = 0.35,
        n_symbols: int = 0,
        n_sectors: int = 0,
        use_symbol_emb: bool = False,
    ) -> None:
        super().__init__()
        del seq_len
        self.in_norm = nn.LayerNorm(n_features)
        self.in_proj = nn.Linear(n_features, hidden_dim)
        self.in_drop = nn.Dropout(dropout)

        blocks = []
        for idx in range(max(1, min(4, int(n_layers)))):
            blocks.append(
                _CausalDilatedBlock(
                    channels=hidden_dim,
                    kernel_size=kernel_size,
                    dilation=2**idx,
                    dropout=dropout,
                )
            )
        self.tcn = nn.Sequential(*blocks)

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

        self.out_norm = nn.LayerNorm(hidden_dim + emb_dim)
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_dim + emb_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

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

        h = self.in_drop(F.gelu(self.in_proj(self.in_norm(x_flat))))
        # Conv1d 输入布局 (M, H, L)
        h_tcn = self.tcn(h.transpose(1, 2))[:, :, -1]  # 取最新因果末步 (M, H)
        parts = [h_tcn]

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

        rep = self.out_norm(torch.cat(parts, dim=-1))
        out = self.head(rep).squeeze(-1)
        return out.reshape(b_dim, n_dim) if is_day_batch else out
