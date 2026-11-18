"""时空解耦双阶段注意力网络 (dl/models/st_attention.py)。

strictly aligns with Design §5.2:
1. **Stage 1（时间编码器）**：共享参数的因果 GRU 提取单标的 `[t-L+1 .. t]` 时序表征；
2. **Stage 2（横截面交互层）**：
   - 当截面规模适中（如期货 `N <= 256`）时，执行严格屏蔽无效标的的
     `Sector-Aware Cross-Sectional MultiheadAttention`；
   - 当截面规模极大（如 A 股全市场 `N ≈ 3000~5000`）时，自动启用 `O(N · M)` 复杂度
     **Induced Market-Guided Routing Attention**（通过 `M=32` 个可学习市场原型 token 双向汇总/广播），
     在捕捉行业轮动与拥挤度状态的同时将单批显存压低至数十 MB 内。
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SpatioTemporalAttentionAlpha(nn.Module):
    """时空解耦双阶段注意力 Alpha 模型（适配小截面期货与大截面 A 股）。"""

    def __init__(
        self,
        n_features: int,
        seq_len: int = 20,
        hidden_dim: int = 48,
        n_layers: int = 1,
        n_heads: int = 4,
        n_routing_tokens: int = 32,
        direct_attn_threshold: int = 256,
        dropout: float = 0.25,
        n_symbols: int = 0,
        n_sectors: int = 0,
        use_symbol_emb: bool = False,
    ) -> None:
        super().__init__()
        del seq_len
        self.hidden_dim = int(hidden_dim)
        self.direct_attn_threshold = int(direct_attn_threshold)

        # Stage 1: 共享单标的因果 GRU
        self.in_norm = nn.LayerNorm(n_features)
        self.in_proj = nn.Linear(n_features, hidden_dim)
        self.in_drop = nn.Dropout(dropout)
        n_gru = max(1, min(2, int(n_layers)))
        self.gru = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=n_gru,
            batch_first=True,
            dropout=dropout if n_gru > 1 else 0.0,
            bidirectional=False,
        )

        # 可选品种与板块嵌入
        self.sym_emb: nn.Embedding | None = None
        self.sec_emb: nn.Embedding | None = None
        if use_symbol_emb and n_symbols > 0:
            self.sym_emb = nn.Embedding(n_symbols, hidden_dim)
        if n_sectors > 1:
            self.sec_emb = nn.Embedding(n_sectors, hidden_dim)

        # Stage 2: 截面交互注意力 (Exact MHA for small N / Market-Guided Routing for large N)
        eff_heads = max(1, min(n_heads, hidden_dim))
        while hidden_dim % eff_heads != 0 and eff_heads > 1:
            eff_heads -= 1

        self.cs_norm1 = nn.LayerNorm(hidden_dim)
        self.cs_mha = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=eff_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.routing_tokens = nn.Parameter(torch.randn(1, n_routing_tokens, hidden_dim) * 0.02)
        self.route_in = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=eff_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.route_out = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=eff_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.cs_drop = nn.Dropout(dropout)

        self.ffn_norm = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Dropout(dropout),
        )
        self.head = nn.Linear(hidden_dim, 1)

    def _cross_sectional_interact(
        self,
        h_bn: torch.Tensor,
        valid_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        b_dim, n_dim, _ = h_bn.shape
        x = self.cs_norm1(h_bn)
        key_pad = ~valid_mask.bool() if valid_mask is not None else None

        # 保护：若某行所有标的都被遮蔽，避免 MultiheadAttention 全 True key_padding_mask 产生 NaN
        if key_pad is not None:
            all_masked = key_pad.all(dim=1)
            if bool(all_masked.any().item()):
                key_pad = key_pad.clone()
                key_pad[all_masked, 0] = False

        if n_dim <= self.direct_attn_threshold:
            attn_out, _ = self.cs_mha(x, x, x, key_padding_mask=key_pad, need_weights=False)
        else:
            # 市场原型路由：O(N * M) 两跳交叉注意力，兼容 3000~5000 支股票
            m_tokens = self.routing_tokens.expand(b_dim, -1, -1)
            market_ctx, _ = self.route_in(
                m_tokens, x, x, key_padding_mask=key_pad, need_weights=False
            )
            attn_out, _ = self.route_out(x, market_ctx, market_ctx, need_weights=False)

        h = h_bn + self.cs_drop(attn_out)
        h = h + self.ffn(self.ffn_norm(h))
        return h

    def forward(
        self,
        x: torch.Tensor,
        *,
        symbol_ids: torch.Tensor | None = None,
        sector_ids: torch.Tensor | None = None,
        valid_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        is_day_batch = x.dim() == 4
        if is_day_batch:
            b_dim, n_dim, l_dim, c_dim = x.shape
            x_flat = x.reshape(b_dim * n_dim, l_dim, c_dim)
        else:
            b_dim, n_dim = 1, x.shape[0]
            x_flat = x

        h_in = self.in_drop(F.gelu(self.in_proj(self.in_norm(x_flat))))
        seq_out, _ = self.gru(h_in)
        h_last = seq_out[:, -1, :]

        if self.sym_emb is not None and symbol_ids is not None:
            sid = symbol_ids.reshape(-1)
            if is_day_batch and sid.shape[0] == n_dim:
                sid = sid.unsqueeze(0).expand(b_dim, n_dim).reshape(-1)
            h_last = h_last + 0.2 * self.sym_emb(
                sid.clamp(min=0, max=self.sym_emb.num_embeddings - 1)
            )

        if self.sec_emb is not None and sector_ids is not None:
            scid = sector_ids.reshape(-1)
            if is_day_batch and scid.shape[0] == n_dim:
                scid = scid.unsqueeze(0).expand(b_dim, n_dim).reshape(-1)
            h_last = h_last + 0.2 * self.sec_emb(
                scid.clamp(min=0, max=self.sec_emb.num_embeddings - 1)
            )

        if is_day_batch:
            h_bn = h_last.reshape(b_dim, n_dim, self.hidden_dim)
            h_cs = self._cross_sectional_interact(h_bn, valid_mask=valid_mask)
            return self.head(h_cs).squeeze(-1)

        h_res = h_last + self.ffn(self.ffn_norm(h_last))
        return self.head(h_res).squeeze(-1)
