"""DL Alpha 分市场模型族注册中心与构建工厂 (dl/models/__init__.py)。"""

from __future__ import annotations

from typing import Any

import torch.nn as nn

from factorlib.dl.models.gated_mlp import GatedMLPAlpha
from factorlib.dl.models.gru import GRUAlpha
from factorlib.dl.models.st_attention import SpatioTemporalAttentionAlpha
from factorlib.dl.models.tcn import CausalTCNAlpha

MODEL_ARCHITECTURES: tuple[str, ...] = ("gated_mlp", "gru", "tcn", "st_attention")


def normalize_model_arch(raw_name: str | None, market: str = "futures") -> str:
    """将 Web/Recipe 中的展示名或别名规范化为标准模型族 key。"""
    s = str(raw_name or "").strip().lower()
    if not s:
        return "gru" if market == "futures" else "st_attention"
    if "mlp" in s or "gated" in s:
        return "gated_mlp"
    if "tcn" in s:
        return "tcn"
    if "attn" in s or "attention" in s or "cross" in s or "st_" in s or "spacetime" in s:
        return "st_attention"
    if "gru" in s:
        return "gru"
    return "gru" if market == "futures" else "st_attention"


def default_model_hparams(market: str, arch: str) -> dict[str, Any]:
    """按架构设计 §5.2 返回受市场有效截面约束的超参默认模板。"""
    norm_m = str(market).strip().lower()
    if norm_m == "futures":
        # 小截面期货防过拟合：轻量隐层 + 高 Dropout + 品种/板块嵌入
        base = {
            "hidden_dim": 32,
            "n_layers": 1 if arch == "gru" else 2,
            "dropout": 0.40,
            "weight_decay": 1e-3,
            "lr": 1.5e-3,
            "batch_days": 16,
            "use_symbol_emb": True,
        }
    else:
        # 大截面 A 股：禁用个股 ID 过拟合，默认时空解耦或中容量 GRU/TCN，按日批训练
        base = {
            "hidden_dim": 64,
            "n_layers": 2 if arch in ("gated_mlp", "tcn") else 1,
            "dropout": 0.25,
            "weight_decay": 5e-4,
            "lr": 1e-3,
            "batch_days": 2,
            "use_symbol_emb": False,
        }
    return base


def build_dl_model(
    arch: str,
    *,
    n_features: int,
    market: str = "futures",
    seq_len: int = 20,
    hidden_dim: int | None = None,
    n_layers: int | None = None,
    dropout: float | None = None,
    n_symbols: int = 0,
    n_sectors: int = 0,
    use_symbol_emb: bool | None = None,
) -> nn.Module:
    """根据标准架构名称与市场策略实例化 PyTorch DL Alpha 模型。"""
    key = normalize_model_arch(arch, market=market)
    defaults = default_model_hparams(market=market, arch=key)
    h_dim = int(hidden_dim if hidden_dim is not None else defaults["hidden_dim"])
    layers = int(n_layers if n_layers is not None else defaults["n_layers"])
    drop = float(dropout if dropout is not None else defaults["dropout"])
    sym_emb = bool(use_symbol_emb if use_symbol_emb is not None else defaults["use_symbol_emb"])

    common = {
        "n_features": int(n_features),
        "seq_len": int(seq_len),
        "hidden_dim": h_dim,
        "n_layers": layers,
        "dropout": drop,
        "n_symbols": int(n_symbols),
        "n_sectors": int(n_sectors),
        "use_symbol_emb": sym_emb,
    }
    if key == "gated_mlp":
        return GatedMLPAlpha(**common)
    if key == "gru":
        return GRUAlpha(**common)
    if key == "tcn":
        return CausalTCNAlpha(**common)
    if key == "st_attention":
        return SpatioTemporalAttentionAlpha(**common)
    raise ValueError(f"未知的 DL 模型架构: {arch!r}")


__all__ = [
    "CausalTCNAlpha",
    "GatedMLPAlpha",
    "GRUAlpha",
    "MODEL_ARCHITECTURES",
    "SpatioTemporalAttentionAlpha",
    "build_dl_model",
    "default_model_hparams",
    "normalize_model_arch",
]
