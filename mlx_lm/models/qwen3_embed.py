"""Qwen3 embedding model for MLX.

This adapts the generation-oriented Qwen3 architecture into an embedding-only
variant compatible with the `Qwen/Qwen3-Embedding-8B` checkpoint.

The implementation is identical to the previous `qwen2_embed.py` scaffold but
renamed to follow the official model designation.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Dict, Union

import mlx.core as mx
import mlx.nn as nn

from .base import (
    BaseModelArgs,
    create_attention_mask,
    scaled_dot_product_attention,
)
from .rope_utils import initialize_rope


@dataclass
class ModelArgs(BaseModelArgs):
    """Arguments mirror HF `config.json` for Qwen3-Embedding-8B."""

    model_type: str
    hidden_size: int
    num_hidden_layers: int
    intermediate_size: int
    num_attention_heads: int
    rms_norm_eps: float
    vocab_size: int
    num_key_value_heads: int
    head_dim: int = 128

    # Extra projector dims
    projector_dim: int = 1024

    # Rope params
    max_position_embeddings: int = 32768
    rope_theta: float = 1_000_000
    rope_traditional: bool = False
    rope_scaling: Optional[Dict[str, Union[float, str]]] = None

    tie_word_embeddings: bool = True  # unused here


class Attention(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        dim = args.hidden_size
        self.n_heads = n_heads = args.num_attention_heads
        self.n_kv_heads = n_kv_heads = args.num_key_value_heads
        # Qwen3-Embedding-4B stores a fixed head_dim (128) even when
        # `hidden_size / num_attention_heads` differs.  Prefer `head_dim` from
        # config if available, otherwise fall back to dim // n_heads.
        head_dim = getattr(args, "head_dim", dim // n_heads)
        self.scale = head_dim ** -0.5

        self.q_proj = nn.Linear(dim, n_heads * head_dim, bias=True)
        self.k_proj = nn.Linear(dim, n_kv_heads * head_dim, bias=True)
        self.v_proj = nn.Linear(dim, n_kv_heads * head_dim, bias=True)
        # Output projection back to hidden_size (dim)
        self.o_proj = nn.Linear(n_heads * head_dim, dim, bias=False)

        self.rope = initialize_rope(
            head_dim,
            base=args.rope_theta,
            traditional=args.rope_traditional,
            scaling_config=args.rope_scaling,
            max_position_embeddings=args.max_position_embeddings,
        )

    def __call__(self, x: mx.array, mask: Optional[mx.array] = None, cache: Optional[Any] = None):
        B, L, _ = x.shape
        q = self.q_proj(x).reshape(B, L, self.n_heads, -1).transpose(0, 2, 1, 3)
        k = self.k_proj(x).reshape(B, L, self.n_kv_heads, -1).transpose(0, 2, 1, 3)
        v = self.v_proj(x).reshape(B, L, self.n_kv_heads, -1).transpose(0, 2, 1, 3)

        if cache is not None:
            q = self.rope(q, offset=cache.offset)
            k = self.rope(k, offset=cache.offset)
            k, v = cache.update_and_fetch(k, v)
        else:
            q = self.rope(q)
            k = self.rope(k)

        out = scaled_dot_product_attention(q, k, v, cache=cache, scale=self.scale, mask=mask)
        out = out.transpose(0, 2, 1, 3).reshape(B, L, -1)
        # Project back to hidden dimension
        out = self.o_proj(out)
        return out


class MLP(nn.Module):
    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        self.gate_proj = nn.Linear(dim, hidden_dim, bias=False)
        self.down_proj = nn.Linear(hidden_dim, dim, bias=False)
        self.up_proj = nn.Linear(dim, hidden_dim, bias=False)

    def __call__(self, x):
        return self.down_proj(nn.silu(self.gate_proj(x)) * self.up_proj(x))


class TransformerBlock(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.self_attn = Attention(args)
        self.mlp = MLP(args.hidden_size, args.intermediate_size)
        self.input_ln = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.post_attn_ln = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)

    def __call__(self, x, mask=None, cache=None):
        h = x + self.self_attn(self.input_ln(x), mask, cache)
        out = h + self.mlp(self.post_attn_ln(h))
        return out


class Qwen3EmbedModel(nn.Module):
    """Encoder-only forward that returns pooled, L2-normalised embeddings."""

    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.embed_tokens = nn.Embedding(args.vocab_size, args.hidden_size)
        self.layers = [TransformerBlock(args) for _ in range(args.num_hidden_layers)]
        self.final_ln = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        # Project hidden_size → projector_dim (1024)
        self.projector = nn.Linear(args.hidden_size, args.projector_dim, bias=False)

    # ------------------------------------------------------------------
    # HF compatibility helpers (input_ids + attention_mask)
    # ------------------------------------------------------------------
    def __call__(self, input_ids: mx.array, attention_mask: Optional[mx.array] = None, cache=None):
        B, L = input_ids.shape
        h = self.embed_tokens(input_ids)
        attn_mask = create_attention_mask(attention_mask, L) if attention_mask is not None else None
        for layer in self.layers:
            h = layer(h, attn_mask, cache)
        h = self.final_ln(h)

        # Pool last token (assumes left-padding aware pre-processing)
        pooled = h[:, -1, :]
        emb = self.projector(pooled)
        # L2 normalise
        emb = emb / mx.sqrt(mx.sum(emb ** 2, axis=-1, keepdims=True))
        return {"embeddings": emb}


# Alias for MLX-LM dynamic loader
Model = Qwen3EmbedModel
