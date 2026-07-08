"""Stage 2 — Causal multi-head self-attention.

Scaled dot-product attention with a causal mask:

    Attention(Q, K, V) = softmax(Q K^T / sqrt(d_k) + M) V

where M sets all positions above the diagonal to -inf so that a token can
only attend to itself and to earlier tokens. Multi-head attention runs
``n_heads`` such attentions in parallel subspaces and concatenates the
results — implemented efficiently here with a single fused Q/K/V projection
per head group and tensor reshaping (no Python loop over heads).
"""

from __future__ import annotations

import torch
import torch.nn as nn


class MultiHeadAttention(nn.Module):
    def __init__(
        self,
        d_in: int,
        d_out: int,
        context_length: int,
        dropout: float,
        num_heads: int,
        qkv_bias: bool = False,
    ) -> None:
        super().__init__()
        if d_out % num_heads != 0:
            raise ValueError("d_out must be divisible by num_heads")

        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads

        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.out_proj = nn.Linear(d_out, d_out)
        self.dropout = nn.Dropout(dropout)

        # Upper-triangular causal mask, cached as a (non-trainable) buffer.
        self.register_buffer(
            "mask",
            torch.triu(torch.ones(context_length, context_length), diagonal=1).bool(),
            persistent=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, num_tokens, _ = x.shape

        # (b, T, d_out) -> (b, n_heads, T, head_dim)
        def split_heads(t: torch.Tensor) -> torch.Tensor:
            return t.view(b, num_tokens, self.num_heads, self.head_dim).transpose(1, 2)

        queries = split_heads(self.W_query(x))
        keys = split_heads(self.W_key(x))
        values = split_heads(self.W_value(x))

        # Scaled dot-product scores: (b, n_heads, T, T)
        attn_scores = queries @ keys.transpose(2, 3) / keys.shape[-1] ** 0.5
        attn_scores = attn_scores.masked_fill(
            self.mask[:num_tokens, :num_tokens], float("-inf")
        )
        attn_weights = torch.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Weighted sum of values, heads re-assembled: (b, T, d_out)
        context = (attn_weights @ values).transpose(1, 2)
        context = context.contiguous().view(b, num_tokens, self.d_out)
        return self.out_proj(context)
