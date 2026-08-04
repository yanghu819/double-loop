from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class FullBidirectionalAttention(nn.Module):
    """A minimal noncausal SDPA mixer used only as the quality ceiling."""

    def __init__(
        self,
        d_model: int,
        layer_idx: int,
        *,
        num_heads: int = 4,
    ) -> None:
        super().__init__()
        del layer_idx
        if d_model % num_heads:
            raise ValueError("d_model must be divisible by num_heads")
        self.num_heads = int(num_heads)
        self.head_dim = d_model // num_heads
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch, length, width = hidden_states.shape
        qkv = self.qkv(hidden_states).view(
            batch, length, 3, self.num_heads, self.head_dim
        )
        query, key, value = qkv.unbind(dim=2)
        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)
        output = F.scaled_dot_product_attention(
            query,
            key,
            value,
            dropout_p=0.0,
            is_causal=False,
        )
        output = output.transpose(1, 2).reshape(batch, length, width)
        return self.out_proj(output)

    def state_size(self, sequence_length: int = 2048) -> int:
        return sequence_length * self.num_heads * self.head_dim * 2
