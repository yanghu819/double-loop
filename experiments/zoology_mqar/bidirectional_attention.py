from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class ParamMatchedBidirectionalAttention(nn.Module):
    """Full noncausal attention ceiling with near-matched mixer parameters."""

    def __init__(
        self,
        d_model: int,
        layer_idx: int,
        *,
        num_heads: int = 4,
        head_dim: int = 57,
    ) -> None:
        super().__init__()
        del layer_idx
        if num_heads <= 0 or head_dim <= 0:
            raise ValueError("num_heads and head_dim must be positive")
        self.num_heads = int(num_heads)
        self.head_dim = int(head_dim)
        self.attention_dim = self.num_heads * self.head_dim
        self.qkv = nn.Linear(d_model, 3 * self.attention_dim, bias=False)
        self.out_proj = nn.Linear(self.attention_dim, d_model, bias=False)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch, length, _width = hidden_states.shape
        with torch.autocast(
            device_type=hidden_states.device.type,
            dtype=torch.bfloat16,
            enabled=hidden_states.is_cuda,
        ):
            qkv = self.qkv(hidden_states).view(
                batch,
                length,
                3,
                self.num_heads,
                self.head_dim,
            )
            query, key, value = qkv.unbind(dim=2)
            output = F.scaled_dot_product_attention(
                query.transpose(1, 2),
                key.transpose(1, 2),
                value.transpose(1, 2),
                dropout_p=0.0,
                is_causal=False,
            )
            output = output.transpose(1, 2).reshape(
                batch,
                length,
                self.attention_dim,
            )
            output = self.out_proj(output)
        return output.to(hidden_states.dtype)

    def state_size(self, sequence_length: int = 2048) -> int:
        return sequence_length * self.attention_dim * 2
