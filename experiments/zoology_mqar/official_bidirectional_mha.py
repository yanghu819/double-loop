from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from zoology.mixers.attention import MHA, SelfAttention


class NonCausalSelfAttention(SelfAttention):
    """The upstream Zoology attention equation without its causal mask."""

    def forward(self, qkv: torch.Tensor) -> torch.Tensor:
        query, key, value = qkv.unbind(dim=2)
        softmax_scale = 1.0 / math.sqrt(query.shape[-1])
        scores = torch.einsum(
            "bthd,bshd->bhts",
            query,
            key * softmax_scale,
        )
        attention = torch.softmax(scores, dim=-1, dtype=value.dtype)
        attention = F.dropout(
            attention,
            self.dropout_p if self.training else 0.0,
        )
        return torch.einsum("bhts,bshd->bthd", attention, value)


class OfficialBidirectionalMHA(MHA):
    """Exact upstream MHA parameters with only attention direction changed."""

    def __init__(
        self,
        d_model: int,
        num_heads: int = 1,
        bias: bool = True,
        dropout: float = 0.0,
        layer_idx: int | None = None,
    ) -> None:
        super().__init__(
            d_model=d_model,
            num_heads=num_heads,
            bias=bias,
            dropout=dropout,
            layer_idx=layer_idx,
        )
        self.inner_attn = NonCausalSelfAttention(attention_dropout=dropout)
