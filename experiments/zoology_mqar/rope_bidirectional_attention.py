from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class ParamMatchedRoPEBidirectionalAttention(nn.Module):
    """Full SDPA attention with parameter-free relative rotary addressing."""

    def __init__(
        self,
        d_model: int,
        layer_idx: int,
        *,
        num_heads: int = 4,
        head_dim: int = 57,
        rope_theta: float = 10_000.0,
        rope_scale: float = 1.0,
    ) -> None:
        super().__init__()
        del layer_idx
        if num_heads <= 0 or head_dim <= 1:
            raise ValueError("num_heads must be positive and head_dim must exceed one")
        self.num_heads = int(num_heads)
        self.head_dim = int(head_dim)
        self.attention_dim = self.num_heads * self.head_dim
        self.rotary_dim = self.head_dim - self.head_dim % 2
        self.rope_scale = float(rope_scale)
        self.qkv = nn.Linear(d_model, 3 * self.attention_dim, bias=False)
        self.out_proj = nn.Linear(self.attention_dim, d_model, bias=False)
        inv_freq = 1.0 / (
            float(rope_theta)
            ** (
                torch.arange(0, self.rotary_dim, 2, dtype=torch.float32)
                / self.rotary_dim
            )
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def _apply_rope(self, tensor: torch.Tensor) -> torch.Tensor:
        if self.rope_scale == 0.0:
            return tensor
        positions = torch.arange(
            tensor.shape[1],
            device=tensor.device,
            dtype=torch.float32,
        )
        angles = torch.outer(positions * self.rope_scale, self.inv_freq.float())
        cosine = angles.cos().to(dtype=tensor.dtype)[None, :, None, :]
        sine = angles.sin().to(dtype=tensor.dtype)[None, :, None, :]
        rotary = tensor[..., : self.rotary_dim].reshape(
            *tensor.shape[:-1],
            self.rotary_dim // 2,
            2,
        )
        even, odd = rotary.unbind(dim=-1)
        rotated = torch.stack(
            (even * cosine - odd * sine, even * sine + odd * cosine),
            dim=-1,
        ).flatten(-2)
        if self.rotary_dim == self.head_dim:
            return rotated
        return torch.cat((rotated, tensor[..., self.rotary_dim :]), dim=-1)

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
            query = self._apply_rope(query)
            key = self._apply_rope(key)
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
