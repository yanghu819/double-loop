from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
from einops import rearrange
from torch.nn import functional as F

from fla.layers.utils import get_layer_cache, update_layer_cache
from fla.ops.gdn2 import chunk_gdn2, fused_recurrent_gdn2

from experiments.zoology_mqar.gdn2_futureseed import ZoologyGDN2FutureSeedMixer


class DualHashGatedDeltaNet2(nn.Module):
    """Encode one payload under two equal-weight address hashes per head."""

    def __init__(self, base: nn.Module) -> None:
        super().__init__()
        if base.num_heads != 4 or base.head_k_dim != 32:
            raise ValueError("P-GDN3-024 requires exactly H4/K32")
        if base.num_v_heads != 4 or base.head_v_dim != 32:
            raise ValueError("P-GDN3-024 requires exactly H4/V32")
        if base.hidden_size != 128:
            raise ValueError("P-GDN3-024 requires the fixed D128 carrier")
        self.base = base
        self.hashes = 2
        self.hash_dim = base.head_k_dim // self.hashes
        self.last_diagnostics: dict[str, torch.Tensor] = {}

    @property
    def num_heads(self) -> int:
        return int(self.base.num_heads)

    @property
    def num_v_heads(self) -> int:
        return int(self.base.num_v_heads)

    @property
    def head_k_dim(self) -> int:
        return int(self.base.head_k_dim)

    @property
    def head_v_dim(self) -> int:
        return int(self.base.head_v_dim)

    def _encode_hashes(self, tensor: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        banks = tensor.float().reshape(
            *tensor.shape[:-1],
            self.hashes,
            self.hash_dim,
        )
        banks = F.normalize(banks, dim=-1, eps=1e-6)
        first, second = banks.unbind(dim=-2)
        indices = torch.arange(self.hash_dim, device=tensor.device)
        second = second.index_select(-1, (5 * indices + 1) % self.hash_dim)
        signs = torch.where(
            indices % 2 == 0,
            torch.ones_like(indices, dtype=torch.float32),
            -torch.ones_like(indices, dtype=torch.float32),
        )
        sketch = F.normalize(first * (second * signs), dim=-1, eps=1e-6)
        encoded = torch.cat((sketch, sketch.roll(3, dims=-1)), dim=-1)
        encoded = F.normalize(encoded, dim=-1, eps=1e-6)
        return encoded.to(dtype=tensor.dtype), banks

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        past_key_values: Any = None,
        use_cache: bool | None = False,
        output_attentions: bool | None = False,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, None, Any]:
        del output_attentions
        if attention_mask is not None:
            raise ValueError("P-GDN3-024 does not use padded MQAR batches")

        layer = self.base
        q_len = hidden_states.shape[1]
        mode = "fused_recurrent" if (q_len <= 64 and not self.training) else layer.mode
        if self.training and mode != "chunk":
            raise RuntimeError("P-GDN3-024 training must use the official chunk kernel")

        last_state = get_layer_cache(layer, past_key_values)
        cu_seqlens = kwargs.get("cu_seqlens")
        if cu_seqlens is not None:
            raise ValueError("P-GDN3-024 uses fixed unpadded sequences")
        conv_state_q, conv_state_k, conv_state_v = None, None, None
        if last_state is not None:
            conv_state_q, conv_state_k, conv_state_v = last_state["conv_state"]
        q, conv_state_q = layer.q_conv1d(
            x=layer.q_proj(hidden_states),
            cache=conv_state_q,
            output_final_state=use_cache,
        )
        k, conv_state_k = layer.k_conv1d(
            x=layer.k_proj(hidden_states),
            cache=conv_state_k,
            output_final_state=use_cache,
        )
        v, conv_state_v = layer.v_conv1d(
            x=layer.v_proj(hidden_states),
            cache=conv_state_v,
            output_final_state=use_cache,
        )

        g = F.softplus(layer.f_proj(hidden_states).float() + layer.dt_bias)
        b = layer.b_proj(hidden_states).sigmoid()
        w = layer.w_proj(hidden_states).sigmoid()
        q, k, g = (
            rearrange(tensor, "... (h d) -> ... h d", d=layer.head_k_dim)
            for tensor in (q, k, g)
        )
        v = rearrange(v, "... (h d) -> ... h d", d=layer.head_v_dim)
        b = rearrange(b, "... (h d) -> ... h d", d=layer.head_k_dim)
        w = rearrange(w, "... (h d) -> ... h d", d=layer.head_v_dim)
        g = -layer.A_log.float().exp().unsqueeze(-1) * g

        q, q_banks = self._encode_hashes(q)
        k, k_banks = self._encode_hashes(k)
        if layer.allow_neg_eigval:
            b = b * 2.0

        recurrent_state = last_state["recurrent_state"] if last_state is not None else None
        operation = chunk_gdn2 if mode == "chunk" else fused_recurrent_gdn2
        output, recurrent_state = operation(
            q=q,
            k=k,
            v=v,
            g=g,
            b=b,
            w=w,
            initial_state=recurrent_state,
            output_final_state=use_cache,
            use_qk_l2norm_in_kernel=True,
        )
        update_layer_cache(
            layer,
            past_key_values,
            recurrent_state=recurrent_state,
            conv_state=(conv_state_q, conv_state_k, conv_state_v),
            offset=q_len,
        )

        with torch.no_grad():
            q_cosine = (q_banks[..., 0, :] * q_banks[..., 1, :]).sum(dim=-1)
            k_cosine = (k_banks[..., 0, :] * k_banks[..., 1, :]).sum(dim=-1)
            q_norms = q.reshape(*q.shape[:-1], self.hashes, self.hash_dim).float().norm(dim=-1)
            k_norms = k.reshape(*k.shape[:-1], self.hashes, self.hash_dim).float().norm(dim=-1)
            self.last_diagnostics = {
                "q_hash_abs_cosine": q_cosine.abs().mean(),
                "k_hash_abs_cosine": k_cosine.abs().mean(),
                "q_hash_token_std": q_cosine.mean(dim=-1).std(unbiased=False),
                "k_hash_token_std": k_cosine.mean(dim=-1).std(unbiased=False),
                "q_hash_norm_imbalance_max": (
                    q_norms[..., 0] - q_norms[..., 1]
                ).abs().max(),
                "k_hash_norm_imbalance_max": (
                    k_norms[..., 0] - k_norms[..., 1]
                ).abs().max(),
            }

        output = layer.o_norm(
            output,
            rearrange(
                layer.g_proj(hidden_states),
                "... (h d) -> ... h d",
                d=layer.head_v_dim,
            ),
        )
        output = rearrange(output, "b t h d -> b t (h d)")
        return layer.o_proj(output), None, past_key_values


class ZoologyDualHashGDN2FutureSeedMixer(ZoologyGDN2FutureSeedMixer):
    """Native FutureSeed over block-contiguous dual-hash GDN2 memory."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.layer = DualHashGatedDeltaNet2(self.layer)


def dual_hash_diagnostics(model: nn.Module) -> dict[str, Any]:
    mixers = [
        block.sequence_mixer
        for block in model.backbone.layers
        if isinstance(block.sequence_mixer, ZoologyDualHashGDN2FutureSeedMixer)
    ]
    if len(mixers) != 2:
        raise RuntimeError(f"Expected two active dual-hash mixers, got {len(mixers)}")
    rows = []
    for mixer in mixers:
        diagnostics = mixer.layer.last_diagnostics
        if not diagnostics:
            raise RuntimeError("Dual-hash diagnostics were not populated")
        rows.append(
            {
                "layer_idx": mixer.layer_idx,
                **{name: float(value.item()) for name, value in diagnostics.items()},
            }
        )
    return {
        "active_layers": len(rows),
        "physical_heads": 4,
        "hashes_per_head": 2,
        "hash_dim": 16,
        "state_values_per_layer": 4 * 32 * 32,
        "parameter_delta": 0,
        "official_scans_per_layer": 1,
        "per_layer": rows,
    }
