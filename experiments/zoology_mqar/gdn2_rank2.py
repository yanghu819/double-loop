from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
from einops import rearrange
from torch.nn import functional as F

from fla.layers.utils import get_layer_cache, update_layer_cache
from fla.ops.gdn2 import chunk_gdn2

from experiments.zoology_mqar.gdn2_futureseed import (
    ZoologyGDN2FutureSeedMixer,
)


class FixedInitProjection(nn.Module):
    """A learned projection whose initialization does not perturb parent RNG."""

    def __init__(self, width: int, *, seed: int) -> None:
        super().__init__()
        generator = torch.Generator(device="cpu")
        generator.manual_seed(seed)
        weight = torch.empty(width, width)
        weight.normal_(mean=0.0, std=0.02, generator=generator)
        self.weight = nn.Parameter(weight)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return F.linear(hidden_states, self.weight)


class TwoEditGatedDeltaNet2(nn.Module):
    """One official chunk scan with two learned rank-one edits per token."""

    def __init__(self, base: nn.Module) -> None:
        super().__init__()
        if base.num_v_heads != base.num_heads:
            raise ValueError("The first two-edit test requires matched QK/V heads")
        if base.key_dim != base.hidden_size or base.value_dim != base.hidden_size:
            raise ValueError("The first two-edit test requires D=H*K=H*V")
        self.base = base
        seed_base = 21_000 + 100 * int(base.layer_idx)
        self.aux_k_proj = FixedInitProjection(base.hidden_size, seed=seed_base + 1)
        self.aux_v_proj = FixedInitProjection(base.hidden_size, seed=seed_base + 2)
        self.aux_b_proj = FixedInitProjection(base.hidden_size, seed=seed_base + 3)
        self.aux_w_proj = FixedInitProjection(base.hidden_size, seed=seed_base + 4)
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

    @staticmethod
    def _interleave(first: torch.Tensor, second: torch.Tensor) -> torch.Tensor:
        batch, length = first.shape[:2]
        return torch.stack((first, second), dim=2).reshape(
            batch, 2 * length, *first.shape[2:]
        )

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
            raise ValueError("P-GDN3-021 does not use padded MQAR batches")

        layer = self.base
        q_len = hidden_states.shape[1]
        if layer.mode != "chunk":
            raise RuntimeError("P-GDN3-021 requires the official chunk kernel")
        last_state = get_layer_cache(layer, past_key_values)
        cu_seqlens = kwargs.get("cu_seqlens")
        if cu_seqlens is not None:
            raise ValueError("P-GDN3-021 uses fixed unpadded sequences")

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

        aux_k = rearrange(
            F.silu(self.aux_k_proj(hidden_states)),
            "... (h d) -> ... h d",
            d=layer.head_k_dim,
        )
        aux_v = rearrange(
            F.silu(self.aux_v_proj(hidden_states)),
            "... (h d) -> ... h d",
            d=layer.head_v_dim,
        )
        aux_b = rearrange(
            self.aux_b_proj(hidden_states).sigmoid(),
            "... (h d) -> ... h d",
            d=layer.head_k_dim,
        )
        aux_w = rearrange(
            self.aux_w_proj(hidden_states).sigmoid(),
            "... (h d) -> ... h d",
            d=layer.head_v_dim,
        )

        recurrent_state = last_state["recurrent_state"] if last_state is not None else None
        interleaved_output, recurrent_state = chunk_gdn2(
            q=self._interleave(q, q),
            k=self._interleave(k, aux_k),
            v=self._interleave(v, aux_v),
            g=self._interleave(g, torch.zeros_like(g)),
            b=self._interleave(b, aux_b),
            w=self._interleave(w, aux_w),
            initial_state=recurrent_state,
            output_final_state=use_cache,
            use_qk_l2norm_in_kernel=True,
        )
        output = interleaved_output[:, 1::2]
        update_layer_cache(
            layer,
            past_key_values,
            recurrent_state=recurrent_state,
            conv_state=(conv_state_q, conv_state_k, conv_state_v),
            offset=q_len,
        )

        with torch.no_grad():
            main_k_unit = F.normalize(k.float(), dim=-1, eps=1e-6)
            aux_k_unit = F.normalize(aux_k.float(), dim=-1, eps=1e-6)
            cosine = (main_k_unit * aux_k_unit).sum(dim=-1).abs()
            main_payload = v.float() * w.float()
            aux_payload = aux_v.float() * aux_w.float()
            main_rms = main_payload.square().mean().sqrt().clamp_min(1e-8)
            aux_board_rms = aux_payload.square().mean(dim=(1, 2, 3)).sqrt()
            self.last_diagnostics = {
                "address_abs_cosine": cosine.mean(),
                "address_separation": 1.0 - cosine.mean(),
                "address_token_std": cosine.mean(dim=2).std(unbiased=False),
                "aux_payload_relative_rms": aux_payload.square().mean().sqrt() / main_rms,
                "aux_payload_board_std": aux_board_rms.std(unbiased=False),
                "aux_erase_mean": aux_b.float().mean(),
                "aux_write_mean": aux_w.float().mean(),
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


class ZoologyTwoEditGDN2FutureSeedMixer(ZoologyGDN2FutureSeedMixer):
    """Native FutureSeed over a two-edit live GDN2 transition."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.layer = TwoEditGatedDeltaNet2(self.layer)


def two_edit_diagnostics(model: nn.Module) -> dict[str, Any]:
    mixers = [
        block.sequence_mixer
        for block in model.backbone.layers
        if isinstance(block.sequence_mixer, ZoologyTwoEditGDN2FutureSeedMixer)
    ]
    if len(mixers) != 2:
        raise RuntimeError(f"Expected two active two-edit mixers, got {len(mixers)}")
    rows = []
    for mixer in mixers:
        diagnostics = mixer.layer.last_diagnostics
        if not diagnostics:
            raise RuntimeError("Two-edit diagnostics were not populated")
        rows.append(
            {
                "layer_idx": mixer.layer_idx,
                **{name: float(value.item()) for name, value in diagnostics.items()},
            }
        )
    return {
        "active_layers": len(rows),
        "parameter_delta": sum(
            projection.weight.numel()
            for mixer in mixers
            for projection in (
                mixer.layer.aux_k_proj,
                mixer.layer.aux_v_proj,
                mixer.layer.aux_b_proj,
                mixer.layer.aux_w_proj,
            )
        ),
        "new_state_values": 0,
        "logical_edits_per_token": 2,
        "per_layer": rows,
    }


def parent_parameter_hash(model: nn.Module) -> str:
    import hashlib

    tensors = []
    auxiliary = ("aux_k_proj", "aux_v_proj", "aux_b_proj", "aux_w_proj")
    for name, parameter in model.named_parameters():
        if any(f".layer.{part}." in name for part in auxiliary):
            continue
        parent_name = name.replace(
            ".sequence_mixer.layer.base.", ".sequence_mixer.layer."
        )
        tensors.append((parent_name, parameter))
    digest = hashlib.sha256()
    for name, parameter in sorted(tensors):
        digest.update(name.encode())
        digest.update(parameter.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()
