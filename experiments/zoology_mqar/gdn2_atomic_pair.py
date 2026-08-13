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
from experiments.zoology_mqar.gdn2_rank2 import FixedInitProjection


POLAR_EPS = 1e-5


def symmetric_weighted_pair(
    first: torch.Tensor,
    second: torch.Tensor,
    erase: torch.Tensor,
    *,
    eps: float = POLAR_EPS,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
    """Whiten two addresses under the positive erase-weighted inner product."""

    pair = torch.stack((first.float(), second.float()), dim=-1)
    weights = erase.float().clamp_min(1e-4).unsqueeze(-1)
    key_dim = pair.shape[-2]
    gram = torch.einsum("...ki,...kj->...ij", pair * weights, pair) / key_dim

    a = gram[..., 0, 0]
    b = 0.5 * (gram[..., 0, 1] + gram[..., 1, 0])
    d = gram[..., 1, 1]
    discriminant = ((a - d).square() + 4.0 * b.square()).clamp_min(0.0).sqrt()
    eig_min = (0.5 * (a + d - discriminant)).clamp_min(1e-12)
    eig_max = (0.5 * (a + d + discriminant)).clamp_min(1e-12)

    a_reg = a + eps
    d_reg = d + eps
    determinant = (a_reg * d_reg - b.square()).clamp_min(eps * eps)
    root_det = determinant.sqrt()
    normalizer = (a_reg + d_reg + 2.0 * root_det).clamp_min(eps).sqrt()
    matrix_a = a_reg + root_det
    matrix_d = d_reg + root_det
    inverse_determinant = (
        matrix_a * matrix_d - b.square()
    ).clamp_min(eps * eps).reciprocal()
    inverse_sqrt = torch.stack(
        (
            normalizer * matrix_d * inverse_determinant,
            -normalizer * b * inverse_determinant,
            -normalizer * b * inverse_determinant,
            normalizer * matrix_a * inverse_determinant,
        ),
        dim=-1,
    ).reshape(*gram.shape)
    whitened = torch.einsum("...ki,...ij->...kj", pair, inverse_sqrt)

    post_gram = (
        torch.einsum("...ki,...kj->...ij", whitened * weights, whitened)
        / key_dim
    )
    diagnostics = {
        "raw_condition": eig_max / eig_min,
        "raw_eigenvalue_min": eig_min,
        "post_cross_abs": post_gram[..., 0, 1].abs(),
        "post_diagonal_error": (
            torch.diagonal(post_gram, dim1=-2, dim2=-1) - 1.0
        ).abs().amax(dim=-1),
    }
    return (
        whitened[..., 0].to(first.dtype),
        whitened[..., 1].to(second.dtype),
        diagnostics,
    )


class AtomicPairGatedDeltaNet2(nn.Module):
    """One shared-payload atomic rank-two update through official GDN2."""

    def __init__(self, base: nn.Module) -> None:
        super().__init__()
        if base.num_v_heads != base.num_heads:
            raise ValueError("P-GDN3-028 requires matched QK/V heads")
        if base.key_dim != base.hidden_size or base.value_dim != base.hidden_size:
            raise ValueError("P-GDN3-028 requires D=H*K=H*V")
        self.base = base
        seed_base = 28_000 + 100 * int(base.layer_idx)
        self.aux_q_proj = FixedInitProjection(base.hidden_size, seed=seed_base + 1)
        self.aux_k_proj = FixedInitProjection(base.hidden_size, seed=seed_base + 2)
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
            raise ValueError("P-GDN3-028 does not use padded MQAR batches")

        layer = self.base
        q_len = hidden_states.shape[1]
        if layer.mode != "chunk":
            raise RuntimeError("P-GDN3-028 requires the official chunk kernel")
        last_state = get_layer_cache(layer, past_key_values)
        if kwargs.get("cu_seqlens") is not None:
            raise ValueError("P-GDN3-028 uses fixed unpadded sequences")

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

        aux_q = rearrange(
            self.aux_q_proj(hidden_states),
            "... (h d) -> ... h d",
            d=layer.head_k_dim,
        )
        aux_k = rearrange(
            self.aux_k_proj(hidden_states),
            "... (h d) -> ... h d",
            d=layer.head_k_dim,
        )
        main_k, aux_k, pair_diagnostics = symmetric_weighted_pair(k, aux_k, b)
        query = (q.float() + aux_q.float()).mul_(2.0**-0.5).to(q.dtype)

        shared_v = v * (2.0**-0.5)
        recurrent_state = last_state["recurrent_state"] if last_state is not None else None
        interleaved_output, recurrent_state = chunk_gdn2(
            q=self._interleave(query, query),
            k=self._interleave(main_k, aux_k),
            v=self._interleave(shared_v, shared_v),
            g=self._interleave(g, torch.zeros_like(g)),
            b=self._interleave(b, b),
            w=self._interleave(w, w),
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
            main_unit = F.normalize(main_k.float(), dim=-1, eps=1e-6)
            aux_unit = F.normalize(aux_k.float(), dim=-1, eps=1e-6)
            weighted_cross = (
                b.float() * main_unit * aux_unit
            ).sum(dim=-1).abs()
            state_rms = (
                torch.zeros((), device=hidden_states.device)
                if recurrent_state is None
                else recurrent_state.float().square().mean().sqrt()
            )
            self.last_diagnostics = {
                "raw_condition_mean": pair_diagnostics["raw_condition"].mean(),
                "raw_condition_max": pair_diagnostics["raw_condition"].amax(),
                "raw_eigenvalue_min": pair_diagnostics[
                    "raw_eigenvalue_min"
                ].amin(),
                "post_weighted_cross_abs": weighted_cross.mean(),
                "post_weighted_cross_abs_max": weighted_cross.amax(),
                "post_diagonal_error": pair_diagnostics[
                    "post_diagonal_error"
                ].amax(),
                "shared_payload_max_diff": (
                    shared_v * w - shared_v * w
                ).abs().amax(),
                "aggregate_payload_energy_ratio": (
                    2.0
                    * (shared_v.float() * w.float()).square().mean()
                    / (v.float() * w.float()).square().mean().clamp_min(1e-12)
                ),
                "terminal_state_rms": state_rms,
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


class ZoologyAtomicPairGDN2FutureSeedMixer(ZoologyGDN2FutureSeedMixer):
    """Native FutureSeed over an atomic shared-payload rank-two GDN update."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.layer = AtomicPairGatedDeltaNet2(self.layer)


def atomic_pair_diagnostics(model: nn.Module) -> dict[str, Any]:
    mixers = [
        block.sequence_mixer
        for block in model.backbone.layers
        if isinstance(block.sequence_mixer, ZoologyAtomicPairGDN2FutureSeedMixer)
    ]
    if len(mixers) != 2:
        raise RuntimeError(f"Expected two active atomic-pair mixers, got {len(mixers)}")
    rows = []
    for mixer in mixers:
        diagnostics = mixer.layer.last_diagnostics
        if not diagnostics:
            raise RuntimeError("Atomic-pair diagnostics were not populated")
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
            for projection in (mixer.layer.aux_q_proj, mixer.layer.aux_k_proj)
        ),
        "new_state_values": 0,
        "logical_block_rank": 2,
        "shared_payload": True,
        "per_layer": rows,
    }


def parent_parameter_hash(model: nn.Module) -> str:
    import hashlib

    tensors = []
    for name, parameter in model.named_parameters():
        if ".layer.aux_q_proj." in name or ".layer.aux_k_proj." in name:
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
