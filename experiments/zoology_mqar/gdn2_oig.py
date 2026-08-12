from __future__ import annotations

import hashlib
import math
from typing import Any

import torch
import torch.nn as nn
from einops import rearrange
from torch.nn import functional as F

from fla.layers.gdn2 import GatedDeltaNet2
from fla.layers.utils import get_layer_cache, update_layer_cache

from experiments.zoology_mqar.gdn2_futureseed import (
    ZoologyGDN2FutureSeedMixer,
)


OIG_CHUNK_SIZE = 16
_NORMALIZATION_EPSILON = 1e-6
_ERASE_ENERGY_THRESHOLD = 1e-8


def _normalize_address(tensor: torch.Tensor) -> torch.Tensor:
    tensor = tensor.float()
    return tensor / torch.sqrt(
        tensor.square().sum(dim=-1, keepdim=True) + _NORMALIZATION_EPSILON
    )


def oig_chunk_eager(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    b: torch.Tensor,
    w: torch.Tensor,
    initial_state: torch.Tensor,
    initial_covariance: torch.Tensor,
    mix: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Run one fixed 16-token OIG chunk in FP32.

    The checker-facing return tuple is ``(output, state, covariance, stats)``.
    ``stats`` contains address-delta squared sum, key squared sum, raw-address
    delta squared sum, minimum Sherman-Morrison denominator, and maximum
    committed-residual constraint error.
    """
    if q.shape[1] != OIG_CHUNK_SIZE:
        raise ValueError(f"OIG chunks must contain exactly {OIG_CHUNK_SIZE} tokens")

    q = _normalize_address(q)
    k = _normalize_address(k)
    v = v.float()
    g = g.float()
    b = b.float()
    w = w.float()
    state = initial_state.float()
    covariance = initial_covariance.float()
    mix = mix.float().reshape(1, -1, 1)

    outputs: list[torch.Tensor] = []
    address_delta_squared = q.new_zeros(())
    key_squared = q.new_zeros(())
    raw_address_delta_squared = q.new_zeros(())
    minimum_denominator = q.new_full((), float("inf"))
    maximum_constraint_error = q.new_zeros(())

    for token_index in range(OIG_CHUNK_SIZE):
        q_t = q[:, token_index]
        k_t = k[:, token_index]
        v_t = v[:, token_index]
        decay_t = torch.exp(g[:, token_index])
        b_t = b[:, token_index]
        w_t = w[:, token_index]

        decayed_state = decay_t.unsqueeze(-1) * state
        erase_address = b_t * k_t
        committed_residual = w_t * v_t - torch.einsum(
            "bhk,bhkv->bhv",
            erase_address,
            decayed_state,
        )

        transported_covariance = covariance * (
            decay_t.unsqueeze(-1) * decay_t.unsqueeze(-2)
        )
        transported_covariance = transported_covariance + torch.diag_embed(
            1.0 - decay_t.square()
        )
        inverse_gram_address = torch.einsum(
            "bhij,bhj->bhi",
            transported_covariance,
            k_t,
        )
        denominator = 1.0 + (k_t * inverse_gram_address).sum(dim=-1)
        covariance = transported_covariance - (
            inverse_gram_address.unsqueeze(-1)
            * inverse_gram_address.unsqueeze(-2)
            / denominator.unsqueeze(-1).unsqueeze(-1)
        )
        covariance = 0.5 * (covariance + covariance.transpose(-1, -2))

        erase_energy = erase_address.square().sum(dim=-1)
        has_erase_direction = erase_energy > _ERASE_ENERGY_THRESHOLD
        safe_erase_energy = torch.where(
            has_erase_direction,
            erase_energy,
            torch.ones_like(erase_energy),
        )
        correction_scale = (
            (erase_address * k_t).sum(dim=-1)
            - (erase_address * inverse_gram_address).sum(dim=-1)
        ) / safe_erase_energy
        correction_scale = torch.where(
            has_erase_direction,
            correction_scale,
            torch.zeros_like(correction_scale),
        )
        raw_write_address = (
            inverse_gram_address
            + erase_address * correction_scale.unsqueeze(-1)
        )
        write_address = k_t + mix * (raw_write_address - k_t)

        state = decayed_state + torch.einsum(
            "bhk,bhv->bhkv",
            write_address,
            committed_residual,
        )
        outputs.append(
            torch.einsum("bhkv,bhk->bhv", state, q_t)
            / math.sqrt(q.shape[-1])
        )

        address_delta_squared = address_delta_squared + (
            write_address - k_t
        ).square().sum()
        key_squared = key_squared + k_t.square().sum()
        raw_address_delta_squared = raw_address_delta_squared + (
            raw_write_address - k_t
        ).square().sum()
        minimum_denominator = torch.minimum(
            minimum_denominator,
            denominator.amin(),
        )
        constraint_error = (
            (erase_address * raw_write_address).sum(dim=-1)
            - (erase_address * k_t).sum(dim=-1)
        ).abs()
        maximum_constraint_error = torch.maximum(
            maximum_constraint_error,
            constraint_error.amax(),
        )

    stats = torch.stack(
        (
            address_delta_squared,
            key_squared,
            raw_address_delta_squared,
            minimum_denominator,
            maximum_constraint_error,
        )
    )
    return torch.stack(outputs, dim=1), state, covariance, stats


oig_chunk_compiled = torch.compile(
    oig_chunk_eager,
    fullgraph=True,
    dynamic=False,
    backend="inductor",
)


def oig_recurrence(
    *,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    b: torch.Tensor,
    w: torch.Tensor,
    initial_state: torch.Tensor | None,
    mix_logit: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Run the formal OIG path as compiled fixed chunks with an outer loop."""
    if q.ndim != 4:
        raise ValueError(f"Expected q in BTHK layout, got {tuple(q.shape)}")
    if q.shape[1] % OIG_CHUNK_SIZE != 0:
        raise ValueError(
            f"Sequence length {q.shape[1]} is not divisible by {OIG_CHUNK_SIZE}"
        )
    if q.shape != k.shape or q.shape != g.shape or q.shape != b.shape:
        raise ValueError("OIG q, k, g, and b shapes must match")
    if v.shape != w.shape or v.shape[:3] != q.shape[:3]:
        raise ValueError("OIG v and w must match the BTH prefix of q")

    batch_size, _, num_heads, key_dim = q.shape
    value_dim = v.shape[-1]
    if initial_state is None:
        state = torch.zeros(
            batch_size,
            num_heads,
            key_dim,
            value_dim,
            device=q.device,
            dtype=torch.float32,
        )
    else:
        expected_state_shape = (batch_size, num_heads, key_dim, value_dim)
        if tuple(initial_state.shape) != expected_state_shape:
            raise ValueError(
                f"Expected initial state {expected_state_shape}, got {tuple(initial_state.shape)}"
            )
        state = initial_state.float()

    identity = torch.eye(key_dim, device=q.device, dtype=torch.float32)
    covariance = identity.reshape(1, 1, key_dim, key_dim).expand(
        batch_size,
        num_heads,
        key_dim,
        key_dim,
    ).clone()
    mix = torch.tanh(mix_logit.float())
    outputs = []
    accumulated = q.new_zeros(3, dtype=torch.float32)
    minimum_denominator = q.new_full((), float("inf"), dtype=torch.float32)
    maximum_constraint_error = q.new_zeros((), dtype=torch.float32)

    for chunk_start in range(0, q.shape[1], OIG_CHUNK_SIZE):
        chunk_end = chunk_start + OIG_CHUNK_SIZE
        output, state, covariance, stats = oig_chunk_compiled(
            q[:, chunk_start:chunk_end],
            k[:, chunk_start:chunk_end],
            v[:, chunk_start:chunk_end],
            g[:, chunk_start:chunk_end],
            b[:, chunk_start:chunk_end],
            w[:, chunk_start:chunk_end],
            state,
            covariance,
            mix,
        )
        outputs.append(output)
        accumulated = accumulated + stats[:3]
        minimum_denominator = torch.minimum(minimum_denominator, stats[3])
        maximum_constraint_error = torch.maximum(
            maximum_constraint_error,
            stats[4],
        )

    stats = torch.cat(
        (
            accumulated,
            minimum_denominator.reshape(1),
            maximum_constraint_error.reshape(1),
        )
    )
    return torch.cat(outputs, dim=1), state, covariance, stats


class OIGGatedDeltaNet2(GatedDeltaNet2):
    """Official GDN2 projections and ShortConv with an OIG live recurrence."""

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
            raise ValueError("P-GDN3-027 uses fixed unpadded MQAR sequences")
        if self.mode != "chunk":
            raise RuntimeError("P-GDN3-027 requires chunk-mode GDN2 projections")
        if kwargs.get("cu_seqlens") is not None:
            raise ValueError("P-GDN3-027 does not use packed sequences")
        if not hasattr(self, "oig_mix_logit"):
            raise RuntimeError("OIG layer was not upgraded with its per-head mix logit")

        q_len = hidden_states.shape[1]
        last_state = get_layer_cache(self, past_key_values)
        conv_state_q, conv_state_k, conv_state_v = None, None, None
        if last_state is not None:
            conv_state_q, conv_state_k, conv_state_v = last_state["conv_state"]

        if self.use_short_conv:
            q, conv_state_q = self.q_conv1d(
                x=self.q_proj(hidden_states),
                cache=conv_state_q,
                output_final_state=use_cache,
            )
            k, conv_state_k = self.k_conv1d(
                x=self.k_proj(hidden_states),
                cache=conv_state_k,
                output_final_state=use_cache,
            )
            v, conv_state_v = self.v_conv1d(
                x=self.v_proj(hidden_states),
                cache=conv_state_v,
                output_final_state=use_cache,
            )
        else:
            q = F.silu(self.q_proj(hidden_states))
            k = F.silu(self.k_proj(hidden_states))
            v = F.silu(self.v_proj(hidden_states))

        projection_dtype = q.dtype
        g = F.softplus(self.f_proj(hidden_states).float() + self.dt_bias)
        b = self.b_proj(hidden_states).sigmoid()
        w = self.w_proj(hidden_states).sigmoid()
        q, k, g = (
            rearrange(tensor, "... (h d) -> ... h d", d=self.head_k_dim)
            for tensor in (q, k, g)
        )
        v = rearrange(v, "... (h d) -> ... h d", d=self.head_v_dim)
        b = rearrange(b, "... (h d) -> ... h d", d=self.head_k_dim)
        w = rearrange(w, "... (h d) -> ... h d", d=self.head_v_dim)
        g = -self.A_log.float().exp().unsqueeze(-1) * g
        if self.allow_neg_eigval:
            b = b * 2.0

        recurrent_state = (
            last_state["recurrent_state"] if last_state is not None else None
        )
        output, recurrent_state, covariance, stats = oig_recurrence(
            q=q,
            k=k,
            v=v,
            g=g,
            b=b,
            w=w,
            initial_state=recurrent_state,
            mix_logit=self.oig_mix_logit,
        )
        update_layer_cache(
            self,
            past_key_values,
            recurrent_state=recurrent_state,
            conv_state=(conv_state_q, conv_state_k, conv_state_v)
            if self.use_short_conv
            else None,
            offset=q_len,
        )

        with torch.no_grad():
            key_squared = stats[1].clamp_min(1e-12)
            covariance_symmetry_error = (
                covariance - covariance.transpose(-1, -2)
            ).abs().amax()
            covariance_board_rms = covariance.square().mean(
                dim=(-1, -2, -3)
            ).sqrt()
            self.last_oig_diagnostics = {
                "mix_abs": torch.tanh(self.oig_mix_logit.float()).abs().mean(),
                "address_relative_rms": torch.sqrt(stats[0] / key_squared),
                "raw_address_relative_rms": torch.sqrt(stats[2] / key_squared),
                "minimum_denominator": stats[3],
                "constraint_max_abs_error": stats[4],
                "covariance_symmetry_max_abs_error": covariance_symmetry_error,
                "covariance_board_rms_std": covariance_board_rms.std(
                    unbiased=False
                ),
            }
            self.last_oig_covariance = covariance.detach()

        output = self.o_norm(
            output.to(dtype=projection_dtype),
            rearrange(
                self.g_proj(hidden_states),
                "... (h d) -> ... h d",
                d=self.head_v_dim,
            ),
        )
        output = rearrange(output, "b t h d -> b t (h d)")
        return self.o_proj(output), None, past_key_values


def _upgrade_to_oig(layer: GatedDeltaNet2) -> OIGGatedDeltaNet2:
    if layer.num_heads != layer.num_v_heads:
        raise ValueError("P-GDN3-027 requires matched QK and value heads")
    if hasattr(layer, "oig_mix_logit"):
        raise RuntimeError("GDN2 layer was already upgraded to OIG")
    layer.register_parameter(
        "oig_mix_logit",
        nn.Parameter(torch.zeros(layer.num_heads)),
    )
    layer.__class__ = OIGGatedDeltaNet2
    layer.last_oig_diagnostics = {}
    layer.last_oig_covariance = None
    return layer


class ZoologyOIGGDN2FutureSeedMixer(ZoologyGDN2FutureSeedMixer):
    """Native FutureSeed over the OIG-preconditioned live GDN2 update."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.layer = _upgrade_to_oig(self.layer)


@torch.no_grad()
def oig_diagnostics(model: nn.Module) -> dict[str, Any]:
    mixers = [
        block.sequence_mixer
        for block in model.backbone.layers
        if isinstance(block.sequence_mixer, ZoologyOIGGDN2FutureSeedMixer)
        and isinstance(block.sequence_mixer.layer, OIGGatedDeltaNet2)
    ]
    if not mixers:
        raise RuntimeError("No active OIG GDN2 mixers found")

    rows = []
    total_active_heads = 0
    for mixer in mixers:
        layer = mixer.layer
        if not layer.last_oig_diagnostics or layer.last_oig_covariance is None:
            raise RuntimeError("OIG diagnostics require at least one completed forward")
        covariance = 0.5 * (
            layer.last_oig_covariance
            + layer.last_oig_covariance.transpose(-1, -2)
        )
        eigenvalues = torch.linalg.eigvalsh(covariance.float())
        positive_minimum = eigenvalues.amin().clamp_min(1e-30)
        diagonal = torch.diag_embed(
            covariance.diagonal(dim1=-2, dim2=-1)
        )
        off_diagonal_relative_rms = (
            (covariance - diagonal).square().mean().sqrt()
            / covariance.square().mean().sqrt().clamp_min(1e-30)
        )
        mix_abs = torch.tanh(layer.oig_mix_logit.float()).abs()
        active_heads = int((mix_abs >= 1e-3).sum().item())
        total_active_heads += active_heads
        cholesky_info = torch.linalg.cholesky_ex(covariance.float()).info
        rows.append(
            {
                "layer_idx": mixer.layer_idx,
                **{
                    name: float(value.item())
                    for name, value in layer.last_oig_diagnostics.items()
                },
                "covariance_eigenvalue_min": float(eigenvalues.amin().item()),
                "covariance_eigenvalue_max": float(eigenvalues.amax().item()),
                "covariance_condition_max": float(
                    (eigenvalues.amax() / positive_minimum).item()
                ),
                "active_heads": active_heads,
                "mix_abs_mean": float(mix_abs.mean().item()),
                "mix_abs_min": float(mix_abs.amin().item()),
                "p_offdiag_relative_rms": float(
                    off_diagonal_relative_rms.item()
                ),
                "a_relative_rms": float(
                    layer.last_oig_diagnostics["address_relative_rms"].item()
                ),
                "constraint_max_error": float(
                    layer.last_oig_diagnostics[
                        "constraint_max_abs_error"
                    ].item()
                ),
                "symmetry_max_error": float(
                    layer.last_oig_diagnostics[
                        "covariance_symmetry_max_abs_error"
                    ].item()
                ),
                "sampled_eigenvalue_min": float(eigenvalues.amin().item()),
                "sampled_eigenvalue_max": float(eigenvalues.amax().item()),
                "cholesky_success": bool((cholesky_info == 0).all().item()),
            }
        )
    return {
        "active_layers": len(rows),
        "active_heads": total_active_heads,
        "parameter_delta": sum(mixer.layer.oig_mix_logit.numel() for mixer in mixers),
        "inverse_gram_state_values_per_layer": (
            mixers[0].layer.num_heads * mixers[0].layer.head_k_dim**2
        ),
        "futureseed_transports_inverse_gram_state": False,
        "inverse_gram_initialization": "identity_per_receiver_scan",
        "chunk_size": OIG_CHUNK_SIZE,
        "compiled_fullgraph": True,
        "compiled_dynamic": False,
        "compiled_backend": "inductor",
        "formal_eager_fallback": False,
        "per_layer": rows,
    }


def parent_parameter_hash(model: nn.Module) -> str:
    """Hash parent parameters without changing their original module names."""
    digest = hashlib.sha256()
    tensors = [
        (name, parameter)
        for name, parameter in model.named_parameters()
        if not name.endswith(".oig_mix_logit")
    ]
    for name, parameter in sorted(tensors):
        digest.update(name.encode())
        digest.update(parameter.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()
