from __future__ import annotations

import hashlib
from typing import Any

import torch
from einops import rearrange
from torch import nn
from torch.nn import functional as F

from fla.layers.utils import get_layer_cache, update_layer_cache
from fla.ops.gdn2 import chunk_gdn2

from experiments.zoology_mqar.gdn2_futureseed import (
    ZoologyGDN2FutureSeedMixer,
)


SLOT_COUNT = 2


def _route_gate(gate: torch.Tensor, mass: torch.Tensor) -> torch.Tensor:
    """Route a [0,1] gate while preserving it exactly at unit slot mass."""

    gate_fp32 = gate.float().unsqueeze(-2)
    mass_fp32 = mass.float().unsqueeze(-1)
    denominator = 1.0 + (mass_fp32 - 1.0) * gate_fp32
    return (mass_fp32 * gate_fp32 / denominator).to(dtype=gate.dtype)


class SlotStateGatedDeltaNet2(nn.Module):
    """One official GDN2 scan over content-partitioned full state slots."""

    def __init__(self, base: nn.Module, *, slot_count: int = SLOT_COUNT) -> None:
        super().__init__()
        if slot_count != SLOT_COUNT:
            raise ValueError("P-GDN3-038 fixes exactly two state slots")
        if base.num_heads != 4 or base.num_v_heads != 4:
            raise ValueError("P-GDN3-038 requires H4")
        if base.head_k_dim != 32 or base.head_v_dim != 32:
            raise ValueError("P-GDN3-038 requires K32/V32")
        if base.hidden_size != 128:
            raise ValueError("P-GDN3-038 requires D128")
        self.base = base
        self.slot_count = int(slot_count)
        self.slot_router = nn.Linear(
            base.hidden_size,
            base.num_heads * self.slot_count,
            bias=False,
        )
        self.last_diagnostics: dict[str, torch.Tensor] = {}

    @property
    def num_heads(self) -> int:
        return int(self.base.num_heads)

    @property
    def num_v_heads(self) -> int:
        return int(self.base.num_v_heads * self.slot_count)

    @property
    def head_k_dim(self) -> int:
        return int(self.base.head_k_dim)

    @property
    def head_v_dim(self) -> int:
        return int(self.base.head_v_dim)

    @property
    def layer_idx(self) -> int:
        return int(self.base.layer_idx)

    def slot_probabilities(self, hidden_states: torch.Tensor) -> torch.Tensor:
        logits = rearrange(
            self.slot_router(hidden_states).float(),
            "b t (h s) -> b t h s",
            h=self.num_heads,
            s=self.slot_count,
        )
        return logits.softmax(dim=-1)

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
            raise ValueError("P-GDN3-038 uses fixed unpadded MQAR sequences")
        if kwargs.get("cu_seqlens") is not None:
            raise ValueError("P-GDN3-038 does not use packed sequences")
        if self.base.mode != "chunk":
            raise RuntimeError("P-GDN3-038 requires the official chunk kernel")

        layer = self.base
        q_len = hidden_states.shape[1]
        last_state = get_layer_cache(layer, past_key_values)
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
        if layer.allow_neg_eigval:
            raise RuntimeError("P-GDN3-038 requires bounded positive erase gates")

        probabilities = self.slot_probabilities(hidden_states)
        slot_mass = self.slot_count * probabilities
        q_slots = rearrange(
            q.unsqueeze(-2).expand(-1, -1, -1, self.slot_count, -1),
            "b t h s d -> b t (h s) d",
        )
        k_slots = rearrange(
            k.unsqueeze(-2).expand(-1, -1, -1, self.slot_count, -1),
            "b t h s d -> b t (h s) d",
        )
        v_slots = rearrange(
            v.unsqueeze(-2).expand(-1, -1, -1, self.slot_count, -1),
            "b t h s d -> b t (h s) d",
        )
        g_slots = rearrange(
            g.unsqueeze(-2).expand(-1, -1, -1, self.slot_count, -1),
            "b t h s d -> b t (h s) d",
        )
        b_slots = rearrange(
            _route_gate(b, slot_mass),
            "b t h s d -> b t (h s) d",
        )
        w_slots = rearrange(
            _route_gate(w, slot_mass),
            "b t h s d -> b t (h s) d",
        )

        recurrent_state = (
            last_state["recurrent_state"] if last_state is not None else None
        )
        if recurrent_state is not None and recurrent_state.shape[1:] != (
            self.num_v_heads,
            self.head_k_dim,
            self.head_v_dim,
        ):
            raise RuntimeError(
                f"Unexpected slot-state shape: {tuple(recurrent_state.shape)}"
            )
        slot_output, recurrent_state = chunk_gdn2(
            q=q_slots,
            k=k_slots,
            v=v_slots,
            g=g_slots,
            b=b_slots,
            w=w_slots,
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

        slot_output = rearrange(
            slot_output,
            "b t (h s) d -> b t h s d",
            h=self.num_heads,
            s=self.slot_count,
        )
        output = (slot_output * probabilities.to(slot_output.dtype).unsqueeze(-1)).sum(
            dim=-2
        )

        with torch.no_grad():
            state = rearrange(
                recurrent_state.float(),
                "b (h s) k v -> b h s k v",
                h=self.num_heads,
                s=self.slot_count,
            )
            first, second = state.unbind(dim=2)
            first_flat = first.flatten(-2)
            second_flat = second.flatten(-2)
            cosine = F.cosine_similarity(first_flat, second_flat, dim=-1)
            state_rms = state.square().mean().sqrt().clamp_min(1e-8)
            entropy = -(probabilities * probabilities.clamp_min(1e-8).log()).sum(
                dim=-1
            ) / torch.log(probabilities.new_tensor(float(self.slot_count)))
            b_change = (b_slots.float() - rearrange(
                b.unsqueeze(-2).expand(-1, -1, -1, self.slot_count, -1),
                "b t h s d -> b t (h s) d",
            ).float()).square().mean().sqrt()
            w_change = (w_slots.float() - rearrange(
                w.unsqueeze(-2).expand(-1, -1, -1, self.slot_count, -1),
                "b t h s d -> b t (h s) d",
            ).float()).square().mean().sqrt()
            self.last_diagnostics = {
                "normalized_router_entropy": entropy.mean(),
                "router_max_probability": probabilities.max(dim=-1).values.mean(),
                "router_token_std": probabilities.std(dim=1, unbiased=False).mean(),
                "router_board_std": probabilities.mean(dim=1).std(
                    dim=0, unbiased=False
                ).mean(),
                "minimum_global_slot_mass": probabilities.mean(
                    dim=(0, 1, 2)
                ).min(),
                "maximum_global_slot_mass": probabilities.mean(
                    dim=(0, 1, 2)
                ).max(),
                "slot_state_rms": state_rms,
                "slot_state_relative_difference": (
                    (first - second).square().mean().sqrt() / state_rms
                ),
                "slot_state_cosine_mean": cosine.mean(),
                "slot_state_cosine_max": cosine.max(),
                "erase_route_change_rms": b_change,
                "write_route_change_rms": w_change,
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


class ZoologySlotStateFutureSeedMixer(ZoologyGDN2FutureSeedMixer):
    """Native FutureSeed over two first-class full GDN2 state slots."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        parent_gate = self.future_seed_logit.detach()
        self.layer = SlotStateGatedDeltaNet2(self.layer)
        self.future_seed_logit = nn.Parameter(
            parent_gate.repeat_interleave(SLOT_COUNT, dim=1)
        )


def load_matched_parent_state(
    model: nn.Module,
    control_state: dict[str, torch.Tensor],
) -> None:
    candidate_state = model.state_dict()
    loaded: dict[str, torch.Tensor] = {}
    router_names = []
    for name, tensor in candidate_state.items():
        if ".sequence_mixer.layer.slot_router." in name:
            loaded[name] = tensor
            router_names.append(name)
            continue
        parent_name = name.replace(
            ".sequence_mixer.layer.base.",
            ".sequence_mixer.layer.",
        )
        if parent_name not in control_state:
            raise RuntimeError(f"Missing matched parent tensor: {parent_name}")
        parent = control_state[parent_name]
        if name.endswith("future_seed_logit"):
            parent = parent.repeat_interleave(SLOT_COUNT, dim=1)
        if parent.shape != tensor.shape:
            raise RuntimeError(
                f"Matched tensor shape mismatch for {name}: "
                f"{tuple(parent.shape)} != {tuple(tensor.shape)}"
            )
        loaded[name] = parent
    if len(router_names) != 2:
        raise RuntimeError(f"Expected two slot-router tensors, got {router_names}")
    model.load_state_dict(loaded, strict=True)


def parent_parameter_hash(model: nn.Module) -> str:
    digest = hashlib.sha256()
    rows: list[tuple[str, torch.Tensor]] = []
    for name, parameter in model.named_parameters():
        if ".sequence_mixer.layer.slot_router." in name:
            continue
        parent_name = name.replace(
            ".sequence_mixer.layer.base.",
            ".sequence_mixer.layer.",
        )
        value = parameter
        if name.endswith("future_seed_logit"):
            value = rearrange(
                parameter,
                "b (h s) k v -> b h s k v",
                s=SLOT_COUNT,
            )[:, :, 0]
        rows.append((parent_name, value))
    for name, value in sorted(rows):
        digest.update(name.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def slot_state_diagnostics(model: nn.Module) -> dict[str, Any]:
    mixers = [block.sequence_mixer for block in model.backbone.layers]
    if len(mixers) != 2 or not all(
        isinstance(mixer, ZoologySlotStateFutureSeedMixer) for mixer in mixers
    ):
        raise RuntimeError("Expected exactly two slot-state mixers")
    rows = []
    for mixer in mixers:
        diagnostics = mixer.layer.last_diagnostics
        if not diagnostics:
            raise RuntimeError("Slot-state diagnostics were not populated")
        rows.append(
            {
                "layer_idx": mixer.layer_idx,
                **{name: float(value.item()) for name, value in diagnostics.items()},
            }
        )
    return {
        "active_layers": len(rows),
        "slot_count": SLOT_COUNT,
        "state_values_per_layer": 4 * SLOT_COUNT * 32 * 32,
        "official_scans_per_layer": 1,
        "router_parameters": sum(
            mixer.layer.slot_router.weight.numel() for mixer in mixers
        ),
        "per_layer": rows,
    }
