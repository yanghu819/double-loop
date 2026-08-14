from __future__ import annotations

import hashlib
from typing import Any

import torch
from einops import rearrange
from torch import nn
from torch.nn import functional as F

from fla.layers.raven import Raven
from fla.layers.utils import get_layer_cache, update_layer_cache
from fla.models.utils import Cache
from fla.ops.gdn2 import chunk_gdn2

from experiments.zoology_mqar.gdn2_futureseed import (
    ZoologyGDN2FutureSeedMixer,
)


RAVEN_SLOTS = 16
RAVEN_TOPK = 1
RAVEN_STATE_VALUES_PER_LAYER = 4 * (2 * 32) * RAVEN_SLOTS
EXPECTED_PARAMETER_DELTA_PER_LAYER = 90_824


def _empty_cache(layer_idx: int) -> Cache:
    cache = Cache()
    for index in range(layer_idx):
        cache.update(layer_idx=index, offset=0)
    return cache


class RavenAddressGatedDeltaNet2(nn.Module):
    """Official Raven context composes one coherent Q/K input for GDN2."""

    def __init__(self, base: nn.Module) -> None:
        super().__init__()
        if base.hidden_size != 128:
            raise ValueError("P-GDN3-039 requires D128")
        if base.num_heads != 4 or base.num_v_heads != 4:
            raise ValueError("P-GDN3-039 requires H4")
        if base.head_k_dim != 32 or base.head_v_dim != 32:
            raise ValueError("P-GDN3-039 requires K32/V32")
        self.base = base
        self.raven = Raven(
            mode="chunk",
            hidden_size=128,
            expand_k=1.0,
            expand_v=1.0,
            num_heads=4,
            num_slots=RAVEN_SLOTS,
            topk=RAVEN_TOPK,
            use_short_conv=False,
            layer_idx=base.layer_idx,
        )
        self.address_adapter = nn.Linear(128, 128, bias=False)
        nn.init.zeros_(self.address_adapter.weight)
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

    @property
    def layer_idx(self) -> int:
        return int(self.base.layer_idx)

    def _raven_context(
        self,
        hidden_states: torch.Tensor,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        cache = _empty_cache(self.layer_idx)
        output, _attention, cache = self.raven(
            hidden_states,
            past_key_values=cache,
            use_cache=True,
        )
        terminal_state = cache[self.layer_idx]["recurrent_state"]
        if terminal_state is None or len(terminal_state) != 2:
            raise RuntimeError("Official Raven did not return a terminal state")
        return output, terminal_state

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
            raise ValueError("P-GDN3-039 uses fixed unpadded MQAR sequences")
        if kwargs.get("cu_seqlens") is not None:
            raise ValueError("P-GDN3-039 does not use packed sequences")
        if self.base.mode != "chunk" or self.raven.mode != "chunk":
            raise RuntimeError("P-GDN3-039 requires official chunk kernels")

        raven_output, raven_terminal = self._raven_context(hidden_states)
        address_residual = self.address_adapter(raven_output)
        address_hidden = hidden_states + address_residual

        layer = self.base
        q_len = hidden_states.shape[1]
        last_state = get_layer_cache(layer, past_key_values)
        conv_state_q, conv_state_k, conv_state_v = None, None, None
        if last_state is not None:
            conv_state_q, conv_state_k, conv_state_v = last_state["conv_state"]
        q, conv_state_q = layer.q_conv1d(
            x=layer.q_proj(address_hidden),
            cache=conv_state_q,
            output_final_state=use_cache,
        )
        k, conv_state_k = layer.k_conv1d(
            x=layer.k_proj(address_hidden),
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
            raise RuntimeError("P-GDN3-039 requires bounded positive erase gates")

        recurrent_state = (
            last_state["recurrent_state"] if last_state is not None else None
        )
        output, recurrent_state = chunk_gdn2(
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
            content_rms = hidden_states.float().square().mean().sqrt().clamp_min(1e-8)
            residual_board_rms = address_residual.float().square().mean(
                dim=(1, 2)
            ).sqrt()
            raven_board_rms = raven_output.float().square().mean(
                dim=(1, 2)
            ).sqrt()
            raven_key_state, raven_value_state = raven_terminal
            terminal_values = torch.cat(
                (
                    raven_key_state.float().flatten(start_dim=1),
                    raven_value_state.float().flatten(start_dim=1),
                ),
                dim=1,
            )
            terminal_board_rms = terminal_values.square().mean(dim=1).sqrt()
            slot_mass = (
                raven_key_state.float().square().sum(dim=2)
                + raven_value_state.float().square().sum(dim=3)
            )
            slot_probability = slot_mass / slot_mass.sum(
                dim=-1,
                keepdim=True,
            ).clamp_min(1e-12)
            entropy = -(
                slot_probability
                * slot_probability.clamp_min(1e-12).log()
            ).sum(dim=-1) / torch.log(
                slot_probability.new_tensor(float(RAVEN_SLOTS))
            )
            parent_q = layer.q_proj(hidden_states).float()
            parent_k = layer.k_proj(hidden_states).float()
            composed_q = layer.q_proj(address_hidden).float()
            composed_k = layer.k_proj(address_hidden).float()
            self.last_diagnostics = {
                "raven_output_rms": raven_board_rms.mean(),
                "raven_output_board_std": raven_board_rms.std(unbiased=False),
                "raven_terminal_rms": terminal_board_rms.mean(),
                "raven_terminal_board_std": terminal_board_rms.std(unbiased=False),
                "raven_slot_entropy": entropy.mean(),
                "raven_slot_max_mass": slot_probability.max(dim=-1).values.mean(),
                "address_residual_relative_rms": residual_board_rms.mean()
                / content_rms,
                "address_residual_token_std": address_residual.float().square().mean(
                    dim=-1
                ).sqrt().std(dim=1, unbiased=False).mean(),
                "address_residual_board_std": residual_board_rms.std(unbiased=False),
                "q_projection_relative_change": (
                    (composed_q - parent_q).square().mean().sqrt()
                    / parent_q.square().mean().sqrt().clamp_min(1e-8)
                ),
                "k_projection_relative_change": (
                    (composed_k - parent_k).square().mean().sqrt()
                    / parent_k.square().mean().sqrt().clamp_min(1e-8)
                ),
                "main_terminal_rms": recurrent_state.float().square().mean().sqrt(),
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


class ZoologyRavenAddressFutureSeedMixer(ZoologyGDN2FutureSeedMixer):
    """Native FutureSeed plus a recurrent Raven address context."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.layer = RavenAddressGatedDeltaNet2(self.layer)

    def state_size(self, sequence_length: int = 2048) -> int:
        return super().state_size(sequence_length) + RAVEN_STATE_VALUES_PER_LAYER


def _is_new_parameter(name: str) -> bool:
    return (
        ".sequence_mixer.layer.raven." in name
        or ".sequence_mixer.layer.address_adapter." in name
    )


def load_matched_parent_state(
    model: nn.Module,
    control_state: dict[str, torch.Tensor],
) -> None:
    loaded: dict[str, torch.Tensor] = {}
    new_names = []
    zero_adapter_names = []
    for name, tensor in model.state_dict().items():
        if _is_new_parameter(name):
            if name.endswith("address_adapter.weight"):
                loaded[name] = torch.zeros_like(tensor)
                zero_adapter_names.append(name)
            else:
                loaded[name] = tensor
            new_names.append(name)
            continue
        parent_name = name.replace(
            ".sequence_mixer.layer.base.",
            ".sequence_mixer.layer.",
        )
        if parent_name not in control_state:
            raise RuntimeError(f"Missing matched parent tensor: {parent_name}")
        parent = control_state[parent_name]
        if parent.shape != tensor.shape:
            raise RuntimeError(
                f"Matched tensor shape mismatch for {name}: "
                f"{tuple(parent.shape)} != {tuple(tensor.shape)}"
            )
        loaded[name] = parent
    if not new_names:
        raise RuntimeError("Raven address candidate has no new parameters")
    if len(zero_adapter_names) != 2:
        raise RuntimeError(
            f"Expected two zero address adapters, got {zero_adapter_names}"
        )
    model.load_state_dict(loaded, strict=True)


def parent_parameter_hash(model: nn.Module) -> str:
    digest = hashlib.sha256()
    rows: list[tuple[str, torch.Tensor]] = []
    for name, parameter in model.named_parameters():
        if _is_new_parameter(name):
            continue
        parent_name = name.replace(
            ".sequence_mixer.layer.base.",
            ".sequence_mixer.layer.",
        )
        rows.append((parent_name, parameter))
    for name, value in sorted(rows):
        digest.update(name.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def raven_address_diagnostics(model: nn.Module) -> dict[str, Any]:
    mixers = [block.sequence_mixer for block in model.backbone.layers]
    if len(mixers) != 2 or not all(
        isinstance(mixer, ZoologyRavenAddressFutureSeedMixer) for mixer in mixers
    ):
        raise RuntimeError("Expected exactly two Raven-address mixers")
    rows = []
    for mixer in mixers:
        diagnostics = mixer.layer.last_diagnostics
        if not diagnostics:
            raise RuntimeError("Raven-address diagnostics were not populated")
        rows.append(
            {
                "layer_idx": mixer.layer_idx,
                **{name: float(value.item()) for name, value in diagnostics.items()},
            }
        )
    return {
        "active_layers": len(rows),
        "raven_slots": RAVEN_SLOTS,
        "raven_topk": RAVEN_TOPK,
        "official_gdn2_scans_per_layer": 1,
        "official_raven_scans_per_layer": 1,
        "raven_state_values_per_layer": RAVEN_STATE_VALUES_PER_LAYER,
        "total_state_values_per_layer": 4_096 + RAVEN_STATE_VALUES_PER_LAYER,
        "new_parameters": sum(
            parameter.numel()
            for name, parameter in model.named_parameters()
            if _is_new_parameter(name)
        ),
        "per_layer": rows,
    }
