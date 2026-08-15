from __future__ import annotations

import hashlib
from typing import Any

import torch
import torch.nn.functional as F
from einops import rearrange
from torch import nn

from fla.layers.utils import get_layer_cache, update_layer_cache
from fla.modules import ShortConvolution
from fla.ops.gdn2 import chunk_gdn2

from experiments.zoology_mqar.gdn2_futureseed import (
    ZoologyGDN2FutureSeedMixer,
)


BANK_COUNT = 2
PARENT_HEADS = 4
HEAD_DIM = 32
PARENT_STATE_VALUES = PARENT_HEADS * HEAD_DIM * HEAD_DIM
REDUNDANT_STATE_VALUES = BANK_COUNT * PARENT_STATE_VALUES
EXPECTED_PARAMETER_DELTA = 67_592


def _rms(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.float().square().mean().sqrt()


def _address_contrast(query: torch.Tensor, key: torch.Tensor) -> torch.Tensor:
    query_unit = F.normalize(query[:1].float(), dim=-1, eps=1e-6)
    key_unit = F.normalize(key[:1].float(), dim=-1, eps=1e-6)
    similarity = torch.einsum("bthd,bshd->bhts", query_unit, key_unit)
    diagonal = similarity.diagonal(dim1=-2, dim2=-1)
    if similarity.shape[-1] <= 1:
        return diagonal.mean()
    off_diagonal = (similarity.sum() - diagonal.sum()) / float(
        similarity.shape[0]
        * similarity.shape[1]
        * similarity.shape[2]
        * (similarity.shape[3] - 1)
    )
    return diagonal.mean() - off_diagonal


class RedundantAddressGatedDeltaNet2(nn.Module):
    """Two independent address banks with one shared GDN2 edit stream."""

    def __init__(self, base: nn.Module) -> None:
        super().__init__()
        if base.hidden_size != 128:
            raise ValueError("P-GDN3-053 requires D128")
        if base.num_heads != PARENT_HEADS or base.num_v_heads != PARENT_HEADS:
            raise ValueError("P-GDN3-053 requires H4")
        if base.head_k_dim != HEAD_DIM or base.head_v_dim != HEAD_DIM:
            raise ValueError("P-GDN3-053 requires K32/V32")
        if base.mode != "chunk" or not base.use_short_conv:
            raise ValueError("P-GDN3-053 requires chunk GDN2 with short convolution")
        if base.allow_neg_eigval:
            raise ValueError("P-GDN3-053 requires bounded positive erase gates")

        self.base = base
        self.secondary_q_proj = nn.Linear(
            base.hidden_size,
            base.key_dim,
            bias=False,
        )
        self.secondary_k_proj = nn.Linear(
            base.hidden_size,
            base.key_dim,
            bias=False,
        )
        self.secondary_q_conv1d = ShortConvolution(
            hidden_size=base.key_dim,
            kernel_size=base.conv_size,
            bias=base.conv_bias,
            activation="silu",
        )
        self.secondary_k_conv1d = ShortConvolution(
            hidden_size=base.key_dim,
            kernel_size=base.conv_size,
            bias=base.conv_bias,
            activation="silu",
        )
        self.last_diagnostics: dict[str, torch.Tensor] = {}

    @property
    def parent_heads(self) -> int:
        return int(self.base.num_heads)

    @property
    def num_heads(self) -> int:
        return int(BANK_COUNT * self.base.num_heads)

    @property
    def num_v_heads(self) -> int:
        return int(BANK_COUNT * self.base.num_v_heads)

    @property
    def head_k_dim(self) -> int:
        return int(self.base.head_k_dim)

    @property
    def head_v_dim(self) -> int:
        return int(self.base.head_v_dim)

    @property
    def layer_idx(self) -> int:
        if self.base.layer_idx is None:
            raise RuntimeError("P-GDN3-053 requires an explicit layer index")
        return int(self.base.layer_idx)

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
            raise ValueError("P-GDN3-053 uses fixed unpadded MQAR batches")
        if kwargs.get("cu_seqlens") is not None:
            raise ValueError("P-GDN3-053 does not use packed sequences")

        layer = self.base
        q_len = hidden_states.shape[1]
        last_state = get_layer_cache(layer, past_key_values)
        conv_state_q, conv_state_k, conv_state_v = None, None, None
        if last_state is not None:
            conv_state_q, conv_state_k, conv_state_v = last_state["conv_state"]

        primary_q, conv_state_q = layer.q_conv1d(
            x=layer.q_proj(hidden_states),
            cache=conv_state_q,
            output_final_state=use_cache,
        )
        primary_k, conv_state_k = layer.k_conv1d(
            x=layer.k_proj(hidden_states),
            cache=conv_state_k,
            output_final_state=use_cache,
        )
        value, conv_state_v = layer.v_conv1d(
            x=layer.v_proj(hidden_states),
            cache=conv_state_v,
            output_final_state=use_cache,
        )
        secondary_q, _ = self.secondary_q_conv1d(
            x=self.secondary_q_proj(hidden_states),
            cache=None,
            output_final_state=False,
        )
        secondary_k, _ = self.secondary_k_conv1d(
            x=self.secondary_k_proj(hidden_states),
            cache=None,
            output_final_state=False,
        )

        decay = F.softplus(layer.f_proj(hidden_states).float() + layer.dt_bias)
        erase = layer.b_proj(hidden_states).sigmoid()
        write = layer.w_proj(hidden_states).sigmoid()
        primary_q, primary_k, secondary_q, secondary_k, decay, erase = (
            rearrange(tensor, "... (h d) -> ... h d", d=layer.head_k_dim)
            for tensor in (
                primary_q,
                primary_k,
                secondary_q,
                secondary_k,
                decay,
                erase,
            )
        )
        value = rearrange(value, "... (h d) -> ... h d", d=layer.head_v_dim)
        write = rearrange(write, "... (h d) -> ... h d", d=layer.head_v_dim)
        decay = -layer.A_log.float().exp().unsqueeze(-1) * decay

        query = torch.cat((primary_q, secondary_q), dim=2)
        key = torch.cat((primary_k, secondary_k), dim=2)
        paired_value = torch.cat((value, value), dim=2)
        paired_decay = torch.cat((decay, decay), dim=2)
        paired_erase = torch.cat((erase, erase), dim=2)
        paired_write = torch.cat((write, write), dim=2)
        recurrent_state = (
            last_state["recurrent_state"] if last_state is not None else None
        )
        if recurrent_state is not None and recurrent_state.shape[1:] != (
            self.num_v_heads,
            self.head_k_dim,
            self.head_v_dim,
        ):
            raise RuntimeError(
                f"Unexpected redundant state shape: {tuple(recurrent_state.shape)}"
            )

        paired_output, recurrent_state = chunk_gdn2(
            q=query,
            k=key,
            v=paired_value,
            g=paired_decay,
            b=paired_erase,
            w=paired_write,
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

        primary_output, secondary_output = paired_output.split(
            self.parent_heads,
            dim=2,
        )
        output = 0.5 * (primary_output + secondary_output)

        with torch.no_grad():
            if recurrent_state is None:
                state = paired_output.new_zeros(
                    hidden_states.shape[0],
                    self.num_v_heads,
                    self.head_k_dim,
                    self.head_v_dim,
                )
            else:
                state = recurrent_state.float()
            primary_state, secondary_state = state.split(
                self.parent_heads,
                dim=1,
            )
            primary_flat = primary_state.flatten(-2)
            secondary_flat = secondary_state.flatten(-2)
            state_difference = secondary_state - primary_state
            state_scale = primary_state.square().mean().sqrt().clamp_min(1e-8)
            secondary_address = torch.cat((secondary_q, secondary_k), dim=-1)
            secondary_board = secondary_address.float().square().mean(
                dim=(1, 2, 3)
            ).sqrt()
            output_scale = primary_output.float().square().mean().sqrt().clamp_min(
                1e-8
            )
            self.last_diagnostics = {
                "primary_q_secondary_q_cosine": F.cosine_similarity(
                    primary_q.float().flatten(1),
                    secondary_q.float().flatten(1),
                    dim=-1,
                    eps=1e-8,
                ).mean(),
                "primary_k_secondary_k_cosine": F.cosine_similarity(
                    primary_k.float().flatten(1),
                    secondary_k.float().flatten(1),
                    dim=-1,
                    eps=1e-8,
                ).mean(),
                "primary_address_contrast": _address_contrast(
                    primary_q,
                    primary_k,
                ),
                "secondary_address_contrast": _address_contrast(
                    secondary_q,
                    secondary_k,
                ),
                "secondary_address_token_std": secondary_address.float().mean(
                    dim=-1
                ).std(dim=1, unbiased=False).mean(),
                "secondary_address_board_std": secondary_board.std(
                    unbiased=False
                ),
                "primary_output_rms": _rms(primary_output),
                "secondary_output_rms": _rms(secondary_output),
                "output_relative_disagreement": (
                    (secondary_output.float() - primary_output.float())
                    .square()
                    .mean()
                    .sqrt()
                    / output_scale
                ),
                "output_cosine": F.cosine_similarity(
                    primary_output.float().flatten(1),
                    secondary_output.float().flatten(1),
                    dim=-1,
                    eps=1e-8,
                ).mean(),
                "primary_state_rms": _rms(primary_state),
                "secondary_state_rms": _rms(secondary_state),
                "state_relative_difference": _rms(state_difference) / state_scale,
                "state_cosine_mean": F.cosine_similarity(
                    primary_flat,
                    secondary_flat,
                    dim=-1,
                    eps=1e-8,
                ).mean(),
                "state_cosine_max": F.cosine_similarity(
                    primary_flat,
                    secondary_flat,
                    dim=-1,
                    eps=1e-8,
                ).max(),
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


class ZoologyRedundantAddressFutureSeedMixer(ZoologyGDN2FutureSeedMixer):
    """Native FutureSeed over two full independently addressed state banks."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        parent_gate = self.future_seed_logit.detach().clone()
        self.layer = RedundantAddressGatedDeltaNet2(self.layer)
        self.future_seed_logit = nn.Parameter(
            torch.cat((parent_gate, parent_gate), dim=1)
        )


def _is_extra_parameter(name: str) -> bool:
    return ".sequence_mixer.layer.secondary_" in name


def load_matched_parent_state(
    model: nn.Module,
    control_state: dict[str, torch.Tensor],
) -> None:
    loaded: dict[str, torch.Tensor] = {}
    extra_names: list[str] = []
    for name, tensor in model.state_dict().items():
        if _is_extra_parameter(name):
            loaded[name] = tensor
            extra_names.append(name)
            continue
        parent_name = name.replace(
            ".sequence_mixer.layer.base.",
            ".sequence_mixer.layer.",
        )
        if parent_name not in control_state:
            raise RuntimeError(f"Missing matched parent tensor: {parent_name}")
        parent = control_state[parent_name]
        if name.endswith("future_seed_logit"):
            parent = torch.cat((parent, parent), dim=1)
        if parent.shape != tensor.shape:
            raise RuntimeError(
                f"Matched tensor shape mismatch for {name}: "
                f"{tuple(parent.shape)} != {tuple(tensor.shape)}"
            )
        loaded[name] = parent
    if len(extra_names) != 8:
        raise RuntimeError(
            f"Expected eight independent address tensors, got {extra_names}"
        )
    model.load_state_dict(loaded, strict=True)


def parent_parameter_hash(model: nn.Module) -> str:
    digest = hashlib.sha256()
    rows: list[tuple[str, torch.Tensor]] = []
    for name, parameter in model.named_parameters():
        if _is_extra_parameter(name):
            continue
        parent_name = name.replace(
            ".sequence_mixer.layer.base.",
            ".sequence_mixer.layer.",
        )
        value = parameter
        if name.endswith("future_seed_logit"):
            value = parameter[:, :PARENT_HEADS]
        rows.append((parent_name, value))
    for name, value in sorted(rows):
        digest.update(name.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def redundant_address_diagnostics(model: nn.Module) -> dict[str, Any]:
    mixers = [block.sequence_mixer for block in model.backbone.layers]
    if len(mixers) != 2 or not all(
        isinstance(mixer, ZoologyRedundantAddressFutureSeedMixer)
        for mixer in mixers
    ):
        raise RuntimeError("Expected exactly two redundant-address mixers")
    rows = []
    for mixer in mixers:
        diagnostics = mixer.layer.last_diagnostics
        if not diagnostics:
            raise RuntimeError("Redundant-address diagnostics were not populated")
        rows.append(
            {
                "layer_idx": mixer.layer_idx,
                **{
                    name: float(value.detach().float().item())
                    for name, value in diagnostics.items()
                },
            }
        )
    return {
        "active_layers": len(rows),
        "bank_count": BANK_COUNT,
        "parent_heads": PARENT_HEADS,
        "physical_heads": BANK_COUNT * PARENT_HEADS,
        "state_values_per_layer": REDUNDANT_STATE_VALUES,
        "official_scans_per_layer": 1,
        "fixed_equal_read": True,
        "token_routing": False,
        "independent_address_parameters": sum(
            parameter.numel()
            for mixer in mixers
            for name, parameter in mixer.layer.named_parameters()
            if name.startswith("secondary_")
        ),
        "per_layer": rows,
    }
