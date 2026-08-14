from __future__ import annotations

import copy
import hashlib
import inspect
import math
import os
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from einops import rearrange
from torch.nn import functional as F

from fla.layers.gdn2 import GatedDeltaNet2
from fla.layers.utils import get_layer_cache, update_layer_cache
from fla.ops.gdn2 import chunk_gdn2

from experiments.zoology_mqar.gdn2_futureseed import (
    PINNED_FLA_SHA,
    FutureSeedLanguageModel,
    ZoologyGDN2FutureSeedMixer,
)


NATIVE_MIXER_NAME = (
    "experiments.zoology_mqar.gdn2_futureseed.ZoologyGDN2FutureSeedMixer"
)


class ChunkLocalBiAxisGatedDeltaNet2(nn.Module):
    """GDN2 with bounded persistent value decay and physical chunk carries."""

    anchor_size = 64
    value_groups = 8
    inverse_bound = 4.0

    def __init__(self, base: GatedDeltaNet2) -> None:
        super().__init__()
        if base.mode != "chunk":
            raise ValueError("Chunk-local Bi-Axis requires official chunk GDN2")
        if base.num_v_heads != base.num_heads:
            raise ValueError("The first Bi-Axis test requires matched K/V heads")
        if base.head_v_dim % self.value_groups:
            raise ValueError("V dimension must be divisible by eight fixed groups")
        self.base = base
        self.value_decay_proj = nn.Linear(
            base.hidden_size,
            base.num_v_heads * self.value_groups,
            bias=False,
        )
        nn.init.zeros_(self.value_decay_proj.weight)
        self.last_diagnostics: dict[str, float | int] | None = None

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
    def max_log_decay_per_token(self) -> float:
        return math.log(self.inverse_bound) / self.anchor_size

    def value_log_decay(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch, length, _ = hidden_states.shape
        with torch.autocast(device_type=hidden_states.device.type, enabled=False):
            raw = self.value_decay_proj(hidden_states.float()).view(
                batch,
                length,
                self.num_v_heads,
                self.value_groups,
            )
            zero = F.softplus(raw.new_zeros(()))
            grouped = torch.clamp(
                zero - F.softplus(raw),
                min=-self.max_log_decay_per_token,
                max=0.0,
            )
            return torch.repeat_interleave(
                grouped,
                self.head_v_dim // self.value_groups,
                dim=-1,
            )

    def run_chunk_local_recurrence(
        self,
        *,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor,
        b: torch.Tensor,
        w: torch.Tensor,
        value_log_decay: torch.Tensor,
        recurrent_state: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        outputs = []
        physical_state = recurrent_state
        scale_min = value_log_decay.new_ones(())
        scale_max = value_log_decay.new_ones(())
        inverse_max = value_log_decay.new_ones(())
        state_rms = value_log_decay.new_zeros(())

        for start in range(0, q.shape[1], self.anchor_size):
            stop = min(start + self.anchor_size, q.shape[1])
            local_log = value_log_decay[:, start:stop].float().cumsum(dim=1)
            local_scale = local_log.exp()
            transformed_v = (
                v[:, start:stop].float() / local_scale
            ).to(dtype=v.dtype)
            transformed_output, transformed_state = chunk_gdn2(
                q=q[:, start:stop],
                k=k[:, start:stop],
                v=transformed_v,
                g=g[:, start:stop],
                b=b[:, start:stop],
                w=w[:, start:stop],
                initial_state=physical_state,
                output_final_state=True,
                use_qk_l2norm_in_kernel=True,
            )
            output = (
                transformed_output.float() * local_scale
            ).to(dtype=transformed_output.dtype)
            physical_state = transformed_state.float() * local_scale[:, -1].unsqueeze(-2)
            outputs.append(output)
            with torch.no_grad():
                scale_min = torch.minimum(scale_min, local_scale.min())
                scale_max = torch.maximum(scale_max, local_scale.max())
                inverse_max = torch.maximum(inverse_max, local_scale.reciprocal().max())
                state_rms = physical_state.square().mean().sqrt()

        if physical_state is None:
            raise RuntimeError("Bi-Axis recurrence did not produce terminal state")
        with torch.no_grad():
            grouped = value_log_decay[..., :: self.head_v_dim // self.value_groups]
            abs_grouped = grouped.abs()
            self.last_diagnostics = {
                "chunks": len(outputs),
                "log_decay_abs": float(abs_grouped.mean().item()),
                "active_fraction": float((abs_grouped > 0).float().mean().item()),
                "group_std": float(abs_grouped.mean(dim=(0, 1, 2)).std(unbiased=False).item()),
                "batch_std": float(abs_grouped.mean(dim=(1, 2, 3)).std(unbiased=False).item()),
                "token_std": float(abs_grouped.mean(dim=(0, 2, 3)).std(unbiased=False).item()),
                "local_scale_min": float(scale_min.item()),
                "local_scale_max": float(scale_max.item()),
                "local_inverse_max": float(inverse_max.item()),
                "terminal_state_rms": float(state_rms.item()),
                "weight_rms": float(
                    self.value_decay_proj.weight.float().square().mean().sqrt().item()
                ),
            }
        return torch.cat(outputs, dim=1), physical_state

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
            raise ValueError("Chunk-local Bi-Axis does not use padded MQAR batches")
        if kwargs.get("cu_seqlens") is not None:
            raise ValueError("The first Bi-Axis test uses fixed-length batches")

        layer = self.base
        q_len = hidden_states.shape[1]
        last_state = get_layer_cache(layer, past_key_values)
        if layer.use_short_conv:
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
        else:
            q = F.silu(layer.q_proj(hidden_states))
            k = F.silu(layer.k_proj(hidden_states))
            v = F.silu(layer.v_proj(hidden_states))
            conv_state_q, conv_state_k, conv_state_v = None, None, None

        g = F.softplus(layer.f_proj(hidden_states).float() + layer.dt_bias)
        b = layer.b_proj(hidden_states).sigmoid()
        w = layer.w_proj(hidden_states).sigmoid()
        q, k, g = (
            rearrange(x, "... (h d) -> ... h d", d=layer.head_k_dim)
            for x in (q, k, g)
        )
        v = rearrange(v, "... (h d) -> ... h d", d=layer.head_v_dim)
        b = rearrange(b, "... (h d) -> ... h d", d=layer.head_k_dim)
        w = rearrange(w, "... (h d) -> ... h d", d=layer.head_v_dim)
        g = -layer.A_log.float().exp().unsqueeze(-1) * g
        if layer.allow_neg_eigval:
            b = b * 2.0

        recurrent_state = last_state["recurrent_state"] if last_state is not None else None
        value_log_decay = self.value_log_decay(hidden_states)
        output, recurrent_state = self.run_chunk_local_recurrence(
            q=q,
            k=k,
            v=v,
            g=g,
            b=b,
            w=w,
            value_log_decay=value_log_decay,
            recurrent_state=recurrent_state,
        )
        update_layer_cache(
            layer,
            past_key_values,
            recurrent_state=recurrent_state,
            conv_state=(conv_state_q, conv_state_k, conv_state_v),
            offset=q_len,
        )
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


class ZoologyChunkLocalBiAxisFutureSeedMixer(ZoologyGDN2FutureSeedMixer):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        module_path = Path(inspect.getfile(GatedDeltaNet2)).resolve()
        if os.environ.get("FLA_EXPECTED_SOURCE_SHA") != PINNED_FLA_SHA:
            raise RuntimeError("Pinned official FLA source SHA was not asserted")
        if Path("/huyang2/double-loop/.cache/fla-versions") not in module_path.parents:
            raise RuntimeError(f"Unexpected FLA GDN2 source: {module_path}")
        self.layer = ChunkLocalBiAxisGatedDeltaNet2(self.layer)


def make_chunk_local_biaxis_model(model_config: Any) -> FutureSeedLanguageModel:
    parent_config = copy.deepcopy(model_config)
    parent_config.sequence_mixer.name = NATIVE_MIXER_NAME
    parent = FutureSeedLanguageModel(parent_config)
    candidate = FutureSeedLanguageModel(copy.deepcopy(model_config))
    candidate_state = candidate.state_dict()
    with torch.no_grad():
        for name, tensor in parent.state_dict().items():
            target = name.replace(
                ".sequence_mixer.layer.",
                ".sequence_mixer.layer.base.",
            )
            if target not in candidate_state:
                target = name
            if target not in candidate_state or candidate_state[target].shape != tensor.shape:
                raise RuntimeError(f"Cannot map native parent tensor {name} -> {target}")
            candidate_state[target].copy_(tensor)
        for name, tensor in candidate_state.items():
            if name.endswith("layer.value_decay_proj.weight"):
                tensor.zero_()
    candidate.load_state_dict(candidate_state)
    return candidate


def parent_parameter_hash(model: nn.Module) -> str:
    tensors = []
    for name, parameter in model.named_parameters():
        if name.endswith("layer.value_decay_proj.weight"):
            continue
        parent_name = name.replace(
            ".sequence_mixer.layer.base.",
            ".sequence_mixer.layer.",
        )
        tensors.append((parent_name, parameter))
    digest = hashlib.sha256()
    for name, parameter in sorted(tensors):
        digest.update(name.encode())
        digest.update(parameter.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def chunk_local_biaxis_diagnostics(model: nn.Module) -> dict[str, Any]:
    mixers = [
        block.sequence_mixer
        for block in model.backbone.layers
        if isinstance(block.sequence_mixer, ZoologyChunkLocalBiAxisFutureSeedMixer)
    ]
    rows = []
    for mixer in mixers:
        layer = mixer.layer
        if layer.last_diagnostics is None:
            raise RuntimeError("Bi-Axis diagnostics were not produced")
        rows.append(dict(layer.last_diagnostics))
    return {
        "active_layers": len(rows),
        "parameter_delta": sum(
            mixer.layer.value_decay_proj.weight.numel() for mixer in mixers
        ),
        "anchor_size": ChunkLocalBiAxisGatedDeltaNet2.anchor_size,
        "value_groups": ChunkLocalBiAxisGatedDeltaNet2.value_groups,
        "inverse_bound": ChunkLocalBiAxisGatedDeltaNet2.inverse_bound,
        "per_layer": rows,
    }
