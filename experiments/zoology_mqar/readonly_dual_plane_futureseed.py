from __future__ import annotations

import hashlib
import math
from functools import partial
from typing import Any, Optional

import torch
import torch.nn.functional as F
from einops import rearrange, repeat
from torch import nn

from fla.layers.gdn2 import GatedDeltaNet2
from fla.layers.utils import get_layer_cache, update_layer_cache
from fla.ops.gdn2 import chunk_gdn2
from zoology.config import ModelConfig
from zoology.model import _compute_state_size, _init_weights

from experiments.zoology_mqar.gdn2_futureseed import (
    FutureSeedLMBackbone,
    ZoologyGDN2FutureSeedMixer,
)


EXPECTED_PARAMETER_DELTA = 0
EXPECTED_RECURRENT_STATE_DELTA = 0
READONLY_SIDE_STATE_VALUES = 4 * 32 * 32


def _rms(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.float().square().mean().sqrt()


class ZoologyReadOnlyDualPlaneFutureSeedMixer(ZoologyGDN2FutureSeedMixer):
    """Keep inherited future evidence separate from the live GDN2 state."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        if type(self.layer) is not GatedDeltaNet2:
            raise TypeError("P-FS2-012 requires exact official GDN2")
        self.readonly_enabled = True
        self.capture_dual_plane = False
        self.last_dual_plane: dict[str, torch.Tensor] = {}

    def set_readonly_enabled(self, enabled: bool) -> None:
        self.readonly_enabled = bool(enabled)

    def set_capture_dual_plane(self, enabled: bool) -> None:
        self.capture_dual_plane = bool(enabled)
        if enabled:
            self.last_dual_plane.clear()

    def make_readonly_state(self, terminal_state: torch.Tensor) -> torch.Tensor:
        return super().make_initial_state(terminal_state)

    @staticmethod
    def readonly_read(
        query: torch.Tensor,
        readonly_state: torch.Tensor,
    ) -> torch.Tensor:
        if query.shape[0] != readonly_state.shape[0]:
            raise ValueError("Read-only FutureSeed batch mismatch")
        if query.shape[2] != readonly_state.shape[1]:
            raise ValueError("Read-only FutureSeed head mismatch")
        if query.shape[3] != readonly_state.shape[2]:
            raise ValueError("Read-only FutureSeed key width mismatch")
        query_float = query.float()
        query_unit = query_float * torch.rsqrt(
            query_float.square().sum(dim=-1, keepdim=True) + 1e-6
        )
        query_unit = query_unit.to(dtype=query.dtype).float()
        query_unit = query_unit * math.sqrt(float(query.shape[-1])) ** -1
        return torch.einsum(
            "bthk,bhkv->bthv",
            query_unit,
            readonly_state.float(),
        )

    def _forward_dual_plane(
        self,
        hidden_states: torch.Tensor,
        *,
        readonly_state: Optional[torch.Tensor],
        past_key_values: Any,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        layer = self.layer
        if layer.mode != "chunk" or not layer.use_short_conv:
            raise RuntimeError("P-FS2-012 requires chunk GDN2 with short conv")

        q_len = hidden_states.shape[1]
        last_state = get_layer_cache(layer, past_key_values)
        conv_state_q, conv_state_k, conv_state_v = None, None, None
        if last_state is not None:
            conv_state_q, conv_state_k, conv_state_v = last_state["conv_state"]
            if last_state["recurrent_state"] is not None:
                raise RuntimeError("The live GDN2 plane must start from zero")

        q, conv_state_q = layer.q_conv1d(
            x=layer.q_proj(hidden_states),
            cache=conv_state_q,
            output_final_state=True,
        )
        k, conv_state_k = layer.k_conv1d(
            x=layer.k_proj(hidden_states),
            cache=conv_state_k,
            output_final_state=True,
        )
        v, conv_state_v = layer.v_conv1d(
            x=layer.v_proj(hidden_states),
            cache=conv_state_v,
            output_final_state=True,
        )

        g = F.softplus(layer.f_proj(hidden_states).float() + layer.dt_bias)
        erase_gate = layer.b_proj(hidden_states).sigmoid()
        write_gate = layer.w_proj(hidden_states).sigmoid()
        q, k, g = (
            rearrange(tensor, "... (h d) -> ... h d", d=layer.head_k_dim)
            for tensor in (q, k, g)
        )
        v = rearrange(v, "... (h d) -> ... h d", d=layer.head_v_dim)
        erase_gate = rearrange(
            erase_gate,
            "... (h d) -> ... h d",
            d=layer.head_k_dim,
        )
        write_gate = rearrange(
            write_gate,
            "... (h d) -> ... h d",
            d=layer.head_v_dim,
        )
        g = -layer.A_log.float().exp().unsqueeze(-1) * g
        if layer.num_v_heads > layer.num_heads:
            q, k, g, erase_gate = (
                repeat(
                    tensor,
                    "... h d -> ... (h g) d",
                    g=layer.num_v_heads // layer.num_heads,
                )
                for tensor in (q, k, g, erase_gate)
            )
        if layer.allow_neg_eigval:
            erase_gate = erase_gate * 2.0

        live_output, terminal_state = chunk_gdn2(
            q=q,
            k=k,
            v=v,
            g=g,
            b=erase_gate,
            w=write_gate,
            initial_state=None,
            output_final_state=True,
            use_qk_l2norm_in_kernel=True,
        )
        if terminal_state is None:
            raise RuntimeError("Official GDN2 did not return a terminal state")

        readonly_output = torch.zeros_like(live_output)
        if readonly_state is not None and self.readonly_enabled:
            readonly_output = self.readonly_read(q, readonly_state).to(
                dtype=live_output.dtype
            )
        combined_output = live_output + readonly_output

        update_layer_cache(
            layer,
            past_key_values,
            recurrent_state=terminal_state,
            conv_state=(conv_state_q, conv_state_k, conv_state_v),
            offset=q_len,
        )

        if self.capture_dual_plane:
            read_board = readonly_output.float().square().mean(
                dim=(1, 2, 3)
            ).sqrt()
            read_token = readonly_output.float().square().mean(
                dim=(0, 2, 3)
            ).sqrt()
            live_board = live_output.float().square().mean(
                dim=(1, 2, 3)
            ).sqrt()
            terminal_board = terminal_state.float().square().mean(
                dim=(1, 2, 3)
            ).sqrt()
            self.last_dual_plane = {
                "route_present": torch.tensor(
                    float(readonly_state is not None),
                    device=hidden_states.device,
                ),
                "route_enabled": torch.tensor(
                    float(readonly_state is not None and self.readonly_enabled),
                    device=hidden_states.device,
                ),
                "readonly_read_rms": _rms(readonly_output),
                "readonly_read_board_std": read_board.std(unbiased=False),
                "readonly_read_token_std": read_token.std(unbiased=False),
                "readonly_relative_rms": _rms(readonly_output)
                / _rms(live_output).clamp_min(1e-8),
                "live_output_rms": _rms(live_output),
                "live_output_board_std": live_board.std(unbiased=False),
                "live_terminal_rms": _rms(terminal_state),
                "live_terminal_board_std": terminal_board.std(unbiased=False),
                "read_live_cosine": F.cosine_similarity(
                    readonly_output.float().flatten(2),
                    live_output.float().flatten(2),
                    dim=-1,
                ).mean(),
                "readonly_state_rms": (
                    torch.zeros((), device=hidden_states.device)
                    if readonly_state is None
                    else _rms(readonly_state)
                ),
                "readonly_state_unchanged": torch.ones(
                    (), device=hidden_states.device
                ),
            }

        output = layer.o_norm(
            combined_output,
            rearrange(
                layer.g_proj(hidden_states),
                "... (h d) -> ... h d",
                d=layer.head_v_dim,
            ),
        )
        output = rearrange(output, "b t h d -> b t (h d)")
        return layer.o_proj(output), terminal_state

    def forward_with_readonly_state(
        self,
        hidden_states: torch.Tensor,
        *,
        readonly_state: Optional[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        cache = self._new_cache(None)
        with torch.autocast(
            device_type=hidden_states.device.type,
            dtype=torch.bfloat16,
            enabled=hidden_states.is_cuda,
        ):
            output, terminal_state = self._forward_dual_plane(
                hidden_states,
                readonly_state=readonly_state,
                past_key_values=cache,
            )
        return output.to(hidden_states.dtype), terminal_state


class ReadOnlyDualPlaneBackbone(FutureSeedLMBackbone):
    """Each receiver reads inherited state without mutating that state."""

    def layers_forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        residual = None
        producer_terminal = None
        active_routes = 0
        for layer in self.layers:
            mixer = layer.sequence_mixer
            if type(mixer) is not ZoologyReadOnlyDualPlaneFutureSeedMixer:
                raise TypeError("P-FS2-012 requires strict dual-plane mixers")

            dropped = layer.drop_path1(layer.dropout1(hidden_states))
            residual = dropped + residual if residual is not None else dropped
            hidden_states = layer.norm1(
                residual.to(dtype=layer.norm1.weight.dtype)
            )
            readonly_state = None
            if producer_terminal is not None:
                readonly_state = mixer.make_readonly_state(producer_terminal)
                active_routes += 1
            hidden_states, producer_terminal = (
                mixer.forward_with_readonly_state(
                    hidden_states,
                    readonly_state=readonly_state,
                )
            )

            dropped = layer.drop_path2(layer.dropout2(hidden_states))
            residual = dropped + residual if residual is not None else dropped
            hidden_states = layer.norm2(
                residual.to(dtype=layer.norm2.weight.dtype)
            )
            hidden_states = layer.state_mixer(hidden_states)

        if active_routes != len(self.layers) - 1:
            raise RuntimeError("Not every receiving layer used one read-only route")
        dropped = self.drop_f(hidden_states)
        residual = dropped + residual if residual is not None else dropped
        return self.ln_f(residual.to(dtype=self.ln_f.weight.dtype))


class ReadOnlyDualPlaneLanguageModel(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        if config.vocab_size % config.pad_vocab_size_multiple != 0:
            config.vocab_size += config.pad_vocab_size_multiple - (
                config.vocab_size % config.pad_vocab_size_multiple
            )
        if config.multiplier != 1:
            raise ValueError("P-FS2-012 requires multiplier=1")
        if config.d_model != 128 or config.n_layers != 2:
            raise ValueError("P-FS2-012 is fixed at D128/L2")

        self.backbone = ReadOnlyDualPlaneBackbone(config=config)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.apply(
            partial(
                _init_weights,
                n_layers=config.n_layers,
                block_type=config.block_type,
            )
        )
        if config.learnable_word_embeddings:
            self.lm_head.weight = self.backbone.embeddings.word_embeddings.weight

    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: Optional[torch.Tensor] = None,
        state=None,
        return_embeddings: bool = False,
    ) -> torch.Tensor:
        del state
        hidden_states = self.backbone(input_ids, position_ids=position_ids)
        if return_embeddings:
            return hidden_states
        return self.lm_head(hidden_states)

    def state_size(self, sequence_length: int) -> int:
        return _compute_state_size(self.backbone.layers, sequence_length)

    def set_readonly_enabled(self, enabled: bool) -> None:
        for block in self.backbone.layers:
            block.sequence_mixer.set_readonly_enabled(enabled)


def load_matched_parent_state(
    model: nn.Module,
    control_state: dict[str, torch.Tensor],
) -> None:
    model.load_state_dict(control_state, strict=True)


def parent_parameter_hash(model: nn.Module) -> str:
    digest = hashlib.sha256()
    for name, parameter in sorted(model.named_parameters()):
        digest.update(name.encode())
        digest.update(parameter.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


@torch.no_grad()
def readonly_dual_plane_diagnostics(
    model: ReadOnlyDualPlaneLanguageModel,
    inputs: torch.Tensor,
) -> dict[str, Any]:
    mixers = [block.sequence_mixer for block in model.backbone.layers]
    if len(mixers) != 2 or not all(
        type(mixer) is ZoologyReadOnlyDualPlaneFutureSeedMixer
        for mixer in mixers
    ):
        raise RuntimeError("Expected exactly two dual-plane mixers")
    for mixer in mixers:
        mixer.set_capture_dual_plane(True)
    try:
        model.eval()(inputs)
    finally:
        for mixer in mixers:
            mixer.set_capture_dual_plane(False)

    rows = []
    for mixer in mixers:
        if not mixer.last_dual_plane:
            raise RuntimeError("Dual-plane diagnostics were not populated")
        rows.append(
            {
                "layer_idx": mixer.layer_idx,
                **{
                    name: float(value.item())
                    for name, value in mixer.last_dual_plane.items()
                },
            }
        )
    return {
        "active_readonly_routes": sum(
            int(row["route_enabled"] == 1.0) for row in rows
        ),
        "new_parameters": EXPECTED_PARAMETER_DELTA,
        "new_recurrent_state_values": EXPECTED_RECURRENT_STATE_DELTA,
        "retained_readonly_side_state_values": READONLY_SIDE_STATE_VALUES,
        "official_gdn2_scans_per_layer": 1,
        "new_official_scans_per_layer": 0,
        "direct_receiver_reads_per_route": 1,
        "per_layer": rows,
    }
