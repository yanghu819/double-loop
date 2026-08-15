from __future__ import annotations

import hashlib
from typing import Any

import torch
import torch.nn.functional as F
from einops import rearrange, repeat
from torch import nn

from fla.layers.gdn2 import GatedDeltaNet2
from fla.layers.utils import get_layer_cache, update_layer_cache
from fla.ops.gdn2 import chunk_gdn2

from experiments.zoology_mqar.gdn2_futureseed import (
    FutureSeedLanguageModel,
    ZoologyGDN2FutureSeedMixer,
)


EXPECTED_NEW_PARAMETERS = 8
EXPECTED_STATE_VALUES = 4 * 32 * 32


def _rms(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.float().square().mean().sqrt()


class LaggedCommitGDN2(nn.Module):
    """Official GDN2 with a coherent causal lag on the commit address."""

    def __init__(self, base: GatedDeltaNet2) -> None:
        super().__init__()
        if type(base) is not GatedDeltaNet2:
            raise TypeError("P-GDN3-051 requires exact official GDN2")
        if base.hidden_size != 128 or base.num_heads != 4:
            raise ValueError("P-GDN3-051 is fixed at D128/H4")
        if base.num_v_heads != 4 or base.head_k_dim != 32:
            raise ValueError("P-GDN3-051 is fixed at H4/K32")
        if base.head_v_dim != 32 or base.mode != "chunk":
            raise ValueError("P-GDN3-051 is fixed at V32/chunk mode")
        if not base.use_short_conv:
            raise ValueError("P-GDN3-051 requires the Triton short convolution")
        self.base = base
        self.lag_mix_logits = nn.Parameter(torch.zeros(base.num_heads))
        self.lag_enabled = True
        self.capture_lagged_commit = False
        self.last_lagged_commit: dict[str, torch.Tensor] = {}

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
        if self.base.layer_idx is None:
            raise RuntimeError("P-GDN3-051 requires an explicit layer index")
        return int(self.base.layer_idx)

    def set_lag_enabled(self, enabled: bool) -> None:
        self.lag_enabled = bool(enabled)

    def set_capture(self, enabled: bool) -> None:
        self.capture_lagged_commit = bool(enabled)
        if enabled:
            self.last_lagged_commit.clear()

    @staticmethod
    def previous_key(native_key: torch.Tensor) -> torch.Tensor:
        if native_key.ndim != 4 or native_key.shape[1] < 1:
            raise ValueError("Expected a nonempty BTHK native key sequence")
        return torch.cat((native_key[:, :1], native_key[:, :-1]), dim=1)

    @classmethod
    def lagged_key(
        cls,
        native_key: torch.Tensor,
        lag_mix: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if lag_mix.shape != (native_key.shape[2],):
            raise ValueError("Lag mix must contain one scalar per head")
        previous = cls.previous_key(native_key)
        mix = lag_mix.to(device=native_key.device, dtype=native_key.dtype)
        mix = mix.view(1, 1, native_key.shape[2], 1)
        committed = native_key + mix * (previous - native_key)
        return committed, previous

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
            raise ValueError("P-GDN3-051 uses fixed unpadded MQAR batches")

        layer = self.base
        q_len = hidden_states.shape[1]
        last_state = get_layer_cache(layer, past_key_values)
        cu_seqlens = kwargs.get("cu_seqlens")
        conv_state_q, conv_state_k, conv_state_v = None, None, None
        if last_state is not None:
            conv_state_q, conv_state_k, conv_state_v = last_state["conv_state"]

        q, conv_state_q = layer.q_conv1d(
            x=layer.q_proj(hidden_states),
            cache=conv_state_q,
            output_final_state=use_cache,
            cu_seqlens=cu_seqlens,
        )
        native_key, conv_state_k = layer.k_conv1d(
            x=layer.k_proj(hidden_states),
            cache=conv_state_k,
            output_final_state=use_cache,
            cu_seqlens=cu_seqlens,
        )
        value, conv_state_v = layer.v_conv1d(
            x=layer.v_proj(hidden_states),
            cache=conv_state_v,
            output_final_state=use_cache,
            cu_seqlens=cu_seqlens,
        )

        g = F.softplus(layer.f_proj(hidden_states).float() + layer.dt_bias)
        erase_gate = layer.b_proj(hidden_states).sigmoid()
        write_gate = layer.w_proj(hidden_states).sigmoid()
        q, native_key, g, erase_gate = (
            rearrange(tensor, "... (h d) -> ... h d", d=layer.head_k_dim)
            for tensor in (q, native_key, g, erase_gate)
        )
        value = rearrange(value, "... (h d) -> ... h d", d=layer.head_v_dim)
        write_gate = rearrange(
            write_gate,
            "... (h d) -> ... h d",
            d=layer.head_v_dim,
        )
        g = -layer.A_log.float().exp().unsqueeze(-1) * g
        if layer.num_v_heads > layer.num_heads:
            q, native_key, g, erase_gate = (
                repeat(
                    tensor,
                    "... h d -> ... (h g) d",
                    g=layer.num_v_heads // layer.num_heads,
                )
                for tensor in (q, native_key, g, erase_gate)
            )
        if layer.allow_neg_eigval:
            erase_gate = erase_gate * 2.0

        lag_mix = torch.tanh(self.lag_mix_logits)
        if not self.lag_enabled:
            lag_mix = torch.zeros_like(lag_mix)
        committed_key, previous_key = self.lagged_key(native_key, lag_mix)
        recurrent_state = (
            last_state["recurrent_state"] if last_state is not None else None
        )
        output, recurrent_state = chunk_gdn2(
            q=q,
            k=committed_key,
            v=value,
            g=g,
            b=erase_gate,
            w=write_gate,
            initial_state=recurrent_state,
            output_final_state=use_cache,
            use_qk_l2norm_in_kernel=True,
            cu_seqlens=cu_seqlens,
        )
        update_layer_cache(
            layer,
            past_key_values,
            recurrent_state=recurrent_state,
            conv_state=(conv_state_q, conv_state_k, conv_state_v),
            offset=q_len,
        )
        if self.capture_lagged_commit:
            if recurrent_state is None:
                raise RuntimeError("Lagged commit capture requires terminal state")
            self.last_lagged_commit = {
                "native_key": native_key.detach(),
                "previous_key": previous_key.detach(),
                "committed_key": committed_key.detach(),
                "lag_mix": lag_mix.detach(),
                "terminal_state": recurrent_state.detach(),
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


class ZoologyLaggedCommitFutureSeedMixer(ZoologyGDN2FutureSeedMixer):
    """Native FutureSeed over a coherent lagged-address official GDN2."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.layer = LaggedCommitGDN2(self.layer)


class LaggedCommitLanguageModel(FutureSeedLanguageModel):
    def __init__(self, config) -> None:
        super().__init__(config)
        for block in self.backbone.layers:
            mixer = block.sequence_mixer
            if type(mixer) is not ZoologyLaggedCommitFutureSeedMixer:
                raise TypeError("P-GDN3-051 requires strict lagged mixers")
            nn.init.zeros_(mixer.layer.lag_mix_logits)

    def set_lag_enabled(self, enabled: bool) -> None:
        for block in self.backbone.layers:
            block.sequence_mixer.layer.set_lag_enabled(enabled)


def load_matched_parent_state(
    model: nn.Module,
    control_state: dict[str, torch.Tensor],
) -> None:
    loaded: dict[str, torch.Tensor] = {}
    for name, tensor in model.state_dict().items():
        if name.endswith(".sequence_mixer.layer.lag_mix_logits"):
            loaded[name] = torch.zeros_like(tensor)
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
    model.load_state_dict(loaded, strict=True)


def parent_parameter_hash(model: nn.Module) -> str:
    digest = hashlib.sha256()
    rows: list[tuple[str, torch.Tensor]] = []
    for name, parameter in model.named_parameters():
        if name.endswith(".sequence_mixer.layer.lag_mix_logits"):
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


@torch.no_grad()
def lagged_commit_diagnostics(
    model: LaggedCommitLanguageModel,
    inputs: torch.Tensor,
) -> dict[str, Any]:
    mixers = [block.sequence_mixer for block in model.backbone.layers]
    if len(mixers) != 2 or not all(
        type(mixer) is ZoologyLaggedCommitFutureSeedMixer for mixer in mixers
    ):
        raise RuntimeError("Expected exactly two lagged-commit mixers")
    for mixer in mixers:
        mixer.layer.set_capture(True)
    try:
        model.eval()(inputs)
    finally:
        for mixer in mixers:
            mixer.layer.set_capture(False)

    value_mask = (
        (inputs[:, 1:] >= 160)
        & (inputs[:, 1:] < 256)
        & (inputs[:, :-1] >= 64)
        & (inputs[:, :-1] < 160)
    )
    if not bool(value_mask.any()):
        raise RuntimeError("No serialized key-value positions were captured")

    rows = []
    all_mix = []
    for mixer in mixers:
        captured = mixer.layer.last_lagged_commit
        required = {
            "native_key",
            "previous_key",
            "committed_key",
            "lag_mix",
            "terminal_state",
        }
        if set(captured) != required:
            raise RuntimeError(f"Incomplete lagged capture: {set(captured)}")
        native = captured["native_key"].float()
        previous = captured["previous_key"].float()
        committed = captured["committed_key"].float()
        state = captured["terminal_state"].float()
        mix = captured["lag_mix"].float()
        all_mix.append(mix)
        current_cos = F.cosine_similarity(
            F.normalize(native[:, 1:], dim=-1),
            F.normalize(previous[:, 1:], dim=-1),
            dim=-1,
        )
        commit_cos = F.cosine_similarity(
            F.normalize(committed[:, 1:], dim=-1),
            F.normalize(previous[:, 1:], dim=-1),
            dim=-1,
        )
        expanded_mask = value_mask.unsqueeze(-1).expand_as(current_cos)
        state_board = state.square().mean(dim=(-1, -2, -3)).sqrt()
        rows.append(
            {
                "layer_idx": mixer.layer_idx,
                "lag_mix": [float(value) for value in mix.cpu().tolist()],
                "positive_heads_at_least_0.02": int((mix >= 0.02).sum().item()),
                "committed_native_relative_rms": float(
                    (_rms(committed - native) / _rms(native).clamp_min(1e-8)).item()
                ),
                "token0_identity_max_error": float(
                    (committed[:, 0] - native[:, 0]).abs().max().item()
                ),
                "value_position_native_to_previous_cosine": float(
                    current_cos[expanded_mask].mean().item()
                ),
                "value_position_commit_to_previous_cosine": float(
                    commit_cos[expanded_mask].mean().item()
                ),
                "value_position_cosine_gain": float(
                    (commit_cos[expanded_mask] - current_cos[expanded_mask])
                    .mean()
                    .item()
                ),
                "terminal_state_rms": float(_rms(state).item()),
                "terminal_state_board_std": float(
                    state_board.std(unbiased=False).item()
                ),
            }
        )
    mix_vector = torch.cat(all_mix)
    return {
        "active_layers": len(rows),
        "finite_lag_logits": bool(
            all(torch.isfinite(mixer.layer.lag_mix_logits).all() for mixer in mixers)
        ),
        "global_lag_mix_mean": float(mix_vector.mean().item()),
        "global_lag_mix_abs_mean": float(mix_vector.abs().mean().item()),
        "positive_heads_at_least_0.02": int((mix_vector >= 0.02).sum().item()),
        "new_parameters": EXPECTED_NEW_PARAMETERS,
        "new_recurrent_state_values": 0,
        "logical_scans_per_layer": 1,
        "per_layer": rows,
    }
