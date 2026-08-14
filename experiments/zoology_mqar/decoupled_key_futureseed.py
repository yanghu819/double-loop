from __future__ import annotations

import copy
import hashlib
from typing import Any

import torch
import torch.nn.functional as F
from einops import rearrange
from torch import nn

from fla.layers.utils import get_layer_cache, update_layer_cache
from fla.modules import ShortConvolution
from fla.ops.generalized_delta_rule.dplr import chunk_dplr_delta_rule

from experiments.zoology_mqar.gdn2_futureseed import (
    FutureSeedLanguageModel,
    ZoologyGDN2FutureSeedMixer,
)


NATIVE_MIXER_NAME = (
    "experiments.zoology_mqar.gdn2_futureseed."
    "ZoologyGDN2FutureSeedMixer"
)


class DirectDecoupledKeyGDN2(nn.Module):
    """GDN2 with a separately learned erase key and the original write key."""

    def __init__(self, base: nn.Module) -> None:
        super().__init__()
        if base.num_heads != base.num_v_heads:
            raise ValueError("Decoupled-key GDN2 requires matched QK/V heads")
        if base.head_k_dim != base.head_v_dim:
            raise ValueError("P-GDN3-031 fixes K32/V32 state geometry")
        if base.mode != "chunk" or not base.use_short_conv:
            raise ValueError("P-GDN3-031 requires the chunk path and short convolution")
        self.base = base
        self.k_erase_proj = nn.Linear(
            base.hidden_size,
            base.key_dim,
            bias=False,
        )
        self.k_erase_conv1d = ShortConvolution(
            hidden_size=base.key_dim,
            kernel_size=base.conv_size,
            bias=base.conv_bias,
            activation="silu",
        )
        self.initial_tie_max_error = float("nan")
        self._capture = False
        self._captured: dict[str, torch.Tensor] = {}

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
            raise RuntimeError("P-GDN3-031 requires an explicit layer index")
        return int(self.base.layer_idx)

    def reset_erase_from_write(self) -> None:
        self.k_erase_proj.load_state_dict(self.base.k_proj.state_dict())
        self.k_erase_conv1d.load_state_dict(self.base.k_conv1d.state_dict())
        errors = []
        for erase, write in zip(
            self.k_erase_proj.parameters(),
            self.base.k_proj.parameters(),
        ):
            errors.append((erase.detach() - write.detach()).abs().max())
        for erase, write in zip(
            self.k_erase_conv1d.parameters(),
            self.base.k_conv1d.parameters(),
        ):
            errors.append((erase.detach() - write.detach()).abs().max())
        self.initial_tie_max_error = float(torch.stack(errors).max().item())

    def set_capture(self, enabled: bool) -> None:
        self._capture = bool(enabled)
        if enabled:
            self._captured.clear()

    @staticmethod
    def _unit(tensor: torch.Tensor) -> torch.Tensor:
        return F.normalize(tensor.float(), dim=-1, eps=1e-6).to(tensor.dtype)

    @staticmethod
    def transition_factors(
        k_erase: torch.Tensor,
        g: torch.Tensor,
        erase_gate: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # DPLR uses S' = D S + beta (alpha^T S) + k_write v^T.
        # alpha=D(b*k_erase), beta=-k_erase exactly reproduces GDN2 when
        # k_erase == k_write.
        alpha = torch.exp(g.float()) * erase_gate.float() * k_erase.float()
        beta = -k_erase.float()
        return alpha.to(k_erase.dtype), beta.to(k_erase.dtype)

    def _conv_states(
        self,
        last_state: dict[str, Any] | None,
    ) -> tuple[Any, Any, Any, Any]:
        if last_state is None:
            return None, None, None, None
        states = last_state["conv_state"]
        if len(states) == 3:
            q_state, write_state, value_state = states
            return q_state, write_state, None, value_state
        if len(states) == 4:
            return tuple(states)
        raise RuntimeError(f"Unexpected decoupled-key conv cache: {len(states)}")

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
            raise ValueError("P-GDN3-031 uses fixed unpadded MQAR batches")

        layer = self.base
        q_len = hidden_states.shape[1]
        last_state = get_layer_cache(layer, past_key_values)
        cu_seqlens = kwargs.get("cu_seqlens")
        q_state, write_state, erase_state, value_state = self._conv_states(last_state)

        q, q_state = layer.q_conv1d(
            x=layer.q_proj(hidden_states),
            cache=q_state,
            output_final_state=use_cache,
            cu_seqlens=cu_seqlens,
        )
        k_write, write_state = layer.k_conv1d(
            x=layer.k_proj(hidden_states),
            cache=write_state,
            output_final_state=use_cache,
            cu_seqlens=cu_seqlens,
        )
        k_erase, erase_state = self.k_erase_conv1d(
            x=self.k_erase_proj(hidden_states),
            cache=erase_state,
            output_final_state=use_cache,
            cu_seqlens=cu_seqlens,
        )
        value, value_state = layer.v_conv1d(
            x=layer.v_proj(hidden_states),
            cache=value_state,
            output_final_state=use_cache,
            cu_seqlens=cu_seqlens,
        )

        g = F.softplus(layer.f_proj(hidden_states).float() + layer.dt_bias)
        erase_gate = layer.b_proj(hidden_states).sigmoid()
        write_gate = layer.w_proj(hidden_states).sigmoid()

        q, k_write, k_erase, g, erase_gate = (
            rearrange(x, "... (h d) -> ... h d", d=layer.head_k_dim)
            for x in (q, k_write, k_erase, g, erase_gate)
        )
        value = rearrange(value, "... (h d) -> ... h d", d=layer.head_v_dim)
        write_gate = rearrange(
            write_gate,
            "... (h d) -> ... h d",
            d=layer.head_v_dim,
        )
        g = -layer.A_log.float().exp().unsqueeze(-1) * g
        q = self._unit(q)
        k_write = self._unit(k_write)
        k_erase = self._unit(k_erase)
        alpha, beta = self.transition_factors(k_erase, g, erase_gate)
        write = write_gate * value

        recurrent_state = (
            last_state["recurrent_state"] if last_state is not None else None
        )
        output, recurrent_state = chunk_dplr_delta_rule(
            q=q,
            k=k_write,
            v=write,
            a=alpha,
            b=beta,
            gk=g,
            scale=layer.head_k_dim**-0.5,
            initial_state=recurrent_state,
            output_final_state=use_cache,
            cu_seqlens=cu_seqlens,
            safe_gate=False,
            chunk_size=16,
        )
        update_layer_cache(
            layer,
            past_key_values,
            recurrent_state=recurrent_state,
            conv_state=(q_state, write_state, erase_state, value_state),
            offset=q_len,
        )
        if self._capture:
            self._captured = {
                "q": q.detach(),
                "k_write": k_write.detach(),
                "k_erase": k_erase.detach(),
                "alpha": alpha.detach(),
                "beta": beta.detach(),
                "g": g.detach(),
                "erase_gate": erase_gate.detach(),
                "write": write.detach(),
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


class ZoologyDecoupledKeyGDN2FutureSeedMixer(ZoologyGDN2FutureSeedMixer):
    """Native FutureSeed over direct function-preserving decoupled-key GDN2."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.layer = DirectDecoupledKeyGDN2(self.layer)


def make_tied_decoupled_key_model(model_config: Any) -> FutureSeedLanguageModel:
    """Build a candidate whose parent parameters exactly match native GDN2."""
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
    candidate.load_state_dict(candidate_state)
    for block in candidate.backbone.layers:
        mixer = block.sequence_mixer
        if not isinstance(mixer, ZoologyDecoupledKeyGDN2FutureSeedMixer):
            raise RuntimeError(f"Unexpected candidate mixer: {type(mixer)}")
        mixer.layer.reset_erase_from_write()
    return candidate


def parent_parameter_hash(model: nn.Module) -> str:
    tensors = []
    for name, parameter in model.named_parameters():
        if ".layer.k_erase_proj." in name or ".layer.k_erase_conv1d." in name:
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


def _rms(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.float().square().mean().sqrt()


def _relative_rms(left: torch.Tensor, right: torch.Tensor) -> float:
    return float((_rms(left - right) / _rms(right).clamp_min(1e-8)).item())


def _module_delta_rms(left: nn.Module, right: nn.Module) -> float:
    deltas = []
    for (left_name, left_parameter), (right_name, right_parameter) in zip(
        left.named_parameters(),
        right.named_parameters(),
    ):
        if left_name != right_name or left_parameter.shape != right_parameter.shape:
            raise RuntimeError("Tied key module structures diverged")
        deltas.append(
            (left_parameter.detach().float() - right_parameter.detach().float()).flatten()
        )
    return float(torch.cat(deltas).square().mean().sqrt().item())


def _layer_diagnostics(
    mixer: ZoologyDecoupledKeyGDN2FutureSeedMixer,
    *,
    max_transition_samples: int = 256,
) -> dict[str, float | int]:
    layer = mixer.layer
    captured = layer._captured
    required = {
        "q",
        "k_write",
        "k_erase",
        "alpha",
        "beta",
        "g",
        "erase_gate",
        "write",
        "terminal_state",
    }
    if set(captured) != required:
        raise RuntimeError(f"Incomplete decoupled-key capture: {set(captured)}")

    k_write = captured["k_write"].float()
    k_erase = captured["k_erase"].float()
    alpha = captured["alpha"].float().flatten(0, -2)
    beta = captured["beta"].float().flatten(0, -2)
    decay = captured["g"].float().exp().flatten(0, -2)
    sample_count = min(max_transition_samples, alpha.shape[0])
    indices = torch.linspace(
        0,
        alpha.shape[0] - 1,
        steps=sample_count,
        device=alpha.device,
    ).round().long()
    transition = (
        torch.diag_embed(decay.index_select(0, indices))
        + beta.index_select(0, indices).unsqueeze(-1)
        * alpha.index_select(0, indices).unsqueeze(-2)
    )
    singular_values = torch.linalg.svdvals(transition)
    terminal = captured["terminal_state"].float()
    terminal_board_rms = terminal.square().mean(dim=(-1, -2, -3)).sqrt()
    cosine = (k_write * k_erase).sum(dim=-1)
    return {
        "layer_idx": mixer.layer_idx,
        "initial_tie_max_error": layer.initial_tie_max_error,
        "key_cosine_mean": float(cosine.mean().item()),
        "key_separation_mean": float((1.0 - cosine).mean().item()),
        "key_relative_rms": _relative_rms(k_erase, k_write),
        "projection_delta_rms": _module_delta_rms(
            layer.k_erase_proj,
            layer.base.k_proj,
        ),
        "convolution_delta_rms": _module_delta_rms(
            layer.k_erase_conv1d,
            layer.base.k_conv1d,
        ),
        "erase_gate_mean": float(captured["erase_gate"].float().mean().item()),
        "erase_gate_std": float(
            captured["erase_gate"].float().std(unbiased=False).item()
        ),
        "write_rms": float(_rms(captured["write"]).item()),
        "transition_samples": int(sample_count),
        "transition_spectral_norm_max": float(singular_values[:, 0].max().item()),
        "terminal_state_rms": float(_rms(terminal).item()),
        "terminal_state_board_std": float(
            terminal_board_rms.std(unbiased=False).item()
        ),
    }


@torch.no_grad()
def decoupled_key_diagnostics(
    model: FutureSeedLanguageModel,
    inputs: torch.Tensor,
) -> dict[str, Any]:
    mixers = [
        block.sequence_mixer
        for block in model.backbone.layers
        if isinstance(
            block.sequence_mixer,
            ZoologyDecoupledKeyGDN2FutureSeedMixer,
        )
    ]
    if len(mixers) != len(model.backbone.layers):
        raise RuntimeError("Every layer must use direct decoupled-key GDN2")
    for mixer in mixers:
        mixer.layer.set_capture(True)
    try:
        model.eval()(inputs)
    finally:
        for mixer in mixers:
            mixer.layer.set_capture(False)
    rows = [_layer_diagnostics(mixer) for mixer in mixers]
    return {
        "active_layers": len(rows),
        "new_persistent_state_values": 0,
        "logical_scans_per_layer": 1,
        "per_layer": rows,
    }
