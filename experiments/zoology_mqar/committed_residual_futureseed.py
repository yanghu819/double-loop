from __future__ import annotations

import copy
import hashlib
from typing import Any

import torch
import torch.nn.functional as F
from einops import rearrange
from torch import nn

from fla.layers.utils import get_layer_cache, update_layer_cache
from fla.ops.generalized_delta_rule.dplr import chunk_dplr_delta_rule

from experiments.zoology_mqar.gdn2_futureseed import (
    FutureSeedLanguageModel,
    ZoologyGDN2FutureSeedMixer,
)


NATIVE_MIXER_NAME = (
    "experiments.zoology_mqar.gdn2_futureseed."
    "ZoologyGDN2FutureSeedMixer"
)


def _rms(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.float().square().mean().sqrt()


def _relative_rms(value: torch.Tensor, reference: torch.Tensor) -> float:
    return float((_rms(value - reference) / _rms(reference).clamp_min(1e-8)).item())


class CommittedResidualGDN2(nn.Module):
    """GDN2 projections with one coherent committed-residual transition."""

    def __init__(self, base: nn.Module) -> None:
        super().__init__()
        if base.num_heads != base.num_v_heads:
            raise ValueError("P-GDN3-037 requires matched QK/V heads")
        if base.head_k_dim != base.head_v_dim:
            raise ValueError("P-GDN3-037 fixes K32/V32 geometry")
        if base.mode != "chunk" or not base.use_short_conv:
            raise ValueError("P-GDN3-037 requires chunk mode and short convolution")
        self.base = base
        self._capture = False
        self._captured: dict[str, torch.Tensor | None] = {}

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
            raise RuntimeError("P-GDN3-037 requires an explicit layer index")
        return int(self.base.layer_idx)

    def set_capture(self, enabled: bool) -> None:
        self._capture = bool(enabled)
        if enabled:
            self._captured.clear()

    @staticmethod
    def transition_inputs(
        k: torch.Tensor,
        g: torch.Tensor,
        erase_gate: torch.Tensor,
        write_target: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        beta = erase_gate.float().mean(dim=-1, keepdim=True)
        alpha = torch.exp(g.float()) * k.float()
        transition_beta = -beta * k.float()
        committed_target = beta * write_target.float()
        return (
            beta.to(k.dtype),
            alpha.to(k.dtype),
            transition_beta.to(k.dtype),
            committed_target.to(write_target.dtype),
        )

    @staticmethod
    def _unit(tensor: torch.Tensor) -> torch.Tensor:
        return F.normalize(tensor.float(), dim=-1, eps=1e-6).to(tensor.dtype)

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
            raise ValueError("P-GDN3-037 uses fixed unpadded MQAR batches")

        layer = self.base
        q_len = hidden_states.shape[1]
        last_state = get_layer_cache(layer, past_key_values)
        cu_seqlens = kwargs.get("cu_seqlens")
        q_state, k_state, v_state = None, None, None
        if last_state is not None:
            q_state, k_state, v_state = last_state["conv_state"]

        q, q_state = layer.q_conv1d(
            x=layer.q_proj(hidden_states),
            cache=q_state,
            output_final_state=use_cache,
            cu_seqlens=cu_seqlens,
        )
        k, k_state = layer.k_conv1d(
            x=layer.k_proj(hidden_states),
            cache=k_state,
            output_final_state=use_cache,
            cu_seqlens=cu_seqlens,
        )
        value, v_state = layer.v_conv1d(
            x=layer.v_proj(hidden_states),
            cache=v_state,
            output_final_state=use_cache,
            cu_seqlens=cu_seqlens,
        )

        g = F.softplus(layer.f_proj(hidden_states).float() + layer.dt_bias)
        erase_gate = layer.b_proj(hidden_states).sigmoid()
        write_gate = layer.w_proj(hidden_states).sigmoid()

        q, k, g, erase_gate = (
            rearrange(x, "... (h d) -> ... h d", d=layer.head_k_dim)
            for x in (q, k, g, erase_gate)
        )
        value = rearrange(value, "... (h d) -> ... h d", d=layer.head_v_dim)
        write_gate = rearrange(
            write_gate,
            "... (h d) -> ... h d",
            d=layer.head_v_dim,
        )
        g = -layer.A_log.float().exp().unsqueeze(-1) * g
        q = self._unit(q)
        k = self._unit(k)
        write_target = write_gate * value
        beta, alpha, transition_beta, committed_target = self.transition_inputs(
            k,
            g,
            erase_gate,
            write_target,
        )

        initial_state = (
            last_state["recurrent_state"] if last_state is not None else None
        )
        output, recurrent_state = chunk_dplr_delta_rule(
            q=q,
            k=k,
            v=committed_target,
            a=alpha,
            b=transition_beta,
            gk=g,
            scale=layer.head_k_dim**-0.5,
            initial_state=initial_state,
            output_final_state=use_cache,
            cu_seqlens=cu_seqlens,
            safe_gate=False,
            chunk_size=16,
        )
        update_layer_cache(
            layer,
            past_key_values,
            recurrent_state=recurrent_state,
            conv_state=(q_state, k_state, v_state),
            offset=q_len,
        )
        if self._capture:
            self._captured = {
                "q": q.detach(),
                "k": k.detach(),
                "g": g.detach(),
                "erase_gate": erase_gate.detach(),
                "write_gate": write_gate.detach(),
                "write_target": write_target.detach(),
                "beta": beta.detach(),
                "alpha": alpha.detach(),
                "transition_beta": transition_beta.detach(),
                "committed_target": committed_target.detach(),
                "initial_state": (
                    None if initial_state is None else initial_state.detach()
                ),
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


class ZoologyCommittedResidualGDN2FutureSeedMixer(ZoologyGDN2FutureSeedMixer):
    """Native FutureSeed over committed-residual target GDN2."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.layer = CommittedResidualGDN2(self.layer)


def make_matched_committed_residual_model(
    model_config: Any,
) -> FutureSeedLanguageModel:
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
    return candidate


def parent_parameter_hash(model: nn.Module) -> str:
    tensors = []
    for name, parameter in model.named_parameters():
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


def _explicit_diagnostics(captured: dict[str, torch.Tensor | None]) -> dict[str, float]:
    k = captured["k"]
    g = captured["g"]
    beta = captured["beta"]
    target = captured["write_target"]
    terminal = captured["terminal_state"]
    if not all(isinstance(x, torch.Tensor) for x in (k, g, beta, target, terminal)):
        raise RuntimeError("Incomplete committed-residual capture")
    k = k.float()
    g = g.float()
    beta = beta.float()
    target = target.float()
    initial = captured["initial_state"]
    if initial is None:
        state = torch.zeros(
            k.shape[0],
            k.shape[2],
            k.shape[3],
            target.shape[3],
            device=k.device,
            dtype=torch.float32,
        )
    else:
        if not isinstance(initial, torch.Tensor):
            raise RuntimeError("Initial state capture changed type")
        state = initial.float().clone()
    residual_sq = torch.zeros((), device=k.device)
    target_sq = torch.zeros((), device=k.device)
    residual_board_sq = torch.zeros(k.shape[0], device=k.device)
    for token in range(k.shape[1]):
        decay = torch.exp(g[:, token]).unsqueeze(-1)
        state = state * decay
        read = torch.einsum("bhk,bhkv->bhv", k[:, token], state)
        residual = target[:, token] - read
        state = state + (
            beta[:, token].unsqueeze(-1)
            * k[:, token].unsqueeze(-1)
            * residual.unsqueeze(-2)
        )
        residual_sq = residual_sq + residual.square().sum()
        target_sq = target_sq + target[:, token].square().sum()
        residual_board_sq = residual_board_sq + residual.square().sum(dim=(1, 2))
    residual_relative_rms = float(
        (residual_sq / target_sq.clamp_min(1e-12)).sqrt().item()
    )
    return {
        "committed_residual_relative_rms": residual_relative_rms,
        "committed_residual_board_std": float(
            residual_board_sq.sqrt().std(unbiased=False).item()
        ),
        "explicit_terminal_relative_rms": _relative_rms(
            state,
            terminal.float(),
        ),
    }


def _layer_diagnostics(
    mixer: ZoologyCommittedResidualGDN2FutureSeedMixer,
) -> dict[str, float | int]:
    captured = mixer.layer._captured
    required = {
        "q",
        "k",
        "g",
        "erase_gate",
        "write_gate",
        "write_target",
        "beta",
        "alpha",
        "transition_beta",
        "committed_target",
        "initial_state",
        "terminal_state",
    }
    if set(captured) != required:
        raise RuntimeError(f"Incomplete committed-residual capture: {set(captured)}")
    k = captured["k"]
    g = captured["g"]
    beta = captured["beta"]
    alpha = captured["alpha"]
    transition_beta = captured["transition_beta"]
    terminal = captured["terminal_state"]
    if not all(
        isinstance(x, torch.Tensor)
        for x in (k, g, beta, alpha, transition_beta, terminal)
    ):
        raise RuntimeError("Committed-residual tensor capture changed type")
    flat_alpha = alpha.float().flatten(0, -2)
    flat_transition_beta = transition_beta.float().flatten(0, -2)
    flat_decay = torch.exp(g.float()).flatten(0, -2)
    sample_count = min(256, flat_alpha.shape[0])
    indices = torch.linspace(
        0,
        flat_alpha.shape[0] - 1,
        steps=sample_count,
        device=flat_alpha.device,
    ).round().long()
    transition = (
        torch.diag_embed(flat_decay.index_select(0, indices))
        + flat_transition_beta.index_select(0, indices).unsqueeze(-1)
        * flat_alpha.index_select(0, indices).unsqueeze(-2)
    )
    singular_values = torch.linalg.svdvals(transition)
    terminal_board_rms = terminal.float().square().mean(
        dim=(-1, -2, -3)
    ).sqrt()
    beta_board_rms = beta.float().square().mean(dim=(1, 2, 3)).sqrt()
    beta_token_rms = beta.float().square().mean(dim=(0, 2, 3)).sqrt()
    row = {
        "layer_idx": mixer.layer_idx,
        "beta_mean": float(beta.float().mean().item()),
        "beta_std": float(beta.float().std(unbiased=False).item()),
        "beta_board_std": float(beta_board_rms.std(unbiased=False).item()),
        "beta_token_std": float(beta_token_rms.std(unbiased=False).item()),
        "write_gate_mean": float(
            captured["write_gate"].float().mean().item()  # type: ignore[union-attr]
        ),
        "write_gate_std": float(
            captured["write_gate"].float().std(unbiased=False).item()  # type: ignore[union-attr]
        ),
        "transition_spectral_norm_max": float(singular_values[:, 0].max().item()),
        "terminal_state_rms": float(_rms(terminal).item()),
        "terminal_state_board_std": float(
            terminal_board_rms.std(unbiased=False).item()
        ),
    }
    row.update(_explicit_diagnostics(captured))
    return row


@torch.no_grad()
def committed_residual_diagnostics(
    model: FutureSeedLanguageModel,
    inputs: torch.Tensor,
) -> dict[str, Any]:
    mixers = [
        block.sequence_mixer
        for block in model.backbone.layers
        if isinstance(
            block.sequence_mixer,
            ZoologyCommittedResidualGDN2FutureSeedMixer,
        )
    ]
    if len(mixers) != len(model.backbone.layers):
        raise RuntimeError("Every layer must use committed-residual GDN2")
    for mixer in mixers:
        mixer.layer.set_capture(True)
    try:
        model.eval()(inputs)
    finally:
        for mixer in mixers:
            mixer.layer.set_capture(False)
    return {
        "active_layers": len(mixers),
        "new_parameters": 0,
        "new_persistent_state_values": 0,
        "logical_scans_per_layer": 1,
        "per_layer": [_layer_diagnostics(mixer) for mixer in mixers],
    }
