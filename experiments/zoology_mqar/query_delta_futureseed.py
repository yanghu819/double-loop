from __future__ import annotations

import hashlib
from typing import Any

import torch
import torch.nn.functional as F
from einops import rearrange
from torch import nn

from fla.layers.utils import get_layer_cache, update_layer_cache
from fla.ops.generalized_delta_rule.dplr import chunk_dplr_delta_rule

from experiments.zoology_mqar.gdn2_futureseed import (
    ZoologyGDN2FutureSeedMixer,
)


LAMBDA_CAP = 0.5
EXPECTED_STATE_VALUES = 4_096


def _rms(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.float().square().mean().sqrt()


class QueryAwareLiveDeltaGDN2(nn.Module):
    """Native GDN2 with query feedback inside the live token transition."""

    def __init__(self, base: nn.Module) -> None:
        super().__init__()
        if base.hidden_size != 128:
            raise ValueError("P-GDN3-041 requires D128")
        if base.num_heads != 4 or base.num_v_heads != 4:
            raise ValueError("P-GDN3-041 requires H4")
        if base.head_k_dim != 32 or base.head_v_dim != 32:
            raise ValueError("P-GDN3-041 requires K32/V32")
        if base.mode != "chunk" or not base.use_short_conv:
            raise ValueError("P-GDN3-041 requires chunk GDN2 with short convolution")
        self.base = base
        self.lambda_proj = nn.Linear(
            base.hidden_size,
            base.num_heads,
            bias=True,
        )
        nn.init.zeros_(self.lambda_proj.weight)
        nn.init.zeros_(self.lambda_proj.bias)
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
            raise RuntimeError("P-GDN3-041 requires an explicit layer index")
        return int(self.base.layer_idx)

    def set_capture(self, enabled: bool) -> None:
        self._capture = bool(enabled)
        if enabled:
            self._captured.clear()

    @staticmethod
    def _unit(tensor: torch.Tensor) -> torch.Tensor:
        return F.normalize(tensor.float(), dim=-1, eps=1e-6).to(tensor.dtype)

    @staticmethod
    def transition_factors(
        q: torch.Tensor,
        k: torch.Tensor,
        g: torch.Tensor,
        erase_gate: torch.Tensor,
        query_feedback: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mixed = k.float() + query_feedback.float().unsqueeze(-1) * q.float()
        alpha = torch.exp(g.float()) * erase_gate.float() * mixed
        beta = -k.float()
        return mixed.to(k.dtype), alpha.to(k.dtype), beta.to(k.dtype)

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
            raise ValueError("P-GDN3-041 uses fixed unpadded MQAR batches")

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
        k, conv_state_k = layer.k_conv1d(
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
        query_feedback = LAMBDA_CAP * torch.tanh(
            self.lambda_proj(hidden_states).float()
        )

        q, k, g, erase_gate = (
            rearrange(tensor, "... (h d) -> ... h d", d=layer.head_k_dim)
            for tensor in (q, k, g, erase_gate)
        )
        value = rearrange(
            value,
            "... (h d) -> ... h d",
            d=layer.head_v_dim,
        )
        write_gate = rearrange(
            write_gate,
            "... (h d) -> ... h d",
            d=layer.head_v_dim,
        )
        g = -layer.A_log.float().exp().unsqueeze(-1) * g
        q = self._unit(q)
        k = self._unit(k)
        mixed, alpha, beta = self.transition_factors(
            q,
            k,
            g,
            erase_gate,
            query_feedback,
        )
        write = write_gate * value

        recurrent_state = (
            last_state["recurrent_state"] if last_state is not None else None
        )
        output, recurrent_state = chunk_dplr_delta_rule(
            q=q,
            k=k,
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
            conv_state=(conv_state_q, conv_state_k, conv_state_v),
            offset=q_len,
        )
        if self._capture:
            if recurrent_state is None:
                raise RuntimeError("P-GDN3-041 capture requires a terminal state")
            self._captured = {
                "q": q.detach(),
                "k": k.detach(),
                "mixed": mixed.detach(),
                "query_feedback": query_feedback.detach(),
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


class ZoologyQueryDeltaFutureSeedMixer(ZoologyGDN2FutureSeedMixer):
    """Native FutureSeed over one query-aware live GDN2 transition."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.layer = QueryAwareLiveDeltaGDN2(self.layer)


def load_matched_parent_state(
    model: nn.Module,
    control_state: dict[str, torch.Tensor],
) -> None:
    loaded: dict[str, torch.Tensor] = {}
    for name, tensor in model.state_dict().items():
        if ".sequence_mixer.layer.lambda_proj." in name:
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
        if ".sequence_mixer.layer.lambda_proj." in name:
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


def _layer_diagnostics(
    mixer: ZoologyQueryDeltaFutureSeedMixer,
    *,
    max_transition_samples: int = 256,
) -> dict[str, Any]:
    captured = mixer.layer._captured
    required = {
        "q",
        "k",
        "mixed",
        "query_feedback",
        "alpha",
        "beta",
        "g",
        "erase_gate",
        "write",
        "terminal_state",
    }
    if set(captured) != required:
        raise RuntimeError(f"Incomplete query-delta capture: {set(captured)}")

    q = captured["q"].float()
    k = captured["k"].float()
    mixed = captured["mixed"].float()
    feedback = captured["query_feedback"].float()
    alpha = captured["alpha"].float()
    beta = captured["beta"].float()
    decay = captured["g"].float().exp()
    native_alpha = decay * captured["erase_gate"].float() * k
    query_alpha = alpha - native_alpha

    alpha_flat = alpha.flatten(0, -2)
    beta_flat = beta.flatten(0, -2)
    decay_flat = decay.flatten(0, -2)
    sample_count = min(max_transition_samples, alpha_flat.shape[0])
    indices = torch.linspace(
        0,
        alpha_flat.shape[0] - 1,
        steps=sample_count,
        device=alpha.device,
    ).round().long()
    transition = (
        torch.diag_embed(decay_flat.index_select(0, indices))
        + beta_flat.index_select(0, indices).unsqueeze(-1)
        * alpha_flat.index_select(0, indices).unsqueeze(-2)
    )
    singular_values = torch.linalg.svdvals(transition)
    terminal = captured["terminal_state"].float()
    terminal_board_rms = terminal.square().mean(dim=(-1, -2, -3)).sqrt()
    feedback_board_rms = feedback.square().mean(dim=(1, 2)).sqrt()
    feedback_token_rms = feedback.square().mean(dim=(0, 2)).sqrt()
    alignment = (k * mixed).sum(dim=-1)
    per_head_rms = feedback.square().mean(dim=(0, 1)).sqrt()
    parameter_values = torch.cat(
        [parameter.detach().float().flatten() for parameter in mixer.layer.lambda_proj.parameters()]
    )
    return {
        "layer_idx": mixer.layer_idx,
        "lambda_rms": float(_rms(feedback).item()),
        "lambda_abs_max": float(feedback.abs().max().item()),
        "lambda_board_std": float(feedback_board_rms.std(unbiased=False).item()),
        "lambda_token_std": float(feedback_token_rms.std(unbiased=False).item()),
        "lambda_per_head_rms": [float(value.item()) for value in per_head_rms],
        "active_heads": int((per_head_rms >= 1e-3).sum().item()),
        "feedback_parameter_rms": float(_rms(parameter_values).item()),
        "query_feedback_relative_rms": float(
            (_rms(feedback.unsqueeze(-1) * q) / _rms(k).clamp_min(1e-8)).item()
        ),
        "query_alpha_relative_rms": float(
            (_rms(query_alpha) / _rms(native_alpha).clamp_min(1e-8)).item()
        ),
        "mixed_alignment_min": float(alignment.min().item()),
        "mixed_alignment_max": float(alignment.max().item()),
        "mixed_alignment_mean": float(alignment.mean().item()),
        "transition_samples": int(sample_count),
        "transition_spectral_norm_max": float(singular_values[:, 0].max().item()),
        "write_rms": float(_rms(captured["write"]).item()),
        "terminal_state_rms": float(_rms(terminal).item()),
        "terminal_state_board_std": float(
            terminal_board_rms.std(unbiased=False).item()
        ),
    }


@torch.no_grad()
def query_delta_diagnostics(
    model: nn.Module,
    inputs: torch.Tensor,
) -> dict[str, Any]:
    mixers = [block.sequence_mixer for block in model.backbone.layers]
    if len(mixers) != 2 or not all(
        isinstance(mixer, ZoologyQueryDeltaFutureSeedMixer) for mixer in mixers
    ):
        raise RuntimeError("Expected exactly two query-delta mixers")
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
        "lambda_cap": LAMBDA_CAP,
        "new_parameters": sum(
            parameter.numel()
            for mixer in mixers
            for parameter in mixer.layer.lambda_proj.parameters()
        ),
        "new_persistent_state_values": 0,
        "logical_scans_per_layer": 1,
        "per_layer": rows,
    }
