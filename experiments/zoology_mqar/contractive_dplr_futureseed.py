from __future__ import annotations

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


class ContractiveSeparateEraseWriteDPLR(nn.Module):
    """GDN projections with one symmetric contractive DPLR transition."""

    def __init__(self, base: nn.Module) -> None:
        super().__init__()
        if base.num_heads != base.num_v_heads:
            raise ValueError("P-GDN3-030 requires matched key and value heads")
        if base.head_k_dim != base.head_v_dim:
            raise ValueError("P-GDN3-030 fixes K32/V32 state geometry")
        if base.mode != "chunk":
            raise ValueError("P-GDN3-030 trains only with the official chunk DPLR kernel")
        self.base = base
        self.beta_proj = nn.Linear(base.hidden_size, base.num_heads, bias=False)
        self._capture = False
        self._captured: dict[str, torch.Tensor] = {}

    @property
    def num_heads(self) -> int:
        return int(self.base.num_heads)

    @property
    def layer_idx(self) -> int:
        if self.base.layer_idx is None:
            raise RuntimeError("P-GDN3-030 requires an explicit layer index")
        return int(self.base.layer_idx)

    @property
    def num_v_heads(self) -> int:
        return int(self.base.num_v_heads)

    @property
    def head_k_dim(self) -> int:
        return int(self.base.head_k_dim)

    @property
    def head_v_dim(self) -> int:
        return int(self.base.head_v_dim)

    def set_capture(self, enabled: bool) -> None:
        self._capture = bool(enabled)
        if enabled:
            self._captured.clear()

    @staticmethod
    def _unit(tensor: torch.Tensor) -> torch.Tensor:
        return F.normalize(tensor.float(), dim=-1, eps=1e-6).to(tensor.dtype)

    @staticmethod
    def transition_factors(
        r: torch.Tensor,
        g: torch.Tensor,
        beta: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        sqrt_decay = torch.exp(0.5 * g.float())
        a = sqrt_decay * r.float()
        b = -beta.float().unsqueeze(-1) * a
        return a.to(r.dtype), b.to(r.dtype)

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
            raise ValueError("P-GDN3-030 uses fixed unpadded MQAR batches")

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
        p, conv_state_k = layer.k_conv1d(
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
        r = layer.b_proj(hidden_states)
        write_gate = layer.w_proj(hidden_states).sigmoid()
        beta = self.beta_proj(hidden_states).sigmoid()

        q, p, g, r = (
            rearrange(x, "... (h d) -> ... h d", d=layer.head_k_dim)
            for x in (q, p, g, r)
        )
        value = rearrange(value, "... (h d) -> ... h d", d=layer.head_v_dim)
        write_gate = rearrange(
            write_gate,
            "... (h d) -> ... h d",
            d=layer.head_v_dim,
        )
        g = -layer.A_log.float().exp().unsqueeze(-1) * g
        q = self._unit(q)
        p = self._unit(p)
        r = self._unit(r)
        a, b = self.transition_factors(r, g, beta)
        write = beta.to(value.dtype).unsqueeze(-1) * write_gate * value

        recurrent_state = (
            last_state["recurrent_state"] if last_state is not None else None
        )
        output, recurrent_state = chunk_dplr_delta_rule(
            q=q,
            k=p,
            v=write,
            a=a,
            b=b,
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
            self._captured = {
                "p": p.detach(),
                "r": r.detach(),
                "a": a.detach(),
                "b": b.detach(),
                "g": g.detach(),
                "beta": beta.detach(),
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


class ZoologyContractiveDPLRFutureSeedMixer(ZoologyGDN2FutureSeedMixer):
    """Native FutureSeed over a separate-erase/write official DPLR core."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.layer = ContractiveSeparateEraseWriteDPLR(self.layer)


def _rms(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.float().square().mean().sqrt()


def _layer_diagnostics(
    mixer: ZoologyContractiveDPLRFutureSeedMixer,
    *,
    max_transition_samples: int = 256,
) -> dict[str, float | int]:
    captured = mixer.layer._captured
    required = {"p", "r", "a", "b", "g", "beta", "write", "terminal_state"}
    if set(captured) != required:
        raise RuntimeError(f"Incomplete contractive DPLR capture: {set(captured)}")

    p = captured["p"].float()
    r = captured["r"].float()
    a = captured["a"].float()
    b = captured["b"].float()
    g = captured["g"].float()
    beta = captured["beta"].float()
    write = captured["write"].float()
    terminal = captured["terminal_state"].float()

    decay = g.exp().flatten(0, -2)
    a_flat = a.flatten(0, -2)
    b_flat = b.flatten(0, -2)
    sample_count = min(max_transition_samples, decay.shape[0])
    indices = torch.linspace(
        0,
        decay.shape[0] - 1,
        steps=sample_count,
        device=decay.device,
    ).round().long()
    decay = decay.index_select(0, indices)
    a_flat = a_flat.index_select(0, indices)
    b_flat = b_flat.index_select(0, indices)
    transition = torch.diag_embed(decay) + b_flat.unsqueeze(-1) * a_flat.unsqueeze(-2)
    symmetric = 0.5 * (transition + transition.transpose(-1, -2))
    eigenvalues = torch.linalg.eigvalsh(symmetric)
    singular_values = torch.linalg.svdvals(transition)
    terminal_board_rms = terminal.square().mean(dim=(-1, -2, -3)).sqrt()
    erase_strength = a.norm(dim=-1) * b.norm(dim=-1)

    return {
        "layer_idx": mixer.layer_idx,
        "erase_write_separation": float(
            (1.0 - (p * r).sum(dim=-1).abs()).mean().item()
        ),
        "beta_mean": float(beta.mean().item()),
        "beta_std": float(beta.std(unbiased=False).item()),
        "erase_strength_mean": float(erase_strength.mean().item()),
        "write_rms": float(_rms(write).item()),
        "transition_samples": int(sample_count),
        "transition_spectral_norm_max": float(singular_values[:, 0].max().item()),
        "transition_eigenvalue_min": float(eigenvalues[:, 0].min().item()),
        "transition_symmetry_error_max": float(
            (transition - transition.transpose(-1, -2)).abs().max().item()
        ),
        "terminal_state_rms": float(_rms(terminal).item()),
        "terminal_state_board_std": float(
            terminal_board_rms.std(unbiased=False).item()
        ),
    }


@torch.no_grad()
def contractive_dplr_diagnostics(
    model: FutureSeedLanguageModel,
    inputs: torch.Tensor,
) -> dict[str, Any]:
    mixers = [
        layer.sequence_mixer
        for layer in model.backbone.layers
        if isinstance(layer.sequence_mixer, ZoologyContractiveDPLRFutureSeedMixer)
    ]
    if len(mixers) != len(model.backbone.layers):
        raise RuntimeError("Every layer must use the contractive DPLR mixer")
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
        "per_layer": rows,
    }
