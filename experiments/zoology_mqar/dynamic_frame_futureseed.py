from __future__ import annotations

import hashlib
import math
from typing import Any

import torch
import torch.nn as nn
from einops import rearrange
from torch.nn import functional as F

from fla.layers.utils import get_layer_cache, update_layer_cache
from fla.ops.gdn2 import chunk_gdn2

from experiments.zoology_mqar.gdn2_futureseed import (
    FutureSeedLanguageModel,
    ZoologyGDN2FutureSeedMixer,
)


FRAME_RANK = 8
LOG_FRAME_RADIUS = 0.5 * math.log(2.0)
EXPECTED_STATE_VALUES = 4_096
EXPECTED_PARAMETER_DELTA = 4_096


def _rms(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.float().square().mean().sqrt()


class DynamicCanonicalAddressFrame(nn.Module):
    """Token-local diagonal address frames with a bounded condition number."""

    def __init__(self, hidden_size: int, num_heads: int, head_dim: int) -> None:
        super().__init__()
        if (hidden_size, num_heads, head_dim) != (128, 4, 32):
            raise ValueError("P-GDN3-044 is fixed at D128/H4/K32")
        self.hidden_size = int(hidden_size)
        self.num_heads = int(num_heads)
        self.head_dim = int(head_dim)
        self.in_proj = nn.Linear(hidden_size, FRAME_RANK, bias=False)
        self.out_proj = nn.Linear(
            FRAME_RANK,
            num_heads * head_dim,
            bias=False,
        )

    def reset_identity(self) -> None:
        nn.init.zeros_(self.out_proj.weight)

    def log_frame(self, hidden_states: torch.Tensor) -> torch.Tensor:
        raw = self.out_proj(F.silu(self.in_proj(hidden_states))).float()
        raw = rearrange(raw, "b t (h k) -> b t h k", h=self.num_heads)
        raw = raw - raw.mean(dim=-1, keepdim=True)
        return LOG_FRAME_RADIUS * torch.tanh(raw)


class DynamicFrameGDN2(nn.Module):
    """Official GDN2 in a token-varying, canonically transported address frame."""

    def __init__(self, base: nn.Module) -> None:
        super().__init__()
        if base.hidden_size != 128:
            raise ValueError("P-GDN3-044 requires D128")
        if base.num_heads != 4 or base.num_v_heads != 4:
            raise ValueError("P-GDN3-044 requires H4")
        if base.head_k_dim != 32 or base.head_v_dim != 32:
            raise ValueError("P-GDN3-044 requires K32/V32")
        if base.mode != "chunk" or not base.use_short_conv:
            raise ValueError("P-GDN3-044 requires chunk GDN2 with ShortConv")
        self.base = base
        self.frame = DynamicCanonicalAddressFrame(
            hidden_size=base.hidden_size,
            num_heads=base.num_heads,
            head_dim=base.head_k_dim,
        )
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
            raise RuntimeError("P-GDN3-044 requires an explicit layer index")
        return int(self.base.layer_idx)

    def reset_identity(self) -> None:
        self.frame.reset_identity()

    def set_capture(self, enabled: bool) -> None:
        self._capture = bool(enabled)
        if enabled:
            self._captured.clear()

    def frame_terms(
        self,
        hidden_states: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        log_frame = self.frame.log_frame(hidden_states)
        first = log_frame[:, :1]
        frame_delta = torch.cat(
            (first, log_frame[:, 1:] - log_frame[:, :-1]),
            dim=1,
        )
        factor = log_frame.exp()
        return log_frame, frame_delta, factor

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
            raise ValueError("P-GDN3-044 uses fixed unpadded MQAR batches")

        layer = self.base
        q_len = hidden_states.shape[1]
        last_state = get_layer_cache(layer, past_key_values)
        cu_seqlens = kwargs.get("cu_seqlens")
        if cu_seqlens is not None:
            raise ValueError("P-GDN3-044 does not use packed sequences")
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
        value, conv_state_v = layer.v_conv1d(
            x=layer.v_proj(hidden_states),
            cache=conv_state_v,
            output_final_state=use_cache,
        )

        g = F.softplus(layer.f_proj(hidden_states).float() + layer.dt_bias)
        erase_gate = layer.b_proj(hidden_states).sigmoid()
        write_gate = layer.w_proj(hidden_states).sigmoid()
        q, k, g, erase_gate = (
            rearrange(tensor, "... (h d) -> ... h d", d=layer.head_k_dim)
            for tensor in (q, k, g, erase_gate)
        )
        value = rearrange(value, "... (h d) -> ... h d", d=layer.head_v_dim)
        write_gate = rearrange(
            write_gate,
            "... (h d) -> ... h d",
            d=layer.head_v_dim,
        )
        native_g = -layer.A_log.float().exp().unsqueeze(-1) * g
        log_frame, frame_delta, factor = self.frame_terms(hidden_states)
        factor_in_projection_dtype = factor.to(dtype=q.dtype)
        transformed_q = q + q * (factor_in_projection_dtype - 1)
        transformed_k = k + k * (factor_in_projection_dtype - 1)
        effective_g = native_g + frame_delta
        if layer.allow_neg_eigval:
            erase_gate = erase_gate * 2.0

        recurrent_state = (
            last_state["recurrent_state"] if last_state is not None else None
        )
        output, recurrent_state = chunk_gdn2(
            q=transformed_q,
            k=transformed_k,
            v=value,
            g=effective_g,
            b=erase_gate,
            w=write_gate,
            initial_state=recurrent_state,
            output_final_state=use_cache,
            use_qk_l2norm_in_kernel=True,
        )
        current_frame_state = recurrent_state
        if recurrent_state is not None:
            inverse_terminal_factor = factor[:, -1].float().reciprocal()
            recurrent_state = recurrent_state + recurrent_state * (
                inverse_terminal_factor.unsqueeze(-1) - 1
            )
        update_layer_cache(
            layer,
            past_key_values,
            recurrent_state=recurrent_state,
            conv_state=(conv_state_q, conv_state_k, conv_state_v),
            offset=q_len,
        )
        if self._capture:
            if recurrent_state is None or current_frame_state is None:
                raise RuntimeError("P-GDN3-044 capture requires terminal state")
            self._captured = {
                "q": q.detach(),
                "k": k.detach(),
                "transformed_q": transformed_q.detach(),
                "transformed_k": transformed_k.detach(),
                "native_g": native_g.detach(),
                "effective_g": effective_g.detach(),
                "log_frame": log_frame.detach(),
                "frame_delta": frame_delta.detach(),
                "factor": factor.detach(),
                "current_frame_state": current_frame_state.detach(),
                "canonical_state": recurrent_state.detach(),
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


class ZoologyDynamicFrameFutureSeedMixer(ZoologyGDN2FutureSeedMixer):
    """Native FutureSeed over a canonical dynamic-frame GDN2 transition."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.layer = DynamicFrameGDN2(self.layer)


class DynamicFrameFutureSeedLanguageModel(FutureSeedLanguageModel):
    def __init__(self, config: Any) -> None:
        super().__init__(config)
        for block in self.backbone.layers:
            mixer = block.sequence_mixer
            if not isinstance(mixer, ZoologyDynamicFrameFutureSeedMixer):
                raise TypeError("P-GDN3-044 requires dynamic-frame mixers")
            mixer.layer.reset_identity()


def load_matched_parent_state(
    model: nn.Module,
    control_state: dict[str, torch.Tensor],
) -> None:
    loaded: dict[str, torch.Tensor] = {}
    for name, tensor in model.state_dict().items():
        if ".sequence_mixer.layer.frame." in name:
            loaded[name] = tensor
            if name.endswith("frame.out_proj.weight"):
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
        if ".sequence_mixer.layer.frame." in name:
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
def dynamic_frame_diagnostics(
    model: nn.Module,
    inputs: torch.Tensor,
) -> dict[str, Any]:
    mixers = [block.sequence_mixer for block in model.backbone.layers]
    if len(mixers) != 2 or not all(
        isinstance(mixer, ZoologyDynamicFrameFutureSeedMixer)
        for mixer in mixers
    ):
        raise RuntimeError("Expected two dynamic-frame GDN2 mixers")
    for mixer in mixers:
        mixer.layer.set_capture(True)
    try:
        model.eval()(inputs)
    finally:
        for mixer in mixers:
            mixer.layer.set_capture(False)

    rows = []
    for mixer in mixers:
        captured = mixer.layer._captured
        required = {
            "q",
            "k",
            "transformed_q",
            "transformed_k",
            "native_g",
            "effective_g",
            "log_frame",
            "frame_delta",
            "factor",
            "current_frame_state",
            "canonical_state",
        }
        if set(captured) != required:
            raise RuntimeError(f"Incomplete dynamic-frame capture: {set(captured)}")
        log_frame = captured["log_frame"].float()
        frame_delta = captured["frame_delta"].float()
        factor = captured["factor"].float()
        cumulative_error = (frame_delta.cumsum(dim=1) - log_frame).abs().max()
        q = captured["q"].float()
        k = captured["k"].float()
        tq = captured["transformed_q"].float()
        tk = captured["transformed_k"].float()
        native_g = captured["native_g"].float()
        effective_g = captured["effective_g"].float()
        canonical = captured["canonical_state"].float()
        current = captured["current_frame_state"].float()
        expected_current = canonical * factor[:, -1].unsqueeze(-1)
        canonicalization_error = (expected_current - current).abs().max()
        terminal_board = canonical.square().mean(dim=(-1, -2, -3)).sqrt()
        frame_board = log_frame.square().mean(dim=(1, 2, 3)).sqrt()
        frame_token = log_frame.square().mean(dim=(0, 2, 3)).sqrt()
        factors_per_head = factor.amax(dim=(0, 1, 3)) / factor.amin(
            dim=(0, 1, 3)
        )
        rows.append(
            {
                "layer_idx": mixer.layer_idx,
                "frame_log_rms": float(_rms(log_frame).item()),
                "frame_log_abs_max": float(log_frame.abs().max().item()),
                "frame_board_std": float(
                    frame_board.std(unbiased=False).item()
                ),
                "frame_token_std": float(
                    frame_token.std(unbiased=False).item()
                ),
                "frame_delta_rms": float(_rms(frame_delta).item()),
                "frame_cumulative_max_error": float(cumulative_error.item()),
                "factor_min": float(factor.min().item()),
                "factor_max": float(factor.max().item()),
                "factor_condition_max": float(factors_per_head.max().item()),
                "q_change_relative_rms": float(
                    (_rms(tq - q) / _rms(q).clamp_min(1e-8)).item()
                ),
                "k_change_relative_rms": float(
                    (_rms(tk - k) / _rms(k).clamp_min(1e-8)).item()
                ),
                "effective_g_change_rms": float(
                    _rms(effective_g - native_g).item()
                ),
                "effective_g_positive_fraction": float(
                    (effective_g > 0).float().mean().item()
                ),
                "effective_g_cumsum_max": float(
                    effective_g.cumsum(dim=1).max().item()
                ),
                "canonicalization_max_error": float(
                    canonicalization_error.item()
                ),
                "canonical_state_rms": float(_rms(canonical).item()),
                "canonical_state_board_std": float(
                    terminal_board.std(unbiased=False).item()
                ),
                "controller_in_rms": float(
                    _rms(mixer.layer.frame.in_proj.weight).item()
                ),
                "controller_out_rms": float(
                    _rms(mixer.layer.frame.out_proj.weight).item()
                ),
            }
        )
    return {
        "active_layers": len(rows),
        "logical_scans_per_layer": 1,
        "frame_rank": FRAME_RANK,
        "log_frame_radius": LOG_FRAME_RADIUS,
        "new_parameters": sum(
            parameter.numel()
            for mixer in mixers
            for parameter in mixer.layer.frame.parameters()
        ),
        "new_persistent_state_values": 0,
        "per_layer": rows,
    }
