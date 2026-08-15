from __future__ import annotations

import hashlib
from typing import Any

import torch
import torch.nn as nn
from einops import rearrange
from torch.nn import functional as F

from fla.layers.utils import get_layer_cache, update_layer_cache
from fla.ops.gdn2 import chunk_gdn2

from experiments.zoology_mqar.gdn2_futureseed import (
    ZoologyGDN2FutureSeedMixer,
)


PAYLOAD_LOG_RADIUS = float(torch.log(torch.tensor(2.0)).item())
EXPECTED_NEW_PARAMETERS = 8
EXPECTED_STATE_VALUES = 4_096


def _rms(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.float().square().mean().sqrt()


class ReciprocalPayloadGaugeGDN2(nn.Module):
    """Official GDN2 with a reciprocal address-conditioned V payload code."""

    def __init__(self, base: nn.Module) -> None:
        super().__init__()
        if base.hidden_size != 128:
            raise ValueError("P-GDN3-048 requires D128")
        if base.num_heads != 4 or base.num_v_heads != 4:
            raise ValueError("P-GDN3-048 requires H4")
        if base.head_k_dim != 32 or base.head_v_dim != 32:
            raise ValueError("P-GDN3-048 requires K32/V32")
        if base.mode != "chunk" or not base.use_short_conv:
            raise ValueError("P-GDN3-048 requires chunk GDN2 with short convolution")
        self.base = base
        self.payload_gauge_logit = nn.Parameter(torch.zeros(base.num_heads, 1))
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
            raise RuntimeError("P-GDN3-048 requires an explicit layer index")
        return int(self.base.layer_idx)

    def set_capture(self, enabled: bool) -> None:
        self._capture = bool(enabled)
        if enabled:
            self._captured.clear()

    def log_strength(self, dtype: torch.dtype | None = None) -> torch.Tensor:
        strength = PAYLOAD_LOG_RADIUS * torch.tanh(
            self.payload_gauge_logit.float()
        )
        return strength if dtype is None else strength.to(dtype=dtype)

    def payload_factor(
        self,
        address: torch.Tensor,
        *,
        inverse: bool = False,
    ) -> torch.Tensor:
        if address.shape[-2:] != (self.num_heads, self.head_k_dim):
            raise ValueError(f"Unexpected address shape: {tuple(address.shape)}")
        unit = F.normalize(address.float(), dim=-1, eps=1e-6)
        strength = self.log_strength().view(
            *((1,) * (address.ndim - 2)), self.num_heads, 1
        )
        log_factor = unit * strength
        if inverse:
            log_factor = -log_factor
        return log_factor.exp().to(dtype=address.dtype)

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
            raise ValueError("P-GDN3-048 uses fixed unpadded MQAR batches")

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
        g = -layer.A_log.float().exp().unsqueeze(-1) * g
        write_factor = self.payload_factor(k)
        read_inverse = self.payload_factor(q, inverse=True)
        coded_value = value * write_factor
        if layer.allow_neg_eigval:
            erase_gate = erase_gate * 2.0

        recurrent_state = (
            last_state["recurrent_state"] if last_state is not None else None
        )
        output, recurrent_state = chunk_gdn2(
            q=q,
            k=k,
            v=coded_value,
            g=g,
            b=erase_gate,
            w=write_gate,
            initial_state=recurrent_state,
            output_final_state=use_cache,
            use_qk_l2norm_in_kernel=True,
            cu_seqlens=cu_seqlens,
        )
        decoded_output = output * read_inverse
        update_layer_cache(
            layer,
            past_key_values,
            recurrent_state=recurrent_state,
            conv_state=(conv_state_q, conv_state_k, conv_state_v),
            offset=q_len,
        )
        if self._capture:
            if recurrent_state is None:
                raise RuntimeError("P-GDN3-048 capture requires terminal state")
            self._captured = {
                "q": q.detach(),
                "k": k.detach(),
                "value": value.detach(),
                "coded_value": coded_value.detach(),
                "raw_output": output.detach(),
                "decoded_output": decoded_output.detach(),
                "write_factor": write_factor.detach(),
                "read_inverse": read_inverse.detach(),
                "log_strength": self.log_strength(q.dtype).detach(),
                "terminal_state": recurrent_state.detach(),
            }

        output = layer.o_norm(
            decoded_output,
            rearrange(
                layer.g_proj(hidden_states),
                "... (h d) -> ... h d",
                d=layer.head_v_dim,
            ),
        )
        output = rearrange(output, "b t h d -> b t (h d)")
        return layer.o_proj(output), None, past_key_values


class ZoologyAddressPayloadGaugeFutureSeedMixer(ZoologyGDN2FutureSeedMixer):
    """Native FutureSeed over address-payload-gauged official GDN2."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.layer = ReciprocalPayloadGaugeGDN2(self.layer)


def load_matched_parent_state(
    model: nn.Module,
    control_state: dict[str, torch.Tensor],
) -> None:
    loaded: dict[str, torch.Tensor] = {}
    for name, tensor in model.state_dict().items():
        if ".sequence_mixer.layer.payload_gauge_logit" in name:
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
        if ".sequence_mixer.layer.payload_gauge_logit" in name:
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
def address_payload_gauge_diagnostics(
    model: nn.Module,
    inputs: torch.Tensor,
) -> dict[str, Any]:
    mixers = [block.sequence_mixer for block in model.backbone.layers]
    if len(mixers) != 2 or not all(
        isinstance(mixer, ZoologyAddressPayloadGaugeFutureSeedMixer) for mixer in mixers
    ):
        raise RuntimeError("Expected two address-payload-gauged GDN2 mixers")
    for mixer in mixers:
        mixer.layer.set_capture(True)
    try:
        model.eval()
        model(inputs)
    finally:
        for mixer in mixers:
            mixer.layer.set_capture(False)

    rows = []
    for mixer in mixers:
        captured = mixer.layer._captured
        required = {
            "q", "k", "value", "coded_value", "raw_output",
            "decoded_output", "write_factor", "read_inverse",
            "log_strength", "terminal_state",
        }
        if set(captured) != required:
            raise RuntimeError(f"Incomplete address payload gauge capture: {set(captured)}")
        q = captured["q"].float()
        k = captured["k"].float()
        value = captured["value"].float()
        coded_value = captured["coded_value"].float()
        raw_output = captured["raw_output"].float()
        decoded_output = captured["decoded_output"].float()
        write_factor = captured["write_factor"].float()
        read_inverse = captured["read_inverse"].float()
        same_address_inverse = mixer.layer.payload_factor(
            captured["k"], inverse=True
        ).float()
        replay_write = mixer.layer.payload_factor(captured["k"]).float()
        replay_read = mixer.layer.payload_factor(
            captured["q"], inverse=True
        ).float()
        terminal = captured["terminal_state"].float()
        terminal_board = terminal.square().mean(dim=(-1, -2, -3)).sqrt()
        q_unit = F.normalize(q, dim=-1, eps=1e-6)
        k_unit = F.normalize(k, dim=-1, eps=1e-6)
        mismatch_factor = write_factor * read_inverse
        factor_board = write_factor.square().mean(dim=(1, 2, 3)).sqrt()
        rows.append(
            {
                "layer_idx": mixer.layer_idx,
                "adapter_rms": float(_rms(mixer.layer.payload_gauge_logit).item()),
                "log_strength_per_head": [
                    float(value.item())
                    for value in captured["log_strength"].float().flatten()
                ],
                "active_heads": int(
                    (captured["log_strength"].float().abs() >= 1e-4).sum().item()
                ),
                "factor_min": float(write_factor.min().item()),
                "factor_max": float(write_factor.max().item()),
                "factor_token_std": float(
                    write_factor.std(dim=1, unbiased=False).mean().item()
                ),
                "factor_board_std": float(factor_board.std(unbiased=False).item()),
                "production_replay_max_error": float(
                    torch.maximum(
                        (replay_write - write_factor).abs().max(),
                        (replay_read - read_inverse).abs().max(),
                    ).item()
                ),
                "same_address_reciprocal_max_error": float(
                    (write_factor * same_address_inverse - 1.0).abs().max().item()
                ),
                "qk_code_mismatch_rms": float(_rms(mismatch_factor - 1.0).item()),
                "coded_value_change_relative_rms": float(
                    (_rms(coded_value - value) / _rms(value).clamp_min(1e-8)).item()
                ),
                "decoded_output_change_relative_rms": float(
                    (
                        _rms(decoded_output - raw_output)
                        / _rms(raw_output).clamp_min(1e-8)
                    ).item()
                ),
                "raw_qk_cosine": float((q_unit * k_unit).sum(dim=-1).mean().item()),
                "terminal_state_rms": float(_rms(terminal).item()),
                "terminal_state_board_std": float(
                    terminal_board.std(unbiased=False).item()
                ),
            }
        )
    return {
        "active_layers": len(rows),
        "logical_scans_per_layer": 1,
        "new_parameters": sum(
            mixer.layer.payload_gauge_logit.numel() for mixer in mixers
        ),
        "new_persistent_state_values": 0,
        "payload_log_radius": PAYLOAD_LOG_RADIUS,
        "factor_bound": 2.0,
        "per_layer": rows,
    }
