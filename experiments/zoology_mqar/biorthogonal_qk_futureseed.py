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
    ZoologyGDN2FutureSeedMixer,
)
from experiments.zoology_mqar.gdn2_log_spd import BoundedLogSPDAddressMetric


LOG_FACTOR_RADIUS = 0.5 * math.log(2.0)
EXPECTED_STATE_VALUES = 4_096


def _rms(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.float().square().mean().sqrt()


class BoundedBiorthogonalQKGauge(BoundedLogSPDAddressMetric):
    """Bounded dual coordinates that preserve the unnormalized Q/K pairing."""

    def matrices(self) -> tuple[torch.Tensor, torch.Tensor]:
        generator = self.generator()
        with torch.autocast(device_type=self.raw.device.type, enabled=False):
            forward = torch.matrix_exp(LOG_FACTOR_RADIUS * generator)
            inverse = torch.matrix_exp(-LOG_FACTOR_RADIUS * generator)
        return forward, inverse

    def applied_matrices(
        self,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        forward, inverse = self.matrices()
        identity = torch.eye(
            self.head_dim,
            device=forward.device,
            dtype=forward.dtype,
        )
        applied_identity = identity.to(dtype=dtype)
        applied_forward = applied_identity + (forward - identity).to(dtype=dtype)
        applied_inverse = applied_identity + (inverse - identity).to(dtype=dtype)
        return applied_forward.float(), applied_inverse.float()

    def transform_pair(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        self._check_shape(q)
        self._check_shape(k)
        forward, inverse = self.matrices()
        identity = torch.eye(
            self.head_dim,
            device=q.device,
            dtype=torch.float32,
        )
        transformed_q = self._transform(q, inverse - identity)
        transformed_k = self._transform(k, forward - identity)
        return transformed_q, transformed_k


class BiorthogonalQKGDN2(nn.Module):
    """Official GDN2 with inverse-transpose Q/K address coordinates."""

    def __init__(self, base: nn.Module) -> None:
        super().__init__()
        if base.hidden_size != 128:
            raise ValueError("P-GDN3-043 requires D128")
        if base.num_heads != 4 or base.num_v_heads != 4:
            raise ValueError("P-GDN3-043 requires H4")
        if base.head_k_dim != 32 or base.head_v_dim != 32:
            raise ValueError("P-GDN3-043 requires K32/V32")
        if base.mode != "chunk" or not base.use_short_conv:
            raise ValueError("P-GDN3-043 requires chunk GDN2 with short convolution")
        self.base = base
        self.dual_gauge = BoundedBiorthogonalQKGauge(
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
            raise RuntimeError("P-GDN3-043 requires an explicit layer index")
        return int(self.base.layer_idx)

    def set_capture(self, enabled: bool) -> None:
        self._capture = bool(enabled)
        if enabled:
            self._captured.clear()

    def transform_pair(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.dual_gauge.transform_pair(q, k)

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
            raise ValueError("P-GDN3-043 uses fixed unpadded MQAR batches")

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
        transformed_q, transformed_k = self.transform_pair(q, k)
        if layer.allow_neg_eigval:
            erase_gate = erase_gate * 2.0

        recurrent_state = (
            last_state["recurrent_state"] if last_state is not None else None
        )
        output, recurrent_state = chunk_gdn2(
            q=transformed_q,
            k=transformed_k,
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
        if self._capture:
            if recurrent_state is None:
                raise RuntimeError("P-GDN3-043 capture requires terminal state")
            forward, inverse = self.dual_gauge.applied_matrices(q.dtype)
            self._captured = {
                "q": q.detach(),
                "k": k.detach(),
                "transformed_q": transformed_q.detach(),
                "transformed_k": transformed_k.detach(),
                "forward": forward.detach(),
                "inverse": inverse.detach(),
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


class ZoologyBiorthogonalQKFutureSeedMixer(ZoologyGDN2FutureSeedMixer):
    """Native FutureSeed over biorthogonal official GDN2 addresses."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.layer = BiorthogonalQKGDN2(self.layer)


def load_matched_parent_state(
    model: nn.Module,
    control_state: dict[str, torch.Tensor],
) -> None:
    loaded: dict[str, torch.Tensor] = {}
    for name, tensor in model.state_dict().items():
        if ".sequence_mixer.layer.dual_gauge.raw" in name:
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
        if ".sequence_mixer.layer.dual_gauge.raw" in name:
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
def biorthogonal_qk_diagnostics(
    model: nn.Module,
    inputs: torch.Tensor,
) -> dict[str, Any]:
    mixers = [block.sequence_mixer for block in model.backbone.layers]
    if len(mixers) != 2 or not all(
        isinstance(mixer, ZoologyBiorthogonalQKFutureSeedMixer)
        for mixer in mixers
    ):
        raise RuntimeError("Expected two biorthogonal Q/K GDN2 mixers")
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
            "q",
            "k",
            "transformed_q",
            "transformed_k",
            "forward",
            "inverse",
            "terminal_state",
        }
        if set(captured) != required:
            raise RuntimeError(
                f"Incomplete biorthogonal Q/K capture: {set(captured)}"
            )
        replay_q, replay_k = mixer.layer.transform_pair(
            captured["q"],
            captured["k"],
        )
        production_replay_error = torch.maximum(
            (replay_q - captured["transformed_q"]).abs().max(),
            (replay_k - captured["transformed_k"]).abs().max(),
        )
        q = captured["q"].float()
        k = captured["k"].float()
        tq = captured["transformed_q"].float()
        tk = captured["transformed_k"].float()
        forward = captured["forward"].float()
        inverse = captured["inverse"].float()
        identity = torch.eye(
            mixer.layer.head_k_dim,
            device=forward.device,
            dtype=forward.dtype,
        )
        inverse_error = (inverse @ forward - identity).abs().max()
        raw_pair = (q * k).sum(dim=-1)
        transformed_pair = (tq * tk).sum(dim=-1)
        pairing_relative_rms = _rms(transformed_pair - raw_pair) / _rms(
            raw_pair
        ).clamp_min(1e-8)
        generator = mixer.layer.dual_gauge.generator()
        theoretical_forward, _ = mixer.layer.dual_gauge.matrices()
        eigenvalues = torch.linalg.eigvalsh(theoretical_forward)
        terminal = captured["terminal_state"].float()
        terminal_board = terminal.square().mean(dim=(-1, -2, -3)).sqrt()
        raw = mixer.layer.dual_gauge.raw.detach().float()
        per_head_adapter = raw.square().mean(dim=-1).sqrt()
        q_unit = F.normalize(q, dim=-1, eps=1e-6)
        k_unit = F.normalize(k, dim=-1, eps=1e-6)
        tq_unit = F.normalize(tq, dim=-1, eps=1e-6)
        tk_unit = F.normalize(tk, dim=-1, eps=1e-6)
        rows.append(
            {
                "layer_idx": mixer.layer_idx,
                "adapter_rms": float(_rms(raw).item()),
                "adapter_per_head_rms": [
                    float(value.item()) for value in per_head_adapter
                ],
                "generator_trace_abs_max": float(
                    generator.diagonal(dim1=-2, dim2=-1)
                    .sum(dim=-1)
                    .abs()
                    .max()
                    .item()
                ),
                "factor_eigenvalue_min": float(eigenvalues.min().item()),
                "factor_eigenvalue_max": float(eigenvalues.max().item()),
                "factor_condition_max": float(
                    (eigenvalues[..., -1] / eigenvalues[..., 0]).max().item()
                ),
                "factor_logdet_abs_max": float(
                    torch.linalg.slogdet(theoretical_forward)
                    .logabsdet.abs()
                    .max()
                    .item()
                ),
                "applied_inverse_max_error": float(inverse_error.item()),
                "pairing_relative_rms_error": float(
                    pairing_relative_rms.item()
                ),
                "production_replay_max_error": float(
                    production_replay_error.item()
                ),
                "q_change_relative_rms": float(
                    (_rms(tq - q) / _rms(q).clamp_min(1e-8)).item()
                ),
                "k_change_relative_rms": float(
                    (_rms(tk - k) / _rms(k).clamp_min(1e-8)).item()
                ),
                "raw_qk_cosine": float(
                    (q_unit * k_unit).sum(dim=-1).mean().item()
                ),
                "transformed_qk_cosine": float(
                    (tq_unit * tk_unit).sum(dim=-1).mean().item()
                ),
                "factor_delta_fro_std": float(
                    (forward - identity)
                    .square()
                    .sum(dim=(-1, -2))
                    .sqrt()
                    .std(unbiased=False)
                    .item()
                ),
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
            mixer.layer.dual_gauge.raw.numel() for mixer in mixers
        ),
        "new_persistent_state_values": 0,
        "log_factor_radius": LOG_FACTOR_RADIUS,
        "per_layer": rows,
    }
