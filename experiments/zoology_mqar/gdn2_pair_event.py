from __future__ import annotations

import hashlib
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from experiments.zoology_mqar.gdn2_futureseed import (
    FutureSeedLanguageModel,
    ZoologyGDN2FutureSeedMixer,
)


MODEL_WIDTH = 128
MODEL_HEADS = 4
HEAD_DIM = MODEL_WIDTH // MODEL_HEADS
PAIR_RANK = 64
EXPECTED_PARAMETER_DELTA_PER_LAYER = (
    2 * MODEL_WIDTH * PAIR_RANK
    + PAIR_RANK * MODEL_WIDTH
    + MODEL_HEADS
)
EXPECTED_PARAMETER_DELTA = 2 * EXPECTED_PARAMETER_DELTA_PER_LAYER


def _rms(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.float().square().mean().sqrt()


def _rms_normalize(tensor: torch.Tensor) -> torch.Tensor:
    scale = tensor.float().square().mean(dim=-1, keepdim=True).add(1e-6).sqrt()
    return tensor / scale.to(dtype=tensor.dtype)


class ZoologyPairEventGDN2FutureSeedMixer(ZoologyGDN2FutureSeedMixer):
    """Raven-style local event formation followed by native GDN2 memory."""

    def __init__(
        self,
        d_model: int,
        layer_idx: int,
        *,
        num_heads: int = MODEL_HEADS,
        head_dim: int = HEAD_DIM,
        **kwargs: Any,
    ) -> None:
        if d_model != MODEL_WIDTH or num_heads != MODEL_HEADS:
            raise ValueError("P-GDN3-057 is fixed to D128/H4")
        if head_dim != HEAD_DIM:
            raise ValueError("P-GDN3-057 is fixed to K32/V32")
        super().__init__(
            d_model=d_model,
            layer_idx=layer_idx,
            num_heads=num_heads,
            head_dim=head_dim,
            **kwargs,
        )
        self.event_current = nn.Linear(MODEL_WIDTH, PAIR_RANK, bias=False)
        self.event_previous = nn.Linear(MODEL_WIDTH, PAIR_RANK, bias=False)
        self.event_output = nn.Linear(PAIR_RANK, MODEL_WIDTH, bias=False)
        self.event_gate = nn.Parameter(torch.zeros(MODEL_HEADS))
        self.capture_pair_event = False
        self._pair_event_capture: dict[str, torch.Tensor] = {}

    def set_pair_event_capture(self, enabled: bool) -> None:
        self.capture_pair_event = bool(enabled)
        if enabled:
            self._pair_event_capture.clear()

    def _pair_delta(self, hidden_states: torch.Tensor) -> torch.Tensor:
        normalized = _rms_normalize(hidden_states)
        previous = F.pad(normalized[:, :-1], (0, 0, 1, 0))
        current_features = F.silu(self.event_current(normalized))
        previous_features = self.event_previous(previous)
        pair_features = current_features * previous_features
        raw = self.event_output(pair_features).reshape(
            hidden_states.shape[0],
            hidden_states.shape[1],
            MODEL_HEADS,
            HEAD_DIM,
        )
        unit = _rms_normalize(raw)
        gate = torch.tanh(self.event_gate).to(dtype=unit.dtype)
        delta = unit * gate.view(1, 1, MODEL_HEADS, 1)
        delta = delta.reshape_as(hidden_states).to(dtype=hidden_states.dtype)

        if self.capture_pair_event:
            with torch.no_grad():
                delta_float = delta.float()
                board_rms = delta_float.square().mean(dim=(1, 2)).sqrt()
                token_rms = delta_float.square().mean(dim=-1).sqrt()
                self._pair_event_capture = {
                    "gate": gate.detach().float(),
                    "pair_feature_rms": _rms(pair_features),
                    "raw_event_rms": _rms(raw),
                    "event_delta_rms": _rms(delta),
                    "event_delta_relative_rms": _rms(delta)
                    / _rms(hidden_states).clamp_min(1e-8),
                    "event_delta_board_std": board_rms.std(unbiased=False),
                    "event_delta_token_std": token_rms.std(unbiased=False),
                    "current_feature_rms": _rms(current_features),
                    "previous_feature_rms": _rms(previous_features),
                }
        return delta

    def _event_hidden(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return hidden_states + self._pair_delta(hidden_states)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return super().forward(self._event_hidden(hidden_states))

    def forward_with_state(
        self,
        hidden_states: torch.Tensor,
        *,
        initial_state: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return super().forward_with_state(
            self._event_hidden(hidden_states),
            initial_state=initial_state,
        )


def _is_extra_parameter(name: str) -> bool:
    return ".sequence_mixer.event_" in name


def load_matched_parent_state(
    model: nn.Module,
    control_state: dict[str, torch.Tensor],
) -> None:
    loaded: dict[str, torch.Tensor] = {}
    extras = 0
    for name, tensor in model.state_dict().items():
        if _is_extra_parameter(name):
            loaded[name] = (
                torch.zeros_like(tensor)
                if name.endswith(".sequence_mixer.event_gate")
                else tensor
            )
            extras += tensor.numel()
            continue
        if name not in control_state:
            raise RuntimeError(f"Missing matched parent tensor: {name}")
        parent = control_state[name]
        if parent.shape != tensor.shape:
            raise RuntimeError(
                f"Matched tensor shape mismatch for {name}: "
                f"{tuple(parent.shape)} != {tuple(tensor.shape)}"
            )
        loaded[name] = parent
    if extras != EXPECTED_PARAMETER_DELTA:
        raise RuntimeError(
            f"Expected {EXPECTED_PARAMETER_DELTA} new parameters, found {extras}"
        )
    model.load_state_dict(loaded, strict=True)


def parent_parameter_hash(model: nn.Module) -> str:
    digest = hashlib.sha256()
    tensors = [
        (name, parameter)
        for name, parameter in model.named_parameters()
        if not _is_extra_parameter(name)
    ]
    for name, parameter in sorted(tensors):
        digest.update(name.encode())
        digest.update(parameter.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


@torch.no_grad()
def pair_event_diagnostics(
    model: FutureSeedLanguageModel,
    inputs: torch.Tensor,
) -> dict[str, Any]:
    mixers = [block.sequence_mixer for block in model.backbone.layers]
    if len(mixers) != len(model.backbone.layers) or not all(
        isinstance(mixer, ZoologyPairEventGDN2FutureSeedMixer)
        for mixer in mixers
    ):
        raise RuntimeError("Every layer must use the pair-event mixer")
    for mixer in mixers:
        mixer.set_pair_event_capture(True)
    try:
        model.eval()(inputs)
    finally:
        for mixer in mixers:
            mixer.set_pair_event_capture(False)

    rows = []
    active_paths = 0
    for mixer in mixers:
        captured = mixer._pair_event_capture
        if not captured:
            raise RuntimeError("Pair-event capture was not produced")
        gates = captured["gate"].flatten()
        active_paths += int((gates.abs() >= 1e-3).sum().item())
        rows.append(
            {
                "layer_idx": mixer.layer_idx,
                "gate_abs_mean": float(gates.abs().mean().item()),
                "gate_abs_min": float(gates.abs().min().item()),
                "gate_abs_max": float(gates.abs().max().item()),
                "gate_values": [float(value) for value in gates.tolist()],
                **{
                    name: float(value.float().mean().item())
                    for name, value in captured.items()
                    if name != "gate"
                },
            }
        )
    return {
        "active_layers": len(rows),
        "active_paths": active_paths,
        "total_paths": len(rows) * MODEL_HEADS,
        "pair_rank": PAIR_RANK,
        "parameter_delta": EXPECTED_PARAMETER_DELTA,
        "persistent_state_delta": 0,
        "official_gdn2_scans_per_layer": 1,
        "layers": rows,
    }
