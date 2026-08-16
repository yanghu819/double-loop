from __future__ import annotations

from typing import Any, Optional

import torch
from torch import nn

from experiments.zoology_mqar.momentum_futureseed import (
    EXPECTED_MDN_SHA,
    EXPECTED_STATE_VALUES_PER_LAYER,
    MODEL_HEADS,
    ZoologyMomentumDeltaFutureSeedMixer,
    load_matched_parent_state,
    momentum_futureseed_diagnostics,
    parent_parameter_hash,
)


EXPECTED_PARAMETER_DELTA_VS_MOMENTUM = MODEL_HEADS


class ZoologyMomentumPhaseFutureSeedMixer(
    ZoologyMomentumDeltaFutureSeedMixer
):
    """Native Momentum FutureSeed with receiver-local S/M phase transport."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        if self.layer_idx == 0:
            self.register_parameter("future_seed_phase", None)
        else:
            self.future_seed_phase = nn.Parameter(
                torch.zeros(1, self.layer.num_v_heads, 1, 1)
            )
        self.last_phase_residual_relative_rms: Optional[torch.Tensor] = None
        self.last_phase_residual_board_std: Optional[torch.Tensor] = None
        self.last_phase_residual_head_std: Optional[torch.Tensor] = None
        self.last_phase_energy_relative_error: Optional[torch.Tensor] = None

    def make_initial_state(self, terminal_state: torch.Tensor) -> torch.Tensor:
        seed = super().make_initial_state(terminal_state)
        if self.future_seed_phase is None:
            return seed
        if seed.ndim != 5 or seed.shape[0] != 2:
            raise RuntimeError(f"Momentum seed must be [2,B,H,K,V], got {seed.shape}")

        phase = self.future_seed_phase.float()
        cosine = torch.cos(phase)
        sine = torch.sin(phase)
        state = seed[0].float()
        momentum = seed[1].float()
        rotated_state = cosine * state - sine * momentum
        rotated_momentum = sine * state + cosine * momentum
        rotated = torch.stack((rotated_state, rotated_momentum), dim=0)

        residual = rotated - seed.float()
        base_rms = seed.float().square().mean().sqrt().clamp_min(1e-8)
        board_rms = residual.square().mean(dim=(0, 2, 3, 4)).sqrt()
        head_rms = residual.square().mean(dim=(0, 1, 3, 4)).sqrt()
        energy_before = seed.float().square().sum(dim=(0, 3, 4))
        energy_after = rotated.square().sum(dim=(0, 3, 4))
        energy_error = (
            (energy_after - energy_before).abs()
            / energy_before.clamp_min(1e-8)
        )
        self.last_phase_residual_relative_rms = (
            residual.square().mean().sqrt() / base_rms
        ).detach()
        self.last_phase_residual_board_std = board_rms.std(unbiased=False).detach()
        self.last_phase_residual_head_std = head_rms.std(unbiased=False).detach()
        self.last_phase_energy_relative_error = energy_error.max().detach()
        return rotated.to(dtype=seed.dtype)


def momentum_phase_diagnostics(model: torch.nn.Module) -> dict[str, Any]:
    momentum = momentum_futureseed_diagnostics(model)
    mixers = [block.sequence_mixer for block in model.backbone.layers]
    if not all(
        type(mixer) is ZoologyMomentumPhaseFutureSeedMixer for mixer in mixers
    ):
        raise RuntimeError("Expected only Momentum phase FutureSeed mixers")

    rows = []
    angle_values = []
    for mixer in mixers:
        phase = mixer.future_seed_phase
        if phase is not None:
            phase_float = phase.detach().float()
            angle_values.append(phase_float.reshape(-1))
            angle = {
                "values": phase_float.reshape(-1).cpu().tolist(),
                "abs_mean": float(phase_float.abs().mean()),
                "abs_min": float(phase_float.abs().min()),
                "sin_abs_mean": float(torch.sin(phase_float).abs().mean()),
                "sin_abs_min": float(torch.sin(phase_float).abs().min()),
            }
        else:
            angle = None
        rows.append(
            {
                "layer_idx": mixer.layer_idx,
                "receiving": phase is not None,
                "angle": angle,
                "residual_relative_rms": (
                    None
                    if mixer.last_phase_residual_relative_rms is None
                    else float(mixer.last_phase_residual_relative_rms)
                ),
                "residual_board_std": (
                    None
                    if mixer.last_phase_residual_board_std is None
                    else float(mixer.last_phase_residual_board_std)
                ),
                "residual_head_std": (
                    None
                    if mixer.last_phase_residual_head_std is None
                    else float(mixer.last_phase_residual_head_std)
                ),
                "energy_relative_error": (
                    None
                    if mixer.last_phase_energy_relative_error is None
                    else float(mixer.last_phase_energy_relative_error)
                ),
            }
        )

    angles = torch.cat(angle_values) if angle_values else torch.empty(0)
    return {
        "momentum": momentum,
        "external_sha": EXPECTED_MDN_SHA,
        "parameter_delta_vs_momentum": int(angles.numel()),
        "persistent_state_delta_vs_momentum": 0,
        "scan_delta_vs_momentum": 0,
        "receiving_routes": len(angle_values),
        "active_receiving_routes": sum(
            row["angle"] is not None and row["angle"]["sin_abs_mean"] > 0.0
            for row in rows
        ),
        "angle_abs_mean": None if not angles.numel() else float(angles.abs().mean()),
        "angle_sin_abs_mean": (
            None if not angles.numel() else float(torch.sin(angles).abs().mean())
        ),
        "angle_sin_abs_min": (
            None if not angles.numel() else float(torch.sin(angles).abs().min())
        ),
        "state_values_per_layer": EXPECTED_STATE_VALUES_PER_LAYER,
        "per_layer_phase": rows,
    }


__all__ = [
    "EXPECTED_PARAMETER_DELTA_VS_MOMENTUM",
    "ZoologyMomentumPhaseFutureSeedMixer",
    "load_matched_parent_state",
    "momentum_phase_diagnostics",
    "parent_parameter_hash",
]
