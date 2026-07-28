from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn as nn


FUTURE_SEED_GATE_MODES = ("head", "state")


class FutureSeedSelectiveGate(nn.Module):
    """Select imported recurrent-state entries without changing their basis."""

    def __init__(
        self,
        *,
        mode: str,
        layers: int,
        heads: int,
        row_dim: int,
        col_dim: int,
    ) -> None:
        super().__init__()
        if mode not in FUTURE_SEED_GATE_MODES:
            raise ValueError(f"Unknown FutureSeed gate mode: {mode}")
        if layers < 1:
            raise ValueError("FutureSeed selective gate requires at least one layer")
        if mode == "state" and layers < 2:
            raise ValueError("State-selective FutureSeed requires at least two layers")
        if heads <= 0 or row_dim <= 0 or col_dim <= 0:
            raise ValueError("FutureSeed state dimensions must be positive")
        self.mode = mode
        self.layers = int(layers)
        self.heads = int(heads)
        self.row_dim = int(row_dim)
        self.col_dim = int(col_dim)
        if mode == "state":
            self.gate_delta = nn.Parameter(
                torch.zeros(layers - 1, heads, row_dim, col_dim)
            )
            self.gate_delta._no_weight_decay = True
        else:
            self.register_parameter("gate_delta", None)

    @staticmethod
    def _rms(value: torch.Tensor) -> torch.Tensor:
        return value.float().square().mean(dim=(-1, -2)).sqrt()

    def forward(
        self,
        state: torch.Tensor,
        *,
        base_logit: torch.Tensor,
        layer_idx: int,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        if state.ndim != 4:
            raise ValueError(
                f"FutureSeed state must be [batch, heads, rows, cols], got {tuple(state.shape)}"
            )
        expected = (self.heads, self.row_dim, self.col_dim)
        if tuple(state.shape[1:]) != expected:
            raise ValueError(
                f"FutureSeed state shape {tuple(state.shape[1:])} does not match {expected}"
            )
        if tuple(base_logit.shape) != (1, self.heads, 1, 1):
            raise ValueError(
                "FutureSeed base logit must have shape "
                f"{(1, self.heads, 1, 1)}, got {tuple(base_logit.shape)}"
            )
        if not 1 <= layer_idx < self.layers:
            raise ValueError(
                f"FutureSeed destination layer must be in [1, {self.layers - 1}], got {layer_idx}"
            )

        base_gate = torch.sigmoid(base_logit)
        zero = state.new_zeros(())
        if self.mode == "head":
            return base_gate, {
                "fs2_gate_delta_rms": zero,
                "fs2_gate_std": zero,
                "fs2_gate_min": base_gate.min().to(dtype=state.dtype),
                "fs2_gate_max": base_gate.max().to(dtype=state.dtype),
                "fs2_seed_relative_change": zero,
            }

        assert self.gate_delta is not None
        delta = self.gate_delta[layer_idx - 1].unsqueeze(0)
        gate = torch.sigmoid(base_logit + delta)
        baseline_seed = state * base_gate
        selected_seed = state * gate
        baseline_rms = self._rms(baseline_seed).clamp_min(1e-8)
        change_rms = self._rms(selected_seed - baseline_seed)
        return gate, {
            "fs2_gate_delta_rms": delta.float().square()
            .mean()
            .sqrt()
            .to(dtype=state.dtype),
            "fs2_gate_std": gate.float()
            .std(dim=(-1, -2), unbiased=False)
            .mean()
            .to(dtype=state.dtype),
            "fs2_gate_min": gate.min().to(dtype=state.dtype),
            "fs2_gate_max": gate.max().to(dtype=state.dtype),
            "fs2_seed_relative_change": (change_rms / baseline_rms)
            .mean()
            .to(dtype=state.dtype),
        }
