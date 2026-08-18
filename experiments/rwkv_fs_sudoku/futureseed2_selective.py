from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn as nn


FUTURE_SEED_GATE_MODES = ("head", "state", "content")


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
        if mode == "content":
            self.content_weight = nn.Parameter(
                torch.zeros(layers - 1, heads, 5)
            )
            self.content_weight._no_weight_decay = True
        else:
            self.register_parameter("content_weight", None)

    @staticmethod
    def _rms(value: torch.Tensor) -> torch.Tensor:
        return value.float().square().mean(dim=(-1, -2)).sqrt()

    @staticmethod
    def _content_features(state: torch.Tensor) -> torch.Tensor:
        value = state.float()
        rms = value.square().mean(dim=(-1, -2)).sqrt().clamp_min(1e-6)
        row_rms = value.square().mean(dim=-1).sqrt()
        col_rms = value.square().mean(dim=-2).sqrt()
        return torch.stack(
            (
                rms.log().clamp(min=-6.0, max=6.0),
                value.mean(dim=(-1, -2)) / rms,
                value.abs().mean(dim=(-1, -2)) / rms,
                row_rms.std(dim=-1, unbiased=False) / rms,
                col_rms.std(dim=-1, unbiased=False) / rms,
            ),
            dim=-1,
        )

    def forward(
        self,
        state: torch.Tensor,
        *,
        base_logit: torch.Tensor,
        layer_idx: int,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        if state.ndim not in {4, 5}:
            raise ValueError(
                "FutureSeed state must be [batch, heads, rows, cols] or "
                f"[planes, batch, heads, rows, cols], got {tuple(state.shape)}"
            )
        if state.ndim == 5 and self.mode != "head":
            raise ValueError(
                "Multi-plane FutureSeed currently supports only the canonical head gate"
            )
        state_shape = state.shape[1:] if state.ndim == 4 else state.shape[2:]
        expected = (self.heads, self.row_dim, self.col_dim)
        if tuple(state_shape) != expected:
            raise ValueError(
                f"FutureSeed state shape {tuple(state_shape)} does not match {expected}"
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
                "fs2_gate_batch_std": zero,
                "fs2_content_feature_std": zero,
                "fs2_gate_min": base_gate.min().to(dtype=state.dtype),
                "fs2_gate_max": base_gate.max().to(dtype=state.dtype),
                "fs2_seed_relative_change": zero,
            }

        if self.mode == "content":
            assert self.content_weight is not None
            features = self._content_features(state)
            weight = self.content_weight[layer_idx - 1]
            raw_delta = torch.einsum("bhf,hf->bh", features, weight)
            delta = torch.tanh(raw_delta).unsqueeze(-1).unsqueeze(-1)
            gate = torch.sigmoid(base_logit + delta)
            baseline_seed = state * base_gate
            selected_seed = state * gate
            baseline_rms = self._rms(baseline_seed).clamp_min(1e-8)
            change_rms = self._rms(selected_seed - baseline_seed)
            return gate, {
                "fs2_gate_delta_rms": delta.float()
                .square()
                .mean()
                .sqrt()
                .to(dtype=state.dtype),
                "fs2_gate_std": zero,
                "fs2_gate_batch_std": gate.float()
                .std(dim=0, unbiased=False)
                .mean()
                .to(dtype=state.dtype),
                "fs2_content_feature_std": features.std(
                    dim=0, unbiased=False
                )
                .mean()
                .to(dtype=state.dtype),
                "fs2_gate_min": gate.min().to(dtype=state.dtype),
                "fs2_gate_max": gate.max().to(dtype=state.dtype),
                "fs2_seed_relative_change": (change_rms / baseline_rms)
                .mean()
                .to(dtype=state.dtype),
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
            "fs2_gate_batch_std": gate.float()
            .std(dim=0, unbiased=False)
            .mean()
            .to(dtype=state.dtype),
            "fs2_content_feature_std": zero,
            "fs2_gate_min": gate.min().to(dtype=state.dtype),
            "fs2_gate_max": gate.max().to(dtype=state.dtype),
            "fs2_seed_relative_change": (change_rms / baseline_rms)
            .mean()
            .to(dtype=state.dtype),
        }
