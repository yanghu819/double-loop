from __future__ import annotations

import math
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


FAST_SLOW_DECAY_MODES = ("none", "external_identity", "positive_causal")


def _logit(probability: float) -> float:
    probability = min(max(float(probability), 1e-6), 1.0 - 1e-6)
    return math.log(probability / (1.0 - probability))


def positive_causal_hazard_smooth(
    hazard: torch.Tensor,
    kernel: torch.Tensor,
) -> torch.Tensor:
    """Apply a per-head positive causal FIR with boundary renormalization."""
    if hazard.ndim != 4:
        raise ValueError(f"hazard must have shape (B,T,H,K), got {tuple(hazard.shape)}")
    if kernel.ndim != 2:
        raise ValueError(f"kernel must have shape (H,L), got {tuple(kernel.shape)}")
    if hazard.shape[2] != kernel.shape[0]:
        raise ValueError(
            f"hazard heads {hazard.shape[2]} do not match kernel heads {kernel.shape[0]}"
        )
    if kernel.shape[1] < 1:
        raise ValueError("kernel length must be positive")
    kernel_sum = kernel.sum(dim=-1, keepdim=True)
    normalized = kernel / kernel_sum
    numerator = torch.zeros_like(hazard)
    _batch, length, heads, _channels = hazard.shape
    available_lags = torch.arange(length, device=hazard.device).clamp_max(
        normalized.shape[1] - 1
    )
    denominator = (
        normalized.cumsum(dim=-1)[:, available_lags]
        .transpose(0, 1)
        .view(1, length, heads, 1)
    )
    for lag in range(normalized.shape[1]):
        if lag == 0:
            shifted_hazard = hazard
        elif lag >= length:
            shifted_hazard = torch.zeros_like(hazard)
        else:
            shifted_hazard = F.pad(
                hazard[:, :-lag],
                (0, 0, 0, 0, lag, 0),
            )
        weight = normalized[:, lag].view(1, 1, heads, 1)
        numerator = numerator + shifted_hazard * weight
    return numerator / denominator.clamp_min(torch.finfo(hazard.dtype).tiny)


class FastSlowDecayController(nn.Module):
    """Slow decay control with an untouched fast erase/write path."""

    def __init__(
        self,
        *,
        heads: int,
        kernel_size: int = 4,
        rho_init: float = 0.10,
        current_weight_init: float = 0.85,
    ) -> None:
        super().__init__()
        if heads < 1:
            raise ValueError("heads must be positive")
        if kernel_size < 1:
            raise ValueError("kernel_size must be positive")
        if not (0.0 < rho_init < 1.0):
            raise ValueError("rho_init must be in (0, 1)")
        if not (0.0 < current_weight_init <= 1.0):
            raise ValueError("current_weight_init must be in (0, 1]")
        if kernel_size == 1 and not math.isclose(current_weight_init, 1.0):
            raise ValueError("kernel_size=1 requires current_weight_init=1")

        if kernel_size == 1:
            probabilities = torch.ones(1)
        else:
            lag_weight = (1.0 - float(current_weight_init)) / float(kernel_size - 1)
            probabilities = torch.tensor(
                [float(current_weight_init)] + [lag_weight] * (kernel_size - 1),
                dtype=torch.float32,
            )
        self.kernel_logits = nn.Parameter(
            probabilities.log().view(1, kernel_size).repeat(heads, 1)
        )
        self.rho_logit = nn.Parameter(
            torch.full((heads,), _logit(rho_init), dtype=torch.float32)
        )
        self.kernel_logits._no_weight_decay = True
        self.rho_logit._no_weight_decay = True

    def coefficients(self) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.kernel_logits.float().softmax(dim=-1), self.rho_logit.float().sigmoid()

    def forward(
        self,
        log_decay: torch.Tensor,
        *,
        mode: str,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        if mode not in FAST_SLOW_DECAY_MODES:
            raise ValueError(f"unknown fast-slow decay mode: {mode}")
        if log_decay.ndim != 4:
            raise ValueError(
                f"log_decay must have shape (B,T,H,K), got {tuple(log_decay.shape)}"
            )
        kernel, rho = self.coefficients()
        if kernel.shape[0] != log_decay.shape[2]:
            raise ValueError(
                f"controller heads {kernel.shape[0]} do not match log_decay "
                f"heads {log_decay.shape[2]}"
            )
        hazard = -log_decay.float()
        slow_hazard = positive_causal_hazard_smooth(hazard, kernel)
        if mode == "positive_causal":
            effective_hazard = hazard + rho.view(1, 1, -1, 1) * (
                slow_hazard - hazard
            )
        else:
            effective_hazard = hazard
        effective_log_decay = -effective_hazard

        # Keep production forward asynchronous. Value-domain invariants are
        # exhaustively checked by the pure-Torch and CUDA contract tests.
        diagnostic_hazard = hazard.detach()[:1, :, :, :1]
        diagnostic_effective = effective_hazard.detach()[:1, :, :, :1]
        raw_tv = (
            (diagnostic_hazard[:, 1:] - diagnostic_hazard[:, :-1]).abs().mean()
            if diagnostic_hazard.shape[1] > 1
            else diagnostic_hazard.new_zeros(())
        )
        effective_tv = (
            (
                diagnostic_effective[:, 1:]
                - diagnostic_effective[:, :-1]
            )
            .abs()
            .mean()
            if diagnostic_effective.shape[1] > 1
            else diagnostic_effective.new_zeros(())
        )
        relative_change = (
            (diagnostic_effective - diagnostic_hazard).square().mean().sqrt()
            / diagnostic_hazard.square().mean().sqrt().clamp_min(1e-8)
        )
        diagnostics = {
            "gdn2_fast_slow_enabled": hazard.new_tensor(
                float(mode == "positive_causal")
            ),
            "gdn2_fast_slow_rho_mean": rho.mean().detach(),
            "gdn2_fast_slow_rho_min": rho.min().detach(),
            "gdn2_fast_slow_rho_max": rho.max().detach(),
            "gdn2_fast_slow_current_weight": kernel[:, 0].mean().detach(),
            "gdn2_fast_slow_lag_mass": (1.0 - kernel[:, 0]).mean().detach(),
            "gdn2_fast_slow_raw_hazard_mean": diagnostic_hazard.mean(),
            "gdn2_fast_slow_effective_hazard_mean": diagnostic_effective.mean(),
            "gdn2_fast_slow_raw_tv": raw_tv.detach(),
            "gdn2_fast_slow_effective_tv": effective_tv.detach(),
            "gdn2_fast_slow_tv_ratio": (
                effective_tv / raw_tv.clamp_min(1e-8)
            ).detach(),
            "gdn2_fast_slow_relative_change": relative_change.detach(),
            "gdn2_fast_slow_alpha_mean": (-diagnostic_effective).exp().mean(),
        }
        return effective_log_decay, diagnostics
