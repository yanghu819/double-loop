from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F


GDN2_PRECONDITION_MODES = (
    "none",
    "tied_atk",
    "futureseed_tied_atk",
)


def futureseed_row_precision(
    initial_state: Optional[torch.Tensor],
) -> Optional[torch.Tensor]:
    """Use recurrent-state row energy as a parameter-free confidence prior."""
    if initial_state is None:
        return None
    if initial_state.ndim != 4:
        raise ValueError(
            "FutureSeed confidence expects [batch, head, key, value] state, "
            f"got {tuple(initial_state.shape)}"
        )
    return initial_state.float().square().mean(dim=-1)


def causal_tied_atk_preconditioner(
    k_unit: torch.Tensor,
    g: torch.Tensor,
    b: torch.Tensor,
    *,
    initial_precision: Optional[torch.Tensor] = None,
    squash_x: float = 1.5,
    eps: float = 1e-6,
    log_center: float = -0.2,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """Compute PGDN-style causal diagonal curvature from GDN2's own gates.

    The auxiliary statistic follows

        A_t = exp(g_t) * A_{t-1} + b_t * k_t**2.

    It deliberately reuses GDN2's per-key decay and erase gates. This adds no
    learned projection and lets an existing GDN2 checkpoint be resumed without
    parameter migration. The log-domain prefix form is parallel over time.
    """
    if k_unit.shape != g.shape or k_unit.shape != b.shape:
        raise ValueError(
            "Tied ATK expects matching [batch,time,head,key] tensors: "
            f"k={tuple(k_unit.shape)} g={tuple(g.shape)} b={tuple(b.shape)}"
        )
    if squash_x < 1.0 or squash_x > 2.0:
        raise ValueError("squash_x must be in [1, 2] for a bounded stable update")
    if eps <= 0:
        raise ValueError("eps must be positive")

    k_float = k_unit.float()
    g_float = g.float()
    b_float = b.float().clamp_min(0.0)
    cumulative_g = g_float.cumsum(dim=1)
    log_evidence = (
        torch.log(b_float.clamp_min(eps))
        + torch.log(k_float.square().clamp_min(eps))
        - cumulative_g
    )
    log_accumulator = torch.logcumsumexp(log_evidence, dim=1)

    seed_precision = None
    if initial_precision is not None:
        expected = (k_unit.shape[0], k_unit.shape[2], k_unit.shape[3])
        if tuple(initial_precision.shape) != expected:
            raise ValueError(
                f"initial_precision shape {tuple(initial_precision.shape)} "
                f"does not match {expected}"
            )
        seed_precision = initial_precision.float().clamp_min(0.0)
        log_seed = torch.log(seed_precision.clamp_min(eps)).unsqueeze(1)
        log_accumulator = torch.logaddexp(log_accumulator, log_seed)

    log_precision = cumulative_g + log_accumulator
    log_eps = torch.tensor(math.log(eps), device=k_unit.device, dtype=torch.float32)
    ell = torch.logaddexp(log_precision, log_eps)
    centered = ell - float(log_center)
    squashed = centered / (1.0 + centered.abs())
    multiplier = torch.exp(-math.log(float(squash_x)) * squashed)

    zero = k_unit.new_zeros((), dtype=torch.float32)
    diagnostics = {
        "gdn2_precondition_enabled": k_unit.new_ones((), dtype=torch.float32),
        "gdn2_precondition_multiplier_mean": multiplier.mean().detach(),
        "gdn2_precondition_multiplier_std": multiplier.std(unbiased=False).detach(),
        "gdn2_precondition_multiplier_min": multiplier.min().detach(),
        "gdn2_precondition_multiplier_max": multiplier.max().detach(),
        "gdn2_precondition_log_precision_mean": ell.mean().detach(),
        "gdn2_precondition_log_precision_std": ell.std(unbiased=False).detach(),
        "gdn2_precondition_seed_precision_mean": (
            seed_precision.mean().detach() if seed_precision is not None else zero
        ),
        "gdn2_precondition_seed_precision_std": (
            seed_precision.std(unbiased=False).detach()
            if seed_precision is not None
            else zero
        ),
    }
    return multiplier, diagnostics


def fold_preconditioned_write_into_gdn2(
    k_unit: torch.Tensor,
    b: torch.Tensor,
    multiplier: torch.Tensor,
    *,
    eps: float = 1e-6,
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
    """Give GDN2 separate erase and write keys without changing its kernel.

    Passing `k_write = m*k` and `b_kernel = b/m` preserves the erase address
    `(b_kernel*k_write) == (b*k)` while the rank-one write uses `m*k`.
    Inputs are quantized back to the official kernel dtype before diagnostics.
    """
    if k_unit.shape != b.shape or k_unit.shape != multiplier.shape:
        raise ValueError(
            "Precondition fold expects matching K-axis tensors: "
            f"k={tuple(k_unit.shape)} b={tuple(b.shape)} "
            f"multiplier={tuple(multiplier.shape)}"
        )
    multiplier_float = multiplier.float().clamp_min(eps)
    k_write = (k_unit.float() * multiplier_float).to(dtype=k_unit.dtype)
    b_kernel = (b.float() / multiplier_float).to(dtype=b.dtype)

    expected_erase = b.float() * k_unit.float()
    actual_erase = b_kernel.float() * k_write.float()
    erase_error = actual_erase - expected_erase
    write_delta = k_write.float() - k_unit.float()
    diagnostics = {
        "gdn2_precondition_write_relative_change": (
            write_delta.norm() / k_unit.float().norm().clamp_min(eps)
        ).detach(),
        "gdn2_precondition_erase_error_max": erase_error.abs().max().detach(),
        "gdn2_precondition_erase_error_rms": erase_error.square().mean().sqrt().detach(),
    }
    return k_write, b_kernel, diagnostics


def normalize_qk_fp32(
    q: torch.Tensor,
    k: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Match FLA's normalized Q/K contract before disabling kernel normalization."""
    return (
        F.normalize(q.float(), p=2.0, dim=-1).to(dtype=q.dtype),
        F.normalize(k.float(), p=2.0, dim=-1).to(dtype=k.dtype),
    )
