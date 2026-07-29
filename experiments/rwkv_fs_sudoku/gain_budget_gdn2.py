from __future__ import annotations

import math
from typing import Dict, Literal, Tuple

import torch


BudgetMode = Literal["none", "fixed_sigma", "decay_funded"]
InfeasiblePolicy = Literal["raise", "relax"]


def fla_l2norm_fp32(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Match FLA's L2Norm formula while retaining an FP32 result."""
    x_fp32 = x.float()
    return x_fp32 * torch.rsqrt(
        x_fp32.square().sum(dim=-1, keepdim=True) + float(eps)
    )


def _fail_closed_assert(condition: torch.Tensor, message: str) -> None:
    condition = condition.reshape(())
    if condition.device.type == "cuda" and hasattr(torch, "_assert_async"):
        torch._assert_async(condition, message)
        return
    if not bool(condition):
        raise ValueError(message)


def rank_one_transition_sigma(
    delta: torch.Tensor,
    shear2: torch.Tensor,
) -> torch.Tensor:
    """Exact largest singular value of I - k (b * k)^T for K >= 2."""
    x = 1.0 - delta.float()
    h2 = shear2.float().clamp_min(0.0)
    trace = x.square() + 1.0 + h2
    discriminant = (trace.square() - 4.0 * x.square()).clamp_min(0.0)
    return (0.5 * (trace + discriminant.sqrt())).clamp_min(0.0).sqrt()


def project_erase_gate(
    key: torch.Tensor,
    erase_gate: torch.Tensor,
    log_decay: torch.Tensor,
    *,
    mode: BudgetMode = "decay_funded",
    sigma_cap: float = 1.10,
    step_gain_cap: float = 1.0,
    sigma_cap_max: float = 3.0,
    norm_floor: float = 1e-12,
    infeasible_policy: InfeasiblePolicy = "raise",
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """Project GDN2 erase anisotropy under a per-token/head gain budget.

    Inputs have shape ``[..., K]``. ``key`` must be the exact normalized key
    used by the recurrence, ``erase_gate`` must already include the optional
    ``allow_neg_eigval`` factor, and ``log_decay`` must be the actual
    non-positive log decay.
    """
    if key.shape != erase_gate.shape or key.shape != log_decay.shape:
        raise ValueError("key, erase_gate, and log_decay must share shape [..., K]")
    if key.ndim < 1 or key.shape[-1] < 2:
        raise ValueError("gain-budget projection requires key dimension K >= 2")
    if mode not in {"none", "fixed_sigma", "decay_funded"}:
        raise ValueError(f"unknown gain-budget mode: {mode}")
    if infeasible_policy not in {"raise", "relax"}:
        raise ValueError(f"unknown infeasible policy: {infeasible_policy}")
    if not math.isfinite(norm_floor) or norm_floor <= 0:
        raise ValueError("norm_floor must be positive")
    if not math.isfinite(sigma_cap) or sigma_cap < 1.0:
        raise ValueError("sigma_cap must be at least 1")
    if not math.isfinite(step_gain_cap) or step_gain_cap < 1.0:
        raise ValueError("step_gain_cap must be at least 1")
    if not math.isfinite(sigma_cap_max) or sigma_cap_max < 1.0:
        raise ValueError("sigma_cap_max must be at least 1")

    key_fp32 = key.float()
    gate_fp32 = erase_gate.float()
    decay_fp32 = log_decay.float()
    finite = (
        torch.isfinite(key_fp32).all()
        & torch.isfinite(gate_fp32).all()
        & torch.isfinite(decay_fp32).all()
    )
    _fail_closed_assert(finite, "gain-budget inputs must be finite")
    _fail_closed_assert(
        (decay_fp32 <= 0).all(),
        "log_decay must contain actual non-positive GDN2 log decay",
    )

    if mode == "none":
        one = gate_fp32.new_ones(gate_fp32.shape[:-1] + (1,))
        zero = gate_fp32.new_zeros(gate_fp32.shape[:-1] + (1,))
        key2 = key_fp32.square()
        n2 = key2.sum(dim=-1, keepdim=True)
        live = n2 > float(norm_floor)
        n_safe = torch.where(live, n2, torch.ones_like(n2))
        delta = (gate_fp32 * key2).sum(dim=-1, keepdim=True)
        mean = delta / n_safe
        shear2 = n2 * (
            (gate_fp32 - mean).square() * key2
        ).sum(dim=-1, keepdim=True)
        alpha_max = decay_fp32.amax(dim=-1, keepdim=True).exp()
        sigma = rank_one_transition_sigma(delta, shear2)
        infinity = torch.full_like(one, torch.inf)
        return gate_fp32, {
            "n2": n2,
            "delta": delta,
            "effective_delta": delta,
            "mean": mean,
            "shear2": shear2,
            "effective_shear2": shear2,
            "cap": infinity,
            "scale": one,
            "tau_requested": infinity,
            "tau_effective": infinity,
            "minimum_feasible_tau": torch.maximum(
                one,
                (1.0 - delta).abs(),
            ),
            "alpha_max": alpha_max,
            "ideal_step_gain_bound": infinity,
            "original_sigma": sigma,
            "effective_sigma": sigma,
            "original_step_gain_bound": sigma * alpha_max,
            "effective_step_gain_bound": sigma * alpha_max,
            "clipped": zero.bool(),
            "infeasible": zero.bool(),
            "live": live,
            "delta_abs_error": zero,
        }

    key2 = key_fp32.square()
    n2 = key2.sum(dim=-1, keepdim=True)
    live = n2 > float(norm_floor)
    n_safe = torch.where(live, n2, torch.ones_like(n2))

    delta = (gate_fp32 * key2).sum(dim=-1, keepdim=True)
    mean = delta / n_safe
    centered = gate_fp32 - mean
    shear2 = n2 * (centered.square() * key2).sum(dim=-1, keepdim=True)

    alpha_max = decay_fp32.amax(dim=-1, keepdim=True).exp()
    if mode == "fixed_sigma":
        tau_requested = torch.full_like(alpha_max, float(sigma_cap))
    else:
        tau_requested = torch.minimum(
            torch.full_like(alpha_max, float(sigma_cap_max)),
            torch.full_like(alpha_max, float(step_gain_cap))
            / alpha_max.clamp_min(torch.finfo(torch.float32).tiny),
        )

    minimum_feasible_tau = torch.maximum(
        torch.ones_like(delta),
        (1.0 - delta).abs(),
    )
    infeasible = live & (tau_requested < minimum_feasible_tau)
    if infeasible_policy == "raise":
        _fail_closed_assert(
            (~infeasible).all(),
            "requested gain budget is infeasible while preserving erase strength",
        )
        tau_effective = tau_requested
    else:
        tau_effective = torch.maximum(tau_requested, minimum_feasible_tau)

    x = 1.0 - delta
    finite_tau = torch.isfinite(tau_effective)
    cap_finite = (
        (tau_effective - 1.0)
        * (tau_effective + 1.0)
        * (1.0 - (x / tau_effective).square())
    ).clamp_min(0.0)
    cap = torch.where(
        finite_tau,
        cap_finite,
        torch.full_like(cap_finite, torch.inf),
    )

    active = live & (shear2 > cap)
    positive_cap = active & (cap > 0)
    ratio = cap / shear2.clamp_min(torch.finfo(torch.float32).tiny)
    # Clamp away from zero before sqrt. The endpoint branch below still returns
    # exactly zero, while its backward avoids sqrt(0)'s infinite derivative.
    active_scale = ratio.clamp(
        min=torch.finfo(torch.float32).tiny,
        max=1.0,
    ).sqrt()
    scale = torch.where(
        active,
        torch.where(positive_cap, active_scale, torch.zeros_like(active_scale)),
        torch.ones_like(active_scale),
    )
    projected = mean + scale * centered
    effective_gate = torch.where(live, projected, gate_fp32)

    effective_delta = (effective_gate * key2).sum(dim=-1, keepdim=True)
    effective_centered = effective_gate - effective_delta / n_safe
    effective_shear2 = (
        n2
        * (effective_centered.square() * key2).sum(dim=-1, keepdim=True)
    )
    delta_abs_error = (effective_delta - delta).abs()
    original_sigma = rank_one_transition_sigma(delta, shear2)
    effective_sigma = rank_one_transition_sigma(
        effective_delta,
        effective_shear2,
    )
    certificate_tolerance = 2e-5 * tau_effective + 2e-6
    _fail_closed_assert(
        (delta_abs_error <= 5e-6).all(),
        "gain-budget projection failed to preserve erase strength in FP32",
    )
    _fail_closed_assert(
        (effective_sigma <= tau_effective + certificate_tolerance).all(),
        "gain-budget FP32 numerical singular-value certificate failed",
    )

    stats = {
        "n2": n2,
        "delta": delta,
        "effective_delta": effective_delta,
        "delta_abs_error": delta_abs_error,
        "mean": mean,
        "shear2": shear2,
        "effective_shear2": effective_shear2,
        "cap": cap,
        "scale": scale,
        "tau_requested": tau_requested,
        "tau_effective": tau_effective,
        "minimum_feasible_tau": minimum_feasible_tau,
        "alpha_max": alpha_max,
        "ideal_step_gain_bound": tau_effective * alpha_max,
        "original_sigma": original_sigma,
        "effective_sigma": effective_sigma,
        "original_step_gain_bound": original_sigma * alpha_max,
        "effective_step_gain_bound": effective_sigma * alpha_max,
        "clipped": active,
        "infeasible": infeasible,
        "live": live,
    }
    return effective_gate, stats


def gain_budgeted_gdn2_reference(
    query_tk: torch.Tensor,
    key_tk: torch.Tensor,
    value_tv: torch.Tensor,
    log_decay_tk: torch.Tensor,
    erase_gate_tk: torch.Tensor,
    write_gate_tv: torch.Tensor,
    *,
    initial_state_kv: torch.Tensor | None = None,
    scale: float = 1.0,
    budget_mode: BudgetMode = "decay_funded",
    sigma_cap: float = 1.10,
    step_gain_cap: float = 1.0,
    sigma_cap_max: float = 3.0,
    allow_neg_eigval: bool = False,
    qk_already_normalized: bool = False,
    infeasible_policy: InfeasiblePolicy = "raise",
    return_stats: bool = False,
):
    """Single-head token recurrence used as the correctness oracle.

    The attachment's recurrence uses an unscaled read, so ``scale`` defaults
    to 1. Pass ``key_dim ** -0.5`` when comparing outputs to FLA's public ops.
    """
    if key_tk.ndim != 2 or value_tv.ndim != 2:
        raise ValueError("key_tk and value_tv must be rank-2 tensors")
    tokens, key_dim = key_tk.shape
    value_tokens, value_dim = value_tv.shape
    expected_key_shape = (tokens, key_dim)
    expected_value_shape = (tokens, value_dim)
    for name, tensor in (
        ("query_tk", query_tk),
        ("log_decay_tk", log_decay_tk),
        ("erase_gate_tk", erase_gate_tk),
    ):
        if tensor.shape != expected_key_shape:
            raise ValueError(f"{name} must have shape {expected_key_shape}")
    if value_tokens != tokens or write_gate_tv.shape != expected_value_shape:
        raise ValueError("value_tv and write_gate_tv must share shape [T, V]")

    if qk_already_normalized:
        query = query_tk.float()
        key = key_tk.float()
    else:
        query = fla_l2norm_fp32(query_tk)
        key = fla_l2norm_fp32(key_tk)
    value = value_tv.float()
    log_decay = log_decay_tk.float()
    erase_gate = erase_gate_tk.float()
    write_gate = write_gate_tv.float()
    if allow_neg_eigval:
        erase_gate = erase_gate * 2.0

    effective_gate, projection_stats = project_erase_gate(
        key,
        erase_gate,
        log_decay,
        mode=budget_mode,
        sigma_cap=sigma_cap,
        step_gain_cap=step_gain_cap,
        sigma_cap_max=sigma_cap_max,
        infeasible_policy=infeasible_policy,
    )

    if initial_state_kv is None:
        state_kv = torch.zeros(
            key_dim,
            value_dim,
            dtype=torch.float32,
            device=key.device,
        )
    else:
        if initial_state_kv.shape != (key_dim, value_dim):
            raise ValueError(
                f"initial_state_kv must have shape {(key_dim, value_dim)}"
            )
        state_kv = initial_state_kv.float().clone()

    outputs = []
    for token in range(tokens):
        state_kv = log_decay[token].exp()[:, None] * state_kv
        erased_value = (
            effective_gate[token] * key[token]
        ) @ state_kv
        written_value = write_gate[token] * value[token]
        state_kv = state_kv + torch.outer(
            key[token],
            written_value - erased_value,
        )
        outputs.append((query[token] * float(scale)) @ state_kv)

    output_tv = (
        torch.stack(outputs)
        if outputs
        else torch.empty(
            0,
            value_dim,
            dtype=torch.float32,
            device=key.device,
        )
    ).to(value_tv.dtype)
    if return_stats:
        return output_tv, state_kv, projection_stats
    return output_tv, state_kv
