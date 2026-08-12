"""Bounded receiver-live contrast utilities for the official GDN2 recurrence.

The module is deliberately independent of model and launcher code.  Callers
retain ``v.grad`` from an unchanged official forward/backward pass, use the
FP32 replay below to construct fixed diagnostic signals, and derive one global
discovery direction before applying a norm-bounded holdout intervention.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Literal, Optional

import torch


Tensor = torch.Tensor


@dataclass(frozen=True)
class GDN2Replay:
    """FP32 token-recurrent quantities, all in ``[B,T,H,*]`` layout."""

    raw_output: Tensor
    committed_edit: Tensor
    full_contrast: Tensor
    frozen_only: Tensor
    board_shuffled_frozen_only: Tensor
    final_state: Tensor
    normalized_scaled_query: Tensor
    normalized_key: Tensor
    board_permutation: Tensor


@dataclass(frozen=True)
class DirectionAggregate:
    """A single discovery direction aggregated across the board axis."""

    mean_gradient: Tensor
    descent_direction: Tensor
    board_cosine_to_mean: Tensor
    board_energy_share: Tensor


@dataclass(frozen=True)
class HoldoutDeltaV:
    """A value residual and its verified causal committed-edit budget."""

    delta_v: Tensor
    modified_v: Tensor
    committed_delta: Tensor
    scale: Tensor
    target_rms: Tensor
    achieved_rms: Tensor


@dataclass(frozen=True)
class PairedBootstrap:
    """Deterministic paired-bootstrap summary for a mean improvement."""

    observed_mean: float
    lower: float
    upper: float
    probability_positive: float
    confidence: float
    samples: int
    pairs: int
    seed: int


def _require_shape(name: str, tensor: Tensor, shape: tuple[int, ...]) -> None:
    if tuple(tensor.shape) != shape:
        raise ValueError(f"{name} has shape {tuple(tensor.shape)}, expected {shape}")


def _rms(tensor: Tensor, dims: int | tuple[int, ...], keepdim: bool) -> Tensor:
    return tensor.float().square().mean(dim=dims, keepdim=keepdim).sqrt()


def _kernel_l2_normalize(tensor: Tensor, eps: float) -> Tensor:
    tensor_f = tensor.float()
    return tensor_f * torch.rsqrt(
        tensor_f.square().sum(dim=-1, keepdim=True) + float(eps)
    )


def deterministic_board_permutation(
    board_count: int,
    *,
    seed: int,
    device: Optional[torch.device] = None,
) -> Tensor:
    """Return a seeded derangement for board-shuffled negative controls.

    A seeded ordering followed by one cyclic step is deterministic and has no
    fixed points whenever at least two boards are present.
    """

    if board_count < 2:
        raise ValueError("a board-shuffled control requires at least two boards")
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    order = torch.randperm(board_count, generator=generator)
    permutation = torch.empty_like(order)
    permutation[order] = order.roll(1)
    return permutation.to(device=device)


def replay_gdn2_fp32(
    *,
    q: Tensor,
    k: Tensor,
    v: Tensor,
    g: Tensor,
    b: Tensor,
    w: Tensor,
    initial_state: Optional[Tensor],
    scale: Optional[float] = None,
    use_qk_l2norm_in_kernel: bool = True,
    l2norm_eps: float = 1e-6,
    board_shuffle_seed: int = 0,
) -> GDN2Replay:
    """Replay the official GDN2 baseline recurrence exactly in FP32.

    ``g`` must be the effective log-decay passed to the official operation;
    this helper intentionally does not reproduce ``use_gate_in_kernel``
    preprocessing. ``committed_edit`` is the V-axis residual ``e_t`` in the
    rank-one update ``k_t e_t^T``. ``raw_output`` includes the default
    ``1/sqrt(K)`` query scale and is read after the token's committed edit.
    """

    if q.ndim != 4:
        raise ValueError(f"q must have shape [B,T,H,K], got {tuple(q.shape)}")
    batch, tokens, heads, key_dim = q.shape
    _require_shape("k", k, (batch, tokens, heads, key_dim))
    _require_shape("g", g, (batch, tokens, heads, key_dim))
    _require_shape("b", b, (batch, tokens, heads, key_dim))
    if v.ndim != 4 or tuple(v.shape[:3]) != (batch, tokens, heads):
        raise ValueError(
            "v must have shape [B,T,H,V] with the same B/T/H as q, got "
            f"{tuple(v.shape)}"
        )
    value_dim = v.shape[-1]
    _require_shape("w", w, (batch, tokens, heads, value_dim))
    state_shape = (batch, heads, key_dim, value_dim)
    if initial_state is not None:
        _require_shape("initial_state", initial_state, state_shape)
    if l2norm_eps <= 0:
        raise ValueError("l2norm_eps must be positive")

    query_scale = key_dim**-0.5 if scale is None else float(scale)
    device_type = q.device.type
    with torch.autocast(device_type=device_type, enabled=False):
        q_f = q.float()
        k_f = k.float()
        if use_qk_l2norm_in_kernel:
            q_f = _kernel_l2_normalize(q_f, l2norm_eps)
            k_f = _kernel_l2_normalize(k_f, l2norm_eps)
        q_scaled = q_f * query_scale
        v_f, g_f, b_f, w_f = (
            tensor.float() for tensor in (v, g, b, w)
        )
        if initial_state is None:
            frozen_state = torch.zeros(state_shape, device=q.device, dtype=torch.float32)
        else:
            frozen_state = initial_state.float().clone()
        state = frozen_state.clone()

        raw_outputs: list[Tensor] = []
        committed_edits: list[Tensor] = []
        for token in range(tokens):
            decayed = state * g_f[:, token].exp().unsqueeze(-1)
            erase = torch.einsum(
                "bhk,bhkv->bhv",
                b_f[:, token] * k_f[:, token],
                decayed,
            )
            committed = w_f[:, token] * v_f[:, token] - erase
            state = decayed + torch.einsum(
                "bhk,bhv->bhkv", k_f[:, token], committed
            )
            raw = torch.einsum("bhk,bhkv->bhv", q_scaled[:, token], state)
            committed_edits.append(committed)
            raw_outputs.append(raw)

        raw_output = torch.stack(raw_outputs, dim=1)
        committed_edit = torch.stack(committed_edits, dim=1)
        frozen_only = torch.einsum("bthk,bhkv->bthv", q_scaled, frozen_state)
        permutation = deterministic_board_permutation(
            batch,
            seed=board_shuffle_seed,
            device=q.device,
        )
        shuffled_frozen = torch.einsum(
            "bthk,bhkv->bthv",
            q_scaled,
            frozen_state.index_select(0, permutation),
        )

    return GDN2Replay(
        raw_output=raw_output,
        committed_edit=committed_edit,
        full_contrast=frozen_only - raw_output,
        frozen_only=frozen_only,
        board_shuffled_frozen_only=shuffled_frozen,
        final_state=state,
        normalized_scaled_query=q_scaled,
        normalized_key=k_f,
        board_permutation=permutation,
    )


def rms_match_signal_to_committed_edit(
    signal: Tensor,
    committed_edit: Tensor,
    *,
    eps: float = 1e-12,
) -> Tensor:
    """Match each token/head signal's V-axis RMS to its baseline edit RMS."""

    _require_shape("committed_edit", committed_edit, tuple(signal.shape))
    if signal.ndim != 4:
        raise ValueError("signal must have shape [B,T,H,V]")
    signal_f = signal.detach().float()
    edit_f = committed_edit.detach().float()
    if not torch.isfinite(signal_f).all() or not torch.isfinite(edit_f).all():
        raise ValueError("signal and committed_edit must be finite")
    signal_rms = _rms(signal_f, -1, True)
    edit_rms = _rms(edit_f, -1, True)
    usable = signal_rms > float(eps)
    normalized = signal_f / signal_rms.clamp_min(float(eps))
    return torch.where(usable, normalized * edit_rms, torch.zeros_like(signal_f))


def per_board_virtual_alpha_gradient(
    *,
    v_grad: Tensor,
    signal: Tensor,
    committed_edit: Tensor,
    rho: float = 0.25,
) -> Tensor:
    """Compute ``dL/dalpha`` per board for a shared ``[H,V]`` alpha.

    At zero initialization, ``d(delta_v)/dalpha = rho * d_hat``.  Summing
    ``v.grad * rho * d_hat`` over tokens therefore preserves all downstream
    effects already represented by the unchanged official backward graph.
    """

    _require_shape("signal", signal, tuple(v_grad.shape))
    _require_shape("committed_edit", committed_edit, tuple(v_grad.shape))
    if v_grad.ndim != 4:
        raise ValueError("v_grad must have shape [B,T,H,V]")
    if not math.isfinite(float(rho)):
        raise ValueError("rho must be finite")
    if not torch.isfinite(v_grad).all():
        raise ValueError("v_grad contains non-finite values")
    matched = rms_match_signal_to_committed_edit(signal, committed_edit)
    return (v_grad.detach().float() * (float(rho) * matched)).sum(dim=1)


def discovery_virtual_alpha_gradients(
    *,
    retained_v: Tensor,
    replay: GDN2Replay,
    rho: float = 0.25,
) -> dict[str, Tensor]:
    """Build full-contrast and two sham per-board gradients from ``v.grad``."""

    if retained_v.grad is None:
        raise ValueError("retained_v.grad is missing; call retain_grad() before backward")
    v_grad = retained_v.grad
    return {
        "full_contrast": per_board_virtual_alpha_gradient(
            v_grad=v_grad,
            signal=replay.full_contrast,
            committed_edit=replay.committed_edit,
            rho=rho,
        ),
        "frozen_only": per_board_virtual_alpha_gradient(
            v_grad=v_grad,
            signal=replay.frozen_only,
            committed_edit=replay.committed_edit,
            rho=rho,
        ),
        "board_shuffled_frozen_only": per_board_virtual_alpha_gradient(
            v_grad=v_grad,
            signal=replay.board_shuffled_frozen_only - replay.raw_output,
            committed_edit=replay.committed_edit,
            rho=rho,
        ),
    }


def aggregate_discovery_direction(
    per_board_gradients: Tensor,
    *,
    eps: float = 1e-12,
) -> DirectionAggregate:
    """Aggregate one global descent direction without board/edge selection."""

    if per_board_gradients.ndim < 2 or per_board_gradients.shape[0] < 1:
        raise ValueError("per_board_gradients must have a non-empty board axis")
    gradients = per_board_gradients.detach().float()
    if not torch.isfinite(gradients).all():
        raise ValueError("per_board_gradients contains non-finite values")
    mean_gradient = gradients.mean(dim=0)
    flat_mean = mean_gradient.reshape(-1)
    mean_norm = flat_mean.norm()
    if float(mean_norm) <= float(eps):
        raise ValueError("aggregate discovery gradient is zero")
    descent = -mean_gradient / _rms(mean_gradient, tuple(range(mean_gradient.ndim)), False)

    flat_boards = gradients.reshape(gradients.shape[0], -1)
    board_norms = flat_boards.norm(dim=1)
    cosine = (flat_boards @ flat_mean) / (board_norms * mean_norm).clamp_min(float(eps))
    energy = flat_boards.square().sum(dim=1)
    energy_share = energy / energy.sum().clamp_min(float(eps))
    return DirectionAggregate(
        mean_gradient=mean_gradient,
        descent_direction=descent,
        board_cosine_to_mean=cosine,
        board_energy_share=energy_share,
    )


def scaled_holdout_delta_v(
    *,
    signal: Tensor,
    committed_edit: Tensor,
    w: Tensor,
    direction: Tensor,
    epsilon: float,
    storage_reference: Optional[Tensor] = None,
    eps: float = 1e-12,
) -> HoldoutDeltaV:
    """Create a holdout residual with an exact causal committed RMS budget.

    The scale is computed independently at each current token, using only that
    token's head/value signal, write gate, and committed edit.  The returned
    residual therefore satisfies
    ``RMS_HV(w * delta_v) = epsilon * RMS_HV(committed_edit)`` at every token.
    This preserves the discovered relative strength across heads and implies
    the same equality over each full board without using future tokens to
    normalize earlier interventions.
    Negate ``direction`` to construct the paired opposite intervention.
    """

    if signal.ndim != 4:
        raise ValueError("signal must have shape [B,T,H,V]")
    _require_shape("committed_edit", committed_edit, tuple(signal.shape))
    _require_shape("w", w, tuple(signal.shape))
    if not math.isfinite(float(epsilon)) or epsilon < 0:
        raise ValueError("epsilon must be non-negative; negate direction for -epsilon")
    if not torch.isfinite(signal).all() or not torch.isfinite(committed_edit).all():
        raise ValueError("signal and committed_edit must be finite")
    if not torch.isfinite(w).all() or not torch.isfinite(direction).all():
        raise ValueError("w and direction must be finite")
    if storage_reference is not None:
        _require_shape("storage_reference", storage_reference, tuple(signal.shape))
    try:
        torch.broadcast_shapes(tuple(signal.shape), tuple(direction.shape))
    except RuntimeError as error:
        raise ValueError(
            f"direction shape {tuple(direction.shape)} cannot broadcast to {tuple(signal.shape)}"
        ) from error

    matched = rms_match_signal_to_committed_edit(signal, committed_edit)
    base_delta_v = matched * direction.detach().float()
    committed_base = w.detach().float() * base_delta_v
    target_rms = float(epsilon) * _rms(committed_edit.detach(), (-2, -1), True)
    base_rms = _rms(committed_base, (-2, -1), True)
    impossible = (target_rms > float(eps)) & (base_rms <= float(eps))
    if bool(impossible.any()):
        boards = impossible.flatten().nonzero().flatten().tolist()
        raise ValueError(f"holdout direction has zero committed energy for boards {boards}")
    scale = torch.where(
        target_rms > float(eps),
        target_rms / base_rms.clamp_min(float(eps)),
        torch.zeros_like(target_rms),
    )
    if storage_reference is None:
        delta_v = base_delta_v * scale
        modified_v = delta_v
        committed_delta = w.detach().float() * delta_v
        achieved_rms = _rms(committed_delta, (-2, -1), True)
    else:
        reference = storage_reference.detach()
        reference_f = reference.float()
        best_error = torch.full_like(target_rms, float("inf"))
        best_scale = scale
        best_modified = reference
        best_delta = torch.zeros_like(reference_f)
        best_committed = torch.zeros_like(reference_f)
        best_achieved = torch.zeros_like(target_rms)
        lower_scale = torch.zeros_like(scale)
        upper_scale = scale

        def record_candidate(candidate_scale: Tensor) -> tuple[Tensor, Tensor]:
            nonlocal best_error, best_scale, best_modified
            nonlocal best_delta, best_committed, best_achieved
            modified = (reference_f + base_delta_v * candidate_scale).to(
                reference.dtype
            )
            effective_delta = modified.float() - reference_f
            effective_committed = w.detach().float() * effective_delta
            achieved = _rms(effective_committed, (-2, -1), True)
            error = (achieved - target_rms).abs() / target_rms.clamp_min(float(eps))
            better = error < best_error
            best_error = torch.where(better, error, best_error)
            best_scale = torch.where(better, candidate_scale, best_scale)
            best_modified = torch.where(better, modified, best_modified)
            best_delta = torch.where(better, effective_delta, best_delta)
            best_committed = torch.where(better, effective_committed, best_committed)
            best_achieved = torch.where(better, achieved, best_achieved)
            return achieved, modified

        # BF16 makes the realized committed RMS a monotone step function of
        # the positive scalar scale. First bracket the target independently at
        # every token, then bisect the representable steps while retaining the
        # closest value on either side. This certifies the actual tensor
        # consumed by the official kernel rather than its FP32 precursor.
        record_candidate(lower_scale)
        for _ in range(8):
            achieved, _ = record_candidate(upper_scale)
            below = (target_rms > float(eps)) & (achieved < target_rms)
            lower_scale = torch.where(below, upper_scale, lower_scale)
            upper_scale = torch.where(below, upper_scale * 2.0, upper_scale)
        achieved, _ = record_candidate(upper_scale)
        if bool(((target_rms > float(eps)) & (achieved < target_rms)).any()):
            raise ValueError("could not bracket quantized committed-edit budget")
        for _ in range(14):
            midpoint = (lower_scale + upper_scale) * 0.5
            achieved, _ = record_candidate(midpoint)
            below = achieved < target_rms
            lower_scale = torch.where(below, midpoint, lower_scale)
            upper_scale = torch.where(below, upper_scale, midpoint)
        scale = best_scale
        modified_v = best_modified
        delta_v = best_delta
        committed_delta = best_committed
        achieved_rms = best_achieved
    return HoldoutDeltaV(
        delta_v=delta_v,
        modified_v=modified_v,
        committed_delta=committed_delta,
        scale=scale,
        target_rms=target_rms,
        achieved_rms=achieved_rms,
    )


def paired_bootstrap_interval(
    differences: Tensor | Iterable[float],
    *,
    seed: int,
    samples: int = 10_000,
    confidence: float = 0.95,
    batch_size: int = 512,
) -> PairedBootstrap:
    """Deterministically bootstrap a one-dimensional paired improvement."""

    values = torch.as_tensor(list(differences) if not isinstance(differences, Tensor) else differences)
    values = values.detach().to(device="cpu", dtype=torch.float64).reshape(-1)
    if values.numel() < 2:
        raise ValueError("paired bootstrap requires at least two pairs")
    if not torch.isfinite(values).all():
        raise ValueError("paired differences contain non-finite values")
    if samples < 1 or batch_size < 1:
        raise ValueError("samples and batch_size must be positive")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must lie strictly between zero and one")

    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    means: list[Tensor] = []
    remaining = int(samples)
    while remaining:
        count = min(int(batch_size), remaining)
        indices = torch.randint(
            0,
            values.numel(),
            (count, values.numel()),
            generator=generator,
        )
        means.append(values[indices].mean(dim=1))
        remaining -= count
    bootstrap_means = torch.cat(means)
    tail = (1.0 - float(confidence)) / 2.0
    lower, upper = torch.quantile(
        bootstrap_means,
        torch.tensor([tail, 1.0 - tail], dtype=torch.float64),
    ).tolist()
    return PairedBootstrap(
        observed_mean=float(values.mean().item()),
        lower=float(lower),
        upper=float(upper),
        probability_positive=float((bootstrap_means > 0).double().mean().item()),
        confidence=float(confidence),
        samples=int(samples),
        pairs=int(values.numel()),
        seed=int(seed),
    )


def paired_bootstrap_comparison(
    *,
    control: Tensor | Iterable[float],
    candidate: Tensor | Iterable[float],
    objective: Literal["minimize", "maximize"],
    seed: int,
    samples: int = 10_000,
    confidence: float = 0.95,
    batch_size: int = 512,
) -> PairedBootstrap:
    """Bootstrap candidate improvement while preserving paired board order."""

    control_values = torch.as_tensor(
        list(control) if not isinstance(control, Tensor) else control,
        dtype=torch.float64,
    ).reshape(-1)
    candidate_values = torch.as_tensor(
        list(candidate) if not isinstance(candidate, Tensor) else candidate,
        dtype=torch.float64,
    ).reshape(-1)
    if control_values.shape != candidate_values.shape:
        raise ValueError(
            "control and candidate must contain the same number of paired values"
        )
    if objective == "minimize":
        differences = control_values - candidate_values
    elif objective == "maximize":
        differences = candidate_values - control_values
    else:
        raise ValueError("objective must be 'minimize' or 'maximize'")
    return paired_bootstrap_interval(
        differences,
        seed=seed,
        samples=samples,
        confidence=confidence,
        batch_size=batch_size,
    )


__all__ = [
    "DirectionAggregate",
    "GDN2Replay",
    "HoldoutDeltaV",
    "PairedBootstrap",
    "aggregate_discovery_direction",
    "deterministic_board_permutation",
    "discovery_virtual_alpha_gradients",
    "paired_bootstrap_comparison",
    "paired_bootstrap_interval",
    "per_board_virtual_alpha_gradient",
    "replay_gdn2_fp32",
    "rms_match_signal_to_committed_edit",
    "scaled_holdout_delta_v",
]
