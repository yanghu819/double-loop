from __future__ import annotations

import importlib
import math
from contextlib import contextmanager
from typing import Any, Iterator, Optional

import torch
import torch.nn.functional as F
import triton
import triton.language as tl

from experiments.zoology_mqar.momentum_futureseed import (
    ZoologyMomentumDeltaFutureSeedMixer,
    load_external_momentum_layer,
    momentum_futureseed_diagnostics,
)


@triton.jit(do_not_specialize=["T"])
def _predictive_momentum_forward_kernel(
    q,
    k,
    v,
    alpha,
    mu,
    beta,
    eta,
    initial_s,
    initial_m,
    output,
    final_s,
    final_m,
    stats,
    T,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
):
    program = tl.program_id(0)
    batch = program // H
    head = program % H
    offsets_k = tl.arange(0, BK)
    offsets_v = tl.arange(0, BV)
    mask_k = offsets_k < K
    mask_v = offsets_v < V
    mask_state = mask_k[:, None] & mask_v[None, :]
    state_offsets = program * K * V + offsets_k[:, None] * V + offsets_v[None, :]

    state_s = tl.load(initial_s + state_offsets, mask=mask_state, other=0.0).to(tl.float32)
    state_m = tl.load(initial_m + state_offsets, mask=mask_state, other=0.0).to(tl.float32)

    for token in range(0, T):
        row = (batch * T + token) * H + head
        q_now = tl.load(q + row * K + offsets_k, mask=mask_k, other=0.0).to(tl.float32)
        k_now = tl.load(k + row * K + offsets_k, mask=mask_k, other=0.0).to(tl.float32)
        v_now = tl.load(v + row * V + offsets_v, mask=mask_v, other=0.0).to(tl.float32)
        alpha_now = tl.load(alpha + row).to(tl.float32)
        mu_now = tl.load(mu + row).to(tl.float32)
        beta_now = tl.load(beta + row).to(tl.float32)
        eta_now = tl.load(eta + row).to(tl.float32)

        base_state = alpha_now * state_s
        old_velocity = beta_now * mu_now * state_m
        lookahead = base_state - old_velocity
        parent_residual = v_now - tl.sum(k_now[:, None] * base_state, axis=0)
        residual = v_now - tl.sum(k_now[:, None] * lookahead, axis=0)
        update = eta_now * residual
        state_m = mu_now * state_m - k_now[:, None] * update[None, :]
        state_s = base_state - beta_now * state_m
        out_now = tl.sum(q_now[:, None] * state_s, axis=0)
        tl.store(output + row * V + offsets_v, out_now, mask=mask_v)

        base_sq = tl.sum(tl.sum(base_state * base_state, axis=0), axis=0) / (K * V)
        velocity_sq = tl.sum(tl.sum(old_velocity * old_velocity, axis=0), axis=0) / (K * V)
        parent_residual_sq = tl.sum(parent_residual * parent_residual, axis=0) / V
        residual_delta = residual - parent_residual
        residual_delta_sq = tl.sum(residual_delta * residual_delta, axis=0) / V
        key_energy = tl.sum(k_now * k_now, axis=0)
        denominator = tl.abs(1.0 - beta_now * eta_now * key_energy)
        tl.store(stats + row * 3, tl.sqrt(velocity_sq + 1e-20) / tl.sqrt(base_sq + 1e-20))
        tl.store(stats + row * 3 + 1, tl.sqrt(residual_delta_sq + 1e-20) / tl.sqrt(parent_residual_sq + 1e-20))
        tl.store(stats + row * 3 + 2, denominator)

    tl.store(final_s + state_offsets, state_s, mask=mask_state)
    tl.store(final_m + state_offsets, state_m, mask=mask_state)


@triton.jit(do_not_specialize=["T"])
def _predictive_momentum_backward_kernel(
    q,
    k,
    v,
    alpha,
    mu,
    beta,
    eta,
    final_s,
    final_m,
    grad_output,
    grad_final_s,
    grad_final_m,
    grad_q,
    grad_k,
    grad_v,
    grad_alpha,
    grad_mu,
    grad_beta,
    grad_eta,
    grad_initial_s,
    grad_initial_m,
    T,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
):
    program = tl.program_id(0)
    batch = program // H
    head = program % H
    offsets_k = tl.arange(0, BK)
    offsets_v = tl.arange(0, BV)
    mask_k = offsets_k < K
    mask_v = offsets_v < V
    mask_state = mask_k[:, None] & mask_v[None, :]
    state_offsets = program * K * V + offsets_k[:, None] * V + offsets_v[None, :]

    state_s = tl.load(final_s + state_offsets, mask=mask_state, other=0.0).to(tl.float32)
    state_m = tl.load(final_m + state_offsets, mask=mask_state, other=0.0).to(tl.float32)
    adjoint_s = tl.load(grad_final_s + state_offsets, mask=mask_state, other=0.0).to(tl.float32)
    adjoint_m = tl.load(grad_final_m + state_offsets, mask=mask_state, other=0.0).to(tl.float32)

    for reverse_index in range(0, T):
        token = T - 1 - reverse_index
        row = (batch * T + token) * H + head
        q_now = tl.load(q + row * K + offsets_k, mask=mask_k, other=0.0).to(tl.float32)
        k_now = tl.load(k + row * K + offsets_k, mask=mask_k, other=0.0).to(tl.float32)
        v_now = tl.load(v + row * V + offsets_v, mask=mask_v, other=0.0).to(tl.float32)
        alpha_now = tl.load(alpha + row).to(tl.float32)
        mu_now = tl.load(mu + row).to(tl.float32)
        beta_now = tl.load(beta + row).to(tl.float32)
        eta_now = tl.load(eta + row).to(tl.float32)

        previous_s = (state_s + beta_now * state_m) / alpha_now
        key_energy = tl.sum(k_now * k_now, axis=0)
        denominator = 1.0 - beta_now * eta_now * key_energy
        residual = (v_now - tl.sum(k_now[:, None] * state_s, axis=0)) / denominator
        update = eta_now * residual
        previous_m = (state_m + k_now[:, None] * update[None, :]) / mu_now
        lookahead = alpha_now * previous_s - beta_now * mu_now * previous_m

        out_grad = tl.load(grad_output + row * V + offsets_v, mask=mask_v, other=0.0).to(tl.float32)
        q_grad = tl.sum(state_s * out_grad[None, :], axis=1)
        state_grad = adjoint_s + q_now[:, None] * out_grad[None, :]

        previous_s_grad = alpha_now * state_grad
        alpha_grad = tl.sum(tl.sum(state_grad * previous_s, axis=0), axis=0)
        state_m_grad = adjoint_m - beta_now * state_grad
        beta_grad = -tl.sum(tl.sum(state_grad * state_m, axis=0), axis=0)

        previous_m_grad = mu_now * state_m_grad
        mu_grad = tl.sum(tl.sum(state_m_grad * previous_m, axis=0), axis=0)
        k_grad = -tl.sum(state_m_grad * update[None, :], axis=1)
        update_grad = -tl.sum(state_m_grad * k_now[:, None], axis=0)

        eta_grad = tl.sum(update_grad * residual, axis=0)
        residual_grad = eta_now * update_grad
        v_grad = residual_grad
        k_grad += -tl.sum(lookahead * residual_grad[None, :], axis=1)
        lookahead_grad = -k_now[:, None] * residual_grad[None, :]

        previous_s_grad += alpha_now * lookahead_grad
        alpha_grad += tl.sum(tl.sum(lookahead_grad * previous_s, axis=0), axis=0)
        previous_m_grad += -beta_now * mu_now * lookahead_grad
        inner = tl.sum(tl.sum(lookahead_grad * previous_m, axis=0), axis=0)
        beta_grad += -mu_now * inner
        mu_grad += -beta_now * inner

        tl.store(grad_q + row * K + offsets_k, q_grad, mask=mask_k)
        tl.store(grad_k + row * K + offsets_k, k_grad, mask=mask_k)
        tl.store(grad_v + row * V + offsets_v, v_grad, mask=mask_v)
        tl.store(grad_alpha + row, alpha_grad)
        tl.store(grad_mu + row, mu_grad)
        tl.store(grad_beta + row, beta_grad)
        tl.store(grad_eta + row, eta_grad)

        state_s = previous_s
        state_m = previous_m
        adjoint_s = previous_s_grad
        adjoint_m = previous_m_grad

    tl.store(grad_initial_s + state_offsets, adjoint_s, mask=mask_state)
    tl.store(grad_initial_m + state_offsets, adjoint_m, mask=mask_state)


class _PredictiveMomentumFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        alpha: torch.Tensor,
        mu: torch.Tensor,
        beta: torch.Tensor,
        eta: torch.Tensor,
        initial_s: torch.Tensor,
        initial_m: torch.Tensor,
    ):
        if q.ndim != 4 or q.shape != k.shape:
            raise ValueError("Predictive Momentum requires aligned [B,T,H,K] Q/K")
        if v.shape[:3] != q.shape[:3]:
            raise ValueError("Predictive Momentum requires aligned [B,T,H,V]")
        if any(gate.shape != q.shape[:3] for gate in (alpha, mu, beta, eta)):
            raise ValueError("Predictive Momentum gates must be [B,T,H]")
        batch, tokens, heads, key_dim = q.shape
        value_dim = v.shape[-1]
        expected_state = (batch, heads, key_dim, value_dim)
        if initial_s.shape != expected_state or initial_m.shape != expected_state:
            raise ValueError(
                f"Predictive Momentum state must be {expected_state}, got "
                f"{initial_s.shape}/{initial_m.shape}"
            )
        tensors = (q, k, v, alpha, mu, beta, eta, initial_s, initial_m)
        if any(not tensor.is_cuda for tensor in tensors):
            raise RuntimeError("Predictive Momentum is CUDA-only")
        if any(not tensor.is_contiguous() for tensor in tensors):
            raise RuntimeError("Predictive Momentum requires contiguous tensors")

        output = torch.empty_like(v)
        final_s = torch.empty(expected_state, device=q.device, dtype=torch.float32)
        final_m = torch.empty_like(final_s)
        stats = torch.empty(
            batch,
            tokens,
            heads,
            3,
            device=q.device,
            dtype=torch.float32,
        )
        block_k = triton.next_power_of_2(key_dim)
        block_v = triton.next_power_of_2(value_dim)
        _predictive_momentum_forward_kernel[(batch * heads,)](
            q,
            k,
            v,
            alpha,
            mu,
            beta,
            eta,
            initial_s,
            initial_m,
            output,
            final_s,
            final_m,
            stats,
            tokens,
            H=heads,
            K=key_dim,
            V=value_dim,
            BK=block_k,
            BV=block_v,
            num_warps=4,
        )
        ctx.save_for_backward(
            q,
            k,
            v,
            alpha,
            mu,
            beta,
            eta,
            final_s,
            final_m,
        )
        ctx.shape = (batch, tokens, heads, key_dim, value_dim)
        ctx.mark_non_differentiable(stats)
        return output, final_s, final_m, stats

    @staticmethod
    def backward(ctx, grad_output, grad_final_s, grad_final_m, _grad_stats):
        q, k, v, alpha, mu, beta, eta, final_s, final_m = ctx.saved_tensors
        batch, tokens, heads, key_dim, value_dim = ctx.shape
        if grad_output is None:
            grad_output = torch.zeros_like(v)
        if grad_final_s is None:
            grad_final_s = torch.zeros_like(final_s)
        if grad_final_m is None:
            grad_final_m = torch.zeros_like(final_m)
        grad_output = grad_output.contiguous()
        grad_final_s = grad_final_s.contiguous()
        grad_final_m = grad_final_m.contiguous()

        grad_q = torch.empty_like(q)
        grad_k = torch.empty_like(k)
        grad_v = torch.empty_like(v)
        grad_alpha = torch.empty_like(alpha)
        grad_mu = torch.empty_like(mu)
        grad_beta = torch.empty_like(beta)
        grad_eta = torch.empty_like(eta)
        grad_initial_s = torch.empty_like(final_s)
        grad_initial_m = torch.empty_like(final_m)
        block_k = triton.next_power_of_2(key_dim)
        block_v = triton.next_power_of_2(value_dim)
        _predictive_momentum_backward_kernel[(batch * heads,)](
            q,
            k,
            v,
            alpha,
            mu,
            beta,
            eta,
            final_s,
            final_m,
            grad_output,
            grad_final_s,
            grad_final_m,
            grad_q,
            grad_k,
            grad_v,
            grad_alpha,
            grad_mu,
            grad_beta,
            grad_eta,
            grad_initial_s,
            grad_initial_m,
            tokens,
            H=heads,
            K=key_dim,
            V=value_dim,
            BK=block_k,
            BV=block_v,
            num_warps=4,
        )
        return (
            grad_q,
            grad_k,
            grad_v,
            grad_alpha,
            grad_mu,
            grad_beta,
            grad_eta,
            grad_initial_s,
            grad_initial_m,
        )


def _normalize(vector: torch.Tensor) -> torch.Tensor:
    value = vector.float()
    return value / (value.square().sum(dim=-1, keepdim=True).sqrt() + 1e-6)


def predictive_momentum_rule(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    log_alpha: torch.Tensor,
    log_mu: torch.Tensor,
    p: Optional[torch.Tensor] = None,
    beta: Optional[torch.Tensor] = None,
    eta: Optional[torch.Tensor] = None,
    scale: Optional[float] = None,
    initial_state: Optional[torch.Tensor] = None,
    output_final_state: bool = False,
    cu_seqlens: Optional[torch.LongTensor] = None,
    use_qk_l2norm_in_kernel: bool = True,
    use_p_times_alpha: bool = True,
) -> tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor]:
    if p is not None or cu_seqlens is not None:
        raise RuntimeError("P-GDN3-062 fixes p=k and equal-length sequences")
    if not use_qk_l2norm_in_kernel or not use_p_times_alpha:
        raise RuntimeError("P-GDN3-062 fixes native Q/K normalization and alpha prediction")
    if beta is None or eta is None:
        raise RuntimeError("P-GDN3-062 requires native beta and eta")
    if scale is None:
        scale = k.shape[-1] ** -0.5
    q_normalized = (_normalize(q) * float(scale)).contiguous()
    k_normalized = _normalize(k).contiguous()
    value = v.contiguous()
    alpha = log_alpha.float().exp().contiguous()
    mu = log_mu.float().exp().contiguous()
    beta = beta.float().contiguous()
    eta = eta.float().contiguous()
    batch, _tokens, heads, key_dim = q.shape
    value_dim = v.shape[-1]
    if initial_state is None:
        initial_s = torch.zeros(
            batch,
            heads,
            key_dim,
            value_dim,
            device=q.device,
            dtype=torch.float32,
        )
        initial_m = torch.zeros_like(initial_s)
    else:
        if initial_state.shape != (2, batch, heads, key_dim, value_dim):
            raise ValueError(f"Unexpected initial state {initial_state.shape}")
        initial_s = initial_state[0].contiguous()
        initial_m = initial_state[1].contiguous()
    output, final_s, final_m, stats = _PredictiveMomentumFunction.apply(
        q_normalized,
        k_normalized,
        value,
        alpha,
        mu,
        beta,
        eta,
        initial_s,
        initial_m,
    )
    final_state = torch.stack((final_s, final_m), dim=0) if output_final_state else None
    return output, final_state, stats


def predictive_momentum_reference(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    log_alpha: torch.Tensor,
    log_mu: torch.Tensor,
    beta: torch.Tensor,
    eta: torch.Tensor,
    *,
    scale: Optional[float] = None,
    initial_state: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if scale is None:
        scale = k.shape[-1] ** -0.5
    query = _normalize(q) * float(scale)
    key = _normalize(k)
    value = v.float()
    alpha = log_alpha.float().exp()
    mu = log_mu.float().exp()
    beta = beta.float()
    eta = eta.float()
    batch, tokens, heads, key_dim = q.shape
    value_dim = v.shape[-1]
    if initial_state is None:
        state_s = torch.zeros(
            batch,
            heads,
            key_dim,
            value_dim,
            device=q.device,
            dtype=torch.float32,
        )
        state_m = torch.zeros_like(state_s)
    else:
        state_s, state_m = initial_state[0].float(), initial_state[1].float()
    outputs = []
    for token in range(tokens):
        alpha_now = alpha[:, token, :, None, None]
        mu_now = mu[:, token, :, None, None]
        beta_now = beta[:, token, :, None, None]
        lookahead = alpha_now * state_s - beta_now * mu_now * state_m
        residual = value[:, token] - torch.einsum(
            "bhk,bhkv->bhv", key[:, token], lookahead
        )
        update = eta[:, token, :, None] * residual
        state_m = mu_now * state_m - key[:, token, :, :, None] * update[:, :, None, :]
        state_s = alpha_now * state_s - beta_now * state_m
        outputs.append(
            torch.einsum("bhk,bhkv->bhv", query[:, token], state_s)
        )
    return torch.stack(outputs, dim=1), torch.stack((state_s, state_m), dim=0)


class PredictiveMomentumOperation:
    def __init__(self, *, collect_diagnostics: bool) -> None:
        self.collect_diagnostics = bool(collect_diagnostics)
        self.last_stats: Optional[dict[str, float | int]] = None

    def __call__(self, *args: Any, **kwargs: Any):
        output, final_state, stats = predictive_momentum_rule(*args, **kwargs)
        if self.collect_diagnostics:
            lookahead = stats[..., 0]
            residual = stats[..., 1]
            denominator = stats[..., 2]
            per_board = residual.mean(dim=(1, 2))
            self.last_stats = {
                "lookahead_relative_rms": float(lookahead.mean().detach()),
                "residual_change_relative_rms": float(residual.mean().detach()),
                "residual_change_board_std": float(per_board.std(unbiased=False).detach()),
                "minimum_inverse_denominator": float(denominator.min().detach()),
                "finite": int(torch.isfinite(stats).all().item()),
            }
        return output, final_state


@contextmanager
def scoped_predictive_momentum(
    *, collect_diagnostics: bool,
) -> Iterator[PredictiveMomentumOperation]:
    layer_class = load_external_momentum_layer()
    layer_module = importlib.import_module(layer_class.__module__)
    original_chunk = layer_module.chunk_mode_rule
    original_recurrent = layer_module.fused_recurrent_mode_rule
    operation = PredictiveMomentumOperation(
        collect_diagnostics=collect_diagnostics,
    )
    layer_module.chunk_mode_rule = operation
    layer_module.fused_recurrent_mode_rule = operation
    try:
        yield operation
    finally:
        if (
            layer_module.chunk_mode_rule is not operation
            or layer_module.fused_recurrent_mode_rule is not operation
        ):
            raise RuntimeError("Predictive Momentum operation changed inside scope")
        layer_module.chunk_mode_rule = original_chunk
        layer_module.fused_recurrent_mode_rule = original_recurrent


class ZoologyPredictiveMomentumFutureSeedMixer(
    ZoologyMomentumDeltaFutureSeedMixer
):
    """One-step Momentum whose residual is evaluated at the velocity lookahead."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.last_predictive_stats: Optional[dict[str, float | int]] = None

    def _record(self, operation: PredictiveMomentumOperation) -> None:
        if operation.last_stats is None:
            raise RuntimeError("Predictive Momentum operation was not called")
        self.last_predictive_stats = dict(operation.last_stats)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        with scoped_predictive_momentum(
            collect_diagnostics=not self.training,
        ) as operation:
            output = super().forward(hidden_states)
        if operation.last_stats is not None:
            self._record(operation)
        return output

    def forward_with_state(
        self,
        hidden_states: torch.Tensor,
        *,
        initial_state: Optional[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        with scoped_predictive_momentum(
            collect_diagnostics=not self.training,
        ) as operation:
            output, terminal_state = super().forward_with_state(
                hidden_states,
                initial_state=initial_state,
            )
        if operation.last_stats is not None:
            self._record(operation)
        return output, terminal_state


def predictive_momentum_diagnostics(model: torch.nn.Module) -> dict[str, Any]:
    base = momentum_futureseed_diagnostics(model)
    mixers = [
        block.sequence_mixer
        for block in model.backbone.layers
        if type(block.sequence_mixer) is ZoologyPredictiveMomentumFutureSeedMixer
    ]
    rows = [
        {
            "layer_idx": mixer.layer_idx,
            "predictive": mixer.last_predictive_stats,
        }
        for mixer in mixers
    ]
    return {
        **base,
        "predictive_layers": len(rows),
        "active_predictive_layers": sum(row["predictive"] is not None for row in rows),
        "predictive_per_layer": rows,
        "parameter_delta_vs_momentum": 0,
        "persistent_state_delta_vs_momentum": 0,
        "kernel": "clean_room_fused_reversible_predictive_momentum",
    }
