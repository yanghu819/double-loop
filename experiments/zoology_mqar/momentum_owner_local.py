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


@triton.jit(do_not_specialize=["T", "N_BLOCKS"])
def _owner_local_momentum_forward_kernel(
    q,
    k,
    v,
    alpha,
    mu,
    beta,
    eta,
    initial_s,
    initial_m,
    checkpoints_s,
    checkpoints_m,
    output,
    final_s,
    final_m,
    stats,
    T,
    N_BLOCKS,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    CHECKPOINT_T: tl.constexpr,
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
        checkpoint = token // CHECKPOINT_T
        checkpoint_offsets = (
            (program * N_BLOCKS + checkpoint) * K * V
            + offsets_k[:, None] * V
            + offsets_v[None, :]
        )
        checkpoint_mask = mask_state & (token % CHECKPOINT_T == 0)
        tl.store(checkpoints_s + checkpoint_offsets, state_s, mask=checkpoint_mask)
        tl.store(checkpoints_m + checkpoint_offsets, state_m, mask=checkpoint_mask)
        row = (batch * T + token) * H + head
        q_now = tl.load(q + row * K + offsets_k, mask=mask_k, other=0.0).to(tl.float32)
        k_now = tl.load(k + row * K + offsets_k, mask=mask_k, other=0.0).to(tl.float32)
        v_now = tl.load(v + row * V + offsets_v, mask=mask_v, other=0.0).to(tl.float32)
        alpha_now = tl.load(alpha + row).to(tl.float32)
        mu_now = tl.load(mu + row).to(tl.float32)
        beta_now = tl.load(beta + row).to(tl.float32)
        eta_now = tl.load(eta + row).to(tl.float32)

        base_state = alpha_now * state_s
        residual = v_now - tl.sum(k_now[:, None] * base_state, axis=0)
        update = eta_now * residual
        state_m = mu_now * state_m - k_now[:, None] * update[None, :]
        key_energy = tl.sum(k_now * k_now, axis=0)
        owner_velocity = tl.sum(k_now[:, None] * state_m, axis=0) / key_energy
        local_momentum = k_now[:, None] * owner_velocity[None, :]
        state_s = base_state - beta_now * local_momentum
        out_now = tl.sum(q_now[:, None] * state_s, axis=0)
        tl.store(output + row * V + offsets_v, out_now, mask=mask_v)

        base_sq = tl.sum(tl.sum(base_state * base_state, axis=0), axis=0) / (K * V)
        local_sq = tl.sum(tl.sum(local_momentum * local_momentum, axis=0), axis=0) / (K * V)
        momentum_sq = tl.sum(tl.sum(state_m * state_m, axis=0), axis=0) / (K * V)
        discarded = state_m - local_momentum
        discarded_sq = tl.sum(tl.sum(discarded * discarded, axis=0), axis=0) / (K * V)
        tl.store(
            stats + row * 3,
            tl.abs(beta_now) * tl.sqrt(local_sq + 1e-20) / tl.sqrt(base_sq + 1e-20),
        )
        tl.store(
            stats + row * 3 + 1,
            tl.sqrt(local_sq + 1e-20) / tl.sqrt(momentum_sq + 1e-20),
        )
        tl.store(
            stats + row * 3 + 2,
            tl.sqrt(discarded_sq + 1e-20) / tl.sqrt(momentum_sq + 1e-20),
        )

    tl.store(final_s + state_offsets, state_s, mask=mask_state)
    tl.store(final_m + state_offsets, state_m, mask=mask_state)


@triton.jit(do_not_specialize=["T", "N_BLOCKS"])
def _owner_local_momentum_backward_kernel(
    q,
    k,
    v,
    alpha,
    mu,
    beta,
    eta,
    checkpoints_s,
    checkpoints_m,
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
    N_BLOCKS,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    CHECKPOINT_T: tl.constexpr,
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

    adjoint_s = tl.load(grad_final_s + state_offsets, mask=mask_state, other=0.0).to(tl.float32)
    adjoint_m = tl.load(grad_final_m + state_offsets, mask=mask_state, other=0.0).to(tl.float32)

    for reverse_block in range(0, N_BLOCKS):
        block = N_BLOCKS - 1 - reverse_block
        checkpoint_offsets = (
            (program * N_BLOCKS + block) * K * V
            + offsets_k[:, None] * V
            + offsets_v[None, :]
        )
        state_s = tl.load(
            checkpoints_s + checkpoint_offsets, mask=mask_state, other=0.0
        ).to(tl.float32)
        state_m = tl.load(
            checkpoints_m + checkpoint_offsets, mask=mask_state, other=0.0
        ).to(tl.float32)
        block_start = block * CHECKPOINT_T

        # Recompute the exact block endpoint from its stored FP32 anchor.
        for local in tl.static_range(0, CHECKPOINT_T):
            token = block_start + local
            valid = token < T
            row = (batch * T + token) * H + head
            k_now = tl.load(
                k + row * K + offsets_k,
                mask=mask_k & valid,
                other=0.0,
            ).to(tl.float32)
            v_now = tl.load(
                v + row * V + offsets_v,
                mask=mask_v & valid,
                other=0.0,
            ).to(tl.float32)
            alpha_now = tl.load(alpha + row, mask=valid, other=1.0).to(tl.float32)
            mu_now = tl.load(mu + row, mask=valid, other=1.0).to(tl.float32)
            beta_now = tl.load(beta + row, mask=valid, other=0.0).to(tl.float32)
            eta_now = tl.load(eta + row, mask=valid, other=0.0).to(tl.float32)
            base_state = alpha_now * state_s
            residual = v_now - tl.sum(k_now[:, None] * base_state, axis=0)
            update = eta_now * residual
            state_m = mu_now * state_m - k_now[:, None] * update[None, :]
            key_energy = tl.where(valid, tl.sum(k_now * k_now, axis=0), 1.0)
            owner_velocity = tl.sum(k_now[:, None] * state_m, axis=0) / key_energy
            local_momentum = k_now[:, None] * owner_velocity[None, :]
            state_s = base_state - beta_now * local_momentum

        # Reverse only this short block, bounding inverse-decay amplification.
        for reverse_local in tl.static_range(0, CHECKPOINT_T):
            token = block_start + CHECKPOINT_T - 1 - reverse_local
            valid = token < T
            row = (batch * T + token) * H + head
            q_now = tl.load(
                q + row * K + offsets_k,
                mask=mask_k & valid,
                other=0.0,
            ).to(tl.float32)
            k_now = tl.load(
                k + row * K + offsets_k,
                mask=mask_k & valid,
                other=0.0,
            ).to(tl.float32)
            v_now = tl.load(
                v + row * V + offsets_v,
                mask=mask_v & valid,
                other=0.0,
            ).to(tl.float32)
            alpha_now = tl.load(alpha + row, mask=valid, other=1.0).to(tl.float32)
            mu_now = tl.load(mu + row, mask=valid, other=1.0).to(tl.float32)
            beta_now = tl.load(beta + row, mask=valid, other=0.0).to(tl.float32)
            eta_now = tl.load(eta + row, mask=valid, other=0.0).to(tl.float32)

            key_energy = tl.where(valid, tl.sum(k_now * k_now, axis=0), 1.0)
            owner_velocity = tl.sum(k_now[:, None] * state_m, axis=0) / key_energy
            local_momentum = k_now[:, None] * owner_velocity[None, :]
            base_state = state_s + beta_now * local_momentum
            previous_s = base_state / alpha_now
            residual = v_now - tl.sum(k_now[:, None] * base_state, axis=0)
            update = eta_now * residual
            previous_m = (state_m + k_now[:, None] * update[None, :]) / mu_now

            out_grad = tl.load(
                grad_output + row * V + offsets_v,
                mask=mask_v & valid,
                other=0.0,
            ).to(tl.float32)
            q_grad = tl.sum(state_s * out_grad[None, :], axis=1)
            state_grad = adjoint_s + q_now[:, None] * out_grad[None, :]

            base_state_grad = state_grad
            beta_grad = -tl.sum(tl.sum(state_grad * local_momentum, axis=0), axis=0)
            local_grad = -beta_now * state_grad
            key_grad = tl.sum(local_grad * owner_velocity[None, :], axis=1)
            owner_velocity_grad = tl.sum(local_grad * k_now[:, None], axis=0)

            numerator = owner_velocity * key_energy
            key_grad += (
                tl.sum(state_m * owner_velocity_grad[None, :], axis=1) / key_energy
            )
            state_m_grad = adjoint_m + k_now[:, None] * (
                owner_velocity_grad / key_energy
            )[None, :]
            key_energy_grad = -tl.sum(owner_velocity_grad * numerator, axis=0) / (
                key_energy * key_energy
            )
            key_grad += 2.0 * k_now * key_energy_grad

            previous_m_grad = mu_now * state_m_grad
            mu_grad = tl.sum(tl.sum(state_m_grad * previous_m, axis=0), axis=0)
            key_grad += -tl.sum(state_m_grad * update[None, :], axis=1)
            update_grad = -tl.sum(state_m_grad * k_now[:, None], axis=0)

            eta_grad = tl.sum(update_grad * residual, axis=0)
            residual_grad = eta_now * update_grad
            v_grad = residual_grad
            key_grad += -tl.sum(base_state * residual_grad[None, :], axis=1)
            base_state_grad += -k_now[:, None] * residual_grad[None, :]

            previous_s_grad = alpha_now * base_state_grad
            alpha_grad = tl.sum(tl.sum(base_state_grad * previous_s, axis=0), axis=0)

            tl.store(grad_q + row * K + offsets_k, q_grad, mask=mask_k & valid)
            tl.store(grad_k + row * K + offsets_k, key_grad, mask=mask_k & valid)
            tl.store(grad_v + row * V + offsets_v, v_grad, mask=mask_v & valid)
            tl.store(grad_alpha + row, alpha_grad, mask=valid)
            tl.store(grad_mu + row, mu_grad, mask=valid)
            tl.store(grad_beta + row, beta_grad, mask=valid)
            tl.store(grad_eta + row, eta_grad, mask=valid)

            state_s = previous_s
            state_m = previous_m
            adjoint_s = previous_s_grad
            adjoint_m = previous_m_grad

    tl.store(grad_initial_s + state_offsets, adjoint_s, mask=mask_state)
    tl.store(grad_initial_m + state_offsets, adjoint_m, mask=mask_state)


class _OwnerLocalMomentumFunction(torch.autograd.Function):
    checkpoint_tokens = 8

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
            raise ValueError("Owner-Local Momentum requires aligned [B,T,H,K] Q/K")
        if v.shape[:3] != q.shape[:3]:
            raise ValueError("Owner-Local Momentum requires aligned [B,T,H,V]")
        if any(gate.shape != q.shape[:3] for gate in (alpha, mu, beta, eta)):
            raise ValueError("Owner-Local Momentum gates must be [B,T,H]")
        batch, tokens, heads, key_dim = q.shape
        value_dim = v.shape[-1]
        expected_state = (batch, heads, key_dim, value_dim)
        if initial_s.shape != expected_state or initial_m.shape != expected_state:
            raise ValueError(
                f"Owner-Local Momentum state must be {expected_state}, got "
                f"{initial_s.shape}/{initial_m.shape}"
            )
        tensors = (q, k, v, alpha, mu, beta, eta, initial_s, initial_m)
        if any(not tensor.is_cuda for tensor in tensors):
            raise RuntimeError("Owner-Local Momentum is CUDA-only")
        if any(not tensor.is_contiguous() for tensor in tensors):
            raise RuntimeError("Owner-Local Momentum requires contiguous tensors")

        output = torch.empty_like(v)
        final_s = torch.empty(expected_state, device=q.device, dtype=torch.float32)
        final_m = torch.empty_like(final_s)
        checkpoint_tokens = _OwnerLocalMomentumFunction.checkpoint_tokens
        num_checkpoints = math.ceil(tokens / checkpoint_tokens)
        checkpoints_s = torch.empty(
            batch,
            heads,
            num_checkpoints,
            key_dim,
            value_dim,
            device=q.device,
            dtype=torch.float32,
        )
        checkpoints_m = torch.empty_like(checkpoints_s)
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
        _owner_local_momentum_forward_kernel[(batch * heads,)](
            q,
            k,
            v,
            alpha,
            mu,
            beta,
            eta,
            initial_s,
            initial_m,
            checkpoints_s,
            checkpoints_m,
            output,
            final_s,
            final_m,
            stats,
            tokens,
            num_checkpoints,
            H=heads,
            K=key_dim,
            V=value_dim,
            BK=block_k,
            BV=block_v,
            CHECKPOINT_T=checkpoint_tokens,
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
            checkpoints_s,
            checkpoints_m,
        )
        ctx.shape = (
            batch,
            tokens,
            heads,
            key_dim,
            value_dim,
            num_checkpoints,
        )
        ctx.mark_non_differentiable(stats)
        return output, final_s, final_m, stats

    @staticmethod
    def backward(ctx, grad_output, grad_final_s, grad_final_m, _grad_stats):
        q, k, v, alpha, mu, beta, eta, checkpoints_s, checkpoints_m = (
            ctx.saved_tensors
        )
        batch, tokens, heads, key_dim, value_dim, num_checkpoints = ctx.shape
        state_shape = (batch, heads, key_dim, value_dim)
        if grad_output is None:
            grad_output = torch.zeros_like(v)
        if grad_final_s is None:
            grad_final_s = torch.zeros(state_shape, device=q.device, dtype=torch.float32)
        if grad_final_m is None:
            grad_final_m = torch.zeros(state_shape, device=q.device, dtype=torch.float32)
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
        grad_initial_s = torch.empty(state_shape, device=q.device, dtype=torch.float32)
        grad_initial_m = torch.empty_like(grad_initial_s)
        block_k = triton.next_power_of_2(key_dim)
        block_v = triton.next_power_of_2(value_dim)
        _owner_local_momentum_backward_kernel[(batch * heads,)](
            q,
            k,
            v,
            alpha,
            mu,
            beta,
            eta,
            checkpoints_s,
            checkpoints_m,
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
            num_checkpoints,
            H=heads,
            K=key_dim,
            V=value_dim,
            BK=block_k,
            BV=block_v,
            CHECKPOINT_T=_OwnerLocalMomentumFunction.checkpoint_tokens,
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


def owner_local_momentum_rule(
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
        raise RuntimeError("P-GDN3-068 fixes p=k and equal-length sequences")
    if not use_qk_l2norm_in_kernel or not use_p_times_alpha:
        raise RuntimeError("P-GDN3-068 fixes native Q/K normalization and alpha prediction")
    if beta is None or eta is None:
        raise RuntimeError("P-GDN3-068 requires native beta and eta")
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
    output, final_s, final_m, stats = _OwnerLocalMomentumFunction.apply(
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


def owner_local_momentum_reference(
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
        base_state = alpha_now * state_s
        residual = value[:, token] - torch.einsum(
            "bhk,bhkv->bhv", key[:, token], base_state
        )
        update = eta[:, token, :, None] * residual
        state_m = mu_now * state_m - key[:, token, :, :, None] * update[:, :, None, :]
        key_now = key[:, token]
        key_energy = key_now.square().sum(dim=-1, keepdim=True)
        owner_velocity = torch.einsum("bhk,bhkv->bhv", key_now, state_m)
        local_momentum = (
            key_now[:, :, :, None]
            * (owner_velocity / key_energy)[:, :, None, :]
        )
        state_s = base_state - beta_now * local_momentum
        outputs.append(
            torch.einsum("bhk,bhkv->bhv", query[:, token], state_s)
        )
    return torch.stack(outputs, dim=1), torch.stack((state_s, state_m), dim=0)


class OwnerLocalMomentumOperation:
    def __init__(self, *, collect_diagnostics: bool) -> None:
        self.collect_diagnostics = bool(collect_diagnostics)
        self.last_stats: Optional[dict[str, float | int]] = None

    def __call__(self, *args: Any, **kwargs: Any):
        output, final_state, stats = owner_local_momentum_rule(*args, **kwargs)
        if self.collect_diagnostics:
            commit = stats[..., 0]
            local_fraction = stats[..., 1]
            discarded_fraction = stats[..., 2]
            per_board = commit.mean(dim=(1, 2))
            self.last_stats = {
                "local_commit_relative_rms": float(commit.mean().detach()),
                "local_commit_board_std": float(per_board.std(unbiased=False).detach()),
                "projected_momentum_fraction": float(local_fraction.mean().detach()),
                "discarded_momentum_fraction": float(discarded_fraction.mean().detach()),
                "finite": int(torch.isfinite(stats).all().item()),
            }
        return output, final_state


@contextmanager
def scoped_owner_local_momentum(
    *, collect_diagnostics: bool,
) -> Iterator[OwnerLocalMomentumOperation]:
    layer_class = load_external_momentum_layer()
    layer_module = importlib.import_module(layer_class.__module__)
    original_chunk = layer_module.chunk_mode_rule
    original_recurrent = layer_module.fused_recurrent_mode_rule
    operation = OwnerLocalMomentumOperation(
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
            raise RuntimeError("Owner-Local Momentum operation changed inside scope")
        layer_module.chunk_mode_rule = original_chunk
        layer_module.fused_recurrent_mode_rule = original_recurrent


class ZoologyOwnerLocalMomentumFutureSeedMixer(
    ZoologyMomentumDeltaFutureSeedMixer
):
    """Momentum DeltaNet whose velocity commits only in the current owner subspace."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.last_owner_local_stats: Optional[dict[str, float | int]] = None

    def _record(self, operation: OwnerLocalMomentumOperation) -> None:
        if operation.last_stats is None:
            raise RuntimeError("Owner-Local Momentum operation was not called")
        self.last_owner_local_stats = dict(operation.last_stats)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        with scoped_owner_local_momentum(
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
        with scoped_owner_local_momentum(
            collect_diagnostics=not self.training,
        ) as operation:
            output, terminal_state = super().forward_with_state(
                hidden_states,
                initial_state=initial_state,
            )
        if operation.last_stats is not None:
            self._record(operation)
        return output, terminal_state


def owner_local_momentum_diagnostics(model: torch.nn.Module) -> dict[str, Any]:
    base = momentum_futureseed_diagnostics(model)
    mixers = [
        block.sequence_mixer
        for block in model.backbone.layers
        if type(block.sequence_mixer) is ZoologyOwnerLocalMomentumFutureSeedMixer
    ]
    rows = [
        {
            "layer_idx": mixer.layer_idx,
            "owner_local": mixer.last_owner_local_stats,
        }
        for mixer in mixers
    ]
    return {
        **base,
        "owner_local_layers": len(rows),
        "active_owner_local_layers": sum(row["owner_local"] is not None for row in rows),
        "owner_local_per_layer": rows,
        "parameter_delta_vs_momentum": 0,
        "persistent_state_delta_vs_momentum": 0,
        "kernel": "clean_room_fused_checkpointed_owner_local_momentum",
    }
