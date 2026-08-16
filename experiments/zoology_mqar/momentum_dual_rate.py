from __future__ import annotations

import importlib
import math
from contextlib import contextmanager
from typing import Any, Iterator, Optional

import torch
import torch.nn.functional as F
import triton
import triton.language as tl

from experiments.zoology_mqar.gdn2_futureseed import ZoologyGDN2FutureSeedMixer
from experiments.zoology_mqar.momentum_futureseed import (
    ZoologyMomentumDeltaFutureSeedMixer,
    load_external_momentum_layer,
)


STATE_COMPONENTS = 3
CHECKPOINT_TOKENS = 8


@triton.jit(do_not_specialize=["T", "N_BLOCKS"])
def _dual_rate_momentum_forward_kernel(
    q,
    k,
    v,
    alpha,
    mu,
    beta,
    eta,
    mix,
    initial_s,
    initial_slow,
    initial_fast,
    checkpoints_s,
    checkpoints_slow,
    checkpoints_fast,
    output,
    final_s,
    final_slow,
    final_fast,
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
    state_slow = tl.load(initial_slow + state_offsets, mask=mask_state, other=0.0).to(tl.float32)
    state_fast = tl.load(initial_fast + state_offsets, mask=mask_state, other=0.0).to(tl.float32)

    for token in range(0, T):
        checkpoint = token // CHECKPOINT_T
        checkpoint_offsets = (
            (program * N_BLOCKS + checkpoint) * K * V
            + offsets_k[:, None] * V
            + offsets_v[None, :]
        )
        checkpoint_mask = mask_state & (token % CHECKPOINT_T == 0)
        tl.store(checkpoints_s + checkpoint_offsets, state_s, mask=checkpoint_mask)
        tl.store(checkpoints_slow + checkpoint_offsets, state_slow, mask=checkpoint_mask)
        tl.store(checkpoints_fast + checkpoint_offsets, state_fast, mask=checkpoint_mask)
        row = (batch * T + token) * H + head
        q_now = tl.load(q + row * K + offsets_k, mask=mask_k, other=0.0).to(tl.float32)
        k_now = tl.load(k + row * K + offsets_k, mask=mask_k, other=0.0).to(tl.float32)
        v_now = tl.load(v + row * V + offsets_v, mask=mask_v, other=0.0).to(tl.float32)
        alpha_now = tl.load(alpha + row).to(tl.float32)
        mu_now = tl.load(mu + row).to(tl.float32)
        beta_now = tl.load(beta + row).to(tl.float32)
        eta_now = tl.load(eta + row).to(tl.float32)
        mix_now = tl.load(mix + row).to(tl.float32)

        base_state = alpha_now * state_s
        residual = v_now - tl.sum(k_now[:, None] * base_state, axis=0)
        update = eta_now * residual
        edit = k_now[:, None] * update[None, :]
        state_slow = mu_now * state_slow - edit
        state_fast = mu_now * mu_now * state_fast - edit
        band = state_fast - state_slow
        blended = state_slow + mix_now * band
        state_s = base_state - beta_now * blended
        out_now = tl.sum(q_now[:, None] * state_s, axis=0)
        tl.store(output + row * V + offsets_v, out_now, mask=mask_v)

        base_sq = tl.sum(tl.sum(base_state * base_state, axis=0), axis=0) / (K * V)
        slow_sq = tl.sum(tl.sum(state_slow * state_slow, axis=0), axis=0) / (K * V)
        band_sq = tl.sum(tl.sum(band * band, axis=0), axis=0) / (K * V)
        mixed_sq = tl.sum(tl.sum((mix_now * band) * (mix_now * band), axis=0), axis=0) / (K * V)
        tl.store(
            stats + row * 3,
            tl.sqrt(band_sq + 1e-20) / tl.sqrt(slow_sq + 1e-20),
        )
        tl.store(
            stats + row * 3 + 1,
            tl.sqrt(mixed_sq + 1e-20) / tl.sqrt(base_sq + 1e-20),
        )
        tl.store(
            stats + row * 3 + 2,
            tl.abs(mix_now),
        )

    tl.store(final_s + state_offsets, state_s, mask=mask_state)
    tl.store(final_slow + state_offsets, state_slow, mask=mask_state)
    tl.store(final_fast + state_offsets, state_fast, mask=mask_state)


@triton.jit(do_not_specialize=["T", "N_BLOCKS"])
def _dual_rate_momentum_backward_kernel(
    q,
    k,
    v,
    alpha,
    mu,
    beta,
    eta,
    mix,
    checkpoints_s,
    checkpoints_slow,
    checkpoints_fast,
    grad_output,
    grad_final_s,
    grad_final_slow,
    grad_final_fast,
    grad_q,
    grad_k,
    grad_v,
    grad_alpha,
    grad_mu,
    grad_beta,
    grad_eta,
    grad_mix,
    grad_initial_s,
    grad_initial_slow,
    grad_initial_fast,
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
    adjoint_slow = tl.load(grad_final_slow + state_offsets, mask=mask_state, other=0.0).to(tl.float32)
    adjoint_fast = tl.load(grad_final_fast + state_offsets, mask=mask_state, other=0.0).to(tl.float32)

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
        state_slow = tl.load(
            checkpoints_slow + checkpoint_offsets, mask=mask_state, other=0.0
        ).to(tl.float32)
        state_fast = tl.load(
            checkpoints_fast + checkpoint_offsets, mask=mask_state, other=0.0
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
            mix_now = tl.load(mix + row, mask=valid, other=0.0).to(tl.float32)
            base_state = alpha_now * state_s
            residual = v_now - tl.sum(k_now[:, None] * base_state, axis=0)
            update = eta_now * residual
            edit = k_now[:, None] * update[None, :]
            state_slow = mu_now * state_slow - edit
            state_fast = mu_now * mu_now * state_fast - edit
            blended = state_slow + mix_now * (state_fast - state_slow)
            state_s = base_state - beta_now * blended

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
            mix_now = tl.load(mix + row, mask=valid, other=0.0).to(tl.float32)

            band = state_fast - state_slow
            blended = state_slow + mix_now * band
            base_state = state_s + beta_now * blended
            previous_s = base_state / alpha_now
            residual = v_now - tl.sum(k_now[:, None] * base_state, axis=0)
            update = eta_now * residual
            edit = k_now[:, None] * update[None, :]
            previous_slow = (state_slow + edit) / mu_now
            previous_fast = (state_fast + edit) / (mu_now * mu_now)

            out_grad = tl.load(
                grad_output + row * V + offsets_v,
                mask=mask_v & valid,
                other=0.0,
            ).to(tl.float32)
            q_grad = tl.sum(state_s * out_grad[None, :], axis=1)
            state_grad = adjoint_s + q_now[:, None] * out_grad[None, :]

            base_state_grad = state_grad
            beta_grad = -tl.sum(tl.sum(state_grad * blended, axis=0), axis=0)
            blended_grad = -beta_now * state_grad
            mix_grad = tl.sum(tl.sum(blended_grad * band, axis=0), axis=0)
            state_slow_grad = adjoint_slow + (1.0 - mix_now) * blended_grad
            state_fast_grad = adjoint_fast + mix_now * blended_grad

            previous_slow_grad = mu_now * state_slow_grad
            previous_fast_grad = mu_now * mu_now * state_fast_grad
            mu_grad = tl.sum(
                tl.sum(state_slow_grad * previous_slow, axis=0), axis=0
            ) + 2.0 * mu_now * tl.sum(
                tl.sum(state_fast_grad * previous_fast, axis=0), axis=0
            )
            edit_grad = -state_slow_grad - state_fast_grad
            key_grad = tl.sum(edit_grad * update[None, :], axis=1)
            update_grad = tl.sum(edit_grad * k_now[:, None], axis=0)

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
            tl.store(grad_mix + row, mix_grad, mask=valid)

            state_s = previous_s
            state_slow = previous_slow
            state_fast = previous_fast
            adjoint_s = previous_s_grad
            adjoint_slow = previous_slow_grad
            adjoint_fast = previous_fast_grad

    tl.store(grad_initial_s + state_offsets, adjoint_s, mask=mask_state)
    tl.store(grad_initial_slow + state_offsets, adjoint_slow, mask=mask_state)
    tl.store(grad_initial_fast + state_offsets, adjoint_fast, mask=mask_state)


class _DualRateMomentumFunction(torch.autograd.Function):
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
        mix: torch.Tensor,
        initial_s: torch.Tensor,
        initial_slow: torch.Tensor,
        initial_fast: torch.Tensor,
    ):
        if q.ndim != 4 or q.shape != k.shape:
            raise ValueError("Dual-Rate Momentum requires aligned [B,T,H,K] Q/K")
        if v.shape[:3] != q.shape[:3]:
            raise ValueError("Dual-Rate Momentum requires aligned [B,T,H,V]")
        if any(gate.shape != q.shape[:3] for gate in (alpha, mu, beta, eta, mix)):
            raise ValueError("Dual-Rate Momentum gates must be [B,T,H]")
        batch, tokens, heads, key_dim = q.shape
        value_dim = v.shape[-1]
        expected_state = (batch, heads, key_dim, value_dim)
        if any(
            state.shape != expected_state
            for state in (initial_s, initial_slow, initial_fast)
        ):
            raise ValueError(
                f"Dual-Rate Momentum state must be {expected_state}, got "
                f"{initial_s.shape}/{initial_slow.shape}/{initial_fast.shape}"
            )
        tensors = (
            q, k, v, alpha, mu, beta, eta, mix,
            initial_s, initial_slow, initial_fast,
        )
        if any(not tensor.is_cuda for tensor in tensors):
            raise RuntimeError("Dual-Rate Momentum is CUDA-only")
        if any(not tensor.is_contiguous() for tensor in tensors):
            raise RuntimeError("Dual-Rate Momentum requires contiguous tensors")

        output = torch.empty_like(v)
        final_s = torch.empty(expected_state, device=q.device, dtype=torch.float32)
        final_slow = torch.empty_like(final_s)
        final_fast = torch.empty_like(final_s)
        checkpoint_tokens = _DualRateMomentumFunction.checkpoint_tokens
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
        checkpoints_slow = torch.empty_like(checkpoints_s)
        checkpoints_fast = torch.empty_like(checkpoints_s)
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
        _dual_rate_momentum_forward_kernel[(batch * heads,)](
            q,
            k,
            v,
            alpha,
            mu,
            beta,
            eta,
            mix,
            initial_s,
            initial_slow,
            initial_fast,
            checkpoints_s,
            checkpoints_slow,
            checkpoints_fast,
            output,
            final_s,
            final_slow,
            final_fast,
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
            mix,
            checkpoints_s,
            checkpoints_slow,
            checkpoints_fast,
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
        return output, final_s, final_slow, final_fast, stats

    @staticmethod
    def backward(
        ctx,
        grad_output,
        grad_final_s,
        grad_final_slow,
        grad_final_fast,
        _grad_stats,
    ):
        (
            q, k, v, alpha, mu, beta, eta, mix,
            checkpoints_s, checkpoints_slow, checkpoints_fast,
        ) = ctx.saved_tensors
        batch, tokens, heads, key_dim, value_dim, num_checkpoints = ctx.shape
        state_shape = (batch, heads, key_dim, value_dim)
        if grad_output is None:
            grad_output = torch.zeros_like(v)
        if grad_final_s is None:
            grad_final_s = torch.zeros(state_shape, device=q.device, dtype=torch.float32)
        if grad_final_slow is None:
            grad_final_slow = torch.zeros(state_shape, device=q.device, dtype=torch.float32)
        if grad_final_fast is None:
            grad_final_fast = torch.zeros(state_shape, device=q.device, dtype=torch.float32)
        grad_output = grad_output.contiguous()
        grad_final_s = grad_final_s.contiguous()
        grad_final_slow = grad_final_slow.contiguous()
        grad_final_fast = grad_final_fast.contiguous()

        grad_q = torch.empty_like(q)
        grad_k = torch.empty_like(k)
        grad_v = torch.empty_like(v)
        grad_alpha = torch.empty_like(alpha)
        grad_mu = torch.empty_like(mu)
        grad_beta = torch.empty_like(beta)
        grad_eta = torch.empty_like(eta)
        grad_mix = torch.empty_like(mix)
        grad_initial_s = torch.empty(state_shape, device=q.device, dtype=torch.float32)
        grad_initial_slow = torch.empty_like(grad_initial_s)
        grad_initial_fast = torch.empty_like(grad_initial_s)
        block_k = triton.next_power_of_2(key_dim)
        block_v = triton.next_power_of_2(value_dim)
        _dual_rate_momentum_backward_kernel[(batch * heads,)](
            q,
            k,
            v,
            alpha,
            mu,
            beta,
            eta,
            mix,
            checkpoints_s,
            checkpoints_slow,
            checkpoints_fast,
            grad_output,
            grad_final_s,
            grad_final_slow,
            grad_final_fast,
            grad_q,
            grad_k,
            grad_v,
            grad_alpha,
            grad_mu,
            grad_beta,
            grad_eta,
            grad_mix,
            grad_initial_s,
            grad_initial_slow,
            grad_initial_fast,
            tokens,
            num_checkpoints,
            H=heads,
            K=key_dim,
            V=value_dim,
            BK=block_k,
            BV=block_v,
            CHECKPOINT_T=_DualRateMomentumFunction.checkpoint_tokens,
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
            grad_mix,
            grad_initial_s,
            grad_initial_slow,
            grad_initial_fast,
        )


def _normalize(vector: torch.Tensor) -> torch.Tensor:
    value = vector.float()
    return value / (value.square().sum(dim=-1, keepdim=True).sqrt() + 1e-6)


def dual_rate_momentum_rule(
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
    *,
    dual_rate_mix: torch.Tensor,
) -> tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor]:
    if p is not None or cu_seqlens is not None:
        raise RuntimeError("P-GDN3-070 fixes p=k and equal-length sequences")
    if not use_qk_l2norm_in_kernel or not use_p_times_alpha:
        raise RuntimeError("P-GDN3-070 fixes native Q/K normalization and alpha prediction")
    if beta is None or eta is None:
        raise RuntimeError("P-GDN3-070 requires native beta and eta")
    if scale is None:
        scale = k.shape[-1] ** -0.5
    q_normalized = (_normalize(q) * float(scale)).contiguous()
    k_normalized = _normalize(k).contiguous()
    value = v.contiguous()
    alpha = log_alpha.float().exp().contiguous()
    mu = log_mu.float().exp().contiguous()
    beta = beta.float().contiguous()
    eta = eta.float().contiguous()
    mix = dual_rate_mix.float().contiguous()
    batch, _tokens, heads, key_dim = q.shape
    value_dim = v.shape[-1]
    if mix.shape != q.shape[:3]:
        raise ValueError(f"Dual-rate mix must be {q.shape[:3]}, got {mix.shape}")
    if initial_state is None:
        initial_s = torch.zeros(
            batch,
            heads,
            key_dim,
            value_dim,
            device=q.device,
            dtype=torch.float32,
        )
        initial_slow = torch.zeros_like(initial_s)
        initial_fast = torch.zeros_like(initial_s)
    else:
        if initial_state.shape != (3, batch, heads, key_dim, value_dim):
            raise ValueError(f"Unexpected initial state {initial_state.shape}")
        initial_s = initial_state[0].contiguous()
        initial_slow = initial_state[1].contiguous()
        initial_fast = initial_state[2].contiguous()
    output, final_s, final_slow, final_fast, stats = _DualRateMomentumFunction.apply(
        q_normalized,
        k_normalized,
        value,
        alpha,
        mu,
        beta,
        eta,
        mix,
        initial_s,
        initial_slow,
        initial_fast,
    )
    final_state = (
        torch.stack((final_s, final_slow, final_fast), dim=0)
        if output_final_state
        else None
    )
    return output, final_state, stats


def dual_rate_momentum_reference(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    log_alpha: torch.Tensor,
    log_mu: torch.Tensor,
    beta: torch.Tensor,
    eta: torch.Tensor,
    dual_rate_mix: torch.Tensor,
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
    mix = dual_rate_mix.float()
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
        state_slow = torch.zeros_like(state_s)
        state_fast = torch.zeros_like(state_s)
    else:
        state_s, state_slow, state_fast = (
            initial_state[0].float(),
            initial_state[1].float(),
            initial_state[2].float(),
        )
    outputs = []
    for token in range(tokens):
        alpha_now = alpha[:, token, :, None, None]
        mu_now = mu[:, token, :, None, None]
        beta_now = beta[:, token, :, None, None]
        mix_now = mix[:, token, :, None, None]
        base_state = alpha_now * state_s
        residual = value[:, token] - torch.einsum(
            "bhk,bhkv->bhv", key[:, token], base_state
        )
        update = eta[:, token, :, None] * residual
        edit = key[:, token, :, :, None] * update[:, :, None, :]
        state_slow = mu_now * state_slow - edit
        state_fast = mu_now.square() * state_fast - edit
        blended = state_slow + mix_now * (state_fast - state_slow)
        state_s = base_state - beta_now * blended
        outputs.append(
            torch.einsum("bhk,bhkv->bhv", query[:, token], state_s)
        )
    return torch.stack(outputs, dim=1), torch.stack(
        (state_s, state_slow, state_fast), dim=0
    )


class DualRateMomentumOperation:
    def __init__(
        self,
        *,
        dual_rate_mix: torch.Tensor,
        collect_diagnostics: bool,
    ) -> None:
        self.dual_rate_mix = dual_rate_mix
        self.collect_diagnostics = bool(collect_diagnostics)
        self.last_stats: Optional[dict[str, float | int]] = None

    def __call__(self, *args: Any, **kwargs: Any):
        output, final_state, stats = dual_rate_momentum_rule(
            *args,
            **kwargs,
            dual_rate_mix=self.dual_rate_mix,
        )
        if self.collect_diagnostics:
            band = stats[..., 0]
            injection = stats[..., 1]
            mix = stats[..., 2]
            per_board = injection.mean(dim=(1, 2))
            per_token = injection.mean(dim=(0, 2))
            self.last_stats = {
                "band_to_slow_relative_rms": float(band.mean().detach()),
                "injection_relative_rms": float(injection.mean().detach()),
                "injection_board_std": float(per_board.std(unbiased=False).detach()),
                "injection_token_std": float(per_token.std(unbiased=False).detach()),
                "mean_abs_mix": float(mix.mean().detach()),
                "max_abs_mix": float(mix.max().detach()),
                "finite": int(torch.isfinite(stats).all().item()),
            }
        return output, final_state


@contextmanager
def scoped_dual_rate_momentum(
    *, dual_rate_mix: torch.Tensor, collect_diagnostics: bool,
) -> Iterator[DualRateMomentumOperation]:
    layer_class = load_external_momentum_layer()
    layer_module = importlib.import_module(layer_class.__module__)
    original_chunk = layer_module.chunk_mode_rule
    original_recurrent = layer_module.fused_recurrent_mode_rule
    operation = DualRateMomentumOperation(
        dual_rate_mix=dual_rate_mix,
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
            raise RuntimeError("Dual-Rate Momentum operation changed inside scope")
        layer_module.chunk_mode_rule = original_chunk
        layer_module.fused_recurrent_mode_rule = original_recurrent


class ZoologyDualRateMomentumFutureSeedMixer(
    ZoologyMomentumDeltaFutureSeedMixer
):
    """Coupled slow/fast Momentum lanes with a learned token-wise band-pass read."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.dual_rate_weight = torch.nn.Parameter(
            torch.zeros(self.layer.num_v_heads, self.layer.hidden_size)
        )
        self.last_dual_rate_stats: Optional[dict[str, float | int]] = None
        self.last_fast_momentum_rms: Optional[torch.Tensor] = None

    def _mix(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return torch.tanh(
            F.linear(hidden_states, self.dual_rate_weight).float()
        ).contiguous()

    def _record(self, operation: DualRateMomentumOperation) -> None:
        if operation.last_stats is None:
            raise RuntimeError("Dual-Rate Momentum operation was not called")
        self.last_dual_rate_stats = dict(operation.last_stats)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        with scoped_dual_rate_momentum(
            dual_rate_mix=self._mix(hidden_states),
            collect_diagnostics=not self.training,
        ) as operation:
            output = ZoologyGDN2FutureSeedMixer.forward(self, hidden_states)
        if operation.last_stats is not None:
            self._record(operation)
        return output

    def forward_with_state(
        self,
        hidden_states: torch.Tensor,
        *,
        initial_state: Optional[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        with scoped_dual_rate_momentum(
            dual_rate_mix=self._mix(hidden_states),
            collect_diagnostics=not self.training,
        ) as operation:
            output, terminal_state = ZoologyGDN2FutureSeedMixer.forward_with_state(
                self,
                hidden_states,
                initial_state=initial_state,
            )
        if terminal_state.ndim != 5 or terminal_state.shape[0] != STATE_COMPONENTS:
            raise RuntimeError(
                f"Dual-rate state must be [3,B,H,K,V], got {terminal_state.shape}"
            )
        state_rms = terminal_state[0].float().square().mean().sqrt()
        slow_rms = terminal_state[1].float().square().mean().sqrt()
        fast_rms = terminal_state[2].float().square().mean().sqrt()
        self.last_state_rms = state_rms.detach()
        self.last_momentum_rms = slow_rms.detach()
        self.last_fast_momentum_rms = fast_rms.detach()
        self.last_momentum_to_state_rms = (
            slow_rms / state_rms.clamp_min(1e-8)
        ).detach()
        if operation.last_stats is not None:
            self._record(operation)
        return output, terminal_state

    def state_size(self, sequence_length: int = 2048) -> int:
        del sequence_length
        return (
            STATE_COMPONENTS
            * self.layer.num_v_heads
            * self.layer.head_k_dim
            * self.layer.head_v_dim
        )


def dual_rate_momentum_diagnostics(model: torch.nn.Module) -> dict[str, Any]:
    mixers = [
        block.sequence_mixer
        for block in model.backbone.layers
        if type(block.sequence_mixer) is ZoologyDualRateMomentumFutureSeedMixer
    ]
    rows = []
    for mixer in mixers:
        rows.append({
            "layer_idx": mixer.layer_idx,
            "dual_rate": mixer.last_dual_rate_stats,
            "state_rms": None if mixer.last_state_rms is None else float(mixer.last_state_rms),
            "slow_momentum_rms": (
                None if mixer.last_momentum_rms is None else float(mixer.last_momentum_rms)
            ),
            "fast_momentum_rms": (
                None
                if mixer.last_fast_momentum_rms is None
                else float(mixer.last_fast_momentum_rms)
            ),
            "slow_to_state_rms": (
                None
                if mixer.last_momentum_to_state_rms is None
                else float(mixer.last_momentum_to_state_rms)
            ),
            "seed_applied": mixer.last_seed_gate is not None,
            "seed_gate": None if mixer.last_seed_gate is None else float(mixer.last_seed_gate),
        })
    state_values = mixers[0].state_size() if mixers else 0
    return {
        "external_sha": "c6e77fa261fb0c002fae1a14b6209a5b28d2edc9",
        "state_components": STATE_COMPONENTS,
        "state_values_per_layer": state_values,
        "active_layers": sum(row["state_rms"] is not None for row in rows),
        "active_futureseed_routes": sum(row["seed_applied"] for row in rows),
        "per_layer": rows,
        "matched_parent": getattr(model, "_momentum_parent_metadata", None),
        "dual_rate_layers": len(rows),
        "active_dual_rate_layers": sum(row["dual_rate"] is not None for row in rows),
        "dual_rate_per_layer": rows,
        "parameter_delta_vs_momentum": sum(
            mixer.dual_rate_weight.numel() for mixer in mixers
        ),
        "persistent_state_delta_vs_momentum": (
            mixers[0].layer.num_v_heads
            * mixers[0].layer.head_k_dim
            * mixers[0].layer.head_v_dim
            if mixers
            else 0
        ),
        "kernel": "clean_room_fused_checkpointed_dual_rate_momentum",
    }
