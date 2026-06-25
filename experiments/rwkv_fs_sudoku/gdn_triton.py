from __future__ import annotations

from typing import Optional, Tuple

import torch
import triton
import triton.language as tl


@triton.jit
def _gdn_recurrent_fwd_kernel(
    q,
    k,
    v,
    g,
    beta,
    h0,
    o,
    ht,
    scale: tl.constexpr,
    T: tl.constexpr,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    HAS_INITIAL: tl.constexpr,
):
    pid_v = tl.program_id(0)
    pid_bh = tl.program_id(1)
    b = pid_bh // H
    h = pid_bh - b * H

    offs_k = tl.arange(0, BK)
    offs_v = pid_v * BV + tl.arange(0, BV)
    mask_k = offs_k < K
    mask_v = offs_v < V
    mask_state = mask_v[:, None] & mask_k[None, :]

    state = tl.zeros((BV, BK), dtype=tl.float32)
    if HAS_INITIAL:
        h0_ptrs = h0 + ((b * H + h) * V + offs_v[:, None]) * K + offs_k[None, :]
        state += tl.load(h0_ptrs, mask=mask_state, other=0.0).to(tl.float32)

    q_base = q + ((b * T) * H + h) * K
    k_base = k + ((b * T) * H + h) * K
    v_base = v + ((b * T) * H + h) * V
    g_base = g + (b * T) * H + h
    beta_base = beta + (b * T) * H + h
    o_base = o + ((b * T) * H + h) * V

    for t in tl.range(0, T):
        q_t = tl.load(q_base + t * H * K + offs_k, mask=mask_k, other=0.0).to(tl.float32)
        k_t = tl.load(k_base + t * H * K + offs_k, mask=mask_k, other=0.0).to(tl.float32)
        v_t = tl.load(v_base + t * H * V + offs_v, mask=mask_v, other=0.0).to(tl.float32)
        decay_t = tl.exp(tl.load(g_base + t * H).to(tl.float32))
        beta_t = tl.load(beta_base + t * H).to(tl.float32)

        state = state * decay_t
        pred = tl.sum(state * k_t[None, :], axis=1)
        delta_v = beta_t * (v_t - pred)
        state = state + delta_v[:, None] * k_t[None, :]
        out_t = tl.sum(state * (q_t[None, :] * scale), axis=1)
        tl.store(o_base + t * H * V + offs_v, out_t, mask=mask_v)

    ht_ptrs = ht + ((b * H + h) * V + offs_v[:, None]) * K + offs_k[None, :]
    tl.store(ht_ptrs, state, mask=mask_state)


def _gdn_recurrent_torch(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    initial_state: Optional[torch.Tensor],
    *,
    scale: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    batch, seq_len, heads, key_dim = q.shape
    value_dim = v.shape[-1]
    if initial_state is None:
        state = q.new_zeros(batch, heads, value_dim, key_dim)
    else:
        state = initial_state.to(torch.float32)
    outs = []
    for t in range(seq_len):
        q_t = q[:, t].to(torch.float32) * scale
        k_t = k[:, t].to(torch.float32)
        v_t = v[:, t].to(torch.float32)
        decay_t = g[:, t].to(torch.float32).exp().unsqueeze(-1).unsqueeze(-1)
        beta_t = beta[:, t].to(torch.float32).unsqueeze(-1)
        state = state * decay_t
        pred = (state * k_t.unsqueeze(-2)).sum(dim=-1)
        delta_v = beta_t * (v_t - pred)
        state = state + delta_v.unsqueeze(-1) * k_t.unsqueeze(-2)
        outs.append((state * q_t.unsqueeze(-2)).sum(dim=-1))
    return torch.stack(outs, dim=1), state


class _GDNRecurrentTritonFn(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor,
        beta: torch.Tensor,
        initial_state: Optional[torch.Tensor],
        scale: float,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if not q.is_cuda:
            raise RuntimeError("gdn_triton_recurrent is CUDA-only.")
        q = q.contiguous()
        k = k.contiguous()
        v = v.contiguous()
        g = g.contiguous()
        beta = beta.contiguous()
        has_initial = initial_state is not None
        if has_initial:
            initial_state = initial_state.contiguous()

        batch, seq_len, heads, key_dim = q.shape
        value_dim = v.shape[-1]
        if k.shape != q.shape:
            raise ValueError(f"k shape {tuple(k.shape)} must match q shape {tuple(q.shape)}")
        if v.shape[:3] != (batch, seq_len, heads):
            raise ValueError(f"v shape {tuple(v.shape)} must start with {(batch, seq_len, heads)}")
        if g.shape != (batch, seq_len, heads):
            raise ValueError(f"g shape {tuple(g.shape)} must be {(batch, seq_len, heads)}")
        if beta.shape != (batch, seq_len, heads):
            raise ValueError(f"beta shape {tuple(beta.shape)} must be {(batch, seq_len, heads)}")
        if has_initial and tuple(initial_state.shape) != (batch, heads, value_dim, key_dim):
            raise ValueError(
                "initial_state must use FutureSeed/GDN external layout "
                f"(B,H,V,K)={(batch, heads, value_dim, key_dim)}, got {tuple(initial_state.shape)}"
            )

        o = torch.empty((batch, seq_len, heads, value_dim), device=q.device, dtype=torch.float32)
        ht = torch.empty((batch, heads, value_dim, key_dim), device=q.device, dtype=torch.float32)
        bk = triton.next_power_of_2(key_dim)
        bv = min(64, triton.next_power_of_2(value_dim))
        dummy_h0 = initial_state if has_initial else q
        grid = (triton.cdiv(value_dim, bv), batch * heads)
        _gdn_recurrent_fwd_kernel[grid](
            q,
            k,
            v,
            g,
            beta,
            dummy_h0,
            o,
            ht,
            scale=float(scale),
            T=seq_len,
            H=heads,
            K=key_dim,
            V=value_dim,
            BK=bk,
            BV=bv,
            HAS_INITIAL=has_initial,
            num_warps=1,
            num_stages=3,
        )
        empty = q.new_empty(0)
        ctx.save_for_backward(q, k, v, g, beta, initial_state if has_initial else empty)
        ctx.has_initial = has_initial
        ctx.scale = float(scale)
        return o, ht

    @staticmethod
    def backward(ctx, do: torch.Tensor, dht: Optional[torch.Tensor]):
        q, k, v, g, beta, initial_state_saved = ctx.saved_tensors
        initial_state = initial_state_saved if ctx.has_initial else None
        with torch.enable_grad():
            q_ = q.detach().requires_grad_(True)
            k_ = k.detach().requires_grad_(True)
            v_ = v.detach().requires_grad_(True)
            g_ = g.detach().requires_grad_(True)
            beta_ = beta.detach().requires_grad_(True)
            inputs = [q_, k_, v_, g_, beta_]
            if initial_state is not None:
                h0_ = initial_state.detach().requires_grad_(True)
                inputs.append(h0_)
            else:
                h0_ = None
            o_ref, ht_ref = _gdn_recurrent_torch(q_, k_, v_, g_, beta_, h0_, scale=ctx.scale)
            loss = (o_ref * do.to(o_ref.dtype)).sum()
            if dht is not None:
                loss = loss + (ht_ref * dht.to(ht_ref.dtype)).sum()
            grads = torch.autograd.grad(loss, inputs, allow_unused=True)
        grad_q, grad_k, grad_v, grad_g, grad_beta = grads[:5]
        grad_h0 = grads[5] if initial_state is not None else None
        return grad_q, grad_k, grad_v, grad_g, grad_beta, grad_h0, None


def gdn_triton_recurrent(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    *,
    initial_state: Optional[torch.Tensor] = None,
    scale: Optional[float] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if scale is None:
        scale = q.shape[-1] ** -0.5
    return _GDNRecurrentTritonFn.apply(q, k, v, g, beta, initial_state, float(scale))
