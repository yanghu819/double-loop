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
    states,
    scale: tl.constexpr,
    T: tl.constexpr,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    HAS_INITIAL: tl.constexpr,
    SAVE_STATES: tl.constexpr,
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
        if SAVE_STATES:
            state_ptrs = states + (((b * T + t) * H + h) * V + offs_v[:, None]) * K + offs_k[None, :]
            tl.store(state_ptrs, state, mask=mask_state)

    ht_ptrs = ht + ((b * H + h) * V + offs_v[:, None]) * K + offs_k[None, :]
    tl.store(ht_ptrs, state, mask=mask_state)


@triton.jit
def _gdn_recurrent_bwd_kernel(
    q,
    k,
    v,
    g,
    beta,
    h0,
    states,
    do,
    dht,
    dq,
    dk,
    dv,
    dg,
    dbeta,
    dh0,
    scale: tl.constexpr,
    T: tl.constexpr,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    HAS_INITIAL: tl.constexpr,
    HAS_DHT: tl.constexpr,
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

    q_base = q + ((b * T) * H + h) * K
    k_base = k + ((b * T) * H + h) * K
    v_base = v + ((b * T) * H + h) * V
    g_base = g + (b * T) * H + h
    beta_base = beta + (b * T) * H + h
    do_base = do + ((b * T) * H + h) * V
    dq_base = dq + ((b * T) * H + h) * K
    dk_base = dk + ((b * T) * H + h) * K
    dv_base = dv + ((b * T) * H + h) * V
    dg_base = dg + (b * T) * H + h
    dbeta_base = dbeta + (b * T) * H + h

    grad_state = tl.zeros((BV, BK), dtype=tl.float32)
    if HAS_DHT:
        dht_ptrs = dht + ((b * H + h) * V + offs_v[:, None]) * K + offs_k[None, :]
        grad_state += tl.load(dht_ptrs, mask=mask_state, other=0.0).to(tl.float32)

    for t_rev in tl.range(0, T):
        t = T - 1 - t_rev
        q_t = tl.load(q_base + t * H * K + offs_k, mask=mask_k, other=0.0).to(tl.float32)
        k_t = tl.load(k_base + t * H * K + offs_k, mask=mask_k, other=0.0).to(tl.float32)
        v_t = tl.load(v_base + t * H * V + offs_v, mask=mask_v, other=0.0).to(tl.float32)
        do_t = tl.load(do_base + t * H * V + offs_v, mask=mask_v, other=0.0).to(tl.float32)
        decay_t = tl.exp(tl.load(g_base + t * H).to(tl.float32))
        beta_t = tl.load(beta_base + t * H).to(tl.float32)

        state_t_ptrs = states + (((b * T + t) * H + h) * V + offs_v[:, None]) * K + offs_k[None, :]
        state_t = tl.load(state_t_ptrs, mask=mask_state, other=0.0).to(tl.float32)
        prev_state = tl.zeros((BV, BK), dtype=tl.float32)
        if t == 0:
            if HAS_INITIAL:
                h0_ptrs = h0 + ((b * H + h) * V + offs_v[:, None]) * K + offs_k[None, :]
                prev_state += tl.load(h0_ptrs, mask=mask_state, other=0.0).to(tl.float32)
        else:
            prev_ptrs = states + (((b * T + (t - 1)) * H + h) * V + offs_v[:, None]) * K + offs_k[None, :]
            prev_state += tl.load(prev_ptrs, mask=mask_state, other=0.0).to(tl.float32)

        q_scaled = q_t * scale
        grad_state += do_t[:, None] * q_scaled[None, :]
        dq_t = tl.sum(state_t * do_t[:, None], axis=0) * scale
        tl.atomic_add(dq_base + t * H * K + offs_k, dq_t, sem="relaxed", mask=mask_k)

        decayed_state = prev_state * decay_t
        pred = tl.sum(decayed_state * k_t[None, :], axis=1)
        residual = v_t - pred
        delta_v = beta_t * residual

        grad_delta = tl.sum(grad_state * k_t[None, :], axis=1)
        grad_k = tl.sum(grad_state * delta_v[:, None], axis=0)
        grad_beta = tl.sum(grad_delta * residual, axis=0)
        grad_v = grad_delta * beta_t
        grad_pred = -grad_v
        grad_k += tl.sum(decayed_state * grad_pred[:, None], axis=0)
        grad_decayed_state = grad_state + grad_pred[:, None] * k_t[None, :]
        grad_decay = tl.sum(tl.sum(grad_decayed_state * prev_state, axis=0), axis=0)
        grad_g = grad_decay * decay_t
        grad_state = grad_decayed_state * decay_t

        tl.atomic_add(dk_base + t * H * K + offs_k, grad_k, sem="relaxed", mask=mask_k)
        tl.store(dv_base + t * H * V + offs_v, grad_v, mask=mask_v)
        tl.atomic_add(dg_base + t * H, grad_g, sem="relaxed")
        tl.atomic_add(dbeta_base + t * H, grad_beta, sem="relaxed")

    if HAS_INITIAL:
        dh0_ptrs = dh0 + ((b * H + h) * V + offs_v[:, None]) * K + offs_k[None, :]
        tl.store(dh0_ptrs, grad_state, mask=mask_state)


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
        save_states = any(ctx.needs_input_grad[:6])
        states = (
            torch.empty((batch, seq_len, heads, value_dim, key_dim), device=q.device, dtype=torch.float32)
            if save_states
            else q.new_empty(0)
        )
        bk = triton.next_power_of_2(key_dim)
        bv = min(64, triton.next_power_of_2(value_dim))
        dummy_h0 = initial_state if has_initial else q
        dummy_states = states if save_states else q
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
            dummy_states,
            scale=float(scale),
            T=seq_len,
            H=heads,
            K=key_dim,
            V=value_dim,
            BK=bk,
            BV=bv,
            HAS_INITIAL=has_initial,
            SAVE_STATES=save_states,
            num_warps=1,
            num_stages=3,
        )
        empty = q.new_empty(0)
        ctx.save_for_backward(q, k, v, g, beta, initial_state if has_initial else empty, states)
        ctx.has_initial = has_initial
        ctx.save_states = save_states
        ctx.scale = float(scale)
        ctx.shape = (batch, seq_len, heads, key_dim, value_dim, bk, bv)
        return o, ht

    @staticmethod
    def backward(ctx, do: torch.Tensor, dht: Optional[torch.Tensor]):
        q, k, v, g, beta, initial_state_saved, states = ctx.saved_tensors
        if not ctx.save_states:
            return None, None, None, None, None, None, None
        batch, seq_len, heads, key_dim, value_dim, bk, bv = ctx.shape
        initial_state = initial_state_saved if ctx.has_initial else None
        do = do.contiguous()
        has_dht = dht is not None
        dht_contig = dht.contiguous() if has_dht else q

        grad_q = torch.zeros(q.shape, device=q.device, dtype=torch.float32)
        grad_k = torch.zeros(k.shape, device=k.device, dtype=torch.float32)
        grad_v = torch.empty(v.shape, device=v.device, dtype=torch.float32)
        grad_g = torch.zeros(g.shape, device=g.device, dtype=torch.float32)
        grad_beta = torch.zeros(beta.shape, device=beta.device, dtype=torch.float32)
        grad_h0 = (
            torch.empty(initial_state.shape, device=initial_state.device, dtype=torch.float32)
            if initial_state is not None
            else None
        )
        dummy_h0 = initial_state if initial_state is not None else q
        dummy_dh0 = grad_h0 if grad_h0 is not None else q
        grid = (triton.cdiv(value_dim, bv), batch * heads)
        _gdn_recurrent_bwd_kernel[grid](
            q,
            k,
            v,
            g,
            beta,
            dummy_h0,
            states,
            do,
            dht_contig,
            grad_q,
            grad_k,
            grad_v,
            grad_g,
            grad_beta,
            dummy_dh0,
            scale=ctx.scale,
            T=seq_len,
            H=heads,
            K=key_dim,
            V=value_dim,
            BK=bk,
            BV=bv,
            HAS_INITIAL=ctx.has_initial,
            HAS_DHT=has_dht,
            num_warps=1,
            num_stages=3,
        )
        return (
            grad_q.to(q.dtype),
            grad_k.to(k.dtype),
            grad_v.to(v.dtype),
            grad_g.to(g.dtype),
            grad_beta.to(beta.dtype),
            grad_h0.to(initial_state.dtype) if grad_h0 is not None else None,
            None,
        )


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
