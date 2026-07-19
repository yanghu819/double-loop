#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Iterable

import torch
import torch.nn.functional as F

from fla.ops.gdn2 import chunk_gdn2, naive_recurrent_gdn2
from fla.ops.kda import chunk_kda
from fla.ops.kda.naive import naive_recurrent_kda
from study_rwkv_futureseed_loop import FLADeltaTimeMix


def max_abs(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a.float() - b.float()).abs().max().item())


def all_finite(tensors: Iterable[torch.Tensor | None]) -> bool:
    return all(t is not None and bool(torch.isfinite(t).all()) for t in tensors)


def clone_leaves(*tensors: torch.Tensor) -> list[torch.Tensor]:
    return [tensor.detach().clone().requires_grad_(True) for tensor in tensors]


def check_gdn2_reference(device: torch.device) -> dict[str, Any]:
    torch.manual_seed(101)
    batch, length, heads, key_dim, value_dim = 1, 81, 2, 32, 32
    q = torch.randn(batch, length, heads, key_dim, device=device)
    k = torch.randn_like(q)
    v = torch.randn(batch, length, heads, value_dim, device=device) * 0.25
    g = torch.empty_like(q).uniform_(-4.0, -0.05)
    b = torch.rand_like(q)
    w = torch.rand_like(v)
    h0_vk = torch.randn(batch, heads, value_dim, key_dim, device=device) * 0.1
    do = torch.randn_like(v)
    dht_vk = torch.randn_like(h0_vk) * 0.1

    ref_leaves = clone_leaves(q, k, v, g, b, w, h0_vk)
    rq, rk, rv, rg, rb, rw, rh0_vk = ref_leaves
    ref, ref_ht_kv = naive_recurrent_gdn2(
        q=F.normalize(rq.float(), dim=-1),
        k=F.normalize(rk.float(), dim=-1),
        v=rv,
        g=rg,
        b=rb,
        w=rw,
        initial_state=rh0_vk.transpose(-1, -2),
        output_final_state=True,
    )
    ref_loss = (ref * do).sum() + (ref_ht_kv.transpose(-1, -2) * dht_vk).sum()
    ref_grads = torch.autograd.grad(ref_loss, ref_leaves)

    tri_leaves = clone_leaves(q, k, v, g, b, w, h0_vk)
    tq, tk, tv, tg, tb, tw, th0_vk = tri_leaves
    tri, tri_ht_vk = chunk_gdn2(
        q=tq,
        k=tk,
        v=tv,
        g=tg,
        b=tb,
        w=tw,
        initial_state=th0_vk,
        output_final_state=True,
        use_qk_l2norm_in_kernel=True,
        state_v_first=True,
    )
    tri_loss = (tri * do).sum() + (tri_ht_vk * dht_vk).sum()
    tri_grads = torch.autograd.grad(tri_loss, tri_leaves)

    grad_errors = [max_abs(a, b) for a, b in zip(ref_grads, tri_grads)]
    result = {
        "output_max_abs": max_abs(ref, tri),
        "state_max_abs": max_abs(ref_ht_kv.transpose(-1, -2), tri_ht_vk),
        "gradient_max_abs": max(grad_errors),
        "gradient_errors": dict(zip(("q", "k", "v", "g", "b", "w", "h0"), grad_errors)),
    }
    if result["output_max_abs"] > 0.02 or result["state_max_abs"] > 0.02 or result["gradient_max_abs"] > 0.08:
        raise AssertionError(f"GDN2 FLA/Torch mismatch: {result}")
    return result


def check_kda_reference(device: torch.device) -> dict[str, Any]:
    torch.manual_seed(202)
    batch, length, heads, key_dim, value_dim = 1, 81, 2, 32, 32
    q = torch.randn(batch, length, heads, key_dim, device=device)
    k = torch.randn_like(q)
    v = torch.randn(batch, length, heads, value_dim, device=device) * 0.25
    g_raw = torch.randn_like(q)
    beta_raw = torch.randn(batch, length, heads, device=device)
    a_log = torch.log(torch.empty(heads, device=device).uniform_(1.0, 16.0))
    dt_bias = torch.randn(heads * key_dim, device=device)
    h0_vk = torch.randn(batch, heads, value_dim, key_dim, device=device) * 0.1
    do = torch.randn_like(v)
    dht_vk = torch.randn_like(h0_vk) * 0.1

    ref_leaves = clone_leaves(q, k, v, g_raw, beta_raw, a_log, dt_bias, h0_vk)
    rq, rk, rv, rg, rbeta, ra, rdt, rh0_vk = ref_leaves
    g_ref = -ra.float().exp().view(1, 1, heads, 1) * F.softplus(
        rg.float() + rdt.float().view(1, 1, heads, key_dim)
    )
    ref, ref_ht_kv = naive_recurrent_kda(
        q=F.normalize(rq.float(), dim=-1),
        k=F.normalize(rk.float(), dim=-1),
        v=rv,
        g=g_ref,
        beta=torch.sigmoid(rbeta),
        initial_state=rh0_vk.transpose(-1, -2),
        output_final_state=True,
    )
    ref_loss = (ref * do).sum() + (ref_ht_kv.transpose(-1, -2) * dht_vk).sum()
    ref_grads = torch.autograd.grad(ref_loss, ref_leaves)

    tri_leaves = clone_leaves(q, k, v, g_raw, beta_raw, a_log, dt_bias, h0_vk)
    tq, tk, tv, tg, tbeta, ta, tdt, th0_vk = tri_leaves
    tri, tri_ht_vk = chunk_kda(
        q=tq,
        k=tk,
        v=tv,
        g=tg,
        beta=tbeta,
        A_log=ta,
        dt_bias=tdt,
        initial_state=th0_vk,
        output_final_state=True,
        use_qk_l2norm_in_kernel=True,
        use_gate_in_kernel=True,
        use_beta_sigmoid_in_kernel=True,
        state_v_first=True,
    )
    tri_loss = (tri * do).sum() + (tri_ht_vk * dht_vk).sum()
    tri_grads = torch.autograd.grad(tri_loss, tri_leaves)

    grad_errors = [max_abs(a, b) for a, b in zip(ref_grads, tri_grads)]
    result = {
        "output_max_abs": max_abs(ref, tri),
        "state_max_abs": max_abs(ref_ht_kv.transpose(-1, -2), tri_ht_vk),
        "gradient_max_abs": max(grad_errors),
        "gradient_errors": dict(zip(("q", "k", "v", "g", "beta", "A_log", "dt_bias", "h0"), grad_errors)),
    }
    if result["output_max_abs"] > 0.02 or result["state_max_abs"] > 0.02 or result["gradient_max_abs"] > 0.08:
        raise AssertionError(f"KDA FLA/Torch mismatch: {result}")
    return result


def check_adapter(backbone: str, device: torch.device) -> dict[str, Any]:
    torch.manual_seed(303 if backbone == "gdn2" else 404)
    batch, length, heads, head_dim = 2, 81, 2, 32
    mixer = FLADeltaTimeMix(
        heads * head_dim,
        heads,
        head_dim,
        backbone=backbone,
        expand_v=1.0,
        mode="chunk",
        use_short_conv=False,
        conv_size=4,
        allow_neg_eigval=False,
    ).to(device)
    x = torch.randn(batch, length, heads * head_dim, device=device, requires_grad=True)
    h0 = torch.randn(batch, heads, head_dim, head_dim, device=device, dtype=torch.float32, requires_grad=True) * 0.05
    h0.retain_grad()
    torch.cuda.reset_peak_memory_stats(device)
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        y, ht = mixer(x, initial_state=h0)
        loss = y.float().square().mean() + ht.float().square().mean() * 1e-3
    loss.backward()
    parameter_grads = [p.grad for p in mixer.parameters() if p.requires_grad]
    if not all_finite([y, ht, x.grad, h0.grad, *parameter_grads]):
        raise AssertionError(f"{backbone} adapter produced non-finite output or gradient")

    mixer.eval()
    split = 37
    with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        full_y, full_ht = mixer(x.detach(), initial_state=h0.detach())
        left_y, left_ht = mixer(x.detach()[:, :split], initial_state=h0.detach())
        right_y, right_ht = mixer(x.detach()[:, split:], initial_state=left_ht)
        split_y = torch.cat((left_y, right_y), dim=1)
    split_output_error = max_abs(full_y, split_y)
    split_state_error = max_abs(full_ht, right_ht)
    if split_output_error > 0.05 or split_state_error > 0.05:
        raise AssertionError(
            f"{backbone} recurrent split mismatch: output={split_output_error}, state={split_state_error}"
        )

    for _ in range(2):
        mixer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            bench_y, bench_ht = mixer(x.detach(), initial_state=h0.detach())
            bench_loss = bench_y.float().square().mean() + bench_ht.float().square().mean() * 1e-3
        bench_loss.backward()
    torch.cuda.synchronize(device)
    repeats = 5
    started = time.perf_counter()
    for _ in range(repeats):
        mixer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            bench_y, bench_ht = mixer(x.detach(), initial_state=h0.detach())
            bench_loss = bench_y.float().square().mean() + bench_ht.float().square().mean() * 1e-3
        bench_loss.backward()
    torch.cuda.synchronize(device)
    elapsed_ms = (time.perf_counter() - started) * 1000.0 / repeats
    return {
        "parameters": sum(p.numel() for p in mixer.parameters()),
        "output_shape": list(y.shape),
        "state_shape": list(ht.shape),
        "state_dtype": str(ht.dtype),
        "initial_state_grad_norm": float(h0.grad.float().norm().item()),
        "split_output_max_abs": split_output_error,
        "split_state_max_abs": split_state_error,
        "forward_backward_ms": elapsed_ms,
        "peak_memory_mb": torch.cuda.max_memory_allocated(device) / (1024**2),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required; CPU smoke is intentionally forbidden.")
    device = torch.device("cuda", 0)
    payload = {
        "torch": torch.__version__,
        "device": torch.cuda.get_device_name(device),
        "gdn2_reference": check_gdn2_reference(device),
        "kda_reference": check_kda_reference(device),
        "gdn2_adapter": check_adapter("gdn2", device),
        "kda_adapter": check_adapter("kda", device),
    }
    text = json.dumps(payload, indent=2, sort_keys=True)
    print(text, flush=True)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
