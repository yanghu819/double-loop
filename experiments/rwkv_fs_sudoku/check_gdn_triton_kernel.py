#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from gdn_triton import _gdn_recurrent_torch, gdn_triton_recurrent


def clone_req(x: torch.Tensor) -> torch.Tensor:
    return x.detach().clone().requires_grad_(True)


def run_check(args: argparse.Namespace) -> dict[str, float | str]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required; CPU fallback is intentionally disabled.")
    torch.manual_seed(args.seed)
    device = torch.device("cuda")
    q = torch.randn(args.batch, args.seq, args.heads, args.dim, device=device, dtype=torch.float32)
    k = torch.randn_like(q)
    q = F.normalize(q, dim=-1)
    k = F.normalize(k, dim=-1)
    v = torch.randn(args.batch, args.seq, args.heads, args.value_dim, device=device, dtype=torch.float32)
    g = -torch.rand(args.batch, args.seq, args.heads, device=device, dtype=torch.float32) * 0.2
    beta = torch.sigmoid(torch.randn(args.batch, args.seq, args.heads, device=device, dtype=torch.float32))
    h0 = torch.randn(args.batch, args.heads, args.value_dim, args.dim, device=device, dtype=torch.float32) * 0.1

    qt, kt, vt, gt, bt, h0t = [clone_req(x) for x in (q, k, v, g, beta, h0)]
    qr, kr, vr, gr, br, h0r = [clone_req(x) for x in (q, k, v, g, beta, h0)]

    # Warm up compile separately from measured timing.
    gdn_triton_recurrent(qt, kt, vt, gt, bt, initial_state=h0t)
    torch.cuda.synchronize()

    t0 = time.time()
    ot, htt = gdn_triton_recurrent(qt, kt, vt, gt, bt, initial_state=h0t)
    torch.cuda.synchronize()
    triton_forward_sec = time.time() - t0

    t1 = time.time()
    or_, htr = _gdn_recurrent_torch(qr, kr, vr, gr, br, h0r, scale=args.dim**-0.5)
    torch.cuda.synchronize()
    torch_forward_sec = time.time() - t1

    w_o = torch.randn_like(ot)
    w_h = torch.randn_like(htt)
    loss_t = (ot * w_o).sum() + (htt * w_h).sum()
    loss_r = (or_ * w_o).sum() + (htr * w_h).sum()
    loss_t.backward()
    loss_r.backward()

    result: dict[str, float | str] = {
        "status": "ok",
        "batch": args.batch,
        "seq": args.seq,
        "heads": args.heads,
        "dim": args.dim,
        "value_dim": args.value_dim,
        "forward_o_max_diff": float((ot - or_).abs().max().item()),
        "forward_h_max_diff": float((htt - htr).abs().max().item()),
        "triton_forward_sec": float(triton_forward_sec),
        "torch_forward_sec": float(torch_forward_sec),
    }
    for name, lhs, rhs in [
        ("q", qt, qr),
        ("k", kt, kr),
        ("v", vt, vr),
        ("g", gt, gr),
        ("beta", bt, br),
        ("h0", h0t, h0r),
    ]:
        result[f"grad_{name}_max_diff"] = float((lhs.grad - rhs.grad).abs().max().item())
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--seq", type=int, default=9)
    parser.add_argument("--heads", type=int, default=3)
    parser.add_argument("--dim", type=int, default=16)
    parser.add_argument("--value_dim", type=int, default=16)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--out", default="")
    args = parser.parse_args()
    result = run_check(args)
    text = json.dumps(result, indent=2, sort_keys=True)
    print(text)
    if args.out:
        Path(args.out).write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
