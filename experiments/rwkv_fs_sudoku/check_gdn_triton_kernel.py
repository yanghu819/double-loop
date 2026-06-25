#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from gdn_triton import _gdn_recurrent_torch, gdn_triton_recurrent


@dataclass(frozen=True)
class KernelCase:
    name: str
    batch: int
    seq: int
    heads: int
    dim: int
    value_dim: int
    use_initial_state: bool = True
    input_dtype: str = "float32"
    decay: str = "mild"
    beta: str = "sigmoid"
    loss: str = "output_plus_state"
    seed_offset: int = 0


def clone_req(x: torch.Tensor) -> torch.Tensor:
    return x.detach().clone().requires_grad_(True)


def max_abs_rel(lhs: torch.Tensor, rhs: torch.Tensor) -> tuple[float, float]:
    diff = (lhs - rhs).abs()
    denom = torch.maximum(lhs.abs(), rhs.abs()).clamp_min(1e-8)
    return float(diff.max().item()), float((diff / denom).max().item())


def grad_or_zeros(value: torch.Tensor | None, like: torch.Tensor) -> torch.Tensor:
    if value is None:
        return torch.zeros_like(like)
    return value


def build_inputs(case: KernelCase, *, seed: int, device: torch.device) -> dict[str, torch.Tensor | None]:
    torch.manual_seed(seed + case.seed_offset)
    q = torch.randn(case.batch, case.seq, case.heads, case.dim, device=device, dtype=torch.float32)
    k = torch.randn_like(q)
    q = F.normalize(q, dim=-1)
    k = F.normalize(k, dim=-1)
    v = torch.randn(case.batch, case.seq, case.heads, case.value_dim, device=device, dtype=torch.float32)
    if case.decay == "mild":
        g = -torch.rand(case.batch, case.seq, case.heads, device=device, dtype=torch.float32) * 0.2
    elif case.decay == "strong":
        g = -torch.rand(case.batch, case.seq, case.heads, device=device, dtype=torch.float32) * 2.0
    elif case.decay == "zero":
        g = torch.zeros(case.batch, case.seq, case.heads, device=device, dtype=torch.float32)
    else:
        raise ValueError(f"unknown decay mode: {case.decay}")
    if case.beta == "sigmoid":
        beta = torch.sigmoid(torch.randn(case.batch, case.seq, case.heads, device=device, dtype=torch.float32))
    elif case.beta == "low":
        beta = torch.full((case.batch, case.seq, case.heads), 0.05, device=device, dtype=torch.float32)
    elif case.beta == "high":
        beta = torch.full((case.batch, case.seq, case.heads), 0.95, device=device, dtype=torch.float32)
    else:
        raise ValueError(f"unknown beta mode: {case.beta}")
    h0 = None
    if case.use_initial_state:
        h0 = torch.randn(case.batch, case.heads, case.value_dim, case.dim, device=device, dtype=torch.float32) * 0.1
    if case.input_dtype == "bfloat16":
        q = q.to(torch.bfloat16)
        k = k.to(torch.bfloat16)
        v = v.to(torch.bfloat16)
        g = g.to(torch.bfloat16)
        beta = beta.to(torch.bfloat16)
        h0 = h0.to(torch.bfloat16) if h0 is not None else None
    elif case.input_dtype != "float32":
        raise ValueError(f"unknown input dtype: {case.input_dtype}")
    return {"q": q, "k": k, "v": v, "g": g, "beta": beta, "h0": h0}


def case_thresholds(case: KernelCase) -> dict[str, float]:
    if case.input_dtype == "bfloat16":
        return {"forward_abs": 3e-3, "forward_rel": 3e-2, "grad_abs": 3e-3, "grad_rel": 5e-2}
    return {"forward_abs": 2e-5, "forward_rel": 2e-4, "grad_abs": 2e-5, "grad_rel": 2e-4}


def run_case(case: KernelCase, *, seed: int) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required; CPU fallback is intentionally disabled.")
    device = torch.device("cuda")
    tensors = build_inputs(case, seed=seed, device=device)
    qt, kt, vt, gt, bt = [clone_req(tensors[name]) for name in ("q", "k", "v", "g", "beta")]
    qr, kr, vr, gr, br = [clone_req(tensors[name]) for name in ("q", "k", "v", "g", "beta")]
    h0 = tensors["h0"]
    h0t = clone_req(h0) if h0 is not None else None
    h0r = clone_req(h0) if h0 is not None else None

    # Warm up compile separately from measured timing.
    gdn_triton_recurrent(qt, kt, vt, gt, bt, initial_state=h0t)
    torch.cuda.synchronize()

    t0 = time.time()
    ot, htt = gdn_triton_recurrent(qt, kt, vt, gt, bt, initial_state=h0t)
    torch.cuda.synchronize()
    triton_forward_sec = time.time() - t0

    t1 = time.time()
    or_, htr = _gdn_recurrent_torch(qr, kr, vr, gr, br, h0r, scale=case.dim**-0.5)
    torch.cuda.synchronize()
    torch_forward_sec = time.time() - t1

    w_o = torch.randn_like(ot)
    w_h = torch.randn_like(htt)
    if case.loss == "output_only":
        loss_t = (ot * w_o).sum()
        loss_r = (or_ * w_o).sum()
    elif case.loss == "state_only":
        loss_t = (htt * w_h).sum()
        loss_r = (htr * w_h).sum()
    elif case.loss == "output_plus_state":
        loss_t = (ot * w_o).sum() + (htt * w_h).sum()
        loss_r = (or_ * w_o).sum() + (htr * w_h).sum()
    else:
        raise ValueError(f"unknown loss mode: {case.loss}")
    loss_t.backward()
    loss_r.backward()

    result: dict[str, Any] = {
        **asdict(case),
        "triton_forward_sec": float(triton_forward_sec),
        "torch_forward_sec": float(torch_forward_sec),
        "speedup_forward": float(torch_forward_sec / max(triton_forward_sec, 1e-12)),
    }
    fwd_o_abs, fwd_o_rel = max_abs_rel(ot, or_)
    fwd_h_abs, fwd_h_rel = max_abs_rel(htt, htr)
    result.update(
        {
            "forward_o_max_abs": fwd_o_abs,
            "forward_o_max_rel": fwd_o_rel,
            "forward_h_max_abs": fwd_h_abs,
            "forward_h_max_rel": fwd_h_rel,
        }
    )
    for name, lhs, rhs in [
        ("q", qt, qr),
        ("k", kt, kr),
        ("v", vt, vr),
        ("g", gt, gr),
        ("beta", bt, br),
    ]:
        abs_diff, rel_diff = max_abs_rel(grad_or_zeros(lhs.grad, lhs), grad_or_zeros(rhs.grad, rhs))
        result[f"grad_{name}_max_abs"] = abs_diff
        result[f"grad_{name}_max_rel"] = rel_diff
    if h0t is not None and h0r is not None:
        abs_diff, rel_diff = max_abs_rel(grad_or_zeros(h0t.grad, h0t), grad_or_zeros(h0r.grad, h0r))
        result["grad_h0_max_abs"] = abs_diff
        result["grad_h0_max_rel"] = rel_diff

    thresholds = case_thresholds(case)
    failed: list[str] = []
    for key in ("forward_o", "forward_h"):
        abs_bad = result[f"{key}_max_abs"] > thresholds["forward_abs"]
        rel_bad = result[f"{key}_max_rel"] > thresholds["forward_rel"]
        if abs_bad and rel_bad:
            failed.append(f"{key}_abs_and_rel")
    for name in ["q", "k", "v", "g", "beta"] + (["h0"] if h0t is not None else []):
        abs_bad = result[f"grad_{name}_max_abs"] > thresholds["grad_abs"]
        rel_bad = result[f"grad_{name}_max_rel"] > thresholds["grad_rel"]
        if abs_bad and rel_bad:
            failed.append(f"grad_{name}_abs_and_rel")
    result["thresholds"] = thresholds
    result["passed"] = not failed
    result["failed_checks"] = failed
    return result


def default_cases() -> list[KernelCase]:
    return [
        KernelCase("t1_no_h0_output_only", 1, 1, 1, 8, 8, False, "float32", "zero", "sigmoid", "output_only", 1),
        KernelCase("small_h0_output_plus_state", 2, 9, 3, 16, 16, True, "float32", "mild", "sigmoid", "output_plus_state", 2),
        KernelCase("non_power_dims_strong_decay", 2, 17, 2, 12, 20, True, "float32", "strong", "sigmoid", "output_plus_state", 3),
        KernelCase("state_only_no_h0", 1, 11, 2, 16, 24, False, "float32", "mild", "high", "state_only", 4),
        KernelCase("longer_t32_k32_v16", 1, 32, 4, 32, 16, True, "float32", "mild", "low", "output_plus_state", 5),
        KernelCase("v64_boundary", 1, 7, 2, 8, 64, True, "float32", "zero", "sigmoid", "output_plus_state", 6),
        KernelCase("bf16_runner_like", 2, 9, 2, 16, 16, True, "bfloat16", "mild", "sigmoid", "output_plus_state", 7),
    ]


def summarize(results: list[dict[str, Any]]) -> dict[str, Any]:
    diff_keys = sorted(
        {
            key
            for row in results
            for key in row
            if key.endswith("_max_abs") or key.endswith("_max_rel")
        }
    )
    return {
        "status": "ok" if all(row["passed"] for row in results) else "failed",
        "num_cases": len(results),
        "num_passed": sum(1 for row in results if row["passed"]),
        "num_failed": sum(1 for row in results if not row["passed"]),
        "maxima": {key: max(float(row.get(key, 0.0)) for row in results) for key in diff_keys},
        "min_forward_speedup": min(float(row["speedup_forward"]) for row in results),
        "failed_cases": [
            {"name": row["name"], "failed_checks": row["failed_checks"]}
            for row in results
            if not row["passed"]
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--matrix", action="store_true", help="Run the fixed multi-shape kernel alignment suite.")
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--seq", type=int, default=9)
    parser.add_argument("--heads", type=int, default=3)
    parser.add_argument("--dim", type=int, default=16)
    parser.add_argument("--value_dim", type=int, default=16)
    parser.add_argument("--no_initial_state", action="store_true")
    parser.add_argument("--input_dtype", choices=("float32", "bfloat16"), default="float32")
    parser.add_argument("--decay", choices=("mild", "strong", "zero"), default="mild")
    parser.add_argument("--beta", choices=("sigmoid", "low", "high"), default="sigmoid")
    parser.add_argument("--loss", choices=("output_only", "state_only", "output_plus_state"), default="output_plus_state")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--out", default="")
    args = parser.parse_args()
    if args.matrix:
        results = [run_case(case, seed=args.seed) for case in default_cases()]
        payload: dict[str, Any] = {"summary": summarize(results), "cases": results}
        if payload["summary"]["status"] != "ok":
            text = json.dumps(payload, indent=2, sort_keys=True)
            print(text)
            if args.out:
                Path(args.out).write_text(text + "\n", encoding="utf-8")
            raise SystemExit(1)
    else:
        case = KernelCase(
            name="single",
            batch=args.batch,
            seq=args.seq,
            heads=args.heads,
            dim=args.dim,
            value_dim=args.value_dim,
            use_initial_state=not args.no_initial_state,
            input_dtype=args.input_dtype,
            decay=args.decay,
            beta=args.beta,
            loss=args.loss,
        )
        result = run_case(case, seed=args.seed)
        payload = {"summary": summarize([result]), "cases": [result]}
        if not result["passed"]:
            text = json.dumps(payload, indent=2, sort_keys=True)
            print(text)
            if args.out:
                Path(args.out).write_text(text + "\n", encoding="utf-8")
            raise SystemExit(1)
    text = json.dumps(payload, indent=2, sort_keys=True)
    print(text)
    if args.out:
        Path(args.out).write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
