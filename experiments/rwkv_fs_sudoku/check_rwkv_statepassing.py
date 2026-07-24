#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Iterable

for cache_var in (
    "XDG_CACHE_HOME",
    "TRITON_CACHE_DIR",
    "TORCHINDUCTOR_CACHE_DIR",
    "TORCH_EXTENSIONS_DIR",
    "TMPDIR",
):
    cache_path = os.environ.get(cache_var, "")
    if not cache_path.startswith("/huyang2/double-loop/"):
        raise RuntimeError(f"{cache_var} must point below /huyang2/double-loop before importing CUDA runtimes")

import torch

from rwkv7_cuda.statepassing import StatePassingRWKV7, statepassing_available


W_SCALE = -0.6065306597


def max_abs(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a.float() - b.float()).abs().max().item())


def rms(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a.float() - b.float()).square().mean().sqrt().item())


def relative_rms(a: torch.Tensor, b: torch.Tensor) -> float:
    denominator = max(float(a.float().square().mean().sqrt().item()), 1e-8)
    return rms(a, b) / denominator


def all_finite(tensors: Iterable[torch.Tensor | None]) -> bool:
    return all(t is not None and bool(torch.isfinite(t).all()) for t in tensors)


def clone_leaves(*tensors: torch.Tensor) -> list[torch.Tensor]:
    return [tensor.detach().clone().requires_grad_(True) for tensor in tensors]


def torch_recurrence(
    s0: torch.Tensor,
    r: torch.Tensor,
    w_raw: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    state = s0.float()
    outputs: list[torch.Tensor] = []
    for t in range(r.shape[1]):
        r_t = r[:, t].float()
        w_t = torch.exp(W_SCALE * torch.sigmoid(w_raw[:, t].float()))
        k_t = k[:, t].float()
        v_t = v[:, t].float()
        a_t = a[:, t].float()
        b_t = b[:, t].float()
        state_a = torch.einsum("bhij,bhj->bhi", state, a_t)
        state = (
            state * w_t.unsqueeze(-2)
            + state_a.unsqueeze(-1) * b_t.unsqueeze(-2)
            + v_t.unsqueeze(-1) * k_t.unsqueeze(-2)
        )
        outputs.append(torch.einsum("bhij,bhj->bhi", state, r_t))
    return torch.stack(outputs, dim=1), state


def compare_tensor(reference: torch.Tensor, actual: torch.Tensor) -> dict[str, Any]:
    return {
        "reference_dtype": str(reference.dtype),
        "actual_dtype": str(actual.dtype),
        "max_abs": max_abs(reference, actual),
        "rms": rms(reference, actual),
        "relative_rms": relative_rms(reference, actual),
    }


def check_forward_backward(device: torch.device, head_dim: int) -> dict[str, Any]:
    torch.manual_seed(20260724)
    batch, length, heads = 1, 32, 2
    shape = (batch, length, heads, head_dim)
    state_shape = (batch, heads, head_dim, head_dim)

    s0 = torch.randn(state_shape, device=device, dtype=torch.float32) * 0.05
    r = torch.randn(shape, device=device, dtype=torch.bfloat16) * 0.2
    w_raw = torch.randn(shape, device=device, dtype=torch.bfloat16)
    k = torch.randn(shape, device=device, dtype=torch.bfloat16) * 0.2
    v = torch.randn(shape, device=device, dtype=torch.bfloat16) * 0.2
    a = torch.randn(shape, device=device, dtype=torch.bfloat16) * 0.05
    b = torch.randn(shape, device=device, dtype=torch.bfloat16) * 0.05
    dy = torch.randn(shape, device=device, dtype=torch.bfloat16) * 0.1
    ds_t = torch.randn(state_shape, device=device, dtype=torch.float32) * 0.01

    ref_leaves = clone_leaves(s0, r, w_raw, k, v, a, b)
    ref_y_float, ref_state = torch_recurrence(*ref_leaves)
    ref_y = ref_y_float.to(torch.bfloat16)
    ref_loss = (ref_y.float() * dy.float()).sum() + (ref_state * ds_t).sum()
    ref_grads = torch.autograd.grad(ref_loss, ref_leaves)

    cuda_leaves = clone_leaves(s0, r, w_raw, k, v, a, b)
    cuda_y, cuda_state = StatePassingRWKV7.apply(*cuda_leaves)
    cuda_loss = (cuda_y.float() * dy.float()).sum() + (cuda_state * ds_t).sum()
    cuda_grads = torch.autograd.grad(cuda_loss, cuda_leaves)

    gradient_names = ("s0", "r", "w_raw", "k", "v", "a", "b")
    gradient_errors = {
        name: compare_tensor(reference, actual)
        for name, reference, actual in zip(gradient_names, ref_grads, cuda_grads)
    }
    result = {
        "output": compare_tensor(ref_y, cuda_y),
        "terminal_state": compare_tensor(ref_state, cuda_state),
        "gradients": gradient_errors,
        "all_gradients_finite": all_finite(cuda_grads),
        "cuda_output_dtype": str(cuda_y.dtype),
        "cuda_state_dtype": str(cuda_state.dtype),
    }
    if result["output"]["max_abs"] > 0.03:
        raise AssertionError(f"RWKV state-passing output mismatch: {result['output']}")
    if result["terminal_state"]["max_abs"] > 0.03:
        raise AssertionError(f"RWKV state-passing terminal-state mismatch: {result['terminal_state']}")
    if max(row["relative_rms"] for row in gradient_errors.values()) > 0.08:
        raise AssertionError(f"RWKV state-passing gradient mismatch: {gradient_errors}")
    if not result["all_gradients_finite"]:
        raise AssertionError("RWKV state-passing returned non-finite gradients")
    return result


@torch.no_grad()
def check_split_continuity(device: torch.device, head_dim: int) -> dict[str, Any]:
    torch.manual_seed(20260725)
    batch, length, heads = 1, 32, 2
    shape = (batch, length, heads, head_dim)
    state_shape = (batch, heads, head_dim, head_dim)
    tensors = [
        torch.randn(state_shape, device=device, dtype=torch.float32) * 0.05,
        torch.randn(shape, device=device, dtype=torch.bfloat16) * 0.2,
        torch.randn(shape, device=device, dtype=torch.bfloat16),
        torch.randn(shape, device=device, dtype=torch.bfloat16) * 0.2,
        torch.randn(shape, device=device, dtype=torch.bfloat16) * 0.2,
        torch.randn(shape, device=device, dtype=torch.bfloat16) * 0.05,
        torch.randn(shape, device=device, dtype=torch.bfloat16) * 0.05,
    ]
    s0, *sequence_tensors = tensors
    full_y, full_state = StatePassingRWKV7.apply(s0, *sequence_tensors)
    first_y, midpoint_state = StatePassingRWKV7.apply(s0, *(tensor[:, :16] for tensor in sequence_tensors))
    second_y, split_state = StatePassingRWKV7.apply(
        midpoint_state,
        *(tensor[:, 16:] for tensor in sequence_tensors),
    )
    split_y = torch.cat((first_y, second_y), dim=1)
    result = {
        "output": compare_tensor(full_y, split_y),
        "terminal_state": compare_tensor(full_state, split_state),
        "midpoint_state_dtype": str(midpoint_state.dtype),
    }
    if result["output"]["max_abs"] != 0.0 or result["terminal_state"]["max_abs"] != 0.0:
        raise AssertionError(f"RWKV state-passing split continuity failed: {result}")
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Strict GPU parity check for the vendored RWKV7 state-passing kernel")
    parser.add_argument("--head-dim", type=int, default=32)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required; this check must not run on CPU")
    device = torch.device("cuda")
    ok, reason = statepassing_available(args.head_dim)
    if not ok:
        raise RuntimeError(reason)

    payload = {
        "device": torch.cuda.get_device_name(device),
        "torch": torch.__version__,
        "head_dim": args.head_dim,
        "forward_backward": check_forward_backward(device, args.head_dim),
        "split_continuity": check_split_continuity(device, args.head_dim),
    }
    encoded = json.dumps(payload, indent=2, sort_keys=True)
    print(encoded)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
