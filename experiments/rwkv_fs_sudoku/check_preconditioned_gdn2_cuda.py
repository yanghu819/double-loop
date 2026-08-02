#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from pathlib import Path
from typing import Any


for cache_var in (
    "XDG_CACHE_HOME",
    "TRITON_CACHE_DIR",
    "TORCHINDUCTOR_CACHE_DIR",
    "TORCH_EXTENSIONS_DIR",
    "TMPDIR",
):
    cache_path = os.environ.get(cache_var, "")
    if not cache_path.startswith("/huyang2/double-loop/"):
        raise RuntimeError(
            f"{cache_var} must point below /huyang2/double-loop before CUDA imports"
        )
if os.environ.get("FLA_DISABLE_BACKEND_DISPATCH") != "1":
    raise RuntimeError("FLA_DISABLE_BACKEND_DISPATCH=1 is required")
if os.environ.get("FLA_CONV_BACKEND") != "triton":
    raise RuntimeError("FLA_CONV_BACKEND=triton is required")

import torch

from fla.ops.gdn2 import chunk_gdn2

from preconditioned_gdn2 import (
    causal_tied_atk_preconditioner,
    fold_preconditioned_write_into_gdn2,
    futureseed_row_precision,
    normalize_qk_fp32,
)
from study_rwkv_futureseed_loop import FLADeltaTimeMix, resolve_strict_fla_source


def max_abs(left: torch.Tensor, right: torch.Tensor) -> float:
    return float((left.float() - right.float()).abs().max().item())


def graph_names(tensor: torch.Tensor) -> list[str]:
    names: list[str] = []
    queue = [tensor.grad_fn]
    seen: set[int] = set()
    while queue:
        fn = queue.pop(0)
        if fn is None or id(fn) in seen:
            continue
        seen.add(id(fn))
        names.append(type(fn).__name__)
        queue.extend(next_fn for next_fn, _ in fn.next_functions if next_fn is not None)
    return names


def sequential_multiplier(
    k_unit: torch.Tensor,
    g: torch.Tensor,
    b: torch.Tensor,
    initial_precision: torch.Tensor,
) -> torch.Tensor:
    precision = initial_precision.float().clone()
    values = []
    log_x = torch.tensor(1.5, device=k_unit.device).log()
    for token in range(k_unit.shape[1]):
        precision = (
            g[:, token].float().exp() * precision
            + b[:, token].float() * k_unit[:, token].float().square()
        )
        ell = (precision + 1e-6).log()
        centered = ell + 0.2
        squashed = centered / (1.0 + centered.abs())
        values.append(torch.exp(-log_x * squashed))
    return torch.stack(values, dim=1)


def direct_recurrence(
    q: torch.Tensor,
    k_write: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    b_kernel: torch.Tensor,
    w: torch.Tensor,
    initial_state: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    state = initial_state.float().clone()
    outputs = []
    scale = q.shape[-1] ** -0.5
    for token in range(q.shape[1]):
        state = state * g[:, token].float().exp().unsqueeze(-1)
        erase = (
            (b_kernel[:, token].float() * k_write[:, token].float()).unsqueeze(-1)
            * state
        ).sum(dim=-2)
        delta = w[:, token].float() * v[:, token].float() - erase
        state = state + k_write[:, token].float().unsqueeze(-1) * delta.unsqueeze(-2)
        outputs.append(
            (q[:, token].float().unsqueeze(-1) * state).sum(dim=-2) * scale
        )
    return torch.stack(outputs, dim=1), state


def check_algebra(device: torch.device) -> dict[str, Any]:
    torch.manual_seed(9201)
    shape = (2, 81, 4, 16)
    k = torch.randn(shape, device=device, dtype=torch.bfloat16)
    _q, k_unit = normalize_qk_fp32(k, k)
    g = -0.01 - 0.04 * torch.rand(shape, device=device)
    b = torch.sigmoid(torch.randn(shape, device=device)).to(torch.bfloat16)
    state = 0.15 * torch.randn(2, 4, 16, 24, device=device)
    initial_precision = futureseed_row_precision(state)
    assert initial_precision is not None
    if tuple(initial_precision.shape) != (2, 4, 16):
        raise AssertionError(
            "FutureSeed precision reduced the wrong recurrent-state axis: "
            f"{tuple(initial_precision.shape)}"
        )
    parallel, diag = causal_tied_atk_preconditioner(
        k_unit,
        g,
        b,
        initial_precision=initial_precision,
    )
    sequential = sequential_multiplier(k_unit, g, b, initial_precision)
    prefix_error = max_abs(parallel, sequential)
    k_write, b_kernel, fold_diag = fold_preconditioned_write_into_gdn2(
        k_unit,
        b,
        parallel,
    )
    erase_error = max_abs(
        b.float() * k_unit.float(),
        b_kernel.float() * k_write.float(),
    )
    if prefix_error > 2e-5:
        raise AssertionError(f"parallel ATK prefix differs from recurrence: {prefix_error}")
    if erase_error > 8e-3:
        raise AssertionError(f"quantized erase address drift is too large: {erase_error}")
    return {
        "parallel_vs_sequential_max_abs": prefix_error,
        "quantized_erase_max_abs": erase_error,
        "multiplier_min": float(parallel.min()),
        "multiplier_max": float(parallel.max()),
        "diagnostics": {
            **{key: float(value) for key, value in diag.items()},
            **{key: float(value) for key, value in fold_diag.items()},
        },
    }


def check_official_kernel(device: torch.device) -> dict[str, Any]:
    torch.manual_seed(9202)
    shape = (2, 81, 4, 16)
    q = torch.randn(shape, device=device, dtype=torch.bfloat16)
    k = torch.randn(shape, device=device, dtype=torch.bfloat16)
    value_shape = (2, 81, 4, 24)
    v = torch.randn(value_shape, device=device, dtype=torch.bfloat16)
    g = -0.01 - 0.04 * torch.rand(shape, device=device)
    b = torch.sigmoid(torch.randn(shape, device=device)).to(torch.bfloat16)
    w = torch.sigmoid(torch.randn(value_shape, device=device)).to(torch.bfloat16)
    state = 0.04 * torch.randn(2, 4, 16, 24, device=device)
    q_unit, k_unit = normalize_qk_fp32(q, k)
    initial_precision = futureseed_row_precision(state)
    multiplier, _diag = causal_tied_atk_preconditioner(
        k_unit,
        g,
        b,
        initial_precision=initial_precision,
    )
    k_write, b_kernel, _fold_diag = fold_preconditioned_write_into_gdn2(
        k_unit,
        b,
        multiplier,
    )
    expected_o, expected_state = direct_recurrence(
        q_unit,
        k_write,
        v,
        g,
        b_kernel,
        w,
        state,
    )
    actual_o, actual_state = chunk_gdn2(
        q=q_unit,
        k=k_write,
        v=v,
        g=g,
        b=b_kernel,
        w=w,
        initial_state=state,
        output_final_state=True,
        use_qk_l2norm_in_kernel=False,
    )
    output_error = max_abs(expected_o, actual_o)
    state_error = max_abs(expected_state, actual_state)
    if output_error > 0.035 or state_error > 0.035:
        raise AssertionError(
            "official GDN2 chunk differs from direct preconditioned recurrence: "
            f"output={output_error} state={state_error}"
        )
    graph = graph_names(actual_state)
    if "ChunkGDN2FunctionBackward" not in graph:
        raise AssertionError(f"official chunk backward is absent: {graph}")
    return {
        "output_max_abs": output_error,
        "terminal_state_max_abs": state_error,
        "autograd_graph": graph[:20],
    }


def check_backward_parity(device: torch.device) -> dict[str, Any]:
    torch.manual_seed(9205)
    shape = (1, 33, 2, 16)
    value_shape = (1, 33, 2, 24)
    state_shape = (1, 2, 16, 24)

    def leaves() -> tuple[torch.Tensor, ...]:
        return (
            torch.randn(shape, device=device, dtype=torch.bfloat16).requires_grad_(True),
            torch.randn(shape, device=device, dtype=torch.bfloat16).requires_grad_(True),
            torch.randn(value_shape, device=device, dtype=torch.bfloat16).requires_grad_(True),
            (-0.01 - 0.04 * torch.rand(shape, device=device)).requires_grad_(True),
            torch.sigmoid(torch.randn(shape, device=device)).to(torch.bfloat16).requires_grad_(True),
            torch.sigmoid(torch.randn(shape, device=device)).to(torch.bfloat16).requires_grad_(True),
            (0.04 * torch.randn(state_shape, device=device)).requires_grad_(True),
        )

    def run(
        tensors: tuple[torch.Tensor, ...],
        *,
        official: bool,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, ...]]:
        q, k, v, g, b, w, state = tensors
        q_unit, k_unit = normalize_qk_fp32(q, k)
        precision = futureseed_row_precision(state)
        multiplier, _diag = causal_tied_atk_preconditioner(
            k_unit,
            g,
            b,
            initial_precision=precision,
        )
        k_write, b_kernel, _fold_diag = fold_preconditioned_write_into_gdn2(
            k_unit,
            b,
            multiplier,
        )
        if official:
            output, terminal = chunk_gdn2(
                q=q_unit,
                k=k_write,
                v=v,
                g=g,
                b=b_kernel,
                w=w,
                initial_state=state,
                output_final_state=True,
                use_qk_l2norm_in_kernel=False,
            )
        else:
            output, terminal = direct_recurrence(
                q_unit,
                k_write,
                v,
                g,
                b_kernel,
                w,
                state,
            )
        loss = output.float().square().mean() + 0.01 * terminal.float().square().mean()
        gradients = torch.autograd.grad(loss, tensors)
        return loss, gradients

    official_tensors = leaves()
    reference_tensors = tuple(
        tensor.detach().clone().requires_grad_(True) for tensor in official_tensors
    )
    official_loss, official_gradients = run(official_tensors, official=True)
    reference_loss, reference_gradients = run(reference_tensors, official=False)
    names = ("q", "k", "v", "g", "b", "w", "initial_state")
    diagnostics = {}
    for name, actual, expected in zip(names, official_gradients, reference_gradients):
        absolute = max_abs(actual, expected)
        relative = float(
            (actual.float() - expected.float()).norm()
            / expected.float().norm().clamp_min(1e-8)
        )
        if not bool(torch.isfinite(actual).all()):
            raise AssertionError(f"non-finite official gradient: {name}")
        if absolute > 0.03 and relative > 0.12:
            raise AssertionError(
                f"official backward differs from direct recurrence for {name}: "
                f"abs={absolute} rel={relative}"
            )
        diagnostics[name] = {
            "max_abs": absolute,
            "relative_l2": relative,
        }
    return {
        "official_loss": float(official_loss.detach()),
        "reference_loss": float(reference_loss.detach()),
        "loss_abs": abs(float(official_loss.detach()) - float(reference_loss.detach())),
        "gradients": diagnostics,
    }


def build(mode: str, device: torch.device) -> FLADeltaTimeMix:
    return FLADeltaTimeMix(
        64,
        4,
        16,
        backbone="gdn2",
        expand_v=1.5,
        mode="chunk",
        use_short_conv=True,
        conv_size=4,
        allow_neg_eigval=False,
        precondition_mode=mode,
    ).to(device)


def check_full_layer_backward(device: torch.device) -> dict[str, Any]:
    torch.manual_seed(9203)
    layer = build("futureseed_tied_atk", device)
    layer.train()
    x = torch.randn(2, 81, 64, device=device, requires_grad=True)
    state = (0.04 * torch.randn(2, 4, 16, 24, device=device)).requires_grad_(True)
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        output, terminal = layer(x, initial_state=state)
        loss = output.float().square().mean() + 0.01 * terminal.float().square().mean()
    graph = graph_names(terminal)
    if "ChunkGDN2FunctionBackward" not in graph:
        raise AssertionError(f"full layer bypassed official GDN2 chunk: {graph}")
    loss.backward()
    gradients = {
        "input": x.grad,
        "initial_state": state.grad,
        "q_proj": layer.core.q_proj.weight.grad,
        "k_proj": layer.core.k_proj.weight.grad,
        "erase_proj": layer.core.b_proj.weight.grad,
        "write_proj": layer.core.w_proj.weight.grad,
    }
    for name, gradient in gradients.items():
        if gradient is None or not bool(torch.isfinite(gradient).all()):
            raise AssertionError(f"missing or non-finite gradient: {name}")
        if float(gradient.abs().max()) == 0.0:
            raise AssertionError(f"zero gradient: {name}")
    diagnostics = {
        key: float(value.detach().float().cpu())
        for key, value in layer.last_gain_budget_diag.items()
        if key.startswith("gdn2_precondition_")
    }
    return {
        "loss": float(loss.detach()),
        "gradient_max_abs": {
            name: float(gradient.detach().float().abs().max())
            for name, gradient in gradients.items()
        },
        "diagnostics": diagnostics,
    }


def benchmark(device: torch.device) -> dict[str, Any]:
    torch.manual_seed(9204)
    base = build("none", device).train()
    candidate = build("futureseed_tied_atk", device).train()
    candidate.core.load_state_dict(base.core.state_dict(), strict=True)
    x = torch.randn(8, 81, 64, device=device)
    state = 0.04 * torch.randn(8, 4, 16, 24, device=device)

    def measure(layer: FLADeltaTimeMix) -> tuple[float, float]:
        for _ in range(3):
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                output, terminal = layer(x, initial_state=state)
                (output.float().square().mean() + terminal.float().square().mean()).backward()
            layer.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats(device)
        start = time.perf_counter()
        for _ in range(10):
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                output, terminal = layer(x, initial_state=state)
                (output.float().square().mean() + terminal.float().square().mean()).backward()
            layer.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        elapsed = (time.perf_counter() - start) / 10.0
        memory = float(torch.cuda.max_memory_allocated(device))
        return elapsed, memory

    base_time, base_memory = measure(base)
    candidate_time, candidate_memory = measure(candidate)
    return {
        "base_sec": base_time,
        "candidate_sec": candidate_time,
        "time_overhead_frac": candidate_time / base_time - 1.0,
        "base_peak_bytes": base_memory,
        "candidate_peak_bytes": candidate_memory,
        "memory_overhead_frac": candidate_memory / base_memory - 1.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required; CPU smoke is forbidden")
    if torch.cuda.device_count() != 1:
        raise RuntimeError(
            f"CUDA_VISIBLE_DEVICES must expose exactly GPU1, got {torch.cuda.device_count()} devices"
        )
    expected_uuid = os.environ.get("GPU1_UUID", "").strip()
    actual_uuid = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=uuid", "--format=csv,noheader"],
        text=True,
    ).strip()
    if expected_uuid and actual_uuid != expected_uuid:
        raise RuntimeError(f"GPU UUID mismatch: {actual_uuid} != {expected_uuid}")
    source_sha, source_root, source_module = resolve_strict_fla_source("gdn2")
    device = torch.device("cuda:0")
    payload = {
        "status": "passed",
        "device": torch.cuda.get_device_name(device),
        "gpu_uuid": actual_uuid,
        "torch_version": torch.__version__,
        "fla_source_sha": source_sha,
        "fla_source_root": source_root,
        "fla_gdn2_module": source_module,
        "algebra": check_algebra(device),
        "official_kernel": check_official_kernel(device),
        "backward_parity": check_backward_parity(device),
        "full_layer_backward": check_full_layer_backward(device),
        "benchmark": benchmark(device),
    }
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
