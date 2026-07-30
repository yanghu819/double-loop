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

from study_rwkv_futureseed_loop import FLADeltaTimeMix


def max_abs(left: torch.Tensor, right: torch.Tensor) -> float:
    return float((left.float() - right.float()).abs().max().item())


def autograd_graph_names(tensor: torch.Tensor) -> list[str]:
    names: list[str] = []
    queue = [tensor.grad_fn]
    seen: set[int] = set()
    while queue:
        fn = queue.pop(0)
        if fn is None or id(fn) in seen:
            continue
        seen.add(id(fn))
        names.append(type(fn).__name__)
        queue.extend(
            next_fn
            for next_fn, _index in fn.next_functions
            if next_fn is not None
        )
    return names


def build_mixer(mode: str, device: torch.device) -> FLADeltaTimeMix:
    return FLADeltaTimeMix(
        64,
        4,
        16,
        backbone="gdn2",
        expand_v=1.0,
        mode="chunk",
        use_short_conv=True,
        conv_size=4,
        allow_neg_eigval=False,
        fast_slow_decay_mode=mode,
        fast_slow_decay_kernel_size=4,
        fast_slow_decay_rho_init=0.10,
        fast_slow_decay_current_weight_init=0.85,
    ).to(device)


def check_identity_case(
    device: torch.device,
    *,
    seed: int,
    with_initial_state: bool,
) -> dict[str, Any]:
    torch.manual_seed(seed)
    official = build_mixer("none", device)
    identity = build_mixer("external_identity", device)
    identity.core.load_state_dict(official.core.state_dict(), strict=True)
    official.train()
    identity.train()
    x_official = torch.randn(2, 81, 64, device=device, requires_grad=True)
    x_identity = x_official.detach().clone().requires_grad_(True)
    state_official = None
    state_identity = None
    if with_initial_state:
        state_official = (
            torch.randn(2, 4, 16, 16, device=device, dtype=torch.float32)
            * 0.05
        ).requires_grad_(True)
        state_identity = state_official.detach().clone().requires_grad_(True)

    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        y_official, h_official = official(
            x_official,
            initial_state=state_official,
        )
        y_identity, h_identity = identity(
            x_identity,
            initial_state=state_identity,
        )
    output_error = max_abs(y_official, y_identity)
    state_error = max_abs(h_official, h_identity)
    probe_y = torch.randn_like(y_official)
    probe_h = torch.randn_like(h_official)
    loss_official = (
        (y_official * probe_y).float().sum()
        + (h_official * probe_h).float().sum()
    )
    loss_identity = (
        (y_identity * probe_y).float().sum()
        + (h_identity * probe_h).float().sum()
    )
    official_parameters = [
        official.core.q_proj.weight,
        official.core.f_proj[0].weight,
        official.core.b_proj.weight,
        official.core.w_proj.weight,
    ]
    identity_parameters = [
        identity.core.q_proj.weight,
        identity.core.f_proj[0].weight,
        identity.core.b_proj.weight,
        identity.core.w_proj.weight,
    ]
    official_inputs = [x_official, *official_parameters]
    identity_inputs = [x_identity, *identity_parameters]
    if state_official is not None and state_identity is not None:
        official_inputs.insert(1, state_official)
        identity_inputs.insert(1, state_identity)
    grads_official = torch.autograd.grad(
        loss_official,
        official_inputs,
    )
    grads_identity = torch.autograd.grad(
        loss_identity,
        identity_inputs,
    )
    gradient_errors = [
        max_abs(left, right)
        for left, right in zip(grads_official, grads_identity)
    ]
    result = {
        "output_max_abs": output_error,
        "state_max_abs": state_error,
        "gradient_max_abs": max(gradient_errors),
        "gradient_errors": gradient_errors,
        "with_initial_state": with_initial_state,
        "identity_enabled": float(
            identity.last_gain_budget_diag["gdn2_fast_slow_enabled"].item()
        ),
    }
    if (
        output_error > 0.02
        or state_error > 0.02
        or result["gradient_max_abs"] > 0.08
        or result["identity_enabled"] != 0.0
    ):
        raise AssertionError(f"external identity differs from official GDN2: {result}")
    return result


def check_identity(device: torch.device) -> dict[str, Any]:
    return {
        "without_initial_state": check_identity_case(
            device,
            seed=7301,
            with_initial_state=False,
        ),
        "with_initial_state": check_identity_case(
            device,
            seed=7302,
            with_initial_state=True,
        ),
    }


def check_candidate(device: torch.device) -> dict[str, Any]:
    torch.manual_seed(7303)
    candidate = build_mixer("positive_causal", device)
    candidate.train()
    x = torch.randn(2, 81, 64, device=device, requires_grad=True)
    state = (
        torch.randn(2, 4, 16, 16, device=device, dtype=torch.float32) * 0.05
    ).requires_grad_(True)
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        output, terminal = candidate(x, initial_state=state)
        loss = output.float().square().mean() + terminal.float().square().mean()
    graph = autograd_graph_names(terminal)
    if "ChunkGDN2FunctionBackward" not in graph:
        raise AssertionError(
            "Fast-Slow candidate did not traverse the official GDN2 chunk kernel: "
            f"{graph}"
        )
    loss.backward()
    assert candidate.fast_slow_decay is not None
    gradients = {
        "kernel_logits": candidate.fast_slow_decay.kernel_logits.grad,
        "rho_logit": candidate.fast_slow_decay.rho_logit.grad,
        "input": x.grad,
        "initial_state": state.grad,
    }
    for name, gradient in gradients.items():
        if gradient is None or not bool(torch.isfinite(gradient).all()):
            raise AssertionError(f"missing or non-finite candidate gradient: {name}")
    if float(gradients["kernel_logits"].abs().max()) == 0.0:
        raise AssertionError("kernel logits received zero gradient")
    if float(gradients["rho_logit"].abs().max()) == 0.0:
        raise AssertionError("rho logit received zero gradient")
    diag = {
        key: float(value.detach().cpu())
        for key, value in candidate.last_gain_budget_diag.items()
        if key.startswith("gdn2_fast_slow_")
    }
    if diag["gdn2_fast_slow_enabled"] != 1.0:
        raise AssertionError(f"candidate did not activate: {diag}")
    if not (0.0 < diag["gdn2_fast_slow_tv_ratio"] < 1.0):
        raise AssertionError(f"candidate did not reduce temporal variation: {diag}")
    return {
        "loss": float(loss.detach().cpu()),
        "diagnostics": diag,
        "gradient_max_abs": {
            name: float(gradient.detach().float().abs().max().cpu())
            for name, gradient in gradients.items()
        },
        "autograd_graph": graph,
    }


def timed_step(
    mixer: FLADeltaTimeMix,
    x: torch.Tensor,
    state: torch.Tensor,
    *,
    iterations: int,
) -> tuple[float, int]:
    samples = []
    peak = 0
    for _ in range(iterations):
        mixer.zero_grad(set_to_none=True)
        x_step = x.detach().clone().requires_grad_(True)
        state_step = state.detach().clone().requires_grad_(True)
        torch.cuda.reset_peak_memory_stats(x.device)
        torch.cuda.synchronize(x.device)
        started = time.perf_counter()
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            output, terminal = mixer(x_step, initial_state=state_step)
            loss = output.float().square().mean() + terminal.float().square().mean()
        loss.backward()
        torch.cuda.synchronize(x.device)
        samples.append(time.perf_counter() - started)
        peak = max(peak, int(torch.cuda.max_memory_allocated(x.device)))
    return sum(samples) / len(samples), peak


def check_benchmark(device: torch.device) -> dict[str, Any]:
    torch.manual_seed(7304)
    control = FLADeltaTimeMix(
        192,
        6,
        32,
        backbone="gdn2",
        expand_v=1.0,
        mode="chunk",
        use_short_conv=True,
        conv_size=4,
        allow_neg_eigval=False,
        fast_slow_decay_mode="external_identity",
        fast_slow_decay_kernel_size=4,
        fast_slow_decay_rho_init=0.10,
        fast_slow_decay_current_weight_init=0.85,
    ).to(device)
    candidate = FLADeltaTimeMix(
        192,
        6,
        32,
        backbone="gdn2",
        expand_v=1.0,
        mode="chunk",
        use_short_conv=True,
        conv_size=4,
        allow_neg_eigval=False,
        fast_slow_decay_mode="positive_causal",
        fast_slow_decay_kernel_size=4,
        fast_slow_decay_rho_init=0.10,
        fast_slow_decay_current_weight_init=0.85,
    ).to(device)
    candidate.core.load_state_dict(control.core.state_dict(), strict=True)
    control.train()
    candidate.train()
    x = torch.randn(8, 81, 192, device=device)
    state = torch.randn(8, 6, 32, 32, device=device, dtype=torch.float32) * 0.05

    timed_step(control, x, state, iterations=2)
    timed_step(candidate, x, state, iterations=2)
    control_sec, control_peak = timed_step(control, x, state, iterations=5)
    candidate_sec, candidate_peak = timed_step(candidate, x, state, iterations=5)
    time_overhead = candidate_sec / control_sec - 1.0
    memory_overhead = candidate_peak / control_peak - 1.0
    result = {
        "control_sec": control_sec,
        "candidate_sec": candidate_sec,
        "time_overhead_frac": time_overhead,
        "control_peak_bytes": control_peak,
        "candidate_peak_bytes": candidate_peak,
        "memory_overhead_frac": memory_overhead,
    }
    if time_overhead > 0.20 or memory_overhead > 0.20:
        raise AssertionError(f"Fast-Slow systems gate failed: {result}")
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--git-sha", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required; CPU model smoke is forbidden")
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible != "0":
        raise RuntimeError(f"CUDA_VISIBLE_DEVICES must be exactly 0, got {visible!r}")
    device = torch.device("cuda", 0)
    payload = {
        "status": "running",
        "git_sha": args.git_sha,
        "torch_version": torch.__version__,
        "device": torch.cuda.get_device_name(device),
        "gpu_uuid": subprocess.check_output(
            [
                "nvidia-smi",
                "-i",
                "0",
                "--query-gpu=uuid",
                "--format=csv,noheader,nounits",
            ],
            text=True,
        ).strip(),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    try:
        payload["identity"] = check_identity(device)
        payload["candidate"] = check_candidate(device)
        payload["benchmark"] = check_benchmark(device)
        payload["status"] = "passed"
    except Exception as exc:
        payload["status"] = "failed"
        payload["error"] = f"{type(exc).__name__}: {exc}"
        args.out.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        raise
    args.out.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
