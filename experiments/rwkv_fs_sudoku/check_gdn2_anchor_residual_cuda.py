#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import subprocess
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

from study_rwkv_futureseed_loop import FLADeltaTimeMix, GAIN_BUDGET_FLA_SHA


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


def build(mode: str, device: torch.device) -> FLADeltaTimeMix:
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
        address_mode=mode,
    ).to(device)


def check_official_identity(
    device: torch.device,
    *,
    with_initial_state: bool,
) -> dict[str, Any]:
    torch.manual_seed(8301 + int(with_initial_state))
    official = build("none", device)
    candidate = build("anchor_residual", device)
    candidate.core.load_state_dict(official.core.state_dict(), strict=True)
    assert candidate.address_residual_proj is not None
    if float(candidate.address_residual_proj.weight.abs().max()) != 0.0:
        raise AssertionError("address residual did not initialize to exact zero")
    official.train()
    candidate.train()
    x_official = torch.randn(2, 81, 64, device=device, requires_grad=True)
    x_candidate = x_official.detach().clone().requires_grad_(True)
    anchor = torch.randn(2, 81, 64, device=device, requires_grad=True)
    order = torch.randperm(81, device=device)
    state_official = None
    state_candidate = None
    if with_initial_state:
        state_official = (
            torch.randn(2, 4, 16, 16, device=device, dtype=torch.float32) * 0.05
        ).requires_grad_(True)
        state_candidate = state_official.detach().clone().requires_grad_(True)

    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        y_official, h_official = official(x_official, initial_state=state_official)
        y_candidate, h_candidate = candidate(
            x_candidate,
            initial_state=state_candidate,
            address=anchor,
            cell_order=order,
        )
    output_error = max_abs(y_official, y_candidate)
    state_error = max_abs(h_official, h_candidate)
    probe_y = torch.randn_like(y_official)
    probe_h = torch.randn_like(h_official)
    loss_official = (y_official * probe_y).float().sum() + (h_official * probe_h).float().sum()
    loss_candidate = (y_candidate * probe_y).float().sum() + (h_candidate * probe_h).float().sum()
    official_inputs = [x_official, official.core.q_proj.weight, official.core.k_proj.weight]
    candidate_inputs = [x_candidate, candidate.core.q_proj.weight, candidate.core.k_proj.weight]
    if state_official is not None and state_candidate is not None:
        official_inputs.insert(1, state_official)
        candidate_inputs.insert(1, state_candidate)
    official_grads = torch.autograd.grad(loss_official, official_inputs)
    candidate_grads = torch.autograd.grad(
        loss_candidate,
        candidate_inputs + [candidate.address_residual_proj.weight, anchor],
    )
    gradient_names = ["input", "q_weight", "k_weight"]
    if state_official is not None:
        gradient_names.insert(1, "initial_state")
    gradient_alignment = {
        name: {
            "max_abs": max_abs(left, right),
            "reference_max_abs": float(left.detach().float().abs().max().item()),
            "allclose_atol_5e-5_rtol_1e-4": bool(
                torch.allclose(left, right, atol=5e-5, rtol=1e-4)
            ),
        }
        for name, left, right in zip(
            gradient_names,
            official_grads,
            candidate_grads[: len(official_grads)],
        )
    }
    gradient_error = max(
        item["max_abs"] for item in gradient_alignment.values()
    )
    residual_weight_grad = candidate_grads[-2]
    anchor_grad = candidate_grads[-1]
    gradients_aligned = all(
        item["allclose_atol_5e-5_rtol_1e-4"]
        for item in gradient_alignment.values()
    )
    if output_error != 0.0 or state_error != 0.0 or not gradients_aligned:
        raise AssertionError(
            "zero-init address residual differs from official GDN2: "
            f"output={output_error} state={state_error} "
            f"gradient_alignment={gradient_alignment}"
        )
    if not bool(torch.isfinite(residual_weight_grad).all()) or float(
        residual_weight_grad.abs().max()
    ) == 0.0:
        raise AssertionError("zero-init address residual cannot learn on its first step")
    if float(anchor_grad.abs().max()) != 0.0:
        raise AssertionError("zero-init residual unexpectedly changed anchor gradient")
    return {
        "with_initial_state": with_initial_state,
        "output_max_abs": output_error,
        "state_max_abs": state_error,
        "base_gradient_max_abs": gradient_error,
        "base_gradient_alignment": gradient_alignment,
        "residual_weight_grad_max_abs": float(
            residual_weight_grad.abs().max().item()
        ),
        "anchor_grad_max_abs": float(anchor_grad.abs().max().item()),
        "parameter_delta": sum(p.numel() for p in candidate.parameters())
        - sum(p.numel() for p in official.parameters()),
    }


def check_active_candidate(device: torch.device) -> dict[str, Any]:
    torch.manual_seed(8304)
    candidate = build("anchor_residual", device)
    assert candidate.address_residual_proj is not None
    torch.nn.init.normal_(candidate.address_residual_proj.weight, mean=0.0, std=0.02)
    candidate.train()
    x = torch.randn(2, 81, 64, device=device, requires_grad=True)
    anchor = torch.randn_like(x, requires_grad=True)
    state = (
        torch.randn(2, 4, 16, 16, device=device, dtype=torch.float32) * 0.05
    ).requires_grad_(True)
    order = torch.randperm(81, device=device)
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        output, terminal = candidate(
            x,
            initial_state=state,
            address=anchor,
            cell_order=order,
        )
        loss = output.float().square().mean() + terminal.float().square().mean()
    graph = graph_names(terminal)
    if "ChunkGDN2FunctionBackward" not in graph:
        raise AssertionError(f"candidate bypassed official GDN2 chunk kernel: {graph}")
    loss.backward()
    gradients = {
        "input": x.grad,
        "anchor": anchor.grad,
        "initial_state": state.grad,
        "residual_weight": candidate.address_residual_proj.weight.grad,
    }
    for name, gradient in gradients.items():
        if gradient is None or not bool(torch.isfinite(gradient).all()):
            raise AssertionError(f"missing or non-finite candidate gradient: {name}")
        if float(gradient.abs().max()) == 0.0:
            raise AssertionError(f"candidate received zero gradient: {name}")
    diagnostics = {
        key: float(value.detach().float().cpu())
        for key, value in candidate.last_gain_budget_diag.items()
        if key.startswith("gdn2_address_")
    }
    required_positive = (
        "gdn2_address_residual_weight_rms",
        "gdn2_address_residual_rms",
        "gdn2_address_residual_token_std",
        "gdn2_address_residual_q_ratio",
        "gdn2_address_residual_k_ratio",
        "gdn2_address_q_relative_change",
        "gdn2_address_k_relative_change",
    )
    if diagnostics["gdn2_address_enabled"] != 1.0 or any(
        diagnostics[key] <= 0.0 for key in required_positive
    ):
        raise AssertionError(f"active residual diagnostics failed: {diagnostics}")

    with torch.no_grad():
        canonical = candidate.address_residual_proj(anchor)
        ordered = candidate.address_residual_proj(anchor.index_select(1, order))
    reorder_error = max_abs(canonical.index_select(1, order), ordered)
    if reorder_error != 0.0:
        raise AssertionError(f"address residual did not travel with its token: {reorder_error}")
    return {
        "loss": float(loss.detach().cpu()),
        "diagnostics": diagnostics,
        "address_reorder_max_abs": reorder_error,
        "gradient_max_abs": {
            name: float(gradient.detach().float().abs().max().cpu())
            for name, gradient in gradients.items()
        },
        "autograd_graph": graph,
    }


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required; CPU smoke is forbidden")
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "0":
        raise RuntimeError("CUDA_VISIBLE_DEVICES=0 is required for GPU1-only validation")
    marker = Path("/huyang2/double-loop/.cache/fla-source-sha")
    marker_sha = marker.read_text(encoding="utf-8").strip()
    if marker_sha != GAIN_BUDGET_FLA_SHA:
        raise RuntimeError(f"strict FLA marker mismatch: {marker_sha} != {GAIN_BUDGET_FLA_SHA}")
    device = torch.device("cuda:0")
    visible_uuid = subprocess.check_output(
        [
            "nvidia-smi",
            "-i",
            "0",
            "--query-gpu=uuid",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    ).strip()
    expected_uuid = os.environ.get("GPU1_UUID", "")
    if not expected_uuid or visible_uuid != expected_uuid:
        raise RuntimeError(f"GPU1 UUID mismatch: visible={visible_uuid} expected={expected_uuid}")
    result = {
        "cuda_device": torch.cuda.get_device_name(device),
        "cuda_uuid": visible_uuid,
        "fla_source_sha": marker_sha,
        "official_identity": {
            "without_initial_state": check_official_identity(
                device, with_initial_state=False
            ),
            "with_initial_state": check_official_identity(
                device, with_initial_state=True
            ),
        },
        "active_candidate": check_active_candidate(device),
    }
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
