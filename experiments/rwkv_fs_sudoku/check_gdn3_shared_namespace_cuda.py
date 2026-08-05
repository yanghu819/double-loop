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

from study_rwkv_futureseed_loop import FutureSeedRWKV, GAIN_BUDGET_FLA_SHA


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


def build(mode: str, device: torch.device) -> FutureSeedRWKV:
    return FutureSeedRWKV(
        64,
        3,
        4,
        16,
        2,
        future_seed_scale=1.0,
        future_seed_decay=0.0,
        future_seed_update="fixed",
        future_seed_norm_mode="unit",
        future_seed_gate_mode="head",
        backbone="gdn2",
        gdn_mode="chunk",
        gdn_expand_v=1.0,
        gdn_use_short_conv=True,
        gdn_conv_size=4,
        gdn_allow_neg_eigval=False,
        gdn2_address_mode=mode,
    ).to(device)


def base_gradient_inputs(
    model: FutureSeedRWKV,
    model_input: torch.Tensor,
) -> tuple[list[str], list[torch.Tensor]]:
    names = ["input"]
    tensors: list[torch.Tensor] = [model_input]
    for layer_idx, block in enumerate(model.blocks):
        names.extend((f"layer{layer_idx}.q_weight", f"layer{layer_idx}.k_weight"))
        tensors.extend((block.time_mix.core.q_proj.weight, block.time_mix.core.k_proj.weight))
    return names, tensors


def check_zero_init_identity(device: torch.device) -> dict[str, Any]:
    torch.manual_seed(9301)
    control = build("none", device)
    candidate = build("shared_namespace", device)
    missing, unexpected = candidate.load_state_dict(control.state_dict(), strict=False)
    if missing != ["shared_address_proj.weight"] or unexpected:
        raise AssertionError(
            f"unexpected zero-init migration: missing={missing} unexpected={unexpected}"
        )
    assert candidate.shared_address_proj is not None
    if float(candidate.shared_address_proj.weight.abs().max()) != 0.0:
        raise AssertionError("shared address projection did not initialize to exact zero")
    shared_parameter_names = [
        name for name, _parameter in candidate.named_parameters()
        if "shared_address_proj" in name
    ]
    if shared_parameter_names != ["shared_address_proj.weight"]:
        raise AssertionError(f"shared projection is not unique: {shared_parameter_names}")

    control.train()
    candidate.train()
    x_control = torch.randn(2, 81, 64, device=device, requires_grad=True)
    x_candidate = x_control.detach().clone().requires_grad_(True)
    anchor = torch.randn(2, 81, 64, device=device, requires_grad=True)
    order = torch.randperm(81, device=device)
    address_ptrs: list[int] = []
    hooks = []

    def capture_address(
        _module: torch.nn.Module,
        _args: tuple[torch.Tensor, ...],
        kwargs: dict[str, Any],
    ) -> None:
        address = kwargs.get("address")
        if address is None:
            raise AssertionError("candidate layer did not receive the shared address")
        address_ptrs.append(address.data_ptr())

    for block in candidate.blocks:
        hooks.append(
            block.time_mix.register_forward_pre_hook(capture_address, with_kwargs=True)
        )
    try:
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            output_control, diag_control, _ = control(
                x_control,
                address=None,
                cell_order=order,
            )
            output_candidate, diag_candidate, _ = candidate(
                x_candidate,
                address=anchor,
                cell_order=order,
            )
    finally:
        for hook in hooks:
            hook.remove()

    if len(address_ptrs) != len(candidate.blocks) or len(set(address_ptrs)) != 1:
        raise AssertionError(f"layers did not consume one shared address tensor: {address_ptrs}")
    output_error = max_abs(output_control, output_candidate)
    if output_error != 0.0:
        raise AssertionError(f"zero-init shared namespace changed model output: {output_error}")

    probe = torch.randn_like(output_control)
    loss_control = (output_control * probe).float().sum()
    loss_candidate = (output_candidate * probe).float().sum()
    gradient_names, control_inputs = base_gradient_inputs(control, x_control)
    _candidate_names, candidate_inputs = base_gradient_inputs(candidate, x_candidate)
    control_grads = torch.autograd.grad(loss_control, control_inputs)
    candidate_grads = torch.autograd.grad(
        loss_candidate,
        candidate_inputs + [candidate.shared_address_proj.weight, anchor],
    )
    gradient_alignment = {
        name: {
            "max_abs": max_abs(left, right),
            "allclose_atol_5e-5_rtol_1e-4": bool(
                torch.allclose(left, right, atol=5e-5, rtol=1e-4)
            ),
        }
        for name, left, right in zip(
            gradient_names,
            control_grads,
            candidate_grads[: len(control_grads)],
        )
    }
    if not all(row["allclose_atol_5e-5_rtol_1e-4"] for row in gradient_alignment.values()):
        raise AssertionError(f"zero-init base gradients changed: {gradient_alignment}")
    shared_grad = candidate_grads[-2]
    anchor_grad = candidate_grads[-1]
    if not bool(torch.isfinite(shared_grad).all()) or float(shared_grad.abs().max()) == 0.0:
        raise AssertionError("shared address projection cannot learn on its first step")
    if float(anchor_grad.abs().max()) != 0.0:
        raise AssertionError("zero-init shared namespace unexpectedly changed anchor gradient")
    if float(diag_candidate["gdn3_shared_address_weight_rms"]) != 0.0:
        raise AssertionError("zero-init shared-address diagnostic is nonzero")
    if max_abs(diag_control["fs_gate_mean"], diag_candidate["fs_gate_mean"]) != 0.0:
        raise AssertionError("zero-init shared namespace changed FutureSeed gating")
    return {
        "output_max_abs": output_error,
        "base_gradient_alignment": gradient_alignment,
        "shared_weight_grad_max_abs": float(shared_grad.float().abs().max().item()),
        "anchor_grad_max_abs": float(anchor_grad.float().abs().max().item()),
        "shared_parameter_names": shared_parameter_names,
        "shared_address_data_ptr_count": len(set(address_ptrs)),
        "parameter_delta": sum(p.numel() for p in candidate.parameters())
        - sum(p.numel() for p in control.parameters()),
    }


def check_active_candidate(device: torch.device) -> dict[str, Any]:
    torch.manual_seed(9302)
    control = build("none", device)
    candidate = build("shared_namespace", device)
    candidate.load_state_dict(control.state_dict(), strict=False)
    assert candidate.shared_address_proj is not None
    torch.nn.init.normal_(candidate.shared_address_proj.weight, mean=0.0, std=0.02)
    candidate.train()
    x = torch.randn(2, 81, 64, device=device, requires_grad=True)
    anchor = torch.randn_like(x, requires_grad=True)
    order = torch.randperm(81, device=device)
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        output, diagnostics, _ = candidate(
            x,
            address=anchor,
            cell_order=order,
        )
        loss = output.float().square().mean()
    graph = graph_names(output)
    if "ChunkGDN2FunctionBackward" not in graph:
        raise AssertionError(f"candidate bypassed official GDN2 chunk kernel: {graph}")
    loss.backward()
    gradients = {
        "input": x.grad,
        "anchor": anchor.grad,
        "shared_weight": candidate.shared_address_proj.weight.grad,
    }
    for name, gradient in gradients.items():
        if gradient is None or not bool(torch.isfinite(gradient).all()):
            raise AssertionError(f"missing or non-finite candidate gradient: {name}")
        if float(gradient.abs().max()) == 0.0:
            raise AssertionError(f"candidate received zero gradient: {name}")
    required_positive = (
        "gdn3_shared_address_weight_rms",
        "gdn3_shared_address_residual_rms",
        "gdn3_shared_address_token_std",
        "gdn2_address_residual_rms",
        "gdn2_address_q_relative_change",
        "gdn2_address_k_relative_change",
    )
    if any(float(diagnostics[key]) <= 0.0 for key in required_positive):
        raise AssertionError(
            "active shared namespace diagnostics failed: "
            f"{ {key: float(diagnostics[key]) for key in required_positive} }"
        )
    return {
        "loss": float(loss.detach().cpu()),
        "diagnostics": {
            key: float(diagnostics[key].detach().float().cpu())
            for key in required_positive
        },
        "gradient_max_abs": {
            name: float(gradient.detach().float().abs().max().cpu())
            for name, gradient in gradients.items()
        },
        "autograd_graph": graph,
    }


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required; CPU model smoke is forbidden")
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
        "zero_init_identity": check_zero_init_identity(device),
        "active_candidate": check_active_candidate(device),
    }
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
