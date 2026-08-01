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
import torch.nn.functional as F

from fla.ops.gdn2 import chunk_gdn2

from study_rwkv_futureseed_loop import (
    FLADeltaTimeMix,
    GAIN_BUDGET_FLA_SHA,
    fold_gdn2_write_carrier_into_official_inputs,
)


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


def direct_carrier_recurrence(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    b: torch.Tensor,
    w: torch.Tensor,
    carrier: torch.Tensor,
    initial_state: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    q_unit = F.normalize(q.float(), dim=-1)
    k_unit = F.normalize(k.float(), dim=-1)
    scale = q.shape[-1] ** -0.5
    state = initial_state.float().clone()
    outputs = []
    for token in range(q.shape[1]):
        state = state * g[:, token].float().exp().unsqueeze(-1)
        erase = (
            (b[:, token].float() * k_unit[:, token]).unsqueeze(-1) * state
        ).sum(dim=-2)
        delta = w[:, token].float() * v[:, token].float() - erase
        write_address = carrier[:, token].float() * k_unit[:, token]
        state = state + write_address.unsqueeze(-1) * delta.unsqueeze(-2)
        outputs.append(
            (
                q_unit[:, token].unsqueeze(-1) * state
            ).sum(dim=-2) * scale
        )
    return torch.stack(outputs, dim=1), state


def check_fold_matches_direct(device: torch.device) -> dict[str, Any]:
    torch.manual_seed(8401)
    shape_k = (2, 81, 4, 16)
    shape_v = (2, 81, 4, 16)
    q = torch.randn(shape_k, device=device, dtype=torch.bfloat16)
    k = torch.randn(shape_k, device=device, dtype=torch.bfloat16)
    v = torch.randn(shape_v, device=device, dtype=torch.bfloat16)
    g = -0.02 * torch.rand(shape_k, device=device)
    b = torch.sigmoid(torch.randn(shape_k, device=device)).to(torch.bfloat16)
    w = torch.sigmoid(torch.randn(shape_v, device=device)).to(torch.bfloat16)
    carrier = (0.55 + 0.44 * torch.rand(shape_k, device=device)).float()
    initial_state = 0.03 * torch.randn(
        2, 4, 16, 16, device=device, dtype=torch.float32
    )

    k_effective, b_effective, w_effective, norm_ratio = (
        fold_gdn2_write_carrier_into_official_inputs(k, b, w, carrier)
    )
    expected_o, expected_state = direct_carrier_recurrence(
        q, k, v, g, b, w, carrier, initial_state
    )
    actual_o, actual_state = chunk_gdn2(
        q=q,
        k=k_effective,
        v=v,
        g=g,
        b=b_effective,
        w=w_effective,
        initial_state=initial_state,
        output_final_state=True,
        use_qk_l2norm_in_kernel=True,
    )
    output_error = max_abs(expected_o, actual_o)
    state_error = max_abs(expected_state, actual_state)
    if output_error > 0.025 or state_error > 0.025:
        raise AssertionError(
            "carrier fold does not match the direct recurrence: "
            f"output={output_error} state={state_error}"
        )
    return {
        "output_max_abs": output_error,
        "state_max_abs": state_error,
        "carrier_mean": float(carrier.mean().item()),
        "write_norm_ratio_mean": float(norm_ratio.mean().item()),
        "b_effective_max": float(b_effective.float().max().item()),
        "official_backward": type(actual_state.grad_fn).__name__,
    }


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


def check_shared_limit(device: torch.device) -> dict[str, Any]:
    torch.manual_seed(8402)
    shared = build("anchor_residual", device)
    carrier = build("anchor_carrier", device)
    carrier.core.load_state_dict(shared.core.state_dict(), strict=True)
    assert shared.address_residual_proj is not None
    assert carrier.address_residual_proj is not None
    carrier.address_residual_proj.load_state_dict(
        shared.address_residual_proj.state_dict(), strict=True
    )
    assert carrier.address_carrier_bias_delta is not None
    assert carrier.address_carrier_scale is not None
    carrier.address_carrier_bias_delta.data.fill_(20.0)
    carrier.address_carrier_scale.data.zero_()
    shared.train()
    carrier.train()
    x_shared = torch.randn(2, 81, 64, device=device, requires_grad=True)
    x_carrier = x_shared.detach().clone().requires_grad_(True)
    anchor_shared = torch.randn_like(x_shared, requires_grad=True)
    anchor_carrier = anchor_shared.detach().clone().requires_grad_(True)
    state_shared = (0.03 * torch.randn(
        2, 4, 16, 16, device=device, dtype=torch.float32
    )).requires_grad_(True)
    state_carrier = state_shared.detach().clone().requires_grad_(True)
    order = torch.randperm(81, device=device)
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        y_shared, h_shared = shared(
            x_shared,
            initial_state=state_shared,
            address=anchor_shared,
            cell_order=order,
        )
        y_carrier, h_carrier = carrier(
            x_carrier,
            initial_state=state_carrier,
            address=anchor_carrier,
            cell_order=order,
        )
    output_error = max_abs(y_shared, y_carrier)
    state_error = max_abs(h_shared, h_carrier)
    if output_error > 5e-4 or state_error > 5e-4:
        raise AssertionError(
            "open carrier failed to recover shared-address GDN2: "
            f"output={output_error} state={state_error}"
        )
    return {
        "output_max_abs": output_error,
        "state_max_abs": state_error,
        "parameter_delta": sum(p.numel() for p in carrier.parameters())
        - sum(p.numel() for p in shared.parameters()),
    }


def check_active_gradients(device: torch.device) -> dict[str, Any]:
    torch.manual_seed(8403)
    candidate = build("anchor_carrier", device)
    assert candidate.address_residual_proj is not None
    assert candidate.address_carrier_scale is not None
    assert candidate.address_carrier_bias_delta is not None
    torch.nn.init.normal_(candidate.address_residual_proj.weight, mean=0.0, std=0.02)
    candidate.address_carrier_scale.data.normal_(mean=0.0, std=0.25)
    candidate.address_carrier_bias_delta.data.fill_(-0.5)
    candidate.train()
    x = torch.randn(2, 81, 64, device=device, requires_grad=True)
    anchor = torch.randn_like(x, requires_grad=True)
    state = (0.03 * torch.randn(
        2, 4, 16, 16, device=device, dtype=torch.float32
    )).requires_grad_(True)
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
        "address_weight": candidate.address_residual_proj.weight.grad,
        "carrier_scale": candidate.address_carrier_scale.grad,
        "carrier_bias_delta": candidate.address_carrier_bias_delta.grad,
    }
    for name, gradient in gradients.items():
        if gradient is None or not bool(torch.isfinite(gradient).all()):
            raise AssertionError(f"missing or non-finite candidate gradient: {name}")
        if float(gradient.abs().max()) == 0.0:
            raise AssertionError(f"candidate received zero gradient: {name}")
    diagnostics = {
        key: float(value.detach().float().cpu())
        for key, value in candidate.last_gain_budget_diag.items()
        if key.startswith("gdn2_address_carrier_")
    }
    if not (0.0 < diagnostics["gdn2_address_carrier_mean"] < 1.0):
        raise AssertionError(f"invalid carrier diagnostics: {diagnostics}")
    if diagnostics["gdn2_address_carrier_token_std"] <= 0.0:
        raise AssertionError(f"carrier is not address specific: {diagnostics}")
    return {
        "loss": float(loss.detach().cpu()),
        "diagnostics": diagnostics,
        "gradient_max_abs": {
            name: float(gradient.detach().float().abs().max().cpu())
            for name, gradient in gradients.items()
        },
        "autograd_graph": graph,
    }


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required; CPU fallback is forbidden")
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
    payload = {
        "fla_source_sha": marker_sha,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpu_name": torch.cuda.get_device_name(device),
        "gpu_uuid": visible_uuid,
        "fold_matches_direct": check_fold_matches_direct(device),
        "shared_limit": check_shared_limit(device),
        "active_gradients": check_active_gradients(device),
    }
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
