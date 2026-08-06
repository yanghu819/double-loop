#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import subprocess
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


EXPECTED_GPU_UUID = "GPU-53e9f3b4-2966-65d3-6614-09c540921519"


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


def build(
    device: torch.device,
    *,
    content_mode: str,
    layers: int = 4,
    heads: int = 4,
    head_dim: int = 16,
) -> FutureSeedRWKV:
    return FutureSeedRWKV(
        heads * head_dim,
        layers,
        heads,
        head_dim,
        4,
        future_seed_scale=1.0,
        future_seed_decay=0.0,
        future_seed_update="fixed",
        future_seed_norm_mode="unit",
        future_seed_gate_mode="head",
        future_seed_scope="layer",
        future_seed_readout_hop=0,
        future_seed_content_mode=content_mode,
        backbone="gdn2",
        gdn_mode="chunk",
        gdn_expand_v=1.0,
        gdn_use_short_conv=True,
        gdn_conv_size=4,
        gdn_allow_neg_eigval=False,
        gdn2_address_mode="position_qk",
        gdn2_cross_layer_init="independent",
    ).to(device)


def copy_shared_state(control: FutureSeedRWKV, candidate: FutureSeedRWKV) -> None:
    missing, unexpected = candidate.load_state_dict(control.state_dict(), strict=False)
    if missing != ["future_seed_innovation_scale"] or unexpected:
        raise AssertionError(
            f"unexpected state migration: missing={missing}, unexpected={unexpected}"
        )


def capture_terminal_states(
    model: FutureSeedRWKV,
    x: torch.Tensor,
    address: torch.Tensor,
    cell_order: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, torch.Tensor], list[torch.Tensor]]:
    states: list[torch.Tensor] = []
    hooks = []

    def capture(
        _module: torch.nn.Module,
        _args: tuple[torch.Tensor, ...],
        _kwargs: dict[str, Any],
        output: tuple[torch.Tensor, torch.Tensor],
    ) -> None:
        states.append(output[1])

    for block in model.blocks:
        hooks.append(block.time_mix.register_forward_hook(capture, with_kwargs=True))
    try:
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            output, diagnostics, _ = model(
                x,
                address=address,
                cell_order=cell_order,
            )
    finally:
        for hook in hooks:
            hook.remove()
    return output, diagnostics, states


def check_zero_identity_and_gradient(device: torch.device) -> dict[str, Any]:
    torch.manual_seed(9404)
    control = build(device, content_mode="terminal")
    torch.manual_seed(9404)
    candidate = build(device, content_mode="innovation_residual")
    copy_shared_state(control, candidate)
    control.eval()
    candidate.eval()

    x = torch.randn(2, 81, 64, device=device)
    address = torch.randn_like(x)
    cell_order = torch.randperm(81, device=device)
    with torch.no_grad():
        control_output, _control_diag, control_states = capture_terminal_states(
            control,
            x,
            address,
            cell_order,
        )
        candidate_output, candidate_diag, candidate_states = capture_terminal_states(
            candidate,
            x,
            address,
            cell_order,
        )
    if not torch.equal(control_output, candidate_output):
        raise AssertionError("zero-init innovation path changed the model output")
    state_errors = [
        float((left.float() - right.float()).abs().max().item())
        for left, right in zip(control_states, candidate_states)
    ]
    if len(state_errors) != len(candidate.blocks) or max(state_errors) != 0.0:
        raise AssertionError(f"zero-init terminal-state mismatch: {state_errors}")
    expected_delta = (len(candidate.blocks) - 2) * candidate.blocks[0].future_seed_logit.shape[1]
    parameter_delta = sum(p.numel() for p in candidate.parameters()) - sum(
        p.numel() for p in control.parameters()
    )
    if parameter_delta != expected_delta:
        raise AssertionError(f"parameter delta {parameter_delta} != {expected_delta}")

    candidate.train()
    candidate.zero_grad(set_to_none=True)
    x_grad = x.detach().clone().requires_grad_(True)
    output, diagnostics, terminal_states = capture_terminal_states(
        candidate,
        x_grad,
        address,
        cell_order,
    )
    target = torch.randn_like(output, dtype=torch.float32)
    loss = (output.float() * target).mean()
    loss.backward()
    scale_grad = candidate.future_seed_innovation_scale.grad
    if scale_grad is None or not bool(torch.isfinite(scale_grad).all()):
        raise AssertionError("innovation-scale gradient is missing or non-finite")
    scale_grad_max = float(scale_grad.float().abs().max().item())
    if scale_grad_max <= 0.0:
        raise AssertionError("innovation-scale gradient is zero")
    graph = graph_names(terminal_states[-1])
    if "ChunkGDN2FunctionBackward" not in graph:
        raise AssertionError(f"official GDN2 chunk kernel is missing: {graph}")
    if float(diagnostics["fs3_innovation_enabled"].item()) != 1.0:
        raise AssertionError("innovation diagnostics did not mark the path active")
    if float(diagnostics["fs3_innovation_scale_abs"].item()) != 0.0:
        raise AssertionError("zero-init innovation scale is not zero")
    address_enabled = float(diagnostics["gdn2_address_enabled"].item())
    if address_enabled != 1.0:
        raise AssertionError("position-QK address path is not active")
    return {
        "output_exact_identity": True,
        "terminal_state_max_abs_errors": state_errors,
        "parameter_delta": parameter_delta,
        "innovation_scale_grad_max_abs": scale_grad_max,
        "innovation_fraction": float(
            diagnostics["fs3_innovation_fraction"].item()
        ),
        "official_chunk_backward": True,
        "position_qk_active": True,
        "initial_diag": {
            key: float(value.detach().float().item())
            for key, value in candidate_diag.items()
            if key.startswith("fs3_")
        },
    }


def check_bounded_orthogonal_formula(device: torch.device) -> dict[str, float]:
    torch.manual_seed(9405)
    candidate = build(device, content_mode="innovation_residual")
    assert candidate.future_seed_innovation_scale is not None
    candidate.future_seed_innovation_scale.data.fill_(0.2)
    terminal = torch.randn(3, 4, 16, 16, device=device, dtype=torch.bfloat16)
    incoming = torch.randn_like(terminal)
    composed, diagnostics = candidate._compose_future_seed_content(
        terminal,
        incoming,
        receiver_layer_idx=2,
    )
    residual = composed.float() - terminal.float()
    cosine_numerator = (residual * incoming.float()).sum(dim=(-1, -2)).abs()
    cosine_denominator = (
        residual.square().sum(dim=(-1, -2)).sqrt()
        * incoming.float().square().sum(dim=(-1, -2)).sqrt()
    ).clamp_min(1e-6)
    max_abs_cosine = float((cosine_numerator / cosine_denominator).max().item())
    residual_rms = residual.square().mean(dim=(-1, -2)).sqrt()
    terminal_rms = terminal.float().square().mean(dim=(-1, -2)).sqrt()
    max_relative_rms = float((residual_rms / terminal_rms).max().item())
    expected_bound = float(torch.tanh(torch.tensor(0.2)).item()) + 0.01
    if max_abs_cosine > 0.01:
        raise AssertionError(f"innovation residual lost orthogonality: {max_abs_cosine}")
    if max_relative_rms > expected_bound:
        raise AssertionError(
            f"innovation residual exceeded its RMS bound: {max_relative_rms}"
        )
    return {
        "max_abs_cosine_with_incoming": max_abs_cosine,
        "max_residual_relative_rms": max_relative_rms,
        "reported_innovation_fraction": float(
            diagnostics["fs3_innovation_fraction"].item()
        ),
    }


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required; CPU smoke is forbidden")
    if torch.cuda.device_count() != 1:
        raise RuntimeError(f"exactly one visible CUDA device is required, got {torch.cuda.device_count()}")
    visible_gpu = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=index,uuid",
            "--format=csv,noheader",
        ],
        text=True,
    ).strip().splitlines()
    if visible_gpu != [f"0, {EXPECTED_GPU_UUID}"]:
        raise RuntimeError(f"unexpected visible GPU contract: {visible_gpu}")
    device = torch.device("cuda:0")
    props = torch.cuda.get_device_properties(device)
    output = {
        "device": str(device),
        "gpu_name": props.name,
        "gpu_uuid": EXPECTED_GPU_UUID,
        "fla_source_sha": GAIN_BUDGET_FLA_SHA,
        "zero_identity_and_gradient": check_zero_identity_and_gradient(device),
        "bounded_orthogonal_formula": check_bounded_orthogonal_formula(device),
    }
    print(json.dumps(output, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
