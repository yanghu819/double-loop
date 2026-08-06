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

from futureseed3_producer_codec import (
    FutureSeedProducerCodec,
    producer_codec_parameter_count,
)
from study_rwkv_futureseed_loop import FutureSeedRWKV, GAIN_BUDGET_FLA_SHA


EXPECTED_GPU_UUID = "GPU-53e9f3b4-2966-65d3-6614-09c540921519"
EXPECTED_FLA_SOURCE_SHA = "9c8e42e762fce087c27b673af4922795d9edb85e"


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


def build(device: torch.device, *, content_mode: str) -> FutureSeedRWKV:
    return FutureSeedRWKV(
        256,
        12,
        8,
        32,
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


def copy_shared_state(
    control: FutureSeedRWKV,
    candidate: FutureSeedRWKV,
) -> list[str]:
    missing, unexpected = candidate.load_state_dict(control.state_dict(), strict=False)
    if candidate.future_seed_producer_codec is None:
        raise AssertionError("candidate producer codec is missing")
    expected = {
        f"future_seed_producer_codec.{name}"
        for name, _parameter in candidate.future_seed_producer_codec.named_parameters()
    }
    if set(missing) != expected or unexpected:
        raise AssertionError(
            f"unexpected state migration: missing={missing}, unexpected={unexpected}"
        )
    return sorted(missing)


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


def backward_once(
    candidate: FutureSeedRWKV,
    x: torch.Tensor,
    address: torch.Tensor,
    cell_order: torch.Tensor,
    target: torch.Tensor,
) -> tuple[dict[str, torch.Tensor], list[str]]:
    candidate.zero_grad(set_to_none=True)
    output, diagnostics, states = capture_terminal_states(
        candidate,
        x.detach().clone().requires_grad_(True),
        address,
        cell_order,
    )
    (output.float() * target).mean().backward()
    return diagnostics, graph_names(states[-1])


def check_zero_identity_and_learning_path(device: torch.device) -> dict[str, Any]:
    torch.manual_seed(9502)
    control = build(device, content_mode="terminal")
    torch.manual_seed(9502)
    candidate = build(device, content_mode="producer_codec")
    missing = copy_shared_state(control, candidate)
    control.eval()
    candidate.eval()

    x = torch.randn(2, 81, 256, device=device)
    address = torch.randn_like(x)
    cell_order = torch.randperm(81, device=device)
    with torch.no_grad():
        control_output, _control_diag, control_states = capture_terminal_states(
            control, x, address, cell_order
        )
        candidate_output, candidate_diag, candidate_states = capture_terminal_states(
            candidate, x, address, cell_order
        )
    if not torch.equal(control_output, candidate_output):
        raise AssertionError("zero-init producer codec changed the model output")
    state_errors = [
        float((left.float() - right.float()).abs().max().item())
        for left, right in zip(control_states, candidate_states)
    ]
    if len(state_errors) != len(candidate.blocks) or max(state_errors) != 0.0:
        raise AssertionError(f"zero-init terminal-state mismatch: {state_errors}")

    expected_delta = producer_codec_parameter_count(32, 32)
    parameter_delta = sum(p.numel() for p in candidate.parameters()) - sum(
        p.numel() for p in control.parameters()
    )
    if parameter_delta != expected_delta:
        raise AssertionError(f"parameter delta {parameter_delta} != {expected_delta}")

    candidate.train()
    target = torch.randn_like(candidate_output, dtype=torch.float32)
    diagnostics, graph = backward_once(
        candidate, x, address, cell_order, target
    )
    codec = candidate.future_seed_producer_codec
    assert codec is not None
    output_grad = codec.cell_decode_out.weight.grad
    if output_grad is None or not bool(torch.isfinite(output_grad).all()):
        raise AssertionError("zero-init decoder output gradient is missing or non-finite")
    output_grad_max = float(output_grad.float().abs().max().item())
    if output_grad_max <= 0.0:
        raise AssertionError("zero-init decoder output gradient is zero")
    with torch.no_grad():
        codec.cell_decode_out.weight.copy_(
            -0.01 * output_grad / output_grad.float().abs().max().clamp_min(1e-12)
        )

    diagnostics_after_open, graph_after_open = backward_once(
        candidate, x, address, cell_order, target
    )
    opened_gradient_max = {}
    for name, parameter in codec.named_parameters():
        if parameter.grad is None or not bool(torch.isfinite(parameter.grad).all()):
            raise AssertionError(f"opened codec gradient is missing or non-finite: {name}")
        value = float(parameter.grad.float().abs().max().item())
        if value <= 0.0:
            raise AssertionError(f"opened codec gradient is zero: {name}")
        opened_gradient_max[name] = value

    if (
        "ChunkGDN2FunctionBackward" not in graph
        or "ChunkGDN2FunctionBackward" not in graph_after_open
    ):
        raise AssertionError("official GDN2 chunk backward is missing")
    official_layers = [
        type(block.time_mix.core).__name__ for block in candidate.blocks
    ]
    if official_layers != ["GatedDeltaNet2"] * 12:
        raise AssertionError(f"unexpected recurrent layers: {official_layers}")
    if float(diagnostics["fs3_codec_enabled"].item()) != 1.0:
        raise AssertionError("producer-codec diagnostics did not mark the path active")
    if float(diagnostics["fs3_codec_residual_relative_rms"].item()) != 0.0:
        raise AssertionError("zero-init producer codec residual is not zero")
    if float(diagnostics_after_open["fs3_codec_residual_relative_rms"].item()) <= 0.0:
        raise AssertionError("opened producer codec residual is zero")
    if float(diagnostics["gdn2_address_enabled"].item()) != 1.0:
        raise AssertionError("position-QK address path is not active")
    return {
        "output_exact_identity": True,
        "terminal_state_max_abs_errors": state_errors,
        "missing_parameters": missing,
        "parameter_delta": parameter_delta,
        "zero_init_decoder_grad_max_abs": output_grad_max,
        "opened_gradient_max_abs": opened_gradient_max,
        "official_chunk_backward": True,
        "official_gdn2_layers": official_layers,
        "position_qk_active": True,
        "initial_diag": {
            key: float(value.detach().float().item())
            for key, value in candidate_diag.items()
            if key.startswith("fs3_codec_")
        },
    }


def check_bounds_and_permutation_equivariance(
    device: torch.device,
) -> dict[str, float]:
    torch.manual_seed(9503)
    codec = FutureSeedProducerCodec(row_dim=16, col_dim=16).to(device)
    codec.cell_decode_out.weight.data.normal_(mean=0.0, std=0.05)
    terminal = torch.randn(3, 4, 16, 16, device=device, dtype=torch.float32)
    incoming = torch.randn_like(terminal)
    baseline, diagnostics = codec(terminal, incoming)

    row_perm = torch.randperm(16, device=device)
    col_perm = torch.randperm(16, device=device)
    permuted, _ = codec(
        terminal.index_select(-2, row_perm).index_select(-1, col_perm),
        incoming.index_select(-2, row_perm).index_select(-1, col_perm),
    )
    inverse_row = torch.empty_like(row_perm)
    inverse_row[row_perm] = torch.arange(16, device=device)
    inverse_col = torch.empty_like(col_perm)
    inverse_col[col_perm] = torch.arange(16, device=device)
    restored = permuted.index_select(-2, inverse_row).index_select(-1, inverse_col)
    permutation_error = float(
        (baseline.float() - restored.float()).abs().max().item()
    )
    if permutation_error > 2e-6:
        raise AssertionError(
            f"producer codec lost permutation equivariance: {permutation_error}"
        )

    residual = baseline.float() - terminal.float()
    terminal_rms = (
        terminal.float().square().mean(dim=(-1, -2)).sqrt().clamp_min(1e-6)
    )
    residual_rms = residual.square().mean(dim=(-1, -2)).sqrt()
    max_relative_rms = float((residual_rms / terminal_rms).max().item())
    if max_relative_rms > 1.01:
        raise AssertionError(f"producer codec exceeded its RMS bound: {max_relative_rms}")
    return {
        "row_and_column_permutation_max_abs_error": permutation_error,
        "max_residual_relative_rms": max_relative_rms,
        "row_attention_entropy": float(
            diagnostics["fs3_codec_row_attention_entropy"].item()
        ),
        "row_attention_max": float(
            diagnostics["fs3_codec_row_attention_max"].item()
        ),
    }


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required; CPU smoke is forbidden")
    if torch.cuda.device_count() != 1:
        raise RuntimeError(
            f"exactly one visible CUDA device is required, got {torch.cuda.device_count()}"
        )
    visible_gpu = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader"],
        text=True,
    ).strip().splitlines()
    if visible_gpu != [f"0, {EXPECTED_GPU_UUID}"]:
        raise RuntimeError(f"unexpected visible GPU contract: {visible_gpu}")
    device = torch.device("cuda:0")
    if GAIN_BUDGET_FLA_SHA != EXPECTED_FLA_SOURCE_SHA:
        raise RuntimeError(
            f"pinned FLA SHA drift: {GAIN_BUDGET_FLA_SHA} != {EXPECTED_FLA_SOURCE_SHA}"
        )
    props = torch.cuda.get_device_properties(device)
    output = {
        "device": str(device),
        "gpu_name": props.name,
        "gpu_uuid": EXPECTED_GPU_UUID,
        "fla_source_sha": GAIN_BUDGET_FLA_SHA,
        "zero_identity_and_learning_path": check_zero_identity_and_learning_path(
            device
        ),
        "bounds_and_permutation_equivariance": (
            check_bounds_and_permutation_equivariance(device)
        ),
    }
    print(json.dumps(output, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
