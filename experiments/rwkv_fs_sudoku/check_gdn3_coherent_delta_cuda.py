#!/usr/bin/env python3
from __future__ import annotations

import json
import math
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
EXPECTED_FLA_SOURCE_SHA = "9c8e42e762fce087c27b673af4922795d9edb85e"
LAYERS = 12
HEADS = 8
HEAD_DIM = 32


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


def build(device: torch.device, *, update_mode: str) -> FutureSeedRWKV:
    return FutureSeedRWKV(
        256,
        LAYERS,
        HEADS,
        HEAD_DIM,
        4,
        future_seed_scale=1.0,
        future_seed_decay=0.0,
        future_seed_update="fixed",
        future_seed_norm_mode="unit",
        future_seed_gate_mode="head",
        future_seed_scope="layer",
        future_seed_readout_hop=0,
        future_seed_content_mode="terminal",
        backbone="gdn2",
        gdn_mode="chunk",
        gdn_expand_v=1.0,
        gdn_use_short_conv=True,
        gdn_conv_size=4,
        gdn_allow_neg_eigval=False,
        gdn2_address_mode="position_qk",
        gdn2_update_mode=update_mode,
        gdn2_cross_layer_init="independent",
    ).to(device)


def copy_shared_state(
    control: FutureSeedRWKV,
    candidate: FutureSeedRWKV,
) -> list[str]:
    missing, unexpected = candidate.load_state_dict(control.state_dict(), strict=False)
    expected = {
        f"blocks.{layer}.time_mix.coherent_delta_mix" for layer in range(LAYERS)
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


def check_initial_state_identity(
    control: FutureSeedRWKV,
    candidate: FutureSeedRWKV,
    device: torch.device,
) -> list[dict[str, float]]:
    rows = []
    torch.manual_seed(9606)
    for layer, (control_block, candidate_block) in enumerate(
        zip(control.blocks, candidate.blocks)
    ):
        x = torch.randn(1, 81, 256, device=device)
        address = torch.randn_like(x)
        cell_order = torch.randperm(81, device=device)
        initial_state = torch.randn(
            1,
            HEADS,
            HEAD_DIM,
            HEAD_DIM,
            device=device,
            dtype=torch.float32,
        )
        with torch.no_grad(), torch.autocast(
            device_type="cuda", dtype=torch.bfloat16
        ):
            control_output, control_state = control_block.time_mix(
                x,
                initial_state=initial_state,
                address=address,
                cell_order=cell_order,
            )
            candidate_output, candidate_state = candidate_block.time_mix(
                x,
                initial_state=initial_state,
                address=address,
                cell_order=cell_order,
            )
        output_error = float(
            (control_output.float() - candidate_output.float()).abs().max().item()
        )
        state_error = float(
            (control_state.float() - candidate_state.float()).abs().max().item()
        )
        if output_error != 0.0 or state_error != 0.0:
            raise AssertionError(
                f"layer {layer} initial-state identity failed: "
                f"output={output_error} state={state_error}"
            )
        rows.append(
            {
                "layer": float(layer),
                "output_max_abs_error": output_error,
                "terminal_state_max_abs_error": state_error,
            }
        )
    return rows


def check_zero_identity_and_learning_path(device: torch.device) -> dict[str, Any]:
    torch.manual_seed(9605)
    control = build(device, update_mode="none")
    torch.manual_seed(9605)
    candidate = build(device, update_mode="coherent_delta")
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
        raise AssertionError("zero-init coherent delta changed the model output")
    state_errors = [
        float((left.float() - right.float()).abs().max().item())
        for left, right in zip(control_states, candidate_states)
    ]
    if len(state_errors) != LAYERS or max(state_errors) != 0.0:
        raise AssertionError(f"zero-init terminal-state mismatch: {state_errors}")

    expected_delta = LAYERS * HEADS
    parameter_delta = sum(p.numel() for p in candidate.parameters()) - sum(
        p.numel() for p in control.parameters()
    )
    if parameter_delta != expected_delta:
        raise AssertionError(f"parameter delta {parameter_delta} != {expected_delta}")

    initial_state_rows = check_initial_state_identity(control, candidate, device)
    candidate.train()
    candidate.zero_grad(set_to_none=True)
    output, diagnostics, states = capture_terminal_states(
        candidate,
        x.detach().clone().requires_grad_(True),
        address,
        cell_order,
    )
    target = torch.randn_like(output, dtype=torch.float32)
    (output.float() * target).mean().backward()
    gradients = {}
    for layer, block in enumerate(candidate.blocks):
        parameter = block.time_mix.coherent_delta_mix
        if parameter is None or parameter.grad is None:
            raise AssertionError(f"layer {layer} coherent delta gradient is missing")
        if not bool(torch.isfinite(parameter.grad).all()):
            raise AssertionError(f"layer {layer} coherent delta gradient is non-finite")
        grad_max = float(parameter.grad.float().abs().max().item())
        if grad_max <= 0.0:
            raise AssertionError(f"layer {layer} coherent delta gradient is zero")
        gradients[str(layer)] = grad_max
    graph = graph_names(states[-1])
    if "ChunkGDN2FunctionBackward" not in graph:
        raise AssertionError("official GDN2 chunk backward is missing")

    official_layers = [type(block.time_mix.core).__name__ for block in candidate.blocks]
    if official_layers != ["GatedDeltaNet2"] * LAYERS:
        raise AssertionError(f"unexpected recurrent layers: {official_layers}")
    if float(diagnostics["gdn2_address_enabled"].item()) != 1.0:
        raise AssertionError("position-QK address path is not active")
    if float(diagnostics["gdn2_coherent_delta_enabled"].item()) != 1.0:
        raise AssertionError("coherent delta diagnostics did not mark the path active")
    if float(diagnostics["gdn2_coherent_delta_mix_abs"].item()) != 0.0:
        raise AssertionError("zero-init coherent delta mix is not zero")
    return {
        "output_exact_identity": True,
        "terminal_state_max_abs_errors": state_errors,
        "initial_state_identity": initial_state_rows,
        "missing_parameters": missing,
        "parameter_delta": parameter_delta,
        "gradient_max_abs_by_layer": gradients,
        "official_chunk_backward": True,
        "official_gdn2_layers": official_layers,
        "position_qk_active": True,
        "zero_init_diag": {
            key: float(value.detach().float().item())
            for key, value in candidate_diag.items()
            if key.startswith("gdn2_coherent_delta_")
        },
    }


def check_gate_contract(device: torch.device) -> dict[str, float]:
    torch.manual_seed(9607)
    model = build(device, update_mode="coherent_delta")
    block = model.blocks[0].time_mix
    assert block.coherent_delta_mix is not None
    with torch.no_grad():
        block.coherent_delta_mix.fill_(math.atanh(0.25))
        b_raw = torch.randn(3, 17, HEADS, HEAD_DIM, device=device)
        w_raw = torch.randn(3, 17, HEADS, HEAD_DIM, device=device)
        b, w, diag = block._coherent_delta_gates(b_raw, w_raw)
    gate_min = float(torch.minimum(b.float().min(), w.float().min()).item())
    gate_max = float(torch.maximum(b.float().max(), w.float().max()).item())
    gap_ratio = float(diag["gdn2_coherent_delta_gap_ratio"].item())
    if gate_min < 0.0 or gate_max > 1.0:
        raise AssertionError(f"coherent delta gate escaped [0,1]: {gate_min}, {gate_max}")
    if not 0.0 < gap_ratio < 1.0:
        raise AssertionError(f"coherent delta did not contract the gate gap: {gap_ratio}")
    return {
        "effective_mix": float(diag["gdn2_coherent_delta_mix_mean"].item()),
        "gate_min": gate_min,
        "gate_max": gate_max,
        "pre_gap_rms": float(diag["gdn2_coherent_delta_pre_gap_rms"].item()),
        "post_gap_rms": float(diag["gdn2_coherent_delta_post_gap_rms"].item()),
        "gap_ratio": gap_ratio,
        "b_relative_change": float(
            diag["gdn2_coherent_delta_b_relative_change"].item()
        ),
        "w_relative_change": float(
            diag["gdn2_coherent_delta_w_relative_change"].item()
        ),
    }


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required; CPU model smoke is forbidden")
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
        "gate_contract": check_gate_contract(device),
    }
    print(json.dumps(output, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
