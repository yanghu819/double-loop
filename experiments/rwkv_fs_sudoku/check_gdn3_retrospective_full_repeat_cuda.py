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

import study_rwkv_futureseed_loop as runner
from study_rwkv_futureseed_loop import FutureSeedRWKV, GAIN_BUDGET_FLA_SHA


EXPECTED_GPU_UUID = os.environ.get("EXPECTED_GPU_UUID", "").strip()
EXPECTED_FLA_SOURCE_SHA = "9c8e42e762fce087c27b673af4922795d9edb85e"
LAYERS = 10
RECEIVING_LAYERS = LAYERS - 1
HEADS = 6
HEAD_DIM = 32
MODEL_DIM = 192
TOKENS = 81


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
        queue.extend(
            next_fn for next_fn, _ in fn.next_functions if next_fn is not None
        )
    return names


def build(device: torch.device, *, update_mode: str) -> FutureSeedRWKV:
    return FutureSeedRWKV(
        MODEL_DIM,
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
        gdn2_address_mode="none",
        gdn2_update_mode=update_mode,
        gdn2_cross_layer_init="independent",
    ).to(device)


def capture_terminal_states(
    model: FutureSeedRWKV,
    x: torch.Tensor,
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
        hooks.append(
            block.time_mix.register_forward_hook(capture, with_kwargs=True)
        )
    try:
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            output, diagnostics, _ = model(x)
    finally:
        for hook in hooks:
            hook.remove()
    return output, diagnostics, states


def check_repeat_contract(device: torch.device) -> dict[str, Any]:
    torch.manual_seed(107401)
    control = build(device, update_mode="none")
    torch.manual_seed(107401)
    candidate = build(device, update_mode="retrospective_full_repeat")
    candidate.load_state_dict(control.state_dict(), strict=True)

    control_names = [name for name, _ in control.named_parameters()]
    candidate_names = [name for name, _ in candidate.named_parameters()]
    if control_names != candidate_names:
        raise AssertionError("full repeat changed the parameter schema")
    parameter_delta = sum(p.numel() for p in candidate.parameters()) - sum(
        p.numel() for p in control.parameters()
    )
    if parameter_delta != 0:
        raise AssertionError(f"parameter delta must be zero, got {parameter_delta}")

    captured: list[dict[str, Any]] = []
    original_chunk = runner.chunk_gdn2
    if original_chunk is None:
        raise RuntimeError("official chunk_gdn2 is unavailable")

    def checked_chunk(**kwargs: Any) -> tuple[torch.Tensor, torch.Tensor]:
        row: dict[str, Any] = {
            "length": int(kwargs["q"].shape[1]),
            "initial_state_rms": float(
                kwargs["initial_state"].float().square().mean().sqrt().item()
            ),
        }
        for key in ("q", "k", "v", "g", "b", "w"):
            stream = kwargs[key]
            if stream.shape[1] != 2 * TOKENS:
                raise AssertionError(
                    f"{key} repeat length {stream.shape[1]} != {2 * TOKENS}"
                )
            first, second = stream.split(TOKENS, dim=1)
            error = float((first.float() - second.float()).abs().max().item())
            if error != 0.0:
                raise AssertionError(f"{key} tied-repeat error is {error}")
            row[f"{key}_repeat_max_abs_error"] = error
        captured.append(row)
        return original_chunk(**kwargs)

    x = torch.randn(3, TOKENS, MODEL_DIM, device=device)
    candidate.train()
    candidate.zero_grad(set_to_none=True)
    runner.chunk_gdn2 = checked_chunk
    try:
        candidate_output, candidate_diag, candidate_states = (
            capture_terminal_states(candidate, x)
        )
    finally:
        runner.chunk_gdn2 = original_chunk

    if len(captured) != RECEIVING_LAYERS:
        raise AssertionError(
            f"expected {RECEIVING_LAYERS} repeat calls, got {len(captured)}"
        )
    if any(row["length"] != 2 * TOKENS for row in captured):
        raise AssertionError(f"unexpected repeat lengths: {captured}")
    if any(row["initial_state_rms"] <= 0.0 for row in captured):
        raise AssertionError("a receiving repeat path has an empty FutureSeed")

    chunk_paths = [
        graph_names(state).count("ChunkGDN2FunctionBackward")
        for state in candidate_states
    ]
    if chunk_paths[1:] != [1] * RECEIVING_LAYERS:
        raise AssertionError(f"receiving official chunk paths: {chunk_paths}")
    official_layers = [type(block.time_mix.core).__name__ for block in candidate.blocks]
    if official_layers != ["GatedDeltaNet2"] * LAYERS:
        raise AssertionError(f"unexpected recurrent layers: {official_layers}")

    loss = candidate_output.float().square().mean()
    loss = loss + 0.01 * torch.stack(
        [state.float().square().mean() for state in candidate_states]
    ).mean()
    loss.backward()
    gradient_rows: dict[str, dict[str, float]] = {}
    for layer, block in enumerate(candidate.blocks[1:], start=1):
        core = block.time_mix.core
        layer_gradients: dict[str, float] = {}
        projection_weights = (
            ("q_proj", core.q_proj.weight),
            ("k_proj", core.k_proj.weight),
            ("v_proj", core.v_proj.weight),
            ("f_proj_in", core.f_proj[0].weight),
            ("f_proj_out", core.f_proj[1].weight),
            ("b_proj", core.b_proj.weight),
            ("w_proj", core.w_proj.weight),
        )
        for name, weight in projection_weights:
            gradient = weight.grad
            if gradient is None or not bool(torch.isfinite(gradient).all()):
                raise AssertionError(f"layer {layer} {name} gradient is invalid")
            maximum = float(gradient.float().abs().max().item())
            if maximum <= 0.0:
                raise AssertionError(f"layer {layer} {name} gradient is zero")
            layer_gradients[name] = maximum
        gradient_rows[str(layer)] = layer_gradients

    active_paths = int(
        candidate_diag["gdn3_retrospective_full_repeat_path_count_sum"].item()
    )
    if active_paths != RECEIVING_LAYERS:
        raise AssertionError(
            f"expected {RECEIVING_LAYERS} active paths, got {active_paths}"
        )
    activation_keys = (
        "gdn3_retrospective_full_repeat_output_relative_rms",
        "gdn3_retrospective_full_repeat_output_batch_std",
        "gdn3_retrospective_full_repeat_first_output_rms",
        "gdn3_retrospective_full_repeat_second_output_rms",
        "gdn3_retrospective_full_repeat_terminal_rms",
        "gdn3_retrospective_full_repeat_terminal_batch_std",
    )
    activation = {
        key: float(candidate_diag[key].detach().float().item())
        for key in activation_keys
    }
    if any(not torch.isfinite(torch.tensor(value)) for value in activation.values()):
        raise AssertionError(f"non-finite repeat activation: {activation}")
    if activation["gdn3_retrospective_full_repeat_output_relative_rms"] < 1e-4:
        raise AssertionError(f"repeat path is inactive: {activation}")
    if activation["gdn3_retrospective_full_repeat_output_batch_std"] <= 0.0:
        raise AssertionError(f"repeat path lacks board variation: {activation}")

    control.eval()
    with torch.no_grad():
        control_output, _control_diag, control_states = capture_terminal_states(
            control, x.detach()
        )
    output_delta = float(
        (candidate_output.detach().float() - control_output.float())
        .square()
        .mean()
        .sqrt()
        .item()
    )
    if output_delta <= 0.0:
        raise AssertionError("full repeat did not change the model output")
    terminal_ratios = [
        float(
            candidate_state.detach().float().square().mean().sqrt().item()
            / control_state.float().square().mean().sqrt().clamp_min(1e-8).item()
        )
        for candidate_state, control_state in zip(candidate_states, control_states)
    ]
    if not all(torch.isfinite(torch.tensor(terminal_ratios))):
        raise AssertionError(f"non-finite state ratios: {terminal_ratios}")
    if max(terminal_ratios) > 8.0:
        raise AssertionError(f"repeat state ratio exceeds 8x: {terminal_ratios}")

    return {
        "parameter_delta": parameter_delta,
        "active_receiving_paths": active_paths,
        "captured_repeat_calls": captured,
        "official_layers": official_layers,
        "chunk_backward_paths_by_layer": chunk_paths,
        "activation": activation,
        "candidate_control_output_delta_rms": output_delta,
        "candidate_control_terminal_rms_ratios": terminal_ratios,
        "gradient_max_by_layer": gradient_rows,
    }


def main() -> None:
    if not EXPECTED_GPU_UUID:
        raise RuntimeError("EXPECTED_GPU_UUID must be set explicitly")
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
    if GAIN_BUDGET_FLA_SHA != EXPECTED_FLA_SOURCE_SHA:
        raise RuntimeError(
            f"pinned FLA SHA drift: {GAIN_BUDGET_FLA_SHA} != {EXPECTED_FLA_SOURCE_SHA}"
        )
    device = torch.device("cuda:0")
    output = {
        "device": str(device),
        "gpu_name": torch.cuda.get_device_properties(device).name,
        "gpu_uuid": EXPECTED_GPU_UUID,
        "fla_source_sha": GAIN_BUDGET_FLA_SHA,
        "repeat_contract": check_repeat_contract(device),
    }
    print(json.dumps(output, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
