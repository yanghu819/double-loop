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


EXPECTED_GPU_UUID = os.environ.get("EXPECTED_GPU_UUID", "").strip()
EXPECTED_FLA_SOURCE_SHA = "9c8e42e762fce087c27b673af4922795d9edb85e"
LAYERS = 10
RECEIVING_LAYERS = LAYERS - 1
HEADS = 6
HEAD_DIM = 32
MODEL_DIM = 192


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


def copy_shared_state(
    control: FutureSeedRWKV,
    candidate: FutureSeedRWKV,
) -> list[str]:
    missing, unexpected = candidate.load_state_dict(
        control.state_dict(), strict=False
    )
    expected = {
        f"blocks.{layer}.time_mix.boundary_recommit_mix"
        for layer in range(LAYERS)
    }
    if set(missing) != expected or unexpected:
        raise AssertionError(
            f"unexpected state migration: missing={missing}, unexpected={unexpected}"
        )
    return sorted(missing)


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


def direct_block_forward(
    block: torch.nn.Module,
    x: torch.Tensor,
    initial_state: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        return block.time_mix(x, initial_state=initial_state)


def check_nonzero_initial_state_identity(
    control: FutureSeedRWKV,
    candidate: FutureSeedRWKV,
    device: torch.device,
) -> list[dict[str, float]]:
    rows = []
    torch.manual_seed(107206)
    for layer, (control_block, candidate_block) in enumerate(
        zip(control.blocks, candidate.blocks)
    ):
        x = torch.randn(1, 81, MODEL_DIM, device=device)
        initial_state = torch.randn(
            1,
            HEADS,
            HEAD_DIM,
            HEAD_DIM,
            device=device,
            dtype=torch.float32,
        )
        with torch.no_grad():
            control_output, control_state = direct_block_forward(
                control_block, x, initial_state
            )
            candidate_output, candidate_state = direct_block_forward(
                candidate_block, x, initial_state
            )
        output_error = float(
            (control_output.float() - candidate_output.float()).abs().max().item()
        )
        state_error = float(
            (control_state.float() - candidate_state.float()).abs().max().item()
        )
        if output_error != 0.0 or state_error != 0.0:
            raise AssertionError(
                f"layer {layer} zero-gate identity failed: "
                f"output={output_error} state={state_error}"
            )
        control_paths = graph_names(control_state).count("ChunkGDN2FunctionBackward")
        candidate_paths = graph_names(candidate_state).count(
            "ChunkGDN2FunctionBackward"
        )
        if control_paths != 1 or candidate_paths != 2:
            raise AssertionError(
                f"layer {layer} official chunk graph mismatch: "
                f"control={control_paths} candidate={candidate_paths}"
            )
        rows.append(
            {
                "layer": float(layer),
                "output_max_abs_error": output_error,
                "terminal_state_max_abs_error": state_error,
                "control_chunk_backward_paths": float(control_paths),
                "candidate_chunk_backward_paths": float(candidate_paths),
            }
        )
    return rows


def open_mix_from_direct_gradients(
    candidate: FutureSeedRWKV,
    device: torch.device,
) -> dict[str, float]:
    torch.manual_seed(107207)
    candidate.train()
    candidate.zero_grad(set_to_none=True)
    losses = []
    for block in candidate.blocks:
        x = torch.randn(1, 81, MODEL_DIM, device=device)
        initial_state = torch.randn(
            1,
            HEADS,
            HEAD_DIM,
            HEAD_DIM,
            device=device,
            dtype=torch.float32,
        )
        output, terminal_state = direct_block_forward(block, x, initial_state)
        losses.append(
            output.float().square().mean()
            + 0.01 * terminal_state.float().square().mean()
        )
    torch.stack(losses).mean().backward()

    gradients: dict[str, float] = {}
    with torch.no_grad():
        for layer, block in enumerate(candidate.blocks):
            mix = block.time_mix.boundary_recommit_mix
            if mix is None:
                raise AssertionError(f"layer {layer} recommit mix is missing")
            gradient = mix.grad
            if gradient is None or not bool(torch.isfinite(gradient).all()):
                raise AssertionError(
                    f"layer {layer} recommit gradient is missing/non-finite"
                )
            gradient_max = float(gradient.float().abs().max().item())
            if gradient_max <= 0.0:
                raise AssertionError(f"layer {layer} recommit gradient is zero")
            gradients[str(layer)] = gradient_max
            mix.copy_(torch.linspace(0.04, 0.10, HEADS, device=device))
    return gradients


def check_geometry_and_dependencies(
    candidate: FutureSeedRWKV,
    device: torch.device,
) -> dict[str, float]:
    torch.manual_seed(107208)
    time_mix = candidate.blocks[1].time_mix
    inherited = torch.randn(
        3,
        HEADS,
        HEAD_DIM,
        HEAD_DIM,
        device=device,
        dtype=torch.float32,
    )
    live = torch.randn_like(inherited)
    mix = time_mix.boundary_recommit_mix
    assert mix is not None
    original_mix = mix.detach().clone()
    permutation = torch.randperm(HEADS, device=device)
    with torch.no_grad():
        base_state, base_diag = time_mix._boundary_recommit_state(
            inherited, live
        )
        mix.copy_(original_mix[permutation])
        permuted_state, _ = time_mix._boundary_recommit_state(
            inherited[:, permutation], live[:, permutation]
        )
        mix.copy_(original_mix)
        inherited_shuffle, _ = time_mix._boundary_recommit_state(
            inherited.roll(1, dims=0), live
        )
        live_shuffle, _ = time_mix._boundary_recommit_state(
            inherited, live.roll(1, dims=0)
        )

    inverse = torch.argsort(permutation)
    equivariance_error = float(
        (base_state.float() - permuted_state[:, inverse].float())
        .abs()
        .max()
        .item()
    )
    inherited_dependency = float(
        (base_state.float() - inherited_shuffle.float())
        .square()
        .mean()
        .sqrt()
        .item()
    )
    live_dependency = float(
        (base_state.float() - live_shuffle.float())
        .square()
        .mean()
        .sqrt()
        .item()
    )
    residual = base_state.float() - live.float()
    orthogonality_error = float(
        (residual * live.float()).sum(dim=(-1, -2)).abs().max().item()
        / live.float().square().sum(dim=(-1, -2)).max().clamp_min(1e-6).item()
    )
    boundary_ratio = float(
        base_diag["gdn3_boundary_recommit_boundary_norm_ratio"].item()
    )
    if equivariance_error > 1e-6:
        raise AssertionError(f"head permutation error: {equivariance_error}")
    if inherited_dependency <= 0.0 or live_dependency <= 0.0:
        raise AssertionError(
            "opened recommit lacks inherited/live-state dependency: "
            f"{inherited_dependency}, {live_dependency}"
        )
    if orthogonality_error > 2e-6:
        raise AssertionError(
            f"recommit residual is not orthogonal to live state: {orthogonality_error}"
        )
    if not 1.0 <= boundary_ratio <= 2.0:
        raise AssertionError(f"boundary norm ratio is not bounded: {boundary_ratio}")
    return {
        "head_permutation_max_abs_error": equivariance_error,
        "inherited_state_dependency_rms": inherited_dependency,
        "live_state_dependency_rms": live_dependency,
        "residual_live_relative_dot_error": orthogonality_error,
        "boundary_norm_ratio_max": boundary_ratio,
    }


def check_zero_identity_and_learning_path(device: torch.device) -> dict[str, Any]:
    torch.manual_seed(107205)
    control = build(device, update_mode="none")
    torch.manual_seed(107205)
    candidate = build(device, update_mode="boundary_recommit")
    missing = copy_shared_state(control, candidate)
    control.eval()
    candidate.eval()

    x = torch.randn(2, 81, MODEL_DIM, device=device)
    with torch.no_grad():
        control_output, _control_diag, control_states = capture_terminal_states(
            control, x
        )
        candidate_output, candidate_diag, candidate_states = (
            capture_terminal_states(candidate, x)
        )
    if not torch.equal(control_output, candidate_output):
        error = float(
            (control_output.float() - candidate_output.float()).abs().max().item()
        )
        raise AssertionError(f"zero-gate model output identity failed: {error}")
    state_errors = [
        float((left.float() - right.float()).abs().max().item())
        for left, right in zip(control_states, candidate_states)
    ]
    if len(state_errors) != LAYERS or max(state_errors) != 0.0:
        raise AssertionError(f"zero-gate terminal-state mismatch: {state_errors}")
    parameter_delta = sum(p.numel() for p in candidate.parameters()) - sum(
        p.numel() for p in control.parameters()
    )
    expected_delta = LAYERS * HEADS
    if parameter_delta != expected_delta:
        raise AssertionError(f"parameter delta {parameter_delta} != {expected_delta}")

    initial_state_rows = check_nonzero_initial_state_identity(
        control, candidate, device
    )
    gradients = open_mix_from_direct_gradients(candidate, device)
    candidate.eval()
    with torch.no_grad():
        opened_output, opened_diag, opened_states = capture_terminal_states(
            candidate, x
        )
    if torch.equal(candidate_output, opened_output) or all(
        torch.equal(left, right)
        for left, right in zip(candidate_states, opened_states)
    ):
        raise AssertionError("opened recommit did not alter output and states")
    active_layers = sum(
        int(
            float(
                block.time_mix.last_gain_budget_diag.get(
                    "gdn3_boundary_recommit_enabled",
                    x.new_zeros(()),
                ).item()
            )
            > 0.0
        )
        for block in candidate.blocks
    )
    if active_layers != RECEIVING_LAYERS:
        raise AssertionError(
            f"expected {RECEIVING_LAYERS} active receiving paths, got {active_layers}"
        )
    for key in (
        "gdn3_boundary_recommit_gate_abs",
        "gdn3_boundary_recommit_gate_head_std",
        "gdn3_boundary_recommit_missing_fraction",
        "gdn3_boundary_recommit_missing_batch_std",
        "gdn3_boundary_recommit_residual_relative_rms",
        "gdn3_boundary_recommit_residual_batch_std",
        "gdn3_boundary_recommit_terminal_rms",
        "gdn3_boundary_recommit_terminal_batch_std",
        "gdn3_boundary_recommit_weight_rms",
    ):
        value = float(opened_diag[key].item())
        if not torch.isfinite(torch.tensor(value)) or value <= 0.0:
            raise AssertionError(f"opened recommit metric {key} is invalid: {value}")
    zero_residual = float(
        candidate_diag["gdn3_boundary_recommit_residual_relative_rms"].item()
    )
    if zero_residual != 0.0:
        raise AssertionError(f"zero-gate recommit residual is nonzero: {zero_residual}")

    official_layers = [type(block.time_mix.core).__name__ for block in candidate.blocks]
    if official_layers != ["GatedDeltaNet2"] * LAYERS:
        raise AssertionError(f"unexpected recurrent layers: {official_layers}")
    terminal_max = max(
        float(state.float().square().mean().sqrt().item())
        for state in opened_states
    )
    if not torch.isfinite(torch.tensor(terminal_max)):
        raise AssertionError(f"non-finite terminal state RMS: {terminal_max}")
    return {
        "output_exact_identity": True,
        "terminal_state_max_abs_errors": state_errors,
        "nonzero_initial_state_identity": initial_state_rows,
        "missing_parameters": missing,
        "parameter_delta": parameter_delta,
        "mix_gradient_max_by_layer": gradients,
        "active_receiving_paths": active_layers,
        "official_layers": official_layers,
        "candidate_direct_chunk_backward_paths": 2 * LAYERS,
        "opened_terminal_rms_max": terminal_max,
        "opened_diagnostics": {
            key: float(value.detach().float().item())
            for key, value in opened_diag.items()
            if key.startswith("gdn3_boundary_recommit_")
        },
        "geometry_and_dependencies": check_geometry_and_dependencies(
            candidate, device
        ),
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
    props = torch.cuda.get_device_properties(device)
    output = {
        "device": str(device),
        "gpu_name": props.name,
        "gpu_uuid": EXPECTED_GPU_UUID,
        "fla_source_sha": GAIN_BUDGET_FLA_SHA,
        "zero_identity_and_learning_path": check_zero_identity_and_learning_path(
            device
        ),
    }
    print(json.dumps(output, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
