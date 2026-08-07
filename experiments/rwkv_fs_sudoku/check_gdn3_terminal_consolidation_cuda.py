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
EXPECTED_FLA_SOURCE_SHA = "9c8e42e762fce087c27b673af4922795d9edb85e"
LAYERS = 12
PRODUCER_LAYERS = LAYERS - 1
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
        queue.extend(
            next_fn for next_fn, _ in fn.next_functions if next_fn is not None
        )
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
    missing, unexpected = candidate.load_state_dict(
        control.state_dict(), strict=False
    )
    expected = {
        f"blocks.{layer}.time_mix.terminal_consolidation_k_proj.weight"
        for layer in range(PRODUCER_LAYERS)
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
        hooks.append(
            block.time_mix.register_forward_hook(capture, with_kwargs=True)
        )
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


def direct_block_forward(
    block: torch.nn.Module,
    x: torch.Tensor,
    address: torch.Tensor,
    cell_order: torch.Tensor,
    initial_state: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        return block.time_mix(
            x,
            initial_state=initial_state,
            address=address,
            cell_order=cell_order,
        )


def check_initial_state_identity(
    control: FutureSeedRWKV,
    candidate: FutureSeedRWKV,
    device: torch.device,
) -> list[dict[str, float]]:
    rows = []
    torch.manual_seed(9806)
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
        with torch.no_grad():
            control_output, control_state = direct_block_forward(
                control_block, x, address, cell_order, initial_state
            )
            candidate_output, candidate_state = direct_block_forward(
                candidate_block, x, address, cell_order, initial_state
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


def open_projections_from_full_model_gradient(
    candidate: FutureSeedRWKV,
    x: torch.Tensor,
    address: torch.Tensor,
    cell_order: torch.Tensor,
) -> dict[str, float]:
    candidate.train()
    candidate.zero_grad(set_to_none=True)
    output, _diagnostics, states = capture_terminal_states(
        candidate, x, address, cell_order
    )
    chunk_counts = [
        graph_names(state).count("ChunkGDN2FunctionBackward")
        for state in states
    ]
    expected_counts = [2] * PRODUCER_LAYERS + [1]
    if chunk_counts != expected_counts:
        raise AssertionError(
            f"official parent/correction graph mismatch: {chunk_counts}"
        )
    output.float().square().mean().backward()

    gradient_max: dict[str, float] = {}
    with torch.no_grad():
        for layer, block in enumerate(candidate.blocks):
            projection = block.time_mix.terminal_consolidation_k_proj
            if layer == LAYERS - 1:
                if projection is not None:
                    raise AssertionError("final layer unexpectedly has a refiner")
                continue
            if projection is None:
                raise AssertionError(f"producer layer {layer} refiner is missing")
            gradient = projection.weight.grad
            if gradient is None or not bool(torch.isfinite(gradient).all()):
                raise AssertionError(
                    f"producer layer {layer} gradient is missing/non-finite"
                )
            value = float(gradient.float().abs().max().item())
            if value <= 0.0:
                raise AssertionError(f"producer layer {layer} gradient is zero")
            gradient_max[str(layer)] = value
            projection.weight.add_(
                -0.01 * gradient / gradient.float().abs().max().clamp_min(1e-8)
            )
    return gradient_max


def check_equivariance_and_dependencies(
    candidate: FutureSeedRWKV,
    device: torch.device,
) -> dict[str, float]:
    torch.manual_seed(9808)
    time_mix = candidate.blocks[0].time_mix
    batch, tokens = 3, 17
    q = torch.randn(
        batch, tokens, HEADS, HEAD_DIM, device=device, dtype=torch.bfloat16
    )
    main_k = torch.randn_like(q)
    first_output = torch.randn_like(q)
    v = torch.randn_like(q)
    b = torch.sigmoid(torch.randn_like(q))
    w = torch.sigmoid(torch.randn_like(q))
    state = torch.randn(
        batch,
        HEADS,
        HEAD_DIM,
        HEAD_DIM,
        device=device,
        dtype=torch.float32,
    )
    permutation = torch.randperm(HEADS, device=device)
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        base_state, _base_diag = time_mix._terminal_consolidation_transition(
            q, main_k, first_output, v, b, w, state
        )
        permuted_state, _permuted_diag = (
            time_mix._terminal_consolidation_transition(
                q[:, :, permutation],
                main_k[:, :, permutation],
                first_output[:, :, permutation],
                v[:, :, permutation],
                b[:, :, permutation],
                w[:, :, permutation],
                state[:, permutation],
            )
        )
    inverse = torch.argsort(permutation)
    equivariance_error = float(
        (base_state.float() - permuted_state[:, inverse].float())
        .abs()
        .max()
        .item()
    )
    if equivariance_error > 2e-6:
        raise AssertionError(f"head permutation error: {equivariance_error}")

    base_residual = base_state.float() - state.float()
    rolled_state = state.roll(1, dims=0)
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        shuffled_state, _ = time_mix._terminal_consolidation_transition(
            q, main_k, first_output, v, b, w, rolled_state
        )
        shuffled_source_state, _ = time_mix._terminal_consolidation_transition(
            q,
            main_k,
            first_output.roll(1, dims=0),
            v,
            b,
            w,
            state,
        )
    state_dependency = float(
        (
            base_residual
            - (shuffled_state.float() - rolled_state.float())
        )
        .square()
        .mean()
        .sqrt()
        .item()
    )
    source_dependency = float(
        (base_state.float() - shuffled_source_state.float())
        .square()
        .mean()
        .sqrt()
        .item()
    )
    if state_dependency <= 0.0 or source_dependency <= 0.0:
        raise AssertionError(
            "opened consolidation lacks state/source dependency: "
            f"{state_dependency}, {source_dependency}"
        )
    return {
        "head_permutation_max_abs_error": equivariance_error,
        "shuffled_terminal_residual_rms": state_dependency,
        "shuffled_first_output_state_rms": source_dependency,
    }


def check_zero_identity_and_learning_path(device: torch.device) -> dict[str, Any]:
    torch.manual_seed(9805)
    control = build(device, update_mode="none")
    torch.manual_seed(9805)
    candidate = build(device, update_mode="terminal_consolidation")
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
        candidate_output, candidate_diag, candidate_states = (
            capture_terminal_states(candidate, x, address, cell_order)
        )
    if not torch.equal(control_output, candidate_output):
        raise AssertionError("zero-init consolidation changed the model output")
    state_errors = [
        float((left.float() - right.float()).abs().max().item())
        for left, right in zip(control_states, candidate_states)
    ]
    if len(state_errors) != LAYERS or max(state_errors) != 0.0:
        raise AssertionError(f"zero-init terminal-state mismatch: {state_errors}")

    expected_delta = PRODUCER_LAYERS * HEAD_DIM * HEAD_DIM
    parameter_delta = sum(p.numel() for p in candidate.parameters()) - sum(
        p.numel() for p in control.parameters()
    )
    if parameter_delta != expected_delta:
        raise AssertionError(f"parameter delta {parameter_delta} != {expected_delta}")

    initial_state_rows = check_initial_state_identity(control, candidate, device)
    gradient_max = open_projections_from_full_model_gradient(
        candidate, x, address, cell_order
    )
    candidate.eval()
    with torch.no_grad():
        _opened_output, opened_diag, opened_states = capture_terminal_states(
            candidate, x, address, cell_order
        )
    active_layers = sum(
        int(
            float(
                block.time_mix.last_gain_budget_diag.get(
                    "gdn3_terminal_consolidation_enabled",
                    x.new_zeros(()),
                ).item()
            )
            > 0.0
        )
        for block in candidate.blocks
    )
    if active_layers != PRODUCER_LAYERS:
        raise AssertionError(
            f"expected {PRODUCER_LAYERS} producer refiners, got {active_layers}"
        )
    for key in (
        "gdn3_terminal_consolidation_k_relative_rms",
        "gdn3_terminal_consolidation_k_batch_std",
        "gdn3_terminal_consolidation_k_token_std",
        "gdn3_terminal_consolidation_state_residual_relative_rms",
        "gdn3_terminal_consolidation_state_residual_batch_std",
        "gdn3_terminal_consolidation_output_rms",
        "gdn3_terminal_consolidation_output_token_std",
        "gdn3_terminal_consolidation_weight_rms",
    ):
        value = float(opened_diag[key].item())
        if not torch.isfinite(torch.tensor(value)) or value <= 0.0:
            raise AssertionError(
                f"opened terminal-consolidation metric {key} is invalid: {value}"
            )
    if all(
        torch.equal(left, right)
        for left, right in zip(candidate_states[:PRODUCER_LAYERS], opened_states)
    ):
        raise AssertionError("opened consolidation did not alter terminal states")

    official_layers = [type(block.time_mix.core).__name__ for block in candidate.blocks]
    if official_layers != ["GatedDeltaNet2"] * LAYERS:
        raise AssertionError(f"unexpected recurrent layers: {official_layers}")
    if float(candidate_diag["gdn2_address_enabled"].item()) != 1.0:
        raise AssertionError("position-QK address path is not active")
    if (
        float(
            candidate_diag[
                "gdn3_terminal_consolidation_state_residual_relative_rms"
            ].item()
        )
        != 0.0
    ):
        raise AssertionError("zero-init consolidation residual is not zero")
    return {
        "output_exact_identity": True,
        "terminal_state_max_abs_errors": state_errors,
        "initial_state_identity": initial_state_rows,
        "missing_parameters": missing,
        "parameter_delta": parameter_delta,
        "projection_gradient_max_by_layer": gradient_max,
        "active_producer_layers": active_layers,
        "official_parent_layers": official_layers,
        "official_chunk_backward_paths": 2 * PRODUCER_LAYERS + 1,
        "position_qk_active": True,
        "opened_diagnostics": {
            key: float(value.detach().float().item())
            for key, value in opened_diag.items()
            if key.startswith("gdn3_terminal_consolidation_")
        },
        "equivariance_and_dependencies": check_equivariance_and_dependencies(
            candidate, device
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
    }
    print(json.dumps(output, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
