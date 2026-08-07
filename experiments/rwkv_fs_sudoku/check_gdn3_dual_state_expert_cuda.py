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
HEADS = 8
MAIN_HEAD_DIM = 32
EXPERT_HEAD_DIM = 16
EXPECTED_PARAMETER_DELTA = 2_479_488
EXPECTED_STATE_VALUES_PER_LAYER = HEADS * EXPERT_HEAD_DIM * EXPERT_HEAD_DIM


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
            next_fn
            for next_fn, _ in fn.next_functions
            if next_fn is not None
        )
    return names


def build(device: torch.device, *, state_expert_mode: str) -> FutureSeedRWKV:
    return FutureSeedRWKV(
        256,
        LAYERS,
        HEADS,
        MAIN_HEAD_DIM,
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
        gdn2_update_mode="none",
        gdn2_state_expert_mode=state_expert_mode,
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
        name
        for name in candidate.state_dict()
        if ".state_expert." in name
    }
    if set(missing) != expected or unexpected:
        raise AssertionError(
            f"unexpected state migration: missing={missing}, unexpected={unexpected}"
        )
    for layer in range(LAYERS):
        prefix = f"blocks.{layer}.state_expert."
        if not any(name.startswith(prefix) for name in expected):
            raise AssertionError(f"layer {layer} has no state-expert migration keys")
    return sorted(missing)


def capture_states(
    model: FutureSeedRWKV,
    x: torch.Tensor,
    address: torch.Tensor,
    cell_order: torch.Tensor,
) -> tuple[
    torch.Tensor,
    dict[str, torch.Tensor],
    list[torch.Tensor],
    list[torch.Tensor],
]:
    main_states: list[torch.Tensor] = []
    expert_states: list[torch.Tensor] = []
    hooks = []

    def capture_main(
        _module: torch.nn.Module,
        _args: tuple[torch.Tensor, ...],
        _kwargs: dict[str, Any],
        output: tuple[torch.Tensor, torch.Tensor],
    ) -> None:
        main_states.append(output[1])

    def capture_expert(
        _module: torch.nn.Module,
        _args: tuple[torch.Tensor, ...],
        _kwargs: dict[str, Any],
        output: tuple[torch.Tensor, torch.Tensor],
    ) -> None:
        expert_states.append(output[1])

    for block in model.blocks:
        hooks.append(
            block.time_mix.register_forward_hook(
                capture_main, with_kwargs=True
            )
        )
        if block.state_expert is not None:
            hooks.append(
                block.state_expert.time_mix.register_forward_hook(
                    capture_expert, with_kwargs=True
                )
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
    return output, diagnostics, main_states, expert_states


def check_nonzero_initial_state_identity(
    control: FutureSeedRWKV,
    candidate: FutureSeedRWKV,
    device: torch.device,
) -> list[dict[str, float]]:
    rows = []
    torch.manual_seed(9608)
    for layer, (control_block, candidate_block) in enumerate(
        zip(control.blocks, candidate.blocks)
    ):
        x = torch.randn(1, 81, 256, device=device)
        address = torch.randn_like(x)
        cell_order = torch.randperm(81, device=device)
        main_state = torch.randn(
            1,
            HEADS,
            MAIN_HEAD_DIM,
            MAIN_HEAD_DIM,
            device=device,
            dtype=torch.float32,
        )
        expert_state = torch.randn(
            1,
            HEADS,
            EXPERT_HEAD_DIM,
            EXPERT_HEAD_DIM,
            device=device,
            dtype=torch.float32,
        )
        with torch.no_grad(), torch.autocast(
            device_type="cuda", dtype=torch.bfloat16
        ):
            control_output, control_state = control_block(
                x,
                initial_state=main_state,
                address=address,
                cell_order=cell_order,
            )
            candidate_output, candidate_state = candidate_block(
                x,
                initial_state=main_state,
                state_expert_initial_state=expert_state,
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
                f"layer {layer} incoming-state identity failed: "
                f"output={output_error} state={state_error}"
            )
        rows.append(
            {
                "layer": float(layer),
                "output_max_abs_error": output_error,
                "main_terminal_state_max_abs_error": state_error,
            }
        )
    return rows


def finite_nonzero_grad(parameter: torch.nn.Parameter, name: str) -> float:
    if parameter.grad is None:
        raise AssertionError(f"{name} gradient is missing")
    grad = parameter.grad.float()
    if not bool(torch.isfinite(grad).all()):
        raise AssertionError(f"{name} gradient is non-finite")
    grad_max = float(grad.abs().max().item())
    if grad_max <= 0.0:
        raise AssertionError(f"{name} gradient is zero")
    return grad_max


def check_two_stage_learning_path(
    candidate: FutureSeedRWKV,
    device: torch.device,
) -> dict[str, Any]:
    for parameter in candidate.parameters():
        parameter.requires_grad_(False)
    expert_parameters = [
        parameter
        for name, parameter in candidate.named_parameters()
        if ".state_expert." in name
    ]
    for parameter in expert_parameters:
        parameter.requires_grad_(True)

    torch.manual_seed(9609)
    x = torch.randn(2, 81, 256, device=device)
    address = torch.randn_like(x)
    cell_order = torch.randperm(81, device=device)
    optimizer = torch.optim.SGD(expert_parameters, lr=0.1)
    candidate.train()

    optimizer.zero_grad(set_to_none=True)
    output, _diagnostics, _main_states, _expert_states = capture_states(
        candidate, x, address, cell_order
    )
    target = torch.randn_like(output, dtype=torch.float32)
    (output.float() * target).mean().backward()
    stage1_up_grads = {}
    for layer, block in enumerate(candidate.blocks):
        assert block.state_expert is not None
        stage1_up_grads[str(layer)] = finite_nonzero_grad(
            block.state_expert.up_proj.weight,
            f"layer {layer} state_expert.up_proj.weight",
        )
    optimizer.step()

    optimizer.zero_grad(set_to_none=True)
    output, diagnostics, main_states, expert_states = capture_states(
        candidate, x, address, cell_order
    )
    (output.float() * target).mean().backward()
    stage2_grads: dict[str, dict[str, float]] = {}
    for layer, block in enumerate(candidate.blocks):
        expert = block.state_expert
        assert expert is not None
        core = expert.time_mix.core
        row = {
            "content_down": finite_nonzero_grad(
                expert.content_down.weight,
                f"layer {layer} state_expert.content_down.weight",
            ),
            "address_down": finite_nonzero_grad(
                expert.address_down.weight,
                f"layer {layer} state_expert.address_down.weight",
            ),
            "q_proj": finite_nonzero_grad(
                core.q_proj.weight,
                f"layer {layer} state_expert.core.q_proj.weight",
            ),
            "k_proj": finite_nonzero_grad(
                core.k_proj.weight,
                f"layer {layer} state_expert.core.k_proj.weight",
            ),
            "v_proj": finite_nonzero_grad(
                core.v_proj.weight,
                f"layer {layer} state_expert.core.v_proj.weight",
            ),
            "f_proj_in": finite_nonzero_grad(
                core.f_proj[0].weight,
                f"layer {layer} state_expert.core.f_proj.0.weight",
            ),
            "f_proj_out": finite_nonzero_grad(
                core.f_proj[1].weight,
                f"layer {layer} state_expert.core.f_proj.1.weight",
            ),
            "b_proj": finite_nonzero_grad(
                core.b_proj.weight,
                f"layer {layer} state_expert.core.b_proj.weight",
            ),
            "w_proj": finite_nonzero_grad(
                core.w_proj.weight,
                f"layer {layer} state_expert.core.w_proj.weight",
            ),
        }
        if layer > 0:
            row["future_seed_logit"] = finite_nonzero_grad(
                expert.future_seed_logit,
                f"layer {layer} state_expert.future_seed_logit",
            )
        stage2_grads[str(layer)] = row

    if len(main_states) != LAYERS:
        raise AssertionError(f"expected {LAYERS} main states, got {len(main_states)}")
    if len(expert_states) != LAYERS:
        raise AssertionError(f"expected {LAYERS} expert states, got {len(expert_states)}")
    main_chunk_backward = []
    for layer, state in enumerate(main_states):
        present = "ChunkGDN2FunctionBackward" in graph_names(state)
        if not present:
            raise AssertionError(
                f"main layer {layer} official GDN2 chunk backward is missing"
            )
        main_chunk_backward.append(present)
    expert_chunk_backward = []
    for layer, state in enumerate(expert_states):
        present = "ChunkGDN2FunctionBackward" in graph_names(state)
        if not present:
            raise AssertionError(
                f"expert layer {layer} official GDN2 chunk backward is missing"
            )
        expert_chunk_backward.append(present)
    diag_keys = (
        "gdn3_state_expert_enabled",
        "gdn3_state_expert_residual_relative_rms",
        "gdn3_state_expert_terminal_rms",
        "gdn3_state_expert_seed_rms",
        "gdn3_state_expert_gate_mean",
        "gdn3_state_expert_up_weight_rms",
        "gdn3_state_expert_address_contrast",
    )
    diag = {
        key: float(diagnostics[key].detach().float().item())
        for key in diag_keys
    }
    for key, value in diag.items():
        if not torch.isfinite(torch.tensor(value)):
            raise AssertionError(f"non-finite state-expert diagnostic {key}: {value}")
    for key in (
        "gdn3_state_expert_residual_relative_rms",
        "gdn3_state_expert_terminal_rms",
        "gdn3_state_expert_seed_rms",
        "gdn3_state_expert_gate_mean",
        "gdn3_state_expert_up_weight_rms",
    ):
        if diag[key] <= 0.0:
            raise AssertionError(f"inactive state-expert diagnostic {key}: {diag[key]}")

    return {
        "stage1_up_projection_gradient_max_abs_by_layer": stage1_up_grads,
        "stage2_inner_gradient_max_abs_by_layer": stage2_grads,
        "main_official_chunk_backward_by_layer": main_chunk_backward,
        "expert_official_chunk_backward_by_layer": expert_chunk_backward,
        "active_diagnostics": diag,
    }


def check_zero_identity_and_learning_path(device: torch.device) -> dict[str, Any]:
    torch.manual_seed(9605)
    control = build(device, state_expert_mode="none")
    torch.manual_seed(9605)
    candidate = build(device, state_expert_mode="dual_state")
    missing = copy_shared_state(control, candidate)
    control.eval()
    candidate.eval()

    x = torch.randn(2, 81, 256, device=device)
    address = torch.randn_like(x)
    cell_order = torch.randperm(81, device=device)
    with torch.no_grad():
        control_output, _control_diag, control_states, _ = capture_states(
            control, x, address, cell_order
        )
        candidate_output, candidate_diag, candidate_states, expert_states = (
            capture_states(candidate, x, address, cell_order)
        )
    if not torch.equal(control_output, candidate_output):
        raise AssertionError("zero-init state expert changed the model output")
    main_state_errors = [
        float((left.float() - right.float()).abs().max().item())
        for left, right in zip(control_states, candidate_states)
    ]
    if len(main_state_errors) != LAYERS or max(main_state_errors) != 0.0:
        raise AssertionError(
            f"zero-init main terminal-state mismatch: {main_state_errors}"
        )
    if len(expert_states) != LAYERS:
        raise AssertionError(f"expected {LAYERS} expert states, got {len(expert_states)}")
    expert_state_rms = [
        float(state.float().square().mean().sqrt().item())
        for state in expert_states
    ]
    if not all(value > 0.0 for value in expert_state_rms):
        raise AssertionError(f"inactive expert terminal states: {expert_state_rms}")

    expected_inserted_parameters = {
        name: parameter
        for name, parameter in candidate.named_parameters()
        if ".state_expert." in name
    }
    parameter_delta = sum(p.numel() for p in candidate.parameters()) - sum(
        p.numel() for p in control.parameters()
    )
    expected_parameter_delta = sum(
        parameter.numel()
        for parameter in expected_inserted_parameters.values()
    )
    if parameter_delta != expected_parameter_delta:
        raise AssertionError(
            f"parameter delta {parameter_delta} != {expected_parameter_delta}"
        )
    if parameter_delta != EXPECTED_PARAMETER_DELTA:
        raise AssertionError(
            f"parameter delta {parameter_delta} != declared {EXPECTED_PARAMETER_DELTA}"
        )

    official_main_layers = [
        type(block.time_mix.core).__name__ for block in candidate.blocks
    ]
    official_expert_layers = [
        type(block.state_expert.time_mix.core).__name__
        for block in candidate.blocks
        if block.state_expert is not None
    ]
    if official_main_layers != ["GatedDeltaNet2"] * LAYERS:
        raise AssertionError(f"unexpected main recurrent layers: {official_main_layers}")
    if official_expert_layers != ["GatedDeltaNet2"] * LAYERS:
        raise AssertionError(
            f"unexpected expert recurrent layers: {official_expert_layers}"
        )
    if float(candidate_diag["gdn3_state_expert_enabled"].item()) != 1.0:
        raise AssertionError("state-expert diagnostics did not mark the path active")
    if float(
        candidate_diag["gdn3_state_expert_residual_relative_rms"].item()
    ) != 0.0:
        raise AssertionError("zero-init state-expert residual is not zero")

    incoming_state_identity = check_nonzero_initial_state_identity(
        control, candidate, device
    )
    learning_path = check_two_stage_learning_path(candidate, device)
    return {
        "output_exact_identity": True,
        "main_terminal_state_max_abs_errors": main_state_errors,
        "expert_terminal_state_rms": expert_state_rms,
        "incoming_state_identity": incoming_state_identity,
        "missing_state_keys": missing,
        "inserted_parameter_count": len(expected_inserted_parameters),
        "parameter_delta": parameter_delta,
        "expert_state_values_per_layer": EXPECTED_STATE_VALUES_PER_LAYER,
        "expert_state_values_all_layers": (
            LAYERS * EXPECTED_STATE_VALUES_PER_LAYER
        ),
        "official_main_gdn2_layers": official_main_layers,
        "official_expert_gdn2_layers": official_expert_layers,
        "learning_path": learning_path,
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
