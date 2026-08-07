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
HEAD_DIM = 32
FEEDBACK_HIDDEN = 16


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
        f"blocks.{layer}.time_mix.state_feedback_{side}.weight"
        for layer in range(LAYERS)
        for side in ("in", "out")
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
    torch.manual_seed(9706)
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


def check_two_stage_gradients(
    candidate: FutureSeedRWKV,
    device: torch.device,
) -> dict[str, Any]:
    torch.manual_seed(9707)
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

    candidate.train()
    candidate.zero_grad(set_to_none=True)
    first_losses = []
    first_states = []
    for block in candidate.blocks:
        output, terminal = direct_block_forward(
            block, x, address, cell_order, initial_state
        )
        first_losses.append(output.float().square().mean())
        first_states.append(terminal)
    torch.stack(first_losses).mean().backward()

    first_stage: dict[str, float] = {}
    with torch.no_grad():
        for layer, block in enumerate(candidate.blocks):
            controller_in = block.time_mix.state_feedback_in
            controller_out = block.time_mix.state_feedback_out
            if controller_in is None or controller_out is None:
                raise AssertionError(f"layer {layer} state feedback is missing")
            grad = controller_out.weight.grad
            if grad is None or not bool(torch.isfinite(grad).all()):
                raise AssertionError(f"layer {layer} output gradient is missing/non-finite")
            grad_max = float(grad.float().abs().max().item())
            if grad_max <= 0.0:
                raise AssertionError(f"layer {layer} output gradient is zero")
            first_stage[str(layer)] = grad_max
            controller_out.weight.add_(
                -0.01 * grad / grad.float().abs().max().clamp_min(1e-8)
            )

    candidate.zero_grad(set_to_none=True)
    second_losses = []
    second_states = []
    for block in candidate.blocks:
        output, terminal = direct_block_forward(
            block, x, address, cell_order, initial_state
        )
        second_losses.append(output.float().square().mean())
        second_states.append(terminal)
    torch.stack(second_losses).mean().backward()

    second_stage: dict[str, dict[str, float]] = {}
    for layer, block in enumerate(candidate.blocks):
        controller_in = block.time_mix.state_feedback_in
        controller_out = block.time_mix.state_feedback_out
        assert controller_in is not None and controller_out is not None
        in_grad = controller_in.weight.grad
        out_grad = controller_out.weight.grad
        if in_grad is None or out_grad is None:
            raise AssertionError(f"layer {layer} second-stage gradient is missing")
        if not bool(torch.isfinite(in_grad).all() and torch.isfinite(out_grad).all()):
            raise AssertionError(f"layer {layer} second-stage gradient is non-finite")
        in_max = float(in_grad.float().abs().max().item())
        out_max = float(out_grad.float().abs().max().item())
        if in_max <= 0.0 or out_max <= 0.0:
            raise AssertionError(
                f"layer {layer} second-stage gradient is zero: {in_max}, {out_max}"
            )
        second_stage[str(layer)] = {"in": in_max, "out": out_max}
    for layer, terminal in enumerate(second_states):
        if "ChunkGDN2FunctionBackward" not in graph_names(terminal):
            raise AssertionError(f"layer {layer} official chunk backward is missing")
    return {
        "zero_output_projection_grad_max_by_layer": first_stage,
        "opened_controller_grad_max_by_layer": second_stage,
        "official_chunk_backward_layers": LAYERS,
    }


def check_equivariance_and_state_dependence(
    candidate: FutureSeedRWKV,
    device: torch.device,
) -> dict[str, float]:
    torch.manual_seed(9708)
    time_mix = candidate.blocks[0].time_mix
    batch, tokens = 3, 17
    q = torch.randn(batch, tokens, HEADS, HEAD_DIM, device=device, dtype=torch.bfloat16)
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    b_raw = torch.randn_like(q)
    w_raw = torch.randn_like(q)
    state = torch.randn(
        batch,
        HEADS,
        HEAD_DIM,
        HEAD_DIM,
        device=device,
        dtype=torch.float32,
    )
    base = time_mix._state_feedback_update(q, k, v, b_raw, w_raw, state)
    permutation = torch.randperm(HEADS, device=device)
    permuted = time_mix._state_feedback_update(
        q[:, :, permutation],
        k[:, :, permutation],
        v[:, :, permutation],
        b_raw[:, :, permutation],
        w_raw[:, :, permutation],
        state[:, permutation],
    )
    inverse = torch.argsort(permutation)
    errors = []
    for original, transformed in zip(base[:4], permuted[:4]):
        errors.append(
            float(
                (original.float() - transformed[:, :, inverse].float())
                .abs()
                .max()
                .item()
            )
        )
    max_equivariance_error = max(errors)
    if max_equivariance_error > 2e-6:
        raise AssertionError(f"head permutation error: {max_equivariance_error}")

    shuffled = time_mix._state_feedback_update(
        q,
        k,
        v,
        b_raw,
        w_raw,
        state.roll(1, dims=0),
    )
    state_dependency = float(
        sum((left.float() - right.float()).square().mean() for left, right in zip(base[:4], shuffled[:4]))
        .sqrt()
        .item()
    )
    if state_dependency <= 0.0:
        raise AssertionError("opened controller is not state dependent")
    return {
        "head_permutation_max_abs_error": max_equivariance_error,
        "shuffled_state_output_rms": state_dependency,
    }


def check_zero_identity_and_learning_path(device: torch.device) -> dict[str, Any]:
    torch.manual_seed(9705)
    control = build(device, update_mode="none")
    torch.manual_seed(9705)
    candidate = build(device, update_mode="state_feedback")
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
        raise AssertionError("zero-init state feedback changed the model output")
    state_errors = [
        float((left.float() - right.float()).abs().max().item())
        for left, right in zip(control_states, candidate_states)
    ]
    if len(state_errors) != LAYERS or max(state_errors) != 0.0:
        raise AssertionError(f"zero-init terminal-state mismatch: {state_errors}")

    per_layer_delta = HEAD_DIM * FEEDBACK_HIDDEN + FEEDBACK_HIDDEN * (4 * HEAD_DIM)
    expected_delta = LAYERS * per_layer_delta
    parameter_delta = sum(p.numel() for p in candidate.parameters()) - sum(
        p.numel() for p in control.parameters()
    )
    if parameter_delta != expected_delta:
        raise AssertionError(f"parameter delta {parameter_delta} != {expected_delta}")

    initial_state_rows = check_initial_state_identity(control, candidate, device)
    learning_path = check_two_stage_gradients(candidate, device)
    candidate.eval()
    with torch.no_grad():
        _output, opened_diag, _states = capture_terminal_states(
            candidate, x, address, cell_order
        )
    active_layers = sum(
        int(float(block.time_mix.last_gain_budget_diag.get(
            "gdn3_state_feedback_enabled", x.new_zeros(())
        ).item()) > 0.0)
        for block in candidate.blocks
    )
    if active_layers != LAYERS - 1:
        raise AssertionError(f"expected 11 receiving layers, got {active_layers}")
    for key in (
        "gdn3_state_feedback_read_rms",
        "gdn3_state_feedback_read_batch_std",
        "gdn3_state_feedback_residual_relative_rms",
        "gdn3_state_feedback_residual_token_std",
        "gdn3_state_feedback_k_relative_change",
        "gdn3_state_feedback_v_relative_change",
        "gdn3_state_feedback_b_relative_change",
        "gdn3_state_feedback_w_relative_change",
    ):
        value = float(opened_diag[key].item())
        if not torch.isfinite(torch.tensor(value)) or value <= 0.0:
            raise AssertionError(f"opened state-feedback metric {key} is invalid: {value}")

    official_layers = [type(block.time_mix.core).__name__ for block in candidate.blocks]
    if official_layers != ["GatedDeltaNet2"] * LAYERS:
        raise AssertionError(f"unexpected recurrent layers: {official_layers}")
    if float(candidate_diag["gdn2_address_enabled"].item()) != 1.0:
        raise AssertionError("position-QK address path is not active")
    if float(candidate_diag["gdn3_state_feedback_residual_relative_rms"].item()) != 0.0:
        raise AssertionError("zero-init state-feedback residual is not zero")
    return {
        "output_exact_identity": True,
        "terminal_state_max_abs_errors": state_errors,
        "initial_state_identity": initial_state_rows,
        "missing_parameters": missing,
        "parameter_delta": parameter_delta,
        "learning_path": learning_path,
        "active_receiving_layers": active_layers,
        "official_gdn2_layers": official_layers,
        "position_qk_active": True,
        "opened_diagnostics": {
            key: float(value.detach().float().item())
            for key, value in opened_diag.items()
            if key.startswith("gdn3_state_feedback_")
        },
        "equivariance_and_state_dependence": check_equivariance_and_state_dependence(
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
