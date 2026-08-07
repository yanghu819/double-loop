#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import subprocess
from typing import Any, Callable

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

import study_rwkv_futureseed_loop as study


EXPECTED_GPU_UUID = "GPU-53e9f3b4-2966-65d3-6614-09c540921519"
EXPECTED_FLA_SOURCE_SHA = "9c8e42e762fce087c27b673af4922795d9edb85e"
LAYERS = 12
HEADS = 8
HEAD_DIM = 32
VALUE_DIM = 32
MODEL_DIM = HEADS * HEAD_DIM
TOKENS = 81
EXPECTED_PARAMETER_DELTA = LAYERS * VALUE_DIM * 3


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


def build(device: torch.device, *, update_mode: str) -> study.FutureSeedRWKV:
    return study.FutureSeedRWKV(
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
        gdn2_address_mode="position_qk",
        gdn2_update_mode=update_mode,
        gdn2_cross_layer_init="independent",
    ).to(device)


def copy_shared_state(
    control: study.FutureSeedRWKV,
    candidate: study.FutureSeedRWKV,
) -> list[str]:
    missing, unexpected = candidate.load_state_dict(
        control.state_dict(), strict=False
    )
    expected = {
        f"blocks.{layer}.time_mix.orthogonal_head_write_proj.weight"
        for layer in range(LAYERS)
    }
    if set(missing) != expected or unexpected:
        raise AssertionError(
            f"unexpected state migration: missing={missing}, unexpected={unexpected}"
        )
    return sorted(missing)


def capture_terminal_states(
    model: study.FutureSeedRWKV,
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


def capture_official_call_lengths(
    call: Callable[
        [], tuple[torch.Tensor, dict[str, torch.Tensor], list[torch.Tensor]]
    ],
) -> tuple[
    tuple[torch.Tensor, dict[str, torch.Tensor], list[torch.Tensor]],
    list[int],
]:
    original = study.chunk_gdn2
    if original is None:
        raise RuntimeError("official chunk_gdn2 is unavailable")
    lengths: list[int] = []

    def audited_chunk(*args: Any, **kwargs: Any) -> Any:
        q = kwargs.get("q")
        if not isinstance(q, torch.Tensor):
            raise AssertionError("official GDN2 call is missing tensor q")
        lengths.append(int(q.shape[1]))
        return original(*args, **kwargs)

    study.chunk_gdn2 = audited_chunk
    try:
        result = call()
    finally:
        study.chunk_gdn2 = original
    return result, lengths


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


def check_nonzero_initial_state_identity(
    control: study.FutureSeedRWKV,
    candidate: study.FutureSeedRWKV,
    device: torch.device,
) -> list[dict[str, float]]:
    rows = []
    torch.manual_seed(10112)
    for layer, (control_block, candidate_block) in enumerate(
        zip(control.blocks, candidate.blocks)
    ):
        x = torch.randn(1, TOKENS, MODEL_DIM, device=device)
        address = torch.randn_like(x)
        cell_order = torch.randperm(TOKENS, device=device)
        initial_state = torch.randn(
            1,
            HEADS,
            HEAD_DIM,
            VALUE_DIM,
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
                f"layer {layer} zero-angle initial-state identity failed: "
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


def open_learning_path(
    candidate: study.FutureSeedRWKV,
    x: torch.Tensor,
    address: torch.Tensor,
    cell_order: torch.Tensor,
) -> dict[str, dict[str, float]]:
    candidate.train()
    candidate.zero_grad(set_to_none=True)
    output, _diagnostics, states = capture_terminal_states(
        candidate, x, address, cell_order
    )
    chunk_counts = [
        graph_names(state).count("ChunkGDN2FunctionBackward")
        for state in states
    ]
    if chunk_counts != [1] * LAYERS:
        raise AssertionError(
            f"single-call official backward graph mismatch: {chunk_counts}"
        )
    output.float().square().mean().backward()

    angle_gradient_max: dict[str, float] = {}
    with torch.no_grad():
        for layer, block in enumerate(candidate.blocks):
            projection = block.time_mix.orthogonal_head_write_proj
            if projection is None:
                raise AssertionError(f"layer {layer} projection is missing")
            gradient = projection.weight.grad
            if gradient is None or not bool(torch.isfinite(gradient).all()):
                raise AssertionError(
                    f"layer {layer} first-stage gradient is missing/non-finite"
                )
            angle_gradient = gradient[-1:]
            value = float(angle_gradient.float().abs().max().item())
            if value <= 0.0:
                raise AssertionError(f"layer {layer} angle gradient is zero")
            angle_gradient_max[str(layer)] = value
            projection.weight[-1:].add_(
                -0.01
                * angle_gradient
                / angle_gradient.float().abs().max().clamp_min(1e-8)
            )

    candidate.zero_grad(set_to_none=True)
    opened_output, _opened_diag, opened_states = capture_terminal_states(
        candidate, x, address, cell_order
    )
    opened_counts = [
        graph_names(state).count("ChunkGDN2FunctionBackward")
        for state in opened_states
    ]
    if opened_counts != [1] * LAYERS:
        raise AssertionError(
            f"opened official backward graph mismatch: {opened_counts}"
        )
    opened_output.float().square().mean().backward()

    plane_gradient_max: dict[str, float] = {}
    for layer, block in enumerate(candidate.blocks):
        projection = block.time_mix.orthogonal_head_write_proj
        assert projection is not None
        gradient = projection.weight.grad
        if gradient is None or not bool(torch.isfinite(gradient).all()):
            raise AssertionError(
                f"layer {layer} opened gradient is missing/non-finite"
            )
        value = float(gradient[:-1].float().abs().max().item())
        if value <= 0.0:
            raise AssertionError(f"layer {layer} plane gradient is zero")
        plane_gradient_max[str(layer)] = value
    return {
        "angle_gradient_max_by_layer": angle_gradient_max,
        "plane_gradient_max_by_layer": plane_gradient_max,
    }


def check_route_geometry(
    candidate: study.FutureSeedRWKV,
    device: torch.device,
) -> dict[str, float]:
    torch.manual_seed(10113)
    time_mix = candidate.blocks[0].time_mix
    value = torch.randn(3, TOKENS, HEADS, VALUE_DIM, device=device)
    permutation = torch.randperm(HEADS, device=device)
    inverse = torch.argsort(permutation)

    routed_fp32, fp32_diag = time_mix._orthogonal_head_write_route(value)
    permuted_fp32, _ = time_mix._orthogonal_head_write_route(
        value[:, :, permutation]
    )
    equivariance_error = float(
        (routed_fp32 - permuted_fp32[:, :, inverse]).abs().max().item()
    )
    if equivariance_error > 3e-6:
        raise AssertionError(f"head permutation error: {equivariance_error}")

    routed_bf16, bf16_diag = time_mix._orthogonal_head_write_route(
        value.to(dtype=torch.bfloat16)
    )
    fp32_norm_error = float(
        fp32_diag[
            "gdn3_orthogonal_head_write_fp32_norm_ratio_max_error"
        ].item()
    )
    storage_norm_error = float(
        bf16_diag[
            "gdn3_orthogonal_head_write_storage_norm_ratio_max_error"
        ].item()
    )
    plane_error = max(
        float(
            fp32_diag[
                "gdn3_orthogonal_head_write_plane_dot_abs_max"
            ].item()
        ),
        float(
            fp32_diag[
                "gdn3_orthogonal_head_write_plane_norm_error_max"
            ].item()
        ),
    )
    if fp32_norm_error > 1e-4:
        raise AssertionError(f"FP32 norm error is too large: {fp32_norm_error}")
    if storage_norm_error > 5e-3:
        raise AssertionError(
            f"BF16 storage norm error is too large: {storage_norm_error}"
        )
    if plane_error > 1e-5:
        raise AssertionError(f"orthogonal plane error is too large: {plane_error}")

    fp32_change = float(
        (routed_fp32 - value).square().mean().sqrt().item()
    )
    bf16_change = float(
        (routed_bf16.float() - value.to(torch.bfloat16).float())
        .square()
        .mean()
        .sqrt()
        .item()
    )
    if fp32_change <= 0.0 or bf16_change <= 0.0:
        raise AssertionError(
            f"opened route did not change V: {fp32_change}, {bf16_change}"
        )
    return {
        "head_permutation_max_abs_error": equivariance_error,
        "fp32_norm_ratio_max_error": fp32_norm_error,
        "bf16_storage_norm_ratio_max_error": storage_norm_error,
        "plane_error_max": plane_error,
        "fp32_route_change_rms": fp32_change,
        "bf16_route_change_rms": bf16_change,
    }


OPEN_METRICS = (
    "gdn3_orthogonal_head_write_angle_abs",
    "gdn3_orthogonal_head_write_angle_batch_std",
    "gdn3_orthogonal_head_write_angle_token_std",
    "gdn3_orthogonal_head_write_v_residual_relative_rms",
    "gdn3_orthogonal_head_write_v_residual_batch_std",
    "gdn3_orthogonal_head_write_v_residual_token_std",
    "gdn3_orthogonal_head_write_terminal_rms",
    "gdn3_orthogonal_head_write_terminal_batch_std",
    "gdn3_orthogonal_head_write_angle_weight_rms",
    "gdn3_orthogonal_head_write_plane_weight_rms",
)


def check_zero_identity_and_learning_path(device: torch.device) -> dict[str, Any]:
    torch.manual_seed(10111)
    control = build(device, update_mode="none")
    torch.manual_seed(10111)
    candidate = build(device, update_mode="orthogonal_head_write")
    missing = copy_shared_state(control, candidate)
    control.eval()
    candidate.eval()

    x = torch.randn(2, TOKENS, MODEL_DIM, device=device)
    address = torch.randn_like(x)
    cell_order = torch.randperm(TOKENS, device=device)
    with torch.no_grad():
        control_output, _control_diag, control_states = capture_terminal_states(
            control, x, address, cell_order
        )
        (
            (candidate_output, candidate_diag, candidate_states),
            official_call_lengths,
        ) = capture_official_call_lengths(
            lambda: capture_terminal_states(candidate, x, address, cell_order)
        )
    if official_call_lengths != [TOKENS] * LAYERS:
        raise AssertionError(
            f"official call lengths are not 12x{TOKENS}: "
            f"{official_call_lengths}"
        )
    if not torch.equal(control_output, candidate_output):
        error = float(
            (control_output.float() - candidate_output.float()).abs().max().item()
        )
        raise AssertionError(f"zero-angle route changed model output: {error}")
    state_errors = [
        float((left.float() - right.float()).abs().max().item())
        for left, right in zip(control_states, candidate_states)
    ]
    if len(state_errors) != LAYERS or max(state_errors) != 0.0:
        raise AssertionError(
            f"zero-angle terminal-state mismatch: {state_errors}"
        )
    if float(
        candidate_diag[
            "gdn3_orthogonal_head_write_v_residual_relative_rms"
        ].item()
    ) != 0.0:
        raise AssertionError("zero-angle V residual is not zero")

    parameter_delta = sum(p.numel() for p in candidate.parameters()) - sum(
        p.numel() for p in control.parameters()
    )
    if parameter_delta != EXPECTED_PARAMETER_DELTA:
        raise AssertionError(
            f"parameter delta {parameter_delta} != {EXPECTED_PARAMETER_DELTA}"
        )
    initial_state_rows = check_nonzero_initial_state_identity(
        control, candidate, device
    )
    gradients = open_learning_path(candidate, x, address, cell_order)

    candidate.eval()
    with torch.no_grad():
        opened_output, opened_diag, opened_states = capture_terminal_states(
            candidate, x, address, cell_order
        )
    if torch.equal(candidate_output, opened_output) or all(
        torch.equal(left, right)
        for left, right in zip(candidate_states, opened_states)
    ):
        raise AssertionError("opened head-write route did not alter output and state")
    active_layers = sum(
        int(
            float(
                block.time_mix.last_gain_budget_diag.get(
                    "gdn3_orthogonal_head_write_enabled", x.new_zeros(())
                ).item()
            )
            > 0.0
        )
        for block in candidate.blocks
    )
    if active_layers != LAYERS:
        raise AssertionError(
            f"expected {LAYERS} active head-write routes, got {active_layers}"
        )
    for key in OPEN_METRICS:
        value = float(opened_diag[key].item())
        if not torch.isfinite(torch.tensor(value)) or value <= 0.0:
            raise AssertionError(f"opened metric {key} is invalid: {value}")

    official_layers = [type(block.time_mix.core).__name__ for block in candidate.blocks]
    if official_layers != ["GatedDeltaNet2"] * LAYERS:
        raise AssertionError(f"unexpected recurrent layers: {official_layers}")
    return {
        "output_exact_identity": True,
        "terminal_state_max_abs_errors": state_errors,
        "nonzero_initial_state_identity": initial_state_rows,
        "missing_parameters": missing,
        "parameter_delta": parameter_delta,
        "official_call_lengths": official_call_lengths,
        "controller_gradients": gradients,
        "active_layers": active_layers,
        "official_layers": official_layers,
        "official_chunk_backward_paths": LAYERS,
        "position_qk_active": float(candidate_diag["gdn2_address_enabled"].item()),
        "opened_diagnostics": {
            key: float(value.detach().float().item())
            for key, value in opened_diag.items()
            if key.startswith("gdn3_orthogonal_head_write_")
        },
        "route_geometry": check_route_geometry(candidate, device),
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
    if study.GAIN_BUDGET_FLA_SHA != EXPECTED_FLA_SOURCE_SHA:
        raise RuntimeError(
            "pinned FLA SHA drift: "
            f"{study.GAIN_BUDGET_FLA_SHA} != {EXPECTED_FLA_SOURCE_SHA}"
        )
    device = torch.device("cuda:0")
    props = torch.cuda.get_device_properties(device)
    output = {
        "device": str(device),
        "gpu_name": props.name,
        "gpu_uuid": EXPECTED_GPU_UUID,
        "fla_source_sha": study.GAIN_BUDGET_FLA_SHA,
        "zero_identity_and_learning_path": check_zero_identity_and_learning_path(
            device
        ),
    }
    print(json.dumps(output, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
