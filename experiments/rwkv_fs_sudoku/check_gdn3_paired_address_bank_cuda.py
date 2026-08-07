#!/usr/bin/env python3
from __future__ import annotations

import json
import math
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
PAIRED_HEADS = 16
HEAD_DIM = 32
VALUE_DIM = 32
MODEL_DIM = HEADS * HEAD_DIM
TOKENS = 81
EXPECTED_PARAMETER_DELTA = LAYERS * (2 * VALUE_DIM * HEAD_DIM + HEADS)
EXPECTED_STATE_VALUE_DELTA = LAYERS * HEADS * HEAD_DIM * VALUE_DIM


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
        gdn_progressive_base_expand_v=0.0,
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
    expected = set()
    for layer in range(LAYERS):
        prefix = f"blocks.{layer}.time_mix.paired_address_"
        expected.update(
            {
                f"{prefix}q_proj.weight",
                f"{prefix}k_proj.weight",
                f"{prefix}read_gate",
            }
        )
    if set(missing) != expected or unexpected:
        raise AssertionError(
            f"unexpected state migration: missing={missing}, unexpected={unexpected}"
        )
    return sorted(missing)


def capture_model(
    model: study.FutureSeedRWKV,
    x: torch.Tensor,
    address: torch.Tensor,
    cell_order: torch.Tensor,
) -> tuple[
    torch.Tensor,
    dict[str, torch.Tensor],
    list[torch.Tensor],
    list[torch.Tensor | None],
]:
    states: list[torch.Tensor] = []
    incoming: list[torch.Tensor | None] = []
    hooks = []

    def capture_input(
        _module: torch.nn.Module,
        _args: tuple[torch.Tensor, ...],
        kwargs: dict[str, Any],
    ) -> None:
        incoming.append(kwargs.get("initial_state"))

    def capture_output(
        _module: torch.nn.Module,
        _args: tuple[torch.Tensor, ...],
        _kwargs: dict[str, Any],
        output: tuple[torch.Tensor, torch.Tensor],
    ) -> None:
        states.append(output[1])

    for block in model.blocks:
        hooks.append(
            block.time_mix.register_forward_pre_hook(
                capture_input, with_kwargs=True
            )
        )
        hooks.append(
            block.time_mix.register_forward_hook(
                capture_output, with_kwargs=True
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
    return output, diagnostics, states, incoming


def capture_official_geometry(
    call: Callable[
        [], tuple[
            torch.Tensor,
            dict[str, torch.Tensor],
            list[torch.Tensor],
            list[torch.Tensor | None],
        ]
    ],
) -> tuple[
    tuple[
        torch.Tensor,
        dict[str, torch.Tensor],
        list[torch.Tensor],
        list[torch.Tensor | None],
    ],
    list[tuple[int, int]],
]:
    original = study.chunk_gdn2
    if original is None:
        raise RuntimeError("official chunk_gdn2 is unavailable")
    geometry: list[tuple[int, int]] = []

    def audited_chunk(*args: Any, **kwargs: Any) -> Any:
        query = kwargs.get("q")
        if not isinstance(query, torch.Tensor):
            raise AssertionError("official GDN2 call is missing tensor q")
        geometry.append((int(query.shape[1]), int(query.shape[2])))
        return original(*args, **kwargs)

    study.chunk_gdn2 = audited_chunk
    try:
        result = call()
    finally:
        study.chunk_gdn2 = original
    return result, geometry


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
    torch.manual_seed(13013)
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
        candidate_base, candidate_companion = candidate_state.split(
            HEADS, dim=1
        )
        errors = {
            "output": float(
                (control_output.float() - candidate_output.float())
                .abs()
                .max()
                .item()
            ),
            "base_state": float(
                (control_state.float() - candidate_base.float())
                .abs()
                .max()
                .item()
            ),
            "companion_state": float(
                (control_state.float() - candidate_companion.float())
                .abs()
                .max()
                .item()
            ),
        }
        if any(value != 0.0 for value in errors.values()):
            raise AssertionError(
                f"layer {layer} nonzero-state identity failed: {errors}"
            )
        rows.append({"layer": float(layer), **errors})
    return rows


def check_zero_identity_and_first_stage(device: torch.device) -> dict[str, Any]:
    torch.manual_seed(13010)
    control = build(device, update_mode="none")
    torch.manual_seed(13010)
    candidate = build(device, update_mode="paired_address_bank")
    missing = copy_shared_state(control, candidate)
    control.eval()
    candidate.eval()

    x = torch.randn(2, TOKENS, MODEL_DIM, device=device)
    address = torch.randn_like(x)
    cell_order = torch.randperm(TOKENS, device=device)
    with torch.no_grad():
        control_result, control_geometry = capture_official_geometry(
            lambda: capture_model(control, x, address, cell_order)
        )
        candidate_result, candidate_geometry = capture_official_geometry(
            lambda: capture_model(candidate, x, address, cell_order)
        )
    control_output, _control_diag, control_states, _control_incoming = control_result
    candidate_output, candidate_diag, candidate_states, candidate_incoming = (
        candidate_result
    )
    if control_geometry != [(TOKENS, HEADS)] * LAYERS:
        raise AssertionError(f"control official geometry changed: {control_geometry}")
    if candidate_geometry != [(TOKENS, PAIRED_HEADS)] * LAYERS:
        raise AssertionError(
            f"candidate official geometry changed: {candidate_geometry}"
        )
    if not torch.equal(control_output, candidate_output):
        raise AssertionError("zero-init paired bank changed full model output")

    state_rows = []
    for layer, (control_state, candidate_state) in enumerate(
        zip(control_states, candidate_states)
    ):
        if tuple(candidate_state.shape[1:]) != (
            PAIRED_HEADS,
            HEAD_DIM,
            VALUE_DIM,
        ):
            raise AssertionError(
                f"layer {layer} candidate state geometry {tuple(candidate_state.shape)}"
            )
        candidate_base, candidate_companion = candidate_state.split(
            HEADS, dim=1
        )
        base_error = float(
            (control_state.float() - candidate_base.float()).abs().max().item()
        )
        companion_error = float(
            (control_state.float() - candidate_companion.float())
            .abs()
            .max()
            .item()
        )
        if base_error != 0.0 or companion_error != 0.0:
            raise AssertionError(
                f"layer {layer} zero-init state identity failed: "
                f"base={base_error} companion={companion_error}"
            )
        state_rows.append(
            {
                "layer": layer,
                "base_max_abs_error": base_error,
                "companion_max_abs_error": companion_error,
            }
        )
    if candidate_incoming[0] is not None:
        raise AssertionError("layer zero unexpectedly received FutureSeed state")
    future_seed_pair_errors = []
    for layer, incoming in enumerate(candidate_incoming[1:], start=1):
        if incoming is None or incoming.shape[1] != PAIRED_HEADS:
            raise AssertionError(f"layer {layer} paired FutureSeed state is missing")
        base, companion = incoming.split(HEADS, dim=1)
        error = float((base.float() - companion.float()).abs().max().item())
        if error != 0.0:
            raise AssertionError(
                f"layer {layer} duplicated FutureSeed gate/state mismatch: {error}"
            )
        future_seed_pair_errors.append(error)

    parameter_delta = sum(p.numel() for p in candidate.parameters()) - sum(
        p.numel() for p in control.parameters()
    )
    if parameter_delta != EXPECTED_PARAMETER_DELTA:
        raise AssertionError(
            f"parameter delta {parameter_delta} != {EXPECTED_PARAMETER_DELTA}"
        )
    state_value_delta = sum(state.numel() for state in candidate_states) - sum(
        state.numel() for state in control_states
    )
    if state_value_delta != EXPECTED_STATE_VALUE_DELTA:
        raise AssertionError(
            "state value delta "
            f"{state_value_delta} != {EXPECTED_STATE_VALUE_DELTA}"
        )

    incoming_identity = check_nonzero_initial_state_identity(
        control, candidate, device
    )
    candidate.train()
    candidate.zero_grad(set_to_none=True)
    output, diagnostics, states, _incoming = capture_model(
        candidate,
        x.detach().clone().requires_grad_(True),
        address,
        cell_order,
    )
    target = torch.randn_like(output, dtype=torch.float32)
    (output.float() * target).mean().backward()
    gate_gradients: dict[str, float] = {}
    chunk_counts = []
    for layer, (block, state) in enumerate(zip(candidate.blocks, states)):
        gate = block.time_mix.paired_address_read_gate
        if gate is None or gate.grad is None:
            raise AssertionError(f"layer {layer} read-gate gradient is missing")
        if not bool(torch.isfinite(gate.grad).all()):
            raise AssertionError(f"layer {layer} read-gate gradient is non-finite")
        grad_min = float(gate.grad.float().abs().min().item())
        if grad_min <= 0.0:
            raise AssertionError(f"layer {layer} read-gate gradient has a dead head")
        gate_gradients[str(layer)] = grad_min
        chunk_counts.append(graph_names(state).count("ChunkGDN2FunctionBackward"))
    if chunk_counts != [1] * LAYERS:
        raise AssertionError(
            f"single-call official backward graph mismatch: {chunk_counts}"
        )

    official_layers = [type(block.time_mix.core).__name__ for block in candidate.blocks]
    if official_layers != ["GatedDeltaNet2"] * LAYERS:
        raise AssertionError(f"unexpected recurrent layers: {official_layers}")
    if float(diagnostics["gdn3_paired_address_bank_enabled"].item()) != 1.0:
        raise AssertionError("paired-bank diagnostics did not mark the path active")
    for key in (
        "gdn3_paired_address_bank_read_gate_abs",
        "gdn3_paired_address_bank_q_residual_relative_rms",
        "gdn3_paired_address_bank_k_residual_relative_rms",
        "gdn3_paired_address_bank_state_residual_relative_rms",
    ):
        if float(diagnostics[key].item()) != 0.0:
            raise AssertionError(f"zero-init diagnostic is not zero: {key}")
    return {
        "output_exact_identity": True,
        "state_identity": state_rows,
        "future_seed_pair_max_abs_errors": future_seed_pair_errors,
        "incoming_state_identity": incoming_identity,
        "missing_parameters": missing,
        "parameter_delta": parameter_delta,
        "state_value_delta": state_value_delta,
        "read_gate_gradient_min_abs_by_layer": gate_gradients,
        "official_geometry": candidate_geometry,
        "official_chunk_backward_counts": chunk_counts,
        "official_gdn2_layers": official_layers,
        "zero_init_diag": {
            key: float(value.detach().float().item())
            for key, value in candidate_diag.items()
            if key.startswith("gdn3_paired_address_bank_")
        },
    }


def check_second_stage_and_equivariance(device: torch.device) -> dict[str, Any]:
    torch.manual_seed(13011)
    model = build(device, update_mode="paired_address_bank")
    model.train()
    with torch.no_grad():
        for block in model.blocks:
            time_mix = block.time_mix
            if time_mix.paired_address_read_gate is None:
                raise AssertionError("paired read gate is missing")
            time_mix.paired_address_read_gate.fill_(math.atanh(0.25))
            if time_mix.paired_address_q_proj is None or time_mix.paired_address_k_proj is None:
                raise AssertionError("paired address projection is missing")
            time_mix.paired_address_q_proj.weight.zero_()
            time_mix.paired_address_k_proj.weight.zero_()

    x = torch.randn(2, TOKENS, MODEL_DIM, device=device)
    address = torch.randn_like(x)
    cell_order = torch.randperm(TOKENS, device=device)
    model.zero_grad(set_to_none=True)
    output, _diagnostics, _states, _incoming = capture_model(
        model,
        x.detach().clone().requires_grad_(True),
        address,
        cell_order,
    )
    output.float().square().mean().backward()
    projection_gradients: dict[str, dict[str, float]] = {}
    for layer, block in enumerate(model.blocks):
        q_projection = block.time_mix.paired_address_q_proj
        k_projection = block.time_mix.paired_address_k_proj
        if q_projection is None or k_projection is None:
            raise AssertionError(f"layer {layer} paired projection is missing")
        row: dict[str, float] = {}
        for name, projection in (("q", q_projection), ("k", k_projection)):
            gradient = projection.weight.grad
            if gradient is None or not bool(torch.isfinite(gradient).all()):
                raise AssertionError(
                    f"layer {layer} {name} projection gradient is missing/non-finite"
                )
            grad_max = float(gradient.float().abs().max().item())
            if grad_max <= 0.0:
                raise AssertionError(
                    f"layer {layer} {name} projection gradient is zero"
                )
            row[name] = grad_max
        projection_gradients[str(layer)] = row

    model.eval()
    with torch.no_grad():
        for block in model.blocks:
            read_gate = block.time_mix.paired_address_read_gate
            assert read_gate is not None
            read_gate.zero_()
        base_reference_output, _base_reference_diag, base_reference_states, _ = (
            capture_model(model, x, address, cell_order)
        )
        for block in model.blocks:
            read_gate = block.time_mix.paired_address_read_gate
            assert read_gate is not None
            read_gate.fill_(math.atanh(0.25))
        zero_output, _zero_diag, zero_states, _zero_incoming = capture_model(
            model, x, address, cell_order
        )
        identity = torch.eye(HEAD_DIM, device=device)
        for layer, block in enumerate(model.blocks):
            q_projection = block.time_mix.paired_address_q_proj
            k_projection = block.time_mix.paired_address_k_proj
            assert q_projection is not None and k_projection is not None
            q_projection.weight.copy_((0.02 + layer * 0.0002) * identity)
            k_projection.weight.copy_((-0.015 - layer * 0.0001) * identity)
            read_gate = block.time_mix.paired_address_read_gate
            assert read_gate is not None
            read_gate.zero_()
        base_open_output, _base_open_diag, base_open_states, _ = capture_model(
            model, x, address, cell_order
        )
        for block in model.blocks:
            read_gate = block.time_mix.paired_address_read_gate
            assert read_gate is not None
            read_gate.fill_(math.atanh(0.25))
        open_output, open_diag, open_states, _open_incoming = capture_model(
            model, x, address, cell_order
        )
    if not torch.equal(base_reference_output, base_open_output):
        raise AssertionError(
            "companion address projections changed the isolated base output"
        )
    base_state_errors = []
    for reference_state, open_state in zip(
        base_reference_states, base_open_states
    ):
        reference_base = reference_state[:, :HEADS]
        open_base = open_state[:, :HEADS]
        error = float(
            (reference_base.float() - open_base.float()).abs().max().item()
        )
        base_state_errors.append(error)
    if any(error != 0.0 for error in base_state_errors):
        raise AssertionError(
            "companion address projections changed isolated base states: "
            f"{base_state_errors}"
        )
    output_change = float(
        (open_output.float() - zero_output.float()).abs().max().item()
    )
    state_pair_residuals = []
    for state in open_states:
        base, companion = state.float().split(HEADS, dim=1)
        relative = float(
            (
                (companion - base).square().mean().sqrt()
                / base.square().mean().sqrt().clamp_min(1e-8)
            ).item()
        )
        state_pair_residuals.append(relative)
    if output_change <= 0.0 or min(state_pair_residuals) <= 0.0:
        raise AssertionError(
            "opened paired bank did not change output/state: "
            f"output={output_change} state={state_pair_residuals}"
        )

    first = model.blocks[0].time_mix
    q_projection = first.paired_address_q_proj
    k_projection = first.paired_address_k_proj
    assert q_projection is not None and k_projection is not None
    value = torch.randn(3, 17, HEADS, VALUE_DIM, device=device)
    permutation = torch.randperm(HEADS, device=device)
    inverse = torch.argsort(permutation)
    equivariance: dict[str, float] = {}
    with torch.no_grad():
        for name, projection in (("q", q_projection), ("k", k_projection)):
            direct = projection(value)
            permuted = projection(value.index_select(2, permutation))
            error = float(
                (direct - permuted.index_select(2, inverse))
                .float()
                .abs()
                .max()
                .item()
            )
            if error > 3e-6:
                raise AssertionError(
                    f"{name} head permutation equivariance error {error}"
                )
            equivariance[name] = error

    required_activation = (
        "gdn3_paired_address_bank_read_gate_abs",
        "gdn3_paired_address_bank_q_residual_relative_rms",
        "gdn3_paired_address_bank_k_residual_relative_rms",
        "gdn3_paired_address_bank_state_residual_relative_rms",
        "gdn3_paired_address_bank_terminal_rms",
    )
    for key in required_activation:
        value_float = float(open_diag[key].item())
        if not math.isfinite(value_float) or value_float <= 0.0:
            raise AssertionError(f"opened activation is invalid: {key}={value_float}")
    return {
        "projection_gradient_max_abs_by_layer": projection_gradients,
        "base_output_exact_isolation": True,
        "base_state_max_abs_errors_by_layer": base_state_errors,
        "output_max_abs_change": output_change,
        "state_pair_relative_rms_by_layer": state_pair_residuals,
        "head_permutation_max_abs_error": equivariance,
        "opened_diag": {
            key: float(value.detach().float().item())
            for key, value in open_diag.items()
            if key.startswith("gdn3_paired_address_bank_")
        },
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
    properties = torch.cuda.get_device_properties(device)
    output = {
        "device": str(device),
        "gpu_name": properties.name,
        "gpu_uuid": EXPECTED_GPU_UUID,
        "fla_source_sha": study.GAIN_BUDGET_FLA_SHA,
        "zero_identity_and_first_stage": check_zero_identity_and_first_stage(
            device
        ),
        "second_stage_and_equivariance": check_second_stage_and_equivariance(
            device
        ),
    }
    print(json.dumps(output, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
