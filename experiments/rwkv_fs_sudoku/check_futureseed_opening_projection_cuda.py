#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import pathlib
import subprocess
from typing import Any


EXPECTED_GPU_UUID = os.environ.get("EXPECTED_GPU_UUID", "").strip()
if not EXPECTED_GPU_UUID:
    raise RuntimeError(
        "EXPECTED_GPU_UUID must be bound to the admitted task GPU before CUDA imports"
    )

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
if os.environ.get("CUDA_VISIBLE_DEVICES") != "0":
    raise RuntimeError("CUDA_VISIBLE_DEVICES=0 is required")
if os.environ.get("FLA_DISABLE_BACKEND_DISPATCH") != "1":
    raise RuntimeError("FLA_DISABLE_BACKEND_DISPATCH=1 is required")
if os.environ.get("FLA_CONV_BACKEND") != "triton":
    raise RuntimeError("FLA_CONV_BACKEND=triton is required")

import torch

import study_rwkv_futureseed_loop as study


EXPECTED_FLA_SOURCE_SHA = "9c8e42e762fce087c27b673af4922795d9edb85e"
LAYERS = 12
HEADS = 8
HEAD_DIM = 32
MODEL_DIM = 256
TOKENS = 81
GATE_TENSORS = 12
GATE_PARAMETERS = 96
ACTIVE_GATE_TENSORS = 11
ACTIVE_GATE_PARAMETERS = 88


def graph_names(tensor: torch.Tensor) -> list[str]:
    names: list[str] = []
    queue = [tensor.grad_fn]
    seen: set[Any] = set()
    while queue:
        fn = queue.pop()
        if fn is None or fn in seen:
            continue
        # Retain node proxies so Python cannot recycle ids during graph traversal.
        seen.add(fn)
        names.append(type(fn).__name__)
        queue.extend(
            next_fn for next_fn, _ in fn.next_functions if next_fn is not None
        )
    return names


def build_model(device: torch.device) -> study.FutureSeedRWKV:
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
        activation_checkpoint=False,
        backbone="gdn2",
        gdn_mode="chunk",
        gdn_expand_v=1.0,
        gdn_progressive_base_expand_v=0.0,
        gdn_use_short_conv=True,
        gdn_conv_size=4,
        gdn_allow_neg_eigval=False,
        gdn2_gain_budget_mode="none",
        gdn2_fast_slow_decay_mode="none",
        gdn2_precondition_mode="none",
        gdn2_address_mode="position_qk",
        gdn2_update_mode="none",
        gdn2_state_expert_mode="none",
        gdn2_cross_layer_init="independent",
    ).to(device)


def capture_forward(
    model: study.FutureSeedRWKV,
    x: torch.Tensor,
    address: torch.Tensor,
    cell_order: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, torch.Tensor], list[torch.Tensor]]:
    terminal_states: list[torch.Tensor] = []
    hooks = []

    def capture_terminal(
        _module: torch.nn.Module,
        _args: tuple[torch.Tensor, ...],
        _kwargs: dict[str, Any],
        output: tuple[torch.Tensor, torch.Tensor],
    ) -> None:
        terminal_states.append(output[1])

    for block in model.blocks:
        hooks.append(
            block.time_mix.register_forward_hook(
                capture_terminal,
                with_kwargs=True,
            )
        )
    try:
        with study.forward_autocast("bfloat16", x.device):
            output, diagnostics, _next_seed = model(
                x,
                address=address,
                cell_order=cell_order,
            )
    finally:
        for hook in hooks:
            hook.remove()
    if len(terminal_states) != LAYERS:
        raise AssertionError(
            f"captured {len(terminal_states)} terminal states, expected {LAYERS}"
        )
    return output, diagnostics, terminal_states


def assert_exact_tensor(left: torch.Tensor, right: torch.Tensor, label: str) -> None:
    if left.shape != right.shape or left.dtype != right.dtype:
        raise AssertionError(
            f"{label} schema changed: "
            f"{tuple(left.shape)}/{left.dtype} != {tuple(right.shape)}/{right.dtype}"
        )
    if not torch.equal(left, right):
        error = float((left.float() - right.float()).abs().max().item())
        raise AssertionError(f"{label} is not bitwise identical; max error={error}")


def check_forward_parameter_and_state_identity(
    device: torch.device,
) -> tuple[study.FutureSeedRWKV, dict[str, Any]]:
    torch.manual_seed(12_002)
    torch.cuda.manual_seed_all(12_002)
    control = build_model(device)
    torch.manual_seed(12_003)
    torch.cuda.manual_seed_all(12_003)
    candidate = build_model(device)
    candidate.load_state_dict(control.state_dict(), strict=True)

    control_parameters = dict(control.named_parameters())
    candidate_parameters = dict(candidate.named_parameters())
    if control_parameters.keys() != candidate_parameters.keys():
        raise AssertionError("training-only mechanism changed parameter names")
    for name in control_parameters:
        left = control_parameters[name]
        right = candidate_parameters[name]
        if left.requires_grad != right.requires_grad:
            raise AssertionError(f"parameter requires_grad changed: {name}")
        assert_exact_tensor(left, right, f"parameter {name}")

    control_state = control.state_dict()
    candidate_state = candidate.state_dict()
    if control_state.keys() != candidate_state.keys():
        raise AssertionError("training-only mechanism changed state_dict keys")
    for name in control_state:
        assert_exact_tensor(
            control_state[name],
            candidate_state[name],
            f"state_dict {name}",
        )

    torch.manual_seed(12_004)
    torch.cuda.manual_seed_all(12_004)
    x = torch.randn(2, TOKENS, MODEL_DIM, device=device)
    address = torch.randn_like(x)
    cell_order = torch.randperm(TOKENS, device=device)
    control.eval()
    candidate.eval()
    with torch.no_grad():
        control_output, control_diag, control_states = capture_forward(
            control,
            x,
            address,
            cell_order,
        )
        candidate_output, candidate_diag, candidate_states = capture_forward(
            candidate,
            x,
            address,
            cell_order,
        )
    assert_exact_tensor(control_output, candidate_output, "model output")
    state_errors = []
    for layer, (left, right) in enumerate(zip(control_states, candidate_states)):
        assert_exact_tensor(left, right, f"terminal state {layer}")
        state_errors.append(0.0)
    if control_diag.keys() != candidate_diag.keys():
        raise AssertionError("forward diagnostic schema changed")
    for name in control_diag:
        assert_exact_tensor(control_diag[name], candidate_diag[name], f"diagnostic {name}")

    parameter_count = sum(parameter.numel() for parameter in candidate.parameters())
    state_dict_elements = sum(value.numel() for value in candidate_state.values())
    del control, control_output, control_diag, control_states
    candidate.train()
    return candidate, {
        "forward_exact_identity": True,
        "terminal_state_max_abs_errors": state_errors,
        "parameter_names_exact_identity": True,
        "parameter_count_exact_identity": True,
        "parameter_count": parameter_count,
        "state_dict_exact_identity": True,
        "state_dict_key_count": len(candidate_state),
        "state_dict_elements": state_dict_elements,
    }


def check_official_runtime(
    model: study.FutureSeedRWKV,
) -> dict[str, Any]:
    if study.GAIN_BUDGET_FLA_SHA != EXPECTED_FLA_SOURCE_SHA:
        raise RuntimeError(
            "trainer pinned FLA SHA drift: "
            f"{study.GAIN_BUDGET_FLA_SHA} != {EXPECTED_FLA_SOURCE_SHA}"
        )
    source_sha, source_root, source_module = study.resolve_strict_fla_source("gdn2")
    if source_sha != EXPECTED_FLA_SOURCE_SHA:
        raise RuntimeError(
            f"runtime pinned FLA SHA drift: {source_sha} != {EXPECTED_FLA_SOURCE_SHA}"
        )

    holder = type("StrictRuntimeHolder", (), {"reasoner": model})()
    runtime = study.strict_fla_runtime_summary(holder, "gdn2")
    rows = runtime["layers"]
    if len(rows) != LAYERS:
        raise AssertionError(f"official layer count {len(rows)} != {LAYERS}")
    expected_class = "fla.layers.gdn2.GatedDeltaNet2"
    for layer, (block, row) in enumerate(zip(model.blocks, rows)):
        if type(block.time_mix.core) is not study.GatedDeltaNet2:
            raise AssertionError(
                f"layer {layer} bypassed exact official GatedDeltaNet2 class"
            )
        if row["class"] != expected_class:
            raise AssertionError(
                f"layer {layer} official class drift: {row['class']}"
            )
        if row["address_mode"] != "position_qk":
            raise AssertionError(
                f"layer {layer} address mode drift: {row['address_mode']}"
            )
        if row["execution_path"] != "canonical_position_qk_then_official_gdn2_chunk":
            raise AssertionError(
                f"layer {layer} execution path drift: {row['execution_path']}"
            )
        if set(row["conv_backends"].values()) != {"triton"}:
            raise AssertionError(
                f"layer {layer} short-conv backend drift: {row['conv_backends']}"
            )
    return {
        "fla_source_sha": source_sha,
        "fla_source_root": source_root,
        "fla_source_module": source_module,
        "official_gdn2_layers": len(rows),
        "official_classes": [row["class"] for row in rows],
        "execution_paths": [row["execution_path"] for row in rows],
        "backend_dispatch_disabled": runtime["backend_dispatch_disabled"],
        "conv_backend": runtime["conv_backend"],
    }


def check_gate_topology_and_official_backward(
    model: study.FutureSeedRWKV,
    device: torch.device,
) -> dict[str, Any]:
    gate_parameters = study.future_seed_gate_parameters(model)
    names = [name for name, _parameter in gate_parameters]
    expected_names = [f"blocks.{layer}.future_seed_logit" for layer in range(LAYERS)]
    if names != expected_names:
        raise AssertionError(f"FutureSeed gate names changed: {names}")
    tensor_count = len(gate_parameters)
    parameter_count = sum(parameter.numel() for _name, parameter in gate_parameters)
    if tensor_count != GATE_TENSORS or parameter_count != GATE_PARAMETERS:
        raise AssertionError(
            f"FutureSeed gate topology is {tensor_count}/{parameter_count}, "
            f"expected {GATE_TENSORS}/{GATE_PARAMETERS}"
        )
    if any(tuple(parameter.shape) != (1, HEADS, 1, 1) for _, parameter in gate_parameters):
        raise AssertionError("FutureSeed gate tensor shape changed")

    torch.manual_seed(12_005)
    torch.cuda.manual_seed_all(12_005)
    model.zero_grad(set_to_none=True)
    x = torch.randn(1, TOKENS, MODEL_DIM, device=device, requires_grad=True)
    address = torch.randn_like(x)
    cell_order = torch.randperm(TOKENS, device=device)
    output, _diag, terminal_states = capture_forward(
        model,
        x,
        address,
        cell_order,
    )
    graph = graph_names(output)
    official_backward_count = graph.count("ChunkGDN2FunctionBackward")
    if official_backward_count != LAYERS:
        raise AssertionError(
            "expected one ChunkGDN2FunctionBackward per official layer, got "
            f"{official_backward_count}"
        )
    loss = output.float().square().mean()
    gate_grads = torch.autograd.grad(
        loss,
        [parameter for _name, parameter in gate_parameters],
        retain_graph=True,
        allow_unused=True,
    )
    active = []
    for (name, parameter), gradient in zip(gate_parameters, gate_grads):
        if gradient is None:
            continue
        if gradient.shape != parameter.shape or not bool(torch.isfinite(gradient).all()):
            raise AssertionError(f"FutureSeed gate gradient invalid: {name}")
        if float(gradient.float().abs().max().item()) <= 0.0:
            raise AssertionError(f"FutureSeed gate gradient is zero: {name}")
        active.append(name)
    expected_active = expected_names[1:]
    if active != expected_active:
        raise AssertionError(
            f"active receiving gates changed: {active} != {expected_active}"
        )
    active_parameters = sum(
        parameter.numel()
        for (_name, parameter), gradient in zip(gate_parameters, gate_grads)
        if gradient is not None
    )
    if len(active) != ACTIVE_GATE_TENSORS or active_parameters != ACTIVE_GATE_PARAMETERS:
        raise AssertionError(
            f"active gate topology is {len(active)}/{active_parameters}, "
            f"expected {ACTIVE_GATE_TENSORS}/{ACTIVE_GATE_PARAMETERS}"
        )

    loss.backward()
    if x.grad is None or not bool(torch.isfinite(x.grad).all()):
        raise AssertionError("full-size input gradient is missing or non-finite")
    q_gradient_max = []
    for layer, block in enumerate(model.blocks):
        gradient = block.time_mix.core.q_proj.weight.grad
        if gradient is None or not bool(torch.isfinite(gradient).all()):
            raise AssertionError(f"official GDN2 Q gradient missing at layer {layer}")
        value = float(gradient.float().abs().max().item())
        if value <= 0.0:
            raise AssertionError(f"official GDN2 Q gradient is zero at layer {layer}")
        q_gradient_max.append(value)
    if any(not bool(torch.isfinite(state).all()) for state in terminal_states):
        raise AssertionError("official GDN2 terminal state became non-finite")
    model.zero_grad(set_to_none=True)
    return {
        "gate_tensor_count": tensor_count,
        "gate_parameter_count": parameter_count,
        "active_receiving_tensor_count": len(active),
        "active_receiving_parameter_count": active_parameters,
        "inactive_gate": expected_names[0],
        "active_gates": active,
        "chunk_gdn2_backward_count": official_backward_count,
        "q_gradient_max_abs_by_layer": q_gradient_max,
    }


def projection_vectors(device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    continuation = torch.linspace(
        0.5,
        1.5,
        ACTIVE_GATE_PARAMETERS,
        device=device,
        dtype=torch.float64,
    )
    raw_orthogonal = torch.cos(
        torch.arange(ACTIVE_GATE_PARAMETERS, device=device, dtype=torch.float64)
    )
    orthogonal = raw_orthogonal - (
        torch.dot(raw_orthogonal, continuation)
        / torch.dot(continuation, continuation)
    ) * continuation
    orthogonal = (
        orthogonal
        / torch.linalg.vector_norm(orthogonal)
        * torch.linalg.vector_norm(continuation)
    )
    opening = -0.75 * continuation + 0.40 * orthogonal
    baseline = (opening + 4.0 * continuation) / 5.0
    return opening, continuation, baseline


def expected_active_projection(
    opening: torch.Tensor,
    continuation: torch.Tensor,
    baseline: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    dot = torch.dot(opening, continuation)
    projected_opening = opening - (
        dot / torch.dot(continuation, continuation)
    ) * continuation
    raw = (projected_opening + 4.0 * continuation) / 5.0
    corrected = raw * (
        torch.linalg.vector_norm(baseline) / torch.linalg.vector_norm(raw)
    )
    return projected_opening, corrected


def make_synthetic_gate_parameters(
    baseline: torch.Tensor,
) -> list[tuple[str, torch.nn.Parameter]]:
    if baseline.numel() != ACTIVE_GATE_PARAMETERS:
        raise AssertionError("synthetic gate vector has the wrong size")
    rows = []
    cursor = 0
    for layer in range(LAYERS):
        parameter = torch.nn.Parameter(
            torch.zeros(
                1,
                HEADS,
                1,
                1,
                device=baseline.device,
                dtype=baseline.dtype,
            )
        )
        if layer > 0:
            parameter.grad = baseline[cursor : cursor + HEADS].reshape_as(parameter).clone()
            cursor += HEADS
        rows.append((f"blocks.{layer}.future_seed_logit", parameter))
    if cursor != baseline.numel():
        raise AssertionError("synthetic gate slicing failed")
    return rows


def split_synthetic_opening(
    opening: torch.Tensor,
) -> list[torch.Tensor | None]:
    gradients: list[torch.Tensor | None] = [None]
    gradients.extend(chunk.reshape(1, HEADS, 1, 1).clone() for chunk in opening.split(HEADS))
    if len(gradients) != LAYERS:
        raise AssertionError("synthetic opening-gradient slicing failed")
    return gradients


def check_projection_algebra_and_isolation(
    device: torch.device,
) -> dict[str, Any]:
    opening, continuation, baseline = projection_vectors(device)
    recovered_continuation = (5.0 * baseline - opening) / 4.0
    recovery_error = float(
        (recovered_continuation - continuation).abs().max().item()
    )
    if recovery_error > 1e-12:
        raise AssertionError(
            f"continuation recovery formula mismatch: {recovery_error}"
        )
    if float(torch.dot(opening, continuation).item()) >= 0.0:
        raise AssertionError("synthetic active case is not conflicting")
    expected_opening, expected_corrected = expected_active_projection(
        opening,
        continuation,
        baseline,
    )
    corrected, active_diag = study.project_future_seed_opening_gradient(
        opening,
        baseline,
    )
    active_error = float((corrected - expected_corrected).abs().max().item())
    if active_diag["active"] != 1.0 or active_error > 1e-12:
        raise AssertionError(
            f"active projection formula mismatch: active={active_diag['active']} "
            f"error={active_error}"
        )
    baseline_norm = torch.linalg.vector_norm(baseline)
    corrected_norm = torch.linalg.vector_norm(corrected)
    norm_error = float((corrected_norm - baseline_norm).abs().item())
    post_dot = float(torch.dot(expected_opening, continuation).item())
    if norm_error > 1e-12 or post_dot < -1e-10:
        raise AssertionError(
            f"active projection contract failed: norm_error={norm_error} post_dot={post_dot}"
        )

    inactive_opening = baseline.clone()
    inactive_corrected, inactive_diag = study.project_future_seed_opening_gradient(
        inactive_opening,
        baseline,
    )
    if inactive_diag["active"] != 0.0 or not torch.equal(inactive_corrected, baseline):
        raise AssertionError("pure no-conflict path changed the baseline gradient")

    inactive_parameters = make_synthetic_gate_parameters(baseline)
    inactive_openings = split_synthetic_opening(baseline)
    inactive_versions = [
        None if parameter.grad is None else parameter.grad._version
        for _name, parameter in inactive_parameters
    ]
    inactive_snapshots = [
        None if parameter.grad is None else parameter.grad.clone()
        for _name, parameter in inactive_parameters
    ]
    inactive_apply_diag = study.apply_future_seed_opening_projection(
        inactive_parameters,
        inactive_openings,
    )
    if inactive_apply_diag["active"] != 0.0:
        raise AssertionError("no-conflict apply path unexpectedly activated")
    for index, (_name, parameter) in enumerate(inactive_parameters):
        snapshot = inactive_snapshots[index]
        if snapshot is None:
            if parameter.grad is not None:
                raise AssertionError("inactive layer-0 gate acquired a gradient")
            continue
        if parameter.grad is None or not torch.equal(parameter.grad, snapshot):
            raise AssertionError(f"no-conflict path changed gate gradient {index}")
        if parameter.grad._version != inactive_versions[index]:
            raise AssertionError(f"no-conflict path wrote gate gradient {index}")

    apply_expected, _apply_expected_diag = (
        study.project_future_seed_opening_gradient(
            opening.float(),
            baseline.float(),
        )
    )
    active_parameters = make_synthetic_gate_parameters(baseline.float())
    active_openings = split_synthetic_opening(opening.float())
    non_fs = torch.nn.Parameter(
        torch.linspace(0.1, 0.8, HEADS, device=device, dtype=torch.float64)
    )
    non_fs.grad = torch.linspace(-0.4, 0.3, HEADS, device=device, dtype=torch.float64)
    non_fs_snapshot = non_fs.grad.clone()
    non_fs_version = non_fs.grad._version
    gate_versions = [
        None if parameter.grad is None else parameter.grad._version
        for _name, parameter in active_parameters
    ]
    active_apply_diag = study.apply_future_seed_opening_projection(
        active_parameters,
        active_openings,
    )
    active_result = torch.cat(
        [
            parameter.grad.reshape(-1)
            for _name, parameter in active_parameters[1:]
            if parameter.grad is not None
        ]
    )
    apply_error = float((active_result - apply_expected).abs().max().item())
    if active_apply_diag["active"] != 1.0 or apply_error != 0.0:
        raise AssertionError(
            f"active apply path mismatch: active={active_apply_diag['active']} "
            f"error={apply_error}"
        )
    if active_parameters[0][1].grad is not None:
        raise AssertionError("active apply path created a layer-0 gradient")
    for index, (_name, parameter) in enumerate(active_parameters[1:], start=1):
        if parameter.grad is None or parameter.grad._version <= gate_versions[index]:
            raise AssertionError(f"active apply path did not write gate gradient {index}")
    if non_fs.grad._version != non_fs_version or not torch.equal(
        non_fs.grad,
        non_fs_snapshot,
    ):
        raise AssertionError("FutureSeed projection changed a non-FS gradient")
    if (
        active_apply_diag["active_tensor_count"] != ACTIVE_GATE_TENSORS
        or active_apply_diag["active_parameter_count"] != ACTIVE_GATE_PARAMETERS
        or active_apply_diag["inactive_tensor_count"] != 1.0
    ):
        raise AssertionError(f"active apply topology mismatch: {active_apply_diag}")
    if active_apply_diag["post_opening_continuation_dot"] < -1e-10:
        raise AssertionError("active apply projection missed the continuation half-space")
    if active_apply_diag["norm_relative_error"] >= 1e-5:
        raise AssertionError("active apply projection changed baseline norm")

    clipped_parameters = make_synthetic_gate_parameters((3.0 * baseline).float())
    clipped_non_fs = torch.nn.Parameter(
        torch.linspace(2.0, 3.0, HEADS, device=device, dtype=torch.float32)
    )
    clipped_non_fs.grad = torch.linspace(
        -4.0, 5.0, HEADS, device=device, dtype=torch.float32
    )
    pre_clip_norm = torch.nn.utils.clip_grad_norm_(
        [
            *(parameter for _name, parameter in clipped_parameters),
            clipped_non_fs,
        ],
        1.0,
    )
    clip_coefficient = min(1.0, 1.0 / (float(pre_clip_norm.item()) + 1e-6))
    clipped_non_fs_snapshot = clipped_non_fs.grad.clone()
    canonical_clipped_norm = float(
        study.current_global_gradient_norm(
            [
                *(parameter for _name, parameter in clipped_parameters),
                clipped_non_fs,
            ]
        ).item()
    )
    clipped_opening = split_synthetic_opening((3.0 * opening).float())
    clipped_opening = [
        None if gradient is None else gradient * clip_coefficient
        for gradient in clipped_opening
    ]
    study.apply_future_seed_opening_projection(
        clipped_parameters,
        clipped_opening,
    )
    if not torch.equal(clipped_non_fs.grad, clipped_non_fs_snapshot):
        raise AssertionError("projection changed a canonically clipped non-FS gradient")
    final_global_norm = float(
        study.current_global_gradient_norm(
            [
                *(parameter for _name, parameter in clipped_parameters),
                clipped_non_fs,
            ]
        ).item()
    )
    clipped_norm_relative_error = abs(
        final_global_norm - canonical_clipped_norm
    ) / max(canonical_clipped_norm, 1e-30)
    if clipped_norm_relative_error > 2e-5:
        raise AssertionError(
            "projection changed the canonical clipped global norm: "
            f"{clipped_norm_relative_error}"
        )
    return {
        "continuation_recovery_max_abs_error": recovery_error,
        "active_formula_max_abs_error": active_error,
        "active_apply_max_abs_error": apply_error,
        "active_diagnostics": active_apply_diag,
        "baseline_norm": float(baseline_norm.item()),
        "corrected_norm": float(corrected_norm.item()),
        "norm_abs_error": norm_error,
        "post_opening_continuation_dot": post_dot,
        "inactive_pure_bitwise_identity": True,
        "inactive_apply_no_write": True,
        "inactive_diagnostics": inactive_apply_diag,
        "non_future_seed_gradient_exact_identity": True,
        "canonical_clip_non_future_seed_exact_identity": True,
        "canonical_clip_global_norm_relative_error": clipped_norm_relative_error,
    }


def run_toy_accumulation(
    samples: torch.Tensor,
    *,
    accumulation_steps: int,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
    if samples.shape != (8, HEADS):
        raise AssertionError(f"unexpected toy sample shape: {tuple(samples.shape)}")
    if accumulation_steps not in {1, 4}:
        raise AssertionError("toy accumulation supports only 1 or 4 steps")
    gate = torch.nn.Parameter(
        torch.linspace(-0.2, 0.3, HEADS, device=samples.device, dtype=torch.float64)
    )
    non_fs = torch.nn.Parameter(
        torch.linspace(0.4, -0.1, HEADS, device=samples.device, dtype=torch.float64)
    )
    opening_coeff = torch.tensor(
        [-1.0, -0.8, -0.4, 0.2, 0.5, 0.7, 0.9, 1.1],
        device=samples.device,
        dtype=torch.float64,
    )
    continuation_coeff = torch.tensor(
        [1.2, 0.9, 0.5, 0.1, -0.2, -0.4, -0.6, -0.8],
        device=samples.device,
        dtype=torch.float64,
    )
    chunks = samples.chunk(accumulation_steps, dim=0)
    opening_accum: torch.Tensor | None = None
    for chunk in chunks:
        opening_term = (chunk * gate * opening_coeff).sum(dim=-1).mean()
        continuation_term = (
            chunk * gate * continuation_coeff
        ).sum(dim=-1).mean()
        non_fs_term = (chunk.square() * non_fs).sum(dim=-1).mean()
        loop_losses = [opening_term + 0.10 * non_fs_term]
        loop_losses.extend(
            continuation_term + coefficient * non_fs_term
            for coefficient in (0.20, 0.30, 0.40, 0.50)
        )
        opening_gradient = torch.autograd.grad(
            loop_losses[0] / float(accumulation_steps),
            [gate],
            retain_graph=True,
        )[0]
        if opening_accum is None:
            opening_accum = opening_gradient.detach().clone()
        else:
            opening_accum.add_(opening_gradient.detach())
        (
            torch.stack(loop_losses).mean() / float(accumulation_steps)
        ).backward()
    if opening_accum is None or gate.grad is None or non_fs.grad is None:
        raise AssertionError("toy accumulation produced missing gradients")
    diagnostics = study.apply_future_seed_opening_projection(
        [("toy.future_seed_logit", gate)],
        [opening_accum],
    )
    if diagnostics["active"] != 1.0:
        raise AssertionError(
            f"toy accumulation did not activate projection: {diagnostics}"
        )
    return gate.grad.detach().clone(), non_fs.grad.detach().clone(), diagnostics


def check_accumulation_equivalence(device: torch.device) -> dict[str, Any]:
    samples = (
        torch.arange(8 * HEADS, device=device, dtype=torch.float64)
        .reshape(8, HEADS)
        .div(64.0)
        .add(0.5)
    )
    gate_one, non_fs_one, diag_one = run_toy_accumulation(
        samples,
        accumulation_steps=1,
    )
    gate_four, non_fs_four, diag_four = run_toy_accumulation(
        samples,
        accumulation_steps=4,
    )
    gate_error = float((gate_one - gate_four).abs().max().item())
    non_fs_error = float((non_fs_one - non_fs_four).abs().max().item())
    if gate_error > 1e-12 or non_fs_error > 1e-12:
        raise AssertionError(
            "gradient accumulation changed the projected update: "
            f"gate={gate_error} non_fs={non_fs_error}"
        )
    for key in (
        "opening_continuation_cosine",
        "post_opening_continuation_dot",
        "relative_correction",
        "norm_relative_error",
    ):
        if abs(diag_one[key] - diag_four[key]) > 1e-12:
            raise AssertionError(
                f"gradient accumulation changed projection diagnostic {key}: "
                f"{diag_one[key]} != {diag_four[key]}"
            )
    return {
        "accumulation_1_vs_4_gate_max_abs_error": gate_error,
        "accumulation_1_vs_4_non_fs_max_abs_error": non_fs_error,
        "accumulation_1_diagnostics": diag_one,
        "accumulation_4_diagnostics": diag_four,
    }


def visible_gpu_contract() -> tuple[str, str]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required; CPU model smoke is forbidden")
    if torch.cuda.device_count() != 1:
        raise RuntimeError(
            f"exactly one visible CUDA device is required, got {torch.cuda.device_count()}"
        )
    rows = [
        row.strip()
        for row in subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=index,uuid",
                "--format=csv,noheader",
            ],
            text=True,
        ).splitlines()
        if row.strip()
    ]
    expected = [f"0, {EXPECTED_GPU_UUID}"]
    if rows != expected:
        raise RuntimeError(f"unexpected visible GPU contract: {rows} != {expected}")
    device = torch.device("cuda:0")
    return EXPECTED_GPU_UUID, torch.cuda.get_device_name(device)


def compute_app_contract() -> list[str]:
    rows = [
        row.strip()
        for row in subprocess.check_output(
            [
                "nvidia-smi",
                "--query-compute-apps=gpu_uuid,pid",
                "--format=csv,noheader",
            ],
            text=True,
        ).splitlines()
        if row.strip()
    ]
    if len(rows) != 1 or not rows[0].startswith(EXPECTED_GPU_UUID):
        raise RuntimeError(f"unexpected concurrent compute applications: {rows}")
    return rows


def main() -> None:
    gpu_uuid, gpu_name = visible_gpu_contract()
    if tuple(study.FUTURE_SEED_GRADIENT_MODES) != (
        "canonical",
        "opening_projection",
    ):
        raise RuntimeError(
            "trainer FutureSeed gradient modes are not canonical|opening_projection"
        )
    device = torch.device("cuda:0")
    torch.cuda.reset_peak_memory_stats(device)
    model, identity = check_forward_parameter_and_state_identity(device)
    provenance = check_official_runtime(model)
    topology = check_gate_topology_and_official_backward(model, device)
    projection = check_projection_algebra_and_isolation(device)
    accumulation = check_accumulation_equivalence(device)
    torch.cuda.synchronize(device)
    result = {
        "status": "passed",
        "device": str(device),
        "gpu_name": gpu_name,
        "gpu_uuid": gpu_uuid,
        "future_seed_gradient_modes": list(study.FUTURE_SEED_GRADIENT_MODES),
        "fla_source_sha": EXPECTED_FLA_SOURCE_SHA,
        "identity": identity,
        "official_runtime": provenance,
        "gate_topology_and_backward": topology,
        "projection_algebra_and_isolation": projection,
        "gradient_accumulation_equivalence": accumulation,
        "peak_allocated_mib": torch.cuda.max_memory_allocated(device) / (1024**2),
        "peak_reserved_mib": torch.cuda.max_memory_reserved(device) / (1024**2),
        "compute_apps": compute_app_contract(),
    }
    output_path = os.environ.get("CONTRACT_OUTPUT", "").strip()
    if output_path:
        pathlib.Path(output_path).write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
