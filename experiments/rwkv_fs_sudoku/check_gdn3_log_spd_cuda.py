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

from gdn3_log_spd import BoundedLogSPDAddressMetric
from study_rwkv_futureseed_loop import (
    FutureSeedRWKV,
    GAIN_BUDGET_FLA_SHA,
    resolve_strict_fla_source,
)


EXPECTED_GPU_UUID = os.environ.get("EXPECTED_GPU_UUID", "").strip()
EXPECTED_FLA_SOURCE_SHA = "9c8e42e762fce087c27b673af4922795d9edb85e"
LAYERS = 12
HEADS = 8
HEAD_DIM = 32
EXPECTED_PARAMETER_DELTA = LAYERS * HEADS * (
    HEAD_DIM * (HEAD_DIM + 1) // 2 - 1
)


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


def copy_parent(control: FutureSeedRWKV, candidate: FutureSeedRWKV) -> list[str]:
    missing, unexpected = candidate.load_state_dict(control.state_dict(), strict=False)
    expected = {
        f"blocks.{layer}.time_mix.log_spd_address_metric.raw"
        for layer in range(LAYERS)
    }
    if set(missing) != expected or unexpected:
        raise AssertionError(
            f"unexpected state migration: missing={missing}, unexpected={unexpected}"
        )
    return sorted(missing)


def capture(
    model: FutureSeedRWKV,
    hidden: torch.Tensor,
    address: torch.Tensor,
    cell_order: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, torch.Tensor], list[torch.Tensor]]:
    states: list[torch.Tensor] = []
    hooks = []

    def save_state(
        _module: torch.nn.Module,
        _args: tuple[torch.Tensor, ...],
        _kwargs: dict[str, Any],
        output: tuple[torch.Tensor, torch.Tensor],
    ) -> None:
        states.append(output[1])

    for block in model.blocks:
        hooks.append(block.time_mix.register_forward_hook(save_state, with_kwargs=True))
    try:
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            output, diagnostics, _ = model(
                hidden,
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
    torch.manual_seed(93401)
    for layer, (control_block, candidate_block) in enumerate(
        zip(control.blocks, candidate.blocks)
    ):
        hidden = torch.randn(1, 81, 256, device=device)
        address = torch.randn_like(hidden)
        order = torch.randperm(81, device=device)
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
                hidden,
                initial_state=initial_state,
                address=address,
                cell_order=order,
            )
            candidate_output, candidate_state = candidate_block.time_mix(
                hidden,
                initial_state=initial_state,
                address=address,
                cell_order=order,
            )
        output_error = float(
            (control_output.float() - candidate_output.float()).abs().max().item()
        )
        state_error = float(
            (control_state.float() - candidate_state.float()).abs().max().item()
        )
        if output_error != 0.0 or state_error != 0.0:
            raise AssertionError(
                f"layer {layer} nonzero-state identity failed: "
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


def check_metric_geometry(device: torch.device) -> dict[str, Any]:
    torch.manual_seed(93402)
    metric = BoundedLogSPDAddressMetric(HEADS, HEAD_DIM).to(device)
    with torch.no_grad():
        metric.raw.normal_(mean=0.0, std=0.25)
    query = torch.randn(2, 19, HEADS, HEAD_DIM, device=device)
    key = torch.randn_like(query)
    transformed_query, transformed_key = metric.transform_pair(query, key)

    permutation = torch.randperm(HEADS, device=device)
    permuted_metric = BoundedLogSPDAddressMetric(HEADS, HEAD_DIM).to(device)
    with torch.no_grad():
        permuted_metric.raw.copy_(metric.raw.index_select(0, permutation))
    permuted_query, permuted_key = permuted_metric.transform_pair(
        query.index_select(-2, permutation),
        key.index_select(-2, permutation),
    )
    query_error = float(
        (
            permuted_query
            - transformed_query.index_select(-2, permutation)
        ).abs().max().item()
    )
    key_error = float(
        (
            permuted_key
            - transformed_key.index_select(-2, permutation)
        ).abs().max().item()
    )
    if query_error > 2e-6 or key_error > 2e-6:
        raise AssertionError(
            f"head permutation equivariance failed: q={query_error} k={key_error}"
        )
    diagnostics = metric.diagnostics(torch.bfloat16)
    eigen_min = float(diagnostics["gdn3_log_spd_eigenvalue_min"].item())
    eigen_max = float(diagnostics["gdn3_log_spd_eigenvalue_max"].item())
    condition = float(diagnostics["gdn3_log_spd_condition_max"].item())
    logdet_error = float(diagnostics["gdn3_log_spd_logdet_abs_max"].item())
    if eigen_min < 0.49 or eigen_max > 2.01 or condition > 4.05:
        raise AssertionError(
            f"bounded SPD contract failed: min={eigen_min} max={eigen_max} cond={condition}"
        )
    if logdet_error > 0.02:
        raise AssertionError(f"BF16 metric logdet drift is too large: {logdet_error}")
    return {
        "query_head_permutation_max_abs_error": query_error,
        "key_head_permutation_max_abs_error": key_error,
        **{key: float(value.item()) for key, value in diagnostics.items()},
    }


def check_model_contract(device: torch.device) -> dict[str, Any]:
    torch.manual_seed(93400)
    control = build(device, update_mode="none")
    torch.manual_seed(93400)
    candidate = build(device, update_mode="log_spd_metric")
    missing = copy_parent(control, candidate)
    control.eval()
    candidate.eval()

    parameter_delta = sum(p.numel() for p in candidate.parameters()) - sum(
        p.numel() for p in control.parameters()
    )
    if parameter_delta != EXPECTED_PARAMETER_DELTA:
        raise AssertionError(
            f"parameter delta {parameter_delta} != {EXPECTED_PARAMETER_DELTA}"
        )
    metrics = [
        block.time_mix.log_spd_address_metric for block in candidate.blocks
    ]
    if any(metric is None for metric in metrics):
        raise AssertionError("one or more per-layer metrics are missing")
    pointers = [int(metric.raw.data_ptr()) for metric in metrics if metric is not None]
    if len(set(pointers)) != LAYERS:
        raise AssertionError(f"per-layer metrics unexpectedly share storage: {pointers}")

    hidden = torch.randn(2, 81, 256, device=device)
    address = torch.randn_like(hidden)
    order = torch.randperm(81, device=device)
    with torch.no_grad():
        control_output, _control_diag, control_states = capture(
            control, hidden, address, order
        )
        candidate_output, candidate_diag, candidate_states = capture(
            candidate, hidden, address, order
        )
    if not torch.equal(control_output, candidate_output):
        raise AssertionError("zero-init Log-SPD changed the full model output")
    state_errors = [
        float((left.float() - right.float()).abs().max().item())
        for left, right in zip(control_states, candidate_states)
    ]
    if len(state_errors) != LAYERS or max(state_errors) != 0.0:
        raise AssertionError(f"zero-init terminal-state mismatch: {state_errors}")
    if float(candidate_diag["gdn3_log_spd_metric_delta_fro"].item()) != 0.0:
        raise AssertionError("zero-init applied metric is not identity")
    initial_state_rows = check_initial_state_identity(control, candidate, device)

    candidate.train()
    candidate.zero_grad(set_to_none=True)
    output, diagnostics, states = capture(
        candidate,
        hidden.detach().clone().requires_grad_(True),
        address,
        order,
    )
    (output.float() * torch.randn_like(output, dtype=torch.float32)).mean().backward()
    gradients: dict[str, float] = {}
    official_backward_layers = 0
    for layer, (metric, state) in enumerate(zip(metrics, states)):
        assert metric is not None
        gradient = metric.raw.grad
        if gradient is None or not bool(torch.isfinite(gradient).all()):
            raise AssertionError(f"layer {layer} metric gradient missing/non-finite")
        grad_max = float(gradient.float().abs().max().item())
        if grad_max <= 0.0:
            raise AssertionError(f"layer {layer} metric gradient is zero")
        gradients[str(layer)] = grad_max
        if "ChunkGDN2FunctionBackward" in graph_names(state):
            official_backward_layers += 1
    if official_backward_layers != LAYERS:
        raise AssertionError(
            f"official chunk backward reached {official_backward_layers}/{LAYERS} layers"
        )
    official_layers = [type(block.time_mix.core).__name__ for block in candidate.blocks]
    if official_layers != ["GatedDeltaNet2"] * LAYERS:
        raise AssertionError(f"unexpected recurrent layers: {official_layers}")

    with torch.no_grad():
        for layer, metric in enumerate(metrics):
            assert metric is not None
            torch.manual_seed(93500 + layer)
            metric.raw.normal_(mean=0.0, std=0.03)
        opened_output, opened_diag, _ = capture(candidate, hidden, address, order)
    output_dependency = float(
        (opened_output.float() - control_output.float()).abs().max().item()
    )
    if output_dependency <= 1e-5:
        raise AssertionError("opened Log-SPD metric has no model-output dependency")
    if float(opened_diag["gdn3_log_spd_enabled"].item()) != 1.0:
        raise AssertionError("Log-SPD diagnostics did not mark the path active")
    return {
        "output_exact_identity": True,
        "terminal_state_max_abs_errors": state_errors,
        "nonzero_initial_state_identity": initial_state_rows,
        "missing_parameters": missing,
        "parameter_delta": parameter_delta,
        "state_delta": 0,
        "scan_delta": 0,
        "per_layer_parameter_pointers": pointers,
        "per_layer_storage_independent": True,
        "gradient_max_abs_by_layer": gradients,
        "official_chunk_backward_layers": official_backward_layers,
        "official_gdn2_layers": official_layers,
        "position_qk_active": float(diagnostics["gdn2_address_enabled"].item()),
        "opened_output_max_abs_change": output_dependency,
        "opened_diagnostics": {
            key: float(value.detach().float().item())
            for key, value in opened_diag.items()
            if key.startswith("gdn3_log_spd_")
        },
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
    source_sha, source_root, source_module = resolve_strict_fla_source("gdn2")
    if source_sha != EXPECTED_FLA_SOURCE_SHA:
        raise RuntimeError(f"unexpected FLA source SHA: {source_sha}")
    device = torch.device("cuda:0")
    props = torch.cuda.get_device_properties(device)
    output = {
        "device": str(device),
        "gpu_name": props.name,
        "gpu_uuid": EXPECTED_GPU_UUID,
        "fla_source_sha": source_sha,
        "fla_source_root": source_root,
        "fla_source_module": source_module,
        "model_contract": check_model_contract(device),
        "metric_geometry": check_metric_geometry(device),
    }
    print(json.dumps(output, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
