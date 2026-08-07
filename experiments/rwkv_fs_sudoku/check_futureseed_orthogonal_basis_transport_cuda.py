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

from futureseed3_orthogonal_basis_transport import (
    FutureSeedOrthogonalBasisTransport,
    basis_transport_parameter_count,
)
import study_rwkv_futureseed_loop as study


EXPECTED_GPU_UUID = "GPU-53e9f3b4-2966-65d3-6614-09c540921519"
EXPECTED_FLA_SOURCE_SHA = "9c8e42e762fce087c27b673af4922795d9edb85e"
LAYERS = 12
EDGES = LAYERS - 1
HEADS = 8
HEAD_DIM = 32
MODEL_DIM = HEADS * HEAD_DIM
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


def build(device: torch.device, *, content_mode: str) -> study.FutureSeedRWKV:
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
        future_seed_content_mode=content_mode,
        backbone="gdn2",
        gdn_mode="chunk",
        gdn_expand_v=1.0,
        gdn_progressive_base_expand_v=0.0,
        gdn_use_short_conv=True,
        gdn_conv_size=4,
        gdn_allow_neg_eigval=False,
        gdn2_address_mode="position_qk",
        gdn2_update_mode="none",
        gdn2_state_expert_mode="none",
        gdn2_cross_layer_init="independent",
    ).to(device)


def copy_shared_state(
    control: study.FutureSeedRWKV,
    candidate: study.FutureSeedRWKV,
) -> list[str]:
    missing, unexpected = candidate.load_state_dict(
        control.state_dict(), strict=False
    )
    module = candidate.future_seed_basis_transport
    if module is None:
        raise AssertionError("candidate basis-transport module is missing")
    expected = {
        f"future_seed_basis_transport.{name}"
        for name, _parameter in module.named_parameters()
    }
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


def check_identity_and_first_order(device: torch.device) -> dict[str, Any]:
    torch.manual_seed(1514)
    control = build(device, content_mode="terminal")
    torch.manual_seed(1514)
    candidate = build(device, content_mode="orthogonal_basis_transport")
    missing = copy_shared_state(control, candidate)
    control.eval()
    candidate.eval()

    x = torch.randn(2, TOKENS, MODEL_DIM, device=device)
    address = torch.randn_like(x)
    cell_order = torch.randperm(TOKENS, device=device)
    with torch.no_grad():
        control_output, _control_diag, control_states = capture_model(
            control, x, address, cell_order
        )
        candidate_output, initial_diag, candidate_states = capture_model(
            candidate, x, address, cell_order
        )
    if not torch.equal(control_output, candidate_output):
        raise AssertionError("zero-init basis transport changed full model output")
    state_errors = [
        float((left.float() - right.float()).abs().max().item())
        for left, right in zip(control_states, candidate_states)
    ]
    if len(state_errors) != LAYERS or max(state_errors) != 0.0:
        raise AssertionError(f"zero-init terminal-state mismatch: {state_errors}")

    module = candidate.future_seed_basis_transport
    assert module is not None
    random_state = torch.randn(
        3, HEADS, HEAD_DIM, HEAD_DIM, device=device, dtype=torch.bfloat16
    )
    with torch.no_grad():
        for edge_idx in range(EDGES):
            transported, transport_diag = module(random_state, edge_idx=edge_idx)
            if not torch.equal(random_state, transported):
                raise AssertionError(
                    f"zero-init nonzero-state identity failed on edge {edge_idx}"
                )
            if float(
                transport_diag[
                    "fs3_basis_transport_state_residual_relative_rms"
                ].item()
            ) != 0.0:
                raise AssertionError("zero-init transport residual is nonzero")

    expected_delta = basis_transport_parameter_count(
        edges=EDGES,
        heads=HEADS,
        row_dim=HEAD_DIM,
        col_dim=HEAD_DIM,
    )
    parameter_delta = sum(p.numel() for p in candidate.parameters()) - sum(
        p.numel() for p in control.parameters()
    )
    if parameter_delta != expected_delta or expected_delta != 87296:
        raise AssertionError(
            f"basis parameter delta {parameter_delta} != {expected_delta}"
        )

    candidate.train()
    candidate.zero_grad(set_to_none=True)
    output, _diag, states = capture_model(
        candidate,
        x.detach().clone().requires_grad_(True),
        address,
        cell_order,
    )
    target = torch.randn_like(output, dtype=torch.float32)
    (output.float() * target).mean().backward()
    gradients = {}
    for name in ("row_angles", "col_angles"):
        grad = getattr(module, name).grad
        if grad is None or not bool(torch.isfinite(grad).all()):
            raise AssertionError(f"basis gradient is missing or non-finite: {name}")
        edge_head_max = grad.float().abs().amax(dim=-1)
        if not bool((edge_head_max > 0).all()):
            raise AssertionError(
                f"basis first-order path is zero for an edge/head: {name}"
            )
        gradients[name] = {
            "min_edge_head_max_abs": float(edge_head_max.min().item()),
            "max_abs": float(edge_head_max.max().item()),
        }

    state_graphs = [graph_names(state) for state in states]
    if any("ChunkGDN2FunctionBackward" not in graph for graph in state_graphs):
        raise AssertionError("an official GDN2 chunk backward is missing")
    official_layers = [type(block.time_mix.core).__name__ for block in candidate.blocks]
    if official_layers != ["GatedDeltaNet2"] * LAYERS:
        raise AssertionError(f"unexpected recurrent layers: {official_layers}")
    return {
        "output_exact_identity": True,
        "terminal_state_max_abs_errors": state_errors,
        "nonzero_state_exact_identity": True,
        "missing_parameters": missing,
        "parameter_delta": parameter_delta,
        "direct_gradient_by_axis": gradients,
        "official_chunk_backward_by_layer": [True] * LAYERS,
        "official_gdn2_layers": official_layers,
        "initial_diag": {
            key: float(value.detach().float().item())
            for key, value in initial_diag.items()
            if key.startswith("fs3_basis_transport_")
        },
    }


def check_opened_geometry(device: torch.device) -> dict[str, Any]:
    torch.manual_seed(1515)
    module = FutureSeedOrthogonalBasisTransport(
        edges=1,
        heads=4,
        row_dim=16,
        col_dim=16,
    ).to(device)
    with torch.no_grad():
        module.row_angles.normal_(mean=0.0, std=0.02)
        module.col_angles.normal_(mean=0.0, std=0.02)
    state = torch.randn(5, 4, 16, 16, device=device, dtype=torch.bfloat16)
    transported, diagnostics = module(state, edge_idx=0)
    if torch.equal(state, transported):
        raise AssertionError("opened basis transport did not change state")
    fp32_error = float(
        diagnostics["fs3_basis_transport_fp32_norm_max_error"].float().item()
    )
    storage_error = float(
        diagnostics["fs3_basis_transport_storage_norm_max_error"].float().item()
    )
    orthogonality_error = float(
        diagnostics[
            "fs3_basis_transport_orthogonality_max_error"
        ].float().item()
    )
    if fp32_error > 3e-5:
        raise AssertionError(f"FP32 norm error is too large: {fp32_error}")
    if storage_error > 5e-3:
        raise AssertionError(f"storage norm error is too large: {storage_error}")
    if orthogonality_error > 3e-5:
        raise AssertionError(
            f"basis orthogonality error is too large: {orthogonality_error}"
        )

    permutation = torch.randperm(4, device=device)
    permuted_module = FutureSeedOrthogonalBasisTransport(
        edges=1,
        heads=4,
        row_dim=16,
        col_dim=16,
    ).to(device)
    with torch.no_grad():
        permuted_module.row_angles.copy_(module.row_angles[:, permutation])
        permuted_module.col_angles.copy_(module.col_angles[:, permutation])
    permuted_output, _ = permuted_module(
        state[:, permutation], edge_idx=0
    )
    head_error = float(
        (transported[:, permutation].float() - permuted_output.float())
        .abs()
        .max()
        .item()
    )
    if head_error > 3e-6:
        raise AssertionError(f"head permutation error is too large: {head_error}")

    activation = {
        key: float(value.detach().float().item())
        for key, value in diagnostics.items()
    }
    required_positive = (
        "fs3_basis_transport_angle_abs",
        "fs3_basis_transport_row_rotation_relative_rms",
        "fs3_basis_transport_col_rotation_relative_rms",
        "fs3_basis_transport_state_residual_relative_rms",
        "fs3_basis_transport_residual_batch_std",
        "fs3_basis_transport_residual_head_std",
    )
    if not all(activation[key] > 0.0 for key in required_positive):
        raise AssertionError(f"opened basis path is not active: {activation}")
    return {
        "activation": activation,
        "head_permutation_max_abs_error": head_error,
    }


def check_opened_full_model(device: torch.device) -> dict[str, Any]:
    torch.manual_seed(1516)
    control = build(device, content_mode="terminal")
    torch.manual_seed(1516)
    candidate = build(device, content_mode="orthogonal_basis_transport")
    copy_shared_state(control, candidate)
    module = candidate.future_seed_basis_transport
    assert module is not None
    with torch.no_grad():
        module.row_angles.normal_(mean=0.0, std=0.01)
        module.col_angles.normal_(mean=0.0, std=0.01)

    control.eval()
    candidate.eval()
    x = torch.randn(2, TOKENS, MODEL_DIM, device=device)
    address = torch.randn_like(x)
    cell_order = torch.randperm(TOKENS, device=device)
    with torch.no_grad():
        control_output, _control_diag, control_states = capture_model(
            control, x, address, cell_order
        )
        candidate_output, diagnostics, candidate_states = capture_model(
            candidate, x, address, cell_order
        )
    if torch.equal(control_output, candidate_output):
        raise AssertionError("opened full-model basis path did not change output")
    if not bool(torch.isfinite(candidate_output).all()):
        raise AssertionError("opened full-model output is non-finite")
    if len(control_states) != LAYERS or len(candidate_states) != LAYERS:
        raise AssertionError("opened full-model state capture is incomplete")

    state_rms_ratios = []
    for layer_idx, (control_state, candidate_state) in enumerate(
        zip(control_states, candidate_states)
    ):
        if not bool(torch.isfinite(candidate_state).all()):
            raise AssertionError(
                f"opened full-model state is non-finite at layer {layer_idx}"
            )
        control_rms = control_state.float().square().mean().sqrt().clamp_min(1e-6)
        candidate_rms = candidate_state.float().square().mean().sqrt()
        state_rms_ratios.append(float((candidate_rms / control_rms).item()))
    if max(state_rms_ratios) > 4.0:
        raise AssertionError(
            f"opened full-model terminal RMS ratio is unbounded: {state_rms_ratios}"
        )

    activation = {
        key: float(value.detach().float().item())
        for key, value in diagnostics.items()
        if key.startswith("fs3_basis_transport_")
    }
    for key in (
        "fs3_basis_transport_angle_abs_min",
        "fs3_basis_transport_row_rotation_relative_rms_min",
        "fs3_basis_transport_col_rotation_relative_rms_min",
        "fs3_basis_transport_state_residual_relative_rms_min",
        "fs3_basis_transport_residual_batch_std",
        "fs3_basis_transport_residual_head_std",
    ):
        if activation.get(key, 0.0) <= 0.0:
            raise AssertionError(
                f"opened full-model path is inactive for {key}: {activation}"
            )
    if activation.get("fs3_basis_transport_fp32_norm_max_error", 1.0) > 3e-5:
        raise AssertionError(f"opened full-model FP32 norm drift: {activation}")
    if activation.get("fs3_basis_transport_storage_norm_max_error", 1.0) > 5e-3:
        raise AssertionError(f"opened full-model storage norm drift: {activation}")
    if activation.get("fs3_basis_transport_orthogonality_max_error", 1.0) > 3e-5:
        raise AssertionError(f"opened full-model orthogonality drift: {activation}")
    return {
        "output_max_abs_change": float(
            (candidate_output.float() - control_output.float()).abs().max().item()
        ),
        "terminal_state_rms_ratios": state_rms_ratios,
        "activation": activation,
    }


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required; CPU model smoke is forbidden")
    if torch.cuda.device_count() != 1:
        raise RuntimeError(
            f"expected one visible CUDA device, got {torch.cuda.device_count()}"
        )
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "0":
        raise RuntimeError("CUDA_VISIBLE_DEVICES must be exactly 0")
    gpu_uuid = study.visible_gpu_uuid()
    if gpu_uuid != EXPECTED_GPU_UUID:
        raise RuntimeError(f"GPU UUID mismatch: {gpu_uuid}")
    fla_source_sha, fla_root, fla_module = study.resolve_strict_fla_source("gdn2")
    if fla_source_sha != EXPECTED_FLA_SOURCE_SHA:
        raise RuntimeError(f"FLA source mismatch: {fla_source_sha}")
    git_sha = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], text=True
    ).strip()
    dirty = subprocess.check_output(
        ["git", "status", "--porcelain"], text=True
    ).strip()
    if dirty:
        raise RuntimeError(f"strict contract requires clean source: {dirty}")

    device = torch.device("cuda:0")
    identity = check_identity_and_first_order(device)
    opened = check_opened_geometry(device)
    opened_full_model = check_opened_full_model(device)
    runtime_model = build(device, content_mode="orthogonal_basis_transport")
    runtime_container = torch.nn.Module()
    runtime_container.reasoner = runtime_model
    runtime = study.strict_fla_runtime_summary(runtime_container, "gdn2")
    result = {
        "status": "passed",
        "git_sha": git_sha,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "gpu_uuid": gpu_uuid,
        "device": torch.cuda.get_device_name(0),
        "fla_source_sha": fla_source_sha,
        "fla_source_root": fla_root,
        "fla_source_module": fla_module,
        "identity_and_first_order": identity,
        "opened_geometry": opened,
        "opened_full_model": opened_full_model,
        "strict_runtime": runtime,
    }
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
