#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
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


COORDINATE_MODULES = (
    "q_proj",
    "k_proj",
    "v_proj",
    "q_conv1d",
    "k_conv1d",
    "v_conv1d",
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
        queue.extend(next_fn for next_fn, _ in fn.next_functions if next_fn is not None)
    return names


def build(
    device: torch.device,
    *,
    d_model: int,
    layers: int,
    heads: int,
    head_dim: int,
    cross_layer_init: str = "coherent_qkv",
) -> FutureSeedRWKV:
    return FutureSeedRWKV(
        d_model,
        layers,
        heads,
        head_dim,
        4,
        future_seed_scale=1.0,
        future_seed_decay=0.0,
        future_seed_update="fixed",
        future_seed_norm_mode="unit",
        future_seed_gate_mode="head",
        future_seed_scope="layer",
        backbone="gdn2",
        gdn_mode="chunk",
        gdn_expand_v=1.0,
        gdn_use_short_conv=True,
        gdn_conv_size=4,
        gdn_allow_neg_eigval=False,
        gdn2_address_mode="none",
        gdn2_cross_layer_init=cross_layer_init,
    ).to(device)


def module_max_error(left: torch.nn.Module, right: torch.nn.Module) -> float:
    errors = []
    left_state = left.state_dict()
    right_state = right.state_dict()
    if left_state.keys() != right_state.keys():
        raise AssertionError("coordinate modules have different state schemas")
    for name, left_value in left_state.items():
        errors.append(
            float((left_value.float() - right_state[name].float()).abs().max().item())
        )
    return max(errors, default=0.0)


def coordinate_snapshot(model: FutureSeedRWKV) -> dict[str, torch.Tensor]:
    snapshot: dict[str, torch.Tensor] = {}
    for layer_idx, block in enumerate(model.blocks):
        core = block.time_mix.core
        for module_name in COORDINATE_MODULES:
            module = getattr(core, module_name, None)
            if module is None:
                continue
            for state_name, value in module.state_dict().items():
                snapshot[f"layer{layer_idx}.{module_name}.{state_name}"] = value.detach().clone()
    return snapshot


def check_initial_contract(device: torch.device) -> dict[str, Any]:
    torch.manual_seed(9303)
    model = build(device, d_model=64, layers=3, heads=4, head_dim=16)
    control = build(
        device,
        d_model=64,
        layers=3,
        heads=4,
        head_dim=16,
        cross_layer_init="independent",
    )
    candidate_parameter_count = sum(parameter.numel() for parameter in model.parameters())
    control_parameter_count = sum(parameter.numel() for parameter in control.parameters())
    if candidate_parameter_count != control_parameter_count:
        raise AssertionError(
            "coherent initialization changed parameter count: "
            f"{candidate_parameter_count} != {control_parameter_count}"
        )
    if model.state_dict().keys() != control.state_dict().keys():
        raise AssertionError("coherent initialization changed the state-dict schema")
    source_core = model.blocks[0].time_mix.core
    max_error = 0.0
    unique_parameter_ptrs: dict[str, int] = {}
    for layer_idx, block in enumerate(model.blocks):
        core = block.time_mix.core
        for module_name in COORDINATE_MODULES:
            source_module = getattr(source_core, module_name, None)
            target_module = getattr(core, module_name, None)
            if source_module is None or target_module is None:
                if source_module is not target_module:
                    raise AssertionError(f"coordinate module mismatch: {module_name}")
                continue
            max_error = max(max_error, module_max_error(source_module, target_module))
            for parameter_name, parameter in target_module.named_parameters():
                key = f"{module_name}.{parameter_name}"
                pointer = parameter.data_ptr()
                if layer_idx > 0 and unique_parameter_ptrs.get(key) == pointer:
                    raise AssertionError(f"coordinate parameters were tied: {key}")
                if layer_idx == 0:
                    unique_parameter_ptrs[key] = pointer
    if max_error != 0.0:
        raise AssertionError(f"coherent initialization was not exact: {max_error}")

    private_error = module_max_error(
        model.blocks[0].time_mix.core.o_proj,
        model.blocks[1].time_mix.core.o_proj,
    )
    if private_error == 0.0:
        raise AssertionError("non-coordinate output projections were accidentally cloned")
    return {
        "coordinate_init_max_abs_error": max_error,
        "private_o_projection_max_abs_difference": private_error,
        "parameter_count": candidate_parameter_count,
        "parameter_delta_vs_independent": candidate_parameter_count - control_parameter_count,
    }


def check_specialization_after_step(device: torch.device) -> dict[str, Any]:
    torch.manual_seed(9304)
    model = build(device, d_model=64, layers=3, heads=4, head_dim=16)
    model.train()
    before = coordinate_snapshot(model)
    x = torch.randn(2, 81, 64, device=device, requires_grad=True)
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
        hooks.append(block.time_mix.register_forward_hook(capture_terminal, with_kwargs=True))
    try:
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            output, diagnostics, _ = model(x)
            loss = output.float().square().mean()
    finally:
        for hook in hooks:
            hook.remove()
    if len(terminal_states) != len(model.blocks):
        raise AssertionError("did not capture every GDN2 terminal state")
    graph = graph_names(terminal_states[-1])
    if "ChunkGDN2FunctionBackward" not in graph:
        raise AssertionError(f"candidate bypassed official GDN2 chunk kernel: {graph}")
    loss.backward()
    if x.grad is None or not bool(torch.isfinite(x.grad).all()):
        raise AssertionError("input gradient is missing or non-finite")
    for layer_idx, state in enumerate(terminal_states):
        if not bool(torch.isfinite(state).all()):
            raise AssertionError(f"terminal state {layer_idx} is non-finite")
    coordinate_gradients = []
    for block in model.blocks:
        gradient = block.time_mix.core.q_proj.weight.grad
        if gradient is None or not bool(torch.isfinite(gradient).all()):
            raise AssertionError("Q projection gradient is missing or non-finite")
        coordinate_gradients.append(float(gradient.float().abs().max().item()))
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
    optimizer.step()

    source_core = model.blocks[0].time_mix.core
    post_step_divergence = 0.0
    for block in model.blocks[1:]:
        target_core = block.time_mix.core
        for module_name in COORDINATE_MODULES:
            source_module = getattr(source_core, module_name, None)
            target_module = getattr(target_core, module_name, None)
            if source_module is not None and target_module is not None:
                post_step_divergence = max(
                    post_step_divergence,
                    module_max_error(source_module, target_module),
                )
    if post_step_divergence <= 0.0:
        raise AssertionError("coordinate parameters did not specialize after one optimizer step")
    parameter_change = max(
        float((value.float() - before[name].float()).abs().max().item())
        for name, value in coordinate_snapshot(model).items()
    )
    if parameter_change <= 0.0:
        raise AssertionError("coordinate parameters did not update")
    return {
        "loss": float(loss.detach().cpu()),
        "future_seed_gate_mean": float(diagnostics["fs_gate_mean"].detach().cpu()),
        "q_gradient_max_abs_by_layer": coordinate_gradients,
        "post_step_coordinate_divergence": post_step_divergence,
        "coordinate_parameter_change": parameter_change,
        "autograd_graph": graph,
    }


def check_full_size_fit(device: torch.device) -> dict[str, Any]:
    torch.manual_seed(9305)
    torch.cuda.reset_peak_memory_stats(device)
    model = build(device, d_model=256, layers=12, heads=8, head_dim=32)
    model.train()
    x = torch.randn(1, 81, 256, device=device, requires_grad=True)
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        output, diagnostics, _ = model(x)
        loss = output.float().square().mean()
    loss.backward()
    if not bool(torch.isfinite(loss)):
        raise AssertionError("full-size loss is non-finite")
    if x.grad is None or not bool(torch.isfinite(x.grad).all()):
        raise AssertionError("full-size input gradient is missing or non-finite")
    return {
        "loss": float(loss.detach().cpu()),
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "peak_allocated_mib": torch.cuda.max_memory_allocated(device) / (1024**2),
        "peak_reserved_mib": torch.cuda.max_memory_reserved(device) / (1024**2),
        "future_seed_gate_mean": float(diagnostics["fs_gate_mean"].detach().cpu()),
    }


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required; CPU model smoke is forbidden")
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "0":
        raise RuntimeError("CUDA_VISIBLE_DEVICES=0 is required for GPU1-only validation")
    marker = Path("/huyang2/double-loop/.cache/fla-source-sha")
    marker_sha = marker.read_text(encoding="utf-8").strip()
    if marker_sha != GAIN_BUDGET_FLA_SHA:
        raise RuntimeError(f"strict FLA marker mismatch: {marker_sha} != {GAIN_BUDGET_FLA_SHA}")
    device = torch.device("cuda:0")
    visible_uuid = subprocess.check_output(
        [
            "nvidia-smi",
            "-i",
            "0",
            "--query-gpu=uuid",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    ).strip()
    expected_uuid = os.environ.get("GPU1_UUID", "")
    if not expected_uuid or visible_uuid != expected_uuid:
        raise RuntimeError(f"GPU1 UUID mismatch: visible={visible_uuid} expected={expected_uuid}")
    result = {
        "cuda_device": torch.cuda.get_device_name(device),
        "cuda_uuid": visible_uuid,
        "fla_source_sha": marker_sha,
        "initial_contract": check_initial_contract(device),
        "specialization_after_step": check_specialization_after_step(device),
        "full_size_fit": check_full_size_fit(device),
    }
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
