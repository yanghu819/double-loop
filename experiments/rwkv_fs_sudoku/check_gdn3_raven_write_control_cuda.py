#!/usr/bin/env python3
from __future__ import annotations

import inspect
import json
import os
import subprocess
from pathlib import Path
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
if os.environ.get("FLA_DISABLE_BACKEND_DISPATCH") != "1":
    raise RuntimeError("FLA_DISABLE_BACKEND_DISPATCH=1 is required")
if os.environ.get("FLA_CONV_BACKEND") != "triton":
    raise RuntimeError("FLA_CONV_BACKEND=triton is required")

import torch

import study_rwkv_futureseed_loop as study


EXPECTED_FLA_SOURCE_SHA = "9c8e42e762fce087c27b673af4922795d9edb85e"
LAYERS = 12
MODEL_DIM = 256
MAIN_HEADS = 8
MAIN_HEAD_DIM = 32
CONTROL_WIDTH = 64
CONTROL_HEADS = 4
CONTROL_HEAD_DIM = 16
CONTROL_SLOTS = 8
CONTROL_TOPK = 1
TOKENS = 81
EXPECTED_RAVEN_PARAMETERS_PER_LAYER = 18_792
EXPECTED_PARAMETER_DELTA_PER_LAYER = 51_564
EXPECTED_PARAMETER_DELTA = LAYERS * EXPECTED_PARAMETER_DELTA_PER_LAYER
EXPECTED_STATE_VALUES_PER_LAYER = (
    CONTROL_HEADS * (2 * CONTROL_HEAD_DIM) * CONTROL_SLOTS
)
EXPECTED_STATE_VALUES = LAYERS * EXPECTED_STATE_VALUES_PER_LAYER


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def finite_nonzero_grad(parameter: torch.nn.Parameter, name: str) -> float:
    if parameter.grad is None:
        raise AssertionError(f"{name} gradient is missing")
    gradient = parameter.grad.float()
    require(bool(torch.isfinite(gradient).all()), f"{name} gradient is non-finite")
    maximum = float(gradient.abs().max().item())
    require(maximum > 0.0, f"{name} gradient is zero")
    return maximum


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


def build(device: torch.device, *, state_expert_mode: str) -> study.FutureSeedRWKV:
    return study.FutureSeedRWKV(
        MODEL_DIM,
        LAYERS,
        MAIN_HEADS,
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
    control: study.FutureSeedRWKV,
    candidate: study.FutureSeedRWKV,
) -> list[str]:
    missing, unexpected = candidate.load_state_dict(
        control.state_dict(),
        strict=False,
    )
    expected = {
        name for name in candidate.state_dict() if ".state_expert." in name
    }
    require(set(missing) == expected, f"unexpected missing keys: {missing}")
    require(not unexpected, f"unexpected candidate keys: {unexpected}")
    for layer in range(LAYERS):
        prefix = f"blocks.{layer}.state_expert."
        require(
            any(name.startswith(prefix) for name in expected),
            f"layer {layer} has no Raven controller migration keys",
        )
    return sorted(missing)


def capture(
    model: study.FutureSeedRWKV,
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
    raven_states: list[torch.Tensor] = []
    hooks = []

    def capture_main(
        _module: torch.nn.Module,
        _args: tuple[torch.Tensor, ...],
        _kwargs: dict[str, Any],
        output: tuple[torch.Tensor, torch.Tensor],
    ) -> None:
        main_states.append(output[1])

    def capture_raven(
        _module: torch.nn.Module,
        _args: tuple[torch.Tensor, ...],
        _kwargs: dict[str, Any],
        output: tuple[torch.Tensor, torch.Tensor],
    ) -> None:
        raven_states.append(output[1])

    for block in model.blocks:
        hooks.append(
            block.time_mix.register_forward_hook(capture_main, with_kwargs=True)
        )
        if block.state_expert is not None:
            hooks.append(
                block.state_expert.time_mix.register_forward_hook(
                    capture_raven,
                    with_kwargs=True,
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
    return output, diagnostics, main_states, raven_states


def check_incoming_state_identity(
    control: study.FutureSeedRWKV,
    candidate: study.FutureSeedRWKV,
    device: torch.device,
) -> list[dict[str, float]]:
    rows: list[dict[str, float]] = []
    torch.manual_seed(18102)
    for layer, (control_block, candidate_block) in enumerate(
        zip(control.blocks, candidate.blocks)
    ):
        x = torch.randn(1, TOKENS, MODEL_DIM, device=device)
        address = torch.randn_like(x)
        order = torch.randperm(TOKENS, device=device)
        main_state = torch.randn(
            1,
            MAIN_HEADS,
            MAIN_HEAD_DIM,
            MAIN_HEAD_DIM,
            device=device,
            dtype=torch.float32,
        )
        raven_state = torch.randn(
            1,
            CONTROL_HEADS,
            2 * CONTROL_HEAD_DIM,
            CONTROL_SLOTS,
            device=device,
            dtype=torch.float32,
        )
        with torch.no_grad(), torch.autocast(
            device_type="cuda",
            dtype=torch.bfloat16,
        ):
            control_output, control_terminal = control_block(
                x,
                initial_state=main_state,
                address=address,
                cell_order=order,
            )
            candidate_output, candidate_terminal = candidate_block(
                x,
                initial_state=main_state,
                state_expert_initial_state=raven_state,
                address=address,
                cell_order=order,
            )
        output_error = float(
            (candidate_output.float() - control_output.float()).abs().max().item()
        )
        state_error = float(
            (candidate_terminal.float() - control_terminal.float()).abs().max().item()
        )
        require(
            output_error == 0.0 and state_error == 0.0,
            f"layer {layer} nonzero-state identity failed: "
            f"output={output_error} state={state_error}",
        )
        rows.append(
            {
                "layer": float(layer),
                "output_max_abs_error": output_error,
                "main_terminal_state_max_abs_error": state_error,
            }
        )
    return rows


def check_state_dependency(
    candidate: study.FutureSeedRWKV,
    device: torch.device,
) -> dict[str, float]:
    block = candidate.blocks[1]
    controller = block.state_expert
    require(
        isinstance(controller, study.RavenWriteController),
        "layer 1 Raven controller is missing",
    )
    controller.eval()
    torch.manual_seed(18103)
    x = torch.randn(2, TOKENS, MODEL_DIM, device=device)
    state = torch.randn(
        2,
        CONTROL_HEADS,
        2 * CONTROL_HEAD_DIM,
        CONTROL_SLOTS,
        device=device,
    )
    shifted = state.roll(1, dims=-1)
    state_rms = state.float().square().mean(
        dim=(-1, -2),
        keepdim=True,
    ).sqrt().clamp_min(1e-6)
    shifted_rms = shifted.float().square().mean(
        dim=(-1, -2),
        keepdim=True,
    ).sqrt().clamp_min(1e-6)
    gate = torch.sigmoid(controller.future_seed_logit.float())
    content = controller.content_down(x)
    with torch.no_grad(), torch.autocast(
        device_type="cuda",
        dtype=torch.bfloat16,
    ):
        raw_a, terminal_a = controller.time_mix(
            content,
            initial_state=state.float() / state_rms * gate,
        )
        raw_b, terminal_b = controller.time_mix(
            content,
            initial_state=shifted.float() / shifted_rms * gate,
        )
    controller_output_change = float(
        (raw_a.float() - raw_b.float()).square().mean().sqrt().item()
    )
    terminal_change = float(
        (terminal_a.float() - terminal_b.float()).square().mean().sqrt().item()
    )
    require(
        controller_output_change > 0.0,
        "Raven controller ignores incoming packed state",
    )
    require(terminal_change > 0.0, "Raven terminal state ignores incoming packed state")

    saved_adapter = controller.value_adapter.weight.detach().clone()
    main_state = torch.randn(
        2,
        MAIN_HEADS,
        MAIN_HEAD_DIM,
        MAIN_HEAD_DIM,
        device=device,
        dtype=torch.float32,
    )
    address = torch.randn_like(x)
    order = torch.randperm(TOKENS, device=device)
    try:
        with torch.no_grad():
            controller.value_adapter.weight.zero_()
            controller.value_adapter.weight[:CONTROL_WIDTH].copy_(
                0.01
                * torch.eye(
                    CONTROL_WIDTH,
                    device=device,
                    dtype=controller.value_adapter.weight.dtype,
                )
            )
        with torch.no_grad(), torch.autocast(
            device_type="cuda",
            dtype=torch.bfloat16,
        ):
            main_output_a, main_terminal_a = block(
                x,
                initial_state=main_state,
                state_expert_initial_state=state,
                address=address,
                cell_order=order,
            )
            main_output_b, main_terminal_b = block(
                x,
                initial_state=main_state,
                state_expert_initial_state=shifted,
                address=address,
                cell_order=order,
            )
    finally:
        with torch.no_grad():
            controller.value_adapter.weight.copy_(saved_adapter)
    main_output_change = float(
        (main_output_a.float() - main_output_b.float()).square().mean().sqrt().item()
    )
    main_terminal_change = float(
        (main_terminal_a.float() - main_terminal_b.float()).square().mean().sqrt().item()
    )
    require(main_output_change > 0.0, "Raven state does not reach the main output")
    require(main_terminal_change > 0.0, "Raven state does not reach the main transition")

    unpacked = controller.time_mix.unpack_recurrent_state(terminal_a)
    roundtrip = controller.time_mix.pack_recurrent_state(unpacked)
    require(torch.equal(roundtrip, terminal_a), "Raven state pack roundtrip is not exact")
    return {
        "controller_output_rms_change": controller_output_change,
        "controller_terminal_rms_change": terminal_change,
        "opened_adapter_main_output_rms_change": main_output_change,
        "opened_adapter_main_terminal_rms_change": main_terminal_change,
        "pack_roundtrip_max_abs_error": 0.0,
    }


def check_two_stage_learning(
    candidate: study.FutureSeedRWKV,
    device: torch.device,
) -> dict[str, Any]:
    for parameter in candidate.parameters():
        parameter.requires_grad_(False)
    controller_parameters = [
        parameter
        for name, parameter in candidate.named_parameters()
        if ".state_expert." in name
    ]
    for parameter in controller_parameters:
        parameter.requires_grad_(True)

    torch.manual_seed(18104)
    x = torch.randn(2, TOKENS, MODEL_DIM, device=device, requires_grad=True)
    address = torch.randn_like(x, requires_grad=True)
    order = torch.randperm(TOKENS, device=device)
    optimizer = torch.optim.SGD(controller_parameters, lr=0.1)
    candidate.train()

    optimizer.zero_grad(set_to_none=True)
    output, _diag, _main_states, _raven_states = capture(
        candidate,
        x,
        address,
        order,
    )
    target = torch.randn_like(output, dtype=torch.float32)
    (output.float() * target).mean().backward()
    stage1_adapter_gradients: dict[str, float] = {}
    for layer, block in enumerate(candidate.blocks):
        controller = block.state_expert
        require(
            isinstance(controller, study.RavenWriteController),
            f"layer {layer} controller is missing",
        )
        stage1_adapter_gradients[str(layer)] = finite_nonzero_grad(
            controller.value_adapter.weight,
            f"layer {layer} value_adapter.weight",
        )
    optimizer.step()

    optimizer.zero_grad(set_to_none=True)
    x.grad = None
    address.grad = None
    output, diagnostics, main_states, raven_states = capture(
        candidate,
        x,
        address,
        order,
    )
    (output.float() * target).mean().backward()
    stage2_gradients: dict[str, dict[str, float]] = {}
    for layer, block in enumerate(candidate.blocks):
        controller = block.state_expert
        require(isinstance(controller, study.RavenWriteController), "controller missing")
        core = controller.time_mix.core
        row = {
            "content_down": finite_nonzero_grad(
                controller.content_down.weight,
                f"layer {layer} content_down.weight",
            ),
            "value_adapter": finite_nonzero_grad(
                controller.value_adapter.weight,
                f"layer {layer} value_adapter.weight",
            ),
            "q_proj": finite_nonzero_grad(core.q_proj.weight, f"layer {layer} q_proj"),
            "k_proj": finite_nonzero_grad(core.k_proj.weight, f"layer {layer} k_proj"),
            "v_proj": finite_nonzero_grad(core.v_proj.weight, f"layer {layer} v_proj"),
            "a_proj": finite_nonzero_grad(core.a_proj.weight, f"layer {layer} a_proj"),
            "r_proj": finite_nonzero_grad(core.r_proj.weight, f"layer {layer} r_proj"),
        }
        if layer > 0:
            row["future_seed_logit"] = finite_nonzero_grad(
                controller.future_seed_logit,
                f"layer {layer} future_seed_logit",
            )
        stage2_gradients[str(layer)] = row

    require(len(main_states) == LAYERS, "main state capture count drifted")
    require(len(raven_states) == LAYERS, "Raven state capture count drifted")
    main_graphs = [graph_names(state) for state in main_states]
    raven_graphs = [graph_names(state) for state in raven_states]
    main_backward = [
        "ChunkGDN2FunctionBackward" in names for names in main_graphs
    ]
    raven_backward = [
        any("GSA" in name and name.endswith("Backward") for name in names)
        for names in raven_graphs
    ]
    require(all(main_backward), f"official GDN2 backward missing: {main_backward}")
    require(all(raven_backward), f"official Raven GSA backward missing: {raven_backward}")

    diag = {
        key: float(diagnostics[key].detach().float().item())
        for key in RAVEN_DIAGNOSTIC_KEYS
    }
    require(
        diag["gdn3_raven_write_control_v_residual_relative_rms_min"] > 0.0,
        "Raven write residual did not activate after the adapter step",
    )
    require(
        diag["gdn3_raven_write_control_seed_rms_receiving_min"] > 0.0,
        "adjacent-layer Raven state path did not activate",
    )
    require(
        diag["gdn3_raven_write_control_incoming_path_count_sum"] == LAYERS - 1,
        "Raven incoming-state path count drifted",
    )
    require(
        0.0 <= diag["gdn3_raven_write_control_slot_entropy_normalized_min"] <= 1.0,
        "Raven slot entropy is out of range",
    )
    require(
        0.0 < diag["gdn3_raven_write_control_slot_max_mass_share_max"] <= 1.0,
        "Raven maximum slot mass share is out of range",
    )
    return {
        "stage1_adapter_gradient_max_abs_by_layer": stage1_adapter_gradients,
        "stage2_inner_gradient_max_abs_by_layer": stage2_gradients,
        "main_official_chunk_backward_by_layer": main_backward,
        "raven_official_gsa_backward_by_layer": raven_backward,
        "active_diagnostics": diag,
    }


RAVEN_DIAGNOSTIC_KEYS = (
    "gdn3_raven_write_control_enabled",
    "gdn3_raven_write_control_v_residual_relative_rms",
    "gdn3_raven_write_control_v_residual_relative_rms_min",
    "gdn3_raven_write_control_v_residual_relative_rms_max",
    "gdn3_raven_write_control_terminal_rms",
    "gdn3_raven_write_control_seed_rms",
    "gdn3_raven_write_control_seed_rms_receiving_min",
    "gdn3_raven_write_control_incoming_path_count_sum",
    "gdn3_raven_write_control_slot_entropy_normalized_min",
    "gdn3_raven_write_control_slot_max_mass_share_max",
    "gdn3_raven_write_control_main_terminal_rms_max",
    "gdn3_raven_write_control_adapter_weight_rms",
)


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required; CPU model smoke is forbidden")
    require(torch.cuda.device_count() == 1, "exactly one CUDA device must be visible")
    visible = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader"],
        text=True,
    ).strip().splitlines()
    require(visible == [f"0, {EXPECTED_GPU_UUID}"], f"GPU contract drifted: {visible}")
    source_sha, source_root, source_module = study.resolve_strict_fla_source("gdn2")
    require(source_sha == EXPECTED_FLA_SOURCE_SHA, f"FLA SHA drifted: {source_sha}")
    require(study.GAIN_BUDGET_FLA_SHA == EXPECTED_FLA_SOURCE_SHA, "source constant drifted")
    require(study.GatedDeltaNet2 is not None, "official GatedDeltaNet2 import failed")
    require(study.FLARaven is not None, "official Raven import failed")
    raven_source = Path(inspect.getfile(study.FLARaven)).resolve()
    require(
        Path(source_root).resolve() in raven_source.parents,
        f"Raven source is outside pinned FLA root: {raven_source}",
    )

    device = torch.device("cuda:0")
    torch.manual_seed(18101)
    torch.cuda.manual_seed_all(18101)
    control = build(device, state_expert_mode="none")
    candidate = build(device, state_expert_mode="raven_write_control")
    missing = copy_shared_state(control, candidate)
    control.eval()
    candidate.eval()

    main_classes = [type(block.time_mix.core) for block in candidate.blocks]
    raven_classes = [
        type(block.state_expert.time_mix.core)
        for block in candidate.blocks
        if block.state_expert is not None
    ]
    require(
        main_classes == [study.GatedDeltaNet2] * LAYERS,
        f"main recurrent classes drifted: {main_classes}",
    )
    require(
        raven_classes == [study.FLARaven] * LAYERS,
        f"Raven controller classes drifted: {raven_classes}",
    )
    for layer, block in enumerate(candidate.blocks):
        controller = block.state_expert
        require(isinstance(controller, study.RavenWriteController), "controller missing")
        core = controller.time_mix.core
        require(core.mode == "chunk", f"layer {layer} Raven mode drifted")
        require(core.num_slots == CONTROL_SLOTS, f"layer {layer} slot count drifted")
        require(core.topk == CONTROL_TOPK, f"layer {layer} top-k drifted")
        require(core.add_gumbel_noise, f"layer {layer} Gumbel routing is disabled")
        core_parameters = sum(parameter.numel() for parameter in core.parameters())
        require(
            core_parameters == EXPECTED_RAVEN_PARAMETERS_PER_LAYER,
            f"layer {layer} Raven parameters {core_parameters} drifted",
        )

    parameter_delta = sum(p.numel() for p in candidate.parameters()) - sum(
        p.numel() for p in control.parameters()
    )
    require(
        parameter_delta == EXPECTED_PARAMETER_DELTA,
        f"parameter delta {parameter_delta} != {EXPECTED_PARAMETER_DELTA}",
    )
    inserted_parameter_count = sum(
        parameter.numel()
        for name, parameter in candidate.named_parameters()
        if ".state_expert." in name
    )
    require(
        inserted_parameter_count == EXPECTED_PARAMETER_DELTA,
        "inserted controller parameter count does not match total delta",
    )

    x = torch.randn(2, TOKENS, MODEL_DIM, device=device)
    address = torch.randn_like(x)
    order = torch.randperm(TOKENS, device=device)
    with torch.no_grad():
        control_output, _control_diag, control_states, _ = capture(
            control,
            x,
            address,
            order,
        )
        candidate_output, candidate_diag, candidate_states, raven_states = capture(
            candidate,
            x,
            address,
            order,
        )
    require(torch.equal(candidate_output, control_output), "zero-init output identity failed")
    state_errors = [
        float((right.float() - left.float()).abs().max().item())
        for left, right in zip(control_states, candidate_states)
    ]
    require(len(state_errors) == LAYERS and max(state_errors) == 0.0, f"main state identity failed: {state_errors}")
    require(len(raven_states) == LAYERS, "Raven terminal state count drifted")
    require(
        all(
            tuple(state.shape)
            == (2, CONTROL_HEADS, 2 * CONTROL_HEAD_DIM, CONTROL_SLOTS)
            for state in raven_states
        ),
        f"Raven packed state shapes drifted: {[tuple(state.shape) for state in raven_states]}",
    )
    require(
        float(candidate_diag["gdn3_raven_write_control_enabled"].item()) == 1.0,
        "Raven write-control path is not marked active",
    )
    require(
        float(candidate_diag["gdn3_raven_write_control_v_residual_relative_rms"].item()) == 0.0,
        "zero-init Raven V residual is not exact zero",
    )

    result = {
        "status": "pass",
        "gpu": {
            "visible": visible,
            "name": torch.cuda.get_device_properties(device).name,
            "uuid": EXPECTED_GPU_UUID,
        },
        "fla": {
            "source_sha": source_sha,
            "source_root": source_root,
            "source_module": source_module,
            "raven_source": str(raven_source),
            "main_class": f"{study.GatedDeltaNet2.__module__}.{study.GatedDeltaNet2.__qualname__}",
            "raven_class": f"{study.FLARaven.__module__}.{study.FLARaven.__qualname__}",
        },
        "migration": {
            "missing_state_keys": missing,
            "parameter_delta_per_layer": EXPECTED_PARAMETER_DELTA_PER_LAYER,
            "parameter_delta": parameter_delta,
            "controller_state_values_per_layer": EXPECTED_STATE_VALUES_PER_LAYER,
            "controller_state_values": EXPECTED_STATE_VALUES,
        },
        "identity": {
            "output_exact": True,
            "main_terminal_state_max_abs_errors": state_errors,
            "nonzero_incoming_state": check_incoming_state_identity(
                control,
                candidate,
                device,
            ),
        },
        "state_dependency": check_state_dependency(candidate, device),
        "learning_path": check_two_stage_learning(candidate, device),
        "cpu_fallback": False,
    }
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
