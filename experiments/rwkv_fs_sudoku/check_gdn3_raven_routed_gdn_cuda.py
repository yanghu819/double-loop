#!/usr/bin/env python3
from __future__ import annotations

import json
import math
import os
import subprocess
from typing import Any, Callable

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
import torch.nn.functional as F

import study_rwkv_futureseed_loop as study


EXPECTED_FLA_SOURCE_SHA = "9c8e42e762fce087c27b673af4922795d9edb85e"
LAYERS = 12
HEADS = 8
HEAD_DIM = 32
VALUE_DIM = 32
SLOTS = 8
SLOT_WIDTH = HEAD_DIM // SLOTS
MODEL_DIM = HEADS * HEAD_DIM
TOKENS = 81
EXPECTED_PARAMETER_DELTA = LAYERS * MODEL_DIM * HEADS * SLOTS


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
        f"blocks.{layer}.time_mix.raven_route_proj.weight"
        for layer in range(LAYERS)
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


def capture_official_calls(
    call: Callable[
        [], tuple[torch.Tensor, dict[str, torch.Tensor], list[torch.Tensor]]
    ],
) -> tuple[
    tuple[torch.Tensor, dict[str, torch.Tensor], list[torch.Tensor]],
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
        geometry.append((int(query.shape[1]), int(query.shape[-1])))
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
    torch.manual_seed(17012)
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
                f"layer {layer} incoming-state identity failed: "
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


def check_zero_identity_and_learning_path(device: torch.device) -> dict[str, Any]:
    torch.manual_seed(17009)
    control = build(device, update_mode="none")
    torch.manual_seed(17009)
    candidate = build(device, update_mode="raven_routed_gdn")
    missing = copy_shared_state(control, candidate)
    control.eval()
    candidate.eval()

    x = torch.randn(2, TOKENS, MODEL_DIM, device=device)
    address = torch.randn_like(x)
    cell_order = torch.randperm(TOKENS, device=device)
    with torch.no_grad():
        control_result, control_calls = capture_official_calls(
            lambda: capture_model(control, x, address, cell_order)
        )
        candidate_result, candidate_calls = capture_official_calls(
            lambda: capture_model(candidate, x, address, cell_order)
        )
    control_output, _control_diag, control_states = control_result
    candidate_output, candidate_diag, candidate_states = candidate_result
    expected_calls = [(TOKENS, HEAD_DIM)] * LAYERS
    if control_calls != expected_calls or candidate_calls != expected_calls:
        raise AssertionError(
            f"official call geometry changed: {control_calls}, {candidate_calls}"
        )
    if not torch.equal(control_output, candidate_output):
        error = float(
            (control_output.float() - candidate_output.float()).abs().max().item()
        )
        raise AssertionError(f"zero-init routed output identity failed: {error}")
    state_errors = [
        float((left.float() - right.float()).abs().max().item())
        for left, right in zip(control_states, candidate_states)
    ]
    if len(state_errors) != LAYERS or max(state_errors) != 0.0:
        raise AssertionError(f"zero-init terminal-state mismatch: {state_errors}")

    parameter_delta = sum(p.numel() for p in candidate.parameters()) - sum(
        p.numel() for p in control.parameters()
    )
    if parameter_delta != EXPECTED_PARAMETER_DELTA:
        raise AssertionError(
            f"parameter delta {parameter_delta} != {EXPECTED_PARAMETER_DELTA}"
        )
    incoming_identity = check_nonzero_initial_state_identity(
        control, candidate, device
    )

    candidate.train()
    candidate.zero_grad(set_to_none=True)
    output, diagnostics, states = capture_model(
        candidate,
        x.detach().clone().requires_grad_(True),
        address,
        cell_order,
    )
    target = torch.randn_like(output, dtype=torch.float32)
    (output.float() * target).mean().backward()
    gradients: dict[str, float] = {}
    chunk_counts = []
    for layer, (block, state) in enumerate(zip(candidate.blocks, states)):
        projection = block.time_mix.raven_route_proj
        if projection is None or projection.weight.grad is None:
            raise AssertionError(f"layer {layer} router gradient is missing")
        gradient = projection.weight.grad.float()
        if not bool(torch.isfinite(gradient).all()):
            raise AssertionError(f"layer {layer} router gradient is non-finite")
        grad_max = float(gradient.abs().max().item())
        if grad_max <= 0.0:
            raise AssertionError(f"layer {layer} router gradient is zero")
        gradients[str(layer)] = grad_max
        chunk_counts.append(
            graph_names(state).count("ChunkGDN2FunctionBackward")
        )
    if chunk_counts != [1] * LAYERS:
        raise AssertionError(
            f"single-call official backward graph mismatch: {chunk_counts}"
        )

    official_layers = [type(block.time_mix.core).__name__ for block in candidate.blocks]
    if official_layers != ["GatedDeltaNet2"] * LAYERS:
        raise AssertionError(f"unexpected recurrent layers: {official_layers}")
    expected_zero = {
        "gdn3_raven_routed_allocation_abs_from_one": 0.0,
        "gdn3_raven_routed_allocation_slot_std": 0.0,
        "gdn3_raven_routed_allocation_batch_std": 0.0,
        "gdn3_raven_routed_allocation_token_std": 0.0,
        "gdn3_raven_routed_k_relative_change": 0.0,
        "gdn3_raven_routed_g_relative_change": 0.0,
        "gdn3_raven_routed_router_weight_rms": 0.0,
        "gdn3_raven_routed_allocation_entropy_normalized": 1.0,
        "gdn3_raven_routed_allocation_min": 1.0,
        "gdn3_raven_routed_allocation_max": 1.0,
    }
    for key, expected in expected_zero.items():
        actual = float(diagnostics[key].item())
        if actual != expected:
            raise AssertionError(
                f"zero-init diagnostic mismatch {key}: {actual} != {expected}"
            )
    if float(diagnostics["gdn3_raven_routed_enabled"].item()) != 1.0:
        raise AssertionError("Raven-routed path did not report enabled")
    return {
        "output_exact_identity": True,
        "terminal_state_max_abs_errors": state_errors,
        "incoming_state_identity": incoming_identity,
        "missing_parameters": missing,
        "parameter_delta": parameter_delta,
        "state_value_delta": 0,
        "gradient_max_abs_by_layer": gradients,
        "official_call_geometry": candidate_calls,
        "official_chunk_backward_counts": chunk_counts,
        "official_gdn2_layers": official_layers,
        "position_qk_active": float(diagnostics["gdn2_address_enabled"].item()),
        "zero_init_diag": {
            key: float(value.detach().float().item())
            for key, value in candidate_diag.items()
            if key.startswith("gdn3_raven_routed_")
        },
    }


def check_route_geometry(device: torch.device) -> dict[str, Any]:
    torch.manual_seed(17010)
    model = build(device, update_mode="raven_routed_gdn")
    model.eval()
    layer = model.blocks[0].time_mix
    projection = layer.raven_route_proj
    if projection is None:
        raise AssertionError("Raven route projection is missing")
    generator = torch.Generator(device=device).manual_seed(17013)
    with torch.no_grad():
        projection.weight.copy_(
            0.002
            * torch.randn(
                projection.weight.shape,
                generator=generator,
                device=device,
            )
        )
    x = torch.randn(3, 23, MODEL_DIM, device=device)
    key = torch.randn(3, 23, HEADS, HEAD_DIM, device=device).to(
        torch.bfloat16
    )
    log_decay = (-torch.rand_like(key.float())).to(torch.bfloat16)
    routed_key, routed_decay, diagnostics = layer._raven_routed_gdn_update(
        x, key, log_decay
    )

    weights = projection.weight.float().view(HEADS, SLOTS, MODEL_DIM)
    logits = F.linear(x.float(), weights.reshape(-1, MODEL_DIM)).view(
        3, 23, HEADS, SLOTS
    )
    allocation = float(SLOTS) * logits.softmax(dim=-1)
    row_allocation = torch.repeat_interleave(
        allocation, SLOT_WIDTH, dim=-1
    )
    reference_key = (key.float() * row_allocation.sqrt()).to(key.dtype)
    reference_decay = (
        log_decay.float() * row_allocation
    ).to(log_decay.dtype)
    formula_error = max(
        float((reference_key - routed_key).abs().max().item()),
        float((reference_decay - routed_decay).abs().max().item()),
    )
    if formula_error != 0.0:
        raise AssertionError(f"route formula mismatch: {formula_error}")
    budget_error = float((allocation.mean(dim=-1) - 1.0).abs().max().item())
    if budget_error > 2e-7:
        raise AssertionError(f"allocation budget drift: {budget_error}")
    if float(routed_decay.float().max().item()) > 0.0:
        raise AssertionError("routing made log decay expansive")

    permutation = torch.randperm(SLOTS, device=device)
    permuted_key = key.view(3, 23, HEADS, SLOTS, SLOT_WIDTH).index_select(
        -2, permutation
    ).reshape_as(key)
    permuted_decay = log_decay.view(
        3, 23, HEADS, SLOTS, SLOT_WIDTH
    ).index_select(-2, permutation).reshape_as(log_decay)
    permuted_weights = weights.index_select(1, permutation)
    permuted_logits = F.linear(
        x.float(), permuted_weights.reshape(-1, MODEL_DIM)
    ).view(3, 23, HEADS, SLOTS)
    permuted_allocation = float(SLOTS) * permuted_logits.softmax(dim=-1)
    permuted_rows = torch.repeat_interleave(
        permuted_allocation, SLOT_WIDTH, dim=-1
    )
    permuted_routed_key = (
        permuted_key.float() * permuted_rows.sqrt()
    ).to(key.dtype)
    permuted_routed_decay = (
        permuted_decay.float() * permuted_rows
    ).to(log_decay.dtype)
    expected_key = routed_key.view(
        3, 23, HEADS, SLOTS, SLOT_WIDTH
    ).index_select(-2, permutation).reshape_as(routed_key)
    expected_decay = routed_decay.view(
        3, 23, HEADS, SLOTS, SLOT_WIDTH
    ).index_select(-2, permutation).reshape_as(routed_decay)
    equivariance_error = max(
        float((permuted_routed_key - expected_key).abs().max().item()),
        float((permuted_routed_decay - expected_decay).abs().max().item()),
    )
    if equivariance_error != 0.0:
        raise AssertionError(
            f"slot permutation equivariance failed: {equivariance_error}"
        )
    required_positive = (
        "gdn3_raven_routed_allocation_abs_from_one",
        "gdn3_raven_routed_allocation_slot_std",
        "gdn3_raven_routed_allocation_batch_std",
        "gdn3_raven_routed_allocation_token_std",
        "gdn3_raven_routed_k_relative_change",
        "gdn3_raven_routed_g_relative_change",
        "gdn3_raven_routed_router_weight_rms",
    )
    values = {
        key: float(diagnostics[key].detach().float().item())
        for key in study.RAVEN_ROUTED_GDN_TRAIN_KEYS
    }
    if min(values[key] for key in required_positive) <= 0.0:
        raise AssertionError(f"opened route lacks variation: {values}")
    entropy = values["gdn3_raven_routed_allocation_entropy_normalized"]
    if not 0.0 < entropy < 1.0:
        raise AssertionError(f"opened route entropy is invalid: {entropy}")
    if not 0.0 < values["gdn3_raven_routed_allocation_min"]:
        raise AssertionError("opened route produced a nonpositive allocation")
    if values["gdn3_raven_routed_allocation_max"] >= float(SLOTS):
        raise AssertionError("opened route escaped the soft allocation budget")
    return {
        "allocation_budget_max_abs_error": budget_error,
        "route_formula_max_abs_error": formula_error,
        "slot_permutation_max_abs_error": equivariance_error,
        "diagnostics": values,
    }


def check_open_model_and_protected_slot(device: torch.device) -> dict[str, Any]:
    torch.manual_seed(17011)
    model = build(device, update_mode="raven_routed_gdn")
    model.eval()
    x = torch.randn(2, TOKENS, MODEL_DIM, device=device)
    address = torch.randn_like(x)
    cell_order = torch.randperm(TOKENS, device=device)
    with torch.no_grad():
        zero_output, _zero_diag, zero_states = capture_model(
            model, x, address, cell_order
        )
        generator = torch.Generator(device=device).manual_seed(17014)
        for block in model.blocks:
            projection = block.time_mix.raven_route_proj
            if projection is None:
                raise AssertionError("Raven route projection is missing")
            projection.weight.copy_(
                0.002
                * torch.randn(
                    projection.weight.shape,
                    generator=generator,
                    device=device,
                )
            )
        open_output, open_diag, open_states = capture_model(
            model, x, address, cell_order
        )
    output_change = float(
        (open_output.float() - zero_output.float()).abs().max().item()
    )
    state_changes = [
        float((opened.float() - zero.float()).abs().max().item())
        for opened, zero in zip(open_states, zero_states)
    ]
    if output_change <= 0.0 or min(state_changes) <= 0.0:
        raise AssertionError(
            f"opened route did not change output/state: {output_change}, {state_changes}"
        )
    terminal_rms = float(
        open_diag["gdn3_raven_routed_terminal_rms"].item()
    )
    zero_terminal_rms = torch.stack(
        [state.float().square().mean().sqrt() for state in zero_states]
    ).mean()
    if not math.isfinite(terminal_rms):
        raise AssertionError("opened terminal RMS is non-finite")
    if terminal_rms > 4.0 * float(zero_terminal_rms.item()):
        raise AssertionError(
            f"opened terminal RMS exceeded 4x parent: {terminal_rms}"
        )

    if study.chunk_gdn2 is None:
        raise RuntimeError("official chunk_gdn2 is unavailable")
    torch.manual_seed(17015)
    batch, tokens, heads, key_dim, value_dim = 1, 17, 2, 8, 8
    q = F.normalize(
        torch.randn(batch, tokens, heads, key_dim, device=device), dim=-1
    ).to(torch.bfloat16)
    k = F.normalize(
        torch.randn(batch, tokens, heads, key_dim, device=device), dim=-1
    ).to(torch.bfloat16)
    v = torch.randn(
        batch, tokens, heads, value_dim, device=device
    ).to(torch.bfloat16)
    g = (-0.1 * torch.rand_like(k.float())).to(torch.bfloat16)
    b = torch.sigmoid(torch.randn_like(k.float())).to(torch.bfloat16)
    w = torch.sigmoid(torch.randn_like(v.float())).to(torch.bfloat16)
    initial_state = torch.randn(
        batch, heads, key_dim, value_dim, device=device, dtype=torch.float32
    )
    protected_width = 2
    k[:, :, :, :protected_width] = 0
    g[:, :, :, :protected_width] = 0
    _output, terminal = study.chunk_gdn2(
        q=q,
        k=k,
        v=v,
        g=g,
        b=b,
        w=w,
        scale=1.0,
        initial_state=initial_state,
        output_final_state=True,
        use_qk_l2norm_in_kernel=False,
    )
    protected_error = float(
        (
            terminal[:, :, :protected_width].float()
            - initial_state[:, :, :protected_width]
        ).abs().max().item()
    )
    active_change = float(
        (
            terminal[:, :, protected_width:].float()
            - initial_state[:, :, protected_width:]
        ).abs().max().item()
    )
    if protected_error > 2e-6 or active_change <= 0.0:
        raise AssertionError(
            "zero-allocation state protection failed: "
            f"protected={protected_error} active={active_change}"
        )
    return {
        "output_max_abs_change": output_change,
        "terminal_state_max_abs_changes": state_changes,
        "terminal_rms": terminal_rms,
        "parent_terminal_rms": float(zero_terminal_rms.item()),
        "protected_slot_max_abs_error": protected_error,
        "active_slot_max_abs_change": active_change,
        "opened_diag": {
            key: float(value.detach().float().item())
            for key, value in open_diag.items()
            if key.startswith("gdn3_raven_routed_")
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
    source_sha, source_root, source_module = study.resolve_strict_fla_source(
        "gdn2"
    )
    if source_sha != EXPECTED_FLA_SOURCE_SHA:
        raise RuntimeError(
            f"pinned FLA SHA drift: {source_sha} != {EXPECTED_FLA_SOURCE_SHA}"
        )
    device = torch.device("cuda:0")
    properties = torch.cuda.get_device_properties(device)
    output = {
        "device": str(device),
        "gpu_name": properties.name,
        "gpu_uuid": EXPECTED_GPU_UUID,
        "fla_source_sha": source_sha,
        "fla_source_root": source_root,
        "fla_source_module": source_module,
        "zero_identity_and_learning_path": (
            check_zero_identity_and_learning_path(device)
        ),
        "route_geometry": check_route_geometry(device),
        "open_model_and_protected_slot": (
            check_open_model_and_protected_slot(device)
        ),
    }
    compute_apps = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-compute-apps=gpu_uuid,pid",
            "--format=csv,noheader",
        ],
        text=True,
    ).strip().splitlines()
    if len(compute_apps) != 1 or not compute_apps[0].startswith(
        EXPECTED_GPU_UUID
    ):
        raise RuntimeError(
            f"unexpected concurrent compute applications: {compute_apps}"
        )
    output["compute_apps"] = compute_apps
    print(json.dumps(output, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
