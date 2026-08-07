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
EXPECTED_PARAMETER_DELTA = LAYERS * VALUE_DIM * HEAD_DIM


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
        f"blocks.{layer}.time_mix.adaptive_signed_erase_proj.weight"
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
    torch.manual_seed(12012)
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
                f"layer {layer} zero-init incoming-state identity failed: "
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
    torch.manual_seed(12009)
    control = build(device, update_mode="none")
    torch.manual_seed(12009)
    candidate = build(device, update_mode="adaptive_signed_erase")
    missing = copy_shared_state(control, candidate)
    control.eval()
    candidate.eval()

    x = torch.randn(2, TOKENS, MODEL_DIM, device=device)
    address = torch.randn_like(x)
    cell_order = torch.randperm(TOKENS, device=device)
    with torch.no_grad():
        (control_result, control_calls) = capture_official_call_lengths(
            lambda: capture_terminal_states(control, x, address, cell_order)
        )
        (candidate_result, candidate_calls) = capture_official_call_lengths(
            lambda: capture_terminal_states(candidate, x, address, cell_order)
        )
    control_output, _control_diag, control_states = control_result
    candidate_output, candidate_diag, candidate_states = candidate_result
    if control_calls != [TOKENS] * LAYERS or candidate_calls != [TOKENS] * LAYERS:
        raise AssertionError(
            f"official call lengths changed: {control_calls}, {candidate_calls}"
        )
    if not torch.equal(control_output, candidate_output):
        raise AssertionError("zero-init signed erase changed the model output")
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
    output, diagnostics, states = capture_terminal_states(
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
        projection = block.time_mix.adaptive_signed_erase_proj
        if projection is None or projection.weight.grad is None:
            raise AssertionError(f"layer {layer} signed-erase gradient is missing")
        gradient = projection.weight.grad
        if not bool(torch.isfinite(gradient).all()):
            raise AssertionError(f"layer {layer} signed-erase gradient is non-finite")
        grad_max = float(gradient.float().abs().max().item())
        if grad_max <= 0.0:
            raise AssertionError(f"layer {layer} signed-erase gradient is zero")
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
    if any(block.time_mix.core.allow_neg_eigval for block in candidate.blocks):
        raise AssertionError("parent allow_neg_eigval contract changed")
    if float(diagnostics["gdn2_address_enabled"].item()) != 1.0:
        raise AssertionError("position-QK address path is not active")
    if float(diagnostics["gdn3_adaptive_signed_erase_enabled"].item()) != 1.0:
        raise AssertionError("signed-erase diagnostics did not mark the path active")
    if float(diagnostics["gdn3_adaptive_signed_erase_residual_abs"].item()) != 0.0:
        raise AssertionError("zero-init signed-erase residual is not zero")
    return {
        "output_exact_identity": True,
        "terminal_state_max_abs_errors": state_errors,
        "incoming_state_identity": incoming_identity,
        "missing_parameters": missing,
        "parameter_delta": parameter_delta,
        "gradient_max_abs_by_layer": gradients,
        "official_call_lengths": candidate_calls,
        "official_chunk_backward_counts": chunk_counts,
        "official_gdn2_layers": official_layers,
        "parent_allow_neg_eigval": False,
        "position_qk_active": True,
        "zero_init_diag": {
            key: float(value.detach().float().item())
            for key, value in candidate_diag.items()
            if key.startswith("gdn3_adaptive_signed_erase_")
        },
    }


def check_open_transition_and_equivariance(
    device: torch.device,
) -> dict[str, Any]:
    torch.manual_seed(12010)
    model = build(device, update_mode="adaptive_signed_erase")
    block = model.blocks[0].time_mix
    projection = block.adaptive_signed_erase_proj
    if projection is None:
        raise AssertionError("adaptive signed-erase projection is missing")
    with torch.no_grad():
        projection.weight.zero_()
        projection.weight.copy_(1.5 * torch.eye(HEAD_DIM, device=device))
        value = torch.randn(3, 17, HEADS, VALUE_DIM, device=device)
        erase = torch.sigmoid(
            torch.randn(3, 17, HEADS, HEAD_DIM, device=device)
        )
        effective, diagnostics = block._adaptive_signed_erase(value, erase)
        permutation = torch.randperm(HEADS, device=device)
        inverse = torch.argsort(permutation)
        permuted, _ = block._adaptive_signed_erase(
            value.index_select(2, permutation),
            erase.index_select(2, permutation),
        )
        equivariance_error = float(
            (
                effective.float()
                - permuted.index_select(2, inverse).float()
            )
            .abs()
            .max()
            .item()
        )
    effective_min = float(effective.float().min().item())
    effective_max = float(effective.float().max().item())
    above_one_frac = float(
        diagnostics["gdn3_adaptive_signed_erase_above_one_frac"].item()
    )
    below_one_frac = float((effective.float() < 1.0).float().mean().item())
    if effective_min < 0.0 or effective_max > 2.0:
        raise AssertionError(
            f"effective erase escaped [0,2]: {effective_min}, {effective_max}"
        )
    if above_one_frac <= 0.0 or below_one_frac <= 0.0:
        raise AssertionError(
            "opened signed erase did not span both sides of one: "
            f"above={above_one_frac} below={below_one_frac}"
        )
    if equivariance_error > 3e-6:
        raise AssertionError(
            f"head permutation equivariance error {equivariance_error}"
        )

    model.eval()
    x = torch.randn(1, TOKENS, MODEL_DIM, device=device)
    address = torch.randn_like(x)
    cell_order = torch.randperm(TOKENS, device=device)
    with torch.no_grad():
        projection.weight.zero_()
        zero_output, _zero_diag, zero_states = capture_terminal_states(
            model, x, address, cell_order
        )
        for layer_block in model.blocks:
            layer_projection = layer_block.time_mix.adaptive_signed_erase_proj
            if layer_projection is None:
                raise AssertionError("opened layer projection is missing")
            layer_projection.weight.copy_(
                0.25 * torch.eye(HEAD_DIM, device=device)
            )
        open_output, open_diag, open_states = capture_terminal_states(
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
            f"opened transition did not change output/state: {output_change}, {state_changes}"
        )
    model.train()
    model.zero_grad(set_to_none=True)
    opened_train_output, _opened_train_diag, _opened_train_states = (
        capture_terminal_states(
            model,
            x.detach().clone().requires_grad_(True),
            address,
            cell_order,
        )
    )
    opened_train_output.float().square().mean().backward()
    opened_gradient_max: dict[str, float] = {}
    for layer, layer_block in enumerate(model.blocks):
        layer_projection = layer_block.time_mix.adaptive_signed_erase_proj
        if layer_projection is None or layer_projection.weight.grad is None:
            raise AssertionError(f"layer {layer} opened gradient is missing")
        gradient = layer_projection.weight.grad
        if not bool(torch.isfinite(gradient).all()):
            raise AssertionError(f"layer {layer} opened gradient is non-finite")
        grad_max = float(gradient.float().abs().max().item())
        if grad_max <= 0.0:
            raise AssertionError(f"layer {layer} opened gradient is zero")
        opened_gradient_max[str(layer)] = grad_max
    return {
        "effective_min": effective_min,
        "effective_max": effective_max,
        "above_one_frac": above_one_frac,
        "below_one_frac": below_one_frac,
        "head_permutation_max_abs_error": equivariance_error,
        "output_max_abs_change": output_change,
        "terminal_state_max_abs_changes": state_changes,
        "opened_gradient_max_abs_by_layer": opened_gradient_max,
        "opened_diag": {
            key: float(value.detach().float().item())
            for key, value in open_diag.items()
            if key.startswith("gdn3_adaptive_signed_erase_")
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
    props = torch.cuda.get_device_properties(device)
    output = {
        "device": str(device),
        "gpu_name": props.name,
        "gpu_uuid": EXPECTED_GPU_UUID,
        "fla_source_sha": study.GAIN_BUDGET_FLA_SHA,
        "zero_identity_and_learning_path": (
            check_zero_identity_and_learning_path(device)
        ),
        "open_transition_and_equivariance": (
            check_open_transition_and_equivariance(device)
        ),
    }
    print(json.dumps(output, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
