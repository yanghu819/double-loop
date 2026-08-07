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
HEAD_DIM = 32
EXPANDED_KEY_DIM = 64
VALUE_DIM = 32
MODEL_DIM = HEADS * HEAD_DIM
TOKENS = 81
EXPECTED_PARAMETER_DELTA = LAYERS * MODEL_DIM * MODEL_DIM
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
    expected = {
        f"blocks.{layer}.time_mix.coupled_address_k_proj.weight"
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
    call: Callable[[], Any],
) -> tuple[Any, list[dict[str, Any]]]:
    original = study.chunk_gdn2
    if original is None:
        raise RuntimeError("official chunk_gdn2 is unavailable")
    geometry: list[dict[str, Any]] = []

    def audited_chunk(*args: Any, **kwargs: Any) -> Any:
        q = kwargs.get("q")
        v = kwargs.get("v")
        if not isinstance(q, torch.Tensor) or not isinstance(v, torch.Tensor):
            raise AssertionError("official GDN2 call is missing q/v tensors")
        geometry.append(
            {
                "tokens": int(q.shape[1]),
                "heads": int(q.shape[2]),
                "key_dim": int(q.shape[3]),
                "value_dim": int(v.shape[3]),
                "scale": float(kwargs.get("scale", 1.0 / math.sqrt(q.shape[-1]))),
                "kernel_l2norm": bool(
                    kwargs.get("use_qk_l2norm_in_kernel", False)
                ),
            }
        )
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
    torch.manual_seed(14014)
    for layer, (control_block, candidate_block) in enumerate(
        zip(control.blocks, candidate.blocks)
    ):
        x = torch.randn(1, TOKENS, MODEL_DIM, device=device)
        address = torch.randn_like(x)
        cell_order = torch.randperm(TOKENS, device=device)
        base_initial = torch.randn(
            1,
            HEADS,
            HEAD_DIM,
            VALUE_DIM,
            device=device,
            dtype=torch.float32,
        )
        with torch.no_grad():
            control_output, control_state = direct_block_forward(
                control_block, x, address, cell_order, base_initial
            )
            candidate_output, candidate_state = direct_block_forward(
                candidate_block, x, address, cell_order, base_initial
            )
        candidate_base, candidate_extra = candidate_state.split(
            HEAD_DIM, dim=-2
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
            "extra_state": float(candidate_extra.float().abs().max().item()),
        }
        if any(value != 0.0 for value in errors.values()):
            raise AssertionError(
                f"layer {layer} nonzero-state identity failed: {errors}"
            )
        rows.append({"layer": float(layer), **errors})
    return rows


def check_identity_and_first_order(device: torch.device) -> dict[str, Any]:
    torch.manual_seed(14010)
    control = build(device, update_mode="none")
    torch.manual_seed(14010)
    candidate = build(device, update_mode="coupled_address_rows")
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
    control_output, _control_diag, control_states, control_incoming = control_result
    candidate_output, candidate_diag, candidate_states, candidate_incoming = (
        candidate_result
    )
    if not torch.equal(control_output, candidate_output):
        raise AssertionError(
            "external K32 normalization plus K64 zero bank changed full output"
        )
    expected_control_geometry = [
        {
            "tokens": TOKENS,
            "heads": HEADS,
            "key_dim": HEAD_DIM,
            "value_dim": VALUE_DIM,
            "scale": 1.0 / math.sqrt(HEAD_DIM),
            "kernel_l2norm": True,
        }
    ] * LAYERS
    if control_geometry != expected_control_geometry:
        raise AssertionError(f"control geometry changed: {control_geometry}")
    for row in candidate_geometry:
        if row != {
            "tokens": TOKENS,
            "heads": HEADS,
            "key_dim": EXPANDED_KEY_DIM,
            "value_dim": VALUE_DIM,
            "scale": 1.0 / math.sqrt(HEAD_DIM),
            "kernel_l2norm": False,
        }:
            raise AssertionError(f"candidate geometry changed: {candidate_geometry}")

    state_rows = []
    for layer, (control_state, candidate_state) in enumerate(
        zip(control_states, candidate_states)
    ):
        if tuple(candidate_state.shape[1:]) != (
            HEADS,
            EXPANDED_KEY_DIM,
            VALUE_DIM,
        ):
            raise AssertionError(
                f"layer {layer} state geometry {tuple(candidate_state.shape)}"
            )
        base, extra = candidate_state.split(HEAD_DIM, dim=-2)
        base_error = float(
            (control_state.float() - base.float()).abs().max().item()
        )
        extra_error = float(extra.float().abs().max().item())
        if base_error != 0.0 or extra_error != 0.0:
            raise AssertionError(
                f"layer {layer} state identity failed: {base_error}/{extra_error}"
            )
        state_rows.append(
            {
                "layer": layer,
                "base_max_abs_error": base_error,
                "extra_max_abs": extra_error,
            }
        )
    if control_incoming[0] is not None or candidate_incoming[0] is not None:
        raise AssertionError("layer zero unexpectedly received FutureSeed state")
    future_seed_rows = []
    for layer, (control_seed, candidate_seed) in enumerate(
        zip(control_incoming[1:], candidate_incoming[1:]), start=1
    ):
        if control_seed is None or candidate_seed is None:
            raise AssertionError(f"layer {layer} FutureSeed state is missing")
        candidate_base, candidate_extra = candidate_seed.split(
            HEAD_DIM, dim=-2
        )
        base_error = float(
            (control_seed.float() - candidate_base.float()).abs().max().item()
        )
        extra_error = float(candidate_extra.float().abs().max().item())
        if base_error != 0.0 or extra_error != 0.0:
            raise AssertionError(
                f"layer {layer} split-row FutureSeed identity failed: "
                f"{base_error}/{extra_error}"
            )
        future_seed_rows.append(
            {"layer": layer, "base_error": base_error, "extra_max_abs": extra_error}
        )

    parameter_delta = sum(p.numel() for p in candidate.parameters()) - sum(
        p.numel() for p in control.parameters()
    )
    if parameter_delta != EXPECTED_PARAMETER_DELTA:
        raise AssertionError(
            f"parameter delta {parameter_delta} != {EXPECTED_PARAMETER_DELTA}"
        )
    state_value_delta = sum(
        state.numel() // state.shape[0] for state in candidate_states
    ) - sum(state.numel() // state.shape[0] for state in control_states)
    if state_value_delta != EXPECTED_STATE_VALUE_DELTA:
        raise AssertionError(
            f"state delta {state_value_delta} != {EXPECTED_STATE_VALUE_DELTA}"
        )

    incoming_identity = check_nonzero_initial_state_identity(
        control, candidate, device
    )
    candidate.train()
    candidate.zero_grad(set_to_none=True)
    output, _diag, states, _incoming = capture_model(
        candidate,
        x.detach().clone().requires_grad_(True),
        address,
        cell_order,
    )
    target = torch.randn_like(output, dtype=torch.float32)
    (output.float() * target).mean().backward()
    gradient_rows: dict[str, dict[str, float]] = {}
    chunk_counts = []
    for layer, (block, state) in enumerate(zip(candidate.blocks, states)):
        projection = block.time_mix.coupled_address_k_proj
        if projection is None or projection.weight.grad is None:
            raise AssertionError(f"layer {layer} first-order K gradient is missing")
        gradient = projection.weight.grad.float().view(
            HEADS, HEAD_DIM, MODEL_DIM
        )
        head_norm = gradient.square().sum(dim=(1, 2)).sqrt()
        if not bool(torch.isfinite(gradient).all()) or float(head_norm.min()) <= 0.0:
            raise AssertionError(
                f"layer {layer} first-order K gradient is dead/non-finite"
            )
        gradient_rows[str(layer)] = {
            "head_norm_min": float(head_norm.min().item()),
            "head_norm_max": float(head_norm.max().item()),
        }
        chunk_counts.append(graph_names(state).count("ChunkGDN2FunctionBackward"))
    if chunk_counts != [1] * LAYERS:
        raise AssertionError(f"official backward graph mismatch: {chunk_counts}")
    if float(candidate_diag["gdn3_coupled_address_rows_enabled"].item()) != 1.0:
        raise AssertionError("coupled-row diagnostics did not mark the path active")
    zero_keys = (
        "gdn3_coupled_address_rows_k_weight_rms",
        "gdn3_coupled_address_rows_k_residual_relative_rms",
        "gdn3_coupled_address_rows_k_residual_batch_std",
        "gdn3_coupled_address_rows_k_residual_token_std",
        "gdn3_coupled_address_rows_k_residual_head_std",
        "gdn3_coupled_address_rows_k_norm_max",
        "gdn3_coupled_address_rows_extra_state_relative_rms",
        "gdn3_coupled_address_rows_extra_state_batch_std",
        "gdn3_coupled_address_rows_extra_state_head_std",
        "gdn3_coupled_address_rows_extra_state_rms",
    )
    for key in zero_keys:
        value = float(candidate_diag[key].detach().float().item())
        if value != 0.0:
            raise AssertionError(f"zero-init diagnostic is not zero: {key}={value}")
    for key in (
        "gdn3_coupled_address_rows_base_state_rms",
        "gdn3_coupled_address_rows_terminal_rms",
    ):
        value = float(candidate_diag[key].detach().float().item())
        if not math.isfinite(value) or value <= 0.0:
            raise AssertionError(f"zero-init parent state diagnostic is invalid: {key}={value}")

    official_layers = [type(block.time_mix.core).__name__ for block in candidate.blocks]
    if official_layers != ["GatedDeltaNet2"] * LAYERS:
        raise AssertionError(f"unexpected recurrent layers: {official_layers}")
    return {
        "output_exact_identity": True,
        "state_identity": state_rows,
        "future_seed_identity": future_seed_rows,
        "incoming_state_identity": incoming_identity,
        "missing_parameters": missing,
        "parameter_delta": parameter_delta,
        "state_value_delta": state_value_delta,
        "first_order_k_gradient_by_layer": gradient_rows,
        "official_geometry": candidate_geometry,
        "official_chunk_backward_counts": chunk_counts,
        "official_gdn2_layers": official_layers,
    }


def check_opened_geometry(device: torch.device) -> dict[str, Any]:
    torch.manual_seed(14011)
    model = build(device, update_mode="coupled_address_rows")
    model.eval()
    x = torch.randn(2, TOKENS, MODEL_DIM, device=device)
    address = torch.randn_like(x)
    cell_order = torch.randperm(TOKENS, device=device)
    with torch.no_grad():
        zero_output, _zero_diag, _zero_states, _ = capture_model(
            model, x, address, cell_order
        )
        identity = torch.eye(MODEL_DIM, device=device)
        for layer, block in enumerate(model.blocks):
            projection = block.time_mix.coupled_address_k_proj
            assert projection is not None
            projection.weight.copy_((0.01 + layer * 0.0001) * identity)
        open_output, open_diag, open_states, _ = capture_model(
            model, x, address, cell_order
        )
    output_change = float(
        (open_output.float() - zero_output.float()).abs().max().item()
    )
    if output_change <= 0.0:
        raise AssertionError("opened K rows did not change model output")
    state_relative = []
    state_ratio = []
    for layer, state in enumerate(open_states):
        base, extra = state.float().split(HEAD_DIM, dim=-2)
        base_rms = base.square().mean().sqrt().clamp_min(1e-8)
        extra_rms = extra.square().mean().sqrt()
        relative = float((extra_rms / base_rms).item())
        if not math.isfinite(relative) or not 0.0 < relative <= 4.0:
            raise AssertionError(
                f"layer {layer} opened state ratio is invalid: {relative}"
            )
        state_relative.append(relative)
        state_ratio.append(float(extra_rms.item()))
    required = (
        "gdn3_coupled_address_rows_k_residual_relative_rms",
        "gdn3_coupled_address_rows_k_residual_batch_std",
        "gdn3_coupled_address_rows_k_residual_token_std",
        "gdn3_coupled_address_rows_k_residual_head_std",
        "gdn3_coupled_address_rows_extra_state_relative_rms",
        "gdn3_coupled_address_rows_terminal_rms",
    )
    for key in required:
        value = float(open_diag[key].detach().float().item())
        if not math.isfinite(value) or value <= 0.0:
            raise AssertionError(f"opened activation is invalid: {key}={value}")
    k_norm_max = float(
        open_diag["gdn3_coupled_address_rows_k_norm_max"].item()
    )
    if not 0.0 < k_norm_max <= 1.000001:
        raise AssertionError(f"soft-unit-ball K norm escaped: {k_norm_max}")

    first = model.blocks[0].time_mix.coupled_address_k_proj
    assert first is not None
    permutation = torch.randperm(HEAD_DIM, device=device)
    weight = first.weight.detach().view(HEADS, HEAD_DIM, MODEL_DIM)
    permuted_weight = weight.index_select(1, permutation)
    direct = torch.einsum("btd,hkd->bthk", address.float(), weight.float())
    permuted = torch.einsum(
        "btd,hkd->bthk", address.float(), permuted_weight.float()
    )
    equivariance_error = float(
        (
            direct.index_select(-1, permutation)
            - permuted
        ).abs().max().item()
    )
    if equivariance_error > 3e-6:
        raise AssertionError(f"K-row permutation error {equivariance_error}")
    return {
        "output_max_abs_change": output_change,
        "extra_state_relative_rms_by_layer": state_relative,
        "extra_state_rms_by_layer": state_ratio,
        "k_row_permutation_max_abs_error": equivariance_error,
        "opened_diag": {
            key: float(value.detach().float().item())
            for key, value in open_diag.items()
            if key.startswith("gdn3_coupled_address_rows_")
        },
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
    fla_source_sha, fla_root, fla_module = study.resolve_strict_fla_source(
        "gdn2"
    )
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
    runtime_model = build(device, update_mode="coupled_address_rows")
    runtime = study.strict_fla_runtime_summary(runtime_model, "gdn2")
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
        "strict_runtime": runtime,
    }
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
