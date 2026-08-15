#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import subprocess
from typing import Any, Optional

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


EXPECTED_GPU_UUID = "GPU-d2877fe4-641c-fe64-2a74-8abca47c292f"
EXPECTED_FLA_SOURCE_SHA = "9c8e42e762fce087c27b673af4922795d9edb85e"
LAYERS = 12
HEADS = 8
HEAD_DIM = 32
TOKENS = 81
MODEL_DIM = HEADS * HEAD_DIM
EXPECTED_PARAMETER_DELTA = (LAYERS - 1) * HEADS


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
        MODEL_DIM,
        LAYERS,
        HEADS,
        HEAD_DIM,
        4,
        future_seed_scale=1.0,
        future_seed_decay=0.0,
        future_seed_update=update_mode,
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
        gdn2_cross_layer_init="independent",
    ).to(device)


def copy_shared_state(
    control: FutureSeedRWKV,
    candidate: FutureSeedRWKV,
) -> list[str]:
    missing, unexpected = candidate.load_state_dict(
        control.state_dict(), strict=False
    )
    expected = {"future_seed_secant_raw"}
    if set(missing) != expected or unexpected:
        raise AssertionError(
            f"unexpected state migration: missing={missing}, unexpected={unexpected}"
        )
    return sorted(missing)


def capture_pass(
    model: FutureSeedRWKV,
    x: torch.Tensor,
    address: torch.Tensor,
    cell_order: torch.Tensor,
    seed_memory: Optional[list[torch.Tensor]],
) -> tuple[
    torch.Tensor,
    dict[str, torch.Tensor],
    list[torch.Tensor],
    Optional[list[torch.Tensor]],
]:
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
            output, diagnostics, next_memory = model(
                x,
                seed_memory=seed_memory,
                address=address,
                cell_order=cell_order,
            )
    finally:
        for hook in hooks:
            hook.remove()
    return output, diagnostics, states, next_memory


def random_seed_memory(
    device: torch.device,
    *,
    batch: int,
) -> list[torch.Tensor]:
    return [
        torch.randn(
            batch,
            HEADS,
            HEAD_DIM,
            HEAD_DIM,
            device=device,
            dtype=torch.bfloat16,
        )
        for _ in range(LAYERS - 1)
    ]


def check_zero_identity(device: torch.device) -> dict[str, Any]:
    torch.manual_seed(12130)
    control = build(device, update_mode="fixed")
    candidate = build(device, update_mode="loop_secant")
    missing = copy_shared_state(control, candidate)
    control.eval()
    candidate.eval()

    parameter_delta = sum(p.numel() for p in candidate.parameters()) - sum(
        p.numel() for p in control.parameters()
    )
    if parameter_delta != EXPECTED_PARAMETER_DELTA:
        raise AssertionError(
            f"parameter delta {parameter_delta} != {EXPECTED_PARAMETER_DELTA}"
        )

    address = torch.randn(2, TOKENS, MODEL_DIM, device=device)
    cell_order = torch.randperm(TOKENS, device=device)
    candidate_memory: Optional[list[torch.Tensor]] = None
    pass_rows = []
    with torch.no_grad():
        for pass_idx in range(5):
            x = torch.randn(2, TOKENS, MODEL_DIM, device=device)
            control_output, _control_diag, control_states, _ = capture_pass(
                control, x, address, cell_order, None
            )
            candidate_output, candidate_diag, candidate_states, next_memory = (
                capture_pass(
                    candidate,
                    x,
                    address,
                    cell_order,
                    candidate_memory,
                )
            )
            if not torch.equal(control_output, candidate_output):
                raise AssertionError(
                    f"zero-init output identity failed at pass {pass_idx + 1}"
                )
            state_errors = [
                float((left.float() - right.float()).abs().max().item())
                for left, right in zip(control_states, candidate_states)
            ]
            if len(state_errors) != LAYERS or max(state_errors) != 0.0:
                raise AssertionError(
                    f"zero-init state identity failed at pass {pass_idx + 1}: "
                    f"{state_errors}"
                )
            if next_memory is None or len(next_memory) != LAYERS - 1:
                raise AssertionError("loop-secant did not return all producer states")
            memory_errors = [
                float((stored.float() - live.float()).abs().max().item())
                for stored, live in zip(next_memory, candidate_states[:-1])
            ]
            if max(memory_errors) != 0.0:
                raise AssertionError(
                    f"producer memory drifted at pass {pass_idx + 1}: {memory_errors}"
                )
            pass_rows.append(
                {
                    "pass": pass_idx + 1,
                    "output_exact": True,
                    "state_max_abs_error": max(state_errors),
                    "memory_max_abs_error": max(memory_errors),
                    "delta_relative_rms": float(
                        candidate_diag[
                            "fs2_loop_secant_delta_relative_rms"
                        ].float().item()
                    ),
                }
            )
            candidate_memory = next_memory

    nonzero_memory = random_seed_memory(device, batch=2)
    x = torch.randn(2, TOKENS, MODEL_DIM, device=device)
    with torch.no_grad():
        control_output, _control_diag, control_states, _ = capture_pass(
            control, x, address, cell_order, None
        )
        candidate_output, candidate_diag, candidate_states, _ = capture_pass(
            candidate, x, address, cell_order, nonzero_memory
        )
    if not torch.equal(control_output, candidate_output):
        raise AssertionError("nonzero-history zero-init output identity failed")
    nonzero_state_errors = [
        float((left.float() - right.float()).abs().max().item())
        for left, right in zip(control_states, candidate_states)
    ]
    if max(nonzero_state_errors) != 0.0:
        raise AssertionError(
            f"nonzero-history zero-init state identity failed: {nonzero_state_errors}"
        )
    if float(candidate_diag["fs2_loop_secant_scale_abs"].float().item()) != 0.0:
        raise AssertionError("zero-init secant scale is not zero")

    return {
        "missing_parameters": missing,
        "parameter_delta": parameter_delta,
        "five_pass_identity": pass_rows,
        "nonzero_history_state_max_abs_error": max(nonzero_state_errors),
    }


def check_gradient_and_open_path(device: torch.device) -> dict[str, Any]:
    torch.manual_seed(12131)
    candidate = build(device, update_mode="loop_secant")
    candidate.train()
    address = torch.randn(3, TOKENS, MODEL_DIM, device=device)
    cell_order = torch.randperm(TOKENS, device=device)
    x = torch.randn(3, TOKENS, MODEL_DIM, device=device)
    prior_memory = random_seed_memory(device, batch=3)

    candidate.zero_grad(set_to_none=True)
    output, diagnostics, states, _ = capture_pass(
        candidate, x, address, cell_order, prior_memory
    )
    target = torch.randn_like(output, dtype=torch.float32)
    (output.float() * target).mean().backward()
    raw = candidate.future_seed_secant_raw
    if raw is None or raw.grad is None or not bool(torch.isfinite(raw.grad).all()):
        raise AssertionError("loop-secant gradient is missing or non-finite")
    gradient = raw.grad.detach().float().abs().reshape(LAYERS - 1, HEADS)
    nonzero_mask = gradient > 0
    if not bool(nonzero_mask.all()):
        raise AssertionError(
            f"not all 88 loop-secant coefficients receive gradient: "
            f"{int(nonzero_mask.sum().item())}/{gradient.numel()}"
        )
    chunk_presence = [
        "ChunkGDN2FunctionBackward" in graph_names(state) for state in states
    ]
    if chunk_presence != [True] * LAYERS:
        raise AssertionError(
            f"official GDN2 chunk backward missing by layer: {chunk_presence}"
        )
    official_layers = [type(block.time_mix.core).__name__ for block in candidate.blocks]
    if official_layers != ["GatedDeltaNet2"] * LAYERS:
        raise AssertionError(f"unexpected recurrent layers: {official_layers}")
    if float(diagnostics["fs2_loop_secant_enabled"].float().item()) != 1.0:
        raise AssertionError("loop-secant diagnostics are not active")

    with torch.no_grad():
        raw.fill_(0.2)
        opened_output, opened_diag, opened_states, _ = capture_pass(
            candidate, x, address, cell_order, prior_memory
        )
    output_change = float(
        (opened_output.float() - output.detach().float()).square().mean().sqrt().item()
    )
    activation = {
        key: float(opened_diag[key].detach().float().item())
        for key in (
            "fs2_loop_secant_scale_abs",
            "fs2_loop_secant_scale_signed",
            "fs2_loop_secant_delta_relative_rms",
            "fs2_loop_secant_residual_relative_rms",
            "fs2_loop_secant_residual_batch_std",
        )
    }
    if output_change <= 0.0:
        raise AssertionError("opened loop-secant path does not change output")
    if not all(torch.isfinite(state).all() for state in opened_states):
        raise AssertionError("opened loop-secant path produced non-finite state")
    if activation["fs2_loop_secant_scale_abs"] <= 0.0:
        raise AssertionError(f"opened scale is inactive: {activation}")
    if activation["fs2_loop_secant_delta_relative_rms"] <= 0.0:
        raise AssertionError(f"opened history delta is inactive: {activation}")
    if activation["fs2_loop_secant_residual_relative_rms"] <= 0.0:
        raise AssertionError(f"opened residual is inactive: {activation}")
    if activation["fs2_loop_secant_residual_batch_std"] <= 0.0:
        raise AssertionError(f"opened residual lacks board variation: {activation}")
    if activation["fs2_loop_secant_scale_abs"] > 0.5 + 1e-6:
        raise AssertionError(f"secant coefficient exceeded its bound: {activation}")
    if activation["fs2_loop_secant_residual_relative_rms"] > 0.5 + 1e-5:
        raise AssertionError(f"secant residual exceeded its RMS bound: {activation}")

    return {
        "all_88_gradients_finite_nonzero": True,
        "gradient_min_abs": float(gradient.min().item()),
        "gradient_max_abs": float(gradient.max().item()),
        "official_chunk_backward_by_layer": chunk_presence,
        "official_gdn2_layers": official_layers,
        "opened_output_rms_change": output_change,
        "opened_activation": activation,
    }


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required; CPU model smoke is forbidden")
    if torch.cuda.device_count() != 1:
        raise RuntimeError(
            f"exactly one visible CUDA device is required, got {torch.cuda.device_count()}"
        )
    visible_gpu = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=index,uuid",
            "--format=csv,noheader",
        ],
        text=True,
    ).strip().splitlines()
    if visible_gpu != [f"0, {EXPECTED_GPU_UUID}"]:
        raise RuntimeError(f"unexpected visible GPU contract: {visible_gpu}")
    if GAIN_BUDGET_FLA_SHA != EXPECTED_FLA_SOURCE_SHA:
        raise RuntimeError(
            f"pinned FLA SHA drift: {GAIN_BUDGET_FLA_SHA} != {EXPECTED_FLA_SOURCE_SHA}"
        )

    device = torch.device("cuda:0")
    result = {
        "plan": "P-FS2-013",
        "status": "pass",
        "gpu": {
            "index": 0,
            "uuid": EXPECTED_GPU_UUID,
            "name": torch.cuda.get_device_name(device),
        },
        "pinned_fla_source_sha": GAIN_BUDGET_FLA_SHA,
        "zero_identity": check_zero_identity(device),
        "learning_path": check_gradient_and_open_path(device),
    }
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
