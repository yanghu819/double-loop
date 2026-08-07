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
PHYSICAL_STEPS = 2 * TOKENS
EXPECTED_PARAMETER_DELTA = LAYERS * 2 * MODEL_DIM * MODEL_DIM


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
        f"blocks.{layer}.time_mix.interleaved_write_{kind}_proj.weight"
        for layer in range(LAYERS)
        for kind in ("k", "v")
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
    call: Callable[[], tuple[torch.Tensor, dict[str, torch.Tensor], list[torch.Tensor]]],
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
    torch.manual_seed(10010)
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
                f"layer {layer} zero-write identity failed: "
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


def open_learning_path(
    candidate: study.FutureSeedRWKV,
    x: torch.Tensor,
    address: torch.Tensor,
    cell_order: torch.Tensor,
) -> dict[str, dict[str, float]]:
    candidate.train()
    candidate.zero_grad(set_to_none=True)
    output, _diagnostics, states = capture_terminal_states(
        candidate, x, address, cell_order
    )
    chunk_counts = [
        graph_names(state).count("ChunkGDN2FunctionBackward")
        for state in states
    ]
    if chunk_counts != [1] * LAYERS:
        raise AssertionError(
            f"single-call official backward graph mismatch: {chunk_counts}"
        )
    output.float().square().mean().backward()

    v_gradient_max: dict[str, float] = {}
    with torch.no_grad():
        for layer, block in enumerate(candidate.blocks):
            time_mix = block.time_mix
            v_projection = time_mix.interleaved_write_v_proj
            k_projection = time_mix.interleaved_write_k_proj
            if v_projection is None or k_projection is None:
                raise AssertionError(f"layer {layer} projections are missing")
            v_gradient = v_projection.weight.grad
            if v_gradient is None or not bool(torch.isfinite(v_gradient).all()):
                raise AssertionError(
                    f"layer {layer} first-stage V gradient is missing/non-finite"
                )
            value = float(v_gradient.float().abs().max().item())
            if value <= 0.0:
                raise AssertionError(f"layer {layer} first-stage V gradient is zero")
            v_gradient_max[str(layer)] = value
            v_projection.weight.add_(
                -0.01
                * v_gradient
                / v_gradient.float().abs().max().clamp_min(1e-8)
            )

    candidate.zero_grad(set_to_none=True)
    opened_v_output, _opened_v_diag, _opened_v_states = capture_terminal_states(
        candidate, x, address, cell_order
    )
    opened_v_output.float().square().mean().backward()
    k_gradient_max: dict[str, float] = {}
    with torch.no_grad():
        for layer, block in enumerate(candidate.blocks):
            projection = block.time_mix.interleaved_write_k_proj
            assert projection is not None
            gradient = projection.weight.grad
            if gradient is None or not bool(torch.isfinite(gradient).all()):
                raise AssertionError(
                    f"layer {layer} opened K gradient is missing/non-finite"
                )
            value = float(gradient.float().abs().max().item())
            if value <= 0.0:
                raise AssertionError(f"layer {layer} opened K gradient is zero")
            k_gradient_max[str(layer)] = value
            projection.weight.add_(
                -0.01
                * gradient
                / gradient.float().abs().max().clamp_min(1e-8)
            )
    return {
        "v_gradient_max_by_layer": v_gradient_max,
        "k_gradient_max_by_layer": k_gradient_max,
    }


def check_dependencies_and_head_geometry(
    candidate: study.FutureSeedRWKV,
    device: torch.device,
) -> dict[str, float]:
    torch.manual_seed(10011)
    block = candidate.blocks[0]
    x = torch.randn(2, TOKENS, MODEL_DIM, device=device)
    address = torch.randn_like(x)
    cell_order = torch.randperm(TOKENS, device=device)
    state = torch.randn(
        2,
        HEADS,
        HEAD_DIM,
        VALUE_DIM,
        device=device,
        dtype=torch.float32,
    )
    with torch.no_grad():
        base_output, base_state = direct_block_forward(
            block, x, address, cell_order, state
        )
        shuffled_source_output, shuffled_source_state = direct_block_forward(
            block, x.roll(1, dims=0), address, cell_order, state
        )
        shuffled_state_output, shuffled_state = direct_block_forward(
            block, x, address, cell_order, state.roll(1, dims=0)
        )
    source_dependency = float(
        (base_state.float() - shuffled_source_state.float())
        .square()
        .mean()
        .sqrt()
        .item()
    )
    state_dependency = float(
        (base_state.float() - shuffled_state.float())
        .square()
        .mean()
        .sqrt()
        .item()
    )
    output_dependency = max(
        float(
            (base_output.float() - shuffled_source_output.float())
            .square()
            .mean()
            .sqrt()
            .item()
        ),
        float(
            (base_output.float() - shuffled_state_output.float())
            .square()
            .mean()
            .sqrt()
            .item()
        ),
    )
    if min(source_dependency, state_dependency, output_dependency) <= 0.0:
        raise AssertionError(
            "opened interleaved write lacks source/state dependency: "
            f"{source_dependency}, {state_dependency}, {output_dependency}"
        )

    q = torch.randn(2, TOKENS, HEADS, HEAD_DIM, device=device)
    k = torch.randn_like(q)
    v = torch.randn(2, TOKENS, HEADS, VALUE_DIM, device=device)
    g = torch.randn_like(q, dtype=torch.float32)
    b = torch.sigmoid(torch.randn_like(q))
    w = torch.sigmoid(torch.randn_like(v))
    permutation = torch.randperm(HEADS, device=device)
    inverse = torch.argsort(permutation)

    def interleave(auxiliary: torch.Tensor, parent: torch.Tensor) -> torch.Tensor:
        return torch.stack((auxiliary, parent), dim=2).reshape(
            auxiliary.shape[0], 2 * TOKENS, *auxiliary.shape[2:]
        )

    original = study.chunk_gdn2
    assert original is not None
    with torch.no_grad(), torch.autocast(
        device_type="cuda", dtype=torch.bfloat16
    ):
        base, _ = original(
            q=interleave(q, q),
            k=interleave(k, k),
            v=interleave(v, v),
            g=interleave(torch.zeros_like(g), g),
            b=interleave(torch.zeros_like(b), b),
            w=interleave(torch.ones_like(w), w),
            initial_state=None,
            output_final_state=True,
            use_qk_l2norm_in_kernel=True,
        )
        permuted, _ = original(
            q=interleave(q[:, :, permutation], q[:, :, permutation]),
            k=interleave(k[:, :, permutation], k[:, :, permutation]),
            v=interleave(v[:, :, permutation], v[:, :, permutation]),
            g=interleave(
                torch.zeros_like(g[:, :, permutation]),
                g[:, :, permutation],
            ),
            b=interleave(
                torch.zeros_like(b[:, :, permutation]),
                b[:, :, permutation],
            ),
            w=interleave(
                torch.ones_like(w[:, :, permutation]),
                w[:, :, permutation],
            ),
            initial_state=None,
            output_final_state=True,
            use_qk_l2norm_in_kernel=True,
        )
    equivariance_error = float(
        (base.float() - permuted[:, :, inverse].float()).abs().max().item()
    )
    if equivariance_error > 3e-6:
        raise AssertionError(f"head permutation error: {equivariance_error}")
    return {
        "source_dependency_rms": source_dependency,
        "state_dependency_rms": state_dependency,
        "output_dependency_rms": output_dependency,
        "head_permutation_max_abs_error": equivariance_error,
    }


def check_zero_identity_and_learning_path(device: torch.device) -> dict[str, Any]:
    torch.manual_seed(10009)
    control = build(device, update_mode="none")
    torch.manual_seed(10009)
    candidate = build(device, update_mode="interleaved_write")
    missing = copy_shared_state(control, candidate)
    control.eval()
    candidate.eval()

    x = torch.randn(2, TOKENS, MODEL_DIM, device=device)
    address = torch.randn_like(x)
    cell_order = torch.randperm(TOKENS, device=device)
    with torch.no_grad():
        control_output, _control_diag, control_states = capture_terminal_states(
            control, x, address, cell_order
        )
        (
            (candidate_output, candidate_diag, candidate_states),
            official_call_lengths,
        ) = capture_official_call_lengths(
            lambda: capture_terminal_states(candidate, x, address, cell_order)
        )
    if official_call_lengths != [PHYSICAL_STEPS] * LAYERS:
        raise AssertionError(
            f"official call lengths are not 12x{PHYSICAL_STEPS}: "
            f"{official_call_lengths}"
        )
    if not torch.equal(control_output, candidate_output):
        error = float(
            (control_output.float() - candidate_output.float()).abs().max().item()
        )
        raise AssertionError(f"zero auxiliary write changed model output: {error}")
    state_errors = [
        float((left.float() - right.float()).abs().max().item())
        for left, right in zip(control_states, candidate_states)
    ]
    if len(state_errors) != LAYERS or max(state_errors) != 0.0:
        raise AssertionError(f"zero-write terminal-state mismatch: {state_errors}")

    parameter_delta = sum(p.numel() for p in candidate.parameters()) - sum(
        p.numel() for p in control.parameters()
    )
    if parameter_delta != EXPECTED_PARAMETER_DELTA:
        raise AssertionError(
            f"parameter delta {parameter_delta} != {EXPECTED_PARAMETER_DELTA}"
        )
    initial_state_rows = check_nonzero_initial_state_identity(
        control, candidate, device
    )
    gradients = open_learning_path(candidate, x, address, cell_order)
    candidate.eval()
    with torch.no_grad():
        opened_output, opened_diag, opened_states = capture_terminal_states(
            candidate, x, address, cell_order
        )
    if torch.equal(candidate_output, opened_output) or all(
        torch.equal(left, right)
        for left, right in zip(candidate_states, opened_states)
    ):
        raise AssertionError("opened interleaved write did not alter output and state")
    active_layers = sum(
        int(
            float(
                block.time_mix.last_gain_budget_diag.get(
                    "gdn3_interleaved_write_enabled", x.new_zeros(())
                ).item()
            )
            > 0.0
        )
        for block in candidate.blocks
    )
    if active_layers != LAYERS:
        raise AssertionError(
            f"expected {LAYERS} active interleaved paths, got {active_layers}"
        )
    for key in INTERLEAVED_OPEN_METRICS:
        value = float(opened_diag[key].item())
        if not torch.isfinite(torch.tensor(value)) or value <= 0.0:
            raise AssertionError(f"opened metric {key} is invalid: {value}")
    for key in (
        "gdn3_interleaved_write_k_residual_rms",
        "gdn3_interleaved_write_v_rms",
        "gdn3_interleaved_write_state_write_relative_rms",
    ):
        if float(candidate_diag[key].item()) != 0.0:
            raise AssertionError(f"zero-write metric {key} is not zero")

    official_layers = [type(block.time_mix.core).__name__ for block in candidate.blocks]
    if official_layers != ["GatedDeltaNet2"] * LAYERS:
        raise AssertionError(f"unexpected recurrent layers: {official_layers}")
    return {
        "output_exact_identity": True,
        "terminal_state_max_abs_errors": state_errors,
        "nonzero_initial_state_identity": initial_state_rows,
        "missing_parameters": missing,
        "parameter_delta": parameter_delta,
        "official_call_lengths": official_call_lengths,
        "controller_gradients": gradients,
        "active_layers": active_layers,
        "official_layers": official_layers,
        "official_chunk_backward_paths": LAYERS,
        "position_qk_active": float(candidate_diag["gdn2_address_enabled"].item()),
        "opened_diagnostics": {
            key: float(value.detach().float().item())
            for key, value in opened_diag.items()
            if key.startswith("gdn3_interleaved_write_")
        },
        "dependencies_and_head_geometry": check_dependencies_and_head_geometry(
            candidate, device
        ),
    }


INTERLEAVED_OPEN_METRICS = (
    "gdn3_interleaved_write_k_residual_rms",
    "gdn3_interleaved_write_k_residual_relative_rms",
    "gdn3_interleaved_write_k_batch_std",
    "gdn3_interleaved_write_k_token_std",
    "gdn3_interleaved_write_v_rms",
    "gdn3_interleaved_write_v_relative_rms",
    "gdn3_interleaved_write_v_batch_std",
    "gdn3_interleaved_write_v_token_std",
    "gdn3_interleaved_write_state_write_relative_rms",
    "gdn3_interleaved_write_state_write_batch_std",
    "gdn3_interleaved_write_terminal_rms",
    "gdn3_interleaved_write_terminal_batch_std",
    "gdn3_interleaved_write_k_weight_rms",
    "gdn3_interleaved_write_v_weight_rms",
)


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
        "zero_identity_and_learning_path": check_zero_identity_and_learning_path(
            device
        ),
    }
    print(json.dumps(output, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
