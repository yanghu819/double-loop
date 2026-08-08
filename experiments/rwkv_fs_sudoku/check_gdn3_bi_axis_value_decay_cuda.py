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
import torch.nn.functional as F

import study_rwkv_futureseed_loop as study


EXPECTED_GPU_UUID = "GPU-53e9f3b4-2966-65d3-6614-09c540921519"
EXPECTED_FLA_SOURCE_SHA = "9c8e42e762fce087c27b673af4922795d9edb85e"
LAYERS = 12
HEADS = 8
HEAD_DIM = 32
VALUE_DIM = 32
VALUE_GROUPS = 8
MODEL_DIM = HEADS * HEAD_DIM
TOKENS = 81
EXPECTED_PARAMETER_DELTA = LAYERS * MODEL_DIM * HEADS * VALUE_GROUPS


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
        f"blocks.{layer}.time_mix.bi_axis_value_decay_proj.weight"
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
    torch.manual_seed(15012)
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
    torch.manual_seed(15009)
    control = build(device, update_mode="none")
    torch.manual_seed(15009)
    candidate = build(device, update_mode="bi_axis_value_decay")
    missing = copy_shared_state(control, candidate)
    control.eval()
    candidate.eval()

    x = torch.randn(2, TOKENS, MODEL_DIM, device=device)
    address = torch.randn_like(x)
    cell_order = torch.randperm(TOKENS, device=device)
    with torch.no_grad():
        control_result, control_calls = capture_official_call_lengths(
            lambda: capture_terminal_states(control, x, address, cell_order)
        )
        candidate_result, candidate_calls = capture_official_call_lengths(
            lambda: capture_terminal_states(candidate, x, address, cell_order)
        )
    control_output, _control_diag, control_states = control_result
    candidate_output, candidate_diag, candidate_states = candidate_result
    if control_calls != [TOKENS] * LAYERS or candidate_calls != [TOKENS] * LAYERS:
        raise AssertionError(
            f"official call lengths changed: {control_calls}, {candidate_calls}"
        )
    if not torch.equal(control_output, candidate_output):
        error = float(
            (control_output.float() - candidate_output.float()).abs().max().item()
        )
        raise AssertionError(f"zero-init Bi-Axis output identity failed: {error}")
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
        projection = block.time_mix.bi_axis_value_decay_proj
        if projection is None or projection.weight.grad is None:
            raise AssertionError(f"layer {layer} Bi-Axis gradient is missing")
        gradient = projection.weight.grad
        if not bool(torch.isfinite(gradient).all()):
            raise AssertionError(f"layer {layer} Bi-Axis gradient is non-finite")
        grad_max = float(gradient.float().abs().max().item())
        if grad_max <= 0.0:
            raise AssertionError(f"layer {layer} Bi-Axis gradient is zero")
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
    if float(diagnostics["gdn2_address_enabled"].item()) != 1.0:
        raise AssertionError("position-QK address path is not active")
    if float(diagnostics["gdn3_bi_axis_value_decay_enabled"].item()) != 1.0:
        raise AssertionError("Bi-Axis diagnostics did not mark the path active")
    if float(diagnostics["gdn3_bi_axis_value_decay_log_decay_abs"].item()) != 0.0:
        raise AssertionError("zero-init value decay is not zero")
    for key in (
        "gdn3_bi_axis_value_decay_cumulative_scale_min",
        "gdn3_bi_axis_value_decay_cumulative_scale_mean",
        "gdn3_bi_axis_value_decay_inverse_scale_max",
    ):
        if float(diagnostics[key].item()) != 1.0:
            raise AssertionError(f"zero-init moving-frame identity failed: {key}")
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
        "position_qk_active": True,
        "zero_init_diag": {
            key: float(value.detach().float().item())
            for key, value in candidate_diag.items()
            if key.startswith("gdn3_bi_axis_value_decay_")
        },
    }


def direct_bi_axis_reference(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    gk: torch.Tensor,
    gv: torch.Tensor,
    b: torch.Tensor,
    w: torch.Tensor,
    initial_state: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    state = initial_state.float().clone()
    outputs = []
    for token in range(q.shape[1]):
        q_t = q[:, token].float()
        k_t = k[:, token].float()
        state = state * gk[:, token].float().exp().unsqueeze(-1)
        state = state * gv[:, token].float().exp().unsqueeze(-2)
        erase = (
            (b[:, token].float() * k_t).unsqueeze(-1) * state
        ).sum(dim=-2)
        update = w[:, token].float() * v[:, token].float() - erase
        state = state + k_t.unsqueeze(-1) * update.unsqueeze(-2)
        outputs.append((q_t.unsqueeze(-1) * state).sum(dim=-2))
    return torch.stack(outputs, dim=1), state


def check_moving_frame_reference(device: torch.device) -> dict[str, float]:
    if study.chunk_gdn2 is None:
        raise RuntimeError("official chunk_gdn2 is unavailable")
    torch.manual_seed(15010)
    batch, tokens, heads, key_dim, value_dim = 1, 17, 2, 8, 8
    q = F.normalize(
        torch.randn(batch, tokens, heads, key_dim, device=device), dim=-1
    ).to(torch.bfloat16)
    k = F.normalize(
        torch.randn(batch, tokens, heads, key_dim, device=device), dim=-1
    ).to(torch.bfloat16)
    v = (0.2 * torch.randn(
        batch, tokens, heads, value_dim, device=device
    )).to(torch.bfloat16)
    gk = (-0.01 * torch.rand(
        batch, tokens, heads, key_dim, device=device
    )).to(torch.bfloat16)
    gv = (-0.004 * torch.rand(
        batch, tokens, heads, value_dim, device=device
    )).to(torch.bfloat16)
    b = torch.sigmoid(torch.randn_like(q.float())).to(torch.bfloat16)
    w = torch.sigmoid(torch.randn_like(v.float())).to(torch.bfloat16)
    initial_state = 0.05 * torch.randn(
        batch, heads, key_dim, value_dim, device=device, dtype=torch.float32
    )
    cumulative_log = gv.float().cumsum(dim=1)
    scale = cumulative_log.exp()
    transformed_value = (v.float() * (-cumulative_log).exp()).to(torch.bfloat16)
    transformed_output, transformed_state = study.chunk_gdn2(
        q=q,
        k=k,
        v=transformed_value,
        g=gk,
        b=b,
        w=w,
        scale=1.0,
        initial_state=initial_state,
        output_final_state=True,
        use_qk_l2norm_in_kernel=False,
    )
    moving_output = transformed_output.float() * scale
    moving_state = transformed_state.float() * scale[:, -1].unsqueeze(-2)
    direct_output, direct_state = direct_bi_axis_reference(
        q, k, v, gk, gv, b, w, initial_state
    )
    output_error = (moving_output - direct_output).abs()
    state_error = (moving_state - direct_state).abs()
    output_max = float(output_error.max().item())
    state_max = float(state_error.max().item())
    output_relative = float(
        output_error.square().mean().sqrt().item()
        / direct_output.square().mean().sqrt().clamp_min(1e-6).item()
    )
    state_relative = float(
        state_error.square().mean().sqrt().item()
        / direct_state.square().mean().sqrt().clamp_min(1e-6).item()
    )
    if output_max > 0.08 or state_max > 0.08:
        raise AssertionError(
            f"moving-frame max error too large: output={output_max} state={state_max}"
        )
    if output_relative > 0.03 or state_relative > 0.03:
        raise AssertionError(
            "moving-frame relative error too large: "
            f"output={output_relative} state={state_relative}"
        )
    return {
        "output_max_abs_error": output_max,
        "state_max_abs_error": state_max,
        "output_relative_rms_error": output_relative,
        "state_relative_rms_error": state_relative,
    }


def check_open_transition_and_geometry(device: torch.device) -> dict[str, Any]:
    torch.manual_seed(15011)
    model = build(device, update_mode="bi_axis_value_decay")
    model.eval()
    x = torch.randn(2, TOKENS, MODEL_DIM, device=device)
    address = torch.randn_like(x)
    cell_order = torch.randperm(TOKENS, device=device)
    with torch.no_grad():
        zero_output, _zero_diag, zero_states = capture_terminal_states(
            model, x, address, cell_order
        )
        generator = torch.Generator(device=device).manual_seed(15013)
        for block in model.blocks:
            projection = block.time_mix.bi_axis_value_decay_proj
            if projection is None:
                raise AssertionError("Bi-Axis projection is missing")
            projection.weight.copy_(
                0.001
                * torch.randn(
                    projection.weight.shape,
                    generator=generator,
                    device=device,
                )
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
            f"opened Bi-Axis path did not change output/state: {output_change}"
        )
    decay_abs = float(
        open_diag["gdn3_bi_axis_value_decay_log_decay_abs"].item()
    )
    scale_min = float(
        open_diag["gdn3_bi_axis_value_decay_cumulative_scale_min"].item()
    )
    inverse_max = float(
        open_diag["gdn3_bi_axis_value_decay_inverse_scale_max"].item()
    )
    variations = {
        key: float(open_diag[key].item())
        for key in (
            "gdn3_bi_axis_value_decay_group_std",
            "gdn3_bi_axis_value_decay_batch_std",
            "gdn3_bi_axis_value_decay_token_std",
        )
    }
    if decay_abs <= 0.0 or min(variations.values()) <= 0.0:
        raise AssertionError(
            f"opened decay lacks activation/variation: {decay_abs}, {variations}"
        )
    if not 0.25 <= scale_min <= 1.0 or not 1.0 <= inverse_max <= 4.0:
        raise AssertionError(
            f"opened moving frame escaped preregistered bounds: {scale_min}, {inverse_max}"
        )
    terminal_rms = float(
        open_diag["gdn3_bi_axis_value_decay_terminal_rms"].item()
    )
    zero_terminal_rms = torch.stack(
        [state.float().square().mean().sqrt() for state in zero_states]
    ).mean()
    if not torch.isfinite(torch.tensor(terminal_rms)):
        raise AssertionError("opened terminal RMS is non-finite")
    if terminal_rms > 4.0 * float(zero_terminal_rms.item()):
        raise AssertionError(
            f"opened terminal RMS exceeded 4x parent: {terminal_rms}"
        )

    projection = model.blocks[0].time_mix.bi_axis_value_decay_proj
    assert projection is not None
    weights = projection.weight.float().view(
        HEADS, VALUE_GROUPS, MODEL_DIM
    )
    permutation = torch.randperm(VALUE_GROUPS, device=device)
    raw = F.linear(x.float(), weights.reshape(-1, MODEL_DIM)).view(
        2, TOKENS, HEADS, VALUE_GROUPS
    )
    permuted_raw = F.linear(
        x.float(),
        weights.index_select(1, permutation).reshape(-1, MODEL_DIM),
    ).view(2, TOKENS, HEADS, VALUE_GROUPS)
    equivariance_error = float(
        (
            permuted_raw
            - raw.index_select(-1, permutation)
        ).abs().max().item()
    )
    if equivariance_error > 3e-6:
        raise AssertionError(
            f"V-group permutation equivariance error {equivariance_error}"
        )
    softplus_zero = F.softplus(raw.new_zeros(()))
    group_decay = torch.clamp(
        softplus_zero - F.softplus(raw), max=0.0
    )
    changed_x = x.clone()
    changed_x[:, -1] += 1.0
    changed_raw = F.linear(
        changed_x.float(), projection.weight.float()
    ).view(2, TOKENS, HEADS, VALUE_GROUPS)
    changed_decay = torch.clamp(
        softplus_zero - F.softplus(changed_raw), max=0.0
    )
    causal_error = float(
        (
            group_decay.cumsum(dim=1)[:, :-1]
            - changed_decay.cumsum(dim=1)[:, :-1]
        ).abs().max().item()
    )
    if causal_error != 0.0 or float(group_decay.max().item()) > 0.0:
        raise AssertionError(
            f"causality/nonpositive decay failed: {causal_error}, {group_decay.max()}"
        )
    return {
        "output_max_abs_change": output_change,
        "terminal_state_max_abs_changes": state_changes,
        "decay_abs": decay_abs,
        "variations": variations,
        "cumulative_scale_min": scale_min,
        "inverse_scale_max": inverse_max,
        "terminal_rms": terminal_rms,
        "group_permutation_max_abs_error": equivariance_error,
        "causality_max_abs_error": causal_error,
        "opened_diag": {
            key: float(value.detach().float().item())
            for key, value in open_diag.items()
            if key.startswith("gdn3_bi_axis_value_decay_")
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
        "moving_frame_reference": check_moving_frame_reference(device),
        "open_transition_and_geometry": (
            check_open_transition_and_geometry(device)
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
    if len(compute_apps) != 1 or not compute_apps[0].startswith(EXPECTED_GPU_UUID):
        raise RuntimeError(f"unexpected concurrent compute applications: {compute_apps}")
    output["compute_apps"] = compute_apps
    print(json.dumps(output, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
