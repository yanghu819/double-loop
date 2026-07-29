#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import importlib.metadata
import json
import os
import statistics
import subprocess
import time
import traceback
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
torch = None
FLACache = None
chunk_gdn2 = None
fused_recurrent_gdn2 = None
naive_recurrent_gdn2 = None
collect_provenance = None
fla_l2norm_fp32 = None
project_erase_gate = None
FLADeltaTimeMix = None


def load_cuda_dependencies() -> None:
    global torch
    global FLACache
    global chunk_gdn2
    global fused_recurrent_gdn2
    global naive_recurrent_gdn2
    global collect_provenance
    global fla_l2norm_fp32
    global project_erase_gate
    global FLADeltaTimeMix

    import torch as torch_module
    from fla.models.utils import Cache
    from fla.ops.gdn2 import (
        chunk_gdn2 as chunk_operation,
        fused_recurrent_gdn2 as fused_operation,
        naive_recurrent_gdn2 as naive_operation,
    )
    from check_fla_delta_backbones import (
        collect_provenance as provenance_collector,
    )
    from gain_budget_gdn2 import (
        fla_l2norm_fp32 as l2norm_operation,
        project_erase_gate as projection_operation,
    )
    from study_rwkv_futureseed_loop import (
        FLADeltaTimeMix as delta_time_mix,
    )

    torch = torch_module
    FLACache = Cache
    chunk_gdn2 = chunk_operation
    fused_recurrent_gdn2 = fused_operation
    naive_recurrent_gdn2 = naive_operation
    collect_provenance = provenance_collector
    fla_l2norm_fp32 = l2norm_operation
    project_erase_gate = projection_operation
    FLADeltaTimeMix = delta_time_mix


def validate_process_environment() -> None:
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
                f"{cache_var} must point below /huyang2/double-loop"
            )
    if os.environ.get("FLA_DISABLE_BACKEND_DISPATCH") != "1":
        raise RuntimeError("FLA_DISABLE_BACKEND_DISPATCH=1 is required")
    if os.environ.get("FLA_CONV_BACKEND") != "triton":
        raise RuntimeError("FLA_CONV_BACKEND=triton is required")


def query_visible_gpu_uuid() -> str:
    output = subprocess.check_output(
        [
            "nvidia-smi",
            "-i",
            "0",
            "--query-gpu=uuid",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    )
    rows = [row.strip() for row in output.splitlines() if row.strip()]
    if len(rows) != 1:
        raise RuntimeError(
            f"Expected exactly one visible physical GPU UUID, got {rows}"
        )
    return rows[0]


def max_abs(left: torch.Tensor, right: torch.Tensor) -> float:
    return float((left.float() - right.float()).abs().max().item())


def relative_l2(left: torch.Tensor, right: torch.Tensor) -> float:
    difference = (left.float() - right.float()).norm()
    denominator = right.float().norm().clamp_min(1e-8)
    return float((difference / denominator).item())


def max_rel(left: torch.Tensor, right: torch.Tensor, floor: float = 1e-4) -> float:
    denominator = right.float().abs().clamp_min(floor)
    return float(((left.float() - right.float()).abs() / denominator).max().item())


def assert_close(
    actual: torch.Tensor,
    expected: torch.Tensor,
    *,
    atol: float,
    rtol: float,
    label: str,
) -> dict[str, float]:
    torch.testing.assert_close(actual.float(), expected.float(), atol=atol, rtol=rtol)
    return {
        f"{label}_max_abs": max_abs(actual, expected),
        f"{label}_max_rel": max_rel(actual, expected),
    }


def make_inputs(
    *,
    tokens: int,
    batch: int = 1,
    heads: int = 2,
    key_dim: int = 32,
    value_dim: int = 32,
    seed: int,
    value_dtype: torch.dtype | None = None,
    requires_grad: bool = False,
) -> dict[str, torch.Tensor]:
    if value_dtype is None:
        value_dtype = torch.bfloat16
    generator = torch.Generator(device="cuda")
    generator.manual_seed(seed)

    def random(shape: tuple[int, ...], dtype: torch.dtype) -> torch.Tensor:
        tensor = torch.randn(
            shape,
            generator=generator,
            device="cuda",
            dtype=dtype,
        )
        return tensor.requires_grad_(requires_grad)

    q_raw = random((batch, tokens, heads, key_dim), torch.float32)
    k_raw = random((batch, tokens, heads, key_dim), torch.float32)
    v = random((batch, tokens, heads, value_dim), value_dtype)
    raw_g = random((batch, tokens, heads, key_dim), torch.float32)
    raw_b = random((batch, tokens, heads, key_dim), torch.float32)
    raw_w = random((batch, tokens, heads, value_dim), value_dtype)
    h0 = random((batch, heads, key_dim, value_dim), torch.float32) * 0.05
    if requires_grad:
        h0 = h0.detach().requires_grad_(True)
    return {
        "q_raw": q_raw,
        "k_raw": k_raw,
        "v": v,
        "raw_g": raw_g,
        "raw_b": raw_b,
        "raw_w": raw_w,
        "h0": h0,
    }


def prepare_projected(
    inputs: dict[str, torch.Tensor],
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    q = fla_l2norm_fp32(inputs["q_raw"])
    k = fla_l2norm_fp32(inputs["k_raw"])
    # Keep decay close enough to identity that anisotropic erase is the active
    # part of the deterministic contract fixture.
    g = -0.003 - 0.002 * inputs["raw_g"].sigmoid()
    b = (2.0 * inputs["raw_b"]).sigmoid()
    # This helper is used only by the strict FP32 fused-forward lane and the
    # small-shape FP32 chunk diagnostic, never by formal BF16 training.
    v = inputs["v"].float()
    w = inputs["raw_w"].sigmoid().float()
    b_effective, stats = project_erase_gate(
        k,
        b,
        g,
        mode="decay_funded",
        step_gain_cap=1.0,
        sigma_cap_max=3.0,
        infeasible_policy="raise",
    )
    return {
        "q": q,
        "k": k,
        "v": v,
        "g": g,
        "b": b_effective,
        "w": w,
        "initial_state": inputs["h0"],
    }, stats


def prepare_projected_bfloat16(
    inputs: dict[str, torch.Tensor],
) -> tuple[
    dict[str, torch.Tensor],
    dict[str, torch.Tensor],
    dict[str, torch.Tensor],
]:
    """Build the exact low-precision lane consumed by official chunk GDN2."""
    q = fla_l2norm_fp32(inputs["q_raw"]).to(torch.bfloat16)
    k = fla_l2norm_fp32(inputs["k_raw"]).to(torch.bfloat16)
    g = -0.003 - 0.002 * inputs["raw_g"].sigmoid()
    b = (2.0 * inputs["raw_b"]).sigmoid().to(torch.bfloat16)
    v = inputs["v"].to(torch.bfloat16)
    w = inputs["raw_w"].sigmoid().to(torch.bfloat16)
    effective_b_fp32, projection = project_erase_gate(
        k.float(),
        b.float(),
        g.float(),
        mode="decay_funded",
        step_gain_cap=1.0,
        sigma_cap_max=3.0,
        infeasible_policy="raise",
    )
    effective_b = effective_b_fp32.to(torch.bfloat16)
    _audited_b, actual = project_erase_gate(
        k.float(),
        effective_b.float(),
        g.float(),
        mode="none",
        infeasible_policy="raise",
    )
    quantization = {
        "delta_abs_error": (
            actual["delta"] - projection["delta"]
        ).abs(),
        "effective_sigma": actual["effective_sigma"],
        "effective_step_gain_bound": actual[
            "effective_step_gain_bound"
        ],
    }
    return {
        "q": q,
        "k": k,
        "v": v,
        "g": g.float(),
        "b": effective_b,
        "w": w,
        "initial_state": inputs["h0"],
    }, projection, quantization


def independent_step_sigma_max(
    projected: dict[str, torch.Tensor],
    *,
    block_size: int = 64,
) -> float:
    key = projected["k"].detach().reshape(-1, projected["k"].shape[-1])
    gate = projected["b"].detach().reshape(-1, projected["b"].shape[-1])
    decay = projected["g"].detach().reshape(-1, projected["g"].shape[-1])
    identity = torch.eye(
        key.shape[-1],
        device=key.device,
        dtype=torch.float64,
    )
    maximum = 0.0
    for start in range(0, key.shape[0], block_size):
        stop = min(start + block_size, key.shape[0])
        key_block = key[start:stop].double()
        gate_block = gate[start:stop].double()
        decay_block = decay[start:stop].double()
        transition = identity.unsqueeze(0) - key_block.unsqueeze(-1) * (
            gate_block * key_block
        ).unsqueeze(-2)
        diagonal_decay = torch.diag_embed(decay_block.exp())
        sigma = torch.linalg.svdvals(transition @ diagonal_decay)[..., 0]
        maximum = max(maximum, float(sigma.max().item()))
    return maximum


def check_chunk_boundaries() -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    cases = (
        (1, 2, 1),
        (63, 2, 1),
        (64, 2, 1),
        (65, 2, 1),
        (81, 2, 1),
        (128, 2, 1),
        (81, 6, 2),
    )
    for tokens, heads, batch in cases:
        inputs = make_inputs(
            tokens=tokens,
            heads=heads,
            batch=batch,
            seed=20260729 + tokens + heads * 100,
        )
        projected, stats, quantization = prepare_projected_bfloat16(inputs)
        actual_o, actual_state = chunk_gdn2(
            **projected,
            output_final_state=True,
            use_qk_l2norm_in_kernel=False,
        )
        expected_o, expected_state = naive_recurrent_gdn2(
            **projected,
            output_final_state=True,
        )
        row: dict[str, Any] = {
            "tokens": tokens,
            "heads": heads,
            "batch": batch,
            "lane": (
                "official_bfloat16_chunk_production_shape"
                if heads == 6
                else "official_bfloat16_chunk_boundary"
            ),
            "kernel_input_dtypes": {
                name: str(projected[name].dtype)
                for name in ("q", "k", "v", "g", "b", "w", "initial_state")
            },
            "clipped_frac": float(stats["clipped"].float().mean().item()),
            "fp32_projection_delta_error_max": float(
                stats["delta_abs_error"].max().item()
            ),
            "quantized_delta_error_max": float(
                quantization["delta_abs_error"].max().item()
            ),
            "effective_step_bound_max": float(
                quantization["effective_step_gain_bound"].max().item()
            ),
            "independent_actual_step_sigma_max": independent_step_sigma_max(
                projected
            ),
        }
        row.update(
            assert_close(
                actual_o,
                expected_o,
                atol=3e-2,
                rtol=3e-2,
                label="output",
            )
        )
        row.update(
            assert_close(
                actual_state,
                expected_state,
                atol=4e-2,
                rtol=4e-2,
                label="state",
            )
        )
        if row["clipped_frac"] <= 0:
            raise AssertionError(f"projection did not activate at T={tokens}")
        if row["fp32_projection_delta_error_max"] > 5e-6:
            raise AssertionError(f"erase-strength preservation failed at T={tokens}")
        if row["quantized_delta_error_max"] > 3e-3:
            raise AssertionError(
                f"BF16 erase-strength tolerance failed at T={tokens}"
            )
        if row["effective_step_bound_max"] > 1.001:
            raise AssertionError(f"BF16 gain-budget tolerance failed at T={tokens}")
        if row["independent_actual_step_sigma_max"] > 1.001:
            raise AssertionError(
                f"independent BF16 P@D SVD tolerance failed at T={tokens}"
            )
        rows.append(row)
    return {"rows": rows}


def check_fused_forward() -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for tokens, heads in ((1, 2), (63, 2), (64, 2), (81, 6)):
        inputs = make_inputs(
            tokens=tokens,
            heads=heads,
            seed=20260829 + tokens + heads * 100,
            value_dtype=torch.float32,
        )
        projected, _stats = prepare_projected(inputs)
        actual_o, actual_state = fused_recurrent_gdn2(
            **projected,
            output_final_state=True,
            use_qk_l2norm_in_kernel=False,
        )
        expected_o, expected_state = naive_recurrent_gdn2(
            **projected,
            output_final_state=True,
        )
        row: dict[str, Any] = {
            "tokens": tokens,
            "heads": heads,
            "lane": "strict_fp32_fused_recurrent_forward_certificate",
            "backward": "not_supported_by_upstream_fused_recurrent_gdn2",
            "independent_actual_step_sigma_max": independent_step_sigma_max(
                projected
            ),
        }
        row.update(
            assert_close(
                actual_o,
                expected_o,
                atol=4e-3,
                rtol=4e-3,
                label="output",
            )
        )
        row.update(
            assert_close(
                actual_state,
                expected_state,
                atol=4e-3,
                rtol=4e-3,
                label="state",
            )
        )
        if row["independent_actual_step_sigma_max"] > 1.0001:
            raise AssertionError(
                "strict FP32 fused-recurrent P@D SVD certificate failed"
            )
        rows.append(row)
    return {"rows": rows}


def _run_kernel_vjp(
    projected: dict[str, torch.Tensor],
    *,
    kernel: str,
    output_cotangent: torch.Tensor,
    state_cotangent: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    leaves = {
        name: value.detach().clone().requires_grad_(True)
        for name, value in projected.items()
    }
    operation = chunk_gdn2 if kernel == "chunk" else naive_recurrent_gdn2
    output, final_state = operation(
        **leaves,
        output_final_state=True,
        **({"use_qk_l2norm_in_kernel": False} if kernel == "chunk" else {}),
    )
    loss = (
        output.float() * output_cotangent
    ).sum() + (final_state * state_cotangent).sum()
    gradient_values = torch.autograd.grad(loss, tuple(leaves.values()))
    gradients = {
        name: gradient.detach()
        for name, gradient in zip(leaves, gradient_values)
    }
    return output, final_state, gradients, leaves


def check_chunk_backward() -> dict[str, Any]:
    base = make_inputs(
        tokens=81,
        heads=2,
        key_dim=32,
        value_dim=32,
        seed=20260929,
        value_dtype=torch.float32,
        requires_grad=False,
    )
    projected, stats = prepare_projected(base)
    generator = torch.Generator(device="cuda")
    generator.manual_seed(20260930)
    output_cotangent = torch.randn(
        projected["v"].shape,
        generator=generator,
        device="cuda",
        dtype=torch.float32,
    )
    state_cotangent = torch.randn(
        projected["initial_state"].shape,
        generator=generator,
        device="cuda",
        dtype=torch.float32,
    )
    chunk_o, chunk_state, chunk_grads, chunk_leaves = _run_kernel_vjp(
        projected,
        kernel="chunk",
        output_cotangent=output_cotangent,
        state_cotangent=state_cotangent,
    )
    naive_o, naive_state, naive_grads, _naive_leaves = _run_kernel_vjp(
        projected,
        kernel="naive",
        output_cotangent=output_cotangent,
        state_cotangent=state_cotangent,
    )
    result: dict[str, Any] = {
        "autograd_fn": type(chunk_o.grad_fn).__name__,
        "clipped_frac": float(stats["clipped"].float().mean().item()),
        "gradients": {},
    }
    if "ChunkGDN2FunctionBackward" not in result["autograd_fn"]:
        raise AssertionError(
            f"unexpected chunk autograd path: {result['autograd_fn']}"
        )
    result.update(
        assert_close(
            chunk_o,
            naive_o,
            atol=2e-3,
            rtol=2e-3,
            label="output",
        )
    )
    result.update(
        assert_close(
            chunk_state,
            naive_state,
            atol=3e-3,
            rtol=3e-3,
            label="state",
        )
    )
    expected_gradient_names = {
        "q",
        "k",
        "v",
        "g",
        "b",
        "w",
        "initial_state",
    }
    if set(chunk_grads) != expected_gradient_names:
        raise AssertionError(
            f"missing chunk gradients: {expected_gradient_names - set(chunk_grads)}"
        )
    if set(naive_grads) != expected_gradient_names:
        raise AssertionError(
            f"missing oracle gradients: {expected_gradient_names - set(naive_grads)}"
        )
    for name in sorted(expected_gradient_names):
        if not bool(torch.isfinite(chunk_grads[name]).all()):
            raise AssertionError(f"non-finite chunk gradient for {name}")
        result["gradients"][name] = assert_close(
            chunk_grads[name],
            naive_grads[name],
            atol=2e-2,
            rtol=8e-2,
            label=name,
        )
        result["gradients"][name]["chunk_norm"] = float(
            chunk_grads[name].float().norm().item()
        )
        if result["gradients"][name]["chunk_norm"] <= 0:
            raise AssertionError(f"zero chunk VJP for {name}")
    result["kernel_input_dtypes"] = {
        name: str(value.dtype) for name, value in chunk_leaves.items()
    }
    result["independent_actual_step_sigma_max"] = independent_step_sigma_max(
        projected,
    )
    return result


def check_chunk_backward_bfloat16() -> dict[str, Any]:
    base = make_inputs(
        tokens=81,
        batch=2,
        heads=6,
        key_dim=32,
        value_dim=32,
        seed=20260931,
        value_dtype=torch.bfloat16,
        requires_grad=False,
    )
    projected, stats, quantization = prepare_projected_bfloat16(base)
    generator = torch.Generator(device="cuda")
    generator.manual_seed(20260932)
    output_cotangent = torch.randn(
        projected["v"].shape,
        generator=generator,
        device="cuda",
        dtype=torch.float32,
    )
    state_cotangent = torch.randn(
        projected["initial_state"].shape,
        generator=generator,
        device="cuda",
        dtype=torch.float32,
    )
    chunk_o, chunk_state, chunk_grads, chunk_leaves = _run_kernel_vjp(
        projected,
        kernel="chunk",
        output_cotangent=output_cotangent,
        state_cotangent=state_cotangent,
    )
    naive_o, naive_state, naive_grads, _naive_leaves = _run_kernel_vjp(
        projected,
        kernel="naive",
        output_cotangent=output_cotangent,
        state_cotangent=state_cotangent,
    )
    result: dict[str, Any] = {
        "lane": "official_bfloat16_chunk_training_tolerance",
        "autograd_fn": type(chunk_o.grad_fn).__name__,
        "clipped_frac": float(stats["clipped"].float().mean().item()),
        "quantized_delta_error_max": float(
            quantization["delta_abs_error"].max().item()
        ),
        "quantized_step_bound_max": float(
            quantization["effective_step_gain_bound"].max().item()
        ),
        "independent_actual_step_sigma_max": independent_step_sigma_max(
            projected
        ),
        "kernel_input_dtypes": {
            name: str(value.dtype) for name, value in chunk_leaves.items()
        },
        "gradients": {},
    }
    result.update(
        assert_close(
            chunk_o,
            naive_o,
            atol=4e-2,
            rtol=4e-2,
            label="output",
        )
    )
    result.update(
        assert_close(
            chunk_state,
            naive_state,
            atol=5e-2,
            rtol=5e-2,
            label="state",
        )
    )
    for name in sorted(chunk_grads):
        gradient = chunk_grads[name]
        if not bool(torch.isfinite(gradient).all()):
            raise AssertionError(
                f"non-finite official BF16 chunk gradient for {name}"
            )
        norm = float(gradient.float().norm().item())
        if norm <= 0:
            raise AssertionError(
                f"zero official BF16 chunk gradient for {name}"
            )
        result["gradients"][name] = assert_close(
            gradient,
            naive_grads[name],
            atol=6e-2,
            rtol=1.5e-1,
            label=name,
        )
        relative_error = relative_l2(gradient, naive_grads[name])
        result["gradients"][name]["chunk_norm"] = norm
        result["gradients"][name]["relative_l2"] = relative_error
        if relative_error > 0.20:
            raise AssertionError(
                "Official BF16 chunk VJP relative L2 exceeds 20% for "
                f"{name}: {relative_error}"
            )
    if result["quantized_delta_error_max"] > 3e-3:
        raise AssertionError("BF16 quantized erase-strength error exceeds tolerance")
    if result["quantized_step_bound_max"] > 1.001:
        raise AssertionError("BF16 quantized gain bound exceeds tolerance")
    if result["independent_actual_step_sigma_max"] > 1.001:
        raise AssertionError("independent BF16 P@D SVD exceeds tolerance")
    return result


def check_chunk_state_carry() -> dict[str, Any]:
    rows = []
    for lane, value_dtype, heads, batch, atol, rtol, seed in (
        ("strict_fp32_chunk_small_shape", torch.float32, 2, 1, 6e-3, 6e-3, 20260940),
        (
            "official_bfloat16_chunk_production_shape",
            torch.bfloat16,
            6,
            2,
            5e-2,
            5e-2,
            20260941,
        ),
    ):
        inputs = make_inputs(
            tokens=81,
            heads=heads,
            batch=batch,
            key_dim=32,
            value_dim=32,
            seed=seed,
            value_dtype=value_dtype,
        )
        if value_dtype == torch.bfloat16:
            projected, _stats, _quantization = prepare_projected_bfloat16(
                inputs
            )
        else:
            projected, _stats = prepare_projected(inputs)
        full_output, full_state = chunk_gdn2(
            **projected,
            output_final_state=True,
            use_qk_l2norm_in_kernel=False,
        )
        split = 37
        first = {
            name: value if name == "initial_state" else value[:, :split]
            for name, value in projected.items()
        }
        first_output, first_state = chunk_gdn2(
            **first,
            output_final_state=True,
            use_qk_l2norm_in_kernel=False,
        )
        second = {
            name: first_state if name == "initial_state" else value[:, split:]
            for name, value in projected.items()
        }
        second_output, second_state = chunk_gdn2(
            **second,
            output_final_state=True,
            use_qk_l2norm_in_kernel=False,
        )
        segmented_output = torch.cat((first_output, second_output), dim=1)
        row = {
            "lane": lane,
            "tokens": 81,
            "split": split,
        }
        row.update(
            assert_close(
                segmented_output,
                full_output,
                atol=atol,
                rtol=rtol,
                label="output",
            )
        )
        row.update(
            assert_close(
                second_state,
                full_state,
                atol=atol,
                rtol=rtol,
                label="state",
            )
        )
        rows.append(row)
    return {"rows": rows}


def check_mode_none_exact() -> dict[str, Any]:
    torch.manual_seed(20261029)
    layer = FLADeltaTimeMix(
        d_model=64,
        heads=2,
        head_dim=32,
        backbone="gdn2",
        expand_v=1.0,
        mode="chunk",
        use_short_conv=True,
        conv_size=4,
        allow_neg_eigval=False,
        gain_budget_mode="none",
    ).cuda().to(torch.bfloat16)
    layer.train()
    x = torch.randn(
        2,
        81,
        64,
        device="cuda",
        dtype=torch.bfloat16,
        requires_grad=True,
    )
    initial_state = (
        torch.randn(
            2,
            2,
            32,
            32,
            device="cuda",
            dtype=torch.float32,
        )
        * 0.05
    ).requires_grad_(True)
    calls = {"count": 0}

    def hook(_module, _inputs, _output):
        calls["count"] += 1

    handle = layer.core.register_forward_hook(hook)
    wrapped_o, wrapped_state = layer(x, initial_state=initial_state)
    handle.remove()
    if calls["count"] != 1:
        raise AssertionError("mode=none did not call the untouched official layer exactly once")

    cache = FLACache()
    cache.update(
        recurrent_state=initial_state,
        conv_state=layer._zero_conv_state(x),
        layer_idx=0,
        offset=0,
    )
    direct_o, _attn, direct_cache = layer.core(
        x,
        past_key_values=cache,
        use_cache=True,
    )
    direct_state = direct_cache[0]["recurrent_state"]
    if not torch.equal(wrapped_o, direct_o):
        raise AssertionError("mode=none output is not bitwise equal to official GDN2")
    if not torch.equal(wrapped_state, direct_state):
        raise AssertionError("mode=none state is not bitwise equal to official GDN2")

    original_mode = layer.gain_budget_mode
    layer.gain_budget_mode = "external_identity"
    external_o, external_state = layer._forward_gain_budget(
        x,
        initial_state=initial_state,
    )
    layer.gain_budget_mode = original_mode
    external_output_max_abs = max_abs(external_o, direct_o)
    external_output_rms = float(
        (external_o.float() - direct_o.float()).square().mean().sqrt().item()
    )
    external_state_max_abs = max_abs(external_state, direct_state)
    external_state_rms = float(
        (external_state.float() - direct_state.float()).square().mean().sqrt().item()
    )
    external_output_relative_l2 = relative_l2(external_o, direct_o)
    external_state_relative_l2 = relative_l2(external_state, direct_state)
    if (
        external_output_rms > 5e-3
        or external_state_rms > 5e-3
        or external_output_max_abs > 5e-2
        or external_state_max_abs > 1e-1
        or external_output_relative_l2 > 2e-2
        or external_state_relative_l2 > 2e-2
    ):
        raise AssertionError(
            "external FP32 normalization plus BF16 recurrence changes the "
            "official no-projection path too much: "
            f"output_rms={external_output_rms}, "
            f"output_max_abs={external_output_max_abs}, "
            f"output_relative_l2={external_output_relative_l2}, "
            f"state_rms={external_state_rms}, "
            f"state_max_abs={external_state_max_abs}, "
            f"state_relative_l2={external_state_relative_l2}"
        )

    generator = torch.Generator(device="cuda")
    generator.manual_seed(20261030)
    output_cotangent = torch.randn(
        wrapped_o.shape,
        generator=generator,
        device="cuda",
        dtype=torch.float32,
    )
    state_cotangent = torch.randn(
        wrapped_state.shape,
        generator=generator,
        device="cuda",
        dtype=torch.float32,
    )

    parameter_rows = tuple(layer.named_parameters())
    parameter_names = tuple(name for name, _parameter in parameter_rows)
    parameters = tuple(parameter for _name, parameter in parameter_rows)

    def official_vjp() -> tuple[torch.Tensor, ...]:
        x_leaf = x.detach().clone().requires_grad_(True)
        state_leaf = initial_state.detach().clone().requires_grad_(True)
        output, state = layer(x_leaf, initial_state=state_leaf)
        loss = (
            output.float() * output_cotangent
        ).sum() + (state * state_cotangent).sum()
        return torch.autograd.grad(
            loss,
            (x_leaf, state_leaf, *parameters),
        )

    def external_vjp() -> tuple[torch.Tensor, ...]:
        x_leaf = x.detach().clone().requires_grad_(True)
        state_leaf = initial_state.detach().clone().requires_grad_(True)
        original = layer.gain_budget_mode
        layer.gain_budget_mode = "external_identity"
        output, state = layer._forward_gain_budget(
            x_leaf,
            initial_state=state_leaf,
        )
        layer.gain_budget_mode = original
        loss = (
            output.float() * output_cotangent
        ).sum() + (state * state_cotangent).sum()
        return torch.autograd.grad(
            loss,
            (x_leaf, state_leaf, *parameters),
        )

    official_gradients = official_vjp()
    external_gradients = external_vjp()
    external_vjp_rows = {}
    gradient_names = ("input", "initial_state", *parameter_names)
    for name, external_gradient, official_gradient in zip(
        gradient_names,
        external_gradients,
        official_gradients,
    ):
        external_vjp_rows[name] = assert_close(
            external_gradient,
            official_gradient,
            atol=4e-2,
            rtol=8e-2,
            label=name,
        )
        external_vjp_rows[name]["relative_l2"] = relative_l2(
            external_gradient,
            official_gradient,
        )

    budget_layer = FLADeltaTimeMix(
        d_model=64,
        heads=2,
        head_dim=32,
        backbone="gdn2",
        expand_v=1.0,
        mode="chunk",
        use_short_conv=True,
        conv_size=4,
        allow_neg_eigval=False,
        gain_budget_mode="decay_funded",
    ).cuda().to(torch.bfloat16)
    budget_layer.load_state_dict(layer.state_dict(), strict=True)
    if set(layer.state_dict()) != set(budget_layer.state_dict()):
        raise AssertionError("gain-budget mode changed checkpoint parameter keys")
    return {
        "official_forward_calls": calls["count"],
        "output_bitwise_equal": True,
        "state_bitwise_equal": True,
        "external_no_projection_output_max_abs": external_output_max_abs,
        "external_no_projection_output_rms": external_output_rms,
        "external_no_projection_output_relative_l2": external_output_relative_l2,
        "external_no_projection_state_max_abs": external_state_max_abs,
        "external_no_projection_state_rms": external_state_rms,
        "external_no_projection_state_relative_l2": external_state_relative_l2,
        "external_no_projection_vjp": external_vjp_rows,
        "parameter_keys_equal": True,
        "parameter_count": sum(parameter.numel() for parameter in layer.parameters()),
    }


def check_budget_layer_backward() -> dict[str, Any]:
    torch.manual_seed(20261129)
    layer = FLADeltaTimeMix(
        d_model=64,
        heads=2,
        head_dim=32,
        backbone="gdn2",
        expand_v=1.0,
        mode="chunk",
        use_short_conv=True,
        conv_size=4,
        allow_neg_eigval=False,
        gain_budget_mode="decay_funded",
    ).cuda().to(torch.bfloat16)
    layer.train()
    x = torch.randn(
        2,
        81,
        64,
        device="cuda",
        dtype=torch.bfloat16,
        requires_grad=True,
    )
    output, final_state = layer(x)
    loss = output.float().square().mean() + 0.01 * final_state.square().mean()
    loss.backward()
    required = (
        "core.q_proj.weight",
        "core.k_proj.weight",
        "core.v_proj.weight",
        "core.f_proj.0.weight",
        "core.f_proj.1.weight",
        "core.b_proj.weight",
        "core.w_proj.weight",
        "core.q_conv1d.weight",
        "core.k_conv1d.weight",
        "core.v_conv1d.weight",
        "core.A_log",
        "core.dt_bias",
    )
    gradient_norms: dict[str, float] = {}
    named_parameters = dict(layer.named_parameters())
    for name in required:
        gradient = named_parameters[name].grad
        if gradient is None or not bool(torch.isfinite(gradient).all()):
            raise AssertionError(f"missing or non-finite full-layer gradient for {name}")
        norm = float(gradient.float().norm().item())
        if norm <= 0:
            raise AssertionError(f"zero full-layer gradient for {name}")
        gradient_norms[name] = norm
    diagnostics = {
        name: float(value.float().item())
        for name, value in layer.last_gain_budget_diag.items()
    }
    if diagnostics["gdn2_gain_budget_clipped_frac"] <= 0:
        raise AssertionError("full layer did not activate gain-budget projection")
    if diagnostics["gdn2_gain_budget_delta_error_max"] > 3e-3:
        raise AssertionError(
            "full-layer BF16 erase-strength tolerance was exceeded"
        )
    if diagnostics["gdn2_gain_budget_effective_step_bound_max"] > 1.001:
        raise AssertionError("full-layer BF16 numerical tolerance was exceeded")
    if (
        diagnostics["gdn2_gain_budget_fp32_projection_delta_error_max"]
        > 5e-6
    ):
        raise AssertionError("full-layer FP32 projection failed its certificate")
    if (
        diagnostics["gdn2_gain_budget_fp32_projection_step_bound_max"]
        > 1.0001
    ):
        raise AssertionError("full-layer FP32 projection exceeded its bound")
    if diagnostics["gdn2_gain_budget_fp32_numerical_certificate"] != 0.0:
        raise AssertionError("BF16 chunk path was mislabeled as hard-certified")
    if diagnostics["gdn2_gain_budget_low_precision_tolerance_path"] != 1.0:
        raise AssertionError("BF16 chunk path did not declare tolerance mode")
    if x.grad is None or not bool(torch.isfinite(x.grad).all()):
        raise AssertionError("missing or non-finite full-layer input gradient")
    input_grad_norm = float(x.grad.float().norm().item())
    if input_grad_norm <= 0:
        raise AssertionError("zero full-layer input gradient")
    return {
        "output_dtype": str(output.dtype),
        "state_dtype": str(final_state.dtype),
        "input_grad_norm": input_grad_norm,
        "parameter_gradient_norms": gradient_norms,
        "diagnostics": diagnostics,
    }


def benchmark_layers(warmup: int, iterations: int) -> dict[str, Any]:
    torch.manual_seed(20261229)
    kwargs = dict(
        d_model=192,
        heads=6,
        head_dim=32,
        backbone="gdn2",
        expand_v=1.0,
        mode="chunk",
        use_short_conv=True,
        conv_size=4,
        allow_neg_eigval=False,
    )
    baseline = FLADeltaTimeMix(
        **kwargs,
        gain_budget_mode="none",
    ).cuda().to(torch.bfloat16)
    identity = FLADeltaTimeMix(
        **kwargs,
        gain_budget_mode="external_identity",
    ).cuda().to(torch.bfloat16)
    budget = FLADeltaTimeMix(
        **kwargs,
        gain_budget_mode="decay_funded",
    ).cuda().to(torch.bfloat16)
    identity.load_state_dict(baseline.state_dict(), strict=True)
    budget.load_state_dict(baseline.state_dict(), strict=True)
    baseline.train()
    identity.train()
    budget.train()
    x_template = torch.randn(32, 81, 192, device="cuda", dtype=torch.bfloat16)
    state_template = torch.randn(
        32,
        6,
        32,
        32,
        device="cuda",
        dtype=torch.float32,
    ) * 0.02

    def run(layer: FLADeltaTimeMix) -> None:
        x = x_template.detach().clone().requires_grad_(True)
        initial_state = state_template.detach().clone().requires_grad_(True)
        layer.zero_grad(set_to_none=True)
        output, state = layer(x, initial_state=initial_state)
        (output.float().square().mean() + 0.01 * state.square().mean()).backward()

    for layer in (baseline, identity, budget):
        for _ in range(warmup):
            run(layer)
        layer.zero_grad(set_to_none=True)
    torch.cuda.synchronize()

    def measure_once(layer: FLADeltaTimeMix) -> tuple[float, int]:
        layer.zero_grad(set_to_none=True)
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        allocated_before = int(torch.cuda.memory_allocated())
        started = torch.cuda.Event(enable_timing=True)
        finished = torch.cuda.Event(enable_timing=True)
        started.record()
        run(layer)
        finished.record()
        torch.cuda.synchronize()
        elapsed_seconds = float(started.elapsed_time(finished)) / 1000.0
        incremental_peak = int(torch.cuda.max_memory_allocated()) - allocated_before
        layer.zero_grad(set_to_none=True)
        return elapsed_seconds, incremental_peak

    samples = {"baseline": [], "identity": [], "budget": []}
    memory_samples = {"baseline": [], "identity": [], "budget": []}
    sequence = (
        "baseline",
        "identity",
        "budget",
        "budget",
        "identity",
        "baseline",
    )
    cursors = {"baseline": 0, "identity": 0, "budget": 0}
    layers = {
        "baseline": baseline,
        "identity": identity,
        "budget": budget,
    }
    while min(cursors.values()) < iterations:
        for name in sequence:
            if cursors[name] >= iterations:
                continue
            seconds, peak = measure_once(layers[name])
            samples[name].append(seconds)
            memory_samples[name].append(peak)
            cursors[name] += 1

    baseline_seconds = statistics.median(samples["baseline"])
    identity_seconds = statistics.median(samples["identity"])
    budget_seconds = statistics.median(samples["budget"])
    baseline_memory = int(statistics.median(memory_samples["baseline"]))
    identity_memory = int(statistics.median(memory_samples["identity"]))
    budget_memory = int(statistics.median(memory_samples["budget"]))
    result = {
        "warmup": warmup,
        "iterations": iterations,
        "order": "ABCCBA",
        "baseline_seconds": baseline_seconds,
        "external_identity_seconds": identity_seconds,
        "budget_seconds": budget_seconds,
        "time_overhead_frac": budget_seconds / baseline_seconds - 1.0,
        "projection_time_overhead_frac": (
            budget_seconds / identity_seconds - 1.0
        ),
        "baseline_incremental_peak_bytes": baseline_memory,
        "external_identity_incremental_peak_bytes": identity_memory,
        "budget_incremental_peak_bytes": budget_memory,
        "memory_overhead_frac": budget_memory / baseline_memory - 1.0,
        "projection_memory_overhead_frac": (
            budget_memory / identity_memory - 1.0
        ),
        "baseline_seconds_samples": samples["baseline"],
        "external_identity_seconds_samples": samples["identity"],
        "budget_seconds_samples": samples["budget"],
        "baseline_peak_samples": memory_samples["baseline"],
        "external_identity_peak_samples": memory_samples["identity"],
        "budget_peak_samples": memory_samples["budget"],
        "budget_clipped_frac": float(
            budget.last_gain_budget_diag[
                "gdn2_gain_budget_clipped_frac"
            ].float().item()
        ),
    }
    if result["projection_time_overhead_frac"] > 0.20:
        raise AssertionError(
            "gain-budget projection time overhead exceeds 20% over the "
            f"matched external identity: {result['projection_time_overhead_frac']}"
        )
    if result["projection_memory_overhead_frac"] > 0.20:
        raise AssertionError(
            "gain-budget projection memory overhead exceeds 20% over the "
            "matched external identity: "
            f"{result['projection_memory_overhead_frac']}"
        )
    if result["budget_clipped_frac"] < 0.01:
        raise AssertionError(
            "gain-budget clipping fraction is below the 1% mechanism gate"
        )
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--wheel",
        type=Path,
        default=REPO_ROOT
        / (
            "wheelhouse/"
            "flash_linear_attention-0.5.2-9c8e42e-py3-none-any.whl"
        ),
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-gpu-uuid", required=True)
    parser.add_argument("--benchmark-warmup", type=int, default=3)
    parser.add_argument("--benchmark-iterations", type=int, default=8)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result: dict[str, Any] = {
        "status": "running",
        "expected_gpu_uuid": args.expected_gpu_uuid,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    try:
        validate_process_environment()
        load_cuda_dependencies()
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required; CPU fallback is forbidden")
        if os.environ.get("CUDA_VISIBLE_DEVICES") != "0":
            raise RuntimeError("CUDA_VISIBLE_DEVICES must be exactly '0' for GPU1")
        if torch.cuda.device_count() != 1 or torch.cuda.current_device() != 0:
            raise RuntimeError(
                "Expose only GPU1 as CUDA device 0 with CUDA_VISIBLE_DEVICES=0"
            )
        if torch.cuda.get_device_properties(0).total_memory < 70 * 1024**3:
            raise RuntimeError("Expected the 80 GB GPU1 row")
        gpu_uuid = query_visible_gpu_uuid()
        if gpu_uuid != args.expected_gpu_uuid:
            raise RuntimeError(
                f"GPU UUID mismatch: {gpu_uuid} != {args.expected_gpu_uuid}"
            )
        torch.manual_seed(20260729)
        torch.cuda.manual_seed_all(20260729)
        result.update(
            {
                "git_sha": subprocess.check_output(
                    ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"],
                    text=True,
                ).strip(),
                "cuda_visible_devices": os.environ.get(
                    "CUDA_VISIBLE_DEVICES"
                ),
                "device": torch.cuda.get_device_name(0),
                "gpu_uuid": gpu_uuid,
                "torch_version": torch.__version__,
                "triton_version": importlib.metadata.version("triton"),
            }
        )
        args.output.write_text(
            json.dumps(result, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        result["provenance"] = collect_provenance(args.wheel)
        checks = (
            ("chunk_boundary_forward", check_chunk_boundaries),
            ("fused_forward", check_fused_forward),
            ("chunk_backward", check_chunk_backward),
            ("chunk_backward_bfloat16", check_chunk_backward_bfloat16),
            ("chunk_state_carry", check_chunk_state_carry),
            ("mode_none_exact", check_mode_none_exact),
            ("budget_layer_backward", check_budget_layer_backward),
            (
                "benchmark",
                lambda: benchmark_layers(
                    args.benchmark_warmup,
                    args.benchmark_iterations,
                ),
            ),
        )
        for name, check in checks:
            started = time.perf_counter()
            result[name] = check()
            result[name]["elapsed_seconds"] = time.perf_counter() - started
            args.output.write_text(
                json.dumps(result, indent=2, sort_keys=True),
                encoding="utf-8",
            )
            print(
                f"PASS {name} ({result[name]['elapsed_seconds']:.2f}s)",
                flush=True,
            )

        result["status"] = "passed"
        args.output.write_text(
            json.dumps(result, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        print(json.dumps(result["benchmark"], indent=2, sort_keys=True))
    except Exception as error:
        result["status"] = "failed"
        result["error_type"] = type(error).__name__
        result["error"] = str(error)
        result["traceback"] = traceback.format_exc()
        args.output.write_text(
            json.dumps(result, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        raise


if __name__ == "__main__":
    main()
