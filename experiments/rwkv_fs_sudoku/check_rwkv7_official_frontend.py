#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Iterable

for cache_var in (
    "XDG_CACHE_HOME",
    "TRITON_CACHE_DIR",
    "TORCHINDUCTOR_CACHE_DIR",
    "TORCH_EXTENSIONS_DIR",
    "TMPDIR",
):
    cache_path = os.environ.get(cache_var, "")
    if not cache_path.startswith("/huyang2/double-loop/"):
        raise RuntimeError(f"{cache_var} must point below /huyang2/double-loop before importing CUDA runtimes")

import torch
import torch.nn.functional as F

from study_rwkv_futureseed_loop import (
    RWKV7_OFFICIAL_SOURCE_BLOB,
    RWKV7_OFFICIAL_SOURCE_COMMIT,
    RWKV7_OFFICIAL_SOURCE_PATH,
    RWKV7_OFFICIAL_KERNEL_BLOB,
    RWKV7_OFFICIAL_KERNEL_PATH,
    RWKV7_STATEPASSING_CUDA_SHA256,
    RWKV7TimeMixOfficial,
    build_adamw,
    statepassing_available,
)


EXPECTED_SOURCE_COMMIT = "952102498e9ed367ea0a59ee64106916d474d30f"
EXPECTED_SOURCE_BLOB = "b4d167fedead2655d253c55eb47b65f00e7193d2"
EXPECTED_SOURCE_PATH = "RWKV-v7/train_temp/src/model.py"
EXPECTED_KERNEL_BLOB = "827faeb06b9d2b6e31b3efe85af6d3ae4cf88905"
EXPECTED_KERNEL_PATH = "RWKV-v7/train_temp/cuda/rwkv7_clampw.cu"
EXPECTED_STATEPASSING_CUDA_SHA256 = "59a90a0521b1851da17c008c685f959d586af1a7d28056b29a7478ab92c1c892"


def max_abs(reference: torch.Tensor, actual: torch.Tensor) -> float:
    return float((reference.float() - actual.float()).abs().max().item())


def rms(reference: torch.Tensor, actual: torch.Tensor) -> float:
    return float((reference.float() - actual.float()).square().mean().sqrt().item())


def relative_rms(reference: torch.Tensor, actual: torch.Tensor) -> float:
    denominator = max(float(reference.float().square().mean().sqrt().item()), 1e-8)
    return rms(reference, actual) / denominator


def compare(reference: torch.Tensor, actual: torch.Tensor) -> dict[str, Any]:
    return {
        "reference_dtype": str(reference.dtype),
        "actual_dtype": str(actual.dtype),
        "max_abs": max_abs(reference, actual),
        "rms": rms(reference, actual),
        "relative_rms": relative_rms(reference, actual),
    }


def all_finite(tensors: Iterable[torch.Tensor | None]) -> bool:
    return all(tensor is not None and bool(torch.isfinite(tensor).all()) for tensor in tensors)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def randomize_open_paths(layer: RWKV7TimeMixOfficial) -> None:
    generator = torch.Generator(device=layer.output.weight.device)
    generator.manual_seed(20260724 + layer.layer_id)
    with torch.no_grad():
        for parameter in (layer.w1, layer.a1, layer.v1, layer.g1):
            parameter.copy_(torch.randn(parameter.shape, generator=generator, device=parameter.device) * 0.01)
        layer.output.weight.copy_(
            torch.randn(
                layer.output.weight.shape,
                generator=generator,
                device=layer.output.weight.device,
            )
            * 0.01
        )


def independent_recurrence(
    *,
    r: torch.Tensor,
    w_raw: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    kk: torch.Tensor,
    a: torch.Tensor,
    initial_state: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    state = initial_state.float()
    outputs: list[torch.Tensor] = []
    for step in range(r.shape[1]):
        decay = torch.exp(-math.exp(-0.5) * torch.sigmoid(w_raw[:, step].float()))
        state_erase = torch.einsum("bhij,bhj->bhi", state, -kk[:, step].float())
        state = (
            state * decay.unsqueeze(-2)
            + state_erase.unsqueeze(-1) * (kk[:, step].float() * a[:, step].float()).unsqueeze(-2)
            + v[:, step].float().unsqueeze(-1) * k[:, step].float().unsqueeze(-2)
        )
        outputs.append(torch.einsum("bhij,bhj->bhi", state, r[:, step].float()))
    return torch.stack(outputs, dim=1).to(r.dtype), state


def independent_frontend(
    layer: RWKV7TimeMixOfficial,
    x: torch.Tensor,
    initial_state: torch.Tensor,
    v_first: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    batch_size, seq_len, channels = x.shape
    shifted = torch.zeros_like(x)
    shifted[:, 1:] = x[:, :-1]
    delta = shifted - x
    xr = x + delta * layer.x_r
    xw = x + delta * layer.x_w
    xk = x + delta * layer.x_k
    xv = x + delta * layer.x_v
    xa = x + delta * layer.x_a
    xg = x + delta * layer.x_g

    r_flat = F.linear(xr, layer.receptance.weight)
    w_raw_flat = layer.w0 + torch.tanh(xw @ layer.w1) @ layer.w2
    k_flat = F.linear(xk, layer.key.weight)
    v_flat = F.linear(xv, layer.value.weight)
    v_flat = v_flat + (v_first - v_flat) * torch.sigmoid(layer.v0 + (xv @ layer.v1) @ layer.v2)
    a_flat = torch.sigmoid(layer.a0 + (xa @ layer.a1) @ layer.a2)
    gate = torch.sigmoid(xg @ layer.g1) @ layer.g2
    kk_flat = F.normalize(
        (k_flat * layer.k_k).view(batch_size, seq_len, layer.heads, layer.head_dim),
        dim=-1,
        p=2.0,
    ).view(batch_size, seq_len, channels)
    k_flat = k_flat * (1.0 + (a_flat - 1.0) * layer.k_a)
    recurrent_out, terminal_state = independent_recurrence(
        r=r_flat.view(batch_size, seq_len, layer.heads, layer.head_dim),
        w_raw=w_raw_flat.view(batch_size, seq_len, layer.heads, layer.head_dim),
        k=k_flat.view(batch_size, seq_len, layer.heads, layer.head_dim),
        v=v_flat.view(batch_size, seq_len, layer.heads, layer.head_dim),
        kk=kk_flat.view(batch_size, seq_len, layer.heads, layer.head_dim),
        a=a_flat.view(batch_size, seq_len, layer.heads, layer.head_dim),
        initial_state=initial_state,
    )
    mixed = F.group_norm(
        recurrent_out.reshape(batch_size * seq_len, channels),
        num_groups=layer.heads,
        weight=layer.ln_x.weight,
        bias=layer.ln_x.bias,
        eps=layer.ln_x.eps,
    ).reshape(batch_size, seq_len, channels)
    local = (
        (
            r_flat.view(batch_size, seq_len, layer.heads, layer.head_dim)
            * k_flat.view(batch_size, seq_len, layer.heads, layer.head_dim)
            * layer.r_k
        ).sum(dim=-1, keepdim=True)
        * v_flat.view(batch_size, seq_len, layer.heads, layer.head_dim)
    ).view(batch_size, seq_len, channels)
    return F.linear((mixed + local) * gate, layer.output.weight), terminal_state, v_first


def check_source_and_initialization(device: torch.device) -> dict[str, Any]:
    if (
        RWKV7_OFFICIAL_SOURCE_COMMIT != EXPECTED_SOURCE_COMMIT
        or RWKV7_OFFICIAL_SOURCE_BLOB != EXPECTED_SOURCE_BLOB
        or RWKV7_OFFICIAL_SOURCE_PATH != EXPECTED_SOURCE_PATH
        or RWKV7_OFFICIAL_KERNEL_BLOB != EXPECTED_KERNEL_BLOB
        or RWKV7_OFFICIAL_KERNEL_PATH != EXPECTED_KERNEL_PATH
        or RWKV7_STATEPASSING_CUDA_SHA256 != EXPECTED_STATEPASSING_CUDA_SHA256
    ):
        raise AssertionError("RWKV7 official source provenance changed")
    statepassing_source = (
        Path(__file__).resolve().parent
        / "rwkv7_cuda"
        / "rwkv_cuda_statepassing"
        / "rwkv7_statepassing_clampw.cu"
    )
    statepassing_sha256 = file_sha256(statepassing_source)
    if statepassing_sha256 != EXPECTED_STATEPASSING_CUDA_SHA256:
        raise AssertionError(
            "Vendored RWKV7 state-passing CUDA changed: "
            f"{statepassing_sha256} != {EXPECTED_STATEPASSING_CUDA_SHA256}"
        )
    torch.manual_seed(100)
    layer = RWKV7TimeMixOfficial(
        192,
        6,
        32,
        layer_id=3,
        layers=10,
        rwkv_kernel="torch",
    ).to(device)
    expected_shapes = {
        "w1": [192, 32],
        "w2": [32, 192],
        "a1": [192, 32],
        "a2": [32, 192],
        "v1": [192, 32],
        "v2": [32, 192],
        "g1": [192, 64],
        "g2": [64, 192],
        "r_k": [6, 32],
    }
    actual_shapes = {name: list(getattr(layer, name).shape) for name in expected_shapes}
    if actual_shapes != expected_shapes:
        raise AssertionError(f"RWKV7 parameter shapes differ from official contract: {actual_shapes}")
    zero_parameters = {
        "w1": float(layer.w1.abs().max().item()),
        "a1": float(layer.a1.abs().max().item()),
        "v1": float(layer.v1.abs().max().item()),
        "g1": float(layer.g1.abs().max().item()),
        "output.weight": float(layer.output.weight.abs().max().item()),
    }
    if any(value != 0.0 for value in zero_parameters.values()):
        raise AssertionError(f"RWKV7 zero initialization contract failed: {zero_parameters}")
    return {
        "source_commit": RWKV7_OFFICIAL_SOURCE_COMMIT,
        "source_blob": RWKV7_OFFICIAL_SOURCE_BLOB,
        "source_path": RWKV7_OFFICIAL_SOURCE_PATH,
        "kernel_blob": RWKV7_OFFICIAL_KERNEL_BLOB,
        "kernel_path": RWKV7_OFFICIAL_KERNEL_PATH,
        "statepassing_cuda_sha256": statepassing_sha256,
        "parameter_shapes": actual_shapes,
        "zero_initialization": zero_parameters,
    }


def check_decay_parameterization(device: torch.device) -> dict[str, Any]:
    raw = torch.linspace(-20.0, 20.0, 4097, device=device, dtype=torch.float64)
    official_pre_kernel = -F.softplus(-raw) - 0.5
    official_two_stage = torch.exp(-torch.exp(official_pre_kernel))
    fused_statepassing = torch.exp(-math.exp(-0.5) * torch.sigmoid(raw))
    row = compare(official_two_stage, fused_statepassing)
    if row["max_abs"] > 2e-15:
        raise AssertionError(f"Fused RWKV7 decay differs from official Python+CUDA mapping: {row}")
    return row


def check_independent_formula(device: torch.device) -> dict[str, Any]:
    torch.manual_seed(101)
    layer = RWKV7TimeMixOfficial(64, 2, 32, layer_id=1, layers=3, rwkv_kernel="torch").to(device)
    randomize_open_paths(layer)
    x = torch.randn(2, 16, 64, device=device, dtype=torch.float32)
    initial_state = torch.randn(2, 2, 32, 32, device=device, dtype=torch.float32) * 0.03
    v_first = torch.randn_like(x) * 0.1
    actual = layer(x, initial_state=initial_state, v_first=v_first)
    reference = independent_frontend(layer, x, initial_state, v_first)
    result = {
        "output": compare(reference[0], actual[0]),
        "terminal_state": compare(reference[1], actual[1]),
        "v_first": compare(reference[2], actual[2]),
    }
    if max(row["max_abs"] for row in result.values()) > 2e-6:
        raise AssertionError(f"RWKV7 frontend differs from independent official-equation reference: {result}")
    return result


def check_layer_zero_value(device: torch.device) -> dict[str, Any]:
    torch.manual_seed(104)
    layer = RWKV7TimeMixOfficial(64, 2, 32, layer_id=0, layers=3, rwkv_kernel="torch").to(device)
    x = torch.randn(1, 16, 64, device=device)
    output, terminal_state, returned_v_first = layer(x, initial_state=None, v_first=None)
    shifted = torch.zeros_like(x)
    shifted[:, 1:] = x[:, :-1]
    xv = x + (shifted - x) * layer.x_v
    expected_v_first = F.linear(xv, layer.value.weight)
    row = compare(expected_v_first, returned_v_first)
    if row["max_abs"] > 2e-6:
        raise AssertionError(f"RWKV7 layer zero did not expose its official value residual source: {row}")
    return {
        "v_first": row,
        "output_finite": bool(torch.isfinite(output).all()),
        "terminal_state_finite": bool(torch.isfinite(terminal_state).all()),
    }


def check_cuda_against_torch(device: torch.device) -> dict[str, Any]:
    torch.manual_seed(102)
    torch_layer = RWKV7TimeMixOfficial(64, 2, 32, layer_id=1, layers=3, rwkv_kernel="torch").to(device)
    randomize_open_paths(torch_layer)
    cuda_layer = copy.deepcopy(torch_layer)
    cuda_layer.rwkv_kernel = "statepassing"
    x = torch.randn(2, 32, 64, device=device, dtype=torch.float32)
    initial_state = torch.randn(2, 2, 32, 32, device=device, dtype=torch.float32) * 0.03
    v_first = torch.randn_like(x) * 0.1
    dy = torch.randn_like(x) * 0.1
    ds = torch.randn_like(initial_state) * 0.01

    def run_layer(
        layer: RWKV7TimeMixOfficial,
    ) -> tuple[tuple[torch.Tensor, torch.Tensor, torch.Tensor], list[torch.Tensor]]:
        inputs = [
            x.detach().clone().requires_grad_(True),
            initial_state.detach().clone().requires_grad_(True),
            v_first.detach().clone().requires_grad_(True),
        ]
        with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
            outputs = layer(inputs[0], initial_state=inputs[1], v_first=inputs[2])
            loss = (outputs[0].float() * dy).sum() + (outputs[1] * ds).sum()
        gradients = torch.autograd.grad(loss, inputs + list(layer.parameters()))
        return outputs, list(gradients)

    torch_outputs, torch_gradients = run_layer(torch_layer)
    cuda_outputs, cuda_gradients = run_layer(cuda_layer)
    gradient_names = ["x", "initial_state", "v_first"] + [name for name, _parameter in torch_layer.named_parameters()]
    gradient_rows = {
        name: compare(reference, actual)
        for name, reference, actual in zip(gradient_names, torch_gradients, cuda_gradients)
    }
    result = {
        "output": compare(torch_outputs[0], cuda_outputs[0]),
        "terminal_state": compare(torch_outputs[1], cuda_outputs[1]),
        "v_first": compare(torch_outputs[2], cuda_outputs[2]),
        "gradients": gradient_rows,
        "all_gradients_finite": all_finite(cuda_gradients),
    }
    if result["output"]["max_abs"] > 0.02 or result["terminal_state"]["max_abs"] > 0.02:
        raise AssertionError(f"RWKV7 CUDA/Torch forward mismatch: {result}")
    failed_gradients = {
        name: row
        for name, row in gradient_rows.items()
        if row["max_abs"] > 0.02 and row["relative_rms"] > 0.08
    }
    if failed_gradients:
        raise AssertionError(f"RWKV7 CUDA/Torch gradient mismatch: {failed_gradients}")
    if not result["all_gradients_finite"]:
        raise AssertionError("RWKV7 CUDA frontend returned non-finite gradients")
    return result


def check_value_residual_and_optimizer(device: torch.device) -> dict[str, Any]:
    torch.manual_seed(103)
    layer = RWKV7TimeMixOfficial(64, 2, 32, layer_id=1, layers=3, rwkv_kernel="torch").to(device)
    randomize_open_paths(layer)
    x = torch.randn(1, 16, 64, device=device)
    initial_state = torch.randn(1, 2, 32, 32, device=device) * 0.03
    v_first = torch.randn_like(x, requires_grad=True)
    output, _state, _returned_first = layer(x, initial_state=initial_state, v_first=v_first)
    zero_output, _zero_state, _zero_first = layer(x, initial_state=initial_state, v_first=torch.zeros_like(v_first))
    sensitivity = float((output - zero_output).float().square().mean().sqrt().item())
    cotangent = torch.randn_like(output)
    value_gradient = torch.autograd.grad((output.float() * cotangent.float()).sum(), v_first)[0]
    value_gradient_norm = float(value_gradient.float().norm().item())
    if sensitivity <= 1e-6 or value_gradient_norm <= 1e-6:
        raise AssertionError("RWKV7 v_first residual is inactive")

    optimizer, optimizer_runtime = build_adamw(
        layer,
        lr=0.0015,
        weight_decay=0.001,
        contract="rwkv7_decay_groups",
    )
    del optimizer
    decay_names = set(optimizer_runtime["decay_names"])
    no_decay_names = set(optimizer_runtime["no_decay_names"])
    required_decay = {"receptance.weight", "key.weight", "value.weight", "output.weight"}
    required_no_decay = {"w0", "w1", "w2", "a0", "a1", "a2", "v0", "v1", "v2", "g1", "g2", "r_k", "ln_x.weight"}
    if not required_decay.issubset(decay_names) or not required_no_decay.issubset(no_decay_names):
        raise AssertionError(f"RWKV7 optimizer grouping differs from official contract: {optimizer_runtime}")
    return {
        "v_first_output_rms_delta": sensitivity,
        "v_first_gradient_norm": value_gradient_norm,
        "optimizer": optimizer_runtime,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Strict GPU-only official RWKV7 frontend contract check")
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "0":
        raise RuntimeError("Set CUDA_VISIBLE_DEVICES=0; this check is GPU1-only")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required; CPU model smoke is forbidden")
    available, reason = statepassing_available(32)
    if not available:
        raise RuntimeError(f"RWKV7 statepassing CUDA is unavailable: {reason}")
    device = torch.device("cuda", 0)
    payload = {
        "device": torch.cuda.get_device_name(device),
        "torch": torch.__version__,
        "cuda_visible_devices": os.environ["CUDA_VISIBLE_DEVICES"],
        "source_and_initialization": check_source_and_initialization(device),
        "decay_parameterization": check_decay_parameterization(device),
        "independent_formula": check_independent_formula(device),
        "layer_zero_value": check_layer_zero_value(device),
        "cuda_against_torch": check_cuda_against_torch(device),
        "value_residual_and_optimizer": check_value_residual_and_optimizer(device),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
