#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import os
import subprocess
from pathlib import Path
from types import SimpleNamespace

import torch

from momentum_delta_sudoku import (
    EXPECTED_CHUNK_SHA256,
    EXPECTED_HOST_FLA_SHA,
    EXPECTED_LAYER_SHA256,
    EXPECTED_MDN_SHA,
    EXPECTED_RECURRENT_SHA256,
    load_external_momentum_layer,
    momentum_source_summary,
)
from study_rwkv_futureseed_loop import (
    FutureSeedRWKV,
    MomentumDeltaBlock,
    resolve_strict_fla_source,
    strict_fla_runtime_summary,
)


LAYERS = 10
D_MODEL = 192
HEADS = 6
HEAD_DIM = 32
CONV_SIZE = 4
EXPECTED_CORE_PARAMETERS = 191_300
EXPECTED_STATE_VALUES_PER_LAYER = 2 * HEADS * HEAD_DIM * HEAD_DIM


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def backward_names(tensor: torch.Tensor) -> list[str]:
    names: list[str] = []
    seen = set()
    stack = [tensor.grad_fn]
    while stack:
        function = stack.pop()
        if function is None or function in seen:
            continue
        seen.add(function)
        names.append(type(function).__name__)
        stack.extend(next_function for next_function, _ in function.next_functions)
    return names


def gradient_rms(parameter: torch.nn.Parameter, label: str) -> float:
    gradient = parameter.grad
    if gradient is None or not torch.isfinite(gradient).all():
        raise RuntimeError(f"{label} has no finite gradient")
    value = float(gradient.float().square().mean().sqrt().item())
    if value <= 0.0:
        raise RuntimeError(f"{label} gradient is inactive")
    return value


def visible_gpu() -> dict[str, str]:
    output = subprocess.check_output(
        [
            "nvidia-smi",
            "-i",
            "0",
            "--query-gpu=index,uuid,name",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    ).strip()
    rows = [row.strip() for row in output.splitlines() if row.strip()]
    require(len(rows) == 1, f"Expected one visible GPU row, got {rows}")
    index, uuid, name = (part.strip() for part in rows[0].split(",", 2))
    return {"index": index, "uuid": uuid, "name": name}


def build_reasoner(transport: str) -> FutureSeedRWKV:
    return FutureSeedRWKV(
        D_MODEL,
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
        momentum_future_seed_transport=transport,
        activation_checkpoint=False,
        backbone="momentum",
        gdn_mode="chunk",
        gdn_expand_v=1.0,
        gdn_use_short_conv=True,
        gdn_conv_size=CONV_SIZE,
        gdn_allow_neg_eigval=False,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--expected_gpu_name", required=True)
    parser.add_argument("--expected_gpu_uuid", required=True)
    args = parser.parse_args()

    require(os.environ.get("CUDA_VISIBLE_DEVICES") == "0", "CUDA index must be 0")
    require(torch.cuda.is_available(), "CUDA is unavailable; CPU fallback is forbidden")
    require(torch.cuda.device_count() == 1, "Exactly one CUDA device must be visible")
    gpu = visible_gpu()
    require(gpu["index"] == "0", f"Unexpected CUDA index: {gpu}")
    require(gpu["name"] == args.expected_gpu_name, f"Unexpected GPU name: {gpu}")
    require(gpu["uuid"] == args.expected_gpu_uuid, f"Unexpected GPU UUID: {gpu}")
    require(
        torch.cuda.get_device_name(0) == args.expected_gpu_name,
        "PyTorch and nvidia-smi disagree on GPU identity",
    )
    require(os.environ.get("FLA_EXPECTED_SOURCE_SHA") == EXPECTED_HOST_FLA_SHA, "Host FLA SHA drifted")
    require(os.environ.get("MDN_EXPECTED_SHA") == EXPECTED_MDN_SHA, "Momentum source SHA drifted")
    require(os.environ.get("FLA_USE_CUDA_GRAPH", "0") == "0", "CUDA graph must stay disabled")

    layer_class = load_external_momentum_layer()
    source = momentum_source_summary()
    layer_path = Path(inspect.getfile(layer_class)).resolve()
    fla_root = Path(os.environ["MDN_FLA_ROOT"]).resolve()
    source_hashes = {
        "layer": sha256(layer_path),
        "chunk": sha256(fla_root / "fla/ops/momentum_delta_rule/chunk.py"),
        "fused_recurrent": sha256(
            fla_root / "fla/ops/momentum_delta_rule/fused_recurrent.py"
        ),
    }
    require(
        source_hashes
        == {
            "layer": EXPECTED_LAYER_SHA256,
            "chunk": EXPECTED_CHUNK_SHA256,
            "fused_recurrent": EXPECTED_RECURRENT_SHA256,
        },
        f"Momentum source hashes drifted: {source_hashes}",
    )
    host_sha, host_root, host_module = resolve_strict_fla_source("momentum")
    require(host_sha == EXPECTED_HOST_FLA_SHA, "Pinned host FLA source drifted")

    torch.manual_seed(71_018)
    torch.cuda.manual_seed_all(71_018)
    device = torch.device("cuda:0")
    reasoner = build_reasoner("full_state").to(device).train()
    blocks = list(reasoner.blocks)
    require(len(blocks) == LAYERS, "Layer count drifted")
    require(
        all(type(block) is MomentumDeltaBlock for block in blocks),
        "A non-Momentum block entered the registered backbone",
    )

    provenance = []
    for index, block in enumerate(blocks):
        core = block.time_mix.core
        require(type(core) is layer_class, f"Layer {index} escaped the pinned external class")
        parameter_count = sum(parameter.numel() for parameter in core.parameters())
        require(
            parameter_count == EXPECTED_CORE_PARAMETERS,
            f"Layer {index} parameter geometry drifted: {parameter_count}",
        )
        require(
            block.time_mix.state_elements_per_head() == 2 * HEAD_DIM * HEAD_DIM,
            f"Layer {index} state geometry drifted",
        )
        provenance.append(
            {
                "layer": index,
                "class": f"{type(core).__module__}.{type(core).__qualname__}",
                "parameters": parameter_count,
                "state_shape": [2, "B", HEADS, HEAD_DIM, HEAD_DIM],
            }
        )

    x = torch.randn(2, 81, D_MODEL, device=device, requires_grad=True)
    output, diagnostics, _memory = reasoner(x)
    require(tuple(output.shape) == tuple(x.shape), "Reasoner output shape drifted")
    loss = output.float().square().mean()
    graph = backward_names(loss)
    chunk_count = sum("Chunkmode_ruleFunctionBackward" in name for name in graph)
    require(chunk_count == LAYERS, f"Expected {LAYERS} Momentum chunk paths, got {chunk_count}")
    loss.backward()

    gradients = []
    for index, block in enumerate(blocks):
        core = block.time_mix.core
        row = {
            "layer": index,
            "q": gradient_rms(core.q_proj.weight, f"layer{index}.q"),
            "k": gradient_rms(core.k_proj.weight, f"layer{index}.k"),
            "v": gradient_rms(core.v_proj.weight, f"layer{index}.v"),
            "alpha": gradient_rms(core.a_proj.weight, f"layer{index}.alpha"),
            "beta": gradient_rms(core.b_proj.weight, f"layer{index}.beta"),
            "momentum": gradient_rms(core.m_proj.weight, f"layer{index}.momentum"),
            "eta": gradient_rms(core.e_proj.weight, f"layer{index}.eta"),
        }
        if index > 0:
            row["future_seed"] = gradient_rms(
                block.future_seed_logit,
                f"layer{index}.future_seed",
            )
        require(
            block.time_mix.last_terminal_state_shape
            == (2, 2, HEADS, HEAD_DIM, HEAD_DIM),
            f"Layer {index} terminal state shape drifted",
        )
        gradients.append(row)

    activation = {
        key: float(value.detach().float().item())
        for key, value in diagnostics.items()
        if key.startswith("momentum_")
    }
    require(activation.get("momentum_active_layers_sum") == LAYERS, "Not all Momentum layers activated")
    require(activation.get("momentum_state_rms", 0.0) > 1e-4, "State plane is inactive")
    require(activation.get("momentum_velocity_rms", 0.0) > 1e-4, "Momentum plane is inactive")
    require(activation.get("momentum_state_board_std", 0.0) > 0.0, "State has no board variation")
    require(activation.get("momentum_velocity_board_std", 0.0) > 0.0, "Momentum has no board variation")
    require(
        activation.get("momentum_future_seed_transport_fraction") == 1.0,
        "Full-state FutureSeed transport is not active",
    )

    reasoner.eval()
    probe = torch.randn(2, 81, D_MODEL, device=device)
    with torch.no_grad():
        reasoner.momentum_future_seed_transport = "full_state"
        full_output, full_diag, _ = reasoner(probe)
        reasoner.momentum_future_seed_transport = "momentum_only"
        momentum_output, momentum_diag, _ = reasoner(probe)
    deployment_effect = float(
        (full_output.float() - momentum_output.float()).square().mean().sqrt().item()
    )
    require(deployment_effect > 1e-7, "Momentum-only transport does not affect the computation")
    require(
        float(momentum_diag["momentum_future_seed_transport_fraction"].item()) == 0.5,
        "Momentum-only transport did not halve the transported state",
    )

    runtime = strict_fla_runtime_summary(
        SimpleNamespace(reasoner=reasoner),
        "momentum",
    )
    require(len(runtime["layers"]) == LAYERS, "Strict runtime summary lost layers")
    payload = {
        "status": "pass",
        "torch_version": torch.__version__,
        "gpu": gpu,
        "host_fla": {
            "sha": host_sha,
            "root": host_root,
            "module": host_module,
        },
        "momentum_source": source,
        "source_hashes": source_hashes,
        "provenance": provenance,
        "backward_chunk_count": chunk_count,
        "gradients": gradients,
        "activation": activation,
        "deployment_transport": {
            "full_fraction": float(
                full_diag["momentum_future_seed_transport_fraction"].item()
            ),
            "momentum_only_fraction": float(
                momentum_diag["momentum_future_seed_transport_fraction"].item()
            ),
            "same_weight_output_rms_difference": deployment_effect,
        },
        "state_values_per_layer_per_board": EXPECTED_STATE_VALUES_PER_LAYER,
        "state_values_all_layers_per_board": LAYERS * EXPECTED_STATE_VALUES_PER_LAYER,
        "strict_runtime": runtime,
        "fallback": False,
    }
    output_path = Path(args.out)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
