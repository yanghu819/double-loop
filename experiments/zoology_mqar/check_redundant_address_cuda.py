from __future__ import annotations

import argparse
import copy
import inspect
import json
import os
import subprocess
from pathlib import Path

import torch
import torch.nn.functional as F
from fla.layers.gdn2 import GatedDeltaNet2
from zoology.utils import set_determinism

from experiments.zoology_mqar.gdn2_futureseed import PINNED_FLA_SHA
from experiments.zoology_mqar.gdn2_redundant_address import (
    EXPECTED_PARAMETER_DELTA,
    PARENT_HEADS,
    REDUNDANT_STATE_VALUES,
    RedundantAddressGatedDeltaNet2,
    ZoologyRedundantAddressFutureSeedMixer,
    load_matched_parent_state,
    parent_parameter_hash,
)
from experiments.zoology_mqar.length_scaling import build_config, make_model


EXPECTED_PARENT_PARAMETERS = 661_584
EXPECTED_PARENT_PARAMETER_HASH = (
    "3e8fe038f1401735168783c2de1d9217a8807e88685ee12010142fc7b05e4c44"
)


def _visible_gpu() -> tuple[str, str]:
    output = subprocess.check_output(
        [
            "nvidia-smi",
            "-i",
            "0",
            "--query-gpu=uuid,name",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    ).strip()
    rows = [row.strip() for row in output.splitlines() if row.strip()]
    if len(rows) != 1:
        raise RuntimeError(f"Expected one visible GPU row, got {rows}")
    uuid, name = (part.strip() for part in rows[0].split(",", maxsplit=1))
    return uuid, name


def _grad_fn_counts(loss: torch.Tensor) -> dict[str, int]:
    counts: dict[str, int] = {}
    pending = [loss.grad_fn]
    seen: set[object] = set()
    while pending:
        node = pending.pop()
        if node is None or node in seen:
            continue
        seen.add(node)
        name = type(node).__name__
        counts[name] = counts.get(name, 0) + 1
        pending.extend(next_node for next_node, _index in node.next_functions)
    return counts


def _gradient_summary(model: torch.nn.Module) -> dict[str, float]:
    wanted = (
        ".secondary_q_proj.weight",
        ".secondary_k_proj.weight",
        ".secondary_q_conv1d.weight",
        ".secondary_k_conv1d.weight",
        ".base.q_proj.weight",
        ".base.k_proj.weight",
        ".base.v_proj.weight",
        ".base.b_proj.weight",
        ".base.w_proj.weight",
    )
    rows: dict[str, float] = {}
    for name, parameter in model.named_parameters():
        if not name.endswith(wanted):
            continue
        if parameter.grad is None:
            raise RuntimeError(f"Missing gradient: {name}")
        gradient = parameter.grad.detach().float()
        if not torch.isfinite(gradient).all() or float(gradient.abs().max()) == 0:
            raise RuntimeError(f"Inactive or nonfinite gradient: {name}")
        rows[name] = float(gradient.square().mean().sqrt().item())
    if len(rows) != 18:
        raise RuntimeError(f"Expected 18 address/edit gradients, got {sorted(rows)}")
    return rows


@torch.no_grad()
def _bank_swap_contract(
    mixer: ZoologyRedundantAddressFutureSeedMixer,
    hidden_states: torch.Tensor,
) -> dict[str, float]:
    layer = mixer.layer
    if not isinstance(layer, RedundantAddressGatedDeltaNet2):
        raise TypeError("Expected redundant-address layer")
    output, state = mixer.forward_with_state(hidden_states, initial_state=None)
    module_pairs = (
        (layer.base.q_proj, layer.secondary_q_proj),
        (layer.base.k_proj, layer.secondary_k_proj),
        (layer.base.q_conv1d, layer.secondary_q_conv1d),
        (layer.base.k_conv1d, layer.secondary_k_conv1d),
    )
    saved = [
        (copy.deepcopy(first.state_dict()), copy.deepcopy(second.state_dict()))
        for first, second in module_pairs
    ]
    try:
        for (first, second), (first_state, second_state) in zip(
            module_pairs,
            saved,
        ):
            first.load_state_dict(second_state)
            second.load_state_dict(first_state)
        swapped_output, swapped_state = mixer.forward_with_state(
            hidden_states,
            initial_state=None,
        )
    finally:
        for (first, second), (first_state, second_state) in zip(
            module_pairs,
            saved,
        ):
            first.load_state_dict(first_state)
            second.load_state_dict(second_state)
    expected_state = torch.cat(
        (
            state[:, PARENT_HEADS:],
            state[:, :PARENT_HEADS],
        ),
        dim=1,
    )
    return {
        "output_max_error": float(
            (swapped_output.float() - output.float()).abs().max().item()
        ),
        "state_max_error": float(
            (swapped_state.float() - expected_state.float()).abs().max().item()
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--matched-init", type=Path, required=True)
    parser.add_argument("--expected-gpu-uuid", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "0":
        raise RuntimeError("P-GDN3-053 contract requires CUDA_VISIBLE_DEVICES=0")
    if torch.cuda.device_count() != 1:
        raise RuntimeError("P-GDN3-053 contract requires exactly one visible GPU")
    gpu_uuid, gpu_name = _visible_gpu()
    if gpu_uuid != args.expected_gpu_uuid or "A800" not in gpu_name:
        raise RuntimeError(f"Unexpected GPU: {gpu_uuid}, {gpu_name}")
    if os.environ.get("FLA_EXPECTED_SOURCE_SHA") != PINNED_FLA_SHA:
        raise RuntimeError("Pinned FLA SHA was not asserted")
    source_path = Path(inspect.getfile(GatedDeltaNet2)).resolve()
    if Path("/huyang2/double-loop/.cache/fla-versions") not in source_path.parents:
        raise RuntimeError(f"Unexpected GDN2 source path: {source_path}")

    config = build_config(
        arm="future_seed_redundant_address_gdn2",
        sequence_length=1024,
        num_kv_pairs=4,
        max_epochs=10,
        batch_size=32,
    )
    set_determinism(123)
    model = make_model(config, "future_seed_redundant_address_gdn2")
    parent_state = torch.load(
        args.matched_init,
        map_location="cpu",
        weights_only=True,
    )
    load_matched_parent_state(model, parent_state)
    parent_hash = parent_parameter_hash(model)
    if parent_hash != EXPECTED_PARENT_PARAMETER_HASH:
        raise RuntimeError(f"Parent initialization drifted: {parent_hash}")
    parameters = sum(parameter.numel() for parameter in model.parameters())
    if parameters != EXPECTED_PARENT_PARAMETERS + EXPECTED_PARAMETER_DELTA:
        raise RuntimeError(
            f"Parameter count drifted: {parameters} != "
            f"{EXPECTED_PARENT_PARAMETERS + EXPECTED_PARAMETER_DELTA}"
        )
    mixers = [block.sequence_mixer for block in model.backbone.layers]
    if len(mixers) != 2 or not all(
        isinstance(mixer, ZoologyRedundantAddressFutureSeedMixer)
        for mixer in mixers
    ):
        raise RuntimeError("Contract requires exactly two redundant-address layers")
    if not all(isinstance(mixer.layer.base, GatedDeltaNet2) for mixer in mixers):
        raise RuntimeError("Every layer must wrap pinned official GatedDeltaNet2")

    model = model.cuda().train()
    inputs = torch.randint(0, 256, (2, 64), device="cuda")
    logits = model(inputs)
    loss = logits.float().square().mean()
    grad_counts = _grad_fn_counts(loss)
    if grad_counts.get("ChunkGDN2FunctionBackward", 0) != 2:
        raise RuntimeError(f"Official backward count drifted: {grad_counts}")
    loss.backward()
    gradients = _gradient_summary(model)

    hidden_states = torch.randn(2, 64, 128, device="cuda")
    output0, state0 = mixers[0].forward_with_state(
        hidden_states,
        initial_state=None,
    )
    seed = mixers[1].make_initial_state(state0)
    output1, state1 = mixers[1].forward_with_state(
        hidden_states,
        initial_state=seed,
    )
    expected_state_shape = (2, 8, 32, 32)
    if (
        tuple(state0.shape) != expected_state_shape
        or tuple(state1.shape) != expected_state_shape
    ):
        raise RuntimeError(
            f"State geometry drifted: {tuple(state0.shape)}, {tuple(state1.shape)}"
        )
    if state0[0].numel() != REDUNDANT_STATE_VALUES:
        raise RuntimeError("Per-board recurrent state count drifted")
    if not all(
        torch.isfinite(tensor).all()
        for tensor in (output0, state0, seed, output1, state1)
    ):
        raise RuntimeError("Nonfinite output or recurrent state")
    if mixers[1].last_seed_gate is None or mixers[1].last_seed_norm is None:
        raise RuntimeError("Native FutureSeed route did not activate")

    bank_swap = _bank_swap_contract(mixers[0], hidden_states)
    if bank_swap["output_max_error"] > 3e-6 or bank_swap["state_max_error"] > 3e-6:
        raise RuntimeError(f"Bank-swap equivariance failed: {bank_swap}")
    independence = []
    for mixer in mixers:
        layer = mixer.layer
        q_cosine = F.cosine_similarity(
            layer.base.q_proj.weight.detach().float().flatten(),
            layer.secondary_q_proj.weight.detach().float().flatten(),
            dim=0,
        )
        k_cosine = F.cosine_similarity(
            layer.base.k_proj.weight.detach().float().flatten(),
            layer.secondary_k_proj.weight.detach().float().flatten(),
            dim=0,
        )
        row = {
            "layer_idx": mixer.layer_idx,
            "q_weight_cosine": float(q_cosine.item()),
            "k_weight_cosine": float(k_cosine.item()),
        }
        if abs(row["q_weight_cosine"]) >= 0.99 or abs(row["k_weight_cosine"]) >= 0.99:
            raise RuntimeError(
                f"Address banks were not independently initialized: {row}"
            )
        independence.append(row)

    payload = {
        "status": "passed",
        "plan": "P-GDN3-053",
        "gpu": {"uuid": gpu_uuid, "name": gpu_name},
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "fla_source_sha": PINNED_FLA_SHA,
        "gdn2_source_path": str(source_path),
        "parameters": parameters,
        "parent_parameters": EXPECTED_PARENT_PARAMETERS,
        "parameter_delta": EXPECTED_PARAMETER_DELTA,
        "state_values_per_layer": REDUNDANT_STATE_VALUES,
        "official_scans_per_layer": 1,
        "official_backward_count": grad_counts["ChunkGDN2FunctionBackward"],
        "parent_parameter_hash": parent_hash,
        "bank_swap": bank_swap,
        "independent_address_initialization": independence,
        "gradient_rms": gradients,
        "future_seed_gate": float(mixers[1].last_seed_gate.item()),
        "future_seed_norm": float(mixers[1].last_seed_norm.item()),
        "token_router": None,
        "fallback": None,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
