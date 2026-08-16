from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
from pathlib import Path

import torch
import torch.nn.functional as F
import triton

from experiments.zoology_mqar.futureseed_directionality import dataset_hash
from experiments.zoology_mqar.length_scaling import (
    build_config,
    make_model,
)
from experiments.zoology_mqar.official_sdm_futureseed import (
    EXPECTED_SDM_SHA,
    EXPECTED_STATE_VALUES_PER_LAYER,
    SDM_BLOCK_SIZE,
    SDM_HEADS,
    SDM_READS,
    SDM_SLOTS,
    SDM_WRITES,
    ZoologyOfficialSDMFutureSeedMixer,
    external_sdm_provenance,
    load_matched_parent_state,
    official_sdm_diagnostics,
)
from zoology.data.utils import prepare_data
from zoology.utils import set_determinism


EXPECTED_MATCHED_INIT_SHA256 = (
    "7402e46c48cbd47070d65262d1a1b62a32ac55a4d644a6fb9644b96b0914850f"
)
EXPECTED_TRAIN_HASH = (
    "647c64ece84984a23dfd817c4f277ea83840dbec57cc18bb6c9bf9eda7cc9a68"
)
EXPECTED_TEST_HASH = (
    "4a8237ba8fe19aaff0d1d72de7b7f6505eaab59cd091442c2f463df34cce278f"
)
EXPECTED_MIXER_PARAMETERS = 66_565


def normalized_uuid(value: str) -> str:
    return value.removeprefix("GPU-").lower()


def _visible_gpu() -> tuple[str, str]:
    row = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=uuid,name",
            "--format=csv,noheader",
        ],
        text=True,
    ).strip()
    uuid, name = [field.strip() for field in row.split(",", maxsplit=1)]
    return uuid, name


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _version(value: str) -> tuple[int, ...]:
    prefix = value.split("+", maxsplit=1)[0]
    return tuple(int(part) for part in prefix.split(".")[:3])


def backward_names(loss: torch.Tensor) -> list[str]:
    names: list[str] = []
    visited = set()
    queue = [loss.grad_fn]
    while queue:
        node = queue.pop()
        if node is None or node in visited:
            continue
        visited.add(node)
        names.append(type(node).__name__)
        queue.extend(next_node for next_node, _index in node.next_functions)
    return names


def _gradient(parameter: torch.Tensor, name: str) -> dict[str, float | bool]:
    gradient = parameter.grad
    if gradient is None or not torch.isfinite(gradient).all():
        raise RuntimeError(f"Missing or nonfinite gradient: {name}")
    rms = float(gradient.float().square().mean().sqrt())
    nonzero = bool((gradient != 0).any())
    if not math.isfinite(rms) or rms <= 0 or not nonzero:
        raise RuntimeError(f"Inactive gradient: {name}")
    return {"rms": rms, "nonzero": nonzero}


def _relative_rms(left: torch.Tensor, right: torch.Tensor) -> float:
    numerator = (left.float() - right.float()).square().mean().sqrt()
    denominator = right.float().square().mean().sqrt().clamp_min(1e-8)
    return float(numerator / denominator)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--matched-init", type=Path, required=True)
    parser.add_argument("--expected-gpu-name", required=True)
    parser.add_argument("--expected-gpu-uuid", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if os.environ.get("CUDA_VISIBLE_DEVICES") != "0":
        raise RuntimeError("CUDA_VISIBLE_DEVICES must be exactly 0")
    if torch.cuda.device_count() != 1 or torch.cuda.current_device() != 0:
        raise RuntimeError("P-GDN3-064 requires exactly CUDA index 0")
    gpu_uuid, gpu_name = _visible_gpu()
    device = torch.cuda.get_device_properties(0)
    if (
        gpu_name != args.expected_gpu_name
        or gpu_uuid != args.expected_gpu_uuid
        or device.name != args.expected_gpu_name
        or normalized_uuid(str(getattr(device, "uuid", "unavailable")))
        != normalized_uuid(args.expected_gpu_uuid)
    ):
        raise RuntimeError(f"Unexpected GPU: {gpu_uuid}, {gpu_name}")
    if _version(torch.__version__)[:2] != (2, 8):
        raise RuntimeError(f"Official SDM requires PyTorch 2.8, got {torch.__version__}")
    if _version(triton.__version__) < (3, 4):
        raise RuntimeError(f"Official SDM requires Triton >=3.4, got {triton.__version__}")
    if not args.matched_init.is_file() or _sha256(args.matched_init) != EXPECTED_MATCHED_INIT_SHA256:
        raise RuntimeError("Matched initialization drifted")
    provenance = external_sdm_provenance()
    if provenance["sha"] != EXPECTED_SDM_SHA:
        raise RuntimeError("Official SDM provenance drifted")

    config = build_config(
        arm="future_seed_official_sdm",
        sequence_length=1024,
        num_kv_pairs=4,
        max_epochs=10,
        batch_size=32,
    )
    train_loader, test_loader = prepare_data(config.data)
    data_hashes = {
        "train": dataset_hash(train_loader),
        "test": dataset_hash(test_loader),
    }
    if data_hashes != {"train": EXPECTED_TRAIN_HASH, "test": EXPECTED_TEST_HASH}:
        raise RuntimeError(f"Directional MQAR data drifted: {data_hashes}")

    parent_state = torch.load(args.matched_init, map_location="cpu", weights_only=True)
    set_determinism(123)
    model = make_model(config, "future_seed_official_sdm")
    load_matched_parent_state(model, parent_state)
    mixers = [block.sequence_mixer for block in model.backbone.layers]
    if len(mixers) != 2 or not all(
        type(mixer) is ZoologyOfficialSDMFutureSeedMixer for mixer in mixers
    ):
        raise RuntimeError("Expected two exact official SDM mixers")
    mixer_parameter_counts = [sum(p.numel() for p in mixer.parameters()) for mixer in mixers]
    if mixer_parameter_counts != [EXPECTED_MIXER_PARAMETERS, EXPECTED_MIXER_PARAMETERS]:
        raise RuntimeError(f"Official SDM parameter accounting drifted: {mixer_parameter_counts}")
    if any(mixer.state_size() != EXPECTED_STATE_VALUES_PER_LAYER for mixer in mixers):
        raise RuntimeError("Official SDM state accounting drifted")
    if any(mixer.layer.memory is not None for mixer in mixers):
        raise RuntimeError("P-GDN3-064 forbids learned initial memory")

    inputs, targets, _slices = next(iter(test_loader))
    inputs, targets = inputs[:4].cuda(), targets[:4].cuda()
    model = model.cuda().train()
    model.zero_grad(set_to_none=True)
    logits = model(inputs)
    mask = targets != -100
    loss = F.cross_entropy(logits[mask], targets[mask])
    graph_names = backward_names(loss)
    official_backward_count = sum(
        "GatedSparseMemoryWriteReadBackward" in name for name in graph_names
    )
    if official_backward_count != 2:
        raise RuntimeError(
            f"Expected two official sparse backward paths, got {official_backward_count}: {sorted(graph_names)}"
        )
    producer_terminal_autograd = {
        "requires_grad": mixers[0].last_terminal_requires_grad,
        "grad_fn": mixers[0].last_terminal_grad_fn,
    }
    loss.backward()
    gradients = []
    for index, mixer in enumerate(mixers):
        layer = mixer.layer
        row = {
            "layer": index,
            "read": _gradient(layer.Wq_read.weight, f"layer{index}.read"),
            "write": _gradient(layer.Wk_write.weight, f"layer{index}.write"),
            "value": _gradient(layer.Wv_write.weight, f"layer{index}.value"),
            "decay": _gradient(layer.a_proj.weight, f"layer{index}.decay"),
            "input_gate": _gradient(layer.b_proj.weight, f"layer{index}.input_gate"),
            "output": _gradient(layer.Wo.weight, f"layer{index}.output"),
            "output_gate": _gradient(layer.Wog.weight, f"layer{index}.output_gate"),
        }
        if index == 1:
            row["future_seed"] = _gradient(
                mixer.future_seed_logit,
                "layer1.future_seed",
            )
        gradients.append(row)

    generator = torch.Generator(device="cuda").manual_seed(64064)
    hidden = torch.randn(2, 128, 128, generator=generator, device="cuda")
    incoming = 0.03 * torch.randn(
        2,
        SDM_HEADS,
        SDM_SLOTS,
        128,
        generator=generator,
        device="cuda",
        dtype=torch.bfloat16,
    )
    receiving = mixers[1].eval()
    with torch.no_grad():
        zero_output, zero_terminal = receiving.forward_with_state(
            hidden,
            initial_state=torch.zeros_like(incoming),
        )
        seeded_output, seeded_terminal = receiving.forward_with_state(
            hidden,
            initial_state=incoming.clone(),
        )
    incoming_dependency = {
        "output_relative_rms": _relative_rms(seeded_output, zero_output),
        "terminal_relative_rms": _relative_rms(seeded_terminal, zero_terminal),
        "finite": bool(
            torch.isfinite(seeded_output).all()
            and torch.isfinite(seeded_terminal).all()
        ),
    }
    if (
        incoming_dependency["output_relative_rms"] <= 1e-4
        or incoming_dependency["terminal_relative_rms"] <= 1e-4
        or not incoming_dependency["finite"]
    ):
        raise RuntimeError(f"Incoming sparse state dependency failed: {incoming_dependency}")

    diagnostics = official_sdm_diagnostics(model, inputs[:2])
    rows = diagnostics["per_layer"]
    if (
        diagnostics["active_layers"] != 2
        or diagnostics["active_futureseed_routes"] != 1
        or diagnostics["state_values_per_layer"] != EXPECTED_STATE_VALUES_PER_LAYER
        or diagnostics["heads"] != SDM_HEADS
        or diagnostics["slots_per_head"] != SDM_SLOTS
        or diagnostics["reads"] != SDM_READS
        or diagnostics["writes"] != SDM_WRITES
        or diagnostics["block_size"] != SDM_BLOCK_SIZE
    ):
        raise RuntimeError(f"Official SDM diagnostic geometry failed: {diagnostics}")
    for row in rows:
        access = row["access"]
        read_slots = access["read_unique_pct"] * SDM_SLOTS / 100.0
        write_slots = access["write_unique_pct"] * SDM_SLOTS / 100.0
        if (
            row["update_autocast_enabled"] is not False
            or row["update_input_dtypes"]["memory"] != "torch.bfloat16"
            or read_slots <= 64
            or write_slots <= 64
            or access["read_slot_entropy_normalized"] <= 0
            or access["write_slot_entropy_normalized"] <= 0
            or row["active_slots"] <= 64
            or not math.isfinite(row["state_rms"])
            or row["state_rms"] <= 1e-4
        ):
            raise RuntimeError(f"Official SDM activation failed: {row}")

    result = {
        "status": "passed",
        "plan": "P-GDN3-064",
        "gpu": {"uuid": gpu_uuid, "name": gpu_name},
        "runtime": {"torch": torch.__version__, "triton": triton.__version__},
        "external": provenance,
        "data_hashes": data_hashes,
        "parameters": {
            "model": sum(parameter.numel() for parameter in model.parameters()),
            "per_mixer": mixer_parameter_counts,
            "expected_per_mixer": EXPECTED_MIXER_PARAMETERS,
        },
        "state_values_per_layer": EXPECTED_STATE_VALUES_PER_LAYER,
        "official_backward_count": official_backward_count,
        "gradients": gradients,
        "producer_terminal_autograd": producer_terminal_autograd,
        "incoming_state_dependency": incoming_dependency,
        "diagnostics": diagnostics,
        "dense_gdn2_scan_count": 0,
        "external_source_redistributed": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
