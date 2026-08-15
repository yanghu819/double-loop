from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from pathlib import Path
from typing import Any

import torch
from zoology.data.utils import prepare_data
from zoology.utils import set_determinism

from experiments.zoology_mqar.canonical_address_companion_endpoint import (
    transition_summary,
)
from experiments.zoology_mqar.futureseed_directionality import dataset_hash
from experiments.zoology_mqar.gdn2_cycle_memory import (
    CycleMemoryBackbone,
    parent_parameter_hash,
)
from experiments.zoology_mqar.gdn2_log_spd_endpoint import wrong_key_swap_summary
from experiments.zoology_mqar.length_scaling import (
    build_config,
    evaluate,
    make_model,
    parameter_hash,
)


SEQUENCE_LENGTH = 1024
NUM_KV_PAIRS = 4
MAX_EPOCHS = 10
BATCH_SIZE = 32
SEED = 123
CANDIDATE_ARM = "future_seed_cycle_memory_gdn2"
EXPECTED_CHECKPOINT_SHA256 = (
    "64bb48df94f0e7baf2102e47ee5ca0df8d032fb172e234815cf91a28d5d5e1fe"
)
EXPECTED_FORMAL_SCORE_SHA256 = (
    "4cdc5bfc99dd3c404da7fa79abb39a48567a20ec27d0f90d955fa4d71652d159"
)
EXPECTED_FORMAL_CASES_SHA256 = (
    "903cddf0bfddd4cf3d45db2067dadf7495c3a8a0ed426a8947d050c7f623bb8f"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _gpu_identity() -> dict[str, Any]:
    rows = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=index,uuid,name",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    ).strip().splitlines()
    if len(rows) != 1:
        raise RuntimeError(f"Expected one visible GPU, found {rows}")
    index, uuid, name = (item.strip() for item in rows[0].split(",", 2))
    expected_name = os.environ["EXPECTED_GPU_NAME"]
    expected_uuid = os.environ["EXPECTED_GPU_UUID"]
    if index != "0" or uuid != expected_uuid or name != expected_name:
        raise RuntimeError(
            f"GPU mismatch: {(index, uuid, name)} != "
            f"{('0', expected_uuid, expected_name)}"
        )
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("The diagnostic requires exactly one CUDA device")
    return {"index": 0, "uuid": uuid, "name": name}


def _cycle_gates(model: torch.nn.Module) -> list[torch.nn.Parameter]:
    if not isinstance(model.backbone, CycleMemoryBackbone):
        raise TypeError("Expected CycleMemoryBackbone")
    gates = [
        block.sequence_mixer.layer.cycle_gate
        for block in model.backbone.layers
    ]
    if len(gates) != 2 or sum(gate.numel() for gate in gates) != 8:
        raise RuntimeError("Expected exactly eight cycle-gate scalars")
    return gates


def _metric_row(metrics: dict[str, Any]) -> dict[str, float]:
    return {
        "balanced_accuracy": float(metrics["balanced_accuracy"]),
        "future_accuracy": float(metrics["future"]["accuracy"]),
        "past_accuracy": float(metrics["past"]["accuracy"]),
        "joint_exact": float(metrics["joint_exact"]),
        "future_ce": float(metrics["future"]["ce"]),
        "past_ce": float(metrics["past"]["ce"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--formal-score", type=Path, required=True)
    parser.add_argument("--formal-cases", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    expected = {
        args.checkpoint: EXPECTED_CHECKPOINT_SHA256,
        args.formal_score: EXPECTED_FORMAL_SCORE_SHA256,
        args.formal_cases: EXPECTED_FORMAL_CASES_SHA256,
    }
    for path, digest in expected.items():
        if not path.is_file() or _sha256(path) != digest:
            raise RuntimeError(f"Frozen artifact mismatch: {path}")

    gpu = _gpu_identity()
    formal = json.loads(args.formal_score.read_text())
    candidate_cases = json.loads(args.formal_cases.read_text())
    candidate = formal["candidate"]
    if candidate["arm"] != CANDIDATE_ARM:
        raise RuntimeError("Formal score is not the frozen P-GDN3-058 arm")

    config = build_config(
        arm=CANDIDATE_ARM,
        sequence_length=SEQUENCE_LENGTH,
        num_kv_pairs=NUM_KV_PAIRS,
        max_epochs=MAX_EPOCHS,
        batch_size=BATCH_SIZE,
    )
    set_determinism(SEED)
    _train_loader, test_loader = prepare_data(config.data)
    test_digest = dataset_hash(test_loader)
    if test_digest != candidate["data_hashes"]["test"]:
        raise RuntimeError("P-GDN3-058 test data hash drifted")

    model = make_model(config, CANDIDATE_ARM).cuda()
    checkpoint = torch.load(args.checkpoint, map_location="cuda", weights_only=True)
    if (
        checkpoint.get("arm") != CANDIDATE_ARM
        or checkpoint.get("carrier_arm") != CANDIDATE_ARM
    ):
        raise RuntimeError("Checkpoint arm metadata is inconsistent")
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    trained_hash = parameter_hash(model)
    if trained_hash != candidate["trained_parameter_hash"]:
        raise RuntimeError("Loaded trained parameter hash does not match formal score")
    parent_hash = parent_parameter_hash(model)

    gates = _cycle_gates(model)
    trained_gates = [gate.detach().float().cpu().flatten().tolist() for gate in gates]
    with torch.no_grad():
        for gate in gates:
            gate.zero_()
    if any(torch.count_nonzero(gate).item() for gate in gates):
        raise RuntimeError("Cycle gates did not become exactly zero")
    if parent_parameter_hash(model) != parent_hash:
        raise RuntimeError("Zeroing cycle gates changed a parent parameter")

    edge_off_metrics, edge_off_cases = evaluate(
        model,
        test_loader,
        sequence_length=SEQUENCE_LENGTH,
    )
    edge_off_swaps = wrong_key_swap_summary(edge_off_cases)
    candidate_swaps = formal["candidate_swap_summary"]
    transitions = transition_summary(candidate_cases, edge_off_cases)

    balanced = float(edge_off_metrics["balanced_accuracy"])
    errors = int(edge_off_swaps["errors"])
    if balanced >= 0.40 and errors <= 2_400:
        diagnosis = "read_time_cycle_reread_is_primary_damage"
    elif balanced <= 0.15 and errors >= 3_200:
        diagnosis = "training_coadaptation_already_collapsed_native_path"
    else:
        diagnosis = "mixed_training_and_read_time_damage"

    result = {
        "status": "complete",
        "plan": "P-DIAG-CYCLE-001",
        "gpu": gpu,
        "protocol": {
            "checkpoint_sha256": EXPECTED_CHECKPOINT_SHA256,
            "formal_score_sha256": EXPECTED_FORMAL_SCORE_SHA256,
            "formal_cases_sha256": EXPECTED_FORMAL_CASES_SHA256,
            "test_data_sha256": test_digest,
            "same_trained_weights": True,
            "only_intervention": "set all eight cycle gates to exact zero",
            "training": False,
        },
        "trained_parameter_hash": trained_hash,
        "parent_parameter_hash": parent_hash,
        "trained_cycle_gates": trained_gates,
        "formal_candidate": {
            "metrics": _metric_row(candidate["metrics"]),
            "swap_summary": candidate_swaps,
        },
        "same_weight_cycle_edge_off": {
            "metrics": _metric_row(edge_off_metrics),
            "swap_summary": edge_off_swaps,
        },
        "candidate_to_edge_off_transitions": transitions,
        "diagnosis": diagnosis,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
