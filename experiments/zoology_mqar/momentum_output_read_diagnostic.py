from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import time
from pathlib import Path
from typing import Any

import torch
from zoology.data.utils import prepare_data
from zoology.utils import set_determinism

from experiments.zoology_mqar.canonical_address_companion_endpoint import (
    transition_summary,
)
from experiments.zoology_mqar.futureseed_directionality import dataset_hash
from experiments.zoology_mqar.gdn2_log_spd_endpoint import (
    wrong_key_swap_summary,
)
from experiments.zoology_mqar.length_scaling import (
    build_config,
    evaluate,
    make_model,
    parameter_hash,
)
from experiments.zoology_mqar.momentum_futureseed import (
    ZoologyMomentumDeltaFutureSeedMixer,
)
from experiments.zoology_mqar.momentum_futureseed_component_diagnostic import (
    _owner_topology,
)


SEQUENCE_LENGTH = 1024
NUM_KV_PAIRS = 4
MAX_EPOCHS = 10
BATCH_SIZE = 32
SEED = 123
CANDIDATE_ARM = "future_seed_momentum_delta"
EXPECTED_CHECKPOINT_SHA256 = (
    "13410aaad3be26fbdf07c7a12e1afa9fb38ced41a1939a519c02973be2b8508f"
)
EXPECTED_FORMAL_SCORE_SHA256 = (
    "7201d32834d3f9af4d1057750ba4eb62df492aa495eec770f9dad85f3d4f6e8a"
)
EXPECTED_FORMAL_CASES_SHA256 = (
    "f64b0ae0e45a65a8da2934ece8cfe208fedd835a373d79230320bdf945826f20"
)
MODES = ("native", "output_correction_off")


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
    expected = ("0", os.environ["EXPECTED_GPU_UUID"], os.environ["EXPECTED_GPU_NAME"])
    if (index, uuid, name) != expected:
        raise RuntimeError(f"GPU mismatch: {(index, uuid, name)} != {expected}")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("The diagnostic requires exactly one CUDA device")
    return {"index": 0, "uuid": uuid, "name": name}


def _metric_row(metrics: dict[str, Any]) -> dict[str, float]:
    return {
        "balanced_accuracy": float(metrics["balanced_accuracy"]),
        "future_accuracy": float(metrics["future"]["accuracy"]),
        "past_accuracy": float(metrics["past"]["accuracy"]),
        "joint_exact": float(metrics["joint_exact"]),
        "future_ce": float(metrics["future"]["ce"]),
        "past_ce": float(metrics["past"]["ce"]),
    }


def _mixers(model: torch.nn.Module) -> list[ZoologyMomentumDeltaFutureSeedMixer]:
    rows = [
        block.sequence_mixer
        for block in model.backbone.layers
        if isinstance(block.sequence_mixer, ZoologyMomentumDeltaFutureSeedMixer)
    ]
    if len(rows) != 2:
        raise RuntimeError(f"Expected two Momentum layers, found {len(rows)}")
    if not all(mixer.layer.use_output_correction for mixer in rows):
        raise RuntimeError("Frozen P059 must enable output correction")
    return rows


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
    formal_cases = json.loads(args.formal_cases.read_text())
    candidate = formal["candidate"]
    if candidate["arm"] != CANDIDATE_ARM:
        raise RuntimeError("Formal score is not the frozen P-GDN3-059 arm")

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
        raise RuntimeError("P-GDN3-059 test data hash drifted")

    model = make_model(config, CANDIDATE_ARM).cuda().eval()
    checkpoint = torch.load(args.checkpoint, map_location="cuda", weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    trained_hash = parameter_hash(model)
    if trained_hash != candidate["trained_parameter_hash"]:
        raise RuntimeError("Loaded trained parameter hash does not match formal score")
    mixers = _mixers(model)

    variants: dict[str, Any] = {}
    variant_cases: dict[str, list[dict[str, Any]]] = {}
    for mode in MODES:
        enabled = mode == "native"
        for mixer in mixers:
            mixer.layer.use_output_correction = enabled
        if parameter_hash(model) != trained_hash:
            raise RuntimeError("Diagnostic toggle changed model parameters")
        torch.cuda.synchronize()
        started = time.perf_counter()
        metrics, cases = evaluate(model, test_loader, sequence_length=SEQUENCE_LENGTH)
        torch.cuda.synchronize()
        variants[mode] = {
            "metrics": _metric_row(metrics),
            "swap_summary": wrong_key_swap_summary(cases),
            "owner_topology": _owner_topology(cases),
            "elapsed_sec": time.perf_counter() - started,
        }
        variant_cases[mode] = cases

    if variant_cases["native"] != formal_cases:
        raise RuntimeError("Native output-correction replay does not match formal cases")
    if variants["native"]["metrics"] != _metric_row(candidate["metrics"]):
        raise RuntimeError("Native output-correction metrics drifted")

    baseline = variants["native"]
    counterfactual = variants["output_correction_off"]
    metric_delta = {
        key: counterfactual["metrics"][key] - baseline["metrics"][key]
        for key in baseline["metrics"]
    }
    swap_reduction = (
        baseline["swap_summary"]["wrong_key_valid_value_swaps"]
        - counterfactual["swap_summary"]["wrong_key_valid_value_swaps"]
    )
    selection_checks = {
        "balanced_gain_at_least_0.005": metric_delta["balanced_accuracy"] >= 0.005,
        "wrong_key_swaps_reduced_at_least_20": swap_reduction >= 20,
        "future_regression_at_most_0.005": metric_delta["future_accuracy"] >= -0.005,
        "past_regression_at_most_0.005": metric_delta["past_accuracy"] >= -0.005,
        "joint_regression_at_most_0.01": metric_delta["joint_exact"] >= -0.01,
        "total_errors_do_not_increase": (
            counterfactual["swap_summary"]["errors"]
            <= baseline["swap_summary"]["errors"]
        ),
    }
    opens_successor = all(selection_checks.values())
    result = {
        "status": "complete",
        "plan": "P-DIAG-MOMREAD-001",
        "gpu": gpu,
        "protocol": {
            "training": False,
            "same_trained_weights": True,
            "checkpoint_sha256": EXPECTED_CHECKPOINT_SHA256,
            "formal_score_sha256": EXPECTED_FORMAL_SCORE_SHA256,
            "formal_cases_sha256": EXPECTED_FORMAL_CASES_SHA256,
            "test_data_sha256": test_digest,
            "intervention": (
                "disable only the trained q-Dk output correction in both P059 layers"
            ),
            "sweep": None,
        },
        "trained_parameter_hash": trained_hash,
        "variants": variants,
        "metric_delta_off_minus_native": metric_delta,
        "wrong_key_swap_reduction": swap_reduction,
        "native_to_off_transitions": transition_summary(
            formal_cases,
            variant_cases["output_correction_off"],
        ),
        "selection_checks": selection_checks,
        "opens_no_output_correction_successor": opens_successor,
        "diagnosis": (
            "output_read_correction_is_causal_residual_owner_bottleneck"
            if opens_successor
            else "output_read_correction_is_not_residual_owner_bottleneck"
        ),
        "next_branch": (
            "one_from_scratch_no_output_correction_successor"
            if opens_successor
            else "distinct_live_momentum_state_organization"
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
