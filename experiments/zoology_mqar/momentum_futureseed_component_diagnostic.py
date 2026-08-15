from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import time
from pathlib import Path
from types import MethodType
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
MODES = ("both", "state_only", "momentum_only", "none")


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


def _install_component_mask(model: torch.nn.Module) -> list[Any]:
    mixers = [
        block.sequence_mixer
        for block in model.backbone.layers
        if isinstance(
            block.sequence_mixer,
            ZoologyMomentumDeltaFutureSeedMixer,
        )
    ]
    if len(mixers) != 2:
        raise RuntimeError(f"Expected two momentum layers, found {len(mixers)}")

    for mixer in mixers:
        mixer._diagnostic_component_mode = "both"
        mixer._diagnostic_native_make_initial_state = mixer.make_initial_state

        def masked_make_initial_state(self, terminal_state: torch.Tensor) -> torch.Tensor:
            seed = self._diagnostic_native_make_initial_state(terminal_state)
            if seed.ndim != 5 or seed.shape[0] != 2:
                raise RuntimeError(f"Expected [2,B,H,K,V] seed, got {seed.shape}")
            mode = self._diagnostic_component_mode
            if mode == "both":
                return seed
            masked = seed.clone()
            if mode == "state_only":
                masked[1].zero_()
            elif mode == "momentum_only":
                masked[0].zero_()
            elif mode == "none":
                masked.zero_()
            else:
                raise RuntimeError(f"Unknown diagnostic mode: {mode}")
            return masked

        mixer.make_initial_state = MethodType(masked_make_initial_state, mixer)
    return mixers


def _owner_topology(cases: list[dict[str, Any]]) -> dict[str, Any]:
    errors = 0
    swaps = 0
    adjacent = 0
    same_direction = 0
    signed_rank_gaps: dict[int, int] = {}
    for case in cases:
        events = case["events"]
        write_order = sorted(
            range(len(events)),
            key=lambda index: events[index]["write_position"],
        )
        ranks = {event_index: rank for rank, event_index in enumerate(write_order)}
        for index, event in enumerate(events):
            if event["correct"]:
                continue
            errors += 1
            donors = [
                donor_index
                for donor_index, donor in enumerate(events)
                if donor_index != index and donor["target"] == event["prediction"]
            ]
            if not donors:
                continue
            donor_index = min(
                donors,
                key=lambda item: abs(
                    events[item]["write_position"] - event["write_position"]
                ),
            )
            donor = events[donor_index]
            rank_gap = ranks[donor_index] - ranks[index]
            swaps += 1
            adjacent += int(abs(rank_gap) == 1)
            same_direction += int(donor["direction"] == event["direction"])
            signed_rank_gaps[rank_gap] = signed_rank_gaps.get(rank_gap, 0) + 1
    return {
        "errors": errors,
        "wrong_key_swaps": swaps,
        "adjacent_write_rank_swaps": adjacent,
        "same_direction_swaps": same_direction,
        "signed_write_rank_gap_histogram": {
            str(key): value for key, value in sorted(signed_rank_gaps.items())
        },
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
    mixers = _install_component_mask(model)
    if parameter_hash(model) != trained_hash:
        raise RuntimeError("Installing the diagnostic mask changed model parameters")

    variants: dict[str, Any] = {}
    variant_cases: dict[str, list[dict[str, Any]]] = {}
    for mode in MODES:
        for mixer in mixers:
            mixer._diagnostic_component_mode = mode
        torch.cuda.synchronize()
        started = time.perf_counter()
        metrics, cases = evaluate(
            model,
            test_loader,
            sequence_length=SEQUENCE_LENGTH,
        )
        torch.cuda.synchronize()
        variants[mode] = {
            "metrics": _metric_row(metrics),
            "swap_summary": wrong_key_swap_summary(cases),
            "owner_topology": _owner_topology(cases),
            "elapsed_sec": time.perf_counter() - started,
        }
        variant_cases[mode] = cases

    if variant_cases["both"] != formal_cases:
        raise RuntimeError("Native [S,M] replay does not exactly match formal cases")
    if variants["both"]["metrics"] != _metric_row(candidate["metrics"]):
        raise RuntimeError("Native [S,M] replay metrics drifted")

    baseline_metrics = variants["both"]["metrics"]
    baseline_swaps = variants["both"]["swap_summary"]
    component_improvements = []
    for mode in MODES[1:]:
        metrics = variants[mode]["metrics"]
        swaps = variants[mode]["swap_summary"]
        if (
            metrics["balanced_accuracy"] - baseline_metrics["balanced_accuracy"]
            >= 0.005
            and baseline_swaps["wrong_key_valid_value_swaps"]
            - swaps["wrong_key_valid_value_swaps"]
            >= 20
            and metrics["future_accuracy"]
            >= baseline_metrics["future_accuracy"] - 0.005
            and metrics["past_accuracy"]
            >= baseline_metrics["past_accuracy"] - 0.005
            and metrics["joint_exact"] >= baseline_metrics["joint_exact"] - 0.01
            and swaps["errors"] <= baseline_swaps["errors"]
        ):
            component_improvements.append(mode)

    if component_improvements:
        diagnosis = "futureseed_component_transport_contributes_to_residual_tail"
        next_branch = "one_component_aware_futureseed_successor"
    else:
        diagnosis = "no_component_ablation_repairs_residual_tail"
        next_branch = "distinct_intralayer_owner_recurrence"

    transitions = {
        mode: transition_summary(formal_cases, variant_cases[mode])
        for mode in MODES[1:]
    }
    result = {
        "status": "complete",
        "plan": "P-DIAG-MOMFS-001",
        "gpu": gpu,
        "protocol": {
            "training": False,
            "same_trained_weights": True,
            "checkpoint_sha256": EXPECTED_CHECKPOINT_SHA256,
            "formal_score_sha256": EXPECTED_FORMAL_SCORE_SHA256,
            "formal_cases_sha256": EXPECTED_FORMAL_CASES_SHA256,
            "test_data_sha256": test_digest,
            "interventions": {
                "both": "native normalized and gated [S,M] FutureSeed",
                "state_only": "zero only transported momentum after native normalization",
                "momentum_only": "zero only transported state after native normalization",
                "none": "zero both transported components",
            },
            "selection_thresholds": {
                "balanced_gain": 0.005,
                "wrong_key_swap_reduction": 20,
                "per_direction_regression_max": 0.005,
                "joint_regression_max": 0.01,
                "total_errors_must_not_increase": True,
            },
        },
        "trained_parameter_hash": trained_hash,
        "variants": variants,
        "both_to_variant_transitions": transitions,
        "component_improvements": component_improvements,
        "diagnosis": diagnosis,
        "next_branch": next_branch,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
