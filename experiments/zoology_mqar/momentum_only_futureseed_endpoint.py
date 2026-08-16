from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

from experiments.zoology_mqar.canonical_address_companion_endpoint import (
    transition_summary,
)
from experiments.zoology_mqar.gdn2_log_spd_endpoint import (
    wrong_key_swap_summary,
)
from experiments.zoology_mqar.length_scaling import run_arm
from experiments.zoology_mqar.momentum_futureseed import (
    EXPECTED_MDN_SHA,
    EXPECTED_STATE_VALUES_PER_LAYER,
)
from experiments.zoology_mqar.momentum_only_futureseed import (
    FULL_STATE_VALUES_PER_ROUTE,
    MOMENTUM_PAYLOAD_VALUES_PER_ROUTE,
)


SEQUENCE_LENGTH = 1024
NUM_KV_PAIRS = 4
MAX_EPOCHS = 10
BATCH_SIZE = 32
CANDIDATE_ARM = "future_seed_momentum_only"
EXPECTED_MATCHED_INIT_SHA256 = (
    "7402e46c48cbd47070d65262d1a1b62a32ac55a4d644a6fb9644b96b0914850f"
)
EXPECTED_FROZEN_SCORE_SHA256 = (
    "7201d32834d3f9af4d1057750ba4eb62df492aa495eec770f9dad85f3d4f6e8a"
)
EXPECTED_FROZEN_CASES_SHA256 = (
    "f64b0ae0e45a65a8da2934ece8cfe208fedd835a373d79230320bdf945826f20"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _finite(value: Any) -> bool:
    return value is not None and math.isfinite(float(value))


def _ratio(candidate: dict[str, Any], control: dict[str, Any], key: str) -> float:
    return float(candidate[key]) / float(control[key])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--matched-init", type=Path, required=True)
    parser.add_argument("--frozen-score", type=Path, required=True)
    parser.add_argument("--frozen-cases", type=Path, required=True)
    parser.add_argument("--max-epochs", type=int, default=MAX_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    args = parser.parse_args()
    if args.max_epochs != MAX_EPOCHS or args.batch_size != BATCH_SIZE:
        raise ValueError("P-FS2-016 fixes one 10-epoch batch32 endpoint")

    expected_hashes = {
        args.matched_init: EXPECTED_MATCHED_INIT_SHA256,
        args.frozen_score: EXPECTED_FROZEN_SCORE_SHA256,
        args.frozen_cases: EXPECTED_FROZEN_CASES_SHA256,
    }
    for path, expected in expected_hashes.items():
        if not path.is_file() or _sha256(path) != expected:
            raise RuntimeError(f"Frozen artifact drifted: {path}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    frozen = json.loads(args.frozen_score.read_text())
    control = frozen["candidate"]
    control_cases = json.loads(args.frozen_cases.read_text())
    control_swaps = frozen["candidate_swap_summary"]

    candidate = run_arm(
        arm=CANDIDATE_ARM,
        sequence_length=SEQUENCE_LENGTH,
        num_kv_pairs=NUM_KV_PAIRS,
        output_dir=args.output_dir,
        max_epochs=MAX_EPOCHS,
        batch_size=BATCH_SIZE,
        save_checkpoint=True,
        matched_init_path=args.matched_init,
    )
    cases_path = (
        args.output_dir / "length_1024" / CANDIDATE_ARM / "cases.json"
    )
    candidate_cases = json.loads(cases_path.read_text())
    candidate_swaps = wrong_key_swap_summary(candidate_cases)
    diagnostics = candidate["momentum_delta"]
    routes = diagnostics["per_route"]
    producer = routes[0]
    receiver = routes[1]

    integrity_checks = {
        "external_source_exact": diagnostics["external_sha"] == EXPECTED_MDN_SHA,
        "shared_parent_mapping_exact": (
            diagnostics["matched_parent"] is not None
            and diagnostics["matched_parent"]["source_hash"]
            == diagnostics["matched_parent"]["loaded_hash"]
            and candidate["parent_init_parameter_hash"]
            == diagnostics["matched_parent"]["loaded_hash"]
        ),
        "matched_data_and_warmup": (
            candidate["data_hashes"] == control["data_hashes"]
            and candidate["warmup_batch_hash"] == control["warmup_batch_hash"]
        ),
        "parameters_scans_and_recurrent_state_unchanged": (
            candidate["parameters"] == control["parameters"]
            and candidate["recurrent_state_values_per_layer"]
            == EXPECTED_STATE_VALUES_PER_LAYER
            and diagnostics["parameter_delta_vs_p059"] == 0
            and diagnostics["persistent_state_delta_vs_p059"] == 0
            and diagnostics["scan_delta_vs_p059"] == 0
        ),
        "transport_exactly_halved": (
            diagnostics["full_state_values_per_route"]
            == FULL_STATE_VALUES_PER_ROUTE
            and diagnostics["transported_values_per_route"]
            == MOMENTUM_PAYLOAD_VALUES_PER_ROUTE
            and diagnostics["transport_ratio"] == 0.5
        ),
    }
    activation_checks = {
        "two_momentum_layers_and_one_seed_route": (
            diagnostics["active_layers"] == 2
            and diagnostics["active_futureseed_routes"] == 1
            and producer["pack_calls"] > 0
        ),
        "only_momentum_is_seeded": (
            receiver["seed_state_rms"] == 0.0
            and _finite(receiver["seed_momentum_rms"])
            and receiver["seed_momentum_rms"] >= 1e-4
            and _finite(producer["payload_rms"])
            and producer["payload_rms"] >= 1e-4
        ),
        "state_and_momentum_bounded": all(
            _finite(row["state_rms"])
            and _finite(row["momentum_rms"])
            and row["state_rms"] >= 1e-4
            and row["momentum_rms"] >= 1e-4
            and 0.01 <= row["momentum_to_state_rms"] <= 20.0
            for row in diagnostics["per_layer"]
        ),
    }

    cm = control["metrics"]
    mm = candidate["metrics"]
    quality_checks = {
        "balanced_quality_equivalent": (
            mm["balanced_accuracy"] >= 0.94
            and mm["balanced_accuracy"] >= cm["balanced_accuracy"] - 0.005
        ),
        "both_directions_within_0.005": (
            mm["future"]["accuracy"] >= cm["future"]["accuracy"] - 0.005
            and mm["past"]["accuracy"] >= cm["past"]["accuracy"] - 0.005
        ),
        "joint_within_0.01": (
            mm["joint_exact"] >= 0.81
            and mm["joint_exact"] >= cm["joint_exact"] - 0.01
        ),
        "errors_in_equivalence_budget": candidate_swaps["errors"] <= 243,
        "wrong_key_swaps_in_equivalence_budget": (
            candidate_swaps["wrong_key_valid_value_swaps"] <= 171
        ),
    }
    cost_ratios = {
        "elapsed": _ratio(
            candidate, control, "elapsed_sec_including_validation"
        ),
        "post_warm_wall": _ratio(
            candidate, control, "post_warm_arm_wall_sec_through_checkpoint"
        ),
        "warmed_step": (
            candidate["warmed_step_benchmark"]["elapsed_sec"]
            / control["warmed_step_benchmark"]["elapsed_sec"]
        ),
        "peak_allocation": _ratio(
            candidate, control, "peak_training_cuda_mem_bytes"
        ),
    }
    cost_checks = {
        "elapsed_below_1.05x": cost_ratios["elapsed"] < 1.05,
        "post_warm_wall_below_1.05x": cost_ratios["post_warm_wall"] < 1.05,
        "warmed_step_below_1.05x": cost_ratios["warmed_step"] < 1.05,
        "peak_allocation_below_1.02x": cost_ratios["peak_allocation"] < 1.02,
    }
    passed = all(
        all(group.values())
        for group in (
            integrity_checks,
            activation_checks,
            quality_checks,
            cost_checks,
        )
    )
    comparison = {
        "status": "complete",
        "decision": "pareto_futureseed" if passed else "closed",
        "protocol": {
            "plan": "P-FS2-016",
            "mechanism": "momentum-only cross-layer FutureSeed transport",
            "candidate_only": True,
            "frozen_control": "future_seed_momentum_delta",
            "sequence_length": SEQUENCE_LENGTH,
            "num_kv_pairs": NUM_KV_PAIRS,
            "epochs": MAX_EPOCHS,
            "batch_size": BATCH_SIZE,
            "seed": 123,
            "sweep": None,
        },
        "frozen_artifacts": {
            "score": str(args.frozen_score),
            "score_sha256": _sha256(args.frozen_score),
            "cases": str(args.frozen_cases),
            "cases_sha256": _sha256(args.frozen_cases),
            "matched_init": str(args.matched_init),
            "matched_init_sha256": _sha256(args.matched_init),
        },
        "control": control,
        "candidate": candidate,
        "control_swap_summary": control_swaps,
        "candidate_swap_summary": candidate_swaps,
        "paired_error_transitions": transition_summary(
            control_cases, candidate_cases
        ),
        "cost_ratios": cost_ratios,
        "registered_gate": {
            "integrity_checks": integrity_checks,
            "activation_checks": activation_checks,
            "quality_checks": quality_checks,
            "cost_checks": cost_checks,
            "passed": passed,
        },
    }
    output_path = args.output_dir / "comparison.json"
    output_path.write_text(json.dumps(comparison, indent=2, sort_keys=True) + "\n")
    print(json.dumps(comparison, indent=2, sort_keys=True))
    raise SystemExit(0 if passed else 2)


if __name__ == "__main__":
    main()
