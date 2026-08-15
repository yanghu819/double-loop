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
from experiments.zoology_mqar.gdn2_pair_event import (
    EXPECTED_PARAMETER_DELTA,
    PAIR_RANK,
)
from experiments.zoology_mqar.gdn2_log_spd_endpoint import (
    wrong_key_swap_summary,
)
from experiments.zoology_mqar.length_scaling import run_arm


SEQUENCE_LENGTH = 1024
NUM_KV_PAIRS = 4
MAX_EPOCHS = 10
BATCH_SIZE = 32
SEED = 123
CANDIDATE_ARM = "future_seed_pair_event_gdn2"
EXPECTED_MATCHED_INIT_SHA256 = (
    "7402e46c48cbd47070d65262d1a1b62a32ac55a4d644a6fb9644b96b0914850f"
)
EXPECTED_FROZEN_SCORE_SHA256 = (
    "df8ae1c212590666440f12029d60c74a6d09cbc059b23e58bcf864c00ae98854"
)
EXPECTED_FROZEN_CASES_SHA256 = (
    "2322ddc61dac4eb1819c6cac160b3a280c17c2bb0dfdde1fae96a6858b9b32bd"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _ratio(value: float, reference: float) -> float:
    if reference <= 0:
        raise ValueError("Cost reference must be positive")
    return value / reference


def _all_finite(rows: list[dict[str, Any]], names: tuple[str, ...]) -> bool:
    return all(
        all(math.isfinite(float(row[name])) for name in names)
        for row in rows
    )


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
        raise ValueError("P-GDN3-057 has one fixed 10-epoch/batch32 endpoint")

    expected_hashes = {
        args.matched_init: EXPECTED_MATCHED_INIT_SHA256,
        args.frozen_score: EXPECTED_FROZEN_SCORE_SHA256,
        args.frozen_cases: EXPECTED_FROZEN_CASES_SHA256,
    }
    for path, expected in expected_hashes.items():
        if not path.is_file():
            raise FileNotFoundError(path)
        actual = _sha256(path)
        if actual != expected:
            raise RuntimeError(
                f"Frozen artifact drifted: {path} {actual} != {expected}"
            )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    frozen = json.loads(args.frozen_score.read_text())
    control = frozen["replay_b"]
    control_swaps = frozen["replay_b_swap_summary"]
    control_cases = json.loads(args.frozen_cases.read_text())
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
    candidate_cases_path = (
        args.output_dir / "length_1024" / CANDIDATE_ARM / "cases.json"
    )
    candidate_cases = json.loads(candidate_cases_path.read_text())
    candidate_swaps = wrong_key_swap_summary(candidate_cases)
    transitions = transition_summary(control_cases, candidate_cases)
    diagnostics = candidate["pair_event"]
    rows = diagnostics["layers"]

    integrity_checks = {
        "candidate_parent_matches_frozen_initialization": (
            candidate["parent_init_parameter_hash"]
            == control["init_parameter_hash"]
        ),
        "matched_data_hashes": candidate["data_hashes"] == control["data_hashes"],
        "matched_warmup_batch": (
            candidate["warmup_batch_hash"] == control["warmup_batch_hash"]
        ),
        "parameter_delta_exact": (
            candidate["parameters"] - control["parameters"]
            == EXPECTED_PARAMETER_DELTA
        ),
        "main_state_geometry_unchanged": (
            candidate["recurrent_state_values_per_layer"]
            == control["recurrent_state_values_per_layer"]
        ),
        "one_main_scan_one_pair_event_path": (
            diagnostics["official_gdn2_scans_per_layer"] == 1
            and diagnostics["pair_rank"] == PAIR_RANK
            and diagnostics["persistent_state_delta"] == 0
        ),
    }
    finite_names = (
        "gate_abs_mean",
        "gate_abs_min",
        "gate_abs_max",
        "pair_feature_rms",
        "raw_event_rms",
        "event_delta_rms",
        "event_delta_relative_rms",
        "event_delta_board_std",
        "event_delta_token_std",
        "current_feature_rms",
        "previous_feature_rms",
    )
    activation_checks = {
        "all_eight_paths_active": (
            diagnostics["active_layers"] == 2
            and diagnostics["total_paths"] == 8
            and diagnostics["active_paths"] == 8
            and all(row["gate_abs_min"] >= 1e-3 for row in rows)
        ),
        "all_diagnostics_finite": _all_finite(rows, finite_names),
        "pair_event_materially_used_and_bounded": all(
            0.02 <= row["event_delta_relative_rms"] <= 0.35
            and row["event_delta_board_std"] >= 1e-4
            and row["event_delta_token_std"] >= 1e-4
            and row["current_feature_rms"] >= 1e-4
            and row["previous_feature_rms"] >= 1e-4
            for row in rows
        ),
        "native_futureseed_active": (
            candidate["future_seed"]["active_seed_routes"] == 1
        ),
    }

    control_metrics = control["metrics"]
    candidate_metrics = candidate["metrics"]
    quality_checks = {
        "balanced_at_least_0.65_and_gain_0.10": (
            candidate_metrics["balanced_accuracy"] >= 0.65
            and candidate_metrics["balanced_accuracy"]
            - control_metrics["balanced_accuracy"]
            >= 0.10
        ),
        "both_directions_at_least_0.62": (
            candidate_metrics["future"]["accuracy"] >= 0.62
            and candidate_metrics["past"]["accuracy"] >= 0.62
        ),
        "joint_at_least_0.15_and_gain_0.10": (
            candidate_metrics["joint_exact"] >= 0.15
            and candidate_metrics["joint_exact"]
            - control_metrics["joint_exact"]
            >= 0.10
        ),
        "total_errors_reduced_at_least_20_percent": (
            candidate_swaps["errors"] <= 0.80 * control_swaps["errors"]
        ),
        "swap_fraction_reduced_at_least_0.10": (
            control_swaps["wrong_key_swap_fraction_of_errors"]
            - candidate_swaps["wrong_key_swap_fraction_of_errors"]
            >= 0.10
        ),
    }
    cost_ratios = {
        "elapsed": _ratio(
            candidate["elapsed_sec_including_validation"],
            control["elapsed_sec_including_validation"],
        ),
        "post_warm_wall": _ratio(
            candidate["post_warm_arm_wall_sec_through_checkpoint"],
            control["post_warm_arm_wall_sec_through_checkpoint"],
        ),
        "warmed_step": _ratio(
            candidate["warmed_step_benchmark"]["elapsed_sec"],
            control["warmed_step_benchmark"]["elapsed_sec"],
        ),
        "peak_allocation": _ratio(
            candidate["peak_training_cuda_mem_bytes"],
            control["peak_training_cuda_mem_bytes"],
        ),
    }
    cost_checks = {
        "elapsed_below_1.50x": cost_ratios["elapsed"] < 1.50,
        "post_warm_wall_below_1.50x": cost_ratios["post_warm_wall"] < 1.50,
        "warmed_step_below_1.50x": cost_ratios["warmed_step"] < 1.50,
        "peak_allocation_below_1.25x": cost_ratios["peak_allocation"] < 1.25,
    }
    passed = (
        all(integrity_checks.values())
        and all(activation_checks.values())
        and all(quality_checks.values())
        and all(cost_checks.values())
    )
    comparison = {
        "status": "complete",
        "protocol": {
            "plan": "P-GDN3-057",
            "mechanism": "Raven-style multiplicative local event formation before native GDN2",
            "candidate_only": True,
            "frozen_control": control["arm"],
            "model": "D128/L2/H4/K32/V32 native GDN2+FutureSeed plus rank64 pair-event encoder",
            "sequence_length": SEQUENCE_LENGTH,
            "num_kv_pairs": NUM_KV_PAIRS,
            "pair_rank": PAIR_RANK,
            "epochs": MAX_EPOCHS,
            "batch_size": BATCH_SIZE,
            "seed": SEED,
            "new_parameters": EXPECTED_PARAMETER_DELTA,
            "new_persistent_state": 0,
            "new_main_recurrent_scans": 0,
            "local_pair_event_paths_per_layer": 1,
            "sweep": None,
        },
        "frozen_artifacts": {
            "matched_init": str(args.matched_init),
            "matched_init_sha256": EXPECTED_MATCHED_INIT_SHA256,
            "score": str(args.frozen_score),
            "score_sha256": EXPECTED_FROZEN_SCORE_SHA256,
            "cases": str(args.frozen_cases),
            "cases_sha256": EXPECTED_FROZEN_CASES_SHA256,
        },
        "control": control,
        "control_swap_summary": control_swaps,
        "candidate": candidate,
        "candidate_swap_summary": candidate_swaps,
        "paired_error_transitions": transitions,
        "cost_ratios": cost_ratios,
        "registered_gate": {
            "passed": passed,
            "integrity_checks": integrity_checks,
            "activation_checks": activation_checks,
            "quality_checks": quality_checks,
            "cost_checks": cost_checks,
        },
        "decision": "admit_one_sudoku_transfer" if passed else "closed",
    }
    (args.output_dir / "comparison.json").write_text(
        json.dumps(comparison, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(comparison, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
