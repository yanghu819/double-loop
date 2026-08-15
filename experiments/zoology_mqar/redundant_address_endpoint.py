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
from experiments.zoology_mqar.gdn2_redundant_address import (
    EXPECTED_PARAMETER_DELTA,
    PARENT_STATE_VALUES,
    REDUNDANT_STATE_VALUES,
)
from experiments.zoology_mqar.length_scaling import run_arm


SEQUENCE_LENGTH = 1024
NUM_KV_PAIRS = 4
MAX_EPOCHS = 10
BATCH_SIZE = 32
SEED = 123
CANDIDATE_ARM = "future_seed_redundant_address_gdn2"
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
        raise ValueError("P-GDN3-053 has one fixed 10-epoch/batch32 endpoint")
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
        args.output_dir
        / "length_1024"
        / CANDIDATE_ARM
        / "cases.json"
    )
    candidate_cases = json.loads(candidate_cases_path.read_text())
    candidate_swaps = wrong_key_swap_summary(candidate_cases)
    transitions = transition_summary(control_cases, candidate_cases)
    diagnostics = candidate["redundant_address"]
    rows = diagnostics["per_layer"]

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
        "state_geometry_exact": (
            control["recurrent_state_values_per_layer"] == PARENT_STATE_VALUES
            and candidate["recurrent_state_values_per_layer"]
            == REDUNDANT_STATE_VALUES
        ),
        "single_official_scan_and_no_router": (
            diagnostics["official_scans_per_layer"] == 1
            and diagnostics["fixed_equal_read"]
            and not diagnostics["token_routing"]
        ),
    }
    finite_names = (
        "primary_q_secondary_q_cosine",
        "primary_k_secondary_k_cosine",
        "primary_address_contrast",
        "secondary_address_contrast",
        "secondary_address_token_std",
        "secondary_address_board_std",
        "primary_output_rms",
        "secondary_output_rms",
        "output_relative_disagreement",
        "output_cosine",
        "primary_state_rms",
        "secondary_state_rms",
        "state_relative_difference",
        "state_cosine_mean",
        "state_cosine_max",
    )
    activation_checks = {
        "two_layers_two_full_banks": (
            diagnostics["active_layers"] == 2
            and diagnostics["bank_count"] == 2
            and diagnostics["physical_heads"] == 8
            and diagnostics["state_values_per_layer"]
            == REDUNDANT_STATE_VALUES
        ),
        "all_diagnostics_finite": _all_finite(rows, finite_names),
        "addresses_independent": all(
            abs(row["primary_q_secondary_q_cosine"]) < 0.995
            and abs(row["primary_k_secondary_k_cosine"]) < 0.995
            and row["secondary_address_token_std"] >= 1e-4
            and row["secondary_address_board_std"] >= 1e-6
            for row in rows
        ),
        "both_reads_active_and_distinct": all(
            row["primary_output_rms"] >= 1e-4
            and row["secondary_output_rms"] >= 1e-4
            and row["output_relative_disagreement"] >= 1e-3
            for row in rows
        ),
        "both_states_active_and_distinct": all(
            row["primary_state_rms"] >= 1e-4
            and row["secondary_state_rms"] >= 1e-4
            and row["state_relative_difference"] >= 1e-3
            and row["state_cosine_max"] < 0.999
            for row in rows
        ),
        "native_futureseed_active": (
            candidate["future_seed"]["active_seed_routes"] == 1
        ),
    }

    control_metrics = control["metrics"]
    candidate_metrics = candidate["metrics"]
    quality_checks = {
        "balanced_at_least_0.65_and_gain_0.15": (
            candidate_metrics["balanced_accuracy"] >= 0.65
            and candidate_metrics["balanced_accuracy"]
            - control_metrics["balanced_accuracy"]
            >= 0.15
        ),
        "both_directions_at_least_0.60": (
            candidate_metrics["future"]["accuracy"] >= 0.60
            and candidate_metrics["past"]["accuracy"] >= 0.60
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
        "wrong_key_swaps_reduced_at_least_25_percent": (
            candidate_swaps["wrong_key_valid_value_swaps"]
            <= 0.75 * control_swaps["wrong_key_valid_value_swaps"]
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
        "elapsed_below_2x": cost_ratios["elapsed"] < 2.0,
        "post_warm_wall_below_2x": cost_ratios["post_warm_wall"] < 2.0,
        "warmed_step_below_2x": cost_ratios["warmed_step"] < 2.0,
        "peak_allocation_below_1.75x": cost_ratios["peak_allocation"] < 1.75,
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
            "plan": "P-GDN3-053",
            "mechanism": (
                "two independent full K32 address banks with shared payload/edit "
                "and fixed equal read"
            ),
            "candidate_only": True,
            "frozen_control": control["arm"],
            "model": "D128/L2/H4x2/K32/V32 official GDN2 plus native FutureSeed",
            "sequence_length": SEQUENCE_LENGTH,
            "num_kv_pairs": NUM_KV_PAIRS,
            "epochs": MAX_EPOCHS,
            "batch_size": BATCH_SIZE,
            "seed": SEED,
            "new_parameters": EXPECTED_PARAMETER_DELTA,
            "state_value_ratio": 2.0,
            "new_scans": 0,
            "token_router": None,
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
