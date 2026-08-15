from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

from experiments.zoology_mqar.gdn2_log_spd_endpoint import wrong_key_swap_summary
from experiments.zoology_mqar.length_scaling import run_arm
from experiments.zoology_mqar.readonly_dual_plane_futureseed import (
    EXPECTED_PARAMETER_DELTA,
    EXPECTED_RECURRENT_STATE_DELTA,
    READONLY_SIDE_STATE_VALUES,
)


SEQUENCE_LENGTH = 1024
NUM_KV_PAIRS = 4
MAX_EPOCHS = 10
BATCH_SIZE = 32
SEED = 123
EXPECTED_NATIVE_PARAMETERS = 661_584
EXPECTED_STATE_VALUES = 4 * 32 * 32
EXPECTED_INIT_SHA256 = (
    "7402e46c48cbd47070d65262d1a1b62a32ac55a4d644a6fb9644b96b0914850f"
)
EXPECTED_CONTROL_SHA256 = (
    "df8ae1c212590666440f12029d60c74a6d09cbc059b23e58bcf864c00ae98854"
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _ratio(value: float, reference: float) -> float:
    if reference <= 0:
        raise ValueError("Cost reference must be positive")
    return value / reference


def _cases(root: Path, arm: str) -> list[dict[str, Any]]:
    return json.loads((root / "length_1024" / arm / "cases.json").read_text())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--frozen-control-root", type=Path, required=True)
    parser.add_argument("--frozen-control-score", type=Path, required=True)
    parser.add_argument("--matched-init-path", type=Path, required=True)
    parser.add_argument("--max-epochs", type=int, default=MAX_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    args = parser.parse_args()
    if args.max_epochs != MAX_EPOCHS or args.batch_size != BATCH_SIZE:
        raise ValueError("P-FS2-012 has one fixed 10-epoch/batch32 endpoint")
    if _sha256(args.matched_init_path) != EXPECTED_INIT_SHA256:
        raise RuntimeError("Frozen matched initialization hash drifted")
    if _sha256(args.frozen_control_score) != EXPECTED_CONTROL_SHA256:
        raise RuntimeError("Frozen P-REPRO-001 score hash drifted")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    frozen = json.loads(args.frozen_control_score.read_text())
    if (
        frozen["decision"] != "reproducible"
        or not frozen["registered_gate"]["passed"]
        or not frozen["trained_parameter_hash_exact_match"]
    ):
        raise RuntimeError("Frozen native replay is not a valid control")
    control = frozen["replay_b"]
    control_swaps = frozen["replay_b_swap_summary"]
    candidate_name = "future_seed_readonly_dual_plane_gdn2"
    candidate = run_arm(
        arm=candidate_name,
        sequence_length=SEQUENCE_LENGTH,
        num_kv_pairs=NUM_KV_PAIRS,
        output_dir=args.output_dir,
        max_epochs=MAX_EPOCHS,
        batch_size=BATCH_SIZE,
        save_checkpoint=True,
        matched_init_path=args.matched_init_path,
    )
    candidate_swaps = wrong_key_swap_summary(
        _cases(args.output_dir, candidate_name)
    )
    edge_off_swaps = wrong_key_swap_summary(
        json.loads(
            (
                args.output_dir
                / "length_1024"
                / candidate_name
                / "readonly_edge_off_cases.json"
            ).read_text()
        )
    )

    diagnostics = candidate["readonly_dual_plane"]
    rows = diagnostics["per_layer"]
    receiving = rows[1]
    parent_hash = control["init_parameter_hash"]
    integrity_checks = {
        "frozen_control_reproducible": frozen["registered_gate"]["passed"],
        "candidate_parent_matches_frozen_initialization": (
            candidate["parent_init_parameter_hash"] == parent_hash
            and candidate["init_parameter_hash"] == parent_hash
        ),
        "matched_data_hashes": candidate["data_hashes"] == control["data_hashes"],
        "matched_warmup_batch": (
            candidate["warmup_batch_hash"] == control["warmup_batch_hash"]
        ),
        "parameter_counts_exact_and_unchanged": (
            control["parameters"] == EXPECTED_NATIVE_PARAMETERS
            and candidate["parameters"] == EXPECTED_NATIVE_PARAMETERS
            and EXPECTED_PARAMETER_DELTA == 0
        ),
        "recurrent_state_counts_exact_and_unchanged": (
            control["recurrent_state_values_per_layer"] == EXPECTED_STATE_VALUES
            and candidate["recurrent_state_values_per_layer"]
            == EXPECTED_STATE_VALUES
            and EXPECTED_RECURRENT_STATE_DELTA == 0
        ),
        "one_official_scan_and_no_new_scan": (
            diagnostics["official_gdn2_scans_per_layer"] == 1
            and diagnostics["new_official_scans_per_layer"] == 0
        ),
        "one_bounded_side_state_only": (
            diagnostics["retained_readonly_side_state_values"]
            == READONLY_SIDE_STATE_VALUES
        ),
    }
    activation_checks = {
        "exactly_one_receiving_route": (
            diagnostics["active_readonly_routes"] == 1
            and rows[0]["route_present"] == 0.0
            and receiving["route_present"] == 1.0
            and receiving["route_enabled"] == 1.0
        ),
        "read_and_live_planes_finite_nonzero": all(
            math.isfinite(receiving[name]) and receiving[name] >= 1e-4
            for name in (
                "readonly_read_rms",
                "readonly_relative_rms",
                "live_output_rms",
                "live_terminal_rms",
                "readonly_state_rms",
            )
        ),
        "read_varies_by_board_and_token": (
            receiving["readonly_read_board_std"] >= 1e-5
            and receiving["readonly_read_token_std"] >= 1e-5
        ),
        "live_state_varies_by_board": (
            receiving["live_output_board_std"] >= 1e-5
            and receiving["live_terminal_board_std"] >= 1e-5
        ),
        "read_is_not_a_duplicate_live_plane": (
            math.isfinite(receiving["read_live_cosine"])
            and abs(receiving["read_live_cosine"]) < 0.995
        ),
        "readonly_state_remains_unchanged": (
            receiving["readonly_state_unchanged"] == 1.0
        ),
        "native_seed_gate_active": (
            candidate["future_seed"]["active_seed_routes"] == 1
            and candidate["future_seed"]["future_seed_gate"] is not None
            and candidate["future_seed"]["future_seed_gate"] >= 1e-4
        ),
    }

    cm = control["metrics"]
    pm = candidate["metrics"]
    em = candidate["readonly_dual_plane_edge_off_metrics"]
    quality_checks = {
        "balanced_at_least_0.60_and_gain_0.10": (
            pm["balanced_accuracy"] >= 0.60
            and pm["balanced_accuracy"] - cm["balanced_accuracy"] >= 0.10
        ),
        "future_at_least_0.60_and_gain_0.14": (
            pm["future"]["accuracy"] >= 0.60
            and pm["future"]["accuracy"] - cm["future"]["accuracy"] >= 0.14
        ),
        "past_regression_at_most_0.03": (
            pm["past"]["accuracy"] >= cm["past"]["accuracy"] - 0.03
        ),
        "joint_exact_at_least_0.10": pm["joint_exact"] >= 0.10,
        "total_errors_reduced_at_least_0.15": (
            candidate_swaps["errors"] <= 0.85 * control_swaps["errors"]
        ),
        "wrong_key_swap_count_reduced_at_least_0.20": (
            candidate_swaps["wrong_key_valid_value_swaps"]
            <= 0.80 * control_swaps["wrong_key_valid_value_swaps"]
        ),
        "readonly_edge_causes_future_gain": (
            pm["future"]["accuracy"] - em["future"]["accuracy"] >= 0.10
            and pm["balanced_accuracy"] - em["balanced_accuracy"] >= 0.05
            and candidate_swaps["errors"] < edge_off_swaps["errors"]
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
        "elapsed_below_1.25x": cost_ratios["elapsed"] < 1.25,
        "post_warm_wall_below_1.25x": cost_ratios["post_warm_wall"] < 1.25,
        "warmed_step_below_1.25x": cost_ratios["warmed_step"] < 1.25,
        "peak_allocation_below_1.10x": cost_ratios["peak_allocation"] < 1.10,
    }
    passed = (
        all(integrity_checks.values())
        and all(activation_checks.values())
        and all(quality_checks.values())
        and all(cost_checks.values())
    )
    comparison = {
        "status": "complete",
        "decision": "admit_one_sudoku_transfer" if passed else "closed",
        "protocol": {
            "plan": "P-FS2-012",
            "mechanism": "read-only dual-plane FutureSeed",
            "candidate_only_against_exact_reproducible_frozen_control": True,
            "sequence_length": SEQUENCE_LENGTH,
            "num_kv_pairs": NUM_KV_PAIRS,
            "epochs": MAX_EPOCHS,
            "batch_size": BATCH_SIZE,
            "seed": SEED,
            "new_parameters": EXPECTED_PARAMETER_DELTA,
            "new_recurrent_state_values": EXPECTED_RECURRENT_STATE_DELTA,
            "retained_readonly_side_state_values": READONLY_SIDE_STATE_VALUES,
            "new_official_scans_per_layer": 0,
            "sweep": None,
        },
        "frozen_control_score_path": str(args.frozen_control_score),
        "frozen_control_root": str(args.frozen_control_root),
        "matched_initialization": {
            "path": str(args.matched_init_path),
            "sha256": EXPECTED_INIT_SHA256,
            "parameter_hash": parent_hash,
        },
        "control": control,
        "control_swap_summary": control_swaps,
        "candidate": candidate,
        "candidate_swap_summary": candidate_swaps,
        "edge_off_swap_summary": edge_off_swaps,
        "cost_ratios": cost_ratios,
        "registered_gate": {
            "passed": passed,
            "integrity_checks": integrity_checks,
            "activation_checks": activation_checks,
            "quality_checks": quality_checks,
            "cost_checks": cost_checks,
        },
    }
    (args.output_dir / "comparison.json").write_text(
        json.dumps(comparison, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(comparison, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
