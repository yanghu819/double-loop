from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

from experiments.zoology_mqar.gdn2_log_spd_endpoint import wrong_key_swap_summary
from experiments.zoology_mqar.lagged_commit_futureseed import (
    EXPECTED_NEW_PARAMETERS,
    EXPECTED_STATE_VALUES,
)
from experiments.zoology_mqar.length_scaling import run_arm


SEQUENCE_LENGTH = 1024
NUM_KV_PAIRS = 4
MAX_EPOCHS = 10
BATCH_SIZE = 32
SEED = 123
EXPECTED_NATIVE_PARAMETERS = 661_584
EXPECTED_CANDIDATE_PARAMETERS = EXPECTED_NATIVE_PARAMETERS + EXPECTED_NEW_PARAMETERS
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
        raise ValueError("P-GDN3-051 has one fixed 10-epoch/batch32 endpoint")
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
    candidate_name = "future_seed_lagged_commit_gdn2"
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
    edge_off_cases = json.loads(
        (
            args.output_dir
            / "length_1024"
            / candidate_name
            / "lag_edge_off_cases.json"
        ).read_text()
    )
    edge_off_swaps = wrong_key_swap_summary(edge_off_cases)

    diagnostics = candidate["lagged_commit"]
    rows = diagnostics["per_layer"]
    parent_hash = control["init_parameter_hash"]
    integrity_checks = {
        "frozen_control_reproducible": frozen["registered_gate"]["passed"],
        "candidate_parent_matches_frozen_initialization": (
            candidate["parent_init_parameter_hash"] == parent_hash
        ),
        "matched_data_hashes": candidate["data_hashes"] == control["data_hashes"],
        "matched_warmup_batch": (
            candidate["warmup_batch_hash"] == control["warmup_batch_hash"]
        ),
        "parameter_counts_exact": (
            control["parameters"] == EXPECTED_NATIVE_PARAMETERS
            and candidate["parameters"] == EXPECTED_CANDIDATE_PARAMETERS
            and candidate["parameters"] - control["parameters"]
            == EXPECTED_NEW_PARAMETERS
        ),
        "state_and_scan_counts_unchanged": (
            control["recurrent_state_values_per_layer"] == EXPECTED_STATE_VALUES
            and candidate["recurrent_state_values_per_layer"]
            == EXPECTED_STATE_VALUES
            and diagnostics["new_recurrent_state_values"] == 0
            and diagnostics["logical_scans_per_layer"] == 1
        ),
    }
    cosine_gain = sum(
        row["value_position_cosine_gain"] for row in rows
    ) / len(rows)
    activation_checks = {
        "both_layers_and_all_logits_finite": (
            diagnostics["active_layers"] == 2
            and diagnostics["finite_lag_logits"]
            and all(
                all(math.isfinite(value) for value in row["lag_mix"])
                for row in rows
            )
        ),
        "positive_causal_lag_is_learned": (
            diagnostics["positive_heads_at_least_0.02"] >= 4
            and diagnostics["global_lag_mix_mean"] >= 0.01
        ),
        "commit_key_moves_materially": all(
            row["committed_native_relative_rms"] >= 0.01 for row in rows
        ),
        "value_positions_align_with_preceding_key": cosine_gain >= 0.02,
        "token_zero_has_exact_identity": all(
            row["token0_identity_max_error"] == 0.0 for row in rows
        ),
        "terminal_states_finite_and_variable": all(
            math.isfinite(row["terminal_state_rms"])
            and row["terminal_state_rms"] >= 1e-4
            and row["terminal_state_board_std"] >= 1e-5
            for row in rows
        ),
        "native_future_seed_active": (
            candidate["future_seed"]["active_seed_routes"] == 1
            and candidate["future_seed"]["future_seed_gate"] is not None
            and candidate["future_seed"]["future_seed_gate"] >= 1e-4
        ),
    }

    cm = control["metrics"]
    pm = candidate["metrics"]
    em = candidate["lagged_commit_edge_off_metrics"]
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
        "lag_edge_causes_balanced_gain": (
            pm["balanced_accuracy"] - em["balanced_accuracy"] >= 0.05
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
        "elapsed_below_1.20x": cost_ratios["elapsed"] < 1.20,
        "post_warm_wall_below_1.20x": cost_ratios["post_warm_wall"] < 1.20,
        "warmed_step_below_1.20x": cost_ratios["warmed_step"] < 1.20,
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
            "plan": "P-GDN3-051",
            "mechanism": "causal lagged-address coherent commit",
            "candidate_only_against_exact_reproducible_frozen_control": True,
            "sequence_length": SEQUENCE_LENGTH,
            "num_kv_pairs": NUM_KV_PAIRS,
            "epochs": MAX_EPOCHS,
            "batch_size": BATCH_SIZE,
            "seed": SEED,
            "new_parameters": EXPECTED_NEW_PARAMETERS,
            "new_recurrent_state_values": 0,
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
        "mean_value_position_cosine_gain": cosine_gain,
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
