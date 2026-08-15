from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

from experiments.zoology_mqar.gdn2_log_spd_endpoint import wrong_key_swap_summary
from experiments.zoology_mqar.gdn2_sparse_delta_slots import (
    EXPECTED_NEW_PARAMETERS,
    MAIN_STATE_VALUES,
    SLOT_COUNT,
    SPARSE_STATE_VALUES,
)
from experiments.zoology_mqar.length_scaling import run_arm


SEQUENCE_LENGTH = 1024
NUM_KV_PAIRS = 4
MAX_EPOCHS = 10
BATCH_SIZE = 32
SEED = 123
EXPECTED_NATIVE_PARAMETERS = 661_584
EXPECTED_CANDIDATE_PARAMETERS = (
    EXPECTED_NATIVE_PARAMETERS + EXPECTED_NEW_PARAMETERS
)
EXPECTED_INIT_SHA256 = (
    "7402e46c48cbd47070d65262d1a1b62a32ac55a4d644a6fb9644b96b0914850f"
)
EXPECTED_CONTROL_SHA256 = (
    "df8ae1c212590666440f12029d60c74a6d09cbc059b23e58bcf864c00ae98854"
)


def _sha256(path: Path) -> str:
    import hashlib

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
        raise ValueError("P-GDN3-050 has one fixed 10-epoch/batch32 endpoint")
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
    candidate_name = "future_seed_sparse_delta_slot_gdn2"
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

    diagnostics = candidate["sparse_delta_slots"]
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
        "state_counts_exact": (
            control["recurrent_state_values_per_layer"] == MAIN_STATE_VALUES
            and candidate["recurrent_state_values_per_layer"]
            == MAIN_STATE_VALUES + SPARSE_STATE_VALUES
        ),
        "official_scan_counts_exact": (
            diagnostics["official_gdn2_scans_per_layer"] == 1
            and diagnostics["official_gsa_scans_per_layer"] == 1
        ),
    }
    activation_checks = {
        "two_active_layers_and_one_sparse_seed_route": (
            diagnostics["active_layers"] == 2
            and diagnostics["sparse_futureseed_routes"] == 1
        ),
        "all_slots_used_with_bounded_concentration": all(
            row["active_slots"] == SLOT_COUNT
            and row["slot_usage_entropy"] >= 0.50
            and row["slot_usage_max"] <= 0.30
            for row in rows
        ),
        "committed_edits_and_sparse_states_finite_nonzero": all(
            math.isfinite(row[name]) and row[name] > 0
            for row in rows
            for name in (
                "committed_edit_rms",
                "sparse_output_relative_rms",
                "sparse_output_token_std",
                "sparse_output_board_std",
                "sparse_key_state_rms",
                "sparse_value_state_rms",
                "sparse_dense_rms",
                "sparse_dense_board_std",
            )
        ),
        "local_read_gates_active": all(
            row["local_read_gate_abs"] >= 1e-4 for row in rows
        ),
        "receiving_sparse_seed_gate_active": (
            rows[0]["sparse_seed_gate_abs"] is None
            and rows[1]["sparse_seed_gate_abs"] is not None
            and rows[1]["sparse_seed_gate_abs"] >= 1e-4
        ),
        "slot_keys_remain_separated": all(
            row["slot_key_abs_cosine_mean"] < 0.90 for row in rows
        ),
        "routing_transition_bounded": all(
            0.0 <= row["slot_s_min"] <= row["slot_s_max"] <= 1.0
            and row["slot_g_max"] <= 0.0
            for row in rows
        ),
        "native_futureseed_active": candidate["future_seed"]["active_seed_routes"] == 1,
    }

    cm = control["metrics"]
    pm = candidate["metrics"]
    quality_checks = {
        "balanced_at_least_0.85_and_gain_0.10": (
            pm["balanced_accuracy"] >= 0.85
            and pm["balanced_accuracy"] - cm["balanced_accuracy"] >= 0.10
        ),
        "future_and_past_each_at_least_0.82": (
            pm["future"]["accuracy"] >= 0.82
            and pm["past"]["accuracy"] >= 0.82
        ),
        "joint_exact_at_least_0.60": pm["joint_exact"] >= 0.60,
        "total_errors_reduced": candidate_swaps["errors"] < control_swaps["errors"],
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
        "elapsed_below_2.0x": cost_ratios["elapsed"] < 2.0,
        "post_warm_wall_below_2.0x": cost_ratios["post_warm_wall"] < 2.0,
        "warmed_step_below_2.0x": cost_ratios["warmed_step"] < 2.0,
        "peak_allocation_below_1.60x": cost_ratios["peak_allocation"] < 1.60,
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
            "plan": "P-GDN3-050",
            "mechanism": "sparse committed-delta pair slots",
            "candidate_only_against_exact_reproducible_frozen_control": True,
            "sequence_length": SEQUENCE_LENGTH,
            "num_kv_pairs": NUM_KV_PAIRS,
            "epochs": MAX_EPOCHS,
            "batch_size": BATCH_SIZE,
            "seed": SEED,
            "slot_count": SLOT_COUNT,
            "new_parameters": EXPECTED_NEW_PARAMETERS,
            "new_state_values_per_layer": SPARSE_STATE_VALUES,
            "new_official_scans_per_layer": 1,
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
