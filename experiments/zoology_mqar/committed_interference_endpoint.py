from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

from experiments.zoology_mqar.committed_interference_credit import (
    AUXILIARY_WEIGHT,
    CAUSAL_WINDOW,
    EXPECTED_INFERENCE_SCAN_DELTA,
    EXPECTED_PARAMETER_DELTA,
    EXPECTED_STATE_DELTA,
    EXPECTED_TRAINING_CONV_DELTA,
)
from experiments.zoology_mqar.gdn2_log_spd_endpoint import wrong_key_swap_summary
from experiments.zoology_mqar.length_scaling import run_arm


SEQUENCE_LENGTH = 1024
NUM_KV_PAIRS = 4
MAX_EPOCHS = 10
BATCH_SIZE = 32
SEED = 123
EXPECTED_PARAMETERS = 661_584
EXPECTED_STATE_VALUES = 4_096
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
        raise ValueError("P-GDN3-052 has one fixed 10-epoch/batch32 endpoint")
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

    candidate_name = "future_seed_committed_interference_gdn2"
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

    diagnostics = candidate["committed_interference"]
    layers = diagnostics["layers"]
    integrity_checks = {
        "frozen_control_reproducible": frozen["registered_gate"]["passed"],
        "candidate_uses_frozen_initialization": (
            candidate["init_parameter_hash"] == control["init_parameter_hash"]
        ),
        "matched_data_hashes": candidate["data_hashes"] == control["data_hashes"],
        "matched_warmup_batch": (
            candidate["warmup_batch_hash"] == control["warmup_batch_hash"]
        ),
        "parameter_count_and_delta_exact": (
            control["parameters"]
            == candidate["parameters"]
            == EXPECTED_PARAMETERS
            and EXPECTED_PARAMETER_DELTA == 0
        ),
        "state_and_inference_graph_unchanged": (
            control["recurrent_state_values_per_layer"]
            == candidate["recurrent_state_values_per_layer"]
            == EXPECTED_STATE_VALUES
            and diagnostics["inference_parameter_delta"]
            == EXPECTED_PARAMETER_DELTA
            and diagnostics["inference_state_delta"] == EXPECTED_STATE_DELTA
            and diagnostics["inference_scan_delta"]
            == EXPECTED_INFERENCE_SCAN_DELTA
            and diagnostics["training_short_conv_delta"]
            == EXPECTED_TRAINING_CONV_DELTA
        ),
    }
    activation_checks = {
        "two_active_layers": diagnostics["active_layers"] == 2,
        "fixed_target_free_causal_credit": (
            diagnostics["auxiliary_weight"] == AUXILIARY_WEIGHT
            and diagnostics["causal_window"] == CAUSAL_WINDOW
            and diagnostics["input_detached"]
            and diagnostics["committed_edit_detached"]
            and not diagnostics["uses_targets"]
        ),
        "credit_finite_nontrivial": (
            math.isfinite(diagnostics["unweighted_loss"])
            and 1e-5 <= diagnostics["unweighted_loss"] <= 10.0
            and math.isfinite(diagnostics["weighted_loss"])
            and diagnostics["weighted_loss"] >= 1e-6
        ),
        "exact_committed_edits_active_and_variable": all(
            math.isfinite(row["committed_edit_rms"])
            and row["committed_edit_rms"] >= 1e-4
            and row["surprise_cv"] >= 0.05
            and row["surprise_board_std"] > 0
            for row in layers
        ),
        "collision_tail_is_present": all(
            math.isfinite(row["relative_interference_mean"])
            and row["above_random_fraction"] >= 0.05
            and row["local_mass_mean"] > 0
            and row["valid_fraction"] > 0.99
            for row in layers
        ),
        "native_futureseed_active": (
            candidate["future_seed"]["active_seed_routes"] == 1
        ),
    }

    cm = control["metrics"]
    pm = candidate["metrics"]
    quality_checks = {
        "balanced_at_least_0.60_and_gain_0.10": (
            pm["balanced_accuracy"] >= 0.60
            and pm["balanced_accuracy"] - cm["balanced_accuracy"] >= 0.10
        ),
        "future_at_least_0.58_and_gain_0.10": (
            pm["future"]["accuracy"] >= 0.58
            and pm["future"]["accuracy"] - cm["future"]["accuracy"] >= 0.10
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
        "elapsed_below_1.35x": cost_ratios["elapsed"] < 1.35,
        "post_warm_wall_below_1.35x": cost_ratios["post_warm_wall"] < 1.35,
        "warmed_step_below_1.35x": cost_ratios["warmed_step"] < 1.35,
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
        "decision": "admit_one_sudoku_transfer" if passed else "closed",
        "protocol": {
            "plan": "P-GDN3-052",
            "mechanism": "committed-edit weighted causal interference credit",
            "candidate_only_against_exact_reproducible_frozen_control": True,
            "sequence_length": SEQUENCE_LENGTH,
            "num_kv_pairs": NUM_KV_PAIRS,
            "epochs": MAX_EPOCHS,
            "batch_size": BATCH_SIZE,
            "seed": SEED,
            "auxiliary_weight": AUXILIARY_WEIGHT,
            "causal_window": CAUSAL_WINDOW,
            "new_parameters": EXPECTED_PARAMETER_DELTA,
            "new_recurrent_state_values": EXPECTED_STATE_DELTA,
            "new_inference_scans_per_layer": EXPECTED_INFERENCE_SCAN_DELTA,
            "training_short_conv_delta": EXPECTED_TRAINING_CONV_DELTA,
            "sweep": None,
        },
        "frozen_control_score_path": str(args.frozen_control_score),
        "frozen_control_root": str(args.frozen_control_root),
        "matched_initialization": {
            "path": str(args.matched_init_path),
            "sha256": EXPECTED_INIT_SHA256,
            "parameter_hash": control["init_parameter_hash"],
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
