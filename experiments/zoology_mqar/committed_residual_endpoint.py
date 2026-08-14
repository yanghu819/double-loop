from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

from experiments.zoology_mqar.gdn2_log_spd_endpoint import wrong_key_swap_summary
from experiments.zoology_mqar.length_scaling import run_arm


SEQUENCE_LENGTH = 1024
NUM_KV_PAIRS = 4
MAX_EPOCHS = 10
BATCH_SIZE = 32
EXPECTED_PARAMETERS = 661_584


def ratio(value: float, reference: float) -> float:
    if reference <= 0.0:
        raise ValueError("Cost reference must be positive")
    return value / reference


def load_cases(output_dir: Path, arm: str) -> list[dict[str, Any]]:
    return json.loads(
        (output_dir / "length_1024" / arm / "cases.json").read_text()
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-epochs", type=int, default=MAX_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    args = parser.parse_args()
    if args.max_epochs != MAX_EPOCHS or args.batch_size != BATCH_SIZE:
        raise ValueError("P-GDN3-037 has one fixed 10-epoch/batch32 endpoint")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    control_name = "future_seed_gdn2_runtime_control"
    candidate_name = "future_seed_committed_residual_gdn2"
    control = run_arm(
        arm="future_seed_gdn2",
        output_arm_name=control_name,
        sequence_length=SEQUENCE_LENGTH,
        num_kv_pairs=NUM_KV_PAIRS,
        output_dir=args.output_dir,
        max_epochs=MAX_EPOCHS,
        batch_size=BATCH_SIZE,
        save_checkpoint=True,
    )
    candidate = run_arm(
        arm=candidate_name,
        sequence_length=SEQUENCE_LENGTH,
        num_kv_pairs=NUM_KV_PAIRS,
        output_dir=args.output_dir,
        max_epochs=MAX_EPOCHS,
        batch_size=BATCH_SIZE,
        save_checkpoint=True,
    )

    control_swaps = wrong_key_swap_summary(load_cases(args.output_dir, control_name))
    candidate_swaps = wrong_key_swap_summary(
        load_cases(args.output_dir, candidate_name)
    )
    diagnostics = candidate["committed_residual"]
    rows = diagnostics["per_layer"]
    integrity_checks = {
        "exact_native_parent_initialization": (
            candidate["parent_init_parameter_hash"]
            == control["init_parameter_hash"]
        ),
        "matched_data_hashes": candidate["data_hashes"] == control["data_hashes"],
        "matched_warmup_batch": (
            candidate["warmup_batch_hash"] == control["warmup_batch_hash"]
        ),
        "equal_parameter_count": (
            control["parameters"] == candidate["parameters"] == EXPECTED_PARAMETERS
        ),
        "equal_state_and_scan_count": (
            diagnostics["new_parameters"] == 0
            and diagnostics["new_persistent_state_values"] == 0
            and diagnostics["logical_scans_per_layer"] == 1
        ),
    }
    activation_checks = {
        "exactly_two_active_layers": diagnostics["active_layers"] == 2,
        "native_futureseed_route_active": (
            candidate["future_seed"]["active_seed_routes"] == 1
        ),
        "beta_is_bounded_and_varied": all(
            0.0 <= row["beta_mean"] <= 1.0
            and row["beta_std"] > 1e-4
            and row["beta_board_std"] > 0.0
            and row["beta_token_std"] > 0.0
            for row in rows
        ),
        "write_target_gate_is_varied": all(
            0.0 <= row["write_gate_mean"] <= 1.0
            and row["write_gate_std"] > 1e-4
            for row in rows
        ),
        "committed_residual_is_nontrivial": all(
            math.isfinite(row["committed_residual_relative_rms"])
            and row["committed_residual_relative_rms"] >= 1e-3
            and row["committed_residual_board_std"] > 0.0
            for row in rows
        ),
        "explicit_recurrence_agrees": all(
            math.isfinite(row["explicit_terminal_relative_rms"])
            and row["explicit_terminal_relative_rms"] <= 0.05
            for row in rows
        ),
        "sampled_transition_is_bounded": all(
            math.isfinite(row["transition_spectral_norm_max"])
            and row["transition_spectral_norm_max"] <= 1.05
            for row in rows
        ),
        "terminal_state_is_finite_and_varies": all(
            math.isfinite(row["terminal_state_rms"])
            and 1e-4 <= row["terminal_state_rms"] <= 1e4
            and row["terminal_state_board_std"] > 1e-6
            for row in rows
        ),
    }

    control_metrics = control["metrics"]
    candidate_metrics = candidate["metrics"]
    quality_checks = {
        "balanced_at_least_0.35": candidate_metrics["balanced_accuracy"] >= 0.35,
        "balanced_gain_at_least_0.10": (
            candidate_metrics["balanced_accuracy"]
            - control_metrics["balanced_accuracy"]
            >= 0.10
        ),
        "future_gain_at_least_0.07": (
            candidate_metrics["future"]["accuracy"]
            - control_metrics["future"]["accuracy"]
            >= 0.07
        ),
        "past_gain_at_least_0.07": (
            candidate_metrics["past"]["accuracy"]
            - control_metrics["past"]["accuracy"]
            >= 0.07
        ),
        "joint_at_least_0.03": candidate_metrics["joint_exact"] >= 0.03,
        "joint_gain_at_least_0.03": (
            candidate_metrics["joint_exact"] - control_metrics["joint_exact"]
            >= 0.03
        ),
        "total_errors_reduced": candidate_swaps["errors"] < control_swaps["errors"],
        "wrong_key_swaps_down_at_least_0.05": (
            float(candidate_swaps["wrong_key_swap_fraction_of_errors"])
            <= float(control_swaps["wrong_key_swap_fraction_of_errors"]) - 0.05
        ),
    }
    cost_ratios = {
        "elapsed": ratio(
            float(candidate["elapsed_sec_including_validation"]),
            float(control["elapsed_sec_including_validation"]),
        ),
        "post_warm_wall": ratio(
            float(candidate["post_warm_arm_wall_sec_through_checkpoint"]),
            float(control["post_warm_arm_wall_sec_through_checkpoint"]),
        ),
        "warmed_step": ratio(
            float(candidate["warmed_step_benchmark"]["elapsed_sec"]),
            float(control["warmed_step_benchmark"]["elapsed_sec"]),
        ),
        "peak_allocation": ratio(
            float(candidate["peak_training_cuda_mem_bytes"]),
            float(control["peak_training_cuda_mem_bytes"]),
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
            "plan": "P-GDN3-037",
            "mechanism": "single coherent committed-residual target transition",
            "equation": "S'=DS+beta*k[(w*v)-k^TDS]^T",
            "run_order": [control_name, candidate_name],
            "sequence_length": SEQUENCE_LENGTH,
            "num_kv_pairs": NUM_KV_PAIRS,
            "epochs": MAX_EPOCHS,
            "batch_size": BATCH_SIZE,
            "seed": 123,
            "model": "D128/L2/H4/K32/V32 GDN2 projections plus native FutureSeed",
            "new_parameters": 0,
            "new_recurrent_state": 0,
            "logical_scans_per_layer": 1,
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
        "decision": "authorize_one_sudoku_transfer" if passed else "closed",
    }
    (args.output_dir / "comparison.json").write_text(
        json.dumps(comparison, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(comparison, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
