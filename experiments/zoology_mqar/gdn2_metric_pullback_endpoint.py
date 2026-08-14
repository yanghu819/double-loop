from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from experiments.zoology_mqar.gdn2_log_spd_endpoint import wrong_key_swap_summary
from experiments.zoology_mqar.length_scaling import run_arm


SEQUENCE_LENGTH = 1024
NUM_KV_PAIRS = 4
MAX_EPOCHS = 10
BATCH_SIZE = 32


def _ratio(value: float, reference: float) -> float:
    if reference <= 0:
        raise ValueError("Cost reference must be positive")
    return value / reference


def _cases(root: Path, arm: str) -> list[dict[str, Any]]:
    return json.loads((root / "length_1024" / arm / "cases.json").read_text())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-epochs", type=int, default=MAX_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    args = parser.parse_args()
    if args.max_epochs != MAX_EPOCHS or args.batch_size != BATCH_SIZE:
        raise ValueError("P-FS2-009 has one fixed 10-epoch/batch32 endpoint")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    control_name = "future_seed_gdn2_log_spd_control"
    candidate_name = "future_seed_gdn2_metric_pullback"
    control = run_arm(
        arm="future_seed_gdn2_log_spd",
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
    for key in ("init_parameter_hash", "data_hashes", "warmup_batch_hash", "parameters"):
        if candidate[key] != control[key]:
            raise RuntimeError(f"Matched arm mismatch for {key}: {control[key]} != {candidate[key]}")

    control_swaps = wrong_key_swap_summary(_cases(args.output_dir, control_name))
    candidate_swaps = wrong_key_swap_summary(_cases(args.output_dir, candidate_name))
    metric_rows = candidate["log_spd"]["per_layer"]
    pullback = candidate["metric_pullback"]
    pullback_rows = pullback["per_receiver"]
    activation_checks = {
        "two_private_metrics_active": (
            candidate["log_spd"]["active_layers"] == 2
            and all(row["actual_metric_delta_fro_mean"] >= 1e-4 for row in metric_rows)
        ),
        "private_metrics_bounded": all(
            row["actual_metric_eigenvalue_min"] >= 0.45
            and row["actual_metric_eigenvalue_max"] <= 2.05
            and row["actual_metric_condition_max"] < 4.60
            for row in metric_rows
        ),
        "exactly_one_pullback_route": pullback["active_routes"] == 1,
        "pullback_active": all(
            row["finite"]
            and row["pullback_delta_fro_mean"] >= 1e-4
            and row["residual_relative_rms_mean"] >= 1e-4
            and row["residual_relative_rms_head_std"] > 0
            and row["residual_relative_rms_board_std"] > 0
            for row in pullback_rows
        ),
        "pullback_bounded": all(
            row["pullback_condition_max"] < 4.60
            and row["transported_to_producer_rms_max"] <= 4.60
            for row in pullback_rows
        ),
        "native_futureseed_active": candidate["future_seed"]["active_seed_routes"] == 1,
    }

    cm = control["metrics"]
    pm = candidate["metrics"]
    quality_checks = {
        "balanced_at_least_0.55": pm["balanced_accuracy"] >= 0.55,
        "balanced_gain_at_least_0.10": pm["balanced_accuracy"] - cm["balanced_accuracy"] >= 0.10,
        "future_gain_at_least_0.07": pm["future"]["accuracy"] - cm["future"]["accuracy"] >= 0.07,
        "past_gain_at_least_0.07": pm["past"]["accuracy"] - cm["past"]["accuracy"] >= 0.07,
        "joint_at_least_0.08": pm["joint_exact"] >= 0.08,
        "joint_gain_at_least_0.04": pm["joint_exact"] - cm["joint_exact"] >= 0.04,
        "total_errors_reduced": candidate_swaps["errors"] < control_swaps["errors"],
        "swap_fraction_reduced_at_least_0.05": (
            control_swaps["wrong_key_swap_fraction_of_errors"]
            - candidate_swaps["wrong_key_swap_fraction_of_errors"]
            >= 0.05
        ),
    }
    cost_ratios = {
        "elapsed": _ratio(candidate["elapsed_sec_including_validation"], control["elapsed_sec_including_validation"]),
        "post_warm_wall": _ratio(candidate["post_warm_arm_wall_sec_through_checkpoint"], control["post_warm_arm_wall_sec_through_checkpoint"]),
        "warmed_step": _ratio(candidate["warmed_step_benchmark"]["elapsed_sec"], control["warmed_step_benchmark"]["elapsed_sec"]),
        "peak_allocation": _ratio(candidate["peak_training_cuda_mem_bytes"], control["peak_training_cuda_mem_bytes"]),
    }
    cost_checks = {
        "elapsed_below_1.30x": cost_ratios["elapsed"] < 1.30,
        "post_warm_wall_below_1.30x": cost_ratios["post_warm_wall"] < 1.30,
        "warmed_step_below_1.30x": cost_ratios["warmed_step"] < 1.30,
        "peak_allocation_below_1.10x": cost_ratios["peak_allocation"] < 1.10,
    }
    passed = all(activation_checks.values()) and all(quality_checks.values()) and all(cost_checks.values())
    comparison = {
        "status": "complete",
        "protocol": {
            "plan": "P-FS2-009",
            "mechanism": "private Log-SPD receiver-read metric pullback FutureSeed",
            "run_order": [control_name, candidate_name],
            "model": "D128/L2/H4/K32/V32 pinned official GDN2 plus native FutureSeed",
            "sequence_length": SEQUENCE_LENGTH,
            "num_kv_pairs": NUM_KV_PAIRS,
            "epochs": MAX_EPOCHS,
            "batch_size": BATCH_SIZE,
            "seed": 123,
            "new_parameters_vs_control": 0,
            "new_recurrent_state": 0,
            "new_scans": 0,
        },
        "control": control,
        "control_swap_summary": control_swaps,
        "candidate": candidate,
        "candidate_swap_summary": candidate_swaps,
        "cost_ratios": cost_ratios,
        "registered_gate": {
            "passed": passed,
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
