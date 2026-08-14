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
        raise ValueError("P-GDN3-033 has one fixed 10-epoch/batch32 endpoint")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    control_name = "future_seed_gdn2_runtime_control"
    candidate_name = "future_seed_gdn2_shared_log_spd"
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

    if candidate["parent_init_parameter_hash"] != control["init_parameter_hash"]:
        raise RuntimeError("Shared metric changed native parent initialization")
    if candidate["data_hashes"] != control["data_hashes"]:
        raise RuntimeError("Matched data hashes differ")
    if candidate["warmup_batch_hash"] != control["warmup_batch_hash"]:
        raise RuntimeError("Matched warmup batch differs")

    control_swaps = wrong_key_swap_summary(load_cases(args.output_dir, control_name))
    candidate_swaps = wrong_key_swap_summary(
        load_cases(args.output_dir, candidate_name)
    )
    metric = candidate["log_spd"]
    metric_rows = metric["per_layer"]
    activation_checks = {
        "exactly_two_active_layers": metric["active_layers"] == 2,
        "one_shared_metric_object": bool(metric["shared_object"]),
        "one_shared_parameter": bool(metric["shared_parameter"]),
        "parameter_delta_exactly_2108": metric["parameter_delta"] == 2_108,
        "all_actual_metrics_active": all(
            row["actual_metric_delta_fro_mean"] >= 1e-4 for row in metric_rows
        ),
        "all_fp32_metrics_bounded": all(
            row["fp32_metric_eigenvalue_min"] >= 0.5 - 1e-5
            and row["fp32_metric_eigenvalue_max"] <= 2.0 + 1e-5
            and row["fp32_metric_condition_max"] < 4.0 + 1e-5
            and row["fp32_metric_logdet_abs_max"] <= 1e-4
            for row in metric_rows
        ),
        "all_bf16_metrics_bounded": all(
            row["actual_metric_eigenvalue_min"] >= 0.45
            and row["actual_metric_eigenvalue_max"] <= 2.05
            and row["actual_metric_condition_max"] < 4.60
            and row["actual_metric_logdet_abs_max"] <= 0.10
            for row in metric_rows
        ),
        "future_seed_active_finite": (
            0.0 < float(candidate["future_seed"]["future_seed_raw_rms"])
            < float("inf")
        ),
    }

    control_metrics = control["metrics"]
    candidate_metrics = candidate["metrics"]
    quality_checks = {
        "balanced_at_least_0.50": candidate_metrics["balanced_accuracy"] >= 0.50,
        "balanced_gain_at_least_0.15": (
            candidate_metrics["balanced_accuracy"]
            - control_metrics["balanced_accuracy"]
            >= 0.15
        ),
        "future_gain_at_least_0.10": (
            candidate_metrics["future"]["accuracy"]
            - control_metrics["future"]["accuracy"]
            >= 0.10
        ),
        "past_gain_at_least_0.10": (
            candidate_metrics["past"]["accuracy"]
            - control_metrics["past"]["accuracy"]
            >= 0.10
        ),
        "joint_at_least_0.10": candidate_metrics["joint_exact"] >= 0.10,
        "joint_gain_at_least_0.10": (
            candidate_metrics["joint_exact"] - control_metrics["joint_exact"]
            >= 0.10
        ),
        "total_errors_reduced": candidate_swaps["errors"] < control_swaps["errors"],
        "wrong_key_swap_fraction_reduced_at_least_0.10": (
            float(control_swaps["wrong_key_swap_fraction_of_errors"])
            - float(candidate_swaps["wrong_key_swap_fraction_of_errors"])
            >= 0.10
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
        "elapsed_below_1.15x": cost_ratios["elapsed"] < 1.15,
        "post_warm_wall_below_1.15x": cost_ratios["post_warm_wall"] < 1.15,
        "warmed_step_below_1.15x": cost_ratios["warmed_step"] < 1.15,
        "peak_allocation_below_1.10x": cost_ratios["peak_allocation"] < 1.10,
    }
    passed = (
        all(activation_checks.values())
        and all(quality_checks.values())
        and all(cost_checks.values())
    )
    comparison = {
        "status": "complete",
        "protocol": {
            "plan": "P-GDN3-033",
            "mechanism": "cross-layer shared bounded Log-SPD Q/K metric",
            "run_order": [control_name, candidate_name],
            "sequence_length": SEQUENCE_LENGTH,
            "num_kv_pairs": NUM_KV_PAIRS,
            "epochs": MAX_EPOCHS,
            "batch_size": BATCH_SIZE,
            "seed": 123,
            "model": "D128/L2/H4/K32/V32 official GDN2 plus native FutureSeed",
            "new_parameters": 2_108,
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
        "decision": "admit_sudoku_transfer" if passed else "closed",
    }
    (args.output_dir / "comparison.json").write_text(
        json.dumps(comparison, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(comparison, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
