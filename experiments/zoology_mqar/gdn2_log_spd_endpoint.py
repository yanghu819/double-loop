from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from experiments.zoology_mqar.length_scaling import run_arm


SEQUENCE_LENGTH = 1024
NUM_KV_PAIRS = 4
MAX_EPOCHS = 10
BATCH_SIZE = 32
HISTORICAL_SOURCE_SHA = "77e5539fc0ef74231cab658bd610b24254fdcfa7"
HISTORICAL_BALANCED = 0.7475
HISTORICAL_JOINT_EXACT = 0.339
HISTORICAL_PAST_ACCURACY = 0.7535
HISTORICAL_FUTURE_ACCURACY = 0.7415
HISTORICAL_SWAP_FRACTION = 0.806930693069307
HISTORICAL_INIT_HASH = "5595ecb17132326ba41be9f872d3abc81c31cd815832c60aa33f878273ba28ae"
HISTORICAL_INIT_PARAMETER_HASH = (
    "3b0c133410eaed1135224cc0acb094705655cc7e97ea6beeba17b40beab1a368"
)
HISTORICAL_TRAIN_HASH = "647c64ece84984a23dfd817c4f277ea83840dbec57cc18bb6c9bf9eda7cc9a68"
HISTORICAL_TEST_HASH = "4a8237ba8fe19aaff0d1d72de7b7f6505eaab59cd091442c2f463df34cce278f"
HISTORICAL_SCORE_SHA256 = "5d65ebe522bb681cf819953255c2eb7c8e12c2f4335479ac66e6511f7db95583"
HISTORICAL_CASES_SHA256 = "af28e5b00721f01db182449101ed015a7c81496d4ec8ea719bf48c87e67e9fd3"


def wrong_key_swap_summary(cases: list[dict[str, Any]]) -> dict[str, float | int]:
    queries = 0
    errors = 0
    swaps = 0
    reciprocal_events = 0
    for case in cases:
        events = case["events"]
        queries += len(events)
        target_to_index = {event["target"]: index for index, event in enumerate(events)}
        for index, event in enumerate(events):
            if event["correct"]:
                continue
            errors += 1
            other = target_to_index.get(event["prediction"])
            if other is None or other == index:
                continue
            swaps += 1
            paired = events[other]
            reciprocal_events += int(
                not paired["correct"] and paired["prediction"] == event["target"]
            )
    return {
        "queries": queries,
        "errors": errors,
        "wrong_key_valid_value_swaps": swaps,
        "wrong_key_swap_fraction_of_queries": swaps / queries if queries else 0.0,
        "wrong_key_swap_fraction_of_errors": swaps / errors if errors else 0.0,
        "reciprocal_swap_event_fraction_of_errors": (
            reciprocal_events / errors if errors else 0.0
        ),
    }


def load_historical(reference_run: Path) -> dict[str, Any]:
    if (reference_run / "git_sha.txt").read_text().strip() != HISTORICAL_SOURCE_SHA:
        raise RuntimeError("Historical directional MQAR source changed")
    score_path = reference_run / "score.json"
    cases_path = (
        reference_run
        / "output"
        / "length_1024"
        / "future_seed_gdn2"
        / "cases.json"
    )
    if hashlib.sha256(score_path.read_bytes()).hexdigest() != HISTORICAL_SCORE_SHA256:
        raise RuntimeError("Historical directional MQAR score hash changed")
    if hashlib.sha256(cases_path.read_bytes()).hexdigest() != HISTORICAL_CASES_SHA256:
        raise RuntimeError("Historical directional MQAR cases hash changed")
    score = json.loads(score_path.read_text())
    arm = score["lengths"]["1024"]["arms"]["future_seed_gdn2"]
    cases = json.loads(cases_path.read_text())
    if arm["init_hash"] != HISTORICAL_INIT_HASH:
        raise RuntimeError("Historical model-state initialization changed")
    if arm["init_parameter_hash"] != HISTORICAL_INIT_PARAMETER_HASH:
        raise RuntimeError("Historical parameter initialization changed")
    if arm["parameters"] != 661_584:
        raise RuntimeError("Historical parameter count changed")
    if arm["data_hashes"] != {
        "train": HISTORICAL_TRAIN_HASH,
        "test": HISTORICAL_TEST_HASH,
    }:
        raise RuntimeError("Historical directional MQAR data hashes changed")
    if abs(float(arm["metrics"]["balanced_accuracy"]) - HISTORICAL_BALANCED) > 1e-12:
        raise RuntimeError("Historical balanced accuracy changed")
    if abs(float(arm["metrics"]["joint_exact"]) - HISTORICAL_JOINT_EXACT) > 1e-12:
        raise RuntimeError("Historical joint exact changed")
    if (
        abs(float(arm["metrics"]["past"]["accuracy"]) - HISTORICAL_PAST_ACCURACY)
        > 1e-12
    ):
        raise RuntimeError("Historical past accuracy changed")
    if (
        abs(
            float(arm["metrics"]["future"]["accuracy"])
            - HISTORICAL_FUTURE_ACCURACY
        )
        > 1e-12
    ):
        raise RuntimeError("Historical future accuracy changed")
    swaps = wrong_key_swap_summary(cases)
    if abs(
        float(swaps["wrong_key_swap_fraction_of_errors"]) - HISTORICAL_SWAP_FRACTION
    ) > 1e-12:
        raise RuntimeError("Historical wrong-key swap evidence changed")
    return {"score": arm, "swap_summary": swaps}


def branch_selector(geometry: dict[str, Any]) -> dict[str, bool]:
    return {
        "exactly_128_examples": geometry["examples"] == 128,
        "rank_or_anisotropy_trigger": (
            geometry["effective_rank_fraction"]["median"] <= 0.50
            or geometry["anisotropy"]["median"] >= 4.0
        ),
        "affected_fraction_at_least_0.75": geometry["affected_fraction"] >= 0.75,
    }


def _binding_not_regressed(control_value: float, candidate_value: float) -> bool:
    return candidate_value >= control_value - 0.05 * max(abs(control_value), 1e-6)


def geometry_gate(
    control_geometry: dict[str, Any],
    candidate_geometry: dict[str, Any],
    metric_diagnostics: dict[str, Any],
) -> dict[str, bool]:
    control_layers = control_geometry["per_layer"]
    candidate_layers = candidate_geometry["per_layer"]
    metric_layers = metric_diagnostics["per_layer"]
    return {
        "exactly_two_matched_layers": (
            len(control_layers) == len(candidate_layers) == len(metric_layers) == 2
        ),
        "all_actual_metrics_active": all(
            row["actual_metric_delta_fro_mean"] >= 1e-4 for row in metric_layers
        ),
        "all_fp32_metrics_analytically_bounded": all(
            row["fp32_metric_eigenvalue_min"] >= 0.5 - 1e-5
            and row["fp32_metric_eigenvalue_max"] <= 2.0 + 1e-5
            and row["fp32_metric_condition_max"] < 4.0 + 1e-5
            and row["fp32_metric_logdet_abs_max"] <= 1e-4
            for row in metric_layers
        ),
        "all_bf16_applied_metrics_bounded": all(
            row["actual_metric_eigenvalue_min"] >= 0.45
            and row["actual_metric_eigenvalue_max"] <= 2.05
            and row["actual_metric_condition_max"] < 4.60
            and row["actual_metric_logdet_abs_max"] <= 0.10
            for row in metric_layers
        ),
        "effective_rank_fraction_gain_at_least_0.05": all(
            candidate["effective_rank_fraction"]["median"]
            >= control["effective_rank_fraction"]["median"] + 0.05
            for control, candidate in zip(control_layers, candidate_layers, strict=True)
        ),
        "anisotropy_reduced_at_least_20_percent": all(
            candidate["anisotropy"]["median"]
            <= 0.8 * control["anisotropy"]["median"]
            for control, candidate in zip(control_layers, candidate_layers, strict=True)
        ),
        "binding_contrast_not_regressed_over_5_percent": all(
            _binding_not_regressed(
                control["binding_contrast"][direction]["median"],
                candidate["binding_contrast"][direction]["median"],
            )
            for control, candidate in zip(control_layers, candidate_layers, strict=True)
            for direction in ("future", "past")
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--historical-reference", type=Path, required=True)
    parser.add_argument("--max-epochs", type=int, default=MAX_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    args = parser.parse_args()
    if args.max_epochs != MAX_EPOCHS or args.batch_size != BATCH_SIZE:
        raise ValueError("P-GDN3-020 has one fixed D128/L2/H4/K32/10-epoch arm")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    historical = load_historical(args.historical_reference.resolve())
    runtime_control = run_arm(
        arm="future_seed_gdn2",
        output_arm_name="future_seed_gdn2_runtime_control",
        sequence_length=SEQUENCE_LENGTH,
        num_kv_pairs=NUM_KV_PAIRS,
        output_dir=args.output_dir,
        max_epochs=MAX_EPOCHS,
        batch_size=BATCH_SIZE,
        save_checkpoint=True,
        capture_address_geometry=True,
    )
    if runtime_control["data_hashes"] != historical["score"]["data_hashes"]:
        raise RuntimeError("Matched runtime-control data changed")
    runtime_control_cases = json.loads(
        (
            args.output_dir
            / "length_1024"
            / "future_seed_gdn2_runtime_control"
            / "cases.json"
        ).read_text()
    )
    runtime_control_swaps = wrong_key_swap_summary(runtime_control_cases)
    carrier_checks = {
        "parameter_count_is_exact": runtime_control["parameters"] == 661_584,
        "state_geometry_is_exact": (
            runtime_control["recurrent_state_values_per_layer"] == 4_096
        ),
        "balanced_accuracy_at_least_0.70": (
            runtime_control["metrics"]["balanced_accuracy"] >= 0.70
        ),
        "joint_exact_at_least_0.25": runtime_control["metrics"]["joint_exact"] >= 0.25,
        "past_accuracy_at_least_0.68": (
            runtime_control["metrics"]["past"]["accuracy"] >= 0.68
        ),
        "future_accuracy_at_least_0.68": (
            runtime_control["metrics"]["future"]["accuracy"] >= 0.68
        ),
    }
    carrier_admission = {
        "passed": all(carrier_checks.values()),
        "checks": carrier_checks,
        "control": runtime_control,
        "control_swap_summary": runtime_control_swaps,
    }
    (args.output_dir / "carrier_admission.json").write_text(
        json.dumps(carrier_admission, indent=2, sort_keys=True) + "\n"
    )
    if not carrier_admission["passed"]:
        raise RuntimeError("Matched directional MQAR carrier admission failed")

    branch_checks = branch_selector(runtime_control["address_geometry"])
    branch_admission = {
        "passed": all(branch_checks.values()),
        "checks": branch_checks,
        "control_address_geometry": runtime_control["address_geometry"],
        "decision": "bounded_native_log_spd" if all(branch_checks.values()) else "closed",
    }
    (args.output_dir / "branch_admission.json").write_text(
        json.dumps(branch_admission, indent=2, sort_keys=True) + "\n"
    )
    if not branch_admission["passed"]:
        raise RuntimeError("Matched runtime geometry did not open the Log-SPD branch")

    candidate = run_arm(
        arm="future_seed_gdn2_log_spd",
        sequence_length=SEQUENCE_LENGTH,
        num_kv_pairs=NUM_KV_PAIRS,
        output_dir=args.output_dir,
        max_epochs=MAX_EPOCHS,
        batch_size=BATCH_SIZE,
        save_checkpoint=True,
        capture_address_geometry=True,
    )
    if candidate["parent_init_parameter_hash"] != runtime_control["init_parameter_hash"]:
        raise RuntimeError("Candidate did not preserve the matched runtime-control init")
    if candidate["data_hashes"] != historical["score"]["data_hashes"]:
        raise RuntimeError("Directional MQAR data changed")

    cases = json.loads(
        (
            args.output_dir
            / "length_1024"
            / "future_seed_gdn2_log_spd"
            / "cases.json"
        ).read_text()
    )
    candidate_swaps = wrong_key_swap_summary(cases)
    metrics = candidate["metrics"]
    geometry_checks = geometry_gate(
        runtime_control["address_geometry"],
        candidate["address_geometry"],
        candidate["log_spd"],
    )
    stability_checks = {
        "future_seed_terminal_rms_finite_nonzero": (
            0.0 < float(candidate["future_seed"]["future_seed_raw_rms"]) < float("inf")
        ),
        "future_seed_terminal_rms_at_most_2x_control": (
            float(candidate["future_seed"]["future_seed_raw_rms"])
            <= 2.0 * float(runtime_control["future_seed"]["future_seed_raw_rms"])
        ),
    }
    control_elapsed = float(runtime_control["elapsed_sec_including_validation"])
    control_arm_wall = float(runtime_control["arm_wall_sec_through_checkpoint"])
    control_peak = float(runtime_control["peak_training_cuda_mem_bytes"])
    quality_checks = {
        "balanced_accuracy_at_least_0.85": metrics["balanced_accuracy"] >= 0.85,
        "balanced_gain_over_historical_at_least_0.10": (
            metrics["balanced_accuracy"] - HISTORICAL_BALANCED >= 0.10
        ),
        "balanced_gain_over_matched_control_at_least_0.10": (
            metrics["balanced_accuracy"]
            - runtime_control["metrics"]["balanced_accuracy"]
            >= 0.10
        ),
        "joint_exact_at_least_0.60": metrics["joint_exact"] >= 0.60,
        "past_accuracy_not_below_0.85": metrics["past"]["accuracy"] >= 0.85,
        "future_accuracy_not_below_0.85": metrics["future"]["accuracy"] >= 0.85,
        "wrong_key_swap_fraction_reduced_at_least_0.10": (
            HISTORICAL_SWAP_FRACTION
            - float(candidate_swaps["wrong_key_swap_fraction_of_errors"])
            >= 0.10
        ),
        "wrong_key_swap_fraction_reduced_vs_control_at_least_0.10": (
            float(runtime_control_swaps["wrong_key_swap_fraction_of_errors"])
            - float(candidate_swaps["wrong_key_swap_fraction_of_errors"])
            >= 0.10
        ),
    }
    cost_checks = {
        "elapsed_overhead_below_15_percent": (
            float(candidate["elapsed_sec_including_validation"]) < 1.15 * control_elapsed
        ),
        "arm_wall_overhead_below_15_percent": (
            float(candidate["arm_wall_sec_through_checkpoint"])
            < 1.15 * control_arm_wall
        ),
        "peak_memory_overhead_below_10_percent": (
            float(candidate["peak_training_cuda_mem_bytes"]) < 1.10 * control_peak
        ),
        "warmed_step_overhead_below_15_percent": (
            float(candidate["warmed_step_benchmark"]["elapsed_sec"])
            < 1.15 * float(runtime_control["warmed_step_benchmark"]["elapsed_sec"])
        ),
    }
    comparison = {
        "status": "complete",
        "protocol": {
            "plan": "P-GDN3-020",
            "mechanism": "bounded trace-free Log-SPD native Q/K address metric",
            "sequence_length": SEQUENCE_LENGTH,
            "num_kv_pairs": NUM_KV_PAIRS,
            "epochs": MAX_EPOCHS,
            "batch_size": BATCH_SIZE,
            "seed": 123,
            "model": "D128/L2/H4/K32 official-FLA GDN2 plus native FutureSeed",
            "new_parameters": 4216,
            "new_recurrent_state": 0,
            "new_scan": 0,
        },
        "historical_reference": historical,
        "carrier_admission": carrier_admission,
        "branch_admission": branch_admission,
        "candidate": candidate,
        "candidate_swap_summary": candidate_swaps,
        "registered_gate": {
            "passed": all(geometry_checks.values())
            and all(stability_checks.values())
            and all(quality_checks.values())
            and all(cost_checks.values()),
            "geometry_checks": geometry_checks,
            "stability_checks": stability_checks,
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
