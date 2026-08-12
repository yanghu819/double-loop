from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from experiments.zoology_mqar.gdn2_log_spd_endpoint import wrong_key_swap_summary
from experiments.zoology_mqar.length_scaling import run_arm


SEQUENCE_LENGTH = 1024
NUM_KV_PAIRS = 4
MAX_EPOCHS = 10
BATCH_SIZE = 32


def load_score_and_cases(score_path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    score = json.loads(score_path.read_text())
    cases_path = score_path.with_name("cases.json")
    return score, json.loads(cases_path.read_text())


def ratio(candidate: float, control: float) -> float:
    return candidate / control


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--runtime-control-score", type=Path, required=True)
    parser.add_argument("--log-spd-score", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    control, control_cases = load_score_and_cases(args.runtime_control_score)
    log_spd, log_spd_cases = load_score_and_cases(args.log_spd_score)
    candidate = run_arm(
        arm="future_seed_gdn2_two_edit",
        sequence_length=SEQUENCE_LENGTH,
        num_kv_pairs=NUM_KV_PAIRS,
        output_dir=args.output_dir,
        max_epochs=MAX_EPOCHS,
        batch_size=BATCH_SIZE,
        save_checkpoint=True,
    )
    candidate_cases_path = (
        args.output_dir
        / "length_1024"
        / "future_seed_gdn2_two_edit"
        / "cases.json"
    )
    candidate_cases = json.loads(candidate_cases_path.read_text())
    if not (
        candidate["data_hashes"]
        == control["data_hashes"]
        == log_spd["data_hashes"]
    ):
        raise RuntimeError("Directional MQAR data hashes changed")
    if candidate["parent_init_parameter_hash"] != control["init_parameter_hash"]:
        raise RuntimeError("Two-edit parent initialization is not matched")

    swaps = {
        "runtime_control": wrong_key_swap_summary(control_cases),
        "log_spd": wrong_key_swap_summary(log_spd_cases),
        "two_edit": wrong_key_swap_summary(candidate_cases),
    }
    metrics = candidate["metrics"]
    diagnostics = candidate["two_edit"]
    activation_checks = {
        "exactly_two_active_layers": diagnostics["active_layers"] == 2,
        "parameter_delta_exactly_131072": diagnostics["parameter_delta"] == 131_072,
        "state_delta_zero": diagnostics["new_state_values"] == 0,
        "two_live_edits_per_token": diagnostics["logical_edits_per_token"] == 2,
        "independent_addresses": all(
            row["address_separation"] >= 0.20 for row in diagnostics["per_layer"]
        ),
        "finite_nonzero_aux_payload": all(
            1e-4 <= row["aux_payload_relative_rms"] < 10.0
            for row in diagnostics["per_layer"]
        ),
    }
    quality_checks = {
        "balanced_accuracy_at_least_0.70": metrics["balanced_accuracy"] >= 0.70,
        "joint_exact_at_least_0.25": metrics["joint_exact"] >= 0.25,
        "past_accuracy_at_least_0.68": metrics["past"]["accuracy"] >= 0.68,
        "future_accuracy_at_least_0.68": metrics["future"]["accuracy"] >= 0.68,
        "balanced_gain_over_log_spd_at_least_0.10": (
            metrics["balanced_accuracy"] - log_spd["metrics"]["balanced_accuracy"]
            >= 0.10
        ),
        "wrong_key_swap_fraction_of_errors_down_0.10": (
            swaps["two_edit"]["wrong_key_swap_fraction_of_errors"]
            <= swaps["log_spd"]["wrong_key_swap_fraction_of_errors"] - 0.10
        ),
    }
    cost = {
        "fit_elapsed_ratio_vs_control": ratio(
            candidate["elapsed_sec_including_validation"],
            control["elapsed_sec_including_validation"],
        ),
        "warmed_step_elapsed_ratio_vs_control": ratio(
            candidate["warmed_step_benchmark"]["elapsed_sec"],
            control["warmed_step_benchmark"]["elapsed_sec"],
        ),
        "peak_training_memory_ratio_vs_control": ratio(
            candidate["peak_training_cuda_mem_bytes"],
            control["peak_training_cuda_mem_bytes"],
        ),
    }
    cost_checks = {
        "fit_elapsed_ratio_at_most_2.5": cost["fit_elapsed_ratio_vs_control"] <= 2.5,
        "warmed_step_ratio_at_most_2.5": cost["warmed_step_elapsed_ratio_vs_control"] <= 2.5,
        "peak_memory_ratio_at_most_1.5": cost["peak_training_memory_ratio_vs_control"] <= 1.5,
    }
    decision = {
        "plan": "P-GDN3-021",
        "mechanism": "two learned rank-one live edits per token in one official chunk scan",
        "candidate": candidate,
        "reference_scores": {
            "runtime_control": control,
            "log_spd": log_spd,
        },
        "wrong_key_swaps": swaps,
        "activation_checks": activation_checks,
        "quality_checks": quality_checks,
        "cost": cost,
        "cost_checks": cost_checks,
        "passed": all(activation_checks.values())
        and all(quality_checks.values())
        and all(cost_checks.values()),
        "strong_result": (
            metrics["balanced_accuracy"] >= 0.85
            and metrics["joint_exact"] >= 0.60
        ),
        "artifact_sha256": {
            "candidate_score": hashlib.sha256(
                (
                    args.output_dir
                    / "length_1024"
                    / "future_seed_gdn2_two_edit"
                    / "score.json"
                ).read_bytes()
            ).hexdigest(),
            "candidate_cases": hashlib.sha256(candidate_cases_path.read_bytes()).hexdigest(),
            "candidate_checkpoint": candidate["checkpoint_sha256"],
        },
    }
    (args.output_dir / "decision.json").write_text(
        json.dumps(decision, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(decision, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
