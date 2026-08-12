from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Callable

from experiments.zoology_mqar.length_scaling import run_arm


SEQUENCE_LENGTH = 1024
NUM_KV_PAIRS = 4
MODEL_WIDTH = 128
MODEL_LAYERS = 2
MODEL_HEADS = 4
GDN2_HEAD_DIM = 32
GDN2_VALUE_DIM = 32
TRAIN_EXAMPLES = 10_000
VALID_EXAMPLES = 1_000
BATCH_SIZE = 32
MAX_EPOCHS = 10
SEED = 123

CONTROL_ARM = "future_seed_gdn2"
CONTROL_OUTPUT_ARM = "future_seed_gdn2_runtime_control"
CANDIDATE_ARM = "future_seed_gdn2_oig"

P020_BALANCED = 0.48225
P020_SWAP_FRACTION = 0.9415741187831965
P020_SCORE_SHA256 = "a9c750e84fc918910c008b679016c3f24a5e821e3c20b84b1f0b07c1c64b9b3e"
P020_CASES_SHA256 = "127fb40e32a14a54f6f90c784dfe728c85dcf200535e0fb300e3e28a55c8751b"


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


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _all_finite(value: Any) -> bool:
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return True
    if isinstance(value, (int, float)):
        return math.isfinite(float(value))
    if isinstance(value, dict):
        return all(_all_finite(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return all(_all_finite(item) for item in value)
    return False


def _diagnostic_value(
    diagnostics: dict[str, Any],
    names: tuple[str, ...],
    reducer: Callable[[list[float]], float],
) -> float:
    for name in names:
        if name in diagnostics:
            return float(diagnostics[name])
    rows = diagnostics.get("per_layer")
    if not isinstance(rows, list) or not rows:
        raise KeyError(f"OIG diagnostic is missing one of {names}")
    values = [
        float(next(row[name] for name in names if name in row))
        for row in rows
    ]
    return reducer(values)


def _active_heads(diagnostics: dict[str, Any]) -> int:
    for name in ("active_heads", "active_layer_heads"):
        if name in diagnostics:
            return int(diagnostics[name])
    rows = diagnostics.get("per_layer")
    if not isinstance(rows, list) or not rows:
        raise KeyError("OIG diagnostic is missing active-head evidence")
    return sum(int(row["active_heads"]) for row in rows)


def _failed_checks(groups: dict[str, dict[str, bool]]) -> dict[str, list[str]]:
    return {
        group: [name for name, passed in checks.items() if not passed]
        for group, checks in groups.items()
        if not all(checks.values())
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--p020-score", type=Path, required=True)
    parser.add_argument("--p020-cases", type=Path, required=True)
    parser.add_argument("--max-epochs", type=int, required=True)
    parser.add_argument("--batch-size", type=int, required=True)
    args = parser.parse_args()
    if args.max_epochs != MAX_EPOCHS or args.batch_size != BATCH_SIZE:
        raise ValueError("P-GDN3-027 has one fixed 10-epoch/batch32 protocol")
    if _sha256(args.p020_score) != P020_SCORE_SHA256:
        raise RuntimeError("Frozen P020 score hash changed")
    if _sha256(args.p020_cases) != P020_CASES_SHA256:
        raise RuntimeError("Frozen P020 cases hash changed")
    p020_score = json.loads(args.p020_score.read_text())
    p020_cases = json.loads(args.p020_cases.read_text())
    p020_swaps = wrong_key_swap_summary(p020_cases)
    if abs(float(p020_score["metrics"]["balanced_accuracy"]) - P020_BALANCED) > 1e-12:
        raise RuntimeError("Frozen P020 balanced accuracy changed")
    if abs(
        float(p020_swaps["wrong_key_swap_fraction_of_errors"])
        - P020_SWAP_FRACTION
    ) > 1e-12:
        raise RuntimeError("Frozen P020 wrong-key swap fraction changed")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    fixed_arm_args = {
        "sequence_length": SEQUENCE_LENGTH,
        "num_kv_pairs": NUM_KV_PAIRS,
        "output_dir": args.output_dir,
        "max_epochs": MAX_EPOCHS,
        "batch_size": BATCH_SIZE,
        "model_width": MODEL_WIDTH,
        "model_heads": MODEL_HEADS,
        "gdn2_head_dim": GDN2_HEAD_DIM,
        "gdn2_expand_v": 1.0,
        "save_checkpoint": True,
    }
    control = run_arm(
        arm=CONTROL_ARM,
        output_arm_name=CONTROL_OUTPUT_ARM,
        **fixed_arm_args,
    )
    candidate = run_arm(arm=CANDIDATE_ARM, **fixed_arm_args)

    if control["data_hashes"] != candidate["data_hashes"]:
        raise RuntimeError("Control and OIG candidate data hashes differ")
    if candidate["parent_init_parameter_hash"] != control["init_parameter_hash"]:
        raise RuntimeError("OIG candidate did not preserve the control parent initialization")

    control_dir = args.output_dir / "length_1024" / CONTROL_OUTPUT_ARM
    candidate_dir = args.output_dir / "length_1024" / CANDIDATE_ARM
    control_cases = json.loads((control_dir / "cases.json").read_text())
    candidate_cases = json.loads((candidate_dir / "cases.json").read_text())
    swaps = {
        "control": wrong_key_swap_summary(control_cases),
        "candidate": wrong_key_swap_summary(candidate_cases),
        "p020_frozen_fraction_of_errors": P020_SWAP_FRACTION,
    }

    diagnostics = candidate["oig"]
    active_layers = int(diagnostics["active_layers"])
    active_heads = _active_heads(diagnostics)
    mean_abs_mix = _diagnostic_value(
        diagnostics,
        ("mean_abs_mix", "mix_abs_mean", "mix_mean_abs"),
        lambda values: sum(values) / len(values),
    )
    p_offdiag = _diagnostic_value(
        diagnostics,
        (
            "p_offdiag_relative_rms",
            "preconditioner_offdiag_relative_rms",
            "p_offdiag_rms",
        ),
        lambda values: sum(values) / len(values),
    )
    a_relative = _diagnostic_value(
        diagnostics,
        (
            "a_relative_rms",
            "write_direction_relative_rms",
            "a_relative_change",
        ),
        lambda values: sum(values) / len(values),
    )
    constraint_error = _diagnostic_value(
        diagnostics,
        (
            "constraint_max_error",
            "write_constraint_max_error",
            "erase_constraint_max_error",
        ),
        max,
    )
    symmetry_error = _diagnostic_value(
        diagnostics,
        ("symmetry_max_error", "preconditioner_symmetry_max_error"),
        max,
    )
    sampled_eigenvalue_min = _diagnostic_value(
        diagnostics,
        ("sampled_eigenvalue_min", "sampled_eig_min"),
        min,
    )
    sampled_eigenvalue_max = _diagnostic_value(
        diagnostics,
        ("sampled_eigenvalue_max", "sampled_eig_max"),
        max,
    )

    candidate_metrics = candidate["metrics"]
    control_metrics = control["metrics"]
    activation_stability_checks = {
        "diagnostics_are_finite": _all_finite(diagnostics),
        "exactly_two_active_layers": active_layers == MODEL_LAYERS,
        "exactly_eight_active_heads": active_heads == MODEL_LAYERS * MODEL_HEADS,
        "mean_abs_mix_at_least_1e-3": mean_abs_mix >= 1e-3,
        "p_offdiag_at_least_1e-3": p_offdiag >= 1e-3,
        "a_relative_change_at_least_0.02": a_relative >= 0.02,
        "every_layer_mix_is_active": all(
            float(row["mix_abs_min"]) >= 1e-3
            for row in diagnostics["per_layer"]
        ),
        "every_layer_p_offdiag_at_least_1e-3": all(
            float(row["p_offdiag_relative_rms"]) >= 1e-3
            for row in diagnostics["per_layer"]
        ),
        "every_layer_a_relative_change_at_least_0.02": all(
            float(row["a_relative_rms"]) >= 0.02
            for row in diagnostics["per_layer"]
        ),
        "constraint_error_at_most_1e-4": (
            math.isfinite(constraint_error) and constraint_error <= 1e-4
        ),
        "symmetry_error_at_most_1e-5": (
            math.isfinite(symmetry_error) and symmetry_error <= 1e-5
        ),
        "sampled_eigenvalues_in_registered_interval": (
            math.isfinite(sampled_eigenvalue_min)
            and math.isfinite(sampled_eigenvalue_max)
            and sampled_eigenvalue_min >= 1e-4
            and sampled_eigenvalue_max <= 1.001
        ),
        "all_sampled_covariances_cholesky_positive": all(
            bool(row["cholesky_success"])
            for row in diagnostics["per_layer"]
        ),
        "all_sherman_morrison_denominators_at_least_1e-4": min(
            float(row["minimum_denominator"])
            for row in diagnostics["per_layer"]
        ) >= 1e-4,
        "exactly_eight_candidate_parameters": diagnostics["parameter_delta"] == 8,
        "exactly_4096_extra_state_values_per_layer": (
            diagnostics["inverse_gram_state_values_per_layer"] == 4096
        ),
        "compiled_fullgraph_inductor_without_eager_fallback": (
            diagnostics["compiled_fullgraph"]
            and diagnostics["compiled_backend"] == "inductor"
            and not diagnostics["formal_eager_fallback"]
        ),
    }
    quality_checks = {
        "candidate_metrics_are_finite": _all_finite(candidate_metrics),
        "balanced_accuracy_at_least_0.60": (
            candidate_metrics["balanced_accuracy"] >= 0.60
        ),
        "balanced_gain_over_control_at_least_0.20": (
            candidate_metrics["balanced_accuracy"]
            >= control_metrics["balanced_accuracy"] + 0.20
        ),
        "balanced_gain_over_p020_at_least_0.10": (
            candidate_metrics["balanced_accuracy"] >= P020_BALANCED + 0.10
        ),
        "past_accuracy_at_least_0.58": candidate_metrics["past"]["accuracy"] >= 0.58,
        "future_accuracy_at_least_0.58": (
            candidate_metrics["future"]["accuracy"] >= 0.58
        ),
        "joint_exact_at_least_0.15": candidate_metrics["joint_exact"] >= 0.15,
        "wrong_key_swap_fraction_reduced_from_p020_by_0.10": (
            float(swaps["candidate"]["wrong_key_swap_fraction_of_errors"])
            <= P020_SWAP_FRACTION - 0.10
        ),
    }

    warmed_elapsed_ratio = (
        float(candidate["warmed_step_benchmark"]["elapsed_sec"])
        / float(control["warmed_step_benchmark"]["elapsed_sec"])
    )
    peak_allocation_ratio = (
        float(candidate["peak_training_cuda_mem_bytes"])
        / float(control["peak_training_cuda_mem_bytes"])
    )
    cost = {
        "warmed_elapsed_ratio_vs_control": warmed_elapsed_ratio,
        "peak_allocation_ratio_vs_control": peak_allocation_ratio,
        "candidate_elapsed_sec_including_validation": candidate[
            "elapsed_sec_including_validation"
        ],
        "control_elapsed_sec_including_validation": control[
            "elapsed_sec_including_validation"
        ],
        "fit_elapsed_ratio_vs_control": (
            float(candidate["elapsed_sec_including_validation"])
            / float(control["elapsed_sec_including_validation"])
        ),
        "post_warm_wall_ratio_vs_control": (
            float(candidate["post_warm_arm_wall_sec_through_checkpoint"])
            / float(control["post_warm_arm_wall_sec_through_checkpoint"])
        ),
    }
    cost_checks = {
        "cost_metrics_are_finite": _all_finite(cost),
        "warmed_elapsed_ratio_below_2.5": warmed_elapsed_ratio < 2.5,
        "fit_elapsed_ratio_below_2.5": cost["fit_elapsed_ratio_vs_control"] < 2.5,
        "post_warm_wall_ratio_below_2.5": (
            cost["post_warm_wall_ratio_vs_control"] < 2.5
        ),
        "peak_allocation_ratio_below_1.8": peak_allocation_ratio < 1.8,
    }
    integrity_checks = {
        "data_hashes_match_exactly": control["data_hashes"] == candidate["data_hashes"],
        "candidate_parent_init_matches_control": (
            candidate["parent_init_parameter_hash"] == control["init_parameter_hash"]
        ),
    }
    check_groups = {
        "integrity": integrity_checks,
        "activation_stability": activation_stability_checks,
        "quality": quality_checks,
        "cost": cost_checks,
    }
    passed = all(all(checks.values()) for checks in check_groups.values())

    comparison = {
        "status": "complete",
        "plan": "P-GDN3-027",
        "mechanism": "online inverse-Gram preconditioned delta GDN2 recurrence",
        "protocol": {
            "arms_in_order": [CONTROL_ARM, CANDIDATE_ARM],
            "sequence_length": SEQUENCE_LENGTH,
            "num_kv_pairs": NUM_KV_PAIRS,
            "model": "D128/L2/H4/K32/V32 GDN2 plus native FutureSeed",
            "train_examples": TRAIN_EXAMPLES,
            "valid_examples": VALID_EXAMPLES,
            "batch_size": BATCH_SIZE,
            "epochs": MAX_EPOCHS,
            "seed": SEED,
        },
        "frozen_p020": {
            "balanced_accuracy": P020_BALANCED,
            "wrong_key_swap_fraction_of_errors": P020_SWAP_FRACTION,
            "score_sha256": P020_SCORE_SHA256,
            "cases_sha256": P020_CASES_SHA256,
        },
        "control": control,
        "candidate": candidate,
        "wrong_key_swaps": swaps,
        "diagnostic_summary": {
            "active_layers": active_layers,
            "active_heads": active_heads,
            "mean_abs_mix": mean_abs_mix,
            "p_offdiag_relative_rms": p_offdiag,
            "a_relative_rms": a_relative,
            "constraint_max_error": constraint_error,
            "symmetry_max_error": symmetry_error,
            "sampled_eigenvalue_min": sampled_eigenvalue_min,
            "sampled_eigenvalue_max": sampled_eigenvalue_max,
        },
        "integrity_checks": integrity_checks,
        "activation_stability_checks": activation_stability_checks,
        "quality_checks": quality_checks,
        "cost": cost,
        "cost_checks": cost_checks,
        "registered_gate": {
            "passed": passed,
            "failed_checks": _failed_checks(check_groups),
        },
        "artifact_sha256": {
            "control_score": _sha256(control_dir / "score.json"),
            "control_cases": _sha256(control_dir / "cases.json"),
            "control_checkpoint": control["checkpoint_sha256"],
            "candidate_score": _sha256(candidate_dir / "score.json"),
            "candidate_cases": _sha256(candidate_dir / "cases.json"),
            "candidate_checkpoint": candidate["checkpoint_sha256"],
        },
    }
    score = {
        "status": "passed" if passed else "scientific_failure",
        "scientific_failure": not passed,
        "passed": passed,
        "plan": comparison["plan"],
        "candidate_metrics": candidate_metrics,
        "control_metrics": control_metrics,
        "balanced_gain_over_control": (
            candidate_metrics["balanced_accuracy"]
            - control_metrics["balanced_accuracy"]
        ),
        "balanced_gain_over_p020": (
            candidate_metrics["balanced_accuracy"] - P020_BALANCED
        ),
        "wrong_key_swap_reduction_from_p020": (
            P020_SWAP_FRACTION
            - float(swaps["candidate"]["wrong_key_swap_fraction_of_errors"])
        ),
        "diagnostic_summary": comparison["diagnostic_summary"],
        "cost": cost,
        "registered_gate": comparison["registered_gate"],
    }
    _write_json(args.output_dir / "comparison.json", comparison)
    _write_json(args.output_dir / "score.json", score)
    print(json.dumps(score, indent=2, sort_keys=True))

    if not passed:
        abort = {
            "status": "scientific_failure",
            "scientific_failure": True,
            "exit_code": 3,
            "plan": comparison["plan"],
            "reason": "one or more preregistered OIG gates failed",
            "failed_checks": comparison["registered_gate"]["failed_checks"],
            "comparison_path": str(args.output_dir / "comparison.json"),
            "score_path": str(args.output_dir / "score.json"),
        }
        _write_json(args.output_dir / "abort.json", abort)
        raise SystemExit(3)


if __name__ == "__main__":
    main()
