from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

from experiments.zoology_mqar.gdn2_log_spd_endpoint import (
    HISTORICAL_BALANCED,
    HISTORICAL_CASES_SHA256,
    HISTORICAL_FUTURE_ACCURACY,
    HISTORICAL_JOINT_EXACT,
    HISTORICAL_PAST_ACCURACY,
    HISTORICAL_SCORE_SHA256,
    HISTORICAL_SWAP_FRACTION,
    load_historical,
    wrong_key_swap_summary,
)
from experiments.zoology_mqar.length_scaling import run_arm


SEQUENCE_LENGTH = 1024
NUM_KV_PAIRS = 4
MODEL_WIDTH = 128
MODEL_LAYERS = 2
MODEL_HEADS = 4
HEAD_KEY_DIM = 32
HEAD_VALUE_DIM = 32
MAX_EPOCHS = 10
BATCH_SIZE = 32
SEED = 123
EXPECTED_PARAMETERS = 695_376
EXPECTED_STATE_VALUES_PER_LAYER = 4_096
CANDIDATE_ARM = "future_seed_decoupled_key_gdn2"
RUNTIME_REFERENCE_DECISION_SHA256 = (
    "0ffde24c326c3a7acfe24298ea2dbb562571c5021d3aea6ade90cc155bbbdcfc"
)


def _ratio(candidate: float, control: float) -> float:
    if control <= 0:
        raise ValueError("Reference cost must be positive")
    return candidate / control


def _all_finite(values: list[float]) -> bool:
    return all(math.isfinite(value) for value in values)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--historical-reference-run", type=Path, required=True)
    parser.add_argument("--runtime-reference-run", type=Path, required=True)
    parser.add_argument("--max-epochs", type=int, default=MAX_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    args = parser.parse_args()
    if args.max_epochs != MAX_EPOCHS or args.batch_size != BATCH_SIZE:
        raise ValueError("P-GDN3-031 fixes one 10-epoch batch32 candidate")

    historical = load_historical(args.historical_reference_run.resolve())
    runtime_decision_path = args.runtime_reference_run.resolve() / "score.json"
    runtime_decision_sha256 = _sha256(runtime_decision_path)
    if runtime_decision_sha256 != RUNTIME_REFERENCE_DECISION_SHA256:
        raise RuntimeError("P-FS2-007 runtime reference decision changed")
    runtime_decision = json.loads(runtime_decision_path.read_text())
    runtime_control = runtime_decision["contemporaneous_baseline"]
    runtime_control_swaps = runtime_decision["wrong_key_swaps"][
        "contemporaneous_native_futureseed"
    ]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    candidate = run_arm(
        arm=CANDIDATE_ARM,
        sequence_length=SEQUENCE_LENGTH,
        num_kv_pairs=NUM_KV_PAIRS,
        output_dir=args.output_dir,
        max_epochs=MAX_EPOCHS,
        batch_size=BATCH_SIZE,
        model_width=MODEL_WIDTH,
        model_heads=MODEL_HEADS,
        gdn2_head_dim=HEAD_KEY_DIM,
        gdn2_expand_v=HEAD_VALUE_DIM / HEAD_KEY_DIM,
        save_checkpoint=True,
    )
    candidate_root = args.output_dir / "length_1024" / CANDIDATE_ARM
    candidate_config_path = candidate_root / "config.json"
    candidate_score_path = candidate_root / "score.json"
    candidate_cases_path = candidate_root / "cases.json"
    candidate_cases = json.loads(candidate_cases_path.read_text())

    diagnostics = candidate["decoupled_key"]
    rows = diagnostics["per_layer"]
    future_seed = candidate["future_seed"]
    initial_tie = [float(row["initial_tie_max_error"]) for row in rows]
    key_separation = [float(row["key_separation_mean"]) for row in rows]
    key_relative_rms = [float(row["key_relative_rms"]) for row in rows]
    projection_delta = [float(row["projection_delta_rms"]) for row in rows]
    convolution_delta = [float(row["convolution_delta_rms"]) for row in rows]
    erase_gate_std = [float(row["erase_gate_std"]) for row in rows]
    write_rms = [float(row["write_rms"]) for row in rows]
    transition_norm = [
        float(row["transition_spectral_norm_max"]) for row in rows
    ]
    terminal_rms = [float(row["terminal_state_rms"]) for row in rows]
    terminal_board_std = [
        float(row["terminal_state_board_std"]) for row in rows
    ]

    integrity_checks = {
        "fixed_candidate_protocol": (
            candidate["arm"] == CANDIDATE_ARM
            and candidate["carrier_arm"] == CANDIDATE_ARM
            and candidate["sequence_length"] == SEQUENCE_LENGTH
            and candidate["num_kv_pairs"] == NUM_KV_PAIRS
            and candidate["model_width"] == MODEL_WIDTH
            and candidate["model_heads"] == MODEL_HEADS
            and candidate["gdn2_head_dim"] == HEAD_KEY_DIM
            and candidate["epochs"] == MAX_EPOCHS
            and candidate["parameters"] == EXPECTED_PARAMETERS
            and candidate["recurrent_state_values_per_layer"]
            == EXPECTED_STATE_VALUES_PER_LAYER
        ),
        "fixed_directional_data": (
            candidate["data_hashes"]
            == runtime_control["data_hashes"]
            == historical["score"]["data_hashes"]
        ),
        "fixed_warmup_batch": (
            candidate["warmup_batch_hash"] == runtime_control["warmup_batch_hash"]
        ),
        "exact_native_parent_initialization": (
            candidate["parent_init_parameter_hash"]
            == runtime_control["init_parameter_hash"]
        ),
        "zero_new_persistent_state": (
            diagnostics["new_persistent_state_values"] == 0
            and diagnostics["logical_scans_per_layer"] == 1
        ),
        "historical_reference_locked": (
            historical["score"]["metrics"]["balanced_accuracy"]
            == HISTORICAL_BALANCED
            and historical["score"]["metrics"]["joint_exact"]
            == HISTORICAL_JOINT_EXACT
            and historical["score"]["metrics"]["future"]["accuracy"]
            == HISTORICAL_FUTURE_ACCURACY
            and historical["score"]["metrics"]["past"]["accuracy"]
            == HISTORICAL_PAST_ACCURACY
        ),
        "runtime_reference_decision_locked": (
            runtime_decision_sha256 == RUNTIME_REFERENCE_DECISION_SHA256
        ),
    }

    activation_checks = {
        "exactly_two_active_layers": (
            diagnostics["active_layers"] == MODEL_LAYERS
            and len(rows) == MODEL_LAYERS
        ),
        "native_futureseed_route_active": (
            future_seed["active_seed_routes"] == MODEL_LAYERS - 1
        ),
        "function_preserving_key_tie_was_exact": (
            _all_finite(initial_tie) and max(initial_tie) == 0.0
        ),
        "erase_write_keys_separated": (
            _all_finite(key_separation)
            and all(value >= 1e-4 for value in key_separation)
            and _all_finite(key_relative_rms)
            and all(value >= 0.01 for value in key_relative_rms)
        ),
        "both_key_pipelines_moved_apart": (
            _all_finite(projection_delta)
            and all(value >= 1e-5 for value in projection_delta)
            and _all_finite(convolution_delta)
            and all(value >= 1e-5 for value in convolution_delta)
        ),
        "channel_erase_gate_varies": (
            _all_finite(erase_gate_std)
            and all(value >= 1e-4 for value in erase_gate_std)
        ),
        "write_is_active": (
            _all_finite(write_rms) and all(value >= 1e-6 for value in write_rms)
        ),
        "sampled_transition_is_bounded": (
            _all_finite(transition_norm)
            and all(value <= 1.25 for value in transition_norm)
        ),
        "terminal_state_is_finite_and_bounded": (
            _all_finite(terminal_rms)
            and all(1e-4 <= value <= 1e4 for value in terminal_rms)
        ),
        "terminal_state_varies_by_board": (
            _all_finite(terminal_board_std)
            and all(value > 1e-6 for value in terminal_board_std)
        ),
    }

    swaps = {
        "historical_native_futureseed": historical["swap_summary"],
        "contemporaneous_native_futureseed": runtime_control_swaps,
        "decoupled_key_gdn2": wrong_key_swap_summary(candidate_cases),
    }
    metrics = candidate["metrics"]
    runtime_metrics = runtime_control["metrics"]
    quality_checks = {
        "balanced_accuracy_at_least_0.85": metrics["balanced_accuracy"] >= 0.85,
        "balanced_gain_over_historical_at_least_0.10": (
            metrics["balanced_accuracy"] >= HISTORICAL_BALANCED + 0.10
        ),
        "balanced_gain_over_runtime_control_at_least_0.10": (
            metrics["balanced_accuracy"]
            >= runtime_metrics["balanced_accuracy"] + 0.10
        ),
        "future_accuracy_at_least_0.85": metrics["future"]["accuracy"] >= 0.85,
        "past_accuracy_at_least_0.85": metrics["past"]["accuracy"] >= 0.85,
        "joint_exact_at_least_0.60": metrics["joint_exact"] >= 0.60,
        "wrong_key_swaps_down_at_least_0.10_from_historical": (
            swaps["decoupled_key_gdn2"]["wrong_key_swap_fraction_of_errors"]
            <= HISTORICAL_SWAP_FRACTION - 0.10
        ),
        "wrong_key_swaps_down_at_least_0.10_from_runtime_control": (
            swaps["decoupled_key_gdn2"]["wrong_key_swap_fraction_of_errors"]
            <= runtime_control_swaps["wrong_key_swap_fraction_of_errors"] - 0.10
        ),
        "total_errors_lower_than_both_references": (
            swaps["decoupled_key_gdn2"]["errors"]
            < swaps["historical_native_futureseed"]["errors"]
            and swaps["decoupled_key_gdn2"]["errors"]
            < swaps["contemporaneous_native_futureseed"]["errors"]
        ),
    }

    cost = {
        "fit_elapsed_ratio": _ratio(
            candidate["elapsed_sec_including_validation"],
            runtime_control["elapsed_sec_including_validation"],
        ),
        "post_warm_wall_ratio": _ratio(
            candidate["post_warm_arm_wall_sec_through_checkpoint"],
            runtime_control["post_warm_arm_wall_sec_through_checkpoint"],
        ),
        "warmed_step_ratio": _ratio(
            candidate["warmed_step_benchmark"]["elapsed_sec"],
            runtime_control["warmed_step_benchmark"]["elapsed_sec"],
        ),
        "peak_allocation_ratio": _ratio(
            candidate["peak_training_cuda_mem_bytes"],
            runtime_control["peak_training_cuda_mem_bytes"],
        ),
    }
    cost_checks = {
        "fit_elapsed_ratio_at_most_1.75": cost["fit_elapsed_ratio"] <= 1.75,
        "post_warm_wall_ratio_at_most_1.75": (
            cost["post_warm_wall_ratio"] <= 1.75
        ),
        "warmed_step_ratio_at_most_1.75": cost["warmed_step_ratio"] <= 1.75,
        "peak_allocation_ratio_at_most_1.50": (
            cost["peak_allocation_ratio"] <= 1.50
        ),
    }

    passed = (
        all(integrity_checks.values())
        and all(activation_checks.values())
        and all(quality_checks.values())
        and all(cost_checks.values())
    )
    decision = {
        "status": "completed_passed" if passed else "completed_rejected",
        "plan": "P-GDN3-031",
        "mechanism": (
            "function-preserving direct decoupled erase/write key GDN2 over "
            "the pinned official DPLR chunk operator"
        ),
        "protocol": {
            "sequence_length": SEQUENCE_LENGTH,
            "num_kv_pairs": NUM_KV_PAIRS,
            "epochs": MAX_EPOCHS,
            "batch_size": BATCH_SIZE,
            "seed": SEED,
            "model": "D128/L2/H4/K32/V32 decoupled-key GDN2 + native FutureSeed",
            "candidate_arm": CANDIDATE_ARM,
            "candidate_only": True,
            "foundational_architecture": True,
            "parent_init_exact_required": True,
        },
        "historical_reference": historical,
        "runtime_control": runtime_control,
        "candidate": candidate,
        "wrong_key_swaps": swaps,
        "integrity_checks": integrity_checks,
        "activation_checks": activation_checks,
        "quality_checks": quality_checks,
        "cost": cost,
        "cost_checks": cost_checks,
        "registered_gate": {"passed": passed},
        "artifact_sha256": {
            "historical_score": HISTORICAL_SCORE_SHA256,
            "historical_cases": HISTORICAL_CASES_SHA256,
            "runtime_reference_decision": runtime_decision_sha256,
            "candidate_config": _sha256(candidate_config_path),
            "candidate_score": _sha256(candidate_score_path),
            "candidate_cases": _sha256(candidate_cases_path),
            "candidate_checkpoint": candidate["checkpoint_sha256"],
        },
        "next_decision": (
            "Transfer the passed direct recurrence to one matched hard-Sudoku gate."
            if passed
            else "Close direct decoupled-key GDN2 without key, gate, init, or training rescue."
        ),
    }
    (args.output_dir / "decision.json").write_text(
        json.dumps(decision, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(decision, indent=2, sort_keys=True))
    raise SystemExit(0 if passed else 3)


if __name__ == "__main__":
    main()
