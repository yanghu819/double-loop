from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

from experiments.zoology_mqar.gdn2_log_spd_endpoint import (
    HISTORICAL_BALANCED,
    HISTORICAL_FUTURE_ACCURACY,
    HISTORICAL_JOINT_EXACT,
    HISTORICAL_PAST_ACCURACY,
    HISTORICAL_SWAP_FRACTION,
    load_historical,
    wrong_key_swap_summary,
)
from experiments.zoology_mqar.length_scaling import run_arm


SEQUENCE_LENGTH = 1024
NUM_KV_PAIRS = 4
MAX_EPOCHS = 10
BATCH_SIZE = 32
RUNTIME_REFERENCE_DECISION_SHA256 = (
    "0ffde24c326c3a7acfe24298ea2dbb562571c5021d3aea6ade90cc155bbbdcfc"
)


def _ratio(candidate: float, control: float) -> float:
    if control <= 0:
        raise ValueError("Reference cost must be positive")
    return candidate / control


def _all_finite(values: list[float]) -> bool:
    return all(math.isfinite(value) for value in values)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--historical-reference-run", type=Path, required=True)
    parser.add_argument("--runtime-reference-run", type=Path, required=True)
    parser.add_argument("--max-epochs", type=int, default=MAX_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    args = parser.parse_args()
    if args.max_epochs != MAX_EPOCHS or args.batch_size != BATCH_SIZE:
        raise ValueError("P-GDN3-028 fixes one 10-epoch batch32 candidate")

    historical = load_historical(args.historical_reference_run.resolve())
    runtime_decision_path = args.runtime_reference_run / "score.json"
    runtime_decision_sha256 = hashlib.sha256(
        runtime_decision_path.read_bytes()
    ).hexdigest()
    if runtime_decision_sha256 != RUNTIME_REFERENCE_DECISION_SHA256:
        raise RuntimeError("P-FS2-007 runtime reference decision changed")
    runtime_decision = json.loads(runtime_decision_path.read_text())
    runtime_control = runtime_decision["contemporaneous_baseline"]
    runtime_cases_path = (
        args.runtime_reference_run
        / "output"
        / "length_1024"
        / "future_seed_gdn2"
        / "cases.json"
    )
    runtime_cases = json.loads(runtime_cases_path.read_text())

    args.output_dir.mkdir(parents=True, exist_ok=True)
    candidate = run_arm(
        arm="future_seed_gdn2_atomic_pair",
        sequence_length=SEQUENCE_LENGTH,
        num_kv_pairs=NUM_KV_PAIRS,
        output_dir=args.output_dir,
        max_epochs=MAX_EPOCHS,
        batch_size=BATCH_SIZE,
        save_checkpoint=True,
    )
    candidate_root = (
        args.output_dir / "length_1024" / "future_seed_gdn2_atomic_pair"
    )
    candidate_cases_path = candidate_root / "cases.json"
    candidate_cases = json.loads(candidate_cases_path.read_text())

    integrity_checks = {
        "fixed_directional_data": (
            candidate["data_hashes"]
            == runtime_control["data_hashes"]
            == historical["score"]["data_hashes"]
        ),
        "fixed_warmup_batch": (
            candidate["warmup_batch_hash"]
            == runtime_control["warmup_batch_hash"]
        ),
        "parent_initialization_exact": (
            candidate["parent_init_parameter_hash"]
            == runtime_control["init_parameter_hash"]
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

    diagnostics = candidate["atomic_pair"]
    rows = diagnostics["per_layer"]
    future_seed = candidate["future_seed"]
    state_rms_values = [row["terminal_state_rms"] for row in rows]
    activation_checks = {
        "exactly_two_active_layers": diagnostics["active_layers"] == 2,
        "parameter_delta_exactly_65536": diagnostics["parameter_delta"] == 65_536,
        "zero_new_recurrent_state": diagnostics["new_state_values"] == 0,
        "logical_rank_exactly_two": diagnostics["logical_block_rank"] == 2,
        "payload_shared_exactly": diagnostics["shared_payload"]
        and all(
            row["shared_payload_max_diff"] == 0.0
            and abs(row["aggregate_payload_energy_ratio"] - 1.0) <= 5e-3
            for row in rows
        ),
        "both_address_pairs_well_conditioned": all(
            row["raw_condition_mean"] <= 4.0
            and row["raw_condition_max"] <= 100.0
            and row["raw_eigenvalue_min"] >= 1e-5
            for row in rows
        ),
        "weighted_cross_erase_cancelled": all(
            row["post_weighted_cross_abs"] <= 1e-3
            and row["post_weighted_cross_abs_max"] <= 5e-3
            for row in rows
        ),
        "polar_diagonal_is_stable": all(
            row["post_diagonal_error"] <= 5e-3 for row in rows
        ),
        "terminal_state_finite_nonzero": (
            _all_finite(state_rms_values)
            and all(1e-4 <= value <= 1e4 for value in state_rms_values)
        ),
        "native_futureseed_route_active": future_seed["active_seed_routes"] == 1,
        "receiver_seed_state_bounded_vs_control": (
            math.isfinite(float(future_seed["future_seed_raw_rms"]))
            and float(future_seed["future_seed_raw_rms"])
            <= 2.0 * float(runtime_control["future_seed"]["future_seed_raw_rms"])
        ),
    }

    swaps = {
        "historical_native_futureseed": historical["swap_summary"],
        "contemporaneous_native_futureseed": wrong_key_swap_summary(runtime_cases),
        "atomic_pair": wrong_key_swap_summary(candidate_cases),
    }
    metrics = candidate["metrics"]
    runtime_metrics = runtime_control["metrics"]
    quality_checks = {
        "balanced_accuracy_at_least_0.85": metrics["balanced_accuracy"] >= 0.85,
        "balanced_gain_over_historical_at_least_0.10": (
            metrics["balanced_accuracy"] >= HISTORICAL_BALANCED + 0.10
        ),
        "balanced_gain_over_current_runtime_at_least_0.10": (
            metrics["balanced_accuracy"]
            >= runtime_metrics["balanced_accuracy"] + 0.10
        ),
        "future_accuracy_at_least_0.85": metrics["future"]["accuracy"] >= 0.85,
        "past_accuracy_at_least_0.85": metrics["past"]["accuracy"] >= 0.85,
        "joint_exact_at_least_0.60": metrics["joint_exact"] >= 0.60,
        "wrong_key_swaps_down_at_least_0.10_from_historical": (
            swaps["atomic_pair"]["wrong_key_swap_fraction_of_errors"]
            <= HISTORICAL_SWAP_FRACTION - 0.10
        ),
        "total_errors_lower_than_both_references": (
            swaps["atomic_pair"]["errors"]
            < swaps["historical_native_futureseed"]["errors"]
            and swaps["atomic_pair"]["errors"]
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
        "fit_elapsed_ratio_at_most_2": cost["fit_elapsed_ratio"] <= 2.0,
        "post_warm_wall_ratio_at_most_2": cost["post_warm_wall_ratio"] <= 2.0,
        "warmed_step_ratio_at_most_2": cost["warmed_step_ratio"] <= 2.0,
        "peak_allocation_ratio_at_most_1.5": cost["peak_allocation_ratio"] <= 1.5,
    }

    passed = (
        all(integrity_checks.values())
        and all(activation_checks.values())
        and all(quality_checks.values())
        and all(cost_checks.values())
    )
    decision = {
        "status": "completed_passed" if passed else "completed_rejected",
        "plan": "P-GDN3-028",
        "mechanism": "atomic shared-payload erase-orthogonal rank-two GDN update",
        "protocol": {
            "sequence_length": SEQUENCE_LENGTH,
            "num_kv_pairs": NUM_KV_PAIRS,
            "epochs": MAX_EPOCHS,
            "batch_size": BATCH_SIZE,
            "seed": 123,
            "model": "D128/L2/H4/K32/V32 pinned official GDN2 + native FutureSeed",
            "candidate_only": True,
            "new_parameters": 65_536,
            "new_persistent_state": 0,
            "shared_payload": True,
            "logical_block_rank": 2,
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
            "candidate_score": hashlib.sha256(
                (candidate_root / "score.json").read_bytes()
            ).hexdigest(),
            "candidate_cases": hashlib.sha256(candidate_cases_path.read_bytes()).hexdigest(),
            "candidate_checkpoint": candidate["checkpoint_sha256"],
        },
        "next_decision": (
            "Transfer the passed atomic update to one matched hard-Sudoku scale gate."
            if passed
            else "Close atomic paired-address updates without rank/projection/whitening/training rescue."
        ),
    }
    (args.output_dir / "decision.json").write_text(
        json.dumps(decision, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(decision, indent=2, sort_keys=True))
    raise SystemExit(0 if passed else 3)


if __name__ == "__main__":
    main()
