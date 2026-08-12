from __future__ import annotations

import argparse
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


def _finite_at_least(value: float, floor: float) -> bool:
    return math.isfinite(value) and value >= floor


def activation_checks(score: dict[str, Any], *, admission: str) -> dict[str, bool]:
    tape = score["event_tape"]
    producer = tape["producer"]
    receiver = tape["receiver"]
    return {
        "expected_admission_mode": tape["admission_mode"] == admission,
        "event_tape_size_exactly_16": tape["event_tape_size"] == 16,
        "zero_new_parameters": tape["new_parameters"] == 0,
        "zero_persistent_state_delta": tape["persistent_state_delta"] == 0,
        "fixed_32_example_activation_prefix": tape["diagnostic_examples"] == 32,
        "exactly_two_main_and_one_replay_official_scans": (
            tape["main_official_scans"] == 2
            and tape["replay_official_scans"] == 1
        ),
        "exact_committed_edit_active": (
            _finite_at_least(producer["committed_edit_rms"], 1e-4)
            and _finite_at_least(producer["surprise_token_std"], 1e-4)
        ),
        "exactly_16_events_per_board": (
            producer["selected_count_per_board"] == 16.0
        ),
        "selected_positions_span_tokens": producer["selected_position_std"] > 0,
        "surprise_positions_vary_across_boards": (
            admission != "surprise"
            or producer["selected_position_board_std"] > 0
        ),
        "selected_surprise_is_finite_nonzero": (
            _finite_at_least(producer["selected_surprise_mean"], 1e-4)
            and _finite_at_least(producer["selected_surprise_fraction"], 1e-4)
            and producer["selected_surprise_fraction"] <= 1.0 + 1e-6
        ),
        "receiver_replay_state_active": (
            _finite_at_least(receiver["replay_terminal_rms"], 1e-4)
            and receiver["replay_terminal_board_std"] > 0
            and _finite_at_least(receiver["replay_seed_rms"], 1e-4)
            and math.isfinite(receiver["replay_seed_abs_max"])
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--historical-reference-run", type=Path, required=True)
    parser.add_argument("--max-epochs", type=int, default=MAX_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    args = parser.parse_args()
    if args.max_epochs != MAX_EPOCHS or args.batch_size != BATCH_SIZE:
        raise ValueError("P-FS2-007 fixes one baseline plus two 10-epoch batch32 arms")

    historical = load_historical(args.historical_reference_run)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    baseline = run_arm(
        arm="future_seed_gdn2",
        sequence_length=SEQUENCE_LENGTH,
        num_kv_pairs=NUM_KV_PAIRS,
        output_dir=args.output_dir,
        max_epochs=MAX_EPOCHS,
        batch_size=BATCH_SIZE,
        save_checkpoint=True,
    )
    recency = run_arm(
        arm="future_seed_gdn2_recency_replay",
        sequence_length=SEQUENCE_LENGTH,
        num_kv_pairs=NUM_KV_PAIRS,
        output_dir=args.output_dir,
        max_epochs=MAX_EPOCHS,
        batch_size=BATCH_SIZE,
        save_checkpoint=True,
    )
    surprise = run_arm(
        arm="future_seed_gdn2_surprise_replay",
        sequence_length=SEQUENCE_LENGTH,
        num_kv_pairs=NUM_KV_PAIRS,
        output_dir=args.output_dir,
        max_epochs=MAX_EPOCHS,
        batch_size=BATCH_SIZE,
        save_checkpoint=True,
    )

    root = args.output_dir / "length_1024"
    baseline_cases = json.loads(
        (root / "future_seed_gdn2" / "cases.json").read_text()
    )
    recency_cases = json.loads(
        (root / "future_seed_gdn2_recency_replay" / "cases.json").read_text()
    )
    surprise_cases = json.loads(
        (root / "future_seed_gdn2_surprise_replay" / "cases.json").read_text()
    )
    swaps = {
        "historical_native_futureseed": historical["swap_summary"],
        "contemporaneous_native_futureseed": wrong_key_swap_summary(baseline_cases),
        "recency_k16": wrong_key_swap_summary(recency_cases),
        "surprise_k16": wrong_key_swap_summary(surprise_cases),
    }

    integrity_checks = {
        "identical_data": (
            baseline["data_hashes"] == recency["data_hashes"] == surprise["data_hashes"]
        ),
        "identical_warmup_and_first_epoch_anchor_batch": (
            baseline["warmup_batch_hash"]
            == recency["warmup_batch_hash"]
            == surprise["warmup_batch_hash"]
        ),
        "identical_initial_model_state": (
            baseline["init_hash"] == recency["init_hash"] == surprise["init_hash"]
        ),
        "identical_initial_parameters": (
            baseline["init_parameter_hash"]
            == recency["init_parameter_hash"]
            == surprise["init_parameter_hash"]
        ),
        "identical_parameter_count": (
            baseline["parameters"] == recency["parameters"] == surprise["parameters"]
        ),
        "identical_persistent_recurrent_state_budget": (
            baseline["recurrent_state_values_per_layer"]
            == recency["recurrent_state_values_per_layer"]
            == surprise["recurrent_state_values_per_layer"]
            == 4096
        ),
        "matched_transient_k16_tape_budget": (
            recency["event_tape"]["selected_evidence_values_per_board"]
            == surprise["event_tape"]["selected_evidence_values_per_board"]
            == 2048
        ),
        "historical_reference_locked": (
            historical["score"]["metrics"]["balanced_accuracy"]
            == HISTORICAL_BALANCED
            and historical["score"]["metrics"]["joint_exact"]
            == HISTORICAL_JOINT_EXACT
            and historical["score"]["metrics"]["past"]["accuracy"]
            == HISTORICAL_PAST_ACCURACY
            and historical["score"]["metrics"]["future"]["accuracy"]
            == HISTORICAL_FUTURE_ACCURACY
        ),
    }
    activation = {
        "recency_k16": activation_checks(recency, admission="recency"),
        "surprise_k16": activation_checks(surprise, admission="surprise"),
    }

    recency_metrics = recency["metrics"]
    baseline_metrics = baseline["metrics"]
    metrics = surprise["metrics"]
    quality_checks = {
        "surprise_balanced_at_least_0.85": metrics["balanced_accuracy"] >= 0.85,
        "surprise_balanced_gain_over_historical_at_least_0.10": (
            metrics["balanced_accuracy"] >= HISTORICAL_BALANCED + 0.10
        ),
        "surprise_balanced_gain_over_contemporaneous_at_least_0.10": (
            metrics["balanced_accuracy"]
            >= baseline_metrics["balanced_accuracy"] + 0.10
        ),
        "surprise_joint_exact_at_least_0.60": metrics["joint_exact"] >= 0.60,
        "surprise_beats_recency_by_0.05_balanced_or_joint": (
            metrics["balanced_accuracy"]
            >= recency_metrics["balanced_accuracy"] + 0.05
            or metrics["joint_exact"] >= recency_metrics["joint_exact"] + 0.05
        ),
        "wrong_key_swap_fraction_down_0.10_from_historical": (
            swaps["surprise_k16"]["wrong_key_swap_fraction_of_errors"]
            <= HISTORICAL_SWAP_FRACTION - 0.10
        ),
        "wrong_key_swap_fraction_down_0.10_from_contemporaneous": (
            swaps["surprise_k16"]["wrong_key_swap_fraction_of_errors"]
            <= swaps["contemporaneous_native_futureseed"][
                "wrong_key_swap_fraction_of_errors"
            ]
            - 0.10
        ),
    }
    cost = {
        "fit_elapsed_ratio_surprise_to_baseline": (
            surprise["elapsed_sec_including_validation"]
            / baseline["elapsed_sec_including_validation"]
        ),
        "post_warm_wall_ratio_surprise_to_baseline": (
            surprise["post_warm_arm_wall_sec_through_checkpoint"]
            / baseline["post_warm_arm_wall_sec_through_checkpoint"]
        ),
        "warmed_step_ratio_surprise_to_baseline": (
            surprise["warmed_step_benchmark"]["elapsed_sec"]
            / baseline["warmed_step_benchmark"]["elapsed_sec"]
        ),
        "peak_allocation_ratio_surprise_to_baseline": (
            surprise["peak_training_cuda_mem_bytes"]
            / baseline["peak_training_cuda_mem_bytes"]
        ),
        "warmed_step_ratio_surprise_to_recency": (
            surprise["warmed_step_benchmark"]["elapsed_sec"]
            / recency["warmed_step_benchmark"]["elapsed_sec"]
        ),
    }
    cost_checks = {
        "fit_elapsed_overhead_at_most_25_percent": (
            cost["fit_elapsed_ratio_surprise_to_baseline"] <= 1.25
        ),
        "post_warm_wall_overhead_at_most_25_percent": (
            cost["post_warm_wall_ratio_surprise_to_baseline"] <= 1.25
        ),
        "warmed_step_overhead_at_most_25_percent": (
            cost["warmed_step_ratio_surprise_to_baseline"] <= 1.25
        ),
        "peak_allocation_overhead_at_most_25_percent": (
            cost["peak_allocation_ratio_surprise_to_baseline"] <= 1.25
        ),
    }
    passed = (
        all(integrity_checks.values())
        and all(all(rows.values()) for rows in activation.values())
        and all(quality_checks.values())
        and all(cost_checks.values())
    )
    decision = {
        "status": "completed_passed" if passed else "completed_rejected",
        "plan": "P-FS2-007",
        "mechanism": "receiver-native exact-surprise event replay FutureSeed K16",
        "protocol": {
            "sequence_length": SEQUENCE_LENGTH,
            "num_kv_pairs": NUM_KV_PAIRS,
            "epochs": MAX_EPOCHS,
            "batch_size": BATCH_SIZE,
            "seed": 123,
            "model": "D128/L2/H4/K32/V32 pinned official GDN2",
            "arms": ["contemporaneous_baseline", "recency_k16", "surprise_k16"],
            "main_recurrence_changed": False,
            "admission_parameters": 0,
            "event_tape_size": 16,
            "transient_event_tape_values_per_board": 2048,
        },
        "historical_reference": historical,
        "contemporaneous_baseline": baseline,
        "recency_k16": recency,
        "surprise_k16": surprise,
        "wrong_key_swaps": swaps,
        "integrity_checks": integrity_checks,
        "activation_checks": activation,
        "quality_checks": quality_checks,
        "cost": cost,
        "cost_checks": cost_checks,
        "registered_gate": {"passed": passed},
        "next_decision": (
            "Implement one fused Sudoku D256/L12 receiver-native K16 replay transfer."
            if passed
            else "Close receiver-native sparse event replay without K/admission/training rescue."
        ),
    }
    (args.output_dir / "decision.json").write_text(
        json.dumps(decision, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(decision, indent=2, sort_keys=True))
    raise SystemExit(0 if passed else 3)


if __name__ == "__main__":
    main()
