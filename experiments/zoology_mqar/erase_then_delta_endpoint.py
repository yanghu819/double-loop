from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

from experiments.zoology_mqar.canonical_address_companion_endpoint import (
    transition_summary,
)
from experiments.zoology_mqar.erase_then_delta_futureseed import (
    HEAD_DIM,
    NUM_HEADS,
    NUM_MICROSTEPS,
    STATE_VALUES_PER_LAYER,
)
from experiments.zoology_mqar.gdn2_log_spd_endpoint import wrong_key_swap_summary
from experiments.zoology_mqar.length_scaling import run_arm


SEQUENCE_LENGTH = 1024
NUM_KV_PAIRS = 4
MAX_EPOCHS = 10
BATCH_SIZE = 32
SEED = 123
CANDIDATE_ARM = "future_seed_erase_then_delta"
EXPECTED_FROZEN_SCORE_SHA256 = (
    "7201d32834d3f9af4d1057750ba4eb62df492aa495eec770f9dad85f3d4f6e8a"
)
EXPECTED_FROZEN_CASES_SHA256 = (
    "f64b0ae0e45a65a8da2934ece8cfe208fedd835a373d79230320bdf945826f20"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _ratio(value: float, reference: float) -> float:
    if reference <= 0:
        raise ValueError("Cost reference must be positive")
    return value / reference


def _finite(value: Any) -> bool:
    return value is not None and math.isfinite(float(value))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--frozen-score", type=Path, required=True)
    parser.add_argument("--frozen-cases", type=Path, required=True)
    parser.add_argument("--max-epochs", type=int, default=MAX_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    args = parser.parse_args()
    if args.max_epochs != MAX_EPOCHS or args.batch_size != BATCH_SIZE:
        raise ValueError("P-GDN3-067 fixes one 10-epoch/batch32 endpoint")
    expected = {
        args.frozen_score: EXPECTED_FROZEN_SCORE_SHA256,
        args.frozen_cases: EXPECTED_FROZEN_CASES_SHA256,
    }
    for path, digest in expected.items():
        if not path.is_file() or _sha256(path) != digest:
            raise RuntimeError(f"Frozen artifact drifted: {path}")

    frozen = json.loads(args.frozen_score.read_text())
    control = frozen["candidate"]
    control_swaps = frozen["candidate_swap_summary"]
    control_cases = json.loads(args.frozen_cases.read_text())
    args.output_dir.mkdir(parents=True, exist_ok=True)
    candidate = run_arm(
        arm=CANDIDATE_ARM,
        sequence_length=SEQUENCE_LENGTH,
        num_kv_pairs=NUM_KV_PAIRS,
        output_dir=args.output_dir,
        max_epochs=MAX_EPOCHS,
        batch_size=BATCH_SIZE,
        save_checkpoint=True,
    )
    candidate_root = args.output_dir / "length_1024" / CANDIDATE_ARM
    cases_path = candidate_root / "cases.json"
    candidate_cases = json.loads(cases_path.read_text())
    candidate_swaps = wrong_key_swap_summary(candidate_cases)
    transitions = transition_summary(control_cases, candidate_cases)

    diagnostics = candidate["erase_then_delta"]
    rows = diagnostics["per_layer"]
    future_seed = candidate["future_seed"]
    integrity_checks = {
        "fixed_protocol": (
            candidate["arm"] == CANDIDATE_ARM
            and candidate["carrier_arm"] == CANDIDATE_ARM
            and candidate["sequence_length"] == SEQUENCE_LENGTH
            and candidate["num_kv_pairs"] == NUM_KV_PAIRS
            and candidate["model_width"] == 128
            and candidate["model_heads"] == NUM_HEADS
            and candidate["gdn2_head_dim"] == HEAD_DIM
            and candidate["epochs"] == MAX_EPOCHS
        ),
        "matched_data_hashes": candidate["data_hashes"] == control["data_hashes"],
        "matched_warmup_batch": (
            candidate["warmup_batch_hash"] == control["warmup_batch_hash"]
        ),
        "foundational_from_scratch": candidate["parent_init_parameter_hash"] is None,
        "exact_state_and_order": (
            diagnostics["microsteps_per_token"] == NUM_MICROSTEPS
            and diagnostics["state_values_per_layer"] == STATE_VALUES_PER_LAYER
            and candidate["recurrent_state_values_per_layer"]
            == STATE_VALUES_PER_LAYER
            and diagnostics["first_microstep"]
            == "independent_erase_zero_payload"
            and diagnostics["second_microstep"]
            == "standard_delta_correction"
        ),
        "checkpoint_complete": (
            candidate["checkpoint_path"] is not None
            and candidate["checkpoint_sha256"] is not None
            and Path(candidate["checkpoint_path"]).is_file()
            and _sha256(Path(candidate["checkpoint_path"]))
            == candidate["checkpoint_sha256"]
        ),
    }
    activation_checks = {
        "two_layers_one_seed_route": (
            diagnostics["active_layers"] == 2
            and len(rows) == 2
            and future_seed["active_seed_routes"] == 1
        ),
        "erase_write_addresses_distinct": all(
            _finite(row["erase_write_key_relative_rms"])
            and row["erase_write_key_relative_rms"] >= 0.10
            and _finite(row["erase_write_key_abs_cosine"])
            and row["erase_write_key_abs_cosine"] <= 0.98
            for row in rows
        ),
        "gates_active_and_variable": all(
            _finite(row["gamma_mean"])
            and 0.01 <= row["gamma_mean"] <= 0.99
            and _finite(row["beta_mean"])
            and 0.01 <= row["beta_mean"] <= 0.99
            and _finite(row["gamma_std"])
            and row["gamma_std"] >= 1e-4
            and _finite(row["beta_std"])
            and row["beta_std"] >= 1e-4
            for row in rows
        ),
        "pure_erase_and_material_effect": all(
            row["first_payload_max_abs"] == 0
            and _finite(row["erase_relative_rms"])
            and row["erase_relative_rms"] >= 1e-4
            for row in rows
        ),
        "explicit_replay_stable": all(
            _finite(row["explicit_terminal_relative_rms"])
            and row["explicit_terminal_relative_rms"] <= 0.05
            for row in rows
        ),
        "terminal_state_bounded_and_variable": all(
            _finite(row["terminal_state_rms"])
            and 1e-4 <= row["terminal_state_rms"] <= 1e4
            and _finite(row["terminal_state_board_std"])
            and row["terminal_state_board_std"] > 1e-6
            for row in rows
        ),
    }

    control_metrics = control["metrics"]
    metrics = candidate["metrics"]
    quality_checks = {
        "balanced_gain_at_least_0.005": (
            metrics["balanced_accuracy"]
            >= control_metrics["balanced_accuracy"] + 0.005
        ),
        "future_regression_at_most_0.003": (
            metrics["future"]["accuracy"]
            >= control_metrics["future"]["accuracy"] - 0.003
        ),
        "past_regression_at_most_0.003": (
            metrics["past"]["accuracy"]
            >= control_metrics["past"]["accuracy"] - 0.003
        ),
        "joint_gain_at_least_0.005": (
            metrics["joint_exact"] >= control_metrics["joint_exact"] + 0.005
        ),
        "errors_at_most_200": candidate_swaps["errors"] <= 200,
        "wrong_key_swaps_at_most_120": (
            candidate_swaps["wrong_key_valid_value_swaps"] <= 120
        ),
        "wrong_key_swap_share_at_most_0.60": (
            candidate_swaps["wrong_key_swap_fraction_of_errors"] <= 0.60
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
        "elapsed_at_most_2.25x": cost_ratios["elapsed"] <= 2.25,
        "post_warm_wall_at_most_2.25x": cost_ratios["post_warm_wall"] <= 2.25,
        "warmed_step_at_most_2.25x": cost_ratios["warmed_step"] <= 2.25,
        "peak_allocation_at_most_1.50x": cost_ratios["peak_allocation"] <= 1.50,
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
            "plan": "P-GDN3-067",
            "mechanism": "structured zero-payload erase then standard delta correction",
            "candidate_only": True,
            "frozen_control": control["arm"],
            "sequence_length": SEQUENCE_LENGTH,
            "num_kv_pairs": NUM_KV_PAIRS,
            "epochs": MAX_EPOCHS,
            "batch_size": BATCH_SIZE,
            "seed": SEED,
            "heads": NUM_HEADS,
            "head_dim": HEAD_DIM,
            "microsteps_per_token": NUM_MICROSTEPS,
            "state_values_per_layer": STATE_VALUES_PER_LAYER,
            "sweep": None,
        },
        "frozen_artifacts": {
            "score": str(args.frozen_score),
            "score_sha256": EXPECTED_FROZEN_SCORE_SHA256,
            "cases": str(args.frozen_cases),
            "cases_sha256": EXPECTED_FROZEN_CASES_SHA256,
        },
        "control": control,
        "control_swap_summary": control_swaps,
        "candidate": candidate,
        "candidate_swap_summary": candidate_swaps,
        "paired_error_transitions": transitions,
        "cost_ratios": cost_ratios,
        "registered_gate": {
            "passed": passed,
            "integrity_checks": integrity_checks,
            "activation_checks": activation_checks,
            "quality_checks": quality_checks,
            "cost_checks": cost_checks,
        },
        "decision": "better_gdn3_pending_fs2_isolation" if passed else "closed",
    }
    (args.output_dir / "comparison.json").write_text(
        json.dumps(comparison, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(comparison, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
