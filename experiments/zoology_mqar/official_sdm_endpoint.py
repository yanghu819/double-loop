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
from experiments.zoology_mqar.gdn2_log_spd_endpoint import (
    wrong_key_swap_summary,
)
from experiments.zoology_mqar.length_scaling import run_arm
from experiments.zoology_mqar.official_sdm_futureseed import (
    EXPECTED_SDM_SHA,
    EXPECTED_SDM_TREE_SHA256,
    EXPECTED_STATE_VALUES_PER_LAYER,
    SDM_BLOCK_SIZE,
    SDM_HEADS,
    SDM_READS,
    SDM_SLOTS,
    SDM_WRITES,
)


SEQUENCE_LENGTH = 1024
NUM_KV_PAIRS = 4
MAX_EPOCHS = 10
BATCH_SIZE = 32
SEED = 123
CANDIDATE_ARM = "future_seed_official_sdm"
EXPECTED_MATCHED_INIT_SHA256 = (
    "7402e46c48cbd47070d65262d1a1b62a32ac55a4d644a6fb9644b96b0914850f"
)
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


def _slot_count(percent: float) -> float:
    return percent * SDM_HEADS * SDM_SLOTS / 100.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--matched-init", type=Path, required=True)
    parser.add_argument("--frozen-score", type=Path, required=True)
    parser.add_argument("--frozen-cases", type=Path, required=True)
    parser.add_argument("--max-epochs", type=int, default=MAX_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    args = parser.parse_args()
    if args.max_epochs != MAX_EPOCHS or args.batch_size != BATCH_SIZE:
        raise ValueError("P-GDN3-064 has one fixed 10-epoch/batch32 endpoint")

    expected_hashes = {
        args.matched_init: EXPECTED_MATCHED_INIT_SHA256,
        args.frozen_score: EXPECTED_FROZEN_SCORE_SHA256,
        args.frozen_cases: EXPECTED_FROZEN_CASES_SHA256,
    }
    for path, expected in expected_hashes.items():
        if not path.is_file() or _sha256(path) != expected:
            raise RuntimeError(f"Frozen artifact drifted: {path}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    frozen = json.loads(args.frozen_score.read_text())
    control = frozen["candidate"]
    control_swaps = frozen["candidate_swap_summary"]
    control_cases = json.loads(args.frozen_cases.read_text())
    candidate = run_arm(
        arm=CANDIDATE_ARM,
        sequence_length=SEQUENCE_LENGTH,
        num_kv_pairs=NUM_KV_PAIRS,
        output_dir=args.output_dir,
        max_epochs=MAX_EPOCHS,
        batch_size=BATCH_SIZE,
        save_checkpoint=True,
        matched_init_path=args.matched_init,
    )
    cases_path = args.output_dir / "length_1024" / CANDIDATE_ARM / "cases.json"
    candidate_cases = json.loads(cases_path.read_text())
    candidate_swaps = wrong_key_swap_summary(candidate_cases)
    transitions = transition_summary(control_cases, candidate_cases)

    diagnostics = candidate["official_sdm"]
    rows = diagnostics["per_layer"]
    matched_parent = diagnostics["matched_parent"]
    integrity_checks = {
        "external_source_exact": (
            diagnostics["external"]["sha"] == EXPECTED_SDM_SHA
            and diagnostics["external"]["tree_sha256"]
            == EXPECTED_SDM_TREE_SHA256
            and diagnostics["external"]["redistributed_source"] is False
        ),
        "shared_parent_mapping_exact": (
            matched_parent is not None
            and matched_parent["tensor_count"] >= 20
            and matched_parent["numel"] >= 300_000
            and matched_parent["source_hash"] == matched_parent["loaded_hash"]
            and candidate["parent_init_parameter_hash"]
            == matched_parent["loaded_hash"]
        ),
        "matched_data_hashes": candidate["data_hashes"] == control["data_hashes"],
        "matched_warmup_batch": (
            candidate["warmup_batch_hash"] == control["warmup_batch_hash"]
        ),
        "state_and_sparse_geometry_exact": (
            diagnostics["heads"] == SDM_HEADS
            and diagnostics["slots_per_head"] == SDM_SLOTS
            and diagnostics["reads"] == SDM_READS
            and diagnostics["writes"] == SDM_WRITES
            and diagnostics["block_size"] == SDM_BLOCK_SIZE
            and diagnostics["state_values_per_layer"]
            == EXPECTED_STATE_VALUES_PER_LAYER
            and candidate["recurrent_state_values_per_layer"]
            == EXPECTED_STATE_VALUES_PER_LAYER
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
        "two_sparse_layers_and_one_seed_route": (
            diagnostics["active_layers"] == 2
            and diagnostics["active_futureseed_routes"] == 1
            and len(rows) == 2
        ),
        "more_than_64_slots_read_and_written_per_layer": all(
            row["access"] is not None
            and _slot_count(row["access"]["read_unique_pct"]) > 64
            and _slot_count(row["access"]["write_unique_pct"]) > 64
            for row in rows
        ),
        "finite_nonzero_access_entropy": all(
            _finite(row["access"]["read_slot_entropy_normalized"])
            and row["access"]["read_slot_entropy_normalized"] > 0
            and _finite(row["access"]["write_slot_entropy_normalized"])
            and row["access"]["write_slot_entropy_normalized"] > 0
            for row in rows
        ),
        "finite_nonzero_state_and_board_variation": all(
            _finite(row["state_rms"])
            and row["state_rms"] >= 1e-4
            and _finite(row["state_board_std"])
            and row["state_board_std"] > 0
            and row["active_slots"] is not None
            and row["active_slots"] > 64
            for row in rows
        ),
        "futureseed_gate_bounded_and_active": (
            rows[0]["seed_applied"] is False
            and rows[1]["seed_applied"] is True
            and _finite(rows[1]["seed_gate"])
            and 0.05 <= rows[1]["seed_gate"] <= 0.95
        ),
    }

    control_metrics = control["metrics"]
    candidate_metrics = candidate["metrics"]
    quality_checks = {
        "balanced_at_least_0.94": candidate_metrics["balanced_accuracy"] >= 0.94,
        "future_at_least_0.93": candidate_metrics["future"]["accuracy"] >= 0.93,
        "past_at_least_0.93": candidate_metrics["past"]["accuracy"] >= 0.93,
        "joint_at_least_0.82": candidate_metrics["joint_exact"] >= 0.82,
        "errors_at_most_223": candidate_swaps["errors"] <= 223,
        "wrong_key_swaps_at_most_100": (
            candidate_swaps["wrong_key_valid_value_swaps"] <= 100
        ),
        "wrong_key_swap_share_at_most_0.60": (
            candidate_swaps["wrong_key_swap_fraction_of_errors"] <= 0.60
        ),
        "strict_pareto_gain": (
            candidate_metrics["balanced_accuracy"]
            > control_metrics["balanced_accuracy"]
            or candidate_metrics["joint_exact"] > control_metrics["joint_exact"]
            or candidate_swaps["errors"] < control_swaps["errors"]
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
        "elapsed_at_most_3x": cost_ratios["elapsed"] <= 3.0,
        "post_warm_wall_at_most_3x": cost_ratios["post_warm_wall"] <= 3.0,
        "warmed_step_at_most_3x": cost_ratios["warmed_step"] <= 3.0,
        "peak_allocation_at_most_4x": cost_ratios["peak_allocation"] <= 4.0,
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
            "plan": "P-GDN3-064",
            "mechanism": "pinned official sparse product-key delta recurrence with full-bank native FutureSeed",
            "candidate_only": True,
            "frozen_control": control["arm"],
            "sequence_length": SEQUENCE_LENGTH,
            "num_kv_pairs": NUM_KV_PAIRS,
            "epochs": MAX_EPOCHS,
            "batch_size": BATCH_SIZE,
            "seed": SEED,
            "external_source_sha": EXPECTED_SDM_SHA,
            "state_values_per_layer": EXPECTED_STATE_VALUES_PER_LAYER,
            "sweep": None,
        },
        "frozen_artifacts": {
            "matched_init": str(args.matched_init),
            "matched_init_sha256": EXPECTED_MATCHED_INIT_SHA256,
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
        "decision": "better_gdn3_and_fs2" if passed else "closed",
    }
    (args.output_dir / "comparison.json").write_text(
        json.dumps(comparison, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(comparison, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
