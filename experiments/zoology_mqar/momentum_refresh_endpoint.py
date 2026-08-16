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
from experiments.zoology_mqar.momentum_futureseed import (
    EXPECTED_MDN_SHA,
    EXPECTED_STATE_VALUES_PER_LAYER,
)


SEQUENCE_LENGTH = 1024
NUM_KV_PAIRS = 4
MAX_EPOCHS = 10
BATCH_SIZE = 32
SEED = 123
CANDIDATE_ARM = "future_seed_momentum_refresh"
EXPECTED_MATCHED_INIT_SHA256 = (
    "7402e46c48cbd47070d65262d1a1b62a32ac55a4d644a6fb9644b96b0914850f"
)
EXPECTED_FROZEN_MOMENTUM_SCORE_SHA256 = (
    "7201d32834d3f9af4d1057750ba4eb62df492aa495eec770f9dad85f3d4f6e8a"
)
EXPECTED_FROZEN_MOMENTUM_CASES_SHA256 = (
    "f64b0ae0e45a65a8da2934ece8cfe208fedd835a373d79230320bdf945826f20"
)
EXPECTED_INIT_PARAMETER_HASH = (
    "dc8f49f92c00f0f08863cf6183692246cf2c44f0c5aa3a35892bb9555ec17bfb"
)
EXPECTED_PARENT_PARAMETER_HASH = (
    "0adf26f657e59a35e90d7e54b905721b10f537b6391d107676248de16b71f13f"
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


def _finite(rows: list[dict[str, Any]], key: str) -> bool:
    return all(row[key] is not None and math.isfinite(float(row[key])) for row in rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--matched-init", type=Path, required=True)
    parser.add_argument("--frozen-momentum-score", type=Path, required=True)
    parser.add_argument("--frozen-momentum-cases", type=Path, required=True)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--max-epochs", type=int, default=MAX_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    args = parser.parse_args()
    if args.max_epochs != MAX_EPOCHS or args.batch_size != BATCH_SIZE:
        raise ValueError("P-GDN3-061 has one fixed 10-epoch/batch32 endpoint")

    expected_hashes = {
        args.matched_init: EXPECTED_MATCHED_INIT_SHA256,
        args.frozen_momentum_score: EXPECTED_FROZEN_MOMENTUM_SCORE_SHA256,
        args.frozen_momentum_cases: EXPECTED_FROZEN_MOMENTUM_CASES_SHA256,
    }
    for path, expected in expected_hashes.items():
        if not path.is_file() or _sha256(path) != expected:
            raise RuntimeError(f"Frozen artifact drifted: {path}")
    contract = json.loads(args.contract.read_text())
    if contract.get("status") != "passed" or contract.get("plan") != "P-GDN3-061":
        raise RuntimeError("P-GDN3-061 strict contract is missing")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    frozen = json.loads(args.frozen_momentum_score.read_text())
    reference = frozen["candidate"]
    reference_swaps = frozen["candidate_swap_summary"]
    reference_cases = json.loads(args.frozen_momentum_cases.read_text())
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
    transitions = transition_summary(reference_cases, candidate_cases)
    diagnostics = candidate["momentum_delta"]
    state_rows = diagnostics["per_layer"]
    refresh_rows = diagnostics["refresh_per_layer"]
    refresh_stats = [row["refresh"] for row in refresh_rows]
    matched_parent = diagnostics["matched_parent"]

    integrity_checks = {
        "external_source_exact": diagnostics["external_sha"] == EXPECTED_MDN_SHA,
        "p059_initialization_exact": (
            reference["init_parameter_hash"] == EXPECTED_INIT_PARAMETER_HASH
            and candidate["init_parameter_hash"] == EXPECTED_INIT_PARAMETER_HASH
            and reference["parent_init_parameter_hash"]
            == EXPECTED_PARENT_PARAMETER_HASH
            and candidate["parent_init_parameter_hash"]
            == EXPECTED_PARENT_PARAMETER_HASH
        ),
        "shared_parent_mapping_exact": (
            matched_parent is not None
            and matched_parent["source_hash"] == matched_parent["loaded_hash"]
            and matched_parent["loaded_hash"] == EXPECTED_PARENT_PARAMETER_HASH
        ),
        "matched_data_and_warmup": (
            candidate["data_hashes"] == reference["data_hashes"]
            and candidate["warmup_batch_hash"] == reference["warmup_batch_hash"]
        ),
        "parameter_and_state_delta_zero": (
            candidate["parameters"] == reference["parameters"]
            and candidate["recurrent_state_values_per_layer"]
            == reference["recurrent_state_values_per_layer"]
            == EXPECTED_STATE_VALUES_PER_LAYER
            and diagnostics["parameter_delta_vs_momentum"] == 0
            and diagnostics["persistent_state_delta_vs_momentum"] == 0
        ),
        "strict_contract_passed": bool(
            contract["registered_contract"]["passed"]
        ),
    }
    activation_checks = {
        "both_momentum_and_refresh_layers_active": (
            diagnostics["active_layers"] == 2
            and diagnostics["refresh_layers"] == 2
            and diagnostics["active_refresh_layers"] == 2
        ),
        "native_futureseed_active": diagnostics["active_futureseed_routes"] == 1,
        "state_and_momentum_finite": (
            _finite(state_rows, "state_rms")
            and _finite(state_rows, "momentum_rms")
            and _finite(state_rows, "momentum_to_state_rms")
        ),
        "exact_refresh_schedule": all(
            row is not None
            and row["parent_tokens"] == SEQUENCE_LENGTH
            and row["refresh_tokens"] == SEQUENCE_LENGTH
            and row["micro_tokens"] == 2 * SEQUENCE_LENGTH
            and row["refresh_log_alpha_max_abs"] == 0.0
            and row["refresh_log_mu_max_abs"] == 0.0
            and row["refresh_beta_max_abs"] == 0.0
            for row in refresh_stats
        ),
        "refresh_cannot_change_state_read": all(
            row["refresh_output_relative_rms"] <= 1e-5
            for row in refresh_stats
        ),
        "synthetic_momentum_refresh_active": (
            contract["synthetic_refresh"]["momentum_change_relative_rms"] >= 1e-4
            and contract["synthetic_refresh"]["state_change_max_abs"] <= 1e-6
            and contract["synthetic_refresh"]["token_variation"] >= 1e-4
        ),
    }

    reference_metrics = reference["metrics"]
    candidate_metrics = candidate["metrics"]
    quality_checks = {
        "balanced_at_least_0.955_and_gain_0.01": (
            candidate_metrics["balanced_accuracy"] >= 0.955
            and candidate_metrics["balanced_accuracy"]
            - reference_metrics["balanced_accuracy"]
            >= 0.01
        ),
        "joint_at_least_0.84_and_gain_0.01": (
            candidate_metrics["joint_exact"] >= 0.84
            and candidate_metrics["joint_exact"]
            - reference_metrics["joint_exact"]
            >= 0.01
        ),
        "directions_retained": (
            candidate_metrics["future"]["accuracy"]
            >= reference_metrics["future"]["accuracy"] - 0.005
            and candidate_metrics["past"]["accuracy"]
            >= reference_metrics["past"]["accuracy"] - 0.005
        ),
        "errors_at_most_180": candidate_swaps["errors"] <= 180,
        "wrong_key_swaps_at_most_105": (
            candidate_swaps["wrong_key_valid_value_swaps"] <= 105
        ),
        "conditional_wrong_key_fraction_reduced_0.07": (
            candidate_swaps["wrong_key_swap_fraction_of_errors"] <= 0.60713
            and reference_swaps["wrong_key_swap_fraction_of_errors"]
            - candidate_swaps["wrong_key_swap_fraction_of_errors"]
            >= 0.07
        ),
    }
    cost_ratios = {
        "elapsed": _ratio(
            candidate["elapsed_sec_including_validation"],
            reference["elapsed_sec_including_validation"],
        ),
        "post_warm_wall": _ratio(
            candidate["post_warm_arm_wall_sec_through_checkpoint"],
            reference["post_warm_arm_wall_sec_through_checkpoint"],
        ),
        "warmed_step": _ratio(
            candidate["warmed_step_benchmark"]["elapsed_sec"],
            reference["warmed_step_benchmark"]["elapsed_sec"],
        ),
        "peak_allocation": _ratio(
            candidate["peak_training_cuda_mem_bytes"],
            reference["peak_training_cuda_mem_bytes"],
        ),
    }
    cost_checks = {
        "elapsed_below_2.25x": cost_ratios["elapsed"] < 2.25,
        "post_warm_wall_below_2.25x": cost_ratios["post_warm_wall"] < 2.25,
        "warmed_step_below_2.25x": cost_ratios["warmed_step"] < 2.25,
        "peak_allocation_below_1.20x": cost_ratios["peak_allocation"] < 1.20,
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
            "plan": "P-GDN3-061",
            "mechanism": "post-commit residual-only Momentum refresh",
            "candidate_only": True,
            "frozen_reference": reference["arm"],
            "sequence_length": SEQUENCE_LENGTH,
            "num_kv_pairs": NUM_KV_PAIRS,
            "epochs": MAX_EPOCHS,
            "batch_size": BATCH_SIZE,
            "seed": SEED,
            "logical_to_micro_token_ratio": 2,
            "parameter_delta_vs_reference": 0,
            "persistent_state_delta_vs_reference": 0,
            "sweep": None,
        },
        "frozen_artifacts": {
            "matched_init": str(args.matched_init),
            "matched_init_sha256": EXPECTED_MATCHED_INIT_SHA256,
            "momentum_score": str(args.frozen_momentum_score),
            "momentum_score_sha256": EXPECTED_FROZEN_MOMENTUM_SCORE_SHA256,
            "momentum_cases": str(args.frozen_momentum_cases),
            "momentum_cases_sha256": EXPECTED_FROZEN_MOMENTUM_CASES_SHA256,
            "contract": str(args.contract),
            "contract_sha256": _sha256(args.contract),
        },
        "reference": reference,
        "reference_swap_summary": reference_swaps,
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
        "decision": "authorize_one_sudoku_transfer" if passed else "closed",
    }
    (args.output_dir / "comparison.json").write_text(
        json.dumps(comparison, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(comparison, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
