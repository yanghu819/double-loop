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
from experiments.zoology_mqar.momentum_futureseed_component_diagnostic import (
    _owner_topology,
)
from experiments.zoology_mqar.momentum_prediction_key import (
    EXPECTED_PARAMETER_DELTA_VS_MOMENTUM,
)


SEQUENCE_LENGTH = 1024
NUM_KV_PAIRS = 4
MAX_EPOCHS = 10
BATCH_SIZE = 32
SEED = 123
CANDIDATE_ARM = "future_seed_momentum_prediction_key"
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
        raise ValueError("P-GDN3-063 has one fixed 10-epoch/batch32 endpoint")

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
    control_topology = _owner_topology(control_cases)
    candidate_topology = _owner_topology(candidate_cases)
    transitions = transition_summary(control_cases, candidate_cases)
    diagnostics = candidate["momentum_delta"]
    momentum = diagnostics["momentum"]
    momentum_rows = momentum["per_layer"]
    prediction_rows = diagnostics["per_layer_prediction"]
    matched_parent = momentum["matched_parent"]

    integrity_checks = {
        "external_source_exact": momentum["external_sha"] == EXPECTED_MDN_SHA,
        "shared_parent_mapping_exact": (
            matched_parent is not None
            and matched_parent["tensor_count"] >= 20
            and matched_parent["numel"] >= 400_000
            and matched_parent["source_hash"] == matched_parent["loaded_hash"]
            and candidate["parent_init_parameter_hash"]
            == matched_parent["loaded_hash"]
            and candidate["parent_init_parameter_hash"]
            == control["parent_init_parameter_hash"]
        ),
        "matched_data_hashes": candidate["data_hashes"] == control["data_hashes"],
        "matched_warmup_batch": (
            candidate["warmup_batch_hash"] == control["warmup_batch_hash"]
        ),
        "parameter_delta_exact": (
            candidate["parameters"] - control["parameters"]
            == EXPECTED_PARAMETER_DELTA_VS_MOMENTUM
            and diagnostics["parameter_delta_vs_momentum"]
            == EXPECTED_PARAMETER_DELTA_VS_MOMENTUM
        ),
        "state_geometry_exact": (
            candidate["recurrent_state_values_per_layer"]
            == EXPECTED_STATE_VALUES_PER_LAYER
            and momentum["state_values_per_layer"]
            == EXPECTED_STATE_VALUES_PER_LAYER
            and momentum["state_components"] == 2
        ),
        "parent_prediction_tie_exact": (
            diagnostics["parent_tie"] is not None
            and diagnostics["parent_tie"]["exact"]
            and len(diagnostics["parent_tie"]["paths"]) == 4
            and diagnostics["persistent_state_delta_vs_momentum"] == 0
        ),
    }
    activation_checks = {
        "both_prediction_layers_active": (
            diagnostics["prediction_layers"] == 2
            and diagnostics["active_prediction_layers"] == 2
        ),
        "prediction_key_meaningfully_diverged": all(
            row["prediction"] is not None
            and _finite(row["prediction"]["relative_rms"])
            and row["prediction"]["relative_rms"] >= 0.01
            and _finite(row["prediction"]["cosine_mean"])
            and 0.80 <= row["prediction"]["cosine_mean"] <= 0.9999
            and _finite(row["prediction"]["board_rms_std"])
            and _finite(row["prediction"]["token_rms_std"])
            and _finite(row["prediction"]["head_rms_std"])
            and row["prediction"]["board_rms_std"] > 0.0
            and row["prediction"]["token_rms_std"] > 0.0
            and row["prediction"]["head_rms_std"] > 0.0
            for row in prediction_rows
        ),
        "both_momentum_layers_active": momentum["active_layers"] == 2,
        "native_futureseed_active": momentum["active_futureseed_routes"] == 1,
        "state_and_momentum_stable": all(
            _finite(row["state_rms"])
            and _finite(row["momentum_rms"])
            and _finite(row["momentum_to_state_rms"])
            and row["state_rms"] >= 1e-4
            and row["momentum_rms"] >= 1e-4
            and 0.01 <= row["momentum_to_state_rms"] <= 20.0
            for row in momentum_rows
        ),
    }

    control_metrics = control["metrics"]
    candidate_metrics = candidate["metrics"]
    quality_checks = {
        "balanced_at_least_0.955_and_gain_0.01": (
            candidate_metrics["balanced_accuracy"] >= 0.955
            and candidate_metrics["balanced_accuracy"]
            - control_metrics["balanced_accuracy"]
            >= 0.01
        ),
        "directions_regress_at_most_0.005": (
            candidate_metrics["future"]["accuracy"]
            >= control_metrics["future"]["accuracy"] - 0.005
            and candidate_metrics["past"]["accuracy"]
            >= control_metrics["past"]["accuracy"] - 0.005
        ),
        "joint_at_least_0.84_and_gain_0.01": (
            candidate_metrics["joint_exact"] >= 0.84
            and candidate_metrics["joint_exact"]
            - control_metrics["joint_exact"]
            >= 0.01
        ),
        "errors_reduced_at_least_20_percent": (
            candidate_swaps["errors"] <= 178
            and candidate_swaps["errors"] <= 0.80 * control_swaps["errors"]
        ),
        "wrong_key_swaps_reduced_at_least_20_percent": (
            candidate_swaps["wrong_key_valid_value_swaps"] <= 120
            and candidate_swaps["wrong_key_valid_value_swaps"]
            <= 0.80 * control_swaps["wrong_key_valid_value_swaps"]
        ),
        "conditional_wrong_key_share_reduced_at_least_0.05": (
            control_swaps["wrong_key_swap_fraction_of_errors"]
            - candidate_swaps["wrong_key_swap_fraction_of_errors"]
            >= 0.05
        ),
        "adjacent_owner_tail_reduced": (
            candidate_topology["adjacent_write_rank_swaps"]
            <= 0.80 * control_topology["adjacent_write_rank_swaps"]
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
        "elapsed_below_1.30x": cost_ratios["elapsed"] < 1.30,
        "post_warm_wall_below_1.30x": cost_ratios["post_warm_wall"] < 1.30,
        "warmed_step_below_1.30x": cost_ratios["warmed_step"] < 1.30,
        "peak_allocation_below_1.10x": cost_ratios["peak_allocation"] < 1.10,
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
            "plan": "P-GDN3-063",
            "mechanism": "second-order Momentum with an independently learned residual-prediction key, unchanged owner/write K, and native [S,M] FutureSeed",
            "candidate_only": True,
            "frozen_control": control["arm"],
            "sequence_length": SEQUENCE_LENGTH,
            "num_kv_pairs": NUM_KV_PAIRS,
            "epochs": MAX_EPOCHS,
            "batch_size": BATCH_SIZE,
            "seed": SEED,
            "external_source_sha": EXPECTED_MDN_SHA,
            "parameter_delta_vs_control": EXPECTED_PARAMETER_DELTA_VS_MOMENTUM,
            "recurrent_state_values_per_layer": EXPECTED_STATE_VALUES_PER_LAYER,
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
        "control_owner_topology": control_topology,
        "candidate": candidate,
        "candidate_swap_summary": candidate_swaps,
        "candidate_owner_topology": candidate_topology,
        "paired_error_transitions": transitions,
        "cost_ratios": cost_ratios,
        "registered_gate": {
            "passed": passed,
            "integrity_checks": integrity_checks,
            "activation_checks": activation_checks,
            "quality_checks": quality_checks,
            "cost_checks": cost_checks,
        },
        "decision": "admit_one_sudoku_transfer" if passed else "closed",
    }
    (args.output_dir / "comparison.json").write_text(
        json.dumps(comparison, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(comparison, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
