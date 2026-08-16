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
from experiments.zoology_mqar.momentum_conv_prefill_futureseed import (
    PREFILL_PARAMETERS,
    PREFILL_TOKENS,
)
from experiments.zoology_mqar.momentum_futureseed import (
    EXPECTED_MDN_SHA,
    EXPECTED_STATE_VALUES_PER_LAYER,
)


SEQUENCE_LENGTH = 1024
NUM_KV_PAIRS = 4
MAX_EPOCHS = 10
BATCH_SIZE = 32
CANDIDATE_ARM = "future_seed_momentum_conv_prefill"
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


def _finite(value: Any) -> bool:
    return value is not None and math.isfinite(float(value))


def _adjacent_wrong_key_swaps(cases: list[dict[str, Any]]) -> int:
    count = 0
    for case in cases:
        events = case["events"]
        ordered = sorted(
            range(len(events)), key=lambda index: events[index]["write_position"]
        )
        rank = {event_index: write_rank for write_rank, event_index in enumerate(ordered)}
        for event_index, event in enumerate(events):
            if event["correct"]:
                continue
            owner = next(
                (
                    index
                    for index, candidate in enumerate(events)
                    if index != event_index
                    and candidate["target"] == event["prediction"]
                ),
                None,
            )
            if owner is not None and abs(rank[event_index] - rank[owner]) == 1:
                count += 1
    return count


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
        raise ValueError("P-FS2-015 fixes one 10-epoch batch32 endpoint")

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
    control_cases = json.loads(args.frozen_cases.read_text())
    control_swaps = frozen["candidate_swap_summary"]
    control_adjacent = _adjacent_wrong_key_swaps(control_cases)

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
    cases_path = (
        args.output_dir
        / "length_1024"
        / CANDIDATE_ARM
        / "cases.json"
    )
    candidate_cases = json.loads(cases_path.read_text())
    candidate_swaps = wrong_key_swap_summary(candidate_cases)
    candidate_adjacent = _adjacent_wrong_key_swaps(candidate_cases)
    diagnostics = candidate["momentum_delta"]
    route = diagnostics["per_route"][0] if diagnostics["per_route"] else {}

    integrity_checks = {
        "external_source_exact": diagnostics["external_sha"] == EXPECTED_MDN_SHA,
        "shared_parent_mapping_exact": (
            diagnostics["matched_parent"] is not None
            and diagnostics["matched_parent"]["source_hash"]
            == diagnostics["matched_parent"]["loaded_hash"]
            and candidate["parent_init_parameter_hash"]
            == diagnostics["matched_parent"]["loaded_hash"]
        ),
        "matched_data_hashes": candidate["data_hashes"] == control["data_hashes"],
        "matched_warmup_batch": (
            candidate["warmup_batch_hash"] == control["warmup_batch_hash"]
        ),
        "parameter_delta_exact": (
            candidate["parameters"] == control["parameters"] + PREFILL_PARAMETERS
        ),
        "state_and_scan_unchanged": (
            candidate["recurrent_state_values_per_layer"]
            == EXPECTED_STATE_VALUES_PER_LAYER
            and diagnostics["state_values_per_layer"]
            == EXPECTED_STATE_VALUES_PER_LAYER
            and diagnostics["persistent_state_delta"] == 0
            and diagnostics["scan_delta"] == 0
            and diagnostics["prefill_tokens"] == PREFILL_TOKENS
        ),
    }
    activation_checks = {
        "two_momentum_layers_active": diagnostics["active_layers"] == 2,
        "one_native_futureseed_route": diagnostics["active_futureseed_routes"] == 1,
        "one_conv_prefill_route": (
            diagnostics["receiving_routes"] == 1
            and diagnostics["active_receiving_routes"] == 1
            and route.get("calls", 0) > 0
        ),
        "prefill_gate_active": (
            _finite(route.get("mix_abs")) and route["mix_abs"] >= 1e-3
        ),
        "evidence_and_cache_vary_by_board": (
            _finite(route.get("evidence_rms"))
            and route["evidence_rms"] >= 1e-4
            and _finite(route.get("evidence_board_std"))
            and route["evidence_board_std"] > 0
            and _finite(route.get("raw_cache_rms"))
            and route["raw_cache_rms"] >= 1e-4
            and _finite(route.get("mixed_cache_rms"))
            and route["mixed_cache_rms"] >= 1e-4
            and _finite(route.get("mixed_cache_board_std"))
            and route["mixed_cache_board_std"] > 0
        ),
        "state_and_momentum_bounded": all(
            _finite(row["state_rms"])
            and _finite(row["momentum_rms"])
            and row["state_rms"] >= 1e-4
            and row["momentum_rms"] >= 1e-4
            and 0.01 <= row["momentum_to_state_rms"] <= 20.0
            for row in diagnostics["per_layer"]
        ),
    }

    cm = control["metrics"]
    mm = candidate["metrics"]
    quality_checks = {
        "balanced_gain_at_least_0.005": (
            mm["balanced_accuracy"] >= cm["balanced_accuracy"] + 0.005
        ),
        "future_gain_at_least_0.005": (
            mm["future"]["accuracy"] >= cm["future"]["accuracy"] + 0.005
        ),
        "past_regression_at_most_0.003": (
            mm["past"]["accuracy"] >= cm["past"]["accuracy"] - 0.003
        ),
        "joint_gain_at_least_0.005": (
            mm["joint_exact"] >= cm["joint_exact"] + 0.005
        ),
        "errors_at_most_200": candidate_swaps["errors"] <= 200,
        "wrong_key_swaps_at_most_120": (
            candidate_swaps["wrong_key_valid_value_swaps"] <= 120
        ),
        "adjacent_swaps_reduced_at_least_20_percent": (
            candidate_adjacent <= int(control_adjacent * 0.8)
        ),
        "conditional_wrong_key_fraction_at_most_0.60": (
            candidate_swaps["wrong_key_swap_fraction_of_errors"] <= 0.60
        ),
    }
    cost_ratios = {
        "elapsed": (
            candidate["elapsed_sec_including_validation"]
            / control["elapsed_sec_including_validation"]
        ),
        "post_warm_wall": (
            candidate["post_warm_arm_wall_sec_through_checkpoint"]
            / control["post_warm_arm_wall_sec_through_checkpoint"]
        ),
        "warmed_step": (
            candidate["warmed_step_benchmark"]["elapsed_sec"]
            / control["warmed_step_benchmark"]["elapsed_sec"]
        ),
        "peak_allocation": (
            candidate["peak_training_cuda_mem_bytes"]
            / control["peak_training_cuda_mem_bytes"]
        ),
    }
    cost_checks = {
        "elapsed_below_1.15x": cost_ratios["elapsed"] < 1.15,
        "post_warm_wall_below_1.15x": cost_ratios["post_warm_wall"] < 1.15,
        "warmed_step_below_1.15x": cost_ratios["warmed_step"] < 1.15,
        "peak_allocation_below_1.05x": cost_ratios["peak_allocation"] < 1.05,
    }
    passed = all(
        all(group.values())
        for group in (
            integrity_checks,
            activation_checks,
            quality_checks,
            cost_checks,
        )
    )
    comparison = {
        "status": "complete",
        "decision": "passed" if passed else "closed",
        "protocol": {
            "plan": "P-FS2-015",
            "mechanism": "receiver-native terminal-hidden short-conv prefill",
            "candidate_only": True,
            "frozen_control": "future_seed_momentum_delta",
            "sequence_length": SEQUENCE_LENGTH,
            "num_kv_pairs": NUM_KV_PAIRS,
            "epochs": MAX_EPOCHS,
            "batch_size": BATCH_SIZE,
            "seed": 123,
            "sweep": None,
        },
        "frozen_artifacts": {
            "score": str(args.frozen_score),
            "score_sha256": _sha256(args.frozen_score),
            "cases": str(args.frozen_cases),
            "cases_sha256": _sha256(args.frozen_cases),
            "matched_init": str(args.matched_init),
            "matched_init_sha256": _sha256(args.matched_init),
        },
        "control": control,
        "candidate": candidate,
        "control_swap_summary": control_swaps,
        "candidate_swap_summary": candidate_swaps,
        "control_adjacent_wrong_key_swaps": control_adjacent,
        "candidate_adjacent_wrong_key_swaps": candidate_adjacent,
        "paired_error_transitions": transition_summary(
            control_cases, candidate_cases
        ),
        "cost_ratios": cost_ratios,
        "registered_gate": {
            "integrity_checks": integrity_checks,
            "activation_checks": activation_checks,
            "quality_checks": quality_checks,
            "cost_checks": cost_checks,
            "passed": passed,
        },
    }
    output_path = args.output_dir / "comparison.json"
    output_path.write_text(json.dumps(comparison, indent=2, sort_keys=True) + "\n")
    print(json.dumps(comparison, indent=2, sort_keys=True))
    raise SystemExit(0 if passed else 2)


if __name__ == "__main__":
    main()
