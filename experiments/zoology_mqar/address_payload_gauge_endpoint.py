from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

import torch
from zoology.utils import set_determinism

from experiments.zoology_mqar.address_payload_gauge_futureseed import (
    EXPECTED_STATE_VALUES,
    EXPECTED_NEW_PARAMETERS,
)
from experiments.zoology_mqar.gdn2_log_spd_endpoint import wrong_key_swap_summary
from experiments.zoology_mqar.length_scaling import (
    build_config,
    make_model,
    parameter_hash,
    run_arm,
)


SEQUENCE_LENGTH = 1024
NUM_KV_PAIRS = 4
MAX_EPOCHS = 10
BATCH_SIZE = 32
SEED = 123
EXPECTED_NATIVE_PARAMETERS = 661_584
EXPECTED_CANDIDATE_PARAMETERS = EXPECTED_NATIVE_PARAMETERS + EXPECTED_NEW_PARAMETERS


def _ratio(value: float, reference: float) -> float:
    if reference <= 0:
        raise ValueError("Cost reference must be positive")
    return value / reference


def _cases(root: Path, arm: str) -> list[dict[str, Any]]:
    return json.loads((root / "length_1024" / arm / "cases.json").read_text())


def _write_matched_initialization(output_dir: Path) -> tuple[Path, str]:
    config = build_config(
        arm="future_seed_gdn2",
        sequence_length=SEQUENCE_LENGTH,
        num_kv_pairs=NUM_KV_PAIRS,
        max_epochs=MAX_EPOCHS,
        batch_size=BATCH_SIZE,
    )
    set_determinism(SEED)
    model = make_model(config, "future_seed_gdn2")
    path = output_dir / "matched_parent_initialization.pt"
    torch.save(model.state_dict(), path)
    digest = parameter_hash(model)
    del model
    return path, digest


def _event_category(event: dict[str, Any], valid_targets: set[int]) -> str:
    if event["correct"]:
        return "correct"
    if int(event["prediction"]) in valid_targets - {int(event["target"])}:
        return "wrong_key"
    return "other_wrong"


def transition_summary(
    control_cases: list[dict[str, Any]],
    candidate_cases: list[dict[str, Any]],
) -> dict[str, int]:
    control_by_id = {case["case_id"]: case for case in control_cases}
    candidate_by_id = {case["case_id"]: case for case in candidate_cases}
    if set(control_by_id) != set(candidate_by_id):
        raise RuntimeError("Control/candidate case IDs differ")
    counts: Counter[str] = Counter()
    for case_id, control_case in control_by_id.items():
        candidate_case = candidate_by_id[case_id]
        control_events = {
            int(event["query_position"]): event for event in control_case["events"]
        }
        candidate_events = {
            int(event["query_position"]): event for event in candidate_case["events"]
        }
        if set(control_events) != set(candidate_events):
            raise RuntimeError(f"Query positions differ for case {case_id}")
        valid_targets = {int(event["target"]) for event in control_events.values()}
        for position, control_event in control_events.items():
            candidate_event = candidate_events[position]
            source = _event_category(control_event, valid_targets)
            target = _event_category(candidate_event, valid_targets)
            counts[f"{source}_to_{target}"] += 1
    return dict(sorted(counts.items()))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-epochs", type=int, default=MAX_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    args = parser.parse_args()
    if args.max_epochs != MAX_EPOCHS or args.batch_size != BATCH_SIZE:
        raise ValueError("P-GDN3-048 has one fixed 10-epoch/batch32 endpoint")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    matched_init_path, matched_init_hash = _write_matched_initialization(
        args.output_dir
    )
    control_name = "future_seed_gdn2_control"
    candidate_name = "future_seed_address_payload_gauge_gdn2"
    control = run_arm(
        arm="future_seed_gdn2",
        output_arm_name=control_name,
        sequence_length=SEQUENCE_LENGTH,
        num_kv_pairs=NUM_KV_PAIRS,
        output_dir=args.output_dir,
        max_epochs=MAX_EPOCHS,
        batch_size=BATCH_SIZE,
        save_checkpoint=True,
        matched_init_path=matched_init_path,
    )
    candidate = run_arm(
        arm=candidate_name,
        sequence_length=SEQUENCE_LENGTH,
        num_kv_pairs=NUM_KV_PAIRS,
        output_dir=args.output_dir,
        max_epochs=MAX_EPOCHS,
        batch_size=BATCH_SIZE,
        save_checkpoint=True,
        matched_init_path=matched_init_path,
    )

    diagnostics = candidate["address_payload_gauge"]
    layers = diagnostics["per_layer"]
    integrity_checks = {
        "control_used_matched_initialization": (
            control["init_parameter_hash"] == matched_init_hash
        ),
        "candidate_parent_used_matched_initialization": (
            candidate["parent_init_parameter_hash"] == matched_init_hash
        ),
        "matched_data_hashes": candidate["data_hashes"] == control["data_hashes"],
        "matched_warmup_batch": (
            candidate["warmup_batch_hash"] == control["warmup_batch_hash"]
        ),
        "parameter_counts_exact": (
            control["parameters"] == EXPECTED_NATIVE_PARAMETERS
            and candidate["parameters"] == EXPECTED_CANDIDATE_PARAMETERS
            and candidate["parameters"] - control["parameters"]
            == EXPECTED_NEW_PARAMETERS
        ),
        "main_state_size_unchanged": (
            control["recurrent_state_values_per_layer"]
            == candidate["recurrent_state_values_per_layer"]
            == EXPECTED_STATE_VALUES
        ),
        "single_scan_topology_exact": (
            diagnostics["logical_scans_per_layer"] == 1
            and diagnostics["new_persistent_state_values"] == 0
        ),
    }
    activation_checks = {
        "two_active_layers_and_eight_heads": (
            diagnostics["active_layers"] == 2
            and sum(row["active_heads"] for row in layers) == 8
        ),
        "bounded_reciprocal_factors": all(
            row["factor_min"] >= 0.5 - 2e-3
            and row["factor_max"] <= 2.0 + 2e-3
            and row["same_address_reciprocal_max_error"] <= 2e-3
            and row["production_replay_max_error"] == 0.0
            for row in layers
        ),
        "address_conditioning_is_effective": all(
            row["factor_token_std"] >= 1e-4
            and row["qk_code_mismatch_rms"] >= 1e-4
            and row["coded_value_change_relative_rms"] >= 1e-4
            and row["decoded_output_change_relative_rms"] >= 1e-4
            for row in layers
        ),
        "state_is_finite_and_variable": all(
            math.isfinite(row["terminal_state_rms"])
            and row["terminal_state_rms"] > 0
            and row["terminal_state_board_std"] > 1e-6
            for row in layers
        ),
        "native_futureseed_active": (
            candidate["future_seed"]["active_seed_routes"] == 1
        ),
    }

    control_cases = _cases(args.output_dir, control_name)
    candidate_cases = _cases(args.output_dir, candidate_name)
    control_swaps = wrong_key_swap_summary(control_cases)
    candidate_swaps = wrong_key_swap_summary(candidate_cases)
    transitions = transition_summary(control_cases, candidate_cases)
    cm = control["metrics"]
    pm = candidate["metrics"]
    quality_checks = {
        "balanced_at_least_0.55_and_gain_0.10": (
            pm["balanced_accuracy"] >= 0.55
            and pm["balanced_accuracy"] - cm["balanced_accuracy"] >= 0.10
        ),
        "future_at_least_0.50_and_gain_0.08": (
            pm["future"]["accuracy"] >= 0.50
            and pm["future"]["accuracy"] - cm["future"]["accuracy"] >= 0.08
        ),
        "past_at_least_0.50_and_gain_0.08": (
            pm["past"]["accuracy"] >= 0.50
            and pm["past"]["accuracy"] - cm["past"]["accuracy"] >= 0.08
        ),
        "joint_at_least_0.06_and_gain_0.04": (
            pm["joint_exact"] >= 0.06
            and pm["joint_exact"] - cm["joint_exact"] >= 0.04
        ),
        "total_errors_reduced": candidate_swaps["errors"] < control_swaps["errors"],
        "swap_fraction_reduced_at_least_0.10": (
            control_swaps["wrong_key_swap_fraction_of_errors"]
            - candidate_swaps["wrong_key_swap_fraction_of_errors"]
            >= 0.10
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
        "elapsed_below_1.35x": cost_ratios["elapsed"] < 1.35,
        "post_warm_wall_below_1.35x": cost_ratios["post_warm_wall"] < 1.35,
        "warmed_step_below_1.35x": cost_ratios["warmed_step"] < 1.35,
        "peak_allocation_below_1.12x": cost_ratios["peak_allocation"] < 1.12,
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
            "plan": "P-GDN3-048",
            "mechanism": "reciprocal address-conditioned diagonal payload gauge",
            "run_order": [control_name, candidate_name],
            "model": "D128/L2/H4/K32/V32 pinned-official GDN2 plus native FutureSeed",
            "sequence_length": SEQUENCE_LENGTH,
            "num_kv_pairs": NUM_KV_PAIRS,
            "epochs": MAX_EPOCHS,
            "batch_size": BATCH_SIZE,
            "seed": SEED,
            "new_parameters": EXPECTED_NEW_PARAMETERS,
            "new_persistent_state_per_layer": 0,
            "new_scans_per_layer": 0,
            "sweep": None,
        },
        "matched_initialization": {
            "path": str(matched_init_path),
            "parameter_hash": matched_init_hash,
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
        "decision": "admit_one_sudoku_scale_transfer" if passed else "closed",
    }
    (args.output_dir / "comparison.json").write_text(
        json.dumps(comparison, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(comparison, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
