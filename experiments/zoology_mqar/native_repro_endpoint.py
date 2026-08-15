from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
from zoology.utils import set_determinism

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
EXPECTED_PARAMETERS = 661_584


def _write_initialization(output_dir: Path) -> tuple[Path, str]:
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


def _cases(root: Path, arm: str) -> list[dict[str, Any]]:
    return json.loads((root / "length_1024" / arm / "cases.json").read_text())


def _prediction_agreement(
    first: list[dict[str, Any]], second: list[dict[str, Any]]
) -> dict[str, float | int]:
    first_by_id = {case["case_id"]: case for case in first}
    second_by_id = {case["case_id"]: case for case in second}
    if set(first_by_id) != set(second_by_id):
        raise RuntimeError("Replay case IDs differ")
    matched = 0
    total = 0
    for case_id, first_case in first_by_id.items():
        second_case = second_by_id[case_id]
        first_events = {
            int(event["query_position"]): event for event in first_case["events"]
        }
        second_events = {
            int(event["query_position"]): event for event in second_case["events"]
        }
        if set(first_events) != set(second_events):
            raise RuntimeError(f"Replay query positions differ for {case_id}")
        for position, first_event in first_events.items():
            matched += int(
                int(first_event["prediction"])
                == int(second_events[position]["prediction"])
            )
            total += 1
    return {
        "matched_predictions": matched,
        "queries": total,
        "fraction": matched / total,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-epochs", type=int, default=MAX_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    args = parser.parse_args()
    if args.max_epochs != MAX_EPOCHS or args.batch_size != BATCH_SIZE:
        raise ValueError("P-REPRO-001 fixes one 10-epoch/batch32 protocol")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    init_path, init_hash = _write_initialization(args.output_dir)
    arms = ("native_replay_a", "native_replay_b")
    results = []
    for name in arms:
        results.append(
            run_arm(
                arm="future_seed_gdn2",
                output_arm_name=name,
                sequence_length=SEQUENCE_LENGTH,
                num_kv_pairs=NUM_KV_PAIRS,
                output_dir=args.output_dir,
                max_epochs=MAX_EPOCHS,
                batch_size=BATCH_SIZE,
                save_checkpoint=True,
                matched_init_path=init_path,
            )
        )

    first, second = results
    first_cases = _cases(args.output_dir, arms[0])
    second_cases = _cases(args.output_dir, arms[1])
    first_swaps = wrong_key_swap_summary(first_cases)
    second_swaps = wrong_key_swap_summary(second_cases)
    agreement = _prediction_agreement(first_cases, second_cases)
    fm = first["metrics"]
    sm = second["metrics"]

    integrity_checks = {
        "same_serialized_initialization": (
            first["init_parameter_hash"]
            == second["init_parameter_hash"]
            == init_hash
        ),
        "same_data_hashes": first["data_hashes"] == second["data_hashes"],
        "same_warmup_batch": (
            first["warmup_batch_hash"] == second["warmup_batch_hash"]
        ),
        "exact_parameter_count": (
            first["parameters"] == second["parameters"] == EXPECTED_PARAMETERS
        ),
        "native_futureseed_active": (
            first["future_seed"]["active_seed_routes"]
            == second["future_seed"]["active_seed_routes"]
            == 1
        ),
    }
    reproducibility_checks = {
        "balanced_delta_at_most_0.01": (
            abs(fm["balanced_accuracy"] - sm["balanced_accuracy"]) <= 0.01
        ),
        "direction_delta_at_most_0.015": (
            abs(fm["future"]["accuracy"] - sm["future"]["accuracy"]) <= 0.015
            and abs(fm["past"]["accuracy"] - sm["past"]["accuracy"]) <= 0.015
        ),
        "joint_delta_at_most_0.01": abs(
            fm["joint_exact"] - sm["joint_exact"]
        ) <= 0.01,
        "error_delta_at_most_40": (
            abs(first_swaps["errors"] - second_swaps["errors"]) <= 40
        ),
        "prediction_agreement_at_least_0.90": agreement["fraction"] >= 0.90,
    }
    cost_ratio = (
        second["post_warm_arm_wall_sec_through_checkpoint"]
        / first["post_warm_arm_wall_sec_through_checkpoint"]
    )
    cost_checks = {"second_replay_wall_below_1.25x": cost_ratio < 1.25}
    passed = (
        all(integrity_checks.values())
        and all(reproducibility_checks.values())
        and all(cost_checks.values())
    )
    comparison = {
        "status": "complete",
        "decision": "reproducible" if passed else "training_bifurcation",
        "protocol": {
            "plan": "P-REPRO-001",
            "mechanism_delta": None,
            "seed_sweep": False,
            "seed": SEED,
            "sequence_length": SEQUENCE_LENGTH,
            "num_kv_pairs": NUM_KV_PAIRS,
            "epochs": MAX_EPOCHS,
            "batch_size": BATCH_SIZE,
            "run_order": list(arms),
        },
        "replay_a": first,
        "replay_b": second,
        "replay_a_swap_summary": first_swaps,
        "replay_b_swap_summary": second_swaps,
        "prediction_agreement": agreement,
        "trained_parameter_hash_exact_match": (
            first["trained_parameter_hash"] == second["trained_parameter_hash"]
        ),
        "post_warm_wall_ratio": cost_ratio,
        "registered_gate": {
            "integrity_checks": integrity_checks,
            "reproducibility_checks": reproducibility_checks,
            "cost_checks": cost_checks,
            "passed": passed,
        },
    }
    (args.output_dir / "comparison.json").write_text(
        json.dumps(comparison, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(comparison, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
