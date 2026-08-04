from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

from experiments.zoology_mqar.length_scaling import run_arm


SEQUENCE_LENGTH = 1024
NUM_KV_PAIRS = 4
MAX_EPOCHS = 10
BATCH_SIZE = 32
CANDIDATE_WIDTH = 256
CANDIDATE_HEADS = 8
CANDIDATE_HEAD_DIM = 32
EXPECTED_REFERENCE_SHA = "77e5539fc0ef74231cab658bd610b24254fdcfa7"
ARMS = ("causal_gdn2_d256", "future_seed_gdn2_d256")


def load_reference(reference_run: Path, output_dir: Path) -> dict[str, Any]:
    if (reference_run / "git_sha.txt").read_text().strip() != EXPECTED_REFERENCE_SHA:
        raise RuntimeError("Frozen D128 reference SHA changed")
    reference = json.loads((reference_run / "score.json").read_text())
    reference_length = reference["lengths"]["1024"]
    destination = output_dir / "reference_d128"
    destination.mkdir(parents=True, exist_ok=True)
    for arm in ("causal_gdn2", "future_seed_gdn2"):
        shutil.copytree(
            reference_run / "output" / "length_1024" / arm,
            destination / arm,
            dirs_exist_ok=False,
        )
    return reference_length


def binding_diagnostics(cases: list[dict[str, Any]]) -> dict[str, Any]:
    by_direction = {}
    for direction in ("past", "future"):
        errors = 0
        wrong_value_from_same_case = 0
        queries = 0
        for case in cases:
            targets = {event["target"] for event in case["events"]}
            for event in case["events"]:
                if event["direction"] != direction:
                    continue
                queries += 1
                if event["correct"]:
                    continue
                errors += 1
                if event["prediction"] in targets:
                    wrong_value_from_same_case += 1
        by_direction[direction] = {
            "queries": queries,
            "errors": errors,
            "wrong_value_from_same_case": wrong_value_from_same_case,
            "wrong_value_from_same_case_fraction_of_errors": (
                wrong_value_from_same_case / errors if errors else 0.0
            ),
            "wrong_value_from_same_case_fraction_of_queries": (
                wrong_value_from_same_case / queries if queries else 0.0
            ),
        }
    return by_direction


def load_cases(path: Path) -> list[dict[str, Any]]:
    return json.loads(path.read_text())


def summarize(
    reference: dict[str, Any],
    candidate_scores: dict[str, dict[str, Any]],
    output_dir: Path,
) -> dict[str, Any]:
    causal = candidate_scores["causal_gdn2_d256"]
    future_seed = candidate_scores["future_seed_gdn2_d256"]
    if causal["init_hash"] != future_seed["init_hash"]:
        raise RuntimeError("D256 matched arms did not start identically")
    if causal["init_parameter_hash"] != future_seed["init_parameter_hash"]:
        raise RuntimeError("D256 matched parameter hashes differ")
    if causal["data_hashes"] != future_seed["data_hashes"]:
        raise RuntimeError("D256 matched arms used different data")
    if causal["data_hashes"] != reference["arms"]["causal_gdn2"]["data_hashes"]:
        raise RuntimeError("D256 data drifted from the frozen D128 endpoint")

    diagnostics = {
        "reference_d128_future_seed": binding_diagnostics(
            load_cases(
                output_dir
                / "reference_d128"
                / "future_seed_gdn2"
                / "cases.json"
            )
        ),
        "candidate_d256_causal": binding_diagnostics(
            load_cases(
                output_dir
                / "length_1024"
                / "causal_gdn2_d256"
                / "cases.json"
            )
        ),
        "candidate_d256_future_seed": binding_diagnostics(
            load_cases(
                output_dir
                / "length_1024"
                / "future_seed_gdn2_d256"
                / "cases.json"
            )
        ),
    }
    base_metrics = reference["arms"]["future_seed_gdn2"]["metrics"]
    candidate_metrics = future_seed["metrics"]
    base_balanced = base_metrics["balanced_accuracy"]
    candidate_balanced = candidate_metrics["balanced_accuracy"]
    base_binding = diagnostics["reference_d128_future_seed"]
    candidate_binding = diagnostics["candidate_d256_future_seed"]
    future_binding_reduction = (
        base_binding["future"]["wrong_value_from_same_case_fraction_of_queries"]
        - candidate_binding["future"][
            "wrong_value_from_same_case_fraction_of_queries"
        ]
    )
    past_binding_reduction = (
        base_binding["past"]["wrong_value_from_same_case_fraction_of_queries"]
        - candidate_binding["past"][
            "wrong_value_from_same_case_fraction_of_queries"
        ]
    )
    checks = {
        "candidate_causal_past_at_least_0.80": (
            causal["metrics"]["past"]["accuracy"] >= 0.80
        ),
        "candidate_causal_future_at_most_0.10": (
            causal["metrics"]["future"]["accuracy"] <= 0.10
        ),
        "candidate_future_seed_past_at_least_0.90": (
            candidate_metrics["past"]["accuracy"] >= 0.90
        ),
        "candidate_future_seed_future_at_least_0.90": (
            candidate_metrics["future"]["accuracy"] >= 0.90
        ),
        "candidate_future_seed_joint_exact_at_least_0.70": (
            candidate_metrics["joint_exact"] >= 0.70
        ),
        "balanced_accuracy_gain_at_least_0.10": (
            candidate_balanced - base_balanced >= 0.10
        ),
        "future_binding_error_rate_reduction_at_least_0.08": (
            future_binding_reduction >= 0.08
        ),
        "past_binding_error_rate_reduction_at_least_0.07": (
            past_binding_reduction >= 0.07
        ),
    }
    partial_signal = (
        candidate_balanced - base_balanced >= 0.05
        and future_binding_reduction >= 0.04
        and past_binding_reduction >= 0.04
    )
    baseline_state = 4 * 32 * 32
    candidate_state = CANDIDATE_HEADS * CANDIDATE_HEAD_DIM * CANDIDATE_HEAD_DIM
    return {
        "status": "complete",
        "protocol": {
            "plan": "P-CAUSAL-013",
            "sequence_length": SEQUENCE_LENGTH,
            "num_kv_pairs": NUM_KV_PAIRS,
            "train_examples_per_candidate_arm": 10_000,
            "validation_examples_per_arm": 1_000,
            "epochs": MAX_EPOCHS,
            "batch_size": BATCH_SIZE,
            "seed": 123,
            "reference_model": "official-FLA GDN2 D128/L2/H4/D32",
            "candidate_model": "official-FLA GDN2 D256/L2/H8/D32",
            "future_seed": "native cross-layer terminal-state seeding",
        },
        "reference_d128": reference,
        "candidate_d256": {
            "arms": candidate_scores,
            "future_seed_vs_causal": {
                "past_accuracy_delta": (
                    future_seed["metrics"]["past"]["accuracy"]
                    - causal["metrics"]["past"]["accuracy"]
                ),
                "future_accuracy_delta": (
                    future_seed["metrics"]["future"]["accuracy"]
                    - causal["metrics"]["future"]["accuracy"]
                ),
                "joint_exact_delta": (
                    future_seed["metrics"]["joint_exact"]
                    - causal["metrics"]["joint_exact"]
                ),
            },
        },
        "capacity_scaling": {
            "reference_state_values_per_layer": baseline_state,
            "candidate_state_values_per_layer": candidate_state,
            "state_value_ratio": candidate_state / baseline_state,
            "reference_parameters": reference["arms"]["future_seed_gdn2"][
                "parameters"
            ],
            "candidate_parameters": future_seed["parameters"],
            "parameter_ratio": (
                future_seed["parameters"]
                / reference["arms"]["future_seed_gdn2"]["parameters"]
            ),
        },
        "binding_diagnostics": diagnostics,
        "registered_gate": {
            "passed": all(checks.values()),
            "partial_signal": partial_signal,
            "checks": checks,
            "balanced_accuracy_gain": candidate_balanced - base_balanced,
            "future_binding_error_rate_reduction": future_binding_reduction,
            "past_binding_error_rate_reduction": past_binding_reduction,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--reference-run", type=Path, required=True)
    parser.add_argument("--max-epochs", type=int, default=MAX_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    args = parser.parse_args()
    if args.max_epochs != MAX_EPOCHS or args.batch_size != BATCH_SIZE:
        raise ValueError("P-CAUSAL-013 uses the preregistered fixed budget")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    reference = load_reference(args.reference_run.resolve(), args.output_dir)
    candidate_scores = {
        "causal_gdn2_d256": run_arm(
            arm="causal_gdn2",
            output_arm_name="causal_gdn2_d256",
            sequence_length=SEQUENCE_LENGTH,
            num_kv_pairs=NUM_KV_PAIRS,
            output_dir=args.output_dir,
            max_epochs=MAX_EPOCHS,
            batch_size=BATCH_SIZE,
            model_width=CANDIDATE_WIDTH,
            model_heads=CANDIDATE_HEADS,
            gdn2_head_dim=CANDIDATE_HEAD_DIM,
        ),
        "future_seed_gdn2_d256": run_arm(
            arm="future_seed_gdn2",
            output_arm_name="future_seed_gdn2_d256",
            sequence_length=SEQUENCE_LENGTH,
            num_kv_pairs=NUM_KV_PAIRS,
            output_dir=args.output_dir,
            max_epochs=MAX_EPOCHS,
            batch_size=BATCH_SIZE,
            model_width=CANDIDATE_WIDTH,
            model_heads=CANDIDATE_HEADS,
            gdn2_head_dim=CANDIDATE_HEAD_DIM,
        ),
    }
    comparison = summarize(reference, candidate_scores, args.output_dir)
    (args.output_dir / "comparison.json").write_text(
        json.dumps(comparison, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(comparison, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
