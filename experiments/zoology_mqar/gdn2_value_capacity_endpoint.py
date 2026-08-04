from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from experiments.zoology_mqar.gdn2_state_capacity_endpoint import (
    binding_diagnostics,
    load_cases,
    load_reference,
)
from experiments.zoology_mqar.length_scaling import run_arm


SEQUENCE_LENGTH = 1024
NUM_KV_PAIRS = 4
MAX_EPOCHS = 10
BATCH_SIZE = 32
MODEL_WIDTH = 128
MODEL_HEADS = 4
HEAD_DIM = 32
EXPAND_V = 2.0


def first_opening_epoch(score: dict[str, Any], threshold: float) -> int | None:
    for row in score["valid_curve"]:
        if float(row["valid/accuracy"]) >= threshold:
            return int(row["epoch"])
    return None


def summarize(
    reference: dict[str, Any],
    candidate_scores: dict[str, dict[str, Any]],
    output_dir: Path,
) -> dict[str, Any]:
    causal = candidate_scores["causal_gdn2_expandv2"]
    future_seed = candidate_scores["future_seed_gdn2_expandv2"]
    if causal["init_hash"] != future_seed["init_hash"]:
        raise RuntimeError("expand-v2 matched arms did not start identically")
    if causal["init_parameter_hash"] != future_seed["init_parameter_hash"]:
        raise RuntimeError("expand-v2 matched parameter hashes differ")
    if causal["data_hashes"] != future_seed["data_hashes"]:
        raise RuntimeError("expand-v2 matched arms used different data")
    if causal["data_hashes"] != reference["arms"]["causal_gdn2"]["data_hashes"]:
        raise RuntimeError("expand-v2 data drifted from the frozen D128 endpoint")

    diagnostics = {
        "reference_d128_future_seed": binding_diagnostics(
            load_cases(
                output_dir
                / "reference_d128"
                / "future_seed_gdn2"
                / "cases.json"
            )
        ),
        "candidate_expandv2_causal": binding_diagnostics(
            load_cases(
                output_dir
                / "length_1024"
                / "causal_gdn2_expandv2"
                / "cases.json"
            )
        ),
        "candidate_expandv2_future_seed": binding_diagnostics(
            load_cases(
                output_dir
                / "length_1024"
                / "future_seed_gdn2_expandv2"
                / "cases.json"
            )
        ),
    }
    base = reference["arms"]["future_seed_gdn2"]
    base_metrics = base["metrics"]
    candidate_metrics = future_seed["metrics"]
    balanced_gain = (
        candidate_metrics["balanced_accuracy"] - base_metrics["balanced_accuracy"]
    )
    base_binding = diagnostics["reference_d128_future_seed"]
    candidate_binding = diagnostics["candidate_expandv2_future_seed"]
    binding_reduction = {
        direction: (
            base_binding[direction]["wrong_value_from_same_case_fraction_of_queries"]
            - candidate_binding[direction][
                "wrong_value_from_same_case_fraction_of_queries"
            ]
        )
        for direction in ("past", "future")
    }
    quality_comparable = (
        candidate_metrics["balanced_accuracy"] >= base_metrics["balanced_accuracy"]
    )
    checks = {
        "candidate_causal_future_at_most_0.10": (
            causal["metrics"]["future"]["accuracy"] <= 0.10
        ),
        "candidate_future_seed_past_at_least_0.85": (
            candidate_metrics["past"]["accuracy"] >= 0.85
        ),
        "candidate_future_seed_future_at_least_0.85": (
            candidate_metrics["future"]["accuracy"] >= 0.85
        ),
        "candidate_future_seed_joint_exact_at_least_0.60": (
            candidate_metrics["joint_exact"] >= 0.60
        ),
        "balanced_accuracy_gain_at_least_0.10": balanced_gain >= 0.10,
        "future_binding_reduction_at_least_0.08": (
            quality_comparable and binding_reduction["future"] >= 0.08
        ),
        "past_binding_reduction_at_least_0.07": (
            quality_comparable and binding_reduction["past"] >= 0.07
        ),
    }
    paper_checks = {
        "past_at_least_0.90": candidate_metrics["past"]["accuracy"] >= 0.90,
        "future_at_least_0.90": candidate_metrics["future"]["accuracy"] >= 0.90,
        "joint_exact_at_least_0.70": candidate_metrics["joint_exact"] >= 0.70,
    }
    partial_signal = (
        quality_comparable
        and balanced_gain >= 0.05
        and binding_reduction["future"] >= 0.04
        and binding_reduction["past"] >= 0.04
    )
    candidate_state = int(future_seed["recurrent_state_values_per_layer"])
    reference_state = MODEL_HEADS * HEAD_DIM * HEAD_DIM
    return {
        "status": "complete",
        "protocol": {
            "plan": "P-CAUSAL-014",
            "sequence_length": SEQUENCE_LENGTH,
            "num_kv_pairs": NUM_KV_PAIRS,
            "train_examples_per_candidate_arm": 10_000,
            "validation_examples_per_arm": 1_000,
            "epochs": MAX_EPOCHS,
            "batch_size": BATCH_SIZE,
            "seed": 123,
            "reference_model": "official-FLA GDN2 D128/L2/H4/D32 expand-v1",
            "candidate_model": "official-FLA GDN2 D128/L2/H4/D32 expand-v2",
            "future_seed": "native cross-layer terminal-state seeding",
        },
        "reference_d128_expandv1": reference,
        "candidate_d128_expandv2": {
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
            "reference_state_values_per_layer": reference_state,
            "candidate_state_values_per_layer": candidate_state,
            "state_value_ratio": candidate_state / reference_state,
            "reference_parameters": base["parameters"],
            "candidate_parameters": future_seed["parameters"],
            "parameter_ratio": future_seed["parameters"] / base["parameters"],
            "model_width_ratio": 1.0,
        },
        "opening_epochs": {
            "reference_accuracy_0.05": first_opening_epoch(base, 0.05),
            "reference_accuracy_0.50": first_opening_epoch(base, 0.50),
            "candidate_accuracy_0.05": first_opening_epoch(future_seed, 0.05),
            "candidate_accuracy_0.50": first_opening_epoch(future_seed, 0.50),
        },
        "binding_diagnostics": diagnostics,
        "registered_gate": {
            "passed": all(checks.values()),
            "paper_candidate": all(paper_checks.values()) and all(checks.values()),
            "partial_signal": partial_signal,
            "checks": checks,
            "paper_checks": paper_checks,
            "quality_comparable": quality_comparable,
            "balanced_accuracy_gain": balanced_gain,
            "future_binding_error_rate_reduction": binding_reduction["future"],
            "past_binding_error_rate_reduction": binding_reduction["past"],
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
        raise ValueError("P-CAUSAL-014 uses the preregistered fixed budget")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    reference = load_reference(args.reference_run.resolve(), args.output_dir)
    candidate_scores = {
        "causal_gdn2_expandv2": run_arm(
            arm="causal_gdn2",
            output_arm_name="causal_gdn2_expandv2",
            sequence_length=SEQUENCE_LENGTH,
            num_kv_pairs=NUM_KV_PAIRS,
            output_dir=args.output_dir,
            max_epochs=MAX_EPOCHS,
            batch_size=BATCH_SIZE,
            model_width=MODEL_WIDTH,
            model_heads=MODEL_HEADS,
            gdn2_head_dim=HEAD_DIM,
            gdn2_expand_v=EXPAND_V,
        ),
        "future_seed_gdn2_expandv2": run_arm(
            arm="future_seed_gdn2",
            output_arm_name="future_seed_gdn2_expandv2",
            sequence_length=SEQUENCE_LENGTH,
            num_kv_pairs=NUM_KV_PAIRS,
            output_dir=args.output_dir,
            max_epochs=MAX_EPOCHS,
            batch_size=BATCH_SIZE,
            model_width=MODEL_WIDTH,
            model_heads=MODEL_HEADS,
            gdn2_head_dim=HEAD_DIM,
            gdn2_expand_v=EXPAND_V,
        ),
    }
    comparison = summarize(reference, candidate_scores, args.output_dir)
    (args.output_dir / "comparison.json").write_text(
        json.dumps(comparison, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(comparison, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
