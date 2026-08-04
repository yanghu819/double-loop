from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

from experiments.zoology_mqar.length_scaling import run_arm


ARMS = ("causal_gdn2", "future_seed_gdn2")
SEQUENCE_LENGTH = 1024
NUM_KV_PAIRS = 4
MAX_EPOCHS = 10
BATCH_SIZE = 32
EXPECTED_REFERENCE_SHA = "4b909629ee3cc32ab1c57e9d3ac1f26ecaa8b273"


def two_arm_summary(scores: dict[str, dict[str, Any]]) -> dict[str, Any]:
    causal = scores["causal_gdn2"]
    future_seed = scores["future_seed_gdn2"]
    if causal["init_hash"] != future_seed["init_hash"]:
        raise RuntimeError("Matched GDN2 arms did not start identically")
    if causal["init_parameter_hash"] != future_seed["init_parameter_hash"]:
        raise RuntimeError("Matched GDN2 parameter hashes differ")
    if causal["data_hashes"] != future_seed["data_hashes"]:
        raise RuntimeError("Matched GDN2 arms did not use identical data")
    return {
        "arms": scores,
        "future_seed_vs_causal": {
            "future_accuracy_delta": (
                future_seed["metrics"]["future"]["accuracy"]
                - causal["metrics"]["future"]["accuracy"]
            ),
            "past_accuracy_delta": (
                future_seed["metrics"]["past"]["accuracy"]
                - causal["metrics"]["past"]["accuracy"]
            ),
            "joint_exact_delta": (
                future_seed["metrics"]["joint_exact"]
                - causal["metrics"]["joint_exact"]
            ),
        },
    }


def copy_length64_reference(reference_run: Path, output_dir: Path) -> dict[str, Any]:
    if (reference_run / "git_sha.txt").read_text().strip() != EXPECTED_REFERENCE_SHA:
        raise RuntimeError("Frozen L64 reference SHA changed")
    reference_score = json.loads((reference_run / "score.json").read_text())
    summary = reference_score["lengths"]["64"]
    frozen_scores = {arm: summary["arms"][arm] for arm in ARMS}
    destination = output_dir / "length_64"
    destination.mkdir(parents=True, exist_ok=True)
    for arm in ARMS:
        shutil.copytree(
            reference_run / "output" / "length_64" / arm,
            destination / arm,
            dirs_exist_ok=False,
        )
    frozen_summary = two_arm_summary(frozen_scores)
    (destination / "comparison.json").write_text(
        json.dumps(frozen_summary, indent=2, sort_keys=True) + "\n"
    )
    return frozen_summary


def endpoint_gate(lengths: dict[str, Any]) -> dict[str, Any]:
    length64 = lengths["64"]["arms"]
    length1024 = lengths["1024"]["arms"]
    causal = length1024["causal_gdn2"]["metrics"]
    future_seed = length1024["future_seed_gdn2"]["metrics"]
    delta = future_seed["future"]["accuracy"] - causal["future"]["accuracy"]
    retention = (
        future_seed["future"]["accuracy"]
        / length64["future_seed_gdn2"]["metrics"]["future"]["accuracy"]
    )
    checks = {
        "causal_past_at_least_0.90": causal["past"]["accuracy"] >= 0.90,
        "causal_future_at_most_0.10": causal["future"]["accuracy"] <= 0.10,
        "future_seed_past_at_least_0.90": future_seed["past"]["accuracy"] >= 0.90,
        "future_seed_future_at_least_0.80": (
            future_seed["future"]["accuracy"] >= 0.80
        ),
        "future_seed_delta_at_least_0.70": delta >= 0.70,
        "future_seed_retention_at_least_0.80": retention >= 0.80,
    }
    partial_signal = (
        future_seed["past"]["accuracy"] >= 0.90
        and future_seed["future"]["accuracy"] >= 0.50
        and delta >= 0.40
    )
    return {
        "passed": all(checks.values()),
        "partial_signal": partial_signal,
        "checks": checks,
        "length1024_future_accuracy_delta": delta,
        "future_seed_future_accuracy_retention_64_to_1024": retention,
    }


def systems_scaling(lengths: dict[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for arm in ARMS:
        short = lengths["64"]["arms"][arm]["warmed_step_benchmark"]
        long = lengths["1024"]["arms"][arm]["warmed_step_benchmark"]
        result[arm] = {
            "tokens_per_sec_length64": short["tokens_per_sec"],
            "tokens_per_sec_length1024": long["tokens_per_sec"],
            "throughput_ratio_1024_over_64": (
                long["tokens_per_sec"] / short["tokens_per_sec"]
            ),
            "peak_memory_length64_bytes": short["peak_cuda_mem_bytes"],
            "peak_memory_length1024_bytes": long["peak_cuda_mem_bytes"],
            "peak_memory_ratio_1024_over_64": (
                long["peak_cuda_mem_bytes"] / short["peak_cuda_mem_bytes"]
            ),
        }
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--length64-reference-run", type=Path, required=True)
    parser.add_argument("--max-epochs", type=int, default=MAX_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    length64 = copy_length64_reference(
        args.length64_reference_run.resolve(),
        args.output_dir,
    )
    scores1024 = {
        arm: run_arm(
            arm=arm,
            sequence_length=SEQUENCE_LENGTH,
            num_kv_pairs=NUM_KV_PAIRS,
            output_dir=args.output_dir,
            max_epochs=args.max_epochs,
            batch_size=args.batch_size,
        )
        for arm in ARMS
    }
    length1024 = two_arm_summary(scores1024)
    (args.output_dir / "length_1024" / "comparison.json").write_text(
        json.dumps(length1024, indent=2, sort_keys=True) + "\n"
    )
    lengths = {"64": length64, "1024": length1024}
    gate = endpoint_gate(lengths)
    comparison = {
        "status": "complete",
        "protocol": {
            "sequence_lengths": [64, 1024],
            "num_kv_pairs": NUM_KV_PAIRS,
            "train_examples_per_new_arm": 10_000,
            "validation_examples_per_arm": 1_000,
            "epochs": args.max_epochs,
            "batch_size": args.batch_size,
            "seed": 123,
            "model": "official-FLA GDN2 D128/L2/H4/D32",
            "future_seed": "native cross-layer terminal-state seeding",
            "length64_source_sha": EXPECTED_REFERENCE_SHA,
            "attention_quality_ceiling": "not included; P-CAUSAL-011 carrier invalid",
        },
        "lengths": lengths,
        "registered_endpoint_gate": gate,
        "systems_scaling": systems_scaling(lengths),
    }
    (args.output_dir / "comparison.json").write_text(
        json.dumps(comparison, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(comparison, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
