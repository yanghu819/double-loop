from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

from zoology.data.utils import prepare_data
from zoology.utils import set_determinism

from experiments.zoology_mqar.length_scaling import (
    benchmark_training_step,
    build_config,
    make_model,
    run_arm,
    warm_model,
)


ARMS = ("causal_gdn2", "future_seed_gdn2")
ALL_LENGTHS = (64, 128, 256, 512, 1024)
TRAIN_LENGTHS = (128, 256, 512)
NUM_KV_PAIRS = 4
MAX_EPOCHS = 10
BATCH_SIZE = 32
REFERENCE_SPECS = {
    64: {
        "git_sha": "4b909629ee3cc32ab1c57e9d3ac1f26ecaa8b273",
        "score_sha256": "d6b54a89738bbbd3a2c9e27e40c50ff9ff86c6845455034e8bd773c795c083e6",
    },
    1024: {
        "git_sha": "77e5539fc0ef74231cab658bd610b24254fdcfa7",
        "score_sha256": "5d65ebe522bb681cf819953255c2eb7c8e12c2f4335479ac66e6511f7db95583",
    },
}
FUTURE_ACCURACY_FLOORS = {64: 0.90, 128: 0.95, 256: 0.90, 512: 0.80, 1024: 0.70}


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


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


def copy_reference(
    *,
    reference_run: Path,
    output_dir: Path,
    sequence_length: int,
) -> dict[str, Any]:
    spec = REFERENCE_SPECS[sequence_length]
    actual_git_sha = (reference_run / "git_sha.txt").read_text().strip()
    actual_score_sha = file_sha256(reference_run / "score.json")
    if actual_git_sha != spec["git_sha"]:
        raise RuntimeError(
            f"Frozen L{sequence_length} reference source changed: {actual_git_sha}"
        )
    if actual_score_sha != spec["score_sha256"]:
        raise RuntimeError(
            f"Frozen L{sequence_length} reference score changed: {actual_score_sha}"
        )
    reference_score = json.loads((reference_run / "score.json").read_text())
    source_summary = reference_score["lengths"][str(sequence_length)]
    scores = {arm: source_summary["arms"][arm] for arm in ARMS}
    destination = output_dir / f"length_{sequence_length}"
    destination.mkdir(parents=True, exist_ok=True)
    for arm in ARMS:
        shutil.copytree(
            reference_run / "output" / f"length_{sequence_length}" / arm,
            destination / arm,
            dirs_exist_ok=False,
        )
    summary = two_arm_summary(scores)
    (destination / "comparison.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    return summary


def load_trained_length(output_dir: Path, sequence_length: int) -> dict[str, Any]:
    scores = {
        arm: json.loads(
            (
                output_dir
                / f"length_{sequence_length}"
                / arm
                / "score.json"
            ).read_text()
        )
        for arm in ARMS
    }
    summary = two_arm_summary(scores)
    (output_dir / f"length_{sequence_length}" / "comparison.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    return summary


def binding_diagnostics(output_dir: Path, sequence_length: int) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for arm in ARMS:
        cases = json.loads(
            (
                output_dir
                / f"length_{sequence_length}"
                / arm
                / "cases.json"
            ).read_text()
        )
        by_direction: dict[str, Any] = {}
        for direction in ("past", "future"):
            errors = 0
            other_sample_value = 0
            for case in cases:
                targets = {int(event["target"]) for event in case["events"]}
                for event in case["events"]:
                    if event["direction"] != direction or event["correct"]:
                        continue
                    errors += 1
                    if int(event["prediction"]) in targets:
                        other_sample_value += 1
            by_direction[direction] = {
                "errors": errors,
                "other_sample_value": other_sample_value,
                "other_sample_value_fraction": (
                    other_sample_value / errors if errors else 0.0
                ),
            }
        result[arm] = by_direction
    return result


def registered_gate(lengths: dict[str, Any]) -> dict[str, Any]:
    checks: dict[str, bool] = {}
    per_length: dict[str, Any] = {}
    for sequence_length in ALL_LENGTHS:
        summary = lengths[str(sequence_length)]
        causal = summary["arms"]["causal_gdn2"]["metrics"]
        future_seed = summary["arms"]["future_seed_gdn2"]["metrics"]
        delta = future_seed["future"]["accuracy"] - causal["future"]["accuracy"]
        floor = FUTURE_ACCURACY_FLOORS[sequence_length]
        length_checks = {
            "causal_future_at_most_0.10": causal["future"]["accuracy"] <= 0.10,
            "future_seed_future_at_least_floor": (
                future_seed["future"]["accuracy"] >= floor
            ),
            "future_seed_future_delta_at_least_0.70": delta >= 0.70,
        }
        per_length[str(sequence_length)] = {
            "future_accuracy_floor": floor,
            "future_accuracy_delta": delta,
            "checks": length_checks,
        }
        for name, passed in length_checks.items():
            checks[f"length{sequence_length}_{name}"] = passed
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "per_length": per_length,
        "kill_if_length128_future_below_0.80": (
            lengths["128"]["arms"]["future_seed_gdn2"]["metrics"]["future"]["accuracy"]
            < 0.80
        ),
    }


def curve_diagnostics(lengths: dict[str, Any]) -> dict[str, Any]:
    fs_future = [
        lengths[str(length)]["arms"]["future_seed_gdn2"]["metrics"]["future"]["accuracy"]
        for length in ALL_LENGTHS
    ]
    drops = [
        {
            "from_length": left,
            "to_length": right,
            "accuracy_drop": fs_future[index] - fs_future[index + 1],
        }
        for index, (left, right) in enumerate(zip(ALL_LENGTHS, ALL_LENGTHS[1:]))
    ]
    largest = max(drops, key=lambda row: row["accuracy_drop"])
    return {
        "future_seed_future_accuracy": {
            str(length): value for length, value in zip(ALL_LENGTHS, fs_future)
        },
        "adjacent_log2_length_drops": drops,
        "largest_adjacent_drop": largest,
        "retention_64_to_1024": fs_future[-1] / fs_future[0],
        "monotonic_nonincreasing": all(
            left >= right for left, right in zip(fs_future, fs_future[1:])
        ),
    }


def systems_metrics(lengths: dict[str, Any]) -> dict[str, Any]:
    return {
        str(length): {
            arm: {
                "parameters": lengths[str(length)]["arms"][arm]["parameters"],
                "train_tokens": lengths[str(length)]["arms"][arm]["train_tokens"],
                "elapsed_sec_including_validation": lengths[str(length)]["arms"][arm][
                    "elapsed_sec_including_validation"
                ],
                "tokens_per_sec": lengths[str(length)]["arms"][arm][
                    "fresh_process_benchmark"
                ]["tokens_per_sec"],
                "peak_cuda_mem_bytes": lengths[str(length)]["arms"][arm][
                    "fresh_process_benchmark"
                ]["peak_cuda_mem_bytes"],
            }
            for arm in ARMS
        }
        for length in ALL_LENGTHS
    }


def run_single_arm(args: argparse.Namespace) -> None:
    score = run_arm(
        arm=args.arm,
        sequence_length=args.sequence_length,
        num_kv_pairs=NUM_KV_PAIRS,
        output_dir=args.output_dir,
        max_epochs=args.max_epochs,
        batch_size=args.batch_size,
    )
    print(json.dumps(score, indent=2, sort_keys=True))


def benchmark_single_arm(args: argparse.Namespace) -> None:
    config = build_config(
        arm=args.arm,
        sequence_length=args.sequence_length,
        num_kv_pairs=NUM_KV_PAIRS,
        max_epochs=MAX_EPOCHS,
        batch_size=BATCH_SIZE,
    )
    set_determinism(config.seed)
    model = make_model(config, args.arm)
    train_loader, _ = prepare_data(config.data)
    fixed_batch = next(iter(train_loader))
    warm_model(model, fixed_batch)
    benchmark = benchmark_training_step(
        model,
        fixed_batch,
        warmup_steps=args.warmup_steps,
        measured_steps=args.measured_steps,
    )
    result = {
        "arm": args.arm,
        "sequence_length": args.sequence_length,
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "warmup_steps": args.warmup_steps,
        **benchmark,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


def aggregate(args: argparse.Namespace) -> None:
    lengths = {
        "64": copy_reference(
            reference_run=args.length64_reference_run.resolve(),
            output_dir=args.output_dir,
            sequence_length=64,
        ),
        "1024": copy_reference(
            reference_run=args.length1024_reference_run.resolve(),
            output_dir=args.output_dir,
            sequence_length=1024,
        ),
    }
    for sequence_length in TRAIN_LENGTHS:
        lengths[str(sequence_length)] = load_trained_length(
            args.output_dir, sequence_length
        )
    lengths = {str(length): lengths[str(length)] for length in ALL_LENGTHS}
    for sequence_length in ALL_LENGTHS:
        for arm in ARMS:
            benchmark = json.loads(
                (
                    args.output_dir
                    / "fresh_process_benchmarks"
                    / f"length_{sequence_length}"
                    / f"{arm}.json"
                ).read_text()
            )
            score = lengths[str(sequence_length)]["arms"][arm]
            if benchmark["parameters"] != score["parameters"]:
                raise RuntimeError(
                    f"Benchmark parameter count drifted at L{sequence_length} {arm}"
                )
            score["fresh_process_benchmark"] = benchmark
    comparison = {
        "status": "complete",
        "protocol": {
            "plan": "P-CAUSAL-016",
            "sequence_lengths": list(ALL_LENGTHS),
            "new_sequence_lengths": list(TRAIN_LENGTHS),
            "num_kv_pairs": NUM_KV_PAIRS,
            "train_examples_per_new_arm": 10_000,
            "validation_examples_per_arm": 1_000,
            "epochs": args.max_epochs,
            "batch_size": args.batch_size,
            "seed": 123,
            "model": "official-FLA GDN2 D128/L2/H4/D32",
            "future_seed": "native cross-layer terminal-state seeding",
            "arm_process_isolation": "one fresh Python process per newly trained arm",
            "benchmark_process_isolation": "one fresh Python process per arm and length",
            "attention": "excluded because P-CAUSAL-015 closed this MQAR carrier",
        },
        "lengths": lengths,
        "registered_gate": registered_gate(lengths),
        "curve_diagnostics": curve_diagnostics(lengths),
        "binding_diagnostics": {
            str(length): binding_diagnostics(args.output_dir, length)
            for length in ALL_LENGTHS
        },
        "systems_metrics": systems_metrics(lengths),
    }
    (args.output_dir / "comparison.json").write_text(
        json.dumps(comparison, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(comparison, indent=2, sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    arm_parser = subparsers.add_parser("arm")
    arm_parser.add_argument("--output-dir", type=Path, required=True)
    arm_parser.add_argument("--arm", choices=ARMS, required=True)
    arm_parser.add_argument("--sequence-length", choices=TRAIN_LENGTHS, type=int, required=True)
    arm_parser.add_argument("--max-epochs", type=int, default=MAX_EPOCHS)
    arm_parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    arm_parser.set_defaults(handler=run_single_arm)

    benchmark_parser = subparsers.add_parser("benchmark")
    benchmark_parser.add_argument("--output", type=Path, required=True)
    benchmark_parser.add_argument("--arm", choices=ARMS, required=True)
    benchmark_parser.add_argument("--sequence-length", choices=ALL_LENGTHS, type=int, required=True)
    benchmark_parser.add_argument("--warmup-steps", type=int, default=5)
    benchmark_parser.add_argument("--measured-steps", type=int, default=20)
    benchmark_parser.set_defaults(handler=benchmark_single_arm)

    aggregate_parser = subparsers.add_parser("aggregate")
    aggregate_parser.add_argument("--output-dir", type=Path, required=True)
    aggregate_parser.add_argument("--length64-reference-run", type=Path, required=True)
    aggregate_parser.add_argument("--length1024-reference-run", type=Path, required=True)
    aggregate_parser.add_argument("--max-epochs", type=int, default=MAX_EPOCHS)
    aggregate_parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    aggregate_parser.set_defaults(handler=aggregate)

    args = parser.parse_args()
    if hasattr(args, "output_dir"):
        args.output_dir.mkdir(parents=True, exist_ok=True)
    args.handler(args)


if __name__ == "__main__":
    main()
