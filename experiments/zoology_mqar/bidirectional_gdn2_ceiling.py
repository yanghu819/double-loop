from __future__ import annotations

import argparse
import copy
import hashlib
import json
import shutil
import statistics
import time
from pathlib import Path
from typing import Any

import torch

from zoology.config import ModuleConfig
from zoology.data.utils import prepare_data
from zoology.model import LanguageModel
from zoology.train import Trainer
from zoology.utils import set_determinism

from experiments.zoology_mqar.bidirectional_gdn2 import (
    ExplicitBidirectionalGDN2Mixer,
    bidirectional_diagnostics,
)
from experiments.zoology_mqar.futureseed_directionality import (
    CaptureLogger,
    dataset_hash,
    model_hash,
)
from experiments.zoology_mqar.length_scaling import (
    GDN2_HEAD_DIM,
    MODEL_HEADS,
    MODEL_WIDTH,
    benchmark_training_step,
    build_config,
    evaluate,
    make_model,
    parameter_hash,
    warm_model,
)


SEQUENCE_LENGTH = 512
NUM_KV_PAIRS = 4
BATCH_SIZE = 32
MAX_EPOCHS = 10
ARMS = ("causal_gdn2", "future_seed_gdn2", "explicit_bidirectional_gdn2")
REFERENCE_GIT_SHA = "cbe7060d1962ef4fd6f9c7df6e9556b34ec9779b"
REFERENCE_SCORE_SHA256 = (
    "a07b4103dee02e10a7c848614b1c2545c6dc6213e0199a02d415e3dbe1e582d8"
)


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_bidirectional_config():
    config = build_config(
        arm="causal_gdn2",
        sequence_length=SEQUENCE_LENGTH,
        num_kv_pairs=NUM_KV_PAIRS,
        max_epochs=MAX_EPOCHS,
        batch_size=BATCH_SIZE,
    )
    mixer = ModuleConfig(
        name=(
            "experiments.zoology_mqar.bidirectional_gdn2."
            "ExplicitBidirectionalGDN2Mixer"
        ),
        kwargs={
            "num_heads": MODEL_HEADS,
            "head_dim": GDN2_HEAD_DIM,
            "expand_v": 1.0,
            "conv_size": 4,
        },
    )
    model = config.model.model_copy(update={"sequence_mixer": mixer})
    return config.model_copy(
        update={
            "model": model,
            "run_id": "directional-explicit-bidirectional-gdn2-l512-k4",
        }
    )


def make_arm_model(arm: str):
    if arm == "explicit_bidirectional_gdn2":
        config = build_bidirectional_config()
    else:
        config = build_config(
            arm=arm,
            sequence_length=SEQUENCE_LENGTH,
            num_kv_pairs=NUM_KV_PAIRS,
            max_epochs=MAX_EPOCHS,
            batch_size=BATCH_SIZE,
        )
    set_determinism(config.seed)
    if arm == "explicit_bidirectional_gdn2":
        return config, LanguageModel(copy.deepcopy(config.model))
    return config, make_model(config, arm)


def run_bidirectional_arm(output_dir: Path) -> dict[str, Any]:
    arm = "explicit_bidirectional_gdn2"
    arm_dir = output_dir / arm
    arm_dir.mkdir(parents=True, exist_ok=True)
    config, model = make_arm_model(arm)
    init_hash = model_hash(model)
    init_parameter_hash = parameter_hash(model)
    train_loader, test_loader = prepare_data(config.data)
    hashes = {
        "train": dataset_hash(train_loader),
        "test": dataset_hash(test_loader),
    }
    fixed_batch = next(iter(train_loader))
    warm_model(model, fixed_batch)
    set_determinism(config.seed)

    logger = CaptureLogger(arm_dir / "metrics.jsonl", arm)
    logger.log_config(config)
    logger.log_model(model, config)
    trainer = Trainer(
        model=model,
        train_dataloader=train_loader,
        test_dataloader=test_loader,
        input_type=config.input_type,
        max_epochs=config.max_epochs,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        early_stopping_metric=config.early_stopping_metric,
        early_stopping_threshold=config.early_stopping_threshold,
        slice_keys=config.slice_keys,
        loss_type=config.loss_type,
        device="cuda",
        logger=logger,
    )
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    started = time.perf_counter()
    trainer.fit()
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    peak = torch.cuda.max_memory_allocated()
    metrics, cases = evaluate(
        model,
        test_loader,
        sequence_length=SEQUENCE_LENGTH,
    )
    score = {
        "arm": arm,
        "sequence_length": SEQUENCE_LENGTH,
        "num_kv_pairs": NUM_KV_PAIRS,
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "init_hash": init_hash,
        "init_parameter_hash": init_parameter_hash,
        "data_hashes": hashes,
        "epochs": MAX_EPOCHS,
        "train_examples": 10_000,
        "train_tokens": 10_000 * SEQUENCE_LENGTH * MAX_EPOCHS,
        "elapsed_sec_including_validation": elapsed,
        "peak_training_cuda_mem_bytes": peak,
        "metrics": metrics,
        "valid_curve": logger.rows,
        "bidirectional": bidirectional_diagnostics(model),
    }
    logger.finish()
    (arm_dir / "config.json").write_text(
        json.dumps(config.model_dump(mode="json"), indent=2, sort_keys=True) + "\n"
    )
    (arm_dir / "score.json").write_text(
        json.dumps(score, indent=2, sort_keys=True) + "\n"
    )
    (arm_dir / "cases.json").write_text(
        json.dumps(cases, separators=(",", ":")) + "\n"
    )
    del trainer, model
    torch.cuda.empty_cache()
    return score


def run_benchmark(args: argparse.Namespace) -> None:
    config, model = make_arm_model(args.arm)
    train_loader, _ = prepare_data(config.data)
    fixed_batch = next(iter(train_loader))
    warm_model(model, fixed_batch)
    result = {
        "arm": args.arm,
        "replicate": args.replicate,
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "sequence_length": SEQUENCE_LENGTH,
        "warmup_steps": args.warmup_steps,
        **benchmark_training_step(
            model,
            fixed_batch,
            warmup_steps=args.warmup_steps,
            measured_steps=args.measured_steps,
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


def benchmark_summary(output_dir: Path, arm: str) -> dict[str, Any]:
    rows = [
        json.loads(path.read_text())
        for path in sorted((output_dir / "benchmarks" / arm).glob("rep*.json"))
    ]
    if len(rows) != 3:
        raise RuntimeError(f"Expected three benchmark replicates for {arm}")
    throughput = [float(row["tokens_per_sec"]) for row in rows]
    memory = [float(row["peak_cuda_mem_bytes"]) for row in rows]
    return {
        "replicates": rows,
        "tokens_per_sec_median": statistics.median(throughput),
        "tokens_per_sec_min": min(throughput),
        "tokens_per_sec_max": max(throughput),
        "tokens_per_sec_relative_range": (
            (max(throughput) - min(throughput)) / statistics.median(throughput)
        ),
        "peak_cuda_mem_bytes_median": statistics.median(memory),
    }


def copy_reference(reference_run: Path, output_dir: Path) -> dict[str, Any]:
    if (reference_run / "git_sha.txt").read_text().strip() != REFERENCE_GIT_SHA:
        raise RuntimeError("P-CAUSAL-016 reference source SHA drifted")
    if file_sha256(reference_run / "score.json") != REFERENCE_SCORE_SHA256:
        raise RuntimeError("P-CAUSAL-016 reference score hash drifted")
    reference = json.loads((reference_run / "score.json").read_text())
    result = {}
    for arm in ("causal_gdn2", "future_seed_gdn2"):
        result[arm] = reference["lengths"]["512"]["arms"][arm]
        shutil.copytree(
            reference_run / "output" / "length_512" / arm,
            output_dir / "reference" / arm,
        )
    return result


def aggregate(args: argparse.Namespace) -> None:
    arms = copy_reference(args.reference_run.resolve(), args.output_dir)
    arms["explicit_bidirectional_gdn2"] = json.loads(
        (args.output_dir / "explicit_bidirectional_gdn2" / "score.json").read_text()
    )
    train_hashes = {arm: score["data_hashes"]["train"] for arm, score in arms.items()}
    test_hashes = {arm: score["data_hashes"]["test"] for arm, score in arms.items()}
    if len(set(train_hashes.values())) != 1 or len(set(test_hashes.values())) != 1:
        raise RuntimeError("Three-way carrier data hashes differ")

    systems = {arm: benchmark_summary(args.output_dir, arm) for arm in ARMS}
    fs = arms["future_seed_gdn2"]["metrics"]
    bidir = arms["explicit_bidirectional_gdn2"]["metrics"]
    quality_checks = {
        "bidirectional_past_at_least_0.95": bidir["past"]["accuracy"] >= 0.95,
        "bidirectional_future_at_least_0.95": bidir["future"]["accuracy"] >= 0.95,
        "bidirectional_joint_exact_at_least_0.90": bidir["joint_exact"] >= 0.90,
        "future_seed_future_within_0.03": (
            fs["future"]["accuracy"] >= bidir["future"]["accuracy"] - 0.03
        ),
        "future_seed_joint_within_0.05": fs["joint_exact"] >= bidir["joint_exact"] - 0.05,
    }
    fs_speed = systems["future_seed_gdn2"]["tokens_per_sec_median"]
    bidir_speed = systems["explicit_bidirectional_gdn2"]["tokens_per_sec_median"]
    fs_memory = systems["future_seed_gdn2"]["peak_cuda_mem_bytes_median"]
    bidir_memory = systems["explicit_bidirectional_gdn2"]["peak_cuda_mem_bytes_median"]
    efficiency_checks = {
        "future_seed_throughput_at_least_1.25x": fs_speed >= 1.25 * bidir_speed,
        "future_seed_peak_memory_at_most_0.80x": fs_memory <= 0.80 * bidir_memory,
        "all_benchmark_relative_ranges_at_most_0.15": all(
            systems[arm]["tokens_per_sec_relative_range"] <= 0.15 for arm in ARMS
        ),
    }
    comparison = {
        "status": "complete",
        "protocol": {
            "plan": "P-CAUSAL-017",
            "sequence_length": SEQUENCE_LENGTH,
            "num_kv_pairs": NUM_KV_PAIRS,
            "model": "D128/L2/H4/D32 official-FLA GDN2",
            "baseline": "two independent forward/reverse GDN2 streams per layer with concatenate-linear fusion",
            "training": "10k/1k, batch32, ten epochs, AdamW1e-3, WD0.1, cosine, seed123",
            "benchmark": (
                "three fresh processes per arm, three compile warmups plus "
                "5 benchmark warmups and 200 measured forward-backward steps"
            ),
            "future_seed": "native cross-layer terminal-state seeding; no reverse scan",
        },
        "arms": arms,
        "systems": systems,
        "quality_gate": {"passed": all(quality_checks.values()), "checks": quality_checks},
        "efficiency_gate": {
            "passed": all(efficiency_checks.values()),
            "checks": efficiency_checks,
            "future_seed_vs_bidirectional_throughput": fs_speed / bidir_speed,
            "future_seed_vs_bidirectional_peak_memory": fs_memory / bidir_memory,
        },
    }
    comparison["registered_gate"] = {
        "passed": comparison["quality_gate"]["passed"]
        and comparison["efficiency_gate"]["passed"],
        "carrier_opened": comparison["quality_gate"]["checks"][
            "bidirectional_future_at_least_0.95"
        ]
        and comparison["quality_gate"]["checks"][
            "bidirectional_past_at_least_0.95"
        ],
    }
    (args.output_dir / "comparison.json").write_text(
        json.dumps(comparison, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(comparison, indent=2, sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train")
    train_parser.add_argument("--output-dir", type=Path, required=True)
    train_parser.set_defaults(
        handler=lambda args: print(
            json.dumps(
                run_bidirectional_arm(args.output_dir),
                indent=2,
                sort_keys=True,
            )
        )
    )

    benchmark_parser = subparsers.add_parser("benchmark")
    benchmark_parser.add_argument("--output", type=Path, required=True)
    benchmark_parser.add_argument("--arm", choices=ARMS, required=True)
    benchmark_parser.add_argument("--replicate", type=int, choices=(1, 2, 3), required=True)
    benchmark_parser.add_argument("--warmup-steps", type=int, default=5)
    benchmark_parser.add_argument("--measured-steps", type=int, default=200)
    benchmark_parser.set_defaults(handler=run_benchmark)

    aggregate_parser = subparsers.add_parser("aggregate")
    aggregate_parser.add_argument("--output-dir", type=Path, required=True)
    aggregate_parser.add_argument("--reference-run", type=Path, required=True)
    aggregate_parser.set_defaults(handler=aggregate)

    args = parser.parse_args()
    args.handler(args)


if __name__ == "__main__":
    main()
