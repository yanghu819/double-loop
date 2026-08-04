from __future__ import annotations

import argparse
import copy
import json
import shutil
import time
from pathlib import Path
from typing import Any

import torch

from zoology.config import DataConfig, ModelConfig, ModuleConfig, TrainConfig
from zoology.data.utils import prepare_data
from zoology.model import LanguageModel
from zoology.train import Trainer
from zoology.utils import set_determinism

from experiments.zoology_mqar.directional_mqar import DirectionalMQARConfig
from experiments.zoology_mqar.futureseed_directionality import (
    CaptureLogger,
    dataset_hash,
    model_hash,
)
from experiments.zoology_mqar.length_scaling import (
    benchmark_training_step,
    evaluate,
    parameter_hash,
    warm_model,
)


SEQUENCE_LENGTH = 64
NUM_KV_PAIRS = 4
TRAIN_EXAMPLES = 10_000
VALID_EXAMPLES = 1_000
MODEL_WIDTH = 128
MODEL_LAYERS = 2
NUM_HEADS = 4
HEAD_DIM = 57
MAX_EPOCHS = 30
BATCH_SIZE = 32
SEED = 123
EXPECTED_ENDPOINT_SHA = "4b909629ee3cc32ab1c57e9d3ac1f26ecaa8b273"
EXPECTED_OFFICIAL_SHA = "40002497ed5e56f1879b996d09608065c963b27d"


def build_config(*, rope_scale: float) -> TrainConfig:
    data = DataConfig(
        train_configs=[
            DirectionalMQARConfig(
                num_examples=TRAIN_EXAMPLES,
                vocab_size=256,
                input_seq_len=SEQUENCE_LENGTH,
                num_kv_pairs=NUM_KV_PAIRS,
                direction="mixed",
            )
        ],
        test_configs=[
            DirectionalMQARConfig(
                num_examples=VALID_EXAMPLES,
                vocab_size=256,
                input_seq_len=SEQUENCE_LENGTH,
                num_kv_pairs=NUM_KV_PAIRS,
                direction="mixed",
            )
        ],
        batch_size=BATCH_SIZE,
        seed=SEED,
        cache_dir=(
            "/huyang2/double-loop/.cache/"
            "zoology-directional-scaling-l64-k4"
        ),
    )
    model = ModelConfig(
        vocab_size=256,
        max_position_embeddings=SEQUENCE_LENGTH,
        d_model=MODEL_WIDTH,
        n_layers=MODEL_LAYERS,
        sequence_mixer=ModuleConfig(
            name=(
                "experiments.zoology_mqar.rope_bidirectional_attention."
                "ParamMatchedRoPEBidirectionalAttention"
            ),
            kwargs={
                "num_heads": NUM_HEADS,
                "head_dim": HEAD_DIM,
                "rope_theta": 10_000.0,
                "rope_scale": rope_scale,
            },
        ),
    )
    return TrainConfig(
        data=data,
        model=model,
        max_epochs=MAX_EPOCHS,
        early_stopping_metric="valid/accuracy",
        early_stopping_threshold=1.1,
        learning_rate=1e-3,
        weight_decay=0.1,
        seed=SEED,
        slice_keys=[],
        run_id="zoology-rope-bidirectional-carrier",
    )


def first_opening_epoch(curve: list[dict[str, Any]], threshold: float) -> int | None:
    for row in curve:
        if float(row["valid/accuracy"]) >= threshold:
            return int(row["epoch"])
    return None


def load_references(endpoint_run: Path, official_run: Path, output_dir: Path) -> dict[str, Any]:
    if (endpoint_run / "git_sha.txt").read_text().strip() != EXPECTED_ENDPOINT_SHA:
        raise RuntimeError("Frozen P-CAUSAL-010 source SHA changed")
    if (official_run / "git_sha.txt").read_text().strip() != EXPECTED_OFFICIAL_SHA:
        raise RuntimeError("Frozen P-CAUSAL-011 source SHA changed")
    references_dir = output_dir / "references"
    references_dir.mkdir(parents=True, exist_ok=True)
    endpoint_source = endpoint_run / "output" / "length_64"
    for arm in ("causal_gdn2", "future_seed_gdn2", "bidirectional_attention"):
        shutil.copytree(
            endpoint_source / arm,
            references_dir / arm,
            dirs_exist_ok=False,
        )
    shutil.copytree(
        official_run / "output",
        references_dir / "official_bidirectional_mha",
        dirs_exist_ok=False,
    )
    scores = {
        arm: json.loads((references_dir / arm / "score.json").read_text())
        for arm in (
            "causal_gdn2",
            "future_seed_gdn2",
            "bidirectional_attention",
            "official_bidirectional_mha",
        )
    }
    hashes = {arm: score["data_hashes"] for arm, score in scores.items()}
    if len({json.dumps(value, sort_keys=True) for value in hashes.values()}) != 1:
        raise RuntimeError(f"Frozen reference data hashes differ: {hashes}")
    return scores


def run_candidate(output_dir: Path) -> dict[str, Any]:
    arm_dir = output_dir / "rope_bidirectional_attention"
    arm_dir.mkdir(parents=True, exist_ok=True)
    config = build_config(rope_scale=1.0)
    set_determinism(config.seed)
    model = LanguageModel(copy.deepcopy(config.model))
    init_hash = model_hash(model)
    init_parameter_hash = parameter_hash(model)
    train_dataloader, test_dataloader = prepare_data(config.data)
    data_hashes = {
        "train": dataset_hash(train_dataloader),
        "test": dataset_hash(test_dataloader),
    }
    fixed_batch = next(iter(train_dataloader))
    warm_model(model, fixed_batch)
    set_determinism(config.seed)

    logger = CaptureLogger(arm_dir / "metrics.jsonl", "rope_bidirectional_attention")
    logger.log_config(config)
    logger.log_model(model, config)
    trainer = Trainer(
        model=model,
        train_dataloader=train_dataloader,
        test_dataloader=test_dataloader,
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
    peak_memory = torch.cuda.max_memory_allocated()
    metrics, cases = evaluate(
        model,
        test_dataloader,
        sequence_length=SEQUENCE_LENGTH,
    )
    benchmark = benchmark_training_step(model, fixed_batch)
    score = {
        "arm": "rope_bidirectional_attention",
        "sequence_length": SEQUENCE_LENGTH,
        "num_kv_pairs": NUM_KV_PAIRS,
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "init_hash": init_hash,
        "init_parameter_hash": init_parameter_hash,
        "data_hashes": data_hashes,
        "epochs": MAX_EPOCHS,
        "train_examples": TRAIN_EXAMPLES,
        "train_tokens": TRAIN_EXAMPLES * SEQUENCE_LENGTH * MAX_EPOCHS,
        "elapsed_sec_including_validation": elapsed,
        "peak_training_cuda_mem_bytes": peak_memory,
        "metrics": metrics,
        "valid_curve": logger.rows,
        "opening_epoch_at_0.90": first_opening_epoch(logger.rows, 0.90),
        "opening_epoch_at_0.99": first_opening_epoch(logger.rows, 0.99),
        "warmed_step_benchmark": benchmark,
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
    return score


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--endpoint-reference", type=Path, required=True)
    parser.add_argument("--official-reference", type=Path, required=True)
    parser.add_argument("--max-epochs", type=int, default=MAX_EPOCHS)
    args = parser.parse_args()
    if args.max_epochs != MAX_EPOCHS:
        raise ValueError("P-CAUSAL-015 uses the preregistered 30-epoch budget")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    references = load_references(
        args.endpoint_reference.resolve(),
        args.official_reference.resolve(),
        args.output_dir,
    )
    candidate = run_candidate(args.output_dir)
    if candidate["data_hashes"] != references["causal_gdn2"]["data_hashes"]:
        raise RuntimeError("Candidate data drifted from frozen references")
    old_attention = references["bidirectional_attention"]
    if candidate["parameters"] != old_attention["parameters"]:
        raise RuntimeError("RoPE changed the parameter budget")
    if candidate["init_hash"] != old_attention["init_hash"]:
        raise RuntimeError("RoPE candidate did not preserve the old initialized tensors")
    metrics = candidate["metrics"]
    checks = {
        "past_accuracy_at_least_0.90": metrics["past"]["accuracy"] >= 0.90,
        "future_accuracy_at_least_0.90": metrics["future"]["accuracy"] >= 0.90,
        "joint_exact_at_least_0.80": metrics["joint_exact"] >= 0.80,
    }
    comparison = {
        "status": "complete",
        "protocol": {
            "plan": "P-CAUSAL-015",
            "sequence_length": SEQUENCE_LENGTH,
            "num_kv_pairs": NUM_KV_PAIRS,
            "epochs": MAX_EPOCHS,
            "seed": SEED,
            "intervention": "parameter-free RoPE on full noncausal SDPA Q/K",
        },
        "references": references,
        "candidate": candidate,
        "registered_gate": {
            "passed": all(checks.values()),
            "checks": checks,
            "balanced_gain_over_plain_sdpa": (
                metrics["balanced_accuracy"]
                - old_attention["metrics"]["balanced_accuracy"]
            ),
            "joint_gain_over_plain_sdpa": (
                metrics["joint_exact"] - old_attention["metrics"]["joint_exact"]
            ),
            "gap_to_future_seed_balanced": (
                references["future_seed_gdn2"]["metrics"]["balanced_accuracy"]
                - metrics["balanced_accuracy"]
            ),
        },
    }
    (args.output_dir / "comparison.json").write_text(
        json.dumps(comparison, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(comparison, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
