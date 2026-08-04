from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from zoology.config import DataConfig, ModelConfig, ModuleConfig, TrainConfig
from zoology.data.utils import prepare_data
from zoology.model import LanguageModel
from zoology.train import Trainer
from zoology.utils import set_determinism

from experiments.zoology_mqar.futureseed_directionality import (
    CaptureLogger,
    dataset_hash,
    model_hash,
)
from experiments.zoology_mqar.gdn2_futureseed import (
    FutureSeedLanguageModel,
    futureseed_diagnostics,
)
from experiments.zoology_mlm.wikitext_byte_mlm import WikiTextByteMLMConfig


VOCAB_SIZE = 257
MASK_TOKEN_ID = 256
SEQ_LEN = 256
TRAIN_EXAMPLES = 20_000
VALID_EXAMPLES = 2_000
MASKED_TOKENS_PER_EXAMPLE = 38
SEED = 123


def build_config(
    *,
    arm: str,
    prepared_path: Path,
    prepared_sha256: str,
    max_epochs: int,
    batch_size: int,
) -> TrainConfig:
    data = DataConfig(
        train_configs=[
            WikiTextByteMLMConfig(
                vocab_size=VOCAB_SIZE,
                num_examples=TRAIN_EXAMPLES,
                input_seq_len=SEQ_LEN,
                prepared_path=str(prepared_path),
                prepared_sha256=prepared_sha256,
                split="train",
            )
        ],
        test_configs=[
            WikiTextByteMLMConfig(
                vocab_size=VOCAB_SIZE,
                num_examples=VALID_EXAMPLES,
                input_seq_len=SEQ_LEN,
                prepared_path=str(prepared_path),
                prepared_sha256=prepared_sha256,
                split="valid",
            )
        ],
        batch_size=batch_size,
        seed=SEED,
        cache_dir=None,
    )
    if arm in {"causal_gdn2", "future_seed_gdn2"}:
        future_seed_scale = 0.0 if arm == "causal_gdn2" else 1.0
        sequence_mixer = ModuleConfig(
            name=(
                "experiments.zoology_mqar.gdn2_futureseed."
                "ZoologyGDN2FutureSeedMixer"
            ),
            kwargs={
                "num_heads": 4,
                "head_dim": 32,
                "expand_v": 1.0,
                "conv_size": 4,
                "future_seed_scale": future_seed_scale,
            },
        )
    elif arm == "bidirectional_attention":
        sequence_mixer = ModuleConfig(
            name=(
                "experiments.zoology_mlm.bidirectional_attention."
                "FullBidirectionalAttention"
            ),
            kwargs={"num_heads": 4},
        )
    else:
        raise ValueError(f"Unknown arm: {arm}")

    model = ModelConfig(
        vocab_size=VOCAB_SIZE,
        max_position_embeddings=SEQ_LEN,
        d_model=128,
        n_layers=2,
        sequence_mixer=sequence_mixer,
    )
    return TrainConfig(
        data=data,
        model=model,
        max_epochs=max_epochs,
        early_stopping_metric=None,
        early_stopping_threshold=1.1,
        learning_rate=1e-3,
        weight_decay=0.1,
        seed=SEED,
        slice_keys=[],
        run_id=f"wikitext-byte-mlm-{arm}",
    )


def _make_model(config: TrainConfig, arm: str) -> torch.nn.Module:
    if arm in {"causal_gdn2", "future_seed_gdn2"}:
        return FutureSeedLanguageModel(config.model)
    return LanguageModel(config.model)


def _parameter_hash(model: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, parameter in sorted(model.named_parameters()):
        digest.update(name.encode())
        digest.update(parameter.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def _warm_model(
    model: torch.nn.Module,
    batch: tuple[torch.Tensor, torch.Tensor, Any],
) -> None:
    model.train().cuda()
    inputs, labels, _slices = batch
    inputs = inputs.cuda()
    labels = labels.cuda()
    for _ in range(3):
        model.zero_grad(set_to_none=True)
        logits = model(inputs)
        loss = F.cross_entropy(logits.flatten(0, 1), labels.flatten())
        loss.backward()
    model.zero_grad(set_to_none=True)
    torch.cuda.synchronize()


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    dataloader,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    model.eval()
    loss_sum = 0.0
    correct = 0
    masked = 0
    exact = 0
    cases: list[dict[str, Any]] = []
    case_index = 0
    for inputs, labels, _slices in dataloader:
        inputs_gpu = inputs.cuda()
        labels_gpu = labels.cuda()
        logits = model(inputs_gpu)
        mask = labels_gpu != -100
        predictions = logits.argmax(dim=-1)
        loss_sum += float(
            F.cross_entropy(
                logits[mask], labels_gpu[mask], reduction="sum"
            ).item()
        )
        correct_mask = (predictions == labels_gpu) & mask
        correct += int(correct_mask.sum().item())
        masked += int(mask.sum().item())
        exact += int((correct_mask.sum(dim=1) == mask.sum(dim=1)).sum().item())

        inputs_cpu = inputs.cpu()
        labels_cpu = labels.cpu()
        predictions_cpu = predictions.cpu()
        correct_cpu = correct_mask.cpu()
        for row in range(inputs.shape[0]):
            positions = torch.nonzero(labels_cpu[row] != -100).flatten()
            errors = int((~correct_cpu[row, positions]).sum().item())
            original = inputs_cpu[row].clone()
            original[positions] = labels_cpu[row, positions]
            cases.append(
                {
                    "case_index": case_index,
                    "case_id": hashlib.sha256(
                        original.contiguous().numpy().tobytes()
                    ).hexdigest()[:16],
                    "errors": errors,
                    "input": inputs_cpu[row].tolist(),
                    "original": original.tolist(),
                    "masked_positions": positions.tolist(),
                    "targets": labels_cpu[row, positions].tolist(),
                    "predictions": predictions_cpu[row, positions].tolist(),
                }
            )
            case_index += 1
    cases.sort(key=lambda row: (-row["errors"], row["case_index"]))
    return (
        {
            "masked_ce": loss_sum / masked,
            "masked_accuracy": correct / masked,
            "masked_exact": exact / len(dataloader.dataset),
            "masked_tokens": masked,
            "examples": len(dataloader.dataset),
        },
        cases,
    )


def _benchmark(
    model: torch.nn.Module,
    batch: tuple[torch.Tensor, torch.Tensor, Any],
    *,
    warmup_steps: int = 5,
    measured_steps: int = 20,
) -> dict[str, float]:
    model.train()
    inputs, labels, _slices = batch
    inputs = inputs.cuda()
    labels = labels.cuda()

    def one_step() -> None:
        model.zero_grad(set_to_none=True)
        logits = model(inputs)
        loss = F.cross_entropy(logits.flatten(0, 1), labels.flatten())
        loss.backward()

    for _ in range(warmup_steps):
        one_step()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    for _ in range(measured_steps):
        one_step()
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    model.zero_grad(set_to_none=True)
    tokens = inputs.numel() * measured_steps
    return {
        "measured_steps": measured_steps,
        "elapsed_sec": elapsed,
        "tokens_per_sec": tokens / elapsed,
        "examples_per_sec": inputs.shape[0] * measured_steps / elapsed,
        "peak_cuda_mem_bytes": float(torch.cuda.max_memory_allocated()),
    }


def _opening_epoch(rows: list[dict[str, Any]]) -> int | None:
    for row in rows:
        if float(row.get("valid/accuracy", 0.0)) >= 0.10:
            return int(row["epoch"])
    return None


def run_arm(
    *,
    arm: str,
    output_dir: Path,
    prepared_path: Path,
    prepared_sha256: str,
    max_epochs: int,
    batch_size: int,
) -> dict[str, Any]:
    arm_dir = output_dir / arm
    arm_dir.mkdir(parents=True, exist_ok=True)
    config = build_config(
        arm=arm,
        prepared_path=prepared_path,
        prepared_sha256=prepared_sha256,
        max_epochs=max_epochs,
        batch_size=batch_size,
    )
    set_determinism(config.seed)
    model = _make_model(config, arm)
    init_hash = model_hash(model)
    parameter_hash = _parameter_hash(model)
    train_dataloader, valid_dataloader = prepare_data(config.data)
    data_hashes = {
        "train": dataset_hash(train_dataloader),
        "validation": dataset_hash(valid_dataloader),
    }
    fixed_batch = next(iter(train_dataloader))
    _warm_model(model, fixed_batch)
    set_determinism(config.seed)

    logger = CaptureLogger(arm_dir / "metrics.jsonl", arm)
    logger.log_config(config)
    logger.log_model(model, config)
    trainer = Trainer(
        model=model,
        train_dataloader=train_dataloader,
        test_dataloader=valid_dataloader,
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
    training_peak = torch.cuda.max_memory_allocated()
    metrics, cases = evaluate(model, valid_dataloader)
    benchmark = _benchmark(model, fixed_batch)
    score: dict[str, Any] = {
        "arm": arm,
        "init_hash": init_hash,
        "parameter_hash": parameter_hash,
        "data_hashes": data_hashes,
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "epochs": max_epochs,
        "train_examples": TRAIN_EXAMPLES,
        "train_tokens": TRAIN_EXAMPLES * SEQ_LEN * max_epochs,
        "elapsed_sec_including_validation": elapsed,
        "peak_training_cuda_mem_bytes": training_peak,
        "metrics": metrics,
        "valid_curve": logger.rows,
        "opening_epoch_at_0.10_accuracy": _opening_epoch(logger.rows),
        "warmed_step_benchmark": benchmark,
    }
    if arm in {"causal_gdn2", "future_seed_gdn2"}:
        score["future_seed"] = futureseed_diagnostics(model)
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


def _safe_ratio(numerator: float, denominator: float) -> float | None:
    if abs(denominator) < 1e-12:
        return None
    return numerator / denominator


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--prepared-path", type=Path, required=True)
    parser.add_argument("--prepared-sha256", required=True)
    parser.add_argument("--max-epochs", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    arms = ("causal_gdn2", "future_seed_gdn2", "bidirectional_attention")
    scores = {
        arm: run_arm(
            arm=arm,
            output_dir=args.output_dir,
            prepared_path=args.prepared_path,
            prepared_sha256=args.prepared_sha256,
            max_epochs=args.max_epochs,
            batch_size=args.batch_size,
        )
        for arm in arms
    }
    causal = scores["causal_gdn2"]
    future_seed = scores["future_seed_gdn2"]
    bidirectional = scores["bidirectional_attention"]
    if causal["init_hash"] != future_seed["init_hash"]:
        raise RuntimeError("Matched GDN2 arms did not start identically")
    if causal["parameter_hash"] != future_seed["parameter_hash"]:
        raise RuntimeError("Matched GDN2 parameter hashes differ")
    if len({score["data_hashes"]["train"] for score in scores.values()}) != 1:
        raise RuntimeError("The three arms did not use the same training data")
    if len({score["data_hashes"]["validation"] for score in scores.values()}) != 1:
        raise RuntimeError("The three arms did not use the same validation data")

    causal_accuracy = causal["metrics"]["masked_accuracy"]
    fs_accuracy = future_seed["metrics"]["masked_accuracy"]
    bidir_accuracy = bidirectional["metrics"]["masked_accuracy"]
    causal_ce = causal["metrics"]["masked_ce"]
    fs_ce = future_seed["metrics"]["masked_ce"]
    bidir_ce = bidirectional["metrics"]["masked_ce"]
    comparison = {
        "arms": scores,
        "future_seed_vs_causal": {
            "masked_accuracy_delta": fs_accuracy - causal_accuracy,
            "masked_ce_delta": fs_ce - causal_ce,
        },
        "bidirectional_vs_causal": {
            "masked_accuracy_delta": bidir_accuracy - causal_accuracy,
            "masked_ce_delta": bidir_ce - causal_ce,
        },
        "future_seed_gap_closure": {
            "accuracy": _safe_ratio(
                fs_accuracy - causal_accuracy,
                bidir_accuracy - causal_accuracy,
            ),
            "ce": _safe_ratio(causal_ce - fs_ce, causal_ce - bidir_ce),
        },
        "protocol": {
            "vocab": "UTF-8 bytes plus one mask token",
            "mask_rate": 0.15,
            "sequence_length": SEQ_LEN,
            "train_examples": TRAIN_EXAMPLES,
            "validation_examples": VALID_EXAMPLES,
            "seed": SEED,
            "loops": "not used; this gate isolates cross-layer state transfer",
        },
    }
    (args.output_dir / "comparison.json").write_text(
        json.dumps(comparison, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(comparison, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
