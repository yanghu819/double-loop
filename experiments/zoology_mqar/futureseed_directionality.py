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
from zoology.train import Trainer
from zoology.utils import set_determinism

from experiments.zoology_mqar.directional_mqar import DirectionalMQARConfig
from experiments.zoology_mqar.gdn2_futureseed import (
    FutureSeedLanguageModel,
    futureseed_diagnostics,
)


def build_config(*, future_seed_scale: float, max_epochs: int) -> TrainConfig:
    data = DataConfig(
        train_configs=[
            DirectionalMQARConfig(
                num_examples=5_000,
                vocab_size=256,
                input_seq_len=64,
                num_kv_pairs=4,
                direction=direction,
            )
            for direction in ("past", "future")
        ],
        test_configs=[
            DirectionalMQARConfig(
                num_examples=500,
                vocab_size=256,
                input_seq_len=64,
                num_kv_pairs=4,
                direction=direction,
            )
            for direction in ("past", "future")
        ],
        batch_size=32,
        seed=123,
        cache_dir="/huyang2/double-loop/.cache/zoology-directional-mqar",
    )
    model = ModelConfig(
        vocab_size=256,
        max_position_embeddings=64,
        d_model=128,
        n_layers=2,
        sequence_mixer=ModuleConfig(
            name=(
                "experiments.zoology_mqar.gdn2_futureseed."
                "ZoologyGDN2FutureSeedMixer"
            ),
            kwargs={
                "num_heads": 4,
                "head_dim": 32,
                "expand_v": 1.0,
                "conv_size": 4,
                "future_seed_scale": float(future_seed_scale),
            },
        ),
    )
    return TrainConfig(
        data=data,
        model=model,
        max_epochs=max_epochs,
        early_stopping_metric=None,
        learning_rate=1e-3,
        weight_decay=0.1,
        seed=123,
        slice_keys=["direction"],
        run_id=f"gdn2-fs{int(future_seed_scale)}-directionality",
    )


def _tensor_hash(tensors: list[torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for tensor in tensors:
        array = tensor.detach().cpu().contiguous().numpy()
        digest.update(str(array.shape).encode())
        digest.update(str(array.dtype).encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


def model_hash(model: torch.nn.Module) -> str:
    return _tensor_hash(
        [tensor for _name, tensor in sorted(model.state_dict().items())]
    )


def dataset_hash(dataloader) -> str:
    tensors = []
    for segment in dataloader.dataset.segments:
        tensors.extend([segment.inputs, segment.labels])
    return _tensor_hash(tensors)


class CaptureLogger:
    def __init__(self, path: Path, arm: str) -> None:
        self.path = path
        self.arm = arm
        self.rows: list[dict[str, Any]] = []
        self.train_counter = 0
        self.handle = path.open("w", buffering=1)

    def _write(self, payload: dict[str, Any]) -> None:
        row = {"arm": self.arm, "time": time.time(), **payload}
        self.handle.write(json.dumps(row, sort_keys=True) + "\n")

    def log_config(self, config: TrainConfig) -> None:
        self._write({"event": "config", "config": config.model_dump(mode="json")})

    def log_model(self, model: torch.nn.Module, config: TrainConfig) -> None:
        del config
        self._write(
            {
                "event": "model",
                "parameters": sum(p.numel() for p in model.parameters()),
            }
        )

    def log(self, metrics: dict[str, Any]) -> None:
        if "train/loss" in metrics:
            self.train_counter += 1
            if self.train_counter % 25:
                return
        payload = {
            key: float(value) if hasattr(value, "item") else value
            for key, value in metrics.items()
        }
        self._write(payload)
        if "valid/accuracy" in payload:
            self.rows.append(payload)

    def finish(self) -> None:
        self.handle.close()


@torch.no_grad()
def evaluate(model, dataloader, device: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    model.eval()
    totals: dict[str, dict[str, float]] = {}
    cases: list[dict[str, Any]] = []
    future_case_index = 0
    for inputs, targets, slices in dataloader:
        direction = str(slices[0]["direction"])
        inputs_gpu = inputs.to(device)
        targets_gpu = targets.to(device)
        logits = model(inputs_gpu)
        mask = targets_gpu != -100
        loss_sum = F.cross_entropy(
            logits[mask],
            targets_gpu[mask],
            reduction="sum",
        )
        preds = logits.argmax(dim=-1)
        correct = (preds == targets_gpu) & mask
        query_count = mask.sum(dim=1)
        example_correct = correct.sum(dim=1)
        exact = example_correct == query_count
        bucket = totals.setdefault(
            direction,
            {"correct": 0.0, "queries": 0.0, "exact": 0.0, "examples": 0.0, "loss": 0.0},
        )
        bucket["correct"] += float(correct.sum().item())
        bucket["queries"] += float(mask.sum().item())
        bucket["exact"] += float(exact.sum().item())
        bucket["examples"] += float(inputs.shape[0])
        bucket["loss"] += float(loss_sum.item())

        if direction == "future":
            inputs_cpu = inputs.cpu()
            targets_cpu = targets.cpu()
            preds_cpu = preds.cpu()
            for index in range(inputs.shape[0]):
                positions = torch.nonzero(targets_cpu[index] != -100).flatten()
                errors = int(
                    (preds_cpu[index, positions] != targets_cpu[index, positions])
                    .sum()
                    .item()
                )
                cases.append(
                    {
                        "case_index": future_case_index,
                        "case_id": hashlib.sha256(
                            inputs_cpu[index].contiguous().numpy().tobytes()
                        ).hexdigest()[:16],
                        "errors": errors,
                        "input": inputs_cpu[index].tolist(),
                        "query_positions": positions.tolist(),
                        "targets": targets_cpu[index, positions].tolist(),
                        "predictions": preds_cpu[index, positions].tolist(),
                    }
                )
                future_case_index += 1

    metrics: dict[str, Any] = {}
    for direction, bucket in totals.items():
        metrics[direction] = {
            "accuracy": bucket["correct"] / bucket["queries"],
            "exact": bucket["exact"] / bucket["examples"],
            "ce": bucket["loss"] / bucket["queries"],
            "queries": int(bucket["queries"]),
            "examples": int(bucket["examples"]),
        }
    metrics["balanced_accuracy"] = sum(
        metrics[direction]["accuracy"] for direction in ("past", "future")
    ) / 2.0
    cases.sort(key=lambda row: (-row["errors"], row["query_positions"]))
    return metrics, cases


def run_arm(
    *,
    arm: str,
    future_seed_scale: float,
    output_dir: Path,
    max_epochs: int,
) -> dict[str, Any]:
    arm_dir = output_dir / arm
    arm_dir.mkdir(parents=True, exist_ok=True)
    config = build_config(
        future_seed_scale=future_seed_scale,
        max_epochs=max_epochs,
    )
    set_determinism(config.seed)
    model = FutureSeedLanguageModel(config.model)
    init_hash = model_hash(model)
    train_dataloader, test_dataloader = prepare_data(config.data)
    data_hashes = {
        "train": dataset_hash(train_dataloader),
        "test": dataset_hash(test_dataloader),
    }
    logger = CaptureLogger(arm_dir / "metrics.jsonl", arm)
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
    metrics, cases = evaluate(model, test_dataloader, "cuda")
    score = {
        "arm": arm,
        "future_seed_scale": future_seed_scale,
        "init_hash": init_hash,
        "data_hashes": data_hashes,
        "parameters": sum(p.numel() for p in model.parameters()),
        "epochs": max_epochs,
        "elapsed_sec": elapsed,
        "train_examples_per_sec": (10_000 * max_epochs) / elapsed,
        "peak_cuda_mem_bytes": torch.cuda.max_memory_allocated(),
        "metrics": metrics,
        "future_seed": futureseed_diagnostics(model),
        "valid_curve": logger.rows,
    }
    logger.finish()
    (arm_dir / "config.json").write_text(
        json.dumps(config.model_dump(mode="json"), indent=2, sort_keys=True) + "\n"
    )
    (arm_dir / "score.json").write_text(
        json.dumps(score, indent=2, sort_keys=True) + "\n"
    )
    (arm_dir / "cases.json").write_text(
        json.dumps(cases, indent=2, sort_keys=True) + "\n"
    )
    del trainer, model
    torch.cuda.empty_cache()
    return score


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-epochs", type=int, default=10)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    scores = [
        run_arm(
            arm="no_future_seed",
            future_seed_scale=0.0,
            output_dir=args.output_dir,
            max_epochs=args.max_epochs,
        ),
        run_arm(
            arm="future_seed",
            future_seed_scale=1.0,
            output_dir=args.output_dir,
            max_epochs=args.max_epochs,
        ),
    ]
    no_fs, fs = scores
    if no_fs["init_hash"] != fs["init_hash"]:
        raise RuntimeError("Matched arms did not start from identical parameters")
    if no_fs["data_hashes"] != fs["data_hashes"]:
        raise RuntimeError("Matched arms did not use identical data")
    no_fs_cases = json.loads(
        (args.output_dir / "no_future_seed" / "cases.json").read_text()
    )
    fs_cases = json.loads(
        (args.output_dir / "future_seed" / "cases.json").read_text()
    )
    fs_cases_by_id = {case["case_id"]: case for case in fs_cases}
    paired_cases = []
    for no_fs_case in no_fs_cases:
        fs_case = fs_cases_by_id.get(no_fs_case["case_id"])
        if fs_case is None:
            raise RuntimeError("Matched case IDs differ across arms")
        paired_cases.append(
            {
                "case_id": no_fs_case["case_id"],
                "case_index": no_fs_case["case_index"],
                "input": no_fs_case["input"],
                "query_positions": no_fs_case["query_positions"],
                "targets": no_fs_case["targets"],
                "no_future_seed_predictions": no_fs_case["predictions"],
                "future_seed_predictions": fs_case["predictions"],
                "no_future_seed_errors": no_fs_case["errors"],
                "future_seed_errors": fs_case["errors"],
            }
        )
    paired_cases.sort(
        key=lambda case: (
            -case["no_future_seed_errors"],
            case["future_seed_errors"],
            case["case_index"],
        )
    )
    (args.output_dir / "paired_hardest_cases.json").write_text(
        json.dumps(paired_cases[:32], indent=2, sort_keys=True) + "\n"
    )
    comparison = {
        "no_future_seed": no_fs,
        "future_seed": fs,
        "future_accuracy_delta": (
            fs["metrics"]["future"]["accuracy"]
            - no_fs["metrics"]["future"]["accuracy"]
        ),
        "past_accuracy_delta": (
            fs["metrics"]["past"]["accuracy"]
            - no_fs["metrics"]["past"]["accuracy"]
        ),
        "balanced_accuracy_delta": (
            fs["metrics"]["balanced_accuracy"]
            - no_fs["metrics"]["balanced_accuracy"]
        ),
    }
    (args.output_dir / "comparison.json").write_text(
        json.dumps(comparison, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(comparison, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
