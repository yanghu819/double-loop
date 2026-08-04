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

from experiments.zoology_mqar.directional_mqar import DirectionalMQARConfig


VOCAB_SIZE = 256
SEQUENCE_LENGTH = 64
NUM_KV_PAIRS = 4
TRAIN_EXAMPLES = 10_000
VALID_EXAMPLES = 1_000
MODEL_WIDTH = 128
MODEL_LAYERS = 2
SEED = 123
P007_LENGTH64_TRAIN_HASH = (
    "31bac228c46a1c85b0b0e164675c7f65e9bb0a3d4b733d4c42d7c057bd6d5bf9"
)
P007_LENGTH64_TEST_HASH = (
    "3fa26a5a04b8302001231451bce23c4f5372648d8cdd9fe6116d8e418f470209"
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


def parameter_hash(model: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, parameter in sorted(model.named_parameters()):
        digest.update(name.encode())
        digest.update(parameter.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def dataset_hash(dataloader) -> str:
    tensors = []
    for segment in dataloader.dataset.segments:
        tensors.extend([segment.inputs, segment.labels])
    return _tensor_hash(tensors)


class CaptureLogger:
    def __init__(self, path: Path, arm: str) -> None:
        self.arm = arm
        self.rows: list[dict[str, Any]] = []
        self.train_counter = 0
        self.handle = path.open("w", buffering=1)

    def _write(self, payload: dict[str, Any]) -> None:
        self.handle.write(
            json.dumps(
                {"arm": self.arm, "time": time.time(), **payload},
                sort_keys=True,
            )
            + "\n"
        )

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


def benchmark_training_step(
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
    return {
        "measured_steps": float(measured_steps),
        "elapsed_sec": elapsed,
        "tokens_per_sec": inputs.numel() * measured_steps / elapsed,
        "examples_per_sec": inputs.shape[0] * measured_steps / elapsed,
        "peak_cuda_mem_bytes": float(torch.cuda.max_memory_allocated()),
    }


def _query_event(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    predictions: torch.Tensor,
    query_position: int,
) -> dict[str, Any]:
    key = int(inputs[query_position].item())
    target = int(targets[query_position].item())
    write_position = None
    for candidate in torch.nonzero(inputs == key).flatten().tolist():
        if candidate == query_position or candidate + 1 >= inputs.numel():
            continue
        if int(inputs[candidate + 1].item()) == target:
            write_position = int(candidate)
            break
    if write_position is None:
        raise RuntimeError("Could not recover write position for query")
    return {
        "direction": "future" if query_position < 16 else "past",
        "query_position": query_position,
        "write_position": write_position,
        "distance": write_position - query_position,
        "key": key,
        "target": target,
        "prediction": int(predictions[query_position].item()),
        "correct": bool(predictions[query_position].item() == target),
    }


@torch.no_grad()
def evaluate(model, dataloader) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    model.eval()
    totals: dict[str, dict[str, float]] = {}
    cases: list[dict[str, Any]] = []
    case_index = 0
    for inputs, targets, _slices in dataloader:
        inputs_gpu = inputs.cuda()
        targets_gpu = targets.cuda()
        logits = model(inputs_gpu)
        predictions = logits.argmax(dim=-1)
        label_mask = targets_gpu != -100
        positions = torch.arange(SEQUENCE_LENGTH, device="cuda")
        direction_masks = {
            "future": label_mask & (positions[None, :] < 16),
            "past": (
                label_mask
                & (positions[None, :] >= 32)
                & (positions[None, :] < 48)
            ),
        }
        for direction, direction_mask in direction_masks.items():
            correct = (predictions == targets_gpu) & direction_mask
            query_count = direction_mask.sum(dim=1)
            loss_sum = F.cross_entropy(
                logits[direction_mask],
                targets_gpu[direction_mask],
                reduction="sum",
            )
            bucket = totals.setdefault(
                direction,
                {
                    "correct": 0.0,
                    "queries": 0.0,
                    "exact": 0.0,
                    "examples": 0.0,
                    "loss": 0.0,
                },
            )
            bucket["correct"] += float(correct.sum().item())
            bucket["queries"] += float(direction_mask.sum().item())
            bucket["exact"] += float(
                (correct.sum(dim=1) == query_count).sum().item()
            )
            bucket["examples"] += float(inputs.shape[0])
            bucket["loss"] += float(loss_sum.item())

        predictions_cpu = predictions.cpu()
        for row in range(inputs.shape[0]):
            query_positions = torch.nonzero(targets[row] != -100).flatten()
            events = [
                _query_event(
                    inputs[row],
                    targets[row],
                    predictions_cpu[row],
                    int(position.item()),
                )
                for position in query_positions
            ]
            cases.append(
                {
                    "case_index": case_index,
                    "case_id": hashlib.sha256(
                        inputs[row].contiguous().numpy().tobytes()
                    ).hexdigest()[:16],
                    "future_errors": sum(
                        int(not event["correct"])
                        for event in events
                        if event["direction"] == "future"
                    ),
                    "past_errors": sum(
                        int(not event["correct"])
                        for event in events
                        if event["direction"] == "past"
                    ),
                    "events": events,
                }
            )
            case_index += 1

    metrics = {
        direction: {
            "accuracy": bucket["correct"] / bucket["queries"],
            "exact": bucket["exact"] / bucket["examples"],
            "ce": bucket["loss"] / bucket["queries"],
            "queries": int(bucket["queries"]),
            "examples": int(bucket["examples"]),
        }
        for direction, bucket in totals.items()
    }
    metrics["balanced_accuracy"] = (
        metrics["past"]["accuracy"] + metrics["future"]["accuracy"]
    ) / 2.0
    metrics["joint_exact"] = sum(
        int(case["future_errors"] + case["past_errors"] == 0)
        for case in cases
    ) / len(cases)
    cases.sort(
        key=lambda case: (
            -case["future_errors"],
            -case["past_errors"],
            case["case_index"],
        )
    )
    return metrics, cases


def build_config(*, bidirectional: bool, max_epochs: int = 30) -> TrainConfig:
    data = DataConfig(
        train_configs=[
            DirectionalMQARConfig(
                num_examples=TRAIN_EXAMPLES,
                vocab_size=VOCAB_SIZE,
                input_seq_len=SEQUENCE_LENGTH,
                num_kv_pairs=NUM_KV_PAIRS,
                direction="mixed",
            )
        ],
        test_configs=[
            DirectionalMQARConfig(
                num_examples=VALID_EXAMPLES,
                vocab_size=VOCAB_SIZE,
                input_seq_len=SEQUENCE_LENGTH,
                num_kv_pairs=NUM_KV_PAIRS,
                direction="mixed",
            )
        ],
        batch_size=32,
        seed=SEED,
        cache_dir="/huyang2/double-loop/.cache/zoology-directional-mqar",
    )
    mixer_name = (
        "experiments.zoology_mqar.official_bidirectional_mha."
        "OfficialBidirectionalMHA"
        if bidirectional
        else "zoology.mixers.attention.MHA"
    )
    model = ModelConfig(
        vocab_size=VOCAB_SIZE,
        max_position_embeddings=SEQUENCE_LENGTH,
        d_model=MODEL_WIDTH,
        n_layers=MODEL_LAYERS,
        sequence_mixer=ModuleConfig(
            name=mixer_name,
            kwargs={"dropout": 0.1, "num_heads": 1},
        ),
    )
    return TrainConfig(
        data=data,
        model=model,
        max_epochs=max_epochs,
        early_stopping_metric="valid/accuracy",
        early_stopping_threshold=0.99,
        learning_rate=1e-3,
        weight_decay=0.1,
        seed=SEED,
        slice_keys=[],
        run_id="official-zoology-bidirectional-directional-mqar",
    )


def first_opening_epoch(
    curve: list[dict[str, Any]],
    threshold: float,
) -> int | None:
    for row in curve:
        if float(row.get("valid/accuracy", -1.0)) >= threshold:
            return int(row["epoch"])
    return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-epochs", type=int, default=30)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    config = build_config(bidirectional=True, max_epochs=args.max_epochs)
    set_determinism(config.seed)
    model = LanguageModel(config.model)
    init_hash = model_hash(model)
    init_parameter_hash = parameter_hash(model)
    train_dataloader, test_dataloader = prepare_data(config.data)
    data_hashes = {
        "train": dataset_hash(train_dataloader),
        "test": dataset_hash(test_dataloader),
    }
    if data_hashes["train"] != P007_LENGTH64_TRAIN_HASH:
        raise RuntimeError("Training data drifted from P-CAUSAL-007")
    if data_hashes["test"] != P007_LENGTH64_TEST_HASH:
        raise RuntimeError("Validation data drifted from P-CAUSAL-007")

    logger = CaptureLogger(
        args.output_dir / "metrics.jsonl",
        "official_bidirectional_mha",
    )
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
    training_peak = torch.cuda.max_memory_allocated()
    metrics, cases = evaluate(model, test_dataloader)
    benchmark = benchmark_training_step(model, next(iter(train_dataloader)))
    completed_epochs = len(logger.rows)
    score = {
        "arm": "official_bidirectional_mha",
        "registered_carrier_pass": (
            metrics["past"]["accuracy"] >= 0.90
            and metrics["future"]["accuracy"] >= 0.90
        ),
        "metrics": metrics,
        "valid_curve": logger.rows,
        "opening_epoch_at_0.90": first_opening_epoch(logger.rows, 0.90),
        "opening_epoch_at_0.99": first_opening_epoch(logger.rows, 0.99),
        "configured_max_epochs": args.max_epochs,
        "completed_epochs": completed_epochs,
        "sequence_length": SEQUENCE_LENGTH,
        "num_kv_pairs": NUM_KV_PAIRS,
        "train_examples": TRAIN_EXAMPLES,
        "validation_examples": VALID_EXAMPLES,
        "train_tokens": TRAIN_EXAMPLES * SEQUENCE_LENGTH * completed_epochs,
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "init_hash": init_hash,
        "init_parameter_hash": init_parameter_hash,
        "data_hashes": data_hashes,
        "elapsed_sec_including_validation": elapsed,
        "peak_training_cuda_mem_bytes": training_peak,
        "warmed_step_benchmark": benchmark,
    }
    logger.finish()
    (args.output_dir / "config.json").write_text(
        json.dumps(config.model_dump(mode="json"), indent=2, sort_keys=True) + "\n"
    )
    (args.output_dir / "score.json").write_text(
        json.dumps(score, indent=2, sort_keys=True) + "\n"
    )
    (args.output_dir / "cases.json").write_text(
        json.dumps(cases, separators=(",", ":")) + "\n"
    )
    print(json.dumps(score, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
