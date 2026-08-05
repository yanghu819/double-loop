from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import math
import subprocess
import time
from pathlib import Path
from typing import Any, Callable

import torch
import torch.nn.functional as F
import transformers
from torch import nn
from transformers import AutoTokenizer, BertForMaskedLM

from fla.layers.gdn2 import GatedDeltaNet2

from experiments.zoology_mlm.wordpiece_futureseed_mlm import (
    EXPECTED_BERT_SOURCE_SHA256,
    EXPECTED_GDN2_SOURCE_SHA256,
    EXPECTED_TRANSFORMERS_VERSION,
    build_model,
    corruption,
    lexical_integrity,
    load_grouped_examples,
    sha256,
    state_hash,
    tensor_digest,
    verify_fla_tree,
)
from experiments.zoology_mqar.gdn2_futureseed import futureseed_diagnostics


EXPECTED_GPU_UUID = "GPU-53e9f3b4-2966-65d3-6614-09c540921519"
EXPECTED_VALIDATION_TENSOR_SHA256 = (
    "26d57156e43b4d7c21036b1340f23b0e418bf4aed32366bccc4ec252685b0ada"
)
EXPECTED_LEXICAL_SHA256 = (
    "c21163c6a84ae83c3ce10f4a7db66f5e63d2468e09330a5794c68ac67c17682d"
)
EXPECTED_QUALITY = {
    "bidirectional_bert": {
        "masked_accuracy": 0.35554618304512864,
        "masked_ce": 4.033268768040252,
    },
    "causal_gdn2": {
        "masked_accuracy": 0.30936313791649095,
        "masked_ce": 4.618275784380272,
    },
    "future_seed_gdn2": {
        "masked_accuracy": 0.37389287220582035,
        "masked_ce": 3.9051307218077813,
    },
}
EXPECTED_PARAMETERS = {
    "bidirectional_bert": 4_416_698,
    "causal_gdn2": 4_965_722,
    "future_seed_gdn2": 4_965_722,
}
QUALITY_CORRECT_TOLERANCE = 1
QUALITY_CE_TOLERANCE = 1e-3


def visible_gpu() -> dict[str, str]:
    output = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,uuid,name,memory.total",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip().splitlines()
    if len(output) != 1:
        raise RuntimeError(f"Expected one visible GPU, got {output}")
    index, uuid, name, memory_total = [part.strip() for part in output[0].split(",")]
    if index != "0" or uuid != EXPECTED_GPU_UUID:
        raise RuntimeError(f"Unexpected visible GPU: {output[0]}")
    return {
        "index": index,
        "uuid": uuid,
        "name": name,
        "memory_total_mib": memory_total,
    }


def token_text(tokenizer: Any, token_id: int) -> str:
    return str(tokenizer.convert_ids_to_tokens(int(token_id)))


def prepare_validation(
    validation_json: Path,
    tokenizer: Any,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    examples = load_grouped_examples(validation_json, tokenizer, 128, 256)
    originals = torch.tensor(
        [example["input_ids"] for example in examples], dtype=torch.long
    ).contiguous()
    inputs, labels = corruption(examples, tokenizer, 0.15, 123)
    actual_hash = tensor_digest([("inputs", inputs), ("labels", labels)])
    if actual_hash != EXPECTED_VALIDATION_TENSOR_SHA256:
        raise RuntimeError(f"Fixed validation tensors drifted: {actual_hash}")
    if int((labels != -100).sum().item()) != 4_742:
        raise RuntimeError("Fixed validation masked-token count drifted")
    return originals, inputs, labels


class InferenceAdapter:
    def __init__(self, arm: str, model: nn.Module) -> None:
        self.arm = arm
        self.model = model

    def encode(self, input_ids: torch.Tensor) -> torch.Tensor:
        if self.arm == "bidirectional_bert":
            attention_mask = torch.ones_like(input_ids)
            return self.model.bert(
                input_ids=input_ids,
                attention_mask=attention_mask,
            ).last_hidden_state
        return self.model.encode(input_ids)

    def masked_head(self, hidden: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        if self.arm == "bidirectional_bert":
            return self.model.cls(hidden[mask])
        return self.model.mlm_head(hidden[mask])


def load_arm(
    arm: str,
    model_dir: Path,
    bert_checkpoint: Path,
    gdn_checkpoint: Path | None,
    gdn_checkpoint_sha256: str | None,
    fla_wheel: Path,
) -> tuple[InferenceAdapter, dict[str, Any]]:
    if transformers.__version__ != EXPECTED_TRANSFORMERS_VERSION:
        raise RuntimeError(f"Unexpected Transformers version: {transformers.__version__}")
    bert_source = Path(inspect.getfile(BertForMaskedLM)).resolve()
    if sha256(bert_source) != EXPECTED_BERT_SOURCE_SHA256:
        raise RuntimeError(f"Unexpected BERT implementation: {bert_source}")

    pretrained = BertForMaskedLM.from_pretrained(
        model_dir,
        local_files_only=True,
        attn_implementation="sdpa",
        use_safetensors=bert_checkpoint.suffix == ".safetensors",
    ).eval()
    if arm == "bidirectional_bert":
        if pretrained.config.is_decoder:
            raise RuntimeError("Registered BERT checkpoint is unexpectedly causal")
        model = pretrained
        provenance: dict[str, Any] = {
            "implementation": "transformers.BertForMaskedLM native bidirectional SDPA",
            "bert_source": str(bert_source),
            "bert_source_sha256": EXPECTED_BERT_SOURCE_SHA256,
            "checkpoint_sha256": sha256(bert_checkpoint),
        }
    else:
        if gdn_checkpoint is None or gdn_checkpoint_sha256 is None:
            raise RuntimeError("GDN2 arm requires a registered checkpoint")
        actual_checkpoint_sha = sha256(gdn_checkpoint)
        if actual_checkpoint_sha != gdn_checkpoint_sha256:
            raise RuntimeError(
                f"GDN2 checkpoint hash drifted: {actual_checkpoint_sha}"
            )
        fla_provenance = verify_fla_tree(fla_wheel)
        gdn2_source = Path(inspect.getfile(GatedDeltaNet2)).resolve()
        if sha256(gdn2_source) != EXPECTED_GDN2_SOURCE_SHA256:
            raise RuntimeError(f"Unexpected GDN2 implementation: {gdn2_source}")
        scale = 0.0 if arm == "causal_gdn2" else 1.0
        model = build_model(pretrained, future_seed_scale=scale)
        checkpoint_state = torch.load(
            gdn_checkpoint, map_location="cpu", weights_only=True
        )
        model.load_state_dict(checkpoint_state, strict=True)
        lexical = lexical_integrity(model, EXPECTED_LEXICAL_SHA256)
        provenance = {
            "implementation": "pinned official-FLA GDN2 chunk/Triton",
            "fla": fla_provenance,
            "checkpoint_sha256": actual_checkpoint_sha,
            "checkpoint_state_sha256": tensor_digest(checkpoint_state.items()),
            "lexical_integrity": lexical,
            "future_seed": futureseed_diagnostics(model),
        }
        del checkpoint_state, pretrained

    parameters = sum(parameter.numel() for parameter in model.parameters())
    if parameters != EXPECTED_PARAMETERS[arm]:
        raise RuntimeError(f"{arm} parameter count drifted: {parameters}")
    model = model.cuda().eval()
    return InferenceAdapter(arm, model), {
        "parameters": parameters,
        "model_state_sha256": state_hash(model),
        **provenance,
    }


@torch.no_grad()
def quality_metrics(
    adapter: InferenceAdapter,
    originals: torch.Tensor,
    inputs: torch.Tensor,
    labels: torch.Tensor,
    tokenizer: Any,
) -> tuple[dict[str, float | int], list[dict[str, Any]]]:
    batch_size = 32 if adapter.arm == "bidirectional_bert" else 64
    loss_sum = 0.0
    correct = 0
    masked = 0
    cases: list[dict[str, Any]] = []
    for start in range(0, inputs.shape[0], batch_size):
        batch_inputs = inputs[start : start + batch_size].cuda()
        batch_labels = labels[start : start + batch_size].cuda()
        mask = batch_labels != -100
        with torch.autocast("cuda", dtype=torch.bfloat16):
            hidden = adapter.encode(batch_inputs)
            logits = adapter.masked_head(hidden, mask).float()
        targets = batch_labels[mask]
        predictions = logits.argmax(dim=-1)
        probabilities = logits.softmax(dim=-1)
        loss_sum += float(F.cross_entropy(logits, targets, reduction="sum").item())
        correct += int((predictions == targets).sum().item())
        masked += int(targets.numel())

        offset = 0
        for local_row in range(batch_inputs.shape[0]):
            positions = torch.nonzero(mask[local_row]).flatten().tolist()
            count = len(positions)
            row_predictions = predictions[offset : offset + count]
            row_probabilities = probabilities[offset : offset + count]
            row_targets = targets[offset : offset + count]
            global_row = start + local_row
            masked_rows = []
            for index, position in enumerate(positions):
                target_id = int(row_targets[index])
                prediction_id = int(row_predictions[index])
                masked_rows.append(
                    {
                        "position": int(position),
                        "target_id": target_id,
                        "target": token_text(tokenizer, target_id),
                        "prediction_id": prediction_id,
                        "prediction": token_text(tokenizer, prediction_id),
                        "correct": prediction_id == target_id,
                        "target_probability": float(
                            row_probabilities[index, target_id].item()
                        ),
                    }
                )
            original_ids = originals[global_row].tolist()
            cases.append(
                {
                    "case_index": global_row,
                    "case_id": hashlib.sha256(
                        originals[global_row].numpy().tobytes()
                    ).hexdigest()[:16],
                    "tokens": [token_text(tokenizer, token) for token in original_ids],
                    "masked_tokens": masked_rows,
                }
            )
            offset += count

    metrics: dict[str, float | int] = {
        "masked_accuracy": correct / masked,
        "masked_ce": loss_sum / masked,
        "masked_correct": correct,
        "masked_tokens": masked,
        "examples": int(inputs.shape[0]),
    }
    expected = EXPECTED_QUALITY[adapter.arm]
    expected_correct = round(expected["masked_accuracy"] * masked)
    if (
        abs(int(metrics["masked_correct"]) - expected_correct)
        > QUALITY_CORRECT_TOLERANCE
    ):
        raise RuntimeError(
            f"{adapter.arm} accuracy did not reproduce: {metrics['masked_accuracy']} "
            f"({metrics['masked_correct']}/{masked}, expected {expected_correct}/{masked})"
        )
    if (
        abs(float(metrics["masked_ce"]) - expected["masked_ce"])
        > QUALITY_CE_TOLERANCE
    ):
        raise RuntimeError(f"{adapter.arm} CE did not reproduce: {metrics['masked_ce']}")
    return metrics, cases


@torch.no_grad()
def future_dependency(
    adapter: InferenceAdapter,
    inputs: torch.Tensor,
    labels: torch.Tensor,
    replacement_id: int,
) -> float:
    rows = []
    positions = []
    for row in range(inputs.shape[0]):
        candidates = torch.nonzero(labels[row] != -100).flatten()
        candidates = candidates[candidates < inputs.shape[1] - 8]
        if len(candidates):
            rows.append(row)
            positions.append(int(candidates[0]))
        if len(rows) == 16:
            break
    original = inputs[rows].cuda()
    altered = original.clone()
    for local_row, position in enumerate(positions):
        altered[local_row, position + 1 :] = replacement_id
    with torch.autocast("cuda", dtype=torch.bfloat16):
        original_hidden = adapter.encode(original)
        altered_hidden = adapter.encode(altered)
        original_masked = torch.stack(
            [original_hidden[row, position] for row, position in enumerate(positions)]
        )
        altered_masked = torch.stack(
            [altered_hidden[row, position] for row, position in enumerate(positions)]
        )
        original_logits = adapter.masked_head(
            original_masked[:, None, :],
            torch.ones(
                (len(positions), 1), dtype=torch.bool, device=original_masked.device
            ),
        ).float()
        altered_logits = adapter.masked_head(
            altered_masked[:, None, :],
            torch.ones(
                (len(positions), 1), dtype=torch.bool, device=altered_masked.device
            ),
        ).float()
    dependency = float((original_logits - altered_logits).abs().mean().item())
    if adapter.arm == "causal_gdn2" and dependency != 0.0:
        raise RuntimeError(f"Causal GDN2 leaks future tokens: {dependency}")
    if adapter.arm != "causal_gdn2" and dependency <= 1e-7:
        raise RuntimeError(f"{adapter.arm} has no future dependency")
    return dependency


def benchmark_call(
    call: Callable[[], torch.Tensor],
    *,
    examples: int,
    tokens: int,
    warmup_steps: int,
    measured_steps: int,
) -> dict[str, float | int]:
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        for _ in range(warmup_steps):
            output = call()
            if not bool(torch.isfinite(output).all()):
                raise RuntimeError("Non-finite benchmark output")
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    baseline_allocated = torch.cuda.memory_allocated()
    baseline_reserved = torch.cuda.memory_reserved()
    started = time.perf_counter()
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        for _ in range(measured_steps):
            output = call()
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    peak_allocated = torch.cuda.max_memory_allocated()
    peak_reserved = torch.cuda.max_memory_reserved()
    return {
        "warmup_steps": warmup_steps,
        "measured_steps": measured_steps,
        "elapsed_seconds": elapsed,
        "latency_ms": elapsed * 1000.0 / measured_steps,
        "examples_per_second": examples * measured_steps / elapsed,
        "input_tokens_per_second": tokens * measured_steps / elapsed,
        "baseline_allocated_bytes": baseline_allocated,
        "baseline_reserved_bytes": baseline_reserved,
        "peak_allocated_bytes": peak_allocated,
        "peak_reserved_bytes": peak_reserved,
        "peak_increment_bytes": peak_allocated - baseline_allocated,
    }


def benchmark_suite(
    adapter: InferenceAdapter,
    inputs: torch.Tensor,
    labels: torch.Tensor,
    warmup_steps: int,
    measured_steps: int,
) -> dict[str, Any]:
    results: dict[str, Any] = {}
    for batch_size in (1, 64):
        batch_inputs = inputs[:batch_size].cuda()
        batch_labels = labels[:batch_size].cuda()
        mask = batch_labels != -100
        steps = measured_steps * (3 if batch_size == 1 else 1)

        def encoder_call() -> torch.Tensor:
            return adapter.encode(batch_inputs)

        def end_to_end_call() -> torch.Tensor:
            hidden = adapter.encode(batch_inputs)
            return adapter.masked_head(hidden, mask)

        for workload, call in (
            ("encoder", encoder_call),
            ("masked_recovery", end_to_end_call),
        ):
            key = f"batch{batch_size}_{workload}"
            results[key] = benchmark_call(
                call,
                examples=batch_size,
                tokens=batch_inputs.numel(),
                warmup_steps=warmup_steps,
                measured_steps=steps,
            )
    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--arm",
        choices=("bidirectional_bert", "causal_gdn2", "future_seed_gdn2"),
        required=True,
    )
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--bert-checkpoint", type=Path, required=True)
    parser.add_argument("--gdn-checkpoint", type=Path)
    parser.add_argument("--gdn-checkpoint-sha256")
    parser.add_argument("--validation-json", type=Path, required=True)
    parser.add_argument("--fla-wheel", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repetition", type=int, required=True)
    parser.add_argument("--warmup-steps", type=int, default=30)
    parser.add_argument("--measured-steps", type=int, default=200)
    parser.add_argument("--quality", action="store_true")
    args = parser.parse_args()

    if args.repetition < 0:
        raise RuntimeError("Repetition must be nonnegative")
    if args.warmup_steps < 1 or args.measured_steps < 1:
        raise RuntimeError("Benchmark step counts must be positive")
    gpu = visible_gpu()
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_dir, local_files_only=True, use_fast=True
    )
    if tokenizer.mask_token_id != 103 or len(tokenizer) != 30_522:
        raise RuntimeError("Registered WordPiece tokenizer drifted")
    originals, inputs, labels = prepare_validation(args.validation_json, tokenizer)
    adapter, model_provenance = load_arm(
        args.arm,
        args.model_dir,
        args.bert_checkpoint,
        args.gdn_checkpoint,
        args.gdn_checkpoint_sha256,
        args.fla_wheel,
    )
    dependency = future_dependency(adapter, inputs, labels, tokenizer.unk_token_id)
    if args.arm != "bidirectional_bert":
        active_diagnostics = futureseed_diagnostics(adapter.model)
        expected_routes = 0 if args.arm == "causal_gdn2" else 3
        if active_diagnostics["active_seed_routes"] != expected_routes:
            raise RuntimeError(
                f"{args.arm} active FutureSeed routes drifted: {active_diagnostics}"
            )
        model_provenance["future_seed_after_forward"] = active_diagnostics
    metrics = None
    cases = None
    if args.quality:
        metrics, cases = quality_metrics(
            adapter, originals, inputs, labels, tokenizer
        )
    benchmarks = benchmark_suite(
        adapter,
        inputs,
        labels,
        args.warmup_steps,
        args.measured_steps,
    )
    result = {
        "plan": "P-CAUSAL-023",
        "arm": args.arm,
        "repetition": args.repetition,
        "gpu": gpu,
        "protocol": {
            "sequence_length": 128,
            "validation_windows": 256,
            "masked_tokens": 4_742,
            "validation_tensor_sha256": EXPECTED_VALIDATION_TENSOR_SHA256,
            "warmup_steps": args.warmup_steps,
            "measured_steps_batch64": args.measured_steps,
            "measured_steps_batch1": args.measured_steps * 3,
            "fresh_process": True,
            "quality_evaluated": args.quality,
        },
        "model": model_provenance,
        "future_dependency_mean_abs": dependency,
        "quality": metrics,
        "benchmarks": benchmarks,
        "cases": cases,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({key: value for key, value in result.items() if key != "cases"}))


if __name__ == "__main__":
    main()
