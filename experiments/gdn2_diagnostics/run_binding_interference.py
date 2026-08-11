#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import hashlib
import html
import importlib
import inspect
import json
import math
import os
import random
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Iterable


PERSIST_ROOT = Path(os.environ.get("PERSIST_ROOT", "/huyang2/double-loop")).resolve()
for cache_name in (
    "XDG_CACHE_HOME",
    "TRITON_CACHE_DIR",
    "TORCHINDUCTOR_CACHE_DIR",
    "TORCH_EXTENSIONS_DIR",
    "TMPDIR",
):
    cache_path = Path(os.environ.get(cache_name, "/invalid")).resolve()
    if PERSIST_ROOT not in (cache_path, *cache_path.parents):
        raise RuntimeError(f"{cache_name} must stay below {PERSIST_ROOT}")
if os.environ.get("CUDA_VISIBLE_DEVICES") != "0":
    raise RuntimeError("P-BIND-001 requires CUDA_VISIBLE_DEVICES=0")
if os.environ.get("FLA_DISABLE_BACKEND_DISPATCH") != "1":
    raise RuntimeError("FLA_DISABLE_BACKEND_DISPATCH=1 is required")
if os.environ.get("FLA_CONV_BACKEND") != "triton":
    raise RuntimeError("FLA_CONV_BACKEND=triton is required")
if os.environ.get("FLA_STRICT_OFFICIAL") != "1":
    raise RuntimeError("FLA_STRICT_OFFICIAL=1 is required")

import torch
import torch.nn.functional as F
from zoology.data.utils import prepare_data
from zoology.train import Trainer
from zoology.utils import set_determinism

from experiments.gdn2_diagnostics.committed_edit import (
    CaptureContext,
    ChunkGDN2Recorder,
    binary_auroc,
    pearson_binary,
    summarize_geometry,
    top_quantile_lift,
)
from experiments.zoology_mqar.futureseed_directionality import (
    CaptureLogger,
    dataset_hash,
    model_hash,
)
from experiments.zoology_mqar.length_scaling import (
    _query_event,
    build_config,
    evaluate,
    make_model,
    parameter_hash,
    warm_model,
)
from experiments.rwkv_fs_sudoku import study_rwkv_futureseed_loop as sudoku


PLAN_ID = "P-BIND-001"
PINNED_FLA_SHA = "9c8e42e762fce087c27b673af4922795d9edb85e"
SUDOKU_PARENT_SHA256 = (
    "6339c3cb2b5fc5230a581d6633716483e35ff8e4522f06a9d7aaf26512f023da"
)
SUDOKU_PARENT_SOURCE_SHA = "9f2ee8d1738032bc5f09b55db0b81d507780b376"
MQAR_SOURCE_SHA = "77e5539fc0ef74231cab658bd610b24254fdcfa7"
MQAR_FROZEN = {
    "init_hash": "5595ecb17132326ba41be9f872d3abc81c31cd815832c60aa33f878273ba28ae",
    "init_parameter_hash": "3b0c133410eaed1135224cc0acb094705655cc7e97ea6beeba17b40beab1a368",
    "train_hash": "647c64ece84984a23dfd817c4f277ea83840dbec57cc18bb6c9bf9eda7cc9a68",
    "test_hash": "4a8237ba8fe19aaff0d1d72de7b7f6505eaab59cd091442c2f463df34cce278f",
    "balanced_accuracy": 0.7475,
    "joint_exact": 0.339,
    "future_accuracy": 0.7415,
    "past_accuracy": 0.7535,
}
RANGES = (
    ("46-50", 46, 50),
    ("51-55", 51, 55),
    ("56-60", 56, 60),
    ("61-64", 61, 64),
)
BRANCH_GATES = {
    "surprise": {
        "corrected_auroc_min": 0.65,
        "corrected_top_quartile_lift_min": 1.50,
        "recency_minus_surprise_survival_min": 0.10,
    },
    "gram": {
        "effective_rank_fraction_median_max": 0.50,
        "anisotropy_median_min": 4.0,
        "affected_record_fraction_min": 0.75,
    },
    "clustered_delta": {
        "long_minus_short_survival_max": -0.20,
        "surprise_top16_concentration_max": 0.40,
    },
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def state_hash(state: dict[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(state.items()):
        digest.update(name.encode())
        raw = value.detach().cpu().contiguous().view(torch.uint8)
        digest.update(raw.numpy().tobytes())
    return digest.hexdigest()


def git_value(repo: Path, *arguments: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(repo), *arguments],
        text=True,
    ).strip()


def gpu_uuid() -> str:
    rows = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=index,uuid",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    ).strip().splitlines()
    if len(rows) != 1:
        raise RuntimeError(f"expected one visible GPU, got {rows}")
    index, uuid = [part.strip() for part in rows[0].split(",", 1)]
    if index != "0":
        raise RuntimeError(f"expected CUDA index0, got {index}")
    return uuid


def fla_provenance() -> dict[str, Any]:
    from fla.layers.gdn2 import GatedDeltaNet2
    from fla.ops.gdn2 import chunk_gdn2

    layer_path = Path(inspect.getfile(GatedDeltaNet2)).resolve()
    operation = inspect.unwrap(chunk_gdn2)
    operation_path = Path(inspect.getfile(operation)).resolve()
    for path in (layer_path, operation_path):
        if PINNED_FLA_SHA not in str(path):
            raise RuntimeError(f"unpinned official FLA source: {path}")
    return {
        "expected_sha": PINNED_FLA_SHA,
        "layer_source": str(layer_path),
        "operation_source": str(operation_path),
        "operation_module": chunk_gdn2.__module__,
        "operation_unwrapped_module": operation.__module__,
    }


def construct_sudoku_model(
    checkpoint: Path,
    *,
    device: torch.device,
) -> tuple[torch.nn.Module, dict[str, Any]]:
    if sha256_file(checkpoint) != SUDOKU_PARENT_SHA256:
        raise RuntimeError("Sudoku parent checkpoint SHA256 mismatch")
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if int(payload.get("saved_at_step", -1)) != 3000:
        raise RuntimeError("Sudoku parent is not exact step3000")
    config = dict(payload["args"])
    signature = inspect.signature(sudoku.FutureSeedLoopSudoku)
    kwargs: dict[str, Any] = {}
    for name, parameter in signature.parameters.items():
        if name == "self":
            continue
        if name in config:
            kwargs[name] = config[name]
        elif parameter.default is inspect.Parameter.empty:
            raise RuntimeError(f"checkpoint lacks required model field {name}")
    model = sudoku.FutureSeedLoopSudoku(**kwargs)
    model.load_state_dict(payload["model"], strict=True)
    model = model.to(device).eval()
    metadata = {
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": SUDOKU_PARENT_SHA256,
        "parent_source_sha": SUDOKU_PARENT_SOURCE_SHA,
        "saved_at_step": int(payload["saved_at_step"]),
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "config": config,
    }
    del payload
    return model, metadata


def sudoku_context(config: dict[str, Any]):
    layers = int(config["layers"])
    streams = int(config["l_cycles"]) + 1
    calls_per_loop = layers * streams

    def context(call_index: int) -> CaptureContext:
        within_loop = call_index % calls_per_loop
        stream = within_loop // layers
        layer = within_loop % layers
        loop = call_index // calls_per_loop
        return CaptureContext(
            selected=stream == streams - 1,
            metadata={"loop": loop, "layer": layer, "stream": "h"},
        )

    return context, calls_per_loop


@torch.no_grad()
def sudoku_forward(
    model: torch.nn.Module,
    inputs: torch.Tensor,
    *,
    loops: int,
    forward_dtype: str,
) -> list[torch.Tensor]:
    with sudoku.forward_autocast(forward_dtype, inputs.device):
        logits, _trace = model.forward_trace(
            inputs,
            loops=loops,
            noise_scale=0.0,
            feature_buffer=None,
            update_feature_buffer=False,
            cell_order=None,
        )
    return logits


def prediction_summary(
    predictions: list[torch.Tensor],
    labels: torch.Tensor,
    clue_mask: torch.Tensor,
) -> dict[str, Any]:
    blank_mask = ~clue_mask
    rows = []
    for loop, prediction in enumerate(predictions, start=1):
        correct = prediction.eq(labels)
        rows.append(
            {
                "loop": loop,
                "exact": float(correct.all(dim=1).float().mean().item()),
                "blank_accuracy": float(correct[blank_mask].float().mean().item()),
                "wrong_cells_mean": float((~correct).sum(dim=1).float().mean().item()),
            }
        )
    return {
        "loops": rows,
        "wrong_cells_loop1_to_last": rows[0]["wrong_cells_mean"] - rows[-1]["wrong_cells_mean"],
    }


def sudoku_identity_contract(
    *,
    checkpoint: Path,
    data_dir: Path,
    device: torch.device,
) -> dict[str, Any]:
    model, metadata = construct_sudoku_model(checkpoint, device=device)
    config = metadata["config"]
    dataset = sudoku.OfficialSudokuDataset(data_dir, config["official_sudoku_eval_split"])
    context, calls_per_loop = sudoku_context(config)
    batch = dataset.fixed_batch_by_blank_range(
        2,
        int(config["seed"]) + int(config["official_eval_seed_offset"]) + 17000 + 46 * 37 + 50,
        holes_min=46,
        holes_max=50,
        device=device,
    )
    initial_hash = state_hash(model.state_dict())
    baseline = sudoku_forward(
        model,
        batch[0],
        loops=int(config["max_loops"]),
        forward_dtype=str(config["forward_dtype"]),
    )
    original_operation = sudoku.chunk_gdn2
    if original_operation is None:
        raise RuntimeError("official Sudoku chunk_gdn2 is unavailable")
    recorder = ChunkGDN2Recorder(original_operation, context=context, top_k=16)
    sudoku.chunk_gdn2 = recorder
    try:
        recorder.reset(enabled=False)
        wrapped = sudoku_forward(
            model,
            batch[0],
            loops=int(config["max_loops"]),
            forward_dtype=str(config["forward_dtype"]),
        )
    finally:
        sudoku.chunk_gdn2 = original_operation
    expected_calls = int(config["max_loops"]) * calls_per_loop
    if recorder.call_index != expected_calls:
        raise RuntimeError(
            f"Sudoku identity expected {expected_calls} official calls, got {recorder.call_index}"
        )
    if not all(torch.equal(left, right) for left, right in zip(baseline, wrapped)):
        raise RuntimeError("read-only Sudoku wrapper changed logits")
    final_hash = state_hash(model.state_dict())
    if final_hash != initial_hash:
        raise RuntimeError("Sudoku identity contract changed model state")
    result = {
        "logits_bit_exact": True,
        "official_calls": expected_calls,
        "model_state_hash_before": initial_hash,
        "model_state_hash_after": final_hash,
        "parameters": metadata["parameters"],
        "parameter_delta": 0,
    }
    del model
    torch.cuda.empty_cache()
    return result


def _finite_stat(values: torch.Tensor, statistic: str) -> float:
    finite = values.float()[torch.isfinite(values.float())]
    if finite.numel() == 0:
        return float("nan")
    if statistic == "mean":
        return float(finite.mean().item())
    if statistic == "median":
        return float(finite.median().item())
    raise ValueError(statistic)


def analyze_sudoku_capture(
    records: list[dict[str, Any]],
    predictions: list[torch.Tensor],
    labels: torch.Tensor,
    clue_mask: torch.Tensor,
    *,
    loops: int,
    layers: int,
) -> tuple[dict[str, Any], dict[str, torch.Tensor]]:
    if len(records) != loops * layers:
        raise RuntimeError(f"expected {loops * layers} selected records, got {len(records)}")
    blank_mask = (~clue_mask).cpu()
    predictions_cpu = [prediction.detach().cpu() for prediction in predictions]
    labels_cpu = labels.detach().cpu()
    per_loop: list[dict[str, Any]] = []
    combined = {
        "scores": [],
        "corrected": [],
        "final_failure": [],
    }
    for loop in range(loops):
        loop_records = [record for record in records if int(record["loop"]) == loop]
        surprise = torch.stack([record["surprise"] for record in loop_records]).mean(dim=0)
        current_wrong = predictions_cpu[loop].ne(labels_cpu)
        final_failure = predictions_cpu[-1].ne(labels_cpu)
        if loop + 1 < loops:
            corrected = current_wrong & predictions_cpu[loop + 1].eq(labels_cpu)
        else:
            corrected = torch.zeros_like(current_wrong)
        selected_scores = surprise[blank_mask]
        selected_corrected = corrected[blank_mask]
        selected_failure = final_failure[blank_mask]
        row = {
            "loop": loop + 1,
            "geometry": summarize_geometry(loop_records),
            "surprise_mean": float(selected_scores.mean().item()),
            "surprise_top16_concentration": _finite_stat(
                torch.cat([record["surprise_topk_concentration"] for record in loop_records]),
                "mean",
            ),
            "surprise_survival": _finite_stat(
                torch.cat([record["surprise_survival"] for record in loop_records]),
                "mean",
            ),
            "recency_survival": _finite_stat(
                torch.cat([record["recency_survival"] for record in loop_records]),
                "mean",
            ),
            "corrected_next_count": int(selected_corrected.sum().item()),
            "corrected_next_auroc": binary_auroc(selected_scores, selected_corrected),
            "corrected_next_pearson": pearson_binary(selected_scores.log1p(), selected_corrected),
            "corrected_next_top_quartile_lift": top_quantile_lift(
                selected_scores, selected_corrected
            ),
            "final_failure_auroc": binary_auroc(selected_scores, selected_failure),
            "final_failure_pearson": pearson_binary(selected_scores.log1p(), selected_failure),
        }
        per_loop.append(row)
        if loop + 1 < loops:
            combined["scores"].append(selected_scores)
            combined["corrected"].append(selected_corrected)
            combined["final_failure"].append(selected_failure)
    tensors = {key: torch.cat(value) for key, value in combined.items()}
    summary = {
        "per_loop": per_loop,
        "corrected_auroc": binary_auroc(tensors["scores"], tensors["corrected"]),
        "corrected_pearson": pearson_binary(tensors["scores"].log1p(), tensors["corrected"]),
        "corrected_top_quartile_lift": top_quantile_lift(
            tensors["scores"], tensors["corrected"]
        ),
        "final_failure_auroc": binary_auroc(
            tensors["scores"], tensors["final_failure"]
        ),
        "final_failure_pearson": pearson_binary(
            tensors["scores"].log1p(), tensors["final_failure"]
        ),
        "recency_minus_surprise_survival": sum(
            row["recency_survival"] - row["surprise_survival"] for row in per_loop
        ) / len(per_loop),
    }
    return summary, tensors


def run_sudoku(
    *,
    checkpoint: Path,
    data_dir: Path,
    batch_size: int,
    device: torch.device,
) -> dict[str, Any]:
    model, metadata = construct_sudoku_model(checkpoint, device=device)
    config = metadata["config"]
    initial_state_hash = state_hash(model.state_dict())
    parameter_count = metadata["parameters"]
    dataset = sudoku.OfficialSudokuDataset(data_dir, config["official_sudoku_eval_split"])
    context, calls_per_loop = sudoku_context(config)
    original_operation = sudoku.chunk_gdn2
    if original_operation is None:
        raise RuntimeError("official Sudoku chunk_gdn2 is unavailable")
    recorder = ChunkGDN2Recorder(original_operation, context=context, top_k=16)

    identity_batch = dataset.fixed_batch_by_blank_range(
        2,
        int(config["seed"]) + int(config["official_eval_seed_offset"]) + 17000 + 46 * 37 + 50,
        holes_min=46,
        holes_max=50,
        device=device,
    )
    baseline_logits = sudoku_forward(
        model,
        identity_batch[0],
        loops=int(config["max_loops"]),
        forward_dtype=str(config["forward_dtype"]),
    )
    sudoku.chunk_gdn2 = recorder
    try:
        recorder.reset(enabled=False)
        wrapped_logits = sudoku_forward(
            model,
            identity_batch[0],
            loops=int(config["max_loops"]),
            forward_dtype=str(config["forward_dtype"]),
        )
        if recorder.call_index != int(config["max_loops"]) * calls_per_loop:
            raise RuntimeError("Sudoku wrapper did not observe the expected official calls")
        if not all(torch.equal(left, right) for left, right in zip(baseline_logits, wrapped_logits)):
            raise RuntimeError("read-only Sudoku wrapper changed logits")

        ranges: dict[str, Any] = {}
        hard_tensors = {"scores": [], "corrected": [], "final_failure": []}
        hard_records: list[dict[str, Any]] = []
        for offset, (label, lower, upper) in enumerate(RANGES):
            seed = (
                int(config["seed"])
                + int(config["official_eval_seed_offset"])
                + 17000
                + offset * 997
                + lower * 37
                + upper
            )
            batch = dataset.fixed_batch_by_blank_range(
                batch_size,
                seed,
                holes_min=lower,
                holes_max=upper,
                device=device,
            )
            recorder.reset(enabled=True)
            logits = sudoku_forward(
                model,
                batch[0],
                loops=int(config["max_loops"]),
                forward_dtype=str(config["forward_dtype"]),
            )
            predictions = [value.argmax(dim=-1) for value in logits]
            capture_summary, tensors = analyze_sudoku_capture(
                recorder.records,
                predictions,
                batch[1],
                batch[2],
                loops=int(config["max_loops"]),
                layers=int(config["layers"]),
            )
            ranges[label] = {
                "blank_range": [lower, upper],
                "batch_size": int(batch[0].shape[0]),
                "seed": seed,
                "predictions": prediction_summary(predictions, batch[1], batch[2]),
                "capture": capture_summary,
            }
            if lower >= 51:
                for key in hard_tensors:
                    hard_tensors[key].append(tensors[key])
                hard_records.extend(recorder.records)
        combined_hard = {key: torch.cat(value) for key, value in hard_tensors.items()}
        hard_loop_rows = [
            row
            for label, range_result in ranges.items()
            if int(range_result["blank_range"][0]) >= 51
            for row in range_result["capture"]["per_loop"]
        ]
        hard_summary = {
            "corrected_auroc": binary_auroc(
                combined_hard["scores"], combined_hard["corrected"]
            ),
            "corrected_pearson": pearson_binary(
                combined_hard["scores"].log1p(), combined_hard["corrected"]
            ),
            "corrected_top_quartile_lift": top_quantile_lift(
                combined_hard["scores"], combined_hard["corrected"]
            ),
            "final_failure_auroc": binary_auroc(
                combined_hard["scores"], combined_hard["final_failure"]
            ),
            "final_failure_pearson": pearson_binary(
                combined_hard["scores"].log1p(), combined_hard["final_failure"]
            ),
            "recency_minus_surprise_survival": sum(
                row["recency_survival"] - row["surprise_survival"]
                for row in hard_loop_rows
            ) / len(hard_loop_rows),
            "surprise_top16_concentration": sum(
                row["surprise_top16_concentration"] for row in hard_loop_rows
            ) / len(hard_loop_rows),
            "geometry": summarize_geometry(hard_records),
        }
    finally:
        sudoku.chunk_gdn2 = original_operation

    final_state_hash = state_hash(model.state_dict())
    if final_state_hash != initial_state_hash:
        raise RuntimeError("Sudoku diagnostic changed model state")
    if sum(parameter.numel() for parameter in model.parameters()) != parameter_count:
        raise RuntimeError("Sudoku diagnostic changed parameter count")
    del model
    torch.cuda.empty_cache()
    return {
        "model": metadata,
        "identity": {
            "logits_bit_exact": True,
            "model_state_hash_before": initial_state_hash,
            "model_state_hash_after": final_state_hash,
            "parameter_delta": 0,
        },
        "ranges": ranges,
        "hard_51_64": hard_summary,
    }


def assert_close_metric(actual: float, expected: float, name: str) -> None:
    if not math.isclose(float(actual), float(expected), rel_tol=0.0, abs_tol=1e-12):
        raise RuntimeError(f"MQAR frozen endpoint drifted for {name}: {actual} != {expected}")


def reconstruct_mqar(
    *,
    out_dir: Path,
) -> tuple[torch.nn.Module, Any, dict[str, Any], list[dict[str, Any]]]:
    config = build_config(
        arm="future_seed_gdn2",
        sequence_length=1024,
        num_kv_pairs=4,
        max_epochs=10,
        batch_size=32,
    )
    set_determinism(config.seed)
    model = make_model(config, "future_seed_gdn2")
    init_hash = model_hash(model)
    init_parameter_hash = parameter_hash(model)
    train_loader, test_loader = prepare_data(config.data)
    hashes = {
        "train": dataset_hash(train_loader),
        "test": dataset_hash(test_loader),
    }
    if init_hash != MQAR_FROZEN["init_hash"]:
        raise RuntimeError("MQAR initialization hash drifted")
    if init_parameter_hash != MQAR_FROZEN["init_parameter_hash"]:
        raise RuntimeError("MQAR parameter initialization hash drifted")
    if hashes["train"] != MQAR_FROZEN["train_hash"] or hashes["test"] != MQAR_FROZEN["test_hash"]:
        raise RuntimeError("MQAR data hash drifted")

    fixed_batch = next(iter(train_loader))
    warm_model(model, fixed_batch)
    set_determinism(config.seed)
    logger = CaptureLogger(out_dir / "mqar_training_metrics.jsonl", "future_seed_gdn2")
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
    torch.cuda.synchronize()
    started = time.perf_counter()
    trainer.fit()
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    metrics, cases = evaluate(model, test_loader, sequence_length=1024)
    logger.finish()
    assert_close_metric(metrics["balanced_accuracy"], MQAR_FROZEN["balanced_accuracy"], "balanced_accuracy")
    assert_close_metric(metrics["joint_exact"], MQAR_FROZEN["joint_exact"], "joint_exact")
    assert_close_metric(metrics["future"]["accuracy"], MQAR_FROZEN["future_accuracy"], "future_accuracy")
    assert_close_metric(metrics["past"]["accuracy"], MQAR_FROZEN["past_accuracy"], "past_accuracy")
    endpoint = {
        "source_sha": MQAR_SOURCE_SHA,
        "init_hash": init_hash,
        "init_parameter_hash": init_parameter_hash,
        "data_hashes": hashes,
        "metrics": metrics,
        "elapsed_sec": elapsed,
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "parameter_hash": parameter_hash(model),
    }
    checkpoint_path = out_dir / "mqar_l1024_frozen_reconstruction.pt"
    torch.save(
        {
            "model": {name: value.detach().cpu() for name, value in model.state_dict().items()},
            "config": config.model_dump(mode="json"),
            "endpoint": endpoint,
        },
        checkpoint_path,
    )
    endpoint["reconstructed_checkpoint"] = str(checkpoint_path)
    endpoint["reconstructed_checkpoint_sha256"] = sha256_file(checkpoint_path)
    return model, test_loader, endpoint, cases


@torch.no_grad()
def run_mqar_capture(
    model: torch.nn.Module,
    test_loader: Iterable[Any],
    *,
    max_examples: int,
) -> dict[str, Any]:
    gdn2_module = importlib.import_module("fla.layers.gdn2")
    original_operation = gdn2_module.chunk_gdn2
    recorder = ChunkGDN2Recorder(
        original_operation,
        context=lambda index: CaptureContext(True, {"layer": index % 2}),
        top_k=16,
    )
    first_batch = next(iter(test_loader))
    baseline = model(first_batch[0].cuda())
    gdn2_module.chunk_gdn2 = recorder
    all_records: list[dict[str, Any]] = []
    event_scores: list[float] = []
    event_errors: list[bool] = []
    wrong_key_labels: list[bool] = []
    incorrect_events = 0
    wrong_key_events = 0
    examples = 0
    try:
        recorder.reset(enabled=False)
        wrapped = model(first_batch[0].cuda())
        if not torch.equal(baseline, wrapped):
            raise RuntimeError("read-only MQAR wrapper changed logits")
        for inputs, targets, _slices in test_loader:
            remaining = max_examples - examples
            if remaining <= 0:
                break
            if int(inputs.shape[0]) > remaining:
                inputs = inputs[:remaining]
                targets = targets[:remaining]
            recorder.reset(enabled=True)
            logits = model(inputs.cuda())
            predictions = logits.argmax(dim=-1).cpu()
            if recorder.call_index != 2 or len(recorder.records) != 2:
                raise RuntimeError("MQAR wrapper did not capture both official GDN2 layers")
            all_records.extend(recorder.records)
            surprise = torch.stack([record["surprise"] for record in recorder.records]).mean(dim=0)
            quarter = inputs.shape[1] // 4
            for row in range(inputs.shape[0]):
                query_positions = torch.nonzero(targets[row] != -100).flatten()
                true_values = {int(targets[row, position].item()) for position in query_positions}
                for position in query_positions:
                    event = _query_event(
                        inputs[row],
                        targets[row],
                        predictions[row],
                        int(position.item()),
                        quarter,
                    )
                    error = not bool(event["correct"])
                    wrong_key = error and int(event["prediction"]) in true_values
                    event_scores.append(float(surprise[row, int(event["write_position"])].item()))
                    event_errors.append(error)
                    wrong_key_labels.append(wrong_key)
                    incorrect_events += int(error)
                    wrong_key_events += int(wrong_key)
            examples += int(inputs.shape[0])
    finally:
        gdn2_module.chunk_gdn2 = original_operation

    scores = torch.tensor(event_scores)
    errors = torch.tensor(event_errors)
    wrong_keys = torch.tensor(wrong_key_labels)
    long_survival: list[torch.Tensor] = []
    short_survival: list[torch.Tensor] = []
    concentrations = []
    for record in all_records:
        indices = record["surprise_indices"]
        survival = record["surprise_survival_by_slot"]
        horizon = 1023 - indices
        if bool((horizon >= 512).any()):
            long_survival.append(survival[horizon >= 512])
        if bool((horizon < 128).any()):
            short_survival.append(survival[horizon < 128])
        concentrations.append(record["surprise_topk_concentration"])
    long_mean = _finite_stat(torch.cat(long_survival), "mean") if long_survival else float("nan")
    short_mean = _finite_stat(torch.cat(short_survival), "mean") if short_survival else float("nan")
    geometry = summarize_geometry(all_records)
    effective_rank_values = torch.cat(
        [record["geometry"]["effective_rank_fraction"].reshape(-1) for record in all_records]
    )
    anisotropy_values = torch.cat(
        [record["geometry"]["anisotropy"].reshape(-1) for record in all_records]
    )
    affected = (effective_rank_values <= 0.50) | (anisotropy_values >= 4.0)
    return {
        "examples": examples,
        "events": len(event_scores),
        "wrong_key_valid_value": {
            "incorrect_events": incorrect_events,
            "wrong_key_events": wrong_key_events,
            "fraction_of_incorrect": wrong_key_events / max(incorrect_events, 1),
        },
        "surprise_vs_error": {
            "auroc": binary_auroc(scores, errors),
            "pearson": pearson_binary(scores.log1p(), errors),
            "top_quartile_lift": top_quantile_lift(scores, errors),
        },
        "surprise_vs_wrong_key": {
            "auroc": binary_auroc(scores, wrong_keys),
            "pearson": pearson_binary(scores.log1p(), wrong_keys),
            "top_quartile_lift": top_quantile_lift(scores, wrong_keys),
        },
        "geometry": geometry,
        "anisotropic_record_fraction": float(affected.float().mean().item()),
        "surprise_top16_concentration": _finite_stat(
            torch.cat(concentrations), "mean"
        ),
        "surprise_survival": _finite_stat(
            torch.cat([record["surprise_survival"] for record in all_records]), "mean"
        ),
        "recency_survival": _finite_stat(
            torch.cat([record["recency_survival"] for record in all_records]), "mean"
        ),
        "long_horizon_survival": long_mean,
        "short_horizon_survival": short_mean,
        "long_minus_short_survival": long_mean - short_mean,
    }


def choose_branch(result: dict[str, Any]) -> dict[str, Any]:
    sudoku_hard = result["sudoku"]["hard_51_64"]
    mqar = result["mqar"]["diagnostic"]
    surprise_checks = {
        "sudoku_corrected_auroc": sudoku_hard["corrected_auroc"]
        >= BRANCH_GATES["surprise"]["corrected_auroc_min"],
        "sudoku_corrected_top_quartile_lift": sudoku_hard[
            "corrected_top_quartile_lift"
        ]
        >= BRANCH_GATES["surprise"]["corrected_top_quartile_lift_min"],
        "overwritten_relative_to_recency": max(
            sudoku_hard["recency_minus_surprise_survival"],
            mqar["recency_survival"] - mqar["surprise_survival"],
        )
        >= BRANCH_GATES["surprise"]["recency_minus_surprise_survival_min"],
    }
    if all(surprise_checks.values()):
        return {
            "branch": "receiver_native_surprise_cache",
            "checks": surprise_checks,
            "next_experiment": (
                "Directional MQAR L1024 from-scratch K16 receiver-native evidence cache, "
                "surprise admission versus matched recency."
            ),
        }

    gram_checks = {
        "rank_or_anisotropy": (
            mqar["geometry"]["effective_rank_fraction_median"]
            <= BRANCH_GATES["gram"]["effective_rank_fraction_median_max"]
            or mqar["geometry"]["anisotropy_median"]
            >= BRANCH_GATES["gram"]["anisotropy_median_min"]
        ),
        "affected_fraction": mqar["anisotropic_record_fraction"]
        >= BRANCH_GATES["gram"]["affected_record_fraction_min"],
    }
    if all(gram_checks.values()):
        return {
            "branch": "bounded_native_pgdn",
            "checks": gram_checks,
            "next_experiment": (
                "Directional MQAR L1024 from-scratch bounded native key conditioner; "
                "no tied pre-scan wrapper."
            ),
        }

    cluster_checks = {
        "long_horizon_survival_drop": mqar["long_minus_short_survival"]
        <= BRANCH_GATES["clustered_delta"]["long_minus_short_survival_max"],
        "surprise_is_diffuse": mqar["surprise_top16_concentration"]
        <= BRANCH_GATES["clustered_delta"]["surprise_top16_concentration_max"],
    }
    if all(cluster_checks.values()):
        return {
            "branch": "clustered_delta_memory",
            "checks": cluster_checks,
            "next_experiment": (
                "Clean-room fused block-contiguous clustered delta memory on directional "
                "MQAR L1024; no third-party source copy."
            ),
        }
    return {
        "branch": "loop_dynamics_training_signal",
        "checks": {
            "surprise": surprise_checks,
            "gram": gram_checks,
            "clustered_delta": cluster_checks,
        },
        "next_experiment": (
            "Stop cache/capacity interventions and test a general loop-dynamics or "
            "training-signal mechanism."
        ),
    }


def graph_function_names(tensor: torch.Tensor) -> set[str]:
    names: set[str] = set()
    pending = [tensor.grad_fn]
    while pending:
        function = pending.pop()
        if function is None:
            continue
        name = type(function).__name__
        if name in names:
            continue
        names.add(name)
        pending.extend(item[0] for item in function.next_functions)
    return names


def synthetic_backward_contract(device: torch.device) -> dict[str, Any]:
    from fla.ops.gdn2 import chunk_gdn2

    generator = torch.Generator(device=device).manual_seed(52001)
    shapes = (1, 81, 2, 32)
    q = torch.randn(*shapes, device=device, dtype=torch.bfloat16, generator=generator, requires_grad=True)
    k = torch.randn(*shapes, device=device, dtype=torch.bfloat16, generator=generator, requires_grad=True)
    v = torch.randn(*shapes, device=device, dtype=torch.bfloat16, generator=generator, requires_grad=True)
    g = (-F.softplus(torch.randn(
        *shapes, device=device, dtype=torch.bfloat16, generator=generator
    ))).requires_grad_()
    b = torch.sigmoid(torch.randn(
        *shapes, device=device, dtype=torch.bfloat16, generator=generator
    )).requires_grad_()
    w = torch.sigmoid(torch.randn(
        *shapes, device=device, dtype=torch.bfloat16, generator=generator
    )).requires_grad_()
    output, final_state = chunk_gdn2(
        q=q,
        k=k,
        v=v,
        g=g,
        b=b,
        w=w,
        output_final_state=True,
        use_qk_l2norm_in_kernel=True,
    )
    names = graph_function_names(output)
    loss = output.float().square().mean() + final_state.float().square().mean()
    loss.backward()
    gradients = {}
    for name, value in (("q", q), ("k", k), ("v", v), ("g", g), ("b", b), ("w", w)):
        if value.grad is None or not torch.isfinite(value.grad).all() or value.grad.abs().max() == 0:
            raise RuntimeError(f"official GDN2 gradient contract failed for {name}")
        gradients[name] = float(value.grad.float().abs().max().item())
    if "ChunkGDN2FunctionBackward" not in names:
        raise RuntimeError(f"official backward function missing: {sorted(names)}")
    return {
        "backward_function": "ChunkGDN2FunctionBackward",
        "graph_functions": sorted(names),
        "gradient_max_abs": gradients,
    }


def render_markdown(result: dict[str, Any]) -> str:
    hard = result["sudoku"]["hard_51_64"]
    mqar = result["mqar"]["diagnostic"]
    decision = result["decision"]
    return f"""# {PLAN_ID}: Binding interference diagnostic

- Decision: `{decision['branch']}`
- Sudoku hard corrected AUROC/lift: `{hard['corrected_auroc']:.4f}` / `{hard['corrected_top_quartile_lift']:.4f}`
- Sudoku recency-minus-surprise survival: `{hard['recency_minus_surprise_survival']:.4f}`
- MQAR balanced/joint exact: `{result['mqar']['endpoint']['metrics']['balanced_accuracy']:.4f}` / `{result['mqar']['endpoint']['metrics']['joint_exact']:.4f}`
- MQAR wrong-key valid-value fraction: `{mqar['wrong_key_valid_value']['fraction_of_incorrect']:.4f}`
- MQAR key effective-rank fraction / anisotropy: `{mqar['geometry']['effective_rank_fraction_median']:.4f}` / `{mqar['geometry']['anisotropy_median']:.4f}`
- MQAR surprise concentration and long-minus-short survival: `{mqar['surprise_top16_concentration']:.4f}` / `{mqar['long_minus_short_survival']:.4f}`
- Next experiment: {decision['next_experiment']}

The diagnostic wrapped the pinned official operation transparently, replayed the
actual committed update in FP32, and made no parameter, optimizer, data-order, or
logit change. The reconstructed directional MQAR endpoint was accepted only after
matching the frozen initialization, data hashes, balanced accuracy, joint exact,
and directional accuracies exactly.
"""


def render_html(result: dict[str, Any]) -> str:
    rows = []
    for label, item in result["sudoku"]["ranges"].items():
        last = item["predictions"]["loops"][-1]
        rows.append(
            "<tr>"
            f"<td>{html.escape(label)}</td>"
            f"<td>{last['exact']:.4f}</td>"
            f"<td>{last['blank_accuracy']:.4f}</td>"
            f"<td>{item['capture']['corrected_auroc']:.4f}</td>"
            f"<td>{item['capture']['recency_minus_surprise_survival']:.4f}</td>"
            "</tr>"
        )
    decision = result["decision"]
    return f"""<!doctype html><html><head><meta charset="utf-8"><title>{PLAN_ID}</title>
<style>body{{font:15px/1.5 system-ui,sans-serif;max-width:980px;margin:36px auto;padding:0 20px;color:#182026}}
table{{border-collapse:collapse;width:100%}}th,td{{border-bottom:1px solid #d9dee2;padding:9px;text-align:right}}th:first-child,td:first-child{{text-align:left}}
.band{{border:1px solid #d9dee2;padding:16px;margin:18px 0;background:#f7f9fa}}code{{background:#edf1f3;padding:2px 5px}}</style></head>
<body><h1>GDN2 binding-interference diagnostic</h1><div class="band">Selected branch: <code>{html.escape(decision['branch'])}</code><br>{html.escape(decision['next_experiment'])}</div>
<h2>Sudoku fixed ranges</h2><table><thead><tr><th>blanks</th><th>loop5 exact</th><th>blank acc</th><th>correction AUROC</th><th>recency-surprise survival</th></tr></thead><tbody>{''.join(rows)}</tbody></table>
<h2>Directional MQAR L1024</h2><pre>{html.escape(json.dumps(result['mqar']['diagnostic'], indent=2))}</pre></body></html>"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("contract", "full"), required=True)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--source-sha", required=True)
    parser.add_argument("--expected-gpu-uuid", required=True)
    parser.add_argument("--sudoku-checkpoint", type=Path, required=True)
    parser.add_argument("--sudoku-data-dir", type=Path, required=True)
    parser.add_argument("--sudoku-batch-size", type=int, default=32)
    parser.add_argument("--mqar-diagnostic-examples", type=int, default=128)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()

    args.repo = args.repo.resolve()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    if git_value(args.repo, "rev-parse", "HEAD") != args.source_sha:
        raise RuntimeError("source HEAD differs from registered SHA")
    if git_value(args.repo, "status", "--porcelain"):
        raise RuntimeError("formal source worktree is dirty")
    actual_uuid = gpu_uuid()
    if actual_uuid != args.expected_gpu_uuid:
        raise RuntimeError(f"GPU UUID drift: {actual_uuid} != {args.expected_gpu_uuid}")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("exactly one CUDA device is required")
    device = torch.device("cuda:0")
    provenance = fla_provenance()
    backward = synthetic_backward_contract(device)
    contract = {
        "plan_id": PLAN_ID,
        "source_sha": args.source_sha,
        "source_clean": True,
        "cuda_visible_devices": os.environ["CUDA_VISIBLE_DEVICES"],
        "gpu_uuid": actual_uuid,
        "gpu_name": torch.cuda.get_device_name(device),
        "pinned_fla": provenance,
        "official_backward": backward,
        "parameter_delta": 0,
        "logit_intervention": "none",
        "branch_gates": BRANCH_GATES,
    }
    contract["sudoku_wrapper_identity"] = sudoku_identity_contract(
        checkpoint=args.sudoku_checkpoint.resolve(),
        data_dir=args.sudoku_data_dir.resolve(),
        device=device,
    )
    contract_path = args.out_dir / "cuda_contract.json"
    contract_path.write_text(json.dumps(contract, indent=2, sort_keys=True) + "\n")
    if args.mode == "contract":
        print(json.dumps({"status": "passed", "contract": str(contract_path)}, indent=2))
        return

    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    sudoku_result = run_sudoku(
        checkpoint=args.sudoku_checkpoint.resolve(),
        data_dir=args.sudoku_data_dir.resolve(),
        batch_size=args.sudoku_batch_size,
        device=device,
    )
    mqar_dir = args.out_dir / "mqar_reconstruction"
    mqar_dir.mkdir(parents=True, exist_ok=True)
    mqar_model, mqar_test_loader, mqar_endpoint, _cases = reconstruct_mqar(out_dir=mqar_dir)
    mqar_diagnostic = run_mqar_capture(
        mqar_model,
        mqar_test_loader,
        max_examples=args.mqar_diagnostic_examples,
    )
    del mqar_model
    torch.cuda.empty_cache()
    elapsed = time.perf_counter() - started
    result = {
        "plan_id": PLAN_ID,
        "generated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "source_sha": args.source_sha,
        "contract": contract,
        "method": {
            "new_parameters": 0,
            "logits_changed": False,
            "sudoku_batch_per_range": args.sudoku_batch_size,
            "mqar_diagnostic_examples": args.mqar_diagnostic_examples,
            "surprise": "Frobenius norm of the exact committed rank-1 edit",
            "top_k": 16,
        },
        "sudoku": sudoku_result,
        "mqar": {"endpoint": mqar_endpoint, "diagnostic": mqar_diagnostic},
        "elapsed_sec": elapsed,
        "peak_allocated_mb": torch.cuda.max_memory_allocated(device) / (1024**2),
    }
    result["decision"] = choose_branch(result)
    result_path = args.out_dir / "diagnostic.json"
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    (args.out_dir / "summary.md").write_text(render_markdown(result))
    (args.out_dir / "index.html").write_text(render_html(result))
    hashes = {
        path.name: sha256_file(path)
        for path in (contract_path, result_path, args.out_dir / "summary.md", args.out_dir / "index.html")
    }
    (args.out_dir / "hashes.json").write_text(json.dumps(hashes, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "status": "passed",
        "out_dir": str(args.out_dir),
        "decision": result["decision"],
        "elapsed_sec": elapsed,
        "peak_allocated_mb": result["peak_allocated_mb"],
        "hashes": hashes,
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
