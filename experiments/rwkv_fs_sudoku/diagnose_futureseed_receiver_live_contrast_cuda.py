#!/usr/bin/env python3
"""Causal admission diagnostic for receiver-native FutureSeed recommit.

The endpoint has two disjoint phases over official *training* rows that were
not seen by step 3000:

1. Discovery runs the unchanged parent once per microbatch and obtains one
   global receiver-edge/head/value descent direction from virtual gradients.
2. Holdout evaluates that fixed direction at a fixed one-percent committed
   edit budget, together with its sign reversal and two sham signals.

No model parameter, optimizer state, checkpoint, or official test row is
created or modified by this program.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import time
import zipfile
from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable, Iterable, Optional

import diagnose_loop_gradient_conflict_cuda as base
import numpy as np
import torch
import torch.nn.functional as F

import study_rwkv_futureseed_loop as study
from experiments.gdn2_diagnostics.receiver_live_contrast import (
    aggregate_discovery_direction,
    deterministic_board_permutation,
    paired_bootstrap_comparison,
    per_board_virtual_alpha_gradient,
    replay_gdn2_fp32,
    scaled_holdout_delta_v,
)


PLAN_ID = "P-FS2-006"
SCHEMA_VERSION = 1
EXPECTED_LAYERS = 12
EXPECTED_LOOPS = 5
EXPECTED_DEPTH_CALLS = 3
EXPECTED_OFFICIAL_BACKWARDS = EXPECTED_LAYERS * EXPECTED_DEPTH_CALLS * EXPECTED_LOOPS
EXPECTED_RECEIVER_CALLS = (EXPECTED_LAYERS - 1) * EXPECTED_DEPTH_CALLS * EXPECTED_LOOPS
DEFAULT_DISCOVERY_PER_RANGE = 128
DEFAULT_HOLDOUT_PER_RANGE = 128
MIN_ROWS_PER_RANGE = 128
MICROBATCH = 64
EPSILON = 0.01
QUANTIZED_BUDGET_REL_ERROR_MAX = 0.10
RHO = 0.25
BOOTSTRAP_SAMPLES = 10_000
REPLAY_REL_RMS_MAX = 2e-3
EXPECTED_GPU_NAME = "NVIDIA A100-SXM4-80GB"
EXPECTED_GPU_MEMORY_MIB = 81920
EXPECTED_FLA_WHEEL_SHA256 = (
    "0280db310981915eb048ece99d7bedca8b5caa9be65c99835a0f912ada977d6a"
)
EXPECTED_FLA_TREE_SHA256 = (
    "dd792d8b5ff490996df04532c9588a69844b1430124dcdf59e47e42a8e45d3d8"
)
EXPECTED_INPUTS_SHA256 = (
    "979f0f27411dfde97b725082cd29da03bea7c3b1c6b2bd2b15bfe95b76549cbc"
)
EXPECTED_LABELS_SHA256 = (
    "84e8a40dbb12d032eb9eba601c378d8388ea949430e07c00e80bc0947a59e48a"
)
REGISTERED_RANGES = ((51, 55), (56, 60), (61, 64))
ARM_NAMES = (
    "baseline",
    "plus_full",
    "minus_full",
    "plus_frozen",
    "plus_shuffled",
)
INTERVENTION_ARMS = ARM_NAMES[1:]


def parameter_sha256(model: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, parameter in model.named_parameters():
        value = parameter.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(str(tuple(value.shape)).encode("ascii"))
        digest.update(value.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def sha256_int64(values: np.ndarray) -> str:
    packed = np.asarray(values, dtype="<i8", order="C")
    return hashlib.sha256(packed.tobytes(order="C")).hexdigest()


def sha256_json(value: Any) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def verify_pinned_fla_wheel_tree(persistent_root: Path) -> dict[str, Any]:
    """Bind every imported FLA file to the pinned official wheel."""

    import fla

    wheel = (
        persistent_root
        / "wheelhouse"
        / "flash_linear_attention-0.5.2-9c8e42e-py3-none-any.whl"
    ).resolve()
    if base.sha256_file(wheel) != EXPECTED_FLA_WHEEL_SHA256:
        raise RuntimeError("pinned FLA wheel SHA mismatch")
    package_root = Path(fla.__file__).resolve().parent
    expected_root = Path(os.environ["FLA_PINNED_ROOT"]).resolve() / "fla"
    if package_root != expected_root:
        raise RuntimeError(f"FLA resolved from {package_root}, expected {expected_root}")
    if [Path(path).resolve() for path in fla.__path__] != [package_root]:
        raise RuntimeError("FLA namespace contains an unexpected package path")

    tree = hashlib.sha256()
    with zipfile.ZipFile(wheel) as archive:
        archive_files = sorted(
            name
            for name in archive.namelist()
            if name.startswith("fla/") and not name.endswith("/")
        )
        for name in archive_files:
            installed = package_root.parent / name
            if not installed.is_file():
                raise RuntimeError(f"pinned FLA file is missing: {name}")
            expected = hashlib.sha256(archive.read(name)).hexdigest()
            actual = base.sha256_file(installed)
            if actual != expected:
                raise RuntimeError(f"pinned FLA file drift: {name}")
            tree.update(name.encode("utf-8"))
            tree.update(b"\0")
            tree.update(bytes.fromhex(actual))
        installed_python = {
            str(path.relative_to(package_root.parent))
            for path in package_root.rglob("*.py")
        }
        wheel_python = {name for name in archive_files if name.endswith(".py")}
        if installed_python != wheel_python:
            raise RuntimeError("pinned FLA Python file set differs from official wheel")
    tree_sha = tree.hexdigest()
    if tree_sha != EXPECTED_FLA_TREE_SHA256:
        raise RuntimeError("pinned FLA installed-tree SHA mismatch")
    return {
        "wheel": str(wheel),
        "wheel_sha256": EXPECTED_FLA_WHEEL_SHA256,
        "package_root": str(package_root),
        "verified_files": len(archive_files),
        "installed_tree_sha256": tree_sha,
    }


def rms_relative_error(actual: torch.Tensor, expected: torch.Tensor) -> float:
    actual_f = actual.detach().float()
    expected_f = expected.detach().float()
    numerator = (actual_f - expected_f).square().mean().sqrt()
    denominator = expected_f.square().mean().sqrt().clamp_min(1e-8)
    return float((numerator / denominator).cpu().item())


def argument(
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    name: str,
    index: int,
    default: Any = None,
) -> Any:
    if name in kwargs:
        return kwargs[name]
    if index < len(args):
        return args[index]
    return default


def replace_argument(
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    name: str,
    index: int,
    value: Any,
) -> tuple[tuple[Any, ...], dict[str, Any]]:
    if name in kwargs:
        updated_kwargs = dict(kwargs)
        updated_kwargs[name] = value
        return args, updated_kwargs
    if index >= len(args):
        raise RuntimeError(f"official operation did not receive required argument {name}")
    updated_args = list(args)
    updated_args[index] = value
    return tuple(updated_args), kwargs


def call_metadata(call_index: int) -> dict[str, int | bool]:
    layer = int(call_index % EXPECTED_LAYERS)
    invocation = int(call_index // EXPECTED_LAYERS)
    return {
        "call_index": int(call_index),
        "layer": layer,
        "edge": layer - 1,
        "invocation": invocation,
        "macro_loop": invocation // EXPECTED_DEPTH_CALLS,
        "depth_call": invocation % EXPECTED_DEPTH_CALLS,
        "selected": layer > 0,
    }


def replay_from_call(
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    *,
    board_shuffle_seed: int,
):
    use_qk_l2norm = bool(
        argument(args, kwargs, "use_qk_l2norm_in_kernel", 9, False)
    )
    if not use_qk_l2norm:
        raise RuntimeError("receiver-live diagnostic requires official in-kernel Q/K normalization")
    return replay_gdn2_fp32(
        q=argument(args, kwargs, "q", 0),
        k=argument(args, kwargs, "k", 1),
        v=argument(args, kwargs, "v", 2),
        g=argument(args, kwargs, "g", 3),
        b=argument(args, kwargs, "b", 4),
        w=argument(args, kwargs, "w", 5),
        initial_state=argument(args, kwargs, "initial_state", 7),
        scale=argument(args, kwargs, "scale", 6, None),
        use_qk_l2norm_in_kernel=use_qk_l2norm,
        board_shuffle_seed=board_shuffle_seed,
    )


def shuffled_full_contrast(replay: Any) -> torch.Tensor:
    """Use a wrong-board inherited state while preserving the live subtraction."""

    return replay.board_shuffled_frozen_only - replay.raw_output


class DiscoveryRecorder:
    """Transparent official-op wrapper retaining only receiver-call inputs."""

    def __init__(self, operation: Callable[..., Any], *, shuffle_seed: int) -> None:
        self.operation = operation
        self.shuffle_seed = int(shuffle_seed)
        self.call_index = 0
        self.records: list[dict[str, Any]] = []

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        call_index = self.call_index
        self.call_index += 1
        output = self.operation(*args, **kwargs)
        metadata = call_metadata(call_index)
        if not metadata["selected"]:
            return output

        initial_state = argument(args, kwargs, "initial_state", 7)
        if initial_state is None:
            raise RuntimeError(
                f"receiver call {call_index} did not receive an inherited FutureSeed"
            )
        v = argument(args, kwargs, "v", 2)
        if not v.requires_grad:
            raise RuntimeError(f"receiver call {call_index} value tensor has no gradient path")
        if not isinstance(output, tuple) or len(output) < 2:
            raise RuntimeError("official GDN2 operation did not return output and terminal state")

        self.records.append(
            {
                **metadata,
                "v": v,
                "q_detached": argument(args, kwargs, "q", 0).detach(),
                "k_detached": argument(args, kwargs, "k", 1).detach(),
                "v_detached": v.detach(),
                "g_detached": argument(args, kwargs, "g", 3).detach(),
                "b_detached": argument(args, kwargs, "b", 4).detach(),
                "w_detached": argument(args, kwargs, "w", 5).detach(),
                "initial_state_detached": initial_state.detach(),
                "scale": argument(args, kwargs, "scale", 6, None),
                "use_qk_l2norm": bool(
                    argument(args, kwargs, "use_qk_l2norm_in_kernel", 9, False)
                ),
                "official_output": output[0].detach(),
                "official_terminal": output[1].detach(),
                "shuffle_seed": self.shuffle_seed,
            }
        )
        return output


def replay_discovery_record(record: dict[str, Any]):
    if not record["use_qk_l2norm"]:
        raise RuntimeError("captured receiver call disabled official Q/K normalization")
    return replay_gdn2_fp32(
        q=record["q_detached"],
        k=record["k_detached"],
        v=record["v_detached"],
        g=record["g_detached"],
        b=record["b_detached"],
        w=record["w_detached"],
        initial_state=record["initial_state_detached"],
        scale=record["scale"],
        use_qk_l2norm_in_kernel=True,
        board_shuffle_seed=int(record["shuffle_seed"]),
    )


class HoldoutIntervention:
    """Apply four fixed-direction arms in one official-op forward.

    The four arms already occupy contiguous blocks of one combined batch.  A
    single FP32 replay therefore gives each arm its own current live and frozen
    state while avoiding four redundant token scans per official call.
    """

    def __init__(
        self,
        operation: Callable[..., Any],
        *,
        direction: torch.Tensor,
        base_batch_size: int,
        shuffle_seed: int,
    ) -> None:
        self.operation = operation
        self.direction = direction.detach().float()
        self.base_batch_size = int(base_batch_size)
        self.shuffle_seed = int(shuffle_seed)
        self.call_index = 0
        self.selected_calls = 0
        self.budget_relative_errors: list[float] = []

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        call_index = self.call_index
        self.call_index += 1
        metadata = call_metadata(call_index)
        if not metadata["selected"]:
            return self.operation(*args, **kwargs)

        self.selected_calls += 1
        v = argument(args, kwargs, "v", 2)
        expected_batch = self.base_batch_size * len(INTERVENTION_ARMS)
        if int(v.shape[0]) != expected_batch:
            raise RuntimeError(
                f"intervention batch {v.shape[0]} does not match {expected_batch}"
            )
        edge = int(metadata["edge"])
        edge_direction = self.direction[edge].view(
            1,
            1,
            self.direction.shape[-2],
            self.direction.shape[-1],
        )
        replay = replay_from_call(
            args,
            kwargs,
            board_shuffle_seed=self.shuffle_seed,
        )
        v_chunks = list(v.split(self.base_batch_size, dim=0))
        modified_chunks: list[torch.Tensor] = []

        for arm_idx, (arm_name, v_chunk) in enumerate(
            zip(INTERVENTION_ARMS, v_chunks)
        ):
            start = arm_idx * self.base_batch_size
            stop = start + self.base_batch_size
            arm_slice = slice(start, stop)
            if arm_name in {"plus_full", "minus_full"}:
                signal = replay.full_contrast[arm_slice]
                signed_direction = (
                    edge_direction if arm_name == "plus_full" else -edge_direction
                )
            elif arm_name == "plus_frozen":
                signal = replay.frozen_only[arm_slice]
                signed_direction = edge_direction
            elif arm_name == "plus_shuffled":
                initial_state = argument(args, kwargs, "initial_state", 7)
                if initial_state is None:
                    raise RuntimeError("shuffled receiver arm requires inherited state")
                permutation = deterministic_board_permutation(
                    self.base_batch_size,
                    seed=self.shuffle_seed,
                    device=v.device,
                )
                frozen = initial_state[arm_slice].float().index_select(0, permutation)
                shuffled_frozen = torch.einsum(
                    "bthk,bhkv->bthv",
                    replay.normalized_scaled_query[arm_slice],
                    frozen,
                )
                signal = shuffled_frozen - replay.raw_output[arm_slice]
                signed_direction = edge_direction
            else:  # pragma: no cover - the tuple above is fixed.
                raise RuntimeError(f"unknown intervention arm {arm_name}")

            scaled = scaled_holdout_delta_v(
                signal=signal,
                committed_edit=replay.committed_edit[arm_slice],
                w=argument(args, kwargs, "w", 5)[arm_slice],
                direction=signed_direction,
                epsilon=EPSILON,
                storage_reference=v_chunk,
            )
            relative = (
                (scaled.achieved_rms - scaled.target_rms).abs()
                / scaled.target_rms.clamp_min(1e-12)
            )
            self.budget_relative_errors.append(float(relative.max().cpu().item()))
            modified_chunks.append(scaled.modified_v)

        modified_v = torch.cat(modified_chunks, dim=0)
        updated_args, updated_kwargs = replace_argument(
            args,
            kwargs,
            "v",
            2,
            modified_v,
        )
        return self.operation(*updated_args, **updated_kwargs)


def per_board_loop_ce(
    logits: torch.Tensor,
    labels: torch.Tensor,
    clue_mask: torch.Tensor,
    *,
    blank_weight: float,
) -> torch.Tensor:
    cell_loss = F.cross_entropy(
        logits.float().reshape(-1, study.N),
        labels.reshape(-1),
        reduction="none",
    ).view_as(labels)
    weights = torch.where(
        clue_mask,
        torch.ones_like(cell_loss),
        torch.full_like(cell_loss, float(blank_weight)),
    )
    return (cell_loss * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)


def exact_loop_metrics(
    predictions: list[torch.Tensor],
    loop_ces: list[torch.Tensor],
    labels: torch.Tensor,
    clue_mask: torch.Tensor,
    *,
    device: torch.device,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    labels_device = labels.to(device)
    clues_device = clue_mask.to(device)
    for loop_idx in range(EXPECTED_LOOPS):
        prediction = predictions[loop_idx].to(device)
        metrics = study.metrics_from_predictions(
            prediction,
            labels_device,
            clues_device,
        )
        wrong = ((prediction != labels_device) & ~clues_device).sum(dim=1).float()
        rows.append(
            {
                "loop": loop_idx + 1,
                "ce": float(loop_ces[loop_idx].mean().item()),
                **asdict(metrics),
                "wrong_blank_cells_mean": float(wrong.mean().item()),
                "wrong_blank_cells_max": int(wrong.max().item()),
            }
        )
    return rows


def make_selection(
    dataset: study.OfficialSudokuDataset,
    blank_ranges: Iterable[tuple[int, int]],
    *,
    discovery_per_range: int,
    holdout_per_range: int,
    discovery_seed: int,
    holdout_seed: int,
) -> dict[str, dict[str, Any]]:
    selection: dict[str, dict[str, Any]] = {}
    all_discovery: set[int] = set()
    all_holdout: set[int] = set()
    for range_idx, (holes_min, holes_max) in enumerate(blank_ranges):
        name = f"{holes_min}-{holes_max}"
        candidates = np.asarray(
            dataset._indices_for_blank_range(holes_min, holes_max),
            dtype=np.int64,
        )
        needed = int(discovery_per_range) + int(holdout_per_range)
        if int(candidates.size) < needed:
            raise RuntimeError(
                f"restricted train split has {candidates.size} rows in {name}, needs {needed}"
            )
        discovery_range_seed = int(discovery_seed) + range_idx * 1_000_003
        holdout_range_seed = int(holdout_seed) + range_idx * 1_000_033
        discovery_rng = np.random.default_rng(discovery_range_seed)
        discovery_positions = candidates[
            discovery_rng.permutation(candidates.size)[:discovery_per_range]
        ]
        remaining_positions = np.setdiff1d(
            candidates,
            discovery_positions,
            assume_unique=True,
        )
        holdout_rng = np.random.default_rng(holdout_range_seed)
        holdout_positions = remaining_positions[
            holdout_rng.permutation(remaining_positions.size)[:holdout_per_range]
        ]
        discovery = np.asarray(
            dataset._base_indices(discovery_positions),
            dtype=np.int64,
        )
        holdout = np.asarray(
            dataset._base_indices(holdout_positions),
            dtype=np.int64,
        )
        if set(discovery.tolist()).intersection(holdout.tolist()):
            raise RuntimeError(f"discovery/holdout overlap in range {name}")
        all_discovery.update(int(value) for value in discovery.tolist())
        all_holdout.update(int(value) for value in holdout.tolist())
        selection[name] = {
            "blank_range": [holes_min, holes_max],
            "discovery_selection_seed": discovery_range_seed,
            "holdout_selection_seed": holdout_range_seed,
            "discovery_base_row_ids": discovery.tolist(),
            "holdout_base_row_ids": holdout.tolist(),
            "discovery_base_row_ids_sha256": sha256_int64(discovery),
            "holdout_base_row_ids_sha256": sha256_int64(holdout),
        }
    if all_discovery.intersection(all_holdout):
        raise RuntimeError("global discovery/holdout base-row overlap")
    return selection


def microbatch_slices(
    total: int,
    *,
    logical_batch_size: int,
) -> Iterable[tuple[int, int]]:
    """Partition the fixed independent sample bank into registered microbatches."""

    for logical_start in range(0, int(total), int(logical_batch_size)):
        logical_stop = min(logical_start + int(logical_batch_size), int(total))
        for start in range(logical_start, logical_stop, MICROBATCH):
            yield start, min(start + MICROBATCH, logical_stop)


def load_base_rows(
    dataset: study.OfficialSudokuDataset,
    row_ids: Iterable[int],
    *,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    indices = np.asarray(list(row_ids), dtype=np.int64)
    return dataset._map_arrays(
        dataset.inputs[indices],
        dataset.labels[indices],
        device=device,
    )


def append_holdout_outputs(
    store: dict[str, dict[str, Any]],
    arm_name: str,
    logits: list[torch.Tensor],
    labels: torch.Tensor,
    clue_mask: torch.Tensor,
    *,
    blank_weight: float,
) -> None:
    arm = store[arm_name]
    per_loop_ce = [
        per_board_loop_ce(
            loop_logits,
            labels,
            clue_mask,
            blank_weight=blank_weight,
        ).detach().cpu()
        for loop_logits in logits
    ]
    arm["equal_ce"].append(torch.stack(per_loop_ce).mean(dim=0))
    for loop_idx, (loop_logits, loop_ce) in enumerate(zip(logits, per_loop_ce)):
        arm["loop_ce"][loop_idx].append(loop_ce)
        arm["predictions"][loop_idx].append(loop_logits.argmax(dim=-1).detach().cpu())


def empty_holdout_store() -> dict[str, dict[str, Any]]:
    return {
        arm: {
            "equal_ce": [],
            "loop_ce": [[] for _ in range(EXPECTED_LOOPS)],
            "predictions": [[] for _ in range(EXPECTED_LOOPS)],
        }
        for arm in ARM_NAMES
    }


def finalize_holdout_store(
    store: dict[str, dict[str, Any]],
    labels: torch.Tensor,
    clues: torch.Tensor,
    *,
    device: torch.device,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    summary: dict[str, Any] = {}
    arrays: dict[str, np.ndarray] = {}
    for arm_name, arm in store.items():
        equal_ce = torch.cat(arm["equal_ce"])
        loop_ce = [torch.cat(chunks) for chunks in arm["loop_ce"]]
        predictions = [torch.cat(chunks) for chunks in arm["predictions"]]
        summary[arm_name] = {
            "equal_loop_ce": float(equal_ce.mean().item()),
            "loop_metrics": exact_loop_metrics(
                predictions,
                loop_ce,
                labels,
                clues,
                device=device,
            ),
        }
        arrays[f"holdout_{arm_name}_equal_loop_ce"] = equal_ce.numpy()
        arrays[f"holdout_{arm_name}_loop_ce"] = torch.stack(loop_ce).numpy()
        arrays[f"holdout_{arm_name}_predictions"] = torch.stack(predictions).numpy()
    return summary, arrays


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Receiver-native inherited-vs-live FutureSeed causal diagnostic"
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--checkpoint-sha256", required=True)
    parser.add_argument("--parent-source-sha", required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--expected-gpu-uuid", required=True)
    parser.add_argument("--batch-size", type=int, default=MICROBATCH)
    parser.add_argument("--eval-seed", type=int, default=52051)
    parser.add_argument(
        "--unseen-index-path",
        type=Path,
        required=True,
    )
    parser.add_argument("--unseen-index-sha256", required=True)
    parser.add_argument("--unseen-manifest-path", type=Path, required=True)
    parser.add_argument("--unseen-manifest-sha256", required=True)
    parser.add_argument(
        "--discovery-per-range",
        type=int,
        default=DEFAULT_DISCOVERY_PER_RANGE,
    )
    parser.add_argument(
        "--holdout-per-range",
        type=int,
        default=DEFAULT_HOLDOUT_PER_RANGE,
    )
    parser.add_argument("--discovery-seed", type=int, default=5205101)
    parser.add_argument("--holdout-seed", type=int, default=5205102)
    parser.add_argument(
        "--blank-ranges",
        type=base.parse_range,
        nargs="+",
        default=list(REGISTERED_RANGES),
    )
    parser.add_argument("--blank-weight", type=float, default=8.0)
    args = parser.parse_args()

    if args.batch_size != MICROBATCH:
        raise RuntimeError(f"registered microbatch is exactly {MICROBATCH}")
    if args.discovery_per_range != DEFAULT_DISCOVERY_PER_RANGE:
        raise RuntimeError(
            f"registered discovery count is exactly {DEFAULT_DISCOVERY_PER_RANGE}"
        )
    if args.holdout_per_range != DEFAULT_HOLDOUT_PER_RANGE:
        raise RuntimeError(
            f"registered holdout count is exactly {DEFAULT_HOLDOUT_PER_RANGE}"
        )
    if args.discovery_seed != 5205101 or args.holdout_seed != 5205102:
        raise RuntimeError("registered discovery/holdout seeds are fixed")
    if tuple(args.blank_ranges) != REGISTERED_RANGES:
        raise RuntimeError(f"registered blank ranges are exactly {REGISTERED_RANGES}")
    if args.blank_weight <= 0:
        raise RuntimeError("blank weight must be positive")

    started = time.perf_counter()
    gpu = base.validate_gpu(args.expected_gpu_uuid)
    if (
        gpu["name"] != EXPECTED_GPU_NAME
        or gpu["memory_total_mib"] != EXPECTED_GPU_MEMORY_MIB
    ):
        raise RuntimeError(f"registered A10080 contract mismatch: {gpu}")
    fla_wheel_provenance = verify_pinned_fla_wheel_tree(
        Path(os.environ.get("PERSIST_ROOT", "/huyang2/double-loop"))
    )
    if study.GAIN_BUDGET_FLA_SHA != base.EXPECTED_FLA_SHA:
        raise RuntimeError("pinned FLA constant drift")
    checkpoint_hash = base.sha256_file(args.checkpoint)
    if checkpoint_hash != args.checkpoint_sha256:
        raise RuntimeError("checkpoint SHA mismatch")

    checkpoint_run = args.checkpoint.resolve().parents[1].name
    persistent_root = Path(os.environ.get("PERSIST_ROOT", "/huyang2/double-loop"))
    source_head_path = persistent_root / "runs" / checkpoint_run / "source_HEAD.txt"
    source_patch_path = persistent_root / "runs" / checkpoint_run / "source.patch"
    if source_head_path.read_text(encoding="utf-8").strip() != args.parent_source_sha:
        raise RuntimeError("parent source provenance mismatch")
    if not source_patch_path.is_file() or source_patch_path.stat().st_size != 0:
        raise RuntimeError("parent source patch is missing or nonempty")

    index_path = args.unseen_index_path.resolve()
    if not index_path.is_file():
        raise RuntimeError(f"registered unseen-row index is missing: {index_path}")
    index_sha = base.sha256_file(index_path)
    if index_sha != args.unseen_index_sha256:
        raise RuntimeError("registered unseen-row index SHA mismatch")
    manifest_path = args.unseen_manifest_path.resolve()
    if not manifest_path.is_file():
        raise RuntimeError(f"unseen-row manifest is missing: {manifest_path}")
    manifest_sha = base.sha256_file(manifest_path)
    if manifest_sha != args.unseen_manifest_sha256:
        raise RuntimeError("registered unseen-row manifest SHA mismatch")
    unseen_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    data_inputs_path = args.data_dir.resolve() / "train" / "all__inputs.npy"
    data_labels_path = args.data_dir.resolve() / "train" / "all__labels.npy"
    data_inputs_sha = base.sha256_file(data_inputs_path)
    data_labels_sha = base.sha256_file(data_labels_path)
    if data_inputs_sha != EXPECTED_INPUTS_SHA256:
        raise RuntimeError("official train input data SHA mismatch")
    if data_labels_sha != EXPECTED_LABELS_SHA256:
        raise RuntimeError("official train label data SHA mismatch")
    manifest_contract = {
        "checkpoint_sha256": args.checkpoint_sha256,
        "saved_step": 3000,
        "seed": 52,
        "microbatch": 32,
        "grad_accum_steps": 4,
        "effective_batch": 128,
        "target_blank_range": [51, 64],
        "rng_matches_checkpoint": True,
        "sampling_mode": "all_strict_unseen",
        "output_indices_sha256": index_sha,
    }
    for key, expected in manifest_contract.items():
        if unseen_manifest.get(key) != expected:
            raise RuntimeError(
                f"unseen manifest mismatch for {key}: "
                f"{unseen_manifest.get(key)!r} != {expected!r}"
            )

    checkpoint = torch.load(
        args.checkpoint,
        map_location="cpu",
        weights_only=False,
        mmap=True,
    )
    saved_args = dict(checkpoint["args"])
    base.validate_parent(saved_args, int(checkpoint.get("saved_at_step", -1)))
    study.configure_sudoku(int(saved_args["size"]), 0, 0)

    device = torch.device("cuda", 0)
    model = base.build_model(saved_args).to(device)
    missing, unexpected = model.load_state_dict(checkpoint["model"], strict=False)
    if missing or unexpected:
        raise RuntimeError(
            f"checkpoint is not exact: missing={missing}, unexpected={unexpected}"
        )
    del checkpoint
    model.eval()
    runtime = study.strict_fla_runtime_summary(model, "gdn2")
    if runtime["fla_source_sha"] != base.EXPECTED_FLA_SHA:
        raise RuntimeError("runtime FLA SHA drift")
    if len(runtime["layers"]) != EXPECTED_LAYERS:
        raise RuntimeError("expected exactly 12 pinned official GDN2 layers")

    dataset = study.OfficialSudokuDataset(
        args.data_dir,
        "train",
        row_indices_path=index_path,
    )
    selection = make_selection(
        dataset,
        args.blank_ranges,
        discovery_per_range=args.discovery_per_range,
        holdout_per_range=args.holdout_per_range,
        discovery_seed=args.discovery_seed,
        holdout_seed=args.holdout_seed,
    )
    selection_sha = sha256_json(selection)
    parameter_hash_before = parameter_sha256(model)
    torch.cuda.reset_peak_memory_stats(device)
    original_operation = study.chunk_gdn2
    if original_operation is None:
        raise RuntimeError("pinned official chunk_gdn2 operation is unavailable")

    discovery_arrays: dict[str, list[torch.Tensor]] = {
        "full": [],
        "frozen": [],
        "shuffled": [],
    }
    discovery_rows: dict[str, Any] = {}
    replay_output_errors: list[float] = []
    replay_terminal_errors: list[float] = []
    backward_counts: list[int] = []

    for range_idx, (holes_min, holes_max) in enumerate(args.blank_ranges):
        range_name = f"{holes_min}-{holes_max}"
        row_ids = selection[range_name]["discovery_base_row_ids"]
        range_gradients = {
            key: torch.zeros(
                args.discovery_per_range,
                EXPECTED_LAYERS - 1,
                int(saved_args["heads"]),
                int(saved_args["head_dim"]),
                dtype=torch.float32,
            )
            for key in discovery_arrays
        }
        range_started = time.perf_counter()

        for microbatch_idx, (start, stop) in enumerate(
            microbatch_slices(
                args.discovery_per_range,
                logical_batch_size=args.batch_size,
            )
        ):
            batch_ids = row_ids[start:stop]
            inputs, labels, clue_mask = load_base_rows(dataset, batch_ids, device=device)
            current_batch = int(inputs.shape[0])
            recorder = DiscoveryRecorder(
                original_operation,
                shuffle_seed=args.eval_seed
                + range_idx * 10_000_019
                + microbatch_idx * 1_009,
            )
            study.chunk_gdn2 = recorder
            try:
                with study.forward_autocast("bfloat16", device):
                    loop_logits, _trace = model.forward_trace(
                        inputs,
                        loops=EXPECTED_LOOPS,
                        noise_scale=0.0,
                    )
            finally:
                study.chunk_gdn2 = original_operation

            if recorder.call_index != EXPECTED_OFFICIAL_BACKWARDS:
                raise RuntimeError(
                    f"discovery captured {recorder.call_index} official calls, "
                    f"expected {EXPECTED_OFFICIAL_BACKWARDS}"
                )
            if len(recorder.records) != EXPECTED_RECEIVER_CALLS:
                raise RuntimeError(
                    f"discovery captured {len(recorder.records)} receivers, "
                    f"expected {EXPECTED_RECEIVER_CALLS}"
                )
            graph = base.graph_names(loop_logits[-1])
            backward_count = graph.count("ChunkGDN2FunctionBackward")
            backward_counts.append(backward_count)
            if backward_count != EXPECTED_OFFICIAL_BACKWARDS:
                raise RuntimeError(
                    f"official backward count {backward_count} != "
                    f"{EXPECTED_OFFICIAL_BACKWARDS}"
                )
            loop_losses = [
                study.loss_from_logits(
                    logits,
                    labels,
                    clue_mask,
                    blank_weight=args.blank_weight,
                )
                for logits in loop_logits
            ]
            equal_loss = torch.stack(loop_losses).mean()
            retained_v = [record["v"] for record in recorder.records]
            v_gradients = torch.autograd.grad(
                equal_loss,
                retained_v,
                retain_graph=False,
                create_graph=False,
                allow_unused=True,
            )
            if any(gradient is None for gradient in v_gradients):
                inactive = [
                    idx for idx, gradient in enumerate(v_gradients) if gradient is None
                ]
                raise RuntimeError(f"receiver value cotangent missing at calls {inactive}")

            per_board = {
                key: torch.zeros(
                    current_batch,
                    EXPECTED_LAYERS - 1,
                    int(saved_args["heads"]),
                    int(saved_args["head_dim"]),
                    dtype=torch.float32,
                    device=device,
                )
                for key in discovery_arrays
            }
            for record, v_gradient in zip(recorder.records, v_gradients):
                assert v_gradient is not None
                replay = replay_discovery_record(record)
                output_error = rms_relative_error(
                    replay.raw_output,
                    record["official_output"],
                )
                terminal_error = rms_relative_error(
                    replay.final_state,
                    record["official_terminal"],
                )
                replay_output_errors.append(output_error)
                replay_terminal_errors.append(terminal_error)
                if max(output_error, terminal_error) > REPLAY_REL_RMS_MAX:
                    raise RuntimeError(
                        "receiver replay exceeded registered relative RMS: "
                        f"output={output_error} terminal={terminal_error}"
                    )
                signals = {
                    "full": replay.full_contrast,
                    "frozen": replay.frozen_only,
                    "shuffled": shuffled_full_contrast(replay),
                }
                edge = int(record["edge"])
                for signal_name, signal in signals.items():
                    per_board[signal_name][:, edge].add_(
                        per_board_virtual_alpha_gradient(
                            v_grad=v_gradient,
                            signal=signal,
                            committed_edit=replay.committed_edit,
                            rho=RHO,
                        )
                    )
            for signal_name in discovery_arrays:
                range_gradients[signal_name][start:stop].copy_(
                    per_board[signal_name].cpu()
                )

            del loop_logits, loop_losses, equal_loss, retained_v, v_gradients
            del recorder, per_board, inputs, labels, clue_mask
            torch.cuda.empty_cache()

        for signal_name in discovery_arrays:
            discovery_arrays[signal_name].append(range_gradients[signal_name])
        discovery_rows[range_name] = {
            "blank_range": [holes_min, holes_max],
            "boards": args.discovery_per_range,
            "logical_batch_size": args.batch_size,
            "microbatch": MICROBATCH,
            "elapsed_sec": time.perf_counter() - range_started,
        }

    discovery_full = torch.cat(discovery_arrays["full"], dim=0)
    discovery_frozen = torch.cat(discovery_arrays["frozen"], dim=0)
    discovery_shuffled = torch.cat(discovery_arrays["shuffled"], dim=0)
    aggregate = aggregate_discovery_direction(discovery_full)
    max_board_energy_share = float(aggregate.board_energy_share.max().item())
    global_direction = aggregate.descent_direction.to(device)

    holdout_rows: dict[str, Any] = {}
    holdout_npz: dict[str, np.ndarray] = {}
    holdout_all_equal_ce: dict[str, list[torch.Tensor]] = {
        arm: [] for arm in ARM_NAMES
    }
    max_budget_relative_error = 0.0

    for range_idx, (holes_min, holes_max) in enumerate(args.blank_ranges):
        range_name = f"{holes_min}-{holes_max}"
        row_ids = selection[range_name]["holdout_base_row_ids"]
        store = empty_holdout_store()
        labels_chunks: list[torch.Tensor] = []
        clue_chunks: list[torch.Tensor] = []
        range_started = time.perf_counter()

        for microbatch_idx, (start, stop) in enumerate(
            microbatch_slices(
                args.holdout_per_range,
                logical_batch_size=args.batch_size,
            )
        ):
            batch_ids = row_ids[start:stop]
            inputs, labels, clue_mask = load_base_rows(dataset, batch_ids, device=device)
            current_batch = int(inputs.shape[0])
            labels_chunks.append(labels.detach().cpu())
            clue_chunks.append(clue_mask.detach().cpu())

            if study.chunk_gdn2 is not original_operation:
                raise RuntimeError("baseline arm is not unwrapped")
            with torch.no_grad(), study.forward_autocast("bfloat16", device):
                baseline_logits, _trace = model.forward_trace(
                    inputs,
                    loops=EXPECTED_LOOPS,
                    noise_scale=0.0,
                )
            append_holdout_outputs(
                store,
                "baseline",
                baseline_logits,
                labels,
                clue_mask,
                blank_weight=args.blank_weight,
            )

            intervention_inputs = inputs.repeat(len(INTERVENTION_ARMS), 1)
            intervention_labels = labels.repeat(len(INTERVENTION_ARMS), 1)
            intervention_clues = clue_mask.repeat(len(INTERVENTION_ARMS), 1)
            wrapper = HoldoutIntervention(
                original_operation,
                direction=global_direction,
                base_batch_size=current_batch,
                shuffle_seed=args.eval_seed
                + 100_000_007
                + range_idx * 10_000_019
                + microbatch_idx * 1_009,
            )
            study.chunk_gdn2 = wrapper
            try:
                with torch.no_grad(), study.forward_autocast("bfloat16", device):
                    intervention_logits, _trace = model.forward_trace(
                        intervention_inputs,
                        loops=EXPECTED_LOOPS,
                        noise_scale=0.0,
                    )
            finally:
                study.chunk_gdn2 = original_operation
            if wrapper.call_index != EXPECTED_OFFICIAL_BACKWARDS:
                raise RuntimeError(
                    f"holdout captured {wrapper.call_index} official calls, "
                    f"expected {EXPECTED_OFFICIAL_BACKWARDS}"
                )
            if wrapper.selected_calls != EXPECTED_RECEIVER_CALLS:
                raise RuntimeError(
                    f"holdout modified {wrapper.selected_calls} receiver calls, "
                    f"expected {EXPECTED_RECEIVER_CALLS}"
                )
            if wrapper.budget_relative_errors:
                max_budget_relative_error = max(
                    max_budget_relative_error,
                    max(wrapper.budget_relative_errors),
                )

            for arm_idx, arm_name in enumerate(INTERVENTION_ARMS):
                arm_slice = slice(
                    arm_idx * current_batch,
                    (arm_idx + 1) * current_batch,
                )
                append_holdout_outputs(
                    store,
                    arm_name,
                    [logits[arm_slice] for logits in intervention_logits],
                    intervention_labels[arm_slice],
                    intervention_clues[arm_slice],
                    blank_weight=args.blank_weight,
                )

            del baseline_logits, intervention_logits, wrapper
            del inputs, labels, clue_mask, intervention_inputs
            del intervention_labels, intervention_clues
            torch.cuda.empty_cache()

        labels_cpu = torch.cat(labels_chunks)
        clues_cpu = torch.cat(clue_chunks)
        range_summary, range_arrays = finalize_holdout_store(
            store,
            labels_cpu,
            clues_cpu,
            device=device,
        )
        baseline_ce = torch.from_numpy(
            range_arrays["holdout_baseline_equal_loop_ce"]
        )
        range_decision: dict[str, Any] = {}
        for arm_name in INTERVENTION_ARMS:
            arm_ce = torch.from_numpy(
                range_arrays[f"holdout_{arm_name}_equal_loop_ce"]
            )
            range_decision[f"{arm_name}_ce_improvement"] = float(
                (baseline_ce - arm_ce).mean().item()
            )
        holdout_rows[range_name] = {
            "blank_range": [holes_min, holes_max],
            "boards": args.holdout_per_range,
            "logical_batch_size": args.batch_size,
            "microbatch": MICROBATCH,
            "arms": range_summary,
            "ce_improvements": range_decision,
            "elapsed_sec": time.perf_counter() - range_started,
        }
        safe_range = range_name.replace("-", "_")
        for key, value in range_arrays.items():
            holdout_npz[f"range_{safe_range}_{key}"] = value
        for arm_name in ARM_NAMES:
            holdout_all_equal_ce[arm_name].append(
                torch.from_numpy(range_arrays[f"holdout_{arm_name}_equal_loop_ce"])
            )
        holdout_npz[f"range_{safe_range}_labels"] = labels_cpu.numpy()
        holdout_npz[f"range_{safe_range}_clue_mask"] = clues_cpu.numpy()

    all_equal_ce = {
        arm: torch.cat(chunks) for arm, chunks in holdout_all_equal_ce.items()
    }
    full_improvement_values = all_equal_ce["baseline"] - all_equal_ce["plus_full"]
    frozen_improvement_values = (
        all_equal_ce["baseline"] - all_equal_ce["plus_frozen"]
    )
    shuffled_improvement_values = (
        all_equal_ce["baseline"] - all_equal_ce["plus_shuffled"]
    )
    minus_improvement_values = (
        all_equal_ce["baseline"] - all_equal_ce["minus_full"]
    )
    full_improvement = float(full_improvement_values.mean().item())
    frozen_improvement = float(frozen_improvement_values.mean().item())
    shuffled_improvement = float(shuffled_improvement_values.mean().item())
    minus_improvement = float(minus_improvement_values.mean().item())
    bootstrap = paired_bootstrap_comparison(
        control=all_equal_ce["baseline"],
        candidate=all_equal_ce["plus_full"],
        objective="minimize",
        seed=args.eval_seed + 777_777_779,
        samples=BOOTSTRAP_SAMPLES,
    )

    reversed_positive_ranges = []
    for range_name, row in holdout_rows.items():
        full_delta = row["ce_improvements"]["plus_full_ce_improvement"]
        minus_delta = row["ce_improvements"]["minus_full_ce_improvement"]
        if full_delta > 0.0 and minus_delta < 0.0:
            reversed_positive_ranges.append(range_name)

    checks = {
        "full_ce_improvement_at_least_0_01": full_improvement >= 0.01,
        "bootstrap_95pct_lower_positive": bootstrap.lower > 0.0,
        "at_least_two_ranges_positive_with_minus_reversal": len(
            reversed_positive_ranges
        )
        >= 2,
        "full_beats_frozen_by_0_005": (
            full_improvement - frozen_improvement
        )
        >= 0.005,
        "shuffled_no_more_than_25pct_full": (
            shuffled_improvement <= 0.25 * full_improvement
        ),
        "max_discovery_board_energy_share_at_most_0_20": (
            max_board_energy_share <= 0.20
        ),
    }
    admitted = all(checks.values())
    status = "admitted" if admitted else "completed_rejected"

    parameter_hash_after = parameter_sha256(model)
    if parameter_hash_after != parameter_hash_before:
        raise RuntimeError("diagnostic mutated model parameters")
    if any(parameter.grad is not None for parameter in model.parameters()):
        raise RuntimeError("autograd.grad populated a model parameter .grad field")
    if study.chunk_gdn2 is not original_operation:
        raise RuntimeError("official operation wrapper was not restored")
    if max_budget_relative_error > QUANTIZED_BUDGET_REL_ERROR_MAX:
        raise RuntimeError(
            f"holdout committed-edit budget drifted by {max_budget_relative_error}"
        )

    current_head = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], text=True
    ).strip()
    source_clean = (
        subprocess.check_output(["git", "status", "--porcelain"], text=True).strip()
        == ""
    )
    if not source_clean:
        raise RuntimeError("diagnostic source worktree became dirty")

    arrays_path = args.out.with_name("cotangent_arrays.npz")
    arrays_payload: dict[str, np.ndarray] = {
        "discovery_full_gradients": discovery_full.numpy(),
        "discovery_frozen_gradients": discovery_frozen.numpy(),
        "discovery_shuffled_gradients": discovery_shuffled.numpy(),
        "discovery_mean_gradient": aggregate.mean_gradient.numpy(),
        "discovery_descent_direction": aggregate.descent_direction.numpy(),
        "discovery_board_cosine_to_mean": aggregate.board_cosine_to_mean.numpy(),
        "discovery_board_energy_share": aggregate.board_energy_share.numpy(),
        **holdout_npz,
    }
    for range_name, row in selection.items():
        safe_range = range_name.replace("-", "_")
        arrays_payload[f"range_{safe_range}_discovery_base_row_ids"] = np.asarray(
            row["discovery_base_row_ids"], dtype=np.int64
        )
        arrays_payload[f"range_{safe_range}_holdout_base_row_ids"] = np.asarray(
            row["holdout_base_row_ids"], dtype=np.int64
        )
    arrays_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(arrays_path, **arrays_payload)

    decision = {
        "status": status,
        "candidate_admitted": admitted,
        "checks": checks,
        "thresholds": {
            "full_ce_improvement_min": 0.01,
            "bootstrap_confidence": 0.95,
            "positive_reversed_ranges_min": 2,
            "full_minus_frozen_improvement_min": 0.005,
            "shuffled_fraction_of_full_max": 0.25,
            "max_discovery_board_energy_share": 0.20,
        },
        "full_ce_improvement": full_improvement,
        "frozen_ce_improvement": frozen_improvement,
        "shuffled_ce_improvement": shuffled_improvement,
        "minus_full_ce_improvement": minus_improvement,
        "full_minus_frozen_ce_improvement": (
            full_improvement - frozen_improvement
        ),
        "shuffled_fraction_of_full": (
            shuffled_improvement / full_improvement
            if full_improvement > 0.0
            else None
        ),
        "reversed_positive_ranges": reversed_positive_ranges,
        "paired_bootstrap": asdict(bootstrap),
        "max_discovery_board_energy_share": max_board_energy_share,
        "next_decision": (
            "Register the directional-MQAR receiver-live candidate; do not train a "
            "Sudoku graft yet."
            if admitted
            else "Close receiver-native inherited-vs-live recommit without rescue."
        ),
    }
    payload = {
        "plan_id": PLAN_ID,
        "schema_version": SCHEMA_VERSION,
        "status": status,
        "kind": "receiver_native_inherited_vs_live_causal_diagnostic",
        "source_sha": current_head,
        "source_clean": source_clean,
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": checkpoint_hash,
        "parent_source_sha": args.parent_source_sha,
        "gpu": gpu,
        "torch_version": torch.__version__,
        "runtime": runtime,
        "fla_wheel_provenance": fla_wheel_provenance,
        "data": {
            "data_dir": str(args.data_dir),
            "split": "train",
            "inputs_sha256": data_inputs_sha,
            "labels_sha256": data_labels_sha,
            "restriction_index": str(index_path),
            "restriction_index_sha256": index_sha,
            "restriction_manifest": str(manifest_path),
            "restriction_manifest_sha256": manifest_sha,
            "restriction_manifest_contract": manifest_contract,
            "parent_rng_reconstruction_exact": True,
            "restriction_index_row_count": int(dataset.row_indices.size),
            "restriction_index_base_row_ids_sha256": sha256_int64(
                np.asarray(dataset.row_indices, dtype=np.int64)
            ),
            "selection_sha256": selection_sha,
            "selection": selection,
            "discovery_rows": len(discovery_full),
            "holdout_rows": len(all_equal_ce["baseline"]),
            "discovery_per_range": args.discovery_per_range,
            "holdout_per_range": args.holdout_per_range,
            "discovery_seed": args.discovery_seed,
            "holdout_seed": args.holdout_seed,
            "logical_batch_size": args.batch_size,
            "microbatch_size": MICROBATCH,
        },
        "protocol": {
            "loops": EXPECTED_LOOPS,
            "equal_loop_ce": True,
            "blank_weight": float(args.blank_weight),
            "rho_for_virtual_gradient": RHO,
            "holdout_committed_edit_epsilon": EPSILON,
            "quantized_budget_relative_error_max": (
                QUANTIZED_BUDGET_REL_ERROR_MAX
            ),
            "global_direction_shape": list(aggregate.descent_direction.shape),
            "global_direction_from": "all discovery ranges and boards; full contrast only",
            "edge_or_range_selection": False,
            "baseline_unwrapped": True,
            "intervention_arms": list(INTERVENTION_ARMS),
            "shuffled_signal": "q^T F_board_permutation - q^T S_live_post_write",
        },
        "integrity": {
            "expected_receiver_calls_per_forward": EXPECTED_RECEIVER_CALLS,
            "expected_official_backward_count": EXPECTED_OFFICIAL_BACKWARDS,
            "discovery_backward_count_min": min(backward_counts),
            "discovery_backward_count_max": max(backward_counts),
            "replay_output_relative_rms_max": max(replay_output_errors),
            "replay_terminal_relative_rms_max": max(replay_terminal_errors),
            "replay_relative_rms_threshold": REPLAY_REL_RMS_MAX,
            "holdout_budget_relative_error_max": max_budget_relative_error,
            "holdout_budget_relative_error_threshold": (
                QUANTIZED_BUDGET_REL_ERROR_MAX
            ),
            "parameter_sha256_before": parameter_hash_before,
            "parameter_sha256_after": parameter_hash_after,
            "all_parameter_grads_none": True,
            "optimizer_created": False,
            "checkpoint_saved": False,
        },
        "discovery": {
            "ranges": discovery_rows,
            "boards": int(discovery_full.shape[0]),
            "direction_rms": float(
                aggregate.descent_direction.float().square().mean().sqrt().item()
            ),
            "mean_gradient_rms": float(
                aggregate.mean_gradient.float().square().mean().sqrt().item()
            ),
            "max_board_energy_share": max_board_energy_share,
        },
        "holdout": holdout_rows,
        "decision": decision,
        "arrays_path": str(arrays_path),
        "arrays_sha256": base.sha256_file(arrays_path),
        "peak_allocated_mib": torch.cuda.max_memory_allocated(device) / (1024.0**2),
        "peak_reserved_mib": torch.cuda.max_memory_reserved(device) / (1024.0**2),
        "elapsed_sec": time.perf_counter() - started,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "out": str(args.out),
                "arrays": str(arrays_path),
                "status": status,
                "decision": decision,
                "elapsed_sec": payload["elapsed_sec"],
                "peak_allocated_mib": payload["peak_allocated_mib"],
            },
            sort_keys=True,
            allow_nan=False,
        )
    )


if __name__ == "__main__":
    main()
