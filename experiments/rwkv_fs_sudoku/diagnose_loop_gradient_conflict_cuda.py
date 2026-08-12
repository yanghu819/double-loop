#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

for cache_var in (
    "XDG_CACHE_HOME",
    "TRITON_CACHE_DIR",
    "TORCHINDUCTOR_CACHE_DIR",
    "TORCH_EXTENSIONS_DIR",
    "TMPDIR",
):
    cache_path = os.environ.get(cache_var, "")
    if not cache_path.startswith("/huyang2/double-loop/"):
        raise RuntimeError(
            f"{cache_var} must point below /huyang2/double-loop before CUDA imports"
        )
if os.environ.get("FLA_DISABLE_BACKEND_DISPATCH") != "1":
    raise RuntimeError("FLA_DISABLE_BACKEND_DISPATCH=1 is required")
if os.environ.get("FLA_CONV_BACKEND") != "triton":
    raise RuntimeError("FLA_CONV_BACKEND=triton is required")
if os.environ.get("FLA_STRICT_OFFICIAL") != "1":
    raise RuntimeError("FLA_STRICT_OFFICIAL=1 is required")

import torch

import study_rwkv_futureseed_loop as study


EXPECTED_FLA_SHA = "9c8e42e762fce087c27b673af4922795d9edb85e"
GROUPS = (
    "future_seed_gate",
    "gdn2_address",
    "gdn2_edit",
    "shared_shell",
)
MECHANISM_GROUPS = GROUPS[:3]
GROUP_EARLY_LATE_FLOORS = {
    "future_seed_gate": -0.10,
    "gdn2_address": -0.05,
    "gdn2_edit": -0.05,
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def saved_value(saved_args: dict[str, Any], name: str, default: Any) -> Any:
    value = saved_args.get(name, default)
    return default if value is None else value


def build_model(saved_args: dict[str, Any]) -> study.FutureSeedLoopSudoku:
    return study.FutureSeedLoopSudoku(
        d_model=int(saved_args["d_model"]),
        layers=int(saved_args["layers"]),
        heads=int(saved_args["heads"]),
        head_dim=int(saved_args["head_dim"]),
        channel_mult=int(saved_args["channel_mult"]),
        l_cycles=int(saved_args["l_cycles"]),
        max_loops=int(saved_args["max_loops"]),
        lambda_=float(saved_args["lambda_"]),
        loop_update_mode=str(saved_value(saved_args, "loop_update_mode", "fixed")),
        loop_update_gate_init=float(
            saved_value(saved_args, "loop_update_gate_init", 0.95)
        ),
        future_seed_scale=float(saved_value(saved_args, "future_seed_scale", 1.0)),
        future_seed_decay=float(saved_value(saved_args, "future_seed_decay", 0.0)),
        future_seed_update=str(saved_value(saved_args, "future_seed_update", "fixed")),
        future_seed_norm_mode=str(
            saved_value(saved_args, "future_seed_norm_mode", "unit")
        ),
        future_seed_gate_mode=str(
            saved_value(saved_args, "future_seed_gate_mode", "head")
        ),
        future_seed_scope=str(saved_value(saved_args, "future_seed_scope", "layer")),
        future_seed_readout_hop=int(
            saved_value(saved_args, "future_seed_readout_hop", 0)
        ),
        future_seed_content_mode=str(
            saved_value(saved_args, "future_seed_content_mode", "terminal")
        ),
        loop_feedback_scale=float(saved_value(saved_args, "loop_feedback_scale", 0.0)),
        loop_feedback_detach=bool(
            saved_value(saved_args, "loop_feedback_detach", False)
        ),
        loop_feedback_corrupt_prob=float(
            saved_value(saved_args, "loop_feedback_corrupt_prob", 0.0)
        ),
        loop_feedback_corrupt_mix=float(
            saved_value(saved_args, "loop_feedback_corrupt_mix", 0.0)
        ),
        loop_feedback_corrupt_mode=str(
            saved_value(saved_args, "loop_feedback_corrupt_mode", "random_token")
        ),
        loop_time_scale=float(saved_value(saved_args, "loop_time_scale", 0.0)),
        scratch_mode=str(saved_value(saved_args, "scratch_mode", "none")),
        scratch_scale=float(saved_value(saved_args, "scratch_scale", 1.0)),
        scratch_noise_scale=float(
            saved_value(saved_args, "scratch_noise_scale", 0.0)
        ),
        scratch_gauss_projections=int(
            saved_value(saved_args, "scratch_gauss_projections", 0)
        ),
        scratch_gate_bias=float(saved_value(saved_args, "scratch_gate_bias", -2.0)),
        scratch_decay_bias=float(
            saved_value(saved_args, "scratch_decay_bias", 2.0)
        ),
        hidden_agg_noise_scale=float(
            saved_value(saved_args, "hidden_agg_noise_scale", 0.0)
        ),
        hidden_agg_noise_temp=float(
            saved_value(saved_args, "hidden_agg_noise_temp", 1.0)
        ),
        hidden_agg_noise_detach=bool(
            saved_value(saved_args, "hidden_agg_noise_detach", True)
        ),
        hidden_agg_noise_mode=str(
            saved_value(saved_args, "hidden_agg_noise_mode", "gumbel")
        ),
        hidden_agg_noise_topk=int(
            saved_value(saved_args, "hidden_agg_noise_topk", 8)
        ),
        hidden_agg_noise_max_norm=float(
            saved_value(saved_args, "hidden_agg_noise_max_norm", 0.0)
        ),
        activation_checkpoint=False,
        rwkv_kernel=str(saved_value(saved_args, "rwkv_kernel", "auto")),
        backbone=str(saved_args["backbone"]),
        gdn_mode=str(saved_args["gdn_mode"]),
        gdn_expand_v=float(saved_args["gdn_expand_v"]),
        gdn_progressive_base_expand_v=float(
            saved_value(saved_args, "gdn_progressive_base_expand_v", 0.0)
        ),
        gdn_use_short_conv=bool(saved_args["gdn_use_short_conv"]),
        gdn_conv_size=int(saved_value(saved_args, "gdn_conv_size", 4)),
        gdn_allow_neg_eigval=bool(
            saved_value(saved_args, "gdn_allow_neg_eigval", False)
        ),
        gdn2_gain_budget_mode="none",
        gdn2_fast_slow_decay_mode="none",
        gdn2_precondition_mode=str(
            saved_value(saved_args, "gdn2_precondition_mode", "none")
        ),
        gdn2_address_mode=str(saved_value(saved_args, "gdn2_address_mode", "none")),
        gdn2_update_mode=str(saved_value(saved_args, "gdn2_update_mode", "none")),
        gdn2_state_expert_mode=str(
            saved_value(saved_args, "gdn2_state_expert_mode", "none")
        ),
        gdn2_cross_layer_init=str(
            saved_value(saved_args, "gdn2_cross_layer_init", "independent")
        ),
        raven_num_slots=int(saved_value(saved_args, "raven_num_slots", 0)),
        raven_topk=int(saved_value(saved_args, "raven_topk", 0)),
    )


def graph_names(tensor: torch.Tensor) -> list[str]:
    names: list[str] = []
    queue = [tensor.grad_fn]
    seen: set[int] = set()
    while queue:
        fn = queue.pop()
        if fn is None or id(fn) in seen:
            continue
        seen.add(id(fn))
        names.append(type(fn).__name__)
        queue.extend(next_fn for next_fn, _ in fn.next_functions if next_fn is not None)
    return names


def parameter_group(name: str) -> str:
    if name.endswith("future_seed_logit"):
        return "future_seed_gate"
    if ".time_mix.core." in name:
        address_terms = (
            ".q_proj.",
            ".k_proj.",
            ".q_conv1d.",
            ".k_conv1d.",
        )
        return "gdn2_address" if any(term in name for term in address_terms) else "gdn2_edit"
    return "shared_shell"


def group_parameters(
    model: torch.nn.Module,
) -> tuple[list[torch.nn.Parameter], list[str], dict[str, Any]]:
    parameters: list[torch.nn.Parameter] = []
    groups: list[str] = []
    summary: dict[str, Any] = {
        group: {"tensor_count": 0, "parameter_count": 0, "names": []}
        for group in GROUPS
    }
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        group = parameter_group(name)
        parameters.append(parameter)
        groups.append(group)
        summary[group]["tensor_count"] += 1
        summary[group]["parameter_count"] += parameter.numel()
        summary[group]["names"].append(name)
    for group in GROUPS:
        if summary[group]["tensor_count"] == 0:
            raise RuntimeError(f"parameter group {group} is empty")
    return parameters, groups, summary


def grouped_gradient_vectors(
    grads: tuple[torch.Tensor | None, ...],
    parameters: list[torch.nn.Parameter],
    groups: list[str],
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    chunks: dict[str, list[torch.Tensor]] = {group: [] for group in GROUPS}
    active_tensors = {group: 0 for group in GROUPS}
    finite = {group: True for group in GROUPS}
    for grad, parameter, group in zip(grads, parameters, groups):
        if grad is None:
            chunk = torch.zeros(parameter.numel(), dtype=torch.float32)
        else:
            chunk = grad.detach().float().reshape(-1).cpu()
            active_tensors[group] += 1
            finite[group] = finite[group] and bool(torch.isfinite(chunk).all())
        chunks[group].append(chunk)
    vectors = {group: torch.cat(chunks[group]) for group in GROUPS}
    diagnostics = {
        group: {
            "active_tensor_count": active_tensors[group],
            "finite": finite[group],
            "norm": float(vectors[group].norm().item()),
        }
        for group in GROUPS
    }
    return vectors, diagnostics


def cosine(left: torch.Tensor, right: torch.Tensor) -> float | None:
    left_norm = float(left.norm().item())
    right_norm = float(right.norm().item())
    if left_norm <= 1e-12 or right_norm <= 1e-12:
        return None
    return float(torch.dot(left, right).item() / (left_norm * right_norm))


def summarize_group(vectors: list[torch.Tensor]) -> dict[str, Any]:
    matrix: list[list[float | None]] = []
    pairs: list[float] = []
    adjacent: list[float] = []
    for left_idx, left in enumerate(vectors):
        row: list[float | None] = []
        for right_idx, right in enumerate(vectors):
            value = cosine(left, right)
            row.append(value)
            if right_idx > left_idx and value is not None:
                pairs.append(value)
            if right_idx == left_idx + 1 and value is not None:
                adjacent.append(value)
        matrix.append(row)
    early = vectors[0] + vectors[1]
    late = vectors[-2] + vectors[-1]
    norm_sum = sum(float(vector.norm().item()) for vector in vectors)
    summed = torch.stack(vectors).sum(dim=0)
    cancellation_ratio = float(summed.norm().item() / max(norm_sum, 1e-12))
    return {
        "grad_norms": [float(vector.norm().item()) for vector in vectors],
        "cosine_matrix": matrix,
        "early_vs_late_cosine": cosine(early, late),
        "all_pair_conflict_fraction": (
            sum(value < 0.0 for value in pairs) / len(pairs) if pairs else None
        ),
        "adjacent_conflict_fraction": (
            sum(value < 0.0 for value in adjacent) / len(adjacent)
            if adjacent
            else None
        ),
        "final_vs_prior_mean_cosine": (
            sum(value for value in (cosine(vectors[-1], prior) for prior in vectors[:-1]) if value is not None)
            / max(
                1,
                sum(cosine(vectors[-1], prior) is not None for prior in vectors[:-1]),
            )
        ),
        "cancellation_ratio": cancellation_ratio,
    }


def group_admitted(group: str, summary: dict[str, Any]) -> bool:
    early_late = summary["early_vs_late_cosine"]
    conflicts = summary["all_pair_conflict_fraction"]
    cancellation = summary["cancellation_ratio"]
    return bool(
        early_late is not None
        and early_late <= GROUP_EARLY_LATE_FLOORS[group]
        and conflicts is not None
        and conflicts >= 0.30
        and cancellation <= 0.75
    )


def validate_parent(saved_args: dict[str, Any], saved_at_step: int) -> None:
    expected = {
        "size": (9, 9),
        "d_model": (256, 256),
        "layers": (12, 12),
        "heads": (8, 8),
        "head_dim": (32, 32),
        "channel_mult": (4, 4),
        "l_cycles": (2, 2),
        "max_loops": (5, 5),
        "backbone": ("gdn2", "gdn2"),
        "gdn_mode": ("chunk", "chunk"),
        "gdn_expand_v": (1.0, 1.0),
        "gdn2_address_mode": ("position_qk", "none"),
        "gdn2_update_mode": ("none", "none"),
        "gdn2_state_expert_mode": ("none", "none"),
        "gdn2_cross_layer_init": ("independent", "independent"),
        "future_seed_scope": ("layer", "layer"),
        "future_seed_content_mode": ("terminal", "terminal"),
        "future_seed_update": ("fixed", "fixed"),
        "future_seed_norm_mode": ("unit", "unit"),
        "loop_update_mode": ("fixed", "fixed"),
        "loop_loss": ("all", "final"),
        "cell_order_train": ("random", "row_major"),
        "seed": (52, 52),
    }
    mismatches = {
        name: {"expected": value, "actual": saved_value(saved_args, name, default)}
        for name, (value, default) in expected.items()
        if saved_value(saved_args, name, default) != value
    }
    if saved_at_step != 3000:
        mismatches["saved_at_step"] = {"expected": 3000, "actual": saved_at_step}
    if mismatches:
        raise RuntimeError(f"parent semantic contract mismatch: {mismatches}")


def validate_gpu(expected_uuid: str) -> dict[str, Any]:
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "0":
        raise RuntimeError("CUDA_VISIBLE_DEVICES must be exactly 0")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("exactly one CUDA device is required; CPU fallback is forbidden")
    rows = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=index,uuid,name,memory.total",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    ).strip().splitlines()
    if len(rows) != 1:
        raise RuntimeError(f"expected one visible nvidia-smi row, got {rows}")
    fields = [part.strip() for part in rows[0].split(",")]
    if fields[0] != "0" or fields[1] != expected_uuid:
        raise RuntimeError(f"unexpected GPU contract: {rows[0]}")
    return {
        "index": int(fields[0]),
        "uuid": fields[1],
        "name": fields[2],
        "memory_total_mib": int(fields[3]),
        "torch_name": torch.cuda.get_device_name(0),
    }


def parse_range(value: str) -> tuple[int, int]:
    lo_text, hi_text = value.split("-", 1)
    lo, hi = int(lo_text), int(hi_text)
    if lo < 1 or hi < lo:
        raise argparse.ArgumentTypeError(f"invalid blank range: {value}")
    return lo, hi


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Zero-parameter per-loop gradient conflict audit for FutureSeed/GDN2"
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--checkpoint-sha256", required=True)
    parser.add_argument("--parent-source-sha", required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--expected-gpu-uuid", required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--eval-seed", type=int, default=52051)
    parser.add_argument("--blank-ranges", type=parse_range, nargs="+", default=[(51, 55), (56, 60), (61, 64)])
    parser.add_argument("--blank-weight", type=float, default=8.0)
    args = parser.parse_args()

    started = time.perf_counter()
    gpu = validate_gpu(args.expected_gpu_uuid)
    if study.GAIN_BUDGET_FLA_SHA != EXPECTED_FLA_SHA:
        raise RuntimeError(
            f"pinned FLA constant drift: {study.GAIN_BUDGET_FLA_SHA} != {EXPECTED_FLA_SHA}"
        )
    checkpoint_hash = sha256_file(args.checkpoint)
    if checkpoint_hash != args.checkpoint_sha256:
        raise RuntimeError(
            f"checkpoint SHA mismatch: {checkpoint_hash} != {args.checkpoint_sha256}"
        )

    checkpoint_run = args.checkpoint.resolve().parents[1].name
    persistent_root = Path(os.environ.get("PERSIST_ROOT", "/huyang2/double-loop"))
    source_head_path = persistent_root / "runs" / checkpoint_run / "source_HEAD.txt"
    source_patch_path = persistent_root / "runs" / checkpoint_run / "source.patch"
    if source_head_path.read_text(encoding="utf-8").strip() != args.parent_source_sha:
        raise RuntimeError("parent source_HEAD.txt does not match the registered source SHA")
    if not source_patch_path.is_file() or source_patch_path.stat().st_size != 0:
        raise RuntimeError("parent source.patch is missing or nonempty")

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False, mmap=True)
    saved_args = dict(checkpoint["args"])
    validate_parent(saved_args, int(checkpoint.get("saved_at_step", -1)))
    study.configure_sudoku(int(saved_args["size"]), 0, 0)

    device = torch.device("cuda", 0)
    model = build_model(saved_args).to(device)
    missing, unexpected = model.load_state_dict(checkpoint["model"], strict=False)
    if missing or unexpected:
        raise RuntimeError(
            f"checkpoint is not exact for this source: missing={missing}, unexpected={unexpected}"
        )
    del checkpoint
    model.train()
    runtime = study.strict_fla_runtime_summary(model, "gdn2")
    if runtime["fla_source_sha"] != EXPECTED_FLA_SHA:
        raise RuntimeError("runtime FLA SHA drift")
    if len(runtime["layers"]) != 12:
        raise RuntimeError(f"expected 12 official layers, got {len(runtime['layers'])}")

    parameters, parameter_groups, parameter_summary = group_parameters(model)
    dataset = study.OfficialSudokuDataset(args.data_dir, "test")
    bucket_rows: dict[str, Any] = {}
    torch.cuda.reset_peak_memory_stats(device)

    for bucket_idx, (holes_min, holes_max) in enumerate(args.blank_ranges):
        bucket_name = f"{holes_min}-{holes_max}"
        inputs, labels, clue_mask = dataset.fixed_batch_by_blank_range(
            args.batch_size,
            args.eval_seed + bucket_idx * 1000,
            holes_min=holes_min,
            holes_max=holes_max,
            device=device,
        )
        bucket_started = time.perf_counter()
        with study.forward_autocast("bfloat16", device):
            loop_logits, _trace = model.forward_trace(
                inputs,
                loops=5,
                noise_scale=0.0,
            )
        graph = graph_names(loop_logits[-1])
        backward_count = graph.count("ChunkGDN2FunctionBackward")
        if backward_count < 12:
            raise RuntimeError(
                f"official GDN2 backward path missing: count={backward_count}"
            )

        losses = [
            study.loss_from_logits(
                logits,
                labels,
                clue_mask,
                blank_weight=args.blank_weight,
            )
            for logits in loop_logits
        ]
        vectors_by_group: dict[str, list[torch.Tensor]] = {
            group: [] for group in GROUPS
        }
        gradient_rows: list[dict[str, Any]] = []
        for loop_idx, loss in enumerate(losses):
            grads = torch.autograd.grad(
                loss,
                parameters,
                retain_graph=loop_idx < len(losses) - 1,
                allow_unused=True,
            )
            vectors, diagnostics = grouped_gradient_vectors(
                grads,
                parameters,
                parameter_groups,
            )
            for group in GROUPS:
                vectors_by_group[group].append(vectors[group])
            gradient_rows.append(diagnostics)

        loop_metrics: list[dict[str, Any]] = []
        for loop_idx, (logits, loss) in enumerate(zip(loop_logits, losses), start=1):
            metrics, prediction = study.metrics_from_logits(logits, labels, clue_mask)
            blank_mask = ~clue_mask
            wrong_cells = ((prediction != labels) & blank_mask).sum(dim=1).float()
            loop_metrics.append(
                {
                    "loop": loop_idx,
                    "ce": float(loss.detach().item()),
                    **asdict(metrics),
                    "wrong_blank_cells_mean": float(wrong_cells.mean().item()),
                    "wrong_blank_cells_max": int(wrong_cells.max().item()),
                }
            )
        group_stats = {
            group: summarize_group(vectors_by_group[group]) for group in GROUPS
        }
        group_admissions = {
            group: group_admitted(group, group_stats[group])
            for group in MECHANISM_GROUPS
        }
        late_correction = bool(
            loop_metrics[-1]["blank_acc"] - loop_metrics[0]["blank_acc"] >= 0.02
            and loop_metrics[-1]["wrong_blank_cells_mean"]
            < loop_metrics[0]["wrong_blank_cells_mean"]
        )
        torch.cuda.synchronize(device)
        bucket_rows[bucket_name] = {
            "blank_range": [holes_min, holes_max],
            "batch_size": int(inputs.shape[0]),
            "official_backward_count": backward_count,
            "loop_metrics": loop_metrics,
            "gradient_loop_diagnostics": gradient_rows,
            "groups": group_stats,
            "group_admissions": group_admissions,
            "late_correction": late_correction,
            "elapsed_sec": time.perf_counter() - bucket_started,
        }
        del loop_logits, losses, vectors_by_group, grads, vectors
        del inputs, labels, clue_mask
        torch.cuda.empty_cache()

    group_bucket_counts = {
        group: sum(
            row["late_correction"] and row["group_admissions"][group]
            for row in bucket_rows.values()
        )
        for group in MECHANISM_GROUPS
    }
    admitted_groups = [
        group for group, count in group_bucket_counts.items() if count >= 2
    ]
    candidate_admitted = bool(admitted_groups)
    decision = {
        "candidate_admitted": candidate_admitted,
        "admitted_groups": admitted_groups,
        "qualified_bucket_counts": group_bucket_counts,
        "required_qualified_buckets_per_group": 2,
        "quality_interpretation": (
            "Register one contemporaneous-control gradient-coordination candidate "
            "on the admitted parameter group only."
            if candidate_admitted
            else "Close loop-gradient coordination; do not run a 100-step candidate."
        ),
        "thresholds": {
            "early_vs_late_cosine_max": GROUP_EARLY_LATE_FLOORS,
            "all_pair_conflict_fraction_min": 0.30,
            "cancellation_ratio_max": 0.75,
            "blank_accuracy_loop5_minus_loop1_min": 0.02,
        },
    }
    current_head = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], text=True
    ).strip()
    payload = {
        "plan_id": "P-LOOP-001",
        "kind": "zero_parameter_gradient_diagnostic",
        "source_sha": current_head,
        "source_clean": subprocess.check_output(
            ["git", "status", "--porcelain"], text=True
        ).strip()
        == "",
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": checkpoint_hash,
        "parent_source_sha": args.parent_source_sha,
        "data_dir": str(args.data_dir),
        "gpu": gpu,
        "torch_version": torch.__version__,
        "runtime": runtime,
        "parameter_groups": parameter_summary,
        "buckets": bucket_rows,
        "decision": decision,
        "peak_allocated_mib": torch.cuda.max_memory_allocated(device) / (1024.0**2),
        "elapsed_sec": time.perf_counter() - started,
    }
    if not payload["source_clean"]:
        raise RuntimeError("diagnostic source worktree became dirty")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "out": str(args.out),
        "decision": decision,
        "elapsed_sec": payload["elapsed_sec"],
        "peak_allocated_mib": payload["peak_allocated_mib"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
