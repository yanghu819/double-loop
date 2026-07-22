#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict

import torch
import torch.nn.functional as F


REPO_ROOT = Path(__file__).resolve().parents[1]
RUNNER_DIR = REPO_ROOT / "experiments" / "rwkv_fs_sudoku"
sys.path.insert(0, str(RUNNER_DIR))

from study_rwkv_futureseed_loop import (  # noqa: E402
    FutureSeedLoopSudoku,
    OfficialSudokuDataset,
    configure_sudoku,
    forward_autocast,
)


def _value(args: Dict[str, Any], name: str, default: Any) -> Any:
    value = args.get(name, default)
    return default if value is None else value


def build_model(saved_args: Dict[str, Any]) -> FutureSeedLoopSudoku:
    return FutureSeedLoopSudoku(
        d_model=int(saved_args["d_model"]),
        layers=int(saved_args["layers"]),
        heads=int(saved_args["heads"]),
        head_dim=int(saved_args["head_dim"]),
        channel_mult=int(saved_args["channel_mult"]),
        l_cycles=int(saved_args["l_cycles"]),
        max_loops=int(saved_args["max_loops"]),
        lambda_=float(saved_args["lambda_"]),
        loop_update_mode=str(_value(saved_args, "loop_update_mode", "fixed")),
        loop_update_gate_init=float(_value(saved_args, "loop_update_gate_init", 0.95)),
        future_seed_scale=float(_value(saved_args, "future_seed_scale", 1.0)),
        future_seed_decay=float(_value(saved_args, "future_seed_decay", 0.0)),
        future_seed_update=str(_value(saved_args, "future_seed_update", "fixed")),
        future_seed_norm_mode=str(_value(saved_args, "future_seed_norm_mode", "unit")),
        loop_feedback_scale=float(_value(saved_args, "loop_feedback_scale", 0.0)),
        loop_feedback_detach=bool(_value(saved_args, "loop_feedback_detach", False)),
        loop_feedback_corrupt_prob=float(_value(saved_args, "loop_feedback_corrupt_prob", 0.0)),
        loop_feedback_corrupt_mix=float(_value(saved_args, "loop_feedback_corrupt_mix", 0.0)),
        loop_feedback_corrupt_mode=str(_value(saved_args, "loop_feedback_corrupt_mode", "random_token")),
        loop_time_scale=float(_value(saved_args, "loop_time_scale", 0.0)),
        scratch_mode=str(_value(saved_args, "scratch_mode", "none")),
        scratch_scale=float(_value(saved_args, "scratch_scale", 1.0)),
        scratch_noise_scale=float(_value(saved_args, "scratch_noise_scale", 0.0)),
        scratch_gauss_projections=int(_value(saved_args, "scratch_gauss_projections", 0)),
        scratch_gate_bias=float(_value(saved_args, "scratch_gate_bias", -2.0)),
        scratch_decay_bias=float(_value(saved_args, "scratch_decay_bias", 2.0)),
        hidden_agg_noise_scale=float(_value(saved_args, "hidden_agg_noise_scale", 0.0)),
        hidden_agg_noise_temp=float(_value(saved_args, "hidden_agg_noise_temp", 1.0)),
        hidden_agg_noise_detach=bool(_value(saved_args, "hidden_agg_noise_detach", True)),
        hidden_agg_noise_mode=str(_value(saved_args, "hidden_agg_noise_mode", "gumbel")),
        hidden_agg_noise_topk=int(_value(saved_args, "hidden_agg_noise_topk", 8)),
        hidden_agg_noise_max_norm=float(_value(saved_args, "hidden_agg_noise_max_norm", 0.0)),
        activation_checkpoint=False,
        rwkv_kernel=str(_value(saved_args, "rwkv_kernel", "auto")),
        backbone=str(saved_args["backbone"]),
        gdn_mode=str(saved_args["gdn_mode"]),
        gdn_expand_v=float(saved_args["gdn_expand_v"]),
        gdn_use_short_conv=bool(saved_args["gdn_use_short_conv"]),
        gdn_conv_size=int(_value(saved_args, "gdn_conv_size", 4)),
        gdn_allow_neg_eigval=bool(_value(saved_args, "gdn_allow_neg_eigval", False)),
    )


def _run_logits(
    checkpoint: Dict[str, Any],
    inputs: torch.Tensor,
    *,
    loops: int,
    forward_dtype: str,
    device: torch.device,
) -> tuple[list[torch.Tensor], float]:
    model = build_model(checkpoint["args"]).to(device)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.eval()
    torch.cuda.reset_peak_memory_stats(device)
    with torch.no_grad(), forward_autocast(forward_dtype, device):
        logits, _trace = model.forward_trace(inputs, loops=loops, noise_scale=0.0)
    outputs = [item.detach().float().cpu() for item in logits]
    peak_mb = torch.cuda.max_memory_allocated(device) / (1024.0**2)
    del model
    torch.cuda.empty_cache()
    return outputs, peak_mb


def _new_branch_grad_norm(model: FutureSeedLoopSudoku, old_expand_v: float) -> float:
    total = 0.0
    heads = model.reasoner.blocks[0].time_mix.heads
    old_head_v = int(model.reasoner.blocks[0].time_mix.head_dim * old_expand_v)
    for block in model.reasoner.blocks:
        grad = block.time_mix.o_proj.weight.grad
        if grad is None:
            raise RuntimeError("Expanded o_proj did not receive a gradient")
        shaped = grad.reshape(grad.shape[0], heads, -1)
        total += float(shaped[:, :, old_head_v:].float().square().sum().cpu())
    return math.sqrt(total)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CUDA equivalence and backward gate for expanded GDN checkpoints")
    parser.add_argument("--original", type=Path, required=True)
    parser.add_argument("--expanded", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--loops", type=int, default=5)
    parser.add_argument("--holes-min", type=int, default=56)
    parser.add_argument("--holes-max", type=int, default=64)
    parser.add_argument("--seed", type=int, default=52035)
    parser.add_argument("--forward-dtype", choices=("float32", "bfloat16"), default="bfloat16")
    parser.add_argument("--max-abs-tol", type=float, default=0.04)
    parser.add_argument("--rms-tol", type=float, default=0.004)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is mandatory; CPU fallback is forbidden")
    device = torch.device("cuda:0")
    if torch.cuda.device_count() != 1:
        raise RuntimeError(f"Expected exactly one visible GPU, got {torch.cuda.device_count()}")

    original = torch.load(args.original, map_location="cpu", weights_only=False, mmap=True)
    expanded = torch.load(args.expanded, map_location="cpu", weights_only=False, mmap=True)
    old_expand_v = float(original["args"]["gdn_expand_v"])
    new_expand_v = float(expanded["args"]["gdn_expand_v"])
    transform = expanded.get("checkpoint_transform", {})
    if transform.get("type") != "function_preserving_gdn_value_state_expansion":
        raise RuntimeError("Expanded checkpoint lacks the expected transform provenance")
    if int(original.get("saved_at_step", -1)) != int(expanded.get("saved_at_step", -2)):
        raise RuntimeError("Expansion changed the global training step")

    configure_sudoku(9, 3, 3)
    dataset = OfficialSudokuDataset(args.data_dir, "test")
    inputs, labels, clue_mask = dataset.fixed_batch_by_blank_range(
        args.batch_size,
        args.seed,
        holes_min=args.holes_min,
        holes_max=args.holes_max,
        device=device,
    )
    original_logits, original_peak_mb = _run_logits(
        original,
        inputs,
        loops=args.loops,
        forward_dtype=args.forward_dtype,
        device=device,
    )
    expanded_logits, expanded_peak_mb = _run_logits(
        expanded,
        inputs,
        loops=args.loops,
        forward_dtype=args.forward_dtype,
        device=device,
    )

    loop_diagnostics = []
    mismatch_total = 0
    max_abs = 0.0
    max_rms = 0.0
    for loop_idx, (before, after) in enumerate(zip(original_logits, expanded_logits), start=1):
        delta = after - before
        loop_max_abs = float(delta.abs().max())
        loop_rms = float(delta.square().mean().sqrt())
        mismatches = int(before.argmax(dim=-1).ne(after.argmax(dim=-1)).sum())
        mismatch_total += mismatches
        max_abs = max(max_abs, loop_max_abs)
        max_rms = max(max_rms, loop_rms)
        loop_diagnostics.append(
            {
                "loop": loop_idx,
                "max_abs_logit_delta": loop_max_abs,
                "rms_logit_delta": loop_rms,
                "prediction_mismatches": mismatches,
            }
        )

    model = build_model(expanded["args"]).to(device)
    model.load_state_dict(expanded["model"], strict=True)
    model.train()
    with forward_autocast(args.forward_dtype, device):
        logits, _trace = model.forward_trace(inputs, loops=args.loops, noise_scale=0.0)
        losses = [F.cross_entropy(item.reshape(-1, item.shape[-1]), labels.reshape(-1)) for item in logits]
        loss = torch.stack(losses).mean()
    loss.backward()
    new_branch_grad_norm = _new_branch_grad_norm(model, old_expand_v)
    finite_grads = all(
        parameter.grad is None or bool(torch.isfinite(parameter.grad).all()) for parameter in model.parameters()
    )
    backward_peak_mb = torch.cuda.max_memory_allocated(device) / (1024.0**2)

    passed = (
        mismatch_total == 0
        and max_abs <= args.max_abs_tol
        and max_rms <= args.rms_tol
        and finite_grads
        and new_branch_grad_norm > 0.0
    )
    result = {
        "passed": passed,
        "device": torch.cuda.get_device_name(device),
        "visible_gpu_count": torch.cuda.device_count(),
        "saved_at_step": int(original["saved_at_step"]),
        "old_expand_v": old_expand_v,
        "new_expand_v": new_expand_v,
        "forward_dtype": args.forward_dtype,
        "batch_size": args.batch_size,
        "loops": args.loops,
        "loop_diagnostics": loop_diagnostics,
        "max_abs_logit_delta": max_abs,
        "max_rms_logit_delta": max_rms,
        "prediction_mismatches": mismatch_total,
        "new_branch_o_proj_grad_norm": new_branch_grad_norm,
        "finite_grads": finite_grads,
        "original_eval_peak_mb": original_peak_mb,
        "expanded_eval_peak_mb": expanded_peak_mb,
        "expanded_backward_peak_mb": backward_peak_mb,
        "tolerances": {"max_abs": args.max_abs_tol, "rms": args.rms_tol},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
