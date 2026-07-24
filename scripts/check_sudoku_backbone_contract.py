#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F


BACKBONES = {
    "rwkv": "rwkv7",
    "gdn": "fla_gdn",
    "gdn2": "gdn2",
    "kda": "kda",
}


def load_runner(repo: Path):
    experiment_dir = repo / "experiments" / "rwkv_fs_sudoku"
    sys.path.insert(0, str(experiment_dir))
    import study_rwkv_futureseed_loop as runner

    return runner


def build_model(runner, backbone: str):
    return runner.FutureSeedLoopSudoku(
        d_model=192,
        layers=10,
        heads=6,
        head_dim=32,
        channel_mult=4,
        l_cycles=2,
        max_loops=5,
        lambda_=0.95,
        loop_update_mode="fixed",
        loop_update_gate_init=0.95,
        future_seed_scale=1.0,
        future_seed_decay=0.0,
        future_seed_update="fixed",
        future_seed_norm_mode="unit",
        loop_feedback_scale=0.0,
        loop_feedback_detach=False,
        loop_feedback_corrupt_prob=0.0,
        loop_feedback_corrupt_mix=0.0,
        loop_feedback_corrupt_mode="random_token",
        loop_time_scale=0.0,
        scratch_mode="none",
        scratch_scale=1.0,
        scratch_noise_scale=0.0,
        scratch_gauss_projections=0,
        scratch_gate_bias=-2.0,
        scratch_decay_bias=2.0,
        hidden_agg_noise_scale=0.0,
        hidden_agg_noise_temp=1.0,
        hidden_agg_noise_detach=True,
        hidden_agg_noise_mode="gumbel",
        hidden_agg_noise_topk=8,
        hidden_agg_noise_max_norm=0.0,
        activation_checkpoint=False,
        rwkv_kernel="statepassing",
        backbone=BACKBONES[backbone],
        gdn_mode="chunk",
        gdn_expand_v=1.0,
        gdn_progressive_base_expand_v=0.0,
        gdn_use_short_conv=True,
        gdn_conv_size=4,
        gdn_allow_neg_eigval=False,
    )


def finite_gradient_summary(model: torch.nn.Module) -> dict[str, Any]:
    missing: list[str] = []
    nonfinite: list[str] = []
    squared_norm = 0.0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if parameter.grad is None:
            missing.append(name)
            continue
        gradient = parameter.grad.detach().float()
        if not bool(torch.isfinite(gradient).all()):
            nonfinite.append(name)
        squared_norm += float(gradient.square().sum().item())
    expected_missing = {"reasoner.blocks.0.future_seed_logit"}
    unexpected_missing = sorted(set(missing) - expected_missing)
    return {
        "global_norm": squared_norm**0.5,
        "unexpected_missing": unexpected_missing,
        "expected_missing": sorted(set(missing) & expected_missing),
        "nonfinite": nonfinite,
    }


def check_backbone(runner, public_name: str, device: torch.device) -> dict[str, Any]:
    torch.manual_seed(52)
    torch.cuda.manual_seed_all(52)
    model = build_model(runner, public_name).to(device).train()
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    trainable_count = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    internal_name = BACKBONES[public_name]

    if public_name == "rwkv":
        available, reason = runner.statepassing_available(32)
        if not available:
            raise RuntimeError(f"RWKV state-passing CUDA kernel is unavailable: {reason}")
        runtime = {
            "implementation": "official_rwkv7_timemix_with_explicit_state_io",
            "kernel": "statepassing",
            "available": available,
            "reason": reason,
            "official_source_commit": runner.RWKV7_OFFICIAL_SOURCE_COMMIT,
            "official_source_blob": runner.RWKV7_OFFICIAL_SOURCE_BLOB,
            "official_kernel_blob": runner.RWKV7_OFFICIAL_KERNEL_BLOB,
            "statepassing_cuda_sha256": runner.RWKV7_STATEPASSING_CUDA_SHA256,
        }
    else:
        runtime = runner.strict_fla_runtime_summary(model, internal_name)

    inputs = torch.randint(0, 10, (2, 81), device=device)
    labels = torch.randint(0, 9, (2, 81), device=device)
    torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
        loop_logits, trace = model.forward_trace(inputs, loops=2, noise_scale=0.0)
        loss = torch.stack(
            [F.cross_entropy(logits.float().reshape(-1, 9), labels.reshape(-1)) for logits in loop_logits]
        ).mean()
    loss.backward()
    torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started
    gradients = finite_gradient_summary(model)
    if gradients["unexpected_missing"] or gradients["nonfinite"]:
        raise AssertionError(f"{public_name} gradient gate failed: {gradients}")

    optimizer, optimizer_runtime = runner.build_adamw(
        model,
        lr=0.0015,
        weight_decay=0.001,
        contract="rwkv7_decay_groups",
    )
    optimizer.zero_grad(set_to_none=True)
    row = {
        "public_name": public_name,
        "internal_backbone": internal_name,
        "parameter_count": parameter_count,
        "trainable_parameter_count": trainable_count,
        "loss": float(loss.detach().cpu()),
        "output_shape": list(loop_logits[-1].shape),
        "fs_gate_mean": float(trace[-1]["fs_gate_mean"].detach().cpu()),
        "fs_state_norm": float(trace[-1]["fs_state_norm"].detach().cpu()),
        "gradient": gradients,
        "smoke_wall_sec": elapsed,
        "peak_allocated_mib": torch.cuda.max_memory_allocated(device) / (1024**2),
        "runtime": runtime,
        "optimizer_runtime": optimizer_runtime,
    }
    del model, loop_logits, trace, loss
    torch.cuda.empty_cache()
    return row


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    if os.environ.get("CUDA_VISIBLE_DEVICES") != "0":
        raise RuntimeError("Set CUDA_VISIBLE_DEVICES=0; this preflight is GPU1-only.")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required; CPU model smoke is forbidden.")

    repo = args.repo.resolve()
    runner = load_runner(repo)
    runner.configure_sudoku(9, 0, 0)
    device = torch.device("cuda", 0)
    payload: dict[str, Any] = {
        "device": torch.cuda.get_device_name(device),
        "torch": torch.__version__,
        "cuda_visible_devices": os.environ["CUDA_VISIBLE_DEVICES"],
        "contract": {
            "d_model": 192,
            "layers": 10,
            "heads": 6,
            "head_dim": 32,
            "gdn_expand_v": 1.0,
            "loops_smoke": 2,
            "future_seed_scale": 1.0,
        },
        "backbones": {},
    }
    for public_name in BACKBONES:
        print(f"[gpu1-contract] start {public_name}", flush=True)
        payload["backbones"][public_name] = check_backbone(runner, public_name, device)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(f"[gpu1-contract] pass {public_name}", flush=True)
    print(json.dumps(payload, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
