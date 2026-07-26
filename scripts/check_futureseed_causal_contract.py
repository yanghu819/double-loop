#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
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
FS_RUNS = {
    "rwkv": "sudoku-backbone-rwkv-s500-sharedinit-20260724T154020Z-6e51f06",
    "gdn": "sudoku-backbone-gdn-s500-sharedinit-20260724T154020Z-6e51f06",
    "gdn2": "sudoku-backbone-gdn2-s500-sharedinit-20260724T154020Z-6e51f06",
    "kda": "sudoku-backbone-kda-s500-sharedinit-20260724T154020Z-6e51f06",
}


def git_output(repo: Path, *args: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(repo), *args],
        text=True,
    ).strip()


def load_runner(repo: Path):
    experiment_dir = repo / "experiments" / "rwkv_fs_sudoku"
    sys.path.insert(0, str(experiment_dir))
    import study_rwkv_futureseed_loop as runner

    return runner


def build_model(runner, public_name: str, future_seed_scale: float):
    model = runner.FutureSeedLoopSudoku(
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
        future_seed_scale=future_seed_scale,
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
        backbone=BACKBONES[public_name],
        gdn_mode="chunk",
        gdn_expand_v=1.0,
        gdn_progressive_base_expand_v=0.0,
        gdn_use_short_conv=True,
        gdn_conv_size=4,
        gdn_allow_neg_eigval=False,
    )
    model.reset_shared_shell_parameters(52)
    return model


def tensor_sha256(tensor: torch.Tensor) -> str:
    value = tensor.detach().contiguous().cpu()
    return hashlib.sha256(value.numpy().tobytes()).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parameter_hashes(model: torch.nn.Module) -> dict[str, str]:
    return {
        name: tensor_sha256(parameter)
        for name, parameter in model.named_parameters()
    }


def parameter_fingerprint(hashes: dict[str, str]) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(hashes.items()):
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(value.encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def runtime_summary(runner, model: torch.nn.Module, public_name: str) -> dict[str, Any]:
    if public_name == "rwkv":
        available, reason = runner.statepassing_available(32)
        if not available:
            raise RuntimeError(f"RWKV state-passing CUDA kernel is unavailable: {reason}")
        return {
            "implementation": "official_rwkv7_timemix_with_explicit_state_io",
            "kernel": "statepassing",
            "official_source_commit": runner.RWKV7_OFFICIAL_SOURCE_COMMIT,
            "official_source_blob": runner.RWKV7_OFFICIAL_SOURCE_BLOB,
            "official_kernel_blob": runner.RWKV7_OFFICIAL_KERNEL_BLOB,
            "statepassing_cuda_sha256": runner.RWKV7_STATEPASSING_CUDA_SHA256,
        }
    return runner.strict_fla_runtime_summary(model, BACKBONES[public_name])


def run_arm(
    runner,
    public_name: str,
    future_seed_scale: float,
    checkpoint_path: Path,
    inputs: torch.Tensor,
    labels: torch.Tensor,
) -> dict[str, Any]:
    torch.manual_seed(52)
    torch.cuda.manual_seed_all(52)
    model = build_model(runner, public_name, future_seed_scale)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if int(checkpoint.get("saved_at_step", -1)) != 500:
        raise AssertionError(f"{public_name} functional checkpoint is not step 500")
    model.load_state_dict(checkpoint["model"], strict=True)
    del checkpoint
    model = model.to(inputs.device).train()
    hashes = parameter_hashes(model)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    runtime = runtime_summary(runner, model, public_name)
    with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
        loop_logits, trace = model.forward_trace(inputs, loops=2, noise_scale=0.0)
        loss = torch.stack(
            [
                F.cross_entropy(logits.float().reshape(-1, 9), labels.reshape(-1))
                for logits in loop_logits
            ]
        ).mean()
    loss.backward()
    torch.cuda.synchronize(inputs.device)
    logits = loop_logits[-1].detach().float()
    future_seed_grad_sq = 0.0
    backbone_grad_sq = 0.0
    future_seed_grad_names = []
    for name, parameter in model.named_parameters():
        if parameter.grad is None:
            continue
        gradient = parameter.grad.detach().float()
        if not bool(torch.isfinite(gradient).all()):
            raise AssertionError(f"{public_name} scale={future_seed_scale} nonfinite gradient: {name}")
        value = float(gradient.square().sum().cpu())
        if name.endswith("future_seed_logit"):
            future_seed_grad_sq += value
            if value > 0:
                future_seed_grad_names.append(name)
        else:
            backbone_grad_sq += value
    row = {
        "future_seed_scale": future_seed_scale,
        "parameter_count": parameter_count,
        "parameter_hashes": hashes,
        "output_sha256": tensor_sha256(logits),
        "output_rms": float(logits.square().mean().sqrt().cpu()),
        "fs_gate_mean": float(trace[-1]["fs_gate_mean"].float().cpu()),
        "fs_state_norm": float(trace[-1]["fs_state_norm"].float().cpu()),
        "loss": float(loss.detach().float().cpu()),
        "future_seed_grad_norm": future_seed_grad_sq**0.5,
        "future_seed_grad_names": future_seed_grad_names,
        "backbone_grad_norm": backbone_grad_sq**0.5,
        "runtime": runtime,
        "_output": logits.cpu(),
    }
    del model, loop_logits, trace, logits, loss
    torch.cuda.empty_cache()
    return row


def check_backbone(
    runner,
    public_name: str,
    checkpoint_path: Path,
    inputs: torch.Tensor,
    labels: torch.Tensor,
) -> dict[str, Any]:
    torch.manual_seed(52)
    torch.cuda.manual_seed_all(52)
    initial_with = parameter_hashes(build_model(runner, public_name, 1.0))
    torch.manual_seed(52)
    torch.cuda.manual_seed_all(52)
    initial_without = parameter_hashes(build_model(runner, public_name, 0.0))
    if initial_with != initial_without:
        differing = sorted(
            name
            for name in set(initial_with) | set(initial_without)
            if initial_with.get(name) != initial_without.get(name)
        )
        raise AssertionError(f"{public_name} initialization differs with scale: {differing}")
    initial_parameter_fingerprint = parameter_fingerprint(initial_with)
    initial_parameter_tensor_count = len(initial_with)
    del initial_with, initial_without

    with_future = run_arm(
        runner,
        public_name,
        1.0,
        checkpoint_path,
        inputs,
        labels,
    )
    without_future = run_arm(
        runner,
        public_name,
        0.0,
        checkpoint_path,
        inputs,
        labels,
    )
    expected_future_seed_grad_names = [
        f"reasoner.blocks.{layer_idx}.future_seed_logit"
        for layer_idx in range(1, 10)
    ]
    if with_future["parameter_hashes"] != without_future["parameter_hashes"]:
        differing = sorted(
            name
            for name in set(with_future["parameter_hashes"])
            | set(without_future["parameter_hashes"])
            if with_future["parameter_hashes"].get(name)
            != without_future["parameter_hashes"].get(name)
        )
        raise AssertionError(f"{public_name} trained checkpoint differs with scale: {differing}")
    if with_future["parameter_count"] != without_future["parameter_count"]:
        raise AssertionError(f"{public_name} parameter count changed with FutureSeed")
    if with_future["runtime"] != without_future["runtime"]:
        raise AssertionError(f"{public_name} runtime identity changed with FutureSeed")
    if with_future["fs_gate_mean"] <= 0 or with_future["fs_state_norm"] <= 0:
        raise AssertionError(f"{public_name} FutureSeed-on path did not inject state")
    if abs(without_future["fs_gate_mean"]) > 1e-8:
        raise AssertionError(f"{public_name} FutureSeed-off gate is nonzero")
    if abs(without_future["fs_state_norm"]) > 1e-8:
        raise AssertionError(f"{public_name} FutureSeed-off state is nonzero")
    if with_future["future_seed_grad_norm"] <= 0:
        raise AssertionError(f"{public_name} FutureSeed-on gate did not receive gradients")
    if with_future["future_seed_grad_names"] != expected_future_seed_grad_names:
        raise AssertionError(
            f"{public_name} FutureSeed-on gradient paths differ: "
            f"{with_future['future_seed_grad_names']} != {expected_future_seed_grad_names}"
        )
    if without_future["future_seed_grad_norm"] > 1e-12:
        raise AssertionError(f"{public_name} FutureSeed-off gate received gradients")
    if without_future["future_seed_grad_names"]:
        raise AssertionError(
            f"{public_name} FutureSeed-off has active gate gradients: "
            f"{without_future['future_seed_grad_names']}"
        )
    if with_future["backbone_grad_norm"] <= 0 or without_future["backbone_grad_norm"] <= 0:
        raise AssertionError(f"{public_name} backbone gradient path is inactive")
    output_rms_delta = float(
        (with_future.pop("_output") - without_future.pop("_output"))
        .square()
        .mean()
        .sqrt()
    )
    if output_rms_delta <= 1e-7:
        raise AssertionError(f"{public_name} FutureSeed intervention did not change output")
    trained_parameter_hashes = with_future.pop("parameter_hashes")
    without_future.pop("parameter_hashes")
    return {
        "status": "PASS",
        "identical_initial_parameter_tensors": True,
        "parameter_tensor_count": initial_parameter_tensor_count,
        "initial_parameter_fingerprint": initial_parameter_fingerprint,
        "identical_functional_checkpoint_tensors": True,
        "functional_checkpoint": str(checkpoint_path),
        "functional_checkpoint_sha256": file_sha256(checkpoint_path),
        "functional_parameter_fingerprint": parameter_fingerprint(trained_parameter_hashes),
        "output_rms_delta": output_rms_delta,
        "with_future": with_future,
        "without_future": without_future,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runner-repo", type=Path, required=True)
    parser.add_argument("--models-root", type=Path, required=True)
    parser.add_argument("--expected-sha", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    if os.environ.get("CUDA_VISIBLE_DEVICES") != "0":
        raise RuntimeError("Set CUDA_VISIBLE_DEVICES=0; this check is GPU1-only.")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required; CPU model smoke is forbidden.")

    repo = args.runner_repo.resolve()
    git_sha = git_output(repo, "rev-parse", "HEAD")
    if git_sha != args.expected_sha:
        raise AssertionError(f"Wrong source SHA: {git_sha} != {args.expected_sha}")
    if git_output(repo, "status", "--porcelain"):
        raise AssertionError("Runner source is dirty")

    runner = load_runner(repo)
    runner.configure_sudoku(9, 0, 0)
    device = torch.device("cuda", 0)
    torch.manual_seed(2052)
    torch.cuda.manual_seed_all(2052)
    inputs = torch.randint(0, 10, (2, 81), device=device)
    labels = torch.randint(0, 9, (2, 81), device=device)
    payload: dict[str, Any] = {
        "status": "IN_PROGRESS",
        "source_sha": git_sha,
        "source_dirty": False,
        "device": torch.cuda.get_device_name(device),
        "cuda_visible_devices": os.environ["CUDA_VISIBLE_DEVICES"],
        "definition": (
            "FutureSeed on/off constructors use byte-identical initial parameters. "
            "The functional check loads the same accepted step-500 FutureSeed checkpoint "
            "into both arms; only future_seed_scale changes recurrent initial-state injection."
        ),
        "backbones": {},
    }
    for public_name in BACKBONES:
        print(f"[futureseed-causal-contract] start {public_name}", flush=True)
        checkpoint_path = (
            args.models_root.resolve()
            / FS_RUNS[public_name]
            / "checkpoints"
            / "train_state_step000500.pt"
        )
        if not checkpoint_path.is_file():
            raise FileNotFoundError(checkpoint_path)
        payload["backbones"][public_name] = check_backbone(
            runner,
            public_name,
            checkpoint_path,
            inputs,
            labels,
        )
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(f"[futureseed-causal-contract] pass {public_name}", flush=True)
    payload["status"] = "PASS"
    args.out.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
