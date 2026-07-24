#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
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
        raise RuntimeError(f"{cache_var} must point below /huyang2/double-loop before importing CUDA runtimes")

import torch
import torch.nn.functional as F


def load_runner(repo: Path):
    experiment_dir = repo / "experiments" / "rwkv_fs_sudoku"
    sys.path.insert(0, str(experiment_dir))
    import study_rwkv_futureseed_loop as runner

    return runner


def tensor_digest(tensor: torch.Tensor) -> str:
    value = tensor.detach().to(device="cpu", dtype=torch.float32).contiguous().numpy()
    return hashlib.sha256(value.tobytes()).hexdigest()


def autograd_graph_names(tensor: torch.Tensor) -> list[str]:
    names: list[str] = []
    queue = [tensor.grad_fn]
    seen: set[int] = set()
    while queue:
        fn = queue.pop(0)
        if fn is None or id(fn) in seen:
            continue
        seen.add(id(fn))
        names.append(type(fn).__name__)
        queue.extend(next_fn for next_fn, _index in fn.next_functions if next_fn is not None)
    return names


def build_model(runner, *, head_dim: int, expand_v: float):
    model = runner.FutureSeedLoopSudoku(
        d_model=192,
        layers=10,
        heads=6,
        head_dim=head_dim,
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
        backbone="fla_gdn",
        gdn_mode="chunk",
        gdn_expand_v=expand_v,
        gdn_progressive_base_expand_v=0.0,
        gdn_use_short_conv=True,
        gdn_conv_size=4,
        gdn_allow_neg_eigval=False,
    )
    model.reset_shared_shell_parameters(52)
    return model


def shared_shell_hashes(model: torch.nn.Module) -> dict[str, str]:
    return {
        name: tensor_digest(parameter)
        for name, parameter in model.named_parameters()
        if ".time_mix." not in name
    }


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
    return {
        "global_norm": squared_norm**0.5,
        "unexpected_missing": sorted(set(missing) - expected_missing),
        "expected_missing": sorted(set(missing) & expected_missing),
        "nonfinite": nonfinite,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--phase",
        choices=("reference", "full_stack", "all"),
        default="all",
    )
    args = parser.parse_args()

    if os.environ.get("CUDA_VISIBLE_DEVICES") != "0":
        raise RuntimeError("Set CUDA_VISIBLE_DEVICES=0; this preflight is GPU1-only")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required; CPU model execution is forbidden")
    if os.environ.get("FLA_DISABLE_BACKEND_DISPATCH") != "1":
        raise RuntimeError("FLA_DISABLE_BACKEND_DISPATCH=1 is required")
    if os.environ.get("FLA_CONV_BACKEND") != "triton":
        raise RuntimeError("FLA_CONV_BACKEND=triton is required")

    repo = args.repo.resolve()
    runner = load_runner(repo)
    runner.configure_sudoku(9, 0, 0)
    device = torch.device("cuda", 0)

    reference = None
    if args.phase in {"reference", "all"}:
        print("[native-gdn-gate] start K24/V48 numerical reference", flush=True)
        import check_fla_delta_backbones as kernel_checks

        reference = kernel_checks.check_gdn_reference(
            device,
            key_dim=24,
            value_dim=48,
            seed=109,
        )
        print("[native-gdn-gate] pass K24/V48 numerical reference", flush=True)
        if args.phase == "reference":
            payload = {
                "status": "PASS",
                "phase": "reference",
                "device": torch.cuda.get_device_name(device),
                "cuda_visible_devices": os.environ["CUDA_VISIBLE_DEVICES"],
                "contract": {
                    "heads": 2,
                    "head_dim_k": 24,
                    "head_dim_v": 48,
                    "state_layout": "VxK",
                    "official_chunk_autograd_node": "ChunkGatedDeltaRuleFunctionBackward",
                },
                "native_shape_reference": reference,
            }
            args.out.parent.mkdir(parents=True, exist_ok=True)
            args.out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            print(json.dumps(payload, indent=2, sort_keys=True))
            return

    print("[native-gdn-gate] start full-stack model construction", flush=True)
    torch.manual_seed(52)
    torch.cuda.manual_seed_all(52)
    matched = build_model(runner, head_dim=32, expand_v=1.0).to(device)
    torch.manual_seed(52)
    torch.cuda.manual_seed_all(52)
    native = build_model(runner, head_dim=24, expand_v=2.0).to(device).train()
    print("[native-gdn-gate] pass full-stack model construction", flush=True)

    print("[native-gdn-gate] start shared-shell hash comparison", flush=True)
    matched_shell = shared_shell_hashes(matched)
    native_shell = shared_shell_hashes(native)
    common_shell = sorted(set(matched_shell) & set(native_shell))
    differing_shell = [
        name for name in common_shell if matched_shell[name] != native_shell[name]
    ]
    if len(common_shell) != 77 or differing_shell:
        raise AssertionError(
            f"Native GDN changed the shared shell: common={len(common_shell)}, differing={differing_shell}"
        )
    print("[native-gdn-gate] pass shared-shell hash comparison", flush=True)

    print("[native-gdn-gate] start official runtime and geometry checks", flush=True)
    runtime = runner.strict_fla_runtime_summary(native, "fla_gdn")
    expected_class = "fla.layers.gated_deltanet.GatedDeltaNet"
    if {row["class"] for row in runtime["layers"]} != {expected_class}:
        raise AssertionError(f"Native GDN class mismatch: {runtime}")
    first_mix = native.reasoner.blocks[0].time_mix
    geometry = {
        "d_model": 192,
        "heads": first_mix.heads,
        "head_dim_k": first_mix.head_dim,
        "head_dim_v": first_mix.head_v_dim,
        "key_dim": int(first_mix.core.key_dim),
        "value_dim": int(first_mix.core.value_dim),
        "state_layout": "VxK" if first_mix.state_v_first else "KxV",
        "state_elements_per_sample_layer": first_mix.heads
        * first_mix.head_v_dim
        * first_mix.head_dim,
    }
    expected_geometry = {
        "d_model": 192,
        "heads": 6,
        "head_dim_k": 24,
        "head_dim_v": 48,
        "key_dim": 144,
        "value_dim": 288,
        "state_layout": "VxK",
        "state_elements_per_sample_layer": 6912,
    }
    if geometry != expected_geometry:
        raise AssertionError(f"Native GDN geometry mismatch: {geometry} != {expected_geometry}")
    print("[native-gdn-gate] pass official runtime and geometry checks", flush=True)

    print("[native-gdn-gate] start adapter initial-state backward", flush=True)
    h0 = torch.randn(
        2,
        6,
        48,
        24,
        device=device,
        dtype=torch.float32,
        requires_grad=True,
    ) * 0.05
    h0.retain_grad()
    mixer_input = torch.randn(2, 81, 192, device=device, requires_grad=True)
    with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
        mixer_output, terminal_state = first_mix(mixer_input, initial_state=h0)
        mixer_loss = (
            mixer_output.float().square().mean()
            + terminal_state.float().square().mean() * 1e-3
        )
    graph_names = autograd_graph_names(terminal_state)
    expected_node = "ChunkGatedDeltaRuleFunctionBackward"
    if expected_node not in graph_names:
        raise AssertionError(
            f"Native GDN did not use the official chunk backward: expected={expected_node}, "
            f"graph={graph_names}"
        )
    mixer_loss.backward()
    if h0.grad is None or not bool(torch.isfinite(h0.grad).all()) or h0.grad.norm() <= 0:
        raise AssertionError("Native GDN initial recurrent state did not receive a finite nonzero gradient")
    native.zero_grad(set_to_none=True)
    print("[native-gdn-gate] pass adapter initial-state backward", flush=True)

    print("[native-gdn-gate] start 10-layer two-loop full-stack backward", flush=True)
    inputs = torch.randint(0, 10, (2, 81), device=device)
    labels = torch.randint(0, 9, (2, 81), device=device)
    torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
        loop_logits, trace = native.forward_trace(inputs, loops=2, noise_scale=0.0)
        loss = torch.stack(
            [
                F.cross_entropy(logits.float().reshape(-1, 9), labels.reshape(-1))
                for logits in loop_logits
            ]
        ).mean()
    loss.backward()
    torch.cuda.synchronize(device)
    gradients = finite_gradient_summary(native)
    if gradients["unexpected_missing"] or gradients["nonfinite"]:
        raise AssertionError(f"Native GDN full-stack gradient gate failed: {gradients}")
    future_seed_grad = native.reasoner.blocks[1].future_seed_logit.grad
    if (
        future_seed_grad is None
        or not bool(torch.isfinite(future_seed_grad).all())
        or future_seed_grad.float().norm() <= 0
    ):
        raise AssertionError("Native GDN FutureSeed gate did not receive a finite nonzero gradient")
    print("[native-gdn-gate] pass 10-layer two-loop full-stack backward", flush=True)

    payload = {
        "status": "PASS",
        "phase": args.phase,
        "device": torch.cuda.get_device_name(device),
        "cuda_visible_devices": os.environ["CUDA_VISIBLE_DEVICES"],
        "hypothesis": (
            "The state-matched GDN arm may be compressed by K32/V32; official native "
            "K24/V48 geometry should improve finite-budget opening if geometry is the cause."
        ),
        "contract": {
            "seed": 52,
            "d_model": 192,
            "layers": 10,
            "heads": 6,
            "head_dim": 24,
            "gdn_expand_v": 2.0,
            "future_seed_scale": 1.0,
            "loops_smoke": 2,
            "shared_shell_init_seed": 52,
            "expected_official_46_50_delta": 0.15,
            "expected_ce_delta": -0.03,
        },
        "geometry": geometry,
        "parameter_count": sum(parameter.numel() for parameter in native.parameters()),
        "matched_parameter_count": sum(parameter.numel() for parameter in matched.parameters()),
        "shared_shell": {
            "common_parameter_tensors": len(common_shell),
            "identical_parameter_tensors": len(common_shell),
            "differing_parameter_tensors": len(differing_shell),
        },
        "runtime": runtime,
        "native_shape_reference": reference,
        "adapter": {
            "official_chunk_autograd_node": expected_node,
            "initial_state_grad_norm": float(h0.grad.float().norm().item()),
            "output_shape": list(mixer_output.shape),
            "state_shape": list(terminal_state.shape),
        },
        "full_stack": {
            "loss": float(loss.detach().cpu()),
            "output_shape": list(loop_logits[-1].shape),
            "fs_gate_mean": float(trace[-1]["fs_gate_mean"].detach().cpu()),
            "fs_state_norm": float(trace[-1]["fs_state_norm"].detach().cpu()),
            "future_seed_gate_grad_norm": float(future_seed_grad.float().norm().item()),
            "gradients": gradients,
            "forward_backward_sec": time.perf_counter() - started,
            "peak_allocated_mib": torch.cuda.max_memory_allocated(device) / (1024**2),
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
