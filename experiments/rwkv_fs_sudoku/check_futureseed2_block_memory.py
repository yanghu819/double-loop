#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import torch


def load_runner(repo: Path):
    experiment_dir = repo / "experiments" / "rwkv_fs_sudoku"
    sys.path.insert(0, str(experiment_dir))
    import study_rwkv_futureseed_loop as runner

    return runner


def build_reasoner(runner, *, scope: str) -> torch.nn.Module:
    return runner.FutureSeedRWKV(
        d_model=64,
        layers=3,
        heads=2,
        head_dim=32,
        channel_mult=2,
        future_seed_scale=1.0,
        future_seed_decay=0.0,
        future_seed_update="fixed",
        future_seed_norm_mode="unit",
        future_seed_gate_mode="head",
        future_seed_scope=scope,
        activation_checkpoint=False,
        backbone="gdn2",
        gdn_mode="chunk",
        gdn_expand_v=1.0,
        gdn_use_short_conv=True,
        gdn_conv_size=4,
        gdn_allow_neg_eigval=False,
    )


def tensor_max_abs(left: torch.Tensor, right: torch.Tensor) -> float:
    return float((left.detach().float() - right.detach().float()).abs().max().cpu())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    if os.environ.get("CUDA_VISIBLE_DEVICES") != "0":
        raise RuntimeError("FutureSeed2 block-memory contract requires CUDA_VISIBLE_DEVICES=0")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required; CPU smoke is forbidden")
    device = torch.device("cuda:0")
    if torch.cuda.current_device() != 0:
        raise RuntimeError(f"Expected visible CUDA device 0, got {torch.cuda.current_device()}")

    runner = load_runner(args.repo.resolve())
    torch.manual_seed(52)
    torch.cuda.manual_seed_all(52)
    layer_model = build_reasoner(runner, scope="layer")
    torch.manual_seed(52)
    torch.cuda.manual_seed_all(52)
    block_model = build_reasoner(runner, scope="block")
    block_model.load_state_dict(layer_model.state_dict(), strict=True)
    layer_model = layer_model.to(device).eval()
    block_model = block_model.to(device).eval()

    if sum(p.numel() for p in layer_model.parameters()) != sum(
        p.numel() for p in block_model.parameters()
    ):
        raise AssertionError("Block memory changed parameter count")

    inputs = torch.randn(2, 81, 64, device=device, dtype=torch.bfloat16)
    with torch.no_grad(), torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
        layer_first, _layer_diag, _ = layer_model(inputs)
        block_first_eval, block_first_diag_eval, block_bank_eval = block_model(inputs)
    first_pass_max_abs = tensor_max_abs(layer_first, block_first_eval)
    if first_pass_max_abs != 0.0:
        raise AssertionError(
            f"Block scope must be bit-exact to FutureSeed1 on its first pass, got {first_pass_max_abs}"
        )
    if block_bank_eval is None or len(block_bank_eval) != len(block_model.blocks):
        raise AssertionError("Block scope did not return one terminal state per layer")
    if float(block_first_diag_eval["fs2_block_seed_active"].cpu()) != 0.0:
        raise AssertionError("First block pass unexpectedly consumed prior block memory")

    block_model.train()
    torch.manual_seed(52)
    torch.cuda.manual_seed_all(52)
    with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
        block_first, _block_first_diag, block_bank = block_model(inputs)
    assert block_bank is not None
    for state in block_bank:
        state.retain_grad()

    captured: dict[int, torch.Tensor] = {}
    hooks = []
    for layer_idx, block in enumerate(block_model.blocks):
        def capture_initial_state(
            _module,
            _args,
            kwargs,
            *,
            index: int = layer_idx,
        ) -> None:
            initial_state = kwargs.get("initial_state")
            if initial_state is None:
                raise AssertionError(f"Layer {index} did not receive block memory")
            captured[index] = initial_state.detach().clone()

        hooks.append(block.register_forward_pre_hook(capture_initial_state, with_kwargs=True))

    with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
        block_second, block_second_diag, _block_bank_second = block_model(
            inputs,
            seed_memory=block_bank,
        )
        loss = block_second.float().square().mean()
    for hook in hooks:
        hook.remove()

    if float(block_second_diag["fs2_block_seed_active"].detach().cpu()) != 1.0:
        raise AssertionError("Second block pass did not report active same-layer memory")
    if len(captured) != len(block_model.blocks):
        raise AssertionError(f"Captured {len(captured)} block seeds, expected {len(block_model.blocks)}")

    routing_max_abs = 0.0
    for layer_idx, state in enumerate(block_bank):
        state_native = state.to(device=inputs.device, dtype=inputs.dtype)
        denom = state_native.square().mean(
            dim=(-1, -2),
            keepdim=True,
        ).sqrt().clamp(min=1e-6)
        gate = torch.sigmoid(block_model.blocks[layer_idx].future_seed_logit)
        expected = state_native / denom * gate
        routing_max_abs = max(
            routing_max_abs,
            tensor_max_abs(captured[layer_idx], expected),
        )
    if routing_max_abs != 0.0:
        raise AssertionError(
            f"Same-layer block-state routing differs from the direct formula: {routing_max_abs}"
        )

    loss.backward()
    torch.cuda.synchronize(device)
    state0_grad = block_bank[0].grad
    block0_gate_grad = block_model.blocks[0].future_seed_logit.grad
    if state0_grad is None or not bool(torch.isfinite(state0_grad).all()) or float(
        state0_grad.float().norm().cpu()
    ) == 0.0:
        raise AssertionError("Block-zero terminal state did not receive a finite nonzero gradient")
    if block0_gate_grad is None or not bool(torch.isfinite(block0_gate_grad).all()) or float(
        block0_gate_grad.float().norm().cpu()
    ) == 0.0:
        raise AssertionError("Block-zero FutureSeed gate did not receive a finite nonzero gradient")

    runtime = runner.strict_fla_runtime_summary(block_model, "gdn2")
    result: dict[str, Any] = {
        "status": "pass",
        "device": torch.cuda.get_device_name(device),
        "torch": torch.__version__,
        "parameter_count": sum(p.numel() for p in block_model.parameters()),
        "first_pass_max_abs": first_pass_max_abs,
        "same_layer_routing_max_abs": routing_max_abs,
        "second_pass_change_rms": float(
            (block_second.detach().float() - block_first.detach().float())
            .square()
            .mean()
            .sqrt()
            .cpu()
        ),
        "block_seed_raw_norm": float(
            block_second_diag["fs2_block_seed_raw_norm"].detach().float().cpu()
        ),
        "state0_grad_norm": float(state0_grad.float().norm().cpu()),
        "block0_gate_grad_norm": float(block0_gate_grad.float().norm().cpu()),
        "state_shapes": [list(state.shape) for state in block_bank],
        "runtime": runtime,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
