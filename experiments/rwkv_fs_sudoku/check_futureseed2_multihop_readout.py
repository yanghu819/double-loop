#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch
import torch.nn.functional as F


def load_runner(repo: Path):
    experiment_dir = repo / "experiments" / "rwkv_fs_sudoku"
    sys.path.insert(0, str(experiment_dir))
    import study_rwkv_futureseed_loop as runner

    return runner


def build_reasoner(runner, *, readout_hop: int) -> torch.nn.Module:
    return runner.FutureSeedRWKV(
        d_model=64,
        layers=4,
        heads=2,
        head_dim=32,
        channel_mult=2,
        future_seed_scale=1.0,
        future_seed_decay=0.0,
        future_seed_update="fixed",
        future_seed_norm_mode="unit",
        future_seed_gate_mode="head",
        future_seed_scope="layer",
        future_seed_readout_hop=readout_hop,
        activation_checkpoint=False,
        backbone="gdn2",
        gdn_mode="chunk",
        gdn_expand_v=1.0,
        gdn_use_short_conv=True,
        gdn_conv_size=4,
        gdn_allow_neg_eigval=False,
    )


def tensor_max_abs(left: torch.Tensor, right: torch.Tensor) -> float:
    return float(
        (left.detach().float() - right.detach().float()).abs().max().cpu()
    )


def direct_readout(
    block: torch.nn.Module,
    hidden: torch.Tensor,
    state: torch.Tensor,
) -> torch.Tensor:
    normalized = block.ln_time(hidden)
    mix = block.time_mix
    core = mix.core
    batch_size, seq_len, _channels = normalized.shape
    q = core.q_proj(normalized)
    if core.use_short_conv:
        q, _conv_state = core.q_conv1d(
            x=q,
            cache=None,
            output_final_state=False,
        )
    else:
        q = F.silu(q)
    q = F.normalize(
        q.float().view(batch_size, seq_len, mix.heads, mix.head_dim),
        dim=-1,
        p=2.0,
    )
    value = torch.einsum("bthk,bhkv->bthv", q, state.float())
    gate = core.g_proj(normalized).view(
        batch_size,
        seq_len,
        mix.heads,
        mix.head_v_dim,
    )
    value = core.o_norm(value.to(dtype=normalized.dtype), gate)
    return core.o_proj(value.reshape(batch_size, seq_len, mix.value_dim))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    if os.environ.get("CUDA_VISIBLE_DEVICES") != "0":
        raise RuntimeError(
            "FutureSeed2 multihop-readout contract requires CUDA_VISIBLE_DEVICES=0"
        )
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required; CPU model smoke is forbidden")
    device = torch.device("cuda:0")
    if torch.cuda.current_device() != 0:
        raise RuntimeError(
            f"Expected visible CUDA device 0, got {torch.cuda.current_device()}"
        )

    runner = load_runner(args.repo.resolve())
    torch.manual_seed(52)
    torch.cuda.manual_seed_all(52)
    baseline = build_reasoner(runner, readout_hop=0)
    torch.manual_seed(52)
    torch.cuda.manual_seed_all(52)
    readout = build_reasoner(runner, readout_hop=2)
    missing, unexpected = readout.load_state_dict(
        baseline.state_dict(),
        strict=False,
    )
    if missing != ["future_seed_readout_scale"] or unexpected:
        raise AssertionError(
            f"Unexpected identity migration: missing={missing}, unexpected={unexpected}"
        )
    baseline = baseline.to(device).eval()
    readout = readout.to(device).eval()

    parameter_delta = sum(p.numel() for p in readout.parameters()) - sum(
        p.numel() for p in baseline.parameters()
    )
    if parameter_delta != 2:
        raise AssertionError(
            f"Expected one scalar for each two-hop edge, got parameter delta {parameter_delta}"
        )

    inputs = torch.randn(
        2,
        81,
        64,
        device=device,
        dtype=torch.bfloat16,
    )
    with torch.no_grad(), torch.amp.autocast(
        device_type="cuda",
        dtype=torch.bfloat16,
    ):
        baseline_output, _baseline_diag, _ = baseline(inputs)
        identity_output, identity_diag, _ = readout(inputs)
    identity_max_abs = tensor_max_abs(baseline_output, identity_output)
    if identity_max_abs != 0.0:
        raise AssertionError(
            f"Zero-init multihop readout must be bit-exact to FutureSeed1, got {identity_max_abs}"
        )
    if float(identity_diag["fs2_readout_enabled"].cpu()) != 1.0:
        raise AssertionError("Multihop readout did not report enabled")
    if float(identity_diag["fs2_readout_scale_abs"].cpu()) != 0.0:
        raise AssertionError("Zero-init multihop scale is not zero")
    if float(identity_diag["fs2_readout_raw_norm"].cpu()) <= 0.0:
        raise AssertionError("Compatible readout produced a zero raw feature")
    if float(identity_diag["fs2_readout_residual_norm"].cpu()) != 0.0:
        raise AssertionError("Zero-init multihop residual is not zero")

    source_block = readout.blocks[0]
    with torch.no_grad(), torch.amp.autocast(
        device_type="cuda",
        dtype=torch.bfloat16,
    ):
        source_hidden, source_state = source_block(
            inputs,
            initial_state=None,
        )
        actual_readout = source_block.read_terminal_state(
            source_hidden,
            source_state,
        )
        expected_readout = direct_readout(
            source_block,
            source_hidden,
            source_state,
        )
    formula_max_abs = tensor_max_abs(actual_readout, expected_readout)
    if formula_max_abs != 0.0:
        raise AssertionError(
            f"Compatible state readout differs from direct formula: {formula_max_abs}"
        )

    assert readout.future_seed_readout_scale is not None
    with torch.no_grad():
        readout.future_seed_readout_scale.fill_(0.125)
    with torch.no_grad(), torch.amp.autocast(
        device_type="cuda",
        dtype=torch.bfloat16,
    ):
        active_output, active_diag, _ = readout(inputs)
    active_change_rms = float(
        (active_output.float() - identity_output.float())
        .square()
        .mean()
        .sqrt()
        .cpu()
    )
    if active_change_rms <= 0.0:
        raise AssertionError("Activated multihop readout did not change model output")
    if float(active_diag["fs2_readout_residual_norm"].cpu()) <= 0.0:
        raise AssertionError("Activated multihop readout has zero residual")

    readout.train()
    readout.zero_grad(set_to_none=True)
    with torch.no_grad():
        readout.future_seed_readout_scale.zero_()
    with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
        train_output, _train_diag, _ = readout(inputs)
        loss = train_output.float().square().mean()
    loss.backward()
    torch.cuda.synchronize(device)
    scale_grad = readout.future_seed_readout_scale.grad
    if (
        scale_grad is None
        or not bool(torch.isfinite(scale_grad).all())
        or float(scale_grad.float().norm().cpu()) == 0.0
    ):
        raise AssertionError(
            "Zero-init multihop readout scale did not receive a finite nonzero gradient"
        )

    runtime = runner.strict_fla_runtime_summary(
        SimpleNamespace(reasoner=readout),
        "gdn2",
    )
    result: dict[str, Any] = {
        "status": "pass",
        "device": torch.cuda.get_device_name(device),
        "torch": torch.__version__,
        "parameter_count": sum(p.numel() for p in readout.parameters()),
        "parameter_delta": parameter_delta,
        "identity_max_abs": identity_max_abs,
        "readout_formula_max_abs": formula_max_abs,
        "identity_raw_readout_norm": float(
            identity_diag["fs2_readout_raw_norm"].float().cpu()
        ),
        "active_change_rms": active_change_rms,
        "active_residual_norm": float(
            active_diag["fs2_readout_residual_norm"].float().cpu()
        ),
        "zero_init_scale_grad_norm": float(
            scale_grad.float().norm().cpu()
        ),
        "runtime": runtime,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
