#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path
from typing import Any

import torch

from study_rwkv_futureseed_loop import (
    FLACache,
    FLADeltaTimeMix,
    FLARaven,
    FutureSeedRWKV,
    GatedDeltaNet2,
    RAVEN_FLA_SHA,
    resolve_strict_fla_source,
)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def finite_grad(parameter: torch.nn.Parameter) -> bool:
    return parameter.grad is not None and bool(torch.isfinite(parameter.grad).all().item())


def gpu_identity() -> dict[str, str]:
    line = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=index,uuid,name",
            "--format=csv,noheader,nounits",
            "-i",
            "0",
        ],
        text=True,
    ).strip()
    index, uuid, name = (part.strip() for part in line.split(",", 2))
    return {"index": index, "uuid": uuid, "name": name}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--expected_gpu_uuid", default="")
    args = parser.parse_args()

    require(os.environ.get("CUDA_VISIBLE_DEVICES") == "0", "Contract requires CUDA_VISIBLE_DEVICES=0")
    require(torch.cuda.is_available(), "CUDA is unavailable; CPU fallback is forbidden")
    require(torch.cuda.device_count() == 1, "The process must expose exactly GPU1")
    require(FLARaven is not None and FLACache is not None, "Official Raven/FLA Cache import failed")
    source_sha, source_root, source_module = resolve_strict_fla_source("raven")
    require(source_sha == RAVEN_FLA_SHA, "Unexpected Raven FLA source SHA")

    gpu = gpu_identity()
    if args.expected_gpu_uuid:
        require(gpu["uuid"] == args.expected_gpu_uuid, "GPU UUID changed before contract execution")

    torch.manual_seed(5201)
    torch.cuda.manual_seed_all(5201)
    device = torch.device("cuda:0")
    dtype = torch.bfloat16
    batch, seq_len, hidden, heads, head_dim = 2, 81, 64, 2, 32
    slots, topk = 16, 2

    mix = FLADeltaTimeMix(
        hidden,
        heads,
        head_dim,
        backbone="raven",
        expand_v=1.0,
        mode="chunk",
        use_short_conv=False,
        conv_size=4,
        allow_neg_eigval=False,
        raven_num_slots=slots,
        raven_topk=topk,
    ).to(device=device, dtype=dtype)
    require(type(mix.core) is FLARaven, "Adapter did not instantiate the exact official Raven class")
    require(mix.core.mode == "chunk", "Official Raven is not using chunk training mode")
    require(mix.core.topk == topk and mix.core.num_slots == slots, "Raven routing budget drifted")
    require(mix.core.add_gumbel_noise, "Official Raven training-time Gumbel routing is disabled")

    mix.eval()
    x = torch.randn(batch, seq_len, hidden, device=device, dtype=dtype)
    with torch.no_grad():
        adapter_output, packed_state = mix(x)
        require(
            tuple(packed_state.shape) == (batch, heads, head_dim * 2, slots),
            f"Unexpected packed Raven state shape: {tuple(packed_state.shape)}",
        )
        unpacked = mix.unpack_recurrent_state(packed_state)
        roundtrip = mix.pack_recurrent_state(unpacked)
        require(torch.equal(roundtrip, packed_state), "Raven state pack/unpack is not lossless")

        direct_cache = FLACache()
        direct_output, _attentions, direct_cache = mix.core(
            x,
            past_key_values=direct_cache,
            use_cache=True,
        )
        direct_state = mix.pack_recurrent_state(direct_cache[0]["recurrent_state"])
        require(torch.equal(adapter_output, direct_output), "Adapter output differs from official Raven")
        require(torch.equal(packed_state, direct_state), "Adapter state differs from official Raven")

        seeded_state = torch.randn_like(packed_state.float()).mul_(0.01)
        seeded_output, _seeded_terminal = mix(x, initial_state=seeded_state)
        initial_state_effect = float((seeded_output.float() - adapter_output.float()).square().mean().sqrt())
        require(initial_state_effect > 1e-6, "Raven ignores the FutureSeed initial state")

    mix.train()
    train_x = torch.randn(batch, seq_len, hidden, device=device, dtype=dtype, requires_grad=True)
    train_output, train_state = mix(train_x)
    train_loss = train_output.float().square().mean() + 1e-5 * train_state.float().square().mean()
    train_loss.backward()
    require(finite_grad(mix.core.r_proj.weight), "No finite gradient reached Raven router projection")
    require(finite_grad(mix.core.k_proj.weight), "No finite gradient reached Raven key projection")
    require(finite_grad(mix.core.v_proj.weight), "No finite gradient reached Raven value projection")

    gdn2 = FLADeltaTimeMix(
        hidden,
        heads,
        head_dim,
        backbone="gdn2",
        expand_v=1.0,
        mode="chunk",
        use_short_conv=True,
        conv_size=4,
        allow_neg_eigval=False,
    ).to(device=device, dtype=dtype)
    require(type(gdn2.core) is GatedDeltaNet2, "Baseline is not the exact official GDN2 class")
    gdn2.train()
    gdn2_x = torch.randn(batch, seq_len, hidden, device=device, dtype=dtype, requires_grad=True)
    gdn2_output, gdn2_state = gdn2(gdn2_x)
    require(
        tuple(gdn2_state.shape) == (batch, heads, head_dim, head_dim),
        f"Unexpected official GDN2 state shape: {tuple(gdn2_state.shape)}",
    )
    gdn2_loss = gdn2_output.float().square().mean() + 1e-5 * gdn2_state.float().square().mean()
    gdn2_loss.backward()
    require(finite_grad(gdn2.core.k_proj.weight), "No finite gradient reached GDN2 key projection")
    require(finite_grad(gdn2.core.v_proj.weight), "No finite gradient reached GDN2 value projection")

    reasoner = FutureSeedRWKV(
        hidden,
        2,
        heads,
        head_dim,
        2,
        future_seed_scale=1.0,
        future_seed_decay=0.0,
        future_seed_update="fixed",
        future_seed_norm_mode="unit",
        future_seed_gate_mode="head",
        future_seed_scope="layer",
        activation_checkpoint=False,
        backbone="raven",
        gdn_mode="chunk",
        gdn_expand_v=1.0,
        gdn_use_short_conv=False,
        gdn_conv_size=4,
        gdn_allow_neg_eigval=False,
        raven_num_slots=slots,
        raven_topk=topk,
    ).to(device=device, dtype=dtype)
    reasoner.eval()
    with torch.no_grad():
        reasoner.future_seed_scale = 0.0
        no_fs_output, _no_fs_diag, _no_fs_memory = reasoner(x)
        reasoner.future_seed_scale = 1.0
        fs_output, fs_diag, _fs_memory = reasoner(x)
    fs_effect = float((fs_output.float() - no_fs_output.float()).square().mean().sqrt())
    require(fs_effect > 1e-6, "FutureSeed does not change the Raven backbone output")
    require(float(fs_diag["fs_gate_mean"].item()) > 0.0, "FutureSeed gate is inactive")

    state_elements_raven = slots * (head_dim + head_dim)
    state_elements_gdn2 = head_dim * head_dim
    require(state_elements_raven == state_elements_gdn2, "Raven/GDN2 state elements are not matched")

    payload: dict[str, Any] = {
        "status": "pass",
        "torch_version": torch.__version__,
        "gpu": gpu,
        "fla_source_sha": source_sha,
        "fla_source_root": source_root,
        "fla_source_module": source_module,
        "official_class": f"{type(mix.core).__module__}.{type(mix.core).__qualname__}",
        "official_kernel": "fla.ops.gsa.chunk_gsa",
        "mode": mix.core.mode,
        "state": {
            "head_dim": head_dim,
            "num_slots": slots,
            "topk": topk,
            "occupancy": topk / slots,
            "raven_elements_per_head": state_elements_raven,
            "gdn2_elements_per_head": state_elements_gdn2,
            "packed_shape": list(packed_state.shape),
            "pack_roundtrip_exact": True,
        },
        "adapter_matches_official_output": True,
        "adapter_matches_official_state": True,
        "initial_state_output_rms_change": initial_state_effect,
        "future_seed_output_rms_change": fs_effect,
        "training_loss": float(train_loss.detach().item()),
        "gradients": {
            "router": finite_grad(mix.core.r_proj.weight),
            "key": finite_grad(mix.core.k_proj.weight),
            "value": finite_grad(mix.core.v_proj.weight),
        },
        "matched_gdn2": {
            "official_class": f"{type(gdn2.core).__module__}.{type(gdn2.core).__qualname__}",
            "state_shape": list(gdn2_state.shape),
            "training_loss": float(gdn2_loss.detach().item()),
            "gradients": {
                "key": finite_grad(gdn2.core.k_proj.weight),
                "value": finite_grad(gdn2.core.v_proj.weight),
            },
        },
        "cpu_fallback": False,
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2), flush=True)


if __name__ == "__main__":
    main()
