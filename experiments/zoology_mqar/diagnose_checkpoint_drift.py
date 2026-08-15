#!/usr/bin/env python3
"""Compare trained checkpoints without constructing or running a model."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tensor_mapping(value: Any) -> dict[str, torch.Tensor]:
    candidates: list[dict[str, torch.Tensor]] = []

    def visit(item: Any) -> None:
        if not isinstance(item, dict):
            return
        tensors = {
            str(key): tensor.detach().cpu()
            for key, tensor in item.items()
            if isinstance(tensor, torch.Tensor)
        }
        if tensors:
            candidates.append(tensors)
        for child in item.values():
            visit(child)

    visit(value)
    if not candidates:
        raise ValueError("checkpoint contains no tensor mapping")
    return max(candidates, key=lambda mapping: sum(t.numel() for t in mapping.values()))


def canonical_name(name: str) -> str:
    return name.replace(".sequence_mixer.layer.base.", ".sequence_mixer.layer.")


def load_state(path: Path) -> dict[str, torch.Tensor]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    raw = tensor_mapping(checkpoint)
    state: dict[str, torch.Tensor] = {}
    for name, tensor in raw.items():
        canonical = canonical_name(name)
        if canonical in state:
            raise ValueError(f"checkpoint key collision after canonicalization: {canonical}")
        state[canonical] = tensor
    return state


def category(name: str) -> str:
    lower = name.lower()
    if "cycle" in lower or "reverse" in lower:
        return "cycle_or_reverse"
    if "future" in lower or "seed_gate" in lower or "fs_gate" in lower:
        return "futureseed"
    if any(part in lower for part in ("embedding", "embeddings", "token_emb", "word_emb")):
        return "embedding"
    if any(part in lower for part in ("q_proj", "q_conv", ".q_", "query")):
        return "query_address"
    if any(part in lower for part in ("k_proj", "k_conv", ".k_", "key")):
        return "write_address"
    if any(part in lower for part in ("v_proj", "v_conv", ".v_", "value")):
        return "value_payload"
    if any(part in lower for part in ("decay", "a_proj", "g_proj", "forget")):
        return "decay"
    if any(part in lower for part in ("b_proj", "erase", "beta")):
        return "erase_gate"
    if any(part in lower for part in ("w_proj", "write_gate")):
        return "write_gate"
    if any(part in lower for part in ("o_proj", "out_proj", "lm_head", "unembed", "decoder")):
        return "output"
    if any(part in lower for part in ("mlp", "ffn", "feed_forward", "mixer.f_proj")):
        return "mlp"
    if "norm" in lower:
        return "norm"
    return "other"


def flattened(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.to(torch.float64).reshape(-1)


def add(stats: dict[str, float], key: str, value: float) -> None:
    stats[key] = stats.get(key, 0.0) + value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--initial", type=Path, required=True)
    parser.add_argument("--native", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    initial = load_state(args.initial)
    native = load_state(args.native)
    candidate = load_state(args.candidate)
    common = sorted(set(initial) & set(native) & set(candidate))
    common = [
        name
        for name in common
        if initial[name].shape == native[name].shape == candidate[name].shape
    ]
    if not common:
        raise RuntimeError("checkpoints have no shape-compatible common tensors")

    grouped: dict[str, dict[str, float]] = defaultdict(dict)
    tensors: list[dict[str, Any]] = []
    for name in common:
        init_value = flattened(initial[name])
        native_value = flattened(native[name])
        candidate_value = flattened(candidate[name])
        native_update = native_value - init_value
        candidate_update = candidate_value - init_value
        divergence = candidate_value - native_value
        group = category(name)
        stats = grouped[group]
        add(stats, "numel", float(init_value.numel()))
        add(stats, "initial_sq", float(torch.dot(init_value, init_value)))
        add(stats, "native_update_sq", float(torch.dot(native_update, native_update)))
        add(stats, "candidate_update_sq", float(torch.dot(candidate_update, candidate_update)))
        add(stats, "divergence_sq", float(torch.dot(divergence, divergence)))
        add(stats, "update_dot", float(torch.dot(native_update, candidate_update)))
        tensors.append(
            {
                "name": name,
                "category": group,
                "numel": init_value.numel(),
                "native_update_rms": float(torch.sqrt(torch.mean(native_update.square()))),
                "candidate_update_rms": float(torch.sqrt(torch.mean(candidate_update.square()))),
                "candidate_native_divergence_rms": float(
                    torch.sqrt(torch.mean(divergence.square()))
                ),
            }
        )

    summary: dict[str, dict[str, float]] = {}
    for group, stats in grouped.items():
        numel = stats["numel"]
        native_norm = math.sqrt(stats["native_update_sq"])
        candidate_norm = math.sqrt(stats["candidate_update_sq"])
        denominator = max(native_norm * candidate_norm, 1e-30)
        summary[group] = {
            "numel": int(numel),
            "native_update_rms": math.sqrt(stats["native_update_sq"] / numel),
            "candidate_update_rms": math.sqrt(stats["candidate_update_sq"] / numel),
            "candidate_native_divergence_rms": math.sqrt(stats["divergence_sq"] / numel),
            "candidate_to_native_update_norm_ratio": candidate_norm / max(native_norm, 1e-30),
            "candidate_native_update_cosine": stats["update_dot"] / denominator,
            "divergence_to_initial_norm": math.sqrt(stats["divergence_sq"])
            / max(math.sqrt(stats["initial_sq"]), 1e-30),
        }

    result = {
        "kind": "checkpoint_tensor_drift_only_no_model_forward",
        "inputs": {
            "initial": {"path": str(args.initial), "sha256": sha256(args.initial)},
            "native": {"path": str(args.native), "sha256": sha256(args.native)},
            "candidate": {"path": str(args.candidate), "sha256": sha256(args.candidate)},
        },
        "tensor_counts": {
            "initial": len(initial),
            "native": len(native),
            "candidate": len(candidate),
            "common_shape_compatible": len(common),
        },
        "candidate_only_tensors": sorted(set(candidate) - set(native)),
        "native_only_tensors": sorted(set(native) - set(candidate)),
        "groups": dict(sorted(summary.items())),
        "largest_divergence_tensors": sorted(
            tensors,
            key=lambda item: item["candidate_native_divergence_rms"],
            reverse=True,
        )[:30],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
