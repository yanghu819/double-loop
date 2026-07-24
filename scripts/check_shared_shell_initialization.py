#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import torch

from check_sudoku_backbone_contract import BACKBONES, build_model, load_runner


def tensor_digest(tensor: torch.Tensor) -> str:
    value = tensor.detach().to(device="cpu", dtype=torch.float32).contiguous().numpy()
    return hashlib.sha256(value.tobytes()).hexdigest()


def is_shared_shell_parameter(name: str) -> bool:
    return ".time_mix." not in name


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    if os.environ.get("CUDA_VISIBLE_DEVICES") != "0":
        raise RuntimeError("Set CUDA_VISIBLE_DEVICES=0; this gate is GPU1-only")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required; CPU model execution is forbidden")

    repo = args.repo.resolve()
    runner = load_runner(repo)
    runner.configure_sudoku(9, 0, 0)
    device = torch.device("cuda", 0)
    hashes: dict[str, dict[str, str]] = {}
    shapes: dict[str, dict[str, list[int]]] = {}

    for public_name in BACKBONES:
        torch.manual_seed(52)
        torch.cuda.manual_seed_all(52)
        model = build_model(runner, public_name).to(device)
        hashes[public_name] = {
            name: tensor_digest(parameter)
            for name, parameter in model.named_parameters()
            if is_shared_shell_parameter(name)
        }
        shapes[public_name] = {
            name: list(parameter.shape)
            for name, parameter in model.named_parameters()
            if is_shared_shell_parameter(name)
        }
        del model
        torch.cuda.empty_cache()

    common_names = sorted(set.intersection(*(set(row) for row in hashes.values())))
    same_shape_names = [
        name
        for name in common_names
        if len({tuple(shapes[backbone][name]) for backbone in BACKBONES}) == 1
    ]
    differing_names = [
        name
        for name in same_shape_names
        if len({hashes[backbone][name] for backbone in BACKBONES}) != 1
    ]
    if differing_names:
        raise AssertionError(f"Shared-shell initialization differs by backbone: {differing_names}")

    payload: dict[str, Any] = {
        "status": "PASS",
        "device": torch.cuda.get_device_name(device),
        "cuda_visible_devices": os.environ["CUDA_VISIBLE_DEVICES"],
        "seed_reset_per_backbone": 52,
        "shared_shell_init_seed": 52,
        "backbones": BACKBONES,
        "definition": "All parameters outside each block's time_mix are the shared shell.",
        "common_parameter_tensors": len(common_names),
        "same_shape_parameter_tensors": len(same_shape_names),
        "identical_parameter_tensors": len(same_shape_names),
        "differing_parameter_tensors": 0,
        "differing_names": [],
        "hashes": hashes,
        "shapes": shapes,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in payload.items() if key not in {"hashes", "shapes"}}, indent=2))


if __name__ == "__main__":
    main()
