#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, MutableMapping, Tuple

import torch


EXPANDED_SUFFIXES = (
    ".time_mix.o_norm_weight",
    ".time_mix.v_proj.weight",
    ".time_mix.g_proj.weight",
    ".time_mix.o_proj.weight",
)


def _repeat_value_rows(tensor: torch.Tensor, heads: int, multiplier: int) -> torch.Tensor:
    old_v = tensor.shape[0] // heads
    shaped = tensor.reshape(heads, old_v, *tensor.shape[1:])
    return torch.cat([shaped] * multiplier, dim=1).reshape(heads * old_v * multiplier, *tensor.shape[1:])


def _repeat_value_vector(tensor: torch.Tensor, multiplier: int) -> torch.Tensor:
    return torch.cat([tensor] * multiplier, dim=0)


def _expand_output_columns(
    tensor: torch.Tensor,
    heads: int,
    multiplier: int,
    *,
    copy_new_channels: bool,
) -> torch.Tensor:
    old_v = tensor.shape[1] // heads
    shaped = tensor.reshape(tensor.shape[0], heads, old_v)
    additions = [shaped] if copy_new_channels else [torch.zeros_like(shaped)]
    pieces = [shaped] + additions * (multiplier - 1)
    return torch.cat(pieces, dim=2).reshape(tensor.shape[0], heads * old_v * multiplier)


def expand_parameter_tensor(
    name: str,
    tensor: torch.Tensor,
    *,
    heads: int,
    multiplier: int,
    optimizer_moment: bool = False,
) -> torch.Tensor:
    if name.endswith((".time_mix.v_proj.weight", ".time_mix.g_proj.weight")):
        if tensor.ndim != 2 or tensor.shape[0] % heads:
            raise ValueError(f"Unexpected value projection shape for {name}: {tuple(tensor.shape)}")
        return _repeat_value_rows(tensor, heads, multiplier)
    if name.endswith(".time_mix.o_norm_weight"):
        if tensor.ndim != 1:
            raise ValueError(f"Unexpected output norm shape for {name}: {tuple(tensor.shape)}")
        return _repeat_value_vector(tensor, multiplier)
    if name.endswith(".time_mix.o_proj.weight"):
        if tensor.ndim != 2 or tensor.shape[1] % heads:
            raise ValueError(f"Unexpected output projection shape for {name}: {tuple(tensor.shape)}")
        return _expand_output_columns(
            tensor,
            heads,
            multiplier,
            copy_new_channels=False,
        )
    raise ValueError(f"Unsupported GDN expansion parameter: {name}")


def _parameter_names(model_state: Mapping[str, torch.Tensor], optimizer_param_count: int) -> list[str]:
    # The clean runner has one persistent buffer; validate the inferred mapping
    # against every materialized Adam moment before trusting it.
    candidates = [name for name in model_state if name != "scratch_projection"]
    if len(candidates) != optimizer_param_count:
        raise ValueError(
            "Cannot infer optimizer parameter names: "
            f"state_dict candidates={len(candidates)}, optimizer params={optimizer_param_count}"
        )
    return candidates


def _optimizer_param_ids(optimizer: Mapping[str, Any]) -> list[int]:
    return [int(param_id) for group in optimizer["param_groups"] for param_id in group["params"]]


def _validate_optimizer_mapping(
    model_state: Mapping[str, torch.Tensor],
    optimizer: Mapping[str, Any],
    names: Iterable[str],
    param_ids: Iterable[int],
) -> None:
    opt_state = optimizer["state"]
    for name, param_id in zip(names, param_ids):
        expected = tuple(model_state[name].shape)
        for state_name, value in opt_state.get(param_id, {}).items():
            if not torch.is_tensor(value) or value.ndim == 0:
                continue
            if tuple(value.shape) != expected:
                raise ValueError(
                    f"Optimizer mapping mismatch for {name}/{state_name}: "
                    f"moment={tuple(value.shape)}, parameter={expected}"
                )


def expand_checkpoint(checkpoint: MutableMapping[str, Any], new_expand_v: float) -> Dict[str, Any]:
    args = checkpoint.get("args")
    if not isinstance(args, MutableMapping):
        raise ValueError("Checkpoint is missing mutable args metadata")
    if args.get("backbone") != "gdn":
        raise ValueError("Function-preserving expansion currently supports BACKBONE=gdn only")
    if int(args.get("gdn_use_short_conv", 0)) != 0:
        raise ValueError("Function-preserving expansion requires gdn_use_short_conv=0")

    old_expand_v = float(args["gdn_expand_v"])
    ratio = float(new_expand_v) / old_expand_v
    multiplier = int(round(ratio))
    if multiplier <= 1 or not math.isclose(ratio, multiplier, rel_tol=0.0, abs_tol=1e-9):
        raise ValueError(
            f"new expand_v must be an integer multiple greater than old expand_v: {old_expand_v} -> {new_expand_v}"
        )

    heads = int(args["heads"])
    model_state = checkpoint["model"]
    optimizer = checkpoint["optimizer"]
    param_ids = _optimizer_param_ids(optimizer)
    param_names = _parameter_names(model_state, len(param_ids))
    _validate_optimizer_mapping(model_state, optimizer, param_names, param_ids)
    param_to_id = dict(zip(param_names, param_ids))

    expanded_names = [name for name in model_state if name.endswith(EXPANDED_SUFFIXES)]
    expected_count = int(args["layers"]) * len(EXPANDED_SUFFIXES)
    if len(expanded_names) != expected_count:
        raise ValueError(f"Expected {expected_count} expandable tensors, found {len(expanded_names)}")

    old_shapes: Dict[str, Tuple[int, ...]] = {}
    new_shapes: Dict[str, Tuple[int, ...]] = {}
    opt_state = optimizer["state"]
    for name in expanded_names:
        old_tensor = model_state[name]
        old_shapes[name] = tuple(old_tensor.shape)
        expanded = expand_parameter_tensor(name, old_tensor, heads=heads, multiplier=multiplier)
        model_state[name] = expanded
        new_shapes[name] = tuple(expanded.shape)

        param_id = param_to_id[name]
        for state_name, value in list(opt_state.get(param_id, {}).items()):
            if not torch.is_tensor(value) or value.ndim == 0:
                continue
            opt_state[param_id][state_name] = expand_parameter_tensor(
                name,
                value,
                heads=heads,
                multiplier=multiplier,
                optimizer_moment=True,
            )
            if tuple(opt_state[param_id][state_name].shape) != tuple(expanded.shape):
                raise ValueError(f"Expanded optimizer state has wrong shape for {name}/{state_name}")

    original_reason = checkpoint.get("reason", "")
    args["gdn_expand_v"] = float(new_expand_v)
    checkpoint["reason"] = "function_preserving_gdn_state_expansion"
    checkpoint["checkpoint_transform"] = {
        "type": "function_preserving_gdn_value_state_expansion",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "old_expand_v": old_expand_v,
        "new_expand_v": float(new_expand_v),
        "multiplier": multiplier,
        "optimizer_state_expanded": True,
        "original_reason": original_reason,
        "construction": {
            "v_proj": "repeat_each_head_value_block",
            "g_proj": "repeat_each_head_value_block",
            "o_norm": "repeat_each_head_value_block",
            "o_proj": "copy_old_head_block_then_zero_new_blocks",
        },
    }
    return {
        "old_expand_v": old_expand_v,
        "new_expand_v": float(new_expand_v),
        "multiplier": multiplier,
        "expanded_tensor_count": len(expanded_names),
        "old_shapes": old_shapes,
        "new_shapes": new_shapes,
        "saved_at_step": int(checkpoint.get("saved_at_step", -1)),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Function-preserving GDN value/state checkpoint expansion")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--new-expand-v", type=float, required=True)
    parser.add_argument("--summary", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite existing checkpoint: {args.output}")
    checkpoint = torch.load(args.input, map_location="cpu", weights_only=False, mmap=True)
    summary = expand_checkpoint(checkpoint, args.new_expand_v)
    summary["input"] = str(args.input.resolve())
    summary["output"] = str(args.output.resolve())

    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    torch.save(checkpoint, temporary)
    os.replace(temporary, args.output)
    summary["output_bytes"] = args.output.stat().st_size

    encoded = json.dumps(summary, indent=2, sort_keys=True)
    if args.summary is not None:
        args.summary.parent.mkdir(parents=True, exist_ok=True)
        args.summary.write_text(encoded + "\n", encoding="utf-8")
    print(encoded)


if __name__ == "__main__":
    main()
