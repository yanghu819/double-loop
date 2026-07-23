#!/usr/bin/env python3
"""Reconstruct previously sampled official Sudoku rows and emit the unseen set."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch


def parse_stages(spec: str) -> list[tuple[int, int, int]]:
    stages: list[tuple[int, int, int]] = []
    for raw_stage in spec.split(","):
        blank_range, raw_steps = raw_stage.strip().split(":", 1)
        raw_lo, raw_hi = blank_range.split("-", 1)
        stages.append((int(raw_lo), int(raw_hi), int(raw_steps)))
    if not stages or any(steps <= 0 for _lo, _hi, steps in stages):
        raise ValueError(f"Invalid hole stages: {spec!r}")
    return stages


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def simulate_stage(
    *,
    rng: random.Random,
    candidates: np.ndarray,
    draws: int,
    seen: np.ndarray,
    chunk_size: int,
) -> int:
    unique_before = int(seen.sum())
    remaining = int(draws)
    while remaining > 0:
        count = min(remaining, chunk_size)
        positions = np.fromiter(
            (rng.randrange(len(candidates)) for _ in range(count)),
            dtype=np.int64,
            count=count,
        )
        seen[candidates[positions]] = True
        remaining -= count
    return int(seen.sum()) - unique_before


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--target-min", type=int, default=51)
    parser.add_argument("--target-max", type=int, default=64)
    parser.add_argument("--chunk-size", type=int, default=500_000)
    args = parser.parse_args()

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    saved_step = int(checkpoint["saved_at_step"])
    checkpoint_args: dict[str, Any] = dict(checkpoint["args"])
    seed = int(checkpoint_args["seed"])
    batch = int(checkpoint_args["batch"])
    grad_accum = int(checkpoint_args.get("grad_accum_steps", 1))
    stages = parse_stages(str(checkpoint_args["hole_stages"]))
    if saved_step > sum(steps for _lo, _hi, steps in stages):
        raise ValueError("Checkpoint step exceeds the recorded curriculum.")

    split = str(checkpoint_args["official_sudoku_train_split"])
    input_path = args.data_dir / split / "all__inputs.npy"
    inputs = np.load(input_path, mmap_mode="r")
    blank_counts = (inputs == 1).sum(axis=1).astype(np.int16, copy=False)
    seen = np.zeros(len(inputs), dtype=np.bool_)
    rng = random.Random(seed + 1000)

    stage_rows: list[dict[str, Any]] = []
    completed_steps = 0
    for holes_min, holes_max, stage_steps in stages:
        executed_steps = min(max(saved_step - completed_steps, 0), stage_steps)
        candidates = np.flatnonzero(
            (blank_counts >= holes_min) & (blank_counts <= holes_max)
        )
        draws = executed_steps * batch * grad_accum
        new_unique = simulate_stage(
            rng=rng,
            candidates=candidates,
            draws=draws,
            seen=seen,
            chunk_size=args.chunk_size,
        )
        stage_rows.append(
            {
                "blank_range": [holes_min, holes_max],
                "executed_steps": executed_steps,
                "candidate_count": int(len(candidates)),
                "draws": draws,
                "new_unique_global_rows": new_unique,
            }
        )
        completed_steps += stage_steps
        if completed_steps >= saved_step:
            break

    rng_matches_checkpoint = rng.getstate() == checkpoint["rng_python"]
    if not rng_matches_checkpoint:
        raise RuntimeError(
            "Reconstructed Python RNG state does not match the checkpoint; "
            "refusing to infer unseen rows."
        )

    target_candidates = np.flatnonzero(
        (blank_counts >= args.target_min) & (blank_counts <= args.target_max)
    )
    unseen = target_candidates[~seen[target_candidates]]
    if unseen.size == 0:
        raise RuntimeError("No unseen target rows remain.")
    if bool((np.diff(unseen) <= 0).any()):
        raise RuntimeError("Generated unseen indices are not strictly increasing.")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.output, unseen)
    manifest = {
        "schema_version": 1,
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": sha256_file(args.checkpoint),
        "saved_step": saved_step,
        "seed": seed,
        "microbatch": batch,
        "grad_accum_steps": grad_accum,
        "effective_batch": batch * grad_accum,
        "stages": stage_rows,
        "rng_matches_checkpoint": rng_matches_checkpoint,
        "total_rows": int(len(inputs)),
        "total_unique_rows_seen": int(seen.sum()),
        "target_blank_range": [args.target_min, args.target_max],
        "target_rows": int(len(target_candidates)),
        "target_unique_rows_seen": int(seen[target_candidates].sum()),
        "target_unseen_rows": int(len(unseen)),
        "target_seen_fraction": float(seen[target_candidates].mean()),
        "output_indices": str(args.output),
        "output_indices_sha256": sha256_file(args.output),
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, sort_keys=True))


if __name__ == "__main__":
    main()
