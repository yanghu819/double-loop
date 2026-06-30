#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch


IGNORE_LABEL_ID = -100


def parse_steps(raw: str) -> List[int]:
    steps = sorted({int(x.strip()) for x in raw.split(",") if x.strip()})
    if not steps or min(steps) < 1:
        raise ValueError("--steps must contain positive integers")
    return steps


def parse_board(raw: str, *, allow_blank: bool) -> np.ndarray:
    text = "".join(ch for ch in raw if ch.isdigit() or ch == ".")
    if len(text) != 81:
        raise ValueError(f"expected 81 cells, got {len(text)} from {raw!r}")
    values: List[int] = []
    for ch in text:
        if ch == ".":
            if not allow_blank:
                raise ValueError("solution board cannot contain blanks")
            values.append(0)
        else:
            val = int(ch)
            if val == 0 and not allow_blank:
                raise ValueError("solution board cannot contain zeros")
            values.append(val)
    arr = np.asarray(values, dtype=np.int64)
    if allow_blank:
        if not np.all((arr >= 0) & (arr <= 9)):
            raise ValueError("puzzle digits must be in 0..9")
    else:
        if not np.all((arr >= 1) & (arr <= 9)):
            raise ValueError("solution digits must be in 1..9")
    return arr


def board_to_string(board: np.ndarray, *, blank: str = ".") -> str:
    chars = [blank if int(x) == 0 else str(int(x)) for x in board.reshape(-1)]
    return "".join(chars)


def jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, torch.Tensor):
        return jsonable(value.detach().cpu().numpy())
    if isinstance(value, Path):
        return str(value)
    return value


def import_official_eqr(repo: Path) -> Dict[str, Any]:
    sys.path.insert(0, str(repo))
    os.chdir(repo)

    from evaluate import (  # type: ignore
        _apply_overrides,
        _apply_runtime_arch_overrides,
        _base_config,
        _load_checkpoint,
        _seed,
    )
    from evaluators.eval_config import EvalConfig  # type: ignore
    from pretrain import create_dataloader, init_train_state  # type: ignore
    from utils.checkpoint import load_ema, load_model_state_dict  # type: ignore

    return {
        "EvalConfig": EvalConfig,
        "_apply_overrides": _apply_overrides,
        "_apply_runtime_arch_overrides": _apply_runtime_arch_overrides,
        "_base_config": _base_config,
        "_load_checkpoint": _load_checkpoint,
        "_seed": _seed,
        "create_dataloader": create_dataloader,
        "init_train_state": init_train_state,
        "load_ema": load_ema,
        "load_model_state_dict": load_model_state_dict,
    }


def conflict_units(board: np.ndarray) -> List[str]:
    grid = board.reshape(9, 9)
    units: List[tuple[str, np.ndarray]] = []
    for i in range(9):
        units.append((f"row {i + 1}", grid[i, :]))
        units.append((f"col {i + 1}", grid[:, i]))
    for br in range(3):
        for bc in range(3):
            units.append((f"box {br + 1},{bc + 1}", grid[br * 3 : br * 3 + 3, bc * 3 : bc * 3 + 3].reshape(-1)))

    out: List[str] = []
    for name, vals in units:
        parts: List[str] = []
        for digit in range(1, 10):
            count = int((vals == digit).sum())
            if count > 1:
                parts.append(f"{digit}x{count}")
        if parts:
            out.append(f"{name}: {'; '.join(parts)}")
    return out


def sudoku_metrics(pred: np.ndarray, solution: np.ndarray, puzzle: np.ndarray) -> Dict[str, Any]:
    pred = pred.reshape(-1).astype(np.int64)
    solution = solution.reshape(-1).astype(np.int64)
    puzzle = puzzle.reshape(-1).astype(np.int64)
    hidden = puzzle == 0
    wrong_mask = (pred != solution) & hidden
    full_wrong = pred != solution
    wrong_coords = []
    for idx in np.flatnonzero(wrong_mask):
        wrong_coords.append(
            {
                "coord": f"R{idx // 9 + 1}C{idx % 9 + 1}",
                "pred": int(pred[idx]),
                "truth": int(solution[idx]),
            }
        )
    conflicts = conflict_units(pred)
    return {
        "hidden_count": int(hidden.sum()),
        "wrong_count": int(wrong_mask.sum()),
        "full_wrong_count": int(full_wrong.sum()),
        "blank_acc": float(1.0 - (wrong_mask.sum() / max(int(hidden.sum()), 1))),
        "full_acc": float(1.0 - (full_wrong.sum() / 81.0)),
        "exact": bool(np.all(pred == solution)),
        "valid_board": bool(len(conflicts) == 0),
        "conflict_unit_count": int(len(conflicts)),
        "conflict_units": conflicts[:20],
        "wrong_blanks": wrong_coords,
    }


def run(args: argparse.Namespace) -> Dict[str, Any]:
    t0 = time.time()
    repo = Path(args.repo).resolve()
    checkpoint = Path(args.checkpoint).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    os.environ.setdefault("DISABLE_COMPILE", "1")
    os.environ.setdefault("WANDB_MODE", "disabled")
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

    puzzle = parse_board(args.puzzle, allow_blank=True)
    solution = parse_board(args.solution, allow_blank=False)
    if np.any((puzzle != 0) & (puzzle != solution)):
        raise ValueError("puzzle clues do not match solution")

    steps = parse_steps(args.steps)
    max_step = max(steps)

    api = import_official_eqr(repo)
    ckpt, ckpt_path = api["_load_checkpoint"](str(checkpoint))
    eval_cfg = api["EvalConfig"](
        checkpoint=str(checkpoint),
        global_batch_size=args.batch_size,
        seed=args.seed,
        dataset_data_path=args.dataset_data_path,
        load_strict=False,
        loss_head_name="losses@InferenceLossHead",
    )
    config = api["_base_config"](eval_cfg, ckpt, ckpt_path, 0)
    overrides: Dict[str, Any] = {
        "arch.halt_max_steps": max_step,
        "arch.noise_scale": args.noise_scale,
        "arch.H_init_std": args.h_init_std,
        "arch.L_init_std": args.l_init_std,
    }
    api["_apply_overrides"](config, overrides, rank=0)
    api["_seed"](config.seed, 0)

    train_loader, train_metadata = api["create_dataloader"](
        config,
        "train",
        test_set_mode=False,
        epochs_per_iter=1,
        global_batch_size=config.global_batch_size,
        rank=0,
        world_size=1,
    )
    del train_loader

    train_state = api["init_train_state"](config, train_metadata, rank=0, world_size=1)
    api["load_model_state_dict"](
        train_state.model,
        ckpt.get("model", ckpt),
        strict=bool(getattr(config, "load_strict", False)),
        assign=True,
    )
    train_state.step = int(ckpt.get("step", 0))
    if config.ema and "ema" in ckpt:
        train_state = api["load_ema"](ckpt, train_state, config, 0)
    train_state.model.eval()
    api["_apply_runtime_arch_overrides"](train_state.model, overrides, 0)

    device = torch.device("cuda")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this visualization export")
    model = train_state.model.to(device)

    official_inputs = torch.as_tensor((puzzle + 1)[None, :], dtype=torch.int32, device=device)
    official_labels = torch.as_tensor((solution + 1)[None, :], dtype=torch.int32, device=device)
    batch = {
        "inputs": official_inputs,
        "labels": official_labels,
        "puzzle_identifiers": torch.zeros((1,), dtype=torch.int32, device=device),
    }

    predictions: Dict[str, List[int]] = {}
    logits_margin: Dict[str, Dict[str, float]] = {}
    with torch.no_grad():
        carry = model.initial_carry(batch)
        for step in range(1, max_step + 1):
            carry, _loss, _metrics, detached, _all_finish = model(
                carry=carry,
                batch=batch,
                return_keys=["preds", "logits"],
            )
            if step in steps:
                pred_tokens = detached["preds"][0].detach().cpu().numpy().astype(np.int64)
                pred_digits = np.clip(pred_tokens - 1, 0, 9)
                predictions[str(step)] = pred_digits.tolist()
                logits = detached["logits"][0].detach().float().cpu().numpy()
                top2 = np.sort(logits, axis=-1)[:, -2:]
                margins = top2[:, 1] - top2[:, 0]
                hidden = puzzle == 0
                logits_margin[str(step)] = {
                    "mean": float(np.mean(margins)),
                    "hidden_mean": float(np.mean(margins[hidden])) if np.any(hidden) else float(np.mean(margins)),
                }

    metrics = {
        step: sudoku_metrics(np.asarray(pred), solution, puzzle)
        for step, pred in predictions.items()
    }
    payload = {
        "run_name": args.run_name,
        "condition": args.condition,
        "repo": str(repo),
        "checkpoint": str(checkpoint),
        "checkpoint_step": int(ckpt.get("step", 0)),
        "dataset": str(config.dataset.data_path),
        "seed": int(config.seed),
        "steps": steps,
        "puzzle": puzzle.tolist(),
        "solution": solution.tolist(),
        "puzzle_string": board_to_string(puzzle),
        "solution_string": board_to_string(solution, blank="0"),
        "predictions": predictions,
        "metrics": metrics,
        "logit_margin": logits_margin,
        "elapsed_sec": time.time() - t0,
        "notes": args.notes,
    }
    (out_dir / "eqr_custom_case.json").write_text(json.dumps(jsonable(payload), indent=2), encoding="utf-8")
    print(json.dumps({"out": str(out_dir / "eqr_custom_case.json"), "elapsed_sec": payload["elapsed_sec"]}, indent=2))
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Export official EqR Sudoku predictions for one custom 9x9 case.")
    parser.add_argument("--repo", required=True, help="Path to the official EqR clone.")
    parser.add_argument("--checkpoint", required=True, help="Path to an official EqR checkpoint or checkpoint dir.")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--run-name", default="official-eqr-sudoku-custom-case")
    parser.add_argument("--condition", default="official-eqr")
    parser.add_argument("--dataset-data-path", default=None)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--steps", default="1,2,3,5,8,16,32,64")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--noise-scale", type=float, default=0.5)
    parser.add_argument("--h-init-std", type=float, default=1.0)
    parser.add_argument("--l-init-std", type=float, default=1.0)
    parser.add_argument("--puzzle", required=True, help="81 chars using 0 or . for blanks.")
    parser.add_argument("--solution", required=True, help="81 solution digits.")
    parser.add_argument("--notes", default="")
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
