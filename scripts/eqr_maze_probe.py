#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import json
import random
import sys
import time
import types
from collections import deque
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F


CHARSET = "# SGo"
PAD_ID = 0
WALL_ID = 1
OPEN_ID = 2
START_ID = 3
GOAL_ID = 4
PATH_ID = 5
VOCAB_SIZE = len(CHARSET) + 1
DIRS = ((1, 0), (-1, 0), (0, 1), (0, -1))


def add_import_paths(repo_root: Path, eqr_dir: Path) -> None:
    sys.path.insert(0, str(eqr_dir))


def install_colorama_fallback() -> None:
    if importlib.util.find_spec("colorama") is not None:
        return

    class _Codes:
        BLACK = RED = GREEN = YELLOW = BLUE = MAGENTA = CYAN = WHITE = RESET = ""
        RESET_ALL = BRIGHT = DIM = NORMAL = ""

    module = types.ModuleType("colorama")
    module.Fore = _Codes()
    module.Style = _Codes()
    module.init = lambda *_args, **_kwargs: None
    sys.modules["colorama"] = module


def _bfs(open_mask: np.ndarray, start: Tuple[int, int]) -> Tuple[np.ndarray, np.ndarray]:
    n = open_mask.shape[0]
    dist = np.full((n, n), -1, dtype=np.int32)
    parent = np.full((n, n, 2), -1, dtype=np.int32)
    queue: deque[Tuple[int, int]] = deque([start])
    dist[start] = 0
    while queue:
        y, x = queue.popleft()
        for dy, dx in DIRS:
            ny, nx = y + dy, x + dx
            if 0 <= ny < n and 0 <= nx < n and open_mask[ny, nx] and dist[ny, nx] < 0:
                dist[ny, nx] = dist[y, x] + 1
                parent[ny, nx] = (y, x)
                queue.append((ny, nx))
    return dist, parent


def _generate_perfect_maze(n: int, rng: np.random.Generator) -> np.ndarray:
    if n < 3 or n % 2 == 0:
        raise ValueError("perfect maze probe requires an odd grid size >= 3")
    open_mask = np.zeros((n, n), dtype=bool)
    cells_h = (n - 1) // 2
    cells_w = (n - 1) // 2
    visited = np.zeros((cells_h, cells_w), dtype=bool)

    def cell_to_grid(r: int, c: int) -> Tuple[int, int]:
        return 2 * r + 1, 2 * c + 1

    start = (int(rng.integers(cells_h)), int(rng.integers(cells_w)))
    stack = [start]
    visited[start] = True
    open_mask[cell_to_grid(*start)] = True
    while stack:
        r, c = stack[-1]
        neighbors = []
        for dr, dc in DIRS:
            nr, nc = r + dr, c + dc
            if 0 <= nr < cells_h and 0 <= nc < cells_w and not visited[nr, nc]:
                neighbors.append((nr, nc))
        if not neighbors:
            stack.pop()
            continue
        nr, nc = neighbors[int(rng.integers(len(neighbors)))]
        visited[nr, nc] = True
        gr, gc = cell_to_grid(r, c)
        ngr, ngc = cell_to_grid(nr, nc)
        open_mask[(gr + ngr) // 2, (gc + ngc) // 2] = True
        open_mask[ngr, ngc] = True
        stack.append((nr, nc))
    return open_mask


def _reconstruct(parent: np.ndarray, start: Tuple[int, int], goal: Tuple[int, int]) -> Optional[List[Tuple[int, int]]]:
    path = [goal]
    cur = goal
    while cur != start:
        py, px = parent[cur]
        if py < 0:
            return None
        cur = (int(py), int(px))
        path.append(cur)
    path.reverse()
    return path


def _sample_random_connected_maze(n: int, wall_prob: float, rng: np.random.Generator) -> np.ndarray:
    open_mask = rng.random((n, n)) >= wall_prob
    if int(open_mask.sum()) < 2:
        open_mask[rng.integers(n), rng.integers(n)] = True
        open_mask[rng.integers(n), rng.integers(n)] = True
    return open_mask


def _sample_maze(
    n: int,
    min_path: int,
    max_path: int,
    mode: str,
    wall_prob: float,
    rng: np.random.Generator,
    max_grid_attempts: int,
    max_start_attempts: int,
) -> Tuple[np.ndarray, Tuple[int, int], Tuple[int, int], List[Tuple[int, int]]]:
    if min_path < 1 or max_path < min_path:
        raise ValueError("path length range must be positive and ordered")
    mode = mode.lower()
    for _ in range(max_grid_attempts):
        if mode == "perfect":
            open_mask = _generate_perfect_maze(n, rng)
        elif mode == "random":
            open_mask = _sample_random_connected_maze(n, wall_prob, rng)
        else:
            raise ValueError(f"unknown maze mode: {mode}")

        open_positions = np.argwhere(open_mask)
        if open_positions.shape[0] < min_path + 1:
            continue
        for _ in range(max_start_attempts):
            sy, sx = open_positions[int(rng.integers(open_positions.shape[0]))]
            start = (int(sy), int(sx))
            dist, parent = _bfs(open_mask, start)
            candidates = np.argwhere((dist >= min_path) & (dist <= max_path))
            if candidates.shape[0] == 0:
                continue
            gy, gx = candidates[int(rng.integers(candidates.shape[0]))]
            goal = (int(gy), int(gx))
            path = _reconstruct(parent, start, goal)
            if path is not None:
                return open_mask, start, goal, path
    raise RuntimeError("failed to sample maze; relax path range or wall probability")


def _maze_to_ids(open_mask: np.ndarray, start: Tuple[int, int], goal: Tuple[int, int], path: List[Tuple[int, int]]) -> Tuple[np.ndarray, np.ndarray]:
    n = open_mask.shape[0]
    inputs = np.full((n, n), OPEN_ID, dtype=np.int64)
    inputs[~open_mask] = WALL_ID
    inputs[start] = START_ID
    inputs[goal] = GOAL_ID
    labels = inputs.copy()
    for y, x in path[1:-1]:
        labels[y, x] = PATH_ID
    return inputs.reshape(-1), labels.reshape(-1)


def make_batch(args: argparse.Namespace, batch_size: int, rng: np.random.Generator, device: torch.device) -> Dict[str, torch.Tensor]:
    inputs: List[np.ndarray] = []
    labels: List[np.ndarray] = []
    for _ in range(batch_size):
        maze = _sample_maze(
            args.grid_size,
            args.min_path_length,
            args.max_path_length,
            args.maze_mode,
            args.wall_prob,
            rng,
            args.max_grid_attempts,
            args.max_start_attempts,
        )
        inp, out = _maze_to_ids(*maze)
        inputs.append(inp)
        labels.append(out)
    return {
        "inputs": torch.as_tensor(np.stack(inputs), dtype=torch.long, device=device),
        "labels": torch.as_tensor(np.stack(labels), dtype=torch.long, device=device),
        "puzzle_identifiers": torch.zeros((batch_size,), dtype=torch.long, device=device),
    }


def set_noise_scale(model: torch.nn.Module, value: float) -> None:
    inner = getattr(model, "inner", model)
    if hasattr(getattr(inner, "config", None), "noise_scale"):
        inner.config.noise_scale = float(value)
    for module in inner.modules():
        if hasattr(module, "noise_scale"):
            module.noise_scale = float(value)


def reset_inner_carry(model: torch.nn.Module, batch: Dict[str, torch.Tensor]) -> Any:
    inner = model.inner
    batch_size = int(batch["inputs"].shape[0])
    carry = inner.empty_carry(batch_size, device=batch["inputs"].device)
    reset = torch.ones((batch_size,), dtype=torch.bool, device=batch["inputs"].device)
    return inner.reset_carry(reset, carry)


def inner_rollout(
    model: torch.nn.Module,
    batch: Dict[str, torch.Tensor],
    steps: int,
    noise_scale: float,
) -> Tuple[List[torch.Tensor], List[Dict[str, float]]]:
    set_noise_scale(model, noise_scale)
    carry = reset_inner_carry(model, batch)
    logits_by_step: List[torch.Tensor] = []
    residuals: List[Dict[str, float]] = []
    inner_batch = {"inputs": batch["inputs"], "puzzle_identifiers": batch["puzzle_identifiers"]}
    for _ in range(steps):
        prev_h = carry.z_H
        prev_l = carry.z_L
        carry, logits, _q = model.inner(carry, inner_batch)
        logits_by_step.append(logits)
        with torch.no_grad():
            residuals.append(
                {
                    "zH_rms": float((carry.z_H - prev_h).to(torch.float32).square().mean().sqrt().detach().cpu()),
                    "zL_rms": float((carry.z_L - prev_l).to(torch.float32).square().mean().sqrt().detach().cpu()),
                }
            )
    return logits_by_step, residuals


def loss_from_logits(logits: torch.Tensor, labels: torch.Tensor, path_weight: float) -> torch.Tensor:
    loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), labels.reshape(-1), reduction="none").view_as(labels)
    if path_weight != 1.0:
        weights = torch.where(labels == PATH_ID, torch.full_like(loss, float(path_weight)), torch.ones_like(loss))
        return (loss * weights).sum() / weights.sum().clamp_min(1.0)
    return loss.mean()


@torch.no_grad()
def metrics_from_logits(logits: torch.Tensor, labels: torch.Tensor) -> Dict[str, float]:
    pred = logits.argmax(dim=-1)
    correct = pred == labels
    path_label = labels == PATH_ID
    path_pred = pred == PATH_ID
    tp = (path_pred & path_label).sum(dim=-1).to(torch.float32)
    fp = (path_pred & ~path_label).sum(dim=-1).to(torch.float32)
    fn = (~path_pred & path_label).sum(dim=-1).to(torch.float32)
    precision = tp / (tp + fp).clamp_min(1.0)
    recall = tp / (tp + fn).clamp_min(1.0)
    f1 = (2 * tp) / (2 * tp + fp + fn).clamp_min(1.0)
    static_mask = labels != PATH_ID
    return {
        "label_exact": float(correct.all(dim=-1).to(torch.float32).mean().cpu()),
        "token_acc": float(correct.to(torch.float32).mean().cpu()),
        "path_exact": float((path_pred == path_label).all(dim=-1).to(torch.float32).mean().cpu()),
        "path_precision": float(precision.mean().cpu()),
        "path_recall": float(recall.mean().cpu()),
        "path_f1": float(f1.mean().cpu()),
        "path_pred_frac": float(path_pred.to(torch.float32).mean().cpu()),
        "path_label_frac": float(path_label.to(torch.float32).mean().cpu()),
        "static_acc": float(correct[static_mask].to(torch.float32).mean().cpu()),
    }


def train(args: argparse.Namespace, model: torch.nn.Module, device: torch.device) -> Dict[str, Any]:
    rng = np.random.default_rng(args.seed + 1000)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    t0 = time.time()
    history: List[Dict[str, Any]] = []
    last_loss = 0.0
    last_loop1 = 0.0
    last_metrics: Dict[str, float] = {}
    for step in range(1, args.steps + 1):
        model.train()
        batch = make_batch(args, args.batch, rng, device)
        logits_by_step, residuals = inner_rollout(model, batch, args.train_loops, noise_scale=args.noise_scale)
        losses = [loss_from_logits(logits, batch["labels"], args.path_loss_weight) for logits in logits_by_step]
        loss = losses[-1] if args.loop_loss == "final" else torch.stack(losses).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        opt.step()
        last_loss = float(losses[-1].detach().cpu())
        last_loop1 = float(losses[0].detach().cpu())
        if args.log_every and step % args.log_every == 0:
            last_metrics = metrics_from_logits(logits_by_step[-1], batch["labels"])
            row = {
                "step": step,
                "ce": last_loss,
                "loop1_ce": last_loop1,
                "path_f1": last_metrics["path_f1"],
                "exact": last_metrics["label_exact"],
                "zH_rms": residuals[-1]["zH_rms"],
                "zL_rms": residuals[-1]["zL_rms"],
                "sec": time.time() - t0,
            }
            history.append(row)
            print(
                f"[eqr_maze] step={step:04d} ce={last_loss:.4f} loop1={last_loop1:.4f} "
                f"path_f1={row['path_f1']:.4f} exact={row['exact']:.4f} "
                f"zH={row['zH_rms']:.4f} zL={row['zL_rms']:.4f}",
                flush=True,
            )
    return {
        "train_ce_loss": last_loss,
        "train_loop1_loss": last_loop1,
        "train_last_metrics": last_metrics,
        "train_sec": time.time() - t0,
        "history": history,
    }


@torch.no_grad()
def evaluate(args: argparse.Namespace, model: torch.nn.Module, device: torch.device, seed_offset: int = 0) -> Dict[str, Any]:
    rng = np.random.default_rng(args.seed + 4000 + seed_offset)
    model.eval()
    batch = make_batch(args, args.eval_n, rng, device)
    logits_by_step, residuals = inner_rollout(model, batch, args.eval_loops, noise_scale=0.0)
    eval_clean: Dict[str, Any] = {}
    for idx, logits in enumerate(logits_by_step, start=1):
        eval_clean[f"loop{idx}"] = metrics_from_logits(logits, batch["labels"])
        eval_clean[f"loop{idx}/residual"] = residuals[idx - 1]
    return eval_clean


def write_report(path: Path, metrics: Dict[str, Any]) -> None:
    task = metrics["task"]
    train = metrics["train"]
    lines = [
        "# EqR Maze Probe",
        "",
        "Question: does recurrent FutureSeed-style state mixing transfer from Sudoku to path propagation?",
        "",
        f"- grid: {task['grid_size']}x{task['grid_size']}",
        f"- path range: {task['min_path_length']}-{task['max_path_length']}",
        f"- mode: `{task['maze_mode']}`",
        f"- future_seed_scale: {task['future_seed_scale']}",
        f"- train loops: {task['train_loops']}",
        f"- eval loops: {task['eval_loops']}",
        f"- train CE: {train['train_ce_loss']:.4f}",
        f"- train seconds: {train['train_sec']:.1f}",
        "",
        "## Clean Eval",
        "",
    ]
    for idx in range(1, task["eval_loops"] + 1):
        row = metrics["eval_clean"][f"loop{idx}"]
        res = metrics["eval_clean"][f"loop{idx}/residual"]
        lines.append(
            f"- loop{idx}: exact={row['label_exact']:.4f}, path_f1={row['path_f1']:.4f}, "
            f"path_exact={row['path_exact']:.4f}, token_acc={row['token_acc']:.4f}, "
            f"zH={res['zH_rms']:.4f}, zL={res['zL_rms']:.4f}"
        )
    lines.extend(["", "## Decision", "", metrics["decision"], ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--repo_root", default=str(Path(__file__).resolve().parents[1]))
    p.add_argument("--eqr_dir", default="")
    p.add_argument("--out_dir", required=True)
    p.add_argument("--grid_size", type=int, default=15)
    p.add_argument("--maze_mode", choices=("perfect", "random"), default="perfect")
    p.add_argument("--wall_prob", type=float, default=0.37)
    p.add_argument("--min_path_length", type=int, default=32)
    p.add_argument("--max_path_length", type=int, default=56)
    p.add_argument("--max_grid_attempts", type=int, default=200)
    p.add_argument("--max_start_attempts", type=int, default=200)
    p.add_argument("--steps", type=int, default=600)
    p.add_argument("--batch", type=int, default=128)
    p.add_argument("--eval_n", type=int, default=512)
    p.add_argument("--hidden_size", type=int, default=128)
    p.add_argument("--heads", type=int, default=4)
    p.add_argument("--layers", type=int, default=1)
    p.add_argument("--h_cycles", type=int, default=2)
    p.add_argument("--l_cycles", type=int, default=4)
    p.add_argument("--train_loops", type=int, default=4)
    p.add_argument("--eval_loops", type=int, default=8)
    p.add_argument("--expansion", type=float, default=4.0)
    p.add_argument("--lambda_", type=float, default=0.95)
    p.add_argument("--noise_scale", type=float, default=0.0)
    p.add_argument("--noise_mode", choices=("gaussian", "feature_diff", "none"), default="none")
    p.add_argument("--future_seed_scale", type=float, default=1.0)
    p.add_argument("--future_seed_gate_bias", type=float, default=-2.0)
    p.add_argument("--forward_dtype", default="bfloat16")
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight_decay", type=float, default=0.1)
    p.add_argument("--path_loss_weight", type=float, default=4.0)
    p.add_argument("--loop_loss", choices=("final", "all"), default="all")
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=52)
    p.add_argument("--log_every", type=int, default=100)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = Path(args.repo_root).resolve()
    eqr_dir = Path(args.eqr_dir or repo_root / "repos" / "eqr").resolve()
    add_import_paths(repo_root, eqr_dir)
    install_colorama_fallback()

    from models.eqr import EqRModel

    if args.grid_size % 2 == 0 and args.maze_mode == "perfect":
        raise ValueError("--grid_size must be odd for perfect mazes")
    if args.hidden_size % args.heads != 0:
        raise ValueError("--hidden_size must be divisible by --heads")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("eqr_maze_probe requires CUDA; CPU smoke/training is intentionally disabled")
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    config = {
        "batch_size": args.batch,
        "seq_len": args.grid_size * args.grid_size,
        "vocab_size": VOCAB_SIZE,
        "num_puzzle_identifiers": 1,
        "H_cycles": args.h_cycles,
        "L_cycles": args.l_cycles,
        "H_layers": 0,
        "L_layers": args.layers,
        "hidden_size": args.hidden_size,
        "expansion": args.expansion,
        "num_heads": args.heads,
        "pos_encodings": "rope2d",
        "board_height": args.grid_size,
        "board_width": args.grid_size,
        "halt_max_steps": args.train_loops,
        "halt_exploration_prob": 0.0,
        "forward_dtype": args.forward_dtype,
        "lambda_": args.lambda_,
        "noise_scale": args.noise_scale,
        "noise_mode": args.noise_mode,
        "feature_noise_buffer_size": 1,
        "feature_noise_buffer_add": 0,
        "feature_noise_fallback": "zero",
        "future_seed_scale": args.future_seed_scale,
        "future_seed_gate_bias": args.future_seed_gate_bias,
        "H_init_std": 1.0,
        "L_init_std": 1.0,
    }
    model = EqRModel(config).to(device)
    param_count = sum(p.numel() for p in model.parameters())
    print(f"device={device} torch={torch.__version__} params={param_count} eqr_dir={eqr_dir}", flush=True)

    train_metrics = train(args, model, device)
    eval_clean = evaluate(args, model, device)
    final = eval_clean[f"loop{args.eval_loops}"]
    loop1 = eval_clean["loop1"]
    loop_gain = float(final["path_f1"]) - float(loop1["path_f1"])
    if train_metrics["train_ce_loss"] > 1.5:
        decision = "Stop: optimization is not yet working on this maze setting; reduce path length or capacity-test before mechanism claims."
    elif loop_gain >= 0.05 and final["path_f1"] >= 0.35:
        decision = "Continue: recurrence is buying real path refinement on maze; compare against base or scale path length next."
    elif final["path_f1"] >= 0.35:
        decision = "Continue cautiously: maze path learning is alive, but loop gain is weak; inspect residuals before scaling."
    else:
        decision = "Pivot: path F1 remains low; shorten path curriculum or increase capacity before judging FutureSeed transfer."

    metrics = {
        "task": {
            "model": "EqRModel",
            "grid_size": args.grid_size,
            "maze_mode": args.maze_mode,
            "min_path_length": args.min_path_length,
            "max_path_length": args.max_path_length,
            "train_loops": args.train_loops,
            "eval_loops": args.eval_loops,
            "hidden_size": args.hidden_size,
            "heads": args.heads,
            "layers": args.layers,
            "h_cycles": args.h_cycles,
            "l_cycles": args.l_cycles,
            "future_seed_scale": args.future_seed_scale,
            "params": param_count,
        },
        "train": train_metrics,
        "eval_clean": eval_clean,
        "decision": decision,
    }
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {"metrics": metrics, "config": vars(args)}
    suffix = f"fs{args.future_seed_scale:g}_seed{args.seed}"
    json_path = out_dir / f"eqr_maze_probe_{suffix}.json"
    md_path = out_dir / f"eqr_maze_probe_{suffix}.md"
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_report(md_path, metrics)
    print(
        f"primary loop{args.eval_loops} exact={final['label_exact']:.4f} "
        f"path_f1={final['path_f1']:.4f} gain={loop_gain:.4f}"
    )
    print(f"decision={decision}")
    print(f"wrote {json_path}")
    print(f"wrote {md_path}")


if __name__ == "__main__":
    main()
