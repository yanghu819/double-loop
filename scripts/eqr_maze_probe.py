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


PathStage = Tuple[int, int, int]


def parse_path_stages(text: str) -> List[PathStage]:
    stages: List[PathStage] = []
    for raw in str(text or "").split(","):
        raw = raw.strip()
        if not raw:
            continue
        if ":" not in raw or "-" not in raw:
            raise ValueError("path stages must look like min-max:steps,min-max:steps")
        range_text, steps_text = raw.split(":", 1)
        min_text, max_text = range_text.split("-", 1)
        min_path = int(min_text)
        max_path = int(max_text)
        steps = int(steps_text)
        if min_path < 1 or max_path < min_path or steps < 1:
            raise ValueError("path stage ranges must be positive and ordered")
        stages.append((min_path, max_path, steps))
    return stages


def path_range_for_step(stages: List[PathStage], step: int, default_range: Tuple[int, int]) -> Tuple[int, int]:
    if not stages:
        return default_range
    total = 0
    for min_path, max_path, duration in stages:
        total += duration
        if step <= total:
            return min_path, max_path
    return stages[-1][0], stages[-1][1]


def make_batch(
    args: argparse.Namespace,
    batch_size: int,
    rng: np.random.Generator,
    device: torch.device,
    path_range: Optional[Tuple[int, int]] = None,
) -> Dict[str, torch.Tensor]:
    inputs: List[np.ndarray] = []
    labels: List[np.ndarray] = []
    min_path, max_path = path_range or (args.min_path_length, args.max_path_length)
    for _ in range(batch_size):
        maze = _sample_maze(
            args.grid_size,
            min_path,
            max_path,
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
    state_update_mode: str = "none",
    state_delta_scale: float = 0.0,
    state_delta_decay: float = 1.0,
) -> Tuple[List[torch.Tensor], List[Dict[str, float]], List[Dict[str, torch.Tensor]]]:
    set_noise_scale(model, noise_scale)
    carry = reset_inner_carry(model, batch)
    logits_by_step: List[torch.Tensor] = []
    residuals: List[Dict[str, float]] = []
    aux_by_step: List[Dict[str, torch.Tensor]] = []
    inner_batch = {"inputs": batch["inputs"], "puzzle_identifiers": batch["puzzle_identifiers"]}
    state_update_mode = str(state_update_mode).lower()
    if state_update_mode not in {"none", "delta_carry", "learned_gate", "state_compete", "state_compete_cross"}:
        raise ValueError(f"unknown state_update_mode: {state_update_mode}")
    for idx in range(steps):
        prev_h = carry.z_H
        prev_l = carry.z_L
        carry, logits, _q = model.inner(carry, inner_batch)
        raw_h = carry.z_H
        raw_l = carry.z_L
        loop_delta_scale = float(state_delta_scale) * (float(state_delta_decay) ** idx)
        if state_update_mode == "delta_carry" and loop_delta_scale != 0.0:
            # Extrapolate a small fraction of the just-computed correction into
            # the next recurrent state. Logits for this loop stay unchanged.
            carry = type(carry)(
                z_H=(raw_h + loop_delta_scale * (raw_h - prev_h)).detach(),
                z_L=(raw_l + loop_delta_scale * (raw_l - prev_l)).detach(),
            )
        logits_by_step.append(logits)
        aux_by_step.append(dict(getattr(model.inner, "latest_state_aux", {})))
        with torch.no_grad():
            raw_zH = (raw_h - prev_h).to(torch.float32).square().mean().sqrt()
            raw_zL = (raw_l - prev_l).to(torch.float32).square().mean().sqrt()
            residuals.append(
                {
                    "zH_rms": float((carry.z_H - prev_h).to(torch.float32).square().mean().sqrt().detach().cpu()),
                    "zL_rms": float((carry.z_L - prev_l).to(torch.float32).square().mean().sqrt().detach().cpu()),
                    "raw_zH_rms": float(raw_zH.detach().cpu()),
                    "raw_zL_rms": float(raw_zL.detach().cpu()),
                    "state_delta_scale": loop_delta_scale,
                    **getattr(model.inner, "latest_state_gate_stats", {}),
                }
            )
    with torch.no_grad():
        for idx in range(len(aux_by_step) - 1):
            context_logits = aux_by_step[idx].get("context_logits")
            if context_logits is None:
                continue
            next_logits = logits_by_step[idx + 1]
            mse = F.mse_loss(context_logits.to(torch.float32), next_logits.detach().to(torch.float32))
            residuals[idx]["state_predict_next_logit_mse"] = float(mse.detach().cpu())
    return logits_by_step, residuals, aux_by_step


def predictive_state_loss(
    logits_by_step: List[torch.Tensor],
    aux_by_step: List[Dict[str, torch.Tensor]],
    horizon: int,
) -> torch.Tensor:
    losses = []
    horizon = max(1, int(horizon))
    for idx, aux in enumerate(aux_by_step):
        target_idx = idx + horizon
        if target_idx >= len(logits_by_step):
            continue
        context_logits = aux.get("context_logits")
        if context_logits is None:
            continue
        target = logits_by_step[target_idx].detach()
        losses.append(F.mse_loss(context_logits.to(torch.float32), target.to(torch.float32)))
    if not losses:
        device = logits_by_step[-1].device if logits_by_step else torch.device("cpu")
        return torch.zeros((), dtype=torch.float32, device=device)
    return torch.stack(losses).mean()


def context_improvement_loss(
    logits_by_step: List[torch.Tensor],
    aux_by_step: List[Dict[str, torch.Tensor]],
    labels: torch.Tensor,
    margin: float,
) -> torch.Tensor:
    losses = []
    margin = float(margin)
    for logits, aux in zip(logits_by_step, aux_by_step):
        context_logits = aux.get("context_logits")
        if context_logits is None:
            continue
        with torch.no_grad():
            current_pred = logits.detach().argmax(dim=-1)
            error_mask = current_pred != labels
            current_ce = F.cross_entropy(
                logits.detach().reshape(-1, logits.shape[-1]),
                labels.reshape(-1),
                reduction="none",
            ).view_as(labels)
        context_ce = F.cross_entropy(
            context_logits.reshape(-1, context_logits.shape[-1]),
            labels.reshape(-1),
            reduction="none",
        ).view_as(labels)
        token_loss = F.relu(context_ce - current_ce + margin)
        if error_mask.any():
            losses.append(token_loss[error_mask].mean())
        else:
            losses.append(token_loss.mean())
    if not losses:
        device = logits_by_step[-1].device if logits_by_step else labels.device
        return torch.zeros((), dtype=torch.float32, device=device)
    return torch.stack(losses).mean()


@torch.no_grad()
def add_context_improvement_diagnostics(
    logits_by_step: List[torch.Tensor],
    aux_by_step: List[Dict[str, torch.Tensor]],
    residuals: List[Dict[str, float]],
    labels: torch.Tensor,
    margin: float,
) -> None:
    margin = float(margin)
    for idx, (logits, aux) in enumerate(zip(logits_by_step, aux_by_step)):
        context_logits = aux.get("context_logits")
        if context_logits is None or idx >= len(residuals):
            continue
        current_pred = logits.detach().argmax(dim=-1)
        error_mask = current_pred != labels
        current_ce = F.cross_entropy(
            logits.detach().reshape(-1, logits.shape[-1]),
            labels.reshape(-1),
            reduction="none",
        ).view_as(labels)
        context_ce = F.cross_entropy(
            context_logits.detach().reshape(-1, context_logits.shape[-1]),
            labels.reshape(-1),
            reduction="none",
        ).view_as(labels)
        token_loss = F.relu(context_ce - current_ce + margin)
        mask = error_mask if error_mask.any() else torch.ones_like(error_mask, dtype=torch.bool)
        current_mean = current_ce[mask].mean()
        context_mean = context_ce[mask].mean()
        residuals[idx].update(
            {
                "context_improve_error_frac": float(error_mask.to(torch.float32).mean().detach().cpu()),
                "context_improve_loss": float(token_loss[mask].mean().detach().cpu()),
                "context_improve_current_ce": float(current_mean.detach().cpu()),
                "context_improve_context_ce": float(context_mean.detach().cpu()),
                "context_improve_advantage": float((current_mean - context_mean).detach().cpu()),
            }
        )


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


@torch.no_grad()
def per_case_path_metrics(pred: torch.Tensor, labels: torch.Tensor) -> Dict[str, torch.Tensor]:
    correct = pred == labels
    path_label = labels == PATH_ID
    path_pred = pred == PATH_ID
    tp = (path_pred & path_label).sum(dim=-1).to(torch.float32)
    fp = (path_pred & ~path_label).sum(dim=-1).to(torch.float32)
    fn = (~path_pred & path_label).sum(dim=-1).to(torch.float32)
    precision = tp / (tp + fp).clamp_min(1.0)
    recall = tp / (tp + fn).clamp_min(1.0)
    f1 = (2 * tp) / (2 * tp + fp + fn).clamp_min(1.0)
    return {
        "label_exact": correct.all(dim=-1).to(torch.float32),
        "path_exact": (path_pred == path_label).all(dim=-1).to(torch.float32),
        "path_f1": f1,
        "path_precision": precision,
        "path_recall": recall,
        "path_pred_frac": path_pred.to(torch.float32).mean(dim=-1),
        "path_label_frac": path_label.to(torch.float32).mean(dim=-1),
        "false_positive_path": fp,
        "false_negative_path": fn,
    }


def parse_loop_list(text: str, max_loop: int) -> List[int]:
    loops: List[int] = []
    for raw in str(text or "").split(","):
        raw = raw.strip()
        if not raw:
            continue
        value = int(raw)
        if 1 <= value <= max_loop and value not in loops:
            loops.append(value)
    if max_loop not in loops:
        loops.append(max_loop)
    return sorted(loops)


def row_chars_from_ids(ids: np.ndarray, n: int, *, target: bool = False) -> List[str]:
    rows: List[str] = []
    chars = {
        WALL_ID: "#",
        OPEN_ID: ".",
        START_ID: "S",
        GOAL_ID: "G",
        PATH_ID: "P" if target else "p",
        PAD_ID: "?",
    }
    grid = ids.reshape(n, n)
    for y in range(n):
        rows.append("".join(chars.get(int(grid[y, x]), "?") for x in range(n)))
    return rows


def comparison_rows(input_ids: np.ndarray, label_ids: np.ndarray, pred_ids: np.ndarray, n: int) -> List[str]:
    rows: List[str] = []
    inp = input_ids.reshape(n, n)
    label_path = label_ids.reshape(n, n) == PATH_ID
    pred_path = pred_ids.reshape(n, n) == PATH_ID
    for y in range(n):
        chars = []
        for x in range(n):
            token = int(inp[y, x])
            if token == WALL_ID:
                chars.append("#")
            elif token == START_ID:
                chars.append("S")
            elif token == GOAL_ID:
                chars.append("G")
            elif label_path[y, x] and pred_path[y, x]:
                chars.append("T")
            elif label_path[y, x]:
                chars.append("M")
            elif pred_path[y, x]:
                chars.append("F")
            else:
                chars.append(".")
        rows.append("".join(chars))
    return rows


def grid_html(rows: List[str], n: int, title: str, stats: Optional[Dict[str, float]] = None) -> str:
    class_map = {
        "#": "wall",
        ".": "open",
        "S": "start",
        "G": "goal",
        "P": "gold",
        "p": "pred",
        "T": "tp",
        "F": "fp",
        "M": "fn",
        "?": "unk",
    }
    stat_text = ""
    if stats:
        stat_text = " ".join(
            f"{key}={value:.4f}" for key, value in stats.items() if isinstance(value, (int, float))
        )
    cells = []
    for row in rows:
        for ch in row:
            cls = class_map.get(ch, "unk")
            cells.append(f'<span class="cell {cls}">{ch}</span>')
    return (
        '<section class="panel">'
        f"<h3>{title}</h3>"
        f'<p class="stats">{stat_text}</p>'
        f'<div class="grid" style="grid-template-columns: repeat({n}, 15px);">'
        + "".join(cells)
        + "</div></section>"
    )


def write_visualizations(out_dir: Path, payload: Dict[str, Any]) -> None:
    viz_dir = out_dir / "visualizations"
    viz_dir.mkdir(parents=True, exist_ok=True)
    (viz_dir / "cases.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_casebook(viz_dir / "casebook.md", payload)
    n = int(payload["grid_size"])
    lines = [
        "<!doctype html>",
        '<html lang="en">',
        "<head>",
        '<meta charset="utf-8">',
        '<meta name="viewport" content="width=device-width, initial-scale=1">',
        "<title>Maze Loop Visualization</title>",
        "<style>",
        "body{font-family:system-ui,-apple-system,Segoe UI,sans-serif;margin:24px;background:#f7f7f4;color:#1f2328}",
        "h1{margin-bottom:4px}.case{border-top:1px solid #d7d7d2;padding-top:18px;margin-top:22px}",
        ".legend{display:flex;flex-wrap:wrap;gap:8px;margin:12px 0 20px}.legend span{padding:2px 7px;border-radius:4px;background:#fff;border:1px solid #ddd}",
        ".panels{display:flex;flex-wrap:wrap;gap:18px;align-items:flex-start}.panel{background:#fff;border:1px solid #d8d8d2;border-radius:8px;padding:10px}",
        ".panel h3{font-size:14px;margin:0 0 4px}.stats{font-size:12px;color:#555;min-height:18px;margin:0 0 8px}",
        ".grid{display:grid;gap:1px;background:#c9c9c2;padding:2px}.cell{width:15px;height:15px;display:inline-flex;align-items:center;justify-content:center;font-size:9px;font-family:ui-monospace,Menlo,monospace;line-height:1}",
        ".wall{background:#252525;color:#252525}.open{background:#f4f1e7;color:#c7c2b6}.start{background:#2f9e44;color:white}.goal{background:#d9480f;color:white}.gold{background:#5c7cfa;color:white}.pred{background:#91a7ff;color:#102a83}.tp{background:#2b8a3e;color:white}.fp{background:#f08c00;color:white}.fn{background:#c2255c;color:white}.unk{background:#aaa;color:white}",
        "</style>",
        "</head>",
        "<body>",
        f"<h1>Maze {n}x{n} Loop Visualization</h1>",
        f"<p>Grid {n}x{n}. T=true path predicted, F=false positive path, M=missed true path.</p>",
        '<div class="legend"><span># wall</span><span>S start</span><span>G goal</span><span>T true positive</span><span>F extra predicted path</span><span>M missed true path</span></div>',
    ]
    for case in payload["cases"]:
        lines.extend(
            [
                '<article class="case">',
                f"<h2>Case {case['case_index']} - {case['reason']}</h2>",
                f"<p>final_f1={case['final_path_f1']:.4f} loop_gain={case['loop_gain']:.4f} exact={case['final_exact']:.4f}</p>",
                '<div class="panels">',
                grid_html(case["target_rows"], n, "target path"),
            ]
        )
        for loop_key in case["loop_order"]:
            lines.append(grid_html(case["comparison_rows"][loop_key], n, loop_key, case["loop_stats"][loop_key]))
        lines.extend(["</div>", "</article>"])
    lines.extend(["</body>", "</html>"])
    (viz_dir / "index.html").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_casebook(path: Path, payload: Dict[str, Any]) -> None:
    n = int(payload["grid_size"])
    lines = [
        f"# Maze {n}x{n} Loop Trajectory Casebook",
        "",
        "Legend: `#` wall, `S` start, `G` goal, `T` correct predicted path, "
        "`F` false-positive path, `M` missed true path, `.` open non-path cell.",
        "",
        "Cases are selected in this order: final failures, hard low-F1 cases, "
        "final over-prediction cases, then largest loop-gain solved cases.",
        "",
    ]
    preferred_loops = {"loop1", "loop2", "loop4", "loop6", "loop10"}
    for case in payload["cases"]:
        stats = case["loop_stats"]
        final_key = case["loop_order"][-1]
        loop1 = stats[case["loop_order"][0]]
        final = stats[final_key]
        lines.extend(
            [
                f"## Case {case['case_index']} ({case['reason']})",
                "",
                (
                    f"Loop gain: `{case['loop_gain']:.4f}`. "
                    f"First loop F1 `{loop1['path_f1']:.4f}` with "
                    f"{int(loop1['false_positive_path'])} false positives and "
                    f"{int(loop1['false_negative_path'])} misses. "
                    f"{final_key} F1 `{final['path_f1']:.4f}` with "
                    f"{int(final['false_positive_path'])} false positives and "
                    f"{int(final['false_negative_path'])} misses. "
                    f"Final exact `{case['final_exact']:.4f}`."
                ),
                "",
            ]
        )
        for loop_key in case["loop_order"]:
            if loop_key not in preferred_loops and loop_key != final_key:
                continue
            lines.extend([f"### {loop_key}", "", "```text"])
            lines.extend(case["comparison_rows"][loop_key])
            lines.extend(["```", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def build_visualization_payload(
    args: argparse.Namespace,
    batch: Dict[str, torch.Tensor],
    logits_by_step: List[torch.Tensor],
    residuals: List[Dict[str, float]],
) -> Dict[str, Any]:
    n = int(args.grid_size)
    loops = parse_loop_list(args.viz_loops, len(logits_by_step))
    labels_cpu = batch["labels"].detach().cpu()
    labels = labels_cpu.numpy()
    inputs = batch["inputs"].detach().cpu().numpy()
    preds_by_loop = {loop: logits_by_step[loop - 1].argmax(dim=-1).detach().cpu() for loop in loops}
    stats_by_loop = {loop: per_case_path_metrics(preds_by_loop[loop], labels_cpu).items() for loop in loops}
    stats_np: Dict[int, Dict[str, np.ndarray]] = {}
    for loop, items in stats_by_loop.items():
        stats_np[loop] = {key: value.detach().cpu().numpy() for key, value in items}
    loop1 = loops[0]
    final_loop = loops[-1]
    final_f1 = stats_np[final_loop]["path_f1"]
    final_exact = stats_np[final_loop]["label_exact"]
    loop_gain = final_f1 - stats_np[loop1]["path_f1"]
    over_pred = stats_np[final_loop]["path_pred_frac"] - stats_np[final_loop]["path_label_frac"]
    final_fp = stats_np[final_loop]["false_positive_path"]
    final_fn = stats_np[final_loop]["false_negative_path"]

    final_failures = np.where(final_exact < 0.5)[0]
    if final_failures.size:
        final_failures = final_failures[
            np.lexsort((-final_fn[final_failures], -final_fp[final_failures], final_f1[final_failures]))
        ]

    orders = [
        ("final failure", final_failures),
        ("hard final low f1", np.argsort(final_f1)),
        ("most final over-prediction", np.argsort(-over_pred)),
        ("largest loop gain", np.argsort(-loop_gain)),
    ]
    selected: List[int] = []
    reasons: Dict[int, str] = {}
    for reason, order in orders:
        for idx in order.tolist():
            if len(selected) >= args.viz_cases:
                break
            if idx in selected:
                continue
            selected.append(int(idx))
            reasons[int(idx)] = reason
        if len(selected) >= args.viz_cases:
            break

    cases = []
    for idx in selected:
        loop_stats: Dict[str, Dict[str, float]] = {}
        comparison: Dict[str, List[str]] = {}
        loop_order: List[str] = []
        for loop in loops:
            key = f"loop{loop}"
            loop_order.append(key)
            loop_stats[key] = {
                stat_key: float(values[idx])
                for stat_key, values in stats_np[loop].items()
                if stat_key
                in {
                    "label_exact",
                    "path_exact",
                    "path_f1",
                    "path_precision",
                    "path_recall",
                    "path_pred_frac",
                    "path_label_frac",
                    "false_positive_path",
                    "false_negative_path",
                }
            }
            comparison[key] = comparison_rows(inputs[idx], labels[idx], preds_by_loop[loop][idx].numpy(), n)
        cases.append(
            {
                "case_index": idx,
                "reason": reasons[idx],
                "loop_order": loop_order,
                "loop_gain": float(loop_gain[idx]),
                "final_path_f1": float(final_f1[idx]),
                "final_exact": float(stats_np[final_loop]["label_exact"][idx]),
                "input_rows": row_chars_from_ids(inputs[idx], n),
                "target_rows": row_chars_from_ids(labels[idx], n, target=True),
                "comparison_rows": comparison,
                "loop_stats": loop_stats,
            }
        )
    return {
        "grid_size": n,
        "viz_loops": loops,
        "residuals": {f"loop{idx}": residuals[idx - 1] for idx in loops},
        "cases": cases,
    }


def train(args: argparse.Namespace, model: torch.nn.Module, device: torch.device) -> Dict[str, Any]:
    rng = np.random.default_rng(args.seed + 1000)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    t0 = time.time()
    history: List[Dict[str, Any]] = []
    path_stages = parse_path_stages(args.path_stages)
    default_range = (args.min_path_length, args.max_path_length)
    last_loss = 0.0
    last_loop1 = 0.0
    last_metrics: Dict[str, float] = {}
    for step in range(1, args.steps + 1):
        model.train()
        train_path_range = path_range_for_step(path_stages, step, default_range)
        batch = make_batch(args, args.batch, rng, device, path_range=train_path_range)
        logits_by_step, residuals, aux_by_step = inner_rollout(
            model,
            batch,
            args.train_loops,
            noise_scale=args.noise_scale,
            state_update_mode=args.state_update_mode,
            state_delta_scale=args.state_delta_scale,
            state_delta_decay=args.state_delta_decay,
        )
        losses = [loss_from_logits(logits, batch["labels"], args.path_loss_weight) for logits in logits_by_step]
        supervised_loss = losses[-1] if args.loop_loss == "final" else torch.stack(losses).mean()
        pred_loss = predictive_state_loss(logits_by_step, aux_by_step, args.predictive_state_horizon)
        improve_loss = context_improvement_loss(
            logits_by_step,
            aux_by_step,
            batch["labels"],
            args.context_improve_margin,
        )
        loss = (
            supervised_loss
            + float(args.predictive_state_weight) * pred_loss
            + float(args.context_improve_weight) * improve_loss
        )
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        opt.step()
        last_loss = float(losses[-1].detach().cpu())
        last_loop1 = float(losses[0].detach().cpu())
        last_pred_loss = float(pred_loss.detach().cpu())
        last_improve_loss = float(improve_loss.detach().cpu())
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
                "predictive_state_loss": last_pred_loss,
                "context_improve_loss": last_improve_loss,
                "train_min_path_length": train_path_range[0],
                "train_max_path_length": train_path_range[1],
                "sec": time.time() - t0,
            }
            history.append(row)
            print(
                f"[eqr_maze] step={step:04d} ce={last_loss:.4f} loop1={last_loop1:.4f} "
                f"path_f1={row['path_f1']:.4f} exact={row['exact']:.4f} "
                f"zH={row['zH_rms']:.4f} zL={row['zL_rms']:.4f} "
                f"pred={row['predictive_state_loss']:.4f} improve={row['context_improve_loss']:.4f} "
                f"path={train_path_range[0]}-{train_path_range[1]}",
                flush=True,
            )
    return {
        "train_ce_loss": last_loss,
        "train_loop1_loss": last_loop1,
        "train_predictive_state_loss": last_pred_loss,
        "train_context_improve_loss": last_improve_loss,
        "train_last_metrics": last_metrics,
        "train_sec": time.time() - t0,
        "history": history,
        "path_stages": [
            {"min_path_length": min_path, "max_path_length": max_path, "steps": steps}
            for min_path, max_path, steps in path_stages
        ],
    }


@torch.no_grad()
def evaluate(args: argparse.Namespace, model: torch.nn.Module, device: torch.device, seed_offset: int = 0) -> Tuple[Dict[str, Any], Optional[Dict[str, Any]]]:
    rng = np.random.default_rng(args.seed + 4000 + seed_offset)
    model.eval()
    batch = make_batch(args, args.eval_n, rng, device)
    logits_by_step, residuals, aux_by_step = inner_rollout(
        model,
        batch,
        args.eval_loops,
        noise_scale=0.0,
        state_update_mode=args.state_update_mode,
        state_delta_scale=args.state_delta_scale,
        state_delta_decay=args.state_delta_decay,
    )
    add_context_improvement_diagnostics(
        logits_by_step,
        aux_by_step,
        residuals,
        batch["labels"],
        args.context_improve_margin,
    )
    eval_clean: Dict[str, Any] = {}
    for idx, logits in enumerate(logits_by_step, start=1):
        eval_clean[f"loop{idx}"] = metrics_from_logits(logits, batch["labels"])
        eval_clean[f"loop{idx}/residual"] = residuals[idx - 1]
    viz_payload = None
    if args.viz_cases > 0:
        viz_payload = build_visualization_payload(args, batch, logits_by_step, residuals)
    return eval_clean, viz_payload


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
        f"- train path stages: `{task['path_stages'] or 'fixed hard range'}`",
        f"- mode: `{task['maze_mode']}`",
        f"- future_seed_scale: {task['future_seed_scale']}",
        f"- state update: `{task['state_update_mode']}` scale={task['state_delta_scale']} decay={task['state_delta_decay']} gate_bias={task.get('state_gate_bias', 2.0)}",
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
    p.add_argument("--path_stages", default="")
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
    p.add_argument(
        "--state_update_mode",
        choices=("none", "delta_carry", "learned_gate", "state_compete", "state_compete_cross"),
        default="none",
    )
    p.add_argument("--state_delta_scale", type=float, default=0.0)
    p.add_argument("--state_delta_decay", type=float, default=1.0)
    p.add_argument("--state_gate_bias", type=float, default=2.0)
    p.add_argument("--forward_dtype", default="bfloat16")
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight_decay", type=float, default=0.1)
    p.add_argument("--path_loss_weight", type=float, default=4.0)
    p.add_argument("--loop_loss", choices=("final", "all"), default="all")
    p.add_argument("--predictive_state_weight", type=float, default=0.0)
    p.add_argument("--predictive_state_horizon", type=int, default=1)
    p.add_argument("--context_improve_weight", type=float, default=0.0)
    p.add_argument("--context_improve_margin", type=float, default=0.01)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=52)
    p.add_argument("--log_every", type=int, default=100)
    p.add_argument("--viz_cases", type=int, default=0)
    p.add_argument("--viz_loops", default="1,2,4,6,10")
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
    parse_path_stages(args.path_stages)
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
        "state_update_mode": args.state_update_mode,
        "state_delta_scale": args.state_delta_scale,
        "state_delta_decay": args.state_delta_decay,
        "state_gate_bias": args.state_gate_bias,
        "H_init_std": 1.0,
        "L_init_std": 1.0,
    }
    model = EqRModel(config).to(device)
    param_count = sum(p.numel() for p in model.parameters())
    print(f"device={device} torch={torch.__version__} params={param_count} eqr_dir={eqr_dir}", flush=True)

    train_metrics = train(args, model, device)
    eval_clean, viz_payload = evaluate(args, model, device)
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
            "path_stages": args.path_stages,
            "train_loops": args.train_loops,
            "eval_loops": args.eval_loops,
            "hidden_size": args.hidden_size,
            "heads": args.heads,
            "layers": args.layers,
            "h_cycles": args.h_cycles,
            "l_cycles": args.l_cycles,
            "future_seed_scale": args.future_seed_scale,
            "state_update_mode": args.state_update_mode,
            "state_delta_scale": args.state_delta_scale,
            "state_delta_decay": args.state_delta_decay,
            "state_gate_bias": args.state_gate_bias,
            "predictive_state_weight": args.predictive_state_weight,
            "predictive_state_horizon": args.predictive_state_horizon,
            "context_improve_weight": args.context_improve_weight,
            "context_improve_margin": args.context_improve_margin,
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
    if viz_payload is not None:
        write_visualizations(out_dir, viz_payload)
    print(
        f"primary loop{args.eval_loops} exact={final['label_exact']:.4f} "
        f"path_f1={final['path_f1']:.4f} gain={loop_gain:.4f}"
    )
    print(f"decision={decision}")
    print(f"wrote {json_path}")
    print(f"wrote {md_path}")


if __name__ == "__main__":
    main()
