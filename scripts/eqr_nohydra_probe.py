#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch
import torch.nn.functional as F


def add_import_paths(repo_root: Path, eqr_dir: Path) -> None:
    sys.path.insert(0, str(eqr_dir))
    sys.path.insert(0, str(repo_root / "experiments" / "rwkv_fs_sudoku"))


def parse_stages(raw: str, holes_min: int, holes_max: int, steps: int) -> List[Tuple[int, int, int]]:
    if not raw.strip():
        return [(holes_min, holes_max, steps)]
    stages: List[Tuple[int, int, int]] = []
    for item in raw.split(","):
        span, stage_steps = item.strip().split(":", 1)
        lo, hi = span.split("-", 1)
        stages.append((int(lo), int(hi), int(stage_steps)))
    for lo, hi, stage_steps in stages:
        if lo < 1 or hi < lo or stage_steps < 1:
            raise ValueError("--hole_stages entries must look like 4-8:100")
    return stages


def set_noise_scale(model: torch.nn.Module, value: float) -> None:
    inner = getattr(model, "inner", model)
    if hasattr(getattr(inner, "config", None), "noise_scale"):
        inner.config.noise_scale = float(value)
    for module in inner.modules():
        if hasattr(module, "noise_scale"):
            module.noise_scale = float(value)


def make_eqr_batch(sudoku: Any, batch_size: int, holes_min: int, holes_max: int, pattern: str, rng: random.Random, device: torch.device) -> Dict[str, torch.Tensor]:
    inputs, labels, clue_mask = sudoku.make_batch(batch_size, holes_min, holes_max, pattern, rng, device=device)
    return {
        "inputs": inputs,
        "labels": labels,
        "clue_mask": clue_mask,
        "puzzle_identifiers": torch.zeros((batch_size,), dtype=torch.long, device=device),
    }


def reset_inner_carry(model: torch.nn.Module, batch: Dict[str, torch.Tensor]) -> Any:
    inner = model.inner
    batch_size = int(batch["inputs"].shape[0])
    carry = inner.empty_carry(batch_size, device=batch["inputs"].device)
    reset = torch.ones((batch_size,), dtype=torch.bool, device=batch["inputs"].device)
    return inner.reset_carry(reset, carry)


def inner_rollout(model: torch.nn.Module, batch: Dict[str, torch.Tensor], steps: int, noise_scale: float) -> List[torch.Tensor]:
    set_noise_scale(model, noise_scale)
    carry = reset_inner_carry(model, batch)
    logits_by_step: List[torch.Tensor] = []
    inner_batch = {
        "inputs": batch["inputs"],
        "puzzle_identifiers": batch["puzzle_identifiers"],
    }
    for _ in range(steps):
        carry, logits, _q = model.inner(carry, inner_batch)
        logits_by_step.append(logits)
    return logits_by_step


def digit_logits(logits: torch.Tensor, num_digits: int) -> torch.Tensor:
    return logits[..., :num_digits].contiguous()


def loss_from_logits(logits: torch.Tensor, labels: torch.Tensor, clue_mask: torch.Tensor, num_digits: int, blank_weight: float) -> torch.Tensor:
    logits = digit_logits(logits, num_digits)
    loss = F.cross_entropy(logits.reshape(-1, num_digits), labels.reshape(-1), reduction="none").view_as(labels)
    if blank_weight == 1.0:
        return loss.mean()
    weights = torch.where(clue_mask, torch.ones_like(loss), torch.full_like(loss, float(blank_weight)))
    return (loss * weights).sum() / weights.sum().clamp_min(1.0)


@torch.no_grad()
def evaluate_clean(
    sudoku: Any,
    model: torch.nn.Module,
    *,
    eval_n: int,
    holes: int,
    pattern: str,
    steps: int,
    seed: int,
    device: torch.device,
) -> Dict[str, Any]:
    rng = random.Random(seed)
    batch = make_eqr_batch(sudoku, eval_n, holes, holes, pattern, rng, device)
    logits_by_step = inner_rollout(model, batch, steps, noise_scale=0.0)
    out: Dict[str, Any] = {}
    for idx, logits in enumerate(logits_by_step, start=1):
        metrics, _pred = sudoku.metrics_from_logits(digit_logits(logits, sudoku.N), batch["labels"], batch["clue_mask"])
        out[f"loop{idx}"] = asdict(metrics)
    return out


def train(args: argparse.Namespace, sudoku: Any, model: torch.nn.Module, device: torch.device) -> Dict[str, Any]:
    rng = random.Random(args.seed + 1000)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    stages = parse_stages(args.hole_stages, args.holes_min, args.holes_max, args.steps)
    t0 = time.time()
    step = 0
    history: List[Dict[str, Any]] = []
    last_loss = 0.0
    last_loop1 = 0.0

    for stage_idx, (holes_min, holes_max, stage_steps) in enumerate(stages, start=1):
        for _ in range(stage_steps):
            step += 1
            model.train()
            batch = make_eqr_batch(sudoku, args.batch, holes_min, holes_max, args.hole_pattern, rng, device)
            logits_by_step = inner_rollout(model, batch, args.train_loops, noise_scale=args.noise_scale)
            losses = [
                loss_from_logits(logits, batch["labels"], batch["clue_mask"], sudoku.N, args.blank_loss_weight)
                for logits in logits_by_step
            ]
            loss = losses[-1] if args.loop_loss == "final" else torch.stack(losses).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            opt.step()
            last_loss = float(losses[-1].detach().cpu())
            last_loop1 = float(losses[0].detach().cpu())
            if args.log_every and step % args.log_every == 0:
                row = {
                    "step": step,
                    "stage": stage_idx,
                    "holes_min": holes_min,
                    "holes_max": holes_max,
                    "ce": last_loss,
                    "loop1_ce": last_loop1,
                    "sec": time.time() - t0,
                }
                history.append(row)
                print(
                    f"[eqr_nohydra stage={stage_idx}:{holes_min}-{holes_max}] "
                    f"step={step:04d} ce={last_loss:.4f} loop1={last_loop1:.4f} sec={row['sec']:.1f}",
                    flush=True,
                )

    return {
        "train_ce_loss": last_loss,
        "train_loop1_loss": last_loop1,
        "train_sec": time.time() - t0,
        "history": history,
        "noise_mode": args.noise_mode,
        "future_seed_scale": args.future_seed_scale,
    }


def write_report(path: Path, metrics: Dict[str, Any]) -> None:
    task = metrics["task"]
    train_metrics = metrics["train"]
    max_loop = int(task["train_loops"])
    lines = [
        f"# EqR No-Hydra FutureSeed Probe",
        "",
        "Mechanism: EqR recurrent attractor update with opt-in FutureSeed cross-level mixers and feature-diff noise.",
        "",
        f"- board: {task['size']}x{task['size']}",
        f"- holes: {task['hole_stages'] or str(task['holes_min']) + '-' + str(task['holes_max'])}",
        f"- noise_mode: `{train_metrics['noise_mode']}`",
        f"- future_seed_scale: {train_metrics['future_seed_scale']}",
        f"- train_ce: {train_metrics['train_ce_loss']:.4f}",
        f"- train_sec: {train_metrics['train_sec']:.1f}",
        "",
        "## Clean Eval",
        "",
    ]
    for holes_key, row in metrics["eval_by_holes"].items():
        loop = row["eval_clean"][f"loop{max_loop}"]
        lines.append(
            f"- {holes_key}: exact={loop['label_exact']:.4f}, "
            f"blank={loop['blank_acc']:.4f}, valid={loop['valid_sudoku']:.4f}"
        )
    lines.extend(["", "## Decision", "", metrics["decision"], ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--repo_root", default=str(Path(__file__).resolve().parents[1]))
    p.add_argument("--eqr_dir", default="")
    p.add_argument("--out_dir", required=True)
    p.add_argument("--size", type=int, default=9)
    p.add_argument("--box_rows", type=int, default=0)
    p.add_argument("--box_cols", type=int, default=0)
    p.add_argument("--steps", type=int, default=120)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--eval_n", type=int, default=128)
    p.add_argument("--holes_min", type=int, default=4)
    p.add_argument("--holes_max", type=int, default=10)
    p.add_argument("--hole_stages", default="")
    p.add_argument("--hole_pattern", choices=("random", "row", "col", "box", "unit"), default="random")
    p.add_argument("--eval_holes", type=int, default=8)
    p.add_argument("--eval_holes_list", default="8,12")
    p.add_argument("--hidden_size", type=int, default=192)
    p.add_argument("--heads", type=int, default=6)
    p.add_argument("--layers", type=int, default=2)
    p.add_argument("--h_cycles", type=int, default=2)
    p.add_argument("--l_cycles", type=int, default=4)
    p.add_argument("--train_loops", type=int, default=4)
    p.add_argument("--expansion", type=float, default=4.0)
    p.add_argument("--lambda_", type=float, default=0.95)
    p.add_argument("--noise_scale", type=float, default=0.01)
    p.add_argument("--noise_mode", choices=("gaussian", "feature_diff", "none"), default="feature_diff")
    p.add_argument("--feature_noise_buffer_size", type=int, default=512)
    p.add_argument("--feature_noise_buffer_add", type=int, default=64)
    p.add_argument("--future_seed_scale", type=float, default=1.0)
    p.add_argument("--future_seed_gate_bias", type=float, default=-2.0)
    p.add_argument("--forward_dtype", default="bfloat16")
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight_decay", type=float, default=0.1)
    p.add_argument("--blank_loss_weight", type=float, default=8.0)
    p.add_argument("--loop_loss", choices=("final", "all"), default="final")
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=52)
    p.add_argument("--log_every", type=int, default=50)
    p.add_argument("--cpu", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = Path(args.repo_root).resolve()
    eqr_dir = Path(args.eqr_dir or repo_root / "repos" / "eqr").resolve()
    add_import_paths(repo_root, eqr_dir)

    import study_rwkv_futureseed_loop as sudoku
    from models.eqr import EqRModel

    sudoku.configure_sudoku(args.size, args.box_rows, args.box_cols)
    if args.hidden_size % args.heads != 0:
        raise ValueError("--hidden_size must be divisible by --heads")
    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    torch.manual_seed(args.seed)

    config = {
        "batch_size": args.batch,
        "seq_len": sudoku.CELLS,
        "vocab_size": sudoku.VOCAB,
        "num_puzzle_identifiers": 1,
        "H_cycles": args.h_cycles,
        "L_cycles": args.l_cycles,
        "H_layers": 0,
        "L_layers": args.layers,
        "hidden_size": args.hidden_size,
        "expansion": args.expansion,
        "num_heads": args.heads,
        "pos_encodings": "rope2d",
        "board_height": sudoku.N,
        "board_width": sudoku.N,
        "halt_max_steps": args.train_loops,
        "halt_exploration_prob": 0.0,
        "forward_dtype": args.forward_dtype,
        "lambda_": args.lambda_,
        "noise_scale": args.noise_scale,
        "noise_mode": args.noise_mode,
        "feature_noise_buffer_size": args.feature_noise_buffer_size,
        "feature_noise_buffer_add": args.feature_noise_buffer_add,
        "feature_noise_fallback": "gaussian",
        "future_seed_scale": args.future_seed_scale,
        "future_seed_gate_bias": args.future_seed_gate_bias,
        "H_init_std": 1.0,
        "L_init_std": 1.0,
    }
    model = EqRModel(config).to(device)
    param_count = sum(p.numel() for p in model.parameters())
    print(f"device={device} torch={torch.__version__} params={param_count} eqr_dir={eqr_dir}", flush=True)

    train_metrics = train(args, sudoku, model, device)
    model.eval()

    eval_holes = []
    for raw in str(args.eval_holes_list or args.eval_holes).split(","):
        raw = raw.strip()
        if raw and int(raw) not in eval_holes:
            eval_holes.append(int(raw))
    if args.eval_holes not in eval_holes:
        eval_holes.insert(0, args.eval_holes)

    eval_by_holes: Dict[str, Any] = {}
    for holes in eval_holes:
        eval_by_holes[f"holes{holes}"] = {
            "eval_clean": evaluate_clean(
                sudoku,
                model,
                eval_n=args.eval_n,
                holes=holes,
                pattern=args.hole_pattern,
                steps=args.train_loops,
                seed=args.seed + 4000 + holes,
                device=device,
            )
        }

    last_key = f"loop{args.train_loops}"
    primary = eval_by_holes[f"holes{args.eval_holes}"]["eval_clean"][last_key]
    loop1 = eval_by_holes[f"holes{args.eval_holes}"]["eval_clean"]["loop1"]
    loop_gain = float(primary["label_exact"]) - float(loop1["label_exact"])
    if train_metrics["train_ce_loss"] > 1.0:
        decision = "Stop: CE remains high, treat this as optimization/config failure before spending more GPU."
    elif loop_gain >= 0.03:
        decision = "Continue: EqR benefits from deeper recurrent refinement with FutureSeed/feature-diff enabled."
    elif float(primary["label_exact"]) > 0.05:
        decision = "Continue cautiously: nonzero exact suggests the mechanism is alive, next tune scale or curriculum."
    else:
        decision = "Pivot: exact remains near zero; inspect global consistency or reduce task difficulty."

    metrics = {
        "task": {
            "model": "EqRModel",
            "size": args.size,
            "hole_pattern": args.hole_pattern,
            "holes_min": args.holes_min,
            "holes_max": args.holes_max,
            "hole_stages": args.hole_stages,
            "train_loops": args.train_loops,
            "hidden_size": args.hidden_size,
            "heads": args.heads,
            "layers": args.layers,
            "h_cycles": args.h_cycles,
            "l_cycles": args.l_cycles,
            "params": param_count,
        },
        "train": train_metrics,
        "eval_by_holes": eval_by_holes,
        "eval_clean": eval_by_holes[f"holes{args.eval_holes}"]["eval_clean"],
        "decision": decision,
    }

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {"metrics": metrics, "config": vars(args)}
    json_path = out_dir / f"eqr_probe_seed{args.seed}.json"
    md_path = out_dir / f"eqr_probe_seed{args.seed}.md"
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_report(md_path, metrics)
    print(f"primary holes{args.eval_holes} {last_key} exact={primary['label_exact']:.4f} blank={primary['blank_acc']:.4f} gain={loop_gain:.4f}")
    print(f"decision={decision}")
    print(f"wrote {json_path}")
    print(f"wrote {md_path}")


if __name__ == "__main__":
    main()
