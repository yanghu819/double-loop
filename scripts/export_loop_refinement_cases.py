#!/usr/bin/env python3
"""Export concrete 9x9 loop-refinement cases from a reproduced FutureSeed run.

This script is intentionally CUDA-only. It retrains the same FutureSeed RWKV
backbone configuration used in the 2026-06-04 GPU1 cliff run, then searches
harder eval batches for boards where recurrent loops visibly refine the answer.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import random
import sys
from datetime import datetime, timezone
from pathlib import Path
from types import ModuleType
from typing import Any

import torch
from PIL import Image, ImageDraw


REPO_ROOT = Path(__file__).resolve().parents[1]
SUDOKU_PATH = REPO_ROOT / "experiments/rwkv_fs_sudoku/study_rwkv_futureseed_loop.py"
VIZ_PATH = REPO_ROOT / "scripts/visualize_sudoku_case.py"


def import_from_path(name: str, path: Path) -> ModuleType:
    parent = str(path.parent)
    if parent not in sys.path:
        sys.path.insert(0, parent)
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot import {name} from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


sudoku = import_from_path("double_loop_sudoku", SUDOKU_PATH)
viz = import_from_path("double_loop_case_viz", VIZ_PATH)


def digit(value: int, *, blank: int) -> str:
    return "." if int(value) == blank else str(int(value) + 1)


def tensor_board(x: torch.Tensor, *, blank: int) -> list[str]:
    return [digit(int(v), blank=blank) for v in x.detach().cpu().reshape(-1).tolist()]


def loop_key(idx: int) -> str:
    return f"loop{idx}"


def case_kind(stats: dict[str, Any]) -> str:
    wrong1 = len(stats["loop1"].wrong_blanks)
    wrong3 = len(stats["loop3"].wrong_blanks)
    wrong5 = len(stats["loop5"].wrong_blanks)
    if wrong1 > 0 and wrong5 == 0:
        return "refined"
    if wrong5 < wrong1 or wrong3 < wrong1:
        return "improved"
    if wrong5 > 0:
        return "hard_failure"
    return "already_solved"


def score_case(kind: str, stats: dict[str, Any]) -> tuple[int, int, int]:
    wrong1 = len(stats["loop1"].wrong_blanks)
    wrong3 = len(stats["loop3"].wrong_blanks)
    wrong5 = len(stats["loop5"].wrong_blanks)
    priority = {"refined": 0, "improved": 1, "hard_failure": 2, "already_solved": 3}[kind]
    return (priority, wrong5, -max(wrong1 - wrong5, wrong1 - wrong3))


def extract_cases(
    model: torch.nn.Module,
    *,
    holes: int,
    eval_n: int,
    seed: int,
    device: torch.device,
    max_loops: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    batch = sudoku.make_batch(eval_n, holes, holes, "random", random.Random(seed), device=device)
    clean, loop_preds = sudoku.evaluate_model(
        model,
        batch,
        max_loops=max_loops,
        noise_scale=0.0,
        seed=seed + 4000,
    )
    inputs, labels, clue_mask = batch
    candidates: list[dict[str, Any]] = []
    for idx in range(inputs.shape[0]):
        puzzle = tensor_board(inputs[idx], blank=sudoku.BLANK)
        solution = tensor_board(labels[idx], blank=sudoku.BLANK)
        loops = {
            loop_key(loop_idx): tensor_board(loop_preds[loop_idx - 1][idx], blank=sudoku.BLANK)
            for loop_idx in range(1, max_loops + 1)
        }
        selected = {name: loops[name] for name in ["loop1", "loop3", "loop5"]}
        stats = {name: viz.analyze(board, solution, puzzle) for name, board in selected.items()}
        kind = case_kind(stats)
        if kind == "already_solved":
            continue
        candidates.append(
            {
                "holes": holes,
                "batch_index": idx,
                "kind": kind,
                "puzzle": viz.board_rows(puzzle),
                "solution": viz.board_rows(solution),
                "loops": {
                    name: {
                        "board": viz.board_rows(board),
                        "wrong_blanks": [viz.coord(cell) for cell in stats[name].wrong_blanks],
                        "conflict_units": stats[name].conflict_units,
                    }
                    for name, board in selected.items()
                },
                "_flat": {
                    "puzzle": puzzle,
                    "solution": solution,
                    **selected,
                },
                "_score": score_case(kind, stats),
            }
        )
    candidates.sort(key=lambda row: row["_score"])
    return clean, candidates


def build_case_html(case: dict[str, Any]) -> str:
    flat = case["_flat"]
    puzzle = flat["puzzle"]
    solution = flat["solution"]
    stats = {
        name: viz.analyze(flat[name], solution, puzzle)
        for name in ["loop1", "loop3", "loop5"]
    }
    panels = [
        viz.panel("Puzzle", f"{case['holes']} hidden cells", puzzle, solution, puzzle, stats=None, mode="puzzle"),
        viz.panel("Solution", "sampled target board", solution, solution, puzzle, stats=None, mode="solution"),
    ]
    for name in ["loop1", "loop3", "loop5"]:
        panels.append(
            viz.panel(
                f"FutureSeed {name}",
                (
                    f"wrong={len(stats[name].wrong_blanks)}/{case['holes']}; "
                    f"conflict units={len(stats[name].conflict_units)}"
                ),
                flat[name],
                solution,
                puzzle,
                stats=stats[name],
                previous=flat["loop1"] if name == "loop3" else flat["loop3"] if name == "loop5" else None,
            )
        )
    wrong_lines = []
    for name in ["loop1", "loop3", "loop5"]:
        wrong = case["loops"][name]["wrong_blanks"]
        wrong_lines.append(f"<li><code>{name}</code>: {html_list(wrong) if wrong else 'none'}</li>")
    return f"""<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>9x9 loop-refinement case h{case['holes']} #{case['batch_index']}</title>
<style>
body {{ margin: 24px; background: #f3f4f6; color: #111827; font-family: ui-sans-serif, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }}
h1 {{ margin: 0 0 8px; font-size: 26px; letter-spacing: 0; }}
h2 {{ margin: 22px 0 10px; font-size: 18px; letter-spacing: 0; }}
h3 {{ margin: 0 0 4px; font-size: 15px; letter-spacing: 0; }}
p {{ margin: 0; color: #4b5563; font-size: 13px; line-height: 1.35; }}
code {{ font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; }}
.row {{ display: flex; flex-wrap: wrap; gap: 14px; align-items: flex-start; }}
.panel {{ background: #fff; border: 1px solid #d1d5db; border-radius: 8px; padding: 12px; }}
.panel p {{ min-height: 36px; max-width: 330px; }}
.grid {{ display: grid; grid-template-columns: repeat(9, 34px); grid-template-rows: repeat(9, 34px); width: max-content; border: 3px solid #111827; margin-top: 10px; }}
.cell {{ width: 34px; height: 34px; box-sizing: border-box; display: flex; align-items: center; justify-content: center; border-right: 1px solid #9ca3af; border-bottom: 1px solid #9ca3af; font-size: 17px; font-weight: 800; position: relative; }}
.clue {{ background: #d1d5db; color: #111827; }}
.hole {{ background: #fff7ed; color: #9a3412; }}
.solution {{ background: #e0f2fe; color: #0c4a6e; }}
.given-solution {{ box-shadow: inset 0 0 0 2px rgba(17,24,39,0.14); }}
.correct {{ background: #bbf7d0; color: #14532d; }}
.wrong {{ background: #fecaca; color: #7f1d1d; }}
.conflict::after {{ content: ""; position: absolute; inset: 3px; border: 3px solid #f59e0b; border-radius: 6px; pointer-events: none; }}
.changed::before {{ content: ""; position: absolute; left: 8px; right: 8px; bottom: 4px; height: 3px; background: #2563eb; border-radius: 4px; }}
.notes {{ max-width: 1130px; background: #fff; border: 1px solid #d1d5db; border-radius: 8px; padding: 12px 14px; margin-top: 14px; }}
.notes ul {{ margin: 8px 0 0 18px; padding: 0; }}
.notes li {{ margin: 4px 0; font-size: 13px; }}
</style>
</head>
<body>
<h1>9x9 loop-refinement case: h{case['holes']}, {case['kind']}</h1>
<p>Batch index {case['batch_index']}. Blue underline marks cells changed since the previous shown loop; orange outline marks Sudoku duplicate conflicts.</p>
<div class="row" style="margin-top: 14px;">{''.join(panels)}</div>
<div class="notes">
  <p>Wrong hidden-cell coordinates by loop:</p>
  <ul>{''.join(wrong_lines)}</ul>
</div>
</body>
</html>
"""


def html_list(items: list[str]) -> str:
    return ", ".join(f"<code>{item}</code>" for item in items)


def draw_case_png(path: Path, case: dict[str, Any]) -> None:
    flat = case["_flat"]
    puzzle = flat["puzzle"]
    solution = flat["solution"]
    stats = {
        name: viz.analyze(flat[name], solution, puzzle)
        for name in ["loop1", "loop3", "loop5"]
    }
    image = Image.new("RGB", (1780, 520), "#f3f4f6")
    draw = ImageDraw.Draw(image)
    draw.text(
        (32, 26),
        f"9x9 loop-refinement case: h{case['holes']} / {case['kind']} / batch {case['batch_index']}",
        fill="#111827",
        font=viz.load_font(30, bold=True),
    )
    draw.text(
        (32, 66),
        "FutureSeed loop trajectory on one concrete puzzle. Green is correct fill, red is wrong fill, orange is duplicate conflict.",
        fill="#4b5563",
        font=viz.load_font(16),
    )
    x_positions = [32, 380, 728, 1076, 1424]
    y = 114
    viz.draw_board(draw, (x_positions[0], y), "Puzzle", f"{case['holes']} hidden cells", puzzle, solution, puzzle, None, mode="puzzle")
    viz.draw_board(draw, (x_positions[1], y), "Solution", "ground truth", solution, solution, puzzle, None, mode="solution")
    for pos, name in zip(x_positions[2:], ["loop1", "loop3", "loop5"]):
        viz.draw_board(
            draw,
            (pos, y),
            f"FutureSeed {name}",
            f"wrong {len(stats[name].wrong_blanks)}/{case['holes']}",
            flat[name],
            solution,
            puzzle,
            stats[name],
        )
    image.save(path)


def write_outputs(out_dir: Path, holes: int, cases: list[dict[str, Any]]) -> list[dict[str, str]]:
    written = []
    for rank, case in enumerate(cases, start=1):
        stem = f"h{holes}_{rank:02d}_{case['kind']}_idx{case['batch_index']}"
        html_path = out_dir / f"{stem}.html"
        png_path = out_dir / f"{stem}.png"
        html_path.write_text(build_case_html(case), encoding="utf-8")
        draw_case_png(png_path, case)
        written.append({"html": str(html_path), "png": str(png_path), "stem": stem})
    return written


def strip_private(case: dict[str, Any]) -> dict[str, Any]:
    public = {key: value for key, value in case.items() if not key.startswith("_")}
    return public


def parse_int_list(raw: str) -> list[int]:
    return [int(item.strip()) for item in raw.split(",") if item.strip()]


def jsonable_config(args: argparse.Namespace) -> dict[str, Any]:
    out = {}
    for key, value in vars(args).items():
        out[key] = str(value) if isinstance(value, Path) else value
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=REPO_ROOT / "runs/rwkv-loop-refinement-cases-20260604")
    parser.add_argument("--seed", type=int, default=52)
    parser.add_argument("--eval-n", type=int, default=512)
    parser.add_argument("--eval-holes-list", default="24,28,32")
    parser.add_argument("--max-cases-per-holes", type=int, default=3)
    parser.add_argument("--steps", type=int, default=600)
    parser.add_argument("--batch", type=int, default=192)
    parser.add_argument("--hole-stages", default="8-16:200,16-24:400")
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--layers", type=int, default=8)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--head-dim", type=int, default=16)
    parser.add_argument("--channel-mult", type=int, default=4)
    parser.add_argument("--l-cycles", type=int, default=2)
    parser.add_argument("--max-loops", type=int, default=5)
    parser.add_argument("--rwkv-kernel", default="cuda", choices=("cuda", "statepassing", "torch", "auto", "wind"))
    parser.add_argument("--log-every", type=int, default=100)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required; refusing CPU case export.")
    if args.d_model != args.heads * args.head_dim:
        raise ValueError("--d-model must equal --heads * --head-dim")
    sudoku.configure_sudoku(9, 0, 0)
    device = torch.device("cuda")
    train_args = argparse.Namespace(
        size=9,
        box_rows=0,
        box_cols=0,
        steps=args.steps,
        batch=args.batch,
        d_model=args.d_model,
        layers=args.layers,
        heads=args.heads,
        head_dim=args.head_dim,
        channel_mult=args.channel_mult,
        l_cycles=args.l_cycles,
        max_loops=args.max_loops,
        lambda_=0.95,
        future_seed_scale=1.0,
        rwkv_kernel=args.rwkv_kernel,
        loop_loss="final",
        noise_scale=0.0,
        feature_buffer_size=8192,
        feature_buffer_add=2048,
        rollout_ks="",
        rollout_loop_values="",
        rollout_noise_scale=0.0,
        lr=2e-3,
        weight_decay=1e-3,
        holes_min=8,
        holes_max=24,
        hole_stages=args.hole_stages,
        hole_pattern="random",
        eval_hole_patterns="",
        eval_holes=24,
        eval_holes_list=args.eval_holes_list,
        eval_n=args.eval_n,
        blank_loss_weight=8.0,
        case_index=0,
        seed=args.seed,
        log_every=args.log_every,
        out_dir=str(args.out_dir),
        cpu=False,
    )
    ok, reason = sudoku.statepassing_available(args.head_dim)
    if args.rwkv_kernel in {"cuda", "statepassing"} and not ok:
        raise RuntimeError(f"RWKV CUDA statepassing unavailable: {reason}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    model, train_stats = sudoku.train_model(train_args, device=device)
    holes_values = parse_int_list(args.eval_holes_list)
    all_cases: dict[str, list[dict[str, Any]]] = {}
    metrics_by_holes: dict[str, Any] = {}
    artifacts: dict[str, list[dict[str, str]]] = {}
    for holes in holes_values:
        clean, candidates = extract_cases(
            model,
            holes=holes,
            eval_n=args.eval_n,
            seed=args.seed + 999 + holes * 17,
            device=device,
            max_loops=args.max_loops,
        )
        selected = candidates[: args.max_cases_per_holes]
        all_cases[f"h{holes}"] = [strip_private(case) for case in selected]
        metrics_by_holes[f"h{holes}"] = {
            loop: clean[loop]
            for loop in ["loop1", "loop3", "loop5"]
        }
        artifacts[f"h{holes}"] = write_outputs(args.out_dir, holes, selected)

    summary = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "purpose": "Concrete loop-refinement cases: loop1 wrong to later-loop improved/exact.",
        "config": jsonable_config(args),
        "train": train_stats,
        "metrics_by_holes": metrics_by_holes,
        "cases": all_cases,
        "artifacts": artifacts,
    }
    summary_path = args.out_dir / "loop_refinement_cases.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    readme_path = args.out_dir / "README.md"
    lines = [
        "# 9x9 FutureSeed Loop-Refinement Cases",
        "",
        "Purpose: show concrete boards where recurrent loop depth refines a FutureSeed RWKV state.",
        "",
        f"Seed: `{args.seed}`; eval_n: `{args.eval_n}`; holes: `{args.eval_holes_list}`.",
        "",
        "## Selected Cases",
        "",
    ]
    for hole_key, rows in artifacts.items():
        lines.append(f"### {hole_key}")
        for item, case in zip(rows, all_cases[hole_key]):
            wrong1 = len(case["loops"]["loop1"]["wrong_blanks"])
            wrong3 = len(case["loops"]["loop3"]["wrong_blanks"])
            wrong5 = len(case["loops"]["loop5"]["wrong_blanks"])
            lines.append(
                f"- `{item['stem']}`: kind={case['kind']}, wrong loop1/3/5={wrong1}/{wrong3}/{wrong5}; "
                f"png=`{Path(item['png']).name}`, html=`{Path(item['html']).name}`"
            )
        lines.append("")
    lines.append("## Metrics")
    lines.append("")
    for hole_key, metrics in metrics_by_holes.items():
        vals = []
        for loop in ["loop1", "loop3", "loop5"]:
            vals.append(f"{loop} exact={metrics[loop]['label_exact']:.4f}")
        lines.append(f"- {hole_key}: " + ", ".join(vals))
    readme_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {summary_path}")
    print(f"wrote {readme_path}")


if __name__ == "__main__":
    main()
