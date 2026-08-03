#!/usr/bin/env python3
"""GPU-only paired FutureSeed checkpoint evaluation and case visualization."""

from __future__ import annotations

import argparse
import gc
import hashlib
import html
import json
import os
import sys
from pathlib import Path
from typing import Any

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
RUNNER_DIR = REPO_ROOT / "experiments" / "rwkv_fs_sudoku"
sys.path.insert(0, str(RUNNER_DIR))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import study_rwkv_futureseed_loop as runner  # noqa: E402
from probe_gdn2_cell_order import build_model  # noqa: E402


RANGES = (("51-55", 51, 55), ("56-60", 56, 60), ("61-64", 61, 64))
LOOPS = (1, 2, 3, 4, 5)
MATCHED_FIELDS = (
    "size",
    "backbone",
    "d_model",
    "layers",
    "heads",
    "head_dim",
    "channel_mult",
    "l_cycles",
    "max_loops",
    "lambda_",
    "loop_update_mode",
    "loop_update_gate_init",
    "loop_loss",
    "batch",
    "grad_accum_steps",
    "seed",
    "gdn_mode",
    "gdn_expand_v",
    "gdn_use_short_conv",
    "gdn_conv_size",
    "official_sudoku_data_dir",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def wrong(case: dict[str, Any], loop: int) -> int:
    return int(case["loops"][f"loop{loop}"]["wrong_blank_count"])


def board_html(
    values: list[int] | None,
    *,
    labels: list[int],
    clue_mask: list[bool],
    title: str,
    puzzle: bool = False,
) -> str:
    cells = []
    for index in range(81):
        is_clue = bool(clue_mask[index])
        classes = ["cell"]
        if index % 3 == 0:
            classes.append("box-left")
        if index // 9 % 3 == 0:
            classes.append("box-top")
        if puzzle:
            classes.append("clue" if is_clue else "blank")
            text = str(int(labels[index]) + 1) if is_clue else ""
        else:
            assert values is not None
            correct = int(values[index]) == int(labels[index])
            classes.append("clue" if is_clue else ("correct" if correct else "wrong"))
            text = str(int(values[index]) + 1)
        cells.append(f'<div class="{" ".join(classes)}">{text}</div>')
    return (
        '<section class="board-wrap">'
        f"<h4>{html.escape(title)}</h4>"
        f'<div class="board">{"".join(cells)}</div>'
        "</section>"
    )


def summarize(cases: list[dict[str, Any]]) -> dict[str, Any]:
    total_blanks = sum(int(case["blank_count"]) for case in cases)
    loops: dict[str, Any] = {}
    for loop in LOOPS:
        exact = sum(bool(case["loops"][f"loop{loop}"]["label_exact"]) for case in cases)
        wrong_blanks = sum(wrong(case, loop) for case in cases)
        loops[f"loop{loop}"] = {
            "label_exact": exact / len(cases),
            "blank_acc": 1.0 - wrong_blanks / total_blanks,
            "wrong_blank_mean": wrong_blanks / len(cases),
        }
    return {
        "eval_n": len(cases),
        "loops": loops,
        "loop1_to_loop5_exact_gain": (
            loops["loop5"]["label_exact"] - loops["loop1"]["label_exact"]
        ),
        "loop1_to_loop5_wrong_blank_reduction": (
            loops["loop1"]["wrong_blank_mean"] - loops["loop5"]["wrong_blank_mean"]
        ),
    }


def paired_cases(
    nofs_payload: dict[str, Any], fs_payload: dict[str, Any]
) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    if nofs_payload["data_hash"] != fs_payload["data_hash"]:
        raise ValueError("Paired case-bank data hashes differ")
    nofs = {case["content_sha256"]: case for case in nofs_payload["cases"]}
    fs = {case["content_sha256"]: case for case in fs_payload["cases"]}
    if set(nofs) != set(fs):
        raise ValueError("Paired case identities differ")
    return [(nofs[key], fs[key]) for key in sorted(nofs)]


def select_cases(
    pairs: list[tuple[dict[str, Any], dict[str, Any]]]
) -> list[tuple[str, dict[str, Any], dict[str, Any]]]:
    largest_fs_gain = max(pairs, key=lambda pair: wrong(pair[0], 5) - wrong(pair[1], 5))
    strongest_fs_refinement = max(
        pairs,
        key=lambda pair: wrong(pair[1], 1) - wrong(pair[1], 5),
    )
    hardest_shared = max(
        pairs,
        key=lambda pair: wrong(pair[0], 5) + wrong(pair[1], 5),
    )
    rows = [
        ("Largest FutureSeed advantage", *largest_fs_gain),
        ("Strongest FutureSeed loop refinement", *strongest_fs_refinement),
        ("Hardest shared case", *hardest_shared),
    ]
    deduped = []
    seen = set()
    for row in rows:
        key = row[1]["content_sha256"]
        if key not in seen:
            seen.add(key)
            deduped.append(row)
    return deduped


def render_html(
    summary: dict[str, Any],
    selected: dict[str, list[tuple[str, dict[str, Any], dict[str, Any]]]],
) -> str:
    metric_rows = []
    for range_label, row in summary["ranges"].items():
        for loop in LOOPS:
            nofs = row["no_future_seed"]["loops"][f"loop{loop}"]
            fs = row["future_seed"]["loops"][f"loop{loop}"]
            metric_rows.append(
                "<tr>"
                f"<td>{range_label}</td><td>{loop}</td>"
                f"<td>{nofs['label_exact']:.4f}</td><td>{fs['label_exact']:.4f}</td>"
                f"<td>{fs['label_exact'] - nofs['label_exact']:+.4f}</td>"
                f"<td>{nofs['blank_acc']:.4f}</td><td>{fs['blank_acc']:.4f}</td>"
                f"<td>{nofs['wrong_blank_mean']:.2f}</td><td>{fs['wrong_blank_mean']:.2f}</td>"
                "</tr>"
            )
    sections = []
    for range_label, rows in selected.items():
        cards = []
        for title, nofs_case, fs_case in rows:
            labels = [int(value) for value in nofs_case["label_tokens"]]
            clue_mask = [bool(value) for value in nofs_case["clue_mask"]]
            boards = [
                board_html(None, labels=labels, clue_mask=clue_mask, title="Input", puzzle=True),
                board_html(labels, labels=labels, clue_mask=clue_mask, title="Target"),
            ]
            for arm_label, case in (("No FutureSeed", nofs_case), ("FutureSeed", fs_case)):
                for loop in (1, 3, 5):
                    boards.append(
                        board_html(
                            case["loops"][f"loop{loop}"]["prediction_tokens"],
                            labels=labels,
                            clue_mask=clue_mask,
                            title=f"{arm_label} loop{loop}: {wrong(case, loop)} wrong",
                        )
                    )
            cards.append(
                '<article class="case">'
                f"<h3>{html.escape(title)}</h3>"
                f'<div class="boards">{"".join(boards)}</div>'
                "</article>"
            )
        sections.append(f"<h2>{range_label} blanks</h2>{''.join(cards)}")
    decision = summary["decision"]
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>FutureSeed causal control at step {summary['checkpoint_step']}</title>
<style>
body{{font-family:Inter,system-ui,sans-serif;margin:0;color:#17202a;background:#f4f6f8;letter-spacing:0}}
main{{max-width:1540px;margin:auto;padding:24px}} h1,h2,h3,h4{{letter-spacing:0}}
.decision{{background:#fff;border-left:5px solid #16794b;padding:15px;margin:16px 0}}
table{{border-collapse:collapse;width:100%;background:#fff;margin:14px 0 28px;font-size:13px}}
th,td{{padding:8px;border:1px solid #d4d9df;text-align:right}} th:first-child,td:first-child{{text-align:left}}
.case{{background:#fff;border:1px solid #d4d9df;padding:13px;margin:14px 0}}
.boards{{display:grid;grid-template-columns:repeat(auto-fit,minmax(178px,1fr));gap:12px}}
.board-wrap h4{{font-size:12px;min-height:30px;margin:0 0 6px}}
.board{{display:grid;grid-template-columns:repeat(9,1fr);aspect-ratio:1;border:2px solid #1f2937}}
.cell{{display:grid;place-items:center;font:600 12px/1 monospace;border-left:1px solid #adb5bd;border-top:1px solid #adb5bd}}
.box-left{{border-left:2px solid #1f2937}} .box-top{{border-top:2px solid #1f2937}}
.clue{{background:#e5e7eb}} .blank{{background:#fff}} .correct{{background:#dff3e5}} .wrong{{background:#ffd8d3;color:#9b1c13}}
code{{background:#e8ebee;padding:2px 4px}}
</style></head><body><main>
<h1>Matched GDN2 causal control, step {summary['checkpoint_step']}</h1>
<p>Same official Sudoku data, architecture, optimizer-step budget, and loop supervision. The only mechanism difference is native FutureSeed state initialization.</p>
<div class="decision"><strong>{html.escape(decision['label'])}</strong><br>{html.escape(decision['reason'])}</div>
<h2>Paired aggregate</h2>
<table><thead><tr><th>Blank range</th><th>Loop</th><th>No-FS exact</th><th>FS exact</th><th>Exact delta</th><th>No-FS blank acc</th><th>FS blank acc</th><th>No-FS wrong</th><th>FS wrong</th></tr></thead>
<tbody>{''.join(metric_rows)}</tbody></table>
<p>Red cells are incorrect hidden cells. The displayed cases are diagnostics selected mechanically from a fixed paired pool; they do not affect aggregate scores.</p>
{''.join(sections)}
</main></body></html>"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-fs-checkpoint", required=True, type=Path)
    parser.add_argument("--no-fs-sha256", required=True)
    parser.add_argument("--fs-checkpoint", required=True, type=Path)
    parser.add_argument("--fs-sha256", required=True)
    parser.add_argument("--data-dir", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--eval-n", type=int, default=128)
    parser.add_argument("--cases-per-kind", type=int, default=2)
    parser.add_argument("--eval-seed", type=int, default=52)
    parser.add_argument("--expected-step", type=int, required=True)
    parser.add_argument("--forward-dtype", choices=("float32", "bfloat16"), default="bfloat16")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "0":
        raise RuntimeError("Set CUDA_VISIBLE_DEVICES=0; this comparison is GPU1-only")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is mandatory; CPU model fallback is forbidden")
    if torch.cuda.device_count() != 1:
        raise RuntimeError(f"Expected exactly one visible GPU, got {torch.cuda.device_count()}")

    specs = {
        "no_future_seed": (args.no_fs_checkpoint.resolve(), args.no_fs_sha256.lower()),
        "future_seed": (args.fs_checkpoint.resolve(), args.fs_sha256.lower()),
    }
    checkpoints: dict[str, dict[str, Any]] = {}
    for name, (path, expected_hash) in specs.items():
        actual_hash = sha256_file(path)
        if actual_hash != expected_hash:
            raise RuntimeError(f"{name} checkpoint hash mismatch: {actual_hash}")
        checkpoint = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
        if int(checkpoint.get("saved_at_step", -1)) != args.expected_step:
            raise RuntimeError(
                f"{name} is not an exact step{args.expected_step} checkpoint"
            )
        checkpoints[name] = checkpoint

    nofs_args = checkpoints["no_future_seed"]["args"]
    fs_args = checkpoints["future_seed"]["args"]
    mismatches = {
        field: (nofs_args.get(field), fs_args.get(field))
        for field in MATCHED_FIELDS
        if nofs_args.get(field) != fs_args.get(field)
    }
    if mismatches:
        raise RuntimeError(f"Matched checkpoint fields differ: {mismatches}")
    if float(nofs_args.get("future_seed_scale", -1.0)) != 0.0:
        raise RuntimeError("no-FS checkpoint has FutureSeed enabled")
    if float(fs_args.get("future_seed_scale", 0.0)) <= 0.0:
        raise RuntimeError("FutureSeed checkpoint has FutureSeed disabled")
    if nofs_args.get("backbone") != "gdn2":
        raise RuntimeError("This comparison is restricted to official FLA GDN2")

    runner.configure_sudoku(int(nofs_args["size"]), 0, 0)
    official_eval = runner.OfficialSudokuDataset(args.data_dir.resolve(), "test")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    arm_metadata: dict[str, Any] = {}
    arm_case_paths: dict[str, dict[str, Path]] = {}
    for name in ("no_future_seed", "future_seed"):
        checkpoint = checkpoints[name]
        saved_args = checkpoint["args"]
        device = torch.device("cuda", 0)
        model = build_model(saved_args).to(device)
        missing, unexpected = model.load_state_dict(checkpoint["model"], strict=False)
        if missing or unexpected:
            raise RuntimeError(f"{name} checkpoint mismatch: missing={missing}, unexpected={unexpected}")
        model.eval()
        runtime = runner.strict_fla_runtime_summary(model, "gdn2")
        arm_out = args.out_dir / "arms" / name
        artifacts = runner.export_case_bank(
            model,
            arm_out,
            holes_values=[],
            official_eval=official_eval,
            official_blank_ranges=[(label.replace("-", "_"), lo, hi) for label, lo, hi in RANGES],
            eval_n=args.eval_n,
            cases_per_kind=args.cases_per_kind,
            loop_values=list(LOOPS),
            seed=args.eval_seed,
            forward_dtype=args.forward_dtype,
        )
        arm_metadata[name] = {
            "checkpoint": str(specs[name][0]),
            "checkpoint_sha256": specs[name][1],
            "future_seed_scale": float(saved_args["future_seed_scale"]),
            "runtime": runtime,
            "case_bank": artifacts,
        }
        arm_case_paths[name] = {
            label: Path(artifacts["holes"][f"official_{label.replace('-', '_')}"]["all_cases_json"])
            for label, _lo, _hi in RANGES
        }
        del model
        del checkpoint
        checkpoints[name] = {}
        gc.collect()
        torch.cuda.empty_cache()

    range_summary: dict[str, Any] = {}
    selected: dict[str, Any] = {}
    for label, _lo, _hi in RANGES:
        nofs_payload = read_json(arm_case_paths["no_future_seed"][label])
        fs_payload = read_json(arm_case_paths["future_seed"][label])
        pairs = paired_cases(nofs_payload, fs_payload)
        nofs_cases = [pair[0] for pair in pairs]
        fs_cases = [pair[1] for pair in pairs]
        range_summary[label] = {
            "data_hash": nofs_payload["data_hash"],
            "no_future_seed": summarize(nofs_cases),
            "future_seed": summarize(fs_cases),
        }
        selected[label] = select_cases(pairs)

    hard_fs_exact = sum(
        row["future_seed"]["loops"]["loop5"]["label_exact"]
        for row in range_summary.values()
    ) / len(range_summary)
    hard_nofs_exact = sum(
        row["no_future_seed"]["loops"]["loop5"]["label_exact"]
        for row in range_summary.values()
    ) / len(range_summary)
    hard_fs_blank = sum(
        row["future_seed"]["loops"]["loop5"]["blank_acc"]
        for row in range_summary.values()
    ) / len(range_summary)
    hard_nofs_blank = sum(
        row["no_future_seed"]["loops"]["loop5"]["blank_acc"]
        for row in range_summary.values()
    ) / len(range_summary)
    supported = (
        hard_fs_exact > hard_nofs_exact
        and hard_fs_blank - hard_nofs_blank >= 0.10
    )
    summary = {
        "schema_version": "futureseed_paired_checkpoint_comparison.v1",
        "checkpoint_step": args.expected_step,
        "hypothesis": "Native FutureSeed changes the finite-compute optimization and information path, rather than merely changing the final readout operating point.",
        "matched_fields": {field: nofs_args.get(field) for field in MATCHED_FIELDS},
        "arms": arm_metadata,
        "ranges": range_summary,
        "aggregate": {
            "no_future_seed_loop5_exact": hard_nofs_exact,
            "future_seed_loop5_exact": hard_fs_exact,
            "exact_delta": hard_fs_exact - hard_nofs_exact,
            "no_future_seed_loop5_blank_acc": hard_nofs_blank,
            "future_seed_loop5_blank_acc": hard_fs_blank,
            "blank_acc_delta": hard_fs_blank - hard_nofs_blank,
        },
        "decision": {
            "supported": supported,
            "label": (
                f"FutureSeed separates from the matched causal control at step {args.expected_step}."
                if supported
                else f"This paired pool is inconclusive at step {args.expected_step}."
            ),
            "reason": (
                "The paired hard-case pool shows both an exact and a large blank-accuracy advantage without changing the data, backbone width/depth, optimizer-step budget, or loop supervision."
                if supported
                else "The paired pool does not simultaneously show a positive exact delta and at least +0.10 blank accuracy; use the preregistered aggregate gate for the scientific decision."
            ),
        },
    }
    (args.out_dir / "comparison.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (args.out_dir / "index.html").write_text(
        render_html(summary, selected),
        encoding="utf-8",
    )
    print(json.dumps(summary["aggregate"], sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
