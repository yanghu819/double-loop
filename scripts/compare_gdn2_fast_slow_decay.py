#!/usr/bin/env python3
"""Compare the single preregistered GDN2 Fast-Slow decay experiment."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import html
import json
import math
import shutil
from pathlib import Path
from typing import Any


RANGES = ("b51_55", "b56_60", "b61_64")
RANGE_LABELS = {
    "b51_55": "51-55 blanks",
    "b56_60": "56-60 blanks",
    "b61_64": "61-64 blanks",
}
CRITICAL_ARGS = (
    "backbone",
    "d_model",
    "layers",
    "heads",
    "head_dim",
    "max_loops",
    "loop_loss",
    "batch",
    "grad_accum_steps",
    "seed",
    "future_seed_scale",
    "future_seed_scope",
    "official_sudoku_data_dir",
    "resume_train_checkpoint",
)


def load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def loop_row(result: dict[str, Any], loop: int) -> dict[str, Any]:
    return result["metrics"]["eval_clean"][f"loop{loop}"]


def official_row(
    result: dict[str, Any], range_key: str, loop: int
) -> dict[str, Any]:
    return result["metrics"]["official_eval_by_blank_range"][range_key][
        "eval_clean"
    ][f"loop{loop}"]


def trace_row(result: dict[str, Any], loop: int) -> dict[str, Any]:
    return result["metrics"]["eval_clean"][f"loop{loop}/future_seed"]


def case_paths(result: dict[str, Any]) -> dict[str, Path]:
    paths: dict[str, Path] = {}
    holes = result["metrics"]["case_bank"]["holes"]
    for key in RANGES:
        group_key = f"official_{key}"
        paths[key] = Path(holes[group_key]["all_cases_json"]).resolve()
    return paths


def wrong(case: dict[str, Any], loop: int) -> int:
    return int(case["loops"][f"loop{loop}"]["wrong_total_count"])


def board_html(
    values: list[int],
    *,
    label: list[int],
    clue_mask: list[bool],
    title: str,
) -> str:
    cells = []
    for index, value in enumerate(values):
        is_clue = bool(clue_mask[index])
        correct = int(value) == int(label[index])
        classes = ["cell"]
        if index % 3 == 0:
            classes.append("box-left")
        if index // 9 % 3 == 0:
            classes.append("box-top")
        classes.append("clue" if is_clue else ("correct" if correct else "wrong"))
        cells.append(
            f'<div class="{" ".join(classes)}">{html.escape(str(int(value) + 1))}</div>'
        )
    return (
        '<section class="board-wrap">'
        f"<h4>{html.escape(title)}</h4>"
        f'<div class="board">{"".join(cells)}</div>'
        "</section>"
    )


def puzzle_html(case: dict[str, Any]) -> str:
    values = [
        int(value) if bool(case["clue_mask"][index]) else -1
        for index, value in enumerate(case["input"])
    ]
    cells = []
    for index, value in enumerate(values):
        classes = ["cell"]
        if index % 3 == 0:
            classes.append("box-left")
        if index // 9 % 3 == 0:
            classes.append("box-top")
        classes.append("clue" if value >= 0 else "blank")
        text = str(value + 1) if value >= 0 else ""
        cells.append(f'<div class="{" ".join(classes)}">{text}</div>')
    return (
        '<section class="board-wrap"><h4>Input</h4>'
        f'<div class="board">{"".join(cells)}</div></section>'
    )


def select_cases(
    control_payload: dict[str, Any],
    candidate_payload: dict[str, Any],
) -> list[tuple[str, dict[str, Any], dict[str, Any]]]:
    control_by_hash = {
        case["content_sha256"]: case for case in control_payload["cases"]
    }
    candidate_by_hash = {
        case["content_sha256"]: case for case in candidate_payload["cases"]
    }
    if set(control_by_hash) != set(candidate_by_hash):
        raise ValueError("paired case bank content hashes do not match")
    paired = [
        (control_by_hash[key], candidate_by_hash[key])
        for key in sorted(control_by_hash)
    ]
    improvement = max(
        paired,
        key=lambda pair: wrong(pair[0], 5) - wrong(pair[1], 5),
    )
    regression = max(
        paired,
        key=lambda pair: wrong(pair[1], 5) - wrong(pair[0], 5),
    )
    hardest = max(
        paired,
        key=lambda pair: wrong(pair[0], 5) + wrong(pair[1], 5),
    )
    return [
        ("Largest candidate improvement", *improvement),
        ("Largest candidate regression", *regression),
        ("Hardest shared case", *hardest),
    ]


def render_visualization(
    output: Path,
    summary: dict[str, Any],
    selected: dict[str, list[tuple[str, dict[str, Any], dict[str, Any]]]],
) -> None:
    metric_rows = []
    for key in RANGES:
        row = summary["official_ranges"][key]
        metric_rows.append(
            "<tr>"
            f"<td>{RANGE_LABELS[key]}</td>"
            f"<td>{row['control_loop5_exact']:.4f}</td>"
            f"<td>{row['candidate_loop5_exact']:.4f}</td>"
            f"<td class=\"{'positive' if row['delta_exact'] >= 0 else 'negative'}\">"
            f"{row['delta_exact']:+.4f}</td>"
            f"<td>{row['control_blank_acc']:.4f}</td>"
            f"<td>{row['candidate_blank_acc']:.4f}</td>"
            "</tr>"
        )
    loop_rows = []
    for loop in range(1, 6):
        row = summary["mixed_by_loop"][f"loop{loop}"]
        loop_rows.append(
            "<tr>"
            f"<td>{loop}</td><td>{row['control_exact']:.4f}</td>"
            f"<td>{row['candidate_exact']:.4f}</td>"
            f"<td>{row['delta_exact']:+.4f}</td>"
            f"<td>{row['control_blank_acc']:.4f}</td>"
            f"<td>{row['candidate_blank_acc']:.4f}</td>"
            "</tr>"
        )
    case_sections = []
    for range_key, cases in selected.items():
        cards = []
        for title, control_case, candidate_case in cases:
            boards = [puzzle_html(control_case)]
            boards.append(
                board_html(
                    control_case["label_tokens"],
                    label=control_case["label_tokens"],
                    clue_mask=control_case["clue_mask"],
                    title="Target",
                )
            )
            for arm, case in (("Control", control_case), ("Fast-Slow", candidate_case)):
                for loop in (1, 3, 5):
                    boards.append(
                        board_html(
                            case["loops"][f"loop{loop}"]["prediction_tokens"],
                            label=case["label_tokens"],
                            clue_mask=case["clue_mask"],
                            title=f"{arm} loop{loop}: {wrong(case, loop)} wrong",
                        )
                    )
            cards.append(
                '<article class="case">'
                f"<h3>{html.escape(title)}</h3>"
                f'<div class="boards">{"".join(boards)}</div>'
                "</article>"
            )
        case_sections.append(
            f"<h2>{RANGE_LABELS[range_key]}</h2>{''.join(cards)}"
        )
    mechanism = summary["mechanism"]
    decision = summary["decision"]
    document = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>GDN2 Fast-Slow Decay</title>
<style>
body{{font-family:Inter,system-ui,sans-serif;margin:0;color:#151515;background:#f5f6f7}}
main{{max-width:1480px;margin:auto;padding:28px}}
h1,h2,h3,h4{{letter-spacing:0}} .lede{{max-width:900px;color:#4b5563}}
.decision{{padding:16px;border-left:5px solid {'#16794b' if decision['accepted'] else '#b42318'};background:white}}
table{{border-collapse:collapse;width:100%;background:white;margin:14px 0 28px}}
th,td{{padding:9px 11px;border:1px solid #d8dde3;text-align:right}}
th:first-child,td:first-child{{text-align:left}} .positive{{color:#16794b}} .negative{{color:#b42318}}
.diagnostics{{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:8px;margin:14px 0 28px}}
.diagnostics div{{background:white;border:1px solid #d8dde3;padding:12px}}
.case{{background:white;border:1px solid #d8dde3;margin:16px 0;padding:14px}}
.boards{{display:grid;grid-template-columns:repeat(auto-fit,minmax(190px,1fr));gap:14px}}
.board-wrap h4{{font-size:13px;margin:0 0 7px;min-height:32px}}
.board{{display:grid;grid-template-columns:repeat(9,1fr);aspect-ratio:1;border:2px solid #111}}
.cell{{display:grid;place-items:center;font:600 12px/1 monospace;border-left:1px solid #b8bec5;border-top:1px solid #b8bec5}}
.box-left{{border-left:2px solid #111}} .box-top{{border-top:2px solid #111}}
.clue{{background:#e8eaed}} .blank{{background:white}} .correct{{background:#e4f3e9}} .wrong{{background:#ffd9d5;color:#9d1c13}}
code{{background:#eef0f2;padding:2px 4px}}
</style>
</head>
<body><main>
<h1>GDN2 + FutureSeed: Fast-Slow Decay Probe</h1>
<p class="lede">The candidate smooths only the forgetting hazard with a positive causal
FIR. Erase and write stay token-fast. The control traverses the same wrapper but
returns the raw hazard unchanged.</p>
<div class="decision"><strong>Decision: {html.escape(decision['label'])}</strong><br>
{html.escape(decision['reason'])}</div>
<h2>Official exact accuracy</h2>
<table><thead><tr><th>Blank range</th><th>Control</th><th>Fast-Slow</th><th>Delta</th><th>Control blank acc</th><th>Fast-Slow blank acc</th></tr></thead>
<tbody>{''.join(metric_rows)}</tbody></table>
<h2>Loop trajectory</h2>
<table><thead><tr><th>Loop</th><th>Control exact</th><th>Fast-Slow exact</th><th>Delta</th><th>Control blank acc</th><th>Fast-Slow blank acc</th></tr></thead>
<tbody>{''.join(loop_rows)}</tbody></table>
<h2>Mechanism diagnostics</h2>
<div class="diagnostics">
<div><strong>Hazard TV ratio</strong><br>{mechanism['tv_ratio']:.4f}</div>
<div><strong>Learned slow mix rho</strong><br>{mechanism['rho_mean']:.4f}</div>
<div><strong>Learned lag mass</strong><br>{mechanism['lag_mass']:.4f}</div>
<div><strong>Relative hazard change</strong><br>{mechanism['relative_change']:.4f}</div>
<div><strong>Time overhead</strong><br>{summary['systems']['time_overhead_frac']:+.2%}</div>
<div><strong>VRAM overhead</strong><br>{summary['systems']['memory_overhead_frac']:+.2%}</div>
</div>
<h2>Paired hard cases</h2>
<p>Gray cells are clues, green blank cells are correct, red cells are wrong.
Each row compares the same official puzzle across loop 1, 3, and 5.</p>
{''.join(case_sections)}
</main></body></html>"""
    output.write_text(document, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--control", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    control_path = args.control.resolve()
    candidate_path = args.candidate.resolve()
    control = load(control_path)
    candidate = load(candidate_path)

    mismatches = {
        key: (control["args"].get(key), candidate["args"].get(key))
        for key in CRITICAL_ARGS
        if control["args"].get(key) != candidate["args"].get(key)
    }
    if mismatches:
        raise ValueError(f"matched configuration mismatch: {mismatches}")
    if control["args"].get("gdn2_fast_slow_decay_mode") != "external_identity":
        raise ValueError("control is not external_identity")
    if candidate["args"].get("gdn2_fast_slow_decay_mode") != "positive_causal":
        raise ValueError("candidate is not positive_causal")

    official_ranges: dict[str, Any] = {}
    official_deltas = []
    for key in RANGES:
        control_final = official_row(control, key, 5)
        candidate_final = official_row(candidate, key, 5)
        delta = float(candidate_final["label_exact"]) - float(
            control_final["label_exact"]
        )
        official_deltas.append(delta)
        official_ranges[key] = {
            "control_loop5_exact": float(control_final["label_exact"]),
            "candidate_loop5_exact": float(candidate_final["label_exact"]),
            "delta_exact": delta,
            "control_blank_acc": float(control_final["blank_acc"]),
            "candidate_blank_acc": float(candidate_final["blank_acc"]),
            "control_loop_gain": float(control_final["label_exact"])
            - float(official_row(control, key, 1)["label_exact"]),
            "candidate_loop_gain": float(candidate_final["label_exact"])
            - float(official_row(candidate, key, 1)["label_exact"]),
        }

    mixed_by_loop: dict[str, Any] = {}
    for loop in range(1, 6):
        control_loop = loop_row(control, loop)
        candidate_loop = loop_row(candidate, loop)
        mixed_by_loop[f"loop{loop}"] = {
            "control_exact": float(control_loop["label_exact"]),
            "candidate_exact": float(candidate_loop["label_exact"]),
            "delta_exact": float(candidate_loop["label_exact"])
            - float(control_loop["label_exact"]),
            "control_blank_acc": float(control_loop["blank_acc"]),
            "candidate_blank_acc": float(candidate_loop["blank_acc"]),
        }

    control_train = control["metrics"]["train"]
    candidate_train = candidate["metrics"]["train"]
    candidate_diag = candidate_train["fast_slow_decay"]
    time_overhead = (
        float(candidate_train["train_sec"]) / float(control_train["train_sec"]) - 1.0
    )
    memory_overhead = (
        float(candidate_train["cuda_max_memory_allocated_mb"])
        / float(control_train["cuda_max_memory_allocated_mb"])
        - 1.0
    )
    hard_mean_delta = sum(official_deltas) / len(official_deltas)
    mixed_delta = mixed_by_loop["loop5"]["delta_exact"]
    mechanism_active = (
        float(candidate_diag["gdn2_fast_slow_enabled"]) == 1.0
        and float(candidate_diag["gdn2_fast_slow_tv_ratio"]) <= 0.995
        and float(candidate_diag["gdn2_fast_slow_relative_change"]) > 0.0
    )
    systems_pass = time_overhead <= 0.20 and memory_overhead <= 0.20
    score_pass = (
        (
            hard_mean_delta >= 0.01
            and min(official_deltas) >= -0.01
            and mixed_delta >= 0.0
        )
        or (
            max(official_deltas[1:]) >= 0.02
            and official_deltas[0] >= -0.01
            and mixed_delta >= 0.0
        )
    )
    accepted = mechanism_active and systems_pass and score_pass
    if accepted:
        label = "accept and continue unchanged to step9300"
        reason = (
            "The slow forgetting path reduced temporal hazard variation and "
            "improved hard exact accuracy without a systems or easy-range regression."
        )
    elif not systems_pass:
        label = "discard at systems gate"
        reason = "The mechanism exceeded the fixed 20% time or memory overhead budget."
    elif not mechanism_active:
        label = "discard: mechanism did not become active"
        reason = "The learned filter failed to materially smooth the forgetting hazard."
    else:
        label = "discard: no hard Sudoku gain"
        reason = (
            "Smoother forgetting did not improve the preregistered hard exact metrics; "
            "do not sweep kernel size, rho, seed, LR, or loss."
        )

    summary = {
        "schema_version": "gdn2_fast_slow_decay_comparison.v1",
        "timestamp_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "control": {
            "path": str(control_path),
            "sha256": sha256(control_path),
            "mode": "external_identity",
        },
        "candidate": {
            "path": str(candidate_path),
            "sha256": sha256(candidate_path),
            "mode": "positive_causal",
        },
        "official_ranges": official_ranges,
        "official_hard_mean_delta": hard_mean_delta,
        "mixed_by_loop": mixed_by_loop,
        "mixed_loop5_delta": mixed_delta,
        "mechanism": {
            "active": mechanism_active,
            "tv_ratio": float(candidate_diag["gdn2_fast_slow_tv_ratio"]),
            "rho_mean": float(candidate_diag["gdn2_fast_slow_rho_mean"]),
            "lag_mass": float(candidate_diag["gdn2_fast_slow_lag_mass"]),
            "relative_change": float(
                candidate_diag["gdn2_fast_slow_relative_change"]
            ),
        },
        "systems": {
            "control_train_sec": float(control_train["train_sec"]),
            "candidate_train_sec": float(candidate_train["train_sec"]),
            "time_overhead_frac": time_overhead,
            "control_peak_mb": float(control_train["cuda_max_memory_allocated_mb"]),
            "candidate_peak_mb": float(
                candidate_train["cuda_max_memory_allocated_mb"]
            ),
            "memory_overhead_frac": memory_overhead,
        },
        "decision": {
            "accepted": accepted,
            "score_pass": score_pass,
            "systems_pass": systems_pass,
            "mechanism_active": mechanism_active,
            "label": label,
            "reason": reason,
        },
    }

    out_dir = args.out_dir.resolve()
    visualization_dir = out_dir / "visualizations"
    evidence_dir = out_dir / "evidence"
    visualization_dir.mkdir(parents=True, exist_ok=True)
    evidence_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(control_path, evidence_dir / "control.json")
    shutil.copy2(candidate_path, evidence_dir / "candidate.json")

    selected: dict[str, Any] = {}
    control_cases = case_paths(control)
    candidate_cases = case_paths(candidate)
    for key in RANGES:
        control_payload = load(control_cases[key])
        candidate_payload = load(candidate_cases[key])
        if control_payload["data_hash"] != candidate_payload["data_hash"]:
            raise ValueError(f"{key}: paired data hashes do not match")
        selected[key] = select_cases(control_payload, candidate_payload)
        shutil.copy2(control_cases[key], evidence_dir / f"{key}-control-cases.json")
        shutil.copy2(
            candidate_cases[key], evidence_dir / f"{key}-candidate-cases.json"
        )

    (out_dir / "comparison.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (out_dir / "score.json").write_text(
        json.dumps(
            {
                "primary_score": sum(
                    row["candidate_loop5_exact"]
                    for row in official_ranges.values()
                )
                / len(RANGES),
                "decision": summary["decision"],
                "official_ranges": official_ranges,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    (out_dir / "config.json").write_text(
        json.dumps(
            {
                "critical_args": {
                    key: control["args"].get(key) for key in CRITICAL_ARGS
                },
                "candidate_intervention": {
                    "mode": "positive_causal",
                    "kernel_size": candidate["args"][
                        "gdn2_fast_slow_decay_kernel_size"
                    ],
                    "rho_init": candidate["args"][
                        "gdn2_fast_slow_decay_rho_init"
                    ],
                    "current_weight_init": candidate["args"][
                        "gdn2_fast_slow_decay_current_weight_init"
                    ],
                },
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    (out_dir / "metadata.json").write_text(
        json.dumps(
            {
                "timestamp_utc": summary["timestamp_utc"],
                "source": "matched GPU1 official FLA GDN2 FutureSeed continuation",
                "control_sha256": summary["control"]["sha256"],
                "candidate_sha256": summary["candidate"]["sha256"],
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    render_visualization(visualization_dir / "index.html", summary, selected)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
