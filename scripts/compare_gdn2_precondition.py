#!/usr/bin/env python3
from __future__ import annotations

import argparse
import html
import json
from pathlib import Path
from typing import Any

from compare_gdn2_address_payload import (
    LOOPS,
    RANGES,
    case_map,
    load_run,
    verify_matched,
    wrong,
)


def loop_metrics(summary: dict[str, Any], group: str, loop: str) -> dict[str, float]:
    metrics = summary["metrics"]["official_eval_by_blank_range"][group][
        "eval_clean"
    ][loop]
    return {
        "blank_acc": float(metrics["blank_acc"]),
        "exact": float(metrics["label_exact"]),
    }


def metric_rows(
    control: dict[str, Any], candidate: dict[str, Any]
) -> list[dict[str, Any]]:
    rows = []
    for group in RANGES:
        blank_range = control["metrics"]["official_eval_by_blank_range"][group][
            "blank_range"
        ]
        control_l1 = loop_metrics(control, group, "loop1")
        control_l5 = loop_metrics(control, group, "loop5")
        candidate_l1 = loop_metrics(candidate, group, "loop1")
        candidate_l5 = loop_metrics(candidate, group, "loop5")
        rows.append(
            {
                "group": group,
                "range": blank_range,
                "control_loop1": control_l1,
                "control_loop5": control_l5,
                "candidate_loop1": candidate_l1,
                "candidate_loop5": candidate_l5,
                "blank_delta": candidate_l5["blank_acc"]
                - control_l5["blank_acc"],
                "candidate_loop_gain": candidate_l5["blank_acc"]
                - candidate_l1["blank_acc"],
            }
        )
    return rows


def select_cases(
    control_bank: dict[str, Any], candidate_bank: dict[str, Any]
) -> list[tuple[str, dict[str, Any], dict[str, Any]]]:
    control = case_map(control_bank)
    candidate = case_map(candidate_bank)
    ids = list(control)
    selectors = (
        (
            "Largest final improvement",
            lambda index: wrong(control[index], "loop5")
            - wrong(candidate[index], "loop5"),
        ),
        (
            "Strongest candidate loop correction",
            lambda index: wrong(candidate[index], "loop1")
            - wrong(candidate[index], "loop5"),
        ),
        (
            "Hardest candidate failure",
            lambda index: wrong(candidate[index], "loop5"),
        ),
        (
            "Largest final regression",
            lambda index: wrong(candidate[index], "loop5")
            - wrong(control[index], "loop5"),
        ),
    )
    selected: list[tuple[str, dict[str, Any], dict[str, Any]]] = []
    seen: set[int] = set()
    for label, score in selectors:
        available = [index for index in ids if index not in seen]
        if not available:
            break
        index = max(available, key=score)
        seen.add(index)
        selected.append((label, control[index], candidate[index]))
    return selected


def board(
    *,
    title: str,
    prediction: list[int],
    target: list[int],
    clue_mask: list[bool],
    previous: list[int] | None,
    subtitle: str,
) -> str:
    cells = []
    for index, value in enumerate(prediction):
        row, col = divmod(index, 9)
        classes = ["cell"]
        if clue_mask[index]:
            classes.append("clue")
        elif value == target[index]:
            classes.append("correct")
        else:
            classes.append("wrong")
        if previous is not None and value != previous[index]:
            classes.append("changed")
        if col in (2, 5):
            classes.append("box-right")
        if row in (2, 5):
            classes.append("box-bottom")
        cells.append(f'<div class="{" ".join(classes)}">{value or "·"}</div>')
    return (
        '<article class="board-card">'
        f"<h4>{html.escape(title)}</h4><p>{html.escape(subtitle)}</p>"
        f'<div class="board">{"".join(cells)}</div></article>'
    )


def prediction_boards(case: dict[str, Any], arm: str) -> list[str]:
    rendered = []
    previous = None
    for loop in LOOPS:
        current = case["loops"][loop]
        rendered.append(
            board(
                title=f"{arm} {loop}",
                prediction=current["prediction"],
                target=case["label"],
                clue_mask=case["clue_mask"],
                previous=previous,
                subtitle=(
                    f"wrong {current['wrong_blank_count']}/{case['blank_count']}; "
                    f"blank acc {current['blank_acc']:.3f}"
                ),
            )
        )
        previous = current["prediction"]
    return rendered


def scalar_train_diagnostics(summary: dict[str, Any]) -> dict[str, float]:
    train = summary["metrics"]["train"]
    result = {}
    sources = (train, train.get("precondition", {}))
    for source in sources:
        if not isinstance(source, dict):
            continue
        for key, value in source.items():
            if key.startswith("gdn2_precondition_") and isinstance(
                value, (int, float)
            ):
                result[key] = float(value)
    return result


def render(
    *,
    control_run: Path,
    candidate_run: Path,
    control: dict[str, Any],
    candidate: dict[str, Any],
    rows: list[dict[str, Any]],
    selected: list[tuple[str, dict[str, Any], dict[str, Any]]],
) -> str:
    control_train = control["metrics"]["train"]
    candidate_train = candidate["metrics"]["train"]
    control_mean = sum(row["control_loop5"]["blank_acc"] for row in rows) / len(rows)
    candidate_mean = sum(row["candidate_loop5"]["blank_acc"] for row in rows) / len(rows)
    mean_delta = candidate_mean - control_mean
    time_overhead = float(candidate_train["train_sec"]) / float(
        control_train["train_sec"]
    ) - 1.0
    range_floor = min(row["blank_delta"] for row in rows)
    passes = mean_delta >= 0.03 and range_floor >= -0.03 and time_overhead <= 0.20
    verdict = "PASS" if passes else "STOP"
    verdict_text = (
        "The one-shot mechanism gate passed."
        if passes
        else "The preregistered gate did not pass; do not tune this mechanism."
    )

    table_rows = []
    for row in rows:
        low, high = row["range"]
        table_rows.append(
            "<tr>"
            f"<td>{low}-{high}</td>"
            f"<td>{row['control_loop1']['blank_acc']:.4f}</td>"
            f"<td>{row['control_loop5']['blank_acc']:.4f}</td>"
            f"<td>{row['candidate_loop1']['blank_acc']:.4f}</td>"
            f"<td>{row['candidate_loop5']['blank_acc']:.4f}</td>"
            f"<td>{row['candidate_loop_gain']:+.4f}</td>"
            f"<td>{row['blank_delta']:+.4f}</td>"
            f"<td>{row['candidate_loop5']['exact']:.4f}</td>"
            "</tr>"
        )

    diagnostic_rows = []
    for key, value in scalar_train_diagnostics(candidate).items():
        diagnostic_rows.append(
            f"<tr><td><code>{html.escape(key)}</code></td><td>{value:.6g}</td></tr>"
        )
    if not diagnostic_rows:
        diagnostic_rows.append(
            '<tr><td colspan="2">No scalar precondition diagnostics found.</td></tr>'
        )

    case_sections = []
    for label, control_case, candidate_case in selected:
        puzzle_mask = [value != 0 for value in control_case["puzzle"]]
        boards = [
            board(
                title="Puzzle",
                prediction=control_case["puzzle"],
                target=control_case["label"],
                clue_mask=puzzle_mask,
                previous=None,
                subtitle=f"{control_case['blank_count']} hidden cells",
            ),
            board(
                title="Target",
                prediction=control_case["label"],
                target=control_case["label"],
                clue_mask=control_case["clue_mask"],
                previous=None,
                subtitle="identical label for both arms",
            ),
            *prediction_boards(control_case, "Control"),
            *prediction_boards(candidate_case, "Preconditioned"),
        ]
        case_sections.append(
            f"<section><h2>{html.escape(label)}: batch {control_case['batch_index']}</h2>"
            "<p>Red is wrong, green is correct, gray is a clue. A blue underline "
            "means the value changed from the preceding loop.</p>"
            f'<div class="boards">{"".join(boards)}</div></section>'
        )

    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>FutureSeed-tied preconditioned GDN2</title>
<style>
*{{box-sizing:border-box}} body{{margin:0;background:#f5f7f8;color:#17202a;font:15px/1.45 system-ui,sans-serif}}
main{{max-width:1560px;margin:auto;padding:24px}} h1{{font-size:28px;margin:0 0 8px;letter-spacing:0}}
h2{{font-size:20px;margin:30px 0 10px;letter-spacing:0}} h4{{font-size:15px;margin:0;letter-spacing:0}}
.decision{{background:#fff;border-left:5px solid {('#087f5b' if passes else '#b42318')};padding:14px 18px}}
.metrics{{display:grid;grid-template-columns:repeat(5,minmax(150px,1fr));gap:10px}}
.metric{{background:#fff;border:1px solid #d8dde2;border-radius:6px;padding:12px}}
.metric b{{display:block;font-size:22px}} table{{width:100%;border-collapse:collapse;background:#fff}}
th,td{{padding:8px;border:1px solid #d8dde2;text-align:left}} th{{background:#edf1f3}}
.boards{{display:grid;grid-template-columns:repeat(4,minmax(220px,1fr));gap:12px}}
.board-card{{background:#fff;border:1px solid #d8dde2;border-radius:6px;padding:10px}}
.board-card p{{height:42px;margin:4px 0 8px}} .board{{display:grid;grid-template-columns:repeat(9,1fr);aspect-ratio:1}}
.cell{{display:grid;place-items:center;border:1px solid #aeb7c2;font-weight:650;font-size:15px;position:relative}}
.box-right{{border-right:2px solid #17202a}} .box-bottom{{border-bottom:2px solid #17202a}}
.clue{{background:#e7edf3}} .correct{{background:#d9f7e7}} .wrong{{background:#ffe0e0;color:#a61b1b}}
.changed::after{{content:"";position:absolute;left:22%;right:22%;bottom:3px;height:3px;background:#168aad}}
code{{background:#e9ecef;padding:2px 4px}} @media(max-width:1000px){{.boards{{grid-template-columns:repeat(2,1fr)}}.metrics{{grid-template-columns:repeat(2,1fr)}}}}
@media(max-width:620px){{main{{padding:12px}}.boards,.metrics{{grid-template-columns:1fr}}.cell{{font-size:13px}}}}
</style></head><body><main>
<h1>FutureSeed confidence + causal write preconditioning</h1>
<p class="decision"><b>{verdict}: {html.escape(verdict_text)}</b><br>
This is one fixed-budget candidate against the frozen position-Q/K control. It changes no parameter count, data, loss, loop budget, seed, or official GDN2 kernel.</p>
<div class="metrics">
<div class="metric"><span>Control mean hard blank acc</span><b>{control_mean:.4f}</b></div>
<div class="metric"><span>Candidate mean hard blank acc</span><b>{candidate_mean:.4f}</b></div>
<div class="metric"><span>Mean delta</span><b>{mean_delta:+.4f}</b></div>
<div class="metric"><span>Worst range delta</span><b>{range_floor:+.4f}</b></div>
<div class="metric"><span>Train-time overhead</span><b>{time_overhead:+.1%}</b></div>
</div>
<h2>Official test, 512 boards per blank range</h2>
<table><thead><tr><th>blanks</th><th>control L1 blank</th><th>control L5 blank</th>
<th>candidate L1 blank</th><th>candidate L5 blank</th><th>candidate L1→L5</th>
<th>candidate-control L5</th><th>candidate L5 exact</th></tr></thead><tbody>{''.join(table_rows)}</tbody></table>
<h2>Mechanism diagnostics</h2><table><thead><tr><th>quantity</th><th>value</th></tr></thead>
<tbody>{''.join(diagnostic_rows)}</tbody></table>
{''.join(case_sections)}
<p>Control: <code>{html.escape(control_run.name)}</code><br>Candidate: <code>{html.escape(candidate_run.name)}</code></p>
</main></body></html>"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--control-run", type=Path, required=True)
    parser.add_argument("--candidate-run", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()

    control, control_banks = load_run(args.control_run)
    candidate, candidate_banks = load_run(args.candidate_run)
    verify_matched(control_banks, candidate_banks)
    rows = metric_rows(control, candidate)
    selected = select_cases(control_banks["b61_64"], candidate_banks["b61_64"])
    control_mean = sum(row["control_loop5"]["blank_acc"] for row in rows) / len(rows)
    candidate_mean = sum(row["candidate_loop5"]["blank_acc"] for row in rows) / len(rows)
    payload = {
        "control_run": str(args.control_run),
        "candidate_run": str(args.candidate_run),
        "matched_case_banks": True,
        "rows": rows,
        "mean_control_blank_acc": control_mean,
        "mean_candidate_blank_acc": candidate_mean,
        "mean_delta_blank_acc": candidate_mean - control_mean,
        "train_time_overhead_frac": float(candidate["metrics"]["train"]["train_sec"])
        / float(control["metrics"]["train"]["train_sec"])
        - 1.0,
        "precondition_diagnostics": scalar_train_diagnostics(candidate),
        "selected_cases": [
            {
                "reason": label,
                "batch_index": control_case["batch_index"],
                "control_wrong": {loop: wrong(control_case, loop) for loop in LOOPS},
                "candidate_wrong": {
                    loop: wrong(candidate_case, loop) for loop in LOOPS
                },
            }
            for label, control_case, candidate_case in selected
        ],
        "data_hashes": {
            group: control_banks[group]["data_hash"] for group in RANGES
        },
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "comparison.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (args.out_dir / "index.html").write_text(
        render(
            control_run=args.control_run,
            candidate_run=args.candidate_run,
            control=control,
            candidate=candidate,
            rows=rows,
            selected=selected,
        ),
        encoding="utf-8",
    )
    print(args.out_dir / "index.html")


if __name__ == "__main__":
    main()
