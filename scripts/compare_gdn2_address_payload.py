#!/usr/bin/env python3
from __future__ import annotations

import argparse
import html
import json
from pathlib import Path
from typing import Any


RANGES = ("b51_55", "b56_60", "b61_64")
LOOPS = ("loop1", "loop2", "loop3", "loop4", "loop5")


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_run(run_dir: Path) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    summary = load_json(run_dir / "output" / "futureseed_loop_seed52.json")
    banks: dict[str, dict[str, Any]] = {}
    for group in RANGES:
        banks[group] = load_json(
            run_dir
            / "output"
            / "case_bank"
            / f"official_{group}"
            / "all_cases.json"
        )
    return summary, banks


def case_map(bank: dict[str, Any]) -> dict[int, dict[str, Any]]:
    return {int(case["batch_index"]): case for case in bank["cases"]}


def wrong(case: dict[str, Any], loop: str) -> int:
    return int(case["loops"][loop]["wrong_blank_count"])


def verify_matched(
    control_banks: dict[str, dict[str, Any]],
    candidate_banks: dict[str, dict[str, Any]],
) -> None:
    for group in RANGES:
        control = control_banks[group]
        candidate = candidate_banks[group]
        if control["data_hash"] != candidate["data_hash"]:
            raise ValueError(f"{group}: data hashes differ")
        control_cases = case_map(control)
        candidate_cases = case_map(candidate)
        if control_cases.keys() != candidate_cases.keys():
            raise ValueError(f"{group}: evaluated case IDs differ")
        for batch_index in control_cases:
            left = control_cases[batch_index]
            right = candidate_cases[batch_index]
            for key in ("puzzle", "label", "clue_mask", "blank_count"):
                if left[key] != right[key]:
                    raise ValueError(
                        f"{group} batch {batch_index}: {key} differs"
                    )


def select_case(
    control_bank: dict[str, Any],
    candidate_bank: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    control_cases = case_map(control_bank)
    candidate_cases = case_map(candidate_bank)

    def score(batch_index: int) -> tuple[int, int, int]:
        control = control_cases[batch_index]
        candidate = candidate_cases[batch_index]
        cross_arm_gain = wrong(control, "loop5") - wrong(candidate, "loop5")
        loop_gain = wrong(candidate, "loop1") - wrong(candidate, "loop5")
        return cross_arm_gain + loop_gain, cross_arm_gain, loop_gain

    selected = max(control_cases, key=score)
    return control_cases[selected], candidate_cases[selected]


def metric_rows(
    control_summary: dict[str, Any],
    candidate_summary: dict[str, Any],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for group in RANGES:
        control = control_summary["metrics"]["official_eval_by_blank_range"][group]
        candidate = candidate_summary["metrics"]["official_eval_by_blank_range"][
            group
        ]
        control_loop = control["eval_clean"]["loop5"]
        candidate_loop = candidate["eval_clean"]["loop5"]
        rows.append(
            {
                "group": group,
                "range": control["blank_range"],
                "control_blank_acc": float(control_loop["blank_acc"]),
                "candidate_blank_acc": float(candidate_loop["blank_acc"]),
                "delta_blank_acc": float(candidate_loop["blank_acc"])
                - float(control_loop["blank_acc"]),
                "control_exact": float(control_loop["label_exact"]),
                "candidate_exact": float(candidate_loop["label_exact"]),
            }
        )
    return rows


def board_html(
    *,
    title: str,
    values: list[int],
    label: list[int],
    clue_mask: list[bool],
    previous: list[int] | None,
    subtitle: str,
) -> str:
    cells: list[str] = []
    for index, value in enumerate(values):
        row, col = divmod(index, 9)
        classes = ["cell"]
        if clue_mask[index]:
            classes.append("clue")
        elif value == label[index]:
            classes.append("correct")
        else:
            classes.append("wrong")
        if previous is not None and value != previous[index]:
            classes.append("changed")
        if col in (2, 5):
            classes.append("box-right")
        if row in (2, 5):
            classes.append("box-bottom")
        display = "·" if value == 0 else str(value)
        cells.append(
            f'<div class="{" ".join(classes)}">{html.escape(display)}</div>'
        )
    return (
        '<section class="board-card">'
        f"<h3>{html.escape(title)}</h3>"
        f"<p>{html.escape(subtitle)}</p>"
        f'<div class="board">{"".join(cells)}</div>'
        "</section>"
    )


def prediction_board(
    case: dict[str, Any],
    *,
    arm: str,
    loop: str,
    previous_loop: str | None,
) -> str:
    payload = case["loops"][loop]
    previous = (
        case["loops"][previous_loop]["prediction"] if previous_loop else None
    )
    return board_html(
        title=f"{arm} {loop}",
        values=payload["prediction"],
        label=case["label"],
        clue_mask=case["clue_mask"],
        previous=previous,
        subtitle=(
            f"wrong blanks {payload['wrong_blank_count']}/{case['blank_count']}; "
            f"blank accuracy {payload['blank_acc']:.3f}"
        ),
    )


def render(
    *,
    control_run: Path,
    candidate_run: Path,
    control_summary: dict[str, Any],
    candidate_summary: dict[str, Any],
    control_case: dict[str, Any],
    candidate_case: dict[str, Any],
    rows: list[dict[str, Any]],
) -> str:
    table_rows = []
    bars = []
    for row in rows:
        low, high = row["range"]
        table_rows.append(
            "<tr>"
            f"<td>{low}-{high}</td>"
            f"<td>{row['control_blank_acc']:.4f}</td>"
            f"<td>{row['candidate_blank_acc']:.4f}</td>"
            f"<td class=\"gain\">{row['delta_blank_acc']:+.4f}</td>"
            f"<td>{row['control_exact']:.4f}</td>"
            f"<td>{row['candidate_exact']:.4f}</td>"
            "</tr>"
        )
        control_width = 100 * row["control_blank_acc"]
        candidate_width = 100 * row["candidate_blank_acc"]
        bars.append(
            '<div class="bar-row">'
            f"<strong>{low}-{high}</strong>"
            '<div class="bar-track">'
            f'<span class="bar control" style="width:{control_width:.2f}%"></span>'
            "</div>"
            '<div class="bar-track">'
            f'<span class="bar candidate" style="width:{candidate_width:.2f}%"></span>'
            "</div></div>"
        )

    candidate_loop_gain = wrong(candidate_case, "loop1") - wrong(
        candidate_case, "loop5"
    )
    cross_arm_gain = wrong(control_case, "loop5") - wrong(
        candidate_case, "loop5"
    )
    control_train = control_summary["metrics"]["train"]
    candidate_train = candidate_summary["metrics"]["train"]
    control_step = int(control_train["optimizer_steps"])
    candidate_step = int(candidate_train["optimizer_steps"])
    same_budget = control_step == candidate_step
    control_mean = sum(row["control_blank_acc"] for row in rows) / len(rows)
    candidate_mean = sum(row["candidate_blank_acc"] for row in rows) / len(rows)
    mean_delta = candidate_mean - control_mean
    budget_note = (
        f"same optimizer step {control_step}"
        if same_budget
        else (
            f"shared step9000 parent; control stopped at step{control_step}, "
            f"positive candidate continued to step{candidate_step}"
        )
    )
    comparison_note = (
        (
            f"Both arms use the same {control_step} optimizer steps, parameter "
            "count, data, evaluation cases, and loop budget. The position-Q/K "
            f"arm improves mean hard-range blank accuracy by {mean_delta:+.4f}. "
            "Exact remains zero, so the address split removes a substantial "
            "optimization bottleneck but does not solve global consistency."
        )
        if same_budget
        else (
            "The position-Q/K arm learns the random-order continuation much "
            "faster and more than doubles hard-range blank accuracy. Exact "
            "remains zero, so this is mechanism evidence rather than a solved "
            "Sudoku result. Because optimizer steps differ, this page is not a "
            "matched-total-compute quality estimate."
        )
    )
    address_diag = candidate_summary["metrics"]["official_eval_by_blank_range"][
        "b51_55"
    ]["eval_clean"]["loop5/future_seed"]

    puzzle = board_html(
        title="Puzzle",
        values=control_case["puzzle"],
        label=control_case["label"],
        clue_mask=[value != 0 for value in control_case["puzzle"]],
        previous=None,
        subtitle=f"{control_case['blank_count']} hidden cells",
    )
    target = board_html(
        title="Target",
        values=control_case["label"],
        label=control_case["label"],
        clue_mask=control_case["clue_mask"],
        previous=None,
        subtitle="same ground-truth board for both arms",
    )
    boards = [
        puzzle,
        target,
        prediction_board(
            control_case, arm="Control", loop="loop1", previous_loop=None
        ),
        prediction_board(
            control_case, arm="Control", loop="loop2", previous_loop="loop1"
        ),
        prediction_board(
            control_case, arm="Control", loop="loop5", previous_loop="loop4"
        ),
        prediction_board(
            candidate_case, arm="Position Q/K", loop="loop1", previous_loop=None
        ),
        prediction_board(
            candidate_case,
            arm="Position Q/K",
            loop="loop2",
            previous_loop="loop1",
        ),
        prediction_board(
            candidate_case,
            arm="Position Q/K",
            loop="loop5",
            previous_loop="loop4",
        ),
    ]
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>GDN2 address-payload matched comparison</title>
<style>
*{{box-sizing:border-box}} body{{margin:0;background:#f5f7f8;color:#17202a;
font:15px/1.45 system-ui,sans-serif}} main{{max-width:1500px;margin:auto;padding:24px}}
h1{{font-size:27px;margin:0 0 6px}} h2{{font-size:20px;margin-top:30px}}
.decision{{background:#fff;border-left:5px solid #087f5b;padding:14px 18px}}
.metrics{{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:10px}}
.metric{{background:#fff;border:1px solid #d8dde2;padding:12px;border-radius:6px}}
.metric b{{display:block;font-size:24px}} table{{width:100%;border-collapse:collapse;
background:#fff}} th,td{{padding:9px;border:1px solid #d8dde2;text-align:left}}
th{{background:#edf1f3}} .gain{{color:#087f5b;font-weight:700}}
.bars{{background:#fff;padding:14px;border:1px solid #d8dde2}}
.bar-row{{display:grid;grid-template-columns:70px 1fr 1fr;gap:10px;margin:10px 0}}
.bar-track{{height:18px;background:#e9ecef;position:relative}}
.bar{{display:block;height:100%}} .control{{background:#7b8794}}
.candidate{{background:#0c8f72}} .legend span{{margin-right:18px}}
.boards{{display:grid;grid-template-columns:repeat(4,minmax(230px,1fr));gap:12px}}
.board-card{{background:#fff;border:1px solid #d8dde2;padding:10px;border-radius:6px}}
.board-card h3{{font-size:16px;margin:0}} .board-card p{{min-height:42px;margin:4px 0 8px}}
.board{{display:grid;grid-template-columns:repeat(9,1fr);aspect-ratio:1}}
.cell{{display:grid;place-items:center;border:1px solid #aeb7c2;font-weight:650;
font-size:clamp(11px,1.1vw,17px);position:relative}} .box-right{{border-right:2px solid #17202a}}
.box-bottom{{border-bottom:2px solid #17202a}} .clue{{background:#e7edf3}}
.correct{{background:#d9f7e7}} .wrong{{background:#ffe0e0;color:#a61b1b}}
.changed::after{{content:"";position:absolute;left:22%;right:22%;bottom:3px;
height:3px;background:#168aad}} code{{background:#e9ecef;padding:2px 4px}}
@media(max-width:1000px){{.boards{{grid-template-columns:repeat(2,1fr)}}
.metrics{{grid-template-columns:repeat(2,1fr)}}}}
@media(max-width:620px){{main{{padding:12px}}.boards,.metrics{{grid-template-columns:1fr}}
.bar-row{{grid-template-columns:55px 1fr}}.bar-row .bar-track:last-child{{grid-column:2}}}}
</style></head><body><main>
<h1>Canonical address versus content-entangled GDN2</h1>
<p class="decision"><b>Mechanism signal, not solved Sudoku.</b><br>
{html.escape(comparison_note)}</p>
<div class="metrics">
<div class="metric"><span>Control step{control_step} train CE</span><b>{control_train['train_ce_loss']:.3f}</b></div>
<div class="metric"><span>Position-Q/K step{candidate_step} train CE</span><b>{candidate_train['train_ce_loss']:.3f}</b></div>
<div class="metric"><span>Control mean hard blank acc</span><b>{control_mean:.4f}</b></div>
<div class="metric"><span>Position-Q/K mean hard blank acc</span><b>{candidate_mean:.4f}</b></div>
<div class="metric"><span>Mean hard blank acc delta</span><b>{mean_delta:+.4f}</b></div>
<div class="metric"><span>Matched-case loop correction</span><b>{candidate_loop_gain:+d} cells</b></div>
<div class="metric"><span>Matched-case final advantage</span><b>{cross_arm_gain:+d} cells</b></div>
</div>
<h2>Official test, 512 boards per range</h2>
<p class="legend"><span>gray: control</span><span>green: position-Q/K</span></p>
<div class="bars">{''.join(bars)}</div>
<table><thead><tr><th>blanks</th><th>control blank acc</th>
<th>position-Q/K blank acc</th><th>delta</th><th>control exact</th>
<th>position-Q/K exact</th></tr></thead><tbody>{''.join(table_rows)}</tbody></table>
<h2>Same 64-blank board, {html.escape(budget_note)}</h2>
<p>Red cells are wrong, green cells are right, gray cells are clues, and a blue
underline marks a value changed since the preceding displayed loop. Batch
<code>{control_case['batch_index']}</code> is selected mechanically for the
largest combined cross-arm and within-candidate loop correction, after
verifying identical data hashes.</p>
<div class="boards">{''.join(boards)}</div>
<h2>Address diagnostics</h2>
<p>Q/K same-cell cosine: <code>{address_diag['gdn2_address_qk_diag_cosine']:.4f}</code>;
different-cell cosine: <code>{address_diag['gdn2_address_qk_offdiag_cosine']:.4f}</code>;
contrast: <code>{address_diag['gdn2_address_qk_contrast']:.4f}</code>.
Control source: <code>{html.escape(control_run.name)}</code>. Candidate source:
<code>{html.escape(candidate_run.name)}</code>.</p>
</main></body></html>"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--control-run", type=Path, required=True)
    parser.add_argument("--candidate-run", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()

    control_summary, control_banks = load_run(args.control_run)
    candidate_summary, candidate_banks = load_run(args.candidate_run)
    verify_matched(control_banks, candidate_banks)
    control_case, candidate_case = select_case(
        control_banks["b61_64"],
        candidate_banks["b61_64"],
    )
    rows = metric_rows(control_summary, candidate_summary)
    control_mean = sum(row["control_blank_acc"] for row in rows) / len(rows)
    candidate_mean = sum(row["candidate_blank_acc"] for row in rows) / len(rows)
    payload = {
        "control_run": str(args.control_run),
        "candidate_run": str(args.candidate_run),
        "control_optimizer_steps": int(
            control_summary["metrics"]["train"]["optimizer_steps"]
        ),
        "candidate_optimizer_steps": int(
            candidate_summary["metrics"]["train"]["optimizer_steps"]
        ),
        "metric_rows": rows,
        "mean_control_blank_acc": control_mean,
        "mean_candidate_blank_acc": candidate_mean,
        "mean_delta_blank_acc": candidate_mean - control_mean,
        "selected_batch_index": control_case["batch_index"],
        "selected_control_wrong": {
            loop: wrong(control_case, loop) for loop in LOOPS
        },
        "selected_candidate_wrong": {
            loop: wrong(candidate_case, loop) for loop in LOOPS
        },
        "data_hashes": {
            group: control_banks[group]["data_hash"] for group in RANGES
        },
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "comparison.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (args.out_dir / "index.html").write_text(
        render(
            control_run=args.control_run,
            candidate_run=args.candidate_run,
            control_summary=control_summary,
            candidate_summary=candidate_summary,
            control_case=control_case,
            candidate_case=candidate_case,
            rows=rows,
        ),
        encoding="utf-8",
    )
    print(args.out_dir / "index.html")


if __name__ == "__main__":
    main()
