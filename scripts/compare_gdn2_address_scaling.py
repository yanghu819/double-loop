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
    board_html,
    case_map,
    load_run,
    verify_matched,
    wrong,
)


def official_rows(
    earlier: dict[str, Any],
    later: dict[str, Any],
) -> list[dict[str, Any]]:
    rows = []
    for group in RANGES:
        left = earlier["metrics"]["official_eval_by_blank_range"][group]
        right = later["metrics"]["official_eval_by_blank_range"][group]
        left_final = left["eval_clean"]["loop5"]
        right_final = right["eval_clean"]["loop5"]
        rows.append(
            {
                "group": group,
                "range": left["blank_range"],
                "earlier_blank_acc": float(left_final["blank_acc"]),
                "later_blank_acc": float(right_final["blank_acc"]),
                "delta_blank_acc": float(right_final["blank_acc"])
                - float(left_final["blank_acc"]),
                "earlier_exact": float(left_final["label_exact"]),
                "later_exact": float(right_final["label_exact"]),
                "delta_exact": float(right_final["label_exact"])
                - float(left_final["label_exact"]),
                "later_loop1_exact": float(
                    right["eval_clean"]["loop1"]["label_exact"]
                ),
            }
        )
    return rows


def select_cases(
    earlier_banks: dict[str, dict[str, Any]],
    later_banks: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    for group, mode in (
        ("b56_60", "exact opening"),
        ("b61_64", "largest hard improvement"),
        ("b61_64", "largest hard regression"),
    ):
        earlier = case_map(earlier_banks[group])
        later = case_map(later_banks[group])
        common = sorted(earlier)
        if mode == "exact opening":
            eligible = [
                index
                for index in common
                if wrong(earlier[index], "loop5") > 0
                and wrong(later[index], "loop5") == 0
            ]
            if not eligible:
                raise ValueError("no matched exact-opening case")
            index = max(
                eligible,
                key=lambda value: (
                    wrong(earlier[value], "loop5"),
                    wrong(later[value], "loop1")
                    - wrong(later[value], "loop5"),
                    -value,
                ),
            )
        elif mode == "largest hard improvement":
            index = max(
                common,
                key=lambda value: (
                    wrong(earlier[value], "loop5")
                    - wrong(later[value], "loop5"),
                    wrong(later[value], "loop1")
                    - wrong(later[value], "loop5"),
                    -value,
                ),
            )
        else:
            index = min(
                common,
                key=lambda value: (
                    wrong(earlier[value], "loop5")
                    - wrong(later[value], "loop5"),
                    value,
                ),
            )
        selected.append(
            {
                "group": group,
                "mode": mode,
                "batch_index": index,
                "earlier": earlier[index],
                "later": later[index],
            }
        )
    return selected


def prediction_board(
    case: dict[str, Any],
    *,
    step: int,
    loop: str,
    previous_loop: str | None,
) -> str:
    payload = case["loops"][loop]
    previous = (
        case["loops"][previous_loop]["prediction"] if previous_loop else None
    )
    return board_html(
        title=f"step{step} {loop}",
        values=payload["prediction"],
        label=case["label"],
        clue_mask=case["clue_mask"],
        previous=previous,
        subtitle=(
            f"wrong blanks {payload['wrong_blank_count']}/{case['blank_count']}; "
            f"blank accuracy {payload['blank_acc']:.3f}"
        ),
    )


def case_section(
    payload: dict[str, Any],
    *,
    earlier_step: int,
    later_step: int,
) -> str:
    earlier = payload["earlier"]
    later = payload["later"]
    boards = [
        board_html(
            title="Puzzle",
            values=earlier["puzzle"],
            label=earlier["label"],
            clue_mask=[value != 0 for value in earlier["puzzle"]],
            previous=None,
            subtitle=f"{earlier['blank_count']} hidden cells",
        ),
        board_html(
            title="Target",
            values=earlier["label"],
            label=earlier["label"],
            clue_mask=earlier["clue_mask"],
            previous=None,
            subtitle="shared ground truth",
        ),
    ]
    for step, case in ((earlier_step, earlier), (later_step, later)):
        boards.extend(
            [
                prediction_board(
                    case, step=step, loop="loop1", previous_loop=None
                ),
                prediction_board(
                    case, step=step, loop="loop3", previous_loop="loop2"
                ),
                prediction_board(
                    case, step=step, loop="loop5", previous_loop="loop4"
                ),
            ]
        )
    earlier_delta = wrong(earlier, "loop1") - wrong(earlier, "loop5")
    later_delta = wrong(later, "loop1") - wrong(later, "loop5")
    cross_delta = wrong(earlier, "loop5") - wrong(later, "loop5")
    return (
        f'<section class="case"><h2>{html.escape(payload["mode"].title())}: '
        f'{html.escape(payload["group"])} batch {payload["batch_index"]}</h2>'
        "<p>"
        f"Final wrong-cell change from step{earlier_step} to step{later_step}: "
        f"<b>{cross_delta:+d}</b>. Loop1-to-loop5 correction: "
        f"<code>{earlier_delta:+d}</code> versus "
        f"<code>{later_delta:+d}</code>."
        "</p>"
        f'<div class="boards">{"".join(boards)}</div></section>'
    )


def render(
    *,
    earlier_run: Path,
    later_run: Path,
    earlier_summary: dict[str, Any],
    later_summary: dict[str, Any],
    rows: list[dict[str, Any]],
    selected: list[dict[str, Any]],
) -> str:
    earlier_train = earlier_summary["metrics"]["train"]
    later_train = later_summary["metrics"]["train"]
    earlier_step = int(earlier_train["optimizer_steps"])
    later_step = int(later_train["optimizer_steps"])
    earlier_mean = sum(row["earlier_blank_acc"] for row in rows) / len(rows)
    later_mean = sum(row["later_blank_acc"] for row in rows) / len(rows)
    table_rows = "".join(
        "<tr>"
        f"<td>{row['range'][0]}-{row['range'][1]}</td>"
        f"<td>{row['earlier_blank_acc']:.4f}</td>"
        f"<td>{row['later_blank_acc']:.4f}</td>"
        f"<td>{row['delta_blank_acc']:+.4f}</td>"
        f"<td>{row['earlier_exact']:.4f}</td>"
        f"<td>{row['later_exact']:.4f}</td>"
        f"<td>{row['delta_exact']:+.4f}</td>"
        f"<td>{row['later_loop1_exact']:.4f}</td>"
        "</tr>"
        for row in rows
    )
    cases = "".join(
        case_section(
            payload,
            earlier_step=earlier_step,
            later_step=later_step,
        )
        for payload in selected
    )
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>GDN2 address hard-stage scaling</title>
<style>
*{{box-sizing:border-box}} body{{margin:0;background:#f5f7f8;color:#17202a;
font:15px/1.45 system-ui,sans-serif}} main{{max-width:1500px;margin:auto;padding:24px}}
h1{{font-size:27px;margin:0 0 6px}} h2{{font-size:20px;margin-top:30px}}
.decision{{background:#fff;border-left:5px solid #c47f00;padding:14px 18px}}
.metrics{{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:10px}}
.metric{{background:#fff;border:1px solid #d8dde2;padding:12px;border-radius:6px}}
.metric b{{display:block;font-size:24px}} table{{width:100%;border-collapse:collapse;
background:#fff}} th,td{{padding:8px;border:1px solid #d8dde2;text-align:left}}
th{{background:#edf1f3}} .boards{{display:grid;
grid-template-columns:repeat(4,minmax(230px,1fr));gap:12px}}
.board-card{{background:#fff;border:1px solid #d8dde2;padding:10px;border-radius:6px}}
.board-card h3{{font-size:16px;margin:0}} .board-card p{{min-height:42px;margin:4px 0 8px}}
.board{{display:grid;grid-template-columns:repeat(9,1fr);aspect-ratio:1}}
.cell{{display:grid;place-items:center;border:1px solid #aeb7c2;font-weight:650;
font-size:clamp(11px,1.1vw,17px);position:relative}} .box-right{{border-right:2px solid #17202a}}
.box-bottom{{border-bottom:2px solid #17202a}} .clue{{background:#e7edf3}}
.correct{{background:#d9f7e7}} .wrong{{background:#ffe0e0;color:#a61b1b}}
.changed::after{{content:"";position:absolute;left:22%;right:22%;bottom:3px;
height:3px;background:#168aad}} code{{background:#e9ecef;padding:2px 4px}}
.case{{border-top:1px solid #cbd2d9;margin-top:28px}}
@media(max-width:1000px){{.boards{{grid-template-columns:repeat(2,1fr)}}}}
@media(max-width:620px){{main{{padding:12px}}.boards{{grid-template-columns:1fr}}}}
</style></head><body><main>
<h1>Clean hard-stage scaling: step{earlier_step} versus step{later_step}</h1>
<p class="decision"><b>Weak opening, below the preregistered continuation gate.</b><br>
Official exact opens to 0.0059 in two ranges and later loops create the solves,
but mean hard blank accuracy improves by only {later_mean - earlier_mean:+.4f}
to {later_mean:.4f}, below the +0.04 partial threshold. The hardest range still
has zero exact, and paired cases include both large improvements and regressions.
Do not authorize another short continuation from this result.</p>
<div class="metrics">
<div class="metric"><span>step{earlier_step} train CE</span><b>{earlier_train['train_ce_loss']:.3f}</b></div>
<div class="metric"><span>step{later_step} train CE</span><b>{later_train['train_ce_loss']:.3f}</b></div>
<div class="metric"><span>step{earlier_step} mean hard blank</span><b>{earlier_mean:.4f}</b></div>
<div class="metric"><span>step{later_step} mean hard blank</span><b>{later_mean:.4f}</b></div>
<div class="metric"><span>mean hard blank delta</span><b>{later_mean - earlier_mean:+.4f}</b></div>
</div>
<h2>Official test, 512 boards per range</h2>
<table><thead><tr><th>blanks</th><th>step{earlier_step} blank</th>
<th>step{later_step} blank</th><th>blank delta</th>
<th>step{earlier_step} exact</th><th>step{later_step} exact</th>
<th>exact delta</th><th>step{later_step} loop1 exact</th></tr></thead>
<tbody>{table_rows}</tbody></table>
<p>Red cells are wrong, green cells are right, gray cells are clues, and blue
underlines mark values changed since the preceding loop. Cases are selected
mechanically after verifying identical board and label hashes.</p>
{cases}
<p>Earlier source: <code>{html.escape(earlier_run.name)}</code>. Later source:
<code>{html.escape(later_run.name)}</code>.</p>
</main></body></html>"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--earlier-run", type=Path, required=True)
    parser.add_argument("--later-run", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()

    earlier_summary, earlier_banks = load_run(args.earlier_run)
    later_summary, later_banks = load_run(args.later_run)
    verify_matched(earlier_banks, later_banks)
    rows = official_rows(earlier_summary, later_summary)
    selected = select_cases(earlier_banks, later_banks)
    earlier_step = int(earlier_summary["metrics"]["train"]["optimizer_steps"])
    later_step = int(later_summary["metrics"]["train"]["optimizer_steps"])
    earlier_mean = sum(row["earlier_blank_acc"] for row in rows) / len(rows)
    later_mean = sum(row["later_blank_acc"] for row in rows) / len(rows)
    payload = {
        "earlier_run": str(args.earlier_run),
        "later_run": str(args.later_run),
        "earlier_optimizer_steps": earlier_step,
        "later_optimizer_steps": later_step,
        "metric_rows": rows,
        "mean_earlier_blank_acc": earlier_mean,
        "mean_later_blank_acc": later_mean,
        "mean_delta_blank_acc": later_mean - earlier_mean,
        "selected_cases": [
            {
                "group": item["group"],
                "mode": item["mode"],
                "batch_index": item["batch_index"],
                "earlier_wrong": {
                    loop: wrong(item["earlier"], loop) for loop in LOOPS
                },
                "later_wrong": {
                    loop: wrong(item["later"], loop) for loop in LOOPS
                },
            }
            for item in selected
        ],
        "data_hashes": {
            group: earlier_banks[group]["data_hash"] for group in RANGES
        },
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "comparison.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (args.out_dir / "index.html").write_text(
        render(
            earlier_run=args.earlier_run,
            later_run=args.later_run,
            earlier_summary=earlier_summary,
            later_summary=later_summary,
            rows=rows,
            selected=selected,
        ),
        encoding="utf-8",
    )
    print(args.out_dir / "index.html")


if __name__ == "__main__":
    main()
