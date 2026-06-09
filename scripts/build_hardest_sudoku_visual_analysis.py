#!/usr/bin/env python3
"""Build a static visual analysis for the current hardest doable Sudoku task.

This script only reads archived experiment metrics. It does not train, evaluate,
or run any model, so it is safe to run on a laptop while preserving the project's
GPU-only rule for model smoke/training.
"""

from __future__ import annotations

import html
import json
import math
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = REPO_ROOT / "runs/hardest-sudoku-frontier-visual-analysis-20260609"

CLEAN_12_JSON = REPO_ROOT / "runs/frontier-12x12-d256-h72-20260604T0920Z-f67e6a2/output/futureseed_loop_seed52.json"
CLEAN_12_CASE = REPO_ROOT / "runs/frontier-12x12-d256-h72-20260604T0920Z-f67e6a2/output/futureseed_loop_case_seed52.html"
CLEAN_16_JSON = REPO_ROOT / "runs/bf16-b32-shaped16x16-d256-l12-loop8-h24-s600-20260609T0638Z-a9a0e9d/output/futureseed_loop_seed52.json"
CLEAN_16_CASE = REPO_ROOT / "runs/bf16-b32-shaped16x16-d256-l12-loop8-h24-s600-20260609T0638Z-a9a0e9d/output/futureseed_loop_case_seed52.html"
H72_CASEBANK_INDEX = REPO_ROOT / "runs/casebank-h72-12x12-d256-l12-loop5-s1100-20260609T1013Z-8a2eb57/output/case_bank/h72/index.html"
H72_CASEBANK_JSON = REPO_ROOT / "runs/casebank-h72-12x12-d256-l12-loop5-s1100-20260609T1013Z-8a2eb57/output/case_bank/h72/cases.json"
UPPERBOUND_SUMMARY = REPO_ROOT / "runs/futureseed-upperbound-visual-analysis-20260608/summary.json"
UNIT_16_JSON = REPO_ROOT / "runs/frontier-16x16-unitmem-h80curr-d192-l10-loop7-b48-20260605T0850Z-0be64f4/output/futureseed_loop_seed52.json"
UNIT_25_JSON = REPO_ROOT / "runs/upperbound-25x25-unitmem-h100curr-d160-l8-loop5-b32-20260607T050029Z-46a0b06/output/futureseed_loop_seed52.json"


LOOP_RE = re.compile(r"^loop(\d+)$")
COLOR_EXACT = "#1f6feb"
COLOR_BLANK = "#2da44e"
COLOR_VALID = "#bf8700"
COLOR_MUTED = "#57606a"


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def loop_number(name: str) -> int:
    match = LOOP_RE.match(name)
    if match is None:
        raise ValueError(f"not a loop key: {name}")
    return int(match.group(1))


def hole_number(name: str) -> int:
    if not name.startswith("holes"):
        raise ValueError(f"not a hole key: {name}")
    return int(name.removeprefix("holes"))


def eval_clean_for_hole(metrics: dict[str, Any], hole_key: str) -> dict[str, Any]:
    raw = metrics["eval_by_holes"][hole_key]
    if "eval_clean" in raw:
        return raw["eval_clean"]
    return raw["eval_clean"] if "eval_clean" in raw else raw


def extract_hole_rows(metrics: dict[str, Any], *, run_name: str, size: int, kind: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for hole_key in sorted(metrics["eval_by_holes"], key=hole_number):
        loops = eval_clean_for_hole(metrics, hole_key)
        loop_keys = sorted([key for key in loops if LOOP_RE.match(key)], key=loop_number)
        if not loop_keys:
            continue
        loop1 = loops[loop_keys[0]]
        final_key = loop_keys[-1]
        final = loops[final_key]
        holes = hole_number(hole_key)
        blank_acc = float(final["blank_acc"])
        exact = float(final["label_exact"])
        independent = blank_acc**holes
        rows.append(
            {
                "run": run_name,
                "size": size,
                "kind": kind,
                "holes": holes,
                "loop_final": final_key,
                "loop1_exact": float(loop1["label_exact"]),
                "loop1_blank_acc": float(loop1["blank_acc"]),
                "loop_final_exact": exact,
                "loop_final_blank_acc": blank_acc,
                "valid": float(final.get("valid_sudoku", 0.0)),
                "independent_exact_estimate": independent,
                "exact_over_independent": exact / max(independent, 1e-12),
            }
        )
    return rows


def extract_loop_curve(metrics: dict[str, Any], hole_key: str) -> list[dict[str, Any]]:
    loops = eval_clean_for_hole(metrics, hole_key)
    curve: list[dict[str, Any]] = []
    for key in sorted([key for key in loops if LOOP_RE.match(key)], key=loop_number):
        item = loops[key]
        curve.append(
            {
                "loop": loop_number(key),
                "exact": float(item["label_exact"]),
                "blank_acc": float(item["blank_acc"]),
                "valid": float(item.get("valid_sudoku", 0.0)),
            }
        )
    return curve


def fs_stats(metrics: dict[str, Any], hole_key: str) -> dict[str, Any]:
    loops = eval_clean_for_hole(metrics, hole_key)
    key = sorted([key for key in loops if key.endswith("/future_seed")], key=lambda row: loop_number(row.split("/")[0]))[0]
    return loops[key]


def pct(value: float, digits: int = 2) -> str:
    return f"{100.0 * value:.{digits}f}%"


def num(value: float, digits: int = 4) -> str:
    if value == 0:
        return "0"
    if abs(value) < 0.001:
        return f"{value:.2e}"
    return f"{value:.{digits}f}"


def rel(path: Path) -> str:
    return path.relative_to(REPO_ROOT).as_posix()


def html_link(path: Path, label: str) -> str:
    return f'<a href="../{html.escape(path.relative_to(OUT_DIR.parent).as_posix())}">{html.escape(label)}</a>'


def svg_line_chart(
    title: str,
    x_values: list[float],
    series: list[tuple[str, str, list[float]]],
    *,
    x_label: str,
    y_label: str,
    width: int = 840,
    height: int = 320,
    y_max: float = 1.0,
) -> str:
    left, right, top, bottom = 58, 24, 38, 44
    inner_w = width - left - right
    inner_h = height - top - bottom
    min_x, max_x = min(x_values), max(x_values)

    def sx(x: float) -> float:
        if max_x == min_x:
            return left + inner_w / 2
        return left + (x - min_x) / (max_x - min_x) * inner_w

    def sy(y: float) -> float:
        return top + (1.0 - min(max(y / y_max, 0.0), 1.0)) * inner_h

    lines: list[str] = []
    for frac in [0, 0.25, 0.5, 0.75, 1.0]:
        y = top + (1.0 - frac) * inner_h
        lines.append(f'<line x1="{left}" x2="{width - right}" y1="{y:.1f}" y2="{y:.1f}" class="gridline"/>')
        lines.append(f'<text x="{left - 10}" y="{y + 4:.1f}" text-anchor="end" class="tick">{int(frac * 100)}</text>')
    for x in x_values:
        x_pos = sx(x)
        lines.append(f'<line x1="{x_pos:.1f}" x2="{x_pos:.1f}" y1="{top}" y2="{height - bottom}" class="xgrid"/>')
        label = int(x) if float(x).is_integer() else x
        lines.append(f'<text x="{x_pos:.1f}" y="{height - bottom + 22}" text-anchor="middle" class="tick">{label}</text>')

    paths: list[str] = []
    legend_x = left
    for name, color, values in series:
        coords = " ".join(f"{sx(x):.1f},{sy(y):.1f}" for x, y in zip(x_values, values))
        paths.append(f'<polyline points="{coords}" fill="none" stroke="{color}" stroke-width="3" stroke-linecap="round" stroke-linejoin="round"/>')
        for x, y in zip(x_values, values):
            paths.append(f'<circle cx="{sx(x):.1f}" cy="{sy(y):.1f}" r="4" fill="{color}"/>')
        paths.append(f'<rect x="{legend_x}" y="12" width="11" height="11" fill="{color}" rx="2"/>')
        paths.append(f'<text x="{legend_x + 17}" y="22" class="legend">{html.escape(name)}</text>')
        legend_x += 138

    return f"""
<figure class="chart">
  <figcaption>{html.escape(title)}</figcaption>
  <svg viewBox="0 0 {width} {height}" role="img" aria-label="{html.escape(title)}">
    <text x="{width / 2:.1f}" y="{height - 6}" text-anchor="middle" class="axislabel">{html.escape(x_label)}</text>
    <text x="16" y="{height / 2:.1f}" text-anchor="middle" class="axislabel rotate">{html.escape(y_label)}</text>
    {''.join(lines)}
    <line x1="{left}" x2="{width - right}" y1="{height - bottom}" y2="{height - bottom}" class="axis"/>
    <line x1="{left}" x2="{left}" y1="{top}" y2="{height - bottom}" class="axis"/>
    {''.join(paths)}
  </svg>
</figure>
"""


def bar_cell(value: float, *, label: str | None = None) -> str:
    display = label if label is not None else pct(value)
    width = max(0.0, min(value * 100.0, 100.0))
    return f'<div class="barcell"><div class="bar" style="width:{width:.2f}%"></div><span>{html.escape(display)}</span></div>'


def frontier_table(rows: list[dict[str, Any]], *, selected_key: tuple[str, int] | None = None) -> str:
    body: list[str] = []
    for row in rows:
        classes = ["selected"] if selected_key == (row["run"], row["holes"]) else []
        note = row.get("note", "")
        body.append(
            "<tr{}><td>{}</td><td>{}</td><td>{}</td><td>{}</td><td>{}</td>"
            "<td>{}</td><td>{}</td><td>{}</td><td>{}</td></tr>".format(
                f' class="{" ".join(classes)}"' if classes else "",
                html.escape(row["run"]),
                row["size"],
                row["holes"],
                html.escape(row.get("kind", "")),
                html.escape(row["loop_final"]),
                bar_cell(float(row["loop_final_exact"])),
                bar_cell(float(row["loop_final_blank_acc"])),
                num(float(row["independent_exact_estimate"])),
                html.escape(note),
            )
        )
    return (
        '<table class="metric-table"><thead><tr><th>run</th><th>size</th><th>holes</th>'
        '<th>line</th><th>final loop</th><th>exact</th><th>blank acc</th>'
        '<th>independent exact estimate</th><th>read</th></tr></thead><tbody>'
        + "".join(body)
        + "</tbody></table>"
    )


def loop_table(curve: list[dict[str, Any]]) -> str:
    body = []
    prev = curve[0]
    for row in curve:
        d_exact = row["exact"] - prev["exact"]
        d_blank = row["blank_acc"] - prev["blank_acc"]
        body.append(
            f"<tr><td>{row['loop']}</td><td>{bar_cell(row['exact'])}</td>"
            f"<td>{bar_cell(row['blank_acc'])}</td><td>{pct(d_exact, 2)}</td>"
            f"<td>{pct(d_blank, 2)}</td></tr>"
        )
        prev = row
    return (
        '<table class="metric-table compact"><thead><tr><th>loop</th><th>exact</th>'
        '<th>blank acc</th><th>exact delta</th><th>blank acc delta</th></tr></thead><tbody>'
        + "".join(body)
        + "</tbody></table>"
    )


def build_summary() -> dict[str, Any]:
    clean12 = load_json(CLEAN_12_JSON)
    clean16 = load_json(CLEAN_16_JSON)
    unit16 = load_json(UNIT_16_JSON)
    unit25 = load_json(UNIT_25_JSON)
    upper = load_json(UPPERBOUND_SUMMARY)

    clean12_metrics = clean12["metrics"]
    clean16_metrics = clean16["metrics"]
    unit16_metrics = unit16["metrics"]
    unit25_metrics = unit25["metrics"]

    clean12_rows = extract_hole_rows(clean12_metrics, run_name="12x12 clean FutureSeed+loop", size=12, kind="clean")
    clean16_rows = extract_hole_rows(clean16_metrics, run_name="16x16 clean bf16", size=16, kind="clean")
    unit16_rows = extract_hole_rows(unit16_metrics, run_name="16x16 unit-memory diagnostic", size=16, kind="unit-memory")
    unit25_rows = extract_hole_rows(unit25_metrics, run_name="25x25 unit-memory diagnostic", size=25, kind="unit-memory")

    nine_h32 = next(
        row
        for row in upper["points"]
        if row["run"] == "9x9 FutureSeed" and int(row["holes"]) == 32
    )

    selected = next(row for row in clean12_rows if row["holes"] == 72)
    h84 = next(row for row in clean12_rows if row["holes"] == 84)
    clean16_h24 = next(row for row in clean16_rows if row["holes"] == 24)
    h60 = next(row for row in clean12_rows if row["holes"] == 60)

    ladder = [
        {
            **nine_h32,
            "kind": "clean",
            "note": "already learnable; not the hardest frontier",
            "independent_exact_estimate": nine_h32["independent_exact_estimate"],
            "loop_final_exact": nine_h32["loop_final_exact"],
            "loop_final_blank_acc": nine_h32["loop_final_blank_acc"],
        },
        {**h60, "note": "strongly doable, but too far from the cliff"},
        {**selected, "note": "selected: hardest clean nonzero exact"},
        {**h84, "note": "closed under this run"},
        {**clean16_h24, "note": "larger, but exact is still closed"},
        {**next(row for row in unit16_rows if row["holes"] == 80), "note": "harder but uses Sudoku unit memory"},
        {**next(row for row in unit25_rows if row["holes"] == 100), "note": "upper-bound diagnostic, not clean mainline"},
    ]
    ladder[0]["run"] = "9x9 clean FutureSeed+loop"
    ladder[0]["loop_final"] = ladder[0].get("loop_final", "loop5")

    h72_curve = extract_loop_curve(clean12_metrics, "holes72")
    h60_curve = extract_loop_curve(clean12_metrics, "holes60")
    fs = fs_stats(clean12_metrics, "holes72")
    case_bank = load_json(H72_CASEBANK_JSON) if H72_CASEBANK_JSON.exists() else None

    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "selected_task": {
            **selected,
            "name": "12x12 random-holes h72",
            "run": "frontier-12x12-d256-h72-20260604T0920Z-f67e6a2",
            "line": selected["run"],
            "source_sha": clean12["args"].get("source_sha", "f67e6a2e909dc39ce5ba48f622d93b1ec43f5ab3"),
            "size": 12,
            "boxes": "3x4",
            "holes": 72,
            "reason": "It is the hardest archived clean random-hole setting with nonzero exact solve rate. The next harder 12x12 point h84 is closed, and clean 16x16 is still closed.",
        },
        "clean_12_rows": clean12_rows,
        "ladder": ladder,
        "h72_curve": h72_curve,
        "h60_curve": h60_curve,
        "future_seed_h72": fs,
        "case_links": {
            "near_frontier_12x12_default_case": rel(CLEAN_12_CASE),
            "h72_case_bank": rel(H72_CASEBANK_INDEX),
            "clean_16x16_failure_case": rel(CLEAN_16_CASE),
        },
        "case_bank_summary": case_bank["summary"] if case_bank else None,
    }


def build_readme(summary: dict[str, Any]) -> str:
    selected = summary["selected_task"]
    curve = summary["h72_curve"]
    loop1 = curve[0]
    loop2 = curve[1]
    loop3 = curve[2]
    loop5 = curve[-1]
    fs = summary["future_seed_h72"]

    return f"""# Hardest Doable Sudoku Frontier Visual Analysis

Generated UTC: `{summary['generated_at_utc']}`

## Selection

Selected task: `{selected['name']}` from `{selected['run']}`.

This is the hardest currently archived clean task that is still doable. It uses random holes only, no row/column/box unit-memory mechanism, and reaches loop5 exact `{pct(selected['loop_final_exact'])}` at h72. The next harder 12x12 point, h84, is closed at exact `0.00%`; clean 16x16 is also still closed. That makes h72 the best current mechanism microscope: it has real successes, real failures, and a visible cliff.

## Mechanism Readout

- Loop1 at h72: exact `{pct(loop1['exact'])}`, blank accuracy `{pct(loop1['blank_acc'])}`.
- Loop2 at h72: exact `{pct(loop2['exact'])}`, blank accuracy `{pct(loop2['blank_acc'])}`.
- Loop3 at h72: exact `{pct(loop3['exact'])}`, blank accuracy `{pct(loop3['blank_acc'])}`.
- Loop5 at h72: exact `{pct(loop5['exact'])}`, blank accuracy `{pct(loop5['blank_acc'])}`.
- Loop gain: exact `{pct(loop5['exact'] - loop1['exact'])}`, blank accuracy `{pct(loop5['blank_acc'] - loop1['blank_acc'])}`.
- Independent-cell exact estimate at h72 is `{num(selected['independent_exact_estimate'])}`; real exact is `{pct(selected['loop_final_exact'])}`. The solved boards are therefore not just independent lucky cells. The loop sometimes lands in a coherent whole-board basin.
- FutureSeed h72 diagnostic: gate mean `{float(fs.get('fs_gate_mean', 0.0)):.3f}`, state norm `{float(fs.get('fs_state_norm', 0.0)):.3f}`. The current archive records only aggregate seed strength, not per-cell or per-loop causal contribution.

## Main Insight

FutureSeed alone is not enough at this frontier: loop1 exact is zero. The loop is the part that turns a noncausal global seed into board-level refinement.

The loop has two regimes. Loop1 to loop2 mostly repairs local cell reliability. Loop2 to loop5 adds much less blank accuracy, but it still moves exact upward, which means later loops are doing the harder board-level coordination work. The current failure mode is not simply "more single-cell accuracy"; h72 already has high blank accuracy and still has low exact because a few coupled mistakes break the whole board.

## Concrete Cases

- 12x12 archived case: `{summary['case_links']['near_frontier_12x12_default_case']}`. This is the concrete case exported by the original run. It is useful for reading loop refinement from the same trained model, but the archive did not save a full h72 case bank.
- New h72 case bank: `{summary['case_links']['h72_case_bank']}`. This reproduced the frontier and exported 4 solved-by-loop, 4 almost-solved, and 4 hard-failure h72 boards.
- 16x16 clean failure case: `{summary['case_links']['clean_16x16_failure_case']}`. This shows why 16x16 clean is not yet the right case-level microscope: exact is still closed.

The case bank confirms the loop story concretely. Solved-by-loop boards start with about 29-32 wrong hidden cells at loop1, drop to 5-11 at loop2, reach 0-3 by loop3, and finish exact by loop5. Almost-solved and hard-failure boards show the current limit: the last 1-5 coupled wrong cells can become sticky.

## Files

- `index.html`
- `frontier_selection.json`

"""


def build_html(summary: dict[str, Any]) -> str:
    selected = summary["selected_task"]
    curve = summary["h72_curve"]
    clean12_rows = summary["clean_12_rows"]
    ladder = summary["ladder"]
    fs = summary["future_seed_h72"]

    loop_x = [row["loop"] for row in curve]
    loop_chart = svg_line_chart(
        "12x12 h72: loops turn local guesses into a few full-board solves",
        loop_x,
        [
            ("exact", COLOR_EXACT, [row["exact"] for row in curve]),
            ("blank acc", COLOR_BLANK, [row["blank_acc"] for row in curve]),
            ("valid board", COLOR_VALID, [row["valid"] for row in curve]),
        ],
        x_label="loop",
        y_label="percent",
    )

    hole_x = [row["holes"] for row in clean12_rows]
    hole_chart = svg_line_chart(
        "12x12 clean ladder: h72 is the last nonzero clean point before collapse",
        hole_x,
        [
            ("loop final exact", COLOR_EXACT, [row["loop_final_exact"] for row in clean12_rows]),
            ("loop final blank acc", COLOR_BLANK, [row["loop_final_blank_acc"] for row in clean12_rows]),
        ],
        x_label="hidden cells",
        y_label="percent",
    )

    selected_key = ("12x12 clean FutureSeed+loop", 72)
    h72_loop_table = loop_table(curve)
    difficulty_table = frontier_table(ladder, selected_key=selected_key)

    h72 = selected
    loop1 = curve[0]
    loop2 = curve[1]
    loop3 = curve[2]
    loop5 = curve[-1]
    fs_gate = float(fs.get("fs_gate_mean", 0.0))
    fs_norm = float(fs.get("fs_state_norm", 0.0))

    clean_case_link = html_link(CLEAN_12_CASE, "open the archived 12x12 loop case")
    casebank_link = html_link(H72_CASEBANK_INDEX, "open the new h72 case bank")
    failure_case_link = html_link(CLEAN_16_CASE, "open the clean 16x16 failure case")

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Hardest Doable Sudoku Frontier</title>
  <style>
    :root {{
      color-scheme: light;
      --bg: #f6f8fa;
      --panel: #ffffff;
      --text: #24292f;
      --muted: #57606a;
      --line: #d0d7de;
      --blue: {COLOR_EXACT};
      --green: {COLOR_BLANK};
      --gold: {COLOR_VALID};
      --red: #cf222e;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      background: var(--bg);
      color: var(--text);
      font-family: ui-sans-serif, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      letter-spacing: 0;
    }}
    main {{ max-width: 1180px; margin: 0 auto; padding: 28px 24px 44px; }}
    header {{ margin-bottom: 18px; }}
    h1 {{ margin: 0 0 8px; font-size: 32px; line-height: 1.1; letter-spacing: 0; }}
    h2 {{ margin: 0 0 12px; font-size: 20px; letter-spacing: 0; }}
    h3 {{ margin: 0 0 8px; font-size: 16px; letter-spacing: 0; }}
    p {{ margin: 0 0 10px; line-height: 1.48; color: var(--muted); }}
    a {{ color: var(--blue); text-decoration-thickness: 1px; }}
    code {{ font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; }}
    .hero {{
      display: grid;
      grid-template-columns: minmax(0, 1.25fr) minmax(280px, 0.75fr);
      gap: 16px;
      align-items: stretch;
      margin-top: 16px;
    }}
    .panel {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 16px;
      box-shadow: 0 1px 1px rgba(31,35,40,0.03);
    }}
    .cards {{
      display: grid;
      grid-template-columns: repeat(4, minmax(0, 1fr));
      gap: 12px;
      margin: 16px 0;
    }}
    .card {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 14px;
      min-height: 108px;
    }}
    .kicker {{ color: var(--muted); font-size: 12px; text-transform: uppercase; font-weight: 700; }}
    .big {{ font-size: 30px; font-weight: 800; margin-top: 4px; color: var(--text); }}
    .small {{ font-size: 13px; color: var(--muted); }}
    .charts {{ display: grid; grid-template-columns: 1fr; gap: 14px; margin: 16px 0; }}
    .chart {{
      margin: 0;
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 14px;
    }}
    .chart figcaption {{ font-weight: 750; margin-bottom: 8px; }}
    svg {{ width: 100%; height: auto; display: block; }}
    .gridline {{ stroke: #d8dee4; stroke-width: 1; }}
    .xgrid {{ stroke: #eef1f4; stroke-width: 1; }}
    .axis {{ stroke: #57606a; stroke-width: 1.2; }}
    .tick, .legend, .axislabel {{ fill: #57606a; font-size: 12px; }}
    .rotate {{ transform: rotate(-90deg); transform-origin: 16px center; }}
    .metric-table {{
      width: 100%;
      border-collapse: collapse;
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      overflow: hidden;
      font-size: 13px;
    }}
    .metric-table th, .metric-table td {{
      padding: 9px 10px;
      border-bottom: 1px solid var(--line);
      text-align: left;
      vertical-align: middle;
    }}
    .metric-table th {{ background: #f0f3f6; color: #57606a; font-size: 12px; }}
    .metric-table tr:last-child td {{ border-bottom: 0; }}
    .metric-table tr.selected td {{ background: #ddf4ff; }}
    .compact td, .compact th {{ padding: 8px 10px; }}
    .barcell {{ position: relative; min-width: 96px; height: 24px; background: #f0f3f6; border-radius: 6px; overflow: hidden; }}
    .bar {{ position: absolute; inset: 0 auto 0 0; background: rgba(31,111,235,0.18); }}
    .barcell span {{ position: relative; display: flex; height: 100%; align-items: center; padding-left: 8px; font-weight: 700; color: #24292f; }}
    .twocol {{ display: grid; grid-template-columns: 1fr 1fr; gap: 14px; margin-top: 14px; }}
    .callout {{
      border-left: 4px solid var(--blue);
      padding: 12px 14px;
      background: #ddf4ff;
      border-radius: 8px;
      color: #24292f;
    }}
    .warn {{ border-left-color: var(--red); background: #ffebe9; }}
    .list {{ margin: 8px 0 0 18px; padding: 0; color: var(--muted); }}
    .list li {{ margin: 6px 0; }}
    @media (max-width: 900px) {{
      .hero, .twocol, .cards {{ grid-template-columns: 1fr; }}
      h1 {{ font-size: 26px; }}
      main {{ padding: 22px 14px 34px; }}
      .metric-table {{ display: block; overflow-x: auto; white-space: nowrap; }}
    }}
  </style>
</head>
<body>
<main>
  <header>
    <h1>Hardest doable Sudoku frontier</h1>
    <p>Generated from archived GPU1 runs at <code>{html.escape(summary['generated_at_utc'])}</code>. No model was run for this report.</p>
  </header>

  <section class="hero">
    <div class="panel">
      <h2>Chosen task: 12x12 random h72</h2>
      <p>{html.escape(selected['reason'])}</p>
      <p>The important property is not that h72 scores well. It barely works: loop5 exact is {pct(h72['loop_final_exact'])}. That is exactly why it is useful. It has enough solved boards to inspect, while h84 and clean 16x16 are still closed.</p>
      <div class="callout">
        <strong>Read:</strong> h72 is the current clean mechanism microscope. It is hard enough to expose the cliff, but not so hard that every board fails.
      </div>
    </div>
    <div class="panel">
      <h2>Selection rule</h2>
      <p><strong>Doable</strong> means nonzero exact solve rate. <strong>Hard</strong> means the next harder clean setting collapses. <strong>Clean</strong> means no Sudoku unit-memory training mechanism.</p>
      <p>16x16 unit-memory and 25x25 unit-memory remain useful diagnostics, but they are not the clean FutureSeed+loop mainline.</p>
    </div>
  </section>

  <section class="cards">
    <div class="card"><div class="kicker">h72 loop5 exact</div><div class="big">{pct(h72['loop_final_exact'])}</div><div class="small">thin but nonzero</div></div>
    <div class="card"><div class="kicker">h72 loop5 blank acc</div><div class="big">{pct(h72['loop_final_blank_acc'])}</div><div class="small">local cells mostly right</div></div>
    <div class="card"><div class="kicker">loop exact gain</div><div class="big">{pct(loop5['exact'] - loop1['exact'])}</div><div class="small">loop1 exact is zero</div></div>
    <div class="card"><div class="kicker">exact / independent</div><div class="big">{h72['exact_over_independent']:.1f}x</div><div class="small">success is not random per-cell luck</div></div>
  </section>

  <section class="charts">
    {loop_chart}
    {hole_chart}
  </section>

  <section class="twocol">
    <div>
      <h2>Loop curve at h72</h2>
      {h72_loop_table}
    </div>
    <div class="panel">
      <h2>Plain-language mechanism read</h2>
      <p>Loop1 to loop2 is the big local-repair jump: blank accuracy moves from {pct(loop1['blank_acc'])} to {pct(loop2['blank_acc'])}. That still barely solves full boards.</p>
      <p>Loop2 to loop5 adds little blank accuracy, but exact keeps moving upward. That is the more interesting part: late loops are not just making more cells right; they are sometimes making the board coherent.</p>
      <p>Failure after loop5 is therefore a whole-board coordination problem. A few coupled errors can keep exact near zero even when most hidden cells are correct.</p>
    </div>
  </section>

  <section style="margin-top: 18px;">
    <h2>Difficulty ladder</h2>
    {difficulty_table}
  </section>

  <section class="twocol">
    <div class="panel">
      <h2>FutureSeed reading</h2>
      <p>The h72 archive reports FutureSeed gate mean {fs_gate:.3f} and seed-state norm {fs_norm:.3f}. That tells us the global seed is active, but it is not enough by itself: loop1 exact is {pct(loop1['exact'])}.</p>
      <p>The useful story is FutureSeed plus recurrence. FutureSeed gives a noncausal global initialization; loop reuses it and iteratively cleans the board. The present archive does not store per-token seed influence, so the next visualization needs richer traces.</p>
    </div>
    <div class="panel">
      <h2>Concrete cases</h2>
      <p>{clean_case_link}</p>
      <p class="small">This is the concrete case exported by the 12x12 frontier run. It is from the same model, but the archive did not save a full h72 case bank.</p>
      <p style="margin-top:10px;">{casebank_link}</p>
      <p class="small">This reproduces the frontier and adds 4 solved-by-loop, 4 almost-solved, and 4 hard-failure h72 boards with per-loop wrong cells, entropy, changed cells, and duplicate conflicts.</p>
      <p style="margin-top:10px;">{failure_case_link}</p>
      <p class="small">This helps explain why clean 16x16 is not yet the case-level target: the current clean run has no exact successes.</p>
      <div class="callout" style="margin-top:12px;">
        <strong>Case-bank read:</strong> solved boards usually go from about 29-32 wrong hidden cells at loop1 to 0-3 by loop3, then exact by loop5. Failures often stall with 1-5 coupled wrong cells.
      </div>
    </div>
  </section>

  <section class="panel" style="margin-top: 18px;">
    <h2>Next read</h2>
    <p>The h72 case bank is now available. The next useful analysis is not another training variant; it is reading the case groups for what actually changes between loops.</p>
    <ul class="list">
      <li>Solved-by-loop: identify what kinds of conflicts disappear between loop2 and loop5.</li>
      <li>Almost-solved: inspect why a single wrong hidden cell remains sticky.</li>
      <li>Hard failures: check whether the remaining 5 wrong cells form one coupled region or multiple independent mistakes.</li>
      <li>FutureSeed tracing is still missing at per-token/layer granularity; that is the next instrumentation gap.</li>
    </ul>
  </section>
</main>
</body>
</html>
"""


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    summary = build_summary()
    (OUT_DIR / "frontier_selection.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    (OUT_DIR / "README.md").write_text(build_readme(summary))
    (OUT_DIR / "index.html").write_text(build_html(summary))
    print(f"wrote {OUT_DIR.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
