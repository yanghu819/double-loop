#!/usr/bin/env python3
from __future__ import annotations

import argparse
import html
import json
import re
from html.parser import HTMLParser
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np


class CasePanelParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.panels: List[Dict[str, Any]] = []
        self._in_section = False
        self._section_depth = 0
        self._current: Dict[str, Any] | None = None
        self._capture_h3 = False
        self._capture_p = False
        self._capture_cell = False
        self._current_cell: Dict[str, str] | None = None

    def handle_starttag(self, tag: str, attrs: List[Tuple[str, str | None]]) -> None:
        attrs_d = {k: v or "" for k, v in attrs}
        if tag == "section" and "case-panel" in attrs_d.get("class", ""):
            self._in_section = True
            self._section_depth = 1
            self._current = {"title": "", "subtitle": "", "cells": []}
            return
        if not self._in_section:
            return
        if tag == "section":
            self._section_depth += 1
        if tag == "h3":
            self._capture_h3 = True
        elif tag == "p" and not self._current.get("subtitle"):
            self._capture_p = True
        elif tag == "div" and "cell" in attrs_d.get("class", ""):
            self._capture_cell = True
            self._current_cell = {
                "class": attrs_d.get("class", ""),
                "title": attrs_d.get("title", ""),
                "text": "",
            }

    def handle_endtag(self, tag: str) -> None:
        if not self._in_section:
            return
        if tag == "h3":
            self._capture_h3 = False
        elif tag == "p":
            self._capture_p = False
        elif tag == "div" and self._capture_cell and self._current_cell is not None:
            assert self._current is not None
            self._current["cells"].append(self._current_cell)
            self._current_cell = None
            self._capture_cell = False
        elif tag == "section":
            self._section_depth -= 1
            if self._section_depth <= 0:
                assert self._current is not None
                self.panels.append(self._current)
                self._current = None
                self._in_section = False

    def handle_data(self, data: str) -> None:
        if not self._in_section or self._current is None:
            return
        if self._capture_h3:
            self._current["title"] += data.strip()
        elif self._capture_p:
            self._current["subtitle"] += data.strip()
        elif self._capture_cell and self._current_cell is not None:
            self._current_cell["text"] += data.strip()


def parse_gdn_html(path: Path) -> Dict[str, Any]:
    parser = CasePanelParser()
    parser.feed(path.read_text(encoding="utf-8"))
    panels = parser.panels
    if not panels:
        raise ValueError(f"no case panels found in {path}")
    boards: Dict[str, List[int]] = {}
    cell_classes: Dict[str, List[str]] = {}
    cell_titles: Dict[str, List[str]] = {}
    for panel in panels:
        title = panel["title"]
        values: List[int] = []
        classes: List[str] = []
        titles: List[str] = []
        for cell in panel["cells"]:
            text = cell["text"]
            values.append(0 if text in ("", ".") else int(text))
            classes.append(cell["class"])
            titles.append(cell["title"])
        if len(values) != 81:
            raise ValueError(f"panel {title!r} has {len(values)} cells, expected 81")
        key = title.lower().replace(" ", "")
        boards[key] = values
        cell_classes[key] = classes
        cell_titles[key] = titles
    loop_boards = {
        re.sub(r"^loop", "", key): value
        for key, value in boards.items()
        if key.startswith("loop")
    }
    loop_classes = {
        re.sub(r"^loop", "", key): value
        for key, value in cell_classes.items()
        if key.startswith("loop")
    }
    return {
        "source_html": str(path),
        "puzzle": boards["puzzle"],
        "solution": boards["solution"],
        "predictions": loop_boards,
        "cell_classes": loop_classes,
        "cell_titles": {
            re.sub(r"^loop", "", key): value
            for key, value in cell_titles.items()
            if key.startswith("loop")
        },
    }


def conflict_units(board: Iterable[int]) -> List[str]:
    grid = np.asarray(list(board), dtype=np.int64).reshape(9, 9)
    units: List[tuple[str, np.ndarray]] = []
    for i in range(9):
        units.append((f"row {i + 1}", grid[i, :]))
        units.append((f"col {i + 1}", grid[:, i]))
    for br in range(3):
        for bc in range(3):
            units.append((f"box {br + 1},{bc + 1}", grid[br * 3 : br * 3 + 3, bc * 3 : bc * 3 + 3].reshape(-1)))
    out: List[str] = []
    for name, vals in units:
        bad = []
        for digit in range(1, 10):
            count = int((vals == digit).sum())
            if count > 1:
                bad.append(f"{digit}x{count}")
        if bad:
            out.append(f"{name}: {'; '.join(bad)}")
    return out


def metrics(pred: Iterable[int], solution: Iterable[int], puzzle: Iterable[int]) -> Dict[str, Any]:
    p = np.asarray(list(pred), dtype=np.int64)
    y = np.asarray(list(solution), dtype=np.int64)
    x = np.asarray(list(puzzle), dtype=np.int64)
    hidden = x == 0
    wrong_hidden = (p != y) & hidden
    changed_clues = (x != 0) & (p != x)
    conflicts = conflict_units(p)
    return {
        "hidden_count": int(hidden.sum()),
        "wrong_count": int(wrong_hidden.sum()),
        "clue_changed_count": int(changed_clues.sum()),
        "blank_acc": float(1.0 - wrong_hidden.sum() / max(int(hidden.sum()), 1)),
        "full_acc": float((p == y).mean()),
        "exact": bool(np.all(p == y)),
        "valid_board": bool(len(conflicts) == 0),
        "conflict_unit_count": int(len(conflicts)),
        "conflict_units": conflicts[:20],
    }


def board_string(board: Iterable[int], *, blank: str = ".") -> str:
    return "".join(blank if int(v) == 0 else str(int(v)) for v in board)


def render_board(
    board: Iterable[int],
    solution: Iterable[int] | None,
    puzzle: Iterable[int] | None,
    title: str,
    subtitle: str,
    *,
    class_hints: List[str] | None = None,
) -> str:
    values = list(int(v) for v in board)
    sol = None if solution is None else list(int(v) for v in solution)
    puz = None if puzzle is None else list(int(v) for v in puzzle)
    cells: List[str] = []
    for idx, val in enumerate(values):
        r, c = divmod(idx, 9)
        cls = ["cell"]
        title_bits = [f"R{r + 1}C{c + 1}"]
        if puz is not None and puz[idx] != 0:
            cls.append("clue")
            title_bits.append("clue")
        elif val == 0:
            cls.append("hole")
        elif sol is not None and val == sol[idx]:
            cls.append("correct")
        elif sol is not None and val != sol[idx]:
            cls.append("wrong")
            title_bits.append(f"truth={sol[idx]}")
        elif "solution" in title.lower():
            cls.append("solution")
        if class_hints is not None:
            hint = class_hints[idx]
            if "changed" in hint:
                cls.append("changed")
            if "conflict" in hint:
                cls.append("conflict")
        style = []
        if c in (2, 5):
            style.append("border-right:2px solid #0f172a")
        if c == 8:
            style.append("border-right:0")
        if r in (2, 5):
            style.append("border-bottom:2px solid #0f172a")
        if r == 8:
            style.append("border-bottom:0")
        cells.append(
            f"<div class=\"{' '.join(cls)}\" style=\"{';'.join(style)}\" "
            f"title=\"{html.escape(' '.join(title_bits))}\">{'.' if val == 0 else val}</div>"
        )
    return (
        "<section class='board-card'>"
        f"<h3>{html.escape(title)}</h3>"
        f"<p>{html.escape(subtitle)}</p>"
        "<div class='grid'>"
        + "".join(cells)
        + "</div></section>"
    )


def render_html(payload: Dict[str, Any]) -> str:
    puzzle = payload["puzzle"]
    solution = payload["solution"]
    eqr = payload["eqr"]
    gdn = payload["gdn"]
    shared_steps = payload["shared_steps"]
    eqr_steps = payload["eqr_only_steps"]

    style = """
body{margin:0;background:#f6f8fa;color:#24292f;font-family:ui-sans-serif,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;letter-spacing:0}
main{max-width:1600px;margin:0 auto;padding:24px}
h1{font-size:25px;margin:0 0 8px} h2{font-size:18px;margin:22px 0 10px} h3{font-size:14px;margin:0 0 5px}
p{margin:0;color:#57606a;font-size:13px;line-height:1.35}
code{font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;background:#eaeef2;border-radius:4px;padding:1px 4px}
.note{background:#fff;border:1px solid #d0d7de;border-left:5px solid #0969da;border-radius:8px;padding:10px 12px;margin:14px 0}
.legend{display:flex;gap:12px;flex-wrap:wrap;margin:12px 0 16px}.legend span{font-size:12px;background:#fff;border:1px solid #d0d7de;border-radius:999px;padding:4px 8px}
.row{display:grid;grid-template-columns:repeat(auto-fit,minmax(315px,1fr));gap:12px;align-items:start;margin-bottom:14px}
.compare-row{display:grid;grid-template-columns:72px minmax(315px,1fr) minmax(315px,1fr);gap:12px;align-items:start;margin:12px 0}
.loop-label{font-weight:800;color:#0f172a;padding-top:12px}
.board-card{background:#fff;border:1px solid #d0d7de;border-radius:8px;padding:10px}
.grid{display:grid;grid-template-columns:repeat(9,34px);grid-template-rows:repeat(9,34px);border:2px solid #0f172a;width:max-content;margin-top:9px}
.cell{box-sizing:border-box;display:flex;align-items:center;justify-content:center;border-right:1px solid #94a3b8;border-bottom:1px solid #94a3b8;font-weight:760;position:relative;width:34px;height:34px;font-size:13px}
.clue{background:#e2e8f0;color:#0f172a}.hole{background:#fff7ed;color:#9a3412}.solution{background:#e0f2fe;color:#0c4a6e}
.correct{background:#bbf7d0;color:#14532d}.wrong{background:#fecaca;color:#7f1d1d}
.conflict::after{content:"";position:absolute;inset:3px;border:3px solid #f59e0b;border-radius:5px;pointer-events:none}
.changed::before{content:"";position:absolute;left:7px;right:7px;bottom:4px;height:3px;background:#1f6feb;border-radius:4px}
table{border-collapse:collapse;width:100%;background:#fff;border:1px solid #d0d7de;border-radius:8px;overflow:hidden;margin:10px 0}
th,td{border-bottom:1px solid #d8dee4;text-align:left;padding:7px 8px;font-size:13px}th{background:#f0f3f6}.bad{color:#b42318}.good{color:#067647}
@media (max-width:800px){.compare-row{grid-template-columns:1fr}.loop-label{padding-top:0}.grid{grid-template-columns:repeat(9,30px);grid-template-rows:repeat(9,30px)}.cell{width:30px;height:30px}}
"""
    metric_rows = []
    all_metric_steps = sorted({*eqr["metrics"].keys(), *gdn["metrics"].keys()}, key=lambda x: int(x))
    for step in all_metric_steps:
        em = eqr["metrics"].get(step)
        gm = gdn["metrics"].get(step)
        metric_rows.append(
            "<tr>"
            f"<td>loop{html.escape(step)}</td>"
            f"<td>{fmt_metric(em)}</td>"
            f"<td>{fmt_metric(gm)}</td>"
            "</tr>"
        )

    rows = []
    for step in shared_steps:
        em = eqr["metrics"][step]
        gm = gdn["metrics"][step]
        rows.append(
            "<div class='compare-row'>"
            f"<div class='loop-label'>loop{html.escape(step)}</div>"
            + render_board(
                eqr["predictions"][step],
                solution,
                puzzle,
                "Official EqR",
                metric_line(em),
            )
            + render_board(
                gdn["predictions"][step],
                solution,
                puzzle,
                "GDN + native FutureSeed",
                metric_line(gm),
                class_hints=gdn.get("cell_classes", {}).get(step),
            )
            + "</div>"
        )
    if eqr_steps:
        rows.append("<h2>EqR Longer Rollout</h2><div class='row'>")
        for step in eqr_steps:
            rows.append(
                render_board(
                    eqr["predictions"][step],
                    solution,
                    puzzle,
                    f"Official EqR loop{step}",
                    metric_line(eqr["metrics"][step]),
                )
            )
        rows.append("</div>")

    notes = payload.get("notes", [])
    notes_html = "".join(f"<li>{html.escape(str(item))}</li>" for item in notes)
    return f"""<!doctype html>
<html><head><meta charset="utf-8"><title>EqR vs FutureSeed Sudoku Loop Case</title><style>{style}</style></head>
<body><main>
<h1>Same Hard Sudoku Case: Official EqR vs GDN + native FutureSeed</h1>
<p>Case source: <code>{html.escape(payload['case_source'])}</code>. Puzzle has {sum(1 for x in puzzle if x == 0)} hidden cells.</p>
<div class="note">This is one exact same Sudoku board. Red cells are wrong hidden predictions; orange outline means duplicate Sudoku conflicts; blue underline means our GDN prediction changed from the previous shown loop. No repair, search, selector, or oracle post-processing is used.</div>
<div class="legend"><span class="good">green = correct hidden</span><span class="bad">red = wrong hidden</span><span>gray = clue</span><span>orange outline = row/col/box conflict</span><span>blue underline = changed from previous shown loop</span></div>
<h2>Puzzle And Target</h2>
<div class="row">
{render_board(puzzle, None, puzzle, "Puzzle", board_string(puzzle))}
{render_board(solution, None, puzzle, "Target", board_string(solution, blank="0"))}
</div>
<h2>Metric Summary</h2>
<table><thead><tr><th>loop</th><th>Official EqR</th><th>GDN + native FutureSeed</th></tr></thead><tbody>{''.join(metric_rows)}</tbody></table>
<h2>Matched Loops</h2>
{''.join(rows)}
<h2>Read</h2>
<ul>{notes_html}</ul>
</main></body></html>"""


def metric_line(item: Dict[str, Any]) -> str:
    return (
        f"wrong={int(item['wrong_count'])}/{int(item['hidden_count'])}; "
        f"blank_acc={float(item['blank_acc']):.3f}; "
        f"valid={bool(item['valid_board'])}; conflicts={int(item['conflict_unit_count'])}"
    )


def fmt_metric(item: Dict[str, Any] | None) -> str:
    if item is None:
        return "not exported"
    exact = "exact" if item.get("exact") else "not exact"
    valid = "valid" if item.get("valid_board") else f"{int(item.get('conflict_unit_count', 0))} conflicts"
    return f"{metric_line(item)}; {exact}; {valid}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Build an HTML comparison for one Sudoku case: official EqR vs GDN FutureSeed.")
    parser.add_argument("--gdn-html", required=True)
    parser.add_argument("--gdn-cases-json", required=True)
    parser.add_argument("--eqr-json", required=True)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()

    gdn_html = Path(args.gdn_html).resolve()
    gdn_case_meta = json.loads(Path(args.gdn_cases_json).read_text(encoding="utf-8"))
    eqr = json.loads(Path(args.eqr_json).read_text(encoding="utf-8"))
    gdn = parse_gdn_html(gdn_html)

    selected = gdn_case_meta["selected"]["hard_failure"][0]
    gdn_metrics = {
        key.replace("loop", ""): value
        for key, value in selected["loops"].items()
    }
    gdn["metrics"] = {
        step: {
            "hidden_count": int(selected["holes"]),
            "wrong_count": int(m["wrong_count"]),
            "blank_acc": float(m["blank_acc"]),
            "exact": bool(m["exact"]),
            "valid_board": bool(m["valid_board"]),
            "conflict_unit_count": int(m["conflict_unit_count"]),
            "conflict_units": m.get("conflict_units", []),
        }
        for step, m in gdn_metrics.items()
    }

    eqr_norm = {
        "predictions": {str(k): v for k, v in eqr["predictions"].items()},
        "metrics": {str(k): v for k, v in eqr["metrics"].items()},
    }
    if gdn["puzzle"] != eqr["puzzle"] or gdn["solution"] != eqr["solution"]:
        raise ValueError("GDN case and EqR export are not the same puzzle/solution")

    shared_steps = sorted(set(gdn["predictions"]) & set(eqr_norm["predictions"]), key=lambda x: int(x))
    eqr_only = sorted(set(eqr_norm["predictions"]) - set(gdn["predictions"]), key=lambda x: int(x))
    payload = {
        "case_source": str(gdn_html),
        "puzzle": gdn["puzzle"],
        "solution": gdn["solution"],
        "gdn": {
            "predictions": gdn["predictions"],
            "metrics": gdn["metrics"],
            "cell_classes": gdn["cell_classes"],
        },
        "eqr": eqr_norm,
        "shared_steps": shared_steps,
        "eqr_only_steps": eqr_only,
        "notes": [
            "GDN+FutureSeed rapidly improves from 29 wrong hidden cells at loop1 to 6 at loop3, then stalls at the same 6 wrong cells through loop5.",
            "Official EqR is evaluated on the same puzzle with the official checkpoint and wrapper; longer rollout shows whether its recurrence keeps correcting after the matched early loops.",
            "The comparison is diagnostic rather than a training result table: the key question is whether later loops move global consistency, not whether a single selected case proves a leaderboard win.",
        ],
    }
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "comparison.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    (out_dir / "index.html").write_text(render_html(payload), encoding="utf-8")
    print(json.dumps({"out_dir": str(out_dir), "shared_steps": shared_steps, "eqr_only_steps": eqr_only}, indent=2))


if __name__ == "__main__":
    main()
