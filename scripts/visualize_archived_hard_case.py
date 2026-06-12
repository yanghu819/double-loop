#!/usr/bin/env python3
"""Build a readable visualization for an archived FutureSeed loop case.

This is an offline artifact parser. It does not import torch, load a model, or
run evaluation, so it preserves the project's GPU-only rule for model work.
"""

from __future__ import annotations

import argparse
import html
import json
import math
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from html.parser import HTMLParser
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CASE_HTML = (
    REPO_ROOT
    / "runs/h120curve-12x12-d256-l12-loop6-s9000-ckpt1k2k3k-20260612T0455Z-2e95bd8/output/futureseed_loop_case_seed52.html"
)
DEFAULT_METRICS_JSON = DEFAULT_CASE_HTML.with_name("futureseed_loop_seed52.json")
DEFAULT_OUT_HTML = DEFAULT_CASE_HTML.with_name("h120_hardest_case_visual.html")
DEFAULT_OUT_JSON = DEFAULT_CASE_HTML.with_name("h120_hardest_case_summary.json")
DEFAULT_OUT_PNG = DEFAULT_CASE_HTML.with_name("h120_hardest_case_visual.png")


@dataclass(frozen=True)
class ParsedCase:
    titles: list[str]
    boards: dict[str, list[str]]
    size: int
    box_rows: int
    box_cols: int


@dataclass(frozen=True)
class UnitConflict:
    unit: str
    repeated: dict[str, list[int]]
    missing: list[str]


@dataclass(frozen=True)
class LoopStats:
    name: str
    wrong_hidden: list[int]
    correct_hidden: int
    conflict_cells: set[int]
    conflicts: list[UnitConflict]
    changed_from_prev: list[int]
    fixed_from_prev: list[int]
    regressed_from_prev: list[int]


class CaseHTMLParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.titles: list[str] = []
        self.cells: list[str] = []
        self._in_h3 = False
        self._in_cell = False
        self._text = ""

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attrs_dict = dict(attrs)
        classes = str(attrs_dict.get("class", "")).split()
        if tag == "h3":
            self._in_h3 = True
            self._text = ""
        elif tag == "div" and "cell" in classes:
            self._in_cell = True
            self._text = ""

    def handle_data(self, data: str) -> None:
        if self._in_h3 or self._in_cell:
            self._text += data.strip()

    def handle_endtag(self, tag: str) -> None:
        if tag == "h3" and self._in_h3:
            self.titles.append(self._text.strip())
            self._in_h3 = False
            self._text = ""
        elif tag == "div" and self._in_cell:
            self.cells.append(self._text.strip() or ".")
            self._in_cell = False
            self._text = ""


def normalize_title(title: str) -> str:
    title = title.strip().lower()
    return title.replace(" ", "")


def infer_box_shape(size: int) -> tuple[int, int]:
    root = int(math.sqrt(size))
    if root * root == size:
        return root, root
    candidates = [value for value in range(root, 1, -1) if size % value == 0]
    if not candidates:
        raise ValueError(f"cannot infer Sudoku box shape for size={size}")
    rows = candidates[0]
    return rows, size // rows


def parse_case(path: Path) -> ParsedCase:
    parser = CaseHTMLParser()
    parser.feed(path.read_text(encoding="utf-8"))
    if len(parser.titles) < 3:
        raise ValueError(f"{path} does not look like a case HTML: no board titles")
    if len(parser.cells) % len(parser.titles) != 0:
        raise ValueError(
            f"{path} has {len(parser.cells)} cells and {len(parser.titles)} titles; cannot split boards"
        )
    cells_per_board = len(parser.cells) // len(parser.titles)
    size = int(math.sqrt(cells_per_board))
    if size * size != cells_per_board:
        raise ValueError(f"{path} has non-square board size: {cells_per_board} cells")
    boards = {
        normalize_title(title): parser.cells[i * cells_per_board : (i + 1) * cells_per_board]
        for i, title in enumerate(parser.titles)
    }
    if "puzzle" not in boards or "solution" not in boards:
        raise ValueError(f"{path} is missing puzzle or solution board")
    box_rows, box_cols = infer_box_shape(size)
    return ParsedCase(
        titles=[normalize_title(title) for title in parser.titles],
        boards=boards,
        size=size,
        box_rows=box_rows,
        box_cols=box_cols,
    )


def coord(idx: int, size: int) -> str:
    row, col = divmod(idx, size)
    return f"R{row + 1}C{col + 1}"


def unit_iter(size: int, box_rows: int, box_cols: int) -> list[tuple[str, list[int]]]:
    units: list[tuple[str, list[int]]] = []
    for row in range(size):
        units.append((f"row {row + 1}", [row * size + col for col in range(size)]))
    for col in range(size):
        units.append((f"col {col + 1}", [row * size + col for row in range(size)]))
    for br in range(size // box_rows):
        for bc in range(size // box_cols):
            idxs = []
            for dr in range(box_rows):
                for dc in range(box_cols):
                    idxs.append((br * box_rows + dr) * size + (bc * box_cols + dc))
            units.append((f"box {br + 1},{bc + 1}", idxs))
    return units


def analyze_loop(
    name: str,
    board: list[str],
    puzzle: list[str],
    solution: list[str],
    previous: list[str] | None,
    *,
    size: int,
    box_rows: int,
    box_cols: int,
) -> LoopStats:
    hidden = [idx for idx, value in enumerate(puzzle) if value == "."]
    wrong_hidden = [idx for idx in hidden if board[idx] != solution[idx]]
    correct_hidden = len(hidden) - len(wrong_hidden)
    conflicts: list[UnitConflict] = []
    conflict_cells: set[int] = set()
    symbols = {str(value) for value in range(1, size + 1)}
    for unit_name, idxs in unit_iter(size, box_rows, box_cols):
        counts: dict[str, list[int]] = {}
        for idx in idxs:
            value = board[idx]
            if value != ".":
                counts.setdefault(value, []).append(idx)
        repeated = {value: locs for value, locs in counts.items() if len(locs) > 1}
        missing = sorted(symbols - set(counts), key=lambda value: int(value))
        if repeated:
            for locs in repeated.values():
                conflict_cells.update(locs)
            conflicts.append(UnitConflict(unit=unit_name, repeated=repeated, missing=missing))

    if previous is None:
        changed_from_prev: list[int] = []
        fixed_from_prev: list[int] = []
        regressed_from_prev: list[int] = []
    else:
        changed_from_prev = [idx for idx in hidden if previous[idx] != board[idx]]
        fixed_from_prev = [
            idx for idx in changed_from_prev if previous[idx] != solution[idx] and board[idx] == solution[idx]
        ]
        regressed_from_prev = [
            idx for idx in changed_from_prev if previous[idx] == solution[idx] and board[idx] != solution[idx]
        ]
    return LoopStats(
        name=name,
        wrong_hidden=wrong_hidden,
        correct_hidden=correct_hidden,
        conflict_cells=conflict_cells,
        conflicts=conflicts,
        changed_from_prev=changed_from_prev,
        fixed_from_prev=fixed_from_prev,
        regressed_from_prev=regressed_from_prev,
    )


def loop_names(case: ParsedCase) -> list[str]:
    names = [name for name in case.titles if name.startswith("loop")]
    return sorted(names, key=lambda value: int(value.removeprefix("loop")))


def cell_classes(
    idx: int,
    board: list[str],
    puzzle: list[str],
    solution: list[str],
    stats: LoopStats | None,
    previous: list[str] | None,
    mode: str,
) -> list[str]:
    classes = ["cell"]
    if mode == "puzzle":
        classes.append("hole" if board[idx] == "." else "clue")
    elif mode == "solution":
        classes.append("solution")
        if puzzle[idx] != ".":
            classes.append("given-solution")
    elif puzzle[idx] != ".":
        classes.append("clue")
    elif board[idx] == solution[idx]:
        classes.append("correct")
    else:
        classes.append("wrong")
    if stats is not None and idx in stats.conflict_cells:
        classes.append("conflict")
    if previous is not None and puzzle[idx] == "." and previous[idx] != board[idx]:
        classes.append("changed")
    if previous is not None and puzzle[idx] == "." and previous[idx] != solution[idx] and board[idx] == solution[idx]:
        classes.append("fixed")
    if previous is not None and puzzle[idx] == "." and previous[idx] == solution[idx] and board[idx] != solution[idx]:
        classes.append("regressed")
    return classes


def border_style(idx: int, size: int, box_rows: int, box_cols: int) -> str:
    row, col = divmod(idx, size)
    styles: list[str] = []
    if col == size - 1:
        styles.append("border-right: 0")
    elif (col + 1) % box_cols == 0:
        styles.append("border-right: 2px solid #111827")
    if row == size - 1:
        styles.append("border-bottom: 0")
    elif (row + 1) % box_rows == 0:
        styles.append("border-bottom: 2px solid #111827")
    return "; ".join(styles)


def render_board(
    name: str,
    board: list[str],
    puzzle: list[str],
    solution: list[str],
    stats: LoopStats | None,
    previous: list[str] | None,
    *,
    size: int,
    box_rows: int,
    box_cols: int,
    mode: str,
) -> str:
    cells: list[str] = []
    for idx, value in enumerate(board):
        title_bits = [coord(idx, size), f"value={value}"]
        if mode == "prediction" and puzzle[idx] == ".":
            title_bits.append(f"truth={solution[idx]}")
        if previous is not None and puzzle[idx] == "." and previous[idx] != value:
            title_bits.append(f"prev={previous[idx]}")
        if stats is not None and idx in stats.conflict_cells:
            title_bits.append("duplicate-conflict")
        style = border_style(idx, size, box_rows, box_cols)
        style_attr = f' style="{style}"' if style else ""
        cells.append(
            '<div class="{}"{} title="{}">{}</div>'.format(
                " ".join(cell_classes(idx, board, puzzle, solution, stats, previous, mode)),
                style_attr,
                html.escape("; ".join(title_bits)),
                html.escape(value),
            )
        )
    return f"""
<section class="board-card">
  <h3>{html.escape(name)}</h3>
  {render_grid_metrics(name, stats, puzzle) if stats is not None else ""}
  <div class="sudoku-grid" style="--n:{size}; --cell:{30 if size <= 12 else 24}px;">{''.join(cells)}</div>
</section>
"""


def render_grid_metrics(name: str, stats: LoopStats | None, puzzle: list[str]) -> str:
    if stats is None:
        return ""
    hidden = sum(1 for value in puzzle if value == ".")
    acc = stats.correct_hidden / max(hidden, 1)
    if name == "loop1":
        delta = "first pass"
    else:
        delta = (
            f"changed {len(stats.changed_from_prev)}, fixed {len(stats.fixed_from_prev)}, "
            f"regressed {len(stats.regressed_from_prev)}"
        )
    return (
        f'<p class="card-meta">{stats.correct_hidden}/{hidden} hidden correct '
        f'({acc * 100:.1f}%), wrong {len(stats.wrong_hidden)}, '
        f'conflict units {len(stats.conflicts)}, {html.escape(delta)}</p>'
    )


def pct(value: float) -> str:
    return f"{100.0 * value:.2f}%"


def load_metrics(path: Path | None) -> dict[str, Any]:
    if path is None or not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def metric_loop_curve(metrics: dict[str, Any]) -> list[dict[str, Any]]:
    eval_clean = metrics.get("metrics", {}).get("eval_clean", {})
    rows = []
    for key, value in eval_clean.items():
        if not key.startswith("loop") or "/" in key:
            continue
        try:
            loop_idx = int(key.removeprefix("loop"))
        except ValueError:
            continue
        rows.append(
            {
                "loop": loop_idx,
                "exact": float(value.get("label_exact", 0.0)),
                "blank_acc": float(value.get("blank_acc", 0.0)),
                "valid": float(value.get("valid_sudoku", 0.0)),
            }
        )
    return sorted(rows, key=lambda row: row["loop"])


def render_metric_table(curve: list[dict[str, Any]]) -> str:
    if not curve:
        return ""
    rows = []
    for item in curve:
        rows.append(
            "<tr><td>loop {loop}</td><td>{exact}</td><td>{blank_acc}</td><td>{valid}</td></tr>".format(
                loop=item["loop"],
                exact=pct(item["exact"]),
                blank_acc=pct(item["blank_acc"]),
                valid=pct(item["valid"]),
            )
        )
    return (
        '<table class="metric-table"><thead><tr><th>eval loop</th><th>exact</th>'
        '<th>blank acc</th><th>valid Sudoku</th></tr></thead><tbody>'
        + "".join(rows)
        + "</tbody></table>"
    )


def render_conflict_table(stats: LoopStats, size: int, limit: int = 16) -> str:
    rows = []
    for conflict in stats.conflicts[:limit]:
        repeated_bits = []
        for value, locs in sorted(conflict.repeated.items(), key=lambda item: int(item[0])):
            repeated_bits.append(f"{value}: " + ", ".join(coord(idx, size) for idx in locs))
        rows.append(
            "<tr><td>{}</td><td>{}</td><td>{}</td></tr>".format(
                html.escape(conflict.unit),
                html.escape("; ".join(repeated_bits)),
                html.escape(", ".join(conflict.missing[:8])),
            )
        )
    if not rows:
        rows.append('<tr><td colspan="3">No duplicate conflicts.</td></tr>')
    more = "" if len(stats.conflicts) <= limit else f"<p class=\"small\">Showing {limit} of {len(stats.conflicts)} conflict units.</p>"
    return (
        '<table class="metric-table"><thead><tr><th>unit</th><th>duplicates</th><th>missing symbols</th></tr></thead><tbody>'
        + "".join(rows)
        + "</tbody></table>"
        + more
    )


def aggregate_wrong_units(
    wrong_hidden: list[int],
    *,
    size: int,
    box_rows: int,
    box_cols: int,
) -> dict[str, list[tuple[str, int]]]:
    wrong_set = set(wrong_hidden)
    buckets: dict[str, list[tuple[str, int]]] = {"rows": [], "cols": [], "boxes": []}
    for row in range(size):
        count = sum(1 for col in range(size) if row * size + col in wrong_set)
        if count:
            buckets["rows"].append((f"R{row + 1}", count))
    for col in range(size):
        count = sum(1 for row in range(size) if row * size + col in wrong_set)
        if count:
            buckets["cols"].append((f"C{col + 1}", count))
    for br in range(size // box_rows):
        for bc in range(size // box_cols):
            count = 0
            for dr in range(box_rows):
                for dc in range(box_cols):
                    idx = (br * box_rows + dr) * size + (bc * box_cols + dc)
                    count += int(idx in wrong_set)
            if count:
                buckets["boxes"].append((f"B{br + 1},{bc + 1}", count))
    for key in buckets:
        buckets[key].sort(key=lambda item: item[1], reverse=True)
    return buckets


def render_wrong_summary(stats: LoopStats, case: ParsedCase, limit: int = 8) -> str:
    buckets = aggregate_wrong_units(
        stats.wrong_hidden,
        size=case.size,
        box_rows=case.box_rows,
        box_cols=case.box_cols,
    )
    chunks = []
    for key, values in buckets.items():
        top = values[:limit]
        chunks.append(
            '<div class="mini-list"><h4>{}</h4><p>{}</p></div>'.format(
                html.escape(key),
                html.escape(", ".join(f"{name}:{count}" for name, count in top) or "none"),
            )
        )
    return '<div class="mini-list-row">' + "".join(chunks) + "</div>"


def render_change_summary(stats_by_loop: dict[str, LoopStats], names: list[str]) -> str:
    rows = []
    for name in names:
        stats = stats_by_loop[name]
        rows.append(
            "<tr><td>{}</td><td>{}</td><td>{}</td><td>{}</td><td>{}</td><td>{}</td></tr>".format(
                html.escape(name),
                stats.correct_hidden,
                len(stats.wrong_hidden),
                len(stats.changed_from_prev),
                len(stats.fixed_from_prev),
                len(stats.regressed_from_prev),
            )
        )
    return (
        '<table class="metric-table"><thead><tr><th>loop</th><th>hidden correct</th>'
        '<th>hidden wrong</th><th>changed vs prev</th><th>fixed vs prev</th>'
        '<th>regressed vs prev</th></tr></thead><tbody>'
        + "".join(rows)
        + "</tbody></table>"
    )


def render_token_hist(board: list[str], puzzle: list[str], size: int) -> str:
    hidden_values = [board[idx] for idx, value in enumerate(puzzle) if value == "."]
    counts = Counter(hidden_values)
    max_count = max(counts.values()) if counts else 1
    bars = []
    for symbol in [str(value) for value in range(1, size + 1)]:
        count = counts.get(symbol, 0)
        width = 100.0 * count / max_count
        bars.append(
            f'<div class="hist-row"><span>{symbol}</span><div><i style="width:{width:.1f}%"></i></div><b>{count}</b></div>'
        )
    return '<div class="histogram">' + "".join(bars) + "</div>"


def load_font(size: int, *, bold: bool = False) -> Any:
    try:
        from PIL import ImageFont

        candidates = [
            "/System/Library/Fonts/Supplemental/Arial Bold.ttf" if bold else "/System/Library/Fonts/Supplemental/Arial.ttf",
            "/System/Library/Fonts/Supplemental/Helvetica Bold.ttf" if bold else "/System/Library/Fonts/Supplemental/Helvetica.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        ]
        for path in candidates:
            if Path(path).exists():
                return ImageFont.truetype(path, size=size)
        return ImageFont.load_default()
    except Exception:
        return None


def text_size(draw: Any, text: str, font: Any) -> tuple[int, int]:
    box = draw.textbbox((0, 0), text, font=font)
    return int(box[2] - box[0]), int(box[3] - box[1])


def draw_text_center(draw: Any, xy: tuple[int, int, int, int], text: str, font: Any, fill: str) -> None:
    x0, y0, x1, y1 = xy
    width, height = text_size(draw, text, font)
    draw.text((x0 + (x1 - x0 - width) / 2, y0 + (y1 - y0 - height) / 2 - 1), text, font=font, fill=fill)


def draw_board_png(
    draw: Any,
    x: int,
    y: int,
    title: str,
    board: list[str],
    puzzle: list[str],
    solution: list[str],
    stats: LoopStats | None,
    previous: list[str] | None,
    *,
    size: int,
    box_rows: int,
    box_cols: int,
    cell: int,
    title_font: Any,
    meta_font: Any,
    cell_font: Any,
    mode: str,
) -> None:
    draw.text((x, y), title, font=title_font, fill="#111827")
    meta_y = y + 24
    if stats is not None:
        hidden = sum(1 for value in puzzle if value == ".")
        meta = (
            f"{stats.correct_hidden}/{hidden} correct, "
            f"{len(stats.wrong_hidden)} wrong, {len(stats.conflicts)} conflicts"
        )
        if previous is not None:
            meta += f", changed {len(stats.changed_from_prev)}"
        draw.text((x, meta_y), meta, font=meta_font, fill="#5b6472")
    grid_y = y + 48
    palette = {
        "clue": "#e5e7eb",
        "hole": "#f8fafc",
        "solution": "#eff6ff",
        "correct": "#b7f7cb",
        "wrong": "#fecaca",
    }
    for idx, value in enumerate(board):
        row, col = divmod(idx, size)
        cx = x + col * cell
        cy = grid_y + row * cell
        if mode == "puzzle":
            color = palette["hole"] if value == "." else palette["clue"]
            fill = "#94a3b8" if value == "." else "#111827"
        elif mode == "solution":
            color = "#dbeafe" if puzzle[idx] != "." else palette["solution"]
            fill = "#1e3a8a"
        elif puzzle[idx] != ".":
            color = palette["clue"]
            fill = "#111827"
        elif board[idx] == solution[idx]:
            color = palette["correct"]
            fill = "#064e3b"
        else:
            color = palette["wrong"]
            fill = "#7f1d1d"
        draw.rectangle((cx, cy, cx + cell, cy + cell), fill=color, outline="#9ca3af", width=1)
        if stats is not None and idx in stats.conflict_cells:
            draw.rectangle((cx + 2, cy + 2, cx + cell - 2, cy + cell - 2), outline="#f59e0b", width=2)
        if previous is not None and puzzle[idx] == "." and previous[idx] != value:
            draw.rectangle((cx + 4, cy + cell - 5, cx + cell - 4, cy + cell - 3), fill="#2563eb")
        if previous is not None and puzzle[idx] == "." and previous[idx] != solution[idx] and board[idx] == solution[idx]:
            draw.rectangle((cx + 4, cy + 4, cx + cell - 4, cy + cell - 4), outline="#16a34a", width=2)
        if previous is not None and puzzle[idx] == "." and previous[idx] == solution[idx] and board[idx] != solution[idx]:
            draw.rectangle((cx + 4, cy + 4, cx + cell - 4, cy + cell - 4), outline="#dc2626", width=2)
        draw_text_center(draw, (cx, cy, cx + cell, cy + cell), value, cell_font, fill)

    grid_w = size * cell
    grid_h = size * cell
    draw.rectangle((x, grid_y, x + grid_w, grid_y + grid_h), outline="#111827", width=2)
    for col in range(box_cols, size, box_cols):
        px = x + col * cell
        draw.line((px, grid_y, px, grid_y + grid_h), fill="#111827", width=2)
    for row in range(box_rows, size, box_rows):
        py = grid_y + row * cell
        draw.line((x, py, x + grid_w, py), fill="#111827", width=2)


def make_png(
    case: ParsedCase,
    stats_by_loop: dict[str, LoopStats],
    names: list[str],
    *,
    out_png: Path,
) -> None:
    try:
        from PIL import Image, ImageDraw
    except Exception as exc:
        print(f"PNG skipped: PIL unavailable: {exc}")
        return

    puzzle = case.boards["puzzle"]
    solution = case.boards["solution"]
    hidden = sum(1 for value in puzzle if value == ".")
    final = stats_by_loop[names[-1]]
    cell = 30 if case.size <= 12 else 24
    board_w = case.size * cell
    card_w = board_w + 34
    card_h = case.size * cell + 76
    gap = 24
    cols = 4
    header_h = 190
    rows = math.ceil((2 + len(names)) / cols)
    width = cols * card_w + (cols + 1) * gap
    height = header_h + rows * card_h + (rows + 1) * gap
    image = Image.new("RGB", (width, height), "#f5f7fb")
    draw = ImageDraw.Draw(image)
    title_font = load_font(28, bold=True)
    subtitle_font = load_font(16)
    stat_font = load_font(18, bold=True)
    small_font = load_font(13)
    board_title_font = load_font(16, bold=True)
    meta_font = load_font(12)
    cell_font = load_font(15, bold=True)

    draw.text((gap, 22), "Hardest archived h120 case: 12x12 / 120 hidden cells", font=title_font, fill="#111827")
    subtitle = (
        "Green=correct, red=wrong, orange=duplicate conflict, blue=changed from previous loop. "
        "The late loops make few edits and leave row/column/box conflicts."
    )
    draw.text((gap, 62), subtitle, font=subtitle_font, fill="#5b6472")
    stat_items = [
        ("hidden", str(hidden)),
        ("clues", str(case.size * case.size - hidden)),
        ("loop6 wrong", str(len(final.wrong_hidden))),
        ("loop6 conflicts", str(len(final.conflicts))),
        ("loop1 correct", str(stats_by_loop[names[0]].correct_hidden)),
        ("loop6 correct", str(final.correct_hidden)),
    ]
    sx = gap
    for label, value in stat_items:
        draw.rounded_rectangle((sx, 104, sx + 176, 158), radius=8, fill="#eef2ff", outline="#c7d2fe", width=1)
        draw.text((sx + 12, 112), value, font=stat_font, fill="#111827")
        draw.text((sx + 12, 136), label, font=small_font, fill="#5b6472")
        sx += 190

    specs: list[tuple[str, list[str], LoopStats | None, list[str] | None, str]] = [
        ("puzzle", puzzle, None, None, "puzzle"),
        ("solution", solution, None, None, "solution"),
    ]
    for idx, name in enumerate(names):
        prev_name = names[idx - 1] if idx else None
        specs.append((name, case.boards[name], stats_by_loop[name], case.boards[prev_name] if prev_name else None, "prediction"))

    for i, (title, board, stats, previous, mode) in enumerate(specs):
        row = i // cols
        col = i % cols
        x = gap + col * (card_w + gap)
        y = header_h + gap + row * (card_h + gap)
        draw.rounded_rectangle((x - 12, y - 12, x + card_w - 12, y + card_h - 12), radius=8, fill="#ffffff", outline="#c9d3e2", width=1)
        draw_board_png(
            draw,
            x,
            y,
            title,
            board,
            puzzle,
            solution,
            stats,
            previous,
            size=case.size,
            box_rows=case.box_rows,
            box_cols=case.box_cols,
            cell=cell,
            title_font=board_title_font,
            meta_font=meta_font,
            cell_font=cell_font,
            mode=mode,
        )

    out_png.parent.mkdir(parents=True, exist_ok=True)
    image.save(out_png)


def build_summary_json(
    case: ParsedCase,
    stats_by_loop: dict[str, LoopStats],
    names: list[str],
    metrics: dict[str, Any],
    *,
    case_html: Path,
) -> dict[str, Any]:
    puzzle = case.boards["puzzle"]
    solution = case.boards["solution"]
    final = stats_by_loop[names[-1]]
    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_case_html": str(case_html.relative_to(REPO_ROOT)),
        "size": case.size,
        "box_rows": case.box_rows,
        "box_cols": case.box_cols,
        "hidden_cells": sum(1 for value in puzzle if value == "."),
        "clues": sum(1 for value in puzzle if value != "."),
        "loop_summary": [
            {
                "loop": name,
                "hidden_correct": stats_by_loop[name].correct_hidden,
                "hidden_wrong": len(stats_by_loop[name].wrong_hidden),
                "changed_from_prev": len(stats_by_loop[name].changed_from_prev),
                "fixed_from_prev": len(stats_by_loop[name].fixed_from_prev),
                "regressed_from_prev": len(stats_by_loop[name].regressed_from_prev),
                "conflict_units": len(stats_by_loop[name].conflicts),
                "conflict_cells": len(stats_by_loop[name].conflict_cells),
            }
            for name in names
        ],
        "final_wrong_cells": [
            {
                "coord": coord(idx, case.size),
                "pred": case.boards[names[-1]][idx],
                "truth": solution[idx],
            }
            for idx in final.wrong_hidden
        ],
        "eval_loop_curve": metric_loop_curve(metrics),
    }


def make_html(
    case: ParsedCase,
    stats_by_loop: dict[str, LoopStats],
    names: list[str],
    metrics: dict[str, Any],
    *,
    case_html: Path,
    out_html: Path,
) -> None:
    puzzle = case.boards["puzzle"]
    solution = case.boards["solution"]
    hidden = sum(1 for value in puzzle if value == ".")
    clues = case.size * case.size - hidden
    final_name = names[-1]
    final_stats = stats_by_loop[final_name]
    curve = metric_loop_curve(metrics)
    fs_state = metrics.get("metrics", {}).get("eval_clean", {}).get("loop1/future_seed", {})

    board_sections = [
        render_board(
            "puzzle",
            puzzle,
            puzzle,
            solution,
            None,
            None,
            size=case.size,
            box_rows=case.box_rows,
            box_cols=case.box_cols,
            mode="puzzle",
        ),
        render_board(
            "solution",
            solution,
            puzzle,
            solution,
            None,
            None,
            size=case.size,
            box_rows=case.box_rows,
            box_cols=case.box_cols,
            mode="solution",
        ),
    ]
    for idx, name in enumerate(names):
        prev_name = names[idx - 1] if idx else None
        board_sections.append(
            render_board(
                name,
                case.boards[name],
                puzzle,
                solution,
                stats_by_loop[name],
                case.boards[prev_name] if prev_name else None,
                size=case.size,
                box_rows=case.box_rows,
                box_cols=case.box_cols,
                mode="prediction",
            )
        )

    fs_bits = []
    if fs_state:
        fs_bits = [
            f"future-seed gate mean: {float(fs_state.get('fs_gate_mean', 0.0)):.3f}",
            f"state norm: {float(fs_state.get('fs_state_norm', 0.0)):.3f}",
            f"decay: {float(fs_state.get('fs_decay', 0.0)):.3f}",
            f"update mean: {float(fs_state.get('fs_update_mean', 0.0)):.3f}",
        ]

    out_html.write_text(
        f"""<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>h120 hardest case visual</title>
<style>
:root {{
  --bg: #f5f7fb;
  --ink: #111827;
  --muted: #5b6472;
  --panel: #ffffff;
  --line: #c9d3e2;
  --clue: #e5e7eb;
  --hole: #f8fafc;
  --correct: #b7f7cb;
  --wrong: #fecaca;
  --conflict: #f59e0b;
  --changed: #2563eb;
  --fixed: #16a34a;
  --regressed: #dc2626;
}}
* {{ box-sizing: border-box; }}
body {{
  margin: 0;
  background: var(--bg);
  color: var(--ink);
  font-family: ui-sans-serif, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
}}
main {{ max-width: 1480px; margin: 0 auto; padding: 24px; }}
h1 {{ margin: 0 0 8px; font-size: 28px; letter-spacing: 0; }}
h2 {{ margin: 28px 0 12px; font-size: 19px; }}
h3 {{ margin: 0 0 8px; font-size: 15px; }}
h4 {{ margin: 0 0 4px; font-size: 13px; }}
p {{ line-height: 1.55; }}
.lead {{ max-width: 940px; color: var(--muted); margin: 0 0 18px; }}
.hero {{
  background: var(--panel);
  border: 1px solid var(--line);
  border-radius: 8px;
  padding: 16px;
}}
.stat-row {{ display: flex; flex-wrap: wrap; gap: 10px; margin: 14px 0 4px; }}
.stat {{
  background: #eef2ff;
  border: 1px solid #c7d2fe;
  border-radius: 8px;
  padding: 10px 12px;
  min-width: 148px;
}}
.stat b {{ display: block; font-size: 20px; }}
.stat span {{ display: block; color: var(--muted); font-size: 12px; }}
.legend {{ display: flex; flex-wrap: wrap; gap: 10px; margin-top: 12px; color: var(--muted); font-size: 13px; }}
.legend i {{ display: inline-block; width: 14px; height: 14px; border-radius: 3px; margin-right: 5px; vertical-align: -2px; border: 1px solid rgba(17, 24, 39, .18); }}
.boards {{ display: flex; flex-wrap: wrap; gap: 14px; align-items: flex-start; }}
.board-card {{
  background: var(--panel);
  border: 1px solid var(--line);
  border-radius: 8px;
  padding: 12px;
}}
.card-meta {{ margin: -2px 0 9px; color: var(--muted); font-size: 12px; max-width: 390px; }}
.sudoku-grid {{
  display: grid;
  grid-template-columns: repeat(var(--n), var(--cell));
  grid-auto-rows: var(--cell);
  border: 2px solid #111827;
  width: max-content;
}}
.cell {{
  display: flex;
  align-items: center;
  justify-content: center;
  border-right: 1px solid #9ca3af;
  border-bottom: 1px solid #9ca3af;
  font-weight: 760;
  font-size: 13px;
  position: relative;
}}
.clue {{ background: var(--clue); color: #111827; }}
.hole {{ background: var(--hole); color: #94a3b8; }}
.solution {{ background: #eff6ff; color: #1e3a8a; }}
.given-solution {{ background: #dbeafe; }}
.correct {{ background: var(--correct); color: #064e3b; }}
.wrong {{ background: var(--wrong); color: #7f1d1d; }}
.conflict::after {{
  content: "";
  position: absolute;
  inset: 2px;
  border: 2px solid var(--conflict);
  pointer-events: none;
}}
.changed::before {{
  content: "";
  position: absolute;
  left: 4px;
  right: 4px;
  bottom: 2px;
  height: 3px;
  background: var(--changed);
}}
.fixed {{ box-shadow: inset 0 0 0 3px rgba(22, 163, 74, .55); }}
.regressed {{ box-shadow: inset 0 0 0 3px rgba(220, 38, 38, .55); }}
.metric-table {{
  border-collapse: collapse;
  background: var(--panel);
  border: 1px solid var(--line);
  width: 100%;
  overflow: hidden;
  border-radius: 8px;
}}
.metric-table th, .metric-table td {{
  border-bottom: 1px solid var(--line);
  padding: 8px 10px;
  text-align: left;
  font-size: 13px;
  vertical-align: top;
}}
.metric-table th {{ background: #eef2f7; }}
.metric-table tr:last-child td {{ border-bottom: 0; }}
.analysis-grid {{ display: grid; grid-template-columns: minmax(0, 1.15fr) minmax(300px, .85fr); gap: 14px; }}
.mini-list-row {{ display: grid; grid-template-columns: repeat(3, 1fr); gap: 10px; margin: 10px 0 14px; }}
.mini-list {{
  background: var(--panel);
  border: 1px solid var(--line);
  border-radius: 8px;
  padding: 10px;
}}
.mini-list p {{ margin: 0; color: var(--muted); font-size: 13px; }}
.histogram {{
  background: var(--panel);
  border: 1px solid var(--line);
  border-radius: 8px;
  padding: 10px;
}}
.hist-row {{
  display: grid;
  grid-template-columns: 24px 1fr 28px;
  align-items: center;
  gap: 8px;
  font-size: 12px;
  margin: 4px 0;
}}
.hist-row div {{ height: 9px; background: #e5e7eb; border-radius: 4px; overflow: hidden; }}
.hist-row i {{ display: block; height: 100%; background: #64748b; }}
.hist-row b {{ font-weight: 650; text-align: right; }}
.small {{ color: var(--muted); font-size: 12px; }}
@media (max-width: 900px) {{
  main {{ padding: 14px; }}
  .analysis-grid {{ grid-template-columns: 1fr; }}
  .mini-list-row {{ grid-template-columns: 1fr; }}
  .sudoku-grid {{ --cell: 25px; }}
}}
</style>
</head>
<body>
<main>
  <section class="hero">
    <h1>Hardest archived h120 case: 12x12 with 120 hidden cells</h1>
    <p class="lead">
      This page reads the saved seed52 case from the h120 hard-stage curve run.
      It shows one concrete failure: the model fills many local cells, but the final board still has many duplicate row/column/box conflicts.
      Red cells differ from the sampled solution; orange outlines mark duplicate conflicts; blue marks cells that changed from the previous loop.
    </p>
    <div class="stat-row">
      <div class="stat"><b>{case.size}x{case.size}</b><span>board, {case.box_rows}x{case.box_cols} boxes</span></div>
      <div class="stat"><b>{hidden}</b><span>hidden cells</span></div>
      <div class="stat"><b>{clues}</b><span>given clues</span></div>
      <div class="stat"><b>{len(final_stats.wrong_hidden)}</b><span>wrong hidden cells at {html.escape(final_name)}</span></div>
      <div class="stat"><b>{len(final_stats.conflicts)}</b><span>duplicate conflict units at {html.escape(final_name)}</span></div>
    </div>
    <div class="legend">
      <span><i style="background:var(--clue)"></i>clue</span>
      <span><i style="background:var(--correct)"></i>correct hidden cell</span>
      <span><i style="background:var(--wrong)"></i>wrong hidden cell</span>
      <span><i style="background:white;border:2px solid var(--conflict)"></i>duplicate conflict</span>
      <span><i style="background:var(--changed)"></i>changed since previous loop</span>
    </div>
  </section>

  <h2>Boards</h2>
  <div class="boards">
    {''.join(board_sections)}
  </div>

  <h2>What the loops did</h2>
  <div class="analysis-grid">
    <div>
      {render_change_summary(stats_by_loop, names)}
      <p class="small">
        Read: the early loops do most of the correction. After that, changes become small and mostly rearrange remaining conflicts instead of completing the board.
      </p>
      {render_metric_table(curve)}
    </div>
    <div>
      <div class="mini-list">
        <h4>FutureSeed state in this run</h4>
        <p>{html.escape("; ".join(fs_bits) if fs_bits else "No FutureSeed scalar stats found in metrics JSON.")}</p>
        <p class="small">
          The saved run used a fixed FutureSeed state: no decay, no loop feedback, no learned state update. That makes this case useful for seeing why late loops run out of useful edits.
        </p>
      </div>
      <h3 style="margin-top:14px;">Final hidden-token histogram</h3>
      {render_token_hist(case.boards[final_name], puzzle, case.size)}
    </div>
  </div>

  <h2>Where final errors concentrate</h2>
  {render_wrong_summary(final_stats, case)}
  {render_conflict_table(final_stats, case.size)}

  <h2>Source</h2>
  <p class="small">
    Parsed from <code>{html.escape(str(case_html.relative_to(REPO_ROOT)))}</code>.
    Generated at {datetime.now(timezone.utc).isoformat()}.
  </p>
</main>
</body>
</html>
""",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case-html", type=Path, default=DEFAULT_CASE_HTML)
    parser.add_argument("--metrics-json", type=Path, default=DEFAULT_METRICS_JSON)
    parser.add_argument("--out-html", type=Path, default=DEFAULT_OUT_HTML)
    parser.add_argument("--out-json", type=Path, default=DEFAULT_OUT_JSON)
    parser.add_argument("--out-png", type=Path, default=DEFAULT_OUT_PNG)
    args = parser.parse_args()

    case = parse_case(args.case_html)
    names = loop_names(case)
    if not names:
        raise ValueError(f"{args.case_html} has no loop boards")
    puzzle = case.boards["puzzle"]
    solution = case.boards["solution"]
    stats_by_loop: dict[str, LoopStats] = {}
    for idx, name in enumerate(names):
        prev_name = names[idx - 1] if idx else None
        stats_by_loop[name] = analyze_loop(
            name,
            case.boards[name],
            puzzle,
            solution,
            case.boards[prev_name] if prev_name else None,
            size=case.size,
            box_rows=case.box_rows,
            box_cols=case.box_cols,
        )

    metrics = load_metrics(args.metrics_json)
    args.out_html.parent.mkdir(parents=True, exist_ok=True)
    make_html(case, stats_by_loop, names, metrics, case_html=args.case_html, out_html=args.out_html)
    make_png(case, stats_by_loop, names, out_png=args.out_png)
    summary = build_summary_json(case, stats_by_loop, names, metrics, case_html=args.case_html)
    args.out_json.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(args.out_html)
    print(args.out_json)
    print(args.out_png)


if __name__ == "__main__":
    main()
