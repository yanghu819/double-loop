#!/usr/bin/env python3
"""Build a human-readable 9x9 Sudoku case visualization from archived run HTML."""

from __future__ import annotations

import argparse
import html
import json
from dataclasses import dataclass
from html.parser import HTMLParser
from pathlib import Path
from typing import Iterable

from PIL import Image, ImageDraw, ImageFont


N = 9
CELLS = N * N
BOARD_NAMES = ["puzzle", "solution", "loop1", "loop2", "loop3", "loop4", "loop5"]


class CellParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.cells: list[str] = []
        self._in_cell = False
        self._text = ""

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attr = dict(attrs)
        if tag == "div" and "cell" in str(attr.get("class", "")).split():
            self._in_cell = True
            self._text = ""

    def handle_data(self, data: str) -> None:
        if self._in_cell:
            self._text += data.strip()

    def handle_endtag(self, tag: str) -> None:
        if tag == "div" and self._in_cell:
            self.cells.append(self._text or ".")
            self._in_cell = False
            self._text = ""


@dataclass(frozen=True)
class BoardStats:
    wrong_blanks: list[int]
    conflict_cells: set[int]
    conflict_units: list[str]

    @property
    def exact(self) -> bool:
        return not self.wrong_blanks


def parse_boards(path: Path) -> dict[str, list[str]]:
    parser = CellParser()
    parser.feed(path.read_text(encoding="utf-8"))
    expected = len(BOARD_NAMES) * CELLS
    if len(parser.cells) != expected:
        raise ValueError(f"{path} has {len(parser.cells)} cells, expected {expected}")
    return {
        name: parser.cells[i * CELLS : (i + 1) * CELLS]
        for i, name in enumerate(BOARD_NAMES)
    }


def board_rows(board: list[str]) -> list[list[str]]:
    return [board[r * N : (r + 1) * N] for r in range(N)]


def units() -> Iterable[tuple[str, list[int]]]:
    for row in range(N):
        yield f"row {row + 1}", [row * N + col for col in range(N)]
    for col in range(N):
        yield f"col {col + 1}", [row * N + col for row in range(N)]
    for box_row in range(3):
        for box_col in range(3):
            idxs = []
            for dr in range(3):
                for dc in range(3):
                    idxs.append((box_row * 3 + dr) * N + (box_col * 3 + dc))
            yield f"box {box_row + 1},{box_col + 1}", idxs


def analyze(board: list[str], solution: list[str], puzzle: list[str]) -> BoardStats:
    blanks = [idx for idx, value in enumerate(puzzle) if value == "."]
    wrong_blanks = [idx for idx in blanks if board[idx] != solution[idx]]
    conflict_cells: set[int] = set()
    conflict_units: list[str] = []
    for name, idxs in units():
        by_digit: dict[str, list[int]] = {}
        for idx in idxs:
            digit = board[idx]
            if digit != ".":
                by_digit.setdefault(digit, []).append(idx)
        repeated = {digit: locs for digit, locs in by_digit.items() if len(locs) > 1}
        if repeated:
            bits = []
            for digit, locs in sorted(repeated.items()):
                conflict_cells.update(locs)
                coords = ",".join(coord(idx) for idx in locs)
                bits.append(f"{digit}@{coords}")
            conflict_units.append(f"{name}: " + "; ".join(bits))
    return BoardStats(wrong_blanks=wrong_blanks, conflict_cells=conflict_cells, conflict_units=conflict_units)


def coord(idx: int) -> str:
    row, col = divmod(idx, N)
    return f"R{row + 1}C{col + 1}"


def board_text(board: list[str]) -> str:
    return "\n".join(" ".join(row) for row in board_rows(board))


def border_style(idx: int) -> str:
    row, col = divmod(idx, N)
    style = []
    if col == N - 1:
        style.append("border-right: 0")
    elif (col + 1) % 3 == 0:
        style.append("border-right: 3px solid #111827")
    if row == N - 1:
        style.append("border-bottom: 0")
    elif (row + 1) % 3 == 0:
        style.append("border-bottom: 3px solid #111827")
    return "; ".join(style)


def render_grid(
    board: list[str],
    solution: list[str],
    puzzle: list[str],
    *,
    stats: BoardStats | None,
    previous: list[str] | None = None,
    mode: str = "prediction",
) -> str:
    cells = []
    conflict_cells = stats.conflict_cells if stats else set()
    for idx, value in enumerate(board):
        classes = ["cell"]
        title = coord(idx)
        if mode == "puzzle":
            if value == ".":
                classes.append("hole")
                title += " hidden"
            else:
                classes.append("clue")
                title += " clue"
        elif mode == "solution":
            classes.append("solution")
            if puzzle[idx] != ".":
                classes.append("given-solution")
        elif puzzle[idx] != ".":
            classes.append("clue")
        elif value == solution[idx]:
            classes.append("correct")
        else:
            classes.append("wrong")
            title += f" truth={solution[idx]}"
        if stats and idx in conflict_cells:
            classes.append("conflict")
            title += " conflict"
        if previous is not None and puzzle[idx] == "." and previous[idx] != value:
            classes.append("changed")
            title += f" changed-from={previous[idx]}"
        style = border_style(idx)
        style_attr = f' style="{style}"' if style else ""
        cells.append(
            f'<div class="{" ".join(classes)}"{style_attr} title="{html.escape(title)}">'
            f"{html.escape(value)}</div>"
        )
    return '<div class="grid">' + "".join(cells) + "</div>"


def panel(
    title: str,
    subtitle: str,
    board: list[str],
    solution: list[str],
    puzzle: list[str],
    *,
    stats: BoardStats | None,
    previous: list[str] | None = None,
    mode: str = "prediction",
) -> str:
    return f"""
<section class="panel">
  <h3>{html.escape(title)}</h3>
  <p>{html.escape(subtitle)}</p>
  {render_grid(board, solution, puzzle, stats=stats, previous=previous, mode=mode)}
</section>
"""


def make_html(
    out_path: Path,
    fs: dict[str, list[str]],
    nofs: dict[str, list[str]],
    summary: dict[str, object],
) -> None:
    puzzle = fs["puzzle"]
    solution = fs["solution"]
    fs_stats = {name: analyze(fs[name], solution, puzzle) for name in ["loop1", "loop3", "loop5"]}
    nofs_stats = {name: analyze(nofs[name], solution, puzzle) for name in ["loop1", "loop3", "loop5"]}
    nofs_wrong = nofs_stats["loop5"].wrong_blanks
    wrong_items = [
        f"<li><code>{coord(idx)}</code>: noFS loop5={html.escape(nofs['loop5'][idx])}, truth={html.escape(solution[idx])}</li>"
        for idx in nofs_wrong
    ]
    conflict_items = [f"<li>{html.escape(item)}</li>" for item in nofs_stats["loop5"].conflict_units]

    fs_panels = "\n".join(
        panel(
            title=f"FutureSeed loop {loop[-1]}",
            subtitle=f"wrong blanks: {len(fs_stats[loop].wrong_blanks)}/16, conflict units: {len(fs_stats[loop].conflict_units)}",
            board=fs[loop],
            solution=solution,
            puzzle=puzzle,
            stats=fs_stats[loop],
            previous=fs["loop1"] if loop == "loop3" else fs["loop3"] if loop == "loop5" else None,
        )
        for loop in ["loop1", "loop3", "loop5"]
    )
    nofs_panels = "\n".join(
        panel(
            title=f"no FutureSeed loop {loop[-1]}",
            subtitle=f"wrong blanks: {len(nofs_stats[loop].wrong_blanks)}/16, conflict units: {len(nofs_stats[loop].conflict_units)}",
            board=nofs[loop],
            solution=solution,
            puzzle=puzzle,
            stats=nofs_stats[loop],
            previous=nofs["loop1"] if loop == "loop3" else nofs["loop3"] if loop == "loop5" else None,
        )
        for loop in ["loop1", "loop3", "loop5"]
    )

    body = f"""<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>9x9 Sudoku case: FutureSeed vs no-FutureSeed</title>
<style>
:root {{
  --ink: #111827;
  --muted: #4b5563;
  --line: #9ca3af;
  --panel: #ffffff;
  --page: #f3f4f6;
  --clue: #d1d5db;
  --hole: #fff7ed;
  --correct: #bbf7d0;
  --wrong: #fecaca;
  --conflict: #f59e0b;
  --changed: #2563eb;
}}
body {{
  margin: 24px;
  background: var(--page);
  color: var(--ink);
  font-family: ui-sans-serif, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
}}
h1 {{ margin: 0 0 8px; font-size: 28px; letter-spacing: 0; }}
h2 {{ margin: 24px 0 10px; font-size: 18px; letter-spacing: 0; }}
h3 {{ margin: 0 0 4px; font-size: 15px; letter-spacing: 0; }}
p {{ margin: 0; color: var(--muted); font-size: 13px; line-height: 1.35; }}
code {{ font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; }}
.lede {{ max-width: 1120px; font-size: 14px; margin-bottom: 14px; }}
.summary {{
  display: grid;
  grid-template-columns: repeat(4, minmax(150px, 1fr));
  gap: 10px;
  max-width: 1120px;
  margin: 14px 0 18px;
}}
.stat {{ background: var(--panel); border: 1px solid #d1d5db; border-radius: 8px; padding: 10px 12px; }}
.stat strong {{ display: block; font-size: 22px; line-height: 1.1; }}
.stat span {{ color: var(--muted); font-size: 12px; }}
.row {{ display: flex; flex-wrap: wrap; gap: 14px; align-items: flex-start; }}
.panel {{ background: var(--panel); border: 1px solid #d1d5db; border-radius: 8px; padding: 12px; }}
.panel p {{ min-height: 36px; max-width: 330px; }}
.grid {{
  display: grid;
  grid-template-columns: repeat(9, 34px);
  grid-template-rows: repeat(9, 34px);
  width: max-content;
  border: 3px solid var(--ink);
  margin-top: 10px;
}}
.cell {{
  width: 34px;
  height: 34px;
  box-sizing: border-box;
  display: flex;
  align-items: center;
  justify-content: center;
  border-right: 1px solid var(--line);
  border-bottom: 1px solid var(--line);
  font-size: 17px;
  font-weight: 800;
  position: relative;
}}
.clue {{ background: var(--clue); color: var(--ink); }}
.hole {{ background: var(--hole); color: #9a3412; }}
.solution {{ background: #e0f2fe; color: #0c4a6e; }}
.given-solution {{ box-shadow: inset 0 0 0 2px rgba(17,24,39,0.14); }}
.correct {{ background: var(--correct); color: #14532d; }}
.wrong {{ background: var(--wrong); color: #7f1d1d; }}
.conflict::after {{
  content: "";
  position: absolute;
  inset: 3px;
  border: 3px solid var(--conflict);
  border-radius: 6px;
  pointer-events: none;
}}
.changed::before {{
  content: "";
  position: absolute;
  left: 8px;
  right: 8px;
  bottom: 4px;
  height: 3px;
  background: var(--changed);
  border-radius: 4px;
}}
.legend {{ display: flex; flex-wrap: wrap; gap: 8px; margin: 10px 0 4px; }}
.chip {{ display: inline-flex; align-items: center; gap: 6px; font-size: 12px; color: var(--muted); }}
.swatch {{ width: 16px; height: 16px; border: 1px solid #9ca3af; border-radius: 4px; display: inline-block; }}
.notes {{ max-width: 1120px; background: var(--panel); border: 1px solid #d1d5db; border-radius: 8px; padding: 12px 14px; }}
.notes ul {{ margin: 8px 0 0 18px; padding: 0; }}
.notes li {{ margin: 4px 0; font-size: 13px; }}
@media (max-width: 760px) {{
  body {{ margin: 14px; }}
  .summary {{ grid-template-columns: repeat(2, minmax(130px, 1fr)); }}
  .grid {{ grid-template-columns: repeat(9, 29px); grid-template-rows: repeat(9, 29px); }}
  .cell {{ width: 29px; height: 29px; font-size: 15px; }}
}}
</style>
</head>
<body>
<h1>9x9 Sudoku case: FutureSeed vs no-FutureSeed</h1>
<p class="lede">One fixed GPU1 evaluation puzzle from seed 52. The model must fill 16 hidden cells. This page shows the concrete board, not an aggregate score.</p>
<div class="summary">
  <div class="stat"><strong>{summary["holes"]}</strong><span>hidden cells</span></div>
  <div class="stat"><strong>{summary["fs_loop5_wrong"]}/16</strong><span>FutureSeed loop5 wrong blanks</span></div>
  <div class="stat"><strong>{summary["nofs_loop5_wrong"]}/16</strong><span>no-FutureSeed loop5 wrong blanks</span></div>
  <div class="stat"><strong>{summary["nofs_loop5_conflicts"]}</strong><span>no-FutureSeed loop5 conflict units</span></div>
</div>
<div class="legend">
  <span class="chip"><span class="swatch" style="background: var(--clue)"></span>given clue</span>
  <span class="chip"><span class="swatch" style="background: var(--hole)"></span>hidden in puzzle</span>
  <span class="chip"><span class="swatch" style="background: var(--correct)"></span>correct fill</span>
  <span class="chip"><span class="swatch" style="background: var(--wrong)"></span>wrong fill</span>
  <span class="chip"><span class="swatch" style="border: 3px solid var(--conflict)"></span>row/col/box duplicate</span>
  <span class="chip"><span class="swatch" style="border-bottom: 4px solid var(--changed)"></span>changed since previous shown loop</span>
</div>
<h2>Puzzle and truth</h2>
<div class="row">
{panel("Puzzle", "Dots are hidden cells. Gray numbers are givens.", puzzle, solution, puzzle, stats=None, mode="puzzle")}
{panel("Solution", "The sampled solved board used for exact-match scoring.", solution, solution, puzzle, stats=None, mode="solution")}
</div>
<h2>FutureSeed backbone</h2>
<div class="row">{fs_panels}</div>
<h2>No-FutureSeed control</h2>
<div class="row">{nofs_panels}</div>
<h2>What fails without FutureSeed</h2>
<div class="notes">
  <p>On this case, no-FutureSeed is not just making isolated digit mistakes. Its final board violates global Sudoku coupling through duplicated digits in rows, columns, and boxes.</p>
  <ul>
    {''.join(wrong_items)}
  </ul>
  <p style="margin-top: 10px;">Conflict units in no-FutureSeed loop5:</p>
  <ul>
    {''.join(conflict_items)}
  </ul>
</div>
</body>
</html>
"""
    out_path.write_text(body, encoding="utf-8")


def load_font(size: int, *, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = [
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf" if bold else "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/Library/Fonts/Arial Unicode.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]
    for path in candidates:
        if Path(path).exists():
            return ImageFont.truetype(path, size=size)
    return ImageFont.load_default()


def draw_board(
    draw: ImageDraw.ImageDraw,
    xy: tuple[int, int],
    title: str,
    subtitle: str,
    board: list[str],
    solution: list[str],
    puzzle: list[str],
    stats: BoardStats | None,
    *,
    mode: str = "prediction",
) -> None:
    x0, y0 = xy
    cell = 34
    title_font = load_font(20, bold=True)
    sub_font = load_font(13)
    digit_font = load_font(18, bold=True)
    draw.text((x0, y0), title, fill="#111827", font=title_font)
    draw.text((x0, y0 + 25), subtitle, fill="#4b5563", font=sub_font)
    grid_x, grid_y = x0, y0 + 50
    conflict_cells = stats.conflict_cells if stats else set()
    for idx, value in enumerate(board):
        row, col = divmod(idx, N)
        x = grid_x + col * cell
        y = grid_y + row * cell
        if mode == "puzzle":
            fill = "#fff7ed" if value == "." else "#d1d5db"
            text_fill = "#9a3412" if value == "." else "#111827"
        elif mode == "solution":
            fill = "#e0f2fe"
            text_fill = "#0c4a6e"
        elif puzzle[idx] != ".":
            fill = "#d1d5db"
            text_fill = "#111827"
        elif value == solution[idx]:
            fill = "#bbf7d0"
            text_fill = "#14532d"
        else:
            fill = "#fecaca"
            text_fill = "#7f1d1d"
        draw.rectangle((x, y, x + cell, y + cell), fill=fill, outline="#9ca3af", width=1)
        if stats and idx in conflict_cells:
            draw.rounded_rectangle((x + 3, y + 3, x + cell - 3, y + cell - 3), radius=5, outline="#f59e0b", width=3)
        bbox = draw.textbbox((0, 0), value, font=digit_font)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        draw.text((x + (cell - tw) / 2, y + (cell - th) / 2 - 1), value, fill=text_fill, font=digit_font)
    for k in range(N + 1):
        width = 3 if k % 3 == 0 else 1
        draw.line((grid_x + k * cell, grid_y, grid_x + k * cell, grid_y + N * cell), fill="#111827" if width == 3 else "#9ca3af", width=width)
        draw.line((grid_x, grid_y + k * cell, grid_x + N * cell, grid_y + k * cell), fill="#111827" if width == 3 else "#9ca3af", width=width)


def make_png(
    out_path: Path,
    fs: dict[str, list[str]],
    nofs: dict[str, list[str]],
    summary: dict[str, object],
) -> None:
    puzzle = fs["puzzle"]
    solution = fs["solution"]
    stats = {
        "fs1": analyze(fs["loop1"], solution, puzzle),
        "fs5": analyze(fs["loop5"], solution, puzzle),
        "nofs1": analyze(nofs["loop1"], solution, puzzle),
        "nofs3": analyze(nofs["loop3"], solution, puzzle),
        "nofs5": analyze(nofs["loop5"], solution, puzzle),
    }
    width, height = 1430, 920
    image = Image.new("RGB", (width, height), "#f3f4f6")
    draw = ImageDraw.Draw(image)
    title_font = load_font(31, bold=True)
    sub_font = load_font(16)
    draw.text((36, 26), "9x9 Sudoku case: FutureSeed vs no-FutureSeed", fill="#111827", font=title_font)
    draw.text(
        (36, 68),
        f"Seed 52, 16 hidden cells. FS loop5 wrong={summary['fs_loop5_wrong']}/16; noFS loop5 wrong={summary['nofs_loop5_wrong']}/16 with {summary['nofs_loop5_conflicts']} conflict units.",
        fill="#4b5563",
        font=sub_font,
    )
    xs = [36, 382, 728, 1074]
    y1 = 120
    y2 = 525
    draw_board(draw, (xs[0], y1), "Puzzle", "16 hidden cells", puzzle, solution, puzzle, None, mode="puzzle")
    draw_board(draw, (xs[1], y1), "Solution", "ground truth", solution, solution, puzzle, None, mode="solution")
    draw_board(draw, (xs[2], y1), "FutureSeed loop1", f"wrong {len(stats['fs1'].wrong_blanks)}/16", fs["loop1"], solution, puzzle, stats["fs1"])
    draw_board(draw, (xs[3], y1), "FutureSeed loop5", f"wrong {len(stats['fs5'].wrong_blanks)}/16", fs["loop5"], solution, puzzle, stats["fs5"])
    draw_board(draw, (xs[0], y2), "noFS loop1", f"wrong {len(stats['nofs1'].wrong_blanks)}/16", nofs["loop1"], solution, puzzle, stats["nofs1"])
    draw_board(draw, (xs[1], y2), "noFS loop3", f"wrong {len(stats['nofs3'].wrong_blanks)}/16", nofs["loop3"], solution, puzzle, stats["nofs3"])
    draw_board(draw, (xs[2], y2), "noFS loop5", f"wrong {len(stats['nofs5'].wrong_blanks)}/16", nofs["loop5"], solution, puzzle, stats["nofs5"])

    note_font = load_font(15)
    note_x = xs[3]
    note_y = y2 + 54
    draw.rounded_rectangle((note_x, note_y, note_x + 306, note_y + 300), radius=8, fill="#ffffff", outline="#d1d5db", width=1)
    draw.text((note_x + 14, note_y + 12), "noFS loop5 wrong cells", fill="#111827", font=load_font(18, bold=True))
    y = note_y + 45
    for idx in stats["nofs5"].wrong_blanks:
        line = f"{coord(idx)}: pred {nofs['loop5'][idx]} / truth {solution[idx]}"
        draw.text((note_x + 14, y), line, fill="#7f1d1d", font=note_font)
        y += 24
    y += 8
    draw.text((note_x + 14, y), "Orange outlines mark", fill="#4b5563", font=note_font)
    draw.text((note_x + 14, y + 22), "row/col/box duplicates.", fill="#4b5563", font=note_font)
    image.save(out_path)


def make_summary(fs: dict[str, list[str]], nofs: dict[str, list[str]]) -> dict[str, object]:
    puzzle = fs["puzzle"]
    solution = fs["solution"]
    if puzzle != nofs["puzzle"] or solution != nofs["solution"]:
        raise ValueError("FS and noFS case files do not describe the same puzzle")
    fs5 = analyze(fs["loop5"], solution, puzzle)
    nofs5 = analyze(nofs["loop5"], solution, puzzle)
    return {
        "holes": puzzle.count("."),
        "puzzle": board_rows(puzzle),
        "solution": board_rows(solution),
        "future_seed": {
            loop: {
                "board": board_rows(fs[loop]),
                "wrong_blanks": [coord(idx) for idx in analyze(fs[loop], solution, puzzle).wrong_blanks],
                "conflict_units": analyze(fs[loop], solution, puzzle).conflict_units,
            }
            for loop in ["loop1", "loop3", "loop5"]
        },
        "no_future_seed": {
            loop: {
                "board": board_rows(nofs[loop]),
                "wrong_blanks": [coord(idx) for idx in analyze(nofs[loop], solution, puzzle).wrong_blanks],
                "conflict_units": analyze(nofs[loop], solution, puzzle).conflict_units,
            }
            for loop in ["loop1", "loop3", "loop5"]
        },
        "fs_loop5_wrong": len(fs5.wrong_blanks),
        "nofs_loop5_wrong": len(nofs5.wrong_blanks),
        "nofs_loop5_conflicts": len(nofs5.conflict_units),
    }


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    default_fs = root / "runs/rwkv-fs-backbone-cliff2-20260604T070945Z-c9c212a-rwkv_fs-s52/output/futureseed_loop_case_seed52.html"
    default_nofs = root / "runs/rwkv-fs-backbone-cliff2-20260604T070945Z-c9c212a-rwkv_nofs-s52/output/futureseed_loop_case_seed52.html"
    default_out = root / "runs/rwkv-fs-backbone-cliff-20260604"
    parser = argparse.ArgumentParser()
    parser.add_argument("--fs-html", type=Path, default=default_fs)
    parser.add_argument("--nofs-html", type=Path, default=default_nofs)
    parser.add_argument("--out-dir", type=Path, default=default_out)
    parser.add_argument("--prefix", default="case_9x9_seed52_fs_vs_nofs")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    fs = parse_boards(args.fs_html)
    nofs = parse_boards(args.nofs_html)
    summary = make_summary(fs, nofs)
    html_path = args.out_dir / f"{args.prefix}.html"
    png_path = args.out_dir / f"{args.prefix}.png"
    json_path = args.out_dir / f"{args.prefix}.json"
    make_html(html_path, fs, nofs, summary)
    make_png(png_path, fs, nofs, summary)
    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"wrote {html_path}")
    print(f"wrote {png_path}")
    print(f"wrote {json_path}")
    print(f"puzzle:\n{board_text(fs['puzzle'])}")
    print(f"solution:\n{board_text(fs['solution'])}")


if __name__ == "__main__":
    main()
