#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Sequence

try:
    from PIL import Image, ImageDraw, ImageFont
except Exception as exc:  # pragma: no cover - optional visualization dependency
    raise SystemExit(f"Pillow is required for inline maze visuals: {exc}")


COLORS = {
    "#": (30, 32, 36),
    ".": (248, 247, 240),
    "S": (54, 120, 220),
    "G": (54, 170, 92),
    "P": (246, 180, 65),
    "T": (246, 180, 65),
    "F": (225, 70, 70),
    "M": (139, 92, 246),
}


def load_font(size: int, *, bold: bool = False) -> ImageFont.ImageFont:
    candidates = [
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf" if bold else "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/System/Library/Fonts/Menlo.ttc",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]
    for path in candidates:
        try:
            return ImageFont.truetype(path, size)
        except Exception:
            continue
    return ImageFont.load_default()


def draw_text(draw: ImageDraw.ImageDraw, xy: tuple[int, int], text: str, *, font: ImageFont.ImageFont, fill=(35, 35, 35)) -> None:
    draw.text(xy, text, fill=fill, font=font)


def draw_grid(draw: ImageDraw.ImageDraw, rows: Sequence[str], x0: int, y0: int, cell: int) -> None:
    for y, row in enumerate(rows):
        for x, ch in enumerate(row):
            draw.rectangle(
                [x0 + x * cell, y0 + y * cell, x0 + (x + 1) * cell - 1, y0 + (y + 1) * cell - 1],
                fill=COLORS.get(ch, (210, 210, 210)),
            )
    draw.rectangle([x0, y0, x0 + len(rows[0]) * cell, y0 + len(rows) * cell], outline=(85, 85, 85), width=1)


def loop_gain(case: Dict[str, Any], final_loop: str) -> float:
    return float(case["loop_stats"][final_loop]["path_f1"]) - float(case["loop_stats"]["loop1"]["path_f1"])


def stat_line(stats: Dict[str, Any]) -> str:
    return (
        f"F1 {float(stats['path_f1']):.3f} | P {float(stats['path_precision']):.3f} | "
        f"R {float(stats['path_recall']):.3f} | FP {int(stats['false_positive_path'])} | "
        f"FN {int(stats['false_negative_path'])}"
    )


def render_case(case: Dict[str, Any], final_loop: str, out_path: Path, subtitle: str) -> None:
    font = load_font(18)
    font_b = load_font(20, bold=True)
    font_s = load_font(14)
    cell = 13
    n = len(case["target_rows"])
    panel_w = n * cell
    panel_h = n * cell
    margin = 28
    gap = 30
    title_h = 92
    stat_h = 52
    legend_h = 78
    width = margin * 2 + panel_w * 3 + gap * 2
    height = title_h + stat_h + panel_h + legend_h + margin
    image = Image.new("RGB", (width, height), (255, 255, 251))
    draw = ImageDraw.Draw(image)

    loop1 = case["loop_stats"]["loop1"]
    final = case["loop_stats"][final_loop]
    gain = float(final["path_f1"]) - float(loop1["path_f1"])
    draw_text(draw, (margin, 18), f"Maze case {case['case_index']} - {subtitle}", font=font_b)
    draw_text(
        draw,
        (margin, 46),
        (
            f"loop1 -> {final_loop}: F1 {float(loop1['path_f1']):.3f} -> {float(final['path_f1']):.3f} "
            f"({gain:+.3f}); FP {int(loop1['false_positive_path'])}->{int(final['false_positive_path'])}, "
            f"FN {int(loop1['false_negative_path'])}->{int(final['false_negative_path'])}"
        ),
        font=font,
    )

    y_head = title_h
    y_grid = title_h + stat_h
    panels = [
        ("Target path", case["target_rows"], None),
        ("Loop 1 prediction", case["comparison_rows"]["loop1"], loop1),
        (f"{final_loop.capitalize()} prediction", case["comparison_rows"][final_loop], final),
    ]
    for idx, (title, rows, stats) in enumerate(panels):
        x = margin + idx * (panel_w + gap)
        draw_text(draw, (x, y_head), title, font=font_b)
        draw_text(draw, (x, y_head + 25), stat_line(stats) if stats else "orange = true solution path", font=font_s)
        draw_grid(draw, rows, x, y_grid, cell)

    lx = margin
    ly = y_grid + panel_h + 22
    legend = [
        ("#", "wall", 145),
        ("P", "target/true path", 185),
        ("T", "true positive", 145),
        ("F", "false positive", 145),
        ("M", "false negative", 185),
        ("S", "start", 115),
        ("G", "goal", 115),
    ]
    for ch, label, step in legend:
        draw.rectangle([lx, ly, lx + 18, ly + 18], fill=COLORS[ch], outline=(80, 80, 80))
        draw_text(draw, (lx + 24, ly - 1), label, font=font_s)
        lx += step
    image.save(out_path)


def mean(values: Sequence[float]) -> float:
    return sum(values) / max(1, len(values))


def render_summary(cases: Sequence[Dict[str, Any]], final_loop: str, eval_clean: Dict[str, Any] | None, out_path: Path) -> None:
    font = load_font(18)
    font_b = load_font(20, bold=True)
    font_s = load_font(14)
    image = Image.new("RGB", (1180, 680), (255, 255, 251))
    draw = ImageDraw.Draw(image)
    draw_text(draw, (36, 24), f"Maze loop summary: loop1 -> {final_loop}", font=font_b)
    draw_text(draw, (36, 56), "Loop behavior should reduce false positives without adding false negatives.", font=font)

    if eval_clean:
        loop1 = eval_clean["loop1"]
        final = eval_clean[final_loop]
        metrics = [
            ("held-out path F1", float(loop1["path_f1"]), float(final["path_f1"])),
            ("precision", float(loop1["path_precision"]), float(final["path_precision"])),
            ("recall", float(loop1["path_recall"]), float(final["path_recall"])),
            ("pred PATH frac", float(loop1["path_pred_frac"]), float(final["path_pred_frac"])),
        ]
        card_y = 105
        for idx, (name, before, after) in enumerate(metrics):
            x = 36 + idx * 275
            delta = after - before
            draw.rounded_rectangle([x, card_y, x + 245, card_y + 110], radius=10, fill=(245, 244, 238), outline=(218, 214, 204))
            draw_text(draw, (x + 16, card_y + 14), name, font=font_b)
            draw_text(draw, (x + 16, card_y + 47), f"{before:.4f} -> {after:.4f}", font=font)
            draw_text(draw, (x + 16, card_y + 75), f"delta {delta:+.4f}", fill=(20, 100, 70) if delta >= 0 else (170, 50, 50), font=font_s)

    rows = []
    for case in cases:
        loop1 = case["loop_stats"]["loop1"]
        final = case["loop_stats"][final_loop]
        rows.append((float(loop1["false_positive_path"]), float(final["false_positive_path"]), float(loop1["false_negative_path"]), float(final["false_negative_path"])))
    fp1, fpf, fn1, fnf = [mean([row[idx] for row in rows]) for idx in range(4)]
    draw_text(draw, (36, 260), "Visual hard-case average counts", font=font_b)
    chart_x, chart_y = 80, 305
    bar_w = 160
    vals = [
        ("FP loop1", fp1, (225, 70, 70)),
        (f"FP {final_loop}", fpf, (190, 50, 50)),
        ("FN loop1", fn1, (139, 92, 246)),
        (f"FN {final_loop}", fnf, (105, 70, 210)),
    ]
    scale = 1.4
    for idx, (name, value, color) in enumerate(vals):
        x = chart_x + idx * 245
        height = value * scale
        draw.rectangle([x, chart_y + 285 - height, x + bar_w, chart_y + 285], fill=color)
        draw_text(draw, (x, chart_y + 292), name, font=font_s)
        draw_text(draw, (x + 38, int(chart_y + 285 - height - 26)), f"{value:.1f}", font=font)
    draw.line([chart_x - 20, chart_y + 285, chart_x + 4 * 245 - 70, chart_y + 285], fill=(90, 90, 90), width=2)
    draw.rounded_rectangle([36, 620, 1144, 664], radius=10, fill=(250, 236, 226), outline=(230, 180, 160))
    draw_text(draw, (54, 632), "Good loop dynamics should push FP down while keeping FN flat or lower.", font=font)
    image.save(out_path)


def infer_final_loop(cases: Sequence[Dict[str, Any]]) -> str:
    loop_order = cases[0].get("loop_order") if cases else None
    if loop_order:
        return str(loop_order[-1])
    keys = [key for key in cases[0]["loop_stats"] if key.startswith("loop")]
    return sorted(keys, key=lambda key: int(key.replace("loop", "")))[-1]


def read_eval_clean(result_json: Path) -> Dict[str, Any] | None:
    try:
        result = json.loads(result_json.read_text(encoding="utf-8"))
        metrics = result.get("metrics", {})
        eval_clean = metrics.get("eval_clean")
        return eval_clean if isinstance(eval_clean, dict) else None
    except Exception:
        return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--result-json", default="")
    parser.add_argument("--cases-json", default="")
    parser.add_argument("--out-dir", default="")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    cases_path = Path(args.cases_json) if args.cases_json else run_dir / "output" / "visualizations" / "cases.json"
    out_dir = Path(args.out_dir) if args.out_dir else run_dir / "output" / "visualizations" / "inline"
    out_dir.mkdir(parents=True, exist_ok=True)

    payload = json.loads(cases_path.read_text(encoding="utf-8"))
    cases: List[Dict[str, Any]] = payload.get("cases", payload) if isinstance(payload, dict) else payload
    if not cases:
        raise SystemExit(f"No cases found in {cases_path}")
    final_loop = infer_final_loop(cases)
    result_json = Path(args.result_json) if args.result_json else next((run_dir / "output").glob("eqr_maze_probe_*.json"), None)
    eval_clean = read_eval_clean(result_json) if result_json else None

    rows = sorted(cases, key=lambda case: loop_gain(case, final_loop))
    render_summary(cases, final_loop, eval_clean, out_dir / "summary_loop_effect.png")
    render_case(rows[-1], final_loop, out_dir / "case_best_loop_gain.png", "best loop gain")
    render_case(rows[0], final_loop, out_dir / "case_worst_loop_regression.png", "worst loop regression")
    mid = min(rows, key=lambda case: abs(loop_gain(case, final_loop)))
    render_case(mid, final_loop, out_dir / "case_typical_failure.png", "typical near-zero loop gain")
    print(out_dir)


if __name__ == "__main__":
    main()
