#!/usr/bin/env python3
from __future__ import annotations

import argparse
import html
import json
import math
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


RESULT_GLOBS = (
    "output/futureseed_loop_seed*.json",
    "**/futureseed_loop_seed*.json",
    "output/eqr_probe_seed*.json",
    "**/eqr_probe_seed*.json",
    "output/eqr_maze_probe_*.json",
    "**/eqr_maze_probe_*.json",
    "output/checkpoint_eval_step*.json",
    "**/checkpoint_eval_step*.json",
)

CASE_GLOBS = (
    "output/visualizations/index.html",
    "output/visualizations/casebook.md",
    "output/visualizations/cases.json",
    "output/case_bank/**/index.html",
    "output/case_bank/**/*.html",
    "output/futureseed_loop_case_seed*.html",
    "**/*case*.html",
    "**/*case*.png",
)

NUMBER_RE = re.compile(r"^-?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?$")
KV_RE = re.compile(r"([A-Za-z_][A-Za-z0-9_/-]*)=([^,\s;]+)")
LOOP_RE = re.compile(r"^loop(\d+)$")
HOLES_RE = re.compile(r"holes?(\d+)|h(\d+)")


def read_json(path: Path) -> Any:
    if not path.exists():
        return None
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
        f.write("\n")


def is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def parse_number(text: str) -> Any:
    value = text.strip()
    if NUMBER_RE.match(value):
        try:
            as_float = float(value)
        except ValueError:
            return value
        if as_float.is_integer() and "." not in value and "e" not in value.lower():
            return int(as_float)
        return as_float
    return value


def rel_link(base: Path, path: Path) -> str:
    try:
        return path.resolve().relative_to(base.resolve()).as_posix()
    except Exception:
        return path.as_posix()


def html_link(href: str, label: str) -> str:
    return f'<a href="{html.escape(href)}">{html.escape(label)}</a>'


def find_repo_from_runs_dir(path: Path) -> Path:
    current = path.resolve()
    while current != current.parent:
        if (current / ".git").exists():
            return current
        current = current.parent
    return path.resolve()


def find_result_json(run_dir: Path) -> Optional[Path]:
    candidates: List[Path] = []
    for pattern in RESULT_GLOBS:
        candidates.extend(run_dir.glob(pattern))
        if candidates:
            break
    if not candidates:
        return None
    return sorted(set(candidates))[-1]


def unwrap_metrics(result: Any) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    if isinstance(result, dict):
        metrics = result.get("metrics")
        config = result.get("config") or result.get("args") or {}
        if isinstance(metrics, dict):
            return metrics, config if isinstance(config, dict) else {}
    return {}, {}


def discover_run_dirs(runs_dir: Path) -> List[Path]:
    dirs: List[Path] = []
    for child in sorted(runs_dir.iterdir() if runs_dir.exists() else []):
        if not child.is_dir():
            continue
        markers = (
            child / "metadata.json",
            child / "score.json",
            child / "config.json",
            child / "README.md",
            child / "logs",
            child / "output",
            child / "abort.json",
        )
        if any(path.exists() for path in markers):
            dirs.append(child)
    return dirs


def parse_log_history(run_dir: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for log_path in sorted((run_dir / "logs").glob("*.log")) + sorted(run_dir.glob("*.log")):
        try:
            text = log_path.read_text(encoding="utf-8", errors="replace")
        except Exception:
            continue
        for line in text.splitlines():
            if "step=" not in line:
                continue
            pairs = {key: parse_number(value) for key, value in KV_RE.findall(line)}
            if "step" not in pairs:
                continue
            row: Dict[str, Any] = {"source_log": rel_link(run_dir, log_path)}
            row.update(pairs)
            stage_match = re.search(r"stage=([^\]\s]+)", line)
            if stage_match:
                row["stage"] = stage_match.group(1)
            rows.append(row)
    dedup: Dict[int, Dict[str, Any]] = {}
    no_step: List[Dict[str, Any]] = []
    for row in rows:
        step = row.get("step")
        if isinstance(step, int):
            dedup[step] = row
        else:
            no_step.append(row)
    return no_step + [dedup[key] for key in sorted(dedup)]


def numeric_history(history: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for row in history:
        if not isinstance(row, dict):
            continue
        clean: Dict[str, Any] = {}
        for key, value in row.items():
            if is_number(value) or key in {"stage", "source_log", "path"}:
                clean[key] = value
        if clean:
            rows.append(clean)
    return rows


def extract_loop_rows(eval_clean: Any, group: str = "eval_clean") -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if not isinstance(eval_clean, dict):
        return rows
    for key, value in eval_clean.items():
        match = LOOP_RE.match(str(key))
        if not match or not isinstance(value, dict):
            continue
        row: Dict[str, Any] = {"group": group, "loop": int(match.group(1))}
        for metric_key, metric_value in value.items():
            if is_number(metric_value):
                row[metric_key] = float(metric_value)
        residual = eval_clean.get(f"{key}/residual")
        if isinstance(residual, dict):
            for metric_key, metric_value in residual.items():
                if is_number(metric_value):
                    row[f"residual.{metric_key}"] = float(metric_value)
        future_seed = eval_clean.get(f"{key}/future_seed")
        if isinstance(future_seed, dict):
            for metric_key, metric_value in future_seed.items():
                if is_number(metric_value):
                    row[f"future_seed.{metric_key}"] = float(metric_value)
        rows.append(row)
    return sorted(rows, key=lambda item: item["loop"])


def holes_sort_key(key: str) -> int:
    match = HOLES_RE.search(key)
    if not match:
        return 0
    return int(match.group(1) or match.group(2))


def extract_hole_rows(eval_by_holes: Any) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if not isinstance(eval_by_holes, dict):
        return rows
    for holes_key, payload in sorted(eval_by_holes.items(), key=lambda item: holes_sort_key(str(item[0]))):
        if not isinstance(payload, dict):
            continue
        holes = holes_sort_key(str(holes_key))
        for loop_row in extract_loop_rows(payload.get("eval_clean"), group=str(holes_key)):
            loop_row["holes"] = holes
            rows.append(loop_row)
    return rows


def extract_case_links(run_dir: Path) -> List[Dict[str, str]]:
    links: List[Dict[str, str]] = []
    seen = set()
    for pattern in CASE_GLOBS:
        for path in sorted(run_dir.glob(pattern)):
            if not path.is_file():
                continue
            rel = rel_link(run_dir, path)
            if rel in seen:
                continue
            seen.add(rel)
            if "output/visualizations/index.html" in rel:
                kind = "maze trajectory html"
            elif rel.endswith("casebook.md"):
                kind = "maze casebook"
            elif "case_bank" in rel and rel.endswith("index.html"):
                kind = "sudoku case-bank"
            elif rel.endswith(".png"):
                kind = "image"
            elif rel.endswith(".json"):
                kind = "case data"
            else:
                kind = "case html"
            links.append({"kind": kind, "path": rel})
    return links


def first_last_delta(rows: Sequence[Dict[str, Any]], metric: str, x_key: str = "loop") -> Optional[Dict[str, float]]:
    valid = [row for row in rows if is_number(row.get(metric))]
    if len(valid) < 2:
        return None
    first = valid[0]
    last = valid[-1]
    return {
        "first_x": float(first.get(x_key, 0.0)),
        "last_x": float(last.get(x_key, 0.0)),
        "first": float(first[metric]),
        "last": float(last[metric]),
        "delta": float(last[metric]) - float(first[metric]),
    }


def choose_history_metrics(history: Sequence[Dict[str, Any]]) -> List[str]:
    preferred = [
        "ce",
        "total",
        "loop1",
        "loop1_ce",
        "loop_last",
        "path_f1",
        "exact",
        "zH",
        "zH_rms",
        "zL",
        "zL_rms",
        "path_mass_loss",
        "self_correction_loss",
        "path_margin_loss",
    ]
    available = {key for row in history for key, value in row.items() if is_number(value) and key != "step"}
    return [key for key in preferred if key in available][:8]


def choose_loop_metrics(rows: Sequence[Dict[str, Any]]) -> List[str]:
    preferred = [
        "path_f1",
        "path_precision",
        "path_recall",
        "path_pred_frac",
        "label_exact",
        "blank_acc",
        "valid_sudoku",
        "solved_valid_clue",
        "token_acc",
        "future_seed.fs_gate_mean",
        "future_seed.fs_state_norm",
        "residual.zH_rms",
        "residual.zL_rms",
    ]
    available = {key for row in rows for key, value in row.items() if is_number(value) and key != "loop"}
    return [key for key in preferred if key in available][:8]


def svg_line_chart(
    rows: Sequence[Dict[str, Any]],
    x_key: str,
    metrics: Sequence[str],
    *,
    width: int = 760,
    height: int = 260,
) -> str:
    points_by_metric: Dict[str, List[Tuple[float, float]]] = {}
    xs: List[float] = []
    ys: List[float] = []
    for metric in metrics:
        pts: List[Tuple[float, float]] = []
        for row in rows:
            x = row.get(x_key)
            y = row.get(metric)
            if is_number(x) and is_number(y):
                pts.append((float(x), float(y)))
                xs.append(float(x))
                ys.append(float(y))
        if pts:
            points_by_metric[metric] = pts
    if not points_by_metric:
        return "<p class=\"muted\">No numeric series available.</p>"
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    if x_min == x_max:
        x_min -= 1.0
        x_max += 1.0
    if y_min == y_max:
        pad = max(0.1, abs(y_min) * 0.1)
        y_min -= pad
        y_max += pad
    y_pad = (y_max - y_min) * 0.08
    y_min -= y_pad
    y_max += y_pad
    left, right, top, bottom = 56, 18, 18, 42
    plot_w = width - left - right
    plot_h = height - top - bottom

    def sx(value: float) -> float:
        return left + (value - x_min) / (x_max - x_min) * plot_w

    def sy(value: float) -> float:
        return top + (y_max - value) / (y_max - y_min) * plot_h

    colors = ["#2563eb", "#dc2626", "#16a34a", "#9333ea", "#f59e0b", "#0891b2", "#be123c", "#4b5563"]
    lines = [
        f'<svg class="chart" viewBox="0 0 {width} {height}" role="img">',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{height-bottom}" class="axis"/>',
        f'<line x1="{left}" y1="{height-bottom}" x2="{width-right}" y2="{height-bottom}" class="axis"/>',
        f'<text x="{left}" y="{height-12}" class="tick">{html.escape(x_key)}</text>',
        f'<text x="4" y="{top+8}" class="tick">{y_max:.3g}</text>',
        f'<text x="4" y="{height-bottom}" class="tick">{y_min:.3g}</text>',
    ]
    legend_x = left
    for idx, (metric, pts) in enumerate(points_by_metric.items()):
        color = colors[idx % len(colors)]
        path = " ".join(f"{sx(x):.1f},{sy(y):.1f}" for x, y in pts)
        lines.append(f'<polyline fill="none" stroke="{color}" stroke-width="2" points="{path}"/>')
        for x, y in pts:
            lines.append(f'<circle cx="{sx(x):.1f}" cy="{sy(y):.1f}" r="2.4" fill="{color}"><title>{html.escape(metric)} {x_key}={x:g} value={y:.6g}</title></circle>')
        lines.append(f'<text x="{legend_x}" y="{height-24}" fill="{color}" class="legend-text">{html.escape(metric)}</text>')
        legend_x += min(150, 28 + len(metric) * 7)
    lines.append("</svg>")
    return "\n".join(lines)


def table_html(rows: Sequence[Dict[str, Any]], columns: Sequence[str], max_rows: int = 18) -> str:
    if not rows:
        return '<p class="muted">No rows available.</p>'
    head = "".join(f"<th>{html.escape(col)}</th>" for col in columns)
    body: List[str] = []
    for row in rows[:max_rows]:
        cells = []
        for col in columns:
            value = row.get(col, "")
            if is_number(value):
                rendered = f"{float(value):.6g}"
            else:
                rendered = str(value)
            cells.append(f"<td>{html.escape(rendered)}</td>")
        body.append("<tr>" + "".join(cells) + "</tr>")
    if len(rows) > max_rows:
        body.append(f'<tr><td colspan="{len(columns)}" class="muted">... {len(rows) - max_rows} more rows in summary.json</td></tr>')
    return f"<table><thead><tr>{head}</tr></thead><tbody>{''.join(body)}</tbody></table>"


def build_run_summary(run_dir: Path) -> Dict[str, Any]:
    metadata = read_json(run_dir / "metadata.json") or {}
    score = read_json(run_dir / "score.json") or {}
    config = read_json(run_dir / "config.json") or {}
    abort = read_json(run_dir / "abort.json")
    result_path = find_result_json(run_dir)
    result = read_json(result_path) if result_path else None
    metrics, result_config = unwrap_metrics(result)

    train = metrics.get("train", {}) if isinstance(metrics, dict) else {}
    history = numeric_history(train.get("history", []) if isinstance(train, dict) else [])
    if not history:
        history = numeric_history(parse_log_history(run_dir))

    eval_clean = metrics.get("eval_clean", {}) if isinstance(metrics, dict) else {}
    loop_rows = extract_loop_rows(eval_clean)
    hole_rows = extract_hole_rows(metrics.get("eval_by_holes", {}) if isinstance(metrics, dict) else {})
    final_by_hole: List[Dict[str, Any]] = []
    by_hole: Dict[int, List[Dict[str, Any]]] = {}
    for row in hole_rows:
        holes = int(row.get("holes", 0))
        by_hole.setdefault(holes, []).append(row)
    for holes, rows in sorted(by_hole.items()):
        if rows:
            final = sorted(rows, key=lambda item: int(item.get("loop", 0)))[-1].copy()
            final_by_hole.append(final)

    case_links = extract_case_links(run_dir)
    result_rel = rel_link(run_dir, result_path) if result_path else ""
    readme_rel = "README.md" if (run_dir / "README.md").exists() else ""
    primary_score = score.get("score", metadata.get("score"))
    score_key = score.get("score_key", metadata.get("score_key", ""))
    loop_deltas = {}
    for metric in ("path_f1", "path_precision", "path_recall", "path_pred_frac", "label_exact", "blank_acc", "valid_sudoku", "solved_valid_clue"):
        delta = first_last_delta(loop_rows, metric)
        if delta is not None:
            loop_deltas[metric] = delta

    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "run_name": run_dir.name,
        "mode": metadata.get("mode") or config.get("mode") or "",
        "git_sha": metadata.get("git_sha") or config.get("git_sha") or score.get("git_sha") or "",
        "git_dirty": metadata.get("git_dirty", config.get("git_dirty", "")),
        "timestamp_utc": metadata.get("timestamp_utc") or config.get("timestamp_utc") or "",
        "score": primary_score,
        "score_key": score_key,
        "notes": metadata.get("notes", ""),
        "has_abort": abort is not None,
        "abort": abort if isinstance(abort, dict) else {},
        "result_json": result_rel,
        "readme": readme_rel,
        "task": metrics.get("task", {}) if isinstance(metrics, dict) else {},
        "config": result_config or config,
        "history": history,
        "loop_rows": loop_rows,
        "hole_rows": hole_rows,
        "final_by_hole": final_by_hole,
        "loop_deltas": loop_deltas,
        "case_links": case_links,
        "available_artifacts": sorted(
            rel_link(run_dir, path)
            for pattern in ("logs/*.log", "output/*.md", "output/*.json", "config.json", "score.json", "metadata.json", "abort.json")
            for path in run_dir.glob(pattern)
            if path.is_file()
        ),
    }


def render_run_html(summary: Dict[str, Any]) -> str:
    history = summary["history"]
    loop_rows = summary["loop_rows"]
    final_by_hole = summary["final_by_hole"]
    history_metrics = choose_history_metrics(history)
    loop_metrics = choose_loop_metrics(loop_rows)
    hole_metrics = choose_loop_metrics(final_by_hole)

    score = summary.get("score")
    score_text = "" if score in (None, "") else f"{float(score):.6g}" if is_number(score) else str(score)
    title = f"Experiment Dashboard - {summary['run_name']}"
    links = []
    if summary.get("readme"):
        links.append(html_link(summary["readme"], "README"))
    if summary.get("result_json"):
        links.append(html_link(summary["result_json"], "result JSON"))
    links.append(html_link("summary.json", "dashboard data"))
    case_links = summary.get("case_links", [])
    case_items = "".join(
        f"<li>{html.escape(link['kind'])}: {html_link(link['path'], link['path'])}</li>"
        for link in case_links
    )
    if not case_items:
        case_items = '<li class="muted">No per-case trajectories were archived for this run. Future Maze runs now default to saving casebooks.</li>'

    delta_rows = []
    for metric, delta in summary.get("loop_deltas", {}).items():
        delta_rows.append(
            {
                "metric": metric,
                "first_loop": delta["first_x"],
                "first": delta["first"],
                "last_loop": delta["last_x"],
                "last": delta["last"],
                "delta": delta["delta"],
            }
        )

    task = summary.get("task", {})
    task_bits = []
    if isinstance(task, dict):
        for key in ("model", "grid_size", "size", "hidden_size", "layers", "train_loops", "eval_loops", "max_loops", "future_seed_scale", "state_update_mode"):
            if key in task:
                task_bits.append(f"<dt>{html.escape(key)}</dt><dd>{html.escape(str(task[key]))}</dd>")

    html_parts = [
        "<!doctype html>",
        '<html lang="en">',
        "<head>",
        '<meta charset="utf-8">',
        '<meta name="viewport" content="width=device-width, initial-scale=1">',
        f"<title>{html.escape(title)}</title>",
        "<style>",
        "body{font-family:system-ui,-apple-system,Segoe UI,sans-serif;margin:24px;background:#f7f7f4;color:#1f2328}",
        "main{max-width:1180px;margin:auto}.card{background:#fff;border:1px solid #ddd8cf;border-radius:8px;padding:14px;margin:14px 0}",
        "h1{font-size:24px;margin:0 0 8px}h2{font-size:18px;margin:0 0 10px}.muted{color:#6b7280}.links{display:flex;flex-wrap:wrap;gap:10px}",
        "table{border-collapse:collapse;width:100%;font-size:13px}th,td{border-bottom:1px solid #e5e7eb;padding:6px 8px;text-align:left}th{background:#f3f4f6}",
        "dl{display:grid;grid-template-columns:max-content 1fr;gap:4px 12px}dt{font-weight:600}.chart{width:100%;height:auto;background:#fbfbf8;border:1px solid #e5e0d8;border-radius:6px}.axis{stroke:#777;stroke-width:1}.tick,.legend-text{font-size:11px;font-family:ui-monospace,Menlo,monospace}",
        ".pill{display:inline-block;background:#eef2ff;border:1px solid #c7d2fe;border-radius:999px;padding:2px 8px;margin-right:6px;font-size:12px}",
        "</style>",
        "</head>",
        "<body><main>",
        f"<h1>{html.escape(summary['run_name'])}</h1>",
        '<p class="links">' + " ".join(links) + "</p>",
        '<section class="card"><h2>Run Summary</h2>',
        "<dl>",
        f"<dt>mode</dt><dd>{html.escape(str(summary.get('mode', '')))}</dd>",
        f"<dt>score</dt><dd>{html.escape(score_text)} <span class=\"muted\">{html.escape(str(summary.get('score_key', '')))}</span></dd>",
        f"<dt>git sha</dt><dd>{html.escape(str(summary.get('git_sha', '')))}</dd>",
        f"<dt>timestamp</dt><dd>{html.escape(str(summary.get('timestamp_utc', '')))}</dd>",
        f"<dt>aborted</dt><dd>{html.escape(str(summary.get('has_abort', False)))}</dd>",
        "</dl>",
        f"<p>{html.escape(str(summary.get('notes', '')))}</p>" if summary.get("notes") else "",
        "</section>",
    ]
    if task_bits:
        html_parts.extend(['<section class="card"><h2>Task / Model</h2><dl>', "".join(task_bits), "</dl></section>"])
    html_parts.extend(
        [
            '<section class="card"><h2>Training Curve</h2>',
            svg_line_chart(history, "step", history_metrics),
            table_html(history, ["step", "stage", "ce", "total", "loop1", "loop1_ce", "loop_last", "path_f1", "exact"], max_rows=12),
            "</section>",
            '<section class="card"><h2>Loop Readout</h2>',
            svg_line_chart(loop_rows, "loop", loop_metrics),
            table_html(loop_rows, ["loop", "path_f1", "path_precision", "path_recall", "path_pred_frac", "label_exact", "blank_acc", "valid_sudoku", "solved_valid_clue"], max_rows=20),
            "</section>",
        ]
    )
    if delta_rows:
        html_parts.extend(
            [
                '<section class="card"><h2>Loop Delta</h2>',
                table_html(delta_rows, ["metric", "first_loop", "first", "last_loop", "last", "delta"], max_rows=20),
                "</section>",
            ]
        )
    if final_by_hole:
        html_parts.extend(
            [
                '<section class="card"><h2>Difficulty Transfer</h2>',
                svg_line_chart(final_by_hole, "holes", hole_metrics),
                table_html(final_by_hole, ["holes", "loop", "label_exact", "blank_acc", "valid_sudoku", "solved_valid_clue"], max_rows=32),
                "</section>",
            ]
        )
    html_parts.extend(
        [
            '<section class="card"><h2>Case Visualizations</h2><ul>',
            case_items,
            "</ul></section>",
            '<section class="card"><h2>Artifacts</h2>',
            table_html([{"path": path} for path in summary.get("available_artifacts", [])], ["path"], max_rows=30),
            "</section>",
            "</main></body></html>",
        ]
    )
    return "\n".join(part for part in html_parts if part != "")


def build_one(run_dir: Path) -> Dict[str, Any]:
    summary = build_run_summary(run_dir)
    viz_dir = run_dir / "visualizations"
    viz_dir.mkdir(parents=True, exist_ok=True)
    write_json(viz_dir / "summary.json", summary)
    (viz_dir / "index.html").write_text(render_run_html(summary) + "\n", encoding="utf-8")
    return summary


def render_global_index(repo: Path, run_summaries: Sequence[Tuple[Path, Dict[str, Any]]]) -> str:
    rows = []
    for run_dir, summary in sorted(
        run_summaries,
        key=lambda item: str(item[1].get("timestamp_utc") or item[1].get("generated_at_utc") or ""),
        reverse=True,
    ):
        rel_run = rel_link(repo, run_dir)
        score = summary.get("score")
        score_text = "" if score in (None, "") else f"{float(score):.6g}" if is_number(score) else str(score)
        case_count = len(summary.get("case_links", []))
        rows.append(
            {
                "run": html_link(f"{rel_run}/visualizations/index.html", run_dir.name),
                "mode": str(summary.get("mode", "")),
                "score": score_text,
                "score_key": str(summary.get("score_key", "")),
                "cases": case_count,
                "sha": str(summary.get("git_sha", ""))[:7],
                "timestamp": str(summary.get("timestamp_utc", "")),
            }
        )
    body_rows = []
    for row in rows:
        body_rows.append(
            "<tr>"
            f"<td>{row['run']}</td>"
            f"<td>{html.escape(row['mode'])}</td>"
            f"<td>{html.escape(row['score'])}</td>"
            f"<td>{html.escape(row['score_key'])}</td>"
            f"<td>{row['cases']}</td>"
            f"<td>{html.escape(row['sha'])}</td>"
            f"<td>{html.escape(row['timestamp'])}</td>"
            "</tr>"
        )
    return "\n".join(
        [
            "<!doctype html>",
            '<html lang="en"><head><meta charset="utf-8">',
            '<meta name="viewport" content="width=device-width, initial-scale=1">',
            "<title>Experiment Visualization Index</title>",
            "<style>body{font-family:system-ui,-apple-system,Segoe UI,sans-serif;margin:24px;background:#f7f7f4;color:#1f2328}main{max-width:1280px;margin:auto}table{border-collapse:collapse;width:100%;font-size:13px;background:white;border:1px solid #ddd8cf}th,td{border-bottom:1px solid #e5e7eb;padding:7px 8px;text-align:left;vertical-align:top}th{background:#f3f4f6;position:sticky;top:0}a{color:#1d4ed8}</style>",
            "</head><body><main>",
            "<h1>Experiment Visualization Index</h1>",
            f"<p>Generated {html.escape(datetime.now(timezone.utc).isoformat())}. Each run links to a dashboard with curves, loop readouts, and case artifacts when available.</p>",
            "<table><thead><tr><th>run</th><th>mode</th><th>score</th><th>score key</th><th>case artifacts</th><th>sha</th><th>timestamp</th></tr></thead><tbody>",
            "".join(body_rows),
            "</tbody></table></main></body></html>",
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", default=str(Path(__file__).resolve().parents[1]))
    parser.add_argument("--runs-dir", default="")
    parser.add_argument("--run-dir", action="append", default=[])
    parser.add_argument("--index-only", action="store_true")
    args = parser.parse_args()

    repo = Path(args.repo_root).resolve()
    runs_dir = Path(args.runs_dir).resolve() if args.runs_dir else repo / "runs"
    run_dirs = [Path(path).resolve() for path in args.run_dir] if args.run_dir else discover_run_dirs(runs_dir)
    summaries: List[Tuple[Path, Dict[str, Any]]] = []
    if args.index_only:
        run_dirs = discover_run_dirs(runs_dir)
    for run_dir in run_dirs:
        if not run_dir.exists() or not run_dir.is_dir():
            continue
        if args.index_only:
            summary = read_json(run_dir / "visualizations" / "summary.json")
            if not isinstance(summary, dict):
                summary = build_run_summary(run_dir)
        else:
            summary = build_one(run_dir)
        summaries.append((run_dir, summary))
    if runs_dir.exists():
        global_html = render_global_index(repo, summaries if not args.run_dir else [(rd, read_json(rd / "visualizations" / "summary.json") or build_run_summary(rd)) for rd in discover_run_dirs(runs_dir)])
        (runs_dir / "visualization_index.html").write_text(global_html + "\n", encoding="utf-8")
    print(f"built visualizations for {len(summaries)} run(s)")


if __name__ == "__main__":
    main()
