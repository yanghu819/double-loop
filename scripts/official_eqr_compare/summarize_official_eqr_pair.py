#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import html
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List


def load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def rel_link(target: Path, out_dir: Path) -> str:
    try:
        return target.resolve().relative_to(out_dir.resolve()).as_posix()
    except ValueError:
        return target.resolve().as_posix()


def step_rows(base: Dict[str, Any], futureseed: Dict[str, Any]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for step in [str(s) for s in base["steps"]]:
        b = base["aggregate_by_step"][step]
        f = futureseed["aggregate_by_step"][step]
        rows.append(
            {
                "loop": step,
                "base_token_acc": b["token_acc"],
                "fs_token_acc": f["token_acc"],
                "d_token_acc": f["token_acc"] - b["token_acc"],
                "base_exact": b["exact"],
                "fs_exact": f["exact"],
                "d_exact": f["exact"] - b["exact"],
                "base_path_f1": b["path_f1"],
                "fs_path_f1": f["path_f1"],
                "d_path_f1": f["path_f1"] - b["path_f1"],
                "base_path_precision": b["path_precision"],
                "fs_path_precision": f["path_precision"],
                "d_path_precision": f["path_precision"] - b["path_precision"],
                "base_path_recall": b["path_recall"],
                "fs_path_recall": f["path_recall"],
                "d_path_recall": f["path_recall"] - b["path_recall"],
                "base_pred_path_frac": b["pred_path_frac"],
                "fs_pred_path_frac": f["pred_path_frac"],
                "d_pred_path_frac": f["pred_path_frac"] - b["pred_path_frac"],
                "base_fp": b["path_fp"],
                "fs_fp": f["path_fp"],
                "d_fp": f["path_fp"] - b["path_fp"],
                "base_fn": b["path_fn"],
                "fs_fn": f["path_fn"],
                "d_fn": f["path_fn"] - b["path_fn"],
            }
        )
    return rows


def fmt(x: Any, digits: int = 4) -> str:
    if isinstance(x, float):
        return f"{x:.{digits}f}"
    return str(x)


def render_index(
    out_dir: Path,
    experiment: str,
    base: Dict[str, Any],
    futureseed: Dict[str, Any],
    rows: List[Dict[str, Any]],
    base_dir: Path,
    futureseed_dir: Path,
    notes: str,
) -> str:
    base_href = html.escape(rel_link(base_dir / "index.html", out_dir))
    fs_href = html.escape(rel_link(futureseed_dir / "index.html", out_dir))
    final = rows[-1]
    style = """
body{font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;margin:0;background:#f6f6f3;color:#202124}
main{max-width:1200px;margin:0 auto;padding:28px}
h1{font-size:26px;margin:0 0 8px} h2{font-size:18px;margin:24px 0 10px}
p{line-height:1.45}.note{background:#fff;border-left:5px solid #475569;padding:11px 13px;margin:14px 0}
.cards{display:grid;grid-template-columns:repeat(4,1fr);gap:10px;margin:16px 0}.card{background:#fff;border:1px solid #ddd;border-radius:8px;padding:10px}.card b{display:block;font-size:21px}
table{width:100%;border-collapse:collapse;background:#fff;border:1px solid #ddd;margin:10px 0 18px}
th,td{border-bottom:1px solid #e5e5e5;padding:8px;text-align:left;font-size:13px}th{background:#eee}
a{color:#155eef;text-decoration:none}code{background:#eee;padding:1px 4px;border-radius:4px}.pos{color:#067647}.neg{color:#b42318}
"""
    body_rows = []
    for row in rows:
        cls = "pos" if row["d_path_f1"] > 0 else "neg" if row["d_path_f1"] < 0 else ""
        body_rows.append(
            "<tr>"
            f"<td>loop{html.escape(str(row['loop']))}</td>"
            f"<td>{fmt(row['base_path_f1'])}</td><td>{fmt(row['fs_path_f1'])}</td>"
            f"<td class='{cls}'>{fmt(row['d_path_f1'], 5)}</td>"
            f"<td>{fmt(row['base_path_precision'])}</td><td>{fmt(row['fs_path_precision'])}</td>"
            f"<td>{fmt(row['base_path_recall'])}</td><td>{fmt(row['fs_path_recall'])}</td>"
            f"<td>{fmt(row['base_pred_path_frac'])}</td><td>{fmt(row['fs_pred_path_frac'])}</td>"
            f"<td>{fmt(row['base_fp'], 1)}</td><td>{fmt(row['fs_fp'], 1)}</td>"
            f"<td>{fmt(row['base_fn'], 1)}</td><td>{fmt(row['fs_fn'], 1)}</td>"
            "</tr>"
        )
    return f"""<!doctype html>
<html><head><meta charset="utf-8"><title>{html.escape(experiment)}</title><style>{style}</style></head>
<body><main>
<h1>{html.escape(experiment)}</h1>
<p>Matched official EqR Maze comparison: base condition versus the same code path patched with FutureSeed. Metrics below are path-aware and computed from official test cases with no selector, search, repair, or maze-specific postprocessing.</p>
<div class="note">{html.escape(notes)}</div>
<div class="cards">
<div class="card"><b>{fmt(final['base_path_f1'])}</b><span>base final path F1</span></div>
<div class="card"><b>{fmt(final['fs_path_f1'])}</b><span>FutureSeed final path F1</span></div>
<div class="card"><b>{fmt(final['d_path_f1'], 5)}</b><span>FutureSeed delta</span></div>
<div class="card"><b>{int(base['processed_cases'])}</b><span>test cases</span></div>
</div>
<p><a href="{base_href}">Open base hard-case visualization</a> | <a href="{fs_href}">Open FutureSeed hard-case visualization</a></p>
<h2>Loop Metrics</h2>
<table><thead><tr><th>loop</th><th>base F1</th><th>FS F1</th><th>delta F1</th><th>base precision</th><th>FS precision</th><th>base recall</th><th>FS recall</th><th>base pred frac</th><th>FS pred frac</th><th>base FP</th><th>FS FP</th><th>base FN</th><th>FS FN</th></tr></thead><tbody>
{''.join(body_rows)}
</tbody></table>
<h2>Provenance</h2>
<table><tbody>
<tr><th>base checkpoint</th><td><code>{html.escape(base['checkpoint'])}</code></td></tr>
<tr><th>FutureSeed checkpoint</th><td><code>{html.escape(futureseed['checkpoint'])}</code></td></tr>
<tr><th>base repo</th><td><code>{html.escape(base['repo'])}</code></td></tr>
<tr><th>FutureSeed repo</th><td><code>{html.escape(futureseed['repo'])}</code></td></tr>
</tbody></table>
</main></body></html>"""


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize a matched official EqR Maze case-visualization pair.")
    parser.add_argument("--base-cases", required=True, type=Path)
    parser.add_argument("--futureseed-cases", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--notes", default="")
    args = parser.parse_args()

    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    base_cases = args.base_cases.resolve()
    futureseed_cases = args.futureseed_cases.resolve()
    base = load_json(base_cases)
    futureseed = load_json(futureseed_cases)
    rows = step_rows(base, futureseed)

    summary = {
        "experiment": args.experiment,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "notes": args.notes,
        "base": {
            "run_name": base["run_name"],
            "checkpoint": base["checkpoint"],
            "checkpoint_step": base.get("checkpoint_step"),
            "processed_cases": base["processed_cases"],
            "visual_dir": str(base_cases.parent),
        },
        "futureseed": {
            "run_name": futureseed["run_name"],
            "checkpoint": futureseed["checkpoint"],
            "checkpoint_step": futureseed.get("checkpoint_step"),
            "processed_cases": futureseed["processed_cases"],
            "visual_dir": str(futureseed_cases.parent),
        },
        "path_case_summary": rows,
        "decision": "",
    }
    final = rows[-1]
    if final["base_path_f1"] < 0.01 and final["fs_path_f1"] < 0.01:
        summary["decision"] = "Both official EqR variants mostly fail to emit PATH under this budget; token accuracy is not a valid Maze success signal."
    elif final["d_path_f1"] > 0.02:
        summary["decision"] = "FutureSeed improves path-aware Maze behavior under the matched official EqR protocol."
    elif final["d_path_f1"] < -0.02:
        summary["decision"] = "FutureSeed hurts path-aware Maze behavior under the matched official EqR protocol."
    else:
        summary["decision"] = "FutureSeed does not clearly beat the matched base condition on path-aware Maze behavior under this budget."

    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_csv(out_dir / "path_case_summary.csv", rows)
    (out_dir / "index.html").write_text(
        render_index(out_dir, args.experiment, base, futureseed, rows, base_cases.parent, futureseed_cases.parent, args.notes),
        encoding="utf-8",
    )
    readme = (
        f"# {args.experiment}\n\n"
        f"{args.notes}\n\n"
        f"Decision: {summary['decision']}\n\n"
        f"Final loop path F1: base {final['base_path_f1']:.6f}, "
        f"FutureSeed {final['fs_path_f1']:.6f}, delta {final['d_path_f1']:.6f}.\n"
    )
    (out_dir / "README.md").write_text(readme, encoding="utf-8")
    print(json.dumps({"out_dir": str(out_dir), "decision": summary["decision"]}, indent=2))


if __name__ == "__main__":
    main()
