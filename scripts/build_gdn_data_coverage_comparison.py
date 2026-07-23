#!/usr/bin/env python3
"""Build the matched control-vs-unseen FutureSeed-GDN comparison dashboard."""

from __future__ import annotations

import argparse
import html
import json
import os
from pathlib import Path
from typing import Any


BUCKETS = ("b46_50", "b51_55", "b56_64")
BUCKET_LABELS = {
    "b46_50": "46-50 blanks",
    "b51_55": "51-55 blanks",
    "b56_64": "56-64 blanks",
}


def read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def metric(data: dict[str, Any], *path: str) -> float:
    value: Any = data
    for key in path:
        value = value[key]
    return float(value)


def summarize(label: str, result_path: Path, out_dir: Path) -> dict[str, Any]:
    data = read_json(result_path)
    metrics = data["metrics"]
    run_dir = result_path.parent.parent
    loops = {
        f"loop{loop}": {
            "exact": metric(metrics, "eval_clean", f"loop{loop}", "label_exact"),
            "blank_acc": metric(metrics, "eval_clean", f"loop{loop}", "blank_acc"),
        }
        for loop in range(1, 6)
    }
    official = {
        bucket: {
            "exact": metric(
                metrics,
                "official_eval_by_blank_range",
                bucket,
                "eval_clean",
                "loop5",
                "label_exact",
            ),
            "blank_acc": metric(
                metrics,
                "official_eval_by_blank_range",
                bucket,
                "eval_clean",
                "loop5",
                "blank_acc",
            ),
        }
        for bucket in BUCKETS
    }
    case_bank = {}
    for bucket in BUCKETS:
        case_path = (
            run_dir
            / "output"
            / "case_bank"
            / f"official_{bucket}"
            / "cases.json"
        )
        if case_path.exists():
            summary = read_json(case_path)["summary"]
            case_bank[bucket] = {
                "exact": float(summary["final_exact"]),
                "blank_acc": float(summary["final_blank_acc"]),
                "eval_n": int(summary["eval_n"]),
            }
    train = metrics.get("train", {})
    return {
        "label": label,
        "run": run_dir.name,
        "result_path": str(result_path),
        "visualization": os.path.relpath(
            run_dir / "visualizations" / "index.html", out_dir
        ),
        "loops": loops,
        "official": official,
        "case_bank": case_bank,
        "train_sec": train.get("train_sec"),
        "cuda_max_memory_allocated_mb": train.get(
            "cuda_max_memory_allocated_mb"
        ),
        "train_ce_loss": train.get("train_ce_loss"),
        "source_sha": data.get("args", {}).get("git_sha"),
    }


def signed(value: float) -> str:
    return f"{value:+.4f}"


def line_chart(arms: list[dict[str, Any]]) -> str:
    width, height = 760, 300
    left, right, top, bottom = 58, 24, 24, 48
    plot_w = width - left - right
    plot_h = height - top - bottom
    colors = {"parent": "#6b7280", "control": "#2563eb", "unseen": "#dc2626"}
    y_max = max(
        0.55,
        max(arm["loops"][f"loop{i}"]["exact"] for arm in arms for i in range(1, 6))
        + 0.03,
    )

    def x(loop: int) -> float:
        return left + (loop - 1) * plot_w / 4

    def y(value: float) -> float:
        return top + (y_max - value) * plot_h / y_max

    grid = []
    for value in (0.0, 0.1, 0.2, 0.3, 0.4, 0.5):
        if value > y_max:
            continue
        yy = y(value)
        grid.append(
            f'<line x1="{left}" y1="{yy:.1f}" x2="{width-right}" '
            f'y2="{yy:.1f}" stroke="#d1d5db" stroke-width="1"/>'
            f'<text x="{left-10}" y="{yy+4:.1f}" text-anchor="end" '
            f'font-size="12" fill="#4b5563">{value:.1f}</text>'
        )
    series = []
    for arm in arms:
        points = [
            (x(loop), y(arm["loops"][f"loop{loop}"]["exact"]))
            for loop in range(1, 6)
        ]
        point_text = " ".join(f"{px:.1f},{py:.1f}" for px, py in points)
        color = colors[arm["label"]]
        series.append(
            f'<polyline points="{point_text}" fill="none" stroke="{color}" '
            f'stroke-width="3"/>'
        )
        for loop, (px, py) in enumerate(points, 1):
            value = arm["loops"][f"loop{loop}"]["exact"]
            series.append(
                f'<circle cx="{px:.1f}" cy="{py:.1f}" r="4" fill="{color}"/>'
                f'<title>{html.escape(arm["label"])} loop{loop}: {value:.4f}</title>'
            )
    x_labels = "".join(
        f'<text x="{x(loop):.1f}" y="{height-18}" text-anchor="middle" '
        f'font-size="13" fill="#374151">loop {loop}</text>'
        for loop in range(1, 6)
    )
    legend = "".join(
        f'<span><i style="background:{colors[arm["label"]]}"></i>'
        f'{html.escape(arm["label"])}</span>'
        for arm in arms
    )
    return (
        f'<div class="legend">{legend}</div><svg viewBox="0 0 {width} {height}" '
        f'role="img" aria-label="Full-board exact accuracy by loop">'
        f'{"".join(grid)}{"".join(series)}{x_labels}</svg>'
    )


def render(payload: dict[str, Any]) -> str:
    arms = payload["arms"]
    control = next(arm for arm in arms if arm["label"] == "control")
    unseen = next(arm for arm in arms if arm["label"] == "unseen")
    decision = payload["decision"]

    official_rows = []
    for bucket in BUCKETS:
        parent = arms[0]["official"][bucket]
        ctl = control["official"][bucket]
        uns = unseen["official"][bucket]
        official_rows.append(
            "<tr>"
            f"<td>{BUCKET_LABELS[bucket]}</td>"
            f"<td>{parent['exact']:.4f}</td>"
            f"<td>{ctl['exact']:.4f}</td>"
            f"<td>{uns['exact']:.4f}</td>"
            f"<td>{signed(uns['exact'] - ctl['exact'])}</td>"
            f"<td>{ctl['blank_acc']:.4f}</td>"
            f"<td>{uns['blank_acc']:.4f}</td>"
            "</tr>"
        )

    case_rows = []
    for bucket in ("b51_55", "b56_64"):
        parent = arms[0]["case_bank"][bucket]
        ctl = control["case_bank"][bucket]
        uns = unseen["case_bank"][bucket]
        case_rows.append(
            "<tr>"
            f"<td>{BUCKET_LABELS[bucket]}</td>"
            f"<td>{parent['exact']:.4f}</td>"
            f"<td>{ctl['exact']:.4f}</td>"
            f"<td>{uns['exact']:.4f}</td>"
            f"<td>{signed(uns['exact'] - ctl['exact'])}</td>"
            f"<td>{uns['eval_n']}</td>"
            "</tr>"
        )

    run_rows = []
    for arm in arms:
        train_sec = arm["train_sec"]
        memory = arm["cuda_max_memory_allocated_mb"]
        run_rows.append(
            "<tr>"
            f"<td>{html.escape(arm['label'])}</td>"
            f"<td>{arm['loops']['loop5']['exact']:.4f}</td>"
            f"<td>{arm['loops']['loop5']['blank_acc']:.4f}</td>"
            f"<td>{'-' if train_sec is None else f'{float(train_sec):.1f}'}</td>"
            f"<td>{'-' if memory is None else f'{float(memory):.0f}'}</td>"
            f'<td><a href="{html.escape(arm["visualization"])}">'
            "hard-case visualizations</a></td>"
            "</tr>"
        )

    manifest = payload["manifest"]
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>GDN Data-Coverage Continuation</title>
<style>
:root {{ color-scheme: light; font-family: Inter, ui-sans-serif, system-ui, sans-serif; }}
body {{ margin: 0; background: #f3f4f6; color: #111827; }}
main {{ max-width: 1120px; margin: 0 auto; padding: 28px 20px 60px; }}
h1 {{ margin: 0 0 8px; font-size: 30px; letter-spacing: 0; }}
h2 {{ margin-top: 30px; font-size: 20px; letter-spacing: 0; }}
p {{ line-height: 1.6; }}
.band {{ background: white; border: 1px solid #d1d5db; border-radius: 6px;
  padding: 18px; margin-top: 16px; }}
.kpis {{ display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 10px; }}
.kpi {{ border-left: 4px solid #2563eb; padding: 8px 12px; background: #f9fafb; }}
.kpi b {{ display: block; font-size: 24px; }}
.kpi span {{ color: #4b5563; font-size: 13px; }}
table {{ width: 100%; border-collapse: collapse; font-variant-numeric: tabular-nums; }}
th, td {{ text-align: right; border-bottom: 1px solid #e5e7eb; padding: 10px 8px; }}
th:first-child, td:first-child {{ text-align: left; }}
th {{ color: #374151; font-size: 13px; }}
.legend {{ display: flex; gap: 18px; margin-bottom: 8px; }}
.legend span {{ display: inline-flex; align-items: center; gap: 6px; }}
.legend i {{ display: inline-block; width: 18px; height: 4px; }}
svg {{ width: 100%; height: auto; }}
.decision {{ border-left-color: {"#15803d" if decision["success"] else "#b45309"}; }}
code {{ background: #e5e7eb; padding: 2px 5px; border-radius: 3px; }}
a {{ color: #1d4ed8; }}
@media (max-width: 760px) {{ .kpis {{ grid-template-columns: 1fr 1fr; }}
  table {{ font-size: 12px; }} th, td {{ padding: 8px 4px; }} }}
</style>
</head>
<body><main>
<h1>Exact unseen-data coverage continuation</h1>
<p>Same step30000 FutureSeed-GDN checkpoint, optimizer, RNG state, model,
five-loop all-loop CE, batch, learning rate, CUDA kernel, and evaluation.
Only the training row pool differs.</p>
<section class="band kpis">
<div class="kpi"><b>{manifest["target_rows"]:,}</b><span>51-64 blank rows</span></div>
<div class="kpi"><b>{manifest["target_seen_fraction"]:.1%}</b><span>actually seen by step30000</span></div>
<div class="kpi"><b>{manifest["output_rows"]:,}</b><span>matched unseen rows used</span></div>
<div class="kpi"><b>{signed(decision["hard_delta"])}</b><span>unseen-control hard exact</span></div>
</section>
<section class="band decision">
<h2>Predeclared decision</h2>
<p><b>{html.escape(decision["label"])}</b>. {html.escape(decision["reason"])}</p>
</section>
<h2>Loop scaling</h2>
<section class="band">{line_chart(arms)}</section>
<h2>Official test distribution</h2>
<section class="band"><table><thead><tr>
<th>range</th><th>parent exact</th><th>control exact</th><th>unseen exact</th>
<th>unseen-control</th><th>control blank acc</th><th>unseen blank acc</th>
</tr></thead><tbody>{"".join(official_rows)}</tbody></table></section>
<h2>Second fixed evaluation batch</h2>
<section class="band"><p>This separate 256-board batch checks whether the
hard-tail direction repeats on different held-out examples.</p>
<table><thead><tr>
<th>range</th><th>parent exact</th><th>control exact</th><th>unseen exact</th>
<th>unseen-control</th><th>boards</th>
</tr></thead><tbody>{"".join(case_rows)}</tbody></table></section>
<h2>Run provenance</h2>
<section class="band"><table><thead><tr>
<th>arm</th><th>mixed exact</th><th>mixed blank acc</th><th>train sec</th>
<th>peak MB</th><th>cases</th>
</tr></thead><tbody>{"".join(run_rows)}</tbody></table>
<p>Index SHA256: <code>{manifest["output_indices_sha256"]}</code><br>
Checkpoint SHA256: <code>{manifest["checkpoint_sha256"]}</code><br>
RNG reconstruction matched checkpoint: <b>{str(manifest["rng_matches_checkpoint"]).lower()}</b>.</p>
</section>
</main></body></html>
"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--control", type=Path, required=True)
    parser.add_argument("--unseen", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    arms = [
        summarize("parent", args.parent, args.out_dir),
        summarize("control", args.control, args.out_dir),
        summarize("unseen", args.unseen, args.out_dir),
    ]
    control = arms[1]
    unseen = arms[2]
    mixed_delta = (
        unseen["loops"]["loop5"]["exact"] - control["loops"]["loop5"]["exact"]
    )
    hard_delta = (
        unseen["official"]["b56_64"]["exact"]
        - control["official"]["b56_64"]["exact"]
    )
    mid_delta = (
        unseen["official"]["b51_55"]["exact"]
        - control["official"]["b51_55"]["exact"]
    )
    case_hard_delta = (
        unseen["case_bank"]["b56_64"]["exact"]
        - control["case_bank"]["b56_64"]["exact"]
    )
    case_mid_delta = (
        unseen["case_bank"]["b51_55"]["exact"]
        - control["case_bank"]["b51_55"]["exact"]
    )
    success = (mixed_delta >= 0.03 or hard_delta >= 0.03) and mid_delta >= -0.03
    low_signal = max(abs(mixed_delta), abs(hard_delta), abs(mid_delta)) < 0.01
    if success:
        label = "continue independent-data coverage"
        reason = "The unseen arm passes the effect-size and preservation gates."
    elif low_signal:
        label = "stop sampler engineering"
        reason = "All primary matched deltas are below 0.01 in magnitude."
    else:
        label = "do not claim a data-coverage win"
        reason = "The unseen arm does not pass the predeclared mechanism gate."

    payload = {
        "schema_version": 1,
        "arms": arms,
        "manifest": read_json(args.manifest),
        "decision": {
            "label": label,
            "reason": reason,
            "success": success,
            "low_signal": low_signal,
            "mixed_delta": mixed_delta,
            "hard_delta": hard_delta,
            "mid_delta": mid_delta,
            "case_hard_delta": case_hard_delta,
            "case_mid_delta": case_mid_delta,
        },
    }
    with (args.out_dir / "comparison.json").open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")
    (args.out_dir / "index.html").write_text(render(payload), encoding="utf-8")
    (args.out_dir / "README.md").write_text(
        "# GDN data-coverage comparison\n\n"
        f"Decision: **{label}**.\n\n"
        f"- mixed loop5 unseen-control: `{mixed_delta:+.4f}`\n"
        f"- official 56-64 unseen-control: `{hard_delta:+.4f}`\n"
        f"- official 51-55 unseen-control: `{mid_delta:+.4f}`\n\n"
        f"- second-batch 56-64 unseen-control: `{case_hard_delta:+.4f}`\n"
        f"- second-batch 51-55 unseen-control: `{case_mid_delta:+.4f}`\n\n"
        "Open `index.html` for loop curves, official blank-range results, "
        "provenance, and hard-case links.\n",
        encoding="utf-8",
    )
    print(json.dumps(payload["decision"], sort_keys=True))


if __name__ == "__main__":
    main()
