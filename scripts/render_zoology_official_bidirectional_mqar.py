from __future__ import annotations

import argparse
import html
import json
from pathlib import Path
from typing import Any


def curve_svg(curve: list[dict[str, Any]]) -> str:
    width, height = 820, 270
    left, right, top, bottom = 58, 18, 28, 40
    plot_width = width - left - right
    plot_height = height - top - bottom
    if not curve:
        return "<p>No validation curve.</p>"
    max_epoch = max(int(row["epoch"]) for row in curve)

    def x(epoch: int) -> float:
        return left + plot_width * epoch / max(1, max_epoch)

    def y(value: float) -> float:
        return top + plot_height * (1.0 - value)

    points = " ".join(
        f"{x(int(row['epoch'])):.1f},{y(float(row['valid/accuracy'])):.1f}"
        for row in curve
    )
    parts = [f'<svg viewBox="0 0 {width} {height}">']
    for tick in range(5):
        value = tick / 4
        yy = y(value)
        parts.append(
            f'<line x1="{left}" y1="{yy:.1f}" x2="{width-right}" '
            f'y2="{yy:.1f}" class="grid" />'
        )
        parts.append(
            f'<text x="{left-8}" y="{yy+4:.1f}" text-anchor="end" '
            f'class="axis">{value:.2f}</text>'
        )
    parts.append(
        f'<polyline points="{points}" fill="none" stroke="#176b55" '
        'stroke-width="3" />'
    )
    for row in curve:
        xx = x(int(row["epoch"]))
        yy = y(float(row["valid/accuracy"]))
        parts.append(f'<circle cx="{xx:.1f}" cy="{yy:.1f}" r="3" fill="#176b55" />')
    parts.append(
        f'<text x="{left}" y="{height-10}" class="axis">epoch 0</text>'
        f'<text x="{width-right}" y="{height-10}" text-anchor="end" '
        f'class="axis">epoch {max_epoch}</text>'
        "</svg>"
    )
    return "".join(parts)


def case_html(case: dict[str, Any]) -> str:
    rows = []
    for event in case["events"]:
        status = "correct" if event["correct"] else "wrong"
        rows.append(
            "<tr>"
            f"<td>{html.escape(event['direction'])}</td>"
            f"<td>{event['query_position']}</td>"
            f"<td>{event['write_position']}</td>"
            f"<td>{event['distance']:+d}</td>"
            f"<td>{event['key']}</td>"
            f"<td>{event['target']}</td>"
            f'<td class="{status}">{event["prediction"]}</td>'
            "</tr>"
        )
    return (
        '<article class="case">'
        f"<h3>Case {html.escape(case['case_id'])}</h3>"
        f"<p>Future errors {case['future_errors']}; past errors {case['past_errors']}.</p>"
        '<div class="table-wrap"><table><thead><tr>'
        "<th>Direction</th><th>Query</th><th>Write</th><th>Distance</th>"
        "<th>Key</th><th>Target</th><th>Prediction</th>"
        "</tr></thead><tbody>"
        + "".join(rows)
        + "</tbody></table></div></article>"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--visual-dir", type=Path, required=True)
    args = parser.parse_args()
    score = json.loads((args.output_dir / "score.json").read_text())
    cases = json.loads((args.output_dir / "cases.json").read_text())
    hardest = sorted(
        cases,
        key=lambda row: (
            -row["future_errors"],
            -row["past_errors"],
            row["case_index"],
        ),
    )[:12]
    args.visual_dir.mkdir(parents=True, exist_ok=True)
    (args.visual_dir / "hardest_cases.json").write_text(
        json.dumps(hardest, indent=2, sort_keys=True) + "\n"
    )
    (args.visual_dir / "summary.json").write_text(
        json.dumps(
            {
                "metrics": score["metrics"],
                "opening_epoch_at_0.90": score["opening_epoch_at_0.90"],
                "opening_epoch_at_0.99": score["opening_epoch_at_0.99"],
                "parameters": score["parameters"],
                "warmed_step_benchmark": score["warmed_step_benchmark"],
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    metrics = score["metrics"]
    bench = score["warmed_step_benchmark"]
    document = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Official Zoology bidirectional MHA carrier</title>
<style>
:root {{ --ink:#18211d; --muted:#5d6762; --line:#d8dedb; --good:#176b55; --bad:#a4331b; }}
* {{ box-sizing:border-box; }} body {{ margin:0; color:var(--ink); background:#fff; font-family:Inter,ui-sans-serif,system-ui,sans-serif; letter-spacing:0; }}
main {{ width:min(1080px,calc(100% - 28px)); margin:28px auto 64px; }}
h1 {{ font-size:28px; margin:0 0 8px; }} h2 {{ font-size:19px; margin:30px 0 12px; }} h3 {{ font-size:14px; margin:0 0 4px; }} p {{ color:var(--muted); margin:0 0 12px; }}
.metrics {{ display:grid; grid-template-columns:repeat(4,minmax(0,1fr)); border-block:1px solid var(--line); }}
.metric {{ padding:16px 10px; }} .metric b {{ display:block; font-size:24px; }} .metric span {{ color:var(--muted); font-size:12px; }}
.chart {{ border:1px solid var(--line); border-radius:6px; padding:8px; }} svg {{ width:100%; height:auto; }} .grid {{ stroke:#e4e9e6; }} .axis {{ font-size:11px; fill:#67716c; }}
.table-wrap {{ overflow-x:auto; }} table {{ width:100%; border-collapse:collapse; font-size:12px; }} th,td {{ border-bottom:1px solid var(--line); padding:8px; text-align:right; }} th:first-child,td:first-child {{ text-align:left; }}
.case {{ border-top:1px solid var(--line); padding:15px 0 6px; }} .correct {{ background:#e4f4ed; color:var(--good); font-weight:700; }} .wrong {{ background:#fde8e2; color:var(--bad); font-weight:700; }}
@media(max-width:720px) {{ .metrics {{ grid-template-columns:1fr 1fr; }} }}
</style></head><body><main>
<h1>Official Zoology MHA, causal mask removed</h1>
<p>Carrier calibration only. The parameters, initialization, data, optimizer and trainer match upstream Zoology; the triangular attention mask is the only semantic change.</p>
<section class="metrics">
<div class="metric"><b>{metrics['past']['accuracy']:.4f}</b><span>past accuracy</span></div>
<div class="metric"><b>{metrics['future']['accuracy']:.4f}</b><span>future accuracy</span></div>
<div class="metric"><b>{metrics['joint_exact']:.4f}</b><span>joint exact</span></div>
<div class="metric"><b>{score['opening_epoch_at_0.90']}</b><span>opening epoch at 0.90</span></div>
</section>
<h2>Validation opening curve</h2><div class="chart">{curve_svg(score['valid_curve'])}</div>
<h2>Cost</h2><div class="table-wrap"><table><thead><tr><th>Parameters</th><th>Completed epochs</th><th>Train tokens</th><th>Tokens/s</th><th>Peak memory</th></tr></thead><tbody><tr><td>{score['parameters']:,}</td><td>{score['completed_epochs']}</td><td>{score['train_tokens']:,}</td><td>{bench['tokens_per_sec']:,.0f}</td><td>{bench['peak_cuda_mem_bytes']/2**20:.1f} MiB</td></tr></tbody></table></div>
<h2>Hardest same-sequence cases</h2>{''.join(case_html(case) for case in hardest)}
</main></body></html>"""
    (args.visual_dir / "index.html").write_text(document)
    print(json.dumps({"visualized_cases": len(hardest)}, sort_keys=True))


if __name__ == "__main__":
    main()
