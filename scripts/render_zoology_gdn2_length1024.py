from __future__ import annotations

import argparse
import html
import json
from pathlib import Path
from typing import Any


ARMS = ("causal_gdn2", "future_seed_gdn2")
LABELS = {
    "causal_gdn2": "Causal GDN2",
    "future_seed_gdn2": "GDN2 + FutureSeed",
}
COLORS = {"causal_gdn2": "#b54125", "future_seed_gdn2": "#14745d"}


def line_chart(
    comparison: dict[str, Any],
    *,
    value_getter,
    title: str,
    percent: bool,
) -> str:
    lengths = [64, 1024]
    width, height = 760, 270
    left, right, top, bottom = 62, 20, 34, 42
    plot_width = width - left - right
    plot_height = height - top - bottom
    values = [
        float(value_getter(comparison["lengths"][str(length)]["arms"][arm]))
        for length in lengths
        for arm in ARMS
    ]
    maximum = 1.0 if percent else max(values) * 1.08

    def x(index: int) -> float:
        return left + plot_width * index

    def y(value: float) -> float:
        return top + plot_height * (maximum - value) / maximum

    parts = [
        f'<svg viewBox="0 0 {width} {height}" role="img" aria-label="{html.escape(title)}">',
        f'<text x="{left}" y="20" class="chart-title">{html.escape(title)}</text>',
    ]
    for tick in range(5):
        value = maximum * tick / 4
        yy = y(value)
        label = f"{value * 100:.0f}%" if percent else f"{value:,.0f}"
        parts.append(
            f'<line x1="{left}" y1="{yy:.1f}" x2="{width-right}" '
            f'y2="{yy:.1f}" class="grid" />'
        )
        parts.append(
            f'<text x="{left-9}" y="{yy+4:.1f}" text-anchor="end" '
            f'class="axis">{label}</text>'
        )
    for index, length in enumerate(lengths):
        xx = x(index)
        parts.append(
            f'<text x="{xx:.1f}" y="{height-15}" text-anchor="middle" '
            f'class="axis">{length}</text>'
        )
    for arm in ARMS:
        points = [
            (
                x(index),
                y(
                    float(
                        value_getter(
                            comparison["lengths"][str(length)]["arms"][arm]
                        )
                    )
                ),
            )
            for index, length in enumerate(lengths)
        ]
        parts.append(
            f'<polyline points="{" ".join(f"{xx:.1f},{yy:.1f}" for xx, yy in points)}" '
            f'fill="none" stroke="{COLORS[arm]}" stroke-width="3" />'
        )
        parts.extend(
            f'<circle cx="{xx:.1f}" cy="{yy:.1f}" r="4" fill="{COLORS[arm]}" />'
            for xx, yy in points
        )
    parts.append("</svg>")
    return "".join(parts)


def merge_cases(output_dir: Path, length: int, limit: int = 8) -> list[dict[str, Any]]:
    by_arm = {}
    for arm in ARMS:
        cases = json.loads(
            (output_dir / f"length_{length}" / arm / "cases.json").read_text()
        )
        by_arm[arm] = {case["case_id"]: case for case in cases}
    shared_ids = set(by_arm[ARMS[0]]) & set(by_arm[ARMS[1]])
    merged = [
        {
            "case_id": case_id,
            "sequence_length": length,
            "arms": {arm: by_arm[arm][case_id] for arm in ARMS},
        }
        for case_id in shared_ids
    ]
    merged.sort(
        key=lambda row: (
            -row["arms"]["future_seed_gdn2"]["future_errors"],
            -row["arms"]["causal_gdn2"]["future_errors"],
            row["case_id"],
        )
    )
    return merged[:limit]


def metric_rows(comparison: dict[str, Any]) -> str:
    rows = []
    for length in (64, 1024):
        for arm in ARMS:
            score = comparison["lengths"][str(length)]["arms"][arm]
            metrics = score["metrics"]
            benchmark = score["warmed_step_benchmark"]
            rows.append(
                "<tr>"
                f"<td>{length}</td><td>{html.escape(LABELS[arm])}</td>"
                f"<td>{metrics['past']['accuracy']:.4f}</td>"
                f"<td>{metrics['future']['accuracy']:.4f}</td>"
                f"<td>{metrics['joint_exact']:.4f}</td>"
                f"<td>{metrics['future']['ce']:.4f}</td>"
                f"<td>{benchmark['tokens_per_sec']:,.0f}</td>"
                f"<td>{benchmark['peak_cuda_mem_bytes']/2**20:.1f} MiB</td>"
                "</tr>"
            )
    return "".join(rows)


def case_html(case: dict[str, Any]) -> str:
    fs_case = case["arms"]["future_seed_gdn2"]
    rows = []
    for index, event in enumerate(fs_case["events"]):
        predictions = []
        for arm in ARMS:
            arm_event = case["arms"][arm]["events"][index]
            css = "correct" if arm_event["correct"] else "wrong"
            predictions.append(
                f'<td class="{css}">{arm_event["prediction"]}</td>'
            )
        rows.append(
            "<tr>"
            f"<td>{event['direction']}</td><td>{event['query_position']}</td>"
            f"<td>{event['write_position']}</td><td>{event['distance']:+d}</td>"
            f"<td>{event['key']}</td><td>{event['target']}</td>"
            + "".join(predictions)
            + "</tr>"
        )
    summary = " / ".join(
        f"{LABELS[arm]} future errors {case['arms'][arm]['future_errors']}, "
        f"past errors {case['arms'][arm]['past_errors']}"
        for arm in ARMS
    )
    return (
        '<article class="case">'
        f"<h3>Case {html.escape(case['case_id'])}</h3><p>{html.escape(summary)}</p>"
        '<div class="table-wrap"><table><thead><tr>'
        "<th>Direction</th><th>Query</th><th>Write</th><th>Distance</th>"
        f"<th>Key</th><th>Target</th><th>{LABELS[ARMS[0]]}</th>"
        f"<th>{LABELS[ARMS[1]]}</th></tr></thead><tbody>"
        + "".join(rows)
        + "</tbody></table></div></article>"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--visual-dir", type=Path, required=True)
    args = parser.parse_args()
    comparison = json.loads((args.output_dir / "comparison.json").read_text())
    hardest = {str(length): merge_cases(args.output_dir, length) for length in (64, 1024)}
    args.visual_dir.mkdir(parents=True, exist_ok=True)
    (args.visual_dir / "hardest_cases.json").write_text(
        json.dumps(hardest, indent=2, sort_keys=True) + "\n"
    )
    (args.visual_dir / "summary.json").write_text(
        json.dumps(
            {
                "registered_endpoint_gate": comparison["registered_endpoint_gate"],
                "systems_scaling": comparison["systems_scaling"],
                "metrics": {
                    length: {
                        arm: comparison["lengths"][length]["arms"][arm]["metrics"]
                        for arm in ARMS
                    }
                    for length in ("64", "1024")
                },
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    legend = "".join(
        f'<span><i style="background:{COLORS[arm]}"></i>{LABELS[arm]}</span>'
        for arm in ARMS
    )
    case_sections = []
    for length in (64, 1024):
        case_sections.append(f"<h2>Hardest same-sequence cases, length {length}</h2>")
        case_sections.extend(case_html(case) for case in hardest[str(length)])
    gate = comparison["registered_endpoint_gate"]
    document = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>GDN2 FutureSeed length endpoint</title><style>
:root {{ --ink:#18211d; --muted:#5b6761; --line:#d8dedb; }} * {{ box-sizing:border-box; }}
body {{ margin:0; color:var(--ink); background:#fff; font-family:Inter,ui-sans-serif,system-ui,sans-serif; letter-spacing:0; }}
main {{ width:min(1180px,calc(100% - 30px)); margin:28px auto 64px; }} h1 {{ font-size:29px; margin:0 0 8px; }} h2 {{ font-size:20px; margin:32px 0 12px; }} h3 {{ font-size:14px; margin:0 0 4px; }} p {{ color:var(--muted); margin:0 0 12px; }}
.gate {{ border-block:1px solid var(--line); padding:14px 0; font-weight:700; }} .legend {{ display:flex; gap:20px; margin:16px 0; }} .legend span {{ display:flex; gap:7px; align-items:center; }} .legend i {{ width:12px; height:12px; border-radius:2px; }}
.charts {{ display:grid; grid-template-columns:1fr 1fr; gap:16px; }} .chart {{ border:1px solid var(--line); border-radius:6px; padding:8px; overflow:hidden; }} svg {{ width:100%; height:auto; }} .chart-title {{ font-size:14px; font-weight:700; }} .axis {{ font-size:11px; fill:#67716c; }} .grid {{ stroke:#e4e9e6; }}
.table-wrap {{ overflow-x:auto; }} table {{ width:100%; border-collapse:collapse; font-size:12px; }} th,td {{ border-bottom:1px solid var(--line); padding:8px; text-align:right; white-space:nowrap; }} th:first-child,td:first-child,th:nth-child(2),td:nth-child(2) {{ text-align:left; }} .case {{ border-top:1px solid var(--line); padding:15px 0 6px; }} .correct {{ background:#e4f4ed; color:#0d6b53; font-weight:700; }} .wrong {{ background:#fde8e2; color:#a4331b; font-weight:700; }}
@media(max-width:760px) {{ .charts {{ grid-template-columns:1fr; }} }}
</style></head><body><main><h1>Native FutureSeed: length 64 to 1024</h1>
<p>Four associations are fixed while irrelevant context grows 16x. L64 is the frozen P-CAUSAL-010 endpoint; only the two L1024 GDN2 arms are newly trained.</p>
<div class="gate">Registered strong gate: {'PASS' if gate['passed'] else 'MISS'}; future accuracy retention {gate['future_seed_future_accuracy_retention_64_to_1024']:.3f}</div>
<div class="legend">{legend}</div><div class="charts">
<div class="chart">{line_chart(comparison,value_getter=lambda s:s['metrics']['future']['accuracy'],title='Future-query accuracy',percent=True)}</div>
<div class="chart">{line_chart(comparison,value_getter=lambda s:s['warmed_step_benchmark']['tokens_per_sec'],title='Warmed training throughput (tokens/s)',percent=False)}</div>
</div><h2>Quality and systems metrics</h2><div class="table-wrap"><table><thead><tr><th>Length</th><th>Model</th><th>Past acc</th><th>Future acc</th><th>Joint exact</th><th>Future CE</th><th>Tokens/s</th><th>Peak memory</th></tr></thead><tbody>{metric_rows(comparison)}</tbody></table></div>
{''.join(case_sections)}</main></body></html>"""
    (args.visual_dir / "index.html").write_text(document)
    print(json.dumps({"visualized_lengths": [64, 1024]}, sort_keys=True))


if __name__ == "__main__":
    main()
