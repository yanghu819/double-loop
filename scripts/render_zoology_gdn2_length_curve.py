from __future__ import annotations

import argparse
import html
import json
from pathlib import Path
from typing import Any, Callable


ARMS = ("causal_gdn2", "future_seed_gdn2")
LENGTHS = (64, 128, 256, 512, 1024)
LABELS = {
    "causal_gdn2": "Causal GDN2",
    "future_seed_gdn2": "GDN2 + FutureSeed",
}
COLORS = {"causal_gdn2": "#717874", "future_seed_gdn2": "#176b55"}


def line_chart(
    comparison: dict[str, Any],
    *,
    title: str,
    value_getter: Callable[[dict[str, Any]], float],
    maximum: float | None = None,
    percent: bool = False,
) -> str:
    width, height = 700, 320
    left, right, top, bottom = 58, 20, 40, 42
    plot_width = width - left - right
    plot_height = height - top - bottom
    values = [
        float(value_getter(comparison["lengths"][str(length)]["arms"][arm]))
        for length in LENGTHS
        for arm in ARMS
    ]
    top_value = maximum if maximum is not None else max(values) * 1.08

    def x(index: int) -> float:
        return left + plot_width * index / (len(LENGTHS) - 1)

    def y(value: float) -> float:
        return top + plot_height * (top_value - value) / max(top_value, 1e-12)

    parts = [
        f'<svg viewBox="0 0 {width} {height}" role="img" aria-label="{html.escape(title)}">',
        f'<text x="{left}" y="22" class="chart-title">{html.escape(title)}</text>',
    ]
    for tick in range(5):
        value = top_value * tick / 4
        yy = y(value)
        label = f"{value * 100:.0f}%" if percent else f"{value:,.0f}"
        parts.append(
            f'<line x1="{left}" y1="{yy:.1f}" x2="{width-right}" '
            f'y2="{yy:.1f}" class="grid" />'
        )
        parts.append(
            f'<text x="{left-8}" y="{yy+4:.1f}" text-anchor="end" '
            f'class="axis">{label}</text>'
        )
    for index, length in enumerate(LENGTHS):
        parts.append(
            f'<text x="{x(index):.1f}" y="{height-14}" text-anchor="middle" '
            f'class="axis">{length}</text>'
        )
    for arm in ARMS:
        points = [
            (
                x(index),
                y(float(value_getter(comparison["lengths"][str(length)]["arms"][arm]))),
            )
            for index, length in enumerate(LENGTHS)
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


def merge_cases(output_dir: Path, sequence_length: int, limit: int = 2) -> list[dict[str, Any]]:
    by_arm = {}
    for arm in ARMS:
        cases = json.loads(
            (output_dir / f"length_{sequence_length}" / arm / "cases.json").read_text()
        )
        by_arm[arm] = {case["case_id"]: case for case in cases}
    shared_ids = set(by_arm[ARMS[0]]) & set(by_arm[ARMS[1]])
    merged = [
        {
            "case_id": case_id,
            "sequence_length": sequence_length,
            "arms": {arm: by_arm[arm][case_id] for arm in ARMS},
        }
        for case_id in shared_ids
    ]
    merged.sort(
        key=lambda row: (
            -row["arms"]["future_seed_gdn2"]["future_errors"],
            -row["arms"]["future_seed_gdn2"]["past_errors"],
            row["case_id"],
        )
    )
    return merged[:limit]


def metric_rows(comparison: dict[str, Any]) -> str:
    rows = []
    for length in LENGTHS:
        for arm in ARMS:
            score = comparison["lengths"][str(length)]["arms"][arm]
            metrics = score["metrics"]
            benchmark = score["fresh_process_benchmark"]
            rows.append(
                "<tr>"
                f"<td>{length}</td><td>{html.escape(LABELS[arm])}</td>"
                f"<td>{metrics['past']['accuracy']:.4f}</td>"
                f"<td>{metrics['future']['accuracy']:.4f}</td>"
                f"<td>{metrics['joint_exact']:.4f}</td>"
                f"<td>{metrics['past']['ce']:.4f}</td>"
                f"<td>{metrics['future']['ce']:.4f}</td>"
                f"<td>{benchmark['tokens_per_sec']:,.0f}</td>"
                f"<td>{benchmark['peak_cuda_mem_bytes']/2**20:.1f} MiB</td>"
                "</tr>"
            )
    return "".join(rows)


def binding_rows(comparison: dict[str, Any]) -> str:
    rows = []
    for length in LENGTHS:
        for arm in ARMS:
            for direction in ("past", "future"):
                row = comparison["binding_diagnostics"][str(length)][arm][direction]
                rows.append(
                    "<tr>"
                    f"<td>{length}</td><td>{html.escape(LABELS[arm])}</td>"
                    f"<td>{direction}</td><td>{row['errors']}</td>"
                    f"<td>{row['other_sample_value']}</td>"
                    f"<td>{row['other_sample_value_fraction']:.1%}</td>"
                    "</tr>"
                )
    return "".join(rows)


def case_html(case: dict[str, Any]) -> str:
    future_seed_case = case["arms"]["future_seed_gdn2"]
    rows = []
    for index, event in enumerate(future_seed_case["events"]):
        prediction_cells = []
        for arm in ARMS:
            arm_event = case["arms"][arm]["events"][index]
            css = "correct" if arm_event["correct"] else "wrong"
            prediction_cells.append(
                f'<td class="{css}">{arm_event["prediction"]}</td>'
            )
        rows.append(
            "<tr>"
            f"<td>{event['direction']}</td><td>{event['query_position']}</td>"
            f"<td>{event['write_position']}</td><td>{event['distance']:+d}</td>"
            f"<td>{event['key']}</td><td>{event['target']}</td>"
            + "".join(prediction_cells)
            + "</tr>"
        )
    summary = " / ".join(
        f"{LABELS[arm]} future/past errors "
        f"{case['arms'][arm]['future_errors']}/{case['arms'][arm]['past_errors']}"
        for arm in ARMS
    )
    return (
        '<article class="case">'
        f"<h3>Case {html.escape(case['case_id'])}</h3>"
        f"<p>{html.escape(summary)}</p>"
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
    hardest = {
        str(length): merge_cases(args.output_dir, length) for length in LENGTHS
    }
    args.visual_dir.mkdir(parents=True, exist_ok=True)
    (args.visual_dir / "hardest_cases.json").write_text(
        json.dumps(hardest, indent=2, sort_keys=True) + "\n"
    )
    (args.visual_dir / "summary.json").write_text(
        json.dumps(
            {
                "registered_gate": comparison["registered_gate"],
                "curve_diagnostics": comparison["curve_diagnostics"],
                "binding_diagnostics": comparison["binding_diagnostics"],
                "systems_metrics": comparison["systems_metrics"],
                "metrics": {
                    str(length): {
                        arm: comparison["lengths"][str(length)]["arms"][arm]["metrics"]
                        for arm in ARMS
                    }
                    for length in LENGTHS
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
    for length in LENGTHS:
        case_sections.append(f"<h2>Hardest same-sequence cases, length {length}</h2>")
        case_sections.extend(case_html(case) for case in hardest[str(length)])
    gate = comparison["registered_gate"]
    curve = comparison["curve_diagnostics"]
    document = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>P-CAUSAL-016: FutureSeed length curve</title><style>
:root {{ --ink:#18211d; --muted:#5b6761; --line:#d8dedb; --good:#176b55; --bad:#a4331b; --soft:#f5f7f6; }} * {{ box-sizing:border-box; }}
body {{ margin:0; color:var(--ink); background:#fff; font-family:Inter,ui-sans-serif,system-ui,sans-serif; letter-spacing:0; }}
main {{ width:min(1220px,calc(100% - 30px)); margin:28px auto 64px; }} h1 {{ font-size:29px; margin:0 0 8px; }} h2 {{ font-size:20px; margin:32px 0 12px; }} h3 {{ font-size:14px; margin:0 0 4px; }} p {{ color:var(--muted); margin:0 0 12px; line-height:1.45; }}
.gate {{ display:grid; grid-template-columns:120px 1fr; gap:18px; align-items:center; border-block:1px solid var(--line); padding:16px 0; }} .badge {{ height:54px; display:grid; place-items:center; border:2px solid currentColor; font-weight:800; font-size:18px; }} .pass {{ color:var(--good); }} .stop {{ color:var(--bad); }}
.legend {{ display:flex; flex-wrap:wrap; gap:20px; margin:16px 0; }} .legend span {{ display:flex; gap:7px; align-items:center; }} .legend i {{ width:12px; height:12px; }}
.charts {{ display:grid; grid-template-columns:1fr 1fr; gap:14px; }} .chart {{ border:1px solid var(--line); border-radius:6px; padding:8px; overflow:hidden; }} svg {{ width:100%; height:auto; }} .chart-title {{ font-size:14px; font-weight:700; }} .axis {{ font-size:11px; fill:#67716c; }} .grid {{ stroke:#e4e9e6; }}
.table-wrap {{ overflow-x:auto; border:1px solid var(--line); }} table {{ width:100%; border-collapse:collapse; font-size:12px; min-width:900px; }} th,td {{ border-bottom:1px solid var(--line); padding:8px; text-align:right; white-space:nowrap; }} th:first-child,td:first-child,th:nth-child(2),td:nth-child(2) {{ text-align:left; }} th {{ background:var(--soft); color:var(--muted); }} .case {{ border-top:1px solid var(--line); padding:15px 0 6px; }} .correct {{ background:#e4f4ed; color:#0d6b53; font-weight:700; }} .wrong {{ background:#fde8e2; color:#a4331b; font-weight:700; }}
@media(max-width:760px) {{ .charts {{ grid-template-columns:1fr; }} .gate {{ grid-template-columns:1fr; }} h1 {{ font-size:24px; }} }}
</style></head><body><main><h1>Native FutureSeed: context-length scaling</h1>
<p>Four key/value associations stay fixed while irrelevant context grows from 64 to 1024 tokens. Lengths 64 and 1024 are frozen evidence; only 128, 256 and 512 are newly trained, each arm in a fresh process.</p>
<section class="gate"><div class="badge {'pass' if gate['passed'] else 'stop'}">{'PASS' if gate['passed'] else 'STOP'}</div><div><strong>Registered scaling gate</strong><p>FutureSeed future accuracy floors are 0.95, 0.90 and 0.80 at lengths 128, 256 and 512; every length must beat its causal future control by at least 0.70. Frozen 64/1024 floors are 0.90/0.70. Retention from 64 to 1024: {curve['retention_64_to_1024']:.3f}.</p></div></section>
<div class="legend">{legend}</div><div class="charts">
<div class="chart">{line_chart(comparison,title='Future-query accuracy',value_getter=lambda s:s['metrics']['future']['accuracy'],maximum=1.0,percent=True)}</div>
<div class="chart">{line_chart(comparison,title='Past-query accuracy',value_getter=lambda s:s['metrics']['past']['accuracy'],maximum=1.0,percent=True)}</div>
<div class="chart">{line_chart(comparison,title='Whole-sample exact',value_getter=lambda s:s['metrics']['joint_exact'],maximum=1.0,percent=True)}</div>
<div class="chart">{line_chart(comparison,title='Fresh-process warmed training throughput (tokens/s)',value_getter=lambda s:s['fresh_process_benchmark']['tokens_per_sec'])}</div>
</div>
<h2>Quality and systems metrics</h2><div class="table-wrap"><table><thead><tr><th>Length</th><th>Model</th><th>Past acc</th><th>Future acc</th><th>Joint exact</th><th>Past CE</th><th>Future CE</th><th>Tokens/s</th><th>Peak memory</th></tr></thead><tbody>{metric_rows(comparison)}</tbody></table></div>
<h2>Address-binding errors</h2><p>"Another valid value" means a wrong prediction selected a value attached to another key in the same sequence. This distinguishes lost content from incorrect key-to-value binding.</p>
<div class="table-wrap"><table><thead><tr><th>Length</th><th>Model</th><th>Direction</th><th>Errors</th><th>Another valid value</th><th>Fraction</th></tr></thead><tbody>{binding_rows(comparison)}</tbody></table></div>
{''.join(case_sections)}</main></body></html>"""
    (args.visual_dir / "index.html").write_text(document)
    print(json.dumps({"visualized_lengths": list(LENGTHS)}, sort_keys=True))


if __name__ == "__main__":
    main()
