from __future__ import annotations

import argparse
import html
import json
from pathlib import Path
from typing import Any


ARMS = ("causal_gdn2", "future_seed_gdn2", "bidirectional_attention")
ARM_LABELS = {
    "causal_gdn2": "Causal GDN2",
    "future_seed_gdn2": "GDN2 + FutureSeed",
    "bidirectional_attention": "Bidirectional attention",
}
ARM_COLORS = {
    "causal_gdn2": "#c73e1d",
    "future_seed_gdn2": "#147d64",
    "bidirectional_attention": "#3155a6",
}


def fmt(value: float, digits: int = 4) -> str:
    return f"{value:.{digits}f}"


def load_cases(output_dir: Path, length: int) -> dict[str, list[dict[str, Any]]]:
    return {
        arm: json.loads(
            (output_dir / f"length_{length}" / arm / "cases.json").read_text()
        )
        for arm in ARMS
    }


def merge_hardest_cases(
    output_dir: Path,
    length: int,
    *,
    limit: int = 8,
) -> list[dict[str, Any]]:
    cases = load_cases(output_dir, length)
    by_arm = {
        arm: {case["case_id"]: case for case in arm_cases}
        for arm, arm_cases in cases.items()
    }
    shared_ids = set.intersection(*(set(mapping) for mapping in by_arm.values()))
    merged = []
    for case_id in shared_ids:
        row = {
            "case_id": case_id,
            "sequence_length": length,
            "arms": {arm: by_arm[arm][case_id] for arm in ARMS},
        }
        merged.append(row)
    merged.sort(
        key=lambda row: (
            -row["arms"]["future_seed_gdn2"]["future_errors"],
            -row["arms"]["causal_gdn2"]["future_errors"],
            row["arms"]["bidirectional_attention"]["future_errors"],
            row["case_id"],
        )
    )
    return merged[:limit]


def line_chart(
    comparison: dict[str, Any],
    *,
    metric_path: tuple[str, ...],
    title: str,
    percent: bool,
) -> str:
    lengths = sorted(int(length) for length in comparison["lengths"])
    width, height = 760, 270
    left, right, top, bottom = 62, 20, 34, 42
    plot_width = width - left - right
    plot_height = height - top - bottom
    values = []
    for length in lengths:
        for arm in ARMS:
            value: Any = comparison["lengths"][str(length)]["arms"][arm]
            for key in metric_path:
                value = value[key]
            values.append(float(value))
    maximum = 1.0 if percent else max(values) * 1.08
    minimum = 0.0

    def x(length: int) -> float:
        if len(lengths) == 1:
            return left + plot_width / 2
        index = lengths.index(length)
        return left + plot_width * index / (len(lengths) - 1)

    def y(value: float) -> float:
        return top + plot_height * (maximum - value) / (maximum - minimum)

    parts = [
        f'<svg viewBox="0 0 {width} {height}" role="img" aria-label="{html.escape(title)}">',
        f'<text x="{left}" y="20" class="chart-title">{html.escape(title)}</text>',
    ]
    for tick in range(5):
        value = maximum * tick / 4
        yy = y(value)
        label = f"{value * 100:.0f}%" if percent else f"{value:,.0f}"
        parts.append(
            f'<line x1="{left}" y1="{yy:.1f}" x2="{width-right}" y2="{yy:.1f}" class="grid" />'
        )
        parts.append(
            f'<text x="{left-9}" y="{yy+4:.1f}" text-anchor="end" class="axis">{label}</text>'
        )
    for length in lengths:
        xx = x(length)
        parts.append(
            f'<text x="{xx:.1f}" y="{height-15}" text-anchor="middle" class="axis">{length}</text>'
        )
    for arm in ARMS:
        points = []
        for length in lengths:
            value: Any = comparison["lengths"][str(length)]["arms"][arm]
            for key in metric_path:
                value = value[key]
            points.append((x(length), y(float(value))))
        point_text = " ".join(f"{xx:.1f},{yy:.1f}" for xx, yy in points)
        parts.append(
            f'<polyline points="{point_text}" fill="none" stroke="{ARM_COLORS[arm]}" stroke-width="3" />'
        )
        for xx, yy in points:
            parts.append(
                f'<circle cx="{xx:.1f}" cy="{yy:.1f}" r="4" fill="{ARM_COLORS[arm]}" />'
            )
    parts.append("</svg>")
    return "".join(parts)


def metric_table(comparison: dict[str, Any]) -> str:
    rows = []
    for length in sorted(int(value) for value in comparison["lengths"]):
        summary = comparison["lengths"][str(length)]
        for arm in ARMS:
            score = summary["arms"][arm]
            metrics = score["metrics"]
            bench = score["warmed_step_benchmark"]
            rows.append(
                "<tr>"
                f"<td>{length}</td>"
                f"<td>{html.escape(ARM_LABELS[arm])}</td>"
                f"<td>{score['parameters']:,}</td>"
                f"<td>{fmt(metrics['past']['accuracy'])}</td>"
                f"<td>{fmt(metrics['future']['accuracy'])}</td>"
                f"<td>{fmt(metrics['joint_exact'])}</td>"
                f"<td>{fmt(metrics['future']['ce'])}</td>"
                f"<td>{bench['tokens_per_sec']:,.0f}</td>"
                f"<td>{bench['peak_cuda_mem_bytes'] / 2**30:.2f} GiB</td>"
                "</tr>"
            )
    return "".join(rows)


def case_html(case: dict[str, Any]) -> str:
    fs_case = case["arms"]["future_seed_gdn2"]
    rows = []
    for event_index, fs_event in enumerate(fs_case["events"]):
        predictions = []
        for arm in ARMS:
            event = case["arms"][arm]["events"][event_index]
            css = "correct" if event["correct"] else "wrong"
            predictions.append(
                f'<td class="{css}">{event["prediction"]}</td>'
            )
        rows.append(
            "<tr>"
            f"<td>{html.escape(fs_event['direction'])}</td>"
            f"<td>{fs_event['query_position']}</td>"
            f"<td>{fs_event['write_position']}</td>"
            f"<td>{fs_event['distance']:+d}</td>"
            f"<td>{fs_event['key']}</td>"
            f"<td>{fs_event['target']}</td>"
            + "".join(predictions)
            + "</tr>"
        )
    error_summary = " / ".join(
        f"{ARM_LABELS[arm]}: future {case['arms'][arm]['future_errors']}, "
        f"past {case['arms'][arm]['past_errors']}"
        for arm in ARMS
    )
    return (
        '<article class="case">'
        f"<h3>Case {html.escape(case['case_id'])}</h3>"
        f"<p>{html.escape(error_summary)}</p>"
        '<div class="table-wrap"><table><thead><tr>'
        "<th>Direction</th><th>Query pos</th><th>Write pos</th><th>Distance</th>"
        "<th>Key</th><th>Target</th>"
        f"<th>{ARM_LABELS['causal_gdn2']}</th>"
        f"<th>{ARM_LABELS['future_seed_gdn2']}</th>"
        f"<th>{ARM_LABELS['bidirectional_attention']}</th>"
        "</tr></thead><tbody>"
        + "".join(rows)
        + "</tbody></table></div></article>"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--visual-dir", type=Path, required=True)
    args = parser.parse_args()
    comparison = json.loads((args.output_dir / "comparison.json").read_text())
    lengths = sorted(int(length) for length in comparison["lengths"])
    hardest = {
        str(length): merge_hardest_cases(args.output_dir, length)
        for length in lengths
    }
    args.visual_dir.mkdir(parents=True, exist_ok=True)
    (args.visual_dir / "hardest_cases.json").write_text(
        json.dumps(hardest, indent=2, sort_keys=True) + "\n"
    )
    summary = {
        "sequence_lengths": lengths,
        "num_kv_pairs": comparison["protocol"]["num_kv_pairs"],
        "metrics": {
            length: {
                arm: comparison["lengths"][str(length)]["arms"][arm]["metrics"]
                for arm in ARMS
            }
            for length in lengths
        },
    }
    (args.visual_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )

    legend = "".join(
        f'<span><i style="background:{ARM_COLORS[arm]}"></i>{ARM_LABELS[arm]}</span>'
        for arm in ARMS
    )
    case_sections = []
    for length in lengths:
        case_sections.append(f"<h2>Hardest same-sequence cases, length {length}</h2>")
        case_sections.extend(case_html(case) for case in hardest[str(length)])
    document = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>FutureSeed directional scaling</title>
<style>
:root {{ color-scheme: light; --ink:#17211d; --muted:#5b6761; --line:#d7ddd9; --paper:#f6f7f5; }}
* {{ box-sizing:border-box; }}
body {{ margin:0; font-family:Inter,ui-sans-serif,system-ui,sans-serif; color:var(--ink); background:white; letter-spacing:0; }}
main {{ width:min(1180px,calc(100% - 32px)); margin:28px auto 64px; }}
h1 {{ font-size:30px; margin:0 0 8px; }}
h2 {{ font-size:20px; margin:34px 0 12px; }}
h3 {{ font-size:15px; margin:0 0 5px; }}
p {{ color:var(--muted); margin:0 0 12px; }}
.legend {{ display:flex; gap:20px; flex-wrap:wrap; margin:18px 0; font-size:13px; }}
.legend span {{ display:flex; align-items:center; gap:7px; }}
.legend i {{ width:12px; height:12px; border-radius:2px; }}
.charts {{ display:grid; grid-template-columns:1fr 1fr; gap:16px; }}
.chart {{ border:1px solid var(--line); border-radius:6px; padding:8px; overflow:hidden; }}
svg {{ width:100%; height:auto; }}
.chart-title {{ font-size:14px; font-weight:700; }}
.axis {{ font-size:11px; fill:#65716b; }}
.grid {{ stroke:#e3e8e5; stroke-width:1; }}
.table-wrap {{ overflow-x:auto; }}
table {{ border-collapse:collapse; width:100%; font-size:12px; }}
th,td {{ border-bottom:1px solid var(--line); padding:8px 9px; text-align:right; white-space:nowrap; }}
th:first-child,td:first-child,th:nth-child(2),td:nth-child(2) {{ text-align:left; }}
.case {{ border-top:1px solid var(--line); padding:16px 0 7px; }}
.correct {{ background:#e3f5ed; color:#075f47; font-weight:700; }}
.wrong {{ background:#fde8e2; color:#9d2c15; font-weight:700; }}
@media (max-width:760px) {{ .charts {{ grid-template-columns:1fr; }} main {{ width:min(100% - 20px,1180px); }} }}
</style>
</head>
<body><main>
<h1>FutureSeed directional scaling</h1>
<p>Fixed four associations per sequence. Only sequence length changes; every arm uses the same examples, optimizer steps, seed, width, depth, and evaluation.</p>
<div class="legend">{legend}</div>
<div class="charts">
<div class="chart">{line_chart(comparison, metric_path=('metrics','future','accuracy'), title='Future-query accuracy vs sequence length', percent=True)}</div>
<div class="chart">{line_chart(comparison, metric_path=('warmed_step_benchmark','tokens_per_sec'), title='Warmed training throughput (tokens/s)', percent=False)}</div>
</div>
<h2>Metrics and systems cost</h2>
<div class="table-wrap"><table><thead><tr><th>Length</th><th>Model</th><th>Parameters</th><th>Past acc</th><th>Future acc</th><th>Joint exact</th><th>Future CE</th><th>Tokens/s</th><th>Peak memory</th></tr></thead><tbody>{metric_table(comparison)}</tbody></table></div>
{''.join(case_sections)}
</main></body></html>
"""
    (args.visual_dir / "index.html").write_text(document)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
