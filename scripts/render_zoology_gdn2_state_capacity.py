from __future__ import annotations

import argparse
import html
import json
from pathlib import Path
from typing import Any


MODELS = (
    "reference_d128_future_seed",
    "candidate_d256_causal",
    "candidate_d256_future_seed",
)
LABELS = {
    "reference_d128_future_seed": "D128 FutureSeed",
    "candidate_d256_causal": "D256 causal",
    "candidate_d256_future_seed": "D256 FutureSeed",
}
COLORS = {
    "reference_d128_future_seed": "#b85b25",
    "candidate_d256_causal": "#326aa8",
    "candidate_d256_future_seed": "#16745a",
}


def score_for(comparison: dict[str, Any], model: str) -> dict[str, Any]:
    if model == "reference_d128_future_seed":
        return comparison["reference_d128"]["arms"]["future_seed_gdn2"]
    arm = {
        "candidate_d256_causal": "causal_gdn2_d256",
        "candidate_d256_future_seed": "future_seed_gdn2_d256",
    }[model]
    return comparison["candidate_d256"]["arms"][arm]


def metric_bars(comparison: dict[str, Any], key: str, title: str) -> str:
    rows = []
    for model in MODELS:
        metrics = score_for(comparison, model)["metrics"]
        if key == "joint_exact":
            value = metrics[key]
        else:
            direction, metric = key.split(".")
            value = metrics[direction][metric]
        rows.append(
            '<div class="bar-row">'
            f'<span>{html.escape(LABELS[model])}</span>'
            '<div class="bar-track">'
            f'<i style="width:{100*value:.2f}%;background:{COLORS[model]}"></i>'
            "</div>"
            f"<strong>{value:.4f}</strong></div>"
        )
    return f'<section class="metric"><h3>{html.escape(title)}</h3>{"".join(rows)}</section>'


def learning_curve(
    comparison: dict[str, Any],
    metric: str,
    title: str,
    *,
    fixed_range: tuple[float, float] | None = None,
) -> str:
    width, height, pad = 620, 220, 38
    series = {
        model: [float(row[metric]) for row in score_for(comparison, model)["valid_curve"]]
        for model in MODELS
    }
    values = [value for model_values in series.values() for value in model_values]
    low, high = fixed_range or (min(values), max(values))
    if high - low < 1e-9:
        high = low + 1.0

    def point(index: int, value: float, count: int) -> tuple[float, float]:
        x = pad + (width - 2 * pad) * index / max(count - 1, 1)
        y = height - pad - (height - 2 * pad) * (value - low) / (high - low)
        return x, y

    grid = []
    for fraction in (0.0, 0.5, 1.0):
        value = low + (high - low) * fraction
        y = height - pad - (height - 2 * pad) * fraction
        grid.append(
            f'<line x1="{pad}" y1="{y:.1f}" x2="{width-pad}" y2="{y:.1f}" '
            'stroke="#d8dfdb" stroke-width="1"/>'
            f'<text x="4" y="{y+4:.1f}" font-size="11" fill="#5d6862">'
            f"{value:.2f}</text>"
        )
    lines = []
    legends = []
    for legend_index, model in enumerate(MODELS):
        points = " ".join(
            f"{x:.1f},{y:.1f}"
            for x, y in (
                point(index, value, len(series[model]))
                for index, value in enumerate(series[model])
            )
        )
        lines.append(
            f'<polyline points="{points}" fill="none" stroke="{COLORS[model]}" '
            'stroke-width="3" vector-effect="non-scaling-stroke"/>'
        )
        legend_x = pad + legend_index * 178
        legends.append(
            f'<line x1="{legend_x}" y1="205" x2="{legend_x+22}" y2="205" '
            f'stroke="{COLORS[model]}" stroke-width="3"/>'
            f'<text x="{legend_x+28}" y="209" font-size="11" fill="#17201c">'
            f"{html.escape(LABELS[model])}</text>"
        )
    return (
        '<section class="curve"><h3>'
        + html.escape(title)
        + f'</h3><svg viewBox="0 0 {width} {height}" role="img" '
        + f'aria-label="{html.escape(title)}">'
        + "".join(grid + lines + legends)
        + "</svg></section>"
    )


def load_case_map(path: Path) -> dict[str, dict[str, Any]]:
    return {case["case_id"]: case for case in json.loads(path.read_text())}


def hardest_cases(output_dir: Path, limit: int = 12) -> list[dict[str, Any]]:
    maps = {
        "reference_d128_future_seed": load_case_map(
            output_dir / "reference_d128" / "future_seed_gdn2" / "cases.json"
        ),
        "candidate_d256_causal": load_case_map(
            output_dir / "length_1024" / "causal_gdn2_d256" / "cases.json"
        ),
        "candidate_d256_future_seed": load_case_map(
            output_dir
            / "length_1024"
            / "future_seed_gdn2_d256"
            / "cases.json"
        ),
    }
    shared = set.intersection(*(set(case_map) for case_map in maps.values()))
    rows = [
        {"case_id": case_id, "models": {name: maps[name][case_id] for name in MODELS}}
        for case_id in shared
    ]
    rows.sort(
        key=lambda row: (
            -row["models"]["candidate_d256_future_seed"]["future_errors"],
            -row["models"]["candidate_d256_future_seed"]["past_errors"],
            -row["models"]["reference_d128_future_seed"]["future_errors"],
            row["case_id"],
        )
    )
    return rows[:limit]


def binding_rows(comparison: dict[str, Any]) -> str:
    rows = []
    diagnostics = comparison["binding_diagnostics"]
    for model in MODELS:
        for direction in ("past", "future"):
            row = diagnostics[model][direction]
            rows.append(
                "<tr>"
                f"<td>{html.escape(LABELS[model])}</td><td>{direction}</td>"
                f"<td>{row['errors']}</td>"
                f"<td>{row['wrong_value_from_same_case']}</td>"
                f"<td>{row['wrong_value_from_same_case_fraction_of_queries']:.1%}</td>"
                f"<td>{row['wrong_value_from_same_case_fraction_of_errors']:.1%}</td>"
                "</tr>"
            )
    return "".join(rows)


def case_html(case: dict[str, Any]) -> str:
    anchor = case["models"]["candidate_d256_future_seed"]
    rows = []
    for index, event in enumerate(anchor["events"]):
        predictions = []
        for model in MODELS:
            model_event = case["models"][model]["events"][index]
            css = "correct" if model_event["correct"] else "wrong"
            predictions.append(f'<td class="{css}">{model_event["prediction"]}</td>')
        rows.append(
            "<tr>"
            f"<td>{event['direction']}</td><td>{event['query_position']}</td>"
            f"<td>{event['write_position']}</td><td>{event['distance']:+d}</td>"
            f"<td>{event['key']}</td><td>{event['target']}</td>"
            + "".join(predictions)
            + "</tr>"
        )
    summary = " / ".join(
        f"{LABELS[model]} errors "
        f"{case['models'][model]['past_errors'] + case['models'][model]['future_errors']}"
        for model in MODELS
    )
    return (
        '<article class="case">'
        f"<h3>Case {html.escape(case['case_id'])}</h3><p>{html.escape(summary)}</p>"
        '<div class="table-wrap"><table><thead><tr>'
        "<th>Direction</th><th>Query</th><th>Write</th><th>Distance</th>"
        "<th>Key</th><th>Target</th>"
        + "".join(f"<th>{html.escape(LABELS[model])}</th>" for model in MODELS)
        + "</tr></thead><tbody>"
        + "".join(rows)
        + "</tbody></table></div></article>"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--visual-dir", type=Path, required=True)
    args = parser.parse_args()
    comparison = json.loads((args.output_dir / "comparison.json").read_text())
    cases = hardest_cases(args.output_dir)
    args.visual_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "capacity_scaling": comparison["capacity_scaling"],
        "registered_gate": comparison["registered_gate"],
        "binding_diagnostics": comparison["binding_diagnostics"],
        "metrics": {
            model: score_for(comparison, model)["metrics"] for model in MODELS
        },
        "binding_diagnostic_interpretable": (
            score_for(comparison, "candidate_d256_future_seed")["metrics"][
                "balanced_accuracy"
            ]
            >= score_for(comparison, "reference_d128_future_seed")["metrics"][
                "balanced_accuracy"
            ]
        ),
    }
    (args.visual_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    (args.visual_dir / "hardest_cases.json").write_text(
        json.dumps(cases, indent=2, sort_keys=True) + "\n"
    )
    gate = comparison["registered_gate"]
    capacity = comparison["capacity_scaling"]
    binding_interpretable = summary["binding_diagnostic_interpretable"]
    binding_note = (
        "The binding-rate delta is interpretable because candidate accuracy did not "
        "fall below the reference."
        if binding_interpretable
        else "Do not interpret the lower same-case swap rate as better binding: the "
        "D256 candidate collapsed to near-random accuracy, and random predictions "
        "rarely equal any value from the same sequence."
    )
    document = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>FutureSeed state capacity at length 1024</title><style>
:root {{ --ink:#17201c; --muted:#5d6862; --line:#d8dfdb; --soft:#f5f7f6; }}
* {{ box-sizing:border-box; }} body {{ margin:0; color:var(--ink); background:#fff; font-family:Inter,ui-sans-serif,system-ui,sans-serif; letter-spacing:0; }}
main {{ width:min(1180px,calc(100% - 30px)); margin:28px auto 64px; }} h1 {{ font-size:29px; margin:0 0 8px; }} h2 {{ font-size:20px; margin:32px 0 12px; }} h3 {{ font-size:14px; margin:0 0 8px; }} p {{ color:var(--muted); margin:0 0 12px; }}
.gate {{ border-block:1px solid var(--line); padding:14px 0; font-weight:700; }} .warning {{ border-left:4px solid #b85b25; padding:10px 12px; color:var(--ink); background:#fff7f1; }} .metrics {{ display:grid; grid-template-columns:repeat(3,1fr); gap:12px; }} .metric {{ border:1px solid var(--line); border-radius:6px; padding:14px; }} .curves {{ display:grid; grid-template-columns:1fr 1fr; gap:12px; }} .curve {{ border:1px solid var(--line); border-radius:6px; padding:12px; }} .curve svg {{ display:block; width:100%; height:auto; }}
.bar-row {{ display:grid; grid-template-columns:135px 1fr 54px; gap:8px; align-items:center; font-size:12px; margin:11px 0; }} .bar-track {{ height:12px; background:var(--soft); }} .bar-track i {{ display:block; height:100%; }}
.table-wrap {{ overflow-x:auto; }} table {{ width:100%; border-collapse:collapse; font-size:12px; }} th,td {{ border-bottom:1px solid var(--line); padding:8px; text-align:right; white-space:nowrap; }} th:first-child,td:first-child,th:nth-child(2),td:nth-child(2) {{ text-align:left; }}
.case {{ border-top:1px solid var(--line); padding:15px 0 6px; }} .correct {{ background:#e3f3eb; color:#0c694f; font-weight:700; }} .wrong {{ background:#fde7e1; color:#a3311a; font-weight:700; }}
@media(max-width:800px) {{ .metrics,.curves {{ grid-template-columns:1fr; }} .bar-row {{ grid-template-columns:120px 1fr 50px; }} }}
</style></head><body><main><h1>FutureSeed state capacity at length 1024</h1>
<p>The data, seed, optimizer, epochs and FutureSeed rule are fixed. D128/H4/D32 is frozen; D256/H8/D32 increases recurrent state values per layer {capacity['state_value_ratio']:.1f}x.</p>
<div class="gate">Registered gate: {'PASS' if gate['passed'] else 'MISS'}; balanced accuracy gain {gate['balanced_accuracy_gain']:+.4f}; future binding-error reduction {gate['future_binding_error_rate_reduction']:+.4f}</div>
<h2>Accuracy</h2><div class="metrics">
{metric_bars(comparison,'past.accuracy','Past-query accuracy')}
{metric_bars(comparison,'future.accuracy','Future-query accuracy')}
{metric_bars(comparison,'joint_exact','Whole-sample exact')}
</div><h2>Training dynamics</h2><div class="curves">
{learning_curve(comparison,'valid/accuracy','Validation accuracy by epoch',fixed_range=(0.0,1.0))}
{learning_curve(comparison,'valid/loss','Validation cross-entropy by epoch')}
</div><h2>Binding errors</h2>
<p>"Another case value" means the predicted value belongs to another key in the same sequence. The query-rate column measures how often this happens over all queries, so it falls only when the model actually removes swaps.</p>
<p class="warning">{html.escape(binding_note)}</p>
<div class="table-wrap"><table><thead><tr><th>Model</th><th>Direction</th><th>Errors</th><th>Another case value</th><th>Per query</th><th>Among errors</th></tr></thead><tbody>{binding_rows(comparison)}</tbody></table></div>
<h2>Hardest same-sequence cases</h2>{''.join(case_html(case) for case in cases)}
</main></body></html>"""
    (args.visual_dir / "index.html").write_text(document)
    print(json.dumps({"visualized_cases": len(cases)}, sort_keys=True))


if __name__ == "__main__":
    main()
