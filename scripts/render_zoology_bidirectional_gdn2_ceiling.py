from __future__ import annotations

import argparse
import html
import json
from pathlib import Path
from typing import Any


ARMS = ("causal_gdn2", "future_seed_gdn2", "explicit_bidirectional_gdn2")
LABELS = {
    "causal_gdn2": "Causal GDN2",
    "future_seed_gdn2": "GDN2 + FutureSeed",
    "explicit_bidirectional_gdn2": "Forward + reverse GDN2",
}


def quality_svg(score: dict[str, Any]) -> str:
    width, height = 900, 310
    left, right, top, bottom = 74, 24, 28, 58
    plot_w, plot_h = width - left - right, height - top - bottom
    groups = ("past", "future", "joint")
    colors = ("#59635e", "#14745a", "#b45f18")
    parts = [f'<svg viewBox="0 0 {width} {height}" aria-label="L512 quality">']
    for tick in range(5):
        value = tick / 4
        y = top + plot_h * (1 - value)
        parts.append(f'<line class="grid" x1="{left}" y1="{y}" x2="{width-right}" y2="{y}"/>')
        parts.append(f'<text class="axis" x="{left-9}" y="{y+4}" text-anchor="end">{value:.2f}</text>')
    group_w = plot_w / len(groups)
    bar_w = group_w / 5
    for group_i, group in enumerate(groups):
        center = left + group_w * (group_i + 0.5)
        for arm_i, arm in enumerate(ARMS):
            metrics = score["arms"][arm]["metrics"]
            value = metrics["joint_exact"] if group == "joint" else metrics[group]["accuracy"]
            x = center + (arm_i - 1) * bar_w - bar_w * 0.42
            y = top + plot_h * (1 - value)
            parts.append(f'<rect x="{x}" y="{y}" width="{bar_w*0.84}" height="{top+plot_h-y}" fill="{colors[arm_i]}"/>')
        parts.append(f'<text class="axis" x="{center}" y="{height-22}" text-anchor="middle">{group}</text>')
    parts.append("</svg>")
    return "".join(parts)


def case_rows(cases: dict[str, dict[str, Any]], case_id: str) -> str:
    by_arm = {arm: { (e["direction"], e["query_position"]): e for e in cases[arm][case_id]["events"] } for arm in ARMS}
    keys = sorted(by_arm["future_seed_gdn2"], key=lambda item: (item[0], item[1]))
    rows = []
    for key in keys:
        base = by_arm["future_seed_gdn2"][key]
        predictions = []
        for arm in ARMS:
            event = by_arm[arm][key]
            cls = "correct" if event["correct"] else "wrong"
            predictions.append(f'<td class="{cls}">{event["prediction"]}</td>')
        rows.append(
            "<tr>"
            f"<td>{html.escape(base['direction'])}</td><td>{base['query_position']}</td>"
            f"<td>{base['write_position']}</td><td>{base['distance']:+d}</td>"
            f"<td>{base['key']}</td><td>{base['target']}</td>"
            + "".join(predictions)
            + "</tr>"
        )
    return "".join(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--visual-dir", type=Path, required=True)
    args = parser.parse_args()
    score = json.loads((args.output_dir / "comparison.json").read_text())
    case_lists = {
        "causal_gdn2": json.loads((args.output_dir / "reference" / "causal_gdn2" / "cases.json").read_text()),
        "future_seed_gdn2": json.loads((args.output_dir / "reference" / "future_seed_gdn2" / "cases.json").read_text()),
        "explicit_bidirectional_gdn2": json.loads((args.output_dir / "explicit_bidirectional_gdn2" / "cases.json").read_text()),
    }
    cases = {arm: {case["case_id"]: case for case in rows} for arm, rows in case_lists.items()}
    shared_ids = set.intersection(*(set(rows) for rows in cases.values()))
    ranked = sorted(
        shared_ids,
        key=lambda case_id: (
            -sum(cases[arm][case_id]["future_errors"] + cases[arm][case_id]["past_errors"] for arm in ARMS),
            case_id,
        ),
    )[:12]
    hardest = {case_id: {arm: cases[arm][case_id] for arm in ARMS} for case_id in ranked}
    args.visual_dir.mkdir(parents=True, exist_ok=True)
    (args.visual_dir / "hardest_cases.json").write_text(json.dumps(hardest, indent=2, sort_keys=True) + "\n")
    (args.visual_dir / "summary.json").write_text(json.dumps(score, indent=2, sort_keys=True) + "\n")

    metric_rows = []
    for arm in ARMS:
        metrics = score["arms"][arm]["metrics"]
        system = score.get("systems", {}).get(arm)
        throughput = (
            f"{system['tokens_per_sec_median']:,.0f}" if system else "not measured"
        )
        peak_memory = (
            f"{system['peak_cuda_mem_bytes_median']/2**20:.1f} MiB"
            if system
            else "not measured"
        )
        timing_range = (
            f"{system['tokens_per_sec_relative_range']:.1%}"
            if system
            else "not measured"
        )
        metric_rows.append(
            "<tr>"
            f"<td>{LABELS[arm]}</td><td>{score['arms'][arm]['parameters']:,}</td>"
            f"<td>{metrics['past']['accuracy']:.4f}</td><td>{metrics['future']['accuracy']:.4f}</td>"
            f"<td>{metrics['joint_exact']:.4f}</td>"
            f"<td>{throughput}</td><td>{peak_memory}</td><td>{timing_range}</td>"
            "</tr>"
        )
    case_articles = []
    for case_id in ranked:
        counts = ", ".join(
            f"{LABELS[arm]} {cases[arm][case_id]['future_errors'] + cases[arm][case_id]['past_errors']} errors"
            for arm in ARMS
        )
        case_articles.append(
            '<article class="case">'
            f"<h3>Case {html.escape(case_id)}</h3><p>{html.escape(counts)}</p>"
            '<div class="table-wrap"><table><thead><tr><th>Direction</th><th>Query</th><th>Write</th><th>Distance</th><th>Key</th><th>Target</th>'
            '<th>Causal</th><th>FutureSeed</th><th>Forward + reverse</th></tr></thead><tbody>'
            + case_rows(cases, case_id)
            + "</tbody></table></div></article>"
        )
    gate = "PASS" if score["registered_gate"]["passed"] else "FAIL"
    gate_class = "pass" if gate == "PASS" else "fail"
    cost_note = (
        "Robust cost benchmarks were not run because the explicit bidirectional "
        "carrier missed its preregistered quality gate."
        if not score.get("systems")
        else "Robust cost uses three independent fresh-process measurements."
    )
    document = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>P-CAUSAL-017 explicit bidirectional GDN2 ceiling</title>
<style>
:root{{--ink:#17201c;--muted:#5c6661;--line:#d8dedb;--good:#14745a;--bad:#a43a22;}}
*{{box-sizing:border-box}}body{{margin:0;background:#fff;color:var(--ink);font-family:Inter,ui-sans-serif,system-ui,sans-serif;letter-spacing:0}}
main{{width:min(1160px,calc(100% - 30px));margin:28px auto 64px}}h1{{font-size:28px;margin:0 0 8px}}h2{{font-size:19px;margin:30px 0 12px}}h3{{font-size:14px;margin:0 0 4px}}p{{color:var(--muted);margin:0 0 12px;line-height:1.45}}
.gate{{display:grid;grid-template-columns:120px 1fr;gap:18px;border-block:1px solid var(--line);padding:16px 0;align-items:center}}.badge{{border:2px solid var(--good);color:var(--good);font-weight:800;text-align:center;padding:14px}}.badge.fail{{border-color:var(--bad);color:var(--bad)}}
.legend{{display:flex;gap:18px;flex-wrap:wrap;margin:18px 0}}.swatch{{width:11px;height:11px;display:inline-block;margin-right:6px}}.chart{{border:1px solid var(--line);border-radius:6px;padding:8px}}svg{{width:100%;height:auto}}.grid{{stroke:#e5eae7}}.axis{{font-size:11px;fill:#68726d}}
.table-wrap{{overflow-x:auto}}table{{width:100%;min-width:900px;border-collapse:collapse;font-size:12px}}th,td{{border-bottom:1px solid var(--line);padding:8px;text-align:right}}th:first-child,td:first-child{{text-align:left}}.case{{border-top:1px solid var(--line);padding:14px 0}}.correct{{background:#e4f4ed;color:var(--good);font-weight:700}}.wrong{{background:#fde8e2;color:var(--bad);font-weight:700}}
@media(max-width:680px){{h1{{font-size:24px}}.gate{{grid-template-columns:1fr}}}}
</style></head><body><main>
<h1>FutureSeed versus explicit bidirectional recurrence</h1>
<p>L512 directional MQAR with four associations. The baseline runs independent official-FLA GDN2 streams forward and backward at every layer, then learns a generic linear fusion. FutureSeed never reverses the sequence.</p>
<section class="gate"><div class="badge {gate_class}">{gate}</div><div><strong>Registered carrier and efficiency gate</strong><p>Bidirectional past/future at least 0.95, joint exact at least 0.90; FutureSeed stays within 0.03 future and 0.05 joint exact while using at least 1.25x throughput and at most 0.80x peak memory. {cost_note}</p></div></section>
<div class="legend"><span><i class="swatch" style="background:#59635e"></i>Causal GDN2</span><span><i class="swatch" style="background:#14745a"></i>GDN2 + FutureSeed</span><span><i class="swatch" style="background:#b45f18"></i>Forward + reverse GDN2</span></div>
<div class="chart">{quality_svg(score)}</div>
<h2>Quality and robust cost</h2><div class="table-wrap"><table><thead><tr><th>Model</th><th>Parameters</th><th>Past acc</th><th>Future acc</th><th>Joint exact</th><th>Median tokens/s</th><th>Peak memory</th><th>Timing range</th></tr></thead><tbody>{''.join(metric_rows)}</tbody></table></div>
<h2>Hardest matched cases</h2>{''.join(case_articles)}
</main></body></html>"""
    (args.visual_dir / "index.html").write_text(document)
    print(json.dumps({"gate": gate, "visualized_cases": len(ranked)}, sort_keys=True))


if __name__ == "__main__":
    main()
