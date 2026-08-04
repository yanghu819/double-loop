from __future__ import annotations

import argparse
import html
import json
from pathlib import Path
from typing import Any


ARM_LABELS = {
    "causal_gdn2": "Causal GDN2",
    "future_seed_gdn2": "GDN2 + FutureSeed",
    "bidirectional_attention": "Plain full SDPA",
    "official_bidirectional_mha": "Official MHA, mask removed",
    "rope_bidirectional_attention": "Full SDPA + RoPE",
}
COLORS = {
    "causal_gdn2": "#6f7773",
    "future_seed_gdn2": "#176b55",
    "bidirectional_attention": "#b56817",
    "official_bidirectional_mha": "#9b3f30",
    "rope_bidirectional_attention": "#245ea8",
}


def fmt(value: float, digits: int = 4) -> str:
    return f"{value:.{digits}f}"


def curve_svg(scores: dict[str, dict[str, Any]]) -> str:
    width, height = 980, 350
    left, right, top, bottom = 62, 22, 28, 48
    plot_width = width - left - right
    plot_height = height - top - bottom
    curves = {
        name: score.get("valid_curve", [])
        for name, score in scores.items()
        if score.get("valid_curve")
    }
    max_epoch = max(
        int(row["epoch"])
        for curve in curves.values()
        for row in curve
    )

    def x(epoch: int) -> float:
        return left + plot_width * epoch / max(1, max_epoch)

    def y(value: float) -> float:
        return top + plot_height * (1.0 - value)

    parts = [f'<svg viewBox="0 0 {width} {height}" role="img">']
    for tick in range(5):
        value = tick / 4
        yy = y(value)
        parts.append(
            f'<line x1="{left}" y1="{yy:.1f}" x2="{width-right}" '
            f'y2="{yy:.1f}" class="grid" />'
        )
        parts.append(
            f'<text x="{left-10}" y="{yy+4:.1f}" text-anchor="end" '
            f'class="axis">{value:.2f}</text>'
        )
    for name, curve in curves.items():
        points = " ".join(
            f"{x(int(row['epoch'])):.1f},{y(float(row['valid/accuracy'])):.1f}"
            for row in curve
        )
        parts.append(
            f'<polyline points="{points}" fill="none" '
            f'stroke="{COLORS[name]}" stroke-width="3" />'
        )
    parts.append(
        f'<text x="{left}" y="{height-12}" class="axis">epoch 0</text>'
        f'<text x="{width-right}" y="{height-12}" text-anchor="end" '
        f'class="axis">epoch {max_epoch}</text>'
        "</svg>"
    )
    return "".join(parts)


def event_map(case: dict[str, Any]) -> dict[tuple[str, int], dict[str, Any]]:
    return {
        (event["direction"], int(event["query_position"])): event
        for event in case["events"]
    }


def error_count(case: dict[str, Any]) -> int:
    return int(case["future_errors"]) + int(case["past_errors"])


def binding_diagnostics(cases: dict[str, dict[str, Any]]) -> dict[str, Any]:
    by_direction: dict[str, dict[str, int]] = {
        "past": {"errors": 0, "other_sample_value": 0},
        "future": {"errors": 0, "other_sample_value": 0},
    }
    for case in cases.values():
        valid_targets = {int(event["target"]) for event in case["events"]}
        for event in case["events"]:
            if event["correct"]:
                continue
            direction = str(event["direction"])
            row = by_direction[direction]
            row["errors"] += 1
            if int(event["prediction"]) in valid_targets:
                row["other_sample_value"] += 1
    total_errors = sum(row["errors"] for row in by_direction.values())
    total_swaps = sum(row["other_sample_value"] for row in by_direction.values())
    return {
        "by_direction": by_direction,
        "errors": total_errors,
        "other_sample_value": total_swaps,
        "other_sample_value_fraction": total_swaps / max(1, total_errors),
    }


def select_cases(
    case_maps: dict[str, dict[str, dict[str, Any]]],
) -> list[str]:
    common = set.intersection(*(set(mapping) for mapping in case_maps.values()))
    plain = case_maps["bidirectional_attention"]
    candidate = case_maps["rope_bidirectional_attention"]
    repairs = sorted(
        common,
        key=lambda case_id: (
            -(error_count(plain[case_id]) - error_count(candidate[case_id])),
            error_count(candidate[case_id]),
            case_id,
        ),
    )
    hardest = sorted(
        common,
        key=lambda case_id: (-error_count(candidate[case_id]), case_id),
    )
    selected: list[str] = []
    for case_id in repairs[:6] + hardest[:8]:
        if case_id not in selected:
            selected.append(case_id)
        if len(selected) == 10:
            break
    return selected


def prediction_cell(event: dict[str, Any]) -> str:
    css = "correct" if event["correct"] else "wrong"
    return f'<td class="{css}">{event["prediction"]}</td>'


def case_html(
    case_id: str,
    case_maps: dict[str, dict[str, dict[str, Any]]],
) -> str:
    candidate = case_maps["rope_bidirectional_attention"][case_id]
    ordered_arms = list(ARM_LABELS)
    event_maps = {
        arm: event_map(case_maps[arm][case_id]) for arm in ordered_arms
    }
    rows = []
    for event in candidate["events"]:
        key = (event["direction"], int(event["query_position"]))
        cells = "".join(prediction_cell(event_maps[arm][key]) for arm in ordered_arms)
        rows.append(
            "<tr>"
            f"<td>{html.escape(event['direction'])}</td>"
            f"<td>{event['query_position']}</td>"
            f"<td>{event['write_position']}</td>"
            f"<td>{event['distance']:+d}</td>"
            f"<td>{event['key']}</td>"
            f"<td>{event['target']}</td>"
            f"{cells}</tr>"
        )
    summary = " / ".join(
        f"{ARM_LABELS[arm]} {error_count(case_maps[arm][case_id])} errors"
        for arm in ordered_arms
    )
    prediction_headers = "".join(
        f"<th>{html.escape(ARM_LABELS[arm])}</th>" for arm in ordered_arms
    )
    return (
        '<article class="case">'
        f"<h3>Case {html.escape(case_id)}</h3>"
        f"<p>{html.escape(summary)}</p>"
        '<div class="table-wrap"><table><thead><tr>'
        "<th>Direction</th><th>Query</th><th>Write</th><th>Distance</th>"
        f"<th>Key</th><th>Target</th>{prediction_headers}"
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
    scores = dict(comparison["references"])
    scores["rope_bidirectional_attention"] = comparison["candidate"]
    case_paths = {
        "causal_gdn2": args.output_dir / "references/causal_gdn2/cases.json",
        "future_seed_gdn2": args.output_dir / "references/future_seed_gdn2/cases.json",
        "bidirectional_attention": args.output_dir / "references/bidirectional_attention/cases.json",
        "official_bidirectional_mha": args.output_dir / "references/official_bidirectional_mha/cases.json",
        "rope_bidirectional_attention": args.output_dir / "rope_bidirectional_attention/cases.json",
    }
    case_maps = {
        arm: {row["case_id"]: row for row in json.loads(path.read_text())}
        for arm, path in case_paths.items()
    }
    binding = {
        arm: binding_diagnostics(cases) for arm, cases in case_maps.items()
    }
    selected = select_cases(case_maps)
    args.visual_dir.mkdir(parents=True, exist_ok=True)
    (args.visual_dir / "selected_cases.json").write_text(
        json.dumps(
            {
                case_id: {arm: case_maps[arm][case_id] for arm in ARM_LABELS}
                for case_id in selected
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    summary = {
        "registered_gate": comparison["registered_gate"],
        "metrics": {arm: score["metrics"] for arm, score in scores.items()},
        "parameters": {arm: score["parameters"] for arm, score in scores.items()},
        "throughput_tokens_per_sec": {
            arm: score["warmed_step_benchmark"]["tokens_per_sec"]
            for arm, score in scores.items()
        },
        "peak_cuda_mem_bytes": {
            arm: score["warmed_step_benchmark"]["peak_cuda_mem_bytes"]
            for arm, score in scores.items()
        },
    }
    (args.visual_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    (args.visual_dir / "binding_diagnostics.json").write_text(
        json.dumps(binding, indent=2, sort_keys=True) + "\n"
    )

    metric_rows = []
    for arm in ARM_LABELS:
        score = scores[arm]
        metrics = score["metrics"]
        benchmark = score["warmed_step_benchmark"]
        metric_rows.append(
            "<tr>"
            f"<td><span class=\"swatch\" style=\"background:{COLORS[arm]}\"></span>"
            f"{html.escape(ARM_LABELS[arm])}</td>"
            f"<td>{fmt(metrics['past']['accuracy'])}</td>"
            f"<td>{fmt(metrics['future']['accuracy'])}</td>"
            f"<td>{fmt(metrics['balanced_accuracy'])}</td>"
            f"<td>{fmt(metrics['joint_exact'])}</td>"
            f"<td>{score['parameters']:,}</td>"
            f"<td>{benchmark['tokens_per_sec'] / 1e3:.1f}k</td>"
            f"<td>{benchmark['peak_cuda_mem_bytes'] / 2**20:.1f} MiB</td>"
            "</tr>"
        )
    legend = "".join(
        f'<span><i style="background:{COLORS[arm]}"></i>{html.escape(ARM_LABELS[arm])}</span>'
        for arm in ARM_LABELS
    )
    gate = comparison["registered_gate"]
    gate_word = "PASS" if gate["passed"] else "STOP"
    gate_class = "pass" if gate["passed"] else "stop"
    binding_rows = []
    for arm in ARM_LABELS:
        diagnostic = binding[arm]
        past = diagnostic["by_direction"]["past"]
        future = diagnostic["by_direction"]["future"]
        binding_rows.append(
            "<tr>"
            f"<td>{html.escape(ARM_LABELS[arm])}</td>"
            f"<td>{future['other_sample_value']:,} / {future['errors']:,}</td>"
            f"<td>{past['other_sample_value']:,} / {past['errors']:,}</td>"
            f"<td>{diagnostic['other_sample_value_fraction']:.1%}</td>"
            "</tr>"
        )
    cases = "".join(case_html(case_id, case_maps) for case_id in selected)
    document = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>P-CAUSAL-015: relative-address bidirectional carrier</title>
<style>
:root {{ --ink:#18211d; --muted:#5d6762; --line:#d8dedb; --good:#176b55; --bad:#a4331b; --soft:#f5f7f6; }}
* {{ box-sizing:border-box; }} body {{ margin:0; color:var(--ink); background:#fff; font-family:Inter,ui-sans-serif,system-ui,sans-serif; letter-spacing:0; }}
main {{ width:min(1240px,calc(100% - 28px)); margin:28px auto 72px; }}
h1 {{ font-size:28px; margin:0 0 8px; }} h2 {{ font-size:20px; margin:34px 0 12px; }} h3 {{ font-size:15px; margin:0 0 5px; }} p {{ color:var(--muted); margin:0 0 12px; line-height:1.5; }}
.decision {{ display:grid; grid-template-columns:150px 1fr; gap:18px; align-items:center; padding:18px 0; border-block:1px solid var(--line); }}
.badge {{ width:120px; height:58px; display:grid; place-items:center; border:2px solid currentColor; font-size:20px; font-weight:800; }} .pass {{ color:var(--good); }} .stop {{ color:var(--bad); }}
.table-wrap {{ overflow:auto; border:1px solid var(--line); }} table {{ width:100%; border-collapse:collapse; min-width:980px; }} th,td {{ padding:9px 10px; border-bottom:1px solid var(--line); text-align:right; font-variant-numeric:tabular-nums; white-space:nowrap; }} th:first-child,td:first-child {{ text-align:left; }} th {{ background:var(--soft); font-size:12px; color:var(--muted); }}
.swatch {{ display:inline-block; width:10px; height:10px; margin-right:8px; }}
.chart {{ border-block:1px solid var(--line); padding:12px 0; }} svg {{ width:100%; height:auto; display:block; }} .grid {{ stroke:#e6eae8; stroke-width:1; }} .axis {{ fill:var(--muted); font-size:12px; }}
.legend {{ display:flex; flex-wrap:wrap; gap:14px; margin:8px 0 0; color:var(--muted); font-size:13px; }} .legend i {{ display:inline-block; width:12px; height:3px; margin-right:5px; vertical-align:middle; }}
.case {{ padding:18px 0 26px; border-top:1px solid var(--line); }} .case .correct {{ color:var(--good); font-weight:700; }} .case .wrong {{ color:var(--bad); font-weight:700; }}
.notes {{ background:var(--soft); padding:14px 16px; border-left:4px solid #245ea8; }} code {{ font-family:ui-monospace,SFMono-Regular,monospace; font-size:.92em; }}
@media (max-width:700px) {{ main {{ width:min(100% - 18px,1240px); margin-top:18px; }} h1 {{ font-size:23px; }} .decision {{ grid-template-columns:1fr; }} }}
</style></head><body><main>
<h1>Does relative addressing rescue the bidirectional ceiling?</h1>
<p>P-CAUSAL-015 keeps the old full-attention parameter tensors, data, initialization and optimizer fixed. The only candidate change is parameter-free RoPE on Q/K.</p>
<section class="decision"><div class="badge {gate_class}">{gate_word}</div><div>
<strong>Registered carrier gate</strong>
<p>Past accuracy >= 0.90, future accuracy >= 0.90, and whole-sample exact >= 0.80. Balanced gain over plain SDPA: {gate['balanced_gain_over_plain_sdpa']:+.4f}; joint gain: {gate['joint_gain_over_plain_sdpa']:+.4f}; balanced gap to FutureSeed: {gate['gap_to_future_seed_balanced']:+.4f}.</p>
</div></section>
<h2>Quality and diagnostic cost</h2>
<div class="table-wrap"><table><thead><tr><th>Model</th><th>Past acc</th><th>Future acc</th><th>Balanced</th><th>Joint exact</th><th>Parameters</th><th>Warm train tok/s</th><th>Warm peak</th></tr></thead><tbody>{''.join(metric_rows)}</tbody></table></div>
<p>Throughput and memory are fixed-order diagnostics after independent warmup, not the final paper cost curve.</p>
<h2>Validation opening</h2><div class="chart">{curve_svg(scores)}</div><div class="legend">{legend}</div>
<h2>Binding failure</h2>
<div class="table-wrap"><table><thead><tr><th>Model</th><th>Future wrong-key / errors</th><th>Past wrong-key / errors</th><th>All errors that select another valid value</th></tr></thead><tbody>{''.join(binding_rows)}</tbody></table></div>
<p>An error counts as a wrong-key selection when its prediction is a valid value attached to another key in the same sequence. The failed attention arms recover the value set, but do not bind each value to its own key.</p>
<h2>What changed</h2><div class="notes"><p><code>rope_scale=0</code> is checked bit-exact against the frozen plain SDPA output. RoPE adds no trainable parameter and the attention remains fully noncausal. This experiment tests whether the failed Transformer ceiling lacked a translation-invariant relative address for the adjacent key/value relation.</p></div>
<h2>Same-sequence decisions</h2><p>Green predictions equal the target; red predictions do not. Cases prioritize candidate repairs over plain SDPA, then the candidate's hardest remaining examples.</p>{cases}
</main></body></html>"""
    (args.visual_dir / "index.html").write_text(document)
    print(args.visual_dir / "index.html")


if __name__ == "__main__":
    main()
