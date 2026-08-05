from __future__ import annotations

import argparse
import html
import json
import math
import statistics
from pathlib import Path
from typing import Any


ARMS = ("bidirectional_bert", "causal_gdn2", "future_seed_gdn2")
PRIMARY_WORKLOADS = (
    "batch1_encoder",
    "batch1_masked_recovery",
    "batch64_encoder",
    "batch64_masked_recovery",
)


def summarize(values: list[float]) -> dict[str, float | int]:
    if not values or any(not math.isfinite(value) for value in values):
        raise RuntimeError(f"Invalid benchmark values: {values}")
    mean = statistics.fmean(values)
    std = statistics.stdev(values) if len(values) > 1 else 0.0
    return {
        "count": len(values),
        "mean": mean,
        "median": statistics.median(values),
        "minimum": min(values),
        "maximum": max(values),
        "coefficient_of_variation": std / mean if mean else 0.0,
        "relative_range": (max(values) - min(values)) / mean if mean else 0.0,
    }


def aggregate_benchmarks(rows: list[dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for workload in PRIMARY_WORKLOADS:
        workload_rows = [row["benchmarks"][workload] for row in rows]
        result[workload] = {
            metric: summarize([float(row[metric]) for row in workload_rows])
            for metric in (
                "latency_ms",
                "examples_per_second",
                "input_tokens_per_second",
                "peak_allocated_bytes",
                "peak_increment_bytes",
            )
        }
    return result


def validate_cases(quality_rows: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    cases_by_arm = {
        arm: {int(case["case_index"]): case for case in row["cases"]}
        for arm, row in quality_rows.items()
    }
    expected_indices = set(range(256))
    if any(set(cases) != expected_indices for cases in cases_by_arm.values()):
        raise RuntimeError("Quality workers did not return the exact 256 windows")

    merged = []
    for index in range(256):
        arm_cases = {arm: cases[index] for arm, cases in cases_by_arm.items()}
        case_ids = {case["case_id"] for case in arm_cases.values()}
        token_rows = {tuple(case["tokens"]) for case in arm_cases.values()}
        if len(case_ids) != 1 or len(token_rows) != 1:
            raise RuntimeError(f"Case identity drift at window {index}")
        masked_by_arm = {
            arm: {
                (int(token["position"]), int(token["target_id"])): token
                for token in case["masked_tokens"]
            }
            for arm, case in arm_cases.items()
        }
        target_keys = {tuple(sorted(tokens)) for tokens in masked_by_arm.values()}
        if len(target_keys) != 1:
            raise RuntimeError(f"Masked targets drifted at window {index}")
        tokens = []
        fs_repairs = 0
        fs_regressions = 0
        fs_vs_bert = 0
        bert_vs_fs = 0
        for key in sorted(next(iter(masked_by_arm.values()))):
            arm_tokens = {arm: rows[key] for arm, rows in masked_by_arm.items()}
            causal_correct = bool(arm_tokens["causal_gdn2"]["correct"])
            fs_correct = bool(arm_tokens["future_seed_gdn2"]["correct"])
            bert_correct = bool(arm_tokens["bidirectional_bert"]["correct"])
            fs_repairs += int(fs_correct and not causal_correct)
            fs_regressions += int(causal_correct and not fs_correct)
            fs_vs_bert += int(fs_correct and not bert_correct)
            bert_vs_fs += int(bert_correct and not fs_correct)
            tokens.append(
                {
                    "position": key[0],
                    "target_id": key[1],
                    "target": arm_tokens["future_seed_gdn2"]["target"],
                    "arms": arm_tokens,
                }
            )
        merged.append(
            {
                "case_index": index,
                "case_id": next(iter(case_ids)),
                "tokens": arm_cases["future_seed_gdn2"]["tokens"],
                "masked_tokens": tokens,
                "fs_repairs_causal": fs_repairs,
                "fs_regressions_causal": fs_regressions,
                "fs_repairs_bert": fs_vs_bert,
                "bert_repairs_fs": bert_vs_fs,
            }
        )
    return merged


def population_diagnostics(cases: list[dict[str, Any]]) -> dict[str, int]:
    return {
        "future_seed_repairs_causal": sum(case["fs_repairs_causal"] for case in cases),
        "future_seed_regressions_causal": sum(
            case["fs_regressions_causal"] for case in cases
        ),
        "future_seed_repairs_bert": sum(case["fs_repairs_bert"] for case in cases),
        "bert_repairs_future_seed": sum(case["bert_repairs_fs"] for case in cases),
    }


def select_cases(cases: list[dict[str, Any]], count: int = 12) -> list[dict[str, Any]]:
    ranked = sorted(
        cases,
        key=lambda case: (
            -(case["fs_repairs_causal"] - case["fs_regressions_causal"]),
            -(case["fs_repairs_bert"] - case["bert_repairs_fs"]),
            case["case_index"],
        ),
    )
    positive = ranked[: count // 2]
    hard = sorted(
        cases,
        key=lambda case: (
            case["fs_repairs_causal"] - case["fs_regressions_causal"],
            case["fs_repairs_bert"] - case["bert_repairs_fs"],
            case["case_index"],
        ),
    )[: count - len(positive)]
    selected = []
    seen = set()
    for case in positive + hard + ranked:
        if case["case_index"] not in seen:
            selected.append(case)
            seen.add(case["case_index"])
        if len(selected) == count:
            break
    return selected


def render_html(score: dict[str, Any], cases: list[dict[str, Any]]) -> str:
    quality = score["quality"]
    comparison = score["comparison"]
    benchmark_rows = []
    for workload in PRIMARY_WORKLOADS:
        for arm in ARMS:
            result = score["benchmarks"][arm][workload]
            benchmark_rows.append(
                "<tr>"
                f"<td>{html.escape(workload)}</td>"
                f"<td>{html.escape(arm)}</td>"
                f"<td>{result['latency_ms']['median']:.3f}</td>"
                f"<td>{result['input_tokens_per_second']['median']:,.0f}</td>"
                f"<td>{result['peak_allocated_bytes']['median'] / 2**20:.1f}</td>"
                f"<td>{result['input_tokens_per_second']['coefficient_of_variation']:.3f}</td>"
                "</tr>"
            )

    cards = []
    labels = {
        "causal_gdn2": "causal GDN2",
        "future_seed_gdn2": "GDN2 + FutureSeed",
        "bidirectional_bert": "bidirectional BERT",
    }
    for case in cases:
        masked = {row["position"]: row for row in case["masked_tokens"]}
        token_spans = []
        for position, token in enumerate(case["tokens"]):
            token_spans.append(
                '<span class="mask">[MASK]</span>'
                if position in masked
                else f"<span>{html.escape(token)}</span>"
            )
        rows = []
        for token in case["masked_tokens"]:
            cells = [f"<td>{token['position']}</td>", f"<td>{html.escape(token['target'])}</td>"]
            for arm in ("causal_gdn2", "future_seed_gdn2", "bidirectional_bert"):
                item = token["arms"][arm]
                css = "good" if item["correct"] else "bad"
                cells.append(
                    f'<td class="{css}">{html.escape(item["prediction"])}</td>'
                )
                cells.append(f"<td>{item['target_probability']:.3f}</td>")
            rows.append(f"<tr>{''.join(cells)}</tr>")
        headers = "".join(
            f"<th>{html.escape(labels[arm])}</th><th>P(target)</th>"
            for arm in ("causal_gdn2", "future_seed_gdn2", "bidirectional_bert")
        )
        cards.append(
            '<section class="case">'
            f"<h2>Window {case['case_index']} | FS vs causal "
            f"+{case['fs_repairs_causal']}/-{case['fs_regressions_causal']} | "
            f"FS vs BERT +{case['fs_repairs_bert']}/-{case['bert_repairs_fs']}</h2>"
            f'<div class="tokens">{" ".join(token_spans)}</div>'
            '<div class="table-wrap"><table><thead><tr><th>pos</th><th>target</th>'
            f"{headers}</tr></thead><tbody>{''.join(rows)}</tbody></table></div></section>"
        )

    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>P-CAUSAL-023 FutureSeed inference frontier</title>
<style>
body{{font:14px/1.45 ui-monospace,SFMono-Regular,Menlo,monospace;margin:0;color:#17202a;background:#f4f6f7}}
header{{background:#17202a;color:#fff;padding:24px max(24px,calc((100% - 1180px)/2))}}
main{{max-width:1180px;margin:0 auto;padding:20px}} .summary{{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:8px}}
.metric,.case,.benchmark{{background:#fff;border:1px solid #ccd1d1;border-radius:6px;padding:14px}} .case,.benchmark{{margin-top:14px}}
.metric b{{display:block;font-size:20px}} .tokens{{padding:12px;background:#f8f9f9;line-height:2;overflow-wrap:anywhere}}
.tokens span{{padding:2px}} .mask,.good{{color:#117864;font-weight:700}} .bad{{color:#b03a2e;font-weight:700}}
.table-wrap{{overflow-x:auto}} table{{border-collapse:collapse;width:100%;margin-top:12px;min-width:900px}}
th,td{{border-bottom:1px solid #e5e7e9;padding:7px;text-align:left;vertical-align:top}}
@media(max-width:720px){{.summary{{grid-template-columns:1fr 1fr}} table{{font-size:12px}}}}
</style></head><body><header><h1>P-CAUSAL-023 fixed L128 quality-cost frontier</h1>
<p>Frozen official BERT-Tiny and frozen P022 GDN2 checkpoints. Five fresh processes per arm; no retraining or length extrapolation.</p></header>
<main><div class="summary">
<div class="metric">BERT accuracy<b>{quality['bidirectional_bert']['masked_accuracy']:.4f}</b></div>
<div class="metric">FutureSeed accuracy<b>{quality['future_seed_gdn2']['masked_accuracy']:.4f}</b></div>
<div class="metric">batch64 speed ratio<b>{comparison['future_seed_vs_bert']['batch64_masked_recovery_throughput_ratio']:.2f}x</b></div>
<div class="metric">cost gate<b>{'PASS' if score['gate']['cost_passed'] else 'MISS'}</b></div>
</div><section class="benchmark"><h2>Robust benchmark medians</h2><div class="table-wrap"><table><thead><tr>
<th>workload</th><th>arm</th><th>latency ms</th><th>tokens/s</th><th>peak MiB</th><th>throughput CV</th>
</tr></thead><tbody>{''.join(benchmark_rows)}</tbody></table></div></section>{''.join(cards)}</main></body></html>"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-repetitions", type=int, default=5)
    args = parser.parse_args()

    rows = [json.loads(path.read_text(encoding="utf-8")) for path in sorted(args.raw_dir.glob("*.json"))]
    by_arm = {arm: [row for row in rows if row["arm"] == arm] for arm in ARMS}
    for arm, arm_rows in by_arm.items():
        repetitions = sorted(int(row["repetition"]) for row in arm_rows)
        if repetitions != list(range(args.expected_repetitions)):
            raise RuntimeError(f"{arm} repetition set drifted: {repetitions}")
        hashes = {row["protocol"]["validation_tensor_sha256"] for row in arm_rows}
        parameters = {int(row["model"]["parameters"]) for row in arm_rows}
        checkpoints = {row["model"]["checkpoint_sha256"] for row in arm_rows}
        if len(hashes) != 1 or len(parameters) != 1 or len(checkpoints) != 1:
            raise RuntimeError(f"{arm} provenance changed across fresh processes")

    quality_rows = {}
    for arm, arm_rows in by_arm.items():
        candidates = [row for row in arm_rows if row["quality"] is not None]
        if len(candidates) != 1:
            raise RuntimeError(f"{arm} must have exactly one quality evaluation")
        quality_rows[arm] = candidates[0]
    quality = {arm: row["quality"] for arm, row in quality_rows.items()}
    cases = validate_cases(quality_rows)
    selected_cases = select_cases(cases)
    population = population_diagnostics(cases)
    benchmarks = {arm: aggregate_benchmarks(arm_rows) for arm, arm_rows in by_arm.items()}

    def median(arm: str, workload: str, metric: str) -> float:
        return float(benchmarks[arm][workload][metric]["median"])

    fs_vs_bert = {
        "batch1_masked_recovery_latency_speedup": (
            median("bidirectional_bert", "batch1_masked_recovery", "latency_ms")
            / median("future_seed_gdn2", "batch1_masked_recovery", "latency_ms")
        ),
        "batch64_masked_recovery_throughput_ratio": (
            median("future_seed_gdn2", "batch64_masked_recovery", "input_tokens_per_second")
            / median("bidirectional_bert", "batch64_masked_recovery", "input_tokens_per_second")
        ),
        "batch64_encoder_throughput_ratio": (
            median("future_seed_gdn2", "batch64_encoder", "input_tokens_per_second")
            / median("bidirectional_bert", "batch64_encoder", "input_tokens_per_second")
        ),
        "batch64_masked_recovery_peak_memory_ratio": (
            median("future_seed_gdn2", "batch64_masked_recovery", "peak_allocated_bytes")
            / median("bidirectional_bert", "batch64_masked_recovery", "peak_allocated_bytes")
        ),
    }
    quality_passed = (
        quality["future_seed_gdn2"]["masked_accuracy"]
        >= quality["bidirectional_bert"]["masked_accuracy"] - 0.02
        and quality["future_seed_gdn2"]["masked_ce"]
        <= quality["bidirectional_bert"]["masked_ce"] + 0.10
    )
    stable = all(
        benchmarks[arm][workload]["input_tokens_per_second"]["coefficient_of_variation"]
        <= 0.10
        for arm in ARMS
        for workload in PRIMARY_WORKLOADS
    )
    primary_speed = fs_vs_bert["batch64_masked_recovery_throughput_ratio"] >= 1.20
    memory_trade = (
        fs_vs_bert["batch64_masked_recovery_peak_memory_ratio"] <= 0.80
        and fs_vs_bert["batch64_masked_recovery_throughput_ratio"] >= 0.90
    )
    cost_passed = stable and (primary_speed or memory_trade)
    status = (
        "quality_and_cost_supported"
        if quality_passed and cost_passed
        else "quality_supported_cost_missed"
        if quality_passed
        else "quality_frontier_missed"
    )
    score = {
        "plan": "P-CAUSAL-023",
        "status": status,
        "claim_boundary": (
            "Frozen length-128 inference quality and systems cost only. The BERT and "
            "GDN2 checkpoints have different pretraining provenance; this is not a "
            "training-efficiency, equal-data, or length-scaling claim."
        ),
        "protocol": {
            "sequence_length": 128,
            "repetitions_per_arm": args.expected_repetitions,
            "fresh_process_per_repetition": True,
            "quality_windows": 256,
            "quality_masked_tokens": 4_742,
            "length_extrapolation": False,
            "length_extrapolation_reason": (
                "The frozen P022 GDN2 checkpoint has learned absolute position "
                "embeddings registered only through position 127."
            ),
        },
        "quality": quality,
        "benchmarks": benchmarks,
        "comparison": {"future_seed_vs_bert": fs_vs_bert},
        "population": population,
        "gate": {
            "quality_passed": quality_passed,
            "timing_stable": stable,
            "cost_passed": cost_passed,
            "checks": {
                "quality_accuracy_within_0.02": (
                    quality["future_seed_gdn2"]["masked_accuracy"]
                    >= quality["bidirectional_bert"]["masked_accuracy"] - 0.02
                ),
                "quality_ce_within_0.10": (
                    quality["future_seed_gdn2"]["masked_ce"]
                    <= quality["bidirectional_bert"]["masked_ce"] + 0.10
                ),
                "all_primary_throughput_cv_at_most_0.10": stable,
                "batch64_end_to_end_speed_at_least_1.20x": primary_speed,
                "memory_at_most_0.80x_with_throughput_at_least_0.90x": memory_trade,
            },
        },
        "raw_files": [path.name for path in sorted(args.raw_dir.glob("*.json"))],
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "score.json").write_text(
        json.dumps(score, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (args.output_dir / "hardest_cases.json").write_text(
        json.dumps(selected_cases, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (args.output_dir / "index.html").write_text(
        render_html(score, selected_cases), encoding="utf-8"
    )
    print(json.dumps(score, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
