#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import html
import json
import math
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from summarize_sudoku_backbone_benchmark import primary_case, train_curve


EXPECTED_SOURCE_SHA = "8662e3ade15f27cba53171fb9c18129f1df08c39"
EXPECTED_FLA_SHA = "fe8fce9fc6984f22905f54cfa885dce1502baf26"
EXPECTED_CLASS = "fla.layers.gated_deltanet.GatedDeltaNet"
ALLOWED_ARG_DIFFS = {
    "head_dim": (32, 24),
    "gdn_expand_v": (1.0, 2.0),
}
PATH_ONLY_ARG_DIFFS = {"out_dir", "train_checkpoint_dir"}
LOG_RE = re.compile(
    r"step=(?P<step>\d+)\s+ce=(?P<ce>[-+0-9.eE]+).*?"
    r"loop1=(?P<loop1>[-+0-9.eE]+)\s+loop_last=(?P<loop_last>[-+0-9.eE]+)"
)


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def one_result(run_dir: Path) -> tuple[Path, dict[str, Any]]:
    paths = sorted((run_dir / "output").glob("futureseed_loop_seed*.json"))
    if len(paths) != 1:
        raise AssertionError(f"Expected exactly one result JSON under {run_dir}, got {paths}")
    return paths[0], read_json(paths[0])


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def close(left: Any, right: Any, tolerance: float = 1e-8) -> bool:
    return math.isclose(float(left), float(right), rel_tol=0.0, abs_tol=tolerance)


def compare_args(baseline: dict[str, Any], native: dict[str, Any]) -> list[dict[str, Any]]:
    differences = []
    for key in sorted(set(baseline) | set(native)):
        left, right = baseline.get(key), native.get(key)
        if left == right:
            continue
        differences.append({"key": key, "baseline": left, "native": right})
        if key in ALLOWED_ARG_DIFFS:
            require(
                (left, right) == ALLOWED_ARG_DIFFS[key],
                f"Unexpected {key} transition: {(left, right)!r}",
            )
        elif key in PATH_ONLY_ARG_DIFFS:
            require(
                isinstance(left, str) and isinstance(right, str),
                f"Path-only argument {key} is not a string",
            )
        else:
            raise AssertionError(
                f"Training contract changed outside native geometry: {key}: {left!r} -> {right!r}"
            )
    require(
        {row["key"] for row in differences}
        == set(ALLOWED_ARG_DIFFS) | PATH_ONLY_ARG_DIFFS,
        f"Expected exactly two substantive and two path-only differences, got {differences}",
    )
    return differences


def detailed_curve(log_path: Path) -> list[dict[str, float]]:
    rows: dict[int, dict[str, float]] = {}
    for line in log_path.read_text(encoding="utf-8", errors="replace").splitlines():
        match = LOG_RE.search(line)
        if match:
            step = int(match.group("step"))
            rows[step] = {
                "step": step,
                "ce": float(match.group("ce")),
                "loop1_ce": float(match.group("loop1")),
                "loop5_ce": float(match.group("loop_last")),
            }
    require(sorted(rows) == [100, 200, 300, 400, 500], f"Unexpected training log steps: {rows}")
    return [rows[step] for step in sorted(rows)]


def runtime_checks(train: dict[str, Any], args: dict[str, Any]) -> dict[str, Any]:
    runtime = train["backbone_runtime"]
    require(runtime["implementation"] == "official_fla_fla_gdn", f"Wrong runtime: {runtime}")
    require(runtime["requested_kernel"] == "chunk", f"Wrong kernel request: {runtime}")
    require(not runtime["silent_fallback_allowed"], "Silent fallback was allowed")
    fla = train["fla_runtime"]
    require(fla["strict"], "FLA strict mode was not active")
    require(fla["fla_source_sha"] == EXPECTED_FLA_SHA, f"Wrong FLA source: {fla}")
    require(fla["backend_dispatch_disabled"], "FLA backend dispatch was enabled")
    require(fla["conv_backend"] == "triton", f"Wrong convolution backend: {fla}")
    layers = fla["layers"]
    require(len(layers) == int(args["layers"]) == 10, f"Wrong official layer count: {len(layers)}")
    for index, layer in enumerate(layers):
        require(layer["layer"] == index, f"Missing/reordered layer report: {layer}")
        require(layer["class"] == EXPECTED_CLASS, f"Wrong official class: {layer}")
        require(
            set(layer["conv_backends"].values()) == {"triton"},
            f"Layer {index} changed convolution backend: {layer}",
        )
    return {
        "implementation": runtime["implementation"],
        "kernel": runtime["requested_kernel"],
        "silent_fallback_allowed": runtime["silent_fallback_allowed"],
        "fla_source_sha": fla["fla_source_sha"],
        "official_class": EXPECTED_CLASS,
        "official_layer_count": len(layers),
        "conv_backend": fla["conv_backend"],
        "backend_dispatch_disabled": fla["backend_dispatch_disabled"],
    }


def loop_rows(metrics: dict[str, Any]) -> list[dict[str, float]]:
    rows = []
    for loop in range(1, 6):
        row = metrics["eval_clean"][f"loop{loop}"]
        rows.append(
            {
                "loop": loop,
                "exact": float(row["label_exact"]),
                "blank_acc": float(row["blank_acc"]),
            }
        )
    return rows


def range_rows(metrics: dict[str, Any]) -> dict[str, dict[str, float]]:
    output = {}
    for key in ("b46_50", "b51_55", "b56_64"):
        value = metrics["official_eval_by_blank_range"][key]
        output[key] = {
            "loop1_exact": float(value["eval_clean"]["loop1"]["label_exact"]),
            "loop5_exact": float(value["eval_clean"]["loop5"]["label_exact"]),
            "loop5_blank_acc": float(value["eval_clean"]["loop5"]["blank_acc"]),
        }
    return output


def series_svg(
    baseline: list[dict[str, float]],
    native: list[dict[str, float]],
    x_key: str,
    y_key: str,
    title: str,
) -> str:
    width, height = 650, 270
    left, right, top, bottom = 58, 18, 40, 42
    rows = baseline + native
    xs = sorted({float(row[x_key]) for row in rows})
    ys = [float(row[y_key]) for row in rows]
    y_min, y_max = min(ys), max(ys)
    pad = max((y_max - y_min) * 0.12, 0.002)
    y_min, y_max = y_min - pad, y_max + pad
    plot_w, plot_h = width - left - right, height - top - bottom

    def xpos(value: float) -> float:
        return left + (value - xs[0]) / max(xs[-1] - xs[0], 1.0) * plot_w

    def ypos(value: float) -> float:
        return top + (y_max - value) / max(y_max - y_min, 1e-12) * plot_h

    parts = [
        f'<svg viewBox="0 0 {width} {height}" role="img" aria-label="{html.escape(title)}">',
        f'<text x="{left}" y="23" class="chart-title">{html.escape(title)}</text>',
    ]
    for tick in range(5):
        value = y_min + (y_max - y_min) * tick / 4
        y = ypos(value)
        parts.append(f'<line x1="{left}" y1="{y:.2f}" x2="{left + plot_w}" y2="{y:.2f}" class="grid"/>')
        parts.append(f'<text x="{left - 7}" y="{y + 4:.2f}" text-anchor="end" class="tick">{value:.3f}</text>')
    for value in xs:
        x = xpos(value)
        parts.append(f'<text x="{x:.2f}" y="{height - 18}" text-anchor="middle" class="tick">{int(value)}</text>')
    for label, rows_for_arm, color in (
        ("state-matched K32/V32", baseline, "#176b55"),
        ("native K24/V48", native, "#b04a3a"),
    ):
        points = " ".join(
            f"{xpos(float(row[x_key])):.2f},{ypos(float(row[y_key])):.2f}"
            for row in rows_for_arm
        )
        parts.append(f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="3"/>')
        for row in rows_for_arm:
            parts.append(
                f'<circle cx="{xpos(float(row[x_key])):.2f}" cy="{ypos(float(row[y_key])):.2f}" '
                f'r="4" fill="{color}"/>'
            )
        legend_x = left if label.startswith("state") else left + 220
        parts.append(f'<line x1="{legend_x}" y1="{height - 4}" x2="{legend_x + 20}" y2="{height - 4}" stroke="{color}" stroke-width="3"/>')
        parts.append(f'<text x="{legend_x + 26}" y="{height}" class="tick">{html.escape(label)}</text>')
    parts.append("</svg>")
    return "".join(parts)


def fmt_pct(value: float) -> str:
    return f"{100.0 * value:.2f}%"


def render_html(payload: dict[str, Any]) -> str:
    baseline, native = payload["baseline"], payload["native"]
    range_rows_html = "".join(
        "<tr>"
        f"<td>{label}</td>"
        f"<td>{fmt_pct(baseline['ranges'][key]['loop5_exact'])}</td>"
        f"<td>{fmt_pct(native['ranges'][key]['loop5_exact'])}</td>"
        f"<td>{fmt_pct(native['ranges'][key]['loop5_exact'] - baseline['ranges'][key]['loop5_exact'])}</td>"
        "</tr>"
        for key, label in (("b46_50", "46-50"), ("b51_55", "51-55"), ("b56_64", "56-64"))
    )
    decision_class = "negative" if not payload["hypothesis"]["supported"] else "positive"
    return f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Native GDN geometry diagnostic</title>
<style>
:root{{--bg:#f2f4f5;--paper:#fff;--ink:#182126;--muted:#5d6970;--line:#d7dde1;--red:#a9352b;--green:#176b55}}
*{{box-sizing:border-box}}body{{margin:0;background:var(--bg);color:var(--ink);font:15px/1.55 ui-sans-serif,system-ui,sans-serif}}
main{{max-width:1420px;margin:auto;padding:24px 18px 56px}}h1{{font-size:28px;margin:0;letter-spacing:0}}h2{{font-size:20px;margin:25px 0 9px;letter-spacing:0}}
.muted{{color:var(--muted)}}.band{{background:var(--paper);border:1px solid var(--line);border-radius:6px;padding:15px;margin-top:12px;overflow:auto}}
.verdict{{border-left:5px solid var(--red)}}.positive{{border-left-color:var(--green)}}table{{width:100%;border-collapse:collapse;white-space:nowrap}}
th,td{{padding:9px;border-bottom:1px solid var(--line);text-align:right}}th:first-child,td:first-child{{text-align:left}}
.charts,.cases{{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:12px}}svg{{width:100%;background:#fff;border:1px solid var(--line);border-radius:6px}}
.grid{{stroke:#e5eaed}}.tick{{font-size:12px;fill:#4f5a62}}.chart-title{{font-size:16px;font-weight:650}}iframe{{width:100%;height:730px;border:1px solid var(--line);border-radius:6px;background:#fff}}
code{{background:#e9eef0;padding:2px 5px;border-radius:3px}}@media(max-width:950px){{.charts,.cases{{grid-template-columns:1fr}}}}
</style></head><body><main>
<h1>GDN native state geometry: strict diagnostic</h1>
<p class="muted">Same official Sudoku data, order, loss, optimizer, five-loop training and seed. Only K/V geometry changes.</p>
<div class="band verdict {decision_class}"><strong>结论：</strong>{html.escape(payload["hypothesis"]["decision"])}</div>
<h2>唯一干预</h2><div class="band"><table><thead><tr><th>arm</th><th>K/head</th><th>V/head</th><th>state/layer/sample</th><th>params</th><th>sec/step</th><th>peak GiB</th></tr></thead><tbody>
<tr><td>state-matched</td><td>32</td><td>32</td><td>6,144</td><td>{baseline['parameter_count'] / 1e6:.3f}M</td><td>{baseline['sec_per_step']:.2f}</td><td>{baseline['peak_mib'] / 1024:.2f}</td></tr>
<tr><td>native GDN</td><td>24</td><td>48</td><td>6,912</td><td>{native['parameter_count'] / 1e6:.3f}M</td><td>{native['sec_per_step']:.2f}</td><td>{native['peak_mib'] / 1024:.2f}</td></tr>
</tbody></table></div>
<h2>结果</h2><div class="band"><table><thead><tr><th>metric</th><th>state-matched</th><th>native</th><th>delta</th><th>success gate</th></tr></thead><tbody>
<tr><td>step-500 train CE</td><td>{baseline['train_ce']:.4f}</td><td>{native['train_ce']:.4f}</td><td>{native['train_ce'] - baseline['train_ce']:+.4f}</td><td>native ≤ baseline - 0.03</td></tr>
<tr><td>mixed loop5 exact</td><td>{fmt_pct(baseline['mixed_loop5_exact'])}</td><td>{fmt_pct(native['mixed_loop5_exact'])}</td><td>{fmt_pct(native['mixed_loop5_exact'] - baseline['mixed_loop5_exact'])}</td><td>diagnostic only</td></tr>
<tr><td>46-50 blank loop5 exact</td><td>{fmt_pct(baseline['ranges']['b46_50']['loop5_exact'])}</td><td>{fmt_pct(native['ranges']['b46_50']['loop5_exact'])}</td><td>{fmt_pct(native['ranges']['b46_50']['loop5_exact'] - baseline['ranges']['b46_50']['loop5_exact'])}</td><td>+15 points</td></tr>
<tr><td>51-55 / 56-64 exact</td><td>0 / 0</td><td>0 / 0</td><td>0</td><td>must open to matter</td></tr>
</tbody></table></div>
<div class="charts">
{series_svg(baseline['curve'], native['curve'], 'step', 'ce', 'Training CE')}
{series_svg(baseline['loops'], native['loops'], 'loop', 'exact', 'Mixed full-board exact by loop')}
</div>
<h2>Official blank ranges</h2><div class="band"><table><thead><tr><th>blanks</th><th>state-matched exact</th><th>native exact</th><th>delta</th></tr></thead><tbody>{range_rows_html}</tbody></table></div>
<h2>同一道题逐 loop</h2><p class="muted">Puzzle hash: <code>{html.escape(native['case']['puzzle_sha256'])}</code>. Wrong cells: baseline {html.escape(str(baseline['case']['wrong_by_loop']))}; native {html.escape(str(native['case']['wrong_by_loop']))}.</p>
<div class="cases"><section><h3>state-matched K32/V32</h3><iframe src="{html.escape(baseline['case']['relative_html'])}"></iframe></section><section><h3>native K24/V48</h3><iframe src="{html.escape(native['case']['relative_html'])}"></iframe></section></div>
<h2>实现可信度</h2><div class="band"><p><strong>PASS.</strong> Exact official FLA class, all 10 layers, Triton short convolution, official chunk backward, no backend dispatch, no silent fallback. Torch reference agreement: output max error {payload['preflight']['output_max_abs']:.2e}, state {payload['preflight']['state_max_abs']:.2e}, gradient {payload['preflight']['gradient_max_abs']:.4f}. Shared-shell tensors identical: 77/77. Source worktree was clean at <code>{EXPECTED_SOURCE_SHA}</code>.</p></div>
<h2>解释边界</h2><div class="band"><p>这条实验只回答“state-matched GDN 是否因为 K/V 压缩而被冤枉”。答案是否定的：native geometry 有更大的 value state 和更多参数，但没有达到预注册的 CE 或 46-50 blank 提升，51-64 blank 仍为零。它不证明 GDN 永远不行，也不比较 FutureSeed on/off；它只关闭继续扫 GDN K/V 比例这条低收益路线。</p></div>
</main></body></html>"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument("--native-run", required=True)
    parser.add_argument("--baseline-run", required=True)
    parser.add_argument("--baseline-comparison", type=Path, required=True)
    parser.add_argument("--reference-gate", type=Path, required=True)
    parser.add_argument("--full-stack-gate", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()

    repo = args.repo.resolve()
    native_dir = repo / "runs" / args.native_run
    baseline_dir = repo / "runs" / args.baseline_run
    baseline_comparison_path = (
        args.baseline_comparison
        if args.baseline_comparison.is_absolute()
        else repo / args.baseline_comparison
    )
    reference_path = args.reference_gate if args.reference_gate.is_absolute() else repo / args.reference_gate
    full_stack_path = args.full_stack_gate if args.full_stack_gate.is_absolute() else repo / args.full_stack_gate
    out_dir = args.out_dir if args.out_dir.is_absolute() else repo / args.out_dir

    _, native_payload = one_result(native_dir)
    _, baseline_payload = one_result(baseline_dir)
    native_args, native_metrics = native_payload["args"], native_payload["metrics"]
    baseline_args = baseline_payload["args"]
    differences = compare_args(baseline_args, native_args)

    config = read_json(native_dir / "config.json")
    metadata = read_json(native_dir / "metadata.json")
    score = read_json(native_dir / "score.json")
    checkpoint = read_json(native_dir / "output" / "checkpoint_eval_step000500.json")
    reference = read_json(reference_path)
    full_stack = read_json(full_stack_path)
    require(config["git_sha"] == EXPECTED_SOURCE_SHA and not config["git_dirty"], f"Bad config: {config}")
    require((native_dir / "source_HEAD.txt").read_text().strip() == EXPECTED_SOURCE_SHA, "source_HEAD differs")
    require((native_dir / "source.patch").read_text() == "", "Formal source patch is not empty")
    require((native_dir / "source_snapshot.ref").is_file(), "Missing source snapshot reference")
    require(metadata["git_sha"] == EXPECTED_SOURCE_SHA and not metadata["git_dirty"], f"Bad metadata: {metadata}")
    require(score["git_sha"] == EXPECTED_SOURCE_SHA, f"Bad score source: {score}")
    require(score["score_key"] == "metrics.eval_clean.loop5.label_exact", f"Bad score key: {score}")

    train = native_metrics["train"]
    require(train["data_source"] == "official_sudoku", "Native run did not use official Sudoku data")
    require(train["official_train_size"] == 3831994, "Official train size differs")
    require(train["official_eval_size"] == 422786, "Official eval size differs")
    require(train["effective_batch"] == 128, "Effective batch differs")
    require(train["optimizer_runtime"]["contract"] == "rwkv7_decay_groups", "Optimizer contract differs")
    require(train["parameter_count"] == 5235348, "Run parameter count differs")
    runtime = runtime_checks(train, native_args)

    require(reference["status"] == "PASS" and reference["cuda_visible_devices"] == "0", "Reference gate failed")
    require(full_stack["status"] == "PASS" and full_stack["cuda_visible_devices"] == "0", "Full-stack gate failed")
    require(full_stack["parameter_count"] == train["parameter_count"], "Preflight/run parameter count differs")
    require(full_stack["geometry"]["state_layout"] == "VxK", "Wrong state layout")
    require(full_stack["geometry"]["head_dim_k"] == 24, "Wrong K geometry")
    require(full_stack["geometry"]["head_dim_v"] == 48, "Wrong V geometry")
    require(full_stack["shared_shell"] == {
        "common_parameter_tensors": 77,
        "differing_parameter_tensors": 0,
        "identical_parameter_tensors": 77,
    }, f"Shared shell differs: {full_stack['shared_shell']}")
    require(reference["native_shape_reference"]["output_max_abs"] < 1e-3, "Reference output error too high")
    require(reference["native_shape_reference"]["state_max_abs"] < 1e-3, "Reference state error too high")
    require(reference["native_shape_reference"]["gradient_max_abs"] < 0.08, "Reference gradient error too high")
    require(
        full_stack["adapter"]["official_chunk_autograd_node"] == "ChunkGatedDeltaRuleFunctionBackward",
        "Official chunk backward was not exercised",
    )

    fixed = checkpoint["eval_by_holes"]["holes53"]
    require(fixed["blank_range"] == [53, 53] and fixed["eval_n"] == 512, "Checkpoint h53 eval differs")
    require(close(checkpoint["train"]["ce_loss"], train["train_ce_loss"]), "Checkpoint/final CE differs")
    require(close(score["score"], native_metrics["eval_clean"]["loop5"]["label_exact"]), "Score/final metric differs")

    baseline_aggregate = read_json(baseline_comparison_path)
    baseline_arm = next(arm for arm in baseline_aggregate["arms"] if arm["key"] == "gdn")
    require(baseline_arm["run_name"] == args.baseline_run, "Baseline aggregate points to another run")
    require(baseline_arm["strict_validation"]["status"] == "PASS", "Baseline strict validation failed")
    require(
        native_metrics["eval_official"] == baseline_payload["metrics"]["eval_official"],
        "Official evaluator identity differs",
    )
    require(
        train["official_train_size"] == baseline_payload["metrics"]["train"]["official_train_size"]
        and train["official_eval_size"] == baseline_payload["metrics"]["train"]["official_eval_size"],
        "Official dataset identity differs",
    )

    native_case = primary_case(native_dir)
    baseline_case = primary_case(baseline_dir)
    require(
        native_case["puzzle_sha256"] == baseline_case["puzzle_sha256"],
        "Primary visualization puzzle differs",
    )

    native_curve = detailed_curve(native_dir / "logs" / "run.log")
    require(
        [(row["step"], row["ce"]) for row in native_curve]
        == [(row["step"], row["ce"]) for row in train_curve(native_dir / "logs" / "run.log")],
        "Independent curve parsers disagree",
    )
    baseline = {
        "run_name": args.baseline_run,
        "parameter_count": int(baseline_arm["parameter_count"]),
        "train_ce": float(baseline_arm["train_ce"]),
        "sec_per_step": float(baseline_arm["train_sec_per_step"]),
        "peak_mib": float(baseline_arm["peak_allocated_mib"]),
        "mixed_loop5_exact": float(baseline_arm["loops"][-1]["exact"]),
        "curve": baseline_arm["curve"],
        "loops": baseline_arm["loops"],
        "ranges": {
            key: {
                "loop1_exact": float(baseline_arm["ranges"][key]["loop1"]["label_exact"]),
                "loop5_exact": float(baseline_arm["ranges"][key]["loop5"]["label_exact"]),
                "loop5_blank_acc": float(baseline_arm["ranges"][key]["loop5"]["blank_acc"]),
            }
            for key in ("b46_50", "b51_55", "b56_64")
        },
        "case": {
            "puzzle_sha256": baseline_case["puzzle_sha256"],
            "wrong_by_loop": baseline_case["wrong_by_loop"],
            "relative_html": f"../{args.baseline_run}/output/{baseline_case['path'].name}",
        },
    }
    loops = loop_rows(native_metrics)
    ranges = range_rows(native_metrics)
    native = {
        "run_name": args.native_run,
        "source_sha": EXPECTED_SOURCE_SHA,
        "parameter_count": int(train["parameter_count"]),
        "train_ce": float(train["train_ce_loss"]),
        "sec_per_step": float(train["train_sec"]) / int(train["optimizer_steps"]),
        "peak_mib": float(train["cuda_max_memory_allocated_mb"]),
        "mixed_loop5_exact": loops[-1]["exact"],
        "curve": native_curve,
        "loops": loops,
        "ranges": ranges,
        "runtime": runtime,
        "case": {
            "puzzle_sha256": native_case["puzzle_sha256"],
            "wrong_by_loop": native_case["wrong_by_loop"],
            "relative_html": f"../{args.native_run}/output/{native_case['path'].name}",
        },
    }
    ce_improvement = baseline["train_ce"] - native["train_ce"]
    opening_delta = ranges["b46_50"]["loop5_exact"] - baseline["ranges"]["b46_50"]["loop5_exact"]
    supported = ce_improvement >= 0.03 or opening_delta >= 0.15
    decision = (
        "SUPPORTED: native GDN geometry crossed a preregistered gate."
        if supported
        else (
            "NOT SUPPORTED: native K24/V48 adds state and parameters but misses both gates; "
            f"CE improvement is {ce_improvement:+.4f} (need +0.03) and 46-50 exact delta is "
            f"{opening_delta:+.4f} (need +0.15). Hard 51-64 exact remains zero."
        )
    )
    require(ranges["b51_55"]["loop5_exact"] == 0.0, "Unexpected hard-range result changed")
    require(ranges["b56_64"]["loop5_exact"] == 0.0, "Unexpected hardest-range result changed")

    payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS",
        "integrity": "PASS",
        "hypothesis": {
            "statement": full_stack["hypothesis"],
            "supported": supported,
            "ce_improvement": ce_improvement,
            "ce_improvement_gate": 0.03,
            "official_46_50_exact_delta": opening_delta,
            "official_46_50_delta_gate": 0.15,
            "decision": decision,
        },
        "argument_differences": differences,
        "preflight": {
            "reference": "PASS",
            "full_stack": "PASS",
            **reference["native_shape_reference"],
            "shared_shell": full_stack["shared_shell"],
            "geometry": full_stack["geometry"],
        },
        "baseline": baseline,
        "native": native,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (native_dir / "strict_run_validation.json").write_text(
        json.dumps(
            {
                "status": "PASS",
                "run_name": args.native_run,
                "git_sha": EXPECTED_SOURCE_SHA,
                "integrity": "PASS",
                "hypothesis_supported": supported,
                "puzzle_sha256": native_case["puzzle_sha256"],
                "parameter_count": native["parameter_count"],
                "train_ce": native["train_ce"],
                "train_sec_per_step": native["sec_per_step"],
                "peak_allocated_mib": native["peak_mib"],
                "fixed_holes53_exact": float(fixed["eval_clean"]["loop5"]["label_exact"]),
                "mixed_loop1_exact": loops[0]["exact"],
                "mixed_loop5_exact": loops[-1]["exact"],
                "mixed_loop5_blank_acc": loops[-1]["blank_acc"],
                "official_46_50_exact": ranges["b46_50"]["loop5_exact"],
                "official_51_55_exact": ranges["b51_55"]["loop5_exact"],
                "official_56_64_exact": ranges["b56_64"]["loop5_exact"],
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    (out_dir / "comparison.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    with (out_dir / "benchmark.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(
            [
                "arm",
                "parameters",
                "train_ce",
                "mixed_loop5_exact",
                "official_46_50_exact",
                "official_51_55_exact",
                "official_56_64_exact",
                "sec_per_step",
                "peak_allocated_mib",
            ]
        )
        for name, arm in (("state_matched_k32_v32", baseline), ("native_k24_v48", native)):
            writer.writerow(
                [
                    name,
                    arm["parameter_count"],
                    arm["train_ce"],
                    arm["mixed_loop5_exact"],
                    arm["ranges"]["b46_50"]["loop5_exact"],
                    arm["ranges"]["b51_55"]["loop5_exact"],
                    arm["ranges"]["b56_64"]["loop5_exact"],
                    arm["sec_per_step"],
                    arm["peak_mib"],
                ]
            )
    (out_dir / "index.html").write_text(render_html(payload), encoding="utf-8")
    (out_dir / "README.md").write_text(
        "# Native GDN geometry diagnostic\n\n"
        f"- Integrity: **{payload['integrity']}**\n"
        f"- Hypothesis supported: **{str(supported).lower()}**\n"
        f"- Decision: {decision}\n"
        f"- Native run: `{args.native_run}` at `{EXPECTED_SOURCE_SHA}`\n"
        "- Only substantive argument changes: `head_dim 32 -> 24` and "
        "`gdn_expand_v 1 -> 2`.\n"
        "- No silent fallback, no backend dispatch, official FLA GDN and chunk backward verified.\n",
        encoding="utf-8",
    )
    print(json.dumps({"status": "PASS", "hypothesis_supported": supported, "out_dir": str(out_dir)}))


if __name__ == "__main__":
    main()
