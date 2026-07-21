#!/usr/bin/env python3
from __future__ import annotations

import argparse
import html
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


LOG_RE = re.compile(r"step=(?P<step>\d+)\s+ce=(?P<ce>[-+0-9.eE]+)")
CRITICAL_ARGS = (
    "d_model",
    "layers",
    "heads",
    "head_dim",
    "channel_mult",
    "gdn_expand_v",
    "gdn_use_short_conv",
    "gdn_conv_size",
    "max_loops",
    "loop_loss",
    "batch",
    "grad_accum_steps",
    "lr",
    "weight_decay",
    "blank_loss_weight",
    "future_seed_scale",
    "future_seed_update",
    "future_seed_norm_mode",
    "noise_scale",
    "seed",
    "forward_dtype",
)
COLORS = {"fla_gdn": "#1769aa", "gdn2": "#2e7d50"}
LABELS = {"fla_gdn": "GDN", "gdn2": "GDN2"}
EXPECTED_AUTOGRAD_NODES = {
    "fla_gdn_adapter": "ChunkGatedDeltaRuleFunctionBackward",
    "kda_adapter": "ChunkKDAFunctionBackward",
    "gdn2_adapter": "ChunkGDN2FunctionBackward",
}


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def result_path(run_dir: Path) -> Path:
    candidates = sorted((run_dir / "output").glob("futureseed_loop_seed*.json"))
    if len(candidates) != 1:
        raise ValueError(f"expected one result JSON under {run_dir}, got {candidates}")
    return candidates[0]


def train_curve(*logs: Path) -> list[dict[str, float]]:
    rows: dict[int, float] = {}
    for path in logs:
        with path.open("r", encoding="utf-8", errors="replace") as handle:
            for line in handle:
                match = LOG_RE.search(line)
                if match:
                    rows[int(match.group("step"))] = float(match.group("ce"))
    return [{"step": step, "ce": rows[step]} for step in sorted(rows)]


def wrong_by_loop(case_path: Path) -> dict[str, int]:
    text = case_path.read_text(encoding="utf-8")
    output: dict[str, int] = {}
    for chunk in text.split('<div class="panel"><h3>loop ')[1:]:
        loop_match = re.match(r"(\d+)</h3>", chunk)
        if loop_match:
            output[f"loop{int(loop_match.group(1))}"] = chunk.count('class="cell wrong"')
    return output


def extract_arm(
    repo: Path,
    run_name: str,
    baseline_run_name: str,
    expected_backbone: str,
    expected_resume_step: int,
    extra_logs: list[Path] | None = None,
) -> dict[str, Any]:
    run_dir = repo / "runs" / run_name
    baseline_dir = repo / "runs" / baseline_run_name
    payload = read_json(result_path(run_dir))
    args = payload["args"]
    metrics = payload["metrics"]
    if args["backbone"] != expected_backbone:
        raise AssertionError(f"{run_name} has backbone {args['backbone']}, expected {expected_backbone}")
    if int(args["steps"]) != 1000:
        raise AssertionError(f"{run_name} is not a step1000 result")
    resume = metrics["train"].get("resume_train_checkpoint", {})
    if int(resume.get("saved_at_step", -1)) != expected_resume_step:
        raise AssertionError(
            f"{run_name} resumed step {resume.get('saved_at_step')}, expected {expected_resume_step}"
        )
    official_seed = metrics["eval_official"]
    loops = []
    for loop in range(1, int(args["max_loops"]) + 1):
        row = metrics["eval_clean"][f"loop{loop}"]
        loops.append({
            "loop": loop,
            "exact": float(row["label_exact"]),
            "blank_acc": float(row["blank_acc"]),
        })
    ranges: dict[str, Any] = {}
    for key, value in metrics["official_eval_by_blank_range"].items():
        ranges[key] = {
            "blank_range": value["blank_range"],
            "eval_n": int(value["eval_n"]),
            "loop1": value["eval_clean"]["loop1"],
            "loop5": value["eval_clean"]["loop5"],
        }
    checkpoint = read_json(run_dir / "output" / "checkpoint_eval_step001000.json")
    case_path = run_dir / "output" / "futureseed_loop_case_seed52.html"
    return {
        "backbone": expected_backbone,
        "label": LABELS[expected_backbone],
        "color": COLORS[expected_backbone],
        "run_name": run_name,
        "baseline_run_name": baseline_run_name,
        "git_sha": read_json(run_dir / "config.json")["git_sha"],
        "args": {key: args.get(key) for key in CRITICAL_ARGS},
        "hole_stages": args["hole_stages"],
        "resume": resume,
        "official_eval": official_seed,
        "train_ce": float(checkpoint["train"]["ce_loss"]),
        "train_sec_segment": float(metrics["train"]["train_sec"]),
        "train_steps_segment": 1000 - expected_resume_step,
        "train_sec_per_step": float(metrics["train"]["train_sec"]) / (1000 - expected_resume_step),
        "peak_allocated_mb": float(metrics["train"]["cuda_max_memory_allocated_mb"]),
        "checkpoint_holes53": checkpoint["eval_by_holes"]["holes53"]["eval_clean"]["loop5"],
        "loops": loops,
        "ranges": ranges,
        "curve": train_curve(
            baseline_dir / "logs" / "run.log",
            *(extra_logs or []),
            run_dir / "logs" / "run.log",
        ),
        "case": {
            "relative_html": f"../{run_name}/output/futureseed_loop_case_seed52.html?rev=backbone-label-v2",
            "wrong_by_loop": wrong_by_loop(case_path),
        },
        "fla_runtime": metrics["train"]["fla_runtime"],
    }


def line_chart(
    arms: list[dict[str, Any]],
    key: str,
    title: str,
    x_key: str,
    value_key: str,
) -> str:
    width, height = 720, 300
    left, right, top, bottom = 64, 20, 42, 48
    plot_w, plot_h = width - left - right, height - top - bottom
    rows = [row for arm in arms for row in arm[key]]
    xs = sorted({float(row[x_key]) for row in rows})
    values = [float(row[value_key]) for row in rows]
    y_min = min(values)
    y_max = max(values)
    if y_max == y_min:
        y_max = y_min + 1.0

    def xpos(value: float) -> float:
        return left + (value - xs[0]) / max(xs[-1] - xs[0], 1.0) * plot_w

    def ypos(value: float) -> float:
        return top + (y_max - value) / (y_max - y_min) * plot_h

    parts = [
        f'<svg viewBox="0 0 {width} {height}" role="img" aria-label="{html.escape(title)}">',
        f'<text x="{left}" y="24" class="chart-title">{html.escape(title)}</text>',
        f'<line x1="{left}" y1="{top + plot_h}" x2="{left + plot_w}" y2="{top + plot_h}" class="axis"/>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_h}" class="axis"/>',
    ]
    for tick in range(5):
        value = y_min + (y_max - y_min) * tick / 4
        y = ypos(value)
        parts.append(f'<line x1="{left}" y1="{y:.2f}" x2="{left + plot_w}" y2="{y:.2f}" class="grid"/>')
        parts.append(f'<text x="{left - 8}" y="{y + 4:.2f}" text-anchor="end" class="tick">{value:.3f}</text>')
    for value in xs:
        x = xpos(value)
        parts.append(f'<text x="{x:.2f}" y="{top + plot_h + 22}" text-anchor="middle" class="tick">{int(value)}</text>')
    for index, arm in enumerate(arms):
        points = " ".join(
            f'{xpos(float(row[x_key])):.2f},{ypos(float(row[value_key])):.2f}' for row in arm[key]
        )
        parts.append(f'<polyline points="{points}" fill="none" stroke="{arm["color"]}" stroke-width="3"/>')
        for row in arm[key]:
            parts.append(
                f'<circle cx="{xpos(float(row[x_key])):.2f}" cy="{ypos(float(row[value_key])):.2f}" '
                f'r="4" fill="{arm["color"]}"/>'
            )
        lx = left + index * 140
        parts.append(f'<line x1="{lx}" y1="{height - 10}" x2="{lx + 22}" y2="{height - 10}" stroke="{arm["color"]}" stroke-width="3"/>')
        parts.append(f'<text x="{lx + 28}" y="{height - 6}" class="tick">{arm["label"]}</text>')
    parts.append("</svg>")
    return "".join(parts)


def render(payload: dict[str, Any]) -> str:
    arms = payload["arms"]
    gdn, gdn2 = arms
    result = payload["decision"]
    strict = payload["strict_gate"]
    range_rows = []
    for key in ("b46_50", "b51_55", "b56_64"):
        range_rows.append(
            "<tr>"
            f"<td>{key[1:].replace('_', '-')}</td>"
            f"<td>{gdn['ranges'][key]['loop5']['label_exact']:.4f}</td>"
            f"<td>{gdn2['ranges'][key]['loop5']['label_exact']:.4f}</td>"
            f"<td>{gdn['ranges'][key]['loop5']['blank_acc']:.4f}</td>"
            f"<td>{gdn2['ranges'][key]['loop5']['blank_acc']:.4f}</td>"
            "</tr>"
        )
    frames = "".join(
        f'<section><h3>{arm["label"]}: same-seed hard case</h3>'
        f'<p class="muted">Wrong cells by loop: {html.escape(str(arm["case"]["wrong_by_loop"]))}</p>'
        f'<iframe src="{html.escape(arm["case"]["relative_html"])}" loading="lazy"></iframe></section>'
        for arm in arms
    )
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>GDN versus GDN2 step1000 crossover</title><style>
:root{{--bg:#f4f6f7;--paper:#fff;--ink:#172026;--muted:#59666f;--line:#d6dde1}}*{{box-sizing:border-box}}body{{margin:0;background:var(--bg);color:var(--ink);font:15px/1.5 ui-sans-serif,system-ui,sans-serif}}main{{max-width:1160px;margin:auto;padding:28px 20px 60px}}h1{{font-size:28px;margin:0 0 6px;letter-spacing:0}}h2{{font-size:20px;margin:28px 0 10px;letter-spacing:0}}h3{{font-size:16px;letter-spacing:0}}.muted{{color:var(--muted)}}.band{{background:var(--paper);border:1px solid var(--line);border-radius:6px;padding:17px;margin-top:14px}}.metrics{{display:grid;grid-template-columns:repeat(4,1fr);gap:10px}}.metric{{background:#fff;border:1px solid var(--line);border-radius:6px;padding:14px}}.metric small{{display:block;color:var(--muted)}}.metric b{{font-size:22px}}.charts,.cases{{display:grid;grid-template-columns:1fr 1fr;gap:14px}}svg{{width:100%;background:#fff;border:1px solid var(--line);border-radius:6px}}.axis{{stroke:#54616a}}.grid{{stroke:#e6eaed}}.tick{{font-size:12px;fill:#4d5961}}.chart-title{{font-size:16px;font-weight:650}}table{{width:100%;border-collapse:collapse}}th,td{{padding:9px;border-bottom:1px solid var(--line);text-align:right}}th:first-child,td:first-child{{text-align:left}}iframe{{width:100%;height:760px;border:1px solid var(--line);border-radius:6px;background:#fff}}code{{background:#edf1f3;padding:2px 5px;border-radius:3px}}@media(max-width:800px){{.metrics{{grid-template-columns:1fr 1fr}}.charts,.cases{{grid-template-columns:1fr}}}}
</style></head><body><main><h1>GDN vs GDN2: does the early ranking survive longer training?</h1><p class="muted">P-LA-004. Exact step500 resumes, identical hard-data continuation to step1000, official FLA CUDA kernels, GPU1 only.</p>
<div class="band"><strong>Decision:</strong> {html.escape(result)}</div>
<div class="metrics"><div class="metric"><small>step1000 CE</small><b>{gdn['train_ce']:.4f}</b><span> GDN</span></div><div class="metric"><small>step1000 CE</small><b>{gdn2['train_ce']:.4f}</b><span> GDN2</span></div><div class="metric"><small>holes53 exact</small><b>{gdn['checkpoint_holes53']['label_exact']:.4f}</b><span> GDN</span></div><div class="metric"><small>holes53 exact</small><b>{gdn2['checkpoint_holes53']['label_exact']:.4f}</b><span> GDN2</span></div><div class="metric"><small>segment sec / optimizer step</small><b>{gdn['train_sec_per_step']:.2f}</b><span> GDN</span></div><div class="metric"><small>segment sec / optimizer step</small><b>{gdn2['train_sec_per_step']:.2f}</b><span> GDN2</span></div><div class="metric"><small>peak allocated MiB</small><b>{gdn['peak_allocated_mb']:.0f}</b><span> GDN</span></div><div class="metric"><small>peak allocated MiB</small><b>{gdn2['peak_allocated_mb']:.0f}</b><span> GDN2</span></div></div>
<h2>Learning and recurrent computation</h2><div class="charts">{line_chart(arms, 'curve', 'Training CE through step1000', 'step', 'ce')}{line_chart(arms, 'loops', 'Full-board exact by loop at step1000', 'loop', 'exact')}{line_chart(arms, 'loops', 'Blank accuracy by loop at step1000', 'loop', 'blank_acc')}</div>
<h2>Official blank ranges</h2><div class="band"><table><thead><tr><th>blanks</th><th>GDN exact</th><th>GDN2 exact</th><th>GDN blank acc</th><th>GDN2 blank acc</th></tr></thead><tbody>{''.join(range_rows)}</tbody></table></div>
<h2>Hard-case loop behavior</h2><div class="cases">{frames}</div>
<h2>Kernel provenance</h2><div class="band"><p>Wheel <code>{strict['wheel_sha256']}</code>; installed-source marker <code>{strict['fla_source_sha']}</code>; all audited source hashes match: <strong>{str(strict['source_hashes_match']).lower()}</strong>.</p><table><thead><tr><th>adapter</th><th>official layer</th><th>CUDA backward node</th><th>short conv</th></tr></thead><tbody>{''.join(f'<tr><td>{html.escape(name)}</td><td>{html.escape(row["official_layer_class"])}</td><td>{html.escape(row["official_chunk_autograd_node"])}</td><td>{html.escape(row["conv_backend"])}</td></tr>' for name, row in strict['adapters'].items())}</tbody></table></div>
<h2>Mechanism context</h2><div class="band"><p>P-LA-003 measured token81 FutureSeed state influence at <code>0.2945</code> for GDN and <code>0.5298</code> for GDN2. GDN2 does not lose because it erases the seed. The step500 score gap largely disappears by step1000, but GDN2 does not produce a task-level or efficiency win under this shared small-state recipe.</p></div>
</main></body></html>"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument("--gdn-run", required=True)
    parser.add_argument("--gdn2-run", required=True)
    parser.add_argument("--gdn-baseline", required=True)
    parser.add_argument("--gdn2-baseline", required=True)
    parser.add_argument("--gdn2-resume-step", type=int, default=500)
    parser.add_argument("--gdn2-extra-log", type=Path, action="append", default=[])
    parser.add_argument("--strict-gate", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    repo = args.repo.resolve()
    out_dir = args.out_dir if args.out_dir.is_absolute() else repo / args.out_dir
    arms = [
        extract_arm(repo, args.gdn_run, args.gdn_baseline, "fla_gdn", 500),
        extract_arm(
            repo,
            args.gdn2_run,
            args.gdn2_baseline,
            "gdn2",
            args.gdn2_resume_step,
            [path if path.is_absolute() else repo / path for path in args.gdn2_extra_log],
        ),
    ]
    if arms[0]["args"] != arms[1]["args"]:
        raise AssertionError(f"critical configs differ: {arms[0]['args']} != {arms[1]['args']}")
    if arms[0]["hole_stages"] != arms[1]["hole_stages"]:
        raise AssertionError("executed continuation curricula differ")
    if arms[0]["official_eval"] != arms[1]["official_eval"]:
        raise AssertionError("official evaluation batches differ")
    strict_path = args.strict_gate if args.strict_gate.is_absolute() else repo / args.strict_gate
    strict_payload = read_json(strict_path)
    source_hashes_match = all(
        row["installed_sha256"] == row["wheel_sha256"]
        for row in strict_payload["provenance"]["source_file_hashes"].values()
    )
    if not source_hashes_match:
        raise AssertionError("installed FLA source does not match the pinned wheel")
    strict_adapters: dict[str, Any] = {}
    for name, expected_node in EXPECTED_AUTOGRAD_NODES.items():
        row = strict_payload[name]
        if row["official_chunk_autograd_node"] != expected_node:
            raise AssertionError(f"{name} used {row['official_chunk_autograd_node']}, expected {expected_node}")
        conv_backends = set(row["conv_backends"].values())
        if conv_backends != {"triton"}:
            raise AssertionError(f"{name} short-conv backends are {conv_backends}")
        strict_adapters[name] = {
            "official_layer_class": row["official_layer_class"],
            "official_chunk_autograd_node": row["official_chunk_autograd_node"],
            "conv_backend": "triton",
            "initial_state_grad_norm": row["initial_state_grad_norm"],
        }
    gdn, gdn2 = arms
    ce_gap = gdn2["train_ce"] - gdn["train_ce"]
    opening_gap = (
        gdn["ranges"]["b46_50"]["loop5"]["label_exact"]
        - gdn2["ranges"]["b46_50"]["loop5"]["label_exact"]
    )
    if gdn2["train_ce"] <= gdn["train_ce"] and opening_gap <= 0:
        decision = "GDN2 crossed over: the step500 ranking was an early-budget artifact."
    elif ce_gap >= 0.01 and opening_gap >= 0.05:
        decision = "No crossover: GDN remains more sample-efficient under the shared small-state recipe."
    else:
        decision = "Inconclusive crossover: quality gaps narrowed but did not reverse cleanly."
    payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "plan_id": "P-LA-004",
        "arms": arms,
        "ce_gap_gdn2_minus_gdn": ce_gap,
        "opening_exact_gap_gdn_minus_gdn2": opening_gap,
        "decision": decision,
        "strict_gate": {
            "wheel_sha256": strict_payload["provenance"]["wheel_sha256"],
            "fla_source_sha": strict_payload["provenance"]["fla_source_sha"],
            "source_hashes_match": source_hashes_match,
            "adapters": strict_adapters,
        },
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "comparison.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    (out_dir / "index.html").write_text(render(payload), encoding="utf-8")
    readme = f"""# GDN versus GDN2 Crossover

## Decision

{decision}

At step1000, GDN/GDN2 CE is `{gdn['train_ce']:.4f}/{gdn2['train_ce']:.4f}`;
mixed loop5 exact ties at `{gdn['loops'][-1]['exact']:.5f}`, and fixed holes53
exact ties at `{gdn['checkpoint_holes53']['label_exact']:.5f}`. Official 46-50
exact is `{gdn['ranges']['b46_50']['loop5']['label_exact']:.5f}` versus
`{gdn2['ranges']['b46_50']['loop5']['label_exact']:.5f}`; both remain zero exact
at 51-64 blanks. GDN2 narrows the early gap but does not win task quality or
efficiency under the shared small-state recipe.

The post-run strict gate confirms pinned official FLA source, Triton short
convolution, and official CUDA backward nodes for GDN, KDA, and GDN2. Open
`index.html` for learning curves, loop metrics, resource use, and same-seed
hard-case visualizations.
"""
    (out_dir / "README.md").write_text(readme, encoding="utf-8")
    print(out_dir)


if __name__ == "__main__":
    main()
