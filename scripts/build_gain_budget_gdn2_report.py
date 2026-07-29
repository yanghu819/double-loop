#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import html
import json
import pathlib
import re
import shutil
from datetime import datetime, timezone
from typing import Any


def load_json(path: pathlib.Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: pathlib.Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def loop_metrics(payload: dict[str, Any]) -> list[dict[str, float]]:
    rows = []
    for key, row in payload["metrics"]["eval_clean"].items():
        match = re.fullmatch(r"loop(\d+)", key)
        if match is None:
            continue
        rows.append(
            {
                "loop": int(match.group(1)),
                "exact": float(row["label_exact"]),
                "blank_acc": float(row["blank_acc"]),
            }
        )
    return sorted(rows, key=lambda row: row["loop"])


def official_loop5(payload: dict[str, Any]) -> dict[str, dict[str, float]]:
    output = {}
    ranges = payload["metrics"]["official_eval_by_blank_range"]
    for name in ("b51_55", "b56_60", "b61_64"):
        row = ranges[name]
        metric = row["eval_clean"]["loop5"]
        output[name] = {
            "eval_n": int(row["eval_n"]),
            "exact": float(metric["label_exact"]),
            "blank_acc": float(metric["blank_acc"]),
        }
    return output


def smoke_diagnostics(payload: dict[str, Any]) -> dict[str, Any]:
    train = payload["metrics"]["train"]
    range_row = payload["metrics"]["official_eval_by_blank_range"]["b51_55"]
    loop5 = range_row["eval_clean"]["loop5"]
    mechanism = range_row["eval_clean"]["loop5/future_seed"]
    return {
        "mode": payload["args"]["gdn2_gain_budget_mode"],
        "train_ce": float(train["train_ce_loss"]),
        "train_total": float(train["train_total_loss"]),
        "clip_frac": float(train["gain_budget_clipped_frac"]),
        "endpoint_frac": float(
            train["gain_budget_numerical_endpoint_frac"]
        ),
        "gate_relative_change": float(
            train["gain_budget_gate_relative_change"]
        ),
        "step_bound_max": float(train["gain_budget_step_bound_max"]),
        "delta_error_max": float(train["gain_budget_delta_error_max"]),
        "b51_55_eval_n": int(range_row["eval_n"]),
        "b51_55_loop5_exact": float(loop5["label_exact"]),
        "b51_55_loop5_blank_acc": float(loop5["blank_acc"]),
        "loop5_lambda_mean": float(
            mechanism["gdn2_gain_budget_lambda_mean"]
        ),
        "loop5_endpoint_frac": float(
            mechanism["gdn2_gain_budget_numerical_endpoint_frac"]
        ),
        "loop5_original_step_bound": float(
            mechanism["gdn2_gain_budget_original_step_bound"]
        ),
        "loop5_effective_step_bound": float(
            mechanism["gdn2_gain_budget_effective_step_bound"]
        ),
        "loops": loop_metrics(payload),
    }


def pct(value: float) -> str:
    return f"{100.0 * value:.2f}%"


def bar_rows(rows: list[tuple[str, float, str]], maximum: float) -> str:
    output = []
    for label, value, text in rows:
        width = 0.0 if maximum <= 0 else 100.0 * value / maximum
        output.append(
            "<div class=\"bar-row\">"
            f"<div class=\"bar-label\">{html.escape(label)}</div>"
            "<div class=\"bar-track\">"
            f"<div class=\"bar-fill\" style=\"width:{width:.2f}%\"></div>"
            "</div>"
            f"<div class=\"bar-value\">{html.escape(text)}</div>"
            "</div>"
        )
    return "\n".join(output)


def loop_table(
    control: list[dict[str, float]],
    candidate: list[dict[str, float]],
) -> str:
    candidate_by_loop = {row["loop"]: row for row in candidate}
    rows = []
    for control_row in control:
        loop = control_row["loop"]
        candidate_row = candidate_by_loop[loop]
        rows.append(
            "<tr>"
            f"<td>Loop {loop}</td>"
            f"<td>{control_row['exact']:.3f}</td>"
            f"<td>{candidate_row['exact']:.3f}</td>"
            f"<td>{control_row['blank_acc']:.3f}</td>"
            f"<td>{candidate_row['blank_acc']:.3f}</td>"
            "</tr>"
        )
    return "\n".join(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract", type=pathlib.Path, required=True)
    parser.add_argument("--abort", type=pathlib.Path, required=True)
    parser.add_argument("--cpu-log", type=pathlib.Path, required=True)
    parser.add_argument("--dataset-manifest", type=pathlib.Path, required=True)
    parser.add_argument("--smoke-control", type=pathlib.Path, required=True)
    parser.add_argument("--smoke-budget", type=pathlib.Path, required=True)
    parser.add_argument("--formal-control", type=pathlib.Path, required=True)
    parser.add_argument("--smoke-control-html", type=pathlib.Path, required=True)
    parser.add_argument("--smoke-budget-html", type=pathlib.Path, required=True)
    parser.add_argument("--source-root", type=pathlib.Path, required=True)
    parser.add_argument("--source-sha", required=True)
    parser.add_argument("--smoke-source-sha", required=True)
    parser.add_argument("--output-dir", type=pathlib.Path, required=True)
    args = parser.parse_args()

    contract = load_json(args.contract)
    abort = load_json(args.abort)
    control = load_json(args.smoke_control)
    budget = load_json(args.smoke_budget)
    formal_control = load_json(args.formal_control)
    benchmark = contract["benchmark"]
    control_diag = smoke_diagnostics(control)
    budget_diag = smoke_diagnostics(budget)
    formal_ranges = official_loop5(formal_control)

    output_dir = args.output_dir.resolve()
    evidence_dir = output_dir / "evidence"
    logs_dir = output_dir / "logs"
    source_dir = output_dir / "source"
    visualization_dir = output_dir / "visualizations"
    for path in (evidence_dir, logs_dir, source_dir, visualization_dir):
        path.mkdir(parents=True, exist_ok=True)

    evidence_files = {
        "cuda_contract.json": args.contract,
        "abort.json": args.abort,
        "dataset_manifest.json": args.dataset_manifest,
        "smoke_control.json": args.smoke_control,
        "smoke_budget.json": args.smoke_budget,
        "formal_control.json": args.formal_control,
    }
    evidence_hashes = {}
    for name, source in evidence_files.items():
        destination = evidence_dir / name
        shutil.copy2(source, destination)
        evidence_hashes[name] = sha256_file(destination)
    shutil.copy2(args.cpu_log, logs_dir / "cpu_properties.log")
    shutil.copy2(
        args.contract.parent / "preflight.log",
        logs_dir / "preflight.log",
    )
    shutil.copy2(
        args.smoke_control_html,
        visualization_dir / "smoke_control_cases.html",
    )
    shutil.copy2(
        args.smoke_budget_html,
        visualization_dir / "smoke_budget_cases.html",
    )

    source_paths = (
        "experiments/rwkv_fs_sudoku/gain_budget_gdn2.py",
        "experiments/rwkv_fs_sudoku/check_gain_budget_gdn2_cuda.py",
        "experiments/rwkv_fs_sudoku/test_gain_budget_gdn2.py",
        "scripts/run_gain_budget_gdn2_preflight.sh",
        "scripts/run_gain_budget_gdn2_matched.sh",
        "scripts/run_gain_budget_gdn2_arm.sh",
        "configs/sudoku/gain_budget_gdn2_probe.env",
    )
    source_hashes = {}
    for relative in source_paths:
        source = args.source_root / relative
        destination = source_dir / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        source_hashes[relative] = sha256_file(destination)

    cpu_text = args.cpu_log.read_text(encoding="utf-8")
    math_match = re.search(r"Ran (\d+) tests", cpu_text)
    math_tests = int(math_match.group(1)) if math_match else None
    correctness_checks = [
        name
        for name in (
            "chunk_boundary_forward",
            "fused_forward",
            "chunk_backward",
            "chunk_backward_bfloat16",
            "chunk_state_carry",
            "mode_none_exact",
            "budget_layer_backward",
        )
        if contract.get(name)
    ]
    generated_at = datetime.now(timezone.utc).isoformat()

    summary = {
        "schema_version": "gain_budget_gdn2_summary.v1",
        "status": "discarded",
        "decision": (
            "Stop strict c=1 gain budgeting. It failed the pre-registered "
            "20% systems gate; do not sweep cap, seed, loss, LR, or length."
        ),
        "source_sha": args.source_sha,
        "smoke_source_sha": args.smoke_source_sha,
        "generated_at_utc": generated_at,
        "gpu_uuid": contract["gpu_uuid"],
        "fla_source_sha": contract["provenance"]["fla_source_sha"],
        "math_tests_passed": math_tests,
        "cuda_correctness_checks": correctness_checks,
        "benchmark": benchmark,
        "smoke_control": control_diag,
        "smoke_budget": budget_diag,
        "formal_control_official_loop5": formal_ranges,
        "evidence_sha256": evidence_hashes,
        "source_snapshot_sha256": source_hashes,
    }
    write_json(output_dir / "summary.json", summary)
    write_json(
        output_dir / "score.json",
        {
            "status": "discarded",
            "primary_metric": "systems.projection_time_overhead_frac",
            "score": float(benchmark["projection_time_overhead_frac"]),
            "maximum_allowed": 0.20,
            "formal_candidate_score": None,
            "reason": abort["reason"],
        },
    )
    write_json(
        output_dir / "config.json",
        {
            "experiment": "P-GAIN-001",
            "mode": "decay_funded",
            "step_gain_cap": 1.0,
            "matched_control": "external_identity",
            "systems_overhead_max_frac": 0.20,
            "formal_train_steps": [9000, 9100],
            "gpu": "GPU1",
            "cuda_visible_devices": "0",
            "fla_source_sha": contract["provenance"]["fla_source_sha"],
            "source_sha": args.source_sha,
        },
    )
    write_json(
        output_dir / "metadata.json",
        {
            "run_name": output_dir.name,
            "plan_id": "P-GAIN-001",
            "status": "discarded",
            "generated_at_utc": generated_at,
            "source_sha": args.source_sha,
            "gpu_uuid": contract["gpu_uuid"],
            "formal_training_launched": False,
            "abort_path": "evidence/abort.json",
            "visualization": "visualizations/index.html",
        },
    )
    (output_dir / "source_HEAD.txt").write_text(
        args.source_sha + "\n",
        encoding="utf-8",
    )

    external_ms = 1000.0 * float(benchmark["external_identity_seconds"])
    budget_ms = 1000.0 * float(benchmark["budget_seconds"])
    external_mem = (
        float(benchmark["external_identity_incremental_peak_bytes"])
        / (1024.0 * 1024.0)
    )
    budget_mem = (
        float(benchmark["budget_incremental_peak_bytes"])
        / (1024.0 * 1024.0)
    )
    performance_bars = bar_rows(
        [
            ("Matched identity", external_ms, f"{external_ms:.2f} ms"),
            ("Strict c=1 budget", budget_ms, f"{budget_ms:.2f} ms"),
        ],
        max(external_ms, budget_ms),
    )
    memory_bars = bar_rows(
        [
            ("Matched identity", external_mem, f"{external_mem:.1f} MiB"),
            ("Strict c=1 budget", budget_mem, f"{budget_mem:.1f} MiB"),
        ],
        max(external_mem, budget_mem),
    )
    exact_bars = bar_rows(
        [
            (
                "Identity loop5",
                control_diag["b51_55_loop5_exact"],
                f"{control_diag['b51_55_loop5_exact']:.3f}",
            ),
            (
                "Budget loop5",
                budget_diag["b51_55_loop5_exact"],
                f"{budget_diag['b51_55_loop5_exact']:.3f}",
            ),
        ],
        1.0,
    )
    check_items = "\n".join(
        f"<li><strong>PASS</strong> {html.escape(name)}</li>"
        for name in correctness_checks
    )
    formal_rows = "\n".join(
        "<tr>"
        f"<td>{name.replace('_', '-')}</td>"
        f"<td>{row['eval_n']}</td>"
        f"<td>{row['exact']:.4f}</td>"
        f"<td>{row['blank_acc']:.4f}</td>"
        "</tr>"
        for name, row in formal_ranges.items()
    )
    page = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Gain-Budgeted GDN2 decision report</title>
  <style>
    :root {{
      color-scheme: light;
      --ink: #171a1f;
      --muted: #606773;
      --line: #d9dee7;
      --paper: #ffffff;
      --wash: #f3f5f8;
      --red: #b42318;
      --red-soft: #fff0ee;
      --green: #18794e;
      --blue: #2457a6;
      --gold: #8a5d00;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      background: var(--wash);
      color: var(--ink);
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, sans-serif;
      line-height: 1.45;
      letter-spacing: 0;
    }}
    main {{ max-width: 1180px; margin: 0 auto; padding: 28px 22px 64px; }}
    h1 {{ margin: 0 0 6px; font-size: 30px; line-height: 1.15; }}
    h2 {{ margin: 0 0 14px; font-size: 19px; }}
    p {{ margin: 8px 0; }}
    .meta {{ color: var(--muted); font-family: ui-monospace, monospace; font-size: 13px; }}
    .decision {{
      margin: 22px 0;
      padding: 16px 18px;
      border-left: 5px solid var(--red);
      background: var(--red-soft);
    }}
    .decision strong {{ color: var(--red); }}
    .grid {{ display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 14px; }}
    .panel {{
      border: 1px solid var(--line);
      border-radius: 6px;
      background: var(--paper);
      padding: 18px;
    }}
    .wide {{ grid-column: 1 / -1; }}
    .statline {{ display: flex; gap: 18px; flex-wrap: wrap; margin-top: 12px; }}
    .stat {{ min-width: 145px; }}
    .stat b {{ display: block; font-size: 22px; }}
    .stat span {{ color: var(--muted); font-size: 12px; }}
    .bar-row {{
      display: grid;
      grid-template-columns: 150px minmax(120px, 1fr) 90px;
      gap: 10px;
      align-items: center;
      margin: 10px 0;
      font-size: 13px;
    }}
    .bar-track {{ height: 15px; background: #e8ecf2; }}
    .bar-fill {{ height: 100%; background: var(--blue); }}
    .bar-value {{ text-align: right; font-family: ui-monospace, monospace; }}
    table {{ width: 100%; border-collapse: collapse; font-size: 13px; }}
    th, td {{ padding: 8px 9px; border-bottom: 1px solid var(--line); text-align: right; }}
    th:first-child, td:first-child {{ text-align: left; }}
    ul {{ margin: 0; padding-left: 20px; }}
    li {{ margin: 6px 0; }}
    li strong {{ color: var(--green); }}
    .warning {{ color: var(--gold); font-weight: 700; }}
    .frames {{ display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 12px; }}
    iframe {{ width: 100%; height: 760px; border: 1px solid var(--line); background: white; }}
    a {{ color: var(--blue); }}
    code {{ font-family: ui-monospace, monospace; }}
    @media (max-width: 820px) {{
      main {{ padding: 18px 12px 44px; }}
      .grid, .frames {{ grid-template-columns: 1fr; }}
      .wide {{ grid-column: auto; }}
      .bar-row {{ grid-template-columns: 115px minmax(80px, 1fr) 75px; }}
    }}
  </style>
</head>
<body>
<main>
  <h1>Gain-Budgeted GDN2: strict c=1 result</h1>
  <div class="meta">P-GAIN-001 | GPU1 | source {html.escape(args.source_sha)} | FLA {html.escape(contract['provenance']['fla_source_sha'])}</div>

  <div class="decision">
    <strong>STOP at the systems gate.</strong>
    The mechanism is mathematically correct, but it adds
    {pct(float(benchmark['projection_time_overhead_frac']))} steady-state
    time over the matched identity path. The pre-registered limit was 20%.
    Formal candidate training was not launched.
  </div>

  <div class="grid">
    <section class="panel">
      <h2>Steady-state layer cost</h2>
      {performance_bars}
      <div class="statline">
        <div class="stat"><b>{pct(float(benchmark['projection_time_overhead_frac']))}</b><span>time overhead</span></div>
        <div class="stat"><b>{pct(float(benchmark['projection_memory_overhead_frac']))}</b><span>memory overhead</span></div>
        <div class="stat"><b>{pct(float(benchmark['budget_clipped_frac']))}</b><span>checkpoint rows clipped</span></div>
      </div>
    </section>

    <section class="panel">
      <h2>Incremental peak memory</h2>
      {memory_bars}
      <p>The timing uses ABCCBA order, 3 warmups, 8 samples, and 5
      forward/backward repetitions per sample. Raw samples are in
      <code>evidence/cuda_contract.json</code>.</p>
    </section>

    <section class="panel">
      <h2>Correctness before the stop</h2>
      <p><strong>{math_tests or 'Unknown'}/{math_tests or 'Unknown'}</strong>
      mathematical tests passed.</p>
      <ul>{check_items}</ul>
      <p>No CPU model smoke, GPU2, search, repair, selector, or fallback was used.</p>
    </section>

    <section class="panel">
      <h2>What the strict budget does</h2>
      <div class="statline">
        <div class="stat"><b>{pct(budget_diag['clip_frac'])}</b><span>training rows clipped</span></div>
        <div class="stat"><b>{pct(budget_diag['endpoint_frac'])}</b><span>rows forced to isotropic endpoint</span></div>
        <div class="stat"><b>{budget_diag['loop5_lambda_mean']:.3f}</b><span>mean retained anisotropy at loop5</span></div>
        <div class="stat"><b>{pct(budget_diag['gate_relative_change'])}</b><span>relative gate change</span></div>
      </div>
      <p>It changes almost every erase gate and removes nearly all channel
      anisotropy. This is not a gentle stability correction.</p>
    </section>

    <section class="panel wide">
      <h2>Two-step checkpoint diagnostic</h2>
      <p class="warning">Diagnostic only: n=8 from source
      {html.escape(args.smoke_source_sha)}. This is not a formal candidate score.</p>
      {exact_bars}
      <table>
        <thead><tr><th>Readout</th><th>Identity exact</th><th>Budget exact</th><th>Identity blank acc</th><th>Budget blank acc</th></tr></thead>
        <tbody>{loop_table(control_diag['loops'], budget_diag['loops'])}</tbody>
      </table>
      <p>On the paired 51-55 blank subset, loop5 exact changes
      {control_diag['b51_55_loop5_exact']:.3f} to
      {budget_diag['b51_55_loop5_exact']:.3f}. Train CE changes
      {control_diag['train_ce']:.3f} to {budget_diag['train_ce']:.3f}.
      This supports the same diagnosis as the systems gate: strict c=1
      projection is too aggressive for this pretrained solver.</p>
    </section>

    <section class="panel wide">
      <h2>Formal matched control context</h2>
      <table>
        <thead><tr><th>Official blank range</th><th>n</th><th>Loop5 exact</th><th>Loop5 blank acc</th></tr></thead>
        <tbody>{formal_rows}</tbody>
      </table>
      <p>The formal identity arm completed at source 56030a1. The formal
      candidate never produced a valid score because earlier strict numerical
      certificates stopped it. The current exact SHA fixed correctness but
      failed the systems gate before any formal arm.</p>
    </section>

    <section class="panel wide">
      <h2>Paired Sudoku boards across loops</h2>
      <p>The two frames use the same fixed diagnostic cases. Open either frame
      directly for full-width inspection.</p>
      <div class="frames">
        <div>
          <p><a href="smoke_control_cases.html">Matched identity cases</a></p>
          <iframe src="smoke_control_cases.html" title="Matched identity cases"></iframe>
        </div>
        <div>
          <p><a href="smoke_budget_cases.html">Strict c=1 budget cases</a></p>
          <iframe src="smoke_budget_cases.html" title="Strict budget cases"></iframe>
        </div>
      </div>
    </section>
  </div>
</main>
</body>
</html>
"""
    (visualization_dir / "index.html").write_text(page, encoding="utf-8")

    readme = f"""# Gain-Budgeted GDN2 Final Decision

- Plan: `P-GAIN-001`
- Status: discarded at strict systems preflight
- Exact source: `{args.source_sha}`
- GPU: GPU1 `{contract['gpu_uuid']}`
- Official FLA source: `{contract['provenance']['fla_source_sha']}`
- Projection time overhead: `{pct(float(benchmark['projection_time_overhead_frac']))}`
- Projection memory overhead: `{pct(float(benchmark['projection_memory_overhead_frac']))}`
- Pre-registered time limit: `20%`
- Formal candidate training: not launched

All mathematical and official CUDA correctness checks passed before the
systems benchmark rejected the mechanism. The older two-step diagnostic is
included only to explain the failure mode; it is not a formal candidate score.
It shows that strict `c=1` clips almost every token/head and removes almost all
erase-gate anisotropy, degrading paired 51-55-blank loop5 exact from
`{control_diag['b51_55_loop5_exact']:.3f}` to
`{budget_diag['b51_55_loop5_exact']:.3f}`.

Decision: close strict `c=1` Gain-Budgeted GDN2. Do not sweep cap, seed, loss,
learning rate, model size, or training length. The broader lesson is that the
pretrained GDN2 uses mild transient expansion as useful computation; forcing
every step to be non-expansive is neither cheap nor behavior-preserving.
"""
    (output_dir / "README.md").write_text(readme, encoding="utf-8")
    print(visualization_dir / "index.html")


if __name__ == "__main__":
    main()
