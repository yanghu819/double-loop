#!/usr/bin/env python3
from __future__ import annotations

import argparse
import html
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from build_fla_crossover_comparison import (
    EXPECTED_AUTOGRAD_NODES,
    extract_arm,
    line_chart,
    read_json,
)


def metric_table(arms: list[dict[str, Any]]) -> str:
    rows = []
    for arm in arms:
        final = arm["loops"][-1]
        rows.append(
            "<tr>"
            f"<td>{arm['label']}</td>"
            f"<td>{arm['train_ce']:.4f}</td>"
            f"<td>{arm['checkpoint_holes53']['label_exact']:.4f}</td>"
            f"<td>{final['exact']:.4f}</td>"
            f"<td>{final['blank_acc']:.4f}</td>"
            f"<td>{arm['train_sec_per_step']:.2f}</td>"
            f"<td>{arm['peak_allocated_mb']:.0f}</td>"
            "</tr>"
        )
    return "".join(rows)


def range_table(arms: list[dict[str, Any]]) -> str:
    rows = []
    for key in ("b46_50", "b51_55", "b56_64"):
        exact = "".join(
            f"<td>{arm['ranges'][key]['loop5']['label_exact']:.4f}</td>" for arm in arms
        )
        blank = "".join(
            f"<td>{arm['ranges'][key]['loop5']['blank_acc']:.4f}</td>" for arm in arms
        )
        rows.append(f"<tr><td>{key[1:].replace('_', '-')}</td>{exact}{blank}</tr>")
    return "".join(rows)


def render(payload: dict[str, Any]) -> str:
    arms = payload["arms"]
    strict = payload["strict_gate"]
    arm_headers = "".join(f"<th>{arm['label']}</th>" for arm in arms)
    frames = "".join(
        f'<section><h3>{arm["label"]}: same-seed hard case</h3>'
        f'<p class="muted">Wrong cells by loop: {html.escape(str(arm["case"]["wrong_by_loop"]))}</p>'
        f'<iframe src="{html.escape(arm["case"]["relative_html"])}" loading="lazy"></iframe></section>'
        for arm in arms
    )
    kernel_rows = "".join(
        f'<tr><td>{html.escape(name)}</td><td>{html.escape(row["official_layer_class"])}</td>'
        f'<td>{html.escape(row["official_chunk_autograd_node"])}</td>'
        f'<td>{html.escape(row["conv_backend"])}</td></tr>'
        for name, row in strict["adapters"].items()
    )
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Official FLA FutureSeed three-way step1000 check</title><style>
:root{{--bg:#f4f6f7;--paper:#fff;--ink:#172026;--muted:#59666f;--line:#d6dde1}}*{{box-sizing:border-box}}body{{margin:0;background:var(--bg);color:var(--ink);font:15px/1.5 ui-sans-serif,system-ui,sans-serif}}main{{max-width:1480px;margin:auto;padding:28px 20px 60px}}h1{{font-size:28px;margin:0 0 6px;letter-spacing:0}}h2{{font-size:20px;margin:28px 0 10px;letter-spacing:0}}h3{{font-size:16px;letter-spacing:0}}.muted{{color:var(--muted)}}.band{{background:var(--paper);border:1px solid var(--line);border-radius:6px;padding:17px;margin-top:14px;overflow:auto}}.charts{{display:grid;grid-template-columns:1fr 1fr;gap:14px}}.cases{{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:14px}}svg{{width:100%;background:#fff;border:1px solid var(--line);border-radius:6px}}.axis{{stroke:#54616a}}.grid{{stroke:#e6eaed}}.tick{{font-size:12px;fill:#4d5961}}.chart-title{{font-size:16px;font-weight:650}}table{{width:100%;border-collapse:collapse;white-space:nowrap}}th,td{{padding:9px;border-bottom:1px solid var(--line);text-align:right}}th:first-child,td:first-child{{text-align:left}}iframe{{width:100%;height:760px;border:1px solid var(--line);border-radius:6px;background:#fff}}code{{background:#edf1f3;padding:2px 5px;border-radius:3px}}@media(max-width:1000px){{.charts,.cases{{grid-template-columns:1fr}}}}
</style></head><body><main><h1>GDN, KDA, and GDN2: does the step500 ranking survive?</h1><p class="muted">P-LA-005. Exact optimizer/RNG resumes to step1000, one shared D192/L10/H6/D32 recipe, official FLA CUDA kernels, GPU1 only.</p>
<div class="band"><strong>Decision:</strong> {html.escape(payload['decision'])}</div>
<h2>Matched result</h2><div class="band"><table><thead><tr><th>backbone</th><th>step1000 CE</th><th>fixed holes53 exact</th><th>mixed loop5 exact</th><th>mixed loop5 blank</th><th>sec/step</th><th>peak MiB</th></tr></thead><tbody>{metric_table(arms)}</tbody></table></div>
<h2>Learning and recurrent computation</h2><div class="charts">{line_chart(arms, 'curve', 'Training CE through step1000', 'step', 'ce')}{line_chart(arms, 'loops', 'Full-board exact by loop at step1000', 'loop', 'exact')}{line_chart(arms, 'loops', 'Blank accuracy by loop at step1000', 'loop', 'blank_acc')}</div>
<h2>Official blank ranges</h2><div class="band"><table><thead><tr><th rowspan="2">blanks</th><th colspan="3">loop5 exact</th><th colspan="3">loop5 blank accuracy</th></tr><tr>{arm_headers}{arm_headers}</tr></thead><tbody>{range_table(arms)}</tbody></table></div>
<h2>Same-seed hard-case behavior</h2><div class="cases">{frames}</div>
<h2>Kernel provenance</h2><div class="band"><p>Wheel <code>{strict['wheel_sha256']}</code>; installed-source marker <code>{strict['fla_source_sha']}</code>; audited installed files equal the pinned wheel: <strong>{str(strict['source_hashes_match']).lower()}</strong>.</p><table><thead><tr><th>adapter</th><th>official layer</th><th>CUDA backward node</th><th>short conv</th></tr></thead><tbody>{kernel_rows}</tbody></table></div>
<h2>Interpretation boundary</h2><div class="band"><p>All three arms contain the same FutureSeed mechanism. This experiment ranks FutureSeed-enabled carrier equations under one deliberately shared small-state recipe. It does not estimate the causal FutureSeed gain and does not test each architecture's native geometry or asymptotic ceiling.</p></div>
</main></body></html>"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument("--gdn-run", required=True)
    parser.add_argument("--kda-run", required=True)
    parser.add_argument("--gdn2-run", required=True)
    parser.add_argument("--gdn-baseline", required=True)
    parser.add_argument("--kda-baseline", required=True)
    parser.add_argument("--gdn2-baseline", required=True)
    parser.add_argument("--gdn2-resume-step", type=int, default=600)
    parser.add_argument("--gdn2-extra-log", type=Path, action="append", default=[])
    parser.add_argument("--strict-gate", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    repo = args.repo.resolve()
    out_dir = args.out_dir if args.out_dir.is_absolute() else repo / args.out_dir
    arms = [
        extract_arm(repo, args.gdn_run, args.gdn_baseline, "fla_gdn", 500),
        extract_arm(repo, args.kda_run, args.kda_baseline, "kda", 500),
        extract_arm(
            repo,
            args.gdn2_run,
            args.gdn2_baseline,
            "gdn2",
            args.gdn2_resume_step,
            [path if path.is_absolute() else repo / path for path in args.gdn2_extra_log],
        ),
    ]
    for arm in arms[1:]:
        if arms[0]["args"] != arm["args"]:
            raise AssertionError(f"critical configs differ: {arms[0]['args']} != {arm['args']}")
        if arms[0]["hole_stages"] != arm["hole_stages"]:
            raise AssertionError("executed continuation curricula differ")
        if arms[0]["official_eval"] != arm["official_eval"]:
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
        if set(row["conv_backends"].values()) != {"triton"}:
            raise AssertionError(f"{name} did not use Triton short convolution")
        strict_adapters[name] = {
            "official_layer_class": row["official_layer_class"],
            "official_chunk_autograd_node": row["official_chunk_autograd_node"],
            "conv_backend": "triton",
            "initial_state_grad_norm": row["initial_state_grad_norm"],
        }

    gdn, kda, gdn2 = arms
    kda_ce_gap = kda["train_ce"] - gdn["train_ce"]
    kda_opening_gap = (
        gdn["ranges"]["b46_50"]["loop5"]["label_exact"]
        - kda["ranges"]["b46_50"]["loop5"]["label_exact"]
    )
    gdn2_ce_gap = gdn2["train_ce"] - gdn["train_ce"]
    gdn2_opening_gap = (
        gdn["ranges"]["b46_50"]["loop5"]["label_exact"]
        - gdn2["ranges"]["b46_50"]["loop5"]["label_exact"]
    )
    kda_caught = kda_ce_gap < 0.02 and kda_opening_gap < 0.05
    gdn2_caught = gdn2_ce_gap < 0.02 and gdn2_opening_gap < 0.05
    if kda_caught and gdn2_caught:
        decision = "The step500 ranking was mostly a finite-budget artifact; no alternative wins hard closure or efficiency at step1000."
    elif gdn2_caught:
        decision = "GDN2 catches GDN by step1000, but KDA remains slower to optimize under the shared D32 recipe."
    elif kda_caught:
        decision = "KDA catches GDN by step1000, while GDN2 retains a measurable finite-budget gap under the shared D32 recipe."
    else:
        decision = "GDN remains the most sample-efficient carrier under this shared D32 recipe; this is not an architecture-ceiling claim."

    payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "plan_id": "P-LA-005",
        "arms": arms,
        "gaps_vs_gdn": {
            "kda": {"ce": kda_ce_gap, "opening_exact": kda_opening_gap},
            "gdn2": {"ce": gdn2_ce_gap, "opening_exact": gdn2_opening_gap},
        },
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
    summary = "\n".join(
        f"- {arm['label']}: CE {arm['train_ce']:.4f}, fixed holes53 exact {arm['checkpoint_holes53']['label_exact']:.5f}, mixed loop5 exact {arm['loops'][-1]['exact']:.5f}, {arm['train_sec_per_step']:.2f} sec/step"
        for arm in arms
    )
    (out_dir / "README.md").write_text(
        f"# Official FLA FutureSeed Three-Way Step1000 Check\n\n## Decision\n\n{decision}\n\n## Matched results\n\n{summary}\n\nOpen `index.html` for curves, official blank ranges, kernel provenance, and same-seed loop visualizations.\n",
        encoding="utf-8",
    )
    print(out_dir)


if __name__ == "__main__":
    main()
