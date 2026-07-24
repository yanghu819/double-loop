#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PUBLIC_TO_INTERNAL = {
    "rwkv": "rwkv7",
    "gdn": "fla_gdn",
    "gdn2": "gdn2",
    "kda": "kda",
}
EXPECTED_FLA_RUNTIME = {
    "gdn": {
        "implementation": "official_fla_fla_gdn",
        "class": "fla.layers.gated_deltanet.GatedDeltaNet",
    },
    "gdn2": {
        "implementation": "official_fla_gdn2",
        "class": "fla.layers.gdn2.GatedDeltaNet2",
    },
    "kda": {
        "implementation": "official_fla_kda",
        "class": "fla.layers.kda.KimiDeltaAttention",
    },
}
LABELS = {
    "rwkv": "RWKV7 TimeMix",
    "gdn": "GDN",
    "gdn2": "GDN2",
    "kda": "KDA",
}
COLORS = {
    "rwkv": "#7c3f92",
    "gdn": "#176b55",
    "gdn2": "#b04a3a",
    "kda": "#2869a6",
}
EXPECTED = {
    "size": 9,
    "batch": 32,
    "grad_accum_steps": 4,
    "d_model": 192,
    "layers": 10,
    "heads": 6,
    "head_dim": 32,
    "channel_mult": 4,
    "l_cycles": 2,
    "max_loops": 5,
    "loop_loss": "all",
    "lr": 0.0015,
    "weight_decay": 0.001,
    "optimizer_contract": "rwkv7_decay_groups",
    "shared_shell_init_seed": 52,
    "blank_loss_weight": 8.0,
    "future_seed_scale": 1.0,
    "future_seed_decay": 0.0,
    "future_seed_update": "fixed",
    "future_seed_norm_mode": "unit",
    "loop_update_mode": "fixed",
    "noise_scale": 0.0,
    "loop_feedback_scale": 0.0,
    "loop_time_scale": 0.0,
    "scratch_mode": "none",
    "hidden_agg_noise_scale": 0.0,
    "exact_margin_weight": 0.0,
    "gdn_mode": "chunk",
    "gdn_expand_v": 1.0,
    "gdn_use_short_conv": 1,
    "gdn_conv_size": 4,
    "forward_dtype": "bfloat16",
    "seed": 52,
    "eval_n": 512,
}
LOG_RE = re.compile(r"step=(?P<step>\d+)\s+ce=(?P<ce>[-+0-9.eE]+)")


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def result_path(run_dir: Path) -> Path:
    candidates = sorted((run_dir / "output").glob("futureseed_loop_seed*.json"))
    if len(candidates) != 1:
        raise ValueError(f"Expected one result JSON under {run_dir}, got {candidates}")
    return candidates[0]


def close_enough(left: Any, right: Any) -> bool:
    if isinstance(right, float):
        return abs(float(left) - right) <= 1e-9
    return left == right


def assert_close(label: str, left: Any, right: Any, tolerance: float = 1e-8) -> None:
    if abs(float(left) - float(right)) > tolerance:
        raise AssertionError(f"{label} differs: {left!r} != {right!r}")


def validate_args(public_name: str, args: dict[str, Any], steps: int) -> None:
    if args.get("backbone") != PUBLIC_TO_INTERNAL[public_name]:
        raise AssertionError(
            f"{public_name} resolved to {args.get('backbone')}, expected {PUBLIC_TO_INTERNAL[public_name]}"
        )
    if int(args.get("steps", -1)) != steps:
        raise AssertionError(f"{public_name} has {args.get('steps')} steps, expected {steps}")
    for key, expected in EXPECTED.items():
        actual = args.get(key)
        if not close_enough(actual, expected):
            raise AssertionError(f"{public_name} config mismatch for {key}: {actual!r} != {expected!r}")
    expected_stages = f"46-50:100,51-55:{steps - 100}"
    if args.get("hole_stages") != expected_stages:
        raise AssertionError(
            f"{public_name} curriculum mismatch: {args.get('hole_stages')!r} != {expected_stages!r}"
        )
    if args.get("official_eval_blank_ranges") != "46-50,51-55,56-64":
        raise AssertionError(f"{public_name} official blank ranges differ")
    if public_name == "rwkv":
        if args.get("rwkv_kernel") != "statepassing":
            raise AssertionError("RWKV benchmark must request the state-passing CUDA kernel")
    elif not bool(args.get("fla_strict_official")):
        raise AssertionError(f"{public_name} did not enable strict official FLA mode")


def train_curve(log_path: Path) -> list[dict[str, float]]:
    rows: dict[int, float] = {}
    with log_path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            match = LOG_RE.search(line)
            if match:
                rows[int(match.group("step"))] = float(match.group("ce"))
    return [{"step": step, "ce": rows[step]} for step in sorted(rows)]


def primary_case(run_dir: Path) -> dict[str, Any]:
    candidates = sorted((run_dir / "output").glob("futureseed_loop_case_seed*.html"))
    if len(candidates) != 1:
        raise ValueError(f"Expected one primary case HTML under {run_dir}, got {candidates}")
    case_path = candidates[0]
    text = case_path.read_text(encoding="utf-8")
    puzzle_match = re.search(
        r'<div class="panel"><h3>puzzle</h3>(.*?)<div class="panel"><h3>solution</h3>',
        text,
        flags=re.DOTALL,
    )
    if puzzle_match is None:
        raise ValueError(f"Could not parse puzzle from {case_path}")
    wrong_by_loop: dict[str, int] = {}
    for chunk in text.split('<div class="panel"><h3>loop ')[1:]:
        loop_match = re.match(r"(\d+)</h3>", chunk)
        if loop_match:
            wrong_by_loop[f"loop{int(loop_match.group(1))}"] = chunk.count('class="cell wrong"')
    return {
        "path": case_path,
        "puzzle_sha256": hashlib.sha256(puzzle_match.group(1).encode("utf-8")).hexdigest(),
        "wrong_by_loop": wrong_by_loop,
    }


def source_patch_files(run_dir: Path) -> list[str]:
    patch_path = run_dir / "source.patch"
    text = patch_path.read_text(encoding="utf-8", errors="replace")
    return sorted(set(re.findall(r"^diff --git a/(.+?) b/", text, flags=re.MULTILINE)))


def parameter_count(
    public_name: str,
    train: dict[str, Any],
    preflight: dict[str, Any],
) -> int:
    reported = train.get("parameter_count")
    checked = preflight["backbones"][public_name]["parameter_count"]
    if reported is not None and int(reported) != int(checked):
        raise AssertionError(
            f"{public_name} parameter count differs between run and preflight: {reported} != {checked}"
        )
    return int(checked)


def extract_run(
    runs_root: Path,
    public_name: str,
    run_name: str,
    steps: int,
    preflight: dict[str, Any],
) -> dict[str, Any]:
    run_dir = runs_root / run_name
    payload = read_json(result_path(run_dir))
    args = payload["args"]
    metrics = payload["metrics"]
    train = metrics["train"]
    validate_args(public_name, args, steps)

    if train.get("data_source") != "official_sudoku":
        raise AssertionError(f"{public_name} did not use official Sudoku data")
    if int(train.get("effective_batch", -1)) != 128:
        raise AssertionError(f"{public_name} effective batch is not 128")
    runtime = train.get("backbone_runtime", {})
    if runtime and bool(runtime.get("silent_fallback_allowed")):
        raise AssertionError(f"{public_name} allowed a silent fallback")
    optimizer_runtime = train.get("optimizer_runtime", {})
    if optimizer_runtime.get("contract") != "rwkv7_decay_groups":
        raise AssertionError(f"{public_name} optimizer contract differs: {optimizer_runtime}")
    if public_name == "rwkv":
        if runtime.get("implementation") != "official_rwkv7_timemix_with_explicit_state_io":
            raise AssertionError(f"RWKV implementation provenance differs: {runtime}")
        if runtime.get("official_source_commit") != "952102498e9ed367ea0a59ee64106916d474d30f":
            raise AssertionError(f"RWKV official source commit differs: {runtime}")
        if runtime.get("official_source_blob") != "b4d167fedead2655d253c55eb47b65f00e7193d2":
            raise AssertionError(f"RWKV official source blob differs: {runtime}")
        if runtime.get("official_kernel_blob") != "827faeb06b9d2b6e31b3efe85af6d3ae4cf88905":
            raise AssertionError(f"RWKV official kernel blob differs: {runtime}")
        if runtime.get("statepassing_cuda_sha256") != "59a90a0521b1851da17c008c685f959d586af1a7d28056b29a7478ab92c1c892":
            raise AssertionError(f"RWKV state-passing CUDA source differs: {runtime}")
    if public_name != "rwkv":
        fla_runtime = train.get("fla_runtime", {})
        if not bool(fla_runtime.get("strict")):
            raise AssertionError(f"{public_name} runtime is not strict official FLA")
        if fla_runtime.get("fla_source_sha") != "fe8fce9fc6984f22905f54cfa885dce1502baf26":
            raise AssertionError(f"{public_name} FLA source marker differs: {fla_runtime}")
        if not bool(fla_runtime.get("backend_dispatch_disabled")):
            raise AssertionError(f"{public_name} allowed FLA backend dispatch")
        if fla_runtime.get("conv_backend") != "triton":
            raise AssertionError(f"{public_name} convolution backend is not Triton")
        classes = {row["class"] for row in fla_runtime.get("layers", [])}
        expected_runtime = EXPECTED_FLA_RUNTIME[public_name]
        if classes != {expected_runtime["class"]}:
            raise AssertionError(
                f"{public_name} official FLA class differs: {classes} != "
                f"{{{expected_runtime['class']!r}}}"
            )
        if len(fla_runtime.get("layers", [])) != int(args["layers"]):
            raise AssertionError(f"{public_name} did not report every official FLA layer")
        for row in fla_runtime["layers"]:
            if set(row.get("conv_backends", {}).values()) != {"triton"}:
                raise AssertionError(f"{public_name} layer changed convolution backend: {row}")
        if runtime.get("implementation") != expected_runtime["implementation"]:
            raise AssertionError(f"{public_name} implementation marker differs: {runtime}")

    checkpoint_path = run_dir / "output" / f"checkpoint_eval_step{steps:06d}.json"
    checkpoint = read_json(checkpoint_path)
    fixed_holes53 = checkpoint["eval_by_holes"]["holes53"]
    if fixed_holes53.get("blank_range") != [53, 53]:
        raise AssertionError(
            f"{public_name} checkpoint holes53 is not a strict 53-blank evaluation: {fixed_holes53}"
        )
    if int(fixed_holes53.get("eval_n", -1)) != 512:
        raise AssertionError(
            f"{public_name} checkpoint holes53 eval_n differs: {fixed_holes53.get('eval_n')}"
        )
    config = read_json(run_dir / "config.json")
    if bool(config.get("git_dirty")):
        raise AssertionError(f"{public_name} formal run reports a dirty source tree")
    patch_files = source_patch_files(run_dir)
    if patch_files:
        raise AssertionError(f"{public_name} formal run has a non-empty source patch: {patch_files}")
    loops = []
    for loop in range(1, int(args["max_loops"]) + 1):
        row = metrics["eval_clean"][f"loop{loop}"]
        loops.append(
            {
                "loop": loop,
                "exact": float(row["label_exact"]),
                "blank_acc": float(row["blank_acc"]),
            }
        )
    ranges: dict[str, Any] = {}
    for key in ("b46_50", "b51_55", "b56_64"):
        value = metrics["official_eval_by_blank_range"][key]
        ranges[key] = {
            "blank_range": value["blank_range"],
            "eval_n": int(value["eval_n"]),
            "loop1": value["eval_clean"]["loop1"],
            "loop5": value["eval_clean"]["loop5"],
        }
    case = primary_case(run_dir)
    elapsed = float(checkpoint["elapsed_sec"])
    arm = {
        "key": public_name,
        "label": LABELS[public_name],
        "color": COLORS[public_name],
        "run_name": run_name,
        "run_dir": f"runs/{run_name}",
        "git_sha": config["git_sha"],
        "git_dirty": bool(config.get("git_dirty")),
        "source_patch_files": patch_files,
        "parameter_count": parameter_count(public_name, train, preflight),
        "train_ce": float(checkpoint["train"]["ce_loss"]),
        "train_loop1_ce": float(checkpoint["train"]["loop1_loss"]),
        "train_loop5_ce": float(checkpoint["train"]["loop_last_loss"]),
        "train_wall_sec": elapsed,
        "train_sec_per_step": elapsed / steps,
        "peak_allocated_mib": float(train["cuda_max_memory_allocated_mb"]),
        "memory_measurement": train.get("cuda_memory_measurement"),
        "fixed_holes53": fixed_holes53["eval_clean"]["loop5"],
        "loops": loops,
        "loop_gain_exact": loops[-1]["exact"] - loops[0]["exact"],
        "loop_gain_blank": loops[-1]["blank_acc"] - loops[0]["blank_acc"],
        "ranges": ranges,
        "curve": train_curve(run_dir / "logs" / "run.log"),
        "case": {
            "relative_html": f"../{run_name}/output/{case['path'].name}",
            "puzzle_sha256": case["puzzle_sha256"],
            "wrong_by_loop": case["wrong_by_loop"],
        },
        "eval_identity": {
            "eval_official": metrics["eval_official"],
            "official_eval_seed_offset": int(args["official_eval_seed_offset"]),
            "train_size": int(train["official_train_size"]),
            "eval_size": int(train["official_eval_size"]),
        },
        "runtime": runtime,
        "fla_runtime": train.get("fla_runtime"),
    }
    strict_path = run_dir / "strict_run_validation.json"
    if not strict_path.is_file():
        raise FileNotFoundError(f"{public_name} is missing strict post-run validation: {strict_path}")
    strict = read_json(strict_path)
    expected_strict = {
        "status": "PASS",
        "backbone": public_name,
        "run_name": run_name,
        "git_sha": arm["git_sha"],
        "puzzle_sha256": arm["case"]["puzzle_sha256"],
        "parameter_count": arm["parameter_count"],
    }
    for key, expected in expected_strict.items():
        if strict.get(key) != expected:
            raise AssertionError(
                f"{public_name} strict validation mismatch for {key}: "
                f"{strict.get(key)!r} != {expected!r}"
            )
    strict_metrics = {
        "train_ce": arm["train_ce"],
        "train_sec_per_step": arm["train_sec_per_step"],
        "peak_allocated_mib": arm["peak_allocated_mib"],
        "fixed_holes53_exact": arm["fixed_holes53"]["label_exact"],
        "mixed_loop1_exact": arm["loops"][0]["exact"],
        "mixed_loop5_exact": arm["loops"][-1]["exact"],
        "mixed_loop5_blank_acc": arm["loops"][-1]["blank_acc"],
        "official_46_50_exact": arm["ranges"]["b46_50"]["loop5"]["label_exact"],
        "official_51_55_exact": arm["ranges"]["b51_55"]["loop5"]["label_exact"],
        "official_56_64_exact": arm["ranges"]["b56_64"]["loop5"]["label_exact"],
    }
    for key, expected in strict_metrics.items():
        assert_close(f"{public_name} strict validation {key}", strict.get(key), expected)
    arm["strict_validation"] = strict
    return arm


def validate_cross_arm(arms: list[dict[str, Any]], allow_mixed_source: bool) -> None:
    reference = arms[0]
    for arm in arms[1:]:
        if arm["eval_identity"] != reference["eval_identity"]:
            raise AssertionError(
                f"Evaluation/data identity differs: {reference['key']} vs {arm['key']}"
            )
        if arm["case"]["puzzle_sha256"] != reference["case"]["puzzle_sha256"]:
            raise AssertionError(
                f"Primary visualization puzzle differs: {reference['key']} vs {arm['key']}"
            )
    shas = {arm["git_sha"] for arm in arms}
    if len(shas) != 1 and not allow_mixed_source:
        raise AssertionError(f"Source SHAs differ across arms: {sorted(shas)}")


def fmt_pct(value: float) -> str:
    return f"{100.0 * value:.2f}%"


def line_chart(
    arms: list[dict[str, Any]],
    key: str,
    x_key: str,
    y_key: str,
    title: str,
) -> str:
    rows = [row for arm in arms for row in arm[key]]
    xs = sorted({float(row[x_key]) for row in rows})
    values = [float(row[y_key]) for row in rows]
    y_min, y_max = min(values), max(values)
    if y_min == y_max:
        y_max = y_min + 1.0
    width, height = 720, 300
    left, right, top, bottom = 62, 18, 42, 46
    plot_w, plot_h = width - left - right, height - top - bottom

    def xpos(value: float) -> float:
        return left + (value - xs[0]) / max(xs[-1] - xs[0], 1.0) * plot_w

    def ypos(value: float) -> float:
        return top + (y_max - value) / (y_max - y_min) * plot_h

    parts = [
        f'<svg viewBox="0 0 {width} {height}" role="img" aria-label="{html.escape(title)}">',
        f'<text x="{left}" y="24" class="chart-title">{html.escape(title)}</text>',
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
            f'{xpos(float(row[x_key])):.2f},{ypos(float(row[y_key])):.2f}' for row in arm[key]
        )
        parts.append(f'<polyline points="{points}" fill="none" stroke="{arm["color"]}" stroke-width="3"/>')
        for row in arm[key]:
            parts.append(
                f'<circle cx="{xpos(float(row[x_key])):.2f}" cy="{ypos(float(row[y_key])):.2f}" '
                f'r="4" fill="{arm["color"]}"/>'
            )
        legend_x = left + index * 140
        parts.append(f'<line x1="{legend_x}" y1="{height - 8}" x2="{legend_x + 20}" y2="{height - 8}" stroke="{arm["color"]}" stroke-width="3"/>')
        parts.append(f'<text x="{legend_x + 26}" y="{height - 4}" class="tick">{arm["label"]}</text>')
    parts.append("</svg>")
    return "".join(parts)


def render_html(payload: dict[str, Any]) -> str:
    arms = payload["arms"]
    quality_rows = "".join(
        "<tr>"
        f"<td><span class=\"swatch\" style=\"background:{arm['color']}\"></span>{arm['label']}</td>"
        f"<td>{arm['parameter_count'] / 1e6:.3f}M</td>"
        f"<td>{arm['train_ce']:.4f}</td>"
        f"<td>{fmt_pct(arm['fixed_holes53']['label_exact'])}</td>"
        f"<td>{fmt_pct(arm['loops'][-1]['exact'])}</td>"
        f"<td>{fmt_pct(arm['loops'][-1]['blank_acc'])}</td>"
        f"<td>{fmt_pct(arm['loop_gain_exact'])}</td>"
        f"<td>{arm['train_sec_per_step']:.2f}</td>"
        f"<td>{arm['peak_allocated_mib'] / 1024:.2f}</td>"
        "</tr>"
        for arm in arms
    )
    range_rows = []
    for key, label in (("b46_50", "46-50"), ("b51_55", "51-55"), ("b56_64", "56-64")):
        cells = "".join(
            f"<td>{fmt_pct(arm['ranges'][key]['loop5']['label_exact'])}</td>"
            f"<td>{fmt_pct(arm['ranges'][key]['loop5']['blank_acc'])}</td>"
            for arm in arms
        )
        range_rows.append(f"<tr><td>{label}</td>{cells}</tr>")
    range_headers = "".join(f"<th colspan=\"2\">{arm['label']}</th>" for arm in arms)
    range_subheaders = "".join("<th>exact</th><th>blank</th>" for _ in arms)
    cases = "".join(
        f'<section><h3>{arm["label"]}</h3><p class="muted">Wrong cells: '
        f'{html.escape(str(arm["case"]["wrong_by_loop"]))}</p>'
        f'<iframe src="{html.escape(arm["case"]["relative_html"])}" loading="lazy"></iframe></section>'
        for arm in arms
    )
    provenance_rows = "".join(
        "<tr>"
        f"<td>{arm['label']}</td>"
        f"<td>{html.escape(str(arm['runtime'].get('implementation', 'official FLA strict runtime')))}</td>"
        f"<td>{arm['git_sha'][:12]}</td>"
        f"<td>{'clean' if not arm['source_patch_files'] else 'tracking only'}</td>"
        f"<td>{arm['strict_validation']['status']}</td>"
        f"<td>{html.escape(str(payload['preflight']['backbones'][arm['key']]['runtime'].get('kernel', 'chunk + Triton conv')))}</td>"
        "</tr>"
        for arm in arms
    )
    memory_notes = "".join(
        f"<li>{arm['label']}: peak allocation recovered from "
        f"<code>{html.escape(str(arm['memory_measurement']['source_run']))}</code> using "
        f"{int(arm['memory_measurement']['probe_optimizer_steps'])} matched optimizer step; "
        "quality and speed remain from the original step-500 run.</li>"
        for arm in arms
        if arm["memory_measurement"]
    )
    memory_note_html = (
        f"<p class=\"muted\">Lease-recovery instrumentation:</p><ul>{memory_notes}</ul>"
        if memory_notes
        else ""
    )
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>FutureSeed Sudoku backbone benchmark</title>
<style>
:root{{--bg:#f2f4f5;--paper:#fff;--ink:#172026;--muted:#5a666e;--line:#d7dde1}}
*{{box-sizing:border-box}}body{{margin:0;background:var(--bg);color:var(--ink);font:15px/1.5 ui-sans-serif,system-ui,sans-serif}}
main{{max-width:1540px;margin:auto;padding:26px 20px 60px}}h1{{font-size:28px;margin:0 0 5px;letter-spacing:0}}
h2{{font-size:20px;margin:28px 0 10px;letter-spacing:0}}h3{{font-size:16px;letter-spacing:0}}
.band{{background:var(--paper);border:1px solid var(--line);border-radius:6px;padding:16px;margin-top:14px;overflow:auto}}
.muted{{color:var(--muted)}}table{{width:100%;border-collapse:collapse;white-space:nowrap}}
th,td{{padding:9px;border-bottom:1px solid var(--line);text-align:right}}th:first-child,td:first-child{{text-align:left}}
.swatch{{display:inline-block;width:10px;height:10px;margin-right:7px}}.charts{{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:12px}}
svg{{width:100%;background:#fff;border:1px solid var(--line);border-radius:6px}}.grid{{stroke:#e5eaed}}.tick{{font-size:12px;fill:#4f5a62}}.chart-title{{font-size:16px;font-weight:650}}
.cases{{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:12px}}iframe{{width:100%;height:720px;border:1px solid var(--line);border-radius:6px;background:#fff}}
code{{background:#edf1f3;padding:2px 5px;border-radius:3px}}@media(max-width:1100px){{.charts,.cases{{grid-template-columns:1fr}}}}
</style></head><body><main>
<h1>FutureSeed: RWKV7 vs GDN vs GDN2 vs KDA</h1>
<p class="muted">One seed, one data order, one training budget, one evaluator. GPU1 only; no CPU model smoke and no silent fallback.</p>
<div class="band"><strong>Decision:</strong> {html.escape(payload["decision"])}</div>
<h2>Fairness contract</h2><div class="band"><p>D192/L10/H6/D32, matched 6x32x32 recurrent state, five loops, CE on every loop, effective batch 128, BF16, seed 52, backbone-independent shared-shell initialization, native FutureSeed scale 1, official full-diversity Sudoku, and curriculum <code>{html.escape(payload["contract"]["hole_stages"])}</code>. Parameters and runtime are measured rather than padded.</p></div>
<h2>Matched result</h2><div class="band"><table><thead><tr><th>backbone</th><th>params</th><th>train CE</th><th>fixed h53 exact</th><th>mixed exact</th><th>mixed blank</th><th>loop exact gain</th><th>sec/step</th><th>peak GiB</th></tr></thead><tbody>{quality_rows}</tbody></table></div>
<h2>Learning and loops</h2><div class="charts">{line_chart(arms, "curve", "step", "ce", "Training CE")}{line_chart(arms, "loops", "loop", "exact", "Full-board exact by loop")}{line_chart(arms, "loops", "loop", "blank_acc", "Blank accuracy by loop")}</div>
<h2>Official blank ranges</h2><div class="band"><table><thead><tr><th rowspan="2">blanks</th>{range_headers}</tr><tr>{range_subheaders}</tr></thead><tbody>{''.join(range_rows)}</tbody></table></div>
<h2>Same puzzle, every loop</h2><div class="cases">{cases}</div>
<h2>Implementation provenance</h2><div class="band"><table><thead><tr><th>backbone</th><th>implementation</th><th>source SHA</th><th>source patch</th><th>strict post-run</th><th>kernel</th></tr></thead><tbody>{provenance_rows}</tbody></table><p>Official RWKV7 model commit: <code>{html.escape(payload["rwkv7_gate"]["source_and_initialization"]["source_commit"])}</code>; model blob: <code>{html.escape(payload["rwkv7_gate"]["source_and_initialization"]["source_blob"])}</code>; kernel blob: <code>{html.escape(payload["rwkv7_gate"]["source_and_initialization"]["kernel_blob"])}</code>. Official FLA wheel SHA256: <code>{html.escape(payload["fla_gate"]["provenance"]["wheel_sha256"])}</code>. Formula, CUDA reference/backward, source, native cache layout, exact official class, and no-fallback gates passed before training. Every displayed metric also matches a separate fail-closed post-run validator.</p>{memory_note_html}</div>
<h2>Interpretation boundary</h2><div class="band"><p>This is a one-seed, finite-budget, matched-state and matched-recipe comparison of FutureSeed-enabled recurrent carriers. It is not a with/without-FutureSeed ablation, an architecture-specific hyperparameter study, or evidence about asymptotic ceilings. In particular, state matching uses <code>H6 x K32 x V32</code> for every carrier; official GDN commonly recommends <code>H x K = 0.75D</code> with <code>expand_v=2</code>, so this run deliberately does not use GDN's native recommended geometry. The separate matched no-FutureSeed result is needed for a causal FutureSeed claim.</p></div>
</main></body></html>"""


def decision_from(arms: list[dict[str, Any]]) -> str:
    hard_scores = {
        arm["key"]: (
            arm["ranges"]["b51_55"]["loop5"]["label_exact"]
            + arm["ranges"]["b56_64"]["loop5"]["label_exact"]
        )
        / 2
        for arm in arms
    }
    best_key = max(hard_scores, key=hard_scores.get)
    ordered = sorted(hard_scores.values(), reverse=True)
    if ordered[0] - ordered[1] >= 0.03:
        return (
            f"{LABELS[best_key]} has a real hard-exact advantage at this budget. "
            "Retain it as the scale carrier and verify the advantage at one longer gate."
        )
    best_opening = max(
        arms,
        key=lambda arm: arm["ranges"]["b46_50"]["loop5"]["label_exact"],
    )
    cheapest = min(arms, key=lambda arm: arm["train_sec_per_step"])
    return (
        "No backbone separates on hard full-board closure at this finite budget. "
        f"{best_opening['label']} has the best 46-50 blank opening exact "
        f"({best_opening['ranges']['b46_50']['loop5']['label_exact']:.4f}); "
        f"{cheapest['label']} is cheapest per optimizer step. "
        "Do not select or claim a universal architecture winner from this gate."
    )


def write_csv(path: Path, arms: list[dict[str, Any]]) -> None:
    fields = [
        "backbone",
        "git_sha",
        "parameters",
        "train_ce",
        "fixed_holes53_exact",
        "mixed_loop5_exact",
        "mixed_loop5_blank_acc",
        "loop1_to_loop5_exact_gain",
        "sec_per_step",
        "peak_allocated_mib",
        "peak_memory_source",
        "official_46_50_exact",
        "official_51_55_exact",
        "official_56_64_exact",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for arm in arms:
            writer.writerow(
                {
                    "backbone": arm["key"],
                    "git_sha": arm["git_sha"],
                    "parameters": arm["parameter_count"],
                    "train_ce": arm["train_ce"],
                    "fixed_holes53_exact": arm["fixed_holes53"]["label_exact"],
                    "mixed_loop5_exact": arm["loops"][-1]["exact"],
                    "mixed_loop5_blank_acc": arm["loops"][-1]["blank_acc"],
                    "loop1_to_loop5_exact_gain": arm["loop_gain_exact"],
                    "sec_per_step": arm["train_sec_per_step"],
                    "peak_allocated_mib": arm["peak_allocated_mib"],
                    "peak_memory_source": (
                        arm["memory_measurement"]["source_run"]
                        if arm["memory_measurement"]
                        else "uninterrupted_run"
                    ),
                    "official_46_50_exact": arm["ranges"]["b46_50"]["loop5"]["label_exact"],
                    "official_51_55_exact": arm["ranges"]["b51_55"]["loop5"]["label_exact"],
                    "official_56_64_exact": arm["ranges"]["b56_64"]["loop5"]["label_exact"],
                }
            )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument(
        "--runs-root",
        type=Path,
        default=None,
        help="Run artifact root; defaults to <repo>/runs.",
    )
    parser.add_argument("--rwkv-run", required=True)
    parser.add_argument("--gdn-run", required=True)
    parser.add_argument("--gdn2-run", required=True)
    parser.add_argument("--kda-run", required=True)
    parser.add_argument("--preflight", type=Path, required=True)
    parser.add_argument("--shared-shell-gate", type=Path, required=True)
    parser.add_argument("--fla-gate", type=Path, required=True)
    parser.add_argument("--rwkv7-gate", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--plan-id", default="P-BASELINE-003")
    parser.add_argument("--allow-mixed-source", action="store_true")
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.steps <= 100:
        raise ValueError("--steps must be greater than the fixed 100-step opening stage")

    repo = args.repo.resolve()
    runs_root = args.runs_root or repo / "runs"
    if not runs_root.is_absolute():
        runs_root = repo / runs_root
    runs_root = runs_root.resolve()
    preflight_path = args.preflight if args.preflight.is_absolute() else repo / args.preflight
    shared_shell_gate_path = (
        args.shared_shell_gate if args.shared_shell_gate.is_absolute() else repo / args.shared_shell_gate
    )
    fla_gate_path = args.fla_gate if args.fla_gate.is_absolute() else repo / args.fla_gate
    rwkv7_gate_path = args.rwkv7_gate if args.rwkv7_gate.is_absolute() else repo / args.rwkv7_gate
    preflight = read_json(preflight_path)
    shared_shell_gate = read_json(shared_shell_gate_path)
    fla_gate = read_json(fla_gate_path)
    rwkv7_gate = read_json(rwkv7_gate_path)
    if preflight.get("cuda_visible_devices") != "0":
        raise AssertionError("Preflight was not bound to GPU1")
    if shared_shell_gate.get("status") != "PASS":
        raise AssertionError("Shared-shell initialization gate did not pass")
    if int(shared_shell_gate.get("differing_parameter_tensors", -1)) != 0:
        raise AssertionError("Shared-shell initialization still differs across backbones")
    if not all(
        row["installed_sha256"] == row["wheel_sha256"]
        for row in fla_gate["provenance"]["source_file_hashes"].values()
    ):
        raise AssertionError("Installed FLA source differs from the pinned wheel")
    if rwkv7_gate.get("source_and_initialization", {}).get("source_commit") != "952102498e9ed367ea0a59ee64106916d474d30f":
        raise AssertionError("RWKV7 official source commit differs from the pinned contract")
    if rwkv7_gate.get("source_and_initialization", {}).get("source_blob") != "b4d167fedead2655d253c55eb47b65f00e7193d2":
        raise AssertionError("RWKV7 official source blob differs from the pinned contract")
    if rwkv7_gate.get("source_and_initialization", {}).get("kernel_blob") != "827faeb06b9d2b6e31b3efe85af6d3ae4cf88905":
        raise AssertionError("RWKV7 official kernel blob differs from the pinned contract")
    if rwkv7_gate.get("source_and_initialization", {}).get("statepassing_cuda_sha256") != "59a90a0521b1851da17c008c685f959d586af1a7d28056b29a7478ab92c1c892":
        raise AssertionError("RWKV7 vendored state-passing CUDA source differs from the pinned contract")

    run_names = {
        "rwkv": args.rwkv_run,
        "gdn": args.gdn_run,
        "gdn2": args.gdn2_run,
        "kda": args.kda_run,
    }
    arms = [
        extract_run(runs_root, public_name, run_names[public_name], args.steps, preflight)
        for public_name in ("rwkv", "gdn", "gdn2", "kda")
    ]
    validate_cross_arm(arms, args.allow_mixed_source)
    payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "plan_id": args.plan_id,
        "contract": {
            **EXPECTED,
            "steps": args.steps,
            "hole_stages": f"46-50:100,51-55:{args.steps - 100}",
            "effective_batch": 128,
            "public_backbones": list(PUBLIC_TO_INTERNAL),
        },
        "decision": decision_from(arms),
        "arms": arms,
        "preflight": preflight,
        "shared_shell_gate": shared_shell_gate,
        "fla_gate": fla_gate,
        "rwkv7_gate": rwkv7_gate,
    }
    out_dir = args.out_dir if args.out_dir.is_absolute() else repo / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    serializable = json.loads(json.dumps(payload, default=str))
    (out_dir / "comparison.json").write_text(
        json.dumps(serializable, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    write_csv(out_dir / "benchmark.csv", arms)
    (out_dir / "index.html").write_text(render_html(payload), encoding="utf-8")
    summary_rows = "\n".join(
        f"| {arm['label']} | {arm['parameter_count'] / 1e6:.3f}M | {arm['train_ce']:.4f} | "
        f"{arm['loops'][-1]['exact']:.5f} | {arm['loops'][-1]['blank_acc']:.5f} | "
        f"{arm['train_sec_per_step']:.2f}s | {arm['peak_allocated_mib'] / 1024:.2f}GiB |"
        for arm in arms
    )
    recovered_memory = [
        f"{arm['label']} peak memory was recovered from "
        f"`{arm['memory_measurement']['source_run']}` with one matched optimizer step; "
        "its quality and speed remain from the original step-500 run."
        for arm in arms
        if arm["memory_measurement"]
    ]
    recovered_memory_note = (
        "\n" + "\n".join(recovered_memory) + "\n" if recovered_memory else ""
    )
    (out_dir / "README.md").write_text(
        "# FutureSeed Sudoku Backbone Benchmark\n\n"
        f"Decision: {payload['decision']}\n\n"
        "| Backbone | Params | Train CE | Mixed exact | Mixed blank | Sec/step | Peak |\n"
        "|---|---:|---:|---:|---:|---:|---:|\n"
        f"{summary_rows}\n\n"
        f"{recovered_memory_note}"
        "Open `index.html` for official blank ranges, loop curves, kernel provenance, and same-puzzle visualizations.\n",
        encoding="utf-8",
    )
    print(out_dir)


if __name__ == "__main__":
    main()
