#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import html
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


FLA_SHA = "31d15f7554bd5df05d3da6f75e09146279d2b1a8"


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def result_path(run_dir: Path) -> Path:
    paths = sorted((run_dir / "output").glob("futureseed_loop_seed*.json"))
    if len(paths) != 1:
        raise RuntimeError(f"Expected one result JSON in {run_dir}, got {paths}")
    return paths[0]


def primary_case(run_dir: Path) -> tuple[Path, str, dict[str, int]]:
    paths = sorted((run_dir / "output").glob("futureseed_loop_case_seed*.html"))
    if len(paths) != 1:
        raise RuntimeError(f"Expected one primary case HTML in {run_dir}, got {paths}")
    text = paths[0].read_text(encoding="utf-8")
    match = re.search(
        r'<div class="panel"><h3>puzzle</h3>(.*?)<div class="panel"><h3>solution</h3>',
        text,
        flags=re.DOTALL,
    )
    if match is None:
        raise RuntimeError(f"Could not parse puzzle from {paths[0]}")
    wrong: dict[str, int] = {}
    for chunk in text.split('<div class="panel"><h3>loop ')[1:]:
        loop = re.match(r"(\d+)</h3>", chunk)
        if loop:
            wrong[f"loop{loop.group(1)}"] = chunk.count('class="cell wrong"')
    return paths[0], hashlib.sha256(match.group(1).encode()).hexdigest(), wrong


def metric(metrics: dict[str, Any], range_key: str, loop: int) -> dict[str, float]:
    return metrics["official_eval_by_blank_range"][range_key]["eval_clean"][f"loop{loop}"]


def extract(label: str, run_dir: Path) -> dict[str, Any]:
    payload = load_json(result_path(run_dir))
    args = payload["args"]
    metrics = payload["metrics"]
    train = metrics["train"]
    runtime = train["fla_runtime"]
    config = load_json(run_dir / "config.json")
    if config.get("git_dirty"):
        raise RuntimeError(f"{label} run used dirty source")
    if (run_dir / "source.patch").read_text(encoding="utf-8").strip():
        raise RuntimeError(f"{label} run has a non-empty source patch")
    if args["seed"] != 52 or args["future_seed_scale"] != 1.0:
        raise RuntimeError(f"{label} is not the single-seed FutureSeed arm")
    if not runtime.get("strict") or runtime.get("fla_source_sha") != FLA_SHA:
        raise RuntimeError(f"{label} did not use the pinned strict official FLA tree")
    if train["backbone_runtime"].get("silent_fallback_allowed"):
        raise RuntimeError(f"{label} allowed a silent fallback")
    layers = runtime.get("layers", [])
    if len(layers) != int(args["layers"]):
        raise RuntimeError(f"{label} did not report every mixer layer")
    expected_class = {
        "gdn2": "fla.layers.gdn2.GatedDeltaNet2",
        "raven": "fla.layers.raven.Raven",
    }[args["backbone"]]
    classes = {row.get("class") for row in layers}
    if classes != {expected_class}:
        raise RuntimeError(f"{label} did not use the exact official class: {classes}")
    expected_path = {
        "gdn2": "official_layer_forward",
        "raven": "official_raven_sparse_slot_router_then_gsa_chunk",
    }[args["backbone"]]
    paths = {row.get("execution_path") for row in layers}
    if paths != {expected_path}:
        raise RuntimeError(f"{label} execution path drifted: {paths}")
    conv_backends = {
        backend
        for row in layers
        for backend in row.get("conv_backends", {}).values()
    }
    expected_conv = {"triton"} if args["backbone"] == "gdn2" else {"disabled"}
    if conv_backends != expected_conv:
        raise RuntimeError(f"{label} short-conv contract drifted: {conv_backends}")
    state_elements = {int(row["state_elements_per_head"]) for row in layers}
    if state_elements != {1024}:
        raise RuntimeError(f"{label} state budget is not 1024 elements/head: {state_elements}")

    case_path, puzzle_sha, wrong = primary_case(run_dir)
    loops = []
    for loop in range(1, 6):
        row = metrics["eval_clean"][f"loop{loop}"]
        loops.append(
            {
                "loop": loop,
                "exact": float(row["label_exact"]),
                "blank_acc": float(row["blank_acc"]),
                "wrong_cells": wrong.get(f"loop{loop}"),
            }
        )
    ranges: dict[str, Any] = {}
    for key in ("b46_50", "b51_55", "b56_64"):
        ranges[key] = {
            "loop1": metric(metrics, key, 1),
            "loop5": metric(metrics, key, 5),
        }
    return {
        "label": label,
        "run_name": run_dir.name,
        "run_dir": str(run_dir),
        "git_sha": config["git_sha"],
        "args": args,
        "parameter_count": int(train["parameter_count"]),
        "train_ce": float(train["train_ce_loss"]),
        "sec_per_step": float(train["train_sec"]) / int(train["optimizer_steps"]),
        "peak_mib": float(train["cuda_max_memory_allocated_mb"]),
        "loops": loops,
        "ranges": ranges,
        "puzzle_sha256": puzzle_sha,
        "primary_case": str(case_path),
        "runtime": runtime,
    }


def pct(value: float) -> str:
    return f"{100 * value:.2f}%"


def rel_link(out_dir: Path, target: Path) -> str:
    return str(Path("..") / target.parent.parent.name / "output" / target.name)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gdn2", type=Path, required=True)
    parser.add_argument("--raven", type=Path, required=True)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--out_dir", type=Path, required=True)
    args = parser.parse_args()

    contract = load_json(args.contract)
    if contract.get("status") != "pass" or contract.get("cpu_fallback"):
        raise RuntimeError("Raven CUDA contract did not pass")
    gdn2 = extract("GDN2 + FutureSeed", args.gdn2)
    raven = extract("Raven + FutureSeed", args.raven)
    if gdn2["git_sha"] != raven["git_sha"]:
        raise RuntimeError("Source SHAs differ across arms")
    if gdn2["puzzle_sha256"] != raven["puzzle_sha256"]:
        raise RuntimeError("Primary visualization puzzle differs across arms")

    ignored = {
        "backbone",
        "gdn_use_short_conv",
        "raven_num_slots",
        "raven_topk",
        "out_dir",
        "train_checkpoint_dir",
    }
    mismatches = {
        key: [gdn2["args"].get(key), raven["args"].get(key)]
        for key in sorted(set(gdn2["args"]) | set(raven["args"]))
        if key not in ignored and gdn2["args"].get(key) != raven["args"].get(key)
    }
    if mismatches:
        raise RuntimeError(f"Cross-arm config mismatch: {mismatches}")

    primary_key = "b51_55"
    g_primary = gdn2["ranges"][primary_key]["loop5"]
    r_primary = raven["ranges"][primary_key]["loop5"]
    exact_delta = float(r_primary["label_exact"] - g_primary["label_exact"])
    blank_delta = float(r_primary["blank_acc"] - g_primary["blank_acc"])
    raven_loop_gain = raven["loops"][-1]["exact"] - raven["loops"][0]["exact"]
    speed_ratio = raven["sec_per_step"] / gdn2["sec_per_step"]
    decision = (
        "Raven is better on the primary hard range at matched recurrent-state capacity."
        if exact_delta > 0
        else "Raven does not beat GDN2 on the primary hard-range exact metric."
    )
    comparison = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "hypothesis": (
            "Sparse content-routed slots preserve FutureSeed context with less interference "
            "than GDN2's dense recurrent memory."
        ),
        "fairness": {
            "same_git_sha": gdn2["git_sha"],
            "same_seed": 52,
            "same_data_optimizer_loops": True,
            "state_elements_per_head": 1024,
            "raven_slots_topk": [16, 2],
            "raven_occupancy": 0.125,
            "official_fla_sha": FLA_SHA,
            "gdn2_short_conv_retained_as_strong_baseline": True,
            "silent_fallback": False,
        },
        "contract": contract,
        "arms": {"gdn2": gdn2, "raven": raven},
        "primary": {
            "range": "51-55 blanks",
            "raven_minus_gdn2_exact": exact_delta,
            "raven_minus_gdn2_blank_acc": blank_delta,
            "raven_loop5_minus_loop1_exact": raven_loop_gain,
            "raven_over_gdn2_sec_per_step": speed_ratio,
        },
        "decision": decision,
    }

    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "comparison.json").write_text(
        json.dumps(comparison, indent=2) + "\n", encoding="utf-8"
    )

    md_lines = [
        "# Raven FutureSeed vs GDN2 FutureSeed",
        "",
        f"- Decision: {decision}",
        f"- 51-55 exact delta: {exact_delta:+.4f}",
        f"- 51-55 blank-accuracy delta: {blank_delta:+.4f}",
        f"- Raven loop5-loop1 exact: {raven_loop_gain:+.4f}",
        f"- Raven/GDN2 sec per step: {speed_ratio:.3f}x",
        "- Both arms use 1024 recurrent-state elements per head and the same source/data/seed/optimizer/loop loss.",
        "- Raven uses 16 slots and top-2 routing; GDN2 keeps its established official short convolution.",
    ]
    (args.out_dir / "README.md").write_text("\n".join(md_lines) + "\n", encoding="utf-8")

    range_rows = []
    for key, label in (("b46_50", "46-50"), ("b51_55", "51-55"), ("b56_64", "56-64")):
        gm = gdn2["ranges"][key]["loop5"]
        rm = raven["ranges"][key]["loop5"]
        range_rows.append(
            "<tr>"
            f"<td>{label}</td><td>{pct(gm['label_exact'])}</td><td>{pct(rm['label_exact'])}</td>"
            f"<td>{float(rm['label_exact'] - gm['label_exact']):+.4f}</td>"
            f"<td>{pct(gm['blank_acc'])}</td><td>{pct(rm['blank_acc'])}</td>"
            "</tr>"
        )
    loop_rows = []
    for g_row, r_row in zip(gdn2["loops"], raven["loops"]):
        loop_rows.append(
            "<tr>"
            f"<td>{g_row['loop']}</td><td>{pct(g_row['exact'])}</td><td>{pct(r_row['exact'])}</td>"
            f"<td>{pct(g_row['blank_acc'])}</td><td>{pct(r_row['blank_acc'])}</td>"
            f"<td>{g_row['wrong_cells']}</td><td>{r_row['wrong_cells']}</td>"
            "</tr>"
        )
    g_case = rel_link(args.out_dir, Path(gdn2["primary_case"]))
    r_case = rel_link(args.out_dir, Path(raven["primary_case"]))
    page = f"""<!doctype html>
<html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Raven FutureSeed vs GDN2</title>
<style>
body{{margin:0;font:14px/1.5 ui-sans-serif,system-ui;color:#171717;background:#f7f7f5}}
header,section{{padding:24px max(24px,calc((100vw - 1180px)/2))}}header{{background:#171717;color:white}}
h1{{margin:0 0 8px;font-size:28px}}h2{{font-size:19px;margin:0 0 14px}}p{{max-width:900px}}
table{{width:100%;border-collapse:collapse;background:white}}th,td{{padding:10px;border:1px solid #d8d8d3;text-align:right}}th:first-child,td:first-child{{text-align:left}}
.verdict{{font-size:17px;font-weight:700;color:{'#176b55' if exact_delta > 0 else '#9b3426'}}}
.facts{{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:1px;background:#d8d8d3;border:1px solid #d8d8d3}}
.fact{{background:white;padding:14px}}.fact b{{display:block;font-size:20px}}
.cases{{display:grid;grid-template-columns:1fr 1fr;gap:12px}}iframe{{width:100%;height:760px;border:1px solid #bbb;background:white}}
@media(max-width:800px){{.facts,.cases{{grid-template-columns:1fr}}iframe{{height:680px}}}}
</style></head><body>
<header><h1>Raven + FutureSeed vs GDN2 + FutureSeed</h1><p>Same source, official Sudoku data, seed, optimizer, loop supervision, and 1024 recurrent-state elements per head.</p></header>
<section><p class="verdict">{html.escape(decision)}</p><div class="facts">
<div class="fact"><span>51-55 exact delta</span><b>{exact_delta:+.4f}</b></div>
<div class="fact"><span>51-55 blank-acc delta</span><b>{blank_delta:+.4f}</b></div>
<div class="fact"><span>Raven loop gain</span><b>{raven_loop_gain:+.4f}</b></div>
<div class="fact"><span>Raven speed ratio</span><b>{speed_ratio:.3f}x</b></div></div></section>
<section><h2>Hardness ranges, loop 5</h2><table><thead><tr><th>blanks</th><th>GDN2 exact</th><th>Raven exact</th><th>exact delta</th><th>GDN2 blank acc</th><th>Raven blank acc</th></tr></thead><tbody>{''.join(range_rows)}</tbody></table></section>
<section><h2>Loop behavior on the shared visual case</h2><table><thead><tr><th>loop</th><th>GDN2 exact</th><th>Raven exact</th><th>GDN2 blank acc</th><th>Raven blank acc</th><th>GDN2 wrong cells</th><th>Raven wrong cells</th></tr></thead><tbody>{''.join(loop_rows)}</tbody></table></section>
<section><h2>Same hard case, side by side</h2><div class="cases"><iframe src="{html.escape(g_case)}" title="GDN2 case"></iframe><iframe src="{html.escape(r_case)}" title="Raven case"></iframe></div></section>
</body></html>"""
    (args.out_dir / "index.html").write_text(page, encoding="utf-8")
    print(json.dumps(comparison["primary"], indent=2))
    print(f"wrote {args.out_dir / 'index.html'}")


if __name__ == "__main__":
    main()
