#!/usr/bin/env python3
"""Build the frozen-control P-GDN3-017 decision and same-board visualization."""

import hashlib
import html
import json
import math
import statistics
from pathlib import Path


ROOT = Path("/huyang2/double-loop")
CONTROL_RUN = "p-fs3-001-terminal-s3100-20260806T200859Z-3e167b6"
CANDIDATE_RUN = "p-gdn3-017-raven-routed-s3100-20260810T065231Z-e884b7d"
OUTPUT = ROOT / "runs/p-gdn3-017-comparison-20260810T071900Z-e884b7d"
SOURCE_SHA = "e884b7df948334ba6ca326717414cb3045b0d5bb"
GPU_UUID = "GPU-bfb964ca-068f-8fd1-8f27-463dca141125"
PARENT_TERMINAL_RMS = 6.697616696357727
RANGES = ("b51_55", "b56_60", "b61_64")
LOOPS = tuple(f"loop{i}" for i in range(1, 6))
ACTIVATION_KEYS = (
    "gdn3_raven_routed_enabled",
    "gdn3_raven_routed_allocation_abs_from_one",
    "gdn3_raven_routed_allocation_slot_std",
    "gdn3_raven_routed_allocation_batch_std",
    "gdn3_raven_routed_allocation_token_std",
    "gdn3_raven_routed_allocation_entropy_normalized",
    "gdn3_raven_routed_allocation_min",
    "gdn3_raven_routed_allocation_max",
    "gdn3_raven_routed_k_relative_change",
    "gdn3_raven_routed_g_relative_change",
    "gdn3_raven_routed_router_weight_rms",
    "gdn3_raven_routed_terminal_rms",
    "gdn3_raven_routed_terminal_batch_std",
)


def load_json(path: Path):
    with path.open() as handle:
        return json.load(handle)


def sha256(path: Path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def artifact(path: Path):
    return {"path": str(path), "sha256": sha256(path), "size": path.stat().st_size}


def loop_metrics(eval_clean):
    return {
        loop: {
            "exact": eval_clean[loop]["label_exact"],
            "blank_acc": eval_clean[loop]["blank_acc"],
        }
        for loop in LOOPS
    }


def case_map(run: str, label: str):
    path = ROOT / f"runs/{run}/output/case_bank/official_{label}/all_cases.json"
    payload = load_json(path)
    return path, payload, {case["case_id"]: case for case in payload["cases"]}


def paired_stats(control_cases, candidate_cases):
    ids = sorted(set(control_cases) & set(candidate_cases))
    if len(ids) != len(control_cases) or len(ids) != len(candidate_cases):
        raise RuntimeError("case-bank identity mismatch")
    for case_id in ids:
        control = control_cases[case_id]
        candidate = candidate_cases[case_id]
        if (
            control["content_sha256"] != candidate["content_sha256"]
            or control["label"] != candidate["label"]
        ):
            raise RuntimeError(f"case content mismatch: {case_id}")

    def means(cases):
        return [
            statistics.fmean(
                cases[case_id]["loops"][loop]["wrong_total_count"] for case_id in ids
            )
            for loop in LOOPS
        ]

    control_means = means(control_cases)
    candidate_means = means(candidate_cases)
    control_late = [
        control_cases[case_id]["loops"]["loop3"]["wrong_total_count"]
        - control_cases[case_id]["loops"]["loop5"]["wrong_total_count"]
        for case_id in ids
    ]
    candidate_late = [
        candidate_cases[case_id]["loops"]["loop3"]["wrong_total_count"]
        - candidate_cases[case_id]["loops"]["loop5"]["wrong_total_count"]
        for case_id in ids
    ]
    loop5_delta = [
        candidate_cases[case_id]["loops"]["loop5"]["wrong_total_count"]
        - control_cases[case_id]["loops"]["loop5"]["wrong_total_count"]
        for case_id in ids
    ]
    late_delta = [new - old for old, new in zip(control_late, candidate_late)]
    return {
        "shared_boards": len(ids),
        "control_mean_wrong_cells_loops1_5": control_means,
        "candidate_mean_wrong_cells_loops1_5": candidate_means,
        "control_loop3_to_5_correction": statistics.fmean(control_late),
        "candidate_loop3_to_5_correction": statistics.fmean(candidate_late),
        "loop5_candidate_better_equal_worse": {
            "better": sum(delta < 0 for delta in loop5_delta),
            "equal": sum(delta == 0 for delta in loop5_delta),
            "worse": sum(delta > 0 for delta in loop5_delta),
        },
        "late_correction_candidate_stronger_equal_weaker": {
            "stronger": sum(delta > 0 for delta in late_delta),
            "equal": sum(delta == 0 for delta in late_delta),
            "weaker": sum(delta < 0 for delta in late_delta),
        },
    }


def board(values, label, clue_mask, title):
    cells = []
    for index, value in enumerate(values):
        kind = "clue" if clue_mask[index] else "wrong" if value != label[index] else "correct"
        cells.append(f'<span class="{kind}">{html.escape(str(value))}</span>')
    return f'<figure><figcaption>{html.escape(title)}</figcaption><div class="board">{"".join(cells)}</div></figure>'


def hardest_case_html(label, control_cases, candidate_cases):
    shared = sorted(set(control_cases) & set(candidate_cases))
    case_id = max(
        shared,
        key=lambda key: (
            max(
                control_cases[key]["loops"]["loop5"]["wrong_total_count"],
                candidate_cases[key]["loops"]["loop5"]["wrong_total_count"],
            ),
            abs(
                candidate_cases[key]["loops"]["loop5"]["wrong_total_count"]
                - control_cases[key]["loops"]["loop5"]["wrong_total_count"]
            ),
        ),
    )
    control = control_cases[case_id]
    candidate = candidate_cases[case_id]
    boards = [board(control["label"], control["label"], control["clue_mask"], "solution")]
    for arm, case in (("control", control), ("raven-routed", candidate)):
        for loop in LOOPS:
            wrong = case["loops"][loop]["wrong_total_count"]
            boards.append(
                board(
                    case["loops"][loop]["prediction"],
                    case["label"],
                    case["clue_mask"],
                    f"{arm} {loop} wrong={wrong}",
                )
            )
    return (
        f'<section><h3>{html.escape(label)} hardest shared board</h3>'
        f'<p><code>{html.escape(case_id)}</code></p><div class="boards">{"".join(boards)}</div></section>'
    )


def main():
    control_dir = ROOT / f"runs/{CONTROL_RUN}"
    candidate_dir = ROOT / f"runs/{CANDIDATE_RUN}"
    control_metrics_path = control_dir / "output/futureseed_loop_seed52.json"
    candidate_metrics_path = candidate_dir / "output/futureseed_loop_seed52.json"
    control = load_json(control_metrics_path)["metrics"]
    candidate = load_json(candidate_metrics_path)["metrics"]

    official = {"control": {}, "candidate": {}}
    for label in ("b46_50",) + RANGES:
        official["control"][label] = loop_metrics(
            control["official_eval_by_blank_range"][label]["eval_clean"]
        )
        official["candidate"][label] = loop_metrics(
            candidate["official_eval_by_blank_range"][label]["eval_clean"]
        )
    full_eval = {
        "control": loop_metrics(control["eval_clean"]),
        "candidate": loop_metrics(candidate["eval_clean"]),
    }
    hard_control = statistics.fmean(
        official["control"][label]["loop5"]["exact"] for label in RANGES
    )
    hard_candidate = statistics.fmean(
        official["candidate"][label]["loop5"]["exact"] for label in RANGES
    )
    mixed_control = full_eval["control"]["loop5"]["exact"]
    mixed_candidate = full_eval["candidate"]["loop5"]["exact"]

    same_board = {}
    case_payloads = {}
    data_hashes = {}
    case_artifacts = {}
    for label in RANGES:
        control_path, control_payload, control_cases = case_map(CONTROL_RUN, label)
        candidate_path, candidate_payload, candidate_cases = case_map(CANDIDATE_RUN, label)
        if control_payload["data_hash"] != candidate_payload["data_hash"]:
            raise RuntimeError(f"case-bank data hash mismatch: {label}")
        same_board[label] = paired_stats(control_cases, candidate_cases)
        case_payloads[label] = (control_cases, candidate_cases)
        data_hashes[label] = control_payload["data_hash"]
        case_artifacts[f"control_cases_{label}"] = control_path
        case_artifacts[f"candidate_cases_{label}"] = candidate_path

    activation = {
        loop: {
            key: candidate["eval_clean"][f"{loop}/future_seed"][key]
            for key in ACTIVATION_KEYS
        }
        for loop in LOOPS
    }
    activation_pass = all(
        abs(values["gdn3_raven_routed_enabled"] - 1.0) < 1e-6
        and values["gdn3_raven_routed_allocation_abs_from_one"] >= 1e-4
        and values["gdn3_raven_routed_allocation_slot_std"] >= 1e-4
        and values["gdn3_raven_routed_allocation_batch_std"] >= 1e-6
        and values["gdn3_raven_routed_allocation_token_std"] >= 1e-6
        and 0.35 <= values["gdn3_raven_routed_allocation_entropy_normalized"] <= 1.0
        and values["gdn3_raven_routed_allocation_min"] > 0.0
        and values["gdn3_raven_routed_allocation_max"] < 7.5
        and values["gdn3_raven_routed_k_relative_change"] >= 1e-4
        and values["gdn3_raven_routed_g_relative_change"] >= 1e-4
        and values["gdn3_raven_routed_router_weight_rms"] > 0.0
        and 0.0 < values["gdn3_raven_routed_terminal_rms"] <= 4.0 * PARENT_TERMINAL_RMS
        and values["gdn3_raven_routed_terminal_batch_std"] > 0.0
        and all(math.isfinite(value) for value in values.values())
        for values in activation.values()
    )
    blank_delta = {
        label: official["candidate"][label]["loop5"]["blank_acc"]
        - official["control"][label]["loop5"]["blank_acc"]
        for label in RANGES
    }
    primary_pass = hard_candidate - hard_control >= 0.02 and all(
        delta >= -0.01 for delta in blank_delta.values()
    )
    late_stronger_all = all(
        same_board[label]["candidate_loop3_to_5_correction"]
        > same_board[label]["control_loop3_to_5_correction"]
        for label in RANGES
    )
    hardest_nonregressive = (
        official["candidate"]["b61_64"]["loop5"]["exact"]
        >= official["control"]["b61_64"]["loop5"]["exact"]
        and official["candidate"]["b61_64"]["loop5"]["blank_acc"]
        >= official["control"]["b61_64"]["loop5"]["blank_acc"]
    )
    alternate_pass = (
        mixed_candidate - mixed_control >= 0.03
        and hardest_nonregressive
        and late_stronger_all
    )

    train = candidate["train"]
    effective_boards = 100 * train["effective_batch"]
    throughput = effective_boards / train["train_sec"]
    allocated = train["cuda_max_memory_allocated_mb"]
    reserved = train["cuda_max_memory_reserved_mb"]
    throughput_pass = throughput >= 12.0
    allocated_pass = allocated < 20 * 1024
    reserved_pass = reserved < 24 * 1024

    artifact_paths = {
        "control_metrics": control_metrics_path,
        "candidate_metrics": candidate_metrics_path,
        "candidate_config": candidate_dir / "config.json",
        "candidate_log": candidate_dir / "logs/run.log",
        "candidate_checkpoint": ROOT / f"models/{CANDIDATE_RUN}/checkpoints/train_state_step003100.pt",
        "candidate_source_snapshot": candidate_dir / "source_snapshot.tar.gz",
        "formal_log": ROOT / "artifacts/launch/p-gdn3-017/formal-20260810T065231Z-e884b7d.log",
        "formal_status": ROOT / "artifacts/launch/p-gdn3-017/formal-20260810T065231Z-e884b7d.status",
        "contract_log": ROOT / "artifacts/launch/p-gdn3-017/contract-e884b7d-r2.log",
        "probe_log": ROOT / "artifacts/launch/p-gdn3-017/probe-20260810T064611Z-e884b7d.log",
        "probe_metrics": ROOT / "runs/p-gdn3-017-raven-routed-probe-s3001-20260810T064611Z-e884b7d/output/futureseed_loop_seed52.json",
        "probe_checkpoint": ROOT / "models/p-gdn3-017-raven-routed-probe-s3001-20260810T064611Z-e884b7d/checkpoints/train_state_step003001.pt",
        **case_artifacts,
    }
    result = {
        "experiment_id": "P-GDN3-017",
        "mechanism": "raven_content_dependent_soft_slot_allocation_inside_official_gdn2",
        "source_sha": SOURCE_SHA,
        "decision": "discarded",
        "decision_reason": (
            "All twelve Raven routers activate, but endpoint allocation approaches a one-slot "
            "collapse and violates the preregistered max-allocation stability bound. Hard 51-64 "
            "macro exact is unchanged and mixed exact regresses, while the alternate same-board "
            "route is weaker in 56-60 and 61-64. The mechanism is closed without temperature, "
            "slot-count, scale, seed, optimizer, or duration rescue."
        ),
        "integrity": {
            "formal_status": 0,
            "single_gpu_only": True,
            "target_gpu_uuid": GPU_UUID,
            "source_clean_detached": True,
            "fallback": False,
            "nan": False,
            "oom": False,
            "case_bank_data_hashes": data_hashes,
        },
        "comparison": {
            "hard_51_64_macro_loop5_exact": {
                "control": hard_control,
                "candidate": hard_candidate,
                "delta": hard_candidate - hard_control,
            },
            "mixed_loop5_exact": {
                "control": mixed_control,
                "candidate": mixed_candidate,
                "delta": mixed_candidate - mixed_control,
            },
            "official_loop5_blank_deltas": blank_delta,
        },
        "registered_gate": {
            "passed": False,
            "activation_and_stability_pass": activation_pass,
            "primary_quality_pass": primary_pass,
            "alternate_quality_pass": alternate_pass,
            "hardest_range_nonregressive": hardest_nonregressive,
            "late_correction_stronger_all_ranges": late_stronger_all,
            "systems": {
                "hardware": "A100-SXM4-40GB",
                "effective_boards_per_sec": throughput,
                "throughput_floor": 12.0,
                "throughput_pass": throughput_pass,
                "peak_allocated_mb": allocated,
                "peak_allocated_limit_mb": 20 * 1024,
                "peak_allocated_pass": allocated_pass,
                "peak_reserved_mb": reserved,
                "peak_reserved_limit_mb": 24 * 1024,
                "peak_reserved_pass": reserved_pass,
                "stable_step_timing_cv": None,
                "stable_step_timing_cv_limit": 0.10,
                "timing_cv_note": (
                    "Not separately measured because activation stability and both quality routes "
                    "already failed; no additional GPU benchmark was justified."
                ),
            },
        },
        "train": {
            "control_ce": control["train"]["train_ce_loss"],
            "candidate_ce": train["train_ce_loss"],
            "candidate_train_sec": train["train_sec"],
            "candidate_parameter_count": train["parameter_count"],
            "candidate_effective_batch": train["effective_batch"],
        },
        "activation": activation,
        "full_eval": full_eval,
        "official_eval": official,
        "same_board_all_256": same_board,
        "artifacts": {name: artifact(path) for name, path in artifact_paths.items()},
    }

    OUTPUT.mkdir(parents=True, exist_ok=True)
    json_path = OUTPUT / "comparison.json"
    json_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    abort_path = OUTPUT / "abort.json"
    abort_path.write_text(
        json.dumps(
            {
                "experiment_id": result["experiment_id"],
                "run": CANDIDATE_RUN,
                "source_sha": SOURCE_SHA,
                "status": "discarded",
                "reason": "science_and_stability_gate_miss",
                "process_action": "formal process exited naturally with status 0; no kill required",
                "failed_gates": {
                    "activation_and_stability": not activation_pass,
                    "primary_quality": not primary_pass,
                    "alternate_quality": not alternate_pass,
                },
                "rescue_forbidden": [
                    "temperature",
                    "slot_count",
                    "route_scale",
                    "seed",
                    "learning_rate",
                    "loss",
                    "batch",
                    "width",
                    "depth",
                    "duration",
                ],
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )

    official_rows = []
    for label in RANGES:
        for loop in LOOPS:
            old = official["control"][label][loop]
            new = official["candidate"][label][loop]
            official_rows.append(
                f"<tr><td>{label}</td><td>{loop[-1]}</td><td>{old['exact']:.6f}</td>"
                f"<td>{new['exact']:.6f}</td><td>{new['exact']-old['exact']:+.6f}</td>"
                f"<td>{old['blank_acc']:.6f}</td><td>{new['blank_acc']:.6f}</td>"
                f"<td>{new['blank_acc']-old['blank_acc']:+.6f}</td></tr>"
            )
    paired_rows = []
    for label in RANGES:
        stats = same_board[label]
        paired_rows.append(
            f"<tr><td>{label}</td><td>{stats['shared_boards']}</td>"
            f"<td>{' / '.join(f'{value:.2f}' for value in stats['control_mean_wrong_cells_loops1_5'])}</td>"
            f"<td>{' / '.join(f'{value:.2f}' for value in stats['candidate_mean_wrong_cells_loops1_5'])}</td>"
            f"<td>{stats['control_loop3_to_5_correction']:.3f}</td>"
            f"<td>{stats['candidate_loop3_to_5_correction']:.3f}</td></tr>"
        )
    activation_rows = []
    for loop in LOOPS:
        values = activation[loop]
        activation_rows.append(
            f"<tr><td>{loop[-1]}</td>"
            f"<td>{values['gdn3_raven_routed_allocation_abs_from_one']:.4f}</td>"
            f"<td>{values['gdn3_raven_routed_allocation_slot_std']:.4f}</td>"
            f"<td>{values['gdn3_raven_routed_allocation_batch_std']:.4f}</td>"
            f"<td>{values['gdn3_raven_routed_allocation_token_std']:.4f}</td>"
            f"<td>{values['gdn3_raven_routed_allocation_entropy_normalized']:.4f}</td>"
            f"<td>{values['gdn3_raven_routed_allocation_min']:.6f}</td>"
            f"<td>{values['gdn3_raven_routed_allocation_max']:.6f}</td>"
            f"<td>{values['gdn3_raven_routed_k_relative_change']:.4f}</td>"
            f"<td>{values['gdn3_raven_routed_g_relative_change']:.4f}</td>"
            f"<td>{values['gdn3_raven_routed_terminal_rms']:.4f}</td></tr>"
        )
    case_sections = "".join(
        hardest_case_html(label, *case_payloads[label]) for label in RANGES
    )
    html_path = OUTPUT / "comparison.html"
    html_path.write_text(
        f"""<!doctype html><html><head><meta charset="utf-8"><title>P-GDN3-017 decision</title>
<style>body{{font:15px system-ui,sans-serif;max-width:1500px;margin:28px auto;padding:0 20px;color:#17202a}}h1,h2,h3{{letter-spacing:0}}table{{border-collapse:collapse;width:100%;margin:12px 0 28px}}th,td{{border:1px solid #cbd5e1;padding:7px 9px;text-align:right}}th:first-child,td:first-child{{text-align:left}}th{{background:#eef2f7}}.fail{{color:#a61b1b;font-weight:700}}code{{background:#f3f4f6;padding:2px 4px}}.boards{{display:flex;flex-wrap:wrap;gap:10px}}figure{{margin:0}}figcaption{{font-size:12px;max-width:126px}}.board{{display:grid;grid-template-columns:repeat(9,14px);grid-template-rows:repeat(9,14px);border:2px solid #1f2937}}.board span{{font:10px/14px ui-monospace,monospace;text-align:center;border-right:1px solid #d1d5db;border-bottom:1px solid #d1d5db}}.board span:nth-child(3n){{border-right-color:#1f2937}}.board span:nth-child(n+19):nth-child(-n+27),.board span:nth-child(n+46):nth-child(-n+54){{border-bottom-color:#1f2937}}.clue{{background:#e5e7eb;font-weight:700}}.wrong{{background:#fecaca;color:#991b1b}}.correct{{background:#dcfce7}}section{{margin:30px 0 42px}}</style></head><body>
<h1>P-GDN3-017 Raven-Routed GDN</h1><p class="fail">Discarded: routing strongly activates and approaches one-slot collapse, but exact quality does not improve.</p>
<ul><li>Hard macro exact {hard_control:.6f} to {hard_candidate:.6f} ({hard_candidate-hard_control:+.6f}).</li><li>Mixed exact {mixed_control:.6f} to {mixed_candidate:.6f} ({mixed_candidate-mixed_control:+.6f}).</li><li>A100 40GB throughput {throughput:.3f} boards/s; allocated/reserved {allocated:.1f}/{reserved:.1f} MiB.</li></ul>
<h2>Routing activation</h2><table><thead><tr><th>Loop</th><th>|a-1|</th><th>Slot std</th><th>Board std</th><th>Token std</th><th>Entropy</th><th>Min</th><th>Max</th><th>K change</th><th>g change</th><th>State RMS</th></tr></thead><tbody>{''.join(activation_rows)}</tbody></table>
<h2>Official metrics</h2><table><thead><tr><th>Range</th><th>Loop</th><th>Control exact</th><th>Candidate exact</th><th>Delta exact</th><th>Control blank</th><th>Candidate blank</th><th>Delta blank</th></tr></thead><tbody>{''.join(official_rows)}</tbody></table>
<h2>Same-board wrong-cell dynamics</h2><table><thead><tr><th>Range</th><th>Boards</th><th>Control loops1-5</th><th>Candidate loops1-5</th><th>Control L3-L5</th><th>Candidate L3-L5</th></tr></thead><tbody>{''.join(paired_rows)}</tbody></table>
<h2>Hardest shared boards</h2>{case_sections}<p>Machine-readable evidence: <code>comparison.json</code>.</p></body></html>"""
    )
    script_path = OUTPUT / Path(__file__).name
    script_path.write_text(Path(__file__).read_text())
    manifest_path = OUTPUT / "manifest.sha256"
    manifest_path.write_text(
        "".join(
            f"{sha256(path)}  {path.name}\n"
            for path in (json_path, abort_path, html_path, script_path)
        )
    )
    print(
        json.dumps(
            {
                "output": str(OUTPUT),
                "activation_pass": activation_pass,
                "hard_macro_delta": hard_candidate - hard_control,
                "mixed_delta": mixed_candidate - mixed_control,
                "throughput": throughput,
                "decision": "discarded",
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
