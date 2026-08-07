#!/usr/bin/env python3
import hashlib
import html
import json
import math
import statistics
from pathlib import Path


CONTROL_RUN = "p-fs3-001-terminal-s3100-20260806T200859Z-3e167b6"
CANDIDATE_RUN = "p-gdn3-012-adaptive-signed-erase-s3100-20260807T083104Z-c09c368"
ROOT = Path("/huyang2/double-loop")
OUTPUT = ROOT / "runs/p-gdn3-012-comparison-20260807T090000Z-c09c368"
SOURCE_SHA = "c09c36851e5a17224eae9d38a40923c77fe6cba2"
GPU_UUID = "GPU-53e9f3b4-2966-65d3-6614-09c540921519"
RANGES = ("b51_55", "b56_60", "b61_64")
LOOPS = tuple(f"loop{i}" for i in range(1, 6))


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


def metric_loops(eval_clean):
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
        c = control_cases[case_id]
        n = candidate_cases[case_id]
        if c["content_sha256"] != n["content_sha256"] or c["label"] != n["label"]:
            raise RuntimeError(f"content mismatch: {case_id}")

    def means(cases):
        return [
            statistics.fmean(cases[case_id]["loops"][loop]["wrong_total_count"] for case_id in ids)
            for loop in LOOPS
        ]

    control_means = means(control_cases)
    candidate_means = means(candidate_cases)
    loop5_deltas = [
        candidate_cases[case_id]["loops"]["loop5"]["wrong_total_count"]
        - control_cases[case_id]["loops"]["loop5"]["wrong_total_count"]
        for case_id in ids
    ]
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
    late_deltas = [n - c for c, n in zip(control_late, candidate_late)]
    return {
        "shared_boards": len(ids),
        "control_mean_wrong_cells_loops1_5": control_means,
        "candidate_mean_wrong_cells_loops1_5": candidate_means,
        "control_loop1_to_5_correction": control_means[0] - control_means[4],
        "candidate_loop1_to_5_correction": candidate_means[0] - candidate_means[4],
        "control_loop3_to_5_correction": statistics.fmean(control_late),
        "candidate_loop3_to_5_correction": statistics.fmean(candidate_late),
        "loop5_candidate_better_equal_worse": {
            "better": sum(delta < 0 for delta in loop5_deltas),
            "equal": sum(delta == 0 for delta in loop5_deltas),
            "worse": sum(delta > 0 for delta in loop5_deltas),
        },
        "late_correction_candidate_stronger_equal_weaker": {
            "stronger": sum(delta > 0 for delta in late_deltas),
            "equal": sum(delta == 0 for delta in late_deltas),
            "weaker": sum(delta < 0 for delta in late_deltas),
        },
    }


def grid(values, label, clue_mask, title):
    cells = []
    for index, value in enumerate(values):
        classes = []
        if clue_mask[index]:
            classes.append("clue")
        elif value != label[index]:
            classes.append("wrong")
        else:
            classes.append("correct")
        cells.append(f'<span class="{" ".join(classes)}">{html.escape(str(value))}</span>')
    return f'<figure><figcaption>{html.escape(title)}</figcaption><div class="board">{"".join(cells)}</div></figure>'


def hardest_case_html(label, control_cases, candidate_cases):
    shared = sorted(set(control_cases) & set(candidate_cases))
    ranked = sorted(
        shared,
        key=lambda case_id: (
            max(
                control_cases[case_id]["loops"]["loop5"]["wrong_total_count"],
                candidate_cases[case_id]["loops"]["loop5"]["wrong_total_count"],
            ),
            abs(
                candidate_cases[case_id]["loops"]["loop5"]["wrong_total_count"]
                - control_cases[case_id]["loops"]["loop5"]["wrong_total_count"]
            ),
        ),
        reverse=True,
    )[:2]
    sections = []
    for case_id in ranked:
        control = control_cases[case_id]
        candidate = candidate_cases[case_id]
        c_traj = [control["loops"][loop]["wrong_total_count"] for loop in LOOPS]
        n_traj = [candidate["loops"][loop]["wrong_total_count"] for loop in LOOPS]
        boards = [grid(control["label"], control["label"], control["clue_mask"], "solution")]
        for arm, case in (("control", control), ("adaptive-signed-erase", candidate)):
            for loop in LOOPS:
                boards.append(
                    grid(
                        case["loops"][loop]["prediction"],
                        case["label"],
                        case["clue_mask"],
                        f"{arm} {loop} wrong={case['loops'][loop]['wrong_total_count']}",
                    )
                )
        sections.append(
            f'<section class="case"><h3>{html.escape(label)} hardest same board</h3>'
            f'<p><code>{html.escape(case_id)}</code>; blanks={control["blank_count"]}; '
            f'control wrong {" / ".join(map(str, c_traj))}; adaptive-signed-erase wrong {" / ".join(map(str, n_traj))}.</p>'
            f'<div class="boards">{"".join(boards)}</div></section>'
        )
    return "".join(sections)


def main():
    control_dir = ROOT / f"runs/{CONTROL_RUN}"
    candidate_dir = ROOT / f"runs/{CANDIDATE_RUN}"
    control_metrics_path = control_dir / "output/futureseed_loop_seed52.json"
    candidate_metrics_path = candidate_dir / "output/futureseed_loop_seed52.json"
    control = load_json(control_metrics_path)["metrics"]
    candidate = load_json(candidate_metrics_path)["metrics"]

    official = {"control": {}, "candidate": {}}
    for label in ("b46_50",) + RANGES:
        official["control"][label] = metric_loops(
            control["official_eval_by_blank_range"][label]["eval_clean"]
        )
        official["candidate"][label] = metric_loops(
            candidate["official_eval_by_blank_range"][label]["eval_clean"]
        )

    full_eval = {
        "control": metric_loops(control["eval_clean"]),
        "candidate": metric_loops(candidate["eval_clean"]),
    }
    hard_macro_control = statistics.fmean(official["control"][label]["loop5"]["exact"] for label in RANGES)
    hard_macro_candidate = statistics.fmean(official["candidate"][label]["loop5"]["exact"] for label in RANGES)
    mixed_control = full_eval["control"]["loop5"]["exact"]
    mixed_candidate = full_eval["candidate"]["loop5"]["exact"]

    same_board = {}
    case_payloads = {}
    data_hashes = {}
    for label in RANGES:
        control_path, control_payload, control_cases = case_map(CONTROL_RUN, label)
        candidate_path, candidate_payload, candidate_cases = case_map(CANDIDATE_RUN, label)
        if control_payload["data_hash"] != candidate_payload["data_hash"]:
            raise RuntimeError(f"data hash mismatch for {label}")
        same_board[label] = paired_stats(control_cases, candidate_cases)
        data_hashes[label] = control_payload["data_hash"]
        case_payloads[label] = (control_cases, candidate_cases)

    control_sec = control["train"]["train_sec"]
    candidate_sec = candidate["train"]["train_sec"]
    control_alloc = control["train"]["cuda_max_memory_allocated_mb"]
    candidate_alloc = candidate["train"]["cuda_max_memory_allocated_mb"]
    control_reserved = control["train"]["cuda_max_memory_reserved_mb"]
    candidate_reserved = candidate["train"]["cuda_max_memory_reserved_mb"]
    effective_boards = 100 * candidate["train"]["effective_batch"]

    activation = {}
    for loop in LOOPS:
        values = candidate["eval_clean"][f"{loop}/future_seed"]
        activation[loop] = {
            key: values[key]
            for key in (
                "gdn3_adaptive_signed_erase_enabled",
                "gdn3_adaptive_signed_erase_residual_abs",
                "gdn3_adaptive_signed_erase_residual_relative_rms",
                "gdn3_adaptive_signed_erase_residual_batch_std",
                "gdn3_adaptive_signed_erase_residual_token_std",
                "gdn3_adaptive_signed_erase_residual_head_std",
                "gdn3_adaptive_signed_erase_b_relative_change",
                "gdn3_adaptive_signed_erase_above_one_frac",
                "gdn3_adaptive_signed_erase_effective_min",
                "gdn3_adaptive_signed_erase_effective_max",
                "gdn3_adaptive_signed_erase_terminal_rms",
                "gdn3_adaptive_signed_erase_terminal_batch_std",
                "gdn3_adaptive_signed_erase_weight_rms",
            )
        }

    artifact_paths = {
        "control_metrics": control_metrics_path,
        "candidate_metrics": candidate_metrics_path,
        "control_config": control_dir / "config.json",
        "candidate_config": candidate_dir / "config.json",
        "control_log": control_dir / "logs/run.log",
        "candidate_log": candidate_dir / "logs/run.log",
        "control_checkpoint": ROOT / f"models/{CONTROL_RUN}/checkpoints/train_state_step003100.pt",
        "candidate_checkpoint": ROOT / f"models/{CANDIDATE_RUN}/checkpoints/train_state_step003100.pt",
        "control_source_snapshot": control_dir / "source_snapshot.tar.gz",
        "candidate_source_snapshot": candidate_dir / "source_snapshot.tar.gz",
        "formal_log": ROOT / "artifacts/launch/p-gdn3-012/formal-p-gdn3-012-adaptive-signed-erase-s3100-20260807T083104Z-c09c368.log",
        "formal_status": ROOT / "artifacts/launch/p-gdn3-012/formal-p-gdn3-012-adaptive-signed-erase-s3100-20260807T083104Z-c09c368.status",
        "cuda_contract_log": ROOT / "artifacts/launch/p-gdn3-012/contract-c09c368-r1.log",
        "full_stack_probe_log": ROOT / "artifacts/launch/p-gdn3-012/probe-p-gdn3-012-adaptive-signed-erase-probe-s3001-20260807T082442Z-c09c368.log",
        "full_stack_probe_metrics": ROOT / "runs/p-gdn3-012-adaptive-signed-erase-probe-s3001-20260807T082442Z-c09c368/output/futureseed_loop_seed52.json",
        "full_stack_probe_checkpoint": ROOT / "models/p-gdn3-012-adaptive-signed-erase-probe-s3001-20260807T082442Z-c09c368/checkpoints/train_state_step003001.pt",
    }

    time_overhead = candidate_sec / control_sec - 1.0
    memory_overhead = candidate_alloc / control_alloc - 1.0
    hard_blank_deltas = {
        label: official["candidate"][label]["loop5"]["blank_acc"]
        - official["control"][label]["loop5"]["blank_acc"]
        for label in RANGES
    }
    activation_pass = all(
        abs(activation[loop]["gdn3_adaptive_signed_erase_enabled"] - 1.0) < 1e-6
        and activation[loop]["gdn3_adaptive_signed_erase_residual_abs"] >= 1e-4
        and activation[loop]["gdn3_adaptive_signed_erase_residual_relative_rms"] >= 1e-4
        and activation[loop]["gdn3_adaptive_signed_erase_residual_batch_std"] > 0.0
        and activation[loop]["gdn3_adaptive_signed_erase_residual_token_std"] > 0.0
        and activation[loop]["gdn3_adaptive_signed_erase_residual_head_std"] > 0.0
        and activation[loop]["gdn3_adaptive_signed_erase_b_relative_change"] >= 1e-4
        and activation[loop]["gdn3_adaptive_signed_erase_above_one_frac"] >= 1e-3
        and activation[loop]["gdn3_adaptive_signed_erase_effective_min"] >= 0.0
        and activation[loop]["gdn3_adaptive_signed_erase_effective_max"] <= 2.0
        and activation[loop]["gdn3_adaptive_signed_erase_terminal_rms"] > 0.0
        and activation[loop]["gdn3_adaptive_signed_erase_terminal_rms"] <= 30.0
        and activation[loop]["gdn3_adaptive_signed_erase_terminal_batch_std"] > 0.0
        and activation[loop]["gdn3_adaptive_signed_erase_weight_rms"] > 0.0
        and all(math.isfinite(value) for value in activation[loop].values())
        for loop in LOOPS
    )
    primary_pass = (
        hard_macro_candidate - hard_macro_control >= 0.02
        and all(delta >= -0.01 for delta in hard_blank_deltas.values())
    )
    late_correction_stronger_all_ranges = all(
        same_board[label]["candidate_loop3_to_5_correction"]
        > same_board[label]["control_loop3_to_5_correction"]
        for label in RANGES
    )
    hardest_range_nonregressive = (
        official["candidate"]["b61_64"]["loop5"]["exact"]
        >= official["control"]["b61_64"]["loop5"]["exact"]
        and official["candidate"]["b61_64"]["loop5"]["blank_acc"]
        >= official["control"]["b61_64"]["loop5"]["blank_acc"]
    )
    alternate_pass = (
        mixed_candidate - mixed_control >= 0.03
        and hardest_range_nonregressive
        and late_correction_stronger_all_ranges
    )
    time_cost_pass = time_overhead < 0.25
    memory_cost_pass = memory_overhead < 0.25
    result = {
        "experiment_id": "P-GDN3-012",
        "mechanism": "adaptive_signed_erase_spectrum",
        "source_sha": SOURCE_SHA,
        "decision": "discarded",
        "decision_reason": (
            "The adaptive signed erase spectrum activates in all twelve official GDN2 layers, opens "
            "a finite negative-retention regime, and stays numerically bounded within its registered "
            "cost limit. Hard 51-64 macro loop5 exact is unchanged and mixed exact regresses by one "
            "board, while same-board late-loop dynamics do not satisfy the alternate route. Both "
            "quality routes fail, so the mechanism is closed without projection, bound, target, "
            "sharing, or training rescue."
        ),
        "integrity": {
            "candidate_status": 0,
            "control_status": 0,
            "single_gpu_only": True,
            "target_gpu_uuid": GPU_UUID,
            "case_bank_data_hashes": data_hashes,
            "fallback": False,
            "nan": False,
            "oom": False,
        },
        "comparison": {
            "hard_51_64_macro_loop5_exact": {"control": hard_macro_control, "candidate": hard_macro_candidate},
            "hard_macro_exact_delta": hard_macro_candidate - hard_macro_control,
            "mixed_loop5_exact": {"control": mixed_control, "candidate": mixed_candidate},
            "mixed_exact_delta": mixed_candidate - mixed_control,
            "continuation_time_sec": {"control": control_sec, "candidate": candidate_sec},
            "continuation_time_overhead_fraction": time_overhead,
            "continuation_throughput_effective_boards_per_sec": {
                "control": effective_boards / control_sec,
                "candidate": effective_boards / candidate_sec,
            },
            "peak_allocated_memory_mb": {"control": control_alloc, "candidate": candidate_alloc},
            "peak_allocated_memory_overhead_fraction": memory_overhead,
            "peak_reserved_memory_mb": {"control": control_reserved, "candidate": candidate_reserved},
            "peak_reserved_memory_overhead_fraction": candidate_reserved / control_reserved - 1.0,
            "timing_scope": "fresh-process 100-step exact continuations with candidate shape precompiled by its registered step3001 probe; includes registered eval",
        },
        "registered_gate": {
            "passed": activation_pass
            and (primary_pass or alternate_pass)
            and time_cost_pass
            and memory_cost_pass,
            "activation_pass": activation_pass,
            "contract_12_layers_active": True,
            "primary_pass": primary_pass,
            "alternate_pass": alternate_pass,
            "hard_range_loop5_blank_deltas": hard_blank_deltas,
            "hardest_range_nonregressive": hardest_range_nonregressive,
            "late_correction_stronger_all_ranges": late_correction_stronger_all_ranges,
            "time_cost_pass": time_cost_pass,
            "memory_cost_pass": memory_cost_pass,
            "time_overhead_limit": 0.25,
            "memory_overhead_limit": 0.25,
        },
        "train": {
            "control_ce": control["train"]["train_ce_loss"],
            "candidate_ce": candidate["train"]["train_ce_loss"],
            "candidate_parameter_count": candidate["train"]["parameter_count"],
            "candidate_effective_batch": candidate["train"]["effective_batch"],
        },
        "activation": activation,
        "full_eval": full_eval,
        "official_eval": official,
        "same_board_all_256": same_board,
        "artifacts": {name: artifact(path) for name, path in artifact_paths.items()},
    }

    OUTPUT.mkdir(parents=True, exist_ok=False)
    json_path = OUTPUT / "comparison.json"
    json_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")

    official_rows = []
    for label in RANGES:
        for loop in LOOPS:
            c = official["control"][label][loop]
            n = official["candidate"][label][loop]
            official_rows.append(
                f"<tr><td>{label}</td><td>{loop[-1]}</td><td>{c['exact']:.6f}</td><td>{n['exact']:.6f}</td>"
                f"<td>{n['exact'] - c['exact']:+.6f}</td><td>{c['blank_acc']:.6f}</td>"
                f"<td>{n['blank_acc']:.6f}</td><td>{n['blank_acc'] - c['blank_acc']:+.6f}</td></tr>"
            )
    paired_rows = []
    for label in RANGES:
        stats = same_board[label]
        counts = stats["loop5_candidate_better_equal_worse"]
        late = stats["late_correction_candidate_stronger_equal_weaker"]
        paired_rows.append(
            f"<tr><td>{label}</td><td>{stats['shared_boards']}</td>"
            f"<td>{' / '.join(f'{v:.2f}' for v in stats['control_mean_wrong_cells_loops1_5'])}</td>"
            f"<td>{' / '.join(f'{v:.2f}' for v in stats['candidate_mean_wrong_cells_loops1_5'])}</td>"
            f"<td>{stats['control_loop3_to_5_correction']:.3f}</td>"
            f"<td>{stats['candidate_loop3_to_5_correction']:.3f}</td>"
            f"<td>{counts['better']} / {counts['equal']} / {counts['worse']}</td>"
            f"<td>{late['stronger']} / {late['equal']} / {late['weaker']}</td></tr>"
        )
    activation_rows = []
    for loop in LOOPS:
        values = activation[loop]
        activation_rows.append(
            f"<tr><td>{loop[-1]}</td><td>{values['gdn3_adaptive_signed_erase_enabled']:.6f}</td>"
            f"<td>{values['gdn3_adaptive_signed_erase_residual_abs']:.6f}</td>"
            f"<td>{values['gdn3_adaptive_signed_erase_residual_relative_rms']:.6f}</td>"
            f"<td>{values['gdn3_adaptive_signed_erase_residual_batch_std']:.6f}</td>"
            f"<td>{values['gdn3_adaptive_signed_erase_residual_token_std']:.6f}</td>"
            f"<td>{values['gdn3_adaptive_signed_erase_residual_head_std']:.6f}</td>"
            f"<td>{values['gdn3_adaptive_signed_erase_b_relative_change']:.6f}</td>"
            f"<td>{values['gdn3_adaptive_signed_erase_above_one_frac']:.6f}</td>"
            f"<td>{values['gdn3_adaptive_signed_erase_effective_min']:.6f} / {values['gdn3_adaptive_signed_erase_effective_max']:.6f}</td>"
            f"<td>{values['gdn3_adaptive_signed_erase_terminal_rms']:.6f}</td>"
            f"<td>{values['gdn3_adaptive_signed_erase_terminal_batch_std']:.6f}</td>"
            f"<td>{values['gdn3_adaptive_signed_erase_weight_rms']:.6f}</td></tr>"
        )
    case_sections = "".join(
        hardest_case_html(label, *case_payloads[label]) for label in RANGES
    )
    html_text = f"""<!doctype html>
<html><head><meta charset="utf-8"><title>P-GDN3-012 matched decision</title>
<style>
body{{font:15px system-ui,sans-serif;max-width:1500px;margin:28px auto;padding:0 20px;color:#17202a}}
h1,h2,h3{{letter-spacing:0}} table{{border-collapse:collapse;width:100%;margin:12px 0 28px}}
th,td{{border:1px solid #cbd5e1;padding:7px 9px;text-align:right}} th:first-child,td:first-child{{text-align:left}} th{{background:#eef2f7}}
.fail{{color:#a61b1b;font-weight:700}} code{{background:#f3f4f6;padding:2px 4px}} .boards{{display:flex;flex-wrap:wrap;gap:10px}}
figure{{margin:0}} figcaption{{font-size:12px;max-width:126px}} .board{{display:grid;grid-template-columns:repeat(9,14px);grid-template-rows:repeat(9,14px);border:2px solid #1f2937}}
.board span{{font:10px/14px ui-monospace,monospace;text-align:center;border-right:1px solid #d1d5db;border-bottom:1px solid #d1d5db}}
.board span:nth-child(3n){{border-right-color:#1f2937}} .board span:nth-child(n+19):nth-child(-n+27),.board span:nth-child(n+46):nth-child(-n+54){{border-bottom-color:#1f2937}}
.clue{{background:#e5e7eb;font-weight:700}} .wrong{{background:#fecaca;color:#991b1b}} .correct{{background:#dcfce7}} .case{{margin:30px 0 42px}}
</style></head><body>
<h1>P-GDN3-012: Adaptive Signed-Erase Spectrum</h1>
<p class="fail">Decision: discarded. The signed erase transition activates and stays bounded, but misses both preregistered exact-quality routes.</p>
<ul>
<li>Hard 51-64 macro loop5 exact: {hard_macro_control:.6f} to {hard_macro_candidate:.6f} ({hard_macro_candidate-hard_macro_control:+.6f}).</li>
<li>Mixed loop5 exact: {mixed_control:.6f} to {mixed_candidate:.6f} ({mixed_candidate-mixed_control:+.6f}).</li>
<li>Loop5 erase residual abs/relative RMS: {activation['loop5']['gdn3_adaptive_signed_erase_residual_abs']:.6f} / {activation['loop5']['gdn3_adaptive_signed_erase_residual_relative_rms']:.6f}; b&gt;1 fraction {activation['loop5']['gdn3_adaptive_signed_erase_above_one_frac']:.6f}.</li>
<li>100-step throughput: {effective_boards/control_sec:.3f} to {effective_boards/candidate_sec:.3f} effective boards/s; time overhead {time_overhead*100:.2f}%.</li>
<li>Peak allocated memory: {control_alloc:.1f} to {candidate_alloc:.1f} MiB ({memory_overhead*100:+.2f}%).</li>
</ul>
<h2>Adaptive signed-erase activation</h2><table><thead><tr><th>Loop</th><th>Enabled</th><th>Residual abs</th><th>Residual rel RMS</th><th>Board std</th><th>Token std</th><th>Head std</th><th>b rel change</th><th>b&gt;1 frac</th><th>Effective min/max</th><th>Terminal RMS</th><th>Terminal board std</th><th>Weight RMS</th></tr></thead><tbody>{''.join(activation_rows)}</tbody></table>
<h2>Official 512-board loop metrics</h2><table><thead><tr><th>Range</th><th>Loop</th><th>Control exact</th><th>Candidate exact</th><th>Delta exact</th><th>Control blank</th><th>Candidate blank</th><th>Delta blank</th></tr></thead><tbody>{''.join(official_rows)}</tbody></table>
<h2>Same-board case-bank dynamics</h2><p>All 256 case IDs and labels match per range.</p>
<table><thead><tr><th>Range</th><th>Boards</th><th>Control wrong loops1-5</th><th>Signed erase wrong loops1-5</th><th>Control L3-L5</th><th>Signed erase L3-L5</th><th>L5 better/equal/worse</th><th>Late stronger/equal/weaker</th></tr></thead><tbody>{''.join(paired_rows)}</tbody></table>
<h2>Hardest same-board loop1-5 views</h2>{case_sections}
<p>Machine-readable evidence: <code>comparison.json</code>. Source SHA <code>{SOURCE_SHA}</code>.</p>
</body></html>"""
    html_path = OUTPUT / "comparison.html"
    html_path.write_text(html_text)
    script_path = OUTPUT / "build_pgd3012_comparison.py"
    script_path.write_text(Path(__file__).read_text())
    manifest_path = OUTPUT / "manifest.sha256"
    manifest_path.write_text(
        "".join(
            f"{sha256(path)}  {path.name}\n"
            for path in (json_path, html_path, script_path)
        )
    )
    print(json.dumps({
        "output": str(OUTPUT),
        "hard_macro_delta": result["comparison"]["hard_macro_exact_delta"],
        "mixed_delta": result["comparison"]["mixed_exact_delta"],
        "time_overhead": time_overhead,
        "memory_overhead": memory_overhead,
        "status": "discarded",
    }, indent=2))


if __name__ == "__main__":
    main()
