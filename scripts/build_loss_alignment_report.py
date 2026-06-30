#!/usr/bin/env python3
from __future__ import annotations

import argparse
import html
import json
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


GDN_RUNS: List[Dict[str, str]] = [
    {
        "label": "D128/L6 no-FS 600",
        "path": "runs/gdn-realbwd-official-sudoku-scale-nofs-20260626T0055-be2815b",
        "group": "opening gate",
    },
    {
        "label": "D128/L6 FS 600",
        "path": "runs/gdn-realbwd-official-sudoku-scale-fs-clean-20260626T0105-be2815b",
        "group": "opening gate",
    },
    {
        "label": "D192/L10 no-FS 600",
        "path": "runs/gdn-d192-official-sudoku-nofs-reusefswt-s600-20260626T0855-b471906",
        "group": "opening gate",
    },
    {
        "label": "D192/L10 FS 600",
        "path": "runs/gdn-d192-official-sudoku-fs-clean-s600-20260626T0825-b471906",
        "group": "opening gate",
    },
    {
        "label": "D192/L10 FS 1500",
        "path": "runs/gdn-d192-official-sudoku-fs-long-s1500-20260626T0910Z-fce43c9",
        "group": "clean scale",
    },
    {
        "label": "D192/L10 FS 3000",
        "path": "runs/gdn-d192-official-sudoku-fs-clean-s3000-20260627T0512Z-e6fe7a0",
        "group": "clean scale",
    },
    {
        "label": "D256/L10 FS 1500",
        "path": "runs/gdn-d256-official-sudoku-fs-capacity-s1500-20260626T1515Z-6ea0b0c",
        "group": "capacity gate",
    },
    {
        "label": "D192/L10 loop-resid 1500",
        "path": "runs/gdn-d192-official-sudoku-fs-loopresid-s1500-20260627T0135Z-629b578",
        "group": "state tweak",
    },
    {
        "label": "D192/L10 hardcurr 4000",
        "path": "runs/gdn-d192-official-sudoku-fs-hardcurr-s4000-20260628T1435Z-d5658e0",
        "group": "data pressure",
    },
    {
        "label": "D192/L10 exact-margin 3000",
        "path": "runs/gdn-d192-official-sudoku-fs-exactmargin-s3000-20260629T0251Z-6dde82a",
        "group": "loss pressure",
    },
    {
        "label": "D192/L10 loop8 1500",
        "path": "runs/gdn-d192-official-sudoku-fs-loop8-dynamics-s1500-20260629T0416Z-83b6bff",
        "group": "loop-depth diagnostic",
    },
    {
        "label": "D192/L10 expand_v2 1500",
        "path": "runs/gdn-d192-official-sudoku-fs-expandv2-s1500-20260629T0455Z-e2375df",
        "group": "state capacity",
    },
    {
        "label": "D192/L10 feedback-attractor 1500",
        "path": "runs/gdn-d192-official-sudoku-fs-feedback-attractor-s1500-20260629T0708Z-4dfba71",
        "group": "feedback objective",
    },
    {
        "label": "D192/L10 latent-noise0.01 1500",
        "path": "runs/gdn-d192-official-sudoku-fs-latentnoise001-s1500-20260630T0315Z-19007fc",
        "group": "EqR-style noise",
    },
]


OFFICIAL_SCORE_FILES = {
    "official_eqr_sudoku": "runs/official-eqr-sudoku-repro-20260623/score.json",
    "official_eqr_maze": "runs/official-eqr-maze-repro-20260623/score.json",
    "official_mixer_e1024": "runs/official-eqr-mixer-replacement-triton-e1024-matched-eval-20260624T1015Z-ce2e76b/score.json",
    "official_native_rwkv_gate": "runs/rwkv_native_fs_sudoku_gate_20260624T121426Z_cb1b5a1-e64-d16b1/score.json",
}


LOG_ROW_RE = re.compile(
    r"step=(?P<step>\d+)\s+ce=(?P<ce>[-+0-9.eE]+)\s+total=(?P<total>[-+0-9.eE]+)"
    r"\s+loop1=(?P<loop1>[-+0-9.eE]+)\s+loop_last=(?P<loop_last>[-+0-9.eE]+)"
)
CKPT_ROW_RE = re.compile(
    r"\[checkpoint_eval step=(?P<step>\d+).*?\]\s+loop(?P<loop>\d+)\s+exact=(?P<exact>[-+0-9.eE]+)\s+blank=(?P<blank>[-+0-9.eE]+)"
)


def git(repo: Path, *args: str) -> str:
    return subprocess.check_output(["git", "-C", str(repo), *args], text=True).strip()


def read_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def find_result(run_dir: Path) -> Optional[Path]:
    candidates = sorted((run_dir / "output").glob("futureseed_loop_seed*.json"))
    if not candidates:
        candidates = sorted(run_dir.glob("**/futureseed_loop_seed*.json"))
    return candidates[-1] if candidates else None


def fnum(value: Any) -> Optional[float]:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(value)
    except Exception:
        return None


def fmt(value: Any, digits: int = 4) -> str:
    number = fnum(value)
    if number is None:
        return "-"
    return f"{number:.{digits}f}"


def fmt_int(value: Any) -> str:
    number = fnum(value)
    if number is None:
        return "-"
    return str(int(round(number)))


def sorted_loop_items(eval_payload: Dict[str, Any]) -> List[Tuple[int, Dict[str, Any]]]:
    out: List[Tuple[int, Dict[str, Any]]] = []
    for key, value in eval_payload.items():
        match = re.fullmatch(r"loop(\d+)", str(key))
        if match and isinstance(value, dict):
            out.append((int(match.group(1)), value))
    return sorted(out)


def parse_log(run_dir: Path) -> Tuple[List[Dict[str, float]], List[Dict[str, float]]]:
    log_path = run_dir / "logs" / "run.log"
    train_rows: List[Dict[str, float]] = []
    ckpt_rows: List[Dict[str, float]] = []
    if not log_path.exists():
        return train_rows, ckpt_rows
    with log_path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            row_match = LOG_ROW_RE.search(line)
            if row_match:
                train_rows.append({key: float(value) for key, value in row_match.groupdict().items()})
                continue
            ckpt_match = CKPT_ROW_RE.search(line)
            if ckpt_match:
                ckpt_rows.append({key: float(value) for key, value in ckpt_match.groupdict().items()})
    return train_rows, ckpt_rows


def extract_case_bank(run_dir: Path, metrics: Dict[str, Any]) -> Dict[str, Any]:
    case_files = sorted((run_dir / "output" / "case_bank").glob("**/cases.json"))
    if case_files:
        payload = read_json(case_files[0])
        summary = payload.get("summary") if isinstance(payload, dict) else None
        cases = payload.get("cases") if isinstance(payload, dict) else None
        return {"summary": summary or {}, "case_count": len(cases) if isinstance(cases, list) else None}

    case_bank = metrics.get("case_bank", {})
    if isinstance(case_bank, dict):
        holes = case_bank.get("holes", {})
        if isinstance(holes, dict):
            for item in holes.values():
                if isinstance(item, dict) and isinstance(item.get("summary"), dict):
                    return {"summary": item["summary"], "case_count": None}
    return {"summary": {}, "case_count": None}


def extract_gdn_row(repo: Path, spec: Dict[str, str]) -> Dict[str, Any]:
    run_dir = repo / spec["path"]
    result_path = find_result(run_dir)
    payload = read_json(result_path) if result_path is not None else {}
    score = read_json(run_dir / "score.json")
    config = read_json(run_dir / "config.json")
    args = payload.get("args", {}) if isinstance(payload.get("args"), dict) else {}
    metrics = payload.get("metrics", {}) if isinstance(payload.get("metrics"), dict) else {}
    task = metrics.get("task", {}) if isinstance(metrics.get("task"), dict) else {}
    train = metrics.get("train", {}) if isinstance(metrics.get("train"), dict) else {}
    eval_clean = metrics.get("eval_clean", {}) if isinstance(metrics.get("eval_clean"), dict) else {}
    eval_noisy = metrics.get("eval_noisy", {}) if isinstance(metrics.get("eval_noisy"), dict) else None
    loops = sorted_loop_items(eval_clean)
    loop1 = loops[0][1] if loops else {}
    final_loop_num, final_loop = loops[-1] if loops else (None, {})
    fs_metrics = eval_clean.get(f"loop{final_loop_num}/future_seed", {}) if final_loop_num is not None else {}
    train_curve, ckpt_rows = parse_log(run_dir)
    case_bank = extract_case_bank(run_dir, metrics)

    final_exact = fnum(final_loop.get("label_exact"))
    loop1_exact = fnum(loop1.get("label_exact"))
    final_blank = fnum(final_loop.get("blank_acc"))
    loop1_blank = fnum(loop1.get("blank_acc"))
    loop_rows = [
        {
            "loop": loop_num,
            "exact": fnum(values.get("label_exact")),
            "blank_acc": fnum(values.get("blank_acc")),
        }
        for loop_num, values in loops
    ]

    noisy_final = None
    if isinstance(eval_noisy, dict):
        noisy_loops = sorted_loop_items(eval_noisy)
        if noisy_loops:
            noisy_final = {
                "loop": noisy_loops[-1][0],
                "exact": fnum(noisy_loops[-1][1].get("label_exact")),
                "blank_acc": fnum(noisy_loops[-1][1].get("blank_acc")),
            }

    return {
        "label": spec["label"],
        "group": spec["group"],
        "run_dir": spec["path"],
        "result_json": str(result_path.relative_to(repo)) if result_path else None,
        "score_json": str((run_dir / "score.json").relative_to(repo)) if (run_dir / "score.json").exists() else None,
        "exists": run_dir.exists(),
        "git_sha": config.get("git_sha") or score.get("git_sha") or args.get("source_sha"),
        "backbone": train.get("backbone") or args.get("backbone"),
        "gdn_mode": train.get("gdn_mode") or args.get("gdn_mode"),
        "d_model": args.get("d_model"),
        "layers": args.get("layers"),
        "heads": args.get("heads"),
        "head_dim": args.get("head_dim"),
        "steps": args.get("steps") or score.get("steps_completed"),
        "future_seed_scale": task.get("future_seed_scale", args.get("future_seed_scale")),
        "noise_scale": args.get("noise_scale"),
        "noise_mode": task.get("noise_mode", train.get("noise_mode")),
        "loop_update_mode": train.get("loop_update_mode") or task.get("loop_update_mode"),
        "future_seed_update": train.get("future_seed_update") or task.get("future_seed_update"),
        "gdn_expand_v": task.get("gdn_expand_v", args.get("gdn_expand_v")),
        "loop_feedback_scale": task.get("loop_feedback_scale", train.get("loop_feedback_scale")),
        "train_ce": fnum(train.get("train_ce_loss")),
        "train_total": fnum(train.get("train_total_loss")),
        "train_loop1": fnum(train.get("train_loop1_loss")),
        "train_loop_last": fnum(train.get("train_loop_last_loss")),
        "train_loop_delta": None
        if fnum(train.get("train_loop1_loss")) is None or fnum(train.get("train_loop_last_loss")) is None
        else fnum(train.get("train_loop1_loss")) - fnum(train.get("train_loop_last_loss")),
        "train_sec": fnum(train.get("train_sec")),
        "cuda_mb": fnum(train.get("cuda_max_memory_allocated_mb")),
        "final_loop": final_loop_num,
        "loop1_exact": loop1_exact,
        "final_exact": final_exact,
        "exact_gain": None if loop1_exact is None or final_exact is None else final_exact - loop1_exact,
        "loop1_blank": loop1_blank,
        "final_blank": final_blank,
        "blank_gain": None if loop1_blank is None or final_blank is None else final_blank - loop1_blank,
        "fs_gate": fnum(fs_metrics.get("fs_gate_mean")),
        "fs_state_norm": fnum(fs_metrics.get("fs_state_norm")),
        "loop_rows": loop_rows,
        "train_curve": train_curve,
        "checkpoint_rows": ckpt_rows,
        "case_bank": case_bank,
        "noisy_final": noisy_final,
    }


def extract_official(repo: Path) -> Dict[str, Any]:
    official = {name: read_json(repo / rel_path) for name, rel_path in OFFICIAL_SCORE_FILES.items()}
    return official


def paired_delta(rows: List[Dict[str, Any]], left_label: str, right_label: str) -> Dict[str, Any]:
    by_label = {row["label"]: row for row in rows}
    left = by_label.get(left_label, {})
    right = by_label.get(right_label, {})
    return {
        "left": left_label,
        "right": right_label,
        "delta_train_ce": None
        if fnum(right.get("train_ce")) is None or fnum(left.get("train_ce")) is None
        else fnum(right.get("train_ce")) - fnum(left.get("train_ce")),
        "delta_final_exact": None
        if fnum(right.get("final_exact")) is None or fnum(left.get("final_exact")) is None
        else fnum(right.get("final_exact")) - fnum(left.get("final_exact")),
        "delta_final_blank": None
        if fnum(right.get("final_blank")) is None or fnum(left.get("final_blank")) is None
        else fnum(right.get("final_blank")) - fnum(left.get("final_blank")),
        "time_ratio_right_over_left": None
        if fnum(right.get("train_sec")) in (None, 0.0) or fnum(left.get("train_sec")) in (None, 0.0)
        else fnum(right.get("train_sec")) / fnum(left.get("train_sec")),
    }


def markdown_table(headers: List[str], rows: Iterable[List[str]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def build_markdown(repo: Path, payload: Dict[str, Any]) -> str:
    rows = payload["gdn_rows"]
    official = payload["official"]
    created = payload["created_at_utc"]
    sha = payload["source_sha"]

    sudoku = official.get("official_eqr_sudoku", {})
    sudoku_metrics = sudoku.get("metrics", {}) if isinstance(sudoku.get("metrics"), dict) else {}
    maze = official.get("official_eqr_maze", {})
    maze_rep = maze.get("reproduced", {}) if isinstance(maze.get("reproduced"), dict) else {}
    mixer = official.get("official_mixer_e1024", {})
    native_rwkv = official.get("official_native_rwkv_gate", {})

    clean1500 = next((row for row in rows if row["label"] == "D192/L10 FS 1500"), {})
    clean3000 = next((row for row in rows if row["label"] == "D192/L10 FS 3000"), {})
    latent = next((row for row in rows if row["label"] == "D192/L10 latent-noise0.01 1500"), {})
    loop8 = next((row for row in rows if row["label"] == "D192/L10 loop8 1500"), {})

    opening_d192 = paired_delta(rows, "D192/L10 no-FS 600", "D192/L10 FS 600")
    clean_long = paired_delta(rows, "D192/L10 FS 1500", "D192/L10 FS 3000")

    table_rows = []
    for row in rows:
        table_rows.append(
            [
                row["label"],
                row["group"],
                fmt_int(row.get("steps")),
                fmt(row.get("train_ce")),
                fmt(row.get("train_loop_delta")),
                fmt(row.get("loop1_exact")),
                fmt(row.get("final_exact")),
                fmt(row.get("exact_gain")),
                fmt(row.get("loop1_blank")),
                fmt(row.get("final_blank")),
                fmt(row.get("blank_gain")),
                fmt(row.get("train_sec"), 1),
            ]
        )

    loop_lines = []
    for row in [clean1500, clean3000, loop8, latent]:
        if not row:
            continue
        loop_curve = ", ".join(
            f"L{item['loop']} {fmt(item.get('exact'))}/{fmt(item.get('blank_acc'))}"
            for item in row.get("loop_rows", [])
        )
        loop_lines.append(f"- {row['label']}: exact/blank by loop = {loop_curve}")

    checkpoint_lines = []
    for row in [clean3000, latent]:
        for ckpt in row.get("checkpoint_rows", []):
            checkpoint_lines.append(
                f"- {row['label']} step{fmt_int(ckpt.get('step'))}: loop{fmt_int(ckpt.get('loop'))} "
                f"exact {fmt(ckpt.get('exact'))}, blank {fmt(ckpt.get('blank'))}"
            )

    lines: List[str] = []
    lines.append("# EqR vs FutureSeed Loss Alignment Report")
    lines.append("")
    lines.append(f"- generated_at_utc: `{created}`")
    lines.append(f"- source_sha: `{sha}`")
    lines.append("- scope: existing artifacts only; no GPU training was launched for this report")
    lines.append("")
    lines.append("## Question")
    lines.append("")
    lines.append(
        "We want to know whether the gap to EqR is a missing hand-crafted trick, a missing loss, "
        "or a more basic scale/state-dynamics issue. The report therefore aligns loss anatomy, "
        "training CE, loop-level behavior, noise robustness, and hard-case evidence."
    )
    lines.append("")
    lines.append("## Official EqR Facts")
    lines.append("")
    lines.append(
        "- Official EqR released checkpoints were reproduced before this report: Sudoku quick gate "
        f"top1 `{fmt(sudoku_metrics.get('convergence_top_k/cumulative_exact_acc_top1'))}`, "
        f"top4 exact `{fmt(sudoku_metrics.get('convergence_top_k/exact_accuracy'))}`, "
        f"majority exact `{fmt(sudoku_metrics.get('majority_vote/exact_accuracy'))}`."
    )
    d16 = maze_rep.get("D16_B1", {}) if isinstance(maze_rep.get("D16_B1"), dict) else {}
    d64 = maze_rep.get("D64_B1", {}) if isinstance(maze_rep.get("D64_B1"), dict) else {}
    d64b = maze_rep.get("D64_B128", {}) if isinstance(maze_rep.get("D64_B128"), dict) else {}
    lines.append(
        "- Maze released checkpoint reproduction: "
        f"D16/B1 exact `{fmt(d16.get('exact_accuracy'))}`, "
        f"D64/B1 exact `{fmt(d64.get('exact_accuracy'))}`, "
        f"D64/B128 convergence exact `{fmt(d64b.get('convergence_top1_exact_accuracy'))}`."
    )
    lines.append(
        "- EqR loss is not a solver: the source path uses token loss plus a halting classifier. "
        "The relevant local audit points are `repos/eqr/models/losses/loss_heads.py:114` "
        "for LM loss, `:115` for halt BCE, and `:124` for `lm_loss + 0.5 * q_halt_loss`."
    )
    lines.append(
        "- EqR's non-clean ingredient is latent recurrence noise: `repos/eqr/models/eqr.py:549-572` "
        "adds Gaussian or feature-diff noise inside recurrent state updates; default train config has "
        "`noise_scale: 0.01` in `repos/eqr/config/arch/eqr.yaml:27`."
    )
    lines.append(
        "- Important eval detail: Sudoku released quick eval uses `noise_scale=0.5`; Maze released "
        "checkpoint needs `noise_scale=0.01`. The Maze report showed `0.5` gives misleading token "
        "accuracy but near-zero exact."
    )
    lines.append("")
    lines.append("## GDN/FutureSeed Run Alignment")
    lines.append("")
    lines.append(
        markdown_table(
            [
                "run",
                "group",
                "steps",
                "train CE",
                "loop CE gain",
                "loop1 exact",
                "final exact",
                "exact gain",
                "loop1 blank",
                "final blank",
                "blank gain",
                "sec",
            ],
            table_rows,
        )
    )
    lines.append("")
    lines.append("## Direct Deltas")
    lines.append("")
    lines.append(
        f"- D192 no-FS -> FS at 600 steps: train CE delta `{fmt(opening_d192['delta_train_ce'])}`, "
        f"exact delta `{fmt(opening_d192['delta_final_exact'])}`, blank delta `{fmt(opening_d192['delta_final_blank'])}`, "
        f"time ratio `{fmt(opening_d192['time_ratio_right_over_left'], 3)}`. This is the cleanest evidence "
        "that FutureSeed opens the GDN backbone."
    )
    lines.append(
        f"- D192 FS 1500 -> 3000: train CE delta `{fmt(clean_long['delta_train_ce'])}`, "
        f"exact delta `{fmt(clean_long['delta_final_exact'])}`, blank delta `{fmt(clean_long['delta_final_blank'])}`. "
        "Soft cell accuracy keeps improving, full-board exact barely moves."
    )
    lines.append(
        f"- Latent-noise0.01 vs clean 1500: final exact `{fmt(latent.get('final_exact'))}` vs "
        f"`{fmt(clean1500.get('final_exact'))}`; final blank `{fmt(latent.get('final_blank'))}` vs "
        f"`{fmt(clean1500.get('final_blank'))}`. It is healthy but not a plateau breaker."
    )
    lines.append("")
    lines.append("## Loop Behavior")
    lines.append("")
    lines.extend(loop_lines)
    lines.append("")
    lines.append("Checkpoint probes:")
    lines.extend(checkpoint_lines)
    lines.append("")
    lines.append("## EqR-Codebase Replacement Boundary")
    lines.append("")
    base = mixer.get("base", {}) if isinstance(mixer.get("base"), dict) else {}
    futureseed = mixer.get("futureseed", {}) if isinstance(mixer.get("futureseed"), dict) else {}
    native_metrics = native_rwkv.get("metrics", {}) if isinstance(native_rwkv.get("metrics"), dict) else {}
    native_delta = native_rwkv.get("delta", {}) if isinstance(native_rwkv.get("delta"), dict) else {}
    lines.append(
        "- Official EqR mixer-replacement e1024 boundary: official mixer base reached "
        f"acc `{fmt(base.get('accuracy'))}`, exact `{fmt(base.get('exact_accuracy'))}`, LM loss `{fmt(base.get('lm_loss'))}`; "
        f"the archived scan/FutureSeed-mixer side branch reached acc `{fmt(futureseed.get('accuracy'))}`, "
        f"exact `{fmt(futureseed.get('exact_accuracy'))}`, LM loss `{fmt(futureseed.get('lm_loss'))}`. "
        "Lower residual there was a stable wrong attractor, not success."
    )
    rwkv_base = native_metrics.get("rwkv-mixer", {}) if isinstance(native_metrics.get("rwkv-mixer"), dict) else {}
    rwkv_fs = (
        native_metrics.get("rwkv-native-fs-mixer", {})
        if isinstance(native_metrics.get("rwkv-native-fs-mixer"), dict)
        else {}
    )
    lines.append(
        "- Corrected native RWKV FutureSeed inside official EqR codebase at the tiny gate was essentially tied: "
        f"no-FS acc `{fmt(rwkv_base.get('eval_accuracy'))}`, FS acc `{fmt(rwkv_fs.get('eval_accuracy'))}`, "
        f"FS-minus-noFS exact `{fmt(native_delta.get('fs_minus_nofs_exact_accuracy'))}`, "
        f"FS eval time ratio `{fmt(native_delta.get('fs_eval_time_over_nofs'), 3)}`."
    )
    lines.append("")
    lines.append("## Diagnosis")
    lines.append("")
    lines.append(
        "1. The missing piece is not an obvious EqR loss trick. EqR's core objective is still token CE plus halt BCE, "
        "with latent noise and a strong recurrent/noncausal architecture."
    )
    lines.append(
        "2. FutureSeed is real as an opening mechanism. In matched GDN official Sudoku, no-FS stays near random-ish "
        "blank accuracy while FS sharply lowers CE and reaches nonzero exact."
    )
    lines.append(
        "3. After opening, our failure is the exact-vs-blank conversion. CE and blank accuracy keep improving, "
        "but full-board exact stays around `0.03`; that means a few cells per board remain wrong and dominate exact."
    )
    lines.append(
        "4. Loop currently front-loads the work. Loop1->2/3 gives most of the gain; later loops usually copy the "
        "same operating point. More loop count alone is low ROI until state dynamics changes."
    )
    lines.append(
        "5. Noise is scale-sensitive but not the answer by itself. `0.03` was too strong; `0.01` trained normally "
        "and reproduced the same plateau."
    )
    lines.append("")
    lines.append("## Next Decision")
    lines.append("")
    lines.append(
        "Do not run another noise table, margin-loss table, feedback table, or same-config long run. The highest-ROI "
        "next experiment is a clean bigger-state/backbone scaling run only if it changes state capacity, not just "
        "token width, plus diagnostic eval CE by loop. If that still leaves exact flat while CE/blank improve, the "
        "core FutureSeed formulation needs a simpler recurrent state update that keeps useful uncertainty across "
        "loops instead of freezing after loop3."
    )
    lines.append("")
    lines.append("## Generated Files")
    lines.append("")
    lines.append("- JSON: `research/reports/loss_alignment_20260630.json`")
    lines.append("- HTML: `research/reports/loss_alignment_20260630.html`")
    return "\n".join(lines) + "\n"


def html_table(headers: List[str], rows: Iterable[List[str]]) -> str:
    out = ["<table>", "<thead><tr>"]
    for header in headers:
        out.append(f"<th>{html.escape(header)}</th>")
    out.append("</tr></thead><tbody>")
    for row in rows:
        out.append("<tr>")
        for cell in row:
            out.append(f"<td>{html.escape(cell)}</td>")
        out.append("</tr>")
    out.append("</tbody></table>")
    return "".join(out)


def build_html(markdown_text: str, payload: Dict[str, Any]) -> str:
    rows = payload["gdn_rows"]
    table_rows = [
        [
            row["label"],
            row["group"],
            fmt_int(row.get("steps")),
            fmt(row.get("train_ce")),
            fmt(row.get("loop1_exact")),
            fmt(row.get("final_exact")),
            fmt(row.get("exact_gain")),
            fmt(row.get("loop1_blank")),
            fmt(row.get("final_blank")),
            fmt(row.get("blank_gain")),
            fmt(row.get("train_sec"), 1),
        ]
        for row in rows
    ]
    loop_json = json.dumps(
        {
            row["label"]: row.get("loop_rows", [])
            for row in rows
            if row.get("loop_rows") and row["label"] in {"D192/L10 FS 1500", "D192/L10 FS 3000", "D192/L10 latent-noise0.01 1500"}
        },
        indent=2,
    )
    return f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>EqR vs FutureSeed Loss Alignment</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, Segoe UI, sans-serif; margin: 28px; color: #202124; }}
    h1, h2 {{ margin: 24px 0 8px; }}
    table {{ border-collapse: collapse; width: 100%; margin: 16px 0; font-size: 13px; }}
    th, td {{ border: 1px solid #d0d7de; padding: 6px 8px; text-align: right; }}
    th:first-child, td:first-child, th:nth-child(2), td:nth-child(2) {{ text-align: left; }}
    th {{ background: #f6f8fa; position: sticky; top: 0; }}
    pre {{ background: #f6f8fa; padding: 12px; overflow: auto; border: 1px solid #d0d7de; }}
    .callout {{ border-left: 4px solid #0969da; padding: 8px 12px; background: #f6f8fa; }}
  </style>
</head>
<body>
  <h1>EqR vs FutureSeed Loss Alignment</h1>
  <div class="callout">Existing artifacts only. No GPU training was launched for this diagnostic.</div>
  {html_table(["run","group","steps","train CE","loop1 exact","final exact","exact gain","loop1 blank","final blank","blank gain","sec"], table_rows)}
  <h2>Loop Curves JSON</h2>
  <pre>{html.escape(loop_json)}</pre>
  <h2>Full Markdown Report</h2>
  <pre>{html.escape(markdown_text)}</pre>
</body>
</html>
"""


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
        f.write("\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument("--out-prefix", type=Path, default=Path("research/reports/loss_alignment_20260630"))
    args = parser.parse_args()

    repo = args.repo.resolve()
    out_prefix = args.out_prefix
    if not out_prefix.is_absolute():
        out_prefix = repo / out_prefix

    rows = [extract_gdn_row(repo, spec) for spec in GDN_RUNS]
    payload = {
        "created_at_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "source_sha": git(repo, "rev-parse", "HEAD"),
        "source_dirty_in_tracked_scope": bool(
            git(
                repo,
                "status",
                "--short",
                "--",
                ".",
                ":(exclude)runs",
                ":(exclude).cache",
                ":(exclude).venv",
                ":(exclude)artifacts",
                ":(exclude)models",
            )
        ),
        "gdn_rows": rows,
        "paired_deltas": {
            "d128_opening": paired_delta(rows, "D128/L6 no-FS 600", "D128/L6 FS 600"),
            "d192_opening": paired_delta(rows, "D192/L10 no-FS 600", "D192/L10 FS 600"),
            "clean_1500_to_3000": paired_delta(rows, "D192/L10 FS 1500", "D192/L10 FS 3000"),
            "latent_noise_vs_clean_1500": paired_delta(
                rows, "D192/L10 FS 1500", "D192/L10 latent-noise0.01 1500"
            ),
        },
        "official": extract_official(repo),
    }
    markdown_text = build_markdown(repo, payload)
    html_text = build_html(markdown_text, payload)

    write_json(out_prefix.with_suffix(".json"), payload)
    out_prefix.with_suffix(".md").write_text(markdown_text, encoding="utf-8")
    out_prefix.with_suffix(".html").write_text(html_text, encoding="utf-8")
    print(out_prefix.with_suffix(".md"))
    print(out_prefix.with_suffix(".json"))
    print(out_prefix.with_suffix(".html"))


if __name__ == "__main__":
    main()
