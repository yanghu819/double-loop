#!/usr/bin/env python3
"""Compare matched Gain-Budget GDN2 runs on official Sudoku case banks."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import html
import json
import math
import re
from pathlib import Path
from typing import Any


RANGES = {
    "51-55": (51, 55),
    "56-60": (56, 60),
    "61-64": (61, 64),
}
LOOPS = tuple(range(1, 6))


class ComparisonError(RuntimeError):
    pass


def fail(message: str) -> None:
    raise ComparisonError(message)


def load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        fail(f"cannot read JSON {path}: {exc}")
    if not isinstance(value, dict):
        fail(f"expected a JSON object: {path}")
    return value


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def normalize_hash(value: Any, *, where: str) -> str:
    if not isinstance(value, str):
        fail(f"{where}: data_hash must be a string")
    normalized = value.lower().removeprefix("sha256:")
    if not re.fullmatch(r"[0-9a-f]{64}", normalized):
        fail(f"{where}: invalid SHA-256 data_hash {value!r}")
    return "sha256:" + normalized


def range_label(value: str) -> str:
    numbers = [int(item) for item in re.findall(r"\d+", value)]
    for label, expected in RANGES.items():
        if tuple(numbers[-2:]) == expected:
            return label
    fail(f"unknown blank range {value!r}; expected one of {sorted(RANGES)}")
    raise AssertionError


def parse_range_paths(values: list[str], flag: str) -> dict[str, Path]:
    parsed: dict[str, Path] = {}
    for value in values:
        raw_range, separator, raw_path = value.partition("=")
        if not separator or not raw_path:
            fail(f"{flag} expects RANGE=PATH, got {value!r}")
        label = range_label(raw_range)
        if label in parsed:
            fail(f"{flag}: duplicate range {label}")
        parsed[label] = Path(raw_path).expanduser().resolve()
    if parsed and set(parsed) != set(RANGES):
        missing = sorted(set(RANGES) - set(parsed))
        fail(f"{flag}: all three ranges are required; missing {missing}")
    return parsed


def metrics_payload(result: dict[str, Any], path: Path) -> dict[str, Any]:
    metrics = result.get("metrics")
    if not isinstance(metrics, dict):
        fail(f"{path}: missing object metrics")
    return metrics


def matching_range_row(
    mapping: Any, label: str, *, where: str
) -> dict[str, Any]:
    if not isinstance(mapping, dict):
        fail(f"{where}: expected a range mapping")
    matches = []
    for key, row in mapping.items():
        try:
            matched = range_label(str(key)) == label
        except ComparisonError:
            matched = False
        if matched:
            matches.append(row)
    if len(matches) != 1 or not isinstance(matches[0], dict):
        fail(f"{where}: expected exactly one row for range {label}")
    return matches[0]


def resolve_artifact_path(raw: Any, result_path: Path) -> Path:
    if not isinstance(raw, str) or not raw:
        fail(f"{result_path}: invalid all_cases_json path {raw!r}")
    candidate = Path(raw).expanduser()
    attempts = [candidate]
    if not candidate.is_absolute():
        attempts.append(result_path.parent / candidate)
    for attempt in attempts:
        if attempt.is_file():
            return attempt.resolve()
    fail(
        f"{result_path}: all_cases artifact does not exist: {raw!r}; "
        "pass explicit --control-all-cases/--candidate-all-cases overrides"
    )
    raise AssertionError


def infer_case_paths(
    result: dict[str, Any], result_path: Path
) -> dict[str, Path]:
    metrics = metrics_payload(result, result_path)
    case_bank = metrics.get("case_bank")
    if not isinstance(case_bank, dict):
        fail(
            f"{result_path}: metrics.case_bank missing; pass explicit all_cases paths"
        )
    holes = case_bank.get("holes")
    paths: dict[str, Path] = {}
    for label in RANGES:
        row = matching_range_row(
            holes, label, where=f"{result_path}: metrics.case_bank.holes"
        )
        paths[label] = resolve_artifact_path(
            row.get("all_cases_json"), result_path
        )
    return paths


def require_list(
    value: Any, *, length: int, where: str
) -> list[Any]:
    if not isinstance(value, list) or len(value) != length:
        fail(f"{where}: expected list of length {length}")
    return value


def require_int(value: Any, *, where: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        fail(f"{where}: expected integer")
    return value


def require_number(value: Any, *, where: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        fail(f"{where}: expected finite number")
    number = float(value)
    if not math.isfinite(number):
        fail(f"{where}: expected finite number")
    return number


def normalize_case(
    raw: Any, *, artifact: Path, label: str, index: int
) -> dict[str, Any]:
    where = f"{artifact}: cases[{index}]"
    if not isinstance(raw, dict):
        fail(f"{where}: expected object")
    label_tokens = [
        require_int(value, where=f"{where}.label_tokens[{cell}]")
        for cell, value in enumerate(
            require_list(raw.get("label_tokens"), length=81, where=where)
        )
    ]
    if any(value < 0 or value > 8 for value in label_tokens):
        fail(f"{where}: label_tokens must be in 0..8")
    input_tokens = [
        require_int(value, where=f"{where}.input[{cell}]")
        for cell, value in enumerate(
            require_list(raw.get("input"), length=81, where=where)
        )
    ]
    clue_mask_raw = require_list(
        raw.get("clue_mask"), length=81, where=f"{where}.clue_mask"
    )
    clue_mask: list[bool] = []
    for cell, value in enumerate(clue_mask_raw):
        if value not in (0, 1, False, True):
            fail(f"{where}.clue_mask[{cell}]: expected bool or 0/1")
        clue_mask.append(bool(value))
    blank_count = sum(not value for value in clue_mask)
    declared_blank_count = require_int(
        raw.get("blank_count"), where=f"{where}.blank_count"
    )
    if blank_count != declared_blank_count:
        fail(
            f"{where}: blank_count={declared_blank_count}, "
            f"but clue_mask implies {blank_count}"
        )
    lo, hi = RANGES[label]
    if not lo <= blank_count <= hi:
        fail(f"{where}: {blank_count} blanks are outside range {label}")
    for cell, is_clue in enumerate(clue_mask):
        if is_clue and input_tokens[cell] != label_tokens[cell]:
            fail(f"{where}: clue cell {cell} disagrees with label")

    content_hash = normalize_hash(
        raw.get("content_sha256"), where=f"{where}.content_sha256"
    )
    loops_raw = raw.get("loops")
    if not isinstance(loops_raw, dict):
        fail(f"{where}.loops: expected object")
    loops: dict[str, dict[str, Any]] = {}
    for loop in LOOPS:
        loop_key = f"loop{loop}"
        row = loops_raw.get(loop_key)
        if not isinstance(row, dict):
            fail(f"{where}.loops.{loop_key}: expected object")
        prediction = [
            require_int(
                value,
                where=(
                    f"{where}.loops.{loop_key}.prediction_tokens[{cell}]"
                ),
            )
            for cell, value in enumerate(
                require_list(
                    row.get("prediction_tokens"),
                    length=81,
                    where=f"{where}.loops.{loop_key}.prediction_tokens",
                )
            )
        ]
        if any(value < 0 or value > 8 for value in prediction):
            fail(f"{where}.loops.{loop_key}: predictions must be in 0..8")
        wrong_total = sum(
            predicted != target
            for predicted, target in zip(prediction, label_tokens)
        )
        wrong_blank = sum(
            predicted != label_tokens[cell]
            for cell, predicted in enumerate(prediction)
            if not clue_mask[cell]
        )
        exact = wrong_total == 0
        blank_acc = 1.0 - wrong_blank / blank_count
        declared_exact = row.get("label_exact")
        if not isinstance(declared_exact, bool) or declared_exact != exact:
            fail(f"{where}.loops.{loop_key}: label_exact is inconsistent")
        if require_int(
            row.get("wrong_count"),
            where=f"{where}.loops.{loop_key}.wrong_count",
        ) != wrong_total:
            fail(f"{where}.loops.{loop_key}: wrong_count is inconsistent")
        if require_int(
            row.get("wrong_blank_count"),
            where=f"{where}.loops.{loop_key}.wrong_blank_count",
        ) != wrong_blank:
            fail(
                f"{where}.loops.{loop_key}: wrong_blank_count is inconsistent"
            )
        if require_int(
            row.get("wrong_clue_count"),
            where=f"{where}.loops.{loop_key}.wrong_clue_count",
        ) != wrong_total - wrong_blank:
            fail(
                f"{where}.loops.{loop_key}: wrong_clue_count is inconsistent"
            )
        if not math.isclose(
            require_number(
                row.get("blank_acc"),
                where=f"{where}.loops.{loop_key}.blank_acc",
            ),
            blank_acc,
            rel_tol=0.0,
            abs_tol=1e-9,
        ):
            fail(f"{where}.loops.{loop_key}: blank_acc is inconsistent")
        loops[loop_key] = {
            "prediction_tokens": prediction,
            "label_exact": exact,
            "blank_acc": blank_acc,
            "wrong_blank_count": wrong_blank,
            "wrong_total_count": wrong_total,
        }

    identity = {
        "input": input_tokens,
        "label_tokens": label_tokens,
        "clue_mask": clue_mask,
    }
    identity_sha256 = canonical_sha256(identity)
    if content_hash != identity_sha256:
        fail(
            f"{where}: content_sha256 does not match input/label/clue content"
        )
    case_id = raw.get("case_id")
    if not isinstance(case_id, str) or not case_id:
        fail(f"{where}.case_id: expected non-empty string")
    batch_index = require_int(
        raw.get("batch_index"), where=f"{where}.batch_index"
    )
    return {
        "case_id": case_id,
        "batch_index": batch_index,
        "content_sha256": content_hash,
        "identity_sha256": identity_sha256,
        "blank_count": blank_count,
        "input_tokens": input_tokens,
        "label_tokens": label_tokens,
        "clue_mask": clue_mask,
        "loops": loops,
    }


def load_case_artifact(path: Path, label: str) -> dict[str, Any]:
    payload = load_json(path)
    if payload.get("schema_version") != "official_sudoku_all_cases.v1":
        fail(f"{path}: unsupported schema_version {payload.get('schema_version')!r}")
    data_hash = normalize_hash(payload.get("data_hash"), where=f"{path}.data_hash")
    cases_raw = payload.get("cases")
    if not isinstance(cases_raw, list) or not cases_raw:
        fail(f"{path}: cases must be a non-empty list")
    cases = [
        normalize_case(raw, artifact=path, label=label, index=index)
        for index, raw in enumerate(cases_raw)
    ]
    ordered_digest = hashlib.sha256()
    for case in cases:
        ordered_digest.update(
            bytes.fromhex(case["content_sha256"].removeprefix("sha256:"))
        )
    recomputed_data_hash = "sha256:" + ordered_digest.hexdigest()
    if data_hash != recomputed_data_hash:
        fail(
            f"{path}: data_hash does not match the ordered per-case hashes"
        )
    eval_row = payload.get("eval")
    if not isinstance(eval_row, dict):
        fail(f"{path}: eval must be an object")
    if require_int(eval_row.get("actual_n"), where=f"{path}.eval.actual_n") != len(
        cases
    ):
        fail(f"{path}: eval.actual_n does not match cases length")
    declared_range = eval_row.get("range")
    if not isinstance(declared_range, dict):
        fail(f"{path}: eval.range must be an object")
    if (
        require_int(
            declared_range.get("holes_min"),
            where=f"{path}.eval.range.holes_min",
        ),
        require_int(
            declared_range.get("holes_max"),
            where=f"{path}.eval.range.holes_max",
        ),
    ) != RANGES[label]:
        fail(f"{path}: eval.range does not match CLI range {label}")
    return {
        "path": path,
        "file_sha256": file_sha256(path),
        "data_hash": data_hash,
        "cases": cases,
    }


def pair_cases(
    control: dict[str, Any],
    candidate: dict[str, Any],
    *,
    label: str,
) -> list[dict[str, Any]]:
    if control["data_hash"] != candidate["data_hash"]:
        fail(
            f"range {label}: control/candidate data_hash mismatch: "
            f"{control['data_hash']} != {candidate['data_hash']}"
        )
    control_cases = control["cases"]
    candidate_cases = candidate["cases"]
    if len(control_cases) != len(candidate_cases):
        fail(f"range {label}: control/candidate case counts differ")
    paired = []
    for index, (control_case, candidate_case) in enumerate(
        zip(control_cases, candidate_cases)
    ):
        if control_case["content_sha256"] != candidate_case["content_sha256"]:
            fail(f"range {label}: ordered case hash mismatch at index {index}")
        if control_case["identity_sha256"] != candidate_case["identity_sha256"]:
            fail(f"range {label}: puzzle content mismatch at index {index}")
        if control_case["case_id"] != candidate_case["case_id"]:
            fail(f"range {label}: case_id mismatch at index {index}")
        if control_case["batch_index"] != candidate_case["batch_index"]:
            fail(f"range {label}: batch_index mismatch at index {index}")
        paired.append(
            {
                "range": label,
                "case_id": control_case["case_id"],
                "content_sha256": control_case["content_sha256"],
                "blank_count": control_case["blank_count"],
                "input_tokens": control_case["input_tokens"],
                "label_tokens": control_case["label_tokens"],
                "clue_mask": control_case["clue_mask"],
                "control": control_case["loops"],
                "candidate": candidate_case["loops"],
            }
        )
    return paired


def summarize_cases(
    cases: list[dict[str, Any]], arm: str
) -> dict[str, Any]:
    total_blanks = sum(case["blank_count"] for case in cases)
    loops: dict[str, Any] = {}
    for loop in LOOPS:
        loop_key = f"loop{loop}"
        exact_count = sum(case[arm][loop_key]["label_exact"] for case in cases)
        wrong_blank = sum(
            case[arm][loop_key]["wrong_blank_count"] for case in cases
        )
        wrong_total = sum(
            case[arm][loop_key]["wrong_total_count"] for case in cases
        )
        loops[loop_key] = {
            "label_exact": exact_count / len(cases),
            "blank_acc": 1.0 - wrong_blank / total_blanks,
            "wrong_blank_total": wrong_blank,
            "wrong_blank_mean": wrong_blank / len(cases),
            "wrong_total": wrong_total,
            "wrong_total_mean": wrong_total / len(cases),
        }
    return {
        "eval_n": len(cases),
        "total_blanks": total_blanks,
        "loops": loops,
        "loop5_exact": loops["loop5"]["label_exact"],
        "loop1_to_loop5_exact_gain": (
            loops["loop5"]["label_exact"]
            - loops["loop1"]["label_exact"]
        ),
        "loop5_blank_acc": loops["loop5"]["blank_acc"],
        "loop1_to_loop5_wrong_blank_reduction": (
            loops["loop1"]["wrong_blank_total"]
            - loops["loop5"]["wrong_blank_total"]
        ),
        "loop1_to_loop5_transitions": {
            "solved_by_loop": sum(
                not case[arm]["loop1"]["label_exact"]
                and case[arm]["loop5"]["label_exact"]
                for case in cases
            ),
            "lost_by_loop": sum(
                case[arm]["loop1"]["label_exact"]
                and not case[arm]["loop5"]["label_exact"]
                for case in cases
            ),
            "stayed_exact": sum(
                case[arm]["loop1"]["label_exact"]
                and case[arm]["loop5"]["label_exact"]
                for case in cases
            ),
            "stayed_inexact": sum(
                not case[arm]["loop1"]["label_exact"]
                and not case[arm]["loop5"]["label_exact"]
                for case in cases
            ),
        },
    }


def paired_outcomes(cases: list[dict[str, Any]]) -> dict[str, int]:
    return {
        "candidate_only_exact_at_loop5": sum(
            case["candidate"]["loop5"]["label_exact"]
            and not case["control"]["loop5"]["label_exact"]
            for case in cases
        ),
        "control_only_exact_at_loop5": sum(
            case["control"]["loop5"]["label_exact"]
            and not case["candidate"]["loop5"]["label_exact"]
            for case in cases
        ),
        "both_exact_at_loop5": sum(
            case["control"]["loop5"]["label_exact"]
            and case["candidate"]["loop5"]["label_exact"]
            for case in cases
        ),
        "neither_exact_at_loop5": sum(
            not case["control"]["loop5"]["label_exact"]
            and not case["candidate"]["loop5"]["label_exact"]
            for case in cases
        ),
    }


def official_range(
    result: dict[str, Any],
    result_path: Path,
    label: str,
) -> dict[str, Any]:
    metrics = metrics_payload(result, result_path)
    row = matching_range_row(
        metrics.get("official_eval_by_blank_range"),
        label,
        where=f"{result_path}: metrics.official_eval_by_blank_range",
    )
    blank_range = row.get("blank_range")
    if (
        not isinstance(blank_range, list)
        or len(blank_range) != 2
        or tuple(blank_range) != RANGES[label]
    ):
        fail(
            f"{result_path}: range {label} has invalid blank_range "
            f"{blank_range!r}"
        )
    eval_clean = row.get("eval_clean")
    if not isinstance(eval_clean, dict):
        fail(f"{result_path}: range {label} has no eval_clean object")
    loops: dict[str, dict[str, float]] = {}
    for loop in LOOPS:
        key = f"loop{loop}"
        loop_row = eval_clean.get(key)
        if not isinstance(loop_row, dict):
            fail(f"{result_path}: range {label} missing eval_clean.{key}")
        loops[key] = {
            "label_exact": require_number(
                loop_row.get("label_exact"),
                where=f"{result_path}: {label}.{key}.label_exact",
            ),
            "blank_acc": require_number(
                loop_row.get("blank_acc"),
                where=f"{result_path}: {label}.{key}.blank_acc",
            ),
        }
        if not 0.0 <= loops[key]["label_exact"] <= 1.0:
            fail(f"{result_path}: {label}.{key}.label_exact outside [0, 1]")
        if not 0.0 <= loops[key]["blank_acc"] <= 1.0:
            fail(f"{result_path}: {label}.{key}.blank_acc outside [0, 1]")
    diagnostics_row = eval_clean.get("loop5/future_seed", {})
    diagnostics = {
        key: value
        for key, value in diagnostics_row.items()
        if "gain_budget" in key and isinstance(value, (bool, int, float, str))
    } if isinstance(diagnostics_row, dict) else {}
    eval_n = require_int(
        row.get("eval_n"), where=f"{result_path}: {label}.eval_n"
    )
    if eval_n <= 0:
        fail(f"{result_path}: {label}.eval_n must be positive")
    return {
        "eval_n": eval_n,
        "loops": loops,
        "gain_diagnostics": diagnostics,
    }


def summarize_official(row: dict[str, Any]) -> dict[str, Any]:
    loops = row["loops"]
    return {
        "eval_n": row["eval_n"],
        "loops": loops,
        "loop5_exact": loops["loop5"]["label_exact"],
        "loop1_to_loop5_exact_gain": (
            loops["loop5"]["label_exact"]
            - loops["loop1"]["label_exact"]
        ),
        "loop5_blank_acc": loops["loop5"]["blank_acc"],
    }


def validate_matched_args(
    control: dict[str, Any],
    candidate: dict[str, Any],
    *,
    control_path: Path,
    candidate_path: Path,
) -> dict[str, Any]:
    control_args = control.get("args")
    candidate_args = candidate.get("args")
    if not isinstance(control_args, dict) or not isinstance(candidate_args, dict):
        fail("both result files must contain args objects")
    allowed_differences = {
        "gdn2_gain_budget_mode",
        "out_dir",
        "train_checkpoint_dir",
    }
    keys = set(control_args) | set(candidate_args)
    mismatches = {}
    for key in sorted(keys - allowed_differences):
        if control_args.get(key) != candidate_args.get(key):
            mismatches[key] = {
                "control": control_args.get(key),
                "candidate": candidate_args.get(key),
            }
    if mismatches:
        fail(f"matched result args differ: {mismatches}")
    if control_args.get("gdn2_gain_budget_mode") != "external_identity":
        fail(
            f"{control_path}: control mode is not external_identity"
        )
    if candidate_args.get("gdn2_gain_budget_mode") != "decay_funded":
        fail(
            f"{candidate_path}: candidate mode is not decay_funded"
        )
    return {
        "allowed_differences": sorted(allowed_differences),
        "compared_field_count": len(keys - allowed_differences),
        "mismatches": {},
    }


def train_summary(
    result: dict[str, Any], result_path: Path
) -> dict[str, Any]:
    train = metrics_payload(result, result_path).get("train")
    if not isinstance(train, dict):
        fail(f"{result_path}: metrics.train missing")
    summary = {
        "train_sec": require_number(
            train.get("train_sec"), where=f"{result_path}: train.train_sec"
        ),
        "cuda_max_memory_allocated_mb": require_number(
            train.get("cuda_max_memory_allocated_mb"),
            where=f"{result_path}: train.cuda_max_memory_allocated_mb",
        ),
        "cuda_max_memory_reserved_mb": require_number(
            train.get("cuda_max_memory_reserved_mb"),
            where=f"{result_path}: train.cuda_max_memory_reserved_mb",
        ),
        "gain_diagnostics": {
            key: value
            for key, value in train.items()
            if "gain_budget" in key
            and isinstance(value, (bool, int, float, str))
        },
    }
    if summary["train_sec"] <= 0:
        fail(f"{result_path}: train_sec must be positive")
    return summary


def numeric_delta(candidate: Any, control: Any) -> float | None:
    if (
        isinstance(candidate, (int, float))
        and not isinstance(candidate, bool)
        and isinstance(control, (int, float))
        and not isinstance(control, bool)
    ):
        return float(candidate) - float(control)
    return None


def diagnostics_comparison(
    control: dict[str, Any], candidate: dict[str, Any]
) -> dict[str, Any]:
    keys = sorted(set(control) | set(candidate))
    return {
        key: {
            "control": control.get(key),
            "candidate": candidate.get(key),
            "delta": numeric_delta(candidate.get(key), control.get(key)),
        }
        for key in keys
    }


def mean(values: list[float]) -> float:
    return sum(values) / len(values)


def aggregate_ranges(
    ranges: dict[str, dict[str, Any]], arm: str
) -> dict[str, Any]:
    rows = [ranges[label][arm] for label in RANGES]
    total_cases = sum(row["eval_n"] for row in rows)
    total_blanks = sum(row["total_blanks"] for row in rows)
    loop1_wrong = sum(
        row["loops"]["loop1"]["wrong_blank_total"] for row in rows
    )
    loop5_wrong = sum(
        row["loops"]["loop5"]["wrong_blank_total"] for row in rows
    )
    return {
        "paired_mean_loop5_exact": mean(
            [row["loop5_exact"] for row in rows]
        ),
        "paired_mean_loop1_to_loop5_exact_gain": mean(
            [row["loop1_to_loop5_exact_gain"] for row in rows]
        ),
        "paired_mean_loop5_blank_acc": mean(
            [row["loop5_blank_acc"] for row in rows]
        ),
        "pooled_eval_n": total_cases,
        "pooled_total_blanks": total_blanks,
        "pooled_loop5_wrong_blank_total": loop5_wrong,
        "pooled_loop5_wrong_blank_mean": loop5_wrong / total_cases,
        "pooled_loop1_to_loop5_wrong_blank_reduction": (
            loop1_wrong - loop5_wrong
        ),
    }


def aggregate_official_ranges(
    ranges: dict[str, dict[str, Any]],
    arm: str,
) -> dict[str, Any]:
    rows = [ranges[label][arm] for label in RANGES]
    return {
        "official_mean_loop5_exact": mean(
            [row["loop5_exact"] for row in rows]
        ),
        "official_mean_loop1_to_loop5_exact_gain": mean(
            [row["loop1_to_loop5_exact_gain"] for row in rows]
        ),
        "official_mean_loop5_blank_acc": mean(
            [row["loop5_blank_acc"] for row in rows]
        ),
        "official_eval_n_per_range": {
            label: ranges[label][arm]["eval_n"] for label in RANGES
        },
        "official_total_evaluations": sum(row["eval_n"] for row in rows),
    }


def metric_deltas(
    control: dict[str, Any], candidate: dict[str, Any]
) -> dict[str, Any]:
    return {
        key: numeric_delta(candidate.get(key), value)
        for key, value in control.items()
        if numeric_delta(candidate.get(key), value) is not None
    }


def select_cases(cases: list[dict[str, Any]]) -> list[dict[str, Any]]:
    def wrong(case: dict[str, Any], arm: str) -> int:
        return case[arm]["loop5"]["wrong_blank_count"]

    hardest = sorted(
        cases,
        key=lambda case: (
            -wrong(case, "control"),
            -case["blank_count"],
            case["content_sha256"],
        ),
    )[0]
    improved = sorted(
        cases,
        key=lambda case: (
            -(wrong(case, "control") - wrong(case, "candidate")),
            -wrong(case, "control"),
            -case["blank_count"],
            case["content_sha256"],
        ),
    )[0]
    regressed = sorted(
        cases,
        key=lambda case: (
            -(wrong(case, "candidate") - wrong(case, "control")),
            -wrong(case, "candidate"),
            -case["blank_count"],
            case["content_sha256"],
        ),
    )[0]
    return [
        case_for_output(
            "control_hardest",
            "Largest control loop5 wrong-cell count",
            hardest,
        ),
        case_for_output(
            "candidate_max_improvement",
            "Largest control-minus-candidate loop5 wrong-cell count",
            improved,
        ),
        case_for_output(
            "candidate_max_regression",
            "Largest candidate-minus-control loop5 wrong-cell count",
            regressed,
        ),
    ]


def case_for_output(
    category: str, rule: str, case: dict[str, Any]
) -> dict[str, Any]:
    return {
        "category": category,
        "selection_rule": rule,
        "range": case["range"],
        "case_id": case["case_id"],
        "content_sha256": case["content_sha256"],
        "blank_count": case["blank_count"],
        "puzzle": [
            case["label_tokens"][cell] + 1 if is_clue else 0
            for cell, is_clue in enumerate(case["clue_mask"])
        ],
        "target": [value + 1 for value in case["label_tokens"]],
        "clue_mask": case["clue_mask"],
        "control": {
            key: {
                **value,
                "prediction": [
                    token + 1 for token in value["prediction_tokens"]
                ],
            }
            for key, value in case["control"].items()
        },
        "candidate": {
            key: {
                **value,
                "prediction": [
                    token + 1 for token in value["prediction_tokens"]
                ],
            }
            for key, value in case["candidate"].items()
        },
        "loop5_wrong_delta_candidate_minus_control": (
            case["candidate"]["loop5"]["wrong_blank_count"]
            - case["control"]["loop5"]["wrong_blank_count"]
        ),
    }


def ratio_delta(candidate: float, control: float) -> float:
    return candidate / control - 1.0


def required_diagnostic(
    diagnostics: dict[str, Any],
    key: str,
    *,
    where: str,
) -> float:
    if key not in diagnostics:
        fail(f"{where}: missing required diagnostic {key}")
    return require_number(diagnostics[key], where=f"{where}.{key}")


def build_decision(
    official_ranges: dict[str, dict[str, Any]],
    aggregate: dict[str, dict[str, Any]],
    system: dict[str, Any],
    candidate_train: dict[str, Any],
) -> dict[str, Any]:
    range_deltas = {
        label: official_ranges[label]["delta"]["loop5_exact"]
        for label in RANGES
    }
    mean_exact_delta = aggregate["delta"]["official_mean_loop5_exact"]
    mean_loop_gain_delta = aggregate["delta"][
        "official_mean_loop1_to_loop5_exact_gain"
    ]
    score_gate_a = (
        mean_exact_delta >= 0.015
        and min(range_deltas.values()) >= -0.01
        and mean_loop_gain_delta >= 0.0
    )
    score_gate_b = (
        max(range_deltas["56-60"], range_deltas["61-64"]) >= 0.02
        and range_deltas["51-55"] >= -0.01
        and mean_loop_gain_delta >= 0.005
    )

    relative = system["relative_delta"]
    systems_gate = all(
        relative[key] <= 0.20
        for key in (
            "train_sec",
            "cuda_max_memory_allocated_mb",
            "cuda_max_memory_reserved_mb",
        )
    )
    diagnostics = candidate_train["gain_diagnostics"]
    clipped = required_diagnostic(
        diagnostics,
        "gain_budget_clipped_frac",
        where="candidate train diagnostics",
    )
    infeasible = required_diagnostic(
        diagnostics,
        "gain_budget_infeasible_frac",
        where="candidate train diagnostics",
    )
    step_bound = required_diagnostic(
        diagnostics,
        "gain_budget_step_bound_max",
        where="candidate train diagnostics",
    )
    delta_error = required_diagnostic(
        diagnostics,
        "gain_budget_delta_error_max",
        where="candidate train diagnostics",
    )
    mechanism_gate = (
        clipped >= 0.01
        and infeasible == 0.0
        and step_bound <= 1.001
        and delta_error <= 3e-3
    )
    return {
        "passed": bool(
            (score_gate_a or score_gate_b) and systems_gate and mechanism_gate
        ),
        "score_gate_passed": bool(score_gate_a or score_gate_b),
        "score_gate_a": {
            "passed": bool(score_gate_a),
            "mean_loop5_exact_delta": mean_exact_delta,
            "minimum_range_loop5_exact_delta": min(range_deltas.values()),
            "mean_loop_gain_delta": mean_loop_gain_delta,
        },
        "score_gate_b": {
            "passed": bool(score_gate_b),
            "harder_range_best_loop5_exact_delta": max(
                range_deltas["56-60"], range_deltas["61-64"]
            ),
            "range_51_55_loop5_exact_delta": range_deltas["51-55"],
            "mean_loop_gain_delta": mean_loop_gain_delta,
        },
        "range_loop5_exact_deltas": range_deltas,
        "systems_gate": {
            "passed": bool(systems_gate),
            "maximum_allowed_relative_overhead": 0.20,
            "relative_deltas": relative,
        },
        "mechanism_gate": {
            "passed": bool(mechanism_gate),
            "clipped_frac": clipped,
            "minimum_clipped_frac": 0.01,
            "infeasible_frac": infeasible,
            "maximum_infeasible_frac": 0.0,
            "step_bound_max": step_bound,
            "maximum_step_bound": 1.001,
            "delta_error_max": delta_error,
            "maximum_delta_error": 3e-3,
            "precision_contract": (
                "official BF16 chunk tolerance; strict FP32 certificate "
                "is enforced separately by formal preflight"
            ),
        },
    }


def result_name(result: dict[str, Any], path: Path) -> str:
    args = result.get("args")
    if isinstance(args, dict) and args.get("run_name"):
        return str(args["run_name"])
    if result.get("run_name"):
        return str(result["run_name"])
    if path.parent.name == "output":
        return path.parent.parent.name
    return path.stem


def source_sha(result: dict[str, Any]) -> str | None:
    for container in (result, result.get("args", {})):
        if isinstance(container, dict):
            for key in ("git_sha", "source_sha"):
                value = container.get(key)
                if isinstance(value, str) and value:
                    return value
    return None


def run_provenance(result: dict[str, Any], result_path: Path) -> dict[str, Any]:
    run_dir = (
        result_path.parent.parent
        if result_path.parent.name == "output"
        else result_path.parent
    )
    source_head_path = run_dir / "source_HEAD.txt"
    config_path = run_dir / "config.json"
    patch_path = run_dir / "source.patch"
    if not source_head_path.is_file() or not config_path.is_file():
        fail(f"{run_dir}: missing source_HEAD.txt or config.json")
    config = load_json(config_path)
    head = source_head_path.read_text(encoding="utf-8").strip()
    embedded = source_sha(result)
    if embedded is not None and embedded != head:
        fail(f"{result_path}: embedded source SHA disagrees with source_HEAD.txt")
    if config.get("git_sha") != head or config.get("git_dirty") is not False:
        fail(f"{run_dir}: run config is not bound to clean source {head}")
    if not patch_path.is_file() or patch_path.stat().st_size != 0:
        fail(f"{run_dir}: source.patch must exist and be empty")
    return {
        "run_dir": str(run_dir),
        "git_sha": head,
        "git_dirty": False,
        "config_path": str(config_path),
        "config_sha256": file_sha256(config_path),
        "source_patch_empty": True,
    }


def markdown_report(comparison: dict[str, Any]) -> str:
    aggregate = comparison["aggregate"]
    decision = comparison["decision"]
    lines = [
        "# Gain-Budget GDN2 Comparison",
        "",
        "All scores are fixed official aggregates. Selected cases are diagnostic-only and never used to choose a score.",
        "",
        "## Decision",
        "",
        f"- Overall preregistered gate: `{'PASS' if decision['passed'] else 'FAIL'}`",
        f"- Score gate: `{'PASS' if decision['score_gate_passed'] else 'FAIL'}`",
        f"- Systems gate: `{'PASS' if decision['systems_gate']['passed'] else 'FAIL'}`",
        f"- Mechanism gate: `{'PASS' if decision['mechanism_gate']['passed'] else 'FAIL'}`",
        "",
        "## Aggregate",
        "",
        "| Metric | Control | Candidate | Delta |",
        "|---|---:|---:|---:|",
    ]
    for key in (
        "official_mean_loop5_exact",
        "official_mean_loop1_to_loop5_exact_gain",
        "official_mean_loop5_blank_acc",
    ):
        lines.append(
            f"| `{key}` | {aggregate['control'][key]:.6f} | "
            f"{aggregate['candidate'][key]:.6f} | "
            f"{aggregate['delta'][key]:+.6f} |"
        )
    lines.extend(
        [
            "",
            "## Official Ranges",
            "",
            "| Range | Arm | Eval N | Loop5 exact | Loop1->5 gain | Blank acc |",
            "|---|---|---:|---:|---:|---:|",
        ]
    )
    for label, row in comparison["ranges"].items():
        for arm in ("control", "candidate"):
            arm_row = row[arm]
            lines.append(
                f"| {label} | {arm} | {arm_row['eval_n']} | "
                f"{arm_row['loop5_exact']:.6f} | "
                f"{arm_row['loop1_to_loop5_exact_gain']:+.6f} | "
                f"{arm_row['loop5_blank_acc']:.6f} |"
            )
    paired = comparison["paired_case_aggregate"]
    lines.extend(
        [
            "",
            "## Paired Diagnostic Pool",
            "",
            "This separate 256-case-per-range pool is used only for paired error analysis and visualization; it never sets the official score.",
            "",
            "| Metric | Control | Candidate | Delta |",
            "|---|---:|---:|---:|",
        ]
    )
    for key in (
        "paired_mean_loop5_exact",
        "paired_mean_loop1_to_loop5_exact_gain",
        "paired_mean_loop5_blank_acc",
        "pooled_loop5_wrong_blank_mean",
        "pooled_loop1_to_loop5_wrong_blank_reduction",
    ):
        lines.append(
            f"| `{key}` | {paired['control'][key]:.6f} | "
            f"{paired['candidate'][key]:.6f} | "
            f"{paired['delta'][key]:+.6f} |"
        )
    system = comparison["system"]
    lines.extend(
        [
            "",
            "## Runtime And Memory",
            "",
            "| Metric | Control | Candidate | Delta | Relative |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for key in (
        "train_sec",
        "cuda_max_memory_allocated_mb",
        "cuda_max_memory_reserved_mb",
    ):
        lines.append(
            f"| `{key}` | {system['control'][key]:.3f} | "
            f"{system['candidate'][key]:.3f} | "
            f"{system['delta'][key]:+.3f} | "
            f"{system['relative_delta'][key]:+.2%} |"
        )
    lines.extend(["", "## Gain Diagnostics", ""])
    lines.append("```json")
    lines.append(
        json.dumps(
            comparison["gain_diagnostics"],
            indent=2,
            sort_keys=True,
        )
    )
    lines.extend(["```", "", "## Fixed Diagnostic Cases", ""])
    for case in comparison["selected_cases"]:
        lines.append(
            f"- `{case['category']}`: {case['range']} / "
            f"{case['case_id']}; loop5 candidate-control wrong delta "
            f"{case['loop5_wrong_delta_candidate_minus_control']:+d}"
        )
    return "\n".join(lines) + "\n"


def board_html(
    title: str,
    board: list[int],
    target: list[int],
    clue_mask: list[bool],
    *,
    target_board: bool = False,
    detail: str = "",
) -> str:
    cells = []
    for index, value in enumerate(board):
        classes = ["cell"]
        if target_board:
            classes.append("clue" if clue_mask[index] else "target-hidden")
        elif value != target[index]:
            classes.append("error")
        else:
            classes.append("clue" if clue_mask[index] else "correct")
        if index % 9 in (2, 5):
            classes.append("box-right")
        if index // 9 in (2, 5):
            classes.append("box-bottom")
        cells.append(
            f'<div class="{" ".join(classes)}">{int(value)}</div>'
        )
    return (
        '<section class="board-panel">'
        f"<h4>{html.escape(title)}</h4>"
        f'<div class="board">{"".join(cells)}</div>'
        f'<div class="board-detail">{html.escape(detail)}</div>'
        "</section>"
    )


def html_report(comparison: dict[str, Any]) -> str:
    aggregate = comparison["aggregate"]
    decision = comparison["decision"]
    aggregate_rows = []
    for key in (
        "official_mean_loop5_exact",
        "official_mean_loop1_to_loop5_exact_gain",
        "official_mean_loop5_blank_acc",
    ):
        aggregate_rows.append(
            "<tr>"
            f"<td>{html.escape(key)}</td>"
            f"<td>{aggregate['control'][key]:.6f}</td>"
            f"<td>{aggregate['candidate'][key]:.6f}</td>"
            f"<td>{aggregate['delta'][key]:+.6f}</td>"
            "</tr>"
        )
    range_rows = []
    for label, row in comparison["ranges"].items():
        for arm in ("control", "candidate"):
            metrics = row[arm]
            range_rows.append(
                "<tr>"
                f"<td>{label}</td><td>{arm}</td>"
                f"<td>{metrics['eval_n']}</td>"
                f"<td>{metrics['loop5_exact']:.6f}</td>"
                f"<td>{metrics['loop1_to_loop5_exact_gain']:+.6f}</td>"
                f"<td>{metrics['loop5_blank_acc']:.6f}</td>"
                "</tr>"
            )
    paired = comparison["paired_case_aggregate"]
    paired_rows = []
    for key in (
        "paired_mean_loop5_exact",
        "paired_mean_loop1_to_loop5_exact_gain",
        "paired_mean_loop5_blank_acc",
        "pooled_loop5_wrong_blank_mean",
        "pooled_loop1_to_loop5_wrong_blank_reduction",
    ):
        paired_rows.append(
            "<tr>"
            f"<td>{html.escape(key)}</td>"
            f"<td>{paired['control'][key]:.6f}</td>"
            f"<td>{paired['candidate'][key]:.6f}</td>"
            f"<td>{paired['delta'][key]:+.6f}</td>"
            "</tr>"
        )
    case_sections = []
    for case in comparison["selected_cases"]:
        panels = [
            board_html(
                "Target",
                case["target"],
                case["target"],
                case["clue_mask"],
                target_board=True,
                detail=f"{case['blank_count']} blanks",
            )
        ]
        for arm in ("control", "candidate"):
            for loop in LOOPS:
                loop_key = f"loop{loop}"
                row = case[arm][loop_key]
                panels.append(
                    board_html(
                        f"{arm.title()} loop {loop}",
                        row["prediction"],
                        case["target"],
                        case["clue_mask"],
                        detail=(
                            f"wrong blanks {row['wrong_blank_count']}; "
                            f"exact {str(row['label_exact']).lower()}"
                        ),
                    )
                )
        case_sections.append(
            '<section class="case">'
            f"<h2>{html.escape(case['category'])}</h2>"
            f"<p>{html.escape(case['selection_rule'])}. "
            f"Range {case['range']}; case {html.escape(case['case_id'])}; "
            "red cells are errors.</p>"
            f'<div class="boards">{"".join(panels)}</div>'
            "</section>"
        )
    system = comparison["system"]
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Gain-Budget GDN2 Comparison</title>
<style>
* {{ box-sizing: border-box; }}
body {{ margin: 0; font: 14px/1.45 system-ui, sans-serif; color: #18212b; background: #f5f7f8; }}
header, main {{ max-width: 1500px; margin: auto; padding: 24px; }}
header {{ background: #18212b; color: white; max-width: none; }}
header > div {{ max-width: 1452px; margin: auto; }}
h1, h2, h3, h4 {{ margin: 0 0 10px; letter-spacing: 0; }}
p {{ margin: 6px 0 16px; }}
.band {{ background: white; border-bottom: 1px solid #d8dee4; }}
table {{ border-collapse: collapse; width: 100%; margin: 8px 0 24px; background: white; }}
th, td {{ padding: 8px 10px; border: 1px solid #d8dee4; text-align: right; }}
th:first-child, td:first-child {{ text-align: left; }}
.case {{ padding: 22px 0; border-top: 1px solid #cbd3da; }}
.boards {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(246px, 1fr)); gap: 14px; align-items: start; }}
.board-panel {{ min-width: 0; }}
.board-panel h4 {{ font-size: 14px; }}
.board {{ display: grid; grid-template-columns: repeat(9, 1fr); width: min(100%, 270px); aspect-ratio: 1; border: 2px solid #18212b; background: #fff; }}
.cell {{ display: grid; place-items: center; min-width: 0; border-right: 1px solid #aab4bd; border-bottom: 1px solid #aab4bd; font-weight: 650; }}
.box-right {{ border-right: 2px solid #18212b; }}
.box-bottom {{ border-bottom: 2px solid #18212b; }}
.clue {{ background: #e3e7ea; color: #18212b; }}
.target-hidden {{ background: #dceef8; color: #12405b; }}
.correct {{ background: #e5f4e8; color: #155c2c; }}
.error {{ background: #ffd9d6; color: #9b1c16; }}
.board-detail {{ margin-top: 6px; min-height: 20px; color: #52606d; font-size: 12px; }}
.note {{ padding: 12px; border-left: 4px solid #277da1; background: #eaf5fa; }}
@media (max-width: 600px) {{ header, main {{ padding: 16px; }} .boards {{ grid-template-columns: 1fr; }} }}
</style>
</head>
<body>
<header><div><h1>Gain-Budget GDN2 Comparison</h1>
<p>Control: {html.escape(comparison['runs']['control']['name'])}<br>
Candidate: {html.escape(comparison['runs']['candidate']['name'])}</p></div></header>
<div class="band"><main>
<p class="note">Scores use every official case. The three displayed puzzles are fixed diagnostics: control hardest, candidate maximum improvement, and candidate maximum regression. They never select or alter a score.</p>
<h2>Decision</h2>
<p><strong>Overall: {'PASS' if decision['passed'] else 'FAIL'}</strong>.
Score gate: {'PASS' if decision['score_gate_passed'] else 'FAIL'};
systems gate: {'PASS' if decision['systems_gate']['passed'] else 'FAIL'};
mechanism gate: {'PASS' if decision['mechanism_gate']['passed'] else 'FAIL'}.</p>
<h2>Aggregate</h2>
<table><thead><tr><th>Metric</th><th>Control</th><th>Candidate</th><th>Delta</th></tr></thead>
	<tbody>{''.join(aggregate_rows)}</tbody></table>
	<h2>Official ranges</h2>
	<table><thead><tr><th>Range</th><th>Arm</th><th>Eval N</th><th>Loop5 exact</th><th>Loop1 to 5 gain</th><th>Blank acc</th></tr></thead>
	<tbody>{''.join(range_rows)}</tbody></table>
	<h2>Paired diagnostic pool</h2>
	<p>This separate 256-case-per-range pool is used only for paired error analysis and visualization; it never sets the official score.</p>
	<table><thead><tr><th>Metric</th><th>Control</th><th>Candidate</th><th>Delta</th></tr></thead>
	<tbody>{''.join(paired_rows)}</tbody></table>
<h2>Runtime and memory</h2>
<table><thead><tr><th>Metric</th><th>Control</th><th>Candidate</th><th>Relative delta</th></tr></thead><tbody>
<tr><td>Train seconds</td><td>{system['control']['train_sec']:.3f}</td><td>{system['candidate']['train_sec']:.3f}</td><td>{system['relative_delta']['train_sec']:+.2%}</td></tr>
<tr><td>Allocated MiB</td><td>{system['control']['cuda_max_memory_allocated_mb']:.3f}</td><td>{system['candidate']['cuda_max_memory_allocated_mb']:.3f}</td><td>{system['relative_delta']['cuda_max_memory_allocated_mb']:+.2%}</td></tr>
<tr><td>Reserved MiB</td><td>{system['control']['cuda_max_memory_reserved_mb']:.3f}</td><td>{system['candidate']['cuda_max_memory_reserved_mb']:.3f}</td><td>{system['relative_delta']['cuda_max_memory_reserved_mb']:+.2%}</td></tr>
</tbody></table>
</main></div>
<main>{''.join(case_sections)}</main>
</body>
</html>
"""


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Compare matched control/candidate official Sudoku runs using "
            "their complete all_cases artifacts."
        )
    )
    parser.add_argument("--control-result", required=True, type=Path)
    parser.add_argument("--candidate-result", required=True, type=Path)
    parser.add_argument(
        "--control-all-cases",
        action="append",
        default=[],
        metavar="RANGE=PATH",
        help="Override control artifacts; repeat for 51-55, 56-60, 61-64.",
    )
    parser.add_argument(
        "--candidate-all-cases",
        action="append",
        default=[],
        metavar="RANGE=PATH",
        help="Override candidate artifacts; repeat for 51-55, 56-60, 61-64.",
    )
    parser.add_argument("--out-dir", required=True, type=Path)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    control_result_path = args.control_result.expanduser().resolve()
    candidate_result_path = args.candidate_result.expanduser().resolve()
    control_result = load_json(control_result_path)
    candidate_result = load_json(candidate_result_path)
    control_paths = parse_range_paths(
        args.control_all_cases, "--control-all-cases"
    ) or infer_case_paths(control_result, control_result_path)
    candidate_paths = parse_range_paths(
        args.candidate_all_cases, "--candidate-all-cases"
    ) or infer_case_paths(candidate_result, candidate_result_path)

    matched_args = validate_matched_args(
        control_result,
        candidate_result,
        control_path=control_result_path,
        candidate_path=candidate_result_path,
    )
    control_provenance = run_provenance(
        control_result, control_result_path
    )
    candidate_provenance = run_provenance(
        candidate_result, candidate_result_path
    )
    if control_provenance["git_sha"] != candidate_provenance["git_sha"]:
        fail(
            "control and candidate were not produced from the same source SHA"
        )

    official_range_results: dict[str, dict[str, Any]] = {}
    paired_range_results: dict[str, dict[str, Any]] = {}
    all_paired_cases: list[dict[str, Any]] = []
    artifact_manifest: dict[str, Any] = {}
    eval_diagnostics: dict[str, Any] = {}
    for label in RANGES:
        control_artifact = load_case_artifact(control_paths[label], label)
        candidate_artifact = load_case_artifact(candidate_paths[label], label)
        paired = pair_cases(control_artifact, candidate_artifact, label=label)
        all_paired_cases.extend(paired)
        control_computed = summarize_cases(paired, "control")
        candidate_computed = summarize_cases(paired, "candidate")
        control_official = official_range(
            control_result, control_result_path, label
        )
        candidate_official = official_range(
            candidate_result, candidate_result_path, label
        )
        if control_official["eval_n"] != candidate_official["eval_n"]:
            fail(
                f"range {label}: official control/candidate eval_n differ"
            )
        expected_eval_n = control_result.get("args", {}).get("eval_n")
        if (
            isinstance(expected_eval_n, int)
            and control_official["eval_n"] != expected_eval_n
        ):
            fail(
                f"range {label}: official eval_n "
                f"{control_official['eval_n']} does not match configured "
                f"eval_n {expected_eval_n}"
            )

        control_official_summary = summarize_official(control_official)
        candidate_official_summary = summarize_official(candidate_official)
        official_range_results[label] = {
            "control": control_official_summary,
            "candidate": candidate_official_summary,
            "delta": metric_deltas(
                control_official_summary, candidate_official_summary
            ),
        }
        paired_range_results[label] = {
            "data_hash": control_artifact["data_hash"],
            "eval_n": len(paired),
            "control": control_computed,
            "candidate": candidate_computed,
            "delta": metric_deltas(control_computed, candidate_computed),
            "paired_outcomes": paired_outcomes(paired),
        }
        artifact_manifest[label] = {
            "data_hash": control_artifact["data_hash"],
            "control": {
                "path": str(control_artifact["path"]),
                "file_sha256": control_artifact["file_sha256"],
            },
            "candidate": {
                "path": str(candidate_artifact["path"]),
                "file_sha256": candidate_artifact["file_sha256"],
            },
        }
        eval_diagnostics[label] = diagnostics_comparison(
            control_official["gain_diagnostics"],
            candidate_official["gain_diagnostics"],
        )

    aggregate_control = aggregate_official_ranges(
        official_range_results, "control"
    )
    aggregate_candidate = aggregate_official_ranges(
        official_range_results, "candidate"
    )
    paired_aggregate_control = aggregate_ranges(
        paired_range_results, "control"
    )
    paired_aggregate_candidate = aggregate_ranges(
        paired_range_results, "candidate"
    )
    control_train = train_summary(control_result, control_result_path)
    candidate_train = train_summary(candidate_result, candidate_result_path)
    system_control = {
        key: control_train[key]
        for key in (
            "train_sec",
            "cuda_max_memory_allocated_mb",
            "cuda_max_memory_reserved_mb",
        )
    }
    system_candidate = {
        key: candidate_train[key]
        for key in system_control
    }
    system = {
        "control": system_control,
        "candidate": system_candidate,
        "delta": metric_deltas(system_control, system_candidate),
        "relative_delta": {
            key: ratio_delta(system_candidate[key], system_control[key])
            for key in system_control
        },
    }
    aggregate = {
        "control": aggregate_control,
        "candidate": aggregate_candidate,
        "delta": metric_deltas(
            aggregate_control, aggregate_candidate
        ),
    }
    paired_case_aggregate = {
        "control": paired_aggregate_control,
        "candidate": paired_aggregate_candidate,
        "delta": metric_deltas(
            paired_aggregate_control, paired_aggregate_candidate
        ),
    }
    decision = build_decision(
        official_range_results,
        aggregate,
        system,
        candidate_train,
    )
    generated_at = dt.datetime.now(dt.timezone.utc).replace(
        microsecond=0
    ).isoformat().replace("+00:00", "Z")
    comparison = {
        "schema_version": "gain_budget_gdn2_comparison.v1",
        "generated_at_utc": generated_at,
        "runs": {
            "control": {
                "name": result_name(control_result, control_result_path),
                "result_path": str(control_result_path),
                "result_sha256": file_sha256(control_result_path),
                **control_provenance,
            },
            "candidate": {
                "name": result_name(candidate_result, candidate_result_path),
                "result_path": str(candidate_result_path),
                "result_sha256": file_sha256(candidate_result_path),
                **candidate_provenance,
            },
        },
        "validation": {
            "matched_args": matched_args,
            "same_source_sha": True,
            "same_ordered_cases_required": True,
            "official_primary_metrics_from_result_eval_n": True,
            "official_metrics_recomputed_from_all_cases": False,
            "paired_case_pool_is_diagnostic_only": True,
            "paired_case_pool_affects_score": False,
            "artifacts": artifact_manifest,
        },
        "ranges": official_range_results,
        "aggregate": aggregate,
        "paired_case_ranges": paired_range_results,
        "paired_case_aggregate": paired_case_aggregate,
        "system": system,
        "gain_diagnostics": {
            "train": diagnostics_comparison(
                control_train["gain_diagnostics"],
                candidate_train["gain_diagnostics"],
            ),
            "eval_loop5_by_range": eval_diagnostics,
        },
        "selection_policy": {
            "diagnostic_only": True,
            "affects_score": False,
            "categories": [
                "control_hardest",
                "candidate_max_improvement",
                "candidate_max_regression",
            ],
        },
        "decision": decision,
        "selected_cases": select_cases(all_paired_cases),
    }

    out_dir = args.out_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    comparison_json = out_dir / "comparison.json"
    comparison_md = out_dir / "comparison.md"
    comparison_html = out_dir / "comparison.html"
    leaderboard_json = out_dir / "leaderboard_row.json"
    write_json(comparison_json, comparison)
    comparison_md.write_text(markdown_report(comparison), encoding="utf-8")
    comparison_html.write_text(html_report(comparison), encoding="utf-8")
    leaderboard = {
        "timestamp_utc": generated_at,
        "run_name": (
            f"{comparison['runs']['candidate']['name']}-vs-"
            f"{comparison['runs']['control']['name']}"
        ),
        "mode": "gain_budget_gdn2_comparison",
        "git_sha": comparison["runs"]["candidate"]["git_sha"],
        "git_dirty": False,
        "score": aggregate_candidate["official_mean_loop5_exact"],
        "score_key": (
            "aggregate.candidate.official_mean_loop5_exact"
        ),
        "run_dir": str(out_dir),
        "notes": (
            "candidate-control mean loop5 exact delta="
            f"{comparison['aggregate']['delta']['official_mean_loop5_exact']:+.6f}; "
            "fixed diagnostic cases do not affect score"
        ),
        "control_score": aggregate_control["official_mean_loop5_exact"],
        "candidate_score": aggregate_candidate["official_mean_loop5_exact"],
        "candidate_minus_control": comparison["aggregate"]["delta"][
            "official_mean_loop5_exact"
        ],
        "decision_passed": decision["passed"],
        "data_hashes": {
            label: row["data_hash"]
            for label, row in paired_range_results.items()
        },
    }
    write_json(leaderboard_json, leaderboard)
    print(
        json.dumps(
            {
                "comparison_json": str(comparison_json),
                "comparison_md": str(comparison_md),
                "comparison_html": str(comparison_html),
                "leaderboard_row_json": str(leaderboard_json),
                "candidate_minus_control_loop5_exact": comparison[
                    "aggregate"
                ]["delta"]["official_mean_loop5_exact"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
