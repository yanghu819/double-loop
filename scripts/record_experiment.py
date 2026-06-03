#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Tuple


def git(repo: Path, *args: str) -> str:
    return subprocess.check_output(["git", "-C", str(repo), *args], text=True).strip()


def load_result(run_dir: Path) -> Optional[Dict[str, Any]]:
    candidates = sorted((run_dir / "output").glob("futureseed_loop_seed*.json"))
    if not candidates:
        candidates = sorted(run_dir.glob("**/futureseed_loop_seed*.json"))
    if not candidates:
        return None
    with candidates[-1].open("r", encoding="utf-8") as f:
        return json.load(f)


def iter_number_paths(obj: Any, prefix: str = "") -> Iterable[Tuple[str, float]]:
    if isinstance(obj, dict):
        for key, value in obj.items():
            next_prefix = f"{prefix}.{key}" if prefix else str(key)
            yield from iter_number_paths(value, next_prefix)
    elif isinstance(obj, list):
        for idx, value in enumerate(obj):
            next_prefix = f"{prefix}[{idx}]"
            yield from iter_number_paths(value, next_prefix)
    elif isinstance(obj, (int, float)) and not isinstance(obj, bool):
        yield prefix, float(obj)


def extract_score(result: Optional[Dict[str, Any]]) -> Tuple[Optional[float], str]:
    if not result:
        return None, "missing_result_json"

    metrics = result.get("metrics", {})
    task = metrics.get("task", {})
    max_loops = task.get("max_loops", 3)
    preferred_key = f"eval_clean.loop{max_loops}.label_exact"
    try:
        preferred = metrics["eval_clean"][f"loop{max_loops}"]["label_exact"]
        return float(preferred), f"metrics.{preferred_key}"
    except Exception:
        pass

    rollout_scores = []
    for key, row in metrics.get("rollouts", {}).items():
        if isinstance(row, dict) and "oracle_exact" in row:
            rollout_scores.append((float(row["oracle_exact"]), f"metrics.rollouts.{key}.oracle_exact"))
    if rollout_scores:
        return max(rollout_scores, key=lambda item: item[0])

    label_scores = [(value, path) for path, value in iter_number_paths(metrics) if path.endswith("label_exact")]
    if label_scores:
        value, path = max(label_scores, key=lambda item: item[0])
        return value, f"metrics.{path}"

    return None, "missing_score"


def read_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
        f.write("\n")


def update_leaderboard(repo: Path, row: Dict[str, Any]) -> None:
    path = repo / "leaderboard.csv"
    fieldnames = [
        "timestamp_utc",
        "run_name",
        "mode",
        "git_sha",
        "git_dirty",
        "score",
        "score_key",
        "run_dir",
        "notes",
    ]
    rows = []
    if path.exists():
        with path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
    rows = [existing for existing in rows if existing.get("run_name") != row["run_name"]]
    rows.append({key: str(row.get(key, "")) for key in fieldnames})
    rows.sort(key=lambda item: item.get("timestamp_utc", ""))
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def source_dirty(repo: Path) -> bool:
    status = git(
        repo,
        "status",
        "--short",
        "--",
        ".",
        ":(exclude).cache",
        ":(exclude).venv",
        ":(exclude)artifacts",
        ":(exclude)models",
        ":(exclude)runs",
    )
    return bool(status)


def maybe_tag(repo: Path, score: Optional[float], sha: str, threshold: Optional[float]) -> Optional[str]:
    if score is None or threshold is None or score < threshold:
        return None
    tag = f"exp/score-{score:.4f}-{sha[:7]}"
    subprocess.check_call(["git", "-C", str(repo), "tag", "-f", tag, sha])
    return tag


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--mode", default="unknown")
    parser.add_argument("--no-leaderboard", action="store_true")
    parser.add_argument("--tag-min", type=float, default=None)
    parser.add_argument("--notes", default="")
    args = parser.parse_args()

    run_dir = Path(args.run_dir).resolve()
    repo = run_dir
    while repo != repo.parent and not (repo / ".git").exists():
        repo = repo.parent
    if not (repo / ".git").exists():
        raise SystemExit(f"Could not find git repository above {run_dir}")

    config = read_json(run_dir / "config.json")
    sha = str(config.get("git_sha") or git(repo, "rev-parse", "HEAD"))
    dirty = bool(config.get("git_dirty", source_dirty(repo)))
    result = load_result(run_dir)
    score, score_key = extract_score(result)

    timestamp = datetime.now(timezone.utc).isoformat()
    run_name = run_dir.name
    metadata_path = run_dir / "metadata.json"
    metadata = read_json(metadata_path)
    metadata.update(
        {
            "timestamp_utc": metadata.get("timestamp_utc", timestamp),
            "recorded_at_utc": timestamp,
            "run_name": run_name,
            "mode": args.mode,
            "git_sha": sha,
            "git_dirty": dirty,
            "score": score,
            "score_key": score_key,
            "notes": args.notes,
        }
    )
    write_json(metadata_path, metadata)
    write_json(run_dir / "score.json", {"score": score, "score_key": score_key, "git_sha": sha})

    tag = maybe_tag(repo, score, sha, args.tag_min)
    if tag:
        metadata["tag"] = tag
        write_json(metadata_path, metadata)

    if not args.no_leaderboard:
        update_leaderboard(
            repo,
            {
                "timestamp_utc": metadata["timestamp_utc"],
                "run_name": run_name,
                "mode": args.mode,
                "git_sha": sha,
                "git_dirty": int(dirty),
                "score": "" if score is None else f"{score:.6f}",
                "score_key": score_key,
                "run_dir": str(run_dir.relative_to(repo)),
                "notes": args.notes,
            },
        )

    print(f"recorded run={run_name} score={score} key={score_key}")


if __name__ == "__main__":
    main()
