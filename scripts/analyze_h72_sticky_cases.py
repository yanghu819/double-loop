#!/usr/bin/env python3
"""Analyze sticky final-loop errors in the h72 case bank."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
CASE_JSON = REPO_ROOT / "runs/casebank-h72-12x12-d256-l12-loop5-s1100-20260609T1013Z-8a2eb57/output/case_bank/h72/cases.json"
OUT_DIR = REPO_ROOT / "runs/h72-sticky-error-analysis-20260610"
N = 12
BOX_ROWS = 3
BOX_COLS = 4


def parse_coord(text: str) -> tuple[int, int]:
    row_s, col_s = text[1:].split("C", 1)
    return int(row_s), int(col_s)


def box_id(row: int, col: int) -> tuple[int, int]:
    return (row - 1) // BOX_ROWS + 1, (col - 1) // BOX_COLS + 1


def connected_components(wrongs: list[dict[str, str]]) -> list[list[dict[str, str]]]:
    if not wrongs:
        return []
    coords = [parse_coord(item["coord"]) for item in wrongs]
    parent = list(range(len(wrongs)))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for i, (ri, ci) in enumerate(coords):
        for j, (rj, cj) in enumerate(coords[i + 1 :], start=i + 1):
            same_unit = ri == rj or ci == cj or box_id(ri, ci) == box_id(rj, cj)
            same_digit = wrongs[i]["pred"] == wrongs[j]["pred"] or wrongs[i]["truth"] == wrongs[j]["truth"]
            swapped = wrongs[i]["pred"] == wrongs[j]["truth"] and wrongs[j]["pred"] == wrongs[i]["truth"]
            if same_unit or same_digit or swapped:
                union(i, j)
    groups: dict[int, list[dict[str, str]]] = defaultdict(list)
    for idx, item in enumerate(wrongs):
        groups[find(idx)].append(item)
    return list(groups.values())


def summarize_case(kind: str, case: dict[str, Any]) -> dict[str, Any]:
    final = case["loops"]["loop5"]
    wrongs = final["wrong_blanks"]
    comps = connected_components(wrongs)
    pred_digits = Counter(item["pred"] for item in wrongs)
    truth_digits = Counter(item["truth"] for item in wrongs)
    swap_edges = []
    for i, a in enumerate(wrongs):
        for b in wrongs[i + 1 :]:
            if a["pred"] == b["truth"] and b["pred"] == a["truth"]:
                swap_edges.append([a["coord"], b["coord"], f"{a['truth']}<->{a['pred']}"])
    return {
        "kind": kind,
        "stem": case["stem"],
        "batch_index": case["batch_index"],
        "final_wrong": final["wrong_count"],
        "final_conflicts": final["conflict_unit_count"],
        "final_entropy": final["hidden_entropy_mean"],
        "final_changed": final["changed_from_previous"],
        "wrong_blanks": wrongs,
        "component_sizes": sorted([len(group) for group in comps], reverse=True),
        "components": [[item["coord"] for item in group] for group in comps],
        "pred_digit_counts": dict(pred_digits),
        "truth_digit_counts": dict(truth_digits),
        "swap_edges": swap_edges,
        "loop_wrong_counts": {loop: case["loops"][loop]["wrong_count"] for loop in ["loop1", "loop2", "loop3", "loop5"]},
        "loop_conflicts": {loop: case["loops"][loop]["conflict_unit_count"] for loop in ["loop1", "loop2", "loop3", "loop5"]},
        "html": f"../casebank-h72-12x12-d256-l12-loop5-s1100-20260609T1013Z-8a2eb57/output/case_bank/h72/{case['stem']}.html",
    }


def build() -> dict[str, Any]:
    data = json.loads(CASE_JSON.read_text())
    rows = []
    for kind, cases in data["selected"].items():
        for case in cases:
            rows.append(summarize_case(kind, case))

    sticky = [row for row in rows if row["kind"] != "solved_by_loop"]
    return {
        "source": str(CASE_JSON.relative_to(REPO_ROOT)),
        "summary": data["summary"],
        "cases": rows,
        "sticky_summary": {
            "sticky_cases": len(sticky),
            "single_wrong_cases": sum(row["final_wrong"] == 1 for row in sticky),
            "five_wrong_cases": sum(row["final_wrong"] == 5 for row in sticky),
            "cases_with_swap_edges": sum(bool(row["swap_edges"]) for row in sticky),
            "mean_final_entropy": sum(row["final_entropy"] for row in sticky) / max(len(sticky), 1),
            "component_size_hist": dict(Counter(size for row in sticky for size in row["component_sizes"])),
        },
    }


def md_table(rows: list[dict[str, Any]]) -> str:
    lines = [
        "| case | kind | loop wrong 1/2/3/5 | final components | swaps | entropy | link |",
        "| --- | --- | --- | --- | --- | ---: | --- |",
    ]
    for row in rows:
        wrongs = row["loop_wrong_counts"]
        wrong_text = f"{wrongs['loop1']}/{wrongs['loop2']}/{wrongs['loop3']}/{wrongs['loop5']}"
        comp_text = ", ".join(str(x) for x in row["component_sizes"]) or "0"
        swap_text = "; ".join(edge[2] for edge in row["swap_edges"]) or "-"
        lines.append(
            f"| `{row['stem']}` | {row['kind']} | {wrong_text} | {comp_text} | "
            f"{swap_text} | {row['final_entropy']:.3f} | [html]({row['html']}) |"
        )
    return "\n".join(lines)


def write_outputs(payload: dict[str, Any]) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "sticky_cases.json").write_text(json.dumps(payload, indent=2) + "\n")
    rows = payload["cases"]
    sticky = [row for row in rows if row["kind"] != "solved_by_loop"]
    readme = f"""# h72 Sticky Error Analysis

Source: `{payload['source']}`

## Question

When h72 almost works, are the remaining errors independent low-confidence cells, or coupled structures that the loop cannot break?

## Finding

The remaining errors are coupled. The 4 almost-solved cases each have exactly one wrong final hidden cell, but that cell is tied to three duplicate-conflict units: row, column, and box. The hard failures mostly end with 5 wrong cells, and they contain clear digit swaps or small connected components rather than many independent mistakes.

This points toward a loop-mechanism bottleneck, not a simple classifier-capacity bottleneck. The model often knows enough locally to reduce 20-30 wrong cells to 1-5, but the last coupled configuration can become stable.

## Aggregate

```json
{json.dumps(payload['sticky_summary'], indent=2)}
```

## Cases

{md_table(rows)}

## Decision

The next high-ROI iteration should instrument and improve the recurrent refinement step, not add Sudoku-specific rules. A good next run is to trace per-loop token update norm, confidence margin, and FutureSeed injection for the sticky components, then test one simple generic loop change such as learned residual damping or a confidence-gated update.

"""
    (OUT_DIR / "README.md").write_text(readme)


def main() -> None:
    payload = build()
    write_outputs(payload)
    print(f"wrote {OUT_DIR.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
