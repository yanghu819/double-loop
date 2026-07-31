#!/usr/bin/env python3
from __future__ import annotations

import argparse
import html
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


RANGES = ("b51_55", "b56_60", "b61_64")
RANGE_LABELS = {
    "b51_55": "51-55 blanks",
    "b56_60": "56-60 blanks",
    "b61_64": "61-64 blanks",
}
LOOPS = (1, 2, 3, 4, 5)
DISPLAY_LOOPS = (1, 2, 5)
MATCHED_ARGS = (
    "backbone",
    "d_model",
    "layers",
    "heads",
    "head_dim",
    "channel_mult",
    "l_cycles",
    "max_loops",
    "batch",
    "grad_accum_steps",
    "lr",
    "weight_decay",
    "optimizer_contract",
    "loop_loss",
    "cell_order_train",
    "future_seed_scale",
    "future_seed_scope",
    "seed",
    "shared_shell_init_seed",
    "steps",
    "eval_n",
    "official_sudoku_data_dir",
    "official_sudoku_train_split",
    "official_sudoku_eval_split",
)


@dataclass(frozen=True)
class ArmSpec:
    key: str
    label: str
    color: str
    run_dir: Path


@dataclass
class ArmData:
    spec: ArmSpec
    summary: dict[str, Any]
    banks: dict[str, dict[str, Any]]


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_arm(spec: ArmSpec) -> ArmData:
    summary = load_json(spec.run_dir / "output" / "futureseed_loop_seed52.json")
    banks = {
        group: load_json(
            spec.run_dir
            / "output"
            / "case_bank"
            / f"official_{group}"
            / "all_cases.json"
        )
        for group in RANGES
    }
    return ArmData(spec=spec, summary=summary, banks=banks)


def case_map(bank: dict[str, Any]) -> dict[int, dict[str, Any]]:
    return {int(case["batch_index"]): case for case in bank["cases"]}


def wrong(case: dict[str, Any], loop: int) -> int:
    return int(case["loops"][f"loop{loop}"]["wrong_blank_count"])


def verify_matched(arms: list[ArmData]) -> dict[str, Any]:
    reference = arms[0]
    contract = {key: reference.summary["args"].get(key) for key in MATCHED_ARGS}
    for arm in arms[1:]:
        other = {key: arm.summary["args"].get(key) for key in MATCHED_ARGS}
        if other != contract:
            differences = {
                key: {reference.spec.key: contract[key], arm.spec.key: other[key]}
                for key in MATCHED_ARGS
                if contract[key] != other[key]
            }
            raise ValueError(f"matched training/eval contract differs: {differences}")

    hashes: dict[str, dict[str, str]] = {}
    for group in RANGES:
        group_hashes = {arm.spec.key: arm.banks[group]["data_hash"] for arm in arms}
        if len(set(group_hashes.values())) != 1:
            raise ValueError(f"{group}: data hashes differ: {group_hashes}")
        hashes[group] = group_hashes
        maps = {arm.spec.key: case_map(arm.banks[group]) for arm in arms}
        reference_ids = set(maps[reference.spec.key])
        if any(set(mapping) != reference_ids for mapping in maps.values()):
            raise ValueError(f"{group}: evaluated case IDs differ")
        for batch_index in reference_ids:
            left = maps[reference.spec.key][batch_index]
            for arm in arms[1:]:
                right = maps[arm.spec.key][batch_index]
                for key in ("puzzle", "label", "clue_mask", "blank_count"):
                    if left[key] != right[key]:
                        raise ValueError(
                            f"{group} batch {batch_index}: {key} differs for {arm.spec.key}"
                        )
    return {"matched_args": contract, "data_hashes": hashes}


def range_metrics(arm: ArmData, group: str) -> list[float]:
    payload = arm.summary["metrics"]["official_eval_by_blank_range"][group][
        "eval_clean"
    ]
    return [float(payload[f"loop{loop}"]["blank_acc"]) for loop in LOOPS]


def arm_metrics(arm: ArmData) -> dict[str, Any]:
    train = arm.summary["metrics"]["train"]
    by_range = {group: range_metrics(arm, group) for group in RANGES}
    loop_means = [
        sum(by_range[group][loop_index] for group in RANGES) / len(RANGES)
        for loop_index in range(len(LOOPS))
    ]
    return {
        "run_name": arm.spec.run_dir.name,
        "git_sha": load_json(arm.spec.run_dir / "score.json")["git_sha"],
        "train_ce": float(train["train_ce_loss"]),
        "train_sec": float(train["train_sec"]),
        "parameters": int(train["parameter_count"]),
        "vram_mb": float(train["cuda_max_memory_allocated_mb"]),
        "range_blank_acc": by_range,
        "mean_blank_acc_by_loop": loop_means,
        "loop_gain": loop_means[-1] - loop_means[0],
        "mean_loop5_blank_acc": loop_means[-1],
        "exact_by_range": {
            group: float(
                arm.summary["metrics"]["official_eval_by_blank_range"][group][
                    "eval_clean"
                ]["loop5"]["label_exact"]
            )
            for group in RANGES
        },
        "address_operator": train.get("address_operator"),
    }


def select_cases(arms: dict[str, ArmData]) -> dict[str, int]:
    selected: dict[str, int] = {}
    for group in RANGES:
        maps = {key: case_map(arm.banks[group]) for key, arm in arms.items()}

        def score(batch_index: int) -> tuple[int, int, int, int]:
            control5 = wrong(maps["control"][batch_index], 5)
            shared1 = wrong(maps["shared"][batch_index], 1)
            shared5 = wrong(maps["shared"][batch_index], 5)
            decoupled5 = wrong(maps["decoupled"][batch_index], 5)
            shared_advantage = control5 - shared5
            shared_loop_gain = shared1 - shared5
            decoupling_penalty = decoupled5 - shared5
            return (
                shared_advantage + shared_loop_gain + decoupling_penalty,
                shared_advantage,
                shared_loop_gain,
                -batch_index,
            )

        selected[group] = max(maps["control"], key=score)
    return selected


def sparkline(values: list[float], color: str) -> str:
    width, height, padding = 210, 54, 5
    low = min(values) - 0.01
    high = max(values) + 0.01
    span = max(high - low, 1e-6)
    points = []
    for index, value in enumerate(values):
        x = padding + index * (width - 2 * padding) / max(len(values) - 1, 1)
        y = height - padding - (value - low) / span * (height - 2 * padding)
        points.append(f"{x:.2f},{y:.2f}")
    circles = "".join(
        f'<circle cx="{point.split(",")[0]}" cy="{point.split(",")[1]}" r="2.8" />'
        for point in points
    )
    return (
        f'<svg class="spark" viewBox="0 0 {width} {height}" role="img" '
        f'aria-label="loop accuracy curve"><polyline points="{" ".join(points)}" '
        f'style="stroke:{color}" />'
        f'<g style="fill:{color}">{circles}</g></svg>'
    )


def board_html(
    *,
    title: str,
    values: list[int],
    label: list[int],
    clue_mask: list[bool],
    previous: list[int] | None,
    subtitle: str,
) -> str:
    cells: list[str] = []
    for index, value in enumerate(values):
        row, col = divmod(index, 9)
        classes = ["cell"]
        if clue_mask[index]:
            classes.append("clue")
        elif value == label[index]:
            classes.append("correct")
        else:
            classes.append("wrong")
        if previous is not None and value != previous[index]:
            classes.append("changed")
        if col in (2, 5):
            classes.append("box-right")
        if row in (2, 5):
            classes.append("box-bottom")
        display = "." if value == 0 else str(value)
        cells.append(f'<div class="{" ".join(classes)}">{display}</div>')
    return (
        '<section class="board-card">'
        f"<h4>{html.escape(title)}</h4>"
        f"<p>{html.escape(subtitle)}</p>"
        f'<div class="board">{"".join(cells)}</div>'
        "</section>"
    )


def prediction_board(case: dict[str, Any], arm: ArmSpec, loop: int) -> str:
    payload = case["loops"][f"loop{loop}"]
    previous = None if loop == 1 else case["loops"][f"loop{loop - 1}"]["prediction"]
    return board_html(
        title=f"{arm.label}, loop {loop}",
        values=payload["prediction"],
        label=case["label"],
        clue_mask=case["clue_mask"],
        previous=previous,
        subtitle=(
            f"wrong blanks {payload['wrong_blank_count']}/{case['blank_count']}; "
            f"accuracy {payload['blank_acc']:.3f}"
        ),
    )


def render(
    arms: list[ArmData],
    metrics: dict[str, dict[str, Any]],
    selected: dict[str, int],
    contract: dict[str, Any],
) -> str:
    arm_by_key = {arm.spec.key: arm for arm in arms}
    control = metrics["control"]
    shared = metrics["shared"]
    decoupled = metrics["decoupled"]
    position = metrics["position"]
    shared_delta = shared["mean_loop5_blank_acc"] - control["mean_loop5_blank_acc"]
    decoupled_delta = (
        decoupled["mean_loop5_blank_acc"] - control["mean_loop5_blank_acc"]
    )
    split_vs_shared = (
        decoupled["mean_loop5_blank_acc"] - shared["mean_loop5_blank_acc"]
    )
    decoupled_runtime = decoupled["train_sec"] / control["train_sec"] - 1.0

    table_rows = []
    curve_rows = []
    for arm in arms:
        m = metrics[arm.spec.key]
        table_rows.append(
            "<tr>"
            f'<td><span class="swatch" style="background:{arm.spec.color}"></span>'
            f"{html.escape(arm.spec.label)}</td>"
            f"<td>{m['train_ce']:.4f}</td>"
            f"<td>{m['mean_loop5_blank_acc']:.4f}</td>"
            f"<td>{m['loop_gain']:+.4f}</td>"
            f"<td>{m['parameters'] / 1e6:.3f}M</td>"
            f"<td>{m['train_sec']:.1f}s</td>"
            f"<td>{m['vram_mb'] / 1024:.2f} GiB</td>"
            "</tr>"
        )
        curves = "".join(
            '<div class="curve-cell">'
            f"<b>{RANGE_LABELS[group]}</b>"
            f"{sparkline(m['range_blank_acc'][group], arm.spec.color)}"
            f"<span>{m['range_blank_acc'][group][0]:.3f} to "
            f"{m['range_blank_acc'][group][-1]:.3f}</span>"
            "</div>"
            for group in RANGES
        )
        curve_rows.append(
            '<section class="curve-row">'
            f"<h3>{html.escape(arm.spec.label)}</h3>{curves}</section>"
        )

    case_sections = []
    for group in RANGES:
        batch_index = selected[group]
        cases = {
            key: case_map(arm_by_key[key].banks[group])[batch_index]
            for key in arm_by_key
        }
        reference = cases["control"]
        intro_boards = [
            board_html(
                title="Puzzle",
                values=reference["puzzle"],
                label=reference["label"],
                clue_mask=[value != 0 for value in reference["puzzle"]],
                previous=None,
                subtitle=f"{reference['blank_count']} hidden cells",
            ),
            board_html(
                title="Target",
                values=reference["label"],
                label=reference["label"],
                clue_mask=reference["clue_mask"],
                previous=None,
                subtitle="same label for every arm",
            ),
        ]
        arm_sections = []
        for arm in arms:
            arm_case = cases[arm.spec.key]
            wrong_trace = [wrong(arm_case, loop) for loop in LOOPS]
            arm_sections.append(
                '<section class="arm-case">'
                f'<h3 style="border-color:{arm.spec.color}">{html.escape(arm.spec.label)}</h3>'
                f'<p class="trace">wrong blanks by loop: {" -> ".join(map(str, wrong_trace))}</p>'
                '<div class="boards">'
                + "".join(
                    prediction_board(arm_case, arm.spec, loop)
                    for loop in DISPLAY_LOOPS
                )
                + "</div></section>"
            )
        case_sections.append(
            '<section class="case-band">'
            f"<h2>{RANGE_LABELS[group]}: mechanically selected batch {batch_index}</h2>"
            '<p class="case-note">Selected by shared-address advantage, shared loop correction, '
            "and the cost of splitting read/write maps. No board was chosen by eye.</p>"
            f'<div class="boards intro">{"".join(intro_boards)}</div>'
            + "".join(arm_sections)
            + "</section>"
        )

    contract_items = "".join(
        f"<li><code>{html.escape(key)}</code>: {html.escape(str(value))}</li>"
        for key, value in contract["matched_args"].items()
    )
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>GDN2 stable address binding</title>
<style>
*{{box-sizing:border-box}} body{{margin:0;background:#f4f6f7;color:#182026;
font:15px/1.45 system-ui,-apple-system,sans-serif;letter-spacing:0}}
main{{max-width:1480px;margin:auto;padding:22px}} h1{{font-size:28px;margin:0 0 6px}}
h2{{font-size:21px;margin:28px 0 8px}} h3{{font-size:17px;margin:0 0 8px}}
p{{max-width:980px}} code{{font-size:13px}} .decision{{background:white;
border-left:5px solid #b42318;padding:14px 18px;margin:18px 0}}
.decision strong{{font-size:18px}} .metrics{{display:grid;
grid-template-columns:repeat(4,minmax(160px,1fr));gap:10px;margin:14px 0}}
.metric{{background:white;border:1px solid #d5dadd;padding:12px;border-radius:6px}}
.metric b{{display:block;font-size:24px}} table{{width:100%;border-collapse:collapse;
background:white}} th,td{{border:1px solid #d5dadd;padding:9px;text-align:left}}
th{{background:#e9edef}} .swatch{{display:inline-block;width:12px;height:12px;
margin-right:8px;vertical-align:-1px}} .curve-row{{display:grid;
grid-template-columns:190px repeat(3,1fr);gap:12px;align-items:center;background:white;
border-bottom:1px solid #d5dadd;padding:10px}} .curve-cell{{display:grid;
grid-template-columns:110px 1fr 90px;gap:8px;align-items:center}}
.spark{{width:100%;height:54px}} .spark polyline{{fill:none;stroke-width:3}}
.case-band{{margin:34px -22px 0;padding:24px 22px;background:#e9edef}}
.case-band:nth-of-type(even){{background:#f8f9fa}} .case-note{{margin-top:0}}
.arm-case{{margin:18px 0 26px}} .arm-case h3{{border-left:5px solid;padding-left:9px}}
.trace{{font-family:ui-monospace,monospace;margin:4px 0 9px}}
.boards{{display:grid;grid-template-columns:repeat(3,minmax(220px,1fr));gap:12px}}
.boards.intro{{grid-template-columns:repeat(2,minmax(220px,330px))}}
.board-card{{background:white;border:1px solid #cfd5d9;padding:10px;border-radius:6px}}
.board-card h4{{font-size:15px;margin:0}} .board-card p{{min-height:40px;margin:3px 0 8px}}
.board{{display:grid;grid-template-columns:repeat(9,1fr);aspect-ratio:1}}
.cell{{display:grid;place-items:center;border:1px solid #aeb7c2;font-weight:650;
font-size:14px;position:relative}} .cell.clue{{background:#dce1e4;color:#182026}}
.cell.correct:not(.clue){{background:#d9f2e6;color:#0b5d3b}}
.cell.wrong{{background:#ffe0dc;color:#9d1c13}} .cell.changed{{outline:3px solid #146eb4;
outline-offset:-3px}} .box-right{{border-right:2px solid #182026}}
.box-bottom{{border-bottom:2px solid #182026}} .legend{{display:flex;gap:15px;flex-wrap:wrap}}
.legend span{{display:flex;align-items:center;gap:6px}} .legend i{{width:16px;height:16px;
display:inline-block;border:1px solid #9da7ad}} details{{background:white;padding:10px 14px;
margin-top:20px;border:1px solid #d5dadd}} details ul{{columns:2}}
@media(max-width:900px){{.metrics{{grid-template-columns:repeat(2,1fr)}}
.curve-row{{grid-template-columns:1fr}} .curve-cell{{grid-template-columns:100px 1fr 85px}}
.boards,.boards.intro{{grid-template-columns:1fr}} details ul{{columns:1}}}}
</style></head><body><main>
<h1>Stable address binding in GDN2 memory</h1>
<p>Same official Sudoku data, step9000 parent, optimizer, random traversal, FutureSeed,
five-loop supervision, and evaluation cases. Only the memory address construction changes.</p>
<section class="decision"><strong>Decision: keep one shared stable address; reject fully split read/write maps.</strong>
<p>The shared residual improves mean hard blank accuracy by {shared_delta:+.4f} over normal
GDN2. Splitting it into independent read and write maps keeps only {decoupled_delta:+.4f}
and falls {split_vs_shared:+.4f} behind shared, while taking {decoupled_runtime:+.1%} more
training time than control. The position-only diagnostic remains the upper signal at
{position['mean_loop5_blank_acc']:.4f}, but it removes content-driven addressing and is not
the retained general mechanism.</p></section>
<div class="metrics">
<div class="metric"><span>Normal mean loop5</span><b>{control['mean_loop5_blank_acc']:.3f}</b></div>
<div class="metric"><span>Shared address</span><b>{shared['mean_loop5_blank_acc']:.3f}</b></div>
<div class="metric"><span>Split read/write</span><b>{decoupled['mean_loop5_blank_acc']:.3f}</b></div>
<div class="metric"><span>Position-only diagnostic</span><b>{position['mean_loop5_blank_acc']:.3f}</b></div>
</div>
<h2>Aggregate result</h2>
<table><thead><tr><th>Arm</th><th>Train CE</th><th>Mean loop5 blank</th>
<th>Loop5 - loop1</th><th>Parameters</th><th>Train time</th><th>Peak VRAM</th></tr></thead>
<tbody>{"".join(table_rows)}</tbody></table>
<h2>Does more loop compute fix cells?</h2>
<p>Each line runs from loop1 to loop5. Rising lines mean later loops correct more hidden cells;
falling lines mean later loops damage the first answer.</p>
<section>{"".join(curve_rows)}</section>
<div class="legend"><span><i style="background:#dce1e4"></i>given clue</span>
<span><i style="background:#d9f2e6"></i>correct hidden cell</span>
<span><i style="background:#ffe0dc"></i>wrong hidden cell</span>
<span><i style="outline:3px solid #146eb4;outline-offset:-3px"></i>changed since prior loop</span></div>
{"".join(case_sections)}
<details><summary>Matched experiment contract and data hashes</summary>
<ul>{contract_items}</ul><pre>{html.escape(json.dumps(contract['data_hashes'], indent=2))}</pre></details>
</main></body></html>"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--control-run", type=Path, required=True)
    parser.add_argument("--shared-run", type=Path, required=True)
    parser.add_argument("--decoupled-run", type=Path, required=True)
    parser.add_argument("--position-run", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    specs = [
        ArmSpec("control", "Normal content Q/K", "#65727a", args.control_run),
        ArmSpec("shared", "Shared stable address", "#087f5b", args.shared_run),
        ArmSpec("decoupled", "Split read/write address", "#b42318", args.decoupled_run),
        ArmSpec("position", "Position-only Q/K diagnostic", "#146eb4", args.position_run),
    ]
    arms = [load_arm(spec) for spec in specs]
    contract = verify_matched(arms)
    arm_by_key = {arm.spec.key: arm for arm in arms}
    metrics = {key: arm_metrics(arm) for key, arm in arm_by_key.items()}
    selected = select_cases(arm_by_key)
    payload = {
        "decision": "retain shared stable address; reject fully split read/write maps",
        "metrics": metrics,
        "selected_cases": selected,
        "contract": contract,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "comparison.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (args.output_dir / "index.html").write_text(
        render(arms, metrics, selected, contract),
        encoding="utf-8",
    )
    print(json.dumps({"output_dir": str(args.output_dir), **payload}, indent=2))


if __name__ == "__main__":
    main()
