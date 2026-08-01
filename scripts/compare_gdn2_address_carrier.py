#!/usr/bin/env python3
from __future__ import annotations

import argparse
import html
import json
from pathlib import Path
from typing import Any

from compare_gdn2_address_binding import (
    DISPLAY_LOOPS,
    LOOPS,
    RANGES,
    RANGE_LABELS,
    ArmData,
    ArmSpec,
    arm_metrics,
    board_html,
    case_map,
    load_arm,
    prediction_board,
    sparkline,
    verify_matched,
    wrong,
)


CARRIER_KEYS = (
    "gdn2_address_carrier_mean",
    "gdn2_address_carrier_std",
    "gdn2_address_carrier_token_std",
    "gdn2_address_carrier_min",
    "gdn2_address_carrier_below_095_frac",
    "gdn2_address_carrier_scale_rms",
    "gdn2_address_carrier_bias_delta_rms",
    "gdn2_address_carrier_write_norm_ratio",
    "gdn2_address_carrier_b_input_relative_change",
    "gdn2_address_carrier_w_input_relative_change",
)


def checkpoint_identity(arm: ArmData) -> dict[str, Any]:
    args = arm.summary["args"]
    return {
        "path": args.get("resume_train_checkpoint"),
        "sha256": args.get("resume_train_checkpoint_sha256"),
    }


def verify_carrier_pair(control: ArmData, carrier: ArmData) -> dict[str, Any]:
    contract = verify_matched([control, carrier])
    modes = {
        "control": control.summary["args"].get("gdn2_address_mode"),
        "carrier": carrier.summary["args"].get("gdn2_address_mode"),
    }
    expected_modes = {"control": "anchor_residual", "carrier": "anchor_carrier"}
    if modes != expected_modes:
        raise ValueError(f"unexpected arm modes: {modes}; expected {expected_modes}")

    checkpoints = {
        "control": checkpoint_identity(control),
        "carrier": checkpoint_identity(carrier),
    }
    if checkpoints["control"] != checkpoints["carrier"]:
        raise ValueError(f"resume checkpoint identity differs: {checkpoints}")
    if not checkpoints["control"]["sha256"]:
        raise ValueError("resume checkpoint SHA256 is absent")

    contract["arm_modes"] = modes
    contract["resume_checkpoint"] = checkpoints["control"]
    return contract


def extended_metrics(arm: ArmData) -> dict[str, Any]:
    metrics = arm_metrics(arm)
    train = arm.summary["metrics"]["train"]
    address_operator = train.get("address_operator") or {}
    metrics["carrier"] = {
        key.removeprefix("gdn2_address_carrier_"): float(
            address_operator.get(key, 0.0)
        )
        for key in CARRIER_KEYS
    }
    metrics["terminal_state_rms"] = float(
        address_operator.get("gdn2_address_terminal_state_rms", 0.0)
    )
    metrics["mean_loop5_exact"] = sum(metrics["exact_by_range"].values()) / len(
        RANGES
    )
    metrics["range_loop5_wrong_mean"] = {}
    for group in RANGES:
        cases = arm.banks[group]["cases"]
        metrics["range_loop5_wrong_mean"][group] = sum(
            wrong(case, 5) for case in cases
        ) / max(len(cases), 1)
    return metrics


def evaluate_gate(metrics: dict[str, dict[str, Any]]) -> dict[str, Any]:
    control = metrics["control"]
    carrier = metrics["carrier"]
    accuracy_delta = (
        carrier["mean_loop5_blank_acc"] - control["mean_loop5_blank_acc"]
    )
    ce_delta = carrier["train_ce"] - control["train_ce"]
    loop_gain_delta = carrier["loop_gain"] - control["loop_gain"]
    runtime_overhead = carrier["train_sec"] / max(control["train_sec"], 1e-9) - 1.0
    token_std = carrier["carrier"]["token_std"]
    state_rms_ratio = carrier["terminal_state_rms"] / max(
        control["terminal_state_rms"], 1e-9
    )
    address_specific = token_std >= 0.002
    cost_ok = runtime_overhead <= 0.25 or accuracy_delta >= 0.05
    state_stable = state_rms_ratio <= 2.0
    primary_win = accuracy_delta >= 0.03
    mechanism_signal = ce_delta <= -0.05 and loop_gain_delta > 0.0
    passed = (
        address_specific
        and cost_ok
        and state_stable
        and (primary_win or mechanism_signal)
    )
    if passed:
        decision = "continue: address-conditioned carrier passed the preregistered gate"
    elif not address_specific:
        decision = "discard: carrier did not become address-specific"
    elif not cost_ok:
        decision = "discard: carrier cost exceeded the 25% budget without a qualifying gain"
    elif not state_stable:
        decision = "discard: carrier terminal state exceeded the 2x stability bound"
    else:
        decision = "discard: carrier did not improve accuracy or recurrent correction enough"
    return {
        "passed": passed,
        "decision": decision,
        "mean_loop5_blank_acc_delta": accuracy_delta,
        "train_ce_delta": ce_delta,
        "loop_gain_delta": loop_gain_delta,
        "runtime_overhead": runtime_overhead,
        "carrier_token_std": token_std,
        "terminal_state_rms_ratio": state_rms_ratio,
        "thresholds": {
            "mean_loop5_blank_acc_delta": 0.03,
            "train_ce_delta": -0.05,
            "carrier_token_std": 0.002,
            "runtime_overhead": 0.25,
            "terminal_state_rms_ratio": 2.0,
        },
    }


def select_cases(control: ArmData, carrier: ArmData) -> dict[str, int]:
    selected: dict[str, int] = {}
    for group in RANGES:
        control_cases = case_map(control.banks[group])
        carrier_cases = case_map(carrier.banks[group])

        def score(batch_index: int) -> tuple[int, int, int, int]:
            control_loop_change = wrong(control_cases[batch_index], 1) - wrong(
                control_cases[batch_index], 5
            )
            carrier_loop_change = wrong(carrier_cases[batch_index], 1) - wrong(
                carrier_cases[batch_index], 5
            )
            final_advantage = wrong(control_cases[batch_index], 5) - wrong(
                carrier_cases[batch_index], 5
            )
            correction_advantage = carrier_loop_change - control_loop_change
            return (
                final_advantage + correction_advantage,
                correction_advantage,
                final_advantage,
                -batch_index,
            )

        selected[group] = max(control_cases, key=score)
    return selected


def matched_case_outcomes(control: ArmData, carrier: ArmData) -> dict[str, Any]:
    outcomes: dict[str, Any] = {}
    for group in RANGES:
        control_cases = case_map(control.banks[group])
        carrier_cases = case_map(carrier.banks[group])
        final_deltas: list[int] = []
        correction_deltas: list[int] = []
        for batch_index, control_case in control_cases.items():
            carrier_case = carrier_cases[batch_index]
            final_deltas.append(wrong(control_case, 5) - wrong(carrier_case, 5))
            control_correction = wrong(control_case, 1) - wrong(control_case, 5)
            carrier_correction = wrong(carrier_case, 1) - wrong(carrier_case, 5)
            correction_deltas.append(carrier_correction - control_correction)
        outcomes[group] = {
            "n": len(final_deltas),
            "carrier_final_better": sum(delta > 0 for delta in final_deltas),
            "carrier_final_equal": sum(delta == 0 for delta in final_deltas),
            "carrier_final_worse": sum(delta < 0 for delta in final_deltas),
            "mean_wrong_cell_reduction": sum(final_deltas) / max(len(final_deltas), 1),
            "carrier_loop_correction_better": sum(
                delta > 0 for delta in correction_deltas
            ),
            "mean_loop_correction_advantage": sum(correction_deltas)
            / max(len(correction_deltas), 1),
        }
    return outcomes


def render(
    arms: list[ArmData],
    metrics: dict[str, dict[str, Any]],
    selected: dict[str, int],
    contract: dict[str, Any],
    gate: dict[str, Any],
    outcomes: dict[str, Any],
) -> str:
    arm_by_key = {arm.spec.key: arm for arm in arms}
    control = metrics["control"]
    carrier = metrics["carrier"]
    decision_class = "pass" if gate["passed"] else "fail"

    table_rows: list[str] = []
    curve_rows: list[str] = []
    for arm in arms:
        metric = metrics[arm.spec.key]
        exact_trace = "/".join(
            f"{metric['exact_by_range'][group]:.3f}" for group in RANGES
        )
        table_rows.append(
            "<tr>"
            f'<td><span class="swatch" style="background:{arm.spec.color}"></span>'
            f"{html.escape(arm.spec.label)}</td>"
            f"<td>{metric['train_ce']:.4f}</td>"
            f"<td>{metric['mean_loop5_blank_acc']:.4f}</td>"
            f"<td>{metric['mean_loop5_exact']:.4f}</td>"
            f"<td>{exact_trace}</td>"
            f"<td>{metric['loop_gain']:+.4f}</td>"
            f"<td>{metric['parameters'] / 1e6:.3f}M</td>"
            f"<td>{metric['train_sec']:.1f}s</td>"
            f"<td>{metric['vram_mb'] / 1024:.2f} GiB</td>"
            "</tr>"
        )
        curves = "".join(
            '<div class="curve-cell">'
            f"<b>{RANGE_LABELS[group]}</b>"
            f"{sparkline(metric['range_blank_acc'][group], arm.spec.color)}"
            f"<span>{metric['range_blank_acc'][group][0]:.3f} to "
            f"{metric['range_blank_acc'][group][-1]:.3f}</span>"
            "</div>"
            for group in RANGES
        )
        curve_rows.append(
            '<section class="curve-row">'
            f"<h3>{html.escape(arm.spec.label)}</h3>{curves}</section>"
        )

    carrier_diag = carrier["carrier"]
    diagnostic_rows = "".join(
        f"<tr><td>{html.escape(key.replace('_', ' '))}</td><td>{value:.6f}</td></tr>"
        for key, value in carrier_diag.items()
    )
    outcome_rows = "".join(
        "<tr>"
        f"<td>{RANGE_LABELS[group]}</td>"
        f"<td>{row['carrier_final_better']}</td>"
        f"<td>{row['carrier_final_equal']}</td>"
        f"<td>{row['carrier_final_worse']}</td>"
        f"<td>{row['mean_wrong_cell_reduction']:+.3f}</td>"
        f"<td>{row['carrier_loop_correction_better']}</td>"
        f"<td>{row['mean_loop_correction_advantage']:+.3f}</td>"
        "</tr>"
        for group, row in outcomes.items()
    )

    case_sections: list[str] = []
    for group in RANGES:
        batch_index = selected[group]
        cases = {
            key: case_map(arm.banks[group])[batch_index]
            for key, arm in arm_by_key.items()
        }
        reference = cases["control"]
        intro = "".join(
            [
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
                    subtitle="identical target for both arms",
                ),
            ]
        )
        arm_sections: list[str] = []
        for arm in arms:
            case = cases[arm.spec.key]
            wrong_trace = [wrong(case, loop) for loop in LOOPS]
            arm_sections.append(
                '<section class="arm-case">'
                f'<h3 style="border-color:{arm.spec.color}">{html.escape(arm.spec.label)}</h3>'
                f'<p class="trace">wrong blanks: {" -> ".join(map(str, wrong_trace))}</p>'
                '<div class="boards">'
                + "".join(
                    prediction_board(case, arm.spec, loop) for loop in DISPLAY_LOOPS
                )
                + "</div></section>"
            )
        case_sections.append(
            '<section class="case-band">'
            f"<h2>{RANGE_LABELS[group]}: batch {batch_index}</h2>"
            '<p class="case-note">Mechanically selected by final carrier advantage plus '
            "loop1-to-loop5 correction advantage. No board was selected by eye.</p>"
            f'<div class="boards intro">{intro}</div>'
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
<title>GDN2 address-conditioned write carrier</title>
<style>
*{{box-sizing:border-box}} body{{margin:0;background:#f3f5f6;color:#182026;
font:15px/1.45 system-ui,-apple-system,sans-serif;letter-spacing:0}}
main{{max-width:1420px;margin:auto;padding:22px}} h1{{font-size:28px;margin:0 0 5px}}
h2{{font-size:21px;margin:28px 0 8px}} h3{{font-size:17px;margin:0 0 8px}}
p{{max-width:1000px}} .decision{{background:#fff;border-left:5px solid;padding:14px 18px;
margin:18px 0}} .decision.pass{{border-color:#087f5b}} .decision.fail{{border-color:#b42318}}
.decision strong{{font-size:18px}} .metrics{{display:grid;
grid-template-columns:repeat(4,minmax(150px,1fr));gap:10px;margin:14px 0}}
.metric{{background:#fff;border:1px solid #d5dadd;padding:12px;border-radius:6px}}
.metric b{{display:block;font-size:23px}} table{{width:100%;border-collapse:collapse;background:#fff}}
th,td{{border:1px solid #d5dadd;padding:9px;text-align:left}} th{{background:#e7ebed}}
.swatch{{display:inline-block;width:12px;height:12px;margin-right:8px;vertical-align:-1px}}
.curve-row{{display:grid;grid-template-columns:190px repeat(3,1fr);gap:12px;
align-items:center;background:#fff;border-bottom:1px solid #d5dadd;padding:10px}}
.curve-cell{{display:grid;grid-template-columns:110px 1fr 90px;gap:8px;align-items:center}}
.spark{{width:100%;height:54px}} .spark polyline{{fill:none;stroke-width:3}}
.case-band{{margin:34px -22px 0;padding:24px 22px;background:#e7ebed}}
.case-band:nth-of-type(even){{background:#f8f9fa}} .case-note{{margin-top:0}}
.arm-case{{margin:18px 0 26px}} .arm-case h3{{border-left:5px solid;padding-left:9px}}
.trace{{font-family:ui-monospace,monospace;margin:4px 0 9px}}
.boards{{display:grid;grid-template-columns:repeat(3,minmax(220px,1fr));gap:12px}}
.boards.intro{{grid-template-columns:repeat(2,minmax(220px,330px))}}
.board-card{{background:#fff;border:1px solid #cfd5d9;padding:10px;border-radius:6px}}
.board-card h4{{font-size:15px;margin:0}} .board-card p{{min-height:40px;margin:3px 0 8px}}
.board{{display:grid;grid-template-columns:repeat(9,1fr);aspect-ratio:1}}
.cell{{display:grid;place-items:center;border:1px solid #aeb7c2;font-weight:650;
font-size:14px;position:relative}} .cell.clue{{background:#dce1e4;color:#182026}}
.cell.correct:not(.clue){{background:#d9f2e6;color:#0b5d3b}}
.cell.wrong{{background:#ffe0dc;color:#9d1c13}} .cell.changed{{outline:3px solid #146eb4;
outline-offset:-3px}} .box-right{{border-right:2px solid #182026}}
.box-bottom{{border-bottom:2px solid #182026}} .legend{{display:flex;gap:15px;flex-wrap:wrap}}
.legend span{{display:flex;align-items:center;gap:6px}} .legend i{{width:16px;height:16px;
display:inline-block;border:1px solid #9da7ad}} details{{background:#fff;padding:10px 14px;
margin-top:20px;border:1px solid #d5dadd}} details ul{{columns:2}}
@media(max-width:900px){{.metrics{{grid-template-columns:repeat(2,1fr)}}
.curve-row{{grid-template-columns:1fr}} .curve-cell{{grid-template-columns:100px 1fr 85px}}
.boards,.boards.intro{{grid-template-columns:1fr}} details ul{{columns:1}}}}
</style></head><body><main>
<h1>Can a sequence address decide where GDN2 writes?</h1>
<p>Both arms start from the same shared-address step9100 checkpoint and receive the same
100 optimizer steps, official hard Sudoku samples, FutureSeed, loop supervision, and evaluation
cases. The candidate adds only a generic per-token, per-head, per-key-channel write carrier.</p>
<section class="decision {decision_class}"><strong>{html.escape(gate['decision'])}</strong>
<p>Mean loop5 blank accuracy delta {gate['mean_loop5_blank_acc_delta']:+.4f}; train CE delta
{gate['train_ce_delta']:+.4f}; recurrent correction delta {gate['loop_gain_delta']:+.4f};
runtime overhead {gate['runtime_overhead']:+.1%}; carrier token variation
{gate['carrier_token_std']:.4f}; terminal-state RMS ratio
{gate['terminal_state_rms_ratio']:.3f}x.</p></section>
<div class="metrics">
<div class="metric"><span>Control mean loop5</span><b>{control['mean_loop5_blank_acc']:.3f}</b></div>
<div class="metric"><span>Carrier mean loop5</span><b>{carrier['mean_loop5_blank_acc']:.3f}</b></div>
<div class="metric"><span>Carrier delta</span><b>{gate['mean_loop5_blank_acc_delta']:+.3f}</b></div>
<div class="metric"><span>Carrier token std</span><b>{gate['carrier_token_std']:.4f}</b></div>
</div>
<h2>Matched aggregate result</h2>
<table><thead><tr><th>Arm</th><th>Train CE</th><th>Mean loop5 blank</th>
<th>Mean loop5 exact</th><th>Exact 51-55/56-60/61-64</th><th>Loop5 - loop1</th>
<th>Parameters</th><th>Train time</th><th>Peak VRAM</th></tr></thead>
<tbody>{''.join(table_rows)}</tbody></table>
<h2>Did later loops correct more cells?</h2>
<p>Each curve runs from loop1 to loop5. A useful recurrent mechanism should rise because later
loops correct hidden cells, not merely because the first answer moved to another operating point.</p>
<section>{''.join(curve_rows)}</section>
<h2>Matched case-level evidence</h2>
<p>Counts use every paired case in each official range. Positive wrong-cell reduction means
the carrier makes fewer final hidden-cell errors; positive correction advantage means its
loop1-to-loop5 improvement is larger than control.</p>
<table><thead><tr><th>Range</th><th>Carrier better</th><th>Equal</th><th>Worse</th>
<th>Mean wrong-cell reduction</th><th>Better loop correction</th>
<th>Mean correction advantage</th></tr></thead><tbody>{outcome_rows}</tbody></table>
<h2>Did the carrier actually use the sequence address?</h2>
<table><thead><tr><th>Diagnostic</th><th>Candidate value</th></tr></thead>
<tbody>{diagnostic_rows}</tbody></table>
<div class="legend"><span><i style="background:#dce1e4"></i>given clue</span>
<span><i style="background:#d9f2e6"></i>correct hidden cell</span>
<span><i style="background:#ffe0dc"></i>wrong hidden cell</span>
<span><i style="outline:3px solid #146eb4;outline-offset:-3px"></i>changed since prior loop</span></div>
{''.join(case_sections)}
<details><summary>Matched contract, parent checkpoint, and data hashes</summary>
<ul>{contract_items}</ul><pre>{html.escape(json.dumps(contract, indent=2))}</pre></details>
</main></body></html>"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--control-run", type=Path, required=True)
    parser.add_argument("--carrier-run", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    specs = [
        ArmSpec("control", "Shared address GDN2", "#65727a", args.control_run),
        ArmSpec(
            "carrier",
            "Address-conditioned write carrier",
            "#087f5b",
            args.carrier_run,
        ),
    ]
    arms = [load_arm(spec) for spec in specs]
    control, carrier = arms
    contract = verify_carrier_pair(control, carrier)
    metrics = {arm.spec.key: extended_metrics(arm) for arm in arms}
    selected = select_cases(control, carrier)
    outcomes = matched_case_outcomes(control, carrier)
    gate = evaluate_gate(metrics)
    payload = {
        "gate": gate,
        "metrics": metrics,
        "selected_cases": selected,
        "matched_case_outcomes": outcomes,
        "contract": contract,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "comparison.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (args.output_dir / "index.html").write_text(
        render(arms, metrics, selected, contract, gate, outcomes), encoding="utf-8"
    )
    print(json.dumps({"output_dir": str(args.output_dir), **payload}, indent=2))


if __name__ == "__main__":
    main()
