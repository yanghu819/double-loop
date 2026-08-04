from __future__ import annotations

import argparse
import html
import json
from pathlib import Path


ARMS = ("causal_gdn2", "future_seed_gdn2", "bidirectional_attention")
LABELS = {
    "causal_gdn2": "Causal GDN2",
    "future_seed_gdn2": "GDN2 + FutureSeed",
    "bidirectional_attention": "Bidirectional attention",
}


def _byte_label(value: int) -> str:
    if value == 10:
        return "\\n"
    if value == 9:
        return "\\t"
    if 32 <= value < 127:
        return chr(value)
    return f"0x{value:02x}"


def _snippet(original: list[int], position: int, radius: int = 26) -> str:
    start = max(0, position - radius)
    end = min(len(original), position + radius + 1)
    pieces = []
    for index in range(start, end):
        token = html.escape(_byte_label(original[index]))
        if index == position:
            pieces.append(f'<mark class="target-byte">{token}</mark>')
        else:
            pieces.append(token)
    return "".join(pieces)


def _metric(value: float) -> str:
    return f"{value:.4f}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--html", type=Path, required=True)
    parser.add_argument("--cases", type=int, default=12)
    args = parser.parse_args()

    comparison = json.loads((args.output_dir / "comparison.json").read_text())
    cases_by_arm = {
        arm: json.loads((args.output_dir / arm / "cases.json").read_text())
        for arm in ARMS
    }
    indexed = {
        arm: {case["case_id"]: case for case in cases}
        for arm, cases in cases_by_arm.items()
    }
    causal_cases = cases_by_arm["causal_gdn2"]
    selected = []
    for causal in causal_cases:
        case_id = causal["case_id"]
        if not all(case_id in indexed[arm] for arm in ARMS):
            continue
        selected.append(causal)
    selected.sort(
        key=lambda case: (
            -case["errors"],
            indexed["future_seed_gdn2"][case["case_id"]]["errors"],
            indexed["bidirectional_attention"][case["case_id"]]["errors"],
            case["case_id"],
        )
    )
    selected = selected[: args.cases]

    metric_blocks = []
    for arm in ARMS:
        score = comparison["arms"][arm]
        metrics = score["metrics"]
        benchmark = score["warmed_step_benchmark"]
        metric_blocks.append(
            f"""
            <section class="metric-block">
              <h2>{html.escape(LABELS[arm])}</h2>
              <dl>
                <div><dt>Masked accuracy</dt><dd>{_metric(metrics['masked_accuracy'])}</dd></div>
                <div><dt>Masked CE</dt><dd>{_metric(metrics['masked_ce'])}</dd></div>
                <div><dt>Exact windows</dt><dd>{_metric(metrics['masked_exact'])}</dd></div>
                <div><dt>Parameters</dt><dd>{score['parameters']:,}</dd></div>
                <div><dt>Warmed tokens/s</dt><dd>{benchmark['tokens_per_sec']:,.0f}</dd></div>
                <div><dt>Peak train memory</dt><dd>{score['peak_training_cuda_mem_bytes'] / 2**30:.2f} GiB</dd></div>
              </dl>
            </section>
            """
        )

    case_blocks = []
    for rank, causal in enumerate(selected, start=1):
        case_id = causal["case_id"]
        rows = {arm: indexed[arm][case_id] for arm in ARMS}
        positions = causal["masked_positions"]
        target_by_position = dict(zip(positions, causal["targets"]))
        pred_by_arm = {
            arm: dict(zip(rows[arm]["masked_positions"], rows[arm]["predictions"]))
            for arm in ARMS
        }
        token_rows = []
        for position in positions:
            target = target_by_position[position]
            predictions = [pred_by_arm[arm][position] for arm in ARMS]
            if all(prediction == target for prediction in predictions):
                continue
            cells = []
            for arm, prediction in zip(ARMS, predictions):
                css = "correct" if prediction == target else "wrong"
                cells.append(
                    f'<td class="{css}">{html.escape(_byte_label(prediction))}</td>'
                )
            token_rows.append(
                "<tr>"
                f"<td>{position}</td>"
                f"<td class=\"target\">{html.escape(_byte_label(target))}</td>"
                + "".join(cells)
                + f'<td class="snippet"><code>{_snippet(causal["original"], position)}</code></td>'
                + "</tr>"
            )
        errors = " / ".join(
            f"{LABELS[arm]} {rows[arm]['errors']}" for arm in ARMS
        )
        case_blocks.append(
            f"""
            <section class="case">
              <header><h2>Case {rank} <code>{case_id}</code></h2><p>{html.escape(errors)} errors out of 38 masks</p></header>
              <div class="table-wrap"><table>
                <thead><tr><th>Pos</th><th>Target</th><th>Causal</th><th>FutureSeed</th><th>Bidir</th><th>Original context</th></tr></thead>
                <tbody>{''.join(token_rows)}</tbody>
              </table></div>
            </section>
            """
        )

    fs_delta = comparison["future_seed_vs_causal"]
    closure = comparison["future_seed_gap_closure"]
    document = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>P-CAUSAL-008: WikiText FutureSeed</title>
<style>
:root{{--ink:#171a1f;--muted:#626a73;--line:#d9dde2;--paper:#f7f8fa;--accent:#0b6bcb;--good:#d9f2e3;--bad:#fde1df;--target:#fff1ad}}
*{{box-sizing:border-box}} body{{margin:0;color:var(--ink);background:var(--paper);font:15px/1.45 ui-sans-serif,system-ui,sans-serif;letter-spacing:0}}
main{{max-width:1500px;margin:0 auto;padding:28px 24px 64px}} h1{{font-size:30px;margin:0 0 6px}} h2{{font-size:17px;margin:0}} p{{color:var(--muted);margin:4px 0 0}}
.summary{{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:12px;margin:24px 0}} .metric-block{{border-top:3px solid var(--accent);background:white;padding:16px}}
dl{{display:grid;grid-template-columns:1fr 1fr;gap:8px 14px;margin:14px 0 0}} dl div{{border-top:1px solid var(--line);padding-top:7px}} dt{{font-size:12px;color:var(--muted)}} dd{{margin:1px 0 0;font-variant-numeric:tabular-nums;font-weight:650}}
.decision{{background:#101820;color:white;padding:16px 18px;margin:0 0 26px}} .decision p{{color:#d8e1e8}} .case{{background:white;border-top:1px solid var(--line);padding:18px 0;margin:0}}
.case header{{padding:0 18px 12px}} .table-wrap{{overflow-x:auto}} table{{width:100%;border-collapse:collapse;table-layout:auto}} th,td{{padding:7px 9px;border-top:1px solid var(--line);text-align:center;white-space:nowrap;font-variant-numeric:tabular-nums}} th{{font-size:12px;color:var(--muted);background:#f2f4f6;position:sticky;top:0}} td.correct{{background:var(--good)}} td.wrong{{background:var(--bad)}} td.target{{background:var(--target);font-weight:700}} td.snippet{{text-align:left;min-width:560px;white-space:normal}} code{{font-family:ui-monospace,SFMono-Regular,Menlo,monospace}} mark.target-byte{{background:var(--target);font-weight:800;padding:1px 2px}}
@media(max-width:800px){{main{{padding:18px 10px 48px}} .summary{{grid-template-columns:1fr}} h1{{font-size:24px}}}}
</style></head><body><main>
<h1>WikiText-103 masked recovery</h1>
<p>P-CAUSAL-008: same text windows, same masks, causal GDN2 versus native FutureSeed versus full bidirectional attention.</p>
<div class="summary">{''.join(metric_blocks)}</div>
<section class="decision"><h2>Registered comparison</h2><p>FutureSeed vs causal: accuracy {fs_delta['masked_accuracy_delta']:+.4f}, CE {fs_delta['masked_ce_delta']:+.4f}. Gap closure: accuracy {closure['accuracy']}, CE {closure['ce']}.</p></section>
{''.join(case_blocks)}
</main></body></html>"""
    args.html.parent.mkdir(parents=True, exist_ok=True)
    args.html.write_text(document)
    print(args.html)


if __name__ == "__main__":
    main()
