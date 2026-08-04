from __future__ import annotations

import argparse
import html
import json
from pathlib import Path


def percent(value: float) -> str:
    return f"{100.0 * value:.2f}%"


def metric_row(label: str, no_fs: float, fs: float) -> str:
    delta = fs - no_fs
    return (
        "<tr>"
        f"<th>{html.escape(label)}</th>"
        f"<td>{percent(no_fs)}</td>"
        f"<td>{percent(fs)}</td>"
        f"<td class='delta'>{delta * 100.0:+.2f} pp</td>"
        "</tr>"
    )


def sequence_cell(position: int, token: int, query_positions: set[int]) -> str:
    if position < 16:
        region = "future-query"
        region_label = "FQ"
    elif position < 32:
        region = "past-write"
        region_label = "PW"
    elif position < 48:
        region = "past-query"
        region_label = "PQ"
    else:
        region = "future-write"
        region_label = "FW"
    token_kind = "fill" if token < 64 else "key" if token < 160 else "value"
    query = " query" if position in query_positions else ""
    return (
        f"<div class='token {region} {token_kind}{query}' title='position {position}'>"
        f"<span class='pos'>{position}</span>"
        f"<strong>{token}</strong>"
        f"<span class='region'>{region_label}</span>"
        "</div>"
    )


def render_case(case: dict) -> str:
    positions = case["query_positions"]
    targets = case["targets"]
    no_fs = case["no_future_seed_predictions"]
    fs = case["future_seed_predictions"]
    results = []
    for position, target, no_prediction, fs_prediction in zip(
        positions, targets, no_fs, fs
    ):
        no_class = "ok" if no_prediction == target else "bad"
        fs_class = "ok" if fs_prediction == target else "bad"
        results.append(
            "<tr>"
            f"<th>position {position}</th><td>{target}</td>"
            f"<td class='{no_class}'>{no_prediction}</td>"
            f"<td class='{fs_class}'>{fs_prediction}</td>"
            "</tr>"
        )
    cells = "".join(
        sequence_cell(index, token, set(positions))
        for index, token in enumerate(case["input"])
    )
    return f"""
    <article class="case">
      <header>
        <h3>Case {case['case_index']} <code>{case['case_id']}</code></h3>
        <span>No-FS errors {case['no_future_seed_errors']} / FS errors {case['future_seed_errors']}</span>
      </header>
      <div class="sequence">{cells}</div>
      <table class="case-results">
        <thead><tr><th>Future query</th><th>Target</th><th>No FutureSeed</th><th>FutureSeed</th></tr></thead>
        <tbody>{''.join(results)}</tbody>
      </table>
    </article>
    """.strip()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    comparison = json.loads((args.run_dir / "output/comparison.json").read_text())
    cases = json.loads((args.run_dir / "output/paired_hardest_cases.json").read_text())
    preflight = json.loads((args.run_dir / "preflight.json").read_text())
    no_fs = comparison["no_future_seed"]
    fs = comparison["future_seed"]

    no_metrics = no_fs["metrics"]
    fs_metrics = fs["metrics"]
    metric_rows = "".join(
        [
            metric_row(
                "Past-query accuracy",
                no_metrics["past"]["accuracy"],
                fs_metrics["past"]["accuracy"],
            ),
            metric_row(
                "Future-query accuracy",
                no_metrics["future"]["accuracy"],
                fs_metrics["future"]["accuracy"],
            ),
            metric_row(
                "Past-query exact",
                no_metrics["past"]["exact"],
                fs_metrics["past"]["exact"],
            ),
            metric_row(
                "Future-query exact",
                no_metrics["future"]["exact"],
                fs_metrics["future"]["exact"],
            ),
            metric_row(
                "Balanced accuracy",
                no_metrics["balanced_accuracy"],
                fs_metrics["balanced_accuracy"],
            ),
        ]
    )
    curve_rows = []
    for no_row, fs_row in zip(no_fs["valid_curve"], fs["valid_curve"]):
        no_value = no_row["valid/accuracy"]
        fs_value = fs_row["valid/accuracy"]
        epoch = int(no_row["epoch"])
        curve_rows.append(
            "<tr>"
            f"<th>{epoch}</th><td><span class='bar nofs' style='width:{100*no_value:.2f}%'></span>{percent(no_value)}</td>"
            f"<td><span class='bar fs' style='width:{100*fs_value:.2f}%'></span>{percent(fs_value)}</td>"
            "</tr>"
        )

    rendered_cases = "".join(render_case(case) for case in cases[:10])
    output_dir = args.run_dir / "visualizations"
    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / "index.html"
    output.write_text(
        f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Native FutureSeed Directionality</title>
<style>
:root {{ color-scheme: light; font-family: Inter, ui-sans-serif, system-ui, sans-serif; color: #17202a; background: #f4f6f7; }}
* {{ box-sizing: border-box; }}
body {{ margin: 0; }}
header.hero {{ background: #16202a; color: #fff; padding: 28px max(24px, calc((100vw - 1180px)/2)); }}
h1 {{ margin: 0 0 8px; font-size: 32px; letter-spacing: 0; }}
.hero p {{ margin: 0; max-width: 920px; color: #d5dde5; line-height: 1.5; }}
main {{ max-width: 1180px; margin: 0 auto; padding: 24px; }}
h2 {{ margin: 32px 0 12px; font-size: 22px; }}
.claim {{ border-left: 5px solid #16805b; background: #eaf7f1; padding: 16px 18px; line-height: 1.55; }}
.numbers {{ display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 12px; margin: 18px 0; }}
.number {{ background: #fff; border: 1px solid #d6dde3; border-radius: 6px; padding: 16px; }}
.number strong {{ display: block; font-size: 28px; color: #126549; }}
.number span {{ color: #5e6b76; font-size: 13px; }}
table {{ width: 100%; border-collapse: collapse; background: #fff; }}
th, td {{ border: 1px solid #d9e0e5; padding: 9px 11px; text-align: right; }}
th:first-child {{ text-align: left; }}
thead th {{ background: #e9edf0; }}
.delta {{ color: #126549; font-weight: 700; }}
.curve td {{ position: relative; overflow: hidden; min-width: 200px; }}
.bar {{ position: absolute; inset: 3px auto 3px 3px; opacity: .22; border-radius: 3px; }}
.bar.nofs {{ background: #b34b4b; }} .bar.fs {{ background: #16805b; }}
.method {{ display: grid; grid-template-columns: 1fr 1fr; gap: 12px; }}
.method > div {{ background: #fff; border: 1px solid #d6dde3; padding: 14px; border-radius: 6px; }}
.case {{ background: #fff; border: 1px solid #ccd5dc; border-radius: 6px; margin: 14px 0; padding: 14px; }}
.case header {{ display: flex; justify-content: space-between; gap: 12px; align-items: baseline; margin-bottom: 10px; }}
.case h3 {{ margin: 0; font-size: 16px; }}
.case header span {{ color: #5d6973; font-size: 13px; }}
.sequence {{ display: grid; grid-template-columns: repeat(16, minmax(36px, 1fr)); gap: 3px; overflow-x: auto; }}
.token {{ min-width: 36px; height: 48px; border: 1px solid #cbd4db; padding: 3px; position: relative; text-align: center; }}
.token .pos {{ position: absolute; top: 2px; left: 3px; font-size: 8px; color: #66737d; }}
.token strong {{ display: block; margin-top: 8px; font-size: 13px; }}
.token .region {{ font-size: 8px; color: #66737d; }}
.token.future-query {{ background: #e8f1fb; }} .token.future-write {{ background: #eaf7f1; }}
.token.past-write {{ background: #f5f0de; }} .token.past-query {{ background: #f3eaf7; }}
.token.key {{ border-bottom: 3px solid #326ea8; }} .token.value {{ border-bottom: 3px solid #16805b; }}
.token.query {{ outline: 2px solid #b02f3a; outline-offset: -2px; }}
.case-results {{ margin-top: 10px; }}
.ok {{ color: #126549; background: #eaf7f1; font-weight: 700; }}
.bad {{ color: #9d2430; background: #fae9eb; font-weight: 700; }}
.foot {{ color: #596671; font-size: 13px; line-height: 1.5; margin: 24px 0; }}
code {{ font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: .9em; }}
@media (max-width: 760px) {{ .numbers, .method {{ grid-template-columns: 1fr; }} main {{ padding: 16px; }} h1 {{ font-size: 26px; }} .sequence {{ grid-template-columns: repeat(8, minmax(38px, 1fr)); }} .case header {{ display: block; }} }}
</style>
</head>
<body>
<header class="hero">
  <h1>Native FutureSeed Directionality</h1>
  <p>Matched official-FLA GDN2 models. The only variable is whether layer 1 starts from layer 0's normalized terminal recurrent state. No reverse scan, oracle, selector, repair, or task rule.</p>
</header>
<main>
  <div class="claim"><strong>Result:</strong> the causal control solves associations written before a query but remains at chance when the random value is written later. Native FutureSeed solves both directions from the same initialization and data.</div>
  <section class="numbers">
    <div class="number"><strong>{percent(no_metrics['future']['accuracy'])} to {percent(fs_metrics['future']['accuracy'])}</strong><span>Future-query accuracy</span></div>
    <div class="number"><strong>{comparison['future_accuracy_delta']*100:+.2f} pp</strong><span>FutureSeed future-direction gain</span></div>
    <div class="number"><strong>{percent(fs_metrics['future']['exact'])}</strong><span>Future-query exact retrieval</span></div>
  </section>

  <h2>Matched Endpoint</h2>
  <table><thead><tr><th>Metric</th><th>No FutureSeed</th><th>FutureSeed</th><th>Delta</th></tr></thead><tbody>{metric_rows}</tbody></table>

  <h2>Learning Curve</h2>
  <table class="curve"><thead><tr><th>Epoch</th><th>No FutureSeed overall</th><th>FutureSeed overall</th></tr></thead><tbody>{''.join(curve_rows)}</tbody></table>

  <h2>Causal Integrity</h2>
  <div class="method">
    <div><strong>No-FS future dependency</strong><br>{preflight['no_fs_future_dependency_max']:.6f}<br><small>Changing all later value tokens changes early future-query logits by exactly zero.</small></div>
    <div><strong>FS future dependency</strong><br>{preflight['fs_future_dependency_mean']:.6f}<br><small>The same perturbation reaches early queries only through cross-layer terminal-state seeding.</small></div>
    <div><strong>Scale-0 baseline diff</strong><br>{preflight['scale0_output_max_diff']:.6f}<br><small>The scale-0 wrapper is output-identical to the validated causal GDN2 baseline.</small></div>
    <div><strong>Gate gradient</strong><br>{preflight['future_seed_gate_grad_max']:.6g}<br><small>The FutureSeed route participates in end-to-end learning.</small></div>
  </div>

  <h2>Same-Sequence Cases</h2>
  <p class="foot">FQ: early future query. PW: past key/value write. PQ: later past query. FW: future key/value write. Blue underline marks keys, green underline marks values, and the red outline marks supervised future queries.</p>
  {rendered_cases}

  <p class="foot">Source <code>{html.escape((args.run_dir / 'source_HEAD.txt').read_text().strip())}</code>; official FLA <code>{html.escape(preflight['fla_sha'])}</code>; matched init hash <code>{html.escape(no_fs['init_hash'])}</code>. Raw fixed-budget wall time is not used as a speed claim because the first arm paid the one-time Triton compile cost.</p>
</main>
</body>
</html>
""",
        encoding="utf-8",
    )
    print(output)


if __name__ == "__main__":
    main()
