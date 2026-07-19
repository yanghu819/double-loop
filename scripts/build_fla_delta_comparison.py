#!/usr/bin/env python3
from __future__ import annotations

import argparse
import html
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


RUN_SPECS = [
    {
        "key": "gdn",
        "label": "GDN v1",
        "run": "runs/gdn-realbwd-official-sudoku-scale-fs-clean-20260626T0105-be2815b",
        "parameters": 1_306_864,
        "color": "#1f6f50",
    },
    {
        "key": "gdn2",
        "label": "FLA GDN2",
        "run": "runs/fla-gdn2-futureseed-sudoku-gate-20260719T042329Z-bf4b338",
        "parameters": 1_443_520,
        "color": "#b44b31",
    },
    {
        "key": "kda",
        "label": "FLA KDA",
        "run": "runs/fla-kda-futureseed-sudoku-gate-20260719T043622Z-bf4b338",
        "parameters": 1_253_056,
        "color": "#3567a8",
    },
]

LOG_RE = re.compile(
    r"step=(?P<step>\d+)\s+ce=(?P<ce>[-+0-9.eE]+).*?loop1=(?P<loop1>[-+0-9.eE]+)"
    r"\s+loop_last=(?P<loop_last>[-+0-9.eE]+)"
)


def read_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def find_result(run_dir: Path) -> Path:
    candidates = sorted((run_dir / "output").glob("futureseed_loop_seed*.json"))
    if not candidates:
        raise FileNotFoundError(f"No result JSON under {run_dir}")
    return candidates[-1]


def parse_train_curve(log_path: Path) -> List[Dict[str, float]]:
    rows: List[Dict[str, float]] = []
    with log_path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            match = LOG_RE.search(line)
            if match:
                rows.append({key: float(value) for key, value in match.groupdict().items()})
    return rows


def extract_same_hard_case(run_dir: Path) -> Optional[Dict[str, Any]]:
    path = run_dir / "output" / "case_bank" / "official_b56_64" / "cases.json"
    if not path.exists():
        return None
    payload = read_json(path)
    hard_cases = payload.get("selected", {}).get("hard_failure", [])
    case = next((item for item in hard_cases if item.get("batch_index") == 0), None)
    if not case:
        return None
    loops = case.get("loops", {})
    loop_rows = []
    for loop_index in range(1, 5):
        values = loops.get(f"loop{loop_index}", {})
        loop_rows.append(
            {
                "loop": loop_index,
                "wrong_count": values.get("wrong_count"),
                "changed_from_previous": values.get("changed_from_previous"),
                "conflict_unit_count": values.get("conflict_unit_count"),
            }
        )
    return {
        "batch_index": 0,
        "holes": case.get("holes"),
        "stem": case.get("stem"),
        "loops": loop_rows,
    }


def extract_run(repo: Path, spec: Dict[str, Any]) -> Dict[str, Any]:
    run_dir = repo / spec["run"]
    payload = read_json(find_result(run_dir))
    args = payload["args"]
    metrics = payload["metrics"]
    train = metrics["train"]
    eval_clean = metrics["eval_clean"]
    loop_rows = []
    for loop_index in range(1, int(args["max_loops"]) + 1):
        values = eval_clean[f"loop{loop_index}"]
        loop_rows.append(
            {
                "loop": loop_index,
                "exact": float(values["label_exact"]),
                "blank_acc": float(values["blank_acc"]),
            }
        )

    final_loop = loop_rows[-1]
    fs_metrics = eval_clean.get(f"loop{final_loop['loop']}/future_seed", {})
    official_rows = []
    official = metrics.get("official_eval_by_blank_range", {})
    for key in ("b46_50", "b51_55", "b56_64"):
        values = official.get(key)
        if not values:
            continue
        loop_values = values["eval_clean"][f"loop{final_loop['loop']}"]
        official_rows.append(
            {
                "range": key.removeprefix("b").replace("_", "-"),
                "exact": float(loop_values["label_exact"]),
                "blank_acc": float(loop_values["blank_acc"]),
            }
        )

    return {
        **spec,
        "git_sha": read_json(run_dir / "config.json").get("git_sha"),
        "backbone": args.get("backbone"),
        "steps": int(args["steps"]),
        "batch": int(args["batch"]),
        "d_model": int(args["d_model"]),
        "layers": int(args["layers"]),
        "heads": int(args["heads"]),
        "head_dim": int(args["head_dim"]),
        "max_loops": int(args["max_loops"]),
        "train_ce": float(train["train_ce_loss"]),
        "train_sec": float(train["train_sec"]),
        "peak_memory_mb": float(train["cuda_max_memory_allocated_mb"]),
        "loop_rows": loop_rows,
        "loop1_exact": loop_rows[0]["exact"],
        "final_exact": final_loop["exact"],
        "exact_gain": final_loop["exact"] - loop_rows[0]["exact"],
        "loop1_blank": loop_rows[0]["blank_acc"],
        "final_blank": final_loop["blank_acc"],
        "blank_gain": final_loop["blank_acc"] - loop_rows[0]["blank_acc"],
        "fs_gate": fs_metrics.get("fs_gate_mean"),
        "fs_state_norm": fs_metrics.get("fs_state_norm"),
        "train_curve": parse_train_curve(run_dir / "logs" / "run.log"),
        "official_rows": official_rows,
        "same_hard_case": extract_same_hard_case(run_dir),
    }


def fmt(value: Any, digits: int = 4) -> str:
    if value is None:
        return "-"
    return f"{float(value):.{digits}f}"


def pct(value: Any, digits: int = 2) -> str:
    if value is None:
        return "-"
    return f"{100.0 * float(value):.{digits}f}%"


def esc(value: Any) -> str:
    return html.escape(str(value))


def svg_line_chart(
    title: str,
    x_labels: Iterable[Any],
    series: List[Dict[str, Any]],
    *,
    y_min: float,
    y_max: float,
    percent: bool = False,
) -> str:
    labels = list(x_labels)
    width, height = 760, 310
    left, right, top, bottom = 64, 22, 42, 50
    plot_w, plot_h = width - left - right, height - top - bottom

    def x_pos(index: int) -> float:
        return left + (plot_w * index / max(1, len(labels) - 1))

    def y_pos(value: float) -> float:
        clipped = min(max(float(value), y_min), y_max)
        return top + (y_max - clipped) * plot_h / (y_max - y_min)

    parts = [
        f'<svg class="chart" viewBox="0 0 {width} {height}" role="img" aria-label="{esc(title)}">',
        f'<text x="{left}" y="24" class="chart-title">{esc(title)}</text>',
    ]
    for tick in range(5):
        value = y_min + (y_max - y_min) * tick / 4
        y = y_pos(value)
        label = f"{value * 100:.1f}%" if percent else f"{value:.2f}"
        parts.append(f'<line x1="{left}" y1="{y:.1f}" x2="{width-right}" y2="{y:.1f}" class="grid"/>')
        parts.append(f'<text x="{left-10}" y="{y+4:.1f}" text-anchor="end" class="tick">{label}</text>')
    for index, label in enumerate(labels):
        x = x_pos(index)
        parts.append(f'<text x="{x:.1f}" y="{height-20}" text-anchor="middle" class="tick">{esc(label)}</text>')

    for item in series:
        values = item["values"]
        points = " ".join(f"{x_pos(i):.1f},{y_pos(v):.1f}" for i, v in enumerate(values))
        color = item["color"]
        parts.append(f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="3"/>')
        for index, value in enumerate(values):
            parts.append(f'<circle cx="{x_pos(index):.1f}" cy="{y_pos(value):.1f}" r="4" fill="{color}"/>')
    legend_x = left
    for item in series:
        parts.append(f'<rect x="{legend_x}" y="{height-8}" width="14" height="4" fill="{item["color"]}"/>')
        parts.append(f'<text x="{legend_x+20}" y="{height-3}" class="legend">{esc(item["label"])}</text>')
        legend_x += 150
    parts.append("</svg>")
    return "".join(parts)


def table(headers: List[str], rows: Iterable[List[str]]) -> str:
    body = []
    for row in rows:
        body.append("<tr>" + "".join(f"<td>{cell}</td>" for cell in row) + "</tr>")
    head = "".join(f"<th>{esc(header)}</th>" for header in headers)
    return f"<div class=\"table-wrap\"><table><thead><tr>{head}</tr></thead><tbody>{''.join(body)}</tbody></table></div>"


def run_link(run_rel: str, suffix: str, label: str) -> str:
    href = Path("..") / Path(run_rel).name / suffix
    return f'<a href="{esc(href.as_posix())}">{esc(label)}</a>'


def build_html(payload: Dict[str, Any]) -> str:
    rows = payload["runs"]
    by_key = {row["key"]: row for row in rows}
    gdn = by_key["gdn"]
    gdn2 = by_key["gdn2"]
    kda = by_key["kda"]

    metric_rows = []
    for row in rows:
        metric_rows.append(
            [
                f'<span class="swatch" style="background:{row["color"]}"></span>{esc(row["label"])}',
                f'{row["parameters"] / 1e6:.3f}M',
                fmt(row["train_ce"]),
                pct(row["loop1_exact"]),
                pct(row["final_exact"]),
                pct(row["exact_gain"]),
                pct(row["final_blank"]),
                f'{row["peak_memory_mb"] / 1024:.2f} GiB',
                f'{row["train_sec"]:.1f}s',
            ]
        )

    ce_chart = svg_line_chart(
        "Training CE: simple GDN learns fastest",
        [100, 200, 300, 400, 500, 600],
        [
            {"label": row["label"], "color": row["color"], "values": [item["ce"] for item in row["train_curve"]]}
            for row in rows
        ],
        y_min=1.0,
        y_max=1.95,
    )
    exact_chart = svg_line_chart(
        "Full-board exact by loop",
        [1, 2, 3, 4],
        [
            {"label": row["label"], "color": row["color"], "values": [item["exact"] for item in row["loop_rows"]]}
            for row in rows
        ],
        y_min=0.0,
        y_max=0.012,
        percent=True,
    )
    blank_chart = svg_line_chart(
        "Blank-cell accuracy by loop",
        [1, 2, 3, 4],
        [
            {"label": row["label"], "color": row["color"], "values": [item["blank_acc"] for item in row["loop_rows"]]}
            for row in rows
        ],
        y_min=0.43,
        y_max=0.51,
        percent=True,
    )

    official_rows = []
    for index, blank_range in enumerate(("46-50", "51-55", "56-64")):
        gdn2_values = gdn2["official_rows"][index]
        kda_values = kda["official_rows"][index]
        official_rows.append(
            [
                esc(blank_range),
                pct(gdn2_values["exact"]),
                pct(gdn2_values["blank_acc"]),
                pct(kda_values["exact"]),
                pct(kda_values["blank_acc"]),
            ]
        )

    kernel = payload["kernel"]
    kernel_rows = [
        [
            "GDN2",
            fmt(kernel["gdn2"]["output_max_abs"], 6),
            fmt(kernel["gdn2"]["state_max_abs"], 6),
            fmt(kernel["gdn2"]["gradient_max_abs"], 6),
            fmt(kernel["gdn2"]["initial_state_grad_norm"], 6),
            f'{kernel["gdn2"]["forward_backward_ms"]:.3f}ms',
        ],
        [
            "KDA",
            fmt(kernel["kda"]["output_max_abs"], 6),
            fmt(kernel["kda"]["state_max_abs"], 6),
            fmt(kernel["kda"]["gradient_max_abs"], 6),
            fmt(kernel["kda"]["initial_state_grad_norm"], 6),
            f'{kernel["kda"]["forward_backward_ms"]:.3f}ms',
        ],
    ]

    hard_links = {}
    for row in (gdn2, kda):
        case = row["same_hard_case"]
        hard_links[row["key"]] = (Path("..") / Path(row["run"]).name / "output" / "case_bank" / "official_b56_64" / f'{case["stem"]}.html').as_posix()

    return f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <link rel="icon" href="data:,">
  <title>FutureSeed Linear Attention Comparison</title>
  <style>
    :root {{ color-scheme: light; --ink:#18212b; --muted:#5f6b76; --line:#ccd4dc; --paper:#ffffff; --band:#f4f6f8; --good:#1f6f50; --bad:#a33d2c; --accent:#3567a8; }}
    * {{ box-sizing:border-box; }}
    body {{ margin:0; color:var(--ink); background:var(--paper); font-family:Inter,ui-sans-serif,system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif; letter-spacing:0; }}
    header, section {{ width:100%; border-bottom:1px solid var(--line); }}
    .inner {{ max-width:1180px; margin:0 auto; padding:28px 24px; }}
    h1 {{ margin:0 0 8px; font-size:30px; line-height:1.2; }}
    h2 {{ margin:0 0 14px; font-size:21px; }}
    h3 {{ margin:0 0 8px; font-size:16px; }}
    p {{ margin:8px 0; line-height:1.58; }}
    .meta {{ color:var(--muted); font-size:13px; }}
    .decision {{ background:#edf5ef; border-left:5px solid var(--good); padding:16px 18px; margin-top:18px; }}
    .decision strong {{ display:block; margin-bottom:5px; }}
    .band {{ background:var(--band); }}
    .grid-2 {{ display:grid; grid-template-columns:repeat(2,minmax(0,1fr)); gap:18px; }}
    .metric-strip {{ display:grid; grid-template-columns:repeat(3,minmax(0,1fr)); gap:1px; background:var(--line); border:1px solid var(--line); margin:18px 0; }}
    .metric {{ background:var(--paper); padding:16px; min-height:112px; }}
    .metric b {{ display:block; font-size:24px; margin:6px 0; }}
    .metric small {{ color:var(--muted); }}
    .table-wrap {{ overflow-x:auto; border:1px solid var(--line); }}
    table {{ border-collapse:collapse; width:100%; font-size:13px; background:var(--paper); }}
    th,td {{ padding:9px 10px; border-bottom:1px solid var(--line); text-align:right; white-space:nowrap; }}
    th:first-child,td:first-child {{ text-align:left; }}
    th {{ background:#e9edf1; color:#303b46; }}
    tr:last-child td {{ border-bottom:0; }}
    .swatch {{ display:inline-block; width:11px; height:11px; margin-right:7px; vertical-align:-1px; }}
    .chart {{ width:100%; min-height:280px; border:1px solid var(--line); background:var(--paper); }}
    .chart-title {{ font-size:16px; font-weight:650; fill:var(--ink); }}
    .grid {{ stroke:#d9e0e6; stroke-width:1; }}
    .tick,.legend {{ font-size:11px; fill:#596673; }}
    .case-grid {{ display:grid; grid-template-columns:repeat(2,minmax(0,1fr)); gap:18px; }}
    .case-pane {{ min-width:0; }}
    iframe {{ width:100%; height:760px; border:1px solid var(--line); background:#fff; }}
    .links {{ display:flex; flex-wrap:wrap; gap:10px 18px; margin-top:12px; }}
    a {{ color:#205b97; text-decoration-thickness:1px; text-underline-offset:2px; }}
    code {{ font-family:ui-monospace,SFMono-Regular,Menlo,monospace; font-size:12px; }}
    .bad {{ color:var(--bad); font-weight:650; }}
    .good {{ color:var(--good); font-weight:650; }}
    @media (max-width:800px) {{ .grid-2,.case-grid,.metric-strip {{ grid-template-columns:1fr; }} .inner {{ padding:22px 15px; }} iframe {{ height:680px; }} }}
  </style>
</head>
<body>
  <header><div class="inner">
    <div class="meta">P-LA-001 · GPU1 A800 · generated {esc(payload["generated_at_utc"])} · official FLA <code>fe8fce9</code></div>
    <h1>FutureSeed 换不同 Linear Attention，谁更有效？</h1>
    <p>固定数据、600 steps、D128/L6、loop4 和每个 loop 的 CE，只替换 recurrent state update。FutureSeed 仍是上一层终态注入下一层初态，不是右到左扫描。</p>
    <div class="decision"><strong>结论：原 GDN v1 胜出；GDN2 次之；KDA 最弱。</strong>更细粒度的 memory edit 在这份预算下没有提高全局闭合，反而学得更慢。按预设规则，不把 GDN2/KDA 升到大模型；回到已证明仍有斜率的 clean data/compute scaling。</div>
  </div></header>

  <section class="band"><div class="inner">
    <h2>最重要的三件事</h2>
    <div class="metric-strip">
      <div class="metric"><small>600-step full-board exact</small><b>1.07% / 0.20% / 0.10%</b><span>GDN v1 / GDN2 / KDA</span></div>
      <div class="metric"><small>loop1 → loop4 exact gain</small><b>+0.98% / +0.20% / 0.00%</b><span>复杂状态没有产生更强的 loop 修正</span></div>
      <div class="metric"><small>hard 51–64 blanks</small><b>0 exact</b><span>GDN2 和 KDA 都没跨过 hard transition</span></div>
    </div>
    {table(["backbone","params","CE@600","loop1 exact","loop4 exact","loop gain","loop4 blank","peak alloc","observed wall"], metric_rows)}
    <p class="meta">Observed wall 包含一次性 Triton 编译/自动调优，不是纯稳态吞吐。质量没有持平，因此显存更低不能算效率胜利。</p>
  </div></section>

  <section><div class="inner">
    <h2>学习和 loop 动力学</h2>
    <div class="grid-2">{ce_chart}{exact_chart}</div>
    <div style="margin-top:18px">{blank_chart}</div>
    <p><span class="bad">关键失败：</span>GDN2 的同一 hard case 是 <code>23→20→20→20</code> 个错格；KDA 是 <code>23→22→21→21</code>。两者在 loop2 附近做完少量修改，后续基本冻结。更复杂的 state update 并没有自动变成更强的反复修错。</p>
  </div></section>

  <section class="band"><div class="inner">
    <h2>难度分段</h2>
    {table(["blank range","GDN2 exact","GDN2 blank","KDA exact","KDA blank"], official_rows)}
    <p>GDN2 在 46–50 blanks 的较容易区间能解出 <strong>7.32%</strong>，KDA 只有 <strong>1.27%</strong>；但进入 51–55 后两者 exact 都归零。这说明 GDN2 确实比 KDA 更好优化，但仍没有解决真正的全局约束闭合。</p>
  </div></section>

  <section><div class="inner">
    <h2>同一个 56-blank case：loop1–4 原地对比</h2>
    <div class="case-grid">
      <div class="case-pane"><h3>FLA GDN2 · 23→20→20→20 wrong</h3><iframe loading="lazy" onload="this.style.height='760px';this.style.height=(this.contentDocument.documentElement.scrollHeight+4)+'px'" src="{esc(hard_links['gdn2'])}" title="GDN2 hard case"></iframe></div>
      <div class="case-pane"><h3>FLA KDA · 23→22→21→21 wrong</h3><iframe loading="lazy" onload="this.style.height='760px';this.style.height=(this.contentDocument.documentElement.scrollHeight+4)+'px'" src="{esc(hard_links['kda'])}" title="KDA hard case"></iframe></div>
    </div>
    <div class="links">
      {run_link(gdn2['run'], 'visualizations/index.html', '打开 GDN2 总览')}
      {run_link(kda['run'], 'visualizations/index.html', '打开 KDA 总览')}
      {run_link(gdn['run'], 'visualizations/index.html', '打开 GDN v1 历史基线')}
    </div>
  </div></section>

  <section class="band"><div class="inner">
    <h2>CUDA kernel 是否可信</h2>
    {table(["kernel","output max abs","state max abs","gradient max abs","h0 grad norm","hot fwd+bwd"], kernel_rows)}
    <p>两个 official FLA kernel 都通过了 Torch reference 的 forward、terminal state、backward、初始状态梯度和分段序列连续性测试；三层 FutureSeed stack 也有非零输入梯度。KDA 在测试 shape 下的 hot forward+backward 比 GDN2 慢约 <strong>59%</strong>。</p>
  </div></section>

  <section><div class="inner">
    <h2>机制判断和下一步</h2>
    <p><strong>判负的假设：</strong>hard Sudoku 的主要瓶颈不是“每个 state channel 缺少更细的 erase/write/decay 控制”。如果它是，预期顺序应为 GDN2 &gt; KDA &gt; GDN v1；实际是 GDN v1 &gt; GDN2 &gt; KDA。</p>
    <p><strong>保留的发现：</strong>原生 FutureSeed 能无损迁移到 GDN2 和 KDA 的矩阵状态，kernel 与梯度都正确；因此 FutureSeed 不是绑定某一个 RWKV/GDN 方程的 hack。但这轮没有 no-FS 新对照，不能把两条新 backbone 的 opening 单独归因给 FutureSeed。</p>
    <p><strong>决策：</strong>停止 GDN2/KDA 的 seed、LR、loss、width 补表，也不立刻上 D224。恢复 P-SCALE-034 的原 GDN v1 clean full-diversity 长训练到 step20000，因为那条曲线在 step16000 仍有真实 exact 正斜率。</p>
  </div></section>
  <script>
    const fitCaseFrames = () => document.querySelectorAll("iframe").forEach((frame) => {{
      const doc = frame.contentDocument;
      if (doc) {{
        frame.style.height = "760px";
        frame.style.height = `${{doc.documentElement.scrollHeight + 4}}px`;
      }}
    }});
    window.addEventListener("load", fitCaseFrames);
    window.addEventListener("resize", () => requestAnimationFrame(fitCaseFrames));
  </script>
</body>
</html>
"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("runs/fla-gdn2-kda-comparison-20260719-bf4b338"),
    )
    args = parser.parse_args()
    repo = args.repo.resolve()
    out_dir = args.out_dir if args.out_dir.is_absolute() else repo / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = [extract_run(repo, spec) for spec in RUN_SPECS]
    kernel_dir = repo / "runs/fla-delta-kernel-gate-20260719-986e725"
    gdn2_kernel = read_json(kernel_dir / "gdn2-reference-adapter.json")
    kda_kernel = read_json(kernel_dir / "kda-all.json")
    kernel = {
        "gdn2": {
            **gdn2_kernel["gdn2_reference"],
            **gdn2_kernel["gdn2_adapter"],
        },
        "kda": {
            **kda_kernel["kda_reference"],
            **kda_kernel["kda_adapter"],
        },
    }
    payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "plan_id": "P-LA-001",
        "source_sha": "bf4b33820599f2f5782c128ac55b5a867d7e9d3d",
        "official_fla_sha": "fe8fce9fc6984f22905f54cfa885dce1502baf26",
        "runs": rows,
        "kernel": kernel,
        "decision": "Stop GDN2/KDA scaling; neither beat the matched GDN v1 quality gate.",
    }
    with (out_dir / "comparison.json").open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    (out_dir / "index.html").write_text(build_html(payload), encoding="utf-8")
    (out_dir / "README.md").write_text(
        "# FLA GDN2/KDA Comparison\n\n"
        "Open `index.html` for the matched 600-step metrics, CUDA checks, loop curves, "
        "and same-case GDN2/KDA visualizations.\n",
        encoding="utf-8",
    )
    print(out_dir)


if __name__ == "__main__":
    main()
