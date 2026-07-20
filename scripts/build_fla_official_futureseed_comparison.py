#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import html
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SOURCE_SHA = "c3342b8326a6e00e012cd6972d8c0e295d82c0b3"
FLA_SOURCE_SHA = "fe8fce9fc6984f22905f54cfa885dce1502baf26"
WHEEL_SHA256 = "65f57bf2aa937991fc497bd63f42883d263ead8646f21f2492735ccddd82d0eb"

RUNS = [
    {
        "key": "gdn",
        "label": "GDN",
        "color": "#167052",
        "parameters": 5_980_468,
        "run": "runs/fla-official-gdn-fs-d192l10-step500-unifiedeval-20260720T1435Z-c3342b8",
        "train_run": "runs/fla-official-gdn-fs-d192l10-s1500-20260720T0825Z-c3342b8",
        "memory_run": "runs/fla-official-gdn-trainmem-step501-20260720T1455Z-c3342b8",
    },
    {
        "key": "kda",
        "label": "KDA",
        "color": "#2f65a7",
        "parameters": 5_851_768,
        "run": "runs/fla-official-kda-fs-d192l10-s500-20260720T0908Z-c3342b8",
        "train_run": "runs/fla-official-kda-fs-d192l10-s500-20260720T0908Z-c3342b8",
    },
    {
        "key": "gdn2",
        "label": "GDN2",
        "color": "#b34934",
        "parameters": 6_946_168,
        "run": "runs/fla-official-gdn2-fs-d192l10-s500-clean-20260720T1034Z-c3342b8",
        "train_run": "runs/fla-official-gdn2-fs-d192l10-s500-clean-20260720T1034Z-c3342b8",
    },
]

LOG_RE = re.compile(
    r"step=(?P<step>\d+)\s+ce=(?P<ce>[-+0-9.eE]+).*?"
    r"loop1=(?P<loop1>[-+0-9.eE]+)\s+loop_last=(?P<loop_last>[-+0-9.eE]+)"
)


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def result_path(run_dir: Path) -> Path:
    candidates = sorted((run_dir / "output").glob("futureseed_loop_seed*.json"))
    if not candidates:
        raise FileNotFoundError(f"No result JSON under {run_dir}")
    return candidates[-1]


def train_curve(log_path: Path) -> list[dict[str, float]]:
    rows: list[dict[str, float]] = []
    with log_path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            match = LOG_RE.search(line)
            if match:
                rows.append({key: float(value) for key, value in match.groupdict().items()})
    return rows


def primary_case(run_dir: Path) -> dict[str, Any]:
    case_path = run_dir / "output" / "futureseed_loop_case_seed52.html"
    text = case_path.read_text(encoding="utf-8")
    puzzle_match = re.search(
        r'<div class="panel"><h3>puzzle</h3>(.*?)<div class="panel"><h3>solution</h3>',
        text,
        flags=re.DOTALL,
    )
    if not puzzle_match:
        raise ValueError(f"Could not find puzzle in {case_path}")
    wrong_by_loop: dict[str, int] = {}
    for chunk in text.split('<div class="panel"><h3>loop ')[1:]:
        loop_match = re.match(r"(\d+)</h3>", chunk)
        if loop_match:
            wrong_by_loop[f"loop{int(loop_match.group(1))}"] = chunk.count('class="cell wrong"')
    return {
        "html": case_path.relative_to(run_dir.parents[1]).as_posix(),
        "puzzle_sha256": hashlib.sha256(puzzle_match.group(1).encode("utf-8")).hexdigest(),
        "wrong_by_loop": wrong_by_loop,
    }


def extract_run(repo: Path, spec: dict[str, Any]) -> dict[str, Any]:
    run_dir = repo / spec["run"]
    train_dir = repo / spec["train_run"]
    payload = read_json(result_path(run_dir))
    args = payload["args"]
    metrics = payload["metrics"]
    train = metrics["train"]
    checkpoint = read_json(train_dir / "output" / "checkpoint_eval_step000500.json")
    loop_rows = []
    for loop in range(1, int(args["max_loops"]) + 1):
        row = metrics["eval_clean"][f"loop{loop}"]
        loop_rows.append(
            {
                "loop": loop,
                "exact": float(row["label_exact"]),
                "blank_acc": float(row["blank_acc"]),
            }
        )
    ranges: dict[str, dict[str, float]] = {}
    for key, value in metrics["official_eval_by_blank_range"].items():
        loop1 = value["eval_clean"]["loop1"]
        final = value["eval_clean"][f"loop{int(args['max_loops'])}"]
        ranges[key] = {
            "loop1_exact": float(loop1["label_exact"]),
            "loop1_blank_acc": float(loop1["blank_acc"]),
            "final_exact": float(final["label_exact"]),
            "final_blank_acc": float(final["blank_acc"]),
        }
    checkpoint53 = checkpoint["eval_by_holes"]["holes53"]["eval_clean"]["loop5"]

    peak_allocated = train.get("cuda_max_memory_allocated_mb")
    peak_reserved = train.get("cuda_max_memory_reserved_mb")
    memory_run = spec.get("memory_run")
    if memory_run:
        memory_dir = repo / memory_run
        if memory_dir.exists():
            memory_payload = read_json(result_path(memory_dir))
            memory_train = memory_payload["metrics"]["train"]
            peak_allocated = memory_train.get("cuda_max_memory_allocated_mb")
            peak_reserved = memory_train.get("cuda_max_memory_reserved_mb")

    final = loop_rows[-1]
    return {
        **spec,
        "git_sha": read_json(run_dir / "config.json")["git_sha"],
        "backbone": args["backbone"],
        "official_class": train["fla_runtime"]["layers"][0]["class"],
        "steps": int(args["steps"]),
        "effective_batch": int(args["batch"]) * int(args["grad_accum_steps"]),
        "d_model": int(args["d_model"]),
        "layers": int(args["layers"]),
        "heads": int(args["heads"]),
        "head_dim": int(args["head_dim"]),
        "max_loops": int(args["max_loops"]),
        "train_ce": float(checkpoint["train"]["ce_loss"]),
        "train_loop1_ce": float(checkpoint["train"]["loop1_loss"]),
        "train_loop5_ce": float(checkpoint["train"]["loop_last_loss"]),
        "train_wall_sec": float(checkpoint["elapsed_sec"]),
        "peak_allocated_mb": None if peak_allocated is None else float(peak_allocated),
        "peak_reserved_mb": None if peak_reserved is None else float(peak_reserved),
        "checkpoint53_exact": float(checkpoint53["label_exact"]),
        "checkpoint53_blank_acc": float(checkpoint53["blank_acc"]),
        "loop_rows": loop_rows,
        "loop_gain_exact": final["exact"] - loop_rows[0]["exact"],
        "loop_gain_blank": final["blank_acc"] - loop_rows[0]["blank_acc"],
        "ranges": ranges,
        "train_curve": train_curve(train_dir / "logs" / "run.log"),
        "case": primary_case(run_dir),
    }


def esc(value: Any) -> str:
    return html.escape(str(value))


def pct(value: float, digits: int = 2) -> str:
    return f"{100.0 * value:.{digits}f}%"


def fmt(value: float, digits: int = 4) -> str:
    return f"{value:.{digits}f}"


def table(headers: list[str], rows: list[list[str]]) -> str:
    head = "".join(f"<th>{esc(item)}</th>" for item in headers)
    body = "".join(
        "<tr>"
        + "".join(f'<td data-label="{esc(headers[index])}">{item}</td>' for index, item in enumerate(row))
        + "</tr>"
        for row in rows
    )
    return f'<div class="table-wrap"><table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>'


def line_chart(
    title: str,
    labels: list[Any],
    series: list[dict[str, Any]],
    *,
    y_min: float,
    y_max: float,
    percent: bool = False,
) -> str:
    width, height = 760, 310
    left, right, top, bottom = 66, 24, 42, 52
    plot_w, plot_h = width - left - right, height - top - bottom

    def x_pos(index: int) -> float:
        return left + plot_w * index / max(1, len(labels) - 1)

    def y_pos(value: float) -> float:
        clipped = min(max(value, y_min), y_max)
        return top + (y_max - clipped) * plot_h / (y_max - y_min)

    parts = [
        f'<svg class="chart" viewBox="0 0 {width} {height}" role="img" aria-label="{esc(title)}">',
        f'<text x="{left}" y="24" class="chart-title">{esc(title)}</text>',
    ]
    for tick in range(5):
        value = y_min + (y_max - y_min) * tick / 4
        y = y_pos(value)
        label = f"{value * 100:.1f}%" if percent else f"{value:.2f}"
        parts.append(f'<line x1="{left}" y1="{y:.1f}" x2="{width-right}" y2="{y:.1f}" class="grid-line"/>')
        parts.append(f'<text x="{left-10}" y="{y+4:.1f}" text-anchor="end" class="tick">{label}</text>')
    for index, label in enumerate(labels):
        x = x_pos(index)
        parts.append(f'<text x="{x:.1f}" y="{height-23}" text-anchor="middle" class="tick">{esc(label)}</text>')
    for item in series:
        values = item["values"]
        points = " ".join(f"{x_pos(i):.1f},{y_pos(float(v)):.1f}" for i, v in enumerate(values))
        parts.append(f'<polyline points="{points}" fill="none" stroke="{item["color"]}" stroke-width="3"/>')
        for index, value in enumerate(values):
            parts.append(f'<circle cx="{x_pos(index):.1f}" cy="{y_pos(float(value)):.1f}" r="4" fill="{item["color"]}"/>')
    legend_x = left
    for item in series:
        parts.append(f'<rect x="{legend_x}" y="{height-10}" width="14" height="4" fill="{item["color"]}"/>')
        parts.append(f'<text x="{legend_x+20}" y="{height-5}" class="legend">{esc(item["label"])}</text>')
        legend_x += 140
    parts.append("</svg>")
    return "".join(parts)


def relative_link(out_dir: Path, repo: Path, target: str) -> str:
    return Path("../" + Path(target).name).as_posix() if target.startswith("runs/") else Path(target).as_posix()


def build_html(repo: Path, out_dir: Path, payload: dict[str, Any]) -> str:
    runs = payload["runs"]
    provenance = payload["provenance"]
    by_key = {row["key"]: row for row in runs}
    gdn, kda, gdn2 = by_key["gdn"], by_key["kda"], by_key["gdn2"]

    quality_rows = []
    for row in runs:
        final = row["loop_rows"][-1]
        quality_rows.append(
            [
                f'<span class="swatch" style="background:{row["color"]}"></span>{esc(row["label"])}',
                f'{row["parameters"] / 1e6:.3f}M',
                fmt(row["train_ce"]),
                pct(row["checkpoint53_exact"]),
                pct(final["exact"]),
                pct(final["blank_acc"]),
                pct(row["loop_gain_exact"]),
                f'{row["train_wall_sec"] / 60:.1f} min',
                "-" if row["peak_allocated_mb"] is None else f'{row["peak_allocated_mb"] / 1024:.2f} GiB',
            ]
        )

    range_rows = []
    for key, label in (("b46_50", "46-50"), ("b51_55", "51-55"), ("b56_64", "56-64")):
        row = [esc(label)]
        for run in runs:
            values = run["ranges"][key]
            row.extend([pct(values["final_exact"]), pct(values["final_blank_acc"])])
        range_rows.append(row)

    kernel_rows = []
    gate_keys = {"gdn": "fla_gdn", "kda": "kda", "gdn2": "gdn2"}
    for run in runs:
        prefix = gate_keys[run["key"]]
        reference = provenance["gate"][f"{prefix}_reference"]
        adapter = provenance["gate"][f"{prefix}_adapter"]
        kernel_rows.append(
            [
                esc(run["label"]),
                esc(adapter["official_layer_class"]),
                esc(adapter["official_chunk_autograd_node"]),
                fmt(reference["output_max_abs"], 6),
                fmt(reference["state_max_abs"], 6),
                fmt(reference["gradient_max_abs"], 6),
                fmt(adapter["initial_state_grad_norm"], 6),
                "V,K" if adapter["state_v_first"] else "K,V",
            ]
        )

    curve_steps = [int(row["step"]) for row in gdn["train_curve"]]
    ce_chart = line_chart(
        "Train CE by optimizer step",
        curve_steps,
        [{"label": row["label"], "color": row["color"], "values": [point["ce"] for point in row["train_curve"]]} for row in runs],
        y_min=0.95,
        y_max=1.72,
    )
    loop_chart = line_chart(
        "53-blank full-board exact by loop",
        [1, 2, 3, 4, 5],
        [{"label": row["label"], "color": row["color"], "values": [point["exact"] for point in row["loop_rows"]]} for row in runs],
        y_min=0.0,
        y_max=0.024,
        percent=True,
    )
    blank_chart = line_chart(
        "53-blank blank-cell accuracy by loop",
        [1, 2, 3, 4, 5],
        [{"label": row["label"], "color": row["color"], "values": [point["blank_acc"] for point in row["loop_rows"]]} for row in runs],
        y_min=0.48,
        y_max=0.51,
        percent=True,
    )

    case_buttons = []
    case_frames = []
    case_rows = []
    for index, row in enumerate(runs):
        active = " active" if index == 0 else ""
        pressed = "true" if index == 0 else "false"
        case_buttons.append(
            f'<button class="case-tab{active}" data-target="case-{row["key"]}" '
            f'aria-pressed="{pressed}">{esc(row["label"])}</button>'
        )
        src = (Path("..") / Path(row["run"]).name / "output" / "futureseed_loop_case_seed52.html").as_posix()
        case_frames.append(
            f'<iframe id="case-{row["key"]}" class="case-frame{active}" src="{esc(src)}" '
            f'title="{esc(row["label"])} same Sudoku case" loading="lazy"></iframe>'
        )
        case_rows.append(
            [esc(row["label"])] + [str(row["case"]["wrong_by_loop"].get(f"loop{loop}", "-")) for loop in range(1, 6)]
        )

    source_hash_count = len(provenance["source_file_hashes"])
    return f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <link rel="icon" href="data:,">
  <title>Official FLA FutureSeed Comparison</title>
  <style>
    :root {{ color-scheme:light; --ink:#17212b; --muted:#5f6c78; --line:#cbd3da; --band:#f3f5f7; --paper:#fff; --ok:#167052; --warn:#9b3d2f; }}
    * {{ box-sizing:border-box; }}
    body {{ margin:0; color:var(--ink); background:var(--paper); font-family:Inter,ui-sans-serif,system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif; letter-spacing:0; }}
    header,section {{ width:100%; border-bottom:1px solid var(--line); }}
    .inner {{ width:min(1220px,100%); margin:0 auto; padding:26px 22px; }}
    h1 {{ margin:0 0 8px; font-size:28px; line-height:1.25; }}
    h2 {{ margin:0 0 13px; font-size:20px; }}
    p {{ margin:7px 0; line-height:1.6; }}
    .meta {{ color:var(--muted); font-size:13px; }}
    .band {{ background:var(--band); }}
    .verdict {{ margin-top:16px; padding:14px 16px; border-left:5px solid var(--ok); background:#edf5f1; }}
    .verdict strong {{ display:block; margin-bottom:4px; }}
    .metric-strip {{ display:grid; grid-template-columns:repeat(3,minmax(0,1fr)); border:1px solid var(--line); background:var(--line); gap:1px; margin:17px 0; }}
    .metric {{ background:#fff; min-height:105px; padding:15px; }}
    .metric b {{ display:block; margin:5px 0; font-size:23px; }}
    .metric small {{ color:var(--muted); }}
    .table-wrap {{ max-width:100%; overflow-x:auto; border:1px solid var(--line); background:#fff; -webkit-overflow-scrolling:touch; }}
    table {{ width:100%; border-collapse:collapse; font-size:13px; }}
    th,td {{ padding:9px 10px; border-bottom:1px solid var(--line); text-align:right; white-space:nowrap; }}
    th:first-child,td:first-child {{ text-align:left; }}
    th {{ background:#e8edf1; color:#34414c; }}
    tr:last-child td {{ border-bottom:0; }}
    .swatch {{ display:inline-block; width:11px; height:11px; margin-right:7px; vertical-align:-1px; }}
    .charts {{ display:grid; grid-template-columns:repeat(2,minmax(0,1fr)); gap:16px; }}
    .chart {{ width:100%; min-height:270px; border:1px solid var(--line); background:#fff; }}
    .chart-title {{ font-size:16px; font-weight:650; fill:var(--ink); }}
    .grid-line {{ stroke:#dbe1e6; stroke-width:1; }}
    .tick,.legend {{ font-size:11px; fill:#596774; }}
    .case-tabs {{ display:flex; gap:4px; margin:14px 0 10px; }}
    .case-tab {{ border:1px solid #aeb9c3; background:#fff; color:#34414c; padding:8px 13px; font:inherit; font-size:13px; cursor:pointer; }}
    .case-tab.active {{ color:#fff; background:#253746; border-color:#253746; }}
    .case-frame {{ display:none; width:100%; min-height:760px; border:1px solid var(--line); background:#fff; }}
    .case-frame.active {{ display:block; }}
    .proof {{ display:grid; grid-template-columns:repeat(2,minmax(0,1fr)); gap:14px 24px; }}
    code {{ font-family:ui-monospace,SFMono-Regular,Menlo,monospace; font-size:12px; }}
    a {{ color:#245f98; text-underline-offset:2px; }}
    .links {{ display:flex; flex-wrap:wrap; gap:8px 18px; margin-top:12px; }}
    @media (max-width:820px) {{
      .metric-strip,.charts,.proof {{ grid-template-columns:1fr; }}
      .inner {{ padding:21px 14px; }}
      .table-wrap {{ border:0; background:transparent; overflow:visible; }}
      table,tbody,tr,td {{ display:block; width:100%; }}
      thead {{ display:none; }}
      tr {{ margin-bottom:10px; border:1px solid var(--line); background:#fff; }}
      tr:last-child {{ margin-bottom:0; }}
      td,td:first-child {{ display:grid; grid-template-columns:minmax(0,0.9fr) minmax(0,1.1fr); gap:10px; padding:7px 9px; text-align:right; white-space:normal; overflow-wrap:anywhere; }}
      td::before {{ content:attr(data-label); color:#4b5965; font-weight:650; text-align:left; }}
      .case-frame {{ min-height:680px; }}
    }}
  </style>
</head>
<body>
  <header><div class="inner">
    <div class="meta">P-LA-002 · GPU1 A800 only · generated {esc(payload["generated_at_utc"])} · source <code>{SOURCE_SHA[:7]}</code></div>
    <h1>官方 FLA 上，FutureSeed 配哪种 Linear Attention？</h1>
    <p>固定 D192/L10/H6、loop5、effective batch128、500 optimizer steps、同一 official Sudoku 数据和同一 seed。只替换 FLA recurrent layer：GDN、KDA、GDN2。</p>
    <div class="verdict"><strong>结论：hard 53-blank 没有可靠赢家；GDN 的有限预算 opening 最好。</strong>三者在预先固定的 checkpoint batch 上 exact 都是 1.76%。另一批 512 cases 的差距只有 3 个 board 以内，不能硬吹架构胜负；但 GDN 的 CE 更低、46-50 blanks exact 更高，KDA 更慢，GDN2 多 16% 参数仍没打开 51-55 blanks。</div>
  </div></header>

  <section class="band"><div class="inner">
    <h2>公平结果</h2>
    <div class="metric-strip">
      <div class="metric"><small>预固定 holes53 checkpoint exact</small><b>1.76% / 1.76% / 1.76%</b><span>GDN / KDA / GDN2，hard 区间打平</span></div>
      <div class="metric"><small>46-50 blanks loop5 exact</small><b>85.94% / 65.04% / 75.39%</b><span>GDN opening 明显更快</span></div>
      <div class="metric"><small>51-64 blanks exact</small><b>全是 0</b><span>更复杂 state update 没跨过全局闭合断崖</span></div>
    </div>
    {table(["backbone","params","CE@500","fixed h53 exact","final exact","final blank","loop exact gain","train wall","peak allocated"], quality_rows)}
    <p class="meta">Train wall 取写 checkpoint 前的同一路径时间。Peak allocated 来自相同 full-model all-loop backward；不混用 nvidia-smi 驱动占用。单 seed 只用于机制门，不用于论文显著性结论。</p>
  </div></section>

  <section><div class="inner">
    <h2>优化与 loop</h2>
    <div class="charts">{ce_chart}{loop_chart}</div>
    <div style="margin-top:16px">{blank_chart}</div>
    <p>共同形态很清楚：loop1 到 loop2 有主要收益，loop3 以后基本饱和。FutureSeed 状态能在三种 recurrence 中传播，但更强的 KDA/GDN2 state update 没自动变成更持续的全局修错。</p>
  </div></section>

  <section class="band"><div class="inner">
    <h2>难度断崖</h2>
    {table(["blank range","GDN exact","GDN blank","KDA exact","KDA blank","GDN2 exact","GDN2 blank"], range_rows)}
    <p>46-50 blanks 能被 loop 大幅改善；51-55 一到，三种 backbone 的 exact 同时归零。当前瓶颈不是换一个更“高级”的 Linear Attention 方程就能解决，而是同等 500-step 计算还不足以形成 hard global closure。</p>
  </div></section>

  <section><div class="inner">
    <h2>同一个 53-blank case</h2>
    <p class="meta">三个页面的 puzzle 区块 SHA256 完全相同：<code>{esc(payload["same_case_sha256"])}</code>。红格是错，绿格是对，灰格是 clue。</p>
    {table(["backbone","loop1 wrong","loop2 wrong","loop3 wrong","loop4 wrong","loop5 wrong"], case_rows)}
    <div class="case-tabs">{''.join(case_buttons)}</div>
    {''.join(case_frames)}
  </div></section>

  <section class="band"><div class="inner">
    <h2>确实是官方 FLA，没有 fallback</h2>
    <div class="proof">
      <p><strong>安装来源</strong><br><code>flash-linear-attention 0.5.2</code><br>wheel SHA256 <code>{WHEEL_SHA256}</code></p>
      <p><strong>源码身份</strong><br>FLA marker <code>{FLA_SOURCE_SHA}</code><br>{source_hash_count} 个关键 installed files 与 pinned wheel 逐字节一致</p>
      <p><strong>后端锁定</strong><br><code>FLA_DISABLE_BACKEND_DISPATCH=1</code><br><code>FLA_CONV_BACKEND=triton</code>，CUDA GPU1 only</p>
      <p><strong>FutureSeed 接口</strong><br>只把上一层终态放进官方 <code>Cache</code> 作为下一层初态；完整调用官方 layer <code>forward</code>，没有左右双扫或本地 Torch fallback。</p>
    </div>
    {table(["backbone","official class","actual backward node","output err","state err","max grad err","h0 grad norm","state layout"], kernel_rows)}
  </div></section>

  <section><div class="inner">
    <h2>应该怎么决策</h2>
    <p><strong>保留 GDN 作为主 scaling backbone。</strong>它参数不大、CE 最低、easy-to-medium closure 最强；GDN2 可表达性更高，但此预算没有换来 hard exact；KDA 在这条任务上更慢且 opening 更差。</p>
    <p><strong>停止继续枚举 Linear Attention 变体。</strong>这轮已经回答“更细 memory gate 是否直接修好 hard Sudoku”：没有。下一条高信息实验应扩大 clean data/steps/model state，并只在曲线仍有斜率时延长，而不是继续换方程补表。</p>
    <p><strong>边界：</strong>这次比较的是三个 <em>FS-enabled</em> backbone 的排序，不是每个 backbone 上 FutureSeed 的因果增益。要写论文中的 FutureSeed gain，仍应在最终选定的 GDN scaling 点做一个 matched no-FS 对照，而不是三套 no-FS 表。</p>
    <div class="links">
      {''.join(f'<a href="../{Path(row["run"]).name}/visualizations/index.html">{esc(row["label"])} run</a>' for row in runs)}
      <a href="../../artifacts/fla-official-futureseed-strict-gate-c3342b8/gate.json">strict gate JSON</a>
    </div>
  </div></section>
  <script>
    const tabs = [...document.querySelectorAll('.case-tab')];
    const frames = [...document.querySelectorAll('.case-frame')];
    function resizeFrame(frame) {{
      try {{
        const root = frame.contentDocument.documentElement;
        frame.style.height = `${{Math.max(680, root.scrollHeight + 4)}}px`;
      }} catch (error) {{
        console.warn('Could not resize case frame', error);
      }}
    }}
    frames.forEach((frame) => frame.addEventListener('load', () => resizeFrame(frame)));
    tabs.forEach((tab) => tab.addEventListener('click', () => {{
      tabs.forEach((item) => {{
        const active = item === tab;
        item.classList.toggle('active', active);
        item.setAttribute('aria-pressed', String(active));
      }});
      frames.forEach((frame) => frame.classList.toggle('active', frame.id === tab.dataset.target));
      requestAnimationFrame(() => resizeFrame(document.getElementById(tab.dataset.target)));
    }}));
    window.addEventListener('resize', () => {{
      const activeFrame = document.querySelector('.case-frame.active');
      if (activeFrame) requestAnimationFrame(() => resizeFrame(activeFrame));
    }});
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
        default=Path("runs/fla-official-futureseed-comparison-20260720-c3342b8"),
    )
    args = parser.parse_args()
    repo = args.repo.resolve()
    out_dir = args.out_dir if args.out_dir.is_absolute() else repo / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    runs = [extract_run(repo, spec) for spec in RUNS]
    case_hashes = {row["case"]["puzzle_sha256"] for row in runs}
    if len(case_hashes) != 1:
        raise ValueError(f"Primary visualization cases differ: {sorted(case_hashes)}")

    gate = read_json(repo / "artifacts" / "fla-official-futureseed-strict-gate-c3342b8" / "gate.json")
    source_hashes = gate["provenance"]["source_file_hashes"]
    mismatches = [name for name, values in source_hashes.items() if values["installed_sha256"] != values["wheel_sha256"]]
    if mismatches:
        raise ValueError(f"Installed FLA source differs from pinned wheel: {mismatches}")

    payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "plan_id": "P-LA-002",
        "source_sha": SOURCE_SHA,
        "official_fla_sha": FLA_SOURCE_SHA,
        "wheel_sha256": WHEEL_SHA256,
        "same_case_sha256": next(iter(case_hashes)),
        "runs": runs,
        "provenance": {
            "source_file_hashes": source_hashes,
            "gate": gate,
            "all_source_hashes_match": True,
            "backend_dispatch_disabled": gate["provenance"]["backend_dispatch_disabled"],
            "conv_backend": gate["provenance"]["conv_backend"],
        },
        "decision": "Keep official GDN as the scaling backbone; stop architecture enumeration at this budget.",
    }
    write_json(out_dir / "comparison.json", payload)
    (out_dir / "index.html").write_text(build_html(repo, out_dir, payload), encoding="utf-8")
    (out_dir / "README.md").write_text(
        "# Official FLA FutureSeed Comparison\n\n"
        "Open `index.html` for the matched GDN/KDA/GDN2 metrics, strict no-fallback proof, "
        "training curves, and the same 53-blank case across loops.\n\n"
        "At the matched 500-step gate, pre-fixed holes53 exact is tied at 1.76%. GDN has "
        "the lowest CE and strongest 46-50-blank opening; all arms remain zero exact at "
        "51-64 blanks. Keep official GDN and stop architecture enumeration at this budget.\n\n"
        "Screenshots: `screenshots/desktop-1440x1000.png` and "
        "`screenshots/mobile-390x844.png`.\n",
        encoding="utf-8",
    )
    print(out_dir)


if __name__ == "__main__":
    main()
