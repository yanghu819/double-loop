#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import html
import json
import math
import os
import random
import statistics
import time
from pathlib import Path
from typing import Any

for cache_var in (
    "XDG_CACHE_HOME",
    "TRITON_CACHE_DIR",
    "TORCHINDUCTOR_CACHE_DIR",
    "TORCH_EXTENSIONS_DIR",
    "TMPDIR",
):
    cache_path = os.environ.get(cache_var, "")
    if not cache_path.startswith("/huyang2/double-loop/"):
        raise RuntimeError(f"{cache_var} must point below /huyang2/double-loop before importing CUDA runtimes")
if os.environ.get("FLA_DISABLE_BACKEND_DISPATCH") != "1":
    raise RuntimeError("FLA_DISABLE_BACKEND_DISPATCH=1 is required")
if os.environ.get("FLA_CONV_BACKEND") != "triton":
    raise RuntimeError("FLA_CONV_BACKEND=triton is required")
if os.environ.get("FLA_STRICT_OFFICIAL") != "1":
    raise RuntimeError("FLA_STRICT_OFFICIAL=1 is required")

import numpy as np
import torch
import torch.nn.functional as F

from check_fla_delta_backbones import collect_provenance, sha256_file
from study_rwkv_futureseed_loop import BLANK, CELLS, FutureSeedRWKV


BACKBONES = ("fla_gdn", "kda", "gdn2")
COLORS = {"fla_gdn": "#1769aa", "kda": "#b44b27", "gdn2": "#2e7d50"}
CRITICAL_CONFIG_FIELDS = (
    "d_model",
    "layers",
    "heads",
    "head_dim",
    "channel_mult",
    "gdn_mode",
    "gdn_expand_v",
    "gdn_use_short_conv",
    "gdn_conv_size",
    "gdn_allow_neg_eigval",
    "l_cycles",
    "max_loops",
    "loop_loss",
    "batch",
    "grad_accum_steps",
    "lr",
    "weight_decay",
    "blank_loss_weight",
    "future_seed_scale",
    "future_seed_decay",
    "future_seed_update",
    "future_seed_norm_mode",
    "noise_scale",
    "official_sudoku_train_split",
    "official_sudoku_eval_split",
    "seed",
    "forward_dtype",
)


def parse_checkpoint(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("checkpoint must be BACKBONE=/absolute/path.pt")
    backbone, path = value.split("=", 1)
    if backbone not in BACKBONES:
        raise argparse.ArgumentTypeError(f"unknown backbone {backbone!r}")
    checkpoint = Path(path).expanduser().resolve()
    if not checkpoint.is_file():
        raise argparse.ArgumentTypeError(f"checkpoint does not exist: {checkpoint}")
    return backbone, checkpoint


def tensor_rms(value: torch.Tensor) -> torch.Tensor:
    return value.float().square().mean().sqrt()


def tensor_quantiles(value: torch.Tensor) -> dict[str, float]:
    flat = value.detach().float().reshape(-1)
    return {
        "mean": float(flat.mean().item()),
        "p10": float(torch.quantile(flat, 0.10).item()),
        "p50": float(torch.quantile(flat, 0.50).item()),
        "p90": float(torch.quantile(flat, 0.90).item()),
    }


def effective_hole_stages(raw: str, completed_steps: int) -> list[dict[str, int]]:
    remaining = int(completed_steps)
    effective: list[dict[str, int]] = []
    for item in str(raw).split(","):
        span, count_text = item.strip().split(":", 1)
        lo_text, hi_text = span.split("-", 1)
        planned = int(count_text)
        executed = min(max(remaining, 0), planned)
        if executed > 0:
            effective.append({"holes_min": int(lo_text), "holes_max": int(hi_text), "steps": executed})
        remaining -= executed
        if remaining <= 0:
            break
    if remaining != 0:
        raise AssertionError(f"hole schedule {raw!r} does not cover checkpoint step {completed_steps}")
    return effective


def cosine_mean(a: torch.Tensor, b: torch.Tensor) -> float:
    a_flat = a.detach().float().flatten(1)
    b_flat = b.detach().float().flatten(1)
    cosine = F.cosine_similarity(a_flat, b_flat, dim=-1, eps=1e-8)
    return float(cosine.mean().item())


def build_reasoner(config: dict[str, Any], *, device: torch.device) -> FutureSeedRWKV:
    reasoner = FutureSeedRWKV(
        int(config["d_model"]),
        int(config["layers"]),
        int(config["heads"]),
        int(config["head_dim"]),
        int(config["channel_mult"]),
        future_seed_scale=float(config["future_seed_scale"]),
        future_seed_decay=float(config.get("future_seed_decay", 0.0)),
        future_seed_update=str(config["future_seed_update"]),
        future_seed_norm_mode=str(config["future_seed_norm_mode"]),
        activation_checkpoint=False,
        rwkv_kernel=str(config.get("rwkv_kernel", "auto")),
        backbone=str(config["backbone"]),
        gdn_mode=str(config["gdn_mode"]),
        gdn_expand_v=float(config["gdn_expand_v"]),
        gdn_use_short_conv=bool(config["gdn_use_short_conv"]),
        gdn_conv_size=int(config["gdn_conv_size"]),
        gdn_allow_neg_eigval=bool(config["gdn_allow_neg_eigval"]),
    )
    # FLA layers choose the same chunk kernel used for training only in train mode.
    # No gradients, noise, or optimizer updates are enabled in this diagnostic.
    reasoner.train()
    return reasoner.to(device)


def load_checkpoint(
    checkpoint_path: Path,
    *,
    expected_backbone: str,
    device: torch.device,
) -> tuple[FutureSeedRWKV, dict[str, torch.Tensor], dict[str, Any]]:
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = dict(payload["args"])
    if config.get("backbone") != expected_backbone:
        raise AssertionError(
            f"checkpoint backbone {config.get('backbone')!r} does not match {expected_backbone!r}"
        )
    if int(payload.get("saved_at_step", -1)) != 500:
        raise AssertionError(f"expected exact step500 checkpoint, got {payload.get('saved_at_step')}")
    if config.get("future_seed_update") != "fixed" or float(config.get("future_seed_decay", -1.0)) != 0.0:
        raise AssertionError("diagnostic expects fixed, no-decay FutureSeed transfer")
    if config.get("future_seed_norm_mode") != "unit":
        raise AssertionError("diagnostic expects the unit-RMS FutureSeed path")
    if float(config.get("noise_scale", 0.0)) != 0.0:
        raise AssertionError("diagnostic expects a clean checkpoint with no feature noise")

    reasoner = build_reasoner(config, device=device)
    reasoner_state = {
        key.removeprefix("reasoner."): value
        for key, value in payload["model"].items()
        if key.startswith("reasoner.")
    }
    reasoner.load_state_dict(reasoner_state, strict=True)
    embedding_state = {
        "embed_weight": payload["model"]["embed.weight"].to(device),
        "position_weight": payload["model"]["position.weight"].to(device),
        "h_init": payload["model"]["h_init"].to(device),
        "l_init": payload["model"]["l_init"].to(device),
    }
    metadata = {
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "saved_at_step": int(payload["saved_at_step"]),
        "config": {field: config.get(field) for field in ("backbone", *CRITICAL_CONFIG_FIELDS)},
        "planned_hole_stages": str(config.get("hole_stages", "")),
        "planned_steps": int(config.get("steps", payload["saved_at_step"])),
        "effective_hole_stages": effective_hole_stages(
            str(config.get("hole_stages", "")), int(payload["saved_at_step"])
        ),
    }
    del payload
    return reasoner, embedding_state, metadata


def load_inputs(
    data_dir: Path,
    split: str,
    *,
    batch_size: int,
    seed: int,
    holes_min: int,
    holes_max: int,
    device: torch.device,
) -> tuple[torch.Tensor, dict[str, Any]]:
    path = data_dir / split / "all__inputs.npy"
    raw = np.load(path, mmap_mode="r")
    blank_counts = (raw == 1).sum(axis=1)
    candidates = np.flatnonzero((blank_counts >= holes_min) & (blank_counts <= holes_max))
    if len(candidates) < batch_size:
        raise ValueError(f"only {len(candidates)} inputs in blank range {holes_min}-{holes_max}")
    rng = random.Random(seed)
    positions = rng.sample(range(len(candidates)), batch_size)
    indices = candidates[positions]
    selected = np.asarray(raw[indices], dtype=np.int64)
    clue_mask = selected != 1
    mapped = np.where(clue_mask, selected - 2, BLANK)
    inputs = torch.as_tensor(mapped, dtype=torch.long, device=device)
    digest = hashlib.sha256(np.ascontiguousarray(mapped).tobytes()).hexdigest()
    return inputs, {
        "path": str(path),
        "split": split,
        "indices": [int(index) for index in indices.tolist()],
        "blank_counts": [int(blank_counts[index]) for index in indices.tolist()],
        "mapped_inputs_sha256": digest,
        "labels_loaded_or_used": False,
    }


def make_first_reasoner_input(inputs: torch.Tensor, embedding_state: dict[str, torch.Tensor]) -> torch.Tensor:
    embedded = F.embedding(inputs, embedding_state["embed_weight"])
    positions = embedding_state["position_weight"][:CELLS].unsqueeze(0)
    return embedded + positions + embedding_state["h_init"] + embedding_state["l_init"]


def gate_statistics(backbone: str, block: torch.nn.Module, x: torch.Tensor) -> dict[str, Any]:
    core = block.time_mix.core
    hidden = block.ln_time(x)
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        if backbone == "fla_gdn":
            raw_g = core.a_proj(hidden).float()
            log_decay = -core.A_log.float().exp().view(1, 1, -1) * F.softplus(
                raw_g + core.dt_bias.float().view(1, 1, -1)
            )
            erase = torch.sigmoid(core.b_proj(hidden).float())
            write = erase
        elif backbone == "kda":
            raw_g = core.f_proj(hidden).float().view(*hidden.shape[:2], core.num_v_heads, core.head_k_dim)
            dt_bias = core.dt_bias.float().view(core.num_v_heads, core.head_k_dim)
            log_decay = -core.A_log.float().exp().view(1, 1, -1, 1) * F.softplus(raw_g + dt_bias)
            erase = torch.sigmoid(core.b_proj(hidden).float())
            write = erase
        elif backbone == "gdn2":
            raw_g = core.f_proj(hidden).float().view(*hidden.shape[:2], core.num_heads, core.head_k_dim)
            dt_bias = core.dt_bias.float().view(core.num_heads, core.head_k_dim)
            log_decay = -core.A_log.float().exp().view(1, 1, -1, 1) * F.softplus(raw_g + dt_bias)
            erase = torch.sigmoid(core.b_proj(hidden).float())
            write = torch.sigmoid(core.w_proj(hidden).float())
        else:  # pragma: no cover - guarded by CLI.
            raise ValueError(backbone)
    return {
        "per_token_retention": tensor_quantiles(log_decay.exp()),
        "erase_gate": tensor_quantiles(erase),
        "write_gate": tensor_quantiles(write),
    }


@torch.no_grad()
def capture_transitions(
    reasoner: FutureSeedRWKV,
    x: torch.Tensor,
) -> list[dict[str, Any]]:
    previous_state: torch.Tensor | None = None
    current = x
    captures: list[dict[str, Any]] = []
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        for layer_idx, block in enumerate(reasoner.blocks):
            initial_state = None
            if layer_idx > 0:
                assert previous_state is not None
                raw_rms = previous_state.float().square().mean(dim=(-1, -2), keepdim=True).sqrt().clamp_min(1e-6)
                gate = torch.sigmoid(block.future_seed_logit) * reasoner.future_seed_scale
                initial_state = previous_state / raw_rms.to(previous_state.dtype) * gate.to(previous_state.dtype)
                captures.append(
                    {
                        "layer": layer_idx,
                        "block": block,
                        "block_input": current.detach(),
                        "initial_state": initial_state.detach(),
                        "raw_seed_rms": float(raw_rms.mean().item()),
                        "injected_seed_rms": float(tensor_rms(initial_state).item()),
                        "future_seed_gate_mean": float(gate.float().mean().item()),
                    }
                )
            current, previous_state = block(current, initial_state=initial_state)
    return captures


@torch.no_grad()
def probe_capture(
    backbone: str,
    capture: dict[str, Any],
    prefixes: list[int],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    block = capture["block"]
    block_input = capture["block_input"]
    initial_state = capture["initial_state"]
    zero_state = torch.zeros_like(initial_state)
    rows: list[dict[str, Any]] = []
    for prefix in prefixes:
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            normalized = block.ln_time(block_input[:, :prefix])
            seeded_output, seeded_state = block.time_mix(normalized, initial_state=initial_state)
            zero_output, zero_terminal = block.time_mix(normalized, initial_state=zero_state)
        output_delta = seeded_output.float() - zero_output.float()
        state_delta = seeded_state.float() - zero_terminal.float()
        seed_rms = tensor_rms(initial_state).clamp_min(1e-8)
        zero_output_rms = tensor_rms(zero_output).clamp_min(1e-8)
        rows.append(
            {
                "layer": int(capture["layer"]),
                "prefix": int(prefix),
                "state_delta_rms": float(tensor_rms(state_delta).item()),
                "state_retention_ratio": float((tensor_rms(state_delta) / seed_rms).item()),
                "state_delta_seed_cosine": cosine_mean(state_delta, initial_state),
                "output_delta_rms": float(tensor_rms(output_delta).item()),
                "output_delta_relative_to_zero": float((tensor_rms(output_delta) / zero_output_rms).item()),
                "last_token_output_delta_rms": float(tensor_rms(output_delta[:, -1:]).item()),
                "zero_output_rms": float(zero_output_rms.item()),
                "raw_seed_rms": float(capture["raw_seed_rms"]),
                "injected_seed_rms": float(capture["injected_seed_rms"]),
                "future_seed_gate_mean": float(capture["future_seed_gate_mean"]),
            }
        )
    return rows, gate_statistics(backbone, block, block_input)


def aggregate_rows(rows: list[dict[str, Any]], prefixes: list[int]) -> list[dict[str, Any]]:
    metrics = (
        "state_retention_ratio",
        "state_delta_seed_cosine",
        "output_delta_rms",
        "output_delta_relative_to_zero",
        "last_token_output_delta_rms",
        "injected_seed_rms",
        "future_seed_gate_mean",
    )
    output: list[dict[str, Any]] = []
    for prefix in prefixes:
        selected = [row for row in rows if int(row["prefix"]) == prefix]
        item: dict[str, Any] = {"prefix": prefix, "layers": len(selected)}
        for metric in metrics:
            values = [float(row[metric]) for row in selected]
            item[metric] = {
                "mean": statistics.fmean(values),
                "median": statistics.median(values),
                "min": min(values),
                "max": max(values),
            }
        output.append(item)
    return output


def check_matched_configs(metadata: dict[str, dict[str, Any]]) -> dict[str, Any]:
    baseline = metadata["fla_gdn"]["config"]
    mismatches: dict[str, dict[str, Any]] = {}
    for field in CRITICAL_CONFIG_FIELDS:
        values = {backbone: metadata[backbone]["config"].get(field) for backbone in BACKBONES}
        if len({json.dumps(value, sort_keys=True) for value in values.values()}) != 1:
            mismatches[field] = values
    if mismatches:
        raise AssertionError(f"critical checkpoint configs differ: {json.dumps(mismatches, indent=2)}")
    effective_schedules = {
        backbone: metadata[backbone]["effective_hole_stages"] for backbone in BACKBONES
    }
    if len({json.dumps(value, sort_keys=True) for value in effective_schedules.values()}) != 1:
        raise AssertionError(
            "effective checkpoint curricula differ: " + json.dumps(effective_schedules, indent=2)
        )
    output = {field: baseline.get(field) for field in CRITICAL_CONFIG_FIELDS}
    output["effective_hole_stages_through_step500"] = effective_schedules["fla_gdn"]
    output["planned_hole_stages_by_backbone"] = {
        backbone: metadata[backbone]["planned_hole_stages"] for backbone in BACKBONES
    }
    output["planned_steps_by_backbone"] = {
        backbone: metadata[backbone]["planned_steps"] for backbone in BACKBONES
    }
    return output


def svg_chart(result: dict[str, Any], metric: str, title: str) -> str:
    width, height = 720, 310
    left, right, top, bottom = 70, 20, 42, 48
    plot_w, plot_h = width - left - right, height - top - bottom
    prefixes = result["prefixes"]
    values = [
        row[metric]["mean"]
        for backbone in BACKBONES
        for row in result["arms"][backbone]["summary"]
    ]
    y_max = max(values) * 1.08 if values and max(values) > 0 else 1.0
    x_positions = {
        prefix: left + index * plot_w / max(len(prefixes) - 1, 1)
        for index, prefix in enumerate(prefixes)
    }

    def y_position(value: float) -> float:
        return top + plot_h * (1.0 - value / y_max)

    parts = [
        f'<svg viewBox="0 0 {width} {height}" role="img" aria-label="{html.escape(title)}">',
        f'<text x="{left}" y="24" class="chart-title">{html.escape(title)}</text>',
        f'<line x1="{left}" y1="{top + plot_h}" x2="{left + plot_w}" y2="{top + plot_h}" class="axis"/>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_h}" class="axis"/>',
    ]
    for tick in range(5):
        value = y_max * tick / 4
        y = y_position(value)
        parts.append(f'<line x1="{left}" y1="{y:.2f}" x2="{left + plot_w}" y2="{y:.2f}" class="grid"/>')
        parts.append(f'<text x="{left - 8}" y="{y + 4:.2f}" text-anchor="end" class="tick">{value:.3g}</text>')
    for prefix, x in x_positions.items():
        parts.append(f'<text x="{x:.2f}" y="{top + plot_h + 24}" text-anchor="middle" class="tick">{prefix}</text>')
    for legend_idx, backbone in enumerate(BACKBONES):
        rows = result["arms"][backbone]["summary"]
        points = " ".join(f'{x_positions[row["prefix"]]:.2f},{y_position(row[metric]["mean"]):.2f}' for row in rows)
        color = COLORS[backbone]
        parts.append(f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="3"/>')
        for row in rows:
            x = x_positions[row["prefix"]]
            y = y_position(row[metric]["mean"])
            parts.append(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="4" fill="{color}"/>')
        lx = left + legend_idx * 150
        parts.append(f'<line x1="{lx}" y1="{height - 12}" x2="{lx + 24}" y2="{height - 12}" stroke="{color}" stroke-width="3"/>')
        parts.append(f'<text x="{lx + 30}" y="{height - 8}" class="tick">{html.escape(backbone)}</text>')
    parts.append("</svg>")
    return "".join(parts)


def render_html(result: dict[str, Any]) -> str:
    prefix = result["prefixes"][-1]
    summary_rows = []
    for backbone in BACKBONES:
        final = next(row for row in result["arms"][backbone]["summary"] if row["prefix"] == prefix)
        gate = result["arms"][backbone]["gate_summary"]
        summary_rows.append(
            "<tr>"
            f"<td><strong>{html.escape(backbone)}</strong></td>"
            f"<td>{final['state_retention_ratio']['mean']:.4f}</td>"
            f"<td>{final['state_delta_seed_cosine']['mean']:.4f}</td>"
            f"<td>{final['output_delta_relative_to_zero']['mean']:.4f}</td>"
            f"<td>{final['future_seed_gate_mean']['mean']:.4f}</td>"
            f"<td>{gate['per_token_retention_mean']:.4f}</td>"
            f"<td>{gate['erase_gate_mean']:.4f}</td>"
            f"<td>{gate['write_gate_mean']:.4f}</td>"
            "</tr>"
        )
    config = result["matched_config"]
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>FLA FutureSeed survival diagnostic</title>
<style>
:root{{--bg:#f5f7f8;--paper:#fff;--ink:#182026;--muted:#5b6770;--line:#d7dde1;}}
*{{box-sizing:border-box}} body{{margin:0;background:var(--bg);color:var(--ink);font:15px/1.5 ui-sans-serif,system-ui,sans-serif}}
main{{max-width:1120px;margin:0 auto;padding:28px 20px 60px}} h1{{font-size:28px;margin:0 0 6px;letter-spacing:0}} h2{{font-size:19px;margin:30px 0 10px;letter-spacing:0}}
p{{max-width:900px}} .muted{{color:var(--muted)}} .band{{background:var(--paper);border:1px solid var(--line);border-radius:6px;padding:18px;margin-top:16px}}
.charts{{display:grid;grid-template-columns:1fr 1fr;gap:14px}} svg{{width:100%;height:auto;background:#fff;border:1px solid var(--line);border-radius:6px}}
.axis{{stroke:#54616a;stroke-width:1}} .grid{{stroke:#e5e9ec;stroke-width:1}} .tick{{font-size:12px;fill:#4d5961}} .chart-title{{font-size:16px;font-weight:650;fill:#182026}}
table{{width:100%;border-collapse:collapse}} th,td{{padding:9px 10px;border-bottom:1px solid var(--line);text-align:right;white-space:nowrap}} th:first-child,td:first-child{{text-align:left}}
code{{background:#edf1f3;padding:2px 5px;border-radius:3px}} @media(max-width:780px){{.charts{{grid-template-columns:1fr}} .table-wrap{{overflow:auto}}}}
</style></head><body><main>
<h1>FutureSeed survival in official FLA recurrences</h1>
<p class="muted">Plan {html.escape(result['plan_id'])}. GPU-only, no labels, no training, no task rule. The diagnostic changes only the recurrent initial state: actual checkpoint FutureSeed versus an all-zero state.</p>
<section class="band"><strong>Matched checkpoint gate:</strong> D{config['d_model']}/L{config['layers']}/H{config['heads']}/Dhead{config['head_dim']}, expand-v{config['gdn_expand_v']}, step500, fixed unit-RMS FutureSeed, official chunk CUDA path. Input digest <code>{result['input']['mapped_inputs_sha256'][:16]}</code>.</section>
<h2>Influence over sequence position</h2><div class="charts">
{svg_chart(result, 'state_retention_ratio', 'Recurrent-state influence retained')}
{svg_chart(result, 'output_delta_relative_to_zero', 'Output change caused by FutureSeed')}
</div>
<h2>Token {prefix} summary</h2><div class="band table-wrap"><table><thead><tr><th>Backbone</th><th>state retained</th><th>state cosine</th><th>relative output effect</th><th>FS gate</th><th>per-token decay</th><th>erase</th><th>write</th></tr></thead><tbody>{''.join(summary_rows)}</tbody></table></div>
<h2>How to read this</h2><div class="band"><p><strong>state retained</strong> is the RMS difference between seeded and zero-state terminal memories, divided by injected-seed RMS. <strong>relative output effect</strong> is the RMS output difference divided by the zero-state output RMS. Lower values mean the recurrence makes FutureSeed matter less. These are descriptive diagnostics, not Sudoku scores.</p></div>
</main></body></html>"""


def summarize_gate_stats(layer_stats: list[dict[str, Any]]) -> dict[str, float]:
    output: dict[str, float] = {}
    for key in ("per_token_retention", "erase_gate", "write_gate"):
        output[f"{key}_mean"] = statistics.fmean(float(item[key]["mean"]) for item in layer_stats)
        output[f"{key}_p10_mean"] = statistics.fmean(float(item[key]["p10"]) for item in layer_stats)
        output[f"{key}_p90_mean"] = statistics.fmean(float(item[key]["p90"]) for item in layer_stats)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="Measure how official FLA recurrences retain a real FutureSeed state")
    parser.add_argument("--checkpoint", action="append", type=parse_checkpoint, required=True)
    parser.add_argument("--wheel", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--input-seed", type=int, default=52003)
    parser.add_argument("--holes-min", type=int, default=51)
    parser.add_argument("--holes-max", type=int, default=55)
    parser.add_argument("--prefixes", default="1,8,32,81")
    parser.add_argument("--plan-id", default="P-LA-003")
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()

    checkpoints = dict(args.checkpoint)
    if tuple(sorted(checkpoints)) != tuple(sorted(BACKBONES)):
        raise ValueError(f"provide exactly one checkpoint for each of {BACKBONES}")
    prefixes = [int(value) for value in args.prefixes.split(",") if value.strip()]
    if not prefixes or sorted(set(prefixes)) != prefixes or prefixes[-1] > CELLS or prefixes[0] < 1:
        raise ValueError(f"prefixes must be sorted unique values in [1, {CELLS}]")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("diagnostic requires exactly one visible CUDA device")
    device = torch.device("cuda:0")
    gpu_name = torch.cuda.get_device_name(device)
    if "A800" not in gpu_name:
        raise RuntimeError(f"expected GPU1 A800, got {gpu_name}")

    started = time.time()
    provenance = collect_provenance(args.wheel)
    inputs, input_metadata = load_inputs(
        args.data_dir.resolve(),
        args.split,
        batch_size=args.batch_size,
        seed=args.input_seed,
        holes_min=args.holes_min,
        holes_max=args.holes_max,
        device=device,
    )
    arms: dict[str, Any] = {}
    checkpoint_metadata: dict[str, dict[str, Any]] = {}
    for backbone in BACKBONES:
        torch.cuda.empty_cache()
        reasoner, embedding_state, metadata = load_checkpoint(
            checkpoints[backbone], expected_backbone=backbone, device=device
        )
        checkpoint_metadata[backbone] = metadata
        first_input = make_first_reasoner_input(inputs, embedding_state)
        captures = capture_transitions(reasoner, first_input)
        if len(captures) != int(metadata["config"]["layers"]) - 1:
            raise AssertionError("did not capture every FutureSeed layer transition")
        rows: list[dict[str, Any]] = []
        layer_gate_stats: list[dict[str, Any]] = []
        for capture in captures:
            capture_rows, gates = probe_capture(backbone, capture, prefixes)
            rows.extend(capture_rows)
            layer_gate_stats.append({"layer": int(capture["layer"]), **gates})
        arms[backbone] = {
            "checkpoint": metadata,
            "rows": rows,
            "summary": aggregate_rows(rows, prefixes),
            "gate_by_layer": layer_gate_stats,
            "gate_summary": summarize_gate_stats(layer_gate_stats),
        }
        del reasoner, embedding_state, first_input, captures
        torch.cuda.empty_cache()

    matched_config = check_matched_configs(checkpoint_metadata)
    result = {
        "plan_id": args.plan_id,
        "generated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "elapsed_sec": time.time() - started,
        "device": {
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "name": gpu_name,
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "peak_allocated_mb": torch.cuda.max_memory_allocated(device) / (1024**2),
        },
        "official_fla": provenance,
        "matched_config": matched_config,
        "input": input_metadata,
        "prefixes": prefixes,
        "method": {
            "labels_loaded_or_used": False,
            "training_or_parameter_updates": False,
            "state_comparison": "actual fixed unit-RMS FutureSeed versus zeros",
            "kernel_mode": "official chunk forced with module.train(); torch.no_grad()",
        },
        "arms": arms,
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "diagnostic.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    (args.out_dir / "index.html").write_text(render_html(result), encoding="utf-8")
    print(json.dumps({
        "out_dir": str(args.out_dir),
        "elapsed_sec": result["elapsed_sec"],
        "summary": {backbone: arms[backbone]["summary"] for backbone in BACKBONES},
        "gate_summary": {backbone: arms[backbone]["gate_summary"] for backbone in BACKBONES},
    }, indent=2))


if __name__ == "__main__":
    main()
