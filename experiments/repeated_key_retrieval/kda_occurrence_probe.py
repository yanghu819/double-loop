#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import html
import json
import math
import os
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


EXPECTED_FLA_SHA = "9c8e42e762fce087c27b673af4922795d9edb85e"
EXPECTED_FLA_WHEEL_SHA256 = "0280db310981915eb048ece99d7bedca8b5caa9be65c99835a0f912ada977d6a"
ADDRESS_MODES = ("content", "oracle_occurrence_rotary")


@dataclass(frozen=True)
class RetrievalBatch:
    key_ids: torch.Tensor
    value_ids: torch.Tensor
    occurrence_ids: torch.Tensor
    query_key_ids: torch.Tensor
    query_occurrence_ids: torch.Tensor
    labels: torch.Tensor
    target_positions: torch.Tensor
    target_values: torch.Tensor


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def apply_occurrence_rotary(
    x: torch.Tensor,
    occurrence_ids: torch.Tensor,
    *,
    base: float = 10_000.0,
) -> torch.Tensor:
    """Apply a norm-preserving rotation indexed by a per-token occurrence id."""
    if x.ndim != 4:
        raise ValueError(f"x must have shape [B,T,H,D], got {tuple(x.shape)}")
    if occurrence_ids.shape != x.shape[:2]:
        raise ValueError(
            "occurrence_ids must match x's [B,T] prefix, got "
            f"{tuple(occurrence_ids.shape)} vs {tuple(x.shape[:2])}"
        )
    rotary_dim = x.shape[-1] - (x.shape[-1] % 2)
    if rotary_dim == 0:
        return x
    half = rotary_dim // 2
    inv_freq = torch.exp(
        -math.log(base)
        * torch.arange(half, device=x.device, dtype=torch.float32)
        / max(half, 1)
    )
    angle = occurrence_ids.float().unsqueeze(-1) * inv_freq
    cos = angle.cos().unsqueeze(-2)
    sin = angle.sin().unsqueeze(-2)
    source = x[..., :rotary_dim].float()
    first, second = source.split(half, dim=-1)
    rotated = torch.cat(
        (first * cos - second * sin, first * sin + second * cos),
        dim=-1,
    ).to(dtype=x.dtype)
    if rotary_dim == x.shape[-1]:
        return rotated
    return torch.cat((rotated, x[..., rotary_dim:]), dim=-1)


def generate_retrieval_batch(
    *,
    batch_size: int,
    sequence_length: int,
    target_repeats: int,
    key_vocab_size: int,
    value_vocab_size: int,
    max_occurrence: int,
    generator: torch.Generator,
    device: torch.device,
) -> RetrievalBatch:
    if key_vocab_size < 2:
        raise ValueError("key_vocab_size must be at least 2")
    if not (1 <= target_repeats <= sequence_length):
        raise ValueError("target_repeats must be in [1, sequence_length]")

    query_keys = torch.randint(
        key_vocab_size,
        (batch_size,),
        generator=generator,
        device=device,
    )
    filler = torch.randint(
        key_vocab_size - 1,
        (batch_size, sequence_length),
        generator=generator,
        device=device,
    )
    key_ids = filler + (filler >= query_keys[:, None]).long()
    position_scores = torch.rand(
        batch_size,
        sequence_length,
        generator=generator,
        device=device,
    )
    target_positions = position_scores.topk(
        target_repeats,
        dim=1,
        largest=False,
    ).indices.sort(dim=1).values
    key_ids.scatter_(1, target_positions, query_keys[:, None].expand_as(target_positions))

    value_ids = torch.randint(
        value_vocab_size,
        (batch_size, sequence_length),
        generator=generator,
        device=device,
    )
    target_values = value_ids.gather(1, target_positions)
    query_occurrences = torch.randint(
        1,
        target_repeats + 1,
        (batch_size,),
        generator=generator,
        device=device,
    )
    labels = target_values.gather(1, query_occurrences[:, None] - 1).squeeze(1)

    one_hot = F.one_hot(key_ids, num_classes=key_vocab_size)
    occurrence_ids = one_hot.cumsum(dim=1).gather(
        2,
        key_ids.unsqueeze(-1),
    ).squeeze(-1)
    observed_max = int(occurrence_ids.max().item())
    if observed_max > max_occurrence:
        raise ValueError(
            f"Observed occurrence id {observed_max} exceeds max_occurrence={max_occurrence}"
        )
    return RetrievalBatch(
        key_ids=key_ids,
        value_ids=value_ids,
        occurrence_ids=occurrence_ids,
        query_key_ids=query_keys,
        query_occurrence_ids=query_occurrences,
        labels=labels,
        target_positions=target_positions,
        target_values=target_values,
    )


class KDAOccurrenceMemory(nn.Module):
    """One official KDA memory layer with an optional parameter-free address binding."""

    def __init__(
        self,
        *,
        key_vocab_size: int,
        value_vocab_size: int,
        max_occurrence: int,
        d_model: int,
        heads: int,
        head_dim: int,
        expand_v: float,
        address_mode: str,
    ) -> None:
        super().__init__()
        if address_mode not in ADDRESS_MODES:
            raise ValueError(f"address_mode must be one of {ADDRESS_MODES}")
        if d_model != heads * head_dim:
            raise ValueError("d_model must equal heads * head_dim for this matched probe")

        from fla.layers.kda import KimiDeltaAttention

        self.address_mode = address_mode
        self.key_vocab_size = int(key_vocab_size)
        self.value_vocab_size = int(value_vocab_size)
        self.max_occurrence = int(max_occurrence)
        self.d_model = int(d_model)
        self.heads = int(heads)
        self.head_dim = int(head_dim)
        self.key_embedding = nn.Embedding(key_vocab_size, d_model)
        self.value_embedding = nn.Embedding(value_vocab_size, d_model)
        self.occurrence_embedding = nn.Embedding(max_occurrence + 1, d_model)
        self.type_embedding = nn.Embedding(2, d_model)
        self.input_norm = nn.LayerNorm(d_model)
        self.core = KimiDeltaAttention(
            hidden_size=d_model,
            expand_v=expand_v,
            head_dim=head_dim,
            num_heads=heads,
            num_v_heads=heads,
            mode="chunk",
            use_short_conv=False,
            allow_neg_eigval=False,
            layer_idx=0,
        )
        self.output_norm = nn.LayerNorm(d_model)
        self.classifier = nn.Linear(d_model, value_vocab_size, bias=False)

    def _embed_writes(self, batch: RetrievalBatch) -> torch.Tensor:
        type_ids = torch.zeros_like(batch.key_ids)
        x = (
            self.key_embedding(batch.key_ids)
            + self.value_embedding(batch.value_ids)
            + self.occurrence_embedding(batch.occurrence_ids)
            + self.type_embedding(type_ids)
        )
        return self.input_norm(x)

    def _embed_query(self, batch: RetrievalBatch) -> torch.Tensor:
        type_ids = torch.ones_like(batch.query_key_ids)
        x = (
            self.key_embedding(batch.query_key_ids)
            + self.occurrence_embedding(batch.query_occurrence_ids)
            + self.type_embedding(type_ids)
        )
        return self.input_norm(x).unsqueeze(1)

    def forward(self, batch: RetrievalBatch) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        from einops import rearrange
        from fla.ops.kda import chunk_kda

        x = self._embed_writes(batch)
        query_x = self._embed_query(batch)
        core = self.core
        batch_size, sequence_length, _ = x.shape

        q = F.silu(core.q_proj(x))
        k = F.silu(core.k_proj(x))
        v = F.silu(core.v_proj(x))
        q = rearrange(q, "b t (h d) -> b t h d", d=core.head_k_dim)
        k = rearrange(k, "b t (h d) -> b t h d", d=core.head_k_dim)
        v = rearrange(v, "b t (h d) -> b t h d", d=core.head_v_dim)

        query_q = F.silu(core.q_proj(query_x))
        query_q = rearrange(
            query_q,
            "b t (h d) -> b t h d",
            d=core.head_k_dim,
        )
        if self.address_mode == "oracle_occurrence_rotary":
            q = apply_occurrence_rotary(q, batch.occurrence_ids)
            k = apply_occurrence_rotary(k, batch.occurrence_ids)
            query_q = apply_occurrence_rotary(
                query_q,
                batch.query_occurrence_ids[:, None],
            )

        g = core.f_proj(x)
        beta = core.b_proj(x)
        g = rearrange(g, "b t (h d) -> b t h d", d=core.head_k_dim)
        _ignored, terminal_state = chunk_kda(
            q=q,
            k=k,
            v=v,
            g=g,
            beta=beta,
            A_log=core.A_log,
            dt_bias=core.dt_bias,
            initial_state=None,
            output_final_state=True,
            use_qk_l2norm_in_kernel=True,
            use_gate_in_kernel=True,
            use_beta_sigmoid_in_kernel=True,
            allow_neg_eigval=core.allow_neg_eigval,
            safe_gate=core.safe_gate,
            lower_bound=core.lower_bound,
            state_v_first=True,
        )
        if terminal_state is None:
            raise RuntimeError("Official chunk_kda did not return terminal state")
        expected_state = (
            batch_size,
            core.num_v_heads,
            core.head_v_dim,
            core.head_k_dim,
        )
        if tuple(terminal_state.shape) != expected_state:
            raise RuntimeError(
                "Official KDA terminal state does not match the audited V-first "
                f"layout: {tuple(terminal_state.shape)} != {expected_state}"
            )

        query_unit = F.normalize(query_q.float(), dim=-1)
        query_read = torch.einsum(
            "bthk,bhvk->bthv",
            query_unit * (core.head_k_dim**-0.5),
            terminal_state.float(),
        ).to(dtype=query_x.dtype)
        output_gate = rearrange(
            core.g_proj(query_x),
            "b t (h d) -> b t h d",
            d=core.head_v_dim,
        )
        readout = core.o_norm(query_read, output_gate)
        readout = core.o_proj(
            readout.reshape(batch_size, 1, core.value_dim)
        ).squeeze(1)
        logits = self.classifier(self.output_norm(readout))

        with torch.no_grad():
            q_norm = query_q.float().norm(dim=-1).mean()
            k_norm = k.float().norm(dim=-1).mean()
            state_rms = terminal_state.float().square().mean().sqrt()
        return logits, {
            "query_q_norm": q_norm,
            "write_k_norm": k_norm,
            "terminal_state_rms": state_rms,
            "sequence_length": x.new_tensor(float(sequence_length)),
        }


def parse_eval_specs(text: str) -> list[tuple[int, int]]:
    specs: list[tuple[int, int]] = []
    for item in text.split(","):
        length_text, repeats_text = item.split(":", 1)
        specs.append((int(length_text), int(repeats_text)))
    if not specs:
        raise ValueError("At least one eval spec is required")
    return specs


@torch.inference_mode()
def evaluate_model(
    model: KDAOccurrenceMemory,
    *,
    batch_size: int,
    batches: int,
    sequence_length: int,
    target_repeats: int,
    key_vocab_size: int,
    value_vocab_size: int,
    max_occurrence: int,
    seed: int,
    device: torch.device,
    collect_cases: bool,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    model.eval()
    total = 0
    correct = 0
    loss_sum = 0.0
    by_occurrence: dict[int, list[int]] = {}
    cases: list[dict[str, Any]] = []
    diagnostics: dict[str, list[float]] = {}
    generator = torch.Generator(device=device).manual_seed(seed)
    for _ in range(batches):
        batch = generate_retrieval_batch(
            batch_size=batch_size,
            sequence_length=sequence_length,
            target_repeats=target_repeats,
            key_vocab_size=key_vocab_size,
            value_vocab_size=value_vocab_size,
            max_occurrence=max_occurrence,
            generator=generator,
            device=device,
        )
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            logits, diag = model(batch)
            loss = F.cross_entropy(logits.float(), batch.labels)
        predictions = logits.argmax(dim=-1)
        matches = predictions.eq(batch.labels)
        total += batch_size
        correct += int(matches.sum().item())
        loss_sum += float(loss.item()) * batch_size
        for occurrence in range(1, target_repeats + 1):
            mask = batch.query_occurrence_ids.eq(occurrence)
            count = int(mask.sum().item())
            if count:
                bucket = by_occurrence.setdefault(occurrence, [0, 0])
                bucket[0] += int(matches[mask].sum().item())
                bucket[1] += count
        for key, value in diag.items():
            diagnostics.setdefault(key, []).append(float(value.item()))
        if collect_cases and len(cases) < 64:
            probabilities = logits.float().softmax(dim=-1)
            confidence = probabilities.gather(1, predictions[:, None]).squeeze(1)
            for index in range(batch_size):
                cases.append(
                    {
                        "query_key": int(batch.query_key_ids[index].item()),
                        "query_occurrence": int(
                            batch.query_occurrence_ids[index].item()
                        ),
                        "target": int(batch.labels[index].item()),
                        "prediction": int(predictions[index].item()),
                        "correct": bool(matches[index].item()),
                        "confidence": float(confidence[index].item()),
                        "target_positions": [
                            int(item)
                            for item in batch.target_positions[index].tolist()
                        ],
                        "target_values": [
                            int(item)
                            for item in batch.target_values[index].tolist()
                        ],
                    }
                )
                if len(cases) >= 64:
                    break
    return {
        "accuracy": correct / max(total, 1),
        "ce": loss_sum / max(total, 1),
        "n": total,
        "sequence_length": sequence_length,
        "target_repeats": target_repeats,
        "by_occurrence": {
            str(key): values[0] / max(values[1], 1)
            for key, values in sorted(by_occurrence.items())
        },
        "diagnostics": {
            key: sum(values) / max(len(values), 1)
            for key, values in diagnostics.items()
        },
    }, cases


def pair_cases(
    base_cases: list[dict[str, Any]],
    candidate_cases: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    paired: list[dict[str, Any]] = []
    for base, candidate in zip(base_cases, candidate_cases, strict=True):
        identity = (
            "query_key",
            "query_occurrence",
            "target",
            "target_positions",
            "target_values",
        )
        if tuple(base[key] for key in identity) != tuple(
            candidate[key] for key in identity
        ):
            raise AssertionError("Matched evaluation cases diverged between arms")
        paired.append(
            {
                **{key: base[key] for key in identity},
                "base_prediction": base["prediction"],
                "base_correct": base["correct"],
                "base_confidence": base["confidence"],
                "candidate_prediction": candidate["prediction"],
                "candidate_correct": candidate["correct"],
                "candidate_confidence": candidate["confidence"],
            }
        )
    paired.sort(
        key=lambda row: (
            not (row["candidate_correct"] and not row["base_correct"]),
            row["base_correct"],
            -row["candidate_confidence"],
        )
    )
    return paired


def curve_svg(curves: list[dict[str, Any]]) -> str:
    if not curves:
        return ""
    width, height = 760, 250
    left, top, right, bottom = 48, 18, 16, 36
    max_step = max(row["step"] for row in curves)

    def point(step: int, value: float) -> tuple[float, float]:
        x = left + (width - left - right) * step / max(max_step, 1)
        y = top + (height - top - bottom) * (1.0 - value)
        return x, y

    lines = []
    for key, color in (
        ("content_acc", "#6b7280"),
        ("oracle_occurrence_rotary_acc", "#0f766e"),
    ):
        points = " ".join(
            f"{x:.1f},{y:.1f}"
            for x, y in (point(row["step"], row[key]) for row in curves)
        )
        lines.append(
            f'<polyline points="{points}" fill="none" stroke="{color}" '
            'stroke-width="3" />'
        )
    return (
        f'<svg viewBox="0 0 {width} {height}" role="img" '
        'aria-label="Training retrieval accuracy">'
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{height-bottom}" '
        'stroke="#9ca3af"/>'
        f'<line x1="{left}" y1="{height-bottom}" x2="{width-right}" '
        f'y2="{height-bottom}" stroke="#9ca3af"/>'
        + "".join(lines)
        + f'<text x="{left}" y="{height-8}" font-size="12">step 0</text>'
        + f'<text x="{width-right-70}" y="{height-8}" font-size="12">'
        f"step {max_step}</text>"
        + f'<text x="8" y="{top+5}" font-size="12">1.0</text>'
        + f'<text x="8" y="{height-bottom}" font-size="12">0.0</text>'
        + "</svg>"
    )


def render_html(
    *,
    config: dict[str, Any],
    score: dict[str, Any],
    curves: list[dict[str, Any]],
    cases: list[dict[str, Any]],
) -> str:
    eval_rows = []
    for name, result in score["eval"].items():
        base = result["content"]
        candidate = result["oracle_occurrence_rotary"]
        eval_rows.append(
            "<tr>"
            f"<td>{html.escape(name)}</td>"
            f"<td>{base['accuracy']:.4f}</td>"
            f"<td>{candidate['accuracy']:.4f}</td>"
            f"<td>{candidate['accuracy'] - base['accuracy']:+.4f}</td>"
            f"<td>{base['ce']:.4f}</td>"
            f"<td>{candidate['ce']:.4f}</td>"
            "</tr>"
        )
    case_rows = []
    for index, row in enumerate(cases[:16], start=1):
        values = " | ".join(
            f"{i + 1}:{value}"
            for i, value in enumerate(row["target_values"])
        )
        row_class = (
            "win"
            if row["candidate_correct"] and not row["base_correct"]
            else "tie"
        )
        case_rows.append(
            f'<tr class="{row_class}"><td>{index}</td>'
            f"<td>K{row['query_key']}</td>"
            f"<td>{values}</td>"
            f"<td>#{row['query_occurrence']}</td>"
            f"<td>V{row['target']}</td>"
            f"<td>V{row['base_prediction']} "
            f"({row['base_confidence']:.2f})</td>"
            f"<td>V{row['candidate_prediction']} "
            f"({row['candidate_confidence']:.2f})</td></tr>"
        )
    decision = score["decision"]
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>KDA occurrence-address causal probe</title>
<style>
body{{font:15px/1.5 system-ui,sans-serif;color:#17202a;margin:0;background:#f6f7f8}}
main{{max-width:1120px;margin:auto;padding:28px}}
h1{{font-size:28px;margin:0 0 8px}} h2{{font-size:19px;margin-top:30px}}
.summary{{border-left:5px solid #0f766e;background:white;padding:14px 18px}}
.grid{{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:12px}}
.metric{{background:white;border:1px solid #d9dde2;padding:12px;border-radius:6px}}
.metric b{{font-size:22px;display:block}}
table{{width:100%;border-collapse:collapse;background:white}}
th,td{{padding:9px;border:1px solid #d9dde2;text-align:left;vertical-align:top}}
th{{background:#eef1f3}} tr.win{{background:#e8f6f1}} tr.tie{{background:#fff}}
code{{background:#eef1f3;padding:2px 4px}} svg{{width:100%;background:white}}
.legend span{{margin-right:20px}} .base{{color:#6b7280}} .candidate{{color:#0f766e}}
@media(max-width:720px){{.grid{{grid-template-columns:1fr}}main{{padding:16px}}table{{font-size:12px}}}}
</style>
</head>
<body><main>
<h1>KDA repeated-key address probe</h1>
<p class="summary"><b>{html.escape(decision['label'])}</b><br>
{html.escape(decision['reason'])}</p>
<div class="grid">
<div class="metric"><span>Primary baseline</span><b>{decision['base_accuracy']:.4f}</b></div>
<div class="metric"><span>Primary occurrence rotary</span><b>{decision['candidate_accuracy']:.4f}</b></div>
<div class="metric"><span>Absolute gain</span><b>{decision['delta']:+.4f}</b></div>
</div>
<h2>Mechanism</h2>
<p>Both arms receive the same key, value, type, and occurrence embeddings.
The candidate adds a parameter-free rotation to KDA Q/K using the occurrence
index. The official pinned FLA <code>chunk_kda</code> recurrence is unchanged.</p>
<h2>Training signal</h2>
<p class="legend"><span class="base">content address</span>
<span class="candidate">oracle occurrence rotary</span></p>
{curve_svg(curves)}
<h2>Matched evaluation</h2>
<table><thead><tr><th>setting</th><th>base acc</th><th>candidate acc</th>
<th>delta</th><th>base CE</th><th>candidate CE</th></tr></thead>
<tbody>{''.join(eval_rows)}</tbody></table>
<h2>Concrete repeated-key cases</h2>
<p>Each list shows every value written to the queried key in chronological
order. The query asks for one numbered occurrence.</p>
<table><thead><tr><th>#</th><th>key</th><th>version history</th><th>query</th>
<th>target</th><th>base</th><th>occurrence rotary</th></tr></thead>
<tbody>{''.join(case_rows)}</tbody></table>
<h2>Provenance</h2>
<pre>{html.escape(json.dumps(config, indent=2, sort_keys=True))}</pre>
</main></body></html>"""


def strict_runtime_metadata(repo_root: Path) -> dict[str, Any]:
    persistent_root = Path(
        os.environ.get("PERSIST_ROOT", "/huyang2/double-loop")
    ).resolve()
    if persistent_root != Path("/huyang2/double-loop"):
        raise RuntimeError(
            "PERSIST_ROOT must resolve to /huyang2/double-loop, got "
            f"{persistent_root}"
        )
    for variable in (
        "XDG_CACHE_HOME",
        "TRITON_CACHE_DIR",
        "TORCHINDUCTOR_CACHE_DIR",
        "TORCH_EXTENSIONS_DIR",
        "TMPDIR",
    ):
        path = os.environ.get(variable, "")
        if not path.startswith("/huyang2/double-loop/"):
            raise RuntimeError(
                f"{variable} must be under /huyang2/double-loop, got {path!r}"
            )
    if os.environ.get("FLA_DISABLE_BACKEND_DISPATCH") != "1":
        raise RuntimeError("FLA_DISABLE_BACKEND_DISPATCH=1 is required")
    if os.environ.get("FLA_CONV_BACKEND") != "triton":
        raise RuntimeError("FLA_CONV_BACKEND=triton is required")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required; CPU model execution is forbidden")
    if torch.cuda.device_count() != 1:
        raise RuntimeError(
            f"Exactly one visible GPU is required, got {torch.cuda.device_count()}"
        )

    import fla
    from fla.ops.backends import _DISPATCH_DISABLED
    from fla.ops.kda import chunk_kda

    if not _DISPATCH_DISABLED:
        raise RuntimeError("FLA backend dispatch is active")
    package_root = Path(fla.__file__).resolve().parent
    if not str(package_root).startswith("/huyang2/double-loop/"):
        raise RuntimeError(f"FLA package is outside persistent root: {package_root}")
    source_marker = persistent_root / ".cache" / "fla-source-sha"
    source_sha = source_marker.read_text(encoding="utf-8").strip()
    if source_sha != EXPECTED_FLA_SHA:
        raise RuntimeError(
            f"FLA source SHA mismatch: {source_sha} != {EXPECTED_FLA_SHA}"
        )
    wheel = (
        repo_root
        / "wheelhouse"
        / "flash_linear_attention-0.5.2-9c8e42e-py3-none-any.whl"
    )
    wheel_sha = sha256_file(wheel)
    if wheel_sha != EXPECTED_FLA_WHEEL_SHA256:
        raise RuntimeError(
            f"FLA wheel SHA mismatch: {wheel_sha} != {EXPECTED_FLA_WHEEL_SHA256}"
        )
    properties = torch.cuda.get_device_properties(0)
    return {
        "cuda_device": torch.cuda.get_device_name(0),
        "cuda_capability": list(torch.cuda.get_device_capability(0)),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "torch_version": torch.__version__,
        "fla_version": fla.__version__,
        "fla_package_root": str(package_root),
        "persistent_root": str(persistent_root),
        "fla_source_sha": source_sha,
        "fla_wheel_sha256": wheel_sha,
        "fla_backend_dispatch_disabled": bool(_DISPATCH_DISABLED),
        "kda_kernel_module": chunk_kda.__module__,
        "total_memory_bytes": int(properties.total_memory),
    }


def train(
    args: argparse.Namespace,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    device = torch.device("cuda:0")
    torch.manual_seed(args.seed)
    base = KDAOccurrenceMemory(
        key_vocab_size=args.key_vocab_size,
        value_vocab_size=args.value_vocab_size,
        max_occurrence=args.max_occurrence,
        d_model=args.d_model,
        heads=args.heads,
        head_dim=args.head_dim,
        expand_v=args.expand_v,
        address_mode="content",
    ).to(device)
    candidate = KDAOccurrenceMemory(
        key_vocab_size=args.key_vocab_size,
        value_vocab_size=args.value_vocab_size,
        max_occurrence=args.max_occurrence,
        d_model=args.d_model,
        heads=args.heads,
        head_dim=args.head_dim,
        expand_v=args.expand_v,
        address_mode="oracle_occurrence_rotary",
    ).to(device)
    candidate.load_state_dict(base.state_dict(), strict=True)
    if sum(p.numel() for p in base.parameters()) != sum(
        p.numel() for p in candidate.parameters()
    ):
        raise AssertionError("Parameter counts differ between matched arms")

    models = {
        "content": base,
        "oracle_occurrence_rotary": candidate,
    }
    optimizers = {
        name: torch.optim.AdamW(
            model.parameters(),
            lr=args.learning_rate,
            betas=(0.9, 0.95),
            weight_decay=args.weight_decay,
        )
        for name, model in models.items()
    }
    curves: list[dict[str, Any]] = []
    train_wall = {name: 0.0 for name in models}
    generator = torch.Generator(device=device).manual_seed(args.data_seed)
    torch.cuda.reset_peak_memory_stats()

    for step in range(1, args.steps + 1):
        batch = generate_retrieval_batch(
            batch_size=args.batch_size,
            sequence_length=args.train_length,
            target_repeats=args.train_repeats,
            key_vocab_size=args.key_vocab_size,
            value_vocab_size=args.value_vocab_size,
            max_occurrence=args.max_occurrence,
            generator=generator,
            device=device,
        )
        row: dict[str, Any] = {"step": step}
        for name, model in models.items():
            model.train()
            optimizer = optimizers[name]
            optimizer.zero_grad(set_to_none=True)
            torch.cuda.synchronize()
            start = time.perf_counter()
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits, diagnostics = model(batch)
                loss = F.cross_entropy(logits.float(), batch.labels)
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                args.grad_clip,
            )
            optimizer.step()
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - start
            train_wall[name] += elapsed
            row[f"{name}_loss"] = float(loss.item())
            row[f"{name}_acc"] = float(
                logits.argmax(dim=-1).eq(batch.labels).float().mean().item()
            )
            row[f"{name}_grad_norm"] = float(grad_norm.item())
            row[f"{name}_state_rms"] = float(
                diagnostics["terminal_state_rms"].item()
            )
        if step == 1 or step % args.log_every == 0 or step == args.steps:
            curves.append(row)
            print(json.dumps(row, sort_keys=True), flush=True)

    eval_specs = parse_eval_specs(args.eval_specs)
    eval_results: dict[str, Any] = {}
    paired_cases: list[dict[str, Any]] = []
    for index, (length, repeats) in enumerate(eval_specs):
        setting = f"len{length}_repeat{repeats}"
        eval_results[setting] = {}
        arm_cases: dict[str, list[dict[str, Any]]] = {}
        for name, model in models.items():
            metrics, cases = evaluate_model(
                model,
                batch_size=args.eval_batch_size,
                batches=args.eval_batches,
                sequence_length=length,
                target_repeats=repeats,
                key_vocab_size=args.key_vocab_size,
                value_vocab_size=args.value_vocab_size,
                max_occurrence=args.max_occurrence,
                seed=args.eval_seed + index,
                device=device,
                collect_cases=index == len(eval_specs) - 1,
            )
            eval_results[setting][name] = metrics
            arm_cases[name] = cases
        if arm_cases["content"]:
            paired_cases = pair_cases(
                arm_cases["content"],
                arm_cases["oracle_occurrence_rotary"],
            )

    primary_name = list(eval_results)[-1]
    primary = eval_results[primary_name]
    base_accuracy = float(primary["content"]["accuracy"])
    candidate_accuracy = float(primary["oracle_occurrence_rotary"]["accuracy"])
    delta = candidate_accuracy - base_accuracy
    supported = (
        delta >= args.success_delta
        and candidate_accuracy >= args.success_accuracy
    )
    rejected = delta < args.reject_delta
    if supported:
        label = "Address-collision hypothesis supported"
        reason = (
            "A parameter-free query-matchable occurrence address produced a "
            "large OOD retrieval gain under the unchanged official KDA recurrence."
        )
    elif rejected:
        label = "Occurrence binding rejected at this gate"
        reason = (
            "Oracle occurrence binding did not produce the preregistered gain; "
            "do not build a learned counter or sweep rotary settings."
        )
    else:
        label = "Inconclusive causal signal"
        reason = (
            "The gain is nonzero but below the strong gate. Inspect the learning "
            "curve once; do not start a hyperparameter table."
        )
    score = {
        "primary_setting": primary_name,
        "eval": eval_results,
        "train_wall_sec": train_wall,
        "peak_cuda_memory_bytes": int(torch.cuda.max_memory_allocated()),
        "parameter_count": sum(p.numel() for p in base.parameters()),
        "decision": {
            "label": label,
            "reason": reason,
            "supported": supported,
            "rejected": rejected,
            "base_accuracy": base_accuracy,
            "candidate_accuracy": candidate_accuracy,
            "delta": delta,
            "success_delta": args.success_delta,
            "success_accuracy": args.success_accuracy,
            "reject_delta": args.reject_delta,
        },
    }
    return score, curves, paired_cases


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--eval-batch-size", type=int, default=128)
    parser.add_argument("--eval-batches", type=int, default=4)
    parser.add_argument("--train-length", type=int, default=128)
    parser.add_argument("--train-repeats", type=int, default=8)
    parser.add_argument("--eval-specs", default="128:8,512:16")
    parser.add_argument("--key-vocab-size", type=int, default=64)
    parser.add_argument("--value-vocab-size", type=int, default=32)
    parser.add_argument("--max-occurrence", type=int, default=64)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--head-dim", type=int, default=32)
    parser.add_argument("--expand-v", type=float, default=2.0)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--log-every", type=int, default=25)
    parser.add_argument("--seed", type=int, default=52)
    parser.add_argument("--data-seed", type=int, default=5201)
    parser.add_argument("--eval-seed", type=int, default=5299)
    parser.add_argument("--success-delta", type=float, default=0.20)
    parser.add_argument("--success-accuracy", type=float, default=0.70)
    parser.add_argument("--reject-delta", type=float, default=0.10)
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    args.out_dir.mkdir(parents=True, exist_ok=True)
    runtime = strict_runtime_metadata(repo_root)
    config = {
        **vars(args),
        "out_dir": str(args.out_dir),
        "source_sha": os.environ.get("SOURCE_SHA", ""),
        "hypothesis": (
            "KDA decay tracks age but cannot create a query-matchable address "
            "for repeated writes to one semantic key. With identical additive "
            "occurrence metadata, parameter-free occurrence rotary Q/K binding "
            "should sharply improve nth-occurrence retrieval."
        ),
        "prediction": (
            "On len512/repeat16, occurrence rotary improves accuracy by at least "
            "+0.20 and reaches at least 0.70."
        ),
        "kill_criteria": (
            "If OOD delta is below +0.10 after 300 steps, stop this direction; "
            "do not implement a learned counter or sweep beta/rotary frequencies."
        ),
        "runtime": runtime,
    }
    (args.out_dir / "config.json").write_text(
        json.dumps(config, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    shutil.copy2(__file__, args.out_dir / "kda_occurrence_probe.py.snapshot")

    started = time.time()
    score, curves, cases = train(args)
    score["wall_sec"] = time.time() - started
    (args.out_dir / "score.json").write_text(
        json.dumps(score, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (args.out_dir / "curves.json").write_text(
        json.dumps(curves, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (args.out_dir / "cases.json").write_text(
        json.dumps(cases, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (args.out_dir / "index.html").write_text(
        render_html(config=config, score=score, curves=curves, cases=cases),
        encoding="utf-8",
    )
    print(json.dumps(score["decision"], sort_keys=True), flush=True)
    print(f"artifact_dir={args.out_dir}", flush=True)


if __name__ == "__main__":
    main()
