from __future__ import annotations

import argparse
import hashlib
import html
import inspect
import json
import math
import subprocess
import time
from itertools import chain
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
import transformers
from transformers import (
    AutoTokenizer,
    BertConfig,
    BertForMaskedLM,
    DataCollatorForLanguageModeling,
    set_seed,
)


EXPECTED_GPU_UUID = "GPU-53e9f3b4-2966-65d3-6614-09c540921519"
EXPECTED_TRANSFORMERS_VERSION = "4.46.3"
EXPECTED_BERT_SOURCE_SHA256 = (
    "3493bff5da90fdcce98dad5c84aafe4d3ce1c550dcd93bc99289309953559eca"
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def visible_gpu_uuid() -> str:
    return subprocess.run(
        ["nvidia-smi", "--query-gpu=uuid", "--format=csv,noheader"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def state_hash(model: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        digest.update(name.encode("utf-8"))
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def load_grouped_examples(
    validation_json: Path,
    tokenizer: Any,
    sequence_length: int,
) -> list[dict[str, list[int]]]:
    texts = [
        json.loads(line)["text"]
        for line in validation_json.read_text(encoding="utf-8").splitlines()
    ]
    tokenized: dict[str, list[list[int]]] = {}
    for start in range(0, len(texts), 256):
        encoded = tokenizer(
            texts[start : start + 256],
            return_special_tokens_mask=True,
        )
        for key, rows in encoded.items():
            tokenized.setdefault(key, []).extend(rows)

    concatenated = {
        key: list(chain.from_iterable(rows)) for key, rows in tokenized.items()
    }
    total_length = len(concatenated["input_ids"])
    total_length = (total_length // sequence_length) * sequence_length
    return [
        {
            key: values[start : start + sequence_length]
            for key, values in concatenated.items()
        }
        for start in range(0, total_length, sequence_length)
    ]


def token_text(tokenizer: Any, token_id: int) -> str:
    return str(tokenizer.convert_ids_to_tokens(int(token_id)))


def build_matched_models(
    model_dir: Path,
    checkpoint_file: Path,
) -> tuple[BertForMaskedLM, BertForMaskedLM, dict[str, Any]]:
    if checkpoint_file.parent.resolve() != model_dir.resolve():
        raise RuntimeError("Registered checkpoint must live directly in model_dir")
    if checkpoint_file.suffix not in {".bin", ".safetensors"}:
        raise RuntimeError(f"Unsupported checkpoint format: {checkpoint_file.name}")
    bidirectional = BertForMaskedLM.from_pretrained(
        model_dir,
        local_files_only=True,
        attn_implementation="sdpa",
        use_safetensors=checkpoint_file.suffix == ".safetensors",
    )
    bidirectional.config.use_cache = False
    if bidirectional.config.is_decoder:
        raise RuntimeError("Official checkpoint unexpectedly uses causal attention")

    expected_shape = {
        "hidden_size": 128,
        "num_hidden_layers": 2,
        "num_attention_heads": 2,
        "intermediate_size": 512,
        "vocab_size": 30_522,
    }
    actual_shape = {
        key: int(getattr(bidirectional.config, key)) for key in expected_shape
    }
    if actual_shape != expected_shape:
        raise RuntimeError(
            f"Checkpoint is not the registered BERT-Tiny architecture: {actual_shape}"
        )

    causal_dict = bidirectional.config.to_dict()
    causal_dict.update(
        {
            "is_decoder": True,
            "add_cross_attention": False,
            "use_cache": False,
            "_attn_implementation": "sdpa",
        }
    )
    causal_config = BertConfig.from_dict(causal_dict)
    causal_config._attn_implementation = "sdpa"
    causal = BertForMaskedLM(causal_config)
    causal.load_state_dict(bidirectional.state_dict(), strict=True)

    if set(causal.state_dict()) != set(bidirectional.state_dict()):
        raise RuntimeError("Causal conversion changed checkpoint state keys")
    max_diff = max(
        float((causal.state_dict()[key] - tensor).abs().max().item())
        for key, tensor in bidirectional.state_dict().items()
    )
    if max_diff != 0.0:
        raise RuntimeError(f"Causal conversion changed pretrained tensors: {max_diff}")
    if sum(p.numel() for p in causal.parameters()) != sum(
        p.numel() for p in bidirectional.parameters()
    ):
        raise RuntimeError("Causal conversion changed parameter count")

    diagnostics = {
        "architecture": expected_shape,
        "parameters": sum(p.numel() for p in bidirectional.parameters()),
        "state_keys": len(bidirectional.state_dict()),
        "shared_weight_max_diff": max_diff,
        "bidirectional_state_hash": state_hash(bidirectional),
        "causal_state_hash": state_hash(causal),
    }
    return bidirectional, causal, diagnostics


def masked_metrics(
    model: BertForMaskedLM,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    labels: torch.Tensor,
    batch_size: int,
) -> tuple[dict[str, float | int], torch.Tensor, torch.Tensor]:
    model.eval()
    target_log_probs_cpu = []
    predictions_cpu = []
    loss_sum = 0.0
    correct = 0
    count = 0
    for start in range(0, input_ids.shape[0], batch_size):
        batch_input = input_ids[start : start + batch_size].cuda()
        batch_attention = attention_mask[start : start + batch_size].cuda()
        batch_labels = labels[start : start + batch_size].cuda()
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model(
                input_ids=batch_input,
                attention_mask=batch_attention,
            ).logits.float()
        mask = batch_labels != -100
        predictions = logits.argmax(dim=-1)
        safe_labels = batch_labels.masked_fill(~mask, 0)
        target_log_probs = logits.log_softmax(dim=-1).gather(
            dim=-1,
            index=safe_labels.unsqueeze(-1),
        ).squeeze(-1)
        target_log_probs = target_log_probs.masked_fill(~mask, float("nan"))
        loss_sum += float(
            F.cross_entropy(
                logits[mask], batch_labels[mask], reduction="sum"
            ).item()
        )
        correct += int(((predictions == batch_labels) & mask).sum().item())
        count += int(mask.sum().item())
        target_log_probs_cpu.append(target_log_probs.cpu())
        predictions_cpu.append(predictions.cpu())
    return (
        {
            "masked_accuracy": correct / count,
            "masked_ce": loss_sum / count,
            "masked_tokens": count,
        },
        torch.cat(target_log_probs_cpu),
        torch.cat(predictions_cpu),
    )


def benchmark_forward(
    model: BertForMaskedLM,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    batch_size: int,
    warmup_steps: int = 5,
    measured_steps: int = 50,
) -> dict[str, float | int]:
    model.eval()
    batch_input = input_ids[:batch_size].cuda()
    batch_attention = attention_mask[:batch_size].cuda()
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        for _ in range(warmup_steps):
            model(input_ids=batch_input, attention_mask=batch_attention)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        for _ in range(measured_steps):
            model(input_ids=batch_input, attention_mask=batch_attention)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    return {
        "warmup_steps": warmup_steps,
        "measured_steps": measured_steps,
        "batch_size": int(batch_input.shape[0]),
        "sequence_length": int(batch_input.shape[1]),
        "elapsed_seconds": elapsed,
        "input_tokens_per_second": (
            batch_input.numel() * measured_steps / elapsed
        ),
        "peak_cuda_memory_bytes": torch.cuda.max_memory_allocated(),
    }


def future_dependency(
    model: BertForMaskedLM,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    labels: torch.Tensor,
    replacement_id: int,
) -> float:
    probe_rows = []
    probe_positions = []
    for row in range(input_ids.shape[0]):
        positions = torch.nonzero(labels[row] != -100).flatten()
        positions = positions[positions < input_ids.shape[1] - 8]
        if len(positions):
            probe_rows.append(row)
            probe_positions.append(int(positions[0]))
        if len(probe_rows) == 16:
            break
    if not probe_rows:
        raise RuntimeError("No fixed-mask position can probe future dependency")

    original = input_ids[probe_rows].cuda()
    mask = attention_mask[probe_rows].cuda()
    altered = original.clone()
    for row, position in enumerate(probe_positions):
        suffix = torch.arange(original.shape[1], device="cuda") > position
        visible_suffix = suffix & mask[row].bool()
        altered[row, visible_suffix] = replacement_id

    model.eval()
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        original_logits = model(input_ids=original, attention_mask=mask).logits.float()
        altered_logits = model(input_ids=altered, attention_mask=mask).logits.float()
    differences = [
        (original_logits[row, position] - altered_logits[row, position])
        .abs()
        .mean()
        for row, position in enumerate(probe_positions)
    ]
    return float(torch.stack(differences).mean().item())


def finite_backward(
    model: BertForMaskedLM,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    labels: torch.Tensor,
) -> dict[str, float]:
    model.train()
    model.zero_grad(set_to_none=True)
    output = model(
        input_ids=input_ids[:2].cuda(),
        attention_mask=attention_mask[:2].cuda(),
        labels=labels[:2].cuda(),
    )
    output.loss.backward()
    gradients = [p.grad for p in model.parameters() if p.grad is not None]
    if not gradients or not all(torch.isfinite(gradient).all() for gradient in gradients):
        raise RuntimeError("Missing or non-finite pretrained BERT gradient")
    return {
        "loss": float(output.loss.item()),
        "max_gradient": max(float(g.abs().max().item()) for g in gradients),
    }


def select_cases(
    tokenizer: Any,
    originals: torch.Tensor,
    labels: torch.Tensor,
    bidirectional_target_log_probs: torch.Tensor,
    bidirectional_predictions: torch.Tensor,
    causal_target_log_probs: torch.Tensor,
    causal_predictions: torch.Tensor,
    count: int,
) -> list[dict[str, Any]]:
    cases = []
    for case_index in range(originals.shape[0]):
        positions = torch.nonzero(labels[case_index] != -100).flatten().tolist()
        rows = []
        ce_gain = 0.0
        repairs = 0
        for position in positions:
            target_id = int(labels[case_index, position])
            bidir_target_log_prob = bidirectional_target_log_probs[
                case_index, position
            ]
            causal_target_log_prob = causal_target_log_probs[case_index, position]
            bidir_prediction = int(bidirectional_predictions[case_index, position])
            causal_prediction = int(causal_predictions[case_index, position])
            token_ce_gain = float(bidir_target_log_prob - causal_target_log_prob)
            ce_gain += token_ce_gain
            repairs += int(
                bidir_prediction == target_id and causal_prediction != target_id
            )
            rows.append(
                {
                    "position": position,
                    "target": token_text(tokenizer, target_id),
                    "bidirectional_prediction": token_text(
                        tokenizer, bidir_prediction
                    ),
                    "causal_prediction": token_text(tokenizer, causal_prediction),
                    "bidirectional_correct": bidir_prediction == target_id,
                    "causal_correct": causal_prediction == target_id,
                    "bidirectional_probability": float(bidir_target_log_prob.exp()),
                    "causal_probability": float(causal_target_log_prob.exp()),
                    "token_ce_gain": token_ce_gain,
                }
            )
        cases.append(
            {
                "case_index": case_index,
                "case_id": hashlib.sha256(
                    originals[case_index].numpy().tobytes()
                ).hexdigest()[:16],
                "repairs": repairs,
                "mean_ce_gain": ce_gain / len(rows),
                "tokens": [
                    token_text(tokenizer, token_id)
                    for token_id in originals[case_index].tolist()
                ],
                "masked_tokens": rows,
            }
        )
    cases.sort(
        key=lambda row: (-row["repairs"], -row["mean_ce_gain"], row["case_index"])
    )
    return cases[:count]


def render_html(score: dict[str, Any], cases: list[dict[str, Any]]) -> str:
    metrics = score["metrics"]
    cards = []
    for case in cases:
        masked = {row["position"]: row for row in case["masked_tokens"]}
        tokens = []
        for position, token in enumerate(case["tokens"]):
            if position in masked:
                tokens.append('<span class="mask">[MASK]</span>')
            else:
                tokens.append(f"<span>{html.escape(token)}</span>")
        rows = []
        for row in case["masked_tokens"]:
            bidir_class = "good" if row["bidirectional_correct"] else "bad"
            causal_class = "good" if row["causal_correct"] else "bad"
            rows.append(
                "<tr>"
                f"<td>{row['position']}</td>"
                f"<td>{html.escape(row['target'])}</td>"
                f'<td class="{bidir_class}">{html.escape(row["bidirectional_prediction"])}</td>'
                f"<td>{row['bidirectional_probability']:.4f}</td>"
                f'<td class="{causal_class}">{html.escape(row["causal_prediction"])}</td>'
                f"<td>{row['causal_probability']:.4f}</td>"
                f"<td>{row['token_ce_gain']:+.4f}</td>"
                "</tr>"
            )
        cards.append(
            '<section class="case">'
            f"<h2>Case {case['case_index']} | right-context repairs "
            f"{case['repairs']} | mean CE gain {case['mean_ce_gain']:+.3f}</h2>"
            f'<div class="tokens">{" ".join(tokens)}</div>'
            "<div class=\"table-wrap\"><table><thead><tr><th>pos</th>"
            "<th>target</th><th>bidirectional</th><th>P target</th>"
            "<th>strict causal</th><th>P target</th><th>CE gain</th>"
            f"</tr></thead><tbody>{''.join(rows)}</tbody></table></div></section>"
        )
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>P-CAUSAL-018 pretrained BERT carrier</title>
<style>
body{{font:14px/1.45 ui-monospace,SFMono-Regular,Menlo,monospace;margin:0;color:#17202a;background:#f4f6f7}}
header{{background:#17202a;color:#fff;padding:24px max(24px,calc((100% - 1180px)/2))}}
main{{max-width:1180px;margin:0 auto;padding:20px}} .summary{{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:8px}}
.metric,.case{{background:#fff;border:1px solid #ccd1d1;border-radius:6px;padding:14px}} .case{{margin-top:14px}}
.metric b{{display:block;font-size:20px}} .tokens{{padding:12px;background:#f8f9f9;line-height:2;overflow-wrap:anywhere}}
.tokens span{{padding:2px}} .mask,.good{{color:#117864;font-weight:700}} .bad{{color:#b03a2e;font-weight:700}}
.table-wrap{{overflow-x:auto}} table{{border-collapse:collapse;width:100%;margin-top:12px;min-width:760px}}
th,td{{border-bottom:1px solid #e5e7e9;padding:7px;text-align:left;vertical-align:top}}
@media(max-width:720px){{.summary{{grid-template-columns:1fr 1fr}} table{{font-size:12px}}}}
</style></head><body><header><h1>P-CAUSAL-018 pretrained BERT carrier gate</h1>
<p>One official pretrained checkpoint, one fixed corruption, and two attention masks. This validates the carrier only; it is not a FutureSeed result.</p></header>
<main><div class="summary">
<div class="metric">bidirectional accuracy<b>{metrics['bidirectional']['masked_accuracy']:.4f}</b></div>
<div class="metric">strict-causal accuracy<b>{metrics['causal']['masked_accuracy']:.4f}</b></div>
<div class="metric">accuracy delta<b>{metrics['delta']['masked_accuracy']:+.4f}</b></div>
<div class="metric">CE improvement<b>{metrics['delta']['causal_minus_bidirectional_ce']:+.4f}</b></div>
</div>{''.join(cards)}</main></body></html>"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--checkpoint-file", type=Path, required=True)
    parser.add_argument("--validation-json", type=Path, required=True)
    parser.add_argument("--data-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--sequence-length", type=int, default=128)
    parser.add_argument("--candidate-windows", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--cases", type=int, default=12)
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args()

    if torch.cuda.device_count() != 1 or visible_gpu_uuid() != EXPECTED_GPU_UUID:
        raise RuntimeError("Evaluation requires the exact single GPU1")
    if transformers.__version__ != EXPECTED_TRANSFORMERS_VERSION:
        raise RuntimeError(f"Unexpected Transformers version: {transformers.__version__}")
    bert_source = Path(inspect.getfile(BertForMaskedLM))
    if sha256(bert_source) != EXPECTED_BERT_SOURCE_SHA256:
        raise RuntimeError("Unexpected BertForMaskedLM implementation")

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_dir,
        local_files_only=True,
        use_fast=True,
    )
    if tokenizer.mask_token_id is None or len(tokenizer) != 30_522:
        raise RuntimeError("Unexpected BERT-Tiny tokenizer")
    examples = load_grouped_examples(
        args.validation_json, tokenizer, args.sequence_length
    )[: args.candidate_windows]
    if len(examples) != args.candidate_windows:
        raise RuntimeError("Validation corpus did not provide the registered windows")

    originals = torch.tensor(
        [example["input_ids"] for example in examples], dtype=torch.long
    )
    set_seed(args.seed)
    collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm_probability=0.15,
        return_tensors="pt",
    )
    corrupted = collator(examples)
    input_ids = corrupted["input_ids"]
    attention_mask = corrupted["attention_mask"]
    labels = corrupted["labels"]

    bidirectional, causal, model_diagnostics = build_matched_models(
        args.model_dir, args.checkpoint_file
    )
    bidirectional = bidirectional.cuda()
    causal = causal.cuda()
    torch.cuda.reset_peak_memory_stats()

    bidir_backward = finite_backward(
        bidirectional, input_ids, attention_mask, labels
    )
    causal_backward = finite_backward(causal, input_ids, attention_mask, labels)
    bidir_dependency = future_dependency(
        bidirectional,
        input_ids,
        attention_mask,
        labels,
        tokenizer.unk_token_id,
    )
    causal_dependency = future_dependency(
        causal,
        input_ids,
        attention_mask,
        labels,
        tokenizer.unk_token_id,
    )
    if causal_dependency != 0.0:
        raise RuntimeError(f"Strict-causal checkpoint leaks future input: {causal_dependency}")
    if bidir_dependency <= 1e-7:
        raise RuntimeError("Pretrained bidirectional checkpoint ignores future input")

    bidir_metrics, bidir_target_log_probs, bidir_predictions = masked_metrics(
        bidirectional,
        input_ids,
        attention_mask,
        labels,
        args.batch_size,
    )
    causal_metrics, causal_target_log_probs, causal_predictions = masked_metrics(
        causal,
        input_ids,
        attention_mask,
        labels,
        args.batch_size,
    )
    bidirectional_performance = benchmark_forward(
        bidirectional, input_ids, attention_mask, args.batch_size
    )
    causal_performance = benchmark_forward(
        causal, input_ids, attention_mask, args.batch_size
    )
    accuracy_delta = (
        float(bidir_metrics["masked_accuracy"])
        - float(causal_metrics["masked_accuracy"])
    )
    ce_improvement = (
        float(causal_metrics["masked_ce"])
        - float(bidir_metrics["masked_ce"])
    )
    checks = {
        "bidirectional_accuracy_at_least_0.10": (
            bidir_metrics["masked_accuracy"] >= 0.10
        ),
        "right_context_quality_delta": (
            accuracy_delta >= 0.03 or ce_improvement >= 0.20
        ),
        "strict_causal_future_dependency_zero": causal_dependency == 0.0,
        "bidirectional_future_dependency_nonzero": bidir_dependency > 1e-7,
        "shared_pretrained_weights_exact": (
            model_diagnostics["shared_weight_max_diff"] == 0.0
            and model_diagnostics["bidirectional_state_hash"]
            == model_diagnostics["causal_state_hash"]
        ),
    }
    passed = all(checks.values())
    score = {
        "plan": "P-CAUSAL-018",
        "status": "carrier_opened" if passed else "discarded",
        "claim_boundary": (
            "Carrier validity only; no GDN2 or FutureSeed model is evaluated."
        ),
        "assets": {
            "checkpoint_file": str(args.checkpoint_file),
            "checkpoint_sha256": sha256(args.checkpoint_file),
            "validation_sha256": sha256(args.validation_json),
            "data_manifest_sha256": sha256(args.data_manifest),
            "bert_source_sha256": sha256(bert_source),
        },
        "protocol": {
            "sequence_length": args.sequence_length,
            "candidate_windows": len(examples),
            "seed": args.seed,
            "mask_probability": 0.15,
            "same_checkpoint_both_arms": True,
            "only_difference": "full bidirectional versus strict causal attention mask",
        },
        "model": model_diagnostics,
        "metrics": {
            "bidirectional": bidir_metrics,
            "causal": causal_metrics,
            "delta": {
                "masked_accuracy": accuracy_delta,
                "causal_minus_bidirectional_ce": ce_improvement,
            },
        },
        "diagnostics": {
            "bidirectional_future_dependency": bidir_dependency,
            "causal_future_dependency": causal_dependency,
            "bidirectional_backward": bidir_backward,
            "causal_backward": causal_backward,
            "performance": {
                "bidirectional": bidirectional_performance,
                "causal": causal_performance,
            },
        },
        "gate": {"passed": passed, "checks": checks},
        "data_manifest": json.loads(args.data_manifest.read_text(encoding="utf-8")),
    }
    cases = select_cases(
        tokenizer,
        originals,
        labels,
        bidir_target_log_probs,
        bidir_predictions,
        causal_target_log_probs,
        causal_predictions,
        args.cases,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "score.json").write_text(
        json.dumps(score, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (args.output_dir / "hardest_cases.json").write_text(
        json.dumps(cases, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (args.output_dir / "index.html").write_text(
        render_html(score, cases), encoding="utf-8"
    )
    print(json.dumps(score, indent=2, sort_keys=True))
    raise SystemExit(0 if passed else 2)


if __name__ == "__main__":
    main()
