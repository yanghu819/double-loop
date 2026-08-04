from __future__ import annotations

import argparse
import hashlib
import html
import json
import math
import subprocess
from itertools import chain
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import (
    AutoTokenizer,
    BertForMaskedLM,
    DataCollatorForLanguageModeling,
    set_seed,
)


EXPECTED_GPU_UUID = "GPU-53e9f3b4-2966-65d3-6614-09c540921519"


def visible_gpu_uuid() -> str:
    return subprocess.run(
        ["nvidia-smi", "--query-gpu=uuid", "--format=csv,noheader"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def load_grouped_examples(
    validation_json: Path,
    tokenizer,
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


def token_text(tokenizer, token_id: int) -> str:
    token = tokenizer.convert_ids_to_tokens(int(token_id))
    return str(token).replace("##", "")


def render_html(summary: dict, cases: list[dict]) -> str:
    cards = []
    for case in cases:
        spans = []
        masked = {row["position"]: row for row in case["masked_tokens"]}
        for position, token in enumerate(case["tokens"]):
            if position in masked:
                row = masked[position]
                css = "correct" if row["correct"] else "wrong"
                title = html.escape(
                    f"target={row['target']} prediction={row['prediction']}"
                )
                spans.append(
                    f'<span class="{css}" title="{title}">[MASK]</span>'
                )
            else:
                spans.append(f"<span>{html.escape(token)}</span>")
        rows = []
        for row in case["masked_tokens"]:
            top = ", ".join(
                f"{html.escape(item['token'])} ({item['probability']:.3f})"
                for item in row["top5"]
            )
            css = "correct" if row["correct"] else "wrong"
            rows.append(
                "<tr>"
                f"<td>{row['position']}</td>"
                f"<td>{html.escape(row['target'])}</td>"
                f'<td class="{css}">{html.escape(row["prediction"])}</td>'
                f"<td>{row['true_probability']:.4f}</td>"
                f"<td>{top}</td>"
                "</tr>"
            )
        cards.append(
            '<section class="case">'
            f"<h2>Case {case['case_index']} | {case['errors']} / "
            f"{case['masked_count']} wrong</h2>"
            f'<div class="tokens">{" ".join(spans)}</div>'
            "<table><thead><tr><th>pos</th><th>target</th>"
            "<th>prediction</th><th>P(target)</th><th>top 5</th>"
            f"</tr></thead><tbody>{''.join(rows)}</tbody></table></section>"
        )
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>P-CAUSAL-009 fixed-mask cases</title>
<style>
body{{font:14px/1.45 ui-monospace,SFMono-Regular,Menlo,monospace;margin:0;color:#17202a;background:#f4f6f7}}
header{{background:#17202a;color:#fff;padding:24px max(24px,calc((100% - 1180px)/2))}}
main{{max-width:1180px;margin:0 auto;padding:20px}} .summary{{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:8px}}
.metric,.case{{background:#fff;border:1px solid #ccd1d1;border-radius:6px;padding:14px}} .case{{margin-top:14px}}
.metric b{{display:block;font-size:20px}} .tokens{{padding:12px;background:#f8f9f9;line-height:2;overflow-wrap:anywhere}}
.tokens span{{padding:2px}} .correct{{color:#117864;font-weight:700}} .wrong{{color:#b03a2e;font-weight:700}}
table{{border-collapse:collapse;width:100%;margin-top:12px}} th,td{{border-bottom:1px solid #e5e7e9;padding:7px;text-align:left;vertical-align:top}}
@media(max-width:720px){{.summary{{grid-template-columns:1fr 1fr}} table{{font-size:12px}}}}
</style></head><body><header><h1>P-CAUSAL-009 fixed-mask audit</h1><p>Deterministic masks on the official grouped validation stream. This is diagnostic, not the registered dynamic-mask metric.</p></header>
<main><div class="summary">
<div class="metric">accuracy<b>{summary['masked_accuracy']:.4f}</b></div>
<div class="metric">cross entropy<b>{summary['masked_ce']:.4f}</b></div>
<div class="metric">masked tokens<b>{summary['masked_tokens']}</b></div>
<div class="metric">candidate windows<b>{summary['candidate_windows']}</b></div>
</div>{''.join(cards)}</main></body></html>"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--validation-json", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--sequence-length", type=int, default=128)
    parser.add_argument("--candidate-windows", type=int, default=64)
    parser.add_argument("--cases", type=int, default=8)
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args()

    if torch.cuda.device_count() != 1 or visible_gpu_uuid() != EXPECTED_GPU_UUID:
        raise RuntimeError("Visualization requires the exact single GPU1")
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer, local_files_only=True, use_fast=True
    )
    model = BertForMaskedLM.from_pretrained(
        args.model_dir, local_files_only=True
    ).cuda().eval()
    if model.config.is_decoder:
        raise RuntimeError("Expected the completed bidirectional BERT arm")

    examples = load_grouped_examples(
        args.validation_json, tokenizer, args.sequence_length
    )[: args.candidate_windows]
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
    labels = corrupted["labels"]
    attention_mask = corrupted["attention_mask"]

    cases = []
    correct_total = 0
    masked_total = 0
    loss_total = 0.0
    for start in range(0, len(examples), 16):
        batch_input = input_ids[start : start + 16].cuda()
        batch_attention = attention_mask[start : start + 16].cuda()
        batch_labels = labels[start : start + 16].cuda()
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model(
                input_ids=batch_input,
                attention_mask=batch_attention,
            ).logits.float()
        mask = batch_labels != -100
        predictions = logits.argmax(dim=-1)
        loss_total += float(
            F.cross_entropy(logits[mask], batch_labels[mask], reduction="sum").item()
        )
        correct_total += int(((predictions == batch_labels) & mask).sum().item())
        masked_total += int(mask.sum().item())

        for local_index in range(batch_input.shape[0]):
            global_index = start + local_index
            positions = torch.nonzero(mask[local_index]).flatten()
            rows = []
            true_nll = []
            for position in positions.tolist():
                distribution = logits[local_index, position].softmax(dim=-1)
                target_id = int(batch_labels[local_index, position].item())
                prediction_id = int(predictions[local_index, position].item())
                probabilities, ids = distribution.topk(5)
                true_probability = float(distribution[target_id].item())
                true_nll.append(-math.log(max(true_probability, 1e-30)))
                rows.append(
                    {
                        "position": position,
                        "target_id": target_id,
                        "target": token_text(tokenizer, target_id),
                        "prediction_id": prediction_id,
                        "prediction": token_text(tokenizer, prediction_id),
                        "correct": prediction_id == target_id,
                        "true_probability": true_probability,
                        "top5": [
                            {
                                "token_id": int(token_id),
                                "token": token_text(tokenizer, int(token_id)),
                                "probability": float(probability),
                            }
                            for probability, token_id in zip(
                                probabilities.tolist(), ids.tolist()
                            )
                        ],
                    }
                )
            original_ids = originals[global_index].tolist()
            cases.append(
                {
                    "case_index": global_index,
                    "case_id": hashlib.sha256(
                        originals[global_index].numpy().tobytes()
                    ).hexdigest()[:16],
                    "errors": sum(not row["correct"] for row in rows),
                    "masked_count": len(rows),
                    "mean_true_nll": sum(true_nll) / len(true_nll),
                    "input_ids": original_ids,
                    "tokens": [token_text(tokenizer, token_id) for token_id in original_ids],
                    "masked_tokens": rows,
                }
            )

    cases.sort(key=lambda row: (-row["errors"], -row["mean_true_nll"], row["case_index"]))
    selected = cases[: args.cases]
    summary = {
        "protocol": "official grouping plus deterministic standard 15% MLM corruption",
        "seed": args.seed,
        "candidate_windows": len(examples),
        "masked_tokens": masked_total,
        "masked_accuracy": correct_total / masked_total,
        "masked_ce": loss_total / masked_total,
        "model_dir": str(args.model_dir),
        "model_config_is_decoder": model.config.is_decoder,
        "gpu_uuid": visible_gpu_uuid(),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (args.output_dir / "hardest_cases.json").write_text(
        json.dumps(selected, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (args.output_dir / "index.html").write_text(
        render_html(summary, selected), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
