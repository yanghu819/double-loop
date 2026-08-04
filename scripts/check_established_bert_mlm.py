from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import subprocess
from pathlib import Path

import torch
import transformers
from transformers import AutoTokenizer, BertConfig, BertForMaskedLM, set_seed


EXPECTED_GPU_UUID = "GPU-53e9f3b4-2966-65d3-6614-09c540921519"
EXPECTED_TRANSFORMERS_VERSION = "4.46.3"
EXPECTED_BERT_SOURCE_SHA256 = (
    "3493bff5da90fdcce98dad5c84aafe4d3ce1c550dcd93bc99289309953559eca"
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def parameter_hash(model: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, parameter in sorted(model.named_parameters()):
        digest.update(name.encode("utf-8"))
        digest.update(parameter.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def build_model(config_path: Path) -> BertForMaskedLM:
    config = BertConfig.from_pretrained(config_path, local_files_only=True)
    config._attn_implementation = "sdpa"
    config.use_cache = False
    set_seed(123)
    return BertForMaskedLM(config)


def future_dependency(
    model: BertForMaskedLM,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    probe_position: int,
    replacement_id: int,
) -> float:
    altered = input_ids.clone()
    suffix_mask = torch.arange(input_ids.shape[1], device=input_ids.device)
    suffix_mask = suffix_mask > probe_position
    altered[:, suffix_mask] = torch.where(
        attention_mask[:, suffix_mask].bool(),
        torch.full_like(altered[:, suffix_mask], replacement_id),
        altered[:, suffix_mask],
    )
    model.eval()
    with torch.no_grad():
        original = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        ).logits[:, probe_position]
        changed = model(
            input_ids=altered,
            attention_mask=attention_mask,
        ).logits[:, probe_position]
    return float((original - changed).abs().mean().item())


def finite_backward(
    model: BertForMaskedLM,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    labels: torch.Tensor,
) -> tuple[float, float]:
    model.train()
    model.zero_grad(set_to_none=True)
    output = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        labels=labels,
    )
    output.loss.backward()
    gradients = [
        parameter.grad
        for parameter in model.parameters()
        if parameter.grad is not None
    ]
    if not gradients or not all(torch.isfinite(grad).all() for grad in gradients):
        raise RuntimeError("Missing or non-finite gradient")
    max_gradient = max(float(grad.abs().max().item()) for grad in gradients)
    return float(output.loss.item()), max_gradient


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bidirectional-config", type=Path, required=True)
    parser.add_argument("--causal-config", type=Path, required=True)
    parser.add_argument("--official-config", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--data-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if torch.cuda.device_count() != 1:
        raise RuntimeError(f"Expected one visible GPU, got {torch.cuda.device_count()}")
    properties = torch.cuda.get_device_properties(0)
    gpu_uuid = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=uuid",
            "--format=csv,noheader",
        ],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if properties.name != "NVIDIA A100-SXM4-80GB" or gpu_uuid != EXPECTED_GPU_UUID:
        raise RuntimeError(f"Unexpected GPU: {properties.name}, {gpu_uuid}")
    if transformers.__version__ != EXPECTED_TRANSFORMERS_VERSION:
        raise RuntimeError(f"Unexpected Transformers version: {transformers.__version__}")
    bert_source = Path(inspect.getfile(BertForMaskedLM))
    bert_source_sha256 = sha256(bert_source)
    if bert_source_sha256 != EXPECTED_BERT_SOURCE_SHA256:
        raise RuntimeError(
            f"Unexpected BertForMaskedLM source SHA256: {bert_source_sha256}"
        )

    bidirectional_config = BertConfig.from_pretrained(
        args.bidirectional_config, local_files_only=True
    )
    causal_config = BertConfig.from_pretrained(
        args.causal_config, local_files_only=True
    )
    official_config = json.loads(args.official_config.read_text(encoding="utf-8"))
    if bidirectional_config.is_decoder or not causal_config.is_decoder:
        raise RuntimeError("Bidirectional/causal is_decoder semantics are wrong")
    bidirectional_dict = bidirectional_config.to_dict()
    causal_dict = causal_config.to_dict()
    bidirectional_dict["is_decoder"] = True
    if bidirectional_dict != causal_dict:
        differing = sorted(
            key
            for key in set(bidirectional_dict) | set(causal_dict)
            if bidirectional_dict.get(key) != causal_dict.get(key)
        )
        raise RuntimeError(f"Configs differ beyond is_decoder: {differing}")
    registered_architecture = json.loads(
        args.bidirectional_config.read_text(encoding="utf-8")
    )
    for explicit_runtime_key in ("add_cross_attention", "is_decoder", "use_cache"):
        registered_architecture.pop(explicit_runtime_key)
    if registered_architecture != official_config:
        differing = sorted(
            key
            for key in set(registered_architecture) | set(official_config)
            if registered_architecture.get(key) != official_config.get(key)
        )
        raise RuntimeError(
            f"Registered architecture differs from Google BERT miniature: {differing}"
        )

    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer,
        local_files_only=True,
        use_fast=True,
    )
    if tokenizer.mask_token_id is None or len(tokenizer) != 30_522:
        raise RuntimeError("Unexpected BERT tokenizer vocabulary")

    bidirectional = build_model(args.bidirectional_config)
    causal = build_model(args.causal_config)
    if set(bidirectional.state_dict()) != set(causal.state_dict()):
        raise RuntimeError("Model state keys differ")
    init_max_diff = max(
        float((bidirectional.state_dict()[key] - causal.state_dict()[key]).abs().max())
        for key in bidirectional.state_dict()
    )
    if init_max_diff != 0.0:
        raise RuntimeError(f"Matched initialization differs: {init_max_diff}")
    parameter_count = sum(parameter.numel() for parameter in causal.parameters())
    if parameter_count != sum(
        parameter.numel() for parameter in bidirectional.parameters()
    ):
        raise RuntimeError("Parameter counts differ")

    token_ids = tokenizer.convert_tokens_to_ids(
        ["the", "capital", "of", "france", "is", "paris", ".", "today"]
    )
    sequence = [tokenizer.cls_token_id, *token_ids[:5], tokenizer.mask_token_id]
    probe_position = len(sequence) - 1
    sequence.extend(token_ids[5:])
    sequence.append(tokenizer.sep_token_id)
    input_ids = torch.full((2, 128), tokenizer.pad_token_id, dtype=torch.long)
    input_ids[:, : len(sequence)] = torch.tensor(sequence, dtype=torch.long)
    attention_mask = input_ids.ne(tokenizer.pad_token_id).long()
    labels = torch.full_like(input_ids, -100)
    labels[:, probe_position] = token_ids[5]
    input_ids = input_ids.cuda()
    attention_mask = attention_mask.cuda()
    labels = labels.cuda()

    torch.cuda.reset_peak_memory_stats()
    bidirectional = bidirectional.cuda()
    causal = causal.cuda()
    causal_dependency = future_dependency(
        causal,
        input_ids,
        attention_mask,
        probe_position,
        token_ids[0],
    )
    bidirectional_dependency = future_dependency(
        bidirectional,
        input_ids,
        attention_mask,
        probe_position,
        token_ids[0],
    )
    if causal_dependency != 0.0:
        raise RuntimeError(f"Causal model leaks future input: {causal_dependency}")
    if bidirectional_dependency <= 1e-7:
        raise RuntimeError("Bidirectional model has no measurable future dependency")
    bidirectional_loss, bidirectional_gradient = finite_backward(
        bidirectional, input_ids, attention_mask, labels
    )
    causal_loss, causal_gradient = finite_backward(
        causal, input_ids, attention_mask, labels
    )

    data_manifest = json.loads(args.data_manifest.read_text(encoding="utf-8"))
    result = {
        "gpu": {
            "index": 0,
            "name": properties.name,
            "uuid": gpu_uuid,
        },
        "transformers": {
            "version": transformers.__version__,
            "bert_source": str(bert_source),
            "bert_source_sha256": bert_source_sha256,
        },
        "assets": {
            "tokenizer_vocab_sha256": sha256(args.tokenizer / "vocab.txt"),
            "official_config_sha256": sha256(args.official_config),
            "bidirectional_config_sha256": sha256(args.bidirectional_config),
            "causal_config_sha256": sha256(args.causal_config),
            "data_manifest": data_manifest,
        },
        "model": {
            "class": f"{BertForMaskedLM.__module__}.{BertForMaskedLM.__name__}",
            "parameters": parameter_count,
            "initialization_hash": parameter_hash(causal),
            "shared_init_max_diff": init_max_diff,
            "bidirectional_is_decoder": bidirectional_config.is_decoder,
            "causal_is_decoder": causal_config.is_decoder,
            "position_embeddings": bidirectional_config.max_position_embeddings,
            "mlm_head": "BertOnlyMLMHead",
        },
        "diagnostics": {
            "causal_future_dependency": causal_dependency,
            "bidirectional_future_dependency": bidirectional_dependency,
            "causal_loss": causal_loss,
            "bidirectional_loss": bidirectional_loss,
            "causal_max_gradient": causal_gradient,
            "bidirectional_max_gradient": bidirectional_gradient,
            "peak_cuda_memory_bytes": torch.cuda.max_memory_allocated(),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
