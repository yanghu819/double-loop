from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import os
import subprocess
from pathlib import Path

import torch
import torch.nn.functional as F

from zoology.data.utils import prepare_data
from zoology.mixers.attention import MHA
from zoology.model import LanguageModel
from zoology.utils import set_determinism

from experiments.zoology_mqar.official_bidirectional_carrier import (
    P007_LENGTH64_TEST_HASH,
    P007_LENGTH64_TRAIN_HASH,
    build_config,
    dataset_hash,
)
from experiments.zoology_mqar.official_bidirectional_mha import (
    NonCausalSelfAttention,
    OfficialBidirectionalMHA,
)


EXPECTED_GPU_NAME = "NVIDIA A100-SXM4-80GB"
EXPECTED_GPU_UUID = "53e9f3b4-2966-65d3-6614-09c540921519"
EXPECTED_ZOOLOGY_SHA = "1ad20d193b6113cae1e8f3c655c300d7b4b3f4bb"


def git_head(path: Path) -> str:
    return subprocess.check_output(
        ["git", "-C", str(path), "rev-parse", "HEAD"],
        text=True,
    ).strip()


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def alter_later_values(inputs: torch.Tensor) -> torch.Tensor:
    altered = inputs.clone()
    value_mask = (altered >= 160) & (altered < 256)
    altered[value_mask] = 160 + ((altered[value_mask] - 160 + 1) % 96)
    return altered


def finite_backward(
    model: torch.nn.Module,
    inputs: torch.Tensor,
    targets: torch.Tensor,
) -> dict[str, float]:
    model.train()
    model.zero_grad(set_to_none=True)
    logits = model(inputs)
    mask = targets != -100
    loss = F.cross_entropy(logits[mask], targets[mask])
    loss.backward()
    gradients = [
        parameter.grad
        for parameter in model.parameters()
        if parameter.grad is not None
    ]
    if not gradients or not all(
        torch.isfinite(gradient).all() for gradient in gradients
    ):
        raise RuntimeError("Non-finite or missing gradients")
    return {
        "loss": float(loss.item()),
        "gradient_max": max(
            float(gradient.abs().max().item()) for gradient in gradients
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if torch.cuda.device_count() != 1:
        raise RuntimeError(f"Expected one visible GPU, got {torch.cuda.device_count()}")
    device = torch.cuda.get_device_properties(0)
    device_uuid = str(getattr(device, "uuid", "unavailable"))
    if device.name != EXPECTED_GPU_NAME or device_uuid != EXPECTED_GPU_UUID:
        raise RuntimeError(f"Unexpected GPU: {device.name}, {device_uuid}")

    zoology_root = Path(os.environ["ZOOLOGY_ROOT"]).resolve()
    zoology_sha = git_head(zoology_root)
    if zoology_sha != EXPECTED_ZOOLOGY_SHA:
        raise RuntimeError(f"Unexpected Zoology SHA: {zoology_sha}")
    source_hashes = {
        "attention": sha256(zoology_root / "zoology/mixers/attention.py"),
        "model": sha256(zoology_root / "zoology/model.py"),
        "train": sha256(zoology_root / "zoology/train.py"),
    }
    expected_hashes = {
        "attention": os.environ["ZOOLOGY_ATTENTION_SHA256"],
        "model": os.environ["ZOOLOGY_MODEL_SHA256"],
        "train": os.environ["ZOOLOGY_TRAIN_SHA256"],
    }
    if source_hashes != expected_hashes:
        raise RuntimeError(
            f"Zoology source hash mismatch: {source_hashes} != {expected_hashes}"
        )

    if OfficialBidirectionalMHA.__mro__[1] is not MHA:
        raise RuntimeError("Bidirectional carrier does not directly inherit official MHA")
    custom_source = inspect.getsource(NonCausalSelfAttention.forward)
    if "causal_mask" in custom_source or "torch.triu" in custom_source:
        raise RuntimeError("Bidirectional attention still contains a causal mask")

    causal_config = build_config(bidirectional=False)
    bidirectional_config = build_config(bidirectional=True)
    set_determinism(123)
    causal = LanguageModel(causal_config.model)
    set_determinism(123)
    bidirectional = LanguageModel(bidirectional_config.model)
    causal_state = causal.state_dict()
    bidirectional_state = bidirectional.state_dict()
    if set(causal_state) != set(bidirectional_state):
        raise RuntimeError("Official and bidirectional state keys differ")
    init_max_diff = max(
        float((causal_state[name] - bidirectional_state[name]).abs().max().item())
        for name in causal_state
    )
    if init_max_diff != 0.0:
        raise RuntimeError(f"Official initialization changed: {init_max_diff}")
    parameter_counts = {
        "official_causal_mha": sum(p.numel() for p in causal.parameters()),
        "official_bidirectional_mha": sum(
            p.numel() for p in bidirectional.parameters()
        ),
    }
    if len(set(parameter_counts.values())) != 1:
        raise RuntimeError(f"Parameter counts differ: {parameter_counts}")

    train_dataloader, test_dataloader = prepare_data(causal_config.data)
    data_hashes = {
        "train": dataset_hash(train_dataloader),
        "test": dataset_hash(test_dataloader),
    }
    if data_hashes != {
        "train": P007_LENGTH64_TRAIN_HASH,
        "test": P007_LENGTH64_TEST_HASH,
    }:
        raise RuntimeError(f"Directional data drifted: {data_hashes}")

    inputs, targets, _slices = next(iter(test_dataloader))
    inputs = inputs.cuda()
    targets = targets.cuda()
    altered = alter_later_values(inputs).cuda()
    positions = torch.arange(inputs.shape[1], device="cuda")
    future_query_mask = (targets != -100) & (positions[None, :] < 16)
    causal = causal.cuda().eval()
    bidirectional = bidirectional.cuda().eval()
    with torch.no_grad():
        causal_logits = causal(inputs)
        causal_altered = causal(altered)
        bidirectional_logits = bidirectional(inputs)
        bidirectional_altered = bidirectional(altered)
    dependencies = {
        "official_causal_future_max": float(
            (causal_logits[future_query_mask] - causal_altered[future_query_mask])
            .abs()
            .max()
            .item()
        ),
        "official_bidirectional_future_mean": float(
            (
                bidirectional_logits[future_query_mask]
                - bidirectional_altered[future_query_mask]
            )
            .abs()
            .mean()
            .item()
        ),
    }
    if dependencies["official_causal_future_max"] > 1e-6:
        raise RuntimeError("Official causal MHA leaked future information")
    if dependencies["official_bidirectional_future_mean"] <= 1e-7:
        raise RuntimeError("Bidirectional MHA has no future dependency")

    backward = {
        "official_causal_mha": finite_backward(causal, inputs, targets),
        "official_bidirectional_mha": finite_backward(
            bidirectional,
            inputs,
            targets,
        ),
    }
    result = {
        "cuda_device_count": torch.cuda.device_count(),
        "device": device.name,
        "device_uuid": device_uuid,
        "zoology_sha": zoology_sha,
        "zoology_source_hashes": source_hashes,
        "official_mha_source": str(Path(inspect.getfile(MHA)).resolve()),
        "official_mha_class": f"{MHA.__module__}.{MHA.__name__}",
        "bidirectional_mha_class": (
            f"{OfficialBidirectionalMHA.__module__}."
            f"{OfficialBidirectionalMHA.__name__}"
        ),
        "only_semantic_change": "remove upstream triangular causal mask",
        "state_keys_identical": True,
        "initialization_max_diff": init_max_diff,
        "parameter_counts": parameter_counts,
        "data_hashes": data_hashes,
        "future_dependencies": dependencies,
        "backward": backward,
        "peak_cuda_mem_bytes": torch.cuda.max_memory_allocated(),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
