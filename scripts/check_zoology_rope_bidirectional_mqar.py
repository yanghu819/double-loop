from __future__ import annotations

import argparse
import copy
import hashlib
import inspect
import json
import os
import subprocess
from pathlib import Path

import torch
import torch.nn.functional as F

from zoology.data.utils import prepare_data
from zoology.model import LanguageModel
from zoology.utils import set_determinism

from experiments.zoology_mqar.bidirectional_attention import (
    ParamMatchedBidirectionalAttention,
)
from experiments.zoology_mqar.futureseed_directionality import (
    dataset_hash,
    model_hash,
)
from experiments.zoology_mqar.length_scaling import (
    P007_LENGTH64_TEST_HASH,
    P007_LENGTH64_TRAIN_HASH,
    parameter_hash,
)
from experiments.zoology_mqar.rope_bidirectional_attention import (
    ParamMatchedRoPEBidirectionalAttention,
)
from experiments.zoology_mqar.rope_bidirectional_carrier import build_config


EXPECTED_GPU_NAME = "NVIDIA A100-SXM4-80GB"
EXPECTED_GPU_UUID = "53e9f3b4-2966-65d3-6614-09c540921519"
EXPECTED_ZOOLOGY_SHA = "1ad20d193b6113cae1e8f3c655c300d7b4b3f4bb"
EXPECTED_PARAMETERS = 539_136
EXPECTED_INIT_HASH = (
    "5e679ccd198047f77d5f15695a49e8968a1a0c2fa4cc8d4c64bcd6a6c05b3a21"
)
EXPECTED_PARAMETER_HASH = (
    "1da2029805ef8d5992bf169b762cf6035b5b20da47445bb269d87abbf7329e2b"
)


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
        raise RuntimeError("Model backward produced non-finite gradients")
    return {
        "loss": float(loss.item()),
        "gradient_max": max(
            float(gradient.abs().max().item()) for gradient in gradients
        ),
    }


def build_model(*, rope_scale: float) -> LanguageModel:
    config = build_config(rope_scale=rope_scale)
    set_determinism(config.seed)
    return LanguageModel(copy.deepcopy(config.model))


def replace_with_plain_attention(model: LanguageModel) -> None:
    for block in model.backbone.layers:
        mixer = block.sequence_mixer
        plain = ParamMatchedBidirectionalAttention(
            d_model=128,
            layer_idx=mixer.layer_idx if hasattr(mixer, "layer_idx") else 0,
            num_heads=4,
            head_dim=57,
        )
        plain.load_state_dict(mixer.state_dict(), strict=True)
        block.sequence_mixer = plain


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reference-score", type=Path, required=True)
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
        "model": sha256(zoology_root / "zoology/model.py"),
        "train": sha256(zoology_root / "zoology/train.py"),
    }
    expected_hashes = {
        "model": os.environ["ZOOLOGY_MODEL_SHA256"],
        "train": os.environ["ZOOLOGY_TRAIN_SHA256"],
    }
    if source_hashes != expected_hashes:
        raise RuntimeError(
            f"Zoology source hash mismatch: {source_hashes} != {expected_hashes}"
        )

    reference = json.loads(args.reference_score.read_text())
    if reference["parameters"] != EXPECTED_PARAMETERS:
        raise RuntimeError("Frozen plain-SDPA parameter count changed")
    if reference["init_hash"] != EXPECTED_INIT_HASH:
        raise RuntimeError("Frozen plain-SDPA initialization hash changed")
    if reference["init_parameter_hash"] != EXPECTED_PARAMETER_HASH:
        raise RuntimeError("Frozen plain-SDPA parameter hash changed")

    config = build_config(rope_scale=1.0)
    train_dataloader, test_dataloader = prepare_data(config.data)
    data_hashes = {
        "train": dataset_hash(train_dataloader),
        "test": dataset_hash(test_dataloader),
    }
    if data_hashes != {
        "train": P007_LENGTH64_TRAIN_HASH,
        "test": P007_LENGTH64_TEST_HASH,
    }:
        raise RuntimeError(f"Directional data drifted: {data_hashes}")
    if data_hashes != reference["data_hashes"]:
        raise RuntimeError("Data no longer match frozen plain SDPA")

    rope_zero = build_model(rope_scale=0.0)
    rope_one = build_model(rope_scale=1.0)
    plain = build_model(rope_scale=0.0)
    replace_with_plain_attention(plain)

    models = {
        "plain_sdpa": plain,
        "rope_scale0": rope_zero,
        "rope_scale1": rope_one,
    }
    parameter_counts = {
        name: sum(parameter.numel() for parameter in model.parameters())
        for name, model in models.items()
    }
    init_hashes = {name: model_hash(model) for name, model in models.items()}
    parameter_hashes = {
        name: parameter_hash(model) for name, model in models.items()
    }
    if set(parameter_counts.values()) != {EXPECTED_PARAMETERS}:
        raise RuntimeError(f"Parameter budget changed: {parameter_counts}")
    if set(init_hashes.values()) != {EXPECTED_INIT_HASH}:
        raise RuntimeError(f"Initialized tensors changed: {init_hashes}")
    if set(parameter_hashes.values()) != {EXPECTED_PARAMETER_HASH}:
        raise RuntimeError(f"Initialized parameters changed: {parameter_hashes}")

    state_keys = {name: set(model.state_dict()) for name, model in models.items()}
    if len({tuple(sorted(keys)) for keys in state_keys.values()}) != 1:
        raise RuntimeError(f"State keys differ: {state_keys}")

    inputs, targets, _slices = next(iter(test_dataloader))
    inputs = inputs.cuda()
    targets = targets.cuda()
    altered = alter_later_values(inputs).cuda()
    positions = torch.arange(inputs.shape[1], device="cuda")
    future_mask = (targets != -100) & (positions[None, :] < 16)
    for model in models.values():
        model.cuda().eval()
    with torch.no_grad():
        outputs = {name: model(inputs) for name, model in models.items()}
        altered_outputs = {
            name: model(altered) for name, model in models.items()
        }
    scale0_max_diff = float(
        (outputs["plain_sdpa"] - outputs["rope_scale0"])
        .abs()
        .max()
        .item()
    )
    if scale0_max_diff != 0.0:
        raise RuntimeError(f"rope_scale=0 is not exact plain SDPA: {scale0_max_diff}")
    future_dependencies = {
        name: float(
            (outputs[name][future_mask] - altered_outputs[name][future_mask])
            .abs()
            .mean()
            .item()
        )
        for name in models
    }
    if future_dependencies["rope_scale1"] <= 1e-7:
        raise RuntimeError("RoPE bidirectional attention has no future dependency")

    candidate_mixers = [
        block.sequence_mixer for block in rope_one.backbone.layers
    ]
    if not all(
        type(mixer) is ParamMatchedRoPEBidirectionalAttention
        for mixer in candidate_mixers
    ):
        raise RuntimeError("Candidate mixer class drifted")
    if not all(
        mixer.rotary_dim == 56 and mixer.rope_scale == 1.0
        for mixer in candidate_mixers
    ):
        raise RuntimeError("Candidate RoPE geometry drifted")
    candidate_source = inspect.getsource(
        ParamMatchedRoPEBidirectionalAttention.forward
    )
    if "is_causal=False" not in candidate_source:
        raise RuntimeError("Candidate is not explicitly noncausal")

    backward = {
        "rope_scale0": finite_backward(rope_zero, inputs, targets),
        "rope_scale1": finite_backward(rope_one, inputs, targets),
    }
    result = {
        "cuda_device_count": torch.cuda.device_count(),
        "device": device.name,
        "device_uuid": device_uuid,
        "zoology_sha": zoology_sha,
        "zoology_source_hashes": source_hashes,
        "data_hashes": data_hashes,
        "plain_attention_class": (
            f"{ParamMatchedBidirectionalAttention.__module__}."
            f"{ParamMatchedBidirectionalAttention.__name__}"
        ),
        "candidate_attention_class": (
            f"{ParamMatchedRoPEBidirectionalAttention.__module__}."
            f"{ParamMatchedRoPEBidirectionalAttention.__name__}"
        ),
        "parameter_counts": parameter_counts,
        "init_hashes": init_hashes,
        "parameter_hashes": parameter_hashes,
        "state_keys_identical": True,
        "rope_scale0_output_max_diff": scale0_max_diff,
        "future_dependencies": future_dependencies,
        "rotary_dim": 56,
        "rope_theta": 10_000.0,
        "rope_scale": 1.0,
        "parameter_free_intervention": True,
        "backward": backward,
        "peak_cuda_mem_bytes": torch.cuda.max_memory_allocated(),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
