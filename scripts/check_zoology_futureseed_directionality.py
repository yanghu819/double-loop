from __future__ import annotations

import argparse
import copy
import inspect
import json
import os
from pathlib import Path

import torch
import torch.nn.functional as F

from fla.layers.gdn2 import GatedDeltaNet2
from zoology.data.utils import prepare_data
from zoology.model import LanguageModel
from zoology.utils import set_determinism

from experiments.zoology_mqar.futureseed_directionality import (
    build_config,
    dataset_hash,
    model_hash,
)
from experiments.zoology_mqar.gdn2_basic import config as baseline_config
from experiments.zoology_mqar.gdn2_futureseed import (
    PINNED_FLA_SHA,
    FutureSeedLanguageModel,
    ZoologyGDN2FutureSeedMixer,
    futureseed_diagnostics,
)


def _shared_state(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: tensor.detach()
        for name, tensor in model.state_dict().items()
        if not name.endswith("future_seed_logit")
    }


def _future_batch(test_dataloader) -> tuple[torch.Tensor, torch.Tensor]:
    segment = test_dataloader.dataset.segments[1]
    if segment.slices["direction"] != "future":
        raise RuntimeError("Expected future segment at index 1")
    return segment.inputs[:32], segment.labels[:32]


def _alter_future_values(inputs: torch.Tensor) -> torch.Tensor:
    altered = inputs.clone()
    value_mask = (altered >= 160) & (altered < 256)
    altered[value_mask] = 160 + ((altered[value_mask] - 160 + 1) % 96)
    return altered


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if torch.cuda.device_count() != 1:
        raise RuntimeError(f"Expected one visible GPU, got {torch.cuda.device_count()}")
    device = torch.cuda.get_device_properties(0)
    if device.name != "NVIDIA A100-SXM4-80GB":
        raise RuntimeError(f"Unexpected GPU: {device.name}")
    if os.environ.get("FLA_EXPECTED_SOURCE_SHA") != PINNED_FLA_SHA:
        raise RuntimeError("Pinned FLA SHA is missing")
    if os.environ.get("FLA_DISABLE_BACKEND_DISPATCH") != "1":
        raise RuntimeError("FLA backend dispatch must be disabled")
    if os.environ.get("FLA_CONV_BACKEND") != "triton":
        raise RuntimeError("Short convolution must use Triton")

    set_determinism(123)
    reference = LanguageModel(copy.deepcopy(baseline_config.model))
    set_determinism(123)
    no_fs_config = build_config(future_seed_scale=0.0, max_epochs=10)
    no_fs = FutureSeedLanguageModel(copy.deepcopy(no_fs_config.model))
    set_determinism(123)
    fs_config = build_config(future_seed_scale=1.0, max_epochs=10)
    fs = FutureSeedLanguageModel(copy.deepcopy(fs_config.model))

    reference_state = reference.state_dict()
    shared_state = _shared_state(no_fs)
    missing = sorted(set(reference_state) - set(shared_state))
    extra = sorted(set(shared_state) - set(reference_state))
    if missing or extra:
        raise RuntimeError(f"Shared state mismatch: missing={missing}, extra={extra}")
    max_init_diff = max(
        float((reference_state[name] - shared_state[name]).abs().max().item())
        for name in reference_state
    )
    if max_init_diff != 0.0:
        raise RuntimeError(f"Shared initialization mismatch: {max_init_diff}")
    if model_hash(no_fs) != model_hash(fs):
        raise RuntimeError("Scale 0 and 1 initialization hashes differ")

    reference = reference.cuda()
    no_fs = no_fs.cuda()
    fs = fs.cuda()

    inputs = torch.randint(0, 256, (32, 64), device="cuda")
    reference.train()
    no_fs.train()
    torch.cuda.manual_seed_all(777)
    reference_logits = reference(inputs)
    torch.cuda.manual_seed_all(777)
    no_fs_logits = no_fs(inputs)
    scale0_max_diff = float((reference_logits - no_fs_logits).abs().max().item())
    if scale0_max_diff != 0.0:
        raise RuntimeError(f"Scale-0 output is not exact: {scale0_max_diff}")

    train_dataloader, test_dataloader = prepare_data(fs_config.data)
    future_inputs, future_targets = _future_batch(test_dataloader)
    future_inputs = future_inputs.cuda()
    future_targets = future_targets.cuda()
    altered_inputs = _alter_future_values(future_inputs).cuda()
    query_mask = future_targets != -100

    no_fs.eval()
    fs.eval()
    no_fs_original = no_fs(future_inputs)
    no_fs_altered = no_fs(altered_inputs)
    fs_original = fs(future_inputs)
    fs_altered = fs(altered_inputs)
    no_fs_future_dependency = float(
        (no_fs_original[query_mask] - no_fs_altered[query_mask]).abs().max().item()
    )
    fs_future_dependency = float(
        (fs_original[query_mask] - fs_altered[query_mask]).abs().mean().item()
    )
    if no_fs_future_dependency > 1e-6:
        raise RuntimeError(f"Scale-0 leaked future values: {no_fs_future_dependency}")
    if fs_future_dependency <= 1e-7:
        raise RuntimeError("FutureSeed path does not affect future-query logits")

    fs.train()
    fs.zero_grad(set_to_none=True)
    logits = fs(future_inputs)
    loss = F.cross_entropy(logits[query_mask], future_targets[query_mask])
    loss.backward()
    gradients = [parameter.grad for parameter in fs.parameters() if parameter.grad is not None]
    if not gradients or not all(torch.isfinite(gradient).all() for gradient in gradients):
        raise RuntimeError("FutureSeed backward produced non-finite gradients")
    gate_grad = fs.backbone.layers[1].sequence_mixer.future_seed_logit.grad
    if gate_grad is None or float(gate_grad.abs().max().item()) == 0.0:
        raise RuntimeError("FutureSeed gate has no gradient")

    train_hash = dataset_hash(train_dataloader)
    test_hash = dataset_hash(test_dataloader)
    source = str(Path(inspect.getfile(GatedDeltaNet2)).resolve())
    mixers = [layer.sequence_mixer for layer in fs.backbone.layers]
    conv_backends = [
        getattr(mixer.layer.q_conv1d, "backend", None)
        for mixer in mixers
        if isinstance(mixer, ZoologyGDN2FutureSeedMixer)
    ]
    result = {
        "cuda_device_count": torch.cuda.device_count(),
        "device": device.name,
        "device_uuid": str(getattr(device, "uuid", "unavailable")),
        "fla_sha": PINNED_FLA_SHA,
        "gdn2_source": source,
        "gdn2_class": f"{GatedDeltaNet2.__module__}.{GatedDeltaNet2.__name__}",
        "mode": [mixer.layer.mode for mixer in mixers],
        "conv_backends": conv_backends,
        "parameters": sum(parameter.numel() for parameter in fs.parameters()),
        "shared_init_max_diff": max_init_diff,
        "scale0_output_max_diff": scale0_max_diff,
        "no_fs_future_dependency_max": no_fs_future_dependency,
        "fs_future_dependency_mean": fs_future_dependency,
        "future_seed_gate_grad_max": float(gate_grad.abs().max().item()),
        "loss": float(loss.item()),
        "train_data_hash": train_hash,
        "test_data_hash": test_hash,
        "future_seed": futureseed_diagnostics(fs),
        "peak_cuda_mem_bytes": torch.cuda.max_memory_allocated(),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
