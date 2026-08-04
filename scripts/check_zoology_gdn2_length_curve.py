from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import subprocess
from pathlib import Path

import torch
import torch.nn.functional as F
from zoology.config import ModuleConfig
from zoology.data.utils import prepare_data
from zoology.model import LanguageModel
from zoology.utils import set_determinism

from experiments.zoology_mqar.gdn2_futureseed import (
    PINNED_FLA_SHA,
    ZoologyGDN2FutureSeedMixer,
)
from experiments.zoology_mqar.length_scaling import (
    build_config,
    dataset_hash,
    make_model,
    model_hash,
    parameter_hash,
)


EXPECTED_GPU_NAME = "NVIDIA A100-SXM4-80GB"
EXPECTED_GPU_UUID = "53e9f3b4-2966-65d3-6614-09c540921519"
EXPECTED_ZOOLOGY_SHA = "1ad20d193b6113cae1e8f3c655c300d7b4b3f4bb"
TRAIN_LENGTHS = (128, 256, 512)
P012_TRAIN_HASH = "647c64ece84984a23dfd817c4f277ea83840dbec57cc18bb6c9bf9eda7cc9a68"
P012_TEST_HASH = "4a8237ba8fe19aaff0d1d72de7b7f6505eaab59cd091442c2f463df34cce278f"


def git_head(path: Path) -> str:
    return subprocess.check_output(
        ["git", "-C", str(path), "rev-parse", "HEAD"], text=True
    ).strip()


def shared_state(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: tensor.detach()
        for name, tensor in model.state_dict().items()
        if not name.endswith("future_seed_logit")
    }


def direct_gdn2_model(sequence_length: int) -> LanguageModel:
    config = build_config(
        arm="causal_gdn2",
        sequence_length=sequence_length,
        num_kv_pairs=4,
        max_epochs=10,
        batch_size=32,
    )
    config.model.sequence_mixer = ModuleConfig(
        name="experiments.zoology_mqar.gdn2_mixer.ZoologyGDN2Mixer",
        kwargs={
            "num_heads": 4,
            "head_dim": 32,
            "expand_v": 1.0,
            "conv_size": 4,
        },
    )
    return LanguageModel(copy.deepcopy(config.model))


def alter_later_values(inputs: torch.Tensor) -> torch.Tensor:
    altered = inputs.clone()
    value_mask = (altered >= 160) & (altered < 256)
    altered[value_mask] = 160 + ((altered[value_mask] - 160 + 1) % 96)
    return altered


def finite_backward(
    model: torch.nn.Module,
    inputs: torch.Tensor,
    targets: torch.Tensor,
) -> tuple[float, float]:
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
    return (
        float(loss.item()),
        max(float(gradient.abs().max().item()) for gradient in gradients),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if torch.cuda.device_count() != 1:
        raise RuntimeError(f"Expected one visible GPU, got {torch.cuda.device_count()}")
    device = torch.cuda.get_device_properties(0)
    device_uuid = str(getattr(device, "uuid", "unavailable"))
    if device.name != EXPECTED_GPU_NAME or device_uuid != EXPECTED_GPU_UUID:
        raise RuntimeError(f"Unexpected GPU: {device.name} {device_uuid}")
    if os.environ.get("FLA_EXPECTED_SOURCE_SHA") != PINNED_FLA_SHA:
        raise RuntimeError("Pinned FLA SHA is missing")
    if os.environ.get("FLA_DISABLE_BACKEND_DISPATCH") != "1":
        raise RuntimeError("FLA backend dispatch must be disabled")
    if os.environ.get("FLA_CONV_BACKEND") != "triton":
        raise RuntimeError("Short convolution must use Triton")

    fla_root = Path(os.environ["FLA_SOURCE_ROOT"]).resolve()
    direct_url = json.loads(
        (fla_root / "flash_linear_attention-0.5.2.dist-info/direct_url.json").read_text()
    )
    wheel_name = Path(direct_url["url"]).name
    wheel_hash = direct_url["archive_info"]["hashes"]["sha256"]
    if PINNED_FLA_SHA[:7] not in wheel_name:
        raise RuntimeError(f"FLA wheel provenance lacks pinned SHA: {wheel_name}")
    if wheel_hash != os.environ["FLA_WHEEL_SHA256"]:
        raise RuntimeError(f"Unexpected FLA wheel hash: {wheel_hash}")
    source_hash = hashlib.sha256((fla_root / "fla/layers/gdn2.py").read_bytes()).hexdigest()
    if source_hash != os.environ["FLA_GDN2_SOURCE_SHA256"]:
        raise RuntimeError(f"Unexpected GDN2 source hash: {source_hash}")
    zoology_root = Path(os.environ["ZOOLOGY_ROOT"]).resolve()
    if git_head(zoology_root) != EXPECTED_ZOOLOGY_SHA:
        raise RuntimeError("Unexpected Zoology SHA")

    data_hashes: dict[str, dict[str, str]] = {}
    for sequence_length in (*TRAIN_LENGTHS, 1024):
        config = build_config(
            arm="causal_gdn2",
            sequence_length=sequence_length,
            num_kv_pairs=4,
            max_epochs=10,
            batch_size=32,
        )
        train_loader, test_loader = prepare_data(config.data)
        data_hashes[str(sequence_length)] = {
            "train": dataset_hash(train_loader),
            "test": dataset_hash(test_loader),
        }
    if data_hashes["1024"] != {
        "train": P012_TRAIN_HASH,
        "test": P012_TEST_HASH,
    }:
        raise RuntimeError(f"Frozen length-1024 data drifted: {data_hashes['1024']}")

    sequence_length = 512
    configs = {
        arm: build_config(
            arm=arm,
            sequence_length=sequence_length,
            num_kv_pairs=4,
            max_epochs=10,
            batch_size=32,
        )
        for arm in ("causal_gdn2", "future_seed_gdn2")
    }
    train_loader, test_loader = prepare_data(configs["causal_gdn2"].data)
    set_determinism(123)
    direct = direct_gdn2_model(sequence_length)
    set_determinism(123)
    causal = make_model(configs["causal_gdn2"], "causal_gdn2")
    set_determinism(123)
    future_seed = make_model(configs["future_seed_gdn2"], "future_seed_gdn2")
    if model_hash(causal) != model_hash(future_seed):
        raise RuntimeError("Matched model hashes differ")
    if parameter_hash(causal) != parameter_hash(future_seed):
        raise RuntimeError("Matched parameter hashes differ")
    parameter_counts = {
        "causal_gdn2": sum(parameter.numel() for parameter in causal.parameters()),
        "future_seed_gdn2": sum(
            parameter.numel() for parameter in future_seed.parameters()
        ),
    }
    if len(set(parameter_counts.values())) != 1:
        raise RuntimeError(f"Matched parameter counts differ: {parameter_counts}")

    direct_state = direct.state_dict()
    causal_state = shared_state(causal)
    if set(direct_state) != set(causal_state):
        raise RuntimeError("Scale-zero wrapper state keys differ from direct GDN2")
    shared_init_max_diff = max(
        float((direct_state[name] - causal_state[name]).abs().max().item())
        for name in direct_state
    )
    if shared_init_max_diff != 0.0:
        raise RuntimeError(f"Scale-zero init mismatch: {shared_init_max_diff}")

    inputs, targets, _ = next(iter(test_loader))
    inputs = inputs.cuda()
    targets = targets.cuda()
    altered = alter_later_values(inputs).cuda()
    quarter = sequence_length // 4
    positions = torch.arange(sequence_length, device="cuda")
    future_mask = (targets != -100) & (positions[None, :] < quarter)
    direct = direct.cuda().eval()
    causal = causal.cuda().eval()
    future_seed = future_seed.cuda().eval()
    torch.cuda.reset_peak_memory_stats()
    with torch.no_grad():
        direct_logits = direct(inputs)
        causal_logits = causal(inputs)
        causal_altered = causal(altered)
        future_logits = future_seed(inputs)
        future_altered = future_seed(altered)
    scale0_output_max_diff = float((direct_logits - causal_logits).abs().max().item())
    if scale0_output_max_diff != 0.0:
        raise RuntimeError(f"Scale-zero output mismatch: {scale0_output_max_diff}")
    future_dependencies = {
        "causal_gdn2_max": float(
            (causal_logits[future_mask] - causal_altered[future_mask]).abs().max().item()
        ),
        "future_seed_gdn2_mean": float(
            (future_logits[future_mask] - future_altered[future_mask]).abs().mean().item()
        ),
    }
    if future_dependencies["causal_gdn2_max"] != 0.0:
        raise RuntimeError(f"Causal path leaks future values: {future_dependencies}")
    if future_dependencies["future_seed_gdn2_mean"] <= 0.0:
        raise RuntimeError("FutureSeed path has no future dependency")

    losses_and_gradients = {}
    for arm, model in (("causal_gdn2", causal), ("future_seed_gdn2", future_seed)):
        loss, gradient_max = finite_backward(model, inputs, targets)
        losses_and_gradients[arm] = {
            "loss": loss,
            "gradient_max": gradient_max,
        }
    gate_gradients = [
        parameter.grad
        for name, parameter in future_seed.named_parameters()
        if name.endswith("future_seed_logit") and parameter.grad is not None
    ]
    future_seed_gate_gradient_max = max(
        float(gradient.abs().max().item()) for gradient in gate_gradients
    )
    if future_seed_gate_gradient_max <= 0.0:
        raise RuntimeError("FutureSeed gate received no gradient")

    mixers = [
        layer.sequence_mixer
        for model in (causal, future_seed)
        for layer in model.backbone.layers
        if isinstance(layer.sequence_mixer, ZoologyGDN2FutureSeedMixer)
    ]
    modes = [mixer.layer.mode for mixer in mixers]
    conv_backends = [getattr(mixer.layer.q_conv1d, "backend", None) for mixer in mixers]
    if any(mode != "chunk" for mode in modes):
        raise RuntimeError(f"Unexpected GDN2 mode: {modes}")
    if any(backend != "triton" for backend in conv_backends):
        raise RuntimeError(f"Unexpected convolution backend: {conv_backends}")

    result = {
        "device": device.name,
        "device_uuid": device_uuid,
        "cuda_device_count": torch.cuda.device_count(),
        "zoology_sha": EXPECTED_ZOOLOGY_SHA,
        "fla_sha": PINNED_FLA_SHA,
        "fla_wheel_sha256": wheel_hash,
        "fla_gdn2_source_sha256": source_hash,
        "data_hashes": data_hashes,
        "full_size_preflight_length": sequence_length,
        "parameter_counts": parameter_counts,
        "shared_init_max_diff": shared_init_max_diff,
        "scale0_output_max_diff": scale0_output_max_diff,
        "future_dependencies": future_dependencies,
        "future_seed_gate_gradient_max": future_seed_gate_gradient_max,
        "losses_and_gradients": losses_and_gradients,
        "gdn2_modes": modes,
        "conv_backends": conv_backends,
        "peak_cuda_mem_bytes": torch.cuda.max_memory_allocated(),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
