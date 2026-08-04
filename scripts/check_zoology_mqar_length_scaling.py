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

from fla.layers.gdn2 import GatedDeltaNet2
from zoology.config import ModuleConfig
from zoology.data.utils import prepare_data
from zoology.model import LanguageModel
from zoology.utils import set_determinism

from experiments.zoology_mqar.bidirectional_attention import (
    ParamMatchedBidirectionalAttention,
)
from experiments.zoology_mqar.gdn2_futureseed import (
    PINNED_FLA_SHA,
    FutureSeedLanguageModel,
    ZoologyGDN2FutureSeedMixer,
)
from experiments.zoology_mqar.length_scaling import (
    P007_LENGTH64_TEST_HASH,
    P007_LENGTH64_TRAIN_HASH,
    build_config,
    dataset_hash,
    make_model,
    model_hash,
    parameter_hash,
)


EXPECTED_GPU_NAME = "NVIDIA A100-SXM4-80GB"
EXPECTED_GPU_UUID = "53e9f3b4-2966-65d3-6614-09c540921519"
EXPECTED_ZOOLOGY_SHA = "1ad20d193b6113cae1e8f3c655c300d7b4b3f4bb"


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
    if not gradients or not all(torch.isfinite(gradient).all() for gradient in gradients):
        raise RuntimeError("Model backward produced non-finite gradients")
    gradient_max = max(float(gradient.abs().max().item()) for gradient in gradients)
    return float(loss.item()), gradient_max


def git_head(path: Path) -> str:
    return subprocess.check_output(
        ["git", "-C", str(path), "rev-parse", "HEAD"],
        text=True,
    ).strip()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if torch.cuda.device_count() != 1:
        raise RuntimeError(f"Expected one visible GPU, got {torch.cuda.device_count()}")
    device = torch.cuda.get_device_properties(0)
    device_uuid = str(getattr(device, "uuid", "unavailable"))
    if device.name != EXPECTED_GPU_NAME:
        raise RuntimeError(f"Unexpected GPU: {device.name}")
    if device_uuid != EXPECTED_GPU_UUID:
        raise RuntimeError(f"Unexpected GPU UUID: {device_uuid}")
    if os.environ.get("FLA_EXPECTED_SOURCE_SHA") != PINNED_FLA_SHA:
        raise RuntimeError("Pinned FLA SHA is missing")
    if os.environ.get("FLA_DISABLE_BACKEND_DISPATCH") != "1":
        raise RuntimeError("FLA backend dispatch must be disabled")
    if os.environ.get("FLA_CONV_BACKEND") != "triton":
        raise RuntimeError("Short convolution must use Triton")

    fla_root = Path(os.environ["FLA_SOURCE_ROOT"]).resolve()
    direct_url = json.loads(
        (
            fla_root
            / "flash_linear_attention-0.5.2.dist-info/direct_url.json"
        ).read_text()
    )
    wheel_name = Path(direct_url["url"]).name
    wheel_hash = direct_url["archive_info"]["hashes"]["sha256"]
    expected_wheel_hash = os.environ["FLA_WHEEL_SHA256"]
    if PINNED_FLA_SHA[:7] not in wheel_name:
        raise RuntimeError(f"FLA wheel provenance lacks pinned SHA: {wheel_name}")
    if wheel_hash != expected_wheel_hash:
        raise RuntimeError(f"Unexpected FLA wheel hash: {wheel_hash}")
    source_hash = hashlib.sha256(
        (fla_root / "fla/layers/gdn2.py").read_bytes()
    ).hexdigest()
    if source_hash != os.environ["FLA_GDN2_SOURCE_SHA256"]:
        raise RuntimeError(f"Unexpected GDN2 source hash: {source_hash}")

    zoology_root = Path(os.environ["ZOOLOGY_ROOT"]).resolve()
    actual_zoology_sha = git_head(zoology_root)
    if actual_zoology_sha != EXPECTED_ZOOLOGY_SHA:
        raise RuntimeError(f"Unexpected Zoology SHA: {actual_zoology_sha}")

    length64_config = build_config(
        arm="causal_gdn2",
        sequence_length=64,
        num_kv_pairs=4,
        max_epochs=10,
        batch_size=32,
    )
    length64_train, length64_test = prepare_data(length64_config.data)
    length64_hashes = {
        "train": dataset_hash(length64_train),
        "test": dataset_hash(length64_test),
    }
    if length64_hashes["train"] != P007_LENGTH64_TRAIN_HASH:
        raise RuntimeError("Length-64 train data does not reproduce P-CAUSAL-007")
    if length64_hashes["test"] != P007_LENGTH64_TEST_HASH:
        raise RuntimeError("Length-64 test data does not reproduce P-CAUSAL-007")

    configs = {
        arm: build_config(
            arm=arm,
            sequence_length=1024,
            num_kv_pairs=4,
            max_epochs=10,
            batch_size=32,
        )
        for arm in ("causal_gdn2", "future_seed_gdn2", "bidirectional_attention")
    }
    set_determinism(123)
    reference = direct_gdn2_model(1024)
    set_determinism(123)
    causal = make_model(configs["causal_gdn2"], "causal_gdn2")
    set_determinism(123)
    future_seed = make_model(configs["future_seed_gdn2"], "future_seed_gdn2")
    set_determinism(123)
    attention = make_model(
        configs["bidirectional_attention"],
        "bidirectional_attention",
    )

    reference_state = reference.state_dict()
    causal_shared = shared_state(causal)
    if set(reference_state) != set(causal_shared):
        raise RuntimeError("Scale-0 shared state keys differ from direct GDN2")
    shared_init_max_diff = max(
        float((reference_state[name] - causal_shared[name]).abs().max().item())
        for name in reference_state
    )
    if shared_init_max_diff != 0.0:
        raise RuntimeError(f"Scale-0 shared init mismatch: {shared_init_max_diff}")
    if model_hash(causal) != model_hash(future_seed):
        raise RuntimeError("Scale-0 and scale-1 model hashes differ")
    if parameter_hash(causal) != parameter_hash(future_seed):
        raise RuntimeError("Scale-0 and scale-1 parameter hashes differ")

    parameter_counts = {
        "causal_gdn2": sum(parameter.numel() for parameter in causal.parameters()),
        "future_seed_gdn2": sum(
            parameter.numel() for parameter in future_seed.parameters()
        ),
        "bidirectional_attention": sum(
            parameter.numel() for parameter in attention.parameters()
        ),
    }
    attention_parameter_delta_fraction = (
        parameter_counts["bidirectional_attention"]
        - parameter_counts["causal_gdn2"]
    ) / parameter_counts["causal_gdn2"]
    if abs(attention_parameter_delta_fraction) > 0.005:
        raise RuntimeError(
            "Bidirectional attention is not within 0.5% of the GDN2 parameter count"
        )

    train_dataloader, test_dataloader = prepare_data(configs["causal_gdn2"].data)
    inputs, targets, _slices = next(iter(test_dataloader))
    inputs = inputs.cuda()
    targets = targets.cuda()
    altered = alter_later_values(inputs).cuda()
    quarter = inputs.shape[1] // 4
    positions = torch.arange(inputs.shape[1], device="cuda")
    future_query_mask = (targets != -100) & (positions[None, :] < quarter)

    reference = reference.cuda().eval()
    causal = causal.cuda().eval()
    future_seed = future_seed.cuda().eval()
    attention = attention.cuda().eval()
    with torch.no_grad():
        reference_logits = reference(inputs)
        causal_logits = causal(inputs)
        scale0_output_max_diff = float(
            (reference_logits - causal_logits).abs().max().item()
        )
        causal_altered = causal(altered)
        future_seed_logits = future_seed(inputs)
        future_seed_altered = future_seed(altered)
        attention_logits = attention(inputs)
        attention_altered = attention(altered)
    if scale0_output_max_diff != 0.0:
        raise RuntimeError(f"Scale-0 output is not exact: {scale0_output_max_diff}")

    dependencies = {
        "causal_gdn2_max": float(
            (causal_logits[future_query_mask] - causal_altered[future_query_mask])
            .abs()
            .max()
            .item()
        ),
        "future_seed_gdn2_mean": float(
            (
                future_seed_logits[future_query_mask]
                - future_seed_altered[future_query_mask]
            )
            .abs()
            .mean()
            .item()
        ),
        "bidirectional_attention_mean": float(
            (attention_logits[future_query_mask] - attention_altered[future_query_mask])
            .abs()
            .mean()
            .item()
        ),
    }
    if dependencies["causal_gdn2_max"] > 1e-6:
        raise RuntimeError("Causal GDN2 leaked future values")
    if dependencies["future_seed_gdn2_mean"] <= 1e-7:
        raise RuntimeError("FutureSeed does not affect early-query logits")
    if dependencies["bidirectional_attention_mean"] <= 1e-7:
        raise RuntimeError("Bidirectional attention does not see later values")

    losses_and_gradients = {}
    for arm, model in (
        ("causal_gdn2", causal),
        ("future_seed_gdn2", future_seed),
        ("bidirectional_attention", attention),
    ):
        loss, gradient_max = finite_backward(model, inputs, targets)
        losses_and_gradients[arm] = {
            "loss": loss,
            "gradient_max": gradient_max,
        }
    gate_gradient = (
        future_seed.backbone.layers[1]
        .sequence_mixer.future_seed_logit.grad
    )
    if gate_gradient is None or float(gate_gradient.abs().max().item()) == 0.0:
        raise RuntimeError("FutureSeed gate has no gradient")

    mixers = [layer.sequence_mixer for layer in future_seed.backbone.layers]
    conv_backends = [
        getattr(mixer.layer.q_conv1d, "backend", None)
        for mixer in mixers
        if isinstance(mixer, ZoologyGDN2FutureSeedMixer)
    ]
    if conv_backends != ["triton", "triton"]:
        raise RuntimeError(f"Unexpected short-conv backends: {conv_backends}")
    if [mixer.layer.mode for mixer in mixers] != ["chunk", "chunk"]:
        raise RuntimeError("GDN2 training mode is not strict chunk recurrence")

    attention_source = inspect.getsource(ParamMatchedBidirectionalAttention.forward)
    if "is_causal=False" not in attention_source:
        raise RuntimeError("Attention ceiling is not explicitly noncausal")
    gdn2_source = str(Path(inspect.getfile(GatedDeltaNet2)).resolve())
    result = {
        "cuda_device_count": torch.cuda.device_count(),
        "device": device.name,
        "device_uuid": device_uuid,
        "zoology_sha": actual_zoology_sha,
        "fla_sha": PINNED_FLA_SHA,
        "fla_wheel_sha256": wheel_hash,
        "fla_gdn2_source_sha256": source_hash,
        "gdn2_source": gdn2_source,
        "gdn2_class": f"{GatedDeltaNet2.__module__}.{GatedDeltaNet2.__name__}",
        "gdn2_modes": [mixer.layer.mode for mixer in mixers],
        "conv_backends": conv_backends,
        "length64_data_hashes": length64_hashes,
        "length1024_data_hashes": {
            "train": dataset_hash(train_dataloader),
            "test": dataset_hash(test_dataloader),
        },
        "parameter_counts": parameter_counts,
        "attention_parameter_delta_fraction": attention_parameter_delta_fraction,
        "shared_init_max_diff": shared_init_max_diff,
        "scale0_output_max_diff": scale0_output_max_diff,
        "future_dependencies": dependencies,
        "losses_and_gradients": losses_and_gradients,
        "future_seed_gate_gradient_max": float(
            gate_gradient.abs().max().item()
        ),
        "attention_is_causal": False,
        "peak_cuda_mem_bytes": torch.cuda.max_memory_allocated(),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
