from __future__ import annotations

import argparse
import copy
import hashlib
import inspect
import json
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from fla.layers.gdn2 import GatedDeltaNet2
from zoology.model import LanguageModel
from zoology.utils import set_determinism

from experiments.zoology_mqar.futureseed_directionality import model_hash
from experiments.zoology_mqar.gdn2_futureseed import (
    PINNED_FLA_SHA,
    FutureSeedLanguageModel,
    ZoologyGDN2FutureSeedMixer,
)
from experiments.zoology_mlm.bidirectional_attention import (
    FullBidirectionalAttention,
)
from experiments.zoology_mlm.futureseed_mlm import build_config


EXPECTED_GPU_UUID = "GPU-53e9f3b4-2966-65d3-6614-09c540921519"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _shared_state(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: tensor.detach()
        for name, tensor in model.state_dict().items()
        if not name.endswith("future_seed_logit")
    }


def _model(
    arm: str,
    prepared_path: Path,
    prepared_sha256: str,
) -> tuple[torch.nn.Module, object]:
    config = build_config(
        arm=arm,
        prepared_path=prepared_path,
        prepared_sha256=prepared_sha256,
        max_epochs=4,
        batch_size=64,
    )
    set_determinism(123)
    if arm == "bidirectional_attention":
        model = LanguageModel(copy.deepcopy(config.model))
    else:
        model = FutureSeedLanguageModel(copy.deepcopy(config.model))
    return model, config


def _future_dependency(
    model: torch.nn.Module,
    inputs: torch.Tensor,
    probe_position: int,
) -> float:
    altered = inputs.clone()
    suffix = altered[:, probe_position + 1 :]
    altered[:, probe_position + 1 :] = torch.where(
        suffix == 256,
        suffix,
        (suffix + 17) % 256,
    )
    model.eval()
    original_logits = model(inputs)[:, probe_position]
    altered_logits = model(altered)[:, probe_position]
    return float((original_logits - altered_logits).abs().mean().item())


def _finite_backward(
    model: torch.nn.Module,
    inputs: torch.Tensor,
    labels: torch.Tensor,
) -> tuple[float, float]:
    model.train()
    model.zero_grad(set_to_none=True)
    logits = model(inputs)
    loss = F.cross_entropy(logits.flatten(0, 1), labels.flatten())
    loss.backward()
    gradients = [
        parameter.grad
        for parameter in model.parameters()
        if parameter.grad is not None
    ]
    if not gradients or not all(torch.isfinite(gradient).all() for gradient in gradients):
        raise RuntimeError("Non-finite or missing gradients")
    return float(loss.item()), max(float(gradient.abs().max().item()) for gradient in gradients)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepared-path", type=Path, required=True)
    parser.add_argument("--prepared-sha256", required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if torch.cuda.device_count() != 1:
        raise RuntimeError(f"Expected one visible GPU, got {torch.cuda.device_count()}")
    device = torch.cuda.get_device_properties(0)
    torch_device_uuid = str(getattr(device, "uuid", "unavailable"))
    device_uuid = (
        torch_device_uuid
        if torch_device_uuid.startswith("GPU-")
        else f"GPU-{torch_device_uuid}"
    )
    if device.name != "NVIDIA A100-SXM4-80GB":
        raise RuntimeError(f"Unexpected GPU: {device.name}")
    if device_uuid != EXPECTED_GPU_UUID:
        raise RuntimeError(f"Unexpected GPU UUID: {device_uuid}")
    if os.environ.get("FLA_EXPECTED_SOURCE_SHA") != PINNED_FLA_SHA:
        raise RuntimeError("Pinned FLA SHA was not asserted")
    if os.environ.get("FLA_DISABLE_BACKEND_DISPATCH") != "1":
        raise RuntimeError("FLA backend dispatch must be disabled")
    if os.environ.get("FLA_CONV_BACKEND") != "triton":
        raise RuntimeError("Short convolution must use Triton")
    if _sha256(args.prepared_path) != args.prepared_sha256:
        raise RuntimeError("Prepared dataset hash mismatch")

    manifest = json.loads(args.manifest.read_text())
    if manifest["prepared"]["sha256"] != args.prepared_sha256:
        raise RuntimeError("Manifest and prepared dataset disagree")
    with np.load(args.prepared_path, allow_pickle=False) as archive:
        shapes = {name: list(archive[name].shape) for name in archive.files}
        train_labels = np.asarray(archive["train_labels"])
        valid_inputs = torch.from_numpy(
            np.asarray(archive["valid_inputs"][:8], dtype=np.int64)
        ).cuda()
        valid_labels = torch.from_numpy(
            np.asarray(archive["valid_labels"][:8], dtype=np.int64)
        ).cuda()
    masked_per_train_example = np.sum(train_labels != -100, axis=1)
    if shapes != {
        "train_inputs": [20_000, 256],
        "train_labels": [20_000, 256],
        "valid_inputs": [2_000, 256],
        "valid_labels": [2_000, 256],
    }:
        raise RuntimeError(f"Unexpected prepared shapes: {shapes}")
    if not np.all(masked_per_train_example == 38):
        raise RuntimeError("Mask count is not exactly 38 per train example")

    reference_config = build_config(
        arm="causal_gdn2",
        prepared_path=args.prepared_path,
        prepared_sha256=args.prepared_sha256,
        max_epochs=4,
        batch_size=64,
    )
    set_determinism(123)
    reference = LanguageModel(copy.deepcopy(reference_config.model))
    no_fs, _ = _model("causal_gdn2", args.prepared_path, args.prepared_sha256)
    fs, _ = _model("future_seed_gdn2", args.prepared_path, args.prepared_sha256)
    bidirectional, _ = _model(
        "bidirectional_attention", args.prepared_path, args.prepared_sha256
    )
    if model_hash(no_fs) != model_hash(fs):
        raise RuntimeError("Matched GDN2 initialization hashes differ")
    reference_state = reference.state_dict()
    shared_state = _shared_state(no_fs)
    if set(reference_state) != set(shared_state):
        raise RuntimeError("Scale-0 shared state keys differ from upstream LanguageModel")
    shared_init_max_diff = max(
        float((reference_state[name] - shared_state[name]).abs().max().item())
        for name in reference_state
    )
    if shared_init_max_diff != 0.0:
        raise RuntimeError("Scale-0 shared initialization is not exact")

    reference = reference.cuda().eval()
    no_fs = no_fs.cuda().eval()
    fs = fs.cuda().eval()
    bidirectional = bidirectional.cuda().eval()
    with torch.no_grad():
        reference_logits = reference(valid_inputs)
        no_fs_logits = no_fs(valid_inputs)
    scale0_output_max_diff = float(
        (reference_logits - no_fs_logits).abs().max().item()
    )
    if scale0_output_max_diff != 0.0:
        raise RuntimeError(f"Scale-0 output mismatch: {scale0_output_max_diff}")

    probe_candidates = torch.nonzero(valid_labels[0] != -100).flatten()
    probe_candidates = probe_candidates[probe_candidates < 128]
    if probe_candidates.numel() == 0:
        raise RuntimeError("No early masked probe position")
    probe_position = int(probe_candidates[0].item())
    dependencies = {
        "causal_gdn2": _future_dependency(no_fs, valid_inputs, probe_position),
        "future_seed_gdn2": _future_dependency(fs, valid_inputs, probe_position),
        "bidirectional_attention": _future_dependency(
            bidirectional, valid_inputs, probe_position
        ),
    }
    if dependencies["causal_gdn2"] > 1e-7:
        raise RuntimeError(f"Causal GDN2 leaks future input: {dependencies}")
    if dependencies["future_seed_gdn2"] <= 1e-7:
        raise RuntimeError("FutureSeed has no future-input dependency")
    if dependencies["bidirectional_attention"] <= 1e-7:
        raise RuntimeError("Bidirectional ceiling is accidentally causal")

    torch.cuda.reset_peak_memory_stats()
    backward = {}
    gate_gradient = None
    for arm, model in (
        ("causal_gdn2", no_fs),
        ("future_seed_gdn2", fs),
        ("bidirectional_attention", bidirectional),
    ):
        loss, gradient_max = _finite_backward(model, valid_inputs, valid_labels)
        backward[arm] = {"loss": loss, "gradient_max": gradient_max}
        if arm == "future_seed_gdn2":
            gate_gradient_tensor = (
                model.backbone.layers[1]
                .sequence_mixer.future_seed_logit.grad
            )
            gate_gradient = (
                None
                if gate_gradient_tensor is None
                else float(gate_gradient_tensor.abs().max().item())
            )
    if gate_gradient is None or gate_gradient == 0.0:
        raise RuntimeError("FutureSeed gate has no gradient")

    mixers = [layer.sequence_mixer for layer in fs.backbone.layers]
    source = str(Path(inspect.getfile(GatedDeltaNet2)).resolve())
    result = {
        "cuda_device_count": torch.cuda.device_count(),
        "device": device.name,
        "device_uuid": device_uuid,
        "torch_device_uuid_raw": torch_device_uuid,
        "fla_sha": PINNED_FLA_SHA,
        "gdn2_source": source,
        "gdn2_class": f"{GatedDeltaNet2.__module__}.{GatedDeltaNet2.__name__}",
        "gdn2_mode": [mixer.layer.mode for mixer in mixers],
        "gdn2_conv_backends": [
            getattr(mixer.layer.q_conv1d, "backend", None)
            for mixer in mixers
            if isinstance(mixer, ZoologyGDN2FutureSeedMixer)
        ],
        "bidirectional_class": (
            f"{FullBidirectionalAttention.__module__}."
            f"{FullBidirectionalAttention.__name__}"
        ),
        "bidirectional_is_causal": False,
        "prepared_sha256": args.prepared_sha256,
        "prepared_shapes": shapes,
        "masked_tokens_per_example": int(masked_per_train_example[0]),
        "gdn2_parameters": sum(parameter.numel() for parameter in fs.parameters()),
        "bidirectional_parameters": sum(
            parameter.numel() for parameter in bidirectional.parameters()
        ),
        "matched_gdn2_init_hash": model_hash(fs),
        "shared_init_max_diff": shared_init_max_diff,
        "scale0_output_max_diff": scale0_output_max_diff,
        "probe_position": probe_position,
        "future_dependency_mean_abs": dependencies,
        "future_seed_gate_gradient_max": gate_gradient,
        "backward": backward,
        "peak_cuda_mem_bytes": torch.cuda.max_memory_allocated(),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
