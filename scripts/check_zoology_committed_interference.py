from __future__ import annotations

import argparse
import hashlib
import importlib
import inspect
import json
import math
import os
import subprocess
from pathlib import Path

import torch
import torch.nn.functional as F

from fla.layers.gdn2 import GatedDeltaNet2
from fla.modules.convolution import ShortConvolution
from fla.ops.gdn2 import chunk_gdn2
from zoology.data.utils import prepare_data
from zoology.utils import set_determinism

from experiments.zoology_mqar.committed_interference_credit import (
    AUXILIARY_WEIGHT,
    CAUSAL_WINDOW,
    EXPECTED_INFERENCE_SCAN_DELTA,
    EXPECTED_PARAMETER_DELTA,
    EXPECTED_STATE_DELTA,
    EXPECTED_TRAINING_CONV_DELTA,
    CommittedInterferenceBackbone,
    CommittedInterferenceLanguageModel,
    committed_interference_terms,
)
from experiments.zoology_mqar.gdn2_futureseed import (
    PINNED_FLA_SHA,
    FutureSeedLanguageModel,
    ZoologyGDN2FutureSeedMixer,
)
from experiments.zoology_mqar.length_scaling import (
    build_config,
    dataset_hash,
    make_model,
    parameter_hash,
)
from scripts.check_zoology_gdn2_log_spd import (
    EXPECTED_TEST_HASH,
    EXPECTED_TRAIN_HASH,
    EXPECTED_ZOOLOGY_SHA,
    backward_names,
    finite_max_abs_difference,
    git_head,
    normalized_uuid,
    python_tree_hash,
)


EXPECTED_PARAMETERS = 661_584
EXPECTED_STATE_VALUES = 4_096
EXPECTED_INIT_SHA256 = (
    "7402e46c48cbd47070d65262d1a1b62a32ac55a4d644a6fb9644b96b0914850f"
)
EXPECTED_INIT_PARAMETER_HASH = (
    "3e8fe038f1401735168783c2de1d9217a8807e88685ee12010142fc7b05e4c44"
)
MODEL_LAYERS = 2


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _gradient_rms(parameter: torch.nn.Parameter, name: str) -> float:
    gradient = parameter.grad
    if gradient is None or not torch.isfinite(gradient).all():
        raise RuntimeError(f"Missing or non-finite gradient: {name}")
    value = float(gradient.float().square().mean().sqrt().item())
    if value <= 0:
        raise RuntimeError(f"Zero gradient: {name}")
    return value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--matched-init-path", type=Path, required=True)
    parser.add_argument("--expected-gpu-name", required=True)
    parser.add_argument("--expected-gpu-uuid", required=True)
    args = parser.parse_args()

    if os.environ.get("CUDA_VISIBLE_DEVICES") != "0":
        raise RuntimeError("CUDA_VISIBLE_DEVICES must be exactly 0")
    if torch.cuda.device_count() != 1 or torch.cuda.current_device() != 0:
        raise RuntimeError("P-GDN3-052 requires exactly CUDA index0")
    device = torch.cuda.get_device_properties(0)
    device_uuid = str(getattr(device, "uuid", "unavailable"))
    if (
        device.name != args.expected_gpu_name
        or normalized_uuid(device_uuid) != normalized_uuid(args.expected_gpu_uuid)
    ):
        raise RuntimeError(f"Unexpected GPU: {device.name} {device_uuid}")
    if os.environ.get("FLA_EXPECTED_SOURCE_SHA") != PINNED_FLA_SHA:
        raise RuntimeError("Pinned FLA SHA is missing")
    if (
        os.environ.get("FLA_DISABLE_BACKEND_DISPATCH") != "1"
        or os.environ.get("FLA_CONV_BACKEND") != "triton"
    ):
        raise RuntimeError("Official backend dispatch/Triton contract failed")
    if _sha256(args.matched_init_path) != EXPECTED_INIT_SHA256:
        raise RuntimeError("Frozen matched initialization hash drifted")

    fla_root = Path(os.environ["FLA_SOURCE_ROOT"]).resolve()
    gdn2_source = Path(inspect.getfile(GatedDeltaNet2)).resolve()
    chunk_module = importlib.import_module("fla.ops.gdn2.chunk")
    chunk_source = Path(chunk_module.__file__).resolve()
    if fla_root not in gdn2_source.parents or fla_root not in chunk_source.parents:
        raise RuntimeError("Official GDN2 escaped the pinned FLA tree")
    if chunk_module.chunk_gdn2 is not chunk_gdn2:
        raise RuntimeError("Official chunk_gdn2 export changed")
    source_hash = hashlib.sha256(gdn2_source.read_bytes()).hexdigest()
    if source_hash != os.environ["FLA_GDN2_SOURCE_SHA256"]:
        raise RuntimeError(f"Unexpected GDN2 source hash: {source_hash}")
    ops_hash = python_tree_hash(fla_root / "fla" / "ops" / "gdn2")
    if ops_hash != os.environ["FLA_GDN2_OPS_SHA256"]:
        raise RuntimeError(f"Unexpected GDN2 recurrence hash: {ops_hash}")
    zoology_root = Path(os.environ["ZOOLOGY_ROOT"]).resolve()
    zoology_status = subprocess.run(
        ["git", "-C", str(zoology_root), "status", "--porcelain"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if git_head(zoology_root) != EXPECTED_ZOOLOGY_SHA or zoology_status:
        raise RuntimeError("Zoology source is not exact and clean")

    native_config = build_config(
        arm="future_seed_gdn2",
        sequence_length=1024,
        num_kv_pairs=4,
        max_epochs=10,
        batch_size=32,
    )
    candidate_config = build_config(
        arm="future_seed_committed_interference_gdn2",
        sequence_length=1024,
        num_kv_pairs=4,
        max_epochs=10,
        batch_size=32,
    )
    train_loader, test_loader = prepare_data(native_config.data)
    data_hashes = {
        "train": dataset_hash(train_loader),
        "test": dataset_hash(test_loader),
    }
    if data_hashes != {"train": EXPECTED_TRAIN_HASH, "test": EXPECTED_TEST_HASH}:
        raise RuntimeError(f"Directional MQAR data drifted: {data_hashes}")

    parent_state = torch.load(
        args.matched_init_path,
        map_location="cpu",
        weights_only=True,
    )
    set_determinism(123)
    native = make_model(native_config, "future_seed_gdn2")
    set_determinism(123)
    candidate = make_model(
        candidate_config,
        "future_seed_committed_interference_gdn2",
    )
    if type(native) is not FutureSeedLanguageModel:
        raise RuntimeError(f"Native model type changed: {type(native)}")
    if type(candidate) is not CommittedInterferenceLanguageModel:
        raise RuntimeError(f"Candidate model type changed: {type(candidate)}")
    if type(candidate.backbone) is not CommittedInterferenceBackbone:
        raise RuntimeError("Candidate backbone changed")
    if set(native.state_dict()) != set(candidate.state_dict()):
        raise RuntimeError("Candidate state topology changed")
    native.load_state_dict(parent_state, strict=True)
    candidate.load_state_dict(parent_state, strict=True)
    parameter_counts = {
        "native": sum(parameter.numel() for parameter in native.parameters()),
        "candidate": sum(parameter.numel() for parameter in candidate.parameters()),
    }
    if parameter_counts != {
        "native": EXPECTED_PARAMETERS,
        "candidate": EXPECTED_PARAMETERS,
    }:
        raise RuntimeError(f"Parameter counts changed: {parameter_counts}")
    if parameter_hash(native) != parameter_hash(candidate):
        raise RuntimeError("Candidate parameters differ from native FutureSeed")
    if parameter_hash(candidate) != EXPECTED_INIT_PARAMETER_HASH:
        raise RuntimeError("Frozen initialization parameter hash drifted")

    provenance = []
    for layer_index, block in enumerate(candidate.backbone.layers):
        mixer = block.sequence_mixer
        if type(mixer) is not ZoologyGDN2FutureSeedMixer:
            raise RuntimeError(f"Layer {layer_index} mixer changed")
        if type(mixer.layer) is not GatedDeltaNet2:
            raise RuntimeError(f"Layer {layer_index} carrier changed")
        if mixer.state_size() != EXPECTED_STATE_VALUES:
            raise RuntimeError(f"Layer {layer_index} state size changed")
        convolutions = {
            name: module.backend
            for name, module in mixer.layer.named_modules()
            if isinstance(module, ShortConvolution)
        }
        if len(convolutions) != 3 or set(convolutions.values()) != {"triton"}:
            raise RuntimeError(f"Layer {layer_index} convolution fallback")
        provenance.append(
            {
                "layer": layer_index,
                "carrier": type(mixer.layer).__qualname__,
                "operator": "ChunkGDN2FunctionBackward",
                "state_values": mixer.state_size(),
                "logical_scans": 1,
                "convolution_backends": convolutions,
            }
        )

    inputs, targets, _slices = next(iter(test_loader))
    inputs = inputs[:4].cuda()
    targets = targets[:4].cuda()
    native = native.cuda().eval()
    candidate = candidate.cuda().eval()
    with torch.no_grad():
        native_logits = native(inputs)
        candidate_logits = candidate(inputs)
    inference_diff = finite_max_abs_difference(
        native_logits,
        candidate_logits,
        "inference logits",
    )
    if inference_diff != 0.0:
        raise RuntimeError(f"Training credit changed inference: {inference_diff}")

    generator = torch.Generator(device="cuda").manual_seed(52052)
    incoming_output_diff = 0.0
    incoming_state_diff = 0.0
    for layer_index in range(MODEL_LAYERS):
        native_mixer = native.backbone.layers[layer_index].sequence_mixer
        candidate_mixer = candidate.backbone.layers[layer_index].sequence_mixer
        hidden = torch.randn(
            2,
            97,
            128,
            generator=generator,
            device="cuda",
            dtype=torch.bfloat16,
        )
        incoming = torch.randn(
            2,
            4,
            32,
            32,
            generator=generator,
            device="cuda",
            dtype=torch.float32,
        )
        with torch.no_grad():
            native_output, native_state = native_mixer.forward_with_state(
                hidden,
                initial_state=incoming,
            )
            candidate_output, candidate_state = candidate_mixer.forward_with_state(
                hidden,
                initial_state=incoming,
            )
        incoming_output_diff = max(
            incoming_output_diff,
            finite_max_abs_difference(
                native_output,
                candidate_output,
                f"layer {layer_index} incoming output",
            ),
        )
        incoming_state_diff = max(
            incoming_state_diff,
            finite_max_abs_difference(
                native_state,
                candidate_state,
                f"layer {layer_index} incoming state",
            ),
        )
    if incoming_output_diff != 0.0 or incoming_state_diff != 0.0:
        raise RuntimeError("Nonzero incoming-state identity failed")

    native.train()
    candidate.train()
    set_determinism(52052)
    native_train_logits = native(inputs)
    set_determinism(52052)
    candidate_train_logits = candidate(inputs)
    train_diff = finite_max_abs_difference(
        native_train_logits,
        candidate_train_logits,
        "train logits",
    )
    if train_diff != 0.0:
        raise RuntimeError(f"Credit changed train logits: {train_diff}")

    auxiliary_loss = candidate.backbone.get_auxiliary_loss()
    diagnostics = candidate.backbone.interference_diagnostics()
    if not torch.isfinite(auxiliary_loss) or auxiliary_loss.item() <= 0:
        raise RuntimeError(f"Invalid auxiliary loss: {auxiliary_loss.item()}")
    if (
        diagnostics["active_layers"] != MODEL_LAYERS
        or diagnostics["auxiliary_weight"] != AUXILIARY_WEIGHT
        or diagnostics["causal_window"] != CAUSAL_WINDOW
        or diagnostics["uses_targets"]
        or not diagnostics["input_detached"]
        or not diagnostics["committed_edit_detached"]
    ):
        raise RuntimeError(f"Credit contract changed: {diagnostics}")
    if not all(
        math.isfinite(row["committed_edit_rms"])
        and row["committed_edit_rms"] > 0
        and row["surprise_cv"] > 0
        and row["above_random_fraction"] > 0
        and row["valid_fraction"] > 0.99
        for row in diagnostics["layers"]
    ):
        raise RuntimeError(f"Invalid committed-interference geometry: {diagnostics}")
    auxiliary_backward = backward_names(auxiliary_loss)
    auxiliary_official_count = auxiliary_backward.count("ChunkGDN2FunctionBackward")
    if auxiliary_official_count != 0:
        raise RuntimeError("Credit must not backpropagate through the recurrent scan")

    candidate.zero_grad(set_to_none=True)
    auxiliary_loss.backward()
    nonzero_auxiliary = {
        name: _gradient_rms(parameter, name)
        for name, parameter in candidate.named_parameters()
        if parameter.grad is not None and parameter.grad.abs().max().item() != 0.0
    }
    unexpected_auxiliary = {
        name
        for name in nonzero_auxiliary
        if ".sequence_mixer.layer.k_proj." not in name
        and ".sequence_mixer.layer.k_conv1d." not in name
    }
    if unexpected_auxiliary:
        raise RuntimeError(
            f"Credit leaked outside K projection/conv: {unexpected_auxiliary}"
        )
    for layer_index in range(MODEL_LAYERS):
        prefix = f"backbone.layers.{layer_index}.sequence_mixer.layer."
        if not any(name.startswith(prefix + "k_proj.") for name in nonzero_auxiliary):
            raise RuntimeError(f"Layer {layer_index} K projection has zero gradient")
        if not any(
            name.startswith(prefix + "k_conv1d.") for name in nonzero_auxiliary
        ):
            raise RuntimeError(f"Layer {layer_index} K convolution has zero gradient")

    candidate.zero_grad(set_to_none=True)
    set_determinism(52053)
    logits = candidate(inputs)
    combined_loss = F.cross_entropy(
        logits.flatten(0, 1),
        targets.flatten(),
    ) + candidate.backbone.get_auxiliary_loss()
    combined_backward = backward_names(combined_loss)
    combined_official_count = combined_backward.count("ChunkGDN2FunctionBackward")
    if combined_official_count != MODEL_LAYERS:
        raise RuntimeError(
            f"Expected two official backward paths, got {combined_official_count}"
        )
    combined_loss.backward()
    missing = {
        name
        for name, parameter in candidate.named_parameters()
        if parameter.grad is None
    }
    expected_missing = {"backbone.layers.0.sequence_mixer.future_seed_logit"}
    if missing != expected_missing:
        raise RuntimeError(f"Combined gradient topology changed: {missing}")
    if not all(
        torch.isfinite(parameter.grad).all()
        for name, parameter in candidate.named_parameters()
        if name not in expected_missing
    ):
        raise RuntimeError("Combined objective produced non-finite gradients")

    synthetic_key = torch.randn(
        2, 257, 4 * 32, generator=generator, device="cuda"
    )
    synthetic_edit = torch.randn(
        2, 257, 4, 32, generator=generator, device="cuda"
    )
    base_loss, base_diagnostics = committed_interference_terms(
        synthetic_key,
        synthetic_edit,
        head_dim=32,
    )
    permutation = torch.tensor([2, 0, 3, 1], device="cuda")
    permuted_key = synthetic_key.view(2, 257, 4, 32)[:, :, permutation].reshape(
        2, 257, 4 * 32
    )
    permuted_edit = synthetic_edit[:, :, permutation]
    permuted_loss, _ = committed_interference_terms(
        permuted_key,
        permuted_edit,
        head_dim=32,
    )
    scaled_loss, _ = committed_interference_terms(
        7.0 * synthetic_key,
        3.0 * synthetic_edit,
        head_dim=32,
    )
    repeated_key = synthetic_key[:, :1].expand(-1, 257, -1).clone()
    repeated_loss, _ = committed_interference_terms(
        repeated_key,
        synthetic_edit,
        head_dim=32,
    )
    head_permutation_error = float((base_loss - permuted_loss).abs().item())
    scale_invariance_error = float((base_loss - scaled_loss).abs().item())
    if head_permutation_error > 1e-5 or scale_invariance_error > 1e-5:
        raise RuntimeError("Credit lost head equivariance or scale invariance")
    if repeated_loss <= base_loss:
        raise RuntimeError("Credit does not detect repeated-address interference")

    result = {
        "status": "passed",
        "plan": "P-GDN3-052",
        "gpu": {
            "name": device.name,
            "uuid": device_uuid,
            "visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        },
        "provenance": {
            "pinned_fla_sha": PINNED_FLA_SHA,
            "gdn2_source": str(gdn2_source),
            "gdn2_source_sha256": source_hash,
            "gdn2_ops_tree_sha256": ops_hash,
            "zoology_sha": EXPECTED_ZOOLOGY_SHA,
            "layers": provenance,
        },
        "data_hashes": data_hashes,
        "matched_initialization": {
            "path": str(args.matched_init_path),
            "sha256": EXPECTED_INIT_SHA256,
            "parameter_hash": parameter_hash(candidate),
        },
        "parameter_counts": parameter_counts,
        "identity": {
            "inference_output_max_diff": inference_diff,
            "train_output_max_diff": train_diff,
            "nonzero_incoming_output_max_diff": incoming_output_diff,
            "nonzero_incoming_state_max_diff": incoming_state_diff,
        },
        "credit": {
            "diagnostics": diagnostics,
            "auxiliary_official_backward_count": auxiliary_official_count,
            "combined_official_backward_count": combined_official_count,
            "nonzero_auxiliary_gradient_rms": nonzero_auxiliary,
            "synthetic": {
                "base_loss": float(base_loss.item()),
                "repeated_address_loss": float(repeated_loss.item()),
                "head_permutation_error": head_permutation_error,
                "scale_invariance_error": scale_invariance_error,
                "diagnostics": base_diagnostics,
            },
        },
        "contract": {
            "new_parameters": EXPECTED_PARAMETER_DELTA,
            "new_inference_state": EXPECTED_STATE_DELTA,
            "new_inference_scans": EXPECTED_INFERENCE_SCAN_DELTA,
            "training_short_conv_delta": EXPECTED_TRAINING_CONV_DELTA,
            "targets_used": False,
            "input_detached": True,
            "committed_edit_detached": True,
            "silent_fallback": False,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
