from __future__ import annotations

import argparse
import hashlib
import importlib
import inspect
import json
import math
import os
from pathlib import Path

import torch
import torch.nn.functional as F

from fla.layers.gdn2 import GatedDeltaNet2
from fla.modules.convolution import ShortConvolution
from fla.ops.gdn2 import chunk_gdn2
from zoology.data.utils import prepare_data
from zoology.utils import set_determinism

from experiments.zoology_mqar.futureseed_read_credit import (
    AUXILIARY_WEIGHT,
    EXPECTED_PARAMETER_DELTA,
    ReceiverReadCreditBackbone,
    ReceiverReadCreditLanguageModel,
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
MODEL_LAYERS = 2


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
    parser.add_argument("--expected-gpu-name", required=True)
    parser.add_argument("--expected-gpu-uuid", required=True)
    args = parser.parse_args()

    if os.environ.get("CUDA_VISIBLE_DEVICES") != "0":
        raise RuntimeError("CUDA_VISIBLE_DEVICES must be exactly 0")
    if torch.cuda.device_count() != 1 or torch.cuda.current_device() != 0:
        raise RuntimeError("P-FS2-011 requires exactly CUDA index0")
    device = torch.cuda.get_device_properties(0)
    device_uuid = str(getattr(device, "uuid", "unavailable"))
    if (
        device.name != args.expected_gpu_name
        or normalized_uuid(device_uuid)
        != normalized_uuid(args.expected_gpu_uuid)
    ):
        raise RuntimeError(f"Unexpected GPU: {device.name} {device_uuid}")
    if os.environ.get("FLA_EXPECTED_SOURCE_SHA") != PINNED_FLA_SHA:
        raise RuntimeError("Pinned FLA SHA is missing")
    if os.environ.get("FLA_DISABLE_BACKEND_DISPATCH") != "1":
        raise RuntimeError("FLA backend dispatch must be disabled")
    if os.environ.get("FLA_CONV_BACKEND") != "triton":
        raise RuntimeError("Short convolution must use Triton")

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
    if git_head(zoology_root) != EXPECTED_ZOOLOGY_SHA:
        raise RuntimeError("Unexpected Zoology SHA")

    native_config = build_config(
        arm="future_seed_gdn2",
        sequence_length=1024,
        num_kv_pairs=4,
        max_epochs=10,
        batch_size=32,
    )
    candidate_config = build_config(
        arm="future_seed_receiver_read_credit_gdn2",
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
    if data_hashes != {
        "train": EXPECTED_TRAIN_HASH,
        "test": EXPECTED_TEST_HASH,
    }:
        raise RuntimeError(f"Directional MQAR data drifted: {data_hashes}")

    set_determinism(123)
    native = make_model(native_config, "future_seed_gdn2")
    set_determinism(123)
    candidate = make_model(
        candidate_config,
        "future_seed_receiver_read_credit_gdn2",
    )
    if type(native) is not FutureSeedLanguageModel:
        raise RuntimeError(f"Native model type changed: {type(native)}")
    if type(candidate) is not ReceiverReadCreditLanguageModel:
        raise RuntimeError(f"Candidate model type changed: {type(candidate)}")
    if type(candidate.backbone) is not ReceiverReadCreditBackbone:
        raise RuntimeError("Candidate backbone changed")
    parameter_counts = {
        "native": sum(parameter.numel() for parameter in native.parameters()),
        "candidate": sum(parameter.numel() for parameter in candidate.parameters()),
    }
    if parameter_counts != {
        "native": EXPECTED_PARAMETERS,
        "candidate": EXPECTED_PARAMETERS,
    }:
        raise RuntimeError(f"Parameter counts changed: {parameter_counts}")
    if parameter_counts["candidate"] - parameter_counts["native"] != EXPECTED_PARAMETER_DELTA:
        raise RuntimeError("Receiver credit changed parameter count")
    if parameter_hash(native) != parameter_hash(candidate):
        raise RuntimeError("Candidate initialization differs from native FutureSeed")
    if set(native.state_dict()) != set(candidate.state_dict()):
        raise RuntimeError("Candidate state dictionary topology changed")
    candidate.load_state_dict(native.state_dict(), strict=True)

    provenance = []
    for layer_index, block in enumerate(candidate.backbone.layers):
        mixer = block.sequence_mixer
        if type(mixer) is not ZoologyGDN2FutureSeedMixer:
            raise RuntimeError(f"Layer {layer_index} mixer changed: {type(mixer)}")
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
            raise RuntimeError(f"Layer {layer_index} convolution fallback: {convolutions}")
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
    if len(provenance) != MODEL_LAYERS:
        raise RuntimeError("Candidate layer count changed")

    inputs, targets, _slices = next(iter(test_loader))
    inputs = inputs[:8].cuda()
    targets = targets[:8].cuda()
    native = native.cuda().eval()
    candidate = candidate.cuda().eval()
    with torch.no_grad():
        native_logits = native(inputs)
        candidate_logits = candidate(inputs)
    inference_identity_max_diff = finite_max_abs_difference(
        native_logits,
        candidate_logits,
        "inference logits",
    )
    if inference_identity_max_diff != 0.0:
        raise RuntimeError(
            f"Training-only credit changed inference: {inference_identity_max_diff}"
        )

    generator = torch.Generator(device="cuda").manual_seed(52011)
    incoming_output_max_diff = 0.0
    incoming_state_max_diff = 0.0
    for layer_index in range(MODEL_LAYERS):
        native_mixer = native.backbone.layers[layer_index].sequence_mixer
        candidate_mixer = candidate.backbone.layers[layer_index].sequence_mixer
        hidden = torch.randn(
            3,
            65,
            128,
            generator=generator,
            device="cuda",
            dtype=torch.bfloat16,
        )
        incoming = torch.randn(
            3,
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
        incoming_output_max_diff = max(
            incoming_output_max_diff,
            finite_max_abs_difference(
                native_output,
                candidate_output,
                f"layer {layer_index} incoming output",
            ),
        )
        incoming_state_max_diff = max(
            incoming_state_max_diff,
            finite_max_abs_difference(
                native_state,
                candidate_state,
                f"layer {layer_index} incoming state",
            ),
        )
    if incoming_output_max_diff != 0.0 or incoming_state_max_diff != 0.0:
        raise RuntimeError("Nonzero incoming-state identity failed")

    native.train()
    candidate.train()
    set_determinism(52011)
    native_train_logits = native(inputs)
    set_determinism(52011)
    candidate_train_logits = candidate(inputs)
    train_identity_max_diff = finite_max_abs_difference(
        native_train_logits,
        candidate_train_logits,
        "train logits",
    )
    if train_identity_max_diff != 0.0:
        raise RuntimeError(f"Credit changed train logits: {train_identity_max_diff}")

    auxiliary_loss = candidate.backbone.get_auxiliary_loss()
    diagnostics = candidate.backbone.credit_diagnostics()
    if not torch.isfinite(auxiliary_loss) or auxiliary_loss.item() <= 0:
        raise RuntimeError(f"Invalid auxiliary loss: {auxiliary_loss.item()}")
    if diagnostics["active_routes"] != 1:
        raise RuntimeError(f"Unexpected credit routes: {diagnostics}")
    if diagnostics["auxiliary_weight"] != AUXILIARY_WEIGHT:
        raise RuntimeError("Auxiliary weight changed")
    if diagnostics["uses_targets"]:
        raise RuntimeError("Receiver credit became target-selective")
    route = diagnostics["routes"][0]
    if route["query_content_board_std"] <= 1e-6:
        raise RuntimeError("Receiver query content does not vary across boards")
    auxiliary_backward = backward_names(auxiliary_loss)
    auxiliary_official_backward_count = auxiliary_backward.count(
        "ChunkGDN2FunctionBackward"
    )
    if auxiliary_official_backward_count != 1:
        raise RuntimeError(
            "Auxiliary credit must traverse exactly the producer official scan: "
            f"{auxiliary_official_backward_count}"
        )

    candidate.zero_grad(set_to_none=True)
    auxiliary_loss.backward()
    named = dict(candidate.named_parameters())
    producer_gradient_names = [
        "backbone.layers.0.sequence_mixer.layer.k_proj.weight",
        "backbone.layers.0.sequence_mixer.layer.v_proj.weight",
        "backbone.layers.0.sequence_mixer.layer.f_proj.0.weight",
        "backbone.layers.0.sequence_mixer.layer.b_proj.weight",
        "backbone.layers.0.sequence_mixer.layer.w_proj.weight",
        "backbone.layers.1.sequence_mixer.future_seed_logit",
    ]
    producer_gradient_rms = {
        name: _gradient_rms(named[name], name)
        for name in producer_gradient_names
    }
    receiver_gradient_names = [
        name
        for name in named
        if name.startswith("backbone.layers.1.sequence_mixer.layer.")
    ]
    receiver_nonzero_gradients = {
        name: float(parameter.grad.float().abs().max().item())
        for name, parameter in named.items()
        if name in receiver_gradient_names
        and parameter.grad is not None
        and parameter.grad.abs().max().item() != 0.0
    }
    if receiver_nonzero_gradients:
        raise RuntimeError(
            "Producer-only credit leaked into receiver parameters: "
            f"{receiver_nonzero_gradients}"
        )

    candidate.zero_grad(set_to_none=True)
    set_determinism(52012)
    logits = candidate(inputs)
    auxiliary_loss = candidate.backbone.get_auxiliary_loss()
    combined_loss = F.cross_entropy(
        logits.flatten(0, 1),
        targets.flatten(),
    ) + auxiliary_loss
    combined_backward = backward_names(combined_loss)
    combined_official_backward_count = combined_backward.count(
        "ChunkGDN2FunctionBackward"
    )
    if combined_official_backward_count != MODEL_LAYERS:
        raise RuntimeError(
            f"Expected two official backward paths, got {combined_official_backward_count}"
        )
    combined_loss.backward()
    missing_combined_gradients = {
        name
        for name, parameter in candidate.named_parameters()
        if parameter.grad is None
    }
    expected_missing = {
        "backbone.layers.0.sequence_mixer.future_seed_logit",
    }
    if missing_combined_gradients != expected_missing:
        raise RuntimeError(
            "Combined objective gradient topology changed: "
            f"{missing_combined_gradients}"
        )
    if not all(
        torch.isfinite(parameter.grad).all()
        for name, parameter in candidate.named_parameters()
        if name not in expected_missing
    ):
        raise RuntimeError("Combined objective produced non-finite gradients")

    result = {
        "status": "passed",
        "plan": "P-FS2-011",
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
        "parameter_counts": parameter_counts,
        "parameter_hash": parameter_hash(candidate),
        "identity": {
            "inference_output_max_diff": inference_identity_max_diff,
            "train_output_max_diff": train_identity_max_diff,
            "nonzero_incoming_output_max_diff": incoming_output_max_diff,
            "nonzero_incoming_state_max_diff": incoming_state_max_diff,
        },
        "credit": {
            "diagnostics": diagnostics,
            "auxiliary_official_backward_count": auxiliary_official_backward_count,
            "combined_official_backward_count": combined_official_backward_count,
            "producer_gradient_rms": producer_gradient_rms,
            "receiver_nonzero_gradients": receiver_nonzero_gradients,
        },
        "contract": {
            "new_parameters": EXPECTED_PARAMETER_DELTA,
            "new_inference_state": 0,
            "new_inference_scans": 0,
            "targets_used": False,
            "receiver_query_detached": True,
            "teacher_detached": True,
            "silent_fallback": False,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
