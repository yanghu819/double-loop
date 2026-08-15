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

from experiments.zoology_mqar.gdn2_futureseed import PINNED_FLA_SHA
from experiments.zoology_mqar.length_scaling import (
    build_config,
    dataset_hash,
    make_model,
    parameter_hash,
)
from experiments.zoology_mqar.address_payload_gauge_futureseed import (
    EXPECTED_NEW_PARAMETERS,
    PAYLOAD_LOG_RADIUS,
    EXPECTED_STATE_VALUES,
    ReciprocalPayloadGaugeGDN2,
    ZoologyAddressPayloadGaugeFutureSeedMixer,
    load_matched_parent_state,
    parent_parameter_hash,
)
from scripts.check_zoology_gdn2_log_spd import (
    EXPECTED_TEST_HASH,
    EXPECTED_TRAIN_HASH,
    EXPECTED_UNUSED_PARENT_GRADIENTS,
    EXPECTED_ZOOLOGY_SHA,
    backward_names,
    finite_max_abs_difference,
    git_head,
    normalized_uuid,
    python_tree_hash,
)


EXPECTED_NATIVE_PARAMETERS = 661_584
EXPECTED_CANDIDATE_PARAMETERS = EXPECTED_NATIVE_PARAMETERS + EXPECTED_NEW_PARAMETERS
EXPECTED_PARAMETER_DELTA = EXPECTED_NEW_PARAMETERS
MODEL_LAYERS = 2
MODEL_HEADS = 4
HEAD_DIM = 32
PARENT_GRADIENT_ATOL = 0.125
PARENT_GRADIENT_REL_RMS_MAX = 0.01
PARENT_GRADIENT_NOISE_MULTIPLIER = 2.0


def _rms(tensor: torch.Tensor) -> float:
    return float(tensor.float().square().mean().sqrt().item())


def _mapped_candidate_parameters(
    model: torch.nn.Module,
) -> dict[str, torch.nn.Parameter]:
    mapped = {}
    for name, parameter in model.named_parameters():
        if name.endswith("layer.payload_gauge_logit"):
            continue
        parent_name = name.replace(
            ".sequence_mixer.layer.base.",
            ".sequence_mixer.layer.",
        )
        mapped[parent_name] = parameter
    return mapped


def _finite_adapter_gradients(
    mixers: list[ZoologyAddressPayloadGaugeFutureSeedMixer],
) -> list[dict[str, object]]:
    rows = []
    for layer_index, mixer in enumerate(mixers):
        gradient = mixer.layer.payload_gauge_logit.grad
        if gradient is None or not torch.isfinite(gradient).all():
            raise RuntimeError(
                f"Layer {layer_index} address payload gauge gradient missing or non-finite"
            )
        per_head = gradient.float().abs().flatten()
        if (per_head <= 0).any():
            raise RuntimeError(
                f"Layer {layer_index} has a zero-gradient address payload gauge head"
            )
        rows.append(
            {
                "layer": layer_index,
                "gradient_rms": _rms(gradient),
                "per_head_gradient_rms": [
                    float(value.item()) for value in per_head
                ],
            }
        )
    return rows


def _gradient_difference_summary(
    reference: dict[str, torch.nn.Parameter],
    compared: dict[str, torch.nn.Parameter],
    *,
    label: str,
) -> dict[str, float | int]:
    if set(reference) != set(compared):
        raise RuntimeError(f"Gradient parameter names changed: {label}")
    max_difference = 0.0
    difference_energy = 0.0
    reference_energy = 0.0
    compared_count = 0
    absent_count = 0
    for name, parameter in reference.items():
        other = compared[name]
        if parameter.grad is None and other.grad is None:
            absent_count += 1
            continue
        if parameter.grad is None or other.grad is None:
            raise RuntimeError(f"Gradient topology changed for {label}: {name}")
        if not torch.isfinite(parameter.grad).all() or not torch.isfinite(
            other.grad
        ).all():
            raise RuntimeError(f"Non-finite gradient for {label}: {name}")
        compared_count += 1
        difference = parameter.grad.float() - other.grad.float()
        max_difference = max(max_difference, float(difference.abs().max().item()))
        difference_energy += float(difference.square().sum().item())
        reference_energy += float(parameter.grad.float().square().sum().item())
    return {
        "max_abs": max_difference,
        "relative_rms": (
            difference_energy / max(reference_energy, 1e-24)
        ) ** 0.5,
        "compared_parameter_count": compared_count,
        "absent_parameter_count": absent_count,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-gpu-name", required=True)
    parser.add_argument("--expected-gpu-uuid", required=True)
    args = parser.parse_args()

    if os.environ.get("CUDA_VISIBLE_DEVICES") != "0":
        raise RuntimeError("CUDA_VISIBLE_DEVICES must be exactly 0")
    if torch.cuda.device_count() != 1 or torch.cuda.current_device() != 0:
        raise RuntimeError("P-GDN3-048 requires exactly CUDA index 0")
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
    model_module = importlib.import_module(
        "experiments.zoology_mqar.address_payload_gauge_futureseed"
    )
    if fla_root not in gdn2_source.parents or fla_root not in chunk_source.parents:
        raise RuntimeError(
            f"Official carrier escaped pinned FLA: {gdn2_source} {chunk_source}"
        )
    if model_module.chunk_gdn2 is not chunk_gdn2:
        raise RuntimeError("Candidate chunk export identity changed")
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
        arm="future_seed_address_payload_gauge_gdn2",
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
    native_replay = make_model(native_config, "future_seed_gdn2")
    native_replay.load_state_dict(native.state_dict(), strict=True)
    set_determinism(123)
    candidate = make_model(
        candidate_config,
        "future_seed_address_payload_gauge_gdn2",
    )
    load_matched_parent_state(candidate, native.state_dict())
    parameter_counts = {
        "native": sum(parameter.numel() for parameter in native.parameters()),
        "candidate": sum(
            parameter.numel() for parameter in candidate.parameters()
        ),
    }
    if parameter_counts != {
        "native": EXPECTED_NATIVE_PARAMETERS,
        "candidate": EXPECTED_CANDIDATE_PARAMETERS,
    }:
        raise RuntimeError(f"Parameter counts changed: {parameter_counts}")
    if (
        parameter_counts["candidate"] - parameter_counts["native"]
        != EXPECTED_PARAMETER_DELTA
    ):
        raise RuntimeError("Q/K address payload gauge parameter delta changed")
    native_hash = parameter_hash(native)
    candidate_parent_hash = parent_parameter_hash(candidate)
    if candidate_parent_hash != native_hash:
        raise RuntimeError("Candidate parent initialization changed")

    native_parameters = dict(native.named_parameters())
    candidate_parameters = _mapped_candidate_parameters(candidate)
    if set(native_parameters) != set(candidate_parameters):
        raise RuntimeError("Candidate parent parameter names changed")
    parent_init_max_diff = max(
        finite_max_abs_difference(
            native_parameters[name],
            candidate_parameters[name],
            f"parent tensor {name}",
        )
        for name in native_parameters
    )
    if parent_init_max_diff != 0.0:
        raise RuntimeError(
            f"Candidate parent tensors changed: {parent_init_max_diff}"
        )

    mixers: list[ZoologyAddressPayloadGaugeFutureSeedMixer] = []
    provenance = []
    for layer_index, block in enumerate(candidate.backbone.layers):
        mixer = block.sequence_mixer
        if type(mixer) is not ZoologyAddressPayloadGaugeFutureSeedMixer:
            raise RuntimeError(f"Layer {layer_index} mixer changed: {type(mixer)}")
        layer = mixer.layer
        if type(layer) is not ReciprocalPayloadGaugeGDN2:
            raise RuntimeError(f"Layer {layer_index} wrapper changed: {type(layer)}")
        if type(layer.base) is not GatedDeltaNet2:
            raise RuntimeError(f"Layer {layer_index} carrier changed")
        if mixer.state_size() != EXPECTED_STATE_VALUES:
            raise RuntimeError(f"Layer {layer_index} state size changed")
        if layer.payload_gauge_logit.numel() != MODEL_HEADS:
            raise RuntimeError(f"Layer {layer_index} adapter shape changed")
        if layer.payload_gauge_logit.detach().abs().max().item() != 0.0:
            raise RuntimeError(f"Layer {layer_index} adapter is not zero-init")
        convolutions = {
            name: module.backend
            for name, module in layer.named_modules()
            if isinstance(module, ShortConvolution)
        }
        if len(convolutions) != 3 or set(convolutions.values()) != {"triton"}:
            raise RuntimeError(
                f"Layer {layer_index} convolution fallback: {convolutions}"
            )
        provenance.append(
            {
                "layer": layer_index,
                "carrier": type(layer.base).__qualname__,
                "operator": "ChunkGDN2FunctionBackward",
                "state_values": mixer.state_size(),
                "logical_scans": 1,
                "convolution_backends": convolutions,
            }
        )
        mixers.append(mixer)
    if len(mixers) != MODEL_LAYERS:
        raise RuntimeError(f"Expected two candidate mixers, got {len(mixers)}")

    inputs, targets, _slices = next(iter(test_loader))
    inputs = inputs.cuda()
    targets = targets.cuda()
    native = native.cuda().eval()
    native_replay = native_replay.cuda().eval()
    candidate = candidate.cuda().eval()
    native_parameters = dict(native.named_parameters())
    candidate_parameters = _mapped_candidate_parameters(candidate)
    with torch.no_grad():
        native_logits = native(inputs)
        candidate_logits = candidate(inputs)
    identity_output_max_diff = finite_max_abs_difference(
        native_logits,
        candidate_logits,
        "zero-adapter full output",
    )
    if identity_output_max_diff != 0.0:
        raise RuntimeError(
            f"Zero adapter changed full output: {identity_output_max_diff}"
        )

    generator = torch.Generator(device="cuda").manual_seed(52042)
    incoming_rows = []
    incoming_output_max_diff = 0.0
    incoming_state_max_diff = 0.0
    for layer_index in range(MODEL_LAYERS):
        native_mixer = native.backbone.layers[layer_index].sequence_mixer
        candidate_mixer = candidate.backbone.layers[layer_index].sequence_mixer
        hidden = torch.randn(
            2,
            128,
            128,
            generator=generator,
            device="cuda",
        )
        incoming = torch.randn(
            2,
            MODEL_HEADS,
            HEAD_DIM,
            HEAD_DIM,
            generator=generator,
            device="cuda",
            dtype=torch.float32,
        )
        with torch.no_grad():
            native_output, native_state = native_mixer.forward_with_state(
                hidden,
                initial_state=incoming,
            )
            candidate_output, candidate_state = (
                candidate_mixer.forward_with_state(
                    hidden,
                    initial_state=incoming,
                )
            )
        output_diff = finite_max_abs_difference(
            native_output,
            candidate_output,
            f"layer-{layer_index} incoming output",
        )
        state_diff = finite_max_abs_difference(
            native_state,
            candidate_state,
            f"layer-{layer_index} incoming terminal state",
        )
        incoming_output_max_diff = max(incoming_output_max_diff, output_diff)
        incoming_state_max_diff = max(incoming_state_max_diff, state_diff)
        incoming_rows.append(
            {
                "layer": layer_index,
                "output_max_diff": output_diff,
                "state_max_diff": state_diff,
            }
        )
    if incoming_output_max_diff != 0.0 or incoming_state_max_diff != 0.0:
        raise RuntimeError(
            "Zero adapter changed nonzero-state path: "
            f"{incoming_output_max_diff} {incoming_state_max_diff}"
        )

    hidden = torch.randn(
        2,
        128,
        128,
        generator=generator,
        device="cuda",
    )
    with torch.no_grad():
        native_first_output, native_first_state = (
            native.backbone.layers[0].sequence_mixer.forward_with_state(
                hidden,
                initial_state=None,
            )
        )
        candidate_first_output, candidate_first_state = (
            candidate.backbone.layers[0].sequence_mixer.forward_with_state(
                hidden,
                initial_state=None,
            )
        )
        native_seed = native.backbone.layers[1].sequence_mixer.make_initial_state(
            native_first_state
        )
        candidate_seed = (
            candidate.backbone.layers[1].sequence_mixer.make_initial_state(
                candidate_first_state
            )
        )
    future_seed_transport_max_diff = max(
        finite_max_abs_difference(
            native_first_output,
            candidate_first_output,
            "FutureSeed producer output",
        ),
        finite_max_abs_difference(
            native_first_state,
            candidate_first_state,
            "FutureSeed producer state",
        ),
        finite_max_abs_difference(
            native_seed,
            candidate_seed,
            "FutureSeed transported state",
        ),
    )
    if future_seed_transport_max_diff != 0.0:
        raise RuntimeError(
            "Zero adapter changed native FutureSeed transport: "
            f"{future_seed_transport_max_diff}"
        )

    native.train().zero_grad(set_to_none=True)
    native_replay.train().zero_grad(set_to_none=True)
    candidate.train().zero_grad(set_to_none=True)
    cpu_rng_state = torch.get_rng_state()
    cuda_rng_state = torch.cuda.get_rng_state()
    native_train_logits = native(inputs)
    torch.set_rng_state(cpu_rng_state)
    torch.cuda.set_rng_state(cuda_rng_state)
    native_replay_train_logits = native_replay(inputs)
    torch.set_rng_state(cpu_rng_state)
    torch.cuda.set_rng_state(cuda_rng_state)
    candidate_train_logits = candidate(inputs)
    mask = targets != -100
    native_loss = F.cross_entropy(native_train_logits[mask], targets[mask])
    native_replay_loss = F.cross_entropy(
        native_replay_train_logits[mask], targets[mask]
    )
    candidate_loss = F.cross_entropy(candidate_train_logits[mask], targets[mask])
    graph_names = backward_names(candidate_loss)
    official_backward_count = sum(
        "ChunkGDN2FunctionBackward" in name for name in graph_names
    )
    if official_backward_count != MODEL_LAYERS:
        raise RuntimeError(
            f"Expected two official GDN2 backward paths, got {official_backward_count}"
        )
    native_loss.backward()
    native_replay_loss.backward()
    candidate_loss.backward()
    adapter_gradients = _finite_adapter_gradients(mixers)

    replay_parameters = dict(native_replay.named_parameters())
    native_replay_gradient_noise = _gradient_difference_summary(
        native_parameters,
        replay_parameters,
        label="native replay",
    )
    candidate_parent_gradient_difference = _gradient_difference_summary(
        native_parameters,
        candidate_parameters,
        label="address-payload parent",
    )
    calibrated_max_abs = max(
        PARENT_GRADIENT_ATOL,
        PARENT_GRADIENT_NOISE_MULTIPLIER
        * float(native_replay_gradient_noise["max_abs"]),
    )
    calibrated_relative_rms = max(
        PARENT_GRADIENT_REL_RMS_MAX,
        PARENT_GRADIENT_NOISE_MULTIPLIER
        * float(native_replay_gradient_noise["relative_rms"]),
    )
    if (
        float(candidate_parent_gradient_difference["max_abs"])
        > calibrated_max_abs
        or float(candidate_parent_gradient_difference["relative_rms"])
        > calibrated_relative_rms
    ):
        raise RuntimeError(
            "Zero payload gauge changed parent gradients beyond calibrated "
            f"BF16 noise: candidate={candidate_parent_gradient_difference} "
            f"native_replay={native_replay_gradient_noise} "
            f"limits={calibrated_max_abs}/{calibrated_relative_rms}"
        )
    absent_in_both = [
        name
        for name, parameter in native_parameters.items()
        if parameter.grad is None and candidate_parameters[name].grad is None
    ]
    if set(absent_in_both) != EXPECTED_UNUSED_PARENT_GRADIENTS:
        raise RuntimeError(f"Unexpected unused parent gradients: {absent_in_both}")

    opened_rows = []
    for layer_index, mixer in enumerate(mixers):
        layer = mixer.layer
        coordinates = torch.linspace(-0.60, 0.60, steps=MODEL_HEADS, device="cuda")
        coordinates = coordinates.reshape(MODEL_HEADS, 1)
        with torch.no_grad():
            layer.payload_gauge_logit.copy_(coordinates + 0.05 * layer_index)
            q = torch.randn(
                3,
                17,
                MODEL_HEADS,
                HEAD_DIM,
                generator=generator,
                device="cuda",
            )
            k = torch.randn(
                3,
                17,
                MODEL_HEADS,
                HEAD_DIM,
                generator=generator,
                device="cuda",
            )
            write_factor = layer.payload_factor(k)
            read_inverse = layer.payload_factor(q, inverse=True)
            same_inverse = layer.payload_factor(k, inverse=True)
            reciprocal_error = finite_max_abs_difference(
                write_factor * same_inverse,
                torch.ones_like(write_factor),
                f"layer-{layer_index} reciprocal factor",
            )
            raw = layer.payload_gauge_logit.clone()
            permutation = torch.tensor([2, 0, 3, 1], device="cuda")
            expected_write = write_factor[:, :, permutation]
            expected_read = read_inverse[:, :, permutation]
            layer.payload_gauge_logit.copy_(raw[permutation])
            actual_write = layer.payload_factor(k[:, :, permutation])
            actual_read = layer.payload_factor(
                q[:, :, permutation], inverse=True
            )
            layer.payload_gauge_logit.copy_(raw)
            head_permutation_error = max(
                finite_max_abs_difference(
                    expected_write,
                    actual_write,
                    f"layer-{layer_index} permuted write factor",
                ),
                finite_max_abs_difference(
                    expected_read,
                    actual_read,
                    f"layer-{layer_index} permuted read factor",
                ),
            )
        if reciprocal_error > 2e-3:
            raise RuntimeError(f"Layer {layer_index} reciprocal code drifted")
        if head_permutation_error != 0.0:
            raise RuntimeError(
                f"Layer {layer_index} head equivariance failed: "
                f"{head_permutation_error}"
            )
        if not (
            0.5 <= write_factor.min().item()
            and write_factor.max().item() <= 2.0
            and write_factor.std(unbiased=False).item() > 1e-3
            and _rms(write_factor * read_inverse - 1.0) > 1e-4
        ):
            raise RuntimeError(f"Layer {layer_index} payload factors collapsed")

        native_mixer = native.backbone.layers[layer_index].sequence_mixer
        mechanism_hidden = torch.randn(
            2,
            128,
            128,
            generator=generator,
            device="cuda",
        )
        mechanism_incoming = torch.randn(
            2,
            MODEL_HEADS,
            HEAD_DIM,
            HEAD_DIM,
            generator=generator,
            device="cuda",
        )
        with torch.no_grad():
            native_output, native_state = native_mixer.forward_with_state(
                mechanism_hidden,
                initial_state=mechanism_incoming,
            )
            opened_output, opened_state = mixer.forward_with_state(
                mechanism_hidden,
                initial_state=mechanism_incoming,
            )
        output_delta = float(
            (opened_output - native_output).float().square().mean().sqrt().item()
        )
        state_delta = float(
            (opened_state - native_state).float().square().mean().sqrt().item()
        )
        if (
            not math.isfinite(output_delta)
            or not math.isfinite(state_delta)
            or output_delta <= 1e-5
            or state_delta <= 1e-5
        ):
            raise RuntimeError(f"Layer {layer_index} opened adapter is inactive")
        opened_rows.append(
            {
                "layer": layer_index,
                "factor_min": float(write_factor.min().item()),
                "factor_max": float(write_factor.max().item()),
                "factor_std": float(write_factor.std(unbiased=False).item()),
                "same_address_reciprocal_max_error": reciprocal_error,
                "qk_code_mismatch_rms": _rms(write_factor * read_inverse - 1.0),
                "head_permutation_max_error": head_permutation_error,
                "output_delta_rms": output_delta,
                "terminal_state_delta_rms": state_delta,
                "terminal_state_rms": _rms(opened_state),
            }
        )

    result = {
        "device": device.name,
        "device_uuid": device_uuid,
        "cuda_device_count": torch.cuda.device_count(),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "zoology_sha": EXPECTED_ZOOLOGY_SHA,
        "fla_sha": PINNED_FLA_SHA,
        "fla_gdn2_source": str(gdn2_source),
        "fla_gdn2_source_sha256": source_hash,
        "fla_gdn2_ops_sha256": ops_hash,
        "chunk_source": str(chunk_source),
        "data_hashes": data_hashes,
        "parameter_counts": parameter_counts,
        "parameter_delta": EXPECTED_PARAMETER_DELTA,
        "state_values_per_layer": EXPECTED_STATE_VALUES,
        "logical_scans_per_layer": 1,
        "control_init_parameter_hash": native_hash,
        "candidate_parent_parameter_hash": candidate_parent_hash,
        "parent_init_max_diff": parent_init_max_diff,
        "identity_output_max_diff": identity_output_max_diff,
        "incoming_output_max_diff": incoming_output_max_diff,
        "incoming_state_max_diff": incoming_state_max_diff,
        "incoming_rows": incoming_rows,
        "future_seed_transport_max_diff": future_seed_transport_max_diff,
        "official_backward_count": official_backward_count,
        "adapter_gradients": adapter_gradients,
        "native_replay_gradient_noise": native_replay_gradient_noise,
        "candidate_parent_gradient_difference": (
            candidate_parent_gradient_difference
        ),
        "calibrated_parent_gradient_max_abs": calibrated_max_abs,
        "calibrated_parent_gradient_relative_rms": calibrated_relative_rms,
        "parent_gradients_absent_in_both": absent_in_both,
        "payload_log_radius": PAYLOAD_LOG_RADIUS,
        "opened_rows": opened_rows,
        "provenance": provenance,
        "peak_cuda_mem_bytes": torch.cuda.max_memory_allocated(),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
