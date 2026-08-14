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

from experiments.zoology_mqar.dynamic_frame_futureseed import (
    EXPECTED_PARAMETER_DELTA,
    EXPECTED_STATE_VALUES,
    FRAME_RANK,
    LOG_FRAME_RADIUS,
    DynamicFrameGDN2,
    ZoologyDynamicFrameFutureSeedMixer,
    load_matched_parent_state,
    parent_parameter_hash,
)
from experiments.zoology_mqar.gdn2_futureseed import PINNED_FLA_SHA
from experiments.zoology_mqar.length_scaling import (
    build_config,
    dataset_hash,
    make_model,
    parameter_hash,
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
EXPECTED_CANDIDATE_PARAMETERS = 665_680
MODEL_LAYERS = 2
MODEL_HEADS = 4
HEAD_DIM = 32


def _rms(tensor: torch.Tensor) -> float:
    return float(tensor.float().square().mean().sqrt().item())


def _mapped_parent_parameters(
    model: torch.nn.Module,
) -> dict[str, torch.nn.Parameter]:
    mapped = {}
    for name, parameter in model.named_parameters():
        if ".sequence_mixer.layer.frame." in name:
            continue
        parent_name = name.replace(
            ".sequence_mixer.layer.base.",
            ".sequence_mixer.layer.",
        )
        mapped[parent_name] = parameter
    return mapped


def _assert_finite_nonzero_gradient(
    parameter: torch.nn.Parameter,
    name: str,
) -> float:
    gradient = parameter.grad
    if gradient is None or not torch.isfinite(gradient).all():
        raise RuntimeError(f"Missing or non-finite gradient: {name}")
    value = _rms(gradient)
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
        raise RuntimeError("P-GDN3-044 requires exactly CUDA index0")
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
        "experiments.zoology_mqar.dynamic_frame_futureseed"
    )
    if fla_root not in gdn2_source.parents or fla_root not in chunk_source.parents:
        raise RuntimeError("Official GDN2 escaped the pinned FLA tree")
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
        arm="future_seed_dynamic_frame_gdn2",
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
    candidate = make_model(candidate_config, "future_seed_dynamic_frame_gdn2")
    load_matched_parent_state(candidate, native.state_dict())
    parameter_counts = {
        "native": sum(parameter.numel() for parameter in native.parameters()),
        "candidate": sum(parameter.numel() for parameter in candidate.parameters()),
    }
    if parameter_counts != {
        "native": EXPECTED_NATIVE_PARAMETERS,
        "candidate": EXPECTED_CANDIDATE_PARAMETERS,
    }:
        raise RuntimeError(f"Parameter counts changed: {parameter_counts}")
    if parameter_counts["candidate"] - parameter_counts["native"] != EXPECTED_PARAMETER_DELTA:
        raise RuntimeError("Dynamic-frame parameter delta changed")
    native_hash = parameter_hash(native)
    candidate_parent_hash = parent_parameter_hash(candidate)
    if candidate_parent_hash != native_hash:
        raise RuntimeError("Candidate parent initialization changed")

    native_parameters = dict(native.named_parameters())
    candidate_parameters = _mapped_parent_parameters(candidate)
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
        raise RuntimeError(f"Candidate parent tensors changed: {parent_init_max_diff}")

    mixers: list[ZoologyDynamicFrameFutureSeedMixer] = []
    provenance = []
    for layer_index, block in enumerate(candidate.backbone.layers):
        mixer = block.sequence_mixer
        if type(mixer) is not ZoologyDynamicFrameFutureSeedMixer:
            raise RuntimeError(f"Layer {layer_index} mixer changed: {type(mixer)}")
        layer = mixer.layer
        if type(layer) is not DynamicFrameGDN2 or type(layer.base) is not GatedDeltaNet2:
            raise RuntimeError(f"Layer {layer_index} carrier changed")
        if mixer.state_size() != EXPECTED_STATE_VALUES:
            raise RuntimeError(f"Layer {layer_index} state size changed")
        if tuple(layer.frame.in_proj.weight.shape) != (FRAME_RANK, 128):
            raise RuntimeError("Frame input projection shape changed")
        if tuple(layer.frame.out_proj.weight.shape) != (128, FRAME_RANK):
            raise RuntimeError("Frame output projection shape changed")
        if layer.frame.out_proj.weight.detach().abs().max().item() != 0.0:
            raise RuntimeError(f"Layer {layer_index} frame is not identity-init")
        convolutions = {
            name: module.backend
            for name, module in layer.named_modules()
            if isinstance(module, ShortConvolution)
        }
        if len(convolutions) != 3 or set(convolutions.values()) != {"triton"}:
            raise RuntimeError(f"Layer {layer_index} convolution fallback: {convolutions}")
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
    candidate = candidate.cuda().eval()
    native_parameters = dict(native.named_parameters())
    candidate_parameters = _mapped_parent_parameters(candidate)
    with torch.no_grad():
        native_logits = native(inputs)
        candidate_logits = candidate(inputs)
    identity_output_max_diff = finite_max_abs_difference(
        native_logits,
        candidate_logits,
        "zero-frame full output",
    )
    if identity_output_max_diff != 0.0:
        raise RuntimeError(f"Zero frame changed full output: {identity_output_max_diff}")

    generator = torch.Generator(device="cuda").manual_seed(52044)
    incoming_rows = []
    incoming_output_max_diff = 0.0
    incoming_state_max_diff = 0.0
    for layer_index in range(MODEL_LAYERS):
        native_mixer = native.backbone.layers[layer_index].sequence_mixer
        candidate_mixer = candidate.backbone.layers[layer_index].sequence_mixer
        hidden = torch.randn(2, 128, 128, generator=generator, device="cuda")
        incoming = torch.randn(
            2,
            MODEL_HEADS,
            HEAD_DIM,
            HEAD_DIM,
            generator=generator,
            device="cuda",
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
        output_diff = finite_max_abs_difference(
            native_output,
            candidate_output,
            f"layer-{layer_index} incoming output",
        )
        state_diff = finite_max_abs_difference(
            native_state,
            candidate_state,
            f"layer-{layer_index} incoming state",
        )
        incoming_output_max_diff = max(incoming_output_max_diff, output_diff)
        incoming_state_max_diff = max(incoming_state_max_diff, state_diff)
        incoming_rows.append(
            {"layer": layer_index, "output_max_diff": output_diff, "state_max_diff": state_diff}
        )
    if incoming_output_max_diff != 0.0 or incoming_state_max_diff != 0.0:
        raise RuntimeError("Zero frame changed the nonzero-state path")

    hidden = torch.randn(2, 128, 128, generator=generator, device="cuda")
    with torch.no_grad():
        native_output, native_state = native.backbone.layers[0].sequence_mixer.forward_with_state(
            hidden,
            initial_state=None,
        )
        candidate_output, candidate_state = candidate.backbone.layers[0].sequence_mixer.forward_with_state(
            hidden,
            initial_state=None,
        )
        native_seed = native.backbone.layers[1].sequence_mixer.make_initial_state(native_state)
        candidate_seed = candidate.backbone.layers[1].sequence_mixer.make_initial_state(candidate_state)
    future_seed_transport_max_diff = max(
        finite_max_abs_difference(native_output, candidate_output, "producer output"),
        finite_max_abs_difference(native_state, candidate_state, "producer state"),
        finite_max_abs_difference(native_seed, candidate_seed, "transported seed"),
    )
    if future_seed_transport_max_diff != 0.0:
        raise RuntimeError("Zero frame changed native FutureSeed transport")

    native.train().zero_grad(set_to_none=True)
    candidate.train().zero_grad(set_to_none=True)
    cpu_rng_state = torch.get_rng_state()
    cuda_rng_state = torch.cuda.get_rng_state()
    native_train_logits = native(inputs)
    torch.set_rng_state(cpu_rng_state)
    torch.cuda.set_rng_state(cuda_rng_state)
    candidate_train_logits = candidate(inputs)
    mask = targets != -100
    native_loss = F.cross_entropy(native_train_logits[mask], targets[mask])
    candidate_loss = F.cross_entropy(candidate_train_logits[mask], targets[mask])
    official_backward_count = sum(
        "ChunkGDN2FunctionBackward" in name
        for name in backward_names(candidate_loss)
    )
    if official_backward_count != MODEL_LAYERS:
        raise RuntimeError(f"Expected two official backwards, got {official_backward_count}")
    native_loss.backward()
    candidate_loss.backward()
    zero_frame_gradient_rows = []
    for layer_index, mixer in enumerate(mixers):
        out_gradient = _assert_finite_nonzero_gradient(
            mixer.layer.frame.out_proj.weight,
            f"layer-{layer_index} frame out",
        )
        in_gradient = mixer.layer.frame.in_proj.weight.grad
        if in_gradient is None or not torch.isfinite(in_gradient).all():
            raise RuntimeError("Identity frame input gradient is invalid")
        if in_gradient.abs().max().item() != 0.0:
            raise RuntimeError("Identity frame input projection should be two-stage")
        zero_frame_gradient_rows.append(
            {"layer": layer_index, "out_gradient_rms": out_gradient, "in_gradient_rms": _rms(in_gradient)}
        )

    parent_gradient_max_diff = 0.0
    absent_in_both = []
    for name, parameter in native_parameters.items():
        candidate_parameter = candidate_parameters[name]
        if parameter.grad is None and candidate_parameter.grad is None:
            absent_in_both.append(name)
            continue
        if parameter.grad is None or candidate_parameter.grad is None:
            raise RuntimeError(f"Parent gradient presence differs: {name}")
        if not torch.isfinite(parameter.grad).all() or not torch.isfinite(candidate_parameter.grad).all():
            raise RuntimeError(f"Non-finite parent gradient: {name}")
        parent_gradient_max_diff = max(
            parent_gradient_max_diff,
            finite_max_abs_difference(
                parameter.grad,
                candidate_parameter.grad,
                f"parent gradient {name}",
            ),
        )
    if parent_gradient_max_diff != 0.0:
        raise RuntimeError(f"Zero frame changed parent gradients: {parent_gradient_max_diff}")
    if set(absent_in_both) != EXPECTED_UNUSED_PARENT_GRADIENTS:
        raise RuntimeError(f"Unexpected unused parent gradients: {absent_in_both}")

    opened_rows = []
    for layer_index, mixer in enumerate(mixers):
        layer = mixer.layer
        with torch.no_grad():
            coordinates = torch.linspace(
                -0.015,
                0.015,
                steps=layer.frame.out_proj.weight.numel(),
                device="cuda",
            ).reshape_as(layer.frame.out_proj.weight)
            layer.frame.out_proj.weight.copy_(coordinates)
        candidate.train().zero_grad(set_to_none=True)
        opened_logits = candidate(inputs)
        opened_loss = F.cross_entropy(opened_logits[mask], targets[mask])
        opened_loss.backward()
        in_gradient_rms = _assert_finite_nonzero_gradient(
            layer.frame.in_proj.weight,
            f"layer-{layer_index} opened frame in",
        )
        out_gradient_rms = _assert_finite_nonzero_gradient(
            layer.frame.out_proj.weight,
            f"layer-{layer_index} opened frame out",
        )

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
        native_mixer = native.backbone.layers[layer_index].sequence_mixer
        with torch.no_grad():
            log_frame, frame_delta, factor = layer.frame_terms(mechanism_hidden)
            native_open_output, native_open_state = native_mixer.forward_with_state(
                mechanism_hidden,
                initial_state=mechanism_incoming,
            )
            opened_output, opened_state = mixer.forward_with_state(
                mechanism_hidden,
                initial_state=mechanism_incoming,
            )
        cumulative_error = float((frame_delta.cumsum(dim=1) - log_frame).abs().max().item())
        factor_min = float(factor.min().item())
        factor_max = float(factor.max().item())
        condition_max = float(
            (
                factor.amax(dim=(0, 1, 3))
                / factor.amin(dim=(0, 1, 3))
            ).max().item()
        )
        output_delta = _rms(opened_output - native_open_output)
        state_delta = _rms(opened_state - native_open_state)
        if cumulative_error > 1e-5:
            raise RuntimeError(f"Layer {layer_index} frame does not telescope")
        if not (0.70 <= factor_min <= factor_max <= 1.42 and condition_max <= 2.05):
            raise RuntimeError(f"Layer {layer_index} frame bound changed")
        if not all(math.isfinite(value) and value > 1e-5 for value in (output_delta, state_delta)):
            raise RuntimeError(f"Layer {layer_index} opened frame is inactive")
        opened_rows.append(
            {
                "layer": layer_index,
                "frame_log_rms": _rms(log_frame),
                "frame_delta_rms": _rms(frame_delta),
                "frame_cumulative_max_error": cumulative_error,
                "factor_min": factor_min,
                "factor_max": factor_max,
                "factor_condition_max": condition_max,
                "output_delta_rms": output_delta,
                "terminal_state_delta_rms": state_delta,
                "terminal_state_rms": _rms(opened_state),
                "opened_in_gradient_rms": in_gradient_rms,
                "opened_out_gradient_rms": out_gradient_rms,
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
        "zero_frame_gradient_rows": zero_frame_gradient_rows,
        "parent_gradient_max_diff": parent_gradient_max_diff,
        "parent_gradients_absent_in_both": absent_in_both,
        "frame_rank": FRAME_RANK,
        "log_frame_radius": LOG_FRAME_RADIUS,
        "opened_rows": opened_rows,
        "provenance": provenance,
        "peak_cuda_mem_bytes": torch.cuda.max_memory_allocated(),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
