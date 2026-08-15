from __future__ import annotations

import argparse
import hashlib
import importlib
import inspect
import json
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
from experiments.zoology_mqar.stable_token_address_futureseed import (
    EXPECTED_NEW_PARAMETERS,
    EXPECTED_STATE_VALUES,
    StableTokenAddressBackbone,
    StableTokenAddressGDN2,
    ZoologyStableTokenAddressFutureSeedMixer,
    load_matched_parent_state,
    parent_parameter_hash,
    stable_token_address_diagnostics,
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


EXPECTED_NATIVE_PARAMETERS = 661_584
EXPECTED_CANDIDATE_PARAMETERS = (
    EXPECTED_NATIVE_PARAMETERS + EXPECTED_NEW_PARAMETERS
)
PARENT_GRADIENT_ATOL = 0.125
PARENT_GRADIENT_REL_RMS_MAX = 0.01


def _mapped_candidate_parameters(
    model: torch.nn.Module,
) -> dict[str, torch.nn.Parameter]:
    mapped = {}
    for name, parameter in model.named_parameters():
        if name == "backbone.shared_address_proj.weight":
            continue
        parent_name = name.replace(
            ".sequence_mixer.layer.base.",
            ".sequence_mixer.layer.",
        )
        mapped[parent_name] = parameter
    return mapped


def _finite_nonzero_gradient(parameter: torch.nn.Parameter, label: str) -> float:
    gradient = parameter.grad
    if gradient is None or not torch.isfinite(gradient).all():
        raise RuntimeError(f"Missing or non-finite gradient: {label}")
    rms = gradient.float().square().mean().sqrt().item()
    if rms <= 0:
        raise RuntimeError(f"Zero gradient: {label}")
    return float(rms)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-gpu-name", required=True)
    parser.add_argument("--expected-gpu-uuid", required=True)
    args = parser.parse_args()

    if os.environ.get("CUDA_VISIBLE_DEVICES") != "0":
        raise RuntimeError("CUDA_VISIBLE_DEVICES must be exactly 0")
    if torch.cuda.device_count() != 1 or torch.cuda.current_device() != 0:
        raise RuntimeError("P-GDN3-046 requires exactly CUDA index 0")
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
        "experiments.zoology_mqar.stable_token_address_futureseed"
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
        arm="future_seed_stable_token_address_gdn2",
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
        "future_seed_stable_token_address_gdn2",
    )
    load_matched_parent_state(candidate, native.state_dict())
    if not isinstance(candidate.backbone, StableTokenAddressBackbone):
        raise RuntimeError("Candidate backbone changed")
    counts = {
        "native": sum(parameter.numel() for parameter in native.parameters()),
        "candidate": sum(parameter.numel() for parameter in candidate.parameters()),
    }
    if counts != {
        "native": EXPECTED_NATIVE_PARAMETERS,
        "candidate": EXPECTED_CANDIDATE_PARAMETERS,
    }:
        raise RuntimeError(f"Parameter counts changed: {counts}")
    if candidate.backbone.shared_address_proj.weight.numel() != EXPECTED_NEW_PARAMETERS:
        raise RuntimeError("Shared address projection size changed")
    if candidate.backbone.shared_address_proj.weight.detach().abs().max().item() != 0:
        raise RuntimeError("Shared address projection is not zero initialized")
    native_hash = parameter_hash(native)
    if parent_parameter_hash(candidate) != native_hash:
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
    if parent_init_max_diff != 0:
        raise RuntimeError("Candidate parent tensors are not exact")

    mixers = []
    provenance = []
    for layer_index, block in enumerate(candidate.backbone.layers):
        mixer = block.sequence_mixer
        if type(mixer) is not ZoologyStableTokenAddressFutureSeedMixer:
            raise RuntimeError(f"Layer {layer_index} mixer changed")
        if type(mixer.layer) is not StableTokenAddressGDN2:
            raise RuntimeError(f"Layer {layer_index} wrapper changed")
        if type(mixer.layer.base) is not GatedDeltaNet2:
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
                "carrier": type(mixer.layer.base).__qualname__,
                "operator": "ChunkGDN2FunctionBackward",
                "state_values": mixer.state_size(),
                "logical_scans": 1,
                "convolution_backends": convolutions,
            }
        )
        mixers.append(mixer)
    if len(mixers) != 2:
        raise RuntimeError("Expected two stable-address mixers")

    inputs, targets, _slices = next(iter(test_loader))
    inputs = inputs.cuda()
    targets = targets.cuda()
    native = native.cuda().eval()
    candidate = candidate.cuda().eval()
    with torch.no_grad():
        native_logits = native(inputs)
        candidate_logits = candidate(inputs)
    identity_output_max_diff = finite_max_abs_difference(
        native_logits,
        candidate_logits,
        "zero-address full output",
    )
    if identity_output_max_diff != 0:
        raise RuntimeError("Zero stable address changed full output")

    generator = torch.Generator(device="cuda").manual_seed(52046)
    incoming_rows = []
    for layer_index in range(2):
        native_mixer = native.backbone.layers[layer_index].sequence_mixer
        candidate_mixer = candidate.backbone.layers[layer_index].sequence_mixer
        hidden = torch.randn(2, 128, 128, generator=generator, device="cuda")
        incoming = torch.randn(
            2,
            4,
            32,
            32,
            generator=generator,
            device="cuda",
        )
        zero_address = torch.zeros_like(hidden)
        with torch.no_grad():
            native_output, native_state = native_mixer.forward_with_state(
                hidden,
                initial_state=incoming,
            )
            candidate_output, candidate_state = (
                candidate_mixer.forward_with_address_state(
                    hidden,
                    address_residual=zero_address,
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
            f"layer-{layer_index} incoming state",
        )
        if output_diff != 0 or state_diff != 0:
            raise RuntimeError("Zero address changed nonzero-state path")
        incoming_rows.append(
            {
                "layer": layer_index,
                "output_max_diff": output_diff,
                "state_max_diff": state_diff,
            }
        )

    native.train()
    candidate.train()
    native.zero_grad(set_to_none=True)
    candidate.zero_grad(set_to_none=True)
    native_loss = F.cross_entropy(
        native(inputs[:4]).flatten(0, 1),
        targets[:4].flatten(),
    )
    candidate_loss = F.cross_entropy(
        candidate(inputs[:4]).flatten(0, 1),
        targets[:4].flatten(),
    )
    native_loss.backward()
    candidate_loss.backward()
    backward_graph = backward_names(candidate_loss)
    official_backward_count = sum(
        name == "ChunkGDN2FunctionBackward" for name in backward_graph
    )
    if official_backward_count != 2:
        raise RuntimeError(
            f"Expected two official backward nodes, got {official_backward_count}"
        )
    shared_gradient_rms = _finite_nonzero_gradient(
        candidate.backbone.shared_address_proj.weight,
        "shared address projection",
    )
    parent_gradient_max_diff = 0.0
    compared_parent_gradients = 0
    parent_gradient_difference_energy = 0.0
    parent_gradient_reference_energy = 0.0
    candidate_parameters = _mapped_candidate_parameters(candidate)
    native_parameters = dict(native.named_parameters())
    for name in native_parameters:
        native_gradient = native_parameters[name].grad
        candidate_gradient = candidate_parameters[name].grad
        if native_gradient is None or candidate_gradient is None:
            if native_gradient is not candidate_gradient:
                raise RuntimeError(f"Parent gradient topology changed: {name}")
            continue
        compared_parent_gradients += 1
        gradient_difference = (
            native_gradient.float() - candidate_gradient.float()
        )
        parent_gradient_difference_energy += float(
            gradient_difference.square().sum().item()
        )
        parent_gradient_reference_energy += float(
            native_gradient.float().square().sum().item()
        )
        parent_gradient_max_diff = max(
            parent_gradient_max_diff,
            finite_max_abs_difference(
                native_gradient,
                candidate_gradient,
                f"parent gradient {name}",
            ),
        )
    parent_gradient_relative_rms = (
        parent_gradient_difference_energy
        / max(parent_gradient_reference_energy, 1e-24)
    ) ** 0.5
    if (
        parent_gradient_max_diff > PARENT_GRADIENT_ATOL
        or parent_gradient_relative_rms > PARENT_GRADIENT_REL_RMS_MAX
    ):
        raise RuntimeError(
            "Zero address changed parent gradients beyond BF16 tolerance: "
            f"max={parent_gradient_max_diff} "
            f"relative_rms={parent_gradient_relative_rms}"
        )

    with torch.no_grad():
        torch.nn.init.normal_(
            candidate.backbone.shared_address_proj.weight,
            mean=0.0,
            std=0.02,
        )
    active_diagnostics = stable_token_address_diagnostics(
        candidate,
        inputs[:8],
    )
    if active_diagnostics["active_layers"] != 2:
        raise RuntimeError("Not all stable-address layers are active")
    if active_diagnostics["shared_residual_data_ptr_count"] != 1:
        raise RuntimeError("Address projection is not shared across layers")
    if active_diagnostics["repeated_token_groups"] <= 0:
        raise RuntimeError("No repeated-token identity case was checked")
    if active_diagnostics["repeated_token_max_error"] > 1e-6:
        raise RuntimeError("Same token received position-dependent addresses")
    if active_diagnostics["address_residual_token_std"] <= 1e-4:
        raise RuntimeError("Stable token addresses do not vary across tokens")
    if not all(
        row["q_change_relative_rms"] > 1e-4
        and row["k_change_relative_rms"] > 1e-4
        and row["terminal_state_rms"] > 0
        and row["terminal_state_board_std"] > 0
        for row in active_diagnostics["per_layer"]
    ):
        raise RuntimeError("Active stable-address path is collapsed")

    result = {
        "status": "passed",
        "plan": "P-GDN3-046",
        "gpu": {
            "name": device.name,
            "uuid": device_uuid,
            "visible_count": torch.cuda.device_count(),
        },
        "pinned_fla_sha": PINNED_FLA_SHA,
        "gdn2_source_sha256": source_hash,
        "gdn2_ops_sha256": ops_hash,
        "zoology_sha": EXPECTED_ZOOLOGY_SHA,
        "data_hashes": data_hashes,
        "parameter_counts": counts,
        "parameter_delta": counts["candidate"] - counts["native"],
        "state_values_per_layer": EXPECTED_STATE_VALUES,
        "parent_parameter_hash": native_hash,
        "parent_init_max_diff": parent_init_max_diff,
        "identity_output_max_diff": identity_output_max_diff,
        "incoming_state_identity": incoming_rows,
        "official_backward_count": official_backward_count,
        "shared_projection_gradient_rms": shared_gradient_rms,
        "compared_parent_gradients": compared_parent_gradients,
        "parent_gradient_max_diff": parent_gradient_max_diff,
        "parent_gradient_relative_rms": parent_gradient_relative_rms,
        "parent_gradient_atol": PARENT_GRADIENT_ATOL,
        "parent_gradient_relative_rms_max": PARENT_GRADIENT_REL_RMS_MAX,
        "active_diagnostics": active_diagnostics,
        "provenance": provenance,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
