from __future__ import annotations

import argparse
import copy
import inspect
import json
import os
import subprocess
from pathlib import Path

import torch
import torch.nn.functional as F
from einops import rearrange
from fla.layers.gdn2 import GatedDeltaNet2
from fla.modules.convolution import ShortConvolution
from zoology.data.utils import prepare_data
from zoology.utils import set_determinism

from experiments.zoology_mqar.gdn2_futureseed import PINNED_FLA_SHA
from experiments.zoology_mqar.gdn2_slot_state import (
    SLOT_COUNT,
    ZoologySlotStateFutureSeedMixer,
    _route_gate,
    load_matched_parent_state,
    parent_parameter_hash,
    slot_state_diagnostics,
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
)


EXPECTED_PARAMETER_DELTA = 2_056


def _mixers(model: torch.nn.Module) -> list[ZoologySlotStateFutureSeedMixer]:
    mixers = [block.sequence_mixer for block in model.backbone.layers]
    if len(mixers) != 2 or not all(
        isinstance(mixer, ZoologySlotStateFutureSeedMixer) for mixer in mixers
    ):
        raise RuntimeError("Expected exactly two slot-state mixers")
    return mixers


def _mapped_parent_parameters(
    model: torch.nn.Module,
) -> dict[str, torch.nn.Parameter]:
    result = {}
    for name, parameter in model.named_parameters():
        if ".sequence_mixer.layer.slot_router." in name:
            continue
        parent_name = name.replace(
            ".sequence_mixer.layer.base.",
            ".sequence_mixer.layer.",
        )
        if name.endswith("future_seed_logit"):
            parameter = rearrange(
                parameter,
                "b (h s) k v -> b h s k v",
                s=SLOT_COUNT,
            )[:, :, 0]
        result[parent_name] = parameter
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-gpu-name", required=True)
    parser.add_argument("--expected-gpu-uuid", required=True)
    args = parser.parse_args()

    if os.environ.get("CUDA_VISIBLE_DEVICES") != "0":
        raise RuntimeError("CUDA_VISIBLE_DEVICES must be exactly 0")
    if torch.cuda.device_count() != 1 or torch.cuda.current_device() != 0:
        raise RuntimeError("P-GDN3-038 requires one visible CUDA index 0")
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
    fla_root = Path(os.environ["FLA_SOURCE_ROOT"]).resolve()
    if fla_root not in Path(inspect.getfile(GatedDeltaNet2)).resolve().parents:
        raise RuntimeError("GDN2 is not imported from the pinned FLA root")
    zoology_root = Path(os.environ["ZOOLOGY_ROOT"]).resolve()
    zoology_status = subprocess.run(
        ["git", "-C", str(zoology_root), "status", "--porcelain"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if git_head(zoology_root) != EXPECTED_ZOOLOGY_SHA or zoology_status:
        raise RuntimeError("Zoology source is not exact and clean")

    control_config = build_config(
        arm="future_seed_gdn2",
        sequence_length=1024,
        num_kv_pairs=4,
        max_epochs=10,
        batch_size=32,
    )
    candidate_config = build_config(
        arm="future_seed_slot_state_gdn2",
        sequence_length=1024,
        num_kv_pairs=4,
        max_epochs=10,
        batch_size=32,
    )
    train_loader, test_loader = prepare_data(control_config.data)
    hashes = {
        "train": dataset_hash(train_loader),
        "test": dataset_hash(test_loader),
    }
    if hashes != {"train": EXPECTED_TRAIN_HASH, "test": EXPECTED_TEST_HASH}:
        raise RuntimeError(f"Directional MQAR data drifted: {hashes}")

    set_determinism(123)
    control = make_model(control_config, "future_seed_gdn2")
    control_state = copy.deepcopy(control.state_dict())
    set_determinism(123)
    candidate = make_model(candidate_config, "future_seed_slot_state_gdn2")
    load_matched_parent_state(candidate, control_state)
    if parent_parameter_hash(candidate) != parameter_hash(control):
        raise RuntimeError("Slot-state insertion changed parent initialization")
    counts = {
        "control": sum(parameter.numel() for parameter in control.parameters()),
        "candidate": sum(parameter.numel() for parameter in candidate.parameters()),
    }
    if counts["candidate"] - counts["control"] != EXPECTED_PARAMETER_DELTA:
        raise RuntimeError(f"Unexpected parameter delta: {counts}")
    control_parameters = dict(control.named_parameters())
    candidate_parameters = _mapped_parent_parameters(candidate)
    if set(control_parameters) != set(candidate_parameters):
        raise RuntimeError("Candidate parent parameter names differ from control")
    parent_init_max_diff = max(
        finite_max_abs_difference(
            control_parameters[name],
            candidate_parameters[name],
            f"parent initialization {name}",
        )
        for name in control_parameters
    )
    if parent_init_max_diff != 0.0:
        raise RuntimeError("Candidate parent tensors differ from control")

    convolution_counts = []
    for model in (control, candidate):
        convolutions = [
            module
            for module in model.modules()
            if module.__class__.__name__ == "ShortConvolution"
        ]
        if len(convolutions) != 6 or not all(
            isinstance(module, ShortConvolution) for module in convolutions
        ):
            raise RuntimeError("Short-convolution provenance failed")
        convolution_counts.append(len(convolutions))

    inputs, targets, _slices = next(iter(test_loader))
    inputs = inputs.cuda()
    targets = targets.cuda()
    control = control.cuda().eval()
    candidate = candidate.cuda().eval()
    candidate_mixers = _mixers(candidate)
    saved_router_weights = [
        mixer.layer.slot_router.weight.detach().clone() for mixer in candidate_mixers
    ]
    with torch.no_grad():
        for mixer in candidate_mixers:
            mixer.layer.slot_router.weight.zero_()
        control_logits = control(inputs)
        uniform_logits = candidate(inputs)
    uniform_output_max_diff = finite_max_abs_difference(
        control_logits,
        uniform_logits,
        "uniform-slot full output",
    )
    if uniform_output_max_diff != 0.0:
        raise RuntimeError(
            f"Uniform two-slot state changed the parent: {uniform_output_max_diff}"
        )

    generator = torch.Generator(device="cuda").manual_seed(52038)
    incoming_results = []
    for layer_index, (control_block, candidate_mixer) in enumerate(
        zip(control.backbone.layers, candidate_mixers)
    ):
        control_mixer = control_block.sequence_mixer
        hidden = torch.randn(
            2, 128, 128, generator=generator, device="cuda"
        )
        incoming = 0.05 * torch.randn(
            2, 4, 32, 32, generator=generator, device="cuda"
        )
        with torch.no_grad():
            control_output, control_terminal = control_mixer.forward_with_state(
                hidden,
                initial_state=incoming,
            )
            slot_output, slot_terminal = candidate_mixer.forward_with_state(
                hidden,
                initial_state=incoming.repeat_interleave(SLOT_COUNT, dim=1),
            )
        slot_terminal = rearrange(
            slot_terminal,
            "b (h s) k v -> b h s k v",
            h=4,
            s=SLOT_COUNT,
        )
        output_diff = finite_max_abs_difference(
            control_output,
            slot_output,
            f"incoming output {layer_index}",
        )
        state_diff = max(
            finite_max_abs_difference(
                control_terminal,
                slot_terminal[:, :, slot],
                f"incoming state {layer_index}/{slot}",
            )
            for slot in range(SLOT_COUNT)
        )
        if output_diff != 0.0 or state_diff != 0.0:
            raise RuntimeError("Uniform slot identity failed for nonzero state")
        incoming_results.append(
            {
                "layer": layer_index,
                "output_max_diff": output_diff,
                "state_max_diff": state_diff,
            }
        )

    with torch.no_grad():
        for mixer, weight in zip(candidate_mixers, saved_router_weights):
            mixer.layer.slot_router.weight.copy_(weight)
        candidate_logits = candidate(inputs)
    opened_output_relative_rms = float(
        (
            (candidate_logits.float() - uniform_logits.float())
            .square()
            .mean()
            .sqrt()
            / uniform_logits.float().square().mean().sqrt().clamp_min(1e-8)
        ).item()
    )
    if opened_output_relative_rms <= 1e-5:
        raise RuntimeError("Registered slot router did not alter the full model")

    candidate.train().zero_grad(set_to_none=True)
    train_logits = candidate(inputs)
    graph = backward_names(train_logits)
    mask = targets != -100
    F.cross_entropy(train_logits[mask], targets[mask]).backward()
    official_backward_count = sum(
        "ChunkGDN2FunctionBackward" in name for name in graph
    )
    if official_backward_count != 2:
        raise RuntimeError(
            f"Expected two official GDN2 backwards, got {official_backward_count}"
        )
    router_gradient_norms = []
    for mixer in candidate_mixers:
        gradient = mixer.layer.slot_router.weight.grad
        if gradient is None or not torch.isfinite(gradient).all():
            raise RuntimeError("Slot-router gradient is missing or non-finite")
        router_gradient_norms.append(float(gradient.float().norm().item()))
    if not all(value > 0 for value in router_gradient_norms):
        raise RuntimeError(f"Slot-router gradients are zero: {router_gradient_norms}")

    candidate.eval()
    with torch.no_grad():
        candidate(inputs[:8])
    diagnostics = slot_state_diagnostics(candidate)
    for row in diagnostics["per_layer"]:
        if (
            row["router_token_std"] <= 0
            or row["router_board_std"] <= 0
            or row["slot_state_relative_difference"] <= 1e-4
            or row["erase_route_change_rms"] <= 0
            or row["write_route_change_rms"] <= 0
            or not 0 < row["minimum_global_slot_mass"] < 1
            or not 0 < row["maximum_global_slot_mass"] < 1
        ):
            raise RuntimeError(f"Slot-state mechanism is inactive: {row}")

    permuted = copy.deepcopy(candidate).eval()
    with torch.no_grad():
        for mixer in _mixers(permuted):
            weight = rearrange(
                mixer.layer.slot_router.weight,
                "(h s) d -> h s d",
                h=4,
                s=SLOT_COUNT,
            )
            mixer.layer.slot_router.weight.copy_(
                rearrange(weight.flip(1), "h s d -> (h s) d")
            )
        permuted_logits = permuted(inputs[:8])
        candidate_small_logits = candidate(inputs[:8])
    slot_permutation_max_diff = finite_max_abs_difference(
        candidate_small_logits,
        permuted_logits,
        "slot permutation",
    )
    if slot_permutation_max_diff != 0.0:
        raise RuntimeError(
            f"Slot permutation changed output: {slot_permutation_max_diff}"
        )

    gate = torch.linspace(0.01, 0.99, 32, device="cuda").view(1, 1, 1, 32)
    mass = torch.tensor([0.0, 2.0], device="cuda").view(1, 1, 1, 2)
    routed = _route_gate(gate, mass)
    unit = _route_gate(gate, torch.ones_like(mass))
    gate_unit_identity_max_diff = finite_max_abs_difference(
        unit[:, :, :, 0], gate, "unit routed gate"
    )
    if (
        gate_unit_identity_max_diff != 0.0
        or routed.min().item() < 0
        or routed.max().item() > 1
    ):
        raise RuntimeError("Bounded routed-gate contract failed")

    result = {
        "status": "passed",
        "gpu": {"name": device.name, "uuid": device_uuid},
        "pinned_fla_sha": PINNED_FLA_SHA,
        "zoology_sha": EXPECTED_ZOOLOGY_SHA,
        "data_hashes": hashes,
        "parameter_counts": counts,
        "parameter_delta": EXPECTED_PARAMETER_DELTA,
        "parent_init_max_diff": parent_init_max_diff,
        "short_convolution_counts": convolution_counts,
        "uniform_output_max_diff": uniform_output_max_diff,
        "incoming_state_identity": incoming_results,
        "opened_output_relative_rms": opened_output_relative_rms,
        "official_backward_count": official_backward_count,
        "router_gradient_norms": router_gradient_norms,
        "slot_state_diagnostics": diagnostics,
        "slot_permutation_max_diff": slot_permutation_max_diff,
        "gate_unit_identity_max_diff": gate_unit_identity_max_diff,
        "routed_gate_min": float(routed.min().item()),
        "routed_gate_max": float(routed.max().item()),
        "state_values_per_layer": 8_192,
        "official_scans_per_layer": 1,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
