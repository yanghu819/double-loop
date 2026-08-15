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

from experiments.zoology_mqar.binding_certificate_futureseed import (
    EXPECTED_CERTIFICATE_STATE_VALUES,
    EXPECTED_MAIN_STATE_VALUES,
    EXPECTED_NEW_PARAMETERS,
    BindingCertificateBackbone,
    BindingCertificateGDN2,
    ZoologyBindingCertificateFutureSeedMixer,
    binding_certificate_diagnostics,
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
    EXPECTED_ZOOLOGY_SHA,
    backward_names,
    finite_max_abs_difference,
    git_head,
    normalized_uuid,
    python_tree_hash,
)


EXPECTED_NATIVE_PARAMETERS = 661_584
EXPECTED_CANDIDATE_PARAMETERS = EXPECTED_NATIVE_PARAMETERS + EXPECTED_NEW_PARAMETERS
PARENT_GRADIENT_ATOL = 0.125
PARENT_GRADIENT_REL_RMS_MAX = 0.01
PARENT_GRADIENT_NOISE_MULTIPLIER = 2.0


def _mapped_candidate_parameters(
    model: torch.nn.Module,
) -> dict[str, torch.nn.Parameter]:
    mapped = {}
    for name, parameter in model.named_parameters():
        if name.endswith(".certificate_read_logit"):
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
    relative_rms = (difference_energy / max(reference_energy, 1e-24)) ** 0.5
    return {
        "max_abs": max_difference,
        "relative_rms": relative_rms,
        "compared_parameter_count": compared_count,
        "absent_parameter_count": absent_count,
    }


@torch.no_grad()
def _head_permutation_contract() -> dict[str, float]:
    generator = torch.Generator(device="cuda").manual_seed(47047)
    q = torch.randn(
        2, 128, 4, 32, generator=generator, device="cuda"
    ).to(torch.bfloat16)
    k = torch.randn(
        2, 128, 4, 32, generator=generator, device="cuda"
    ).to(torch.bfloat16)
    v = torch.randn(
        2, 128, 4, 32, generator=generator, device="cuda"
    ).to(torch.bfloat16)
    g = -torch.rand(2, 128, 4, 32, generator=generator, device="cuda")
    b = torch.sigmoid(
        torch.randn(2, 128, 4, 32, generator=generator, device="cuda")
    ).to(torch.bfloat16)
    w = torch.sigmoid(
        torch.randn(2, 128, 4, 32, generator=generator, device="cuda")
    ).to(torch.bfloat16)
    permutation = torch.tensor([2, 0, 3, 1], device="cuda")
    inverse = torch.argsort(permutation)
    output, state = chunk_gdn2(
        q=q,
        k=k,
        v=v,
        g=g,
        b=b,
        w=w,
        initial_state=None,
        output_final_state=True,
        use_qk_l2norm_in_kernel=True,
    )
    permuted_output, permuted_state = chunk_gdn2(
        q=q.index_select(2, permutation),
        k=k.index_select(2, permutation),
        v=v.index_select(2, permutation),
        g=g.index_select(2, permutation),
        b=b.index_select(2, permutation),
        w=w.index_select(2, permutation),
        initial_state=None,
        output_final_state=True,
        use_qk_l2norm_in_kernel=True,
    )
    output_error = float(
        (output - permuted_output.index_select(2, inverse)).float().abs().max().item()
    )
    state_error = float(
        (state - permuted_state.index_select(1, inverse)).float().abs().max().item()
    )
    if output_error > 3e-3 or state_error > 3e-3:
        raise RuntimeError(
            f"Head permutation contract failed: {output_error}/{state_error}"
        )
    return {"output_max_error": output_error, "state_max_error": state_error}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-gpu-name", required=True)
    parser.add_argument("--expected-gpu-uuid", required=True)
    args = parser.parse_args()

    if os.environ.get("CUDA_VISIBLE_DEVICES") != "0":
        raise RuntimeError("CUDA_VISIBLE_DEVICES must be exactly 0")
    if torch.cuda.device_count() != 1 or torch.cuda.current_device() != 0:
        raise RuntimeError("P-GDN3-047 requires exactly CUDA index 0")
    device = torch.cuda.get_device_properties(0)
    device_uuid = str(getattr(device, "uuid", "unavailable"))
    if (
        device.name != args.expected_gpu_name
        or normalized_uuid(device_uuid) != normalized_uuid(args.expected_gpu_uuid)
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
        "experiments.zoology_mqar.binding_certificate_futureseed"
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
        arm="future_seed_binding_certificate_gdn2",
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
        "future_seed_binding_certificate_gdn2",
    )
    load_matched_parent_state(candidate, native.state_dict())
    set_determinism(123)
    native_replay = make_model(native_config, "future_seed_gdn2")
    native_replay.load_state_dict(native.state_dict(), strict=True)
    if not isinstance(candidate.backbone, BindingCertificateBackbone):
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
        if type(mixer) is not ZoologyBindingCertificateFutureSeedMixer:
            raise RuntimeError(f"Layer {layer_index} mixer changed")
        if type(mixer.layer) is not BindingCertificateGDN2:
            raise RuntimeError(f"Layer {layer_index} wrapper changed")
        if type(mixer.layer.base) is not GatedDeltaNet2:
            raise RuntimeError(f"Layer {layer_index} carrier changed")
        if mixer.state_size() != EXPECTED_MAIN_STATE_VALUES:
            raise RuntimeError(f"Layer {layer_index} main state size changed")
        if mixer.layer.certificate_read_logit.numel() != 4:
            raise RuntimeError(f"Layer {layer_index} gate shape changed")
        if mixer.layer.certificate_read_logit.detach().abs().max().item() != 0:
            raise RuntimeError(f"Layer {layer_index} gate is not zero initialized")
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
                "main_state_values": EXPECTED_MAIN_STATE_VALUES,
                "certificate_state_values": EXPECTED_CERTIFICATE_STATE_VALUES,
                "logical_scans": 2,
                "convolution_backends": convolutions,
            }
        )
        mixers.append(mixer)
    if len(mixers) != 2:
        raise RuntimeError("Expected two binding-certificate mixers")

    inputs, targets, _slices = next(iter(test_loader))
    inputs = inputs.cuda()
    targets = targets.cuda()
    native = native.cuda().eval()
    native_replay = native_replay.cuda().eval()
    candidate = candidate.cuda().eval()
    with torch.no_grad():
        native_logits = native(inputs)
        candidate_logits = candidate(inputs)
    identity_output_max_diff = finite_max_abs_difference(
        native_logits,
        candidate_logits,
        "zero-certificate full output",
    )
    if identity_output_max_diff != 0:
        raise RuntimeError("Zero certificate gate changed full output")

    generator = torch.Generator(device="cuda").manual_seed(52047)
    incoming_rows = []
    certificate_dependency_rows = []
    for layer_index in range(2):
        native_mixer = native.backbone.layers[layer_index].sequence_mixer
        candidate_mixer = candidate.backbone.layers[layer_index].sequence_mixer
        hidden = torch.randn(2, 128, 128, generator=generator, device="cuda")
        payload = F.normalize(
            torch.randn(2, 128, 128, generator=generator, device="cuda"),
            dim=-1,
        )
        incoming = torch.randn(
            2, 4, 32, 32, generator=generator, device="cuda"
        )
        candidate_mixer.layer.set_capture(True)
        with torch.no_grad():
            native_output, native_state = native_mixer.forward_with_state(
                hidden,
                initial_state=incoming,
            )
            candidate_output, candidate_state = (
                candidate_mixer.forward_with_certificate_state(
                    hidden,
                    certificate_payload=payload,
                    initial_state=incoming,
                )
            )
            first_certificate_state = candidate_mixer.layer._captured[
                "certificate_state"
            ].clone()
            candidate_mixer.forward_with_certificate_state(
                hidden,
                certificate_payload=payload.roll(1, dims=1),
                initial_state=incoming,
            )
            second_certificate_state = candidate_mixer.layer._captured[
                "certificate_state"
            ].clone()
        candidate_mixer.layer.set_capture(False)
        output_diff = finite_max_abs_difference(
            native_output,
            candidate_output,
            f"layer-{layer_index} incoming output",
        )
        state_diff = finite_max_abs_difference(
            native_state,
            candidate_state,
            f"layer-{layer_index} incoming main state",
        )
        if output_diff != 0 or state_diff != 0:
            raise RuntimeError("Zero certificate gate changed nonzero-state path")
        dependency = float(
            (first_certificate_state - second_certificate_state)
            .float()
            .square()
            .mean()
            .sqrt()
            .item()
        )
        if dependency <= 1e-4:
            raise RuntimeError("Certificate state ignored token payload order")
        incoming_rows.append(
            {
                "layer": layer_index,
                "output_max_diff": output_diff,
                "main_state_max_diff": state_diff,
            }
        )
        certificate_dependency_rows.append(
            {"layer": layer_index, "payload_shuffle_state_rms": dependency}
        )

    native.train()
    native_replay.train()
    candidate.train()
    native.zero_grad(set_to_none=True)
    native_replay.zero_grad(set_to_none=True)
    candidate.zero_grad(set_to_none=True)
    native_loss = F.cross_entropy(
        native(inputs[:4]).flatten(0, 1), targets[:4].flatten()
    )
    native_replay_loss = F.cross_entropy(
        native_replay(inputs[:4]).flatten(0, 1), targets[:4].flatten()
    )
    candidate_loss = F.cross_entropy(
        candidate(inputs[:4]).flatten(0, 1), targets[:4].flatten()
    )
    native_loss.backward()
    native_replay_loss.backward()
    candidate_loss.backward()
    backward_graph = backward_names(candidate_loss)
    official_backward_count = sum(
        name == "ChunkGDN2FunctionBackward" for name in backward_graph
    )
    if official_backward_count != 4:
        raise RuntimeError(
            f"Expected four official backward nodes, got {official_backward_count}"
        )
    gate_gradient_rows = []
    for index, mixer in enumerate(mixers):
        rms = _finite_nonzero_gradient(
            mixer.layer.certificate_read_logit,
            f"layer-{index} certificate gate",
        )
        gradient_abs = (
            mixer.layer.certificate_read_logit.grad.detach().float().abs().flatten()
        )
        if (gradient_abs <= 0).any():
            raise RuntimeError(f"Layer {index} has an inactive gate head gradient")
        gate_gradient_rows.append(
            {
                "layer": index,
                "rms": rms,
                "min_abs": float(gradient_abs.min().item()),
                "max_abs": float(gradient_abs.max().item()),
            }
        )
    candidate_parameters = _mapped_candidate_parameters(candidate)
    native_parameters = dict(native.named_parameters())
    replay_parameters = dict(native_replay.named_parameters())
    native_replay_gradient_noise = _gradient_difference_summary(
        native_parameters,
        replay_parameters,
        label="native replay",
    )
    candidate_parent_gradient_difference = _gradient_difference_summary(
        native_parameters,
        candidate_parameters,
        label="binding-certificate parent",
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
            "Zero certificate gate changed parent gradients beyond calibrated "
            f"BF16 noise: candidate={candidate_parent_gradient_difference} "
            f"native_replay={native_replay_gradient_noise} "
            f"limits={calibrated_max_abs}/{calibrated_relative_rms}"
        )

    with torch.no_grad():
        for mixer in mixers:
            mixer.layer.certificate_read_logit.fill_(0.20)
    with torch.no_grad():
        active_logits = candidate.eval()(inputs[:8])
    active_output_relative_rms = float(
        (
            (active_logits - native_logits[:8]).float().square().mean().sqrt()
            / native_logits[:8].float().square().mean().sqrt().clamp_min(1e-8)
        ).item()
    )
    if active_output_relative_rms <= 1e-4:
        raise RuntimeError("Opened certificate gate did not affect model output")
    active_diagnostics = binding_certificate_diagnostics(candidate, inputs[:8])
    if active_diagnostics["active_layers"] != 2:
        raise RuntimeError("Not all certificate layers are active")
    if active_diagnostics["active_gate_heads"] != 8:
        raise RuntimeError("Not all certificate head gates are active")
    if active_diagnostics["repeated_token_groups"] <= 0:
        raise RuntimeError("No repeated-token identity case was checked")
    if active_diagnostics["repeated_token_max_error"] > 1e-6:
        raise RuntimeError("Same token received position-dependent payload")
    if not all(
        row["certificate_output_rms"] > 0
        and row["certificate_output_token_std"] > 1e-4
        and row["certificate_state_rms"] > 0
        and row["certificate_state_board_std"] > 0
        for row in active_diagnostics["per_layer"]
    ):
        raise RuntimeError("Active certificate state is collapsed")
    head_permutation = _head_permutation_contract()

    result = {
        "status": "passed",
        "plan": "P-GDN3-047",
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
        "main_state_values_per_layer": EXPECTED_MAIN_STATE_VALUES,
        "certificate_state_values_per_layer": EXPECTED_CERTIFICATE_STATE_VALUES,
        "parent_parameter_hash": native_hash,
        "parent_init_max_diff": parent_init_max_diff,
        "identity_output_max_diff": identity_output_max_diff,
        "incoming_state_identity": incoming_rows,
        "certificate_payload_dependency": certificate_dependency_rows,
        "official_backward_count": official_backward_count,
        "gate_gradients": gate_gradient_rows,
        "native_replay_gradient_noise": native_replay_gradient_noise,
        "candidate_parent_gradient_difference": candidate_parent_gradient_difference,
        "parent_gradient_atol": PARENT_GRADIENT_ATOL,
        "parent_gradient_relative_rms_max": PARENT_GRADIENT_REL_RMS_MAX,
        "parent_gradient_noise_multiplier": PARENT_GRADIENT_NOISE_MULTIPLIER,
        "calibrated_parent_gradient_max_abs": calibrated_max_abs,
        "calibrated_parent_gradient_relative_rms": calibrated_relative_rms,
        "active_output_relative_rms": active_output_relative_rms,
        "active_diagnostics": active_diagnostics,
        "head_permutation": head_permutation,
        "provenance": provenance,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
