from __future__ import annotations

import argparse
import copy
import hashlib
import importlib
import inspect
import json
import math
import os
import subprocess
from pathlib import Path

import torch
from fla.layers.gdn2 import GatedDeltaNet2
from zoology.data.utils import prepare_data
from zoology.utils import set_determinism

from experiments.zoology_mqar.gdn2_futureseed import PINNED_FLA_SHA
from experiments.zoology_mqar.gdn2_sparse_delta_slots import (
    EXPECTED_NEW_PARAMETERS,
    MAIN_STATE_VALUES,
    SLOT_COUNT,
    SPARSE_STATE_VALUES,
    ZoologySparseDeltaSlotFutureSeedMixer,
    load_matched_parent_state,
    parent_parameter_hash,
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


EXPECTED_NATIVE_PARAMETERS = 661_584
EXPECTED_GSA_OPS_SHA256 = (
    "84ac3394e447ae5d615eb03201dd27e2df68b502d4e4f6bb411546ffea331d72"
)
EXPECTED_INIT_SHA256 = (
    "7402e46c48cbd47070d65262d1a1b62a32ac55a4d644a6fb9644b96b0914850f"
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _mixers(model: torch.nn.Module) -> list[ZoologySparseDeltaSlotFutureSeedMixer]:
    mixers = [block.sequence_mixer for block in model.backbone.layers]
    if len(mixers) != 2 or not all(
        isinstance(mixer, ZoologySparseDeltaSlotFutureSeedMixer)
        for mixer in mixers
    ):
        raise RuntimeError("Expected exactly two sparse-delta-slot mixers")
    return mixers


def _gradient_abs_max(parameter: torch.nn.Parameter, label: str) -> float:
    if parameter.grad is None:
        raise RuntimeError(f"Missing gradient: {label}")
    gradient = parameter.grad.float()
    if not torch.isfinite(gradient).all():
        raise RuntimeError(f"Non-finite gradient: {label}")
    value = float(gradient.abs().max().item())
    if value <= 0:
        raise RuntimeError(f"Zero gradient: {label}")
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
        raise RuntimeError("P-GDN3-050 requires one visible CUDA index 0")
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
    gdn2_source = Path(inspect.getfile(GatedDeltaNet2)).resolve()
    gsa_source = Path(
        importlib.import_module("fla.ops.gsa.chunk").__file__
    ).resolve()
    if fla_root not in gdn2_source.parents or fla_root not in gsa_source.parents:
        raise RuntimeError("GDN2 or GSA was imported outside the pinned FLA root")
    if hashlib.sha256(gdn2_source.read_bytes()).hexdigest() != os.environ[
        "FLA_GDN2_SOURCE_SHA256"
    ]:
        raise RuntimeError("Pinned GDN2 layer source hash drifted")
    if python_tree_hash(fla_root / "fla" / "ops" / "gdn2") != os.environ[
        "FLA_GDN2_OPS_SHA256"
    ]:
        raise RuntimeError("Pinned GDN2 ops hash drifted")
    if python_tree_hash(fla_root / "fla" / "ops" / "gsa") != EXPECTED_GSA_OPS_SHA256:
        raise RuntimeError("Pinned GSA ops hash drifted")
    zoology_root = Path(os.environ["ZOOLOGY_ROOT"]).resolve()
    zoology_status = subprocess.run(
        ["git", "-C", str(zoology_root), "status", "--porcelain"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if git_head(zoology_root) != EXPECTED_ZOOLOGY_SHA or zoology_status:
        raise RuntimeError("Zoology source is not exact and clean")
    if _sha256(args.matched_init_path) != EXPECTED_INIT_SHA256:
        raise RuntimeError("Frozen matched initialization hash drifted")

    control_config = build_config(
        arm="future_seed_gdn2",
        sequence_length=1024,
        num_kv_pairs=4,
        max_epochs=10,
        batch_size=32,
    )
    candidate_config = build_config(
        arm="future_seed_sparse_delta_slot_gdn2",
        sequence_length=1024,
        num_kv_pairs=4,
        max_epochs=10,
        batch_size=32,
    )
    train_loader, test_loader = prepare_data(control_config.data)
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
    control = make_model(control_config, "future_seed_gdn2")
    control.load_state_dict(parent_state, strict=True)
    set_determinism(123)
    candidate = make_model(
        candidate_config,
        "future_seed_sparse_delta_slot_gdn2",
    )
    load_matched_parent_state(candidate, parent_state)
    if parent_parameter_hash(candidate) != parameter_hash(control):
        raise RuntimeError("Sparse slots changed the frozen parent initialization")
    parameter_counts = {
        "control": sum(parameter.numel() for parameter in control.parameters()),
        "candidate": sum(parameter.numel() for parameter in candidate.parameters()),
    }
    if (
        parameter_counts["control"] != EXPECTED_NATIVE_PARAMETERS
        or parameter_counts["candidate"] - parameter_counts["control"]
        != EXPECTED_NEW_PARAMETERS
    ):
        raise RuntimeError(f"Unexpected parameter counts: {parameter_counts}")
    mixers = _mixers(candidate)
    if any(
        mixer.state_size() != MAIN_STATE_VALUES + SPARSE_STATE_VALUES
        for mixer in mixers
    ):
        raise RuntimeError("Sparse factor-state size drifted")
    if len([module for module in candidate.modules() if isinstance(module, GatedDeltaNet2)]) != 2:
        raise RuntimeError("Expected exactly two pinned official GDN2 layers")

    control = control.cuda().eval()
    candidate = candidate.cuda().eval()
    inputs, _labels, _slices = next(iter(test_loader))
    inputs = inputs[:2].cuda()
    with torch.no_grad():
        control_logits = control(inputs)
        candidate_logits = candidate(inputs)
    zero_gate_output_difference = finite_max_abs_difference(
        control_logits,
        candidate_logits,
        "zero-gate full-model output",
    )
    if zero_gate_output_difference != 0.0:
        raise RuntimeError("Zero-gate candidate is not bit-exact with the parent")
    zero_state_differences = []
    generator = torch.Generator(device="cuda").manual_seed(50050)
    hidden = torch.randn(
        2,
        73,
        128,
        generator=generator,
        device="cuda",
    )
    incoming = torch.randn(
        2,
        4,
        32,
        32,
        generator=generator,
        device="cuda",
        dtype=torch.bfloat16,
    )
    with torch.no_grad():
        for control_block, candidate_mixer in zip(control.backbone.layers, mixers):
            control_output, control_terminal = (
                control_block.sequence_mixer.forward_with_state(
                    hidden,
                    initial_state=incoming,
                )
            )
            candidate_output, candidate_terminal, _sparse = (
                candidate_mixer.forward_with_sparse_state(
                    hidden,
                    initial_state=incoming,
                )
            )
            zero_state_differences.append(
                max(
                    finite_max_abs_difference(
                        control_output,
                        candidate_output,
                        "nonzero-incoming output identity",
                    ),
                    finite_max_abs_difference(
                        control_terminal,
                        candidate_terminal,
                        "nonzero-incoming state identity",
                    ),
                )
            )
    if max(zero_state_differences) != 0.0:
        raise RuntimeError("Zero gates changed nonzero-incoming parent behavior")

    candidate.train()
    candidate.zero_grad(set_to_none=True)
    logits = candidate(inputs)
    names = backward_names(logits)
    if names.count("ChunkGDN2FunctionBackward") != 2:
        raise RuntimeError(f"Unexpected GDN2 autograd provenance: {names}")
    if names.count("ChunkGSAFunctionBackward") != 2:
        raise RuntimeError(f"Unexpected GSA autograd provenance: {names}")
    logits.float().square().mean().backward()
    zero_gate_gradients = {
        f"layer_{mixer.layer_idx}_local": _gradient_abs_max(
            mixer.layer.local_read_logit,
            f"layer {mixer.layer_idx} local gate",
        )
        for mixer in mixers
    }
    zero_gate_gradients["layer_1_seed"] = _gradient_abs_max(
        mixers[1].layer.seed_read_logit,
        "receiving sparse FutureSeed gate",
    )

    for mixer in mixers:
        mixer.layer.local_read_logit.data.fill_(0.20)
        if mixer.layer.seed_read_logit is not None:
            mixer.layer.seed_read_logit.data.fill_(0.20)
    candidate.zero_grad(set_to_none=True)
    candidate(inputs).float().square().mean().backward()
    opened_anchor_gradients = {
        f"layer_{mixer.layer_idx}": _gradient_abs_max(
            mixer.layer.slot_anchors,
            f"layer {mixer.layer_idx} anchors",
        )
        for mixer in mixers
    }

    wrapper = mixers[0].layer
    raw_key = torch.randn(
        2,
        29,
        4,
        32,
        generator=generator,
        device="cuda",
        dtype=torch.bfloat16,
    )
    with torch.no_grad():
        route, hard, probabilities = wrapper.sparse_route(raw_key)
        original_anchors = wrapper.slot_anchors.detach().clone()
        permutation = torch.tensor([2, 0, 3, 1], device="cuda")
        wrapper.slot_anchors.copy_(original_anchors[permutation])
        permuted_route, _permuted_hard, _permuted_probabilities = (
            wrapper.sparse_route(raw_key[:, :, permutation])
        )
        wrapper.slot_anchors.copy_(original_anchors)
    head_permutation_error = finite_max_abs_difference(
        route[:, :, permutation],
        permuted_route,
        "head permutation equivariance",
    )
    if head_permutation_error != 0.0:
        raise RuntimeError("Sparse routing is not head-permutation equivariant")
    if (
        not torch.equal(hard.sum(dim=-1), torch.ones_like(hard[..., 0]))
        or not torch.isfinite(probabilities).all()
    ):
        raise RuntimeError("Top-1 routing contract failed")

    diagnostics = []
    for mixer in mixers:
        row = mixer.layer.last_diagnostics
        if not row:
            raise RuntimeError("Sparse diagnostics were not populated")
        diagnostics.append({name: float(value.item()) for name, value in row.items()})
    finite_state_geometry = all(
        math.isfinite(row[name]) and row[name] > 0
        for row in diagnostics
        for name in (
            "sparse_key_state_rms",
            "sparse_value_state_rms",
            "sparse_dense_rms",
        )
    )
    bounded_transition = all(
        0.0 <= row["slot_s_min"] <= row["slot_s_max"] <= 1.0
        and row["slot_g_max"] <= 0.0
        for row in diagnostics
    )
    if not finite_state_geometry or not bounded_transition:
        raise RuntimeError("Sparse slot state geometry is invalid")

    mechanism_source = Path(inspect.getfile(type(wrapper))).read_text()
    if "fallback" in mechanism_source.lower():
        raise RuntimeError("Mechanism source contains a fallback path")
    result = {
        "status": "passed",
        "gpu": {
            "index": 0,
            "name": device.name,
            "uuid": device_uuid,
        },
        "provenance": {
            "fla_sha": PINNED_FLA_SHA,
            "gdn2_source": str(gdn2_source),
            "gsa_source": str(gsa_source),
            "gsa_ops_sha256": EXPECTED_GSA_OPS_SHA256,
            "zoology_sha": EXPECTED_ZOOLOGY_SHA,
            "backward_counts": {
                "ChunkGDN2FunctionBackward": names.count("ChunkGDN2FunctionBackward"),
                "ChunkGSAFunctionBackward": names.count("ChunkGSAFunctionBackward"),
            },
        },
        "data_hashes": data_hashes,
        "matched_parent_parameter_hash": parameter_hash(control),
        "parameter_counts": parameter_counts,
        "parameter_delta": parameter_counts["candidate"] - parameter_counts["control"],
        "state_values_per_layer": {
            "main": MAIN_STATE_VALUES,
            "sparse": SPARSE_STATE_VALUES,
            "total": MAIN_STATE_VALUES + SPARSE_STATE_VALUES,
        },
        "zero_gate_output_max_difference": zero_gate_output_difference,
        "zero_gate_nonzero_incoming_max_differences": zero_state_differences,
        "zero_gate_gradients": zero_gate_gradients,
        "opened_anchor_gradients": opened_anchor_gradients,
        "head_permutation_error": head_permutation_error,
        "slot_count": SLOT_COUNT,
        "routing_probability_sum_error": float(
            (probabilities.sum(dim=-1) - 1.0).abs().max().item()
        ),
        "finite_state_geometry": finite_state_geometry,
        "bounded_transition": bounded_transition,
        "diagnostics": diagnostics,
        "no_fallback": True,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
