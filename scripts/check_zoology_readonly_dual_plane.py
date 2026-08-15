from __future__ import annotations

import argparse
import hashlib
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
from experiments.zoology_mqar.length_scaling import (
    build_config,
    dataset_hash,
    make_model,
    parameter_hash,
)
from experiments.zoology_mqar.readonly_dual_plane_futureseed import (
    EXPECTED_PARAMETER_DELTA,
    EXPECTED_RECURRENT_STATE_DELTA,
    READONLY_SIDE_STATE_VALUES,
    ZoologyReadOnlyDualPlaneFutureSeedMixer,
    load_matched_parent_state,
    parent_parameter_hash,
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
EXPECTED_STATE_VALUES = 4 * 32 * 32
EXPECTED_INIT_SHA256 = (
    "7402e46c48cbd47070d65262d1a1b62a32ac55a4d644a6fb9644b96b0914850f"
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


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


def _mixers(model: torch.nn.Module) -> list[ZoologyReadOnlyDualPlaneFutureSeedMixer]:
    mixers = [block.sequence_mixer for block in model.backbone.layers]
    if len(mixers) != 2 or not all(
        type(mixer) is ZoologyReadOnlyDualPlaneFutureSeedMixer
        for mixer in mixers
    ):
        raise RuntimeError("Expected exactly two read-only dual-plane mixers")
    return mixers


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
        raise RuntimeError("P-FS2-012 requires one visible CUDA index 0")
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
    if fla_root not in gdn2_source.parents:
        raise RuntimeError("GDN2 was imported outside the pinned FLA root")
    if hashlib.sha256(gdn2_source.read_bytes()).hexdigest() != os.environ[
        "FLA_GDN2_SOURCE_SHA256"
    ]:
        raise RuntimeError("Pinned GDN2 layer source hash drifted")
    if python_tree_hash(fla_root / "fla" / "ops" / "gdn2") != os.environ[
        "FLA_GDN2_OPS_SHA256"
    ]:
        raise RuntimeError("Pinned GDN2 ops hash drifted")
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

    native_config = build_config(
        arm="future_seed_gdn2",
        sequence_length=1024,
        num_kv_pairs=4,
        max_epochs=10,
        batch_size=32,
    )
    causal_config = build_config(
        arm="causal_gdn2",
        sequence_length=1024,
        num_kv_pairs=4,
        max_epochs=10,
        batch_size=32,
    )
    candidate_config = build_config(
        arm="future_seed_readonly_dual_plane_gdn2",
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
    native.load_state_dict(parent_state, strict=True)
    set_determinism(123)
    causal = make_model(causal_config, "causal_gdn2")
    causal.load_state_dict(parent_state, strict=True)
    set_determinism(123)
    candidate = make_model(
        candidate_config,
        "future_seed_readonly_dual_plane_gdn2",
    )
    load_matched_parent_state(candidate, parent_state)
    native_hash = parameter_hash(native)
    if (
        parent_parameter_hash(candidate) != native_hash
        or parameter_hash(candidate) != native_hash
        or parameter_hash(causal) != native_hash
    ):
        raise RuntimeError("Dual-plane model changed the frozen initialization")
    parameter_counts = {
        "native": sum(parameter.numel() for parameter in native.parameters()),
        "causal": sum(parameter.numel() for parameter in causal.parameters()),
        "candidate": sum(parameter.numel() for parameter in candidate.parameters()),
    }
    if set(parameter_counts.values()) != {EXPECTED_NATIVE_PARAMETERS}:
        raise RuntimeError(f"Unexpected parameter counts: {parameter_counts}")
    if EXPECTED_PARAMETER_DELTA != 0 or EXPECTED_RECURRENT_STATE_DELTA != 0:
        raise RuntimeError("P-FS2-012 delta constants drifted")
    mixers = _mixers(candidate)
    if any(mixer.state_size() != EXPECTED_STATE_VALUES for mixer in mixers):
        raise RuntimeError("Native recurrent state size drifted")
    if len(
        [module for module in candidate.modules() if isinstance(module, GatedDeltaNet2)]
    ) != 2:
        raise RuntimeError("Expected exactly two pinned official GDN2 layers")

    del native
    causal = causal.cuda().eval()
    candidate = candidate.cuda().eval()
    inputs, _labels, _slices = next(iter(test_loader))
    inputs = inputs[:2].cuda()
    candidate.set_readonly_enabled(False)
    with torch.no_grad():
        causal_logits = causal(inputs)
        edge_off_logits = candidate(inputs)
    edge_off_parent_difference = finite_max_abs_difference(
        causal_logits,
        edge_off_logits,
        "edge-off causal parent identity",
    )
    if edge_off_parent_difference != 0.0:
        raise RuntimeError("Edge-off dual-plane path is not bit-exact causal GDN2")

    candidate.set_readonly_enabled(True)
    candidate.train()
    candidate.zero_grad(set_to_none=True)
    logits = candidate(inputs)
    names = backward_names(logits)
    if names.count("ChunkGDN2FunctionBackward") != 2:
        raise RuntimeError(f"Unexpected GDN2 autograd provenance: {names}")
    logits.float().square().mean().backward()
    gradient_checks = {
        "receiving_future_seed_gate": _gradient_abs_max(
            mixers[1].future_seed_logit,
            "receiving FutureSeed gate",
        ),
        "producer_key_projection": _gradient_abs_max(
            mixers[0].layer.k_proj.weight,
            "producer key projection",
        ),
        "producer_value_projection": _gradient_abs_max(
            mixers[0].layer.v_proj.weight,
            "producer value projection",
        ),
        "receiver_query_projection": _gradient_abs_max(
            mixers[1].layer.q_proj.weight,
            "receiver query projection",
        ),
    }

    generator = torch.Generator(device="cuda").manual_seed(52012)
    query = torch.randn(
        2,
        37,
        4,
        32,
        generator=generator,
        device="cuda",
        dtype=torch.bfloat16,
    )
    readonly_state = torch.randn(
        2,
        4,
        32,
        32,
        generator=generator,
        device="cuda",
        dtype=torch.float32,
    )
    state_before = readonly_state.clone()
    with torch.no_grad():
        read = mixers[1].readonly_read(query, readonly_state)
        query_float = query.float()
        normalized_query = query_float * torch.rsqrt(
            query_float.square().sum(dim=-1, keepdim=True) + 1e-6
        )
        normalized_query = normalized_query.to(dtype=query.dtype).float()
        manual = torch.einsum(
            "bthk,bhkv->bthv",
            normalized_query * (32.0**-0.5),
            readonly_state,
        )
    direct_formula_error = finite_max_abs_difference(
        read,
        manual,
        "direct receiver read formula",
    )
    readonly_mutation_error = finite_max_abs_difference(
        readonly_state,
        state_before,
        "read-only state mutation",
    )
    if direct_formula_error != 0.0 or readonly_mutation_error != 0.0:
        raise RuntimeError("Read-only FutureSeed formula or immutability failed")

    head_permutation = torch.tensor([2, 0, 3, 1], device="cuda")
    with torch.no_grad():
        permuted_read = mixers[1].readonly_read(
            query[:, :, head_permutation],
            readonly_state[:, head_permutation],
        )
    head_permutation_error = finite_max_abs_difference(
        read[:, :, head_permutation],
        permuted_read,
        "head permutation equivariance",
    )
    if head_permutation_error != 0.0:
        raise RuntimeError("Read-only FutureSeed is not head-equivariant")

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
    )
    incoming_before = incoming.clone()
    mixers[1].set_readonly_enabled(True)
    with torch.no_grad():
        output, terminal = mixers[1].forward_with_readonly_state(
            hidden,
            readonly_state=incoming,
        )
        shuffled_output, _shuffled_terminal = (
            mixers[1].forward_with_readonly_state(
                hidden,
                readonly_state=incoming.flip(0),
            )
        )
    full_state_mutation_error = finite_max_abs_difference(
        incoming,
        incoming_before,
        "full receiver read-only state mutation",
    )
    shuffle_dependency = float(
        (output.float() - shuffled_output.float()).square().mean().sqrt().item()
    )
    if full_state_mutation_error != 0.0 or shuffle_dependency <= 1e-4:
        raise RuntimeError("Dual-plane state dependency or immutability failed")
    if not torch.isfinite(terminal).all() or float(terminal.float().square().mean().sqrt()) <= 0:
        raise RuntimeError("Live official terminal state is invalid")

    mechanism_source = Path(inspect.getfile(type(mixers[0]))).read_text()
    if "fallback" in mechanism_source.lower():
        raise RuntimeError("Mechanism source contains a fallback path")
    result = {
        "status": "passed",
        "gpu": {"index": 0, "name": device.name, "uuid": device_uuid},
        "provenance": {
            "fla_sha": PINNED_FLA_SHA,
            "gdn2_source": str(gdn2_source),
            "zoology_sha": EXPECTED_ZOOLOGY_SHA,
            "ChunkGDN2FunctionBackward": names.count(
                "ChunkGDN2FunctionBackward"
            ),
        },
        "data_hashes": data_hashes,
        "matched_parent_parameter_hash": native_hash,
        "parameter_counts": parameter_counts,
        "parameter_delta": EXPECTED_PARAMETER_DELTA,
        "recurrent_state_values_per_layer": EXPECTED_STATE_VALUES,
        "recurrent_state_delta": EXPECTED_RECURRENT_STATE_DELTA,
        "retained_readonly_side_state_values": READONLY_SIDE_STATE_VALUES,
        "official_scans_per_layer": 1,
        "new_official_scans_per_layer": 0,
        "edge_off_parent_max_difference": edge_off_parent_difference,
        "gradient_abs_max": gradient_checks,
        "direct_formula_max_error": direct_formula_error,
        "readonly_mutation_max_error": readonly_mutation_error,
        "full_state_mutation_max_error": full_state_mutation_error,
        "head_permutation_error": head_permutation_error,
        "state_shuffle_output_rms": shuffle_dependency,
        "live_terminal_rms": float(terminal.float().square().mean().sqrt().item()),
        "all_values_finite": all(
            math.isfinite(value)
            for value in (
                shuffle_dependency,
                float(terminal.float().square().mean().sqrt().item()),
                *gradient_checks.values(),
            )
        ),
        "no_fallback": True,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
