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
import torch.nn.functional as F
from fla.layers.gdn2 import GatedDeltaNet2
from zoology.data.utils import prepare_data
from zoology.utils import set_determinism

from experiments.zoology_mqar.gdn2_block_gram import (
    BlockCausalGramConditioner,
    ZoologyBlockGramGDN2FutureSeedMixer,
    parent_parameter_hash,
)
from experiments.zoology_mqar.gdn2_futureseed import PINNED_FLA_SHA
from experiments.zoology_mqar.gdn2_log_spd import (
    ZoologyLogSPDGDN2FutureSeedMixer,
    capture_chunk_addresses,
)
from experiments.zoology_mqar.length_scaling import (
    build_config,
    dataset_hash,
    make_model,
    parameter_hash,
)


EXPECTED_ZOOLOGY_SHA = "1ad20d193b6113cae1e8f3c655c300d7b4b3f4bb"
EXPECTED_TRAIN_HASH = "647c64ece84984a23dfd817c4f277ea83840dbec57cc18bb6c9bf9eda7cc9a68"
EXPECTED_TEST_HASH = "4a8237ba8fe19aaff0d1d72de7b7f6505eaab59cd091442c2f463df34cce278f"
EXPECTED_PARAMETER_DELTA = 8
EXPECTED_UNUSED_PARENT_GRADIENTS = {
    "backbone.layers.0.sequence_mixer.future_seed_logit",
}


def normalized_uuid(value: str) -> str:
    return value.removeprefix("GPU-").lower()


def max_diff(left: torch.Tensor, right: torch.Tensor, label: str) -> float:
    difference = (left - right).abs()
    if not torch.isfinite(difference).all():
        raise RuntimeError(f"Non-finite {label} difference")
    return float(difference.max().item())


def tree_hash(path: Path) -> str:
    digest = hashlib.sha256()
    for source in sorted(path.rglob("*.py")):
        digest.update(str(source.relative_to(path)).encode())
        digest.update(source.read_bytes())
    return digest.hexdigest()


def backward_names(tensor: torch.Tensor) -> list[str]:
    names, seen, stack = [], set(), [tensor.grad_fn]
    while stack:
        function = stack.pop()
        if function is None or function in seen:
            continue
        seen.add(function)
        names.append(type(function).__name__)
        stack.extend(next_function for next_function, _index in function.next_functions)
    return names


def parent_parameters(model: torch.nn.Module) -> dict[str, torch.nn.Parameter]:
    return {
        name: parameter
        for name, parameter in model.named_parameters()
        if not name.endswith("layer.block_gram.raw_strength")
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-gpu-name", required=True)
    parser.add_argument("--expected-gpu-uuid", required=True)
    args = parser.parse_args()

    if torch.cuda.device_count() != 1:
        raise RuntimeError(f"Expected one visible GPU, got {torch.cuda.device_count()}")
    device = torch.cuda.get_device_properties(0)
    device_uuid = str(getattr(device, "uuid", "unavailable"))
    if device.name != args.expected_gpu_name or normalized_uuid(device_uuid) != normalized_uuid(
        args.expected_gpu_uuid
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
    source_hash = hashlib.sha256(gdn2_source.read_bytes()).hexdigest()
    ops_hash = tree_hash(fla_root / "fla" / "ops" / "gdn2")
    if fla_root not in gdn2_source.parents:
        raise RuntimeError(f"Unexpected GDN2 source: {gdn2_source}")
    if source_hash != os.environ["FLA_GDN2_SOURCE_SHA256"]:
        raise RuntimeError("Official GDN2 layer source drifted")
    if ops_hash != os.environ["FLA_GDN2_OPS_SHA256"]:
        raise RuntimeError("Official GDN2 ops source drifted")
    zoology_root = Path(os.environ["ZOOLOGY_ROOT"]).resolve()
    zoology_sha = subprocess.check_output(
        ["git", "-C", str(zoology_root), "rev-parse", "HEAD"], text=True
    ).strip()
    if zoology_sha != EXPECTED_ZOOLOGY_SHA:
        raise RuntimeError(f"Unexpected Zoology SHA: {zoology_sha}")

    control_config = build_config(
        arm="future_seed_gdn2_log_spd",
        sequence_length=1024,
        num_kv_pairs=4,
        max_epochs=10,
        batch_size=32,
    )
    candidate_config = build_config(
        arm="future_seed_gdn2_log_spd_block_gram",
        sequence_length=1024,
        num_kv_pairs=4,
        max_epochs=10,
        batch_size=32,
    )
    train_loader, test_loader = prepare_data(control_config.data)
    data_hashes = {"train": dataset_hash(train_loader), "test": dataset_hash(test_loader)}
    if data_hashes != {"train": EXPECTED_TRAIN_HASH, "test": EXPECTED_TEST_HASH}:
        raise RuntimeError(f"Directional MQAR data drifted: {data_hashes}")

    set_determinism(123)
    control = make_model(control_config, "future_seed_gdn2_log_spd")
    set_determinism(123)
    candidate = make_model(candidate_config, "future_seed_gdn2_log_spd_block_gram")
    if parent_parameter_hash(candidate) != parameter_hash(control):
        raise RuntimeError("Block-Gram insertion changed the P020 parent initialization")
    counts = {
        "control": sum(parameter.numel() for parameter in control.parameters()),
        "candidate": sum(parameter.numel() for parameter in candidate.parameters()),
    }
    if counts["candidate"] - counts["control"] != EXPECTED_PARAMETER_DELTA:
        raise RuntimeError(f"Unexpected parameter delta: {counts}")
    control_parent = dict(control.named_parameters())
    candidate_parent = parent_parameters(candidate)
    if set(control_parent) != set(candidate_parent):
        raise RuntimeError("Candidate parent parameter names differ from P020")
    parent_init_diff = max(
        max_diff(control_parent[name], candidate_parent[name], f"init {name}")
        for name in control_parent
    )
    if parent_init_diff != 0.0:
        raise RuntimeError(f"Candidate parent tensors changed: {parent_init_diff}")

    inputs, targets, _slices = next(iter(test_loader))
    inputs, targets = inputs.cuda(), targets.cuda()
    control, candidate = control.cuda(), candidate.cuda()
    control.eval()
    candidate.eval()
    with torch.no_grad(), capture_chunk_addresses() as recorder:
        candidate_logits = candidate(inputs)
    with torch.no_grad():
        control_logits = control(inputs)
    identity_output_diff = max_diff(control_logits, candidate_logits, "zero output")
    if identity_output_diff != 0.0 or len(recorder.records) != 2:
        raise RuntimeError(
            f"Zero identity/call count failed: diff={identity_output_diff} calls={len(recorder.records)}"
        )

    generator = torch.Generator(device="cuda").manual_seed(52026)
    incoming_diffs = []
    for layer_index in range(2):
        control_mixer = control.backbone.layers[layer_index].sequence_mixer
        candidate_mixer = candidate.backbone.layers[layer_index].sequence_mixer
        if not isinstance(control_mixer, ZoologyLogSPDGDN2FutureSeedMixer) or not isinstance(
            candidate_mixer, ZoologyBlockGramGDN2FutureSeedMixer
        ):
            raise RuntimeError("Mixer types changed")
        hidden = torch.randn(2, 128, 128, generator=generator, device="cuda")
        incoming = torch.randn(2, 4, 32, 32, generator=generator, device="cuda")
        with torch.no_grad():
            control_output, control_state = control_mixer.forward_with_state(
                hidden, initial_state=incoming
            )
            candidate_output, candidate_state = candidate_mixer.forward_with_state(
                hidden, initial_state=incoming
            )
        row = {
            "layer": layer_index,
            "output_diff": max_diff(control_output, candidate_output, "incoming output"),
            "state_diff": max_diff(control_state, candidate_state, "incoming state"),
        }
        incoming_diffs.append(row)
    if any(row["output_diff"] != 0.0 or row["state_diff"] != 0.0 for row in incoming_diffs):
        raise RuntimeError(f"Zero conditioner changed incoming-state path: {incoming_diffs}")

    control.train().zero_grad(set_to_none=True)
    candidate.train().zero_grad(set_to_none=True)
    cpu_rng, cuda_rng = torch.get_rng_state(), torch.cuda.get_rng_state()
    control_train_logits = control(inputs)
    torch.set_rng_state(cpu_rng)
    torch.cuda.set_rng_state(cuda_rng)
    candidate_train_logits = candidate(inputs)
    graph_names = backward_names(candidate_train_logits)
    mask = targets != -100
    F.cross_entropy(control_train_logits[mask], targets[mask]).backward()
    F.cross_entropy(candidate_train_logits[mask], targets[mask]).backward()
    official_backward_count = sum("ChunkGDN2FunctionBackward" in name for name in graph_names)
    if official_backward_count != 2:
        raise RuntimeError(f"Expected two official GDN2 backwards, got {official_backward_count}")
    gate_gradients = [
        layer.sequence_mixer.layer.block_gram.raw_strength.grad
        for layer in candidate.backbone.layers
    ]
    gate_gradient_min = min(
        float(gradient.abs().amin().item())
        for gradient in gate_gradients
        if gradient is not None
    )
    if len(gate_gradients) != 2 or any(
        gradient is None or not torch.isfinite(gradient).all() for gradient in gate_gradients
    ) or gate_gradient_min <= 0.0:
        raise RuntimeError("Block-Gram gates did not all receive finite nonzero gradients")
    control_parent = dict(control.named_parameters())
    candidate_parent = parent_parameters(candidate)
    parent_gradient_diff = 0.0
    absent = []
    for name, parameter in control_parent.items():
        other = candidate_parent[name]
        if parameter.grad is None and other.grad is None:
            absent.append(name)
            continue
        if parameter.grad is None or other.grad is None:
            raise RuntimeError(f"Parent gradient presence differs: {name}")
        parent_gradient_diff = max(
            parent_gradient_diff,
            max_diff(parameter.grad, other.grad, f"parent gradient {name}"),
        )
    if parent_gradient_diff != 0.0 or set(absent) != EXPECTED_UNUSED_PARENT_GRADIENTS:
        raise RuntimeError(
            f"Zero conditioner changed parent gradients: {parent_gradient_diff}, absent={absent}"
        )

    conditioner = BlockCausalGramConditioner(4, 32, 64).cuda()
    conditioner.raw_strength.data.fill_(1.0)
    q = torch.randn(3, 192, 4, 32, generator=generator, device="cuda")
    k = torch.randn(3, 192, 4, 32, generator=generator, device="cuda")
    base = conditioner.transform_queries(q, k)
    factors = conditioner.last_factors.clone()
    first_block_identity_error = max_diff(factors[0], torch.eye(32, device="cuda"), "first factor")
    changed_k = k.clone()
    changed_k[:, 64:128] += torch.randn(
        changed_k[:, 64:128].shape, generator=generator, device="cuda"
    )
    changed = conditioner.transform_queries(q, changed_k)
    earlier_block_causality_error = max_diff(base[:, :128], changed[:, :128], "causality")
    future_block_dependency = float((base[:, 128:] - changed[:, 128:]).abs().mean().item())
    if first_block_identity_error != 0.0 or earlier_block_causality_error != 0.0:
        raise RuntimeError("Block causality or first-block identity failed")
    if future_block_dependency <= 1e-6:
        raise RuntimeError("Future block did not depend on completed-block Gram")
    permutation = torch.randperm(64, generator=torch.Generator().manual_seed(52026)).cuda()
    permuted_k = k.clone()
    permuted_k[:, :64] = permuted_k[:, :64][:, permutation]
    permuted = conditioner.transform_queries(q, permuted_k)
    block_permutation_error = max_diff(base[:, 64:], permuted[:, 64:], "block permutation")
    if block_permutation_error > 2e-6:
        raise RuntimeError(f"Block token permutation invariance failed: {block_permutation_error}")
    head_permutation = torch.tensor([2, 0, 3, 1], device="cuda")
    raw = conditioner.raw_strength.detach().clone()
    expected = base[:, :, head_permutation]
    conditioner.raw_strength.data.copy_(raw[head_permutation])
    actual = conditioner.transform_queries(q[:, :, head_permutation], k[:, :, head_permutation])
    conditioner.raw_strength.data.copy_(raw)
    head_permutation_error = max_diff(expected, actual, "head permutation")
    if head_permutation_error > 2e-6:
        raise RuntimeError(f"Head permutation equivariance failed: {head_permutation_error}")
    opened = conditioner.diagnostics()
    if not (
        opened["factor_eigenvalue_min"] >= 0.75 - 1e-5
        and opened["factor_eigenvalue_max"] <= 1.25 + 1e-5
        and opened["factor_condition_max"] < 1.67
    ):
        raise RuntimeError(f"Opened conditioner violated bounds: {opened}")

    with torch.no_grad():
        for layer in candidate.backbone.layers:
            layer.sequence_mixer.layer.block_gram.raw_strength.fill_(1.0)
        opened_logits = candidate(inputs)
    mechanism_output_delta = float((opened_logits - control_logits).abs().mean().item())
    if not math.isfinite(mechanism_output_delta) or mechanism_output_delta <= 1e-5:
        raise RuntimeError("Opened Block-Gram mechanism did not affect model output")

    result = {
        "device": device.name,
        "device_uuid": device_uuid,
        "cuda_device_count": torch.cuda.device_count(),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "zoology_sha": zoology_sha,
        "fla_sha": PINNED_FLA_SHA,
        "fla_gdn2_source_sha256": source_hash,
        "fla_gdn2_ops_sha256": ops_hash,
        "data_hashes": data_hashes,
        "parameter_counts": counts,
        "parameter_delta_over_p020": EXPECTED_PARAMETER_DELTA,
        "parent_init_max_diff": parent_init_diff,
        "identity_output_max_diff": identity_output_diff,
        "incoming_state_paths": incoming_diffs,
        "parent_gradient_max_diff": parent_gradient_diff,
        "gate_gradient_min": gate_gradient_min,
        "official_backward_count": official_backward_count,
        "official_call_count": len(recorder.records),
        "first_block_identity_error": first_block_identity_error,
        "earlier_block_causality_error": earlier_block_causality_error,
        "future_block_dependency": future_block_dependency,
        "block_token_permutation_error": block_permutation_error,
        "head_permutation_error": head_permutation_error,
        "opened_conditioner": opened,
        "mechanism_output_delta": mechanism_output_delta,
        "conv_backends": [
            layer.sequence_mixer.layer.base.q_conv1d.backend
            for layer in candidate.backbone.layers
        ],
        "peak_cuda_mem_bytes": torch.cuda.max_memory_allocated(),
    }
    if any(backend != "triton" for backend in result["conv_backends"]):
        raise RuntimeError(f"Short-convolution fallback detected: {result['conv_backends']}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
