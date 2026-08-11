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

from experiments.zoology_mqar.gdn2_futureseed import PINNED_FLA_SHA
from experiments.zoology_mqar.gdn2_log_spd import (
    BoundedLogSPDAddressMetric,
    ZoologyLogSPDGDN2FutureSeedMixer,
    capture_chunk_addresses,
    parent_parameter_hash,
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
EXPECTED_PARAMETER_DELTA = 4216


def normalized_uuid(value: str) -> str:
    return value.removeprefix("GPU-").lower()


def finite_max_abs_difference(
    left: torch.Tensor,
    right: torch.Tensor,
    label: str,
) -> float:
    difference = (left - right).abs()
    if not torch.isfinite(difference).all():
        raise RuntimeError(f"Non-finite {label} difference")
    return float(difference.max().item())


def git_head(path: Path) -> str:
    return subprocess.check_output(
        ["git", "-C", str(path), "rev-parse", "HEAD"], text=True
    ).strip()


def python_tree_hash(path: Path) -> str:
    digest = hashlib.sha256()
    for source in sorted(path.rglob("*.py")):
        digest.update(str(source.relative_to(path)).encode())
        digest.update(source.read_bytes())
    return digest.hexdigest()


def mapped_candidate_parameters(model: torch.nn.Module) -> dict[str, torch.nn.Parameter]:
    result = {}
    for name, parameter in model.named_parameters():
        if name.endswith("layer.address_metric.raw"):
            continue
        result[name.replace(".sequence_mixer.layer.base.", ".sequence_mixer.layer.")] = parameter
    return result


def backward_names(tensor: torch.Tensor) -> list[str]:
    names = []
    seen = set()
    stack = [tensor.grad_fn]
    while stack:
        function = stack.pop()
        if function is None or function in seen:
            continue
        seen.add(function)
        names.append(type(function).__name__)
        stack.extend(next_function for next_function, _index in function.next_functions)
    return names


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
    if fla_root not in gdn2_source.parents:
        raise RuntimeError(f"Unexpected GDN2 source: {gdn2_source}")
    source_hash = hashlib.sha256(gdn2_source.read_bytes()).hexdigest()
    if source_hash != os.environ["FLA_GDN2_SOURCE_SHA256"]:
        raise RuntimeError(f"Unexpected GDN2 source hash: {source_hash}")
    ops_hash = python_tree_hash(fla_root / "fla" / "ops" / "gdn2")
    if ops_hash != os.environ["FLA_GDN2_OPS_SHA256"]:
        raise RuntimeError(f"Unexpected GDN2 recurrence source hash: {ops_hash}")
    zoology_root = Path(os.environ["ZOOLOGY_ROOT"]).resolve()
    if git_head(zoology_root) != EXPECTED_ZOOLOGY_SHA:
        raise RuntimeError("Unexpected Zoology SHA")

    control_config = build_config(
        arm="future_seed_gdn2",
        sequence_length=1024,
        num_kv_pairs=4,
        max_epochs=10,
        batch_size=32,
    )
    candidate_config = build_config(
        arm="future_seed_gdn2_log_spd",
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

    set_determinism(123)
    control = make_model(control_config, "future_seed_gdn2")
    set_determinism(123)
    candidate = make_model(candidate_config, "future_seed_gdn2_log_spd")
    control_parent_hash = parameter_hash(control)
    candidate_parent_hash = parent_parameter_hash(candidate)
    if candidate_parent_hash != control_parent_hash:
        raise RuntimeError("Log-SPD insertion changed parent initialization")
    parameter_counts = {
        "control": sum(parameter.numel() for parameter in control.parameters()),
        "candidate": sum(parameter.numel() for parameter in candidate.parameters()),
    }
    if parameter_counts["candidate"] - parameter_counts["control"] != EXPECTED_PARAMETER_DELTA:
        raise RuntimeError(f"Unexpected parameter delta: {parameter_counts}")

    control_parameters = dict(control.named_parameters())
    candidate_parameters = mapped_candidate_parameters(candidate)
    if set(control_parameters) != set(candidate_parameters):
        raise RuntimeError("Candidate parent parameter names differ from control")
    init_max_diff = max(
        float((control_parameters[name] - candidate_parameters[name]).abs().max().item())
        for name in control_parameters
    )
    if init_max_diff != 0.0:
        raise RuntimeError(f"Candidate parent tensors changed: {init_max_diff}")

    inputs, targets, _slices = next(iter(test_loader))
    inputs = inputs.cuda()
    targets = targets.cuda()
    control = control.cuda().eval()
    candidate = candidate.cuda().eval()
    control_parameters = dict(control.named_parameters())
    candidate_parameters = mapped_candidate_parameters(candidate)
    with torch.no_grad():
        control_logits = control(inputs)
        candidate_logits = candidate(inputs)
    identity_output_max_diff = finite_max_abs_difference(
        control_logits,
        candidate_logits,
        "zero-metric output",
    )
    if identity_output_max_diff != 0.0:
        raise RuntimeError(f"Zero metric changed full output: {identity_output_max_diff}")

    with torch.no_grad(), capture_chunk_addresses() as recorder:
        recorded_control_logits = control(inputs)
    recorder_output_max_diff = finite_max_abs_difference(
        control_logits,
        recorded_control_logits,
        "address-recorder output",
    )
    if recorder_output_max_diff != 0.0 or len(recorder.records) != 2:
        raise RuntimeError(
            "Address recorder changed control output or missed official calls: "
            f"diff={recorder_output_max_diff} calls={len(recorder.records)}"
        )

    generator = torch.Generator(device="cuda").manual_seed(52020)
    incoming_layer_results = []
    incoming_output_max_diff = 0.0
    incoming_state_max_diff = 0.0
    for layer_index in range(2):
        control_mixer = control.backbone.layers[layer_index].sequence_mixer
        candidate_mixer = candidate.backbone.layers[layer_index].sequence_mixer
        if not isinstance(candidate_mixer, ZoologyLogSPDGDN2FutureSeedMixer):
            raise RuntimeError("Candidate mixer type changed")
        hidden = torch.randn(2, 128, 128, generator=generator, device="cuda")
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
            control_output, control_state = control_mixer.forward_with_state(
                hidden, initial_state=incoming
            )
            candidate_output, candidate_state = candidate_mixer.forward_with_state(
                hidden, initial_state=incoming
            )
        output_diff = finite_max_abs_difference(
            control_output,
            candidate_output,
            f"layer-{layer_index} incoming-state output",
        )
        state_diff = finite_max_abs_difference(
            control_state,
            candidate_state,
            f"layer-{layer_index} incoming-state terminal state",
        )
        incoming_output_max_diff = max(incoming_output_max_diff, output_diff)
        incoming_state_max_diff = max(incoming_state_max_diff, state_diff)
        incoming_layer_results.append(
            {
                "layer_index": layer_index,
                "output_max_diff": output_diff,
                "state_max_diff": state_diff,
            }
        )
    if incoming_output_max_diff != 0.0 or incoming_state_max_diff != 0.0:
        raise RuntimeError(
            "Zero metric changed nonzero-state path: "
            f"{incoming_output_max_diff} {incoming_state_max_diff}"
        )

    hidden = torch.randn(2, 128, 128, generator=generator, device="cuda")
    with torch.no_grad():
        control_first_output, control_first_state = (
            control.backbone.layers[0].sequence_mixer.forward_with_state(
                hidden, initial_state=None
            )
        )
        candidate_first_output, candidate_first_state = (
            candidate.backbone.layers[0].sequence_mixer.forward_with_state(
                hidden, initial_state=None
            )
        )
        control_seed = control.backbone.layers[1].sequence_mixer.make_initial_state(
            control_first_state
        )
        candidate_seed = candidate.backbone.layers[1].sequence_mixer.make_initial_state(
            candidate_first_state
        )
    future_seed_transport_max_diff = max(
        finite_max_abs_difference(
            control_first_output,
            candidate_first_output,
            "FutureSeed producer output",
        ),
        finite_max_abs_difference(
            control_first_state,
            candidate_first_state,
            "FutureSeed producer state",
        ),
        finite_max_abs_difference(
            control_seed,
            candidate_seed,
            "FutureSeed transported state",
        ),
    )
    if future_seed_transport_max_diff != 0.0:
        raise RuntimeError(
            f"Zero metric changed native FutureSeed transport: {future_seed_transport_max_diff}"
        )

    control.train().zero_grad(set_to_none=True)
    candidate.train().zero_grad(set_to_none=True)
    cpu_rng_state = torch.get_rng_state()
    cuda_rng_state = torch.cuda.get_rng_state()
    control_train_logits = control(inputs)
    torch.set_rng_state(cpu_rng_state)
    torch.cuda.set_rng_state(cuda_rng_state)
    candidate_train_logits = candidate(inputs)
    graph_names = backward_names(candidate_train_logits)
    mask = targets != -100
    control_loss = F.cross_entropy(control_train_logits[mask], targets[mask])
    candidate_loss = F.cross_entropy(candidate_train_logits[mask], targets[mask])
    control_loss.backward()
    candidate_loss.backward()
    official_backward_count = sum("ChunkGDN2FunctionBackward" in name for name in graph_names)
    if official_backward_count != 2:
        raise RuntimeError(f"Expected two official GDN2 backwards, got {official_backward_count}")

    raw_gradients = [
        block.sequence_mixer.layer.address_metric.raw.grad
        for block in candidate.backbone.layers
    ]
    metric_gradient_status = [
        {
            "layer_index": layer_index,
            "present": gradient is not None,
            "finite": gradient is not None and bool(torch.isfinite(gradient).all()),
            "abs_max": (
                None if gradient is None else float(gradient.abs().amax().item())
            ),
        }
        for layer_index, gradient in enumerate(raw_gradients)
    ]
    if not all(row["present"] and row["finite"] for row in metric_gradient_status):
        raise RuntimeError(
            f"Log-SPD metric gradient is missing or non-finite: {metric_gradient_status}"
        )
    metric_gradient_per_head = [
        float(value)
        for gradient in raw_gradients
        for value in gradient.abs().amax(dim=-1).tolist()
    ]
    metric_gradient_min = min(metric_gradient_per_head)
    if metric_gradient_min <= 0.0:
        raise RuntimeError("A Log-SPD layer received zero gradient")

    parent_gradient_max_diff = 0.0
    for name, parameter in control_parameters.items():
        candidate_parameter = candidate_parameters[name]
        if parameter.grad is None or candidate_parameter.grad is None:
            raise RuntimeError(f"Missing parent gradient: {name}")
        if not torch.isfinite(parameter.grad).all() or not torch.isfinite(
            candidate_parameter.grad
        ).all():
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
        raise RuntimeError(f"Zero metric changed parent gradients: {parent_gradient_max_diff}")

    candidate_mixer = candidate.backbone.layers[1].sequence_mixer
    control_mixer = control.backbone.layers[1].sequence_mixer
    metric: BoundedLogSPDAddressMetric = candidate_mixer.layer.address_metric
    mechanism_hidden = torch.randn(
        2, 128, 128, generator=generator, device="cuda"
    )
    mechanism_incoming = torch.randn(
        2,
        4,
        32,
        32,
        generator=generator,
        device="cuda",
        dtype=torch.float32,
    )
    with torch.no_grad():
        control_output, _control_state = control_mixer.forward_with_state(
            mechanism_hidden, initial_state=mechanism_incoming
        )
        metric.raw.copy_(0.1 * torch.randn(metric.raw.shape, generator=generator, device="cuda"))
        opened = metric.diagnostics()
        opened_output, _opened_state = candidate_mixer.forward_with_state(
            mechanism_hidden, initial_state=mechanism_incoming
        )
    mechanism_difference = (opened_output - control_output).abs()
    if not torch.isfinite(mechanism_difference).all():
        raise RuntimeError("Opened Log-SPD metric produced a non-finite output")
    mechanism_output_delta = float(mechanism_difference.mean().item())
    if not math.isfinite(mechanism_output_delta) or mechanism_output_delta <= 1e-5:
        raise RuntimeError("Opened Log-SPD metric does not affect the model")
    if not (
        opened["fp32_metric_eigenvalue_min"] >= 0.5 - 1e-5
        and opened["fp32_metric_eigenvalue_max"] <= 2.0 + 1e-5
        and opened["fp32_metric_condition_max"] < 4.0 + 1e-5
        and opened["fp32_metric_logdet_abs_max"] <= 1e-4
        and opened["actual_metric_eigenvalue_min"] >= 0.45
        and opened["actual_metric_eigenvalue_max"] <= 2.05
        and opened["actual_metric_condition_max"] < 4.60
        and opened["actual_metric_logdet_abs_max"] <= 0.10
    ):
        raise RuntimeError(f"Opened metric violated its analytic bounds: {opened}")

    probe = torch.randn(3, 11, 4, 32, generator=generator, device="cuda")
    permutation = torch.tensor([2, 0, 3, 1], device="cuda")
    with torch.no_grad():
        raw = metric.raw.clone()
        expected = metric(probe)[:, :, permutation]
        metric.raw.copy_(raw[permutation])
        actual = metric(probe[:, :, permutation])
        metric.raw.copy_(raw)
    head_permutation_error = finite_max_abs_difference(
        expected,
        actual,
        "head permutation",
    )
    if head_permutation_error > 2e-6:
        raise RuntimeError(f"Head permutation equivariance failed: {head_permutation_error}")

    result = {
        "device": device.name,
        "device_uuid": device_uuid,
        "cuda_device_count": torch.cuda.device_count(),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "zoology_sha": EXPECTED_ZOOLOGY_SHA,
        "fla_sha": PINNED_FLA_SHA,
        "fla_gdn2_source_sha256": source_hash,
        "fla_gdn2_ops_sha256": ops_hash,
        "data_hashes": data_hashes,
        "parameter_counts": parameter_counts,
        "parameter_delta": EXPECTED_PARAMETER_DELTA,
        "control_init_parameter_hash": control_parent_hash,
        "parent_init_parameter_hash": candidate_parent_hash,
        "parent_init_max_diff": init_max_diff,
        "identity_output_max_diff": identity_output_max_diff,
        "recorder_output_max_diff": recorder_output_max_diff,
        "recorder_official_call_count": len(recorder.records),
        "incoming_output_max_diff": incoming_output_max_diff,
        "incoming_state_max_diff": incoming_state_max_diff,
        "incoming_layer_results": incoming_layer_results,
        "future_seed_transport_max_diff": future_seed_transport_max_diff,
        "parent_gradient_max_diff": parent_gradient_max_diff,
        "metric_gradient_min": metric_gradient_min,
        "metric_gradient_per_head": metric_gradient_per_head,
        "metric_gradient_status": metric_gradient_status,
        "official_backward_count": official_backward_count,
        "mechanism_output_delta": mechanism_output_delta,
        "opened_metric": opened,
        "head_permutation_error": head_permutation_error,
        "conv_backends": [
            block.sequence_mixer.layer.base.q_conv1d.backend
            for block in candidate.backbone.layers
        ],
        "peak_cuda_mem_bytes": torch.cuda.max_memory_allocated(),
    }
    if any(backend != "triton" for backend in result["conv_backends"]):
        raise RuntimeError(f"Short convolution fallback detected: {result['conv_backends']}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
