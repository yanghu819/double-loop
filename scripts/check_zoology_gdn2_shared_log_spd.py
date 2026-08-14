from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import math
import os
from pathlib import Path

import torch
import torch.nn.functional as F
from fla.layers.gdn2 import GatedDeltaNet2
from zoology.data.utils import prepare_data
from zoology.utils import set_determinism

from experiments.zoology_mqar.gdn2_futureseed import PINNED_FLA_SHA
from experiments.zoology_mqar.gdn2_log_spd import capture_chunk_addresses
from experiments.zoology_mqar.gdn2_shared_log_spd import (
    parent_parameter_hash,
    shared_log_spd_diagnostics,
    shared_log_spd_mixers,
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
    EXPECTED_UNUSED_PARENT_GRADIENTS,
    EXPECTED_ZOOLOGY_SHA,
    backward_names,
    finite_max_abs_difference,
    git_head,
    mapped_candidate_parameters,
    normalized_uuid,
    python_tree_hash,
)


EXPECTED_PARAMETER_DELTA = 2_108


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
        raise RuntimeError(f"Unexpected GDN2 recurrence hash: {ops_hash}")
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
        arm="future_seed_gdn2_shared_log_spd",
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
    candidate = make_model(candidate_config, "future_seed_gdn2_shared_log_spd")
    mixers = shared_log_spd_mixers(candidate)
    metric_ids = [id(mixer.layer.address_metric) for mixer in mixers]
    raw_ids = [id(mixer.layer.address_metric.raw) for mixer in mixers]
    if len(set(metric_ids)) != 1 or len(set(raw_ids)) != 1:
        raise RuntimeError("Candidate layers do not share one metric parameter")

    control_parent_hash = parameter_hash(control)
    candidate_parent_hash = parent_parameter_hash(candidate)
    if candidate_parent_hash != control_parent_hash:
        raise RuntimeError("Shared metric insertion changed parent initialization")
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
    mixers = shared_log_spd_mixers(candidate)
    cuda_raw_pointers = [mixer.layer.address_metric.raw.data_ptr() for mixer in mixers]
    if len(set(cuda_raw_pointers)) != 1:
        raise RuntimeError("Shared metric storage split after CUDA transfer")

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
        recorded_candidate_logits = candidate(inputs)
    recorder_output_max_diff = finite_max_abs_difference(
        candidate_logits,
        recorded_candidate_logits,
        "address-recorder output",
    )
    if recorder_output_max_diff != 0.0 or len(recorder.records) != 2:
        raise RuntimeError(
            "Address recorder changed output or missed official calls: "
            f"diff={recorder_output_max_diff} calls={len(recorder.records)}"
        )

    generator = torch.Generator(device="cuda").manual_seed(52033)
    incoming_results = []
    incoming_output_max_diff = 0.0
    incoming_state_max_diff = 0.0
    layer_gradient_contributions = []
    shared_raw = mixers[0].layer.address_metric.raw
    for layer_index in range(2):
        control_mixer = control.backbone.layers[layer_index].sequence_mixer
        candidate_mixer = candidate.backbone.layers[layer_index].sequence_mixer
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
                hidden,
                initial_state=incoming,
            )
            candidate_output, candidate_state = candidate_mixer.forward_with_state(
                hidden,
                initial_state=incoming,
            )
        output_diff = finite_max_abs_difference(
            control_output,
            candidate_output,
            f"layer-{layer_index} incoming output",
        )
        state_diff = finite_max_abs_difference(
            control_state,
            candidate_state,
            f"layer-{layer_index} incoming state",
        )
        incoming_output_max_diff = max(incoming_output_max_diff, output_diff)
        incoming_state_max_diff = max(incoming_state_max_diff, state_diff)
        incoming_results.append(
            {
                "layer_index": layer_index,
                "output_max_diff": output_diff,
                "state_max_diff": state_diff,
            }
        )

        candidate.zero_grad(set_to_none=True)
        layer_output, layer_state = candidate_mixer.forward_with_state(
            hidden,
            initial_state=incoming,
        )
        layer_loss = (
            layer_output.float().square().mean()
            + 1e-3 * layer_state.float().square().mean()
        )
        layer_loss.backward()
        gradient = shared_raw.grad
        layer_gradient_contributions.append(
            {
                "layer_index": layer_index,
                "present": gradient is not None,
                "finite": gradient is not None and bool(torch.isfinite(gradient).all()),
                "abs_max": None if gradient is None else float(gradient.abs().amax().item()),
            }
        )
    if incoming_output_max_diff != 0.0 or incoming_state_max_diff != 0.0:
        raise RuntimeError("Zero metric changed a nonzero incoming-state path")
    if not all(
        row["present"] and row["finite"] and float(row["abs_max"]) > 0.0
        for row in layer_gradient_contributions
    ):
        raise RuntimeError(
            f"A layer does not train the shared metric: {layer_gradient_contributions}"
        )

    hidden = torch.randn(2, 128, 128, generator=generator, device="cuda")
    with torch.no_grad():
        control_first_output, control_first_state = (
            control.backbone.layers[0].sequence_mixer.forward_with_state(
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
        finite_max_abs_difference(control_seed, candidate_seed, "FutureSeed seed"),
    )
    if future_seed_transport_max_diff != 0.0:
        raise RuntimeError("Zero metric changed native FutureSeed transport")

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
    official_backward_count = sum(
        "ChunkGDN2FunctionBackward" in name for name in graph_names
    )
    if official_backward_count != 2:
        raise RuntimeError(f"Expected two official GDN2 backwards, got {official_backward_count}")
    if shared_raw.grad is None or not torch.isfinite(shared_raw.grad).all():
        raise RuntimeError("Shared metric gradient is missing or non-finite")
    shared_metric_gradient_min_per_head = float(
        shared_raw.grad.abs().amax(dim=-1).amin().item()
    )
    if shared_metric_gradient_min_per_head <= 0.0:
        raise RuntimeError("A shared metric head received zero gradient")

    control_parameters = dict(control.named_parameters())
    candidate_parameters = mapped_candidate_parameters(candidate)
    parent_gradient_max_diff = 0.0
    parent_gradients_absent_in_both = []
    for name, parameter in control_parameters.items():
        candidate_parameter = candidate_parameters[name]
        if parameter.grad is None and candidate_parameter.grad is None:
            parent_gradients_absent_in_both.append(name)
            continue
        if parameter.grad is None or candidate_parameter.grad is None:
            raise RuntimeError(f"Parent gradient presence differs: {name}")
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
    if set(parent_gradients_absent_in_both) != EXPECTED_UNUSED_PARENT_GRADIENTS:
        raise RuntimeError(
            f"Unexpected jointly unused parent gradients: {parent_gradients_absent_in_both}"
        )

    metric = mixers[0].layer.address_metric
    with torch.no_grad():
        metric.raw.copy_(
            0.1 * torch.randn(metric.raw.shape, generator=generator, device="cuda")
        )
        opened = metric.diagnostics()
    opened_layer_deltas = []
    for layer_index in range(2):
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
            control_output, _ = control.backbone.layers[
                layer_index
            ].sequence_mixer.forward_with_state(hidden, initial_state=incoming)
            opened_output, _ = candidate.backbone.layers[
                layer_index
            ].sequence_mixer.forward_with_state(hidden, initial_state=incoming)
        value = float((opened_output - control_output).abs().mean().item())
        if not math.isfinite(value) or value <= 1e-5:
            raise RuntimeError(f"Opened shared metric misses layer {layer_index}: {value}")
        opened_layer_deltas.append(value)
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
        raise RuntimeError(f"Opened metric violated bounds: {opened}")

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

    diagnostics = shared_log_spd_diagnostics(candidate)
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
        "incoming_results": incoming_results,
        "future_seed_transport_max_diff": future_seed_transport_max_diff,
        "parent_gradient_max_diff": parent_gradient_max_diff,
        "parent_gradients_absent_in_both": parent_gradients_absent_in_both,
        "shared_metric_gradient_min_per_head": shared_metric_gradient_min_per_head,
        "layer_gradient_contributions": layer_gradient_contributions,
        "official_backward_count": official_backward_count,
        "opened_layer_output_deltas": opened_layer_deltas,
        "opened_metric": opened,
        "head_permutation_error": head_permutation_error,
        "shared_diagnostics": diagnostics,
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
