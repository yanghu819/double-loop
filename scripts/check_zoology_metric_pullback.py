from __future__ import annotations

import argparse
import inspect
import json
import os
from pathlib import Path
import subprocess

import torch
import torch.nn.functional as F
from fla.layers.gdn2 import GatedDeltaNet2
from fla.modules.convolution import ShortConvolution
from zoology.data.utils import prepare_data
from zoology.utils import set_determinism

from experiments.zoology_mqar.gdn2_futureseed import PINNED_FLA_SHA
from experiments.zoology_mqar.gdn2_log_spd import (
    BoundedLogSPDAddressMetric,
    ZoologyLogSPDGDN2FutureSeedMixer,
)
from experiments.zoology_mqar.gdn2_metric_pullback_futureseed import (
    metric_pullback_diagnostics,
    metric_read_pullback,
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


EXPECTED_PARAMETERS = 665_800


def _mixers(model: torch.nn.Module) -> list[ZoologyLogSPDGDN2FutureSeedMixer]:
    mixers = [layer.sequence_mixer for layer in model.backbone.layers]
    if len(mixers) != 2 or not all(
        isinstance(mixer, ZoologyLogSPDGDN2FutureSeedMixer) for mixer in mixers
    ):
        raise RuntimeError("Expected exactly two private Log-SPD GDN2 mixers")
    return mixers


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-gpu-name", required=True)
    parser.add_argument("--expected-gpu-uuid", required=True)
    args = parser.parse_args()

    if os.environ.get("CUDA_VISIBLE_DEVICES") != "0":
        raise RuntimeError("CUDA_VISIBLE_DEVICES must be exactly 0")
    if torch.cuda.device_count() != 1 or torch.cuda.current_device() != 0:
        raise RuntimeError("P-FS2-009 requires one visible CUDA index 0")
    device = torch.cuda.get_device_properties(0)
    device_uuid = str(getattr(device, "uuid", "unavailable"))
    if device.name != args.expected_gpu_name or normalized_uuid(device_uuid) != normalized_uuid(args.expected_gpu_uuid):
        raise RuntimeError(f"Unexpected GPU: {device.name} {device_uuid}")
    if os.environ.get("FLA_EXPECTED_SOURCE_SHA") != PINNED_FLA_SHA:
        raise RuntimeError("Pinned FLA SHA is missing")
    if os.environ.get("FLA_DISABLE_BACKEND_DISPATCH") != "1" or os.environ.get("FLA_CONV_BACKEND") != "triton":
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
        arm="future_seed_gdn2_log_spd",
        sequence_length=1024,
        num_kv_pairs=4,
        max_epochs=10,
        batch_size=32,
    )
    candidate_config = build_config(
        arm="future_seed_gdn2_metric_pullback",
        sequence_length=1024,
        num_kv_pairs=4,
        max_epochs=10,
        batch_size=32,
    )
    train_loader, test_loader = prepare_data(control_config.data)
    hashes = {"train": dataset_hash(train_loader), "test": dataset_hash(test_loader)}
    if hashes != {"train": EXPECTED_TRAIN_HASH, "test": EXPECTED_TEST_HASH}:
        raise RuntimeError(f"Directional MQAR data drifted: {hashes}")

    set_determinism(123)
    control = make_model(control_config, "future_seed_gdn2_log_spd")
    set_determinism(123)
    candidate = make_model(candidate_config, "future_seed_gdn2_metric_pullback")
    if parameter_hash(control) != parameter_hash(candidate):
        raise RuntimeError("Matched initialization differs")
    if sum(p.numel() for p in control.parameters()) != EXPECTED_PARAMETERS or sum(p.numel() for p in candidate.parameters()) != EXPECTED_PARAMETERS:
        raise RuntimeError("Unexpected parameter count")
    control_parameters = dict(control.named_parameters())
    candidate_parameters = dict(candidate.named_parameters())
    if set(control_parameters) != set(candidate_parameters):
        raise RuntimeError("Parameter names differ")
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
            raise RuntimeError(
                f"Short convolution provenance failed: {len(convolutions)}"
            )
        convolution_counts.append(len(convolutions))

    inputs, targets, _slices = next(iter(test_loader))
    inputs = inputs.cuda()
    targets = targets.cuda()
    control = control.cuda().eval()
    candidate = candidate.cuda().eval()
    control_mixers = _mixers(control)
    candidate_mixers = _mixers(candidate)
    with torch.no_grad():
        control_logits = control(inputs)
        candidate_logits = candidate(inputs)
    identity_output_max_diff = finite_max_abs_difference(control_logits, candidate_logits, "identity output")
    if identity_output_max_diff != 0.0:
        raise RuntimeError(f"Identity pullback changed output: {identity_output_max_diff}")

    generator = torch.Generator(device="cuda").manual_seed(52009)
    incoming_results = []
    for layer_index, (cm, pm) in enumerate(zip(control_mixers, candidate_mixers)):
        hidden = torch.randn(2, 128, 128, generator=generator, device="cuda")
        incoming = torch.randn(2, 4, 32, 32, generator=generator, device="cuda")
        with torch.no_grad():
            co, cs = cm.forward_with_state(hidden, initial_state=incoming)
            po, ps = pm.forward_with_state(hidden, initial_state=incoming)
        output_diff = finite_max_abs_difference(co, po, f"incoming output {layer_index}")
        state_diff = finite_max_abs_difference(cs, ps, f"incoming state {layer_index}")
        if output_diff != 0.0 or state_diff != 0.0:
            raise RuntimeError("Identity pullback changed an individual layer")
        incoming_results.append({"layer": layer_index, "output_max_diff": output_diff, "state_max_diff": state_diff})

    control.train().zero_grad(set_to_none=True)
    candidate.train().zero_grad(set_to_none=True)
    cpu_rng = torch.get_rng_state()
    cuda_rng = torch.cuda.get_rng_state()
    control_train_logits = control(inputs)
    torch.set_rng_state(cpu_rng)
    torch.cuda.set_rng_state(cuda_rng)
    candidate_train_logits = candidate(inputs)
    if finite_max_abs_difference(control_train_logits, candidate_train_logits, "train output") != 0.0:
        raise RuntimeError("Identity pullback changed train output")
    graph_names = backward_names(candidate_train_logits)
    mask = targets != -100
    F.cross_entropy(control_train_logits[mask], targets[mask]).backward()
    F.cross_entropy(candidate_train_logits[mask], targets[mask]).backward()
    official_backward_count = sum("ChunkGDN2FunctionBackward" in name for name in graph_names)
    if official_backward_count != 2:
        raise RuntimeError(f"Expected two official GDN2 backwards, got {official_backward_count}")
    parent_gradient_max_diff = 0.0
    metric_gradient_differences = []
    for name, cp in control.named_parameters():
        pp = candidate_parameters[name]
        if cp.grad is None and pp.grad is None:
            continue
        if cp.grad is None or pp.grad is None or not torch.isfinite(cp.grad).all() or not torch.isfinite(pp.grad).all():
            raise RuntimeError(f"Gradient contract failed for {name}")
        diff = finite_max_abs_difference(cp.grad, pp.grad, f"gradient {name}")
        if name.endswith("address_metric.raw"):
            metric_gradient_differences.append(diff)
        else:
            parent_gradient_max_diff = max(parent_gradient_max_diff, diff)
    if parent_gradient_max_diff != 0.0:
        raise RuntimeError(f"Identity pullback changed parent gradients: {parent_gradient_max_diff}")
    if len(metric_gradient_differences) != 2 or not all(value > 0 for value in metric_gradient_differences):
        raise RuntimeError(f"Transport metric gradient is absent: {metric_gradient_differences}")

    with torch.no_grad():
        for cm, pm in zip(control_mixers, candidate_mixers):
            raw = 0.08 * torch.randn(pm.layer.address_metric.raw.shape, generator=generator, device="cuda")
            cm.layer.address_metric.raw.copy_(raw)
            pm.layer.address_metric.raw.copy_(raw)
        control_open = control.eval()(inputs[:8])
        candidate_open = candidate.eval()(inputs[:8])
    opened_output_relative_rms = float(
        (candidate_open.float() - control_open.float()).square().mean().sqrt().item()
        / control_open.float().square().mean().sqrt().clamp_min(1e-8).item()
    )
    diagnostics = metric_pullback_diagnostics(candidate)
    if opened_output_relative_rms <= 1e-4 or diagnostics["active_routes"] != 1:
        raise RuntimeError("Opened pullback did not affect the full model")
    row = diagnostics["per_receiver"][0]
    if not row["finite"] or row["pullback_delta_fro_mean"] <= 1e-4 or row["pullback_condition_max"] >= 4.60 or row["transported_to_producer_rms_max"] > 4.60:
        raise RuntimeError(f"Opened pullback is inactive or unbounded: {row}")

    producer = BoundedLogSPDAddressMetric(4, 32).cuda()
    receiver = BoundedLogSPDAddressMetric(4, 32).cuda()
    producer_permuted = BoundedLogSPDAddressMetric(4, 32).cuda()
    receiver_permuted = BoundedLogSPDAddressMetric(4, 32).cuda()
    state = torch.randn(3, 4, 32, 32, generator=generator, device="cuda")
    permutation = torch.tensor([2, 0, 3, 1], device="cuda")
    with torch.no_grad():
        producer.raw.copy_(0.1 * torch.randn(producer.raw.shape, generator=generator, device="cuda"))
        receiver.raw.copy_(0.1 * torch.randn(receiver.raw.shape, generator=generator, device="cuda"))
        producer_permuted.raw.copy_(producer.raw[permutation])
        receiver_permuted.raw.copy_(receiver.raw[permutation])
        transported, direct = metric_read_pullback(state, producer_metric=producer, receiver_metric=receiver)
        transported_permuted, _ = metric_read_pullback(state[:, permutation], producer_metric=producer_permuted, receiver_metric=receiver_permuted)
    head_permutation_max_diff = finite_max_abs_difference(transported[:, permutation], transported_permuted, "head permutation")
    direct_solve = torch.linalg.solve(receiver.applied_matrix(torch.bfloat16), producer.applied_matrix(torch.bfloat16))
    solve_relative_error = float(
        (direct["pullback"] - direct_solve).square().mean().sqrt().item()
        / direct_solve.square().mean().sqrt().clamp_min(1e-8).item()
    )
    if head_permutation_max_diff != 0.0 or solve_relative_error > 1e-6:
        raise RuntimeError("Pullback solve/equivariance contract failed")

    result = {
        "status": "passed",
        "gpu": {"name": device.name, "uuid": device_uuid},
        "pinned_fla_sha": PINNED_FLA_SHA,
        "zoology_sha": EXPECTED_ZOOLOGY_SHA,
        "data_hashes": hashes,
        "parameters": EXPECTED_PARAMETERS,
        "short_convolution_counts": convolution_counts,
        "official_backward_count": official_backward_count,
        "identity_output_max_diff": identity_output_max_diff,
        "incoming_state_identity": incoming_results,
        "parent_gradient_max_diff": parent_gradient_max_diff,
        "metric_gradient_differences": metric_gradient_differences,
        "opened_output_relative_rms": opened_output_relative_rms,
        "opened_diagnostics": diagnostics,
        "head_permutation_max_diff": head_permutation_max_diff,
        "solve_relative_error": solve_relative_error,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
