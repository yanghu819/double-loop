from __future__ import annotations

import argparse
import inspect
import json
import os
import subprocess
from pathlib import Path

import torch
import torch.nn.functional as F
from zoology.data.utils import prepare_data
from zoology.utils import set_determinism

from experiments.zoology_mqar.length_scaling import (
    build_config,
    dataset_hash,
    make_model,
)
from experiments.zoology_mqar.momentum_futureseed import (
    EXPECTED_CHUNK_SHA256,
    EXPECTED_LAYER_SHA256,
    EXPECTED_MDN_SHA,
    EXPECTED_RECURRENT_SHA256,
    EXPECTED_STATE_VALUES_PER_LAYER,
    ZoologyMomentumDeltaFutureSeedMixer,
    load_external_momentum_layer,
    load_matched_parent_state,
    momentum_fla_compatibility,
)
from experiments.zoology_mqar.momentum_log_spd import (
    EXPECTED_PARAMETER_DELTA_VS_MOMENTUM,
    ZoologyMomentumLogSPDFutureSeedMixer,
)
from scripts.check_zoology_contractive_dplr import (
    backward_names,
    normalized_uuid,
)


EXPECTED_TRAIN_HASH = "647c64ece84984a23dfd817c4f277ea83840dbec57cc18bb6c9bf9eda7cc9a68"
EXPECTED_TEST_HASH = "4a8237ba8fe19aaff0d1d72de7b7f6505eaab59cd091442c2f463df34cce278f"
EXPECTED_MOMENTUM_PARAMETERS = 599_672
EXPECTED_CANDIDATE_PARAMETERS = (
    EXPECTED_MOMENTUM_PARAMETERS + EXPECTED_PARAMETER_DELTA_VS_MOMENTUM
)


def _visible_gpu() -> tuple[str, str]:
    output = subprocess.check_output(
        [
            "nvidia-smi",
            "-i",
            "0",
            "--query-gpu=uuid,name",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    ).strip()
    rows = [row.strip() for row in output.splitlines() if row.strip()]
    if len(rows) != 1:
        raise RuntimeError(f"Expected one visible GPU row, got {rows}")
    return tuple(part.strip() for part in rows[0].split(",", maxsplit=1))


def _sha256(path: Path) -> str:
    import hashlib

    return hashlib.sha256(path.read_bytes()).hexdigest()


def _gradient(parameter: torch.Tensor, name: str) -> float:
    if parameter.grad is None:
        raise RuntimeError(f"Missing gradient: {name}")
    gradient = parameter.grad.detach().float()
    rms = gradient.square().mean().sqrt()
    if not torch.isfinite(gradient).all() or float(rms) == 0.0:
        raise RuntimeError(f"Inactive or nonfinite gradient: {name}")
    return float(rms)


def _max_abs(left: torch.Tensor, right: torch.Tensor) -> float:
    if not torch.isfinite(left).all() or not torch.isfinite(right).all():
        raise RuntimeError("Nonfinite tensor in parity check")
    return float((left.float() - right.float()).abs().max().item())


def _relative_rms(left: torch.Tensor, right: torch.Tensor) -> float:
    delta = (left.float() - right.float()).square().mean().sqrt()
    scale = right.float().square().mean().sqrt().clamp_min(1e-8)
    return float((delta / scale).item())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--matched-init", type=Path, required=True)
    parser.add_argument("--expected-gpu-name", required=True)
    parser.add_argument("--expected-gpu-uuid", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if os.environ.get("CUDA_VISIBLE_DEVICES") != "0":
        raise RuntimeError("CUDA_VISIBLE_DEVICES must be exactly 0")
    if torch.cuda.device_count() != 1 or torch.cuda.current_device() != 0:
        raise RuntimeError("P-GDN3-060 requires exactly CUDA index 0")
    gpu_uuid, gpu_name = _visible_gpu()
    device = torch.cuda.get_device_properties(0)
    if (
        gpu_name != args.expected_gpu_name
        or gpu_uuid != args.expected_gpu_uuid
        or device.name != args.expected_gpu_name
        or normalized_uuid(str(getattr(device, "uuid", "unavailable")))
        != normalized_uuid(args.expected_gpu_uuid)
    ):
        raise RuntimeError(f"Unexpected GPU: {gpu_uuid}, {gpu_name}")

    layer_class = load_external_momentum_layer()
    compatibility = momentum_fla_compatibility()
    if (
        compatibility["missing_required_symbols"]
        or not compatibility["injected_use_cuda_graph"]
        or compatibility["use_cuda_graph"]
    ):
        raise RuntimeError(f"Unexpected FLA bridge state: {compatibility}")
    repo_root = Path(os.environ["MDN_REPO_ROOT"]).resolve()
    fla_root = Path(os.environ["MDN_FLA_ROOT"]).resolve()
    if subprocess.check_output(
        ["git", "-C", str(repo_root), "rev-parse", "HEAD"], text=True
    ).strip() != EXPECTED_MDN_SHA:
        raise RuntimeError("External Momentum DeltaNet SHA changed")
    source_hashes = {
        "layer": _sha256(Path(inspect.getfile(layer_class)).resolve()),
        "chunk": _sha256(
            fla_root / "fla" / "ops" / "momentum_delta_rule" / "chunk.py"
        ),
        "fused_recurrent": _sha256(
            fla_root
            / "fla"
            / "ops"
            / "momentum_delta_rule"
            / "fused_recurrent.py"
        ),
    }
    if source_hashes != {
        "layer": EXPECTED_LAYER_SHA256,
        "chunk": EXPECTED_CHUNK_SHA256,
        "fused_recurrent": EXPECTED_RECURRENT_SHA256,
    }:
        raise RuntimeError(f"External source drifted: {source_hashes}")

    momentum_config = build_config(
        arm="future_seed_momentum_delta",
        sequence_length=1024,
        num_kv_pairs=4,
        max_epochs=10,
        batch_size=32,
    )
    candidate_config = build_config(
        arm="future_seed_momentum_log_spd",
        sequence_length=1024,
        num_kv_pairs=4,
        max_epochs=10,
        batch_size=32,
    )
    train_loader, test_loader = prepare_data(candidate_config.data)
    data_hashes = {
        "train": dataset_hash(train_loader),
        "test": dataset_hash(test_loader),
    }
    if data_hashes != {"train": EXPECTED_TRAIN_HASH, "test": EXPECTED_TEST_HASH}:
        raise RuntimeError(f"Directional MQAR data drifted: {data_hashes}")

    parent_state = torch.load(args.matched_init, map_location="cpu", weights_only=True)
    set_determinism(123)
    momentum = make_model(momentum_config, "future_seed_momentum_delta")
    load_matched_parent_state(momentum, parent_state)
    set_determinism(123)
    candidate = make_model(candidate_config, "future_seed_momentum_log_spd")
    load_matched_parent_state(candidate, parent_state)
    momentum_parameters = sum(parameter.numel() for parameter in momentum.parameters())
    candidate_parameters = sum(parameter.numel() for parameter in candidate.parameters())
    if momentum_parameters != EXPECTED_MOMENTUM_PARAMETERS:
        raise RuntimeError(f"Momentum parameter count drifted: {momentum_parameters}")
    if candidate_parameters != EXPECTED_CANDIDATE_PARAMETERS:
        raise RuntimeError(f"Candidate parameter count drifted: {candidate_parameters}")

    momentum_state = momentum.state_dict()
    candidate_state = candidate.state_dict()
    shared_names = sorted(momentum_state)
    if any(
        name not in candidate_state
        or not torch.equal(momentum_state[name], candidate_state[name])
        for name in shared_names
    ):
        raise RuntimeError("Zero-metric candidate changed a Momentum parent tensor")
    metric_names = sorted(name for name in candidate_state if name.endswith("address_metric.raw"))
    if len(metric_names) != 2 or any(torch.count_nonzero(candidate_state[name]) for name in metric_names):
        raise RuntimeError(f"Metric initialization is not exact zero: {metric_names}")

    momentum_mixers = [block.sequence_mixer for block in momentum.backbone.layers]
    candidate_mixers = [block.sequence_mixer for block in candidate.backbone.layers]
    if len(momentum_mixers) != 2 or not all(
        type(mixer) is ZoologyMomentumDeltaFutureSeedMixer
        for mixer in momentum_mixers
    ):
        raise RuntimeError("Expected two exact Momentum parent mixers")
    if len(candidate_mixers) != 2 or not all(
        type(mixer) is ZoologyMomentumLogSPDFutureSeedMixer
        for mixer in candidate_mixers
    ):
        raise RuntimeError("Expected two exact Momentum Log-SPD mixers")
    for mixer in candidate_mixers:
        if type(mixer.layer) is not layer_class:
            raise RuntimeError("Candidate escaped the pinned external layer")
        if mixer.state_size() != EXPECTED_STATE_VALUES_PER_LAYER:
            raise RuntimeError("Candidate state geometry changed")

    inputs, targets, _slices = next(iter(test_loader))
    inputs = inputs[:4].cuda()
    targets = targets[:4].cuda()
    momentum = momentum.cuda().eval()
    candidate = candidate.cuda().eval()
    with torch.no_grad():
        momentum_logits = momentum(inputs)
        candidate_logits = candidate(inputs)
    full_model_identity = {
        "max_abs": _max_abs(candidate_logits, momentum_logits),
        "exact": bool(torch.equal(candidate_logits, momentum_logits)),
    }
    if not full_model_identity["exact"]:
        raise RuntimeError(f"Zero metric changed full model output: {full_model_identity}")

    generator = torch.Generator(device="cuda").manual_seed(60060)
    hidden = torch.randn(
        2,
        128,
        128,
        generator=generator,
        device="cuda",
        dtype=torch.float32,
    )
    initial_state = 0.05 * torch.randn(
        2,
        2,
        4,
        32,
        32,
        generator=generator,
        device="cuda",
        dtype=torch.float32,
    )
    layer_identity = []
    for index, (parent_mixer, candidate_mixer) in enumerate(
        zip(momentum_mixers, candidate_mixers, strict=True)
    ):
        parent_mixer.eval()
        candidate_mixer.eval()
        with torch.no_grad():
            parent_output, parent_terminal = parent_mixer.forward_with_state(
                hidden,
                initial_state=initial_state,
            )
            candidate_output, candidate_terminal = candidate_mixer.forward_with_state(
                hidden,
                initial_state=initial_state,
            )
        row = {
            "layer": index,
            "output_exact": bool(torch.equal(candidate_output, parent_output)),
            "state_exact": bool(torch.equal(candidate_terminal, parent_terminal)),
            "output_max_abs": _max_abs(candidate_output, parent_output),
            "state_max_abs": _max_abs(candidate_terminal, parent_terminal),
        }
        if not row["output_exact"] or not row["state_exact"]:
            raise RuntimeError(f"Nonzero-state identity failed: {row}")
        layer_identity.append(row)

    candidate.train().zero_grad(set_to_none=True)
    logits = candidate(inputs)
    mask = targets != -100
    loss = F.cross_entropy(logits[mask], targets[mask])
    graph_names = backward_names(loss)
    chunk_backward_count = sum(
        "Chunkmode_ruleFunctionBackward" in name for name in graph_names
    )
    if chunk_backward_count != 2:
        raise RuntimeError(f"Expected two momentum chunk backwards, got {chunk_backward_count}")
    loss.backward()
    gradients = []
    for index, mixer in enumerate(candidate_mixers):
        row = {
            "layer": index,
            "metric": _gradient(mixer.address_metric.raw, f"layer{index}.metric"),
            "q": _gradient(mixer.layer.q_proj.weight, f"layer{index}.q"),
            "k": _gradient(mixer.layer.k_proj.weight, f"layer{index}.k"),
            "v": _gradient(mixer.layer.v_proj.weight, f"layer{index}.v"),
            "momentum": _gradient(mixer.layer.m_proj.weight, f"layer{index}.momentum"),
        }
        if index > 0:
            row["future_seed"] = _gradient(
                mixer.future_seed_logit,
                f"layer{index}.future_seed",
            )
        gradients.append(row)

    metric = candidate_mixers[0].address_metric
    with torch.no_grad():
        opened = torch.linspace(
            -0.08,
            0.08,
            metric.raw.numel(),
            device=metric.raw.device,
            dtype=metric.raw.dtype,
        ).reshape_as(metric.raw)
        metric.raw.copy_(opened)
        delta = metric.matrix() - torch.eye(32, device="cuda", dtype=torch.float32)
        address = torch.randn(
            3,
            7,
            4,
            32,
            generator=generator,
            device="cuda",
            dtype=torch.bfloat16,
        )
        transformed = metric.transform_with_delta(address, delta)
        permutation = torch.tensor([2, 0, 3, 1], device="cuda")
        permuted = metric.transform_with_delta(
            address[..., permutation, :],
            delta[permutation],
        )
        expected_permuted = transformed[..., permutation, :]
        opened_diagnostics = metric.diagnostics()
    synthetic = {
        "relative_rms": _relative_rms(transformed, address),
        "head_permutation_max_abs": _max_abs(permuted, expected_permuted),
        "diagnostics": opened_diagnostics,
    }
    if (
        synthetic["relative_rms"] <= 1e-4
        or synthetic["head_permutation_max_abs"] != 0.0
        or opened_diagnostics["actual_metric_eigenvalue_min"] <= 0.0
        or opened_diagnostics["actual_metric_condition_max"] > 4.1
        or opened_diagnostics["actual_metric_logdet_abs_max"] > 0.05
    ):
        raise RuntimeError(f"Opened metric contract failed: {synthetic}")

    seed = candidate_mixers[1].make_initial_state(initial_state)
    seed_geometry = {
        "shape": list(seed.shape),
        "state_rms": float(seed[0].float().square().mean().sqrt()),
        "momentum_rms": float(seed[1].float().square().mean().sqrt()),
        "finite": bool(torch.isfinite(seed).all()),
    }
    if seed_geometry["shape"] != [2, 2, 4, 32, 32] or not seed_geometry["finite"]:
        raise RuntimeError(f"FutureSeed geometry changed: {seed_geometry}")

    result = {
        "status": "passed",
        "plan": "P-GDN3-060",
        "gpu": {"uuid": gpu_uuid, "name": gpu_name},
        "external": {
            "repo": str(repo_root),
            "sha": EXPECTED_MDN_SHA,
            "source_hashes": source_hashes,
            "redistributed_source": False,
        },
        "host_fla_compatibility": compatibility,
        "data_hashes": data_hashes,
        "parameters": {
            "momentum": momentum_parameters,
            "candidate": candidate_parameters,
            "delta": candidate_parameters - momentum_parameters,
        },
        "metric_parameter_names": metric_names,
        "full_model_zero_metric_identity": full_model_identity,
        "nonzero_state_zero_metric_identity": layer_identity,
        "chunk_backward_count": chunk_backward_count,
        "gradients": gradients,
        "opened_metric": synthetic,
        "futureseed_state_transport": seed_geometry,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
