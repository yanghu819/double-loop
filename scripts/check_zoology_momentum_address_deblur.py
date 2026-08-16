from __future__ import annotations

import argparse
import importlib
import inspect
import json
import math
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
    EXPECTED_PARAMETER_DELTA_VS_GDN2,
    EXPECTED_RECURRENT_SHA256,
    EXPECTED_STATE_VALUES_PER_LAYER,
    load_external_momentum_layer,
    load_matched_parent_state,
    momentum_fla_compatibility,
)
from experiments.zoology_mqar.momentum_address_deblur import (
    REMOVED_QK_CONV_PARAMETERS_TOTAL,
    PointwiseSiluAddress,
    ZoologyMomentumAddressDeblurFutureSeedMixer,
)
from scripts.check_zoology_contractive_dplr import (
    backward_names,
    normalized_uuid,
)


EXPECTED_TRAIN_HASH = "647c64ece84984a23dfd817c4f277ea83840dbec57cc18bb6c9bf9eda7cc9a68"
EXPECTED_TEST_HASH = "4a8237ba8fe19aaff0d1d72de7b7f6505eaab59cd091442c2f463df34cce278f"
EXPECTED_PARENT_PARAMETERS = 661_584
EXPECTED_P059_PARAMETERS = EXPECTED_PARENT_PARAMETERS + EXPECTED_PARAMETER_DELTA_VS_GDN2
EXPECTED_CANDIDATE_PARAMETERS = (
    EXPECTED_P059_PARAMETERS - REMOVED_QK_CONV_PARAMETERS_TOTAL
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
    uuid, name = (part.strip() for part in rows[0].split(",", maxsplit=1))
    return uuid, name


def _max_abs(left: torch.Tensor, right: torch.Tensor) -> float:
    if not torch.isfinite(left).all() or not torch.isfinite(right).all():
        raise RuntimeError("Nonfinite tensor in parity check")
    return float((left.float() - right.float()).abs().max().item())


def _relative_rms(left: torch.Tensor, right: torch.Tensor) -> float:
    delta = (left.float() - right.float()).square().mean().sqrt()
    scale = right.float().square().mean().sqrt().clamp_min(1e-8)
    return float((delta / scale).item())


def _gradient(parameter: torch.Tensor, name: str) -> float:
    if parameter.grad is None:
        raise RuntimeError(f"Missing gradient: {name}")
    gradient = parameter.grad.detach().float()
    rms = gradient.square().mean().sqrt()
    if not torch.isfinite(gradient).all() or float(rms) == 0.0:
        raise RuntimeError(f"Inactive or nonfinite gradient: {name}")
    return float(rms)


def _sha256(path: Path) -> str:
    import hashlib

    return hashlib.sha256(path.read_bytes()).hexdigest()


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
        raise RuntimeError("P-GDN3-069 requires exactly CUDA index 0")
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
    fla_compatibility = momentum_fla_compatibility()
    if (
        fla_compatibility["missing_required_symbols"]
        or not fla_compatibility["injected_use_cuda_graph"]
        or fla_compatibility["use_cuda_graph"]
    ):
        raise RuntimeError(
            f"Unexpected Momentum FLA compatibility state: {fla_compatibility}"
        )
    repo_root = Path(os.environ["MDN_REPO_ROOT"]).resolve()
    fla_root = Path(os.environ["MDN_FLA_ROOT"]).resolve()
    if subprocess.check_output(
        ["git", "-C", str(repo_root), "rev-parse", "HEAD"], text=True
    ).strip() != EXPECTED_MDN_SHA:
        raise RuntimeError("External Momentum DeltaNet SHA changed")
    layer_path = Path(inspect.getfile(layer_class)).resolve()
    chunk_path = fla_root / "fla" / "ops" / "momentum_delta_rule" / "chunk.py"
    recurrent_path = fla_root / "fla" / "ops" / "momentum_delta_rule" / "fused_recurrent.py"
    source_hashes = {
        "layer": _sha256(layer_path),
        "chunk": _sha256(chunk_path),
        "fused_recurrent": _sha256(recurrent_path),
    }
    if source_hashes != {
        "layer": EXPECTED_LAYER_SHA256,
        "chunk": EXPECTED_CHUNK_SHA256,
        "fused_recurrent": EXPECTED_RECURRENT_SHA256,
    }:
        raise RuntimeError(f"External source drifted: {source_hashes}")

    native_config = build_config(
        arm="future_seed_gdn2",
        sequence_length=1024,
        num_kv_pairs=4,
        max_epochs=10,
        batch_size=32,
    )
    candidate_config = build_config(
        arm="future_seed_momentum_address_deblur",
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
    native = make_model(native_config, "future_seed_gdn2")
    native.load_state_dict(parent_state, strict=True)
    set_determinism(123)
    candidate = make_model(candidate_config, "future_seed_momentum_address_deblur")
    load_matched_parent_state(candidate, parent_state)
    native_parameters = sum(parameter.numel() for parameter in native.parameters())
    candidate_parameters = sum(parameter.numel() for parameter in candidate.parameters())
    if native_parameters != EXPECTED_PARENT_PARAMETERS:
        raise RuntimeError(f"Native parameter count drifted: {native_parameters}")
    if candidate_parameters != EXPECTED_CANDIDATE_PARAMETERS:
        raise RuntimeError(
            f"Candidate parameter count drifted: {candidate_parameters}"
        )
    metadata = candidate._momentum_parent_metadata
    if (
        metadata["tensor_count"] < 20
        or metadata["numel"] < 400_000
        or metadata["source_hash"] != metadata["loaded_hash"]
    ):
        raise RuntimeError(f"Shared parent mapping failed: {metadata}")

    mixers = [block.sequence_mixer for block in candidate.backbone.layers]
    if len(mixers) != 2 or not all(
        type(mixer) is ZoologyMomentumAddressDeblurFutureSeedMixer
        for mixer in mixers
    ):
        raise RuntimeError("Contract requires two exact deblurred Momentum mixers")
    provenance = []
    for index, mixer in enumerate(mixers):
        if type(mixer.layer) is not layer_class:
            raise RuntimeError(f"Layer {index} escaped the pinned external class")
        if mixer.state_size() != EXPECTED_STATE_VALUES_PER_LAYER:
            raise RuntimeError(f"Layer {index} state geometry changed")
        if not isinstance(mixer.layer.q_conv1d, PointwiseSiluAddress):
            raise RuntimeError(f"Layer {index} Q path still mixes tokens")
        if not isinstance(mixer.layer.k_conv1d, PointwiseSiluAddress):
            raise RuntimeError(f"Layer {index} K path still mixes tokens")
        if mixer.layer.v_conv1d.backend != "triton":
            raise RuntimeError(f"Layer {index} V path left Triton")
        if tuple(mixer.layer.v_conv1d.kernel_size) != (4,):
            raise RuntimeError(f"Layer {index} V convolution width drifted")
        provenance.append(
            {
                "layer": index,
                "class": type(mixer.layer).__qualname__,
                "module": type(mixer.layer).__module__,
                "state_shape": [2, "B", 4, 32, 32],
                "kernel": "Chunkmode_ruleFunctionBackward",
                "q_address": "pointwise_silu",
                "k_address": "pointwise_silu",
                "v_conv_backend": mixer.layer.v_conv1d.backend,
                "v_conv_kernel_size": list(mixer.layer.v_conv1d.kernel_size),
            }
        )

    generator = torch.Generator(device="cuda").manual_seed(69069)
    pointwise_input = torch.randn(
        2, 17, 128, generator=generator, device="cuda", dtype=torch.bfloat16
    )
    changed_neighbor = pointwise_input.clone()
    changed_neighbor[:, 8] = torch.randn(
        changed_neighbor[:, 8].shape,
        generator=generator,
        device="cuda",
        dtype=changed_neighbor.dtype,
    )
    with torch.no_grad():
        pointwise_output, pointwise_cache = mixers[0].layer.q_conv1d(
            x=pointwise_input,
            output_final_state=True,
        )
        changed_output, changed_cache = mixers[0].layer.q_conv1d(
            x=changed_neighbor,
            output_final_state=True,
        )
    pointwise_contract = {
        "silu_max_abs_error": _max_abs(
            pointwise_output, F.silu(pointwise_input)
        ),
        "neighbor_independence_max_abs_error": _max_abs(
            pointwise_output[:, 9:], changed_output[:, 9:]
        ),
        "cache_is_none": pointwise_cache is None and changed_cache is None,
    }
    if (
        pointwise_contract["silu_max_abs_error"] != 0.0
        or pointwise_contract["neighbor_independence_max_abs_error"] != 0.0
        or not pointwise_contract["cache_is_none"]
    ):
        raise RuntimeError(f"Pointwise address contract failed: {pointwise_contract}")

    inputs, targets, _slices = next(iter(test_loader))
    inputs = inputs[:4].cuda()
    targets = targets[:4].cuda()
    candidate = candidate.cuda().train()
    logits = candidate(inputs)
    mask = targets != -100
    loss = F.cross_entropy(logits[mask], targets[mask])
    graph_names = backward_names(loss)
    chunk_backward_count = sum(
        "Chunkmode_ruleFunctionBackward" in name for name in graph_names
    )
    if chunk_backward_count != 2:
        raise RuntimeError(
            f"Expected two momentum chunk backward paths, got {chunk_backward_count}"
        )
    loss.backward()
    gradient_rows = []
    for index, mixer in enumerate(mixers):
        layer = mixer.layer
        row = {
            "layer": index,
            "q": _gradient(layer.q_proj.weight, f"layer{index}.q"),
            "k": _gradient(layer.k_proj.weight, f"layer{index}.k"),
            "v": _gradient(layer.v_proj.weight, f"layer{index}.v"),
            "alpha": _gradient(layer.a_proj.weight, f"layer{index}.alpha"),
            "beta": _gradient(layer.b_proj.weight, f"layer{index}.beta"),
            "momentum": _gradient(layer.m_proj.weight, f"layer{index}.momentum"),
            "eta": _gradient(layer.e_proj.weight, f"layer{index}.eta"),
        }
        if index > 0:
            row["future_seed"] = _gradient(
                mixer.future_seed_logit,
                f"layer{index}.future_seed",
            )
        gradient_rows.append(row)

    if mixers[0].last_state_rms is None or mixers[0].last_momentum_rms is None:
        raise RuntimeError("Momentum terminal state was not observed")
    full_stack_state = {
        "state_rms": float(mixers[0].last_state_rms),
        "momentum_rms": float(mixers[0].last_momentum_rms),
        "momentum_to_state_rms": float(mixers[0].last_momentum_to_state_rms),
        "seed_routes": sum(mixer.last_seed_gate is not None for mixer in mixers),
    }
    if (
        full_stack_state["state_rms"] <= 1e-4
        or full_stack_state["momentum_rms"] <= 1e-4
        or not 0.01 <= full_stack_state["momentum_to_state_rms"] <= 20.0
        or full_stack_state["seed_routes"] != 1
    ):
        raise RuntimeError(f"Full-stack state activation failed: {full_stack_state}")

    op_module = importlib.import_module("fla.ops.momentum_delta_rule")
    chunk_rule = op_module.chunk_mode_rule
    recurrent_rule = op_module.fused_recurrent_mode_rule
    generator = torch.Generator(device="cuda").manual_seed(59059)
    shape = (2, 128, 4)
    q = torch.randn(*shape, 32, generator=generator, device="cuda", dtype=torch.bfloat16)
    key = torch.randn_like(q)
    value = torch.randn_like(q)
    log_alpha = -0.05 - 0.2 * torch.rand(*shape, generator=generator, device="cuda")
    log_mu = -0.05 - 0.2 * torch.rand(*shape, generator=generator, device="cuda")
    beta = 0.2 + 0.6 * torch.rand(*shape, generator=generator, device="cuda")
    eta = 0.5 + torch.rand(*shape, generator=generator, device="cuda")
    initial = 0.05 * torch.randn(
        2, 2, 4, 32, 32,
        generator=generator,
        device="cuda",
        dtype=torch.float32,
    )
    with torch.no_grad():
        chunk_output, chunk_state = chunk_rule(
            q=q,
            k=key,
            v=value,
            log_alpha=log_alpha,
            log_mu=log_mu,
            beta=beta,
            eta=eta,
            initial_state=initial,
            output_final_state=True,
        )
        recurrent_output, recurrent_state = recurrent_rule(
            q=q,
            k=key,
            v=value,
            log_alpha=log_alpha,
            log_mu=log_mu,
            beta=beta,
            eta=eta,
            initial_state=initial,
            output_final_state=True,
        )
        zero_output, _ = recurrent_rule(
            q=q,
            k=key,
            v=value,
            log_alpha=log_alpha,
            log_mu=log_mu,
            beta=beta,
            eta=eta,
            initial_state=torch.zeros_like(initial),
            output_final_state=True,
        )
    parity = {
        "output_max_abs": _max_abs(chunk_output, recurrent_output),
        "output_relative_rms": _relative_rms(chunk_output, recurrent_output),
        "state_max_abs": _max_abs(chunk_state, recurrent_state),
        "state_relative_rms": _relative_rms(chunk_state, recurrent_state),
        "initial_state_dependency": _relative_rms(recurrent_output, zero_output),
    }
    if (
        parity["output_relative_rms"] > 0.05
        or parity["state_relative_rms"] > 0.05
        or parity["initial_state_dependency"] <= 1e-4
    ):
        raise RuntimeError(f"Chunk/recurrent contract failed: {parity}")

    seed = mixers[1].make_initial_state(recurrent_state)
    seed_geometry = {
        "shape": list(seed.shape),
        "state_component_rms": float(seed[0].float().square().mean().sqrt()),
        "momentum_component_rms": float(seed[1].float().square().mean().sqrt()),
        "finite": bool(torch.isfinite(seed).all()),
    }
    if (
        seed_geometry["shape"] != [2, 2, 4, 32, 32]
        or seed_geometry["state_component_rms"] <= 1e-4
        or seed_geometry["momentum_component_rms"] <= 1e-4
        or not seed_geometry["finite"]
    ):
        raise RuntimeError(f"FutureSeed state transport failed: {seed_geometry}")

    result = {
        "status": "passed",
        "gpu": {"uuid": gpu_uuid, "name": gpu_name},
        "external": {
            "repo": str(repo_root),
            "sha": EXPECTED_MDN_SHA,
            "source_hashes": source_hashes,
            "redistributed_source": False,
        },
        "host_fla_compatibility": fla_compatibility,
        "data_hashes": data_hashes,
        "parameters": {
            "native": native_parameters,
            "p059": EXPECTED_P059_PARAMETERS,
            "candidate": candidate_parameters,
            "delta": candidate_parameters - native_parameters,
            "delta_vs_p059": candidate_parameters - EXPECTED_P059_PARAMETERS,
        },
        "matched_parent": metadata,
        "provenance": provenance,
        "pointwise_address_contract": pointwise_contract,
        "chunk_backward_count": chunk_backward_count,
        "gradients": gradient_rows,
        "full_stack_state": full_stack_state,
        "chunk_recurrent_parity": parity,
        "futureseed_state_transport": seed_geometry,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
