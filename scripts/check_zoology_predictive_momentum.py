from __future__ import annotations

import argparse
import importlib
import inspect
import json
import math
import os
import subprocess
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from zoology.data.utils import prepare_data
from zoology.utils import set_determinism

from experiments.zoology_mqar.length_scaling import (
    build_config,
    dataset_hash,
    make_model,
    parameter_hash,
)
from experiments.zoology_mqar.momentum_futureseed import (
    EXPECTED_CHUNK_SHA256,
    EXPECTED_LAYER_SHA256,
    EXPECTED_MDN_SHA,
    EXPECTED_RECURRENT_SHA256,
    EXPECTED_STATE_VALUES_PER_LAYER,
    load_external_momentum_layer,
    load_matched_parent_state,
    momentum_fla_compatibility,
)
from experiments.zoology_mqar.momentum_predictive import (
    ZoologyPredictiveMomentumFutureSeedMixer,
    predictive_momentum_reference,
    predictive_momentum_rule,
)
from scripts.check_zoology_contractive_dplr import (
    backward_names,
    normalized_uuid,
)


EXPECTED_TRAIN_HASH = "647c64ece84984a23dfd817c4f277ea83840dbec57cc18bb6c9bf9eda7cc9a68"
EXPECTED_TEST_HASH = "4a8237ba8fe19aaff0d1d72de7b7f6505eaab59cd091442c2f463df34cce278f"
EXPECTED_PARAMETERS = 599_672
EXPECTED_MATCHED_INIT_SHA256 = (
    "7402e46c48cbd47070d65262d1a1b62a32ac55a4d644a6fb9644b96b0914850f"
)
EXPECTED_PARENT_PARAMETER_HASH = (
    "0adf26f657e59a35e90d7e54b905721b10f537b6391d107676248de16b71f13f"
)
EXPECTED_INITIAL_PARAMETER_HASH = (
    "dc8f49f92c00f0f08863cf6183692246cf2c44f0c5aa3a35892bb9555ec17bfb"
)


def _sha256(path: Path) -> str:
    import hashlib

    return hashlib.sha256(path.read_bytes()).hexdigest()


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


def _relative_rms(left: torch.Tensor, right: torch.Tensor) -> float:
    if not torch.isfinite(left).all() or not torch.isfinite(right).all():
        raise RuntimeError("Nonfinite parity tensor")
    delta = (left.float() - right.float()).square().mean().sqrt()
    scale = right.float().square().mean().sqrt().clamp_min(1e-8)
    return float((delta / scale).item())


def _gradient_rms(tensor: torch.Tensor, name: str) -> float:
    if tensor.grad is None:
        raise RuntimeError(f"Missing gradient {name}")
    gradient = tensor.grad.detach().float()
    rms = gradient.square().mean().sqrt()
    if not torch.isfinite(gradient).all() or float(rms) == 0.0:
        raise RuntimeError(f"Inactive gradient {name}")
    return float(rms)


def _leaf(value: torch.Tensor) -> torch.Tensor:
    return value.detach().clone().requires_grad_(True)


def _synthetic_parity() -> dict[str, Any]:
    generator = torch.Generator(device="cuda").manual_seed(62062)
    shape = (2, 9, 2)
    q0 = torch.randn(*shape, 8, generator=generator, device="cuda")
    k0 = torch.randn_like(q0)
    v0 = torch.randn_like(q0)
    log_alpha0 = -0.15 - 0.25 * torch.rand(*shape, generator=generator, device="cuda")
    log_mu0 = -0.15 - 0.25 * torch.rand(*shape, generator=generator, device="cuda")
    beta0 = 0.05 + 0.20 * torch.rand(*shape, generator=generator, device="cuda")
    eta0 = 0.40 + 0.40 * torch.rand(*shape, generator=generator, device="cuda")
    initial0 = 0.03 * torch.randn(
        2,
        2,
        2,
        8,
        8,
        generator=generator,
        device="cuda",
    )
    output_weight = torch.randn_like(v0)
    state_weight = torch.randn_like(initial0)

    custom_inputs = [
        _leaf(value)
        for value in (q0, k0, v0, log_alpha0, log_mu0, beta0, eta0, initial0)
    ]
    q, k, v, log_alpha, log_mu, beta, eta, initial = custom_inputs
    custom_output, custom_state, custom_stats = predictive_momentum_rule(
        q=q,
        k=k,
        v=v,
        log_alpha=log_alpha,
        log_mu=log_mu,
        beta=beta,
        eta=eta,
        initial_state=initial,
        output_final_state=True,
    )
    custom_loss = (
        custom_output.float().mul(output_weight).sum()
        + custom_state.float().mul(state_weight).sum()
    )
    custom_loss.backward()

    reference_inputs = [
        _leaf(value)
        for value in (q0, k0, v0, log_alpha0, log_mu0, beta0, eta0, initial0)
    ]
    rq, rk, rv, ra, rm, rb, re, ri = reference_inputs
    reference_output, reference_state = predictive_momentum_reference(
        rq,
        rk,
        rv,
        ra,
        rm,
        rb,
        re,
        initial_state=ri,
    )
    reference_loss = (
        reference_output.float().mul(output_weight).sum()
        + reference_state.float().mul(state_weight).sum()
    )
    reference_loss.backward()

    labels = ("q", "k", "v", "log_alpha", "log_mu", "beta", "eta", "initial")
    gradient_parity = {
        name: _relative_rms(custom.grad, reference.grad)
        for name, custom, reference in zip(labels, custom_inputs, reference_inputs)
    }
    parity = {
        "output_relative_rms": _relative_rms(custom_output, reference_output),
        "state_relative_rms": _relative_rms(custom_state, reference_state),
        "gradient_relative_rms": gradient_parity,
        "minimum_inverse_denominator": float(custom_stats[..., 2].min()),
    }
    if (
        parity["output_relative_rms"] > 2e-4
        or parity["state_relative_rms"] > 2e-4
        or max(gradient_parity.values()) > 5e-3
        or parity["minimum_inverse_denominator"] < 0.05
    ):
        raise RuntimeError(f"Predictive Triton/Torch parity failed: {parity}")
    return parity


def _structural_checks(external_recurrent) -> dict[str, Any]:
    generator = torch.Generator(device="cuda").manual_seed(62063)
    shape = (2, 17, 2)
    q = torch.randn(*shape, 8, generator=generator, device="cuda", dtype=torch.bfloat16)
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    log_alpha = -0.10 - 0.15 * torch.rand(*shape, generator=generator, device="cuda")
    log_mu = -0.10 - 0.15 * torch.rand(*shape, generator=generator, device="cuda")
    beta = 0.05 + 0.15 * torch.rand(*shape, generator=generator, device="cuda")
    eta = 0.50 + 0.25 * torch.rand(*shape, generator=generator, device="cuda")
    initial = 0.02 * torch.randn(
        2, 2, 2, 8, 8, generator=generator, device="cuda"
    )

    with torch.no_grad():
        beta_zero = torch.zeros_like(beta)
        predictive_zero, predictive_zero_state, _ = predictive_momentum_rule(
            q=q,
            k=k,
            v=v,
            log_alpha=log_alpha,
            log_mu=log_mu,
            beta=beta_zero,
            eta=eta,
            initial_state=initial,
            output_final_state=True,
        )
        parent_zero, parent_zero_state = external_recurrent(
            q=q,
            k=k,
            v=v,
            log_alpha=log_alpha,
            log_mu=log_mu,
            beta=beta_zero,
            eta=eta,
            initial_state=initial,
            output_final_state=True,
        )

        one_predictive, one_predictive_state, _ = predictive_momentum_rule(
            q=q[:, :1],
            k=k[:, :1],
            v=v[:, :1],
            log_alpha=log_alpha[:, :1],
            log_mu=log_mu[:, :1],
            beta=beta[:, :1],
            eta=eta[:, :1],
            initial_state=torch.stack((initial[0], torch.zeros_like(initial[1]))),
            output_final_state=True,
        )
        one_parent, one_parent_state = external_recurrent(
            q=q[:, :1],
            k=k[:, :1],
            v=v[:, :1],
            log_alpha=log_alpha[:, :1],
            log_mu=log_mu[:, :1],
            beta=beta[:, :1],
            eta=eta[:, :1],
            initial_state=torch.stack((initial[0], torch.zeros_like(initial[1]))),
            output_final_state=True,
        )

        permutation = torch.tensor([1, 0], device="cuda")
        permuted_output, permuted_state, _ = predictive_momentum_rule(
            q=q[:, :, permutation],
            k=k[:, :, permutation],
            v=v[:, :, permutation],
            log_alpha=log_alpha[:, :, permutation],
            log_mu=log_mu[:, :, permutation],
            beta=beta[:, :, permutation],
            eta=eta[:, :, permutation],
            initial_state=initial[:, :, permutation],
            output_final_state=True,
        )
        native_output, native_state, _ = predictive_momentum_rule(
            q=q,
            k=k,
            v=v,
            log_alpha=log_alpha,
            log_mu=log_mu,
            beta=beta,
            eta=eta,
            initial_state=initial,
            output_final_state=True,
        )
        changed_v = v.clone()
        changed_v[:, 9:] = torch.randn(
            changed_v[:, 9:].shape,
            generator=generator,
            device="cuda",
            dtype=changed_v.dtype,
        )
        changed_output, _changed_state, _ = predictive_momentum_rule(
            q=q,
            k=k,
            v=changed_v,
            log_alpha=log_alpha,
            log_mu=log_mu,
            beta=beta,
            eta=eta,
            initial_state=initial,
            output_final_state=True,
        )

    checks = {
        "beta_zero_output_relative_rms": _relative_rms(predictive_zero, parent_zero),
        "beta_zero_state_relative_rms": _relative_rms(predictive_zero_state, parent_zero_state),
        "zero_old_momentum_output_relative_rms": _relative_rms(one_predictive, one_parent),
        "zero_old_momentum_state_relative_rms": _relative_rms(one_predictive_state, one_parent_state),
        "head_permutation_output_relative_rms": _relative_rms(
            permuted_output, native_output[:, :, permutation]
        ),
        "head_permutation_state_relative_rms": _relative_rms(
            permuted_state, native_state[:, :, permutation]
        ),
        "causal_prefix_max_abs": float(
            (native_output[:, :9].float() - changed_output[:, :9].float()).abs().max()
        ),
    }
    if (
        checks["beta_zero_output_relative_rms"] > 0.02
        or checks["beta_zero_state_relative_rms"] > 0.02
        or checks["zero_old_momentum_output_relative_rms"] > 0.02
        or checks["zero_old_momentum_state_relative_rms"] > 0.02
        or checks["head_permutation_output_relative_rms"] > 1e-6
        or checks["head_permutation_state_relative_rms"] > 1e-6
        or checks["causal_prefix_max_abs"] != 0.0
    ):
        raise RuntimeError(f"Predictive structural contract failed: {checks}")
    return checks


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
        raise RuntimeError("P-GDN3-062 requires exactly CUDA index 0")
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
    repo_root = Path(os.environ["MDN_REPO_ROOT"]).resolve()
    fla_root = Path(os.environ["MDN_FLA_ROOT"]).resolve()
    if subprocess.check_output(
        ["git", "-C", str(repo_root), "rev-parse", "HEAD"], text=True
    ).strip() != EXPECTED_MDN_SHA:
        raise RuntimeError("External Momentum SHA changed")
    source_hashes = {
        "layer": _sha256(Path(inspect.getfile(layer_class)).resolve()),
        "chunk": _sha256(fla_root / "fla" / "ops" / "momentum_delta_rule" / "chunk.py"),
        "fused_recurrent": _sha256(
            fla_root / "fla" / "ops" / "momentum_delta_rule" / "fused_recurrent.py"
        ),
    }
    if source_hashes != {
        "layer": EXPECTED_LAYER_SHA256,
        "chunk": EXPECTED_CHUNK_SHA256,
        "fused_recurrent": EXPECTED_RECURRENT_SHA256,
    }:
        raise RuntimeError(f"External source drifted: {source_hashes}")
    external_recurrent = importlib.import_module(
        "fla.ops.momentum_delta_rule"
    ).fused_recurrent_mode_rule

    synthetic = _synthetic_parity()
    structural = _structural_checks(external_recurrent)

    config = build_config(
        arm="future_seed_predictive_momentum",
        sequence_length=1024,
        num_kv_pairs=4,
        max_epochs=10,
        batch_size=32,
    )
    train_loader, test_loader = prepare_data(config.data)
    data_hashes = {
        "train": dataset_hash(train_loader),
        "test": dataset_hash(test_loader),
    }
    if data_hashes != {"train": EXPECTED_TRAIN_HASH, "test": EXPECTED_TEST_HASH}:
        raise RuntimeError(f"Directional MQAR data drifted: {data_hashes}")
    if (
        not args.matched_init.is_file()
        or _sha256(args.matched_init) != EXPECTED_MATCHED_INIT_SHA256
    ):
        raise RuntimeError(f"Matched initialization drifted: {args.matched_init}")
    parent_state = torch.load(args.matched_init, map_location="cpu", weights_only=True)
    set_determinism(123)
    model = make_model(config, "future_seed_predictive_momentum")
    load_matched_parent_state(model, parent_state)
    if sum(parameter.numel() for parameter in model.parameters()) != EXPECTED_PARAMETERS:
        raise RuntimeError("Predictive Momentum parameter count changed")
    if parameter_hash(model) != EXPECTED_INITIAL_PARAMETER_HASH:
        raise RuntimeError("Predictive Momentum initialization hash changed")
    metadata = model._momentum_parent_metadata
    if metadata["loaded_hash"] != EXPECTED_PARENT_PARAMETER_HASH:
        raise RuntimeError(f"Shared parent mapping changed: {metadata}")
    mixers = [block.sequence_mixer for block in model.backbone.layers]
    if len(mixers) != 2 or not all(
        type(mixer) is ZoologyPredictiveMomentumFutureSeedMixer for mixer in mixers
    ):
        raise RuntimeError("Contract requires two exact predictive mixers")
    if any(mixer.state_size() != EXPECTED_STATE_VALUES_PER_LAYER for mixer in mixers):
        raise RuntimeError("Predictive Momentum state geometry changed")

    inputs, targets, _slices = next(iter(test_loader))
    inputs = inputs[:4].cuda()
    targets = targets[:4].cuda()
    model = model.cuda().train()
    logits = model(inputs)
    mask = targets != -100
    loss = F.cross_entropy(logits[mask], targets[mask])
    graph_names = backward_names(loss)
    backward_count = sum(
        "PredictiveMomentumFunctionBackward" in name for name in graph_names
    )
    if backward_count != 2:
        raise RuntimeError(f"Expected two predictive backward paths, got {backward_count}")
    loss.backward()
    gradient_rows = []
    for index, mixer in enumerate(mixers):
        layer = mixer.layer
        gradient_rows.append(
            {
                "layer": index,
                "q": _gradient_rms(layer.q_proj.weight, f"layer{index}.q"),
                "k": _gradient_rms(layer.k_proj.weight, f"layer{index}.k"),
                "v": _gradient_rms(layer.v_proj.weight, f"layer{index}.v"),
                "alpha": _gradient_rms(layer.a_proj.weight, f"layer{index}.alpha"),
                "mu": _gradient_rms(layer.m_proj.weight, f"layer{index}.mu"),
                "beta": _gradient_rms(layer.b_proj.weight, f"layer{index}.beta"),
                "eta": _gradient_rms(layer.e_proj.weight, f"layer{index}.eta"),
                "future_seed": (
                    None
                    if index == 0
                    else _gradient_rms(mixer.future_seed_logit, "future_seed")
                ),
            }
        )

    model.eval()
    with torch.no_grad():
        evaluation_logits = model(inputs)
    if not torch.isfinite(evaluation_logits).all():
        raise RuntimeError("Nonfinite full-stack evaluation")
    diagnostic_rows = [mixer.last_predictive_stats for mixer in mixers]
    if any(row is None for row in diagnostic_rows):
        raise RuntimeError("Predictive diagnostics were not collected")
    if any(
        row["finite"] != 1
        or row["lookahead_relative_rms"] < 0.01
        or row["residual_change_relative_rms"] < 0.01
        or row["minimum_inverse_denominator"] < 0.05
        for row in diagnostic_rows
    ):
        raise RuntimeError(f"Predictive activation/stability failed: {diagnostic_rows}")

    result = {
        "status": "passed",
        "plan": "P-GDN3-062",
        "gpu": {"uuid": gpu_uuid, "name": gpu_name},
        "external": {
            "sha": EXPECTED_MDN_SHA,
            "source_hashes": source_hashes,
            "redistributed_source": False,
        },
        "host_fla_compatibility": compatibility,
        "kernel": {
            "implementation": "clean_room_fused_reversible_predictive_momentum",
            "backward_paths": backward_count,
            "one_scan_per_layer": True,
        },
        "data_hashes": data_hashes,
        "matched_init_sha256": EXPECTED_MATCHED_INIT_SHA256,
        "parameters": EXPECTED_PARAMETERS,
        "state_values_per_layer": EXPECTED_STATE_VALUES_PER_LAYER,
        "matched_parent": metadata,
        "initial_parameter_hash": parameter_hash(model),
        "synthetic_parity": synthetic,
        "structural_checks": structural,
        "full_stack_gradients": gradient_rows,
        "full_stack_diagnostics": diagnostic_rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
