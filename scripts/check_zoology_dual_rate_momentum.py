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
    load_external_momentum_layer,
    load_matched_parent_state,
    momentum_fla_compatibility,
)
from experiments.zoology_mqar.momentum_dual_rate import (
    STATE_COMPONENTS,
    ZoologyDualRateMomentumFutureSeedMixer,
    dual_rate_momentum_reference,
    dual_rate_momentum_rule,
)
from scripts.check_zoology_contractive_dplr import (
    backward_names,
    normalized_uuid,
)


EXPECTED_TRAIN_HASH = "647c64ece84984a23dfd817c4f277ea83840dbec57cc18bb6c9bf9eda7cc9a68"
EXPECTED_TEST_HASH = "4a8237ba8fe19aaff0d1d72de7b7f6505eaab59cd091442c2f463df34cce278f"
EXPECTED_PARAMETERS = 600_696
EXPECTED_STATE_VALUES_PER_LAYER = 12_288
EXPECTED_MATCHED_INIT_SHA256 = (
    "7402e46c48cbd47070d65262d1a1b62a32ac55a4d644a6fb9644b96b0914850f"
)
EXPECTED_PARENT_PARAMETER_HASH = (
    "0adf26f657e59a35e90d7e54b905721b10f537b6391d107676248de16b71f13f"
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
    mix0 = 0.20 * torch.tanh(
        torch.randn(*shape, generator=generator, device="cuda")
    )
    initial0 = 0.03 * torch.randn(
        STATE_COMPONENTS,
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
        for value in (
            q0, k0, v0, log_alpha0, log_mu0, beta0, eta0, mix0, initial0
        )
    ]
    q, k, v, log_alpha, log_mu, beta, eta, mix, initial = custom_inputs
    custom_output, custom_state, custom_stats = dual_rate_momentum_rule(
        q=q,
        k=k,
        v=v,
        log_alpha=log_alpha,
        log_mu=log_mu,
        beta=beta,
        eta=eta,
        initial_state=initial,
        output_final_state=True,
        dual_rate_mix=mix,
    )
    custom_loss = (
        custom_output.float().mul(output_weight).sum()
        + custom_state.float().mul(state_weight).sum()
    )
    custom_loss.backward()

    reference_inputs = [
        _leaf(value)
        for value in (
            q0, k0, v0, log_alpha0, log_mu0, beta0, eta0, mix0, initial0
        )
    ]
    rq, rk, rv, ra, rm, rb, re, rx, ri = reference_inputs
    reference_output, reference_state = dual_rate_momentum_reference(
        rq,
        rk,
        rv,
        ra,
        rm,
        rb,
        re,
        rx,
        initial_state=ri,
    )
    reference_loss = (
        reference_output.float().mul(output_weight).sum()
        + reference_state.float().mul(state_weight).sum()
    )
    reference_loss.backward()

    labels = (
        "q", "k", "v", "log_alpha", "log_mu", "beta", "eta", "mix", "initial"
    )
    gradient_parity = {
        name: _relative_rms(custom.grad, reference.grad)
        for name, custom, reference in zip(labels, custom_inputs, reference_inputs)
    }
    parity = {
        "output_relative_rms": _relative_rms(custom_output, reference_output),
        "state_relative_rms": _relative_rms(custom_state, reference_state),
        "gradient_relative_rms": gradient_parity,
        "band_to_slow_relative_rms": float(custom_stats[..., 0].mean()),
        "injection_relative_rms": float(custom_stats[..., 1].mean()),
        "mean_abs_mix": float(custom_stats[..., 2].mean()),
        "stats_finite": bool(torch.isfinite(custom_stats).all()),
    }
    if (
        parity["output_relative_rms"] > 2e-4
        or parity["state_relative_rms"] > 2e-4
        or max(gradient_parity.values()) > 5e-3
        or not parity["stats_finite"]
        or parity["band_to_slow_relative_rms"] <= 0.0
        or parity["injection_relative_rms"] <= 0.0
        or parity["mean_abs_mix"] <= 0.0
    ):
        raise RuntimeError(f"Dual-rate Triton/Torch parity failed: {parity}")
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
    initial_parent = 0.02 * torch.randn(
        2, 2, 2, 8, 8, generator=generator, device="cuda"
    )
    initial = torch.cat((initial_parent, initial_parent[1:2].clone()), dim=0)
    zero_mix = torch.zeros(shape, device="cuda")
    active_mix = 0.25 * torch.tanh(
        torch.randn(*shape, generator=generator, device="cuda")
    )

    with torch.no_grad():
        parent_limit, parent_limit_state, _ = dual_rate_momentum_rule(
            q=q,
            k=k,
            v=v,
            log_alpha=log_alpha,
            log_mu=log_mu,
            beta=beta,
            eta=eta,
            initial_state=initial,
            output_final_state=True,
            dual_rate_mix=zero_mix,
        )
        parent_output, parent_state = external_recurrent(
            q=q,
            k=k,
            v=v,
            log_alpha=log_alpha,
            log_mu=log_mu,
            beta=beta,
            eta=eta,
            initial_state=initial_parent,
            output_final_state=True,
        )

        permutation = torch.tensor([1, 0], device="cuda")
        permuted_output, permuted_state, _ = dual_rate_momentum_rule(
            q=q[:, :, permutation],
            k=k[:, :, permutation],
            v=v[:, :, permutation],
            log_alpha=log_alpha[:, :, permutation],
            log_mu=log_mu[:, :, permutation],
            beta=beta[:, :, permutation],
            eta=eta[:, :, permutation],
            initial_state=initial[:, :, permutation],
            output_final_state=True,
            dual_rate_mix=active_mix[:, :, permutation],
        )
        native_output, native_state, _ = dual_rate_momentum_rule(
            q=q,
            k=k,
            v=v,
            log_alpha=log_alpha,
            log_mu=log_mu,
            beta=beta,
            eta=eta,
            initial_state=initial,
            output_final_state=True,
            dual_rate_mix=active_mix,
        )
        perturbed_initial = initial.clone()
        perturbed_initial[2] = perturbed_initial[2] + 0.1 * torch.randn(
            perturbed_initial[2].shape,
            generator=generator,
            device="cuda",
        )
        perturbed_output, _perturbed_state, _ = dual_rate_momentum_rule(
            q=q[:, :1],
            k=k[:, :1],
            v=v[:, :1],
            log_alpha=log_alpha[:, :1],
            log_mu=log_mu[:, :1],
            beta=beta[:, :1],
            eta=eta[:, :1],
            initial_state=perturbed_initial,
            output_final_state=True,
            dual_rate_mix=active_mix[:, :1],
        )
        unperturbed_output, _unperturbed_state, _ = dual_rate_momentum_rule(
            q=q[:, :1],
            k=k[:, :1],
            v=v[:, :1],
            log_alpha=log_alpha[:, :1],
            log_mu=log_mu[:, :1],
            beta=beta[:, :1],
            eta=eta[:, :1],
            initial_state=initial,
            output_final_state=True,
            dual_rate_mix=active_mix[:, :1],
        )
        changed_v = v.clone()
        changed_v[:, 9:] = torch.randn(
            changed_v[:, 9:].shape,
            generator=generator,
            device="cuda",
            dtype=changed_v.dtype,
        )
        changed_output, _changed_state, _ = dual_rate_momentum_rule(
            q=q,
            k=k,
            v=changed_v,
            log_alpha=log_alpha,
            log_mu=log_mu,
            beta=beta,
            eta=eta,
            initial_state=initial,
            output_final_state=True,
            dual_rate_mix=active_mix,
        )

    checks = {
        "zero_mix_parent_output_relative_rms": _relative_rms(
            parent_limit, parent_output
        ),
        "zero_mix_parent_state_relative_rms": _relative_rms(
            parent_limit_state[:2], parent_state
        ),
        "head_permutation_output_relative_rms": _relative_rms(
            permuted_output, native_output[:, :, permutation]
        ),
        "head_permutation_state_relative_rms": _relative_rms(
            permuted_state, native_state[:, :, permutation]
        ),
        "fast_state_dependency_relative_rms": _relative_rms(
            perturbed_output, unperturbed_output
        ),
        "active_mix_output_change_relative_rms": _relative_rms(
            native_output, parent_limit
        ),
        "fast_slow_terminal_difference_relative_rms": _relative_rms(
            native_state[2], native_state[1]
        ),
        "fast_state_first_output_max_abs": float(
            (perturbed_output.float() - unperturbed_output.float()).abs().max()
        ),
        "causal_prefix_max_abs": float(
            (native_output[:, :9].float() - changed_output[:, :9].float()).abs().max()
        ),
    }
    if (
        checks["zero_mix_parent_output_relative_rms"] > 0.02
        or checks["zero_mix_parent_state_relative_rms"] > 0.02
        or checks["head_permutation_output_relative_rms"] > 1e-6
        or checks["head_permutation_state_relative_rms"] > 1e-6
        or checks["fast_state_dependency_relative_rms"] < 1e-4
        or checks["active_mix_output_change_relative_rms"] < 1e-4
        or checks["fast_slow_terminal_difference_relative_rms"] < 1e-4
        or checks["fast_state_first_output_max_abs"] <= 0.0
        or checks["causal_prefix_max_abs"] != 0.0
    ):
        raise RuntimeError(f"Dual-rate structural contract failed: {checks}")
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
        raise RuntimeError("P-GDN3-070 requires exactly CUDA index 0")
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
        arm="future_seed_dual_rate_momentum",
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
    model = make_model(config, "future_seed_dual_rate_momentum")
    load_matched_parent_state(model, parent_state)
    if sum(parameter.numel() for parameter in model.parameters()) != EXPECTED_PARAMETERS:
        raise RuntimeError("Dual-Rate Momentum parameter count changed")
    metadata = model._momentum_parent_metadata
    if metadata["loaded_hash"] != EXPECTED_PARENT_PARAMETER_HASH:
        raise RuntimeError(f"Shared parent mapping changed: {metadata}")
    mixers = [block.sequence_mixer for block in model.backbone.layers]
    if len(mixers) != 2 or not all(
        type(mixer) is ZoologyDualRateMomentumFutureSeedMixer for mixer in mixers
    ):
        raise RuntimeError("Contract requires two exact dual_rate mixers")
    if any(mixer.state_size() != EXPECTED_STATE_VALUES_PER_LAYER for mixer in mixers):
        raise RuntimeError("Dual-Rate Momentum state geometry changed")
    if any(
        mixer.dual_rate_weight.shape != (4, 128)
        or torch.count_nonzero(mixer.dual_rate_weight).item() != 0
        for mixer in mixers
    ):
        raise RuntimeError("Dual-rate zero-init parameter contract changed")
    initial_parameter_hash = parameter_hash(model)

    inputs, targets, _slices = next(iter(test_loader))
    inputs = inputs[:4].cuda()
    targets = targets[:4].cuda()
    model = model.cuda().train()
    logits = model(inputs)
    mask = targets != -100
    loss = F.cross_entropy(logits[mask], targets[mask])
    graph_names = backward_names(loss)
    backward_count = sum(
        "DualRateMomentumFunctionBackward" in name for name in graph_names
    )
    if backward_count != 2:
        raise RuntimeError(f"Expected two dual_rate backward paths, got {backward_count}")
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
                "dual_rate_mix": _gradient_rms(
                    mixer.dual_rate_weight, f"layer{index}.dual_rate_mix"
                ),
                "future_seed": (
                    None
                    if index == 0
                    else _gradient_rms(mixer.future_seed_logit, "future_seed")
                ),
            }
        )

    with torch.no_grad():
        for mixer in mixers:
            gradient = mixer.dual_rate_weight.grad.float()
            scale = gradient.square().mean().sqrt().clamp_min(1e-8)
            mixer.dual_rate_weight.add_(-0.01 * gradient / scale)

    model.eval()
    with torch.no_grad():
        evaluation_logits = model(inputs)
    if not torch.isfinite(evaluation_logits).all():
        raise RuntimeError("Nonfinite full-stack evaluation")
    diagnostic_rows = [mixer.last_dual_rate_stats for mixer in mixers]
    if any(row is None for row in diagnostic_rows):
        raise RuntimeError("Dual-rate diagnostics were not collected")
    if any(
        row["finite"] != 1
        or row["band_to_slow_relative_rms"] < 0.01
        or row["injection_relative_rms"] < 1e-4
        or row["injection_board_std"] <= 0.0
        or row["injection_token_std"] <= 0.0
        or row["mean_abs_mix"] < 1e-4
        or row["max_abs_mix"] > 0.95
        for row in diagnostic_rows
    ):
        raise RuntimeError(f"Dual-rate activation/stability failed: {diagnostic_rows}")

    result = {
        "status": "passed",
        "plan": "P-GDN3-070",
        "gpu": {"uuid": gpu_uuid, "name": gpu_name},
        "external": {
            "sha": EXPECTED_MDN_SHA,
            "source_hashes": source_hashes,
            "redistributed_source": False,
        },
        "host_fla_compatibility": compatibility,
        "kernel": {
            "implementation": "clean_room_fused_checkpointed_dual_rate_momentum",
            "checkpoint_tokens": 8,
            "backward_paths": backward_count,
            "one_scan_per_layer": True,
        },
        "data_hashes": data_hashes,
        "matched_init_sha256": EXPECTED_MATCHED_INIT_SHA256,
        "parameters": EXPECTED_PARAMETERS,
        "state_values_per_layer": EXPECTED_STATE_VALUES_PER_LAYER,
        "state_components": STATE_COMPONENTS,
        "matched_parent": metadata,
        "initial_parameter_hash": initial_parameter_hash,
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
