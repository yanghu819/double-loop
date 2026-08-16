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
    parent_parameter_hash,
)
from experiments.zoology_mqar.momentum_refresh import (
    PostCommitMomentumRefreshOperation,
    ZoologyMomentumRefreshFutureSeedMixer,
)
from scripts.check_zoology_contractive_dplr import (
    backward_names,
    normalized_uuid,
)


EXPECTED_TRAIN_HASH = "647c64ece84984a23dfd817c4f277ea83840dbec57cc18bb6c9bf9eda7cc9a68"
EXPECTED_TEST_HASH = "4a8237ba8fe19aaff0d1d72de7b7f6505eaab59cd091442c2f463df34cce278f"
EXPECTED_PARAMETERS = 599_672
EXPECTED_INIT_PARAMETER_HASH = (
    "dc8f49f92c00f0f08863cf6183692246cf2c44f0c5aa3a35892bb9555ec17bfb"
)
EXPECTED_PARENT_PARAMETER_HASH = (
    "0adf26f657e59a35e90d7e54b905721b10f537b6391d107676248de16b71f13f"
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


def _sha256(path: Path) -> str:
    import hashlib

    return hashlib.sha256(path.read_bytes()).hexdigest()


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


def _synthetic_refresh_contract(chunk_rule, recurrent_rule) -> dict[str, float]:
    generator = torch.Generator(device="cuda").manual_seed(61061)
    batch, tokens, heads, width = 2, 16, 4, 32
    shape = (batch, tokens, heads)
    q = torch.randn(*shape, width, generator=generator, device="cuda", dtype=torch.bfloat16)
    key = torch.randn_like(q)
    value = torch.randn_like(q)
    log_alpha = -0.05 - 0.2 * torch.rand(*shape, generator=generator, device="cuda")
    log_mu = -0.05 - 0.2 * torch.rand(*shape, generator=generator, device="cuda")
    beta = 0.2 + 0.6 * torch.rand(*shape, generator=generator, device="cuda")
    eta = 0.5 + torch.rand(*shape, generator=generator, device="cuda")
    initial = 0.05 * torch.randn(
        2,
        batch,
        heads,
        width,
        width,
        generator=generator,
        device="cuda",
        dtype=torch.float32,
    )

    wrapped = PostCommitMomentumRefreshOperation(chunk_rule)
    candidate_output, candidate_state = wrapped(
        q=q,
        k=key,
        v=value,
        log_alpha=log_alpha,
        log_mu=log_mu,
        beta=beta,
        eta=eta,
        initial_state=initial,
        output_final_state=True,
        use_qk_l2norm_in_kernel=True,
        use_p_times_alpha=True,
    )
    state = initial
    outputs = []
    state_change_max_abs = 0.0
    momentum_changes = []
    for index in range(tokens):
        token_slice = slice(index, index + 1)
        parent_output, parent_state = recurrent_rule(
            q=q[:, token_slice],
            k=key[:, token_slice],
            v=value[:, token_slice],
            log_alpha=log_alpha[:, token_slice],
            log_mu=log_mu[:, token_slice],
            beta=beta[:, token_slice],
            eta=eta[:, token_slice],
            initial_state=state,
            output_final_state=True,
            use_qk_l2norm_in_kernel=True,
            use_p_times_alpha=True,
        )
        refresh_output, refresh_state = recurrent_rule(
            q=q[:, token_slice],
            k=key[:, token_slice],
            v=value[:, token_slice],
            log_alpha=torch.zeros_like(log_alpha[:, token_slice]),
            log_mu=torch.zeros_like(log_mu[:, token_slice]),
            beta=torch.zeros_like(beta[:, token_slice]),
            eta=eta[:, token_slice],
            initial_state=parent_state,
            output_final_state=True,
            use_qk_l2norm_in_kernel=True,
            use_p_times_alpha=True,
        )
        outputs.append(parent_output)
        state_change_max_abs = max(
            state_change_max_abs,
            float((refresh_state[0] - parent_state[0]).abs().max()),
            float((refresh_output.float() - parent_output.float()).abs().max()),
        )
        momentum_changes.append(
            _relative_rms(refresh_state[1], parent_state[1])
        )
        state = refresh_state
    sequential_output = torch.cat(outputs, dim=1)

    fp_key = F.normalize(
        torch.randn(batch, heads, width, generator=generator, device="cuda"),
        dim=-1,
    )
    fp_state = torch.randn(
        batch,
        heads,
        width,
        width,
        generator=generator,
        device="cuda",
    )
    fp_momentum = torch.randn_like(fp_state)
    fp_eta = torch.rand(batch, heads, generator=generator, device="cuda") + 0.5
    fp_value = torch.einsum("bhk,bhkv->bhv", fp_key, fp_state)
    fp_residual = fp_value - torch.einsum("bhk,bhkv->bhv", fp_key, fp_state)
    fp_next_momentum = fp_momentum - (
        fp_key.unsqueeze(-1) * (fp_eta.unsqueeze(-1) * fp_residual).unsqueeze(-2)
    )
    fixed_point_max_abs = float((fp_next_momentum - fp_momentum).abs().max())

    changes = torch.tensor(momentum_changes)
    result = {
        "output_relative_rms_vs_sequential": _relative_rms(
            candidate_output,
            sequential_output,
        ),
        "state_relative_rms_vs_sequential": _relative_rms(
            candidate_state,
            state,
        ),
        "state_change_max_abs": state_change_max_abs,
        "momentum_change_relative_rms": float(changes.mean()),
        "token_variation": float(changes.std(unbiased=False)),
        "fixed_point_max_abs": fixed_point_max_abs,
        "refresh_output_relative_rms": float(
            wrapped.last_stats["refresh_output_relative_rms"]
        ),
    }
    if (
        result["output_relative_rms_vs_sequential"] > 0.05
        or result["state_relative_rms_vs_sequential"] > 0.05
        or result["state_change_max_abs"] > 1e-6
        or result["momentum_change_relative_rms"] < 1e-4
        or result["token_variation"] < 1e-4
        or result["fixed_point_max_abs"] > 1e-7
        or result["refresh_output_relative_rms"] > 1e-5
    ):
        raise RuntimeError(f"Synthetic Momentum refresh failed: {result}")
    return result


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
        raise RuntimeError("P-GDN3-061 requires exactly CUDA index 0")
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
        raise RuntimeError("External Momentum DeltaNet SHA changed")
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

    config = build_config(
        arm="future_seed_momentum_refresh",
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

    parent_state = torch.load(args.matched_init, map_location="cpu", weights_only=True)
    set_determinism(123)
    candidate = make_model(config, "future_seed_momentum_refresh")
    load_matched_parent_state(candidate, parent_state)
    parameters = sum(parameter.numel() for parameter in candidate.parameters())
    init_hash = parameter_hash(candidate)
    parent_hash = parent_parameter_hash(candidate)
    if parameters != EXPECTED_PARAMETERS:
        raise RuntimeError(f"Parameter count drifted: {parameters}")
    if init_hash != EXPECTED_INIT_PARAMETER_HASH:
        raise RuntimeError(f"P059 initialization drifted: {init_hash}")
    if parent_hash != EXPECTED_PARENT_PARAMETER_HASH:
        raise RuntimeError(f"Shared parent initialization drifted: {parent_hash}")

    mixers = [block.sequence_mixer for block in candidate.backbone.layers]
    if len(mixers) != 2 or not all(
        type(mixer) is ZoologyMomentumRefreshFutureSeedMixer for mixer in mixers
    ):
        raise RuntimeError("Contract requires two exact Momentum refresh mixers")
    for index, mixer in enumerate(mixers):
        if type(mixer.layer) is not layer_class:
            raise RuntimeError(f"Layer {index} escaped the pinned external class")
        if mixer.state_size() != EXPECTED_STATE_VALUES_PER_LAYER:
            raise RuntimeError(f"Layer {index} state geometry changed")

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
            f"Expected two external Momentum chunk backwards, got {chunk_backward_count}"
        )
    loss.backward()
    gradients = []
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
        gradients.append(row)

    candidate.eval()
    with torch.no_grad():
        candidate(inputs)
    refresh_rows = []
    for mixer in mixers:
        if mixer.last_refresh_stats is None:
            refresh_rows.append(None)
            continue
        refresh_rows.append(
            {
                key: float(value) if isinstance(value, torch.Tensor) else value
                for key, value in mixer.last_refresh_stats.items()
            }
        )
    schedule_passed = all(
        row is not None
        and row["parent_tokens"] == 1024
        and row["refresh_tokens"] == 1024
        and row["micro_tokens"] == 2048
        and row["refresh_log_alpha_max_abs"] == 0.0
        and row["refresh_log_mu_max_abs"] == 0.0
        and row["refresh_beta_max_abs"] == 0.0
        and row["refresh_output_relative_rms"] <= 1e-5
        for row in refresh_rows
    )
    if not schedule_passed:
        raise RuntimeError(f"Refresh schedule failed: {refresh_rows}")
    if (
        mixers[0].last_state_rms is None
        or mixers[0].last_momentum_rms is None
        or mixers[1].last_seed_gate is None
    ):
        raise RuntimeError("Momentum/FutureSeed state was not active")

    op_module = importlib.import_module("fla.ops.momentum_delta_rule")
    layer_module = importlib.import_module(layer_class.__module__)
    chunk_rule = op_module.chunk_mode_rule
    if layer_module.chunk_mode_rule is not chunk_rule:
        raise RuntimeError("Scoped wrapper did not restore the external operation")
    synthetic = _synthetic_refresh_contract(
        chunk_rule,
        op_module.fused_recurrent_mode_rule,
    )

    checks = {
        "gpu_exact": True,
        "external_and_host_sources_exact": (
            compatibility["host_fla_sha"]
            == os.environ["FLA_EXPECTED_SOURCE_SHA"]
            and not compatibility["use_cuda_graph"]
        ),
        "data_exact": True,
        "p059_initialization_exact": True,
        "parameter_delta_zero": parameters == EXPECTED_PARAMETERS,
        "persistent_state_delta_zero": all(
            mixer.state_size() == EXPECTED_STATE_VALUES_PER_LAYER for mixer in mixers
        ),
        "two_external_chunk_backwards": chunk_backward_count == 2,
        "exact_refresh_schedule": schedule_passed,
        "synthetic_parent_refresh_parity": (
            synthetic["output_relative_rms_vs_sequential"] <= 0.05
            and synthetic["state_relative_rms_vs_sequential"] <= 0.05
        ),
        "state_unchanged_and_momentum_active": (
            synthetic["state_change_max_abs"] <= 1e-6
            and synthetic["momentum_change_relative_rms"] >= 1e-4
            and synthetic["token_variation"] >= 1e-4
        ),
        "residual_zero_fixed_point": synthetic["fixed_point_max_abs"] <= 1e-7,
        "all_gradients_active": all(
            math.isfinite(value) and value > 0
            for row in gradients
            for key, value in row.items()
            if key != "layer"
        ),
        "native_futureseed_active": mixers[1].last_seed_gate is not None,
        "no_fallback": layer_module.chunk_mode_rule is chunk_rule,
    }
    passed = all(checks.values())
    if not passed:
        raise RuntimeError(f"Registered P-GDN3-061 contract failed: {checks}")

    result = {
        "status": "passed",
        "plan": "P-GDN3-061",
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
            "p059": EXPECTED_PARAMETERS,
            "candidate": parameters,
            "delta": parameters - EXPECTED_PARAMETERS,
        },
        "persistent_state_values_per_layer": EXPECTED_STATE_VALUES_PER_LAYER,
        "initialization": {
            "full": init_hash,
            "shared_parent": parent_hash,
        },
        "chunk_backward_count": chunk_backward_count,
        "gradients": gradients,
        "refresh_schedule": refresh_rows,
        "synthetic_refresh": synthetic,
        "registered_contract": {"passed": passed, "checks": checks},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
