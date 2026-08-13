from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import os
import subprocess
from pathlib import Path

import torch
import torch.nn.functional as F

from fla.layers.gdn2 import GatedDeltaNet2
from fla.modules.convolution import ShortConvolution
from zoology.data.utils import prepare_data
from zoology.utils import set_determinism

from experiments.zoology_mqar.gdn2_committed_delta import capture_committed_edit
from experiments.zoology_mqar.gdn2_futureseed import PINNED_FLA_SHA
from experiments.zoology_mqar.gdn2_surprise_regression_seed import (
    RIDGE_SCALE,
    SurpriseRegressionFutureSeedLanguageModel,
    ZoologySurpriseRegressionGDN2FutureSeedMixer,
    surprise_regression_diagnostics,
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
EXPECTED_FLA_ROOT = Path(
    "/huyang2/double-loop/.cache/fla-versions/"
    "9c8e42e762fce087c27b673af4922795d9edb85e-0280db310981915e"
)
EXPECTED_GDN2_SHA256 = "4d001b6a8903cc7acb42b908ab3ade0f1edca0a17275630fc9c080c0232bf910"
EXPECTED_GDN2_OPS_SHA256 = "b9267a516290a30b72be06954651b4221a1191402531a9749a99e50b14f8944c"
CONTROL_ARM = "future_seed_gdn2"
CANDIDATE_ARM = "future_seed_gdn2_surprise_regression"
MODEL_LAYERS = 2
HEADS = 4
KEY_DIM = 32
VALUE_DIM = 32


def normalized_uuid(value: str) -> str:
    return value.removeprefix("GPU-").lower()


def git_head(path: Path) -> str:
    return subprocess.check_output(
        ["git", "-C", str(path), "rev-parse", "HEAD"], text=True
    ).strip()


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def python_tree_hash(path: Path) -> str:
    digest = hashlib.sha256()
    sources = sorted(path.rglob("*.py"))
    if not sources:
        raise RuntimeError(f"No Python sources under {path}")
    for source in sources:
        digest.update(str(source.relative_to(path)).encode())
        digest.update(source.read_bytes())
    return digest.hexdigest()


def max_diff(left: torch.Tensor, right: torch.Tensor, label: str) -> float:
    difference = (left.float() - right.float()).abs()
    if not torch.isfinite(difference).all():
        raise RuntimeError(f"Non-finite {label} difference")
    return float(difference.amax().item())


def backward_names(tensor: torch.Tensor) -> list[str]:
    names: list[str] = []
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


def finite_gradient_rms(tensor: torch.Tensor, label: str) -> float:
    gradient = tensor.grad
    if gradient is None or not torch.isfinite(gradient).all():
        raise RuntimeError(f"{label} has no finite gradient")
    rms = float(gradient.float().square().mean().sqrt().item())
    if rms <= 0.0:
        raise RuntimeError(f"{label} gradient is inactive")
    return rms


def finite_module_gradient_rms(module: torch.nn.Module, label: str) -> float:
    squared_sum = 0.0
    count = 0
    for name, parameter in module.named_parameters():
        if not parameter.requires_grad:
            continue
        gradient = parameter.grad
        if gradient is None or not torch.isfinite(gradient).all():
            raise RuntimeError(f"{label}.{name} has no finite gradient")
        squared_sum += float(gradient.float().square().sum().item())
        count += gradient.numel()
    if count == 0 or squared_sum <= 0.0:
        raise RuntimeError(f"{label} gradient is inactive")
    return (squared_sum / count) ** 0.5


def require_identical_state(control: torch.nn.Module, candidate: torch.nn.Module) -> dict[str, float]:
    control_parameters = dict(control.named_parameters())
    candidate_parameters = dict(candidate.named_parameters())
    if tuple(control_parameters) != tuple(candidate_parameters):
        raise RuntimeError("Candidate/control parameter names differ")
    parameter_difference = max(
        max_diff(control_parameters[name], candidate_parameters[name], name)
        for name in control_parameters
    )
    control_buffers = dict(control.named_buffers())
    candidate_buffers = dict(candidate.named_buffers())
    if tuple(control_buffers) != tuple(candidate_buffers):
        raise RuntimeError("Candidate/control buffer names differ")
    buffer_difference = max(
        (
            max_diff(control_buffers[name], candidate_buffers[name], name)
            for name in control_buffers
            if control_buffers[name].numel()
        ),
        default=0.0,
    )
    if parameter_difference != 0.0 or buffer_difference != 0.0:
        raise RuntimeError(
            f"Candidate changed parent tensors: {parameter_difference=} {buffer_difference=}"
        )
    return {
        "parameter_max_diff": parameter_difference,
        "buffer_max_diff": buffer_difference,
    }


def projected_receiver_system(
    mixer: ZoologySurpriseRegressionGDN2FutureSeedMixer,
    evidence: torch.Tensor,
    edit: torch.Tensor,
    native_state: torch.Tensor,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    scores = edit.detach().float().square().sum(dim=(-1, -2)).sqrt()
    weights = (float(evidence.shape[1]) * scores / scores.sum(dim=1, keepdim=True)).detach()
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        keys, _ = mixer.layer.k_conv1d(
            x=mixer.layer.k_proj(evidence), cache=None, output_final_state=False
        )
        values, _ = mixer.layer.v_conv1d(
            x=mixer.layer.v_proj(evidence), cache=None, output_final_state=False
        )
        keys = keys.unflatten(-1, (HEADS, KEY_DIM))
        values = values.unflatten(-1, (HEADS, VALUE_DIM))
        writes = mixer.layer.w_proj(evidence).sigmoid().unflatten(-1, (HEADS, VALUE_DIM))
    keys = F.normalize(keys.float(), dim=-1, eps=1e-6)
    payload = values.float() * writes.float()
    native_seed = mixer.make_initial_state(native_state)
    native_prediction = torch.einsum("blhk,bhkv->blhv", keys, native_seed.float())
    residual_target = payload - native_prediction
    gram = torch.einsum("bl,blhk,blhj->bhkj", weights, keys, keys)
    gram = 0.5 * (gram + gram.transpose(-1, -2))
    cross = torch.einsum("bl,blhk,blhv->bhkv", weights, keys, residual_target)
    ridge = (
        RIDGE_SCALE
        * (gram.diagonal(dim1=-2, dim2=-1).sum(-1) / KEY_DIM + 1e-6)
    ).detach()
    system = gram + ridge[..., None, None] * torch.eye(KEY_DIM, device=evidence.device)
    return weights, keys, payload, native_seed, system, cross


def solve_projected_tuples(
    keys: torch.Tensor,
    payload: torch.Tensor,
    weights: torch.Tensor,
    native_seed: torch.Tensor,
) -> torch.Tensor:
    gram = torch.einsum("bl,blhk,blhj->bhkj", weights, keys, keys)
    gram = 0.5 * (gram + gram.transpose(-1, -2))
    native_prediction = torch.einsum("blhk,bhkv->blhv", keys, native_seed.float())
    cross = torch.einsum(
        "bl,blhk,blhv->bhkv", weights, keys, payload - native_prediction
    )
    ridge = RIDGE_SCALE * (
        gram.diagonal(dim1=-2, dim2=-1).sum(-1) / KEY_DIM + 1e-6
    )
    system = gram + ridge[..., None, None] * torch.eye(
        KEY_DIM, device=keys.device
    )
    delta = torch.cholesky_solve(cross, torch.linalg.cholesky(system))
    native_rms = native_seed.float().square().mean(
        dim=(-1, -2), keepdim=True
    ).sqrt().clamp_min(1e-6)
    delta_rms = delta.square().mean(
        dim=(-1, -2), keepdim=True
    ).sqrt().clamp_min(1e-6)
    return native_seed.float() + delta * (native_rms / delta_rms).clamp(max=1.0)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-gpu-name", required=True)
    parser.add_argument("--expected-gpu-uuid", required=True)
    args = parser.parse_args()

    if os.environ.get("CUDA_VISIBLE_DEVICES") != "0":
        raise RuntimeError("CUDA_VISIBLE_DEVICES must be exactly 0")
    if torch.cuda.device_count() != 1 or torch.cuda.current_device() != 0:
        raise RuntimeError("P-FS2-008 requires exactly CUDA index 0")
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
    if fla_root != EXPECTED_FLA_ROOT:
        raise RuntimeError(f"Unexpected FLA root: {fla_root}")
    gdn2_source = Path(inspect.getfile(GatedDeltaNet2)).resolve()
    if gdn2_source != fla_root / "fla" / "layers" / "gdn2.py":
        raise RuntimeError(f"Unexpected GDN2 source: {gdn2_source}")
    if sha256(gdn2_source) != EXPECTED_GDN2_SHA256:
        raise RuntimeError("Official GDN2 source changed")
    if python_tree_hash(fla_root / "fla" / "ops" / "gdn2") != EXPECTED_GDN2_OPS_SHA256:
        raise RuntimeError("Official GDN2 recurrence tree changed")
    if os.environ.get("FLA_GDN2_SOURCE_SHA256") != EXPECTED_GDN2_SHA256:
        raise RuntimeError("Registered GDN2 source hash changed")
    if os.environ.get("FLA_GDN2_OPS_SHA256") != EXPECTED_GDN2_OPS_SHA256:
        raise RuntimeError("Registered GDN2 ops hash changed")
    zoology_root = Path(os.environ["ZOOLOGY_ROOT"]).resolve()
    if git_head(zoology_root) != EXPECTED_ZOOLOGY_SHA:
        raise RuntimeError("Unexpected Zoology SHA")

    control_config = build_config(
        arm=CONTROL_ARM, sequence_length=1024, num_kv_pairs=4, max_epochs=10, batch_size=32
    )
    candidate_config = build_config(
        arm=CANDIDATE_ARM, sequence_length=1024, num_kv_pairs=4, max_epochs=10, batch_size=32
    )
    train_loader, test_loader = prepare_data(control_config.data)
    data_hashes = {"train": dataset_hash(train_loader), "test": dataset_hash(test_loader)}
    if data_hashes != {"train": EXPECTED_TRAIN_HASH, "test": EXPECTED_TEST_HASH}:
        raise RuntimeError(f"Directional MQAR data drifted: {data_hashes}")

    set_determinism(123)
    control = make_model(control_config, CONTROL_ARM)
    set_determinism(123)
    candidate = make_model(candidate_config, CANDIDATE_ARM)
    if type(candidate) is not SurpriseRegressionFutureSeedLanguageModel:
        raise RuntimeError("Candidate model type changed")
    parameter_counts = {
        CONTROL_ARM: sum(parameter.numel() for parameter in control.parameters()),
        CANDIDATE_ARM: sum(parameter.numel() for parameter in candidate.parameters()),
    }
    parameter_hashes = {
        CONTROL_ARM: parameter_hash(control), CANDIDATE_ARM: parameter_hash(candidate)
    }
    if parameter_counts != {CONTROL_ARM: 661_584, CANDIDATE_ARM: 661_584}:
        raise RuntimeError(f"Parameter count changed: {parameter_counts}")
    identity = require_identical_state(control, candidate)
    if parameter_hashes[CONTROL_ARM] != parameter_hashes[CANDIDATE_ARM]:
        raise RuntimeError("Candidate/control parameter hashes differ")

    provenance = []
    for layer_index, block in enumerate(candidate.backbone.layers):
        mixer = block.sequence_mixer
        if type(mixer) is not ZoologySurpriseRegressionGDN2FutureSeedMixer:
            raise RuntimeError(f"Layer {layer_index} candidate mixer changed")
        if type(mixer.layer) is not GatedDeltaNet2 or mixer.layer.mode != "chunk":
            raise RuntimeError(f"Layer {layer_index} is not exact official chunk GDN2")
        convolutions = {
            name: getattr(getattr(mixer.layer, name), "backend", None)
            for name in ("q_conv1d", "k_conv1d", "v_conv1d")
        }
        if not all(isinstance(getattr(mixer.layer, name), ShortConvolution) for name in convolutions):
            raise RuntimeError("Official ShortConvolution type changed")
        if set(convolutions.values()) != {"triton"}:
            raise RuntimeError(f"Short-convolution fallback: {convolutions}")
        provenance.append({"layer": layer_index, "convolution_backends": convolutions})

    candidate_source = Path(inspect.getfile(ZoologySurpriseRegressionGDN2FutureSeedMixer)).read_text()
    forbidden = (
        "topk(",
        "argsort(",
        ".gather(",
        "EventTape",
        "Cache(",
        "replay_seed",
        "pinv(",
        "lstsq(",
    )
    found = [token for token in forbidden if token in candidate_source]
    if found:
        raise RuntimeError(f"Replay/top-k/cache path present: {found}")

    control = control.cuda().eval()
    candidate = candidate.cuda().eval()
    generator = torch.Generator(device="cuda").manual_seed(52028)
    hidden = torch.randn(2, 128, 128, generator=generator, device="cuda")
    incoming = torch.randn(2, HEADS, KEY_DIM, VALUE_DIM, generator=generator, device="cuda")
    parity = []
    for layer_index in range(MODEL_LAYERS):
        control_mixer = control.backbone.layers[layer_index].sequence_mixer
        candidate_mixer = candidate.backbone.layers[layer_index].sequence_mixer
        for label, state in (("zero", None), ("nonzero", incoming)):
            with torch.no_grad(), capture_committed_edit() as control_edit:
                control_output, control_state = control_mixer.forward_with_state(
                    hidden, initial_state=state
                )
            with torch.no_grad():
                candidate_output, candidate_state, candidate_edit = (
                    candidate_mixer.forward_with_committed_edit(hidden, initial_state=state)
                )
            if len(control_edit) != 1:
                raise RuntimeError("Control committed-edit capture count changed")
            row = {
                "layer": layer_index,
                "state": label,
                "output_max_diff": max_diff(control_output, candidate_output, label),
                "terminal_state_max_diff": max_diff(control_state, candidate_state, label),
                "committed_edit_max_diff": max_diff(control_edit[0], candidate_edit, label),
            }
            if max(value for key, value in row.items() if key.endswith("max_diff")) != 0.0:
                raise RuntimeError(f"Producer parity failed: {row}")
            parity.append(row)

    inputs, targets, _slices = next(iter(test_loader))
    inputs, targets = inputs[:2].cuda(), targets[:2].cuda()
    candidate.train()
    for block in candidate.backbone.layers:
        block.sequence_mixer.collect_regression_diagnostics = True
    candidate.zero_grad(set_to_none=True)
    logits = candidate(inputs)
    if not torch.isfinite(logits).all():
        raise RuntimeError("Candidate logits are non-finite")
    mask = targets != -100
    loss = F.cross_entropy(logits[mask], targets[mask])
    official_backward_count = backward_names(loss).count("ChunkGDN2FunctionBackward")
    if official_backward_count != MODEL_LAYERS:
        raise RuntimeError(
            f"Expected exactly two official GDN2 backward paths, got {official_backward_count}"
        )
    receiver = candidate.backbone.layers[1].sequence_mixer
    evidence = receiver.last_regression_input
    scores = receiver.last_surprise_scores
    if evidence is None or scores is None or scores.requires_grad:
        raise RuntimeError("Canonical evidence/stop-gradient surprise wiring changed")
    loss.backward()
    full_evidence_gradient_rms = finite_gradient_rms(evidence, "canonical evidence")
    full_future_seed_gate_gradient_rms = finite_gradient_rms(
        receiver.future_seed_logit,
        "receiver.future_seed_logit",
    )
    receiver_gradients = {
        name: finite_module_gradient_rms(getattr(receiver.layer, f"{name}_proj"), f"receiver.{name}_proj")
        for name in ("k", "v", "w")
    }
    diagnostics = surprise_regression_diagnostics(candidate)
    if diagnostics["additional_official_scans"] != 0 or diagnostics["main_official_scans"] != 2:
        raise RuntimeError(f"Unexpected scan accounting: {diagnostics}")
    row = diagnostics["per_receiver"][0]
    if row["condition_max"] > 1e4 or row["solve_relative_residual"] > 1e-4:
        raise RuntimeError(f"Regression system gate failed: {row}")
    if row["weighted_seed_write_fit_mse_ratio"] > 0.75:
        raise RuntimeError(
            "Injected candidate seed did not reduce weighted write-fit MSE by 25%"
        )
    if row["bounded_residual_to_native_head_rms_max"] > 1.0 + 1e-5:
        raise RuntimeError("Per-board/head residual RMS bound failed")
    if row["candidate_to_native_head_rms_ratio_max"] > 2.0 + 1e-5:
        raise RuntimeError("Per-board/head candidate seed RMS bound failed")

    receiver.zero_grad(set_to_none=True)
    evidence_probe = torch.randn(2, 64, 128, generator=generator, device="cuda", requires_grad=True)
    edit_probe = (
        torch.rand(2, 64, HEADS, VALUE_DIM, generator=generator, device="cuda")
        + 0.1
    ).requires_grad_(True)
    native_probe = torch.randn(2, HEADS, KEY_DIM, VALUE_DIM, generator=generator, device="cuda")
    seed = receiver.regression_seed(evidence_probe, edit_probe, native_probe)
    if tuple(seed.shape) != (2, HEADS, KEY_DIM, VALUE_DIM) or not torch.isfinite(seed).all():
        raise RuntimeError(f"Regression seed shape/finite check failed: {seed.shape}")
    weights, keys, payload, native_seed, system, cross = projected_receiver_system(
        receiver, evidence_probe, edit_probe, native_probe
    )
    weight_sum_error = float((weights.sum(dim=1) - 64.0).abs().amax().item())
    if weights.requires_grad or weight_sum_error > 1e-5:
        raise RuntimeError(f"Detached normalized weights failed: {weight_sum_error}")
    factor, info = torch.linalg.cholesky_ex(system, check_errors=False)
    if int((info != 0).sum().item()) != 0:
        raise RuntimeError("Cholesky failed in checker reconstruction")
    solved = torch.cholesky_solve(cross, factor)
    singular = torch.linalg.svdvals(system)
    condition_max = float((singular[..., 0] / singular[..., -1]).amax().item())
    solve_residual = float(((system @ solved - cross).norm() / cross.norm().clamp_min(1e-12)).item())
    if condition_max > 1e4 or solve_residual > 1e-4:
        raise RuntimeError(f"Reconstructed solve failed: {condition_max=} {solve_residual=}")
    seed.float().square().mean().backward()
    if edit_probe.grad is not None:
        raise RuntimeError("Detached surprise admission leaked a gradient to edits")
    isolated_gradients = {
        "canonical_evidence": finite_gradient_rms(evidence_probe, "isolated canonical evidence"),
        "future_seed_gate": finite_gradient_rms(
            receiver.future_seed_logit,
            "isolated receiver.future_seed_logit",
        ),
        **{
            name: finite_module_gradient_rms(getattr(receiver.layer, f"{name}_proj"), f"isolated receiver.{name}_proj")
            for name in ("k", "v", "w")
        },
    }

    permutation = torch.randperm(64, generator=torch.Generator().manual_seed(52029)).cuda()
    with torch.no_grad():
        base_seed = solve_projected_tuples(keys, payload, weights, native_seed)
        permuted_seed = solve_projected_tuples(
            keys[:, permutation],
            payload[:, permutation],
            weights[:, permutation],
            native_seed,
        )
    permutation_error = max_diff(base_seed, permuted_seed, "joint token permutation")
    reconstruction_error = max_diff(
        seed.detach().float(), base_seed, "checker/core ridge residual reconstruction"
    )
    if reconstruction_error > 2e-4:
        raise RuntimeError(f"Checker/core reconstruction failed: {reconstruction_error}")
    if permutation_error > 2e-4:
        raise RuntimeError(f"Joint permutation invariance failed: {permutation_error}")

    original_weight = receiver.layer.k_proj.weight.detach().clone()
    with torch.no_grad():
        receiver.layer.k_proj.weight.copy_(original_weight.roll(1, dims=0))
        changed_seed = receiver.regression_seed(evidence_probe.detach(), edit_probe, native_probe)
        receiver.layer.k_proj.weight.copy_(original_weight)
    receiver_dependency = max_diff(seed.detach(), changed_seed, "receiver coordinates")
    if receiver_dependency < 1e-4:
        raise RuntimeError("Regression seed does not depend on receiver coordinates")

    result = {
        "status": "passed",
        "gpu": {"name": device.name, "uuid": device_uuid, "count": 1},
        "pinned_fla_sha": PINNED_FLA_SHA,
        "zoology_sha": EXPECTED_ZOOLOGY_SHA,
        "data_hashes": data_hashes,
        "parameter_counts": parameter_counts,
        "parameter_hashes": parameter_hashes,
        **identity,
        "official_module_provenance": provenance,
        "producer_parity": parity,
        "official_chunk_backward_count": official_backward_count,
        "full_evidence_gradient_rms": full_evidence_gradient_rms,
        "full_future_seed_gate_gradient_rms": full_future_seed_gate_gradient_rms,
        "full_receiver_projection_gradient_rms": receiver_gradients,
        "isolated_gradient_rms": isolated_gradients,
        "weight_sum_max_error": weight_sum_error,
        "joint_permutation_max_error": permutation_error,
        "core_reconstruction_max_error": reconstruction_error,
        "permutation_scope": "post_causal_conv_receiver_k_payload_weight_tuples",
        "receiver_coordinate_dependency_max_diff": receiver_dependency,
        "cholesky_failure_count": int((info != 0).sum().item()),
        "condition_max": condition_max,
        "solve_relative_residual": solve_residual,
        "seed_shape": list(seed.shape),
        "weighted_candidate_seed_write_fit_mse": row[
            "weighted_candidate_seed_write_fit_mse"
        ],
        "weighted_native_seed_write_fit_mse": row[
            "weighted_native_seed_write_fit_mse"
        ],
        "weighted_seed_write_fit_mse_ratio": row[
            "weighted_seed_write_fit_mse_ratio"
        ],
        "bounded_residual_to_native_head_rms_max": row[
            "bounded_residual_to_native_head_rms_max"
        ],
        "candidate_to_native_head_rms_ratio_max": row[
            "candidate_to_native_head_rms_ratio_max"
        ],
        "no_replay_scan_topk_or_cache": True,
        "new_parameters": 0,
        "persistent_state_delta": 0,
        "fallback_used": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
