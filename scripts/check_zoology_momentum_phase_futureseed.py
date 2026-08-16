from __future__ import annotations

import argparse
import importlib
import inspect
import json
import os
import subprocess
from pathlib import Path

import torch
import torch.nn.functional as F
from zoology.data.utils import prepare_data
from zoology.utils import set_determinism

from experiments.zoology_mqar.length_scaling import build_config, dataset_hash, make_model
from experiments.zoology_mqar.momentum_futureseed import (
    EXPECTED_CHUNK_SHA256,
    EXPECTED_LAYER_SHA256,
    EXPECTED_MDN_SHA,
    EXPECTED_RECURRENT_SHA256,
    EXPECTED_STATE_VALUES_PER_LAYER,
    ZoologyMomentumDeltaFutureSeedMixer,
    load_external_momentum_layer,
    load_matched_parent_state as load_momentum_parent_state,
    momentum_fla_compatibility,
)
from experiments.zoology_mqar.momentum_phase_futureseed import (
    EXPECTED_PARAMETER_DELTA_VS_MOMENTUM,
    ZoologyMomentumPhaseFutureSeedMixer,
    load_matched_parent_state as load_phase_parent_state,
)
from scripts.check_zoology_contractive_dplr import backward_names, normalized_uuid


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


def _gradient(parameter: torch.Tensor, name: str, *, every: bool = False) -> dict:
    if parameter.grad is None:
        raise RuntimeError(f"Missing gradient: {name}")
    gradient = parameter.grad.detach().float()
    rms = gradient.square().mean().sqrt()
    if (
        not torch.isfinite(gradient).all()
        or float(rms) == 0.0
        or (every and not bool((gradient != 0).all()))
    ):
        raise RuntimeError(f"Inactive or nonfinite gradient: {name}")
    return {
        "rms": float(rms),
        "min_abs": float(gradient.abs().min()),
        "all_nonzero": bool((gradient != 0).all()),
    }


def _max_abs(left: torch.Tensor, right: torch.Tensor) -> float:
    if not torch.isfinite(left).all() or not torch.isfinite(right).all():
        raise RuntimeError("Nonfinite tensor in contract check")
    return float((left.float() - right.float()).abs().max())


def _relative_rms(left: torch.Tensor, right: torch.Tensor) -> float:
    delta = (left.float() - right.float()).square().mean().sqrt()
    scale = right.float().square().mean().sqrt().clamp_min(1e-8)
    return float(delta / scale)


def _rotate(seed: torch.Tensor, phase: torch.Tensor) -> torch.Tensor:
    state, momentum = seed[0].float(), seed[1].float()
    cosine, sine = torch.cos(phase.float()), torch.sin(phase.float())
    return torch.stack(
        (
            cosine * state - sine * momentum,
            sine * state + cosine * momentum,
        ),
        dim=0,
    )


def _energy_error(before: torch.Tensor, after: torch.Tensor) -> float:
    before_energy = before.float().square().sum(dim=(0, 3, 4))
    after_energy = after.float().square().sum(dim=(0, 3, 4))
    relative = (
        (after_energy - before_energy).abs()
        / before_energy.clamp_min(1e-8)
    )
    return float(relative.max())


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
        raise RuntimeError("P-FS2-014 requires exactly CUDA index 0")
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

    parent_config = build_config(
        arm="future_seed_momentum_delta",
        sequence_length=1024,
        num_kv_pairs=4,
        max_epochs=10,
        batch_size=32,
    )
    candidate_config = build_config(
        arm="future_seed_momentum_phase",
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
    parent = make_model(parent_config, "future_seed_momentum_delta")
    load_momentum_parent_state(parent, parent_state)
    set_determinism(123)
    candidate = make_model(candidate_config, "future_seed_momentum_phase")
    load_phase_parent_state(candidate, parent_state)

    parent_parameters = sum(parameter.numel() for parameter in parent.parameters())
    candidate_parameters = sum(parameter.numel() for parameter in candidate.parameters())
    if parent_parameters != EXPECTED_MOMENTUM_PARAMETERS:
        raise RuntimeError(f"Momentum parameter count drifted: {parent_parameters}")
    if candidate_parameters != EXPECTED_CANDIDATE_PARAMETERS:
        raise RuntimeError(f"Candidate parameter count drifted: {candidate_parameters}")

    parent_sd = parent.state_dict()
    candidate_sd = candidate.state_dict()
    if any(
        name not in candidate_sd or not torch.equal(tensor, candidate_sd[name])
        for name, tensor in parent_sd.items()
    ):
        raise RuntimeError("Phase candidate changed a parent tensor")
    phase_names = sorted(name for name in candidate_sd if "future_seed_phase" in name)
    if len(phase_names) != 1 or candidate_sd[phase_names[0]].numel() != 4:
        raise RuntimeError(f"Unexpected phase parameters: {phase_names}")
    if torch.count_nonzero(candidate_sd[phase_names[0]]) != 0:
        raise RuntimeError("FutureSeed phase must initialize to exact zero")

    parent_mixers = [block.sequence_mixer for block in parent.backbone.layers]
    candidate_mixers = [block.sequence_mixer for block in candidate.backbone.layers]
    if len(parent_mixers) != 2 or not all(
        type(mixer) is ZoologyMomentumDeltaFutureSeedMixer for mixer in parent_mixers
    ):
        raise RuntimeError("Expected two exact Momentum parent mixers")
    if len(candidate_mixers) != 2 or not all(
        type(mixer) is ZoologyMomentumPhaseFutureSeedMixer for mixer in candidate_mixers
    ):
        raise RuntimeError("Expected two exact Momentum phase mixers")
    if candidate_mixers[0].future_seed_phase is not None:
        raise RuntimeError("Non-receiving layer must not own phase parameters")
    if candidate_mixers[1].future_seed_phase is None:
        raise RuntimeError("Receiving layer is missing phase parameters")
    if any(
        type(mixer.layer) is not layer_class
        or mixer.state_size() != EXPECTED_STATE_VALUES_PER_LAYER
        for mixer in candidate_mixers
    ):
        raise RuntimeError("Candidate source or state geometry changed")

    inputs, targets, _slices = next(iter(test_loader))
    inputs, targets = inputs[:4].cuda(), targets[:4].cuda()
    parent, candidate = parent.cuda().eval(), candidate.cuda().eval()
    with torch.no_grad():
        parent_logits = parent(inputs)
        candidate_logits = candidate(inputs)
    full_model_identity = {
        "exact": bool(torch.equal(candidate_logits, parent_logits)),
        "max_abs": _max_abs(candidate_logits, parent_logits),
    }
    if not full_model_identity["exact"]:
        raise RuntimeError(f"Zero phase changed model output: {full_model_identity}")

    generator = torch.Generator(device="cuda").manual_seed(214014)
    hidden = torch.randn(2, 128, 128, generator=generator, device="cuda")
    incoming = 0.05 * torch.randn(
        2, 2, 4, 32, 32, generator=generator, device="cuda"
    )
    parent_seed = parent_mixers[1].make_initial_state(incoming)
    candidate_seed = candidate_mixers[1].make_initial_state(incoming)
    seed_identity = {
        "exact": bool(torch.equal(candidate_seed, parent_seed)),
        "max_abs": _max_abs(candidate_seed, parent_seed),
    }
    if not seed_identity["exact"]:
        raise RuntimeError(f"Zero phase changed nonzero seed: {seed_identity}")
    with torch.no_grad():
        parent_output, parent_terminal = parent_mixers[1].forward_with_state(
            hidden, initial_state=parent_seed
        )
        candidate_output, candidate_terminal = candidate_mixers[1].forward_with_state(
            hidden, initial_state=candidate_seed
        )
    nonzero_identity = {
        "output_exact": bool(torch.equal(candidate_output, parent_output)),
        "state_exact": bool(torch.equal(candidate_terminal, parent_terminal)),
        "output_max_abs": _max_abs(candidate_output, parent_output),
        "state_max_abs": _max_abs(candidate_terminal, parent_terminal),
    }
    if not nonzero_identity["output_exact"] or not nonzero_identity["state_exact"]:
        raise RuntimeError(f"Nonzero-state identity failed: {nonzero_identity}")

    candidate.train().zero_grad(set_to_none=True)
    logits = candidate(inputs)
    mask = targets != -100
    loss = F.cross_entropy(logits[mask], targets[mask])
    graph_names = backward_names(loss)
    chunk_backward_count = sum(
        "Chunkmode_ruleFunctionBackward" in name for name in graph_names
    )
    if chunk_backward_count != 2:
        raise RuntimeError(f"Expected two Momentum chunk backwards, got {chunk_backward_count}")
    loss.backward()
    gradients = []
    for index, mixer in enumerate(candidate_mixers):
        row = {
            "layer": index,
            "q": _gradient(mixer.layer.q_proj.weight, f"layer{index}.q"),
            "k": _gradient(mixer.layer.k_proj.weight, f"layer{index}.k"),
            "v": _gradient(mixer.layer.v_proj.weight, f"layer{index}.v"),
            "alpha": _gradient(mixer.layer.a_proj.weight, f"layer{index}.alpha"),
            "momentum": _gradient(mixer.layer.m_proj.weight, f"layer{index}.momentum"),
            "erase": _gradient(mixer.layer.b_proj.weight, f"layer{index}.erase"),
            "write": _gradient(mixer.layer.e_proj.weight, f"layer{index}.write"),
        }
        if index == 1:
            row["future_seed"] = _gradient(
                mixer.future_seed_logit, "layer1.future_seed"
            )
            row["phase"] = _gradient(
                mixer.future_seed_phase, "layer1.phase", every=True
            )
        gradients.append(row)

    phase = torch.tensor(
        [[[[0.10]], [[-0.20]], [[0.30]], [[-0.40]]]],
        device="cuda",
    )
    opened_fp32 = _rotate(parent_seed, phase)
    opened_bf16 = opened_fp32.to(torch.bfloat16)
    base_bf16 = parent_seed.to(torch.bfloat16)
    energy = {
        "fp32_relative_error": _energy_error(parent_seed.float(), opened_fp32),
        "bf16_relative_error": _energy_error(base_bf16, opened_bf16),
    }
    if energy["fp32_relative_error"] > 1e-5 or energy["bf16_relative_error"] > 5e-3:
        raise RuntimeError(f"Phase rotation energy contract failed: {energy}")

    permutation = torch.tensor([2, 0, 3, 1], device="cuda")
    permuted_seed = parent_seed.index_select(2, permutation)
    permuted_phase = phase.index_select(1, permutation)
    permuted_opened = _rotate(permuted_seed, permuted_phase)
    equivariance_error = _max_abs(
        permuted_opened,
        opened_fp32.index_select(2, permutation),
    )
    if equivariance_error != 0.0:
        raise RuntimeError(f"Head equivariance failed: {equivariance_error}")

    phase_parameter = candidate_mixers[1].future_seed_phase
    candidate.eval()
    with torch.no_grad():
        phase_parameter.copy_(phase)
        opened_logits = candidate(inputs)
        opened_seed = candidate_mixers[1].make_initial_state(incoming)
        opened_output, opened_terminal = candidate_mixers[1].forward_with_state(
            hidden, initial_state=opened_seed
        )
        phase_parameter.zero_()
    opened_dependency = {
        "seed_relative_rms": _relative_rms(opened_seed, parent_seed),
        "model_output_relative_rms": _relative_rms(opened_logits, parent_logits),
        "layer_output_relative_rms": _relative_rms(opened_output, parent_output),
        "terminal_relative_rms": _relative_rms(opened_terminal, parent_terminal),
        "owner_k_exact": all(
            torch.equal(
                candidate.state_dict()[name],
                parent_sd[name].to(candidate.state_dict()[name].device),
            )
            for name in parent_sd
            if ".layer.k_" in name
        ),
    }
    if (
        opened_dependency["seed_relative_rms"] <= 1e-4
        or opened_dependency["model_output_relative_rms"] <= 1e-6
        or opened_dependency["layer_output_relative_rms"] <= 1e-6
        or opened_dependency["terminal_relative_rms"] <= 1e-6
        or not opened_dependency["owner_k_exact"]
    ):
        raise RuntimeError(f"Opened phase dependency failed: {opened_dependency}")

    result = {
        "status": "passed",
        "plan": "P-FS2-014",
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
            "momentum": parent_parameters,
            "candidate": candidate_parameters,
            "delta": candidate_parameters - parent_parameters,
        },
        "phase_parameter_names": phase_names,
        "full_model_zero_phase_identity": full_model_identity,
        "nonzero_seed_zero_phase_identity": seed_identity,
        "nonzero_state_zero_phase_identity": nonzero_identity,
        "chunk_backward_count": chunk_backward_count,
        "gradients": gradients,
        "opened_phase": opened_dependency,
        "rotation_energy": energy,
        "head_equivariance_max_abs": equivariance_error,
        "state_values_per_layer": EXPECTED_STATE_VALUES_PER_LAYER,
        "scan_delta": 0,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
