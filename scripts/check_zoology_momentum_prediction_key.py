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
from experiments.zoology_mqar.momentum_prediction_key import (
    EXPECTED_PARAMETER_DELTA_VS_MOMENTUM,
    ZoologyMomentumPredictionKeyFutureSeedMixer,
    load_matched_parent_state as load_prediction_parent_state,
)
from scripts.check_zoology_contractive_dplr import backward_names, normalized_uuid


EXPECTED_TRAIN_HASH = "647c64ece84984a23dfd817c4f277ea83840dbec57cc18bb6c9bf9eda7cc9a68"
EXPECTED_TEST_HASH = "4a8237ba8fe19aaff0d1d72de7b7f6505eaab59cd091442c2f463df34cce278f"
EXPECTED_MOMENTUM_PARAMETERS = 599_672
EXPECTED_CANDIDATE_PARAMETERS = EXPECTED_MOMENTUM_PARAMETERS + EXPECTED_PARAMETER_DELTA_VS_MOMENTUM


def _visible_gpu() -> tuple[str, str]:
    output = subprocess.check_output(
        ["nvidia-smi", "-i", "0", "--query-gpu=uuid,name", "--format=csv,noheader,nounits"],
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
        raise RuntimeError("Nonfinite tensor in contract check")
    return float((left.float() - right.float()).abs().max())


def _relative_rms(left: torch.Tensor, right: torch.Tensor) -> float:
    delta = (left.float() - right.float()).square().mean().sqrt()
    scale = right.float().square().mean().sqrt().clamp_min(1e-8)
    return float(delta / scale)


def _per_head(tensor: torch.Tensor, permutation: torch.Tensor) -> torch.Tensor:
    return tensor.index_select(2, permutation)


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
        raise RuntimeError("P-GDN3-063 requires exactly CUDA index 0")
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

    parent_config = build_config(
        arm="future_seed_momentum_delta", sequence_length=1024, num_kv_pairs=4,
        max_epochs=10, batch_size=32,
    )
    candidate_config = build_config(
        arm="future_seed_momentum_prediction_key", sequence_length=1024, num_kv_pairs=4,
        max_epochs=10, batch_size=32,
    )
    train_loader, test_loader = prepare_data(candidate_config.data)
    data_hashes = {"train": dataset_hash(train_loader), "test": dataset_hash(test_loader)}
    if data_hashes != {"train": EXPECTED_TRAIN_HASH, "test": EXPECTED_TEST_HASH}:
        raise RuntimeError(f"Directional MQAR data drifted: {data_hashes}")

    parent_state = torch.load(args.matched_init, map_location="cpu", weights_only=True)
    set_determinism(123)
    parent = make_model(parent_config, "future_seed_momentum_delta")
    load_momentum_parent_state(parent, parent_state)
    set_determinism(123)
    candidate = make_model(candidate_config, "future_seed_momentum_prediction_key")
    load_prediction_parent_state(candidate, parent_state)

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
        raise RuntimeError("Prediction-key candidate changed a parent tensor")
    prediction_names = sorted(name for name in candidate_sd if "prediction_key" in name)
    if len(prediction_names) != 4:
        raise RuntimeError(f"Unexpected prediction-key tensors: {prediction_names}")

    parent_mixers = [block.sequence_mixer for block in parent.backbone.layers]
    candidate_mixers = [block.sequence_mixer for block in candidate.backbone.layers]
    if len(parent_mixers) != 2 or not all(
        type(mixer) is ZoologyMomentumDeltaFutureSeedMixer for mixer in parent_mixers
    ):
        raise RuntimeError("Expected two exact Momentum parent mixers")
    if len(candidate_mixers) != 2 or not all(
        type(mixer) is ZoologyMomentumPredictionKeyFutureSeedMixer for mixer in candidate_mixers
    ):
        raise RuntimeError("Expected two exact prediction-key mixers")
    tied_paths = []
    for mixer in candidate_mixers:
        if type(mixer.layer) is not layer_class or mixer.state_size() != EXPECTED_STATE_VALUES_PER_LAYER:
            raise RuntimeError("Candidate source or state geometry changed")
        proj_exact = all(
            torch.equal(left, right)
            for left, right in zip(
                mixer.prediction_key_proj.state_dict().values(),
                mixer.layer.k_proj.state_dict().values(),
                strict=True,
            )
        )
        conv_exact = all(
            torch.equal(left, right)
            for left, right in zip(
                mixer.prediction_key_conv.state_dict().values(),
                mixer.layer.k_conv1d.state_dict().values(),
                strict=True,
            )
        )
        if not proj_exact or not conv_exact:
            raise RuntimeError("Prediction path is not exactly tied to owner K")
        tied_paths.append({"layer": mixer.layer_idx, "proj": proj_exact, "conv": conv_exact})

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
        raise RuntimeError(f"Tied prediction key changed model output: {full_model_identity}")

    generator = torch.Generator(device="cuda").manual_seed(63063)
    hidden = torch.randn(2, 128, 128, generator=generator, device="cuda")
    initial_state = 0.05 * torch.randn(2, 2, 4, 32, 32, generator=generator, device="cuda")
    layer_identity = []
    for index, (parent_mixer, candidate_mixer) in enumerate(
        zip(parent_mixers, candidate_mixers, strict=True)
    ):
        parent_mixer.eval()
        candidate_mixer.eval()
        with torch.no_grad():
            parent_output, parent_terminal = parent_mixer.forward_with_state(
                hidden, initial_state=initial_state
            )
            candidate_output, candidate_terminal = candidate_mixer.forward_with_state(
                hidden, initial_state=initial_state
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
    chunk_backward_count = sum("Chunkmode_ruleFunctionBackward" in name for name in graph_names)
    if chunk_backward_count != 2:
        raise RuntimeError(f"Expected two Momentum chunk backwards, got {chunk_backward_count}")
    loss.backward()
    gradients = []
    for index, mixer in enumerate(candidate_mixers):
        conv_parameters = list(mixer.prediction_key_conv.parameters())
        if len(conv_parameters) != 1:
            raise RuntimeError("Prediction ShortConv parameter geometry drifted")
        row = {
            "layer": index,
            "prediction_projection": _gradient(
                mixer.prediction_key_proj.weight, f"layer{index}.prediction_projection"
            ),
            "prediction_convolution": _gradient(
                conv_parameters[0], f"layer{index}.prediction_convolution"
            ),
            "q": _gradient(mixer.layer.q_proj.weight, f"layer{index}.q"),
            "k": _gradient(mixer.layer.k_proj.weight, f"layer{index}.k"),
            "v": _gradient(mixer.layer.v_proj.weight, f"layer{index}.v"),
            "alpha": _gradient(mixer.layer.a_proj.weight, f"layer{index}.alpha"),
            "momentum": _gradient(mixer.layer.m_proj.weight, f"layer{index}.momentum"),
            "erase": _gradient(mixer.layer.b_proj.weight, f"layer{index}.erase"),
            "write": _gradient(mixer.layer.e_proj.weight, f"layer{index}.write"),
        }
        if index > 0:
            row["future_seed"] = _gradient(mixer.future_seed_logit, f"layer{index}.future_seed")
        gradients.append(row)

    operation = importlib.import_module(layer_class.__module__).chunk_mode_rule
    q = torch.randn(2, 96, 4, 32, generator=generator, device="cuda", dtype=torch.bfloat16)
    k, v = torch.randn_like(q), torch.randn_like(q)
    p = k.clone()
    opened_p = F.normalize(p.float() + 0.25 * torch.roll(p.float(), 1, -1), dim=-1).to(p.dtype)
    log_alpha = -0.1 * torch.rand(2, 96, 4, generator=generator, device="cuda")
    log_mu = -0.1 * torch.rand(2, 96, 4, generator=generator, device="cuda")
    beta = torch.rand(2, 96, 4, generator=generator, device="cuda")
    eta = 0.5 + torch.rand(2, 96, 4, generator=generator, device="cuda")
    synthetic_state = 0.02 * torch.randn(
        2, 2, 4, 32, 32, generator=generator, device="cuda", dtype=torch.bfloat16
    )
    with torch.no_grad():
        tied_output, tied_terminal = operation(
            q=q, k=k, v=v, p=p, log_alpha=log_alpha, log_mu=log_mu,
            beta=beta, eta=eta, initial_state=synthetic_state, output_final_state=True,
        )
        opened_output, opened_terminal = operation(
            q=q, k=k, v=v, p=opened_p, log_alpha=log_alpha, log_mu=log_mu,
            beta=beta, eta=eta, initial_state=synthetic_state, output_final_state=True,
        )
        permutation = torch.tensor([2, 0, 3, 1], device="cuda")
        permuted_output, permuted_terminal = operation(
            q=_per_head(q, permutation), k=_per_head(k, permutation),
            v=_per_head(v, permutation), p=_per_head(opened_p, permutation),
            log_alpha=_per_head(log_alpha.unsqueeze(-1), permutation).squeeze(-1),
            log_mu=_per_head(log_mu.unsqueeze(-1), permutation).squeeze(-1),
            beta=_per_head(beta.unsqueeze(-1), permutation).squeeze(-1),
            eta=_per_head(eta.unsqueeze(-1), permutation).squeeze(-1),
            initial_state=synthetic_state.index_select(2, permutation),
            output_final_state=True,
        )
    synthetic = {
        "output_relative_rms": _relative_rms(opened_output, tied_output),
        "state_relative_rms": _relative_rms(opened_terminal, tied_terminal),
        "head_output_max_abs": _max_abs(
            permuted_output, opened_output.index_select(2, permutation)
        ),
        "head_state_max_abs": _max_abs(
            permuted_terminal, opened_terminal.index_select(2, permutation)
        ),
    }
    if (
        synthetic["output_relative_rms"] <= 1e-4
        or synthetic["state_relative_rms"] <= 1e-4
        or synthetic["head_output_max_abs"] != 0.0
        or synthetic["head_state_max_abs"] != 0.0
    ):
        raise RuntimeError(f"Opened prediction-key contract failed: {synthetic}")

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
        "plan": "P-GDN3-063",
        "gpu": {"uuid": gpu_uuid, "name": gpu_name},
        "external": {
            "repo": str(repo_root), "sha": EXPECTED_MDN_SHA,
            "source_hashes": source_hashes, "redistributed_source": False,
        },
        "host_fla_compatibility": compatibility,
        "data_hashes": data_hashes,
        "parameters": {
            "momentum": parent_parameters, "candidate": candidate_parameters,
            "delta": candidate_parameters - parent_parameters,
        },
        "prediction_parameter_names": prediction_names,
        "tied_paths": tied_paths,
        "full_model_tied_identity": full_model_identity,
        "nonzero_state_tied_identity": layer_identity,
        "chunk_backward_count": chunk_backward_count,
        "gradients": gradients,
        "opened_prediction_key": synthetic,
        "futureseed_state_transport": seed_geometry,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
