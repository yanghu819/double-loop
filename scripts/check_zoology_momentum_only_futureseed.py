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
from experiments.zoology_mqar.momentum_only_futureseed import (
    FULL_STATE_VALUES_PER_ROUTE,
    MOMENTUM_PAYLOAD_VALUES_PER_ROUTE,
    ZoologyMomentumOnlyFutureSeedMixer,
)
from scripts.check_zoology_contractive_dplr import (
    backward_names,
    normalized_uuid,
)


EXPECTED_TRAIN_HASH = "647c64ece84984a23dfd817c4f277ea83840dbec57cc18bb6c9bf9eda7cc9a68"
EXPECTED_TEST_HASH = "4a8237ba8fe19aaff0d1d72de7b7f6505eaab59cd091442c2f463df34cce278f"
EXPECTED_PARAMETERS = 599_672


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


def _max_abs(left: torch.Tensor, right: torch.Tensor) -> float:
    return float((left.float() - right.float()).abs().max())


def _relative_rms(left: torch.Tensor, right: torch.Tensor) -> float:
    delta = (left.float() - right.float()).square().mean().sqrt()
    scale = right.float().square().mean().sqrt().clamp_min(1e-8)
    return float(delta / scale)


def _gradient(parameter: torch.Tensor, name: str) -> float:
    if parameter.grad is None:
        raise RuntimeError(f"Missing gradient: {name}")
    gradient = parameter.grad.detach().float()
    rms = gradient.square().mean().sqrt()
    if not torch.isfinite(gradient).all() or float(rms) == 0.0:
        raise RuntimeError(f"Inactive or nonfinite gradient: {name}")
    return float(rms)


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
        raise RuntimeError("P-FS2-016 requires exactly CUDA index 0")
    gpu_uuid, gpu_name = _visible_gpu()
    device = torch.cuda.get_device_properties(0)
    if (
        gpu_uuid != args.expected_gpu_uuid
        or gpu_name != args.expected_gpu_name
        or device.name != args.expected_gpu_name
        or normalized_uuid(str(getattr(device, "uuid", "unavailable")))
        != normalized_uuid(args.expected_gpu_uuid)
    ):
        raise RuntimeError(f"Unexpected GPU: {gpu_uuid}, {gpu_name}")

    layer_class = load_external_momentum_layer()
    compatibility = momentum_fla_compatibility()
    if compatibility["missing_required_symbols"] or compatibility["use_cuda_graph"]:
        raise RuntimeError(f"Unexpected FLA compatibility: {compatibility}")
    fla_root = Path(os.environ["MDN_FLA_ROOT"]).resolve()
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
    if subprocess.check_output(
        ["git", "-C", os.environ["MDN_REPO_ROOT"], "rev-parse", "HEAD"],
        text=True,
    ).strip() != EXPECTED_MDN_SHA:
        raise RuntimeError("External Momentum SHA drifted")

    control_config = build_config(
        arm="future_seed_momentum_delta",
        sequence_length=1024,
        num_kv_pairs=4,
        max_epochs=10,
        batch_size=32,
    )
    candidate_config = build_config(
        arm="future_seed_momentum_only",
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
        raise RuntimeError(f"Data drifted: {data_hashes}")

    parent_state = torch.load(args.matched_init, map_location="cpu", weights_only=True)
    set_determinism(123)
    control = make_model(control_config, "future_seed_momentum_delta")
    load_matched_parent_state(control, parent_state)
    set_determinism(123)
    candidate = make_model(candidate_config, "future_seed_momentum_only")
    load_matched_parent_state(candidate, parent_state)
    if sum(p.numel() for p in control.parameters()) != EXPECTED_PARAMETERS:
        raise RuntimeError("P059 parameter count drifted")
    if sum(p.numel() for p in candidate.parameters()) != EXPECTED_PARAMETERS:
        raise RuntimeError("Momentum-only parameter count drifted")
    control_state = control.state_dict()
    candidate_state = candidate.state_dict()
    if set(control_state) != set(candidate_state):
        raise RuntimeError("Momentum-only model changed parameter names")
    parent_parameter_max_abs = max(
        _max_abs(control_state[name], candidate_state[name])
        for name in control_state
    )
    if parent_parameter_max_abs != 0.0:
        raise RuntimeError("Momentum-only model changed parent initialization")

    control_mixers = [block.sequence_mixer for block in control.backbone.layers]
    candidate_mixers = [block.sequence_mixer for block in candidate.backbone.layers]
    if not all(type(mixer) is ZoologyMomentumDeltaFutureSeedMixer for mixer in control_mixers):
        raise RuntimeError("Control escaped exact P059 mixer")
    if not all(type(mixer) is ZoologyMomentumOnlyFutureSeedMixer for mixer in candidate_mixers):
        raise RuntimeError("Candidate escaped exact Momentum-only mixer")
    for mixer in candidate_mixers:
        if type(mixer.layer) is not layer_class:
            raise RuntimeError("Candidate escaped pinned Momentum layer")
        if mixer.state_size() != EXPECTED_STATE_VALUES_PER_LAYER:
            raise RuntimeError("Persistent state geometry changed")

    inputs, targets, _slices = next(iter(test_loader))
    inputs = inputs[:4].cuda()
    targets = targets[:4].cuda()
    control = control.cuda().eval()
    candidate = candidate.cuda().eval()

    generator = torch.Generator(device="cuda").manual_seed(16016)
    hidden = torch.randn(
        2, 1024, 128, generator=generator, device="cuda", dtype=torch.float32
    )
    with torch.no_grad():
        control_output, control_terminal = control_mixers[0].forward_with_state(
            hidden, initial_state=None
        )
        candidate_output, candidate_terminal = candidate_mixers[0].forward_with_state(
            hidden, initial_state=None
        )
    producer_identity = {
        "output_max_abs": _max_abs(control_output, candidate_output),
        "terminal_max_abs": _max_abs(control_terminal, candidate_terminal),
    }
    if any(value != 0.0 for value in producer_identity.values()):
        raise RuntimeError(f"Producer path changed: {producer_identity}")

    synthetic_terminal = 0.05 * torch.randn(
        2,
        2,
        4,
        32,
        32,
        generator=generator,
        device="cuda",
        dtype=torch.float32,
    )
    with torch.no_grad():
        control_seed = control_mixers[1].make_initial_state(synthetic_terminal)
        payload = candidate_mixers[0].pack_future_seed(synthetic_terminal)
        candidate_seed = candidate_mixers[1].make_initial_state(payload)
    seed_contract = {
        "full_values_per_board": int(synthetic_terminal[:, 0].numel()),
        "payload_values_per_board": int(payload[0].numel()),
        "state_seed_max_abs": float(candidate_seed[0].abs().max()),
        "momentum_seed_max_abs_vs_parent": _max_abs(
            candidate_seed[1], control_seed[1]
        ),
    }
    if seed_contract != {
        "full_values_per_board": FULL_STATE_VALUES_PER_ROUTE,
        "payload_values_per_board": MOMENTUM_PAYLOAD_VALUES_PER_ROUTE,
        "state_seed_max_abs": 0.0,
        "momentum_seed_max_abs_vs_parent": 0.0,
    }:
        raise RuntimeError(f"Momentum-only transport failed: {seed_contract}")

    candidate.train()
    candidate.zero_grad(set_to_none=True)
    logits = candidate(inputs)
    mask = targets != -100
    loss = F.cross_entropy(logits[mask], targets[mask])
    graph_names = backward_names(loss)
    backward_count = sum(
        "Chunkmode_ruleFunctionBackward" in name for name in graph_names
    )
    if backward_count != 2:
        raise RuntimeError(f"Expected two Momentum backwards, got {backward_count}")
    loss.backward()
    gradient_rows = []
    for index, mixer in enumerate(candidate_mixers):
        layer = mixer.layer
        row = {
            "layer": index,
            "q": _gradient(layer.q_proj.weight, f"layer{index}.q"),
            "k": _gradient(layer.k_proj.weight, f"layer{index}.k"),
            "v": _gradient(layer.v_proj.weight, f"layer{index}.v"),
            "momentum": _gradient(layer.m_proj.weight, f"layer{index}.momentum"),
        }
        if index == 1:
            row["future_seed"] = _gradient(
                mixer.future_seed_logit, "layer1.future_seed"
            )
        gradient_rows.append(row)

    terminal_for_gradient = synthetic_terminal.detach().requires_grad_(True)
    payload_for_gradient = candidate_mixers[0].pack_future_seed(
        terminal_for_gradient
    )
    seed_for_gradient = candidate_mixers[1].make_initial_state(
        payload_for_gradient
    )
    candidate_mixers[1].zero_grad(set_to_none=True)
    routed_output, routed_state = candidate_mixers[1].forward_with_state(
        hidden,
        initial_state=seed_for_gradient,
    )
    (routed_output.float().square().mean() + routed_state.float().square().mean()).backward()
    seed_gradient = terminal_for_gradient.grad
    if seed_gradient is None or not torch.isfinite(seed_gradient).all():
        raise RuntimeError("Missing finite transport gradient")
    transport_gradient = {
        "state_max_abs": float(seed_gradient[0].abs().max()),
        "momentum_rms": float(seed_gradient[1].float().square().mean().sqrt()),
    }
    if transport_gradient["state_max_abs"] != 0.0 or transport_gradient["momentum_rms"] <= 0:
        raise RuntimeError(f"Transport gradient contract failed: {transport_gradient}")

    with torch.no_grad():
        candidate.eval()(inputs)
    full_stack = {
        "producer_payload_rms": float(candidate_mixers[0].last_payload_rms),
        "receiver_state_seed_rms": float(candidate_mixers[1].last_seed_state_rms),
        "receiver_momentum_seed_rms": float(
            candidate_mixers[1].last_seed_momentum_rms
        ),
        "active_routes": sum(
            mixer.last_seed_gate is not None for mixer in candidate_mixers
        ),
    }
    if (
        full_stack["producer_payload_rms"] <= 1e-4
        or full_stack["receiver_state_seed_rms"] != 0.0
        or full_stack["receiver_momentum_seed_rms"] <= 1e-4
        or full_stack["active_routes"] != 1
    ):
        raise RuntimeError(f"Full-stack activation failed: {full_stack}")

    result = {
        "status": "passed",
        "gpu": {"uuid": gpu_uuid, "name": gpu_name},
        "external_sha": EXPECTED_MDN_SHA,
        "source_hashes": source_hashes,
        "host_fla_compatibility": compatibility,
        "data_hashes": data_hashes,
        "parameters": {
            "p059": EXPECTED_PARAMETERS,
            "candidate": EXPECTED_PARAMETERS,
            "delta": 0,
            "parent_parameter_max_abs": parent_parameter_max_abs,
        },
        "recurrent_state_values_per_layer": EXPECTED_STATE_VALUES_PER_LAYER,
        "transport": seed_contract,
        "transport_ratio": 0.5,
        "producer_identity": producer_identity,
        "transport_gradient": transport_gradient,
        "full_stack": full_stack,
        "official_momentum_backward_count": backward_count,
        "gradient_rows": gradient_rows,
        "scan_delta": 0,
        "fallback": None,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
