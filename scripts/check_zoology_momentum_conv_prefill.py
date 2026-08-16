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
from experiments.zoology_mqar.momentum_conv_prefill_futureseed import (
    PREFILL_PARAMETERS,
    PREFILL_TOKENS,
    ZoologyMomentumConvPrefillFutureSeedMixer,
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
from scripts.check_zoology_contractive_dplr import (
    backward_names,
    normalized_uuid,
)


EXPECTED_TRAIN_HASH = "647c64ece84984a23dfd817c4f277ea83840dbec57cc18bb6c9bf9eda7cc9a68"
EXPECTED_TEST_HASH = "4a8237ba8fe19aaff0d1d72de7b7f6505eaab59cd091442c2f463df34cce278f"
EXPECTED_P059_PARAMETERS = 599_672
EXPECTED_CANDIDATE_PARAMETERS = EXPECTED_P059_PARAMETERS + PREFILL_PARAMETERS


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
    if not torch.isfinite(left).all() or not torch.isfinite(right).all():
        raise RuntimeError("Nonfinite tensor in parity check")
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
        raise RuntimeError("P-FS2-015 requires exactly CUDA index 0")
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
    expected_sources = {
        "layer": EXPECTED_LAYER_SHA256,
        "chunk": EXPECTED_CHUNK_SHA256,
        "fused_recurrent": EXPECTED_RECURRENT_SHA256,
    }
    if source_hashes != expected_sources:
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
        arm="future_seed_momentum_conv_prefill",
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
    candidate = make_model(candidate_config, "future_seed_momentum_conv_prefill")
    load_matched_parent_state(candidate, parent_state)
    control_parameters = sum(parameter.numel() for parameter in control.parameters())
    candidate_parameters = sum(parameter.numel() for parameter in candidate.parameters())
    if control_parameters != EXPECTED_P059_PARAMETERS:
        raise RuntimeError(f"P059 parameter count drifted: {control_parameters}")
    if candidate_parameters != EXPECTED_CANDIDATE_PARAMETERS:
        raise RuntimeError(f"Candidate parameter count drifted: {candidate_parameters}")

    control_state = control.state_dict()
    candidate_state = candidate.state_dict()
    common_names = sorted(set(control_state) & set(candidate_state))
    common_max_diff = max(
        _max_abs(control_state[name], candidate_state[name]) for name in common_names
    )
    if common_max_diff != 0.0:
        raise RuntimeError(f"Candidate parent initialization drifted: {common_max_diff}")
    extra_names = sorted(set(candidate_state) - set(control_state))
    if extra_names != ["backbone.layers.1.sequence_mixer.conv_prefill_raw"]:
        raise RuntimeError(f"Unexpected candidate tensors: {extra_names}")

    control_mixers = [block.sequence_mixer for block in control.backbone.layers]
    candidate_mixers = [block.sequence_mixer for block in candidate.backbone.layers]
    if not all(type(mixer) is ZoologyMomentumDeltaFutureSeedMixer for mixer in control_mixers):
        raise RuntimeError("Control escaped exact P059 mixer")
    if not all(
        type(mixer) is ZoologyMomentumConvPrefillFutureSeedMixer
        for mixer in candidate_mixers
    ):
        raise RuntimeError("Candidate escaped exact conv-prefill mixer")
    for mixer in candidate_mixers:
        if type(mixer.layer) is not layer_class:
            raise RuntimeError("Candidate escaped pinned external Momentum layer")
        if mixer.state_size() != EXPECTED_STATE_VALUES_PER_LAYER:
            raise RuntimeError("Persistent state geometry changed")
        for convolution in (
            mixer.layer.q_conv1d,
            mixer.layer.k_conv1d,
            mixer.layer.v_conv1d,
        ):
            if convolution.backend != "triton" or tuple(convolution.kernel_size) != (4,):
                raise RuntimeError("Short-convolution provenance drifted")

    inputs, targets, _slices = next(iter(test_loader))
    inputs = inputs[:4].cuda()
    targets = targets[:4].cuda()
    control = control.cuda().eval()
    candidate = candidate.cuda().eval()
    with torch.no_grad():
        control_logits = control(inputs)
        candidate_logits = candidate(inputs)
    zero_gate_output_max_abs = _max_abs(control_logits, candidate_logits)
    if zero_gate_output_max_abs != 0.0:
        raise RuntimeError(
            f"Zero conv-prefill gate changed parent logits: {zero_gate_output_max_abs}"
        )

    generator = torch.Generator(device="cuda").manual_seed(15015)
    hidden = torch.randn(
        2, 128, 128, generator=generator, device="cuda", dtype=torch.float32
    )
    evidence = torch.randn(
        2,
        PREFILL_TOKENS,
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
    control_receiver = control.backbone.layers[1].sequence_mixer
    candidate_receiver = candidate.backbone.layers[1].sequence_mixer
    with torch.no_grad():
        control_output, control_terminal = control_receiver.forward_with_state(
            hidden, initial_state=initial_state
        )
        zero_prefill = candidate_receiver.make_conv_prefill(evidence)
        candidate_output, candidate_terminal = candidate_receiver.forward_with_prefill(
            hidden,
            initial_state=initial_state,
            initial_conv_state=zero_prefill,
        )
    zero_state_contract = {
        "output_max_abs": _max_abs(control_output, candidate_output),
        "terminal_max_abs": _max_abs(control_terminal, candidate_terminal),
        "raw_cache_rms": float(candidate_receiver.last_prefill_raw_cache_rms),
        "mixed_cache_rms": float(candidate_receiver.last_prefill_cache_rms),
    }
    if (
        zero_state_contract["output_max_abs"] != 0.0
        or zero_state_contract["terminal_max_abs"] != 0.0
        or zero_state_contract["raw_cache_rms"] <= 1e-4
        or zero_state_contract["mixed_cache_rms"] != 0.0
    ):
        raise RuntimeError(f"Zero/nonzero-state identity failed: {zero_state_contract}")

    candidate.train()
    candidate.zero_grad(set_to_none=True)
    logits = candidate(inputs)
    mask = targets != -100
    loss = F.cross_entropy(logits[mask], targets[mask])
    graph_names = backward_names(loss)
    chunk_backward_count = sum(
        "Chunkmode_ruleFunctionBackward" in name for name in graph_names
    )
    if chunk_backward_count != 2:
        raise RuntimeError(f"Expected two Momentum backwards, got {chunk_backward_count}")
    loss.backward()
    zero_gate_gradient = _gradient(
        candidate.backbone.layers[1].sequence_mixer.conv_prefill_raw,
        "conv_prefill_raw",
    )

    candidate_receiver = candidate.backbone.layers[1].sequence_mixer
    with torch.no_grad():
        candidate_receiver.conv_prefill_raw.fill_(0.25)
    evidence_a = evidence.detach().clone().requires_grad_(True)
    evidence_b = evidence.detach().flip(1)
    cache_a = candidate_receiver.make_conv_prefill(evidence_a)
    output_a, state_a = candidate_receiver.forward_with_prefill(
        hidden,
        initial_state=initial_state,
        initial_conv_state=cache_a,
    )
    cache_b = candidate_receiver.make_conv_prefill(evidence_b)
    output_b, state_b = candidate_receiver.forward_with_prefill(
        hidden,
        initial_state=initial_state,
        initial_conv_state=cache_b,
    )
    active_dependency = {
        "output_relative_rms": _relative_rms(output_a, output_b),
        "state_relative_rms": _relative_rms(state_a, state_b),
    }
    if (
        active_dependency["output_relative_rms"] <= 1e-4
        or active_dependency["state_relative_rms"] <= 1e-4
    ):
        raise RuntimeError(f"Prefill does not depend on ordered evidence: {active_dependency}")
    candidate_receiver.zero_grad(set_to_none=True)
    if evidence_a.grad is not None:
        evidence_a.grad = None
    output_a.float().square().mean().backward()
    evidence_gradient_rms = (
        0.0
        if evidence_a.grad is None
        else float(evidence_a.grad.float().square().mean().sqrt())
    )
    if evidence_gradient_rms <= 0 or not torch.isfinite(evidence_a.grad).all():
        raise RuntimeError("Active conv-prefill evidence has no finite gradient")

    result = {
        "status": "passed",
        "gpu": {"uuid": gpu_uuid, "name": gpu_name},
        "external_sha": EXPECTED_MDN_SHA,
        "source_hashes": source_hashes,
        "host_fla_compatibility": compatibility,
        "data_hashes": data_hashes,
        "parameters": {
            "control": control_parameters,
            "candidate": candidate_parameters,
            "delta": candidate_parameters - control_parameters,
        },
        "common_parent_tensor_count": len(common_names),
        "common_parent_max_abs": common_max_diff,
        "extra_parameter_names": extra_names,
        "zero_gate_output_max_abs": zero_gate_output_max_abs,
        "zero_nonzero_state_contract": zero_state_contract,
        "zero_gate_gradient_rms": zero_gate_gradient,
        "active_ordered_evidence_dependency": active_dependency,
        "active_evidence_gradient_rms": evidence_gradient_rms,
        "official_momentum_backward_count": chunk_backward_count,
        "prefill_tokens": PREFILL_TOKENS,
        "persistent_state_delta": 0,
        "scan_delta": 0,
        "fallback": None,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
