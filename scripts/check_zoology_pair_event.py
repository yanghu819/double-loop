from __future__ import annotations

import argparse
import hashlib
import importlib
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

from experiments.zoology_mqar.gdn2_futureseed import PINNED_FLA_SHA
from experiments.zoology_mqar.gdn2_pair_event import (
    EXPECTED_PARAMETER_DELTA,
    MODEL_HEADS,
    PAIR_RANK,
    ZoologyPairEventGDN2FutureSeedMixer,
    load_matched_parent_state,
    parent_parameter_hash,
)
from experiments.zoology_mqar.length_scaling import (
    build_config,
    dataset_hash,
    make_model,
    parameter_hash,
)
from scripts.check_zoology_contractive_dplr import (
    backward_names,
    git_head,
    normalized_uuid,
)
from scripts.check_zoology_gdn2_log_spd import python_tree_hash


EXPECTED_ZOOLOGY_SHA = "1ad20d193b6113cae1e8f3c655c300d7b4b3f4bb"
EXPECTED_TRAIN_HASH = "647c64ece84984a23dfd817c4f277ea83840dbec57cc18bb6c9bf9eda7cc9a68"
EXPECTED_TEST_HASH = "4a8237ba8fe19aaff0d1d72de7b7f6505eaab59cd091442c2f463df34cce278f"
EXPECTED_FLA_ROOT = Path(
    "/huyang2/double-loop/.cache/fla-versions/"
    "9c8e42e762fce087c27b673af4922795d9edb85e-0280db310981915e"
)
EXPECTED_PARENT_PARAMETERS = 661_584
EXPECTED_PARENT_PARAMETER_HASH = (
    "3e8fe038f1401735168783c2de1d9217a8807e88685ee12010142fc7b05e4c44"
)
MODEL_LAYERS = 2
HEAD_DIM = 32


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


def _finite_gradient(parameter: torch.Tensor, name: str) -> float:
    if parameter.grad is None:
        raise RuntimeError(f"Missing gradient: {name}")
    gradient = parameter.grad.detach().float()
    if not torch.isfinite(gradient).all() or float(gradient.abs().max()) == 0.0:
        raise RuntimeError(f"Inactive or nonfinite gradient: {name}")
    return float(gradient.square().mean().sqrt().item())


def _module_gradient(module: torch.nn.Module, name: str) -> float:
    gradients = [
        parameter.grad.detach().float().reshape(-1)
        for parameter in module.parameters()
        if parameter.grad is not None
    ]
    if not gradients:
        raise RuntimeError(f"Missing module gradient: {name}")
    gradient = torch.cat(gradients)
    if not torch.isfinite(gradient).all() or float(gradient.abs().max()) == 0.0:
        raise RuntimeError(f"Inactive or nonfinite module gradient: {name}")
    return float(gradient.square().mean().sqrt().item())


def _is_extra(name: str) -> bool:
    return ".sequence_mixer.event_" in name


def _mapped_parent_parameters(
    model: torch.nn.Module,
) -> dict[str, torch.nn.Parameter]:
    return {
        name: parameter
        for name, parameter in model.named_parameters()
        if not _is_extra(name)
    }


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
        raise RuntimeError("P-GDN3-057 requires exactly CUDA index 0")
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
    if os.environ.get("FLA_EXPECTED_SOURCE_SHA") != PINNED_FLA_SHA:
        raise RuntimeError("Pinned FLA SHA is missing")
    if os.environ.get("FLA_DISABLE_BACKEND_DISPATCH") != "1":
        raise RuntimeError("FLA backend dispatch must be disabled")
    if os.environ.get("FLA_CONV_BACKEND") != "triton":
        raise RuntimeError("Short convolution must use Triton")

    fla_root = Path(os.environ["FLA_SOURCE_ROOT"]).resolve()
    if fla_root != EXPECTED_FLA_ROOT:
        raise RuntimeError(f"Unexpected pinned FLA root: {fla_root}")
    gdn2_source = Path(inspect.getfile(GatedDeltaNet2)).resolve()
    chunk_module = importlib.import_module("fla.ops.gdn2.chunk")
    chunk_source = Path(chunk_module.__file__).resolve()
    if fla_root not in gdn2_source.parents or fla_root not in chunk_source.parents:
        raise RuntimeError(
            f"Official carrier escaped pinned FLA: {gdn2_source} {chunk_source}"
        )
    source_hash = hashlib.sha256(gdn2_source.read_bytes()).hexdigest()
    if source_hash != os.environ["FLA_GDN2_SOURCE_SHA256"]:
        raise RuntimeError(f"Unexpected GDN2 source hash: {source_hash}")
    ops_hash = python_tree_hash(fla_root / "fla" / "ops" / "gdn2")
    if ops_hash != os.environ["FLA_GDN2_OPS_SHA256"]:
        raise RuntimeError(f"Unexpected GDN2 recurrence hash: {ops_hash}")
    zoology_root = Path(os.environ["ZOOLOGY_ROOT"]).resolve()
    zoology_sha = git_head(zoology_root)
    if zoology_sha != EXPECTED_ZOOLOGY_SHA:
        raise RuntimeError(f"Unexpected Zoology SHA: {zoology_sha}")

    native_config = build_config(
        arm="future_seed_gdn2",
        sequence_length=1024,
        num_kv_pairs=4,
        max_epochs=10,
        batch_size=32,
    )
    candidate_config = build_config(
        arm="future_seed_pair_event_gdn2",
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
    if data_hashes != {
        "train": EXPECTED_TRAIN_HASH,
        "test": EXPECTED_TEST_HASH,
    }:
        raise RuntimeError(f"Directional MQAR data drifted: {data_hashes}")

    parent_state = torch.load(args.matched_init, map_location="cpu", weights_only=True)
    set_determinism(123)
    native = make_model(native_config, "future_seed_gdn2")
    native.load_state_dict(parent_state, strict=True)
    set_determinism(123)
    candidate = make_model(candidate_config, "future_seed_pair_event_gdn2")
    load_matched_parent_state(candidate, parent_state)
    native_parameters = sum(parameter.numel() for parameter in native.parameters())
    candidate_parameters = sum(
        parameter.numel() for parameter in candidate.parameters()
    )
    if native_parameters != EXPECTED_PARENT_PARAMETERS:
        raise RuntimeError(f"Native parameter count drifted: {native_parameters}")
    if candidate_parameters - native_parameters != EXPECTED_PARAMETER_DELTA:
        raise RuntimeError("Pair-event parameter delta changed")
    if parent_parameter_hash(candidate) != parameter_hash(native):
        raise RuntimeError("Candidate parent initialization differs from native")
    if parent_parameter_hash(candidate) != EXPECTED_PARENT_PARAMETER_HASH:
        raise RuntimeError("Frozen parent parameter hash changed")

    native_parent = dict(native.named_parameters())
    candidate_parent = _mapped_parent_parameters(candidate)
    if set(native_parent) != set(candidate_parent):
        raise RuntimeError("Candidate parent parameter names changed")
    parent_tensor_max_diff = max(
        _max_abs(native_parent[name], candidate_parent[name])
        for name in native_parent
    )
    if parent_tensor_max_diff != 0.0:
        raise RuntimeError(f"Candidate parent tensors changed: {parent_tensor_max_diff}")

    mixers = [block.sequence_mixer for block in candidate.backbone.layers]
    if len(mixers) != MODEL_LAYERS or not all(
        type(mixer) is ZoologyPairEventGDN2FutureSeedMixer for mixer in mixers
    ):
        raise RuntimeError("Contract requires two exact pair-event mixers")
    provenance = []
    for layer_index, mixer in enumerate(mixers):
        if type(mixer.layer) is not GatedDeltaNet2:
            raise RuntimeError(f"Layer {layer_index} carrier changed")
        if mixer.event_gate.numel() != MODEL_HEADS:
            raise RuntimeError(f"Layer {layer_index} gate shape changed")
        if float(mixer.event_gate.detach().abs().max()) != 0.0:
            raise RuntimeError(f"Layer {layer_index} gate is not zero-init")
        convolutions = {
            name: module.backend
            for name, module in mixer.layer.named_modules()
            if isinstance(module, ShortConvolution)
        }
        if len(convolutions) != 3 or set(convolutions.values()) != {"triton"}:
            raise RuntimeError(
                f"Layer {layer_index} convolution fallback: {convolutions}"
            )
        provenance.append(
            {
                "layer": layer_index,
                "carrier": type(mixer.layer).__qualname__,
                "operator": "ChunkGDN2FunctionBackward",
                "local_operator": "causal_shift_multiplicative_pair",
                "pair_rank": PAIR_RANK,
                "logical_main_scans": 1,
                "persistent_state_values": mixer.state_size(),
                "convolution_backends": convolutions,
            }
        )

    inputs, targets, _slices = next(iter(test_loader))
    inputs = inputs[:4].cuda()
    targets = targets[:4].cuda()
    native = native.cuda().eval()
    candidate = candidate.cuda().eval()
    with torch.no_grad():
        native_logits = native(inputs)
        candidate_logits = candidate(inputs)
    identity_output_max_diff = _max_abs(candidate_logits, native_logits)
    if identity_output_max_diff != 0.0:
        raise RuntimeError(
            f"Zero gate changed full output: {identity_output_max_diff}"
        )

    generator = torch.Generator(device="cuda").manual_seed(55055)
    incoming_rows = []
    for layer_index in range(MODEL_LAYERS):
        hidden = torch.randn(
            2,
            256,
            128,
            generator=generator,
            device="cuda",
        )
        incoming = torch.randn(
            2,
            MODEL_HEADS,
            HEAD_DIM,
            HEAD_DIM,
            generator=generator,
            device="cuda",
        )
        with torch.no_grad():
            native_output, native_state = native.backbone.layers[
                layer_index
            ].sequence_mixer.forward_with_state(hidden, initial_state=incoming)
            candidate_output, candidate_state = mixers[
                layer_index
            ].forward_with_state(hidden, initial_state=incoming)
        row = {
            "layer": layer_index,
            "output_max_diff": _max_abs(candidate_output, native_output),
            "state_max_diff": _max_abs(candidate_state, native_state),
        }
        if row["output_max_diff"] != 0.0 or row["state_max_diff"] != 0.0:
            raise RuntimeError(f"Nonzero incoming-state identity failed: {row}")
        incoming_rows.append(row)

    native.train().zero_grad(set_to_none=True)
    candidate.train().zero_grad(set_to_none=True)
    cpu_rng_state = torch.get_rng_state()
    cuda_rng_state = torch.cuda.get_rng_state()
    native_logits = native(inputs)
    torch.set_rng_state(cpu_rng_state)
    torch.cuda.set_rng_state(cuda_rng_state)
    candidate_logits = candidate(inputs)
    mask = targets != -100
    native_loss = F.cross_entropy(native_logits[mask], targets[mask])
    candidate_loss = F.cross_entropy(candidate_logits[mask], targets[mask])
    graph_names = backward_names(candidate_loss)
    official_backward_count = sum(
        "ChunkGDN2FunctionBackward" in name for name in graph_names
    )
    if official_backward_count != MODEL_LAYERS:
        raise RuntimeError(
            f"Expected two official GDN2 backward paths, got {official_backward_count}"
        )
    native_loss.backward()
    candidate_loss.backward()
    zero_gate_gradients = []
    for layer_index, mixer in enumerate(mixers):
        gate_gradient = _finite_gradient(
            mixer.event_gate,
            f"layer{layer_index}.event_gate",
        )
        if not torch.all(mixer.event_gate.grad.detach() != 0):
            raise RuntimeError(f"Layer {layer_index} has an inactive gate head")
        zero_gate_gradients.append(
            {"layer": layer_index, "event_gate": gate_gradient}
        )

    parent_gradient_max_diff = 0.0
    for name, parameter in dict(native.named_parameters()).items():
        candidate_parameter = _mapped_parent_parameters(candidate)[name]
        if parameter.grad is None and candidate_parameter.grad is None:
            continue
        if parameter.grad is None or candidate_parameter.grad is None:
            raise RuntimeError(f"Parent gradient presence differs: {name}")
        parent_gradient_max_diff = max(
            parent_gradient_max_diff,
            _max_abs(parameter.grad, candidate_parameter.grad),
        )
    if parent_gradient_max_diff != 0.0:
        raise RuntimeError(
            f"Zero gate changed parent gradients: {parent_gradient_max_diff}"
        )

    candidate.zero_grad(set_to_none=True)
    with torch.no_grad():
        for mixer in mixers:
            mixer.event_gate.fill_(torch.atanh(torch.tensor(0.125)).item())
    opened_logits = candidate(inputs)
    opened_loss = F.cross_entropy(opened_logits[mask], targets[mask])
    opened_loss.backward()
    opened_gradients = []
    for layer_index, mixer in enumerate(mixers):
        opened_gradients.append(
            {
                "layer": layer_index,
                "gate": _finite_gradient(mixer.event_gate, "event_gate"),
                "current": _module_gradient(
                    mixer.event_current, "event_current"
                ),
                "previous": _module_gradient(
                    mixer.event_previous, "event_previous"
                ),
                "output": _module_gradient(mixer.event_output, "event_output"),
            }
        )
    opened_output_relative_rms = _relative_rms(opened_logits, native_logits)
    if opened_output_relative_rms <= 1e-4:
        raise RuntimeError("Opened pair-event path is inactive")

    mixer = mixers[0]
    hidden = torch.randn(
        2,
        256,
        128,
        generator=generator,
        device="cuda",
    )
    with torch.no_grad():
        baseline = mixer._pair_delta(hidden)
        suffix_changed = hidden.clone()
        suffix_changed[:, 128:] = torch.randn(
            suffix_changed[:, 128:].shape,
            generator=generator,
            device="cuda",
        )
        suffix_output = mixer._pair_delta(suffix_changed)
        previous_changed = hidden.clone()
        previous_changed[:, 99] = torch.randn(
            previous_changed[:, 99].shape,
            generator=generator,
            device="cuda",
        )
        previous_output = mixer._pair_delta(previous_changed)
        two_back_changed = hidden.clone()
        two_back_changed[:, 98] = torch.randn(
            two_back_changed[:, 98].shape,
            generator=generator,
            device="cuda",
        )
        two_back_output = mixer._pair_delta(two_back_changed)
        current_changed = hidden.clone()
        current_changed[:, 100] = torch.randn(
            current_changed[:, 100].shape,
            generator=generator,
            device="cuda",
        )
        current_output = mixer._pair_delta(current_changed)
    causal_prefix_max_diff = _max_abs(baseline[:, :128], suffix_output[:, :128])
    previous_token_dependency = _relative_rms(
        previous_output[:, 100], baseline[:, 100]
    )
    current_token_dependency = _relative_rms(
        current_output[:, 100], baseline[:, 100]
    )
    two_back_isolation_max_diff = _max_abs(
        two_back_output[:, 100], baseline[:, 100]
    )
    first_token_delta_max_abs = float(baseline[:, 0].float().abs().max().item())
    if causal_prefix_max_diff != 0.0:
        raise RuntimeError("Pair-event branch leaked future tokens")
    if previous_token_dependency <= 1e-4:
        raise RuntimeError("Pair-event branch lacks previous-token dependency")
    if current_token_dependency <= 1e-4:
        raise RuntimeError("Pair-event branch lacks current-token dependency")
    if two_back_isolation_max_diff != 0.0:
        raise RuntimeError("Pair-event branch leaked beyond one causal shift")
    if first_token_delta_max_abs != 0.0:
        raise RuntimeError("Pair-event branch invented a previous token at t=0")

    result = {
        "status": "passed",
        "plan": "P-GDN3-057",
        "gpu": {"name": gpu_name, "uuid": gpu_uuid, "count": 1},
        "pinned_fla_sha": PINNED_FLA_SHA,
        "official_gdn2_source": str(gdn2_source),
        "official_gdn2_source_sha256": source_hash,
        "official_gdn2_ops_sha256": ops_hash,
        "zoology_sha": zoology_sha,
        "data_hashes": data_hashes,
        "candidate_parameters": candidate_parameters,
        "native_parameters": native_parameters,
        "parameter_delta": candidate_parameters - native_parameters,
        "persistent_state_delta": 0,
        "pair_rank": PAIR_RANK,
        "official_scans_per_layer": 1,
        "parent_parameter_hash": parent_parameter_hash(candidate.cpu()),
        "native_parameter_hash": parameter_hash(native.cpu()),
        "parent_tensor_max_diff": parent_tensor_max_diff,
        "official_module_provenance": provenance,
        "official_backward_count": official_backward_count,
        "zero_gate_model_output_max_diff": identity_output_max_diff,
        "zero_gate_nonzero_incoming_identity": incoming_rows,
        "zero_gate_parent_gradient_max_diff": parent_gradient_max_diff,
        "zero_gate_gradient_rms": zero_gate_gradients,
        "opened_gradient_rms": opened_gradients,
        "opened_output_relative_rms": opened_output_relative_rms,
        "causal_prefix_max_diff": causal_prefix_max_diff,
        "previous_token_dependency_relative_rms": previous_token_dependency,
        "current_token_dependency_relative_rms": current_token_dependency,
        "two_back_isolation_max_diff": two_back_isolation_max_diff,
        "first_token_delta_max_abs": first_token_delta_max_abs,
        "fallback": False,
    }
    if result["parent_parameter_hash"] != result["native_parameter_hash"]:
        raise RuntimeError("Parent hash changed while serializing contract")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
