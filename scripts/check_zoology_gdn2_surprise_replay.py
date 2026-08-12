from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import os
import subprocess
from pathlib import Path

import torch

from fla.layers.gdn2 import GatedDeltaNet2
from fla.modules.convolution import ShortConvolution
from zoology.data.utils import prepare_data
from zoology.utils import set_determinism

from experiments.zoology_mqar.gdn2_committed_delta import capture_committed_edit
from experiments.zoology_mqar.gdn2_futureseed import PINNED_FLA_SHA
from experiments.zoology_mqar.gdn2_surprise_replay import (
    EVENT_TAPE_SIZE,
    ZoologyEventTapeGDN2FutureSeedMixer,
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


def normalized_uuid(value: str) -> str:
    return value.removeprefix("GPU-").lower()


def git_head(path: Path) -> str:
    return subprocess.check_output(
        ["git", "-C", str(path), "rev-parse", "HEAD"], text=True
    ).strip()


def python_tree_hash(path: Path) -> str:
    digest = hashlib.sha256()
    for source in sorted(path.rglob("*.py")):
        digest.update(str(source.relative_to(path)).encode())
        digest.update(source.read_bytes())
    return digest.hexdigest()


def max_diff(left: torch.Tensor, right: torch.Tensor, label: str) -> float:
    difference = (left.float() - right.float()).abs()
    if not torch.isfinite(difference).all():
        raise RuntimeError(f"Non-finite {label} difference")
    return float(difference.max().item())


def backward_names(tensor: torch.Tensor) -> list[str]:
    names = []
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-gpu-name", required=True)
    parser.add_argument("--expected-gpu-uuid", required=True)
    args = parser.parse_args()

    if os.environ.get("CUDA_VISIBLE_DEVICES") != "0":
        raise RuntimeError("CUDA_VISIBLE_DEVICES must be exactly 0")
    if torch.cuda.device_count() != 1:
        raise RuntimeError(f"Expected one visible GPU, got {torch.cuda.device_count()}")
    if torch.cuda.current_device() != 0:
        raise RuntimeError("The only visible GPU must be CUDA index 0")
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
    gdn2_source = Path(inspect.getfile(GatedDeltaNet2)).resolve()
    if fla_root not in gdn2_source.parents:
        raise RuntimeError(f"Unexpected GDN2 source: {gdn2_source}")
    if hashlib.sha256(gdn2_source.read_bytes()).hexdigest() != os.environ[
        "FLA_GDN2_SOURCE_SHA256"
    ]:
        raise RuntimeError("Official GDN2 layer source hash changed")
    if python_tree_hash(fla_root / "fla" / "ops" / "gdn2") != os.environ[
        "FLA_GDN2_OPS_SHA256"
    ]:
        raise RuntimeError("Official GDN2 recurrence source hash changed")
    zoology_root = Path(os.environ["ZOOLOGY_ROOT"]).resolve()
    if git_head(zoology_root) != EXPECTED_ZOOLOGY_SHA:
        raise RuntimeError("Unexpected Zoology SHA")

    arms = (
        "future_seed_gdn2",
        "future_seed_gdn2_recency_replay",
        "future_seed_gdn2_surprise_replay",
    )
    configs = {
        arm: build_config(
            arm=arm,
            sequence_length=1024,
            num_kv_pairs=4,
            max_epochs=10,
            batch_size=32,
        )
        for arm in arms
    }
    train_loader, test_loader = prepare_data(configs[arms[0]].data)
    data_hashes = {
        "train": dataset_hash(train_loader),
        "test": dataset_hash(test_loader),
    }
    if data_hashes != {"train": EXPECTED_TRAIN_HASH, "test": EXPECTED_TEST_HASH}:
        raise RuntimeError(f"Directional MQAR data drifted: {data_hashes}")

    models = {}
    for arm in arms:
        set_determinism(123)
        models[arm] = make_model(configs[arm], arm)
    parameter_counts = {
        arm: sum(parameter.numel() for parameter in model.parameters())
        for arm, model in models.items()
    }
    parameter_hashes = {arm: parameter_hash(model) for arm, model in models.items()}
    if len(set(parameter_counts.values())) != 1 or len(set(parameter_hashes.values())) != 1:
        raise RuntimeError(
            f"Event replay changed parent parameters: {parameter_counts} {parameter_hashes}"
        )
    if set(parameter_counts.values()) != {661_584}:
        raise RuntimeError(f"Historical parent parameter count changed: {parameter_counts}")
    named = {arm: dict(model.named_parameters()) for arm, model in models.items()}
    if len({tuple(parameters) for parameters in named.values()}) != 1:
        raise RuntimeError("Event-replay parent parameter names changed")
    parameter_max_diff = max(
        float(
            (named[arms[0]][name] - named[arm][name]).abs().max().item()
        )
        for arm in arms[1:]
        for name in named[arms[0]]
    )
    if parameter_max_diff != 0.0:
        raise RuntimeError(f"Event-replay parent tensors changed: {parameter_max_diff}")
    named_buffers = {arm: dict(model.named_buffers()) for arm, model in models.items()}
    if len({tuple(buffers) for buffers in named_buffers.values()}) != 1:
        raise RuntimeError("Event-replay parent buffer names changed")
    buffer_differences = [
        float(
            (
                named_buffers[arms[0]][name].float()
                - named_buffers[arm][name].float()
            )
            .abs()
            .max()
            .item()
        )
        for arm in arms[1:]
        for name in named_buffers[arms[0]]
        if named_buffers[arms[0]][name].numel() > 0
    ]
    buffer_max_diff = max(buffer_differences, default=0.0)
    if buffer_max_diff != 0.0:
        raise RuntimeError(f"Event-replay parent buffers changed: {buffer_max_diff}")

    projection_names = (
        "q_proj",
        "k_proj",
        "v_proj",
        "f_proj",
        "b_proj",
        "w_proj",
        "g_proj",
        "o_proj",
    )
    provenance = []
    for arm in arms:
        for layer_index, block in enumerate(models[arm].backbone.layers):
            mixer = block.sequence_mixer
            layer = mixer.layer
            if type(layer) is not GatedDeltaNet2:
                raise RuntimeError(
                    f"{arm} layer {layer_index} is not exact official GatedDeltaNet2"
                )
            control_layer = models[arms[0]].backbone.layers[
                layer_index
            ].sequence_mixer.layer
            projection_types = {}
            for name in projection_names:
                candidate_projection = getattr(layer, name)
                control_projection = getattr(control_layer, name)
                if type(candidate_projection) is not type(control_projection):
                    raise RuntimeError(
                        f"Official projection type changed: {arm} layer={layer_index} {name}"
                    )
                projection_types[name] = type(candidate_projection).__qualname__
            conv_backends = {}
            for name in ("q_conv1d", "k_conv1d", "v_conv1d"):
                convolution = getattr(layer, name)
                if not isinstance(convolution, ShortConvolution):
                    raise RuntimeError(
                        f"Official ShortConvolution type changed: {arm} layer={layer_index} {name}"
                    )
                conv_backends[name] = getattr(convolution, "backend", None)
            if set(conv_backends.values()) != {"triton"}:
                raise RuntimeError(
                    f"Short-conv fallback: {arm} layer={layer_index} {conv_backends}"
                )
            provenance.append(
                {
                    "arm": arm,
                    "layer": layer_index,
                    "layer_type": type(layer).__qualname__,
                    "projection_types": projection_types,
                    "conv_backends": conv_backends,
                }
            )

    control = models[arms[0]].cuda().eval()
    surprise = models[arms[2]].cuda().eval()
    generator = torch.Generator(device="cuda").manual_seed(52027)
    hidden = torch.randn(2, 128, 128, generator=generator, device="cuda")
    incoming = torch.randn(
        2, 4, 32, 32, generator=generator, device="cuda", dtype=torch.float32
    )
    parity = []
    for layer_index in range(2):
        control_mixer = control.backbone.layers[layer_index].sequence_mixer
        surprise_mixer = surprise.backbone.layers[layer_index].sequence_mixer
        if not isinstance(surprise_mixer, ZoologyEventTapeGDN2FutureSeedMixer):
            raise RuntimeError("Surprise mixer type changed")
        for label, state in (("zero", None), ("nonzero", incoming)):
            with torch.no_grad(), capture_committed_edit() as control_edit:
                control_output, control_state = control_mixer.forward_with_state(
                    hidden, initial_state=state
                )
            with torch.no_grad():
                surprise_output, surprise_state, surprise_edit = (
                    surprise_mixer.forward_with_committed_edit(
                        hidden, initial_state=state
                    )
                )
            if len(control_edit) != 1:
                raise RuntimeError("Control committed-edit capture count changed")
            row = {
                "layer": layer_index,
                "label": label,
                "output_max_diff": max_diff(control_output, surprise_output, label),
                "state_max_diff": max_diff(control_state, surprise_state, label),
                "committed_edit_max_diff": max_diff(
                    control_edit[0], surprise_edit, f"{label} committed edit"
                ),
            }
            if any(
                row[name] != 0.0
                for name in row
                if name not in ("layer", "label")
            ):
                raise RuntimeError(f"Main official scan parity failed: {row}")
            parity.append(row)

    surprise_mixer = surprise.backbone.layers[0].sequence_mixer

    synthetic_hidden = torch.randn(
        2, 64, 128, generator=generator, device="cuda", dtype=torch.float32
    )
    synthetic_edit = torch.ones(2, 64, 4, 32, device="cuda")
    synthetic_edit[0, :EVENT_TAPE_SIZE] = 10.0
    synthetic_edit[1, EVENT_TAPE_SIZE : 2 * EVENT_TAPE_SIZE] = 10.0
    surprise_evidence = surprise_mixer.select_event_tape(
        synthetic_hidden, synthetic_edit
    )
    surprise_indices = surprise_mixer.last_selected_indices
    recency_mixer = models[arms[1]].cuda().eval().backbone.layers[0].sequence_mixer
    recency_evidence = recency_mixer.select_event_tape(
        synthetic_hidden, synthetic_edit
    )
    recency_indices = recency_mixer.last_selected_indices
    expected_surprise = torch.stack(
        (
            torch.arange(0, 16, device="cuda"),
            torch.arange(16, 32, device="cuda"),
        )
    )
    expected_recency = torch.arange(48, 64, device="cuda").expand(2, -1)
    if not torch.equal(surprise_indices, expected_surprise):
        raise RuntimeError(f"Surprise admission changed: {surprise_indices}")
    if not torch.equal(recency_indices, expected_recency):
        raise RuntimeError(f"Recency admission changed: {recency_indices}")
    expected_surprise_evidence = synthetic_hidden.gather(
        1, expected_surprise.unsqueeze(-1).expand(-1, -1, 128)
    )
    expected_recency_evidence = synthetic_hidden.gather(
        1, expected_recency.unsqueeze(-1).expand(-1, -1, 128)
    )
    if max_diff(
        surprise_evidence,
        expected_surprise_evidence,
        "surprise gathered evidence",
    ) != 0.0:
        raise RuntimeError("Surprise event gather is not batch-local and chronological")
    if max_diff(
        recency_evidence,
        expected_recency_evidence,
        "recency gathered evidence",
    ) != 0.0:
        raise RuntimeError("Recency event gather is not batch-local and chronological")

    replay_seed = surprise.backbone.layers[1].sequence_mixer.replay_seed(
        synthetic_hidden[:, :EVENT_TAPE_SIZE]
    )
    shuffled_seed = surprise.backbone.layers[1].sequence_mixer.replay_seed(
        synthetic_hidden[:, :EVENT_TAPE_SIZE].flip(1)
    )
    replay_dependency = max_diff(replay_seed, shuffled_seed, "replay dependency")
    if replay_dependency < 1e-4:
        raise RuntimeError("Receiver replay does not depend on ordered evidence")

    inputs, _targets, _slices = next(iter(test_loader))
    inputs = inputs[:2].cuda()
    surprise.train()
    surprise.zero_grad(set_to_none=True)
    logits = surprise(inputs)
    if not torch.isfinite(logits).all():
        raise RuntimeError("Full event-replay logits are non-finite")
    loss = logits.float().square().mean()
    graph_names = backward_names(loss)
    official_backward_count = graph_names.count("ChunkGDN2FunctionBackward")
    if official_backward_count != 3:
        raise RuntimeError(
            f"Expected three official chunk backward paths, got {official_backward_count}"
        )
    producer_mixer = surprise.backbone.layers[0].sequence_mixer
    receiver_mixer = surprise.backbone.layers[1].sequence_mixer
    selected_evidence = producer_mixer.last_selected_evidence
    replay_input = receiver_mixer.last_replay_input
    if selected_evidence is None or replay_input is None:
        raise RuntimeError("Full forward did not populate the event-tape edge")
    full_wiring_max_diff = max_diff(
        selected_evidence, replay_input, "full producer-to-receiver event tape"
    )
    if full_wiring_max_diff != 0.0 or selected_evidence is not replay_input:
        raise RuntimeError(
            "Receiver replay input is not the exact producer-selected evidence: "
            f"diff={full_wiring_max_diff} identity={selected_evidence is replay_input}"
        )
    loss.backward()
    selected_evidence_gradient_rms = (
        selected_evidence.grad.float().square().mean().sqrt().item()
        if selected_evidence.grad is not None
        else 0.0
    )
    if (
        selected_evidence.grad is None
        or not torch.isfinite(selected_evidence.grad).all()
        or selected_evidence_gradient_rms <= 0
    ):
        raise RuntimeError("Selected producer evidence has no finite nonzero gradient")
    finite_nonzero_gradient_count = sum(
        int(
            parameter.grad is not None
            and torch.isfinite(parameter.grad).all()
            and parameter.grad.abs().max().item() > 0
        )
        for parameter in surprise.parameters()
    )
    if finite_nonzero_gradient_count == 0:
        raise RuntimeError("Event-replay full model has no finite nonzero gradients")

    result = {
        "status": "passed",
        "gpu": {"name": device.name, "uuid": device_uuid, "count": 1},
        "pinned_fla_sha": PINNED_FLA_SHA,
        "zoology_sha": EXPECTED_ZOOLOGY_SHA,
        "data_hashes": data_hashes,
        "parameter_counts": parameter_counts,
        "parameter_hashes": parameter_hashes,
        "parameter_max_diff": parameter_max_diff,
        "buffer_max_diff": buffer_max_diff,
        "official_module_provenance": provenance,
        "main_scan_parity": parity,
        "event_tape_size": EVENT_TAPE_SIZE,
        "surprise_indices": surprise_indices[0].tolist(),
        "recency_indices": recency_indices[0].tolist(),
        "replay_order_dependency_max_diff": replay_dependency,
        "full_event_tape_wiring_max_diff": full_wiring_max_diff,
        "selected_evidence_gradient_rms": selected_evidence_gradient_rms,
        "official_chunk_backward_count": official_backward_count,
        "finite_nonzero_gradient_count": finite_nonzero_gradient_count,
        "main_recurrence_changed": False,
        "new_parameters": 0,
        "persistent_state_delta": 0,
        "transient_event_tape_values_per_board": 2048,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
