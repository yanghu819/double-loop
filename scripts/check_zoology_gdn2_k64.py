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
from zoology.data.utils import prepare_data
from zoology.utils import set_determinism

from experiments.zoology_mqar.gdn2_futureseed import (
    PINNED_FLA_SHA,
    ZoologyGDN2FutureSeedMixer,
    futureseed_diagnostics,
)
from experiments.zoology_mqar.length_scaling import (
    build_config,
    dataset_hash,
    make_model,
)
from scripts.check_zoology_contractive_dplr import (
    backward_names,
    git_head,
    normalized_uuid,
    python_tree_hash,
)


EXPECTED_ZOOLOGY_SHA = "1ad20d193b6113cae1e8f3c655c300d7b4b3f4bb"
EXPECTED_TRAIN_HASH = "647c64ece84984a23dfd817c4f277ea83840dbec57cc18bb6c9bf9eda7cc9a68"
EXPECTED_TEST_HASH = "4a8237ba8fe19aaff0d1d72de7b7f6505eaab59cd091442c2f463df34cce278f"
EXPECTED_FLA_ROOT = Path(
    "/huyang2/double-loop/.cache/fla-versions/"
    "9c8e42e762fce087c27b673af4922795d9edb85e-0280db310981915e"
)
EXPECTED_PARAMETERS = 770_384
EXPECTED_STATE_VALUES = 8_192
MODEL_LAYERS = 2
MODEL_HEADS = 4
HEAD_KEY_DIM = 64
HEAD_VALUE_DIM = 32


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _gradient_rms(tensor: torch.Tensor) -> float:
    if tensor.grad is None or not torch.isfinite(tensor.grad).all():
        raise RuntimeError("Required tensor has a missing or non-finite gradient")
    value = float(tensor.grad.float().square().mean().sqrt().item())
    if value <= 0.0:
        raise RuntimeError("Required tensor has a zero gradient")
    return value


def _slice_gradient_rms(tensor: torch.Tensor, row_start: int) -> float:
    if tensor.grad is None or not torch.isfinite(tensor.grad).all():
        raise RuntimeError("Required projection has a missing or non-finite gradient")
    value = float(
        tensor.grad[row_start:].float().square().mean().sqrt().item()
    )
    if value <= 0.0:
        raise RuntimeError("Extra K64 projection rows received no gradient")
    return value


def _alter_values(inputs: torch.Tensor) -> torch.Tensor:
    altered = inputs.clone()
    value = (altered >= 160) & (altered < 256)
    altered[value] = 160 + ((altered[value] - 160 + 1) % 96)
    return altered


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-gpu-name", required=True)
    parser.add_argument("--expected-gpu-uuid", required=True)
    args = parser.parse_args()

    if os.environ.get("CUDA_VISIBLE_DEVICES") != "0":
        raise RuntimeError("CUDA_VISIBLE_DEVICES must be exactly 0")
    if torch.cuda.device_count() != 1 or torch.cuda.current_device() != 0:
        raise RuntimeError("P-GDN3-032 requires exactly CUDA index 0")
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
        raise RuntimeError(f"Unexpected pinned FLA root: {fla_root}")
    direct_url = json.loads(
        (fla_root / "flash_linear_attention-0.5.2.dist-info/direct_url.json")
        .read_text()
    )
    wheel_name = Path(direct_url["url"]).name
    wheel_hash = direct_url["archive_info"]["hashes"]["sha256"]
    if PINNED_FLA_SHA[:7] not in wheel_name:
        raise RuntimeError(f"FLA wheel provenance lacks pinned SHA: {wheel_name}")
    if wheel_hash != os.environ["FLA_WHEEL_SHA256"]:
        raise RuntimeError(f"Unexpected FLA wheel hash: {wheel_hash}")
    gdn2_source = Path(inspect.getfile(GatedDeltaNet2)).resolve()
    if gdn2_source != fla_root / "fla/layers/gdn2.py":
        raise RuntimeError(f"Unexpected GDN2 carrier source: {gdn2_source}")
    gdn2_source_hash = _sha256(gdn2_source)
    if gdn2_source_hash != os.environ["FLA_GDN2_SOURCE_SHA256"]:
        raise RuntimeError(f"Unexpected GDN2 source hash: {gdn2_source_hash}")
    gdn2_ops_hash, gdn2_ops_sources = python_tree_hash(fla_root / "fla/ops/gdn2")
    if gdn2_ops_hash != os.environ["FLA_GDN2_OPS_SHA256"]:
        raise RuntimeError(f"Unexpected GDN2 ops tree hash: {gdn2_ops_hash}")
    zoology_root = Path(os.environ["ZOOLOGY_ROOT"]).resolve()
    zoology_sha = git_head(zoology_root)
    if zoology_sha != EXPECTED_ZOOLOGY_SHA:
        raise RuntimeError(f"Unexpected Zoology SHA: {zoology_sha}")

    config = build_config(
        arm="future_seed_gdn2",
        sequence_length=1024,
        num_kv_pairs=4,
        max_epochs=10,
        batch_size=32,
        model_width=128,
        model_heads=MODEL_HEADS,
        gdn2_head_dim=HEAD_KEY_DIM,
        gdn2_expand_v=HEAD_VALUE_DIM / HEAD_KEY_DIM,
    )
    train_loader, test_loader = prepare_data(config.data)
    data_hashes = {
        "train": dataset_hash(train_loader),
        "test": dataset_hash(test_loader),
    }
    if data_hashes != {
        "train": EXPECTED_TRAIN_HASH,
        "test": EXPECTED_TEST_HASH,
    }:
        raise RuntimeError(f"Directional MQAR data drifted: {data_hashes}")

    set_determinism(123)
    model = make_model(config, "future_seed_gdn2")
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    if parameter_count != EXPECTED_PARAMETERS:
        raise RuntimeError(f"Candidate parameter count changed: {parameter_count}")
    mixers: list[ZoologyGDN2FutureSeedMixer] = []
    geometry = []
    for layer_index, block in enumerate(model.backbone.layers):
        mixer = block.sequence_mixer
        if type(mixer) is not ZoologyGDN2FutureSeedMixer:
            raise RuntimeError(f"Layer {layer_index} mixer changed: {type(mixer)}")
        if type(mixer.layer) is not GatedDeltaNet2:
            raise RuntimeError(f"Layer {layer_index} is not official GatedDeltaNet2")
        if mixer.layer.mode != "chunk" or mixer.state_size() != EXPECTED_STATE_VALUES:
            raise RuntimeError("Candidate scan mode or recurrent state size changed")
        row = {
            "num_heads": mixer.layer.num_heads,
            "num_v_heads": mixer.layer.num_v_heads,
            "head_k_dim": mixer.layer.head_k_dim,
            "head_v_dim": mixer.layer.head_v_dim,
            "state_values": mixer.state_size(),
            "q_conv_backend": getattr(mixer.layer.q_conv1d, "backend", None),
            "k_conv_backend": getattr(mixer.layer.k_conv1d, "backend", None),
            "v_conv_backend": getattr(mixer.layer.v_conv1d, "backend", None),
        }
        if row != {
            "num_heads": 4,
            "num_v_heads": 4,
            "head_k_dim": 64,
            "head_v_dim": 32,
            "state_values": 8192,
            "q_conv_backend": "triton",
            "k_conv_backend": "triton",
            "v_conv_backend": "triton",
        }:
            raise RuntimeError(f"Unexpected K64/V32 geometry: {row}")
        geometry.append(row)
        mixers.append(mixer)
    if len(mixers) != MODEL_LAYERS:
        raise RuntimeError("Candidate layer count changed")

    inputs, targets, _ = next(iter(test_loader))
    inputs = inputs[:2].cuda()
    targets = targets[:2].cuda()
    model = model.cuda().eval()
    altered = _alter_values(inputs.cpu()).cuda()
    with torch.no_grad():
        logits = model(inputs)
        altered_logits = model(altered)
    if not torch.isfinite(logits).all():
        raise RuntimeError("Candidate produced non-finite logits")
    positions = torch.arange(inputs.shape[1], device="cuda")
    future_mask = (targets != -100) & (positions[None, :] < inputs.shape[1] // 4)
    future_dependency = float(
        (logits[future_mask] - altered_logits[future_mask]).abs().mean().item()
    )
    if future_dependency <= 0.0:
        raise RuntimeError("Native FutureSeed has no future-value dependency")
    seed = futureseed_diagnostics(model)
    if seed["active_seed_routes"] != MODEL_LAYERS - 1:
        raise RuntimeError(f"Native FutureSeed route did not activate: {seed}")

    hidden = torch.randn(2, 65, 128, device="cuda", dtype=torch.float32)
    incoming = torch.randn(2, 4, 64, 32, device="cuda", dtype=torch.bfloat16)
    with torch.no_grad():
        direct_output, direct_state = mixers[0].forward_with_state(
            hidden,
            initial_state=incoming,
        )
    if direct_state.shape != incoming.shape or not torch.isfinite(direct_state).all():
        raise RuntimeError("Official K64 GDN2 nonzero-state path failed")

    model.train().zero_grad(set_to_none=True)
    train_logits = model(inputs)
    mask = targets != -100
    loss = F.cross_entropy(train_logits[mask], targets[mask])
    graph_names = backward_names(loss)
    official_backward_count = graph_names.count("ChunkGDN2FunctionBackward")
    if official_backward_count != MODEL_LAYERS:
        raise RuntimeError(
            f"Expected two official GDN2 backward nodes, got {official_backward_count}"
        )
    loss.backward()
    gradient_rows = []
    for layer_index, mixer in enumerate(mixers):
        layer = mixer.layer
        gradient_rows.append(
            {
                "layer": layer_index,
                "q_extra_rows": _slice_gradient_rms(layer.q_proj.weight, 128),
                "k_extra_rows": _slice_gradient_rms(layer.k_proj.weight, 128),
                "decay_extra_rows": _slice_gradient_rms(layer.f_proj[1].weight, 128),
                "erase_extra_rows": _slice_gradient_rms(layer.b_proj.weight, 128),
                "value": _gradient_rms(layer.v_proj.weight),
                "write": _gradient_rms(layer.w_proj.weight),
            }
        )
    future_seed_gradient = _gradient_rms(mixers[1].future_seed_logit)

    result = {
        "plan": "P-GDN3-032",
        "device": device.name,
        "device_uuid": device_uuid,
        "cuda_device_count": torch.cuda.device_count(),
        "zoology_sha": zoology_sha,
        "fla_sha": PINNED_FLA_SHA,
        "fla_wheel_sha256": wheel_hash,
        "fla_gdn2_source_sha256": gdn2_source_hash,
        "fla_gdn2_ops_tree_sha256": gdn2_ops_hash,
        "fla_gdn2_ops_sources": gdn2_ops_sources,
        "data_hashes": data_hashes,
        "parameters": parameter_count,
        "parameter_delta_vs_k32": parameter_count - 661_584,
        "geometry": geometry,
        "state_values_per_layer": EXPECTED_STATE_VALUES,
        "state_ratio_vs_k32": 2.0,
        "logical_scans_per_layer": 1,
        "official_backward_count": official_backward_count,
        "future_dependency_mean": future_dependency,
        "future_seed": seed,
        "future_seed_gradient_rms": future_seed_gradient,
        "gradient_rms": gradient_rows,
        "nonzero_incoming_state": {
            "output_rms": float(direct_output.float().square().mean().sqrt().item()),
            "terminal_state_rms": float(
                direct_state.float().square().mean().sqrt().item()
            ),
        },
        "peak_cuda_mem_bytes": torch.cuda.max_memory_allocated(),
        "no_fallback": True,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
