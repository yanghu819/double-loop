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

from experiments.zoology_mqar.gdn2_futureseed import (
    PINNED_FLA_SHA,
    ZoologyGDN2FutureSeedMixer,
)
from experiments.zoology_mqar.length_scaling import (
    build_config,
    dataset_hash,
    make_model,
    parameter_hash,
)
from experiments.zoology_mqar.producer_readout_futureseed import (
    EXPECTED_PARAMETER_DELTA,
    READOUT_RANK,
    RESIDUAL_CAP,
    ProducerNativeReadoutFusion,
    ProducerReadoutFutureSeedLanguageModel,
    load_matched_parent_state,
    parent_parameter_hash,
)


EXPECTED_ZOOLOGY_SHA = "1ad20d193b6113cae1e8f3c655c300d7b4b3f4bb"
EXPECTED_TRAIN_HASH = "647c64ece84984a23dfd817c4f277ea83840dbec57cc18bb6c9bf9eda7cc9a68"
EXPECTED_TEST_HASH = "4a8237ba8fe19aaff0d1d72de7b7f6505eaab59cd091442c2f463df34cce278f"
EXPECTED_NATIVE_PARAMETERS = 661_584
EXPECTED_CANDIDATE_PARAMETERS = 673_872


def normalized_uuid(value: str) -> str:
    return value.removeprefix("GPU-").lower()


def git_head(path: Path) -> str:
    return subprocess.check_output(
        ["git", "-C", str(path), "rev-parse", "HEAD"],
        text=True,
    ).strip()


def python_tree_hash(path: Path) -> str:
    digest = hashlib.sha256()
    for source in sorted(path.rglob("*.py")):
        digest.update(str(source.relative_to(path)).encode())
        digest.update(source.read_bytes())
    return digest.hexdigest()


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


def gradient_rms(parameter: torch.nn.Parameter, label: str) -> float:
    gradient = parameter.grad
    if gradient is None or not torch.isfinite(gradient).all():
        raise RuntimeError(f"Missing or non-finite gradient: {label}")
    value = float(gradient.float().square().mean().sqrt().item())
    return value


def relative_rms(left: torch.Tensor, right: torch.Tensor) -> float:
    numerator = (left.float() - right.float()).square().mean().sqrt()
    denominator = right.float().square().mean().sqrt().clamp_min(1e-8)
    return float((numerator / denominator).item())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-gpu-name", required=True)
    parser.add_argument("--expected-gpu-uuid", required=True)
    args = parser.parse_args()

    if os.environ.get("CUDA_VISIBLE_DEVICES") != "0":
        raise RuntimeError("CUDA_VISIBLE_DEVICES must be exactly 0")
    if torch.cuda.device_count() != 1 or torch.cuda.current_device() != 0:
        raise RuntimeError("P-FS2-010 requires exactly CUDA index 0")
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
    gdn2_ops_hash = python_tree_hash(fla_root / "fla" / "ops" / "gdn2")
    if gdn2_ops_hash != os.environ["FLA_GDN2_OPS_SHA256"]:
        raise RuntimeError("Official GDN2 recurrence source hash changed")

    zoology_root = Path(os.environ["ZOOLOGY_ROOT"]).resolve()
    zoology_sha = git_head(zoology_root)
    if zoology_sha != EXPECTED_ZOOLOGY_SHA:
        raise RuntimeError(f"Unexpected Zoology SHA: {zoology_sha}")

    control_config = build_config(
        arm="future_seed_gdn2",
        sequence_length=1024,
        num_kv_pairs=4,
        max_epochs=10,
        batch_size=32,
    )
    candidate_config = build_config(
        arm="future_seed_producer_readout_gdn2",
        sequence_length=1024,
        num_kv_pairs=4,
        max_epochs=10,
        batch_size=32,
    )
    train_loader, test_loader = prepare_data(control_config.data)
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
    control = make_model(control_config, "future_seed_gdn2")
    set_determinism(123)
    candidate = make_model(
        candidate_config,
        "future_seed_producer_readout_gdn2",
    )
    if type(candidate) is not ProducerReadoutFutureSeedLanguageModel:
        raise RuntimeError(f"Candidate model type changed: {type(candidate)}")
    load_matched_parent_state(candidate, control.state_dict())

    parameter_counts = {
        "control": sum(parameter.numel() for parameter in control.parameters()),
        "candidate": sum(parameter.numel() for parameter in candidate.parameters()),
    }
    if parameter_counts != {
        "control": EXPECTED_NATIVE_PARAMETERS,
        "candidate": EXPECTED_CANDIDATE_PARAMETERS,
    }:
        raise RuntimeError(f"Parameter counts changed: {parameter_counts}")
    if parameter_counts["candidate"] - parameter_counts["control"] != EXPECTED_PARAMETER_DELTA:
        raise RuntimeError("Producer-readout parameter delta changed")
    control_hash = parameter_hash(control)
    if parent_parameter_hash(candidate) != control_hash:
        raise RuntimeError("Candidate parent initialization differs from control")

    if len(candidate.backbone.readout_fusions) != 1:
        raise RuntimeError("P-FS2-010 requires exactly one readout edge")
    fusion = candidate.backbone.readout_fusions[0]
    if type(fusion) is not ProducerNativeReadoutFusion:
        raise RuntimeError(f"Readout fusion type changed: {type(fusion)}")
    if fusion.rank != READOUT_RANK:
        raise RuntimeError("Readout rank changed")
    if sum(parameter.numel() for parameter in fusion.parameters()) != EXPECTED_PARAMETER_DELTA:
        raise RuntimeError("Readout fusion parameter count changed")
    if fusion.out_proj.weight.detach().abs().max().item() != 0.0:
        raise RuntimeError("Readout output projection is not exact zero")

    provenance = []
    mixers: list[ZoologyGDN2FutureSeedMixer] = []
    for layer_index, block in enumerate(candidate.backbone.layers):
        mixer = block.sequence_mixer
        if type(mixer) is not ZoologyGDN2FutureSeedMixer:
            raise RuntimeError(f"Layer {layer_index} mixer changed: {type(mixer)}")
        if type(mixer.layer) is not GatedDeltaNet2:
            raise RuntimeError(f"Layer {layer_index} carrier is not exact GDN2")
        convolutions = {
            name: getattr(mixer.layer, name).backend
            for name in ("q_conv1d", "k_conv1d", "v_conv1d")
        }
        if not all(
            isinstance(getattr(mixer.layer, name), ShortConvolution)
            for name in convolutions
        ) or set(convolutions.values()) != {"triton"}:
            raise RuntimeError(f"Layer {layer_index} convolution fallback: {convolutions}")
        provenance.append(
            {
                "layer": layer_index,
                "carrier": type(mixer.layer).__qualname__,
                "operator": "ChunkGDN2FunctionBackward",
                "convolution_backends": convolutions,
                "state_values": mixer.state_size(),
            }
        )
        mixers.append(mixer)

    control = control.cuda().eval()
    candidate = candidate.cuda().eval()
    inputs, targets, _slices = next(iter(test_loader))
    inputs = inputs[:2].cuda()
    targets = targets[:2].cuda()
    with torch.no_grad():
        control_logits = control(inputs)
        candidate_logits = candidate(inputs)
    zero_init_max_error = float(
        (candidate_logits.float() - control_logits.float()).abs().max().item()
    )
    if zero_init_max_error != 0.0:
        raise RuntimeError(
            f"Zero-init producer readout lost exact parent identity: {zero_init_max_error}"
        )

    candidate.train().zero_grad(set_to_none=True)
    logits = candidate(inputs)
    mask = targets != -100
    loss = F.cross_entropy(logits[mask], targets[mask])
    graph = backward_names(loss)
    official_backward_count = graph.count("ChunkGDN2FunctionBackward")
    if official_backward_count != 2:
        raise RuntimeError(
            f"Expected two official GDN2 backward paths, got {official_backward_count}"
        )
    loss.backward()
    stage1_gradients = {
        "hidden_projection": gradient_rms(
            fusion.hidden_proj.weight,
            "stage1.hidden_proj",
        ),
        "read_projection": gradient_rms(
            fusion.read_proj.weight,
            "stage1.read_proj",
        ),
        "output_projection": gradient_rms(
            fusion.out_proj.weight,
            "stage1.out_proj",
        ),
    }
    if stage1_gradients["output_projection"] <= 1e-8:
        raise RuntimeError(f"Zero-init output projection has no gradient: {stage1_gradients}")
    if stage1_gradients["hidden_projection"] != 0.0 or stage1_gradients["read_projection"] != 0.0:
        raise RuntimeError(f"Zero-init gradient staging changed: {stage1_gradients}")

    with torch.no_grad():
        values = torch.linspace(
            -1e-3,
            1e-3,
            fusion.out_proj.weight.numel(),
            device="cuda",
            dtype=fusion.out_proj.weight.dtype,
        ).reshape_as(fusion.out_proj.weight)
        fusion.out_proj.weight.copy_(values)
    candidate.zero_grad(set_to_none=True)
    logits = candidate(inputs)
    stage2_loss = F.cross_entropy(logits[mask], targets[mask])
    stage2_loss.backward()
    stage2_gradients = {
        "hidden_projection": gradient_rms(
            fusion.hidden_proj.weight,
            "stage2.hidden_proj",
        ),
        "read_projection": gradient_rms(
            fusion.read_proj.weight,
            "stage2.read_proj",
        ),
        "output_projection": gradient_rms(
            fusion.out_proj.weight,
            "stage2.out_proj",
        ),
    }
    if min(stage2_gradients.values()) <= 1e-8:
        raise RuntimeError(f"Two-stage fusion gradients are inactive: {stage2_gradients}")
    future_seed_gradient = gradient_rms(
        mixers[1].future_seed_logit,
        "receiver.future_seed_logit",
    )
    if future_seed_gradient <= 1e-8:
        raise RuntimeError("Native FutureSeed path has no gradient")

    generator = torch.Generator(device="cuda").manual_seed(52010)
    hidden = torch.randn(
        3,
        64,
        128,
        generator=generator,
        device="cuda",
    )
    state = torch.randn(
        3,
        4,
        32,
        32,
        generator=generator,
        device="cuda",
        dtype=torch.bfloat16,
    )
    with torch.no_grad():
        decoded = fusion.decode(mixers[0], hidden, state)
        decoded_shuffled = fusion.decode(mixers[0], hidden, state.roll(1, 0))
        decoded_zero = fusion.decode(mixers[0], hidden, torch.zeros_like(state))
        fused, feature, residual = fusion.fuse(hidden, decoded)
        _fused_shuffled, _feature_shuffled, residual_shuffled = fusion.fuse(
            hidden,
            decoded_shuffled,
        )
        permuted = torch.tensor([2, 0, 1], device="cuda")
        decoded_permuted = fusion.decode(
            mixers[0],
            hidden.index_select(0, permuted),
            state.index_select(0, permuted),
        )
    state_dependency_relative_rms = relative_rms(decoded, decoded_shuffled)
    residual_state_dependency_relative_rms = relative_rms(
        residual,
        residual_shuffled,
    )
    zero_state_readout_max_abs = float(decoded_zero.float().abs().max().item())
    batch_permutation_max_error = float(
        (
            decoded_permuted.float()
            - decoded.index_select(0, permuted).float()
        ).abs().max().item()
    )
    hidden_token_rms = hidden.float().square().mean(dim=-1).sqrt().clamp_min(1e-8)
    residual_bound_ratio = float(
        (residual.float().abs().amax(dim=-1) / hidden_token_rms).max().item()
    )
    read_board_std = float(
        decoded.float().square().mean(dim=(1, 2)).sqrt().std(unbiased=False).item()
    )
    read_token_std = float(
        decoded.float().square().mean(dim=(0, 2)).sqrt().std(unbiased=False).item()
    )
    if state_dependency_relative_rms <= 1e-3:
        raise RuntimeError("Producer-native decode ignores transferred state")
    if residual_state_dependency_relative_rms <= 1e-3:
        raise RuntimeError("Fusion residual ignores transferred state")
    if zero_state_readout_max_abs != 0.0:
        raise RuntimeError(f"Zero state produced readout: {zero_state_readout_max_abs}")
    if batch_permutation_max_error != 0.0:
        raise RuntimeError(f"Batch equivariance changed: {batch_permutation_max_error}")
    if residual_bound_ratio > RESIDUAL_CAP + 1e-4:
        raise RuntimeError(f"Fusion residual exceeded cap: {residual_bound_ratio}")
    if read_board_std <= 1e-6 or read_token_std <= 1e-6:
        raise RuntimeError("Producer-native readout lacks board/token variation")
    if not torch.isfinite(fused).all() or not torch.isfinite(feature).all():
        raise RuntimeError("Producer-native fusion is non-finite")

    result = {
        "status": "passed",
        "plan": "P-FS2-010",
        "gpu": {"name": device.name, "uuid": device_uuid, "count": 1},
        "pinned_fla_sha": PINNED_FLA_SHA,
        "gdn2_source": str(gdn2_source),
        "gdn2_ops_tree_sha256": gdn2_ops_hash,
        "zoology_sha": zoology_sha,
        "data_hashes": data_hashes,
        "parameter_counts": parameter_counts,
        "parameter_delta": parameter_counts["candidate"] - parameter_counts["control"],
        "control_parameter_hash": control_hash,
        "candidate_parent_parameter_hash": parent_parameter_hash(candidate),
        "official_module_provenance": provenance,
        "official_chunk_backward_count": official_backward_count,
        "zero_init_parent_max_error": zero_init_max_error,
        "stage1_gradients": stage1_gradients,
        "stage2_gradients": stage2_gradients,
        "future_seed_gradient_rms": future_seed_gradient,
        "state_dependency_relative_rms": state_dependency_relative_rms,
        "residual_state_dependency_relative_rms": residual_state_dependency_relative_rms,
        "zero_state_readout_max_abs": zero_state_readout_max_abs,
        "batch_permutation_max_error": batch_permutation_max_error,
        "residual_bound_ratio": residual_bound_ratio,
        "read_board_std": read_board_std,
        "read_token_std": read_token_std,
        "readout_rank": READOUT_RANK,
        "residual_cap": RESIDUAL_CAP,
        "persistent_state_delta": 0,
        "official_scan_delta": 0,
        "fallback": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
