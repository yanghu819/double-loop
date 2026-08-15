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
from experiments.zoology_mqar.gdn2_shared_eligibility import (
    COMPANION_STATE_VALUES,
    EXPECTED_PARAMETER_DELTA,
    HEAD_DIM,
    MODEL_HEADS,
    MODEL_WIDTH,
    SharedEligibilityBackbone,
    SharedEligibilityCompanionGDN2,
    ZoologySharedEligibilityFutureSeedMixer,
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
    return (
        name == "backbone.shared_event_projection.weight"
        or name == "backbone.shared_eligibility_conv.weight"
        or name.endswith(".sequence_mixer.layer.companion_read_logit")
        or name.endswith(".sequence_mixer.layer.companion_seed_logit")
    )


def _mapped_parent_parameters(
    model: torch.nn.Module,
) -> dict[str, torch.nn.Parameter]:
    mapped = {}
    for name, parameter in model.named_parameters():
        if _is_extra(name):
            continue
        parent_name = name.replace(
            ".sequence_mixer.layer.base.",
            ".sequence_mixer.layer.",
        )
        mapped[parent_name] = parameter
    return mapped


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
        raise RuntimeError("P-GDN3-056 requires exactly CUDA index 0")
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
        arm="future_seed_shared_eligibility_gdn2",
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
    candidate = make_model(
        candidate_config,
        "future_seed_shared_eligibility_gdn2",
    )
    load_matched_parent_state(candidate, parent_state)
    native_parameters = sum(parameter.numel() for parameter in native.parameters())
    candidate_parameters = sum(
        parameter.numel() for parameter in candidate.parameters()
    )
    if native_parameters != EXPECTED_PARENT_PARAMETERS:
        raise RuntimeError(f"Native parameter count drifted: {native_parameters}")
    if candidate_parameters - native_parameters != EXPECTED_PARAMETER_DELTA:
        raise RuntimeError("Shared-eligibility parameter delta changed")
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

    if not isinstance(candidate.backbone, SharedEligibilityBackbone):
        raise RuntimeError("Candidate backbone type changed")
    mixers = [block.sequence_mixer for block in candidate.backbone.layers]
    if len(mixers) != MODEL_LAYERS or not all(
        type(mixer) is ZoologySharedEligibilityFutureSeedMixer for mixer in mixers
    ):
        raise RuntimeError("Contract requires two exact shared-eligibility mixers")
    if candidate.backbone.shared_eligibility_conv.backend != "triton":
        raise RuntimeError("Shared eligibility trace escaped Triton")
    provenance = []
    for layer_index, mixer in enumerate(mixers):
        if type(mixer.layer) is not SharedEligibilityCompanionGDN2:
            raise RuntimeError(f"Layer {layer_index} wrapper changed")
        if type(mixer.layer.base) is not GatedDeltaNet2:
            raise RuntimeError(f"Layer {layer_index} carrier changed")
        if mixer.layer.companion_read_logit.numel() != MODEL_HEADS:
            raise RuntimeError(f"Layer {layer_index} read-gate shape changed")
        if mixer.layer.companion_seed_logit.numel() != MODEL_HEADS:
            raise RuntimeError(f"Layer {layer_index} seed-gate shape changed")
        if float(mixer.layer.companion_read_logit.detach().abs().max()) != 0.0:
            raise RuntimeError(f"Layer {layer_index} read gate is not zero-init")
        if float(mixer.layer.companion_seed_logit.detach().abs().max()) != 0.0:
            raise RuntimeError(f"Layer {layer_index} seed gate is not zero-init")
        convolutions = {
            name: module.backend
            for name, module in mixer.layer.base.named_modules()
            if isinstance(module, ShortConvolution)
        }
        if len(convolutions) != 3 or set(convolutions.values()) != {"triton"}:
            raise RuntimeError(
                f"Layer {layer_index} convolution fallback: {convolutions}"
            )
        provenance.append(
            {
                "layer": layer_index,
                "carrier": type(mixer.layer.base).__qualname__,
                "operator": "ChunkGDN2FunctionBackward",
                "logical_official_scans": 2,
                "main_state_values": mixer.state_size(),
                "companion_state_values": COMPANION_STATE_VALUES,
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
            f"Zero read gate changed full output: {identity_output_max_diff}"
        )

    generator = torch.Generator(device="cuda").manual_seed(56056)
    incoming_rows = []
    for layer_index in range(MODEL_LAYERS):
        hidden = torch.randn(
            2,
            256,
            MODEL_WIDTH,
            generator=generator,
            device="cuda",
        )
        address = torch.randn(
            hidden.shape,
            generator=generator,
            device="cuda",
        )
        main_incoming = torch.randn(
            2,
            MODEL_HEADS,
            HEAD_DIM,
            HEAD_DIM,
            generator=generator,
            device="cuda",
        )
        companion_incoming = torch.randn(
            main_incoming.shape,
            generator=generator,
            device="cuda",
        )
        with torch.no_grad():
            native_output, native_state = native.backbone.layers[
                layer_index
            ].sequence_mixer.forward_with_state(
                hidden,
                initial_state=main_incoming,
            )
            candidate_output, candidate_state, _companion_state = mixers[
                layer_index
            ].forward_with_dual_state(
                hidden,
                read_address=address,
                write_address=address,
                main_initial_state=main_incoming,
                companion_initial_state=companion_incoming,
            )
        row = {
            "layer": layer_index,
            "output_max_diff": _max_abs(candidate_output, native_output),
            "main_state_max_diff": _max_abs(candidate_state, native_state),
        }
        if row["output_max_diff"] != 0.0 or row["main_state_max_diff"] != 0.0:
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
    if official_backward_count != 2 * MODEL_LAYERS:
        raise RuntimeError(
            f"Expected four official GDN2 backward paths, got {official_backward_count}"
        )
    native_loss.backward()
    candidate_loss.backward()
    zero_gate_gradients = []
    for layer_index, mixer in enumerate(mixers):
        gate_gradient = _finite_gradient(
            mixer.layer.companion_read_logit,
            f"layer{layer_index}.companion_read_logit",
        )
        if not torch.all(mixer.layer.companion_read_logit.grad.detach() != 0):
            raise RuntimeError(f"Layer {layer_index} has an inactive read-gate head")
        zero_gate_gradients.append(
            {"layer": layer_index, "companion_read_gate": gate_gradient}
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
            f"Zero read gate changed parent gradients: {parent_gradient_max_diff}"
        )

    candidate.zero_grad(set_to_none=True)
    with torch.no_grad():
        for mixer in mixers:
            mixer.layer.companion_read_logit.fill_(
                torch.atanh(torch.tensor(0.125)).item()
            )
    opened_logits = candidate(inputs)
    opened_loss = F.cross_entropy(opened_logits[mask], targets[mask])
    opened_loss.backward()
    opened_gradients = []
    for layer_index, mixer in enumerate(mixers):
        row = {
            "layer": layer_index,
            "read_gate": _finite_gradient(
                mixer.layer.companion_read_logit,
                "companion_read_logit",
            ),
        }
        if layer_index > 0:
            row["seed_gate"] = _finite_gradient(
                mixer.layer.companion_seed_logit,
                "companion_seed_logit",
            )
        opened_gradients.append(row)
    projection_gradient = _module_gradient(
        candidate.backbone.shared_event_projection,
        "shared_event_projection",
    )
    trace_gradient = _module_gradient(
        candidate.backbone.shared_eligibility_conv,
        "shared_eligibility_conv",
    )
    opened_output_relative_rms = _relative_rms(opened_logits, native_logits)
    if opened_output_relative_rms <= 1e-4:
        raise RuntimeError("Opened shared-eligibility path is inactive")

    anchor = torch.randn(
        2,
        32,
        MODEL_WIDTH,
        generator=generator,
        device="cuda",
    )
    anchor = candidate.backbone.normalize_anchor(anchor)
    with torch.no_grad(), torch.autocast(
        device_type="cuda",
        dtype=torch.bfloat16,
    ):
        event_read = candidate.backbone.shared_event_projection(anchor)
        event_write, _ = candidate.backbone.shared_eligibility_conv(
            x=event_read,
            cache=None,
            output_final_state=False,
        )
    initial_trace_identity_max_diff = _max_abs(event_write, event_read)
    if initial_trace_identity_max_diff != 0.0:
        raise RuntimeError(
            f"Initial shared trace is not exact identity: {initial_trace_identity_max_diff}"
        )

    with torch.no_grad():
        candidate.backbone.shared_eligibility_conv.weight[:, 0, -2] = 0.25
    with torch.no_grad(), torch.autocast(
        device_type="cuda",
        dtype=torch.bfloat16,
    ):
        baseline, _ = candidate.backbone.shared_eligibility_conv(
            x=event_read,
            cache=None,
            output_final_state=False,
        )
        suffix_changed = event_read.clone()
        suffix_changed[:, 16:] = torch.randn(
            suffix_changed[:, 16:].shape,
            generator=generator,
            device="cuda",
            dtype=suffix_changed.dtype,
        )
        suffix_output, _ = candidate.backbone.shared_eligibility_conv(
            x=suffix_changed,
            cache=None,
            output_final_state=False,
        )
        previous_changed = event_read.clone()
        previous_changed[:, 5] = torch.randn(
            previous_changed[:, 5].shape,
            generator=generator,
            device="cuda",
            dtype=previous_changed.dtype,
        )
        previous_output, _ = candidate.backbone.shared_eligibility_conv(
            x=previous_changed,
            cache=None,
            output_final_state=False,
        )
    causal_prefix_max_diff = _max_abs(baseline[:, :16], suffix_output[:, :16])
    previous_token_dependency = _relative_rms(
        previous_output[:, 6],
        baseline[:, 6],
    )
    if causal_prefix_max_diff != 0.0:
        raise RuntimeError("Eligibility trace leaked future tokens")
    if previous_token_dependency <= 1e-4:
        raise RuntimeError("Eligibility trace lacks previous-token dependency")

    mixer = mixers[1]
    hidden = torch.randn(
        2,
        256,
        MODEL_WIDTH,
        generator=generator,
        device="cuda",
    )
    address = torch.randn(
        hidden.shape,
        generator=generator,
        device="cuda",
    )
    companion_incoming = torch.randn(
        2,
        MODEL_HEADS,
        HEAD_DIM,
        HEAD_DIM,
        generator=generator,
        device="cuda",
    )
    with torch.no_grad():
        state_output, _main_state, state_terminal = mixer.forward_with_dual_state(
            hidden,
            read_address=address,
            write_address=address,
            main_initial_state=None,
            companion_initial_state=companion_incoming,
        )
        zero_output, _zero_main, zero_terminal = mixer.forward_with_dual_state(
            hidden,
            read_address=address,
            write_address=address,
            main_initial_state=None,
            companion_initial_state=torch.zeros_like(companion_incoming),
        )
    companion_initial_output_dependency = _relative_rms(state_output, zero_output)
    companion_initial_state_dependency = _relative_rms(state_terminal, zero_terminal)
    if companion_initial_output_dependency <= 1e-4:
        raise RuntimeError("Receiving output ignores companion FutureSeed")
    if companion_initial_state_dependency <= 1e-4:
        raise RuntimeError("Companion transition ignores incoming state")

    result = {
        "status": "passed",
        "plan": "P-GDN3-056",
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
        "persistent_state_delta_per_layer": COMPANION_STATE_VALUES,
        "official_scans_per_layer": 2,
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
        "shared_projection_gradient_rms": projection_gradient,
        "shared_trace_gradient_rms": trace_gradient,
        "opened_output_relative_rms": opened_output_relative_rms,
        "initial_trace_identity_max_diff": initial_trace_identity_max_diff,
        "causal_prefix_max_diff": causal_prefix_max_diff,
        "previous_token_dependency_relative_rms": previous_token_dependency,
        "companion_initial_output_dependency_relative_rms": (
            companion_initial_output_dependency
        ),
        "companion_initial_state_dependency_relative_rms": (
            companion_initial_state_dependency
        ),
        "fallback": False,
    }
    if result["parent_parameter_hash"] != result["native_parameter_hash"]:
        raise RuntimeError("Parent hash changed while serializing contract")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
