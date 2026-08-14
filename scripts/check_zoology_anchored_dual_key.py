from __future__ import annotations

import argparse
import hashlib
import importlib
import inspect
import json
import os
from pathlib import Path

import torch
import torch.nn.functional as F

from fla.layers.gdn2 import GatedDeltaNet2
from fla.modules.convolution import ShortConvolution
from fla.ops.generalized_delta_rule.dplr import chunk_dplr_delta_rule
from zoology.data.utils import prepare_data
from zoology.utils import set_determinism

from experiments.zoology_mqar.anchored_dual_key_futureseed import (
    COSINE_FLOOR,
    RESIDUAL_CAP,
    AnchoredDualKeyGDN2,
    ZoologyAnchoredDualKeyGDN2FutureSeedMixer,
    _rms_ratio,
    parent_parameter_hash,
)
from experiments.zoology_mqar.decoupled_key_futureseed import (
    DirectDecoupledKeyGDN2,
)
from experiments.zoology_mqar.gdn2_futureseed import PINNED_FLA_SHA
from experiments.zoology_mqar.length_scaling import (
    build_config,
    dataset_hash,
    make_model,
    parameter_hash,
)
from scripts.check_zoology_contractive_dplr import (
    backward_names,
    finite_gradient_rms,
    finite_module_gradient_rms,
    git_head,
    max_abs_difference,
    normalized_uuid,
    python_tree_hash,
    relative_rms,
    run_official,
    transition_diagnostics,
)
from scripts.check_zoology_decoupled_key import (
    EXPECTED_DPLR_TREE_SHA256,
    EXPECTED_FLA_ROOT,
    EXPECTED_TEST_HASH,
    EXPECTED_TRAIN_HASH,
    EXPECTED_ZOOLOGY_SHA,
    _explicit_gdn2,
    _module_max_error,
    _run_gdn2,
)


EXPECTED_PARAMETERS = 695_376
EXPECTED_NATIVE_PARAMETERS = 661_584
EXPECTED_MIXER_PARAMETERS = 133_416
EXPECTED_PARAMETER_DELTA = 33_792
EXPECTED_STATE_VALUES = 4_096
MODEL_LAYERS = 2
MODEL_HEADS = 4
HEAD_DIM = 32


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-gpu-name", required=True)
    parser.add_argument("--expected-gpu-uuid", required=True)
    args = parser.parse_args()

    if os.environ.get("CUDA_VISIBLE_DEVICES") != "0":
        raise RuntimeError("CUDA_VISIBLE_DEVICES must be exactly 0")
    if torch.cuda.device_count() != 1 or torch.cuda.current_device() != 0:
        raise RuntimeError("P-GDN3-036 requires exactly CUDA index 0")
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
    dplr_root = fla_root / "fla" / "ops" / "generalized_delta_rule" / "dplr"
    dplr_module = importlib.import_module(
        "fla.ops.generalized_delta_rule.dplr.chunk"
    )
    if getattr(dplr_module, "chunk_dplr_delta_rule") is not chunk_dplr_delta_rule:
        raise RuntimeError("Exported DPLR chunk identity changed")
    dplr_source = Path(dplr_module.__file__).resolve()
    if dplr_source != dplr_root / "chunk.py":
        raise RuntimeError(f"Unexpected DPLR source: {dplr_source}")
    dplr_tree_sha256, dplr_source_hashes = python_tree_hash(dplr_root)
    if dplr_tree_sha256 != EXPECTED_DPLR_TREE_SHA256:
        raise RuntimeError(f"Official DPLR tree changed: {dplr_tree_sha256}")
    carrier_sources = {
        "gdn2": Path(inspect.getfile(GatedDeltaNet2)).resolve(),
        "short_convolution": Path(inspect.getfile(ShortConvolution)).resolve(),
    }
    if any(fla_root not in source.parents for source in carrier_sources.values()):
        raise RuntimeError(f"Carrier escaped pinned FLA: {carrier_sources}")

    zoology_root = Path(os.environ["ZOOLOGY_ROOT"]).resolve()
    zoology_sha = git_head(zoology_root)
    if zoology_sha != EXPECTED_ZOOLOGY_SHA:
        raise RuntimeError(f"Unexpected Zoology SHA: {zoology_sha}")

    candidate_config = build_config(
        arm="future_seed_anchored_dual_key_gdn2",
        sequence_length=1024,
        num_kv_pairs=4,
        max_epochs=10,
        batch_size=32,
    )
    direct_config = build_config(
        arm="future_seed_decoupled_key_gdn2",
        sequence_length=1024,
        num_kv_pairs=4,
        max_epochs=10,
        batch_size=32,
    )
    native_config = build_config(
        arm="future_seed_gdn2",
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

    set_determinism(123)
    candidate = make_model(candidate_config, "future_seed_anchored_dual_key_gdn2")
    set_determinism(123)
    direct_parent = make_model(direct_config, "future_seed_decoupled_key_gdn2")
    set_determinism(123)
    native = make_model(native_config, "future_seed_gdn2")
    candidate_parameters = sum(p.numel() for p in candidate.parameters())
    direct_parameters = sum(p.numel() for p in direct_parent.parameters())
    native_parameters = sum(p.numel() for p in native.parameters())
    if candidate_parameters != EXPECTED_PARAMETERS:
        raise RuntimeError(f"Candidate parameter count changed: {candidate_parameters}")
    if native_parameters != EXPECTED_NATIVE_PARAMETERS:
        raise RuntimeError(f"Native parameter count changed: {native_parameters}")
    if candidate_parameters != direct_parameters:
        raise RuntimeError("Anchored and direct dual-key parameter counts differ")
    if candidate_parameters - native_parameters != EXPECTED_PARAMETER_DELTA:
        raise RuntimeError("Anchored dual-key parameter delta changed")
    native_hash = parameter_hash(native)
    if parent_parameter_hash(candidate) != native_hash:
        raise RuntimeError("Candidate parent initialization differs from native GDN2")
    if parent_parameter_hash(direct_parent) != native_hash:
        raise RuntimeError("Direct DPLR reference parent initialization drifted")

    mixers: list[ZoologyAnchoredDualKeyGDN2FutureSeedMixer] = []
    tie_rows = []
    provenance = []
    for layer_index, block in enumerate(candidate.backbone.layers):
        mixer = block.sequence_mixer
        if type(mixer) is not ZoologyAnchoredDualKeyGDN2FutureSeedMixer:
            raise RuntimeError(f"Layer {layer_index} mixer changed: {type(mixer)}")
        layer = mixer.layer
        if type(layer) is not AnchoredDualKeyGDN2:
            raise RuntimeError(f"Layer {layer_index} wrapper changed: {type(layer)}")
        if type(layer.base) is not GatedDeltaNet2:
            raise RuntimeError(f"Layer {layer_index} carrier changed")
        if sum(p.numel() for p in mixer.parameters()) != EXPECTED_MIXER_PARAMETERS:
            raise RuntimeError(f"Layer {layer_index} mixer parameter count changed")
        if mixer.state_size() != EXPECTED_STATE_VALUES:
            raise RuntimeError(f"Layer {layer_index} state size changed")
        projection_error = _module_max_error(
            layer.k_erase_proj,
            layer.base.k_proj,
        )
        convolution_error = _module_max_error(
            layer.k_erase_conv1d,
            layer.base.k_conv1d,
        )
        if max(projection_error, convolution_error, layer.initial_tie_max_error) != 0.0:
            raise RuntimeError(f"Layer {layer_index} key tie is not exact")
        convolutions = {
            name: module.backend
            for name, module in layer.named_modules()
            if isinstance(module, ShortConvolution)
        }
        if len(convolutions) != 4 or set(convolutions.values()) != {"triton"}:
            raise RuntimeError(f"Layer {layer_index} convolution fallback: {convolutions}")
        tie_rows.append(
            {
                "layer": layer_index,
                "projection_max_error": projection_error,
                "convolution_max_error": convolution_error,
                "recorded_initial_tie_max_error": layer.initial_tie_max_error,
            }
        )
        provenance.append(
            {
                "layer": layer_index,
                "carrier": type(layer.base).__qualname__,
                "operator": "ChunkDPLRDeltaRuleFunctionBackward",
                "state_values": mixer.state_size(),
                "convolution_backends": convolutions,
            }
        )
        mixers.append(mixer)

    candidate = candidate.cuda().eval()
    direct_parent = direct_parent.cuda().eval()
    native = native.cuda().eval()
    inputs, targets, _slices = next(iter(test_loader))
    inputs, targets = inputs[:2].cuda(), targets[:2].cuda()
    with torch.no_grad():
        candidate_logits = candidate(inputs)
        direct_logits = direct_parent(inputs)
        native_logits = native(inputs)
    exact_dplr_parent_max_error = max_abs_difference(
        candidate_logits,
        direct_logits,
    )
    if exact_dplr_parent_max_error != 0.0:
        raise RuntimeError(
            "Anchored zero-tangent model lost exact DPLR parent identity: "
            f"{exact_dplr_parent_max_error}"
        )
    native_function_relative_rms = relative_rms(candidate_logits, native_logits)
    if native_function_relative_rms > 0.10:
        raise RuntimeError(
            f"Tied DPLR recurrence lost native function parity: {native_function_relative_rms}"
        )

    candidate.train().zero_grad(set_to_none=True)
    model_module = importlib.import_module(
        "experiments.zoology_mqar.decoupled_key_futureseed"
    )
    original_chunk = model_module.chunk_dplr_delta_rule
    official_calls: list[dict[str, torch.Tensor]] = []

    def recording_chunk(*call_args, **call_kwargs):
        if call_args:
            raise RuntimeError("Anchored dual-key GDN2 must call DPLR by keyword")
        official_calls.append(
            {
                name: call_kwargs[name].detach()
                for name in ("q", "k", "v", "a", "b", "gk")
            }
        )
        return original_chunk(**call_kwargs)

    for mixer in mixers:
        mixer.layer.set_capture(True)
    model_module.chunk_dplr_delta_rule = recording_chunk
    try:
        logits = candidate(inputs)
    finally:
        model_module.chunk_dplr_delta_rule = original_chunk
        for mixer in mixers:
            mixer.layer.set_capture(False)
    mask = targets != -100
    loss = F.cross_entropy(logits[mask], targets[mask])
    graph_names = backward_names(loss)
    official_backward_count = graph_names.count(
        "ChunkDPLRDeltaRuleFunctionBackward"
    )
    if official_backward_count != MODEL_LAYERS:
        raise RuntimeError(f"Expected two DPLR backward paths, got {official_backward_count}")
    loss.backward()

    gradient_rows = []
    production_rows = []
    for layer_index, mixer in enumerate(mixers):
        layer = mixer.layer
        row = {"layer": layer_index}
        for name in ("q", "k", "v", "f", "b", "w"):
            row[f"{name}_gradient_rms"] = finite_module_gradient_rms(
                getattr(layer.base, f"{name}_proj"),
                f"layer{layer_index}.{name}_proj",
            )
        row["erase_key_projection_gradient_rms"] = finite_module_gradient_rms(
            layer.k_erase_proj,
            f"layer{layer_index}.k_erase_proj",
        )
        row["erase_key_convolution_gradient_rms"] = finite_module_gradient_rms(
            layer.k_erase_conv1d,
            f"layer{layer_index}.k_erase_conv1d",
        )
        write_gradient = layer.base.k_proj.weight.grad.float()
        erase_gradient = layer.k_erase_proj.weight.grad.float()
        row["write_erase_gradient_relative_rms"] = relative_rms(
            erase_gradient,
            write_gradient,
        )
        if row["write_erase_gradient_relative_rms"] <= 1e-4:
            raise RuntimeError("Anchored key branches receive indistinguishable gradients")
        if layer_index == 1:
            row["future_seed_gradient_rms"] = finite_gradient_rms(
                mixer.future_seed_logit,
                "layer1.future_seed_logit",
            )
        gradient_rows.append(row)

        captured = layer._captured
        call = official_calls[layer_index]
        expected_key, tangent, correction = layer.anchored_key(
            captured["k_write"],
            captured["k_erase_raw"],
        )
        alpha, beta = layer.transition_factors(
            captured["k_erase"],
            captured["g"],
            captured["erase_gate"],
        )
        mapping_errors = {
            "anchored_key": max_abs_difference(
                captured["k_erase"], expected_key
            ),
            "tangent": max_abs_difference(captured["erase_tangent"], tangent),
            "correction": max_abs_difference(
                captured["erase_correction"], correction
            ),
            "write_key": max_abs_difference(call["k"], captured["k_write"]),
            "write_value": max_abs_difference(call["v"], captured["write"]),
            "erase_alpha": max_abs_difference(call["a"], alpha),
            "erase_beta": max_abs_difference(call["b"], beta),
            "decay": max_abs_difference(call["gk"], captured["g"]),
        }
        if max(mapping_errors.values()) != 0.0:
            raise RuntimeError(f"Layer {layer_index} call mapping changed: {mapping_errors}")
        cosine = F.cosine_similarity(
            captured["k_write"].float(),
            captured["k_erase"].float(),
            dim=-1,
        )
        if float(cosine.min().item()) < 0.89:
            raise RuntimeError(f"Layer {layer_index} ownership floor failed")
        tangent_error = float(
            (
                captured["erase_tangent"].float()
                * captured["k_write"].float()
            ).sum(dim=-1).abs().max().item()
        )
        if tangent_error > 5e-3:
            raise RuntimeError(f"Layer {layer_index} tangent projection failed")
        stability = transition_diagnostics(call["a"], call["b"], call["gk"])
        if stability["spectral_norm_max"] > 1.25:
            raise RuntimeError(f"Layer {layer_index} transition is unbounded: {stability}")
        production_rows.append(
            {
                "layer": layer_index,
                "mapping_max_errors": mapping_errors,
                "key_cosine_min": float(cosine.min().item()),
                "key_cosine_mean": float(cosine.mean().item()),
                "tangent_orthogonality_max_error": tangent_error,
                "correction_relative_rms": _rms_ratio(
                    captured["erase_correction"],
                    captured["k_write"],
                ),
                "transition": stability,
                "terminal_state_rms": float(
                    captured["terminal_state"].float().square().mean().sqrt().item()
                ),
            }
        )

    generator = torch.Generator(device="cuda").manual_seed(53036)
    shape = (2, 64, MODEL_HEADS, HEAD_DIM)
    q = F.normalize(torch.randn(shape, generator=generator, device="cuda"), dim=-1)
    key = F.normalize(torch.randn(shape, generator=generator, device="cuda"), dim=-1)
    raw_erase = F.normalize(
        torch.randn(shape, generator=generator, device="cuda"),
        dim=-1,
    )
    anchored_key, tangent, correction = AnchoredDualKeyGDN2.anchored_key(
        key,
        raw_erase,
    )
    tied_key, tied_tangent, tied_correction = AnchoredDualKeyGDN2.anchored_key(
        key,
        key,
    )
    structural = {
        "residual_cap": RESIDUAL_CAP,
        "registered_cosine_floor": COSINE_FLOOR,
        "opened_cosine_min": float(
            F.cosine_similarity(key, anchored_key, dim=-1).min().item()
        ),
        "tangent_orthogonality_max_error": float(
            (tangent * key).sum(dim=-1).abs().max().item()
        ),
        "tied_key_max_error": max_abs_difference(tied_key, key),
        "tied_tangent_max_abs": float(tied_tangent.abs().max().item()),
        "tied_correction_max_abs": float(tied_correction.abs().max().item()),
    }
    if structural["opened_cosine_min"] < COSINE_FLOOR - 1e-5:
        raise RuntimeError(f"Synthetic cosine floor failed: {structural}")
    if structural["tangent_orthogonality_max_error"] > 1e-5:
        raise RuntimeError(f"Synthetic tangent projection failed: {structural}")
    if max(
        structural["tied_key_max_error"],
        structural["tied_tangent_max_abs"],
        structural["tied_correction_max_abs"],
    ) != 0.0:
        raise RuntimeError(f"Synthetic tied identity failed: {structural}")

    value = 0.1 * torch.randn(shape, generator=generator, device="cuda")
    erase_gate = torch.sigmoid(torch.randn(shape, generator=generator, device="cuda"))
    write_gate = torch.sigmoid(torch.randn(shape, generator=generator, device="cuda"))
    decay = 0.94 + 0.05 * torch.sigmoid(
        torch.randn(shape, generator=generator, device="cuda")
    )
    g = decay.log()
    q, key, anchored_key, value, erase_gate, write_gate, g = (
        tensor.to(torch.bfloat16)
        for tensor in (q, key, anchored_key, value, erase_gate, write_gate, g)
    )
    initial_state = 0.1 * torch.randn(
        2,
        MODEL_HEADS,
        HEAD_DIM,
        HEAD_DIM,
        generator=generator,
        device="cuda",
        dtype=torch.float32,
    )
    tied_alpha, tied_beta = DirectDecoupledKeyGDN2.transition_factors(
        key,
        g,
        erase_gate,
    )
    write = write_gate * value
    tied_output, tied_state = run_official(
        q=q,
        k=key,
        v=write,
        a=tied_alpha,
        b=tied_beta,
        gk=g,
        initial_state=initial_state,
    )
    gdn2_output, gdn2_state = _run_gdn2(
        q=q,
        k=key,
        v=value,
        g=g,
        erase_gate=erase_gate,
        write_gate=write_gate,
        initial_state=initial_state,
    )
    explicit_output, explicit_state = _explicit_gdn2(
        q=q,
        k=key,
        v=value,
        g=g,
        erase_gate=erase_gate,
        write_gate=write_gate,
        initial_state=initial_state,
    )
    tied_recurrence_parity = {
        "dplr_vs_gdn2_output_relative_rms": relative_rms(
            tied_output, gdn2_output
        ),
        "dplr_vs_gdn2_state_relative_rms": relative_rms(tied_state, gdn2_state),
        "dplr_vs_explicit_output_relative_rms": relative_rms(
            tied_output, explicit_output
        ),
        "dplr_vs_explicit_state_relative_rms": relative_rms(
            tied_state, explicit_state
        ),
    }
    if max(tied_recurrence_parity.values()) > 0.03:
        raise RuntimeError(f"Tied recurrence parity failed: {tied_recurrence_parity}")

    opened_alpha, opened_beta = DirectDecoupledKeyGDN2.transition_factors(
        anchored_key,
        g,
        erase_gate,
    )
    opened_output, opened_state = run_official(
        q=q,
        k=key,
        v=write,
        a=opened_alpha,
        b=opened_beta,
        gk=g,
        initial_state=initial_state,
    )
    erase_key_dependency = {
        "output_relative_rms": relative_rms(opened_output, tied_output),
        "state_relative_rms": relative_rms(opened_state, tied_state),
    }
    if min(erase_key_dependency.values()) <= 1e-4:
        raise RuntimeError(f"Anchored erase key is inactive: {erase_key_dependency}")

    permutation = torch.tensor([2, 0, 3, 1], device="cuda")
    permuted_output, permuted_state = run_official(
        q=q[:, :, permutation],
        k=key[:, :, permutation],
        v=write[:, :, permutation],
        a=opened_alpha[:, :, permutation],
        b=opened_beta[:, :, permutation],
        gk=g[:, :, permutation],
        initial_state=initial_state[:, permutation],
    )
    permutation_errors = {
        "output_max_error": max_abs_difference(
            opened_output[:, :, permutation],
            permuted_output,
        ),
        "state_max_error": max_abs_difference(
            opened_state[:, permutation],
            permuted_state,
        ),
    }
    if max(permutation_errors.values()) > 2e-3:
        raise RuntimeError(f"Head equivariance failed: {permutation_errors}")

    result = {
        "status": "passed",
        "gpu": {"name": device.name, "uuid": device_uuid, "count": 1},
        "pinned_fla_sha": PINNED_FLA_SHA,
        "official_dplr_root": str(dplr_root),
        "official_dplr_tree_sha256": dplr_tree_sha256,
        "official_dplr_source_sha256": dplr_source_hashes,
        "official_chunk_source": str(dplr_source),
        "official_chunk_sha256": _sha256(dplr_source),
        "official_carrier_sources": {
            name: str(source) for name, source in carrier_sources.items()
        },
        "zoology_sha": zoology_sha,
        "data_hashes": data_hashes,
        "candidate_parameter_count": candidate_parameters,
        "native_parameter_count": native_parameters,
        "parameter_delta": candidate_parameters - native_parameters,
        "recurrent_state_values_per_layer": EXPECTED_STATE_VALUES,
        "persistent_state_delta": 0,
        "logical_scans_per_layer": 1,
        "parent_parameter_hash": parent_parameter_hash(candidate.cpu()),
        "native_parameter_hash": parameter_hash(native.cpu()),
        "initial_tie": tie_rows,
        "exact_dplr_parent_output_max_error": exact_dplr_parent_max_error,
        "native_function_output_relative_rms": native_function_relative_rms,
        "official_module_provenance": provenance,
        "official_chunk_backward_count": official_backward_count,
        "gradient_rows": gradient_rows,
        "production_rows": production_rows,
        "structural_geometry": structural,
        "tied_recurrence_parity": tied_recurrence_parity,
        "erase_key_dependency": erase_key_dependency,
        "head_permutation": permutation_errors,
        "fallback": False,
    }
    if result["parent_parameter_hash"] != result["native_parameter_hash"]:
        raise RuntimeError("Parent hash changed while serializing contract")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
