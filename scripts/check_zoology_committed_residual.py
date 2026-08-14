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

from experiments.zoology_mqar.committed_residual_futureseed import (
    CommittedResidualGDN2,
    ZoologyCommittedResidualGDN2FutureSeedMixer,
    parent_parameter_hash,
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
    explicit_dplr_recurrence,
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
)


EXPECTED_PARAMETERS = 661_584
EXPECTED_STATE_VALUES = 4_096
MODEL_LAYERS = 2
MODEL_HEADS = 4
HEAD_DIM = 32


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _explicit_call_errors(
    call: dict[str, torch.Tensor | None],
) -> dict[str, float]:
    q = call["q"]
    k = call["k"]
    v = call["v"]
    a = call["a"]
    b = call["b"]
    gk = call["gk"]
    official_output = call["output"]
    official_state = call["state"]
    if not all(
        isinstance(x, torch.Tensor)
        for x in (q, k, v, a, b, gk, official_output, official_state)
    ):
        raise RuntimeError("Incomplete official DPLR call record")
    initial = call["initial_state"]
    if initial is None:
        initial = torch.zeros(
            q.shape[0],
            q.shape[2],
            q.shape[3],
            v.shape[3],
            device=q.device,
            dtype=q.dtype,
        )
    if not isinstance(initial, torch.Tensor):
        raise RuntimeError("Initial state changed type")
    explicit_output, explicit_state = explicit_dplr_recurrence(
        q=q,
        k=k,
        v=v,
        a=a,
        b=b,
        gk=gk,
        initial_state=initial,
    )
    return {
        "output_relative_rms": relative_rms(official_output, explicit_output),
        "state_relative_rms": relative_rms(official_state, explicit_state),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-gpu-name", required=True)
    parser.add_argument("--expected-gpu-uuid", required=True)
    args = parser.parse_args()

    if os.environ.get("CUDA_VISIBLE_DEVICES") != "0":
        raise RuntimeError("CUDA_VISIBLE_DEVICES must be exactly 0")
    if torch.cuda.device_count() != 1 or torch.cuda.current_device() != 0:
        raise RuntimeError("P-GDN3-037 requires exactly CUDA index 0")
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
    dplr_module = importlib.import_module("fla.ops.generalized_delta_rule.dplr.chunk")
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
    if git_head(zoology_root) != EXPECTED_ZOOLOGY_SHA:
        raise RuntimeError("Unexpected Zoology source SHA")

    candidate_config = build_config(
        arm="future_seed_committed_residual_gdn2",
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
    candidate = make_model(candidate_config, "future_seed_committed_residual_gdn2")
    set_determinism(123)
    native = make_model(native_config, "future_seed_gdn2")
    candidate_parameters = sum(p.numel() for p in candidate.parameters())
    native_parameters = sum(p.numel() for p in native.parameters())
    if candidate_parameters != EXPECTED_PARAMETERS or native_parameters != EXPECTED_PARAMETERS:
        raise RuntimeError(
            f"Parameter count changed: candidate={candidate_parameters}, native={native_parameters}"
        )
    native_hash = parameter_hash(native)
    if parent_parameter_hash(candidate) != native_hash:
        raise RuntimeError("Candidate initialization is not byte-identical to native GDN2")

    mixers: list[ZoologyCommittedResidualGDN2FutureSeedMixer] = []
    provenance = []
    for layer_index, block in enumerate(candidate.backbone.layers):
        mixer = block.sequence_mixer
        if type(mixer) is not ZoologyCommittedResidualGDN2FutureSeedMixer:
            raise RuntimeError(f"Layer {layer_index} mixer changed: {type(mixer)}")
        layer = mixer.layer
        if type(layer) is not CommittedResidualGDN2:
            raise RuntimeError(f"Layer {layer_index} wrapper changed: {type(layer)}")
        if type(layer.base) is not GatedDeltaNet2:
            raise RuntimeError(f"Layer {layer_index} carrier changed")
        if mixer.state_size() != EXPECTED_STATE_VALUES:
            raise RuntimeError(f"Layer {layer_index} state size changed")
        convolutions = {
            name: module.backend
            for name, module in layer.named_modules()
            if isinstance(module, ShortConvolution)
        }
        if len(convolutions) != 3 or set(convolutions.values()) != {"triton"}:
            raise RuntimeError(f"Layer {layer_index} convolution fallback: {convolutions}")
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

    candidate = candidate.cuda().train()
    inputs, targets, _slices = next(iter(test_loader))
    inputs, targets = inputs[:2].cuda(), targets[:2].cuda()
    model_module = importlib.import_module(
        "experiments.zoology_mqar.committed_residual_futureseed"
    )
    original_chunk = model_module.chunk_dplr_delta_rule
    official_calls: list[dict[str, torch.Tensor | None]] = []

    def recording_chunk(*call_args, **call_kwargs):
        if call_args:
            raise RuntimeError("Committed-residual GDN2 must call DPLR by keyword")
        result = original_chunk(**call_kwargs)
        output, state = result
        official_calls.append(
            {
                **{
                    name: call_kwargs[name].detach()
                    for name in ("q", "k", "v", "a", "b", "gk")
                },
                "initial_state": (
                    None
                    if call_kwargs["initial_state"] is None
                    else call_kwargs["initial_state"].detach()
                ),
                "output": output.detach(),
                "state": state.detach(),
            }
        )
        return result

    for mixer in mixers:
        mixer.layer.set_capture(True)
    model_module.chunk_dplr_delta_rule = recording_chunk
    try:
        logits = candidate(inputs)
    finally:
        model_module.chunk_dplr_delta_rule = original_chunk
        for mixer in mixers:
            mixer.layer.set_capture(False)
    if len(official_calls) != MODEL_LAYERS:
        raise RuntimeError(
            f"Expected {MODEL_LAYERS} official DPLR calls, got {len(official_calls)}"
        )
    mask = targets != -100
    loss = F.cross_entropy(logits[mask], targets[mask])
    graph_names = backward_names(loss)
    official_backward_count = graph_names.count("ChunkDPLRDeltaRuleFunctionBackward")
    if official_backward_count != MODEL_LAYERS:
        raise RuntimeError(f"Expected two DPLR backward paths, got {official_backward_count}")
    loss.backward()

    gradient_rows = []
    production_rows = []
    for layer_index, (mixer, call) in enumerate(zip(mixers, official_calls)):
        layer = mixer.layer
        captured = layer._captured
        expected_beta, expected_alpha, expected_transition_beta, expected_target = (
            layer.transition_inputs(
                captured["k"],  # type: ignore[arg-type]
                captured["g"],  # type: ignore[arg-type]
                captured["erase_gate"],  # type: ignore[arg-type]
                captured["write_target"],  # type: ignore[arg-type]
            )
        )
        mapping_errors = {
            "query": max_abs_difference(call["q"], captured["q"]),  # type: ignore[arg-type]
            "coherent_key": max_abs_difference(call["k"], captured["k"]),  # type: ignore[arg-type]
            "alpha": max_abs_difference(call["a"], expected_alpha),  # type: ignore[arg-type]
            "transition_beta": max_abs_difference(
                call["b"], expected_transition_beta  # type: ignore[arg-type]
            ),
            "committed_target": max_abs_difference(call["v"], expected_target),  # type: ignore[arg-type]
            "decay": max_abs_difference(call["gk"], captured["g"]),  # type: ignore[arg-type]
            "beta_capture": max_abs_difference(captured["beta"], expected_beta),  # type: ignore[arg-type]
        }
        if max(mapping_errors.values()) != 0.0:
            raise RuntimeError(f"Layer {layer_index} call mapping changed: {mapping_errors}")
        transition = transition_diagnostics(call["a"], call["b"], call["gk"])  # type: ignore[arg-type]
        if transition["spectral_norm_max"] > 1.05:
            raise RuntimeError(f"Layer {layer_index} transition is not bounded: {transition}")
        recurrence_errors = _explicit_call_errors(call)
        if max(recurrence_errors.values()) > 0.05:
            raise RuntimeError(
                f"Layer {layer_index} explicit recurrence mismatch: {recurrence_errors}"
            )
        beta = captured["beta"]
        terminal = captured["terminal_state"]
        if not isinstance(beta, torch.Tensor) or not isinstance(terminal, torch.Tensor):
            raise RuntimeError("Production capture changed tensor type")
        if beta.min() < 0.0 or beta.max() > 1.0 or beta.std() <= 1e-4:
            raise RuntimeError(f"Layer {layer_index} beta is inactive or unbounded")
        if not torch.isfinite(terminal).all():
            raise RuntimeError(f"Layer {layer_index} terminal state is non-finite")

        gradient_row = {"layer": layer_index}
        for name in ("q", "k", "v", "f", "b", "w"):
            gradient_row[f"{name}_gradient_rms"] = finite_module_gradient_rms(
                getattr(layer.base, f"{name}_proj"),
                f"layer{layer_index}.{name}_proj",
            )
        if layer_index == 1:
            gradient_row["future_seed_gradient_rms"] = finite_gradient_rms(
                mixer.future_seed_logit,
                "layer1.future_seed_logit",
            )
        gradient_rows.append(gradient_row)
        production_rows.append(
            {
                "layer": layer_index,
                "mapping_errors": mapping_errors,
                "recurrence_errors": recurrence_errors,
                "transition": transition,
                "beta_mean": float(beta.float().mean().item()),
                "beta_std": float(beta.float().std(unbiased=False).item()),
                "terminal_state_rms": float(
                    terminal.float().square().mean().sqrt().item()
                ),
            }
        )

    torch.manual_seed(37037)
    batch, length, heads, key_dim, value_dim = 2, 64, 4, 32, 32
    q = F.normalize(torch.randn(batch, length, heads, key_dim, device="cuda"), dim=-1).bfloat16()
    k = F.normalize(torch.randn(batch, length, heads, key_dim, device="cuda"), dim=-1).bfloat16()
    g = (-0.01 - 0.04 * torch.rand(batch, length, heads, key_dim, device="cuda")).bfloat16()
    erase_gate = torch.rand(batch, length, heads, key_dim, device="cuda").bfloat16()
    write_target = torch.randn(batch, length, heads, value_dim, device="cuda").bfloat16()
    beta, alpha, transition_beta, committed_target = CommittedResidualGDN2.transition_inputs(
        k, g, erase_gate, write_target
    )
    initial_state = (0.05 * torch.randn(batch, heads, key_dim, value_dim, device="cuda")).bfloat16()
    official_output, official_state = run_official(
        q=q,
        k=k,
        v=committed_target,
        a=alpha,
        b=transition_beta,
        gk=g,
        initial_state=initial_state,
    )
    explicit_output, explicit_state = explicit_dplr_recurrence(
        q=q,
        k=k,
        v=committed_target,
        a=alpha,
        b=transition_beta,
        gk=g,
        initial_state=initial_state,
    )
    synthetic_errors = {
        "output_relative_rms": relative_rms(official_output, explicit_output),
        "state_relative_rms": relative_rms(official_state, explicit_state),
    }
    if max(synthetic_errors.values()) > 0.03:
        raise RuntimeError(f"Synthetic nonzero-state recurrence mismatch: {synthetic_errors}")

    result = {
        "status": "pass",
        "plan": "P-GDN3-037",
        "gpu": {
            "name": device.name,
            "uuid": device_uuid,
            "visible_devices": os.environ["CUDA_VISIBLE_DEVICES"],
        },
        "source": {
            "fla_sha": PINNED_FLA_SHA,
            "fla_root": str(fla_root),
            "dplr_source": str(dplr_source),
            "dplr_tree_sha256": dplr_tree_sha256,
            "dplr_source_hashes": dplr_source_hashes,
            "zoology_sha": git_head(zoology_root),
            "candidate_source": str(Path(inspect.getfile(CommittedResidualGDN2)).resolve()),
            "candidate_source_sha256": _sha256(
                Path(inspect.getfile(CommittedResidualGDN2)).resolve()
            ),
        },
        "data_hashes": data_hashes,
        "parameters": {
            "native": native_parameters,
            "candidate": candidate_parameters,
            "delta": candidate_parameters - native_parameters,
            "native_hash": native_hash,
            "candidate_parent_hash": parent_parameter_hash(candidate),
        },
        "state_values_per_layer": EXPECTED_STATE_VALUES,
        "logical_scans_per_layer": 1,
        "official_backward_count": official_backward_count,
        "provenance": provenance,
        "gradients": gradient_rows,
        "production": production_rows,
        "synthetic_nonzero_state": {
            "errors": synthetic_errors,
            "beta_mean": float(beta.float().mean().item()),
            "beta_std": float(beta.float().std(unbiased=False).item()),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
