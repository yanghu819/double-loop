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
from fla.ops.generalized_delta_rule.dplr import chunk_dplr_delta_rule
from zoology.data.utils import prepare_data
from zoology.utils import set_determinism

from experiments.zoology_mqar.contractive_dplr_futureseed import (
    ZoologyContractiveDPLRFutureSeedMixer,
)
from experiments.zoology_mqar.gdn2_futureseed import PINNED_FLA_SHA
from experiments.zoology_mqar.length_scaling import (
    build_config,
    dataset_hash,
    make_model,
)


EXPECTED_ZOOLOGY_SHA = "1ad20d193b6113cae1e8f3c655c300d7b4b3f4bb"
EXPECTED_TRAIN_HASH = "647c64ece84984a23dfd817c4f277ea83840dbec57cc18bb6c9bf9eda7cc9a68"
EXPECTED_TEST_HASH = "4a8237ba8fe19aaff0d1d72de7b7f6505eaab59cd091442c2f463df34cce278f"
EXPECTED_DPLR_TREE_SHA256 = (
    "686ca42b82ba098f9187fcacfe8ab689b9038938b1edae2809a21b6f03ec8643"
)
EXPECTED_FLA_ROOT = Path(
    "/huyang2/double-loop/.cache/fla-versions/"
    "9c8e42e762fce087c27b673af4922795d9edb85e-0280db310981915e"
)
EXPECTED_PARAMETERS = 662_608
EXPECTED_MIXER_PARAMETERS = 117_032
EXPECTED_STATE_VALUES = 4_096
MODEL_LAYERS = 2
MODEL_HEADS = 4
HEAD_DIM = 32


def normalized_uuid(value: str) -> str:
    return value.removeprefix("GPU-").lower()


def git_head(path: Path) -> str:
    return subprocess.check_output(
        ["git", "-C", str(path), "rev-parse", "HEAD"], text=True
    ).strip()


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def python_tree_hash(path: Path) -> tuple[str, dict[str, str]]:
    digest = hashlib.sha256()
    source_hashes: dict[str, str] = {}
    sources = sorted(path.rglob("*.py"))
    if not sources:
        raise RuntimeError(f"No Python sources found under {path}")
    for source in sources:
        relative = str(source.relative_to(path))
        content = source.read_bytes()
        digest.update(relative.encode())
        digest.update(b"\0")
        digest.update(content)
        source_hashes[relative] = hashlib.sha256(content).hexdigest()
    return digest.hexdigest(), source_hashes


def relative_rms(left: torch.Tensor, right: torch.Tensor) -> float:
    numerator = (left.float() - right.float()).square().mean().sqrt()
    denominator = right.float().square().mean().sqrt().clamp_min(1e-8)
    return float((numerator / denominator).item())


def max_abs_difference(left: torch.Tensor, right: torch.Tensor) -> float:
    difference = (left.float() - right.float()).abs()
    if not torch.isfinite(difference).all():
        raise RuntimeError("Non-finite tensor difference")
    return float(difference.amax().item())


def backward_names(tensor: torch.Tensor) -> list[str]:
    names: list[str] = []
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


def finite_gradient_rms(parameter: torch.nn.Parameter, label: str) -> float:
    gradient = parameter.grad
    if gradient is None or not torch.isfinite(gradient).all():
        raise RuntimeError(f"{label} has no finite gradient")
    rms = float(gradient.float().square().mean().sqrt().item())
    if rms <= 0.0:
        raise RuntimeError(f"{label} gradient is inactive: {rms}")
    return rms


def finite_module_gradient_rms(module: torch.nn.Module, label: str) -> float:
    squared_sum = 0.0
    element_count = 0
    parameter_count = 0
    for name, parameter in module.named_parameters():
        if not parameter.requires_grad:
            continue
        gradient = parameter.grad
        if gradient is None or not torch.isfinite(gradient).all():
            raise RuntimeError(f"{label}.{name} has no finite gradient")
        parameter_squared_sum = float(gradient.float().square().sum().item())
        parameter_rms = (parameter_squared_sum / gradient.numel()) ** 0.5
        if parameter_rms <= 0.0:
            raise RuntimeError(
                f"{label}.{name} gradient is inactive: {parameter_rms}"
            )
        squared_sum += parameter_squared_sum
        element_count += gradient.numel()
        parameter_count += 1
    if parameter_count == 0:
        raise RuntimeError(f"{label} has no trainable parameters")
    rms = (squared_sum / element_count) ** 0.5
    if rms <= 0.0:
        raise RuntimeError(f"{label} gradient is inactive: {rms}")
    return rms


def transition_diagnostics(
    a: torch.Tensor,
    b: torch.Tensor,
    gk: torch.Tensor,
    *,
    max_samples: int = 128,
) -> dict[str, float | int]:
    a_flat = a.float().flatten(0, -2)
    b_flat = b.float().flatten(0, -2)
    g_flat = gk.float().flatten(0, -2)
    sample_count = min(max_samples, a_flat.shape[0])
    indices = torch.linspace(
        0,
        a_flat.shape[0] - 1,
        steps=sample_count,
        device=a.device,
    ).round().long()
    a_flat = a_flat.index_select(0, indices)
    b_flat = b_flat.index_select(0, indices)
    g_flat = g_flat.index_select(0, indices)
    transition = (
        g_flat.exp().diag_embed()
        + b_flat.unsqueeze(-1) * a_flat.unsqueeze(-2)
    )
    symmetry_error = float(
        (transition - transition.transpose(-1, -2)).abs().amax().item()
    )
    symmetric = 0.5 * (transition + transition.transpose(-1, -2))
    spectral_norm = float(torch.linalg.svdvals(transition).amax().item())
    eigenvalue_min = float(torch.linalg.eigvalsh(symmetric).amin().item())
    return {
        "samples": sample_count,
        "symmetry_error_max": symmetry_error,
        "spectral_norm_max": spectral_norm,
        "symmetric_eigenvalue_min": eigenvalue_min,
    }


def explicit_dplr_recurrence(
    *,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    gk: torch.Tensor,
    initial_state: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    qf, kf, vf, af, bf, gf = (
        tensor.float() for tensor in (q, k, v, a, b, gk)
    )
    state = initial_state.float().clone()
    outputs = []
    scale = q.shape[-1] ** -0.5
    for token in range(q.shape[1]):
        previous = state
        erase_read = torch.einsum(
            "bhk,bhkv->bhv", af[:, token], previous
        )
        state = (
            previous * gf[:, token].exp().unsqueeze(-1)
            + bf[:, token].unsqueeze(-1) * erase_read.unsqueeze(-2)
            + kf[:, token].unsqueeze(-1) * vf[:, token].unsqueeze(-2)
        )
        outputs.append(
            torch.einsum("bhk,bhkv->bhv", qf[:, token] * scale, state)
        )
    return torch.stack(outputs, dim=1), state


def run_official(
    *,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    gk: torch.Tensor,
    initial_state: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        output, state = chunk_dplr_delta_rule(
            q=q,
            k=k,
            v=v,
            a=a,
            b=b,
            gk=gk,
            initial_state=initial_state,
            output_final_state=True,
            safe_gate=False,
            chunk_size=16,
        )
    if state is None:
        raise RuntimeError("Official DPLR did not return a final state")
    for label, tensor in {"output": output, "state": state}.items():
        if not torch.isfinite(tensor).all():
            raise RuntimeError(f"Official DPLR {label} is non-finite")
    return output, state


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-gpu-name", required=True)
    parser.add_argument("--expected-gpu-uuid", required=True)
    args = parser.parse_args()

    if os.environ.get("CUDA_VISIBLE_DEVICES") != "0":
        raise RuntimeError("CUDA_VISIBLE_DEVICES must be exactly 0")
    if torch.cuda.device_count() != 1 or torch.cuda.current_device() != 0:
        raise RuntimeError("P-GDN3-030 requires exactly CUDA index 0")
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
    chunk_module = importlib.import_module(
        "fla.ops.generalized_delta_rule.dplr.chunk"
    )
    if getattr(chunk_module, "chunk_dplr_delta_rule") is not chunk_dplr_delta_rule:
        raise RuntimeError("Exported DPLR chunk function identity changed")
    chunk_source = Path(chunk_module.__file__).resolve()
    if chunk_source != dplr_root / "chunk.py":
        raise RuntimeError(f"Unexpected official DPLR chunk source: {chunk_source}")
    carrier_sources = {
        "gdn2": Path(inspect.getfile(GatedDeltaNet2)).resolve(),
        "short_convolution": Path(inspect.getfile(ShortConvolution)).resolve(),
    }
    if any(fla_root not in source.parents for source in carrier_sources.values()):
        raise RuntimeError(f"Projection carrier escaped pinned FLA: {carrier_sources}")
    dplr_tree_sha256, dplr_source_hashes = python_tree_hash(dplr_root)
    if dplr_tree_sha256 != EXPECTED_DPLR_TREE_SHA256:
        raise RuntimeError(f"Official DPLR tree changed: {dplr_tree_sha256}")
    if os.environ.get("FLA_DPLR_TREE_SHA256") != EXPECTED_DPLR_TREE_SHA256:
        raise RuntimeError("Registered DPLR tree hash changed")

    zoology_root = Path(os.environ["ZOOLOGY_ROOT"]).resolve()
    zoology_sha = git_head(zoology_root)
    if zoology_sha != EXPECTED_ZOOLOGY_SHA:
        raise RuntimeError(f"Unexpected Zoology SHA: {zoology_sha}")

    config = build_config(
        arm="future_seed_contractive_dplr",
        sequence_length=1024,
        num_kv_pairs=4,
        max_epochs=10,
        batch_size=32,
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
    model = make_model(config, "future_seed_contractive_dplr")
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    if parameter_count != EXPECTED_PARAMETERS:
        raise RuntimeError(f"Candidate parameter count changed: {parameter_count}")

    provenance = []
    mixers: list[ZoologyContractiveDPLRFutureSeedMixer] = []
    for layer_index, block in enumerate(model.backbone.layers):
        mixer = block.sequence_mixer
        if type(mixer) is not ZoologyContractiveDPLRFutureSeedMixer:
            raise RuntimeError(f"Layer {layer_index} mixer type changed: {type(mixer)}")
        if sum(parameter.numel() for parameter in mixer.parameters()) != EXPECTED_MIXER_PARAMETERS:
            raise RuntimeError(f"Layer {layer_index} mixer parameter count changed")
        if mixer.state_size() != EXPECTED_STATE_VALUES:
            raise RuntimeError(f"Layer {layer_index} state size changed")
        layer = mixer.layer
        base = layer.base
        if type(base) is not GatedDeltaNet2:
            raise RuntimeError(f"Layer {layer_index} projection carrier changed")
        if (
            base.mode != "chunk"
            or base.num_heads != MODEL_HEADS
            or base.num_v_heads != MODEL_HEADS
            or base.head_k_dim != HEAD_DIM
            or base.head_v_dim != HEAD_DIM
        ):
            raise RuntimeError(f"Layer {layer_index} state geometry drifted")
        base_projection_names = (
            "q_proj",
            "k_proj",
            "v_proj",
            "f_proj",
            "b_proj",
            "w_proj",
        )
        missing = [name for name in base_projection_names if not hasattr(base, name)]
        if not hasattr(layer, "beta_proj"):
            missing.append("beta_proj")
        if missing:
            raise RuntimeError(f"Layer {layer_index} projections missing: {missing}")
        convolutions = {
            name: module.backend
            for name, module in layer.named_modules()
            if isinstance(module, ShortConvolution)
        }
        if not convolutions or set(convolutions.values()) != {"triton"}:
            raise RuntimeError(
                f"Layer {layer_index} short-convolution fallback: {convolutions}"
            )
        provenance.append(
            {
                "layer": layer_index,
                "mixer_type": type(mixer).__qualname__,
                "layer_type": type(layer).__qualname__,
                "projection_carrier_type": type(base).__qualname__,
                "state_values": mixer.state_size(),
                "convolution_backends": convolutions,
            }
        )
        mixers.append(mixer)
    if len(mixers) != MODEL_LAYERS:
        raise RuntimeError(f"Expected two DPLR layers, got {len(mixers)}")

    model = model.cuda().train()
    inputs, targets, _slices = next(iter(test_loader))
    inputs, targets = inputs[:2].cuda(), targets[:2].cuda()
    model.zero_grad(set_to_none=True)
    model_module = importlib.import_module(
        "experiments.zoology_mqar.contractive_dplr_futureseed"
    )
    original_chunk = model_module.chunk_dplr_delta_rule
    official_call_records: list[dict[str, torch.Tensor]] = []
    production_components: list[dict[str, torch.Tensor]] = [
        {} for _layer in range(MODEL_LAYERS)
    ]
    hook_handles = []

    def capture_component(layer_index: int, name: str):
        def hook(_module, _inputs, output):
            tensor = output[0] if isinstance(output, tuple) else output
            production_components[layer_index][name] = tensor.detach()

        return hook

    for layer_index, mixer in enumerate(mixers):
        hook_handles.append(
            mixer.layer.base.v_conv1d.register_forward_hook(
                capture_component(layer_index, "value")
            )
        )
        hook_handles.append(
            mixer.layer.base.w_proj.register_forward_hook(
                capture_component(layer_index, "write_logits")
            )
        )

    def recording_chunk(*call_args, **call_kwargs):
        if call_args:
            raise RuntimeError("Contractive DPLR must call the official kernel by keyword")
        official_call_records.append(
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
        logits = model(inputs)
    finally:
        model_module.chunk_dplr_delta_rule = original_chunk
        for mixer in mixers:
            mixer.layer.set_capture(False)
        for handle in hook_handles:
            handle.remove()
    if not torch.isfinite(logits).all():
        raise RuntimeError("Full Contractive DPLR logits are non-finite")
    if len(official_call_records) != MODEL_LAYERS:
        raise RuntimeError(
            f"Expected two official DPLR calls, got {len(official_call_records)}"
        )
    mask = targets != -100
    loss = F.cross_entropy(logits[mask], targets[mask])
    graph_names = backward_names(loss)
    official_backward_count = graph_names.count(
        "ChunkDPLRDeltaRuleFunctionBackward"
    )
    if official_backward_count != MODEL_LAYERS:
        raise RuntimeError(
            f"Expected two official DPLR backward paths, got {official_backward_count}"
        )
    loss.backward()

    gradient_rows = []
    production_rows = []
    for layer_index, block in enumerate(model.backbone.layers):
        layer = block.sequence_mixer.layer
        row = {"layer": layer_index}
        for name in ("q", "k", "v", "f", "b", "w"):
            projection = getattr(layer.base, f"{name}_proj")
            row[f"{name}_gradient_rms"] = finite_module_gradient_rms(
                projection, f"layer{layer_index}.{name}_proj"
            )
        row["beta_gradient_rms"] = finite_gradient_rms(
            layer.beta_proj.weight, f"layer{layer_index}.beta_proj.weight"
        )
        if layer_index == 1:
            row["future_seed_gradient_rms"] = finite_gradient_rms(
                block.sequence_mixer.future_seed_logit,
                "layer1.future_seed_logit",
            )
        gradient_rows.append(row)

        captured = layer._captured
        required = {"p", "r", "a", "b", "g", "beta", "write", "terminal_state"}
        if set(captured) != required:
            raise RuntimeError(
                f"Layer {layer_index} production capture changed: {set(captured)}"
            )
        call = official_call_records[layer_index]
        components = production_components[layer_index]
        if set(components) != {"value", "write_logits"}:
            raise RuntimeError(
                f"Layer {layer_index} production components changed: {set(components)}"
            )
        value_component = components["value"].reshape(
            *components["value"].shape[:-1], MODEL_HEADS, HEAD_DIM
        )
        write_gate_component = components["write_logits"].sigmoid().reshape(
            *components["write_logits"].shape[:-1], MODEL_HEADS, HEAD_DIM
        )
        expected_write = (
            captured["beta"].to(value_component.dtype).unsqueeze(-1)
            * write_gate_component
            * value_component
        )
        call_mapping_errors = {
            "write_key_p": max_abs_difference(call["k"], captured["p"]),
            "write_value": max_abs_difference(call["v"], captured["write"]),
            "write_value_formula": max_abs_difference(
                captured["write"], expected_write
            ),
            "erase_a": max_abs_difference(call["a"], captured["a"]),
            "erase_b": max_abs_difference(call["b"], captured["b"]),
            "decay_g": max_abs_difference(call["gk"], captured["g"]),
        }
        if max(call_mapping_errors.values()) != 0.0:
            raise RuntimeError(
                f"Layer {layer_index} official call mapping changed: {call_mapping_errors}"
            )
        expected_a_fp32 = (
            torch.exp(0.5 * captured["g"].float()) * captured["r"].float()
        )
        expected_a = expected_a_fp32.to(captured["a"].dtype)
        expected_b = (
            -captured["beta"].float().unsqueeze(-1) * expected_a_fp32
        ).to(captured["b"].dtype)
        factor_errors = {
            "a": max_abs_difference(captured["a"], expected_a),
            "b": max_abs_difference(captured["b"], expected_b),
        }
        if max(factor_errors.values()) != 0.0:
            raise RuntimeError(
                f"Layer {layer_index} production factor mapping changed: {factor_errors}"
            )
        stability = transition_diagnostics(
            call["a"], call["b"], call["gk"]
        )
        if (
            stability["symmetry_error_max"] > 2e-3
            or stability["spectral_norm_max"] > 1.001
            or stability["symmetric_eigenvalue_min"] < -0.005
        ):
            raise RuntimeError(
                f"Layer {layer_index} production transition is unstable: {stability}"
            )
        terminal = captured["terminal_state"].float()
        terminal_board_rms = terminal.square().mean(dim=(-1, -2, -3)).sqrt()
        production_rows.append(
            {
                "layer": layer_index,
                "official_call_mapping_max_errors": call_mapping_errors,
                "factor_max_errors": factor_errors,
                "erase_write_separation": float(
                    (
                        1.0
                        - (
                            captured["p"].float() * captured["r"].float()
                        ).sum(dim=-1).abs()
                    ).mean().item()
                ),
                "beta_mean": float(captured["beta"].float().mean().item()),
                "write_rms": float(
                    captured["write"].float().square().mean().sqrt().item()
                ),
                "terminal_state_rms": float(
                    terminal.square().mean().sqrt().item()
                ),
                "terminal_state_board_std": float(
                    terminal_board_rms.std(unbiased=False).item()
                ),
                "transition": stability,
            }
        )

    generator = torch.Generator(device="cuda").manual_seed(53030)
    batch, length, heads, key_dim, value_dim = 2, 64, MODEL_HEADS, HEAD_DIM, HEAD_DIM
    q = F.normalize(
        torch.randn(batch, length, heads, key_dim, generator=generator, device="cuda"),
        dim=-1,
    ).to(torch.bfloat16)
    erase = F.normalize(
        torch.randn(batch, length, heads, key_dim, generator=generator, device="cuda"),
        dim=-1,
    )
    write = F.normalize(
        torch.randn(batch, length, heads, key_dim, generator=generator, device="cuda"),
        dim=-1,
    )
    value_base = 0.1 * torch.randn(
        batch, length, heads, value_dim, generator=generator, device="cuda"
    )
    write_gate = torch.sigmoid(
        torch.randn(
            batch,
            length,
            heads,
            value_dim,
            generator=generator,
            device="cuda",
        )
    )
    beta = 0.1 + 0.7 * torch.sigmoid(
        torch.randn(batch, length, heads, 1, generator=generator, device="cuda")
    )
    decay = 0.95 + 0.04 * torch.sigmoid(
        torch.randn(batch, length, heads, key_dim, generator=generator, device="cuda")
    )
    gk = decay.log()
    a, b = model.backbone.layers[0].sequence_mixer.layer.transition_factors(
        erase.to(torch.bfloat16), gk.to(torch.bfloat16), beta.squeeze(-1)
    )
    erase_bf16 = erase.to(torch.bfloat16)
    gk_bf16 = gk.to(torch.bfloat16)
    expected_sqrt_decay_erase_fp32 = (
        torch.exp(0.5 * gk_bf16.float()) * erase_bf16.float()
    )
    expected_sqrt_decay_erase = expected_sqrt_decay_erase_fp32.to(
        torch.bfloat16
    )
    expected_b = (
        -beta.squeeze(-1).float().unsqueeze(-1)
        * expected_sqrt_decay_erase_fp32
    ).to(torch.bfloat16)
    factor_a_error = max_abs_difference(a, expected_sqrt_decay_erase)
    factor_b_error = max_abs_difference(b, expected_b)
    if factor_a_error != 0.0 or factor_b_error != 0.0:
        raise RuntimeError(
            f"Contractive factor mapping changed: a={factor_a_error} b={factor_b_error}"
        )
    k = write
    v = beta * write_gate * value_base
    q, k, v, a, b, gk = (
        tensor.to(torch.bfloat16).detach().requires_grad_(True)
        for tensor in (q, k, v, a, b, gk)
    )
    initial_state = (0.1 * torch.randn(
        batch,
        heads,
        key_dim,
        value_dim,
        generator=generator,
        device="cuda",
        dtype=torch.float32,
    )).requires_grad_(True)

    output, state = run_official(
        q=q, k=k, v=v, a=a, b=b, gk=gk, initial_state=initial_state
    )
    explicit_output, explicit_state = explicit_dplr_recurrence(
        q=q, k=k, v=v, a=a, b=b, gk=gk, initial_state=initial_state
    )
    recurrence_output_relative_rms = relative_rms(output, explicit_output)
    recurrence_state_relative_rms = relative_rms(state, explicit_state)
    if recurrence_output_relative_rms > 0.03 or recurrence_state_relative_rms > 0.03:
        raise RuntimeError(
            "Official DPLR differs from explicit FP32 recurrence: "
            f"{recurrence_output_relative_rms} {recurrence_state_relative_rms}"
        )

    zero_output, zero_state = run_official(
        q=q,
        k=k,
        v=v,
        a=a,
        b=b,
        gk=gk,
        initial_state=torch.zeros_like(initial_state),
    )
    explicit_zero_output, explicit_zero_state = explicit_dplr_recurrence(
        q=q,
        k=k,
        v=v,
        a=a,
        b=b,
        gk=gk,
        initial_state=torch.zeros_like(initial_state),
    )
    zero_recurrence_output_relative_rms = relative_rms(
        zero_output, explicit_zero_output
    )
    zero_recurrence_state_relative_rms = relative_rms(
        zero_state, explicit_zero_state
    )
    if (
        zero_recurrence_output_relative_rms > 0.03
        or zero_recurrence_state_relative_rms > 0.03
    ):
        raise RuntimeError(
            "Official zero-state DPLR differs from explicit FP32 recurrence: "
            f"{zero_recurrence_output_relative_rms} "
            f"{zero_recurrence_state_relative_rms}"
        )
    incoming_output_dependency = relative_rms(output, zero_output)
    incoming_state_dependency = relative_rms(state, zero_state)
    if min(incoming_output_dependency, incoming_state_dependency) <= 1e-4:
        raise RuntimeError("Official DPLR does not depend on the incoming state")

    synthetic_loss = output.float().square().mean() + state.float().square().mean()
    synthetic_loss.backward()
    synthetic_gradient_rms = {
        name: finite_gradient_rms(tensor, f"synthetic.{name}")
        for name, tensor in {
            "q": q,
            "k": k,
            "v": v,
            "a": a,
            "b": b,
            "gk": gk,
            "initial_state": initial_state,
        }.items()
    }

    no_erase_output, no_erase_state = run_official(
        q=q,
        k=k,
        v=v,
        a=a,
        b=torch.zeros_like(b),
        gk=gk,
        initial_state=initial_state,
    )
    no_write_output, no_write_state = run_official(
        q=q,
        k=torch.zeros_like(k),
        v=v,
        a=a,
        b=b,
        gk=gk,
        initial_state=initial_state,
    )
    component_dependencies = {
        "erase_output_relative_rms": relative_rms(output, no_erase_output),
        "erase_state_relative_rms": relative_rms(state, no_erase_state),
        "write_output_relative_rms": relative_rms(output, no_write_output),
        "write_state_relative_rms": relative_rms(state, no_write_state),
    }
    if min(component_dependencies.values()) <= 1e-4:
        raise RuntimeError(f"Erase/write component is inactive: {component_dependencies}")

    permutation = torch.tensor([2, 0, 3, 1], device="cuda")
    permuted_output, permuted_state = run_official(
        q=q[:, :, permutation],
        k=k[:, :, permutation],
        v=v[:, :, permutation],
        a=a[:, :, permutation],
        b=b[:, :, permutation],
        gk=gk[:, :, permutation],
        initial_state=initial_state[:, permutation],
    )
    head_permutation_output_error = max_abs_difference(
        output[:, :, permutation], permuted_output
    )
    head_permutation_state_error = max_abs_difference(
        state[:, permutation], permuted_state
    )
    if max(head_permutation_output_error, head_permutation_state_error) > 2e-3:
        raise RuntimeError(
            "Head permutation equivariance failed: "
            f"{head_permutation_output_error} {head_permutation_state_error}"
        )

    synthetic_transition = transition_diagnostics(a, b, gk)
    if (
        synthetic_transition["symmetry_error_max"] > 2e-3
        or synthetic_transition["spectral_norm_max"] > 1.001
        or synthetic_transition["symmetric_eigenvalue_min"] < -0.005
    ):
        raise RuntimeError(
            f"Synthetic contractive transition is unstable: {synthetic_transition}"
        )

    result = {
        "status": "passed",
        "gpu": {"name": device.name, "uuid": device_uuid, "count": 1},
        "pinned_fla_sha": PINNED_FLA_SHA,
        "official_dplr_root": str(dplr_root),
        "official_dplr_tree_sha256": dplr_tree_sha256,
        "official_dplr_source_sha256": dplr_source_hashes,
        "official_chunk_source": str(chunk_source),
        "official_chunk_sha256": sha256(chunk_source),
        "official_projection_carrier_sources": {
            name: str(source) for name, source in carrier_sources.items()
        },
        "zoology_sha": zoology_sha,
        "data_hashes": data_hashes,
        "parameter_count": parameter_count,
        "mixer_parameter_count": EXPECTED_MIXER_PARAMETERS,
        "recurrent_state_values_per_layer": EXPECTED_STATE_VALUES,
        "persistent_state_delta": 0,
        "official_module_provenance": provenance,
        "official_chunk_backward_count": official_backward_count,
        "gradient_rows": gradient_rows,
        "synthetic_gradient_rms": synthetic_gradient_rms,
        "production_rows": production_rows,
        "recurrence_output_relative_rms": recurrence_output_relative_rms,
        "recurrence_state_relative_rms": recurrence_state_relative_rms,
        "zero_recurrence_output_relative_rms": (
            zero_recurrence_output_relative_rms
        ),
        "zero_recurrence_state_relative_rms": (
            zero_recurrence_state_relative_rms
        ),
        "incoming_state_output_relative_rms": incoming_output_dependency,
        "incoming_state_final_relative_rms": incoming_state_dependency,
        "component_dependencies": component_dependencies,
        "factor_a_max_error": factor_a_error,
        "factor_b_max_error": factor_b_error,
        "head_permutation_output_max_error": head_permutation_output_error,
        "head_permutation_state_max_error": head_permutation_state_error,
        "synthetic_transition": synthetic_transition,
        "fallback": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
