from __future__ import annotations

import argparse
import copy
import inspect
import json
import os
import subprocess
from pathlib import Path

import torch
import torch.nn.functional as F
from einops import rearrange
from fla.layers.gdn2 import GatedDeltaNet2
from fla.layers.raven import Raven
from fla.models.utils import Cache
from fla.modules.convolution import ShortConvolution
from zoology.data.utils import prepare_data
from zoology.utils import set_determinism

from experiments.zoology_mqar.gdn2_futureseed import PINNED_FLA_SHA
from experiments.zoology_mqar.gdn2_raven_address import (
    EXPECTED_PARAMETER_DELTA_PER_LAYER,
    RAVEN_SLOTS,
    ZoologyRavenAddressFutureSeedMixer,
    load_matched_parent_state,
    parent_parameter_hash,
    raven_address_diagnostics,
)
from experiments.zoology_mqar.length_scaling import (
    build_config,
    dataset_hash,
    make_model,
    parameter_hash,
)
from scripts.check_zoology_gdn2_log_spd import (
    EXPECTED_TEST_HASH,
    EXPECTED_TRAIN_HASH,
    EXPECTED_ZOOLOGY_SHA,
    backward_names,
    finite_max_abs_difference,
    git_head,
    normalized_uuid,
)


EXPECTED_PARAMETER_DELTA = 2 * EXPECTED_PARAMETER_DELTA_PER_LAYER


def _mixers(model: torch.nn.Module) -> list[ZoologyRavenAddressFutureSeedMixer]:
    mixers = [block.sequence_mixer for block in model.backbone.layers]
    if len(mixers) != 2 or not all(
        isinstance(mixer, ZoologyRavenAddressFutureSeedMixer) for mixer in mixers
    ):
        raise RuntimeError("Expected exactly two Raven-address mixers")
    return mixers


def _mapped_parent_parameters(
    model: torch.nn.Module,
) -> dict[str, torch.nn.Parameter]:
    result = {}
    for name, parameter in model.named_parameters():
        if (
            ".sequence_mixer.layer.raven." in name
            or ".sequence_mixer.layer.address_adapter." in name
        ):
            continue
        parent_name = name.replace(
            ".sequence_mixer.layer.base.",
            ".sequence_mixer.layer.",
        )
        result[parent_name] = parameter
    return result


def _finite_nonzero_gradient(parameter: torch.nn.Parameter, name: str) -> float:
    if parameter.grad is None:
        raise RuntimeError(f"Missing gradient: {name}")
    gradient = parameter.grad.float()
    if not torch.isfinite(gradient).all():
        raise RuntimeError(f"Non-finite gradient: {name}")
    value = float(gradient.abs().max().item())
    if value <= 0:
        raise RuntimeError(f"Zero gradient: {name}")
    return value


def _raven_with_state(
    raven: Raven,
    hidden: torch.Tensor,
    state: tuple[torch.Tensor, torch.Tensor],
) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
    cache = Cache()
    for index in range(int(raven.layer_idx)):
        cache.update(layer_idx=index, offset=0)
    cache.update(
        recurrent_state=state,
        conv_state=None,
        layer_idx=int(raven.layer_idx),
        offset=0,
    )
    output, _attention, cache = raven(
        hidden,
        past_key_values=cache,
        use_cache=True,
    )
    return output, cache[int(raven.layer_idx)]["recurrent_state"]


def _permute_raven_heads(raven: Raven, permutation: torch.Tensor) -> None:
    heads = int(raven.num_heads)
    head_dim = int(raven.head_k_dim)
    slots = int(raven.num_slots)

    def permute_rows(weight: torch.Tensor, width: int) -> None:
        view = weight.view(heads, width, *weight.shape[1:])
        weight.copy_(view[permutation].clone().reshape_as(weight))

    with torch.no_grad():
        permute_rows(raven.q_proj.weight, head_dim)
        permute_rows(raven.k_proj.weight, head_dim)
        permute_rows(raven.v_proj.weight, int(raven.head_v_dim))
        permute_rows(raven.a_proj.weight, 1)
        permute_rows(raven.r_proj.weight, slots)
        raven.A_log.copy_(raven.A_log[permutation].clone())
        raven.dt_bias.copy_(raven.dt_bias[permutation].clone())
        if raven.g_norm.weight is not None:
            weight = raven.g_norm.weight.view(heads, int(raven.head_v_dim))
            raven.g_norm.weight.copy_(
                weight[permutation].clone().reshape_as(raven.g_norm.weight)
            )
        columns = raven.o_proj.weight.view(
            raven.hidden_size,
            heads,
            int(raven.head_v_dim),
        )
        raven.o_proj.weight.copy_(
            columns[:, permutation].clone().reshape_as(raven.o_proj.weight)
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-gpu-name", required=True)
    parser.add_argument("--expected-gpu-uuid", required=True)
    args = parser.parse_args()

    if os.environ.get("CUDA_VISIBLE_DEVICES") != "0":
        raise RuntimeError("CUDA_VISIBLE_DEVICES must be exactly 0")
    if torch.cuda.device_count() != 1 or torch.cuda.current_device() != 0:
        raise RuntimeError("P-GDN3-039 requires one visible CUDA index 0")
    device = torch.cuda.get_device_properties(0)
    device_uuid = str(getattr(device, "uuid", "unavailable"))
    if (
        device.name != args.expected_gpu_name
        or normalized_uuid(device_uuid) != normalized_uuid(args.expected_gpu_uuid)
    ):
        raise RuntimeError(f"Unexpected GPU: {device.name} {device_uuid}")
    if os.environ.get("FLA_EXPECTED_SOURCE_SHA") != PINNED_FLA_SHA:
        raise RuntimeError("Pinned FLA SHA is missing")
    if (
        os.environ.get("FLA_DISABLE_BACKEND_DISPATCH") != "1"
        or os.environ.get("FLA_CONV_BACKEND") != "triton"
    ):
        raise RuntimeError("Official backend dispatch/Triton contract failed")
    fla_root = Path(os.environ["FLA_SOURCE_ROOT"]).resolve()
    if fla_root not in Path(inspect.getfile(GatedDeltaNet2)).resolve().parents:
        raise RuntimeError("GDN2 is not imported from the pinned FLA root")
    if fla_root not in Path(inspect.getfile(Raven)).resolve().parents:
        raise RuntimeError("Raven is not imported from the pinned FLA root")
    zoology_root = Path(os.environ["ZOOLOGY_ROOT"]).resolve()
    zoology_status = subprocess.run(
        ["git", "-C", str(zoology_root), "status", "--porcelain"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if git_head(zoology_root) != EXPECTED_ZOOLOGY_SHA or zoology_status:
        raise RuntimeError("Zoology source is not exact and clean")

    control_config = build_config(
        arm="future_seed_gdn2",
        sequence_length=1024,
        num_kv_pairs=4,
        max_epochs=10,
        batch_size=32,
    )
    candidate_config = build_config(
        arm="future_seed_raven_address_gdn2",
        sequence_length=1024,
        num_kv_pairs=4,
        max_epochs=10,
        batch_size=32,
    )
    train_loader, test_loader = prepare_data(control_config.data)
    hashes = {
        "train": dataset_hash(train_loader),
        "test": dataset_hash(test_loader),
    }
    if hashes != {"train": EXPECTED_TRAIN_HASH, "test": EXPECTED_TEST_HASH}:
        raise RuntimeError(f"Directional MQAR data drifted: {hashes}")

    set_determinism(123)
    control = make_model(control_config, "future_seed_gdn2")
    control_state = copy.deepcopy(control.state_dict())
    set_determinism(123)
    candidate = make_model(candidate_config, "future_seed_raven_address_gdn2")
    load_matched_parent_state(candidate, control_state)
    if parent_parameter_hash(candidate) != parameter_hash(control):
        raise RuntimeError("Raven-address insertion changed parent initialization")
    counts = {
        "control": sum(parameter.numel() for parameter in control.parameters()),
        "candidate": sum(parameter.numel() for parameter in candidate.parameters()),
    }
    if counts["candidate"] - counts["control"] != EXPECTED_PARAMETER_DELTA:
        raise RuntimeError(f"Unexpected parameter delta: {counts}")
    control_parameters = dict(control.named_parameters())
    candidate_parameters = _mapped_parent_parameters(candidate)
    if set(control_parameters) != set(candidate_parameters):
        raise RuntimeError("Candidate parent parameter names differ from control")
    parent_init_max_diff = max(
        finite_max_abs_difference(
            control_parameters[name],
            candidate_parameters[name],
            f"parent initialization {name}",
        )
        for name in control_parameters
    )
    if parent_init_max_diff != 0.0:
        raise RuntimeError("Candidate parent tensors differ from control")

    main_layers = [module for module in candidate.modules() if isinstance(module, GatedDeltaNet2)]
    raven_layers = [module for module in candidate.modules() if isinstance(module, Raven)]
    convolutions = [
        module for module in candidate.modules() if isinstance(module, ShortConvolution)
    ]
    if len(main_layers) != 2 or len(raven_layers) != 2 or len(convolutions) != 6:
        raise RuntimeError("Official module provenance/count drifted")

    inputs, targets, _slices = next(iter(test_loader))
    inputs = inputs.cuda()
    targets = targets.cuda()
    control = control.cuda().eval()
    candidate = candidate.cuda().eval()
    mixers = _mixers(candidate)
    with torch.no_grad():
        control_logits = control(inputs)
        identity_logits = candidate(inputs)
    identity_output_max_diff = finite_max_abs_difference(
        control_logits,
        identity_logits,
        "zero-adapter full output",
    )
    if identity_output_max_diff != 0.0:
        raise RuntimeError("Zero Raven address adapter changed the parent output")

    generator = torch.Generator(device="cuda").manual_seed(52039)
    incoming_results = []
    for layer_index, (control_block, candidate_mixer) in enumerate(
        zip(control.backbone.layers, mixers)
    ):
        hidden = torch.randn(2, 128, 128, generator=generator, device="cuda")
        incoming = 0.05 * torch.randn(
            2, 4, 32, 32, generator=generator, device="cuda"
        )
        with torch.no_grad():
            control_output, control_terminal = (
                control_block.sequence_mixer.forward_with_state(
                    hidden,
                    initial_state=incoming,
                )
            )
            candidate_output, candidate_terminal = candidate_mixer.forward_with_state(
                hidden,
                initial_state=incoming,
            )
        output_diff = finite_max_abs_difference(
            control_output,
            candidate_output,
            f"incoming output {layer_index}",
        )
        state_diff = finite_max_abs_difference(
            control_terminal,
            candidate_terminal,
            f"incoming state {layer_index}",
        )
        if output_diff != 0.0 or state_diff != 0.0:
            raise RuntimeError("Zero-adapter nonzero-state identity failed")
        incoming_results.append(
            {
                "layer": layer_index,
                "output_max_diff": output_diff,
                "state_max_diff": state_diff,
            }
        )

    state_dependency = []
    for layer_index, mixer in enumerate(mixers):
        hidden = torch.randn(2, 128, 128, generator=generator, device="cuda")
        state_a = (
            0.05
            * torch.randn(
                2, 4, 32, RAVEN_SLOTS, generator=generator, device="cuda"
            ),
            0.05
            * torch.randn(
                2, 4, RAVEN_SLOTS, 32, generator=generator, device="cuda"
            ),
        )
        state_b = (state_a[0].roll(1, dims=-1), state_a[1].roll(1, dims=-2))
        with torch.no_grad():
            output_a, terminal_a = _raven_with_state(mixer.layer.raven, hidden, state_a)
            output_b, terminal_b = _raven_with_state(mixer.layer.raven, hidden, state_b)
        output_change = float(
            (output_a.float() - output_b.float()).square().mean().sqrt().item()
        )
        terminal_change = max(
            float(
                (left.float() - right.float()).square().mean().sqrt().item()
            )
            for left, right in zip(terminal_a, terminal_b)
        )
        if output_change <= 0 or terminal_change <= 0:
            raise RuntimeError("Raven ignores its recurrent state")
        state_dependency.append(
            {
                "layer": layer_index,
                "output_rms_change": output_change,
                "terminal_rms_change": terminal_change,
            }
        )

    candidate.train().zero_grad(set_to_none=True)
    stage1_logits = candidate(inputs)
    mask = targets != -100
    F.cross_entropy(stage1_logits[mask], targets[mask]).backward()
    stage1_adapter_gradients = [
        _finite_nonzero_gradient(
            mixer.layer.address_adapter.weight,
            f"layer{index}.address_adapter",
        )
        for index, mixer in enumerate(mixers)
    ]
    with torch.no_grad():
        for mixer in mixers:
            gradient = mixer.layer.address_adapter.weight.grad
            mixer.layer.address_adapter.weight.add_(-0.1 * gradient)

    candidate.train().zero_grad(set_to_none=True)
    opened_logits = candidate(inputs)
    graph = backward_names(opened_logits)
    F.cross_entropy(opened_logits[mask], targets[mask]).backward()
    official_gdn2_backward_count = sum(
        "ChunkGDN2FunctionBackward" in name for name in graph
    )
    official_raven_backward_count = sum(
        "GSA" in name and name.endswith("Backward") for name in graph
    )
    if official_gdn2_backward_count != 2 or official_raven_backward_count != 2:
        raise RuntimeError(
            "Official backward provenance failed: "
            f"GDN2={official_gdn2_backward_count} Raven={official_raven_backward_count}"
        )
    stage2_gradients = []
    for index, mixer in enumerate(mixers):
        raven = mixer.layer.raven
        stage2_gradients.append(
            {
                "layer": index,
                "adapter": _finite_nonzero_gradient(
                    mixer.layer.address_adapter.weight,
                    f"layer{index}.address_adapter.stage2",
                ),
                "q_proj": _finite_nonzero_gradient(raven.q_proj.weight, f"layer{index}.raven.q"),
                "k_proj": _finite_nonzero_gradient(raven.k_proj.weight, f"layer{index}.raven.k"),
                "v_proj": _finite_nonzero_gradient(raven.v_proj.weight, f"layer{index}.raven.v"),
                "a_proj": _finite_nonzero_gradient(raven.a_proj.weight, f"layer{index}.raven.a"),
                "r_proj": _finite_nonzero_gradient(raven.r_proj.weight, f"layer{index}.raven.r"),
            }
        )

    opened_output_relative_rms = float(
        (
            (opened_logits.float() - identity_logits.float()).square().mean().sqrt()
            / identity_logits.float().square().mean().sqrt().clamp_min(1e-8)
        ).item()
    )
    if opened_output_relative_rms <= 1e-5:
        raise RuntimeError("Synthetic adapter step did not change full output")

    candidate.eval()
    with torch.no_grad():
        candidate(inputs[:8])
    diagnostics = raven_address_diagnostics(candidate)
    for row in diagnostics["per_layer"]:
        if (
            row["raven_output_rms"] <= 0
            or row["raven_terminal_rms"] <= 0
            or row["address_residual_relative_rms"] <= 0
            or row["address_residual_token_std"] <= 0
            or row["q_projection_relative_change"] <= 0
            or row["k_projection_relative_change"] <= 0
        ):
            raise RuntimeError(f"Raven-address mechanism is inactive: {row}")

    permutation = torch.tensor([2, 0, 3, 1], device="cuda")
    permuted = copy.deepcopy(candidate).eval()
    with torch.no_grad():
        for mixer in _mixers(permuted):
            _permute_raven_heads(mixer.layer.raven, permutation)
        candidate_small = candidate(inputs[:8])
        permuted_small = permuted(inputs[:8])
    head_permutation_max_diff = finite_max_abs_difference(
        candidate_small,
        permuted_small,
        "Raven head permutation",
    )
    if head_permutation_max_diff > 2e-3:
        raise RuntimeError(
            f"Raven head permutation error is too large: {head_permutation_max_diff}"
        )

    result = {
        "status": "passed",
        "gpu": {"name": device.name, "uuid": device_uuid},
        "pinned_fla_sha": PINNED_FLA_SHA,
        "zoology_sha": EXPECTED_ZOOLOGY_SHA,
        "data_hashes": hashes,
        "parameter_counts": counts,
        "parameter_delta": EXPECTED_PARAMETER_DELTA,
        "parent_init_max_diff": parent_init_max_diff,
        "official_module_counts": {
            "gdn2": len(main_layers),
            "raven": len(raven_layers),
            "short_convolution": len(convolutions),
        },
        "identity_output_max_diff": identity_output_max_diff,
        "incoming_main_state_identity": incoming_results,
        "raven_state_dependency": state_dependency,
        "stage1_adapter_gradient_max_abs": stage1_adapter_gradients,
        "stage2_gradient_max_abs": stage2_gradients,
        "opened_output_relative_rms": opened_output_relative_rms,
        "official_gdn2_backward_count": official_gdn2_backward_count,
        "official_raven_backward_count": official_raven_backward_count,
        "raven_address_diagnostics": diagnostics,
        "head_permutation_max_diff": head_permutation_max_diff,
        "state_values_per_layer": 8_192,
        "official_scans_per_layer": {"gdn2": 1, "raven": 1},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
