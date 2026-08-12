from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import os
import subprocess
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from fla.layers.gdn2 import GatedDeltaNet2
from fla.modules.convolution import ShortConvolution
from fla.ops.gdn2 import chunk_gdn2
from zoology.utils import set_determinism

from experiments.zoology_mqar.gdn2_futureseed import PINNED_FLA_SHA
from experiments.zoology_mqar.gdn2_oig import (
    OIG_CHUNK_SIZE,
    OIGGatedDeltaNet2,
    ZoologyOIGGDN2FutureSeedMixer,
    oig_chunk_compiled,
    oig_chunk_eager,
    oig_diagnostics,
    oig_recurrence,
    parent_parameter_hash,
)
from experiments.zoology_mqar.length_scaling import (
    build_config,
    make_model,
    parameter_hash,
)


EXPECTED_ZOOLOGY_SHA = "1ad20d193b6113cae1e8f3c655c300d7b4b3f4bb"
EXPECTED_PARAMETER_DELTA = 8
BATCH = 2
LENGTH = 16
HEADS = 4
KEY_DIM = 32
VALUE_DIM = 32
FORWARD_ATOL = 5e-4
FORWARD_RTOL = 5e-4
GRAD_ATOL = 5e-3
GRAD_RTOL = 5e-3
BF16_ATOL = 3e-2
BF16_RTOL = 3e-2


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


def max_abs_difference(left: torch.Tensor, right: torch.Tensor, label: str) -> float:
    difference = (left.float() - right.float()).abs()
    if not torch.isfinite(difference).all():
        raise RuntimeError(f"Non-finite {label} difference")
    return float(difference.max().item())


def require_close(
    left: torch.Tensor,
    right: torch.Tensor,
    label: str,
    *,
    atol: float,
    rtol: float,
) -> float:
    difference = max_abs_difference(left, right, label)
    if not torch.allclose(left.float(), right.float(), atol=atol, rtol=rtol):
        raise RuntimeError(
            f"{label} mismatch: max_abs={difference} atol={atol} rtol={rtol}"
        )
    return difference


def normalize_address(tensor: torch.Tensor) -> torch.Tensor:
    tensor = tensor.float()
    return tensor / torch.sqrt(tensor.square().sum(dim=-1, keepdim=True) + 1e-6)


def make_chunk_inputs(generator: torch.Generator) -> tuple[torch.Tensor, ...]:
    q = torch.randn(
        BATCH, LENGTH, HEADS, KEY_DIM, generator=generator, device="cuda"
    )
    k = torch.randn(
        BATCH, LENGTH, HEADS, KEY_DIM, generator=generator, device="cuda"
    )
    v = torch.randn(
        BATCH, LENGTH, HEADS, VALUE_DIM, generator=generator, device="cuda"
    )
    g = -0.01 - 0.10 * torch.rand(
        BATCH, LENGTH, HEADS, KEY_DIM, generator=generator, device="cuda"
    )
    b = torch.sigmoid(
        torch.randn(
            BATCH, LENGTH, HEADS, KEY_DIM, generator=generator, device="cuda"
        )
    )
    w = torch.sigmoid(
        torch.randn(
            BATCH, LENGTH, HEADS, VALUE_DIM, generator=generator, device="cuda"
        )
    )
    state = 0.1 * torch.randn(
        BATCH, HEADS, KEY_DIM, VALUE_DIM, generator=generator, device="cuda"
    )
    covariance = torch.eye(KEY_DIM, device="cuda").reshape(
        1, 1, KEY_DIM, KEY_DIM
    ).expand(BATCH, HEADS, KEY_DIM, KEY_DIM).clone()
    mix = torch.linspace(-0.2, 0.2, HEADS, device="cuda")
    return q, k, v, g, b, w, state, covariance, mix


def clone_with_grad(tensors: tuple[torch.Tensor, ...]) -> tuple[torch.Tensor, ...]:
    return tuple(tensor.detach().clone().requires_grad_(True) for tensor in tensors)


def validate_chunk_result(
    result: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    label: str,
) -> None:
    if len(result) != 4:
        raise RuntimeError(f"{label} must return output/state/covariance/stats")
    expected_shapes = (
        (BATCH, LENGTH, HEADS, VALUE_DIM),
        (BATCH, HEADS, KEY_DIM, VALUE_DIM),
        (BATCH, HEADS, KEY_DIM, KEY_DIM),
        (5,),
    )
    for index, (tensor, shape) in enumerate(zip(result, expected_shapes, strict=True)):
        if tuple(tensor.shape) != shape:
            raise RuntimeError(
                f"{label} result {index} shape changed: {tuple(tensor.shape)} != {shape}"
            )
        if not tensor.is_cuda or not torch.isfinite(tensor).all():
            raise RuntimeError(f"{label} result {index} is not finite CUDA output")


def reset_compiler_counters() -> None:
    import torch._dynamo
    from torch._dynamo.utils import counters
    from torch._inductor import metrics

    torch._dynamo.reset()
    counters.clear()
    metrics.reset()


def compiler_counters() -> dict[str, Any]:
    from torch._dynamo.utils import counters
    from torch._inductor import metrics

    stats = dict(counters.get("stats", {}))
    graph_breaks = dict(counters.get("graph_break", {}))
    return {
        "unique_graphs": int(stats.get("unique_graphs", 0)),
        "calls_captured": int(stats.get("calls_captured", 0)),
        "generated_kernel_count": int(metrics.generated_kernel_count),
        "graph_break_count": int(sum(graph_breaks.values())),
        "graph_breaks": {str(key): int(value) for key, value in graph_breaks.items()},
    }


def check_compiled_reference(
    generator: torch.Generator,
) -> tuple[dict[str, Any], dict[str, float], dict[str, float]]:
    eager_inputs = clone_with_grad(make_chunk_inputs(generator))
    compiled_inputs = clone_with_grad(tuple(tensor.detach() for tensor in eager_inputs))

    reset_compiler_counters()
    compiled = oig_chunk_compiled(*compiled_inputs)
    torch.cuda.synchronize()
    compiler = compiler_counters()
    if (
        compiler["unique_graphs"] <= 0
        or compiler["calls_captured"] <= 0
        or compiler["generated_kernel_count"] <= 0
        or compiler["graph_break_count"] != 0
    ):
        raise RuntimeError(f"OIG chunk did not compile as a strict graph: {compiler}")
    eager = oig_chunk_eager(*eager_inputs)
    validate_chunk_result(compiled, "compiled chunk")
    validate_chunk_result(eager, "eager chunk")

    names = ("output", "state", "covariance", "stats")
    forward_differences = {
        name: require_close(
            compiled_tensor,
            eager_tensor,
            f"compiled/eager {name}",
            atol=FORWARD_ATOL,
            rtol=FORWARD_RTOL,
        )
        for name, compiled_tensor, eager_tensor in zip(
            names, compiled, eager, strict=True
        )
    }
    probes = tuple(
        torch.randn(tensor.shape, generator=generator, device="cuda")
        for tensor in eager
    )
    eager_loss = sum(
        (tensor.float() * probe).mean()
        for tensor, probe in zip(eager, probes, strict=True)
    )
    compiled_loss = sum(
        (tensor.float() * probe).mean()
        for tensor, probe in zip(compiled, probes, strict=True)
    )
    eager_loss.backward()
    compiled_loss.backward()

    input_names = (
        "q",
        "k",
        "v",
        "g",
        "b",
        "w",
        "initial_state",
        "initial_covariance",
        "mix",
    )
    gradient_differences = {}
    for name, eager_input, compiled_input in zip(
        input_names, eager_inputs, compiled_inputs, strict=True
    ):
        if eager_input.grad is None or compiled_input.grad is None:
            raise RuntimeError(f"Missing compiled/eager gradient for {name}")
        if not (
            torch.isfinite(eager_input.grad).all()
            and torch.isfinite(compiled_input.grad).all()
        ):
            raise RuntimeError(f"Non-finite compiled/eager gradient for {name}")
        gradient_differences[name] = require_close(
            compiled_input.grad,
            eager_input.grad,
            f"compiled/eager {name} gradient",
            atol=GRAD_ATOL,
            rtol=GRAD_RTOL,
        )
    return compiler, forward_differences, gradient_differences


def check_covariance(
    covariance: torch.Tensor,
) -> dict[str, float]:
    covariance = covariance.float()
    symmetry_error = float(
        (covariance - covariance.transpose(-1, -2)).abs().max().item()
    )
    eigenvalues = torch.linalg.eigvalsh(covariance)
    eigenvalue_min = float(eigenvalues.min().item())
    eigenvalue_max = float(eigenvalues.max().item())
    _factor, cholesky_info = torch.linalg.cholesky_ex(covariance)
    cholesky_failure_count = int((cholesky_info != 0).sum().item())
    if symmetry_error > 1e-5 or eigenvalue_min <= 1e-5 or eigenvalue_max > 1.0001:
        raise RuntimeError(
            "OIG covariance violated symmetry/SPD bounds: "
            f"sym={symmetry_error} eig=[{eigenvalue_min}, {eigenvalue_max}]"
        )
    if cholesky_failure_count != 0:
        raise RuntimeError(
            f"OIG covariance failed {cholesky_failure_count} Cholesky checks"
        )
    return {
        "symmetry_max_error": symmetry_error,
        "eigenvalue_min": eigenvalue_min,
        "eigenvalue_max": eigenvalue_max,
        "cholesky_failure_count": cholesky_failure_count,
    }


def check_b_zero_first_write(generator: torch.Generator) -> dict[str, float]:
    q, k, v, g, _b, _w, _state, covariance, mix = make_chunk_inputs(generator)
    b = torch.zeros_like(k)
    w = torch.zeros_like(v)
    w[:, 0] = 1.0
    v = torch.zeros_like(v).scatter(
        1,
        torch.zeros(
            BATCH, 1, HEADS, VALUE_DIM, dtype=torch.long, device="cuda"
        ),
        torch.randn(
            BATCH,
            1,
            HEADS,
            VALUE_DIM,
            generator=generator,
            device="cuda",
        ),
    )
    g = torch.zeros_like(g)
    initial_state = torch.zeros(BATCH, HEADS, KEY_DIM, VALUE_DIM, device="cuda")
    output, final_state, _final_covariance, stats = oig_chunk_eager(
        q,
        k,
        v,
        g,
        b,
        w,
        initial_state,
        covariance,
        mix,
    )
    expected = torch.einsum(
        "bhk,bhv->bhkv",
        normalize_address(k)[:, 0],
        v[:, 0],
    )
    state_difference = require_close(
        final_state,
        expected,
        "b=0 first-token direct write",
        atol=1e-5,
        rtol=1e-5,
    )
    write_rms = float(final_state.square().mean().sqrt().item())
    if write_rms <= 1e-4 or not torch.isfinite(output).all():
        raise RuntimeError("b=0 suppressed or corrupted the first-token write")
    return {
        "state_max_diff": state_difference,
        "write_rms": write_rms,
        "constraint_max_abs_error": float(stats[4].item()),
    }


def check_head_equivariance(
    generator: torch.Generator,
) -> dict[str, float]:
    inputs = make_chunk_inputs(generator)
    permutation = torch.tensor([2, 0, 3, 1], device="cuda")
    permuted = (
        *(tensor[:, :, permutation] for tensor in inputs[:6]),
        inputs[6][:, permutation],
        inputs[7][:, permutation],
        inputs[8][permutation],
    )
    with torch.no_grad():
        original = oig_chunk_compiled(*inputs)
        actual = oig_chunk_compiled(*permuted)
    expected = (
        original[0][:, :, permutation],
        original[1][:, permutation],
        original[2][:, permutation],
        original[3],
    )
    return {
        name: require_close(
            actual_tensor,
            expected_tensor,
            f"head permutation {name}",
            atol=5e-4,
            rtol=5e-4,
        )
        for name, actual_tensor, expected_tensor in zip(
            ("output", "state", "covariance", "stats"),
            actual,
            expected,
            strict=True,
        )
    }


def check_low_level_zero_mix(
    generator: torch.Generator,
) -> dict[str, float]:
    q, k, v, g, b, w, state, _covariance, _mix = make_chunk_inputs(generator)
    q_bf16 = q.to(torch.bfloat16)
    k_bf16 = k.to(torch.bfloat16)
    v_bf16 = v.to(torch.bfloat16)
    b_bf16 = b.to(torch.bfloat16)
    w_bf16 = w.to(torch.bfloat16)
    mix_logit = torch.zeros(HEADS, device="cuda")
    with torch.no_grad():
        candidate_output, candidate_state, covariance, stats = oig_recurrence(
            q=q_bf16,
            k=k_bf16,
            v=v_bf16,
            g=g,
            b=b_bf16,
            w=w_bf16,
            initial_state=state,
            mix_logit=mix_logit,
        )
        official_output, official_state = chunk_gdn2(
            q=normalize_address(q_bf16).to(torch.bfloat16),
            k=normalize_address(k_bf16).to(torch.bfloat16),
            v=v_bf16,
            g=g,
            b=b_bf16,
            w=w_bf16,
            initial_state=state,
            output_final_state=True,
            use_qk_l2norm_in_kernel=False,
        )
    if float(stats[4].item()) > 1e-5:
        raise RuntimeError(f"OIG committed-residual constraint failed: {stats[4].item()}")
    return {
        "output_max_diff": require_close(
            candidate_output,
            official_output,
            "zero-mix official output",
            atol=BF16_ATOL,
            rtol=BF16_RTOL,
        ),
        "state_max_diff": require_close(
            candidate_state,
            official_state,
            "zero-mix official state",
            atol=BF16_ATOL,
            rtol=BF16_RTOL,
        ),
        "constraint_max_abs_error": float(stats[4].item()),
        **{f"covariance_{key}": value for key, value in check_covariance(covariance).items()},
    }


def candidate_parent_parameters(
    model: torch.nn.Module,
) -> tuple[dict[str, torch.nn.Parameter], dict[str, torch.nn.Parameter]]:
    parent = {}
    logits = {}
    for name, parameter in model.named_parameters():
        if name.endswith(".oig_mix_logit"):
            logits[name] = parameter
        else:
            parent[name] = parameter
    return parent, logits


def projection_gradient_status(model: torch.nn.Module) -> list[dict[str, Any]]:
    rows = []
    for layer_index, block in enumerate(model.backbone.layers):
        layer = block.sequence_mixer.layer
        for projection_name in (
            "q_proj",
            "k_proj",
            "v_proj",
            "f_proj",
            "b_proj",
            "w_proj",
            "g_proj",
            "o_proj",
        ):
            projection = getattr(layer, projection_name)
            parameters = list(projection.parameters())
            gradients = [parameter.grad for parameter in parameters]
            present = bool(parameters) and all(
                gradient is not None for gradient in gradients
            )
            finite = present and all(
                bool(torch.isfinite(gradient).all())
                for gradient in gradients
                if gradient is not None
            )
            abs_max = (
                max(float(gradient.abs().max().item()) for gradient in gradients)
                if finite
                else None
            )
            rows.append(
                {
                    "layer": layer_index,
                    "projection": projection_name,
                    "parameter_tensors": len(parameters),
                    "present": present,
                    "finite": finite,
                    "abs_max": abs_max,
                }
            )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    parser.add_argument("--expected-gpu-name", required=True)
    parser.add_argument("--expected-gpu-uuid", required=True)
    args = parser.parse_args()

    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError(
            f"Expected exactly one CUDA device, got {torch.cuda.device_count()}"
        )
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "0":
        raise RuntimeError("CUDA_VISIBLE_DEVICES must be exactly 0")
    if torch.cuda.current_device() != 0:
        raise RuntimeError("OIG contract must execute on CUDA index 0")
    device = torch.cuda.get_device_properties(0)
    device_uuid = str(getattr(device, "uuid", "unavailable"))
    if (
        device.name != args.expected_gpu_name
        or normalized_uuid(device_uuid) != normalized_uuid(args.expected_gpu_uuid)
    ):
        raise RuntimeError(f"Unexpected GPU: {device.name} {device_uuid}")
    if OIG_CHUNK_SIZE != LENGTH:
        raise RuntimeError(f"Expected OIG chunk size {LENGTH}, got {OIG_CHUNK_SIZE}")

    if os.environ.get("FLA_EXPECTED_SOURCE_SHA") != PINNED_FLA_SHA:
        raise RuntimeError("Pinned FLA SHA is missing")
    if os.environ.get("FLA_DISABLE_BACKEND_DISPATCH") != "1":
        raise RuntimeError("FLA backend dispatch must be disabled")
    if os.environ.get("FLA_CONV_BACKEND") != "triton":
        raise RuntimeError("Short convolution must use Triton")

    fla_root = Path(os.environ["FLA_SOURCE_ROOT"]).resolve()
    gdn2_source = Path(inspect.getfile(GatedDeltaNet2)).resolve()
    short_conv_source = Path(inspect.getfile(ShortConvolution)).resolve()
    if fla_root not in gdn2_source.parents or fla_root not in short_conv_source.parents:
        raise RuntimeError(
            f"Official FLA provenance failed: {gdn2_source} {short_conv_source}"
        )
    gdn2_source_hash = hashlib.sha256(gdn2_source.read_bytes()).hexdigest()
    if gdn2_source_hash != os.environ["FLA_GDN2_SOURCE_SHA256"]:
        raise RuntimeError("Official GDN2 layer source drifted")
    gdn2_ops_hash = python_tree_hash(fla_root / "fla" / "ops" / "gdn2")
    if gdn2_ops_hash != os.environ["FLA_GDN2_OPS_SHA256"]:
        raise RuntimeError("Official GDN2 recurrence source drifted")
    zoology_root = Path(os.environ["ZOOLOGY_ROOT"]).resolve()
    if git_head(zoology_root) != EXPECTED_ZOOLOGY_SHA:
        raise RuntimeError("Unexpected Zoology SHA")

    compiled_original = getattr(oig_chunk_compiled, "_torchdynamo_orig_callable", None)
    if compiled_original is not oig_chunk_eager:
        raise RuntimeError("oig_chunk_compiled is not the strict compiled eager chunk")
    recurrence_source = inspect.getsource(oig_recurrence)
    if "oig_chunk_compiled(" not in recurrence_source or "oig_chunk_eager(" in recurrence_source:
        raise RuntimeError("Formal OIG recurrence contains an eager fallback")

    generator = torch.Generator(device="cuda").manual_seed(27027)
    compiler, forward_differences, gradient_differences = check_compiled_reference(
        generator
    )
    b_zero = check_b_zero_first_write(generator)
    head_equivariance = check_head_equivariance(generator)
    low_level_zero_mix = check_low_level_zero_mix(generator)

    control_config = build_config(
        arm="future_seed_gdn2",
        sequence_length=LENGTH,
        num_kv_pairs=1,
        max_epochs=1,
        batch_size=BATCH,
    )
    candidate_config = build_config(
        arm="future_seed_gdn2_oig",
        sequence_length=LENGTH,
        num_kv_pairs=1,
        max_epochs=1,
        batch_size=BATCH,
    )
    set_determinism(123)
    control = make_model(control_config, "future_seed_gdn2")
    set_determinism(123)
    candidate = make_model(candidate_config, "future_seed_gdn2_oig")
    if parent_parameter_hash(candidate) != parameter_hash(control):
        raise RuntimeError("OIG insertion changed the parent initialization hash")
    control_parameters = dict(control.named_parameters())
    candidate_parent, oig_logits = candidate_parent_parameters(candidate)
    if set(candidate_parent) != set(control_parameters):
        raise RuntimeError("OIG changed official parent parameter names")
    parent_max_diff = max(
        max_abs_difference(
            candidate_parent[name], control_parameters[name], f"parent tensor {name}"
        )
        for name in control_parameters
    )
    if parent_max_diff != 0.0:
        raise RuntimeError(f"OIG changed an official parent tensor: {parent_max_diff}")
    if (
        len(oig_logits) != 2
        or sum(parameter.numel() for parameter in oig_logits.values()) != 8
        or any(parameter.detach().count_nonzero().item() != 0 for parameter in oig_logits.values())
    ):
        raise RuntimeError(f"OIG must add exactly eight zero logits: {oig_logits.keys()}")
    parameter_counts = {
        "control": sum(parameter.numel() for parameter in control.parameters()),
        "candidate": sum(parameter.numel() for parameter in candidate.parameters()),
    }
    if parameter_counts["candidate"] - parameter_counts["control"] != EXPECTED_PARAMETER_DELTA:
        raise RuntimeError(f"Unexpected OIG parameter delta: {parameter_counts}")

    mixers = [block.sequence_mixer for block in candidate.backbone.layers]
    if len(mixers) != 2 or not all(
        isinstance(mixer, ZoologyOIGGDN2FutureSeedMixer) for mixer in mixers
    ):
        raise RuntimeError("length_scaling did not build two OIG mixers")
    provenance = []
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
    for layer_index, mixer in enumerate(mixers):
        layer = mixer.layer
        if not isinstance(layer, OIGGatedDeltaNet2) or not isinstance(
            layer, GatedDeltaNet2
        ):
            raise RuntimeError(f"Layer {layer_index} is not in-place official OIG GDN2")
        if hasattr(layer, "base"):
            raise RuntimeError("OIG must mutate official GDN2 in place, not wrap .base")
        control_layer = control.backbone.layers[layer_index].sequence_mixer.layer
        projection_types = {}
        for name in projection_names:
            candidate_projection = getattr(layer, name)
            control_projection = getattr(control_layer, name)
            if type(candidate_projection) is not type(control_projection):
                raise RuntimeError(f"Official projection type changed: layer={layer_index} {name}")
            projection_types[name] = type(candidate_projection).__qualname__
        conv_backends = {}
        for name in ("q_conv1d", "k_conv1d", "v_conv1d"):
            convolution = getattr(layer, name)
            if not isinstance(convolution, ShortConvolution):
                raise RuntimeError(f"Official ShortConv type changed: layer={layer_index} {name}")
            conv_backends[name] = getattr(convolution, "backend", None)
        if set(conv_backends.values()) != {"triton"}:
            raise RuntimeError(f"ShortConv fallback in layer {layer_index}: {conv_backends}")
        provenance.append(
            {
                "layer": layer_index,
                "layer_type": type(layer).__qualname__,
                "projection_types": projection_types,
                "conv_backends": conv_backends,
            }
        )

    control = control.cuda().eval()
    candidate = candidate.cuda().eval()
    hidden = torch.randn(BATCH, LENGTH, 128, generator=generator, device="cuda")
    model_parity = []
    for layer_index in range(2):
        incoming = torch.randn(
            BATCH, HEADS, KEY_DIM, VALUE_DIM, generator=generator, device="cuda"
        )
        control_mixer = control.backbone.layers[layer_index].sequence_mixer
        candidate_mixer = candidate.backbone.layers[layer_index].sequence_mixer
        with torch.no_grad():
            control_output, control_state = control_mixer.forward_with_state(
                hidden, initial_state=incoming
            )
            candidate_output, candidate_state = candidate_mixer.forward_with_state(
                hidden, initial_state=incoming
            )
        model_parity.append(
            {
                "layer": layer_index,
                "incoming_state_rms": float(incoming.square().mean().sqrt().item()),
                "output_max_diff": require_close(
                    candidate_output,
                    control_output,
                    f"layer {layer_index} zero-mix nonzero-state output",
                    atol=BF16_ATOL,
                    rtol=BF16_RTOL,
                ),
                "state_max_diff": require_close(
                    candidate_state,
                    control_state,
                    f"layer {layer_index} zero-mix nonzero-state terminal",
                    atol=BF16_ATOL,
                    rtol=BF16_RTOL,
                ),
            }
        )

    candidate.train().zero_grad(set_to_none=True)
    token_inputs = torch.randint(
        0,
        candidate_config.model.vocab_size,
        (BATCH, LENGTH),
        generator=generator,
        device="cuda",
    )
    token_targets = torch.randint(
        0,
        candidate_config.model.vocab_size,
        (BATCH, LENGTH),
        generator=generator,
        device="cuda",
    )
    logits = candidate(token_inputs)
    loss = F.cross_entropy(logits.flatten(0, 1), token_targets.flatten())
    loss.backward()
    mix_gradient_status = []
    for name, parameter in oig_logits.items():
        gradient = parameter.grad
        row = {
            "name": name,
            "present": gradient is not None,
            "finite": gradient is not None and bool(torch.isfinite(gradient).all()),
            "per_head_abs": None
            if gradient is None
            else [float(value) for value in gradient.abs().reshape(-1).tolist()],
        }
        mix_gradient_status.append(row)
    if not all(
        row["present"]
        and row["finite"]
        and all(value > 0 for value in row["per_head_abs"])
        for row in mix_gradient_status
    ):
        raise RuntimeError(f"OIG mix gradients failed: {mix_gradient_status}")
    projection_gradients = projection_gradient_status(candidate)
    if not all(
        row["present"] and row["finite"] and row["abs_max"] > 0
        for row in projection_gradients
    ):
        raise RuntimeError(f"Official projection gradients failed: {projection_gradients}")

    diagnostics = oig_diagnostics(candidate)
    if (
        diagnostics.get("active_layers") != 2
        or diagnostics.get("parameter_delta") != 8
        or diagnostics.get("compiled_fullgraph") is not True
        or diagnostics.get("compiled_backend") != "inductor"
        or diagnostics.get("formal_eager_fallback") is not False
        or len(diagnostics.get("per_layer", [])) != 2
    ):
        raise RuntimeError(f"OIG formal diagnostics contract failed: {diagnostics}")
    for row in diagnostics["per_layer"]:
        if (
            not all(torch.isfinite(torch.tensor(value)) for value in row.values() if isinstance(value, float))
            or row["minimum_denominator"] <= 1.0
            or row["constraint_max_abs_error"] > 1e-5
            or row["covariance_symmetry_max_abs_error"] > 1e-5
            or row["covariance_eigenvalue_min"] <= 1e-5
            or row["covariance_eigenvalue_max"] > 1.0001
        ):
            raise RuntimeError(f"OIG per-layer invariant failed: {row}")

    result = {
        "status": "passed",
        "gpu": {
            "name": device.name,
            "uuid": device_uuid,
            "cuda_index": torch.cuda.current_device(),
            "visible_devices": os.environ["CUDA_VISIBLE_DEVICES"],
        },
        "torch": {"version": torch.__version__, "cuda": torch.version.cuda},
        "fla": {
            "source_sha": PINNED_FLA_SHA,
            "gdn2_source": str(gdn2_source),
            "gdn2_source_sha256": gdn2_source_hash,
            "gdn2_ops_sha256": gdn2_ops_hash,
            "short_conv_source": str(short_conv_source),
        },
        "zoology_sha": EXPECTED_ZOOLOGY_SHA,
        "parameter_counts": parameter_counts,
        "parameter_delta": 8,
        "parent_parameter_hash": parent_parameter_hash(candidate),
        "parent_tensor_max_diff": parent_max_diff,
        "oig_mix_logit_names": sorted(oig_logits),
        "provenance": provenance,
        "compiler": compiler,
        "compiled_eager_forward_max_diff": forward_differences,
        "compiled_eager_gradient_max_diff": gradient_differences,
        "low_level_zero_mix": low_level_zero_mix,
        "nonzero_incoming_state_model_parity": model_parity,
        "b_zero_first_write": b_zero,
        "head_equivariance_max_diff": head_equivariance,
        "mix_gradients": mix_gradient_status,
        "projection_gradients": projection_gradients,
        "loss": float(loss.item()),
        "diagnostics": diagnostics,
        "peak_cuda_mem_bytes": torch.cuda.max_memory_allocated(),
    }
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
