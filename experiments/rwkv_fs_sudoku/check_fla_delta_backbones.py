#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.metadata
import inspect
import json
import os
import time
import zipfile
from pathlib import Path
from typing import Any, Iterable

for cache_var in (
    "XDG_CACHE_HOME",
    "TRITON_CACHE_DIR",
    "TORCHINDUCTOR_CACHE_DIR",
    "TORCH_EXTENSIONS_DIR",
    "TMPDIR",
):
    cache_path = os.environ.get(cache_var, "")
    if not cache_path.startswith("/huyang2/double-loop/"):
        raise RuntimeError(f"{cache_var} must point below /huyang2/double-loop before importing CUDA runtimes")
if os.environ.get("FLA_DISABLE_BACKEND_DISPATCH") != "1":
    raise RuntimeError("Set FLA_DISABLE_BACKEND_DISPATCH=1 so no alternate FLA backend can be selected silently")
if os.environ.get("FLA_CONV_BACKEND") != "triton":
    raise RuntimeError("Set FLA_CONV_BACKEND=triton so ShortConvolution cannot switch backends silently")

import torch
import torch.nn.functional as F

import fla
from fla.layers.gated_deltanet import GatedDeltaNet as FLAGatedDeltaNet
from fla.layers.gdn2 import GatedDeltaNet2
from fla.layers.kda import KimiDeltaAttention
from fla.ops.backends import _DISPATCH_DISABLED
from fla.ops.gated_delta_rule.chunk import chunk_gated_delta_rule
from fla.ops.gated_delta_rule.naive import naive_recurrent_gated_delta_rule
from fla.ops.gdn2 import chunk_gdn2, naive_recurrent_gdn2
from fla.ops.kda import chunk_kda
from fla.ops.kda.naive import naive_recurrent_kda
from study_rwkv_futureseed_loop import FLADeltaTimeMix, FutureSeedRWKV


EXPECTED_FLA_SHA = "9c8e42e762fce087c27b673af4922795d9edb85e"
EXPECTED_WHEEL_SHA256 = "0280db310981915eb048ece99d7bedca8b5caa9be65c99835a0f912ada977d6a"
PROVENANCE_FILES = (
    "fla/layers/gated_deltanet.py",
    "fla/layers/gdn2.py",
    "fla/layers/kda.py",
    "fla/models/utils.py",
    "fla/ops/backends/__init__.py",
    "fla/ops/gated_delta_rule/chunk.py",
    "fla/ops/gdn2/chunk.py",
    "fla/ops/gdn2/chunk_fwd.py",
    "fla/ops/gdn2/chunk_bwd.py",
    "fla/ops/gdn2/chunk_intra.py",
    "fla/ops/gdn2/chunk_intra_token_parallel.py",
    "fla/ops/gdn2/fused_recurrent.py",
    "fla/ops/gdn2/naive.py",
    "fla/ops/gdn2/wy_fast.py",
    "fla/ops/kda/chunk.py",
    "fla/modules/l2norm.py",
    "fla/modules/conv/short_conv.py",
    "fla/modules/conv/triton/ops.py",
)


def max_abs(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a.float() - b.float()).abs().max().item())


def all_finite(tensors: Iterable[torch.Tensor | None]) -> bool:
    return all(t is not None and bool(torch.isfinite(t).all()) for t in tensors)


def clone_leaves(*tensors: torch.Tensor) -> list[torch.Tensor]:
    return [tensor.detach().clone().requires_grad_(True) for tensor in tensors]


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def collect_provenance(wheel_path: Path) -> dict[str, Any]:
    wheel_path = wheel_path.resolve()
    if not wheel_path.is_file():
        raise FileNotFoundError(f"Pinned FLA wheel not found: {wheel_path}")
    wheel_sha = sha256_file(wheel_path)
    if wheel_sha != EXPECTED_WHEEL_SHA256:
        raise AssertionError(f"FLA wheel SHA mismatch: {wheel_sha} != {EXPECTED_WHEEL_SHA256}")
    if not _DISPATCH_DISABLED:
        raise AssertionError("FLA backend dispatch is active despite strict no-fallback mode")

    package_root = Path(fla.__file__).resolve().parent
    if not str(package_root).startswith("/huyang2/double-loop/"):
        raise AssertionError(f"FLA was imported outside the persistent project root: {package_root}")
    package_paths = [Path(path).resolve() for path in fla.__path__]
    if package_paths != [package_root]:
        raise AssertionError(f"FLA has unexpected namespace/package paths: {package_paths}")
    source_marker = Path("/huyang2/double-loop/.cache/fla-source-sha")
    marker_sha = source_marker.read_text(encoding="utf-8").strip() if source_marker.is_file() else ""
    if marker_sha != EXPECTED_FLA_SHA:
        raise AssertionError(f"FLA source marker mismatch: {marker_sha!r} != {EXPECTED_FLA_SHA}")

    file_hashes: dict[str, dict[str, str]] = {}
    wheel_tree_digest = hashlib.sha256()
    installed_tree_digest = hashlib.sha256()
    with zipfile.ZipFile(wheel_path) as archive:
        wheel_fla_files = sorted(
            name
            for name in archive.namelist()
            if name.startswith("fla/") and not name.endswith("/")
        )
        for archive_name in wheel_fla_files:
            installed_path = package_root.parent / archive_name
            if not installed_path.is_file():
                raise FileNotFoundError(f"Installed FLA source missing: {installed_path}")
            wheel_bytes = archive.read(archive_name)
            installed_bytes = installed_path.read_bytes()
            wheel_file_sha = sha256_bytes(wheel_bytes)
            installed_file_sha = sha256_bytes(installed_bytes)
            if wheel_file_sha != installed_file_sha:
                raise AssertionError(
                    f"Installed FLA source differs from pinned wheel for {archive_name}: "
                    f"{installed_file_sha} != {wheel_file_sha}"
                )
            file_hashes[archive_name] = {
                "wheel_sha256": wheel_file_sha,
                "installed_sha256": installed_file_sha,
            }
            encoded_name = archive_name.encode("utf-8")
            wheel_tree_digest.update(encoded_name)
            wheel_tree_digest.update(b"\0")
            wheel_tree_digest.update(bytes.fromhex(wheel_file_sha))
            installed_tree_digest.update(encoded_name)
            installed_tree_digest.update(b"\0")
            installed_tree_digest.update(bytes.fromhex(installed_file_sha))
        installed_python_files = {
            str(path.relative_to(package_root.parent))
            for path in package_root.rglob("*.py")
        }
        wheel_python_files = {
            name for name in wheel_fla_files if name.endswith(".py")
        }
        extra_python_files = sorted(installed_python_files - wheel_python_files)
        if extra_python_files:
            raise AssertionError(
                f"Installed FLA contains Python files absent from pinned wheel: {extra_python_files}"
            )
        missing_audited_files = sorted(set(PROVENANCE_FILES) - set(wheel_fla_files))
        if missing_audited_files:
            raise AssertionError(
                f"Pinned wheel lacks required audited files: {missing_audited_files}"
            )

    symbols = {
        "fla_gdn_layer": FLAGatedDeltaNet,
        "gdn2_layer": GatedDeltaNet2,
        "kda_layer": KimiDeltaAttention,
        "fla_gdn_op": chunk_gated_delta_rule,
        "gdn2_op": chunk_gdn2,
        "kda_op": chunk_kda,
    }
    symbol_rows = {}
    for name, symbol in symbols.items():
        module = importlib.import_module(symbol.__module__)
        module_path = Path(inspect.getfile(module)).resolve()
        if package_root not in module_path.parents:
            raise AssertionError(f"{name} module resolved outside installed FLA: {module_path}")
        exported_symbol = getattr(module, symbol.__name__, None)
        if exported_symbol is not symbol:
            raise AssertionError(f"{name} is not the callable exported by {symbol.__module__}.{symbol.__name__}")

        wrapper_path = Path(inspect.getfile(symbol)).resolve()
        unwrapped_symbol = inspect.unwrap(symbol)
        unwrapped_path = Path(inspect.getfile(unwrapped_symbol)).resolve()
        symbol_rows[name] = {
            "module": symbol.__module__,
            "qualname": symbol.__qualname__,
            "module_path": str(module_path),
            "wrapper_path": str(wrapper_path),
            "unwrapped_path": str(unwrapped_path),
            "wrapped": unwrapped_symbol is not symbol,
        }
    return {
        "fla_version": importlib.metadata.version("flash-linear-attention"),
        "fla_package_root": str(package_root),
        "fla_package_paths": [str(path) for path in package_paths],
        "fla_source_sha": marker_sha,
        "wheel_path": str(wheel_path),
        "wheel_sha256": wheel_sha,
        "backend_dispatch_disabled": bool(_DISPATCH_DISABLED),
        "conv_backend": os.environ["FLA_CONV_BACKEND"],
        "symbols": symbol_rows,
        "source_file_hashes": file_hashes,
        "wheel_fla_tree_sha256": wheel_tree_digest.hexdigest(),
        "installed_fla_tree_sha256": installed_tree_digest.hexdigest(),
        "verified_fla_file_count": len(file_hashes),
    }


def autograd_graph_names(tensor: torch.Tensor) -> list[str]:
    names: list[str] = []
    queue = [tensor.grad_fn]
    seen: set[int] = set()
    while queue:
        fn = queue.pop(0)
        if fn is None or id(fn) in seen:
            continue
        seen.add(id(fn))
        names.append(type(fn).__name__)
        queue.extend(next_fn for next_fn, _index in fn.next_functions if next_fn is not None)
    return names


def check_gdn_reference(
    device: torch.device,
    *,
    key_dim: int = 32,
    value_dim: int = 64,
    seed: int = 100,
) -> dict[str, Any]:
    torch.manual_seed(seed)
    batch, length, heads = 1, 81, 2
    q = torch.randn(batch, length, heads, key_dim, device=device)
    k = torch.randn_like(q)
    v = torch.randn(batch, length, heads, value_dim, device=device) * 0.25
    g_raw = torch.randn(batch, length, heads, device=device)
    beta_raw = torch.randn(batch, length, heads, device=device)
    a_log = torch.log(torch.empty(heads, device=device).uniform_(1.0, 16.0))
    dt_bias = torch.randn(heads, device=device)
    h0_vk = torch.randn(batch, heads, value_dim, key_dim, device=device) * 0.1
    do = torch.randn_like(v)
    dht_vk = torch.randn_like(h0_vk) * 0.1

    ref_leaves = clone_leaves(q, k, v, g_raw, beta_raw, a_log, dt_bias, h0_vk)
    rq, rk, rv, rg, rbeta, ra, rdt, rh0_vk = ref_leaves
    g_ref = -ra.float().exp().view(1, 1, heads) * F.softplus(rg.float() + rdt.float().view(1, 1, heads))
    ref, ref_ht_kv = naive_recurrent_gated_delta_rule(
        q=F.normalize(rq.float(), dim=-1),
        k=F.normalize(rk.float(), dim=-1),
        v=rv,
        g=g_ref,
        beta=torch.sigmoid(rbeta),
        initial_state=rh0_vk.transpose(-1, -2),
        output_final_state=True,
    )
    ref_loss = (ref * do).sum() + (ref_ht_kv.transpose(-1, -2) * dht_vk).sum()
    ref_grads = torch.autograd.grad(ref_loss, ref_leaves)

    tri_leaves = clone_leaves(q, k, v, g_raw, beta_raw, a_log, dt_bias, h0_vk)
    tq, tk, tv, tg, tbeta, ta, tdt, th0_vk = tri_leaves
    tri, tri_ht_vk = chunk_gated_delta_rule(
        q=tq,
        k=tk,
        v=tv,
        g=tg,
        beta=tbeta,
        A_log=ta,
        dt_bias=tdt,
        initial_state=th0_vk,
        output_final_state=True,
        use_qk_l2norm_in_kernel=True,
        use_gate_in_kernel=True,
        use_beta_sigmoid_in_kernel=True,
        state_v_first=True,
    )
    tri_loss = (tri * do).sum() + (tri_ht_vk * dht_vk).sum()
    tri_grads = torch.autograd.grad(tri_loss, tri_leaves)

    grad_errors = [max_abs(a, b) for a, b in zip(ref_grads, tri_grads)]
    result = {
        "output_max_abs": max_abs(ref, tri),
        "state_max_abs": max_abs(ref_ht_kv.transpose(-1, -2), tri_ht_vk),
        "gradient_max_abs": max(grad_errors),
        "gradient_errors": dict(zip(("q", "k", "v", "g", "beta", "A_log", "dt_bias", "h0"), grad_errors)),
    }
    if result["output_max_abs"] > 0.02 or result["state_max_abs"] > 0.02 or result["gradient_max_abs"] > 0.08:
        raise AssertionError(f"GDN FLA/Torch mismatch: {result}")
    return result


def check_gdn2_reference(device: torch.device) -> dict[str, Any]:
    torch.manual_seed(101)
    batch, length, heads, key_dim, value_dim = 1, 81, 2, 32, 32
    q = torch.randn(batch, length, heads, key_dim, device=device)
    k = torch.randn_like(q)
    v = torch.randn(batch, length, heads, value_dim, device=device) * 0.25
    g = torch.empty_like(q).uniform_(-4.0, -0.05)
    b = torch.rand_like(q)
    w = torch.rand_like(v)
    h0_vk = torch.randn(batch, heads, value_dim, key_dim, device=device) * 0.1
    do = torch.randn_like(v)
    dht_vk = torch.randn_like(h0_vk) * 0.1

    ref_leaves = clone_leaves(q, k, v, g, b, w, h0_vk)
    rq, rk, rv, rg, rb, rw, rh0_vk = ref_leaves
    ref, ref_ht_kv = naive_recurrent_gdn2(
        q=F.normalize(rq.float(), dim=-1),
        k=F.normalize(rk.float(), dim=-1),
        v=rv,
        g=rg,
        b=rb,
        w=rw,
        initial_state=rh0_vk.transpose(-1, -2),
        output_final_state=True,
    )
    ref_loss = (ref * do).sum() + (ref_ht_kv.transpose(-1, -2) * dht_vk).sum()
    ref_grads = torch.autograd.grad(ref_loss, ref_leaves)

    tri_leaves = clone_leaves(q, k, v, g, b, w, h0_vk)
    tq, tk, tv, tg, tb, tw, th0_vk = tri_leaves
    tri, tri_ht_vk = chunk_gdn2(
        q=tq,
        k=tk,
        v=tv,
        g=tg,
        b=tb,
        w=tw,
        initial_state=th0_vk,
        output_final_state=True,
        use_qk_l2norm_in_kernel=True,
        state_v_first=True,
    )
    tri_loss = (tri * do).sum() + (tri_ht_vk * dht_vk).sum()
    tri_grads = torch.autograd.grad(tri_loss, tri_leaves)

    grad_errors = [max_abs(a, b) for a, b in zip(ref_grads, tri_grads)]
    result = {
        "output_max_abs": max_abs(ref, tri),
        "state_max_abs": max_abs(ref_ht_kv.transpose(-1, -2), tri_ht_vk),
        "gradient_max_abs": max(grad_errors),
        "gradient_errors": dict(zip(("q", "k", "v", "g", "b", "w", "h0"), grad_errors)),
    }
    if result["output_max_abs"] > 0.02 or result["state_max_abs"] > 0.02 or result["gradient_max_abs"] > 0.08:
        raise AssertionError(f"GDN2 FLA/Torch mismatch: {result}")
    return result


def check_kda_reference(device: torch.device) -> dict[str, Any]:
    torch.manual_seed(202)
    batch, length, heads, key_dim, value_dim = 1, 81, 2, 32, 32
    q = torch.randn(batch, length, heads, key_dim, device=device)
    k = torch.randn_like(q)
    v = torch.randn(batch, length, heads, value_dim, device=device) * 0.25
    g_raw = torch.randn_like(q)
    beta_raw = torch.randn(batch, length, heads, device=device)
    a_log = torch.log(torch.empty(heads, device=device).uniform_(1.0, 16.0))
    dt_bias = torch.randn(heads * key_dim, device=device)
    h0_vk = torch.randn(batch, heads, value_dim, key_dim, device=device) * 0.1
    do = torch.randn_like(v)
    dht_vk = torch.randn_like(h0_vk) * 0.1

    ref_leaves = clone_leaves(q, k, v, g_raw, beta_raw, a_log, dt_bias, h0_vk)
    rq, rk, rv, rg, rbeta, ra, rdt, rh0_vk = ref_leaves
    g_ref = -ra.float().exp().view(1, 1, heads, 1) * F.softplus(
        rg.float() + rdt.float().view(1, 1, heads, key_dim)
    )
    ref, ref_ht_kv = naive_recurrent_kda(
        q=F.normalize(rq.float(), dim=-1),
        k=F.normalize(rk.float(), dim=-1),
        v=rv,
        g=g_ref,
        beta=torch.sigmoid(rbeta),
        initial_state=rh0_vk.transpose(-1, -2),
        output_final_state=True,
    )
    ref_loss = (ref * do).sum() + (ref_ht_kv.transpose(-1, -2) * dht_vk).sum()
    ref_grads = torch.autograd.grad(ref_loss, ref_leaves)

    tri_leaves = clone_leaves(q, k, v, g_raw, beta_raw, a_log, dt_bias, h0_vk)
    tq, tk, tv, tg, tbeta, ta, tdt, th0_vk = tri_leaves
    tri, tri_ht_vk = chunk_kda(
        q=tq,
        k=tk,
        v=tv,
        g=tg,
        beta=tbeta,
        A_log=ta,
        dt_bias=tdt,
        initial_state=th0_vk,
        output_final_state=True,
        use_qk_l2norm_in_kernel=True,
        use_gate_in_kernel=True,
        use_beta_sigmoid_in_kernel=True,
        state_v_first=True,
    )
    tri_loss = (tri * do).sum() + (tri_ht_vk * dht_vk).sum()
    tri_grads = torch.autograd.grad(tri_loss, tri_leaves)

    grad_errors = [max_abs(a, b) for a, b in zip(ref_grads, tri_grads)]
    result = {
        "output_max_abs": max_abs(ref, tri),
        "state_max_abs": max_abs(ref_ht_kv.transpose(-1, -2), tri_ht_vk),
        "gradient_max_abs": max(grad_errors),
        "gradient_errors": dict(zip(("q", "k", "v", "g", "beta", "A_log", "dt_bias", "h0"), grad_errors)),
    }
    if result["output_max_abs"] > 0.02 or result["state_max_abs"] > 0.02 or result["gradient_max_abs"] > 0.08:
        raise AssertionError(f"KDA FLA/Torch mismatch: {result}")
    return result


def check_adapter(backbone: str, device: torch.device) -> dict[str, Any]:
    seeds = {"fla_gdn": 303, "gdn2": 404, "kda": 505}
    expected_layers = {
        "fla_gdn": FLAGatedDeltaNet,
        "gdn2": GatedDeltaNet2,
        "kda": KimiDeltaAttention,
    }
    expected_autograd = {
        "fla_gdn": "ChunkGatedDeltaRuleFunctionBackward",
        "gdn2": "ChunkGDN2FunctionBackward",
        "kda": "ChunkKDAFunctionBackward",
    }
    torch.manual_seed(seeds[backbone])
    batch, length, heads, head_dim = 2, 81, 4, 16
    mixer = FLADeltaTimeMix(
        heads * head_dim,
        heads,
        head_dim,
        backbone=backbone,
        expand_v=2.0,
        mode="chunk",
        use_short_conv=True,
        conv_size=4,
        allow_neg_eigval=False,
    ).to(device)
    if type(mixer.core) is not expected_layers[backbone]:
        raise AssertionError(f"{backbone} did not instantiate the exact official FLA layer class: {type(mixer.core)}")
    conv_backends = {
        name: getattr(mixer.core, name).backend
        for name in ("q_conv1d", "k_conv1d", "v_conv1d")
    }
    if set(conv_backends.values()) != {"triton"}:
        raise AssertionError(f"{backbone} short convolution silently changed backend: {conv_backends}")
    official_forward_calls = 0

    def count_forward(_module, _inputs, _output):
        nonlocal official_forward_calls
        official_forward_calls += 1

    hook = mixer.core.register_forward_hook(count_forward)
    x = torch.randn(batch, length, heads * head_dim, device=device, requires_grad=True)
    state_tail = (head_dim * 2, head_dim) if mixer.state_v_first else (head_dim, head_dim * 2)
    h0 = torch.randn(
        batch,
        heads,
        *state_tail,
        device=device,
        dtype=torch.float32,
        requires_grad=True,
    ) * 0.05
    h0.retain_grad()
    torch.cuda.reset_peak_memory_stats(device)
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        y, ht = mixer(x, initial_state=h0)
        loss = y.float().square().mean() + ht.float().square().mean() * 1e-3
    graph_names = autograd_graph_names(ht)
    if expected_autograd[backbone] not in graph_names:
        raise AssertionError(
            f"{backbone} terminal state did not traverse the expected official FLA chunk op; "
            f"expected={expected_autograd[backbone]}, graph={graph_names}"
        )
    loss.backward()
    hook.remove()
    if official_forward_calls != 1:
        raise AssertionError(f"{backbone} official layer forward count was {official_forward_calls}, expected exactly 1")
    if tuple(ht.shape) != (batch, heads, *state_tail):
        raise AssertionError(
            f"{backbone} official cache state shape {tuple(ht.shape)} does not preserve native layout "
            f"{(batch, heads, *state_tail)}"
        )
    parameter_grads = [p.grad for p in mixer.parameters() if p.requires_grad]
    if not all_finite([y, ht, x.grad, h0.grad, *parameter_grads]):
        raise AssertionError(f"{backbone} adapter produced non-finite output or gradient")

    for _ in range(2):
        mixer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            bench_y, bench_ht = mixer(x.detach(), initial_state=h0.detach())
            bench_loss = bench_y.float().square().mean() + bench_ht.float().square().mean() * 1e-3
        bench_loss.backward()
    torch.cuda.synchronize(device)
    repeats = 5
    started = time.perf_counter()
    for _ in range(repeats):
        mixer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            bench_y, bench_ht = mixer(x.detach(), initial_state=h0.detach())
            bench_loss = bench_y.float().square().mean() + bench_ht.float().square().mean() * 1e-3
        bench_loss.backward()
    torch.cuda.synchronize(device)
    elapsed_ms = (time.perf_counter() - started) * 1000.0 / repeats
    return {
        "parameters": sum(p.numel() for p in mixer.parameters()),
        "output_shape": list(y.shape),
        "state_shape": list(ht.shape),
        "state_v_first": mixer.state_v_first,
        "state_dtype": str(ht.dtype),
        "initial_state_grad_norm": float(h0.grad.float().norm().item()),
        "official_layer_class": f"{type(mixer.core).__module__}.{type(mixer.core).__qualname__}",
        "official_layer_forward_calls": official_forward_calls,
        "official_chunk_autograd_node": expected_autograd[backbone],
        "conv_backends": conv_backends,
        "forward_backward_ms": elapsed_ms,
        "peak_memory_mb": torch.cuda.max_memory_allocated(device) / (1024**2),
    }


def check_futureseed_stack(backbone: str, device: torch.device) -> dict[str, Any]:
    seeds = {"fla_gdn": 606, "gdn2": 707, "kda": 808}
    torch.manual_seed(seeds[backbone])
    batch, length, heads, head_dim = 2, 81, 4, 16
    model = FutureSeedRWKV(
        d_model=heads * head_dim,
        layers=3,
        heads=heads,
        head_dim=head_dim,
        channel_mult=2,
        future_seed_scale=1.0,
        future_seed_decay=0.0,
        future_seed_update="fixed",
        future_seed_norm_mode="unit",
        activation_checkpoint=False,
        rwkv_kernel="torch",
        backbone=backbone,
        gdn_mode="chunk",
        gdn_expand_v=2.0,
        gdn_use_short_conv=True,
        gdn_conv_size=4,
        gdn_allow_neg_eigval=False,
    ).to(device)
    x = torch.randn(batch, length, heads * head_dim, device=device, requires_grad=True)
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        y, diagnostics, next_memory = model(x)
        loss = y.float().square().mean()
    loss.backward()
    missing_parameter_grads = [name for name, p in model.named_parameters() if p.requires_grad and p.grad is None]
    nonfinite_parameter_grads = [
        name
        for name, p in model.named_parameters()
        if p.requires_grad and p.grad is not None and not bool(torch.isfinite(p.grad).all())
    ]
    # Layer zero consumes no FutureSeed because no preceding terminal state exists.
    expected_missing = {"blocks.0.future_seed_logit"}
    unexpected_missing = sorted(set(missing_parameter_grads) - expected_missing)
    if not all_finite([y, x.grad]) or unexpected_missing or nonfinite_parameter_grads:
        raise AssertionError(
            f"{backbone} FutureSeed stack failed: "
            f"unexpected_missing={unexpected_missing}, nonfinite={nonfinite_parameter_grads}, "
            f"output_finite={bool(torch.isfinite(y).all())}, input_grad_finite={bool(torch.isfinite(x.grad).all())}"
        )
    if next_memory is not None:
        raise AssertionError("fixed FutureSeed update unexpectedly returned loop memory")
    fs_gate = float(diagnostics["fs_gate_mean"].item())
    fs_state_norm = float(diagnostics["fs_state_norm"].item())
    if not (0.0 < fs_gate < 1.0) or fs_state_norm <= 0.0:
        raise AssertionError(f"{backbone} FutureSeed path is inactive: {diagnostics}")
    return {
        "parameters": sum(p.numel() for p in model.parameters()),
        "output_shape": list(y.shape),
        "fs_gate_mean": fs_gate,
        "fs_state_norm": fs_state_norm,
        "input_grad_norm": float(x.grad.float().norm().item()),
        "expected_missing_parameter_grads": sorted(set(missing_parameter_grads) & expected_missing),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backbone", choices=("all", "fla_gdn", "gdn2", "kda"), default="all")
    parser.add_argument("--check", choices=("all", "reference", "adapter", "futureseed_stack"), default="all")
    parser.add_argument(
        "--wheel",
        type=Path,
        default=Path(
            "/huyang2/double-loop/wheelhouse/"
            "flash_linear_attention-0.5.2-9c8e42e-py3-none-any.whl"
        ),
    )
    parser.add_argument("--out", type=Path, default=None)
    return parser.parse_args()


def write_payload(payload: dict[str, Any], out: Path | None) -> None:
    text = json.dumps(payload, indent=2, sort_keys=True)
    if out is not None:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required; CPU smoke is intentionally forbidden.")
    device = torch.device("cuda", 0)
    payload: dict[str, Any] = {
        "torch": torch.__version__,
        "device": torch.cuda.get_device_name(device),
        "requested_backbone": args.backbone,
        "requested_check": args.check,
        "provenance": collect_provenance(args.wheel),
    }
    selected = ("fla_gdn", "kda", "gdn2") if args.backbone == "all" else (args.backbone,)
    checks = {
        "fla_gdn": (
            ("fla_gdn_reference", lambda: check_gdn_reference(device)),
            ("fla_gdn_adapter", lambda: check_adapter("fla_gdn", device)),
            ("fla_gdn_futureseed_stack", lambda: check_futureseed_stack("fla_gdn", device)),
        ),
        "gdn2": (
            ("gdn2_reference", lambda: check_gdn2_reference(device)),
            ("gdn2_adapter", lambda: check_adapter("gdn2", device)),
            ("gdn2_futureseed_stack", lambda: check_futureseed_stack("gdn2", device)),
        ),
        "kda": (
            ("kda_reference", lambda: check_kda_reference(device)),
            ("kda_adapter", lambda: check_adapter("kda", device)),
            ("kda_futureseed_stack", lambda: check_futureseed_stack("kda", device)),
        ),
    }
    for backbone in selected:
        for name, check in checks[backbone]:
            check_kind = name.removeprefix(f"{backbone}_")
            if args.check != "all" and args.check != check_kind:
                continue
            print(f"[cuda-check] start {name}", flush=True)
            started = time.perf_counter()
            payload[name] = check()
            payload[name]["check_wall_sec"] = time.perf_counter() - started
            write_payload(payload, args.out)
            print(f"[cuda-check] pass {name} in {payload[name]['check_wall_sec']:.2f}s", flush=True)
    write_payload(payload, args.out)
    print(json.dumps(payload, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
