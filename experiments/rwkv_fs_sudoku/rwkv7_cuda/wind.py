from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Tuple

import torch
from torch.utils.cpp_extension import load


_SRC_DIR = Path(__file__).resolve().parent / "rwkv_cuda_wind"
_LOADED: Dict[int, bool] = {}


def wind_available(head_dim: int) -> Tuple[bool, str]:
    if not torch.cuda.is_available():
        return False, "CUDA is not available"
    if not torch.cuda.is_bf16_supported():
        return False, "CUDA device does not support bfloat16"
    if int(head_dim) % 16 != 0:
        return False, "RWKV7 wind kernel requires head_dim divisible by 16"
    if not (_SRC_DIR / "wind_rwkv7.cu").exists():
        return False, "vendored RWKV7 CUDA sources are missing"
    return True, "ok"


def _load_wind(head_dim: int, *, verbose: bool = False) -> None:
    head_dim = int(head_dim)
    if head_dim in _LOADED:
        return
    ok, reason = wind_available(head_dim)
    if not ok:
        raise RuntimeError(reason)

    cache_root = Path(os.environ.get("TORCH_EXTENSIONS_DIR", "")).expanduser()
    if cache_root:
        cache_root.mkdir(parents=True, exist_ok=True)

    cuda_flags = [
        "-res-usage",
        "--use_fast_math",
        "-O3",
        "-Xptxas",
        "-O3",
        "--extra-device-vectorization",
        f"-D_C_={head_dim}",
    ]
    load(
        name=f"rwkv7_wind_c{head_dim}",
        sources=[str(_SRC_DIR / "wind_rwkv7.cu"), str(_SRC_DIR / "wind_rwkv7.cpp")],
        is_python_module=False,
        verbose=verbose,
        extra_cuda_cflags=cuda_flags,
        extra_include_paths=[str(_SRC_DIR)],
    )
    _LOADED[head_dim] = True


class _WindRWKV7Fn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, w, q, k, v, z, a, s0):
        batch, seq_len, heads, head_dim = w.shape
        if seq_len % 16 != 0:
            raise ValueError("RWKV7 wind kernel requires sequence length divisible by 16")
        if any(t.dtype != torch.bfloat16 for t in (w, q, k, v, z, a, s0)):
            raise TypeError("RWKV7 wind kernel expects bfloat16 tensors")

        _load_wind(head_dim)
        w, q, k, v, z, a, s0 = [t.contiguous() for t in (w, q, k, v, z, a, s0)]
        y = torch.empty_like(v)
        states = torch.empty(
            batch,
            heads,
            seq_len // 16,
            head_dim,
            head_dim,
            dtype=torch.bfloat16,
            device=w.device,
        )
        sT = torch.empty_like(s0)
        torch.ops.wind.forward(w, q, k, v, z, a, s0, y, states, sT)
        ctx.save_for_backward(w, q, k, v, z, a, states)
        return y, sT

    @staticmethod
    def backward(ctx, dy, dsT):
        w, q, k, v, z, a, states = ctx.saved_tensors
        dy = dy.contiguous().to(torch.bfloat16)
        dsT = dsT.contiguous().to(torch.bfloat16)
        dw, dq, dk, dv, dz, da, ds0 = [torch.empty_like(t) for t in (w, q, k, v, z, a, dsT)]
        torch.ops.wind.backward(w, q, k, v, z, a, dy, states, dsT, dw, dq, dk, dv, dz, da, ds0)
        return dw, dq, dk, dv, dz, da, ds0


class WindRWKV7:
    @staticmethod
    def apply(w, q, k, v, z, a, s0):
        return _WindRWKV7Fn.apply(w, q, k, v, z, a, s0)
