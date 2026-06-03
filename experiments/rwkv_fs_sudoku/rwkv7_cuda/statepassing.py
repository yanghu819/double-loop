from __future__ import annotations

import os
from pathlib import Path
from typing import Optional, Tuple

import torch
from torch.utils.cpp_extension import load


_SRC_DIR = Path(__file__).resolve().parent / "rwkv_cuda_statepassing"
_LOADED_HEAD_DIM: Optional[int] = None
_CHUNK_LEN = 16


def statepassing_available(head_dim: int) -> Tuple[bool, str]:
    head_dim = int(head_dim)
    if not torch.cuda.is_available():
        return False, "CUDA is not available"
    if not torch.cuda.is_bf16_supported():
        return False, "CUDA device does not support bfloat16"
    if head_dim % 4 != 0:
        return False, "RWKV7 state-passing kernel requires head_dim divisible by 4"
    if not (_SRC_DIR / "rwkv7_statepassing_clampw.cu").exists():
        return False, "vendored RWKV7 state-passing CUDA sources are missing"
    return True, "ok"


def _load_statepassing(head_dim: int, *, verbose: bool = False) -> None:
    global _LOADED_HEAD_DIM
    head_dim = int(head_dim)
    if _LOADED_HEAD_DIM == head_dim:
        return
    if _LOADED_HEAD_DIM is not None and _LOADED_HEAD_DIM != head_dim:
        raise RuntimeError(
            "rwkv7_statepassing_clampw registers a fixed torch.ops namespace; "
            f"already loaded head_dim={_LOADED_HEAD_DIM}, cannot also load head_dim={head_dim}"
        )
    ok, reason = statepassing_available(head_dim)
    if not ok:
        raise RuntimeError(reason)

    cache_root = Path(os.environ.get("TORCH_EXTENSIONS_DIR", "")).expanduser()
    if cache_root:
        cache_root.mkdir(parents=True, exist_ok=True)

    cuda_flags = [
        "-res-usage",
        f"-D_N_={head_dim}",
        f"-D_CHUNK_LEN_={_CHUNK_LEN}",
        "--use_fast_math",
        "-O3",
        "-Xptxas",
        "-O3",
        "--extra-device-vectorization",
    ]
    load(
        name=f"rwkv7_statepassing_clampw_n{head_dim}",
        sources=[
            str(_SRC_DIR / "rwkv7_statepassing_clampw.cu"),
            str(_SRC_DIR / "rwkv7_statepassing_clampw.cpp"),
        ],
        is_python_module=False,
        verbose=verbose,
        extra_cuda_cflags=cuda_flags,
        extra_include_paths=[str(_SRC_DIR)],
    )
    _LOADED_HEAD_DIM = head_dim


class _StatePassingRWKV7Fn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, s0, r, w, k, v, a, b):
        batch, seq_len, heads, head_dim = r.shape
        if seq_len % _CHUNK_LEN != 0:
            raise ValueError("RWKV7 state-passing kernel requires sequence length divisible by 16")
        if any(t.dtype != torch.bfloat16 for t in (r, w, k, v, a, b)):
            raise TypeError("RWKV7 state-passing kernel expects bfloat16 recurrent tensors")
        if s0.dtype != torch.float32:
            raise TypeError("RWKV7 state-passing kernel expects float32 initial state")

        _load_statepassing(head_dim)
        s0, r, w, k, v, a, b = [t.contiguous() for t in (s0, r, w, k, v, a, b)]
        y = torch.empty_like(v)
        sT = torch.empty_like(s0)
        states = torch.empty(
            batch,
            heads,
            seq_len // _CHUNK_LEN,
            head_dim,
            head_dim,
            dtype=torch.float32,
            device=w.device,
        )
        sa = torch.empty(batch, seq_len, heads, head_dim, dtype=torch.float32, device=w.device)
        torch.ops.rwkv7_statepassing_clampw.forward(s0, r, w, k, v, a, b, y, sT, states, sa)
        ctx.save_for_backward(r, w, k, v, a, b, states, sa)
        return y, sT

    @staticmethod
    def backward(ctx, dy, dsT):
        r, w, k, v, a, b, states, sa = ctx.saved_tensors
        dy = dy.contiguous().to(torch.bfloat16)
        dsT = dsT.contiguous().to(torch.float32)
        ds0 = torch.empty_like(dsT)
        dr, dw, dk, dv, da, db = [torch.empty_like(t) for t in (r, w, k, v, a, b)]
        torch.ops.rwkv7_statepassing_clampw.backward(
            r,
            w,
            k,
            v,
            a,
            b,
            dy,
            dsT,
            states,
            sa,
            ds0,
            dr,
            dw,
            dk,
            dv,
            da,
            db,
        )
        return ds0, dr, dw, dk, dv, da, db


class StatePassingRWKV7:
    @staticmethod
    def apply(s0, r, w, k, v, a, b):
        return _StatePassingRWKV7Fn.apply(s0, r, w, k, v, a, b)
