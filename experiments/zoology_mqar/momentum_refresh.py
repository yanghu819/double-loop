from __future__ import annotations

import importlib
from contextlib import contextmanager
from typing import Any, Iterator, Optional

import torch

from experiments.zoology_mqar.momentum_futureseed import (
    ZoologyMomentumDeltaFutureSeedMixer,
    load_external_momentum_layer,
    momentum_futureseed_diagnostics,
)


def interleave_microsteps(
    parent: torch.Tensor,
    refresh: torch.Tensor,
) -> torch.Tensor:
    if parent.shape != refresh.shape or parent.ndim < 2:
        raise ValueError(
            f"Interleaved tensors must share [B,T,...], got "
            f"{parent.shape} and {refresh.shape}"
        )
    return torch.stack((parent, refresh), dim=2).flatten(1, 2).contiguous()


class PostCommitMomentumRefreshOperation:
    """Append one residual-only Momentum refresh after each parent microstep."""

    def __init__(self, operation, *, collect_diagnostics: bool = True) -> None:
        self.operation = operation
        self.collect_diagnostics = bool(collect_diagnostics)
        self.last_stats: Optional[dict[str, Any]] = None

    def __call__(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        log_alpha: torch.Tensor,
        log_mu: torch.Tensor,
        p: Optional[torch.Tensor] = None,
        beta: Optional[torch.Tensor] = None,
        eta: Optional[torch.Tensor] = None,
        scale: Optional[float] = None,
        initial_state: Optional[torch.Tensor] = None,
        output_final_state: bool = False,
        cu_seqlens: Optional[torch.LongTensor] = None,
        use_qk_l2norm_in_kernel: bool = True,
        use_p_times_alpha: bool = True,
    ):
        if cu_seqlens is not None:
            raise RuntimeError("P-GDN3-061 fixes equal-length inputs")
        if q.ndim != 4 or q.shape[:3] != k.shape[:3] or q.shape[:3] != v.shape[:3]:
            raise ValueError("Momentum refresh requires aligned [B,T,H,*] tensors")
        if log_alpha.shape != log_mu.shape or log_alpha.shape != q.shape[:3]:
            raise ValueError("Momentum refresh gate geometry drifted")

        parent_beta = torch.ones_like(log_alpha) if beta is None else beta
        parent_eta = torch.ones_like(log_alpha) if eta is None else eta
        zeros_alpha = torch.zeros_like(log_alpha)
        zeros_mu = torch.zeros_like(log_mu)
        zeros_beta = torch.zeros_like(parent_beta)

        micro_q = interleave_microsteps(q, q)
        micro_k = interleave_microsteps(k, k)
        micro_v = interleave_microsteps(v, v)
        micro_p = None if p is None else interleave_microsteps(p, p)
        micro_log_alpha = interleave_microsteps(log_alpha, zeros_alpha)
        micro_log_mu = interleave_microsteps(log_mu, zeros_mu)
        micro_beta = interleave_microsteps(parent_beta, zeros_beta)
        micro_eta = interleave_microsteps(parent_eta, parent_eta)

        micro_output, final_state = self.operation(
            q=micro_q,
            k=micro_k,
            v=micro_v,
            p=micro_p,
            log_alpha=micro_log_alpha,
            log_mu=micro_log_mu,
            beta=micro_beta,
            eta=micro_eta,
            scale=scale,
            initial_state=initial_state,
            output_final_state=output_final_state,
            cu_seqlens=None,
            use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
            use_p_times_alpha=use_p_times_alpha,
        )
        parent_output = micro_output[:, 0::2]
        refresh_output = micro_output[:, 1::2]
        refresh_output_relative_rms = None
        refresh_output_max_abs = None
        if self.collect_diagnostics:
            parent_diag = parent_output.detach().float()
            refresh_diag = refresh_output.detach().float()
            delta = (refresh_diag - parent_diag).square().mean().sqrt()
            reference = parent_diag.square().mean().sqrt().clamp_min(1e-8)
            refresh_output_relative_rms = (delta / reference).detach()
            refresh_output_max_abs = (refresh_diag - parent_diag).abs().max().detach()
        self.last_stats = {
            "calls": 1,
            "parent_tokens": int(q.shape[1]),
            "refresh_tokens": int(q.shape[1]),
            "micro_tokens": int(micro_q.shape[1]),
            "refresh_log_alpha_max_abs": 0.0,
            "refresh_log_mu_max_abs": 0.0,
            "refresh_beta_max_abs": 0.0,
            "refresh_output_relative_rms": refresh_output_relative_rms,
            "refresh_output_max_abs": refresh_output_max_abs,
        }
        return parent_output, final_state


@contextmanager
def scoped_post_commit_momentum_refresh(
    *,
    collect_diagnostics: bool,
) -> Iterator[PostCommitMomentumRefreshOperation]:
    layer_class = load_external_momentum_layer()
    layer_module = importlib.import_module(layer_class.__module__)
    original = layer_module.chunk_mode_rule
    if isinstance(original, PostCommitMomentumRefreshOperation):
        raise RuntimeError("Nested Momentum refresh scope is not supported")
    wrapped = PostCommitMomentumRefreshOperation(
        original,
        collect_diagnostics=collect_diagnostics,
    )
    layer_module.chunk_mode_rule = wrapped
    try:
        yield wrapped
    finally:
        if layer_module.chunk_mode_rule is not wrapped:
            raise RuntimeError("Momentum chunk operation changed inside refresh scope")
        layer_module.chunk_mode_rule = original


class ZoologyMomentumRefreshFutureSeedMixer(
    ZoologyMomentumDeltaFutureSeedMixer
):
    """Momentum Delta with an exact post-commit residual refresh of M only."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.last_refresh_stats: Optional[dict[str, Any]] = None

    def _record_refresh(
        self,
        operation: PostCommitMomentumRefreshOperation,
    ) -> None:
        if operation.last_stats is None:
            raise RuntimeError("Momentum refresh operation was not called")
        self.last_refresh_stats = dict(operation.last_stats)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        with scoped_post_commit_momentum_refresh(
            collect_diagnostics=not self.training,
        ) as operation:
            output = super().forward(hidden_states)
        self._record_refresh(operation)
        return output

    def forward_with_state(
        self,
        hidden_states: torch.Tensor,
        *,
        initial_state: Optional[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        with scoped_post_commit_momentum_refresh(
            collect_diagnostics=not self.training,
        ) as operation:
            output, terminal_state = super().forward_with_state(
                hidden_states,
                initial_state=initial_state,
            )
        self._record_refresh(operation)
        return output, terminal_state


def momentum_refresh_diagnostics(model: torch.nn.Module) -> dict[str, Any]:
    base = momentum_futureseed_diagnostics(model)
    mixers = [
        block.sequence_mixer
        for block in model.backbone.layers
        if type(block.sequence_mixer) is ZoologyMomentumRefreshFutureSeedMixer
    ]
    rows = []
    for mixer in mixers:
        refresh = mixer.last_refresh_stats
        if refresh is not None:
            refresh = {
                key: (
                    float(value)
                    if isinstance(value, torch.Tensor)
                    else value
                )
                for key, value in refresh.items()
            }
        row = {
            "layer_idx": mixer.layer_idx,
            "refresh": refresh,
        }
        rows.append(row)
    return {
        **base,
        "refresh_layers": len(rows),
        "active_refresh_layers": sum(row["refresh"] is not None for row in rows),
        "refresh_per_layer": rows,
        "parameter_delta_vs_momentum": 0,
        "persistent_state_delta_vs_momentum": 0,
    }
