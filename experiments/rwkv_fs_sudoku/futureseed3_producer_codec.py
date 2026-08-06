from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn as nn


class FutureSeedProducerCodec(nn.Module):
    """Compress a producer update and decode a bounded seed-content residual."""

    ROW_FEATURES = 5
    CELL_FEATURES = 4

    def __init__(self, *, row_dim: int, col_dim: int) -> None:
        super().__init__()
        if row_dim <= 1 or col_dim <= 1:
            raise ValueError("FutureSeed producer codec needs nontrivial state axes")
        self.row_dim = int(row_dim)
        self.col_dim = int(col_dim)

        # Feature-sized hidden layers avoid a tunable codec-width axis.
        self.row_score_in = nn.Linear(self.ROW_FEATURES, self.ROW_FEATURES)
        self.row_score_out = nn.Linear(self.ROW_FEATURES, 1, bias=False)
        self.cell_decode_in = nn.Linear(self.CELL_FEATURES, self.CELL_FEATURES)
        self.cell_decode_out = nn.Linear(self.CELL_FEATURES, 1, bias=False)
        nn.init.zeros_(self.cell_decode_out.weight)

    @staticmethod
    def _rms(value: torch.Tensor, dims: tuple[int, ...]) -> torch.Tensor:
        return value.square().mean(dim=dims, keepdim=True).sqrt()

    def forward(
        self,
        terminal_state: torch.Tensor,
        producer_initial_state: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        expected = (self.row_dim, self.col_dim)
        if terminal_state.ndim != 4 or tuple(terminal_state.shape[-2:]) != expected:
            raise ValueError(
                "FutureSeed producer terminal state must end in "
                f"{expected}, got {tuple(terminal_state.shape)}"
            )
        if producer_initial_state.shape != terminal_state.shape:
            raise ValueError(
                "producer initial-state shape does not match terminal state: "
                f"{tuple(producer_initial_state.shape)} != {tuple(terminal_state.shape)}"
            )

        terminal = terminal_state.float()
        incoming = producer_initial_state.float()
        terminal_rms = self._rms(terminal, (-1, -2)).clamp_min(1e-6)
        terminal_unit = terminal / terminal_rms
        update_unit = (terminal - incoming) / terminal_rms

        terminal_row_rms = (
            self._rms(terminal_unit, (-1,)).squeeze(-1).clamp_min(1e-6)
        )
        update_row_rms = (
            self._rms(update_unit, (-1,)).squeeze(-1).clamp_min(1e-6)
        )
        row_alignment = (
            (terminal_unit * update_unit).mean(dim=-1)
            / (terminal_row_rms * update_row_rms).clamp_min(1e-6)
        ).clamp(min=-1.0, max=1.0)
        row_features = torch.stack(
            (
                terminal_unit.mean(dim=-1),
                terminal_row_rms.log().clamp(min=-6.0, max=6.0),
                update_unit.mean(dim=-1),
                update_row_rms.log().clamp(min=-6.0, max=6.0),
                row_alignment,
            ),
            dim=-1,
        )
        row_scores = self.row_score_out(
            torch.nn.functional.silu(self.row_score_in(row_features))
        ).squeeze(-1)
        row_weights = torch.softmax(row_scores, dim=-1)

        # This col_dim code is shared across layers and heads. The row scorer is
        # permutation-equivariant, so it does not assume a fixed K-axis basis.
        producer_code = torch.sum(
            row_weights.unsqueeze(-1) * update_unit,
            dim=-2,
        )
        expanded_code = producer_code.unsqueeze(-2).expand_as(update_unit)
        cell_features = torch.stack(
            (
                terminal_unit,
                update_unit,
                expanded_code,
                update_unit * expanded_code,
            ),
            dim=-1,
        )
        raw_residual = self.cell_decode_out(
            torch.nn.functional.silu(self.cell_decode_in(cell_features))
        ).squeeze(-1)
        bounded_residual = torch.tanh(raw_residual) * terminal_rms
        candidate = terminal_state + bounded_residual.to(dtype=terminal_state.dtype)

        normalized_entropy = -(
            row_weights * row_weights.clamp_min(1e-12).log()
        ).sum(dim=-1) / torch.log(row_weights.new_tensor(float(self.row_dim)))
        residual_relative_rms = self._rms(
            bounded_residual,
            (-1, -2),
        ) / terminal_rms
        update_relative_rms = self._rms(
            terminal - incoming,
            (-1, -2),
        ) / terminal_rms
        return candidate, {
            "fs3_codec_enabled": terminal_state.new_ones(()),
            "fs3_codec_code_relative_rms": producer_code.detach()
            .square()
            .mean(dim=-1)
            .sqrt()
            .mean()
            .to(dtype=terminal_state.dtype),
            "fs3_codec_row_attention_entropy": normalized_entropy.detach()
            .mean()
            .to(dtype=terminal_state.dtype),
            "fs3_codec_row_attention_max": row_weights.detach()
            .max(dim=-1)
            .values
            .mean()
            .to(dtype=terminal_state.dtype),
            "fs3_codec_row_attention_batch_std": row_weights.detach()
            .std(dim=0, unbiased=False)
            .mean()
            .to(dtype=terminal_state.dtype),
            "fs3_codec_update_relative_rms": update_relative_rms.detach()
            .mean()
            .to(dtype=terminal_state.dtype),
            "fs3_codec_residual_relative_rms": residual_relative_rms.detach()
            .mean()
            .to(dtype=terminal_state.dtype),
            "fs3_codec_residual_batch_std": residual_relative_rms.detach()
            .squeeze(-1)
            .squeeze(-1)
            .std(dim=0, unbiased=False)
            .mean()
            .to(dtype=terminal_state.dtype),
        }


def producer_codec_parameter_count(row_dim: int, col_dim: int) -> int:
    del row_dim, col_dim
    return (
        FutureSeedProducerCodec.ROW_FEATURES**2
        + FutureSeedProducerCodec.ROW_FEATURES
        + FutureSeedProducerCodec.ROW_FEATURES
        + FutureSeedProducerCodec.CELL_FEATURES**2
        + FutureSeedProducerCodec.CELL_FEATURES
        + FutureSeedProducerCodec.CELL_FEATURES
    )
