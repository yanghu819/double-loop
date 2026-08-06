from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn as nn


class FutureSeedAddressLocalUpdate(nn.Module):
    """Route a producer update independently at every recurrent address row."""

    ROW_FEATURES = 5

    def __init__(self, *, row_dim: int, col_dim: int) -> None:
        super().__init__()
        if row_dim <= 1 or col_dim <= 1:
            raise ValueError("Address-local FutureSeed needs nontrivial state axes")
        self.row_dim = int(row_dim)
        self.col_dim = int(col_dim)

        # The feature-sized hidden layer avoids a tunable router-width axis.
        self.row_gate_in = nn.Linear(self.ROW_FEATURES, self.ROW_FEATURES)
        self.row_gate_out = nn.Linear(self.ROW_FEATURES, 1, bias=False)
        nn.init.zeros_(self.row_gate_out.weight)

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
        update = terminal - incoming
        terminal_rms = self._rms(terminal, (-1, -2)).clamp_min(1e-6)
        terminal_unit = terminal / terminal_rms
        update_unit = update / terminal_rms

        terminal_row_rms = self._rms(terminal_unit, (-1,)).squeeze(-1).clamp_min(1e-6)
        update_row_rms = self._rms(update_unit, (-1,)).squeeze(-1).clamp_min(1e-6)
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
        row_gain = torch.tanh(
            self.row_gate_out(
                torch.nn.functional.silu(self.row_gate_in(row_features))
            ).squeeze(-1)
        )
        residual = row_gain.unsqueeze(-1) * update
        candidate = terminal_state + residual.to(dtype=terminal_state.dtype)

        residual_relative_rms = self._rms(residual, (-1, -2)) / terminal_rms
        update_relative_rms = self._rms(update, (-1, -2)) / terminal_rms
        return candidate, {
            "fs3_address_local_enabled": terminal_state.new_ones(()),
            "fs3_address_local_gain_abs": row_gain.detach()
            .abs()
            .mean()
            .to(dtype=terminal_state.dtype),
            "fs3_address_local_gain_row_std": row_gain.detach()
            .std(dim=-1, unbiased=False)
            .mean()
            .to(dtype=terminal_state.dtype),
            "fs3_address_local_gain_batch_std": row_gain.detach()
            .std(dim=0, unbiased=False)
            .mean()
            .to(dtype=terminal_state.dtype),
            "fs3_address_local_update_relative_rms": update_relative_rms.detach()
            .mean()
            .to(dtype=terminal_state.dtype),
            "fs3_address_local_residual_relative_rms": residual_relative_rms.detach()
            .mean()
            .to(dtype=terminal_state.dtype),
            "fs3_address_local_residual_batch_std": residual_relative_rms.detach()
            .squeeze(-1)
            .squeeze(-1)
            .std(dim=0, unbiased=False)
            .mean()
            .to(dtype=terminal_state.dtype),
        }


def address_local_update_parameter_count(row_dim: int, col_dim: int) -> int:
    del row_dim, col_dim
    return (
        FutureSeedAddressLocalUpdate.ROW_FEATURES**2
        + FutureSeedAddressLocalUpdate.ROW_FEATURES
        + FutureSeedAddressLocalUpdate.ROW_FEATURES
    )
