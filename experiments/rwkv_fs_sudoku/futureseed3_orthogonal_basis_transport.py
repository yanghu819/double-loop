from __future__ import annotations

import math
from typing import Dict, Tuple

import torch
import torch.nn as nn


class FutureSeedOrthogonalBasisTransport(nn.Module):
    """Rotate a recurrent state into the receiving layer's K/V coordinates."""

    def __init__(
        self,
        *,
        edges: int,
        heads: int,
        row_dim: int,
        col_dim: int,
    ) -> None:
        super().__init__()
        if edges <= 0 or heads <= 0:
            raise ValueError("Basis transport needs at least one edge and head")
        if row_dim <= 1 or col_dim <= 1:
            raise ValueError("Basis transport needs nontrivial state axes")
        self.edges = int(edges)
        self.heads = int(heads)
        self.row_dim = int(row_dim)
        self.col_dim = int(col_dim)

        row_i, row_j = torch.triu_indices(self.row_dim, self.row_dim, offset=1)
        col_i, col_j = torch.triu_indices(self.col_dim, self.col_dim, offset=1)
        self.register_buffer("row_i", row_i, persistent=False)
        self.register_buffer("row_j", row_j, persistent=False)
        self.register_buffer("col_i", col_i, persistent=False)
        self.register_buffer("col_j", col_j, persistent=False)
        self.row_angles = nn.Parameter(
            torch.zeros(self.edges, self.heads, row_i.numel())
        )
        self.col_angles = nn.Parameter(
            torch.zeros(self.edges, self.heads, col_i.numel())
        )

    @staticmethod
    def _rms(value: torch.Tensor, dims: tuple[int, ...]) -> torch.Tensor:
        return value.square().mean(dim=dims, keepdim=True).sqrt()

    @staticmethod
    def _rotation(
        angles: torch.Tensor,
        *,
        dim: int,
        row_i: torch.Tensor,
        row_j: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        skew = angles.new_zeros(angles.shape[0], dim, dim)
        skew[:, row_i, row_j] = angles
        skew[:, row_j, row_i] = -angles
        identity = torch.eye(dim, device=angles.device, dtype=angles.dtype).expand(
            angles.shape[0], dim, dim
        )
        half_skew = 0.5 * skew
        rotation = torch.linalg.solve(identity - half_skew, identity + half_skew)
        delta = rotation - identity
        orthogonality_error = (
            rotation.transpose(-1, -2) @ rotation - identity
        ).abs().amax(dim=(-1, -2))
        relative_rms = delta.square().sum(dim=(-1, -2)).sqrt() / math.sqrt(dim)
        return rotation, relative_rms, orthogonality_error

    def forward(
        self,
        terminal_state: torch.Tensor,
        *,
        edge_idx: int,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        expected = (self.heads, self.row_dim, self.col_dim)
        if terminal_state.ndim != 4 or tuple(terminal_state.shape[-3:]) != expected:
            raise ValueError(
                "FutureSeed terminal state must end in "
                f"{expected}, got {tuple(terminal_state.shape)}"
            )
        if not 0 <= edge_idx < self.edges:
            raise IndexError(f"FutureSeed edge index {edge_idx} is out of range")

        with torch.autocast(device_type=terminal_state.device.type, enabled=False):
            state = terminal_state.float()
            row_rotation, row_rotation_rms, row_orthogonality_error = (
                self._rotation(
                    self.row_angles[edge_idx].float(),
                    dim=self.row_dim,
                    row_i=self.row_i,
                    row_j=self.row_j,
                )
            )
            col_rotation, col_rotation_rms, col_orthogonality_error = (
                self._rotation(
                    self.col_angles[edge_idx].float(),
                    dim=self.col_dim,
                    row_i=self.col_i,
                    row_j=self.col_j,
                )
            )
            row_delta = row_rotation - torch.eye(
                self.row_dim, device=state.device, dtype=state.dtype
            )
            col_delta = col_rotation - torch.eye(
                self.col_dim, device=state.device, dtype=state.dtype
            )

            left_residual = torch.einsum("hkr,bhrv->bhkv", row_delta, state)
            row_mapped = state + left_residual
            right_residual = torch.einsum(
                "bhkv,hvw->bhkw", row_mapped, col_delta.transpose(-1, -2)
            )
            residual = left_residual + right_residual
            transformed = state + residual

            state_rms = self._rms(state, (-1, -2)).clamp_min(1e-6)
            transformed_rms = self._rms(transformed, (-1, -2))
            residual_relative_rms = self._rms(residual, (-1, -2)) / state_rms
            fp32_norm_error = (transformed_rms / state_rms - 1.0).abs()

        candidate = terminal_state + residual.to(dtype=terminal_state.dtype)
        candidate_rms = self._rms(candidate.float(), (-1, -2))
        storage_norm_error = (candidate_rms / state_rms - 1.0).abs()
        angle_abs = torch.cat(
            (self.row_angles[edge_idx], self.col_angles[edge_idx]), dim=-1
        ).float().abs().mean()
        return candidate, {
            "fs3_basis_transport_enabled": terminal_state.new_ones(()),
            "fs3_basis_transport_angle_abs": angle_abs.detach().to(
                dtype=terminal_state.dtype
            ),
            "fs3_basis_transport_row_rotation_relative_rms": (
                row_rotation_rms.detach().mean().to(dtype=terminal_state.dtype)
            ),
            "fs3_basis_transport_col_rotation_relative_rms": (
                col_rotation_rms.detach().mean().to(dtype=terminal_state.dtype)
            ),
            "fs3_basis_transport_state_residual_relative_rms": (
                residual_relative_rms.detach().mean().to(dtype=terminal_state.dtype)
            ),
            "fs3_basis_transport_residual_batch_std": (
                residual_relative_rms.detach()
                .squeeze(-1)
                .squeeze(-1)
                .std(dim=0, unbiased=False)
                .mean()
                .to(dtype=terminal_state.dtype)
            ),
            "fs3_basis_transport_residual_head_std": (
                residual_relative_rms.detach()
                .squeeze(-1)
                .squeeze(-1)
                .std(dim=1, unbiased=False)
                .mean()
                .to(dtype=terminal_state.dtype)
            ),
            "fs3_basis_transport_fp32_norm_max_error": (
                fp32_norm_error.detach().max().to(dtype=terminal_state.dtype)
            ),
            "fs3_basis_transport_storage_norm_max_error": (
                storage_norm_error.detach().max().to(dtype=terminal_state.dtype)
            ),
            "fs3_basis_transport_orthogonality_max_error": (
                torch.maximum(
                    row_orthogonality_error.max(),
                    col_orthogonality_error.max(),
                )
                .detach()
                .to(dtype=terminal_state.dtype)
            ),
        }


def basis_transport_parameter_count(
    *, edges: int, heads: int, row_dim: int, col_dim: int
) -> int:
    row_pairs = row_dim * (row_dim - 1) // 2
    col_pairs = col_dim * (col_dim - 1) // 2
    return int(edges * heads * (row_pairs + col_pairs))
