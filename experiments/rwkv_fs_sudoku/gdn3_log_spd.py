from __future__ import annotations

import math
from typing import Dict, Tuple

import torch
import torch.nn as nn


class BoundedLogSPDAddressMetric(nn.Module):
    """Volume-preserving bounded SPD geometry for one layer's GDN2 keys."""

    def __init__(self, heads: int, head_dim: int) -> None:
        super().__init__()
        self.heads = int(heads)
        self.head_dim = int(head_dim)
        indices = torch.triu_indices(self.head_dim, self.head_dim)
        keep = ~(
            (indices[0] == self.head_dim - 1)
            & (indices[1] == self.head_dim - 1)
        )
        self.register_buffer("row_indices", indices[0, keep], persistent=False)
        self.register_buffer("col_indices", indices[1, keep], persistent=False)
        self.raw = nn.Parameter(torch.zeros(self.heads, int(keep.sum().item())))

    @property
    def parameters_per_head(self) -> int:
        return self.head_dim * (self.head_dim + 1) // 2 - 1

    def generator(self) -> torch.Tensor:
        raw = self.raw.float()
        matrix = raw.new_zeros(self.heads, self.head_dim, self.head_dim)
        matrix[:, self.row_indices, self.col_indices] = raw
        off_diagonal = self.row_indices != self.col_indices
        matrix[
            :, self.col_indices[off_diagonal], self.row_indices[off_diagonal]
        ] = raw[:, off_diagonal]
        diagonal = matrix.diagonal(dim1=-2, dim2=-1)
        diagonal[:, -1] = -diagonal[:, :-1].sum(dim=-1)
        frobenius = matrix.square().sum(dim=(-1, -2), keepdim=True).sqrt()
        return matrix / (1.0 + frobenius)

    def factor(self) -> torch.Tensor:
        with torch.autocast(device_type=self.raw.device.type, enabled=False):
            return torch.matrix_exp(0.5 * math.log(2.0) * self.generator())

    def transform_pair(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        expected = (self.heads, self.head_dim)
        if query.shape[-2:] != expected or key.shape[-2:] != expected:
            raise ValueError(
                "Log-SPD Q/K shape mismatch: "
                f"query={tuple(query.shape)} key={tuple(key.shape)} expected={expected}"
            )
        factor = self.factor()
        identity = torch.eye(
            self.head_dim,
            device=factor.device,
            dtype=factor.dtype,
        )
        delta = (factor - identity).to(dtype=query.dtype)
        transformed_query = query + torch.einsum(
            "...hk,hkj->...hj", query, delta
        )
        transformed_key = key + torch.einsum(
            "...hk,hkj->...hj", key, delta.to(dtype=key.dtype)
        )
        return transformed_query, transformed_key

    @torch.no_grad()
    def diagnostics(self, storage_dtype: torch.dtype) -> Dict[str, torch.Tensor]:
        factor = self.factor()
        identity = torch.eye(
            self.head_dim,
            device=factor.device,
            dtype=factor.dtype,
        )
        applied_factor = (
            torch.eye(
                self.head_dim,
                device=factor.device,
                dtype=storage_dtype,
            )
            + (factor - identity).to(dtype=storage_dtype)
        ).float()
        metric = applied_factor.transpose(-1, -2) @ applied_factor
        eigenvalues = torch.linalg.eigvalsh(metric)
        return {
            "gdn3_log_spd_enabled": factor.new_ones(()),
            "gdn3_log_spd_raw_rms": self.raw.float().square().mean().sqrt(),
            "gdn3_log_spd_metric_delta_fro": (
                metric - identity
            ).square().sum(dim=(-1, -2)).sqrt().mean(),
            "gdn3_log_spd_eigenvalue_min": eigenvalues.amin(),
            "gdn3_log_spd_eigenvalue_max": eigenvalues.amax(),
            "gdn3_log_spd_condition_max": (
                eigenvalues[..., -1] / eigenvalues[..., 0]
            ).amax(),
            "gdn3_log_spd_logdet_abs_max": torch.linalg.slogdet(
                metric
            ).logabsdet.abs().amax(),
        }
