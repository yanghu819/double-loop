#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import hashlib
import html
import importlib.metadata
import json
import math
import os
import random
import subprocess
import time
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as torch_checkpoint


FUTURE_SEED_GRADIENT_MODES = ("canonical", "opening_projection")

from futureseed2_selective import (
    FUTURE_SEED_GATE_MODES,
    FutureSeedSelectiveGate,
)
from futureseed3_address_local_update import FutureSeedAddressLocalUpdate
from futureseed3_orthogonal_basis_transport import (
    FutureSeedOrthogonalBasisTransport,
)
from futureseed3_producer_codec import FutureSeedProducerCodec
from gdn3_log_spd import BoundedLogSPDAddressMetric
from fast_slow_decay_gdn2 import (
    FAST_SLOW_DECAY_MODES,
    FastSlowDecayController,
)
from gain_budget_gdn2 import fla_l2norm_fp32, project_erase_gate
from momentum_delta_sudoku import (
    MomentumDeltaTimeMix,
    load_external_momentum_layer,
    momentum_source_summary,
)
from preconditioned_gdn2 import (
    GDN2_PRECONDITION_MODES,
    causal_tied_atk_preconditioner,
    fold_preconditioned_write_into_gdn2,
    futureseed_row_precision,
    normalize_qk_fp32,
)

try:
    from rwkv7_cuda import StatePassingRWKV7, WindRWKV7, statepassing_available, wind_available
except Exception:  # pragma: no cover - CUDA extension is optional for CPU smoke.
    StatePassingRWKV7 = None
    WindRWKV7 = None

    def statepassing_available(head_dim: int) -> Tuple[bool, str]:
        return False, "rwkv7_cuda import failed"

    def wind_available(head_dim: int) -> Tuple[bool, str]:
        return False, "rwkv7_cuda import failed"

try:
    from fla.ops.gated_delta_rule.chunk import chunk_gated_delta_rule
    from fla.ops.gated_delta_rule.naive import naive_recurrent_gated_delta_rule
except Exception as exc:  # pragma: no cover - optional CUDA/Triton dependency.
    chunk_gated_delta_rule = None
    naive_recurrent_gated_delta_rule = None
    FLA_IMPORT_ERROR = exc
else:
    FLA_IMPORT_ERROR = None

try:
    from fla.layers.gated_deltanet import GatedDeltaNet as FLAGatedDeltaNet
    from fla.layers.gdn2 import GatedDeltaNet2
    from fla.layers.kda import KimiDeltaAttention
    from fla.models.utils import Cache as FLACache
    from fla.ops.gdn2 import chunk_gdn2, fused_recurrent_gdn2
    from fla.ops.kda import chunk_kda
except Exception as exc:  # pragma: no cover - optional CUDA/Triton dependency.
    FLAGatedDeltaNet = None
    GatedDeltaNet2 = None
    KimiDeltaAttention = None
    FLACache = None
    chunk_gdn2 = None
    fused_recurrent_gdn2 = None
    chunk_kda = None
    FLA_DELTA_IMPORT_ERROR = exc
else:
    FLA_DELTA_IMPORT_ERROR = None

try:
    from fla.layers.raven import Raven as FLARaven
    from fla.ops.gsa import chunk_gsa
except Exception as exc:  # pragma: no cover - Raven requires a newer pinned FLA tree.
    FLARaven = None
    chunk_gsa = None
    FLA_RAVEN_IMPORT_ERROR = exc
else:
    FLA_RAVEN_IMPORT_ERROR = None

try:
    from gdn_triton import gdn_triton_recurrent
except Exception as exc:  # pragma: no cover - optional CUDA/Triton dependency.
    gdn_triton_recurrent = None
    GDN_TRITON_IMPORT_ERROR = exc
else:
    GDN_TRITON_IMPORT_ERROR = None


def load_fla_short_convolution() -> type:
    from fla.modules.convolution import ShortConvolution

    return ShortConvolution


def fla_gdn_available() -> Tuple[bool, str]:
    if chunk_gated_delta_rule is None or naive_recurrent_gated_delta_rule is None:
        return False, f"flash-linear-attention import failed: {FLA_IMPORT_ERROR}"
    return True, "ok"


def fla_delta_available(backbone: str) -> Tuple[bool, str]:
    if backbone == "momentum":
        try:
            load_external_momentum_layer()
        except Exception as exc:
            return False, f"pinned Momentum DeltaNet import failed: {exc}"
        return True, "ok"
    implementations = {
        "fla_gdn": (FLAGatedDeltaNet, chunk_gated_delta_rule),
        "gdn2": (GatedDeltaNet2, chunk_gdn2),
        "kda": (KimiDeltaAttention, chunk_kda),
        "raven": (FLARaven, chunk_gsa),
    }
    if backbone not in implementations:
        return False, f"unknown FLA delta backbone: {backbone}"
    layer, kernel = implementations[backbone]
    if layer is None or kernel is None:
        import_error = (
            FLA_RAVEN_IMPORT_ERROR if backbone == "raven" else FLA_DELTA_IMPORT_ERROR
        )
        return False, f"flash-linear-attention import failed: {import_error}"
    return True, "ok"


def gdn_triton_available() -> Tuple[bool, str]:
    if gdn_triton_recurrent is None:
        return False, f"local GDN Triton import failed: {GDN_TRITON_IMPORT_ERROR}"
    return True, "ok"


def apply_anchor_rotary_address(
    tensor: torch.Tensor,
    anchor: torch.Tensor,
    head_scale: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Bind a stable address to Q/K with a norm-preserving pairwise rotation."""
    if tensor.shape != anchor.shape:
        raise ValueError(
            f"Address rotation shape mismatch: tensor={tuple(tensor.shape)} "
            f"anchor={tuple(anchor.shape)}"
        )
    if tensor.ndim != 4 or tensor.shape[-1] % 2:
        raise ValueError("Address rotation expects [B, T, H, even K]")
    if head_scale.shape != (tensor.shape[-2],):
        raise ValueError(
            f"Address rotation scale {tuple(head_scale.shape)} does not match "
            f"{tensor.shape[-2]} heads"
        )

    pair_shape = (*tensor.shape[:-1], tensor.shape[-1] // 2, 2)
    tensor_pair = tensor.float().reshape(pair_shape)
    anchor_pair = anchor.float().reshape(pair_shape)
    phase_source = math.pi * torch.tanh(anchor_pair.mean(dim=-1))
    phase = phase_source * torch.tanh(head_scale.float()).view(1, 1, -1, 1)
    cosine = torch.cos(phase)
    sine = torch.sin(phase)
    first, second = tensor_pair.unbind(dim=-1)
    rotated = torch.stack(
        (first * cosine - second * sine, first * sine + second * cosine),
        dim=-1,
    ).reshape_as(tensor)
    return rotated.to(dtype=tensor.dtype), phase


def apply_address_phase_rotation(
    tensor: torch.Tensor,
    phase: torch.Tensor,
) -> torch.Tensor:
    """Apply an address-specific orthogonal rotation to paired Q/K channels."""
    if tensor.ndim != 4 or tensor.shape[-1] % 2:
        raise ValueError("Address phase rotation expects [B, T, H, even K]")
    expected_phase_shape = (*tensor.shape[:-1], tensor.shape[-1] // 2)
    if tuple(phase.shape) != expected_phase_shape:
        raise ValueError(
            f"Address phase shape {tuple(phase.shape)} does not match "
            f"{expected_phase_shape}"
        )

    tensor_pair = tensor.float().reshape(*tensor.shape[:-1], tensor.shape[-1] // 2, 2)
    phase_f = phase.float()
    cosine = torch.cos(phase_f)
    sine = torch.sin(phase_f)
    first, second = tensor_pair.unbind(dim=-1)
    rotated = torch.stack(
        (first * cosine - second * sine, first * sine + second * cosine),
        dim=-1,
    ).reshape_as(tensor)
    return rotated.to(dtype=tensor.dtype)


def fold_gdn2_write_carrier_into_official_inputs(
    k: torch.Tensor,
    b: torch.Tensor,
    w: torch.Tensor,
    carrier: torch.Tensor,
    *,
    eps: float = 1e-6,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Express an independent K-row write carrier through the official GDN2 op.

    The official kernel L2-normalizes K, then uses the same unit key for erase
    and rank-one write. For c in (0, 1], let r = ||c * k|| / ||k||. Passing
    k'=c*k, b'=r^2*b/c, and w'=r*w makes the normalized official recurrence
    exactly equal to writing with c*k_unit while keeping erase at b*k_unit.
    """
    if k.shape != b.shape or k.shape != carrier.shape:
        raise ValueError(
            "GDN2 carrier expects matching K-axis tensors: "
            f"k={tuple(k.shape)} b={tuple(b.shape)} carrier={tuple(carrier.shape)}"
        )
    if w.shape[:-1] != k.shape[:-1]:
        raise ValueError(
            "GDN2 carrier expects W to share batch/time/head axes with K: "
            f"k={tuple(k.shape)} w={tuple(w.shape)}"
        )
    k_float = k.float()
    carrier_float = carrier.float().clamp(min=eps, max=1.0)
    carried_k_float = carrier_float * k_float
    base_norm = k_float.norm(dim=-1, keepdim=True).clamp_min(eps)
    carried_norm = carried_k_float.norm(dim=-1, keepdim=True).clamp_min(eps)
    write_norm_ratio = carried_norm / base_norm
    b_effective = (
        b.float() * write_norm_ratio.square() / carrier_float
    ).to(dtype=b.dtype)
    w_effective = (
        w.float() * write_norm_ratio
    ).to(dtype=w.dtype)
    return (
        carried_k_float.to(dtype=k.dtype),
        b_effective,
        w_effective,
        write_norm_ratio,
    )


N = 9
BOX_ROWS = 3
BOX_COLS = 3
CELLS = 81
BLANK = 9
VOCAB = 10
FAST_SLOW_TRAIN_KEYS = (
    "gdn2_fast_slow_enabled",
    "gdn2_fast_slow_rho_mean",
    "gdn2_fast_slow_current_weight",
    "gdn2_fast_slow_lag_mass",
    "gdn2_fast_slow_raw_hazard_mean",
    "gdn2_fast_slow_effective_hazard_mean",
    "gdn2_fast_slow_raw_tv",
    "gdn2_fast_slow_effective_tv",
    "gdn2_fast_slow_tv_ratio",
    "gdn2_fast_slow_relative_change",
    "gdn2_fast_slow_alpha_mean",
)
ADDRESS_TRAIN_KEYS = (
    "gdn2_address_enabled",
    "gdn2_address_rotation_scale_abs",
    "gdn2_address_phase_abs",
    "gdn2_address_q_relative_change",
    "gdn2_address_k_relative_change",
    "gdn2_address_q_norm_error",
    "gdn2_address_k_norm_error",
    "gdn2_address_phase_weight_rms",
    "gdn2_address_phase_token_std",
    "gdn2_address_phase_plane_std",
    "gdn2_address_residual_weight_rms",
    "gdn2_address_residual_rms",
    "gdn2_address_residual_token_std",
    "gdn2_address_residual_q_ratio",
    "gdn2_address_residual_k_ratio",
    "gdn2_address_q_residual_weight_rms",
    "gdn2_address_k_residual_weight_rms",
    "gdn2_address_q_residual_rms",
    "gdn2_address_k_residual_rms",
    "gdn2_address_terminal_state_rms",
    "gdn2_address_carrier_mean",
    "gdn2_address_carrier_std",
    "gdn2_address_carrier_token_std",
    "gdn2_address_carrier_min",
    "gdn2_address_carrier_below_095_frac",
    "gdn2_address_carrier_scale_rms",
    "gdn2_address_carrier_bias_delta_rms",
    "gdn2_address_carrier_write_norm_ratio",
    "gdn2_address_carrier_b_input_relative_change",
    "gdn2_address_carrier_w_input_relative_change",
    "gdn3_shared_address_weight_rms",
    "gdn3_shared_address_residual_rms",
    "gdn3_shared_address_token_std",
    "gdn3_log_spd_enabled",
    "gdn3_log_spd_raw_rms",
    "gdn3_log_spd_metric_delta_fro",
    "gdn3_log_spd_eigenvalue_min",
    "gdn3_log_spd_eigenvalue_max",
    "gdn3_log_spd_condition_max",
    "gdn3_log_spd_logdet_abs_max",
)
PRECONDITION_TRAIN_KEYS = (
    "gdn2_precondition_enabled",
    "gdn2_precondition_multiplier_mean",
    "gdn2_precondition_multiplier_std",
    "gdn2_precondition_multiplier_min",
    "gdn2_precondition_multiplier_max",
    "gdn2_precondition_log_precision_mean",
    "gdn2_precondition_log_precision_std",
    "gdn2_precondition_seed_precision_mean",
    "gdn2_precondition_seed_precision_std",
    "gdn2_precondition_write_relative_change",
    "gdn2_precondition_erase_error_max",
    "gdn2_precondition_erase_error_rms",
)
COHERENT_DELTA_TRAIN_KEYS = (
    "gdn2_coherent_delta_enabled",
    "gdn2_coherent_delta_mix_mean",
    "gdn2_coherent_delta_mix_abs",
    "gdn2_coherent_delta_mix_min",
    "gdn2_coherent_delta_mix_max",
    "gdn2_coherent_delta_target_mean",
    "gdn2_coherent_delta_pre_gap_rms",
    "gdn2_coherent_delta_post_gap_rms",
    "gdn2_coherent_delta_gap_ratio",
    "gdn2_coherent_delta_b_relative_change",
    "gdn2_coherent_delta_w_relative_change",
)
STATE_FEEDBACK_TRAIN_KEYS = (
    "gdn3_state_feedback_enabled",
    "gdn3_state_feedback_read_rms",
    "gdn3_state_feedback_read_batch_std",
    "gdn3_state_feedback_hidden_rms",
    "gdn3_state_feedback_residual_relative_rms",
    "gdn3_state_feedback_residual_token_std",
    "gdn3_state_feedback_k_relative_change",
    "gdn3_state_feedback_v_relative_change",
    "gdn3_state_feedback_b_relative_change",
    "gdn3_state_feedback_w_relative_change",
    "gdn3_state_feedback_in_weight_rms",
    "gdn3_state_feedback_out_weight_rms",
)
TERMINAL_CONSOLIDATION_TRAIN_KEYS = (
    "gdn3_terminal_consolidation_enabled",
    "gdn3_terminal_consolidation_k_relative_rms",
    "gdn3_terminal_consolidation_k_batch_std",
    "gdn3_terminal_consolidation_k_token_std",
    "gdn3_terminal_consolidation_state_residual_relative_rms",
    "gdn3_terminal_consolidation_state_residual_batch_std",
    "gdn3_terminal_consolidation_output_rms",
    "gdn3_terminal_consolidation_output_token_std",
    "gdn3_terminal_consolidation_weight_rms",
)
ORTHOGONAL_CHUNK_STATE_TRAIN_KEYS = (
    "gdn3_orthogonal_chunk_state_enabled",
    "gdn3_orthogonal_chunk_state_angle_abs",
    "gdn3_orthogonal_chunk_state_angle_batch_std",
    "gdn3_orthogonal_chunk_state_angle_head_std",
    "gdn3_orthogonal_chunk_state_plane_dot_abs_max",
    "gdn3_orthogonal_chunk_state_plane_norm_error_max",
    "gdn3_orthogonal_chunk_state_state_residual_relative_rms",
    "gdn3_orthogonal_chunk_state_state_residual_batch_std",
    "gdn3_orthogonal_chunk_state_boundary_norm_ratio_mean",
    "gdn3_orthogonal_chunk_state_boundary_norm_ratio_max_error",
    "gdn3_orthogonal_chunk_state_read_residual_relative_rms",
    "gdn3_orthogonal_chunk_state_read_residual_batch_std",
    "gdn3_orthogonal_chunk_state_terminal_rms",
    "gdn3_orthogonal_chunk_state_terminal_batch_std",
    "gdn3_orthogonal_chunk_state_angle_weight_rms",
    "gdn3_orthogonal_chunk_state_plane_weight_rms",
)
ORTHOGONAL_HEAD_WRITE_TRAIN_KEYS = (
    "gdn3_orthogonal_head_write_enabled",
    "gdn3_orthogonal_head_write_angle_abs",
    "gdn3_orthogonal_head_write_angle_batch_std",
    "gdn3_orthogonal_head_write_angle_token_std",
    "gdn3_orthogonal_head_write_v_residual_relative_rms",
    "gdn3_orthogonal_head_write_v_residual_batch_std",
    "gdn3_orthogonal_head_write_v_residual_token_std",
    "gdn3_orthogonal_head_write_plane_dot_abs_max",
    "gdn3_orthogonal_head_write_plane_norm_error_max",
    "gdn3_orthogonal_head_write_fp32_norm_ratio_mean",
    "gdn3_orthogonal_head_write_fp32_norm_ratio_max_error",
    "gdn3_orthogonal_head_write_storage_norm_ratio_mean",
    "gdn3_orthogonal_head_write_storage_norm_ratio_max_error",
    "gdn3_orthogonal_head_write_terminal_rms",
    "gdn3_orthogonal_head_write_terminal_batch_std",
    "gdn3_orthogonal_head_write_angle_weight_rms",
    "gdn3_orthogonal_head_write_plane_weight_rms",
)
ADAPTIVE_SIGNED_ERASE_TRAIN_KEYS = (
    "gdn3_adaptive_signed_erase_enabled",
    "gdn3_adaptive_signed_erase_residual_abs",
    "gdn3_adaptive_signed_erase_residual_relative_rms",
    "gdn3_adaptive_signed_erase_residual_batch_std",
    "gdn3_adaptive_signed_erase_residual_token_std",
    "gdn3_adaptive_signed_erase_residual_head_std",
    "gdn3_adaptive_signed_erase_b_relative_change",
    "gdn3_adaptive_signed_erase_effective_mean",
    "gdn3_adaptive_signed_erase_effective_min",
    "gdn3_adaptive_signed_erase_effective_max",
    "gdn3_adaptive_signed_erase_above_one_frac",
    "gdn3_adaptive_signed_erase_clipped_low_frac",
    "gdn3_adaptive_signed_erase_clipped_high_frac",
    "gdn3_adaptive_signed_erase_terminal_rms",
    "gdn3_adaptive_signed_erase_terminal_batch_std",
    "gdn3_adaptive_signed_erase_weight_rms",
)
BI_AXIS_VALUE_DECAY_TRAIN_KEYS = (
    "gdn3_bi_axis_value_decay_enabled",
    "gdn3_bi_axis_value_decay_potential_abs",
    "gdn3_bi_axis_value_decay_common_k_log_decay_abs",
    "gdn3_bi_axis_value_decay_log_decay_abs",
    "gdn3_bi_axis_value_decay_active_frac",
    "gdn3_bi_axis_value_decay_group_std",
    "gdn3_bi_axis_value_decay_batch_std",
    "gdn3_bi_axis_value_decay_token_std",
    "gdn3_bi_axis_value_decay_cumulative_scale_min",
    "gdn3_bi_axis_value_decay_cumulative_scale_mean",
    "gdn3_bi_axis_value_decay_cumulative_scale_max",
    "gdn3_bi_axis_value_decay_inverse_scale_max",
    "gdn3_bi_axis_value_decay_write_frame_relative_rms",
    "gdn3_bi_axis_value_decay_output_restore_relative_rms",
    "gdn3_bi_axis_value_decay_state_restore_relative_rms",
    "gdn3_bi_axis_value_decay_terminal_rms",
    "gdn3_bi_axis_value_decay_terminal_batch_std",
    "gdn3_bi_axis_value_decay_weight_rms",
)
RAVEN_ROUTED_GDN_TRAIN_KEYS = (
    "gdn3_raven_routed_enabled",
    "gdn3_raven_routed_allocation_abs_from_one",
    "gdn3_raven_routed_allocation_slot_std",
    "gdn3_raven_routed_allocation_batch_std",
    "gdn3_raven_routed_allocation_token_std",
    "gdn3_raven_routed_allocation_entropy_normalized",
    "gdn3_raven_routed_allocation_min",
    "gdn3_raven_routed_allocation_max",
    "gdn3_raven_routed_k_relative_change",
    "gdn3_raven_routed_g_relative_change",
    "gdn3_raven_routed_router_weight_rms",
    "gdn3_raven_routed_terminal_rms",
    "gdn3_raven_routed_terminal_batch_std",
)
PAIRED_ADDRESS_BANK_TRAIN_KEYS = (
    "gdn3_paired_address_bank_enabled",
    "gdn3_paired_address_bank_read_gate_abs",
    "gdn3_paired_address_bank_read_gate_head_std",
    "gdn3_paired_address_bank_q_residual_relative_rms",
    "gdn3_paired_address_bank_q_residual_batch_std",
    "gdn3_paired_address_bank_q_residual_token_std",
    "gdn3_paired_address_bank_q_residual_head_std",
    "gdn3_paired_address_bank_k_residual_relative_rms",
    "gdn3_paired_address_bank_k_residual_batch_std",
    "gdn3_paired_address_bank_k_residual_token_std",
    "gdn3_paired_address_bank_k_residual_head_std",
    "gdn3_paired_address_bank_state_residual_relative_rms",
    "gdn3_paired_address_bank_state_residual_batch_std",
    "gdn3_paired_address_bank_state_residual_head_std",
    "gdn3_paired_address_bank_base_address_contrast",
    "gdn3_paired_address_bank_companion_address_contrast",
    "gdn3_paired_address_bank_output_cosine",
    "gdn3_paired_address_bank_terminal_rms",
    "gdn3_paired_address_bank_terminal_batch_std",
    "gdn3_paired_address_bank_q_weight_rms",
    "gdn3_paired_address_bank_k_weight_rms",
)
COUPLED_ADDRESS_ROWS_TRAIN_KEYS = (
    "gdn3_coupled_address_rows_enabled",
    "gdn3_coupled_address_rows_k_weight_rms",
    "gdn3_coupled_address_rows_k_residual_relative_rms",
    "gdn3_coupled_address_rows_k_residual_batch_std",
    "gdn3_coupled_address_rows_k_residual_token_std",
    "gdn3_coupled_address_rows_k_residual_head_std",
    "gdn3_coupled_address_rows_k_norm_max",
    "gdn3_coupled_address_rows_extra_state_relative_rms",
    "gdn3_coupled_address_rows_extra_state_batch_std",
    "gdn3_coupled_address_rows_extra_state_head_std",
    "gdn3_coupled_address_rows_base_state_rms",
    "gdn3_coupled_address_rows_extra_state_rms",
    "gdn3_coupled_address_rows_terminal_rms",
    "gdn3_coupled_address_rows_terminal_batch_std",
)
INTERLEAVED_WRITE_TRAIN_KEYS = (
    "gdn3_interleaved_write_enabled",
    "gdn3_interleaved_write_k_residual_rms",
    "gdn3_interleaved_write_k_residual_relative_rms",
    "gdn3_interleaved_write_k_batch_std",
    "gdn3_interleaved_write_k_token_std",
    "gdn3_interleaved_write_v_rms",
    "gdn3_interleaved_write_v_relative_rms",
    "gdn3_interleaved_write_v_batch_std",
    "gdn3_interleaved_write_v_token_std",
    "gdn3_interleaved_write_state_write_relative_rms",
    "gdn3_interleaved_write_state_write_batch_std",
    "gdn3_interleaved_write_terminal_rms",
    "gdn3_interleaved_write_terminal_batch_std",
    "gdn3_interleaved_write_k_weight_rms",
    "gdn3_interleaved_write_v_weight_rms",
)
DUAL_STATE_EXPERT_TRAIN_KEYS = (
    "gdn3_state_expert_enabled",
    "gdn3_state_expert_residual_relative_rms",
    "gdn3_state_expert_residual_batch_std",
    "gdn3_state_expert_terminal_rms",
    "gdn3_state_expert_terminal_batch_std",
    "gdn3_state_expert_seed_rms",
    "gdn3_state_expert_gate_mean",
    "gdn3_state_expert_up_weight_rms",
    "gdn3_state_expert_address_contrast",
    "gdn3_state_expert_output_cosine",
)
RAVEN_WRITE_CONTROL_TRAIN_KEYS = (
    "gdn3_raven_write_control_enabled",
    "gdn3_raven_write_control_v_residual_relative_rms",
    "gdn3_raven_write_control_v_residual_relative_rms_min",
    "gdn3_raven_write_control_v_residual_relative_rms_max",
    "gdn3_raven_write_control_v_residual_batch_std",
    "gdn3_raven_write_control_v_residual_token_std",
    "gdn3_raven_write_control_output_rms",
    "gdn3_raven_write_control_output_batch_std",
    "gdn3_raven_write_control_terminal_rms",
    "gdn3_raven_write_control_terminal_batch_std",
    "gdn3_raven_write_control_seed_rms",
    "gdn3_raven_write_control_seed_gate_mean",
    "gdn3_raven_write_control_seed_rms_receiving_min",
    "gdn3_raven_write_control_incoming_path_count_sum",
    "gdn3_raven_write_control_slot_entropy_normalized",
    "gdn3_raven_write_control_slot_entropy_normalized_min",
    "gdn3_raven_write_control_slot_max_mass_share",
    "gdn3_raven_write_control_slot_max_mass_share_max",
    "gdn3_raven_write_control_main_terminal_rms",
    "gdn3_raven_write_control_main_terminal_rms_max",
    "gdn3_raven_write_control_main_terminal_batch_std",
    "gdn3_raven_write_control_adapter_weight_rms",
)
STATE_EXPERT_TRAIN_KEYS = (
    DUAL_STATE_EXPERT_TRAIN_KEYS + RAVEN_WRITE_CONTROL_TRAIN_KEYS
)
ROWS: List[List[int]] = []
COLS: List[List[int]] = []
BOXES: List[List[int]] = []
UNITS: List[List[int]] = []
BACKBONE_DISPLAY_NAMES = {
    "rwkv": "RWKV-style (deprecated)",
    "rwkv7": "RWKV7 TimeMix (official contract)",
    "gdn": "GDN-Triton",
    "fla_gdn": "GDN (official FLA)",
    "gdn2": "GDN2 (official FLA)",
    "kda": "KDA (official FLA)",
    "raven": "Raven (official FLA)",
    "momentum": "Momentum DeltaNet (pinned external FLA)",
}
RWKV7_OFFICIAL_SOURCE_COMMIT = "952102498e9ed367ea0a59ee64106916d474d30f"
RWKV7_OFFICIAL_SOURCE_BLOB = "b4d167fedead2655d253c55eb47b65f00e7193d2"
RWKV7_OFFICIAL_SOURCE_PATH = "RWKV-v7/train_temp/src/model.py"
RWKV7_OFFICIAL_KERNEL_BLOB = "827faeb06b9d2b6e31b3efe85af6d3ae4cf88905"
RWKV7_OFFICIAL_KERNEL_PATH = "RWKV-v7/train_temp/cuda/rwkv7_clampw.cu"
RWKV7_STATEPASSING_CUDA_SHA256 = "59a90a0521b1851da17c008c685f959d586af1a7d28056b29a7478ab92c1c892"
GAIN_BUDGET_FLA_SHA = "9c8e42e762fce087c27b673af4922795d9edb85e"
RAVEN_FLA_SHA = "31d15f7554bd5df05d3da6f75e09146279d2b1a8"
ALLOWED_FLA_SOURCE_SHAS = frozenset((GAIN_BUDGET_FLA_SHA, RAVEN_FLA_SHA))
GAIN_BUDGET_WHEEL_SHA256 = "0280db310981915eb048ece99d7bedca8b5caa9be65c99835a0f912ada977d6a"
GDN2_ADDRESS_MODES = (
    "none",
    "position_qk",
    "anchor_rotary",
    "anchor_phase",
    "anchor_residual",
    "shared_namespace",
    "anchor_qk_residual",
    "anchor_carrier",
)
GDN2_CROSS_LAYER_INIT_MODES = ("independent", "coherent_qkv")
GDN2_UPDATE_MODES = (
    "none",
    "log_spd_metric",
    "coherent_delta",
    "state_feedback",
    "terminal_consolidation",
    "orthogonal_chunk_state",
    "orthogonal_head_write",
    "adaptive_signed_erase",
    "bi_axis_value_decay",
    "gauge_balanced_bi_axis",
    "raven_routed_gdn",
    "paired_address_bank",
    "coupled_address_rows",
    "interleaved_write",
)
GDN2_STATE_EXPERT_MODES = ("none", "dual_state", "raven_write_control")
FUTURE_SEED_CONTENT_MODES = (
    "terminal",
    "innovation_residual",
    "producer_codec",
    "address_local_update",
    "orthogonal_basis_transport",
)
CELL_ORDER_TRAIN_MODES = ("row_major", "random")


def configure_sudoku(size: int, box_rows: int, box_cols: int) -> None:
    global N, BOX_ROWS, BOX_COLS, CELLS, BLANK, VOCAB, ROWS, COLS, BOXES, UNITS
    if box_rows <= 0 or box_cols <= 0:
        inferred = {4: (2, 2), 6: (2, 3), 9: (3, 3), 12: (3, 4), 16: (4, 4), 25: (5, 5)}
        if size not in inferred:
            raise ValueError("Use a supported Sudoku size, or pass --box_rows and --box_cols.")
        box_rows, box_cols = inferred[size]
    if box_rows <= 1 or box_cols <= 1 or box_rows * box_cols != size:
        raise ValueError("Sudoku needs size == box_rows * box_cols with both box dimensions > 1.")

    N = int(size)
    BOX_ROWS = int(box_rows)
    BOX_COLS = int(box_cols)
    CELLS = N * N
    BLANK = N
    VOCAB = N + 1
    ROWS = [[r * N + c for c in range(N)] for r in range(N)]
    COLS = [[r * N + c for r in range(N)] for c in range(N)]
    BOXES = [
        [(br + dr) * N + (bc + dc) for dr in range(BOX_ROWS) for dc in range(BOX_COLS)]
        for br in range(0, N, BOX_ROWS)
        for bc in range(0, N, BOX_COLS)
    ]
    UNITS = ROWS + COLS + BOXES


def normalize_cell_order(
    cell_order: Optional[torch.Tensor],
    *,
    cells: int,
    device: torch.device,
) -> Optional[torch.Tensor]:
    if cell_order is None:
        return None
    order = cell_order.to(device=device, dtype=torch.long)
    if order.ndim != 1 or order.numel() != cells:
        raise ValueError(
            f"cell_order must be a 1D permutation of {cells} cells, got {tuple(order.shape)}"
        )
    expected = torch.arange(cells, device=device)
    if not torch.equal(order.sort().values, expected):
        raise ValueError("cell_order must contain every canonical cell exactly once")
    return order


def restore_canonical_cell_order(
    sequence_tensor: torch.Tensor,
    cell_order: Optional[torch.Tensor],
) -> torch.Tensor:
    if cell_order is None:
        return sequence_tensor
    inverse = torch.empty_like(cell_order)
    inverse[cell_order] = torch.arange(cell_order.numel(), device=cell_order.device)
    return sequence_tensor.index_select(1, inverse)


def training_cell_order(
    *,
    mode: str,
    seed: int,
    global_step: int,
    accumulation_index: int,
    device: torch.device,
) -> Optional[torch.Tensor]:
    if mode == "row_major":
        return None
    if mode != "random":
        raise ValueError(f"Unknown cell-order training mode: {mode}")
    generator = torch.Generator(device="cpu")
    generator.manual_seed(
        int(seed)
        + 1_000_003 * int(global_step)
        + 10_007 * int(accumulation_index)
    )
    return torch.randperm(CELLS, generator=generator).to(device=device)


configure_sudoku(N, BOX_ROWS, BOX_COLS)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def visible_gpu_uuid() -> str:
    output = subprocess.check_output(
        [
            "nvidia-smi",
            "-i",
            "0",
            "--query-gpu=uuid",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    )
    rows = [row.strip() for row in output.splitlines() if row.strip()]
    if len(rows) != 1:
        raise RuntimeError(
            f"Expected exactly one visible physical GPU UUID, got {rows}"
        )
    return rows[0]


def verify_installed_fla_against_manifest(
    provenance: Dict[str, Any],
) -> Dict[str, Any]:
    import fla

    package_root = Path(fla.__file__).resolve().parent
    package_paths = [Path(path).resolve() for path in fla.__path__]
    if package_paths != [package_root]:
        raise RuntimeError(
            f"FLA has unexpected namespace/package paths: {package_paths}"
        )
    if not str(package_root).startswith("/huyang2/double-loop/"):
        raise RuntimeError(
            f"FLA resolved outside the persistent project root: {package_root}"
        )
    expected_rows = provenance.get("source_file_hashes")
    if not isinstance(expected_rows, dict) or not expected_rows:
        raise RuntimeError("Gain-Budget manifest lacks FLA source file hashes")
    tree_digest = hashlib.sha256()
    for relative_path in sorted(expected_rows):
        expected = expected_rows[relative_path].get("installed_sha256")
        installed_path = package_root.parent / relative_path
        if not installed_path.is_file():
            raise RuntimeError(f"Installed FLA file is missing: {installed_path}")
        actual = sha256_file(installed_path)
        if actual != expected:
            raise RuntimeError(
                "Installed FLA changed after the CUDA contract: "
                f"{relative_path} {actual} != {expected}"
            )
        tree_digest.update(relative_path.encode("utf-8"))
        tree_digest.update(b"\0")
        tree_digest.update(bytes.fromhex(actual))
    installed_python_files = {
        str(path.relative_to(package_root.parent))
        for path in package_root.rglob("*.py")
    }
    expected_python_files = {
        relative_path
        for relative_path in expected_rows
        if relative_path.endswith(".py")
    }
    extra_python_files = sorted(
        installed_python_files - expected_python_files
    )
    if extra_python_files:
        raise RuntimeError(
            "Installed FLA gained Python files after the CUDA contract: "
            f"{extra_python_files}"
        )
    tree_sha256 = tree_digest.hexdigest()
    if tree_sha256 != provenance.get("installed_fla_tree_sha256"):
        raise RuntimeError(
            "Installed FLA aggregate hash changed after the CUDA contract: "
            f"{tree_sha256} != "
            f"{provenance.get('installed_fla_tree_sha256')}"
        )
    return {
        "package_root": str(package_root),
        "package_paths": [str(path) for path in package_paths],
        "verified_file_count": len(expected_rows),
        "installed_fla_tree_sha256": tree_sha256,
    }


def validate_gain_budget_contract_manifest() -> Dict[str, Any]:
    manifest_value = os.environ.get("GAIN_BUDGET_CONTRACT_JSON", "").strip()
    if not manifest_value:
        raise RuntimeError(
            "Gain-Budget strict mode requires GAIN_BUDGET_CONTRACT_JSON"
        )
    manifest_path = Path(manifest_value).resolve()
    if not manifest_path.is_file():
        raise RuntimeError(f"Gain-Budget CUDA contract is missing: {manifest_path}")
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    repo_root = Path(__file__).resolve().parents[2]
    current_sha = subprocess.check_output(
        ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
        text=True,
    ).strip()
    provenance = payload.get("provenance", {})
    runtime_fla = verify_installed_fla_against_manifest(provenance)
    current_gpu_uuid = visible_gpu_uuid()
    current_gpu_name = torch.cuda.get_device_name(0)
    current_wheel = (
        repo_root
        / "wheelhouse"
        / "flash_linear_attention-0.5.2-9c8e42e-py3-none-any.whl"
    )
    current_wheel_sha256 = (
        sha256_file(current_wheel) if current_wheel.is_file() else ""
    )
    checks = {
        "status": payload.get("status") == "passed",
        "git_sha": payload.get("git_sha") == current_sha,
        "runtime_cuda_visible_devices": os.environ.get(
            "CUDA_VISIBLE_DEVICES"
        )
        == "0",
        "cuda_visible_devices": payload.get("cuda_visible_devices") == "0",
        "runtime_single_gpu": torch.cuda.device_count() == 1,
        "gpu_uuid": payload.get("gpu_uuid") == current_gpu_uuid,
        "expected_gpu_uuid": payload.get("expected_gpu_uuid")
        == current_gpu_uuid,
        "gpu_name": payload.get("device") == current_gpu_name,
        "gpu1": "A800" in current_gpu_name,
        "fla_source": provenance.get("fla_source_sha")
        == GAIN_BUDGET_FLA_SHA,
        "wheel_manifest": provenance.get("wheel_sha256")
        == GAIN_BUDGET_WHEEL_SHA256,
        "wheel_runtime": current_wheel_sha256
        == GAIN_BUDGET_WHEEL_SHA256,
        "fla_tree": runtime_fla["installed_fla_tree_sha256"]
        == provenance.get("installed_fla_tree_sha256"),
        "fla_package_root": runtime_fla["package_root"]
        == provenance.get("fla_package_root"),
        "torch_version": payload.get("torch_version") == torch.__version__,
        "triton_version": payload.get("triton_version")
        == importlib.metadata.version("triton"),
        "chunk": "chunk_boundary_forward" in payload,
        "backward": "chunk_backward" in payload,
        "backward_bfloat16": "chunk_backward_bfloat16" in payload,
        "state_carry": "chunk_state_carry" in payload,
        "mode_none": payload.get("mode_none_exact", {}).get(
            "output_bitwise_equal"
        )
        is True,
        "budget_layer": "budget_layer_backward" in payload,
        "benchmark_projection_time": payload.get("benchmark", {}).get(
            "projection_time_overhead_frac",
            float("inf"),
        )
        <= 0.20,
        "benchmark_projection_memory": payload.get("benchmark", {}).get(
            "projection_memory_overhead_frac",
            float("inf"),
        )
        <= 0.20,
    }
    failed = sorted(name for name, passed in checks.items() if not passed)
    if failed:
        raise RuntimeError(
            f"Gain-Budget CUDA contract does not match this run: {failed}"
        )
    digest = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    return {
        "path": str(manifest_path),
        "sha256": digest,
        "git_sha": current_sha,
        "checks": checks,
        "torch_version": payload.get("torch_version"),
        "device": payload.get("device"),
        "gpu_uuid": current_gpu_uuid,
        "runtime_fla": runtime_fla,
        "time_overhead_frac": payload["benchmark"]["time_overhead_frac"],
        "memory_overhead_frac": payload["benchmark"]["memory_overhead_frac"],
        "projection_time_overhead_frac": payload["benchmark"][
            "projection_time_overhead_frac"
        ],
        "projection_memory_overhead_frac": payload["benchmark"][
            "projection_memory_overhead_frac"
        ],
    }


def resolve_strict_fla_source(backbone: str) -> Tuple[str, str, str]:
    expected_sha = os.environ.get("FLA_EXPECTED_SOURCE_SHA", GAIN_BUDGET_FLA_SHA).strip()
    if expected_sha not in ALLOWED_FLA_SOURCE_SHAS:
        raise RuntimeError(f"Unapproved FLA source SHA: {expected_sha!r}")
    if backbone == "raven" and expected_sha != RAVEN_FLA_SHA:
        raise RuntimeError(
            f"Raven requires the pinned latest FLA SHA {RAVEN_FLA_SHA}, got {expected_sha}"
        )

    expected_layer = {
        "fla_gdn": FLAGatedDeltaNet,
        "gdn2": GatedDeltaNet2,
        "kda": KimiDeltaAttention,
        "raven": FLARaven,
        "momentum": GatedDeltaNet2,
    }[backbone]
    if expected_layer is None:
        raise RuntimeError(f"Official FLA layer for {backbone} is unavailable")
    module = __import__(expected_layer.__module__, fromlist=[expected_layer.__name__])
    module_path = Path(module.__file__).resolve()

    source_root_value = os.environ.get("FLA_SOURCE_ROOT", "").strip()
    if source_root_value:
        source_root = Path(source_root_value).resolve()
        expected_package_root = (source_root / "fla").resolve()
        if expected_package_root not in module_path.parents:
            raise RuntimeError(
                f"FLA module {module_path} is outside pinned source root {source_root}"
            )
        source_sha = subprocess.check_output(
            ["git", "-C", str(source_root), "rev-parse", "HEAD"],
            text=True,
        ).strip()
    else:
        marker = Path(
            os.environ.get(
                "FLA_SOURCE_SHA_MARKER",
                "/huyang2/double-loop/.cache/fla-source-sha",
            )
        )
        source_sha = marker.read_text(encoding="utf-8").strip() if marker.is_file() else ""
        source_root = module_path.parents[1]
    if source_sha != expected_sha:
        raise RuntimeError(f"Pinned FLA source mismatch: {source_sha!r} != {expected_sha}")
    return source_sha, str(source_root), str(module_path)


def strict_fla_runtime_summary(model: nn.Module, backbone: str) -> Dict[str, Any]:
    if os.environ.get("FLA_DISABLE_BACKEND_DISPATCH") != "1":
        raise RuntimeError("Strict FLA mode requires FLA_DISABLE_BACKEND_DISPATCH=1 before Python starts")
    if os.environ.get("FLA_CONV_BACKEND") != "triton":
        raise RuntimeError("Strict FLA mode requires FLA_CONV_BACKEND=triton")
    from fla.ops.backends import _DISPATCH_DISABLED

    if not _DISPATCH_DISABLED:
        raise RuntimeError("FLA backend dispatch is active; refusing a run with possible silent backend substitution")
    source_sha, source_root, source_module = resolve_strict_fla_source(backbone)
    expected_layers = {
        "fla_gdn": FLAGatedDeltaNet,
        "gdn2": GatedDeltaNet2,
        "kda": KimiDeltaAttention,
        "raven": FLARaven,
        "momentum": load_external_momentum_layer(),
    }
    expected = expected_layers[backbone]
    rows = []
    reasoner = getattr(model, "reasoner", None)
    if reasoner is None:
        raise RuntimeError("Strict FLA mode could not find model.reasoner")
    for layer_idx, block in enumerate(reasoner.blocks):
        core = getattr(getattr(block, "time_mix", None), "core", None)
        if type(core) is not expected:
            raise RuntimeError(
                f"Layer {layer_idx} is {type(core)}, expected exact official class {expected}"
            )
        use_short_conv = bool(getattr(core, "use_short_conv", False))
        conv_backends = (
            {
                name: getattr(core, name).backend
                for name in ("q_conv1d", "k_conv1d", "v_conv1d")
            }
            if use_short_conv
            else {"q_conv1d": "disabled", "k_conv1d": "disabled", "v_conv1d": "disabled"}
        )
        if use_short_conv and set(conv_backends.values()) != {"triton"}:
            raise RuntimeError(f"Layer {layer_idx} changed short-conv backend: {conv_backends}")
        time_mix = getattr(block, "time_mix", None)
        address_mode = getattr(time_mix, "address_mode", "none")
        fast_slow_mode = getattr(time_mix, "fast_slow_decay_mode", "none")
        gain_budget_mode = getattr(time_mix, "gain_budget_mode", "none")
        precondition_mode = getattr(time_mix, "precondition_mode", "none")
        update_mode = getattr(time_mix, "update_mode", "none")
        if backbone == "momentum":
            execution_path = (
                "pinned_external_second_order_momentum_delta_chunk_with_"
                "explicit_state_and_velocity"
            )
        elif update_mode == "log_spd_metric" and address_mode == "position_qk":
            execution_path = (
                "canonical_position_qk_plus_per_layer_bounded_log_spd_"
                "then_one_official_gdn2_chunk"
            )
        elif update_mode == "coupled_address_rows" and address_mode == "position_qk":
            execution_path = (
                "canonical_position_qk_coupled_k64_address_rows_in_one_"
                "official_gdn2_chunk"
            )
        elif update_mode == "paired_address_bank" and address_mode == "position_qk":
            execution_path = (
                "canonical_position_qk_paired_address_state_bank_in_one_"
                "official_gdn2_chunk"
            )
        elif update_mode == "adaptive_signed_erase" and address_mode == "position_qk":
            execution_path = (
                "canonical_position_qk_plus_adaptive_signed_erase_"
                "then_one_official_gdn2_chunk"
            )
        elif update_mode == "bi_axis_value_decay" and address_mode == "position_qk":
            execution_path = (
                "canonical_position_qk_plus_grouped_value_decay_moving_frame_"
                "around_one_official_gdn2_chunk"
            )
        elif update_mode == "gauge_balanced_bi_axis" and address_mode == "position_qk":
            execution_path = (
                "canonical_position_qk_plus_gauge_balanced_grouped_value_decay_"
                "around_one_official_gdn2_chunk"
            )
        elif update_mode == "raven_routed_gdn" and address_mode == "position_qk":
            execution_path = (
                "canonical_position_qk_plus_raven_soft_slot_allocation_"
                "inside_one_official_gdn2_chunk"
            )
        elif update_mode == "orthogonal_head_write" and address_mode == "position_qk":
            execution_path = (
                "canonical_position_qk_plus_orthogonal_head_write_routing_"
                "then_one_official_gdn2_chunk"
            )
        elif update_mode == "interleaved_write" and address_mode == "position_qk":
            execution_path = (
                "canonical_position_qk_interleaved_auxiliary_write_then_"
                "parent_update_in_one_official_gdn2_chunk"
            )
        elif update_mode == "orthogonal_chunk_state" and address_mode == "position_qk":
            execution_path = (
                "canonical_position_qk_two_official_gdn2_chunks_with_"
                "orthogonal_live_state_transport"
            )
        elif update_mode == "terminal_consolidation" and address_mode == "position_qk":
            execution_path = (
                "canonical_position_qk_then_official_gdn2_chunk_plus_"
                "terminal_consolidation_official_gdn2_chunk"
            )
        elif update_mode == "state_feedback" and address_mode == "position_qk":
            execution_path = (
                "canonical_position_qk_plus_closed_loop_state_feedback_"
                "then_official_gdn2_chunk"
            )
        elif update_mode == "coherent_delta" and address_mode == "position_qk":
            execution_path = (
                "canonical_position_qk_plus_coherent_delta_gates_"
                "then_official_gdn2_chunk"
            )
        elif precondition_mode != "none" and address_mode == "position_qk":
            execution_path = (
                "canonical_position_qk_plus_futureseed_tied_preconditioner_"
                "then_official_gdn2_chunk"
            )
        elif precondition_mode != "none":
            execution_path = (
                "futureseed_tied_preconditioner_then_official_gdn2_chunk"
            )
        elif address_mode == "position_qk":
            execution_path = "canonical_position_qk_then_official_gdn2_chunk"
        elif address_mode == "anchor_rotary":
            execution_path = "content_qk_anchor_rotation_then_official_gdn2_chunk"
        elif address_mode == "anchor_phase":
            execution_path = "content_qk_learned_address_phase_then_official_gdn2_chunk"
        elif address_mode == "anchor_residual":
            execution_path = "content_qk_shared_address_residual_then_official_gdn2_chunk"
        elif address_mode == "shared_namespace":
            execution_path = (
                "cross_layer_shared_address_namespace_then_official_gdn2_chunk"
            )
        elif address_mode == "anchor_qk_residual":
            execution_path = "content_qk_decoupled_address_residual_then_official_gdn2_chunk"
        elif address_mode == "anchor_carrier":
            execution_path = (
                "shared_address_conditioned_write_carrier_folded_into_"
                "official_gdn2_chunk"
            )
        elif fast_slow_mode == "positive_causal":
            execution_path = "positive_causal_hazard_then_official_gdn2_chunk"
        elif fast_slow_mode == "external_identity":
            execution_path = "external_identity_hazard_then_official_gdn2_chunk"
        elif backbone == "raven":
            execution_path = "official_raven_sparse_slot_router_then_gsa_chunk"
        elif gain_budget_mode == "none":
            execution_path = "official_layer_forward"
        elif gain_budget_mode == "external_identity":
            execution_path = (
                "external_fp32_normalization_then_bfloat16_"
                "tolerance_official_gdn2_chunk"
            )
        else:
            execution_path = (
                "fp32_gain_budget_projected_then_bfloat16_"
                "tolerance_audited_official_gdn2_chunk"
            )
        rows.append(
            {
                "layer": layer_idx,
                "class": f"{type(core).__module__}.{type(core).__qualname__}",
                "conv_backends": conv_backends,
                "address_mode": address_mode,
                "gain_budget_mode": gain_budget_mode,
                "fast_slow_decay_mode": fast_slow_mode,
                "precondition_mode": precondition_mode,
                "update_mode": update_mode,
                "execution_path": execution_path,
                "state_elements_per_head": (
                    int(core.num_slots) * (int(core.head_k_dim) + int(core.head_v_dim))
                    if backbone == "raven"
                    else int(time_mix.state_elements_per_head())
                    if backbone == "momentum"
                    else int(getattr(time_mix, "head_dim"))
                    * int(getattr(time_mix, "head_v_dim"))
                ),
                "num_slots": int(core.num_slots) if backbone == "raven" else None,
                "topk": int(core.topk) if backbone == "raven" else None,
            }
        )
    gain_budget_enabled = any(
        row["gain_budget_mode"] != "none" for row in rows
    )
    gain_budget_contract_required = (
        gain_budget_enabled
        or os.environ.get("REQUIRE_GAIN_BUDGET_CONTRACT") == "1"
    )
    gain_budget_contract = (
        validate_gain_budget_contract_manifest()
        if gain_budget_contract_required
        else None
    )
    return {
        "strict": True,
        "fla_source_sha": source_sha,
        "fla_source_root": source_root,
        "fla_source_module": source_module,
        "backend_dispatch_disabled": bool(_DISPATCH_DISABLED),
        "conv_backend": "triton",
        "layers": rows,
        "gain_budget_contract": gain_budget_contract,
        "momentum_source": (
            momentum_source_summary() if backbone == "momentum" else None
        ),
    }


@dataclass
class FullMetrics:
    label_exact: float
    valid_sudoku: float
    solved_valid_clue: float
    clue_ok: float
    blank_acc: float
    blank_acc_early: float
    blank_acc_mid: float
    blank_acc_late: float
    blank_acc_early_late_gap: float
    blank_frac_early: float
    blank_frac_mid: float
    blank_frac_late: float
    avg_filled: float


def choose_device(force_cpu: bool = False) -> torch.device:
    if force_cpu:
        return torch.device("cpu")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def forward_autocast(forward_dtype: str, device: torch.device):
    if forward_dtype == "float32":
        return nullcontext()
    if forward_dtype != "bfloat16":
        raise ValueError("--forward_dtype must be 'float32' or 'bfloat16'")
    if device.type != "cuda":
        raise ValueError("--forward_dtype=bfloat16 requires a CUDA device")
    return torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)


def pattern(row: int, col: int) -> int:
    return (BOX_COLS * (row % BOX_ROWS) + row // BOX_ROWS + col) % N


def shuffled_solution(rng: random.Random) -> List[int]:
    rows = [
        g * BOX_ROWS + r
        for g in rng.sample(range(BOX_COLS), BOX_COLS)
        for r in rng.sample(range(BOX_ROWS), BOX_ROWS)
    ]
    cols = [
        g * BOX_COLS + c
        for g in rng.sample(range(BOX_ROWS), BOX_ROWS)
        for c in rng.sample(range(BOX_COLS), BOX_COLS)
    ]
    digits = rng.sample(range(N), N)
    return [digits[pattern(r, c)] for r in rows for c in cols]


def choose_blank_cells(holes: int, rng: random.Random, hole_pattern: str) -> List[int]:
    if hole_pattern == "random":
        return rng.sample(range(CELLS), holes)
    raise ValueError("--hole_pattern now supports only 'random'; structured hole patterns were Sudoku-specific probes.")


def make_batch(
    batch_size: int,
    holes_min: int,
    holes_max: int,
    hole_pattern: str,
    rng: random.Random,
    *,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    inputs = torch.full((batch_size, CELLS), BLANK, dtype=torch.long)
    labels = torch.empty((batch_size, CELLS), dtype=torch.long)
    clue_mask = torch.zeros((batch_size, CELLS), dtype=torch.bool)

    for b in range(batch_size):
        solution = shuffled_solution(rng)
        holes = rng.randint(holes_min, holes_max)
        blank_cells = set(choose_blank_cells(holes, rng, hole_pattern))
        puzzle = [BLANK if i in blank_cells else solution[i] for i in range(CELLS)]
        inputs[b] = torch.tensor(puzzle, dtype=torch.long)
        labels[b] = torch.tensor(solution, dtype=torch.long)
        clue_mask[b] = torch.tensor([i not in blank_cells for i in range(CELLS)], dtype=torch.bool)

    return inputs.to(device), labels.to(device), clue_mask.to(device)


class OfficialSudokuDataset:
    """Read official EqR Sudoku npy arrays and map tokens to this runner's ids."""

    def __init__(
        self,
        data_dir: Path,
        split: str,
        *,
        row_indices_path: Optional[Path] = None,
    ) -> None:
        try:
            import numpy as np
        except Exception as exc:  # pragma: no cover - dependency is required only for official data.
            raise RuntimeError("Official Sudoku data mode requires numpy.") from exc
        self._np = np
        self.split = str(split)
        split_dir = Path(data_dir) / self.split
        self.inputs = np.load(split_dir / "all__inputs.npy", mmap_mode="r")
        self.labels = np.load(split_dir / "all__labels.npy", mmap_mode="r")
        if self.inputs.shape != self.labels.shape:
            raise ValueError(f"Official Sudoku inputs/labels shape mismatch: {self.inputs.shape} vs {self.labels.shape}")
        if len(self.inputs.shape) != 2 or int(self.inputs.shape[1]) != CELLS:
            raise ValueError(f"Official Sudoku expects shape [N, {CELLS}], got {self.inputs.shape}")
        self.base_count = int(self.inputs.shape[0])
        self.row_indices = None
        if row_indices_path is not None:
            row_indices = np.load(Path(row_indices_path), mmap_mode="r")
            if row_indices.ndim != 1 or row_indices.dtype.kind not in ("i", "u"):
                raise ValueError(
                    "Official Sudoku row indices must be a one-dimensional integer npy array."
                )
            if int(row_indices.size) == 0:
                raise ValueError("Official Sudoku row indices must not be empty.")
            if int(row_indices[0]) < 0 or int(row_indices[-1]) >= self.base_count:
                raise ValueError(
                    f"Official Sudoku row indices must be within [0, {self.base_count})."
                )
            if bool((np.diff(row_indices) <= 0).any()):
                raise ValueError(
                    "Official Sudoku row indices must be unique and strictly increasing."
                )
            self.row_indices = row_indices
        base_blank_counts = (self.inputs == 1).sum(axis=1).astype("int16", copy=False)
        self.blank_counts = (
            base_blank_counts
            if self.row_indices is None
            else base_blank_counts[self.row_indices]
        )
        self._range_cache: Dict[Tuple[int, int], Any] = {}

    def blank_summary(self) -> Dict[str, Any]:
        counts = self.blank_counts
        hist: Dict[str, int] = {}
        unique, freq = self._np.unique(counts, return_counts=True)
        for blank_count, count in zip(unique.tolist(), freq.tolist()):
            hist[str(int(blank_count))] = int(count)
        return {
            "min": int(counts.min()),
            "max": int(counts.max()),
            "mean": float(counts.mean()),
            "histogram": hist,
        }

    def _indices_for_blank_range(self, holes_min: int, holes_max: int):
        lo = max(0, int(holes_min))
        hi = min(CELLS, int(holes_max))
        if hi < lo:
            raise ValueError(f"Invalid official Sudoku blank-count range: {holes_min}-{holes_max}")
        key = (lo, hi)
        cached = self._range_cache.get(key)
        if cached is not None:
            return cached
        mask = (self.blank_counts >= lo) & (self.blank_counts <= hi)
        indices = self._np.flatnonzero(mask)
        if indices.size == 0:
            summary = self.blank_summary()
            raise ValueError(
                "Official Sudoku split "
                f"{self.split!r} has no samples with blank_count in {lo}-{hi}; "
                f"available range is {summary['min']}-{summary['max']}."
            )
        self._range_cache[key] = indices
        return indices

    def __len__(self) -> int:
        return int(self.blank_counts.shape[0])

    def _base_indices(self, indices):
        if self.row_indices is None:
            return indices
        return self.row_indices[indices]

    def _map_arrays(
        self,
        input_values,
        label_values,
        *,
        device: torch.device,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        input_tensor = torch.as_tensor(input_values.astype("int64", copy=False), dtype=torch.long)
        label_tensor = torch.as_tensor(label_values.astype("int64", copy=False), dtype=torch.long)
        clue_mask = input_tensor.ne(1)
        mapped_inputs = torch.where(clue_mask, input_tensor - 2, torch.full_like(input_tensor, BLANK))
        mapped_labels = label_tensor - 2
        if bool((mapped_labels < 0).any() or (mapped_labels >= N).any()):
            raise ValueError("Official Sudoku labels must map into [0, N).")
        if bool((mapped_inputs < 0).any() or (mapped_inputs > BLANK).any()):
            raise ValueError("Official Sudoku inputs must map into [0, BLANK].")
        return mapped_inputs.to(device), mapped_labels.to(device), clue_mask.to(device)

    def sample_batch(
        self,
        batch_size: int,
        rng: random.Random,
        *,
        holes_min: Optional[int] = None,
        holes_max: Optional[int] = None,
        device: torch.device,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if holes_min is None or holes_max is None:
            indices = [rng.randrange(len(self)) for _ in range(batch_size)]
        else:
            candidates = self._indices_for_blank_range(int(holes_min), int(holes_max))
            indices = candidates[[rng.randrange(len(candidates)) for _ in range(batch_size)]]
        base_indices = self._base_indices(indices)
        return self._map_arrays(
            self.inputs[base_indices],
            self.labels[base_indices],
            device=device,
        )

    def fixed_batch(
        self,
        batch_size: int,
        seed: int,
        *,
        device: torch.device,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if batch_size > len(self):
            raise ValueError(f"Requested eval_n={batch_size}, but official split {self.split!r} has only {len(self)} rows.")
        rng = random.Random(seed)
        indices = rng.sample(range(len(self)), batch_size)
        base_indices = self._base_indices(indices)
        return self._map_arrays(
            self.inputs[base_indices],
            self.labels[base_indices],
            device=device,
        )

    def fixed_batch_by_blank_range(
        self,
        batch_size: int,
        seed: int,
        *,
        holes_min: int,
        holes_max: int,
        device: torch.device,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        candidates = self._indices_for_blank_range(int(holes_min), int(holes_max))
        rng = random.Random(seed)
        count = min(int(batch_size), int(len(candidates)))
        positions = rng.sample(range(len(candidates)), count)
        indices = candidates[positions]
        base_indices = self._base_indices(indices)
        return self._map_arrays(
            self.inputs[base_indices],
            self.labels[base_indices],
            device=device,
        )


def official_sudoku_enabled(args: argparse.Namespace) -> bool:
    return bool(str(args.official_sudoku_data_dir).strip())


def make_train_batch(
    args: argparse.Namespace,
    official_train: Optional[OfficialSudokuDataset],
    batch_size: int,
    holes_min: int,
    holes_max: int,
    rng: random.Random,
    *,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if official_train is not None:
        return official_train.sample_batch(
            batch_size,
            rng,
            holes_min=holes_min,
            holes_max=holes_max,
            device=device,
        )
    return make_batch(batch_size, holes_min, holes_max, args.hole_pattern, rng, device=device)


def make_eval_batch(
    args: argparse.Namespace,
    official_eval: Optional[OfficialSudokuDataset],
    batch_size: int,
    holes: int,
    seed: int,
    *,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if official_eval is not None:
        return official_eval.fixed_batch(batch_size, seed, device=device)
    return make_batch(batch_size, holes, holes, args.hole_pattern, random.Random(seed), device=device)


class RWKVTimeMix(nn.Module):
    """RWKV-style WKV block with CUDA RWKV7 paths and a torch fallback."""

    def __init__(
        self,
        d_model: int,
        heads: int,
        head_dim: int,
        *,
        layer_id: int,
        layers: int,
        rwkv_kernel: str,
    ) -> None:
        super().__init__()
        if d_model != heads * head_dim:
            raise ValueError("d_model must equal heads * head_dim")
        self.d_model = d_model
        self.heads = heads
        self.head_dim = head_dim
        self.rwkv_kernel = rwkv_kernel
        self._fallback_warned = False

        ratio_0_to_1 = layer_id / max(layers - 1, 1)
        ratio_1_to_almost0 = 1.0 - (layer_id / max(layers, 1))
        ddd = torch.arange(d_model, dtype=torch.float32).view(1, 1, d_model) / max(d_model - 1, 1)
        self.mix_r = nn.Parameter(1.0 - torch.pow(ddd, 0.2 * ratio_1_to_almost0))
        self.mix_w = nn.Parameter(1.0 - torch.pow(ddd, 0.9 * ratio_1_to_almost0))
        self.mix_k = nn.Parameter(1.0 - torch.pow(ddd, 0.7 * ratio_1_to_almost0))
        self.mix_v = nn.Parameter(1.0 - torch.pow(ddd, 0.7 * ratio_1_to_almost0))
        self.mix_a = nn.Parameter(1.0 - torch.pow(ddd, 0.9 * ratio_1_to_almost0))
        self.mix_g = nn.Parameter(1.0 - torch.pow(ddd, 0.2 * ratio_1_to_almost0))

        base_decay = torch.zeros(d_model)
        zigzag = torch.zeros(d_model)
        linear = torch.zeros(d_model)
        for n in range(d_model):
            frac = n / max(d_model - 1, 1)
            linear[n] = frac - 0.5
            base_decay[n] = -6.0 + 6.0 * frac ** (1.0 + ratio_0_to_1**0.3)
            local = ((n % head_dim) - ((head_dim - 1) / 2.0)) / max((head_dim - 1) / 2.0, 1.0)
            zigzag[n] = local * abs(local)
        self.time_decay = nn.Parameter((base_decay + 0.5 + zigzag * 2.5).view(1, 1, d_model))
        self.decay_delta = nn.Linear(d_model, d_model, bias=False)
        self.state_lr = nn.Linear(d_model, d_model, bias=True)
        self.key_scale = nn.Parameter((0.71 - linear * 0.1).view(1, 1, d_model))
        self.key_lr_mix = nn.Parameter(torch.full((1, 1, d_model), 1.02))

        self.receptance = nn.Linear(d_model, d_model, bias=False)
        self.key = nn.Linear(d_model, d_model, bias=False)
        self.value = nn.Linear(d_model, d_model, bias=False)
        self.gate = nn.Linear(d_model, d_model, bias=False)
        self.out = nn.Linear(d_model, d_model, bias=False)
        self.group_norm = nn.GroupNorm(heads, d_model, eps=64e-5)

        scale = d_model**0.5
        self.receptance.weight.data.uniform_(-0.5 / scale, 0.5 / scale)
        self.key.weight.data.uniform_(-0.05 / scale, 0.05 / scale)
        self.value.weight.data.uniform_(-0.5 / scale, 0.5 / scale)
        nn.init.zeros_(self.decay_delta.weight)
        nn.init.zeros_(self.state_lr.weight)
        self.state_lr.bias.data.copy_(-0.19 + zigzag * 0.3 + linear * 0.4)
        nn.init.zeros_(self.out.weight)

    def forward(
        self,
        x: torch.Tensor,
        *,
        initial_state: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size, seq_len, channels = x.shape
        heads, head_dim = self.heads, self.head_dim

        shifted = torch.zeros_like(x)
        shifted[:, 1:] = x[:, :-1]
        delta = shifted - x

        xr = x + delta * self.mix_r
        xw = x + delta * self.mix_w
        xk = x + delta * self.mix_k
        xv = x + delta * self.mix_v
        xa = x + delta * self.mix_a
        xg = x + delta * self.mix_g

        r = torch.sigmoid(self.receptance(xr)).view(batch_size, seq_len, heads, head_dim)
        w_raw = self.time_decay + torch.tanh(self.decay_delta(xw))
        w_log = -F.softplus(-w_raw) - 0.5
        state_lr = torch.sigmoid(self.state_lr(xa)).view(batch_size, seq_len, heads, head_dim)
        k_flat = self.key(xk)
        kk = F.normalize((k_flat * self.key_scale).view(batch_size, seq_len, heads, head_dim), dim=-1, p=2.0)
        k = (k_flat * (1.0 + (state_lr.reshape(batch_size, seq_len, channels) - 1.0) * self.key_lr_mix)).view(
            batch_size, seq_len, heads, head_dim
        )
        v = self.value(xv).view(batch_size, seq_len, heads, head_dim)
        g = torch.sigmoid(self.gate(xg))

        if self.rwkv_kernel in {"auto", "cuda", "statepassing"}:
            if self._can_use_statepassing(x, initial_state):
                y, terminal_state = self._forward_statepassing(
                    r=r,
                    w_raw=w_raw.view(batch_size, seq_len, heads, head_dim),
                    k=k,
                    v=v,
                    kk=kk,
                    state_lr=state_lr,
                    gate=g,
                    initial_state=initial_state,
                    original_dtype=x.dtype,
                )
                return y, terminal_state
            if self.rwkv_kernel in {"cuda", "statepassing"}:
                ok, reason = statepassing_available(self.head_dim)
                if not ok:
                    raise RuntimeError(f"RWKV_KERNEL={self.rwkv_kernel} requested but unavailable: {reason}")
                raise RuntimeError(f"RWKV_KERNEL={self.rwkv_kernel} requested but inputs are not CUDA tensors")

        if self.rwkv_kernel in {"auto", "wind"}:
            if self._can_use_wind(x, initial_state):
                y, terminal_state = self._forward_wind(
                    r=r,
                    w_log=w_log.view(batch_size, seq_len, heads, head_dim),
                    k=k,
                    v=v,
                    kk=kk,
                    state_lr=state_lr,
                    gate=g,
                    initial_state=initial_state,
                    original_dtype=x.dtype,
                )
                return y, terminal_state
            if self.rwkv_kernel == "wind":
                ok, reason = wind_available(self.head_dim)
                if not ok:
                    raise RuntimeError(f"RWKV_KERNEL=wind requested but unavailable: {reason}")
                raise RuntimeError("RWKV_KERNEL=wind requested but inputs are not CUDA tensors")
            if not self._fallback_warned:
                sp_ok, sp_reason = statepassing_available(self.head_dim)
                ok, reason = wind_available(self.head_dim)
                print(
                    f"rwkv_kernel=auto using torch fallback: statepassing={sp_ok}:{sp_reason}; wind={ok}:{reason}",
                    flush=True,
                )
                self._fallback_warned = True

        return self._forward_torch(
            r=r,
            w_log=w_log,
            k=k,
            v=v,
            kk=kk,
            state_lr=state_lr,
            gate=g,
            initial_state=initial_state,
        )

    def _can_use_statepassing(self, x: torch.Tensor, initial_state: Optional[torch.Tensor]) -> bool:
        if StatePassingRWKV7 is None or not x.is_cuda:
            return False
        ok, _reason = statepassing_available(self.head_dim)
        if not ok:
            return False
        if initial_state is not None:
            expected = (x.shape[0], self.heads, self.head_dim, self.head_dim)
            if tuple(initial_state.shape) != expected:
                return False
        return True

    def _can_use_wind(self, x: torch.Tensor, initial_state: Optional[torch.Tensor]) -> bool:
        if WindRWKV7 is None or not x.is_cuda:
            return False
        ok, _reason = wind_available(self.head_dim)
        if not ok:
            return False
        if initial_state is not None:
            expected = (x.shape[0], self.heads, self.head_dim, self.head_dim)
            if tuple(initial_state.shape) != expected:
                return False
        return True

    @staticmethod
    def _pad_time(t: torch.Tensor, pad_len: int, value: float = 0.0) -> torch.Tensor:
        if pad_len <= 0:
            return t
        pad_shape = (t.shape[0], pad_len, *t.shape[2:])
        pad = t.new_full(pad_shape, value)
        return torch.cat([t, pad], dim=1)

    def _forward_wind(
        self,
        *,
        r: torch.Tensor,
        w_log: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        kk: torch.Tensor,
        state_lr: torch.Tensor,
        gate: torch.Tensor,
        initial_state: Optional[torch.Tensor],
        original_dtype: torch.dtype,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size, seq_len, heads, head_dim = r.shape
        pad_len = (-seq_len) % 16
        q_bf16 = self._pad_time(r, pad_len).to(torch.bfloat16)
        w_bf16 = self._pad_time(w_log, pad_len, value=-60.0).to(torch.bfloat16)
        k_bf16 = self._pad_time(k, pad_len).to(torch.bfloat16)
        v_bf16 = self._pad_time(v, pad_len).to(torch.bfloat16)
        z_bf16 = self._pad_time(-kk, pad_len).to(torch.bfloat16)
        a_bf16 = self._pad_time(kk * state_lr, pad_len).to(torch.bfloat16)

        if initial_state is None:
            s0 = torch.zeros(batch_size, heads, head_dim, head_dim, device=r.device, dtype=torch.bfloat16)
        else:
            s0 = initial_state.to(device=r.device, dtype=torch.bfloat16)

        y_bf16, terminal_state_bf16 = WindRWKV7.apply(w_bf16, q_bf16, k_bf16, v_bf16, z_bf16, a_bf16, s0)
        y = y_bf16[:, :seq_len].reshape(batch_size, seq_len, heads * head_dim).to(original_dtype)
        y = self.group_norm(y.reshape(batch_size * seq_len, heads * head_dim)).reshape(batch_size, seq_len, heads * head_dim)
        return self.out(y * gate), terminal_state_bf16.to(original_dtype)

    def _forward_statepassing(
        self,
        *,
        r: torch.Tensor,
        w_raw: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        kk: torch.Tensor,
        state_lr: torch.Tensor,
        gate: torch.Tensor,
        initial_state: Optional[torch.Tensor],
        original_dtype: torch.dtype,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size, seq_len, heads, head_dim = r.shape
        pad_len = (-seq_len) % 16
        r_bf16 = self._pad_time(r, pad_len).to(torch.bfloat16)
        w_bf16 = self._pad_time(w_raw, pad_len, value=-60.0).to(torch.bfloat16)
        k_bf16 = self._pad_time(k, pad_len).to(torch.bfloat16)
        v_bf16 = self._pad_time(v, pad_len).to(torch.bfloat16)
        a_bf16 = self._pad_time(-kk, pad_len).to(torch.bfloat16)
        b_bf16 = self._pad_time(kk * state_lr, pad_len).to(torch.bfloat16)

        if initial_state is None:
            s0 = torch.zeros(batch_size, heads, head_dim, head_dim, device=r.device, dtype=torch.float32)
        else:
            expected = (batch_size, heads, head_dim, head_dim)
            if tuple(initial_state.shape) != expected:
                raise ValueError(f"initial_state shape {tuple(initial_state.shape)} does not match {expected}")
            s0 = initial_state.to(device=r.device, dtype=torch.float32)

        y_bf16, terminal_state = StatePassingRWKV7.apply(s0, r_bf16, w_bf16, k_bf16, v_bf16, a_bf16, b_bf16)
        y = y_bf16[:, :seq_len].reshape(batch_size, seq_len, heads * head_dim).to(original_dtype)
        y = self.group_norm(y.reshape(batch_size * seq_len, heads * head_dim)).reshape(batch_size, seq_len, heads * head_dim)
        return self.out(y * gate), terminal_state.to(original_dtype)

    def _forward_torch(
        self,
        *,
        r: torch.Tensor,
        w_log: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        kk: torch.Tensor,
        state_lr: torch.Tensor,
        gate: torch.Tensor,
        initial_state: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size, seq_len, channels = gate.shape
        heads, head_dim = self.heads, self.head_dim
        w = torch.exp(-torch.exp(w_log.float())).to(gate.dtype).view(batch_size, seq_len, heads, head_dim)

        if initial_state is None:
            memory = torch.zeros(batch_size, heads, head_dim, head_dim, device=gate.device, dtype=gate.dtype)
        else:
            expected = (batch_size, heads, head_dim, head_dim)
            if tuple(initial_state.shape) != expected:
                raise ValueError(f"initial_state shape {tuple(initial_state.shape)} does not match {expected}")
            memory = initial_state.to(device=gate.device, dtype=gate.dtype)

        outputs: List[torch.Tensor] = []
        for t in range(seq_len):
            erase = torch.einsum("bhij,bhj->bhi", memory, -kk[:, t])
            write_back = (kk[:, t] * state_lr[:, t]).unsqueeze(-2)
            memory = (
                memory * w[:, t].unsqueeze(-2)
                + erase.unsqueeze(-1) * write_back
                + v[:, t].unsqueeze(-1) * k[:, t].unsqueeze(-2)
            )
            y_t = torch.einsum("bhij,bhj->bhi", memory, r[:, t]).reshape(batch_size, channels)
            outputs.append(y_t)

        y = torch.stack(outputs, dim=1)
        y = self.group_norm(y.reshape(batch_size * seq_len, channels)).reshape(batch_size, seq_len, channels)
        return self.out(y * gate), memory


def rwkv7_lora_width(multiplier: float, d_model: int) -> int:
    return max(32, int(round((multiplier * (d_model**0.5)) / 32.0) * 32))


def rwkv7_ortho_init(tensor: torch.Tensor, scale: float) -> torch.Tensor:
    with torch.no_grad():
        rows, cols = tensor.shape
        gain = math.sqrt(rows / cols) if rows > cols else 1.0
        nn.init.orthogonal_(tensor, gain=gain * scale)
    return tensor


class RWKV7TimeMixOfficial(nn.Module):
    """Official RWKV7 TimeMix equations with explicit recurrent-state I/O."""

    def __init__(
        self,
        d_model: int,
        heads: int,
        head_dim: int,
        *,
        layer_id: int,
        layers: int,
        rwkv_kernel: str,
    ) -> None:
        super().__init__()
        if d_model != heads * head_dim:
            raise ValueError("d_model must equal heads * head_dim")
        if layers < 2:
            raise ValueError("official RWKV7 TimeMix requires at least two layers")
        if rwkv_kernel not in {"statepassing", "torch"}:
            raise ValueError("official RWKV7 TimeMix allows only explicit statepassing or torch kernels")
        self.d_model = int(d_model)
        self.heads = int(heads)
        self.head_dim = int(head_dim)
        self.layer_id = int(layer_id)
        self.rwkv_kernel = rwkv_kernel

        ratio_0_to_1 = layer_id / (layers - 1)
        ratio_1_to_almost0 = 1.0 - (layer_id / layers)
        ddd = torch.arange(d_model, dtype=torch.float32).view(1, 1, d_model) / d_model
        self.x_r = nn.Parameter(1.0 - torch.pow(ddd, 0.2 * ratio_1_to_almost0))
        self.x_w = nn.Parameter(1.0 - torch.pow(ddd, 0.9 * ratio_1_to_almost0))
        self.x_k = nn.Parameter(1.0 - torch.pow(ddd, 0.7 * ratio_1_to_almost0))
        self.x_v = nn.Parameter(1.0 - torch.pow(ddd, 0.7 * ratio_1_to_almost0))
        self.x_a = nn.Parameter(1.0 - torch.pow(ddd, 0.9 * ratio_1_to_almost0))
        self.x_g = nn.Parameter(1.0 - torch.pow(ddd, 0.2 * ratio_1_to_almost0))

        decay = torch.zeros(d_model)
        zigzag = torch.zeros(d_model)
        linear = torch.zeros(d_model)
        for index in range(d_model):
            linear[index] = index / max(d_model - 1, 1) - 0.5
            local = ((index % head_dim) - ((head_dim - 1) / 2.0)) / max((head_dim - 1) / 2.0, 1.0)
            zigzag[index] = local * abs(local)
            decay[index] = -6.0 + 6.0 * (index / max(d_model - 1, 1)) ** (1.0 + ratio_0_to_1**0.3)

        decay_width = rwkv7_lora_width(2.5, d_model)
        update_width = rwkv7_lora_width(2.5, d_model)
        value_width = rwkv7_lora_width(1.7, d_model)
        gate_width = rwkv7_lora_width(5.0, d_model)
        self.w1 = nn.Parameter(torch.zeros(d_model, decay_width))
        self.w2 = nn.Parameter(rwkv7_ortho_init(torch.zeros(decay_width, d_model), 0.1))
        self.w0 = nn.Parameter((decay + 0.5 + zigzag * 2.5).view(1, 1, d_model))
        self.a1 = nn.Parameter(torch.zeros(d_model, update_width))
        self.a2 = nn.Parameter(rwkv7_ortho_init(torch.zeros(update_width, d_model), 0.1))
        self.a0 = nn.Parameter((-0.19 + zigzag * 0.3 + linear * 0.4).view(1, 1, d_model))
        self.v1 = nn.Parameter(torch.zeros(d_model, value_width))
        self.v2 = nn.Parameter(rwkv7_ortho_init(torch.zeros(value_width, d_model), 0.1))
        self.v0 = nn.Parameter((0.73 - linear * 0.4).view(1, 1, d_model))
        self.g1 = nn.Parameter(torch.zeros(d_model, gate_width))
        self.g2 = nn.Parameter(rwkv7_ortho_init(torch.zeros(gate_width, d_model), 0.1))
        self.k_k = nn.Parameter((0.71 - linear * 0.1).view(1, 1, d_model))
        self.k_a = nn.Parameter(torch.full((1, 1, d_model), 1.02))
        self.r_k = nn.Parameter(torch.full((heads, head_dim), -0.04))

        self.receptance = nn.Linear(d_model, d_model, bias=False)
        self.key = nn.Linear(d_model, d_model, bias=False)
        self.value = nn.Linear(d_model, d_model, bias=False)
        self.output = nn.Linear(d_model, d_model, bias=False)
        self.ln_x = nn.GroupNorm(heads, d_model, eps=64e-5)
        scale = d_model**0.5
        self.receptance.weight.data.uniform_(-0.5 / scale, 0.5 / scale)
        self.key.weight.data.uniform_(-0.05 / scale, 0.05 / scale)
        self.value.weight.data.uniform_(-0.5 / scale, 0.5 / scale)
        nn.init.zeros_(self.output.weight)

    @staticmethod
    def _pad_time(tensor: torch.Tensor, pad_len: int, value: float = 0.0) -> torch.Tensor:
        if pad_len <= 0:
            return tensor
        pad_shape = (tensor.shape[0], pad_len, *tensor.shape[2:])
        return torch.cat((tensor, tensor.new_full(pad_shape, value)), dim=1)

    def _statepassing(
        self,
        *,
        r: torch.Tensor,
        w_raw: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        kk: torch.Tensor,
        a: torch.Tensor,
        initial_state: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if StatePassingRWKV7 is None or not r.is_cuda:
            raise RuntimeError("official RWKV7 requested statepassing CUDA, but CUDA inputs or extension are unavailable")
        available, reason = statepassing_available(self.head_dim)
        if not available:
            raise RuntimeError(f"official RWKV7 statepassing CUDA unavailable: {reason}")
        batch_size, seq_len, heads, head_dim = r.shape
        pad_len = (-seq_len) % 16
        recurrent = [
            self._pad_time(r, pad_len),
            self._pad_time(w_raw, pad_len, value=-60.0),
            self._pad_time(k, pad_len),
            self._pad_time(v, pad_len),
            self._pad_time(-kk, pad_len),
            self._pad_time(kk * a, pad_len),
        ]
        recurrent_bf16 = [tensor.to(torch.bfloat16) for tensor in recurrent]
        if initial_state is None:
            state = torch.zeros(batch_size, heads, head_dim, head_dim, device=r.device, dtype=torch.float32)
        else:
            expected = (batch_size, heads, head_dim, head_dim)
            if tuple(initial_state.shape) != expected:
                raise ValueError(f"initial_state shape {tuple(initial_state.shape)} does not match {expected}")
            state = initial_state.to(device=r.device, dtype=torch.float32)
        output, terminal_state = StatePassingRWKV7.apply(state, *recurrent_bf16)
        return output[:, :seq_len].to(r.dtype), terminal_state

    @staticmethod
    def _torch_recurrence(
        *,
        r: torch.Tensor,
        w_raw: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        kk: torch.Tensor,
        a: torch.Tensor,
        initial_state: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size, seq_len, heads, head_dim = r.shape
        if initial_state is None:
            state = torch.zeros(batch_size, heads, head_dim, head_dim, device=r.device, dtype=torch.float32)
        else:
            state = initial_state.float()
        outputs: List[torch.Tensor] = []
        for step in range(seq_len):
            decay = torch.exp(-math.exp(-0.5) * torch.sigmoid(w_raw[:, step].float()))
            erase = torch.einsum("bhij,bhj->bhi", state, -kk[:, step].float())
            state = (
                state * decay.unsqueeze(-2)
                + erase.unsqueeze(-1) * (kk[:, step].float() * a[:, step].float()).unsqueeze(-2)
                + v[:, step].float().unsqueeze(-1) * k[:, step].float().unsqueeze(-2)
            )
            outputs.append(torch.einsum("bhij,bhj->bhi", state, r[:, step].float()))
        return torch.stack(outputs, dim=1).to(r.dtype), state

    def forward(
        self,
        x: torch.Tensor,
        *,
        initial_state: Optional[torch.Tensor] = None,
        v_first: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size, seq_len, channels = x.shape
        shifted = torch.zeros_like(x)
        shifted[:, 1:] = x[:, :-1]
        delta = shifted - x
        xr = x + delta * self.x_r
        xw = x + delta * self.x_w
        xk = x + delta * self.x_k
        xv = x + delta * self.x_v
        xa = x + delta * self.x_a
        xg = x + delta * self.x_g

        r_flat = self.receptance(xr)
        w_raw_flat = self.w0 + torch.tanh(xw @ self.w1) @ self.w2
        k_flat = self.key(xk)
        v_flat = self.value(xv)
        if self.layer_id == 0:
            v_first = v_flat
        else:
            if v_first is None or tuple(v_first.shape) != tuple(v_flat.shape):
                raise ValueError("official RWKV7 layer > 0 requires the first layer's value tensor")
            v_flat = v_flat + (v_first - v_flat) * torch.sigmoid(self.v0 + (xv @ self.v1) @ self.v2)
        a_flat = torch.sigmoid(self.a0 + (xa @ self.a1) @ self.a2)
        gate = torch.sigmoid(xg @ self.g1) @ self.g2
        kk_flat = F.normalize(
            (k_flat * self.k_k).view(batch_size, seq_len, self.heads, self.head_dim),
            dim=-1,
            p=2.0,
        ).view(batch_size, seq_len, channels)
        k_flat = k_flat * (1.0 + (a_flat - 1.0) * self.k_a)

        recurrent = {
            "r": r_flat.view(batch_size, seq_len, self.heads, self.head_dim),
            "w_raw": w_raw_flat.view(batch_size, seq_len, self.heads, self.head_dim),
            "k": k_flat.view(batch_size, seq_len, self.heads, self.head_dim),
            "v": v_flat.view(batch_size, seq_len, self.heads, self.head_dim),
            "kk": kk_flat.view(batch_size, seq_len, self.heads, self.head_dim),
            "a": a_flat.view(batch_size, seq_len, self.heads, self.head_dim),
            "initial_state": initial_state,
        }
        if self.rwkv_kernel == "statepassing":
            recurrent_out, terminal_state = self._statepassing(**recurrent)
        else:
            recurrent_out, terminal_state = self._torch_recurrence(**recurrent)

        mixed = self.ln_x(recurrent_out.reshape(batch_size * seq_len, channels)).reshape(
            batch_size, seq_len, channels
        )
        local = (
            (
                r_flat.view(batch_size, seq_len, self.heads, self.head_dim)
                * k_flat.view(batch_size, seq_len, self.heads, self.head_dim)
                * self.r_k
            ).sum(dim=-1, keepdim=True)
            * v_flat.view(batch_size, seq_len, self.heads, self.head_dim)
        ).view(batch_size, seq_len, channels)
        assert v_first is not None
        return self.output((mixed + local) * gate), terminal_state, v_first


class ChannelMix(nn.Module):
    def __init__(self, d_model: int, mult: int) -> None:
        super().__init__()
        self.key = nn.Linear(d_model, mult * d_model, bias=False)
        self.value = nn.Linear(mult * d_model, d_model, bias=False)
        self.key.weight.data.uniform_(-0.5 / (d_model**0.5), 0.5 / (d_model**0.5))
        nn.init.zeros_(self.value.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.value(F.relu(self.key(x)).square())


class RWKV7OfficialBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        heads: int,
        head_dim: int,
        channel_mult: int,
        *,
        layer_id: int,
        layers: int,
        rwkv_kernel: str,
    ) -> None:
        super().__init__()
        self.ln_time = nn.LayerNorm(d_model)
        self.ln_channel = nn.LayerNorm(d_model)
        self.time_mix = RWKV7TimeMixOfficial(
            d_model,
            heads,
            head_dim,
            layer_id=layer_id,
            layers=layers,
            rwkv_kernel=rwkv_kernel,
        )
        self.channel_mix = ChannelMix(d_model, channel_mult)
        self.future_seed_logit = nn.Parameter(torch.zeros(1, heads, 1, 1))

    def forward(
        self,
        x: torch.Tensor,
        *,
        initial_state: Optional[torch.Tensor] = None,
        v_first: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        time_out, terminal_state, v_first = self.time_mix(
            self.ln_time(x),
            initial_state=initial_state,
            v_first=v_first,
        )
        x = x + time_out
        x = x + self.channel_mix(self.ln_channel(x))
        return x, terminal_state, v_first


class RWKVBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        heads: int,
        head_dim: int,
        channel_mult: int,
        *,
        layer_id: int,
        layers: int,
        rwkv_kernel: str,
    ) -> None:
        super().__init__()
        self.ln_time = nn.LayerNorm(d_model)
        self.ln_channel = nn.LayerNorm(d_model)
        self.time_mix = RWKVTimeMix(d_model, heads, head_dim, layer_id=layer_id, layers=layers, rwkv_kernel=rwkv_kernel)
        self.channel_mix = ChannelMix(d_model, channel_mult)
        self.future_seed_logit = nn.Parameter(torch.zeros(1, heads, 1, 1))

    def forward(
        self,
        x: torch.Tensor,
        *,
        initial_state: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        time_out, terminal_state = self.time_mix(self.ln_time(x), initial_state=initial_state)
        x = x + time_out
        x = x + self.channel_mix(self.ln_channel(x))
        return x, terminal_state


class GDNTimeMix(nn.Module):
    """FLA Gated DeltaNet token mixer with explicit recurrent state I/O."""

    def __init__(
        self,
        d_model: int,
        heads: int,
        head_dim: int,
        *,
        expand_v: float,
        progressive_base_expand_v: float,
        mode: str,
        use_short_conv: bool,
        conv_size: int,
        allow_neg_eigval: bool,
        norm_eps: float = 1e-5,
    ) -> None:
        super().__init__()
        if mode in {"chunk", "naive_recurrent"}:
            ok, reason = fla_gdn_available()
            if not ok:
                raise RuntimeError(f"BACKBONE=gdn requires flash-linear-attention: {reason}")
        elif mode == "triton_recurrent":
            ok, reason = gdn_triton_available()
            if not ok:
                raise RuntimeError(f"BACKBONE=gdn GDN_MODE=triton_recurrent requires local Triton kernel: {reason}")
            if use_short_conv:
                ok, reason = fla_gdn_available()
                if not ok:
                    raise RuntimeError(f"--gdn_use_short_conv 1 still requires flash-linear-attention: {reason}")
        else:
            raise ValueError("GDN mode must be 'chunk', 'naive_recurrent', or 'triton_recurrent'.")
        if d_model != heads * head_dim:
            raise ValueError("GDN baseline currently keeps d_model == heads * head_dim for matched state size.")
        self.d_model = d_model
        self.heads = heads
        self.head_dim = head_dim
        self.head_v_dim = int(head_dim * float(expand_v))
        if not math.isclose(float(self.head_v_dim), head_dim * float(expand_v), rel_tol=1e-5):
            raise ValueError("--gdn_expand_v must produce an integer value head dimension.")
        self.value_dim = heads * self.head_v_dim
        self.progressive_base_head_v_dim = int(head_dim * float(progressive_base_expand_v))
        if progressive_base_expand_v > 0:
            if not math.isclose(
                float(self.progressive_base_head_v_dim),
                head_dim * float(progressive_base_expand_v),
                rel_tol=1e-5,
            ):
                raise ValueError("--gdn_progressive_base_expand_v must produce an integer value head dimension.")
            if self.progressive_base_head_v_dim * 2 != self.head_v_dim:
                raise ValueError("Progressive GDN state expansion currently requires total expand_v == 2 * base expand_v.")
        self.progressive_value_dim = heads * self.progressive_base_head_v_dim
        self.progressive_expansion = self.progressive_base_head_v_dim > 0
        self.mode = mode
        self.use_short_conv = bool(use_short_conv)
        self.allow_neg_eigval = bool(allow_neg_eigval)
        self.norm_eps = float(norm_eps)

        self.q_proj = nn.Linear(d_model, heads * head_dim, bias=False)
        self.k_proj = nn.Linear(d_model, heads * head_dim, bias=False)
        projection_value_dim = self.progressive_value_dim if self.progressive_expansion else self.value_dim
        self.v_proj = nn.Linear(d_model, projection_value_dim, bias=False)
        self.a_proj = nn.Linear(d_model, heads, bias=False)
        self.b_proj = nn.Linear(d_model, heads, bias=False)
        self.g_proj = nn.Linear(d_model, projection_value_dim, bias=False)
        self.o_proj = nn.Linear(projection_value_dim, d_model, bias=False)
        self.o_norm_weight = nn.Parameter(torch.ones(self.progressive_base_head_v_dim or self.head_v_dim))
        if self.progressive_expansion:
            self.v_proj_extra = nn.Linear(d_model, self.progressive_value_dim, bias=False)
            self.g_proj_extra = nn.Linear(d_model, self.progressive_value_dim, bias=False)
            self.o_proj_extra = nn.Linear(self.progressive_value_dim, d_model, bias=False)
            self.o_norm_weight_extra = nn.Parameter(torch.ones(self.progressive_base_head_v_dim))
        else:
            self.v_proj_extra = None
            self.g_proj_extra = None
            self.o_proj_extra = None
            self.register_parameter("o_norm_weight_extra", None)

        a = torch.empty(heads, dtype=torch.float32).uniform_(1.0, 16.0)
        self.A_log = nn.Parameter(torch.log(a))
        self.A_log._no_weight_decay = True
        dt = torch.exp(torch.rand(heads, dtype=torch.float32) * (math.log(0.1) - math.log(0.001)) + math.log(0.001))
        dt = torch.clamp(dt, min=1e-4)
        self.dt_bias = nn.Parameter(dt + torch.log(-torch.expm1(-dt)))
        self.dt_bias._no_weight_decay = True

        if self.use_short_conv:
            ShortConvolution = load_fla_short_convolution()
            self.q_conv1d = ShortConvolution(heads * head_dim, kernel_size=int(conv_size), bias=False, activation="silu")
            self.k_conv1d = ShortConvolution(heads * head_dim, kernel_size=int(conv_size), bias=False, activation="silu")
            self.v_conv1d = ShortConvolution(self.value_dim, kernel_size=int(conv_size), bias=False, activation="silu")
        else:
            self.q_conv1d = None
            self.k_conv1d = None
            self.v_conv1d = None

        scale = d_model**0.5
        self.q_proj.weight.data.uniform_(-0.5 / scale, 0.5 / scale)
        self.k_proj.weight.data.uniform_(-0.5 / scale, 0.5 / scale)
        self.v_proj.weight.data.uniform_(-0.5 / scale, 0.5 / scale)
        self.a_proj.weight.data.zero_()
        self.b_proj.weight.data.zero_()
        self.g_proj.weight.data.uniform_(-0.5 / scale, 0.5 / scale)
        nn.init.zeros_(self.o_proj.weight)
        if self.progressive_expansion:
            self.initialize_progressive_from_base()

    def initialize_progressive_from_base(self) -> None:
        if not self.progressive_expansion:
            return
        assert self.v_proj_extra is not None
        assert self.g_proj_extra is not None
        assert self.o_proj_extra is not None
        assert self.o_norm_weight_extra is not None
        with torch.no_grad():
            self.v_proj_extra.weight.copy_(self.v_proj.weight)
            self.g_proj_extra.weight.copy_(self.g_proj.weight)
            self.o_proj_extra.weight.zero_()
            self.o_norm_weight_extra.copy_(self.o_norm_weight)

    def _join_progressive_values(self, base: torch.Tensor, extra: torch.Tensor) -> torch.Tensor:
        batch, seq_len, _channels = base.shape
        base = base.view(batch, seq_len, self.heads, self.progressive_base_head_v_dim)
        extra = extra.view(batch, seq_len, self.heads, self.progressive_base_head_v_dim)
        return torch.cat((base, extra), dim=-1).reshape(batch, seq_len, self.value_dim)

    def _split_progressive_values(self, values: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        batch, seq_len, _heads, _channels = values.shape
        base, extra = values.split(self.progressive_base_head_v_dim, dim=-1)
        return (
            base.reshape(batch, seq_len, self.progressive_value_dim),
            extra.reshape(batch, seq_len, self.progressive_value_dim),
        )

    def _project_qkv(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        if self.progressive_expansion:
            assert self.v_proj_extra is not None
            v = self._join_progressive_values(v, self.v_proj_extra(x))
        if self.use_short_conv:
            assert self.q_conv1d is not None and self.k_conv1d is not None and self.v_conv1d is not None
            q, _ = self.q_conv1d(q, output_final_state=False)
            k, _ = self.k_conv1d(k, output_final_state=False)
            v, _ = self.v_conv1d(v, output_final_state=False)
        else:
            q = F.silu(q)
            k = F.silu(k)
            v = F.silu(v)
        return q, k, v

    def _norm_gate(self, o: torch.Tensor, gate_source: torch.Tensor) -> torch.Tensor:
        gate = self.g_proj(gate_source)
        if self.progressive_expansion:
            assert self.g_proj_extra is not None
            gate = self._join_progressive_values(gate, self.g_proj_extra(gate_source))
        gate = gate.view(gate_source.shape[0], gate_source.shape[1], self.heads, self.head_v_dim)
        if self.progressive_expansion:
            assert self.o_norm_weight_extra is not None
            base_o, extra_o = o.split(self.progressive_base_head_v_dim, dim=-1)
            base_gate, extra_gate = gate.split(self.progressive_base_head_v_dim, dim=-1)
            base_rms = base_o.float().square().mean(dim=-1, keepdim=True).add(self.norm_eps).rsqrt().to(o.dtype)
            extra_rms = extra_o.float().square().mean(dim=-1, keepdim=True).add(self.norm_eps).rsqrt().to(o.dtype)
            base_weight = self.o_norm_weight.to(device=o.device, dtype=o.dtype).view(
                1, 1, 1, self.progressive_base_head_v_dim
            )
            extra_weight = self.o_norm_weight_extra.to(device=o.device, dtype=o.dtype).view(
                1, 1, 1, self.progressive_base_head_v_dim
            )
            return torch.cat(
                (
                    base_o * base_rms * base_weight * F.silu(base_gate),
                    extra_o * extra_rms * extra_weight * F.silu(extra_gate),
                ),
                dim=-1,
            )
        rms = o.float().square().mean(dim=-1, keepdim=True).add(self.norm_eps).rsqrt().to(o.dtype)
        weight = self.o_norm_weight.to(device=o.device, dtype=o.dtype).view(1, 1, 1, self.head_v_dim)
        return o * rms * weight * F.silu(gate)

    def forward(
        self,
        x: torch.Tensor,
        *,
        initial_state: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if not x.is_cuda:
            raise RuntimeError("BACKBONE=gdn is CUDA-only; CPU fallback is intentionally disabled.")
        q, k, v = self._project_qkv(x)
        batch_size, seq_len, _channels = x.shape
        q = q.view(batch_size, seq_len, self.heads, self.head_dim)
        k = k.view(batch_size, seq_len, self.heads, self.head_dim)
        v = v.view(batch_size, seq_len, self.heads, self.head_v_dim)
        expected = (batch_size, self.heads, self.head_v_dim, self.head_dim)
        if initial_state is not None and tuple(initial_state.shape) != expected:
            raise ValueError(f"GDN initial_state shape {tuple(initial_state.shape)} does not match {expected}")
        if self.mode == "chunk":
            o, terminal_state = chunk_gated_delta_rule(
                q=q,
                k=k,
                v=v,
                g=self.a_proj(x),
                beta=self.b_proj(x),
                initial_state=initial_state,
                output_final_state=True,
                use_qk_l2norm_in_kernel=True,
                use_gate_in_kernel=True,
                A_log=self.A_log,
                dt_bias=self.dt_bias,
                use_beta_sigmoid_in_kernel=True,
                allow_neg_eigval=self.allow_neg_eigval,
                state_v_first=True,
            )
        elif self.mode == "naive_recurrent":
            if self.allow_neg_eigval:
                raise ValueError("GDN naive_recurrent mode does not support --gdn_allow_neg_eigval.")
            assert naive_recurrent_gated_delta_rule is not None
            g_raw = self.a_proj(x).float()
            g = -self.A_log.float().exp().view(1, 1, self.heads) * F.softplus(
                g_raw + self.dt_bias.float().view(1, 1, self.heads)
            )
            beta = torch.sigmoid(self.b_proj(x).float())
            internal_initial = initial_state.transpose(-1, -2).contiguous() if initial_state is not None else None
            o, internal_terminal = naive_recurrent_gated_delta_rule(
                q=F.normalize(q.float(), dim=-1, p=2.0),
                k=F.normalize(k.float(), dim=-1, p=2.0),
                v=v.float(),
                g=g,
                beta=beta,
                initial_state=internal_initial,
                output_final_state=True,
            )
            terminal_state = internal_terminal.transpose(-1, -2).contiguous()
        else:
            assert gdn_triton_recurrent is not None
            g_raw = self.a_proj(x).float()
            g = -self.A_log.float().exp().view(1, 1, self.heads) * F.softplus(
                g_raw + self.dt_bias.float().view(1, 1, self.heads)
            )
            beta = torch.sigmoid(self.b_proj(x).float())
            if self.allow_neg_eigval:
                beta = beta * 2.0
            o, terminal_state = gdn_triton_recurrent(
                q=F.normalize(q.float(), dim=-1, p=2.0),
                k=F.normalize(k.float(), dim=-1, p=2.0),
                v=v.float(),
                g=g,
                beta=beta,
                initial_state=initial_state,
            )
        y = self._norm_gate(o, x).reshape(batch_size, seq_len, self.value_dim)
        if self.progressive_expansion:
            assert self.o_proj_extra is not None
            base_y, extra_y = self._split_progressive_values(
                y.view(batch_size, seq_len, self.heads, self.head_v_dim)
            )
            mixed = self.o_proj(base_y) + self.o_proj_extra(extra_y)
        else:
            mixed = self.o_proj(y)
        return mixed, terminal_state.to(x.dtype)


class GDNBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        heads: int,
        head_dim: int,
        channel_mult: int,
        *,
        gdn_mode: str,
        gdn_expand_v: float,
        gdn_progressive_base_expand_v: float = 0.0,
        gdn_use_short_conv: bool,
        gdn_conv_size: int,
        gdn_allow_neg_eigval: bool,
    ) -> None:
        super().__init__()
        self.ln_time = nn.LayerNorm(d_model)
        self.ln_channel = nn.LayerNorm(d_model)
        self.time_mix = GDNTimeMix(
            d_model,
            heads,
            head_dim,
            expand_v=gdn_expand_v,
            progressive_base_expand_v=gdn_progressive_base_expand_v,
            mode=gdn_mode,
            use_short_conv=gdn_use_short_conv,
            conv_size=gdn_conv_size,
            allow_neg_eigval=gdn_allow_neg_eigval,
        )
        self.channel_mix = ChannelMix(d_model, channel_mult)
        self.future_seed_logit = nn.Parameter(torch.zeros(1, heads, 1, 1))

    def forward(
        self,
        x: torch.Tensor,
        *,
        initial_state: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        time_out, terminal_state = self.time_mix(self.ln_time(x), initial_state=initial_state)
        x = x + time_out
        x = x + self.channel_mix(self.ln_channel(x))
        return x, terminal_state


class FLADeltaTimeMix(nn.Module):
    """Official FLA delta-rule layer using each layer's native cache state layout."""

    def __init__(
        self,
        d_model: int,
        heads: int,
        head_dim: int,
        *,
        backbone: str,
        expand_v: float,
        mode: str,
        use_short_conv: bool,
        conv_size: int,
        allow_neg_eigval: bool,
        gain_budget_mode: str = "none",
        gain_budget_sigma_cap: float = 1.10,
        gain_budget_step_cap: float = 1.0,
        gain_budget_sigma_cap_max: float = 3.0,
        gain_budget_infeasible_policy: str = "raise",
        fast_slow_decay_mode: str = "none",
        fast_slow_decay_kernel_size: int = 4,
        fast_slow_decay_rho_init: float = 0.10,
        fast_slow_decay_current_weight_init: float = 0.85,
        precondition_mode: str = "none",
        address_mode: str = "none",
        update_mode: str = "none",
        terminal_consolidation_enabled: bool = True,
        raven_num_slots: int = 0,
        raven_topk: int = 0,
    ) -> None:
        super().__init__()
        if backbone not in {"fla_gdn", "gdn2", "kda", "raven"}:
            raise ValueError(
                "FLA backbone must be 'fla_gdn', 'gdn2', 'kda', or 'raven'."
            )
        if mode != "chunk":
            raise ValueError(f"BACKBONE={backbone} supports only GDN_MODE=chunk during training.")
        ok, reason = fla_delta_available(backbone)
        if not ok:
            raise RuntimeError(f"BACKBONE={backbone} requires current flash-linear-attention: {reason}")
        if backbone != "fla_gdn" and d_model != heads * head_dim:
            raise ValueError(f"{backbone.upper()} keeps d_model == heads * head_dim for matched state size.")
        if backbone == "raven" and use_short_conv:
            raise ValueError("Official Raven does not use short convolution; set GDN_USE_SHORT_CONV=0.")
        if gain_budget_mode not in {
            "none",
            "external_identity",
            "fixed_sigma",
            "decay_funded",
        }:
            raise ValueError(
                "gain_budget_mode must be none, external_identity, "
                "fixed_sigma, or decay_funded"
            )
        if gain_budget_mode != "none" and backbone != "gdn2":
            raise ValueError("Gain-Budget is restricted to the official GDN2 backbone")
        if gain_budget_infeasible_policy not in {"raise", "relax"}:
            raise ValueError("gain_budget_infeasible_policy must be raise or relax")
        if gain_budget_mode != "none" and gain_budget_infeasible_policy != "raise":
            raise ValueError("Integrated Gain-Budget is fail-closed and requires policy=raise")
        if fast_slow_decay_mode not in FAST_SLOW_DECAY_MODES:
            raise ValueError(
                "fast_slow_decay_mode must be one of: "
                f"{', '.join(FAST_SLOW_DECAY_MODES)}"
            )
        if fast_slow_decay_mode != "none" and backbone != "gdn2":
            raise ValueError("Fast-Slow decay is restricted to the official GDN2 backbone")
        if fast_slow_decay_mode != "none" and gain_budget_mode != "none":
            raise ValueError("Fast-Slow decay and Gain-Budget cannot be enabled together")
        if precondition_mode not in GDN2_PRECONDITION_MODES:
            raise ValueError(
                "precondition_mode must be one of: "
                f"{', '.join(GDN2_PRECONDITION_MODES)}"
            )
        if precondition_mode != "none" and backbone != "gdn2":
            raise ValueError("Tied preconditioning is restricted to GDN2")
        if precondition_mode != "none" and (
            gain_budget_mode != "none" or fast_slow_decay_mode != "none"
        ):
            raise ValueError(
                "Tied preconditioning cannot be mixed with Gain-Budget or Fast-Slow decay"
            )
        if address_mode not in GDN2_ADDRESS_MODES:
            raise ValueError(
                f"address_mode must be one of: {', '.join(GDN2_ADDRESS_MODES)}"
            )
        if address_mode != "none" and backbone != "gdn2":
            raise ValueError("Address-payload separation is restricted to GDN2")
        if address_mode in {"anchor_rotary", "anchor_phase"} and head_dim % 2:
            raise ValueError("Anchor address rotation requires an even GDN2 head dimension")
        if address_mode != "none" and (
            gain_budget_mode != "none" or fast_slow_decay_mode != "none"
        ):
            raise ValueError(
                "Address-payload separation cannot be mixed with Gain-Budget or Fast-Slow decay"
            )
        if precondition_mode != "none" and address_mode not in {"none", "position_qk"}:
            raise ValueError(
                "Tied preconditioning currently composes only with none or position_qk addressing"
            )
        if update_mode not in GDN2_UPDATE_MODES:
            raise ValueError(
                f"update_mode must be one of: {', '.join(GDN2_UPDATE_MODES)}"
            )
        if update_mode != "none" and backbone != "gdn2":
            raise ValueError("GDN2 update extensions are restricted to GDN2")
        if update_mode != "none" and address_mode != "position_qk":
            raise ValueError(
                "GDN2 update extensions compose only with position_qk"
            )
        if update_mode != "none" and (
            gain_budget_mode != "none"
            or fast_slow_decay_mode != "none"
            or precondition_mode != "none"
        ):
            raise ValueError(
                "GDN2 update extensions cannot be mixed with Gain-Budget, "
                "Fast-Slow decay, or tied preconditioning"
            )

        self.backbone = backbone
        self.heads = int(heads)
        self.head_dim = int(head_dim)
        self.head_v_dim = int(head_dim * float(expand_v))
        if not math.isclose(float(self.head_v_dim), head_dim * float(expand_v), rel_tol=1e-5):
            raise ValueError("--gdn_expand_v must produce an integer value head dimension.")
        self.value_dim = self.heads * self.head_v_dim
        self.gain_budget_mode = gain_budget_mode
        self.gain_budget_sigma_cap = float(gain_budget_sigma_cap)
        self.gain_budget_step_cap = float(gain_budget_step_cap)
        self.gain_budget_sigma_cap_max = float(gain_budget_sigma_cap_max)
        self.gain_budget_infeasible_policy = gain_budget_infeasible_policy
        self.fast_slow_decay_mode = fast_slow_decay_mode
        self.precondition_mode = precondition_mode
        self.address_mode = address_mode
        self.update_mode = update_mode
        self.state_heads = self.heads * (2 if update_mode == "paired_address_bank" else 1)
        self.terminal_consolidation_enabled = bool(
            terminal_consolidation_enabled
        )
        if backbone == "raven":
            matched_state_elements = self.head_dim * self.head_v_dim
            slot_width = self.head_dim + self.head_v_dim
            if raven_num_slots <= 0:
                if matched_state_elements % slot_width:
                    raise ValueError(
                        "Automatic Raven state matching is not integral; set --raven_num_slots explicitly."
                    )
                raven_num_slots = matched_state_elements // slot_width
            if raven_num_slots < 1:
                raise ValueError("Raven requires at least one memory slot.")
            if raven_topk <= 0:
                raven_topk = max(1, int(round(raven_num_slots / 8)))
            if not 1 <= raven_topk <= raven_num_slots:
                raise ValueError("Raven top-k must be between 1 and the number of slots.")
        self.raven_num_slots = int(raven_num_slots)
        self.raven_topk = int(raven_topk)
        self.address_rotation_scale = (
            nn.Parameter(torch.zeros(self.heads))
            if address_mode == "anchor_rotary"
            else None
        )
        self.address_phase_proj = (
            nn.Linear(d_model, self.heads * (self.head_dim // 2), bias=False)
            if address_mode == "anchor_phase"
            else None
        )
        if self.address_phase_proj is not None:
            nn.init.zeros_(self.address_phase_proj.weight)
        self.address_residual_proj = (
            nn.Linear(d_model, d_model, bias=False)
            if address_mode in {"anchor_residual", "anchor_carrier"}
            else None
        )
        if self.address_residual_proj is not None:
            nn.init.zeros_(self.address_residual_proj.weight)
        self.address_q_residual_proj = (
            nn.Linear(d_model, d_model, bias=False)
            if address_mode == "anchor_qk_residual"
            else None
        )
        self.address_k_residual_proj = (
            nn.Linear(d_model, d_model, bias=False)
            if address_mode == "anchor_qk_residual"
            else None
        )
        if self.address_q_residual_proj is not None:
            nn.init.zeros_(self.address_q_residual_proj.weight)
        if self.address_k_residual_proj is not None:
            nn.init.zeros_(self.address_k_residual_proj.weight)
        self.address_carrier_scale = (
            nn.Parameter(torch.zeros(self.heads, self.head_dim))
            if address_mode == "anchor_carrier"
            else None
        )
        self.address_carrier_bias_delta = (
            nn.Parameter(torch.zeros(self.heads, self.head_dim))
            if address_mode == "anchor_carrier"
            else None
        )
        self.address_carrier_base_logit = 6.0
        self.coherent_delta_mix = (
            nn.Parameter(torch.zeros(self.heads))
            if update_mode == "coherent_delta"
            else None
        )
        if self.coherent_delta_mix is not None:
            self.coherent_delta_mix._no_weight_decay = True
        self.log_spd_address_metric = (
            BoundedLogSPDAddressMetric(self.heads, self.head_dim)
            if update_mode == "log_spd_metric"
            else None
        )
        feedback_hidden = max(8, self.head_v_dim // 2)
        self.state_feedback_in = (
            nn.Linear(self.head_v_dim, feedback_hidden, bias=False)
            if update_mode == "state_feedback"
            else None
        )
        self.state_feedback_out = (
            nn.Linear(
                feedback_hidden,
                2 * self.head_dim + 2 * self.head_v_dim,
                bias=False,
            )
            if update_mode == "state_feedback"
            else None
        )
        if self.state_feedback_out is not None:
            nn.init.zeros_(self.state_feedback_out.weight)
        self.terminal_consolidation_k_proj = (
            nn.Linear(self.head_v_dim, self.head_dim, bias=False)
            if update_mode == "terminal_consolidation"
            and self.terminal_consolidation_enabled
            else None
        )
        if self.terminal_consolidation_k_proj is not None:
            nn.init.zeros_(self.terminal_consolidation_k_proj.weight)
        self.orthogonal_chunk_state_proj = (
            nn.Linear(
                self.head_v_dim,
                2 * self.head_dim + 1,
                bias=False,
            )
            if update_mode == "orthogonal_chunk_state"
            else None
        )
        if self.orthogonal_chunk_state_proj is not None:
            with torch.no_grad():
                self.orthogonal_chunk_state_proj.weight[-1:].zero_()
        self.orthogonal_head_write_proj = (
            nn.Linear(self.head_v_dim, 3, bias=False)
            if update_mode == "orthogonal_head_write"
            else None
        )
        if self.orthogonal_head_write_proj is not None:
            with torch.no_grad():
                self.orthogonal_head_write_proj.weight[-1:].zero_()
        self.adaptive_signed_erase_proj = (
            nn.Linear(self.head_v_dim, self.head_dim, bias=False)
            if update_mode == "adaptive_signed_erase"
            else None
        )
        if self.adaptive_signed_erase_proj is not None:
            nn.init.zeros_(self.adaptive_signed_erase_proj.weight)
        self.bi_axis_value_decay_groups = 8
        self.bi_axis_value_decay_potential_cap = math.log(4.0)
        if (
            update_mode in {"bi_axis_value_decay", "gauge_balanced_bi_axis"}
            and self.head_v_dim % self.bi_axis_value_decay_groups
        ):
            raise ValueError(
                "Bi-Axis value decay requires V width divisible by eight"
            )
        self.bi_axis_value_decay_proj = (
            nn.Linear(
                d_model,
                self.heads * self.bi_axis_value_decay_groups,
                bias=False,
            )
            if update_mode in {"bi_axis_value_decay", "gauge_balanced_bi_axis"}
            else None
        )
        if self.bi_axis_value_decay_proj is not None:
            nn.init.zeros_(self.bi_axis_value_decay_proj.weight)
        self.raven_routed_slots = 8
        if (
            update_mode == "raven_routed_gdn"
            and self.head_dim % self.raven_routed_slots
        ):
            raise ValueError(
                "Raven-routed GDN requires K width divisible by eight"
            )
        self.raven_route_proj = (
            nn.Linear(
                d_model,
                self.heads * self.raven_routed_slots,
                bias=False,
            )
            if update_mode == "raven_routed_gdn"
            else None
        )
        if self.raven_route_proj is not None:
            nn.init.zeros_(self.raven_route_proj.weight)
        self.paired_address_q_proj = (
            nn.Linear(self.head_v_dim, self.head_dim, bias=False)
            if update_mode == "paired_address_bank"
            else None
        )
        self.paired_address_k_proj = (
            nn.Linear(self.head_v_dim, self.head_dim, bias=False)
            if update_mode == "paired_address_bank"
            else None
        )
        self.paired_address_read_gate = (
            nn.Parameter(torch.zeros(self.heads))
            if update_mode == "paired_address_bank"
            else None
        )
        if self.paired_address_q_proj is not None:
            nn.init.zeros_(self.paired_address_q_proj.weight)
        if self.paired_address_k_proj is not None:
            nn.init.zeros_(self.paired_address_k_proj.weight)
        if self.paired_address_read_gate is not None:
            self.paired_address_read_gate._no_weight_decay = True
        self.coupled_address_k_proj = (
            nn.Linear(d_model, self.heads * self.head_dim, bias=False)
            if update_mode == "coupled_address_rows"
            else None
        )
        if self.coupled_address_k_proj is not None:
            nn.init.zeros_(self.coupled_address_k_proj.weight)
        if update_mode == "interleaved_write" and self.value_dim != d_model:
            raise ValueError(
                "Interleaved write requires matched K/V width (gdn_expand_v=1)"
            )
        self.interleaved_write_k_proj = (
            nn.Linear(d_model, d_model, bias=False)
            if update_mode == "interleaved_write"
            else None
        )
        self.interleaved_write_v_proj = (
            nn.Linear(d_model, self.value_dim, bias=False)
            if update_mode == "interleaved_write"
            else None
        )
        if self.interleaved_write_k_proj is not None:
            nn.init.zeros_(self.interleaved_write_k_proj.weight)
        if self.interleaved_write_v_proj is not None:
            nn.init.zeros_(self.interleaved_write_v_proj.weight)
        self.last_gain_budget_diag: Dict[str, torch.Tensor] = {}
        # Official GDN/KDA layers request V-first states from their kernels;
        # official GDN2 currently keeps its default K-first cache layout.
        self.state_v_first = backbone != "gdn2"

        layer_types = {
            "fla_gdn": FLAGatedDeltaNet,
            "gdn2": GatedDeltaNet2,
            "kda": KimiDeltaAttention,
            "raven": FLARaven,
        }
        layer_type = layer_types[backbone]
        assert layer_type is not None
        if backbone == "raven":
            layer_kwargs = dict(
                hidden_size=d_model,
                expand_k=1.0,
                expand_v=expand_v,
                num_heads=heads,
                num_kv_heads=heads,
                num_slots=self.raven_num_slots,
                topk=self.raven_topk,
                mode="chunk",
                decay_type="Mamba2",
                add_gumbel_noise=True,
                router_score="sigmoid",
                router_type="lin",
                use_rope=False,
                use_short_conv=False,
                layer_idx=0,
            )
        else:
            layer_kwargs = dict(
                hidden_size=d_model,
                expand_v=expand_v,
                head_dim=head_dim,
                num_heads=heads,
                num_v_heads=heads,
                mode="chunk",
                use_short_conv=bool(use_short_conv),
                allow_neg_eigval=bool(allow_neg_eigval),
                conv_size=int(conv_size),
                conv_bias=False,
                layer_idx=0,
            )
            if backbone == "fla_gdn":
                layer_kwargs["use_gate"] = True
        self.core = layer_type(**layer_kwargs)
        self.fast_slow_decay = (
            FastSlowDecayController(
                heads=heads,
                kernel_size=fast_slow_decay_kernel_size,
                rho_init=fast_slow_decay_rho_init,
                current_weight_init=fast_slow_decay_current_weight_init,
            )
            if fast_slow_decay_mode != "none"
            else None
        )

    @staticmethod
    def _zero_address_diag(x: torch.Tensor) -> Dict[str, torch.Tensor]:
        zero = x.new_zeros(())
        return {
            "gdn2_address_enabled": zero,
            "gdn2_address_qk_diag_cosine": zero,
            "gdn2_address_qk_offdiag_cosine": zero,
            "gdn2_address_qk_contrast": zero,
            "gdn2_address_order_displacement": zero,
            "gdn2_address_rotation_scale_abs": zero,
            "gdn2_address_phase_abs": zero,
            "gdn2_address_q_relative_change": zero,
            "gdn2_address_k_relative_change": zero,
            "gdn2_address_q_norm_error": zero,
            "gdn2_address_k_norm_error": zero,
            "gdn2_address_phase_weight_rms": zero,
            "gdn2_address_phase_token_std": zero,
            "gdn2_address_phase_plane_std": zero,
            "gdn2_address_residual_weight_rms": zero,
            "gdn2_address_residual_rms": zero,
            "gdn2_address_residual_token_std": zero,
            "gdn2_address_residual_q_ratio": zero,
            "gdn2_address_residual_k_ratio": zero,
            "gdn2_address_q_residual_weight_rms": zero,
            "gdn2_address_k_residual_weight_rms": zero,
            "gdn2_address_q_residual_rms": zero,
            "gdn2_address_k_residual_rms": zero,
            "gdn2_address_terminal_state_rms": zero,
            "gdn2_address_carrier_mean": zero,
            "gdn2_address_carrier_std": zero,
            "gdn2_address_carrier_token_std": zero,
            "gdn2_address_carrier_min": zero,
            "gdn2_address_carrier_below_095_frac": zero,
            "gdn2_address_carrier_scale_rms": zero,
            "gdn2_address_carrier_bias_delta_rms": zero,
            "gdn2_address_carrier_write_norm_ratio": zero,
            "gdn2_address_carrier_b_input_relative_change": zero,
            "gdn2_address_carrier_w_input_relative_change": zero,
            "gdn3_log_spd_enabled": zero,
            "gdn3_log_spd_raw_rms": zero,
            "gdn3_log_spd_metric_delta_fro": zero,
            "gdn3_log_spd_eigenvalue_min": x.new_ones(()),
            "gdn3_log_spd_eigenvalue_max": x.new_ones(()),
            "gdn3_log_spd_condition_max": x.new_ones(()),
            "gdn3_log_spd_logdet_abs_max": zero,
        }

    @staticmethod
    def _zero_precondition_diag(x: torch.Tensor) -> Dict[str, torch.Tensor]:
        zero = x.new_zeros((), dtype=torch.float32)
        return {key: zero for key in PRECONDITION_TRAIN_KEYS}

    @staticmethod
    def _zero_coherent_delta_diag(x: torch.Tensor) -> Dict[str, torch.Tensor]:
        zero = x.new_zeros((), dtype=torch.float32)
        values = {key: zero for key in COHERENT_DELTA_TRAIN_KEYS}
        values["gdn2_coherent_delta_gap_ratio"] = x.new_ones(
            (), dtype=torch.float32
        )
        return values

    @staticmethod
    def _zero_state_feedback_diag(x: torch.Tensor) -> Dict[str, torch.Tensor]:
        zero = x.new_zeros((), dtype=torch.float32)
        return {key: zero for key in STATE_FEEDBACK_TRAIN_KEYS}

    @staticmethod
    def _zero_terminal_consolidation_diag(
        x: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        zero = x.new_zeros((), dtype=torch.float32)
        return {key: zero for key in TERMINAL_CONSOLIDATION_TRAIN_KEYS}

    @staticmethod
    def _zero_orthogonal_chunk_state_diag(
        x: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        zero = x.new_zeros((), dtype=torch.float32)
        values = {key: zero for key in ORTHOGONAL_CHUNK_STATE_TRAIN_KEYS}
        values["gdn3_orthogonal_chunk_state_boundary_norm_ratio_mean"] = (
            x.new_ones((), dtype=torch.float32)
        )
        return values

    @staticmethod
    def _zero_orthogonal_head_write_diag(
        x: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        zero = x.new_zeros((), dtype=torch.float32)
        values = {key: zero for key in ORTHOGONAL_HEAD_WRITE_TRAIN_KEYS}
        values["gdn3_orthogonal_head_write_fp32_norm_ratio_mean"] = (
            x.new_ones((), dtype=torch.float32)
        )
        values["gdn3_orthogonal_head_write_storage_norm_ratio_mean"] = (
            x.new_ones((), dtype=torch.float32)
        )
        return values

    @staticmethod
    def _zero_adaptive_signed_erase_diag(
        x: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        zero = x.new_zeros((), dtype=torch.float32)
        return {key: zero for key in ADAPTIVE_SIGNED_ERASE_TRAIN_KEYS}

    @staticmethod
    def _zero_bi_axis_value_decay_diag(
        x: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        zero = x.new_zeros((), dtype=torch.float32)
        values = {key: zero for key in BI_AXIS_VALUE_DECAY_TRAIN_KEYS}
        for key in (
            "gdn3_bi_axis_value_decay_cumulative_scale_min",
            "gdn3_bi_axis_value_decay_cumulative_scale_mean",
            "gdn3_bi_axis_value_decay_cumulative_scale_max",
            "gdn3_bi_axis_value_decay_inverse_scale_max",
        ):
            values[key] = x.new_ones((), dtype=torch.float32)
        return values

    @staticmethod
    def _zero_raven_routed_gdn_diag(
        x: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        zero = x.new_zeros((), dtype=torch.float32)
        values = {key: zero for key in RAVEN_ROUTED_GDN_TRAIN_KEYS}
        values["gdn3_raven_routed_allocation_entropy_normalized"] = (
            x.new_ones((), dtype=torch.float32)
        )
        values["gdn3_raven_routed_allocation_min"] = x.new_ones(
            (), dtype=torch.float32
        )
        values["gdn3_raven_routed_allocation_max"] = x.new_ones(
            (), dtype=torch.float32
        )
        return values

    def _raven_routed_gdn_update(
        self,
        x: torch.Tensor,
        key: torch.Tensor,
        log_decay: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        projection = self.raven_route_proj
        if self.update_mode != "raven_routed_gdn" or projection is None:
            return key, log_decay, self._zero_raven_routed_gdn_diag(x)
        if key.shape != log_decay.shape or key.ndim != 4:
            raise RuntimeError(
                "Raven-routed GDN requires matched rank-4 K and decay tensors"
            )
        if key.shape[-2:] != (self.heads, self.head_dim):
            raise RuntimeError(
                "Raven-routed GDN received unexpected K geometry: "
                f"{tuple(key.shape)}"
            )

        batch_size, seq_len, _channels = x.shape
        slots = self.raven_routed_slots
        slot_width = self.head_dim // slots
        with torch.autocast(device_type=x.device.type, enabled=False):
            logits = F.linear(x.float(), projection.weight.float()).view(
                batch_size,
                seq_len,
                self.heads,
                slots,
            )
            probabilities = logits.softmax(dim=-1)
            allocation = probabilities * float(slots)
            row_allocation = torch.repeat_interleave(
                allocation,
                slot_width,
                dim=-1,
            )
            routed_key_float = key.float() * row_allocation.sqrt()
            routed_decay_float = log_decay.float() * row_allocation

        routed_key = routed_key_float.to(dtype=key.dtype)
        routed_decay = routed_decay_float.to(dtype=log_decay.dtype)
        with torch.no_grad():
            key_float = key.float()
            decay_float = log_decay.float()
            allocation_delta = allocation - 1.0
            allocation_board = allocation_delta.square().mean(
                dim=(1, 2, 3)
            ).sqrt()
            allocation_token = allocation_delta.square().mean(
                dim=(0, 2, 3)
            ).sqrt()
            allocation_slot = allocation.mean(dim=(0, 1, 2))
            entropy = -(
                probabilities.clamp_min(1e-12)
                * probabilities.clamp_min(1e-12).log()
            ).sum(dim=-1) / math.log(float(slots))
            diagnostics = {
                "gdn3_raven_routed_enabled": x.new_ones(
                    (), dtype=torch.float32
                ),
                "gdn3_raven_routed_allocation_abs_from_one": (
                    allocation - 1.0
                ).abs().mean(),
                "gdn3_raven_routed_allocation_slot_std": (
                    allocation_slot.std(unbiased=False)
                ),
                "gdn3_raven_routed_allocation_batch_std": (
                    allocation_board.std(unbiased=False)
                ),
                "gdn3_raven_routed_allocation_token_std": (
                    allocation_token.std(unbiased=False)
                ),
                "gdn3_raven_routed_allocation_entropy_normalized": (
                    entropy.mean()
                ),
                "gdn3_raven_routed_allocation_min": allocation.min(),
                "gdn3_raven_routed_allocation_max": allocation.max(),
                "gdn3_raven_routed_k_relative_change": (
                    (routed_key_float - key_float).square().mean().sqrt()
                    / key_float.square().mean().sqrt().clamp_min(1e-6)
                ),
                "gdn3_raven_routed_g_relative_change": (
                    (routed_decay_float - decay_float).square().mean().sqrt()
                    / decay_float.square().mean().sqrt().clamp_min(1e-6)
                ),
                "gdn3_raven_routed_router_weight_rms": (
                    projection.weight.float().square().mean().sqrt()
                ),
                "gdn3_raven_routed_terminal_rms": x.new_zeros(
                    (), dtype=torch.float32
                ),
                "gdn3_raven_routed_terminal_batch_std": x.new_zeros(
                    (), dtype=torch.float32
                ),
            }
        return routed_key, routed_decay, diagnostics

    @staticmethod
    def _zero_paired_address_bank_diag(
        x: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        zero = x.new_zeros((), dtype=torch.float32)
        return {key: zero for key in PAIRED_ADDRESS_BANK_TRAIN_KEYS}

    @staticmethod
    def _zero_coupled_address_rows_diag(
        x: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        zero = x.new_zeros((), dtype=torch.float32)
        return {key: zero for key in COUPLED_ADDRESS_ROWS_TRAIN_KEYS}

    @staticmethod
    def _zero_interleaved_write_diag(
        x: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        zero = x.new_zeros((), dtype=torch.float32)
        return {key: zero for key in INTERLEAVED_WRITE_TRAIN_KEYS}

    def _orthogonal_head_write_route(
        self,
        value: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        projection = self.orthogonal_head_write_proj
        if self.update_mode != "orthogonal_head_write" or projection is None:
            return value, self._zero_orthogonal_head_write_diag(value)
        if value.ndim != 4 or value.shape[-2:] != (
            self.heads,
            self.head_v_dim,
        ):
            raise RuntimeError(
                "Orthogonal head-write routing requires V shaped "
                f"[B,T,{self.heads},{self.head_v_dim}], got "
                f"{tuple(value.shape)}"
            )

        value_float = value.float()
        controls = F.linear(
            value_float,
            projection.weight.float(),
        )
        u_raw, v_raw, angle_raw = controls.unbind(dim=-1)
        u = F.normalize(u_raw, dim=-1, eps=1e-6)
        v_residual = v_raw - (u * v_raw).sum(dim=-1, keepdim=True) * u
        v_plane = F.normalize(v_residual, dim=-1, eps=1e-6)
        theta = math.pi * torch.tanh(angle_raw.mean(dim=-1))

        u_component = torch.einsum("bth,bthv->btv", u, value_float)
        v_component = torch.einsum("bth,bthv->btv", v_plane, value_float)
        cosine_delta = (torch.cos(theta) - 1.0).unsqueeze(-1)
        sine = torch.sin(theta).unsqueeze(-1)
        u_delta = cosine_delta * u_component - sine * v_component
        v_delta = sine * u_component + cosine_delta * v_component
        routed_float = (
            value_float
            + torch.einsum("bth,btv->bthv", u, u_delta)
            + torch.einsum("bth,btv->bthv", v_plane, v_delta)
        )
        routed = routed_float.to(dtype=value.dtype)

        with torch.no_grad():
            residual = routed_float - value_float
            value_board_rms = value_float.square().mean(
                dim=(1, 2, 3)
            ).sqrt().clamp_min(1e-6)
            residual_board_relative = residual.square().mean(
                dim=(1, 2, 3)
            ).sqrt() / value_board_rms
            value_token_rms = value_float.square().mean(
                dim=(0, 2, 3)
            ).sqrt().clamp_min(1e-6)
            residual_token_relative = residual.square().mean(
                dim=(0, 2, 3)
            ).sqrt() / value_token_rms
            routed_board_rms = routed_float.square().mean(
                dim=(1, 2, 3)
            ).sqrt()
            storage_board_rms = routed.float().square().mean(
                dim=(1, 2, 3)
            ).sqrt()
            fp32_norm_ratio = routed_board_rms / value_board_rms
            storage_norm_ratio = storage_board_rms / value_board_rms
            u_norm_error = (u.square().sum(dim=-1) - 1.0).abs()
            v_norm_error = (v_plane.square().sum(dim=-1) - 1.0).abs()
            plane_dot_abs = (u * v_plane).sum(dim=-1).abs()
            angle_weight = projection.weight[-1:].float()
            plane_weight = projection.weight[:-1].float()
            diagnostics = {
                "gdn3_orthogonal_head_write_enabled": value.new_ones(
                    (), dtype=torch.float32
                ),
                "gdn3_orthogonal_head_write_angle_abs": theta.abs().mean(),
                "gdn3_orthogonal_head_write_angle_batch_std": theta.abs().mean(
                    dim=1
                ).std(unbiased=False),
                "gdn3_orthogonal_head_write_angle_token_std": theta.abs().mean(
                    dim=0
                ).std(unbiased=False),
                "gdn3_orthogonal_head_write_v_residual_relative_rms": (
                    residual_board_relative.mean()
                ),
                "gdn3_orthogonal_head_write_v_residual_batch_std": (
                    residual_board_relative.std(unbiased=False)
                ),
                "gdn3_orthogonal_head_write_v_residual_token_std": (
                    residual_token_relative.std(unbiased=False)
                ),
                "gdn3_orthogonal_head_write_plane_dot_abs_max": (
                    plane_dot_abs.max()
                ),
                "gdn3_orthogonal_head_write_plane_norm_error_max": torch.maximum(
                    u_norm_error.max(), v_norm_error.max()
                ),
                "gdn3_orthogonal_head_write_fp32_norm_ratio_mean": (
                    fp32_norm_ratio.mean()
                ),
                "gdn3_orthogonal_head_write_fp32_norm_ratio_max_error": (
                    (fp32_norm_ratio - 1.0).abs().max()
                ),
                "gdn3_orthogonal_head_write_storage_norm_ratio_mean": (
                    storage_norm_ratio.mean()
                ),
                "gdn3_orthogonal_head_write_storage_norm_ratio_max_error": (
                    (storage_norm_ratio - 1.0).abs().max()
                ),
                "gdn3_orthogonal_head_write_terminal_rms": value.new_zeros(
                    (), dtype=torch.float32
                ),
                "gdn3_orthogonal_head_write_terminal_batch_std": value.new_zeros(
                    (), dtype=torch.float32
                ),
                "gdn3_orthogonal_head_write_angle_weight_rms": (
                    angle_weight.square().mean().sqrt()
                ),
                "gdn3_orthogonal_head_write_plane_weight_rms": (
                    plane_weight.square().mean().sqrt()
                ),
            }
        return routed, diagnostics

    def _adaptive_signed_erase(
        self,
        value: torch.Tensor,
        erase: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        projection = self.adaptive_signed_erase_proj
        if self.update_mode != "adaptive_signed_erase" or projection is None:
            return erase, self._zero_adaptive_signed_erase_diag(value)
        if value.ndim != 4 or erase.ndim != 4:
            raise RuntimeError(
                "Adaptive signed erase requires rank-4 V and erase tensors"
            )
        if value.shape[:-1] != erase.shape[:-1]:
            raise RuntimeError(
                "Adaptive signed erase V/erase prefix mismatch: "
                f"{tuple(value.shape)} != {tuple(erase.shape)}"
            )
        if value.shape[-1] != self.head_v_dim or erase.shape[-1] != self.head_dim:
            raise RuntimeError(
                "Adaptive signed erase requires V32->K32 per-head tensors, got "
                f"V={tuple(value.shape)} erase={tuple(erase.shape)}"
            )

        residual = torch.tanh(projection(value))
        candidate = erase + residual
        effective = candidate.clamp(min=0.0, max=2.0)

        with torch.no_grad():
            applied = effective.float() - erase.float()
            erase_float = erase.float()
            erase_board_rms = erase_float.square().mean(
                dim=(1, 2, 3)
            ).sqrt().clamp_min(1e-6)
            applied_board_rms = applied.square().mean(dim=(1, 2, 3)).sqrt()
            applied_board_relative = applied_board_rms / erase_board_rms
            residual_token_abs = applied.abs().mean(dim=(0, 2, 3))
            residual_head_abs = applied.abs().mean(dim=(0, 1, 3))
            diagnostics = {
                "gdn3_adaptive_signed_erase_enabled": value.new_ones(
                    (), dtype=torch.float32
                ),
                "gdn3_adaptive_signed_erase_residual_abs": applied.abs().mean(),
                "gdn3_adaptive_signed_erase_residual_relative_rms": (
                    applied_board_relative.mean()
                ),
                "gdn3_adaptive_signed_erase_residual_batch_std": (
                    applied_board_relative.std(unbiased=False)
                ),
                "gdn3_adaptive_signed_erase_residual_token_std": (
                    residual_token_abs.std(unbiased=False)
                ),
                "gdn3_adaptive_signed_erase_residual_head_std": (
                    residual_head_abs.std(unbiased=False)
                ),
                "gdn3_adaptive_signed_erase_b_relative_change": (
                    applied.square().mean().sqrt()
                    / erase_float.square().mean().sqrt().clamp_min(1e-6)
                ),
                "gdn3_adaptive_signed_erase_effective_mean": effective.float().mean(),
                "gdn3_adaptive_signed_erase_effective_min": effective.float().min(),
                "gdn3_adaptive_signed_erase_effective_max": effective.float().max(),
                "gdn3_adaptive_signed_erase_above_one_frac": (
                    effective.float() > 1.0
                ).float().mean(),
                "gdn3_adaptive_signed_erase_clipped_low_frac": (
                    candidate.float() < 0.0
                ).float().mean(),
                "gdn3_adaptive_signed_erase_clipped_high_frac": (
                    candidate.float() > 2.0
                ).float().mean(),
                "gdn3_adaptive_signed_erase_terminal_rms": value.new_zeros(
                    (), dtype=torch.float32
                ),
                "gdn3_adaptive_signed_erase_terminal_batch_std": value.new_zeros(
                    (), dtype=torch.float32
                ),
                "gdn3_adaptive_signed_erase_weight_rms": (
                    projection.weight.float().square().mean().sqrt()
                ),
            }
        return effective, diagnostics

    def _bi_axis_value_decay_transition(
        self,
        operation: Any,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor,
        b: torch.Tensor,
        w: torch.Tensor,
        x: torch.Tensor,
        initial_state: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        projection = self.bi_axis_value_decay_proj
        if (
            self.update_mode
            not in {"bi_axis_value_decay", "gauge_balanced_bi_axis"}
            or projection is None
        ):
            output, terminal_state = operation(
                q=q,
                k=k,
                v=v,
                g=g,
                b=b,
                w=w,
                initial_state=initial_state,
                output_final_state=True,
                use_qk_l2norm_in_kernel=True,
            )
            return output, terminal_state, self._zero_bi_axis_value_decay_diag(x)

        batch_size, seq_len, _channels = x.shape
        groups = self.bi_axis_value_decay_groups
        group_width = self.head_v_dim // groups
        with torch.autocast(device_type=x.device.type, enabled=False):
            raw = F.linear(x.float(), projection.weight.float()).view(
                batch_size,
                seq_len,
                self.heads,
                groups,
            )
            if self.update_mode == "bi_axis_value_decay":
                softplus_zero = F.softplus(raw.new_zeros(()))
                group_log_decay = torch.clamp(
                    softplus_zero - F.softplus(raw),
                    max=0.0,
                )
                value_frame_log = torch.repeat_interleave(
                    group_log_decay,
                    group_width,
                    dim=-1,
                ).cumsum(dim=1)
                group_potential = torch.zeros_like(group_log_decay)
                common_k_log_decay = raw.new_zeros(
                    batch_size,
                    seq_len,
                    self.heads,
                    1,
                )
                operation_g = g
            else:
                group_potential = (
                    self.bi_axis_value_decay_potential_cap * torch.tanh(raw)
                )
                previous_potential = torch.cat(
                    (
                        torch.zeros_like(group_potential[:, :1]),
                        group_potential[:, :-1],
                    ),
                    dim=1,
                )
                potential_delta = group_potential - previous_potential
                common_k_log_decay = torch.relu(
                    potential_delta.amax(dim=-1, keepdim=True)
                )
                group_log_decay = potential_delta - common_k_log_decay
                value_frame_log = torch.repeat_interleave(
                    group_potential,
                    group_width,
                    dim=-1,
                )
                operation_g = (
                    g.float() - common_k_log_decay
                ).to(dtype=g.dtype)
            cumulative_scale = value_frame_log.exp()
            inverse_scale = (-value_frame_log).exp()
            transformed_value = (v.float() * inverse_scale).to(dtype=v.dtype)

        transformed_output, transformed_terminal = operation(
            q=q,
            k=k,
            v=transformed_value,
            g=operation_g,
            b=b,
            w=w,
            initial_state=initial_state,
            output_final_state=True,
            use_qk_l2norm_in_kernel=True,
        )
        with torch.autocast(device_type=x.device.type, enabled=False):
            output = (
                transformed_output.float() * cumulative_scale
            ).to(dtype=transformed_output.dtype)
            terminal_scale = cumulative_scale[:, -1].unsqueeze(-2)
            terminal_state = (
                transformed_terminal.float() * terminal_scale
            ).to(dtype=transformed_terminal.dtype)

        with torch.no_grad():
            log_decay_abs = group_log_decay.abs()
            board_mean = log_decay_abs.mean(dim=(1, 2, 3))
            token_mean = log_decay_abs.mean(dim=(0, 2, 3))
            group_mean = log_decay_abs.mean(dim=(0, 1, 2))
            value_float = v.float()
            transformed_value_float = transformed_value.float()
            transformed_output_float = transformed_output.float()
            transformed_terminal_float = transformed_terminal.float()
            terminal_board_rms = terminal_state.float().square().mean(
                dim=(1, 2, 3)
            ).sqrt()
            diagnostics = {
                "gdn3_bi_axis_value_decay_enabled": x.new_ones(
                    (), dtype=torch.float32
                ),
                "gdn3_bi_axis_value_decay_potential_abs": (
                    group_potential.abs().mean()
                ),
                "gdn3_bi_axis_value_decay_common_k_log_decay_abs": (
                    common_k_log_decay.abs().mean()
                ),
                "gdn3_bi_axis_value_decay_log_decay_abs": log_decay_abs.mean(),
                "gdn3_bi_axis_value_decay_active_frac": (
                    log_decay_abs > 0
                ).float().mean(),
                "gdn3_bi_axis_value_decay_group_std": group_mean.std(
                    unbiased=False
                ),
                "gdn3_bi_axis_value_decay_batch_std": board_mean.std(
                    unbiased=False
                ),
                "gdn3_bi_axis_value_decay_token_std": token_mean.std(
                    unbiased=False
                ),
                "gdn3_bi_axis_value_decay_cumulative_scale_min": (
                    cumulative_scale.min()
                ),
                "gdn3_bi_axis_value_decay_cumulative_scale_mean": (
                    cumulative_scale.mean()
                ),
                "gdn3_bi_axis_value_decay_cumulative_scale_max": (
                    cumulative_scale.max()
                ),
                "gdn3_bi_axis_value_decay_inverse_scale_max": inverse_scale.max(),
                "gdn3_bi_axis_value_decay_write_frame_relative_rms": (
                    (transformed_value_float - value_float).square().mean().sqrt()
                    / value_float.square().mean().sqrt().clamp_min(1e-6)
                ),
                "gdn3_bi_axis_value_decay_output_restore_relative_rms": (
                    (output.float() - transformed_output_float)
                    .square()
                    .mean()
                    .sqrt()
                    / transformed_output_float.square().mean().sqrt().clamp_min(1e-6)
                ),
                "gdn3_bi_axis_value_decay_state_restore_relative_rms": (
                    (terminal_state.float() - transformed_terminal_float)
                    .square()
                    .mean()
                    .sqrt()
                    / transformed_terminal_float.square().mean().sqrt().clamp_min(1e-6)
                ),
                "gdn3_bi_axis_value_decay_terminal_rms": terminal_board_rms.mean(),
                "gdn3_bi_axis_value_decay_terminal_batch_std": (
                    terminal_board_rms.std(unbiased=False)
                ),
                "gdn3_bi_axis_value_decay_weight_rms": (
                    projection.weight.float().square().mean().sqrt()
                ),
            }
        return output, terminal_state, diagnostics

    def _paired_address_bank_transition(
        self,
        operation: Any,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor,
        b: torch.Tensor,
        w: torch.Tensor,
        initial_state: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        q_projection = self.paired_address_q_proj
        k_projection = self.paired_address_k_proj
        read_gate_parameter = self.paired_address_read_gate
        if (
            self.update_mode != "paired_address_bank"
            or q_projection is None
            or k_projection is None
            or read_gate_parameter is None
        ):
            raise RuntimeError("Paired address-state bank parameters are unavailable")
        if operation is None:
            raise RuntimeError("Paired address-state bank requires the official GDN2 op")
        expected_prefix = q.shape[:3]
        if (
            q.ndim != 4
            or k.shape != q.shape
            or v.ndim != 4
            or g.shape != q.shape
            or b.shape != q.shape
            or v.shape[:3] != expected_prefix
            or w.shape != v.shape
        ):
            raise RuntimeError(
                "Paired address-state bank requires matched rank-4 Q/K/V/g/b/w tensors"
            )
        if q.shape[2] != self.heads or v.shape[2] != self.heads:
            raise RuntimeError(
                "Paired address-state bank requires one K/V stream per parent head"
            )
        if q.shape[-1] != self.head_dim or v.shape[-1] != self.head_v_dim:
            raise RuntimeError("Paired address-state bank received unexpected head dimensions")

        batch_size = q.shape[0]
        base_state_shape = (
            batch_size,
            self.heads,
            self.head_dim,
            self.head_v_dim,
        )
        paired_state_shape = (
            batch_size,
            2 * self.heads,
            self.head_dim,
            self.head_v_dim,
        )
        paired_initial_state: Optional[torch.Tensor]
        if initial_state is None:
            paired_initial_state = None
        elif tuple(initial_state.shape) == base_state_shape:
            paired_initial_state = torch.cat((initial_state, initial_state), dim=1)
        elif tuple(initial_state.shape) == paired_state_shape:
            paired_initial_state = initial_state
        else:
            raise RuntimeError(
                "Paired address-state bank incoming state mismatch: "
                f"{tuple(initial_state.shape)} not in "
                f"{{{base_state_shape}, {paired_state_shape}}}"
            )

        q_residual = q_projection(v)
        k_residual = k_projection(v)
        companion_q = q + q_residual.to(dtype=q.dtype)
        companion_k = k + k_residual.to(dtype=k.dtype)
        paired_q = torch.cat((q, companion_q), dim=2)
        paired_k = torch.cat((k, companion_k), dim=2)
        paired_v = torch.cat((v, v), dim=2)
        paired_g = torch.cat((g, g), dim=2)
        paired_b = torch.cat((b, b), dim=2)
        paired_w = torch.cat((w, w), dim=2)
        paired_output, terminal_state = operation(
            q=paired_q,
            k=paired_k,
            v=paired_v,
            g=paired_g,
            b=paired_b,
            w=paired_w,
            initial_state=paired_initial_state,
            output_final_state=True,
            use_qk_l2norm_in_kernel=True,
        )
        if tuple(terminal_state.shape) != paired_state_shape:
            raise RuntimeError(
                "Paired address-state bank returned unexpected state shape: "
                f"{tuple(terminal_state.shape)} != {paired_state_shape}"
            )
        if paired_output.shape[2] != 2 * self.heads:
            raise RuntimeError(
                "Paired address-state bank returned unexpected output head count"
            )
        base_output, companion_output = paired_output.split(self.heads, dim=2)
        read_gate = torch.tanh(read_gate_parameter).view(1, 1, self.heads, 1)
        output = base_output + read_gate.to(dtype=base_output.dtype) * companion_output

        with torch.no_grad():
            q_residual_float = q_residual.float()
            k_residual_float = k_residual.float()
            q_float = q.float()
            k_float = k.float()
            base_state, companion_state = terminal_state.float().split(
                self.heads, dim=1
            )
            state_residual = companion_state - base_state
            state_base_rms = base_state.square().mean(
                dim=(1, 2, 3)
            ).sqrt().clamp_min(1e-8)
            state_residual_board = state_residual.square().mean(
                dim=(1, 2, 3)
            ).sqrt() / state_base_rms
            state_residual_head = state_residual.square().mean(
                dim=(2, 3)
            ).sqrt() / base_state.square().mean(
                dim=(2, 3)
            ).sqrt().clamp_min(1e-8)

            def residual_stats(
                residual: torch.Tensor,
                base: torch.Tensor,
            ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
                base_board = base.square().mean(
                    dim=(1, 2, 3)
                ).sqrt().clamp_min(1e-8)
                residual_board = residual.square().mean(
                    dim=(1, 2, 3)
                ).sqrt() / base_board
                residual_token = residual.square().mean(
                    dim=(2, 3)
                ).sqrt()
                residual_head = residual.square().mean(
                    dim=(1, 3)
                ).sqrt()
                return (
                    residual_board.mean(),
                    residual_board.std(unbiased=False),
                    residual_token.std(dim=1, unbiased=False).mean(),
                    residual_head.std(dim=1, unbiased=False).mean(),
                )

            q_relative, q_batch_std, q_token_std, q_head_std = residual_stats(
                q_residual_float, q_float
            )
            k_relative, k_batch_std, k_token_std, k_head_std = residual_stats(
                k_residual_float, k_float
            )

            def address_contrast(
                query: torch.Tensor,
                key: torch.Tensor,
            ) -> torch.Tensor:
                query_unit = F.normalize(query[:1].float(), dim=-1, eps=1e-6)
                key_unit = F.normalize(key[:1].float(), dim=-1, eps=1e-6)
                similarity = torch.einsum(
                    "bthd,bshd->bhts", query_unit, key_unit
                )
                diagonal = similarity.diagonal(dim1=-2, dim2=-1)
                diagonal_mean = diagonal.mean()
                if similarity.shape[-1] <= 1:
                    return diagonal_mean
                offdiag = (
                    similarity.sum() - diagonal.sum()
                ) / float(
                    similarity.shape[0]
                    * similarity.shape[1]
                    * similarity.shape[2]
                    * (similarity.shape[3] - 1)
                )
                return diagonal_mean - offdiag

            base_output_float = base_output.float()
            companion_output_float = companion_output.float()
            output_cosine = (
                (base_output_float * companion_output_float).mean()
                / (
                    base_output_float.square().mean().sqrt()
                    * companion_output_float.square().mean().sqrt()
                ).clamp_min(1e-8)
            )
            terminal_board_rms = terminal_state.float().square().mean(
                dim=(1, 2, 3)
            ).sqrt()
            read_gate_float = torch.tanh(read_gate_parameter.float())
            diagnostics = {
                "gdn3_paired_address_bank_enabled": q.new_ones(
                    (), dtype=torch.float32
                ),
                "gdn3_paired_address_bank_read_gate_abs": read_gate_float.abs().mean(),
                "gdn3_paired_address_bank_read_gate_head_std": read_gate_float.std(
                    unbiased=False
                ),
                "gdn3_paired_address_bank_q_residual_relative_rms": q_relative,
                "gdn3_paired_address_bank_q_residual_batch_std": q_batch_std,
                "gdn3_paired_address_bank_q_residual_token_std": q_token_std,
                "gdn3_paired_address_bank_q_residual_head_std": q_head_std,
                "gdn3_paired_address_bank_k_residual_relative_rms": k_relative,
                "gdn3_paired_address_bank_k_residual_batch_std": k_batch_std,
                "gdn3_paired_address_bank_k_residual_token_std": k_token_std,
                "gdn3_paired_address_bank_k_residual_head_std": k_head_std,
                "gdn3_paired_address_bank_state_residual_relative_rms": (
                    state_residual_board.mean()
                ),
                "gdn3_paired_address_bank_state_residual_batch_std": (
                    state_residual_board.std(unbiased=False)
                ),
                "gdn3_paired_address_bank_state_residual_head_std": (
                    state_residual_head.std(dim=1, unbiased=False).mean()
                ),
                "gdn3_paired_address_bank_base_address_contrast": address_contrast(
                    q, k
                ),
                "gdn3_paired_address_bank_companion_address_contrast": address_contrast(
                    companion_q, companion_k
                ),
                "gdn3_paired_address_bank_output_cosine": output_cosine,
                "gdn3_paired_address_bank_terminal_rms": terminal_board_rms.mean(),
                "gdn3_paired_address_bank_terminal_batch_std": terminal_board_rms.std(
                    unbiased=False
                ),
                "gdn3_paired_address_bank_q_weight_rms": (
                    q_projection.weight.float().square().mean().sqrt()
                ),
                "gdn3_paired_address_bank_k_weight_rms": (
                    k_projection.weight.float().square().mean().sqrt()
                ),
            }
        return output, terminal_state, diagnostics

    def _coupled_address_rows_transition(
        self,
        operation: Any,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor,
        b: torch.Tensor,
        w: torch.Tensor,
        address: torch.Tensor,
        cell_order: Optional[torch.Tensor],
        initial_state: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        projection = self.coupled_address_k_proj
        if self.update_mode != "coupled_address_rows" or projection is None:
            raise RuntimeError("Coupled address-row projection is unavailable")
        if operation is None:
            raise RuntimeError("Coupled address rows require the official GDN2 op")
        if (
            q.ndim != 4
            or k.shape != q.shape
            or g.shape != q.shape
            or b.shape != q.shape
            or v.ndim != 4
            or w.shape != v.shape
            or q.shape[:3] != v.shape[:3]
            or q.shape[2] != self.heads
            or q.shape[-1] != self.head_dim
            or v.shape[-1] != self.head_v_dim
        ):
            raise RuntimeError(
                "Coupled address rows require matched HxK32/HxV32 official inputs"
            )
        if address.ndim != 3 or address.shape[:2] != q.shape[:2]:
            raise RuntimeError("Coupled address rows require the canonical address stream")

        batch_size, seq_len = q.shape[:2]
        address_ordered = (
            address
            if cell_order is None
            else address.index_select(1, cell_order)
        )
        extra_k_raw = projection(address_ordered).view(
            batch_size,
            seq_len,
            self.heads,
            self.head_dim,
        )
        extra_k_float = extra_k_raw.float()
        extra_k = (
            extra_k_float
            * torch.rsqrt(
                1.0
                + extra_k_float.square().sum(dim=-1, keepdim=True)
            )
        ).to(dtype=k.dtype)

        q_base = fla_l2norm_fp32(q).to(dtype=q.dtype)
        k_base = fla_l2norm_fp32(k).to(dtype=k.dtype)
        q_expanded = torch.cat((q_base, q_base), dim=-1)
        k_expanded = torch.cat((k_base, extra_k), dim=-1)
        g_expanded = torch.cat((g, g), dim=-1)
        b_expanded = torch.cat((b, b), dim=-1)

        base_state_shape = (
            batch_size,
            self.heads,
            self.head_dim,
            self.head_v_dim,
        )
        expanded_state_shape = (
            batch_size,
            self.heads,
            2 * self.head_dim,
            self.head_v_dim,
        )
        expanded_initial_state: Optional[torch.Tensor]
        if initial_state is None:
            expanded_initial_state = None
        elif tuple(initial_state.shape) == base_state_shape:
            expanded_initial_state = torch.cat(
                (initial_state, torch.zeros_like(initial_state)),
                dim=-2,
            )
        elif tuple(initial_state.shape) == expanded_state_shape:
            expanded_initial_state = initial_state
        else:
            raise RuntimeError(
                "Coupled address-row incoming state mismatch: "
                f"{tuple(initial_state.shape)} not in "
                f"{{{base_state_shape}, {expanded_state_shape}}}"
            )

        output, terminal_state = operation(
            q=q_expanded,
            k=k_expanded,
            v=v,
            g=g_expanded,
            b=b_expanded,
            w=w,
            scale=1.0 / math.sqrt(float(self.head_dim)),
            initial_state=expanded_initial_state,
            output_final_state=True,
            use_qk_l2norm_in_kernel=False,
        )
        if tuple(terminal_state.shape) != expanded_state_shape:
            raise RuntimeError(
                "Coupled address rows returned unexpected state shape: "
                f"{tuple(terminal_state.shape)} != {expanded_state_shape}"
            )

        with torch.no_grad():
            extra_k_value = extra_k.float()
            k_base_value = k_base.float()
            k_board = extra_k_value.square().mean(
                dim=(1, 2, 3)
            ).sqrt()
            k_token = extra_k_value.square().mean(
                dim=(2, 3)
            ).sqrt()
            k_head = extra_k_value.square().mean(
                dim=(1, 3)
            ).sqrt()
            base_state, extra_state = terminal_state.float().split(
                self.head_dim,
                dim=-2,
            )
            base_board = base_state.square().mean(
                dim=(1, 2, 3)
            ).sqrt().clamp_min(1e-8)
            extra_board = extra_state.square().mean(
                dim=(1, 2, 3)
            ).sqrt()
            extra_relative = extra_board / base_board
            base_head = base_state.square().mean(
                dim=(2, 3)
            ).sqrt().clamp_min(1e-8)
            extra_head = extra_state.square().mean(
                dim=(2, 3)
            ).sqrt() / base_head
            terminal_comparable = torch.maximum(base_board, extra_board)
            diagnostics = {
                "gdn3_coupled_address_rows_enabled": q.new_ones(
                    (), dtype=torch.float32
                ),
                "gdn3_coupled_address_rows_k_weight_rms": (
                    projection.weight.float().square().mean().sqrt()
                ),
                "gdn3_coupled_address_rows_k_residual_relative_rms": (
                    extra_k_value.square().mean().sqrt()
                    / k_base_value.square().mean().sqrt().clamp_min(1e-8)
                ),
                "gdn3_coupled_address_rows_k_residual_batch_std": (
                    k_board.std(unbiased=False)
                ),
                "gdn3_coupled_address_rows_k_residual_token_std": (
                    k_token.std(dim=1, unbiased=False).mean()
                ),
                "gdn3_coupled_address_rows_k_residual_head_std": (
                    k_head.std(dim=1, unbiased=False).mean()
                ),
                "gdn3_coupled_address_rows_k_norm_max": (
                    extra_k_value.norm(dim=-1).max()
                ),
                "gdn3_coupled_address_rows_extra_state_relative_rms": (
                    extra_relative.mean()
                ),
                "gdn3_coupled_address_rows_extra_state_batch_std": (
                    extra_relative.std(unbiased=False)
                ),
                "gdn3_coupled_address_rows_extra_state_head_std": (
                    extra_head.std(dim=1, unbiased=False).mean()
                ),
                "gdn3_coupled_address_rows_base_state_rms": base_board.mean(),
                "gdn3_coupled_address_rows_extra_state_rms": extra_board.mean(),
                "gdn3_coupled_address_rows_terminal_rms": (
                    terminal_comparable.mean()
                ),
                "gdn3_coupled_address_rows_terminal_batch_std": (
                    terminal_comparable.std(unbiased=False)
                ),
            }
        return output, terminal_state, diagnostics

    def _orthogonal_chunk_state_transport(
        self,
        first_output: torch.Tensor,
        boundary_state: torch.Tensor,
        next_q: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        projection = self.orthogonal_chunk_state_proj
        if self.update_mode != "orthogonal_chunk_state" or projection is None:
            return boundary_state, self._zero_orthogonal_chunk_state_diag(
                first_output
            )
        if boundary_state.ndim != 4 or boundary_state.shape[-2:] != (
            self.head_dim,
            self.head_v_dim,
        ):
            raise RuntimeError(
                "Orthogonal chunk-state transport requires K-first GDN2 state "
                f"[B,H,{self.head_dim},{self.head_v_dim}], got "
                f"{tuple(boundary_state.shape)}"
            )
        if first_output.ndim != 4 or next_q.ndim != 4:
            raise RuntimeError(
                "Orthogonal chunk-state transport requires rank-4 chunk tensors"
            )

        summary = F.normalize(
            first_output.float().mean(dim=1), dim=-1, eps=1e-6
        )
        controls = projection(summary.to(dtype=projection.weight.dtype)).float()
        u_raw, v_raw, angle_raw = torch.split(
            controls,
            (self.head_dim, self.head_dim, 1),
            dim=-1,
        )
        u = F.normalize(u_raw, dim=-1, eps=1e-6)
        v_residual = v_raw - (u * v_raw).sum(dim=-1, keepdim=True) * u
        v_plane = F.normalize(v_residual, dim=-1, eps=1e-6)
        theta = math.pi * torch.tanh(angle_raw.squeeze(-1))
        cosine = torch.cos(theta).unsqueeze(-1)
        sine = torch.sin(theta).unsqueeze(-1)

        state_float = boundary_state.float()
        u_component = torch.einsum("bhk,bhkv->bhv", u, state_float)
        v_component = torch.einsum("bhk,bhkv->bhv", v_plane, state_float)
        rotated_u = cosine * u_component - sine * v_component
        rotated_v = sine * u_component + cosine * v_component
        rotated_state = (
            state_float
            + torch.einsum("bhk,bhv->bhkv", u, rotated_u - u_component)
            + torch.einsum(
                "bhk,bhv->bhkv", v_plane, rotated_v - v_component
            )
        )

        with torch.no_grad():
            state_board_rms = state_float.square().mean(
                dim=(1, 2, 3)
            ).sqrt().clamp_min(1e-6)
            state_residual = rotated_state - state_float
            state_residual_relative = state_residual.square().mean(
                dim=(1, 2, 3)
            ).sqrt() / state_board_rms
            rotated_board_rms = rotated_state.square().mean(
                dim=(1, 2, 3)
            ).sqrt()
            norm_ratio = rotated_board_rms / state_board_rms
            q_unit = F.normalize(next_q.float(), dim=-1, eps=1e-6)
            base_read = torch.einsum(
                "bthk,bhkv->bthv", q_unit, state_float
            )
            rotated_read = torch.einsum(
                "bthk,bhkv->bthv", q_unit, rotated_state
            )
            read_residual = rotated_read - base_read
            read_base_board_rms = base_read.square().mean(
                dim=(1, 2, 3)
            ).sqrt().clamp_min(1e-6)
            read_residual_relative = read_residual.square().mean(
                dim=(1, 2, 3)
            ).sqrt() / read_base_board_rms
            u_norm_error = (u.square().sum(dim=-1) - 1.0).abs()
            v_norm_error = (v_plane.square().sum(dim=-1) - 1.0).abs()
            plane_dot_abs = (u * v_plane).sum(dim=-1).abs()
            angle_weight = projection.weight[-1:].float()
            plane_weight = projection.weight[:-1].float()
            diagnostics = {
                "gdn3_orthogonal_chunk_state_enabled": first_output.new_ones(
                    (), dtype=torch.float32
                ),
                "gdn3_orthogonal_chunk_state_angle_abs": theta.abs().mean(),
                "gdn3_orthogonal_chunk_state_angle_batch_std": theta.abs().mean(
                    dim=1
                ).std(unbiased=False),
                "gdn3_orthogonal_chunk_state_angle_head_std": theta.abs().mean(
                    dim=0
                ).std(unbiased=False),
                "gdn3_orthogonal_chunk_state_plane_dot_abs_max": (
                    plane_dot_abs.max()
                ),
                "gdn3_orthogonal_chunk_state_plane_norm_error_max": torch.maximum(
                    u_norm_error.max(), v_norm_error.max()
                ),
                "gdn3_orthogonal_chunk_state_state_residual_relative_rms": (
                    state_residual_relative.mean()
                ),
                "gdn3_orthogonal_chunk_state_state_residual_batch_std": (
                    state_residual_relative.std(unbiased=False)
                ),
                "gdn3_orthogonal_chunk_state_boundary_norm_ratio_mean": (
                    norm_ratio.mean()
                ),
                "gdn3_orthogonal_chunk_state_boundary_norm_ratio_max_error": (
                    (norm_ratio - 1.0).abs().max()
                ),
                "gdn3_orthogonal_chunk_state_read_residual_relative_rms": (
                    read_residual_relative.mean()
                ),
                "gdn3_orthogonal_chunk_state_read_residual_batch_std": (
                    read_residual_relative.std(unbiased=False)
                ),
                "gdn3_orthogonal_chunk_state_terminal_rms": (
                    first_output.new_zeros((), dtype=torch.float32)
                ),
                "gdn3_orthogonal_chunk_state_terminal_batch_std": (
                    first_output.new_zeros((), dtype=torch.float32)
                ),
                "gdn3_orthogonal_chunk_state_angle_weight_rms": (
                    angle_weight.square().mean().sqrt()
                ),
                "gdn3_orthogonal_chunk_state_plane_weight_rms": (
                    plane_weight.square().mean().sqrt()
                ),
            }
        return rotated_state, diagnostics

    def _interleaved_write_transition(
        self,
        x: torch.Tensor,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor,
        b: torch.Tensor,
        w: torch.Tensor,
        initial_state: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        k_projection = self.interleaved_write_k_proj
        v_projection = self.interleaved_write_v_proj
        if (
            self.update_mode != "interleaved_write"
            or k_projection is None
            or v_projection is None
        ):
            raise RuntimeError("Interleaved-write projections are unavailable")
        if chunk_gdn2 is None:
            raise RuntimeError(
                "Interleaved write requires the official GDN2 chunk op"
            )
        if q.shape != k.shape or q.shape[:2] != x.shape[:2]:
            raise RuntimeError(
                "Interleaved write requires matched position Q/K token geometry"
            )
        if q.shape[2] != v.shape[2] or q.shape[2] != self.heads:
            raise RuntimeError(
                "Interleaved write requires one matched K/V head per model head"
            )

        batch_size, seq_len, _channels = x.shape
        k_residual = k_projection(x).view(
            batch_size, seq_len, self.heads, self.head_dim
        )
        auxiliary_k = k + k_residual.to(dtype=k.dtype)
        auxiliary_v = v_projection(x).view(
            batch_size, seq_len, self.heads, self.head_v_dim
        )

        def interleave(auxiliary: torch.Tensor, parent: torch.Tensor) -> torch.Tensor:
            return torch.stack((auxiliary, parent), dim=2).reshape(
                batch_size, 2 * seq_len, *parent.shape[2:]
            )

        interleaved_q = interleave(q, q)
        interleaved_k = interleave(auxiliary_k, k)
        interleaved_v = interleave(auxiliary_v, v)
        interleaved_g = interleave(torch.zeros_like(g), g)
        interleaved_b = interleave(torch.zeros_like(b), b)
        interleaved_w = interleave(torch.ones_like(w), w)
        interleaved_output, terminal_state = chunk_gdn2(
            q=interleaved_q,
            k=interleaved_k,
            v=interleaved_v,
            g=interleaved_g,
            b=interleaved_b,
            w=interleaved_w,
            initial_state=initial_state,
            output_final_state=True,
            use_qk_l2norm_in_kernel=True,
        )
        output = interleaved_output[:, 1::2]
        if output.shape[1] != seq_len:
            raise RuntimeError(
                "Interleaved write did not preserve the logical sequence length"
            )

        with torch.no_grad():
            k_residual_float = k_residual.float()
            auxiliary_v_float = auxiliary_v.float()
            k_float = k.float()
            v_float = v.float()
            k_board_rms = k_residual_float.square().mean(
                dim=(1, 2, 3)
            ).sqrt()
            k_token_rms = k_residual_float.square().mean(
                dim=(2, 3)
            ).sqrt()
            v_board_rms = auxiliary_v_float.square().mean(
                dim=(1, 2, 3)
            ).sqrt()
            v_token_rms = auxiliary_v_float.square().mean(
                dim=(2, 3)
            ).sqrt()
            auxiliary_k_unit = F.normalize(
                auxiliary_k.float(), dim=-1, eps=1e-6
            )
            parent_k_unit = F.normalize(k_float, dim=-1, eps=1e-6)
            auxiliary_write = torch.einsum(
                "bthk,bthv->bthkv", auxiliary_k_unit, auxiliary_v_float
            )
            parent_write = torch.einsum(
                "bthk,bthv->bthkv", parent_k_unit, v_float * w.float()
            )
            auxiliary_write_board_rms = auxiliary_write.square().mean(
                dim=(1, 2, 3, 4)
            ).sqrt()
            parent_write_board_rms = parent_write.square().mean(
                dim=(1, 2, 3, 4)
            ).sqrt().clamp_min(1e-8)
            write_relative = auxiliary_write_board_rms / parent_write_board_rms
            terminal_board_rms = terminal_state.float().square().mean(
                dim=(1, 2, 3)
            ).sqrt()
            diagnostics = {
                "gdn3_interleaved_write_enabled": x.new_ones(
                    (), dtype=torch.float32
                ),
                "gdn3_interleaved_write_k_residual_rms": (
                    k_residual_float.square().mean().sqrt()
                ),
                "gdn3_interleaved_write_k_residual_relative_rms": (
                    k_residual_float.square().mean().sqrt()
                    / k_float.square().mean().sqrt().clamp_min(1e-8)
                ),
                "gdn3_interleaved_write_k_batch_std": k_board_rms.std(
                    unbiased=False
                ),
                "gdn3_interleaved_write_k_token_std": k_token_rms.std(
                    dim=1, unbiased=False
                ).mean(),
                "gdn3_interleaved_write_v_rms": (
                    auxiliary_v_float.square().mean().sqrt()
                ),
                "gdn3_interleaved_write_v_relative_rms": (
                    auxiliary_v_float.square().mean().sqrt()
                    / v_float.square().mean().sqrt().clamp_min(1e-8)
                ),
                "gdn3_interleaved_write_v_batch_std": v_board_rms.std(
                    unbiased=False
                ),
                "gdn3_interleaved_write_v_token_std": v_token_rms.std(
                    dim=1, unbiased=False
                ).mean(),
                "gdn3_interleaved_write_state_write_relative_rms": (
                    write_relative.mean()
                ),
                "gdn3_interleaved_write_state_write_batch_std": (
                    write_relative.std(unbiased=False)
                ),
                "gdn3_interleaved_write_terminal_rms": (
                    terminal_board_rms.mean()
                ),
                "gdn3_interleaved_write_terminal_batch_std": (
                    terminal_board_rms.std(unbiased=False)
                ),
                "gdn3_interleaved_write_k_weight_rms": (
                    k_projection.weight.float().square().mean().sqrt()
                ),
                "gdn3_interleaved_write_v_weight_rms": (
                    v_projection.weight.float().square().mean().sqrt()
                ),
            }
        return output, terminal_state, diagnostics

    def _terminal_consolidation_transition(
        self,
        q: torch.Tensor,
        main_k: torch.Tensor,
        first_output: torch.Tensor,
        v: torch.Tensor,
        b: torch.Tensor,
        w: torch.Tensor,
        terminal_state: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        projection = self.terminal_consolidation_k_proj
        if self.update_mode != "terminal_consolidation" or projection is None:
            return terminal_state, self._zero_terminal_consolidation_diag(q)
        if chunk_gdn2 is None:
            raise RuntimeError(
                "Terminal consolidation requires the official GDN2 chunk op"
            )

        correction_k = projection(first_output.to(dtype=q.dtype))
        correction_q = F.normalize(q.float(), dim=-1).to(dtype=q.dtype)
        correction_g = torch.zeros_like(q, dtype=torch.float32)
        correction_output, consolidated_state = chunk_gdn2(
            q=correction_q,
            k=correction_k,
            v=v,
            g=correction_g,
            b=b,
            w=w,
            initial_state=terminal_state,
            output_final_state=True,
            use_qk_l2norm_in_kernel=False,
        )

        with torch.no_grad():
            correction_k_float = correction_k.float()
            main_k_rms = main_k.float().square().mean().sqrt().clamp_min(1e-6)
            board_k_rms = correction_k_float.square().mean(
                dim=(1, 2, 3)
            ).sqrt()
            token_k_rms = correction_k_float.square().mean(
                dim=(-1, -2)
            ).sqrt()
            state_before = terminal_state.float()
            state_residual = consolidated_state.float() - state_before
            board_state_rms = state_before.square().mean(
                dim=(1, 2, 3)
            ).sqrt().clamp_min(1e-6)
            board_residual_relative = state_residual.square().mean(
                dim=(1, 2, 3)
            ).sqrt() / board_state_rms
            correction_output_float = correction_output.float()
            correction_output_token_rms = correction_output_float.square().mean(
                dim=(-1, -2)
            ).sqrt()
            diagnostics = {
                "gdn3_terminal_consolidation_enabled": q.new_ones(
                    (), dtype=torch.float32
                ),
                "gdn3_terminal_consolidation_k_relative_rms": (
                    correction_k_float.square().mean().sqrt() / main_k_rms
                ).to(dtype=torch.float32),
                "gdn3_terminal_consolidation_k_batch_std": board_k_rms.std(
                    unbiased=False
                ).to(dtype=torch.float32),
                "gdn3_terminal_consolidation_k_token_std": token_k_rms.std(
                    dim=1, unbiased=False
                ).mean().to(dtype=torch.float32),
                "gdn3_terminal_consolidation_state_residual_relative_rms": (
                    board_residual_relative.mean().to(dtype=torch.float32)
                ),
                "gdn3_terminal_consolidation_state_residual_batch_std": (
                    board_residual_relative.std(unbiased=False).to(
                        dtype=torch.float32
                    )
                ),
                "gdn3_terminal_consolidation_output_rms": (
                    correction_output_float.square().mean().sqrt().to(
                        dtype=torch.float32
                    )
                ),
                "gdn3_terminal_consolidation_output_token_std": (
                    correction_output_token_rms.std(
                        dim=1, unbiased=False
                    ).mean().to(dtype=torch.float32)
                ),
                "gdn3_terminal_consolidation_weight_rms": (
                    projection.weight.float().square().mean().sqrt().to(
                        dtype=torch.float32
                    )
                ),
            }
        return consolidated_state, diagnostics

    def _coherent_delta_gates(
        self,
        b_raw: torch.Tensor,
        w_raw: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        if self.update_mode != "coherent_delta" or self.coherent_delta_mix is None:
            raise RuntimeError("Coherent delta gates selected without their controller")
        if b_raw.shape[:-1] != w_raw.shape[:-1]:
            raise ValueError(
                "Coherent delta gates require matching batch/time/head axes: "
                f"b={tuple(b_raw.shape)} w={tuple(w_raw.shape)}"
            )

        b_base = b_raw.sigmoid()
        w_base = w_raw.sigmoid()
        target = 0.5 * (
            b_base.float().mean(dim=-1, keepdim=True)
            + w_base.float().mean(dim=-1, keepdim=True)
        )
        mix = torch.tanh(self.coherent_delta_mix.float()).view(1, 1, -1, 1)
        b_effective = (
            b_base.float() + mix * (target - b_base.float())
        ).clamp(0.0, 1.0).to(dtype=b_base.dtype)
        w_effective = (
            w_base.float() + mix * (target - w_base.float())
        ).clamp(0.0, 1.0).to(dtype=w_base.dtype)

        with torch.no_grad():
            pre_gap = torch.cat(
                (b_base.float() - target, w_base.float() - target), dim=-1
            )
            post_gap = torch.cat(
                (b_effective.float() - target, w_effective.float() - target),
                dim=-1,
            )
            pre_gap_rms = pre_gap.square().mean().sqrt()
            post_gap_rms = post_gap.square().mean().sqrt()
            diag = {
                "gdn2_coherent_delta_enabled": b_raw.new_ones(
                    (), dtype=torch.float32
                ),
                "gdn2_coherent_delta_mix_mean": mix.mean(),
                "gdn2_coherent_delta_mix_abs": mix.abs().mean(),
                "gdn2_coherent_delta_mix_min": mix.min(),
                "gdn2_coherent_delta_mix_max": mix.max(),
                "gdn2_coherent_delta_target_mean": target.mean(),
                "gdn2_coherent_delta_pre_gap_rms": pre_gap_rms,
                "gdn2_coherent_delta_post_gap_rms": post_gap_rms,
                "gdn2_coherent_delta_gap_ratio": (
                    post_gap_rms / pre_gap_rms.clamp_min(1e-8)
                ),
                "gdn2_coherent_delta_b_relative_change": (
                    (b_effective.float() - b_base.float()).square().mean().sqrt()
                    / b_base.float().square().mean().sqrt().clamp_min(1e-8)
                ),
                "gdn2_coherent_delta_w_relative_change": (
                    (w_effective.float() - w_base.float()).square().mean().sqrt()
                    / w_base.float().square().mean().sqrt().clamp_min(1e-8)
                ),
            }
        return b_effective, w_effective, diag

    def _state_feedback_update(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        b_raw: torch.Tensor,
        w_raw: torch.Tensor,
        initial_state: Optional[torch.Tensor],
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        Dict[str, torch.Tensor],
    ]:
        if (
            self.update_mode != "state_feedback"
            or self.state_feedback_in is None
            or self.state_feedback_out is None
        ):
            raise RuntimeError("State-feedback update selected without its controller")
        b_base = b_raw.sigmoid()
        w_base = w_raw.sigmoid()
        if initial_state is None:
            return k, v, b_base, w_base, self._zero_state_feedback_diag(q)
        expected_state = (
            q.shape[0],
            self.heads,
            self.head_dim,
            self.head_v_dim,
        )
        if tuple(initial_state.shape) != expected_state:
            raise ValueError(
                "State-feedback initial state shape mismatch: "
                f"{tuple(initial_state.shape)} != {expected_state}"
            )
        if q.shape[:-1] != k.shape[:-1] or q.shape[:-1] != v.shape[:-1]:
            raise ValueError(
                "State-feedback requires aligned batch/time/head axes: "
                f"q={tuple(q.shape)} k={tuple(k.shape)} v={tuple(v.shape)}"
            )

        q_unit = F.normalize(q.float(), dim=-1)
        state_read = torch.einsum(
            "bthk,bhkv->bthv",
            q_unit,
            initial_state.float(),
        )
        hidden = F.silu(self.state_feedback_in(state_read.to(dtype=q.dtype)))
        residual = self.state_feedback_out(hidden).float()
        delta_k, delta_v, delta_b, delta_w = residual.split(
            (self.head_dim, self.head_v_dim, self.head_dim, self.head_v_dim),
            dim=-1,
        )
        # Preserve the parent's exact dtype path when the zero-initialized
        # controller emits zero. The strict CUDA contract checks bit identity.
        k_effective = k + delta_k.to(dtype=k.dtype)
        v_effective = v + delta_v.to(dtype=v.dtype)
        b_effective = (b_raw + delta_b.to(dtype=b_raw.dtype)).sigmoid()
        w_effective = (w_raw + delta_w.to(dtype=w_raw.dtype)).sigmoid()

        with torch.no_grad():
            base = torch.cat(
                (k.float(), v.float(), b_base.float(), w_base.float()),
                dim=-1,
            )
            board_read_rms = state_read.square().mean(dim=(1, 2, 3)).sqrt()
            token_residual_rms = residual.square().mean(dim=(2, 3)).sqrt()
            diag = {
                "gdn3_state_feedback_enabled": q.new_ones(
                    (), dtype=torch.float32
                ),
                "gdn3_state_feedback_read_rms": state_read.square()
                .mean()
                .sqrt(),
                "gdn3_state_feedback_read_batch_std": board_read_rms.std(
                    unbiased=False
                ),
                "gdn3_state_feedback_hidden_rms": hidden.float()
                .square()
                .mean()
                .sqrt(),
                "gdn3_state_feedback_residual_relative_rms": (
                    residual.square().mean().sqrt()
                    / base.square().mean().sqrt().clamp_min(1e-8)
                ),
                "gdn3_state_feedback_residual_token_std": token_residual_rms.std(
                    dim=1, unbiased=False
                ).mean(),
                "gdn3_state_feedback_k_relative_change": (
                    delta_k.square().mean().sqrt()
                    / k.float().square().mean().sqrt().clamp_min(1e-8)
                ),
                "gdn3_state_feedback_v_relative_change": (
                    delta_v.square().mean().sqrt()
                    / v.float().square().mean().sqrt().clamp_min(1e-8)
                ),
                "gdn3_state_feedback_b_relative_change": (
                    (b_effective.float() - b_base.float()).square().mean().sqrt()
                    / b_base.float().square().mean().sqrt().clamp_min(1e-8)
                ),
                "gdn3_state_feedback_w_relative_change": (
                    (w_effective.float() - w_base.float()).square().mean().sqrt()
                    / w_base.float().square().mean().sqrt().clamp_min(1e-8)
                ),
                "gdn3_state_feedback_in_weight_rms": self.state_feedback_in.weight.float()
                .square()
                .mean()
                .sqrt(),
                "gdn3_state_feedback_out_weight_rms": self.state_feedback_out.weight.float()
                .square()
                .mean()
                .sqrt(),
            }
        return k_effective, v_effective, b_effective, w_effective, diag

    def _forward_preconditioned_gdn2(
        self,
        x: torch.Tensor,
        *,
        initial_state: Optional[torch.Tensor],
        address: Optional[torch.Tensor],
        cell_order: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Run a tied curvature-aware write through the unchanged GDN2 kernel."""
        if self.backbone != "gdn2" or chunk_gdn2 is None:
            raise RuntimeError("Tied preconditioning requires the official GDN2 chunk op")
        if self.precondition_mode == "none":
            raise RuntimeError("Preconditioned GDN2 path selected with mode=none")
        use_position_address = self.address_mode == "position_qk"
        if use_position_address and (address is None or address.shape != x.shape):
            raise ValueError(
                "position_qk preconditioning requires an address stream matching x"
            )

        core = self.core
        batch_size, seq_len, _channels = x.shape
        qk_source = address if use_position_address else x
        assert qk_source is not None
        if core.use_short_conv:
            conv_q, conv_k, conv_v = (
                self._zero_conv_state(x)
                if initial_state is not None
                else (None, None, None)
            )
            q, _ = core.q_conv1d(
                x=core.q_proj(qk_source),
                cache=conv_q,
                output_final_state=True,
            )
            k, _ = core.k_conv1d(
                x=core.k_proj(qk_source),
                cache=conv_k,
                output_final_state=True,
            )
            v, _ = core.v_conv1d(
                x=core.v_proj(x),
                cache=conv_v,
                output_final_state=True,
            )
        else:
            q = F.silu(core.q_proj(qk_source))
            k = F.silu(core.k_proj(qk_source))
            v = F.silu(core.v_proj(x))

        q = q.view(batch_size, seq_len, core.num_heads, core.head_k_dim)
        k = k.view(batch_size, seq_len, core.num_heads, core.head_k_dim)
        q_canonical = q
        k_canonical = k
        if use_position_address and cell_order is not None:
            q = q.index_select(1, cell_order)
            k = k.index_select(1, cell_order)

        g = F.softplus(core.f_proj(x).float() + core.dt_bias)
        b = core.b_proj(x).sigmoid()
        w = core.w_proj(x).sigmoid()
        g = g.view(batch_size, seq_len, core.num_heads, core.head_k_dim)
        b = b.view(batch_size, seq_len, core.num_heads, core.head_k_dim)
        v = v.view(batch_size, seq_len, core.num_v_heads, core.head_v_dim)
        w = w.view(batch_size, seq_len, core.num_v_heads, core.head_v_dim)
        g = -core.A_log.float().exp().view(1, 1, core.num_heads, 1) * g

        if core.num_v_heads > core.num_heads:
            groups = core.num_v_heads // core.num_heads
            q = torch.repeat_interleave(q, groups, dim=-2)
            k = torch.repeat_interleave(k, groups, dim=-2)
            g = torch.repeat_interleave(g, groups, dim=-2)
            b = torch.repeat_interleave(b, groups, dim=-2)
        if core.allow_neg_eigval:
            b = b * 2.0

        q_unit, k_unit = normalize_qk_fp32(q, k)
        initial_precision = (
            futureseed_row_precision(initial_state)
            if self.precondition_mode == "futureseed_tied_atk"
            else None
        )
        multiplier, precondition_diag = causal_tied_atk_preconditioner(
            k_unit,
            g,
            b,
            initial_precision=initial_precision,
        )
        k_write, b_kernel, fold_diag = fold_preconditioned_write_into_gdn2(
            k_unit,
            b,
            multiplier,
        )

        operation = (
            fused_recurrent_gdn2
            if seq_len <= 64 and not self.training
            else chunk_gdn2
        )
        if operation is None:
            raise RuntimeError("The required official GDN2 kernel is unavailable")
        o, terminal_state = operation(
            q=q_unit,
            k=k_write,
            v=v,
            g=g,
            b=b_kernel,
            w=w,
            initial_state=initial_state,
            output_final_state=True,
            use_qk_l2norm_in_kernel=False,
        )

        address_diag = self._zero_address_diag(x)
        if use_position_address:
            with torch.no_grad():
                q_address = F.normalize(q_canonical[:1].float(), dim=-1)
                k_address = F.normalize(k_canonical[:1].float(), dim=-1)
                similarity = torch.einsum("bthd,bshd->bhts", q_address, k_address)
                diagonal_sum = similarity.diagonal(dim1=-2, dim2=-1).sum()
                diagonal_count = core.num_heads * seq_len
                diagonal_mean = diagonal_sum / max(diagonal_count, 1)
                offdiag_count = core.num_heads * seq_len * max(seq_len - 1, 1)
                offdiag_mean = (
                    (similarity.sum() - diagonal_sum) / max(offdiag_count, 1)
                    if seq_len > 1
                    else similarity.new_zeros(())
                )
            address_diag.update(
                {
                    "gdn2_address_enabled": x.new_ones(()),
                    "gdn2_address_qk_diag_cosine": diagonal_mean.detach().to(x.dtype),
                    "gdn2_address_qk_offdiag_cosine": offdiag_mean.detach().to(x.dtype),
                    "gdn2_address_qk_contrast": (
                        diagonal_mean - offdiag_mean
                    ).detach().to(x.dtype),
                }
            )

        zero = x.new_zeros(())
        self.last_gain_budget_diag = {
            "gdn2_gain_budget_enabled": zero,
            "gdn2_fast_slow_enabled": zero,
            **address_diag,
            **precondition_diag,
            **fold_diag,
        }
        output_gate = core.g_proj(x).view(
            batch_size,
            seq_len,
            core.num_v_heads,
            core.head_v_dim,
        )
        o = core.o_norm(o.to(dtype=x.dtype), output_gate)
        o = core.o_proj(o.reshape(batch_size, seq_len, core.value_dim))
        return o, terminal_state

    def _forward_gain_budget(
        self,
        x: torch.Tensor,
        *,
        initial_state: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.backbone != "gdn2" or chunk_gdn2 is None:
            raise RuntimeError("Gain-Budget requires the official GDN2 chunk op")
        core = self.core
        batch_size, seq_len, _channels = x.shape

        if core.use_short_conv:
            conv_q, conv_k, conv_v = (
                self._zero_conv_state(x)
                if initial_state is not None
                else (None, None, None)
            )
            q, _ = core.q_conv1d(
                x=core.q_proj(x),
                cache=conv_q,
                output_final_state=True,
            )
            k, _ = core.k_conv1d(
                x=core.k_proj(x),
                cache=conv_k,
                output_final_state=True,
            )
            v, _ = core.v_conv1d(
                x=core.v_proj(x),
                cache=conv_v,
                output_final_state=True,
            )
        else:
            q = F.silu(core.q_proj(x))
            k = F.silu(core.k_proj(x))
            v = F.silu(core.v_proj(x))

        g = F.softplus(core.f_proj(x).float() + core.dt_bias)
        b = core.b_proj(x).sigmoid()
        w = core.w_proj(x).sigmoid()
        q = q.view(batch_size, seq_len, core.num_heads, core.head_k_dim)
        k = k.view(batch_size, seq_len, core.num_heads, core.head_k_dim)
        g = g.view(batch_size, seq_len, core.num_heads, core.head_k_dim)
        b = b.view(batch_size, seq_len, core.num_heads, core.head_k_dim)
        v = v.view(batch_size, seq_len, core.num_v_heads, core.head_v_dim)
        w = w.view(batch_size, seq_len, core.num_v_heads, core.head_v_dim)
        g = -core.A_log.float().exp().view(1, 1, core.num_heads, 1) * g

        if core.allow_neg_eigval:
            b = b * 2.0

        with torch.autocast(device_type="cuda", enabled=False):
            if not (
                q.dtype
                == k.dtype
                == v.dtype
                == b.dtype
                == w.dtype
                == torch.bfloat16
            ):
                raise RuntimeError(
                    "Integrated Gain-Budget expects BF16 model projections; "
                    f"got q/k/v/b/w={q.dtype}/{k.dtype}/{v.dtype}/"
                    f"{b.dtype}/{w.dtype}"
                )
            # Normalize in FP32, then quantize before projection so the
            # projection sees the exact key consumed by the official BF16
            # chunk kernel. The strict FP32 certificate remains a separate
            # fused-recurrent contract lane because upstream chunk training
            # does not safely support a full FP32 recurrence.
            q_kernel = fla_l2norm_fp32(q).to(dtype=q.dtype)
            k_kernel = fla_l2norm_fp32(k).to(dtype=k.dtype)
            g_kernel = g.float()
            effective_b, projection = project_erase_gate(
                k_kernel.float(),
                b.float(),
                g_kernel,
                mode=(
                    "none"
                    if self.gain_budget_mode == "external_identity"
                    else self.gain_budget_mode
                ),
                sigma_cap=self.gain_budget_sigma_cap,
                step_gain_cap=self.gain_budget_step_cap,
                sigma_cap_max=self.gain_budget_sigma_cap_max,
                infeasible_policy=self.gain_budget_infeasible_policy,
            )
            gate_delta = effective_b - b.float()
            b_kernel = effective_b.to(dtype=b.dtype)
            _audited_b, quantized = project_erase_gate(
                k_kernel.float(),
                b_kernel.float(),
                g_kernel,
                mode="none",
                infeasible_policy="raise",
            )
            quantized_delta_error = (
                quantized["delta"] - projection["delta"]
            ).abs()
            if core.num_v_heads > core.num_heads:
                groups = core.num_v_heads // core.num_heads
                q_kernel = torch.repeat_interleave(q_kernel, groups, dim=-2)
                k_kernel = torch.repeat_interleave(k_kernel, groups, dim=-2)
                g_kernel = torch.repeat_interleave(g_kernel, groups, dim=-2)
                b_kernel = torch.repeat_interleave(
                    b_kernel,
                    groups,
                    dim=-2,
                )
            operation = (
                fused_recurrent_gdn2
                if seq_len <= 64 and not self.training
                else chunk_gdn2
            )
            if operation is None:
                raise RuntimeError("The required official GDN2 kernel is unavailable")
            o, terminal_state = operation(
                q=q_kernel,
                k=k_kernel,
                v=v,
                g=g_kernel,
                b=b_kernel,
                w=w,
                initial_state=initial_state,
                output_final_state=True,
                use_qk_l2norm_in_kernel=False,
            )

        self.last_gain_budget_diag = {
            "gdn2_gain_budget_enabled": x.new_tensor(
                float(self.gain_budget_mode != "external_identity")
            ),
            "gdn2_gain_budget_clipped_frac": projection["clipped"]
            .float()
            .mean()
            .detach(),
            "gdn2_gain_budget_infeasible_frac": projection["infeasible"]
            .float()
            .mean()
            .detach(),
            "gdn2_gain_budget_numerical_endpoint_frac": projection[
                "numerical_endpoint"
            ]
            .float()
            .mean()
            .detach(),
            "gdn2_gain_budget_lambda_mean": projection["scale"]
            .mean()
            .detach(),
            "gdn2_gain_budget_lambda_min": projection["scale"].min().detach(),
            "gdn2_gain_budget_tau_mean": projection["tau_requested"]
            .mean()
            .detach(),
            "gdn2_gain_budget_tau_effective_mean": projection["tau_effective"]
            .mean()
            .detach(),
            "gdn2_gain_budget_alpha_mean": projection["alpha_max"]
            .mean()
            .detach(),
            "gdn2_gain_budget_original_sigma_mean": projection[
                "original_sigma"
            ]
            .mean()
            .detach(),
            "gdn2_gain_budget_effective_sigma_mean": quantized[
                "effective_sigma"
            ]
            .mean()
            .detach(),
            "gdn2_gain_budget_original_step_bound": projection[
                "original_step_gain_bound"
            ]
            .mean()
            .detach(),
            "gdn2_gain_budget_effective_step_bound": quantized[
                "effective_step_gain_bound"
            ]
            .mean()
            .detach(),
            "gdn2_gain_budget_effective_step_bound_max": quantized[
                "effective_step_gain_bound"
            ]
            .max()
            .detach(),
            "gdn2_gain_budget_delta_error_max": quantized_delta_error
            .max()
            .detach(),
            "gdn2_gain_budget_fp32_projection_step_bound_max": projection[
                "effective_step_gain_bound"
            ]
            .max()
            .detach(),
            "gdn2_gain_budget_fp32_projection_delta_error_max": projection[
                "delta_abs_error"
            ]
            .max()
            .detach(),
            "gdn2_gain_budget_gate_relative_change": (
                gate_delta.square().mean().sqrt()
                / b.float().square().mean().sqrt().clamp_min(1e-8)
            ).detach(),
            "gdn2_gain_budget_fp32_numerical_certificate": x.new_zeros(()),
            "gdn2_gain_budget_low_precision_tolerance_path": x.new_ones(()),
            "gdn2_fast_slow_enabled": x.new_zeros(()),
            "gdn2_fast_slow_rho_mean": x.new_zeros(()),
            "gdn2_fast_slow_rho_min": x.new_zeros(()),
            "gdn2_fast_slow_rho_max": x.new_zeros(()),
            "gdn2_fast_slow_current_weight": x.new_ones(()),
            "gdn2_fast_slow_lag_mass": x.new_zeros(()),
            "gdn2_fast_slow_raw_hazard_mean": x.new_zeros(()),
            "gdn2_fast_slow_effective_hazard_mean": x.new_zeros(()),
            "gdn2_fast_slow_raw_tv": x.new_zeros(()),
            "gdn2_fast_slow_effective_tv": x.new_zeros(()),
            "gdn2_fast_slow_tv_ratio": x.new_ones(()),
            "gdn2_fast_slow_relative_change": x.new_zeros(()),
            "gdn2_fast_slow_alpha_mean": x.new_ones(()),
            **self._zero_address_diag(x),
        }

        output_gate = core.g_proj(x).view(
            batch_size,
            seq_len,
            core.num_v_heads,
            core.head_v_dim,
        )
        o = core.o_norm(o.to(dtype=x.dtype), output_gate)
        o = core.o_proj(o.reshape(batch_size, seq_len, core.value_dim))
        return o, terminal_state

    def _forward_fast_slow_decay(
        self,
        x: torch.Tensor,
        *,
        initial_state: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.backbone != "gdn2" or chunk_gdn2 is None:
            raise RuntimeError("Fast-Slow decay requires the official GDN2 chunk op")
        if self.fast_slow_decay is None:
            raise RuntimeError("Fast-Slow decay controller is missing")
        core = self.core
        batch_size, seq_len, _channels = x.shape

        if core.use_short_conv:
            conv_q, conv_k, conv_v = (
                self._zero_conv_state(x)
                if initial_state is not None
                else (None, None, None)
            )
            q, _ = core.q_conv1d(
                x=core.q_proj(x),
                cache=conv_q,
                output_final_state=True,
            )
            k, _ = core.k_conv1d(
                x=core.k_proj(x),
                cache=conv_k,
                output_final_state=True,
            )
            v, _ = core.v_conv1d(
                x=core.v_proj(x),
                cache=conv_v,
                output_final_state=True,
            )
        else:
            q = F.silu(core.q_proj(x))
            k = F.silu(core.k_proj(x))
            v = F.silu(core.v_proj(x))

        g = F.softplus(core.f_proj(x).float() + core.dt_bias)
        b = core.b_proj(x).sigmoid()
        w = core.w_proj(x).sigmoid()
        q = q.view(batch_size, seq_len, core.num_heads, core.head_k_dim)
        k = k.view(batch_size, seq_len, core.num_heads, core.head_k_dim)
        g = g.view(batch_size, seq_len, core.num_heads, core.head_k_dim)
        b = b.view(batch_size, seq_len, core.num_heads, core.head_k_dim)
        v = v.view(batch_size, seq_len, core.num_v_heads, core.head_v_dim)
        w = w.view(batch_size, seq_len, core.num_v_heads, core.head_v_dim)
        g = -core.A_log.float().exp().view(1, 1, core.num_heads, 1) * g
        effective_g, fast_slow_diag = self.fast_slow_decay(
            g,
            mode=self.fast_slow_decay_mode,
        )

        if core.num_v_heads > core.num_heads:
            groups = core.num_v_heads // core.num_heads
            q = torch.repeat_interleave(q, groups, dim=-2)
            k = torch.repeat_interleave(k, groups, dim=-2)
            effective_g = torch.repeat_interleave(effective_g, groups, dim=-2)
            b = torch.repeat_interleave(b, groups, dim=-2)
        if core.allow_neg_eigval:
            b = b * 2.0

        operation = (
            fused_recurrent_gdn2
            if seq_len <= 64 and not self.training
            else chunk_gdn2
        )
        if operation is None:
            raise RuntimeError("The required official GDN2 kernel is unavailable")
        o, terminal_state = operation(
            q=q,
            k=k,
            v=v,
            g=effective_g,
            b=b,
            w=w,
            initial_state=initial_state,
            output_final_state=True,
            use_qk_l2norm_in_kernel=True,
        )
        zero = x.new_zeros(())
        self.last_gain_budget_diag = {
            "gdn2_gain_budget_enabled": zero,
            "gdn2_gain_budget_clipped_frac": zero,
            "gdn2_gain_budget_infeasible_frac": zero,
            "gdn2_gain_budget_numerical_endpoint_frac": zero,
            "gdn2_gain_budget_lambda_mean": x.new_ones(()),
            "gdn2_gain_budget_lambda_min": x.new_ones(()),
            "gdn2_gain_budget_tau_mean": zero,
            "gdn2_gain_budget_tau_effective_mean": zero,
            "gdn2_gain_budget_alpha_mean": zero,
            "gdn2_gain_budget_original_sigma_mean": zero,
            "gdn2_gain_budget_effective_sigma_mean": zero,
            "gdn2_gain_budget_original_step_bound": zero,
            "gdn2_gain_budget_effective_step_bound": zero,
            "gdn2_gain_budget_effective_step_bound_max": zero,
            "gdn2_gain_budget_delta_error_max": zero,
            "gdn2_gain_budget_fp32_projection_step_bound_max": zero,
            "gdn2_gain_budget_fp32_projection_delta_error_max": zero,
            "gdn2_gain_budget_gate_relative_change": zero,
            "gdn2_gain_budget_fp32_numerical_certificate": zero,
            "gdn2_gain_budget_low_precision_tolerance_path": zero,
            **fast_slow_diag,
            **self._zero_address_diag(x),
        }

        output_gate = core.g_proj(x).view(
            batch_size,
            seq_len,
            core.num_v_heads,
            core.head_v_dim,
        )
        o = core.o_norm(o.to(dtype=x.dtype), output_gate)
        o = core.o_proj(o.reshape(batch_size, seq_len, core.value_dim))
        return o, terminal_state

    def _forward_position_qk(
        self,
        x: torch.Tensor,
        *,
        address: torch.Tensor,
        cell_order: Optional[torch.Tensor],
        initial_state: Optional[torch.Tensor],
        value_residual: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.backbone != "gdn2" or chunk_gdn2 is None:
            raise RuntimeError("Position-addressed Q/K requires the official GDN2 chunk op")
        if address.shape != x.shape:
            raise ValueError(
                f"Canonical address stream shape {tuple(address.shape)} "
                f"does not match hidden stream {tuple(x.shape)}"
            )
        core = self.core
        batch_size, seq_len, _channels = x.shape

        if core.use_short_conv:
            conv_q, conv_k, conv_v = (
                self._zero_conv_state(x)
                if initial_state is not None
                else (None, None, None)
            )
            # Q/K are computed in canonical position order, then gathered into
            # the current traversal. V and all state-edit gates remain content driven.
            q, _ = core.q_conv1d(
                x=core.q_proj(address),
                cache=conv_q,
                output_final_state=True,
            )
            k, _ = core.k_conv1d(
                x=core.k_proj(address),
                cache=conv_k,
                output_final_state=True,
            )
            v, _ = core.v_conv1d(
                x=core.v_proj(x),
                cache=conv_v,
                output_final_state=True,
            )
        else:
            q = F.silu(core.q_proj(address))
            k = F.silu(core.k_proj(address))
            v = F.silu(core.v_proj(x))

        q = q.view(batch_size, seq_len, core.num_heads, core.head_k_dim)
        k = k.view(batch_size, seq_len, core.num_heads, core.head_k_dim)
        if self.update_mode == "log_spd_metric":
            if self.log_spd_address_metric is None:
                raise RuntimeError("log_spd_metric requires a per-layer address metric")
            q, k = self.log_spd_address_metric.transform_pair(q, k)
            log_spd_diag = self.log_spd_address_metric.diagnostics(q.dtype)
        else:
            log_spd_diag = {
                "gdn3_log_spd_enabled": q.new_zeros((), dtype=torch.float32),
                "gdn3_log_spd_raw_rms": q.new_zeros((), dtype=torch.float32),
                "gdn3_log_spd_metric_delta_fro": q.new_zeros((), dtype=torch.float32),
                "gdn3_log_spd_eigenvalue_min": q.new_ones((), dtype=torch.float32),
                "gdn3_log_spd_eigenvalue_max": q.new_ones((), dtype=torch.float32),
                "gdn3_log_spd_condition_max": q.new_ones((), dtype=torch.float32),
                "gdn3_log_spd_logdet_abs_max": q.new_zeros((), dtype=torch.float32),
            }
        q_canonical = q
        k_canonical = k
        if cell_order is not None:
            q = q.index_select(1, cell_order)
            k = k.index_select(1, cell_order)

        g = F.softplus(core.f_proj(x).float() + core.dt_bias)
        b_raw = core.b_proj(x).view(
            batch_size, seq_len, core.num_heads, core.head_k_dim
        )
        w_raw = core.w_proj(x).view(
            batch_size, seq_len, core.num_v_heads, core.head_v_dim
        )
        g = g.view(batch_size, seq_len, core.num_heads, core.head_k_dim)
        v = v.view(batch_size, seq_len, core.num_v_heads, core.head_v_dim)
        raven_write_value_diag: Dict[str, torch.Tensor] = {}
        if value_residual is not None:
            expected_residual = (batch_size, seq_len, core.value_dim)
            if tuple(value_residual.shape) != expected_residual:
                raise ValueError(
                    "External GDN2 value residual shape "
                    f"{tuple(value_residual.shape)} does not match "
                    f"{expected_residual}"
                )
            value_residual_heads = value_residual.view(
                batch_size,
                seq_len,
                core.num_v_heads,
                core.head_v_dim,
            ).to(dtype=v.dtype)
            with torch.no_grad():
                base_board_rms = v.float().square().mean(
                    dim=(1, 2, 3)
                ).sqrt().clamp_min(1e-6)
                residual_board_rms = value_residual_heads.float().square().mean(
                    dim=(1, 2, 3)
                ).sqrt()
                residual_ratio = residual_board_rms / base_board_rms
                base_token_rms = v.float().square().mean(
                    dim=(2, 3)
                ).sqrt().clamp_min(1e-6)
                residual_token_rms = value_residual_heads.float().square().mean(
                    dim=(2, 3)
                ).sqrt()
                token_ratio = residual_token_rms / base_token_rms
                ratio_mean = residual_ratio.mean()
                raven_write_value_diag = {
                    "gdn3_raven_write_control_v_residual_relative_rms": ratio_mean,
                    "gdn3_raven_write_control_v_residual_relative_rms_min": ratio_mean,
                    "gdn3_raven_write_control_v_residual_relative_rms_max": ratio_mean,
                    "gdn3_raven_write_control_v_residual_batch_std": residual_ratio.std(
                        unbiased=False
                    ),
                    "gdn3_raven_write_control_v_residual_token_std": token_ratio.std(
                        dim=1,
                        unbiased=False,
                    ).mean(),
                }
            v = v + value_residual_heads
        if self.update_mode == "orthogonal_head_write":
            v, orthogonal_head_write_diag = self._orthogonal_head_write_route(v)
        else:
            orthogonal_head_write_diag = self._zero_orthogonal_head_write_diag(x)
        if self.update_mode == "coherent_delta":
            b, w, coherent_delta_diag = self._coherent_delta_gates(b_raw, w_raw)
            state_feedback_diag = self._zero_state_feedback_diag(x)
        elif self.update_mode == "state_feedback":
            k, v, b, w, state_feedback_diag = self._state_feedback_update(
                q,
                k,
                v,
                b_raw,
                w_raw,
                initial_state,
            )
            coherent_delta_diag = self._zero_coherent_delta_diag(x)
        else:
            b = b_raw.sigmoid()
            w = w_raw.sigmoid()
            coherent_delta_diag = self._zero_coherent_delta_diag(x)
            state_feedback_diag = self._zero_state_feedback_diag(x)
        if self.update_mode == "adaptive_signed_erase":
            b, adaptive_signed_erase_diag = self._adaptive_signed_erase(v, b)
        else:
            adaptive_signed_erase_diag = (
                self._zero_adaptive_signed_erase_diag(x)
            )
        paired_address_bank_diag = self._zero_paired_address_bank_diag(x)
        coupled_address_rows_diag = self._zero_coupled_address_rows_diag(x)
        bi_axis_value_decay_diag = self._zero_bi_axis_value_decay_diag(x)
        raven_routed_gdn_diag = self._zero_raven_routed_gdn_diag(x)
        g = -core.A_log.float().exp().view(1, 1, core.num_heads, 1) * g

        if self.update_mode == "raven_routed_gdn":
            k, g, raven_routed_gdn_diag = self._raven_routed_gdn_update(
                x,
                k,
                g,
            )

        if core.num_v_heads > core.num_heads:
            groups = core.num_v_heads // core.num_heads
            q = torch.repeat_interleave(q, groups, dim=-2)
            k = torch.repeat_interleave(k, groups, dim=-2)
            g = torch.repeat_interleave(g, groups, dim=-2)
            b = torch.repeat_interleave(b, groups, dim=-2)
        if core.allow_neg_eigval:
            b = b * 2.0

        if self.update_mode == "coupled_address_rows":
            operation = (
                fused_recurrent_gdn2
                if seq_len <= 64 and not self.training
                else chunk_gdn2
            )
            o, terminal_state, coupled_address_rows_diag = (
                self._coupled_address_rows_transition(
                    operation,
                    q,
                    k,
                    v,
                    g,
                    b,
                    w,
                    address,
                    cell_order,
                    initial_state,
                )
            )
            terminal_consolidation_diag = (
                self._zero_terminal_consolidation_diag(x)
            )
            orthogonal_chunk_state_diag = (
                self._zero_orthogonal_chunk_state_diag(x)
            )
            interleaved_write_diag = self._zero_interleaved_write_diag(x)
        elif self.update_mode == "paired_address_bank":
            operation = (
                fused_recurrent_gdn2
                if seq_len <= 64 and not self.training
                else chunk_gdn2
            )
            o, terminal_state, paired_address_bank_diag = (
                self._paired_address_bank_transition(
                    operation,
                    q,
                    k,
                    v,
                    g,
                    b,
                    w,
                    initial_state,
                )
            )
            terminal_consolidation_diag = (
                self._zero_terminal_consolidation_diag(x)
            )
            orthogonal_chunk_state_diag = (
                self._zero_orthogonal_chunk_state_diag(x)
            )
            interleaved_write_diag = self._zero_interleaved_write_diag(x)
        elif self.update_mode == "interleaved_write":
            o, terminal_state, interleaved_write_diag = (
                self._interleaved_write_transition(
                    x,
                    q,
                    k,
                    v,
                    g,
                    b,
                    w,
                    initial_state,
                )
            )
            terminal_consolidation_diag = (
                self._zero_terminal_consolidation_diag(x)
            )
            orthogonal_chunk_state_diag = (
                self._zero_orthogonal_chunk_state_diag(x)
            )
        elif self.update_mode == "orthogonal_chunk_state":
            if chunk_gdn2 is None:
                raise RuntimeError(
                    "Orthogonal chunk-state transport requires the official "
                    "GDN2 chunk op"
                )
            chunk_boundary = 64
            if seq_len <= chunk_boundary:
                raise RuntimeError(
                    "Orthogonal chunk-state transport requires a sequence "
                    f"longer than its registered boundary {chunk_boundary}, "
                    f"got {seq_len}"
                )
            first_output, boundary_state = chunk_gdn2(
                q=q[:, :chunk_boundary],
                k=k[:, :chunk_boundary],
                v=v[:, :chunk_boundary],
                g=g[:, :chunk_boundary],
                b=b[:, :chunk_boundary],
                w=w[:, :chunk_boundary],
                initial_state=initial_state,
                output_final_state=True,
                use_qk_l2norm_in_kernel=True,
            )
            transported_state, orthogonal_chunk_state_diag = (
                self._orthogonal_chunk_state_transport(
                    first_output,
                    boundary_state,
                    q[:, chunk_boundary:],
                )
            )
            second_output, terminal_state = chunk_gdn2(
                q=q[:, chunk_boundary:],
                k=k[:, chunk_boundary:],
                v=v[:, chunk_boundary:],
                g=g[:, chunk_boundary:],
                b=b[:, chunk_boundary:],
                w=w[:, chunk_boundary:],
                initial_state=transported_state,
                output_final_state=True,
                use_qk_l2norm_in_kernel=True,
            )
            o = torch.cat((first_output, second_output), dim=1)
            with torch.no_grad():
                terminal_board_rms = terminal_state.float().square().mean(
                    dim=(1, 2, 3)
                ).sqrt()
                orthogonal_chunk_state_diag[
                    "gdn3_orthogonal_chunk_state_terminal_rms"
                ] = terminal_board_rms.mean()
                orthogonal_chunk_state_diag[
                    "gdn3_orthogonal_chunk_state_terminal_batch_std"
                ] = terminal_board_rms.std(unbiased=False)
            terminal_consolidation_diag = (
                self._zero_terminal_consolidation_diag(x)
            )
            interleaved_write_diag = self._zero_interleaved_write_diag(x)
        else:
            operation = (
                fused_recurrent_gdn2
                if seq_len <= 64 and not self.training
                else chunk_gdn2
            )
            if operation is None:
                raise RuntimeError("The required official GDN2 kernel is unavailable")
            o, terminal_state, bi_axis_value_decay_diag = (
                self._bi_axis_value_decay_transition(
                    operation,
                    q,
                    k,
                    v,
                    g,
                    b,
                    w,
                    x,
                    initial_state,
                )
            )
            terminal_state, terminal_consolidation_diag = (
                self._terminal_consolidation_transition(
                    q,
                    k,
                    o,
                    v,
                    b,
                    w,
                    terminal_state,
                )
            )
            orthogonal_chunk_state_diag = (
                self._zero_orthogonal_chunk_state_diag(x)
            )
            interleaved_write_diag = self._zero_interleaved_write_diag(x)

        if self.update_mode == "orthogonal_head_write":
            with torch.no_grad():
                terminal_board_rms = terminal_state.float().square().mean(
                    dim=(1, 2, 3)
                ).sqrt()
                orthogonal_head_write_diag[
                    "gdn3_orthogonal_head_write_terminal_rms"
                ] = terminal_board_rms.mean()
                orthogonal_head_write_diag[
                    "gdn3_orthogonal_head_write_terminal_batch_std"
                ] = terminal_board_rms.std(unbiased=False)
        if self.update_mode == "adaptive_signed_erase":
            with torch.no_grad():
                terminal_board_rms = terminal_state.float().square().mean(
                    dim=(1, 2, 3)
                ).sqrt()
                adaptive_signed_erase_diag[
                    "gdn3_adaptive_signed_erase_terminal_rms"
                ] = terminal_board_rms.mean()
                adaptive_signed_erase_diag[
                    "gdn3_adaptive_signed_erase_terminal_batch_std"
                ] = terminal_board_rms.std(unbiased=False)
        if self.update_mode == "raven_routed_gdn":
            with torch.no_grad():
                terminal_board_rms = terminal_state.float().square().mean(
                    dim=(1, 2, 3)
                ).sqrt()
                raven_routed_gdn_diag[
                    "gdn3_raven_routed_terminal_rms"
                ] = terminal_board_rms.mean()
                raven_routed_gdn_diag[
                    "gdn3_raven_routed_terminal_batch_std"
                ] = terminal_board_rms.std(unbiased=False)
        if value_residual is not None:
            with torch.no_grad():
                terminal_board_rms = terminal_state.float().square().mean(
                    dim=(1, 2, 3)
                ).sqrt()
                terminal_rms_mean = terminal_board_rms.mean()
                raven_write_value_diag.update(
                    {
                        "gdn3_raven_write_control_main_terminal_rms": terminal_rms_mean,
                        "gdn3_raven_write_control_main_terminal_rms_max": terminal_rms_mean,
                        "gdn3_raven_write_control_main_terminal_batch_std": terminal_board_rms.std(
                            unbiased=False
                        ),
                    }
                )

        with torch.no_grad():
            q_unit = F.normalize(q_canonical[:1].float(), dim=-1)
            k_unit = F.normalize(k_canonical[:1].float(), dim=-1)
            similarity = torch.einsum("bthd,bshd->bhts", q_unit, k_unit)
            diagonal_sum = similarity.diagonal(dim1=-2, dim2=-1).sum()
            diagonal_count = core.num_heads * seq_len
            diagonal_mean = diagonal_sum / max(diagonal_count, 1)
            offdiag_count = core.num_heads * seq_len * max(seq_len - 1, 1)
            offdiag_mean = (
                (similarity.sum() - diagonal_sum) / max(offdiag_count, 1)
                if seq_len > 1
                else similarity.new_zeros(())
            )
        if cell_order is None or seq_len <= 1:
            order_displacement = x.new_zeros(())
        else:
            canonical = torch.arange(seq_len, device=cell_order.device)
            order_displacement = (
                (cell_order - canonical).abs().float().mean() / float(seq_len - 1)
            ).to(dtype=x.dtype)
        self.last_gain_budget_diag = {
            **self._zero_address_diag(x),
            **coherent_delta_diag,
            **state_feedback_diag,
            **terminal_consolidation_diag,
            **orthogonal_chunk_state_diag,
            **orthogonal_head_write_diag,
            **adaptive_signed_erase_diag,
            **bi_axis_value_decay_diag,
            **raven_routed_gdn_diag,
            **paired_address_bank_diag,
            **coupled_address_rows_diag,
            **interleaved_write_diag,
            **raven_write_value_diag,
            **log_spd_diag,
            "gdn2_address_enabled": x.new_ones(()),
            "gdn2_address_qk_diag_cosine": diagonal_mean.detach().to(dtype=x.dtype),
            "gdn2_address_qk_offdiag_cosine": offdiag_mean.detach().to(dtype=x.dtype),
            "gdn2_address_qk_contrast": (
                diagonal_mean - offdiag_mean
            ).detach().to(dtype=x.dtype),
            "gdn2_address_order_displacement": order_displacement.detach(),
        }

        output_gate = core.g_proj(x).view(
            batch_size,
            seq_len,
            core.num_v_heads,
            core.head_v_dim,
        )
        o = core.o_norm(o.to(dtype=x.dtype), output_gate)
        o = core.o_proj(o.reshape(batch_size, seq_len, core.value_dim))
        return o, terminal_state

    def _forward_anchor_rotary(
        self,
        x: torch.Tensor,
        *,
        address: torch.Tensor,
        cell_order: Optional[torch.Tensor],
        initial_state: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.backbone != "gdn2" or chunk_gdn2 is None:
            raise RuntimeError("Anchor-addressed Q/K requires the official GDN2 chunk op")
        if address.shape != x.shape:
            raise ValueError(
                f"Canonical anchor stream shape {tuple(address.shape)} "
                f"does not match hidden stream {tuple(x.shape)}"
            )
        if self.address_mode == "anchor_rotary":
            if self.address_rotation_scale is None:
                raise RuntimeError("anchor_rotary requires a learned rotation scale")
        elif self.address_mode == "anchor_phase":
            if self.address_phase_proj is None:
                raise RuntimeError("anchor_phase requires a learned phase projection")
        elif self.address_mode in {"anchor_residual", "anchor_carrier"}:
            if self.address_residual_proj is None:
                raise RuntimeError(
                    f"{self.address_mode} requires a learned residual projection"
                )
            if self.address_mode == "anchor_carrier" and (
                self.address_carrier_scale is None
                or self.address_carrier_bias_delta is None
            ):
                raise RuntimeError(
                    "anchor_carrier requires address-conditioned carrier parameters"
                )
        elif self.address_mode == "shared_namespace":
            pass
        elif self.address_mode == "anchor_qk_residual":
            if (
                self.address_q_residual_proj is None
                or self.address_k_residual_proj is None
            ):
                raise RuntimeError(
                    "anchor_qk_residual requires learned Q and K residual projections"
                )
        else:
            raise RuntimeError(f"unsupported anchor address mode: {self.address_mode}")
        core = self.core
        batch_size, seq_len, _channels = x.shape

        if core.use_short_conv:
            conv_q, conv_k, conv_v = (
                self._zero_conv_state(x)
                if initial_state is not None
                else (None, None, None)
            )
            q, _ = core.q_conv1d(
                x=core.q_proj(x),
                cache=conv_q,
                output_final_state=True,
            )
            k, _ = core.k_conv1d(
                x=core.k_proj(x),
                cache=conv_k,
                output_final_state=True,
            )
            v, _ = core.v_conv1d(
                x=core.v_proj(x),
                cache=conv_v,
                output_final_state=True,
            )
        else:
            q = F.silu(core.q_proj(x))
            k = F.silu(core.k_proj(x))
            v = F.silu(core.v_proj(x))

        q = q.view(batch_size, seq_len, core.num_heads, core.head_k_dim)
        k = k.view(batch_size, seq_len, core.num_heads, core.head_k_dim)
        q_before = q
        k_before = k
        q_address_vector: Optional[torch.Tensor] = None
        k_address_vector: Optional[torch.Tensor] = None
        if self.address_mode == "anchor_rotary":
            anchor_heads = address.reshape(
                batch_size,
                seq_len,
                core.num_heads,
                core.head_k_dim,
            )
            if cell_order is not None:
                anchor_heads = anchor_heads.index_select(1, cell_order)
            assert self.address_rotation_scale is not None
            q, phase = apply_anchor_rotary_address(
                q,
                anchor_heads,
                self.address_rotation_scale,
            )
            k, _ = apply_anchor_rotary_address(
                k,
                anchor_heads,
                self.address_rotation_scale,
            )
        elif self.address_mode == "anchor_phase":
            address_ordered = (
                address
                if cell_order is None
                else address.index_select(1, cell_order)
            )
            assert self.address_phase_proj is not None
            phase = math.pi * torch.tanh(
                self.address_phase_proj(address_ordered).float().view(
                    batch_size,
                    seq_len,
                    core.num_heads,
                    core.head_k_dim // 2,
                )
            )
            q = apply_address_phase_rotation(q, phase)
            k = apply_address_phase_rotation(k, phase)
        elif self.address_mode in {"anchor_residual", "anchor_carrier"}:
            address_ordered = (
                address
                if cell_order is None
                else address.index_select(1, cell_order)
            )
            assert self.address_residual_proj is not None
            q_address_vector = self.address_residual_proj(address_ordered).view(
                batch_size,
                seq_len,
                core.num_heads,
                core.head_k_dim,
            )
            k_address_vector = q_address_vector
            q = q + q_address_vector.to(dtype=q.dtype)
            k = k + k_address_vector.to(dtype=k.dtype)
            phase = q.new_zeros(
                batch_size,
                seq_len,
                core.num_heads,
                1,
                dtype=torch.float32,
            )
        elif self.address_mode == "shared_namespace":
            address_ordered = (
                address
                if cell_order is None
                else address.index_select(1, cell_order)
            )
            q_address_vector = address_ordered.view(
                batch_size,
                seq_len,
                core.num_heads,
                core.head_k_dim,
            )
            k_address_vector = q_address_vector
            q = q + q_address_vector.to(dtype=q.dtype)
            k = k + k_address_vector.to(dtype=k.dtype)
            phase = q.new_zeros(
                batch_size,
                seq_len,
                core.num_heads,
                1,
                dtype=torch.float32,
            )
        else:
            address_ordered = (
                address
                if cell_order is None
                else address.index_select(1, cell_order)
            )
            assert self.address_q_residual_proj is not None
            assert self.address_k_residual_proj is not None
            q_address_vector = self.address_q_residual_proj(address_ordered).view(
                batch_size,
                seq_len,
                core.num_heads,
                core.head_k_dim,
            )
            k_address_vector = self.address_k_residual_proj(address_ordered).view(
                batch_size,
                seq_len,
                core.num_heads,
                core.head_k_dim,
            )
            q = q + q_address_vector.to(dtype=q.dtype)
            k = k + k_address_vector.to(dtype=k.dtype)
            phase = q.new_zeros(
                batch_size,
                seq_len,
                core.num_heads,
                1,
                dtype=torch.float32,
            )
        q_rotated = q
        k_rotated = k

        g = F.softplus(core.f_proj(x).float() + core.dt_bias)
        b = core.b_proj(x).sigmoid()
        w = core.w_proj(x).sigmoid()
        g = g.view(batch_size, seq_len, core.num_heads, core.head_k_dim)
        b = b.view(batch_size, seq_len, core.num_heads, core.head_k_dim)
        v = v.view(batch_size, seq_len, core.num_v_heads, core.head_v_dim)
        w = w.view(batch_size, seq_len, core.num_v_heads, core.head_v_dim)
        g = -core.A_log.float().exp().view(1, 1, core.num_heads, 1) * g

        carrier: Optional[torch.Tensor] = None
        if self.address_mode == "anchor_carrier":
            assert q_address_vector is not None
            assert self.address_carrier_scale is not None
            assert self.address_carrier_bias_delta is not None
            carrier_logit = (
                self.address_carrier_base_logit
                + self.address_carrier_bias_delta.float().view(
                    1, 1, core.num_heads, core.head_k_dim
                )
                + self.address_carrier_scale.float().view(
                    1, 1, core.num_heads, core.head_k_dim
                )
                * q_address_vector.float()
            )
            carrier = torch.sigmoid(carrier_logit)

        if core.num_v_heads > core.num_heads:
            groups = core.num_v_heads // core.num_heads
            q = torch.repeat_interleave(q, groups, dim=-2)
            k = torch.repeat_interleave(k, groups, dim=-2)
            g = torch.repeat_interleave(g, groups, dim=-2)
            b = torch.repeat_interleave(b, groups, dim=-2)
            if carrier is not None:
                carrier = torch.repeat_interleave(carrier, groups, dim=-2)
        if core.allow_neg_eigval:
            b = b * 2.0

        b_before_carrier = b
        w_before_carrier = w
        write_norm_ratio = k.new_ones(
            batch_size,
            seq_len,
            core.num_v_heads,
            1,
            dtype=torch.float32,
        )
        if carrier is not None:
            k, b, w, write_norm_ratio = fold_gdn2_write_carrier_into_official_inputs(
                k,
                b,
                w,
                carrier,
            )

        operation = (
            fused_recurrent_gdn2
            if seq_len <= 64 and not self.training
            else chunk_gdn2
        )
        if operation is None:
            raise RuntimeError("The required official GDN2 kernel is unavailable")
        o, terminal_state = operation(
            q=q,
            k=k,
            v=v,
            g=g,
            b=b,
            w=w,
            initial_state=initial_state,
            output_final_state=True,
            use_qk_l2norm_in_kernel=True,
        )

        with torch.no_grad():
            q_before_f = q_before.float()
            k_before_f = k_before.float()
            q_after_f = q_rotated.float()
            k_after_f = k_rotated.float()
            q_relative_change = (
                (q_after_f - q_before_f).norm()
                / q_before_f.norm().clamp(min=1e-6)
            )
            k_relative_change = (
                (k_after_f - k_before_f).norm()
                / k_before_f.norm().clamp(min=1e-6)
            )
            q_norm_error = (
                q_after_f.norm(dim=-1) - q_before_f.norm(dim=-1)
            ).abs().max()
            k_norm_error = (
                k_after_f.norm(dim=-1) - k_before_f.norm(dim=-1)
            ).abs().max()
        if cell_order is None or seq_len <= 1:
            order_displacement = x.new_zeros(())
        else:
            canonical = torch.arange(seq_len, device=cell_order.device)
            order_displacement = (
                (cell_order - canonical).abs().float().mean() / float(seq_len - 1)
            ).to(dtype=x.dtype)
        zero_float = x.new_zeros((), dtype=torch.float32)
        if self.address_residual_proj is not None:
            shared_weight_rms = (
                self.address_residual_proj.weight.float().square().mean().sqrt()
            )
            q_residual_weight_rms = shared_weight_rms
            k_residual_weight_rms = shared_weight_rms
        else:
            q_residual_weight_rms = (
                self.address_q_residual_proj.weight.float().square().mean().sqrt()
                if self.address_q_residual_proj is not None
                else zero_float
            )
            k_residual_weight_rms = (
                self.address_k_residual_proj.weight.float().square().mean().sqrt()
                if self.address_k_residual_proj is not None
                else zero_float
            )
        q_residual_rms = (
            q_address_vector.float().square().mean().sqrt()
            if q_address_vector is not None
            else zero_float
        )
        k_residual_rms = (
            k_address_vector.float().square().mean().sqrt()
            if k_address_vector is not None
            else zero_float
        )
        residual_token_std = (
            0.5
            * (
                q_address_vector.float().std(dim=1, unbiased=False).mean()
                + k_address_vector.float().std(dim=1, unbiased=False).mean()
            )
            if q_address_vector is not None and k_address_vector is not None
            else zero_float
        )
        carrier_mean = carrier.float().mean() if carrier is not None else zero_float
        carrier_std = (
            carrier.float().std(unbiased=False) if carrier is not None else zero_float
        )
        carrier_token_std = (
            carrier.float().std(dim=1, unbiased=False).mean()
            if carrier is not None
            else zero_float
        )
        carrier_min = carrier.float().min() if carrier is not None else zero_float
        carrier_below_095_frac = (
            (carrier.float() < 0.95).float().mean()
            if carrier is not None
            else zero_float
        )
        carrier_b_input_relative_change = (
            (b.float() - b_before_carrier.float()).norm()
            / b_before_carrier.float().norm().clamp(min=1e-6)
            if carrier is not None
            else zero_float
        )
        carrier_w_input_relative_change = (
            (w.float() - w_before_carrier.float()).norm()
            / w_before_carrier.float().norm().clamp(min=1e-6)
            if carrier is not None
            else zero_float
        )
        self.last_gain_budget_diag = {
            **self._zero_address_diag(x),
            "gdn2_address_enabled": x.new_ones(()),
            "gdn2_address_order_displacement": order_displacement.detach(),
            "gdn2_address_rotation_scale_abs": (
                torch.tanh(self.address_rotation_scale.float()).abs().mean()
                if self.address_rotation_scale is not None
                else x.new_zeros((), dtype=torch.float32)
            ).detach().to(dtype=x.dtype),
            "gdn2_address_phase_abs": phase.abs().mean().detach().to(dtype=x.dtype),
            "gdn2_address_q_relative_change": q_relative_change.detach().to(dtype=x.dtype),
            "gdn2_address_k_relative_change": k_relative_change.detach().to(dtype=x.dtype),
            "gdn2_address_q_norm_error": q_norm_error.detach().to(dtype=x.dtype),
            "gdn2_address_k_norm_error": k_norm_error.detach().to(dtype=x.dtype),
            "gdn2_address_phase_weight_rms": (
                self.address_phase_proj.weight.float().square().mean().sqrt()
                if self.address_phase_proj is not None
                else x.new_zeros((), dtype=torch.float32)
            ).detach().to(dtype=x.dtype),
            "gdn2_address_phase_token_std": phase.float().std(
                dim=1, unbiased=False
            ).mean().detach().to(dtype=x.dtype),
            "gdn2_address_phase_plane_std": phase.float().std(
                dim=-1, unbiased=False
            ).mean().detach().to(dtype=x.dtype),
            "gdn2_address_residual_weight_rms": (
                0.5 * (q_residual_weight_rms + k_residual_weight_rms)
            ).detach().to(dtype=x.dtype),
            "gdn2_address_residual_rms": (
                0.5 * (q_residual_rms + k_residual_rms)
            ).detach().to(dtype=x.dtype),
            "gdn2_address_residual_token_std": residual_token_std.detach().to(
                dtype=x.dtype
            ),
            "gdn2_address_residual_q_ratio": (
                q_address_vector.float().norm()
                / q_before.float().norm().clamp(min=1e-6)
                if q_address_vector is not None
                else zero_float
            ).detach().to(dtype=x.dtype),
            "gdn2_address_residual_k_ratio": (
                k_address_vector.float().norm()
                / k_before.float().norm().clamp(min=1e-6)
                if k_address_vector is not None
                else zero_float
            ).detach().to(dtype=x.dtype),
            "gdn2_address_q_residual_weight_rms": q_residual_weight_rms.detach().to(
                dtype=x.dtype
            ),
            "gdn2_address_k_residual_weight_rms": k_residual_weight_rms.detach().to(
                dtype=x.dtype
            ),
            "gdn2_address_q_residual_rms": q_residual_rms.detach().to(dtype=x.dtype),
            "gdn2_address_k_residual_rms": k_residual_rms.detach().to(dtype=x.dtype),
            "gdn2_address_terminal_state_rms": (
                terminal_state.float().square().mean().sqrt()
            ).detach().to(dtype=x.dtype),
            "gdn2_address_carrier_mean": carrier_mean.detach().to(dtype=x.dtype),
            "gdn2_address_carrier_std": carrier_std.detach().to(dtype=x.dtype),
            "gdn2_address_carrier_token_std": carrier_token_std.detach().to(
                dtype=x.dtype
            ),
            "gdn2_address_carrier_min": carrier_min.detach().to(dtype=x.dtype),
            "gdn2_address_carrier_below_095_frac": (
                carrier_below_095_frac.detach().to(dtype=x.dtype)
            ),
            "gdn2_address_carrier_scale_rms": (
                self.address_carrier_scale.float().square().mean().sqrt()
                if self.address_carrier_scale is not None
                else zero_float
            ).detach().to(dtype=x.dtype),
            "gdn2_address_carrier_bias_delta_rms": (
                self.address_carrier_bias_delta.float().square().mean().sqrt()
                if self.address_carrier_bias_delta is not None
                else zero_float
            ).detach().to(dtype=x.dtype),
            "gdn2_address_carrier_write_norm_ratio": (
                write_norm_ratio.float().mean().detach().to(dtype=x.dtype)
            ),
            "gdn2_address_carrier_b_input_relative_change": (
                carrier_b_input_relative_change.detach().to(dtype=x.dtype)
            ),
            "gdn2_address_carrier_w_input_relative_change": (
                carrier_w_input_relative_change.detach().to(dtype=x.dtype)
            ),
        }

        output_gate = core.g_proj(x).view(
            batch_size,
            seq_len,
            core.num_v_heads,
            core.head_v_dim,
        )
        o = core.o_norm(o.to(dtype=x.dtype), output_gate)
        o = core.o_proj(o.reshape(batch_size, seq_len, core.value_dim))
        return o, terminal_state

    def pack_recurrent_state(self, state: Any) -> torch.Tensor:
        """Pack Raven's dual GSA cache into one lossless FutureSeed tensor."""
        if self.backbone != "raven":
            if not isinstance(state, torch.Tensor):
                raise TypeError(f"{self.backbone} recurrent state must be a tensor")
            return state
        if not isinstance(state, (tuple, list)) or len(state) != 2:
            raise TypeError("Raven recurrent state must be the official (key_state, value_state) tuple")
        key_state, value_state = state
        if not isinstance(key_state, torch.Tensor) or not isinstance(value_state, torch.Tensor):
            raise TypeError("Raven key/value recurrent states must be tensors")
        expected_key = (key_state.shape[0], self.heads, self.head_dim, self.raven_num_slots)
        expected_value = (
            key_state.shape[0],
            self.heads,
            self.raven_num_slots,
            self.head_v_dim,
        )
        if tuple(key_state.shape) != expected_key or tuple(value_state.shape) != expected_value:
            raise ValueError(
                "Raven recurrent state shape mismatch: "
                f"key={tuple(key_state.shape)} expected={expected_key}, "
                f"value={tuple(value_state.shape)} expected={expected_value}"
            )
        return torch.cat((key_state, value_state.transpose(-1, -2)), dim=-2)

    def unpack_recurrent_state(self, state: torch.Tensor) -> Any:
        if self.backbone != "raven":
            return state
        expected = (
            state.shape[0],
            self.heads,
            self.head_dim + self.head_v_dim,
            self.raven_num_slots,
        )
        if tuple(state.shape) != expected:
            raise ValueError(
                f"Raven packed state shape {tuple(state.shape)} does not match {expected}"
            )
        key_state, value_state_t = state.split(
            (self.head_dim, self.head_v_dim),
            dim=-2,
        )
        return key_state.contiguous(), value_state_t.transpose(-1, -2).contiguous()

    def _zero_conv_state(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size = x.shape[0]
        width = int(self.core.conv_size)
        return (
            x.new_zeros(batch_size, int(self.core.key_dim), width),
            x.new_zeros(batch_size, int(self.core.key_dim), width),
            x.new_zeros(batch_size, int(self.core.value_dim), width),
        )

    def read_recurrent_state(
        self,
        x: torch.Tensor,
        state: torch.Tensor,
    ) -> torch.Tensor:
        """Read a GDN2 state in its producer layer's native coordinate system."""
        if self.backbone != "gdn2":
            raise ValueError("Compatible FutureSeed readout is initially restricted to GDN2.")
        if not x.is_cuda:
            raise RuntimeError("GDN2 FutureSeed readout is CUDA-only; CPU fallback is disabled.")
        batch_size, seq_len, _channels = x.shape
        expected = (batch_size, self.heads, self.head_dim, self.head_v_dim)
        if tuple(state.shape) != expected:
            raise ValueError(
                f"GDN2 readout state shape {tuple(state.shape)} does not match {expected}"
            )

        q = self.core.q_proj(x)
        if self.core.use_short_conv:
            q, _conv_state = self.core.q_conv1d(
                x=q,
                cache=None,
                output_final_state=False,
            )
        else:
            q = F.silu(q)
        q = F.normalize(
            q.float().view(batch_size, seq_len, self.heads, self.head_dim),
            dim=-1,
            p=2.0,
        )
        readout = torch.einsum("bthk,bhkv->bthv", q, state.float())
        gate = self.core.g_proj(x).view(
            batch_size,
            seq_len,
            self.heads,
            self.head_v_dim,
        )
        readout = self.core.o_norm(readout.to(dtype=x.dtype), gate)
        return self.core.o_proj(
            readout.reshape(batch_size, seq_len, self.value_dim)
        )

    def forward(
        self,
        x: torch.Tensor,
        *,
        initial_state: Optional[torch.Tensor] = None,
        address: Optional[torch.Tensor] = None,
        cell_order: Optional[torch.Tensor] = None,
        value_residual: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if not x.is_cuda:
            raise RuntimeError(f"BACKBONE={self.backbone} is CUDA-only; CPU fallback is intentionally disabled.")
        batch_size, seq_len, _channels = x.shape
        if self.backbone == "raven":
            expected = (
                batch_size,
                self.heads,
                self.head_dim + self.head_v_dim,
                self.raven_num_slots,
            )
        else:
            state_key_dim = (
                2 * self.head_dim
                if self.update_mode == "coupled_address_rows"
                else self.head_dim
            )
            expected = (
                (batch_size, self.state_heads, self.head_v_dim, state_key_dim)
                if self.state_v_first
                else (batch_size, self.state_heads, state_key_dim, self.head_v_dim)
            )
        if initial_state is not None:
            base_expected = (
                (batch_size, self.heads, self.head_v_dim, self.head_dim)
                if self.state_v_first
                else (batch_size, self.heads, self.head_dim, self.head_v_dim)
            )
            allowed = {expected}
            if self.update_mode in {
                "paired_address_bank",
                "coupled_address_rows",
            }:
                allowed.add(base_expected)
            if tuple(initial_state.shape) not in allowed:
                raise ValueError(
                    f"{self.backbone.upper()} initial_state shape {tuple(initial_state.shape)} "
                    f"does not match one of {sorted(allowed)}"
                )
            initial_state = initial_state.float()

        if value_residual is not None and not (
            self.backbone == "gdn2"
            and self.address_mode == "position_qk"
            and self.update_mode == "none"
        ):
            raise ValueError(
                "External value residuals are restricted to the unmodified "
                "position-QK GDN2 transition"
            )

        if self.gain_budget_mode != "none":
            return self._forward_gain_budget(
                x,
                initial_state=initial_state,
            )
        if self.fast_slow_decay_mode != "none":
            return self._forward_fast_slow_decay(
                x,
                initial_state=initial_state,
            )
        if self.precondition_mode != "none":
            if self.address_mode == "position_qk" and address is None:
                raise ValueError(
                    "position_qk preconditioning requires a canonical address stream"
                )
            return self._forward_preconditioned_gdn2(
                x,
                initial_state=initial_state,
                address=address,
                cell_order=cell_order,
            )
        if self.address_mode == "position_qk":
            if address is None:
                raise ValueError("position_qk address mode requires a canonical address stream")
            return self._forward_position_qk(
                x,
                address=address,
                cell_order=cell_order,
                initial_state=initial_state,
                value_residual=value_residual,
            )
        if self.address_mode in {
            "anchor_rotary",
            "anchor_phase",
            "anchor_residual",
            "shared_namespace",
            "anchor_qk_residual",
            "anchor_carrier",
        }:
            if address is None:
                raise ValueError(
                    f"{self.address_mode} address mode requires a canonical anchor stream"
                )
            return self._forward_anchor_rotary(
                x,
                address=address,
                cell_order=cell_order,
                initial_state=initial_state,
            )

        assert FLACache is not None
        cache = FLACache()
        if initial_state is not None:
            cache.update(
                recurrent_state=self.unpack_recurrent_state(initial_state),
                conv_state=(
                    self._zero_conv_state(x)
                    if bool(getattr(self.core, "use_short_conv", False))
                    else None
                ),
                layer_idx=0,
                offset=0,
            )
        y, _attentions, returned_cache = self.core(
            x,
            past_key_values=cache,
            use_cache=True,
        )
        if returned_cache is None or len(returned_cache) == 0:
            raise RuntimeError(f"{self.backbone.upper()} official layer did not return its FLA cache.")
        raw_terminal_state = returned_cache[0]["recurrent_state"]
        if raw_terminal_state is None:
            raise RuntimeError(f"{self.backbone.upper()} kernel did not return a terminal state.")
        terminal_state = self.pack_recurrent_state(raw_terminal_state)
        zero = x.new_zeros(())
        self.last_gain_budget_diag = {
            "gdn2_gain_budget_enabled": zero,
            "gdn2_gain_budget_clipped_frac": zero,
            "gdn2_gain_budget_infeasible_frac": zero,
            "gdn2_gain_budget_numerical_endpoint_frac": zero,
            "gdn2_gain_budget_lambda_mean": x.new_ones(()),
            "gdn2_gain_budget_lambda_min": x.new_ones(()),
            "gdn2_gain_budget_tau_mean": zero,
            "gdn2_gain_budget_tau_effective_mean": zero,
            "gdn2_gain_budget_alpha_mean": zero,
            "gdn2_gain_budget_original_sigma_mean": zero,
            "gdn2_gain_budget_effective_sigma_mean": zero,
            "gdn2_gain_budget_original_step_bound": zero,
            "gdn2_gain_budget_effective_step_bound": zero,
            "gdn2_gain_budget_effective_step_bound_max": zero,
            "gdn2_gain_budget_delta_error_max": zero,
            "gdn2_gain_budget_fp32_projection_step_bound_max": zero,
            "gdn2_gain_budget_fp32_projection_delta_error_max": zero,
            "gdn2_gain_budget_gate_relative_change": zero,
            "gdn2_gain_budget_fp32_numerical_certificate": zero,
            "gdn2_gain_budget_low_precision_tolerance_path": zero,
            "gdn2_fast_slow_enabled": zero,
            "gdn2_fast_slow_rho_mean": zero,
            "gdn2_fast_slow_rho_min": zero,
            "gdn2_fast_slow_rho_max": zero,
            "gdn2_fast_slow_current_weight": x.new_ones(()),
            "gdn2_fast_slow_lag_mass": zero,
            "gdn2_fast_slow_raw_hazard_mean": zero,
            "gdn2_fast_slow_effective_hazard_mean": zero,
            "gdn2_fast_slow_raw_tv": zero,
            "gdn2_fast_slow_effective_tv": zero,
            "gdn2_fast_slow_tv_ratio": x.new_ones(()),
            "gdn2_fast_slow_relative_change": zero,
            "gdn2_fast_slow_alpha_mean": x.new_ones(()),
            **self._zero_address_diag(x),
            **self._zero_precondition_diag(x),
        }
        return y, terminal_state


class GDN2ResidualStateExpert(nn.Module):
    """Compact official-GDN2 state expert with a zero-init residual readout."""

    def __init__(
        self,
        d_model: int,
        heads: int,
        *,
        expert_width: int,
        use_short_conv: bool,
        conv_size: int,
    ) -> None:
        super().__init__()
        if expert_width <= 0 or expert_width % heads:
            raise ValueError("State-expert width must be positive and divisible by heads")
        self.d_model = int(d_model)
        self.heads = int(heads)
        self.expert_width = int(expert_width)
        self.head_dim = self.expert_width // self.heads
        self.content_down = nn.Linear(d_model, expert_width, bias=False)
        self.address_down = nn.Linear(d_model, expert_width, bias=False)
        self.time_mix = FLADeltaTimeMix(
            expert_width,
            heads,
            self.head_dim,
            backbone="gdn2",
            expand_v=1.0,
            mode="chunk",
            use_short_conv=use_short_conv,
            conv_size=conv_size,
            allow_neg_eigval=False,
            address_mode="position_qk",
            update_mode="none",
        )
        self.up_proj = nn.Linear(expert_width, d_model, bias=False)
        nn.init.zeros_(self.up_proj.weight)
        self.future_seed_logit = nn.Parameter(torch.zeros(1, heads, 1, 1))

    def forward(
        self,
        x: torch.Tensor,
        *,
        address: torch.Tensor,
        cell_order: Optional[torch.Tensor],
        incoming_state: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        if x.shape != address.shape:
            raise ValueError(
                "State-expert content/address shape mismatch: "
                f"{tuple(x.shape)} != {tuple(address.shape)}"
            )
        content = self.content_down(x)
        expert_address = self.address_down(address)
        zero = x.new_zeros(())
        initial_state: Optional[torch.Tensor] = None
        seed_rms = zero
        gate_mean = zero
        if incoming_state is not None:
            expected = (
                x.shape[0],
                self.heads,
                self.head_dim,
                self.head_dim,
            )
            if tuple(incoming_state.shape) != expected:
                raise ValueError(
                    "State-expert incoming shape mismatch: "
                    f"{tuple(incoming_state.shape)} != {expected}"
                )
            denom = incoming_state.square().mean(
                dim=(-1, -2), keepdim=True
            ).sqrt().clamp_min(1e-6)
            gate = torch.sigmoid(self.future_seed_logit).to(
                device=incoming_state.device,
                dtype=incoming_state.dtype,
            )
            initial_state = incoming_state / denom * gate
            seed_rms = initial_state.float().square().mean().sqrt().to(dtype=x.dtype)
            gate_mean = gate.mean().to(dtype=x.dtype)

        expert_output, terminal_state = self.time_mix(
            content,
            initial_state=initial_state,
            address=expert_address,
            cell_order=cell_order,
        )
        residual = self.up_proj(expert_output)
        residual_board_rms = residual.float().square().mean(
            dim=(-1, -2)
        ).sqrt()
        content_rms = x.float().square().mean().sqrt().clamp_min(1e-6)
        terminal_board_rms = terminal_state.float().square().mean(
            dim=(-1, -2, -3)
        ).sqrt()
        address_diag = self.time_mix.last_gain_budget_diag
        return residual, terminal_state, {
            "gdn3_state_expert_enabled": x.new_ones(()),
            "gdn3_state_expert_residual_relative_rms": (
                residual_board_rms.mean() / content_rms
            ).to(dtype=x.dtype),
            "gdn3_state_expert_residual_batch_std": residual_board_rms.std(
                unbiased=False
            ).to(dtype=x.dtype),
            "gdn3_state_expert_terminal_rms": terminal_board_rms.mean().to(
                dtype=x.dtype
            ),
            "gdn3_state_expert_terminal_batch_std": terminal_board_rms.std(
                unbiased=False
            ).to(dtype=x.dtype),
            "gdn3_state_expert_seed_rms": seed_rms,
            "gdn3_state_expert_gate_mean": gate_mean,
            "gdn3_state_expert_up_weight_rms": self.up_proj.weight.float().square().mean().sqrt().to(
                dtype=x.dtype
            ),
            "gdn3_state_expert_address_contrast": address_diag[
                "gdn2_address_qk_contrast"
            ].to(dtype=x.dtype),
            "gdn3_state_expert_output_cosine": zero,
        }


class RavenWriteController(nn.Module):
    """Persistent official-Raven control state that writes into GDN2 V."""

    WIDTH = 64
    HEADS = 4
    HEAD_DIM = 16
    # The pinned official chunk-GSA Triton kernel requires each dot axis >= 16.
    NUM_SLOTS = 16
    TOPK = 1

    def __init__(self, d_model: int) -> None:
        super().__init__()
        if d_model != 256:
            raise ValueError(
                "The registered Raven write controller requires D256"
            )
        self.d_model = int(d_model)
        self.content_down = nn.Linear(d_model, self.WIDTH, bias=False)
        self.time_mix = FLADeltaTimeMix(
            self.WIDTH,
            self.HEADS,
            self.HEAD_DIM,
            backbone="raven",
            expand_v=1.0,
            mode="chunk",
            use_short_conv=False,
            conv_size=4,
            allow_neg_eigval=False,
            address_mode="none",
            update_mode="none",
            raven_num_slots=self.NUM_SLOTS,
            raven_topk=self.TOPK,
        )
        self.value_adapter = nn.Linear(self.WIDTH, d_model, bias=False)
        nn.init.zeros_(self.value_adapter.weight)
        self.future_seed_logit = nn.Parameter(
            torch.zeros(1, self.HEADS, 1, 1)
        )

    def forward(
        self,
        x: torch.Tensor,
        *,
        incoming_state: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        content = self.content_down(x)
        zero = x.new_zeros(())
        initial_state: Optional[torch.Tensor] = None
        seed_rms = zero
        seed_gate_mean = zero
        if incoming_state is not None:
            expected = (
                x.shape[0],
                self.HEADS,
                2 * self.HEAD_DIM,
                self.NUM_SLOTS,
            )
            if tuple(incoming_state.shape) != expected:
                raise ValueError(
                    "Raven write-control incoming shape mismatch: "
                    f"{tuple(incoming_state.shape)} != {expected}"
                )
            incoming = incoming_state.float()
            denom = incoming.square().mean(
                dim=(-1, -2),
                keepdim=True,
            ).sqrt().clamp_min(1e-6)
            gate = torch.sigmoid(self.future_seed_logit.float())
            initial_state = incoming / denom * gate
            seed_rms = initial_state.square().mean().sqrt().to(dtype=x.dtype)
            seed_gate_mean = gate.mean().to(dtype=x.dtype)

        controller_output, terminal_state = self.time_mix(
            content,
            initial_state=initial_state,
        )
        value_residual = self.value_adapter(controller_output)

        with torch.no_grad():
            output_board_rms = controller_output.float().square().mean(
                dim=(1, 2)
            ).sqrt()
            terminal_board_rms = terminal_state.float().square().mean(
                dim=(1, 2, 3)
            ).sqrt()
            slot_mass = terminal_state.float().square().sum(dim=(1, 2))
            slot_probability = slot_mass / slot_mass.sum(
                dim=-1,
                keepdim=True,
            ).clamp_min(1e-12)
            slot_entropy = -(
                slot_probability
                * slot_probability.clamp_min(1e-12).log()
            ).sum(dim=-1) / math.log(float(self.NUM_SLOTS))
            slot_max_mass_share = slot_probability.max(dim=-1).values

        diagnostics = {
            "gdn3_raven_write_control_enabled": x.new_ones(()),
            "gdn3_raven_write_control_output_rms": output_board_rms.mean().to(
                dtype=x.dtype
            ),
            "gdn3_raven_write_control_output_batch_std": output_board_rms.std(
                unbiased=False
            ).to(dtype=x.dtype),
            "gdn3_raven_write_control_terminal_rms": terminal_board_rms.mean().to(
                dtype=x.dtype
            ),
            "gdn3_raven_write_control_terminal_batch_std": terminal_board_rms.std(
                unbiased=False
            ).to(dtype=x.dtype),
            "gdn3_raven_write_control_seed_rms": seed_rms,
            "gdn3_raven_write_control_seed_gate_mean": seed_gate_mean,
            "gdn3_raven_write_control_slot_entropy_normalized": slot_entropy.mean().to(
                dtype=x.dtype
            ),
            "gdn3_raven_write_control_slot_entropy_normalized_min": slot_entropy.mean().to(
                dtype=x.dtype
            ),
            "gdn3_raven_write_control_slot_max_mass_share": slot_max_mass_share.mean().to(
                dtype=x.dtype
            ),
            "gdn3_raven_write_control_slot_max_mass_share_max": slot_max_mass_share.max().to(
                dtype=x.dtype
            ),
            "gdn3_raven_write_control_adapter_weight_rms": self.value_adapter.weight.float().square().mean().sqrt().to(
                dtype=x.dtype
            ),
        }
        if incoming_state is not None:
            diagnostics[
                "gdn3_raven_write_control_seed_rms_receiving_min"
            ] = seed_rms
            diagnostics[
                "gdn3_raven_write_control_incoming_path_count_sum"
            ] = x.new_ones(())
        return value_residual, terminal_state, diagnostics


class FLADeltaBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        heads: int,
        head_dim: int,
        channel_mult: int,
        *,
        backbone: str,
        gdn_mode: str,
        gdn_expand_v: float,
        gdn_use_short_conv: bool,
        gdn_conv_size: int,
        gdn_allow_neg_eigval: bool,
        gdn2_gain_budget_mode: str = "none",
        gdn2_gain_budget_sigma_cap: float = 1.10,
        gdn2_gain_budget_step_cap: float = 1.0,
        gdn2_gain_budget_sigma_cap_max: float = 3.0,
        gdn2_gain_budget_infeasible_policy: str = "raise",
        gdn2_fast_slow_decay_mode: str = "none",
        gdn2_fast_slow_decay_kernel_size: int = 4,
        gdn2_fast_slow_decay_rho_init: float = 0.10,
        gdn2_fast_slow_decay_current_weight_init: float = 0.85,
        gdn2_precondition_mode: str = "none",
        gdn2_address_mode: str = "none",
        gdn2_update_mode: str = "none",
        gdn2_terminal_consolidation_enabled: bool = True,
        gdn2_state_expert_mode: str = "none",
        raven_num_slots: int = 0,
        raven_topk: int = 0,
    ) -> None:
        super().__init__()
        self.ln_time = nn.LayerNorm(d_model)
        self.ln_channel = nn.LayerNorm(d_model)
        self.time_mix = FLADeltaTimeMix(
            d_model,
            heads,
            head_dim,
            backbone=backbone,
            expand_v=gdn_expand_v,
            mode=gdn_mode,
            use_short_conv=gdn_use_short_conv,
            conv_size=gdn_conv_size,
            allow_neg_eigval=gdn_allow_neg_eigval,
            gain_budget_mode=gdn2_gain_budget_mode,
            gain_budget_sigma_cap=gdn2_gain_budget_sigma_cap,
            gain_budget_step_cap=gdn2_gain_budget_step_cap,
            gain_budget_sigma_cap_max=gdn2_gain_budget_sigma_cap_max,
            gain_budget_infeasible_policy=gdn2_gain_budget_infeasible_policy,
            fast_slow_decay_mode=gdn2_fast_slow_decay_mode,
            fast_slow_decay_kernel_size=gdn2_fast_slow_decay_kernel_size,
            fast_slow_decay_rho_init=gdn2_fast_slow_decay_rho_init,
            fast_slow_decay_current_weight_init=(
                gdn2_fast_slow_decay_current_weight_init
            ),
            precondition_mode=gdn2_precondition_mode,
            address_mode=gdn2_address_mode,
            update_mode=gdn2_update_mode,
            terminal_consolidation_enabled=(
                gdn2_terminal_consolidation_enabled
            ),
            raven_num_slots=raven_num_slots,
            raven_topk=raven_topk,
        )
        self.channel_mix = ChannelMix(d_model, channel_mult)
        self.future_seed_logit = nn.Parameter(torch.zeros(1, heads, 1, 1))
        if gdn2_state_expert_mode not in GDN2_STATE_EXPERT_MODES:
            raise ValueError(
                "gdn2_state_expert_mode must be one of: "
                f"{', '.join(GDN2_STATE_EXPERT_MODES)}"
            )
        if gdn2_state_expert_mode != "none" and (
            backbone != "gdn2" or gdn2_address_mode != "position_qk"
        ):
            raise ValueError(
                "GDN3 auxiliary state modes require position-QK official GDN2"
            )
        self.gdn2_state_expert_mode = gdn2_state_expert_mode
        if gdn2_state_expert_mode == "dual_state":
            self.state_expert: Optional[nn.Module] = GDN2ResidualStateExpert(
                d_model,
                heads,
                expert_width=d_model // 2,
                use_short_conv=gdn_use_short_conv,
                conv_size=gdn_conv_size,
            )
        elif gdn2_state_expert_mode == "raven_write_control":
            self.state_expert = RavenWriteController(d_model)
        else:
            self.state_expert = None
        self.last_state_expert_terminal: Optional[torch.Tensor] = None
        self.last_state_expert_diag: Dict[str, torch.Tensor] = {}

    def forward(
        self,
        x: torch.Tensor,
        *,
        initial_state: Optional[torch.Tensor] = None,
        state_expert_initial_state: Optional[torch.Tensor] = None,
        address: Optional[torch.Tensor] = None,
        cell_order: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        normalized_address = (
            address
            if address is not None
            and self.time_mix.address_mode == "shared_namespace"
            else self.ln_time(address)
            if address is not None
            else None
        )
        normalized_x = self.ln_time(x)
        if self.gdn2_state_expert_mode == "raven_write_control":
            if not isinstance(self.state_expert, RavenWriteController):
                raise RuntimeError("Raven write controller is missing")
            value_residual, expert_terminal, expert_diag = self.state_expert(
                normalized_x,
                incoming_state=state_expert_initial_state,
            )
            time_out, terminal_state = self.time_mix(
                normalized_x,
                initial_state=initial_state,
                address=normalized_address,
                cell_order=cell_order,
                value_residual=value_residual,
            )
            self.last_state_expert_terminal = expert_terminal
            self.last_state_expert_diag = expert_diag
        else:
            time_out, terminal_state = self.time_mix(
                normalized_x,
                initial_state=initial_state,
                address=normalized_address,
                cell_order=cell_order,
            )
        if self.gdn2_state_expert_mode == "dual_state":
            if not isinstance(self.state_expert, GDN2ResidualStateExpert):
                raise RuntimeError("Dual-state expert is missing")
            if normalized_address is None:
                raise ValueError("Dual-state expert requires a canonical address stream")
            expert_residual, expert_terminal, expert_diag = self.state_expert(
                normalized_x,
                address=normalized_address,
                cell_order=cell_order,
                incoming_state=state_expert_initial_state,
            )
            time_norm = time_out.float().square().mean().sqrt().clamp_min(1e-6)
            expert_norm = expert_residual.float().square().mean().sqrt()
            cosine_denom = time_norm * expert_norm
            expert_diag["gdn3_state_expert_output_cosine"] = torch.where(
                cosine_denom > 0,
                (time_out.float() * expert_residual.float()).mean() / cosine_denom,
                cosine_denom.new_zeros(()),
            ).to(dtype=x.dtype)
            time_out = time_out + expert_residual
            self.last_state_expert_terminal = expert_terminal
            self.last_state_expert_diag = expert_diag
        elif self.gdn2_state_expert_mode == "none":
            if state_expert_initial_state is not None:
                raise ValueError("State-expert input was provided while the expert is disabled")
            zero = x.new_zeros(())
            self.last_state_expert_terminal = None
            self.last_state_expert_diag = {
                key: zero for key in STATE_EXPERT_TRAIN_KEYS
            }
        x = x + time_out
        x = x + self.channel_mix(self.ln_channel(x))
        return x, terminal_state

    def read_terminal_state(
        self,
        x: torch.Tensor,
        state: torch.Tensor,
    ) -> torch.Tensor:
        return self.time_mix.read_recurrent_state(self.ln_time(x), state)


class MomentumDeltaBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        heads: int,
        head_dim: int,
        channel_mult: int,
        *,
        layer_idx: int,
        gdn_expand_v: float,
        gdn_use_short_conv: bool,
        gdn_conv_size: int,
    ) -> None:
        super().__init__()
        self.ln_time = nn.LayerNorm(d_model)
        self.ln_channel = nn.LayerNorm(d_model)
        self.time_mix = MomentumDeltaTimeMix(
            d_model,
            heads,
            head_dim,
            layer_idx=layer_idx,
            expand_v=gdn_expand_v,
            use_short_conv=gdn_use_short_conv,
            conv_size=gdn_conv_size,
        )
        self.channel_mix = ChannelMix(d_model, channel_mult)
        self.future_seed_logit = nn.Parameter(torch.zeros(1, heads, 1, 1))

    def forward(
        self,
        x: torch.Tensor,
        *,
        initial_state: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        time_out, terminal_state = self.time_mix(
            self.ln_time(x),
            initial_state=initial_state,
        )
        x = x + time_out
        x = x + self.channel_mix(self.ln_channel(x))
        return x, terminal_state


class FutureSeedRWKV(nn.Module):
    def __init__(
        self,
        d_model: int,
        layers: int,
        heads: int,
        head_dim: int,
        channel_mult: int,
        *,
        future_seed_scale: float = 1.0,
        future_seed_decay: float = 0.0,
        future_seed_update: str = "fixed",
        future_seed_norm_mode: str = "unit",
        future_seed_gate_mode: str = "head",
        future_seed_scope: str = "layer",
        future_seed_readout_hop: int = 0,
        future_seed_content_mode: str = "terminal",
        momentum_future_seed_transport: str = "full_state",
        activation_checkpoint: bool = False,
        rwkv_kernel: str = "auto",
        backbone: str = "rwkv",
        gdn_mode: str = "chunk",
        gdn_expand_v: float = 1.0,
        gdn_progressive_base_expand_v: float = 0.0,
        gdn_use_short_conv: bool = True,
        gdn_conv_size: int = 4,
        gdn_allow_neg_eigval: bool = False,
        gdn2_gain_budget_mode: str = "none",
        gdn2_gain_budget_sigma_cap: float = 1.10,
        gdn2_gain_budget_step_cap: float = 1.0,
        gdn2_gain_budget_sigma_cap_max: float = 3.0,
        gdn2_gain_budget_infeasible_policy: str = "raise",
        gdn2_fast_slow_decay_mode: str = "none",
        gdn2_fast_slow_decay_kernel_size: int = 4,
        gdn2_fast_slow_decay_rho_init: float = 0.10,
        gdn2_fast_slow_decay_current_weight_init: float = 0.85,
        gdn2_precondition_mode: str = "none",
        gdn2_address_mode: str = "none",
        gdn2_update_mode: str = "none",
        gdn2_state_expert_mode: str = "none",
        gdn2_cross_layer_init: str = "independent",
        raven_num_slots: int = 0,
        raven_topk: int = 0,
    ) -> None:
        super().__init__()
        if layers < 2:
            raise ValueError("FutureSeed needs at least two layers.")
        if future_seed_update not in {
            "fixed",
            "learned",
            "loop_residual",
            "loop_secant",
        }:
            raise ValueError(
                "future_seed_update must be one of: fixed, learned, "
                "loop_residual, loop_secant."
            )
        if future_seed_norm_mode not in {"unit", "adaptive_rms"}:
            raise ValueError("future_seed_norm_mode must be one of: unit, adaptive_rms.")
        if future_seed_gate_mode not in FUTURE_SEED_GATE_MODES:
            raise ValueError(
                f"future_seed_gate_mode must be one of: {', '.join(FUTURE_SEED_GATE_MODES)}."
            )
        if future_seed_scope not in {"layer", "block"}:
            raise ValueError("future_seed_scope must be one of: layer, block.")
        if future_seed_content_mode not in FUTURE_SEED_CONTENT_MODES:
            raise ValueError(
                "future_seed_content_mode must be one of: "
                f"{', '.join(FUTURE_SEED_CONTENT_MODES)}."
            )
        if future_seed_scope == "block" and future_seed_update != "fixed":
            raise ValueError("block FutureSeed currently requires future_seed_update=fixed.")
        if future_seed_scope == "block" and future_seed_gate_mode != "head":
            raise ValueError("block FutureSeed currently requires the canonical head gate.")
        if future_seed_readout_hop < 0 or future_seed_readout_hop == 1:
            raise ValueError("future_seed_readout_hop must be 0 or at least 2.")
        if future_seed_readout_hop >= layers:
            raise ValueError("future_seed_readout_hop must be smaller than the layer count.")
        if future_seed_readout_hop > 0 and backbone != "gdn2":
            raise ValueError("compatible FutureSeed readout is initially restricted to GDN2.")
        if future_seed_readout_hop > 0 and future_seed_scope != "layer":
            raise ValueError("compatible FutureSeed readout requires future_seed_scope=layer.")
        if future_seed_readout_hop > 0 and future_seed_update != "fixed":
            raise ValueError("compatible FutureSeed readout requires future_seed_update=fixed.")
        if future_seed_readout_hop > 0 and future_seed_gate_mode != "head":
            raise ValueError("compatible FutureSeed readout requires the canonical head gate.")
        if future_seed_content_mode in {
            "innovation_residual",
            "producer_codec",
            "address_local_update",
            "orthogonal_basis_transport",
        }:
            if layers < 3:
                raise ValueError(
                    f"{future_seed_content_mode} FutureSeed needs at least three layers."
                )
            if backbone != "gdn2":
                raise ValueError(
                    f"{future_seed_content_mode} FutureSeed is restricted to GDN2."
                )
            if future_seed_scope != "layer":
                raise ValueError(
                    f"{future_seed_content_mode} FutureSeed requires layer scope."
                )
            if future_seed_update != "fixed" or future_seed_decay != 0:
                raise ValueError(
                    f"{future_seed_content_mode} FutureSeed requires fixed, "
                    "zero-decay updates."
                )
            if future_seed_norm_mode != "unit" or future_seed_gate_mode != "head":
                raise ValueError(
                    f"{future_seed_content_mode} FutureSeed requires unit normalization "
                    "and head gates."
                )
            if future_seed_readout_hop != 0:
                raise ValueError(
                    f"{future_seed_content_mode} FutureSeed cannot be mixed with "
                    "multihop readout."
                )
        if future_seed_content_mode == "orthogonal_basis_transport" and (
            gdn2_address_mode != "position_qk"
            or gdn2_update_mode != "none"
            or gdn2_state_expert_mode != "none"
            or gdn2_cross_layer_init != "independent"
            or not math.isclose(gdn_expand_v, 1.0)
            or not math.isclose(gdn_progressive_base_expand_v, 0.0)
        ):
            raise ValueError(
                "Orthogonal basis transport composes only with matched-width "
                "independent position-QK GDN2 and the unmodified recurrent update"
            )
        if backbone not in {
            "rwkv",
            "rwkv7",
            "gdn",
            "fla_gdn",
            "gdn2",
            "kda",
            "raven",
            "momentum",
        }:
            raise ValueError(
                "backbone must be one of: rwkv, rwkv7, gdn, fla_gdn, gdn2, "
                "kda, raven, momentum."
            )
        if momentum_future_seed_transport not in {"full_state", "momentum_only"}:
            raise ValueError(
                "momentum_future_seed_transport must be full_state or momentum_only"
            )
        if backbone == "momentum" and (
            future_seed_update != "fixed"
            or future_seed_norm_mode != "unit"
            or future_seed_gate_mode != "head"
            or future_seed_scope != "layer"
            or future_seed_readout_hop != 0
            or future_seed_content_mode != "terminal"
            or gdn2_address_mode != "none"
            or gdn2_update_mode != "none"
            or gdn2_state_expert_mode != "none"
            or gdn2_cross_layer_init != "independent"
        ):
            raise ValueError(
                "Momentum Sudoku composes only with fixed adjacent-layer terminal "
                "FutureSeed and no GDN2 wrapper mechanisms"
            )
        if backbone != "momentum" and momentum_future_seed_transport != "full_state":
            raise ValueError("momentum_only transport requires the Momentum backbone")
        if gdn2_cross_layer_init not in GDN2_CROSS_LAYER_INIT_MODES:
            raise ValueError(
                "gdn2_cross_layer_init must be one of: "
                f"{', '.join(GDN2_CROSS_LAYER_INIT_MODES)}."
            )
        if gdn2_cross_layer_init != "independent" and backbone != "gdn2":
            raise ValueError("Cross-layer coordinate initialization is restricted to GDN2.")
        if gdn2_cross_layer_init != "independent" and gdn2_address_mode != "none":
            raise ValueError(
                "Cross-layer coordinate initialization cannot be mixed with GDN2 addressing."
            )
        if gdn2_state_expert_mode not in GDN2_STATE_EXPERT_MODES:
            raise ValueError(
                "gdn2_state_expert_mode must be one of: "
                f"{', '.join(GDN2_STATE_EXPERT_MODES)}"
            )
        if gdn2_state_expert_mode != "none" and (
            backbone != "gdn2"
            or gdn2_address_mode != "position_qk"
            or gdn2_update_mode != "none"
            or gdn2_cross_layer_init != "independent"
            or not math.isclose(future_seed_scale, 1.0)
            or not math.isclose(future_seed_decay, 0.0)
            or future_seed_update != "fixed"
            or future_seed_norm_mode != "unit"
            or future_seed_gate_mode != "head"
            or future_seed_scope != "layer"
            or future_seed_readout_hop != 0
            or future_seed_content_mode != "terminal"
        ):
            raise ValueError(
                "GDN3 auxiliary state modes compose only with independent "
                "position-QK GDN2, the unmodified main update, and fixed "
                "adjacent-layer terminal FutureSeed"
            )
        if gdn2_state_expert_mode == "raven_write_control" and (
            d_model != 256
            or layers != 12
            or heads != 8
            or head_dim != 32
            or not math.isclose(gdn_expand_v, 1.0)
            or not math.isclose(gdn_progressive_base_expand_v, 0.0)
        ):
            raise ValueError(
                "The registered Raven write controller is fixed to "
                "D256/L12/H8/K32/V32"
            )
        if gdn2_update_mode == "paired_address_bank" and (
            backbone != "gdn2"
            or gdn2_address_mode != "position_qk"
            or gdn2_state_expert_mode != "none"
            or gdn2_cross_layer_init != "independent"
            or not math.isclose(gdn_expand_v, 1.0)
            or not math.isclose(gdn_progressive_base_expand_v, 0.0)
            or not math.isclose(future_seed_scale, 1.0)
            or not math.isclose(future_seed_decay, 0.0)
            or future_seed_update != "fixed"
            or future_seed_norm_mode != "unit"
            or future_seed_gate_mode != "head"
            or future_seed_scope != "layer"
            or future_seed_readout_hop != 0
            or future_seed_content_mode != "terminal"
        ):
            raise ValueError(
                "The paired address-state bank composes only with matched-width "
                "independent position-QK GDN2 and fixed adjacent-layer terminal "
                "FutureSeed"
            )
        if gdn2_update_mode == "coupled_address_rows" and (
            backbone != "gdn2"
            or gdn2_address_mode != "position_qk"
            or gdn2_state_expert_mode != "none"
            or gdn2_cross_layer_init != "independent"
            or not math.isclose(gdn_expand_v, 1.0)
            or not math.isclose(gdn_progressive_base_expand_v, 0.0)
            or not math.isclose(future_seed_scale, 1.0)
            or not math.isclose(future_seed_decay, 0.0)
            or future_seed_update != "fixed"
            or future_seed_norm_mode != "unit"
            or future_seed_gate_mode != "head"
            or future_seed_scope != "layer"
            or future_seed_readout_hop != 0
            or future_seed_content_mode != "terminal"
        ):
            raise ValueError(
                "Coupled address-row expansion composes only with matched-width "
                "independent position-QK GDN2 and fixed adjacent-layer terminal "
                "FutureSeed"
            )
        self.backbone = backbone
        self.future_seed_scale = float(future_seed_scale)
        self.future_seed_decay = float(future_seed_decay)
        self.future_seed_update = future_seed_update
        self.future_seed_norm_mode = future_seed_norm_mode
        self.future_seed_gate_mode = future_seed_gate_mode
        self.future_seed_scope = future_seed_scope
        self.future_seed_readout_hop = int(future_seed_readout_hop)
        self.future_seed_content_mode = future_seed_content_mode
        self.momentum_future_seed_transport = momentum_future_seed_transport
        self.activation_checkpoint = bool(activation_checkpoint)
        self.gdn2_precondition_mode = gdn2_precondition_mode
        self.gdn2_address_mode = gdn2_address_mode
        self.gdn2_update_mode = gdn2_update_mode
        self.gdn2_state_expert_mode = gdn2_state_expert_mode
        self.gdn2_cross_layer_init = gdn2_cross_layer_init
        self.shared_address_proj = (
            nn.Linear(d_model, d_model, bias=False)
            if gdn2_address_mode == "shared_namespace"
            else None
        )
        if self.shared_address_proj is not None:
            nn.init.zeros_(self.shared_address_proj.weight)
        self.gdn_progressive_base_head_v_dim = int(head_dim * float(gdn_progressive_base_expand_v))
        if future_seed_update in {"learned", "loop_residual"}:
            update_init = min(max(1.0 - self.future_seed_decay, 1e-4), 1.0 - 1e-4)
            update_logit = math.log(update_init / (1.0 - update_init))
            self.future_seed_update_logit = nn.Parameter(
                torch.full((layers - 1, 1, heads, 1, 1), float(update_logit))
            )
        else:
            self.register_parameter("future_seed_update_logit", None)
        if future_seed_update == "loop_secant":
            self.future_seed_secant_raw = nn.Parameter(
                torch.zeros(layers - 1, 1, heads, 1, 1)
            )
            self.future_seed_secant_raw._no_weight_decay = True
        else:
            self.register_parameter("future_seed_secant_raw", None)
        if future_seed_norm_mode == "adaptive_rms":
            norm_shape = (layers - 1, 1, heads, 1, 1)
            self.future_seed_norm_slope = nn.Parameter(torch.zeros(norm_shape))
            self.future_seed_norm_bias = nn.Parameter(torch.zeros(norm_shape))
        else:
            self.register_parameter("future_seed_norm_slope", None)
            self.register_parameter("future_seed_norm_bias", None)
        if self.future_seed_readout_hop > 0:
            self.future_seed_readout_scale = nn.Parameter(
                torch.zeros(layers - self.future_seed_readout_hop)
            )
            self.future_seed_readout_scale._no_weight_decay = True
        else:
            self.register_parameter("future_seed_readout_scale", None)
        if self.future_seed_content_mode == "innovation_residual":
            self.future_seed_innovation_scale = nn.Parameter(
                torch.zeros(layers - 2, 1, heads, 1, 1)
            )
            self.future_seed_innovation_scale._no_weight_decay = True
        else:
            self.register_parameter("future_seed_innovation_scale", None)
        expanded_head_dim = int(head_dim * float(gdn_expand_v))
        if backbone in {"gdn", "fla_gdn", "kda"}:
            state_row_dim = expanded_head_dim
            state_col_dim = head_dim
        elif backbone in {"gdn2", "momentum"}:
            state_row_dim = (
                2 * head_dim
                if gdn2_update_mode == "coupled_address_rows"
                else head_dim
            )
            state_col_dim = expanded_head_dim
        elif backbone == "raven":
            state_row_dim = head_dim + expanded_head_dim
            if raven_num_slots <= 0:
                matched_state_elements = head_dim * expanded_head_dim
                slot_width = head_dim + expanded_head_dim
                if matched_state_elements % slot_width:
                    raise ValueError(
                        "Automatic Raven state matching is not integral; set --raven_num_slots."
                    )
                raven_num_slots = matched_state_elements // slot_width
            state_col_dim = int(raven_num_slots)
        else:
            state_row_dim = head_dim
            state_col_dim = head_dim
        self.future_seed_producer_codec = (
            FutureSeedProducerCodec(
                row_dim=state_row_dim,
                col_dim=state_col_dim,
            )
            if self.future_seed_content_mode == "producer_codec"
            else None
        )
        self.future_seed_address_local_update = (
            FutureSeedAddressLocalUpdate(
                row_dim=state_row_dim,
                col_dim=state_col_dim,
            )
            if self.future_seed_content_mode == "address_local_update"
            else None
        )
        future_seed_state_heads = (
            2 * heads if gdn2_update_mode == "paired_address_bank" else heads
        )
        self.future_seed_basis_transport = (
            FutureSeedOrthogonalBasisTransport(
                edges=layers - 1,
                heads=future_seed_state_heads,
                row_dim=state_row_dim,
                col_dim=state_col_dim,
            )
            if self.future_seed_content_mode == "orthogonal_basis_transport"
            else None
        )
        self.future_seed_row_bank_dim = (
            head_dim if gdn2_update_mode == "coupled_address_rows" else 0
        )
        self.future_seed_selector = FutureSeedSelectiveGate(
            mode=future_seed_gate_mode,
            layers=layers,
            heads=future_seed_state_heads,
            row_dim=state_row_dim,
            col_dim=state_col_dim,
        )
        blocks: List[nn.Module] = []
        for layer_id in range(layers):
            if backbone == "rwkv":
                blocks.append(
                    RWKVBlock(
                        d_model,
                        heads,
                        head_dim,
                        channel_mult,
                        layer_id=layer_id,
                        layers=layers,
                        rwkv_kernel=rwkv_kernel,
                    )
                )
            elif backbone == "rwkv7":
                blocks.append(
                    RWKV7OfficialBlock(
                        d_model,
                        heads,
                        head_dim,
                        channel_mult,
                        layer_id=layer_id,
                        layers=layers,
                        rwkv_kernel=rwkv_kernel,
                    )
                )
            elif backbone == "gdn":
                blocks.append(
                    GDNBlock(
                        d_model,
                        heads,
                        head_dim,
                        channel_mult,
                        gdn_mode=gdn_mode,
                        gdn_expand_v=gdn_expand_v,
                        gdn_progressive_base_expand_v=gdn_progressive_base_expand_v,
                        gdn_use_short_conv=gdn_use_short_conv,
                        gdn_conv_size=gdn_conv_size,
                        gdn_allow_neg_eigval=gdn_allow_neg_eigval,
                    )
                )
            elif backbone == "momentum":
                blocks.append(
                    MomentumDeltaBlock(
                        d_model,
                        heads,
                        head_dim,
                        channel_mult,
                        layer_idx=layer_id,
                        gdn_expand_v=gdn_expand_v,
                        gdn_use_short_conv=gdn_use_short_conv,
                        gdn_conv_size=gdn_conv_size,
                    )
                )
            else:
                blocks.append(
                    FLADeltaBlock(
                        d_model,
                        heads,
                        head_dim,
                        channel_mult,
                        backbone=backbone,
                        gdn_mode=gdn_mode,
                        gdn_expand_v=gdn_expand_v,
                        gdn_use_short_conv=gdn_use_short_conv,
                        gdn_conv_size=gdn_conv_size,
                        gdn_allow_neg_eigval=gdn_allow_neg_eigval,
                        gdn2_gain_budget_mode=gdn2_gain_budget_mode,
                        gdn2_gain_budget_sigma_cap=gdn2_gain_budget_sigma_cap,
                        gdn2_gain_budget_step_cap=gdn2_gain_budget_step_cap,
                        gdn2_gain_budget_sigma_cap_max=gdn2_gain_budget_sigma_cap_max,
                        gdn2_gain_budget_infeasible_policy=gdn2_gain_budget_infeasible_policy,
                        gdn2_fast_slow_decay_mode=gdn2_fast_slow_decay_mode,
                        gdn2_fast_slow_decay_kernel_size=(
                            gdn2_fast_slow_decay_kernel_size
                        ),
                        gdn2_fast_slow_decay_rho_init=(
                            gdn2_fast_slow_decay_rho_init
                        ),
                        gdn2_fast_slow_decay_current_weight_init=(
                            gdn2_fast_slow_decay_current_weight_init
                        ),
                        gdn2_precondition_mode=gdn2_precondition_mode,
                        gdn2_address_mode=gdn2_address_mode,
                        gdn2_update_mode=gdn2_update_mode,
                        gdn2_terminal_consolidation_enabled=(
                            layer_id < layers - 1
                        ),
                        gdn2_state_expert_mode=gdn2_state_expert_mode,
                        raven_num_slots=raven_num_slots,
                        raven_topk=raven_topk,
                    )
                )
        self.blocks = nn.ModuleList(blocks)
        self._initialize_gdn2_cross_layer_coordinates()

    def _initialize_gdn2_cross_layer_coordinates(self) -> None:
        if self.gdn2_cross_layer_init == "independent":
            return
        if self.gdn2_cross_layer_init != "coherent_qkv":
            raise AssertionError(
                f"unhandled cross-layer initialization: {self.gdn2_cross_layer_init}"
            )
        source_core = self.blocks[0].time_mix.core
        coordinate_modules = (
            "q_proj",
            "k_proj",
            "v_proj",
            "q_conv1d",
            "k_conv1d",
            "v_conv1d",
        )
        for block in self.blocks[1:]:
            target_core = block.time_mix.core
            for module_name in coordinate_modules:
                source_module = getattr(source_core, module_name, None)
                target_module = getattr(target_core, module_name, None)
                if source_module is None or target_module is None:
                    if source_module is not target_module:
                        raise RuntimeError(
                            f"GDN2 coordinate module mismatch for {module_name}."
                        )
                    continue
                target_module.load_state_dict(source_module.state_dict())

    def _compose_future_seed_content(
        self,
        terminal_state: torch.Tensor,
        producer_initial_state: Optional[torch.Tensor],
        *,
        receiver_layer_idx: int,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        zero = terminal_state.new_zeros(())
        zero_diag = {
            "fs3_innovation_enabled": zero,
            "fs3_innovation_scale_abs": zero,
            "fs3_innovation_fraction": zero,
            "fs3_innovation_residual_relative_rms": zero,
            "fs3_codec_enabled": zero,
            "fs3_codec_code_relative_rms": zero,
            "fs3_codec_row_attention_entropy": zero,
            "fs3_codec_row_attention_max": zero,
            "fs3_codec_row_attention_batch_std": zero,
            "fs3_codec_update_relative_rms": zero,
            "fs3_codec_residual_relative_rms": zero,
            "fs3_codec_residual_batch_std": zero,
            "fs3_address_local_enabled": zero,
            "fs3_address_local_gain_abs": zero,
            "fs3_address_local_gain_row_std": zero,
            "fs3_address_local_gain_batch_std": zero,
            "fs3_address_local_update_relative_rms": zero,
            "fs3_address_local_residual_relative_rms": zero,
            "fs3_address_local_residual_batch_std": zero,
            "fs3_basis_transport_enabled": zero,
            "fs3_basis_transport_angle_abs": zero,
            "fs3_basis_transport_row_rotation_relative_rms": zero,
            "fs3_basis_transport_col_rotation_relative_rms": zero,
            "fs3_basis_transport_state_residual_relative_rms": zero,
            "fs3_basis_transport_residual_batch_std": zero,
            "fs3_basis_transport_residual_head_std": zero,
            "fs3_basis_transport_fp32_norm_max_error": zero,
            "fs3_basis_transport_storage_norm_max_error": zero,
            "fs3_basis_transport_orthogonality_max_error": zero,
        }
        if self.future_seed_content_mode == "terminal":
            return terminal_state, zero_diag
        if self.future_seed_content_mode == "orthogonal_basis_transport":
            if self.future_seed_basis_transport is None:
                raise RuntimeError("orthogonal-basis FutureSeed module is missing")
            candidate, basis_diag = self.future_seed_basis_transport(
                terminal_state,
                edge_idx=receiver_layer_idx - 1,
            )
            return candidate, {**zero_diag, **basis_diag}
        if receiver_layer_idx < 2:
            return terminal_state, zero_diag
        if producer_initial_state is None:
            raise RuntimeError(
                f"{self.future_seed_content_mode} FutureSeed is missing the "
                "producer initial state"
            )
        if producer_initial_state.shape != terminal_state.shape:
            raise ValueError(
                "producer initial-state shape does not match its terminal state: "
                f"{tuple(producer_initial_state.shape)} != {tuple(terminal_state.shape)}"
            )
        if self.future_seed_content_mode == "producer_codec":
            if self.future_seed_producer_codec is None:
                raise RuntimeError("producer-codec FutureSeed module is missing")
            candidate, codec_diag = self.future_seed_producer_codec(
                terminal_state,
                producer_initial_state,
            )
            return candidate, {**zero_diag, **codec_diag}
        if self.future_seed_content_mode == "address_local_update":
            if self.future_seed_address_local_update is None:
                raise RuntimeError("address-local FutureSeed module is missing")
            candidate, address_local_diag = self.future_seed_address_local_update(
                terminal_state,
                producer_initial_state,
            )
            return candidate, {**zero_diag, **address_local_diag}
        if self.future_seed_content_mode != "innovation_residual":
            raise AssertionError(
                f"unhandled FutureSeed content mode: {self.future_seed_content_mode}"
            )
        assert self.future_seed_innovation_scale is not None

        terminal = terminal_state.float()
        incoming = producer_initial_state.float()
        incoming_energy = incoming.square().sum(
            dim=(-1, -2), keepdim=True
        ).clamp_min(1e-12)
        projection = (
            (terminal * incoming).sum(dim=(-1, -2), keepdim=True)
            / incoming_energy
        ) * incoming
        innovation = terminal - projection
        terminal_rms = terminal.square().mean(
            dim=(-1, -2), keepdim=True
        ).sqrt().clamp_min(1e-6)
        innovation_rms = innovation.square().mean(
            dim=(-1, -2), keepdim=True
        ).sqrt()
        innovation_floor = 0.1 * terminal_rms
        bounded_innovation = (
            innovation
            / torch.maximum(innovation_rms, innovation_floor)
            * terminal_rms
        )
        scale = torch.tanh(
            self.future_seed_innovation_scale[receiver_layer_idx - 2]
        ).to(device=terminal.device, dtype=terminal.dtype)
        residual = scale * bounded_innovation
        candidate = terminal_state + residual.to(dtype=terminal_state.dtype)
        residual_rms = residual.square().mean(
            dim=(-1, -2), keepdim=True
        ).sqrt()
        return candidate, {
            **zero_diag,
            "fs3_innovation_enabled": terminal_state.new_ones(()),
            "fs3_innovation_scale_abs": scale.detach().abs().mean().to(
                dtype=terminal_state.dtype
            ),
            "fs3_innovation_fraction": (
                innovation_rms / terminal_rms
            ).detach().mean().to(dtype=terminal_state.dtype),
            "fs3_innovation_residual_relative_rms": (
                residual_rms / terminal_rms
            ).detach().mean().to(dtype=terminal_state.dtype),
        }

    def forward(
        self,
        x: torch.Tensor,
        *,
        seed_memory: Optional[List[Optional[torch.Tensor]]] = None,
        address: Optional[torch.Tensor] = None,
        cell_order: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], Optional[List[torch.Tensor]]]:
        shared_address_residual: Optional[torch.Tensor] = None
        if self.gdn2_address_mode == "shared_namespace":
            if address is None or self.shared_address_proj is None:
                raise ValueError(
                    "shared_namespace requires a canonical anchor and shared projection"
                )
            address_rms = address.float().square().mean(
                dim=-1,
                keepdim=True,
            ).add(1e-6).sqrt()
            normalized_address = address / address_rms.to(dtype=address.dtype)
            shared_address_residual = self.shared_address_proj(normalized_address)
            address = shared_address_residual
        expected_seed_count = (
            len(self.blocks) if self.future_seed_scope == "block" else len(self.blocks) - 1
        )
        if seed_memory is not None and len(seed_memory) != expected_seed_count:
            raise ValueError(
                f"seed_memory has {len(seed_memory)} entries, expected {expected_seed_count}"
            )
        previous_state: Optional[torch.Tensor] = None
        previous_state_expert: Optional[torch.Tensor] = None
        previous_initial_state: Optional[torch.Tensor] = None
        seed_state: Optional[torch.Tensor] = None
        v_first: Optional[torch.Tensor] = None
        next_seed_memory: Optional[List[torch.Tensor]] = (
            []
            if self.future_seed_scope == "block"
            or self.future_seed_update in {"loop_residual", "loop_secant"}
            else None
        )
        gates = []
        update_gates = []
        state_norms = []
        raw_rms_means = []
        raw_rms_stds = []
        norm_gain_means = []
        norm_gain_stds = []
        memory_norms = []
        memory_delta_norms = []
        secant_enabled = []
        secant_scale_abs = []
        secant_scale_signed = []
        secant_delta_relative_rms = []
        secant_residual_relative_rms = []
        secant_residual_batch_std = []
        selective_delta_rms = []
        selective_gate_stds = []
        selective_gate_batch_stds = []
        selective_content_feature_stds = []
        selective_gate_mins = []
        selective_gate_maxs = []
        selective_seed_changes = []
        block_seed_active = []
        block_seed_raw_norms = []
        readout_scales = []
        readout_raw_norms = []
        readout_residual_norms = []
        innovation_enabled = []
        innovation_scales = []
        innovation_fractions = []
        innovation_residual_relative_rms = []
        codec_enabled = []
        codec_code_relative_rms = []
        codec_row_attention_entropy = []
        codec_row_attention_max = []
        codec_row_attention_batch_std = []
        codec_update_relative_rms = []
        codec_residual_relative_rms = []
        codec_residual_batch_std = []
        address_local_enabled = []
        address_local_gain_abs = []
        address_local_gain_row_std = []
        address_local_gain_batch_std = []
        address_local_update_relative_rms = []
        address_local_residual_relative_rms = []
        address_local_residual_batch_std = []
        basis_transport_enabled = []
        basis_transport_angle_abs = []
        basis_transport_row_rotation_relative_rms = []
        basis_transport_col_rotation_relative_rms = []
        basis_transport_state_residual_relative_rms = []
        basis_transport_residual_batch_std = []
        basis_transport_residual_head_std = []
        basis_transport_fp32_norm_max_error = []
        basis_transport_storage_norm_max_error = []
        basis_transport_orthogonality_max_error = []
        momentum_transport_fractions = []
        state_history: List[torch.Tensor] = []
        gain_budget_values: Dict[str, List[torch.Tensor]] = {}
        for layer_idx, block in enumerate(self.blocks):
            if (
                self.future_seed_readout_hop > 0
                and self.future_seed_scale > 0
                and layer_idx >= self.future_seed_readout_hop
            ):
                assert self.future_seed_readout_scale is not None
                source_idx = layer_idx - self.future_seed_readout_hop
                source_block = self.blocks[source_idx]
                if not isinstance(source_block, FLADeltaBlock):
                    raise TypeError(
                        "Compatible FutureSeed readout requires an official FLA delta block."
                    )
                source_state = state_history[source_idx]
                raw_readout = source_block.read_terminal_state(x, source_state)
                readout_scale = self.future_seed_readout_scale[source_idx].to(
                    device=x.device,
                    dtype=x.dtype,
                )
                readout_residual = (
                    raw_readout
                    * readout_scale
                    * self.future_seed_scale
                )
                x = x + readout_residual
                readout_scales.append(readout_scale.abs())
                readout_raw_norms.append(
                    raw_readout.float().norm(dim=-1).mean().to(dtype=x.dtype)
                )
                readout_residual_norms.append(
                    readout_residual.float().norm(dim=-1).mean().to(dtype=x.dtype)
                )
            initial_state = None
            candidate_seed_state: Optional[torch.Tensor] = None
            using_block_seed = (
                self.future_seed_scope == "block" and seed_memory is not None
            )
            if using_block_seed:
                prior_state = seed_memory[layer_idx]
                if prior_state is None:
                    raise ValueError(
                        f"block FutureSeed is missing the state for layer {layer_idx}"
                    )
                candidate_seed_state = prior_state.to(
                    device=x.device,
                    dtype=x.dtype,
                )
                update_gates.append(x.new_ones(()))
                block_seed_active.append(x.new_ones(()))
                block_seed_raw_norms.append(
                    candidate_seed_state.norm(dim=(-1, -2)).mean()
                )
            elif layer_idx > 0:
                assert previous_state is not None
                if self.future_seed_scale > 0:
                    if self.future_seed_update == "loop_secant":
                        assert self.future_seed_secant_raw is not None
                        assert next_seed_memory is not None
                        prior_state = (
                            None
                            if seed_memory is None
                            else seed_memory[layer_idx - 1]
                        )
                        scale = 0.5 * torch.tanh(
                            self.future_seed_secant_raw[layer_idx - 1]
                        ).to(
                            device=previous_state.device,
                            dtype=previous_state.dtype,
                        )
                        if prior_state is None:
                            delta = torch.zeros_like(previous_state)
                            bounded_delta = delta
                            residual = delta
                            candidate_seed_state = previous_state
                        else:
                            prior_state = prior_state.to(
                                device=previous_state.device,
                                dtype=previous_state.dtype,
                            )
                            delta = previous_state - prior_state
                            previous_rms = previous_state.float().square().mean(
                                dim=(-1, -2), keepdim=True
                            ).sqrt().clamp_min(1e-6)
                            delta_rms = delta.float().square().mean(
                                dim=(-1, -2), keepdim=True
                            ).sqrt()
                            bound = torch.minimum(
                                torch.ones_like(delta_rms),
                                previous_rms / delta_rms.clamp_min(1e-6),
                            )
                            bounded_delta = delta.float() * bound
                            residual = scale.float() * bounded_delta
                            candidate_seed_state = previous_state + residual.to(
                                dtype=previous_state.dtype
                            )
                        next_seed_memory.append(previous_state)
                        previous_rms = previous_state.float().square().mean(
                            dim=(-1, -2), keepdim=True
                        ).sqrt().clamp_min(1e-6)
                        delta_rms = delta.float().square().mean(
                            dim=(-1, -2), keepdim=True
                        ).sqrt()
                        residual_rms = residual.float().square().mean(
                            dim=(-1, -2), keepdim=True
                        ).sqrt()
                        residual_board = residual.float().square().mean(
                            dim=(-1, -2, -3)
                        ).sqrt()
                        secant_enabled.append(x.new_ones(()))
                        secant_scale_abs.append(scale.float().abs().mean())
                        secant_scale_signed.append(scale.float().mean())
                        secant_delta_relative_rms.append(
                            (delta_rms / previous_rms).mean()
                        )
                        secant_residual_relative_rms.append(
                            (residual_rms / previous_rms).mean()
                        )
                        secant_residual_batch_std.append(
                            residual_board.std(unbiased=False)
                        )
                        update_gates.append(x.new_ones(()))
                        memory_norms.append(
                            candidate_seed_state.norm(dim=(-1, -2)).mean()
                        )
                        memory_delta_norms.append(
                            delta.norm(dim=(-1, -2)).mean()
                        )
                        seed_state = candidate_seed_state
                    elif self.future_seed_update == "loop_residual":
                        assert self.future_seed_update_logit is not None
                        update_gate = torch.sigmoid(self.future_seed_update_logit[layer_idx - 1]).to(
                            device=previous_state.device,
                            dtype=previous_state.dtype,
                        )
                        prior_state = None if seed_memory is None else seed_memory[layer_idx - 1]
                        if prior_state is None:
                            seed_state = previous_state
                            memory_delta_norms.append(x.new_zeros(()))
                        else:
                            prior_state = prior_state.to(device=previous_state.device, dtype=previous_state.dtype)
                            delta = previous_state - prior_state
                            seed_state = prior_state + update_gate * delta
                            memory_delta_norms.append(delta.norm(dim=(-1, -2)).mean())
                        update_gates.append(update_gate.mean())
                        memory_norms.append(seed_state.norm(dim=(-1, -2)).mean())
                        assert next_seed_memory is not None
                        next_seed_memory.append(seed_state)
                    elif seed_state is None:
                        seed_state = previous_state
                        update_gates.append(x.new_ones(()))
                    elif self.future_seed_update == "learned":
                        assert self.future_seed_update_logit is not None
                        update_gate = torch.sigmoid(self.future_seed_update_logit[layer_idx - 1]).to(
                            device=previous_state.device,
                            dtype=previous_state.dtype,
                        )
                        seed_state = seed_state + update_gate * (previous_state - seed_state)
                        update_gates.append(update_gate.mean())
                    elif self.future_seed_decay <= 0:
                        seed_state = previous_state
                        update_gates.append(x.new_ones(()))
                    else:
                        keep = self.future_seed_decay
                        seed_state = keep * seed_state + (1.0 - keep) * previous_state
                        update_gates.append(x.new_tensor(1.0 - keep))
                    candidate_seed_state = seed_state
            if candidate_seed_state is not None and self.future_seed_scale > 0:
                if self.backbone == "momentum":
                    if candidate_seed_state.ndim != 5 or candidate_seed_state.shape[0] != 2:
                        raise RuntimeError(
                            "Momentum FutureSeed must carry [state, momentum] planes"
                        )
                    if self.momentum_future_seed_transport == "momentum_only":
                        candidate_seed_state = torch.stack(
                            (
                                torch.zeros_like(candidate_seed_state[0]),
                                candidate_seed_state[1],
                            ),
                            dim=0,
                        )
                        momentum_transport_fractions.append(x.new_tensor(0.5))
                    else:
                        momentum_transport_fractions.append(x.new_tensor(1.0))
                candidate_seed_state, innovation_diag = self._compose_future_seed_content(
                    candidate_seed_state,
                    previous_initial_state,
                    receiver_layer_idx=layer_idx,
                )
                innovation_enabled.append(innovation_diag["fs3_innovation_enabled"])
                innovation_scales.append(innovation_diag["fs3_innovation_scale_abs"])
                innovation_fractions.append(innovation_diag["fs3_innovation_fraction"])
                innovation_residual_relative_rms.append(
                    innovation_diag["fs3_innovation_residual_relative_rms"]
                )
                codec_enabled.append(innovation_diag["fs3_codec_enabled"])
                codec_code_relative_rms.append(
                    innovation_diag["fs3_codec_code_relative_rms"]
                )
                codec_row_attention_entropy.append(
                    innovation_diag["fs3_codec_row_attention_entropy"]
                )
                codec_row_attention_max.append(
                    innovation_diag["fs3_codec_row_attention_max"]
                )
                codec_row_attention_batch_std.append(
                    innovation_diag["fs3_codec_row_attention_batch_std"]
                )
                codec_update_relative_rms.append(
                    innovation_diag["fs3_codec_update_relative_rms"]
                )
                codec_residual_relative_rms.append(
                    innovation_diag["fs3_codec_residual_relative_rms"]
                )
                codec_residual_batch_std.append(
                    innovation_diag["fs3_codec_residual_batch_std"]
                )
                address_local_enabled.append(
                    innovation_diag["fs3_address_local_enabled"]
                )
                address_local_gain_abs.append(
                    innovation_diag["fs3_address_local_gain_abs"]
                )
                address_local_gain_row_std.append(
                    innovation_diag["fs3_address_local_gain_row_std"]
                )
                address_local_gain_batch_std.append(
                    innovation_diag["fs3_address_local_gain_batch_std"]
                )
                address_local_update_relative_rms.append(
                    innovation_diag["fs3_address_local_update_relative_rms"]
                )
                address_local_residual_relative_rms.append(
                    innovation_diag["fs3_address_local_residual_relative_rms"]
                )
                address_local_residual_batch_std.append(
                    innovation_diag["fs3_address_local_residual_batch_std"]
                )
                basis_transport_enabled.append(
                    innovation_diag["fs3_basis_transport_enabled"]
                )
                basis_transport_angle_abs.append(
                    innovation_diag["fs3_basis_transport_angle_abs"]
                )
                basis_transport_row_rotation_relative_rms.append(
                    innovation_diag[
                        "fs3_basis_transport_row_rotation_relative_rms"
                    ]
                )
                basis_transport_col_rotation_relative_rms.append(
                    innovation_diag[
                        "fs3_basis_transport_col_rotation_relative_rms"
                    ]
                )
                basis_transport_state_residual_relative_rms.append(
                    innovation_diag[
                        "fs3_basis_transport_state_residual_relative_rms"
                    ]
                )
                basis_transport_residual_batch_std.append(
                    innovation_diag["fs3_basis_transport_residual_batch_std"]
                )
                basis_transport_residual_head_std.append(
                    innovation_diag["fs3_basis_transport_residual_head_std"]
                )
                basis_transport_fp32_norm_max_error.append(
                    innovation_diag[
                        "fs3_basis_transport_fp32_norm_max_error"
                    ]
                )
                basis_transport_storage_norm_max_error.append(
                    innovation_diag[
                        "fs3_basis_transport_storage_norm_max_error"
                    ]
                )
                basis_transport_orthogonality_max_error.append(
                    innovation_diag[
                        "fs3_basis_transport_orthogonality_max_error"
                    ]
                )
                future_seed_base_logit = block.future_seed_logit
                if self.gdn2_update_mode == "paired_address_bank":
                    future_seed_base_logit = torch.cat(
                        (future_seed_base_logit, future_seed_base_logit), dim=1
                    )
                if layer_idx == 0:
                    gate = torch.sigmoid(future_seed_base_logit)
                    selective_zero = x.new_zeros(())
                    selective_diag = {
                        "fs2_gate_delta_rms": selective_zero,
                        "fs2_gate_std": selective_zero,
                        "fs2_gate_batch_std": selective_zero,
                        "fs2_content_feature_std": selective_zero,
                        "fs2_gate_min": gate.min().to(dtype=x.dtype),
                        "fs2_gate_max": gate.max().to(dtype=x.dtype),
                        "fs2_seed_relative_change": selective_zero,
                    }
                else:
                    gate, selective_diag = self.future_seed_selector(
                        candidate_seed_state,
                        base_logit=future_seed_base_logit,
                        layer_idx=layer_idx,
                    )
                gate = gate * self.future_seed_scale
                gates.append(gate.mean())
                selective_delta_rms.append(
                    selective_diag["fs2_gate_delta_rms"]
                )
                selective_gate_stds.append(
                    selective_diag["fs2_gate_std"]
                )
                selective_gate_batch_stds.append(
                    selective_diag["fs2_gate_batch_std"]
                )
                selective_content_feature_stds.append(
                    selective_diag["fs2_content_feature_std"]
                )
                selective_gate_mins.append(
                    selective_diag["fs2_gate_min"]
                )
                selective_gate_maxs.append(
                    selective_diag["fs2_gate_max"]
                )
                selective_seed_changes.append(
                    selective_diag["fs2_seed_relative_change"]
                )
                if self.future_seed_row_bank_dim > 0:
                    bank_states = candidate_seed_state.split(
                        self.future_seed_row_bank_dim,
                        dim=-2,
                    )
                    if len(bank_states) != 2:
                        raise RuntimeError(
                            "Coupled address-row FutureSeed requires exactly "
                            "two equal K-row banks"
                        )
                    bank_denoms_native = [
                        bank.square().mean(
                            dim=(-1, -2), keepdim=True
                        ).sqrt().clamp(min=1e-6)
                        for bank in bank_states
                    ]
                    bank_denoms = [
                        bank.float().square().mean(
                            dim=(-1, -2), keepdim=True
                        ).sqrt().clamp(min=1e-6)
                        for bank in bank_states
                    ]
                    normalized_state = torch.cat(
                        [
                            bank / bank_denom
                            for bank, bank_denom in zip(
                                bank_states, bank_denoms_native
                            )
                        ],
                        dim=-2,
                    )
                    denom = torch.cat(bank_denoms, dim=-2)
                elif self.gdn_progressive_base_head_v_dim > 0:
                    base_state, extra_state = candidate_seed_state.split(
                        self.gdn_progressive_base_head_v_dim,
                        dim=-2,
                    )
                    bank_states = (base_state, extra_state)
                    bank_denoms_native = [
                        bank.square().mean(dim=(-1, -2), keepdim=True).sqrt().clamp(min=1e-6)
                        for bank in bank_states
                    ]
                    bank_denoms = [
                        bank.float().square().mean(dim=(-1, -2), keepdim=True).sqrt().clamp(min=1e-6)
                        for bank in bank_states
                    ]
                    normalized_state = torch.cat(
                        [bank / bank_denom for bank, bank_denom in zip(bank_states, bank_denoms_native)],
                        dim=-2,
                    )
                    denom = torch.cat(bank_denoms, dim=-2)
                else:
                    denom_native = candidate_seed_state.square().mean(
                        dim=(-1, -2), keepdim=True
                    ).sqrt().clamp(min=1e-6)
                    denom = candidate_seed_state.float().square().mean(
                        dim=(-1, -2), keepdim=True
                    ).sqrt().clamp(min=1e-6)
                    normalized_state = candidate_seed_state / denom_native
                raw_rms_means.append(denom.mean().to(dtype=x.dtype))
                batch_dim = 1 if candidate_seed_state.ndim == 5 else 0
                raw_rms_stds.append(
                    denom.std(dim=batch_dim, unbiased=False).mean().to(dtype=x.dtype)
                )
                if self.future_seed_norm_mode == "adaptive_rms":
                    assert self.future_seed_norm_slope is not None
                    assert self.future_seed_norm_bias is not None
                    if layer_idx == 0:
                        raise ValueError(
                            "adaptive_rms is not supported for layer-zero block FutureSeed"
                        )
                    log_rms = denom.log().clamp(min=-6.0, max=6.0)
                    slope = self.future_seed_norm_slope[layer_idx - 1].to(
                        device=candidate_seed_state.device,
                        dtype=log_rms.dtype,
                    )
                    bias = self.future_seed_norm_bias[layer_idx - 1].to(
                        device=candidate_seed_state.device,
                        dtype=log_rms.dtype,
                    )
                    norm_gain = torch.exp(0.5 * torch.tanh(slope * log_rms + bias))
                else:
                    norm_gain = torch.ones_like(denom)
                norm_gain_means.append(norm_gain.mean().to(dtype=x.dtype))
                norm_gain_stds.append(norm_gain.std(dim=0, unbiased=False).mean().to(dtype=x.dtype))
                if self.future_seed_row_bank_dim > 0:
                    norm_gain_state = torch.repeat_interleave(
                        norm_gain,
                        self.future_seed_row_bank_dim,
                        dim=-2,
                    )
                elif self.gdn_progressive_base_head_v_dim > 0:
                    norm_gain_state = torch.repeat_interleave(
                        norm_gain,
                        self.gdn_progressive_base_head_v_dim,
                        dim=-2,
                    )
                else:
                    norm_gain_state = norm_gain
                initial_state = normalized_state * gate * norm_gain_state.to(dtype=normalized_state.dtype)
                state_norms.append(initial_state.norm(dim=(-1, -2)).mean())
            elif layer_idx > 0:
                gates.append(x.new_zeros(()))
                update_gates.append(x.new_zeros(()))
                state_norms.append(x.new_zeros(()))
                if (
                    next_seed_memory is not None
                    and self.future_seed_scope == "layer"
                ):
                    next_seed_memory.append(previous_state)
            if self.backbone == "rwkv7":
                assert isinstance(block, RWKV7OfficialBlock)
                if self.activation_checkpoint and self.training and torch.is_grad_enabled():
                    if initial_state is None and v_first is None:
                        x, previous_state, v_first = torch_checkpoint(
                            lambda block_input: block(block_input, initial_state=None, v_first=None),
                            x,
                            use_reentrant=False,
                            preserve_rng_state=False,
                        )
                    elif initial_state is None:
                        assert v_first is not None
                        x, previous_state, v_first = torch_checkpoint(
                            lambda block_input, first_value: block(
                                block_input,
                                initial_state=None,
                                v_first=first_value,
                            ),
                            x,
                            v_first,
                            use_reentrant=False,
                            preserve_rng_state=False,
                        )
                    else:
                        assert v_first is not None
                        x, previous_state, v_first = torch_checkpoint(
                            lambda block_input, block_state, first_value: block(
                                block_input,
                                initial_state=block_state,
                                v_first=first_value,
                            ),
                            x,
                            initial_state,
                            v_first,
                            use_reentrant=False,
                            preserve_rng_state=False,
                        )
                else:
                    x, previous_state, v_first = block(
                        x,
                        initial_state=initial_state,
                        v_first=v_first,
                    )
            elif self.activation_checkpoint and self.training and torch.is_grad_enabled():
                if address is not None:
                    raise RuntimeError(
                        "GDN2 address mode currently forbids activation checkpointing"
                    )
                if initial_state is None:
                    x, previous_state = torch_checkpoint(
                        lambda block_input: block(block_input, initial_state=None),
                        x,
                        use_reentrant=False,
                        preserve_rng_state=False,
                    )
                else:
                    x, previous_state = torch_checkpoint(
                        lambda block_input, block_state: block(block_input, initial_state=block_state),
                        x,
                        initial_state,
                        use_reentrant=False,
                        preserve_rng_state=False,
                    )
            elif isinstance(block, FLADeltaBlock):
                x, previous_state = block(
                    x,
                    initial_state=initial_state,
                    state_expert_initial_state=(
                        previous_state_expert
                        if layer_idx > 0 and self.future_seed_scale > 0
                        else None
                    ),
                    address=address,
                    cell_order=cell_order,
                )
            else:
                x, previous_state = block(x, initial_state=initial_state)
            assert previous_state is not None
            previous_initial_state = initial_state
            state_history.append(previous_state)
            if isinstance(block, FLADeltaBlock):
                if block.state_expert is not None:
                    if block.last_state_expert_terminal is None:
                        raise RuntimeError("Dual-state expert did not return a terminal state")
                    previous_state_expert = block.last_state_expert_terminal
                else:
                    previous_state_expert = None
                for key, value in block.time_mix.last_gain_budget_diag.items():
                    gain_budget_values.setdefault(key, []).append(value)
                for key, value in block.last_state_expert_diag.items():
                    gain_budget_values.setdefault(key, []).append(value)
            elif isinstance(block, MomentumDeltaBlock):
                for key, value in block.time_mix.last_diagnostics.items():
                    gain_budget_values.setdefault(key, []).append(value)
            if self.future_seed_scope == "block":
                assert next_seed_memory is not None
                next_seed_memory.append(previous_state)

        if gates:
            out = {
                "fs_gate_mean": torch.stack(gates).mean(),
                "fs_update_mean": torch.stack(update_gates).mean(),
                "fs_state_norm": torch.stack(state_norms).mean(),
                "fs_decay": x.new_tensor(self.future_seed_decay),
            }
            if memory_norms:
                out["fs_memory_norm"] = torch.stack(memory_norms).mean()
            if memory_delta_norms:
                out["fs_memory_delta_norm"] = torch.stack(memory_delta_norms).mean()
            out["fs2_loop_secant_enabled"] = (
                torch.stack(secant_enabled).max()
                if secant_enabled
                else x.new_zeros(())
            )
            out["fs2_loop_secant_scale_abs"] = (
                torch.stack(secant_scale_abs).mean()
                if secant_scale_abs
                else x.new_zeros(())
            )
            out["fs2_loop_secant_scale_signed"] = (
                torch.stack(secant_scale_signed).mean()
                if secant_scale_signed
                else x.new_zeros(())
            )
            out["fs2_loop_secant_delta_relative_rms"] = (
                torch.stack(secant_delta_relative_rms).mean()
                if secant_delta_relative_rms
                else x.new_zeros(())
            )
            out["fs2_loop_secant_residual_relative_rms"] = (
                torch.stack(secant_residual_relative_rms).mean()
                if secant_residual_relative_rms
                else x.new_zeros(())
            )
            out["fs2_loop_secant_residual_batch_std"] = (
                torch.stack(secant_residual_batch_std).mean()
                if secant_residual_batch_std
                else x.new_zeros(())
            )
            if raw_rms_means:
                out["fs_raw_rms_mean"] = torch.stack(raw_rms_means).mean()
                out["fs_raw_rms_std"] = torch.stack(raw_rms_stds).mean()
                out["fs_norm_gain_mean"] = torch.stack(norm_gain_means).mean()
                out["fs_norm_gain_std"] = torch.stack(norm_gain_stds).mean()
            if selective_delta_rms:
                out["fs2_gate_delta_rms"] = torch.stack(
                    selective_delta_rms
                ).mean()
                out["fs2_gate_std"] = torch.stack(selective_gate_stds).mean()
                out["fs2_gate_batch_std"] = torch.stack(
                    selective_gate_batch_stds
                ).mean()
                out["fs2_content_feature_std"] = torch.stack(
                    selective_content_feature_stds
                ).mean()
                out["fs2_gate_min"] = torch.stack(selective_gate_mins).min()
                out["fs2_gate_max"] = torch.stack(selective_gate_maxs).max()
                out["fs2_seed_relative_change"] = torch.stack(
                    selective_seed_changes
                ).mean()
            out["fs2_block_seed_active"] = (
                torch.stack(block_seed_active).mean()
                if block_seed_active
                else x.new_zeros(())
            )
            out["fs2_block_seed_raw_norm"] = (
                torch.stack(block_seed_raw_norms).mean()
                if block_seed_raw_norms
                else x.new_zeros(())
            )
            out["fs2_readout_enabled"] = x.new_tensor(
                1.0 if self.future_seed_readout_hop > 0 else 0.0
            )
            out["fs2_readout_scale_abs"] = (
                torch.stack(readout_scales).mean()
                if readout_scales
                else x.new_zeros(())
            )
            out["fs2_readout_raw_norm"] = (
                torch.stack(readout_raw_norms).mean()
                if readout_raw_norms
                else x.new_zeros(())
            )
            out["fs2_readout_residual_norm"] = (
                torch.stack(readout_residual_norms).mean()
                if readout_residual_norms
                else x.new_zeros(())
            )
            out["fs3_innovation_enabled"] = (
                torch.stack(innovation_enabled).max()
                if innovation_enabled
                else x.new_zeros(())
            )
            out["fs3_innovation_scale_abs"] = (
                torch.stack(innovation_scales).mean()
                if innovation_scales
                else x.new_zeros(())
            )
            out["fs3_innovation_fraction"] = (
                torch.stack(innovation_fractions).mean()
                if innovation_fractions
                else x.new_zeros(())
            )
            out["fs3_innovation_residual_relative_rms"] = (
                torch.stack(innovation_residual_relative_rms).mean()
                if innovation_residual_relative_rms
                else x.new_zeros(())
            )
            out["fs3_codec_enabled"] = (
                torch.stack(codec_enabled).max()
                if codec_enabled
                else x.new_zeros(())
            )
            out["fs3_codec_code_relative_rms"] = (
                torch.stack(codec_code_relative_rms).mean()
                if codec_code_relative_rms
                else x.new_zeros(())
            )
            out["fs3_codec_row_attention_entropy"] = (
                torch.stack(codec_row_attention_entropy).mean()
                if codec_row_attention_entropy
                else x.new_zeros(())
            )
            out["fs3_codec_row_attention_max"] = (
                torch.stack(codec_row_attention_max).mean()
                if codec_row_attention_max
                else x.new_zeros(())
            )
            out["fs3_codec_row_attention_batch_std"] = (
                torch.stack(codec_row_attention_batch_std).mean()
                if codec_row_attention_batch_std
                else x.new_zeros(())
            )
            out["fs3_codec_update_relative_rms"] = (
                torch.stack(codec_update_relative_rms).mean()
                if codec_update_relative_rms
                else x.new_zeros(())
            )
            out["fs3_codec_residual_relative_rms"] = (
                torch.stack(codec_residual_relative_rms).mean()
                if codec_residual_relative_rms
                else x.new_zeros(())
            )
            out["fs3_codec_residual_batch_std"] = (
                torch.stack(codec_residual_batch_std).mean()
                if codec_residual_batch_std
                else x.new_zeros(())
            )
            out["fs3_address_local_enabled"] = (
                torch.stack(address_local_enabled).max()
                if address_local_enabled
                else x.new_zeros(())
            )
            out["fs3_address_local_gain_abs"] = (
                torch.stack(address_local_gain_abs).mean()
                if address_local_gain_abs
                else x.new_zeros(())
            )
            out["fs3_address_local_gain_row_std"] = (
                torch.stack(address_local_gain_row_std).mean()
                if address_local_gain_row_std
                else x.new_zeros(())
            )
            out["fs3_address_local_gain_batch_std"] = (
                torch.stack(address_local_gain_batch_std).mean()
                if address_local_gain_batch_std
                else x.new_zeros(())
            )
            out["fs3_address_local_update_relative_rms"] = (
                torch.stack(address_local_update_relative_rms).mean()
                if address_local_update_relative_rms
                else x.new_zeros(())
            )
            out["fs3_address_local_residual_relative_rms"] = (
                torch.stack(address_local_residual_relative_rms).mean()
                if address_local_residual_relative_rms
                else x.new_zeros(())
            )
            out["fs3_address_local_residual_batch_std"] = (
                torch.stack(address_local_residual_batch_std).mean()
                if address_local_residual_batch_std
                else x.new_zeros(())
            )
            out["fs3_basis_transport_enabled"] = (
                torch.stack(basis_transport_enabled).max()
                if basis_transport_enabled
                else x.new_zeros(())
            )
            out["fs3_basis_transport_angle_abs"] = (
                torch.stack(basis_transport_angle_abs).mean()
                if basis_transport_angle_abs
                else x.new_zeros(())
            )
            out["fs3_basis_transport_angle_abs_min"] = (
                torch.stack(basis_transport_angle_abs).min()
                if basis_transport_angle_abs
                else x.new_zeros(())
            )
            out["fs3_basis_transport_row_rotation_relative_rms"] = (
                torch.stack(basis_transport_row_rotation_relative_rms).mean()
                if basis_transport_row_rotation_relative_rms
                else x.new_zeros(())
            )
            out["fs3_basis_transport_row_rotation_relative_rms_min"] = (
                torch.stack(basis_transport_row_rotation_relative_rms).min()
                if basis_transport_row_rotation_relative_rms
                else x.new_zeros(())
            )
            out["fs3_basis_transport_col_rotation_relative_rms"] = (
                torch.stack(basis_transport_col_rotation_relative_rms).mean()
                if basis_transport_col_rotation_relative_rms
                else x.new_zeros(())
            )
            out["fs3_basis_transport_col_rotation_relative_rms_min"] = (
                torch.stack(basis_transport_col_rotation_relative_rms).min()
                if basis_transport_col_rotation_relative_rms
                else x.new_zeros(())
            )
            out["fs3_basis_transport_state_residual_relative_rms"] = (
                torch.stack(basis_transport_state_residual_relative_rms).mean()
                if basis_transport_state_residual_relative_rms
                else x.new_zeros(())
            )
            out["fs3_basis_transport_state_residual_relative_rms_min"] = (
                torch.stack(basis_transport_state_residual_relative_rms).min()
                if basis_transport_state_residual_relative_rms
                else x.new_zeros(())
            )
            out["fs3_basis_transport_residual_batch_std"] = (
                torch.stack(basis_transport_residual_batch_std).mean()
                if basis_transport_residual_batch_std
                else x.new_zeros(())
            )
            out["fs3_basis_transport_residual_head_std"] = (
                torch.stack(basis_transport_residual_head_std).mean()
                if basis_transport_residual_head_std
                else x.new_zeros(())
            )
            out["fs3_basis_transport_fp32_norm_max_error"] = (
                torch.stack(basis_transport_fp32_norm_max_error).max()
                if basis_transport_fp32_norm_max_error
                else x.new_zeros(())
            )
            out["fs3_basis_transport_storage_norm_max_error"] = (
                torch.stack(basis_transport_storage_norm_max_error).max()
                if basis_transport_storage_norm_max_error
                else x.new_zeros(())
            )
            out["fs3_basis_transport_orthogonality_max_error"] = (
                torch.stack(basis_transport_orthogonality_max_error).max()
                if basis_transport_orthogonality_max_error
                else x.new_zeros(())
            )
            out["momentum_future_seed_transport_fraction"] = (
                torch.stack(momentum_transport_fractions).mean()
                if momentum_transport_fractions
                else x.new_zeros(())
            )
            for key, values in gain_budget_values.items():
                stacked = torch.stack([value.float() for value in values])
                out[key] = (
                    stacked.sum()
                    if key.endswith("_sum")
                    else stacked.max()
                    if key.endswith("_max")
                    else stacked.min()
                    if key.endswith("_min")
                    else stacked.mean()
                )
            if self.shared_address_proj is not None:
                assert shared_address_residual is not None
                out["gdn3_shared_address_weight_rms"] = (
                    self.shared_address_proj.weight.float().square().mean().sqrt()
                ).to(dtype=x.dtype)
                out["gdn3_shared_address_residual_rms"] = (
                    shared_address_residual.float().square().mean().sqrt()
                ).detach().to(dtype=x.dtype)
                out["gdn3_shared_address_token_std"] = (
                    shared_address_residual.float().std(dim=1, unbiased=False).mean()
                ).detach().to(dtype=x.dtype)
            return x, out, next_seed_memory
        zero = x.new_zeros(())
        zero_out = {
            "fs_gate_mean": zero,
            "fs_update_mean": zero,
            "fs_state_norm": zero,
            "fs_decay": zero,
            "fs_raw_rms_mean": zero,
            "fs_raw_rms_std": zero,
            "fs_norm_gain_mean": zero,
            "fs_norm_gain_std": zero,
            "fs2_gate_delta_rms": zero,
            "fs2_gate_std": zero,
            "fs2_gate_batch_std": zero,
            "fs2_content_feature_std": zero,
            "fs2_gate_min": zero,
            "fs2_gate_max": zero,
            "fs2_seed_relative_change": zero,
            "fs2_block_seed_active": zero,
            "fs2_block_seed_raw_norm": zero,
            "fs2_readout_enabled": zero,
            "fs2_readout_scale_abs": zero,
            "fs2_readout_raw_norm": zero,
            "fs2_readout_residual_norm": zero,
            "fs2_loop_secant_enabled": zero,
            "fs2_loop_secant_scale_abs": zero,
            "fs2_loop_secant_scale_signed": zero,
            "fs2_loop_secant_delta_relative_rms": zero,
            "fs2_loop_secant_residual_relative_rms": zero,
            "fs2_loop_secant_residual_batch_std": zero,
            "fs3_innovation_enabled": zero,
            "fs3_innovation_scale_abs": zero,
            "fs3_innovation_fraction": zero,
            "fs3_innovation_residual_relative_rms": zero,
            "fs3_codec_enabled": zero,
            "fs3_codec_code_relative_rms": zero,
            "fs3_codec_row_attention_entropy": zero,
            "fs3_codec_row_attention_max": zero,
            "fs3_codec_row_attention_batch_std": zero,
            "fs3_codec_update_relative_rms": zero,
            "fs3_codec_residual_relative_rms": zero,
            "fs3_codec_residual_batch_std": zero,
            "fs3_address_local_enabled": zero,
            "fs3_address_local_gain_abs": zero,
            "fs3_address_local_gain_row_std": zero,
            "fs3_address_local_gain_batch_std": zero,
            "fs3_address_local_update_relative_rms": zero,
            "fs3_address_local_residual_relative_rms": zero,
            "fs3_address_local_residual_batch_std": zero,
            "fs3_basis_transport_enabled": zero,
            "fs3_basis_transport_angle_abs": zero,
            "fs3_basis_transport_angle_abs_min": zero,
            "fs3_basis_transport_row_rotation_relative_rms": zero,
            "fs3_basis_transport_row_rotation_relative_rms_min": zero,
            "fs3_basis_transport_col_rotation_relative_rms": zero,
            "fs3_basis_transport_col_rotation_relative_rms_min": zero,
            "fs3_basis_transport_state_residual_relative_rms": zero,
            "fs3_basis_transport_state_residual_relative_rms_min": zero,
            "fs3_basis_transport_residual_batch_std": zero,
            "fs3_basis_transport_residual_head_std": zero,
            "fs3_basis_transport_fp32_norm_max_error": zero,
            "fs3_basis_transport_storage_norm_max_error": zero,
            "fs3_basis_transport_orthogonality_max_error": zero,
            "momentum_future_seed_transport_fraction": zero,
        }
        for key, values in gain_budget_values.items():
            stacked = torch.stack([value.float() for value in values])
            zero_out[key] = (
                stacked.max()
                if key.endswith("_max")
                else stacked.min()
                if key.endswith("_min")
                else stacked.mean()
            )
        return (
            x,
            zero_out,
            next_seed_memory,
        )


class FeatureNoiseBuffer:
    def __init__(self, *, capacity: int, feature_dim: int, device: torch.device, dtype: torch.dtype = torch.float32) -> None:
        self.capacity = int(capacity)
        self.feature_dim = int(feature_dim)
        self.device = device
        self.dtype = dtype
        self.data = torch.empty(self.capacity, self.feature_dim, device=device, dtype=dtype)
        self.count = 0
        self.write_pos = 0

    def add(self, features: torch.Tensor, *, max_items: int) -> None:
        if self.capacity <= 0 or max_items <= 0:
            return
        flat = features.detach().reshape(-1, self.feature_dim).to(device=self.device, dtype=self.dtype)
        if flat.shape[0] > max_items:
            idx = torch.randperm(flat.shape[0], device=flat.device)[:max_items]
            flat = flat.index_select(0, idx)
        n_items = min(flat.shape[0], self.capacity)
        if n_items <= 0:
            return
        flat = flat[:n_items]
        first = min(n_items, self.capacity - self.write_pos)
        self.data[self.write_pos : self.write_pos + first].copy_(flat[:first])
        remaining = n_items - first
        if remaining > 0:
            self.data[:remaining].copy_(flat[first:])
        self.write_pos = (self.write_pos + n_items) % self.capacity
        self.count = min(self.capacity, self.count + n_items)

    def sample_diff_like(self, reference: torch.Tensor) -> torch.Tensor:
        if self.count < 2:
            return torch.zeros_like(reference)
        flat_ref = reference.detach().reshape(-1, self.feature_dim)
        n_items = flat_ref.shape[0]
        idx_a = torch.randint(0, self.count, (n_items,), device=self.device)
        idx_b = torch.randint(0, self.count, (n_items,), device=self.device)
        diff = self.data.index_select(0, idx_a) - self.data.index_select(0, idx_b)
        diff_rms = diff.square().mean(dim=-1, keepdim=True).sqrt().clamp(min=1e-6)
        ref_rms = flat_ref.to(self.dtype).square().mean(dim=-1, keepdim=True).sqrt().clamp(min=1e-6)
        perturb = diff / diff_rms * ref_rms
        return perturb.to(dtype=reference.dtype).reshape_as(reference)

    def state_dict(self) -> Dict[str, Any]:
        return {
            "capacity": self.capacity,
            "feature_dim": self.feature_dim,
            "count": self.count,
            "write_pos": self.write_pos,
            "data": self.data.detach().cpu(),
            "dtype": str(self.dtype),
        }

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        if int(state.get("capacity", self.capacity)) != self.capacity:
            raise ValueError("FeatureNoiseBuffer capacity mismatch in training checkpoint")
        if int(state.get("feature_dim", self.feature_dim)) != self.feature_dim:
            raise ValueError("FeatureNoiseBuffer feature_dim mismatch in training checkpoint")
        data = state.get("data")
        if data is not None:
            self.data.copy_(data.to(device=self.device, dtype=self.dtype))
        self.count = int(state.get("count", 0))
        self.write_pos = int(state.get("write_pos", 0))


def train_checkpoint_dir(args: argparse.Namespace) -> Path:
    if str(args.train_checkpoint_dir).strip():
        return Path(args.train_checkpoint_dir)
    return Path(args.out_dir).parent / "checkpoints"


def capture_rng_state(device: torch.device) -> Dict[str, Any]:
    state: Dict[str, Any] = {"torch_cpu": torch.get_rng_state()}
    if device.type == "cuda":
        state["torch_cuda"] = torch.cuda.get_rng_state(device)
    return state


def restore_rng_state(state: Dict[str, Any], device: torch.device) -> None:
    if "torch_cpu" in state:
        torch.set_rng_state(state["torch_cpu"].cpu())
    if device.type == "cuda" and "torch_cuda" in state:
        torch.cuda.set_rng_state(state["torch_cuda"].cpu(), device)


def save_training_checkpoint(
    *,
    args: argparse.Namespace,
    model: "FutureSeedLoopSudoku",
    opt: torch.optim.Optimizer,
    feature_buffer: FeatureNoiseBuffer,
    rng: random.Random,
    device: torch.device,
    global_step: int,
    stage_idx: int,
    holes_min: int,
    holes_max: int,
    t0: float,
    checkpoint_evals: Dict[str, Any],
    last_metrics: Dict[str, Any],
    reason: str,
) -> Path:
    out_dir = train_checkpoint_dir(args)
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": 1,
        "saved_at_step": int(global_step),
        "stage": int(stage_idx),
        "holes_min": int(holes_min),
        "holes_max": int(holes_max),
        "elapsed_sec": time.time() - t0,
        "reason": reason,
        "args": vars(args),
        "model": model.state_dict(),
        "optimizer": opt.state_dict(),
        "feature_buffer": feature_buffer.state_dict(),
        "rng_python": rng.getstate(),
        "rng_torch": capture_rng_state(device),
        "checkpoint_evals": checkpoint_evals,
        "last_metrics": last_metrics,
    }
    step_path = out_dir / f"train_state_step{global_step:06d}.pt"
    tmp_path = out_dir / f".train_state_step{global_step:06d}.pt.tmp"
    torch.save(payload, tmp_path)
    tmp_path.replace(step_path)
    latest_tmp = out_dir / ".latest.pt.tmp"
    latest_path = out_dir / "latest.pt"
    torch.save(payload, latest_tmp)
    latest_tmp.replace(latest_path)
    return step_path


def load_training_checkpoint(
    path: str,
    *,
    model: "FutureSeedLoopSudoku",
    opt: torch.optim.Optimizer,
    feature_buffer: FeatureNoiseBuffer,
    rng: random.Random,
    device: torch.device,
    expected_args: argparse.Namespace | None = None,
) -> Dict[str, Any]:
    def parse_curriculum_spec(raw: str) -> List[Tuple[int, int, int]]:
        stages = []
        for item in raw.split(","):
            item = item.strip()
            if not item:
                continue
            span, duration = item.split(":", 1)
            lo, hi = span.split("-", 1)
            stage = (int(lo), int(hi), int(duration))
            if stage[0] < 1 or stage[1] < stage[0] or stage[2] < 1:
                raise RuntimeError(
                    f"Invalid checkpoint curriculum stage: {item}"
                )
            stages.append(stage)
        if not stages:
            raise RuntimeError("Exact checkpoint resume requires hole_stages")
        return stages

    checkpoint_sha256 = ""
    checkpoint_source_sha = ""
    if expected_args is not None:
        expected_checkpoint_sha256 = str(
            expected_args.resume_train_checkpoint_sha256
        ).strip()
        if not expected_checkpoint_sha256:
            raise RuntimeError(
                "Exact checkpoint resume requires "
                "--resume_train_checkpoint_sha256"
            )
        digest = hashlib.sha256()
        with open(path, "rb") as checkpoint_file:
            for chunk in iter(
                lambda: checkpoint_file.read(8 * 1024 * 1024),
                b"",
            ):
                digest.update(chunk)
        checkpoint_sha256 = digest.hexdigest()
        if checkpoint_sha256 != expected_checkpoint_sha256:
            raise RuntimeError(
                "Resume checkpoint SHA256 mismatch: "
                f"{checkpoint_sha256} != {expected_checkpoint_sha256}"
            )
        expected_source_sha = str(
            expected_args.resume_train_source_sha
        ).strip()
        if not expected_source_sha:
            raise RuntimeError(
                "Exact checkpoint resume requires --resume_train_source_sha"
            )
        checkpoint_path = Path(path).resolve()
        checkpoint_run_name = checkpoint_path.parents[1].name
        persistent_root = Path(
            os.environ.get("PERSIST_ROOT", str(Path(__file__).resolve().parents[2]))
        ).resolve()
        source_run_dir = (
            persistent_root
            / "runs"
            / checkpoint_run_name
        )
        source_head_path = source_run_dir / "source_HEAD.txt"
        if not source_head_path.is_file():
            raise RuntimeError(
                f"Resume source provenance is missing: {source_head_path}"
            )
        checkpoint_source_sha = source_head_path.read_text(
            encoding="utf-8"
        ).strip()
        if checkpoint_source_sha != expected_source_sha:
            raise RuntimeError(
                "Resume source SHA mismatch: "
                f"{checkpoint_source_sha} != {expected_source_sha}"
            )
        source_config_path = source_run_dir / "config.json"
        source_patch_path = source_run_dir / "source.patch"
        if not source_config_path.is_file():
            raise RuntimeError(
                f"Resume source config is missing: {source_config_path}"
            )
        source_config = json.loads(
            source_config_path.read_text(encoding="utf-8")
        )
        if (
            source_config.get("run_name") != checkpoint_run_name
            or source_config.get("git_sha") != expected_source_sha
            or source_config.get("git_dirty") is not False
        ):
            raise RuntimeError(
                "Resume run was not recorded from the expected clean source: "
                f"{source_config}"
            )
        if not source_patch_path.is_file() or source_patch_path.stat().st_size != 0:
            raise RuntimeError(
                "Exact resume requires an empty recorded source.patch: "
                f"{source_patch_path}"
            )
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    resume_contract: Dict[str, Any] = {}
    if expected_args is not None:
        contract_fields = (
            "batch",
            "grad_accum_steps",
            "d_model",
            "layers",
            "heads",
            "head_dim",
            "channel_mult",
            "l_cycles",
            "max_loops",
            "backbone",
            "gdn_mode",
            "gdn_expand_v",
            "gdn_use_short_conv",
            "gdn_conv_size",
            "gdn_allow_neg_eigval",
            "gdn2_precondition_mode",
            "gdn2_address_mode",
            "gdn2_update_mode",
            "gdn2_state_expert_mode",
            "gdn2_cross_layer_init",
            "raven_num_slots",
            "raven_topk",
            "cell_order_train",
            "future_seed_scale",
            "future_seed_decay",
            "future_seed_update",
            "future_seed_norm_mode",
            "future_seed_gate_mode",
            "future_seed_scope",
            "future_seed_readout_hop",
            "future_seed_content_mode",
            "momentum_future_seed_transport",
            "future_seed_gradient_mode",
            "lambda_",
            "loop_update_mode",
            "loop_update_gate_init",
            "loop_loss",
            "loop_loss_start",
            "loop_loss_power",
            "loop_loss_min_weight",
            "loop_feedback_scale",
            "loop_feedback_detach",
            "loop_feedback_corrupt_prob",
            "loop_feedback_corrupt_mix",
            "loop_feedback_corrupt_mode",
            "loop_time_scale",
            "scratch_mode",
            "scratch_scale",
            "scratch_noise_scale",
            "scratch_gauss_weight",
            "scratch_gauss_projections",
            "scratch_gate_bias",
            "scratch_decay_bias",
            "hidden_agg_noise_scale",
            "hidden_agg_noise_temp",
            "hidden_agg_noise_detach",
            "hidden_agg_noise_mode",
            "hidden_agg_noise_topk",
            "hidden_agg_noise_max_norm",
            "exact_margin_weight",
            "exact_margin_tau",
            "exact_margin_target",
            "exact_margin_start_step",
            "activation_checkpoint",
            "forward_dtype",
            "optimizer_contract",
            "lr",
            "weight_decay",
            "blank_loss_weight",
            "noise_scale",
            "rollout_noise_scale",
            "feature_buffer_size",
            "feature_buffer_add",
            "official_sudoku_data_dir",
            "official_sudoku_train_split",
            "official_sudoku_train_indices",
            "hole_pattern",
            "gdn_progressive_base_expand_v",
            "fla_strict_official",
            "shared_shell_init_seed",
            "seed",
        )
        legacy_missing_defaults = {
            "future_seed_gate_mode": "head",
            "future_seed_scope": "layer",
            "future_seed_readout_hop": 0,
            "future_seed_content_mode": "terminal",
            "momentum_future_seed_transport": "full_state",
            "future_seed_gradient_mode": "canonical",
            "gdn2_precondition_mode": "none",
            "gdn2_address_mode": "none",
            "gdn2_update_mode": "none",
            "gdn2_state_expert_mode": "none",
            "gdn2_cross_layer_init": "independent",
            "raven_num_slots": 0,
            "raven_topk": 0,
            "cell_order_train": "row_major",
        }
        saved_args = checkpoint.get("args")
        if not isinstance(saved_args, dict):
            raise RuntimeError("Exact checkpoint resume requires saved args")
        mismatches = {}
        accepted_legacy_defaults = {}
        accepted_future_seed_content_upgrade = False
        accepted_future_seed_gradient_upgrade = False
        accepted_future_seed_update_upgrade = False
        accepted_gdn2_update_upgrade = False
        accepted_gdn2_state_expert_upgrade = False
        for field in contract_fields:
            current_value = getattr(expected_args, field)
            if field not in saved_args:
                if (
                    field == "future_seed_gradient_mode"
                    and bool(expected_args.resume_allow_future_seed_gradient_upgrade)
                    and current_value == "opening_projection"
                ):
                    accepted_future_seed_gradient_upgrade = True
                    continue
                if (
                    field == "future_seed_content_mode"
                    and bool(expected_args.resume_allow_future_seed_content_upgrade)
                    and current_value in {
                        "innovation_residual",
                        "producer_codec",
                        "address_local_update",
                        "orthogonal_basis_transport",
                    }
                ):
                    accepted_future_seed_content_upgrade = True
                    continue
                if (
                    field == "gdn2_update_mode"
                    and bool(expected_args.resume_allow_gdn2_update_upgrade)
                    and current_value in {
                        "log_spd_metric",
                        "coherent_delta",
                        "state_feedback",
                        "terminal_consolidation",
                        "orthogonal_chunk_state",
                        "orthogonal_head_write",
                        "adaptive_signed_erase",
                        "bi_axis_value_decay",
                        "gauge_balanced_bi_axis",
                        "raven_routed_gdn",
                        "paired_address_bank",
                        "coupled_address_rows",
                        "interleaved_write",
                    }
                ):
                    accepted_gdn2_update_upgrade = True
                    continue
                if (
                    field == "gdn2_state_expert_mode"
                    and bool(expected_args.resume_allow_gdn2_state_expert_upgrade)
                    and current_value in {"dual_state", "raven_write_control"}
                ):
                    accepted_gdn2_state_expert_upgrade = True
                    continue
                if (
                    field in legacy_missing_defaults
                    and current_value == legacy_missing_defaults[field]
                ):
                    accepted_legacy_defaults[field] = current_value
                    continue
                mismatches[field] = {
                    "saved": "__MISSING__",
                    "current": current_value,
                }
                continue
            saved_value = saved_args[field]
            if isinstance(current_value, float):
                matches = math.isclose(
                    float(saved_value),
                    current_value,
                    rel_tol=0.0,
                    abs_tol=1e-12,
                )
            else:
                matches = saved_value == current_value
            if (
                not matches
                and field == "future_seed_update"
                and bool(expected_args.resume_allow_future_seed_update_upgrade)
                and saved_value == "fixed"
                and current_value == "loop_secant"
            ):
                accepted_future_seed_update_upgrade = True
                matches = True
            if (
                not matches
                and field == "future_seed_gradient_mode"
                and bool(expected_args.resume_allow_future_seed_gradient_upgrade)
                and saved_value == "canonical"
                and current_value == "opening_projection"
            ):
                accepted_future_seed_gradient_upgrade = True
                matches = True
            if (
                not matches
                and field == "future_seed_content_mode"
                and bool(expected_args.resume_allow_future_seed_content_upgrade)
                and saved_value == "terminal"
                and current_value in {
                    "innovation_residual",
                    "producer_codec",
                    "address_local_update",
                    "orthogonal_basis_transport",
                }
            ):
                accepted_future_seed_content_upgrade = True
                matches = True
            if (
                not matches
                and field == "gdn2_update_mode"
                and bool(expected_args.resume_allow_gdn2_update_upgrade)
                and saved_value == "none"
                and current_value in {
                    "log_spd_metric",
                    "coherent_delta",
                    "state_feedback",
                    "terminal_consolidation",
                    "orthogonal_chunk_state",
                    "orthogonal_head_write",
                    "adaptive_signed_erase",
                    "bi_axis_value_decay",
                    "gauge_balanced_bi_axis",
                    "raven_routed_gdn",
                    "paired_address_bank",
                    "coupled_address_rows",
                    "interleaved_write",
                }
            ):
                accepted_gdn2_update_upgrade = True
                matches = True
            if (
                not matches
                and field == "gdn2_state_expert_mode"
                and bool(expected_args.resume_allow_gdn2_state_expert_upgrade)
                and saved_value == "none"
                and current_value in {"dual_state", "raven_write_control"}
            ):
                accepted_gdn2_state_expert_upgrade = True
                matches = True
            if not matches:
                mismatches[field] = {
                    "saved": saved_value,
                    "current": current_value,
                }
        if mismatches:
            raise RuntimeError(
                f"Exact checkpoint semantic contract mismatch: {mismatches}"
            )
        saved_stages = parse_curriculum_spec(
            str(saved_args.get("hole_stages", ""))
        )
        current_stages = parse_curriculum_spec(
            str(expected_args.hole_stages)
        )
        saved_at_step = int(checkpoint.get("saved_at_step", -1))
        if saved_at_step < 0:
            raise RuntimeError("Exact checkpoint lacks a valid saved_at_step")
        consumed = 0
        stage_context = []
        for stage_index, saved_stage in enumerate(saved_stages):
            if consumed >= saved_at_step:
                break
            if stage_index >= len(current_stages):
                raise RuntimeError(
                    "Current curriculum ends before the resume step"
                )
            current_stage = current_stages[stage_index]
            saved_lo, saved_hi, saved_duration = saved_stage
            current_lo, current_hi, current_duration = current_stage
            if (saved_lo, saved_hi) != (current_lo, current_hi):
                raise RuntimeError(
                    "Resume curriculum range mismatch before checkpoint: "
                    f"saved={saved_stage}, current={current_stage}"
                )
            required_in_stage = min(
                saved_duration,
                saved_at_step - consumed,
            )
            if current_duration < required_in_stage:
                raise RuntimeError(
                    "Current curriculum is shorter than the consumed "
                    f"checkpoint prefix at stage {stage_index}: "
                    f"{current_duration} < {required_in_stage}"
                )
            if required_in_stage == saved_duration and (
                current_duration != saved_duration
            ):
                raise RuntimeError(
                    "A completed pre-checkpoint curriculum stage changed "
                    f"duration: saved={saved_stage}, current={current_stage}"
                )
            stage_context.append(
                {
                    "stage_index": stage_index,
                    "holes_min": saved_lo,
                    "holes_max": saved_hi,
                    "consumed_steps": required_in_stage,
                    "saved_duration": saved_duration,
                    "current_duration": current_duration,
                }
            )
            consumed += saved_duration
        if sum(row["consumed_steps"] for row in stage_context) != saved_at_step:
            raise RuntimeError(
                "Resume curriculum does not cover the checkpoint step"
            )
        resume_contract = {
            "fields": list(contract_fields),
            "checkpoint_sha256": checkpoint_sha256,
            "checkpoint_source_sha": checkpoint_source_sha,
            "checkpoint_source_clean": True,
            "checkpoint_source_config": str(source_config_path),
            "checkpoint_source_patch": str(source_patch_path),
            "accepted_legacy_defaults": accepted_legacy_defaults,
            "accepted_future_seed_content_upgrade": (
                accepted_future_seed_content_upgrade
            ),
            "accepted_future_seed_gradient_upgrade": (
                accepted_future_seed_gradient_upgrade
            ),
            "accepted_future_seed_update_upgrade": (
                accepted_future_seed_update_upgrade
            ),
            "accepted_gdn2_update_upgrade": accepted_gdn2_update_upgrade,
            "accepted_gdn2_state_expert_upgrade": (
                accepted_gdn2_state_expert_upgrade
            ),
            "curriculum_prefix": stage_context,
            "saved_at_step": saved_at_step,
            "matched": True,
        }
    missing, unexpected = model.load_state_dict(checkpoint["model"], strict=False)
    allowed_missing = {
        "loop_update_logit",
        "reasoner.future_seed_norm_slope",
        "reasoner.future_seed_norm_bias",
        "reasoner.future_seed_selector.gate_delta",
        "reasoner.future_seed_selector.content_weight",
        "reasoner.future_seed_readout_scale",
        "reasoner.future_seed_innovation_scale",
        "reasoner.future_seed_secant_raw",
        "reasoner.future_seed_producer_codec.row_score_in.weight",
        "reasoner.future_seed_producer_codec.row_score_in.bias",
        "reasoner.future_seed_producer_codec.row_score_out.weight",
        "reasoner.future_seed_producer_codec.cell_decode_in.weight",
        "reasoner.future_seed_producer_codec.cell_decode_in.bias",
        "reasoner.future_seed_producer_codec.cell_decode_out.weight",
        "reasoner.future_seed_address_local_update.row_gate_in.weight",
        "reasoner.future_seed_address_local_update.row_gate_in.bias",
        "reasoner.future_seed_address_local_update.row_gate_out.weight",
        "reasoner.future_seed_basis_transport.row_angles",
        "reasoner.future_seed_basis_transport.col_angles",
        "reasoner.shared_address_proj.weight",
    }
    progressive_suffixes = (
        ".time_mix.o_norm_weight_extra",
        ".time_mix.v_proj_extra.weight",
        ".time_mix.g_proj_extra.weight",
        ".time_mix.o_proj_extra.weight",
    )
    fast_slow_suffixes = (
        ".time_mix.fast_slow_decay.kernel_logits",
        ".time_mix.fast_slow_decay.rho_logit",
    )
    address_operator_suffixes = (
        ".time_mix.address_rotation_scale",
        ".time_mix.address_phase_proj.weight",
        ".time_mix.address_residual_proj.weight",
        ".time_mix.address_q_residual_proj.weight",
        ".time_mix.address_k_residual_proj.weight",
        ".time_mix.address_carrier_scale",
        ".time_mix.address_carrier_bias_delta",
    )
    gdn2_update_suffixes = (
        ".time_mix.log_spd_address_metric.raw",
        ".time_mix.coherent_delta_mix",
        ".time_mix.state_feedback_in.weight",
        ".time_mix.state_feedback_out.weight",
        ".time_mix.terminal_consolidation_k_proj.weight",
        ".time_mix.orthogonal_chunk_state_proj.weight",
        ".time_mix.orthogonal_head_write_proj.weight",
        ".time_mix.adaptive_signed_erase_proj.weight",
        ".time_mix.bi_axis_value_decay_proj.weight",
        ".time_mix.raven_route_proj.weight",
        ".time_mix.paired_address_q_proj.weight",
        ".time_mix.paired_address_k_proj.weight",
        ".time_mix.paired_address_read_gate",
        ".time_mix.coupled_address_k_proj.weight",
        ".time_mix.interleaved_write_k_proj.weight",
        ".time_mix.interleaved_write_v_proj.weight",
    )
    state_expert_marker = ".state_expert."
    allowed_unexpected = {"loop_update_logit"}
    bad_missing = [
        key
        for key in missing
        if key not in allowed_missing
        and not key.endswith(progressive_suffixes)
        and not key.endswith(fast_slow_suffixes)
        and not key.endswith(address_operator_suffixes)
        and not key.endswith(gdn2_update_suffixes)
        and state_expert_marker not in key
    ]
    bad_unexpected = [key for key in unexpected if key not in allowed_unexpected]
    if bad_missing or bad_unexpected:
        raise RuntimeError(
            "Checkpoint model state mismatch: "
            f"missing={bad_missing}, unexpected={bad_unexpected}"
        )
    progressive_missing = [key for key in missing if key.endswith(progressive_suffixes)]
    if progressive_missing:
        for block in model.reasoner.blocks:
            time_mix = getattr(block, "time_mix", None)
            if isinstance(time_mix, GDNTimeMix) and time_mix.progressive_expansion:
                time_mix.initialize_progressive_from_base()
    optimizer_state = checkpoint["optimizer"]
    inserted_parameters = set(missing)
    if inserted_parameters:
        current_state = opt.state_dict()
        saved_groups = optimizer_state.get("param_groups", [])
        current_groups = current_state.get("param_groups", [])
        name_by_parameter_id = {
            id(parameter): name for name, parameter in model.named_parameters()
        }
        current_group_names = [
            [name_by_parameter_id[id(parameter)] for parameter in group["params"]]
            for group in opt.param_groups
        ]
        current_param_count = sum(len(group.get("params", [])) for group in current_groups)
        can_expand = (
            len(saved_groups) == len(current_groups) == len(current_group_names)
            and sum(len(names) for names in current_group_names) == current_param_count
        )
        if not can_expand:
            raise ValueError("Cannot expand optimizer state for missing model parameters")
        expanded_optimizer_state = copy.deepcopy(optimizer_state)
        all_saved_params = [
            param_id
            for group in optimizer_state.get("param_groups", [])
            for param_id in group.get("params", [])
        ]
        synthetic_param_id = (max(all_saved_params) + 1) if all_saved_params else 0
        for saved, current, names in zip(
            expanded_optimizer_state["param_groups"],
            current_groups,
            current_group_names,
        ):
            old_params = list(saved.get("params", []))
            new_params = []
            old_cursor = 0
            current_params = list(current.get("params", []))
            if len(names) != len(current_params):
                raise ValueError(
                    "Optimizer parameter names do not match the current parameter group"
                )
            for name, _param_id in zip(names, current_params):
                if name in inserted_parameters:
                    new_params.append(synthetic_param_id)
                    synthetic_param_id += 1
                else:
                    if old_cursor >= len(old_params):
                        raise ValueError("Optimizer state is missing old parameters while inserting new parameters")
                    new_params.append(old_params[old_cursor])
                    old_cursor += 1
            if old_cursor != len(old_params):
                raise ValueError("Optimizer state has leftover old parameters after inserting new parameters")
            saved["params"] = new_params
        optimizer_state = expanded_optimizer_state
    opt.load_state_dict(optimizer_state)
    feature_buffer.load_state_dict(checkpoint.get("feature_buffer", {}))
    if "rng_python" in checkpoint:
        rng.setstate(checkpoint["rng_python"])
    restore_rng_state(checkpoint.get("rng_torch", {}), device)
    checkpoint["_load_migration"] = {
        "missing_parameters": sorted(missing),
        "unexpected_parameters": sorted(unexpected),
        "optimizer_groups_expanded": bool(inserted_parameters),
    }
    checkpoint["_resume_contract"] = resume_contract
    return checkpoint


class FutureSeedLoopSudoku(nn.Module):
    def __init__(
        self,
        *,
        d_model: int,
        layers: int,
        heads: int,
        head_dim: int,
        channel_mult: int,
        l_cycles: int,
        max_loops: int,
        lambda_: float,
        loop_update_mode: str,
        loop_update_gate_init: float,
        future_seed_scale: float,
        future_seed_decay: float,
        future_seed_update: str,
        future_seed_norm_mode: str,
        future_seed_gate_mode: str,
        loop_feedback_scale: float,
        loop_feedback_detach: bool,
        loop_feedback_corrupt_prob: float,
        loop_feedback_corrupt_mix: float,
        loop_feedback_corrupt_mode: str,
        loop_time_scale: float,
        scratch_mode: str,
        scratch_scale: float,
        scratch_noise_scale: float,
        scratch_gauss_projections: int,
        scratch_gate_bias: float,
        scratch_decay_bias: float,
        hidden_agg_noise_scale: float,
        hidden_agg_noise_temp: float,
        hidden_agg_noise_detach: bool,
        hidden_agg_noise_mode: str,
        hidden_agg_noise_topk: int,
        hidden_agg_noise_max_norm: float,
        activation_checkpoint: bool,
        rwkv_kernel: str,
        backbone: str,
        gdn_mode: str,
        gdn_expand_v: float,
        gdn_progressive_base_expand_v: float = 0.0,
        gdn_use_short_conv: bool,
        gdn_conv_size: int,
        gdn_allow_neg_eigval: bool,
        gdn2_gain_budget_mode: str = "none",
        gdn2_gain_budget_sigma_cap: float = 1.10,
        gdn2_gain_budget_step_cap: float = 1.0,
        gdn2_gain_budget_sigma_cap_max: float = 3.0,
        gdn2_gain_budget_infeasible_policy: str = "raise",
        gdn2_fast_slow_decay_mode: str = "none",
        gdn2_fast_slow_decay_kernel_size: int = 4,
        gdn2_fast_slow_decay_rho_init: float = 0.10,
        gdn2_fast_slow_decay_current_weight_init: float = 0.85,
        gdn2_precondition_mode: str = "none",
        gdn2_address_mode: str = "none",
        gdn2_update_mode: str = "none",
        gdn2_state_expert_mode: str = "none",
        gdn2_cross_layer_init: str = "independent",
        raven_num_slots: int = 0,
        raven_topk: int = 0,
        future_seed_scope: str = "layer",
        future_seed_readout_hop: int = 0,
        future_seed_content_mode: str = "terminal",
        momentum_future_seed_transport: str = "full_state",
    ) -> None:
        super().__init__()
        self.l_cycles = int(l_cycles)
        self.lambda_ = float(lambda_)
        self.max_loops = int(max_loops)
        if loop_update_mode not in {"fixed", "learned_gate"}:
            raise ValueError("loop_update_mode must be 'fixed' or 'learned_gate'.")
        self.loop_update_mode = loop_update_mode
        self.loop_feedback_scale = float(loop_feedback_scale)
        self.loop_feedback_detach = bool(loop_feedback_detach)
        self.loop_feedback_corrupt_prob = float(loop_feedback_corrupt_prob)
        self.loop_feedback_corrupt_mix = float(loop_feedback_corrupt_mix)
        if loop_feedback_corrupt_mode not in {"random_token", "uniform"}:
            raise ValueError("loop_feedback_corrupt_mode must be 'random_token' or 'uniform'.")
        self.loop_feedback_corrupt_mode = loop_feedback_corrupt_mode
        self.loop_time_scale = float(loop_time_scale)
        if scratch_mode not in {"none", "gated"}:
            raise ValueError("scratch_mode must be 'none' or 'gated'.")
        self.scratch_mode = scratch_mode
        self.scratch_scale = float(scratch_scale)
        self.scratch_noise_scale = float(scratch_noise_scale)
        self.hidden_agg_noise_scale = float(hidden_agg_noise_scale)
        self.hidden_agg_noise_temp = float(hidden_agg_noise_temp)
        self.hidden_agg_noise_detach = bool(hidden_agg_noise_detach)
        if hidden_agg_noise_mode not in {"gumbel", "soft_group"}:
            raise ValueError("hidden_agg_noise_mode must be 'gumbel' or 'soft_group'.")
        self.hidden_agg_noise_mode = hidden_agg_noise_mode
        self.hidden_agg_noise_topk = int(hidden_agg_noise_topk)
        self.hidden_agg_noise_max_norm = float(hidden_agg_noise_max_norm)
        self.gdn2_precondition_mode = gdn2_precondition_mode
        self.gdn2_address_mode = gdn2_address_mode
        self.gdn2_update_mode = gdn2_update_mode
        self.gdn2_state_expert_mode = gdn2_state_expert_mode
        self.gdn2_cross_layer_init = gdn2_cross_layer_init
        self.embed = nn.Embedding(VOCAB, d_model)
        self.position = nn.Embedding(CELLS, d_model)
        self.reasoner = FutureSeedRWKV(
            d_model,
            layers,
            heads,
            head_dim,
            channel_mult,
            future_seed_scale=future_seed_scale,
            future_seed_decay=future_seed_decay,
            future_seed_update=future_seed_update,
            future_seed_norm_mode=future_seed_norm_mode,
            future_seed_gate_mode=future_seed_gate_mode,
            future_seed_scope=future_seed_scope,
            future_seed_readout_hop=future_seed_readout_hop,
            future_seed_content_mode=future_seed_content_mode,
            momentum_future_seed_transport=momentum_future_seed_transport,
            activation_checkpoint=activation_checkpoint,
            rwkv_kernel=rwkv_kernel,
            backbone=backbone,
            gdn_mode=gdn_mode,
            gdn_expand_v=gdn_expand_v,
            gdn_progressive_base_expand_v=gdn_progressive_base_expand_v,
            gdn_use_short_conv=gdn_use_short_conv,
            gdn_conv_size=gdn_conv_size,
            gdn_allow_neg_eigval=gdn_allow_neg_eigval,
            gdn2_gain_budget_mode=gdn2_gain_budget_mode,
            gdn2_gain_budget_sigma_cap=gdn2_gain_budget_sigma_cap,
            gdn2_gain_budget_step_cap=gdn2_gain_budget_step_cap,
            gdn2_gain_budget_sigma_cap_max=gdn2_gain_budget_sigma_cap_max,
            gdn2_gain_budget_infeasible_policy=gdn2_gain_budget_infeasible_policy,
            gdn2_fast_slow_decay_mode=gdn2_fast_slow_decay_mode,
            gdn2_fast_slow_decay_kernel_size=gdn2_fast_slow_decay_kernel_size,
            gdn2_fast_slow_decay_rho_init=gdn2_fast_slow_decay_rho_init,
            gdn2_fast_slow_decay_current_weight_init=(
                gdn2_fast_slow_decay_current_weight_init
            ),
            gdn2_precondition_mode=gdn2_precondition_mode,
            gdn2_address_mode=gdn2_address_mode,
            gdn2_update_mode=gdn2_update_mode,
            gdn2_state_expert_mode=gdn2_state_expert_mode,
            gdn2_cross_layer_init=gdn2_cross_layer_init,
            raven_num_slots=raven_num_slots,
            raven_topk=raven_topk,
        )
        self.h_init = nn.Parameter(torch.zeros(1, 1, d_model))
        self.l_init = nn.Parameter(torch.zeros(1, 1, d_model))
        self.out_norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, N, bias=False)
        if self.loop_time_scale > 0:
            self.loop_time = nn.Linear(2, d_model, bias=False)
            nn.init.normal_(self.loop_time.weight, mean=0.0, std=0.02)
        else:
            self.loop_time = None
        if self.loop_feedback_scale > 0:
            self.loop_feedback = nn.Linear(N, d_model, bias=False)
            nn.init.zeros_(self.loop_feedback.weight)
        else:
            self.loop_feedback = None
        if self.scratch_mode == "gated":
            self.scratch_init = nn.Parameter(torch.zeros(1, 1, d_model))
            self.scratch_norm = nn.LayerNorm(d_model)
            self.scratch_update_norm = nn.LayerNorm(d_model)
            self.scratch_residual = nn.Linear(d_model, d_model, bias=False)
            self.scratch_gate = nn.Linear(d_model, d_model)
            self.scratch_decay = nn.Linear(d_model, d_model)
            nn.init.normal_(self.scratch_residual.weight, mean=0.0, std=0.02)
            nn.init.zeros_(self.scratch_gate.weight)
            nn.init.zeros_(self.scratch_decay.weight)
            nn.init.constant_(self.scratch_gate.bias, float(scratch_gate_bias))
            nn.init.constant_(self.scratch_decay.bias, float(scratch_decay_bias))
            if scratch_gauss_projections > 0:
                projection = torch.randn(int(scratch_gauss_projections), d_model)
                projection = projection / projection.norm(dim=-1, keepdim=True).clamp(min=1e-6)
            else:
                projection = torch.empty(0, d_model)
            self.register_buffer("scratch_projection", projection)
        else:
            self.register_parameter("scratch_init", None)
            self.scratch_norm = None
            self.scratch_update_norm = None
            self.scratch_residual = None
            self.scratch_gate = None
            self.scratch_decay = None
            self.register_buffer("scratch_projection", torch.empty(0, d_model))
        if self.loop_update_mode == "learned_gate":
            gate_init = min(max(float(loop_update_gate_init), 1e-4), 1.0 - 1e-4)
            gate_logit = math.log(gate_init / (1.0 - gate_init))
            self.loop_update_logit = nn.Parameter(torch.full((max(1, self.max_loops), 2), float(gate_logit)))
        else:
            self.register_parameter("loop_update_logit", None)

    def reset_shared_shell_parameters(self, seed: int) -> None:
        """Initialize backbone-independent random parameters from one fixed stream."""
        cuda_devices = sorted(
            {
                int(parameter.device.index)
                for parameter in self.parameters()
                if parameter.is_cuda and parameter.device.index is not None
            }
        )
        with torch.random.fork_rng(devices=cuda_devices):
            torch.manual_seed(int(seed))
            if cuda_devices:
                torch.cuda.manual_seed_all(int(seed))

            self.embed.reset_parameters()
            self.position.reset_parameters()
            for block in self.reasoner.blocks:
                channel_mix = block.channel_mix
                d_model = int(channel_mix.key.in_features)
                channel_mix.key.weight.data.uniform_(
                    -0.5 / (d_model**0.5),
                    0.5 / (d_model**0.5),
                )
                nn.init.zeros_(channel_mix.value.weight)
            self.head.reset_parameters()

            if self.loop_time is not None:
                nn.init.normal_(self.loop_time.weight, mean=0.0, std=0.02)
            if self.loop_feedback is not None:
                nn.init.zeros_(self.loop_feedback.weight)
            if self.scratch_residual is not None:
                nn.init.normal_(self.scratch_residual.weight, mean=0.0, std=0.02)
            if self.scratch_gate is not None:
                nn.init.zeros_(self.scratch_gate.weight)
            if self.scratch_decay is not None:
                nn.init.zeros_(self.scratch_decay.weight)
            if self.scratch_projection.numel() > 0:
                projection = torch.randn_like(self.scratch_projection)
                projection = projection / projection.norm(dim=-1, keepdim=True).clamp(min=1e-6)
                self.scratch_projection.copy_(projection)

    def input_sequence(
        self,
        inputs: torch.Tensor,
        *,
        cell_order: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        positions = torch.arange(CELLS, dtype=torch.long, device=inputs.device)
        if cell_order is None:
            ordered_inputs = inputs
            ordered_positions = positions
        else:
            ordered_inputs = inputs.index_select(1, cell_order)
            ordered_positions = cell_order
        return self.embed(ordered_inputs) + self.position(ordered_positions).unsqueeze(0)

    def canonical_address_sequence(
        self,
        *,
        batch_size: int,
        device: torch.device,
    ) -> torch.Tensor:
        positions = torch.arange(CELLS, dtype=torch.long, device=device)
        return self.position(positions).unsqueeze(0).expand(batch_size, -1, -1)

    def canonical_input_anchor_sequence(self, inputs: torch.Tensor) -> torch.Tensor:
        """Stable token-plus-position address anchor, independent of recurrent loops."""
        positions = torch.arange(CELLS, dtype=torch.long, device=inputs.device)
        return self.embed(inputs) + self.position(positions).unsqueeze(0)

    def scratch_gaussian_loss(self, residual: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        if self.scratch_projection.numel() == 0:
            zero = residual.new_zeros(())
            return zero, {
                "scratch_proj_mean_abs": zero,
                "scratch_proj_var_mean": zero,
                "scratch_proj_var_rank": zero,
            }
        z = F.layer_norm(residual.float(), (residual.shape[-1],))
        projection = self.scratch_projection.to(device=z.device, dtype=z.dtype)
        projected = z.reshape(-1, z.shape[-1]) @ projection.t()
        mean = projected.mean(dim=0)
        var = projected.var(dim=0, unbiased=False)
        loss = mean.square().mean() + (var - 1.0).square().mean()
        var_detached = var.detach().clamp(min=1e-8)
        var_rank = var_detached.sum().square() / var_detached.square().sum().clamp(min=1e-8)
        return loss.to(dtype=residual.dtype), {
            "scratch_proj_mean_abs": mean.detach().abs().mean().to(dtype=residual.dtype),
            "scratch_proj_var_mean": var.detach().mean().to(dtype=residual.dtype),
            "scratch_proj_var_rank": var_rank.to(dtype=residual.dtype),
        }

    def scratch_input(
        self,
        scratch: Optional[torch.Tensor],
    ) -> Tuple[Optional[torch.Tensor], torch.Tensor]:
        if scratch is None or self.scratch_mode == "none":
            zero = self.h_init.new_zeros(())
            return None, zero
        if self.training and self.scratch_noise_scale > 0:
            noise = torch.randn_like(scratch) * self.scratch_noise_scale
            return scratch + noise, noise.norm(dim=-1).mean()
        return scratch, scratch.new_zeros(())

    def update_scratch(
        self,
        scratch: Optional[torch.Tensor],
        hidden: torch.Tensor,
    ) -> Tuple[Optional[torch.Tensor], Dict[str, torch.Tensor]]:
        if self.scratch_mode == "none":
            zero = hidden.new_zeros(())
            return None, {
                "scratch_gate_mean": zero,
                "scratch_decay_mean": zero,
                "scratch_residual_norm": zero,
                "scratch_delta_norm": zero,
                "scratch_gauss_loss": zero,
                "scratch_proj_mean_abs": zero,
                "scratch_proj_var_mean": zero,
                "scratch_proj_var_rank": zero,
            }
        assert scratch is not None
        assert self.scratch_norm is not None
        assert self.scratch_update_norm is not None
        assert self.scratch_residual is not None
        assert self.scratch_gate is not None
        assert self.scratch_decay is not None
        h = self.scratch_norm(hidden)
        residual = torch.tanh(self.scratch_residual(h))
        gate = torch.sigmoid(self.scratch_gate(h))
        decay = torch.sigmoid(self.scratch_decay(h))
        updated = self.scratch_update_norm(decay * scratch + gate * residual)
        delta = updated - scratch
        gauss_loss, gauss_diag = self.scratch_gaussian_loss(residual)
        diag = {
            "scratch_gate_mean": gate.mean(),
            "scratch_decay_mean": decay.mean(),
            "scratch_residual_norm": residual.norm(dim=-1).mean(),
            "scratch_delta_norm": delta.norm(dim=-1).mean(),
            "scratch_gauss_loss": gauss_loss,
            **gauss_diag,
        }
        return updated, diag

    def feedback_probs_from_logits(self, logits: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        probs = logits.float().softmax(dim=-1)
        if self.loop_feedback_detach:
            probs = probs.detach()
        zero = probs.new_zeros(())
        diag: Dict[str, torch.Tensor] = {
            "loop_feedback_clean_confidence": probs.max(dim=-1).values.mean().detach(),
            "loop_feedback_entropy": (-(probs.clamp_min(1e-8).log() * probs).sum(dim=-1).mean()).detach(),
            "loop_feedback_corrupt_frac": zero,
            "loop_feedback_corrupt_mix": probs.new_tensor(float(self.loop_feedback_corrupt_mix)),
        }
        if self.training and self.loop_feedback_corrupt_prob > 0 and self.loop_feedback_corrupt_mix > 0:
            mask = torch.rand((*probs.shape[:-1], 1), device=probs.device) < float(self.loop_feedback_corrupt_prob)
            if self.loop_feedback_corrupt_mode == "uniform":
                corrupt_target = torch.full_like(probs, 1.0 / float(probs.shape[-1]))
            else:
                random_ids = torch.randint(0, probs.shape[-1], probs.shape[:-1], device=probs.device)
                corrupt_target = F.one_hot(random_ids, num_classes=probs.shape[-1]).to(dtype=probs.dtype)
            mix = min(max(float(self.loop_feedback_corrupt_mix), 0.0), 1.0)
            corrupted = probs * (1.0 - mix) + corrupt_target * mix
            probs = torch.where(mask, corrupted, probs)
            diag["loop_feedback_corrupt_frac"] = mask.float().mean().detach()
            diag["loop_feedback_corrupt_confidence"] = probs.max(dim=-1).values.mean().detach()
            diag["loop_feedback_corrupt_entropy"] = (
                -(probs.clamp_min(1e-8).log() * probs).sum(dim=-1).mean()
            ).detach()
        else:
            diag["loop_feedback_corrupt_confidence"] = diag["loop_feedback_clean_confidence"]
            diag["loop_feedback_corrupt_entropy"] = diag["loop_feedback_entropy"]
        return probs.to(dtype=logits.dtype), diag

    def hidden_aggregate_noise(self, hidden: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        zero = hidden.new_zeros(())
        if not (self.training and self.hidden_agg_noise_scale > 0):
            return hidden, {
                "hidden_agg_noise_norm": zero,
                "hidden_agg_noise_raw_norm": zero,
                "hidden_agg_noise_entropy": zero,
                "hidden_agg_noise_max_weight": zero,
                "hidden_agg_noise_clip_frac": zero,
            }
        temp = max(float(self.hidden_agg_noise_temp), 1e-4)
        score = hidden.detach().float().square().mean(dim=-1)
        source = hidden.detach() if self.hidden_agg_noise_detach else hidden
        if self.hidden_agg_noise_mode == "soft_group":
            centered = score - score.mean(dim=-1, keepdim=True)
            normalized = centered / score.std(dim=-1, keepdim=True, unbiased=False).clamp(min=1e-6)
            noisy_score = normalized + -torch.empty_like(score).exponential_().log()
            k = min(max(int(self.hidden_agg_noise_topk), 1), int(hidden.shape[1]))
            if k == int(hidden.shape[1]):
                weights = F.softmax(noisy_score / temp, dim=-1)
                pooled = torch.sum(weights.to(dtype=source.dtype).unsqueeze(-1) * source, dim=1, keepdim=True)
                entropy_denom = math.log(max(k, 2))
                max_weight = weights.max(dim=-1).values
                weight_entropy = -(weights.clamp_min(1e-8).log() * weights).sum(dim=-1)
            else:
                top_values, top_idx = torch.topk(noisy_score, k=k, dim=-1)
                weights = F.softmax(top_values / temp, dim=-1)
                gather_idx = top_idx.unsqueeze(-1).expand(-1, -1, source.shape[-1])
                selected = torch.gather(source, dim=1, index=gather_idx)
                pooled = torch.sum(weights.to(dtype=source.dtype).unsqueeze(-1) * selected, dim=1, keepdim=True)
                entropy_denom = math.log(max(k, 2))
                max_weight = weights.max(dim=-1).values
                weight_entropy = -(weights.clamp_min(1e-8).log() * weights).sum(dim=-1)
        else:
            gumbel = -torch.empty_like(score).exponential_().log()
            weights = F.softmax((score + gumbel) / temp, dim=-1)
            pooled = torch.sum(weights.to(dtype=source.dtype).unsqueeze(-1) * source, dim=1, keepdim=True)
            entropy_denom = math.log(max(int(hidden.shape[1]), 2))
            max_weight = weights.max(dim=-1).values
            weight_entropy = -(weights.clamp_min(1e-8).log() * weights).sum(dim=-1)
        direction = pooled - source.mean(dim=1, keepdim=True)
        direction_rms = direction.float().square().mean(dim=-1, keepdim=True).sqrt().clamp(min=1e-6)
        ref_rms = hidden.detach().float().square().mean(dim=-1, keepdim=True).sqrt().mean(dim=1, keepdim=True)
        perturb = direction.float() / direction_rms * ref_rms * float(self.hidden_agg_noise_scale)
        raw_norm = perturb.detach().float().norm(dim=-1, keepdim=True)
        clip_frac = zero
        if self.hidden_agg_noise_max_norm > 0:
            cap = perturb.new_tensor(float(self.hidden_agg_noise_max_norm))
            scale = (cap / raw_norm.clamp(min=1e-6)).clamp(max=1.0)
            clip_frac = (raw_norm > cap).float().mean().to(dtype=hidden.dtype)
            perturb = perturb * scale
        perturb = perturb.to(dtype=hidden.dtype).expand_as(hidden)
        diag = {
            "hidden_agg_noise_norm": perturb.detach().float().norm(dim=-1).mean().to(dtype=hidden.dtype),
            "hidden_agg_noise_raw_norm": raw_norm.mean().to(dtype=hidden.dtype),
            "hidden_agg_noise_entropy": (
                weight_entropy.mean() / entropy_denom
            ).to(dtype=hidden.dtype),
            "hidden_agg_noise_max_weight": max_weight.mean().to(dtype=hidden.dtype),
            "hidden_agg_noise_clip_frac": clip_frac,
        }
        return hidden + perturb, diag

    def depth_update(
        self,
        hidden: torch.Tensor,
        injection: torch.Tensor,
        noise_scale: float,
        *,
        feature_buffer: Optional[FeatureNoiseBuffer],
        update_feature_buffer: bool,
        feature_buffer_add: int,
        allow_noise: bool,
        seed_memory: Optional[List[Optional[torch.Tensor]]] = None,
        loop_idx: int = 0,
        stream_idx: int = 0,
        address: Optional[torch.Tensor] = None,
        cell_order: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], Optional[List[torch.Tensor]]]:
        updated, diag, next_seed_memory = self.reasoner(
            hidden + injection,
            seed_memory=seed_memory,
            address=address,
            cell_order=cell_order,
        )
        if self.loop_update_logit is None:
            update_gate = hidden.new_tensor(self.lambda_)
        else:
            loop_i = min(max(int(loop_idx), 0), self.loop_update_logit.shape[0] - 1)
            stream_i = min(max(int(stream_idx), 0), self.loop_update_logit.shape[1] - 1)
            update_gate = torch.sigmoid(self.loop_update_logit[loop_i, stream_i]).to(
                device=hidden.device,
                dtype=hidden.dtype,
            )
        out = hidden + update_gate * (updated - hidden)
        if allow_noise and noise_scale > 0:
            if feature_buffer is None:
                raise ValueError("feature-difference noise requires a FeatureNoiseBuffer")
            out = out + feature_buffer.sample_diff_like(out) * float(noise_scale)
        if allow_noise:
            out, hidden_agg_diag = self.hidden_aggregate_noise(out)
        else:
            zero = out.new_zeros(())
            hidden_agg_diag = {
                "hidden_agg_noise_norm": zero,
                "hidden_agg_noise_raw_norm": zero,
                "hidden_agg_noise_entropy": zero,
                "hidden_agg_noise_max_weight": zero,
                "hidden_agg_noise_clip_frac": zero,
            }
        if allow_noise and update_feature_buffer and feature_buffer is not None:
            feature_buffer.add(out, max_items=feature_buffer_add)
        diag = dict(diag)
        diag.update(hidden_agg_diag)
        diag["loop_update_gate"] = update_gate.detach()
        return out, diag, next_seed_memory

    def forward_trace(
        self,
        inputs: torch.Tensor,
        *,
        loops: int,
        noise_scale: float,
        feature_buffer: Optional[FeatureNoiseBuffer] = None,
        update_feature_buffer: bool = False,
        feature_buffer_add: int = 2048,
        cell_order: Optional[torch.Tensor] = None,
    ) -> Tuple[List[torch.Tensor], List[Dict[str, torch.Tensor]]]:
        cell_order = normalize_cell_order(
            cell_order,
            cells=CELLS,
            device=inputs.device,
        )
        x = self.input_sequence(inputs, cell_order=cell_order)
        batch_size, seq_len, _channels = x.shape
        if self.gdn2_address_mode == "position_qk":
            address = self.canonical_address_sequence(
                batch_size=batch_size,
                device=inputs.device,
            )
        elif self.gdn2_address_mode in {
            "anchor_rotary",
            "anchor_phase",
            "anchor_residual",
            "shared_namespace",
            "anchor_qk_residual",
            "anchor_carrier",
        }:
            address = self.canonical_input_anchor_sequence(inputs)
        else:
            address = None
        z_h = self.h_init.expand(batch_size, seq_len, -1)
        z_l = self.l_init.expand(batch_size, seq_len, -1)
        feedback: Optional[torch.Tensor] = None
        scratch: Optional[torch.Tensor] = None
        if self.scratch_mode == "gated":
            assert self.scratch_init is not None
            scratch = self.scratch_init.expand(batch_size, seq_len, -1)
        h_seed_memory: Optional[List[torch.Tensor]] = None
        l_seed_memory: Optional[List[torch.Tensor]] = None
        zero = x.new_zeros(())

        loop_logits: List[torch.Tensor] = []
        fs_trace: List[Dict[str, torch.Tensor]] = []
        loop_count = int(loops)
        for loop_idx in range(loop_count):
            feedback_in_norm = zero if feedback is None else feedback.norm(dim=-1).mean()
            loop_context = x if feedback is None else x + feedback
            scratch_context, scratch_noise_norm = self.scratch_input(scratch)
            if scratch_context is not None and self.scratch_scale > 0:
                loop_context = loop_context + scratch_context * self.scratch_scale
            loop_time_norm = zero
            if self.loop_time is not None and self.loop_time_scale > 0:
                phase = float(loop_idx + 1) / float(max(loop_count, 1))
                features = x.new_tensor([[math.sin(math.pi * phase), math.cos(math.pi * phase)]])
                loop_time = self.loop_time(features).to(dtype=x.dtype).view(1, 1, -1) * self.loop_time_scale
                loop_context = loop_context + loop_time
                loop_time_norm = loop_time.norm(dim=-1).mean()
            l_update_gates: List[torch.Tensor] = []
            for _ in range(self.l_cycles):
                z_l, l_diag, l_seed_memory = self.depth_update(
                    z_l,
                    z_h + loop_context,
                    noise_scale,
                    feature_buffer=feature_buffer,
                    update_feature_buffer=update_feature_buffer,
                    feature_buffer_add=feature_buffer_add,
                    allow_noise=True,
                    seed_memory=l_seed_memory,
                    loop_idx=loop_idx,
                    stream_idx=0,
                    address=address,
                    cell_order=cell_order,
                )
                if "loop_update_gate" in l_diag:
                    l_update_gates.append(l_diag["loop_update_gate"])
            z_h, fs_diag, h_seed_memory = self.depth_update(
                z_h,
                z_l,
                noise_scale,
                feature_buffer=feature_buffer,
                update_feature_buffer=update_feature_buffer,
                feature_buffer_add=feature_buffer_add,
                allow_noise=True,
                seed_memory=h_seed_memory,
                loop_idx=loop_idx,
                stream_idx=1,
                address=address,
                cell_order=cell_order,
            )
            board_h = self.out_norm(z_h[:, :CELLS])
            sequence_logits = self.head(board_h)
            logits = restore_canonical_cell_order(sequence_logits, cell_order)
            loop_logits.append(logits)
            fs_diag = dict(fs_diag)
            fs_diag["loop_update_gate_h"] = fs_diag.pop("loop_update_gate", zero)
            if l_update_gates:
                fs_diag["loop_update_gate_l"] = torch.stack(l_update_gates).mean()
            else:
                fs_diag["loop_update_gate_l"] = zero
            fs_diag["loop_feedback_in_norm"] = feedback_in_norm
            fs_diag["loop_time_norm"] = loop_time_norm
            fs_diag["scratch_noise_norm"] = scratch_noise_norm
            scratch, scratch_diag = self.update_scratch(scratch, z_h)
            fs_diag.update(scratch_diag)
            if self.loop_feedback is not None and self.loop_feedback_scale > 0:
                feedback_probs, feedback_diag = self.feedback_probs_from_logits(
                    sequence_logits
                )
                feedback = self.loop_feedback(feedback_probs) * self.loop_feedback_scale
                fs_diag.update(feedback_diag)
                fs_diag["loop_feedback_next_norm"] = feedback.norm(dim=-1).mean()
            else:
                feedback = None
                fs_diag["loop_feedback_next_norm"] = zero
                fs_diag["loop_feedback_clean_confidence"] = zero
                fs_diag["loop_feedback_entropy"] = zero
                fs_diag["loop_feedback_corrupt_frac"] = zero
                fs_diag["loop_feedback_corrupt_mix"] = zero
                fs_diag["loop_feedback_corrupt_confidence"] = zero
                fs_diag["loop_feedback_corrupt_entropy"] = zero
            fs_trace.append(fs_diag)
        return loop_logits, fs_trace


def valid_board(row: torch.Tensor) -> bool:
    board = [int(x) for x in row.detach().cpu().tolist()]
    target = set(range(N))
    if any(x < 0 or x >= N for x in board):
        return False
    return all(set(board[i] for i in unit) == target for unit in UNITS)


@torch.no_grad()
def metrics_from_logits(logits: torch.Tensor, labels: torch.Tensor, clue_mask: torch.Tensor) -> Tuple[FullMetrics, torch.Tensor]:
    pred = logits.argmax(dim=-1)
    return metrics_from_predictions(pred, labels, clue_mask), pred


@torch.no_grad()
def metrics_from_predictions(pred: torch.Tensor, labels: torch.Tensor, clue_mask: torch.Tensor) -> FullMetrics:
    exact = (pred == labels).all(dim=1)
    clue_ok = ((pred == labels) | ~clue_mask).all(dim=1)
    valid = torch.tensor([valid_board(pred[i]) for i in range(pred.shape[0])], device=pred.device)
    blanks = ~clue_mask
    blank_correct = (pred == labels) & blanks
    blank_acc = blank_correct.sum().float() / blanks.sum().clamp_min(1)
    positions = torch.arange(CELLS, device=pred.device)
    first_cut = CELLS // 3
    second_cut = (2 * CELLS) // 3

    def bucket_stats(mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        bucket = blanks & mask.unsqueeze(0)
        denom = bucket.sum().clamp_min(1)
        acc = (blank_correct & bucket).sum().float() / denom
        frac = bucket.sum().float() / blanks.sum().clamp_min(1)
        return acc, frac

    early_acc, early_frac = bucket_stats(positions < first_cut)
    mid_acc, mid_frac = bucket_stats((positions >= first_cut) & (positions < second_cut))
    late_acc, late_frac = bucket_stats(positions >= second_cut)
    return FullMetrics(
        label_exact=exact.float().mean().item(),
        valid_sudoku=valid.float().mean().item(),
        solved_valid_clue=(valid & clue_ok).float().mean().item(),
        clue_ok=clue_ok.float().mean().item(),
        blank_acc=blank_acc.item(),
        blank_acc_early=early_acc.item(),
        blank_acc_mid=mid_acc.item(),
        blank_acc_late=late_acc.item(),
        blank_acc_early_late_gap=(early_acc - late_acc).item(),
        blank_frac_early=early_frac.item(),
        blank_frac_mid=mid_frac.item(),
        blank_frac_late=late_frac.item(),
        avg_filled=pred.ne(BLANK).float().mean().item(),
    )


def loss_from_logits(
    logits: torch.Tensor,
    labels: torch.Tensor,
    clue_mask: torch.Tensor,
    *,
    blank_weight: float,
) -> torch.Tensor:
    loss = F.cross_entropy(logits.float().reshape(-1, N), labels.reshape(-1), reduction="none").view_as(labels)
    if blank_weight == 1.0:
        return loss.mean()
    weights = torch.where(clue_mask, torch.ones_like(loss), torch.full_like(loss, float(blank_weight)))
    return (loss * weights).sum() / weights.sum().clamp_min(1.0)


def exact_margin_loss_from_logits(
    logits: torch.Tensor,
    labels: torch.Tensor,
    clue_mask: torch.Tensor,
    *,
    tau: float,
    target: float,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    logits_f = logits.float()
    blank_mask = ~clue_mask
    blank_counts = blank_mask.sum(dim=-1)
    valid = blank_counts > 0
    zero = logits_f.new_zeros(())
    if not bool(valid.any()):
        return zero, {
            "exact_margin_loss": zero,
            "exact_margin_softmin": zero,
            "exact_margin_hardmin": zero,
            "exact_margin_mean": zero,
        }

    safe_labels = labels.clamp(min=0, max=logits_f.shape[-1] - 1)
    true_logits = logits_f.gather(-1, safe_labels.unsqueeze(-1)).squeeze(-1)
    true_class = F.one_hot(safe_labels, num_classes=logits_f.shape[-1]).to(dtype=torch.bool)
    other_logits = logits_f.masked_fill(true_class, -torch.inf).amax(dim=-1)
    margins = true_logits - other_logits

    masked_margins = margins.masked_fill(~blank_mask, torch.inf)
    tau_value = max(float(tau), 1e-4)
    valid_margins = masked_margins[valid]
    valid_counts = blank_counts[valid].to(dtype=logits_f.dtype)
    # Soft minimum over blank cells per sample. The log(count) correction makes the
    # diagnostic comparable when official samples have different blank counts.
    softmin = -tau_value * torch.logsumexp(-valid_margins / tau_value, dim=-1)
    softmin = softmin + tau_value * valid_counts.clamp_min(1).log()
    hardmin = valid_margins.amin(dim=-1)
    loss = F.softplus(logits_f.new_tensor(float(target)) - softmin).mean()
    selected = margins[blank_mask]
    return loss, {
        "exact_margin_loss": loss.detach(),
        "exact_margin_softmin": softmin.detach().mean(),
        "exact_margin_hardmin": hardmin.detach().mean(),
        "exact_margin_mean": selected.detach().mean() if selected.numel() else zero,
    }


def loop_weight_tensor(
    num_loops: int,
    *,
    mode: str,
    start: int,
    power: float,
    min_weight: float,
    device: torch.device,
) -> torch.Tensor:
    if num_loops <= 0:
        raise ValueError("num_loops must be positive")
    weights = torch.zeros(num_loops, dtype=torch.float32, device=device)
    if mode == "final":
        weights[-1] = 1.0
        return weights
    if mode == "all":
        weights.fill_(1.0 / num_loops)
        return weights

    first = min(max(int(start), 1), num_loops) - 1
    active = num_loops - first
    ramp = torch.linspace(1.0 / active, 1.0, active, dtype=torch.float32, device=device).pow(float(power))
    if mode == "shaped":
        early = torch.full((first,), float(min_weight), dtype=torch.float32, device=device)
        weights = torch.cat((early, torch.clamp(ramp, min=float(min_weight))))
    elif mode == "delayed":
        weights[first:] = ramp
    else:
        raise ValueError(f"Unknown loop loss mode {mode!r}")
    return weights / weights.sum().clamp_min(1e-8)


def weighted_loop_loss(losses: List[torch.Tensor], args: argparse.Namespace) -> Tuple[torch.Tensor, torch.Tensor]:
    weights = loop_weight_tensor(
        len(losses),
        mode=args.loop_loss,
        start=args.loop_loss_start,
        power=args.loop_loss_power,
        min_weight=args.loop_loss_min_weight,
        device=losses[-1].device,
    ).to(dtype=losses[-1].dtype)
    stacked = torch.stack(losses)
    return (stacked * weights).sum(), weights


def project_future_seed_opening_gradient(
    opening_gradient: torch.Tensor,
    baseline_gradient: torch.Tensor,
    *,
    num_loops: int = 5,
    eps: float = 1e-30,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Protect the continuation direction without changing the FS gradient norm."""
    if opening_gradient.ndim != 1 or baseline_gradient.ndim != 1:
        raise ValueError("FutureSeed gradient projection expects flat vectors")
    if opening_gradient.shape != baseline_gradient.shape:
        raise ValueError(
            "FutureSeed opening/baseline gradient shape mismatch: "
            f"{tuple(opening_gradient.shape)} != {tuple(baseline_gradient.shape)}"
        )
    if num_loops != 5:
        raise ValueError("FutureSeed opening projection is registered for five loops")

    opening = opening_gradient.detach().to(dtype=torch.float64)
    baseline = baseline_gradient.detach().to(dtype=torch.float64)
    if not bool(torch.isfinite(opening).all()) or not bool(torch.isfinite(baseline).all()):
        raise RuntimeError("FutureSeed gradient projection received non-finite input")

    continuation = (float(num_loops) * baseline - opening) / float(num_loops - 1)
    opening_norm = torch.linalg.vector_norm(opening)
    baseline_norm = torch.linalg.vector_norm(baseline)
    continuation_norm = torch.linalg.vector_norm(continuation)
    continuation_norm_sq = torch.dot(continuation, continuation)
    if (
        float(opening_norm.item()) <= eps
        or float(baseline_norm.item()) <= eps
        or float(continuation_norm_sq.item()) <= eps
    ):
        raise RuntimeError(
            "FutureSeed gradient projection encountered a near-zero active gradient"
        )

    opening_continuation_dot = torch.dot(opening, continuation)
    cosine = opening_continuation_dot / (opening_norm * continuation_norm)
    active = bool(opening_continuation_dot.item() < 0.0)
    projected_opening = opening
    corrected = baseline
    projection_coefficient = 0.0
    removed_fraction = 0.0
    relative_correction = 0.0
    if active:
        coefficient = -opening_continuation_dot / continuation_norm_sq
        projected_opening = opening + coefficient * continuation
        raw = (
            projected_opening
            + float(num_loops - 1) * continuation
        ) / float(num_loops)
        raw_norm = torch.linalg.vector_norm(raw)
        if float(raw_norm.item()) <= eps or not bool(torch.isfinite(raw_norm)):
            raise RuntimeError("FutureSeed projected gradient has invalid norm")
        corrected = raw * (baseline_norm / raw_norm)
        projection_coefficient = float(coefficient.item())
        removed_fraction = float(
            (torch.linalg.vector_norm(projected_opening - opening) / opening_norm).item()
        )
        relative_correction = float(
            (torch.linalg.vector_norm(corrected - baseline) / baseline_norm).item()
        )

    corrected_norm = torch.linalg.vector_norm(corrected)
    norm_relative_error = float(
        (torch.abs(corrected_norm - baseline_norm) / baseline_norm).item()
    )
    post_dot = float(torch.dot(projected_opening, continuation).item())
    if not bool(torch.isfinite(corrected).all()):
        raise RuntimeError("FutureSeed gradient projection produced non-finite output")
    return corrected.to(dtype=baseline_gradient.dtype), {
        "active": float(active),
        "opening_continuation_cosine": float(cosine.item()),
        "opening_continuation_dot": float(opening_continuation_dot.item()),
        "post_opening_continuation_dot": post_dot,
        "projection_coefficient": projection_coefficient,
        "removed_opening_fraction": removed_fraction,
        "relative_correction": relative_correction,
        "norm_relative_error": norm_relative_error,
        "opening_norm": float(opening_norm.item()),
        "continuation_norm": float(continuation_norm.item()),
        "baseline_norm": float(baseline_norm.item()),
    }


def future_seed_gate_parameters(
    model: nn.Module,
) -> List[Tuple[str, nn.Parameter]]:
    return [
        (name, parameter)
        for name, parameter in model.named_parameters()
        if name.endswith(".future_seed_logit")
    ]


def apply_future_seed_opening_projection(
    gate_parameters: List[Tuple[str, nn.Parameter]],
    opening_gradients: List[Optional[torch.Tensor]],
) -> Dict[str, float]:
    if len(gate_parameters) != len(opening_gradients):
        raise ValueError("FutureSeed gate/opening-gradient count mismatch")
    active_rows: List[Tuple[nn.Parameter, torch.Tensor, torch.Tensor]] = []
    inactive_tensors = 0
    for (_name, parameter), opening in zip(gate_parameters, opening_gradients):
        baseline = parameter.grad
        if opening is None and baseline is None:
            inactive_tensors += 1
            continue
        if opening is None or baseline is None:
            raise RuntimeError(
                "FutureSeed opening and canonical gradients disagree on active tensors"
            )
        if opening.shape != parameter.shape or baseline.shape != parameter.shape:
            raise RuntimeError("FutureSeed gate gradient shape changed")
        active_rows.append((parameter, opening, baseline))
    if not active_rows:
        raise RuntimeError("FutureSeed opening projection found no active receiving edges")

    opening_vector = torch.cat(
        [opening.detach().float().reshape(-1) for _p, opening, _b in active_rows]
    )
    baseline_vector = torch.cat(
        [baseline.detach().float().reshape(-1) for _p, _o, baseline in active_rows]
    )
    corrected, diagnostics = project_future_seed_opening_gradient(
        opening_vector,
        baseline_vector,
    )
    if diagnostics["active"]:
        cursor = 0
        for parameter, _opening, baseline in active_rows:
            count = parameter.numel()
            baseline.copy_(
                corrected[cursor : cursor + count]
                .reshape_as(parameter)
                .to(device=baseline.device, dtype=baseline.dtype)
            )
            cursor += count
        if cursor != corrected.numel():
            raise RuntimeError("FutureSeed projected-gradient slicing mismatch")
        actual = torch.cat(
            [
                baseline.detach().float().reshape(-1)
                for _parameter, _opening, baseline in active_rows
            ]
        ).to(dtype=torch.float64)
        baseline_norm = torch.linalg.vector_norm(
            baseline_vector.to(dtype=torch.float64)
        ).clamp_min(1e-30)
        diagnostics["norm_relative_error"] = float(
            (
                torch.abs(torch.linalg.vector_norm(actual) - baseline_norm)
                / baseline_norm
            ).item()
        )
    diagnostics.update(
        {
            "active_tensor_count": float(len(active_rows)),
            "active_parameter_count": float(opening_vector.numel()),
            "inactive_tensor_count": float(inactive_tensors),
        }
    )
    return diagnostics


def new_future_seed_projection_stats() -> Dict[str, float]:
    return {
        "steps": 0.0,
        "active_steps": 0.0,
        "cosine_sum": 0.0,
        "cosine_min": 1.0,
        "removed_fraction_sum": 0.0,
        "removed_fraction_min": 1.0,
        "relative_correction_sum": 0.0,
        "relative_correction_min": 1.0,
        "norm_relative_error_max": 0.0,
        "post_dot_min": 0.0,
        "active_tensor_count": 0.0,
        "active_parameter_count": 0.0,
        "inactive_tensor_count": 0.0,
        "canonical_global_grad_norm_sum": 0.0,
        "canonical_clip_steps": 0.0,
        "canonical_clip_coefficient_min": 1.0,
        "final_global_grad_norm_max": 0.0,
        "final_global_grad_norm_relative_error_max": 0.0,
    }


def update_future_seed_projection_stats(
    state: Dict[str, float],
    diagnostics: Dict[str, float],
) -> None:
    previous_steps = state["steps"]
    previous_active_steps = state["active_steps"]
    state["steps"] += 1.0
    state["active_steps"] += diagnostics["active"]
    state["cosine_sum"] += diagnostics["opening_continuation_cosine"]
    state["cosine_min"] = (
        diagnostics["opening_continuation_cosine"]
        if previous_steps == 0.0
        else min(
            state["cosine_min"], diagnostics["opening_continuation_cosine"]
        )
    )
    if diagnostics["active"]:
        state["removed_fraction_sum"] += diagnostics["removed_opening_fraction"]
        state["removed_fraction_min"] = (
            diagnostics["removed_opening_fraction"]
            if previous_active_steps == 0.0
            else min(
                state["removed_fraction_min"],
                diagnostics["removed_opening_fraction"],
            )
        )
        state["relative_correction_sum"] += diagnostics["relative_correction"]
        state["relative_correction_min"] = (
            diagnostics["relative_correction"]
            if previous_active_steps == 0.0
            else min(
                state["relative_correction_min"],
                diagnostics["relative_correction"],
            )
        )
    state["norm_relative_error_max"] = max(
        state["norm_relative_error_max"], diagnostics["norm_relative_error"]
    )
    state["post_dot_min"] = (
        diagnostics["post_opening_continuation_dot"]
        if previous_steps == 0.0
        else min(
            state["post_dot_min"],
            diagnostics["post_opening_continuation_dot"],
        )
    )
    for key in (
        "active_tensor_count",
        "active_parameter_count",
        "inactive_tensor_count",
    ):
        state[key] = diagnostics[key]
    state["canonical_global_grad_norm_sum"] += diagnostics[
        "canonical_global_grad_norm"
    ]
    state["canonical_clip_steps"] += float(
        diagnostics["canonical_clip_coefficient"] < 1.0
    )
    state["canonical_clip_coefficient_min"] = min(
        state["canonical_clip_coefficient_min"],
        diagnostics["canonical_clip_coefficient"],
    )
    state["final_global_grad_norm_max"] = max(
        state["final_global_grad_norm_max"],
        diagnostics["final_global_grad_norm"],
    )
    state["final_global_grad_norm_relative_error_max"] = max(
        state["final_global_grad_norm_relative_error_max"],
        diagnostics["final_global_grad_norm_relative_error"],
    )


def summarize_future_seed_projection_stats(
    state: Dict[str, float],
) -> Dict[str, float]:
    steps = state["steps"]
    active_steps = state["active_steps"]
    return {
        "steps": steps,
        "active_steps": active_steps,
        "activation_rate": active_steps / steps if steps else 0.0,
        "opening_continuation_cosine_mean": (
            state["cosine_sum"] / steps if steps else 0.0
        ),
        "opening_continuation_cosine_min": (
            state["cosine_min"] if steps else 0.0
        ),
        "active_removed_opening_fraction_mean": (
            state["removed_fraction_sum"] / active_steps if active_steps else 0.0
        ),
        "active_removed_opening_fraction_min": (
            state["removed_fraction_min"] if active_steps else 0.0
        ),
        "active_relative_correction_mean": (
            state["relative_correction_sum"] / active_steps if active_steps else 0.0
        ),
        "active_relative_correction_min": (
            state["relative_correction_min"] if active_steps else 0.0
        ),
        "norm_relative_error_max": state["norm_relative_error_max"],
        "post_opening_continuation_dot_min": (
            state["post_dot_min"] if steps else 0.0
        ),
        "active_tensor_count": state["active_tensor_count"],
        "active_parameter_count": state["active_parameter_count"],
        "inactive_tensor_count": state["inactive_tensor_count"],
        "canonical_global_grad_norm_mean": (
            state["canonical_global_grad_norm_sum"] / steps if steps else 0.0
        ),
        "canonical_clip_rate": (
            state["canonical_clip_steps"] / steps if steps else 0.0
        ),
        "canonical_clip_coefficient_min": (
            state["canonical_clip_coefficient_min"] if steps else 1.0
        ),
        "final_global_grad_norm_max": state["final_global_grad_norm_max"],
        "final_global_grad_norm_relative_error_max": state[
            "final_global_grad_norm_relative_error_max"
        ],
    }


def current_global_gradient_norm(parameters: Iterable[nn.Parameter]) -> torch.Tensor:
    gradient_norms = [
        torch.linalg.vector_norm(parameter.grad.detach().float())
        for parameter in parameters
        if parameter.grad is not None
    ]
    if not gradient_norms:
        raise RuntimeError("Cannot measure an empty global gradient")
    return torch.linalg.vector_norm(torch.stack(gradient_norms))


@torch.no_grad()
def fs_metrics_from_trace(trace: Dict[str, torch.Tensor]) -> Dict[str, float]:
    return {key: float(value.detach().cpu()) for key, value in trace.items()}


def metric_line(m: Dict[str, float]) -> str:
    return (
        f"exact={m['label_exact']:.4f}, valid={m['valid_sudoku']:.4f}, "
        f"solved={m['solved_valid_clue']:.4f}, blank_acc={m['blank_acc']:.4f}, "
        f"early={m.get('blank_acc_early', 0.0):.4f}, "
        f"late={m.get('blank_acc_late', 0.0):.4f}, "
        f"early_late_gap={m.get('blank_acc_early_late_gap', 0.0):+.4f}"
    )


def coupling_line(m: Dict[str, float], holes: int) -> str:
    independent = float(m["blank_acc"]) ** int(holes)
    ratio = float(m["label_exact"]) / independent if independent > 0 else float("nan")
    return f"indep={independent:.4f}, exact/indep={ratio:.2f}"


def fs_line(m: Dict[str, float]) -> str:
    parts = [
        f"fs_gate={m['fs_gate_mean']:.3f}",
        f"fs_update={m.get('fs_update_mean', 0.0):.3f}",
        f"fs_state_norm={m['fs_state_norm']:.3f}",
    ]
    if "fs_decay" in m:
        parts.append(f"fs_decay={m['fs_decay']:.2f}")
    if "fs_memory_norm" in m:
        parts.append(f"fs_mem={m['fs_memory_norm']:.3f}")
    if "fs_memory_delta_norm" in m:
        parts.append(f"fs_mem_delta={m['fs_memory_delta_norm']:.3f}")
    if m.get("fs2_loop_secant_enabled", 0.0) > 0:
        parts.append(
            "fs2_secant="
            f"{m.get('fs2_loop_secant_scale_signed', 0.0):+.4f}/"
            f"{m.get('fs2_loop_secant_residual_relative_rms', 0.0):.4f}"
        )
    if "fs_raw_rms_mean" in m:
        parts.append(f"fs_raw_rms={m['fs_raw_rms_mean']:.3f}")
        parts.append(f"fs_raw_rms_std={m.get('fs_raw_rms_std', 0.0):.3f}")
    if "fs_norm_gain_mean" in m:
        parts.append(f"fs_norm_gain={m['fs_norm_gain_mean']:.3f}")
        parts.append(f"fs_norm_gain_std={m.get('fs_norm_gain_std', 0.0):.3f}")
    if "fs2_gate_delta_rms" in m:
        parts.append(f"fs2_delta={m['fs2_gate_delta_rms']:.4f}")
        parts.append(f"fs2_gate_std={m.get('fs2_gate_std', 0.0):.4f}")
        parts.append(
            f"fs2_batch_std={m.get('fs2_gate_batch_std', 0.0):.4f}"
        )
        parts.append(
            f"fs2_feature_std={m.get('fs2_content_feature_std', 0.0):.4f}"
        )
        parts.append(
            "fs2_gate_range="
            f"{m.get('fs2_gate_min', 0.0):.3f}:{m.get('fs2_gate_max', 0.0):.3f}"
        )
        parts.append(
            f"fs2_seed_change={m.get('fs2_seed_relative_change', 0.0):.4f}"
        )
    if "fs2_block_seed_active" in m:
        parts.append(
            f"fs2_block_active={m.get('fs2_block_seed_active', 0.0):.1f}"
        )
        parts.append(
            f"fs2_block_raw_norm={m.get('fs2_block_seed_raw_norm', 0.0):.3f}"
        )
    if m.get("fs2_readout_enabled", 0.0) > 0:
        parts.append(
            f"fs2_readout_scale={m.get('fs2_readout_scale_abs', 0.0):.4f}"
        )
        parts.append(
            f"fs2_readout_raw={m.get('fs2_readout_raw_norm', 0.0):.3f}"
        )
        parts.append(
            f"fs2_readout_resid={m.get('fs2_readout_residual_norm', 0.0):.3f}"
        )
    if m.get("fs3_innovation_enabled", 0.0) > 0:
        parts.append(
            f"fs3_innov_scale={m.get('fs3_innovation_scale_abs', 0.0):.4f}"
        )
        parts.append(
            f"fs3_innov_frac={m.get('fs3_innovation_fraction', 0.0):.3f}"
        )
        parts.append(
            "fs3_innov_resid="
            f"{m.get('fs3_innovation_residual_relative_rms', 0.0):.4f}"
        )
    if m.get("fs3_codec_enabled", 0.0) > 0:
        parts.append(
            f"fs3_codec_code={m.get('fs3_codec_code_relative_rms', 0.0):.4f}"
        )
        parts.append(
            "fs3_codec_attn="
            f"{m.get('fs3_codec_row_attention_entropy', 0.0):.4f}/"
            f"{m.get('fs3_codec_row_attention_max', 0.0):.4f}/"
            f"{m.get('fs3_codec_row_attention_batch_std', 0.0):.4f}"
        )
        parts.append(
            f"fs3_codec_update={m.get('fs3_codec_update_relative_rms', 0.0):.4f}"
        )
        parts.append(
            "fs3_codec_resid="
            f"{m.get('fs3_codec_residual_relative_rms', 0.0):.4f}/"
            f"{m.get('fs3_codec_residual_batch_std', 0.0):.4f}"
        )
    if m.get("fs3_address_local_enabled", 0.0) > 0:
        parts.append(
            "fs3_addr_gain="
            f"{m.get('fs3_address_local_gain_abs', 0.0):.4f}/"
            f"{m.get('fs3_address_local_gain_row_std', 0.0):.4f}/"
            f"{m.get('fs3_address_local_gain_batch_std', 0.0):.4f}"
        )
        parts.append(
            "fs3_addr_update="
            f"{m.get('fs3_address_local_update_relative_rms', 0.0):.4f}"
        )
        parts.append(
            "fs3_addr_resid="
            f"{m.get('fs3_address_local_residual_relative_rms', 0.0):.4f}/"
            f"{m.get('fs3_address_local_residual_batch_std', 0.0):.4f}"
        )
    if m.get("fs3_basis_transport_enabled", 0.0) > 0:
        parts.append(
            "fs3_basis_angle="
            f"{m.get('fs3_basis_transport_angle_abs', 0.0):.4f}/"
            f"{m.get('fs3_basis_transport_angle_abs_min', 0.0):.4f}"
        )
        parts.append(
            "fs3_basis_kv="
            f"{m.get('fs3_basis_transport_row_rotation_relative_rms', 0.0):.4f}/"
            f"{m.get('fs3_basis_transport_col_rotation_relative_rms', 0.0):.4f}"
        )
        parts.append(
            "fs3_basis_resid="
            f"{m.get('fs3_basis_transport_state_residual_relative_rms', 0.0):.4f}/"
            f"{m.get('fs3_basis_transport_state_residual_relative_rms_min', 0.0):.4f}"
        )
        parts.append(
            "fs3_basis_geom="
            f"{m.get('fs3_basis_transport_fp32_norm_max_error', 0.0):.2e}/"
            f"{m.get('fs3_basis_transport_storage_norm_max_error', 0.0):.2e}/"
            f"{m.get('fs3_basis_transport_orthogonality_max_error', 0.0):.2e}"
        )
    if m.get("gdn3_state_expert_enabled", 0.0) > 0:
        parts.append(
            "state_expert_resid="
            f"{m.get('gdn3_state_expert_residual_relative_rms', 0.0):.4f}/"
            f"{m.get('gdn3_state_expert_residual_batch_std', 0.0):.4f}"
        )
        parts.append(
            "state_expert_state="
            f"{m.get('gdn3_state_expert_terminal_rms', 0.0):.4f}/"
            f"{m.get('gdn3_state_expert_terminal_batch_std', 0.0):.4f}/"
            f"{m.get('gdn3_state_expert_seed_rms', 0.0):.4f}"
        )
        parts.append(
            "state_expert_addr/cos="
            f"{m.get('gdn3_state_expert_address_contrast', 0.0):.4f}/"
            f"{m.get('gdn3_state_expert_output_cosine', 0.0):.4f}"
        )
    if m.get("gdn3_raven_write_control_enabled", 0.0) > 0:
        parts.append(
            "raven_write_v="
            f"{m.get('gdn3_raven_write_control_v_residual_relative_rms', 0.0):.4f}/"
            f"{m.get('gdn3_raven_write_control_v_residual_relative_rms_min', 0.0):.4f}/"
            f"{m.get('gdn3_raven_write_control_v_residual_relative_rms_max', 0.0):.4f}"
        )
        parts.append(
            "raven_write_slot="
            f"{m.get('gdn3_raven_write_control_slot_entropy_normalized_min', 0.0):.4f}/"
            f"{m.get('gdn3_raven_write_control_slot_max_mass_share_max', 0.0):.4f}"
        )
        parts.append(
            "raven_write_state="
            f"{m.get('gdn3_raven_write_control_terminal_rms', 0.0):.4f}/"
            f"{m.get('gdn3_raven_write_control_seed_rms_receiving_min', 0.0):.4f}/"
            f"{m.get('gdn3_raven_write_control_main_terminal_rms_max', 0.0):.4f}"
        )
    if m.get("gdn3_state_feedback_enabled", 0.0) > 0:
        parts.append(
            "state_fb_read="
            f"{m.get('gdn3_state_feedback_read_rms', 0.0):.4f}/"
            f"{m.get('gdn3_state_feedback_read_batch_std', 0.0):.4f}"
        )
        parts.append(
            "state_fb_resid="
            f"{m.get('gdn3_state_feedback_residual_relative_rms', 0.0):.4f}/"
            f"{m.get('gdn3_state_feedback_residual_token_std', 0.0):.4f}"
        )
        parts.append(
            "state_fb_kv/bw="
            f"{m.get('gdn3_state_feedback_k_relative_change', 0.0):.4f}/"
            f"{m.get('gdn3_state_feedback_v_relative_change', 0.0):.4f}/"
            f"{m.get('gdn3_state_feedback_b_relative_change', 0.0):.4f}/"
            f"{m.get('gdn3_state_feedback_w_relative_change', 0.0):.4f}"
        )
    if m.get("gdn3_terminal_consolidation_enabled", 0.0) > 0:
        parts.append(
            "term_cons_k="
            f"{m.get('gdn3_terminal_consolidation_k_relative_rms', 0.0):.4f}/"
            f"{m.get('gdn3_terminal_consolidation_k_batch_std', 0.0):.4f}/"
            f"{m.get('gdn3_terminal_consolidation_k_token_std', 0.0):.4f}"
        )
        parts.append(
            "term_cons_state="
            f"{m.get('gdn3_terminal_consolidation_state_residual_relative_rms', 0.0):.4f}/"
            f"{m.get('gdn3_terminal_consolidation_state_residual_batch_std', 0.0):.4f}"
        )
        parts.append(
            "term_cons_out="
            f"{m.get('gdn3_terminal_consolidation_output_rms', 0.0):.4f}/"
            f"{m.get('gdn3_terminal_consolidation_output_token_std', 0.0):.4f}"
        )
    if m.get("gdn3_orthogonal_chunk_state_enabled", 0.0) > 0:
        parts.append(
            "orth_chunk_angle="
            f"{m.get('gdn3_orthogonal_chunk_state_angle_abs', 0.0):.4f}/"
            f"{m.get('gdn3_orthogonal_chunk_state_angle_batch_std', 0.0):.4f}/"
            f"{m.get('gdn3_orthogonal_chunk_state_angle_head_std', 0.0):.4f}"
        )
        parts.append(
            "orth_chunk_state="
            f"{m.get('gdn3_orthogonal_chunk_state_state_residual_relative_rms', 0.0):.4f}/"
            f"{m.get('gdn3_orthogonal_chunk_state_read_residual_relative_rms', 0.0):.4f}/"
            f"{m.get('gdn3_orthogonal_chunk_state_terminal_rms', 0.0):.4f}"
        )
        parts.append(
            "orth_chunk_geometry="
            f"{m.get('gdn3_orthogonal_chunk_state_plane_dot_abs_max', 0.0):.2e}/"
            f"{m.get('gdn3_orthogonal_chunk_state_plane_norm_error_max', 0.0):.2e}/"
            f"{m.get('gdn3_orthogonal_chunk_state_boundary_norm_ratio_max_error', 0.0):.2e}"
        )
    if m.get("gdn3_orthogonal_head_write_enabled", 0.0) > 0:
        parts.append(
            "orth_head_angle="
            f"{m.get('gdn3_orthogonal_head_write_angle_abs', 0.0):.4f}/"
            f"{m.get('gdn3_orthogonal_head_write_angle_batch_std', 0.0):.4f}/"
            f"{m.get('gdn3_orthogonal_head_write_angle_token_std', 0.0):.4f}"
        )
        parts.append(
            "orth_head_v/state="
            f"{m.get('gdn3_orthogonal_head_write_v_residual_relative_rms', 0.0):.4f}/"
            f"{m.get('gdn3_orthogonal_head_write_terminal_rms', 0.0):.4f}"
        )
        parts.append(
            "orth_head_geometry="
            f"{m.get('gdn3_orthogonal_head_write_plane_dot_abs_max', 0.0):.2e}/"
            f"{m.get('gdn3_orthogonal_head_write_plane_norm_error_max', 0.0):.2e}/"
            f"{m.get('gdn3_orthogonal_head_write_fp32_norm_ratio_max_error', 0.0):.2e}/"
            f"{m.get('gdn3_orthogonal_head_write_storage_norm_ratio_max_error', 0.0):.2e}"
        )
    if m.get("gdn3_adaptive_signed_erase_enabled", 0.0) > 0:
        parts.append(
            "signed_erase_residual="
            f"{m.get('gdn3_adaptive_signed_erase_residual_abs', 0.0):.4f}/"
            f"{m.get('gdn3_adaptive_signed_erase_residual_relative_rms', 0.0):.4f}/"
            f"{m.get('gdn3_adaptive_signed_erase_residual_batch_std', 0.0):.4f}"
        )
        parts.append(
            "signed_erase_b="
            f"{m.get('gdn3_adaptive_signed_erase_effective_mean', 0.0):.4f}/"
            f"{m.get('gdn3_adaptive_signed_erase_effective_min', 0.0):.4f}/"
            f"{m.get('gdn3_adaptive_signed_erase_effective_max', 0.0):.4f}/"
            f"{m.get('gdn3_adaptive_signed_erase_above_one_frac', 0.0):.4f}"
        )
        parts.append(
            "signed_erase_state="
            f"{m.get('gdn3_adaptive_signed_erase_terminal_rms', 0.0):.4f}/"
            f"{m.get('gdn3_adaptive_signed_erase_terminal_batch_std', 0.0):.4f}"
        )
    if m.get("gdn3_bi_axis_value_decay_enabled", 0.0) > 0:
        parts.append(
            "bi_axis_potential/common/decay="
            f"{m.get('gdn3_bi_axis_value_decay_potential_abs', 0.0):.4f}/"
            f"{m.get('gdn3_bi_axis_value_decay_common_k_log_decay_abs', 0.0):.4f}/"
            f"{m.get('gdn3_bi_axis_value_decay_log_decay_abs', 0.0):.4f}/"
            f"{m.get('gdn3_bi_axis_value_decay_group_std', 0.0):.4f}/"
            f"{m.get('gdn3_bi_axis_value_decay_batch_std', 0.0):.4f}/"
            f"{m.get('gdn3_bi_axis_value_decay_token_std', 0.0):.4f}"
        )
        parts.append(
            "bi_axis_scale="
            f"{m.get('gdn3_bi_axis_value_decay_cumulative_scale_min', 1.0):.4f}/"
            f"{m.get('gdn3_bi_axis_value_decay_cumulative_scale_mean', 1.0):.4f}/"
            f"{m.get('gdn3_bi_axis_value_decay_cumulative_scale_max', 1.0):.4f}/"
            f"{m.get('gdn3_bi_axis_value_decay_inverse_scale_max', 1.0):.4f}"
        )
        parts.append(
            "bi_axis_frame/state="
            f"{m.get('gdn3_bi_axis_value_decay_write_frame_relative_rms', 0.0):.4f}/"
            f"{m.get('gdn3_bi_axis_value_decay_state_restore_relative_rms', 0.0):.4f}/"
            f"{m.get('gdn3_bi_axis_value_decay_terminal_rms', 0.0):.4f}"
        )
    if m.get("gdn3_raven_routed_enabled", 0.0) > 0:
        parts.append(
            "raven_alloc="
            f"{m.get('gdn3_raven_routed_allocation_abs_from_one', 0.0):.4f}/"
            f"{m.get('gdn3_raven_routed_allocation_slot_std', 0.0):.4f}/"
            f"{m.get('gdn3_raven_routed_allocation_batch_std', 0.0):.4f}/"
            f"{m.get('gdn3_raven_routed_allocation_token_std', 0.0):.4f}"
        )
        parts.append(
            "raven_entropy/range="
            f"{m.get('gdn3_raven_routed_allocation_entropy_normalized', 1.0):.4f}/"
            f"{m.get('gdn3_raven_routed_allocation_min', 1.0):.4f}/"
            f"{m.get('gdn3_raven_routed_allocation_max', 1.0):.4f}"
        )
        parts.append(
            "raven_kg/state="
            f"{m.get('gdn3_raven_routed_k_relative_change', 0.0):.4f}/"
            f"{m.get('gdn3_raven_routed_g_relative_change', 0.0):.4f}/"
            f"{m.get('gdn3_raven_routed_terminal_rms', 0.0):.4f}"
        )
    if m.get("gdn3_paired_address_bank_enabled", 0.0) > 0:
        parts.append(
            "paired_bank_gate/q/k="
            f"{m.get('gdn3_paired_address_bank_read_gate_abs', 0.0):.4f}/"
            f"{m.get('gdn3_paired_address_bank_q_residual_relative_rms', 0.0):.4f}/"
            f"{m.get('gdn3_paired_address_bank_k_residual_relative_rms', 0.0):.4f}"
        )
        parts.append(
            "paired_bank_state/terminal="
            f"{m.get('gdn3_paired_address_bank_state_residual_relative_rms', 0.0):.4f}/"
            f"{m.get('gdn3_paired_address_bank_terminal_rms', 0.0):.4f}"
        )
        parts.append(
            "paired_bank_contrast="
            f"{m.get('gdn3_paired_address_bank_base_address_contrast', 0.0):.4f}/"
            f"{m.get('gdn3_paired_address_bank_companion_address_contrast', 0.0):.4f}"
        )
    if m.get("gdn3_coupled_address_rows_enabled", 0.0) > 0:
        parts.append(
            "coupled_rows_k="
            f"{m.get('gdn3_coupled_address_rows_k_residual_relative_rms', 0.0):.4f}/"
            f"{m.get('gdn3_coupled_address_rows_k_residual_batch_std', 0.0):.4f}/"
            f"{m.get('gdn3_coupled_address_rows_k_norm_max', 0.0):.4f}"
        )
        parts.append(
            "coupled_rows_state="
            f"{m.get('gdn3_coupled_address_rows_extra_state_relative_rms', 0.0):.4f}/"
            f"{m.get('gdn3_coupled_address_rows_terminal_rms', 0.0):.4f}"
        )
    if m.get("gdn3_interleaved_write_enabled", 0.0) > 0:
        parts.append(
            "interleave_k/v="
            f"{m.get('gdn3_interleaved_write_k_residual_relative_rms', 0.0):.4f}/"
            f"{m.get('gdn3_interleaved_write_v_relative_rms', 0.0):.4f}"
        )
        parts.append(
            "interleave_write/state="
            f"{m.get('gdn3_interleaved_write_state_write_relative_rms', 0.0):.4f}/"
            f"{m.get('gdn3_interleaved_write_terminal_rms', 0.0):.4f}"
        )
    if m.get("gdn2_gain_budget_enabled", 0.0) > 0:
        parts.append(
            "gain_clip="
            f"{m.get('gdn2_gain_budget_clipped_frac', 0.0):.3f}"
        )
        parts.append(
            "gain_lambda="
            f"{m.get('gdn2_gain_budget_lambda_mean', 1.0):.3f}/"
            f"{m.get('gdn2_gain_budget_lambda_min', 1.0):.3f}"
        )
        parts.append(
            "gain_bound="
            f"{m.get('gdn2_gain_budget_original_step_bound', 0.0):.3f}->"
            f"{m.get('gdn2_gain_budget_effective_step_bound', 0.0):.3f}"
        )
        parts.append(
            "gain_delta_err="
            f"{m.get('gdn2_gain_budget_delta_error_max', 0.0):.1e}"
        )
    if "gdn2_fast_slow_enabled" in m:
        parts.append(
            f"slow_decay={m.get('gdn2_fast_slow_enabled', 0.0):.0f}"
        )
        parts.append(
            "slow_rho/lag="
            f"{m.get('gdn2_fast_slow_rho_mean', 0.0):.3f}/"
            f"{m.get('gdn2_fast_slow_lag_mass', 0.0):.3f}"
        )
        parts.append(
            "slow_tv="
            f"{m.get('gdn2_fast_slow_raw_tv', 0.0):.4f}->"
            f"{m.get('gdn2_fast_slow_effective_tv', 0.0):.4f}"
        )
        parts.append(
            f"slow_change={m.get('gdn2_fast_slow_relative_change', 0.0):.4f}"
        )
    if m.get("gdn2_address_enabled", 0.0) > 0:
        parts.append(
            f"addr_scale={m.get('gdn2_address_rotation_scale_abs', 0.0):.4f}"
        )
        parts.append(
            f"addr_phase={m.get('gdn2_address_phase_abs', 0.0):.4f}"
        )
        parts.append(
            "addr_qk_change="
            f"{m.get('gdn2_address_q_relative_change', 0.0):.4f}/"
            f"{m.get('gdn2_address_k_relative_change', 0.0):.4f}"
        )
        parts.append(
            "addr_norm_err="
            f"{m.get('gdn2_address_q_norm_error', 0.0):.1e}/"
            f"{m.get('gdn2_address_k_norm_error', 0.0):.1e}"
        )
        if m.get("gdn2_address_phase_weight_rms", 0.0) > 0:
            parts.append(
                "addr_phase_field="
                f"{m.get('gdn2_address_phase_weight_rms', 0.0):.4f}/"
                f"{m.get('gdn2_address_phase_token_std', 0.0):.4f}/"
                f"{m.get('gdn2_address_phase_plane_std', 0.0):.4f}"
            )
        if m.get("gdn2_address_residual_weight_rms", 0.0) > 0:
            parts.append(
                "addr_residual="
                f"{m.get('gdn2_address_residual_weight_rms', 0.0):.4f}/"
                f"{m.get('gdn2_address_residual_rms', 0.0):.4f}/"
                f"{m.get('gdn2_address_residual_token_std', 0.0):.4f}/"
                f"{m.get('gdn2_address_residual_q_ratio', 0.0):.4f}/"
                f"{m.get('gdn2_address_residual_k_ratio', 0.0):.4f}"
            )
        if m.get("gdn2_address_q_residual_weight_rms", 0.0) > 0:
            parts.append(
                "addr_qk_residual="
                f"{m.get('gdn2_address_q_residual_weight_rms', 0.0):.4f}/"
                f"{m.get('gdn2_address_k_residual_weight_rms', 0.0):.4f}/"
                f"{m.get('gdn2_address_q_residual_rms', 0.0):.4f}/"
                f"{m.get('gdn2_address_k_residual_rms', 0.0):.4f}"
            )
        if m.get("gdn2_address_carrier_mean", 0.0) > 0:
            parts.append(
                "addr_carrier="
                f"{m.get('gdn2_address_carrier_mean', 0.0):.4f}/"
                f"{m.get('gdn2_address_carrier_std', 0.0):.4f}/"
                f"{m.get('gdn2_address_carrier_token_std', 0.0):.4f}/"
                f"{m.get('gdn2_address_carrier_min', 0.0):.4f}"
            )
            parts.append(
                "addr_carrier_effect="
                f"{m.get('gdn2_address_carrier_write_norm_ratio', 0.0):.4f}/"
                f"{m.get('gdn2_address_carrier_b_input_relative_change', 0.0):.4f}/"
                f"{m.get('gdn2_address_carrier_w_input_relative_change', 0.0):.4f}"
            )
    if "gdn3_shared_address_weight_rms" in m:
        parts.append(
            "gdn3_shared_addr="
            f"{m.get('gdn3_shared_address_weight_rms', 0.0):.4f}/"
            f"{m.get('gdn3_shared_address_residual_rms', 0.0):.4f}/"
            f"{m.get('gdn3_shared_address_token_std', 0.0):.4f}"
        )
    if "loop_feedback_in_norm" in m:
        parts.append(f"fb_in={m['loop_feedback_in_norm']:.3f}")
    if "loop_feedback_next_norm" in m:
        parts.append(f"fb_next={m['loop_feedback_next_norm']:.3f}")
    if m.get("loop_feedback_corrupt_mix", 0.0) > 0:
        parts.append(f"fb_corrupt={m.get('loop_feedback_corrupt_frac', 0.0):.3f}")
        parts.append(f"fb_conf={m.get('loop_feedback_corrupt_confidence', 0.0):.3f}")
    if m.get("hidden_agg_noise_norm", 0.0) > 0:
        parts.append(f"hagg_norm={m['hidden_agg_noise_norm']:.3f}")
        parts.append(f"hagg_ent={m.get('hidden_agg_noise_entropy', 0.0):.3f}")
        parts.append(f"hagg_maxw={m.get('hidden_agg_noise_max_weight', 0.0):.3f}")
    if "loop_time_norm" in m:
        parts.append(f"loop_time={m['loop_time_norm']:.3f}")
    if "loop_update_gate_l" in m:
        parts.append(f"upd_l={m['loop_update_gate_l']:.3f}")
    if "loop_update_gate_h" in m:
        parts.append(f"upd_h={m['loop_update_gate_h']:.3f}")
    if "scratch_gate_mean" in m:
        parts.append(f"scratch_gate={m['scratch_gate_mean']:.3f}")
        parts.append(f"scratch_decay={m.get('scratch_decay_mean', 0.0):.3f}")
        parts.append(f"scratch_delta={m.get('scratch_delta_norm', 0.0):.3f}")
        parts.append(f"scratch_resid={m.get('scratch_residual_norm', 0.0):.3f}")
    if m.get("scratch_noise_norm", 0.0) > 0:
        parts.append(f"scratch_noise={m['scratch_noise_norm']:.3f}")
    if m.get("scratch_proj_var_rank", 0.0) > 0:
        parts.append(f"scratch_var={m.get('scratch_proj_var_mean', 0.0):.3f}")
        parts.append(f"scratch_rank={m['scratch_proj_var_rank']:.1f}")
    return ", ".join(parts)


def build_adamw(
    model: nn.Module,
    *,
    lr: float,
    weight_decay: float,
    contract: str,
) -> Tuple[torch.optim.AdamW, Dict[str, Any]]:
    if contract == "uniform":
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        return optimizer, {
            "contract": contract,
            "decay_parameter_count": sum(parameter.numel() for parameter in model.parameters()),
            "no_decay_parameter_count": 0,
            "decay_names": ["*"],
            "no_decay_names": [],
        }
    if contract != "rwkv7_decay_groups":
        raise ValueError(f"unknown optimizer contract: {contract}")

    norm_parameter_ids = {
        id(parameter)
        for module in model.modules()
        if isinstance(module, (nn.LayerNorm, nn.GroupNorm))
        for parameter in module.parameters(recurse=False)
    }
    decay_parameters: List[nn.Parameter] = []
    no_decay_parameters: List[nn.Parameter] = []
    decay_names: List[str] = []
    no_decay_names: List[str] = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        explicit_no_decay = bool(getattr(parameter, "_no_weight_decay", False))
        is_matrix_weight = name.endswith(".weight")
        use_decay = is_matrix_weight and id(parameter) not in norm_parameter_ids and not explicit_no_decay
        if use_decay:
            decay_parameters.append(parameter)
            decay_names.append(name)
        else:
            no_decay_parameters.append(parameter)
            no_decay_names.append(name)
    if not decay_parameters or not no_decay_parameters:
        raise AssertionError("rwkv7_decay_groups requires both decay and no-decay parameters")
    assigned_ids = {id(parameter) for parameter in decay_parameters + no_decay_parameters}
    expected_ids = {id(parameter) for parameter in model.parameters() if parameter.requires_grad}
    if assigned_ids != expected_ids or len(assigned_ids) != len(decay_parameters) + len(no_decay_parameters):
        raise AssertionError("optimizer parameter grouping is incomplete or contains duplicates")

    optimizer = torch.optim.AdamW(
        [
            {"params": decay_parameters, "weight_decay": weight_decay},
            {"params": no_decay_parameters, "weight_decay": 0.0},
        ],
        lr=lr,
    )
    return optimizer, {
        "contract": contract,
        "decay_parameter_count": sum(parameter.numel() for parameter in decay_parameters),
        "no_decay_parameter_count": sum(parameter.numel() for parameter in no_decay_parameters),
        "decay_names": decay_names,
        "no_decay_names": no_decay_names,
    }


def train_model(args: argparse.Namespace, *, device: torch.device) -> Tuple[FutureSeedLoopSudoku, Dict[str, Any]]:
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    rng = random.Random(args.seed + 1000)
    official_train = (
        OfficialSudokuDataset(
            Path(args.official_sudoku_data_dir),
            args.official_sudoku_train_split,
            row_indices_path=(
                Path(args.official_sudoku_train_indices)
                if str(args.official_sudoku_train_indices).strip()
                else None
            ),
        )
        if official_sudoku_enabled(args)
        else None
    )
    official_eval = (
        OfficialSudokuDataset(Path(args.official_sudoku_data_dir), args.official_sudoku_eval_split)
        if official_sudoku_enabled(args)
        else None
    )
    model = FutureSeedLoopSudoku(
        d_model=args.d_model,
        layers=args.layers,
        heads=args.heads,
        head_dim=args.head_dim,
        channel_mult=args.channel_mult,
        l_cycles=args.l_cycles,
        max_loops=args.max_loops,
        lambda_=args.lambda_,
        loop_update_mode=args.loop_update_mode,
        loop_update_gate_init=args.loop_update_gate_init,
        future_seed_scale=args.future_seed_scale,
        future_seed_decay=args.future_seed_decay,
        future_seed_update=args.future_seed_update,
        future_seed_norm_mode=args.future_seed_norm_mode,
        future_seed_gate_mode=args.future_seed_gate_mode,
        future_seed_scope=args.future_seed_scope,
        future_seed_readout_hop=args.future_seed_readout_hop,
        future_seed_content_mode=args.future_seed_content_mode,
        momentum_future_seed_transport=args.momentum_future_seed_transport,
        loop_feedback_scale=args.loop_feedback_scale,
        loop_feedback_detach=bool(args.loop_feedback_detach),
        loop_feedback_corrupt_prob=args.loop_feedback_corrupt_prob,
        loop_feedback_corrupt_mix=args.loop_feedback_corrupt_mix,
        loop_feedback_corrupt_mode=args.loop_feedback_corrupt_mode,
        loop_time_scale=args.loop_time_scale,
        scratch_mode=args.scratch_mode,
        scratch_scale=args.scratch_scale,
        scratch_noise_scale=args.scratch_noise_scale,
        scratch_gauss_projections=args.scratch_gauss_projections,
        scratch_gate_bias=args.scratch_gate_bias,
        scratch_decay_bias=args.scratch_decay_bias,
        hidden_agg_noise_scale=args.hidden_agg_noise_scale,
        hidden_agg_noise_temp=args.hidden_agg_noise_temp,
        hidden_agg_noise_detach=bool(args.hidden_agg_noise_detach),
        hidden_agg_noise_mode=args.hidden_agg_noise_mode,
        hidden_agg_noise_topk=args.hidden_agg_noise_topk,
        hidden_agg_noise_max_norm=args.hidden_agg_noise_max_norm,
        activation_checkpoint=args.activation_checkpoint,
        rwkv_kernel=args.rwkv_kernel,
        backbone=args.backbone,
        gdn_mode=args.gdn_mode,
        gdn_expand_v=args.gdn_expand_v,
        gdn_progressive_base_expand_v=args.gdn_progressive_base_expand_v,
        gdn_use_short_conv=args.gdn_use_short_conv,
        gdn_conv_size=args.gdn_conv_size,
        gdn_allow_neg_eigval=args.gdn_allow_neg_eigval,
        gdn2_gain_budget_mode=args.gdn2_gain_budget_mode,
        gdn2_gain_budget_sigma_cap=args.gdn2_gain_budget_sigma_cap,
        gdn2_gain_budget_step_cap=args.gdn2_gain_budget_step_cap,
        gdn2_gain_budget_sigma_cap_max=args.gdn2_gain_budget_sigma_cap_max,
        gdn2_gain_budget_infeasible_policy=args.gdn2_gain_budget_infeasible_policy,
        gdn2_fast_slow_decay_mode=args.gdn2_fast_slow_decay_mode,
        gdn2_fast_slow_decay_kernel_size=args.gdn2_fast_slow_decay_kernel_size,
        gdn2_fast_slow_decay_rho_init=args.gdn2_fast_slow_decay_rho_init,
        gdn2_fast_slow_decay_current_weight_init=(
            args.gdn2_fast_slow_decay_current_weight_init
        ),
        gdn2_precondition_mode=args.gdn2_precondition_mode,
        gdn2_address_mode=args.gdn2_address_mode,
        gdn2_update_mode=args.gdn2_update_mode,
        gdn2_state_expert_mode=args.gdn2_state_expert_mode,
        gdn2_cross_layer_init=args.gdn2_cross_layer_init,
        raven_num_slots=args.raven_num_slots,
        raven_topk=args.raven_topk,
    )
    if args.shared_shell_init_seed >= 0:
        model.reset_shared_shell_parameters(args.shared_shell_init_seed)
    model = model.to(device)
    gate_parameters = future_seed_gate_parameters(model)
    if args.future_seed_gradient_mode == "opening_projection":
        if len(gate_parameters) != 12 or sum(
            parameter.numel() for _name, parameter in gate_parameters
        ) != 96:
            raise RuntimeError(
                "Registered FutureSeed opening projection requires exactly "
                "12 gate tensors and 96 gate parameters"
            )
    post_init_torch_seed = args.seed + 2000
    torch.manual_seed(post_init_torch_seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(post_init_torch_seed)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    trainable_parameter_count = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    fla_runtime = (
        strict_fla_runtime_summary(model, args.backbone)
        if args.fla_strict_official
        else {"strict": False}
    )
    if args.backbone in {"rwkv", "rwkv7"}:
        statepassing_ok, statepassing_reason = statepassing_available(args.head_dim)
        backbone_runtime = {
            "implementation": (
                "official_rwkv7_timemix_with_explicit_state_io"
                if args.backbone == "rwkv7"
                else "deprecated_local_rwkv_style_statepassing"
            ),
            "requested_kernel": args.rwkv_kernel,
            "statepassing_available": statepassing_ok,
            "statepassing_reason": statepassing_reason,
            "silent_fallback_allowed": args.rwkv_kernel == "auto",
        }
        if args.backbone == "rwkv7":
            backbone_runtime.update(
                {
                    "official_source_commit": RWKV7_OFFICIAL_SOURCE_COMMIT,
                    "official_source_blob": RWKV7_OFFICIAL_SOURCE_BLOB,
                    "official_source_path": RWKV7_OFFICIAL_SOURCE_PATH,
                    "official_kernel_blob": RWKV7_OFFICIAL_KERNEL_BLOB,
                    "official_kernel_path": RWKV7_OFFICIAL_KERNEL_PATH,
                    "statepassing_cuda_sha256": RWKV7_STATEPASSING_CUDA_SHA256,
                    "shared_shell_channel_mix": True,
                }
            )
    elif args.backbone == "gdn":
        backbone_runtime = {
            "implementation": "local_gdn_triton",
            "requested_kernel": args.gdn_mode,
            "silent_fallback_allowed": False,
        }
    else:
        backbone_runtime = {
            "implementation": f"official_fla_{args.backbone}",
            "requested_kernel": args.gdn_mode,
            "silent_fallback_allowed": not bool(args.fla_strict_official),
        }
    feature_buffer = FeatureNoiseBuffer(
        capacity=args.feature_buffer_size,
        feature_dim=args.d_model,
        device=device,
        dtype=torch.float32,
    )
    model.feature_noise_buffer = feature_buffer
    opt, optimizer_runtime = build_adamw(
        model,
        lr=args.lr,
        weight_decay=args.weight_decay,
        contract=args.optimizer_contract,
    )
    t0 = time.time()
    resume_info: Dict[str, Any] = {}
    saved_train_checkpoints: List[str] = []
    last_ce_loss = 0.0
    last_total_loss = 0.0
    last_loop1_loss = 0.0
    last_loop_last_loss = 0.0
    last_loop_weights: List[float] = []
    last_exact_margin_loss = 0.0
    last_exact_margin_softmin = 0.0
    last_exact_margin_hardmin = 0.0
    last_exact_margin_mean = 0.0
    last_scratch_gauss_loss = 0.0
    last_scratch_gate = 0.0
    last_scratch_decay = 0.0
    last_scratch_delta = 0.0
    last_scratch_residual = 0.0
    last_scratch_proj_var = 0.0
    last_scratch_proj_rank = 0.0
    last_loop_update_gate_l = 0.0
    last_loop_update_gate_h = 0.0
    last_loop_feedback_next_norm = 0.0
    last_loop_feedback_corrupt_frac = 0.0
    last_loop_feedback_corrupt_confidence = 0.0
    last_hidden_agg_noise_norm = 0.0
    last_hidden_agg_noise_raw_norm = 0.0
    last_hidden_agg_noise_entropy = 0.0
    last_hidden_agg_noise_max_weight = 0.0
    last_hidden_agg_noise_clip_frac = 0.0
    last_gain_budget_clipped_frac = 0.0
    last_gain_budget_infeasible_frac = 0.0
    last_gain_budget_numerical_endpoint_frac = 0.0
    last_gain_budget_step_bound_max = 0.0
    last_gain_budget_delta_error_max = 0.0
    last_gain_budget_gate_relative_change = 0.0
    last_fast_slow_diag = {
        key: 0.0 for key in FAST_SLOW_TRAIN_KEYS
    }
    last_precondition_diag = {
        key: 0.0 for key in PRECONDITION_TRAIN_KEYS
    }
    last_coherent_delta_diag = {
        key: 0.0 for key in COHERENT_DELTA_TRAIN_KEYS
    }
    last_coherent_delta_diag["gdn2_coherent_delta_gap_ratio"] = 1.0
    last_state_feedback_diag = {
        key: 0.0 for key in STATE_FEEDBACK_TRAIN_KEYS
    }
    last_terminal_consolidation_diag = {
        key: 0.0 for key in TERMINAL_CONSOLIDATION_TRAIN_KEYS
    }
    last_orthogonal_chunk_state_diag = {
        key: 0.0 for key in ORTHOGONAL_CHUNK_STATE_TRAIN_KEYS
    }
    last_orthogonal_chunk_state_diag[
        "gdn3_orthogonal_chunk_state_boundary_norm_ratio_mean"
    ] = 1.0
    last_orthogonal_head_write_diag = {
        key: 0.0 for key in ORTHOGONAL_HEAD_WRITE_TRAIN_KEYS
    }
    last_orthogonal_head_write_diag[
        "gdn3_orthogonal_head_write_fp32_norm_ratio_mean"
    ] = 1.0
    last_orthogonal_head_write_diag[
        "gdn3_orthogonal_head_write_storage_norm_ratio_mean"
    ] = 1.0
    last_adaptive_signed_erase_diag = {
        key: 0.0 for key in ADAPTIVE_SIGNED_ERASE_TRAIN_KEYS
    }
    last_bi_axis_value_decay_diag = {
        key: 0.0 for key in BI_AXIS_VALUE_DECAY_TRAIN_KEYS
    }
    for key in (
        "gdn3_bi_axis_value_decay_cumulative_scale_min",
        "gdn3_bi_axis_value_decay_cumulative_scale_mean",
        "gdn3_bi_axis_value_decay_cumulative_scale_max",
        "gdn3_bi_axis_value_decay_inverse_scale_max",
    ):
        last_bi_axis_value_decay_diag[key] = 1.0
    last_raven_routed_gdn_diag = {
        key: 0.0 for key in RAVEN_ROUTED_GDN_TRAIN_KEYS
    }
    for key in (
        "gdn3_raven_routed_allocation_entropy_normalized",
        "gdn3_raven_routed_allocation_min",
        "gdn3_raven_routed_allocation_max",
    ):
        last_raven_routed_gdn_diag[key] = 1.0
    last_paired_address_bank_diag = {
        key: 0.0 for key in PAIRED_ADDRESS_BANK_TRAIN_KEYS
    }
    last_coupled_address_rows_diag = {
        key: 0.0 for key in COUPLED_ADDRESS_ROWS_TRAIN_KEYS
    }
    last_interleaved_write_diag = {
        key: 0.0 for key in INTERLEAVED_WRITE_TRAIN_KEYS
    }
    last_state_expert_diag = {
        key: 0.0 for key in STATE_EXPERT_TRAIN_KEYS
    }
    last_address_diag = {
        key: 0.0 for key in ADDRESS_TRAIN_KEYS
    }
    future_seed_projection_stats = new_future_seed_projection_stats()
    last_future_seed_projection_diag: Dict[str, float] = {}
    stages = parse_hole_stages(args)
    checkpoint_steps = parse_eval_checkpoint_steps(args, stages)
    checkpoint_step_set = set(checkpoint_steps)
    checkpoint_evals: Dict[str, Any] = {}
    checkpoint_batches = {}
    if checkpoint_steps:
        out_dir = Path(args.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_holes = parse_eval_checkpoint_holes(args)
        checkpoint_batches = {}
        for holes in checkpoint_holes:
            checkpoint_seed = args.seed + 13000 + holes * 17
            if official_eval is not None:
                checkpoint_batches[holes] = official_eval.fixed_batch_by_blank_range(
                    args.eval_n,
                    checkpoint_seed,
                    holes_min=holes,
                    holes_max=holes,
                    device=device,
                )
            else:
                checkpoint_batches[holes] = make_eval_batch(
                    args,
                    None,
                    args.eval_n,
                    holes,
                    checkpoint_seed,
                    device=device,
                )
        print(
            "checkpoint_eval "
            f"steps={','.join(str(step) for step in checkpoint_steps)} "
            f"holes={','.join(str(holes) for holes in checkpoint_holes)}",
            flush=True,
        )

    global_step = 0
    if str(args.resume_train_checkpoint).strip():
        checkpoint = load_training_checkpoint(
            args.resume_train_checkpoint,
            model=model,
            opt=opt,
            feature_buffer=feature_buffer,
            rng=rng,
            device=device,
            expected_args=args if args.resume_require_exact_state else None,
        )
        migration = checkpoint.get("_load_migration", {})
        expected_fast_slow_insertions = {
            name
            for name, _parameter in model.named_parameters()
            if ".time_mix.fast_slow_decay." in name
        }
        migrated_missing = set(migration.get("missing_parameters", []))
        declared_fast_slow_migration = (
            bool(expected_fast_slow_insertions)
            and migrated_missing == expected_fast_slow_insertions
            and not migration.get("unexpected_parameters")
            and migration.get("optimizer_groups_expanded") is True
        )
        expected_content_upgrade_parameters = (
            {"reasoner.future_seed_innovation_scale"}
            if args.future_seed_content_mode == "innovation_residual"
            else {
                name
                for name, _parameter in model.named_parameters()
                if name.startswith("reasoner.future_seed_producer_codec.")
            }
            if args.future_seed_content_mode == "producer_codec"
            else {
                name
                for name, _parameter in model.named_parameters()
                if name.startswith("reasoner.future_seed_address_local_update.")
            }
            if args.future_seed_content_mode == "address_local_update"
            else {
                name
                for name, _parameter in model.named_parameters()
                if name.startswith("reasoner.future_seed_basis_transport.")
            }
            if args.future_seed_content_mode == "orthogonal_basis_transport"
            else set()
        )
        declared_future_seed_content_upgrade = (
            bool(args.resume_allow_future_seed_content_upgrade)
            and bool(expected_content_upgrade_parameters)
            and migrated_missing == expected_content_upgrade_parameters
            and not migration.get("unexpected_parameters")
            and migration.get("optimizer_groups_expanded") is True
            and checkpoint.get("_resume_contract", {}).get(
                "accepted_future_seed_content_upgrade"
            )
            is True
        )
        expected_future_seed_update_parameters = (
            {"reasoner.future_seed_secant_raw"}
            if args.future_seed_update == "loop_secant"
            else set()
        )
        declared_future_seed_update_upgrade = (
            bool(args.resume_allow_future_seed_update_upgrade)
            and args.future_seed_update == "loop_secant"
            and migrated_missing == expected_future_seed_update_parameters
            and not migration.get("unexpected_parameters")
            and migration.get("optimizer_groups_expanded") is True
            and checkpoint.get("_resume_contract", {}).get(
                "accepted_future_seed_update_upgrade"
            )
            is True
        )
        declared_future_seed_gradient_upgrade = (
            bool(args.resume_allow_future_seed_gradient_upgrade)
            and args.future_seed_gradient_mode == "opening_projection"
            and not migration.get("missing_parameters")
            and not migration.get("unexpected_parameters")
            and migration.get("optimizer_groups_expanded") is False
            and checkpoint.get("_resume_contract", {}).get(
                "accepted_future_seed_gradient_upgrade"
            )
            is True
        )
        if (
            args.resume_allow_future_seed_gradient_upgrade
            and not declared_future_seed_gradient_upgrade
        ):
            raise RuntimeError(
                "FutureSeed gradient semantic upgrade was not accepted exactly"
            )
        if args.gdn2_update_mode == "log_spd_metric":
            expected_gdn2_update_insertions = {
                name
                for name, _parameter in model.named_parameters()
                if name.endswith(".time_mix.log_spd_address_metric.raw")
            }
        elif args.gdn2_update_mode == "coherent_delta":
            expected_gdn2_update_insertions = {
                name
                for name, _parameter in model.named_parameters()
                if name.endswith(".time_mix.coherent_delta_mix")
            }
        elif args.gdn2_update_mode == "state_feedback":
            expected_gdn2_update_insertions = {
                name
                for name, _parameter in model.named_parameters()
                if ".time_mix.state_feedback_" in name
            }
        elif args.gdn2_update_mode == "terminal_consolidation":
            expected_gdn2_update_insertions = {
                name
                for name, _parameter in model.named_parameters()
                if name.endswith(
                    ".time_mix.terminal_consolidation_k_proj.weight"
                )
            }
        elif args.gdn2_update_mode == "orthogonal_chunk_state":
            expected_gdn2_update_insertions = {
                name
                for name, _parameter in model.named_parameters()
                if name.endswith(
                    ".time_mix.orthogonal_chunk_state_proj.weight"
                )
            }
        elif args.gdn2_update_mode == "orthogonal_head_write":
            expected_gdn2_update_insertions = {
                name
                for name, _parameter in model.named_parameters()
                if name.endswith(
                    ".time_mix.orthogonal_head_write_proj.weight"
                )
            }
        elif args.gdn2_update_mode == "adaptive_signed_erase":
            expected_gdn2_update_insertions = {
                name
                for name, _parameter in model.named_parameters()
                if name.endswith(
                    ".time_mix.adaptive_signed_erase_proj.weight"
                )
            }
        elif args.gdn2_update_mode in {
            "bi_axis_value_decay",
            "gauge_balanced_bi_axis",
        }:
            expected_gdn2_update_insertions = {
                name
                for name, _parameter in model.named_parameters()
                if name.endswith(
                    ".time_mix.bi_axis_value_decay_proj.weight"
                )
            }
        elif args.gdn2_update_mode == "raven_routed_gdn":
            expected_gdn2_update_insertions = {
                name
                for name, _parameter in model.named_parameters()
                if name.endswith(".time_mix.raven_route_proj.weight")
            }
        elif args.gdn2_update_mode == "paired_address_bank":
            expected_gdn2_update_insertions = {
                name
                for name, _parameter in model.named_parameters()
                if ".time_mix.paired_address_" in name
            }
        elif args.gdn2_update_mode == "coupled_address_rows":
            expected_gdn2_update_insertions = {
                name
                for name, _parameter in model.named_parameters()
                if name.endswith(
                    ".time_mix.coupled_address_k_proj.weight"
                )
            }
        elif args.gdn2_update_mode == "interleaved_write":
            expected_gdn2_update_insertions = {
                name
                for name, _parameter in model.named_parameters()
                if ".time_mix.interleaved_write_" in name
            }
        else:
            expected_gdn2_update_insertions = set()
        declared_gdn2_update_upgrade = (
            bool(args.resume_allow_gdn2_update_upgrade)
            and args.gdn2_update_mode
            in {
                "log_spd_metric",
                "coherent_delta",
                "state_feedback",
                "terminal_consolidation",
                "orthogonal_chunk_state",
                "orthogonal_head_write",
                "adaptive_signed_erase",
                "bi_axis_value_decay",
                "gauge_balanced_bi_axis",
                "raven_routed_gdn",
                "paired_address_bank",
                "coupled_address_rows",
                "interleaved_write",
            }
            and bool(expected_gdn2_update_insertions)
            and migrated_missing == expected_gdn2_update_insertions
            and not migration.get("unexpected_parameters")
            and migration.get("optimizer_groups_expanded") is True
            and checkpoint.get("_resume_contract", {}).get(
                "accepted_gdn2_update_upgrade"
            )
            is True
        )
        expected_gdn2_state_expert_insertions = {
            name
            for name, _parameter in model.named_parameters()
            if ".state_expert." in name
        }
        declared_gdn2_state_expert_upgrade = (
            bool(args.resume_allow_gdn2_state_expert_upgrade)
            and args.gdn2_state_expert_mode
            in {"dual_state", "raven_write_control"}
            and bool(expected_gdn2_state_expert_insertions)
            and migrated_missing == expected_gdn2_state_expert_insertions
            and not migration.get("unexpected_parameters")
            and migration.get("optimizer_groups_expanded") is True
            and checkpoint.get("_resume_contract", {}).get(
                "accepted_gdn2_state_expert_upgrade"
            )
            is True
        )
        if args.resume_require_exact_state and (
            (
                migration.get("missing_parameters")
                or migration.get("unexpected_parameters")
                or migration.get("optimizer_groups_expanded")
            )
            and not declared_fast_slow_migration
            and not declared_future_seed_content_upgrade
            and not declared_future_seed_update_upgrade
            and not declared_gdn2_update_upgrade
            and not declared_gdn2_state_expert_upgrade
        ):
            raise RuntimeError(
                "Exact checkpoint resume required, but migration was needed: "
                f"{migration}"
            )
        global_step = int(checkpoint.get("saved_at_step", 0))
        checkpoint_evals = dict(checkpoint.get("checkpoint_evals", {}))
        last_metrics = dict(checkpoint.get("last_metrics", {}))
        last_ce_loss = float(last_metrics.get("ce_loss", 0.0))
        last_total_loss = float(last_metrics.get("total_loss", 0.0))
        last_loop1_loss = float(last_metrics.get("loop1_loss", 0.0))
        last_loop_last_loss = float(last_metrics.get("loop_last_loss", 0.0))
        last_loop_weights = [float(x) for x in last_metrics.get("loop_weights", [])]
        last_exact_margin_loss = float(last_metrics.get("exact_margin_loss", 0.0))
        last_exact_margin_softmin = float(last_metrics.get("exact_margin_softmin", 0.0))
        last_exact_margin_hardmin = float(last_metrics.get("exact_margin_hardmin", 0.0))
        last_exact_margin_mean = float(last_metrics.get("exact_margin_mean", 0.0))
        last_scratch_gauss_loss = float(last_metrics.get("scratch_gauss_loss", 0.0))
        last_scratch_gate = float(last_metrics.get("scratch_gate", 0.0))
        last_scratch_decay = float(last_metrics.get("scratch_decay", 0.0))
        last_scratch_delta = float(last_metrics.get("scratch_delta", 0.0))
        last_scratch_residual = float(last_metrics.get("scratch_residual", 0.0))
        last_scratch_proj_var = float(last_metrics.get("scratch_proj_var", 0.0))
        last_scratch_proj_rank = float(last_metrics.get("scratch_proj_rank", 0.0))
        last_loop_update_gate_l = float(last_metrics.get("loop_update_gate_l", 0.0))
        last_loop_update_gate_h = float(last_metrics.get("loop_update_gate_h", 0.0))
        last_loop_feedback_next_norm = float(last_metrics.get("loop_feedback_next_norm", 0.0))
        last_loop_feedback_corrupt_frac = float(last_metrics.get("loop_feedback_corrupt_frac", 0.0))
        last_loop_feedback_corrupt_confidence = float(last_metrics.get("loop_feedback_corrupt_confidence", 0.0))
        last_hidden_agg_noise_norm = float(last_metrics.get("hidden_agg_noise_norm", 0.0))
        last_hidden_agg_noise_raw_norm = float(last_metrics.get("hidden_agg_noise_raw_norm", 0.0))
        last_hidden_agg_noise_entropy = float(last_metrics.get("hidden_agg_noise_entropy", 0.0))
        last_hidden_agg_noise_max_weight = float(last_metrics.get("hidden_agg_noise_max_weight", 0.0))
        last_hidden_agg_noise_clip_frac = float(last_metrics.get("hidden_agg_noise_clip_frac", 0.0))
        last_gain_budget_clipped_frac = float(
            last_metrics.get("gain_budget_clipped_frac", 0.0)
        )
        last_gain_budget_infeasible_frac = float(
            last_metrics.get("gain_budget_infeasible_frac", 0.0)
        )
        last_gain_budget_numerical_endpoint_frac = float(
            last_metrics.get("gain_budget_numerical_endpoint_frac", 0.0)
        )
        last_gain_budget_step_bound_max = float(
            last_metrics.get("gain_budget_step_bound_max", 0.0)
        )
        last_gain_budget_delta_error_max = float(
            last_metrics.get("gain_budget_delta_error_max", 0.0)
        )
        last_gain_budget_gate_relative_change = float(
            last_metrics.get("gain_budget_gate_relative_change", 0.0)
        )
        saved_fast_slow_diag = last_metrics.get("fast_slow_decay", {})
        if isinstance(saved_fast_slow_diag, dict):
            last_fast_slow_diag = {
                key: float(saved_fast_slow_diag.get(key, 0.0))
                for key in FAST_SLOW_TRAIN_KEYS
            }
        saved_precondition_diag = last_metrics.get("precondition", {})
        if isinstance(saved_precondition_diag, dict):
            last_precondition_diag = {
                key: float(saved_precondition_diag.get(key, 0.0))
                for key in PRECONDITION_TRAIN_KEYS
            }
        saved_coherent_delta_diag = last_metrics.get("coherent_delta", {})
        if isinstance(saved_coherent_delta_diag, dict):
            last_coherent_delta_diag = {
                key: float(
                    saved_coherent_delta_diag.get(
                        key,
                        1.0 if key == "gdn2_coherent_delta_gap_ratio" else 0.0,
                    )
                )
                for key in COHERENT_DELTA_TRAIN_KEYS
            }
        saved_state_feedback_diag = last_metrics.get("state_feedback", {})
        if isinstance(saved_state_feedback_diag, dict):
            last_state_feedback_diag = {
                key: float(saved_state_feedback_diag.get(key, 0.0))
                for key in STATE_FEEDBACK_TRAIN_KEYS
            }
        saved_terminal_consolidation_diag = last_metrics.get(
            "terminal_consolidation", {}
        )
        if isinstance(saved_terminal_consolidation_diag, dict):
            last_terminal_consolidation_diag = {
                key: float(saved_terminal_consolidation_diag.get(key, 0.0))
                for key in TERMINAL_CONSOLIDATION_TRAIN_KEYS
            }
        saved_orthogonal_chunk_state_diag = last_metrics.get(
            "orthogonal_chunk_state", {}
        )
        if isinstance(saved_orthogonal_chunk_state_diag, dict):
            last_orthogonal_chunk_state_diag = {
                key: float(
                    saved_orthogonal_chunk_state_diag.get(
                        key,
                        1.0
                        if key
                        == "gdn3_orthogonal_chunk_state_boundary_norm_ratio_mean"
                        else 0.0,
                    )
                )
                for key in ORTHOGONAL_CHUNK_STATE_TRAIN_KEYS
            }
        saved_orthogonal_head_write_diag = last_metrics.get(
            "orthogonal_head_write", {}
        )
        if isinstance(saved_orthogonal_head_write_diag, dict):
            last_orthogonal_head_write_diag = {
                key: float(
                    saved_orthogonal_head_write_diag.get(
                        key,
                        1.0
                        if key
                        in {
                            "gdn3_orthogonal_head_write_fp32_norm_ratio_mean",
                            "gdn3_orthogonal_head_write_storage_norm_ratio_mean",
                        }
                        else 0.0,
                    )
                )
                for key in ORTHOGONAL_HEAD_WRITE_TRAIN_KEYS
            }
        saved_adaptive_signed_erase_diag = last_metrics.get(
            "adaptive_signed_erase", {}
        )
        if isinstance(saved_adaptive_signed_erase_diag, dict):
            last_adaptive_signed_erase_diag = {
                key: float(saved_adaptive_signed_erase_diag.get(key, 0.0))
                for key in ADAPTIVE_SIGNED_ERASE_TRAIN_KEYS
            }
        saved_bi_axis_value_decay_diag = last_metrics.get(
            "bi_axis_value_decay", {}
        )
        if isinstance(saved_bi_axis_value_decay_diag, dict):
            last_bi_axis_value_decay_diag = {
                key: float(
                    saved_bi_axis_value_decay_diag.get(
                        key,
                        1.0
                        if key
                        in {
                            "gdn3_bi_axis_value_decay_cumulative_scale_min",
                            "gdn3_bi_axis_value_decay_cumulative_scale_mean",
                            "gdn3_bi_axis_value_decay_cumulative_scale_max",
                            "gdn3_bi_axis_value_decay_inverse_scale_max",
                        }
                        else 0.0,
                    )
                )
                for key in BI_AXIS_VALUE_DECAY_TRAIN_KEYS
            }
        saved_raven_routed_gdn_diag = last_metrics.get(
            "raven_routed_gdn", {}
        )
        if isinstance(saved_raven_routed_gdn_diag, dict):
            last_raven_routed_gdn_diag = {
                key: float(
                    saved_raven_routed_gdn_diag.get(
                        key,
                        1.0
                        if key
                        in {
                            "gdn3_raven_routed_allocation_entropy_normalized",
                            "gdn3_raven_routed_allocation_min",
                            "gdn3_raven_routed_allocation_max",
                        }
                        else 0.0,
                    )
                )
                for key in RAVEN_ROUTED_GDN_TRAIN_KEYS
            }
        saved_paired_address_bank_diag = last_metrics.get(
            "paired_address_bank", {}
        )
        if isinstance(saved_paired_address_bank_diag, dict):
            last_paired_address_bank_diag = {
                key: float(saved_paired_address_bank_diag.get(key, 0.0))
                for key in PAIRED_ADDRESS_BANK_TRAIN_KEYS
            }
        saved_coupled_address_rows_diag = last_metrics.get(
            "coupled_address_rows", {}
        )
        if isinstance(saved_coupled_address_rows_diag, dict):
            last_coupled_address_rows_diag = {
                key: float(saved_coupled_address_rows_diag.get(key, 0.0))
                for key in COUPLED_ADDRESS_ROWS_TRAIN_KEYS
            }
        saved_interleaved_write_diag = last_metrics.get(
            "interleaved_write", {}
        )
        if isinstance(saved_interleaved_write_diag, dict):
            last_interleaved_write_diag = {
                key: float(saved_interleaved_write_diag.get(key, 0.0))
                for key in INTERLEAVED_WRITE_TRAIN_KEYS
            }
        saved_state_expert_diag = last_metrics.get("state_expert", {})
        if isinstance(saved_state_expert_diag, dict):
            last_state_expert_diag = {
                key: float(saved_state_expert_diag.get(key, 0.0))
                for key in STATE_EXPERT_TRAIN_KEYS
            }
        saved_address_diag = last_metrics.get("address_operator", {})
        if isinstance(saved_address_diag, dict):
            last_address_diag = {
                key: float(saved_address_diag.get(key, 0.0))
                for key in ADDRESS_TRAIN_KEYS
            }
        saved_projection_stats = last_metrics.get(
            "future_seed_gradient_projection_state", {}
        )
        if isinstance(saved_projection_stats, dict) and (
            args.future_seed_gradient_mode == "opening_projection"
        ):
            for key in future_seed_projection_stats:
                if key in saved_projection_stats:
                    future_seed_projection_stats[key] = float(
                        saved_projection_stats[key]
                    )
        resume_info = {
            "path": str(args.resume_train_checkpoint),
            "saved_at_step": global_step,
            "checkpoint_version": checkpoint.get("version"),
            "checkpoint_reason": checkpoint.get("reason"),
            "checkpoint_elapsed_sec": checkpoint.get("elapsed_sec"),
            "load_migration": checkpoint.get("_load_migration", {}),
            "semantic_contract": checkpoint.get("_resume_contract", {}),
        }
        print(
            f"resumed_train_checkpoint path={args.resume_train_checkpoint} "
            f"step={global_step} reason={checkpoint.get('reason', '')}",
            flush=True,
        )

    total_steps = sum(stage_steps for _lo, _hi, stage_steps in stages)
    if global_step > total_steps:
        raise ValueError(f"resume step {global_step} exceeds configured total steps {total_steps}")
    last_saved_step = -1
    completed_before_stage = 0
    for stage_idx, (holes_min, holes_max, stage_steps) in enumerate(stages, start=1):
        stage_end = completed_before_stage + stage_steps
        if global_step >= stage_end:
            completed_before_stage = stage_end
            continue
        for _ in range(global_step - completed_before_stage, stage_steps):
            global_step += 1
            model.train()
            opt.zero_grad(set_to_none=True)
            accum_count = max(1, int(args.grad_accum_steps))
            accum_ce_loss = 0.0
            accum_total_loss = 0.0
            accum_loop1_loss = 0.0
            accum_loop_last_loss = 0.0
            accum_exact_margin_loss = 0.0
            accum_exact_margin_softmin = 0.0
            accum_exact_margin_hardmin = 0.0
            accum_exact_margin_mean = 0.0
            accum_scratch_gauss_loss = 0.0
            accum_scratch_gate = 0.0
            accum_scratch_decay = 0.0
            accum_scratch_delta = 0.0
            accum_scratch_residual = 0.0
            accum_scratch_proj_var = 0.0
            accum_scratch_proj_rank = 0.0
            accum_loop_update_gate_l = 0.0
            accum_loop_update_gate_h = 0.0
            accum_loop_feedback_next_norm = 0.0
            accum_loop_feedback_corrupt_frac = 0.0
            accum_loop_feedback_corrupt_confidence = 0.0
            accum_hidden_agg_noise_norm = 0.0
            accum_hidden_agg_noise_raw_norm = 0.0
            accum_hidden_agg_noise_entropy = 0.0
            accum_hidden_agg_noise_max_weight = 0.0
            accum_hidden_agg_noise_clip_frac = 0.0
            accum_gain_budget_clipped_frac = 0.0
            accum_gain_budget_infeasible_frac = 0.0
            accum_gain_budget_numerical_endpoint_frac = 0.0
            accum_gain_budget_step_bound_max = 0.0
            accum_gain_budget_delta_error_max = 0.0
            accum_gain_budget_gate_relative_change = 0.0
            accum_fast_slow_diag = {
                key: 0.0 for key in FAST_SLOW_TRAIN_KEYS
            }
            accum_precondition_diag = {
                key: 0.0 for key in PRECONDITION_TRAIN_KEYS
            }
            accum_coherent_delta_diag = {
                key: 0.0 for key in COHERENT_DELTA_TRAIN_KEYS
            }
            accum_state_feedback_diag = {
                key: 0.0 for key in STATE_FEEDBACK_TRAIN_KEYS
            }
            accum_terminal_consolidation_diag = {
                key: 0.0 for key in TERMINAL_CONSOLIDATION_TRAIN_KEYS
            }
            accum_orthogonal_chunk_state_diag = {
                key: 0.0 for key in ORTHOGONAL_CHUNK_STATE_TRAIN_KEYS
            }
            accum_orthogonal_head_write_diag = {
                key: 0.0 for key in ORTHOGONAL_HEAD_WRITE_TRAIN_KEYS
            }
            accum_adaptive_signed_erase_diag = {
                key: 0.0 for key in ADAPTIVE_SIGNED_ERASE_TRAIN_KEYS
            }
            accum_bi_axis_value_decay_diag = {
                key: 0.0 for key in BI_AXIS_VALUE_DECAY_TRAIN_KEYS
            }
            accum_raven_routed_gdn_diag = {
                key: 0.0 for key in RAVEN_ROUTED_GDN_TRAIN_KEYS
            }
            accum_paired_address_bank_diag = {
                key: 0.0 for key in PAIRED_ADDRESS_BANK_TRAIN_KEYS
            }
            accum_coupled_address_rows_diag = {
                key: 0.0 for key in COUPLED_ADDRESS_ROWS_TRAIN_KEYS
            }
            accum_interleaved_write_diag = {
                key: 0.0 for key in INTERLEAVED_WRITE_TRAIN_KEYS
            }
            accum_state_expert_diag = {
                key: 0.0 for key in STATE_EXPERT_TRAIN_KEYS
            }
            accum_address_diag = {
                key: 0.0 for key in ADDRESS_TRAIN_KEYS
            }
            opening_gradient_accum: List[Optional[torch.Tensor]] = [
                None for _name, _parameter in gate_parameters
            ]
            for _accum_idx in range(accum_count):
                inputs, labels, clue_mask = make_train_batch(
                    args,
                    official_train,
                    args.batch,
                    holes_min,
                    holes_max,
                    rng,
                    device=device,
                )
                cell_order = training_cell_order(
                    mode=args.cell_order_train,
                    seed=args.seed,
                    global_step=global_step,
                    accumulation_index=_accum_idx,
                    device=device,
                )
                with forward_autocast(args.forward_dtype, device):
                    loop_logits, _fs_trace = model.forward_trace(
                        inputs,
                        loops=args.max_loops,
                        noise_scale=args.noise_scale,
                        feature_buffer=feature_buffer,
                        update_feature_buffer=model.training,
                        feature_buffer_add=args.feature_buffer_add,
                        cell_order=cell_order,
                    )
                loop_losses = [
                    loss_from_logits(logits, labels, clue_mask, blank_weight=args.blank_loss_weight)
                    for logits in loop_logits
                ]
                ce_loss = loop_losses[-1]
                supervised_loss, loop_weights = weighted_loop_loss(loop_losses, args)
                scratch_gauss_terms = [
                    trace["scratch_gauss_loss"].to(dtype=supervised_loss.dtype)
                    for trace in _fs_trace
                    if "scratch_gauss_loss" in trace
                ]
                scratch_gauss_loss = (
                    torch.stack(scratch_gauss_terms).mean()
                    if scratch_gauss_terms
                    else supervised_loss.new_zeros(())
                )
                if float(args.exact_margin_weight) > 0 and global_step >= int(args.exact_margin_start_step):
                    exact_margin_loss, exact_margin_diag = exact_margin_loss_from_logits(
                        loop_logits[-1],
                        labels,
                        clue_mask,
                        tau=args.exact_margin_tau,
                        target=args.exact_margin_target,
                    )
                else:
                    exact_margin_loss = supervised_loss.new_zeros(())
                    exact_margin_diag = {
                        "exact_margin_loss": exact_margin_loss.detach(),
                        "exact_margin_softmin": exact_margin_loss.detach(),
                        "exact_margin_hardmin": exact_margin_loss.detach(),
                        "exact_margin_mean": exact_margin_loss.detach(),
                    }
                loss = (
                    supervised_loss
                    + float(args.scratch_gauss_weight) * scratch_gauss_loss
                    + float(args.exact_margin_weight) * exact_margin_loss
                )
                if args.future_seed_gradient_mode == "opening_projection":
                    opening_gradients = torch.autograd.grad(
                        loop_losses[0] / float(accum_count),
                        [parameter for _name, parameter in gate_parameters],
                        retain_graph=True,
                        create_graph=False,
                        allow_unused=True,
                    )
                    for gradient_idx, gradient in enumerate(opening_gradients):
                        if gradient is None:
                            continue
                        detached = gradient.detach().float()
                        if opening_gradient_accum[gradient_idx] is None:
                            opening_gradient_accum[gradient_idx] = detached.clone()
                        else:
                            opening_gradient_accum[gradient_idx].add_(detached)
                (loss / float(accum_count)).backward()
                accum_ce_loss += float(ce_loss.detach().cpu())
                accum_total_loss += float(loss.detach().cpu())
                accum_loop1_loss += float(loop_losses[0].detach().cpu())
                accum_loop_last_loss += float(loop_losses[-1].detach().cpu())
                accum_exact_margin_loss += float(exact_margin_diag["exact_margin_loss"].cpu())
                accum_exact_margin_softmin += float(exact_margin_diag["exact_margin_softmin"].cpu())
                accum_exact_margin_hardmin += float(exact_margin_diag["exact_margin_hardmin"].cpu())
                accum_exact_margin_mean += float(exact_margin_diag["exact_margin_mean"].cpu())
                accum_scratch_gauss_loss += float(scratch_gauss_loss.detach().cpu())
                if _fs_trace:
                    trace_last = _fs_trace[-1]
                    accum_scratch_gate += float(
                        trace_last.get("scratch_gate_mean", ce_loss.new_zeros(())).detach().cpu()
                    )
                    accum_scratch_decay += float(
                        trace_last.get("scratch_decay_mean", ce_loss.new_zeros(())).detach().cpu()
                    )
                    accum_scratch_delta += float(
                        trace_last.get("scratch_delta_norm", ce_loss.new_zeros(())).detach().cpu()
                    )
                    accum_scratch_residual += float(
                        trace_last.get("scratch_residual_norm", ce_loss.new_zeros(())).detach().cpu()
                    )
                    accum_scratch_proj_var += float(
                        trace_last.get("scratch_proj_var_mean", ce_loss.new_zeros(())).detach().cpu()
                    )
                    accum_scratch_proj_rank += float(
                        trace_last.get("scratch_proj_var_rank", ce_loss.new_zeros(())).detach().cpu()
                    )
                    accum_loop_update_gate_l += float(
                        trace_last.get("loop_update_gate_l", ce_loss.new_zeros(())).detach().cpu()
                    )
                    accum_loop_update_gate_h += float(
                        trace_last.get("loop_update_gate_h", ce_loss.new_zeros(())).detach().cpu()
                    )
                    accum_loop_feedback_next_norm += float(
                        trace_last.get("loop_feedback_next_norm", ce_loss.new_zeros(())).detach().cpu()
                    )
                    accum_loop_feedback_corrupt_frac += float(
                        trace_last.get("loop_feedback_corrupt_frac", ce_loss.new_zeros(())).detach().cpu()
                    )
                    accum_loop_feedback_corrupt_confidence += float(
                        trace_last.get("loop_feedback_corrupt_confidence", ce_loss.new_zeros(())).detach().cpu()
                    )
                    accum_hidden_agg_noise_norm += float(
                        trace_last.get("hidden_agg_noise_norm", ce_loss.new_zeros(())).detach().cpu()
                    )
                    accum_hidden_agg_noise_raw_norm += float(
                        trace_last.get("hidden_agg_noise_raw_norm", ce_loss.new_zeros(())).detach().cpu()
                    )
                    accum_hidden_agg_noise_entropy += float(
                        trace_last.get("hidden_agg_noise_entropy", ce_loss.new_zeros(())).detach().cpu()
                    )
                    accum_hidden_agg_noise_max_weight += float(
                        trace_last.get("hidden_agg_noise_max_weight", ce_loss.new_zeros(())).detach().cpu()
                    )
                    accum_hidden_agg_noise_clip_frac += float(
                        trace_last.get("hidden_agg_noise_clip_frac", ce_loss.new_zeros(())).detach().cpu()
                    )
                    accum_gain_budget_clipped_frac += float(
                        trace_last.get(
                            "gdn2_gain_budget_clipped_frac",
                            ce_loss.new_zeros(()),
                        )
                        .detach()
                        .cpu()
                    )
                    accum_gain_budget_infeasible_frac += float(
                        trace_last.get(
                            "gdn2_gain_budget_infeasible_frac",
                            ce_loss.new_zeros(()),
                        )
                        .detach()
                        .cpu()
                    )
                    accum_gain_budget_numerical_endpoint_frac += float(
                        trace_last.get(
                            "gdn2_gain_budget_numerical_endpoint_frac",
                            ce_loss.new_zeros(()),
                        )
                        .detach()
                        .cpu()
                    )
                    accum_gain_budget_step_bound_max = max(
                        accum_gain_budget_step_bound_max,
                        float(
                            trace_last.get(
                                "gdn2_gain_budget_effective_step_bound_max",
                                ce_loss.new_zeros(()),
                            )
                            .detach()
                            .cpu()
                        ),
                    )
                    accum_gain_budget_delta_error_max = max(
                        accum_gain_budget_delta_error_max,
                        float(
                            trace_last.get(
                                "gdn2_gain_budget_delta_error_max",
                                ce_loss.new_zeros(()),
                            )
                            .detach()
                            .cpu()
                        ),
                    )
                    accum_gain_budget_gate_relative_change += float(
                        trace_last.get(
                            "gdn2_gain_budget_gate_relative_change",
                            ce_loss.new_zeros(()),
                        )
                        .detach()
                        .cpu()
                    )
                    for key in FAST_SLOW_TRAIN_KEYS:
                        accum_fast_slow_diag[key] += float(
                            trace_last.get(
                                key,
                                ce_loss.new_zeros(()),
                            )
                            .detach()
                            .cpu()
                        )
                    for key in PRECONDITION_TRAIN_KEYS:
                        accum_precondition_diag[key] += float(
                            trace_last.get(
                                key,
                                ce_loss.new_zeros(()),
                            )
                            .detach()
                            .cpu()
                        )
                    for key in COHERENT_DELTA_TRAIN_KEYS:
                        accum_coherent_delta_diag[key] += float(
                            trace_last.get(
                                key,
                                ce_loss.new_tensor(
                                    1.0
                                    if key == "gdn2_coherent_delta_gap_ratio"
                                    else 0.0
                                ),
                            )
                            .detach()
                            .cpu()
                        )
                    for key in STATE_FEEDBACK_TRAIN_KEYS:
                        accum_state_feedback_diag[key] += float(
                            trace_last.get(
                                key,
                                ce_loss.new_zeros(()),
                            )
                            .detach()
                            .cpu()
                        )
                    for key in TERMINAL_CONSOLIDATION_TRAIN_KEYS:
                        accum_terminal_consolidation_diag[key] += float(
                            trace_last.get(
                                key,
                                ce_loss.new_zeros(()),
                            )
                            .detach()
                            .cpu()
                        )
                    for key in ORTHOGONAL_CHUNK_STATE_TRAIN_KEYS:
                        accum_orthogonal_chunk_state_diag[key] += float(
                            trace_last.get(
                                key,
                                ce_loss.new_tensor(
                                    1.0
                                    if key
                                    == "gdn3_orthogonal_chunk_state_boundary_norm_ratio_mean"
                                    else 0.0
                                ),
                            )
                            .detach()
                            .cpu()
                        )
                    for key in ORTHOGONAL_HEAD_WRITE_TRAIN_KEYS:
                        accum_orthogonal_head_write_diag[key] += float(
                            trace_last.get(
                                key,
                                ce_loss.new_tensor(
                                    1.0
                                    if key
                                    in {
                                        "gdn3_orthogonal_head_write_fp32_norm_ratio_mean",
                                        "gdn3_orthogonal_head_write_storage_norm_ratio_mean",
                                    }
                                    else 0.0
                                ),
                            )
                            .detach()
                            .cpu()
                        )
                    for key in ADAPTIVE_SIGNED_ERASE_TRAIN_KEYS:
                        accum_adaptive_signed_erase_diag[key] += float(
                            trace_last.get(
                                key,
                                ce_loss.new_zeros(()),
                            )
                            .detach()
                            .cpu()
                        )
                    for key in BI_AXIS_VALUE_DECAY_TRAIN_KEYS:
                        accum_bi_axis_value_decay_diag[key] += float(
                            trace_last.get(
                                key,
                                ce_loss.new_tensor(
                                    1.0
                                    if key
                                    in {
                                        "gdn3_bi_axis_value_decay_cumulative_scale_min",
                                        "gdn3_bi_axis_value_decay_cumulative_scale_mean",
                                        "gdn3_bi_axis_value_decay_cumulative_scale_max",
                                        "gdn3_bi_axis_value_decay_inverse_scale_max",
                                    }
                                    else 0.0
                                ),
                            )
                            .detach()
                            .cpu()
                        )
                    for key in RAVEN_ROUTED_GDN_TRAIN_KEYS:
                        accum_raven_routed_gdn_diag[key] += float(
                            trace_last.get(
                                key,
                                ce_loss.new_tensor(
                                    1.0
                                    if key
                                    in {
                                        "gdn3_raven_routed_allocation_entropy_normalized",
                                        "gdn3_raven_routed_allocation_min",
                                        "gdn3_raven_routed_allocation_max",
                                    }
                                    else 0.0
                                ),
                            )
                            .detach()
                            .cpu()
                        )
                    for key in PAIRED_ADDRESS_BANK_TRAIN_KEYS:
                        accum_paired_address_bank_diag[key] += float(
                            trace_last.get(
                                key,
                                ce_loss.new_zeros(()),
                            )
                            .detach()
                            .cpu()
                        )
                    for key in COUPLED_ADDRESS_ROWS_TRAIN_KEYS:
                        accum_coupled_address_rows_diag[key] += float(
                            trace_last.get(
                                key,
                                ce_loss.new_zeros(()),
                            )
                            .detach()
                            .cpu()
                        )
                    for key in INTERLEAVED_WRITE_TRAIN_KEYS:
                        accum_interleaved_write_diag[key] += float(
                            trace_last.get(
                                key,
                                ce_loss.new_zeros(()),
                            )
                            .detach()
                            .cpu()
                        )
                    for key in STATE_EXPERT_TRAIN_KEYS:
                        accum_state_expert_diag[key] += float(
                            trace_last.get(
                                key,
                                ce_loss.new_zeros(()),
                            )
                            .detach()
                            .cpu()
                        )
                    for key in ADDRESS_TRAIN_KEYS:
                        accum_address_diag[key] += float(
                            trace_last.get(
                                key,
                                ce_loss.new_zeros(()),
                            )
                            .detach()
                            .cpu()
                        )
            canonical_global_grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), 1.0
            )
            if args.future_seed_gradient_mode == "opening_projection":
                canonical_grad_norm_value = float(
                    canonical_global_grad_norm.detach().float().cpu().item()
                )
                if not math.isfinite(canonical_grad_norm_value):
                    raise RuntimeError("Canonical global gradient norm is non-finite")
                canonical_clip_coefficient = min(
                    1.0,
                    1.0 / (canonical_grad_norm_value + 1e-6),
                )
                clipped_opening_gradients = [
                    (
                        None
                        if gradient is None
                        else gradient * canonical_clip_coefficient
                    )
                    for gradient in opening_gradient_accum
                ]
                last_future_seed_projection_diag = (
                    apply_future_seed_opening_projection(
                        gate_parameters,
                        clipped_opening_gradients,
                    )
                )
                last_future_seed_projection_diag.update(
                    {
                        "canonical_global_grad_norm": canonical_grad_norm_value,
                        "canonical_clip_coefficient": canonical_clip_coefficient,
                    }
                )
                expected_final_grad_norm = min(1.0, canonical_grad_norm_value)
                final_global_grad_norm = float(
                    current_global_gradient_norm(model.parameters())
                    .detach()
                    .cpu()
                    .item()
                )
                final_global_grad_norm_relative_error = abs(
                    final_global_grad_norm - expected_final_grad_norm
                ) / max(expected_final_grad_norm, 1e-30)
                last_future_seed_projection_diag.update(
                    {
                        "final_global_grad_norm": final_global_grad_norm,
                        "final_global_grad_norm_relative_error": (
                            final_global_grad_norm_relative_error
                        ),
                    }
                )
                if (
                    last_future_seed_projection_diag["active_tensor_count"] != 11.0
                    or last_future_seed_projection_diag["active_parameter_count"] != 88.0
                    or last_future_seed_projection_diag["inactive_tensor_count"] != 1.0
                ):
                    raise RuntimeError(
                        "FutureSeed opening projection expected exactly 11 active "
                        "receiving gates (88 parameters) and one inactive gate"
                    )
                if last_future_seed_projection_diag["norm_relative_error"] >= 1e-5:
                    raise RuntimeError(
                        "FutureSeed opening projection violated its norm contract"
                    )
                if final_global_grad_norm_relative_error >= 2e-5:
                    raise RuntimeError(
                        "FutureSeed opening projection changed the canonical global "
                        "gradient norm"
                    )
                if (
                    last_future_seed_projection_diag["active"]
                    and last_future_seed_projection_diag[
                        "post_opening_continuation_dot"
                    ]
                    < -1e-6
                ):
                    raise RuntimeError(
                        "FutureSeed opening projection violated its continuation "
                        "half-space contract"
                    )
                update_future_seed_projection_stats(
                    future_seed_projection_stats,
                    last_future_seed_projection_diag,
                )
            opt.step()
            last_ce_loss = accum_ce_loss / float(accum_count)
            last_total_loss = accum_total_loss / float(accum_count)
            last_loop1_loss = accum_loop1_loss / float(accum_count)
            last_loop_last_loss = accum_loop_last_loss / float(accum_count)
            last_loop_weights = [float(x) for x in loop_weights.detach().cpu().tolist()]
            last_exact_margin_loss = accum_exact_margin_loss / float(accum_count)
            last_exact_margin_softmin = accum_exact_margin_softmin / float(accum_count)
            last_exact_margin_hardmin = accum_exact_margin_hardmin / float(accum_count)
            last_exact_margin_mean = accum_exact_margin_mean / float(accum_count)
            last_scratch_gauss_loss = accum_scratch_gauss_loss / float(accum_count)
            last_scratch_gate = accum_scratch_gate / float(accum_count)
            last_scratch_decay = accum_scratch_decay / float(accum_count)
            last_scratch_delta = accum_scratch_delta / float(accum_count)
            last_scratch_residual = accum_scratch_residual / float(accum_count)
            last_scratch_proj_var = accum_scratch_proj_var / float(accum_count)
            last_scratch_proj_rank = accum_scratch_proj_rank / float(accum_count)
            last_loop_update_gate_l = accum_loop_update_gate_l / float(accum_count)
            last_loop_update_gate_h = accum_loop_update_gate_h / float(accum_count)
            last_loop_feedback_next_norm = accum_loop_feedback_next_norm / float(accum_count)
            last_loop_feedback_corrupt_frac = accum_loop_feedback_corrupt_frac / float(accum_count)
            last_loop_feedback_corrupt_confidence = accum_loop_feedback_corrupt_confidence / float(accum_count)
            last_hidden_agg_noise_norm = accum_hidden_agg_noise_norm / float(accum_count)
            last_hidden_agg_noise_raw_norm = accum_hidden_agg_noise_raw_norm / float(accum_count)
            last_hidden_agg_noise_entropy = accum_hidden_agg_noise_entropy / float(accum_count)
            last_hidden_agg_noise_max_weight = accum_hidden_agg_noise_max_weight / float(accum_count)
            last_hidden_agg_noise_clip_frac = accum_hidden_agg_noise_clip_frac / float(accum_count)
            last_gain_budget_clipped_frac = (
                accum_gain_budget_clipped_frac / float(accum_count)
            )
            last_gain_budget_infeasible_frac = (
                accum_gain_budget_infeasible_frac / float(accum_count)
            )
            last_gain_budget_numerical_endpoint_frac = (
                accum_gain_budget_numerical_endpoint_frac / float(accum_count)
            )
            last_gain_budget_step_bound_max = (
                accum_gain_budget_step_bound_max
            )
            last_gain_budget_delta_error_max = (
                accum_gain_budget_delta_error_max
            )
            last_gain_budget_gate_relative_change = (
                accum_gain_budget_gate_relative_change / float(accum_count)
            )
            last_fast_slow_diag = {
                key: value / float(accum_count)
                for key, value in accum_fast_slow_diag.items()
            }
            last_precondition_diag = {
                key: value / float(accum_count)
                for key, value in accum_precondition_diag.items()
            }
            last_coherent_delta_diag = {
                key: value / float(accum_count)
                for key, value in accum_coherent_delta_diag.items()
            }
            last_state_feedback_diag = {
                key: value / float(accum_count)
                for key, value in accum_state_feedback_diag.items()
            }
            last_terminal_consolidation_diag = {
                key: value / float(accum_count)
                for key, value in accum_terminal_consolidation_diag.items()
            }
            last_orthogonal_chunk_state_diag = {
                key: value / float(accum_count)
                for key, value in accum_orthogonal_chunk_state_diag.items()
            }
            last_orthogonal_head_write_diag = {
                key: value / float(accum_count)
                for key, value in accum_orthogonal_head_write_diag.items()
            }
            last_adaptive_signed_erase_diag = {
                key: value / float(accum_count)
                for key, value in accum_adaptive_signed_erase_diag.items()
            }
            last_bi_axis_value_decay_diag = {
                key: value / float(accum_count)
                for key, value in accum_bi_axis_value_decay_diag.items()
            }
            last_raven_routed_gdn_diag = {
                key: value / float(accum_count)
                for key, value in accum_raven_routed_gdn_diag.items()
            }
            last_paired_address_bank_diag = {
                key: value / float(accum_count)
                for key, value in accum_paired_address_bank_diag.items()
            }
            last_coupled_address_rows_diag = {
                key: value / float(accum_count)
                for key, value in accum_coupled_address_rows_diag.items()
            }
            last_interleaved_write_diag = {
                key: value / float(accum_count)
                for key, value in accum_interleaved_write_diag.items()
            }
            last_state_expert_diag = {
                key: value / float(accum_count)
                for key, value in accum_state_expert_diag.items()
            }
            last_address_diag = {
                key: value / float(accum_count)
                for key, value in accum_address_diag.items()
            }
            if args.log_every and global_step % args.log_every == 0:
                projection_summary = summarize_future_seed_projection_stats(
                    future_seed_projection_stats
                )
                print(
                    f"[future_seed_loop stage={stage_idx}:{holes_min}-{holes_max}] "
                    f"step={global_step:04d} ce={last_ce_loss:.4f} total={last_total_loss:.4f} "
                    f"loop1={last_loop1_loss:.4f} loop_last={last_loop_last_loss:.4f} "
                    f"exact_margin={last_exact_margin_loss:.4f} "
                    f"softmin={last_exact_margin_softmin:.3f} "
                    f"upd_l={last_loop_update_gate_l:.3f} upd_h={last_loop_update_gate_h:.3f} "
                    f"fb_next={last_loop_feedback_next_norm:.3f} "
                    f"fb_corrupt={last_loop_feedback_corrupt_frac:.3f} "
                    f"fb_conf={last_loop_feedback_corrupt_confidence:.3f} "
                    f"hagg_norm={last_hidden_agg_noise_norm:.3f} "
                    f"hagg_raw={last_hidden_agg_noise_raw_norm:.3f} "
                    f"hagg_ent={last_hidden_agg_noise_entropy:.3f} "
                    f"hagg_maxw={last_hidden_agg_noise_max_weight:.3f} "
                    f"hagg_clip={last_hidden_agg_noise_clip_frac:.3f} "
                    f"gb_clip={last_gain_budget_clipped_frac:.4f} "
                    f"gb_infeas={last_gain_budget_infeasible_frac:.4f} "
                    f"gb_endpoint={last_gain_budget_numerical_endpoint_frac:.4f} "
                    f"gb_bound={last_gain_budget_step_bound_max:.6f} "
                    f"gb_delta={last_gain_budget_delta_error_max:.2e} "
                    f"gb_change={last_gain_budget_gate_relative_change:.4f} "
                    f"fsproj_rate={projection_summary['activation_rate']:.4f} "
                    f"fsproj_cos={projection_summary['opening_continuation_cosine_mean']:.4f} "
                    f"fsproj_remove={projection_summary['active_removed_opening_fraction_mean']:.4f} "
                    f"fsproj_corr={projection_summary['active_relative_correction_mean']:.4f} "
                    f"fsproj_norm={projection_summary['norm_relative_error_max']:.2e} "
                    f"slow_rho={last_fast_slow_diag['gdn2_fast_slow_rho_mean']:.4f} "
                    f"slow_lag={last_fast_slow_diag['gdn2_fast_slow_lag_mass']:.4f} "
                    f"slow_tv={last_fast_slow_diag['gdn2_fast_slow_tv_ratio']:.4f} "
                    f"slow_change={last_fast_slow_diag['gdn2_fast_slow_relative_change']:.4f} "
                    f"pre_m={last_precondition_diag['gdn2_precondition_multiplier_mean']:.4f}/"
                    f"{last_precondition_diag['gdn2_precondition_multiplier_std']:.4f} "
                    f"pre_seed={last_precondition_diag['gdn2_precondition_seed_precision_mean']:.4f} "
                    f"pre_write={last_precondition_diag['gdn2_precondition_write_relative_change']:.4f} "
                    f"pre_erase={last_precondition_diag['gdn2_precondition_erase_error_max']:.2e} "
                    f"coh_mix={last_coherent_delta_diag['gdn2_coherent_delta_mix_mean']:.4f}/"
                    f"{last_coherent_delta_diag['gdn2_coherent_delta_mix_abs']:.4f} "
                    f"coh_gap={last_coherent_delta_diag['gdn2_coherent_delta_gap_ratio']:.4f} "
                    f"coh_b={last_coherent_delta_diag['gdn2_coherent_delta_b_relative_change']:.4f} "
                    f"coh_w={last_coherent_delta_diag['gdn2_coherent_delta_w_relative_change']:.4f} "
                    f"state_fb={last_state_feedback_diag['gdn3_state_feedback_read_rms']:.4f}/"
                    f"{last_state_feedback_diag['gdn3_state_feedback_residual_relative_rms']:.4f} "
                    f"state_fb_kv={last_state_feedback_diag['gdn3_state_feedback_k_relative_change']:.4f}/"
                    f"{last_state_feedback_diag['gdn3_state_feedback_v_relative_change']:.4f} "
                    f"state_fb_bw={last_state_feedback_diag['gdn3_state_feedback_b_relative_change']:.4f}/"
                    f"{last_state_feedback_diag['gdn3_state_feedback_w_relative_change']:.4f} "
                    f"term_cons_k={last_terminal_consolidation_diag['gdn3_terminal_consolidation_k_relative_rms']:.4f}/"
                    f"{last_terminal_consolidation_diag['gdn3_terminal_consolidation_k_batch_std']:.4f}/"
                    f"{last_terminal_consolidation_diag['gdn3_terminal_consolidation_k_token_std']:.4f} "
                    f"term_cons_state={last_terminal_consolidation_diag['gdn3_terminal_consolidation_state_residual_relative_rms']:.4f}/"
                    f"{last_terminal_consolidation_diag['gdn3_terminal_consolidation_state_residual_batch_std']:.4f} "
                    f"orth_chunk_angle={last_orthogonal_chunk_state_diag['gdn3_orthogonal_chunk_state_angle_abs']:.4f}/"
                    f"{last_orthogonal_chunk_state_diag['gdn3_orthogonal_chunk_state_angle_batch_std']:.4f} "
                    f"orth_chunk_state={last_orthogonal_chunk_state_diag['gdn3_orthogonal_chunk_state_state_residual_relative_rms']:.4f}/"
                    f"{last_orthogonal_chunk_state_diag['gdn3_orthogonal_chunk_state_read_residual_relative_rms']:.4f} "
                    f"orth_chunk_norm={last_orthogonal_chunk_state_diag['gdn3_orthogonal_chunk_state_boundary_norm_ratio_mean']:.6f}/"
                    f"{last_orthogonal_chunk_state_diag['gdn3_orthogonal_chunk_state_boundary_norm_ratio_max_error']:.2e} "
                    f"orth_head_angle={last_orthogonal_head_write_diag['gdn3_orthogonal_head_write_angle_abs']:.4f}/"
                    f"{last_orthogonal_head_write_diag['gdn3_orthogonal_head_write_angle_batch_std']:.4f} "
                    f"orth_head_v={last_orthogonal_head_write_diag['gdn3_orthogonal_head_write_v_residual_relative_rms']:.4f}/"
                    f"{last_orthogonal_head_write_diag['gdn3_orthogonal_head_write_v_residual_batch_std']:.4f} "
                    f"orth_head_norm={last_orthogonal_head_write_diag['gdn3_orthogonal_head_write_fp32_norm_ratio_mean']:.6f}/"
                    f"{last_orthogonal_head_write_diag['gdn3_orthogonal_head_write_storage_norm_ratio_max_error']:.2e} "
                    f"signed_erase={last_adaptive_signed_erase_diag['gdn3_adaptive_signed_erase_residual_abs']:.4f}/"
                    f"{last_adaptive_signed_erase_diag['gdn3_adaptive_signed_erase_residual_batch_std']:.4f}/"
                    f"{last_adaptive_signed_erase_diag['gdn3_adaptive_signed_erase_residual_token_std']:.4f} "
                    f"signed_erase_b={last_adaptive_signed_erase_diag['gdn3_adaptive_signed_erase_effective_mean']:.4f}/"
                    f"{last_adaptive_signed_erase_diag['gdn3_adaptive_signed_erase_effective_max']:.4f}/"
                    f"{last_adaptive_signed_erase_diag['gdn3_adaptive_signed_erase_above_one_frac']:.4f} "
                    f"signed_erase_state={last_adaptive_signed_erase_diag['gdn3_adaptive_signed_erase_terminal_rms']:.4f} "
                    f"bi_axis_potential={last_bi_axis_value_decay_diag['gdn3_bi_axis_value_decay_potential_abs']:.4f}/"
                    f"{last_bi_axis_value_decay_diag['gdn3_bi_axis_value_decay_common_k_log_decay_abs']:.4f} "
                    f"bi_axis_decay={last_bi_axis_value_decay_diag['gdn3_bi_axis_value_decay_log_decay_abs']:.4f}/"
                    f"{last_bi_axis_value_decay_diag['gdn3_bi_axis_value_decay_batch_std']:.4f}/"
                    f"{last_bi_axis_value_decay_diag['gdn3_bi_axis_value_decay_token_std']:.4f} "
                    f"bi_axis_scale={last_bi_axis_value_decay_diag['gdn3_bi_axis_value_decay_cumulative_scale_min']:.4f}/"
                    f"{last_bi_axis_value_decay_diag['gdn3_bi_axis_value_decay_cumulative_scale_max']:.4f}/"
                    f"{last_bi_axis_value_decay_diag['gdn3_bi_axis_value_decay_inverse_scale_max']:.4f} "
                    f"bi_axis_frame={last_bi_axis_value_decay_diag['gdn3_bi_axis_value_decay_write_frame_relative_rms']:.4f}/"
                    f"{last_bi_axis_value_decay_diag['gdn3_bi_axis_value_decay_terminal_rms']:.4f} "
                    f"raven_alloc={last_raven_routed_gdn_diag['gdn3_raven_routed_allocation_abs_from_one']:.4f}/"
                    f"{last_raven_routed_gdn_diag['gdn3_raven_routed_allocation_slot_std']:.4f}/"
                    f"{last_raven_routed_gdn_diag['gdn3_raven_routed_allocation_batch_std']:.4f}/"
                    f"{last_raven_routed_gdn_diag['gdn3_raven_routed_allocation_token_std']:.4f} "
                    f"raven_entropy={last_raven_routed_gdn_diag['gdn3_raven_routed_allocation_entropy_normalized']:.4f}/"
                    f"{last_raven_routed_gdn_diag['gdn3_raven_routed_allocation_min']:.4f}/"
                    f"{last_raven_routed_gdn_diag['gdn3_raven_routed_allocation_max']:.4f} "
                    f"raven_kg={last_raven_routed_gdn_diag['gdn3_raven_routed_k_relative_change']:.4f}/"
                    f"{last_raven_routed_gdn_diag['gdn3_raven_routed_g_relative_change']:.4f} "
                    f"raven_state={last_raven_routed_gdn_diag['gdn3_raven_routed_terminal_rms']:.4f} "
                    f"paired_bank_gate={last_paired_address_bank_diag['gdn3_paired_address_bank_read_gate_abs']:.4f} "
                    f"paired_bank_qk={last_paired_address_bank_diag['gdn3_paired_address_bank_q_residual_relative_rms']:.4f}/"
                    f"{last_paired_address_bank_diag['gdn3_paired_address_bank_k_residual_relative_rms']:.4f} "
                    f"paired_bank_state={last_paired_address_bank_diag['gdn3_paired_address_bank_state_residual_relative_rms']:.4f}/"
                    f"{last_paired_address_bank_diag['gdn3_paired_address_bank_terminal_rms']:.4f} "
                    f"coupled_rows_k={last_coupled_address_rows_diag['gdn3_coupled_address_rows_k_residual_relative_rms']:.4f}/"
                    f"{last_coupled_address_rows_diag['gdn3_coupled_address_rows_k_residual_batch_std']:.4f} "
                    f"coupled_rows_state={last_coupled_address_rows_diag['gdn3_coupled_address_rows_extra_state_relative_rms']:.4f}/"
                    f"{last_coupled_address_rows_diag['gdn3_coupled_address_rows_terminal_rms']:.4f} "
                    f"interleave_kv={last_interleaved_write_diag['gdn3_interleaved_write_k_residual_relative_rms']:.4f}/"
                    f"{last_interleaved_write_diag['gdn3_interleaved_write_v_relative_rms']:.4f} "
                    f"interleave_write={last_interleaved_write_diag['gdn3_interleaved_write_state_write_relative_rms']:.4f}/"
                    f"{last_interleaved_write_diag['gdn3_interleaved_write_terminal_rms']:.4f} "
                    f"expert_resid={last_state_expert_diag['gdn3_state_expert_residual_relative_rms']:.4f}/"
                    f"{last_state_expert_diag['gdn3_state_expert_residual_batch_std']:.4f} "
                    f"expert_state={last_state_expert_diag['gdn3_state_expert_terminal_rms']:.4f}/"
                    f"{last_state_expert_diag['gdn3_state_expert_seed_rms']:.4f} "
                    f"expert_addr={last_state_expert_diag['gdn3_state_expert_address_contrast']:.4f} "
                    f"raven_write_v={last_state_expert_diag['gdn3_raven_write_control_v_residual_relative_rms']:.4f}/"
                    f"{last_state_expert_diag['gdn3_raven_write_control_v_residual_relative_rms_min']:.4f} "
                    f"raven_write_slot={last_state_expert_diag['gdn3_raven_write_control_slot_entropy_normalized_min']:.4f}/"
                    f"{last_state_expert_diag['gdn3_raven_write_control_slot_max_mass_share_max']:.4f} "
                    f"addr_scale={last_address_diag['gdn2_address_rotation_scale_abs']:.4f} "
                    f"addr_phase={last_address_diag['gdn2_address_phase_abs']:.4f} "
                    f"addr_qchg={last_address_diag['gdn2_address_q_relative_change']:.4f} "
                    f"addr_kchg={last_address_diag['gdn2_address_k_relative_change']:.4f} "
                    f"addr_wrms={last_address_diag['gdn2_address_phase_weight_rms']:.4f} "
                    f"addr_tstd={last_address_diag['gdn2_address_phase_token_std']:.4f} "
                    f"addr_pstd={last_address_diag['gdn2_address_phase_plane_std']:.4f} "
                    f"addr_rrms={last_address_diag['gdn2_address_residual_rms']:.4f} "
                    f"addr_rstd={last_address_diag['gdn2_address_residual_token_std']:.4f} "
                    f"addr_rq={last_address_diag['gdn2_address_residual_q_ratio']:.4f} "
                    f"addr_rk={last_address_diag['gdn2_address_residual_k_ratio']:.4f} "
                    f"addr_qwrms={last_address_diag['gdn2_address_q_residual_weight_rms']:.4f} "
                    f"addr_kwrms={last_address_diag['gdn2_address_k_residual_weight_rms']:.4f} "
                    f"carrier={last_address_diag['gdn2_address_carrier_mean']:.4f}/"
                    f"{last_address_diag['gdn2_address_carrier_std']:.4f}/"
                    f"{last_address_diag['gdn2_address_carrier_token_std']:.4f} "
                    f"carrier_min={last_address_diag['gdn2_address_carrier_min']:.4f} "
                    f"carrier_r={last_address_diag['gdn2_address_carrier_write_norm_ratio']:.4f} "
                    f"addr_state={last_address_diag['gdn2_address_terminal_state_rms']:.4f} "
                    f"scratch_delta={last_scratch_delta:.3f} scratch_gauss={last_scratch_gauss_loss:.4f}",
                    flush=True,
                )
            if global_step in checkpoint_step_set:
                model.eval()
                checkpoint = {
                    "step": global_step,
                    "stage": stage_idx,
                    "holes_min": holes_min,
                    "holes_max": holes_max,
                    "elapsed_sec": time.time() - t0,
                    "train": {
                        "ce_loss": last_ce_loss,
                        "total_loss": last_total_loss,
                        "loop1_loss": last_loop1_loss,
                        "loop_last_loss": last_loop_last_loss,
                        "exact_margin_loss": last_exact_margin_loss,
                        "exact_margin_softmin": last_exact_margin_softmin,
                        "exact_margin_hardmin": last_exact_margin_hardmin,
                        "exact_margin_mean": last_exact_margin_mean,
                        "scratch_gauss_loss": last_scratch_gauss_loss,
                        "scratch_gate": last_scratch_gate,
                        "scratch_decay": last_scratch_decay,
                        "scratch_delta": last_scratch_delta,
                        "scratch_residual": last_scratch_residual,
                        "scratch_proj_var": last_scratch_proj_var,
                        "scratch_proj_rank": last_scratch_proj_rank,
                        "loop_update_gate_l": last_loop_update_gate_l,
                        "loop_update_gate_h": last_loop_update_gate_h,
                        "loop_feedback_next_norm": last_loop_feedback_next_norm,
                        "loop_feedback_corrupt_frac": last_loop_feedback_corrupt_frac,
                        "loop_feedback_corrupt_confidence": last_loop_feedback_corrupt_confidence,
                        "hidden_agg_noise_norm": last_hidden_agg_noise_norm,
                        "hidden_agg_noise_raw_norm": last_hidden_agg_noise_raw_norm,
                        "hidden_agg_noise_entropy": last_hidden_agg_noise_entropy,
                        "hidden_agg_noise_max_weight": last_hidden_agg_noise_max_weight,
                        "hidden_agg_noise_clip_frac": last_hidden_agg_noise_clip_frac,
                        "gain_budget_clipped_frac": last_gain_budget_clipped_frac,
                        "gain_budget_infeasible_frac": last_gain_budget_infeasible_frac,
                        "gain_budget_numerical_endpoint_frac": last_gain_budget_numerical_endpoint_frac,
                        "gain_budget_step_bound_max": last_gain_budget_step_bound_max,
                        "gain_budget_delta_error_max": last_gain_budget_delta_error_max,
                        "gain_budget_gate_relative_change": last_gain_budget_gate_relative_change,
                        "fast_slow_decay": dict(last_fast_slow_diag),
                        "precondition": dict(last_precondition_diag),
                        "coherent_delta": dict(last_coherent_delta_diag),
                        "state_feedback": dict(last_state_feedback_diag),
                        "terminal_consolidation": dict(
                            last_terminal_consolidation_diag
                        ),
                        "orthogonal_chunk_state": dict(
                            last_orthogonal_chunk_state_diag
                        ),
                        "orthogonal_head_write": dict(
                            last_orthogonal_head_write_diag
                        ),
                        "adaptive_signed_erase": dict(
                            last_adaptive_signed_erase_diag
                        ),
                        "bi_axis_value_decay": dict(
                            last_bi_axis_value_decay_diag
                        ),
                        "raven_routed_gdn": dict(
                            last_raven_routed_gdn_diag
                        ),
                        "paired_address_bank": dict(
                            last_paired_address_bank_diag
                        ),
                        "coupled_address_rows": dict(
                            last_coupled_address_rows_diag
                        ),
                        "interleaved_write": dict(
                            last_interleaved_write_diag
                        ),
                        "state_expert": dict(last_state_expert_diag),
                        "address_operator": dict(last_address_diag),
                        "future_seed_gradient_projection": (
                            summarize_future_seed_projection_stats(
                                future_seed_projection_stats
                            )
                        ),
                        "future_seed_gradient_projection_state": dict(
                            future_seed_projection_stats
                        ),
                    },
                    "eval_by_holes": {},
                }
                for holes, eval_batch in checkpoint_batches.items():
                    clean, _preds = evaluate_model(
                        model,
                        eval_batch,
                        max_loops=args.max_loops,
                        noise_scale=0.0,
                        seed=args.seed + 14000 + global_step + holes,
                        forward_dtype=args.forward_dtype,
                    )
                    checkpoint["eval_by_holes"][f"holes{holes}"] = {
                        "blank_range": [holes, holes],
                        "eval_n": int(eval_batch[0].shape[0]),
                        "eval_clean": clean,
                    }
                checkpoint_key = f"step{global_step}"
                checkpoint_evals[checkpoint_key] = checkpoint
                checkpoint_path = Path(args.out_dir) / f"checkpoint_eval_step{global_step:06d}.json"
                checkpoint_path.write_text(json.dumps(checkpoint, indent=2, sort_keys=True) + "\n", encoding="utf-8")
                target_key = f"holes{args.eval_holes}"
                if target_key in checkpoint["eval_by_holes"]:
                    last_key = f"loop{args.max_loops}"
                    target_metrics = checkpoint["eval_by_holes"][target_key]["eval_clean"][last_key]
                    print(
                        f"[checkpoint_eval step={global_step:04d} {target_key}] "
                        f"loop{args.max_loops} exact={target_metrics['label_exact']:.4f} "
                        f"blank={target_metrics['blank_acc']:.4f}",
                        flush=True,
                    )
                model.train()
                if args.save_train_checkpoint_every >= 0:
                    path = save_training_checkpoint(
                        args=args,
                        model=model,
                        opt=opt,
                        feature_buffer=feature_buffer,
                        rng=rng,
                        device=device,
                        global_step=global_step,
                        stage_idx=stage_idx,
                        holes_min=holes_min,
                        holes_max=holes_max,
                        t0=t0,
                        checkpoint_evals=checkpoint_evals,
                        last_metrics={
                            "ce_loss": last_ce_loss,
                            "total_loss": last_total_loss,
                            "loop1_loss": last_loop1_loss,
                            "loop_last_loss": last_loop_last_loss,
                            "loop_weights": last_loop_weights,
                            "exact_margin_loss": last_exact_margin_loss,
                            "exact_margin_softmin": last_exact_margin_softmin,
                            "exact_margin_hardmin": last_exact_margin_hardmin,
                            "exact_margin_mean": last_exact_margin_mean,
                            "scratch_gauss_loss": last_scratch_gauss_loss,
                            "scratch_gate": last_scratch_gate,
                            "scratch_decay": last_scratch_decay,
                            "scratch_delta": last_scratch_delta,
                            "scratch_residual": last_scratch_residual,
                            "scratch_proj_var": last_scratch_proj_var,
                            "scratch_proj_rank": last_scratch_proj_rank,
                            "loop_update_gate_l": last_loop_update_gate_l,
                            "loop_update_gate_h": last_loop_update_gate_h,
                            "loop_feedback_next_norm": last_loop_feedback_next_norm,
                            "loop_feedback_corrupt_frac": last_loop_feedback_corrupt_frac,
                            "loop_feedback_corrupt_confidence": last_loop_feedback_corrupt_confidence,
                            "hidden_agg_noise_norm": last_hidden_agg_noise_norm,
                            "hidden_agg_noise_raw_norm": last_hidden_agg_noise_raw_norm,
                            "hidden_agg_noise_entropy": last_hidden_agg_noise_entropy,
                            "hidden_agg_noise_max_weight": last_hidden_agg_noise_max_weight,
                            "hidden_agg_noise_clip_frac": last_hidden_agg_noise_clip_frac,
                            "gain_budget_clipped_frac": last_gain_budget_clipped_frac,
                            "gain_budget_infeasible_frac": last_gain_budget_infeasible_frac,
                            "gain_budget_numerical_endpoint_frac": last_gain_budget_numerical_endpoint_frac,
                            "gain_budget_step_bound_max": last_gain_budget_step_bound_max,
                            "gain_budget_delta_error_max": last_gain_budget_delta_error_max,
                            "gain_budget_gate_relative_change": last_gain_budget_gate_relative_change,
                            "fast_slow_decay": dict(last_fast_slow_diag),
                            "precondition": dict(last_precondition_diag),
                            "coherent_delta": dict(last_coherent_delta_diag),
                            "state_feedback": dict(last_state_feedback_diag),
                            "terminal_consolidation": dict(
                                last_terminal_consolidation_diag
                            ),
                            "orthogonal_chunk_state": dict(
                                last_orthogonal_chunk_state_diag
                            ),
                            "orthogonal_head_write": dict(
                                last_orthogonal_head_write_diag
                            ),
                            "adaptive_signed_erase": dict(
                                last_adaptive_signed_erase_diag
                            ),
                            "bi_axis_value_decay": dict(
                                last_bi_axis_value_decay_diag
                            ),
                            "raven_routed_gdn": dict(
                                last_raven_routed_gdn_diag
                            ),
                            "paired_address_bank": dict(
                                last_paired_address_bank_diag
                            ),
                            "coupled_address_rows": dict(
                                last_coupled_address_rows_diag
                            ),
                            "interleaved_write": dict(
                                last_interleaved_write_diag
                            ),
                            "state_expert": dict(last_state_expert_diag),
                            "address_operator": dict(last_address_diag),
                            "future_seed_gradient_projection": (
                                summarize_future_seed_projection_stats(
                                    future_seed_projection_stats
                                )
                            ),
                            "future_seed_gradient_projection_state": dict(
                                future_seed_projection_stats
                            ),
                        },
                        reason="eval_checkpoint",
                    )
                    saved_train_checkpoints.append(str(path))
                    last_saved_step = global_step
                    print(f"[train_checkpoint step={global_step:04d}] path={path}", flush=True)

            if (
                args.save_train_checkpoint_every > 0
                and global_step % args.save_train_checkpoint_every == 0
                and global_step != last_saved_step
            ):
                path = save_training_checkpoint(
                    args=args,
                    model=model,
                    opt=opt,
                    feature_buffer=feature_buffer,
                    rng=rng,
                    device=device,
                    global_step=global_step,
                    stage_idx=stage_idx,
                    holes_min=holes_min,
                    holes_max=holes_max,
                    t0=t0,
                    checkpoint_evals=checkpoint_evals,
                    last_metrics={
                        "ce_loss": last_ce_loss,
                        "total_loss": last_total_loss,
                        "loop1_loss": last_loop1_loss,
                        "loop_last_loss": last_loop_last_loss,
                        "loop_weights": last_loop_weights,
                        "exact_margin_loss": last_exact_margin_loss,
                        "exact_margin_softmin": last_exact_margin_softmin,
                        "exact_margin_hardmin": last_exact_margin_hardmin,
                        "exact_margin_mean": last_exact_margin_mean,
                        "scratch_gauss_loss": last_scratch_gauss_loss,
                        "scratch_gate": last_scratch_gate,
                        "scratch_decay": last_scratch_decay,
                        "scratch_delta": last_scratch_delta,
                        "scratch_residual": last_scratch_residual,
                        "scratch_proj_var": last_scratch_proj_var,
                        "scratch_proj_rank": last_scratch_proj_rank,
                        "loop_update_gate_l": last_loop_update_gate_l,
                        "loop_update_gate_h": last_loop_update_gate_h,
                        "loop_feedback_next_norm": last_loop_feedback_next_norm,
                        "loop_feedback_corrupt_frac": last_loop_feedback_corrupt_frac,
                        "loop_feedback_corrupt_confidence": last_loop_feedback_corrupt_confidence,
                        "hidden_agg_noise_norm": last_hidden_agg_noise_norm,
                        "hidden_agg_noise_raw_norm": last_hidden_agg_noise_raw_norm,
                        "hidden_agg_noise_entropy": last_hidden_agg_noise_entropy,
                        "hidden_agg_noise_max_weight": last_hidden_agg_noise_max_weight,
                        "hidden_agg_noise_clip_frac": last_hidden_agg_noise_clip_frac,
                        "gain_budget_clipped_frac": last_gain_budget_clipped_frac,
                        "gain_budget_infeasible_frac": last_gain_budget_infeasible_frac,
                        "gain_budget_numerical_endpoint_frac": last_gain_budget_numerical_endpoint_frac,
                        "gain_budget_step_bound_max": last_gain_budget_step_bound_max,
                        "gain_budget_delta_error_max": last_gain_budget_delta_error_max,
                        "gain_budget_gate_relative_change": last_gain_budget_gate_relative_change,
                        "fast_slow_decay": dict(last_fast_slow_diag),
                        "precondition": dict(last_precondition_diag),
                        "coherent_delta": dict(last_coherent_delta_diag),
                        "state_feedback": dict(last_state_feedback_diag),
                        "terminal_consolidation": dict(
                            last_terminal_consolidation_diag
                        ),
                        "orthogonal_chunk_state": dict(
                            last_orthogonal_chunk_state_diag
                        ),
                        "orthogonal_head_write": dict(
                            last_orthogonal_head_write_diag
                        ),
                        "adaptive_signed_erase": dict(
                            last_adaptive_signed_erase_diag
                        ),
                        "bi_axis_value_decay": dict(
                            last_bi_axis_value_decay_diag
                        ),
                        "raven_routed_gdn": dict(
                            last_raven_routed_gdn_diag
                        ),
                        "paired_address_bank": dict(
                            last_paired_address_bank_diag
                        ),
                        "coupled_address_rows": dict(
                            last_coupled_address_rows_diag
                        ),
                        "interleaved_write": dict(
                            last_interleaved_write_diag
                        ),
                        "state_expert": dict(last_state_expert_diag),
                        "address_operator": dict(last_address_diag),
                        "future_seed_gradient_projection": (
                            summarize_future_seed_projection_stats(
                                future_seed_projection_stats
                            )
                        ),
                        "future_seed_gradient_projection_state": dict(
                            future_seed_projection_stats
                        ),
                    },
                    reason="periodic",
                )
                saved_train_checkpoints.append(str(path))
                last_saved_step = global_step
                print(f"[train_checkpoint step={global_step:04d}] path={path}", flush=True)
        completed_before_stage = stage_end

    train_stats = {
        "fla_runtime": fla_runtime,
        "shared_shell_init_seed": args.shared_shell_init_seed,
        "post_init_torch_seed": post_init_torch_seed,
        "train_ce_loss": last_ce_loss,
        "train_total_loss": last_total_loss,
        "train_loop1_loss": last_loop1_loss,
        "train_loop_last_loss": last_loop_last_loss,
        "loop_loss_mode": args.loop_loss,
        "loop_loss_start": args.loop_loss_start,
        "loop_loss_power": args.loop_loss_power,
        "loop_loss_min_weight": args.loop_loss_min_weight,
        "loop_loss_weights": last_loop_weights,
        "exact_margin_weight": args.exact_margin_weight,
        "exact_margin_tau": args.exact_margin_tau,
        "exact_margin_target": args.exact_margin_target,
        "exact_margin_start_step": args.exact_margin_start_step,
        "exact_margin_loss": last_exact_margin_loss,
        "exact_margin_softmin": last_exact_margin_softmin,
        "exact_margin_hardmin": last_exact_margin_hardmin,
        "exact_margin_mean": last_exact_margin_mean,
        "noise_mode": "+".join(
            mode
            for enabled, mode in (
                (args.noise_scale > 0, "feature_diff"),
                (args.hidden_agg_noise_scale > 0, "hidden_aggregate"),
            )
            if enabled
        )
        or "none",
        "scratch_mode": args.scratch_mode,
        "scratch_scale": args.scratch_scale,
        "scratch_noise_scale": args.scratch_noise_scale,
        "scratch_gauss_weight": args.scratch_gauss_weight,
        "scratch_gauss_projections": args.scratch_gauss_projections,
        "scratch_gate_bias": args.scratch_gate_bias,
        "scratch_decay_bias": args.scratch_decay_bias,
        "scratch_gauss_loss": last_scratch_gauss_loss,
        "scratch_gate": last_scratch_gate,
        "scratch_decay": last_scratch_decay,
        "scratch_delta": last_scratch_delta,
        "scratch_residual": last_scratch_residual,
        "scratch_proj_var": last_scratch_proj_var,
        "scratch_proj_rank": last_scratch_proj_rank,
        "loop_update_mode": args.loop_update_mode,
        "loop_update_gate_init": args.loop_update_gate_init,
        "loop_update_gate_l": last_loop_update_gate_l,
        "loop_update_gate_h": last_loop_update_gate_h,
        "loop_feedback_next_norm": last_loop_feedback_next_norm,
        "loop_feedback_corrupt_frac": last_loop_feedback_corrupt_frac,
        "loop_feedback_corrupt_confidence": last_loop_feedback_corrupt_confidence,
        "hidden_agg_noise_scale": args.hidden_agg_noise_scale,
        "hidden_agg_noise_temp": args.hidden_agg_noise_temp,
        "hidden_agg_noise_detach": bool(args.hidden_agg_noise_detach),
        "hidden_agg_noise_mode": args.hidden_agg_noise_mode,
        "hidden_agg_noise_topk": args.hidden_agg_noise_topk,
        "hidden_agg_noise_max_norm": args.hidden_agg_noise_max_norm,
        "hidden_agg_noise_norm": last_hidden_agg_noise_norm,
        "hidden_agg_noise_raw_norm": last_hidden_agg_noise_raw_norm,
        "hidden_agg_noise_entropy": last_hidden_agg_noise_entropy,
        "hidden_agg_noise_max_weight": last_hidden_agg_noise_max_weight,
        "hidden_agg_noise_clip_frac": last_hidden_agg_noise_clip_frac,
        "gain_budget_clipped_frac": last_gain_budget_clipped_frac,
        "gain_budget_infeasible_frac": last_gain_budget_infeasible_frac,
        "gain_budget_numerical_endpoint_frac": last_gain_budget_numerical_endpoint_frac,
        "gain_budget_step_bound_max": last_gain_budget_step_bound_max,
        "gain_budget_delta_error_max": last_gain_budget_delta_error_max,
        "gain_budget_gate_relative_change": last_gain_budget_gate_relative_change,
        "fast_slow_decay": dict(last_fast_slow_diag),
        "precondition": dict(last_precondition_diag),
        "coherent_delta": dict(last_coherent_delta_diag),
        "state_feedback": dict(last_state_feedback_diag),
        "terminal_consolidation": dict(last_terminal_consolidation_diag),
        "orthogonal_chunk_state": dict(last_orthogonal_chunk_state_diag),
        "orthogonal_head_write": dict(last_orthogonal_head_write_diag),
        "adaptive_signed_erase": dict(last_adaptive_signed_erase_diag),
        "bi_axis_value_decay": dict(last_bi_axis_value_decay_diag),
        "raven_routed_gdn": dict(last_raven_routed_gdn_diag),
        "paired_address_bank": dict(last_paired_address_bank_diag),
        "coupled_address_rows": dict(last_coupled_address_rows_diag),
        "interleaved_write": dict(last_interleaved_write_diag),
        "state_expert": dict(last_state_expert_diag),
        "address_operator": dict(last_address_diag),
        "future_seed_gradient_projection": (
            summarize_future_seed_projection_stats(
                future_seed_projection_stats
            )
        ),
        "feature_buffer_count": feature_buffer.count,
        "train_sec": time.time() - t0,
        "optimizer_steps": total_steps,
        "parameter_count": parameter_count,
        "trainable_parameter_count": trainable_parameter_count,
        "backbone_runtime": backbone_runtime,
        "optimizer_runtime": optimizer_runtime,
        "microbatch": args.batch,
        "grad_accum_steps": args.grad_accum_steps,
        "effective_batch": args.batch * max(1, int(args.grad_accum_steps)),
        "rwkv_kernel": args.rwkv_kernel,
        "backbone": args.backbone,
        "gdn_mode": args.gdn_mode,
        "gdn_expand_v": args.gdn_expand_v,
        "gdn_progressive_base_expand_v": args.gdn_progressive_base_expand_v,
        "gdn_use_short_conv": bool(args.gdn_use_short_conv),
        "gdn_conv_size": args.gdn_conv_size,
        "gdn_allow_neg_eigval": bool(args.gdn_allow_neg_eigval),
        "gdn2_gain_budget_mode": args.gdn2_gain_budget_mode,
        "gdn2_gain_budget_sigma_cap": args.gdn2_gain_budget_sigma_cap,
        "gdn2_gain_budget_step_cap": args.gdn2_gain_budget_step_cap,
        "gdn2_gain_budget_sigma_cap_max": args.gdn2_gain_budget_sigma_cap_max,
        "gdn2_gain_budget_infeasible_policy": args.gdn2_gain_budget_infeasible_policy,
        "gdn2_fast_slow_decay_mode": args.gdn2_fast_slow_decay_mode,
        "gdn2_fast_slow_decay_kernel_size": args.gdn2_fast_slow_decay_kernel_size,
        "gdn2_fast_slow_decay_rho_init": args.gdn2_fast_slow_decay_rho_init,
        "gdn2_fast_slow_decay_current_weight_init": (
            args.gdn2_fast_slow_decay_current_weight_init
        ),
        "gdn2_precondition_mode": args.gdn2_precondition_mode,
        "gdn2_address_mode": args.gdn2_address_mode,
        "gdn2_update_mode": args.gdn2_update_mode,
        "gdn2_bi_axis_value_decay_groups": (
            8
            if args.gdn2_update_mode
            in {"bi_axis_value_decay", "gauge_balanced_bi_axis"}
            else 0
        ),
        "gdn2_bi_axis_value_decay_potential_cap": (
            math.log(4.0)
            if args.gdn2_update_mode == "gauge_balanced_bi_axis"
            else 0.0
        ),
        "gdn2_raven_routed_slots": (
            8 if args.gdn2_update_mode == "raven_routed_gdn" else 0
        ),
        "gdn2_state_expert_mode": args.gdn2_state_expert_mode,
        "gdn2_cross_layer_init": args.gdn2_cross_layer_init,
        "raven_num_slots": args.raven_num_slots,
        "raven_topk": args.raven_topk,
        "cell_order_train": args.cell_order_train,
        "forward_dtype": args.forward_dtype,
        "activation_checkpoint": bool(args.activation_checkpoint),
        "resume_train_checkpoint": resume_info,
        "resume_require_exact_state": bool(args.resume_require_exact_state),
        "resume_allow_future_seed_content_upgrade": bool(
            args.resume_allow_future_seed_content_upgrade
        ),
        "resume_allow_future_seed_update_upgrade": bool(
            args.resume_allow_future_seed_update_upgrade
        ),
        "resume_allow_future_seed_gradient_upgrade": bool(
            args.resume_allow_future_seed_gradient_upgrade
        ),
        "resume_allow_gdn2_update_upgrade": bool(
            args.resume_allow_gdn2_update_upgrade
        ),
        "resume_allow_gdn2_state_expert_upgrade": bool(
            args.resume_allow_gdn2_state_expert_upgrade
        ),
        "save_train_checkpoint_every": args.save_train_checkpoint_every,
        "saved_train_checkpoints": saved_train_checkpoints,
        "cuda_max_memory_allocated_mb": (
            torch.cuda.max_memory_allocated(device) / (1024**2) if device.type == "cuda" else 0.0
        ),
        "cuda_max_memory_reserved_mb": (
            torch.cuda.max_memory_reserved(device) / (1024**2) if device.type == "cuda" else 0.0
        ),
        "future_seed_update": args.future_seed_update,
        "future_seed_norm_mode": args.future_seed_norm_mode,
        "future_seed_gate_mode": args.future_seed_gate_mode,
        "future_seed_scope": args.future_seed_scope,
        "future_seed_readout_hop": args.future_seed_readout_hop,
        "future_seed_content_mode": args.future_seed_content_mode,
        "momentum_future_seed_transport": args.momentum_future_seed_transport,
        "future_seed_gradient_mode": args.future_seed_gradient_mode,
        "loop_feedback_scale": args.loop_feedback_scale,
        "loop_feedback_detach": bool(args.loop_feedback_detach),
        "loop_feedback_corrupt_prob": args.loop_feedback_corrupt_prob,
        "loop_feedback_corrupt_mix": args.loop_feedback_corrupt_mix,
        "loop_feedback_corrupt_mode": args.loop_feedback_corrupt_mode,
        "loop_time_scale": args.loop_time_scale,
        "data_source": "official_sudoku" if official_train is not None else "generated_random_holes",
        "official_sudoku_data_dir": str(args.official_sudoku_data_dir) if official_train is not None else "",
        "official_sudoku_train_split": str(args.official_sudoku_train_split) if official_train is not None else "",
        "official_sudoku_train_indices": (
            str(args.official_sudoku_train_indices) if official_train is not None else ""
        ),
        "official_sudoku_eval_split": str(args.official_sudoku_eval_split) if official_train is not None else "",
        "official_train_size": len(official_train) if official_train is not None else 0,
        "official_eval_size": len(official_eval) if official_eval is not None else 0,
        "official_train_blank_summary": official_train.blank_summary() if official_train is not None else {},
        "official_eval_blank_summary": official_eval.blank_summary() if official_eval is not None else {},
        "eval_checkpoint_steps": checkpoint_steps,
        "eval_checkpoint_holes": sorted(checkpoint_batches),
        "checkpoint_evals": checkpoint_evals,
    }
    return model.eval(), train_stats


@torch.no_grad()
def evaluate_model(
    model: FutureSeedLoopSudoku,
    batch,
    *,
    max_loops: int,
    noise_scale: float,
    seed: int,
    forward_dtype: str = "float32",
) -> Tuple[Dict[str, Any], List[torch.Tensor]]:
    if noise_scale > 0:
        torch.manual_seed(seed)
    inputs, labels, clue_mask = batch
    feature_buffer = getattr(model, "feature_noise_buffer", None)
    with forward_autocast(forward_dtype, inputs.device):
        loop_logits, fs_trace = model.forward_trace(
            inputs,
            loops=max_loops,
            noise_scale=noise_scale,
            feature_buffer=feature_buffer,
            update_feature_buffer=False,
        )
    out: Dict[str, Any] = {}
    loop_preds: List[torch.Tensor] = []
    for loop_idx, logits in enumerate(loop_logits, start=1):
        metrics, pred = metrics_from_logits(logits, labels, clue_mask)
        out[f"loop{loop_idx}"] = asdict(metrics)
        out[f"loop{loop_idx}/future_seed"] = fs_metrics_from_trace(fs_trace[loop_idx - 1])
        loop_preds.append(pred)
    return out, loop_preds


def parse_rollout_ks(args: argparse.Namespace) -> List[int]:
    raw = str(args.rollout_ks).strip()
    if not raw:
        return []
    values = []
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        value = int(item)
        if value < 1:
            raise ValueError("--rollout_ks values must be positive")
        if value not in values:
            values.append(value)
    return sorted(values)


def parse_rollout_loop_values(args: argparse.Namespace) -> List[int]:
    raw = str(args.rollout_loop_values).strip()
    if not raw:
        return []
    values = []
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        value = int(item)
        if value < 1 or value > args.max_loops:
            raise ValueError(f"--rollout_loop_values entries must be in [1, {args.max_loops}]")
        if value not in values:
            values.append(value)
    return sorted(values)


@torch.no_grad()
def selected_metrics(
    logits: torch.Tensor,
    scores: torch.Tensor,
    labels: torch.Tensor,
    clue_mask: torch.Tensor,
) -> Dict[str, float]:
    batch_size = labels.shape[0]
    selected = scores.argmax(dim=0)
    batch_idx = torch.arange(batch_size, device=labels.device)
    selected_logits = logits[selected, batch_idx]
    metrics, _pred = metrics_from_logits(selected_logits, labels, clue_mask)
    return asdict(metrics)


@torch.no_grad()
def evaluate_rollouts(
    model: FutureSeedLoopSudoku,
    batch,
    *,
    max_loops: int,
    noise_scale: float,
    ks: List[int],
    seed: int,
    forward_dtype: str = "float32",
) -> Dict[str, Any]:
    if not ks:
        return {}
    inputs, labels, clue_mask = batch
    max_k = max(ks)
    last_logits_list = []
    prev_logits_list = []
    feature_buffer = getattr(model, "feature_noise_buffer", None)
    for rollout_idx in range(max_k):
        if noise_scale > 0:
            torch.manual_seed(seed + rollout_idx)
        with forward_autocast(forward_dtype, inputs.device):
            loop_logits, _fs_trace = model.forward_trace(
                inputs,
                loops=max_loops,
                noise_scale=noise_scale,
                feature_buffer=feature_buffer,
                update_feature_buffer=False,
            )
        last_logits_list.append(loop_logits[-1])
        prev_logits_list.append(loop_logits[-2] if max_loops > 1 else loop_logits[-1])

    last_logits = torch.stack(last_logits_list, dim=0)
    prev_logits = torch.stack(prev_logits_list, dim=0)
    preds = last_logits.argmax(dim=-1)
    labels_k = labels.unsqueeze(0)
    clue_k = clue_mask.unsqueeze(0)
    blank_k = ~clue_k
    exact = (preds == labels_k).all(dim=-1)
    clue_ok = ((preds == labels_k) | ~clue_k).all(dim=-1)
    valid_rows = []
    for rollout_idx in range(max_k):
        valid_rows.append(torch.tensor([valid_board(preds[rollout_idx, i]) for i in range(labels.shape[0])], device=labels.device))
    valid = torch.stack(valid_rows, dim=0)
    solved = valid & clue_ok

    probs = last_logits.float().softmax(dim=-1)
    max_prob = probs.max(dim=-1).values
    confidence_scores = (max_prob * blank_k).sum(dim=-1) / blank_k.sum(dim=-1).clamp_min(1)
    residual = (last_logits.float().softmax(dim=-1) - prev_logits.float().softmax(dim=-1)).square().mean(dim=-1)
    residual_scores = -((residual * blank_k).sum(dim=-1) / blank_k.sum(dim=-1).clamp_min(1))

    out: Dict[str, Any] = {}
    for k in ks:
        subset_logits = last_logits[:k]
        subset_preds = preds[:k]
        subset_exact = exact[:k]
        subset_solved = solved[:k]
        mode_pred = F.one_hot(subset_preds, num_classes=N).sum(dim=0).argmax(dim=-1)
        consistency_scores = ((subset_preds == mode_pred.unsqueeze(0)) & blank_k).float().sum(dim=-1)
        consistency_scores = consistency_scores / blank_k.sum(dim=-1).clamp_min(1)
        mode_metrics = asdict(metrics_from_predictions(mode_pred, labels, clue_mask))
        token_disagreement = 1.0 - float(consistency_scores.mean().detach().cpu())

        confidence_metrics = selected_metrics(subset_logits, confidence_scores[:k], labels, clue_mask)
        consistency_metrics = selected_metrics(subset_logits, consistency_scores, labels, clue_mask)
        residual_metrics = selected_metrics(subset_logits, residual_scores[:k], labels, clue_mask)

        oracle_exact = float(subset_exact.any(dim=0).float().mean().detach().cpu())
        oracle_solved = float(subset_solved.any(dim=0).float().mean().detach().cpu())
        out[f"K{k}"] = {
            "oracle_label_exact": oracle_exact,
            "oracle_solved_valid_clue": oracle_solved,
            "trajectory_token_disagreement": token_disagreement,
            "selector_confidence": confidence_metrics,
            "selector_consistency": consistency_metrics,
            "selector_residual": residual_metrics,
            "selector_majority": mode_metrics,
            "selector_gap_confidence": oracle_exact - float(confidence_metrics["label_exact"]),
            "selector_gap_consistency": oracle_exact - float(consistency_metrics["label_exact"]),
            "selector_gap_residual": oracle_exact - float(residual_metrics["label_exact"]),
            "selector_gap_majority": oracle_exact - float(mode_metrics["label_exact"]),
        }
    return out


@torch.no_grad()
def evaluate_rollouts_by_loop(
    model: FutureSeedLoopSudoku,
    batch,
    *,
    loop_values: List[int],
    noise_scale: float,
    ks: List[int],
    seed: int,
    forward_dtype: str = "float32",
) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for loops in loop_values:
        out[f"loop{loops}"] = evaluate_rollouts(
            model,
            batch,
            max_loops=loops,
            noise_scale=noise_scale,
            ks=ks,
            seed=seed + loops * 1009,
            forward_dtype=forward_dtype,
        )
    return out


def board_list(x: torch.Tensor) -> List[int]:
    return [int(v) for v in x.detach().cpu().view(-1).tolist()]


def digit(v: int) -> str:
    return "." if int(v) == BLANK else str(int(v) + 1)


def coord(idx: int) -> str:
    row, col = divmod(int(idx), N)
    return f"R{row + 1}C{col + 1}"


def grid_html(board: List[int], solution: List[int], clue: List[bool]) -> str:
    cells = []
    for i, value in enumerate(board):
        row, col = divmod(i, N)
        cls = ["cell"]
        if clue[i]:
            cls.append("clue")
        elif value == solution[i]:
            cls.append("correct")
        else:
            cls.append("wrong")
        style = []
        if col == N - 1:
            style.append("border-right: 0")
        elif (col + 1) % BOX_COLS == 0:
            style.append("border-right: 2px solid #0f172a")
        if row == N - 1:
            style.append("border-bottom: 0")
        elif (row + 1) % BOX_ROWS == 0:
            style.append("border-bottom: 2px solid #0f172a")
        style_attr = f' style="{"; ".join(style)}"' if style else ""
        cells.append(f'<div class="{" ".join(cls)}"{style_attr}>{html.escape(digit(value))}</div>')
    cols = " ".join(["24px"] * N)
    rows = " ".join(["24px"] * N)
    grid_style = f"grid-template-columns: {cols}; grid-template-rows: {rows}"
    return f'<div class="grid" style="{grid_style}">' + "".join(cells) + "</div>"


def write_case_html(
    path: Path,
    *,
    title: str,
    model_label: str,
    future_seed_enabled: bool,
    inputs: torch.Tensor,
    labels: torch.Tensor,
    clue_mask: torch.Tensor,
    loop_preds: List[torch.Tensor],
) -> None:
    clue = [bool(x) for x in clue_mask.detach().cpu().tolist()]
    solution = board_list(labels)
    puzzle = board_list(inputs)
    sections = []
    for loop_idx, pred in enumerate(loop_preds, start=1):
        sections.append(
            f'<div class="panel"><h3>loop {loop_idx}</h3>{grid_html(board_list(pred), solution, clue)}</div>'
        )
    mechanism = "with FutureSeed" if future_seed_enabled else "without FutureSeed"

    body = f"""<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>{html.escape(title)}</title>
<style>
body {{ font-family: ui-sans-serif, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; margin: 22px; background: #f8fafc; color: #0f172a; }}
h1 {{ margin: 0 0 6px; font-size: 22px; }}
h3 {{ margin: 0 0 8px; font-size: 13px; }}
.meta {{ color: #475569; margin: 0 0 16px; }}
.row {{ display: flex; gap: 12px; flex-wrap: wrap; align-items: flex-start; }}
.panel {{ background: white; border: 1px solid #cbd5e1; border-radius: 8px; padding: 10px; }}
.grid {{ display: grid; border: 2px solid #0f172a; width: max-content; }}
.cell {{ width: 24px; height: 24px; box-sizing: border-box; display: flex; align-items: center; justify-content: center; border-right: 1px solid #94a3b8; border-bottom: 1px solid #94a3b8; font-size: 13px; font-weight: 750; }}
.clue {{ background: #e2e8f0; color: #0f172a; }}
.correct {{ background: #bbf7d0; color: #14532d; }}
.wrong {{ background: #fecaca; color: #7f1d1d; }}
</style>
</head>
<body>
<h1>{html.escape(title)}</h1>
<p class="meta">{html.escape(model_label)} {mechanism} and depth-loop refinement. Green matches sampled solution, red differs, gray is clue.</p>
<div class="row">
  <div class="panel"><h3>puzzle</h3>{grid_html(puzzle, solution, clue)}</div>
  <div class="panel"><h3>solution</h3>{grid_html(solution, solution, [False] * CELLS)}</div>
</div>
<div class="row" style="margin-top: 12px;">{"".join(sections)}</div>
</body>
</html>
"""
    path.write_text(body, encoding="utf-8")


def parse_int_csv(raw: str, *, name: str, low: int, high: int) -> List[int]:
    values: List[int] = []
    if not str(raw).strip():
        return values
    for item in str(raw).split(","):
        item = item.strip()
        if not item:
            continue
        value = int(item)
        if value < low or value > high:
            raise ValueError(f"{name} entries must be in [{low}, {high}]")
        if value not in values:
            values.append(value)
    return values


def conflict_info(board: List[int]) -> Tuple[set[int], List[str]]:
    conflict_cells: set[int] = set()
    conflict_units: List[str] = []
    unit_names = [f"row {i + 1}" for i in range(N)] + [f"col {i + 1}" for i in range(N)] + [
        f"box {r + 1},{c + 1}" for r in range(N // BOX_ROWS) for c in range(N // BOX_COLS)
    ]
    for name, unit in zip(unit_names, UNITS):
        by_digit: Dict[int, List[int]] = {}
        for idx in unit:
            value = int(board[idx])
            if 0 <= value < N:
                by_digit.setdefault(value, []).append(idx)
        repeated = {digit_value: locs for digit_value, locs in by_digit.items() if len(locs) > 1}
        if not repeated:
            continue
        bits = []
        for digit_value, locs in sorted(repeated.items()):
            conflict_cells.update(locs)
            bits.append(f"{digit(digit_value)}@" + ",".join(coord(idx) for idx in locs))
        conflict_units.append(f"{name}: " + "; ".join(bits))
    return conflict_cells, conflict_units


def case_grid_html(
    board: List[int],
    solution: List[int],
    clue: List[bool],
    *,
    conflict_cells: set[int] | None = None,
    previous: List[int] | None = None,
    mode: str = "prediction",
) -> str:
    conflict_cells = conflict_cells or set()
    cell_px = 34 if N <= 12 else 30 if N <= 16 else 26
    font_px = 13 if N <= 16 else 11
    cells = []
    for i, value in enumerate(board):
        row, col = divmod(i, N)
        cls = ["cell"]
        title = coord(i)
        if mode == "puzzle":
            if clue[i]:
                cls.append("clue")
                title += " clue"
            else:
                cls.append("hole")
                title += " hidden"
        elif mode == "solution":
            cls.append("solution")
            if clue[i]:
                cls.append("given-solution")
        elif clue[i]:
            cls.append("clue")
        elif value == solution[i]:
            cls.append("correct")
        else:
            cls.append("wrong")
            title += f" truth={digit(solution[i])}"
        if i in conflict_cells:
            cls.append("conflict")
            title += " duplicate-conflict"
        if previous is not None and not clue[i] and int(previous[i]) != int(value):
            cls.append("changed")
            title += f" changed-from={digit(previous[i])}"
        style = [f"width:{cell_px}px", f"height:{cell_px}px", f"font-size:{font_px}px"]
        if col == N - 1:
            style.append("border-right:0")
        elif (col + 1) % BOX_COLS == 0:
            style.append("border-right:2px solid #0f172a")
        if row == N - 1:
            style.append("border-bottom:0")
        elif (row + 1) % BOX_ROWS == 0:
            style.append("border-bottom:2px solid #0f172a")
        cells.append(
            f'<div class="{" ".join(cls)}" style="{";".join(style)}" title="{html.escape(title)}">'
            f"{html.escape(digit(value))}</div>"
        )
    grid_style = f"grid-template-columns: repeat({N}, {cell_px}px); grid-template-rows: repeat({N}, {cell_px}px)"
    return f'<div class="grid" style="{grid_style}">' + "".join(cells) + "</div>"


def case_panel_html(
    title: str,
    subtitle: str,
    board: List[int],
    solution: List[int],
    clue: List[bool],
    *,
    conflict_cells: set[int] | None = None,
    previous: List[int] | None = None,
    mode: str = "prediction",
) -> str:
    return (
        '<section class="case-panel">'
        f"<h3>{html.escape(title)}</h3>"
        f"<p>{html.escape(subtitle)}</p>"
        f"{case_grid_html(board, solution, clue, conflict_cells=conflict_cells, previous=previous, mode=mode)}"
        "</section>"
    )


def write_case_bank_case_html(path: Path, case: Dict[str, Any], loop_values: List[int]) -> None:
    puzzle = case["_puzzle"]
    solution = case["_solution"]
    clue = case["_clue"]
    panels = [
        case_panel_html("Puzzle", f"{case['holes']} hidden cells", puzzle, solution, clue, mode="puzzle"),
        case_panel_html("Solution", "sampled target board", solution, solution, clue, mode="solution"),
    ]
    previous: Optional[List[int]] = None
    for loop in loop_values:
        loop_key = f"loop{loop}"
        row = case["loops"][loop_key]
        panels.append(
            case_panel_html(
                f"Loop {loop}",
                (
                    f"wrong={row['wrong_count']}/{case['holes']}; "
                    f"changed={row['changed_from_previous']}; "
                    f"entropy={row['hidden_entropy_mean']:.3f}; "
                    f"conflicts={row['conflict_unit_count']}"
                ),
                row["_board"],
                solution,
                clue,
                conflict_cells=set(row["_conflict_cells"]),
                previous=previous,
            )
        )
        previous = row["_board"]
    wrong_lines = []
    for loop in loop_values:
        loop_key = f"loop{loop}"
        wrong = case["loops"][loop_key]["wrong_blanks"]
        wrong_lines.append(
            f"<li><code>{loop_key}</code>: "
            + (", ".join(f"<code>{html.escape(item['coord'])}</code>" for item in wrong[:32]) if wrong else "none")
            + (" ..." if len(wrong) > 32 else "")
            + "</li>"
        )

    body = f"""<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>{html.escape(case['title'])}</title>
<style>
body {{ margin: 22px; background: #f6f8fa; color: #24292f; font-family: ui-sans-serif, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; letter-spacing: 0; }}
h1 {{ margin: 0 0 8px; font-size: 24px; letter-spacing: 0; }}
h2 {{ margin: 18px 0 8px; font-size: 17px; letter-spacing: 0; }}
h3 {{ margin: 0 0 5px; font-size: 14px; letter-spacing: 0; }}
p {{ margin: 0; color: #57606a; font-size: 13px; line-height: 1.35; }}
code {{ font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; }}
.row {{ display: flex; gap: 12px; flex-wrap: wrap; align-items: flex-start; }}
.case-panel {{ background: #fff; border: 1px solid #d0d7de; border-radius: 8px; padding: 10px; }}
.case-panel p {{ min-height: 35px; max-width: 420px; }}
.grid {{ display: grid; border: 2px solid #0f172a; width: max-content; margin-top: 9px; }}
.cell {{ box-sizing: border-box; display: flex; align-items: center; justify-content: center; border-right: 1px solid #94a3b8; border-bottom: 1px solid #94a3b8; font-weight: 760; position: relative; }}
.clue {{ background: #e2e8f0; color: #0f172a; }}
.hole {{ background: #fff7ed; color: #9a3412; }}
.solution {{ background: #e0f2fe; color: #0c4a6e; }}
.given-solution {{ box-shadow: inset 0 0 0 2px rgba(15,23,42,0.13); }}
.correct {{ background: #bbf7d0; color: #14532d; }}
.wrong {{ background: #fecaca; color: #7f1d1d; }}
.conflict::after {{ content: ""; position: absolute; inset: 3px; border: 3px solid #f59e0b; border-radius: 5px; pointer-events: none; }}
.changed::before {{ content: ""; position: absolute; left: 7px; right: 7px; bottom: 4px; height: 3px; background: #1f6feb; border-radius: 4px; }}
.notes {{ max-width: 1200px; margin-top: 14px; background: #fff; border: 1px solid #d0d7de; border-radius: 8px; padding: 12px 14px; }}
.notes ul {{ margin: 8px 0 0 18px; padding: 0; }}
.notes li {{ margin: 4px 0; font-size: 13px; color: #57606a; }}
</style>
</head>
<body>
<h1>{html.escape(case['title'])}</h1>
<p>Kind: <code>{html.escape(case['kind'])}</code>. Batch index {case['batch_index']}. Blue underline marks hidden cells changed since the previous shown loop; orange outline marks duplicate conflicts.</p>
<div class="row" style="margin-top: 14px;">{''.join(panels)}</div>
<div class="notes">
  <h2>Wrong hidden cells</h2>
  <ul>{''.join(wrong_lines)}</ul>
</div>
</body>
</html>
"""
    path.write_text(body, encoding="utf-8")


def json_ready_case(case: Dict[str, Any]) -> Dict[str, Any]:
    out = {key: value for key, value in case.items() if not key.startswith("_")}
    loops = {}
    for loop_key, row in case["loops"].items():
        loops[loop_key] = {key: value for key, value in row.items() if not key.startswith("_")}
    out["loops"] = loops
    return out


def case_cell_diagnostic(
    *,
    cell: int,
    loop: int,
    board: List[int],
    solution: List[int],
    preds_cpu: List[torch.Tensor],
    margins: List[torch.Tensor],
    entropies: List[torch.Tensor],
    idx: int,
    loop_values: List[int],
) -> Dict[str, Any]:
    pred_value = board[cell]
    previous_pred: Optional[int] = None
    previous_loop: Optional[int] = None
    for candidate_loop in reversed([value for value in loop_values if value < loop]):
        previous_pred = int(preds_cpu[candidate_loop - 1][idx].view(-1)[cell].item())
        previous_loop = candidate_loop
        break

    stable_since = loop
    for candidate_loop in reversed([value for value in loop_values if value < loop]):
        candidate_pred = int(preds_cpu[candidate_loop - 1][idx].view(-1)[cell].item())
        if candidate_pred != pred_value:
            break
        stable_since = candidate_loop

    row = {
        "coord": coord(cell),
        "pred": digit(pred_value),
        "truth": digit(solution[cell]),
        "margin": float(margins[loop - 1][idx].view(-1)[cell].item()),
        "entropy": float(entropies[loop - 1][idx].view(-1)[cell].item()),
        "stable_since_loop": stable_since,
        "changed_from_previous_loop": previous_pred is not None and previous_pred != pred_value,
    }
    if previous_pred is not None and previous_loop is not None:
        row["previous_loop"] = previous_loop
        row["previous_pred"] = digit(previous_pred)
    return row


def write_case_bank_index(
    path: Path,
    *,
    label: str,
    future_seed_enabled: bool,
    selected: Dict[str, List[Dict[str, Any]]],
    summary: Dict[str, Any],
) -> None:
    cards = []
    for kind, cases in selected.items():
        rows = []
        for case in cases:
            final_loop = f"loop{summary['final_loop']}"
            row = case["loops"][final_loop]
            rows.append(
                "<tr>"
                f"<td><a href=\"{html.escape(Path(case['html_path']).name)}\">{html.escape(case['stem'])}</a></td>"
                f"<td>{case['batch_index']}</td>"
                f"<td>{row['wrong_count']}</td>"
                f"<td>{row['blank_acc']:.3f}</td>"
                f"<td>{row['hidden_entropy_mean']:.3f}</td>"
                f"<td>{row['conflict_unit_count']}</td>"
                "</tr>"
            )
        if not rows:
            rows.append('<tr><td colspan="6">no cases found in this eval sample</td></tr>')
        cards.append(
            f"""
<section class="panel">
  <h2>{html.escape(kind.replace("_", " "))}</h2>
  <table><thead><tr><th>case</th><th>batch</th><th>final wrong</th><th>blank acc</th><th>entropy</th><th>conflict units</th></tr></thead><tbody>{''.join(rows)}</tbody></table>
</section>
"""
        )

    mechanism = "FutureSeed" if future_seed_enabled else "No-FutureSeed"
    body = f"""<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>{N}x{N} {html.escape(label)} case bank</title>
<style>
body {{ margin: 24px; background: #f6f8fa; color: #24292f; font-family: ui-sans-serif, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; letter-spacing: 0; }}
h1 {{ margin: 0 0 8px; font-size: 26px; letter-spacing: 0; }}
h2 {{ margin: 0 0 10px; font-size: 17px; letter-spacing: 0; }}
p {{ margin: 0 0 14px; color: #57606a; line-height: 1.45; }}
code {{ font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; }}
.grid {{ display: grid; gap: 12px; grid-template-columns: repeat(auto-fit, minmax(360px, 1fr)); }}
.panel {{ background: #fff; border: 1px solid #d0d7de; border-radius: 8px; padding: 14px; }}
table {{ width: 100%; border-collapse: collapse; font-size: 13px; }}
th, td {{ border-bottom: 1px solid #d0d7de; padding: 8px 7px; text-align: left; }}
th {{ color: #57606a; background: #f0f3f6; }}
tr:last-child td {{ border-bottom: 0; }}
a {{ color: #1f6feb; }}
</style>
</head>
<body>
<h1>{N}x{N} {html.escape(label)} {mechanism} loop case bank</h1>
<p>Eval sample: <code>{summary['eval_n']}</code>. Final loop exact: <code>{summary['final_exact']:.4f}</code>. Final loop blank accuracy: <code>{summary['final_blank_acc']:.4f}</code>. This artifact is diagnostic only; it does not change training.</p>
<div class="grid">{''.join(cards)}</div>
</body>
</html>
"""
    path.write_text(body, encoding="utf-8")


@torch.no_grad()
def export_case_bank(
    model: "FutureSeedLoopSudoku",
    out_dir: Path,
    *,
    holes_values: List[int],
    official_eval: Optional[OfficialSudokuDataset],
    official_blank_ranges: Optional[List[Tuple[str, int, int]]] = None,
    eval_n: int,
    cases_per_kind: int,
    loop_values: List[int],
    seed: int,
    forward_dtype: str,
) -> Dict[str, Any]:
    if cases_per_kind <= 0:
        return {}
    if official_eval is None and not holes_values:
        return {}
    bank_root = out_dir / "case_bank"
    bank_root.mkdir(parents=True, exist_ok=True)
    device = next(model.parameters()).device
    feature_buffer = getattr(model, "feature_noise_buffer", None)
    artifacts: Dict[str, Any] = {"case_bank_root": str(bank_root.resolve()), "holes": {}}
    groups: List[
        Tuple[
            int,
            str,
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
            Optional[Dict[str, Any]],
        ]
    ] = []
    if official_eval is not None:
        if official_blank_ranges:
            for offset, (label, lo, hi) in enumerate(official_blank_ranges):
                eval_seed = seed + 91000 + offset * 997 + lo * 37 + hi
                batch = official_eval.fixed_batch_by_blank_range(
                    eval_n,
                    eval_seed,
                    holes_min=lo,
                    holes_max=hi,
                    device=device,
                )
                groups.append(
                    (
                        hi,
                        f"official_{label}",
                        batch,
                        {
                            "eval_seed": int(eval_seed),
                            "range_label": str(label),
                            "holes_min": int(lo),
                            "holes_max": int(hi),
                        },
                    )
                )
        else:
            eval_seed = seed + 91000
            groups.append(
                (
                    0,
                    "official",
                    official_eval.fixed_batch(eval_n, eval_seed, device=device),
                    {
                        "eval_seed": int(eval_seed),
                        "range_label": "all",
                        "holes_min": None,
                        "holes_max": None,
                    },
                )
            )
    else:
        for holes in holes_values:
            batch = make_batch(eval_n, holes, holes, "random", random.Random(seed + 91000 + holes * 97), device=device)
            groups.append((holes, f"h{holes}", batch, None))
    for holes, group_label, batch, official_metadata in groups:
        inputs, labels, clue_mask = batch
        max_loop = max(loop_values)
        if official_metadata is not None and max_loop < 5:
            raise ValueError(
                "official all_cases export requires case_bank loop values "
                "with a maximum of at least 5"
            )
        with forward_autocast(forward_dtype, device):
            loop_logits, _fs_trace = model.forward_trace(
                inputs,
                loops=max_loop,
                noise_scale=0.0,
                feature_buffer=feature_buffer,
                update_feature_buffer=False,
            )
        loop_preds = [logits.argmax(dim=-1) for logits in loop_logits]
        entropies = []
        margins = []
        for logits in loop_logits:
            probs = logits.float().softmax(dim=-1)
            entropies.append((-(probs * probs.clamp_min(1e-9).log()).sum(dim=-1)).detach().cpu())
            top2 = probs.topk(k=2, dim=-1).values
            margins.append((top2[..., 0] - top2[..., 1]).detach().cpu())

        holes_dir = bank_root / group_label
        holes_dir.mkdir(parents=True, exist_ok=True)
        clue_cpu = clue_mask.detach().cpu()
        labels_cpu = labels.detach().cpu()
        inputs_cpu = inputs.detach().cpu()
        preds_cpu = [pred.detach().cpu() for pred in loop_preds]
        all_cases: List[Dict[str, Any]] = []
        all_case_content_hashes: List[str] = []
        selected: Dict[str, List[Dict[str, Any]]] = {"solved_by_loop": [], "almost_solved": [], "hard_failure": []}
        candidates: Dict[str, List[Tuple[Tuple[float, ...], Dict[str, Any]]]] = {key: [] for key in selected}

        for idx in range(eval_n):
            clue = [bool(x) for x in clue_cpu[idx].tolist()]
            blank_indices = [cell for cell, is_clue in enumerate(clue) if not is_clue]
            solution = [int(x) for x in labels_cpu[idx].view(-1).tolist()]
            puzzle = [int(x) for x in inputs_cpu[idx].view(-1).tolist()]
            loops: Dict[str, Any] = {}
            previous: Optional[List[int]] = None
            for loop in loop_values:
                board = [int(x) for x in preds_cpu[loop - 1][idx].view(-1).tolist()]
                wrong = [cell for cell in blank_indices if board[cell] != solution[cell]]
                conflict_cells, conflict_units = conflict_info(board)
                hidden_entropy = float(entropies[loop - 1][idx][blank_indices].mean().item()) if blank_indices else 0.0
                changed = (
                    sum(1 for cell in blank_indices if previous is not None and previous[cell] != board[cell])
                    if previous is not None
                    else 0
                )
                loops[f"loop{loop}"] = {
                    "wrong_count": len(wrong),
                    "blank_acc": 1.0 - (len(wrong) / max(len(blank_indices), 1)),
                    "exact": len(wrong) == 0 and all(board[cell] == solution[cell] for cell in range(CELLS)),
                    "valid_board": valid_board(torch.tensor(board)),
                    "changed_from_previous": changed,
                    "hidden_entropy_mean": hidden_entropy,
                    "conflict_unit_count": len(conflict_units),
                    "conflict_units": conflict_units[:32],
                    "wrong_blanks": [
                        case_cell_diagnostic(
                            cell=cell,
                            loop=loop,
                            board=board,
                            solution=solution,
                            preds_cpu=preds_cpu,
                            margins=margins,
                            entropies=entropies,
                            idx=idx,
                            loop_values=loop_values,
                        )
                        for cell in wrong[:64]
                    ],
                    "_board": board,
                    "_conflict_cells": sorted(conflict_cells),
                }
                previous = board

            if official_metadata is not None:
                input_tokens = list(puzzle)
                label_tokens = list(solution)
                clue_values = list(clue)
                content_payload = {
                    "clue_mask": clue_values,
                    "input": input_tokens,
                    "label_tokens": label_tokens,
                }
                content_sha256 = hashlib.sha256(
                    json.dumps(
                        content_payload,
                        sort_keys=True,
                        separators=(",", ":"),
                    ).encode("utf-8")
                ).hexdigest()
                case_identity = {
                    "batch_index": int(idx),
                    "content_sha256": content_sha256,
                    "eval_seed": int(official_metadata["eval_seed"]),
                    "holes_max": official_metadata["holes_max"],
                    "holes_min": official_metadata["holes_min"],
                    "range_label": str(official_metadata["range_label"]),
                    "split": str(official_eval.split),
                }
                case_id = "sha256:" + hashlib.sha256(
                    json.dumps(
                        case_identity,
                        sort_keys=True,
                        separators=(",", ":"),
                    ).encode("utf-8")
                ).hexdigest()
                blank_count = len(blank_indices)
                all_case_loops: Dict[str, Any] = {}
                for loop in range(1, 6):
                    prediction_tokens = [
                        int(value)
                        for value in preds_cpu[loop - 1][idx].view(-1).tolist()
                    ]
                    wrong_positions = [
                        cell
                        for cell, (predicted, target) in enumerate(
                            zip(prediction_tokens, label_tokens)
                        )
                        if predicted != target
                    ]
                    wrong_blank_positions = [
                        cell
                        for cell in blank_indices
                        if prediction_tokens[cell] != label_tokens[cell]
                    ]
                    all_case_loops[f"loop{loop}"] = {
                        "blank_acc": 1.0
                        - len(wrong_blank_positions) / max(blank_count, 1),
                        "blank_count": int(blank_count),
                        "label_exact": len(wrong_positions) == 0,
                        "prediction": [
                            value + 1 for value in prediction_tokens
                        ],
                        "prediction_tokens": prediction_tokens,
                        "wrong_blank_count": int(len(wrong_blank_positions)),
                        "wrong_clue_count": int(
                            len(wrong_positions) - len(wrong_blank_positions)
                        ),
                        "wrong_count": int(len(wrong_positions)),
                        "wrong_total_count": int(len(wrong_positions)),
                    }
                all_cases.append(
                    {
                        "batch_index": int(idx),
                        "blank_count": int(blank_count),
                        "case_id": case_id,
                        "clue_mask": clue_values,
                        "content_sha256": "sha256:" + content_sha256,
                        "input": input_tokens,
                        "label": [value + 1 for value in label_tokens],
                        "label_tokens": label_tokens,
                        "loops": all_case_loops,
                        "puzzle": [
                            value + 1 if clue_values[cell] else 0
                            for cell, value in enumerate(input_tokens)
                        ],
                    }
                )
                all_case_content_hashes.append(content_sha256)

            first_key = f"loop{loop_values[0]}"
            final_key = f"loop{loop_values[-1]}"
            first_wrong = loops[first_key]["wrong_count"]
            final_wrong = loops[final_key]["wrong_count"]
            final_blank_acc = loops[final_key]["blank_acc"]
            final_conflicts = loops[final_key]["conflict_unit_count"]
            blank_count = len(blank_indices)
            case = {
                "holes": blank_count,
                "group": group_label,
                "batch_index": idx,
                "loops": loops,
                "_puzzle": puzzle,
                "_solution": solution,
                "_clue": clue,
            }
            if first_wrong > 0 and final_wrong == 0:
                kind = "solved_by_loop"
                score = (-float(first_wrong), float(loops[final_key]["hidden_entropy_mean"]), float(idx))
            elif 1 <= final_wrong <= 4:
                kind = "almost_solved"
                score = (float(final_wrong), float(final_conflicts), -float(final_blank_acc), float(idx))
            elif final_wrong > 4:
                kind = "hard_failure"
                score = (-float(final_blank_acc), float(final_wrong), float(final_conflicts), float(idx))
            else:
                continue
            case["kind"] = kind
            candidates[kind].append((score, case))

        used: set[int] = set()
        for kind, rows in candidates.items():
            rows.sort(key=lambda item: item[0])
            for _score, case in rows:
                if case["batch_index"] in used:
                    continue
                used.add(case["batch_index"])
                stem = f"{group_label}_{kind}_{len(selected[kind]) + 1:02d}_b{case['batch_index']:04d}"
                case["stem"] = stem
                case["title"] = f"{N}x{N} {group_label} {kind} #{len(selected[kind]) + 1}"
                html_path = holes_dir / f"{stem}.html"
                write_case_bank_case_html(html_path, case, loop_values)
                case["html_path"] = str(html_path.resolve())
                selected[kind].append(case)
                if len(selected[kind]) >= cases_per_kind:
                    break

        final_logits = loop_logits[loop_values[-1] - 1]
        final_metrics, _final_preds = metrics_from_logits(final_logits, labels, clue_mask)
        blank_counts = (~clue_mask).sum(dim=-1).detach().cpu().float()
        holes_summary = {
            "holes": holes,
            "group": group_label,
            "eval_n": eval_n,
            "blank_count_min": int(blank_counts.min().item()) if blank_counts.numel() else 0,
            "blank_count_max": int(blank_counts.max().item()) if blank_counts.numel() else 0,
            "blank_count_mean": float(blank_counts.mean().item()) if blank_counts.numel() else 0.0,
            "loop_values": loop_values,
            "final_loop": loop_values[-1],
            "final_exact": final_metrics.label_exact,
            "final_blank_acc": final_metrics.blank_acc,
            "selected_counts": {kind: len(cases) for kind, cases in selected.items()},
        }
        json_path = holes_dir / "cases.json"
        index_path = holes_dir / "index.html"
        json_path.write_text(
            json.dumps(
                {
                    "summary": holes_summary,
                    "selected": {kind: [json_ready_case(case) for case in cases] for kind, cases in selected.items()},
                },
                indent=2,
            )
            + "\n",
                encoding="utf-8",
        )
        write_case_bank_index(
            index_path,
            label=group_label,
            future_seed_enabled=model.reasoner.future_seed_scale > 0,
            selected=selected,
            summary=holes_summary,
        )
        group_artifacts = {
            "index_html": str(index_path.resolve()),
            "cases_json": str(json_path.resolve()),
            "summary": holes_summary,
        }
        if official_metadata is not None:
            data_hasher = hashlib.sha256()
            for content_sha256 in all_case_content_hashes:
                data_hasher.update(bytes.fromhex(content_sha256))
            data_hash = "sha256:" + data_hasher.hexdigest()
            all_cases_path = holes_dir / "all_cases.json"
            all_cases_payload = {
                "arm": {
                    "future_seed_enabled": bool(
                        model.reasoner.future_seed_scale > 0
                    ),
                    "future_seed_scale": float(
                        model.reasoner.future_seed_scale
                    ),
                    "future_seed_content_mode": (
                        model.reasoner.future_seed_content_mode
                    ),
                    "momentum_future_seed_transport": (
                        model.reasoner.momentum_future_seed_transport
                    ),
                },
                "cases": all_cases,
                "data_hash": data_hash,
                "data_hash_scope": (
                    "sha256 over ordered per-case hashes of input, "
                    "label_tokens, and clue_mask"
                ),
                "encoding": {
                    "display_digits": f"1..{N}",
                    "input_tokens": f"0..{N - 1}; blank={BLANK}",
                    "prediction_tokens": f"0..{N - 1}",
                    "puzzle_blank": 0,
                },
                "eval": {
                    "actual_n": int(len(all_cases)),
                    "eval_seed": int(official_metadata["eval_seed"]),
                    "range": {
                        "holes_max": official_metadata["holes_max"],
                        "holes_min": official_metadata["holes_min"],
                        "label": str(official_metadata["range_label"]),
                    },
                    "requested_n": int(eval_n),
                    "split": str(official_eval.split),
                },
                "prediction_loops": [1, 2, 3, 4, 5],
                "forward_trace_total_loops": int(max_loop),
                "schema_version": "official_sudoku_all_cases.v1",
            }
            all_cases_path.write_text(
                json.dumps(
                    all_cases_payload,
                    sort_keys=True,
                    separators=(",", ":"),
                )
                + "\n",
                encoding="utf-8",
            )
            group_artifacts["all_cases_json"] = str(
                all_cases_path.resolve()
            )
            group_artifacts["data_hash"] = data_hash
        artifacts["holes"][group_label] = group_artifacts
    return artifacts


def parse_eval_holes(args: argparse.Namespace) -> List[int]:
    values = [int(args.eval_holes)]
    if str(args.eval_holes_list).strip():
        for raw in str(args.eval_holes_list).split(","):
            raw = raw.strip()
            if raw:
                values.append(int(raw))
    deduped = []
    for value in values:
        if value < 1 or value > CELLS:
            raise ValueError(f"--eval_holes values must be in [1, {CELLS}]")
        if value not in deduped:
            deduped.append(value)
    return deduped


def parse_official_eval_blank_ranges(raw: str) -> List[Tuple[str, int, int]]:
    ranges: List[Tuple[str, int, int]] = []
    for item in str(raw).split(","):
        item = item.strip()
        if not item:
            continue
        if "-" in item:
            lo_raw, hi_raw = item.split("-", 1)
            lo = int(lo_raw.strip())
            hi = int(hi_raw.strip())
        else:
            lo = hi = int(item)
        if lo < 0 or hi > CELLS or hi < lo:
            raise ValueError(f"--official_eval_blank_ranges entries must be blank ranges within [0, {CELLS}], got {item!r}")
        label = f"b{lo}" if lo == hi else f"b{lo}_{hi}"
        if label not in {existing[0] for existing in ranges}:
            ranges.append((label, lo, hi))
    return ranges


def parse_eval_checkpoint_holes(args: argparse.Namespace) -> List[int]:
    raw = str(args.eval_checkpoint_holes_list).strip()
    if not raw:
        return parse_eval_holes(args)
    values: List[int] = []
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        value = int(item)
        if value < 1 or value > CELLS:
            raise ValueError(f"--eval_checkpoint_holes_list values must be in [1, {CELLS}]")
        if value not in values:
            values.append(value)
    return values


def parse_hole_stages(args: argparse.Namespace) -> List[Tuple[int, int, int]]:
    raw = str(args.hole_stages).strip()
    if not raw:
        return [(int(args.holes_min), int(args.holes_max), int(args.steps))]
    stages = []
    for item in raw.split(","):
        span, steps = item.split(":", 1)
        lo, hi = span.split(":", 1) if ":" in span else span.split("-", 1)
        stages.append((int(lo), int(hi), int(steps)))
    for lo, hi, steps in stages:
        if lo < 1 or hi < lo or steps < 1:
            raise ValueError("--hole_stages entries must look like 2-4:100 with 1 <= lo <= hi and steps > 0")
    return stages


def parse_eval_checkpoint_steps(args: argparse.Namespace, stages: List[Tuple[int, int, int]]) -> List[int]:
    total_steps = sum(stage_steps for _lo, _hi, stage_steps in stages)
    steps: List[int] = []
    for value in parse_int_csv(args.eval_checkpoint_steps, name="--eval_checkpoint_steps", low=1, high=total_steps):
        if value not in steps:
            steps.append(value)
    final_stage_start = total_steps - stages[-1][2]
    for offset in parse_int_csv(
        args.eval_checkpoint_stage_offsets,
        name="--eval_checkpoint_stage_offsets",
        low=1,
        high=stages[-1][2],
    ):
        step = final_stage_start + offset
        if step not in steps:
            steps.append(step)
    return sorted(steps)


def write_report(path: Path, metrics: Dict[str, Any], artifacts: Dict[str, str]) -> None:
    task = metrics.get("task", {})
    backbone = str(task.get("backbone", "rwkv"))
    model_label = BACKBONE_DISPLAY_NAMES.get(backbone, backbone.upper())
    lines = [
        f"# {N}x{N} {model_label} FutureSeed Loop Study",
        "",
        f"Mainline mechanism: {model_label} recurrent backbone, FutureSeed cross-layer terminal-state initialization, and depth-loop iterative refinement.",
        "",
        f"Board: {N}x{N}, box: {BOX_ROWS}x{BOX_COLS}, hole pattern: `{task.get('hole_pattern', 'random')}`.",
        "",
        "## Loop Metrics",
        "",
    ]
    train = metrics.get("train", {})
    if train:
        lines.append(
            "- train: "
            f"ce={train.get('train_ce_loss', 0.0):.4f}, "
            f"total={train.get('train_total_loss', 0.0):.4f}, "
            f"loop_loss={train.get('loop_loss_mode', 'final')}, "
            f"noise={train.get('noise_mode', 'feature_diff')}, "
            f"buffer={train.get('feature_buffer_count', 0)}, "
            f"dtype={train.get('forward_dtype', 'float32')}, "
            f"fs_update={train.get('future_seed_update', 'fixed')}, "
            f"loop_update={train.get('loop_update_mode', 'fixed')}, "
            f"upd_h={train.get('loop_update_gate_h', 0.0):.3f}, "
            f"loop_fb={train.get('loop_feedback_scale', 0.0):.2f}, "
            f"loop_time={train.get('loop_time_scale', 0.0):.2f}, "
            f"scratch={train.get('scratch_mode', 'none')}, "
            f"scratch_delta={train.get('scratch_delta', 0.0):.3f}, "
            f"scratch_gauss={train.get('scratch_gauss_loss', 0.0):.4f}, "
            f"exact_margin={train.get('exact_margin_loss', 0.0):.4f}, "
            f"exact_softmin={train.get('exact_margin_softmin', 0.0):.3f}, "
            f"loop1_loss={train.get('train_loop1_loss', 0.0):.4f}, "
            f"loop_last_loss={train.get('train_loop_last_loss', 0.0):.4f}, "
            f"sec={train.get('train_sec', 0.0):.1f}"
        )
    for loop in range(1, int(task.get("max_loops", 3)) + 1):
        loop_key = f"loop{loop}"
        if loop_key not in metrics["eval_clean"]:
            continue
        lines.append(f"- loop {loop}: {metric_line(metrics['eval_clean'][loop_key])}")
        fs = metrics["eval_clean"].get(f"{loop_key}/future_seed")
        if fs:
            lines.append(f"  future_seed: {fs_line(fs)}")
    if "eval_noisy" in metrics:
        lines.extend(["", "## Eval Noise", ""])
        max_loop = task.get("max_loops", 3)
        last_loop_key = f"loop{max_loop}"
        lines.append(f"- clean eval loop{max_loop}: {metric_line(metrics['eval_clean'][last_loop_key])}")
        lines.append(f"- noisy eval loop{max_loop}: {metric_line(metrics['eval_noisy'][last_loop_key])}")
    if metrics.get("rollouts"):
        lines.extend(["", "## Stochastic Rollouts", ""])
        for key, row in metrics["rollouts"].items():
            lines.append(
                f"- {key}: oracle_exact={row['oracle_label_exact']:.4f}, "
                f"oracle_solved={row['oracle_solved_valid_clue']:.4f}, "
                f"disagree={row['trajectory_token_disagreement']:.4f}"
            )
            lines.append(
                f"  confidence: {metric_line(row['selector_confidence'])}; "
                f"gap={row['selector_gap_confidence']:.4f}"
            )
            lines.append(
                f"  consistency: {metric_line(row['selector_consistency'])}; "
                f"gap={row['selector_gap_consistency']:.4f}"
            )
            lines.append(
                f"  residual: {metric_line(row['selector_residual'])}; "
                f"gap={row['selector_gap_residual']:.4f}"
            )
            lines.append(
                f"  majority: {metric_line(row['selector_majority'])}; "
                f"gap={row['selector_gap_majority']:.4f}"
            )
    if metrics.get("rollouts_by_loop"):
        lines.extend(["", "## Compute Scaling", ""])
        for loop_key, by_k in metrics["rollouts_by_loop"].items():
            lines.append(f"### {loop_key}")
            for key, row in by_k.items():
                lines.append(
                    f"- {key}: oracle_exact={row['oracle_label_exact']:.4f}, "
                    f"confidence={row['selector_confidence']['label_exact']:.4f}, "
                    f"residual={row['selector_residual']['label_exact']:.4f}, "
                    f"disagree={row['trajectory_token_disagreement']:.4f}"
                )
    if metrics.get("case_bank"):
        lines.extend(["", "## Case Bank", ""])
        case_bank = metrics["case_bank"]
        for hole_key, row in case_bank.get("holes", {}).items():
            summary = row.get("summary", {})
            lines.append(
                f"- {hole_key}: index={row.get('index_html', '')}; "
                f"cases={row.get('cases_json', '')}; "
                f"final_loop={summary.get('final_loop')}, "
                f"exact={summary.get('final_exact', 0.0):.4f}, "
                f"blank_acc={summary.get('final_blank_acc', 0.0):.4f}, "
                f"selected={summary.get('selected_counts', {})}"
            )
    if len(metrics.get("eval_by_holes", {})) > 1:
        lines.extend(["", "## Hole Transfer", ""])
        for hole_key, hole_metrics in metrics["eval_by_holes"].items():
            holes = int(hole_key.removeprefix("holes"))
            loop3 = hole_metrics["eval_clean"][f"loop{task.get('max_loops', 3)}"]
            lines.append(f"- {hole_key}: {metric_line(loop3)}; {coupling_line(loop3, holes)}")
    if metrics.get("official_eval_by_blank_range"):
        lines.extend(["", "## Official Blank-Range Eval", ""])
        for range_key, row in metrics["official_eval_by_blank_range"].items():
            loop_last = row["eval_clean"][f"loop{task.get('max_loops', 3)}"]
            lo, hi = row.get("blank_range", [0, 0])
            lines.append(
                f"- {range_key} ({lo}-{hi}, n={row.get('eval_n', 0)}): "
                f"{metric_line(loop_last)}"
            )
    lines.extend(["", "## Decision", "", metrics["decision"], "", "## Artifacts", ""])
    for name, value in artifacts.items():
        lines.append(f"- {name}: {value}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> Dict[str, Any]:
    if args.backbone != "fla_gdn" and args.d_model != args.heads * args.head_dim:
        raise ValueError("--d_model must equal --heads * --head_dim except for official FLA GDN geometry")
    if args.future_seed_scale < 0:
        raise ValueError("--future_seed_scale must be non-negative")
    if args.future_seed_gate_mode in {"state", "content"} and args.backbone != "gdn2":
        raise ValueError(
            "--future_seed_gate_mode state/content is initially restricted to the audited GDN2 state layout"
        )
    if args.future_seed_readout_hop > 0 and args.backbone != "gdn2":
        raise ValueError("--future_seed_readout_hop is initially restricted to GDN2")
    if args.resume_allow_future_seed_content_upgrade:
        if not str(args.resume_train_checkpoint).strip() or not args.resume_require_exact_state:
            raise ValueError(
                "--resume_allow_future_seed_content_upgrade requires an exact checkpoint resume"
            )
        if args.future_seed_content_mode not in {
            "innovation_residual",
            "producer_codec",
            "address_local_update",
            "orthogonal_basis_transport",
        }:
            raise ValueError(
                "--resume_allow_future_seed_content_upgrade requires a nonterminal "
                "FutureSeed content mode"
            )
    if args.resume_allow_future_seed_update_upgrade:
        if not str(args.resume_train_checkpoint).strip() or not args.resume_require_exact_state:
            raise ValueError(
                "--resume_allow_future_seed_update_upgrade requires an exact "
                "checkpoint resume"
            )
        if args.future_seed_update != "loop_secant":
            raise ValueError(
                "--resume_allow_future_seed_update_upgrade requires "
                "--future_seed_update loop_secant"
            )
    if args.future_seed_update == "loop_secant" and (
        not str(args.resume_train_checkpoint).strip()
        or not args.resume_require_exact_state
        or not args.resume_allow_future_seed_update_upgrade
    ):
        raise ValueError(
            "Loop-secant FutureSeed is candidate-only and requires its "
            "explicit exact-resume semantic upgrade"
        )
    if args.resume_allow_future_seed_gradient_upgrade:
        if not str(args.resume_train_checkpoint).strip() or not args.resume_require_exact_state:
            raise ValueError(
                "--resume_allow_future_seed_gradient_upgrade requires an exact "
                "checkpoint resume"
            )
        if args.future_seed_gradient_mode != "opening_projection":
            raise ValueError(
                "--resume_allow_future_seed_gradient_upgrade requires "
                "--future_seed_gradient_mode opening_projection"
            )
    if args.future_seed_gradient_mode == "opening_projection":
        if (
            not str(args.resume_train_checkpoint).strip()
            or not args.resume_require_exact_state
            or not args.resume_allow_future_seed_gradient_upgrade
        ):
            raise ValueError(
                "FutureSeed opening projection is candidate-only and requires "
                "its explicit exact-resume semantic upgrade"
            )
        if args.max_loops != 5 or args.loop_loss != "all":
            raise ValueError(
                "FutureSeed opening projection requires max_loops=5 and loop_loss=all"
            )
        if args.future_seed_content_mode != "terminal":
            raise ValueError(
                "FutureSeed opening projection requires native terminal content"
            )
        if args.future_seed_scope != "layer" or args.future_seed_gate_mode != "head":
            raise ValueError(
                "FutureSeed opening projection requires layer scope and head gates"
            )
        if args.scratch_gauss_weight != 0.0 or args.exact_margin_weight != 0.0:
            raise ValueError(
                "FutureSeed opening projection forbids auxiliary gradient terms"
            )
        if args.activation_checkpoint:
            raise ValueError(
                "FutureSeed opening projection forbids activation checkpointing"
            )
    if args.resume_allow_gdn2_update_upgrade:
        if not str(args.resume_train_checkpoint).strip() or not args.resume_require_exact_state:
            raise ValueError(
                "--resume_allow_gdn2_update_upgrade requires an exact checkpoint resume"
            )
        if args.gdn2_update_mode not in {
            "log_spd_metric",
            "coherent_delta",
            "state_feedback",
            "terminal_consolidation",
            "orthogonal_chunk_state",
            "orthogonal_head_write",
            "adaptive_signed_erase",
            "bi_axis_value_decay",
            "gauge_balanced_bi_axis",
            "raven_routed_gdn",
            "paired_address_bank",
            "coupled_address_rows",
            "interleaved_write",
        }:
            raise ValueError(
                "--resume_allow_gdn2_update_upgrade requires "
                "a non-default --gdn2_update_mode"
            )
    if args.resume_allow_gdn2_state_expert_upgrade:
        if not str(args.resume_train_checkpoint).strip() or not args.resume_require_exact_state:
            raise ValueError(
                "--resume_allow_gdn2_state_expert_upgrade requires an exact "
                "checkpoint resume"
            )
        if args.gdn2_state_expert_mode not in {
            "dual_state",
            "raven_write_control",
        }:
            raise ValueError(
                "--resume_allow_gdn2_state_expert_upgrade requires "
                "a non-default --gdn2_state_expert_mode"
            )
    if not (0.0 < args.loop_update_gate_init < 1.0):
        raise ValueError("--loop_update_gate_init must be in (0, 1)")
    if not (0.0 <= args.future_seed_decay < 1.0):
        raise ValueError("--future_seed_decay must be in [0, 1)")
    if args.loop_feedback_scale < 0:
        raise ValueError("--loop_feedback_scale must be non-negative")
    if args.loop_feedback_detach not in {0, 1}:
        raise ValueError("--loop_feedback_detach must be 0 or 1")
    if not (0.0 <= args.loop_feedback_corrupt_prob <= 1.0):
        raise ValueError("--loop_feedback_corrupt_prob must be in [0, 1]")
    if not (0.0 <= args.loop_feedback_corrupt_mix <= 1.0):
        raise ValueError("--loop_feedback_corrupt_mix must be in [0, 1]")
    if args.loop_feedback_corrupt_prob > 0 and args.loop_feedback_scale <= 0:
        raise ValueError("--loop_feedback_corrupt_prob > 0 requires --loop_feedback_scale > 0")
    if args.loop_time_scale < 0:
        raise ValueError("--loop_time_scale must be non-negative")
    if args.scratch_scale < 0:
        raise ValueError("--scratch_scale must be non-negative")
    if args.scratch_noise_scale < 0:
        raise ValueError("--scratch_noise_scale must be non-negative")
    if args.scratch_gauss_weight < 0:
        raise ValueError("--scratch_gauss_weight must be non-negative")
    if args.scratch_gauss_projections < 0:
        raise ValueError("--scratch_gauss_projections must be non-negative")
    if args.scratch_gauss_weight > 0 and args.scratch_mode == "none":
        raise ValueError("--scratch_gauss_weight requires --scratch_mode gated")
    if args.scratch_gauss_weight > 0 and args.scratch_gauss_projections == 0:
        raise ValueError("--scratch_gauss_weight requires positive --scratch_gauss_projections")
    if args.hidden_agg_noise_scale < 0:
        raise ValueError("--hidden_agg_noise_scale must be non-negative")
    if args.hidden_agg_noise_temp <= 0:
        raise ValueError("--hidden_agg_noise_temp must be positive")
    if args.hidden_agg_noise_detach not in {0, 1}:
        raise ValueError("--hidden_agg_noise_detach must be 0 or 1")
    if args.hidden_agg_noise_topk < 1:
        raise ValueError("--hidden_agg_noise_topk must be positive")
    if args.hidden_agg_noise_max_norm < 0:
        raise ValueError("--hidden_agg_noise_max_norm must be non-negative")
    if args.exact_margin_weight < 0:
        raise ValueError("--exact_margin_weight must be non-negative")
    if args.exact_margin_tau <= 0:
        raise ValueError("--exact_margin_tau must be positive")
    if args.exact_margin_start_step < 0:
        raise ValueError("--exact_margin_start_step must be non-negative")
    if args.forward_dtype not in {"float32", "bfloat16"}:
        raise ValueError("--forward_dtype must be 'float32' or 'bfloat16'")
    if args.gdn2_gain_budget_mode != "none" and args.backbone != "gdn2":
        raise ValueError("--gdn2_gain_budget_mode requires --backbone gdn2")
    if args.gdn2_gain_budget_sigma_cap < 1.0:
        raise ValueError("--gdn2_gain_budget_sigma_cap must be at least 1")
    if args.gdn2_gain_budget_step_cap < 1.0:
        raise ValueError("--gdn2_gain_budget_step_cap must be at least 1")
    if args.gdn2_gain_budget_sigma_cap_max < 1.0:
        raise ValueError("--gdn2_gain_budget_sigma_cap_max must be at least 1")
    if (
        args.gdn2_gain_budget_mode != "none"
        and args.gdn2_gain_budget_infeasible_policy != "raise"
    ):
        raise ValueError("Integrated Gain-Budget requires infeasible_policy=raise")
    if (
        args.gdn2_gain_budget_mode != "none"
        and not args.fla_strict_official
    ):
        raise ValueError("Gain-Budget requires --fla_strict_official")
    if (
        args.gdn2_fast_slow_decay_mode != "none"
        and args.backbone != "gdn2"
    ):
        raise ValueError("--gdn2_fast_slow_decay_mode requires --backbone gdn2")
    if (
        args.gdn2_fast_slow_decay_mode != "none"
        and args.gdn2_gain_budget_mode != "none"
    ):
        raise ValueError("Fast-Slow decay and Gain-Budget cannot be enabled together")
    if (
        args.gdn2_fast_slow_decay_mode != "none"
        and not args.fla_strict_official
    ):
        raise ValueError("Fast-Slow decay requires --fla_strict_official")
    if args.gdn2_fast_slow_decay_kernel_size < 1:
        raise ValueError("--gdn2_fast_slow_decay_kernel_size must be positive")
    if not (0.0 < args.gdn2_fast_slow_decay_rho_init < 1.0):
        raise ValueError("--gdn2_fast_slow_decay_rho_init must be in (0, 1)")
    if not (
        0.0 < args.gdn2_fast_slow_decay_current_weight_init <= 1.0
    ):
        raise ValueError(
            "--gdn2_fast_slow_decay_current_weight_init must be in (0, 1]"
        )
    if args.gdn2_precondition_mode != "none" and args.backbone != "gdn2":
        raise ValueError("--gdn2_precondition_mode requires --backbone gdn2")
    if args.gdn2_precondition_mode != "none" and not args.fla_strict_official:
        raise ValueError("Tied preconditioning requires --fla_strict_official")
    if args.gdn2_precondition_mode != "none" and (
        args.gdn2_gain_budget_mode != "none"
        or args.gdn2_fast_slow_decay_mode != "none"
    ):
        raise ValueError(
            "Tied preconditioning cannot be mixed with Gain-Budget or Fast-Slow decay"
        )
    if args.gdn2_precondition_mode != "none" and args.gdn2_address_mode not in {
        "none",
        "position_qk",
    }:
        raise ValueError(
            "Tied preconditioning currently composes only with none or position_qk addressing"
        )
    if args.gdn2_update_mode != "none":
        if args.backbone != "gdn2":
            raise ValueError("--gdn2_update_mode requires --backbone gdn2")
        if not args.fla_strict_official:
            raise ValueError("--gdn2_update_mode requires --fla_strict_official")
        if args.gdn2_address_mode != "position_qk":
            raise ValueError(
                "GDN2 update extensions require --gdn2_address_mode position_qk"
            )
        if args.future_seed_content_mode != "terminal":
            raise ValueError(
                "GDN2 update extensions require terminal FutureSeed content"
            )
        if (
            args.gdn2_gain_budget_mode != "none"
            or args.gdn2_fast_slow_decay_mode != "none"
            or args.gdn2_precondition_mode != "none"
        ):
            raise ValueError(
                "GDN2 update extensions cannot be mixed with Gain-Budget, "
                "Fast-Slow decay, or tied preconditioning"
            )
    if args.gdn2_state_expert_mode != "none":
        if args.backbone != "gdn2":
            raise ValueError("--gdn2_state_expert_mode requires --backbone gdn2")
        if not args.fla_strict_official:
            raise ValueError(
                "--gdn2_state_expert_mode requires --fla_strict_official"
            )
        if args.gdn2_address_mode != "position_qk":
            raise ValueError(
                "GDN3 auxiliary state modes require "
                "--gdn2_address_mode position_qk"
            )
        if args.gdn2_update_mode != "none":
            raise ValueError(
                "GDN3 auxiliary state modes require an unmodified main GDN2 update"
            )
        if args.gdn2_cross_layer_init != "independent":
            raise ValueError(
                "GDN3 auxiliary state modes require independent main-state coordinates"
            )
        if args.future_seed_content_mode != "terminal":
            raise ValueError(
                "GDN3 auxiliary state modes require terminal FutureSeed content"
            )
        if args.future_seed_norm_mode != "unit" or args.future_seed_gate_mode != "head":
            raise ValueError(
                "GDN3 auxiliary state modes require unit FutureSeed normalization "
                "and head gates"
            )
        if args.future_seed_scope != "layer" or args.future_seed_readout_hop != 0:
            raise ValueError(
                "GDN3 auxiliary state modes require adjacent-layer FutureSeed routing"
            )
        if (
            not math.isclose(args.future_seed_scale, 1.0)
            or not math.isclose(args.future_seed_decay, 0.0)
            or args.future_seed_update != "fixed"
        ):
            raise ValueError(
                "GDN3 auxiliary state modes require unit-scale fixed terminal "
                "FutureSeed without decay"
            )
        if (
            args.gdn2_gain_budget_mode != "none"
            or args.gdn2_fast_slow_decay_mode != "none"
            or args.gdn2_precondition_mode != "none"
        ):
            raise ValueError(
                "GDN3 auxiliary state modes cannot be mixed with Gain-Budget, "
                "Fast-Slow decay, or tied preconditioning"
            )
    if args.gdn2_address_mode != "none" and args.backbone != "gdn2":
        raise ValueError("--gdn2_address_mode requires --backbone gdn2")
    if args.gdn2_address_mode != "none" and not args.fla_strict_official:
        raise ValueError("--gdn2_address_mode requires --fla_strict_official")
    if args.gdn2_address_mode != "none" and (
        args.gdn2_gain_budget_mode != "none"
        or args.gdn2_fast_slow_decay_mode != "none"
    ):
        raise ValueError(
            "Address-payload separation cannot be mixed with Gain-Budget or Fast-Slow decay"
        )
    if args.gdn2_address_mode != "none" and args.activation_checkpoint:
        raise ValueError(
            "--gdn2_address_mode currently forbids --activation_checkpoint"
        )
    if args.gdn2_address_mode != "none" and args.future_seed_readout_hop > 0:
        raise ValueError(
            "--gdn2_address_mode currently forbids compatible FutureSeed readout"
        )
    if args.gdn2_cross_layer_init != "independent" and args.backbone != "gdn2":
        raise ValueError("--gdn2_cross_layer_init requires --backbone gdn2")
    if args.gdn2_cross_layer_init != "independent" and not args.fla_strict_official:
        raise ValueError("--gdn2_cross_layer_init requires --fla_strict_official")
    if args.gdn2_cross_layer_init != "independent" and args.gdn2_address_mode != "none":
        raise ValueError(
            "--gdn2_cross_layer_init cannot be mixed with --gdn2_address_mode"
        )
    if args.raven_num_slots < 0 or args.raven_topk < 0:
        raise ValueError("--raven_num_slots and --raven_topk must be non-negative")
    if args.backbone == "raven":
        if not args.fla_strict_official:
            raise ValueError("Raven requires --fla_strict_official")
        if args.gdn_use_short_conv:
            raise ValueError("Official Raven requires --gdn_use_short_conv 0")
        if args.activation_checkpoint:
            raise ValueError(
                "Raven activation checkpointing is disabled until route RNG replay is audited"
            )
    elif args.raven_num_slots or args.raven_topk:
        raise ValueError("Raven slot arguments require --backbone raven")
    configure_sudoku(args.size, args.box_rows, args.box_cols)
    if args.layers < 2:
        raise ValueError("--layers must be at least 2 for FutureSeed.")
    if args.gdn_progressive_base_expand_v < 0:
        raise ValueError("--gdn_progressive_base_expand_v must be non-negative.")
    if args.gdn_progressive_base_expand_v > 0:
        if args.backbone != "gdn":
            raise ValueError("Progressive GDN state expansion requires --backbone gdn.")
        if args.gdn_use_short_conv:
            raise ValueError("Progressive GDN state expansion requires --gdn_use_short_conv 0.")
        if not math.isclose(
            args.gdn_expand_v,
            2.0 * args.gdn_progressive_base_expand_v,
            rel_tol=0.0,
            abs_tol=1e-9,
        ):
            raise ValueError("Progressive GDN state expansion requires expand_v == 2 * base expand_v.")
    if args.backbone == "gdn":
        if args.gdn_mode in {"chunk", "naive_recurrent"}:
            ok, reason = fla_gdn_available()
            if not ok:
                raise ValueError(f"--backbone gdn is unavailable: {reason}")
        elif args.gdn_mode == "triton_recurrent":
            ok, reason = gdn_triton_available()
            if not ok:
                raise ValueError(f"--gdn_mode triton_recurrent is unavailable: {reason}")
            if args.gdn_use_short_conv:
                ok, reason = fla_gdn_available()
                if not ok:
                    raise ValueError(f"--gdn_use_short_conv 1 is unavailable: {reason}")
        else:
            raise ValueError("--gdn_mode must be chunk, naive_recurrent, or triton_recurrent.")
        if args.gdn_expand_v <= 0:
            raise ValueError("--gdn_expand_v must be positive.")
        if args.gdn_conv_size < 1:
            raise ValueError("--gdn_conv_size must be positive.")
    if args.backbone in {"fla_gdn", "gdn2", "kda", "raven", "momentum"}:
        if args.gdn_mode != "chunk":
            raise ValueError(f"--backbone {args.backbone} requires --gdn_mode chunk during training.")
        ok, reason = fla_delta_available(args.backbone)
        if not ok:
            raise ValueError(f"--backbone {args.backbone} is unavailable: {reason}")
        if args.gdn_expand_v <= 0:
            raise ValueError("--gdn_expand_v must be positive.")
        if args.gdn_conv_size < 1:
            raise ValueError("--gdn_conv_size must be positive.")
    if args.fla_strict_official and args.backbone not in {
        "fla_gdn",
        "gdn2",
        "kda",
        "raven",
        "momentum",
    }:
        raise ValueError(
            "--fla_strict_official is valid only for fla_gdn, gdn2, kda, "
            "raven, or momentum"
        )
    if args.backbone == "rwkv7":
        if args.rwkv_kernel not in {"statepassing", "torch"}:
            raise ValueError("--backbone rwkv7 requires an explicit statepassing or torch kernel; fallback is forbidden")
        if args.rwkv_kernel == "statepassing":
            ok, reason = statepassing_available(args.head_dim)
            if not ok:
                raise ValueError(f"--backbone rwkv7 statepassing kernel is unavailable: {reason}")
    if args.backbone == "rwkv" and args.rwkv_kernel in {"cuda", "statepassing"}:
        ok, reason = statepassing_available(args.head_dim)
        if not ok:
            raise ValueError(f"--rwkv_kernel {args.rwkv_kernel} is unavailable: {reason}")
    if args.backbone == "rwkv" and args.rwkv_kernel == "wind":
        ok, reason = wind_available(args.head_dim)
        if not ok:
            raise ValueError(f"--rwkv_kernel wind is unavailable: {reason}")
    if args.hole_pattern != "random":
        raise ValueError("--hole_pattern now supports only 'random'; structured hole probes were removed from the clean mainline.")
    if official_sudoku_enabled(args) and N != 9:
        raise ValueError("--official_sudoku_data_dir currently supports only 9x9 official EqR Sudoku arrays.")

    device = choose_device(args.cpu)
    if args.forward_dtype == "bfloat16" and device.type != "cuda":
        raise ValueError("--forward_dtype=bfloat16 requires CUDA; CPU/MPS fallback is disabled for this run")
    print(
        f"device={device} torch={torch.__version__} board={N}x{N} box={BOX_ROWS}x{BOX_COLS} "
        f"mainline=future_seed_loop backbone={args.backbone} rwkv_kernel={args.rwkv_kernel} "
        f"future_seed_gate_mode={args.future_seed_gate_mode} future_seed_scope={args.future_seed_scope} "
        f"future_seed_readout_hop={args.future_seed_readout_hop} "
        f"future_seed_content_mode={args.future_seed_content_mode} "
        f"momentum_future_seed_transport={args.momentum_future_seed_transport} "
        f"future_seed_gradient_mode={args.future_seed_gradient_mode} "
        f"gdn_mode={args.gdn_mode} gdn_progressive_base_expand_v={args.gdn_progressive_base_expand_v} "
        f"gdn2_gain_budget={args.gdn2_gain_budget_mode} "
        f"gdn2_fast_slow_decay={args.gdn2_fast_slow_decay_mode} "
        f"gdn2_precondition={args.gdn2_precondition_mode} "
        f"gdn2_address={args.gdn2_address_mode} "
        f"gdn2_update={args.gdn2_update_mode} "
        f"gdn2_state_expert={args.gdn2_state_expert_mode} "
        f"gdn2_cross_layer_init={args.gdn2_cross_layer_init} "
        f"raven_slots/topk={args.raven_num_slots}/{args.raven_topk} "
        f"cell_order_train={args.cell_order_train} "
        f"forward_dtype={args.forward_dtype}",
        flush=True,
    )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    official_eval = (
        OfficialSudokuDataset(Path(args.official_sudoku_data_dir), args.official_sudoku_eval_split)
        if official_sudoku_enabled(args)
        else None
    )

    model, train_stats = train_model(args, device=device)
    checkpoint_evals = train_stats.pop("checkpoint_evals", {})
    batch = make_eval_batch(
        args,
        official_eval,
        args.eval_n,
        args.eval_holes,
        args.seed + int(args.official_eval_seed_offset),
        device=device,
    )
    inputs, labels, clue_mask = batch
    clean, loop_preds = evaluate_model(
        model,
        batch,
        max_loops=args.max_loops,
        noise_scale=0.0,
        seed=args.seed + 4000,
        forward_dtype=args.forward_dtype,
    )
    metrics: Dict[str, Any] = {
        "train": train_stats,
        "eval_clean": clean,
        "eval_by_holes": {},
        "checkpoint_evals": checkpoint_evals,
    }
    if args.noise_scale > 0:
        noisy, _noisy_preds = evaluate_model(
            model,
            batch,
            max_loops=args.max_loops,
            noise_scale=args.noise_scale,
            seed=args.seed + 5000,
            forward_dtype=args.forward_dtype,
        )
        metrics["eval_noisy"] = noisy

    rollout_ks = parse_rollout_ks(args)
    if rollout_ks:
        rollout_noise = args.noise_scale if args.rollout_noise_scale < 0 else args.rollout_noise_scale
        metrics["rollouts"] = evaluate_rollouts(
            model,
            batch,
            max_loops=args.max_loops,
            noise_scale=rollout_noise,
            ks=rollout_ks,
            seed=args.seed + 6000,
            forward_dtype=args.forward_dtype,
        )
        rollout_loop_values = parse_rollout_loop_values(args)
        if rollout_loop_values:
            metrics["rollouts_by_loop"] = evaluate_rollouts_by_loop(
                model,
                batch,
                loop_values=rollout_loop_values,
                noise_scale=rollout_noise,
                ks=rollout_ks,
                seed=args.seed + 6500,
                forward_dtype=args.forward_dtype,
            )

    eval_holes_values = parse_eval_holes(args)
    official_eval_blank_ranges = parse_official_eval_blank_ranges(args.official_eval_blank_ranges)
    if official_eval is None:
        for holes in eval_holes_values:
            if holes == args.eval_holes:
                metrics["eval_by_holes"][f"holes{holes}"] = {"eval_clean": clean}
                continue
            extra_batch = make_eval_batch(
                args,
                None,
                args.eval_n,
                holes,
                args.seed + 999 + holes * 17,
                device=device,
            )
            hole_clean, _hole_preds = evaluate_model(
                model,
                extra_batch,
                max_loops=args.max_loops,
                noise_scale=0.0,
                seed=args.seed + 4000 + holes,
                forward_dtype=args.forward_dtype,
            )
            metrics["eval_by_holes"][f"holes{holes}"] = {"eval_clean": hole_clean}
    else:
        metrics["eval_official"] = {
            "split": args.official_sudoku_eval_split,
            "eval_n": args.eval_n,
            "seed": args.seed + int(args.official_eval_seed_offset),
        }
        if official_eval_blank_ranges:
            metrics["official_eval_by_blank_range"] = {}
            for offset, (label, lo, hi) in enumerate(official_eval_blank_ranges):
                range_batch = official_eval.fixed_batch_by_blank_range(
                    args.eval_n,
                    args.seed + int(args.official_eval_seed_offset) + 17000 + offset * 997 + lo * 37 + hi,
                    holes_min=lo,
                    holes_max=hi,
                    device=device,
                )
                range_clean, _range_preds = evaluate_model(
                    model,
                    range_batch,
                    max_loops=args.max_loops,
                    noise_scale=0.0,
                    seed=args.seed + 9000 + offset,
                    forward_dtype=args.forward_dtype,
                )
                actual_n = int(range_batch[0].shape[0])
                metrics["official_eval_by_blank_range"][label] = {
                    "blank_range": [lo, hi],
                    "eval_n": actual_n,
                    "eval_clean": range_clean,
                }

    l1 = metrics["eval_clean"]["loop1"]["label_exact"]
    last_key = f"loop{args.max_loops}"
    l_last = metrics["eval_clean"][last_key]["label_exact"]
    if l_last > l1:
        decision = "Depth loop adds useful full-board refinement."
    elif l_last == l1:
        decision = "Depth loop is neutral on full-board exact in this run."
    else:
        decision = "Depth loop is not yet stable; later loops reduce full-board exact."
    metrics["decision"] = decision
    metrics["task"] = {
        "size": N,
        "box_rows": BOX_ROWS,
        "box_cols": BOX_COLS,
        "hole_pattern": args.hole_pattern,
        "hole_stages": parse_hole_stages(args),
        "data_source": "official_sudoku" if official_eval is not None else "generated_random_holes",
        "official_sudoku_data_dir": str(args.official_sudoku_data_dir) if official_eval is not None else "",
        "official_sudoku_train_split": str(args.official_sudoku_train_split) if official_eval is not None else "",
        "official_sudoku_eval_split": str(args.official_sudoku_eval_split) if official_eval is not None else "",
        "official_train_blank_summary": train_stats.get("official_train_blank_summary", {}),
        "official_eval_blank_summary": train_stats.get("official_eval_blank_summary", {}),
        "official_eval_blank_ranges": [
            {"label": label, "min": lo, "max": hi} for label, lo, hi in official_eval_blank_ranges
        ],
        "official_eval_seed_offset": int(args.official_eval_seed_offset),
        "max_loops": args.max_loops,
        "loop_loss": args.loop_loss,
        "loop_loss_start": args.loop_loss_start,
        "loop_loss_power": args.loop_loss_power,
        "loop_loss_min_weight": args.loop_loss_min_weight,
        "exact_margin_weight": args.exact_margin_weight,
        "exact_margin_tau": args.exact_margin_tau,
        "exact_margin_target": args.exact_margin_target,
        "exact_margin_start_step": args.exact_margin_start_step,
        "noise_mode": "feature_diff",
        "feature_buffer_size": args.feature_buffer_size,
        "future_seed_scale": args.future_seed_scale,
        "future_seed_decay": args.future_seed_decay,
        "future_seed_update": args.future_seed_update,
        "future_seed_norm_mode": args.future_seed_norm_mode,
        "future_seed_gate_mode": args.future_seed_gate_mode,
        "future_seed_scope": args.future_seed_scope,
        "future_seed_readout_hop": args.future_seed_readout_hop,
        "future_seed_content_mode": args.future_seed_content_mode,
        "momentum_future_seed_transport": args.momentum_future_seed_transport,
        "future_seed_gradient_mode": args.future_seed_gradient_mode,
        "loop_update_mode": args.loop_update_mode,
        "loop_update_gate_init": args.loop_update_gate_init,
        "loop_feedback_scale": args.loop_feedback_scale,
        "loop_feedback_detach": bool(args.loop_feedback_detach),
        "loop_feedback_corrupt_prob": args.loop_feedback_corrupt_prob,
        "loop_feedback_corrupt_mix": args.loop_feedback_corrupt_mix,
        "loop_feedback_corrupt_mode": args.loop_feedback_corrupt_mode,
        "loop_time_scale": args.loop_time_scale,
        "scratch_mode": args.scratch_mode,
        "scratch_scale": args.scratch_scale,
        "scratch_noise_scale": args.scratch_noise_scale,
        "scratch_gauss_weight": args.scratch_gauss_weight,
        "scratch_gauss_projections": args.scratch_gauss_projections,
        "scratch_gate_bias": args.scratch_gate_bias,
        "scratch_decay_bias": args.scratch_decay_bias,
        "hidden_agg_noise_scale": args.hidden_agg_noise_scale,
        "hidden_agg_noise_temp": args.hidden_agg_noise_temp,
        "hidden_agg_noise_detach": bool(args.hidden_agg_noise_detach),
        "hidden_agg_noise_mode": args.hidden_agg_noise_mode,
        "hidden_agg_noise_topk": args.hidden_agg_noise_topk,
        "hidden_agg_noise_max_norm": args.hidden_agg_noise_max_norm,
        "rwkv_kernel": args.rwkv_kernel,
        "backbone": args.backbone,
        "gdn_mode": args.gdn_mode,
        "gdn_expand_v": args.gdn_expand_v,
        "gdn_progressive_base_expand_v": args.gdn_progressive_base_expand_v,
        "gdn_use_short_conv": bool(args.gdn_use_short_conv),
        "gdn_conv_size": args.gdn_conv_size,
        "gdn_allow_neg_eigval": bool(args.gdn_allow_neg_eigval),
        "gdn2_gain_budget_mode": args.gdn2_gain_budget_mode,
        "gdn2_gain_budget_sigma_cap": args.gdn2_gain_budget_sigma_cap,
        "gdn2_gain_budget_step_cap": args.gdn2_gain_budget_step_cap,
        "gdn2_gain_budget_sigma_cap_max": args.gdn2_gain_budget_sigma_cap_max,
        "gdn2_gain_budget_infeasible_policy": args.gdn2_gain_budget_infeasible_policy,
        "gdn2_fast_slow_decay_mode": args.gdn2_fast_slow_decay_mode,
        "gdn2_fast_slow_decay_kernel_size": args.gdn2_fast_slow_decay_kernel_size,
        "gdn2_fast_slow_decay_rho_init": args.gdn2_fast_slow_decay_rho_init,
        "gdn2_fast_slow_decay_current_weight_init": (
            args.gdn2_fast_slow_decay_current_weight_init
        ),
        "gdn2_precondition_mode": args.gdn2_precondition_mode,
        "gdn2_address_mode": args.gdn2_address_mode,
        "gdn2_update_mode": args.gdn2_update_mode,
        "gdn2_bi_axis_value_decay_groups": (
            8
            if args.gdn2_update_mode
            in {"bi_axis_value_decay", "gauge_balanced_bi_axis"}
            else 0
        ),
        "gdn2_bi_axis_value_decay_potential_cap": (
            math.log(4.0)
            if args.gdn2_update_mode == "gauge_balanced_bi_axis"
            else 0.0
        ),
        "gdn2_raven_routed_slots": (
            8 if args.gdn2_update_mode == "raven_routed_gdn" else 0
        ),
        "gdn2_state_expert_mode": args.gdn2_state_expert_mode,
        "gdn2_cross_layer_init": args.gdn2_cross_layer_init,
        "raven_num_slots": args.raven_num_slots,
        "raven_topk": args.raven_topk,
        "cell_order_train": args.cell_order_train,
        "fla_strict_official": bool(args.fla_strict_official),
        "fla_runtime": train_stats.get("fla_runtime", {"strict": False}),
        "forward_dtype": args.forward_dtype,
        "rollout_ks": rollout_ks,
        "rollout_loop_values": parse_rollout_loop_values(args),
        "rollout_noise_scale": args.noise_scale if args.rollout_noise_scale < 0 else args.rollout_noise_scale,
        "case_bank_holes": args.case_bank_holes,
        "case_bank_n": args.case_bank_n,
        "case_bank_eval_n": args.case_bank_eval_n,
        "case_bank_loop_values": args.case_bank_loop_values,
        "eval_checkpoint_steps": train_stats.get("eval_checkpoint_steps", []),
        "eval_checkpoint_holes": train_stats.get("eval_checkpoint_holes", []),
        "mainline": "future_seed_loop",
    }

    case_index = min(max(int(args.case_index), 0), inputs.shape[0] - 1)
    html_path = out_dir / f"futureseed_loop_case_seed{args.seed}.html"
    model_label = BACKBONE_DISPLAY_NAMES[args.backbone]
    future_seed_enabled = args.future_seed_scale > 0
    mechanism = "FutureSeed" if future_seed_enabled else "No-FutureSeed"
    write_case_html(
        html_path,
        title=f"{N}x{N} {model_label} {mechanism} loop case",
        model_label=model_label,
        future_seed_enabled=future_seed_enabled,
        inputs=inputs[case_index],
        labels=labels[case_index],
        clue_mask=clue_mask[case_index],
        loop_preds=[pred[case_index] for pred in loop_preds],
    )
    artifacts = {"case_html": str(html_path.resolve())}
    case_bank_holes = parse_int_csv(args.case_bank_holes, name="--case_bank_holes", low=1, high=CELLS)
    case_bank_loop_values = parse_int_csv(
        args.case_bank_loop_values,
        name="--case_bank_loop_values",
        low=1,
        high=args.max_loops,
    )
    if not case_bank_loop_values:
        case_bank_loop_values = sorted({1, min(2, args.max_loops), min(3, args.max_loops), args.max_loops})
    elif args.max_loops not in case_bank_loop_values:
        case_bank_loop_values = sorted(set(case_bank_loop_values + [args.max_loops]))
    if args.case_bank_n > 0 and (case_bank_holes or official_eval is not None):
        case_bank = export_case_bank(
            model,
            out_dir,
            holes_values=case_bank_holes,
            official_eval=official_eval,
            official_blank_ranges=official_eval_blank_ranges,
            eval_n=args.case_bank_eval_n,
            cases_per_kind=args.case_bank_n,
            loop_values=case_bank_loop_values,
            seed=args.seed,
            forward_dtype=args.forward_dtype,
        )
        if case_bank:
            metrics["case_bank"] = case_bank
            artifacts["case_bank_root"] = case_bank["case_bank_root"]
    metrics["artifacts"] = artifacts

    json_path = out_dir / f"futureseed_loop_seed{args.seed}.json"
    md_path = out_dir / f"futureseed_loop_seed{args.seed}.md"
    json_path.write_text(json.dumps({"args": vars(args), "metrics": metrics}, indent=2), encoding="utf-8")
    write_report(md_path, metrics, artifacts)

    print("\nfull-board metrics")
    for loop in range(1, args.max_loops + 1):
        loop_key = f"loop{loop}"
        print(f"loop={loop} {metric_line(metrics['eval_clean'][loop_key])}")
        print(f"        {fs_line(metrics['eval_clean'][f'{loop_key}/future_seed'])}")
    if "eval_noisy" in metrics:
        print(f"eval_noise loop={args.max_loops} {metric_line(metrics['eval_noisy'][last_key])}")
    if metrics.get("rollouts"):
        print("\nstochastic rollout metrics")
        for key, row in metrics["rollouts"].items():
            print(
                f"{key} oracle_exact={row['oracle_label_exact']:.4f} "
                f"oracle_solved={row['oracle_solved_valid_clue']:.4f} "
                f"disagree={row['trajectory_token_disagreement']:.4f}"
            )
            print(f"  confidence {metric_line(row['selector_confidence'])}; gap={row['selector_gap_confidence']:.4f}")
            print(f"  consistency {metric_line(row['selector_consistency'])}; gap={row['selector_gap_consistency']:.4f}")
            print(f"  residual    {metric_line(row['selector_residual'])}; gap={row['selector_gap_residual']:.4f}")
            print(f"  majority    {metric_line(row['selector_majority'])}; gap={row['selector_gap_majority']:.4f}")
    if metrics.get("rollouts_by_loop"):
        print("\ncompute-scaling rollout metrics")
        for loop_key, by_k in metrics["rollouts_by_loop"].items():
            print(loop_key)
            for key, row in by_k.items():
                print(
                    f"  {key} oracle_exact={row['oracle_label_exact']:.4f} "
                    f"confidence_exact={row['selector_confidence']['label_exact']:.4f} "
                    f"residual_exact={row['selector_residual']['label_exact']:.4f} "
                    f"disagree={row['trajectory_token_disagreement']:.4f}"
                )
    if official_eval is None and len(eval_holes_values) > 1:
        print("\nhole-transfer metrics")
        for holes in eval_holes_values:
            loop_last = metrics["eval_by_holes"][f"holes{holes}"]["eval_clean"][last_key]
            print(f"holes={holes:<2d} {metric_line(loop_last)}; {coupling_line(loop_last, holes)}")
    if official_eval is not None:
        print(
            f"\nofficial sudoku eval split={args.official_sudoku_eval_split} "
            f"eval_n={args.eval_n} seed={args.seed + int(args.official_eval_seed_offset)}"
        )
        if metrics.get("official_eval_by_blank_range"):
            print("\nofficial blank-range metrics")
            for range_key, row in metrics["official_eval_by_blank_range"].items():
                lo, hi = row["blank_range"]
                loop_last = row["eval_clean"][last_key]
                print(f"{range_key} blanks={lo}-{hi} n={row['eval_n']} {metric_line(loop_last)}")
    if metrics.get("case_bank"):
        print("\ncase-bank artifacts")
        for hole_key, row in metrics["case_bank"]["holes"].items():
            summary = row["summary"]
            print(
                f"{hole_key} exact={summary['final_exact']:.4f} "
                f"blank_acc={summary['final_blank_acc']:.4f} selected={summary['selected_counts']}"
            )
            print(f"  index={row['index_html']}")
            print(f"  cases={row['cases_json']}")
    print(metrics["decision"])
    print(f"wrote {json_path}")
    print(f"wrote {md_path}")
    print(f"wrote {html_path}")
    return metrics


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--size", type=int, default=9)
    p.add_argument("--box_rows", type=int, default=0)
    p.add_argument("--box_cols", type=int, default=0)
    p.add_argument("--steps", type=int, default=400)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--grad_accum_steps", type=int, default=1)
    p.add_argument("--d_model", type=int, default=48)
    p.add_argument("--layers", type=int, default=4)
    p.add_argument("--heads", type=int, default=4)
    p.add_argument("--head_dim", type=int, default=12)
    p.add_argument("--channel_mult", type=int, default=2)
    p.add_argument("--l_cycles", type=int, default=2)
    p.add_argument("--max_loops", type=int, default=3)
    p.add_argument("--lambda_", type=float, default=0.95)
    p.add_argument("--loop_update_mode", choices=("fixed", "learned_gate"), default="fixed")
    p.add_argument("--loop_update_gate_init", type=float, default=0.95)
    p.add_argument("--future_seed_scale", type=float, default=1.0)
    p.add_argument("--future_seed_decay", type=float, default=0.0)
    p.add_argument(
        "--future_seed_update",
        choices=("fixed", "learned", "loop_residual", "loop_secant"),
        default="fixed",
    )
    p.add_argument("--future_seed_norm_mode", choices=("unit", "adaptive_rms"), default="unit")
    p.add_argument(
        "--future_seed_gate_mode",
        choices=FUTURE_SEED_GATE_MODES,
        default="head",
    )
    p.add_argument("--future_seed_scope", choices=("layer", "block"), default="layer")
    p.add_argument("--future_seed_readout_hop", type=int, default=0)
    p.add_argument(
        "--future_seed_content_mode",
        choices=FUTURE_SEED_CONTENT_MODES,
        default="terminal",
    )
    p.add_argument(
        "--momentum_future_seed_transport",
        choices=("full_state", "momentum_only"),
        default="full_state",
    )
    p.add_argument(
        "--future_seed_gradient_mode",
        choices=FUTURE_SEED_GRADIENT_MODES,
        default="canonical",
    )
    p.add_argument("--loop_feedback_scale", type=float, default=0.0)
    p.add_argument("--loop_feedback_detach", type=int, choices=(0, 1), default=0)
    p.add_argument("--loop_feedback_corrupt_prob", type=float, default=0.0)
    p.add_argument("--loop_feedback_corrupt_mix", type=float, default=0.0)
    p.add_argument("--loop_feedback_corrupt_mode", choices=("random_token", "uniform"), default="random_token")
    p.add_argument("--loop_time_scale", type=float, default=0.0)
    p.add_argument("--scratch_mode", choices=("none", "gated"), default="none")
    p.add_argument("--scratch_scale", type=float, default=1.0)
    p.add_argument("--scratch_noise_scale", type=float, default=0.0)
    p.add_argument("--scratch_gauss_weight", type=float, default=0.0)
    p.add_argument("--scratch_gauss_projections", type=int, default=0)
    p.add_argument("--scratch_gate_bias", type=float, default=-2.0)
    p.add_argument("--scratch_decay_bias", type=float, default=2.0)
    p.add_argument("--hidden_agg_noise_scale", type=float, default=0.0)
    p.add_argument("--hidden_agg_noise_temp", type=float, default=1.0)
    p.add_argument("--hidden_agg_noise_detach", type=int, choices=(0, 1), default=1)
    p.add_argument("--hidden_agg_noise_mode", choices=("gumbel", "soft_group"), default="gumbel")
    p.add_argument("--hidden_agg_noise_topk", type=int, default=8)
    p.add_argument("--hidden_agg_noise_max_norm", type=float, default=0.0)
    p.add_argument("--exact_margin_weight", type=float, default=0.0)
    p.add_argument("--exact_margin_tau", type=float, default=0.5)
    p.add_argument("--exact_margin_target", type=float, default=0.0)
    p.add_argument("--exact_margin_start_step", type=int, default=0)
    p.add_argument("--activation_checkpoint", action="store_true")
    p.add_argument("--resume_train_checkpoint", default="")
    p.add_argument("--resume_train_checkpoint_sha256", default="")
    p.add_argument("--resume_train_source_sha", default="")
    p.add_argument("--resume_require_exact_state", action="store_true")
    p.add_argument(
        "--resume_allow_future_seed_content_upgrade",
        action="store_true",
    )
    p.add_argument(
        "--resume_allow_future_seed_update_upgrade",
        action="store_true",
    )
    p.add_argument(
        "--resume_allow_future_seed_gradient_upgrade",
        action="store_true",
    )
    p.add_argument(
        "--resume_allow_gdn2_update_upgrade",
        action="store_true",
    )
    p.add_argument(
        "--resume_allow_gdn2_state_expert_upgrade",
        action="store_true",
    )
    p.add_argument("--train_checkpoint_dir", default="")
    p.add_argument("--save_train_checkpoint_every", type=int, default=0)
    p.add_argument("--forward_dtype", choices=("float32", "bfloat16"), default="float32")
    p.add_argument(
        "--backbone",
        choices=(
            "rwkv",
            "rwkv7",
            "gdn",
            "fla_gdn",
            "gdn2",
            "kda",
            "raven",
            "momentum",
        ),
        default="rwkv",
    )
    p.add_argument("--rwkv_kernel", choices=("auto", "torch", "cuda", "statepassing", "wind"), default="auto")
    p.add_argument("--gdn_mode", choices=("chunk", "naive_recurrent", "triton_recurrent"), default="chunk")
    p.add_argument("--gdn_expand_v", type=float, default=1.0)
    p.add_argument("--gdn_progressive_base_expand_v", type=float, default=0.0)
    p.add_argument("--gdn_use_short_conv", type=int, choices=(0, 1), default=1)
    p.add_argument("--gdn_conv_size", type=int, default=4)
    p.add_argument("--gdn_allow_neg_eigval", action="store_true")
    p.add_argument(
        "--gdn2_gain_budget_mode",
        choices=(
            "none",
            "external_identity",
            "fixed_sigma",
            "decay_funded",
        ),
        default="none",
    )
    p.add_argument("--gdn2_gain_budget_sigma_cap", type=float, default=1.10)
    p.add_argument("--gdn2_gain_budget_step_cap", type=float, default=1.0)
    p.add_argument("--gdn2_gain_budget_sigma_cap_max", type=float, default=3.0)
    p.add_argument(
        "--gdn2_gain_budget_infeasible_policy",
        choices=("raise", "relax"),
        default="raise",
    )
    p.add_argument(
        "--gdn2_fast_slow_decay_mode",
        choices=FAST_SLOW_DECAY_MODES,
        default="none",
    )
    p.add_argument("--gdn2_fast_slow_decay_kernel_size", type=int, default=4)
    p.add_argument("--gdn2_fast_slow_decay_rho_init", type=float, default=0.10)
    p.add_argument(
        "--gdn2_fast_slow_decay_current_weight_init",
        type=float,
        default=0.85,
    )
    p.add_argument(
        "--gdn2_precondition_mode",
        choices=GDN2_PRECONDITION_MODES,
        default="none",
    )
    p.add_argument(
        "--gdn2_address_mode",
        choices=GDN2_ADDRESS_MODES,
        default="none",
    )
    p.add_argument(
        "--gdn2_update_mode",
        choices=GDN2_UPDATE_MODES,
        default="none",
    )
    p.add_argument(
        "--gdn2_state_expert_mode",
        choices=GDN2_STATE_EXPERT_MODES,
        default="none",
    )
    p.add_argument(
        "--gdn2_cross_layer_init",
        choices=GDN2_CROSS_LAYER_INIT_MODES,
        default="independent",
    )
    p.add_argument(
        "--raven_num_slots",
        type=int,
        default=0,
        help="Raven slots per head; 0 exactly matches GDN2 state elements.",
    )
    p.add_argument(
        "--raven_topk",
        type=int,
        default=0,
        help="Raven write slots per token; 0 uses 12.5 percent occupancy.",
    )
    p.add_argument(
        "--cell_order_train",
        choices=CELL_ORDER_TRAIN_MODES,
        default="row_major",
    )
    p.add_argument("--fla_strict_official", action="store_true")
    p.add_argument("--loop_loss", choices=("final", "all", "shaped", "delayed"), default="final")
    p.add_argument("--loop_loss_start", type=int, default=1)
    p.add_argument("--loop_loss_power", type=float, default=2.0)
    p.add_argument("--loop_loss_min_weight", type=float, default=0.05)
    p.add_argument("--noise_scale", type=float, default=0.0)
    p.add_argument("--feature_buffer_size", type=int, default=8192)
    p.add_argument("--feature_buffer_add", type=int, default=2048)
    p.add_argument("--rollout_ks", default="")
    p.add_argument("--rollout_loop_values", default="")
    p.add_argument("--rollout_noise_scale", type=float, default=-1.0)
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--weight_decay", type=float, default=1e-3)
    p.add_argument("--optimizer_contract", choices=("uniform", "rwkv7_decay_groups"), default="uniform")
    p.add_argument("--shared_shell_init_seed", type=int, default=-1)
    p.add_argument("--holes_min", type=int, default=4)
    p.add_argument("--holes_max", type=int, default=12)
    p.add_argument("--hole_stages", default="")
    p.add_argument("--hole_pattern", choices=("random",), default="random")
    p.add_argument("--eval_holes", type=int, default=8)
    p.add_argument("--eval_holes_list", default="")
    p.add_argument("--eval_n", type=int, default=128)
    p.add_argument("--official_sudoku_data_dir", default="")
    p.add_argument("--official_sudoku_train_split", default="train")
    p.add_argument("--official_sudoku_train_indices", default="")
    p.add_argument("--official_sudoku_eval_split", default="test")
    p.add_argument("--official_eval_seed_offset", type=int, default=999)
    p.add_argument("--official_eval_blank_ranges", default="")
    p.add_argument("--eval_checkpoint_steps", default="")
    p.add_argument("--eval_checkpoint_stage_offsets", default="")
    p.add_argument("--eval_checkpoint_holes_list", default="")
    p.add_argument("--blank_loss_weight", type=float, default=8.0)
    p.add_argument("--case_index", type=int, default=0)
    p.add_argument("--case_bank_holes", default="")
    p.add_argument("--case_bank_n", type=int, default=0)
    p.add_argument("--case_bank_eval_n", type=int, default=256)
    p.add_argument("--case_bank_loop_values", default="")
    p.add_argument("--seed", type=int, default=52)
    p.add_argument("--log_every", type=int, default=100)
    p.add_argument("--out_dir", default="runs/mainline")
    p.add_argument("--cpu", action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
