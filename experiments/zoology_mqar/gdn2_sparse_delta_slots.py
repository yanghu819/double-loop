from __future__ import annotations

import hashlib
import math
from functools import partial
from typing import Any, Optional

import torch
import torch.nn as nn
from einops import rearrange
from torch.nn import functional as F

from fla.layers.utils import get_layer_cache, update_layer_cache
from fla.ops.gdn2 import chunk_gdn2
from fla.ops.gsa import chunk_gsa
from zoology.config import ModelConfig
from zoology.model import _compute_state_size, _init_weights

from experiments.zoology_mqar.gdn2_committed_delta import capture_committed_edit
from experiments.zoology_mqar.gdn2_futureseed import (
    FutureSeedLMBackbone,
    ZoologyGDN2FutureSeedMixer,
)


SLOT_COUNT = 16
MAIN_STATE_VALUES = 4 * 32 * 32
SPARSE_STATE_VALUES = 4 * (32 * SLOT_COUNT + SLOT_COUNT * 32)
ANCHOR_PARAMETERS_PER_LAYER = 4 * SLOT_COUNT * 32
LOCAL_GATE_PARAMETERS_PER_LAYER = 4
SEED_GATE_PARAMETERS = 4
EXPECTED_NEW_PARAMETERS = (
    2 * (ANCHOR_PARAMETERS_PER_LAYER + LOCAL_GATE_PARAMETERS_PER_LAYER)
    + SEED_GATE_PARAMETERS
)


def _rms(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.float().square().mean().sqrt()


def _normalize_last(tensor: torch.Tensor) -> torch.Tensor:
    rms = tensor.float().square().mean(dim=-1, keepdim=True).add(1e-6).sqrt()
    return tensor / rms.to(dtype=tensor.dtype)


class SparseDeltaSlotGDN2(nn.Module):
    """Native GDN2 plus a sparse factorized state of exact committed edits."""

    def __init__(self, base: nn.Module) -> None:
        super().__init__()
        if base.hidden_size != 128:
            raise ValueError("P-GDN3-050 requires D128")
        if base.num_heads != 4 or base.num_v_heads != 4:
            raise ValueError("P-GDN3-050 requires H4")
        if base.head_k_dim != 32 or base.head_v_dim != 32:
            raise ValueError("P-GDN3-050 requires K32/V32")
        if base.mode != "chunk" or not base.use_short_conv:
            raise ValueError("P-GDN3-050 requires chunk GDN2 with short conv")
        self.base = base
        self.slot_anchors = nn.Parameter(
            torch.empty(base.num_heads, SLOT_COUNT, base.head_k_dim)
        )
        for head in range(base.num_heads):
            nn.init.orthogonal_(self.slot_anchors.data[head])
        self.local_read_logit = nn.Parameter(
            torch.zeros(1, 1, base.num_heads, 1)
        )
        self.seed_read_logit = (
            nn.Parameter(torch.zeros(1, base.num_heads, 1, 1))
            if int(base.layer_idx) > 0
            else None
        )
        self.last_sparse_dense: Optional[torch.Tensor] = None
        self.last_diagnostics: dict[str, torch.Tensor] = {}

    @property
    def num_heads(self) -> int:
        return int(self.base.num_heads)

    @property
    def num_v_heads(self) -> int:
        return int(self.base.num_v_heads)

    @property
    def head_k_dim(self) -> int:
        return int(self.base.head_k_dim)

    @property
    def head_v_dim(self) -> int:
        return int(self.base.head_v_dim)

    @property
    def layer_idx(self) -> int:
        return int(self.base.layer_idx)

    def sparse_route(
        self,
        key: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        key_unit = F.normalize(key.float(), dim=-1, eps=1e-6)
        anchors = F.normalize(self.slot_anchors.float(), dim=-1, eps=1e-6)
        logits = torch.einsum("bthk,hmk->bthm", key_unit, anchors)
        logits = logits * math.sqrt(float(self.head_k_dim))
        probabilities = logits.softmax(dim=-1)
        route_index = probabilities.argmax(dim=-1, keepdim=True)
        hard = torch.zeros_like(probabilities).scatter_(-1, route_index, 1.0)
        route = hard + probabilities - probabilities.detach()
        return route.to(dtype=key.dtype), hard.to(dtype=key.dtype), probabilities

    def make_sparse_seed(
        self,
        main_seed: torch.Tensor,
        sparse_dense: torch.Tensor,
    ) -> torch.Tensor:
        if self.seed_read_logit is None:
            raise RuntimeError("Only receiving layers have a sparse seed gate")
        sparse_rms = sparse_dense.float().square().mean(
            dim=(-1, -2), keepdim=True
        ).sqrt().clamp_min(1e-6)
        gate = torch.tanh(self.seed_read_logit).to(
            device=sparse_dense.device,
            dtype=sparse_dense.dtype,
        )
        sparse_seed = sparse_dense / sparse_rms.to(dtype=sparse_dense.dtype)
        return main_seed + gate * sparse_seed

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        past_key_values: Any = None,
        use_cache: bool | None = False,
        output_attentions: bool | None = False,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, None, Any]:
        del output_attentions
        if attention_mask is not None:
            raise ValueError("P-GDN3-050 uses fixed unpadded MQAR batches")
        if kwargs.get("cu_seqlens") is not None:
            raise ValueError("P-GDN3-050 does not use packed sequences")

        layer = self.base
        q_len = hidden_states.shape[1]
        last_state = get_layer_cache(layer, past_key_values)
        conv_state_q, conv_state_k, conv_state_v = None, None, None
        if last_state is not None:
            conv_state_q, conv_state_k, conv_state_v = last_state["conv_state"]
        q, conv_state_q = layer.q_conv1d(
            x=layer.q_proj(hidden_states),
            cache=conv_state_q,
            output_final_state=use_cache,
        )
        k, conv_state_k = layer.k_conv1d(
            x=layer.k_proj(hidden_states),
            cache=conv_state_k,
            output_final_state=use_cache,
        )
        v, conv_state_v = layer.v_conv1d(
            x=layer.v_proj(hidden_states),
            cache=conv_state_v,
            output_final_state=use_cache,
        )

        g = F.softplus(layer.f_proj(hidden_states).float() + layer.dt_bias)
        erase_gate = layer.b_proj(hidden_states).sigmoid()
        write_gate = layer.w_proj(hidden_states).sigmoid()
        q, k, g, erase_gate = (
            rearrange(tensor, "... (h d) -> ... h d", d=layer.head_k_dim)
            for tensor in (q, k, g, erase_gate)
        )
        v = rearrange(v, "... (h d) -> ... h d", d=layer.head_v_dim)
        write_gate = rearrange(
            write_gate,
            "... (h d) -> ... h d",
            d=layer.head_v_dim,
        )
        g = -layer.A_log.float().exp().unsqueeze(-1) * g
        if layer.allow_neg_eigval:
            raise RuntimeError("P-GDN3-050 requires positive erase gates")

        recurrent_state = (
            last_state["recurrent_state"] if last_state is not None else None
        )
        with capture_committed_edit() as captured:
            main_output, recurrent_state = chunk_gdn2(
                q=q,
                k=k,
                v=v,
                g=g,
                b=erase_gate,
                w=write_gate,
                initial_state=recurrent_state,
                output_final_state=use_cache,
                use_qk_l2norm_in_kernel=True,
            )
        if len(captured) != 1:
            raise RuntimeError(
                f"Expected one exact committed edit, received {len(captured)}"
            )
        committed_edit = captured[0].detach()

        q_unit = F.normalize(q.float(), dim=-1, eps=1e-6).to(dtype=q.dtype)
        k_unit = F.normalize(k.float(), dim=-1, eps=1e-6).to(dtype=k.dtype)
        route, hard_route, route_probabilities = self.sparse_route(k)
        selected_decay = g.float().mean(dim=-1, keepdim=True).clamp(
            min=-20.0,
            max=-1e-4,
        )
        slot_g = (hard_route.float() * selected_decay).to(dtype=q.dtype)
        slot_s = (
            route.float() * (1.0 - selected_decay.exp())
        ).to(dtype=q.dtype)
        sparse_output, sparse_state = chunk_gsa(
            q=q_unit,
            k=k_unit,
            v=committed_edit,
            s=slot_s,
            g=slot_g,
            initial_state=None,
            output_final_state=True,
        )
        if sparse_state is None or len(sparse_state) != 2:
            raise RuntimeError("Official GSA did not return its factorized state")
        key_state, value_state = sparse_state
        sparse_dense = torch.einsum("bhkm,bhmv->bhkv", key_state, value_state)
        self.last_sparse_dense = sparse_dense

        update_layer_cache(
            layer,
            past_key_values,
            recurrent_state=recurrent_state,
            conv_state=(conv_state_q, conv_state_k, conv_state_v),
            offset=q_len,
        )
        local_gate = torch.tanh(self.local_read_logit).to(
            dtype=main_output.dtype
        )
        output = main_output + local_gate * _normalize_last(sparse_output)

        with torch.no_grad():
            usage = hard_route.float().mean(dim=(0, 1, 2))
            usage_entropy = -(
                usage * usage.clamp_min(1e-12).log()
            ).sum() / math.log(float(SLOT_COUNT))
            key_columns = F.normalize(
                key_state.float().transpose(-1, -2), dim=-1, eps=1e-6
            )
            gram = key_columns @ key_columns.transpose(-1, -2)
            eye = torch.eye(SLOT_COUNT, device=gram.device, dtype=torch.bool)
            off_diagonal = gram.masked_select(~eye.view(1, 1, SLOT_COUNT, SLOT_COUNT))
            sparse_board = sparse_output.float().square().mean(
                dim=(1, 2, 3)
            ).sqrt()
            dense_board = sparse_dense.float().square().mean(
                dim=(1, 2, 3)
            ).sqrt()
            self.last_diagnostics = {
                "committed_edit_rms": _rms(committed_edit),
                "local_read_gate_abs": local_gate.float().abs().mean(),
                "sparse_output_relative_rms": _rms(sparse_output)
                / _rms(main_output).clamp_min(1e-8),
                "sparse_output_token_std": sparse_output.float()
                .square()
                .mean(dim=-1)
                .sqrt()
                .std(unbiased=False),
                "sparse_output_board_std": sparse_board.std(unbiased=False),
                "sparse_key_state_rms": _rms(key_state),
                "sparse_value_state_rms": _rms(value_state),
                "sparse_dense_rms": _rms(sparse_dense),
                "sparse_dense_board_std": dense_board.std(unbiased=False),
                "route_probability_max_mean": route_probabilities.max(
                    dim=-1
                ).values.mean(),
                "route_probability_token_std": route_probabilities.max(
                    dim=-1
                ).values.std(unbiased=False),
                "slot_usage_entropy": usage_entropy,
                "slot_usage_max": usage.max(),
                "slot_usage_min": usage.min(),
                "active_slots": (usage > 0).sum(),
                "slot_key_abs_cosine_mean": off_diagonal.abs().mean(),
                "slot_key_abs_cosine_max": off_diagonal.abs().max(),
                "slot_s_min": slot_s.float().min(),
                "slot_s_max": slot_s.float().max(),
                "slot_g_max": slot_g.float().max(),
                "slot_g_min": slot_g.float().min(),
            }

        output = layer.o_norm(
            output,
            rearrange(
                layer.g_proj(hidden_states),
                "... (h d) -> ... h d",
                d=layer.head_v_dim,
            ),
        )
        output = rearrange(output, "b t h d -> b t (h d)")
        return layer.o_proj(output), None, past_key_values


class ZoologySparseDeltaSlotFutureSeedMixer(ZoologyGDN2FutureSeedMixer):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.layer = SparseDeltaSlotGDN2(self.layer)
        self.last_sparse_seed_gate: Optional[torch.Tensor] = None
        self.last_sparse_seed_rms: Optional[torch.Tensor] = None

    def make_combined_initial_state(
        self,
        main_terminal: torch.Tensor,
        sparse_dense: torch.Tensor,
    ) -> torch.Tensor:
        main_seed = super().make_initial_state(main_terminal)
        if self.layer.seed_read_logit is None:
            raise RuntimeError("Receiving sparse seed gate is missing")
        sparse_rms = sparse_dense.float().square().mean(
            dim=(-1, -2), keepdim=True
        ).sqrt().clamp_min(1e-6)
        gate = torch.tanh(self.layer.seed_read_logit).to(
            device=sparse_dense.device,
            dtype=sparse_dense.dtype,
        )
        self.last_sparse_seed_gate = gate.detach().float().mean()
        self.last_sparse_seed_rms = sparse_rms.detach().float().mean()
        return self.layer.make_sparse_seed(main_seed, sparse_dense)

    def forward_with_sparse_state(
        self,
        hidden_states: torch.Tensor,
        *,
        initial_state: Optional[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        cache = self._new_cache(initial_state)
        with torch.autocast(
            device_type=hidden_states.device.type,
            dtype=torch.bfloat16,
            enabled=hidden_states.is_cuda,
        ):
            output, _attentions, cache = self.layer(
                hidden_states,
                past_key_values=cache,
                use_cache=True,
            )
        terminal_state = cache[self.layer_idx]["recurrent_state"]
        sparse_dense = self.layer.last_sparse_dense
        if terminal_state is None or sparse_dense is None:
            raise RuntimeError("P-GDN3-050 did not produce both terminal states")
        return output.to(hidden_states.dtype), terminal_state, sparse_dense

    def state_size(self, sequence_length: int = 2048) -> int:
        del sequence_length
        return MAIN_STATE_VALUES + SPARSE_STATE_VALUES


class SparseDeltaSlotBackbone(FutureSeedLMBackbone):
    def layers_forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        residual = None
        main_terminal = None
        sparse_terminal = None
        for layer in self.layers:
            mixer = layer.sequence_mixer
            if not isinstance(mixer, ZoologySparseDeltaSlotFutureSeedMixer):
                raise TypeError("Sparse-delta backbone requires strict mixers")

            dropped = layer.drop_path1(layer.dropout1(hidden_states))
            residual = dropped + residual if residual is not None else dropped
            hidden_states = layer.norm1(
                residual.to(dtype=layer.norm1.weight.dtype)
            )
            initial_state = None
            if main_terminal is not None:
                if sparse_terminal is None:
                    raise RuntimeError("Sparse terminal state was lost")
                initial_state = mixer.make_combined_initial_state(
                    main_terminal,
                    sparse_terminal,
                )
            hidden_states, main_terminal, sparse_terminal = (
                mixer.forward_with_sparse_state(
                    hidden_states,
                    initial_state=initial_state,
                )
            )

            dropped = layer.drop_path2(layer.dropout2(hidden_states))
            residual = dropped + residual if residual is not None else dropped
            hidden_states = layer.norm2(
                residual.to(dtype=layer.norm2.weight.dtype)
            )
            hidden_states = layer.state_mixer(hidden_states)

        dropped = self.drop_f(hidden_states)
        residual = dropped + residual if residual is not None else dropped
        return self.ln_f(residual.to(dtype=self.ln_f.weight.dtype))


class SparseDeltaSlotLanguageModel(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        if config.vocab_size % config.pad_vocab_size_multiple != 0:
            config.vocab_size += config.pad_vocab_size_multiple - (
                config.vocab_size % config.pad_vocab_size_multiple
            )
        if config.multiplier != 1:
            raise ValueError("Discrete FutureSeed model requires multiplier=1")

        self.backbone = SparseDeltaSlotBackbone(config=config)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.apply(
            partial(
                _init_weights,
                n_layers=config.n_layers,
                block_type=config.block_type,
            )
        )
        for block in self.backbone.layers:
            wrapper = block.sequence_mixer.layer
            nn.init.zeros_(wrapper.local_read_logit)
            if wrapper.seed_read_logit is not None:
                nn.init.zeros_(wrapper.seed_read_logit)
        if config.learnable_word_embeddings:
            self.lm_head.weight = self.backbone.embeddings.word_embeddings.weight

    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: Optional[torch.Tensor] = None,
        state=None,
        return_embeddings: bool = False,
    ) -> torch.Tensor:
        del state
        hidden_states = self.backbone(input_ids, position_ids=position_ids)
        if return_embeddings:
            return hidden_states
        return self.lm_head(hidden_states)

    def state_size(self, sequence_length: int) -> int:
        return _compute_state_size(self.backbone.layers, sequence_length)


def _is_extra_parameter(name: str) -> bool:
    return (
        name.endswith(".slot_anchors")
        or name.endswith(".local_read_logit")
        or name.endswith(".seed_read_logit")
    )


def load_matched_parent_state(
    model: nn.Module,
    control_state: dict[str, torch.Tensor],
) -> None:
    loaded: dict[str, torch.Tensor] = {}
    for name, tensor in model.state_dict().items():
        if _is_extra_parameter(name):
            loaded[name] = tensor
            continue
        parent_name = name.replace(
            ".sequence_mixer.layer.base.",
            ".sequence_mixer.layer.",
        )
        if parent_name not in control_state:
            raise RuntimeError(f"Missing matched parent tensor: {parent_name}")
        parent = control_state[parent_name]
        if parent.shape != tensor.shape:
            raise RuntimeError(
                f"Matched tensor shape mismatch for {name}: "
                f"{tuple(parent.shape)} != {tuple(tensor.shape)}"
            )
        loaded[name] = parent
    model.load_state_dict(loaded, strict=True)


def parent_parameter_hash(model: nn.Module) -> str:
    digest = hashlib.sha256()
    rows: list[tuple[str, torch.Tensor]] = []
    for name, parameter in model.named_parameters():
        if _is_extra_parameter(name):
            continue
        parent_name = name.replace(
            ".sequence_mixer.layer.base.",
            ".sequence_mixer.layer.",
        )
        rows.append((parent_name, parameter))
    for name, value in sorted(rows):
        digest.update(name.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


@torch.no_grad()
def sparse_delta_slot_diagnostics(
    model: nn.Module,
    inputs: torch.Tensor,
) -> dict[str, Any]:
    if not isinstance(model.backbone, SparseDeltaSlotBackbone):
        raise TypeError("Expected SparseDeltaSlotBackbone")
    model.eval()(inputs)
    mixers = [block.sequence_mixer for block in model.backbone.layers]
    if len(mixers) != 2 or not all(
        isinstance(mixer, ZoologySparseDeltaSlotFutureSeedMixer)
        for mixer in mixers
    ):
        raise RuntimeError("Expected exactly two sparse-delta-slot mixers")

    rows = []
    for mixer in mixers:
        diagnostics = mixer.layer.last_diagnostics
        if not diagnostics:
            raise RuntimeError("Sparse-delta diagnostics were not populated")
        anchors = F.normalize(
            mixer.layer.slot_anchors.float(), dim=-1, eps=1e-6
        )
        anchor_gram = anchors @ anchors.transpose(-1, -2)
        eye = torch.eye(SLOT_COUNT, device=anchor_gram.device, dtype=torch.bool)
        off_diagonal = anchor_gram.masked_select(
            ~eye.view(1, SLOT_COUNT, SLOT_COUNT)
        )
        rows.append(
            {
                "layer_idx": mixer.layer_idx,
                **{name: float(value.item()) for name, value in diagnostics.items()},
                "anchor_abs_cosine_mean": float(
                    off_diagonal.abs().mean().item()
                ),
                "anchor_abs_cosine_max": float(
                    off_diagonal.abs().max().item()
                ),
                "sparse_seed_gate_abs": (
                    None
                    if mixer.last_sparse_seed_gate is None
                    else float(mixer.last_sparse_seed_gate.abs().item())
                ),
                "sparse_seed_rms": (
                    None
                    if mixer.last_sparse_seed_rms is None
                    else float(mixer.last_sparse_seed_rms.item())
                ),
            }
        )
    return {
        "active_layers": 2,
        "slot_count": SLOT_COUNT,
        "main_state_values_per_layer": MAIN_STATE_VALUES,
        "sparse_factor_state_values_per_layer": SPARSE_STATE_VALUES,
        "official_gdn2_scans_per_layer": 1,
        "official_gsa_scans_per_layer": 1,
        "sparse_futureseed_routes": 1,
        "new_parameters": sum(
            parameter.numel()
            for name, parameter in model.named_parameters()
            if _is_extra_parameter(name)
        ),
        "per_layer": rows,
    }
