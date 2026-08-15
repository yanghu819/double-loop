from __future__ import annotations

import hashlib
from functools import partial
from typing import Any, Optional

import torch
import torch.nn as nn
from einops import rearrange
from torch.nn import functional as F

from fla.layers.utils import get_layer_cache, update_layer_cache
from fla.ops.gdn2 import chunk_gdn2
from zoology.config import ModelConfig
from zoology.model import _compute_state_size, _init_weights

from experiments.zoology_mqar.gdn2_futureseed import (
    FutureSeedLMBackbone,
    ZoologyGDN2FutureSeedMixer,
)


EXPECTED_NEW_PARAMETERS = 8
EXPECTED_MAIN_STATE_VALUES = 4_096
EXPECTED_CERTIFICATE_STATE_VALUES = 4_096


def _rms(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.float().square().mean().sqrt()


def _normalize_last(tensor: torch.Tensor) -> torch.Tensor:
    rms = tensor.float().square().mean(dim=-1, keepdim=True).add(1e-6).sqrt()
    return tensor / rms.to(dtype=tensor.dtype)


class BindingCertificateGDN2(nn.Module):
    """Native GDN2 plus a same-address state carrying stable token identity."""

    def __init__(self, base: nn.Module) -> None:
        super().__init__()
        if base.hidden_size != 128:
            raise ValueError("P-GDN3-047 requires D128")
        if base.num_heads != 4 or base.num_v_heads != 4:
            raise ValueError("P-GDN3-047 requires H4")
        if base.head_k_dim != 32 or base.head_v_dim != 32:
            raise ValueError("P-GDN3-047 requires K32/V32")
        if base.mode != "chunk" or not base.use_short_conv:
            raise ValueError("P-GDN3-047 requires chunk GDN2 with short convolution")
        self.base = base
        self.certificate_read_logit = nn.Parameter(torch.zeros(1, 1, 4, 1))
        self._capture = False
        self._captured: dict[str, torch.Tensor] = {}

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
        if self.base.layer_idx is None:
            raise RuntimeError("P-GDN3-047 requires an explicit layer index")
        return int(self.base.layer_idx)

    def set_capture(self, enabled: bool) -> None:
        self._capture = bool(enabled)
        if enabled:
            self._captured.clear()

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
            raise ValueError("P-GDN3-047 uses fixed unpadded MQAR batches")
        certificate_payload = kwargs.pop("certificate_payload", None)
        if certificate_payload is None:
            raise ValueError("Binding certificate payload is required")
        if certificate_payload.shape != hidden_states.shape:
            raise ValueError(
                "Certificate payload shape mismatch: "
                f"{tuple(certificate_payload.shape)} != {tuple(hidden_states.shape)}"
            )

        layer = self.base
        q_len = hidden_states.shape[1]
        cu_seqlens = kwargs.get("cu_seqlens")
        if cu_seqlens is not None:
            raise ValueError("P-GDN3-047 does not use packed sequences")
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
        value, conv_state_v = layer.v_conv1d(
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
        value = rearrange(value, "... (h d) -> ... h d", d=layer.head_v_dim)
        write_gate = rearrange(
            write_gate,
            "... (h d) -> ... h d",
            d=layer.head_v_dim,
        )
        certificate_value = rearrange(
            certificate_payload,
            "b t (h d) -> b t h d",
            h=layer.num_v_heads,
            d=layer.head_v_dim,
        ).to(dtype=value.dtype)
        g = -layer.A_log.float().exp().unsqueeze(-1) * g
        if layer.allow_neg_eigval:
            erase_gate = erase_gate * 2.0

        recurrent_state = (
            last_state["recurrent_state"] if last_state is not None else None
        )
        main_output, recurrent_state = chunk_gdn2(
            q=q,
            k=k,
            v=value,
            g=g,
            b=erase_gate,
            w=write_gate,
            initial_state=recurrent_state,
            output_final_state=use_cache,
            use_qk_l2norm_in_kernel=True,
        )
        certificate_output, certificate_state = chunk_gdn2(
            q=q,
            k=k,
            v=certificate_value,
            g=g,
            b=erase_gate,
            w=write_gate,
            initial_state=None,
            output_final_state=True,
            use_qk_l2norm_in_kernel=True,
        )
        update_layer_cache(
            layer,
            past_key_values,
            recurrent_state=recurrent_state,
            conv_state=(conv_state_q, conv_state_k, conv_state_v),
            offset=q_len,
        )

        gate = torch.tanh(self.certificate_read_logit).to(dtype=main_output.dtype)
        certificate_unit = _normalize_last(certificate_output)
        output = main_output + gate * certificate_unit
        if self._capture:
            if recurrent_state is None:
                raise RuntimeError("P-GDN3-047 capture requires main terminal state")
            self._captured = {
                "main_output": main_output.detach(),
                "main_state": recurrent_state.detach(),
                "certificate_output": certificate_output.detach(),
                "certificate_state": certificate_state.detach(),
                "certificate_payload": certificate_value.detach(),
                "certificate_gate": gate.detach(),
                "q": q.detach(),
                "k": k.detach(),
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


class ZoologyBindingCertificateFutureSeedMixer(ZoologyGDN2FutureSeedMixer):
    """Native FutureSeed over main state; certificate state stays layer-local."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.layer = BindingCertificateGDN2(self.layer)

    def forward_with_certificate_state(
        self,
        hidden_states: torch.Tensor,
        *,
        certificate_payload: torch.Tensor,
        initial_state: Optional[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        cache = self._new_cache(initial_state)
        with torch.autocast(
            device_type=hidden_states.device.type,
            dtype=torch.bfloat16,
            enabled=hidden_states.is_cuda,
        ):
            output, _attentions, cache = self.layer(
                hidden_states,
                certificate_payload=certificate_payload,
                past_key_values=cache,
                use_cache=True,
            )
        layer_cache = cache[self.layer_idx]
        terminal_state = layer_cache["recurrent_state"]
        if terminal_state is None:
            raise RuntimeError("Official GDN2 did not return a terminal state")
        return output.to(hidden_states.dtype), terminal_state


class BindingCertificateBackbone(FutureSeedLMBackbone):
    """Backbone that exposes position-free token identity to every layer."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__(config)
        self.last_certificate_payload: Optional[torch.Tensor] = None

    @staticmethod
    def normalize_payload(payload: torch.Tensor) -> torch.Tensor:
        return _normalize_last(payload)

    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        token_payload = self.embeddings.word_embeddings(input_ids)
        if self.embeddings.project_in is not None:
            token_payload = self.embeddings.project_in(token_payload)
        certificate_payload = self.normalize_payload(token_payload)
        self.last_certificate_payload = certificate_payload.detach()

        hidden_states = self.embeddings(input_ids, position_ids=position_ids)
        residual = None
        terminal_state = None
        for layer in self.layers:
            mixer = layer.sequence_mixer
            if not isinstance(mixer, ZoologyBindingCertificateFutureSeedMixer):
                raise TypeError("Binding-certificate backbone requires strict mixers")

            dropped = layer.drop_path1(layer.dropout1(hidden_states))
            residual = dropped + residual if residual is not None else dropped
            hidden_states = layer.norm1(
                residual.to(dtype=layer.norm1.weight.dtype)
            )
            initial_state = (
                None
                if terminal_state is None
                else mixer.make_initial_state(terminal_state)
            )
            hidden_states, terminal_state = mixer.forward_with_certificate_state(
                hidden_states,
                certificate_payload=certificate_payload,
                initial_state=initial_state,
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


class BindingCertificateLanguageModel(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        if config.vocab_size % config.pad_vocab_size_multiple != 0:
            config.vocab_size += config.pad_vocab_size_multiple - (
                config.vocab_size % config.pad_vocab_size_multiple
            )
        if config.multiplier != 1:
            raise ValueError("Discrete FutureSeed model requires multiplier=1")

        self.backbone = BindingCertificateBackbone(config=config)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.apply(
            partial(
                _init_weights,
                n_layers=config.n_layers,
                block_type=config.block_type,
            )
        )
        for block in self.backbone.layers:
            nn.init.zeros_(
                block.sequence_mixer.layer.certificate_read_logit
            )
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


def _is_certificate_parameter(name: str) -> bool:
    return name.endswith(".certificate_read_logit")


def load_matched_parent_state(
    model: nn.Module,
    control_state: dict[str, torch.Tensor],
) -> None:
    loaded: dict[str, torch.Tensor] = {}
    for name, tensor in model.state_dict().items():
        if _is_certificate_parameter(name):
            loaded[name] = torch.zeros_like(tensor)
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
        if _is_certificate_parameter(name):
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


def _repeated_token_max_error(
    input_ids: torch.Tensor,
    payload: torch.Tensor,
) -> tuple[float, int]:
    max_error = 0.0
    groups = 0
    ids = input_ids.detach().cpu()
    values = payload.detach().float().cpu()
    for board_ids, board_values in zip(ids, values):
        for token in board_ids.unique():
            positions = torch.nonzero(board_ids == token, as_tuple=False).flatten()
            if positions.numel() < 2:
                continue
            groups += 1
            reference = board_values[positions[0]]
            max_error = max(
                max_error,
                float((board_values[positions] - reference).abs().max().item()),
            )
    return max_error, groups


@torch.no_grad()
def binding_certificate_diagnostics(
    model: nn.Module,
    inputs: torch.Tensor,
) -> dict[str, Any]:
    if not isinstance(model.backbone, BindingCertificateBackbone):
        raise TypeError("Expected BindingCertificateBackbone")
    mixers = [block.sequence_mixer for block in model.backbone.layers]
    if len(mixers) != 2 or not all(
        isinstance(mixer, ZoologyBindingCertificateFutureSeedMixer)
        for mixer in mixers
    ):
        raise RuntimeError("Expected two binding-certificate mixers")
    for mixer in mixers:
        mixer.layer.set_capture(True)
    try:
        model.eval()(inputs)
    finally:
        for mixer in mixers:
            mixer.layer.set_capture(False)

    payload = model.backbone.last_certificate_payload
    if payload is None:
        raise RuntimeError("Certificate payload capture is missing")
    repeat_error, repeat_groups = _repeated_token_max_error(inputs, payload)
    payload_board = payload.float().square().mean(dim=(-1, -2)).sqrt()
    rows = []
    for mixer in mixers:
        captured = mixer.layer._captured
        required = {
            "main_output",
            "main_state",
            "certificate_output",
            "certificate_state",
            "certificate_payload",
            "certificate_gate",
            "q",
            "k",
        }
        if set(captured) != required:
            raise RuntimeError(f"Incomplete certificate capture: {set(captured)}")
        main_output = captured["main_output"].float()
        certificate_output = captured["certificate_output"].float()
        certificate_state = captured["certificate_state"].float()
        main_state = captured["main_state"].float()
        output_board = certificate_output.square().mean(dim=(1, 2, 3)).sqrt()
        state_board = certificate_state.square().mean(dim=(1, 2, 3)).sqrt()
        output_cosine = F.cosine_similarity(
            main_output.flatten(1),
            certificate_output.flatten(1),
            dim=-1,
            eps=1e-8,
        )
        rows.append(
            {
                "layer_idx": mixer.layer_idx,
                "certificate_gate_abs_mean": float(
                    captured["certificate_gate"].float().abs().mean().item()
                ),
                "active_gate_heads": int(
                    (
                        captured["certificate_gate"].float().abs().flatten()
                        >= 1e-4
                    ).sum().item()
                ),
                "certificate_output_rms": float(_rms(certificate_output).item()),
                "certificate_output_token_std": float(
                    certificate_output.mean(dim=-1).std(dim=1, unbiased=False).mean().item()
                ),
                "certificate_output_board_std": float(
                    output_board.std(unbiased=False).item()
                ),
                "certificate_state_rms": float(_rms(certificate_state).item()),
                "certificate_state_board_std": float(
                    state_board.std(unbiased=False).item()
                ),
                "certificate_to_main_state_rms": float(
                    (_rms(certificate_state) / _rms(main_state).clamp_min(1e-8)).item()
                ),
                "main_certificate_output_cosine_mean": float(
                    output_cosine.mean().item()
                ),
            }
        )
    return {
        "active_layers": len(rows),
        "active_gate_heads": sum(row["active_gate_heads"] for row in rows),
        "new_parameters": sum(
            mixer.layer.certificate_read_logit.numel() for mixer in mixers
        ),
        "main_persistent_state_values_per_layer": EXPECTED_MAIN_STATE_VALUES,
        "certificate_transient_state_values_per_layer": (
            EXPECTED_CERTIFICATE_STATE_VALUES
        ),
        "official_scans_per_layer": 2,
        "certificate_futureseed_routes": 0,
        "token_only_payload": True,
        "uses_position_payload": False,
        "payload_rms": float(_rms(payload).item()),
        "payload_token_std": float(payload.float().std(dim=1, unbiased=False).mean().item()),
        "payload_board_std": float(payload_board.std(unbiased=False).item()),
        "repeated_token_groups": repeat_groups,
        "repeated_token_max_error": repeat_error,
        "per_layer": rows,
    }
