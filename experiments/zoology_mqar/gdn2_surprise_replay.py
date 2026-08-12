from __future__ import annotations

from functools import partial
from typing import Any, Optional

import torch
from torch import nn

from zoology.config import ModelConfig
from zoology.model import _compute_state_size, _init_weights

from experiments.zoology_mqar.gdn2_committed_delta import capture_committed_edit
from experiments.zoology_mqar.gdn2_futureseed import (
    FutureSeedLMBackbone,
    ZoologyGDN2FutureSeedMixer,
)


EVENT_TAPE_SIZE = 16


class ZoologyEventTapeGDN2FutureSeedMixer(ZoologyGDN2FutureSeedMixer):
    """Official GDN2 with a zero-parameter receiver-native event tape."""

    admission_mode = "abstract"

    def __init__(self, *args: Any, event_tape_size: int = EVENT_TAPE_SIZE, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        if event_tape_size != EVENT_TAPE_SIZE:
            raise ValueError("P-FS2-007 fixes the event tape at K=16")
        self.event_tape_size = int(event_tape_size)
        self.last_event_diagnostics: dict[str, torch.Tensor] = {}
        self.last_replay_diagnostics: dict[str, torch.Tensor] = {}
        self.last_selected_indices: Optional[torch.Tensor] = None

    def forward_with_committed_edit(
        self,
        hidden_states: torch.Tensor,
        *,
        initial_state: Optional[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        with capture_committed_edit() as captured:
            output, terminal_state = super().forward_with_state(
                hidden_states,
                initial_state=initial_state,
            )
        if len(captured) != 1:
            raise RuntimeError(
                f"Expected one official committed-edit tensor, got {len(captured)}"
            )
        committed_edit = captured[0].detach()
        if committed_edit.shape[:2] != hidden_states.shape[:2]:
            raise RuntimeError(
                "Official committed-edit shape does not match the producer tokens: "
                f"{committed_edit.shape} versus {hidden_states.shape}"
            )
        return output, terminal_state, committed_edit

    def select_event_tape(
        self,
        hidden_states: torch.Tensor,
        committed_edit: torch.Tensor,
    ) -> torch.Tensor:
        batch, tokens, width = hidden_states.shape
        if tokens < self.event_tape_size:
            raise ValueError(
                f"Event tape K={self.event_tape_size} exceeds sequence length {tokens}"
            )
        scores = committed_edit.float().square().sum(dim=(-1, -2)).sqrt()
        if not torch.isfinite(scores).all():
            raise RuntimeError("Committed-edit surprise contains non-finite values")

        if self.admission_mode == "surprise":
            indices = torch.topk(
                scores,
                k=self.event_tape_size,
                dim=1,
                largest=True,
                sorted=False,
            ).indices
            indices = indices.sort(dim=1).values
        elif self.admission_mode == "recency":
            indices = torch.arange(
                tokens - self.event_tape_size,
                tokens,
                device=hidden_states.device,
            ).expand(batch, -1)
        else:
            raise RuntimeError(f"Unknown event-tape admission: {self.admission_mode}")

        gather = indices.unsqueeze(-1).expand(-1, -1, width)
        evidence = hidden_states.gather(dim=1, index=gather)
        self.last_selected_indices = indices.detach()
        selected_scores = scores.gather(dim=1, index=indices)
        total_score = scores.sum(dim=1).clamp_min(1e-12)
        position_std = indices.float().std(dim=1, unbiased=False)
        with torch.no_grad():
            self.last_event_diagnostics = {
                "committed_edit_rms": committed_edit.float().square().mean().sqrt(),
                "surprise_token_std": scores.std(unbiased=False),
                "selected_surprise_mean": selected_scores.mean(),
                "selected_surprise_fraction": (selected_scores.sum(dim=1) / total_score).mean(),
                "selected_position_mean": indices.float().mean(),
                "selected_position_std": position_std.mean(),
                "selected_position_board_std": indices.float().mean(dim=1).std(
                    unbiased=False
                ),
                "selected_index_min": indices.min().float(),
                "selected_index_max": indices.max().float(),
                "selected_count_per_board": scores.new_tensor(
                    float(indices.shape[1])
                ),
            }
        return evidence

    def replay_seed(self, evidence: torch.Tensor) -> torch.Tensor:
        _output, terminal_state = super().forward_with_state(
            evidence,
            initial_state=None,
        )
        seed = self.make_initial_state(terminal_state)
        with torch.no_grad():
            board_rms = terminal_state.float().square().mean(
                dim=(-1, -2, -3)
            ).sqrt()
            self.last_replay_diagnostics = {
                "replay_terminal_rms": board_rms.mean(),
                "replay_terminal_board_std": board_rms.std(unbiased=False),
                "replay_seed_rms": seed.float().square().mean().sqrt(),
                "replay_seed_abs_max": seed.float().abs().max(),
            }
        return seed


class ZoologyRecencyReplayGDN2FutureSeedMixer(
    ZoologyEventTapeGDN2FutureSeedMixer
):
    """Matched K16 event tape admitted by recency."""

    admission_mode = "recency"


class ZoologySurpriseReplayGDN2FutureSeedMixer(
    ZoologyEventTapeGDN2FutureSeedMixer
):
    """K16 event tape admitted by exact official committed-edit surprise."""

    admission_mode = "surprise"


class EventTapeFutureSeedLMBackbone(FutureSeedLMBackbone):
    """Replace producer-basis state transport with receiver-native replay."""

    def layers_forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        residual = None
        event_tape = None
        for layer_index, layer in enumerate(self.layers):
            mixer = layer.sequence_mixer
            if not isinstance(mixer, ZoologyEventTapeGDN2FutureSeedMixer):
                raise TypeError("Event-tape backbone requires event-tape GDN2 mixers")
            if mixer.future_seed_scale != 1.0:
                raise RuntimeError("Event-tape FutureSeed scale must remain exactly one")

            dropped = layer.drop_path1(layer.dropout1(hidden_states))
            residual = dropped + residual if residual is not None else dropped
            mixer_input = layer.norm1(residual.to(dtype=layer.norm1.weight.dtype))
            initial_state = None if event_tape is None else mixer.replay_seed(event_tape)
            if layer_index + 1 < len(self.layers):
                hidden_states, _terminal_state, committed_edit = (
                    mixer.forward_with_committed_edit(
                        mixer_input,
                        initial_state=initial_state,
                    )
                )
                event_tape = mixer.select_event_tape(mixer_input, committed_edit)
            else:
                hidden_states, _terminal_state = mixer.forward_with_state(
                    mixer_input,
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


class EventTapeFutureSeedLanguageModel(nn.Module):
    """Zoology LM with receiver-native FutureSeed event replay."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        if config.vocab_size % config.pad_vocab_size_multiple != 0:
            config.vocab_size += config.pad_vocab_size_multiple - (
                config.vocab_size % config.pad_vocab_size_multiple
            )
        if config.multiplier != 1:
            raise ValueError("Event-tape model requires multiplier=1")

        self.backbone = EventTapeFutureSeedLMBackbone(config=config)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.apply(
            partial(
                _init_weights,
                n_layers=config.n_layers,
                block_type=config.block_type,
            )
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


def event_tape_diagnostics(model: nn.Module) -> dict[str, Any]:
    mixers = [
        block.sequence_mixer
        for block in model.backbone.layers
        if isinstance(block.sequence_mixer, ZoologyEventTapeGDN2FutureSeedMixer)
    ]
    if len(mixers) != 2:
        raise RuntimeError(f"Expected two event-tape mixers, got {len(mixers)}")
    producer = mixers[0]
    receiver = mixers[1]
    if not producer.last_event_diagnostics or not receiver.last_replay_diagnostics:
        raise RuntimeError("Event-tape diagnostics were not populated")
    return {
        "admission_mode": producer.admission_mode,
        "event_tape_size": producer.event_tape_size,
        "new_parameters": 0,
        "persistent_state_delta": 0,
        "main_official_scans": 2,
        "replay_official_scans": 1,
        "selected_evidence_values_per_board": producer.event_tape_size * 128,
        "committed_edit_gradient_policy": "detached admission only",
        "selected_evidence_gradient_policy": "receiver replay remains differentiable",
        "producer": {
            name: float(value.item())
            for name, value in producer.last_event_diagnostics.items()
        },
        "receiver": {
            name: float(value.item())
            for name, value in receiver.last_replay_diagnostics.items()
        },
    }
