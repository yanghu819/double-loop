from __future__ import annotations

import argparse
import copy
import hashlib
import json
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from zoology.config import DataConfig, ModelConfig, ModuleConfig, TrainConfig
from zoology.data.utils import prepare_data
from zoology.model import LanguageModel
from zoology.train import Trainer
from zoology.utils import set_determinism

from experiments.zoology_mqar.directional_mqar import DirectionalMQARConfig
from experiments.zoology_mqar.futureseed_directionality import (
    CaptureLogger,
    dataset_hash,
    model_hash,
)
from experiments.zoology_mqar.gdn2_futureseed import (
    FutureSeedLanguageModel,
    futureseed_diagnostics,
)


VOCAB_SIZE = 256
TRAIN_EXAMPLES = 10_000
VALID_EXAMPLES = 1_000
SEED = 123
MODEL_WIDTH = 128
MODEL_LAYERS = 2
MODEL_HEADS = 4
GDN2_HEAD_DIM = 32
ATTENTION_HEAD_DIM = 57
GDN2_ARMS = (
    "causal_gdn2",
    "future_seed_gdn2",
    "future_seed_gdn2_log_spd",
    "future_seed_gdn2_metric_pullback",
    "future_seed_gdn2_shared_log_spd",
    "future_seed_gdn2_log_spd_block_gram",
    "future_seed_gdn2_two_edit",
    "future_seed_gdn2_dual_hash",
    "future_seed_gdn2_shared_committed_delta",
    "future_seed_gdn2_clustered_committed_delta",
    "future_seed_gdn2_oig",
    "future_seed_gdn2_recency_replay",
    "future_seed_gdn2_surprise_replay",
    "future_seed_gdn2_surprise_regression",
    "future_seed_gdn2_atomic_pair",
    "future_seed_gated_delta_product_n2",
    "future_seed_erase_then_delta",
    "future_seed_contractive_dplr",
    "future_seed_decoupled_key_gdn2",
    "future_seed_anchored_dual_key_gdn2",
    "future_seed_committed_residual_gdn2",
    "future_seed_chunk_local_biaxis_gdn2",
    "future_seed_slot_state_gdn2",
    "future_seed_raven_address_gdn2",
    "future_seed_linear_product_state_gdn2",
    "future_seed_query_delta_gdn2",
    "future_seed_qk_coherence_gdn2",
    "future_seed_biorthogonal_qk_gdn2",
    "future_seed_dynamic_frame_gdn2",
    "future_seed_receiver_read_credit_gdn2",
    "future_seed_coherent_key_spectrum_gdn2",
    "future_seed_stable_token_address_gdn2",
    "future_seed_binding_certificate_gdn2",
    "future_seed_canonical_address_companion_gdn2",
    "future_seed_address_payload_gauge_gdn2",
    "future_seed_producer_readout_gdn2",
    "future_seed_sparse_delta_slot_gdn2",
    "future_seed_readonly_dual_plane_gdn2",
    "future_seed_lagged_commit_gdn2",
    "future_seed_committed_interference_gdn2",
    "future_seed_redundant_address_gdn2",
    "future_seed_block_rls_gdn2",
    "future_seed_local_binding_gdn2",
    "future_seed_shared_eligibility_gdn2",
    "future_seed_pair_event_gdn2",
    "future_seed_cycle_memory_gdn2",
    "future_seed_momentum_delta",
    "future_seed_momentum_phase",
    "future_seed_momentum_prediction_key",
    "future_seed_momentum_log_spd",
    "future_seed_momentum_refresh",
    "future_seed_predictive_momentum",
    "future_seed_official_sdm",
    "future_seed_comba",
    "future_seed_mesa",
)
ARMS = ("causal_gdn2", "future_seed_gdn2", "bidirectional_attention")
P007_LENGTH64_TRAIN_HASH = (
    "31bac228c46a1c85b0b0e164675c7f65e9bb0a3d4b733d4c42d7c057bd6d5bf9"
)
P007_LENGTH64_TEST_HASH = (
    "3fa26a5a04b8302001231451bce23c4f5372648d8cdd9fe6116d8e418f470209"
)


def build_config(
    *,
    arm: str,
    sequence_length: int,
    num_kv_pairs: int,
    max_epochs: int,
    batch_size: int,
    model_width: int = MODEL_WIDTH,
    model_heads: int = MODEL_HEADS,
    gdn2_head_dim: int = GDN2_HEAD_DIM,
    gdn2_expand_v: float = 1.0,
) -> TrainConfig:
    data = DataConfig(
        train_configs=[
            DirectionalMQARConfig(
                num_examples=TRAIN_EXAMPLES,
                vocab_size=VOCAB_SIZE,
                input_seq_len=sequence_length,
                num_kv_pairs=num_kv_pairs,
                direction="mixed",
            )
        ],
        test_configs=[
            DirectionalMQARConfig(
                num_examples=VALID_EXAMPLES,
                vocab_size=VOCAB_SIZE,
                input_seq_len=sequence_length,
                num_kv_pairs=num_kv_pairs,
                direction="mixed",
            )
        ],
        batch_size=batch_size,
        seed=SEED,
        cache_dir=(
            "/huyang2/double-loop/.cache/"
            f"zoology-directional-scaling-l{sequence_length}-k{num_kv_pairs}"
        ),
    )
    if arm in GDN2_ARMS:
        value_width = model_heads * gdn2_head_dim * gdn2_expand_v
        if not float(value_width).is_integer():
            raise ValueError(
                "model_heads * gdn2_head_dim * gdn2_expand_v must be an integer"
            )
        if arm in (
            "future_seed_gdn2_log_spd",
            "future_seed_gdn2_metric_pullback",
            "future_seed_gdn2_shared_log_spd",
        ):
            mixer_name = (
                "experiments.zoology_mqar.gdn2_log_spd."
                "ZoologyLogSPDGDN2FutureSeedMixer"
            )
        elif arm == "future_seed_gdn2_log_spd_block_gram":
            mixer_name = (
                "experiments.zoology_mqar.gdn2_block_gram."
                "ZoologyBlockGramGDN2FutureSeedMixer"
            )
        elif arm == "future_seed_gdn2_two_edit":
            mixer_name = (
                "experiments.zoology_mqar.gdn2_rank2."
                "ZoologyTwoEditGDN2FutureSeedMixer"
            )
        elif arm == "future_seed_gdn2_dual_hash":
            mixer_name = (
                "experiments.zoology_mqar.gdn2_dual_hash."
                "ZoologyDualHashGDN2FutureSeedMixer"
            )
        elif arm == "future_seed_gdn2_shared_committed_delta":
            mixer_name = (
                "experiments.zoology_mqar.gdn2_committed_delta."
                "ZoologySharedCommittedDeltaFutureSeedMixer"
            )
        elif arm == "future_seed_gdn2_clustered_committed_delta":
            mixer_name = (
                "experiments.zoology_mqar.gdn2_committed_delta."
                "ZoologyClusteredCommittedDeltaFutureSeedMixer"
            )
        elif arm == "future_seed_gdn2_oig":
            mixer_name = (
                "experiments.zoology_mqar.gdn2_oig."
                "ZoologyOIGGDN2FutureSeedMixer"
            )
        elif arm == "future_seed_gdn2_recency_replay":
            mixer_name = (
                "experiments.zoology_mqar.gdn2_surprise_replay."
                "ZoologyRecencyReplayGDN2FutureSeedMixer"
            )
        elif arm == "future_seed_gdn2_surprise_replay":
            mixer_name = (
                "experiments.zoology_mqar.gdn2_surprise_replay."
                "ZoologySurpriseReplayGDN2FutureSeedMixer"
            )
        elif arm == "future_seed_gdn2_surprise_regression":
            mixer_name = (
                "experiments.zoology_mqar.gdn2_surprise_regression_seed."
                "ZoologySurpriseRegressionGDN2FutureSeedMixer"
            )
        elif arm == "future_seed_gdn2_atomic_pair":
            mixer_name = (
                "experiments.zoology_mqar.gdn2_atomic_pair."
                "ZoologyAtomicPairGDN2FutureSeedMixer"
            )
        elif arm == "future_seed_gated_delta_product_n2":
            mixer_name = (
                "experiments.zoology_mqar.gated_delta_product_futureseed."
                "ZoologyGatedDeltaProductFutureSeedMixer"
            )
        elif arm == "future_seed_erase_then_delta":
            mixer_name = (
                "experiments.zoology_mqar.erase_then_delta_futureseed."
                "ZoologyEraseThenDeltaFutureSeedMixer"
            )
        elif arm == "future_seed_contractive_dplr":
            mixer_name = (
                "experiments.zoology_mqar.contractive_dplr_futureseed."
                "ZoologyContractiveDPLRFutureSeedMixer"
            )
        elif arm == "future_seed_decoupled_key_gdn2":
            mixer_name = (
                "experiments.zoology_mqar.decoupled_key_futureseed."
                "ZoologyDecoupledKeyGDN2FutureSeedMixer"
            )
        elif arm == "future_seed_anchored_dual_key_gdn2":
            mixer_name = (
                "experiments.zoology_mqar.anchored_dual_key_futureseed."
                "ZoologyAnchoredDualKeyGDN2FutureSeedMixer"
            )
        elif arm == "future_seed_committed_residual_gdn2":
            mixer_name = (
                "experiments.zoology_mqar.committed_residual_futureseed."
                "ZoologyCommittedResidualGDN2FutureSeedMixer"
            )
        elif arm == "future_seed_chunk_local_biaxis_gdn2":
            mixer_name = (
                "experiments.zoology_mqar.gdn2_chunk_local_biaxis."
                "ZoologyChunkLocalBiAxisFutureSeedMixer"
            )
        elif arm == "future_seed_slot_state_gdn2":
            mixer_name = (
                "experiments.zoology_mqar.gdn2_slot_state."
                "ZoologySlotStateFutureSeedMixer"
            )
        elif arm == "future_seed_raven_address_gdn2":
            mixer_name = (
                "experiments.zoology_mqar.gdn2_raven_address."
                "ZoologyRavenAddressFutureSeedMixer"
            )
        elif arm == "future_seed_linear_product_state_gdn2":
            mixer_name = (
                "experiments.zoology_mqar.gdn2_linear_product_state."
                "ZoologyLinearProductStateFutureSeedMixer"
            )
        elif arm == "future_seed_query_delta_gdn2":
            mixer_name = (
                "experiments.zoology_mqar.query_delta_futureseed."
                "ZoologyQueryDeltaFutureSeedMixer"
            )
        elif arm == "future_seed_qk_coherence_gdn2":
            mixer_name = (
                "experiments.zoology_mqar.qk_coherence_futureseed."
                "ZoologyQKCoherenceFutureSeedMixer"
            )
        elif arm == "future_seed_biorthogonal_qk_gdn2":
            mixer_name = (
                "experiments.zoology_mqar.biorthogonal_qk_futureseed."
                "ZoologyBiorthogonalQKFutureSeedMixer"
            )
        elif arm == "future_seed_dynamic_frame_gdn2":
            mixer_name = (
                "experiments.zoology_mqar.dynamic_frame_futureseed."
                "ZoologyDynamicFrameFutureSeedMixer"
            )
        elif arm == "future_seed_stable_token_address_gdn2":
            mixer_name = (
                "experiments.zoology_mqar.stable_token_address_futureseed."
                "ZoologyStableTokenAddressFutureSeedMixer"
            )
        elif arm == "future_seed_binding_certificate_gdn2":
            mixer_name = (
                "experiments.zoology_mqar.binding_certificate_futureseed."
                "ZoologyBindingCertificateFutureSeedMixer"
            )
        elif arm == "future_seed_canonical_address_companion_gdn2":
            mixer_name = (
                "experiments.zoology_mqar.canonical_address_companion."
                "ZoologyCanonicalAddressCompanionFutureSeedMixer"
            )
        elif arm == "future_seed_address_payload_gauge_gdn2":
            mixer_name = (
                "experiments.zoology_mqar.address_payload_gauge_futureseed."
                "ZoologyAddressPayloadGaugeFutureSeedMixer"
            )
        elif arm == "future_seed_sparse_delta_slot_gdn2":
            mixer_name = (
                "experiments.zoology_mqar.gdn2_sparse_delta_slots."
                "ZoologySparseDeltaSlotFutureSeedMixer"
            )
        elif arm == "future_seed_readonly_dual_plane_gdn2":
            mixer_name = (
                "experiments.zoology_mqar.readonly_dual_plane_futureseed."
                "ZoologyReadOnlyDualPlaneFutureSeedMixer"
            )
        elif arm == "future_seed_lagged_commit_gdn2":
            mixer_name = (
                "experiments.zoology_mqar.lagged_commit_futureseed."
                "ZoologyLaggedCommitFutureSeedMixer"
            )
        elif arm == "future_seed_redundant_address_gdn2":
            mixer_name = (
                "experiments.zoology_mqar.gdn2_redundant_address."
                "ZoologyRedundantAddressFutureSeedMixer"
            )
        elif arm == "future_seed_block_rls_gdn2":
            mixer_name = (
                "experiments.zoology_mqar.gdn2_block_rls."
                "ZoologyBlockRLSGDN2FutureSeedMixer"
            )
        elif arm == "future_seed_local_binding_gdn2":
            mixer_name = (
                "experiments.zoology_mqar.gdn2_local_binding."
                "ZoologyLocalBindingGDN2FutureSeedMixer"
            )
        elif arm == "future_seed_shared_eligibility_gdn2":
            mixer_name = (
                "experiments.zoology_mqar.gdn2_shared_eligibility."
                "ZoologySharedEligibilityFutureSeedMixer"
            )
        elif arm == "future_seed_pair_event_gdn2":
            mixer_name = (
                "experiments.zoology_mqar.gdn2_pair_event."
                "ZoologyPairEventGDN2FutureSeedMixer"
            )
        elif arm == "future_seed_cycle_memory_gdn2":
            mixer_name = (
                "experiments.zoology_mqar.gdn2_cycle_memory."
                "ZoologyCycleMemoryFutureSeedMixer"
            )
        elif arm == "future_seed_momentum_delta":
            mixer_name = (
                "experiments.zoology_mqar.momentum_futureseed."
                "ZoologyMomentumDeltaFutureSeedMixer"
            )
        elif arm == "future_seed_momentum_phase":
            mixer_name = (
                "experiments.zoology_mqar.momentum_phase_futureseed."
                "ZoologyMomentumPhaseFutureSeedMixer"
            )
        elif arm == "future_seed_momentum_prediction_key":
            mixer_name = (
                "experiments.zoology_mqar.momentum_prediction_key."
                "ZoologyMomentumPredictionKeyFutureSeedMixer"
            )
        elif arm == "future_seed_momentum_log_spd":
            mixer_name = (
                "experiments.zoology_mqar.momentum_log_spd."
                "ZoologyMomentumLogSPDFutureSeedMixer"
            )
        elif arm == "future_seed_momentum_refresh":
            mixer_name = (
                "experiments.zoology_mqar.momentum_refresh."
                "ZoologyMomentumRefreshFutureSeedMixer"
            )
        elif arm == "future_seed_predictive_momentum":
            mixer_name = (
                "experiments.zoology_mqar.momentum_predictive."
                "ZoologyPredictiveMomentumFutureSeedMixer"
            )
        elif arm == "future_seed_official_sdm":
            mixer_name = (
                "experiments.zoology_mqar.official_sdm_futureseed."
                "ZoologyOfficialSDMFutureSeedMixer"
            )
        elif arm == "future_seed_comba":
            mixer_name = (
                "experiments.zoology_mqar.comba_futureseed."
                "ZoologyCombaFutureSeedMixer"
            )
        elif arm == "future_seed_mesa":
            mixer_name = (
                "experiments.zoology_mqar.mesa_futureseed."
                "ZoologyMesaFutureSeedMixer"
            )
        else:
            mixer_name = (
                "experiments.zoology_mqar.gdn2_futureseed."
                "ZoologyGDN2FutureSeedMixer"
            )
        sequence_mixer = ModuleConfig(
            name=mixer_name,
            kwargs={
                "num_heads": model_heads,
                "head_dim": gdn2_head_dim,
                "expand_v": gdn2_expand_v,
                "conv_size": 4,
                "future_seed_scale": (
                    0.0 if arm == "causal_gdn2" else 1.0
                ),
            },
        )
    elif arm == "bidirectional_attention":
        sequence_mixer = ModuleConfig(
            name=(
                "experiments.zoology_mqar.bidirectional_attention."
                "ParamMatchedBidirectionalAttention"
            ),
            kwargs={
                "num_heads": MODEL_HEADS,
                "head_dim": ATTENTION_HEAD_DIM,
            },
        )
    else:
        raise ValueError(f"Unknown arm: {arm}")

    model = ModelConfig(
        vocab_size=VOCAB_SIZE,
        max_position_embeddings=sequence_length,
        d_model=model_width,
        n_layers=MODEL_LAYERS,
        sequence_mixer=sequence_mixer,
    )
    return TrainConfig(
        data=data,
        model=model,
        max_epochs=max_epochs,
        early_stopping_metric="valid/accuracy",
        early_stopping_threshold=1.1,
        learning_rate=1e-3,
        weight_decay=0.1,
        seed=SEED,
        slice_keys=[],
        run_id=f"directional-scaling-{arm}-l{sequence_length}-k{num_kv_pairs}",
    )


def make_model(config: TrainConfig, arm: str) -> torch.nn.Module:
    if arm == "future_seed_cycle_memory_gdn2":
        from experiments.zoology_mqar.gdn2_cycle_memory import (
            CycleMemoryLanguageModel,
        )

        return CycleMemoryLanguageModel(copy.deepcopy(config.model))
    if arm == "future_seed_shared_eligibility_gdn2":
        from experiments.zoology_mqar.gdn2_shared_eligibility import (
            SharedEligibilityLanguageModel,
        )

        return SharedEligibilityLanguageModel(copy.deepcopy(config.model))
    if arm == "future_seed_lagged_commit_gdn2":
        from experiments.zoology_mqar.lagged_commit_futureseed import (
            LaggedCommitLanguageModel,
        )

        return LaggedCommitLanguageModel(copy.deepcopy(config.model))
    if arm == "future_seed_readonly_dual_plane_gdn2":
        from experiments.zoology_mqar.readonly_dual_plane_futureseed import (
            ReadOnlyDualPlaneLanguageModel,
        )

        return ReadOnlyDualPlaneLanguageModel(copy.deepcopy(config.model))
    if arm == "future_seed_sparse_delta_slot_gdn2":
        from experiments.zoology_mqar.gdn2_sparse_delta_slots import (
            SparseDeltaSlotLanguageModel,
        )

        return SparseDeltaSlotLanguageModel(copy.deepcopy(config.model))
    if arm == "future_seed_canonical_address_companion_gdn2":
        from experiments.zoology_mqar.canonical_address_companion import (
            CanonicalAddressCompanionLanguageModel,
        )

        return CanonicalAddressCompanionLanguageModel(
            copy.deepcopy(config.model)
        )
    if arm == "future_seed_binding_certificate_gdn2":
        from experiments.zoology_mqar.binding_certificate_futureseed import (
            BindingCertificateLanguageModel,
        )

        return BindingCertificateLanguageModel(copy.deepcopy(config.model))
    if arm == "future_seed_stable_token_address_gdn2":
        from experiments.zoology_mqar.stable_token_address_futureseed import (
            StableTokenAddressLanguageModel,
        )

        return StableTokenAddressLanguageModel(copy.deepcopy(config.model))
    if arm == "future_seed_coherent_key_spectrum_gdn2":
        from experiments.zoology_mqar.coherent_key_spectrum_credit import (
            CoherentKeySpectrumLanguageModel,
        )

        return CoherentKeySpectrumLanguageModel(copy.deepcopy(config.model))
    if arm == "future_seed_committed_interference_gdn2":
        from experiments.zoology_mqar.committed_interference_credit import (
            CommittedInterferenceLanguageModel,
        )

        return CommittedInterferenceLanguageModel(copy.deepcopy(config.model))
    if arm == "future_seed_receiver_read_credit_gdn2":
        from experiments.zoology_mqar.futureseed_read_credit import (
            ReceiverReadCreditLanguageModel,
        )

        return ReceiverReadCreditLanguageModel(copy.deepcopy(config.model))
    if arm == "future_seed_producer_readout_gdn2":
        from experiments.zoology_mqar.producer_readout_futureseed import (
            ProducerReadoutFutureSeedLanguageModel,
        )

        return ProducerReadoutFutureSeedLanguageModel(copy.deepcopy(config.model))
    if arm == "future_seed_dynamic_frame_gdn2":
        from experiments.zoology_mqar.dynamic_frame_futureseed import (
            DynamicFrameFutureSeedLanguageModel,
        )

        return DynamicFrameFutureSeedLanguageModel(copy.deepcopy(config.model))
    if arm == "future_seed_gdn2_metric_pullback":
        from experiments.zoology_mqar.gdn2_metric_pullback_futureseed import (
            MetricPullbackFutureSeedLanguageModel,
        )

        return MetricPullbackFutureSeedLanguageModel(copy.deepcopy(config.model))
    if arm == "future_seed_gdn2_shared_log_spd":
        from experiments.zoology_mqar.gdn2_shared_log_spd import (
            SharedLogSPDFutureSeedLanguageModel,
        )

        return SharedLogSPDFutureSeedLanguageModel(copy.deepcopy(config.model))
    if arm == "future_seed_decoupled_key_gdn2":
        from experiments.zoology_mqar.decoupled_key_futureseed import (
            make_tied_decoupled_key_model,
        )

        return make_tied_decoupled_key_model(copy.deepcopy(config.model))
    if arm == "future_seed_anchored_dual_key_gdn2":
        from experiments.zoology_mqar.anchored_dual_key_futureseed import (
            make_tied_anchored_dual_key_model,
        )

        return make_tied_anchored_dual_key_model(copy.deepcopy(config.model))
    if arm == "future_seed_committed_residual_gdn2":
        from experiments.zoology_mqar.committed_residual_futureseed import (
            make_matched_committed_residual_model,
        )

        return make_matched_committed_residual_model(copy.deepcopy(config.model))
    if arm == "future_seed_chunk_local_biaxis_gdn2":
        from experiments.zoology_mqar.gdn2_chunk_local_biaxis import (
            make_chunk_local_biaxis_model,
        )

        return make_chunk_local_biaxis_model(copy.deepcopy(config.model))
    if arm == "future_seed_gdn2_surprise_regression":
        from experiments.zoology_mqar.gdn2_surprise_regression_seed import (
            SurpriseRegressionFutureSeedLanguageModel,
        )

        return SurpriseRegressionFutureSeedLanguageModel(
            copy.deepcopy(config.model)
        )
    if arm in (
        "future_seed_gdn2_recency_replay",
        "future_seed_gdn2_surprise_replay",
    ):
        from experiments.zoology_mqar.gdn2_surprise_replay import (
            EventTapeFutureSeedLanguageModel,
        )

        return EventTapeFutureSeedLanguageModel(copy.deepcopy(config.model))
    if arm in GDN2_ARMS:
        return FutureSeedLanguageModel(copy.deepcopy(config.model))
    return LanguageModel(copy.deepcopy(config.model))


def parameter_hash(model: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, parameter in sorted(model.named_parameters()):
        digest.update(name.encode())
        digest.update(parameter.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def fixed_batch_hash(batch: tuple[torch.Tensor, torch.Tensor, Any]) -> str:
    digest = hashlib.sha256()
    for tensor in batch[:2]:
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def warm_model(
    model: torch.nn.Module,
    batch: tuple[torch.Tensor, torch.Tensor, Any],
) -> None:
    model.train().cuda()
    inputs, labels, _slices = batch
    inputs = inputs.cuda()
    labels = labels.cuda()
    for _ in range(3):
        model.zero_grad(set_to_none=True)
        logits = model(inputs)
        loss = F.cross_entropy(logits.flatten(0, 1), labels.flatten())
        loss = loss + _model_auxiliary_loss(model)
        loss.backward()
    model.zero_grad(set_to_none=True)
    torch.cuda.synchronize()


def benchmark_training_step(
    model: torch.nn.Module,
    batch: tuple[torch.Tensor, torch.Tensor, Any],
    *,
    warmup_steps: int = 5,
    measured_steps: int = 20,
) -> dict[str, float]:
    model.train()
    inputs, labels, _slices = batch
    inputs = inputs.cuda()
    labels = labels.cuda()

    def one_step() -> None:
        model.zero_grad(set_to_none=True)
        logits = model(inputs)
        loss = F.cross_entropy(logits.flatten(0, 1), labels.flatten())
        loss = loss + _model_auxiliary_loss(model)
        loss.backward()

    for _ in range(warmup_steps):
        one_step()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    for _ in range(measured_steps):
        one_step()
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    model.zero_grad(set_to_none=True)
    return {
        "measured_steps": float(measured_steps),
        "elapsed_sec": elapsed,
        "tokens_per_sec": inputs.numel() * measured_steps / elapsed,
        "examples_per_sec": inputs.shape[0] * measured_steps / elapsed,
        "peak_cuda_mem_bytes": float(torch.cuda.max_memory_allocated()),
    }


def _model_auxiliary_loss(model: torch.nn.Module) -> torch.Tensor:
    losses: list[torch.Tensor] = []

    def collect(module: torch.nn.Module) -> None:
        if hasattr(module, "get_auxiliary_loss"):
            losses.append(module.get_auxiliary_loss())

    model.apply(collect)
    if not losses:
        return torch.zeros((), device=next(model.parameters()).device)
    return torch.stack(losses).sum()


def _query_event(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    predictions: torch.Tensor,
    query_position: int,
    quarter: int,
) -> dict[str, Any]:
    key = int(inputs[query_position].item())
    target = int(targets[query_position].item())
    candidate_positions = torch.nonzero(inputs == key).flatten().tolist()
    write_position = None
    for candidate in candidate_positions:
        if candidate == query_position or candidate + 1 >= inputs.numel():
            continue
        if int(inputs[candidate + 1].item()) == target:
            write_position = int(candidate)
            break
    if write_position is None:
        raise RuntimeError("Could not recover the write position for a query")
    direction = "future" if query_position < quarter else "past"
    return {
        "direction": direction,
        "query_position": query_position,
        "write_position": write_position,
        "distance": write_position - query_position,
        "key": key,
        "target": target,
        "prediction": int(predictions[query_position].item()),
        "correct": bool(predictions[query_position].item() == target),
    }


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    dataloader,
    *,
    sequence_length: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    model.eval()
    quarter = sequence_length // 4
    totals: dict[str, dict[str, float]] = {}
    cases: list[dict[str, Any]] = []
    case_index = 0
    for inputs, targets, _slices in dataloader:
        inputs_gpu = inputs.cuda()
        targets_gpu = targets.cuda()
        logits = model(inputs_gpu)
        predictions = logits.argmax(dim=-1)
        label_mask = targets_gpu != -100
        positions = torch.arange(sequence_length, device="cuda")
        direction_masks = {
            "future": label_mask & (positions[None, :] < quarter),
            "past": (
                label_mask
                & (positions[None, :] >= 2 * quarter)
                & (positions[None, :] < 3 * quarter)
            ),
        }
        for direction, direction_mask in direction_masks.items():
            loss_sum = F.cross_entropy(
                logits[direction_mask],
                targets_gpu[direction_mask],
                reduction="sum",
            )
            correct = (predictions == targets_gpu) & direction_mask
            query_count = direction_mask.sum(dim=1)
            exact = correct.sum(dim=1) == query_count
            bucket = totals.setdefault(
                direction,
                {
                    "correct": 0.0,
                    "queries": 0.0,
                    "exact": 0.0,
                    "examples": 0.0,
                    "loss": 0.0,
                },
            )
            bucket["correct"] += float(correct.sum().item())
            bucket["queries"] += float(direction_mask.sum().item())
            bucket["exact"] += float(exact.sum().item())
            bucket["examples"] += float(inputs.shape[0])
            bucket["loss"] += float(loss_sum.item())

        inputs_cpu = inputs.cpu()
        targets_cpu = targets.cpu()
        predictions_cpu = predictions.cpu()
        for row in range(inputs.shape[0]):
            query_positions = torch.nonzero(targets_cpu[row] != -100).flatten()
            events = [
                _query_event(
                    inputs_cpu[row],
                    targets_cpu[row],
                    predictions_cpu[row],
                    int(position.item()),
                    quarter,
                )
                for position in query_positions
            ]
            future_errors = sum(
                int(not event["correct"])
                for event in events
                if event["direction"] == "future"
            )
            past_errors = sum(
                int(not event["correct"])
                for event in events
                if event["direction"] == "past"
            )
            cases.append(
                {
                    "case_index": case_index,
                    "case_id": hashlib.sha256(
                        inputs_cpu[row].contiguous().numpy().tobytes()
                    ).hexdigest()[:16],
                    "sequence_length": sequence_length,
                    "future_errors": future_errors,
                    "past_errors": past_errors,
                    "events": events,
                }
            )
            case_index += 1

    metrics: dict[str, Any] = {}
    for direction, bucket in totals.items():
        metrics[direction] = {
            "accuracy": bucket["correct"] / bucket["queries"],
            "exact": bucket["exact"] / bucket["examples"],
            "ce": bucket["loss"] / bucket["queries"],
            "queries": int(bucket["queries"]),
            "examples": int(bucket["examples"]),
        }
    metrics["balanced_accuracy"] = sum(
        metrics[direction]["accuracy"] for direction in ("past", "future")
    ) / 2.0
    metrics["joint_exact"] = sum(
        int(case["future_errors"] + case["past_errors"] == 0) for case in cases
    ) / len(cases)
    cases.sort(
        key=lambda case: (
            -case["future_errors"],
            -case["past_errors"],
            case["case_index"],
        )
    )
    return metrics, cases


def run_arm(
    *,
    arm: str,
    sequence_length: int,
    num_kv_pairs: int,
    output_dir: Path,
    max_epochs: int,
    batch_size: int,
    model_width: int = MODEL_WIDTH,
    model_heads: int = MODEL_HEADS,
    gdn2_head_dim: int = GDN2_HEAD_DIM,
    gdn2_expand_v: float = 1.0,
    output_arm_name: str | None = None,
    save_checkpoint: bool = False,
    matched_init_path: Path | None = None,
) -> dict[str, Any]:
    run_arm_name = output_arm_name or arm
    arm_dir = output_dir / f"length_{sequence_length}" / run_arm_name
    arm_dir.mkdir(parents=True, exist_ok=True)
    cold_arm_started = time.perf_counter()
    config = build_config(
        arm=arm,
        sequence_length=sequence_length,
        num_kv_pairs=num_kv_pairs,
        max_epochs=max_epochs,
        batch_size=batch_size,
        model_width=model_width,
        model_heads=model_heads,
        gdn2_head_dim=gdn2_head_dim,
        gdn2_expand_v=gdn2_expand_v,
    )
    set_determinism(config.seed)
    model = make_model(config, arm)
    if matched_init_path is not None:
        matched_state = torch.load(
            matched_init_path,
            map_location="cpu",
            weights_only=True,
        )
        if arm == "future_seed_slot_state_gdn2":
            from experiments.zoology_mqar.gdn2_slot_state import (
                load_matched_parent_state,
            )

            load_matched_parent_state(model, matched_state)
        elif arm == "future_seed_raven_address_gdn2":
            from experiments.zoology_mqar.gdn2_raven_address import (
                load_matched_parent_state,
            )

            load_matched_parent_state(model, matched_state)
        elif arm == "future_seed_linear_product_state_gdn2":
            from experiments.zoology_mqar.gdn2_linear_product_state import (
                load_matched_parent_state,
            )

            load_matched_parent_state(model, matched_state)
        elif arm == "future_seed_query_delta_gdn2":
            from experiments.zoology_mqar.query_delta_futureseed import (
                load_matched_parent_state,
            )

            load_matched_parent_state(model, matched_state)
        elif arm == "future_seed_qk_coherence_gdn2":
            from experiments.zoology_mqar.qk_coherence_futureseed import (
                load_matched_parent_state,
            )

            load_matched_parent_state(model, matched_state)
        elif arm == "future_seed_biorthogonal_qk_gdn2":
            from experiments.zoology_mqar.biorthogonal_qk_futureseed import (
                load_matched_parent_state,
            )

            load_matched_parent_state(model, matched_state)
        elif arm == "future_seed_dynamic_frame_gdn2":
            from experiments.zoology_mqar.dynamic_frame_futureseed import (
                load_matched_parent_state,
            )

            load_matched_parent_state(model, matched_state)
        elif arm == "future_seed_producer_readout_gdn2":
            from experiments.zoology_mqar.producer_readout_futureseed import (
                load_matched_parent_state,
            )

            load_matched_parent_state(model, matched_state)
        elif arm == "future_seed_stable_token_address_gdn2":
            from experiments.zoology_mqar.stable_token_address_futureseed import (
                load_matched_parent_state,
            )

            load_matched_parent_state(model, matched_state)
        elif arm == "future_seed_binding_certificate_gdn2":
            from experiments.zoology_mqar.binding_certificate_futureseed import (
                load_matched_parent_state,
            )

            load_matched_parent_state(model, matched_state)
        elif arm == "future_seed_canonical_address_companion_gdn2":
            from experiments.zoology_mqar.canonical_address_companion import (
                load_matched_parent_state,
            )

            load_matched_parent_state(model, matched_state)
        elif arm == "future_seed_address_payload_gauge_gdn2":
            from experiments.zoology_mqar.address_payload_gauge_futureseed import (
                load_matched_parent_state,
            )

            load_matched_parent_state(model, matched_state)
        elif arm == "future_seed_sparse_delta_slot_gdn2":
            from experiments.zoology_mqar.gdn2_sparse_delta_slots import (
                load_matched_parent_state,
            )

            load_matched_parent_state(model, matched_state)
        elif arm == "future_seed_readonly_dual_plane_gdn2":
            from experiments.zoology_mqar.readonly_dual_plane_futureseed import (
                load_matched_parent_state,
            )

            load_matched_parent_state(model, matched_state)
        elif arm == "future_seed_lagged_commit_gdn2":
            from experiments.zoology_mqar.lagged_commit_futureseed import (
                load_matched_parent_state,
            )

            load_matched_parent_state(model, matched_state)
        elif arm == "future_seed_redundant_address_gdn2":
            from experiments.zoology_mqar.gdn2_redundant_address import (
                load_matched_parent_state,
            )

            load_matched_parent_state(model, matched_state)
        elif arm == "future_seed_block_rls_gdn2":
            from experiments.zoology_mqar.gdn2_block_rls import (
                load_matched_parent_state,
            )

            load_matched_parent_state(model, matched_state)
        elif arm == "future_seed_local_binding_gdn2":
            from experiments.zoology_mqar.gdn2_local_binding import (
                load_matched_parent_state,
            )

            load_matched_parent_state(model, matched_state)
        elif arm == "future_seed_shared_eligibility_gdn2":
            from experiments.zoology_mqar.gdn2_shared_eligibility import (
                load_matched_parent_state,
            )

            load_matched_parent_state(model, matched_state)
        elif arm == "future_seed_pair_event_gdn2":
            from experiments.zoology_mqar.gdn2_pair_event import (
                load_matched_parent_state,
            )

            load_matched_parent_state(model, matched_state)
        elif arm == "future_seed_cycle_memory_gdn2":
            from experiments.zoology_mqar.gdn2_cycle_memory import (
                load_matched_parent_state,
            )

            load_matched_parent_state(model, matched_state)
        elif arm == "future_seed_momentum_prediction_key":
            from experiments.zoology_mqar.momentum_prediction_key import (
                load_matched_parent_state,
            )

            load_matched_parent_state(model, matched_state)
        elif arm == "future_seed_momentum_phase":
            from experiments.zoology_mqar.momentum_phase_futureseed import (
                load_matched_parent_state,
            )

            load_matched_parent_state(model, matched_state)
        elif arm in (
            "future_seed_momentum_delta",
            "future_seed_momentum_log_spd",
            "future_seed_momentum_refresh",
            "future_seed_predictive_momentum",
        ):
            from experiments.zoology_mqar.momentum_futureseed import (
                load_matched_parent_state,
            )

            load_matched_parent_state(model, matched_state)
        elif arm == "future_seed_official_sdm":
            from experiments.zoology_mqar.official_sdm_futureseed import (
                load_matched_parent_state,
            )

            load_matched_parent_state(model, matched_state)
        elif arm == "future_seed_comba":
            from experiments.zoology_mqar.comba_futureseed import (
                load_matched_parent_state,
            )

            load_matched_parent_state(model, matched_state)
        elif arm == "future_seed_mesa":
            from experiments.zoology_mqar.mesa_futureseed import (
                load_matched_parent_state,
            )

            load_matched_parent_state(model, matched_state)
        else:
            model.load_state_dict(matched_state, strict=True)
    contractive_dplr_initial_beta_weights = None
    if arm == "future_seed_contractive_dplr":
        contractive_dplr_initial_beta_weights = [
            block.sequence_mixer.layer.beta_proj.weight.detach()
            .float()
            .cpu()
            .clone()
            for block in model.backbone.layers
        ]
    init_hash = model_hash(model)
    init_parameter_hash = parameter_hash(model)
    parent_init_parameter_hash = None
    if arm in (
        "future_seed_gdn2_log_spd",
        "future_seed_gdn2_metric_pullback",
    ):
        from experiments.zoology_mqar.gdn2_log_spd import parent_parameter_hash

        parent_init_parameter_hash = parent_parameter_hash(model)
    elif arm == "future_seed_gdn2_shared_log_spd":
        from experiments.zoology_mqar.gdn2_shared_log_spd import parent_parameter_hash

        parent_init_parameter_hash = parent_parameter_hash(model)
    elif arm == "future_seed_gdn2_log_spd_block_gram":
        from experiments.zoology_mqar.gdn2_block_gram import parent_parameter_hash

        parent_init_parameter_hash = parent_parameter_hash(model)
    elif arm == "future_seed_gdn2_two_edit":
        from experiments.zoology_mqar.gdn2_rank2 import parent_parameter_hash

        parent_init_parameter_hash = parent_parameter_hash(model)
    elif arm in (
        "future_seed_gdn2_shared_committed_delta",
        "future_seed_gdn2_clustered_committed_delta",
    ):
        from experiments.zoology_mqar.gdn2_committed_delta import parent_parameter_hash

        parent_init_parameter_hash = parent_parameter_hash(model)
    elif arm == "future_seed_official_sdm":
        from experiments.zoology_mqar.official_sdm_futureseed import (
            parent_parameter_hash,
        )

        parent_init_parameter_hash = parent_parameter_hash(model)
    elif arm == "future_seed_gdn2_oig":
        from experiments.zoology_mqar.gdn2_oig import parent_parameter_hash

        parent_init_parameter_hash = parent_parameter_hash(model)
    elif arm == "future_seed_gdn2_atomic_pair":
        from experiments.zoology_mqar.gdn2_atomic_pair import parent_parameter_hash

        parent_init_parameter_hash = parent_parameter_hash(model)
    elif arm == "future_seed_decoupled_key_gdn2":
        from experiments.zoology_mqar.decoupled_key_futureseed import (
            parent_parameter_hash,
        )

        parent_init_parameter_hash = parent_parameter_hash(model)
    elif arm == "future_seed_anchored_dual_key_gdn2":
        from experiments.zoology_mqar.anchored_dual_key_futureseed import (
            parent_parameter_hash,
        )

        parent_init_parameter_hash = parent_parameter_hash(model)
    elif arm == "future_seed_committed_residual_gdn2":
        from experiments.zoology_mqar.committed_residual_futureseed import (
            parent_parameter_hash,
        )

        parent_init_parameter_hash = parent_parameter_hash(model)
    elif arm == "future_seed_chunk_local_biaxis_gdn2":
        from experiments.zoology_mqar.gdn2_chunk_local_biaxis import (
            parent_parameter_hash,
        )

        parent_init_parameter_hash = parent_parameter_hash(model)
    elif arm == "future_seed_slot_state_gdn2":
        from experiments.zoology_mqar.gdn2_slot_state import parent_parameter_hash

        parent_init_parameter_hash = parent_parameter_hash(model)
    elif arm == "future_seed_raven_address_gdn2":
        from experiments.zoology_mqar.gdn2_raven_address import (
            parent_parameter_hash,
        )

        parent_init_parameter_hash = parent_parameter_hash(model)
    elif arm == "future_seed_linear_product_state_gdn2":
        from experiments.zoology_mqar.gdn2_linear_product_state import (
            parent_parameter_hash,
        )

        parent_init_parameter_hash = parent_parameter_hash(model)
    elif arm == "future_seed_query_delta_gdn2":
        from experiments.zoology_mqar.query_delta_futureseed import (
            parent_parameter_hash,
        )

        parent_init_parameter_hash = parent_parameter_hash(model)
    elif arm == "future_seed_qk_coherence_gdn2":
        from experiments.zoology_mqar.qk_coherence_futureseed import (
            parent_parameter_hash,
        )

        parent_init_parameter_hash = parent_parameter_hash(model)
    elif arm == "future_seed_biorthogonal_qk_gdn2":
        from experiments.zoology_mqar.biorthogonal_qk_futureseed import (
            parent_parameter_hash,
        )

        parent_init_parameter_hash = parent_parameter_hash(model)
    elif arm == "future_seed_dynamic_frame_gdn2":
        from experiments.zoology_mqar.dynamic_frame_futureseed import (
            parent_parameter_hash,
        )

        parent_init_parameter_hash = parent_parameter_hash(model)
    elif arm == "future_seed_producer_readout_gdn2":
        from experiments.zoology_mqar.producer_readout_futureseed import (
            parent_parameter_hash,
        )

        parent_init_parameter_hash = parent_parameter_hash(model)
    elif arm == "future_seed_stable_token_address_gdn2":
        from experiments.zoology_mqar.stable_token_address_futureseed import (
            parent_parameter_hash,
        )

        parent_init_parameter_hash = parent_parameter_hash(model)
    elif arm == "future_seed_binding_certificate_gdn2":
        from experiments.zoology_mqar.binding_certificate_futureseed import (
            parent_parameter_hash,
        )

        parent_init_parameter_hash = parent_parameter_hash(model)
    elif arm == "future_seed_canonical_address_companion_gdn2":
        from experiments.zoology_mqar.canonical_address_companion import (
            parent_parameter_hash,
        )

        parent_init_parameter_hash = parent_parameter_hash(model)
    elif arm == "future_seed_address_payload_gauge_gdn2":
        from experiments.zoology_mqar.address_payload_gauge_futureseed import (
            parent_parameter_hash,
        )

        parent_init_parameter_hash = parent_parameter_hash(model)
    elif arm == "future_seed_sparse_delta_slot_gdn2":
        from experiments.zoology_mqar.gdn2_sparse_delta_slots import (
            parent_parameter_hash,
        )

        parent_init_parameter_hash = parent_parameter_hash(model)
    elif arm == "future_seed_readonly_dual_plane_gdn2":
        from experiments.zoology_mqar.readonly_dual_plane_futureseed import (
            parent_parameter_hash,
        )

        parent_init_parameter_hash = parent_parameter_hash(model)
    elif arm == "future_seed_lagged_commit_gdn2":
        from experiments.zoology_mqar.lagged_commit_futureseed import (
            parent_parameter_hash,
        )

        parent_init_parameter_hash = parent_parameter_hash(model)
    elif arm == "future_seed_redundant_address_gdn2":
        from experiments.zoology_mqar.gdn2_redundant_address import (
            parent_parameter_hash,
        )

        parent_init_parameter_hash = parent_parameter_hash(model)
    elif arm == "future_seed_block_rls_gdn2":
        from experiments.zoology_mqar.gdn2_block_rls import parent_parameter_hash

        parent_init_parameter_hash = parent_parameter_hash(model)
    elif arm == "future_seed_local_binding_gdn2":
        from experiments.zoology_mqar.gdn2_local_binding import (
            parent_parameter_hash,
        )

        parent_init_parameter_hash = parent_parameter_hash(model)
    elif arm == "future_seed_shared_eligibility_gdn2":
        from experiments.zoology_mqar.gdn2_shared_eligibility import (
            parent_parameter_hash,
        )

        parent_init_parameter_hash = parent_parameter_hash(model)
    elif arm == "future_seed_pair_event_gdn2":
        from experiments.zoology_mqar.gdn2_pair_event import parent_parameter_hash

        parent_init_parameter_hash = parent_parameter_hash(model)
    elif arm == "future_seed_cycle_memory_gdn2":
        from experiments.zoology_mqar.gdn2_cycle_memory import (
            parent_parameter_hash,
        )

        parent_init_parameter_hash = parent_parameter_hash(model)
    elif arm in (
        "future_seed_momentum_delta",
        "future_seed_momentum_phase",
        "future_seed_momentum_prediction_key",
        "future_seed_momentum_log_spd",
        "future_seed_momentum_refresh",
        "future_seed_predictive_momentum",
    ):
        from experiments.zoology_mqar.momentum_futureseed import (
            parent_parameter_hash,
        )

        parent_init_parameter_hash = parent_parameter_hash(model)
    elif arm == "future_seed_comba":
        from experiments.zoology_mqar.comba_futureseed import (
            parent_parameter_hash,
        )

        parent_init_parameter_hash = parent_parameter_hash(model)
    elif arm == "future_seed_mesa":
        from experiments.zoology_mqar.mesa_futureseed import (
            parent_parameter_hash,
        )

        parent_init_parameter_hash = parent_parameter_hash(model)
    train_dataloader, test_dataloader = prepare_data(config.data)
    data_hashes = {
        "train": dataset_hash(train_dataloader),
        "test": dataset_hash(test_dataloader),
    }
    fixed_batch = next(iter(train_dataloader))
    warmup_batch_hash = fixed_batch_hash(fixed_batch)
    warm_model(model, fixed_batch)
    set_determinism(config.seed)
    post_warm_arm_started = time.perf_counter()

    logger = CaptureLogger(arm_dir / "metrics.jsonl", run_arm_name)
    logger.log_config(config)
    logger.log_model(model, config)
    trainer = Trainer(
        model=model,
        train_dataloader=train_dataloader,
        test_dataloader=test_dataloader,
        input_type=config.input_type,
        max_epochs=config.max_epochs,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        early_stopping_metric=config.early_stopping_metric,
        early_stopping_threshold=config.early_stopping_threshold,
        slice_keys=config.slice_keys,
        loss_type=config.loss_type,
        device="cuda",
        logger=logger,
    )

    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    started = time.perf_counter()
    trainer.fit()
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    training_peak = torch.cuda.max_memory_allocated()
    metrics, cases = evaluate(
        model,
        test_dataloader,
        sequence_length=sequence_length,
    )
    surprise_regression = None
    if arm == "future_seed_gdn2_surprise_regression":
        from experiments.zoology_mqar.gdn2_surprise_regression_seed import (
            surprise_regression_diagnostics,
        )

        diagnostic_inputs, _diagnostic_labels, _diagnostic_slices = next(
            iter(test_dataloader)
        )
        for block in model.backbone.layers:
            block.sequence_mixer.collect_regression_diagnostics = True
        with torch.no_grad():
            model.eval()(diagnostic_inputs[:8].cuda())
        for block in model.backbone.layers:
            block.sequence_mixer.collect_regression_diagnostics = False
        surprise_regression = surprise_regression_diagnostics(model)
    future_seed = futureseed_diagnostics(model) if arm in GDN2_ARMS else None
    log_spd = None
    if arm in (
        "future_seed_gdn2_log_spd",
        "future_seed_gdn2_metric_pullback",
        "future_seed_gdn2_log_spd_block_gram",
    ):
        from experiments.zoology_mqar.gdn2_log_spd import log_spd_diagnostics

        log_spd = log_spd_diagnostics(model)
    elif arm == "future_seed_gdn2_shared_log_spd":
        from experiments.zoology_mqar.gdn2_shared_log_spd import (
            shared_log_spd_diagnostics,
        )

        log_spd = shared_log_spd_diagnostics(model)
    block_gram = None
    metric_pullback = None
    if arm == "future_seed_gdn2_metric_pullback":
        from experiments.zoology_mqar.gdn2_metric_pullback_futureseed import (
            metric_pullback_diagnostics,
        )

        metric_pullback = metric_pullback_diagnostics(model)
    if arm == "future_seed_gdn2_log_spd_block_gram":
        from experiments.zoology_mqar.gdn2_block_gram import block_gram_diagnostics

        block_gram = block_gram_diagnostics(model)
    two_edit = None
    if arm == "future_seed_gdn2_two_edit":
        from experiments.zoology_mqar.gdn2_rank2 import two_edit_diagnostics

        two_edit = two_edit_diagnostics(model)
    dual_hash = None
    if arm == "future_seed_gdn2_dual_hash":
        from experiments.zoology_mqar.gdn2_dual_hash import dual_hash_diagnostics

        dual_hash = dual_hash_diagnostics(model)
    committed_delta = None
    if arm in (
        "future_seed_gdn2_shared_committed_delta",
        "future_seed_gdn2_clustered_committed_delta",
    ):
        from experiments.zoology_mqar.gdn2_committed_delta import (
            committed_delta_diagnostics,
        )

        committed_delta = committed_delta_diagnostics(model)
    oig = None
    if arm == "future_seed_gdn2_oig":
        from experiments.zoology_mqar.gdn2_oig import oig_diagnostics

        oig = oig_diagnostics(model)
    event_tape = None
    if arm in (
        "future_seed_gdn2_recency_replay",
        "future_seed_gdn2_surprise_replay",
    ):
        from experiments.zoology_mqar.gdn2_surprise_replay import (
            event_tape_diagnostics,
        )

        diagnostic_inputs, _diagnostic_labels, _diagnostic_slices = next(
            iter(test_dataloader)
        )
        with torch.no_grad():
            model.eval()(diagnostic_inputs.cuda())
        event_tape = event_tape_diagnostics(model)
    atomic_pair = None
    if arm == "future_seed_gdn2_atomic_pair":
        from experiments.zoology_mqar.gdn2_atomic_pair import (
            atomic_pair_diagnostics,
        )

        atomic_pair = atomic_pair_diagnostics(model)
    gated_delta_product = None
    if arm == "future_seed_gated_delta_product_n2":
        from experiments.zoology_mqar.gated_delta_product_futureseed import (
            gated_delta_product_diagnostics,
        )

        diagnostic_inputs, _diagnostic_labels, _diagnostic_slices = next(
            iter(test_dataloader)
        )
        gated_delta_product = gated_delta_product_diagnostics(
            model,
            diagnostic_inputs[:8].cuda(),
        )
    erase_then_delta = None
    if arm == "future_seed_erase_then_delta":
        from experiments.zoology_mqar.erase_then_delta_futureseed import (
            erase_then_delta_diagnostics,
        )

        diagnostic_inputs, _diagnostic_labels, _diagnostic_slices = next(
            iter(test_dataloader)
        )
        erase_then_delta = erase_then_delta_diagnostics(
            model,
            diagnostic_inputs[:8].cuda(),
        )
    contractive_dplr = None
    if arm == "future_seed_contractive_dplr":
        from experiments.zoology_mqar.contractive_dplr_futureseed import (
            contractive_dplr_diagnostics,
        )

        diagnostic_inputs, _diagnostic_labels, _diagnostic_slices = next(
            iter(test_dataloader)
        )
        contractive_dplr = contractive_dplr_diagnostics(
            model,
            diagnostic_inputs[:8].cuda(),
            initial_beta_weights=contractive_dplr_initial_beta_weights,
        )
    decoupled_key = None
    if arm == "future_seed_decoupled_key_gdn2":
        from experiments.zoology_mqar.decoupled_key_futureseed import (
            decoupled_key_diagnostics,
        )

        diagnostic_inputs, _diagnostic_labels, _diagnostic_slices = next(
            iter(test_dataloader)
        )
        decoupled_key = decoupled_key_diagnostics(
            model,
            diagnostic_inputs[:8].cuda(),
        )
    anchored_dual_key = None
    if arm == "future_seed_anchored_dual_key_gdn2":
        from experiments.zoology_mqar.anchored_dual_key_futureseed import (
            anchored_dual_key_diagnostics,
        )

        diagnostic_inputs, _diagnostic_labels, _diagnostic_slices = next(
            iter(test_dataloader)
        )
        anchored_dual_key = anchored_dual_key_diagnostics(
            model,
            diagnostic_inputs[:8].cuda(),
        )
    committed_residual = None
    if arm == "future_seed_committed_residual_gdn2":
        from experiments.zoology_mqar.committed_residual_futureseed import (
            committed_residual_diagnostics,
        )

        diagnostic_inputs, _diagnostic_labels, _diagnostic_slices = next(
            iter(test_dataloader)
        )
        committed_residual = committed_residual_diagnostics(
            model,
            diagnostic_inputs[:4].cuda(),
        )
    chunk_local_biaxis = None
    if arm == "future_seed_chunk_local_biaxis_gdn2":
        from experiments.zoology_mqar.gdn2_chunk_local_biaxis import (
            chunk_local_biaxis_diagnostics,
        )

        diagnostic_inputs, _diagnostic_labels, _diagnostic_slices = next(
            iter(test_dataloader)
        )
        with torch.no_grad():
            model.eval()(diagnostic_inputs[:8].cuda())
        chunk_local_biaxis = chunk_local_biaxis_diagnostics(model)
    slot_state = None
    if arm == "future_seed_slot_state_gdn2":
        from experiments.zoology_mqar.gdn2_slot_state import (
            slot_state_diagnostics,
        )

        diagnostic_inputs, _diagnostic_labels, _diagnostic_slices = next(
            iter(test_dataloader)
        )
        with torch.no_grad():
            model.eval()(diagnostic_inputs[:8].cuda())
        slot_state = slot_state_diagnostics(model)
    raven_address = None
    if arm == "future_seed_raven_address_gdn2":
        from experiments.zoology_mqar.gdn2_raven_address import (
            raven_address_diagnostics,
        )

        diagnostic_inputs, _diagnostic_labels, _diagnostic_slices = next(
            iter(test_dataloader)
        )
        with torch.no_grad():
            model.eval()(diagnostic_inputs[:8].cuda())
        raven_address = raven_address_diagnostics(model)
    linear_product_state = None
    if arm == "future_seed_linear_product_state_gdn2":
        from experiments.zoology_mqar.gdn2_linear_product_state import (
            linear_product_state_diagnostics,
        )

        diagnostic_inputs, _diagnostic_labels, _diagnostic_slices = next(
            iter(test_dataloader)
        )
        with torch.no_grad():
            model.eval()(diagnostic_inputs[:8].cuda())
        linear_product_state = linear_product_state_diagnostics(model)
    query_delta = None
    if arm == "future_seed_query_delta_gdn2":
        from experiments.zoology_mqar.query_delta_futureseed import (
            query_delta_diagnostics,
        )

        diagnostic_inputs, _diagnostic_labels, _diagnostic_slices = next(
            iter(test_dataloader)
        )
        query_delta = query_delta_diagnostics(
            model,
            diagnostic_inputs[:8].cuda(),
        )
    qk_coherence = None
    if arm == "future_seed_qk_coherence_gdn2":
        from experiments.zoology_mqar.qk_coherence_futureseed import (
            qk_coherence_diagnostics,
        )

        diagnostic_inputs, _diagnostic_labels, _diagnostic_slices = next(
            iter(test_dataloader)
        )
        qk_coherence = qk_coherence_diagnostics(
            model,
            diagnostic_inputs[:8].cuda(),
        )
    biorthogonal_qk = None
    if arm == "future_seed_biorthogonal_qk_gdn2":
        from experiments.zoology_mqar.biorthogonal_qk_futureseed import (
            biorthogonal_qk_diagnostics,
        )

        diagnostic_inputs, _diagnostic_labels, _diagnostic_slices = next(
            iter(test_dataloader)
        )
        biorthogonal_qk = biorthogonal_qk_diagnostics(
            model,
            diagnostic_inputs[:8].cuda(),
        )
    dynamic_frame = None
    if arm == "future_seed_dynamic_frame_gdn2":
        from experiments.zoology_mqar.dynamic_frame_futureseed import (
            dynamic_frame_diagnostics,
        )

        diagnostic_inputs, _diagnostic_labels, _diagnostic_slices = next(
            iter(test_dataloader)
        )
        dynamic_frame = dynamic_frame_diagnostics(
            model,
            diagnostic_inputs[:8].cuda(),
        )
    receiver_read_credit = None
    if arm == "future_seed_receiver_read_credit_gdn2":
        from experiments.zoology_mqar.futureseed_read_credit import (
            receiver_read_credit_diagnostics,
        )

        receiver_read_credit = receiver_read_credit_diagnostics(model)
    coherent_key_spectrum = None
    if arm == "future_seed_coherent_key_spectrum_gdn2":
        from experiments.zoology_mqar.coherent_key_spectrum_credit import (
            coherent_key_spectrum_diagnostics,
        )

        coherent_key_spectrum = coherent_key_spectrum_diagnostics(model)
    committed_interference = None
    if arm == "future_seed_committed_interference_gdn2":
        from experiments.zoology_mqar.committed_interference_credit import (
            committed_interference_diagnostics,
        )

        committed_interference = committed_interference_diagnostics(model)
    stable_token_address = None
    if arm == "future_seed_stable_token_address_gdn2":
        from experiments.zoology_mqar.stable_token_address_futureseed import (
            stable_token_address_diagnostics,
        )

        diagnostic_inputs, _diagnostic_labels, _diagnostic_slices = next(
            iter(test_dataloader)
        )
        stable_token_address = stable_token_address_diagnostics(
            model,
            diagnostic_inputs[:8].cuda(),
        )
    binding_certificate = None
    if arm == "future_seed_binding_certificate_gdn2":
        from experiments.zoology_mqar.binding_certificate_futureseed import (
            binding_certificate_diagnostics,
        )

        diagnostic_inputs, _diagnostic_labels, _diagnostic_slices = next(
            iter(test_dataloader)
        )
        binding_certificate = binding_certificate_diagnostics(
            model,
            diagnostic_inputs[:8].cuda(),
        )
    canonical_address_companion = None
    if arm == "future_seed_canonical_address_companion_gdn2":
        from experiments.zoology_mqar.canonical_address_companion import (
            canonical_address_companion_diagnostics,
        )

        diagnostic_inputs, _diagnostic_labels, _diagnostic_slices = next(
            iter(test_dataloader)
        )
        canonical_address_companion = (
            canonical_address_companion_diagnostics(
                model,
                diagnostic_inputs[:8].cuda(),
            )
        )
    sparse_delta_slots = None
    if arm == "future_seed_sparse_delta_slot_gdn2":
        from experiments.zoology_mqar.gdn2_sparse_delta_slots import (
            sparse_delta_slot_diagnostics,
        )

        diagnostic_inputs, _diagnostic_labels, _diagnostic_slices = next(
            iter(test_dataloader)
        )
        sparse_delta_slots = sparse_delta_slot_diagnostics(
            model,
            diagnostic_inputs[:8].cuda(),
        )
    address_payload_gauge = None
    if arm == "future_seed_address_payload_gauge_gdn2":
        from experiments.zoology_mqar.address_payload_gauge_futureseed import (
            address_payload_gauge_diagnostics,
        )

        diagnostic_inputs, _diagnostic_labels, _diagnostic_slices = next(
            iter(test_dataloader)
        )
        address_payload_gauge = address_payload_gauge_diagnostics(
            model,
            diagnostic_inputs[:8].cuda(),
        )
    producer_readout = None
    producer_readout_edge_off = None
    producer_readout_edge_off_cases = None
    readonly_dual_plane = None
    readonly_dual_plane_edge_off = None
    readonly_dual_plane_edge_off_cases = None
    lagged_commit = None
    lagged_commit_edge_off = None
    lagged_commit_edge_off_cases = None
    diagnostic_extra_wall_sec = 0.0
    redundant_address = None
    if arm == "future_seed_redundant_address_gdn2":
        from experiments.zoology_mqar.gdn2_redundant_address import (
            redundant_address_diagnostics,
        )

        diagnostic_inputs, _diagnostic_labels, _diagnostic_slices = next(
            iter(test_dataloader)
        )
        with torch.no_grad():
            model.eval()(diagnostic_inputs[:8].cuda())
        redundant_address = redundant_address_diagnostics(model)
    block_rls = None
    if arm == "future_seed_block_rls_gdn2":
        from experiments.zoology_mqar.gdn2_block_rls import block_rls_diagnostics

        diagnostic_inputs, _diagnostic_labels, _diagnostic_slices = next(
            iter(test_dataloader)
        )
        block_rls = block_rls_diagnostics(
            model,
            diagnostic_inputs[:8].cuda(),
        )
    local_binding = None
    if arm == "future_seed_local_binding_gdn2":
        from experiments.zoology_mqar.gdn2_local_binding import (
            local_binding_diagnostics,
        )

        diagnostic_inputs, _diagnostic_labels, _diagnostic_slices = next(
            iter(test_dataloader)
        )
        local_binding = local_binding_diagnostics(
            model,
            diagnostic_inputs[:8].cuda(),
        )
    shared_eligibility = None
    if arm == "future_seed_shared_eligibility_gdn2":
        from experiments.zoology_mqar.gdn2_shared_eligibility import (
            shared_eligibility_diagnostics,
        )

        diagnostic_inputs, _diagnostic_labels, _diagnostic_slices = next(
            iter(test_dataloader)
        )
        shared_eligibility = shared_eligibility_diagnostics(
            model,
            diagnostic_inputs[:8].cuda(),
        )
    pair_event = None
    if arm == "future_seed_pair_event_gdn2":
        from experiments.zoology_mqar.gdn2_pair_event import (
            pair_event_diagnostics,
        )

        diagnostic_inputs, _diagnostic_labels, _diagnostic_slices = next(
            iter(test_dataloader)
        )
        pair_event = pair_event_diagnostics(
            model,
            diagnostic_inputs[:8].cuda(),
        )
    cycle_memory = None
    if arm == "future_seed_cycle_memory_gdn2":
        from experiments.zoology_mqar.gdn2_cycle_memory import (
            cycle_memory_diagnostics,
        )

        diagnostic_inputs, _diagnostic_labels, _diagnostic_slices = next(
            iter(test_dataloader)
        )
        cycle_memory = cycle_memory_diagnostics(
            model,
            diagnostic_inputs[:8].cuda(),
        )
    momentum_delta = None
    if arm in (
        "future_seed_momentum_delta",
        "future_seed_momentum_phase",
        "future_seed_momentum_prediction_key",
        "future_seed_momentum_log_spd",
        "future_seed_momentum_refresh",
        "future_seed_predictive_momentum",
    ):
        from experiments.zoology_mqar.momentum_futureseed import (
            momentum_futureseed_diagnostics,
        )

        diagnostic_inputs, _diagnostic_labels, _diagnostic_slices = next(
            iter(test_dataloader)
        )
        with torch.no_grad():
            model.eval()(diagnostic_inputs[:8].cuda())
        if arm == "future_seed_momentum_phase":
            from experiments.zoology_mqar.momentum_phase_futureseed import (
                momentum_phase_diagnostics,
            )

            momentum_delta = momentum_phase_diagnostics(model)
        elif arm == "future_seed_momentum_prediction_key":
            from experiments.zoology_mqar.momentum_prediction_key import (
                momentum_prediction_key_diagnostics,
            )

            momentum_delta = momentum_prediction_key_diagnostics(model)
        elif arm == "future_seed_momentum_log_spd":
            from experiments.zoology_mqar.momentum_log_spd import (
                momentum_log_spd_diagnostics,
            )

            momentum_delta = momentum_log_spd_diagnostics(model)
        elif arm == "future_seed_momentum_refresh":
            from experiments.zoology_mqar.momentum_refresh import (
                momentum_refresh_diagnostics,
            )

            momentum_delta = momentum_refresh_diagnostics(model)
        elif arm == "future_seed_predictive_momentum":
            from experiments.zoology_mqar.momentum_predictive import (
                predictive_momentum_diagnostics,
            )

            momentum_delta = predictive_momentum_diagnostics(model)
        else:
            momentum_delta = momentum_futureseed_diagnostics(model)
    official_sdm = None
    if arm == "future_seed_official_sdm":
        from experiments.zoology_mqar.official_sdm_futureseed import (
            official_sdm_diagnostics,
        )

        diagnostic_inputs, _diagnostic_labels, _diagnostic_slices = next(
            iter(test_dataloader)
        )
        official_sdm = official_sdm_diagnostics(
            model,
            diagnostic_inputs[:8].cuda(),
        )
    comba = None
    if arm == "future_seed_comba":
        from experiments.zoology_mqar.comba_futureseed import comba_diagnostics

        diagnostic_inputs, _diagnostic_labels, _diagnostic_slices = next(
            iter(test_dataloader)
        )
        comba = comba_diagnostics(model, diagnostic_inputs[:8].cuda())
    mesa = None
    if arm == "future_seed_mesa":
        from experiments.zoology_mqar.mesa_futureseed import (
            mesa_futureseed_diagnostics,
        )

        diagnostic_inputs, _diagnostic_labels, _diagnostic_slices = next(
            iter(test_dataloader)
        )
        mesa = mesa_futureseed_diagnostics(
            model,
            diagnostic_inputs[:8].cuda(),
        )
    if arm == "future_seed_producer_readout_gdn2":
        from experiments.zoology_mqar.producer_readout_futureseed import (
            producer_readout_diagnostics,
        )

        diagnostic_inputs, _diagnostic_labels, _diagnostic_slices = next(
            iter(test_dataloader)
        )
        diagnostic_started = time.perf_counter()
        producer_readout = producer_readout_diagnostics(
            model,
            diagnostic_inputs[:8].cuda(),
        )
        model.set_producer_readout_enabled(False)
        try:
            producer_readout_edge_off, producer_readout_edge_off_cases = evaluate(
                model,
                test_dataloader,
                sequence_length=sequence_length,
            )
        finally:
            model.set_producer_readout_enabled(True)
        diagnostic_extra_wall_sec = time.perf_counter() - diagnostic_started
    if arm == "future_seed_readonly_dual_plane_gdn2":
        from experiments.zoology_mqar.readonly_dual_plane_futureseed import (
            readonly_dual_plane_diagnostics,
        )

        diagnostic_inputs, _diagnostic_labels, _diagnostic_slices = next(
            iter(test_dataloader)
        )
        diagnostic_started = time.perf_counter()
        readonly_dual_plane = readonly_dual_plane_diagnostics(
            model,
            diagnostic_inputs[:8].cuda(),
        )
        model.set_readonly_enabled(False)
        try:
            (
                readonly_dual_plane_edge_off,
                readonly_dual_plane_edge_off_cases,
            ) = evaluate(
                model,
                test_dataloader,
                sequence_length=sequence_length,
            )
        finally:
            model.set_readonly_enabled(True)
        diagnostic_extra_wall_sec = time.perf_counter() - diagnostic_started
    if arm == "future_seed_lagged_commit_gdn2":
        from experiments.zoology_mqar.lagged_commit_futureseed import (
            lagged_commit_diagnostics,
        )

        diagnostic_inputs, _diagnostic_labels, _diagnostic_slices = next(
            iter(test_dataloader)
        )
        diagnostic_started = time.perf_counter()
        lagged_commit = lagged_commit_diagnostics(
            model,
            diagnostic_inputs[:8].cuda(),
        )
        model.set_lag_enabled(False)
        try:
            lagged_commit_edge_off, lagged_commit_edge_off_cases = evaluate(
                model,
                test_dataloader,
                sequence_length=sequence_length,
            )
        finally:
            model.set_lag_enabled(True)
        diagnostic_extra_wall_sec = time.perf_counter() - diagnostic_started
    benchmark = benchmark_training_step(model, fixed_batch)
    trained_parameter_hash = parameter_hash(model)
    checkpoint_path = None
    checkpoint_sha256 = None
    if save_checkpoint:
        checkpoint_path = arm_dir / "model_state.pt"
        torch.save(
            {
                "arm": run_arm_name,
                "carrier_arm": arm,
                "model_state_dict": {
                    name: tensor.detach().cpu()
                    for name, tensor in model.state_dict().items()
                },
            },
            checkpoint_path,
        )
        checkpoint_sha256 = hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()
    torch.cuda.synchronize()
    post_warm_arm_wall_sec = (
        time.perf_counter() - post_warm_arm_started - diagnostic_extra_wall_sec
    )
    cold_arm_wall_sec = (
        time.perf_counter() - cold_arm_started - diagnostic_extra_wall_sec
    )
    score: dict[str, Any] = {
        "arm": run_arm_name,
        "carrier_arm": arm,
        "sequence_length": sequence_length,
        "num_kv_pairs": num_kv_pairs,
        "model_width": model_width,
        "model_heads": model_heads,
        "gdn2_head_dim": gdn2_head_dim,
        "gdn2_expand_v": gdn2_expand_v,
        "recurrent_state_values_per_layer": (
            131_072
            if arm == "future_seed_official_sdm"
            else 8_192
            if arm == "future_seed_mesa"
            else int(model_heads * gdn2_head_dim * gdn2_head_dim * gdn2_expand_v)
            * (
                2
                if arm
                in (
                    "future_seed_slot_state_gdn2",
                    "future_seed_redundant_address_gdn2",
                    "future_seed_momentum_delta",
                    "future_seed_momentum_phase",
                    "future_seed_momentum_prediction_key",
                    "future_seed_momentum_log_spd",
                    "future_seed_momentum_refresh",
                    "future_seed_predictive_momentum",
                )
                else 1
            )
            + (4_096 if arm == "future_seed_raven_address_gdn2" else 0)
            + (8_192 if arm == "future_seed_linear_product_state_gdn2" else 0)
            + (4_096 if arm == "future_seed_sparse_delta_slot_gdn2" else 0)
            + (4_096 if arm == "future_seed_shared_eligibility_gdn2" else 0)
            + (4_096 if arm == "future_seed_cycle_memory_gdn2" else 0)
        ),
        "init_hash": init_hash,
        "init_parameter_hash": init_parameter_hash,
        "parent_init_parameter_hash": parent_init_parameter_hash,
        "trained_parameter_hash": trained_parameter_hash,
        "data_hashes": data_hashes,
        "warmup_batch_hash": warmup_batch_hash,
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "epochs": max_epochs,
        "train_examples": TRAIN_EXAMPLES,
        "train_tokens": TRAIN_EXAMPLES * sequence_length * max_epochs,
        "elapsed_sec_including_validation": elapsed,
        "cold_arm_wall_sec_through_checkpoint": cold_arm_wall_sec,
        "post_warm_arm_wall_sec_through_checkpoint": post_warm_arm_wall_sec,
        "peak_training_cuda_mem_bytes": training_peak,
        "metrics": metrics,
        "valid_curve": logger.rows,
        "warmed_step_benchmark": benchmark,
        "checkpoint_path": None if checkpoint_path is None else str(checkpoint_path),
        "checkpoint_sha256": checkpoint_sha256,
    }
    if future_seed is not None:
        score["future_seed"] = future_seed
    if momentum_delta is not None:
        score["momentum_delta"] = momentum_delta
    if official_sdm is not None:
        score["official_sdm"] = official_sdm
    if comba is not None:
        score["comba"] = comba
    if mesa is not None:
        score["mesa"] = mesa
    if slot_state is not None:
        score["slot_state"] = slot_state
    if raven_address is not None:
        score["raven_address"] = raven_address
    if linear_product_state is not None:
        score["linear_product_state"] = linear_product_state
    if query_delta is not None:
        score["query_delta"] = query_delta
    if qk_coherence is not None:
        score["qk_coherence"] = qk_coherence
    if biorthogonal_qk is not None:
        score["biorthogonal_qk"] = biorthogonal_qk
    if dynamic_frame is not None:
        score["dynamic_frame"] = dynamic_frame
    if receiver_read_credit is not None:
        score["receiver_read_credit"] = receiver_read_credit
    if coherent_key_spectrum is not None:
        score["coherent_key_spectrum"] = coherent_key_spectrum
    if committed_interference is not None:
        score["committed_interference"] = committed_interference
    if stable_token_address is not None:
        score["stable_token_address"] = stable_token_address
    if binding_certificate is not None:
        score["binding_certificate"] = binding_certificate
    if canonical_address_companion is not None:
        score["canonical_address_companion"] = canonical_address_companion
    if sparse_delta_slots is not None:
        score["sparse_delta_slots"] = sparse_delta_slots
    if address_payload_gauge is not None:
        score["address_payload_gauge"] = address_payload_gauge
    if producer_readout is not None:
        score["producer_readout"] = producer_readout
        score["producer_readout_edge_off_metrics"] = producer_readout_edge_off
        score["producer_readout_diagnostic_extra_wall_sec"] = (
            diagnostic_extra_wall_sec
        )
    if readonly_dual_plane is not None:
        score["readonly_dual_plane"] = readonly_dual_plane
        score["readonly_dual_plane_edge_off_metrics"] = (
            readonly_dual_plane_edge_off
        )
        score["readonly_dual_plane_diagnostic_extra_wall_sec"] = (
            diagnostic_extra_wall_sec
        )
    if lagged_commit is not None:
        score["lagged_commit"] = lagged_commit
        score["lagged_commit_edge_off_metrics"] = lagged_commit_edge_off
        score["lagged_commit_diagnostic_extra_wall_sec"] = (
            diagnostic_extra_wall_sec
        )
    if redundant_address is not None:
        score["redundant_address"] = redundant_address
    if block_rls is not None:
        score["block_rls"] = block_rls
    if local_binding is not None:
        score["local_binding"] = local_binding
    if shared_eligibility is not None:
        score["shared_eligibility"] = shared_eligibility
    if pair_event is not None:
        score["pair_event"] = pair_event
    if cycle_memory is not None:
        score["cycle_memory"] = cycle_memory
    if arm in (
        "future_seed_gdn2_log_spd",
        "future_seed_gdn2_metric_pullback",
        "future_seed_gdn2_shared_log_spd",
        "future_seed_gdn2_log_spd_block_gram",
    ):
        score["log_spd"] = log_spd
    if metric_pullback is not None:
        score["metric_pullback"] = metric_pullback
    if block_gram is not None:
        score["block_gram"] = block_gram
    if arm == "future_seed_gdn2_two_edit":
        score["two_edit"] = two_edit
    if arm == "future_seed_gdn2_dual_hash":
        score["dual_hash"] = dual_hash
    if committed_delta is not None:
        score["committed_delta"] = committed_delta
    if oig is not None:
        score["oig"] = oig
    if event_tape is not None:
        score["event_tape"] = event_tape
    if atomic_pair is not None:
        score["atomic_pair"] = atomic_pair
    if gated_delta_product is not None:
        score["gated_delta_product"] = gated_delta_product
    if erase_then_delta is not None:
        score["erase_then_delta"] = erase_then_delta
    if contractive_dplr is not None:
        score["contractive_dplr"] = contractive_dplr
    if decoupled_key is not None:
        score["decoupled_key"] = decoupled_key
    if anchored_dual_key is not None:
        score["anchored_dual_key"] = anchored_dual_key
    if committed_residual is not None:
        score["committed_residual"] = committed_residual
    if chunk_local_biaxis is not None:
        score["chunk_local_biaxis"] = chunk_local_biaxis
    if surprise_regression is not None:
        score["surprise_regression"] = surprise_regression
    logger.finish()
    (arm_dir / "config.json").write_text(
        json.dumps(config.model_dump(mode="json"), indent=2, sort_keys=True) + "\n"
    )
    (arm_dir / "score.json").write_text(
        json.dumps(score, indent=2, sort_keys=True) + "\n"
    )
    (arm_dir / "cases.json").write_text(
        json.dumps(cases, separators=(",", ":")) + "\n"
    )
    if producer_readout_edge_off_cases is not None:
        (arm_dir / "edge_off_cases.json").write_text(
            json.dumps(
                producer_readout_edge_off_cases,
                separators=(",", ":"),
            )
            + "\n"
        )
    if readonly_dual_plane_edge_off_cases is not None:
        (arm_dir / "readonly_edge_off_cases.json").write_text(
            json.dumps(
                readonly_dual_plane_edge_off_cases,
                separators=(",", ":"),
            )
            + "\n"
        )
    if lagged_commit_edge_off_cases is not None:
        (arm_dir / "lag_edge_off_cases.json").write_text(
            json.dumps(
                lagged_commit_edge_off_cases,
                separators=(",", ":"),
            )
            + "\n"
        )
    del trainer, model
    torch.cuda.empty_cache()
    return score


def _safe_ratio(numerator: float, denominator: float) -> float | None:
    if abs(denominator) < 1e-12:
        return None
    return numerator / denominator


def summarize_length(scores: dict[str, dict[str, Any]]) -> dict[str, Any]:
    causal = scores["causal_gdn2"]
    future_seed = scores["future_seed_gdn2"]
    attention = scores["bidirectional_attention"]
    if causal["init_hash"] != future_seed["init_hash"]:
        raise RuntimeError("Matched GDN2 arms did not start identically")
    if causal["init_parameter_hash"] != future_seed["init_parameter_hash"]:
        raise RuntimeError("Matched GDN2 parameter hashes differ")
    if len({score["data_hashes"]["train"] for score in scores.values()}) != 1:
        raise RuntimeError("The three arms did not use identical training data")
    if len({score["data_hashes"]["test"] for score in scores.values()}) != 1:
        raise RuntimeError("The three arms did not use identical test data")

    causal_future = causal["metrics"]["future"]["accuracy"]
    fs_future = future_seed["metrics"]["future"]["accuracy"]
    attention_future = attention["metrics"]["future"]["accuracy"]
    return {
        "arms": scores,
        "future_seed_vs_causal": {
            "future_accuracy_delta": fs_future - causal_future,
            "past_accuracy_delta": (
                future_seed["metrics"]["past"]["accuracy"]
                - causal["metrics"]["past"]["accuracy"]
            ),
            "joint_exact_delta": (
                future_seed["metrics"]["joint_exact"]
                - causal["metrics"]["joint_exact"]
            ),
        },
        "future_seed_gap_closure_vs_attention": _safe_ratio(
            fs_future - causal_future,
            attention_future - causal_future,
        ),
        "attention_parameter_delta_fraction_vs_gdn2": (
            attention["parameters"] - causal["parameters"]
        )
        / causal["parameters"],
    }


def length64_carrier_gate(summary: dict[str, Any]) -> dict[str, Any]:
    arms = summary["arms"]
    causal = arms["causal_gdn2"]["metrics"]
    future_seed = arms["future_seed_gdn2"]["metrics"]
    attention = arms["bidirectional_attention"]["metrics"]
    checks = {
        "causal_past_at_least_0.90": causal["past"]["accuracy"] >= 0.90,
        "causal_future_at_most_0.10": causal["future"]["accuracy"] <= 0.10,
        "future_seed_past_at_least_0.90": future_seed["past"]["accuracy"] >= 0.90,
        "future_seed_future_at_least_0.90": (
            future_seed["future"]["accuracy"] >= 0.90
        ),
        "attention_past_at_least_0.90": attention["past"]["accuracy"] >= 0.90,
        "attention_future_at_least_0.90": (
            attention["future"]["accuracy"] >= 0.90
        ),
    }
    return {"passed": all(checks.values()), "checks": checks}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--sequence-lengths",
        type=int,
        nargs="+",
        default=[64, 1024],
    )
    parser.add_argument("--num-kv-pairs", type=int, default=4)
    parser.add_argument("--max-epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--enforce-length64-gate", action="store_true")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    summaries: dict[str, Any] = {}
    for sequence_length in args.sequence_lengths:
        scores = {
            arm: run_arm(
                arm=arm,
                sequence_length=sequence_length,
                num_kv_pairs=args.num_kv_pairs,
                output_dir=args.output_dir,
                max_epochs=args.max_epochs,
                batch_size=args.batch_size,
            )
            for arm in ARMS
        }
        summary = summarize_length(scores)
        summaries[str(sequence_length)] = summary
        (args.output_dir / f"length_{sequence_length}" / "comparison.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n"
        )
        if sequence_length == 64:
            train_hash = scores["causal_gdn2"]["data_hashes"]["train"]
            test_hash = scores["causal_gdn2"]["data_hashes"]["test"]
            if train_hash != P007_LENGTH64_TRAIN_HASH:
                raise RuntimeError("Length-64 training data drifted from P-CAUSAL-007")
            if test_hash != P007_LENGTH64_TEST_HASH:
                raise RuntimeError("Length-64 test data drifted from P-CAUSAL-007")
            gate = length64_carrier_gate(summary)
            (args.output_dir / "length_64" / "carrier_gate.json").write_text(
                json.dumps(gate, indent=2, sort_keys=True) + "\n"
            )
            if args.enforce_length64_gate and not gate["passed"]:
                raise RuntimeError(f"Length-64 carrier gate failed: {gate['checks']}")

    protocol = {
        "sequence_lengths": args.sequence_lengths,
        "num_kv_pairs": args.num_kv_pairs,
        "train_examples_per_arm": TRAIN_EXAMPLES,
        "validation_examples_per_arm": VALID_EXAMPLES,
        "epochs": args.max_epochs,
        "batch_size": args.batch_size,
        "seed": SEED,
        "model_width": MODEL_WIDTH,
        "model_layers": MODEL_LAYERS,
        "gdn2_heads": MODEL_HEADS,
        "gdn2_head_dim": GDN2_HEAD_DIM,
        "gdn2_expand_v": 1.0,
        "attention_heads": MODEL_HEADS,
        "attention_head_dim": ATTENTION_HEAD_DIM,
        "loops": "not used; this experiment isolates cross-layer state seeding",
    }
    comparison = {"protocol": protocol, "lengths": summaries}
    (args.output_dir / "comparison.json").write_text(
        json.dumps(comparison, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(comparison, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
