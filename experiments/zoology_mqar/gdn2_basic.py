from zoology.config import DataConfig, ModelConfig, ModuleConfig, TrainConfig
from zoology.data.multiquery_ar import MQARConfig


config = TrainConfig(
    data=DataConfig(
        train_configs=[
            MQARConfig(
                num_examples=10_000,
                vocab_size=256,
                input_seq_len=64,
                num_kv_pairs=4,
            )
        ],
        test_configs=[
            MQARConfig(
                num_examples=1_000,
                vocab_size=256,
                input_seq_len=64,
                num_kv_pairs=4,
            )
        ],
    ),
    model=ModelConfig(
        vocab_size=256,
        max_position_embeddings=64,
        sequence_mixer=ModuleConfig(
            name="experiments.zoology_mqar.gdn2_mixer.ZoologyGDN2Mixer",
            kwargs={
                "num_heads": 4,
                "head_dim": 32,
                "expand_v": 1.0,
                "conv_size": 4,
            },
        ),
    ),
)

configs = [config]
