"""
Central configuration for the DNA encoding optimisation project.
Hyperparameters, seeds, file paths, and experimental settings
are stored here for reproducibility and easy tuning.

All variable names use full words — no abbreviations.
"""

from dataclasses import dataclass, field
from typing import List
import os


@dataclass
class Config:
    """Project-wide configuration dataclass."""

    # reproducibility
    seed: int = 42
    number_of_seeds: int = 7                             # independent runs for statistics

    # sequence parameters
    sequence_lengths: List[int] = field(default_factory=lambda: [64, 128, 256, 512, 1024])
    trials_per_length: int = 100                         # evaluation trials per length

    # error simulation probabilities
    substitution_rate: float = 0.02
    insertion_rate: float = 0.01
    deletion_rate: float = 0.01

    # PPO hyperparameters
    ppo_learning_rate: float = 3e-4
    ppo_discount_factor: float = 0.99
    ppo_clipping_epsilon: float = 0.2
    ppo_update_passes: int = 4                           # PPO update passes per batch
    ppo_training_epochs: int = 10000                     # total training epochs
    ppo_batch_size: int = 64
    ppo_hidden_dimension: int = 64
    ppo_entropy_coefficient: float = 0.01
    ppo_entropy_coefficient_min: float = 0.005           # anneal to this
    ppo_value_coefficient: float = 0.5
    ppo_maximum_gradient_norm: float = 0.5
    ppo_gae_lambda: float = 0.95

    # error predictor hyperparameters
    predictor_learning_rate: float = 1e-3
    predictor_hidden_dimension: int = 64
    predictor_training_interval: int = 200               # train predictor every N epochs
    predictor_batch_size: int = 128
    predictor_update_passes: int = 10
    predictor_buffer_size: int = 10000
    use_error_predictor: bool = True                     # toggle Layer 2

    # reward weights
    reward_local_weight: float = 0.5                     # local block predicted error
    reward_terminal_weight: float = 1.0                  # terminal full-sequence accuracy
    reward_accuracy_weight: float = 2.0
    reward_gc_bonus_weight: float = 1.0
    reward_homopolymer_penalty_weight: float = 0.5
    reward_error_rate_weight: float = 1.0
    reward_decode_failure_penalty: float = 2.0

    # GC content target range
    gc_target_low: float = 0.40
    gc_target_high: float = 0.60

    # output paths
    output_directory: str = "results"
    csv_filename: str = "results.csv"
    statistics_filename: str = "statistics_summary.csv"

    # training sequence length used during the reinforcement learning loop
    # lengths are sampled per sequence if using curriculum/length varying
    training_sequence_length: int = 128

    # multi-block adaptive encoding
    number_of_encoding_blocks: int = 4                   # N blocks for fixed_count policy
    block_policy: str = "fixed_count"                    # "fixed_count" or "fixed_size"
    fixed_block_size_nt: int = 32                        # nt per block for fixed_size policy
    
    use_header_free_encoding: bool = False               # if True, mapping k = f(decoded block k-1)
    header_repetition_code_length: int = 3               # times mapping index is repeated in overhead
    
    # multi-copy redundancy
    number_of_copies: int = 3                            # copies stored per data block (1-5)

    # anti-overfitting measures
    validation_split: float = 0.2                        # fraction of data used for validation
    early_stopping_patience: int = 5000                  # epochs without improvement before stopping
    dropout_rate: float = 0.1                            # dropout probability in neural networks
    use_learning_rate_scheduling: bool = True             # cosine annealing learning rate schedule
    learning_rate_warmup_epochs: int = 1000              # linear warmup before cosine annealing
    minimum_learning_rate: float = 1e-6                  # floor for learning rate scheduling

    def results_path(self, filename: str) -> str:
        os.makedirs(self.output_directory, exist_ok=True)
        return os.path.join(self.output_directory, filename)
