"""
Gym-style reinforcement learning environment for DNA encoding.

Wraps the full encode -> error-simulate -> decode pipeline.
Each epoch generates binary data using a curriculum schedule
(balanced -> mixed -> all profiles), computes dibit distribution
features as the state, and lets the agent pick an encoding strategy
conditioned on the data properties.
"""

import numpy as np
from typing import Tuple, Dict

from config import Config
from encoding import (
    generate_curriculum_binary,
    generate_biased_binary,
    encode_with_action,
    decode_with_action,
    decode_action,
    ALL_MAPPINGS,
    NUMBER_OF_ACTIONS,
    NUMBER_OF_MAPPINGS,
)
from error_simulation import simulate_errors, compute_error_rate
from decoder import compute_bit_accuracy
from sequence_metrics import (
    gc_content,
    normalized_max_homopolymer,
    shannon_entropy,
    gc_balance_score,
)


def _compute_dibit_distribution(binary_bits: np.ndarray) -> np.ndarray:
    """
    Compute the frequency of each dibit pair in the binary data.
    Returns a 4-element array: [frequency_00, frequency_01, frequency_10, frequency_11].
    """
    number_of_dibits = len(binary_bits) // 2
    counts = np.zeros(4, dtype=np.float32)
    for index in range(0, len(binary_bits), 2):
        dibit_index = int(binary_bits[index]) * 2 + int(binary_bits[index + 1])
        counts[dibit_index] += 1
    return counts / max(number_of_dibits, 1)


class DNAEncodingEnvironment:
    """
    Gym-style environment for DNA encoding optimisation.

    State vector (6 floats):
        [0] dibit-00 frequency
        [1] dibit-01 frequency
        [2] dibit-10 frequency
        [3] dibit-11 frequency
        [4] max dibit frequency (dominance measure)
        [5] sequence length / 256 (normalised)

    The dibit distribution tells the agent which dibits dominate.
    Different mappings assign dominant dibits to different nucleotides,
    producing DNA with different GC content and homopolymer profiles.
    The agent learns to pick the mapping that avoids homopolymers
    and keeps GC balanced for the given data.

    Action: integer in [0, NUMBER_OF_ACTIONS)

    Reward: weighted combination of accuracy, GC balance, homopolymer
            quality, and error rate.
    """

    STATE_DIMENSION = 6
    ACTION_DIMENSION = NUMBER_OF_ACTIONS

    def __init__(self, config: Config, random_generator: np.random.Generator):
        self.config = config
        self.random_generator = random_generator
        self.sequence_length = config.training_sequence_length
        self.number_of_bits = self.sequence_length * 2

        self._state: np.ndarray = np.zeros(self.STATE_DIMENSION, dtype=np.float32)
        self._binary_bits: np.ndarray = np.array([], dtype=np.int8)
        self._predicted_error: float = 0.0

    def set_predicted_error(self, prediction: float) -> None:
        self._predicted_error = prediction

    def set_sequence_length(self, sequence_length: int) -> None:
        self.sequence_length = sequence_length
        self.number_of_bits = sequence_length * 2

    def reset(self, epoch: int = 0, total_epochs: int = 1) -> np.ndarray:
        """
        Generate new binary data and return its features as the state.
        Pass epoch/total_epochs during training to use curriculum.
        """
        self._predicted_error = 0.0

        # curriculum during training, biased for evaluation
        if total_epochs > 1:
            self._binary_bits = generate_curriculum_binary(
                self.number_of_bits, self.random_generator, epoch, total_epochs
            )
        else:
            self._binary_bits = generate_biased_binary(self.number_of_bits, self.random_generator)

        # compute data features for the state
        dibit_distribution = _compute_dibit_distribution(self._binary_bits)
        self._state = np.array([
            dibit_distribution[0],
            dibit_distribution[1],
            dibit_distribution[2],
            dibit_distribution[3],
            float(np.max(dibit_distribution)),
            self.sequence_length / 256.0,
        ], dtype=np.float32)

        return self._state.copy()

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict]:
        """
        Encode the current epoch's binary data with the chosen action,
        simulate errors, decode, and compute reward.
        """
        configuration = self.config
        binary_bits = self._binary_bits

        # 1. encode with the chosen block-adaptive action
        dna_sequence, primary_index, secondary_index = encode_with_action(
            binary_bits, action, self.random_generator
        )

        # 2. error simulation (context-dependent)
        mutated_sequence, error_statistics = simulate_errors(
            dna_sequence,
            substitution_rate=configuration.substitution_rate,
            insertion_rate=configuration.insertion_rate,
            deletion_rate=configuration.deletion_rate,
            random_generator=self.random_generator,
        )

        # 3. decode using the same action
        decoded_bits = decode_with_action(mutated_sequence, action, self.sequence_length)
        bit_accuracy = compute_bit_accuracy(binary_bits, decoded_bits)
        error_rate = compute_error_rate(dna_sequence, mutated_sequence)
        decode_failure = (bit_accuracy < 0.5)

        # 4. sequence metrics
        gc_value = gc_content(dna_sequence)
        homopolymer_value = normalized_max_homopolymer(dna_sequence)
        entropy_value = shannon_entropy(dna_sequence)
        gc_bonus = gc_balance_score(dna_sequence, configuration.gc_target_low, configuration.gc_target_high)

        # 5. reward
        reward = (
            configuration.reward_accuracy_weight * bit_accuracy
            + configuration.reward_gc_bonus_weight * gc_bonus
            - configuration.reward_homopolymer_penalty_weight * homopolymer_value
            - configuration.reward_error_rate_weight * error_rate
            - configuration.reward_predicted_error_weight * self._predicted_error
            - configuration.reward_decode_failure_penalty * (1.0 if decode_failure else 0.0)
        )

        done = True

        information = {
            "bit_accuracy": bit_accuracy,
            "error_rate": error_rate,
            "gc_content": gc_value,
            "max_homopolymer": homopolymer_value,
            "entropy": entropy_value,
            "gc_balance": gc_bonus,
            "decode_failure": decode_failure,
            "error_statistics": error_statistics,
            "dna_sequence": dna_sequence,
            "mutated_sequence": mutated_sequence,
            "primary_index": primary_index,
            "secondary_index": secondary_index,
            "predicted_error": self._predicted_error,
            "action": action,
        }

        return self._state.copy(), reward, done, information
