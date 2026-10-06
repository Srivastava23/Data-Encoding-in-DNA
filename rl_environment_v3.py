import numpy as np
from typing import Tuple, Dict

from config import Config
from encoding import (
    generate_curriculum_binary,
    generate_biased_binary,
    encode_binary_to_dna,
    BASELINE_MAPPING,
    ALL_MAPPINGS,
    get_block_splits,
    get_net_information_density,
    encode_block_sequence,
    decode_block_sequence
)
from error_simulation import simulate_errors, compute_error_rate
from decoder import compute_bit_accuracy
from sequence_metrics import (
    gc_content,
    normalized_max_homopolymer,
    shannon_entropy,
)

class DNAEncodingEnvironmentV3:
    STATE_DIMENSION = 8
    ACTION_DIMENSION = 24

    def __init__(self, config: Config, random_generator: np.random.Generator, predictor=None):
        self.config = config
        self.random_generator = random_generator
        self.predictor = predictor
        
        self.sequence_length = config.training_sequence_length
        self.number_of_bits = self.sequence_length * 2
        
        self.current_block = 0
        self.N = 1
        self.splits = []
        self._binary_bits = np.array([], dtype=np.int8)
        self.block_actions = []
        self.dna_so_far = ""
        
        # for normalization against random baseline
        self.baseline_accuracy = 0.0
        self.baseline_predicted_errors = []

    def set_sequence_length(self, sequence_length: int) -> None:
        self.sequence_length = sequence_length
        self.number_of_bits = sequence_length * 2

    def _get_state(self) -> np.ndarray:
        k = self.current_block
        start, end = self.splits[k]
        block_bits = self._binary_bits[start:end]
        
        # dummy encode for features
        dummy_dna = encode_binary_to_dna(block_bits, BASELINE_MAPPING)
        gc = gc_content(dummy_dna)
        hp = normalized_max_homopolymer(dummy_dna)
        ent = shannon_entropy(dummy_dna)

        prev_map = -1.0
        j1 = -1.0
        j2 = -1.0

        if k > 0:
            prev_map = float(self.block_actions[-1]) / 23.0
            if len(self.dna_so_far) >= 1:
                j1 = float(["A", "C", "G", "T"].index(self.dna_so_far[-1])) / 3.0
            if len(self.dna_so_far) >= 2:
                j2 = float(["A", "C", "G", "T"].index(self.dna_so_far[-2])) / 3.0

        return np.array([gc, hp, ent, float(k)/max(1, self.N), float(self.N)/32.0, prev_map, j1, j2], dtype=np.float32)

    def reset(self, epoch: int = 0, total_epochs: int = 1) -> np.ndarray:
        if total_epochs > 1:
            self._binary_bits = generate_curriculum_binary(
                self.number_of_bits, self.random_generator, epoch, total_epochs
            )
        else:
            self._binary_bits = generate_biased_binary(self.number_of_bits, self.random_generator)

        self.splits = get_block_splits(len(self._binary_bits), self.config)
        self.N = len(self.splits)
        self.current_block = 0
        self.block_actions = []
        self.dna_so_far = ""
        
        # compute random baseline for normalization
        random_actions = [self.random_generator.integers(0, 24) for _ in range(self.N)]
        rand_dna = encode_block_sequence(self._binary_bits, random_actions, self.config)
        mut_rand, _ = simulate_errors(rand_dna, self.config.substitution_rate, self.config.insertion_rate, self.config.deletion_rate, self.random_generator)
        dec_rand = decode_block_sequence(mut_rand, random_actions, self.config)
        self.baseline_accuracy = compute_bit_accuracy(self._binary_bits, dec_rand)
        
        self.baseline_predicted_errors = []
        if self.predictor is not None:
            for i, act in enumerate(random_actions):
                # We need features for random action
                b_start, b_end = self.splits[i]
                b_bits = self._binary_bits[b_start:b_end]
                # compute true mapping idx
                map_idx = act
                if self.config.use_header_free_encoding and i > 0:
                    prev_start, prev_end = self.splits[i-1]
                    prev_bits = self._binary_bits[prev_start:prev_end]
                    offset = int(np.sum(prev_bits)) % 24
                    map_idx = (map_idx + offset) % 24
                
                blk_dna = encode_binary_to_dna(b_bits, ALL_MAPPINGS[map_idx])
                # We need predictor to have build_features static method
                feats = self.predictor.build_features(gc_content(blk_dna), shannon_entropy(blk_dna), normalized_max_homopolymer(blk_dna), len(blk_dna), map_idx)
                err = self.predictor.predict(feats)
                self.baseline_predicted_errors.append(err)

        return self._get_state()

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict]:
        k = self.current_block
        self.block_actions.append(action)

        start, end = self.splits[k]
        block_bits = self._binary_bits[start:end]

        mapping_idx = action
        if self.config.use_header_free_encoding and k > 0:
            prev_start, prev_end = self.splits[k-1]
            prev_bits = self._binary_bits[prev_start:prev_end]
            offset = int(np.sum(prev_bits)) % 24
            mapping_idx = (mapping_idx + offset) % 24

        mapping = ALL_MAPPINGS[mapping_idx]
        block_dna = encode_binary_to_dna(block_bits, mapping)
        self.dna_so_far += block_dna
        
        # Local reward
        local_reward = 0.0
        local_pred_err = 0.0
        if self.predictor is not None:
            feats = self.predictor.build_features(gc_content(block_dna), shannon_entropy(block_dna), normalized_max_homopolymer(block_dna), len(block_dna), mapping_idx)
            local_pred_err = self.predictor.predict(feats)
            base_err = self.baseline_predicted_errors[k]
            # normalize against baseline error: positive if lower error than random
            local_reward = base_err - local_pred_err
            
        self.current_block += 1
        done = (self.current_block == self.N)
        
        reward = self.config.reward_local_weight * local_reward
        info = {
            "local_pred_error": local_pred_err,
            "action": action,
            "mapping_idx": mapping_idx
        }
        
        # Terminal reward
        if done:
            mutated_sequence, error_statistics = simulate_errors(
                self.dna_so_far,
                substitution_rate=self.config.substitution_rate,
                insertion_rate=self.config.insertion_rate,
                deletion_rate=self.config.deletion_rate,
                random_generator=self.random_generator,
            )
            
            decoded_bits = decode_block_sequence(mutated_sequence, self.block_actions, self.config)
            bit_accuracy = compute_bit_accuracy(self._binary_bits, decoded_bits)
            error_rate = compute_error_rate(self.dna_so_far, mutated_sequence)
            
            # terminal reward is normalized accuracy
            term_reward = bit_accuracy - self.baseline_accuracy
            reward += self.config.reward_terminal_weight * term_reward
            
            info.update({
                "bit_accuracy": bit_accuracy,
                "error_rate": error_rate,
                "dna_sequence": self.dna_so_far,
                "mutated_sequence": mutated_sequence,
                "baseline_accuracy": self.baseline_accuracy
            })
            
            next_state = np.zeros(self.STATE_DIMENSION, dtype=np.float32)
        else:
            next_state = self._get_state()

        return next_state, float(reward), done, info
