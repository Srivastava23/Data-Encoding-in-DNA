"""
Multi-copy redundancy system for DNA data storage.

Stores N independent copies of each encoded DNA strand and uses
position-wise majority voting during decoding to correct retrieval
errors. Even if one copy has an error at a given position, the
other copies will outvote it.

This module works alongside the existing encoding/decoding pipeline.
"""

import numpy as np
from typing import Dict, List, Tuple
from collections import Counter

from training_pipeline_v3 import _compute_block_state
from encoding import get_block_splits, decode_block_sequence, encode_binary_to_dna, ALL_MAPPINGS
import torch
from config import Config
from encoding import encode_with_action, decode_with_action
from error_simulation import simulate_errors, compute_error_rate
from decoder import decode_dna_to_binary, compute_bit_accuracy


def create_redundant_copies(
    dna_sequence: str,
    number_of_copies: int,
    substitution_rate: float,
    insertion_rate: float,
    deletion_rate: float,
    random_generator: np.random.Generator,
) -> List[Dict]:
    """
    Simulate storing multiple copies of a DNA strand. Each copy
    independently goes through the error simulation (representing
    independent synthesis/sequencing events).

    Returns a list of dictionaries, one per copy, each containing:
        - "mutated_sequence": the copy after errors
        - "error_statistics": counts of substitutions, insertions, deletions
        - "error_rate": positional error rate versus the original
    """
    copies = []
    for copy_index in range(number_of_copies):
        mutated_sequence, error_statistics = simulate_errors(
            dna_sequence,
            substitution_rate=substitution_rate,
            insertion_rate=insertion_rate,
            deletion_rate=deletion_rate,
            random_generator=random_generator,
        )
        error_rate = compute_error_rate(dna_sequence, mutated_sequence)
        copies.append({
            "copy_index": copy_index,
            "mutated_sequence": mutated_sequence,
            "error_statistics": error_statistics,
            "error_rate": error_rate,
        })
    return copies


def consensus_vote(copies: List[str]) -> str:
    """
    Perform position-wise majority voting across multiple DNA strand copies.

    At each position, the nucleotide that appears most frequently across
    all copies is chosen for the consensus sequence. If copies have
    different lengths (due to insertions/deletions), we align from the
    start and use the median length.

    Returns the consensus DNA string.
    """
    if not copies:
        return ""

    if len(copies) == 1:
        return copies[0]

    # use the median length to handle insertion/deletion variations
    lengths = [len(copy) for copy in copies]
    consensus_length = int(np.median(lengths))

    consensus_nucleotides = []
    for position in range(consensus_length):
        nucleotide_votes = []
        for copy in copies:
            if position < len(copy):
                nucleotide_votes.append(copy[position])

        if nucleotide_votes:
            # majority vote: pick the most common nucleotide
            vote_counts = Counter(nucleotide_votes)
            winning_nucleotide = vote_counts.most_common(1)[0][0]
            consensus_nucleotides.append(winning_nucleotide)

    return "".join(consensus_nucleotides)


def encode_with_redundancy(
    binary_bits: np.ndarray,
    action: int,
    random_generator: np.random.Generator,
    config: Config,
    agent = None,
) -> Dict:
    splits = get_block_splits(len(binary_bits), config)
    N = len(splits)
    block_actions = []
    dna_so_far = ""
    if agent:
        actor_hx, critic_hx = None, None
        agent.network.eval()
        with torch.no_grad():
            for k in range(N):
                state = _compute_block_state(binary_bits, splits, k, N, block_actions, dna_so_far)
                state_t = torch.tensor(state, dtype=torch.float32).to(agent.device)
                logits, _, actor_hx, critic_hx = agent.network(state_t, actor_hx, critic_hx)
                action_int = torch.argmax(logits).item()
                
                b_start, b_end = splits[k]
                block_bits = binary_bits[b_start:b_end]
                mapping_idx = action_int
                if config.use_header_free_encoding and k > 0:
                    prev_s, prev_e = splits[k - 1]
                    offset = int(np.sum(binary_bits[prev_s:prev_e])) % 24
                    mapping_idx = (mapping_idx + offset) % 24
                dna_so_far += encode_binary_to_dna(block_bits, ALL_MAPPINGS[mapping_idx])
                block_actions.append(action_int)
    dna_sequence = dna_so_far
    
    copies_data = create_redundant_copies(
        dna_sequence, number_of_copies=config.number_of_copies,
        substitution_rate=config.substitution_rate, insertion_rate=config.insertion_rate,
        deletion_rate=config.deletion_rate, random_generator=random_generator,
    )
    mutated_sequences = [c["mutated_sequence"] for c in copies_data]
    consensus_sequence = consensus_vote(mutated_sequences)
    
    single_copy_decoded = decode_block_sequence(mutated_sequences[0], block_actions, config)
    single_copy_accuracy = compute_bit_accuracy(binary_bits, single_copy_decoded)
    consensus_decoded = decode_block_sequence(consensus_sequence, block_actions, config)
    redundancy_accuracy = compute_bit_accuracy(binary_bits, consensus_decoded)
    consensus_error_rate = compute_error_rate(dna_sequence, consensus_sequence)

    return {
        "original_dna": dna_sequence,
        "primary_index": 0, "secondary_index": 0,
        "copies": copies_data,
        "consensus_sequence": consensus_sequence,
        "consensus_error_rate": consensus_error_rate,
        "single_copy_decoded_bits": single_copy_decoded,
        "consensus_decoded_bits": consensus_decoded,
        "single_copy_accuracy": single_copy_accuracy,
        "redundancy_accuracy": redundancy_accuracy,
        "number_of_copies": config.number_of_copies,
        "average_copy_error_rate": float(np.mean([c["error_rate"] for c in copies_data])),
    }



def evaluate_redundancy_comparison(
    binary_bits: np.ndarray, action: int, random_generator: np.random.Generator,
    config: Config, agent = None, copy_counts: List[int] = None,
) -> List[Dict]:
    if copy_counts is None: copy_counts = [1, 3, 5]
    results = []
    for count in copy_counts:
        test_config = Config(number_of_copies=count, substitution_rate=config.substitution_rate,
                             insertion_rate=config.insertion_rate, deletion_rate=config.deletion_rate)
        result = encode_with_redundancy(binary_bits, action, random_generator, test_config, agent=agent)
        results.append({
            "number_of_copies": count,
            "single_copy_accuracy": result["single_copy_accuracy"],
            "redundancy_accuracy": result["redundancy_accuracy"],
            "consensus_error_rate": result["consensus_error_rate"],
            "average_copy_error_rate": result["average_copy_error_rate"],
        })
    return results

