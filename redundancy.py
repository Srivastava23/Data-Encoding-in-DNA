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
) -> Dict:
    """
    Full redundancy pipeline:
    1. Encode binary data to DNA using the given action
    2. Create N copies, each with independent error simulation
    3. Build consensus via majority voting
    4. Decode the consensus strand

    Returns a dictionary with all intermediate data for visualisation:
        - "original_dna": the clean encoded DNA
        - "copies": list of per-copy data (mutated sequences, errors)
        - "consensus_sequence": the majority-voted consensus
        - "decoded_bits": binary data recovered from consensus
        - "single_copy_accuracy": accuracy from just the first copy (no redundancy)
        - "redundancy_accuracy": accuracy from the consensus (with redundancy)
        - "number_of_copies": how many copies were used
    """
    # 1. encode
    dna_sequence, primary_index, secondary_index = encode_with_action(
        binary_bits, action, random_generator
    )

    # 2. create redundant copies with independent errors
    copies_data = create_redundant_copies(
        dna_sequence,
        number_of_copies=config.number_of_copies,
        substitution_rate=config.substitution_rate,
        insertion_rate=config.insertion_rate,
        deletion_rate=config.deletion_rate,
        random_generator=random_generator,
    )

    # 3. majority voting across all copies
    mutated_sequences = [copy_data["mutated_sequence"] for copy_data in copies_data]
    consensus_sequence = consensus_vote(mutated_sequences)

    # 4. decode
    # single copy decode (first copy, no redundancy)
    single_copy_decoded = decode_with_action(
        mutated_sequences[0], action, len(dna_sequence)
    )
    single_copy_accuracy = compute_bit_accuracy(binary_bits, single_copy_decoded)

    # redundancy decode (consensus)
    consensus_decoded = decode_with_action(
        consensus_sequence, action, len(dna_sequence)
    )
    redundancy_accuracy = compute_bit_accuracy(binary_bits, consensus_decoded)

    # consensus error rate versus original
    consensus_error_rate = compute_error_rate(dna_sequence, consensus_sequence)

    return {
        "original_dna": dna_sequence,
        "primary_index": primary_index,
        "secondary_index": secondary_index,
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
    binary_bits: np.ndarray,
    action: int,
    random_generator: np.random.Generator,
    config: Config,
    copy_counts: List[int] = None,
) -> List[Dict]:
    """
    Compare decoding accuracy across different numbers of copies.

    Returns a list of results, one per copy count, showing how
    accuracy improves with more copies.
    """
    if copy_counts is None:
        copy_counts = [1, 3, 5]

    results = []
    for count in copy_counts:
        # create a temporary config with this copy count
        test_config = Config(
            number_of_copies=count,
            substitution_rate=config.substitution_rate,
            insertion_rate=config.insertion_rate,
            deletion_rate=config.deletion_rate,
        )
        result = encode_with_redundancy(
            binary_bits, action, random_generator, test_config
        )
        results.append({
            "number_of_copies": count,
            "single_copy_accuracy": result["single_copy_accuracy"],
            "redundancy_accuracy": result["redundancy_accuracy"],
            "consensus_error_rate": result["consensus_error_rate"],
            "average_copy_error_rate": result["average_copy_error_rate"],
        })

    return results
