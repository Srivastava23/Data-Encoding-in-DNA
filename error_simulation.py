"""
Simulates DNA sequencing errors.

Models three mutation types seen in real DNA synthesis/sequencing:
    - Substitution: a nucleotide is replaced by a different one
    - Insertion: an extra nucleotide is inserted after the current one
    - Deletion: the current nucleotide is removed

Error probabilities are modulated by local sequence context.
Homopolymer runs and extreme GC content locally increase error rates,
reflecting real Illumina / nanopore behaviour.

A numpy random Generator is required for deterministic reproducibility.
"""

import numpy as np
from typing import Tuple

NUCLEOTIDES = ["A", "C", "G", "T"]


def _local_gc(sequence: str, position: int, window: int = 10) -> float:
    """Compute GC content in a window around the given position."""
    start = max(0, position - window // 2)
    end = min(len(sequence), position + window // 2)
    segment = sequence[start:end]
    if len(segment) == 0:
        return 0.5
    return sum(1 for character in segment if character in ("G", "C")) / len(segment)


def _local_homopolymer(sequence: str, position: int) -> int:
    """Return the run length of the homopolymer that the given position belongs to."""
    if position >= len(sequence):
        return 1
    nucleotide = sequence[position]
    run_length = 1
    # extend left
    index = position - 1
    while index >= 0 and sequence[index] == nucleotide:
        run_length += 1
        index -= 1
    # extend right
    index = position + 1
    while index < len(sequence) and sequence[index] == nucleotide:
        run_length += 1
        index += 1
    return run_length


def simulate_errors(
    dna_sequence: str,
    substitution_rate: float = 0.02,
    insertion_rate: float = 0.01,
    deletion_rate: float = 0.01,
    random_generator: np.random.Generator = None,
) -> Tuple[str, dict]:
    """
    Apply stochastic, context-dependent sequencing errors.

    Local error modulation:
      - Homopolymer runs >= 3 multiply error rates by
        (1 + 0.3 * (run_length - 2))
      - GC content outside [0.35, 0.65] multiplies error rates by
        (1 + 2 * |gc - 0.5|)

    Mappings that produce fewer homopolymers and more balanced GC
    content will experience genuinely lower error rates, which is
    the signal the reinforcement learning agent exploits.
    """
    if random_generator is None:
        random_generator = np.random.default_rng()

    result = []
    statistics = {"substitutions": 0, "insertions": 0, "deletions": 0}

    for position, nucleotide in enumerate(dna_sequence):
        # context-dependent rate multiplier
        homopolymer_length = _local_homopolymer(dna_sequence, position)
        gc_local = _local_gc(dna_sequence, position)

        multiplier = 1.0
        # homopolymer penalty: runs >= 3 increase errors
        if homopolymer_length >= 3:
            multiplier *= (1.0 + 0.3 * (homopolymer_length - 2))
        # GC imbalance penalty
        if gc_local < 0.35 or gc_local > 0.65:
            multiplier *= (1.0 + 2.0 * abs(gc_local - 0.5))

        local_substitution_rate = min(substitution_rate * multiplier, 0.5)
        local_insertion_rate = min(insertion_rate * multiplier, 0.3)
        local_deletion_rate = min(deletion_rate * multiplier, 0.3)

        random_value = random_generator.random()

        # priority: deletion > substitution > pass-through
        if random_value < local_deletion_rate:
            statistics["deletions"] += 1
            continue
        elif random_value < local_deletion_rate + local_substitution_rate:
            alternatives = [n for n in NUCLEOTIDES if n != nucleotide]
            nucleotide = alternatives[random_generator.integers(0, len(alternatives))]
            statistics["substitutions"] += 1

        result.append(nucleotide)

        # independent insertion check
        if random_generator.random() < local_insertion_rate:
            inserted_nucleotide = NUCLEOTIDES[random_generator.integers(0, 4)]
            result.append(inserted_nucleotide)
            statistics["insertions"] += 1

    mutated_sequence = "".join(result)
    return mutated_sequence, statistics


def compute_error_rate(original: str, mutated: str) -> float:
    """
    Simple error rate based on positional mismatches.
    Compares character-by-character up to the shorter length, then counts
    remaining characters in the longer string as errors.
    """
    minimum_length = min(len(original), len(mutated))
    maximum_length = max(len(original), len(mutated))
    if maximum_length == 0:
        return 0.0
    mismatches = sum(1 for index in range(minimum_length) if original[index] != mutated[index])
    length_difference = maximum_length - minimum_length
    return (mismatches + length_difference) / maximum_length
