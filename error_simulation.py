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
    """
    if random_generator is None:
        random_generator = np.random.default_rng()

    seq_len = len(dna_sequence)
    if seq_len == 0:
        return "", {"substitutions": 0, "insertions": 0, "deletions": 0}

    seq_arr = np.array(list(dna_sequence))
    
    # Vectorized homopolymer calculation
    run_lengths = np.ones(seq_len, dtype=int)
    changes = np.concatenate(([True], seq_arr[1:] != seq_arr[:-1], [True]))
    change_idx = np.where(changes)[0]
    runs = np.diff(change_idx)
    for start, length in zip(change_idx[:-1], runs):
        run_lengths[start:start+length] = length

    # Vectorized GC calculation
    is_gc = (seq_arr == 'G') | (seq_arr == 'C')
    cum_gc = np.concatenate(([0], np.cumsum(is_gc)))
    starts = np.maximum(0, np.arange(seq_len) - 5)
    ends = np.minimum(seq_len, np.arange(seq_len) + 5)
    gc_local = (cum_gc[ends] - cum_gc[starts]) / (ends - starts)

    # Multipliers
    multiplier = np.ones(seq_len, dtype=float)
    
    mask_hp = run_lengths >= 3
    multiplier[mask_hp] *= (1.0 + 0.3 * (run_lengths[mask_hp] - 2))
    
    mask_gc = (gc_local < 0.35) | (gc_local > 0.65)
    multiplier[mask_gc] *= (1.0 + 2.0 * np.abs(gc_local[mask_gc] - 0.5))

    local_sub = np.minimum(substitution_rate * multiplier, 0.5)
    local_ins = np.minimum(insertion_rate * multiplier, 0.3)
    local_del = np.minimum(deletion_rate * multiplier, 0.3)

    r1 = random_generator.random(seq_len)
    r2 = random_generator.random(seq_len)

    deletions = r1 < local_del
    substitutions = (r1 >= local_del) & (r1 < local_del + local_sub)
    insertions = r2 < local_ins

    stats = {
        "substitutions": int(np.sum(substitutions)),
        "insertions": int(np.sum(insertions)),
        "deletions": int(np.sum(deletions))
    }

    # Pre-generate mutations
    sub_choices = []
    if stats["substitutions"] > 0:
        for i in np.where(substitutions)[0]:
            orig = seq_arr[i]
            alts = [n for n in NUCLEOTIDES if n != orig]
            sub_choices.append(alts[random_generator.integers(0, 3)])
            
    ins_choices = []
    if stats["insertions"] > 0:
        ins_choices = [NUCLEOTIDES[i] for i in random_generator.integers(0, 4, size=stats["insertions"])]

    # Build sequence
    out = []
    sub_idx = 0
    ins_idx = 0
    for i in range(seq_len):
        if not deletions[i]:
            if substitutions[i]:
                out.append(sub_choices[sub_idx])
                sub_idx += 1
            else:
                out.append(seq_arr[i])
        
        if insertions[i]:
            out.append(ins_choices[ins_idx])
            ins_idx += 1

    return "".join(out), stats


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
