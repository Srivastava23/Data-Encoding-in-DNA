"""
Biophysical quality metrics for DNA sequences.

Captures properties influencing real-world sequencing error rates:
    - GC content (ideal around 0.40-0.60)
    - Homopolymer runs (shorter is better)
    - Shannon entropy (higher is better)
"""

import math
from collections import Counter
from typing import Dict


def gc_content(sequence: str) -> float:
    """Return the GC content ratio in [0, 1]."""
    if len(sequence) == 0:
        return 0.0
    guanine_cytosine_count = sum(1 for nucleotide in sequence if nucleotide in ("G", "C"))
    return guanine_cytosine_count / len(sequence)


def max_homopolymer(sequence: str) -> int:
    """Return the length of the longest homopolymer run."""
    if len(sequence) == 0:
        return 0
    maximum_run = 1
    current_run = 1
    for index in range(1, len(sequence)):
        if sequence[index] == sequence[index - 1]:
            current_run += 1
            maximum_run = max(maximum_run, current_run)
        else:
            current_run = 1
    return maximum_run


def normalized_max_homopolymer(sequence: str) -> float:
    """Max homopolymer length divided by sequence length (in [0, 1])."""
    if len(sequence) == 0:
        return 0.0
    return max_homopolymer(sequence) / len(sequence)


def shannon_entropy(sequence: str) -> float:
    """Shannon entropy of the nucleotide distribution (bits, max ~2.0 for 4 symbols)."""
    if len(sequence) == 0:
        return 0.0
    nucleotide_counts: Dict[str, int] = Counter(sequence)
    total_length = len(sequence)
    entropy_value = 0.0
    for count in nucleotide_counts.values():
        probability = count / total_length
        if probability > 0:
            entropy_value -= probability * math.log2(probability)
    return entropy_value


def gc_balance_score(sequence: str, low: float = 0.40, high: float = 0.60) -> float:
    """
    Return a score in [-1, 1].
    +1.0 if GC content is within [low, high], negative otherwise
    proportional to how far outside the range it falls.
    """
    gc_value = gc_content(sequence)
    if low <= gc_value <= high:
        return 1.0
    midpoint = (low + high) / 2.0
    return -abs(midpoint - gc_value) / midpoint   # scaled penalty


def stability_score(sequence: str) -> float:
    """
    Composite stability heuristic in [0, 1].
    Combines GC balance (positive) and homopolymer penalty (negative).
    """
    gc_value = gc_content(sequence)
    gc_component = 1.0 - 2.0 * abs(gc_value - 0.5)                    # peaks at 0.5
    homopolymer_component = 1.0 - normalized_max_homopolymer(sequence)  # lower homopolymer -> better
    return max(0.0, 0.5 * gc_component + 0.5 * homopolymer_component)


def compute_all_metrics(sequence: str) -> Dict[str, float]:
    """Return a dictionary of all metrics for convenience."""
    return {
        "gc_content": gc_content(sequence),
        "max_homopolymer": float(max_homopolymer(sequence)),
        "normalized_homopolymer": normalized_max_homopolymer(sequence),
        "entropy": shannon_entropy(sequence),
        "gc_balance": gc_balance_score(sequence),
        "stability": stability_score(sequence),
    }
