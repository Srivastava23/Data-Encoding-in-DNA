"""
DNA-to-binary decoding and accuracy metrics.
Reverses the encoding step: converts a (possibly mutated) DNA string
back to a binary array and measures accuracy against the original.
"""

import numpy as np
from typing import Dict, Tuple


def build_reverse_mapping(mapping: Dict[Tuple[int, int], str]) -> Dict[str, Tuple[int, int]]:
    """Invert a dibit-to-nucleotide mapping to nucleotide-to-dibit."""
    return {nucleotide: dibit for dibit, nucleotide in mapping.items()}


def decode_dna_to_binary(dna_sequence: str,
                         mapping: Dict[Tuple[int, int], str]) -> np.ndarray:
    """
    Decode a DNA string back to a binary array using the given mapping.
    Unknown nucleotides (not in the mapping) are decoded as (0, 0).
    """
    reverse_mapping = build_reverse_mapping(mapping)
    binary_bits = []
    for nucleotide in dna_sequence:
        dibit = reverse_mapping.get(nucleotide, (0, 0))  # fallback for unexpected characters
        binary_bits.extend(dibit)
    return np.array(binary_bits, dtype=np.int8)


def compute_bit_accuracy(original: np.ndarray, decoded: np.ndarray) -> float:
    """
    Fraction of bits that match between original and decoded arrays.
    Handles length mismatches by comparing up to the shorter length
    and penalising the excess.
    """
    minimum_length = min(len(original), len(decoded))
    maximum_length = max(len(original), len(decoded))
    if maximum_length == 0:
        return 1.0
    matches = int(np.sum(original[:minimum_length] == decoded[:minimum_length]))
    return matches / maximum_length


def compute_recovery_rate(original: np.ndarray, decoded: np.ndarray) -> float:
    """
    Returns 1.0 if the decoded array exactly matches the original, else 0.0.
    Averaged over many trials this gives a 'sequence recovery rate'.
    """
    if len(original) != len(decoded):
        return 0.0
    return 1.0 if np.array_equal(original, decoded) else 0.0


def compute_error_percentage(original: np.ndarray, decoded: np.ndarray) -> float:
    """Percentage of bits that differ (complement of bit accuracy * 100)."""
    return (1.0 - compute_bit_accuracy(original, decoded)) * 100.0


def compute_all_decode_metrics(original: np.ndarray,
                                decoded: np.ndarray) -> Dict[str, float]:
    """Return all decoding metrics in a single dictionary."""
    return {
        "bit_accuracy": compute_bit_accuracy(original, decoded),
        "recovery_rate": compute_recovery_rate(original, decoded),
        "error_percentage": compute_error_percentage(original, decoded),
    }
