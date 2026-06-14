"""
Binary-to-DNA encoding strategies.

Provides:
    - Fixed baseline mapping (00->A, 01->C, 10->G, 11->T)
    - All 24 permutation-based mappings of the four nucleotides
    - Block-level adaptive encoding where each half of the data
      can use a different mapping to improve sequence quality
    - Composite action space: mapping x block-level adaptation

Post-processing (GC balancing, homopolymer smoothing) is intentionally
not applied because it would break the one-to-one mapping needed for
accurate decoding. The reinforcement learning agent instead learns to
pick the mapping permutation that naturally yields better-quality sequences.
"""

import itertools
import numpy as np
from typing import Dict, List, Tuple

from sequence_metrics import gc_content, max_homopolymer


NUCLEOTIDES = ["A", "C", "G", "T"]

# the four dibits in canonical order
DIBITS: List[Tuple[int, int]] = [(0, 0), (0, 1), (1, 0), (1, 1)]


# all 24 permutation mappings

def _build_all_mappings() -> List[Dict[Tuple[int, int], str]]:
    """Generate all 24 permutations of nucleotide assignments to dibits."""
    mappings: List[Dict[Tuple[int, int], str]] = []
    for permutation in itertools.permutations(NUCLEOTIDES):
        mapping = {dibit: nucleotide for dibit, nucleotide in zip(DIBITS, permutation)}
        mappings.append(mapping)
    return mappings

ALL_MAPPINGS = _build_all_mappings()

# index 0 is the canonical baseline mapping (A, C, G, T)
BASELINE_MAPPING_INDEX = 0
BASELINE_MAPPING = ALL_MAPPINGS[BASELINE_MAPPING_INDEX]

NUMBER_OF_MAPPINGS = len(ALL_MAPPINGS)         # 24


# action space
#
# The agent picks:
#   - a primary mapping (0-23)
#   - a secondary mapping for the second half (0-23)
# This gives block-level adaptive encoding without post-processing.
# Total actions = 24 (primary) x 4 (secondary group) = 96

NUMBER_OF_SECONDARY_GROUPS = 4    # secondary mapping = (primary + offset) mod 24
NUMBER_OF_ACTIONS = NUMBER_OF_MAPPINGS * NUMBER_OF_SECONDARY_GROUPS  # 96


def decode_action(action: int) -> Tuple[int, int]:
    """
    Decode a flat action index in [0, 96) into
    (primary_mapping_index, secondary_mapping_index).
    """
    primary = action % NUMBER_OF_MAPPINGS
    offset = action // NUMBER_OF_MAPPINGS
    secondary = (primary + offset * 6) % NUMBER_OF_MAPPINGS
    return primary, secondary


def encode_action(primary: int, secondary_offset: int) -> int:
    """Inverse of decode_action (offset group index, not raw secondary)."""
    return primary + NUMBER_OF_MAPPINGS * secondary_offset


# core encoding functions

def generate_random_binary(number_of_bits: int, random_generator: np.random.Generator) -> np.ndarray:
    """Generate a random binary array of length number_of_bits (must be even)."""
    assert number_of_bits % 2 == 0, "number_of_bits must be even for dibit encoding"
    return random_generator.integers(0, 2, size=number_of_bits, dtype=np.int8)


def generate_biased_binary(number_of_bits: int, random_generator: np.random.Generator) -> np.ndarray:
    """
    Generate binary data with non-uniform dibit distribution.
    Uses a random Dirichlet draw to create skewed data each call.
    """
    assert number_of_bits % 2 == 0
    number_of_dibits = number_of_bits // 2
    weights = random_generator.dirichlet(alpha=[0.5, 0.5, 0.5, 0.5])
    dibit_indices = random_generator.choice(4, size=number_of_dibits, p=weights)
    dibit_table = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.int8)
    return dibit_table[dibit_indices].flatten()


# controlled data distribution profiles
#
# With baseline mapping (00->A, 01->C, 10->G, 11->T):
#   dibit 00 -> A (AT),  01 -> C (GC),  10 -> G (GC),  11 -> T (AT)
# So GC content = frequency(01) + frequency(10)

def generate_high_gc_binary(number_of_bits: int, random_generator: np.random.Generator) -> np.ndarray:
    """Data biased toward dibits 01 and 10 -> high GC under baseline mapping."""
    assert number_of_bits % 2 == 0
    number_of_dibits = number_of_bits // 2
    weights = np.array([0.10, 0.35, 0.35, 0.20])
    # add small noise for variety
    noise = random_generator.dirichlet(alpha=[2.0, 2.0, 2.0, 2.0])
    weights = 0.7 * weights + 0.3 * noise
    weights /= weights.sum()
    dibit_indices = random_generator.choice(4, size=number_of_dibits, p=weights)
    dibit_table = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.int8)
    return dibit_table[dibit_indices].flatten()


def generate_low_gc_binary(number_of_bits: int, random_generator: np.random.Generator) -> np.ndarray:
    """Data biased toward dibits 00 and 11 -> low GC under baseline mapping."""
    assert number_of_bits % 2 == 0
    number_of_dibits = number_of_bits // 2
    weights = np.array([0.35, 0.10, 0.20, 0.35])
    noise = random_generator.dirichlet(alpha=[2.0, 2.0, 2.0, 2.0])
    weights = 0.7 * weights + 0.3 * noise
    weights /= weights.sum()
    dibit_indices = random_generator.choice(4, size=number_of_dibits, p=weights)
    dibit_table = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.int8)
    return dibit_table[dibit_indices].flatten()


def generate_balanced_binary(number_of_bits: int, random_generator: np.random.Generator) -> np.ndarray:
    """Data with roughly equal dibit frequencies -> ~50% GC under any mapping."""
    assert number_of_bits % 2 == 0
    number_of_dibits = number_of_bits // 2
    weights = np.array([0.25, 0.25, 0.25, 0.25])
    noise = random_generator.dirichlet(alpha=[10.0, 10.0, 10.0, 10.0])  # high alpha = tight
    weights = 0.8 * weights + 0.2 * noise
    weights /= weights.sum()
    dibit_indices = random_generator.choice(4, size=number_of_dibits, p=weights)
    dibit_table = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.int8)
    return dibit_table[dibit_indices].flatten()


# the three profiles as a list for curriculum selection
DATA_PROFILES = [generate_high_gc_binary, generate_low_gc_binary, generate_balanced_binary]
PROFILE_NAMES = ["high_gc", "low_gc", "balanced"]


def generate_curriculum_binary(
    number_of_bits: int, random_generator: np.random.Generator,
    epoch: int, total_epochs: int,
) -> np.ndarray:
    """
    Curriculum-based data generation:
      Early   (0-33%):  balanced data only
      Middle (33-66%):  50% balanced, 50% random profile
      Late   (66-100%): all three profiles equally
    """
    progress = epoch / max(total_epochs, 1)

    if progress < 0.33:
        return generate_balanced_binary(number_of_bits, random_generator)
    elif progress < 0.66:
        if random_generator.random() < 0.5:
            return generate_balanced_binary(number_of_bits, random_generator)
        else:
            profile_index = random_generator.integers(0, len(DATA_PROFILES))
            return DATA_PROFILES[profile_index](number_of_bits, random_generator)
    else:
        profile_index = random_generator.integers(0, len(DATA_PROFILES))
        return DATA_PROFILES[profile_index](number_of_bits, random_generator)


def encode_binary_to_dna(binary_bits: np.ndarray,
                         mapping: Dict[Tuple[int, int], str]) -> str:
    """
    Convert a binary array to a DNA string using the given dibit-to-nucleotide
    mapping. Pure reversible encoding, no post-processing applied.
    """
    assert len(binary_bits) % 2 == 0
    sequence: List[str] = []
    for index in range(0, len(binary_bits), 2):
        dibit = (int(binary_bits[index]), int(binary_bits[index + 1]))
        sequence.append(mapping[dibit])
    return "".join(sequence)


# block-adaptive encoder (used by the reinforcement learning agent)

def encode_with_action(binary_bits: np.ndarray, action: int,
                       random_generator: np.random.Generator) -> Tuple[str, int, int]:
    """
    Block-adaptive encoding:
      - first half of the bit array uses the primary mapping
      - second half uses the secondary mapping

    This gives the agent control over sequence-level properties
    without corrupting the encoding.

    Returns (dna_string, primary_index, secondary_index).
    """
    primary_index, secondary_index = decode_action(action)
    primary_mapping = ALL_MAPPINGS[primary_index]
    secondary_mapping = ALL_MAPPINGS[secondary_index]

    half = (len(binary_bits) // 4) * 2    # ensure even split point (in bits)
    first_half = encode_binary_to_dna(binary_bits[:half], primary_mapping)
    second_half = encode_binary_to_dna(binary_bits[half:], secondary_mapping)

    return first_half + second_half, primary_index, secondary_index


def decode_with_action(dna_sequence: str, action: int,
                       sequence_length: int) -> np.ndarray:
    """
    Decode a block-adaptively encoded DNA sequence.
    Uses the same primary/secondary split as encode_with_action.
    """
    primary_index, secondary_index = decode_action(action)
    primary_mapping = ALL_MAPPINGS[primary_index]
    secondary_mapping = ALL_MAPPINGS[secondary_index]

    # sequence_length is DNA length; half in DNA nucleotides
    dna_half = sequence_length // 2

    from decoder import decode_dna_to_binary
    first_bits = decode_dna_to_binary(dna_sequence[:dna_half], primary_mapping)
    second_bits = decode_dna_to_binary(dna_sequence[dna_half:], secondary_mapping)

    return np.concatenate([first_bits, second_bits])
