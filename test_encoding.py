import numpy as np
from config import Config
from encoding import (
    ALL_MAPPINGS,
    encode_with_action,
    decode_with_action,
    decode_action,
    encode_block_sequence,
    decode_block_sequence,
    get_block_splits
)

def test_encoding_match():
    print("Testing block splits...")
    config = Config()
    
    # test N=1
    config.number_of_encoding_blocks = 1
    splits = get_block_splits(128, config)
    assert splits == [(0, 128)], f"N=1 splits mismatch: {splits}"
    
    # test N=2
    config.number_of_encoding_blocks = 2
    splits = get_block_splits(128, config)
    assert splits == [(0, 64), (64, 128)], f"N=2 splits mismatch: {splits}"
    
    print("Testing equivalence...")
    rng = np.random.default_rng(42)
    binary_bits = rng.integers(0, 2, size=128, dtype=np.int8)
    
    for action in range(96):
        # old way
        legacy_dna, pi, si = encode_with_action(binary_bits, action, rng)
        legacy_decoded = decode_with_action(legacy_dna, action, 64)
        
        # new way
        config.number_of_encoding_blocks = 2
        new_dna = encode_block_sequence(binary_bits, [pi, si], config)
        new_decoded = decode_block_sequence(new_dna, [pi, si], config)
        
        assert legacy_dna == new_dna, f"DNA mismatch at action {action}"
        assert np.array_equal(legacy_decoded, new_decoded), f"Decode mismatch at action {action}"
        assert np.array_equal(binary_bits, new_decoded), f"Decode mismatch with original at action {action}"

    print("Testing header-free encoding...")
    config.use_header_free_encoding = True
    config.number_of_encoding_blocks = 4
    binary_bits = rng.integers(0, 2, size=256, dtype=np.int8)
    mappings = [1, 2, 3, 4]
    new_dna = encode_block_sequence(binary_bits, mappings, config)
    new_decoded = decode_block_sequence(new_dna, mappings, config)
    assert np.array_equal(binary_bits, new_decoded), "Header-free decode failed"

    print("All tests passed!")

if __name__ == "__main__":
    test_encoding_match()
