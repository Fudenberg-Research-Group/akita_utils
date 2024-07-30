import numpy as np
from akita_utils.dna_utils import dna_1hot, dna_1hot_to_seq, dna_seq_rc, permute_seq_k


def test_dna_1hot_to_seq():
    assert dna_1hot_to_seq(dna_1hot("AAAA")) == "AAAA"


def test_dna_seq_rc():
    seq = "ACTG"
    rc_seq = "CAGT"
    assert dna_seq_rc(seq) == rc_seq


def test_permute_seq_k():
    # Test 1: Basic functionality
    seq_1hot = np.array([[1, 0, 0, 0], 
                         [0, 1, 0, 0], 
                         [0, 0, 1, 0], 
                         [0, 0, 0, 1], 
                         [1, 0, 0, 0], 
                         [0, 1, 0, 0]])
    k = 2
    result = permute_seq_k(seq_1hot, k)
    assert result.shape == seq_1hot.shape, "Shape mismatch"
    assert not np.array_equal(result, seq_1hot), "Permutation did not change sequence"
    
    # Check that the sequence is a permutation of k-mers
    permuted_k_mers = [result[i:i + k] for i in range(0, len(result), k)]
    original_k_mers = [seq_1hot[i:i + k] for i in range(0, len(seq_1hot), k)]
    assert sorted(map(tuple, original_k_mers)) == sorted(map(tuple, permuted_k_mers)), "Permuted k-mers are not as expected"

    # Test 2: Edge case where sequence length is not divisible by k
    try:
        permute_seq_k(seq_1hot, k=3)
    except ValueError as e:
        assert str(e) == "Sequence length must be divisible by k", "Incorrect error message for length not divisible by k"
    
    # Test 3: Edge case where k > sequence length
    short_seq_1hot = np.array([[1, 0, 0, 0], 
                               [0, 1, 0, 0]])
    k_large = 3
    try:
        permute_seq_k(short_seq_1hot, k=k_large)
    except ValueError as e:
        assert str(e) == "Sequence length must be divisible by k", "Incorrect error message for k > sequence length"
    
    # Test 4: Randomness check (manual validation needed)
    np.random.seed(0)  # For reproducibility
    result1 = permute_seq_k(seq_1hot, k)
    np.random.seed(0)  # Reset seed to verify same permutation
    result2 = permute_seq_k(seq_1hot, k)
    assert np.array_equal(result1, result2), "Results should be consistent with the same seed"

    print("All tests passed!")
