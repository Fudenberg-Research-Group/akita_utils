from akita_utils.dna_utils import dna_1hot, dna_1hot_to_seq, dna_seq_rc
from akita_utils.seq_gens import _insert_casette
import akita_utils
import pytest
import numpy as np

def test_insert_casette():

    motif1 = "GTAC"
    motif2 = "CGTCG"
    toy_genome_open = dna_1hot("TT" + motif1 + "TT" + motif2 + "TT")
    background_1hot = dna_1hot("AAAAAAAAAA")

    # first test
    first_flank_bp = 1
    first_spacer_bp = 0
    first_seq_1hot_insertion = toy_genome_open[
        (2 - first_flank_bp) : (6 + first_flank_bp)
    ]
    # toy_genome_open[(2-flank_bp):(6+flank_bp) corresponds to "TGTACT" (motif1 + 1bp-flanks)
    first_orientation_string = ">"

    assert (
        dna_1hot_to_seq(
            _insert_casette(
                background_1hot,
                first_seq_1hot_insertion,
                first_spacer_bp,
                first_orientation_string,
            )
        )
        == "AA" + "T" + motif1 + "T" + "AA"
    )
    # Ts around the motif are the 1bp flanks

    # second test
    second_flank_bp = 0
    second_spacer_bp = 0
    second_seq_1hot_insertion = toy_genome_open[
        (8 - second_flank_bp) : (13 + second_flank_bp)
    ]
    second_orientation_string = "<"
    # dna_1hot_to_seq(toy_genome_open[(8 - second_spacer_bp) : (13 + second_spacer_bp)]) corresponds to "CGTCG" (motif2)

    assert (
        dna_1hot_to_seq(
            _insert_casette(
                background_1hot,
                second_seq_1hot_insertion,
                second_spacer_bp,
                second_orientation_string,
            )
        )
        == "AAA" + dna_seq_rc(motif2) + "AA"
    )
    
    # third test (two casettes)
    third_flank_bp = 0
    third_spacer_bp = 0
    third_orientation_string = "><"
    third_seq_1hot_insertion = toy_genome_open[(2 - third_flank_bp) : (6 + third_flank_bp)]
    # toy_genome_open[(2 - third_flank_bp) : (6 + third_flank_bp)] corresponds to motif1
    
    assert (
    dna_1hot_to_seq(
        _insert_casette(
            background_1hot,
            third_seq_1hot_insertion,
            third_spacer_bp,
            third_orientation_string,
        )
    )
    == "A" + motif1 + dna_seq_rc(motif1) + "A"
    )
    

def test_multi_insert_offsets_casette():
    
    # Test that the function correctly handles a single insertion at the center of the sequence, in the forward orientation.
    seq_1hot = np.zeros((10, 4))
    seq_1hot_insertions = [np.array([[0, 0, 0, 1], [0, 0, 1, 0], [0, 1, 0, 0]])]
    offsets_bp = [1]
    orientation_string = [">"]
    expected_output = np.zeros((10, 4))
    expected_output[6:9, :] = seq_1hot_insertions[0]
    output_seq = akita_utils.seq_gens._multi_insert_offsets_casette(
        seq_1hot, seq_1hot_insertions, offsets_bp, orientation_string
    )
    assert np.array_equal(output_seq, expected_output)
    
    # Test that the function correctly handles a single insertion at the center of the sequence, in the reverse orientation.
    seq_1hot = np.zeros((10, 4))
    seq_1hot_insertions = [np.array([[0, 0, 0, 1], [0, 0, 1, 0], [0, 1, 0, 0]])]
    offsets_bp = [1]
    orientation_string = ["<"]
    expected_output = np.zeros((10, 4))
    expected_output[6:9, :] = akita_utils.dna_utils.hot1_rc(seq_1hot_insertions[0])
    output_seq = akita_utils.seq_gens._multi_insert_offsets_casette(
        seq_1hot, seq_1hot_insertions, offsets_bp, orientation_string
    )
    assert np.array_equal(output_seq, expected_output)
    
    # Test that the function correctly handles multiple non-overlapping insertions, in different orientations and at different offsets.
    seq_1hot = np.zeros((10, 4))
    seq_1hot_insertions = [
        np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]]),
        np.array([[0, 0, 0, 1], [0, 0, 1, 0], [0, 1, 0, 0]])
    ]
    offsets_bp = [-2, 2]
    orientation_string = [">", "<"]
    expected_output = np.zeros((10, 4))
    expected_output[3:6, :] = seq_1hot_insertions[0]
    expected_output[7:10, :] = akita_utils.dna_utils.hot1_rc(seq_1hot_insertions[1])
    output_seq = akita_utils.seq_gens._multi_insert_offsets_casette(
        seq_1hot, seq_1hot_insertions, offsets_bp, orientation_string
    )
    assert np.array_equal(output_seq, expected_output)
    
    # Test that the function correctly raises a ValueError when two inserted sequences overlap.
    seq_1hot = np.zeros((10, 4))
    seq_1hot_insertions = [
        np.array([[0, 0, 0, 1], [0, 0, 1, 0], [0, 1, 0, 0]]),
        np.array([[0, 1, 0, 0], [0, 0, 1, 0]])
    ]
    offsets_bp = [1, 3]
    orientation_string = [">", ">"]
    with pytest.raises(ValueError) as exc_info:
        output_seq = akita_utils.seq_gens._multi_insert_offsets_casette(
            seq_1hot, seq_1hot_insertions, offsets_bp, orientation_string
        )
    assert str(exc_info.value) == "Overlap found between inserted sequences: (6, 9), (8, 10)"
    
    # Test that the function returns the expected output when all possible arguments are empty.
    seq_1hot = np.zeros((10, 4))
    seq_1hot_insertions = []
    offsets_bp = []
    orientation_string = ""
    expected_output = np.zeros((10, 4))
    output_seq = akita_utils.seq_gens._multi_insert_offsets_casette(
        seq_1hot, seq_1hot_insertions, offsets_bp, orientation_string
    )
    assert np.array_equal(output_seq, expected_output)    