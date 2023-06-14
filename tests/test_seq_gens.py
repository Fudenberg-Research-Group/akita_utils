import pytest
import numpy as np
from .Toy_genome import ToyGenomeOpen
from akita_utils.dna_utils import dna_1hot, dna_1hot_to_seq, dna_seq_rc, hot1_rc
from akita_utils.seq_gens import (
    _insert_casette,
    _multi_insert_offsets_casette,
    mask_spans,
    mask_spans_from_start_positions,
    permute_spans,
    mask_central_seq,
    permute_central_seq,
    fetch_centered_padded_seq_and_new_start_position,
    randomise_spans_from_start_positions,
    permute_spans_from_start_positions,
)

genome_data = {
    "chr1": "AGCTCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCG",
    "chr2": "GATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGAT",
}

toy_genome = ToyGenomeOpen(genome_data)


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
    third_seq_1hot_insertion = toy_genome_open[
        (2 - third_flank_bp) : (6 + third_flank_bp)
    ]
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


def test_mask_spans():
    seq_1hot = np.array(
        [
            [0, 1, 0, 0],
            [1, 0, 0, 0],
            [0, 0, 1, 0],
            [1, 0, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ]
    )
    spans = [(1, 3), (4, 5)]
    expected_output = [
        [0, 1, 0, 0],
        [0, 0, 0, 0],  # this should be masked
        [0, 0, 0, 0],  # this should be masked
        [1, 0, 0, 0],
        [0, 0, 0, 0],  # this should be masked
        [0, 0, 0, 1],
    ]
    result = mask_spans(seq_1hot, spans)
    assert np.array_equal(result, expected_output)


def test_mask_spans_from_start_positions():
    seq_1hot = np.array(
        [
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 1, 0, 0],
            [0, 1, 0, 0],  # Input sequence in one-hot encoding format
            [0, 1, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ]
    )
    spans = [0, 2]  # Indices of the spans to be masked
    motif_window = 2  # Size of the motif window
    expected_result = [
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],  # Expected output after masking the spans
        [0, 1, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1],
    ]

    result = mask_spans_from_start_positions(
        seq_1hot, spans, motif_window
    )  # Call the function being tested
    assert np.array_equal(
        result, expected_result
    )  # Check if the result matches the expected output


def test_mask_central_seq():
    seq_1hot = np.array(  # Input sequence in one-hot encoding format
        [
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ]
    )
    motif_width = 2  # Width of the motif
    expected_result = np.array(  # Expected output after masking the central sequence
        [
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ]
    )

    result = mask_central_seq(seq_1hot, motif_width)  # Call the function being tested
    assert np.array_equal(
        result, expected_result
    )  # Check if the result matches the expected output


@pytest.mark.genome
def test_permute_central_seq():
    seq_1hot = dna_1hot("ATGC")  # Input DNA sequence in one-hot encoding format
    motif_width = 2  # Width of the motif
    expected_result = dna_1hot(
        "AGTC"
    )  # Expected output after permuting the central sequence "TG" --> "GT"

    result = permute_central_seq(
        seq_1hot, motif_width
    )  # Call the function being tested
    assert np.array_equal(
        result, expected_result
    )  # Check if the result matches the expected output


@pytest.mark.genome
def test_permute_spans():
    seq_1hot = dna_1hot("ATGC")  # Input DNA sequence in one-hot encoding format
    spans = [(1, 3)]  # List of spans to permute
    expected_result = dna_1hot("AGTC")

    result = permute_spans(seq_1hot, spans)  # Call the function being tested
    assert np.array_equal(
        result, expected_result
    )  # Check if the result matches the expected output

    # Expand the spans into individual indices
    span_indices = np.concatenate([np.arange(start, end) for start, end in spans])

    # Check that the non-span regions remain the same
    non_span_indices = np.setdiff1d(np.arange(seq_1hot.shape[0]), span_indices)
    assert np.array_equal(result[non_span_indices], seq_1hot[non_span_indices])

    # Check that the permuted spans are different from the original spans
    for start, end in spans:
        assert not np.array_equal(result[start:end], seq_1hot[start:end])


@pytest.mark.genome
def test_fetch_centered_padded_seq_and_new_start_position():
    chrom = "chr1"
    start = 5
    end = 15
    seq_length = 10

    result = fetch_centered_padded_seq_and_new_start_position(
        chrom, start, end, seq_length, toy_genome
    )

    assert isinstance(result, tuple), "Result should be a tuple"
    assert len(result) == 2, "Result tuple should contain two elements"

    # Validate the start position
    start_centered = result[0]
    assert isinstance(start_centered, int), "Start position should be an integer"
    assert (
        start_centered == (start + end) // 2 - seq_length // 2
    ), "start_centered should be offset by half the sequence length"

    # Validate the padded sequence
    seq = result[1]
    assert isinstance(seq, str), "Sequence should be a string"
    assert len(seq) == seq_length, "Sequence length should match the specified length"

    # Check if the generated sequence matches the expected sequence from the genome data
    genome_seq = toy_genome.fetch(
        chrom, start_centered, start_centered + seq_length
    ).upper()
    assert seq == genome_seq, "Generated sequence does not match genome data"

    # Additional assertions can be added to cover different scenarios
    # TODO padded seqs


@pytest.mark.genome
def test_permute_spans_from_start_positions():
    # Test case 2 (motif window is less than the shuffle parameter)
    seq_1hot = dna_1hot(toy_genome.fetch("chr1", 0, 60).upper())
    spans_start_positions = [23]
    motif_window = 10
    shuffle_parameter = 20
    with pytest.raises(AssertionError):
        _ = permute_spans_from_start_positions(
            seq_1hot, spans_start_positions, motif_window, shuffle_parameter
        )
    return

    # Test case 3 (motif window is reapeated)
    seq_1hot = dna_1hot("AAGTC")  # Input DNA sequence in one-hot encoding format
    spans_start_positions = [0]
    motif_window = 2
    shuffle_parameter = 1
    with pytest.raises(ValueError):
        _ = permute_spans_from_start_positions(
            seq_1hot, spans_start_positions, motif_window, shuffle_parameter
        )
    return

    # Test case 4 (motif window is non-divisible by the shuffle parameter)
    seq_1hot = dna_1hot(
        "ACGTGACTAGACATA"
    )  # Input DNA sequence in one-hot encoding format
    spans_start_positions = [0, 5]
    motif_window = 4
    shuffle_parameter = 3
    with pytest.raises(ValueError):
        _ = permute_spans_from_start_positions(
            seq_1hot, spans_start_positions, motif_window, shuffle_parameter
        )
    return

    # Test case 4 (positive control experiment)
    seq_1hot = dna_1hot("ATGC")  # Input DNA sequence in one-hot encoding format
    spans_start_positions = [1]
    motif_window = 2
    expected_result = dna_1hot("AGTC")  # TG --> GT

    result = permute_spans_from_start_positions(
        seq_1hot, spans_start_positions, motif_window, shuffle_parameter
    )
    span_indices = np.concatenate(
        [np.arange(s, s + motif_window) for s in spans_start_positions]
    )

    # Check that the non-span regions remain the same
    non_span_indices = np.setdiff1d(np.arange(seq_1hot.shape[0]), span_indices)
    assert np.array_equal(result[non_span_indices], seq_1hot[non_span_indices])

    # Check that the spans have been permuted
    for s in spans_start_positions:
        start, end = s, s + motif_window
        assert not np.array_equal(result[start:end], seq_1hot[start:end])

    # Check that the result is as expected
    assert np.array_equal(result, expected_result)


@pytest.mark.genome
def test_randomise_spans_from_start_positions():
    # Test case 1 (test randomisation of repeated snippets)
    seq_1hot = dna_1hot("AAT")
    spans_start_positions = [0]
    motif_window = 2

    # Check if window is repeated
    with pytest.raises(ValueError):
        _ = randomise_spans_from_start_positions(
            seq_1hot, spans_start_positions, motif_window
        )
    return

    # Test case 2 (test randomisation of non  snippets)
    seq_1hot = dna_1hot("ACT")
    spans_start_positions = [0]
    motif_window = 2
    expected_result = dna_1hot(
        "CAT"
    )  # ie first two nucleotides randomised "AC" --> "CA"

    # Randomize the spans
    result = randomise_spans_from_start_positions(
        seq_1hot, spans_start_positions, motif_window
    )

    # Create an array of indices for the spans
    span_indices = np.concatenate(
        [np.arange(s, s + motif_window) for s in spans_start_positions]
    )

    # Check that the result has the same shape as the input
    assert result.shape == seq_1hot.shape

    # Check that the spans have been randomized
    for s in spans_start_positions:
        start, end = s, s + motif_window
        assert not np.array_equal(result[start:end], seq_1hot[start:end])

    # Check that the non-span regions remain the same
    non_span_indices = np.setdiff1d(np.arange(seq_1hot.shape[0]), span_indices)
    assert np.array_equal(result[non_span_indices], seq_1hot[non_span_indices])

    # Check that the result is as expected
    assert np.array_equal(result, expected_result)


def test_multi_insert_offsets_casette():

    # Test that the function correctly handles a single insertion at the center of the sequence, in the forward orientation.
    seq_1hot = np.zeros((10, 4))
    seq_1hot_insertions = [np.array([[0, 0, 0, 1], [0, 0, 1, 0], [0, 1, 0, 0]])]
    offsets_bp = [1]
    orientation_string = [">"]
    expected_output = np.zeros((10, 4))
    expected_output[6:9, :] = seq_1hot_insertions[0]
    output_seq = _multi_insert_offsets_casette(
        seq_1hot, seq_1hot_insertions, offsets_bp, orientation_string
    )
    assert np.array_equal(output_seq, expected_output)

    # Test that the function correctly handles a single insertion at the center of the sequence, in the reverse orientation.
    seq_1hot = np.zeros((10, 4))
    seq_1hot_insertions = [np.array([[0, 0, 0, 1], [0, 0, 1, 0], [0, 1, 0, 0]])]
    offsets_bp = [1]
    orientation_string = ["<"]
    expected_output = np.zeros((10, 4))
    expected_output[6:9, :] = hot1_rc(seq_1hot_insertions[0])
    output_seq = _multi_insert_offsets_casette(
        seq_1hot, seq_1hot_insertions, offsets_bp, orientation_string
    )
    assert np.array_equal(output_seq, expected_output)

    # Test that the function correctly handles multiple non-overlapping insertions, in different orientations and at different offsets.
    seq_1hot = np.zeros((10, 4))
    seq_1hot_insertions = [
        np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]]),
        np.array([[0, 0, 0, 1], [0, 0, 1, 0], [0, 1, 0, 0]]),
    ]
    offsets_bp = [-2, 2]
    orientation_string = [">", "<"]
    expected_output = np.zeros((10, 4))
    expected_output[3:6, :] = seq_1hot_insertions[0]
    expected_output[7:10, :] = hot1_rc(seq_1hot_insertions[1])
    output_seq = _multi_insert_offsets_casette(
        seq_1hot, seq_1hot_insertions, offsets_bp, orientation_string
    )
    assert np.array_equal(output_seq, expected_output)

    # Test that the function returns the expected output when all possible arguments are empty.
    seq_1hot = np.zeros((10, 4))
    seq_1hot_insertions = []
    offsets_bp = []
    orientation_string = ""
    expected_output = np.zeros((10, 4))
    output_seq = _multi_insert_offsets_casette(
        seq_1hot, seq_1hot_insertions, offsets_bp, orientation_string
    )
    assert np.array_equal(output_seq, expected_output)
