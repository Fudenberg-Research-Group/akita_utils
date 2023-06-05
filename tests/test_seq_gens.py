import pytest
import numpy as np
from .Toy_genome import ToyGenomeOpen
from akita_utils.dna_utils import dna_1hot, dna_1hot_to_seq, dna_seq_rc
from akita_utils.seq_gens import (
    _insert_casette,
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
    seq_1hot = [
        [0, 1, 0, 0],
        [1, 0, 0, 0],
        [0, 0, 1, 0],
        [1, 0, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1],
    ]
    spans = [[1, 3]]
    expected_output = [
        [0, 1, 0, 0],
        [0, 0, 0, 0],  # this column should be masked
        [0, 0, 0, 0],  # this column should be masked
        [1, 0, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1],
    ]
    result = mask_spans(seq_1hot, spans)
    assert result == expected_output


@pytest.mark.parametrize(
    "seq_1hot, spans, motif_window, expected_result",  # Order of the parameters being passed
    [
        (  # Test case 1 (span start from start of seq)
            np.array(  # Input sequence in one-hot encoding format
                [
                    [1, 0, 0, 0],
                    [0, 1, 0, 0],
                    [0, 1, 0, 0],
                    [0, 1, 0, 0],
                    [0, 1, 0, 0],
                    [0, 1, 0, 0],
                    [0, 0, 1, 0],
                    [0, 0, 0, 1],
                ]
            ),
            [0, 2],  # Indices of the spans to be masked
            2,  # Size of the motif window
            np.array(  # Expected output after masking the spans
                [
                    [0, 0, 0, 0],
                    [0, 0, 0, 0],
                    [0, 0, 0, 0],
                    [0, 0, 0, 0],
                    [0, 1, 0, 0],
                    [0, 1, 0, 0],
                    [0, 0, 1, 0],
                    [0, 0, 0, 1],
                ]
            ),
        ),
        (  # Test case 2 (span in middle of seq)
            np.array(  # Input sequence in one-hot encoding format
                [
                    [1, 0, 0, 0],
                    [0, 1, 0, 0],
                    [0, 0, 1, 0],
                    [0, 0, 0, 1],
                    [0, 0, 0, 1],
                    [0, 0, 0, 1],
                    [0, 0, 0, 1],
                ]
            ),
            [1, 3],  # Indices of the spans to be masked
            2,  # Size of the motif window
            np.array(  # Expected output after masking the spans
                [
                    [1, 0, 0, 0],
                    [0, 0, 0, 0],
                    [0, 0, 0, 0],
                    [0, 0, 0, 0],
                    [0, 0, 0, 0],
                    [0, 0, 0, 1],
                    [0, 0, 0, 1],
                ]
            ),
        ),
    ],
)
def test_mask_spans_from_start_positions(
    seq_1hot, spans, motif_window, expected_result
):
    result = mask_spans_from_start_positions(
        seq_1hot, spans, motif_window
    )  # Call the function being tested
    assert np.array_equal(
        result, expected_result
    )  # Check if the result matches the expected output


@pytest.mark.parametrize(
    "seq_1hot, motif_width, expected_result",  # Order of the parameters being passed
    [
        (  # Test case 1
            np.array(  # Input sequence in one-hot encoding format
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
            ),
            2,  # Width of the motif
            np.array(  # Expected output after masking the central sequence
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
            ),
        )
    ],
)
def test_mask_central_seq(seq_1hot, motif_width, expected_result):
    result = mask_central_seq(seq_1hot, motif_width)  # Call the function being tested
    assert np.array_equal(
        result, expected_result
    )  # Check if the result matches the expected output


@pytest.mark.genome
@pytest.mark.parametrize(
    "seq_1hot, motif_width, expected_result",  # Order of the parameters being passed
    [
        (  # Test case 1
            dna_1hot("ATGC"),  # Input DNA sequence in one-hot encoding format
            2,  # Width of the motif
            dna_1hot(
                "AGTC"
            ),  # Expected output after permuting the central sequence "TG" --> "GT"
        ),
    ],
)
def test_permute_central_seq(seq_1hot, motif_width, expected_result):
    result = permute_central_seq(
        seq_1hot, motif_width
    )  # Call the function being tested
    assert np.array_equal(
        result, expected_result
    )  # Check if the result matches the expected output


@pytest.mark.genome
@pytest.mark.parametrize(
    "seq_1hot, spans, control",  # Order of the parameters being passed
    [
        (  # Test case 1
            dna_1hot(
                toy_genome.fetch("chr2", 0, 60).upper()
            ),  # Input DNA sequence in one-hot encoding format
            [(0, 20), (23, 51)],  # List of spans to permute
            False,  # Control flag indicating whether to check for specific conditions
        ),
        (  # Test case 2
            dna_1hot(
                toy_genome.fetch("chr2", 0, 60).upper()
            ),  # Input DNA sequence in one-hot encoding format
            [(0, 10)],  # List of spans to permute
            False,  # Control flag indicating whether to check for specific conditions
        ),
        (  # Test case 3
            dna_1hot("ATGC"),  # Input DNA sequence in one-hot encoding format
            [(1, 3)],  # List of spans to permute
            True,  # Control flag indicating whether to check for specific conditions
        ),
    ],
)
def test_permute_spans(seq_1hot, spans, control):
    if control:
        result = permute_spans(seq_1hot, spans)  # Call the function being tested
        assert np.array_equal(
            result, dna_1hot("AGTC")
        )  # Check if the result matches the expected output (specific to test case 3)
    else:
        result = permute_spans(seq_1hot, spans)  # Call the function being tested

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
    ), "Incorrect start position calculation"

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
@pytest.mark.parametrize(
    "seq_1hot, spans_start_positions, motif_window, shuffle_parameter, is_non_divisible, is_cyclical, is_less_than_shuffle_parameter",
    [
        (  # Test case 1 (nothing special)
            dna_1hot(
                toy_genome.fetch("chr1", 0, 60).upper()
            ),  # Input DNA sequence in one-hot encoding format
            [10],  # List of start positions of spans
            15,  # Width of the motif window
            3,  # Shuffle parameter
            False,  # Flag indicating whether the motif window is non-divisible by the shuffle parameter
            False,  # Flag indicating whether the motif window is cyclical with shuffling
            False,  # Flag indicating whether the motif window is less than the shuffle parameter
        ),
        (  # Test case 2 (motif window is less than the shuffle parameter)
            dna_1hot(
                toy_genome.fetch("chr1", 0, 60).upper()
            ),  # Input DNA sequence in one-hot encoding format
            [23],  # List of start positions of spans
            10,  # Width of the motif window
            20,  # Shuffle parameter
            False,  # Flag indicating whether the motif window is non-divisible by the shuffle parameter
            False,  # Flag indicating whether the motif window is cyclical with shuffling
            True,  # Flag indicating whether the motif window is less than the shuffle parameter
        ),
        (  # Test case 3 (motif window is cyclical with shuffling)
            dna_1hot("AAGTC"),  # Input DNA sequence in one-hot encoding format
            [0],  # List of start positions of spans
            2,  # Width of the motif window
            1,  # Shuffle parameter
            False,  # Flag indicating whether the motif window is non-divisible by the shuffle parameter
            True,  # Flag indicating whether the motif window is cyclical with shuffling
            False,  # Flag indicating whether the motif window is less than the shuffle parameter
        ),
        (  # Test case 4 (motif window is non-divisible by the shuffle parameter)
            dna_1hot(
                "ACGTGACTAGACATA"
            ),  # Input DNA sequence in one-hot encoding format
            [0, 5],  # List of start positions of spans
            4,  # Width of the motif window
            3,  # Shuffle parameter
            True,  # Flag indicating whether the motif window is non-divisible by the shuffle parameter
            False,  # Flag indicating whether the motif window is cyclical with shuffling
            False,  # Flag indicating whether the motif window is less than the shuffle parameter
        ),
    ],
)
def test_permute_spans_from_start_positions(
    seq_1hot,
    spans_start_positions,
    motif_window,
    shuffle_parameter,
    is_non_divisible,
    is_cyclical,
    is_less_than_shuffle_parameter,
):
    # Check control experiment one, window is non-divisible by shuffle parameter
    if is_non_divisible:
        with pytest.raises(ValueError):
            _ = permute_spans_from_start_positions(
                seq_1hot, spans_start_positions, motif_window, shuffle_parameter
            )
        return

    # Check control experiment two, window is cyclical with shuffling
    if is_cyclical:
        with pytest.raises(ValueError):
            _ = permute_spans_from_start_positions(
                seq_1hot, spans_start_positions, motif_window, shuffle_parameter
            )
        return

    # Check control experiment three, window is less than the shuffle parameter
    if is_less_than_shuffle_parameter:
        with pytest.raises(AssertionError):
            _ = permute_spans_from_start_positions(
                seq_1hot, spans_start_positions, motif_window, shuffle_parameter
            )
        return

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


@pytest.mark.genome
@pytest.mark.parametrize(
    "seq_1hot, spans_start_positions, motif_window, is_control, is_cyclical",
    [
        (   # Test case 1 (nothing special)
            dna_1hot(
                toy_genome.fetch("chr1", 0, 60).upper()
            ),  # Input sequence in one-hot encoding
            [0, 20, 39],  # Starting positions of the spans
            16,  # Width of the motif window
            False,  # Flag indicating if it's a control experiment
            False,  # Flag indicating if the window is cyclical
        ),
        (   # Test case 2 (test randomisation of non cyclical snippets)
            dna_1hot("ACT"),  # Input sequence in one-hot encoding
            [0],  # Starting positions of the spans
            2,  # Width of the motif window
            True,  # Flag indicating if it's a control experiment
            False,  # Flag indicating if the window is cyclical
        ),
        (   # Test case 3 (test randomisation of cyclical snippets)
            dna_1hot("AAT"),  # Input sequence in one-hot encoding
            [0],  # Starting positions of the spans
            2,  # Width of the motif window
            False,  # Flag indicating if it's a control experiment
            True,  # Flag indicating if the window is cyclical
        ),
    ],
)
def test_randomise_spans_from_start_positions(
    seq_1hot, spans_start_positions, motif_window, is_control, is_cyclical
):
    # Check control experiment two, window is cyclical with shuffling
    if is_cyclical:
        with pytest.raises(ValueError):
            _ = randomise_spans_from_start_positions(
                seq_1hot, spans_start_positions, motif_window
            )
        return

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

    # Check that the spans have been randomized for non-cyclical case
    for s in spans_start_positions:
        start, end = s, s + motif_window
        assert not np.array_equal(result[start:end], seq_1hot[start:end])

    # Check that the non-span regions remain the same
    non_span_indices = np.setdiff1d(np.arange(seq_1hot.shape[0]), span_indices)
    assert np.array_equal(result[non_span_indices], seq_1hot[non_span_indices])

    # Check that the result of control experiment(second last) is as expected ie first two nucleotides randomised "AC" --> "CA"
    if is_control:
        assert np.array_equal(result, dna_1hot("CAT"))
