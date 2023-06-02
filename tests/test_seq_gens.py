from akita_utils.dna_utils import dna_1hot, dna_1hot_to_seq, dna_seq_rc
from akita_utils.seq_gens import _insert_casette
from akita_utils.seq_gens import *
import pytest

class ToyGenomeOpen:
    def __init__(self, genome_data):
        self.genome_data = genome_data

    def fetch(self, chrom, start, end):
        chromosome = self.genome_data.get(chrom)
        if chromosome:
            return chromosome[start:end]
        else:
            raise ValueError(f"Chromosome {chrom} not found")
    def get_reference_length(self, chrom):
        chromosome = self.genome_data.get(chrom)
        if chromosome:
            return len(chromosome)
        else:
            raise ValueError(f"Chromosome {chrom} not found")

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
    

def test_mask_spans():
    seq_1hot = [[0, 1, 0, 0], 
                [1, 0, 0, 0], 
                [0, 0, 1, 0], 
                [1, 0, 0, 0], 
                [0, 0, 1, 0], 
                [0, 0, 0, 1]]
    spans = [[1, 3]]
    expected_output = [ [0, 1, 0, 0], 
                        [0, 0, 0, 0], 
                        [0, 0, 0, 0], 
                        [1, 0, 0, 0], 
                        [0, 0, 1, 0], 
                        [0, 0, 0, 1]]
    result = mask_spans(seq_1hot, spans)
    assert(result == expected_output)


@pytest.mark.parametrize(
    "seq_1hot, spans, motif_window, expected_result",
    [
        (
            np.array([
                [1, 0, 0, 0], 
                [0, 1, 0, 0],
                [0, 1, 0, 0],
                [0, 1, 0, 0],
                [0, 1, 0, 0],
                [0, 1, 0, 0],
                [0, 0, 1, 0], 
                [0, 0, 0, 1]
            ]),
            [0, 2],
            2,
            np.array([
                [0, 0, 0, 0], 
                [0, 0, 0, 0], 
                [0, 0, 0, 0], 
                [0, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 1, 0, 0],
                [0, 0, 1, 0], 
                [0, 0, 0, 1]
            ]),
        ),
        (
            np.array([
                [1, 0, 0, 0], 
                [0, 1, 0, 0], 
                [0, 0, 1, 0], 
                [0, 0, 0, 1], 
                [0, 0, 0, 1], 
                [0, 0, 0, 1], 
                [0, 0, 0, 1]
            ]),
            [1, 3],
            2,
            np.array([
                [1, 0, 0, 0], 
                [0, 0, 0, 0], 
                [0, 0, 0, 0], 
                [0, 0, 0, 0], 
                [0, 0, 0, 0], 
                [0, 0, 0, 1], 
                [0, 0, 0, 1]
            ]),
        ),
    ],
)
def test_mask_spans_from_start_positions(seq_1hot, spans, motif_window, expected_result):
    result = mask_spans_from_start_positions(seq_1hot, spans, motif_window)
    assert np.array_equal(result, expected_result)

    
@pytest.mark.parametrize(
    "seq_1hot, motif_width, expected_result",
    [
        (
            np.array([
                [1, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1],
                [1, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1]
            ]),
            2,
            np.array([
                [1, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 0],
                [0, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1]
            ]),
        ),
        (
            np.array([
                [1, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1],
                [1, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1]
            ]),
            4,
            np.array([
                [1, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 0, 0, 0],
                [0, 0, 0, 0],
                [0, 0, 0, 0],
                [0, 0, 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1]
            ]),
        ),
    ],
)
def test_mask_central_seq(seq_1hot, motif_width, expected_result):
    result = mask_central_seq(seq_1hot, motif_width)
    assert np.array_equal(result, expected_result)


@pytest.mark.genome
@pytest.mark.parametrize(
    "seq_1hot, motif_width, control",
    [
        (dna_1hot(toy_genome.fetch("chr1", 0, 60).upper()),16,False),
        (dna_1hot(toy_genome.fetch("chr2", 0, 60).upper()),10,False),
        (dna_1hot("ATGC"),2,True),
    ],
)
def test_permute_central_seq(seq_1hot, motif_width, control):
    if control:
        result = permute_central_seq(seq_1hot, motif_width)
        assert np.array_equal(result, dna_1hot("AGTC"))
    else:
        result = permute_central_seq(seq_1hot, motif_width)
        seq_length = len(seq_1hot)
        assert not np.array_equal(result[seq_length // 2 - motif_width // 2 : seq_length // 2 + motif_width // 2, :], seq_1hot[seq_length // 2 - motif_width // 2 : seq_length // 2 + motif_width // 2, :])


@pytest.mark.genome
@pytest.mark.parametrize(
    "seq_1hot, spans, control",
    [
        (dna_1hot(toy_genome.fetch("chr2", 0, 60).upper()),[(0, 20),(23, 51)],False),
        (dna_1hot(toy_genome.fetch("chr2", 0, 60).upper()),[(0, 10)],False),
        (dna_1hot("ATGC"),[(1,3)],True),
    ],
)
def test_permute_spans(seq_1hot, spans, control):
    if control:
        result = permute_spans(seq_1hot, spans)
        assert np.array_equal(result, dna_1hot("AGTC"))
    else:    
        result = permute_spans(seq_1hot, spans)

        # Expand the spans into individual indices
        span_indices = np.concatenate([np.arange(start, end) for start, end in spans])

        # Check that the non-span regions remain the same
        non_span_indices = np.setdiff1d(np.arange(seq_1hot.shape[0]), span_indices)    
        assert np.array_equal(result[non_span_indices], seq_1hot[non_span_indices])

        # Check that the permuted spans are different from the original spans
        # (if the seq_1hot and spans are very short this result might be same as original seq thus failing the following test)
        for start, end in spans:
            assert not np.array_equal(result[start:end], seq_1hot[start:end])
        
        
@pytest.mark.genome
def test_fetch_centered_padded_seq_and_new_start_position():

    chrom = "chr1"
    start = 5
    end = 15
    seq_length = 10

    result = fetch_centered_padded_seq_and_new_start_position(chrom, start, end, seq_length, toy_genome)

    assert isinstance(result, tuple), "Result should be a tuple"
    assert len(result) == 2, "Result tuple should contain two elements"

    # Validate the start position
    start_centered = result[0]
    assert isinstance(start_centered, int), "Start position should be an integer"
    assert start_centered == (start + end) // 2 - seq_length // 2, "Incorrect start position calculation"

    # Validate the padded sequence
    seq = result[1]
    assert isinstance(seq, str), "Sequence should be a string"
    assert len(seq) == seq_length, "Sequence length should match the specified length"
    
    # Check if the generated sequence matches the expected sequence from the genome data
    genome_seq = toy_genome.fetch(chrom, start_centered, start_centered + seq_length).upper()
    assert seq == genome_seq, "Generated sequence does not match genome data"


    # Additional assertions can be added to cover different scenarios
    # TODO padded seqs
    
    
@pytest.mark.genome_1
@pytest.mark.parametrize(
    "seq_1hot, spans_start_positions, motif_window, shuffle_parameter, is_non_divisible, is_cyclical, is_less_than_shiffle_parameter",
    [
        (dna_1hot(toy_genome.fetch("chr1", 0, 60).upper()), [10], 15, 3, False, False, False),
        (dna_1hot(toy_genome.fetch("chr1", 0, 60).upper()), [23], 10, 20, False, False, True), # window is_less_than_shiffle_parameter
        (dna_1hot("AAGTC"), [0], 2, 1, False, True, False), # window is_cyclical
        (dna_1hot("ACGTGACTAGACATA"), [0, 5], 4, 3, True, False, False), # window is_non_divisible
    ],
)
def test_permute_spans_from_start_positions(seq_1hot, spans_start_positions, motif_window, shuffle_parameter, is_non_divisible, is_cyclical, is_less_than_shiffle_parameter):
    
    # Check control experiment one, window is_non_divisible shuffle parameter
    if is_non_divisible:
        with pytest.raises(ValueError):
            _ = akita_utils.seq_gens.permute_spans_from_start_positions(seq_1hot, spans_start_positions, motif_window, shuffle_parameter)
        return
    
    # Check control experiment two, window is cyclical with shuffling 
    if is_cyclical:
        with pytest.raises(ValueError):
            _ = akita_utils.seq_gens.permute_spans_from_start_positions(seq_1hot, spans_start_positions, motif_window, shuffle_parameter)
        return
    
    # Check control experiment three, window is_less_than_shiffle_parameter 
    if is_less_than_shiffle_parameter:
        with pytest.raises(AssertionError):
            _ = akita_utils.seq_gens.permute_spans_from_start_positions(seq_1hot, spans_start_positions, motif_window, shuffle_parameter)
        return    
        
    result = akita_utils.seq_gens.permute_spans_from_start_positions(seq_1hot, spans_start_positions, motif_window, shuffle_parameter)
    span_indicies = np.concatenate([np.arange(s, s+motif_window) for s in spans_start_positions])
    
    # Check that the non-span regions remain the same
    non_span_indices = np.setdiff1d(np.arange(seq_1hot.shape[0]), span_indicies)
    assert np.array_equal(result[non_span_indices], seq_1hot[non_span_indices])
    
    # Check that the spans have been permuted  
    for s in spans_start_positions:
        start, end = s, s + motif_window
        assert not np.array_equal(result[start:end], seq_1hot[start:end])
        

@pytest.mark.genome_1
@pytest.mark.parametrize(
    "seq_1hot, spans_start_positions, motif_window, is_control, is_cyclical",
    [
        (dna_1hot(toy_genome.fetch("chr1", 0, 60).upper()), [0, 20, 39], 16, False, False),
        (dna_1hot(toy_genome.fetch("chr2", 0, 60).upper()), [0, 20, 41], 13, False, False),
        (dna_1hot(toy_genome.fetch("chr2", 0, 60).upper()), [0], 20, False, False),
        (dna_1hot("ACT"), [0], 2, True, False), # control experiment
        (dna_1hot("AAT"), [0], 2, False, True), # cyclical experiment
    ],
)
def test_randomise_spans_from_start_positions(seq_1hot, spans_start_positions, motif_window, is_control, is_cyclical):
    
    # Check control experiment two, window is cyclical with shuffling 
    if is_cyclical: 
        with pytest.raises(ValueError):
            _ = akita_utils.seq_gens.randomise_spans_from_start_positions(seq_1hot, spans_start_positions, motif_window)
        return
    
    result = randomise_spans_from_start_positions(seq_1hot, spans_start_positions, motif_window)    
    span_indicies = np.concatenate([np.arange(s, s+motif_window) for s in spans_start_positions])
    
    # Check that the result has the same shape as the input
    assert result.shape == seq_1hot.shape

    # Check that the spans have been randomized for non cyclicalI'm 
    for s in spans_start_positions:
        start, end = s, s + motif_window
        assert not np.array_equal(result[start:end], seq_1hot[start:end])

    # Check that the non-span regions remain the same
    non_span_indices = np.setdiff1d(np.arange(seq_1hot.shape[0]), span_indicies)
    assert np.array_equal(result[non_span_indices], seq_1hot[non_span_indices])
    
    if is_control:
        assert np.array_equal(result, dna_1hot("CAT"))
