from akita_utils.dna_utils import dna_1hot, dna_1hot_to_seq, dna_seq_rc
from akita_utils.seq_gens import _insert_casette, _multi_insert_casette 

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
    


def test_multi_insert_casette():
    # Test insertion of a single sequence
    seq_1hot = dna_1hot("ACGT")
    seq_1hot_insertions =[dna_1hot("GT")]
    spacer_bp = 1
    orientation_string = ">"
    expected_output = "AGTT" 
    assert dna_1hot_to_seq(_multi_insert_casette(seq_1hot, seq_1hot_insertions, spacer_bp, orientation_string)) == expected_output
    
    # Test insertion of multiple sequences with reverse-complemented sequences
    seq_1hot = dna_1hot("ACATAGT")
    seq_1hot_insertions = [dna_1hot("GT"),dna_1hot("A")] 
    spacer_bp = 1
    orientation_string = "<>"
    expected_output = "AACTAGT" 
    assert dna_1hot_to_seq(_multi_insert_casette(seq_1hot, seq_1hot_insertions, spacer_bp, orientation_string)) == expected_output
    
    # Test insertion with longer spacers with reverse-complemented sequences
    seq_1hot = dna_1hot("ACGTACGTACGTTACGT")
    seq_1hot_insertions = [dna_1hot("GT"),dna_1hot("A"),dna_1hot("GT")] 
    spacer_bp = 3
    orientation_string = "<<>"
    expected_output = "ACGACCGTTCGTGTCGT" 
    assert dna_1hot_to_seq(_multi_insert_casette(seq_1hot, seq_1hot_insertions, spacer_bp, orientation_string)) == expected_output
        
    # Test insertion with mismatched number of insertions and orientations
    seq_1hot = dna_1hot("ACGTACGTACGT") 
    seq_1hot_insertions = [dna_1hot("CTG"), dna_1hot,("ATG")] 
    spacer_bp = 1
    orientation_string = "<"
    try:
        _multi_insert_casette(seq_1hot, seq_1hot_insertions, spacer_bp, orientation_string)
    except AssertionError:
        pass
    else:
        assert False, "Expected AssertionError for mismatched number of insertions and orientations"

