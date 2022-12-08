from akita_utils.dna_utils import dna_1hot, dna_1hot_to_seq, dna_seq_rc
from akita_utils.seq_gens import _insert_casette


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
    # spacer_bp is 0

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
