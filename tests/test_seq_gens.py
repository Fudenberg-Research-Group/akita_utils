from dna_utils import dna_1hot, reverse_dna_1hot
from seq_gens import _insert_casette


def test_insert_casette():

    toy_genome_open = dna_1hot("TTGTACTTCGTCGTT")
    seq_1hot = dna_1hot("AAAAAAAAAA")

    # first test
    first_spacer_bp = 1
    first_seq_1hot_insertion = toy_genome_open[
        (2 - first_spacer_bp) : (6 + first_spacer_bp)
    ]
    first_orientation_string = ">"

    assert (
        reverse_dna_1hot(
            _insert_casette(
                seq_1hot,
                first_seq_1hot_insertion,
                first_spacer_bp,
                first_orientation_string,
            )
        )
        == "AATGTACTAA"
    )

    # second test
    second_spacer_bp = 0
    second_seq_1hot_insertion = toy_genome_open[
        (8 - second_spacer_bp) : (13 + second_spacer_bp)
    ]
    second_orientation_string = "<"

    assert (
        reverse_dna_1hot(
            _insert_casette(
                seq_1hot,
                second_seq_1hot_insertion,
                second_spacer_bp,
                second_orientation_string,
            )
        )
        == "AAACGACGAA"
    )
