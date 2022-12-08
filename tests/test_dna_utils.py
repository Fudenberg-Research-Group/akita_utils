from akita_utils.dna_utils import dna_1hot, dna_1hot_to_seq


def test_reverse_dna_1hot():
    assert dna_1hot_to_seq(dna_1hot("AAAA")) == "AAAA"
