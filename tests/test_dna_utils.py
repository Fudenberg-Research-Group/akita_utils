from dna_utils import dna_1hot, reverse_dna_1hot


def test_reverse_dna_1hot():
    assert reverse_dna_1hot(dna_1hot("AAAA")) == "AAAA"
