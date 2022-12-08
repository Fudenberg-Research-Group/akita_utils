from akita_utils.dna_utils import dna_1hot, dna_1hot_to_seq, dna_seq_rc


def test_dna_1hot_to_seq():
    assert dna_1hot_to_seq(dna_1hot("AAAA")) == "AAAA"


def test_dna_seq_rc():
    seq = "ACTG"
    rc_seq = "CAGT"
    assert dna_seq_rc(seq) == rc_seq
