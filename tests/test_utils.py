import pandas as pd
import numpy as np
from akita_utils import filter_by_chrmlen, ut_dense, split_df_equally, symmertic_insertion_seqs_gen
from io import StringIO
import pysam

def test_ut_dense():

    # toy output representing upper-triangular output for two targets
    ut_vecs = np.vstack(([4, 5, 6], [-2, -3, -4])).T

    # (3 entries) x (2 targets) input, two empty diagonals --> 4x4x2 output
    assert np.shape(ut_dense(ut_vecs, 2)) == (4, 4, 2)

    # (3 entries) x (2 targets) input, one empty diagonals --> 3x3x2 output
    dense_mats = ut_dense(ut_vecs, 1)
    assert np.shape(dense_mats) == (3, 3, 2)

    # outputs are symmetric dense matrices with the 3 original entries
    # and zeros at the diagonal
    target_0 = np.array([[0, 4, 5], [4, 0, 6], [5, 6, 0]])
    target_1 = np.array([[0, -2, -3], [-2, 0, -4], [-3, -4, 0]])

    assert (dense_mats[:, :, 0] == target_0).all()
    assert (dense_mats[:, :, 1] == target_1).all()


def test_split_df_equally():
    
    df = pd.DataFrame(np.linspace(0, 99, 100), columns=['col1'])
    fifth_chunk = split_df_equally(df, 20, 5)
    assert (fifth_chunk["col1"].to_numpy() == np.linspace(25, 29, 5)).all() == True

    
def test_filter_by_chrmlen():

    df1 = pd.DataFrame(
        [["chrX", 3, 8], ["chr1", 4, 5], ["chrX", 1, 5]],
        columns=["chrom", "start", "end"],
    )

    # get the same result with chrmsizes provided as dict or via StringIO

    # one interval is dropped for exceeding chrX len of 7
    assert filter_by_chrmlen(df1, {"chr1": 10, "chrX": 7}, 0).shape == (2, 3)

    # both chrX intervals are dropped if the buffer_bp are increased
    assert filter_by_chrmlen(df1, {"chr1": 10, "chrX": 7}, 3).shape == (1, 3)

    # no intervals remain if all of chr1 is excluded as well
    assert filter_by_chrmlen(df1, {"chr1": 10, "chrX": 7}, 5).shape == (0, 3)

    
def test_symmertic_insertion_seqs_gen():
    
    correct_insertion_starting_positions = {"subtest-0" : [655206, 655455],
                                            "subtest-1" : [653272, 655331, 657390],
                                            "subtest-2" : [155301, 1155360]}
    
    test_tsv_path = "/home1/smaruj/akita_utils/tests/test_symmertic_insertion_seqs_gen_df.tsv"
    seq_coords_df = pd.read_csv(test_tsv_path, sep="\t")
    
    genome_path = "/project/fudenber_735/genomes/mm10/mm10.fa"
    genome_open = pysam.Fastafile(genome_path)
    
    background_file_path = "/project/fudenber_735/tensorflow_models/akita/v2/analysis/background_seqs.fa"
    background_seqs = []
    with open(background_file_path, "r") as f:
        for line in f.readlines():
            if ">" in line:
                continue
            background_seqs.append(dna_io.dna_1hot(line.strip()))
    num_insert_backgrounds = seq_coords_df["background_index"].max()
    if len(background_seqs) < num_insert_backgrounds:
        raise ValueError(
            "must provide a background file with at least as many"
            + "backgrounds as those specified in the insert seq_coords tsv."
            + "\nThe provided background file has {len(background_seqs)} sequences."
        )
    
    outputted_insertion_starting_positions = {}
    
    i = 0
    for value in symmertic_insertion_seqs_gen(seq_coords_df, background_seqs, genome_open, test=True):
        outputted_insertion_starting_positions[f"subtest-{i}"] = value
        i += 1
    
    for key in outputted_insertion_starting_positions:
        assert outputted_insertion_starting_positions[key] == correct_insertion_starting_positions[key]
    
    genome_open.close()


