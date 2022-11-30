### akita utilities

import bioframe
import pandas as pd
import numpy as np
import tensorflow as tf
from basenji import dna_io
from io import StringIO
import pysam
import time 
from scipy.stats import spearmanr, pearsonr
import scipy.signal


### numeric utilites

def absmaxND(a, axis=None):
    amax = a.max(axis)
    amin = a.min(axis)
    return np.where(-amin > amax, amin, amax)


def smooth(y, box_pts):
    box = np.ones(box_pts) / box_pts
    y_smooth = np.convolve(y, box, mode="same")
    return y_smooth


def set_diag(arr, x, i=0, copy=False):
    if copy:
        arr = arr.copy()
    start = max(i, -arr.shape[1] * i)
    stop = max(0, (arr.shape[1] - i)) * arr.shape[1]
    step = arr.shape[1] + 1
    arr.flat[start:stop:step] = x
    return arr


### model i/o
def from_upper_triu(vector_repr, matrix_len, num_diags):
    z = np.zeros((matrix_len, matrix_len))
    triu_tup = np.triu_indices(matrix_len, num_diags)
    z[triu_tup] = vector_repr
    for i in range(-num_diags + 1, num_diags):
        set_diag(z, np.nan, i)
    return z + z.T


### score i/o
import h5py


def h5_to_df(filename):
    scd_out = h5py.File(filename, "r")
    s = []
    scd_stats = ["SCD", "SSD", "INS-16", "INS-32", "INS-64", "INS-128", "INS-256"]
    for key in scd_out.keys():
        if key.replace("ref_", "").replace("alt_", "") in scd_stats:
            s.append(pd.Series(scd_out[key][()].mean(axis=1), name=key))
        else:
            s.append(pd.Series(scd_out[key][()], name=key))
            # print(len(scd_out[key][()]))

    ins_stats = ["INS-16", "INS-32", "INS-64", "INS-128", "INS-256"]
    for key in ins_stats:
        if "ref_" + key in scd_out.keys():
            diff = scd_out["ref_" + key][()].mean(axis=1) - scd_out["alt_" + key][
                ()
            ].mean(axis=1)
            s.append(pd.Series(diff, name=key))
    seq_coords_df = pd.concat(s, axis=1)
    for key in ["chrom", "strand_2"]:  #'rownames','strand','chrom','TF']:
        seq_coords_df[key] = seq_coords_df[key].str.decode("utf8").copy()
    scd_out.close()

    len_orig = len(seq_coords_df)
    seq_coords_df.drop_duplicates("index", inplace=True)
    print(len_orig - len(seq_coords_df), "duplicates removed for ", filename)
    seq_coords_df.rename(columns={"index": "mut_index"}, inplace=True)
    seq_coords_df.reset_index(inplace=True, drop=True)
    return seq_coords_df


import glob
from io import StringIO


def filter_boundary_h5(
    h5_dirs="/project/fudenber_735/tensorflow_models/akita/v2/analysis/permute_boundaries_motifs_ctcf_mm10_model*/scd.h5",
    score_key="SCD",
):
    ## load scores from boundary mutagenesis, average chosen score across models
    dfs = []
    for h5_file in glob.glob(h5_dirs):
        dfs.append(h5_to_df(h5_file))
    score_key = "SCD"
    df = dfs[0].copy()
    df[score_key] = np.mean([df[score_key] for df in dfs], axis=0)
    df["span"] = df["span"].str.decode("utf8")

    ## append scores for full mut and all ctcf mut to table
    print("annotating each site with boundary-wide scores")
    score_10k = np.zeros((len(df),))
    score_all_ctcf = np.zeros((len(df),))
    for i in np.unique(df["boundary_index"].values):
        inds = df["boundary_index"].values == i
        df_boundary = df.iloc[inds]
        score_10k[inds] = df_boundary.iloc[-1]["SCD"]
        if len(df_boundary) > 2:
            score_all_ctcf[inds] = df_boundary.iloc[-2]["SCD"]
    df["score_all_ctcf"] = score_all_ctcf
    df["score_10k"] = score_10k

    # considering only single ctcf mutations
    # require that they fall in an overall boundary that has some saliency
    # TODO: maybe also require that the neighboring bins don't have a more salient boundary?
    # suffix _2 means _motif
    sites = df.iloc[
        (df["strand_2"].values != "nan") * (df["score_all_ctcf"].values > 5)
    ].copy()

    # extracting start/end of motif from span
    sites = pd.concat(
        [
            sites,
            sites["span"]
            .str.split("-", expand=True)
            .astype(int)
            .rename(columns={0: "start_2", 1: "end_2"})
            .copy(),
        ],
        axis=1,
    )
    sites.reset_index(inplace=True, drop=True)

    print("filtering sites by overlap with rmsk")
    # require that sites don't overlap rmsk !
    # this is important for sineB2 in mice, maybe other things as well
    rmsk_cols = pd.read_csv(
        StringIO(
            "bin	swScore	milliDiv	milliDel	milliIns	genoName	genoStart	genoEnd	genoLeft	strand	repName	repClass	repFamily	repStart	repEnd	repLeft	id"
        ),
        sep="\t",
    )
    rmsk = pd.read_table(
        "/project/fudenber_735/genomes/mm10/database/rmsk.txt.gz",
        names=rmsk_cols.keys(),
    )
    rmsk.rename(
        columns={"genoName": "chrom", "genoStart": "start", "genoEnd": "end"},
        inplace=True,
    )

    sites = bioframe.count_overlaps(
        sites, rmsk[["chrom", "start", "end"]], cols1=["chrom", "start_2", "end_2"]
    )
    sites = sites.iloc[sites["count"].values == 0]
    sites.reset_index(inplace=True, drop=True)
    if sites.duplicated().sum() > 0:
        raise ValueError("no duplicates allowed")

    return sites


def filter_sites_by_score(
    sites,
    score_key="SCD",
    weak_thresh_pct=1,  # don't use sites weaker than this, might be artifacts
    weak_num=500,
    strong_thresh_pct=99,  # don't use sites weaker than this, might be artifacts
    strong_num=500,
):
    """chooses a specified number of strong and weak sites exluding low and/or high outliers which may contain more artifacts."""
    if (weak_num < 1) or (strong_num < 1):
        raise ValueError("must select a postive number of sites")
    strong_thresh = np.percentile(sites[score_key].values, strong_thresh_pct)
    weak_thresh = np.percentile(sites[score_key].values, weak_thresh_pct)
    weak_sites = (
        sites.loc[sites[score_key] > weak_thresh]
        .copy()
        .sort_values(score_key)[:weak_num]
    )
    strong_sites = (
        sites.loc[sites[score_key] < strong_thresh]
        .copy()
        .sort_values(score_key)[-strong_num:][::-1]
    )
    strong_sites.reset_index(inplace=True, drop=True)
    weak_sites.reset_index(inplace=True, drop=True)
    return strong_sites, weak_sites


# def prepare_insertion_tsv(
#     h5_dirs="/project/fudenber_735/tensorflow_models/akita/v2/analysis/permute_boundaries_motifs_ctcf_mm10_model*/scd.h5",
#     score_key="SCD",
#     flank_pad=60,  # how much flanking sequence around the sites to include
#     weak_thresh_pct=1,  # don't use sites weaker than this, might be artifacts
#     weak_num=500,
#     strong_thresh_pct=99,  # don't use sites weaker than this, might be artifacts
#     strong_num=500,
#     save_tsv=None,  # optional filename to save a tsv
# ):
#     """creates a tsv with strong followed by weak sequences, which can be used as input to akita_insert.py or akita_flat_map.py"""

#     sites = filter_boundary_h5(h5_dirs=h5_dirs, score_key=score_key)

#     strong_sites, weak_sites = filter_sites_by_score(
#         sites,
#         score_key=score_key,
#         weak_thresh_pct=weak_thresh_pct,
#         weak_num=weak_num,
#         strong_thresh_pct=strong_thresh_pct,
#         strong_num=strong_num,
#     )

#     site_df = pd.concat([strong_sites.copy(), weak_sites.copy()])
#     seq_coords_df = (
#         site_df[["chrom", "start_2", "end_2", "strand_2", "SCD"]]
#         .copy()
#         .rename(
#             columns={
#                 "start_2": "start",
#                 "end_2": "end",
#                 "strand_2": "strand",
#                 "SCD": "genomic_SCD",
#             }
#         )
#     )
#     seq_coords_df.reset_index(inplace=True)
#     seq_coords_df = bioframe.expand(seq_coords_df, pad=flank_pad)
#     print("df prepared")
#     if save_tsv is not None:
#         seq_coords_df.to_csv(save_tsv, sep="\t", index=False)
#     return seq_coords_df


### sequence handling


def dna_rc(seq):
    return seq.translate(str.maketrans("ATCGatcg", "TAGCtagc"))[::-1]


def permute_seq_k(seq_1hot, k=2):
    if np.mod(k, 2) != 0:
        raise ValueError("current implementation only works for multiples of 2")
    seq_1hot_perm = np.zeros(np.shape(seq_1hot)).astype(int)
    perm_inds = k * np.random.permutation(np.arange(len(seq_1hot) // k))
    for i in range(k):
        seq_1hot_perm[i::k] = seq_1hot[perm_inds + i, :].copy()
    return seq_1hot_perm


### motif handling
def scan_motif(seq_1hot, motif, strand=None):
    if motif.shape[-1] != 4:
        raise ValueError("motif should be n_postions x 4 bases, A=0, C=1, G=2, T=3")
    if seq_1hot.shape[-1] != 4:
        raise ValueError("seq_1hot should be n_postions x 4 bases, A=0, C=1, G=2, T=3")
    scan_forward = tf.nn.conv1d(
        np.expand_dims(seq_1hot, 0).astype(float),
        np.expand_dims(motif, -1).astype(float),
        stride=1,
        padding="SAME",
    ).numpy()[0]
    if strand == "forward":
        return scan_forward
    scan_reverse = tf.nn.conv1d(
        np.expand_dims(seq_1hot, 0).astype(float),
        np.expand_dims(dna_io.hot1_rc(motif), -1).astype(float),
        stride=1,
        padding="SAME",
    ).numpy()[0]
    if strand == "reverse":
        return scan_reverse
    return np.maximum(scan_forward, scan_reverse).flatten()


def read_jaspar_to_numpy(motif_file, normalize=True):
    ## read jaspar pfm
    with open(motif_file, "r") as f:
        motif = []
        for line in f.readlines():
            if ">" in line:
                continue
            else:
                motif.append(line.strip().replace("[", "").replace("]", "").split())
    motif = pd.DataFrame(motif).set_index(0).astype(float).values.T
    if normalize == True:
        motif /= motif.sum(axis=1)[:, None]
    if motif.shape[1] != 4:
        raise ValueError("motif returned should be have n_positions x 4 bases")
    return motif



