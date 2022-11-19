### akita utilities

import bioframe
import pandas as pd
import numpy as np
import tensorflow as tf
import glob
from io import StringIO
import h5py
import random

### numeric utilites
def absmaxND(a, axis=None):
    """
    https://stackoverflow.com/a/39152275
    """
    amax = a.max(axis)
    amin = a.min(axis)
    return np.where(-amin > amax, amin, amax)


def smooth(y, box_pts):
    box = np.ones(box_pts) / box_pts
    y_smooth = np.convolve(y, box, mode="same")
    return y_smooth


def ut_dense(preds_ut, diagonal_offset):
    """Construct symmetric dense prediction matrices from upper triangular vectors.

    Parameters
    -----------
    preds_ut : ( M x O) numpy array
        Upper triangular matrix to convert. M is the number of upper triangular entries,
        and O corresponds to the number of different targets.
    diagonal_offset : int
        Number of diagonals that are added as zeros in the conversion.
        Typically 2 diagonals are ignored in Hi-C data processing.

    Returns
    --------
    preds_dense : (D x D x O) numpy array
        Each output upper-triangular vector is converted to a symmetric D x D matrix.
        Output matrices have zeros at the diagonal for `diagonal_offset` number of diagonals.

    """
    ut_len, num_targets = preds_ut.shape

    # infer original sequence length
    seq_len = int(np.sqrt(2 * ut_len + 0.25) - 0.5)
    seq_len += diagonal_offset

    # get triu indexes
    ut_indexes = np.triu_indices(seq_len, diagonal_offset)
    assert len(ut_indexes[0]) == ut_len

    # assign to dense matrix
    preds_dense = np.zeros(shape=(seq_len, seq_len, num_targets), dtype=preds_ut.dtype)
    preds_dense[ut_indexes] = preds_ut

    # symmetrize
    preds_dense += np.transpose(preds_dense, axes=[1, 0, 2])

    return preds_dense


# def prepare_insertion_tsv(
#     h5_dirs="/project/fudenber_735/tensorflow_models/akita/v2/analysis/permute_boundaries_motifs_ctcf_mm10_model*/scd.h5",
#     score_key="SCD",
#     threshold_all_ctcf=5,  # reasonable cutoff for ctcf sites with impactful genomic SCD
#     pad_flank=60,  # how much flanking sequence around the sites to include
#     weak_thresh_pct=1,  # don't use sites weaker than this, might be artifacts
#     weak_num=500,
#     strong_thresh_pct=99,  # don't use sites stronger than this, might be artifacts
#     strong_num=500,
#     save_tsv=None,  # optional filename to save a tsv
# ):
#     """creates a tsv with strong followed by weak sequences, which can be used as input to akita_insert.py or akita_flat_map.py"""

#     sites = filter_boundary_ctcfs_from_h5(
#         h5_dirs=h5_dirs, score_key=score_key, threshold_all_ctcf=threshold_all_ctcf
#     )

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
#         site_df[["chrom", "start_2", "end_2", "strand_2", score_key]]
#         .copy()
#         .rename(
#             columns={
#                 "start_2": "start",
#                 "end_2": "end",
#                 "strand_2": "strand",
#                 score_key: "genomic_" + score_key,
#             }
#         )
#     )
#     seq_coords_df.reset_index(drop=True, inplace=True)
#     seq_coords_df.reset_index(inplace=True)
#     seq_coords_df = bioframe.expand(seq_coords_df, pad=pad_flank)
#     print("df prepared")
#     if save_tsv is not None:
#         seq_coords_df.to_csv(save_tsv, sep="\t", index=False)
#     return seq_coords_df


# def _generate_paired_core_flank_df(site_df, pad_core=0, pad_flank=60):
#     pair_columns = (
#         list(site_df.columns + "_core")
#         + ["pad_core"]
#         + list(site_df.columns + "_flank")
#         + ["pad_flank"]
#     )
#     all_pairs = []
#     for core_ind, core_site in site_df.copy().iterrows():
#         for flank_ind, flank_site in site_df.copy().iterrows():
#             all_pairs.append(
#                 list(core_site.values)
#                 + list([pad_core])
#                 + list(flank_site.values)
#                 + list([pad_flank])
#             )
#     return pd.DataFrame(all_pairs, columns=pair_columns)


# def prepare_paired_insertion_df(
#     h5_dirs="/project/fudenber_735/tensorflow_models/akita/v2/analysis/permute_boundaries_motifs_ctcf_mm10_model*/scd.h5",
#     score_key="SCD",
#     pad_core=0,
#     pad_flank=60,
#     weak_thresh_pct=1,  # don't use sites weaker than this, might be artifacts
#     weak_num=500,
#     strong_thresh_pct=99,  # don't use sites stronger than this, might be artifacts
#     strong_num=500,
#     save_tsv=None,  # optional filename to save a tsv
# ):
#     """
#     Prepare a DataFrame for paired motif insertions into a background sequence.

#     Parameters
#     ----------
#     As for prepare_insertion_tsv, but with the addition of pad_core and pad_flank.

#     Returns
#     -------
#     df_site_pairs : pd.DataFrame
#         14-column dataframe with:
#         index_core, chrom_core, start_core, end_core, strand_core score_key_core, pad_core,
#         index_core, chrom_flank, start_flank, end_flank, strand_flank, score_key_flank, pad_flank

#     """
#     if pad_core > pad_flank:
#         raise ValueError("the flanking sequence must be longer than the core sequence")
#     site_df = prepare_insertion_tsv(
#         h5_dirs=h5_dirs,
#         score_key=score_key,
#         weak_thresh_pct=weak_thresh_pct,
#         weak_num=weak_num,
#         strong_num=strong_num,
#         strong_thresh_pct=strong_thresh_pct,
#         save_tsv=None,
#         pad_flank=0,
#     )
#     df_site_pairs = _generate_paired_core_flank_df(
#         site_df, pad_core=pad_core, pad_flank=pad_flank
#     )

#     print("paired df prepared")
#     if save_tsv is not None:
#         df_site_pairs.to_csv(save_tsv, sep="\t", index=False)
#     return df_site_pairs


def split_df_equally(df, num_chunks, chunk_idx):
    
    df_len = len(df)    # but indices are 0 -> 198

    chunks_bounds = np.linspace(
        0, df_len, num_chunks + 1, dtype="int"
    )

    df_chunk = df.loc[
        chunks_bounds[chunk_idx] : (chunks_bounds[chunk_idx + 1]-1), :
    ]
    
    return df_chunk

