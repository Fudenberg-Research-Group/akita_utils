### akita utilities

import numpy as np

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

def dna_rc(seq):
    return seq.translate(str.maketrans("ATCGatcg", "TAGCtagc"))[::-1]


def dna_1hot_index(seq, n_sample=False):
  """ dna_1hot_index
    Args:
      seq:       nucleotide sequence.
    Returns:
      seq_code:  index int array representation.
    """
  seq_len = len(seq)
  seq = seq.upper()

  # map nt's to a len(seq) of 0,1,2,3
  seq_code = np.zeros(seq_len, dtype='uint8')
    
  for i in range(seq_len):
    nt = seq[i]
    if nt == 'A':
      seq_code[i] = 0
    elif nt == 'C':
      seq_code[i] = 1
    elif nt == 'G':
      seq_code[i] = 2
    elif nt == 'T':
      seq_code[i] = 3
    else:
      if n_sample:
        seq_code[i] = random.randint(0,3)
      else:
        seq_code[i] = 4

  return seq_code

def dna_1hot_GC(seq):
  seq_len = len(seq)
  seq = seq.upper()
  # map nt's to a len(seq) of 0 = A,T; 1 = G,C
  seq_code = np.zeros(seq_len, dtype='uint8')
  for i in range(seq_len):
    nt = seq[i]
    if nt == 'A':
      seq_code[i] = 0
    elif nt == 'C':
      seq_code[i] = 1
    elif nt == 'G':
      seq_code[i] = 1
    elif nt == 'T':
      seq_code[i] = 0
    else:
      seq_code[i] = random.randint(0,1)         
  return seq_code


def dna_1hot(seq, seq_len=None, n_uniform=False, n_sample=False):
  """ dna_1hot
    Args:
      seq:       nucleotide sequence.
      seq_len:   length to extend/trim sequences to.
      n_uniform: represent N's as 0.25, forcing float16,
      n_sample:  sample ACGT for N
    Returns:
      seq_code: length by nucleotides array representation.
    """
  if seq_len is None:
    seq_len = len(seq)
    seq_start = 0
  else:
    if seq_len <= len(seq):
      # trim the sequence
      seq_trim = (len(seq) - seq_len) // 2
      seq = seq[seq_trim:seq_trim + seq_len]
      seq_start = 0
    else:
      seq_start = (seq_len - len(seq)) // 2

  seq = seq.upper()

  # map nt's to a matrix len(seq)x4 of 0's and 1's.
  if n_uniform:
    seq_code = np.zeros((seq_len, 4), dtype='float16')
  else:
    seq_code = np.zeros((seq_len, 4), dtype='bool')
    
  for i in range(seq_len):
    if i >= seq_start and i - seq_start < len(seq):
      nt = seq[i - seq_start]
      if nt == 'A':
        seq_code[i, 0] = 1
      elif nt == 'C':
        seq_code[i, 1] = 1
      elif nt == 'G':
        seq_code[i, 2] = 1
      elif nt == 'T':
        seq_code[i, 3] = 1
      else:
        if n_uniform:
          seq_code[i, :] = 0.25
        elif n_sample:
          ni = random.randint(0,3)
          seq_code[i, ni] = 1

  return seq_code


def permute_seq_k(seq_1hot, k=2):
    """
    Permute a 1hot encoded sequence by k-mers.
    Parameters
    ----------
    seq_1hot : numpy.array
        n_bases x 4 array
    k : int
        number of bases kept together in permutations.
    """

    if np.mod(k, 2) != 0:
        raise ValueError("current implementation only works for multiples of 2")
    seq_1hot_perm = np.zeros(np.shape(seq_1hot)).astype(int)
    perm_inds = k * np.random.permutation(np.arange(len(seq_1hot) // k))
    for i in range(k):
        seq_1hot_perm[i::k] = seq_1hot[perm_inds + i, :].copy()
    return seq_1hot_perm


def hot1_rc(seqs_1hot):
    """Reverse complement a batch of one hot coded sequences,
    while being robust to additional tracks beyond the four
    nucleotides."""

    if seqs_1hot.ndim == 2:
        singleton = True
        seqs_1hot = np.expand_dims(seqs_1hot, axis=0)
    else:
        singleton = False

    seqs_1hot_rc = seqs_1hot.copy()

    # reverse
    seqs_1hot_rc = seqs_1hot_rc[:, ::-1, :]

    # swap A and T
    seqs_1hot_rc[:, :, [0, 3]] = seqs_1hot_rc[:, :, [3, 0]]

    # swap C and G
    seqs_1hot_rc[:, :, [1, 2]] = seqs_1hot_rc[:, :, [2, 1]]

    if singleton:
        seqs_1hot_rc = seqs_1hot_rc[0]

    return seqs_1hot_rc


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
        np.expand_dims(hot1_rc(motif), -1).astype(float),
        stride=1,
        padding="SAME",
    ).numpy()[0]
    if strand == "reverse":
        return scan_reverse
    return np.maximum(scan_forward, scan_reverse).flatten()


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
