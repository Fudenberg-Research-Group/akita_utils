
import numpy as np


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


