import numpy as np
import random
import tensorflow as tf


def dna_rc(seq):
    """
    Generate the reverse complement of a DNA sequence.

    Parameters
    ------------
    seq : str
        DNA sequence composed of 'A', 'T', 'C', and 'G' characters (case insensitive).

    Returns
    ---------
    str
        Reverse complement of the input DNA sequence.
    """
    return seq.translate(str.maketrans("ATCGatcg", "TAGCtagc"))[::-1]


def dna_1hot_index(seq, n_sample=False):
    """
    Convert a DNA sequence to a numerical index array, with an option for handling unknown nucleotides.

    Parameters
    ------------
    seq : str
        DNA sequence composed of 'A', 'T', 'C', and 'G' characters.
    n_sample : bool, optional
        If True, randomly assign a value of 0, 1, 2, or 3 to unknown nucleotides.
        If False, assign a value of 4 to unknown nucleotides. Default is False.

    Returns
    ---------
    seq_code : numpy array
        Array of numerical indices representing the DNA sequence, where 'A' -> 0, 'C' -> 1, 'G' -> 2, 'T' -> 3.
        Unknown nucleotides are assigned either a random value (0-3) or 4 based on the `n_sample` parameter.
    """
    seq_len = len(seq)
    seq = seq.upper()

    # map nt's to a len(seq) of 0,1,2,3
    seq_code = np.zeros(seq_len, dtype="uint8")

    for i in range(seq_len):
        nt = seq[i]
        if nt == "A":
            seq_code[i] = 0
        elif nt == "C":
            seq_code[i] = 1
        elif nt == "G":
            seq_code[i] = 2
        elif nt == "T":
            seq_code[i] = 3
        else:
            if n_sample:
                seq_code[i] = random.randint(0, 3)
            else:
                seq_code[i] = 4

    return seq_code


def dna_1hot_GC(seq):
    """
    Convert a DNA sequence to a binary array indicating GC content.

    Parameters
    ------------
    seq : str
        DNA sequence composed of 'A', 'T', 'C', and 'G' characters.

    Returns
    ---------
    seq_code : numpy array
        Array of binary values representing the DNA sequence, where 'A' and 'T' -> 0, 'G' and 'C' -> 1.
        Unknown nucleotides are assigned a random value of 0 or 1.
    """
    seq_len = len(seq)
    seq = seq.upper()
    # map nt's to a len(seq) of 0 = A,T; 1 = G,C
    seq_code = np.zeros(seq_len, dtype="uint8")
    for i in range(seq_len):
        nt = seq[i]
        if nt == "A":
            seq_code[i] = 0
        elif nt == "C":
            seq_code[i] = 1
        elif nt == "G":
            seq_code[i] = 1
        elif nt == "T":
            seq_code[i] = 0
        else:
            seq_code[i] = random.randint(0, 1)
    return seq_code


def dna_1hot(seq, seq_len=None, n_uniform=False, n_sample=False):
    """
    Convert a DNA sequence to a one-hot encoded matrix with optional sequence length adjustment and handling of unknown nucleotides.

    Parameters
    ------------
    seq : str
        DNA sequence composed of 'A', 'T', 'C', and 'G' characters.
    seq_len : int, optional
        Desired length of the output sequence. If None, the length of the input sequence is used.
        If specified and shorter than the input sequence, the sequence is trimmed. If longer, the sequence is centered.
    n_uniform : bool, optional
        If True, encode unknown nucleotides as [0.25, 0.25, 0.25, 0.25]. If False, encode unknown nucleotides as [0, 0, 0, 0] or one-hot based on `n_sample`.
    n_sample : bool, optional
        If True, randomly assign one of the four nucleotides (one-hot encoded) to unknown nucleotides. Used only if `n_uniform` is False. Default is False.

    Returns
    ---------
    seq_code : numpy array
        A matrix of shape (seq_len, 4) with one-hot encoded DNA sequence, where 'A' -> [1, 0, 0, 0], 'C' -> [0, 1, 0, 0],
        'G' -> [0, 0, 1, 0], 'T' -> [0, 0, 0, 1]. Unknown nucleotides are handled according to `n_uniform` and `n_sample`.
    """
    if seq_len is None:
        seq_len = len(seq)
        seq_start = 0
    else:
        if seq_len <= len(seq):
            # trim the sequence
            seq_trim = (len(seq) - seq_len) // 2
            seq = seq[seq_trim : seq_trim + seq_len]
            seq_start = 0
        else:
            seq_start = (seq_len - len(seq)) // 2

    seq = seq.upper()

    # map nt's to a matrix len(seq)x4 of 0's and 1's.
    if n_uniform:
        seq_code = np.zeros((seq_len, 4), dtype="float16")
    else:
        seq_code = np.zeros((seq_len, 4), dtype="bool")

    for i in range(seq_len):
        if i >= seq_start and i - seq_start < len(seq):
            nt = seq[i - seq_start]
            if nt == "A":
                seq_code[i, 0] = 1
            elif nt == "C":
                seq_code[i, 1] = 1
            elif nt == "G":
                seq_code[i, 2] = 1
            elif nt == "T":
                seq_code[i, 3] = 1
            else:
                if n_uniform:
                    seq_code[i, :] = 0.25
                elif n_sample:
                    ni = random.randint(0, 3)
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

    seq_length = len(seq_1hot)
    if seq_length % k != 0:
        raise ValueError("Sequence length must be divisible by k")

    seq_1hot_perm = np.zeros_like(seq_1hot)

    num_permutations = seq_length // k
    perm_inds = np.arange(num_permutations) * k
    np.random.shuffle(perm_inds)

    for i in range(k):
        seq_1hot_perm[i::k] = seq_1hot[perm_inds + i, :].copy()

    return seq_1hot_perm


def hot1_rc(seqs_1hot):
    """
    Generate the reverse complement of one-hot encoded DNA sequences.

    Parameters
    ------------
    seqs_1hot : numpy array
        A 2D array of shape (sequence_length, 4) or a 3D array of shape (num_sequences, sequence_length, 4)
        representing one-hot encoded DNA sequences, where 'A' -> [1, 0, 0, 0], 'C' -> [0, 1, 0, 0],
        'G' -> [0, 0, 1, 0], 'T' -> [0, 0, 0, 1].

    Returns
    ---------
    seqs_1hot_rc : numpy array
        Array of the same shape as the input, representing the reverse complement of the input one-hot encoded DNA sequences.
    """
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
    """
    Scan a one-hot encoded DNA sequence for a given motif and return the match scores.

    Parameters
    ------------
    seq_1hot : numpy array
        A 2D array of shape (sequence_length, 4) representing a one-hot encoded DNA sequence,
        where 'A' -> [1, 0, 0, 0], 'C' -> [0, 1, 0, 0], 'G' -> [0, 0, 1, 0], 'T' -> [0, 0, 0, 1].
    motif : numpy array
        A 2D array of shape (motif_length, 4) representing the motif to scan for, where each row is
        the nucleotide probabilities for a position in the motif.
    strand : str, optional
        Specify 'forward' to scan only the forward strand, 'reverse' to scan only the reverse strand,
        or None to scan both and return the maximum score. Default is None.

    Returns
    ---------
    numpy array
        An array of match scores for the motif along the DNA sequence. If `strand` is specified, returns
        the scores for the specified strand. If `strand` is None, returns the maximum scores from both strands.
    """
    if motif.shape[-1] != 4:
        raise ValueError(
            "motif should be n_postions x 4 bases, A=0, C=1, G=2, T=3"
        )
    if seq_1hot.shape[-1] != 4:
        raise ValueError(
            "seq_1hot should be n_postions x 4 bases, A=0, C=1, G=2, T=3"
        )
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


def hot1_get(seqs_1hot, pos):
    """
    Retrieve the nucleotide at a specific position from a one-hot encoded DNA sequence.

    Parameters
    ------------
    seqs_1hot : numpy array
        A 2D array of shape (sequence_length, 4) representing a one-hot encoded DNA sequence,
        where 'A' -> [1, 0, 0, 0], 'C' -> [0, 1, 0, 0], 'G' -> [0, 0, 1, 0], 'T' -> [0, 0, 0, 1].
    pos : int
        Position in the sequence to retrieve the nucleotide from.

    Returns
    ---------
    nt : str
        The nucleotide ('A', 'C', 'G', 'T') at the specified position. Returns 'N' if no valid nucleotide is found.
    """
    if seqs_1hot[pos, 0] == 1:
        nt = "A"
    elif seqs_1hot[pos, 1] == 1:
        nt = "C"
    elif seqs_1hot[pos, 2] == 1:
        nt = "G"
    elif seqs_1hot[pos, 3] == 1:
        nt = "T"
    else:
        nt = "N"
    return nt


def dna_1hot_to_seq(seq_1hot):
    """
    Convert a one-hot encoded DNA sequence back to its nucleotide sequence.

    Parameters
    ------------
    seq_1hot : numpy array
        A 2D array of shape (sequence_length, 4) representing a one-hot encoded DNA sequence,
        where 'A' -> [1, 0, 0, 0], 'C' -> [0, 1, 0, 0], 'G' -> [0, 0, 1, 0], 'T' -> [0, 0, 0, 1].

    Returns
    ---------
    ACTG_seq : str
        The nucleotide sequence corresponding to the one-hot encoded input sequence.
    """
    ACTG_seq = str()
    for pos in range(len(seq_1hot)):
        ACTG_seq = ACTG_seq + hot1_get(seq_1hot, pos)
    return ACTG_seq


def dna_seq_rc(seq):
    """
    Generate the reverse complement of a DNA sequence.

    Parameters
    ------------
    seq : str
        DNA sequence composed of 'A', 'T', 'C', and 'G' characters.

    Returns
    ---------
    rc_seq : str
        The reverse complement of the input DNA sequence.
    """
    rc_seq = ""
    for nt in seq:
        if nt == "A":
            rc_seq = rc_seq + "T"
        elif nt == "C":
            rc_seq = rc_seq + "G"
        elif nt == "G":
            rc_seq = rc_seq + "C"
        elif nt == "T":
            rc_seq = rc_seq + "A"
    return rc_seq[::-1]
