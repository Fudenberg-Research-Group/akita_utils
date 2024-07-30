import numpy as np
from .dna_utils import hot1_rc, dna_1hot, permute_seq_k, dna_rc


# reference (wild type)


def reference_seqs_gen(background_seqs):
    """
    Generate one-hot encoded reference sequences from a list of background DNA sequences.

    This iterator function takes a list of background DNA sequences `background_seqs`
    and yields one-hot encoded reference sequences that can be used as input to Akita
    via PredStreamGen.

    Parameters:
    - background_seqs (List[str]): List of background DNA sequences.

    Yields:
    numpy.ndarray: One-hot encoded DNA reference sequence.
    """

    for background_index in range(len(background_seqs)):
        seq_1hot = background_seqs[background_index].copy()
        yield seq_1hot


# insertion experiment


def _insert_casette(
    seq_1hot, seq_1hot_insertion, spacer_bp, orientation_string
):
    """
    Insert a casette sequence into a given one-hot encoded DNA sequence.

    This function takes a one-hot encoded DNA sequence `seq_1hot`, an insertion
    sequence `seq_1hot_insertion`, the number of base pairs for intert-insert spacers
    `spacer_bp`, and an orientation string `orientation_string` specifying the orientation
    and number of insertions. It inserts the given casette sequence into the original sequence
    based on the specified orientations and returns the modified sequence.

    Parameters:
    - seq_1hot (numpy.ndarray): One-hot encoded DNA sequence to be modified.
    - seq_1hot_insertion (numpy.ndarray): One-hot encoded DNA sequence to be inserted.
    - spacer_bp (int): Number of base pairs for intert-insert spacers.
    - orientation_string (str): String specifying the orientation and number of insertions.
                               '>' denotes forward orientation, and '<' denotes reverse.

    Returns:
    numpy.ndarray: One-hot encoded DNA sequence with the casette insertion.

    Raises:
    AssertionError: If the insertion offset is outside the valid range or if the length
                    of the insert and inter-insert spacing leads to an invalid offset.
    """

    seq_length = seq_1hot.shape[0]
    insert_bp = len(seq_1hot_insertion)
    num_inserts = len(orientation_string)

    insert_plus_spacer_bp = insert_bp + 2 * spacer_bp
    multi_insert_bp = num_inserts * insert_plus_spacer_bp
    insert_start_bp = seq_length // 2 - multi_insert_bp // 2

    output_seq = seq_1hot.copy()
    insertion_starting_positions = []
    for i in range(num_inserts):
        offset = insert_start_bp + i * insert_plus_spacer_bp + spacer_bp
        insertion_starting_positions.append(offset)

        assert (
            offset >= 0 and offset < seq_length - insert_bp
        ), f"offset = {offset}. Please, check length of insert and inter-insert spacing."

        for orientation_arrow in orientation_string[i]:
            if orientation_arrow == ">":
                output_seq[offset : offset + insert_bp] = seq_1hot_insertion
            else:
                output_seq[offset : offset + insert_bp] = hot1_rc(
                    seq_1hot_insertion
                )

    return output_seq


def symmertic_insertion_seqs_gen(
    seq_coords_df, background_seqs, genome_open, nproc=1, map=map
):
    """
    Generate sequences with symmetric insertions for a given set of coordinates.

    This generator function takes a DataFrame `seq_coords_df` containing genomic
    coordinates, a list of background sequences `background_seqs`, and a genome file
    handler `genome_open`. It yields one-hot encoded DNA sequences with symmetric
    insertions based on the specified coordinates.

    Parameters:
    - seq_coords_df (pandas.DataFrame): DataFrame with columns 'chrom', 'start', 'end',
                                         'strand', 'flank_bp', 'spacer_bp', 'orientation'.
                                         Represents genomic coordinates and insertion parameters.
    - background_seqs (List[numpy.ndarray]): List of background sequences to be modified.
    - genome_open (GenomeFileHandler): A file handler for the genome to fetch sequences.

    Yields:
    numpy.ndarray: One-hot encoded DNA sequence with symmetric insertions.
    """
    for s in seq_coords_df.itertuples():
        flank_bp = s.flank_bp
        spacer_bp = s.spacer_bp
        orientation_string = s.orientation

        seq_1hot_insertion = dna_1hot(
            genome_open.fetch(
                s.chrom, s.start - flank_bp, s.end + flank_bp
            ).upper()
        )

        if s.strand == "-":
            seq_1hot_insertion = hot1_rc(seq_1hot_insertion)
            # now, all motifs are standarized to this orientation ">"

        seq_1hot = background_seqs[s.background_index].copy()

        seq_1hot = _insert_casette(
            seq_1hot, seq_1hot_insertion, spacer_bp, orientation_string
        )

        yield seq_1hot


# disruption experiment


def expand_and_check_window(s, chrom_sizes_table, shift=0, seq_length=1310720):
    """
    Expands a genomic window to a specified sequence length and checks its validity against chromosome sizes.

    Given a genomic window (defined by chromosome, start, and end), this function expands the window to a specified
    sequence length. It ensures that the expanded window is symmetric around the original window and optionally
    applies a shift. It then checks if the expanded window is valid (i.e., within the bounds of the chromosome).

    Parameters:
    - s (Series): A pandas Series or similar object containing the original genomic window, with 'chrom', 'start',
      and 'end' fields.
    - chrom_sizes_table (DataFrame): A pandas DataFrame containing chromosome sizes with 'chrom' and 'size' columns.
    - shift (int, optional): The number of base pairs to shift the center of the window. Default is 0.
    - seq_length (int, optional): The desired total length of the expanded window. Default is 1310720.

    Returns:
    - tuple: A tuple (chrom, up_start, down_end) representing the chromosome, the start, and the end of the
      expanded and potentially shifted window.

    Note: If the shift is too large or if the expanded window extends beyond the chromosome boundaries, an
    Exception is raised.
    """
    chrom, start, end = s.chrom, s.start, s.end
    if abs(end - start) % 2 != 0:
        start = start - 1

    span_length = abs(end - start)
    length_diff = seq_length - span_length
    up_length = down_length = length_diff // 2

    if shift > up_length:
        raise Exception(
            "For the following window of interest: ",
            chrom,
            start,
            end,
            "shift excludes the window of interest from the prediction window.",
        )

    # start and end in genomic coordinates (optionally shifted)
    up_start, down_end = start - up_length - shift, end + down_length - shift

    # checking if a genomic prediction can be centered around the span
    chr_size = int(
        chrom_sizes_table.loc[
            chrom_sizes_table["chrom"] == chrom, "size"
        ].iloc[0]
    )

    if up_start < 0 or down_end > chr_size:
        raise Exception(
            "The prediction window for the following window of interest: ",
            chrom,
            start,
            end,
            "cannot be centered.",
        )
    return chrom, up_start, down_end


def get_relative_window_coordinates(s, shift=0, seq_length=1310720):
    """
    Calculates the relative coordinates of a genomic window within an expanded and optionally shifted sequence.

    This function takes a genomic window (defined by start and end positions) and calculates its relative
    position within an expanded sequence of a specified length. The expansion is symmetric around the original
    window, and an optional shift can be applied. The function returns the relative start and end positions
    of the original window within this expanded sequence.

    Parameters:
    - s (Series): A pandas Series or similar object containing the original genomic window, with 'start'
      and 'end' fields.
    - shift (int, optional): The number of base pairs to shift the center of the window. Default is 0.
    - seq_length (int, optional): The total length of the expanded sequence. Default is 1310720.

    Returns:
    - tuple: A tuple (relative_start, relative_end) representing the relative start and end positions
      of the original window within the expanded sequence.
    """
    start, end = s.start, s.end
    if abs(end - start) % 2 != 0:
        start = start - 1

    span_length = abs(end - start)
    length_diff = seq_length - span_length
    up_length = length_diff // 2

    # relative start and end of the span of interest in the prediction window
    relative_start = up_length + 1 + shift
    relative_end = relative_start + span_length

    return relative_start, relative_end


def central_permutation_seqs_gen(
    seq_coords_df,
    genome_open,
    chrom_sizes_table,
    permutation_window_shift=0,
    revcomp=False,
    seq_length=1310720,
):
    """
    Generates sequences for a set of genomic coordinates, applying central permutations and optionally
    operating on reverse complements, with an additional option for shifting the permutation window.

    This generator function takes a DataFrame `seq_coords_df` containing genomic coordinates
    (chromosome, start, end, strand), a genome file handler `genome_open` to fetch sequences, and
    a table of chromosome sizes `chrom_sizes_table`. It yields sequences with central permutations
    around the coordinates specified in `seq_coords_df`, considering an optional shift for the
    permutation window. If `rc` is True, the reverse complement of these sequences is generated.

    Parameters:
    - seq_coords_df (pandas.DataFrame): DataFrame with columns 'chrom', 'start', 'end', 'strand',
                                        representing genomic coordinates of interest.
    - genome_open (GenomeFileHandler): A file handler for the genome to fetch sequences.
    - chrom_sizes_table (pandas.DataFrame): DataFrame with columns 'chrom' and 'size', representing
                                            the sizes of chromosomes in the genome.
    - permutation_window_shift (int, optional): The number of base pairs to shift the center of the
                                                 permutation window. Default is 0.
    - rc (bool, optional): If True, operates on reverse complement of the sequences. Default is False.
    - seq_length (int, optional): The total length of the sequence to be generated. Default is 1310720.

    Yields:
    numpy.ndarray: One-hot encoded DNA sequences. Each sequence is either the original or its central
                   permutation, with or without reverse complement as specified by `rc`.

    Raises:
    Exception: If the prediction window for a given span cannot be centered within the chromosome.
    """

    for s in seq_coords_df.itertuples():
        list_1hot = []

        chrom, window_start, window_end = expand_and_check_window(
            s, chrom_sizes_table, shift=permutation_window_shift
        )
        permutation_start, permutation_end = get_relative_window_coordinates(
            s, shift=permutation_window_shift
        )

        wt_seq_1hot = dna_1hot(
            genome_open.fetch(chrom, window_start, window_end).upper()
        )
        if revcomp:
            rc_wt_seq_1hot = hot1_rc(wt_seq_1hot)
            list_1hot.append(rc_wt_seq_1hot.copy())
        else:
            list_1hot.append(wt_seq_1hot.copy())

        alt_seq_1hot = wt_seq_1hot.copy()
        permuted_span = permute_seq_k(
            alt_seq_1hot[permutation_start:permutation_end], k=1
        )
        alt_seq_1hot[permutation_start:permutation_end] = permuted_span

        if revcomp:
            rc_alt_seq_1hot = hot1_rc(alt_seq_1hot.copy())
            list_1hot.append(rc_alt_seq_1hot)
        else:
            list_1hot.append(alt_seq_1hot)

        # yielding first the reference, then the permuted sequence
        for sequence in list_1hot:
            yield sequence


def sliding_disruption_seq_gen(
    seq_coords_df, genome_open, split=10, bin_size=2048, seq_length=1310720
):
    """
    Generates wild-type and permuted sequences in one-hot encoding format for given genomic coordinates.

    This function iterates over a DataFrame containing genomic coordinates, fetches the wild-type sequence from
    a genome file, and generates permuted versions of the sequence by disrupting specific regions. The sequences
    are then returned in a one-hot encoding format.

    Args:
        seq_coords_df (pd.DataFrame): DataFrame containing genomic coordinates with columns 'chr', 'start', 'end', and 'genome_window_start'.
        genome_open (pysam.Fasta): Open genome file from which sequences will be fetched.
        split (int, optional): Number of splits to create for permutation within each sequence window. Default is 10.
        bin_size (int, optional): Size of the bins used for permutation. Default is 2048.
        seq_length (int, optional): Length of the sequence to be fetched from the genome. Default is 1310720.

    Yields:
        np.ndarray: One-hot encoded representation of the wild-type sequence and permuted sequences.
    """

    rel_start_permutation_bin = (seq_length // 2) - bin_size

    for s in seq_coords_df.itertuples():
        list_1hot = []
        window_start = s.genome_window_start
        chrom = s.chr

        # wild type
        wt_seq_1hot = dna_1hot(
            genome_open.fetch(
                chrom, window_start, window_start + seq_length
            ).upper()
        )
        list_1hot.append(wt_seq_1hot)

        for perm_index in range(split):
            permutation_start = (
                rel_start_permutation_bin + (bin_size // split) * perm_index
            )
            permutation_end = rel_start_permutation_bin + (
                bin_size // split
            ) * (perm_index + 1)

            alt_seq_1hot = wt_seq_1hot.copy()
            permuted_span = permute_seq_k(
                alt_seq_1hot[permutation_start:permutation_end], k=1
            )
            alt_seq_1hot[permutation_start:permutation_end] = permuted_span
            list_1hot.append(alt_seq_1hot)

        # yielding first the reference, then the permuted sequence
        for sequence in list_1hot:
            yield sequence


# generating shuffled sequences (once!)


def shuffled_sequences_gen(seq_coords_df, genome_open, jasper_motif_file=None):
    """
    Generates sequences with shuffled backgrounds based on the input sequences' coordinates.

    This generator function takes a DataFrame of sequence coordinates, an open genome file,
    and an optional JASPAR motif file. It iterates through each row in the DataFrame,
    fetches the corresponding DNA sequence from the genome, converts it into a one-hot encoding,
    and then applies a shuffling mutation method as specified by the 'mutation_method' column
    in the DataFrame. The function currently supports only the "permute_whole_seq" mutation method.
    If the 'jasper_motif_file' parameter is provided, the function raises a NotImplementedError
    as the handling for this parameter has not been included.

    Parameters:
    - seq_coords_df (DataFrame): A pandas DataFrame containing the sequence coordinates.
      It must have the columns 'chrom', 'start', 'end', and 'mutation_method'. Optionally,
      a 'shuffle_parameter' column is required for the "permute_whole_seq" mutation method.
    - genome_open (file-like): An open file-like object of the genome sequences, supporting
      the fetch method to retrieve DNA sequences given chromosome, start, and end positions.
    - jasper_motif_file (str, optional): Path to a JASPAR motif file. Currently, the use of this
      file is not implemented, and providing it will raise a NotImplementedError.

    Yields:
    - numpy.ndarray: A one-hot encoded numpy array representing the shuffled sequence.

    Raises:
    - NotImplementedError: If 'jasper_motif_file' is provided or if a mutation method other
      than "permute_whole_seq" is encountered.
    """

    if jasper_motif_file is not None:
        raise NotImplementedError(
            "Background generation using jaspar_motif_file not implemented."
        )

    for s in seq_coords_df.itertuples():
        chrom, start, end = s.chrom, s.start, s.end
        seq_dna = genome_open.fetch(chrom, int(start), int(end))
        wt_seq_1hot = dna_1hot(seq_dna)
        alt_seq_1hot = wt_seq_1hot.copy()
        mutation_method = s.mutation_method

        if mutation_method == "permute_whole_seq":
            permuted_alt_seq_1hot = permute_seq_k(
                alt_seq_1hot, k=s.shuffle_parameter
            )
            yield permuted_alt_seq_1hot
        else:
            raise NotImplementedError(
                "Alternative background generation methods not implemented."
            )


# tests on shuffling and its impact on insertion score


def unshuffled_insertion_gen(
    seq_coords_df,
    genome_open,
    ctcf_site_coordinates,
    flank_bp=30,
    orientation=">",
):
    """
    Generates sequences with CTCF site insertions at specified coordinates within genomic sequences.

    This function iterates over a DataFrame containing genomic sequence coordinates (seq_coords_df),
    retrieves each sequence from a genome file (genome_open), and inserts a specific CTCF binding site sequence
    into these sequences. The CTCF site is extended by a specified number of base pairs (flank_bp) on each side
    and can be inserted in a specific orientation. If the CTCF site is on the negative strand, the sequence
    is reverse complemented before insertion.

    Parameters:
    - seq_coords_df (pd.DataFrame): DataFrame with columns ['chrom', 'start', 'end'] specifying the chromosomes and
      start/end coordinates of sequences to process.
    - genome_open (pysam.Fastafile): An open pysam Fastafile object for the genome from which sequences are fetched.
    - ctcf_site_coordinates (tuple): A tuple containing the chromosome (str), start (int), end (int),
      and strand ('+' or '-') of the CTCF site to be inserted.
    - flank_bp (int, optional): The number of base pairs to extend on each side of the CTCF site. Default is 30.
    - orientation (str, optional): The orientation of the CTCF site insertion relative to the genomic sequence.
      Can be '>' for the same orientation or '<' for the opposite orientation. Default is '>'.

    Yields:
    - seq (np.array): 1-hot encoded numpy array representing a genomic sequence with the CTCF site insertion.
      The function yields two versions for each input sequence: one with the wild-type sequence and one with the
      CTCF site inserted.
    """

    ctcf_chrom, ctcf_start, ctcf_end, ctcf_strand = ctcf_site_coordinates

    seq_1hot_insertion = dna_1hot(
        genome_open.fetch(
            ctcf_chrom, ctcf_start - flank_bp, ctcf_end + flank_bp
        ).upper()
    )

    if ctcf_strand == "-":
        seq_1hot_insertion = hot1_rc(seq_1hot_insertion)

    for s in seq_coords_df.itertuples():
        sequences_to_yield = []

        # getting genomic sequence
        chrom, start, end = s.chrom, s.start, s.end
        seq_dna = genome_open.fetch(chrom, int(start), int(end))
        wt_seq_1hot = dna_1hot(seq_dna)
        sequences_to_yield.append(wt_seq_1hot)

        inserted_seq_1hot = _insert_casette(
            wt_seq_1hot, seq_1hot_insertion, 0, orientation
        )
        sequences_to_yield.append(inserted_seq_1hot)

        for seq in sequences_to_yield:
            yield seq


def shuffled_insertion_gen(
    seq_coords_df,
    genome_open,
    ctcf_site_coordinates,
    flank_bp=30,
    orientation=">",
):
    """
    Generates sequences with CTCF site insertions after shuffling genomic sequences at specified coordinates.

    This function iterates over a DataFrame containing genomic sequence coordinates (seq_coords_df),
    and for each sequence, it either shuffles the entire sequence or applies another mutation method
    specified in the DataFrame before inserting a specific CTCF binding site sequence into it. The CTCF
    site is extended by a specified number of base pairs (flank_bp) on each side and can be inserted in
    a specific orientation. If the CTCF site is on the negative strand, the sequence is reverse complemented
    before insertion.

    Parameters:
    - seq_coords_df (pd.DataFrame): DataFrame with columns ['chrom', 'start', 'end', 'mutation_method',
      'shuffle_parameter'] specifying the chromosomes, start/end coordinates of sequences to process, the
      method of mutation or shuffling to apply, and parameters for the shuffling.
    - genome_open (pysam.Fastafile): An open pysam Fastafile object for the genome from which sequences
      are fetched.
    - ctcf_site_coordinates (tuple): A tuple containing the chromosome (str), start (int), end (int),
      and strand ('+' or '-') of the CTCF site to be inserted.
    - flank_bp (int, optional): The number of base pairs to extend on each side of the CTCF site. Default
      is 30.
    - orientation (str, optional): The orientation of the CTCF site insertion relative to the genomic
      sequence. Can be '>' for the same orientation or '<' for the opposite orientation. Default is '>'.

    Yields:
    - seq (np.array): 1-hot encoded numpy array representing a genomic sequence with the CTCF site insertion
      after applying the specified mutation or shuffling method.
    """

    ctcf_chrom, ctcf_start, ctcf_end, ctcf_strand = ctcf_site_coordinates

    seq_1hot_insertion = dna_1hot(
        genome_open.fetch(
            ctcf_chrom, ctcf_start - flank_bp, ctcf_end + flank_bp
        ).upper()
    )

    if ctcf_strand == "-":
        seq_1hot_insertion = hot1_rc(seq_1hot_insertion)

    for s in seq_coords_df.itertuples():
        sequences_to_yield = []

        # getting genomic sequence
        chrom, start, end = s.chrom, s.start, s.end
        seq_dna = genome_open.fetch(chrom, int(start), int(end))
        wt_seq_1hot = dna_1hot(seq_dna)

        mutation_method = s.mutation_method

        if mutation_method == "permute_whole_seq":
            permuted_alt_seq_1hot = permute_seq_k(
                wt_seq_1hot, k=s.shuffle_parameter
            )
            sequences_to_yield.append(permuted_alt_seq_1hot)
        else:
            raise NotImplementedError(
                "Other background generation methods have not been included."
            )

        inserted_seq_1hot = _insert_casette(
            permuted_alt_seq_1hot, seq_1hot_insertion, 0, orientation
        )
        sequences_to_yield.append(inserted_seq_1hot)

        for seq in sequences_to_yield:
            yield seq


# flank-core compatibility


def flank_core_compatibility_seqs_gen(
    seq_coords_df, background_seqs, genome_open
):
    """
    Generates sequences by flanking a core sequence with upstream and downstream sequences.

    This function iterates over each row in the provided `seq_coords_df` DataFrame, and for each row:
        1. Retrieves the core sequence and its flanks from the genome.
        2. Converts these sequences into one-hot encoded representations.
        3. If necessary, reverses the sequences based on the strand information.
        4. Concatenates the upstream flank, core, and downstream flank sequences.
        5. Inserts this concatenated sequence into a background sequence.

    Parameters
    ----------
    seq_coords_df : pandas.DataFrame
        A DataFrame containing the coordinates and other relevant information for each sequence.
        Expected columns include chrom_core, start_core, end_core, strand_core, chrom_flank,
        start_flank, end_flank, strand_flank, flank_bp, background_index, spacer_bp, and orientation.
    background_seqs : list or array-like
        A list or array of background sequences (in one-hot encoded format) into which the
        concatenated sequences will be inserted.
    genome_open : object
        An open genome file object that allows fetching sequences from specific chromosomal coordinates.

    Yields
    ------
    numpy.ndarray
        A one-hot encoded numpy array representing the final sequence after insertion into the background sequence.
    """

    for s in seq_coords_df.itertuples():
        flank_bp = s.flank_bp

        # getting core
        seq_1hot_core = dna_1hot(
            genome_open.fetch(s.chrom_core, s.start_core, s.end_core).upper()
        )
        if s.strand_core == "-":
            seq_1hot_core = hot1_rc(seq_1hot_core)

        # getting flanks
        seq_1hot_flank = dna_1hot(
            genome_open.fetch(
                s.chrom_flank, s.start_flank - flank_bp, s.end_flank + flank_bp
            ).upper()
        )
        if s.strand_flank == "-":
            seq_1hot_flank = hot1_rc(seq_1hot_flank)

        seq_1hot_flank_upstream = seq_1hot_flank[:flank_bp, :]
        seq_1hot_flank_downstream = seq_1hot_flank[-flank_bp:, :]

        # joining all the chunks together
        seq_1hot_insertion = np.concatenate(
            (
                seq_1hot_flank_upstream,
                seq_1hot_core,
                seq_1hot_flank_downstream,
            ),
            axis=0,
        )
        seq_1hot = background_seqs[s.background_index].copy()
        seq_1hot = _insert_casette(
            seq_1hot, seq_1hot_insertion, s.spacer_bp, s.orientation
        )

        yield seq_1hot


# mutagenesis


def single_mutagenesis_seqs_gen(seq_coords_df, background_seqs, genome_open):
    """
    Generate sequences with single mutagenesis applied, based on specified sequence coordinates,
    background sequences, and a reference genome.

    This function iterates through a DataFrame of sequence coordinates (seq_coords_df), applying
    single mutagenesis to each sequence. For each sequence, it adjusts for flanking regions,
    retrieves the sequence from a reference genome, applies specified mutations, and then
    incorporates the mutated sequence into a background sequence.

    Parameters:
    - seq_coords_df (pandas.DataFrame): A DataFrame containing the sequence coordinates and
      mutation details. Expected columns include 'start', 'end', 'flank_bp', 'chrom', 'position',
      'original_nucleotide', 'mutated_nucleotide', 'strand', 'background_index', 'spacer_bp', and
      'orientation'.
    - background_seqs (list): A list of background sequences (e.g., as one-hot encoded arrays)
      into which the mutated sequences will be inserted.
    - genome_open (pysam.FastaFile): An open FastaFile object of the reference genome from which
      sequences will be fetched.

    Yields:
    - numpy.array: A one-hot encoded array representing the background sequence with the
      mutated sequence inserted at the specified position and orientation.
    """

    for s in seq_coords_df.itertuples():
        # Adjust start and end positions to include flanking regions
        start_adj = s.start - s.flank_bp
        end_adj = s.end + s.flank_bp

        # Fetch the sequence from the genome
        sequence = genome_open.fetch(s.chrom, start_adj, end_adj).upper()

        # Apply mutations
        # Adjust position to be relative to the fetched sequence, considering flanking regions
        pos_adj = s.position + s.flank_bp

        if s.strand == "-":
            # Reverse complement the sequence for negative strand
            sequence = dna_rc(sequence).upper()

        # Mutate the sequence at specified position
        sequence_list = list(sequence)
        sequence_list[pos_adj] = s.mutated_nucleotide
        mutated_sequence = "".join(sequence_list)

        seq_1hot_insertion = dna_1hot(mutated_sequence)
        seq_1hot = background_seqs[s.background_index].copy()
        seq_1hot = _insert_casette(
            seq_1hot, seq_1hot_insertion, s.spacer_bp, s.orientation
        )

        yield seq_1hot


def pairwise_mutagenesis_seqs_gen(seq_coords_df, background_seqs, genome_open):
    """
    Generate sequences with pairwise mutagenesis applied, based on specified sequence coordinates,
    background sequences, and a reference genome.

    This function iterates through a DataFrame of sequence coordinates (seq_coords_df), applying
    pairwise mutagenesis to each sequence. For each sequence, it adjusts for flanking regions,
    retrieves the sequence from a reference genome, applies specified mutations, and then
    incorporates the mutated sequence into a background sequence.

    Parameters:
    - seq_coords_df (pandas.DataFrame): A DataFrame containing the sequence coordinates and
      mutation details. Expected columns include 'start', 'end', 'flank_bp', 'chrom', 'pos1',
      'pos2', 'MutatedNuc1', 'MutatedNuc2', 'strand', 'background_index', 'spacer_bp', and
      'orientation'.
    - background_seqs (list): A list of background sequences (e.g., as one-hot encoded arrays)
      into which the mutated sequences will be inserted.
    - genome_open (pysam.FastaFile): An open FastaFile object of the reference genome from which
      sequences will be fetched.

    Yields:
    - numpy.array: A one-hot encoded array representing the background sequence with the
      mutated sequence inserted at the specified position and orientation.
    """

    for s in seq_coords_df.itertuples():
        # Adjust start and end positions to include flanking regions
        start_adj = s.start - s.flank_bp
        end_adj = s.end + s.flank_bp

        # Fetch the sequence from the genome
        sequence = genome_open.fetch(s.chrom, start_adj, end_adj).upper()

        # Apply mutations
        # Adjust positions to be relative to the fetched sequence, considering flanking regions
        pos1_adj = s.pos1 + s.flank_bp
        pos2_adj = s.pos2 + s.flank_bp

        if s.strand == "-":
            # Reverse complement the sequence for negative strand
            sequence = dna_rc(sequence).upper()

        # Mutate the sequence at specified positions
        sequence_list = list(sequence)
        sequence_list[pos1_adj] = s.MutatedNuc1
        sequence_list[pos2_adj] = s.MutatedNuc2
        mutated_sequence = "".join(sequence_list)

        seq_1hot_insertion = dna_1hot(mutated_sequence)
        seq_1hot = background_seqs[s.background_index].copy()
        seq_1hot = _insert_casette(
            seq_1hot, seq_1hot_insertion, s.spacer_bp, s.orientation
        )

        yield seq_1hot
