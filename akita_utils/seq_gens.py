from akita_utils.dna_utils import hot1_rc, dna_1hot
from akita_utils.dna_utils import permute_seq_k


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


def symmertic_insertion_seqs_gen(seq_coords_df, background_seqs, genome_open):
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
            "The background generation methods using jaspar_motif_file have not been included."
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
                "Other background generation methods have not been included."
            )
