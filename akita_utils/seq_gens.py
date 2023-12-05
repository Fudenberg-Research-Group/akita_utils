from akita_utils.dna_utils import hot1_rc, dna_1hot
import numpy as np
import akita_utils.format_io
import pandas as pd
import logging
import math
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


def _multi_insert_offsets_casette(
    seq_1hot, seq_1hot_insertions, offsets_bp, orientation_string
):
    """Insert multiple DNA sequences into a given DNA sequence.

    Args:
        seq_1hot (numpy.ndarray): The DNA sequence to be modified, in one-hot encoding format.
        seq_1hot_insertions (list of numpy.ndarray): A list of DNA sequences to be inserted into `seq_1hot`.
        offsets_bp (list of int): A list of offsets (in base pairs) for each insertion site.
        orientation_string (str): A string of ">" and "<" characters indicating the orientation of each insertion.

    Returns:
        numpy.ndarray: The modified DNA sequence, in one-hot encoding format, with all insertions included.

    Raises:
        ValueError: If any of the inserted sequences overlap with each other.

    This function takes in a DNA sequence in one-hot encoding format, along with a list of other DNA sequences to be inserted into it.
    The function inserts each of the given sequences into the given sequence at specified locations, according to the given orientation and offset.
    The function then returns the modified DNA sequence in one-hot encoding format.

    If any of the inserted sequences overlap with each other, the function raises a `ValueError` with a message indicating which pairs of sequences overlap.
    """
    assert (
        len(seq_1hot_insertions) == len(orientation_string) == len(offsets_bp)
    ), "insertions, orientations and/or offsets dont match, please check"
    seq_length = seq_1hot.shape[0]
    output_seq = seq_1hot.copy()
    insertion_start_bp = seq_length // 2
    insert_limits = []
    for insertion_index, insertion in enumerate(seq_1hot_insertions):
        insert_bp = len(seq_1hot_insertions[insertion_index])
        insertion_orientation_arrow = orientation_string[insertion_index]
        insertion_offset = offsets_bp[insertion_index]

        if insertion_orientation_arrow == ">":

            insert_limits += [
                (
                    insertion_start_bp + insertion_offset,
                    insertion_start_bp + insertion_offset + insert_bp,
                )
            ]

            output_seq[
                insertion_start_bp
                + insertion_offset : insertion_start_bp
                + insertion_offset
                + insert_bp
            ] = seq_1hot_insertions[insertion_index]
        else:

            insert_limits += [
                (
                    insertion_start_bp + insertion_offset,
                    insertion_start_bp + insertion_offset + insert_bp,
                )
            ]

            output_seq[
                insertion_start_bp
                + insertion_offset : insertion_start_bp
                + insertion_offset
                + insert_bp
            ] = akita_utils.dna_utils.hot1_rc(
                seq_1hot_insertions[insertion_index]
            )

    _check_overlaps(insert_limits)

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


def central_permutation_seqs_gen(seq_coords_df, genome_open, chrom_sizes_table, seq_length=1310720):

    """
    Generate sequences for a given set of coordinates performing central permutations.

    This generator function takes a DataFrame `seq_coords_df` containing genomic
    coordinates (chromosome, start, end, strand), a genome file handler `genome_open`
    to fetch sequences, and a table of chromosome sizes `chrom_sizes_table`. It yields
    sequences with central permutations around the coordinates specified in `seq_coords_df`.

    Parameters:
    - seq_coords_df (pandas.DataFrame): DataFrame with columns 'chrom', 'start', 'end', 'strand'
                                         representing genomic coordinates of interest.
    - genome_open (GenomeFileHandler): A file handler for the genome to fetch sequences.
    - chrom_sizes_table (pandas.DataFrame): DataFrame with columns 'chrom' and 'size' representing
                                            the sizes of chromosomes in the genome.
    - seq_length: int; the length of generated sequence (usually, the standard length of Akita's input).

    Yields:
    numpy.ndarray: One-hot encoded DNA sequences with central permutations around the specified
                   coordinates. The first sequence yielded is the reference, followed by the
                   sequence with a central permutation.

    Raises:
    Exception: If the prediction window for a given span cannot be centered within the chromosome.
    ```
    """
    
    for s in seq_coords_df.itertuples():

        list_1hot = []
        
        chrom, start, end, strand = s.chrom, s.start, s.end, s.strand

        if abs(end - start) % 2 != 0:
            start = start - 1
        
        span_length = abs(end - start)
        length_diff = seq_length - span_length
        
        up_length = down_length = length_diff // 2

        # start and end in genomic coordinates
        up_start = start - up_length
        down_end = end + down_length

        # relative start and end of the span of interest in the prediction window
        relative_start = up_length + 1
        relative_end = relative_start + span_length
        
        # checking if a genomic prediction can be centered around the span
        chr_size = int(chrom_sizes_table[chrom_sizes_table["chrom"] == chrom]["size"])
        if up_start < 0 or down_end > chr_size:
            raise Exception("The prediction window for the following span: ", chrom, start, end, strand, "cannot be centered.")
        
        seq_1hot = dna_1hot(genome_open.fetch(chrom, up_start, down_end).upper())
        if strand == "-":
            seq_1hot = hot1_rc(seq_1hot)
        list_1hot.append(seq_1hot)
        
        permuted_span = permute_seq_k(seq_1hot[relative_start:relative_end], k=1)
        seq_1hot[relative_start:relative_end] = permuted_span
        list_1hot.append(seq_1hot)

        # yielding first the reference, then the permuted sequence
        for seq_1hot in list_1hot:
            yield seq_1hot



