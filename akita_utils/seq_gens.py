from akita_utils.dna_utils import hot1_rc, dna_1hot
import numpy as np
import akita_utils.format_io
import pandas as pd

def _insert_casette(seq_1hot, seq_1hot_insertion, spacer_bp, orientation_string):
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

        for orientation_arrow in orientation_string[i]:
            if orientation_arrow == ">":
                output_seq[offset : offset + insert_bp] = seq_1hot_insertion
            else:
                output_seq[offset : offset + insert_bp] = hot1_rc(seq_1hot_insertion)

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
    """
    assert (
        len(seq_1hot_insertions) == len(orientation_string) == len(offsets_bp)
    ), "insertions, orientations and/or offsets dont match, please check"
    seq_length = seq_1hot.shape[0]
    output_seq = seq_1hot.copy()
    insertion_start_bp = seq_length // 2
    for insertion_index, insertion in enumerate(seq_1hot_insertions):
        insert_bp = len(seq_1hot_insertions[insertion_index])
        insertion_orientation_arrow = orientation_string[insertion_index]
        insertion_offset = offsets_bp[insertion_index]

        if insertion_orientation_arrow == ">":
            output_seq[
                insertion_start_bp
                + insertion_offset : insertion_start_bp
                + insertion_offset
                + insert_bp
            ] = seq_1hot_insertions[insertion_index]
        else:
            output_seq[
                insertion_start_bp
                + insertion_offset : insertion_start_bp
                + insertion_offset
                + insert_bp
            ] = akita_utils.dna_utils.hot1_rc(seq_1hot_insertions[insertion_index])
                
    return output_seq



def symmertic_insertion_seqs_gen(seq_coords_df, background_seqs, genome_open):
    """sequence generator for making insertions from tsvs
    construct an iterator that yields a one-hot encoded sequence
    that can be used as input to akita via PredStreamGen
    """

    for s in seq_coords_df.itertuples():
        flank_bp = s.flank_bp
        spacer_bp = s.spacer_bp
        orientation_string = s.orientation

        seq_1hot_insertion = dna_1hot(
            genome_open.fetch(s.chrom, s.start - flank_bp, s.end + flank_bp).upper()
        )

        if s.strand == "-":
            seq_1hot_insertion = hot1_rc(seq_1hot_insertion)
            # now, all motifs are standarized to this orientation ">"

        seq_1hot = background_seqs[s.background_index].copy()

        seq_1hot = _insert_casette(
            seq_1hot, seq_1hot_insertion, spacer_bp, orientation_string
        )

        yield seq_1hot

        
# define sequence generator
def generate_spans_start_positions(seq_1hot, motif, threshold):
    index_scores_array = akita_utils.dna_utils.scan_motif(seq_1hot, motif)
    motif_window = len(motif)
    half_motif_window = int(np.ceil(motif_window / 2))
    spans = []
    for i in np.where(index_scores_array > threshold)[0]:
        if half_motif_window < i < len(seq_1hot) - half_motif_window:
            spans.append(i)
    return spans


def permute_spans(seq_1hot, spans, motif_window, shuffle_parameter):
    seq_1hot_mut = seq_1hot.copy()
    half_motif_window = int(np.ceil(motif_window / 2))
    for s in spans:
        start, end = s - half_motif_window, s + half_motif_window
        seq_1hot_mut[start:end] = akita_utils.dna_utils.permute_seq_k(
            seq_1hot_mut[start:end], k=shuffle_parameter
        )
    return seq_1hot_mut


def mask_spans(seq_1hot, spans, motif_window):
    seq_1hot_perm = seq_1hot.copy()
    half_motif_window = int(np.ceil(motif_window / 2))
    for s in spans:
        start, end = s - half_motif_window, s + half_motif_window
        seq_1hot_perm[start:end, :] = 0
    return seq_1hot_perm


def randomise_spans(seq_1hot, spans, motif_window):
    seq_1hot_perm = seq_1hot.copy()
    half_motif_window = int(np.ceil(motif_window / 2))
    for s in spans:
        start, end = s - half_motif_window, s + half_motif_window
        seq_1hot_perm[start:end] = random_seq_permutation(seq_1hot_perm[start:end])
    return seq_1hot_perm


def random_seq_permutation(seq_1hot):
    seq_1hot_perm = seq_1hot.copy()
    random_inds = np.random.permutation(range(len(seq_1hot)))
    seq_1hot_perm = seq_1hot[random_inds, :].copy()
    return seq_1hot_perm


def background_exploration_seqs_gen(seq_coords_df, genome_open, use_span=True):
    """
    Generate mutated sequences based on a reference genome and a set of mutation specifications.

    Args:
        seq_coords_df (pandas.DataFrame): A pandas DataFrame with one row per mutation specification.
            The DataFrame should have columns `locus_specification`, `mutation_method`, `ctcf_selection_threshold`,
            and `shuffle_parameter`. The `locus_specification` column should have values in the format
            "chrom,start,end". The `mutation_method` column should have values among "mask_motif", "permute_motif",
            "randomise_motif", "permute_whole_seq", and "randomise_whole_seq". The `ctcf_detection_threshold` column
            should have a float value, and is used only when `mutation_method` is "mask_motif" or "permute_motif". The
            `shuffle_parameter` column should have a positive integer value i.e 2 4 8, and is used only when `mutation_method`
            is "permute_motif" or "permute_whole_seq".
        genome_open (pysam.FastaFile): A `pysam.FastaFile` object representing the reference genome.
        use_span (bool, optional): Whether to restrict the mutation operations to the regions where a motif is found
            (True) or apply them to the entire sequence (False). Defaults to True.

    Yields:
        numpy.ndarray: A mutated sequence in one-hot encoding format, as a numpy array with shape (n_positions, 4).

    Raises:
        ValueError: If any of the values in `seq_coords_df` is invalid or if any of the mutation methods is not
            recognized.
    """
    motif = akita_utils.format_io.read_jaspar_to_numpy()
    # motif_window = (
    #     len(motif) - 3
    # )  # for compartibility ie (19-3=16 which is a multiple of 2,4,8 the shuffle parameters)
    motif_window = 2 ** (math.ceil(math.log2(len(motif) - 1)))
    for s in seq_coords_df.itertuples():
        chrom, start, end = s.locus_specification.split(",")
        seq_dna = genome_open.fetch(chrom, int(start), int(end))
        wt_1hot = akita_utils.dna_utils.dna_1hot(seq_dna)
        mutation_method = s.mutation_method
        spans = generate_spans_start_positions(
            wt_1hot, motif, s.ctcf_detection_threshold
        )
        if mutation_method == "mask_motif":
            yield mask_spans(wt_1hot, spans, motif_window)
        elif mutation_method == "permute_motif":
            yield permute_spans(wt_1hot, spans, motif_window, s.shuffle_parameter)
        elif mutation_method == "randomise_motif":
            yield randomise_spans(wt_1hot, spans, motif_window)
        elif mutation_method == "permute_whole_seq":
            yield akita_utils.dna_utils.permute_seq_k(wt_1hot, k=s.shuffle_parameter)
        elif mutation_method == "randomise_whole_seq":
            yield random_seq_permutation(wt_1hot)


# -----------------------------modifying this function-----------------------------


def modular_offsets_insertion_seqs_gen(seq_coords_df, background_seqs, genome_open):
    """
    Generate modified DNA sequences by inserting variable-length DNA segments into specified locations in a background sequence.

    Args:
        seq_coords_df (pandas DataFrame): A DataFrame with one row per desired sequence modification, and columns specifying the location and orientation of each insertion.
            Each row should contain the following columns:
            - 'background_seqs': the index of the background sequence to modify (must match the index of a row in the background_seqs DataFrame).
            - One or more columns with names in the format 'insert_{i}', where 'i' is an integer (starting from 1) and the remaining values specify the location and orientation of the DNA segment to insert.
              The values in each 'insert' column should be separated by the '$' delimiter, and should have the following format:
              - chrom: chromosome name (string)
              - start: 1-based start position of the DNA segment to insert (integer)
              - end: 1-based end position of the DNA segment to insert (integer)
              - strand: '+' or '-' to indicate the orientation of the DNA segment relative to the reference genome.
        background_seqs (pandas DataFrame): A DataFrame with the DNA sequences to be modified, indexed by integer values matching the 'background_seqs' column in seq_coords_df.
        genome_open (pysam.FastaFile): An open pysam FastaFile object representing the reference genome. Used to retrieve the DNA sequences to be inserted.

    Yields:
        numpy.ndarray: A 4D numpy array with shape (1, seq_length, 4, 1), representing the modified DNA sequence.
        seq_length is the length of the modified sequence in base pairs, and the final axis is always 1.

    Raises:
        ValueError: If any 'insert' column in seq_coords_df has an invalid format or refers to a region outside the boundaries of the reference genome.

    """
    
    for experiment_index in seq_coords_df.itertuples(index=False):
        
        ref_and_pred =[]
        
        seq_1hot = background_seqs[experiment_index.background_seqs].copy()
        ref_and_pred += [seq_1hot]
        
        seq_1hot_insertions = []
        offsets_bp = []
        orientation_string = []
        experiment_index_df = pd.DataFrame([experiment_index], columns=seq_coords_df.columns.to_list())

        for col_name in seq_coords_df.columns:
            
            if "insert" in col_name:
                
                (   chrom, start, end, strand,
                    insert_flank_bp,
                    insert_offset,
                    insert_orientation,
                ) = experiment_index_df[col_name][0].split(",")
                seq_1hot_insertion = akita_utils.dna_utils.dna_1hot(
                    genome_open.fetch(
                        chrom,
                        int(start) - int(insert_flank_bp),
                        int(end) + int(insert_flank_bp),
                    ).upper()
                )

                if strand == "-":
                    seq_1hot_insertion = akita_utils.dna_utils.hot1_rc(
                        seq_1hot_insertion
                    )

                seq_1hot_insertions.append(seq_1hot_insertion)
                offsets_bp.append(int(insert_offset))
                orientation_string.append(insert_orientation)

        seq_1hot = _multi_insert_offsets_casette(
            seq_1hot, seq_1hot_insertions, offsets_bp, orientation_string
        )
        ref_and_pred += [seq_1hot]
        
        for seq_1hot in ref_and_pred:
            yield seq_1hot

        
def _inserts_overlap_check_pre_simulation(dataframe):
    
    for experiment in dataframe.itertuples(index=False):
        offsets_bp = []
        insertions_length_list = []
        insertions_name_list = []
        experiment_df = pd.DataFrame([experiment], columns=dataframe.columns.to_list())
        for col_name in dataframe.columns:
            if "insert" in col_name:

                (   chrom, start, end, strand,
                    insert_flank_bp,
                    insert_offset,
                    insert_orientation,
                ) = experiment_df[col_name][0].split(",")

                offsets_bp.append(int(insert_offset))
                insertions_length_list.append(int(end)-int(start)+1+2*int(insert_flank_bp))
                insertions_name_list.append(col_name)
        _check_overlaps_pre_simulation(insertions_length_list, offsets_bp, insertions_name_list)
                

def _check_overlaps_pre_simulation(insertions_length_list, offsets_bp, insertions_name_list):
        assert (
            len(insertions_length_list) == len(offsets_bp)
        ), "insertions and offsets dont match, please check"
        insertion_start_bp = 0
        insert_limits = []
        
        for insertion_index, insertion_length in enumerate(insertions_length_list):
            
            insert_bp = insertion_length
            insertion_offset = offsets_bp[insertion_index]
            insert_limits += [(
                insertion_start_bp
                + insertion_offset , insertion_start_bp
                + insertion_offset
                + insert_bp
            )]
        _check_overlaps(insert_limits, insertions_name_list)
            
            
def _check_overlaps(insert_limits, insertions_name_list=None):
    sorted_insert_limits = sorted(zip(insert_limits, insertions_name_list))
    sorted_insertions_name_list = [name for _, name in sorted_insert_limits]
    sorted_insert_limits = [limits for limits, _ in sorted_insert_limits]
    for i in range(len(sorted_insert_limits) - 1):
        if sorted_insert_limits[i][1] > sorted_insert_limits[i+1][0]:
            raise ValueError(f"Overlap found between inserted sequences: {sorted_insertions_name_list[i]} --> {sorted_insert_limits[i]}, {sorted_insertions_name_list[i+1]} --> {sorted_insert_limits[i+1]}")
