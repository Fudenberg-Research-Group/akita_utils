from akita_utils.dna_utils import hot1_rc, dna_1hot
import numpy as np
import akita_utils.format_io
########################################
#           insertion utils            #
########################################


def _insert_casette(
    seq_1hot, seq_1hot_insertion, spacer_bp, orientation_string
):

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
                output_seq[offset : offset + insert_bp] = hot1_rc(
                    seq_1hot_insertion
                )

    return output_seq


def _multi_insert_casette(seq_1hot, seq_1hot_insertions, spacer_bp, orientation_string):
        
    assert len(seq_1hot_insertions)==len(orientation_string), "insertions dont match orientations, please check"
    seq_length = seq_1hot.shape[0]
    total_insert_bp = sum([len(insertion) for insertion in seq_1hot_insertions])
    num_inserts = len(seq_1hot_insertions)
    inserts_plus_spacer_bp = total_insert_bp + (2 * spacer_bp)*num_inserts
    insert_start_bp = seq_length // 2 - inserts_plus_spacer_bp // 2
    output_seq = seq_1hot.copy()
    
    length_of_previous_insert = 0
    for i in range(num_inserts):
        insert_bp = len(seq_1hot_insertions[i])
        orientation_arrow = orientation_string[i]
        offset = insert_start_bp + length_of_previous_insert + spacer_bp # i * inserts_plus_spacer_bp + spacer_bp
        length_of_previous_insert += len(seq_1hot_insertions[i]) + 2*spacer_bp
        if orientation_arrow == ">":
            output_seq[offset : offset + insert_bp] = seq_1hot_insertions[i]
        else:
            output_seq[offset : offset + insert_bp] = akita_utils.dna_utils.hot1_rc(seq_1hot_insertions[i])
    return output_seq

def _multi_insert_offsets_casette(seq_1hot, seq_1hot_insertions, offsets_bp, orientation_string):
    """insert multiple inserts in the seq at once

    Args:
        seq_1hot : seq to be modified
        seq_1hot_insertions (list): inserts to be inserted in the string
        spacer_bp (int): seperating basepairs between the inserts
        orientation_string (string): string with orientations of the inserts

    Returns:
        modified seq with all the insertions
    """
    
    assert len(seq_1hot_insertions)==len(orientation_string)==len(offsets_bp), "insertions, orientations and/or offsets dont match, please check"
    seq_length = seq_1hot.shape[0]
    output_seq = seq_1hot.copy()
    insertion_start_bp = seq_length // 2
    for insertion_index, insertion in enumerate(seq_1hot_insertions):
        insert_bp = len(seq_1hot_insertions[insertion_index])
        insertion_orientation_arrow = orientation_string[insertion_index]
        insertion_offset = offsets_bp[insertion_index]
        
        if insertion_orientation_arrow == ">":
            output_seq[insertion_start_bp+insertion_offset : insertion_start_bp+insertion_offset+insert_bp] = seq_1hot_insertions[insertion_index]
        else:
            output_seq[insertion_start_bp+insertion_offset : insertion_start_bp+insertion_offset+insert_bp] = akita_utils.dna_utils.hot1_rc(seq_1hot_insertions[insertion_index])
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
            genome_open.fetch(
                s.chrom, s.start - flank_bp, s.end + flank_bp
            ).upper()
        )

        if s.strand == "-":
            seq_1hot_insertion = hot1_rc(seq_1hot_insertion)
            # now, all motifs are standarized to this orientation ">"

        seq_1hot = background_seqs[s.background_index].copy()
        
        seq_1hot = _insert_casette(seq_1hot, seq_1hot_insertion, spacer_bp, orientation_string)
        
        yield seq_1hot
        


        
def modular_offsets_insertion_seqs_gen(seq_coords_df, background_seqs, genome_open):# delimiter specification
    """ sequence generator for making modular insertions from tsvs
        yields a one-hot encoded sequence
        that can be used as input to akita via PredStreamGen

    Args:
        seq_coords_df (dataframe): important colums spacer_bp,locus_orientation,background_seqs,insert_strand,insert_flank_bp,insert_loci 
        background_seqs (fasta): file containing background sequences
        genome_open (opened_fasta): fasta with chrom data

    Yields:
        one-hot encoded sequence: sequence containing specified insertions
    """
    
    for s in seq_coords_df.itertuples():
        seq_1hot_insertions = []
        offsets_bp = []
        orientation_string = [] # s.locus_orientation
        seq_1hot = background_seqs[s.background_seqs].copy()        

        for module_number in range(len(s.insert_loci.split("$"))):
            locus_specification = s.insert_loci.split("$")[module_number]
            if locus_specification != "None":
                chrom,start,end,strand = locus_specification.split(",")
                insert_offset_bp = int(s.insert_offsets.split("$")[module_number])
                insert_orientation = s.insert_orientations.split("$")[module_number]
                 
                flank_bp = int(s.insert_flank_bp.split("$")[module_number])
                seq_1hot_insertion = akita_utils.dna_utils.dna_1hot(genome_open.fetch(chrom, int(start) - flank_bp, int(end) + flank_bp).upper())
                if strand == "-":
                    seq_1hot_insertion = akita_utils.dna_utils.hot1_rc(seq_1hot_insertion)
                
#                 if s.gene_locus_specification is not np.nan:
#                     if s.num_of_motifs > 5:
#                         motif = akita_utils.format_io.read_jaspar_to_numpy()
#                         motif_window = len(motif)-3 #for compartibility ie (19-3=16 which is a multiple of 2,4,8 the shuffle parameters)
#                         spans = generate_spans_start_positions(seq_1hot_insertion, motif, 8)
#                         seq_1hot_insertion = mask_spans(seq_1hot_insertion, spans, motif_window)
                
                seq_1hot_insertions.append(seq_1hot_insertion)
                offsets_bp.append(insert_offset_bp)
                orientation_string.append(insert_orientation)

        seq_1hot = _multi_insert_offsets_casette(seq_1hot, seq_1hot_insertions, offsets_bp, orientation_string)
        yield seq_1hot
        
        


# define sequence generator
def generate_spans_start_positions(seq_1hot, motif, threshold):
    index_scores_array = akita_utils.dna_utils.scan_motif(seq_1hot, motif)
    motif_window = len(motif)
    half_motif_window = int(np.ceil(motif_window/2))
    spans = []
    for i in np.where(index_scores_array > threshold)[0]:
        if  half_motif_window < i < len(seq_1hot)-half_motif_window:
            spans.append(i)
    return spans

def permute_spans(seq_1hot, spans, motif_window, shuffle_parameter):
    seq_1hot_mut = seq_1hot.copy()
    half_motif_window = int(np.ceil(motif_window/2))
    for s in spans:
        start, end = s-half_motif_window, s+half_motif_window
        seq_1hot_mut[start:end] = akita_utils.dna_utils.permute_seq_k(seq_1hot_mut[start:end], k=shuffle_parameter)
    return seq_1hot_mut

def mask_spans(seq_1hot, spans, motif_window):
    seq_1hot_perm = seq_1hot.copy()
    half_motif_window = int(np.ceil(motif_window/2))
    for s in spans:
        start, end = s-half_motif_window, s+half_motif_window
        seq_1hot_perm[start : end, :] = 0
    return seq_1hot_perm

def randomise_spans(seq_1hot, spans, motif_window):
    seq_1hot_perm = seq_1hot.copy()
    half_motif_window = int(np.ceil(motif_window/2))
    for s in spans:
        start, end = s-half_motif_window, s+half_motif_window
        seq_1hot_perm[start : end] = random_seq_permutation(seq_1hot_perm[start : end])
    return seq_1hot_perm

def random_seq_permutation(seq_1hot):
    seq_1hot_perm = seq_1hot.copy()
    random_inds = np.random.permutation(range(len(seq_1hot)))
    seq_1hot_perm = seq_1hot[random_inds, :].copy()
    return seq_1hot_perm

def background_exploration_seqs_gen(seq_coords_df, genome_open, use_span=True):
    motif = akita_utils.format_io.read_jaspar_to_numpy()
    motif_window = len(motif)-3 #for compartibility ie (19-3=16 which is a multiple of 2,4,8 the shuffle parameters)
    for s in seq_coords_df.itertuples():
        chrom,start,end = s.locus_specification.split(",")
        seq_dna = genome_open.fetch(chrom, int(start), int(end))
        wt_1hot = akita_utils.dna_utils.dna_1hot(seq_dna)
        mutation_method = s.mutation_method
        spans = generate_spans_start_positions(wt_1hot, motif, s.ctcf_selection_threshold)
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


########################################
#           deletion utils             #
########################################