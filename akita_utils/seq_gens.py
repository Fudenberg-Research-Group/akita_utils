
from akita_utils.dna_utils import hot1_rc, dna_1hot
import numpy as np
import akita_utils.format_io
from akita_utils.program_setup import Locus, Gene, CTCF, create_insertions_sequences
########################################
#           insertion utils            #
########################################


def _insert_casette(
    seq_1hot, seq_1hot_insertion, spacer_bp, orientation_string
):
    """insert an insert (or multiple instances of same insert) into a seq

    Args:
        seq_1hot : seq to be mofied
        seq_1hot_insertion : seq to be inserted
        spacer_bp (int): seperation basepairs
        orientation_string (str): string with orientation of insert(s)

    Returns:
        seq modified with the inserts
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

        for orientation_arrow in orientation_string[i]:
            if orientation_arrow == ">":
                output_seq[offset : offset + insert_bp] = seq_1hot_insertion
            else:
                output_seq[offset : offset + insert_bp] = hot1_rc(
                    seq_1hot_insertion
                )

    return output_seq


def _multi_insert_casette(seq_1hot, seq_1hot_insertions, spacer_bp, orientation_string):
    """insert multiple inserts in the seq at once

    Args:
        seq_1hot : seq to be modified
        seq_1hot_insertions (list): inserts to be inserted in the string
        spacer_bp (int): seperating basepairs between the inserts
        orientation_string (string): string with orientations of the inserts

    Returns:
        modified seq with all the insertions
    """
      
    assert len(seq_1hot_insertions)==len(orientation_string), "insertions dont match orientations, please check"
    seq_length = seq_1hot.shape[0]
    total_insert_bp = sum([len(insertion) for insertion in seq_1hot_insertions])
    num_inserts = len(seq_1hot_insertions)
    inserts_plus_spacer_bp = total_insert_bp + (spacer_bp)*num_inserts + spacer_bp
    insert_start_bp = seq_length // 2 - inserts_plus_spacer_bp // 2
    output_seq = seq_1hot.copy()
    
    length_of_previous_insert = 0
    for i in range(num_inserts):
        insert_bp = len(seq_1hot_insertions[i])
        orientation_arrow = orientation_string[i]
        offset = insert_start_bp + length_of_previous_insert + spacer_bp 
        length_of_previous_insert += len(seq_1hot_insertions[i]) + spacer_bp
        if orientation_arrow == ">":
            output_seq[offset : offset + insert_bp] = seq_1hot_insertions[i]
        else:
            output_seq[offset : offset + insert_bp] = akita_utils.dna_utils.hot1_rc(seq_1hot_insertions[i])
    return output_seq

        
def symmertic_insertion_seqs_gen(seq_coords_df, background_seqs, genome_open):
    """sequence generator for making insertions from tsvs
    construct an iterator that yields a one-hot encoded sequence
    that can be used as input to akita via PredStreamGen

    Args:
        seq_coords_df (dataframe): important colums spacer_bp,locus_orientation,background_seqs,insert_strand,insert_flank_bp,insert_loci 
        background_seqs (fasta): file containing background sequences
        genome_open (opened_fasta): fasta with chrom data

    Yields:
        one-hot encoded sequence: sequence containing specified insertion
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
        

def modular_insertion_seqs_gen(seq_coords_df, background_seqs, genome_open):
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
        spacer_bp = s.spacer_bp
        orientation_string = s.locus_orientation
        seq_1hot = background_seqs[s.background_seqs].copy()        

        for module_number in range(len(s.insert_strand.split("$"))):
            locus = s.insert_loci.split("$")[module_number]
            flank_bp = int(s.insert_flank_bp.split("$")[module_number])
            chrom,start,end = locus.split(",")
            seq_1hot_insertion = akita_utils.dna_utils.dna_1hot(genome_open.fetch(chrom, int(start) - flank_bp, int(end) + flank_bp).upper())
            if s.insert_strand.split("$")[module_number] == "-":
                seq_1hot_insertion = akita_utils.dna_utils.hot1_rc(seq_1hot_insertion)
            seq_1hot_insertions.append(seq_1hot_insertion)

        seq_1hot = _multi_insert_casette(seq_1hot, seq_1hot_insertions, spacer_bp, orientation_string)
        yield seq_1hot


def flexible_flank_modular_insertion_seqs_gen(seq_coords_df, background_seqs, genome_open):
    """ sequence generator for making modular insertions from tsvs with more twist to flanks generation
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
        spacer_bp = s.spacer_bp
        orientation_string = s.locus_orientation
        seq_1hot = background_seqs[s.background_seqs].copy()        
        custom_locus = Locus([CTCF,Gene])
        swapping_flanks = s.swap_flanks  # whether/how we are swapping flanks

        for module_number in range(len(s.insert_strand.split("$"))):
            # figuring out a way to tell if module is a ctcf or gene, currently it is easy to tell from tsv
            locus = s.insert_loci.split("$")[module_number]
            flank_bp = int(s.insert_flank_bp.split("$")[module_number])
            chrom,start,end = locus.split(",")
            strand = s.insert_strand.split("$")[module_number]
            ctcf_score = s.ctcf_genomic_score
            insert = create_insertion(module_number, locus, strand, flank_bp, ctcf_score, swapping_flanks)
            custom_locus.insert(insert)

        seq_1hot_insertions = create_insertions_sequences(custom_locus, genome_open)
        seq_1hot = _multi_insert_casette(seq_1hot, seq_1hot_insertions, spacer_bp, orientation_string)
        yield seq_1hot
        
def create_insertion(module_number, locus, strand, flank_bp, ctcf_score, swapping_flanks=None):
    """
    Creates an Insertion object (either a Gene or a CTCF) based on the given module number,
    locus, strand, and flank_bp values.
    """
    threshold_for_strong_ctcf = 15
    known_strong_ctcf = CTCF("strong","chr11",22206811,22206830,[flank_bp,flank_bp],"+") # default known strong ctcf to pick flanks
    known_weak_ctcf = CTCF("weak","chr4",88777220,88777239,[flank_bp,flank_bp],"+") # default known weak ctcf to pick flanks 
    chrom, start, end = locus.split(",")
    
    if module_number == 0: # gene
        return Gene("unkwown", chrom, int(start), int(end), strand)
    elif module_number == 1: # ctcf
        ctcf = CTCF("unknown", chrom, int(start), int(end), [flank_bp, flank_bp], strand)
        if swapping_flanks=="weak_for_strong" and ctcf_score <= threshold_for_strong_ctcf:
            ctcf.replace_flanks(known_strong_ctcf)
        if swapping_flanks=="strong_for_weak" and ctcf_score > threshold_for_strong_ctcf:
            ctcf.replace_flanks(known_weak_ctcf)
        return ctcf        
        
        
def generate_spans_start_positions(seq_1hot, motif, threshold):
    """get span positions after search for a specified motif and its threshold
    Args:
        seq_1hot (one-hot encoded sequence): one-hot encoded sequence to be scanned
        motif (ndarry): motif to scan for
        threshold (float): threshold to consider

    Returns:
        list: motif found positions
    """
    index_scores_array = akita_utils.dna_utils.scan_motif(seq_1hot, motif)
    motif_window = len(motif)
    half_motif_window = int(np.ceil(motif_window/2))
    spans = []
    for i in np.where(index_scores_array > threshold)[0]:
        if  half_motif_window < i < len(seq_1hot)-half_motif_window:
            spans.append(i)
    return spans

def permute_spans(seq_1hot, spans, motif_window, shuffle_parameter):
    """permute spans of given seq

    Args:
        seq_1hot : one-hot encoded sequence to be scanned
        spans (list): motif found positions
        motif_window (int): length on motif window
        shuffle_parameter (int): basepairs to shuffle by

    Returns:
        one-hot encoded sequence: one-hot encoded sequence with permutated spans
    """
    seq_1hot_mut = seq_1hot.copy()
    half_motif_window = int(np.ceil(motif_window/2))
    for s in spans:
        start, end = s-half_motif_window, s+half_motif_window
        seq_1hot_mut[start:end] = akita_utils.dna_utils.permute_seq_k(seq_1hot_mut[start:end], k=shuffle_parameter)
    return seq_1hot_mut

def mask_spans(seq_1hot, spans, motif_window):
    """mask spans of given seq

    Args:
        seq_1hot : one-hot encoded sequence to be scanned
        spans (list): motif found positions
        motif_window (int): length on motif window

    Returns:
       one-hot encoded sequence: one-hot encoded sequence with masked spans
    """
    seq_1hot_perm = seq_1hot.copy()
    half_motif_window = int(np.ceil(motif_window/2))
    for s in spans:
        start, end = s-half_motif_window, s+half_motif_window
        seq_1hot_perm[start : end, :] = 0
    return seq_1hot_perm

def randomise_spans(seq_1hot, spans, motif_window):
    """randomise spans of given seq

    Args:
        seq_1hot : one-hot encoded sequence to be scanned
        spans (list): motif found positions
        motif_window (int): length on motif window

    Returns:
       one-hot encoded sequence: one-hot encoded sequence with randomised spans
    """
    seq_1hot_perm = seq_1hot.copy()
    half_motif_window = int(np.ceil(motif_window/2))
    for s in spans:
        start, end = s-half_motif_window, s+half_motif_window
        seq_1hot_perm[start : end] = random_seq_permutation(seq_1hot_perm[start : end])
    return seq_1hot_perm

def random_seq_permutation(seq_1hot):
    """randomise a given seq

    Args:
        seq_1hot : one-hot encoded sequence to be randomised

    Returns:
        one-hot encoded sequence: randomised one-hot encoded sequence
    """
    seq_1hot_perm = seq_1hot.copy()
    random_inds = np.random.permutation(range(len(seq_1hot)))
    seq_1hot_perm = seq_1hot[random_inds, :].copy()
    return seq_1hot_perm

def background_exploration_seqs_gen(seq_coords_df, genome_open):
    """unpacks the given dataframe and creats an iterator for a sequence from each row generated by specified values in the columns

    Args:
        seq_coords_df (dataframe): important colums spacer_bp,locus_orientation,background_seqs,insert_strand,insert_flank_bp,insert_loci
        genome_open (opened_fasta): fasta with chrom data

    Yields:
        seq_1hot generator : one-hot encoded sequences
    """

    motif = akita_utils.format_io.read_jaspar_to_numpy()
    motif_window = len(motif)-3 #for compartibility ie (19-3=16 which is a multiple of 2,4,8 the shuffle parameters)
    for s in seq_coords_df.itertuples():
        chrom,start,end = s.locus_specification.split(",") # Split the locus specification into chromosome, start, and end coordinates
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
        else:
            raise ValueError(f"Unrecognized parameters, check your dataframe")

########################################
#           deletion utils             #
########################################
