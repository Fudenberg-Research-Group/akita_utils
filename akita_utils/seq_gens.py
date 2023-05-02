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
        

def modular_insertion_seqs_gen(seq_coords_df, background_seqs, genome_open):
    """ sequence generator for making modular insertions from tsvs
        construct an iterator that yields a one-hot encoded sequence
        that can be used as input to akita via PredStreamGen
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

# define sequence generator
def mask_central_seq(seq_1hot, motif_width=20):
    seq_1hot_perm = seq_1hot.copy()
    mask_inds = np.arange(
        seq_length // 2 - motif_width // 2, seq_length // 2 + motif_width // 2
    )
    seq_1hot_perm[mask_inds, :] = 0
    return seq_1hot_perm

def permute_central_seq(seq_1hot, motif_width=20):
    seq_1hot_perm = seq_1hot.copy()
    central_inds = np.arange(
        seq_length // 2 - motif_width // 2, seq_length // 2 + motif_width // 2
    )
    mask_inds = np.random.permutation(central_inds)
    seq_1hot_perm[mask_inds, :] = seq_1hot[central_inds, :].copy()
    return seq_1hot_perm

def mask_spans(seq_1hot, spans):
    seq_1hot_perm = seq_1hot.copy()
    for s in spans:
        seq_1hot_perm[s[0] : s[1], :] = 0
    return seq_1hot_perm

def permute_spans(seq_1hot, spans):
    seq_1hot_perm = seq_1hot.copy()
    spans_flat = np.array([]).astype(int)
    for s in spans:
        spans_flat = np.hstack((spans_flat, np.arange(s[0], s[1])))
    spans_permuted = np.random.permutation(spans_flat)
    seq_1hot_perm[spans_permuted, :] = seq_1hot[spans_flat, :].copy()
    return seq_1hot_perm

def split_span(span_string):
    spans = []
    for j in span_string.split(","):
        spans.append([int(j.split("-")[0]), int(j.split("-")[1])])
    return spans

def fetch_centered_padded_seq(chrom, start, end):
    mid = (start + end) // 2
    start_centered, end_centered = int(mid - seq_length // 2), int(
        mid + seq_length // 2
    )
    seq_dna = genome_open.fetch(chrom, start_centered, end_centered).upper()
    chromosome_length = genome_open.get_reference_length(chrom)
    pad_upstream = "N" * max(-start_centered, 0)
    pad_downstream = "N" * max(end_centered - chromosome_length, 0)
    return pad_upstream + seq_dna + pad_downstream
    def seqs_gen(motif_width):
        for s in seq_coords_df.itertuples():
            list_1hot = []
            seq_dna = fetch_centered_padded_seq(s.chrom, s.start, s.end)
            wt_1hot = dna_io.dna_1hot(seq_dna)
            list_1hot.append(wt_1hot)

            if use_span:
                spans = split_span(s.span)
                spans = np.array(spans) - start
                if mutation_method == "mask":
                    list_1hot.append(mask_spans(wt_1hot, spans))
                elif mutation_method == "permute":
                    list_1hot.append(permute_spans(wt_1hot, spans))

            else:  ## just mask central motif
                if mutation_method == "mask":
                    list_1hot.append(mask_central_seq(wt_1hot, motif_width=motif_width))
                elif mutation_method == "permute":
                    list_1hot.append(
                        permute_central_seq(wt_1hot, motif_width=motif_width)
                    )

            for seq_1hot in list_1hot:
                yield seq_1hot
                
def disruption_seqs_gen(motif_width):
    for s in seq_coords_df.itertuples():
        list_1hot = []
        seq_dna = fetch_centered_padded_seq(s.chrom, s.start, s.end)
        wt_1hot = dna_io.dna_1hot(seq_dna)
        list_1hot.append(wt_1hot)

        if use_span:
            spans = split_span(s.span)
            spans = np.array(spans) - start
            if mutation_method == "mask":
                list_1hot.append(mask_spans(wt_1hot, spans))
            elif mutation_method == "permute":
                list_1hot.append(permute_spans(wt_1hot, spans))

        else:  ## just mask central motif
            if mutation_method == "mask":
                list_1hot.append(mask_central_seq(wt_1hot, motif_width=motif_width))
            elif mutation_method == "permute":
                list_1hot.append(
                    permute_central_seq(wt_1hot, motif_width=motif_width)
                )

        for seq_1hot in list_1hot:
            yield seq_1hot
########################################
#           deletion utils             #
########################################