
import pandas as pd
import akita_utils

########################################
#           insertion utils            #
########################################

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
                output_seq[offset : offset + insert_bp] = dna_io.hot1_rc(seq_1hot_insertion)

    return output_seq


def _multi_insert_casette(seq_1hot, seq_1hot_insertions, spacer_bp, orientation_string):
        
    assert len(seq_1hot_insertions)==len(orientation_string), "insertions dont match orientations, please check"
    seq_length = seq_1hot.shape[0]
    total_insert_bp = 0
    for insertion in seq_1hot_insertions:
        total_insert_bp += len(insertion)
        
    num_inserts = len(seq_1hot_insertions)
    inserts_plus_spacer_bp = total_insert_bp + (2 * spacer_bp)*num_inserts
    insert_start_bp = seq_length // 2 - inserts_plus_spacer_bp // 2
    output_seq = seq_1hot.copy()
    
    insertion_starting_positions = []
    
    for i in range(num_inserts):
        insert_bp = len(seq_1hot_insertions[i])
        offset = insert_start_bp + i * inserts_plus_spacer_bp + spacer_bp
        insertion_starting_positions.append(offset)
        for orientation_arrow in orientation_string[i]:
            if orientation_arrow == ">":
                output_seq[offset : offset + insert_bp] = seq_1hot_insertions[i]
            else:
                output_seq[offset : offset + insert_bp] = dna_io.hot1_rc(seq_1hot_insertions[i])
    # print(insertion_starting_positions)
    return output_seq

        
def symmertic_insertion_seqs_gen(seq_coords_df, background_seqs, genome_open):
    """ sequence generator for making insertions from tsvs
        construct an iterator that yields a one-hot encoded sequence
        that can be used as input to akita via PredStreamGen
    """
        
    list_seq_1hot = []
    
    for s in seq_coords_df.itertuples():
        
        flank_bp = s.flank_bp
        spacer_bp = s.spacer_bp
        orientation_string = s.orientation
                
        seq_1hot_insertion = dna_io.dna_1hot(
            genome_open.fetch(s.chrom, s.start - flank_bp, s.end + flank_bp).upper()
        )
        
        if s.strand == "-":
            seq_1hot_insertion = dna_io.hot1_rc(seq_1hot_insertion)
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
        flank_bp = s.flank_bp
        spacer_bp = s.spacer_bp
        orientation_string = s.orientation
        seq_1hot = background_seqs[s.background_index].copy()        

        for module_number in range(len(s.chrom.split("#"))):
            # print(type(s.start))
            
            seq_1hot_insertion = akita_utils.dna_utils.dna_1hot(
                genome_open.fetch(s.chrom.split("#")[module_number], int(s.start.split("#")[module_number]) - flank_bp, int(s.end.split("#")[module_number]) + flank_bp).upper()
            )
            if s.strand.split("#")[module_number] == "-":
                seq_1hot_insertion = akita_utils.dna_utils.hot1_rc(seq_1hot_insertion)
            seq_1hot_insertions.append(seq_1hot_insertion)

        seq_1hot = _multi_insert_casette(seq_1hot, seq_1hot_insertions, spacer_bp, orientation_string)

        yield seq_1hot
        
########################################
#           deletion utils             #
########################################