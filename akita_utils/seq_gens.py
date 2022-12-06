
import pandas as pd
from basenji import dna_io

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

        
########################################
#           deletion utils             #
########################################