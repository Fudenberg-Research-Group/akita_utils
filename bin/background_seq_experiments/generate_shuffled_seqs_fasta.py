import numpy as np
import pandas as pd
import pysam
import argparse

from akita_utils.dna_utils import dna_1hot, permute_seq_k, dna_1hot_to_seq


import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
log = logging.getLogger(__name__)

################################################################################
# main
# This scripts takes the shuffled seqs tsv and output dir to generate each shuffled seq and store it in fasta format
################################################################################

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', dest='genome_fasta', help='fasta file', required=True)
    parser.add_argument('-tsv', dest='shffled_seqs_tsv', help='tsv with locus specifications', required=True)
    parser.add_argument('-o', dest='output_dir', help='where to store the fasta file', required=True)
    parser.add_argument('-sample', dest='sample', type=int, help='specific number of seqs to sample')
    parser.add_argument('-specific_locus', type=int, dest='specific_locus', help='only sample a specific seq in dataframe')
    
    args = parser.parse_args()
    seqs_df = pd.read_csv(args.shffled_seqs_tsv, sep="\t")
    create_shuffled_seqs(args.genome_fasta, dataframe=seqs_df, output_dir=args.output_dir,sample=args.sample,specific_locus=args.specific_locus)
    
    
def create_shuffled_seqs(
    genome_fasta,
    dataframe,
    output_dir,
    sample=None,
    specific_locus=None,
):
    """This function creates and saves shuffled sequences by permutating experimental sequences

    Args:
        genome_fasta(str) : path to fasta file
        dataframe : dataframe of experimental sequences' parameters
        output_dir : where to store fatsa file
        sample : specific number of seqs to sample (dafaults to whole dataframe if not specified)
        specific_locus : only sample a specific seq in dataframe (defaults to all seqs in dataframe)

    Returns:
        shuffled_seqs : fasta file of shuffled sequences
    """
    num_seqs = dataframe.shape[0]
    genome_open = pysam.Fastafile(genome_fasta)
    
    if sample:
        num_seqs = sample
        
    with open(f'{output_dir}','w') as f:

        for ind in range(num_seqs):
            if specific_locus:
                ind = specific_locus
            locus_specification, shuffle_k, ctcf_thresh, scores_thresh,scores_pixelwise_thresh,GC_content = dataframe.iloc[ind][["locus_specification","shuffle_parameter","ctcf_selection_threshold","map_score_threshold",'scores_pixelwise_thresh','GC_content']]
            chrom, start, end = locus_specification.split(",")
            seq = genome_open.fetch(chrom, int(start), int(end)).upper()
            seq_1hot = dna_1hot(seq)
            shuffled_seq_1hot = permute_seq_k(seq_1hot, k=shuffle_k)

            f.write(f">shuffled:{chrom},{start},{end}#{GC_content} \n")
            f.write(f"{dna_1hot_to_seq(shuffled_seq_1hot)} \n")
            log.info(f"finished saving seq_{ind}")

################################################################################
# __main__
################################################################################
if __name__ == '__main__':
      main()