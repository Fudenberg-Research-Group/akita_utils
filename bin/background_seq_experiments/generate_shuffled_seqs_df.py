# import general libraries
import os
import itertools
import pandas as pd
import numpy as np
import bioframe
import argparse
import akita_utils.tsv_gen_utils 

            
################################################################################
'''
This script generates and saves a dataframe specifying sequence chrom, start, end, along with how they should be shuffled. The saved dataframe can then be used as (i) input to scripts that save shuffled fasta files, or (ii) as an input to shuffle sequences on the fly to generate respective scores.

The inputs to this script are:
(1) fasta file of appropriate genome. 
(2) set of intervals to prepare for shuffling, in the bed format. 

Other parameters include:

'shuffle_parameter', which specifies the kmer size to shuffle by
'ctcf_selection_threshold', which specifies the accuracy in idenfying motifs from a seq
'mutation_method', which can be any of ['permute_whole_seq','randomise_whole_seq','mask_motif','permute_motif','randomise_motif']
'output_filename', which is the name of the output tsv file
'num_sites', which is sample size
'mode', locus GC content selection criteria which maybe 'flat', 'tail', 'head', 'random'


The output dataframe has all possible combinations of the provided parameters.

'''
################################################################################
    
# ---------------- typical arguments for choice of bed (mouse or human) ------------------------
# these are bed files with intervals the models were trained on.

# seq_bed_file = '/project/fudenber_735/tensorflow_models/akita/v2/data/mm10/sequences.bed' #mouse
# genome_fasta = '/project/fudenber_735/genomes/mm10/mm10.fa' #mouse
# seq_bed_file = '/project/fudenber_735/tensorflow_models/akita/v2/data/hg38/sequences.bed' #human
# genome_fasta = '/project/fudenber_735/genomes/hg38/hg38.fa'#human
# -------------------------------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', dest='genome_fasta', help='fasta file', required=True)
    parser.add_argument('-seq_bed_file', dest='seq_bed_file', help='bed file for the seqs under investigation', required=True)
    parser.add_argument('--output_filename', dest='output_filename', default='data/shuffled_seqs.tsv', help='output_filename')
    parser.add_argument('--shuffle_parameter', nargs='+', default=[8], type=int, help='List of integers')
    parser.add_argument('--ctcf_selection_threshold', default=[8], nargs='+', type=int, help='List of integers')
    parser.add_argument('--mutation_method', nargs='+', default=['permute_whole_seq'], help='List of strings')
    parser.add_argument('--num_sites', type=int, default=20, help='number of loci to select')
    parser.add_argument('--mode', default='uniform', help='loci selection criteria') 
    args = parser.parse_args()
    
    # prepare dataframe with chromosomes and calculate GC content(using bioframe)
    seq_df = pd.read_csv(args.seq_bed_file, sep='\t', header=None, names=['chrom', 'start', 'end', 'fold'])
    general_seq_gc_df = bioframe.frac_gc(seq_df, bioframe.load_fasta(args.genome_fasta), return_input=True)
    
    #-------------------------------------------------------------------------------------------------
    # INSTRUCTIONS TO FILL IN PARAMETERS BELOW
    # Provide as many appropriate parameters as possible.
    # To generate multiple maps, provide individual parameters in list format, [first_value, second_value, third_value, etc]
    
    grid_search_params = {
        'shuffle_parameter': args.shuffle_parameter,
        'ctcf_selection_threshold': args.ctcf_selection_threshold,
        'mutation_method': args.mutation_method,
    }
    
    # sampling seq_df dataframe respecting GC content
    seq_gc_df = akita_utils.tsv_gen_utils.filter_sites_by_score(general_seq_gc_df,score_key="GC",upper_threshold=99,lower_threshold=1,mode=args.mode,num_sites=args.num_sites) 
    
    # fixing locus specific chacteristics together before grid_search
    seq_gc_df = seq_gc_df["chrom"].map(str) +","+ seq_gc_df["start"].map(str) + "," + seq_gc_df["end"].map(str)+"-"+seq_gc_df["GC"].map(str)
    locus_list = seq_gc_df.values.tolist()
    
    grid_search_params['locus_specification']= locus_list

    grid_search_param_set = list(itertools.product(*[v for v in grid_search_params.values()]))
    parameters_combo_dataframe = pd.DataFrame(grid_search_param_set, columns=grid_search_params.keys())
    parameters_combo_dataframe[["locus_specification",'GC_content']] = parameters_combo_dataframe["locus_specification"].str.split('-',expand=True)
    
    # os.makedirs(, exist_ok=True)
    parameters_combo_dataframe.to_csv(f'{args.output_filename}', sep='\t', index=False)
            
################################################################################
# __main__
################################################################################

if __name__ == "__main__":
    main()