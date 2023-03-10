mport general libraries
import os
import itertools
import pandas as pd
import numpy as np
import bioframe
import argparse
from akita_utils.tsv_gen_utils import calculate_GC, filter_sites_by_score  # reminder. this function needs to be renamed to a more general name (filter dataframe by key)

            
################################################################################
# __main__
# This scripts generates shuffled seqs with specified parameters, generally used to search for appropriate parameters later used in creating flat seqs
################################################################################
    
# --------------------------on boarding(become args) choose mouse or human ------------------------
# chrom_seq_bed_file = '/project/fudenber_735/tensorflow_models/akita/v2/data/mm10/sequences.bed' #mouse
# genome_fasta = '/project/fudenber_735/genomes/mm10/mm10.fa' #mouse
# chrom_seq_bed_file = '/project/fudenber_735/tensorflow_models/akita/v2/data/hg38/sequences.bed' #human
# genome_fasta = '/project/fudenber_735/genomes/hg38/hg38.fa'#human
# -------------------------------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', dest='genome_fasta', help='fasta file', required=True)
    parser.add_argument('-chrom_seq_bed_file', dest='chrom_seq_bed_file', help='bed file for the seqs under investigation', required=True)
    args = parser.parse_args()
    
    # prepare dataframe for desired chromosomes and calculate GC content(using bioframe)
    dataframe = calculate_GC(args.chrom_seq_bed_file,args.genome_fasta)
    
    # carefull with parameters here, sample selection
    new_dataframe = filter_sites_by_score(dataframe,score_key="GC",upper_threshold=99,lower_threshold=1,mode="flat",num_sites=20) # seperate pull request
    
    
    # polishing the dataframe to fit expected input
    new_dataframe = new_dataframe["chrom"].map(str) +","+ new_dataframe["start"].map(str) + "," + new_dataframe["end"].map(str)+"-"+new_dataframe["GC"].map(str)
    locus_specification_list = new_dataframe.values.tolist()
    #-------------------------------------------------------------------------------------------------
    
    # INSTRUCTIONS TO FILL IN PARAMETERS BELOW
    # Provide appropriate parameters as possible.
    # Comment the entire line of the parameter you dont want to provide(or dont know).
    # If you comment a parameter, check that you are confertable with the default value!
    # To simulate multiple values of the same parameter, provide in a list format i.e [first value,second value,third value, etc]
    
    cli_params = {
        'locus_specification': locus_specification_list, 
        'shuffle_parameter': [2,4,8],
        'ctcf_selection_threshold': [4,8,12],
        'mutation_method': ['permute_whole_seq','randomise_whole_seq','mask_motif','permute_motif','randomise_motif'],
    }
    cli_param_set = list(itertools.product(*[v for v in cli_params.values()]))
    parameters_combo_dataframe = pd.DataFrame(cli_param_set, columns=cli_params.keys())
    fill_in_default_values(parameters_combo_dataframe)
    parameters_combo_dataframe[["locus_specification",'GC_content']] = parameters_combo_dataframe["locus_specification"].str.split('-',expand=True)
    os.makedirs(parameters_combo_dataframe.out_folder[0], exist_ok=True)
    storage_folder = parameters_combo_dataframe.out_folder[0]
    parameters_combo_dataframe.to_csv(f'{storage_folder}/shuffled_seqs_human.tsv', sep='\t', index=False)
    
    
def fill_in_default_values(dataframe):
    "function to fill in missing important parameters"
    parameter_space = [('out_folder', 'data'),
                       ('locus_specification', 'chr12,113_500_000,118_500_000-0.35'),
                       ('shuffle_parameter', 2),
                       ('ctcf_selection_threshold', 8),
                       ('mutation_method', 'randomise_whole_seq'),
                       ]
    for (parameter, default_value) in parameter_space:
        if parameter not in dataframe.columns:
            dataframe.insert(1, parameter, default_value,
                             allow_duplicates=False)
            
################################################################################
# __main__
################################################################################

if __name__ == "__main__":
    main()
