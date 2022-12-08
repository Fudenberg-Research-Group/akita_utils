# import general libraries
import os
import itertools
import pandas as pd
import numpy as np
import bioframe

# INSTRUCTIONS TO FILL IN PARAMETERS BELOW
# Provide as many parameters as possible.
# Comment the entire line of the parameter you dont want to provide.
# If you comment a parameter, check that you are confertable with the default value!
# To simulate multiple values of the same parameter, provide in a list format i.e [first value,second value,third value, etc]


chromsizes = bioframe.read_chromsizes('/project/fudenber_735/tensorflow_models/akita/v2/data/mm10/sequences.bed')
dframe = pd.DataFrame(chromsizes)
dframe['end'] = dframe['length']+ 1310720
dframe = dframe.reset_index()
dframe.rename(columns = {'index' : 'chrom', 'length':'start'}, inplace = True)
df = bioframe.frac_gc(dframe, bioframe.load_fasta('/project/fudenber_735/genomes/mm10/mm10.fa'), return_input=True)`


# function_for_choosing_loci_by_gc_content()
# num_loci : int
# gc_content_range :  None or tuple
# sampling_strategy : str, default 'uniform'

def locus_filter_by_GC_content(dataframe,error = 0.0001,sample_size=50):
    "takes a dataframe with colomns |chrom | start | end | GC"
    super_set = []
    for gc in np.percentile(dataframe['GC'].dropna().values, np.linspace(1,99,sample_size)):
        for i in range(dataframe.shape[0]):
            if gc-error <= dataframe['GC'].values[i] <= gc+error:
                super_set += [i]
                break
    new_dataframe = dataframe.iloc[[ind for ind in set(super_set)]]
    new_dataframe = new_dataframe["chrom"].map(str) +","+ new_dataframe["start"].map(str) + "," + new_dataframe["end"].map(str)+"-"+new_dataframe["GC"].map(str)
    return new_dataframe.values.tolist()


locus_specification_list = locus_filter_by_GC_content(df)


cli_params = {
    # should have appropriate permissions in case folder is absent (otherwise provide already existing folder), automaticaly will create ./locus_output if commented
    'out_folder': ["data"],
    'locus_specification': locus_specification_list, 
    'shuffle_parameter': [2,4,8],
    'mutation_method': ['permute_whole_seq','mask_motif','random','permute_motif'], 
    'ctcf_selection_threshold': [5,8,12],
    'map_score_threshold': [5500,6000,6500],
}


def fill_in_default_values(dataframe):

    parameter_space = [('out_folder', 'data'),
                       ('locus_specification', 'chr12,113_500_000,118_500_000-20'),
                       ('shuffle_parameter', 2),
                       ('mutation_method', 'mask'),
                       ('ctcf_selection_threshold', 8),
                       ('map_score_threshold', 5500),                       
                       ]
    
    for (parameter, default_value) in parameter_space:
        if parameter not in dataframe.columns:
            dataframe.insert(1, parameter, default_value,
                             allow_duplicates=False)
            
cli_param_set = list(itertools.product(*[v for v in cli_params.values()]))
parameters_combo_dataframe = pd.DataFrame(cli_param_set, columns=cli_params.keys())
fill_in_default_values(parameters_combo_dataframe)

parameters_combo_dataframe[["locus_specification",'GC_content']] = parameters_combo_dataframe["locus_specification"].str.split('-',expand=True)
os.makedirs(parameters_combo_dataframe.out_folder[0], exist_ok=True)
storage_folder = parameters_combo_dataframe.out_folder[0]

parameters_combo_dataframe.to_csv(f'{storage_folder}/parameters_combo.tsv', sep='\t', index=False)