# import general libraries
import itertools
import os

import bioframe
import gtfparse
import numpy as np
import pandas as pd

import akita_utils.tsv_gen_utils
import gtfparse
from pathlib import Path

# loading motifs
score_key = "SCD"
weak_thresh_pct = 10
strong_thresh_pct = 90
rmsk_file = "/project/fudenber_735/genomes/mm10/database/rmsk.txt.gz"
jaspar_file = "/project/fudenber_735/motifs/mm10/jaspar/MA0139.1.tsv.gz"


sites = akita_utils.tsv_gen_utils.filter_boundary_ctcfs_from_h5(
    h5_dirs="/project/fudenber_735/tensorflow_models/akita/v2/analysis/permute_boundaries_motifs_ctcf_mm10_model*/scd.h5",
    score_key=score_key,
    threshold_all_ctcf=5,
)

sites = akita_utils.tsv_gen_utils.filter_by_rmsk(
    sites,
    rmsk_file = rmsk_file, 
    verbose=True)

sites = akita_utils.tsv_gen_utils.filter_by_ctcf(sites,
    ctcf_file = jaspar_file,
    verbose=True)

strong_sites = akita_utils.tsv_gen_utils.filter_sites_by_score(
    sites,
    score_key=score_key,
    lower_threshold=weak_thresh_pct,
    upper_threshold=strong_thresh_pct,
    mode="head",
    num_sites=5
)

weak_sites = akita_utils.tsv_gen_utils.filter_sites_by_score(
    sites,
    score_key=score_key,
    lower_threshold=weak_thresh_pct,
    upper_threshold=strong_thresh_pct,
    mode="tail",
    num_sites=5
)


site_df = pd.concat([strong_sites.copy(), weak_sites.copy()])

seq_coords_df = (site_df[["chrom", "start_2", "end_2", "strand_2", score_key]].copy().rename(
        columns={
            "start_2": "start",
            "end_2": "end",
            "strand_2": "strand",
            score_key: "genomic_" + score_key,
        }
    )
)

seq_coords_df.reset_index(drop=True, inplace=True)
seq_coords_df.reset_index(inplace=True)

seq_coords_df = (
    seq_coords_df["chrom"].map(str)
    + ","
    + seq_coords_df["start"].map(str)
    + ","
    + seq_coords_df["end"].map(str)
    + "#"
    + seq_coords_df["strand"].map(str)
    + "#"
    + seq_coords_df["genomic_" + score_key].map(str)
)

ctcf_locus_specification_list = seq_coords_df.values.tolist()  # to be modified to have modules attached


#---------------importing genes + cropping upstream bps workflow-------------------------------
current_file_path = Path(__file__)
feature_data_tsv = current_file_path.parent / "data/feature_dataframe.tsv"

up_stream_bps = 10000 # Number of basepairs upstream, will become an option
feature_dataframe = pd.read_csv(feature_data_tsv, sep="\t")
feature_dataframe["start"] = feature_dataframe["start"]-[up_stream_bps]
feature_dataframe.reset_index(drop=True, inplace=True)
feature_dataframe["locus_specification"] = feature_dataframe["chrom"].map(str) +","+ feature_dataframe["start"].map(str) + "," + feature_dataframe["end"].map(str)+"#"+feature_dataframe["strand"].map(str)+"#"+feature_dataframe["SYMBOL"].map(str)
gene_locus_specification_list = feature_dataframe["locus_specification"].values.tolist()[7:12]
#-------------------------------------------------------------------------------------------------

# setting up a grid search over parameters

# INSTRUCTIONS TO FILL IN PARAMETERS BELOW
# Provide as many parameters as appropriate.(the more, the more time for results generation)
# Comment the entire line of the parameter you dont want to provide.
# If you comment a parameter, check that you are confertable with the default value!
# To simulate multiple values of the same parameter, provide in a list format i.e [first value,second value,third value, etc]

cli_params = {
    'out_folder': ["data"],# should have appropriate permissions in case folder is absent (otherwise provide already existing folder), automaticaly will create ./data if commented
    'ctcf_locus_specification': ctcf_locus_specification_list,
    'gene_locus_specification': gene_locus_specification_list,
    'ctcf_flank_bp': [i for i in range(0,61,10)],
    'gene_flank_bp': [i for i in range(0,101,20)],
    'background_seqs': [2], 
    'spacer_bp': [i for i in range(0,61,10)],
    'locus_orientation':[">>","<<","<>","><"],
    'swap_flanks': ["all_for_strong","all_for_weak","no"] # threshold(15) is hard coded in akita_utils.seq_gens.create_insertion (should i open to user?) 
}

cli_param_set = list(itertools.product(*[v for v in cli_params.values()]))

# -------------------------------------------------------------------------------------------------
# filling missing table values (if a key was deleted in dictionary above or commented)

parameters_combo_dataframe = pd.DataFrame(cli_param_set, columns=cli_params.keys())


def fill_in_default_values(dataframe):
    "filling default values in ungiven or commented parameters"

    parameter_space = [('out_folder', 'data'),
                       ('ctcf_locus_specification', 'chr12,113_500_000,118_500_000#-#4'),
                       ('gene_locus_specification', "chr1,3063252,3073252#+#ENSMUSG00000102693.1"),
                       ('ctcf_flank_bp', 25),
                       ('gene_flank_bp', 0),
                       ('background_seqs', 0),
                       ('spacer_bp', 50),
                       ('locus_orientation', ">>"),
                       ('swap_flanks', None),
                       ]
    
    for (parameter, default_value) in parameter_space:
        if parameter not in dataframe.columns:
            dataframe.insert(1, parameter, default_value, allow_duplicates=False)


fill_in_default_values(parameters_combo_dataframe)

# -------------------------------------------------------------------------------------------------

# adapting dataframe to desired look
parameters_combo_dataframe[["ctcf_locus_specification",'ctcf_strand','ctcf_genomic_score']] = parameters_combo_dataframe["ctcf_locus_specification"].str.split('#',expand=True)
parameters_combo_dataframe[["gene_locus_specification",'gene_strand','gene_symbol']] = parameters_combo_dataframe["gene_locus_specification"].str.split('#',expand=True)

parameters_combo_dataframe["insert_strand"] = (
    parameters_combo_dataframe["ctcf_strand"] + "$" + parameters_combo_dataframe["gene_strand"]
)
parameters_combo_dataframe["insert_loci"] = (
    parameters_combo_dataframe["ctcf_locus_specification"]
    + "$"
    + parameters_combo_dataframe["gene_locus_specification"]
)
parameters_combo_dataframe["insert_flank_bp"] = (
    parameters_combo_dataframe["ctcf_flank_bp"].map(str) + "$" + parameters_combo_dataframe["gene_flank_bp"].map(str)
)

os.makedirs(parameters_combo_dataframe.out_folder[0], exist_ok=True)
storage_folder = parameters_combo_dataframe.out_folder[0]
parameters_combo_dataframe.to_csv(f'{storage_folder}/parameters_combo_all.tsv', sep='\t', index=False)
