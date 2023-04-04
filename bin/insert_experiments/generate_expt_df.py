"""
This script takes numerous parameters i.e

(1) 'ctcf_locus_specification_1' 
(2) 'ctcf_locus_specification_2',
(3) 'gene_locus_specification',
(4) 'enhancer_locus_specification',
(5) 'ctcf_flank_bp_1',
(6) 'ctcf_flank_bp_2',
(7) 'gene_flank_bp',
(8) 'enhancer_flank_bp',
(9) 'background_seqs',
(10) 'ctcf_offset_1',
(11) 'ctcf_offset_2',
(12) 'gene_offset',
(13) 'enhancer_offset',
(14) 'ctcf_orientation_1',
(15) 'ctcf_orientation_2',
(16) 'gene_orientation',
(17) 'enhancer_orientation',


and creats a dataframe with different permutation of these parameters which can be used to generate scores for different scenarios in experimental setting i.e

Background with; 

    CTCFs alone
    Enhancers alone
    Promoters alone
    Ehancer-Promoter
    CTCF-Ehancer-Promoter
    
    
By changing offsets of CTCFs, on can experiment with creating;

    Boundary(close CTCFs): ['ctcf_offset_1' = 0,'ctcf_offset_2' = 120]
    TADs:  ['ctcf_offset_1' = -490000,'ctcf_offset_2' = 490000]


sample paths to respective files are:

    -ctcf_h5_dirs: /project/fudenber_735/tensorflow_models/akita/v2/analysis/permute_boundaries_motifs_ctcf_mm10_model1/scd.h5
    -promoters_df: ~/akita_utils/bin/insert_experiments/data/promoter_score_sample.csv
    -enhancers_df: ~/akita_utils/bin/insert_experiments/data/enhancer_score_sample.csv
    
"""
# import general libraries
import itertools
import os
import pysam
import bioframe
import gtfparse
import numpy as np
import pandas as pd
import akita_utils
import akita_utils.tsv_gen_utils
import gtfparse
import argparse
from pathlib import Path
from io import StringIO

import logging

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
log = logging.getLogger(__name__)


######################################################################
# __main__
######################################################################
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-rmsk_file",
        dest="rmsk_file",
        help="rmsk_file",
        default="/project/fudenber_735/genomes/mm10/database/rmsk.txt.gz",
    )
    parser.add_argument(
        "-jaspar_file",
        dest="jaspar_file",
        help="jaspar_file",
        default="/project/fudenber_735/motifs/mm10/jaspar/MA0139.1.tsv.gz",
    )
    parser.add_argument(
        "-f",
        dest="genome_fasta",
        help="fasta file",
        default="/project/fudenber_735/genomes/mm10/mm10.fa",
    )
    parser.add_argument("-score_key", dest="score_key", default="SCD")
    parser.add_argument(
        "-ctcf_h5_dirs",
        dest="h5_dirs",
        help="h5_dirs",
        default=None)
    parser.add_argument("-mode", dest="mode", default="tail")
    parser.add_argument("-num_sites", dest="num_sites", default=2, type=int)
    parser.add_argument(
        "-background_seqs", nargs="+", dest="background_seqs", default=[0], type=int
    )
    parser.add_argument(
        "-o", dest="out_dir", default="default_exp_data", help="where to store output"
    )
    parser.add_argument(
        "-ctcfs_df", dest="ctcfs_dataframe", help="ctcfs_dataframe path"
    )
    parser.add_argument(
        "-promoters_df", dest="promoters_dataframe", help="promoters_dataframe path", default=None
    )
    parser.add_argument(
        "-enhancers_df", dest="enhancers_dataframe", help="enhancers_dataframe path", default=None
    )
    parser.add_argument(
        "-ctcf_offset_1", dest="ctcf_offset_1", help="ctcf_offset_1", nargs="+", default=[0]
    )
    parser.add_argument(
        "-ctcf_orientation_1", dest="ctcf_orientation_1", help="ctcf_orientation_1", nargs="+", default=[">"]
    )
    parser.add_argument(
        "-ctcf_flank_bp_1", dest="ctcf_flank_bp_1", help="ctcf_flank_bp_1", nargs="+", default=[20]
    )
    parser.add_argument(
        "-ctcf_offset_2", dest="ctcf_offset_2", help="ctcf_offset_2", nargs="+", default=[0]
    )
    parser.add_argument(
        "-ctcf_orientation_2", dest="ctcf_orientation_2", help="ctcf_orientation_2", nargs="+", default=[">"]
    )
    parser.add_argument(
        "-ctcf_flank_bp_2", dest="ctcf_flank_bp_2", help="ctcf_flank_bp_2", nargs="+", default=[20]
    )
    parser.add_argument(
        "-enhancer_offset", dest="enhancer_offset", help="enhancer_offset", nargs="+", default=[50000]
    )
    parser.add_argument(
        "-enhancer_orientation", dest="enhancer_orientation", help="enhancer_orientation", nargs="+", default=[">"]
    )
    parser.add_argument(
        "-enhancer_flank_bp", dest="enhancer_flank_bp", help="enhancer_flank_bp", nargs="+", default=[20]
    )
    parser.add_argument(
        "-gene_offset", dest="gene_offset", help="gene_offset", nargs="+", default=[100000]
    )
    parser.add_argument(
        "-gene_orientation", dest="gene_orientation", help="gene_orientation", nargs="+", default=[">"]
    )
    parser.add_argument(
        "-gene_flank_bp", dest="gene_flank_bp", help="gene_flank_bp", nargs="+", default=[20]
    )

    args = parser.parse_args()
    

    genome_open = pysam.Fastafile(args.genome_fasta)

    
    # ---------------setting up a grid search over parameters-------------------------------
    grid_search_params = {
        "background_seqs": args.background_seqs,
    }

    
    # ---------------importing CTCF motifs if given ctcf h5 file--------------------------------
    if args.h5_dirs: 
        ctcf_locus_specification_list = akita_utils.tsv_gen_utils.generate_ctcf_positons(
            args.h5_dirs,
            args.rmsk_file,
            args.jaspar_file,
            args.score_key,
            args.mode,
            args.num_sites,
        )

        # configure the offsets carefully to create either boundaries or TADs (look at script doc string)
        grid_search_params["ctcf_locus_specification_1"] = ctcf_locus_specification_list
        grid_search_params["ctcf_flank_bp_1"] = args.ctcf_flank_bp_1
        grid_search_params["ctcf_offset_1"] = args.ctcf_offset_1
        grid_search_params["ctcf_orientation_1"] = args.ctcf_orientation_1

        grid_search_params["ctcf_locus_specification_2"] = ctcf_locus_specification_list
        grid_search_params["ctcf_flank_bp_2"] = args.ctcf_flank_bp_2
        grid_search_params["ctcf_offset_2"] = args.ctcf_offset_2
        grid_search_params["ctcf_orientation_2"] = args.ctcf_orientation_2

    
    # ---------------importing promoters if given path--------------------------------
    if args.promoters_dataframe:
        promoter_data_csv = args.promoters_dataframe
        promoter_dataframe = pd.read_csv(promoter_data_csv, sep=",")
        gene_locus_specification_list = akita_utils.tsv_gen_utils.generate_promoter_list(
            promoter_dataframe, genome_open, motif_threshold=0, specification_list=[0]
        )
        grid_search_params["gene_locus_specification"] = gene_locus_specification_list
        grid_search_params["gene_flank_bp"] = args.gene_flank_bp
        grid_search_params["gene_offset"] = args.gene_offset # np.logspace(5, 5.0, num=1, dtype = int)
        grid_search_params["gene_orientation"] = args.gene_orientation

    
    # ---------------importing enhancers if given path-------------------------------
    if args.enhancers_dataframe:
        enhancer_data_csv = args.enhancers_dataframe
        enhancer_dataframe = pd.read_csv(enhancer_data_csv, sep=",")
        enhancer_locus_specification_list = akita_utils.tsv_gen_utils.generate_enhancer_list(
            enhancer_dataframe, genome_open, motif_threshold=0, specification_list=[0]
        )  #   
        grid_search_params[
            "enhancer_locus_specification"
        ] = enhancer_locus_specification_list
        grid_search_params["enhancer_flank_bp"] = args.enhancer_flank_bp
        grid_search_params["enhancer_offset"] = args.enhancer_offset
        grid_search_params["enhancer_orientation"] = args.enhancer_orientation
    
    
    # --------------- grid search of provided parameters -------------------------------
    grid_search_params_set = list(
        itertools.product(*[v for v in grid_search_params.values()])
    )
    parameters_combo_dataframe = pd.DataFrame(
        grid_search_params_set, columns=grid_search_params.keys()
    )
    fill_in_default_values(parameters_combo_dataframe)
    parameters_combo_dataframe = akita_utils.tsv_gen_utils.parameter_dataframe_reorganisation(
        parameters_combo_dataframe
    )
    parameters_combo_dataframe.to_csv(f"{args.out_dir}.tsv", sep="\t", index=False)


# -------------------------------------------------------------------------------------------------
# used functions below

def fill_in_default_values(dataframe):
    "filling default values in ungiven or commented parameters"
    parameter_space = [
        ("ctcf_locus_specification_1", None),
        ("ctcf_locus_specification_2", None),
        ("gene_locus_specification", None),
        ("enhancer_locus_specification", None),
        ("ctcf_flank_bp_1", 0),
        ("ctcf_flank_bp_2", 0),
        ("gene_flank_bp", 0),
        ("enhancer_flank_bp", 0),
        ("background_seqs", 0),
        ("ctcf_offset_1", None),
        ("ctcf_offset_2", None),
        ("gene_offset", None),
        ("enhancer_offset", None),
        ("ctcf_orientation_1", None),
        ("ctcf_orientation_2", None),
        ("gene_orientation", None),
        ("enhancer_orientation", None),
    ]

    for parameter, default_value in parameter_space:
        if parameter not in dataframe.columns:
            dataframe.insert(1, parameter, default_value, allow_duplicates=False)

if __name__ == "__main__":
    main()
