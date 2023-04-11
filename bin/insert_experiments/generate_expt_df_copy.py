"""
This script takes various parameters as input, including background sequences and directories for CTCF data in h5 format. These directories can be provided using the 'ctcf_h5_dirs' argument. A JSON file, which includes file paths for various insert data tsv files, can also be provided using the 'json-file' argument.

This script allows for the creation of different virtual insertions by changing the offsets and orientation of CTCFs. For example, to create a boundary with close CTCFs, the 'ctcf_offset_1' parameter would be set to 0, the 'ctcf_offset_2' parameter would be set to 120, and the 'ctcf_orientation_1' and 'ctcf_orientation_2' parameters would be set to ">".

The script outputs a dataframe with different permutations of inserts, which can be used to generate scores for different virtual insertions. These inserts include background with CTCFs alone, enhancers alone, promoters alone, enhancer-promoter, enhancer-CTCF, promoter-CTCF, CTCF-enhancer-promoter, and others. Note that the maximum offset on either positive or negative is around 500,000 base pairs.
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
import sys
import json

import logging

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
log = logging.getLogger(__name__)


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
    parser.add_argument(
        "-score_key", 
        dest="score_key", 
        default="SCD"
    )
    parser.add_argument(
        "-ctcf_h5_dirs", 
        dest="h5_dirs", 
        help="h5_dirs", 
        default=None
    )
    parser.add_argument(
        "-mode", 
        dest="mode", 
        default="uniform"
    )
    parser.add_argument(
        "-num_sites", 
        dest="num_sites", 
        default=2, 
        type=int
    )
    parser.add_argument(
        "-background_seqs", 
        nargs="+", 
        dest="background_seqs", 
        default=[0], 
        type=int
    )
    parser.add_argument(
        "-o", 
        dest="out_dir", 
        default="default_exp_data", 
        help="where to store output"
    )
    parser.add_argument(
        "--json-file", 
        help="Path to JSON file with insert(s) dataframe(s)"
    )
    parser.add_argument(
        "-ctcf_offset_1",
        dest="ctcf_offset_1",
        help="ctcf_offset_1",
        nargs="+",
        default=[0],
    )
    parser.add_argument(
        "-ctcf_orientation_1",
        dest="ctcf_orientation_1",
        help="ctcf_orientation_1",
        nargs="+",
        default=[">"],
    )
    parser.add_argument(
        "-ctcf_flank_bp_1",
        dest="ctcf_flank_bp_1",
        help="ctcf_flank_bp_1",
        nargs="+",
        default=[20],
    )
    parser.add_argument(
        "-ctcf_offset_2",
        dest="ctcf_offset_2",
        help="ctcf_offset_2",
        nargs="+",
        default=[0],
    )
    parser.add_argument(
        "-ctcf_orientation_2",
        dest="ctcf_orientation_2",
        help="ctcf_orientation_2",
        nargs="+",
        default=[">"],
    )
    parser.add_argument(
        "-ctcf_flank_bp_2",
        dest="ctcf_flank_bp_2",
        help="ctcf_flank_bp_2",
        nargs="+",
        default=[20],
    )

    args = parser.parse_args()

    genome_open = pysam.Fastafile(args.genome_fasta)

    insert_names_list = []

    # ---------------setting up a grid search over parameters-------------------------------
    grid_search_params = {
        "background_seqs": args.background_seqs,
    }

    # ---------------importing CTCF motifs if given ctcf h5 file--------------------------------
    if args.h5_dirs:
        ctcf_locus_specification_list = (
            akita_utils.tsv_gen_utils.generate_ctcf_positons(
                args.h5_dirs,
                args.rmsk_file,
                args.jaspar_file,
                args.score_key,
                args.mode,
                args.num_sites,
            )
        )

        # configure the offsets carefully to create either boundaries or TADs (look at script doc string)
        insert_names_list += ["ctcf_1"]
        grid_search_params["ctcf_1_locus_specification"] = ctcf_locus_specification_list
        grid_search_params["ctcf_1_flank_bp"] = args.ctcf_flank_bp_1
        grid_search_params["ctcf_1_offset"] = args.ctcf_offset_1
        grid_search_params["ctcf_1_orientation"] = args.ctcf_orientation_1

        insert_names_list += ["ctcf_2"]
        grid_search_params["ctcf_2_locus_specification"] = ctcf_locus_specification_list
        grid_search_params["ctcf_2_flank_bp"] = args.ctcf_flank_bp_2
        grid_search_params["ctcf_2_offset"] = args.ctcf_offset_2
        grid_search_params["ctcf_2_orientation"] = args.ctcf_orientation_2

    # ---------------importing other inserts from json file if given----------------------------
    if args.json_file:
        dfs = []
        # Load the JSON data
        with open(args.json_file) as f:
            data = json.load(f)

        subparsers = parser.add_subparsers(dest="subcommand")

        # Loop through each dataframe in the JSON data
        for i, df_data in enumerate(data):
            
            parser_df = subparsers.add_parser(f"df{i+1}")
            parser_df.add_argument(
                f"--data", default=None, help=f"Path to dataframe {i+1} CSV file"
            )
            parser_df.add_argument(
                f"--offset",
                type=int,
                default=None,
                help=f"Offset value for dataframe {i+1}",
            )
            parser_df.add_argument(
                f"--flank_bp",
                type=int,
                default=None,
                help=f"Flank value for dataframe {i+1}",
            )
            parser_df.add_argument(
                f"--orientation",
                choices=["forward", "reverse"],
                default=None,
                help=f"Orientation value for dataframe {i+1}",
            )

            # Set the default values for the subparser based on the JSON data
            parser_df.set_defaults(data=df_data["data"])
            parser_df.set_defaults(offset=df_data["offset"])
            parser_df.set_defaults(flank_bp=df_data["flank_bp"])
            parser_df.set_defaults(orientation=df_data["orientation"])

            # Parse the command-line arguments for this dataframe
            secondary_args = parser_df.parse_args(args=[])

            # Access the dataframe using the parsed command-line arguments or defaults from JSON data
            df_path = secondary_args.data or df_data["data"]
            offset = secondary_args.offset or df_data["offset"]
            flank_bp = secondary_args.flank_bp or df_data["flank_bp"]
            orientation = secondary_args.orientation or df_data["orientation"]

            df = pd.read_csv(df_path)
            insert_names_list += [f"{i+1}"]
            dfs.append((df, offset, flank_bp, orientation, i + 1))

            print(f"finished processing dataset {i}")

        for df, offset, flank_bp, orientation, dataframe_number in dfs:
            locus_specification_list = (
                akita_utils.tsv_gen_utils.generate_locus_specification_list(
                    df,
                    genome_open,
                    motif_threshold=0,
                    specification_list=[0, 2],
                    unique_identifier=dataframe_number,
                )
            )
            grid_search_params[
                f"{dataframe_number}_locus_specification"
            ] = locus_specification_list
            grid_search_params[f"{dataframe_number}_flank_bp"] = flank_bp
            grid_search_params[
                f"{dataframe_number}_offset"
            ] = offset  
            grid_search_params[f"{dataframe_number}_orientation"] = orientation

    print("Done preprocessing")

    # --------------- grid search of provided parameters -------------------------------
    grid_search_params_set = list(
        itertools.product(*[v for v in grid_search_params.values()])
    )
    parameters_combo_dataframe = pd.DataFrame(
        grid_search_params_set, columns=grid_search_params.keys()
    )
    parameters_combo_dataframe = (
        akita_utils.tsv_gen_utils.parameter_dataframe_reorganisation(
            parameters_combo_dataframe, insert_names_list
        )
    )
    parameters_combo_dataframe.to_csv(f"{args.out_dir}.tsv", sep="\t", index=False)


if __name__ == "__main__":
    main()
