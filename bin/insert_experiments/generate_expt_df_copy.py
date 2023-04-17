"""
This script takes various parameters as input, including background sequences and directories for CTCF data in h5 format. These directories can be provided using the 'ctcf_h5_dirs' argument. A JSON file, which includes file paths for various insert data tsv files, can also be provided using the 'json-file' argument.

This script allows for the creation of different virtual insertions by changing the offsets and orientation of CTCFs. For example, to create a boundary with close CTCFs, the 'ctcf_offset_1' parameter would be set to 0, the 'ctcf_offset_2' parameter would be set to 120, and the 'ctcf_orientation_1' and 'ctcf_orientation_2' parameters would be set to ">".

The script outputs a dataframe with different permutations of inserts, which can be used to generate scores for different virtual insertions. These inserts include background with CTCFs alone, enhancers alone, promoters alone, enhancer-promoter, enhancer-CTCF, promoter-CTCF, CTCF-enhancer-promoter, and others. Note that the maximum offset on either positive or negative is around 500,000 base pairs.
"""
# import general libraries
import itertools
import os
import pandas as pd
import akita_utils
import akita_utils.tsv_gen_utils
import argparse
import json

import logging

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
log = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-background_seqs", nargs="+", dest="background_seqs", default=[0], type=int
    )
    parser.add_argument(
        "-o", dest="out_dir", default="default_exp_data", help="where to store output"
    )
    parser.add_argument(
        "--json-file", help="Path to JSON file with insert(s) dataframe(s)"
    )

    args = parser.parse_args()

    insert_names_list = []

    # ---------------setting up a grid search over parameters-------------------------------
    grid_search_params = {
        "background_seqs": args.background_seqs,
    }

    # ---------------importing inserts from json file if given----------------------------
    if args.json_file:
        with open(args.json_file) as f:
            data = json.load(f)

        # Loop through each dataframe in the JSON data
        for dataframe_number, df_data in enumerate(data):

            insert_names_list += [f"{dataframe_number}"]
            # Access the dataframe from JSON data
            df_path = df_data["data"]
            root, ext = os.path.splitext(df_path)

            # Check if the file path is a CSV or HDF5 file
            if ext == ".csv":
                df = pd.read_csv(df_path)
                offset=df_data["offset"]
                flank_bp=df_data["flank_bp"]
                orientation=df_data["orientation"]
                locus_specification_list = (
                    akita_utils.tsv_gen_utils.generate_locus_specification_list(
                        df,
                        motif_threshold=0,
                        specification_list=[0, 2],
                        unique_identifier=dataframe_number,
                    )
                )
                grid_search_params[
                    f"{dataframe_number}_locus_specification"
                ] = locus_specification_list
                grid_search_params[f"{dataframe_number}_flank_bp"] = flank_bp
                grid_search_params[f"{dataframe_number}_offset"] = offset
                grid_search_params[f"{dataframe_number}_orientation"] = orientation

                print(f"finished processing dataset {dataframe_number}")
            
            elif ext == ".h5":
                offset=df_data["offset"]
                flank_bp=df_data["flank_bp"]
                orientation=df_data["orientation"]
                rmsk_file =df_data["rmsk_file"] 
                jaspar_file =df_data["jaspar_file"]
                score_key = df_data["score_key"]
                mode = df_data["mode"]
                num_sites = df_data["num_sites"]
                ctcf_locus_specification_list = (
                        akita_utils.tsv_gen_utils.generate_ctcf_positons(
                            df_path,
                            rmsk_file,
                            jaspar_file,
                            score_key,
                            mode,
                            num_sites,
                        )
                    )
                grid_search_params[f"{dataframe_number}_flank_bp"] = flank_bp
                grid_search_params[f"{dataframe_number}_offset"] = offset
                grid_search_params[f"{dataframe_number}_orientation"] = orientation
                grid_search_params[f"{dataframe_number}_locus_specification"]  = ctcf_locus_specification_list

                log.info(f"finished processing dataset {dataframe_number}")

            else:
                raise ValueError("File must be a CSV or HDF5 file.")

            
    log.info("Done preprocessing")

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
