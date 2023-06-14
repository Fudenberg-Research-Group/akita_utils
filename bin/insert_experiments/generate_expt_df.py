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
from akita_utils.tsv_gen_utils import (
    generate_locus_specification_list,
    parameter_dataframe_reorganisation,
)
from akita_utils.tsv_gen_utils import inserts_overlap_check
import argparse
import json
import logging

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
log = logging.getLogger(__name__)

DEFAULT_FLANK_BP = [0]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--background_seqs",
        nargs="+",
        dest="background_seqs",
        default=[0],
        type=int,
        help="list of which background seqs to pick, from the given set of background fasta seqs",
    )
    parser.add_argument(
        "-o", dest="out_dir", default="default_exp_data", help="where to store output"
    )
    parser.add_argument(
        "--json-file", help="Path to JSON file with insert(s) dataframe(s)"
    )
    parser.add_argument(
        "--dont_raise_error_with_ovelaps",
        help="Use this flag to skip searching for overlaps(and raising overlap error) between inserts",
        action="store_true",
    )

    args = parser.parse_args()

    insert_names_list = []
    grid_params = {
        "background_seqs": args.background_seqs,
    }

    if args.json_file:
        with open(args.json_file) as f:
            inserts = json.load(f)

        for insert_identification_number, insert_specs in enumerate(inserts):
            insert_identifier = insert_specs.get(
                "feature_name", f"{insert_identification_number}"
            )
            insert_names_list += [insert_identifier]
            df_path = insert_specs["path"]
            root, ext = os.path.splitext(df_path)

            # Check if the file path is a TSV
            if ext == ".tsv":
                grid_params[
                    f"{insert_identifier}_locus_specification"
                ] = generate_locus_specification_list(
                    dataframe=pd.read_csv(df_path, delimiter="\t"),
                    filter_out_ctcf_motifs=insert_specs.get(
                        "filter_out_inserts_with_ctcf_motifs", False
                    ),
                    specification_list=insert_specs.get("specification_list", None),
                    unique_identifier=insert_identifier,
                    ctcf_jaspar_file=insert_specs.get("ctcf_jaspar_file", None),
                )

            else:
                raise ValueError("Input file must be a TSV.")

            grid_params[f"{insert_identifier}_flank_bp"] = insert_specs.get(
                "flank_bp", DEFAULT_FLANK_BP
            )
            grid_params[f"{insert_identifier}_offset"] = insert_specs["offset"]
            grid_params[f"{insert_identifier}_orientation"] = insert_specs[
                "orientation"
            ]

            log.info(f"finished processing {insert_identifier} dataset")

    # --------------- create a grid over all provided parameters ---------------------
    grid_params_set = list(itertools.product(*[v for v in grid_params.values()]))
    grid_params_dataframe = pd.DataFrame(grid_params_set, columns=grid_params.keys())

    grid_params_dataframe = parameter_dataframe_reorganisation(
        grid_params_dataframe, insert_names_list
    )

    raise_error_with_ovelaps = not args.dont_raise_error_with_ovelaps

    if raise_error_with_ovelaps:
        log.info(
            "Searching for overlaps between inserts, it might take time depending on the number of inserts and parameters ..."
        )
        inserts_overlap_check(grid_params_dataframe)

    grid_params_dataframe.to_csv(f"{args.out_dir}", sep="\t", index=False)

    log.info("Done preprocessing")


if __name__ == "__main__":
    main()
