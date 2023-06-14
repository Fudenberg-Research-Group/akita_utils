#!/usr/bin/env python

# Copyright 2017 Calico LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =========================================================================

###################################################

"""
generate_log-spacing_df.py

This script creates a tsv (default) or csv file that can be used as an input to the spacing experiment script - virtual_symmetric_flank_experiment.py.     # should be changed?

tsv / csv table columns:
chrom | start | end | strand | genomic_SCD | orientation | background_index | flank_bp | spacer_bp

This way one row represents a single experiment.

At the next step - while running virtual_symmetric_flank_experiment.py, calculated statistics metrics e.g. SCD, INS-16, will be added as next columns. 

The script requires the following input:
- number of strong CTCF binding sites
- number of weak CTCF binding sites
- orientation string
- flank range
- desired sum of the length of (flank + spacer)
- (optional) an argument "all_permutations" storing True/False
- (optional) number of background sequences

If the provided orientation_string is a single string of length N, for example orientation_string=">>>", N=3.
    a) if all_permutations == True
    - all possible orientation string permutations of length N are created and tested for each strong and weak CTCF binding site.
    b) otherwise, only the single orientation_string is tested with each possible strong and weak CTCF binding site.
If the orientation_string is a comma-separated list of multiple string, e.g. orientation_string=">>>,>>,<>"
    - then each strong and weak motif will be tested with each orientation on the provided list.

"""

################################################################################
# imports
################################################################################

from __future__ import print_function
import os

os.environ["OPENBLAS_NUM_THREADS"] = "1"

import random
from optparse import OptionParser
import pandas as pd
import numpy as np

import bioframe

from akita_utils.format_io import read_rmsk
from akita_utils.tsv_gen_utils import (
    filter_boundary_ctcfs_from_h5,
    filter_by_overlap_num,
    add_orientation,
    add_background,
    add_const_flank_and_diff_spacer,
    validate_df_lenght,
    filter_dataframe_by_column,
)

from cooltools.lib import numutils

################################################################################
# main
################################################################################


def main():
    usage = "usage: %prog [options]"
    parser = OptionParser(usage)
    parser.add_option(
        "--num-strong",
        dest="num_strong_motifs",
        default=2,
        type="int",
        help="Specify number of strong CTCFs to be tested",
    )
    parser.add_option(
        "--num-weak",
        dest="num_weak_motifs",
        default=2,
        type="int",
        help="Specify number of weak CTCFs to be tested",
    )
    parser.add_option(
        "--orientation-string",
        dest="orientation_string",
        default=">>",
        type="string",
        help="Specify orientation string - one string that will be tested for each CTCF or a list of orientation strings",
    )
    parser.add_option(
        "--flank",
        dest="flank",
        default=20,
        type="int",
        help="Specify range of right and left flank to be tested",
    )
    parser.add_option(
        "--space-range",
        dest="space_range",
        default="1,1000",
        type="string",
        help="Specify range of spacing to be tested",
    )
    parser.add_option(
        "--num_log-intervals",
        dest="log_space_range",
        default=200,
        type="int",
        help="Specify number of intervals to divide the space-range into",
    )
    parser.add_option(
        "--backgrounds-indices",
        dest="backgrounds_indices",
        default="0,1,2,3,4,5,6,7,8,9",
        type="string",
        help="Specify number of background sequences that CTCFs will be inserted into",
    )
    parser.add_option(
        "--jaspar-file",
        dest="jaspar_file",
        default="/project/fudenber_735/motifs/mm10/jaspar/MA0139.1.tsv.gz",
        help="Specify path to the file with coordinates of CTCF-binding sites in the tested genome",
    )
    parser.add_option(
        "--rmsk-file",
        dest="rmsk_file",
        default="/project/fudenber_735/genomes/mm10/database/rmsk.txt.gz",
        help="Specify path to the file with coordinates of repeated elements in the tested genome",
    )
    parser.add_option(
        "--all-permutations",
        dest="all_permutations",
        default=False,
        action="store_true",
        help="Test all possible permutations of N = length of provided orientation_string",
    )
    parser.add_option(
        "--filename",
        dest="filename",
        default="out",
        help="Filename for output",
    )
    parser.add_option(
        "--verbose",
        dest="verbose",
        default=False,
        action="store_true",
        help="Print tsv file summary",
    )
    parser.add_option(
        "--tsv",
        dest="tsv",
        default=True,
        action="store_true",
        help="Save dataframe as tsv",
    )
    parser.add_option(
        "--csv",
        dest="csv",
        default=False,
        action="store_true",
        help="Save dataframe as csv",
    )

    (options, args) = parser.parse_args()

    orient_list = options.orientation_string.split(",")
    num_orients = len(orient_list)
    N = len(orient_list[0])
    all_permutations = options.all_permutations

    flank = options.flank
    spacing_start, spacing_end = [int(num) for num in options.space_range.split(",")]

    spacing_list = list(
        np.unique(
            numutils.logbins(
                lo=spacing_start,
                hi=spacing_end,
                N=options.log_space_range,
                version=2,
            )
            - 1
        )
    )

    rmsk_exclude_window = flank
    ctcf_exclude_window = 2 * flank

    background_indices_list = options.backgrounds_indices.split(",")

    if options.all_permutations == True:
        assert len(orient_list) == 1
        num_orients = 2**N

    random.seed(44)

    # loading motifs
    score_key = "SCD"

    sites = filter_boundary_ctcfs_from_h5(
        h5_dirs="/project/fudenber_735/tensorflow_models/akita/v2/analysis/permute_boundaries_motifs_ctcf_mm10_model*/scd.h5",
        score_key=score_key,
        threshold_all_ctcf=5,
    )

    # filtering by rmsk
    rmsk_df = read_rmsk(options.rmsk_file)

    sites = filter_by_overlap_num(
        sites,
        rmsk_df,
        expand_window=rmsk_exclude_window,
        working_df_cols=["chrom", "start_2", "end_2"],
    )

    # filtering by ctcf
    ctcf_df = bioframe.read_table(options.jaspar_file, schema="jaspar")

    sites = filter_by_overlap_num(
        sites,
        ctcf_df,
        expand_window=ctcf_exclude_window,
        working_df_cols=["chrom", "start_2", "end_2"],
    )

    # chosing strong and weak sites
    strong_sites = filter_dataframe_by_column(
        sites,
        column_name="SCD",
        upper_threshold=99,
        lower_threshold=1,
        filter_mode="head",
        num_rows=options.num_strong_motifs,
    )

    weak_sites = filter_dataframe_by_column(
        sites,
        column_name="SCD",
        upper_threshold=99,
        lower_threshold=1,
        filter_mode="tail",
        num_rows=options.num_weak_motifs,
    )

    site_df = pd.concat([strong_sites.copy(), weak_sites.copy()])
    seq_coords_df = (
        site_df[["chrom", "start_2", "end_2", "strand_2", score_key]]
        .copy()
        .rename(
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

    # adding orientation, background index, information about flanks and spacers
    df_with_orientation = add_orientation(
        seq_coords_df,
        orientation_strings=orient_list,
        all_permutations=all_permutations,
    )

    df_with_background = add_background(df_with_orientation, background_indices_list)

    df_with_flanks_spacers = add_const_flank_and_diff_spacer(
        df_with_background, flank, spacing_list
    )

    df_with_flanks_spacers = df_with_flanks_spacers.drop(columns="index")
    df_with_flanks_spacers.index.name = "experiment_id"

    (expected_df_len, observed_df_len) = validate_df_lenght(
        options.num_strong_motifs,
        options.num_weak_motifs,
        num_orients,
        len(background_indices_list),
        len(spacing_list),
        df_with_flanks_spacers,
    )

    if options.verbose:
        print("\nSummary")
        print(
            "Number of CTCF binding sites: ",
            options.num_strong_motifs + options.num_weak_motifs,
        )
        print("Number of orientations: ", num_orients)
        print("Number of background sequences: ", len(background_indices_list))
        print(
            "Number of different spacings: ",
            len(spacing_list),
        )
        print("===============")

        print("Expected length of dataframe: ", expected_df_len)
        print("True length of dataframe: ", observed_df_len, "\n")

    if options.csv:
        df_with_flanks_spacers.to_csv(f"./{options.filename}.csv", index=True)

    if options.tsv:
        df_with_flanks_spacers.to_csv(f"./{options.filename}.tsv", sep="\t", index=True)


################################################################################
# __main__
################################################################################
if __name__ == "__main__":
    main()
