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
generate_symmetric_flank_df.py

This script creates a tsv (default) or csv file that can be used as an input to the padding experiment script - virtual_symmetric_flank_experiment.py.

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
import math
import random
from optparse import OptionParser
import pandas as pd
import itertools

import bioframe
import akita_utils

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
        "--flank-range",
        dest="flank_range",
        default="0,30",
        type="string",
        help="Specify range of right and left flank to be tested",
    )
    parser.add_option(
        "--flank-spacer-sum",
        dest="flank_spacer_sum",
        default=90,
        type="int",
        help="Specify sum of flank and spacer so that distances between CTCFs binding sites are kept constant. 
        \n2xflank-spacer-sum=distance between two consecutive CTCFs.",
    )
    parser.add_option(
        "--number-backgrounds",
        dest="number_backgrounds",
        default=10,
        type="int",
        help="Specify number of background sequences that CTCFs will be inserted into",
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

    if options.all_permutations == True:
        assert len(orient_list) == 1
        num_orients = 2**N

    random.seed(44)

    # loading motifs
    score_key = "SCD"
    weak_thresh_pct = 1
    strong_thresh_pct = 99
    pad_flank = 0

    sites = akita_utils.filter_boundary_ctcfs_from_h5(
        h5_dirs="/project/fudenber_735/tensorflow_models/akita/v2/analysis/permute_boundaries_motifs_ctcf_mm10_model*/scd.h5",
        score_key=score_key,
        threshold_all_ctcf=5,
    )

    strong_sites, weak_sites = akita_utils.filter_sites_by_score(
        sites,
        score_key=score_key,
        weak_thresh_pct=weak_thresh_pct,
        weak_num=options.num_strong_motifs,
        strong_thresh_pct=strong_thresh_pct,
        strong_num=options.num_strong_motifs,
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
    seq_coords_df = bioframe.expand(seq_coords_df, pad=pad_flank)

    # adding orientation, background index, information about flanks and spacers
    df_with_orientation = add_orientation(
        seq_coords_df,
        orientation_strings=orient_list,
        all_permutations=all_permutations,
    )

    df_with_background = add_background(
        df_with_orientation, options.number_backgrounds
    )

    df_with_flanks_spacers = add_flanks_and_spacers(
        df_with_background, options.flank_range, options.flank_spacer_sum
    )

    df_with_flanks_spacers = df_with_flanks_spacers.drop(columns="index")
    df_with_flanks_spacers.index.name = "experiment_id"

    expected = (
        (options.num_strong_motifs + options.num_weak_motifs)
        * num_orients
        * options.number_backgrounds
        * (
            int(options.flank_range.split(",")[1])
            - int(options.flank_range.split(",")[0])
            + 1
        )
    )
    observed = len(df_with_flanks_spacers)

    assert expected == observed

    if options.verbose:
        print("\nSummary")
        print(
            "Number of CTCF binding sites: ",
            options.num_strong_motifs + options.num_weak_motifs,
        )
        print("Number of orientations: ", num_orients)
        print("Number of background sequences: ", options.number_backgrounds)
        print(
            "Number of flanks: ",
            int(options.flank_range.split(",")[1])
            - int(options.flank_range.split(",")[0])
            + 1,
        )
        print("===============")

        print("Expected length of dataframe: ", expected)
        print("True length of dataframe: ", observed, "\n")

    if options.csv:
        df_with_flanks_spacers.to_csv(f"./{options.filename}.csv", index=True)

    if options.tsv:
        df_with_flanks_spacers.to_csv(
            f"./{options.filename}.tsv", sep="\t", index=True
        )


#################################################################


def generate_all_orientation_strings(N):
    """
    Function generates all possible orientations of N-long string consisting of binary characters (> and <) only.
    Example: for N=2 the result is ['>>', '><', '<>', '<<'].
    """
    def _binary_to_orientation_string_map(binary_list):

        binary_to_orientation_dict = {0: ">", 1: "<"}
        orientation_list = [
            binary_to_orientation_dict[number] for number in binary_list
        ]

        return "".join(orientation_list)

    binary_list = [list(i) for i in itertools.product([0, 1], repeat=N)]

    return [
        _binary_to_orientation_string_map(binary_element)
        for binary_element in binary_list
    ]


def add_orientation(seq_coords_df, orientation_strings, all_permutations):

    """
    Function adds an additional column named 'orientation', to the given dataframe where each row corresponds to one CTCF-binding site.
    """

    df_len = len(seq_coords_df)

    if len(orientation_strings) > 1:

        orientation_ls = []
        rep_unit = seq_coords_df

        for ind in range(len(orientation_strings)):
            orientation = orientation_strings[ind]
            orientation_ls = orientation_ls + [orientation] * df_len
            if len(seq_coords_df) != len(orientation_ls):
                seq_coords_df = pd.concat(
                    [seq_coords_df, rep_unit], ignore_index=True
                )

        seq_coords_df["orientation"] = orientation_ls

    else:
        if all_permutations:

            N = len(orientation_strings[0])

            orientation_strings = generate_all_orientation_strings(N)

            orientation_ls = []
            rep_unit = seq_coords_df

            for ind in range(len(orientation_strings)):
                orientation = orientation_strings[ind]
                orientation_ls = orientation_ls + [orientation] * df_len
                if len(seq_coords_df) != len(orientation_ls):
                    seq_coords_df = pd.concat(
                        [seq_coords_df, rep_unit], ignore_index=True
                    )

            seq_coords_df["orientation"] = orientation_ls

        else:
            orientation_ls = []
            orientation_ls = orientation_strings * df_len

            seq_coords_df["orientation"] = orientation_ls

    return seq_coords_df


def add_flanks_and_spacers(seq_coords_df, flank_range, flank_spacer_sum):

    l, h = [int(num) for num in flank_range.split(",")]

    rep_unit = seq_coords_df
    df_len = len(rep_unit)

    flank_ls = []
    spacer_ls = []

    for flank in range(l, h + 1):
        spacer = flank_spacer_sum - flank
        flank_ls = flank_ls + [flank] * df_len
        spacer_ls = spacer_ls + [spacer] * df_len

        if len(seq_coords_df) != len(flank_ls):
            seq_coords_df = pd.concat(
                [seq_coords_df, rep_unit], ignore_index=True
            )

    seq_coords_df["flank_bp"] = flank_ls
    seq_coords_df["spacer_bp"] = spacer_ls

    return seq_coords_df


def add_background(seq_coords_df, number_backgrounds):

    rep_unit = seq_coords_df
    df_len = len(rep_unit)

    background_ls = []

    for background_ind in range(number_backgrounds):
        background_ls = background_ls + [background_ind] * df_len

        if len(seq_coords_df) != len(background_ls):
            seq_coords_df = pd.concat(
                [seq_coords_df, rep_unit], ignore_index=True
            )

    seq_coords_df["background_index"] = background_ls

    return seq_coords_df


################################################################################
# __main__
################################################################################
if __name__ == "__main__":
    main()
