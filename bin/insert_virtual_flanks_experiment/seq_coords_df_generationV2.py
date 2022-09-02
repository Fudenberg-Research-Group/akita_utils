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
# imports #

from __future__ import print_function

# general

import os
import sys
import time
import math
import random

import json
from optparse import OptionParser
import pickle

# DS
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# bioinf

import h5py
import pysam

sys.path.insert(0, "/home1/smaruj/akita_utils/")
import akita_utils

# others
import itertools

# redundant ?
# import pdb
# from skimage.measure import


###################################################
# styles #

sns.set(style="ticks", font_scale=1.3)


###################################################
# description #

# """
# seq_coords_df_generationV2.py
#
# Given the parameters the scirpt creates a tsv (default) or csv file - an input to the padding experiment script.
#
# tsv / csv table columns
# chrom | start | end | strand | orientation | left_flank | right_flank | left_spacer | right_spacer | back_id
#
# This way one row is one single experiment.
# At the next step, calculated statistics metrics e.g. SCD, INS-16, will be added as next columns. 
#
# """

################################################################################
# main
################################################################################

def main():
    usage = "usage: %prog [options] <params_file> <model_file> <vcf_file>"
    parser = OptionParser(usage)
    parser.add_option(
        "-f",
        dest="genome_fasta",
        default=None,
        help="Genome FASTA for sequences [Default: %default]",
    )
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
        "--mode",
        dest="mode",
        default="same",
        type="str",
        help="Specify mode of adding orientation strings list: same, list, all_possible",
    )
    parser.add_option(
        "--orientation-string",
        dest="orientation_string",
        default=">>",
        type="string",
        help="Specify orientation string - one string that will be tested for each CTCF or a list of orientation strings",
    )
    ### NOTE: Argument after --orientation-list flag has to be written within ""
    parser.add_option(
        "--orientation-N",
        dest="orientation_N",
        default=2,
        type="int",
        help="If mode chosen is all_possible, specify number of CTCF binding sites for which all possible permutations will be tested",
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
        help="Specify sum of flank and spacer so that distances between CTCFs binding sites are kept constant",
    )
    parser.add_option(
        "--number-backgrounds",
        dest="number_backgrounds",
        default=10,
        type="int",
        help="Specify number of background sequences that CTCFs will be inserted into",
    )
    parser.add_option(
        "--filename",
        dest="filename",
        default="out",
        help="Filename for output",
    )
    parser.add_option(
        "--tsv",
        dest="tsv",
        default=False,
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
        
    # orientation modes checkpoints
    
    possible_modes = ["same", "list", "all_possible"]
    
    if options.mode == "same" and len(orient_list) != 1:
        parser.error("When the mode 'same' was chosen, only one orientation has to be provided!")
    
    if options.mode == "list" and len(orient_list) != (options.num_strong_motifs + options.num_weak_motifs):
        parser.error("When the mode 'list' was chosen, length of the orientation list provided has to be equal to the sum of number of strong and weak CTCF motifs!")
    
    if options.mode == "all_possible":
        print("When the mode 'all_possible' was chosen, you should provide ONLY a number of CTCF sites, all possible permutations of the orientation string will be generated automatically!")
    
    if options.mode not in possible_modes:
        parser.error("Invalid mode. Expected one of: %s" % possible_modes)
    
    random.seed(44)
    
    # opening genome FASTA
    genome_open = pysam.Fastafile(options.genome_fasta)
    
    # loading motifs
    
    seq_coords_df = akita_utils.prepare_insertion_tsv(
        h5_dirs = "/project/fudenber_735/tensorflow_models/akita/v2/analysis/permute_boundaries_motifs_ctcf_mm10_model*/scd.h5",
        score_key = "SCD",
        pad_flank = 0, #how much flanking sequence around the sites to include
        weak_thresh_pct = 1, # don't use sites weaker than this, might be artifacts
        weak_num = options.num_strong_motifs ,
        strong_thresh_pct = 99, # don't use sites stronger than this, might be artifacts
        strong_num = options.num_weak_motifs ,
        save_tsv = None
    )
        
    df_with_orientation = add_orientation(
        seq_coords_df,
        nr_strong = options.num_strong_motifs,
        nr_weak = options.num_weak_motifs,
        mode = options.mode,
        orientation_strings = options.orientation_string,
        N = options.orientation_N)
    
    df_with_background = add_background(
        seq_coords_df,
        options.number_backgrounds
    )
    
    df_with_flanks_spacers = add_flanks_and_spacers(
        df_with_background,
        options.flank_range,
        options.flank_spacer_sum
    )
    
    if options.csv:
        df_with_flanks_spacers.to_csv(f"./{options.filename}.csv", index=False)
    
    if options.tsv:
        df_with_flanks_spacers.to_csv(f"./{options.filename}.tsv", sep="\t", index=False)

#################################################################


def generate_all_orientation_strings_manual(N):
    
    # generates all orientation strings given the length N
    # orientation strings are all possible permutations of a string made of two characters : >, <
    # assuming that characters of the same type are indinguishable
    
    def _all_unique_variants(N):

        collected_variants = []
        n_possible_variants = 2**N
        n_unique_variants = 0

        for i in range(N+1):
            variant = []
            j = N - i

            # i = number of ">"
            # j = numer of "<"

            for k in range(i):
                variant.append(">")
            for l in range(j):
                variant.append("<")

            # number of permutations is smaller if objects are indistinguishable
            # it as assumed that all >'s and <'s are exactly the same
            # so, the number of permutations has to be divided by number of permutations 
            # within >'s and <'s (separatelly) for each string
            n_unique_variants += math.factorial(N) / (math.factorial(i) * math.factorial(j))

            collected_variants.append(variant)

        assert n_unique_variants == n_possible_variants
        return collected_variants
    
    
    def _shuffle_characters(list_of_characters):

        # returns all possible permutations of a list of characters in a recursive manner

        llen = len(list_of_characters)

        if llen == 0:
            return []

        elif llen == 1:
            return [list_of_characters]

        list_of_unique_permutations = []

        for i in range(llen):
            # keep the i-th character and shuffle everything around
            rest_characters = list_of_characters[:i] + list_of_characters[(i+1):]

            for permutation in _shuffle_characters(rest_characters):
                if [list_of_characters[i]] + permutation not in list_of_unique_permutations:
                    list_of_unique_permutations.append([list_of_characters[i]] + permutation)

        return list_of_unique_permutations
    
    orientation_list = []
    
    all_variants = _all_unique_variants(N)

    for variant in all_variants:
        for permuted_variant in _shuffle_characters(variant):
            orientation_list.append(permuted_variant)
    
    return orientation_list
    
    
def generate_all_orientation_strings(N):

    def _binary_to_orientation_string_map(binary_list):
        
        binary_to_orientation_dict = {0 : ">", 1 : "<"}
        orientation_list = [binary_to_orientation_dict[number] for number in binary_list]
        
        return "".join(orientation_list)        
    
    binary_list = [list(i) for i in itertools.product([0, 1], repeat=N)]
    
    return [_binary_to_orientation_string_map(binary_element) for binary_element in binary_list]
    
    
def add_orientation(seq_coords_df, 
                    nr_strong,
                    nr_weak,
                    mode="same", 
                    orientation_strings=[">>"],
                    N=2):

    df_len = len(seq_coords_df)
        
    if mode == "same":
        seq_coords_df["orientation"] = [">>" for i in range(df_len)]
    
    elif mode == "list":
        if len(orientation_strings) == df_len:
            seq_coords_df["orientation"] = orientation_strings
        else:
            raise ValueError("Check the length of the customized list of orientation strings. Expected length: %s" % df_len)
        
    else:
        # all possible unique orientations of length N 
        
        # manual version
        # orientation_strings = generate_all_orientation_strings_manual(N)
        
        orientation_strings = generate_all_orientation_strings(N)
        
        orientation_ls = []
        rep_unit = seq_coords_df
        
        for ind in range(len(orientation_strings)):
            orientation = orientation_strings[ind]
            orientation_string = "".join(orientation)
            orientation_ls = orientation_ls + [orientation_string for i in range(df_len)]
            if len(seq_coords_df) != len(orientation_ls):
                seq_coords_df = pd.concat([seq_coords_df, rep_unit], ignore_index=True)
            
        seq_coords_df["orientation"] = orientation_ls
        
    seq_coords_df = seq_coords_df.drop(["index"], axis=1)
    
    return seq_coords_df


def add_flanks_and_spacers(
    seq_coords_df,
    flank_range,
    flank_spacer_sum):
    
    l, h = [int(num) for num in flank_range.split(",")]
    
    rep_unit = seq_coords_df
    df_len = len(rep_unit)
    
    flank_ls = []
    spacer_ls = []
    
    for flank in range(l, h):
        spacer = flank_spacer_sum - flank
        flank_ls = flank_ls + [flank for i in range(df_len)]
        spacer_ls = spacer_ls + [spacer for i in range(df_len)]
        
        if len(seq_coords_df) != len(flank_ls):
            seq_coords_df = pd.concat([seq_coords_df, rep_unit], ignore_index=True)
    
    seq_coords_df["flank_lenght"] = flank_ls
    seq_coords_df["spacer_lenght"] = spacer_ls
    
    return seq_coords_df
   
    
def add_background(
        seq_coords_df,
        number_backgrounds
    ):
    
    rep_unit = seq_coords_df
    df_len = len(rep_unit)
    
    background_ls = []
    
    for background_ind in range(number_backgrounds):
        background_ls = background_ls + [background_ind for i in range(df_len)]
        
        if len(seq_coords_df) != len(background_ls):
            seq_coords_df = pd.concat([seq_coords_df, rep_unit], ignore_index=True)
    
    seq_coords_df["background_index"] = background_ls
    
    return seq_coords_df
        
################################################################################
# __main__
################################################################################
if __name__ == "__main__":
    main()
