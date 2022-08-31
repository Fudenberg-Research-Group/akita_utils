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

from __future__ import print_function

from optparse import OptionParser
import json
import os
import pdb
import pickle
import random
import sys
import time
import math

import h5py
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pysam
from skimage.measure import block_reduce
import seaborn as sns

#################
import sys

sys.path.insert(0, "/home1/smaruj/akita_utils/")

# from akita_utils import *
import akita_utils

#################

sns.set(style="ticks", font_scale=1.3)


# """
# seq_coords_df_generationV2.py
#
# Given the parameters the scirpt creates a tsv or csv file - an input to the padding experiment script.
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
    
    print("options: ", options)
    print("args", args)
    
    orient_list = options.orientation_string.split(",")
    
    # print(orient_list)
    
    ### CHECKPOINT ###
    
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
    
    # open genome FASTA
    genome_open = pysam.Fastafile(options.genome_fasta)
    
    # load motifs
    
    seq_coords_df = akita_utils.prepare_insertion_tsv(
        h5_dirs = "/project/fudenber_735/tensorflow_models/akita/v2/analysis/permute_boundaries_motifs_ctcf_mm10_model*/scd.h5",
        score_key = 'SCD',
        pad_flank = 0, #how much flanking sequence around the sites to include
        weak_thresh_pct = 1, # don't use sites weaker than this, might be artifacts
        weak_num = options.num_strong_motifs ,
        strong_thresh_pct = 99, # don't use sites stronger than this, might be artifacts
        strong_num = options.num_weak_motifs ,
        save_tsv=None, # optional filename to save a tsv
    )
        
    out = add_orientation(seq_coords_df, 
                nr_strong = options.num_strong_motifs,
                nr_weak = options.num_weak_motifs,
                mode=options.mode,
                orient_list=orient_list,
                N = options.orientation_N)
    
    if options.csv:
        out.to_csv(f"./{options.filename}.csv", index=False)
    
    if options.tsv:
        out.to_csv(f"./{options.filename}.tsv", sep="\t", index=False)

#################################################################

# functions: all_orient_strings() and permute() will be added later to one of the utils modules

def all_orient_strings(N):
    
    all_strings = []
    n_pos_perms = 2**N
    n_unique_perms = 0
    
    for i in range(N+1):
        string = []
        j = N - i
        
        # i = number of ">"
        # j = numer of "<"
        
        # print(i, j)
        
        for k in range(i):
            string.append(">")
        for l in range(j):
            string.append("<")
        
        # number of permutations is smaller if objects are indistinguishable
        # it as assumed that all >'s and <'s are exactly the same
        # so, the number of permutations has to be divided by number of permutations 
        # within >'s and <'s (separatelly) for each string
        n_unique_perms += math.factorial(N) / (math.factorial(i) * math.factorial(j))
        
        # print(string)
        all_strings.append(string)
    
    assert n_unique_perms == n_pos_perms
    return all_strings

def permute(l):
    
    # returns all possible permutations of a list of characters in a recursive manner
    
    llen = len(l)
    
    if llen == 0:
        return []
    
    elif llen == 1:
        return [l]

    perm_list = []
    
    for i in range(llen):
        rest = l[:i] + l[(i+1):]
        
        for rest_perm in permute(rest):
            if [l[i]] + rest_perm not in perm_list:
                perm_list.append([l[i]] + rest_perm)
    
    return perm_list

def add_orientation(seq_coords_df, 
                    nr_strong,
                    nr_weak,
                    mode="same", 
                    orient_list=[">>"],
                    N=2):
    
    df_len = len(seq_coords_df)
        
    if mode == "same":
        seq_coords_df["orientation"] = [">>" for i in range(df_len)]
    
    elif mode == "customize":
        if len(orient_list) == df_len:
            seq_coords_df["orientation"] = orient_list
        else:
            raise ValueError("Check the length of the customized list of orientation strings. Expected length: %s" % df_len)
        
    else:
        # all possible unique orientations of length N 
        orient_list = []
    
        strings = all_orient_strings(N)

        for ls in strings:
            for perm in permute(ls):
                orient_list.append(perm)
                
        # print(unique_permutations)
        # print(options.orientation_N, len(unique_permutations))
        
        rep_unit = seq_coords_df
        orientation_ls = []
        
        for o in range(len(orient_list)):
            orientation = orient_list[o]
            orientation_string = "".join(orientation)
            # print(orientation)
            orientation_ls = orientation_ls + [orientation_string for i in range(df_len)]
            if len(seq_coords_df) != len(orientation_ls):
                seq_coords_df = pd.concat([seq_coords_df, rep_unit], ignore_index=True)
            
        seq_coords_df["orientation"] = orientation_ls
    
    seq_coords_df = seq_coords_df.drop(["index"], axis=1)
    
    return seq_coords_df

################################################################################
# __main__
################################################################################
if __name__ == "__main__":
    main()
