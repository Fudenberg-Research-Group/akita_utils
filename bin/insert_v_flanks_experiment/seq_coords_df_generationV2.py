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
        "--csv",
        dest="csv",
        default=False,
        action="store_true",
        help="Save dataframe as csv",
    )
    parser.add_option(
        "--tsv",
        dest="tsv",
        default=False,
        action="store_true",
        help="Save dataframe as tsv",
    )
    parser.add_option(
        "--h5",
        dest="h5",
        default=False,
        action="store_true",
        help="Save dataframe as h5",
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
                orient_list=orient_list)
    
    num_options = len(out)
    
    print("Length of data frame to be prepared: ", num_options)
    
    if options.csv:
        out.to_csv(f"./{options.filename}.csv", index=False)
    
    if options.tsv:
        out.to_csv(f"./{options.filename}.tsv", sep="\t", index=False)
    
    if options.h5:
        save_h5(out, filename=f"./{options.filename}.h5")

#################################################################
# adding orientation

def add_orientation(seq_coords_df, 
                    nr_strong,
                    nr_weak,
                    mode="same", 
                    orient_list=[">>"]):
    
    df_len = len(seq_coords_df)
    
    # mode_types = ["same", "customize", "weak_strong", "for_each"]
    # if mode not in mode_types:
    #     raise ValueError("Invalid mode. Expected one of: %s" % mode_types)
    
    if mode == "same":
        seq_coords_df["orientation"] = [">>" for i in range(df_len)]
    
    elif mode == "customize":
        if len(orient_list) == df_len:
            seq_coords_df["orientation"] = orient_list
        else:
            raise ValueError("Check the length of the customized list of orientation strings. Expected length: %s" % df_len)
    
    elif mode == "weak_strong":
        
        if len(orient_list) != 2:
            raise ValueError("You should provide a list of two orientation strings: first of strong motifs, second - for weak")
        
        strong_orientation = orient_list[0]
        weak_orientation = orient_list[1]
        
        real_orient_list = [strong_orientation for j in range(nr_strong)] + [weak_orientation for k in range(nr_weak)] 
        
        if len(real_orient_list) == df_len:
            seq_coords_df["orientation"] = real_orient_list
        else:
            raise ValueError("Check the numeber of weak and strong sites given. Expected sum of those two numbers is: %s" % df_len)
    
    elif mode == "for_each":
        rep_unit = seq_coords_df
        orientation_ls = []
        
        for o in range(len(orient_list)):
            orientation = orient_list[o]
            orientation_ls = orientation_ls + [orientation for i in range(df_len)]
            if len(seq_coords_df) != len(orientation_ls):
                seq_coords_df = pd.concat([seq_coords_df, rep_unit], ignore_index=True)
            
        seq_coords_df["orientation"] = orientation_ls
    
    seq_coords_df = seq_coords_df.drop(["index"], axis=1)
    
    return seq_coords_df

#################################################################
# saving in the h5 format

def save_h5(seq_coords_df, filename="out.h5"):
    chrom = seq_coords_df.chrom.to_numpy()
    start = seq_coords_df.start.to_numpy().astype(np.int32)
    end = seq_coords_df.end.to_numpy().astype(np.int32)
    strand = seq_coords_df.strand.to_numpy()
    genSCD = seq_coords_df.genomic_SCD.to_numpy().astype(np.float64)
    orientation = seq_coords_df.orientation.to_numpy()
    
    with h5py.File("./" + filename, "w") as hf:
        hf.create_dataset("Chromosome",  data=chrom.astype('S'))
        hf.create_dataset("Start",  data=start)
        hf.create_dataset("End",  data=end)
        hf.create_dataset("Strand",  data=strand.astype('S'))
        hf.create_dataset("genomic_SCD",  data=genSCD)
        hf.create_dataset("orientation",  data=orientation.astype('S'))

################################################################################
# __main__
################################################################################
if __name__ == "__main__":
    main()
