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

import tensorflow as tf

if tf.__version__[0] == "1":
    tf.compat.v1.enable_eager_execution()
gpus = tf.config.experimental.list_physical_devices("GPU")
print(gpus)

from basenji import seqnn
from basenji import stream
from basenji import dna_io

# from basenji import vcf as bvcf

"""
PS_akita_insert.py

Compute SNP Contact Difference (SCD) scores, and INS scores for motif insertions with different paddings from a tsv file with chrom, start, end, strand.

"""

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
        "-v",
        dest="verbose",
        default=False,
        action="store_true",
        help="If true, the stat metrics are printed out",
    )
    # parser.add_option(
    #     "-l",
    #     dest="plot_lim_min",
    #     default=0.1,
    #     type="float",
    #     help="Heatmap plot limit [Default: %default]",
    # )
    # parser.add_option(
    #     "--plot-freq",
    #     dest="plot_freq",
    #     default=100,
    #     type="int",
    #     help="Heatmap plot freq [Default: %default]",
    # )
    parser.add_option(
        "-m",
        dest="plot_map",
        default=False,
        action="store_true",
        help="Plot contact map for each allele [Default: %default]",
    )
    parser.add_option(
        "-o",
        dest="out_dir",
        default="scd",
        help="Output directory for tables and plots [Default: %default]",
    )
    # parser.add_option(
    #     "-p",
    #     dest="processes",
    #     default=None,
    #     type="int",
    #     help="Number of processes, passed by multi script",
    # )
    parser.add_option(
        "--rc",
        dest="rc",
        default=False,
        action="store_true",
        help="Average forward and reverse complement predictions [Default: %default]",
    )
    parser.add_option(
        "--shifts",
        dest="shifts",
        default="0",
        type="str",
        help="Ensemble prediction shifts [Default: %default]",
    )
    parser.add_option(
        "--stats",
        dest="scd_stats",
        default="SCD",
        help="Comma-separated list of stats to save. [Default: %default]",
    )
    parser.add_option(
        "-t",
        dest="targets_file",
        default=None,
        type="str",
        help="File specifying target indexes and labels in table format",
    )
    parser.add_option(
        "--batch-size",
        dest="batch_size",
        default=None,
        type="int",
        help="Specify batch size",
    )
    parser.add_option(
        "--head-index",
        dest="head_index",
        default=0,
        type="int",
        help="Specify head index (0=human 1=mus) ",
    )

    ## insertion-specific options
    parser.add_option(
        "--background-file",
        dest="background_file",
        default=None,
        help="file with insertion seqs in fasta format",
    )
    parser.add_option(
        "--one_side_radius",
        dest="one_side_radius",
        default=0,
        type="int",
        help="Specify radius around motif without overlap with next CTCF",
    )
    parser.add_option(
        "--table",
        dest="table",
        default=None,
        help="Specify a path to the tsv/csv table",
    )
        
    parser.add_option(
        "--num_background",
        dest="num_background",
        default=1,
        type="int",
        help="Specify number of background sequences",
    )
    
    parser.add_option(
        "--paddings_start",
        dest="paddings_start",
        default=0,
        type="int",
        help="Specify paddings at the start",
    )
    
    parser.add_option(
        "--paddings_end",
        dest="paddings_end",
        default=2,
        type="int",
        help="Specify paddings at the end",
    )
    

    (options, args) = parser.parse_args()
    
    # print(options)
    # print(args)
    
    if len(args) == 2:
        # single worker
        params_file = args[0]
        model_file = args[1]
        # motif_file = args[2]

    elif len(args) == 4:
        # multi worker
        options_pkl_file = args[0]
        params_file = args[1]
        model_file = args[2]
        # motif_file = args[3]
        worker_index = int(args[3])

        # load options
        options_pkl = open(options_pkl_file, "rb")
        options = pickle.load(options_pkl)
        options_pkl.close()

        # update output directory
        options.out_dir = "%s/job%d" % (options.out_dir, worker_index)

    else:
        parser.error("Must provide parameters and model files and QTL VCF file")

    if not os.path.isdir(options.out_dir):
        os.mkdir(options.out_dir)
    if options.plot_map:
        plot_dir = options.out_dir
    else:
        plot_dir = None

    options.shifts = [int(shift) for shift in options.shifts.split(",")]
    options.scd_stats = options.scd_stats.split(",")

    random.seed(44)

    #################################################################
    # read parameters and targets

    # read model parameters
    with open(params_file) as params_open:
        params = json.load(params_open)
    params_train = params["train"]
    params_model = params["model"]
    seq_length = params_model["seq_length"]
    
    ###### start : added ######
    hic_diags = params_model["diagonal_offset"]
    try:
        target_crop = params_model["trunk"][-2]["cropping"]
    except:
        target_crop = params_model["target_crop"]
    target_length_cropped = int((seq_length//2048 - target_crop*2 - hic_diags) * ((seq_length//2048 - target_crop*2 - hic_diags) +1)/2) 
    target_map_size = seq_length//2048  - target_crop*2
    
    ###### end : added ######
    
    if options.batch_size is None:
        batch_size = params_train["batch_size"]
    else:
        batch_size = options.batch_size
    print(batch_size)

    if options.targets_file is not None:
        targets_df = pd.read_csv(options.targets_file, sep="\t", index_col=0)
        target_ids = targets_df.identifier
        target_labels = targets_df.description

    #################################################################
    # setup model

    # load model
    seqnn_model = seqnn.SeqNN(params_model)
    seqnn_model.restore(model_file, head_i=options.head_index)
    seqnn_model.build_ensemble(options.rc, options.shifts)
    seq_length = int(params_model["seq_length"])

    # dummy target info
    if options.targets_file is None:
        num_targets = seqnn_model.num_targets()
        target_ids = ["t%d" % ti for ti in range(num_targets)]
        target_labels = [""] * len(target_ids)

    #################################################################
    
    # open genome FASTA
    genome_open = pysam.Fastafile(options.genome_fasta)

    background_seqs = []
    with open(options.background_file, "r") as f:
        for line in f.readlines():
            if ">" in line:
                continue
            background_seqs.append(dna_io.dna_1hot(line.strip()))

    # print(len(background_seqs))
    
    background_seqs = background_seqs[0:options.num_background]
    
    # load motifs info
    
    table_path = options.table
    if table_path[-3:] == "csv":
        seq_coords_df = pd.read_csv(table_path)
    elif table_path[-3:] == "tsv":
        seq_coords_df = pd.read_csv(table_path, sep="\t")
        
    num_seqs = len(seq_coords_df)
    
    print("Number of all sequences to be tested: ", num_seqs)
    
    pad_list = [k for k in range(options.paddings_start, options.paddings_end)]
    
    # print(options.scd_stats)
    
    out = multiple_padding(seq_coords_df, background_seqs, pad_list, target_map_size, hic_diags,
                           seq_length=seq_length,
                           genome_open=genome_open,
                           seqnn_model=seqnn_model,
                           batch_size = options.batch_size,
                           out_dir=plot_dir,
                           head = options.head_index,
                           stat = options.scd_stats,
                           verbose=True,
                           plotting=options.plot_map, 
                           one_side_radius=options.one_side_radius, 
                           motif_len=19)

    save_h5(seq_coords_df=seq_coords_df, 
            out_dir=options.out_dir, 
            stat=options.scd_stats, 
            prediction=out)

#################################################################
# matrics (SCD, INS) caculation

def _insul_diamond_central(mat, window=10):
    """calculate insulation in a diamond around the central pixel"""
    N = mat.shape[0]
    if window > N // 2:
        raise ValueError("window cannot be larger than matrix")
    mid = N // 2
    lo = max(0, mid + 1 - window)
    hi = min(mid + window, N)
    score = np.nanmean(mat[lo : (mid + 1), mid:hi])
    return score


def plot_for_target(predictions, target_map_size, hic_diags, seq_nr, padding, out_dir, size=7, target = 0, vlim = .5, window = 50):
    
    nr_bg = predictions.shape[0]
    bin_mid = target_map_size//2
    
    plt.figure(figsize=(size*nr_bg, size))

    for i in range(nr_bg):
        insert_pred = predictions[i]

        plt.subplot(1, nr_bg, i+1)
        im = plt.matshow(
                akita_utils.from_upper_triu(  
                insert_pred[:,target], target_map_size, hic_diags),
                vmin=-1*vlim, vmax=vlim, fignum=False,cmap='RdBu_r')
        plt.colorbar(im, fraction=0.046, pad=0.04)
        plt.title('insert-scd: '+str(  np.sqrt( (insert_pred**2).sum(axis=0)  )[target] )
                 + '\n background: '+str(i)) 
    plt.tight_layout()
    # plt.close(fig)
    plt.savefig(out_dir + "/" + f"Seq-{seq_nr}_pad-{padding}_target{target}.png")
    # plt.close(fig)

#################################################################
# prediction for multiple paddings, background, sequences, targets, stat metrics
    
def multiple_padding(seq_coords_df, 
                     background_seqs, 
                     pad_list,
                     target_map_size, 
                     hic_diags,
                     seq_length,
                     genome_open,
                     seqnn_model,
                     batch_size,
                     out_dir,
                     head = 1, #mouse
                     stat = ["SCD"],
                     verbose=True,
                     plotting=False, 
                     one_side_radius=100, 
                     motif_len=19):
    
    def split_arrows(string):
        char_list = []
        for char in string:
            char_list.append(char)
        return char_list
        
    nr_seq = len(seq_coords_df)
    nr_bg = len(background_seqs)
    
    nr_targets = 6
    
    if head == 0:  # human -> 5 targets
        nr_targets = 5
    
    nr_pads = len(pad_list)
        
    INS_windows = []
                
    for stat_matric in stat:
        if stat_matric[:3] == "INS":
            window = int(stat_matric.split("-")[1])
            INS_windows.append(window)
    
    INS_all_paddings = {}
    for window in INS_windows:
        INS_all_paddings[window] = np.zeros(shape=(nr_pads, nr_seq, nr_bg, nr_targets))
                
    # if "SCD" in stat_matric:
    SCD_all_paddings = np.zeros(shape=(nr_pads, nr_seq, nr_bg, nr_targets))
    
    for p in range(len(pad_list)):
        
        padding = pad_list[p]
        
        print(f"padding = {padding} || Started working on padding = {padding}")
                
        INS_all_sequences = {}
        for window in INS_windows:
            INS_all_sequences[window] = np.zeros(shape=(nr_seq, nr_bg, nr_targets))
                
        # if "SCD" in stat_matric:
        SCD_all_sequences = np.zeros(shape=(nr_seq, nr_bg, nr_targets))
        
        
        for i in seq_coords_df.index:      #iteraton over sequences
            
            orientation_string = seq_coords_df.iloc[i].orientation
            orientation_list = split_arrows(orientation_string)
            num_inserts = len(orientation_string)
            
            # print(orientation_list)
            # print(orientation_string)
            # print(type(orientation_string))
            # print(num_inserts)
            
            spacer = one_side_radius - padding
            
            multi_insert_length = num_inserts * (motif_len + 2 * one_side_radius)
            
            offsets = []

            for i in range(num_inserts):
                offsets.append( seq_length//2 - multi_insert_length//2 + i * (multi_insert_length//2) + spacer)
            
            # print(offsets)
            
            all_inserts = []

            for bi in range(len(background_seqs)):
                seq_1hot = background_seqs[bi].copy()
                this_row = seq_coords_df.iloc[i]
                
                # CTCF motif in the ">" orientation (heading right)
                seq_1hot_CTCF_pad_right = dna_io.dna_1hot(genome_open.fetch(this_row.chrom, this_row.start-padding, this_row.end+padding).upper())

                if this_row.strand == '-': 
                    seq_1hot_CTCF_pad_right = dna_io.hot1_rc(seq_1hot_CTCF_pad_right)
                
                # CTCF motif in the "<" orientation (headin left)
                seq_1hot_CTCF_pad_left = dna_io.hot1_rc(seq_1hot_CTCF_pad_right)
                
                for o in range(len(offsets)):
                    offset = offsets[o]
                    
                    if orientation_list[o] == ">":
                        seq_1hot[offset : offset + len(seq_1hot_CTCF_pad_right)] = seq_1hot_CTCF_pad_right
                    
                    else: # "<"
                        seq_1hot[offset : offset + len(seq_1hot_CTCF_pad_left)] = seq_1hot_CTCF_pad_left
                    
                all_inserts.append(seq_1hot)
                
            all_inserts = np.array(all_inserts) 

            print(all_inserts.shape) 

            pred = seqnn_model.predict(all_inserts, batch_size=10)  

            print(pred.shape) 
            
            INS_all_backgrounds = {}
            for window in INS_windows:
                INS_all_backgrounds[window] = np.zeros(shape=(nr_bg, nr_targets))
                
            # if "SCD" in stat_matric:
            SCD_all_backgrounds = np.zeros(shape=(nr_bg, nr_targets))
            
            
            for bi in range(nr_bg):      # iteration over background sequences
                
                insert_pred = pred[bi]
                
                if verbose == True:
                
                    print("\n**********")
                    print("Padding: ", padding, "\nSequence: ", i, "\nBackground: ", bi, "\nSCD averaged over targets: ", np.sqrt( (insert_pred**2).sum(axis=0)  ).mean())
                    print("**********\n")

                    print("Metrics: ")
                                                
                INS_all_targets = {}
                for window in INS_windows:
                    INS_all_targets[window] = np.zeros(insert_pred.shape[1])
                
                # if "SCD" in stat_matric:
                SCD_all_targets = np.zeros(insert_pred.shape[1])
                
                for target_ind in range(insert_pred.shape[1]):       # iteration over targets

                    # print(target_ind, "SCD: ", np.sqrt( (insert_pred**2).sum(axis=0))[target_ind])
                    
                    INS_dict = {}
                    mat = akita_utils.from_upper_triu(insert_pred[:,target_ind], target_map_size, hic_diags)
                    
                    for stat_matric in stat:
                                            
                        if stat_matric == "SCD":
                    
                            SCD = np.sqrt( (insert_pred**2).sum(axis=0))[target_ind]      # sequence = i background = bi, target = target_ind
                            SCD_all_targets[target_ind] = SCD
                            
                            if verbose == True:
                                print("target_ind: ", target_ind, "SCD: ", np.sqrt( (insert_pred**2).sum(axis=0))[target_ind])
                        
                        elif stat_matric[:3] == "INS":
                            
                            window = int(stat_matric.split("-")[1])
                            
                            INS_all_targets[window][target_ind] = _insul_diamond_central(mat, window=window)
                            
                            if verbose == True:
                                print("target_ind: ", target_ind, f"INS-{window}: ", INS_all_targets[window][target_ind])
                
                # print(SCD_all_targets)
                # print(INS_all_targets)
                
                for window in INS_windows:
                    INS_all_backgrounds[window][bi] = INS_all_targets[window]
                
                SCD_all_backgrounds[bi] = SCD_all_targets
                
            print(f"padding = {padding} || Plotting")
            
            if plotting == True:
                
                print(f"padding = {padding} || Plotting")
                
                for target_i in range(pred.shape[2]):
                    plot_for_target(pred, target_map_size=target_map_size, hic_diags=hic_diags, seq_nr=i, padding=padding, out_dir=out_dir, size=15, target=target_i)
                                    
            # print(SCD_all_backgrounds)
            # for window in INS_windows:
            #     print(INS_all_backgrounds[window])
            
            for window in INS_windows:
                INS_all_sequences[window][i] = INS_all_backgrounds[window]
                
            SCD_all_sequences[i] = SCD_all_backgrounds
                
        # print("INS_all_sequences ", INS_all_sequences)
        # print("SCD_all_sequences ", SCD_all_sequences)
        
        
        for window in INS_windows:
            INS_all_paddings[window][p] = INS_all_sequences[window]
                
        SCD_all_paddings[p] = SCD_all_sequences
        
    # print(SCD_all_paddings.shape)
    # for window in INS_windows:
    #     print(INS_all_paddings[window].shape)
    
    return(SCD_all_paddings, INS_all_paddings)
    
#################################################################
# saving in the h5 format

def save_h5(seq_coords_df, out_dir, stat, prediction, filename="out.h5"):
    chrom = seq_coords_df.chrom.to_numpy()
    start = seq_coords_df.start.to_numpy().astype(np.int32)
    end = seq_coords_df.end.to_numpy().astype(np.int32)
    strand = seq_coords_df.strand.to_numpy()
    genSCD = seq_coords_df.genomic_SCD.to_numpy().astype(np.float64)
    
    with h5py.File(out_dir + "/" + filename, "w") as hf:
        hf.create_dataset("SCD",  data=prediction[0])
        for key in prediction[1]:
            hf.create_dataset("INS-" + str(key),  data=prediction[1][key])
        hf.create_dataset("Chromosome",  data=chrom.astype('S'))
        hf.create_dataset("Start",  data=start)
        hf.create_dataset("End",  data=end)
        hf.create_dataset("Strand",  data=strand.astype('S'))
        hf.create_dataset("genomic_SCD",  data=genSCD)

################################################################################
# __main__
################################################################################
if __name__ == "__main__":
    main()
