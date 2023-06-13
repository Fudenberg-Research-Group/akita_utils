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
virtual_symmetric_experiment.py
derived from akita_insert.py (https://github.com/Fudenberg-Research-Group/akita_utils/blob/flank_experiment/bin/akita_insert.py)


This scripts computes insertion scores for motif insertions from a tsv file with:
chrom | start | end | strand | genomic_SCD | orientation | background_index | flank_bp | spacer_bp
where one row represents a single experiment.
The insertion scores are added as next keys in the h5 format file.

The script requires the following input:

Parameters:
-----------
<params_file> - parameters for the akita model
<model_file> - model in the h5 format
<motifs_file> - tsv/csv table specifying designed experiemensts

Options:
-----------
- path to the mouse or human genome in the fasta format
- comma-separated list of statistic scores (stats, e.g. --stats SCD,INS-16)
- head index (depending if predictions are to be made in the mouse (--head_index 0) or human genome-context (--head_index 1))
- model index (same as specified one by the model_file)
- batch size 
- path to the background file (in the fasta format)
- output directory for tables and plots
- flag -m to plot contact map for some of the performed experiments
- (optional, specific for plotting) heatmap plot limit
- (optional, specific for plotting) heatmap plot frequency
- (optional) add option --rc to average forward and reverse complement predictions
- (optional) adding --shifts k ensembles prediction shifts by k


"""

################################################################################
# imports
################################################################################

from __future__ import print_function

from optparse import OptionParser
import json
import os

# import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import pickle
import random

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pysam
import seaborn as sns

sns.set(style="ticks", font_scale=1.3)

import tensorflow as tf

if tf.__version__[0] == "1":
    tf.compat.v1.enable_eager_execution()
gpus = tf.config.experimental.list_physical_devices("GPU")
# for gpu in gpus:
#  tf.config.experimental.set_memory_growth(gpu, True)
print(gpus)

from basenji import seqnn, stream, dna_io

from akita_utils.seq_gens import symmertic_insertion_seqs_gen
from akita_utils.utils import split_df_equally
from akita_utils.h5_utils import initialize_output_h5, write_to_h5_stats_for_prediction


################################################################################
# main
################################################################################
def main():
    usage = "usage: %prog [options] <params_file> <model_file> <motifs_file>"
    parser = OptionParser(usage)
    parser.add_option(
        "-f",
        dest="genome_fasta",
        default=None,
        help="Genome FASTA for sequences [Default: %default]",
    )
    parser.add_option(
        "-l",
        dest="plot_lim_min",
        default=0.1,
        type="float",
        help="Heatmap plot limit [Default: %default]",
    )
    parser.add_option(
        "--plot-freq",
        dest="plot_freq",
        default=100,
        type="int",
        help="Heatmap plot freq [Default: %default]",
    )
    parser.add_option(
        "-m",
        dest="plot_map",
        default=False,
        action="store_true",
        help="Plot contact map for each allele [Default: %default]",
    )
    parser.add_option(
        "-o",
        dest="out_dir",      # to be changed?
        default="./",
        help="Output directory for tables and plots [Default: %default]",
    )
    parser.add_option(
        "-p",
        dest="processes",
        default=None,
        type="int",
        help="Number of processes, passed by multi script",
    )
    parser.add_option(       # reverse complement
        "--rc",
        dest="rc",
        default=False,
        action="store_true",
        help="Average forward and reverse complement predictions [Default: %default]",
    )
    parser.add_option(      # shifts
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
        default=4,
        type="int",
        help="Specify batch size",
    )
    parser.add_option(
        "--head-index",
        dest="head_index",
        default=0,
        type="int",
        help="Specify head index (0=human 1=mus)",
    )
    parser.add_option(
        "--model-index",
        dest="model_index",
        default=0,
        type="int",
        help="Specify model index (from 0 to 7)",
    )
    ## insertion-specific options
    parser.add_option(
        "--background-file",
        dest="background_file",
        default="/project/fudenber_735/tensorflow_models/akita/v2/analysis/background_seqs.fa",
        help="file with insertion seqs in fasta format",
    )

    (options, args) = parser.parse_args()
    
    print("\n++++++++++++++++++\n")
    print("INPUT")
    print("\n++++++++++++++++++\n")
    print("options")
    print(options)
    print("\n++++++++++++++++++\n")
    print("args", args)
    print(args)
    print("\n++++++++++++++++++\n")
    
    if len(args) == 3:
        # single worker
        params_file = args[0]
        model_file = args[1]
        motif_file = args[2]

    elif len(args) == 5:                 # muliti-GPU option
        # multi worker
        options_pkl_file = args[0]
        params_file = args[1]
        model_file = args[2]
        motif_file = args[3]
        worker_index = int(args[4])

        # load options
        options_pkl = open(options_pkl_file, "rb")
        options = pickle.load(options_pkl)
        options_pkl.close()

        # update output directory
        options.out_dir = "%s/job%d" % (options.out_dir, worker_index)

    else:
        parser.error("Must provide parameters and model files and insertion TSV file")

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
    with open(params_file) as params_open:s
        params = json.load(params_open)
    params_train = params["train"]
    params_model = params["model"]

    if options.batch_size is None:
        batch_size = params_train["batch_size"]
    else:
        batch_size = options.batch_size

    if options.targets_file is not None:
        targets_df = pd.read_csv(options.targets_file, sep="\t", index_col=0)
        target_ids = targets_df.identifier
        target_labels = targets_df.description

    #################################################################
    # setup model
    head_index = options.head_index
    model_index = options.model_index
    
    # load model
    seqnn_model = seqnn.SeqNN(params_model)
    seqnn_model.restore(model_file, head_i=head_index)
    seqnn_model.build_ensemble(options.rc, options.shifts)
    seq_length = int(params_model["seq_length"])

    # dummy target info
    if options.targets_file is None:
        num_targets = seqnn_model.num_targets()
        target_ids = [ti for ti in range(num_targets)]      # checkpoint? to be sure that the langth of given targets_file is compatibile with the requested head?
        target_labels = [""] * len(target_ids)

    #################################################################
    # load motifs    
    
    # filter for worker motifs
    if options.processes is not None:                    # multi-GPU option
        # determine boundaries from motif file
        seq_coords_full = pd.read_csv(motif_file, sep="\t")
        seq_coords_df = split_df_equally(seq_coords_full, options.processes, worker_index)
        
    else:
        # read motif positions from csv
        seq_coords_df = pd.read_csv(motif_file, sep="\t")
        
    num_experiments = len(seq_coords_df)
    
    print("===================================")
    print("Number of experiements = ", num_experiments)         # Warning! It's not number of predictions. Num of predictions is this number x5 or x6
    
    # open genome FASTA
    genome_open = pysam.Fastafile(options.genome_fasta)          # needs to be closed at some point

    background_seqs = []
    with open(options.background_file, "r") as f:
        for line in f.readlines():
            if ">" in line:
                continue
            background_seqs.append(dna_io.dna_1hot(line.strip()))
    num_insert_backgrounds = seq_coords_df["background_index"].max()
    if len(background_seqs) < num_insert_backgrounds:
        raise ValueError(
            "must provide a background file with at least as many"
            + "backgrounds as those specified in the insert seq_coords tsv."
            + "\nThe provided background file has {len(background_seqs)} sequences."
        )
            
    #################################################################
    # setup output
    
    scd_out = initialize_output_h5(
        options.out_dir,
        params["model"],
        options.genome_fasta,
        seqnn_model,
        options.scd_stats,
        seq_coords_df
    )

    print("initialized")

    #################################################################
    # predict SNP scores, write output

    # initialize predictions stream
    preds_stream = stream.PredStreamGen(
        seqnn_model, symmertic_insertion_seqs_gen(seq_coords_df, background_seqs, genome_open), batch_size
    )
            
    for exp in range(num_experiments):
        # get predictions
        preds = preds_stream[exp]
        # save stat metrics for each prediction
        write_to_h5_stats_for_prediction(
            preds,
            scd_out,
            exp,
            head_index,
            model_index,
            seqnn_model.diagonal_offset,
            options.scd_stats,
            plot_dir,
            options.plot_lim_min,
            options.plot_freq
        )
    
    genome_open.close()
    scd_out.close()
        
################################################################################
# __main__
################################################################################


if __name__ == "__main__":
    main()