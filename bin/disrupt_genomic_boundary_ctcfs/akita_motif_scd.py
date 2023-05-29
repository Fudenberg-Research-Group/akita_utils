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
import re
import h5py
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pysam
from skimage.measure import block_reduce
import seaborn as sns
from akita_utils.stats_utils import insul_diamonds_scores
import akita_utils.h5_utils

sns.set(style="ticks", font_scale=1.3)

import tensorflow as tf

if tf.__version__[0] == "1":
    tf.compat.v1.enable_eager_execution()
gpus = tf.config.experimental.list_physical_devices("GPU")
# for gpu in gpus:
#  tf.config.experimental.set_memory_growth(gpu, True)
print(gpus)

from basenji import seqnn
from basenji import stream
from basenji import dna_io

from akita_utils.utils import ut_dense, split_df_equally
from akita_utils.seq_gens import disruption_seqs_gen

import logging

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
log = logging.getLogger(__name__)

"""
akita_motif_scd.py

Compute Squared Contact Difference (SCD) scores for motifs in TSV motif file.

"""


################################################################################
# main
################################################################################
def main():
    usage = "usage: %prog [options] <params_file> <model_file> <tsv_file>"
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
        dest="out_dir",
        default="scd",
        help="Output directory for tables and plots [Default: %default]",
    )
    parser.add_option(
        "-p",
        dest="processes",
        default=None,
        type="int",
        help="Number of processes, passed by multi script",
    )
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
    parser.add_option(
        "--model-index",
        dest="model_index",
        default=None,
        type="int",
        help="Specify head index (0=human 1=mus) ",
    )
    parser.add_option(
        "--mut-method",
        dest="mutation_method",
        default="mask",
        type="str",
        help="specify mutation method",
    )

    parser.add_option(
        "--use-span",
        dest="use_span",
        default=False,
        action="store_true",
        help="specify if using spans",
    )

    parser.add_option(
        "--motif-width", dest="motif_width", default=18, type="int", help="motif width"
    )
    (options, args) = parser.parse_args()

    if len(args) == 3:
        # single worker
        params_file = args[0]
        model_file = args[1]
        motif_file = args[2]
        
    elif len(args) == 5:
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
        parser.error("Must provide parameters and models file and TSV file")
        
    if options.model_index is None:
        pattern = r"/f(\d+)c0/"  # pattern to match the model_index
        match = re.search(pattern, model_dir) 
        if match:
            options.model_index = int(match.group(1))  # Extract the model_index from the matched group
        else:
            print("Could not extract model index from given model file, and the model index was not provided by user. please specify the model index i.e --model-index")
            exit(1)

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

    if options.batch_size is None:
        batch_size = params_train["batch_size"]
    else:
        batch_size = options.batch_size

    mutation_method, motif_width, use_span = (
        options.mutation_method,
        options.motif_width,
        True,
    )

    if not mutation_method in ["mask", "permute"]:
        raise ValueError("undefined mutation method:", mutation_method)

    if options.use_span:
        log.info(f"using SPANS")
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
    # load motifs

    # filter for worker motifs
    if options.processes is not None:  # multi-GPU option
        # determine boundaries from motif file
        seq_coords_full = pd.read_csv(motif_file, sep="\t")
        seq_coords_df = split_df_equally(
            seq_coords_full, options.processes, worker_index
        )

    else:
        # read motif positions from csv
        seq_coords_df = pd.read_csv(motif_file, sep="\t")

    num_motifs = len(seq_coords_df)

    log.info("====================================================")
    log.info(
        f"This script will run {num_motifs} experiments, each experiment has two predictions i.e reference and alternate"
    )
    log.info(f"OPTIONS {options}")
    log.info("====================================================")

    # open genome FASTA
    genome_open = pysam.Fastafile(options.genome_fasta)

    #################################################################
    # setup output
    scd_out = akita_utils.h5_utils.initialize_output_h5_v2(
    options.out_dir,
    options.scd_stats,
    seq_coords_df,
    target_ids,
    target_labels,
    options.head_index,
    options.model_index,
)

    #################################################################
    # predict SNP scores, write output

    # initialize predictions stream
    preds_stream = stream.PredStreamGen(
        seqnn_model,
        disruption_seqs_gen(
            seq_coords_df,
            mutation_method,
            motif_width,
            seq_length,
            genome_open,
            use_span,
        ),
        batch_size,
    )

    # predictions index
    pi = 0

    for si in range(num_motifs):
        # get predictions
        ref_preds = preds_stream[pi]
        pi += 1
        alt_preds = preds_stream[pi]
        pi += 1

        # process SNP
        akita_utils.h5_utils.write_snp_v2(
        ref_preds,
        alt_preds,
        scd_out,
        si,
        options.head_index,
        options.model_index,
        seqnn_model.diagonal_offset,
        options.scd_stats,
        plot_dir,
        options.plot_lim_min,
        options.plot_freq,
        )

    genome_open.close()
    scd_out.close()
    
################################################################################
# __main__
################################################################################
if __name__ == "__main__":
    main()
