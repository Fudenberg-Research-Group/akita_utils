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
virtual_symmetric_experiment_compare_SCD.py
derived from virtual_symmetric_experiment.py


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
- head index (depending if predictions are to be made in the human (--head_index 0) or mouse genome-context (--head_index 1))
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

import pickle
import random

import numpy as np
import pandas as pd
import pysam

import tensorflow as tf

import h5py
from datetime import date

if tf.__version__[0] == "1":
    tf.compat.v1.enable_eager_execution()
gpus = tf.config.experimental.list_physical_devices("GPU")

print(gpus)

from basenji import seqnn, stream, dna_io

from akita_utils.utils import ut_dense
from akita_utils.seq_gens import (
    symmertic_insertion_seqs_gen,
    reference_seqs_gen,
)
from akita_utils.utils import split_df_equally
from akita_utils.h5_utils import initialize_output_h5, write_stat_metrics_to_h5


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
        dest="out_dir",  # to be changed?
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
    parser.add_option(  # reverse complement
        "--rc",
        dest="rc",
        default=False,
        action="store_true",
        help="Average forward and reverse complement predictions [Default: %default]",
    )
    parser.add_option(  # shifts
        "--shifts",
        dest="shifts",
        default="0",
        type="str",
        help="Ensemble prediction shifts [Default: %default]",
    )
    parser.add_option(
        "--stats",
        dest="scd_stats",
        default="SCD,diffSCD",
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

    elif len(args) == 5:  # muliti-GPU option
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
        parser.error(
            "Must provide parameters and model files and insertion TSV file"
        )

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
        target_ids = [
            ti for ti in range(num_targets)
        ]  # checkpoint? to be sure that the langth of given targets_file is compatibile with the requested head?
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

    num_experiments = len(seq_coords_df)

    print("===================================")
    print(
        "Number of experiements = ", num_experiments
    )  # Warning! It's not number of predictions. Num of predictions is this number x5 or x6

    # open genome FASTA
    genome_open = pysam.Fastafile(
        options.genome_fasta
    )  # needs to be closed at some point

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
        model_file,
        options.genome_fasta,
        seqnn_model,
        options.scd_stats,
        seq_coords_df,
        options.plot_map
    )

    print("initialized")

    #################################################################
    # predict SCD scores

    num_backgrounds = len(background_seqs)

    # initialize predictions stream for reference (background) sequences
    refs_stream = stream.PredStreamGen(
        seqnn_model, reference_seqs_gen(background_seqs), batch_size
    )

    background_preds_maps = np.zeros((num_backgrounds, 512, 512, num_targets))

    for background_index in range(num_backgrounds):
        
        bg_preds_matrix = refs_stream[background_index]
        bg_map_matrix = ut_dense(bg_preds_matrix, seqnn_model.diagonal_offset)
        
        background_preds_maps[background_index, :, :, :] = bg_map_matrix
        
        if options.plot_map:
            write_maps_to_h5(
                bg_map_matrix,
                scd_out,
                background_index,
                head_index,
                model_index,
                reference=True
            )

    # initialize predictions stream for alternate (ctcf-inserted) sequences
    preds_stream = stream.PredStreamGen(
        seqnn_model,
        symmertic_insertion_seqs_gen(
            seq_coords_df, background_seqs, genome_open
        ),
        batch_size,
    )
    
    for exp in range(num_experiments):
        # get predictions
        preds_matrix = preds_stream[exp]
        map_matrix = ut_dense(preds_matrix, seqnn_model.diagonal_offset)
        background_index = seq_coords_df.iloc[exp].background_index
    
        # save stat metrics for each prediction
        write_stat_metrics_to_h5(
            preds_matrix,
            background_preds_maps[background_index, :, :, :],
            scd_out,
            exp,
            head_index,
            model_index,
            seqnn_model.diagonal_offset,
            options.scd_stats,
        )
        
        if options.plot_map:
            write_maps_to_h5(
                map_matrix,
                scd_out,
                exp,
                head_index,
                model_index
            )
        
    genome_open.close()
    scd_out.close()
    
    
def initialize_output_h5(out_dir, 
                         model_file, 
                         genome_fasta,
                         seqnn_model,
                         stat_metrics,
                         seq_coords_df,
                        save_maps=False,
                        map_size=512):
    """
    Initializes an h5 file to save statistical metrics calculated from Akita's predicftions.

    Parameters
    ------------
    out_dir : str
        Path to the desired location of the output h5 file.
    model_file : str
        Path to the model file.
    genome_fasta : str
        Path to the genome file (mouse or human).
    seqnn_model : object
        Loaded model.
    stat_metrics : list
        List of stratistical metrics that are supposed to be calculated.
    seq_coords_df : dataFrame
        Pandas dataframe where each row represents one experiment (so one set of prediction).
    save_maps : Boolean
        Stores True, if maps are supposed to be saved.
    map_size : int
        Size of a predicted (obs/exp) map matrix.
        
    Returns
    ---------
    h5_outfile : h5py object
        An initialized h5 file.
    """
    
    h5_outfile = h5py.File(f"%s/OUT.h5" % out_dir, "w")
    seq_coords_df_dtypes = seq_coords_df.dtypes
    
    head_index = int(model_file.split("model")[-1][0])
    model_index = int(model_file.split("c0")[0][-1]) 
    
    h5_outfile.attrs["date"] = str(date.today())
    
    num_backgrounds = len(seq_coords_df.background_index.unique())
    
    metadata_dict = {
        "model_index" : model_index,
        "head_index" : head_index,
        "genome" : genome_fasta.split("/")[-1],
        "seq_length" : seqnn_model.seq_length,
        "diagonal_offset" : seqnn_model.diagonal_offset,             
        "prediction_vector_length" : seqnn_model.target_lengths[0],
        "target_crops" : seqnn_model.target_crops,
        "num_targets" : seqnn_model.num_targets()}
    
    h5_outfile.attrs.update(metadata_dict)
    
    num_targets = seqnn_model.num_targets()
    target_ids = [ti for ti in range(num_targets)]   
                                   
    num_experiments = len(seq_coords_df)

    for key in seq_coords_df:
        if seq_coords_df_dtypes[key] is np.dtype("O"):
            h5_outfile.create_dataset(
                key, data=seq_coords_df[key].values.astype("S")
            )
        else:
            h5_outfile.create_dataset(key, data=seq_coords_df[key])
    
    if save_maps:
        h5_outfile.create_dataset(
                f"map_h{head_index}_m{model_index}",
                shape=(num_experiments, map_size, map_size, num_targets),
                dtype="float16"
            )
        
        h5_outfile.create_dataset(
                f"refmap_h{head_index}_m{model_index}",
                shape=(num_backgrounds, map_size, map_size, num_targets),
                dtype="float16"
            )
    
    # initialize keys for statistical metrics collection
    for stat_metric in stat_metrics:

        if stat_metric in seq_coords_df.keys():
            raise KeyError("check input tsv for clashing score name")

        for target_index in target_ids:
            h5_outfile.create_dataset(
                f"{stat_metric}_h{head_index}_m{model_index}_t{target_index}",
                shape=(num_experiments,),
                dtype="float16",
                compression=None,
            )

    return h5_outfile  

   
def write_maps_to_h5(
    map_matrix,
    h5_outfile,
    experiment_index,
    head_index,
    model_index,
    diagonal_offset=2,
    reference=False
):
    """
    Writes entire maps to an h5 file.

    Parameters
    ------------
    map_matrix : numpy matrix
        Matrix collecting Akita's prediction maps. Shape: (map_size, map_size, num_targets).
    h5_outfile : h5py object
        An initialized h5 file.
    experiment_index : int
        Index identifying one experiment.
    head_index : int
        Head index used to get a prediction (Mouse: head_index=1; Human: head_index=0).
    model_index : int
        Index of one of 8 models that has been used to make predictions (an index between 0 and 7).
    diagonal_offset : int
        Number of diagonals that are added as zeros in the conversion.
        Typically 2 diagonals are ignored in Hi-C data processing.
    """
    prefix = "map"
    if reference:
        prefix = "refmap"

    for target_index in range(map_matrix.shape[-1]):
        h5_outfile[f"{prefix}_h{head_index}_m{model_index}"][experiment_index, :, :, target_index] += map_matrix[:, :, target_index]


################################################################################
# __main__
################################################################################


if __name__ == "__main__":
    main()
    
    