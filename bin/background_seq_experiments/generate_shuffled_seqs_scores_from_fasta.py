#!/usr/bin/env python

# =========================================================================
from __future__ import print_function


import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
log = logging.getLogger(__name__)

from optparse import OptionParser
import json
import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"
import pickle
import random
import h5py
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pysam
from skimage.measure import block_reduce
import seaborn as sns
sns.set(style='ticks', font_scale=1.3)

import tensorflow as tf
if tf.__version__[0] == '1':
  tf.compat.v1.enable_eager_execution()
gpus = tf.config.experimental.list_physical_devices('GPU')
#for gpu in gpus:
#  tf.config.experimental.set_memory_growth(gpu, True)
log.info(gpus)

from basenji import seqnn, stream, dna_io
import akita_utils
from akita_utils.utils import ut_dense, split_df_equally 
from akita_utils.seq_gens import fasta_shuffled_seq_gen
'''
generating scores for shuffled seqs from fasta

'''
################################################################################
# main
# This script generates scores for the shuffled seqs in the input fasta file
################################################################################
def main():
    usage = "usage: %prog [options] <params_file> <model_file> <shuffled_seqs_fasta_file>"
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
        default="SCD,MSS,MPS,CS",
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
        default=1,
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


    (options, args) = parser.parse_args()
    
    log.info("\n++++++++++++++++++\n")
    log.info("INPUT")
    log.info("\n++++++++++++++++++\n")
    log.info(f"options \n {options}")
    log.info("\n++++++++++++++++++\n")
    log.info(f"args \n {args}")
    log.info("\n++++++++++++++++++\n")
    
    if len(args) == 3:
        # single worker
        params_file = args[0]
        model_file = args[1]
        shuffled_seqs_fasta_file = args[2]

    elif len(args) == 5:                 # muliti-GPU option
        # multi worker
        options_pkl_file = args[0]
        params_file = args[1]
        model_file = args[2]
        shuffled_seqs_fasta_file = args[3]
        worker_index = int(args[4])

        # load options
        options_pkl = open(options_pkl_file, "rb")
        options = pickle.load(options_pkl)
        options_pkl.close()

        # update output directory
        options.out_dir = "%s/job%d" % (options.out_dir, worker_index)

    else:
        parser.error("Must provide parameters and model files and insertion fasta file")

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
    seqnn_model.restore(model_file, head_i = options.head_index)
    seqnn_model.build_ensemble(options.rc, options.shifts)
    seq_length = int(params_model['seq_length'])

    # dummy target info
    if options.targets_file is None:
        num_targets = seqnn_model.num_targets()
        target_ids = ['t%d' % ti for ti in range(num_targets)]
        target_labels = ['']*len(target_ids)

        
    #################################################################
    # setup output

    scd_out = initialize_output_h5(
        options.out_dir,
        options.scd_stats,
        shuffled_seqs_fasta_file,
        target_ids,
        target_labels,
        head_index,
        model_index
    )

    log.info("initialized")

        
    seq_IDs = [line for line in open(shuffled_seqs_fasta_file) if line.startswith(">")]
    num_experiments = num = len(seq_IDs)
    #################################################################
    # predict SNP scores, write output

    # initialize predictions stream
    preds_stream = stream.PredStreamGen(seqnn_model, fasta_shuffled_seq_gen(shuffled_seqs_fasta_file), batch_size, stream_seqs=32, verbose=True)
    
    for exp in range(num_experiments):
        # get predictions
        preds = preds_stream[exp]
        
        # log.info(f"preds look -------{preds_stream[exp]}")
        # process SNP
        write_snp(
            preds,
            scd_out,
            exp,
            head_index,
            model_index,
            seqnn_model.diagonal_offset,
            options.scd_stats,
            plot_dir,
            options.plot_lim_min,
            options.plot_freq,
        )
    
    scd_out.close()


def initialize_output_h5(out_dir, scd_stats, shuffled_seqs_fasta_file, target_ids, target_labels, head_index, model_index):
    """Initialize an output HDF5 file for SCD stats."""

    num_targets = len(target_ids)
    seq_IDs = [line for line in open(shuffled_seqs_fasta_file) if line.startswith(">")]
    num_experiments = num = len(seq_IDs)
    scd_out = h5py.File(f"{out_dir}/scd.h5", "w")
    

    scd_out.create_dataset("locus_specification" ,shape=(num_experiments,),data=np.array(seq_IDs).astype("S")) # 
        

    # initialize scd stats
    for scd_stat in scd_stats:
        log.info(f"initialised stats {scd_stat}")
        
        for target_ind in range(num_targets):
            scd_out.create_dataset(
                f"{scd_stat}_h{head_index}_m{model_index}_t{target_ind}",
                shape=(num_experiments,),
                dtype="float16",
                compression=None,
            )

    return scd_out


def write_snp(
    ref_preds,
    scd_out,
    si,
    head_index,
    model_index,
    diagonal_offset,
    scd_stats=["SCD"],
    plot_dir=None,
    plot_lim_min=0.1,
    plot_freq=100,
):
    """Write SNP predictions to HDF."""
    
    log.info(f"writting SNP predictions for experiment {si}")
    
    # increase dtype
    ref_preds = ref_preds.astype("float32")
    # log.info(f"****** ref_preds {ref_preds.shape}")
    
    if "MPS" in scd_stats:
        # current standard map selection scores
        Max_scores_pixelwise = np.max(ref_preds, axis=0)
        for target_ind in range(ref_preds.shape[1]):
            scd_out[f"MPS_h{head_index}_m{model_index}_t{target_ind}"][si] = Max_scores_pixelwise[target_ind].astype("float16")
            
    if "CS" in scd_stats: 
        # customised scores for exploration
        std = np.std(ref_preds, axis=0)
        mean = np.mean(ref_preds, axis=0)
        Custom_score = 3/mean + 2/std 
        for target_ind in range(ref_preds.shape[1]):
            scd_out[f"CS_h{head_index}_m{model_index}_t{target_ind}"][si] = Custom_score[target_ind].astype("float16")
        
    # compare reference to alternative via mean subtraction
    if "SCD" in scd_stats:
        # sum of squared diffs
        sd2_preds = np.sqrt((ref_preds**2).sum(axis=0))
        # log.info(f"scd----- {sd2_preds}")
        for target_ind in range(ref_preds.shape[1]):
            scd_out[f"SCD_h{head_index}_m{model_index}_t{target_ind}"][si] = sd2_preds[target_ind].astype("float16")
            
            
    if "MSS" in scd_stats:
        # sum of square diffs
        s_preds = np.sum(ref_preds**2, axis=0)
        
        # log.info(f"score----- {s_preds}")
        for target_ind in range(ref_preds.shape[1]):
            scd_out[f"MSS_h{head_index}_m{model_index}_t{target_ind}"][si] = s_preds[target_ind].astype("float16")

    if np.any((["INS" in i for i in scd_stats])):
        ref_map = ut_dense(ref_preds, diagonal_offset)
        for stat in scd_stats:
            if "INS" in stat:
                insul_window = int(stat.split("-")[1])
                
                for target_ind in range(ref_preds.shape[1]):
                    scd_out[f"{stat}_h{head_index}_m{model_index}_t{target_ind}"][si] = akita_utils.stats_utils.insul_diamonds_scores(ref_map, window=insul_window)[target_ind].astype("float16")

    if (plot_dir is not None) and (np.mod(si, plot_freq) == 0):
        log.info(f"plotting {si}")
        # convert back to dense
        ref_map = ut_dense(ref_preds, diagonal_offset)
        _, axs = plt.subplots(1, ref_preds.shape[-1], figsize=(24, 4))
        for ti in range(ref_preds.shape[-1]):
            ref_map_ti = ref_map[..., ti]
            # TEMP: reduce resolution
            ref_map_ti = block_reduce(ref_map_ti, (2, 2), np.mean)
            # vmin = min(ref_map_ti.min(), ref_map_ti.min())
            # vmax = max(ref_map_ti.max(), ref_map_ti.max())
            # vmin = min(-plot_lim_min, vmin)
            # vmax = max(plot_lim_min, vmax)
            
            vmin,vmax = -0.1,0.1
            
            
            
            sns.heatmap(
                ref_map_ti,
                ax=axs[ti],
                center=0,
                vmin=vmin,
                vmax=vmax,
                cmap="RdBu_r",
                xticklabels=False,
                yticklabels=False,
            )
            
            std = np.std(ref_preds, axis=0)
            mean = np.mean(ref_preds, axis=0)
            Custom_score = 3/mean + 2/std 
            
            axs[ti].set_title(f"SCD {np.sqrt((ref_preds**2).sum(axis=0))[ti]}\nMSS {np.sum(ref_preds**2, axis=0)[ti]}\nCS {Custom_score[ti]}")

        plt.tight_layout()
        plt.savefig(f"{plot_dir}/seq_{si}_max-SCD_{np.max(np.sqrt((ref_preds**2).sum(axis=0)))}.pdf")
        plt.close()

################################################################################
# __main__
################################################################################


if __name__ == "__main__":
    main()