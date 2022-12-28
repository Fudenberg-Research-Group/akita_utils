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
# =========================================================================
from __future__ import print_function


import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
log = logging.getLogger(__name__)


from optparse import OptionParser
import json
import os
import pdb
import pickle
import random
import sys
import time
import bioframe 
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

from basenji import seqnn
from basenji import stream
from basenji import dna_io

import akita_utils
from akita_utils import ut_dense, split_df_equally 

'''
creating flat maps/seqs

'''

################################################################################
# main 
################################################################################
def main():
    usage = 'usage: %prog [options] <params_file> <model_file> <vcf_file>'
    parser = OptionParser(usage)
    
    parser.add_option('-f', dest='genome_fasta',
      default=None,
      help='Genome FASTA for sequences [Default: %default]')
    
    parser.add_option('-l', dest='plot_lim_min',
      default=0.1, type='float',
      help='Heatmap plot limit [Default: %default]')
    
    parser.add_option('--plot-freq', dest='plot_freq',
      default=100, type='int',
      help='Heatmap plot freq [Default: %default]')
    
    parser.add_option('-m', dest='plot_map',
      default=None, action='store_true',
      help='Plot contact map for each allele [Default: %default]')
    
    parser.add_option('-o',dest='out_dir',
      default='scd',
      help='Output directory for tables and plots [Default: %default]')
    
    parser.add_option('-p', dest='processes',
      default=None, type='int',
      help='Number of processes, passed by multi script')
    
    parser.add_option('--rc', dest='rc',
      default=False, action='store_true',
      help='Average forward and reverse complement predictions [Default: %default]')
    
    parser.add_option('--shifts', dest='shifts',
      default='0', type='str',
      help='Ensemble prediction shifts [Default: %default]')
    
    parser.add_option('--stats', dest='scd_stats',
      default='SCD',
      help='Comma-separated list of stats to save. [Default: %default]')
    
    parser.add_option('-t', dest='targets_file',
      default=None, type='str',
      help='File specifying target indexes and labels in table format')
    
    parser.add_option('--batch-size', dest='batch_size',
      default=6, type='int',
      help='Specify batch size')
    
    parser.add_option('--head-index', dest='head_index',
      default=0, type='int',
      help='Specify head index (0=human 1=mus) ')

    parser.add_option('-s', dest='save_seqs',
      default=None,
      help='Save the final seqs in fasta format')  

    parser.add_option('--shuffle_k', dest='shuffle_k',
      default=8, type='int',
      help='basepairs considered for shuffling')

    parser.add_option('--ctcf-thresh', dest='ctcf_thresh',
      default=8, type='int',
      help='maximum alowable ctcf motifs in a flat seq')

    parser.add_option('--scores-thresh', dest='scores_thresh',
      default=5500, type='int',
      help='maximum alowable score for a flat seq')

    parser.add_option('--scores-pixelwise-thresh', dest='scores_pixelwise_thresh',
      default=0.04, type='float',
      help='maximum alowable pixel score for a flat seq')
    
    parser.add_option('--max_iters', dest='max_iters',
      default=10,              
      help='maximum iterations')

    parser.add_option('--flat_seq_tsv_file', dest='flat_seq_tsv_file',
      default='/home1/kamulege/akita_utils/bin/background_seq_experiments/data/flat_seqs.tsv',           
      help='flat_se tsv file')

    (options, args) = parser.parse_args()

    if len(args) == 3:
        # single worker
        params_file = args[0]
        model_file = args[1]
        background_file = args[2]
    elif len(args) == 5:
        # multi worker
        options_pkl_file = args[0]
        params_file = args[1]
        model_file = args[2]
        background_file = args[3]
        worker_index = int(args[4])

        # load options
        options_pkl = open(options_pkl_file, 'rb')
        options = pickle.load(options_pkl)
        options_pkl.close()

        # update output directory
        options.out_dir = '%s/job%d' % (options.out_dir, worker_index)
    else:
        parser.error('Must provide parameters and model files and QTL VCF file')

    if not os.path.isdir(options.out_dir):
        os.mkdir(options.out_dir)

    options.shifts = [int(shift) for shift in options.shifts.split(',')]

    random.seed(44)

    #################################################################
    # read parameters and targets

    # read model parameters
    with open(params_file) as params_open:
        params = json.load(params_open)
        params_train = params['train']
        params_model = params['model']

    if options.batch_size is None:
        batch_size = params_train['batch_size']
    else: 
        batch_size = options.batch_size
    log.info(f"Batch size {batch_size}")

    if options.targets_file is not None:
        targets_df = pd.read_csv(options.targets_file, sep='\t', index_col=0)
        target_ids = targets_df.identifier
        target_labels = targets_df.description

    #################################################################
    # setup model

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
    # load sequences    
    
    # filter for worker motifs
    if options.processes is not None:                    # multi-GPU option
        seq_coords_full = pd.read_csv(options.flat_seq_tsv_file, sep="\t")
        seq_coords_df = split_df_equally(seq_coords_full, options.processes, worker_index)
    else:
        seq_coords_df = pd.read_csv(options.backgroung_chromosome_tsv_file, sep="\t")
    
    num_experiments = len(seq_coords_df)
    log.info("===================================")
    log.info(f"Number of experiements = {num_experiments} \n It's not number of predictions. Num of predictions is upper bounded by {options.max_iters} x {batch_size} for each experiment")
    log.info("===================================")    
    #################################################################
    # create flat sequences
    
    flat_seqs = akita_utils.background_utils.create_flat_seqs_gen(seqnn_model, 
                                                                    options.genome_fasta,
                                                                    seq_coords_df,
                                                                    max_iters=options.max_iters,
                                                                    batch_size=batch_size)    

    #################################################################
    # save flat sequences in fasta format if requested

    if options.save_seqs is not None:  
        with open(f'{options.out_dir}/background_seqs.fa','w') as f:
            for i in range(len(flat_seqs)):
                f.write('>shuffled_chr'+str(i)+'_score'+str(int(flat_seqs[i][2]))+'_pixelwise'+str(int(flat_seqs[i][3]))+'\n')
                f.write(dna_io.hot1_dna(flat_seqs[i][0])+'\n')
        log.info(f"finished saving! \n plotting next if requested")

    #################################################################
    # plot flat sequences

    if options.plot_map is not None:
        preds = [flat_seqs[i][1] for i in range(len(flat_seqs))]
        plot_lim_min=0.1
        hic_diags = params_model['diagonal_offset']
        for no,pred in enumerate(preds):
            ref_preds = pred
            ref_map = ut_dense(ref_preds, hic_diags) # convert back to dense
            _, axs = plt.subplots(1, ref_preds.shape[-1], figsize=(24, 4))
            for ti in range(ref_preds.shape[-1]):
                ref_map_ti = ref_map[..., ti]
                # TEMP: reduce resolution
                ref_map_ti = block_reduce(ref_map_ti, (2, 2), np.mean)
                vmin = min(ref_map_ti.min(), ref_map_ti.min())
                vmax = max(ref_map_ti.max(), ref_map_ti.max())
                vmin = min(-plot_lim_min, vmin)
                vmax = max(plot_lim_min, vmax)
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

            plt.tight_layout()
            plt.savefig(f"{options.out_dir}/seq_{no}.pdf")
            plt.close()
        log.info(f"finished plotting! \ncheck {options.out_dir} for plots")
    
################################################################################
# __main__
################################################################################
if __name__ == '__main__':
      main()



 