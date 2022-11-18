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

print(tf.config.list_physical_devices('GPU'))
#for gpu in gpus:
#  tf.config.experimental.set_memory_growth(gpu, True)
print(gpus)

from basenji import seqnn
from basenji import stream
from basenji import dna_io

#from basenji import vcf as bvcf

import akita_utils
from io import StringIO
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

  parser.add_option('-m', dest='plot_map',
      default=False, action='store_true',
      help='Plot contact map for each allele [Default: %default]')
    
  parser.add_option('-o',dest='out_dir',
      default='scd',
      help='Output directory for tables and plots [Default: %default]')

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
      default=None, type='int',
      help='Specify batch size')

  parser.add_option('--head-index', dest='head_index',
      default=0, type='int',
      help='Specify head index (0=human 1=mus) ')

  parser.add_option('--mut-method', dest='mutation_method',
      default='mask', type='str',
      help='specify mutation method')
    
  parser.add_option('--chrom-data', dest='chrom_data',
      default=None,              
      help='crom_data_directory')

  parser.add_option('--max-iters', dest='max_iters',
      default=5, type='int',
      help='maximum number of iterations')  
    
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
    
  parser.add_option('--rc', dest='rc',
      default=False, action='store_true',
      help='Average forward and reverse complement predictions [Default: %default]')

    
  (options, args) = parser.parse_args()

  if len(args) == 3:
    # single worker
    params_file = args[0]
    model_file = args[1]
    motif_file = args[2]
  else:
    parser.error('Must provide parameters and model files and QTL VCF file')

  if not os.path.isdir(options.out_dir):
    os.mkdir(options.out_dir)
  if options.plot_map:
    plot_dir = options.out_dir
  else:
    plot_dir = None

  options.shifts = [int(shift) for shift in options.shifts.split(',')]
  options.scd_stats = options.scd_stats.split(',')

  random.seed(44)

  #################################################################
  # read parameters and targets

  with open(params_file) as params_open:
    params = json.load(params_open)
  params_train = params['train']
  params_model = params['model']

  if options.batch_size is None:
    batch_size = params_train['batch_size']
  else: 
    batch_size = options.batch_size
  print(batch_size)

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
  # fetching chromosome data

  chromsizes = bioframe.read_chromsizes(options.chrom_data)
  dframe = pd.DataFrame(chromsizes)
  dframe['end'] = dframe['length']+ 1310720
  dframe = dframe.reset_index()
  dframe.rename(columns = {'index' : 'chrom', 'length':'start'}, inplace = True)
  df = bioframe.frac_gc(dframe, bioframe.load_fasta(options.genome_fasta), return_input=True)

  #################################################################
  # Generating a sample for down stream analysis
  super_set = []
  error = 0.001

  for gc in np.percentile(df['GC'].dropna().values, np.linspace(1,99,)15):
    for i in range(df.shape[0]):
        if gc-error <= df['GC'].values[i] <= gc+error:
            super_set += [i]
            break

  super_set = list(set(super_set)); print(f'Whole distribution: {super_set}')
  sample_set = super_set; print(f'Sampled datapoints: {sample_set}')
  new_dataframe = df.iloc[[ind for ind in set(sample_set)]]  
  #################################################################
  # calculating initial scores
    
  def mutation_search(seqnn_model,
                    genome_fasta,
                    seq_length,
                    dataframe,
                    max_iters=1,
                    batch_size=6,
                    shuffle_k=8,
                    ctcf_thresh=8,
                    scores_thresh = 5500,
                    scores_pixelwise_thresh = .04,
                    masking=False,
                    timing = False
                    ):
    
    """
    This function creates flat sequences
    """
    flat_seqs = []
    scores_storage = {}
    num_seqs = dataframe.shape[0]

    mot = ">CCAsyAGrkGGCr\n0.0000\t1.0000\t0.0000\t0.0000\n0.0000\t1.0000\t0.0000\t0.0000\n1.0000\t0.0000\t0.0000\t0.0000\n0.0000\t0.5000\t0.5000\t0.0000\n0.0000\t0.5000\t0.0000\t0.5000\n1.0000\t0.0000\t0.0000\t0.0000\n0.0000\t0.0000\t1.0000\t0.0000\n0.5000\t0.0000\t0.5000\t0.0000\n0.0000\t0.0000\t0.5000\t0.5000\n0.0000\t0.0000\t1.0000\t0.0000\n0.0000\t0.0000\t1.0000\t0.0000\n0.0000\t1.0000\t0.0000\t0.0000\n0.5000\t0.0000\t0.5000\t0.0000"
    motif = pd.read_csv(StringIO(mot), sep="\t", header=0, names=["A", "C", "G", "T"]).values
    motif_window = int(np.ceil(len(motif) / 2))
    mot_shuf = [12, 0, 1, 11, 10, 3, 2, 8, 9, 4, 5, 7, 6, 13]

    for ind in range(num_seqs):
        chrom, start, end, gc = dataframe.iloc[ind][["chrom", "start", "end", "GC"]]
        genome_open = pysam.Fastafile(genome_fasta)
        seq = genome_open.fetch(chrom, start, end).upper()
        seq_1hot = dna_io.dna_1hot(seq)

        t0 = time.time()
        num_iters = 0
        while num_iters < max_iters:
            name = f'{num_iters}'
            # scores_set = []
            print("ind",ind,", iter ",num_iters,", k ",shuffle_k,", for", chrom, start, end,)
            seq_1hot_batch = []
            for i in range(batch_size):
                seq_1hot_mut = akita_utils.permute_seq_k(seq_1hot, k=shuffle_k)
                s = akita_utils.scan_motif(seq_1hot_mut, motif)
                if masking == 0:
                    for i in np.where(s > ctcf_thresh)[0]:
                        if len(seq_1hot_mut[i-motif_window:i+motif_window]) == len(mot_shuf):
                            seq_1hot_mut[i-motif_window:i+motif_window] = akita_utils.permute_seq_k(seq_1hot_mut[i-motif_window:i+motif_window], k=2)
                    seq_1hot_batch.append(seq_1hot_mut)
                elif masking == 1:
                    for i in np.where(s > ctcf_thresh)[0]:
                        if len(seq_1hot_mut[i-motif_window:i+motif_window]) == len(mot_shuf):
                            seq_1hot_mut[i-motif_window:i+motif_window] = seq_1hot_mut[i-motif_window:i+motif_window][mot_shuf]

                    seq_1hot_batch.append(seq_1hot_mut)
                else:
                    seq_1hot_batch.append(seq_1hot_mut)

            seq_1hot_batch = np.array(seq_1hot_batch)
            pred = seqnn_model.predict(seq_1hot_batch, batch_size=batch_size)
            scores = np.sum(pred**2, axis=-1).sum(axis=-1)
            scores_pixelwise = np.max(pred**2, axis=-1).max(axis=-1)  
            t1 = time.time()
            # scores_set += [scores]
            scores_storage[name] = [scores]
            
            if np.any([
                (np.min(scores) < scores_thresh)
                , (np.min(scores_pixelwise) < scores_pixelwise_thresh)]
            ):
                # scores_set += [scores]
                best_ind = np.argmin(scores_pixelwise)
                best_seq = seq_1hot_batch[best_ind]
                best_pred = pred[best_ind]
                best_score, best_score_pixelwise = (
                    scores[best_ind],
                    scores_pixelwise[best_ind],
                )
                
                print(
                    "success: best seq, thresh",
                    np.min(scores),
                    " pixelwise",
                    np.min(scores_pixelwise),
                    "time",
                    t1 - t0
                )
                if timing == True:
                    num_iters = max_iters
            else:
                best_ind = np.argmin(scores_pixelwise)
                best_seq = seq_1hot_batch[best_ind]
                best_pred = pred[best_ind]
                best_score, best_score_pixelwise = (
                    scores[best_ind],
                    scores_pixelwise[best_ind],
                )
                print(
                "trying: best seq, thresh",
                np.min(scores),
                " pixelwise",
                np.min(scores_pixelwise),
                )
                
            num_iters += 1
            if num_iters >= max_iters:
                # scores_set += [scores]
                print(f"max iters exceeded, final time {t1 - t0}")
                if gc :
                    flat_seqs.append([
                    best_seq,
                    best_pred,
                    best_score,
                    best_score_pixelwise,
                    t1 - t0,
                    gc
                ])
                else:
                    flat_seqs.append([
                    best_seq,
                    best_pred,
                    best_score,
                    best_score_pixelwise,
                    t1 - t0
                ])


    return scores_storage, flat_seqs

  scores_storage_random_masking, flat_seqs_random_masking = mutation_search( seqnn_model=seqnn_model,
                                                                            genome_fasta=options.genome_fasta,
                                                                            seq_length=seq_length,
                                                                            dataframe= new_dataframe,
                                                                            max_iters=options.max_iters,
                                                                            batch_size=options.batch_size,
                                                                            masking=0,
                                                                            timing = True)


  scores_storage_motif_masking, flat_seqs_motif_masking = mutation_search( seqnn_model=seqnn_model,
                                                                            genome_fasta=options.genome_fasta,
                                                                            seq_length=seq_length,
                                                                            dataframe= new_dataframe,
                                                                            max_iters=options.max_iters,
                                                                            batch_size=options.batch_size,
                                                                            masking=1,
                                                                            timing = True)


  scores_storage_all_random, flat_seqs_no_mask = mutation_search( seqnn_model=seqnn_model,
                                                                genome_fasta=options.genome_fasta,
                                                                seq_length=seq_length,
                                                                dataframe= new_dataframe,
                                                                max_iters=options.max_iters,
                                                                batch_size=options.batch_size,
                                                                timing = True)
        
  #################################################################
    
  try:
    target_crop = params_model['trunk'][-2]['cropping']
  except:
    target_crop = params_model['target_crop']
  
  target_map_size = seq_length//2048  - target_crop*2 
  hic_diags = params_model['diagonal_offset']  

  #################################################################
  # plots
    
  if plot_dir is not None:
    
    random_masking = []
    motif_masking = []
    all_random = []
    
    for i in scores_storage_random_masking:
        for j in scores_storage_random_masking[i]:
            random_masking += j.tolist()

    for i in scores_storage_motif_masking:
        for j in scores_storage_motif_masking[i]:
            motif_masking += j.tolist()        

    for i in scores_storage_all_random:
        for j in scores_storage_all_random[i]:
            all_random += j.tolist()

    kde_1 = pd.DataFrame(random_masking, columns=["score"])
    kde_2 = pd.DataFrame(motif_masking, columns=["score"])
    kde_3 = pd.DataFrame(all_random, columns=["score"])
    sns.kdeplot(data=kde_1, x="score", bw_adjust=.2, label='randomly shuffle motif', fill=True)  
    sns.kdeplot(data=kde_2, x="score", bw_adjust=.2, label='manually shuffle motif', fill=True)    
    sns.kdeplot(data=kde_3, x="score", bw_adjust=.2, label='no mask', fill=True)
    plt.legend()
    plt.savefig(f'{plot_dir}/muation_method_scores_results.pdf',dpi=300)
    plt.close()
    
    # ------------------------------------------------------------------------
    
    flat_pred_time_random_masking = []
    flat_pred_time_motif_masking = []
    flat_pred_time_no_mask = []

    for i in range(len(sample_set)):
        flat_pred_time_no_mask += [flat_seqs_no_mask[i][4]]
        flat_pred_time_random_masking += [flat_seqs_random_masking[i][4]]
        flat_pred_time_motif_masking += [flat_seqs_motif_masking[i][4]]

    kde_1 = pd.DataFrame(flat_pred_time_no_mask, columns=["success time (s)"])  
    kde_2 = pd.DataFrame(flat_pred_time_motif_masking, columns=["success time (s)"])
    kde_3 = pd.DataFrame(flat_pred_time_random_masking, columns=["success time (s)"])  
    sns.kdeplot(data=kde_1, x="success time (s)", bw_adjust=.2, label='all random', fill=True)    
    sns.kdeplot(data=kde_2, x="success time (s)", bw_adjust=.2, label='manually shuffle motif', fill=True) 
    sns.kdeplot(data=kde_3, x="success time (s)", bw_adjust=.2, label='randomly shuffle motif', fill=True)  
    plt.legend()
    plt.savefig(f'{plot_dir}/muation_method_timing_results.pdf',dpi=300)
    plt.close()

################################################################################
# __main__
################################################################################
if __name__ == '__main__':
  main()