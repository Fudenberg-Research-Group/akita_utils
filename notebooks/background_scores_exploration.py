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
  error = 0.01

  for gc in np.percentile(df['GC'].dropna().values, np.linspace(1,99,50)):
    for i in range(df.shape[0]):
        if gc-error <= df['GC'].values[i] <= gc+error:
            super_set += [i]
            break

  super_set = list(set(super_set)) 
  sample_set = [28]#super_set[2:3]

  #################################################################
  # Making predictions for the sampled data
    
  predictions=[]

  for ind in set(sample_set):
        chrom, start, end, gc = df.iloc[ind][['chrom','start','end','GC']]
        genome_open = pysam.Fastafile(options.genome_fasta)
        seq = genome_open.fetch(chrom, start, end).upper()
        seq_1hot = dna_io.dna_1hot(seq)
        predictions.append(seq_1hot)

  predictions = np.array(predictions)
  predictions = seqnn_model.predict(predictions, batch_size=6)#len(sample_set)

  #################################################################
  # For comparison further down
    
  shuffle_set = [8] # shuffling basepairs to sample for comparison
  scores_thresh_set = [3500,5500,7500]
    
  #################################################################
  # calculating initial scores
    
  scores_before = {}
  for gc in sample_set:
    new_dataframe = df.iloc[[gc]]
    for k in shuffle_set:
        print(gc,k)
        scores_before[gc,k] = akita_utils.custom_calculate_scores(seqnn_model=seqnn_model, 
                                                        genome_fasta=options.genome_fasta, 
                                                        seq_length=seq_length, 
                                                        dataframe=new_dataframe, 
                                                        max_iters = options.max_iters, 
                                                        batch_size = options.batch_size, 
                                                        shuffle_k = k, 
                                                        ctcf_thresh = options.ctcf_thresh, 
                                                        scores_thresh = options.scores_thresh, 
                                                        scores_pixelwise_thresh = options.scores_pixelwise_thresh,
                                                        success_scores = 1)
    
    
  #################################################################  
  # calculating scores after shuffling
    
  scores_shuffle_after = {}
  for gc in sample_set:
    new_dataframe = df.iloc[[gc]]
    for k in shuffle_set:
        scores_shuffle_after[gc,k] = akita_utils.custom_calculate_scores(seqnn_model = seqnn_model, 
                                                        genome_fasta = options.genome_fasta, 
                                                        seq_length = seq_length, 
                                                        dataframe = new_dataframe, 
                                                        max_iters = options.max_iters, 
                                                        batch_size = options.batch_size, 
                                                        shuffle_k = k, 
                                                        ctcf_thresh = options.ctcf_thresh, 
                                                        scores_thresh = options.scores_thresh, 
                                                        scores_pixelwise_thresh = options.scores_pixelwise_thresh,
                                                        success_scores = 1)
    
    
  #################################################################  
  # calculating scores after changing threshold
    
  scores_thresh_after = {}
  for gc in sample_set:
    new_dataframe = df.iloc[[gc]]
    for score in scores_thresh_set:
        print(gc,score)
        scores_thresh_after[gc,score] = akita_utils.custom_calculate_scores(seqnn_model = seqnn_model, 
                                                        genome_fasta = options.genome_fasta, 
                                                        seq_length = seq_length, 
                                                        dataframe = new_dataframe, 
                                                        max_iters = options.max_iters, 
                                                        batch_size = options.batch_size, 
                                                        shuffle_k = options.shuffle_k, 
                                                        ctcf_thresh = options.ctcf_thresh, 
                                                        scores_thresh = score, 
                                                        scores_pixelwise_thresh = options.scores_pixelwise_thresh,
                                                        success_scores = 1)
        
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

    fig = plt.figure(figsize=(6* len(shuffle_set) , 6 *  len(sample_set) ), constrained_layout=True)     
    spec = fig.add_gridspec(ncols=len(shuffle_set), nrows=len(sample_set), hspace=0.1, wspace=0.1)#

    target_ind = 0
    vmin=-2; vmax=2

    for ind in sample_set:        
        chrom, start, end, gc = df.iloc[ind][['chrom','start','end','GC']]
        for k in shuffle_set:
            
            ax = fig.add_subplot(spec[sample_set.index(ind),shuffle_set.index(k)])
            temp_scores_before = []
            temp_scores_shuffle_after = []
            for i in scores_before[ind,k]:
                temp_scores_before =+ i
            for i in scores_shuffle_after[ind,k]:
                temp_scores_shuffle_after =+ i

            kde_df_after = pd.DataFrame(temp_scores_shuffle_after, columns=["score"])
            kde_df_before = pd.DataFrame(temp_scores_before, columns=["score"])
            sns.kdeplot(data=kde_df_after, x="score", bw_adjust=.3, label='after', fill=True)
            sns.kdeplot(data=kde_df_before, x="score", bw_adjust=.3, label='before', fill=True)
            ax.legend()
            plt.title(f'GC_{gc} k_{k} before and after masking')
    plt.savefig(f'{plot_dir}/shuffle_parameter_results.pdf')
    plt.close()
    
    fig = plt.figure(figsize=(6* len(scores_thresh_set) , 6 *  len(sample_set) ), constrained_layout=True)     
    spec = fig.add_gridspec(ncols=len(scores_thresh_set), nrows=len(sample_set), hspace=0.1, wspace=0.1)
    
    for ind in sample_set:        
        chrom, start, end, gc = df.iloc[ind][['chrom','start','end','GC']]            
        for score in scores_thresh_set:
            
            ax = fig.add_subplot(spec[sample_set.index(ind),scores_thresh_set.index(score)])
            temp_scores_before = []
            temp_scores_thresh_after = []

            for i in scores_before[ind,8]:
                temp_scores_before =+ i
            for i in scores_thresh_after[ind,score]:
                temp_scores_thresh_after =+ i

            kde_df_after = pd.DataFrame(temp_scores_thresh_after, columns=["score"])
            kde_df_before = pd.DataFrame(temp_scores_before, columns=["score"])
            sns.kdeplot(data=kde_df_after, x="score", bw_adjust=.3, label='masking', fill=True)
            sns.kdeplot(data=kde_df_before, x="score", bw_adjust=.3, label='permutation', fill=True)
            ax.legend()
            plt.title(f'GC_{gc} scores_thresh_{score}')       
    plt.savefig(f'{plot_dir}/masking_threshold_parameter_results.pdf')
    plt.close()

################################################################################
# __main__
################################################################################
if __name__ == '__main__':
  main()